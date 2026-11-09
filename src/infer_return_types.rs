// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of Boring.
// Boring is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// See the LICENSE file at the project root for the full text.

//! Phase 1 of the implicit-generics design (docs/book.md, "Implicit generics: design notes"):
//! a top-level `def` written without a return type whose body says `return <expr>` gets the
//! type of that expression written back into its `return_ty`, so every later stage sees an
//! ordinary annotated function.
//!
//! ```text
//! def fetchGreeting():
//!     return Response(status = 200, payload = "hello world")   # -> Response<int, string>
//! ```
//!
//! Deliberately conservative, because a wrong guess would be worse than the status quo (an
//! unannotated function is `void`, and a `return <expr>` in one is already broken):
//!
//! - **Only explicit `return <expr>`** counts. A trailing expression does not: with a written
//!   return type it is already the result, but with none the function is `void` and inferring
//!   from the last expression would change the signature of every function ending on a call.
//! - **Best effort, never an error.** When the type of any returned expression cannot be
//!   determined here, the function is left exactly as written (no diagnostic: an expression
//!   this pass cannot type may still be a perfectly valid `void` one, e.g. `return g()` where
//!   `g` returns nothing). Reporting the omission is `@api`'s job (`checker/api.rs`).
//! - **Skipped**: `@api` functions (their signature must be written), generic functions,
//!   methods (`def Type.name()` and struct/enum/ext members), `task`/`stream`/`native`
//!   functions, `def mut`, functions in `mod` blocks, and anything recursive (a function whose
//!   inferred type depends on itself, directly or through other functions, keeps no type).
//! - **Types it can produce**: scalars from literals and typed bindings, strings, arrays and
//!   tuples of those, a struct constructed with `Name(...)` (a generic struct's parameters are
//!   unified from the field types of the arguments, so `Box(v = 3)` is `Box<int>`), an enum
//!   variant, a field read, arithmetic and comparisons, a cast, a call to another function
//!   whose return type is written or inferred, and a `type def` factory call. Qualified,
//!   borrowed and `mut` types are never produced (their rules belong to the qualifier
//!   inference, not to this pass).
//! - A call to a function or constructor defined in another file is unknown to the run inside
//!   `parser::parse` (each file is parsed on its own); `infer_return_types_with_imports`
//!   re-runs the pass from `desugar_array_block_with_imports`, which knows the imports.

use crate::ast::*;
use crate::ast_walk::Walker;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

type Env = HashMap<String, Option<Type>>;

/// Rewrites every eligible top-level `def` of `program` that has no return type.
pub fn infer_return_types(program: &mut Program) {
    infer_items(&mut program.items);
}

/// [`infer_return_types`] for a file whose `use`d files' items are known (`imported`, see
/// `desugar_inject::collect_imported_items`): a call or construction of an imported function or
/// struct is typed like a local one. Local declarations shadow imports of the same name. Only
/// the file's own items are rewritten; imported ones were inferred when their own file parsed.
pub fn infer_return_types_with_imports(program: &mut Program, imported: &[Item]) {
    if imported.is_empty() {
        return;
    }
    let local: HashSet<String> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) => Some(f.name.clone()),
            Item::Struct(s) => Some(s.name.clone()),
            Item::Enum(e) => Some(e.name.clone()),
            _ => None,
        })
        .collect();
    let mut combined: Vec<Item> = imported
        .iter()
        .filter(|i| match i {
            Item::Fn(f) => !local.contains(&f.name),
            Item::Struct(s) => !local.contains(&s.name),
            Item::Enum(e) => !local.contains(&e.name),
            _ => true,
        })
        .cloned()
        .collect();
    let n = combined.len();
    combined.append(&mut program.items);
    // Bounds flow from imported functions into the local callers (their own were fixed when their
    // file parsed), before the return types are inferred.
    crate::infer_param_generics::propagate_in(&mut combined);
    infer_items(&mut combined);
    program.items = combined.split_off(n);
}

/// The functions, structs and enums `infer_return_types_with_imports` can see from `items`'
/// `use`s, following the import rules of `desugar_array_block_with_imports`: a bare `use name`
/// (same-project sibling) imports every top-level item, a multi-segment or `[deps]`-qualified one
/// only the `pub` items (narrowed to the named ones by `use mod(A, B)`), the embedded stdlib
/// (`use boring.<module>`) is read from its embedded source. Imports are followed transitively
/// (each file's own `use`s resolve against that file's directory); unreadable or unparsable
/// files are skipped.
pub fn collect_inference_imports(
    items: &[Item],
    dir: &std::path::Path,
    deps: &HashMap<String, std::path::PathBuf>,
) -> Vec<Item> {
    let mut out = Vec::new();
    walk_imports(items, dir, deps, &mut HashSet::new(), &mut out);
    out
}

fn walk_imports(
    items: &[Item],
    dir: &std::path::Path,
    deps: &HashMap<String, std::path::PathBuf>,
    visited: &mut HashSet<(std::path::PathBuf, bool)>,
    out: &mut Vec<Item>,
) {
    for item in items {
        let Item::Use(u) = item else {
            if let Item::Mod(m) = item { walk_imports(&m.items, dir, deps, visited, out); }
            continue;
        };
        let Some(first) = u.path.first() else { continue };
        if first == "boring" {
            let module = u.path.get(1).map(String::as_str).unwrap_or("");
            if let Some(src) = crate::stdlib_embed::lookup(module) {
                if visited.insert((crate::stdlib_embed::synthetic_path(module), true)) {
                    import_src(src, dir, true, &u.items, deps, visited, out);
                }
            }
        } else if let Some(root) = deps.get(first.as_str()) {
            let rel: std::path::PathBuf = u.path[1..].iter().collect();
            import_path(&root.join(rel).with_extension("br"), true, &u.items, deps, visited, out);
        } else {
            let rel: std::path::PathBuf = u.path.iter().collect();
            import_path(&dir.join(rel).with_extension("br"), u.path.len() > 1, &u.items, deps, visited, out);
        }
    }
}

fn import_path(
    path: &std::path::Path,
    pub_only: bool,
    only: &[String],
    deps: &HashMap<String, std::path::PathBuf>,
    visited: &mut HashSet<(std::path::PathBuf, bool)>,
    out: &mut Vec<Item>,
) {
    let Ok(path) = path.canonicalize() else { return };
    if !visited.insert((path.clone(), pub_only)) { return; }
    let Ok(src) = std::fs::read_to_string(&path) else { return };
    let Some(file_dir) = path.parent() else { return };
    import_src(&src, file_dir, pub_only, only, deps, visited, out);
}

fn import_src(
    src: &str,
    dir: &std::path::Path,
    pub_only: bool,
    only: &[String],
    deps: &HashMap<String, std::path::PathBuf>,
    visited: &mut HashSet<(std::path::PathBuf, bool)>,
    out: &mut Vec<Item>,
) {
    let Ok(tokens) = crate::lexer::lex_all(src) else { return };
    let Ok(file) = crate::parser::parse(tokens) else { return };
    fn take(items: &[Item], ok: &dyn Fn(&str, bool) -> bool, out: &mut Vec<Item>) {
        for it in items {
            match it {
                Item::Mod(m) => take(&m.items, ok, out),
                Item::Fn(f) if ok(&f.name, f.is_pub) => out.push(it.clone()),
                Item::Struct(s) if ok(&s.name, s.is_pub) => out.push(it.clone()),
                Item::Enum(e) if ok(&e.name, e.is_pub) => out.push(it.clone()),
                _ => {}
            }
        }
    }
    take(&file.items, &|name, is_pub| (!pub_only || is_pub) && (only.is_empty() || only.iter().any(|n| n == name)), out);
    walk_imports(&file.items, dir, deps, visited, out);
}

/// The checks of `parse` that look at the declarations a call or construction refers to
/// ([`check_untyped_constructions`], [`check_homogeneous_arrays`]), re-run with the imported
/// ones visible. Imported items only supply declarations; every body checked here is the file's
/// own, since an imported file was already checked when it was parsed.
pub fn check_with_imports(program: &Program, imported: &[Item]) -> Result<(), crate::parser::ParseError> {
    if imported.is_empty() {
        return Ok(());
    }
    let local: Vec<&Item> = program.items.iter().collect();
    let shadowed = |i: &Item| match i {
        Item::Fn(f) => local.iter().any(|l| matches!(l, Item::Fn(g) if g.name == f.name)),
        Item::Struct(s) => local.iter().any(|l| matches!(l, Item::Struct(g) if g.name == s.name)),
        Item::Enum(e) => local.iter().any(|l| matches!(l, Item::Enum(g) if g.name == e.name)),
        _ => false,
    };
    let mut items: Vec<Item> = imported.iter().filter(|i| !shadowed(i)).cloned().collect();
    let skip = items.len();
    items.extend(program.items.iter().cloned());
    check_constructions_in(&items, skip)?;
    check_arrays_in(&items, skip)
}

/// One scope's worth of [`infer_return_types`]; a `mod` is its own scope (names resolve
/// against its own members only), so each is handled by recursion.
fn infer_items(items: &mut [Item]) {
    for item in items.iter_mut() {
        if let Item::Mod(m) = item {
            infer_items(&mut m.items);
        }
    }
    // A name declared once is inferred by name; an overloaded one (resolved by argument type)
    // per declaration, each body typed with its own parameters. A call to an overloaded name
    // from another body stays untyped (`fn_ret` refuses it), so such a return is left alone.
    let (found, per_overload): (HashMap<String, Type>, HashMap<usize, Type>) = {
        let cx = Ctx::new(items);
        let names: Vec<String> = cx
            .fns
            .iter()
            .filter(|(_, v)| v.len() == 1 && omitted(v[0]))
            .map(|(k, _)| k.clone())
            .collect();
        let found = names
            .into_iter()
            .filter_map(|n| cx.fn_ret(&n).map(|t| (n, t)))
            .collect();
        let mut per = HashMap::new();
        for (i, item) in items.iter().enumerate() {
            if let Item::Fn(f) = item {
                let overloaded = cx.fns.get(&f.name).is_some_and(|v| v.len() > 1);
                if overloaded && omitted(f) && eligible(f) {
                    if let Some(t) = cx.infer_fn(f) {
                        per.insert(i, t);
                    }
                }
            }
        }
        (found, per)
    };
    if found.is_empty() && per_overload.is_empty() {
        return;
    }
    for (i, item) in items.iter_mut().enumerate() {
        if let Item::Fn(f) = item {
            if omitted(f) && eligible(f) {
                if let Some(t) = per_overload.get(&i).or_else(|| found.get(&f.name)) {
                    f.return_ty = Some(t.clone());
                }
            }
        }
    }
}

/// An overloaded function (one name, several declarations) with no written return type whose
/// body returns a value [`infer_return_types`] could not type, per overload, is the error
/// `write the return type`, as for a recursive function ([`check_recursive_returns`]): left
/// alone it silently became `void`. Run after [`infer_return_types`].
pub fn check_overloaded_returns(program: &Program) -> Result<(), crate::parser::ParseError> {
    check_overloads_in(&program.items)
}

fn check_overloads_in(scope: &[Item]) -> Result<(), crate::parser::ParseError> {
    for item in scope {
        if let Item::Mod(m) = item {
            check_overloads_in(&m.items)?;
        }
    }
    let fns: Vec<&FnDecl> = scope.iter().filter_map(|i| match i { Item::Fn(f) => Some(f), _ => None }).collect();
    for f in &fns {
        let overloaded = fns.iter().filter(|g| g.name == f.name && g.qualifier.is_none()).count() > 1;
        if !overloaded || !omitted(f) || f.is_native || f.task || f.stream || f.qualifier.is_some() {
            continue;
        }
        let Some(returns) = valued_returns(&f.body) else { continue };
        if let Some(bad) = returns.iter().find(|e| certainly_valued(e)) {
            return Err(crate::parser::ParseError::Generic {
                msg: format!(
                    "`{0}` is overloaded and returns a value whose type cannot be inferred, so its return type must be written — put it before the name, e.g. `<type> {0}(...)`",
                    f.name
                ),
                line: bad.line, col: bad.col, len: bad.len.max(1),
            });
        }
    }
    Ok(())
}

/// Whether `e` certainly produces a value, so that a function returning it cannot be a `void`
/// one that the pass merely failed to type: a literal, a variable, a field or index read, an
/// array or tuple, a cast, arithmetic. (A call may be a perfectly valid `void` call.)
fn certainly_valued(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::StringInterp(_) | ExprKind::Bool(_)
        | ExprKind::Var(_) | ExprKind::Field(..) | ExprKind::Index(..) | ExprKind::Array(_) | ExprKind::Tuple(_)
        | ExprKind::Cast(..) => true,
        ExprKind::UnaryOp(UnaryOp::Neg, _) => true,
        ExprKind::BinOp(op, ..) => matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem),
        _ => false,
    }
}

/// D4 of the implicit-generics design: a recursive function (directly, or through other functions
/// of its scope) whose return type was left out and could not be inferred must have it written.
/// Inference would need a fixpoint over the cycle; without it the function silently became
/// `void` (`boring run` then failed far from the cause, `boring run` of `even`/`odd` printed
/// `void`). As elsewhere, only a returned expression that certainly produces a value counts
/// ([`certainly_valued`]), so a recursive `void` function is never reported. Run after
/// [`infer_return_types`]: a recursive function that did get its type is no longer omitted.
pub fn check_recursive_returns(program: &Program) -> Result<(), crate::parser::ParseError> {
    check_recursion_in(&program.items)
}

fn check_recursion_in(scope: &[Item]) -> Result<(), crate::parser::ParseError> {
    for item in scope {
        if let Item::Mod(m) = item {
            check_recursion_in(&m.items)?;
        }
    }
    let fns: Vec<&FnDecl> = scope.iter().filter_map(|i| match i { Item::Fn(f) => Some(f), _ => None }).collect();
    let mut edges: HashMap<&str, HashSet<String>> = HashMap::new();
    for f in &fns {
        let mut callees: HashSet<String> = HashSet::new();
        let mut body = f.body.clone();
        {
            let mut on_expr = |e: &mut Expr| {
                if let ExprKind::Call(c, _) = &e.kind {
                    if let ExprKind::Var(n) = &c.kind {
                        callees.insert(n.clone());
                    }
                }
            };
            let mut on_stmt = |_: &mut Stmt| {};
            Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
        }
        edges.entry(f.name.as_str()).or_default().extend(callees);
    }
    let recursive = |start: &str| -> bool {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = edges.get(start).map(|s| s.iter().map(String::as_str).collect()).unwrap_or_default();
        while let Some(n) = stack.pop() {
            if n == start { return true; }
            if seen.insert(n) {
                if let Some(next) = edges.get(n) {
                    stack.extend(next.iter().map(String::as_str));
                }
            }
        }
        false
    };
    for f in &fns {
        // An overloaded name (resolved by argument type) merges several functions in `edges`: a
        // cycle through it may not be one, so it is left alone.
        let overloaded = fns.iter().filter(|g| g.name == f.name).count() > 1;
        if !omitted(f) || f.is_native || f.task || f.stream || f.qualifier.is_some() || overloaded || !recursive(&f.name) {
            continue;
        }
        let Some(returns) = valued_returns(&f.body) else { continue };
        if let Some(bad) = returns.iter().find(|e| certainly_valued(e)) {
            return Err(crate::parser::ParseError::Generic {
                msg: format!(
                    "`{0}` is recursive and returns a value, so its return type cannot be inferred — write it before the name, e.g. `<type> {0}(...)`",
                    f.name
                ),
                line: bad.line, col: bad.col, len: bad.len.max(1),
            });
        }
    }
    Ok(())
}

/// The "cannot infer" error for a function whose parameters are untyped (implicit generics,
/// phase 2): such a function is never `void` by accident, and without this check a return the
/// pass could not type (`def add(a, b): a + b` — two independent parameters, so no known output)
/// silently becomes `void`: `boring run` yields nil, `boring build` fails in rustc. Only a
/// return whose expression certainly produces a value counts ([`certainly_valued`]), so a
/// `void` function such as `def log(x): print x` is never reported. Run after
/// [`infer_return_types`]; a function that got its type there no longer has an omitted one.
pub fn check_untyped_returns(program: &Program) -> Result<(), crate::parser::ParseError> {
    check_untyped_returns_in(&program.items)
}

fn check_untyped_returns_in(items: &[Item]) -> Result<(), crate::parser::ParseError> {
    for item in items {
        if let Item::Mod(m) = item {
            check_untyped_returns_in(&m.items)?;
            continue;
        }
        let Item::Fn(f) = item else { continue };
        let implicit = f.params.iter().any(|p| matches!(&p.ty, Some(Type::TypeParam(n)) if n.starts_with("T_") && f.type_params.contains(n)));
        if !implicit || !omitted(f) || f.is_native || f.task || f.stream || f.qualifier.is_some() {
            continue;
        }
        let Some(returns) = valued_returns(&f.body) else { continue };
        if let Some(bad) = returns.iter().find(|e| certainly_valued(e)) {
            return Err(crate::parser::ParseError::Generic {
                msg: format!(
                    "cannot infer the return type of `{}`: it returns a value whose type is not determined by its untyped parameters (each untyped parameter is a separate type) — write the return type, e.g. `def T {}(T x, ...)`",
                    f.name, f.name
                ),
                line: bad.line, col: bad.col, len: bad.len.max(1),
            });
        }
    }
    Ok(())
}

/// D8 of the implicit-generics design: an array is homogeneous, so a literal mixing two instances
/// of one implicit-parameter struct (`[a(), b()]` with `Response<int, string>` and
/// `Response<int, int>`) is a Boring error, not a rustc mismatch on generated code. Only elements
/// whose type is certain count (a constructor or a call this pass can type); run after
/// [`infer_return_types`] so the written-back return types are visible.
pub fn check_homogeneous_arrays(program: &Program) -> Result<(), crate::parser::ParseError> {
    check_arrays_in(&program.items, 0)
}

/// The first `skip` items only supply declarations (imported ones): their bodies are not checked.
fn check_arrays_in(scope: &[Item], skip: usize) -> Result<(), crate::parser::ParseError> {
    for item in scope {
        if let Item::Mod(m) = item {
            check_arrays_in(&m.items, 0)?;
        }
    }
    let cx = Ctx::new(scope);
    let implicit = |n: &str| cx.structs.get(n).is_some_and(|s| s.type_params.iter().any(|p| p.starts_with("T_")));
    let mut error: Option<crate::parser::ParseError> = None;
    let mut on_expr = |e: &mut Expr| {
        if error.is_some() { return; }
        let ExprKind::Array(elems) = &e.kind else { return };
        let mut first: Option<(String, Type, usize, usize)> = None;
        for el in elems {
            let Some(t) = cx.ty_of(el, &Env::new()) else { continue };
            let Type::Generic(n, _) = &t else { continue };
            let n = n.clone();
            if !implicit(&n) { continue; }
            match &first {
                None => first = Some((n, t, el.line, el.col)),
                Some((fname, ft, fline, fcol)) if *fname == n && *ft != t => {
                    error = Some(crate::parser::ParseError::Generic {
                        msg: format!(
                            "an array holds one type, but this one mixes `{}` (line {fline}:{fcol}) and `{}` (here) — the untyped fields of `{n}` are fixed by each construction, so two constructions with different field types are different types",
                            crate::desugar_array_block::describe_type(ft), crate::desugar_array_block::describe_type(&t)
                        ),
                        line: el.line, col: el.col, len: el.len.max(1),
                    });
                    return;
                }
                _ => {}
            }
        }
    };
    let mut on_stmt = |_: &mut Stmt| {};
    let mut items = scope[skip..].to_vec();
    for item in &mut items {
        match item {
            Item::Fn(f) => Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut f.body),
            Item::Stmt(s) => Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmt(s),
            Item::Let(l) => {
                if let Some(v) = &mut l.value {
                    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.expr(v);
                }
            }
            _ => {}
        }
    }
    match error { Some(e) => Err(e), None => Ok(()) }
}

/// The construction half of D3 (implicit generics, phase 3): an untyped struct field fixed by an
/// argument that carries no type of its own (`[]`, `{}`, `{=}`, `nil`) can never be resolved,
/// and rustc's message about it names a generated parameter. Reported here instead, naming the
/// field. Only direct constructions of a struct declared in this file are looked at.
pub fn check_untyped_constructions(program: &Program) -> Result<(), crate::parser::ParseError> {
    check_constructions_in(&program.items, 0)
}

/// One scope (the file, or one `mod`): its own structs, and the constructions of its own items.
/// The first `skip` items only supply declarations (imported ones): their bodies are not checked.
fn check_constructions_in(scope: &[Item], skip: usize) -> Result<(), crate::parser::ParseError> {
    for item in scope {
        if let Item::Mod(m) = item {
            check_constructions_in(&m.items, 0)?;
        }
    }
    let mut structs: HashMap<String, Vec<(String, bool)>> = HashMap::new();
    for item in scope {
        if let Item::Struct(s) = item {
            let fields = s.fields.iter()
                .map(|f| (f.name.clone(), matches!(&f.ty, Type::TypeParam(n) if n.starts_with("T_") && s.type_params.contains(n))))
                .collect::<Vec<_>>();
            if fields.iter().any(|(_, implicit)| *implicit) {
                structs.insert(s.name.clone(), fields);
            }
        }
    }
    if structs.is_empty() {
        return Ok(());
    }
    let mut error: Option<crate::parser::ParseError> = None;
    let mut check = |e: &mut Expr| {
        if error.is_some() { return; }
        let ExprKind::Call(callee, args) = &e.kind else { return };
        let ExprKind::Var(name) = &callee.kind else { return };
        let Some(fields) = structs.get(name) else { return };
        for (i, a) in args.iter().enumerate() {
            let field = match &a.label {
                Some(l) => fields.iter().find(|(n, _)| n == l),
                None => fields.get(i),
            };
            let Some((fname, true)) = field else { continue };
            let typeless = match &a.value.kind {
                ExprKind::Nil => true,
                ExprKind::Array(v) | ExprKind::Set(v) => v.is_empty(),
                ExprKind::Dict(v) => v.is_empty(),
                _ => false,
            };
            if typeless {
                error = Some(crate::parser::ParseError::Generic {
                    msg: format!(
                        "cannot infer the type of `{name}.{fname}`: the value given has no type of its own (an empty collection or `nil`) and the field is untyped — give the field a type in `struct {name}`, or pass a value that has one"
                    ),
                    line: a.value.line, col: a.value.col, len: a.value.len.max(1),
                });
                return;
            }
        }
    };
    let mut on_stmt = |_: &mut Stmt| {};
    let mut items = scope[skip..].to_vec();
    for item in &mut items {
        match item {
            Item::Fn(f) => Walker { expr: &mut check, stmt: &mut on_stmt }.stmts(&mut f.body),
            Item::Stmt(s) => Walker { expr: &mut check, stmt: &mut on_stmt }.stmt(s),
            Item::Let(l) => {
                if let Some(v) = &mut l.value {
                    Walker { expr: &mut check, stmt: &mut on_stmt }.expr(v);
                }
            }
            _ => {}
        }
    }
    match error { Some(e) => Err(e), None => Ok(()) }
}

/// Own `return <value>` statements of `f` (not those of a closure or nested function), as
/// their expressions. `None` when the answer is ambiguous: a bare `return` mixed with valued
/// ones, or a `return` inside a block-bearing expression this pass does not model.
pub(crate) fn valued_returns(stmts: &[Stmt]) -> Option<Vec<Expr>> {
    let mut body = stmts.to_vec();
    let mut returns: Vec<(usize, usize, Option<Expr>)> = Vec::new();
    // Shared by both walker callbacks, hence the cells.
    let nested: RefCell<HashSet<(usize, usize)>> = RefCell::new(HashSet::new());
    let ambiguous: RefCell<HashSet<(usize, usize)>> = RefCell::new(HashSet::new());
    {
        let mut on_expr = |e: &mut Expr| match &mut e.kind {
            ExprKind::Closure(_, _, ClosureBody::Block(stmts), _, _) => collect_positions(stmts, &mut nested.borrow_mut()),
            ExprKind::TrailingArrayBlock { body, .. } => collect_positions(body, &mut ambiguous.borrow_mut()),
            ExprKind::TryElseBlock(a, b) => {
                collect_positions(a, &mut ambiguous.borrow_mut());
                collect_positions(b, &mut ambiguous.borrow_mut());
            }
            _ => {}
        };
        let mut on_stmt = |s: &mut Stmt| match s {
            Stmt::Return(r) => returns.push((r.line, r.col, r.value.clone())),
            Stmt::Fn(inner) => collect_positions(&mut inner.body, &mut nested.borrow_mut()),
            _ => {}
        };
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
    }
    let own: Vec<_> = returns.into_iter().filter(|(l, c, _)| !nested.borrow().contains(&(*l, *c))).collect();
    if own.iter().any(|(l, c, _)| ambiguous.borrow().contains(&(*l, *c))) {
        return None;
    }
    let valued: Vec<Expr> = own.iter().filter_map(|(_, _, v)| v.clone()).collect();
    if !valued.is_empty() && valued.len() != own.len() {
        return None;
    }
    Some(valued)
}

fn collect_positions(stmts: &mut [Stmt], out: &mut HashSet<(usize, usize)>) {
    let mut on_expr = |_: &mut Expr| {};
    let mut on_stmt = |s: &mut Stmt| {
        if let Stmt::Return(r) = s {
            out.insert((r.line, r.col));
        }
    };
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(stmts);
}

/// The parser gives every `def` without a written return type `Some(Void)`, so "omitted" cannot be
/// told from a written `void`. The two only differ when the body says `return <value>`, which is
/// contradictory under an explicit `void` anyway; in that case the written-back type replaces it.
pub(crate) fn omitted(f: &FnDecl) -> bool {
    matches!(f.return_ty, None | Some(Type::Void))
}

fn eligible(f: &FnDecl) -> bool {
    omitted(f)
        && f.qualifier.is_none()
        // Only the fresh parameters of untyped parameters (`infer_param_generics`): the rest of a
        // generic function's signature is the author's.
        && f.type_params.iter().all(|p| p.starts_with("T_"))
        && f.implicit_quals.is_empty()
        && !f.is_native
        && !f.task
        && !f.stream
        && !f.return_mutable
        && !f.attrs.iter().any(|a| a.name == "api")
}

/// Scalar and string spellings as the parser produces them for a written annotation.
fn named(n: &str) -> Type {
    Type::Named(n.to_string())
}

fn is_numeric(t: &Type) -> bool {
    matches!(t, Type::Named(n) if matches!(n.as_str(),
        "int" | "uint" | "uint8" | "uint16" | "uint32" | "uint64" | "uint128"
        | "int8" | "int16" | "int32" | "int64" | "int128" | "float" | "float32" | "float64"))
}

/// A type built only from names, generics, arrays, sets, dicts, optionals and tuples: nothing
/// this pass would have to reason about ownership for.
/// A bare type parameter of the function itself (the fresh one of an untyped parameter).
fn own_param(f: &FnDecl, t: &Type) -> bool {
    matches!(t, Type::TypeParam(n) if f.type_params.contains(n))
}

fn is_plain(t: &Type) -> bool {
    match t {
        Type::Named(_) => true,
        Type::Generic(_, args) => args.iter().all(is_plain),
        Type::Optional(i) | Type::Array(i) | Type::Set(i) => is_plain(i),
        Type::Dict(k, v) => is_plain(k) && is_plain(v),
        Type::Tuple(es) => es.iter().all(is_plain),
        _ => false,
    }
}

fn mentions(t: &Type, params: &[String]) -> bool {
    match t {
        Type::TypeParam(n) | Type::Named(n) => params.contains(n),
        Type::Generic(n, args) => params.contains(n) || args.iter().any(|a| mentions(a, params)),
        Type::Optional(i) | Type::Array(i) | Type::Set(i) | Type::Qualified(i, _) | Type::Mut(i) => mentions(i, params),
        Type::Dict(k, v) => mentions(k, params) || mentions(v, params),
        Type::Tuple(es) => es.iter().any(|e| mentions(e, params)),
        _ => false,
    }
}

/// Binds the type parameters named in `pat` against `act`; `false` on a conflict or a shape this
/// pass does not model.
fn unify(pat: &Type, act: &Type, params: &[String], b: &mut HashMap<String, Type>) -> bool {
    if !mentions(pat, params) {
        return true;
    }
    match pat {
        Type::TypeParam(n) | Type::Named(n) if params.contains(n) => match b.get(n) {
            Some(prev) => prev == act,
            None => {
                b.insert(n.clone(), act.clone());
                true
            }
        },
        Type::Optional(p) => match act {
            Type::Optional(a) => unify(p, a, params, b),
            other => unify(p, other, params, b),
        },
        Type::Array(p) => matches!(act, Type::Array(a) if unify(p, a, params, b)),
        Type::Set(p) => matches!(act, Type::Set(a) if unify(p, a, params, b)),
        Type::Dict(pk, pv) => {
            matches!(act, Type::Dict(ak, av) if unify(pk, ak, params, b) && unify(pv, av, params, b))
        }
        Type::Tuple(ps) => match act {
            Type::Tuple(as_) if ps.len() == as_.len() => ps.iter().zip(as_).all(|(p, a)| unify(p, a, params, b)),
            _ => false,
        },
        Type::Generic(pn, pargs) => match act {
            Type::Generic(an, aargs) if pn == an && pargs.len() == aargs.len() => {
                pargs.iter().zip(aargs).all(|(p, a)| unify(p, a, params, b))
            }
            _ => false,
        },
        _ => false,
    }
}

/// `t` with every type parameter replaced by its binding; `None` if one is unbound.
fn subst(t: &Type, b: &HashMap<String, Type>) -> Option<Type> {
    Some(match t {
        Type::TypeParam(n) => b.get(n)?.clone(),
        Type::Named(n) => b.get(n).cloned().unwrap_or_else(|| t.clone()),
        Type::Generic(n, args) => {
            Type::Generic(n.clone(), args.iter().map(|a| subst(a, b)).collect::<Option<_>>()?)
        }
        Type::Optional(i) => Type::Optional(Box::new(subst(i, b)?)),
        Type::Array(i) => Type::Array(Box::new(subst(i, b)?)),
        Type::Set(i) => Type::Set(Box::new(subst(i, b)?)),
        Type::Dict(k, v) => Type::Dict(Box::new(subst(k, b)?), Box::new(subst(v, b)?)),
        Type::Tuple(es) => Type::Tuple(es.iter().map(|e| subst(e, b)).collect::<Option<_>>()?),
        _ => return None,
    })
}

/// A plain identifier type parameter (not a qualifier parameter `^q`, nor a const parameter).
fn is_simple_param(p: &str) -> bool {
    !p.is_empty() && p.chars().all(|c| c.is_alphanumeric() || c == '_')
}

struct Ctx<'a> {
    structs: HashMap<String, &'a StructDecl>,
    enums: HashMap<String, &'a EnumDecl>,
    fns: HashMap<String, Vec<&'a FnDecl>>,
    memo: RefCell<HashMap<String, Option<Type>>>,
    in_progress: RefCell<HashSet<String>>,
}

impl<'a> Ctx<'a> {
    fn new(items: &'a [Item]) -> Self {
        let mut cx = Ctx {
            structs: HashMap::new(),
            enums: HashMap::new(),
            fns: HashMap::new(),
            memo: RefCell::new(HashMap::new()),
            in_progress: RefCell::new(HashSet::new()),
        };
        let mut duplicates: HashSet<String> = HashSet::new();
        #[allow(clippy::collapsible_match)]
        for item in items {
            match item {
                Item::Struct(s) => {
                    if cx.structs.insert(s.name.clone(), s).is_some() {
                        duplicates.insert(s.name.clone());
                    }
                }
                Item::Enum(e) => {
                    if cx.enums.insert(e.name.clone(), e).is_some() {
                        duplicates.insert(e.name.clone());
                    }
                }
                Item::Fn(f) if f.qualifier.is_none() => cx.fns.entry(f.name.clone()).or_default().push(f),
                _ => {}
            }
        }
        // A name declared twice is ambiguous: forget it rather than pick one.
        for d in duplicates {
            cx.structs.remove(&d);
            cx.enums.remove(&d);
        }
        cx
    }

    /// Return type of the top-level function `name`, written or inferred. `None` for void,
    /// unknown, overloaded, generic and recursive functions.
    fn fn_ret(&self, name: &str) -> Option<Type> {
        let cands = self.fns.get(name)?;
        if cands.len() != 1 {
            return None;
        }
        let f = cands[0];
        if f.type_params.iter().any(|p| !p.starts_with("T_")) || f.qualifier.is_some() {
            return None;
        }
        if let (false, Some(t)) = (omitted(f), &f.return_ty) {
            return (is_plain(t) && !matches!(t, Type::Named(n) if n == "void" || n == "Self")).then(|| t.clone());
        }
        if let Some(done) = self.memo.borrow().get(name) {
            return done.clone();
        }
        if !self.in_progress.borrow_mut().insert(name.to_string()) {
            return None; // a cycle: recursion keeps its type unwritten
        }
        let result = if eligible(f) { self.infer_fn(f) } else { None };
        self.in_progress.borrow_mut().remove(name);
        self.memo.borrow_mut().insert(name.to_string(), result.clone());
        result
    }

    /// The flat environment of one body: its parameters, then each `let` in source order
    /// (a name bound twice with different types is unknown).
    fn body_env(&self, f: Option<&FnDecl>, stmts: &[Stmt]) -> Env {
        let mut env: Env = HashMap::new();
        if let Some(f) = f {
            for p in &f.params {
                let t = p.ty.clone().filter(|t| is_plain(t) || own_param(f, t));
                bind(&mut env, &p.name, t);
            }
        }
        let mut body = stmts.to_vec();
        {
            let mut on_expr = |_: &mut Expr| {};
            let mut on_stmt = |s: &mut Stmt| {
                if let Stmt::Let(l) = s {
                    let t = match (&l.ty, &l.value) {
                        (Some(t), _) => Some(t.clone()).filter(is_plain),
                        (None, Some(v)) => self.ty_of(v, &env),
                        (None, None) => None,
                    };
                    bind(&mut env, &l.name, t);
                }
            };
            Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
        }
        env
    }

    fn infer_fn(&self, f: &FnDecl) -> Option<Type> {
        let returns = valued_returns(&f.body)?;
        if returns.is_empty() {
            return None;
        }
        let env = self.body_env(Some(f), &f.body);
        let mut result: Option<Type> = None;
        for r in &returns {
            let t = self.ty_of(r, &env)?;
            match &result {
                None => result = Some(t),
                Some(prev) if *prev == t => {}
                Some(_) => return None,
            }
        }
        result.filter(|t| is_plain(t) || own_param(f, t))
    }

    fn ty_of(&self, e: &Expr, env: &Env) -> Option<Type> {
        match &e.kind {
            ExprKind::Int(_) => Some(named("int")),
            ExprKind::Float(_) => Some(named("float")),
            ExprKind::Str(_) | ExprKind::StringInterp(_) => Some(named("string")),
            ExprKind::Bool(_) => Some(named("bool")),
            ExprKind::Var(n) => env.get(n).cloned().flatten(),
            ExprKind::Cast(_, t) => Some(t.clone()).filter(is_plain),
            ExprKind::Array(es) => {
                let first = self.ty_of(es.first()?, env)?;
                for x in &es[1..] {
                    if self.ty_of(x, env)? != first {
                        return None;
                    }
                }
                Some(Type::Array(Box::new(first)))
            }
            ExprKind::Tuple(es) => {
                Some(Type::Tuple(es.iter().map(|x| self.ty_of(x, env)).collect::<Option<Vec<_>>>()?))
            }
            ExprKind::UnaryOp(op, x) => match op {
                UnaryOp::Not => Some(named("bool")),
                UnaryOp::Neg | UnaryOp::BitNot => self.ty_of(x, env).filter(is_numeric),
            },
            ExprKind::BinOp(op, l, r) => match op {
                BinOp::Eq | BinOp::NotEq | BinOp::RefEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq
                | BinOp::GtEq | BinOp::And | BinOp::Or | BinOp::Is | BinOp::IsNot => Some(named("bool")),
                _ => {
                    let (lt, rt) = (self.ty_of(l, env)?, self.ty_of(r, env)?);
                    // `x + 1` on an untyped parameter: its `Add<isize, Output = T>` bound keeps the type.
                    if matches!(lt, Type::TypeParam(_)) && matches!(r.kind, ExprKind::Int(_) | ExprKind::Float(_)) {
                        return Some(lt);
                    }
                    let text_add = *op == BinOp::Add && lt == named("string");
                    (lt == rt && (is_numeric(&lt) || text_add || matches!(lt, Type::TypeParam(_)))).then_some(lt)
                }
            },
            ExprKind::Field(base, field) => self.field_ty(base, field, env),
            ExprKind::Call(callee, args) => match &callee.kind {
                ExprKind::Var(name) => match self.structs.get(name) {
                    Some(s) => self.construct_ty(s, args, env),
                    // A callee's own type parameter means nothing in the caller.
                    None => self.fn_ret(name).filter(is_plain),
                },
                _ => None,
            },
            ExprKind::GenericCall(callee, targs, _) => match &callee.kind {
                ExprKind::Var(name) => {
                    let s = self.structs.get(name)?;
                    (s.type_params.len() == targs.len() && targs.iter().all(is_plain))
                        .then(|| Type::Generic(name.clone(), targs.clone()))
                }
                _ => None,
            },
            // `x.len()` on an untyped parameter: its `BoringLen` bound returns a `uint`.
            ExprKind::MethodCall(recv, method, a)
                if method == "len" && a.is_empty() && matches!(self.ty_of(recv, env), Some(Type::TypeParam(_))) => Some(named("uint")),
            ExprKind::MethodCall(recv, method, _) => self.type_member_ty(recv, method),
            _ => None,
        }
    }

    /// `Enum.Variant` (a unit variant) as a value.
    fn field_ty(&self, base: &Expr, field: &str, env: &Env) -> Option<Type> {
        if let ExprKind::Var(n) = &base.kind {
            if !env.contains_key(n) {
                if let Some(en) = self.enums.get(n) {
                    return (en.type_params.is_empty() && en.variants.iter().any(|v| v.name == field))
                        .then(|| named(n));
                }
            }
        }
        let (sname, args) = match self.ty_of(base, env)? {
            Type::Named(n) => (n, Vec::new()),
            Type::Generic(n, a) => (n, a),
            _ => return None,
        };
        let s = self.structs.get(&sname)?;
        let fd = s.fields.iter().find(|f| f.name == field)?;
        if s.type_params.len() != args.len() || !s.type_params.iter().all(|p| is_simple_param(p)) {
            return None;
        }
        let b: HashMap<String, Type> = s.type_params.iter().cloned().zip(args).collect();
        subst(&fd.ty, &b).filter(is_plain)
    }

    /// `Enum.Variant(...)` and a `type def` factory call `Type.make(...)`.
    fn type_member_ty(&self, recv: &Expr, method: &str) -> Option<Type> {
        let ExprKind::Var(n) = &recv.kind else { return None };
        if let Some(en) = self.enums.get(n) {
            if en.type_params.is_empty() && en.variants.iter().any(|v| v.name == method) {
                return Some(named(n));
            }
            let tm = en.type_methods.iter().find(|m| m.name == method)?;
            return plain_member_ret(tm, &en.type_params);
        }
        let s = self.structs.get(n)?;
        let tm = s.type_methods.iter().find(|m| m.name == method)?;
        plain_member_ret(tm, &s.type_params)
    }

    fn construct_ty(&self, s: &StructDecl, args: &[Arg], env: &Env) -> Option<Type> {
        self.construct_detail(s, args, env).map(|(t, _)| t)
    }

    /// [`Self::construct_ty`], plus the parameters that were fixed by a field's default value
    /// because the construction left that field out (D11), for `--show-types`.
    fn construct_detail(&self, s: &StructDecl, args: &[Arg], env: &Env) -> Option<(Type, HashSet<String>)> {
        let mut from_default: HashSet<String> = HashSet::new();
        if s.type_params.is_empty() {
            return Some((named(&s.name), from_default));
        }
        if !s.type_params.iter().all(|p| is_simple_param(p)) || !s.inits.is_empty() {
            return None;
        }
        let mut bindings: HashMap<String, Type> = HashMap::new();
        for (i, arg) in args.iter().enumerate() {
            if arg.spread {
                return None;
            }
            let field = match &arg.label {
                Some(l) => s.fields.iter().find(|f| &f.name == l)?,
                None => s.fields.get(i)?,
            };
            if mentions(&field.ty, &s.type_params) {
                // `nil` for an optional field binds nothing (D4): the other fields decide.
                if matches!(arg.value.kind, ExprKind::Nil) && matches!(field.ty, Type::Optional(_)) {
                    continue;
                }
                let at = self.ty_of(&arg.value, env)?;
                if !is_plain(&at) || !unify(&field.ty, &at, &s.type_params, &mut bindings) {
                    return None;
                }
            }
        }
        // A field the construction leaves out falls back on its default, which then types the
        // parameter (D11: a supplied argument has already won above).
        for (i, field) in s.fields.iter().enumerate() {
            let supplied = args.iter().enumerate().any(|(j, a)| match &a.label {
                Some(l) => *l == field.name,
                None => j == i,
            });
            let Some(default) = &field.default else { continue };
            if supplied || !mentions(&field.ty, &s.type_params) {
                continue;
            }
            let dt = self.ty_of(default, &Env::new())?;
            let before: HashSet<String> = bindings.keys().cloned().collect();
            if !is_plain(&dt) || !unify(&field.ty, &dt, &s.type_params, &mut bindings) {
                return None;
            }
            from_default.extend(bindings.keys().filter(|k| !before.contains(*k)).cloned());
        }
        let targs: Vec<Type> =
            s.type_params.iter().map(|p| bindings.get(p).cloned()).collect::<Option<Vec<_>>>()?;
        Some((Type::Generic(s.name.clone(), targs), from_default))
    }
}

fn plain_member_ret(tm: &TypeMethod, owner_params: &[String]) -> Option<Type> {
    let t = tm.return_ty.as_ref()?;
    let ok = owner_params.is_empty() && is_plain(t) && !mentions_self(t);
    ok.then(|| t.clone())
}

fn mentions_self(t: &Type) -> bool {
    match t {
        Type::Named(n) => n == "Self" || n == "void",
        Type::Generic(n, a) => n == "Self" || a.iter().any(mentions_self),
        Type::Optional(i) | Type::Array(i) | Type::Set(i) => mentions_self(i),
        Type::Dict(k, v) => mentions_self(k) || mentions_self(v),
        Type::Tuple(es) => es.iter().any(mentions_self),
        _ => false,
    }
}

/// Records a binding; a second binding of the same name with a different type makes it unknown.
fn bind(env: &mut Env, name: &str, ty: Option<Type>) {
    match env.get(name) {
        None => {
            env.insert(name.to_string(), ty);
        }
        Some(prev) if *prev == ty => {}
        Some(_) => {
            env.insert(name.to_string(), None);
        }
    }
}

/// One construction of a struct with untyped fields, with the type each of them was fixed to
/// (`Response<status = int, payload = string>`), for `boring check --show-types` (D7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instantiation {
    pub line: usize,
    pub col: usize,
    pub text: String,
}

/// Every construction of an implicit-parameter struct whose parameters can be determined, in
/// source order. Same typing as the return-type inference: a construction it cannot type is
/// left out. A `mod` is its own scope.
pub fn instantiations(program: &Program) -> Vec<Instantiation> {
    instantiations_with_imports(program, &[])
}

/// [`instantiations`] for a file whose `use`d files' items are known (see
/// [`collect_inference_imports`]): a construction of an imported struct is typed too. Local
/// declarations shadow imports of the same name; only the file's own bodies are listed.
pub fn instantiations_with_imports(program: &Program, imported: &[Item]) -> Vec<Instantiation> {
    let mut out = Vec::new();
    instantiations_in(&program.items, imported, "", &mut out);
    out.sort_by_key(|i| (i.line, i.col));
    out
}

fn instantiations_in(items: &[Item], imported: &[Item], prefix: &str, out: &mut Vec<Instantiation>) {
    let local: HashSet<String> = items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) => Some(f.name.clone()),
            Item::Struct(s) => Some(s.name.clone()),
            Item::Enum(e) => Some(e.name.clone()),
            _ => None,
        })
        .collect();
    let mut scope: Vec<Item> = imported
        .iter()
        .filter(|i| match i {
            Item::Fn(f) => !local.contains(&f.name),
            Item::Struct(s) => !local.contains(&s.name),
            Item::Enum(e) => !local.contains(&e.name),
            _ => true,
        })
        .cloned()
        .collect();
    scope.extend(items.iter().cloned());
    let cx = Ctx::new(&scope);
    let mut found = |cx: &Ctx, f: Option<&FnDecl>, body: &[Stmt]| {
        let env = cx.body_env(f, body);
        let mut body = body.to_vec();
        let mut on_stmt = |_: &mut Stmt| {};
        let mut on_expr = |e: &mut Expr| {
            let ExprKind::Call(callee, args) = &e.kind else { return };
            let ExprKind::Var(name) = &callee.kind else { return };
            let Some(s) = cx.structs.get(name) else { return };
            if !s.type_params.iter().any(|p| p.starts_with("T_")) {
                return;
            }
            let Some((Type::Generic(_, targs), defaulted)) = cx.construct_detail(s, args, &env) else { return };
            let parts: Vec<String> = s
                .type_params
                .iter()
                .zip(&targs)
                .map(|(p, t)| {
                    let mark = if defaulted.contains(p) { " (default)" } else { "" };
                    format!("{} = {}{mark}", p.strip_prefix("T_").unwrap_or(p), crate::show_types::describe_type(t))
                })
                .collect();
            out.push(Instantiation { line: e.line, col: e.col, text: format!("{prefix}{name}<{}>", parts.join(", ")) });
        };
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
    };
    let mut top: Vec<Stmt> = Vec::new();
    for item in items {
        match item {
            Item::Fn(f) if f.qualifier.is_none() => found(&cx, Some(f), &f.body),
            Item::Stmt(s) => top.push(s.clone()),
            Item::Let(l) => top.push(Stmt::Let(l.clone())),
            _ => {}
        }
    }
    found(&cx, None, &top);
    for item in items {
        if let Item::Mod(m) = item {
            instantiations_in(&m.items, &[], &format!("{prefix}{}.", m.name), out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn inferred(src: &str, fn_name: &str) -> Option<Type> {
        let program = parse(lex(src).expect("lex error")).expect("parse error");
        let ty = program.items.iter().find_map(|i| match i {
            Item::Fn(f) if f.name == fn_name => Some(f.return_ty.clone()),
            _ => None,
        })?;
        // An untouched function keeps the parser's default `Some(Void)`.
        ty.filter(|t| *t != Type::Void)
    }

    fn n(s: &str) -> Type {
        Type::Named(s.to_string())
    }

    #[test]
    fn typeless_argument_is_also_caught_inside_a_mod() {
        let src = "mod m:\n    struct R:\n        payload\n\n    def make():\n        return R(payload = nil)\n";
        assert!(parse_err(src).expect("should be rejected").contains("`R.payload`"));
    }

    #[test]
    fn typeless_argument_for_an_untyped_field_is_an_error() {
        for v in ["[]", "{}", "{=}", "nil"] {
            let src = format!("struct R:\n    status\n    payload\n\ndef main():\n    let r = R(status = 1, payload = {v})\n");
            let e = parse_err(&src).expect("should be rejected");
            assert!(e.contains("cannot infer the type of `R.payload`"), "{v}: {e}");
        }
        let top = "struct R:\n    status\n    payload\n\nlet r = R(status = 1, payload = [])\n";
        assert!(parse_err(top).expect("top-level let").contains("`R.payload`"));
        // A typed field, a typed argument, and a positional empty argument for a typed field are fine.
        let ok = "struct R:\n    status\n    [int] xs\n\ndef main():\n    let r = R(status = 1, xs = [])\n    let q = R(2, [])\n";
        assert!(parse_err(ok).is_none());
    }

    fn parse_err(src: &str) -> Option<String> {
        parse(lex(src).expect("lex error")).err().map(|e| e.to_string())
    }

    #[test]
    fn untyped_params_with_an_untypable_value_return_are_an_error() {
        let larger = "def larger(a, b):\n    if a < b:\n        return b\n    return a\n";
        let e = parse_err(larger).expect("should be rejected");
        assert!(e.contains("cannot infer the return type of `larger`"), "{e}");
        // `a + b` mixes two separate types: the operator check (D5) rejects it first.
        let e = parse_err("def add(a, b):\n    return a + b\n").expect("should be rejected");
        assert!(e.contains("another untyped parameter"), "{e}");
    }

    #[test]
    fn recursive_functions_with_a_value_need_a_written_return_type() {
        let e = parse_err("def fact(int n):\n    if n < 2:\n        return 1\n    return n * fact(n - 1)\n").expect("rejected");
        assert!(e.contains("`fact` is recursive"), "{e}");
        // Mutual recursion.
        let e = parse_err("def even(int n):\n    if n == 0:\n        return true\n    return odd(n - 1)\n\ndef odd(int n):\n    if n == 0:\n        return false\n    return even(n - 1)\n").expect("rejected");
        assert!(e.contains("is recursive"), "{e}");
        // Written return type, a void recursive function, and a non-recursive one are fine.
        assert!(parse_err("int fact(int n):\n    if n < 2:\n        return 1\n    return n * fact(n - 1)\n").is_none());
        assert!(parse_err("def countdown(int n):\n    if n > 0:\n        print n\n        countdown(n - 1)\n").is_none());
        assert!(parse_err("def one(int n):\n    return 1\n").is_none());
    }

    #[test]
    fn overloaded_names_are_not_taken_for_recursion() {
        let src = "def f(int a):\n    if a > 0:\n        return 1\n    return f(1.5)\n\ndef f(float a):\n    return 2\n";
        // Not a recursion error; the untypable `f(1.5)` makes it the overload one.
        let e = parse_err(src).expect("rejected");
        assert!(e.contains("is overloaded") && !e.contains("recursive"), "{e}");
        // Written on the overload that calls the other, it is fine.
        let ok = "int f(int a):\n    if a > 0:\n        return 1\n    return f(1.5)\n\ndef f(float a):\n    return 2\n";
        assert!(parse_err(ok).is_none());
    }

    #[test]
    fn an_array_cannot_mix_two_instances_of_an_implicit_struct() {
        let base = "struct R:\n    status\n    payload\n\ndef a():\n    return R(status = 1, payload = \"x\")\n\ndef b():\n    return R(status = 1, payload = 2)\n\n";
        let e = parse_err(&format!("{base}let xs = [a(), b()]\n")).expect("rejected");
        assert!(e.contains("mixes `R<int, string>` (line 11:") && e.contains("`R<int, int>` (here)"), "{e}");
        let e = parse_err(&format!("{base}let ys = [R(status = 1, payload = 2), R(status = 1, payload = \"y\")]\n")).expect("rejected");
        assert!(e.contains("an array holds one type"), "{e}");
        // The same instance twice, and one instance with an unknown call, are fine.
        assert!(parse_err(&format!("{base}let xs = [a(), a()]\n")).is_none());
        assert!(parse_err(&format!("{base}let zs = [a(), unknown()]\n")).is_none());
    }

    #[test]
    fn typable_or_void_untyped_functions_are_not_reported() {
        assert!(parse_err("def keep(x):\n    return x\n").is_none());
        assert!(parse_err("def twice(n):\n    return n + n\n").is_none());
        assert!(parse_err("def log(x):\n    print x\n").is_none());
        assert!(parse_err("def log(x): print x\n").is_none());
        // A call may be void: not reported.
        assert!(parse_err("def g(x):\n    print x\n\ndef f(x):\n    return g(x)\n").is_none());
    }

    #[test]
    fn mod_members_get_the_same_checks() {
        let e = parse_err("mod m:\n    def pick(a, b):\n        if a < b:\n            return b\n        return a\n\nprint m.pick(1, 2)\n").expect("should be rejected");
        assert!(e.contains("cannot infer the return type of `pick`"), "{e}");
        // Nested mod too.
        assert!(parse_err("mod a:\n    mod b:\n        def pick(a, b):\n            if a < b:\n                return b\n            return a\n").is_some());
        // Typable and void ones inside a mod are fine (and get inferred).
        assert!(parse_err("mod m:\n    def keep(x):\n        return x\n    def log(x):\n        print x\n").is_none());
        let p = parse(lex("mod m:\n    def keep(x):\n        return x\n").unwrap()).unwrap();
        let Item::Mod(m) = &p.items[0] else { panic!() };
        let Item::Fn(f) = &m.items[0] else { panic!() };
        assert!(!matches!(f.return_ty, None | Some(Type::Void)));
    }

    #[test]
    fn typed_functions_are_never_reported() {
        assert!(parse_err("def f(int a, int b):\n    return a + b\n").is_none());
    }

    #[test]
    fn untyped_field_default_types_it_unless_overridden() {
        let decl = "struct Req:\n    url\n    retries = 0\n\n";
        let g = |a: Type, b: Type| Some(Type::Generic("Req".into(), vec![a, b]));
        assert_eq!(
            inferred(&format!("{decl}def f():\n    return Req(url = \"/x\")\n"), "f"),
            g(n("string"), n("int"))
        );
        assert_eq!(
            inferred(&format!("{decl}def f():\n    return Req(url = \"/x\", retries = 1.5)\n"), "f"),
            g(n("string"), n("float"))
        );
        // Positionally supplied: the argument wins too.
        assert_eq!(
            inferred(&format!("{decl}def f():\n    return Req(\"/x\", true)\n"), "f"),
            g(n("string"), n("bool"))
        );
    }

    const BOX: &str = "struct Box<T>:\n    let T v\n\n";

    #[test]
    fn literal_return() {
        assert_eq!(inferred("def f():\n    return 3\n", "f"), Some(n("int")));
        assert_eq!(inferred("def f():\n    return \"hi\"\n", "f"), Some(n("string")));
        assert_eq!(inferred("def f():\n    return 1.5\n", "f"), Some(n("float")));
        assert_eq!(inferred("def f():\n    return true\n", "f"), Some(n("bool")));
    }

    #[test]
    fn generic_constructor_return_is_unified() {
        let src = format!("{BOX}def mk():\n    return Box(v = 3)\n");
        assert_eq!(inferred(&src, "mk"), Some(Type::Generic("Box".into(), vec![n("int")])));
        let src = format!("{BOX}def mk():\n    return Box(\"s\")\n");
        assert_eq!(inferred(&src, "mk"), Some(Type::Generic("Box".into(), vec![n("string")])));
    }

    #[test]
    fn one_struct_two_instantiations() {
        let src = format!("{BOX}def a():\n    return Box(v = 1)\n\ndef b():\n    return Box(v = [1, 2])\n");
        assert_eq!(inferred(&src, "a"), Some(Type::Generic("Box".into(), vec![n("int")])));
        assert_eq!(
            inferred(&src, "b"),
            Some(Type::Generic("Box".into(), vec![Type::Array(Box::new(n("int")))]))
        );
    }

    #[test]
    fn non_generic_struct_return() {
        let src = "struct P:\n    int x = 0\n\ndef mk():\n    return P()\n";
        assert_eq!(inferred(src, "mk"), Some(n("P")));
    }

    #[test]
    fn chain_through_another_inferred_function() {
        let src = format!("{BOX}def inner():\n    return Box(v = 1)\n\ndef outer():\n    return inner()\n");
        assert_eq!(inferred(&src, "outer"), Some(Type::Generic("Box".into(), vec![n("int")])));
    }

    #[test]
    fn local_binding_and_field_read() {
        let src = format!("{BOX}def f():\n    let b = Box(v = 7)\n    return b.v\n");
        assert_eq!(inferred(&src, "f"), Some(n("int")));
    }

    #[test]
    fn typed_parameter_flows_into_the_result() {
        let src = format!("{BOX}def wrap(string s):\n    return Box(v = s)\n");
        assert_eq!(inferred(&src, "wrap"), Some(Type::Generic("Box".into(), vec![n("string")])));
    }

    #[test]
    fn arithmetic_and_comparison() {
        assert_eq!(inferred("def f(int a):\n    return a + 1\n", "f"), Some(n("int")));
        assert_eq!(inferred("def f(int a):\n    return a < 3\n", "f"), Some(n("bool")));
    }

    #[test]
    fn enum_variant() {
        let src = "enum Level:\n    Low\n    High\n\ndef f():\n    return Level.High\n";
        assert_eq!(inferred(src, "f"), Some(n("Level")));
    }

    #[test]
    fn agreeing_branches_are_accepted_and_disagreeing_ones_are_not() {
        let ok = "def f(bool c):\n    if c:\n        return 1\n    return 2\n";
        assert_eq!(inferred(ok, "f"), Some(n("int")));
        let bad = "def f(bool c):\n    if c:\n        return 1\n    return \"x\"\n";
        assert_eq!(inferred(bad, "f"), None);
    }

    #[test]
    fn unconstrained_value_is_left_alone() {
        let src = format!("{BOX}def mk():\n    return Box(v = [])\n");
        assert_eq!(inferred(&src, "mk"), None);
    }

    #[test]
    fn recursion_keeps_its_type_unwritten() {
        // Not inferred, and (D4) an error as soon as a returned expression certainly has a value.
        let src = "def f(int n):\n    if n == 0:\n        return 0\n    return f(n - 1)\n";
        assert!(parse_err(src).expect("rejected").contains("`f` is recursive"));
        // Only calls to each other: nothing certainly valued, left alone.
        let mutual = "def a():\n    return b()\n\ndef b():\n    return a()\n";
        assert_eq!(inferred(mutual, "a"), None);
        assert_eq!(inferred(mutual, "b"), None);
    }

    #[test]
    fn written_type_is_never_touched() {
        let src = format!("{BOX}Box<float> mk():\n    return Box(v = 3)\n");
        assert_eq!(inferred(&src, "mk"), Some(Type::Generic("Box".into(), vec![n("float")])));
    }

    #[test]
    fn tail_expression_does_not_make_a_return_type() {
        assert_eq!(inferred("def f():\n    3\n", "f"), None);
    }

    #[test]
    fn bare_return_mixed_with_a_valued_one_is_left_alone() {
        let src = "def f(bool c):\n    if c:\n        return\n    return 1\n";
        assert_eq!(inferred(src, "f"), None);
    }

    #[test]
    fn nested_function_returns_are_not_the_functions() {
        // (A closure cannot contain `return` in Boring, so a nested `def` is the case that matters.)
        let src = "def f():\n    def inner():\n        return \"x\"\n    return 1\n";
        assert_eq!(inferred(src, "f"), Some(n("int")));
    }

    #[test]
    fn api_functions_are_not_inferred() {
        let src = "@api\npub def f():\n    return 1\n";
        assert_eq!(inferred(src, "f"), None);
    }

    #[test]
    fn unknown_call_leaves_the_function_alone() {
        assert_eq!(inferred("def f():\n    return somewhere()\n", "f"), None);
    }

    #[test]
    fn shadowed_binding_with_two_types_is_unknown() {
        let src = "def f(bool c):\n    if c:\n        let x = 1\n    else:\n        let x = \"a\"\n    return x\n";
        assert_eq!(inferred(src, "f"), None);
    }

    fn overload_types(src: &str, fn_name: &str) -> Vec<Option<Type>> {
        let program = parse(lex(src).expect("lex error")).expect("parse error");
        program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Fn(f) if f.name == fn_name => Some(f.return_ty.clone().filter(|t| *t != Type::Void)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn each_overload_is_inferred_from_its_own_parameters() {
        let src = "def f(int a):\n    return a\n\ndef f(string a):\n    return a\n\ndef f(float a):\n    return 2.5\n";
        assert_eq!(overload_types(src, "f"), vec![Some(n("int")), Some(n("string")), Some(n("float"))]);
    }

    #[test]
    fn overload_with_early_return_is_inferred() {
        let src = "def f(int a):\n    if a > 0:\n        return 1\n    return 2\n\ndef f(float a):\n    return 2\n";
        assert_eq!(overload_types(src, "f"), vec![Some(n("int")), Some(n("int"))]);
    }

    #[test]
    fn untypable_overload_return_is_an_error() {
        let src = "def f(int a):\n    return a\n\ndef f(a, b):\n    return a\n\ndef f(float a):\n    return a.x\n";
        let e = parse_err(src).expect("should be rejected");
        assert!(e.contains("`f` is overloaded"), "{e}");
    }

    #[test]
    fn overload_returning_a_call_or_nothing_is_never_reported() {
        let src = "def f(int a):\n    print a\n\ndef f(float a):\n    return g(a)\n";
        assert!(parse_err(src).is_none());
        assert_eq!(overload_types(src, "f"), vec![None, None]);
    }

    #[test]
    fn overload_with_conflicting_returns_is_an_error() {
        let src = "def f(int a):\n    if a > 0:\n        return 1\n    return \"x\"\n\ndef f(float a):\n    return 2\n";
        assert!(parse_err(src).expect("should be rejected").contains("`f` is overloaded"));
    }
}
