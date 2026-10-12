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

//! Resolves and desugars `ExprKind::TrailingArrayBlock` (docs/book.md,
//! "Trailing array-block sugar") — the `Column:` / `Column(args):` sugar
//! that generalizes the existing trailing-closure sugar to a trailing
//! `[dyn Trait]` array argument. Runs once, on the whole `Program`, right
//! after parsing (same pipeline slot as `desugar_labeled_array`, and for the
//! same reason: every consumer downstream — checker, interpreter, all
//! transpiler backends — only ever sees plain, ordinary AST nodes it already
//! knows how to handle).
//!
//! ## What this pass actually is
//!
//! This is simultaneously the "checker" step called for in this feature's
//! design (resolving, per call site, what the ambiguous `IDENT: <block>` /
//! `IDENT(args): <block>` shape actually means) and the "desugar" step
//! (rewriting it away). Both live here, in one pass, because resolving the
//! sugar and rewriting it away are the same piece of information.
//!
//! A `TrailingArrayBlock` node is never passed through unresolved — it is
//! always rewritten into a plain, ordinary node, or this pass returns a hard
//! `ParseError` (reusing that type purely for its existing `line`/`col`/
//! `len`/`msg` accessors and `main.rs`'s existing `report_error` plumbing —
//! this is a resolution error, not a lexical/grammatical one).
//!
//! ## Resolution — decision table
//!
//! `callee`'s name is looked up in this file's own signature table
//! (`Signatures::collect`, gathered from top-level `fn`/`struct`
//! declarations — see "Known limitation" below). What that lookup finds
//! decides the rewrite, and — for the *unresolved* case only — whether the
//! call was written with an explicit (possibly empty) `(...)` argument list
//! also matters (`has_parens`, recorded by the parser):
//!
//! | `callee` resolves to...                                   | Rewrite |
//! |---|---|
//! | known fn/struct-ctor, last param `[dyn SomeTrait]`         | **Collect**: flat array literal, or an imperative `Vec` builder when the block contains `if`/`elif`/`else`/`for` |
//! | known fn/struct-ctor, last param `Fn(...)`                 | **Tail**: ordinary ("pre-existing") ordinary trailing-closure sugar — `callee(...args, (): body)` |
//! | known fn/struct-ctor, any other last param type            | **Error** — clearly names the mismatched type |
//! | not a known callable at all, `has_parens == true`          | **Tail** (same as the `Fn(...)` row above) — matches this shape's pre-existing default meaning (an ordinary zero-arg trailing body/closure) from before this sugar existed, for any callee this pass simply doesn't know about (a builtin, an external/stdlib function, anything not declared in *this* file) |
//! | not a known callable at all, `has_parens == false`          | **Closure literal**: the pre-existing no-paren closure shorthand this exact bare `Ident:` shape always meant before this sugar existed — `callee`'s own name becomes the closure's single implicit parameter |
//!
//! Only the "known callable, wrong last-param type" row is a hard error —
//! every other outcome resolves to *something* runnable, so this sugar never
//! silently breaks a call site whose callee this pass doesn't happen to know
//! about (crucially including builtins/external functions used with the
//! pre-existing "zero-arg trailing body" sugar, e.g. `timeout(...): body` —
//! see docs/book.md's "Trailing closures").
//!
//! ## The rewrite, in detail
//!
//! - **Collect, flat case** (no `if`/`for` anywhere in the block, even
//!   nested): rewrites to a plain `Call(callee, [...args, Array(element_exprs)])`
//!   — textually identical to what a user would have written by hand
//!   (`Column([Text(...), Row([...])])`). No further special-casing is
//!   needed anywhere downstream: struct-construction and free-function call
//!   sites already box each array-literal element into `Box<dyn Trait>` when
//!   the parameter's declared type is `[Trait]` (`emit_let_value`'s
//!   `Type::Array(elem_ty)` arm, `box_if_trait_typed`) — the exact same
//!   machinery that already backs `docs/book.md`'s "Traits as types".
//! - **Collect, control-flow case** (`if`/`elif`/`else`/`for` present
//!   anywhere in the block): rewrites to a `Do` expression (`ExprKind::Do` —
//!   "own scope, last expression is the value", the same semantics `Block`
//!   documents, but with a transpiler emission path that actually handles
//!   every statement kind — see `desugar_trailing_array_block`'s own comment
//!   on why `Do` and not `Block`) that imperatively builds a
//!   `mut [Trait] __children = []` local, one `.push(...)` per leaf item
//!   (conditionally/looped per the original `if`/`for` structure), then
//!   tail-evaluates to `Call(callee, [...args, Var(__children)])`. Pushing a
//!   value into a `[Trait]`-typed local wasn't itself already trait-boxed
//!   anywhere in the transpiler — a small, narrowly-scoped addition to
//!   `emit_methods.rs`'s generic (non-actor) `.push()`/`.extend()` emission
//!   handles that.
//! - **Tail**: rewrites to `Call(callee, [...args, Closure([], None,
//!   ClosureBody::Block(body), throws, task)])` — a zero-parameter trailing
//!   closure appended after `args`, `throws`/`task` inferred exactly like
//!   any other closure body (`infer_closure_throws_task`). This is *exactly*
//!   what the pre-existing "zero-arg trailing body" sugar (`parse_trailing_body`)
//!   already builds for the single-line/no-block forms — this pass produces
//!   the same shape for the multi-line-block form once resolution allows it.
//! - **Closure literal**: rewrites to `Closure([Param { name: <callee's own
//!   name> }], None, ClosureBody::Block(body), throws, task)` — precisely
//!   what the bare `Ident:` shape has always meant whenever `Ident` isn't
//!   being used as a call target, unchanged from before this sugar existed.
//!
//! Nesting (a block-array line that is itself another block-array call,
//! e.g. `Row:` inside `Column:`) falls out for free: every statement in
//! `body` is desugared bottom-up (`desugar_body`) *before* this node's own
//! resolution runs, so a nested `TrailingArrayBlock` is fully resolved and
//! rewritten by the time its parent is examined — by then it's just an
//! ordinary `Call`/`Do`/`Closure` expression like any other.
//!
//! ## Imported callees
//!
//! `desugar_array_block_with_imports` also feeds the signature table from files
//! reached through `use` (same-project siblings, `[deps]` projects, the embedded
//! `boring.*` stdlib), so a library `Column` declared elsewhere resolves exactly
//! like a local one. Visibility follows ordinary import resolution (bare `use`:
//! everything; multi-segment/dep `use`: `pub` items, narrowed by an explicit item
//! list); local declarations shadow imports. Transparent `use Alias as Original`
//! aliases follow the original callable's signature (including local `Stmt::Alias`
//! declarations); `type … as` newtypes remain distinct constructors.
//! Without a `source_dir` (`desugar_array_block`), resolution stays single-file.

use crate::ast::*;
use crate::parser::{infer_closure_throws_task, ParseError};
use std::collections::{HashMap, HashSet};

/// Per-callee signature info needed to resolve the sugar — only the LAST
/// entry of each `Vec<Type>` is ever consulted.
#[derive(Default)]
struct Signatures {
    /// Free-function name -> parameter types, in declaration order.
    fns: HashMap<String, Vec<Type>>,
    /// Struct name -> field types, in declaration order — the implicit,
    /// fully-positional constructor's parameter list (mirrors the same
    /// "no explicit init" convention the checker's own move-check already
    /// relies on for struct construction — see checker/mod.rs's doc comment
    /// on its "committed 'owned" constructor-param table).
    structs: HashMap<String, Vec<Type>>,
    /// Declared trait names — a `[Named(x)]` last-parameter type only
    /// triggers collect semantics when `x` names a real trait (never a
    /// plain struct/enum or a concrete element type — see the decision
    /// table's "any other last param type" row).
    traits: HashSet<String>,
    /// Struct/view name -> what key inference needs: field types, declared
    /// conformances, and zero-argument method return types.
    defs: HashMap<String, StructInfo>,
    /// Struct name -> its last field's name: lets a trailing array-block be passed as a
    /// *labeled* argument when the call already carries labeled arguments (a positional
    /// argument after labeled ones would be matched by position, not as "the last field").
    last_field: HashMap<String, String>,
    /// Best-effort local type environment for the function/method body currently being
    /// desugared: parameters, the enclosing struct's fields, annotated `let`s and keyed-loop
    /// variables. Flat per body (no block scoping) — it only feeds `for … with` key checks.
    locals: std::cell::RefCell<HashMap<String, Type>>,
}

#[derive(Default, Clone)]
struct StructInfo {
    fields: HashMap<String, Type>,
    protocols: Vec<String>,
    methods: HashMap<String, Type>,
}

impl Signatures {
    fn collect_items(&mut self, items: &[Item]) {
        self.collect_items_filtered(items, &|_, _| true);
    }

    /// Make every transparent `use Alias as Original` name resolve like its
    /// original callable. The parser stores top-level aliases as `Item::Alias`
    /// and function-local aliases as `Stmt::Alias`; both carry the original in
    /// `AliasDecl::ty`. Iterate to a fixed point so alias chains work regardless
    /// of declaration order.
    fn collect_aliases(&mut self, items: &[Item]) {
        let mut aliases = Vec::new();
        collect_alias_decls(items, &mut aliases);
        for _ in 0..aliases.len() {
            let mut changed = false;
            for alias in &aliases {
                if alias.newtype { continue; }
                let Type::Named(original) = &alias.ty else { continue; };
                if let Some(params) = self.fns.get(original).cloned() {
                    changed |= self.fns.insert(alias.name.clone(), params).is_none();
                } else if let Some(params) = self.structs.get(original).cloned() {
                    changed |= self.structs.insert(alias.name.clone(), params).is_none();
                }
            }
            if !changed { break; }
        }
    }

    /// Same as `collect_items`, but only records a top-level fn/struct/trait
    /// when `accept(name, is_pub)` says so — used for imported files, where
    /// `pub` and a selective `use x.(A, B)` list gate what is visible.
    fn collect_items_filtered(&mut self, items: &[Item], accept: &dyn Fn(&str, bool) -> bool) {
        let fn_tys = |f: &FnDecl| -> Vec<Type> {
            f.params.iter()
                .map(|p| p.ty.clone().unwrap_or_else(|| Type::Named("_".to_string())))
                .collect()
        };
        for item in items {
            match item {
                Item::Fn(f) | Item::Stmt(Stmt::Fn(f)) if accept(&f.name, f.is_pub) => {
                    self.fns.insert(f.name.clone(), fn_tys(f));
                }
                Item::Struct(s) | Item::Stmt(Stmt::Struct(s)) if accept(&s.name, s.is_pub) => {
                    self.structs.insert(s.name.clone(), s.fields.iter().map(|f| f.ty.clone()).collect());
                    self.defs.insert(s.name.clone(), StructInfo {
                        fields: s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect(),
                        protocols: s.protocols.clone(),
                        methods: s.methods.iter()
                            .filter(|m| m.params.is_empty())
                            .filter_map(|m| Some((m.name.clone(), m.return_ty.clone()?)))
                            .collect(),
                    });
                }
                // Trait names are always recorded: an imported signature's own
                // `[dyn View]` parameter must classify as "collect" even when
                // the importing file never names `View` itself.
                Item::Trait(t) => { self.traits.insert(t.name.clone()); }
                Item::Mod(m) | Item::Stmt(Stmt::Mod(m)) => self.collect_items_filtered(&m.items, accept),
                _ => {}
            }
        }
    }

    /// The resolved last-parameter type for a plain callee name, if it names
    /// a known free function or struct (implicit positional constructor).
    /// `None` means "not a known callable at all" (decision table's bottom
    /// two rows) — not an error by itself.
    fn last_param_type(&self, name: &str) -> Option<&Type> {
        self.fns.get(name).or_else(|| self.structs.get(name)).and_then(|v| v.last())
    }

    fn is_trait(&self, name: &str) -> bool {
        self.traits.contains(name)
    }

    /// Best-effort static type of `e` from the local environment and struct table.
    /// `None` means "unknown" — callers must never reject on it.
    fn infer(&self, e: &Expr) -> Option<Type> {
        match &e.kind {
            ExprKind::Int(_) => Some(Type::Int),
            ExprKind::Float(_) => Some(Type::Float64),
            ExprKind::Var(n) => self.locals.borrow().get(n).cloned(),
            ExprKind::Field(obj, f) => {
                if matches!(&obj.kind, ExprKind::Var(n) if n == "self") {
                    return self.locals.borrow().get(f).cloned();
                }
                let Type::Named(sn) = strip_perm(&self.infer(obj)?) else { return None };
                self.defs.get(&sn)?.fields.get(f).cloned()
            }
            ExprKind::MethodCall(obj, m, args) if args.is_empty() => {
                let Type::Named(sn) = strip_perm(&self.infer(obj)?) else { return None };
                self.defs.get(&sn)?.methods.get(m).cloned()
            }
            ExprKind::Call(callee, _) => match &callee.kind {
                ExprKind::Var(n) if self.defs.contains_key(n) => Some(Type::Named(n.clone())),
                _ => None,
            },
            _ => None,
        }
    }

    /// Element type of an iterable whose type is a known array.
    fn element_type(&self, iterable: &Expr) -> Option<Type> {
        match strip_perm(&self.infer(iterable)?) {
            Type::Array(t) | Type::ArrayN(t, _) => Some(strip_perm(&t)),
            _ => None,
        }
    }

    /// Records parameter types (and, inside a struct/view, its fields) for a body.
    fn enter_body(&self, params: &[Param], owner: Option<&StructDecl>) -> HashMap<String, Type> {
        let mut env = HashMap::new();
        if let Some(sd) = owner {
            for f in &sd.fields { env.insert(f.name.clone(), f.ty.clone()); }
        }
        for p in params {
            if let Some(t) = &p.ty { env.insert(p.name.clone(), t.clone()); }
        }
        std::mem::replace(&mut *self.locals.borrow_mut(), env)
    }

    fn leave_body(&self, saved: HashMap<String, Type>) {
        *self.locals.borrow_mut() = saved;
    }
}

/// Drops `mut`/qualifier wrappers that don't change which type something is.
fn strip_perm(ty: &Type) -> Type {
    match ty {
        Type::Mut(inner) => strip_perm(inner),
        Type::Qualified(inner, OwnerQual::Borrow | OwnerQual::BorrowMut | OwnerQual::Inline
            | OwnerQual::Owned | OwnerQual::Shared) => strip_perm(inner),
        other => other.clone(),
    }
}

/// Why `ty` can't be an `IdentityKey` (docs/design-notes/boring-ui-draft.md, "`IdentityKey` and
/// `Identifiable`"): floats have no total equality; dictionaries, sets and function values are
/// not `Hash`; a mutable-shared qualifier isn't an owned snapshot.
fn non_key_reason(ty: &Type) -> Option<&'static str> {
    match ty {
        Type::Float32 | Type::Float64 => Some("floating-point values have no total equality (NaN)"),
        Type::Named(n) if matches!(n.as_str(), "float" | "float32" | "float64") =>
            Some("floating-point values have no total equality (NaN)"),
        Type::Dict(..) | Type::Set(_) => Some("dictionaries and sets are not hashable"),
        Type::Fn(..) => Some("function values are not hashable"),
        Type::Mut(inner) => non_key_reason(inner),
        Type::Qualified(_, OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Observed) =>
            Some("mutable shared qualifiers are not owned key snapshots"),
        Type::Qualified(inner, _) | Type::Optional(inner) => non_key_reason(inner),
        Type::Tuple(ts) => ts.iter().find_map(non_key_reason),
        _ => None,
    }
}

/// Validates explicit `for … with <key>` expressions and resolves an omitted `with` for an
/// `Identifiable` element (`item.id()`), recursively through a collect block's elements.
fn resolve_identity_keys(elems: &mut [ArrayBlockElem], sigs: &Signatures) -> Result<(), ParseError> {
    for elem in elems {
        match elem {
            ArrayBlockElem::Item(_) => {}
            ArrayBlockElem::If { branches, else_body } => {
                for (_, b) in branches.iter_mut() { resolve_identity_keys(b, sigs)?; }
                if let Some(b) = else_body { resolve_identity_keys(b, sigs)?; }
            }
            ArrayBlockElem::For { vars, iterable, key, body } => {
                let elem_ty = sigs.element_type(iterable);
                if let (Some(t), [v]) = (&elem_ty, vars.as_slice()) {
                    sigs.locals.borrow_mut().insert(v.clone(), t.clone());
                }
                match key {
                    Some(k) => {
                        if let Some(reason) = sigs.infer(k).as_ref().and_then(non_key_reason) {
                            return Err(ParseError::Generic {
                                line: k.line, col: k.col, len: k.len.max(1),
                                msg: format!(
                                    "`for … with` key is not an `IdentityKey`: {reason} — \
                                     key on an integer, string, enum, newtype or other owned `Clone + Eq + Hash` value"
                                ),
                            });
                        }
                    }
                    None => {
                        if let (Some(Type::Named(sn)), [v]) = (&elem_ty, vars.as_slice()) {
                            let identifiable = sigs.defs.get(sn)
                                .is_some_and(|d| d.protocols.iter().any(|p| p == "Identifiable"));
                            if identifiable {
                                let (line, col, len) = (iterable.line, iterable.col, iterable.len);
                                let var = Expr { kind: ExprKind::Var(v.clone()), line, col, len };
                                *key = Some(Expr {
                                    kind: ExprKind::MethodCall(Box::new(var), "id".to_string(), vec![]),
                                    line, col, len,
                                });
                            }
                        }
                    }
                }
                resolve_identity_keys(body, sigs)?;
            }
        }
    }
    Ok(())
}

/// A short, human-readable rendering of `ty` for error messages — deliberately
/// minimal (this pass only ever needs to name the type it rejected, not emit
/// valid Boring syntax back).
fn describe_type(ty: &Type) -> String {
    match ty {
        Type::Int => "int".into(),
        Type::Uint => "uint".into(),
        Type::Uint8 => "uint8".into(),
        Type::Int8 => "int8".into(),
        Type::Int16 => "int16".into(),
        Type::Int32 => "int32".into(),
        Type::Int64 => "int64".into(),
        Type::Int128 => "int128".into(),
        Type::Uint16 => "uint16".into(),
        Type::Uint32 => "uint32".into(),
        Type::Uint64 => "uint64".into(),
        Type::Uint128 => "uint128".into(),
        Type::Float32 => "float32".into(),
        Type::Float64 => "float".into(),
        Type::Str => "string".into(),
        Type::Bool => "bool".into(),
        Type::Nil => "nil".into(),
        Type::Void => "void".into(),
        Type::Never => "never".into(),
        Type::Named(n) => n.clone(),
        Type::Optional(t) => format!("{}?", describe_type(t)),
        Type::Array(t) => format!("[{}]", describe_type(t)),
        Type::ArrayN(t, n) => format!("[{}, {}]", describe_type(t), n),
        Type::Tuple(ts) => format!("({})", ts.iter().map(describe_type).collect::<Vec<_>>().join(", ")),
        Type::Dict(k, v) => format!("{{{}={}}}", describe_type(k), describe_type(v)),
        Type::Set(t) => format!("{{{}}}", describe_type(t)),
        Type::Dyn(t) => describe_type(t),
        Type::Impl(t) => format!("<{}>", describe_type(t)),
        Type::Qualified(t, _) => format!("{}'...", describe_type(t)),
        other => format!("{:?}", other),
    }
}

/// A short, human-readable name for a statement kind that isn't allowed
/// inside a collect block — used only in `stmt_to_array_block_elem`'s error
/// message. `Expr`/`If`/`For` never reach this (they're handled, not
/// rejected, by the caller).
fn describe_stmt_kind(s: &Stmt) -> &'static str {
    match s {
        Stmt::Let(_) | Stmt::LetDestructure(_) => "let",
        Stmt::Return(_) => "return",
        Stmt::Break(..) => "break",
        Stmt::Continue(_) => "continue",
        Stmt::Throw(_) => "throw",
        Stmt::IfLet(_) => "if let",
        Stmt::Match(_) => "match",
        Stmt::While(_) => "while",
        Stmt::WhileLet(_) => "while let",
        Stmt::DoWhile(_) => "do-while",
        Stmt::Loop(_) => "loop",
        Stmt::Wait(..) => "wait",
        Stmt::Guard(_) => "guard",
        Stmt::Try(_) => "try",
        Stmt::Defer(_) => "defer",
        Stmt::Fn(_) => "nested function declaration",
        Stmt::Struct(_) => "nested struct declaration",
        Stmt::Enum(_) => "nested enum declaration",
        Stmt::Mod(_) => "nested module",
        Stmt::Alias(_) => "type alias",
        Stmt::Yield(..) => "yield",
        Stmt::Comment(_) => "comment",
        Stmt::KernelBlock(_) => "kernel block",
        Stmt::With(_) => "with",
        Stmt::Expr(_) | Stmt::If(_) | Stmt::For(_) => unreachable!("handled by the caller, never described"),
    }
}

/// Whether an element list contains an `if`/`for`, anywhere including
/// nested inside another `if`/`for` — determines flat vs. control-flow
/// desugaring (scoping is per-`TrailingArrayBlock` node: a nested block-array
/// call's own control flow doesn't count, since it desugars independently
/// into its own `Call`/`Do`/`Closure` before this check ever runs on the
/// outer one).
fn elems_have_control_flow(elems: &[ArrayBlockElem]) -> bool {
    elems.iter().any(|e| matches!(e, ArrayBlockElem::If { .. } | ArrayBlockElem::For { .. }))
}

/// Converts a resolved-as-collect `TrailingArrayBlock`'s (already-desugared)
/// `Vec<Stmt>` body into the restricted `ArrayBlockElem` shape the
/// flat/control-flow builder below expects: `Stmt::Expr` becomes one array
/// element, `Stmt::If`/`Stmt::For` become the matching `ArrayBlockElem`
/// variant (recursing into their own bodies the same way), and every other
/// statement kind is rejected — a collect block only ever means "one array
/// element per line, optionally under `if`/`elif`/`else`/`for`", the same
/// restriction this sugar has always documented (docs/book.md's scoping
/// rules), just enforced here instead of by a dedicated parser grammar (see
/// `parse_array_block.rs`'s module doc comment for why: the block's meaning
/// isn't known until resolution runs, so it has to be parsed generically
/// first).
fn stmts_to_array_block_elems(stmts: Vec<Stmt>, line: usize, col: usize, len: usize) -> Result<Vec<ArrayBlockElem>, ParseError> {
    stmts.into_iter().map(|s| stmt_to_array_block_elem(s, line, col, len)).collect()
}

fn stmt_to_array_block_elem(stmt: Stmt, line: usize, col: usize, len: usize) -> Result<ArrayBlockElem, ParseError> {
    match stmt {
        Stmt::Expr(e) => Ok(ArrayBlockElem::Item(e)),
        Stmt::If(i) => {
            let mut branches = Vec::with_capacity(i.branches.len());
            for (c, b) in i.branches { branches.push((c, stmts_to_array_block_elems(b, line, col, len)?)); }
            let else_body = i.else_body.map(|b| stmts_to_array_block_elems(b, line, col, len)).transpose()?;
            Ok(ArrayBlockElem::If { branches, else_body })
        }
        Stmt::For(f) => Ok(ArrayBlockElem::For {
            vars: f.vars, iterable: f.iterable, key: f.key, body: stmts_to_array_block_elems(f.body, line, col, len)?,
        }),
        other => Err(ParseError::Generic {
            line, col, len,
            msg: format!(
                "trailing array-block sugar: this block collects into an array, so each line must \
                 be a plain expression, or `if`/`elif`/`else`/`for` — found a `{}` statement, which \
                 isn't meaningful inside a collect block",
                describe_stmt_kind(&other),
            ),
        }),
    }
}

pub fn desugar_array_block(program: Program) -> Result<Program, ParseError> {
    desugar_array_block_with_imports(program, None, &HashMap::new())
}

/// Like `desugar_array_block`, but also resolves callee signatures declared in
/// files reached through `use` — same-project siblings (bare `use name`,
/// `use sub.mod.*`), `boring.toml` `[deps]` projects (`use dep.mod.Item`), and
/// the embedded first-party stdlib (`use boring.mod.*`). `source_dir` is the
/// directory of the file being compiled (`None` skips every on-disk import).
///
/// Visibility follows ordinary import resolution: a bare single-segment `use`
/// imports every top-level item (book.md, "Same-project multi-file"); any
/// multi-segment or dep-qualified `use` imports only `pub` items, narrowed to
/// the named ones for `use mod.A, B`. Local declarations always shadow imports.
/// An unreadable or unparseable import is skipped — the checker/transpiler
/// re-parse it independently and report the real error. Transparent
/// `use Alias as Original` aliases inherit the original callable's signature.
pub fn desugar_array_block_with_imports(
    mut program: Program,
    source_dir: Option<&std::path::Path>,
    deps: &HashMap<String, std::path::PathBuf>,
) -> Result<Program, ParseError> {
    let mut sigs = Signatures::default();
    if let Some(dir) = source_dir {
        let mut visited = HashSet::new();
        sigs.collect_imports(&program.items, dir, deps, &mut visited);
    }
    // Local declarations last, so they override any imported name.
    sigs.collect_items(&program.items);
    sigs.collect_aliases(&program.items);
    program.items = desugar_items(program.items, &sigs)?;
    Ok(program)
}

fn collect_alias_decls<'a>(items: &'a [Item], out: &mut Vec<&'a AliasDecl>) {
    for item in items {
        match item {
            Item::Alias(alias) => out.push(alias),
            Item::Fn(f) => collect_stmt_alias_decls(&f.body, out),
            Item::Struct(s) => {
                for method in &s.methods { collect_stmt_alias_decls(&method.body, out); }
                for init in &s.inits { collect_stmt_alias_decls(&init.body, out); }
                for method in &s.type_methods { collect_stmt_alias_decls(&method.body, out); }
            }
            Item::Enum(e) => {
                for method in &e.methods { collect_stmt_alias_decls(&method.body, out); }
            }
            Item::Ext(e) => {
                for method in &e.methods { collect_stmt_alias_decls(&method.body, out); }
            }
            Item::Mod(m) => collect_alias_decls(&m.items, out),
            Item::Stmt(stmt) => collect_stmt_alias_decl(stmt, out),
            _ => {}
        }
    }
}

fn collect_stmt_alias_decls<'a>(stmts: &'a [Stmt], out: &mut Vec<&'a AliasDecl>) {
    for stmt in stmts { collect_stmt_alias_decl(stmt, out); }
}

fn collect_stmt_alias_decl<'a>(stmt: &'a Stmt, out: &mut Vec<&'a AliasDecl>) {
    match stmt {
        Stmt::Alias(alias) => out.push(alias),
        Stmt::Fn(f) => collect_stmt_alias_decls(&f.body, out),
        Stmt::Mod(m) => collect_alias_decls(&m.items, out),
        _ => {}
    }
}

impl Signatures {
    /// Parses `path` and records its (filtered) signatures, then recurses into
    /// its own `use` items relative to `dir`.
    fn import_file(
        &mut self,
        path: &std::path::Path,
        pub_only: bool,
        only: &[String],
        deps: &HashMap<String, std::path::PathBuf>,
        visited: &mut HashSet<(std::path::PathBuf, bool)>,
    ) {
        let Ok(path) = path.canonicalize() else { return };
        if !visited.insert((path.clone(), pub_only)) { return; }
        let Ok(src) = std::fs::read_to_string(&path) else { return };
        // A bare `use` inside the imported file resolves against that file's own directory.
        let Some(file_dir) = path.parent() else { return };
        self.import_source(&src, file_dir, pub_only, only, deps, visited);
    }

    fn import_source(
        &mut self,
        src: &str,
        dir: &std::path::Path,
        pub_only: bool,
        only: &[String],
        deps: &HashMap<String, std::path::PathBuf>,
        visited: &mut HashSet<(std::path::PathBuf, bool)>,
    ) {
        let Ok(tokens) = crate::lexer::lex_all(src) else { return };
        let Ok(file) = crate::parser::parse(tokens) else { return };
        self.collect_items_filtered(&file.items, &|name, is_pub| {
            (!pub_only || is_pub) && (only.is_empty() || only.iter().any(|n| n == name))
        });
        self.collect_imports(&file.items, dir, deps, visited);
    }

    fn collect_imports(
        &mut self,
        items: &[Item],
        dir: &std::path::Path,
        deps: &HashMap<String, std::path::PathBuf>,
        visited: &mut HashSet<(std::path::PathBuf, bool)>,
    ) {
        for item in items {
            let Item::Use(u) = item else {
                if let Item::Mod(m) = item { self.collect_imports(&m.items, dir, deps, visited); }
                continue;
            };
            let Some(first) = u.path.first() else { continue };
            // `use boring.<module>` — embedded first-party stdlib.
            if first == "boring" {
                let module = u.path.get(1).map(String::as_str).unwrap_or("");
                if let Some(src) = crate::stdlib_embed::lookup(module) {
                    let key = (crate::stdlib_embed::synthetic_path(module), true);
                    if visited.insert(key) {
                        self.import_source(src, dir, true, &u.items, deps, visited);
                    }
                }
                continue;
            }
            // `use <dep>.<mod>.…` — a `[deps]` project; its own bare `use`s resolve
            // against that dependency's root.
            if let Some(dep_root) = deps.get(first.as_str()) {
                let rel: std::path::PathBuf = u.path[1..].iter().collect();
                self.import_file(&dep_root.join(rel).with_extension("br"), true, &u.items, deps, visited);
                continue;
            }
            let rel: std::path::PathBuf = u.path.iter().collect();
            let bare = u.path.len() == 1;
            self.import_file(&dir.join(rel).with_extension("br"), !bare, &u.items, deps, visited);
        }
    }
}

fn desugar_items(items: Vec<Item>, sigs: &Signatures) -> Result<Vec<Item>, ParseError> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(desugar_item(item, sigs)?);
    }
    Ok(out)
}

fn desugar_item(item: Item, sigs: &Signatures) -> Result<Item, ParseError> {
    Ok(match item {
        Item::Fn(mut f) => {
            let saved = sigs.enter_body(&f.params, None);
            let r = desugar_body(std::mem::take(&mut f.body), sigs);
            sigs.leave_body(saved);
            f.body = r?;
            Item::Fn(f)
        }
        Item::Struct(mut s) => {
            let owner = s.clone();
            for m in s.methods.iter_mut() {
                let saved = sigs.enter_body(&m.params, Some(&owner));
                let r = desugar_body(std::mem::take(&mut m.body), sigs);
                sigs.leave_body(saved);
                m.body = r?;
            }
            for i in s.inits.iter_mut() {
                let body = std::mem::take(&mut i.body);
                i.body = desugar_body(body, sigs)?;
            }
            for tm in s.type_methods.iter_mut() {
                let body = std::mem::take(&mut tm.body);
                tm.body = desugar_body(body, sigs)?;
            }
            Item::Struct(s)
        }
        Item::Enum(mut e) => {
            for m in e.methods.iter_mut() {
                let body = std::mem::take(&mut m.body);
                m.body = desugar_body(body, sigs)?;
            }
            Item::Enum(e)
        }
        Item::Ext(mut ext) => {
            for m in ext.methods.iter_mut() {
                let body = std::mem::take(&mut m.body);
                m.body = desugar_body(body, sigs)?;
            }
            Item::Ext(ext)
        }
        Item::Mod(mut m) => { m.items = desugar_items(m.items, sigs)?; Item::Mod(m) }
        Item::Let(mut s) => {
            if let Some(v) = s.value.take() { s.value = Some(desugar_expr(v, sigs)?); }
            Item::Let(s)
        }
        Item::Stmt(stmt) => Item::Stmt(desugar_stmt(stmt, sigs)?),
        other => other,
    })
}

fn desugar_body(stmts: Vec<Stmt>, sigs: &Signatures) -> Result<Vec<Stmt>, ParseError> {
    stmts.into_iter().map(|s| desugar_stmt(s, sigs)).collect()
}

fn desugar_arg(a: Arg, sigs: &Signatures) -> Result<Arg, ParseError> {
    Ok(Arg { label: a.label, value: desugar_expr(a.value, sigs)?, spread: a.spread, default_rest: a.default_rest })
}

fn desugar_args(args: Vec<Arg>, sigs: &Signatures) -> Result<Vec<Arg>, ParseError> {
    args.into_iter().map(|a| desugar_arg(a, sigs)).collect()
}

fn desugar_stmt(stmt: Stmt, sigs: &Signatures) -> Result<Stmt, ParseError> {
    Ok(match stmt {
        Stmt::Let(mut s) => {
            if let Some(v) = s.value.take() { s.value = Some(desugar_expr(v, sigs)?); }
            if let Some(t) = &s.ty { sigs.locals.borrow_mut().insert(s.name.clone(), t.clone()); }
            Stmt::Let(s)
        }
        Stmt::LetDestructure(mut s) => { s.value = desugar_expr(s.value, sigs)?; Stmt::LetDestructure(s) }
        Stmt::Return(mut r) => { r.value = r.value.map(|v| desugar_expr(v, sigs)).transpose()?; Stmt::Return(r) }
        Stmt::Break(line, val) => Stmt::Break(line, val.map(|v| desugar_expr(v, sigs)).transpose()?),
        Stmt::Throw(mut t) => { t.value = t.value.map(|v| desugar_expr(v, sigs)).transpose()?; Stmt::Throw(t) }
        Stmt::If(mut i) => {
            let mut branches = Vec::with_capacity(i.branches.len());
            for (c, b) in i.branches { branches.push((desugar_expr(c, sigs)?, desugar_body(b, sigs)?)); }
            i.branches = branches;
            i.else_body = i.else_body.map(|b| desugar_body(b, sigs)).transpose()?;
            Stmt::If(i)
        }
        Stmt::IfLet(mut i) => {
            i.then_body = desugar_body(i.then_body, sigs)?;
            let mut elifs = Vec::with_capacity(i.elif_branches.len());
            for mut b in i.elif_branches { b.body = desugar_body(b.body, sigs)?; elifs.push(b); }
            i.elif_branches = elifs;
            i.else_body = i.else_body.map(|b| desugar_body(b, sigs)).transpose()?;
            Stmt::IfLet(i)
        }
        Stmt::Match(mut m) => {
            m.subject = desugar_expr(m.subject, sigs)?;
            let mut arms = Vec::with_capacity(m.arms.len());
            for mut arm in m.arms {
                arm.guard = arm.guard.map(|g| desugar_expr(g, sigs)).transpose()?;
                arm.body = match arm.body {
                    MatchBody::Expr(e) => MatchBody::Expr(desugar_expr(e, sigs)?),
                    MatchBody::Block(b) => MatchBody::Block(desugar_body(b, sigs)?),
                };
                arms.push(arm);
            }
            m.arms = arms;
            Stmt::Match(m)
        }
        Stmt::While(mut w) => { w.condition = desugar_expr(w.condition, sigs)?; w.body = desugar_body(w.body, sigs)?; Stmt::While(w) }
        Stmt::WhileLet(mut w) => { w.body = desugar_body(w.body, sigs)?; Stmt::WhileLet(w) }
        Stmt::DoWhile(mut d) => { d.body = desugar_body(d.body, sigs)?; d.condition = desugar_expr(d.condition, sigs)?; Stmt::DoWhile(d) }
        Stmt::Loop(mut l) => { l.body = desugar_body(l.body, sigs)?; Stmt::Loop(l) }
        Stmt::Wait(e, line) => Stmt::Wait(desugar_expr(e, sigs)?, line),
        Stmt::For(mut f) => { f.iterable = desugar_expr(f.iterable, sigs)?; f.body = desugar_body(f.body, sigs)?; Stmt::For(f) }
        Stmt::Guard(mut g) => { g.else_body = desugar_body(g.else_body, sigs)?; Stmt::Guard(g) }
        Stmt::Try(mut t) => {
            t.body = desugar_body(t.body, sigs)?;
            let mut clauses = Vec::with_capacity(t.catch_clauses.len());
            for mut c in t.catch_clauses { c.body = desugar_body(c.body, sigs)?; clauses.push(c); }
            t.catch_clauses = clauses;
            Stmt::Try(t)
        }
        Stmt::Defer(body) => Stmt::Defer(desugar_body(body, sigs)?),
        Stmt::Expr(e) => Stmt::Expr(desugar_expr(e, sigs)?),
        Stmt::Fn(mut f) => { f.body = desugar_body(f.body, sigs)?; Stmt::Fn(f) }
        Stmt::Struct(mut s) => {
            for m in s.methods.iter_mut() {
                let body = std::mem::take(&mut m.body);
                m.body = desugar_body(body, sigs)?;
            }
            Stmt::Struct(s)
        }
        Stmt::Mod(mut m) => { m.items = desugar_items(m.items, sigs)?; Stmt::Mod(m) }
        Stmt::With(mut w) => { w.body = desugar_body(w.body, sigs)?; Stmt::With(w) }
        Stmt::KernelBlock(mut k) => { k.body = desugar_body(k.body, sigs)?; Stmt::KernelBlock(k) }
        Stmt::Yield(e, line) => Stmt::Yield(desugar_expr(e, sigs)?, line),
        other => other,
    })
}

fn desugar_expr(e: Expr, sigs: &Signatures) -> Result<Expr, ParseError> {
    let Expr { kind, line, col, len } = e;
    let kind = match kind {
        ExprKind::TrailingArrayBlock { callee, args, has_parens, body } => {
            return desugar_trailing_array_block(*callee, args, has_parens, body, sigs, line, col, len);
        }
        ExprKind::BinOp(op, l, r) => ExprKind::BinOp(op, Box::new(desugar_expr(*l, sigs)?), Box::new(desugar_expr(*r, sigs)?)),
        ExprKind::UnaryOp(op, v) => ExprKind::UnaryOp(op, Box::new(desugar_expr(*v, sigs)?)),
        ExprKind::Assign(l, r) => ExprKind::Assign(Box::new(desugar_expr(*l, sigs)?), Box::new(desugar_expr(*r, sigs)?)),
        ExprKind::QuestionAssign(l, r) => ExprKind::QuestionAssign(Box::new(desugar_expr(*l, sigs)?), Box::new(desugar_expr(*r, sigs)?)),
        ExprKind::Field(o, name) => ExprKind::Field(Box::new(desugar_expr(*o, sigs)?), name),
        ExprKind::OptionalField(o, name) => ExprKind::OptionalField(Box::new(desugar_expr(*o, sigs)?), name),
        ExprKind::Index(a, i) => ExprKind::Index(Box::new(desugar_expr(*a, sigs)?), Box::new(desugar_expr(*i, sigs)?)),
        ExprKind::LabeledIndex(o, args) => ExprKind::LabeledIndex(Box::new(desugar_expr(*o, sigs)?), desugar_args(args, sigs)?),
        ExprKind::Call(callee, args) => ExprKind::Call(Box::new(desugar_expr(*callee, sigs)?), desugar_args(args, sigs)?),
        ExprKind::MethodCall(obj, m, args) => ExprKind::MethodCall(Box::new(desugar_expr(*obj, sigs)?), m, desugar_args(args, sigs)?),
        ExprKind::OptionalMethodCall(obj, m, args) => ExprKind::OptionalMethodCall(Box::new(desugar_expr(*obj, sigs)?), m, desugar_args(args, sigs)?),
        ExprKind::GenericCall(callee, tys, args) => ExprKind::GenericCall(Box::new(desugar_expr(*callee, sigs)?), tys, desugar_args(args, sigs)?),
        ExprKind::Pipe(l, name, args) => ExprKind::Pipe(Box::new(desugar_expr(*l, sigs)?), name, desugar_args(args, sigs)?),
        ExprKind::New { arena, ctor } => ExprKind::New {
            arena: arena.map(|a| desugar_expr(*a, sigs)).transpose()?.map(Box::new),
            ctor: Box::new(desugar_expr(*ctor, sigs)?),
        },
        ExprKind::KernelLaunch { mut config, kernel } => {
            config.block = config.block.take().map(|e| desugar_expr(e, sigs)).transpose()?;
            config.grid = config.grid.take().map(|e| desugar_expr(e, sigs)).transpose()?;
            config.after = config.after.take().map(|e| desugar_expr(e, sigs)).transpose()?;
            ExprKind::KernelLaunch { config, kernel: Box::new(desugar_expr(*kernel, sigs)?) }
        }
        ExprKind::TryElse(a, b) => ExprKind::TryElse(Box::new(desugar_expr(*a, sigs)?), Box::new(desugar_expr(*b, sigs)?)),
        ExprKind::TryElseBlock(body, els) => ExprKind::TryElseBlock(desugar_body(body, sigs)?, desugar_body(els, sigs)?),
        ExprKind::Array(elems) => ExprKind::Array(elems.into_iter().map(|x| desugar_expr(x, sigs)).collect::<Result<_, _>>()?),
        ExprKind::ArrayFill { value, count } => ExprKind::ArrayFill {
            value: Box::new(desugar_expr(*value, sigs)?), count: Box::new(desugar_expr(*count, sigs)?),
        },
        ExprKind::ArrayAlloc { count } => ExprKind::ArrayAlloc { count: Box::new(desugar_expr(*count, sigs)?) },
        ExprKind::ArrayComp { expr, var, count } => ExprKind::ArrayComp {
            expr: Box::new(desugar_expr(*expr, sigs)?), var, count: Box::new(desugar_expr(*count, sigs)?),
        },
        ExprKind::ArrayCompIter { expr, var, iter } => ExprKind::ArrayCompIter {
            expr: Box::new(desugar_expr(*expr, sigs)?), var, iter: Box::new(desugar_expr(*iter, sigs)?),
        },
        ExprKind::LabeledArrayComp { expr, clauses } => {
            let mut new_clauses = Vec::with_capacity(clauses.len());
            for (label, count) in clauses { new_clauses.push((label, Box::new(desugar_expr(*count, sigs)?))); }
            ExprKind::LabeledArrayComp { expr: Box::new(desugar_expr(*expr, sigs)?), clauses: new_clauses }
        }
        ExprKind::Tuple(xs) => ExprKind::Tuple(xs.into_iter().map(|x| desugar_expr(x, sigs)).collect::<Result<_, _>>()?),
        ExprKind::Dict(pairs) => {
            let mut out = Vec::with_capacity(pairs.len());
            for (k, v) in pairs { out.push((desugar_expr(k, sigs)?, desugar_expr(v, sigs)?)); }
            ExprKind::Dict(out)
        }
        ExprKind::Set(elems) => ExprKind::Set(elems.into_iter().map(|x| desugar_expr(x, sigs)).collect::<Result<_, _>>()?),
        ExprKind::Range { start, end, inclusive } => ExprKind::Range {
            start: Box::new(desugar_expr(*start, sigs)?), end: Box::new(desugar_expr(*end, sigs)?), inclusive,
        },
        ExprKind::SliceRange { start, end, inclusive } => ExprKind::SliceRange {
            start: start.map(|s| desugar_expr(*s, sigs)).transpose()?.map(Box::new),
            end: end.map(|e| desugar_expr(*e, sigs)).transpose()?.map(Box::new),
            inclusive,
        },
        ExprKind::Cast(inner, ty) => ExprKind::Cast(Box::new(desugar_expr(*inner, sigs)?), ty),
        ExprKind::RelabelCast(inner, pairs) => ExprKind::RelabelCast(Box::new(desugar_expr(*inner, sigs)?), pairs),
        ExprKind::Else(a, b) => ExprKind::Else(Box::new(desugar_expr(*a, sigs)?), Box::new(desugar_expr(*b, sigs)?)),
        ExprKind::Closure(params, ret, body, throws, task) => {
            let body = match body {
                ClosureBody::Expr(e) => ClosureBody::Expr(Box::new(desugar_expr(*e, sigs)?)),
                ClosureBody::Block(b) => ClosureBody::Block(desugar_body(b, sigs)?),
            };
            ExprKind::Closure(params, ret, body, throws, task)
        }
        ExprKind::If(mut i) => {
            let mut branches = Vec::with_capacity(i.branches.len());
            for (c, b) in i.branches { branches.push((desugar_expr(c, sigs)?, desugar_body(b, sigs)?)); }
            i.branches = branches;
            i.else_body = i.else_body.map(|b| desugar_body(b, sigs)).transpose()?;
            ExprKind::If(i)
        }
        ExprKind::Match(mut m) => {
            m.subject = desugar_expr(m.subject, sigs)?;
            let mut arms = Vec::with_capacity(m.arms.len());
            for mut arm in m.arms {
                arm.guard = arm.guard.map(|g| desugar_expr(g, sigs)).transpose()?;
                arm.body = match arm.body {
                    MatchBody::Expr(e) => MatchBody::Expr(desugar_expr(e, sigs)?),
                    MatchBody::Block(b) => MatchBody::Block(desugar_body(b, sigs)?),
                };
                arms.push(arm);
            }
            m.arms = arms;
            ExprKind::Match(m)
        }
        ExprKind::Block(stmts) => ExprKind::Block(desugar_body(stmts, sigs)?),
        ExprKind::Do(stmts) => ExprKind::Do(desugar_body(stmts, sigs)?),
        ExprKind::Loop(mut l) => { l.body = desugar_body(l.body, sigs)?; ExprKind::Loop(l) }
        ExprKind::Task(e) => ExprKind::Task(Box::new(desugar_expr(*e, sigs)?)),
        ExprKind::TaskWithTimeout(a, b) => ExprKind::TaskWithTimeout(Box::new(desugar_expr(*a, sigs)?), Box::new(desugar_expr(*b, sigs)?)),
        ExprKind::JoinAll(exprs) => ExprKind::JoinAll(exprs.into_iter().map(|e| desugar_expr(e, sigs)).collect::<Result<_, _>>()?),
        ExprKind::MacroCall { name, args } => ExprKind::MacroCall {
            name, args: args.into_iter().map(|e| desugar_expr(e, sigs)).collect::<Result<_, _>>()?,
        },
        ExprKind::StringInterp(segs) => ExprKind::StringInterp(segs.into_iter().map(|seg| Ok(match seg {
            StringSegment::Expr(e) => StringSegment::Expr(Box::new(desugar_expr(*e, sigs)?)),
            StringSegment::FormattedExpr(e, f) => StringSegment::FormattedExpr(Box::new(desugar_expr(*e, sigs)?), f),
            other @ StringSegment::Lit(_) => other,
        })).collect::<Result<_, ParseError>>()?),

        // Leaves: Int/Float/Str/Bool/Nil/Void/Var/DotIdent/UInt64 — nothing to recurse into.
        other => other,
    };
    Ok(Expr { kind, line, col, len })
}

/// Resolves one `TrailingArrayBlock` node per the module doc comment's
/// decision table, and rewrites it away accordingly.
#[allow(clippy::too_many_arguments)]
fn desugar_trailing_array_block(
    callee: Expr,
    args: Vec<Arg>,
    has_parens: bool,
    body: Vec<Stmt>,
    sigs: &Signatures,
    line: usize,
    col: usize,
    len: usize,
) -> Result<Expr, ParseError> {
    // Desugar sub-expressions/nested block-array calls bottom-up first, so a
    // nested `TrailingArrayBlock` (e.g. `Row:` inside `Column:`, reached as an
    // ordinary `Stmt::Expr` inside `body`) is fully resolved and rewritten
    // before this node's own resolution runs.
    let callee = desugar_expr(callee, sigs)?;
    let args = desugar_args(args, sigs)?;
    let body = desugar_body(body, sigs)?;

    // The parser only ever builds this node with a plain `Var` callee (see
    // `Parser::parse_array_block_tail`'s call sites) — this is defense in
    // depth, not a reachable path today.
    let ExprKind::Var(name) = &callee.kind else {
        return Err(ParseError::Generic {
            line, col, len,
            msg: "trailing array-block sugar requires a plain function or type name before \
                  the colon (e.g. `Column:`), not a more complex expression".into(),
        });
    };

    enum Interp { Collect(String), Tail, ClosureLiteral }

    let interp = match sigs.last_param_type(name) {
        Some(Type::Array(inner)) => match inner.as_ref() {
            Type::Named(n) if sigs.is_trait(n) => Interp::Collect(n.clone()),
            other => {
                return Err(ParseError::Generic {
                    line, col, len,
                    msg: format!(
                        "trailing array-block sugar: `{}`'s last parameter is `[{}]`, not a \
                         trait-object array `[dyn Trait]` — this sugar only applies when the \
                         last parameter's element type is a declared `trait`",
                        name, describe_type(other)
                    ),
                });
            }
        },
        Some(Type::Fn(..)) => Interp::Tail,
        Some(other) => {
            return Err(ParseError::Generic {
                line, col, len,
                msg: format!(
                    "trailing array-block sugar: `{}`'s last parameter is `{}`, not a \
                     trait-object array `[dyn Trait]` — this sugar only applies when the last \
                     parameter is an array of a declared `trait`",
                    name, describe_type(other)
                ),
            });
        }
        // Not a known callable in this file's own signature table (see this
        // module's "Known limitation" doc comment) — fall back to whatever
        // this exact token shape meant before this sugar existed: an
        // ordinary zero-arg trailing closure when there's an explicit
        // (possibly empty) argument list, or the no-paren closure-literal
        // shorthand when there's no parentheses at all.
        None => if has_parens { Interp::Tail } else { Interp::ClosureLiteral },
    };

    match interp {
        Interp::Collect(trait_name) => {
            let mut elems = stmts_to_array_block_elems(body, line, col, len)?;
            resolve_identity_keys(&mut elems, sigs)?;
            if !elems_have_control_flow(&elems) {
                // Flat case: plain array-literal trailing argument.
                let mut flat = Vec::with_capacity(elems.len());
                for elem in elems {
                    match elem {
                        ArrayBlockElem::Item(e) => flat.push(e),
                        // Unreachable: `elems_have_control_flow` already returned false.
                        ArrayBlockElem::If { .. } | ArrayBlockElem::For { .. } => unreachable!(),
                    }
                }
                let mut new_args = args;
                new_args.push(Arg {
                    label: trailing_label(sigs, name, &new_args),
                    value: Expr { kind: ExprKind::Array(flat), line, col, len },
                    spread: false, default_rest: false,
                });
                return Ok(Expr { kind: ExprKind::Call(Box::new(callee), new_args), line, col, len });
            }

            // Control-flow case: imperative `Vec` builder.
            const CHILDREN_VAR: &str = "__children";
            let mut stmts = Vec::new();
            stmts.push(Stmt::Let(LetStmt {
                binding: BindingKind::Mut,
                is_pub: false,
                is_static: false,
                name: CHILDREN_VAR.to_string(),
                // `Type::Mut` wrapper, exactly like a parser-built `mut` local.
                ty: Some(Type::Mut(Box::new(Type::Array(Box::new(Type::Named(trait_name.clone())))))),
                var_mut: false,
                value: Some(Expr { kind: ExprKind::Array(vec![]), line, col, len: 0 }),
                is_lazy: false,
                line, col,
            }));
            stmts.extend(elems_to_push_stmts(&elems, CHILDREN_VAR, &trait_name, line, col));
            let mut new_args = args;
            new_args.push(Arg {
                label: trailing_label(sigs, name, &new_args),
                value: Expr { kind: ExprKind::Var(CHILDREN_VAR.to_string()), line, col, len: 0 },
                spread: false, default_rest: false,
            });
            stmts.push(Stmt::Expr(Expr { kind: ExprKind::Call(Box::new(callee), new_args), line, col, len }));
            // `Do`, not `Block`: both are documented as "evaluates stmts, last
            // expression is the value", but `ExprKind::Block`'s transpiler emission
            // (`emit_stmt_inline`) only actually handles `Let`/`Return`/`If`/`Expr`
            // statements — a `for` (as this control-flow case needs) silently
            // degrades to a useless `/* complex stmt */` comment instead of real
            // code (a pre-existing, separately filed transpiler gap). `Do`'s own
            // emission path (`emit_body` via a sub-emitter) already handles every
            // statement kind correctly, so it's used here as a working, semantically
            // equivalent substitute rather than fixing `Block` itself.
            Ok(Expr { kind: ExprKind::Do(stmts), line, col, len })
        }
        Interp::Tail => {
            let (throws, task) = infer_closure_throws_task(&ClosureBody::Block(body.clone()));
            let closure = Expr {
                kind: ExprKind::Closure(vec![], None, ClosureBody::Block(body), throws, task),
                line, col, len,
            };
            let mut new_args = args;
            new_args.push(Arg { label: None, value: closure, spread: false, default_rest: false });
            Ok(Expr { kind: ExprKind::Call(Box::new(callee), new_args), line, col, len })
        }
        Interp::ClosureLiteral => {
            let param = Param {
                name: name.clone(), ty: None, mutable: false, rebindable: false, var_mut: false, owned: false,
                variadic: false, default: None, line, col,
            };
            let (throws, task) = infer_closure_throws_task(&ClosureBody::Block(body.clone()));
            Ok(Expr {
                kind: ExprKind::Closure(vec![param], None, ClosureBody::Block(body), throws, task),
                line, col, len,
            })
        }
    }
}

/// Label for the synthesized trailing array argument: the last field's name when `name` is a
/// struct constructor and the call already has labeled arguments, else positional.
fn trailing_label(sigs: &Signatures, name: &str, args: &[Arg]) -> Option<String> {
    if args.iter().any(|a| a.label.is_some()) && !sigs.fns.contains_key(name) {
        sigs.last_field.get(name).cloned()
    } else {
        None
    }
}

/// Builds the imperative push-statement list for the control-flow collect
/// case: `ArrayBlockElem::Item(e)` -> `__children.push(e)`; `If`/`For` -> the
/// matching `Stmt::If`/`Stmt::For`, recursing into their own bodies.
fn elems_to_push_stmts(elems: &[ArrayBlockElem], children_var: &str, trait_name: &str, line: usize, col: usize) -> Vec<Stmt> {
    elems.iter().map(|elem| match elem {
        ArrayBlockElem::Item(e) => {
            let push_arg = Arg { label: None, value: e.clone(), spread: false, default_rest: false };
            let recv = Expr { kind: ExprKind::Var(children_var.to_string()), line, col, len: 0 };
            Stmt::Expr(Expr {
                kind: ExprKind::MethodCall(Box::new(recv), "push".to_string(), vec![push_arg]),
                line, col, len: 0,
            })
        }
        ArrayBlockElem::If { branches, else_body } => {
            let branches = branches.iter()
                .map(|(c, b)| (c.clone(), elems_to_push_stmts(b, children_var, trait_name, line, col)))
                .collect();
            let else_body = else_body.as_ref().map(|b| elems_to_push_stmts(b, children_var, trait_name, line, col));
            Stmt::If(IfStmt { branches, else_body, line, col })
        }
        ArrayBlockElem::For { vars, iterable, key: None, body } => {
            Stmt::For(ForStmt {
                vars: vars.clone(),
                iterable: iterable.clone(),
                key: None,
                body: elems_to_push_stmts(body, children_var, trait_name, line, col),
                line, col,
            })
        }
        // Keyed iteration: the iteration's items are gathered into their own array and
        // pushed as ONE `KeyedGroup(key, items)` element, so the key identifies the whole
        // iteration subtree (docs/design-notes/boring-ui-draft.md, "Identity & lifecycle").
        // `KeyedGroup` is an ordinary library symbol the caller must have in scope; the
        // compiler assigns it no meaning beyond this call shape.
        ArrayBlockElem::For { vars, iterable, key: Some(key), body } => {
            let iter_var = format!("{children_var}_it");
            let var = |name: &str| Expr { kind: ExprKind::Var(name.to_string()), line, col, len: 0 };
            let mut inner = vec![Stmt::Let(LetStmt {
                binding: BindingKind::Mut,
                is_pub: false,
                is_static: false,
                name: iter_var.clone(),
                ty: Some(Type::Mut(Box::new(Type::Array(Box::new(Type::Named(trait_name.to_string())))))),
                var_mut: false,
                value: Some(Expr { kind: ExprKind::Array(vec![]), line, col, len: 0 }),
                is_lazy: false,
                line, col,
            })];
            inner.extend(elems_to_push_stmts(body, &iter_var, trait_name, line, col));
            let group = Expr {
                kind: ExprKind::Call(Box::new(var("KeyedGroup")), vec![
                    Arg { label: None, value: key.clone(), spread: false, default_rest: false },
                    Arg { label: None, value: var(&iter_var), spread: false, default_rest: false },
                ]),
                line, col, len: 0,
            };
            inner.push(Stmt::Expr(Expr {
                kind: ExprKind::MethodCall(Box::new(var(children_var)), "push".to_string(),
                    vec![Arg { label: None, value: group, spread: false, default_rest: false }]),
                line, col, len: 0,
            }));
            Stmt::For(ForStmt { vars: vars.clone(), iterable: iterable.clone(), key: None, body: inner, line, col })
        }
    }).collect()
}
