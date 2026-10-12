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

//! Lowers `view` declarations to ordinary structs (docs/design-notes/boring-ui-draft.md, "State
//! slots and binding handles"). Runs once on the merged `Program`, after `view_perm`, so the
//! checker, interpreter and transpilers only ever see plain structs.
//!
//! ```text
//! view Badge:                        struct Badge__State:           # mounted-state layout
//!     string title                       var int count = 0          # `state` fields, declaration order
//!     state int count = 0        ==>
//!                                    struct Badge:                  # ephemeral description
//!     def bump():                        string title               # inputs only
//!         count += 1                     def bump(mut Badge__State __st):
//!     body():                                __st.count += 1
//!         Text("{title}{count}")         body(mut Badge__State __st):
//!                                            Text("{title}{__st.count}")
//! ```
//!
//! The description (`Badge`) holds inputs only; the persistent values live in `Badge__State`,
//! one instance per mounted node, seeded once at mount by `Badge__State()` (the declaration's
//! default seed). Every method and `body()` takes the mounted state as an explicit trailing
//! `__st` parameter; bare reads/writes of a `state` field and `self.field` are rewritten to
//! `__st.field`, every method becomes `req` (the description itself is never written), and `self.method(...)` calls forward `__st`. Callback closures capture `__st`
//! like any other local, which is the "callback `self` as mounted-view context" of the draft.
//!
//! ## `bind` and `env`
//!
//! - **`bind T name`** becomes an input field `name: Slot<T>'actor` (`Slot` is `boring.ui`'s shared
//!   cell, so `use boring.ui` is required). Bare reads/writes of `name` in the view become
//!   `name.value`. At a call site `Child(name = x)` whose callee field is a `bind`, the argument
//!   must be a `state` or `bind` of the enclosing view; the *slot* is passed instead of the value.
//!   A `state` that is bound somewhere is itself stored as `Slot<T>'actor` in `V__State` (so a
//!   child's write is visible to the parent), and the parent's own accesses go through `.value`.
//!   Child callee info comes from the same program and from embedded `boring.*` modules named by
//!   a `use`; sibling-file views are not visible to the call-site rewrite yet.
//! - **`env T name`** becomes a read-only field of a generated `V__Env` struct, passed as a second
//!   trailing parameter `__env` (after `__st`); the runtime fills it from the nearest providers.
//!
//! ## Slice scope (what is NOT lowered yet)
//!
//! - Native views (`body(): native`, no `state`) lower to a plain struct conforming to `View`.
//! - **`bind` projections** (`profile.name`, keyed elements), binding a `state let`, and
//!   `bind`'s permission levels (the child can always replace and mutate: the slot is `var mut`).
//! - `boring build`: the interpreter runs a lowered `bind` end to end, but the transpiler cannot
//!   yet emit field access through `'actor` (see "Known gaps" in boring-ui-progress.md).
//! - **Parent-supplied seeds** (`Counter(count = expensive())`): a `state` field is no longer a
//!   constructor argument, so that is an ordinary "unknown field" error today.
//! - **Invalidation**: there is no write-site invalidation call. The runtime (layer 2) treats any
//!   callback that received `__st` as having possibly written it, which is conservative but
//!   correct; precise per-slot invalidation comes with the runtime's slot API.
//! - A `state` field needs an initializer (there is no default seed otherwise), and a local,
//!   parameter or loop variable may not shadow a `state` field name inside the view.

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::ast_walk::Walker;
use crate::parser::ParseError;

const ST: &str = "__st";
const ENV: &str = "__env";

fn err(msg: String, line: usize, col: usize) -> ParseError {
    ParseError::Generic { msg, line, col, len: 1 }
}

/// A `[mut] <Ty> <name>` parameter, built by parsing a one-line signature so its exact
/// permission encoding always matches what the parser produces for source-written code.
fn context_param(modifier: &str, ty: &str, name: &str, line: usize, col: usize) -> Param {
    let src = format!("def f({modifier}{ty} {name}):\n    pass\n");
    let tokens = crate::lexer::lex(&src).expect("generated signature lexes");
    let program = crate::parser::parse(tokens).expect("generated signature parses");
    for item in program.items {
        if let Item::Fn(mut f) = item {
            let mut p = f.params.remove(0);
            p.line = line;
            p.col = col;
            return p;
        }
    }
    unreachable!("generated signature has one fn")
}

/// Input fields of a view that has at least one `bind`, in declaration order: `(name, is_bind)`.
type BindViews = HashMap<String, Vec<(String, bool)>>;

fn collect_bind_views(items: &[Item], out: &mut BindViews) {
    for item in items {
        match item {
            Item::Struct(s) if s.is_view => {
                let inputs: Vec<(String, bool)> = s.fields.iter()
                    .filter_map(|f| match f.view_kind {
                        None => Some((f.name.clone(), false)),
                        Some(ViewFieldKind::Bind) => Some((f.name.clone(), true)),
                        _ => None,
                    })
                    .collect();
                if inputs.iter().any(|(_, b)| *b) { out.insert(s.name.clone(), inputs); }
            }
            Item::Mod(m) => collect_bind_views(&m.items, out),
            // `use boring.<module>`: the embedded stdlib is not merged into the program, so read
            // its view signatures directly (names are exported unqualified by `use`).
            Item::Use(u) if u.path.first().map(String::as_str) == Some("boring") => {
                let module = u.path.get(1).map(String::as_str).unwrap_or("");
                if let Some(src) = crate::stdlib_embed::lookup(module) {
                    if let Ok(tokens) = crate::lexer::lex(src) {
                        if let Ok(p) = crate::parser::parse(tokens) {
                            collect_bind_views(&p.items, out);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn declares(items: &[Item], is_it: &dyn Fn(&Item) -> bool) -> bool {
    items.iter().any(|i| match i {
        Item::Mod(m) => declares(&m.items, is_it),
        other => is_it(other),
    })
}

struct Env {
    view_trait: bool,
    has_slot: bool,
    bind_views: BindViews,
}

fn lower_items(items: Vec<Item>, env: &Env) -> Result<Vec<Item>, ParseError> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Item::Struct(s) if s.is_view => {
                let lowered = lower_view(s, env)?;
                out.extend(lowered.into_iter().map(Item::Struct));
            }
            Item::Mod(mut m) => { m.items = lower_items(m.items, env)?; out.push(Item::Mod(m)); }
            other => out.push(other),
        }
    }
    Ok(out)
}

fn slot_type(inner: &Type) -> Type {
    let inner = match inner { Type::Mut(t) => (**t).clone(), t => t.clone() };
    Type::Mut(Box::new(Type::Qualified(
        Box::new(Type::Generic("Slot".to_string(), vec![inner])),
        OwnerQual::Actor,
    )))
}

fn slot_seed(default: &Expr) -> Expr {
    let (line, col, len) = (default.line, default.col, default.len);
    Expr {
        kind: ExprKind::Call(
            Box::new(Expr { kind: ExprKind::Var("Slot".to_string()), line, col, len }),
            vec![Arg { label: Some("value".to_string()), value: default.clone(), spread: false, default_rest: false }],
        ),
        line, col, len,
    }
}

fn plain_struct(name: String, is_pub: bool, fields: Vec<FieldDecl>, line: usize, col: usize) -> StructDecl {
    StructDecl {
        name, is_pub, is_view: false, is_native: false, protocols: vec![], fields,
        inits: vec![], methods: vec![], conversions: vec![], type_params: vec![],
        where_clause: vec![], setters: vec![], type_methods: vec![], type_vars: vec![],
        assoc_type_defs: vec![], attrs: vec![], line, col,
    }
}

/// Which `state` fields of `s` are handed to a child's `bind` (those must live in a `Slot`),
/// validating every bind argument on the way.
fn find_bound_states(
    s: &StructDecl,
    states: &HashSet<String>,
    binds: &HashSet<String>,
    bind_views: &BindViews,
) -> Result<HashSet<String>, ParseError> {
    let mut bound = HashSet::new();
    let mut failure: Option<ParseError> = None;
    let mut methods = s.methods.clone();
    for m in &mut methods {
        let mut on_expr = |e: &mut Expr| {
            let ExprKind::Call(callee, args) = &e.kind else { return };
            let ExprKind::Var(name) = &callee.kind else { return };
            let Some(inputs) = bind_views.get(name) else { return };
            for (i, a) in args.iter().enumerate() {
                let target = match &a.label {
                    Some(l) => inputs.iter().find(|(n, _)| n == l),
                    None => inputs.get(i),
                };
                let Some((field, true)) = target else { continue };
                match &a.value.kind {
                    ExprKind::Var(n) if states.contains(n) => { bound.insert(n.clone()); }
                    ExprKind::Var(n) if binds.contains(n) => {}
                    _ => {
                        failure.get_or_insert_with(|| err(
                            format!(
                                "`{name}`'s `bind {field}` needs a `state` or `bind` field of view '{}' as its argument \
                                 (a literal, calculated value, local or projection has no persistent storage)",
                                s.name
                            ),
                            a.value.line, a.value.col,
                        ));
                    }
                }
            }
        };
        let mut noop = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut noop }.stmts(&mut m.body);
    }
    match failure {
        Some(e) => Err(e),
        None => Ok(bound),
    }
}

fn lower_view(mut s: StructDecl, env: &Env) -> Result<Vec<StructDecl>, ParseError> {
    // A view automatically conforms to `View` — when the trait is in the program (`use boring.ui`).
    if env.view_trait && !s.protocols.iter().any(|p| p == "View") {
        s.protocols.push("View".to_string());
    }
    // A native view's adapter is supplied by the UI runtime: it keeps its inputs, loses the
    // placeholder `body(): native`, and may not own application `state`.
    if s.methods.iter().any(|m| m.is_native) {
        if let Some(f) = s.fields.iter().find(|f| f.view_kind == Some(ViewFieldKind::State)) {
            return Err(err(
                format!("native view '{}' cannot declare `state {}`: persistent implementation details \
                         belong to the native mounted component — compose a non-native view instead", s.name, f.name),
                f.line, f.col,
            ));
        }
        s.methods.retain(|m| !m.is_native);
    }

    let of_kind = |k: ViewFieldKind| -> Vec<FieldDecl> {
        s.fields.iter().filter(|f| f.view_kind == Some(k)).cloned().collect()
    };
    let state_fields = of_kind(ViewFieldKind::State);
    let bind_fields = of_kind(ViewFieldKind::Bind);
    let env_fields = of_kind(ViewFieldKind::Env);
    for f in &state_fields {
        if f.default.is_none() {
            return Err(err(
                format!("`state {}` needs an initializer: its seed is the only source of its first value", f.name),
                f.line, f.col,
            ));
        }
    }
    if !bind_fields.is_empty() && !env.has_slot {
        let f = &bind_fields[0];
        return Err(err(
            format!("`bind {}` needs the shared `Slot` cell: add `use boring.ui.*`", f.name),
            f.line, f.col,
        ));
    }
    let states: HashSet<String> = state_fields.iter().map(|f| f.name.clone()).collect();
    let binds: HashSet<String> = bind_fields.iter().map(|f| f.name.clone()).collect();
    let envs: HashSet<String> = env_fields.iter().map(|f| f.name.clone()).collect();
    let bound = find_bound_states(&s, &states, &binds, &env.bind_views)?;
    for f in state_fields.iter().filter(|f| bound.contains(&f.name) && !f.mutable && !f.ty.is_mut()) {
        return Err(err(
            format!("`state {}` is read-only (`state let`) but is bound to a child's `bind`", f.name),
            f.line, f.col,
        ));
    }

    let state_name = format!("{}__State", s.name);
    let env_name = format!("{}__Env", s.name);
    let mut generated = Vec::new();

    if !state_fields.is_empty() {
        let fields = state_fields.iter().map(|f| {
            let mut f = f.clone();
            f.view_kind = None;
            f.explicit_perm = false;
            f.is_pub = true;
            if bound.contains(&f.name) {
                let seed = slot_seed(f.default.as_ref().expect("checked above"));
                f.ty = slot_type(&f.ty);
                f.mutable = true;
                f.default = Some(seed);
            }
            f
        }).collect();
        generated.push(plain_struct(state_name.clone(), s.is_pub, fields, s.line, s.col));
    }
    if !env_fields.is_empty() {
        let fields = env_fields.iter().map(|f| {
            let mut f = f.clone();
            f.view_kind = None;
            f.explicit_perm = false;
            f.is_pub = true;
            f.mutable = false;
            f
        }).collect();
        generated.push(plain_struct(env_name.clone(), s.is_pub, fields, s.line, s.col));
    }

    // The description keeps inputs; a `bind` input becomes a shared slot handle.
    s.fields = s.fields.iter().filter_map(|f| match f.view_kind {
        None => Some(f.clone()),
        Some(ViewFieldKind::Bind) => {
            let mut f = f.clone();
            f.view_kind = None;
            f.explicit_perm = false;
            f.ty = slot_type(&f.ty);
            f.mutable = true;
            Some(f)
        }
        _ => None,
    }).collect();
    s.is_view = false;

    let has_state = !state_fields.is_empty();
    let has_env = !env_fields.is_empty();
    if has_state || has_env || !binds.is_empty() {
        let ctx = Ctx {
            states: &states, binds: &binds, envs: &envs, bound: &bound,
            own_methods: s.methods.iter().map(|m| m.name.clone()).collect(),
            bind_views: &env.bind_views,
            has_state, has_env,
        };
        let mut failure: Option<ParseError> = None;
        let view_name = s.name.clone();
        for m in &mut s.methods {
            for p in &m.params {
                if states.contains(&p.name) || binds.contains(&p.name) || envs.contains(&p.name) {
                    failure.get_or_insert_with(|| err(
                        format!("parameter `{}` shadows a `state`/`bind`/`env` field of view '{}'", p.name, view_name),
                        m.line, m.col,
                    ));
                }
            }
            if has_state { m.params.push(context_param("mut ", &state_name, ST, m.line, m.col)); }
            if has_env { m.params.push(context_param("", &env_name, ENV, m.line, m.col)); }
            // The description is never mutated (inputs are immutable, state lives in `__st`), so
            // even a method that writes state only needs `&self`.
            m.mutating = false;
            rewrite_body(&mut m.body, &ctx, &view_name, &mut failure);
        }
        if let Some(e) = failure { return Err(e); }
    }
    generated.push(s);
    Ok(generated)
}

struct Ctx<'a> {
    states: &'a HashSet<String>,
    binds: &'a HashSet<String>,
    envs: &'a HashSet<String>,
    bound: &'a HashSet<String>,
    own_methods: HashSet<String>,
    bind_views: &'a BindViews,
    has_state: bool,
    has_env: bool,
}

/// Marker prefix for names that pass 2 must not rewrite again; stripped by pass 3.
const KEEP: char = '@';

fn rewrite_body(body: &mut [Stmt], ctx: &Ctx, view: &str, failure: &mut Option<ParseError>) {
    let is_field = |n: &str| ctx.states.contains(n) || ctx.binds.contains(n) || ctx.envs.contains(n);

    // Pass 1 — on the untouched body: a binding that reuses a field name would be rewritten
    // by pass 2, silently changing what it refers to.
    let mut shadow: Vec<(String, usize, usize)> = Vec::new();
    {
        let shadow_cell = std::cell::RefCell::new(&mut shadow);
        let mut on_expr = |e: &mut Expr| match &e.kind {
            ExprKind::Closure(params, ..) => {
                for p in params.iter().filter(|p| is_field(&p.name)) {
                    shadow_cell.borrow_mut().push((p.name.clone(), p.line, p.col));
                }
            }
            ExprKind::ArrayComp { var, .. } | ExprKind::ArrayCompIter { var, .. } if is_field(var) => {
                shadow_cell.borrow_mut().push((var.clone(), e.line, e.col));
            }
            _ => {}
        };
        let mut on_stmt = |st: &mut Stmt| match st {
            Stmt::Let(l) if is_field(&l.name) => {
                shadow_cell.borrow_mut().push((l.name.clone(), l.line, l.col));
            }
            Stmt::LetDestructure(l) => {
                for b in l.bindings.iter().filter(|b| is_field(&b.name)) {
                    shadow_cell.borrow_mut().push((b.name.clone(), l.line, l.col));
                }
            }
            Stmt::For(f) => {
                for v in f.vars.iter().filter(|v| is_field(v)) {
                    shadow_cell.borrow_mut().push((v.clone(), f.line, f.col));
                }
            }
            _ => {}
        };
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
    }
    if let Some((n, line, col)) = shadow.into_iter().next() {
        failure.get_or_insert_with(|| err(
            format!("local `{n}` shadows a `state`/`bind`/`env` field `{n}` of view '{view}' — rename it"),
            line, col,
        ));
        return;
    }

    // Pass 2 — rewrite field accesses, hand slots to child `bind`s, forward the contexts.
    let var = |name: &str, line, col, len| Expr { kind: ExprKind::Var(name.to_string()), line, col, len };
    let field = |obj: Expr, name: &str| {
        let (line, col, len) = (obj.line, obj.col, obj.len);
        Expr { kind: ExprKind::Field(Box::new(obj), name.to_string()), line, col, len }
    };
    let mut on_expr = |e: &mut Expr| {
        let (line, col, len) = (e.line, e.col, e.len);
        let is_self = |o: &Expr| matches!(&o.kind, ExprKind::Var(n) if n == "self");
        match &mut e.kind {
            ExprKind::Call(callee, args) => {
                let ExprKind::Var(name) = &callee.kind else { return };
                let Some(inputs) = ctx.bind_views.get(name) else { return };
                for (i, a) in args.iter_mut().enumerate() {
                    let target = match &a.label {
                        Some(l) => inputs.iter().find(|(n, _)| n == l),
                        None => inputs.get(i),
                    };
                    if !matches!(target, Some((_, true))) { continue; }
                    let ExprKind::Var(n) = &a.value.kind else { continue };
                    let (vl, vc, vn) = (a.value.line, a.value.col, a.value.len);
                    if ctx.states.contains(n) {
                        let n = n.clone();
                        a.value = field(var(ST, vl, vc, vn), &n);
                    } else if ctx.binds.contains(n) {
                        a.value = var(&format!("{KEEP}{n}"), vl, vc, vn);
                    }
                }
            }
            ExprKind::Var(n) if ctx.states.contains(n) && ctx.has_state => {
                let n = n.clone();
                let slot = field(var(ST, line, col, len), &n);
                e.kind = if ctx.bound.contains(&n) { field(slot, "value").kind } else { slot.kind };
            }
            ExprKind::Var(n) if ctx.binds.contains(n) => {
                let n = format!("{KEEP}{n}");
                e.kind = field(var(&n, line, col, len), "value").kind;
            }
            ExprKind::Var(n) if ctx.envs.contains(n) => {
                let n = n.clone();
                e.kind = field(var(ENV, line, col, len), &n).kind;
            }
            ExprKind::Field(obj, f) if is_self(obj) && ctx.states.contains(f) => {
                let f = f.clone();
                let slot = field(var(ST, line, col, len), &f);
                e.kind = if ctx.bound.contains(&f) { field(slot, "value").kind } else { slot.kind };
            }
            ExprKind::Field(obj, f) if is_self(obj) && ctx.binds.contains(f) => {
                let kept = format!("{KEEP}{f}");
                let inner = field(var("self", line, col, len), &kept);
                e.kind = field(inner, "value").kind;
            }
            ExprKind::Field(obj, f) if is_self(obj) && ctx.envs.contains(f) => {
                obj.kind = ExprKind::Var(ENV.to_string());
            }
            ExprKind::MethodCall(obj, m, args) if ctx.own_methods.contains(m) && is_self(obj) => {
                if ctx.has_state {
                    args.push(Arg { label: None, value: var(ST, line, col, len), spread: false, default_rest: false });
                }
                if ctx.has_env {
                    args.push(Arg { label: None, value: var(ENV, line, col, len), spread: false, default_rest: false });
                }
            }
            _ => {}
        }
    };
    let mut noop = |_: &mut Stmt| {};
    Walker { expr: &mut on_expr, stmt: &mut noop }.stmts(body);

    // Pass 3 — drop the `KEEP` markers.
    let mut strip = |e: &mut Expr| match &mut e.kind {
        ExprKind::Var(n) if n.starts_with(KEEP) => { n.remove(0); }
        ExprKind::Field(_, f) if f.starts_with(KEEP) => { f.remove(0); }
        _ => {}
    };
    let mut noop = |_: &mut Stmt| {};
    Walker { expr: &mut strip, stmt: &mut noop }.stmts(body);
}

/// Lowers every `view` (state, `bind`, `env`, native facades) to ordinary structs.
pub fn lower_views(mut program: Program) -> Result<Program, ParseError> {
    if let Some(e) = crate::checker::check_view_declarations(&program).into_iter().next() {
        return Err(err(e.message, e.line, e.col));
    }
    let mut bind_views = BindViews::new();
    collect_bind_views(&program.items, &mut bind_views);
    let env = Env {
        view_trait: declares(&program.items, &|i| matches!(i, Item::Trait(t) if t.name == "View"))
            || has_ui_use(&program.items),
        has_slot: declares(&program.items, &|i| matches!(i, Item::Struct(s) if s.name == "Slot"))
            || has_ui_use(&program.items),
        bind_views,
    };
    program.items = lower_items(program.items, &env)?;
    Ok(program)
}

fn has_ui_use(items: &[Item]) -> bool {
    items.iter().any(|i| match i {
        Item::Use(u) => u.path.first().map(String::as_str) == Some("boring")
            && u.path.get(1).map(String::as_str) == Some("ui"),
        Item::Mod(m) => has_ui_use(&m.items),
        _ => false,
    })
}

/// Everything a freshly parsed module needs before it is executed or emitted: default view
/// permissions, then view lowering. The main pipeline calls the two passes itself; module
/// loaders (`use` of a sibling file or `boring.*`) call this right after parsing.
pub fn lower_module(program: Program) -> Result<Program, ParseError> {
    crate::view_perm::resolve_view_permissions(program).and_then(lower_views)
}
