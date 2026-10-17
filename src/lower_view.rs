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
//! ## Callbacks in transpiled output
//!
//! A callback is stored in the mounted tree, so in Rust it must be `'static`: it cannot borrow
//! the `&self` / `&mut __st` of the `body()` call that built it. `boring build` therefore (a)
//! stores *every* `state` in a `Slot<T>'actor` (not only bound ones), making `V__State` a bundle
//! of shared handles, and (b) has the transpiler build a stored closure from clones — `self`
//! becomes a `self.clone()` snapshot of the inputs (the draft's "ordinary inputs come from the
//! callback's ephemeral snapshot"), and each captured user-struct local (`__st`, a keyed `for`
//! element) a `.clone()` that shares the same slots. The interpreter keeps plain state fields.
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
//! - **Typed `bind` projections** (`profile.name` into a state struct, keyed elements; a field projection
//!   is only passed through and must already evaluate to a `Slot`), binding a `state let`, and
//!   `bind`'s permission levels (the child can always replace and mutate: the slot is `var mut`).
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

static RUNTIME_METHODS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Whether lowered views get the generated runtime-facing `View` methods (kind, children, state
/// seed, `expand`). On by default (`boring run`); `boring build` turns it off, because the
/// transpiled Rust cannot express them yet: returning a native view's owned `[View]` children
/// needs a by-reference/`clone_box` story for `Box<dyn View>`, and an expanded body that borrows
/// its mounted state cannot outlive the call. The `View` trait's methods all have defaults, so
/// views without the generated impls still compile.
pub fn set_runtime_methods(on: bool) {
    RUNTIME_METHODS.store(on, std::sync::atomic::Ordering::Relaxed);
}

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
                let (aliases, lowered) = lower_view(s, env)?;
                out.extend(aliases.into_iter().map(Item::Alias));
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
    proj_vars: &HashMap<String, usize>,
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
                    ExprKind::Var(n) if proj_vars.contains_key(n) => {}
                    // A projection (`item.completed`) is passed through untouched: it must evaluate
                    // to a `Slot` (an input struct's shared field) — typed element projections are
                    // not supported yet, so this is the application's responsibility.
                    ExprKind::Field(..) => {}
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

fn lower_view(mut s: StructDecl, env: &Env) -> Result<(Vec<AliasDecl>, Vec<StructDecl>), ParseError> {
    // A view automatically conforms to `View` — when the trait is in the program (`use boring.ui`).
    if env.view_trait && !s.protocols.iter().any(|p| p == "View") {
        s.protocols.push("View".to_string());
    }
    // A native view's adapter is supplied by the UI runtime: it keeps its inputs, loses the
    // placeholder `body(): native`, and may not own application `state`.
    let native = s.methods.iter().any(|m| m.is_native);
    if native {
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
    // Transpiled output (no runtime methods) stores *every* `state` in a `Slot`: `V__State` is
    // then a bundle of shared handles, so a callback — `'static`, stored in the mounted tree —
    // captures a clone of it that still reads and writes the mounted storage (see the module
    // docs, "Callbacks"). The interpreter keeps plain fields: its closures capture by reference.
    let slot_all = env.has_slot && !RUNTIME_METHODS.load(std::sync::atomic::Ordering::Relaxed);
    let binds: HashSet<String> = bind_fields.iter().map(|f| f.name.clone()).collect();
    let envs: HashSet<String> = env_fields.iter().map(|f| f.name.clone()).collect();
    // Keyed-loop elements handed to a child's `bind` (see "bind projections" below). Detected on the
    // untouched view; the hidden `__proj` registry state is added before the field lists are read.
    let state_tys: HashMap<String, Type> = state_fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
    let projections = if env.view_trait { find_projections(&s, &state_tys, &env.bind_views)? } else { vec![] };
    let mut proj_vars: HashMap<String, usize> = HashMap::new();
    for (k, p) in projections.iter().enumerate() {
        if let Some(prev) = proj_vars.insert(p.loop_var.clone(), k) {
            if projections[prev].coll != p.coll {
                return Err(err(
                    format!("loop variable `{}` projects two different collections (`{}` and `{}`) in view '{}'; rename one", p.loop_var, projections[prev].coll, p.coll, s.name),
                    p.line, p.col,
                ));
            }
        }
    }
    let mut state_fields = state_fields;
    let mut states = states;
    for (k, _) in projections.iter().enumerate() {
        let registry = projection_registry_field(&s.name, k);
        states.insert(registry.name.clone());
        state_fields.push(registry);
    }
    let mut bound = find_bound_states(&s, &states, &binds, &env.bind_views, &proj_vars)?;
    // A projected collection is a shared slot: the projections write back into it by key.
    for p in &projections { bound.insert(p.coll.clone()); }
    if let Some(f) = state_fields.iter().find(|f| bound.contains(&f.name) && !f.mutable && !f.ty.is_mut()) {
        return Err(err(
            format!("`state {}` is read-only (`state let`) but is bound to a child's `bind`", f.name),
            f.line, f.col,
        ));
    }

    let slotted: &HashSet<String> = if slot_all { &states } else { &bound };
    let state_name = format!("{}__State", s.name);
    let env_name = format!("{}__Env", s.name);
    let mut generated = Vec::new();

    if !state_fields.is_empty() {
        let fields = state_fields.iter().map(|f| {
            let mut f = f.clone();
            f.view_kind = None;
            f.explicit_perm = false;
            f.is_pub = true;
            if slot_all || bound.contains(&f.name) {
                let seed = slot_seed(f.default.as_ref().expect("checked above"));
                f.ty = slot_type(&f.ty);
                f.mutable = true;
                f.default = Some(seed);
            }
            f
        }).collect();
        let mut state = plain_struct(state_name.clone(), s.is_pub, fields, s.line, s.col);
        if env.view_trait { state.protocols.push("StateBox".to_string()); }
        generated.push(state);
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
    // Every `state` can be seeded by the parent at construction: an optional lazy seed closure
    // input per state field (see "parent-supplied state seeds" below).
    let mut seeds: Vec<(String, bool, Type)> = Vec::new();
    let mut aliases: Vec<AliasDecl> = Vec::new();
    if env.view_trait {
        for f in &state_fields {
            let mut seed = f.clone();
            seed.name = seed_field_name(&f.name);
            seed.view_kind = None;
            seed.explicit_perm = false;
            seed.is_pub = true;
            seed.mutable = false;
            // A named function-type alias per seed: a bare `def T()` field type is not a storable
            // closure type in transpiled output, an alias is (like `UIAction`).
            let alias = format!("{}__{}__Seed", s.name, f.name);
            aliases.push(AliasDecl {
                name: alias.clone(), type_params: vec![], newtype: false, line: f.line, col: f.col,
                ty: Type::Fn(Some(Box::new(strip_mut(&f.ty).clone())), vec![], false, false, false),
            });
            // Absent unless the parent supplied one: `newState()` then uses the declaration's initializer.
            seed.ty = Type::Optional(Box::new(Type::Named(alias)));
            seed.default = Some(Expr { kind: ExprKind::Nil, line: f.line, col: f.col, len: 3 });
            s.fields.push(seed);
            seeds.push((f.name.clone(), slotted.contains(&f.name), strip_mut(&f.ty).clone()));
        }
    }
    s.is_view = false;

    let has_state = !state_fields.is_empty();
    let has_env = !env_fields.is_empty();
    if has_state || has_env || !binds.is_empty() {
        let ctx = Ctx {
            states: &states, binds: &binds, envs: &envs, bound: slotted,
            own_methods: s.methods.iter().map(|m| m.name.clone()).collect(),
            bind_views: &env.bind_views,
            has_state, has_env,
            proj_vars: &proj_vars,
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
    if !projections.is_empty() {
        let registry_refs: Vec<String> = (0..projections.len()).map(|k| {
            if slotted.contains(&format!("__proj{k}")) { format!("__st.__proj{k}.value") } else { format!("__st.__proj{k}") }
        }).collect();
        let (structs, methods) = projection_items(&s.name, &state_name, &projections, &registry_refs, slot_all)?;
        generated.extend(structs);
        // Each build of the body starts fresh registries: only the last build's projections are live.
        for m in s.methods.iter_mut().filter(|m| m.name == "body") {
            let reset: Vec<Stmt> = registry_refs.iter().flat_map(|r| parse_stmts(&format!("{r} = []"))).collect();
            for (i, st) in reset.into_iter().enumerate() { m.body.insert(i, st); }
        }
        s.methods.extend(methods);
    }
    if env.view_trait {
        s.methods.extend(view_methods(&s, native, has_state, has_env, &state_name, &seeds, &env_fields));
    }
    // Transpiled output: `body()` is the `<View>` opaque return the parser gave it, which Rust
    // can only satisfy with ONE concrete type — but a body may return `Row` on one branch and
    // `Column` on another. A bare `View` is the dynamic (`Arc<dyn View>`) return instead.
    if env.view_trait && !RUNTIME_METHODS.load(std::sync::atomic::Ordering::Relaxed) {
        for m in s.methods.iter_mut().filter(|m| m.name == "body") {
            if matches!(&m.return_ty, Some(Type::Impl(t)) if matches!(&**t, Type::Named(n) if n == "View")) {
                m.return_ty = Some(Type::Named("View".to_string()));
            }
        }
    }
    generated.push(s);
    Ok((aliases, generated))
}

/// The `View` trait's methods for a lowered view (see `stdlib/ui.br`): kind, structural children
/// of a native container, a native `KeyedGroup`'s key/site, and — for a composite view — the
/// state seed and `expand`, which re-enters the lowered `body(__st)` with the node's own state.
/// Parsed from source text so the generated methods are exactly what the parser would produce.
fn view_methods(s: &StructDecl, native: bool, has_state: bool, has_env: bool, state_name: &str, seeds: &[(String, bool, Type)], env_fields: &[FieldDecl]) -> Vec<FnDecl> {
    let field_ty = |name: &str| s.fields.iter().find(|f| f.name == name).map(|f| strip_mut(&f.ty).clone());
    let mut src = format!("struct G:\n    req string viewKind(): \"{}\"\n", s.name);
    if native {
        if matches!(field_ty("children"), Some(Type::Array(t)) if matches!(*t, Type::Named(ref n) if n == "View")) {
            src.push_str("    req [View] viewChildren(): children\n");
        }
        // Scalar fields, stringified, are the props a layout/paint pass reads.
        let mut props = String::new();
        for f in s.fields.iter().filter(|f| f.name != "key" && f.name != "site") {
            let ty = strip_mut(&f.ty).clone();
            match &ty {
                Type::Array(_) | Type::Dict(..) | Type::Set(_) | Type::Fn(..) | Type::Tuple(_) => {}
                Type::Optional(inner) if !matches!(**inner, Type::Array(_)) => {
                    props.push_str(&format!("        if let v = {name}:\n            p[\"{name}\"] = \"{{v}}\"\n", name = f.name));
                }
                Type::Named(n) if n == "View" || n == "UIAction" || n == "TaskAction" || n == "EnvValue" => {}
                // A lowered `bind` field is a shared `Slot`: its prop is the current value.
                Type::Qualified(inner, OwnerQual::Actor) if matches!(&**inner, Type::Generic(g, _) if g == "Slot") => {
                    props.push_str(&format!("        p[\"{name}\"] = \"{{{name}.value}}\"\n", name = f.name));
                }
                _ => props.push_str(&format!("        p[\"{name}\"] = \"{{{name}}}\"\n", name = f.name)),
            }
        }
        // Semantic actions: a native control's reaction, derived from its facade's shape.
        let mut acts = String::new();
        if s.fields.iter().any(|f| f.name == "onActivate" && is_named(strip_mut(&f.ty), "UIAction")) {
            acts.push_str("        if action == \"activate\":\n            onActivate()\n");
        }
        // Lifecycle modifiers (`OnMount`/`OnUnmount`): the runtime delivers the hook as an action.
        for (field, action) in [("onMount", "mount"), ("onUnmount", "unmount")] {
            if s.fields.iter().any(|f| f.name == field && is_named(strip_mut(&f.ty), "UIAction")) {
                acts.push_str(&format!("        if action == \"{action}\":\n            {field}()\n"));
            }
        }
        if let Some(f) = s.fields.iter().find(|f| f.name == "value") {
            if let Type::Qualified(inner, OwnerQual::Actor) = strip_mut(&f.ty) {
                if let Type::Generic(g, args) = &**inner {
                    if g == "Slot" && args.len() == 1 && is_named(&args[0], "bool") {
                        acts.push_str("        if action == \"toggle\":\n            value.value = !value.value\n");
                    }
                    if g == "Slot" && args.len() == 1 && is_named(&args[0], "string") {
                        acts.push_str("        if action == \"setText\":\n            value.value = arg\n");
                    }
                }
            }
        }
        // `OnTask`: the runtime starts a task run by handing the body its `TaskRun`.
        if s.fields.iter().any(|f| f.name == "onTask" && is_named(strip_mut(&f.ty), "TaskAction")) {
            src.push_str("    req void viewTask(TaskRun run):\n        onTask(run)\n");
        }
        if !acts.is_empty() {
            src.push_str(&format!("    req void viewAct(string action, string arg):\n{acts}"));
        }
        if !props.is_empty() {
            src.push_str(&format!("    req {{string=string}} viewProps():\n        mut {{string=string}} p = {{=}}\n{props}        p\n"));
        }
        // `Provide`: the value it installs for its subtree's `env` fields.
        if field_ty("value").is_some_and(|t| is_named(&t, "EnvValue")) {
            src.push_str("    req EnvValue? viewProvides(): value\n");
        }
        if field_ty("key").is_some_and(|t| is_named(&t, "string")) {
            src.push_str("    req string viewKey(): key\n");
        }
        if field_ty("site").is_some_and(|t| is_named(&t, "int")) {
            src.push_str("    req int viewSite(): site\n");
        }
    } else if s.methods.iter().any(|m| m.name == "body") && (!has_env || env_expandable(env_fields)) {
        // (A view with `env` is expandable when every `env` type is a named struct the runtime's
        // provider scope can be searched for; a primitive `env` has nothing to look up by type.)
        src.push_str("    req bool isComposite(): true\n");
        if has_state {
            // The mount-time state: per `state` field, the parent's seed when it supplied one (a lazy
            // closure, evaluated here — once, because a rebuild of the same identity never calls
            // `newState`), else the declaration's own initializer, read off `V__State()`'s default.
            // A constructor expression, not assignments: whole-field replacement is forbidden on `mut`
            // fields, and transpiled output cannot return a mutated local as a `'shared` trait
            // object. `seededState` is the same state before it is boxed. (The `__initial` helpers'
            // return types are patched in below: a `Type` cannot be spelled back into source.)
            let mut args = Vec::new();
            for (name, slotted, _) in seeds {
                let tail = if *slotted { format!("{name}.value") } else { name.clone() };
                src.push_str(&format!(
                    "    req int {name}__initial():\n        if let seed = {}:\n            return seed()\n        {state_name}().{tail}\n",
                    seed_field_name(name)
                ));
                args.push(if *slotted { format!("{name} = Slot(value = self.{name}__initial())") } else { format!("{name} = self.{name}__initial()") });
            }
            src.push_str(&format!(
                "    req {state_name} seededState(): {state_name}({})\n    req StateBox'shared newState(): self.seededState()\n",
                args.join(", ")
            ));
            if !has_env {
                src.push_str(&format!(
                    "    req View expand(StateBox'shared st):\n        mut s = st as {state_name} else ({state_name}())\n        self.body(s)\n"
                ));
            }
        } else if !has_env {
            src.push_str("    req View expand(StateBox'shared st):\n        self.body()\n");
        }
        if has_env {
            // The runtime hands a composite the provider scope of its ancestors (nearest first); each
            // `env` field is the first value in it that downcasts to the field's type. A missing
            // provider renders an `EnvMissing` placeholder the runtime reports. `expand` is the same
            // with no providers, so a bare `expand(st)` still works (and reports the missing env).
            src.push_str("    req bool usesEnv(): true\n");
            let mut body = String::from("    req View expandIn(StateBox'shared st, [EnvValue] scope):\n");
            if has_state {
                body.push_str(&format!("        mut s = st as {state_name} else ({state_name}())\n"));
            }
            let mut args = Vec::new();
            for f in env_fields {
                let Type::Named(ty) = strip_mut(&f.ty) else { continue };
                let n = &f.name;
                body.push_str(&format!(
                    "        var {ty}? found__{n} = nil\n        for v in scope:\n            if found__{n} == nil:\n                if let t = v as {ty}:\n                    found__{n} = t\n        guard let {n} = found__{n} else:\n            return EnvMissing(what = \"{ty}\")\n"
                ));
                args.push(format!("{n} = {n}"));
            }
            let env_name = format!("{}__Env", s.name);
            body.push_str(&format!("        let e = {env_name}({})\n", args.join(", ")));
            body.push_str(if has_state { "        self.body(s, e)\n" } else { "        self.body(e)\n" });
            src.push_str(&body);
            src.push_str("    req View expand(StateBox'shared st): self.expandIn(st, [])\n");
        }
    }
    let tokens = crate::lexer::lex(&src).expect("generated View methods lex");
    let program = crate::parser::parse(tokens).expect("generated View methods parse");
    for item in program.items {
        if let Item::Struct(mut g) = item {
            for m in &mut g.methods {
                if let Some(name) = m.name.strip_suffix("__initial") {
                    if let Some((_, _, ty)) = seeds.iter().find(|(n, _, _)| n == name) {
                        m.return_ty = Some(ty.clone());
                    }
                }
            }
            return g.methods;
        }
    }
    vec![]
}

fn is_named(t: &Type, name: &str) -> bool {
    matches!(t, Type::Named(n) if n == name)
}

fn strip_mut(t: &Type) -> &Type {
    match t { Type::Mut(inner) => strip_mut(inner), other => other }
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
    /// Keyed-loop variables passed to a child's `bind`: loop var -> projection index.
    proj_vars: &'a HashMap<String, usize>,
}

/// Marker prefix for names that pass 2 must not rewrite again; stripped by pass 3.
const KEEP: char = '@';

/// A closure bound with `let` and only ever passed along as a call argument
/// (`let gone = (): ...` then `OnMount(onMount = gone)`) is moved into each argument position.
/// Transpiled output builds a callback's `'static` capture list (cloned state handles, the
/// `self` snapshot) and types its untyped parameters from the *field it is stored in* only
/// where the closure literal itself is the argument; a plain `let` closure gets neither.
fn inline_let_closures(body: &mut [Stmt]) -> Vec<(String, Expr)> {
    fn is_closure_let(l: &LetStmt) -> bool {
        matches!(&l.value, Some(Expr { kind: ExprKind::Closure(..), .. }))
    }
    let mut lets: HashMap<String, (Expr, usize)> = HashMap::new();
    {
        let mut on_expr = |_: &mut Expr| {};
        let mut on_stmt = |st: &mut Stmt| {
            if let Stmt::Let(l) = st {
                if is_closure_let(l) {
                    lets.entry(l.name.clone()).or_insert((l.value.clone().unwrap(), 0)).1 += 1;
                }
            }
        };
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
    }
    if lets.is_empty() { return vec![]; }
    // Any other binding of the same name (let, closure parameter, loop variable) is ambiguous.
    let mut other_defs: HashSet<String> = HashSet::new();
    let mut total: HashMap<String, usize> = HashMap::new();
    let mut as_arg: HashMap<String, usize> = HashMap::new();
    {
        let (total, as_arg, other) = (
            std::cell::RefCell::new(&mut total),
            std::cell::RefCell::new(&mut as_arg),
            std::cell::RefCell::new(&mut other_defs),
        );
        let mut on_expr = |e: &mut Expr| match &e.kind {
            ExprKind::Var(n) => { *total.borrow_mut().entry(n.clone()).or_default() += 1; }
            ExprKind::Closure(params, ..) => {
                for p in params { other.borrow_mut().insert(p.name.clone()); }
            }
            ExprKind::Call(_, args) | ExprKind::GenericCall(_, _, args)
            | ExprKind::MethodCall(_, _, args) | ExprKind::Pipe(_, _, args)
            | ExprKind::OptionalMethodCall(_, _, args) | ExprKind::TrailingArrayBlock { args, .. } => {
                for a in args.iter().filter(|a| !a.spread) {
                    if let ExprKind::Var(n) = &a.value.kind {
                        *as_arg.borrow_mut().entry(n.clone()).or_default() += 1;
                    }
                }
            }
            _ => {}
        };
        let mut on_stmt = |st: &mut Stmt| match st {
            Stmt::Let(l) if !is_closure_let(l) => { other.borrow_mut().insert(l.name.clone()); }
            Stmt::LetDestructure(l) => {
                for b in &l.bindings { other.borrow_mut().insert(b.name.clone()); }
            }
            Stmt::For(f) => { for v in &f.vars { other.borrow_mut().insert(v.clone()); } }
            _ => {}
        };
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
    }
    // Bound closures that are passed as an argument but cannot be inlined (also called or read
    // elsewhere): the caller reports the ones a stored callback could not use as they are.
    let mut stuck: Vec<(String, Expr)> = lets.iter()
        .filter(|(n, (_, defs))| as_arg.get(*n).copied().unwrap_or(0) > 0
            && (*defs != 1 || other_defs.contains(*n) || total.get(*n) != as_arg.get(*n)))
        .map(|(n, (c, _))| (n.clone(), c.clone()))
        .collect();
    stuck.sort_by_key(|(_, c)| (c.line, c.col));
    lets.retain(|n, (_, defs)| {
        *defs == 1 && !other_defs.contains(n)
            && total.get(n).copied().unwrap_or(0) > 0
            && total.get(n) == as_arg.get(n)
    });
    if lets.is_empty() { return stuck; }
    let mut on_expr = |e: &mut Expr| match &mut e.kind {
        ExprKind::Call(_, args) | ExprKind::GenericCall(_, _, args)
        | ExprKind::MethodCall(_, _, args) | ExprKind::Pipe(_, _, args)
        | ExprKind::OptionalMethodCall(_, _, args) | ExprKind::TrailingArrayBlock { args, .. } => {
            for a in args.iter_mut().filter(|a| !a.spread) {
                if let ExprKind::Var(n) = &a.value.kind {
                    if let Some((c, _)) = lets.get(n) { a.value = c.clone(); }
                }
            }
        }
        _ => {}
    };
    let mut on_stmt = |st: &mut Stmt| {
        if let Stmt::Let(l) = st {
            if is_closure_let(l) && lets.contains_key(&l.name) {
                *st = Stmt::Comment(format!("closure `{}` inlined at its use sites", l.name));
            }
        }
    };
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
    stuck
}

fn rewrite_body(body: &mut [Stmt], ctx: &Ctx, view: &str, failure: &mut Option<ParseError>) {
    let is_field = |n: &str| ctx.states.contains(n) || ctx.binds.contains(n) || ctx.envs.contains(n);
    for (name, mut c) in inline_let_closures(body) {
        if RUNTIME_METHODS.load(std::sync::atomic::Ordering::Relaxed) { break; }
        let ExprKind::Closure(params, ..) = &c.kind else { continue };
        let untyped = params.iter().any(|p| p.ty.is_none());
        let mut captures = false;
        {
            let mut on_expr = |e: &mut Expr| {
                if let ExprKind::Var(n) = &e.kind { captures |= n == "self" || is_field(n); }
            };
            let mut on_stmt = |_: &mut Stmt| {};
            Walker { expr: &mut on_expr, stmt: &mut on_stmt }.expr(&mut c);
        }
        if untyped || captures {
            failure.get_or_insert_with(|| err(
                format!(
                    "closure `{name}` is passed as a callback and also used elsewhere in view '{view}': a stored \
                     callback must be built where it is passed (it captures the view's state handles and takes its \
                     parameter types from the field) — write the closure inline at the call, or use it only as a \
                     callback argument"
                ),
                c.line, c.col,
            ));
            return;
        }
    }

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
                    if let Some(k) = ctx.proj_vars.get(n) {
                        // A keyed-loop element handed to a child's `bind`: a projection slot for it.
                        let arg = |v: Expr| Arg { label: None, value: v, spread: false, default_rest: false };
                        let item = var(n, vl, vc, vn);
                        let made = Expr {
                            kind: ExprKind::MethodCall(
                                Box::new(var("self", vl, vc, vn)),
                                format!("__project{k}"),
                                vec![arg(item), arg(var(ST, vl, vc, vn))],
                            ),
                            line: vl, col: vc, len: vn,
                        };
                        a.value = field(made, "slot");
                    } else if ctx.states.contains(n) {
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
    crate::desugar_view_modifiers::desugar_view_modifiers(&mut program.items);
    if declares(&program.items, &|i| matches!(i, Item::Trait(t) if t.name == "View")) || has_ui_use(&program.items) {
        rewrite_seed_calls(&mut program.items);
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

// ── parent-supplied state seeds ────────────────────────────────────────────────────────────────
//
// `Counter(title = t, count = expensive())` seeds the `state int count` of that identity on its
// FIRST mount only (draft, "Initialization and seeds"). The description therefore carries each seed
// as an optional lazy closure field `<state>__seed: Seed<T>?` (an alias of `def T()` in
// `boring.ui`): the call site's argument is wrapped in `(): <expr>`, the generated `newState()`
// evaluates it exactly once, and a later rebuild of the same identity never evaluates it, because
// the runtime only calls `newState()` at mount and `expand` reuses the persistent state.

const SEED_SUFFIX: &str = "__seed";

fn seed_field_name(state: &str) -> String { format!("{state}{SEED_SUFFIX}") }

fn collect_state_views(items: &[Item], out: &mut HashMap<String, HashMap<String, Type>>) {
    for item in items {
        match item {
            Item::Struct(s) if s.is_view => {
                let states: HashMap<String, Type> = s.fields.iter()
                    .filter(|f| f.view_kind == Some(ViewFieldKind::State))
                    .map(|f| (f.name.clone(), f.ty.clone()))
                    .collect();
                if !states.is_empty() { out.insert(s.name.clone(), states); }
            }
            Item::Mod(m) => collect_state_views(&m.items, out),
            _ => {}
        }
    }
}

/// `(): <value>` — built from a parsed template so the closure's flags match source-written ones.
fn seed_closure(value: Expr) -> Expr {
    let tokens = crate::lexer::lex("let f = (): 0\n").expect("seed template lexes");
    let mut program = crate::parser::parse(tokens).expect("seed template parses");
    let mut template: Option<Expr> = None;
    let mut grab = |e: &mut Expr| { if template.is_none() && matches!(e.kind, ExprKind::Closure(..)) { template = Some(e.clone()); } };
    let mut noop = |_: &mut Stmt| {};
    crate::desugar_view_modifiers::walk_bodies(&mut program.items, &mut |body| {
        Walker { expr: &mut grab, stmt: &mut noop }.stmts(body);
    });
    let mut closure = template.expect("seed template has a closure");
    let (line, col, len) = (value.line, value.col, value.len);
    if let ExprKind::Closure(_, _, body, _, _) = &mut closure.kind {
        *body = ClosureBody::Expr(Box::new(value));
    }
    closure.line = line;
    closure.col = col;
    closure.len = len;
    closure
}

fn rewrite_seed_calls(items: &mut [Item]) {
    let mut table = HashMap::new();
    collect_state_views(items, &mut table);
    if table.is_empty() { return; }
    let wrap = |name: &str, args: &mut Vec<Arg>| {
        let Some(states) = table.get(name) else { return };
        for a in args.iter_mut() {
            if let Some(l) = a.label.clone() {
                if let Some(ty) = states.get(&l) {
                    a.label = Some(seed_field_name(&l));
                    let value = std::mem::replace(&mut a.value, Expr { kind: ExprKind::Nil, line: 0, col: 0, len: 0 });
                    a.value = seed_closure(value);
                }
            }
        }
    };
    let mut on_expr = |e: &mut Expr| match &mut e.kind {
        ExprKind::Call(callee, args) => {
            if let ExprKind::Var(n) = &callee.kind { wrap(n, args); }
        }
        ExprKind::TrailingArrayBlock { callee, args, .. } => {
            if let ExprKind::Var(n) = &callee.kind { wrap(n, args); }
        }
        _ => {}
    };
    let mut noop = |_: &mut Stmt| {};
    crate::desugar_view_modifiers::walk_bodies(items, &mut |body| {
        Walker { expr: &mut on_expr, stmt: &mut noop }.stmts(body);
    });
}

fn env_expandable(env_fields: &[FieldDecl]) -> bool {
    const PRIMITIVES: &[&str] = &["int", "uint", "float", "double", "bool", "string", "char", "byte", "usize", "isize"];
    !env_fields.is_empty() && env_fields.iter().all(|f| {
        matches!(strip_mut(&f.ty), Type::Named(n) if !PRIMITIVES.contains(&n.as_str()))
    })
}

// ── bind projections (keyed collection elements) ───────────────────────────────────────────────
//
//     for item in items with item.id:        # `items` is a `state [Todo]` of this view
//         TodoRow(item = item)               # TodoRow declares `bind Todo item`
//
// The child's `bind Todo item` stays an ordinary `Slot<Todo>'actor`. For a keyed-loop element passed to
// it, the parent hands a *projection slot*: a fresh slot holding the element, registered in the
// parent's hidden `__proj` state together with the collection slot and the element's KEY. At the end
// of an event, before the update pass rebuilds anything, the runtime asks every mounted view to
// `commitProjections`: a projection whose slot changed since it was made is written back into the
// collection at the element whose key still matches (the index is never captured, so a reorder cannot
// redirect a write); a key that is gone is a stale projection and the write is dropped and reported.
// The registry is rebuilt by every `body()`.
//
// "Changed" is decided by comparing the printed form of the slot against the base copy taken at
// projection time (`==` on structs is identity in the interpreter, content in transpiled output).
//
// Known limits: the write-back is transactional (the collection is updated at commit, not at the
// child's write), a child's change wins over a parent change to the same element in the same event,
// and the key expression may only use the loop variable.

struct Projection {
    coll: String,
    elem: String,
    loop_var: String,
    key: Expr,
    line: usize,
    col: usize,
}

fn parse_stmts(body: &str) -> Vec<Stmt> {
    let src = format!("def f():\n    {}\n", body.replace('\n', "\n    "));
    let tokens = crate::lexer::lex(&src).expect("generated statements lex");
    let program = crate::parser::parse(tokens).expect("generated statements parse");
    for item in program.items {
        if let Item::Fn(f) = item { return f.body; }
    }
    vec![]
}

fn projection_registry_field(view: &str, k: usize) -> FieldDecl {
    let src = format!("view G:\n    state var mut [{view}__Proj{k}] __proj{k} = []\n");
    let src = src.as_str();
    let tokens = crate::lexer::lex(src).expect("registry field lexes");
    let program = crate::parser::parse(tokens).expect("registry field parses");
    for item in program.items {
        if let Item::Struct(g) = item { return g.fields.into_iter().next().expect("registry field"); }
    }
    unreachable!("registry field parses to a view")
}

fn find_projections(
    s: &StructDecl,
    state_tys: &HashMap<String, Type>,
    bind_views: &BindViews,
) -> Result<Vec<Projection>, ParseError> {
    let mut found: Vec<Projection> = Vec::new();
    let mut failure: Option<ParseError> = None;
    for m in &s.methods {
        let mut body = m.body.clone();
        let mut on_stmt = |st: &mut Stmt| {
            let Stmt::For(f) = st else { return };
            if f.vars.len() != 1 { return; }
            let ExprKind::Var(coll) = &f.iterable.kind else { return };
            let Some(ty) = state_tys.get(coll) else { return };
            let Type::Array(elem) = strip_mut(ty) else { return };
            let Type::Named(elem) = &**elem else { return };
            let loop_var = f.vars[0].clone();
            let mut inner = f.body.clone();
            let mut uses = false;
            let mut key: Option<Expr> = None;
            let mut on_expr = |e: &mut Expr| {
                let ExprKind::Call(callee, args) = &e.kind else { return };
                let ExprKind::Var(name) = &callee.kind else { return };
                if name == "KeyedGroup" {
                    if key.is_none() { key = args.first().map(|a| a.value.clone()); }
                } else if let Some(inputs) = bind_views.get(name) {
                    for (i, a) in args.iter().enumerate() {
                        let target = match &a.label {
                            Some(l) => inputs.iter().find(|(n, _)| n == l),
                            None => inputs.get(i),
                        };
                        if matches!(target, Some((_, true))) && matches!(&a.value.kind, ExprKind::Var(n) if *n == loop_var) {
                            uses = true;
                        }
                    }
                }
            };
            let mut noop = |_: &mut Stmt| {};
            Walker { expr: &mut on_expr, stmt: &mut noop }.stmts(&mut inner);
            if !uses { return; }
            match key {
                Some(key) => found.push(Projection {
                    coll: coll.clone(), elem: elem.clone(), loop_var, key, line: f.line, col: f.col,
                }),
                None => {
                    failure.get_or_insert_with(|| err(
                        format!(
                            "`{loop_var}` is passed to a child's `bind` but the loop has no identity key: write \
                             `for {loop_var} in {coll} with <key>:` so the binding follows the element, not its index"
                        ),
                        f.line, f.col,
                    ));
                }
            }
        };
        let mut noop = |_: &mut Expr| {};
        Walker { expr: &mut noop, stmt: &mut on_stmt }.stmts(&mut body);
    }
    match failure {
        Some(e) => Err(e),
        None => Ok(found),
    }
}

/// The generated pieces of every projection: one concrete struct per site (the collection slot, the
/// projection slot, a base copy to detect a change, the key, and `writeBack`), held in a typed array
/// of the parent's hidden state — no trait objects and no closures, neither of which `boring build`
/// can carry through a shared state — plus the view methods that make a projection slot and write
/// them all back.
fn projection_items(
    view: &str,
    state_name: &str,
    projs: &[Projection],
    registry: &[String],
    shared: bool,
) -> Result<(Vec<StructDecl>, Vec<FnDecl>), ParseError> {
    // Transpiled output keeps every state in shared handles: pushing into one needs no `&mut` state,
    // and demanding it would clash with the loop that is iterating the (locked) collection slot.
    let st_param = if shared { "" } else { "mut " };
    let mut structs = Vec::new();
    let mut src = String::from("struct G:\n");
    let mut key_patch: Vec<Expr> = Vec::new();
    for (k, p) in projs.iter().enumerate() {
        let (coll, elem, lv) = (&p.coll, &p.elem, &p.loop_var);
        let site = format!("{view}__Proj{k}");
        let struct_src = format!(
            "struct {site}:\n\
             \x20   pub Slot<[{elem}]>'actor coll\n\
             \x20   pub Slot<{elem}>'actor slot\n\
             \x20   pub Slot<{elem}> base\n\
             \x20   pub string key\n\
             \x20   req string keyOf({elem} {lv}):\n\
             \x20       \"__KEY__\"\n\
             \x20   req bool writeBack():\n\
             \x20       if \"{{slot.value}}\" == \"{{base.value}}\":\n\
             \x20           return true\n\
             \x20       for i in 0..<coll.value.length:\n\
             \x20           let candidate = coll.value[i]\n\
             \x20           if self.keyOf(candidate) == key:\n\
             \x20               coll.value[i] = slot.value\n\
             \x20               return true\n\
             \x20       false\n"
        );
        let tokens = crate::lexer::lex(&struct_src).expect("projection struct lexes");
        let program = crate::parser::parse(tokens).expect("projection struct parses");
        for item in program.items {
            if let Item::Struct(mut g) = item {
                for m in &mut g.methods { patch_key(&mut m.body, &p.key); }
                structs.push(g);
            }
        }
        // Returns the whole record: the call site reads `.slot` off it (a field of a shared handle is
        // passed on as that handle, where a bare call result would be wrapped a second time).
        src.push_str(&format!(
            "    req {site} __project{k}({elem} {lv}, {st_param}{state_name} __st):\n\
             \x20       let key = \"__KEY__\"\n\
             \x20       let made = {site}(coll = __st.{coll}, slot = Slot(value = {lv}), base = Slot(value = {lv}), key = key)\n\
             \x20       {reg}.push(made)\n\
             \x20       made\n",
            reg = registry[k],
        ));
        key_patch.push(p.key.clone());
    }
    src.push_str(&format!("    req int commitProjections({st_param}{state_name} __st):\n        var stale = 0\n"));
    for reg in registry {
        src.push_str(&format!("        for p in {reg}:\n            if !p.writeBack():\n                stale += 1\n"));
    }
    src.push_str(&format!(
        "        stale\n\
         \x20   req int commitProjectionsIn(StateBox'shared st):\n\
         \x20       mut s = st as {state_name} else ({state_name}())\n\
         \x20       self.commitProjections(s)\n"
    ));
    let tokens = crate::lexer::lex(&src).expect("projection methods lex");
    let program = crate::parser::parse(tokens).expect("projection methods parse");
    for item in program.items {
        if let Item::Struct(mut g) = item {
            // The per-site key expressions are patched into `__project<k>`, in order.
            let mut idx = 0;
            for m in &mut g.methods {
                if m.name.starts_with("__project") {
                    patch_key(&mut m.body, &key_patch[idx]);
                    idx += 1;
                }
            }
            return Ok((structs, g.methods));
        }
    }
    Ok((structs, vec![]))
}

/// Replaces the `"__KEY__"` placeholder expression with the loop's key expression.
fn patch_key(body: &mut [Stmt], key: &Expr) {
    let mut on_expr = |e: &mut Expr| {
        if matches!(&e.kind, ExprKind::Str(t) if t == "__KEY__") {
            *e = key.clone();
        }
    };
    let mut noop = |_: &mut Stmt| {};
    Walker { expr: &mut on_expr, stmt: &mut noop }.stmts(body);
}
