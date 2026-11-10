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

//! `TypeTable`: what the emitter learns about the type of an expression, kept by the position of the
//! expression instead of in the per-local tables it fills while it walks (`var_struct_types`,
//! `var_types`, ... , see section 16 of docs/design-notes/typed-specialisation.md).
//!
//! First step toward typing before emission. The table is built by *recording* the answers the emitter
//! gives (`TypeTableMode::Record`), then a second emission *replays* them (`Replay`) instead of
//! consulting its own tables for the same question. If the Rust is the same, the question is answerable
//! from a table; the walk that fills the table can then be written on its own, one construct at a
//! time, and checked against the recorded one (`BORING_TYPE_TABLE=check`).
//!
//! Off by default (`BORING_TYPE_TABLE` unset): nothing changes for a normal build.
//!
//! The key is `(scope, line, col)`: `scope` is the enclosing function (`Owner::name`, a specialised
//! copy has its own name), because the copies made by the specialisations keep the positions of the
//! declaration they come from.

use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TypeTableMode {
    Off,
    /// Remember what the emitter answers.
    Record,
    /// Answer from the table when it has the question; ask the emitter otherwise (a miss).
    Replay,
    /// Answer from the walk of its own (`build_struct_names`) where it claims the position; ask the
    /// emitter for the rest.
    Walk,
    /// The emitter answers; where the walk claims a position and differs, the difference is logged
    /// (`BORING_TYPE_TABLE_LOG=<file>`), whatever the program: a gate for the walk over every program a
    /// build or a test transpiles, with no change of output.
    Check,
}

/// `(scope, line, col, what)`: `what` tells apart the expressions one source token becomes (`self` and
/// `items` of an implicit `self.items` share a position): `v:<name>` or `f:<name>`.
pub(crate) type Key = (String, usize, usize, String);

#[derive(Clone, Default, Debug)]
pub(crate) struct TypeTable {
    /// Struct (or enum) name a variable or field expression resolves to; `None` is a recorded
    /// "not known here", which is also an answer. `Err(())` is a position asked twice with different
    /// answers (an expression the emitter asks before and after a binding is registered, or a scope the
    /// key does not tell apart): the position alone does not decide, so it is never replayed.
    pub struct_names: HashMap<Key, Result<Option<String>, ()>>,
    /// Positions that became ambiguous.
    pub conflicts: usize,
    pub hits: usize,
    pub misses: usize,
    /// Walk mode: questions the walk did not answer but the emitter did (a struct name).
    pub missed_known: usize,
    pub local_hits: usize,
    pub local_misses: usize,
    pub local_conflicts: usize,
    pub decision_hits: usize,
    pub decision_conflicts: usize,
}

impl TypeTable {
    pub fn record_struct(&mut self, key: Key, answer: Option<String>) {
        match self.struct_names.get(&key) {
            Some(Ok(previous)) if *previous != answer => {
                self.conflicts += 1;
                if std::env::var_os("BORING_TYPE_TABLE_WHY").is_some() { eprintln!("CONFLICT {:?}: {:?} then {:?}", key, previous, answer); }
                self.struct_names.insert(key, Err(()));
            }
            Some(_) => {}
            None => { self.struct_names.insert(key, Ok(answer)); }
        }
    }

    /// The recorded answer, if the position decides.
    pub fn struct_answer(&self, key: &Key) -> Option<Option<String>> {
        match self.struct_names.get(key) {
            Some(Ok(answer)) => Some(answer.clone()),
            _ => None,
        }
    }
}

// ─── The walk that builds the table on its own ───────────────────────────────────────────────

use crate::ast::*;
use crate::ast_walk::Walker;

/// What a body binds: the declared type of each local, `None` when the name is bound more than once
/// to different things (so no claim is made).
type Env = HashMap<String, Option<Type>>;

/// `(scope, local name)` -> qualifier decided by `infer_qualifiers` (the speculative per-function runs of
/// `pre_infer_fn_qualifiers`): the decisions the emitter otherwise recomputes while it walks a body.
pub(crate) type Decisions = HashMap<(String, String), OwnerQual>;

/// `(scope, local name)` -> struct name: the per-body table of locals.
pub(crate) type Locals = HashMap<(String, String), Option<String>>;

struct Decls {
    /// struct name -> its fields' declared types
    fields: HashMap<String, Vec<(String, Type)>>,
    /// struct/enum names
    types: std::collections::HashSet<String>,
    /// `(owner, method)` -> declared return type (a name declared twice for one owner is left out)
    methods: HashMap<(String, String), Option<Type>>,
    /// free function -> declared return type (`None` for an overloaded name)
    fns: HashMap<String, Option<Type>>,
    /// variant name -> `(enum, payload types)`; `None` for a variant name more than one enum declares
    variants: HashMap<String, Option<(String, Vec<Type>)>>,
    /// generic structs: `Stack` makes `Stack_int` (an instance the monomorphizer names, declared in
    /// the file that declares the generic, so in no list of items)
    generics: std::collections::HashSet<String>,
    /// trait names: a local declared with one holds a struct the initialiser says
    traits: std::collections::HashSet<String>,
    enums: std::collections::HashSet<String>,
    /// enum -> variant -> payload types
    enum_variants: HashMap<String, HashMap<String, Vec<Type>>>,
}

impl Decls {
    fn known(&self, n: &str) -> bool {
        self.types.contains(n)
            || self.traits.contains(n)
            || self.generics.iter().any(|g| n.len() > g.len() && n.starts_with(g.as_str()) && n.as_bytes()[g.len()] == b'_')
    }
}

fn collect_decls(items: &[Item], d: &mut Decls) {
    for item in items {
        match item {
            Item::Struct(s) => {
                d.types.insert(s.name.clone());
                if !s.type_params.is_empty() { d.generics.insert(s.name.clone()); }
                d.fields.insert(s.name.clone(), s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect());
                for m in &s.methods { add_method(d, &s.name, m); }
            }
            Item::Enum(e) => {
                d.types.insert(e.name.clone());
                d.enums.insert(e.name.clone());
                d.enum_variants.insert(e.name.clone(), e.variants.iter().map(|v| (v.name.clone(), v.fields.iter().map(|f| f.ty.clone()).collect())).collect());
                for m in &e.methods { add_method(d, &e.name, m); }
                for v in &e.variants {
                    let payload = Some((e.name.clone(), v.fields.iter().map(|f| f.ty.clone()).collect()));
                    d.variants.entry(v.name.clone()).and_modify(|x| *x = None).or_insert(payload);
                }
            }
            Item::Ext(e) => for m in &e.methods { add_method(d, &e.type_name, m); },
            Item::Trait(t) => { d.traits.insert(t.name.clone()); }
            Item::Fn(f) if f.qualifier.is_none() => {
                let rt = f.return_ty.clone();
                d.fns.entry(f.name.clone()).and_modify(|v| *v = None).or_insert(rt);
            }
            Item::Mod(m) => collect_decls(&m.items, d),
            _ => {}
        }
    }
}

/// The types the transpiler registers itself (`Transpiler::new`, "v2 Introspect handle types"): hand-emitted
/// Rust, never declared in Boring source.
fn seed_builtins(d: &mut Decls) {
    let named = |n: &str| Type::Named(n.to_string());
    let list = |n: &str| Type::Array(Box::new(Type::Named(n.to_string())));
    let reg = |d: &mut Decls, name: &str, fields: Vec<(&str, Type)>| {
        d.types.insert(name.to_string());
        d.fields.insert(name.to_string(), fields.into_iter().map(|(n, t)| (n.to_string(), t)).collect());
    };
    reg(d, "IntrospectInfo", vec![
        ("typeName", named("string")), ("variantName", named("string")), ("variantIndex", named("uint")),
        ("fields", list("Field")), ("methods", list("Method")),
    ]);
    reg(d, "Field", vec![("name", named("string")), ("isPublic", Type::Bool), ("isInstance", Type::Bool), ("isMutable", Type::Bool), ("isRebindable", Type::Bool)]);
    reg(d, "Method", vec![
        ("name", named("string")), ("isPublic", Type::Bool), ("isInstance", Type::Bool), ("isMutable", Type::Bool),
        ("isCallable", Type::Bool), ("isThrowing", Type::Bool), ("kind", named("MethodKind")),
    ]);
}

fn add_method(d: &mut Decls, owner: &str, m: &FnDecl) {
    d.methods.entry((owner.to_string(), m.name.clone())).and_modify(|v| *v = None).or_insert(m.return_ty.clone());
}

/// The name a type is known by to the emitter's struct question: `Counter`, `Counter'actor`, `mut Counter`,
/// `Response<int, string>`, a trait object `dyn Trait`.
fn named(t: &Type) -> Option<String> {
    let mut cur = t.without_mut();
    loop {
        match cur {
            // `'observed` values go through the emitter's own notification path, not its struct question.
            Type::Qualified(_, OwnerQual::Observed) => return None,
            Type::Qualified(inner, _) | Type::Dyn(inner) | Type::Impl(inner) => cur = inner.without_mut(),
            Type::Named(n) => return Some(n.clone()),
            _ => return None,
        }
    }
}

fn bind(env: &mut Env, name: &str, ty: Option<Type>) {
    match env.get(name) {
        Some(prev) if *prev != ty => { env.insert(name.to_string(), None); }
        Some(_) => {}
        None => { env.insert(name.to_string(), ty); }
    }
}

/// The declared type of an expression, when declarations alone say it.
fn type_of(e: &Expr, env: &Env, owner: Option<&str>, d: &Decls) -> Option<Type> {
    use super::emit_methods::indexed_elem_type;
    let struct_of = |t: &Type| named(t).filter(|n| d.known(n));
    match &e.kind {
        ExprKind::Var(v) if v == "self" => owner.map(|o| Type::Named(o.to_string())),
        ExprKind::Var(v) => match env.get(v) {
            Some(known) => known.clone(),
            // A bare field name inside a method means `self.<name>`.
            None => d.fields.get(owner?)?.iter().find(|(n, _)| n == v).map(|(_, t)| t.clone()),
        },
        // `Enum.Variant` without a payload.
        ExprKind::Field(inner, f) if matches!(&inner.kind, ExprKind::Var(en) if !env.contains_key(en) && d.enum_variants.get(en).is_some_and(|vs| vs.contains_key(f))) => {
            let ExprKind::Var(en) = &inner.kind else { return None };
            Some(Type::Named(en.clone()))
        }
        ExprKind::Field(inner, f) => {
            let owner_ty = struct_of(&type_of(inner, env, owner, d)?)?;
            d.fields.get(&owner_ty)?.iter().find(|(n, _)| n == f).map(|(_, t)| t.clone())
        }
        ExprKind::Cast(_, ty) => Some(ty.clone()),
        // `[A(..), B(..)]`: an array of what its first element is.
        ExprKind::Array(elems) => type_of(elems.first()?, env, owner, d).map(|t| Type::Array(Box::new(t))),
        ExprKind::Index(inner, _) | ExprKind::LabeledIndex(inner, _) => {
            indexed_elem_type(&type_of(inner, env, owner, d)?).cloned()
        }
        ExprKind::Call(callee, _) => match &callee.kind {
            ExprKind::Var(n) if d.known(n) => Some(Type::Named(n.clone())),
            // `Enum.Variant(..)`: a variant of a declared enum builds that enum.
            ExprKind::Field(recv, variant) => match &recv.kind {
                ExprKind::Var(en) if !env.contains_key(en) && d.enum_variants.get(en).is_some_and(|vs| vs.contains_key(variant)) => Some(Type::Named(en.clone())),
                _ => None,
            },
            ExprKind::Var(n) => d.fns.get(n)?.clone(),
            _ => None,
        },
        // `Enum.Variant(..)` parsed as a method call on the enum's name.
        ExprKind::MethodCall(recv, variant, _) if matches!(&recv.kind, ExprKind::Var(en) if !env.contains_key(en) && d.enum_variants.get(en).is_some_and(|vs| vs.contains_key(variant))) => {
            let ExprKind::Var(en) = &recv.kind else { return None };
            Some(Type::Named(en.clone()))
        }
        // `value.introspect()`: the reflection handle, whatever `value` is.
        ExprKind::MethodCall(recv, m, args) if m == "introspect" && args.is_empty() => match &recv.kind {
            // `Shape.Unit.introspect()`: a type-level path, not a value the emitter tracks.
            ExprKind::Field(inner, _) if matches!(&inner.kind, ExprKind::Var(n) if d.types.contains(n) && !env.contains_key(n)) => None,
            _ => Some(Type::Named("IntrospectInfo".to_string())),
        },
        ExprKind::MethodCall(recv, m, _) => {
            let o = struct_of(&type_of(recv, env, owner, d)?)?;
            d.methods.get(&(o, m.clone()))?.clone()
        }
        _ => None,
    }
}

/// Binds the names of `pat` matched against a value of type `ty` (when known).
fn bind_pattern(pat: &Pattern, ty: Option<&Type>, env: &mut Env, d: &Decls) {
    match pat {
        Pattern::Bind(n) => bind(env, n, ty.cloned()),
        Pattern::Some(inner) => {
            let inner_ty = match ty.map(|t| t.without_mut()) {
                Some(Type::Optional(i)) => Some((**i).clone()),
                _ => None,
            };
            bind_pattern(inner, inner_ty.as_ref(), env, d);
        }
        Pattern::Tuple(ps) => {
            let tys: Vec<Type> = match ty.map(|t| t.without_mut()) {
                Some(Type::Tuple(ts)) if ts.len() == ps.len() => ts.clone(),
                _ => Vec::new(),
            };
            for (i, p) in ps.iter().enumerate() { bind_pattern(p, tys.get(i), env, d); }
        }
        Pattern::Variant(name, subs) => {
            // `Enum.Variant`, a bare variant of the matched enum, or a variant name only one enum has.
            let bare = name.rsplit('.').next().unwrap_or(name);
            let enum_named = name.rsplit_once('.').map(|(e, _)| e.to_string()).or_else(|| ty.and_then(named));
            let payload = enum_named.and_then(|e| d.enum_variants.get(&e).and_then(|vs| vs.get(bare)).cloned())
                .or_else(|| d.variants.get(bare).cloned().flatten().map(|(_, tys)| tys))
                .unwrap_or_default();
            for (i, p) in subs.iter().enumerate() { bind_pattern(p, payload.get(i), env, d); }
        }
        Pattern::Wildcard | Pattern::Lit(_) | Pattern::None => {}
    }
}

fn bind_clauses(clauses: &[CondClause], env: &mut Env, owner: Option<&str>, d: &Decls) {
    for c in clauses {
        match c {
            CondClause::Let(n, e) => {
                let t = type_of(e, env, owner, d);
                let t = match t { Some(Type::Optional(i)) => Some(*i), other => other };
                bind(env, n, t);
            }
            CondClause::LetPat(p, e) => {
                let t = type_of(e, env, owner, d);
                bind_pattern(p, t.as_ref(), env, d);
            }
            CondClause::Expr(_) => {}
        }
    }
}

fn bind_match(m: &MatchStmt, env: &mut Env, owner: Option<&str>, d: &Decls) {
    let t = type_of(&m.subject, env, owner, d);
    for arm in &m.arms {
        for p in &arm.patterns { bind_pattern(p, t.as_ref(), env, d); }
    }
}

fn walk_body(scope: &str, owner: Option<&str>, params: &[Param], body: &[Stmt], d: &Decls, out: &mut HashMap<Key, Option<String>>, locals: &mut Locals) {
    use super::emit_methods::receiver_type_name;
    let mut env: Env = HashMap::new();
    // Locals declared with an enum type: the emitter does not answer its struct question for them (it does
    // for one built from a value, `let c = Color.Red`), so the walk does not either.
    let mut no_claim: std::collections::HashSet<String> = std::collections::HashSet::new();
    let param_names: std::collections::HashSet<&str> = params.iter().map(|p| p.name.as_str()).collect();
    // Shapes the emitter does not track in its by-name table of struct locals while its by-position question still
    // answers them (a plain cast, an alias of a local): only the table of locals leaves them out.
    let mut by_name_no_claim: std::collections::HashSet<String> = std::collections::HashSet::new();
    let declared_enum = |t: &Type| named(t).is_some_and(|n| d.enums.contains(&n));
    for p in params {
        if p.ty.as_ref().is_some_and(declared_enum) { no_claim.insert(p.name.clone()); }
        bind(&mut env, &p.name, p.ty.clone());
    }
    // Bindings first (flow-insensitive: one table per body), in source order, each value typed with
    // what is bound before it.
    let mut stmts = body.to_vec();
    {
        let env_cell = std::cell::RefCell::new(&mut env);
        let no_claim_cell = std::cell::RefCell::new(&mut no_claim);
        let by_name_no_claim_cell = std::cell::RefCell::new(&mut by_name_no_claim);
        let mut on_expr = |e: &mut Expr| match &e.kind {
            ExprKind::Closure(params, ..) => {
                let mut env = env_cell.borrow_mut();
                // An untyped parameter an array method already bound to the element type (the call is visited first)
                // keeps it; any other untyped one is unknown.
                for p in params {
                    let untyped = p.ty.is_none() || matches!(p.ty, Some(Type::TypeParam(_)));
                    if untyped && env.get(&p.name).is_some_and(|t| t.is_some()) { continue; }
                    bind(&mut env, &p.name, p.ty.clone());
                }
            }
            // `items.map((x): ..)`: the first parameter of a closure given to an array method is an element.
            ExprKind::MethodCall(recv, m, args) if matches!(m.as_str(), "map" | "filter" | "any" | "all" | "forEach" | "each" | "find" | "count" | "flatMap" | "sortBy" | "reduce" | "contains") => {
                let mut env = env_cell.borrow_mut();
                let elem = type_of(recv, &env, owner, d).and_then(|t| super::emit_methods::indexed_elem_type(&t).cloned());
                for a in args {
                    if let ExprKind::Closure(params, ..) = &a.value.kind {
                        if let Some(p) = params.first().filter(|p| p.ty.is_none() || matches!(p.ty, Some(Type::TypeParam(_)))) {
                            bind(&mut env, &p.name, elem.clone());
                        }
                    }
                }
            }
            ExprKind::Match(m) => bind_match(m, &mut env_cell.borrow_mut(), owner, d),
            _ => {}
        };
        let mut on_stmt = |s: &mut Stmt| match s {
            Stmt::Match(m) => bind_match(m, &mut env_cell.borrow_mut(), owner, d),
            Stmt::IfLet(i) => {
                let mut env = env_cell.borrow_mut();
                bind_clauses(&i.clauses, &mut env, owner, d);
                for b in &i.elif_branches { bind_clauses(&b.clauses, &mut env, owner, d); }
                // `if let t = x as T`: a downcast; the emitter does not track the struct of what it binds.
                for c in i.clauses.iter().chain(i.elif_branches.iter().flat_map(|b| b.clauses.iter())) {
                    if let CondClause::Let(n, e) = c {
                        if matches!(e.kind, ExprKind::Cast(..)) { no_claim_cell.borrow_mut().insert(n.clone()); }
                    }
                }
            }
            Stmt::WhileLet(w) => {
                let mut env = env_cell.borrow_mut();
                let t = type_of(&w.value, &env, owner, d);
                match &w.pattern {
                    Some(p) => bind_pattern(p, t.as_ref(), &mut env, d),
                    None => { let t = match t { Some(Type::Optional(i)) => Some(*i), other => other }; bind(&mut env, &w.name, t) }
                }
            }
            Stmt::Guard(g) => {
                if let GuardCond::Clauses(cs) = &g.cond { bind_clauses(cs, &mut env_cell.borrow_mut(), owner, d); }
            }
            Stmt::Let(l) => {
                let mut env = env_cell.borrow_mut();
                let from_value = l.value.as_ref().and_then(|v| type_of(v, &env, owner, d));
                let ty = match &l.ty {
                    // `Visitor v = A()`: the emitter keeps the struct the value is built from.
                    Some(t) if named(t).is_some_and(|n| d.traits.contains(&n)) && from_value.as_ref().and_then(named).is_some_and(|n| d.known(&n)) => from_value,
                    Some(t) => Some(t.clone()),
                    None => from_value,
                };
                if l.ty.as_ref().is_some_and(declared_enum) { no_claim_cell.borrow_mut().insert(l.name.clone()); }
                // `let d = denominator` / `let x = self.field` / `let e = items[0]`: the emitter does not track
                // the struct of a local read out of a field or an element (and what it clones depends on it).
                // `let sa = x as T` (a conversion) and `T c = b` (an alias of a local): not tracked as a struct local.
                if l.value.as_ref().is_some_and(|v| match &v.kind {
                    ExprKind::Cast(..) => true,
                    ExprKind::Var(n) => l.ty.is_some() && n != "self" && env.contains_key(n),
                    _ => false,
                }) { by_name_no_claim_cell.borrow_mut().insert(l.name.clone()); }
                if l.ty.is_none() && l.value.as_ref().is_some_and(|v| match &v.kind {
                    ExprKind::Field(..) | ExprKind::Index(..) | ExprKind::LabeledIndex(..) => true,
                    ExprKind::Var(n) => n != "self" && !env.contains_key(n),
                    _ => false,
                }) { by_name_no_claim_cell.borrow_mut().insert(l.name.clone()); }
                bind(&mut env, &l.name, ty);
            }
            Stmt::For(f) => {
                let mut env = env_cell.borrow_mut();
                let elem = match (f.vars.len(), f.key.is_none()) {
                    (1, true) => type_of(&f.iterable, &env, owner, d)
                        .and_then(|t| super::emit_methods::indexed_elem_type(&t).cloned()),
                    _ => None,
                };
                for (i, v) in f.vars.iter().enumerate() { bind(&mut env, v, if i == 0 { elem.clone() } else { None }); }
            }
            _ => {}
        };
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut stmts);
    }
    // The struct each local of this body is, by name (what the emitter's `var_struct_types` holds): `None`
    // when the name is bound to different things in the body.
    for (name, ty) in &env {
        // A local the emitter keeps behind a lock or a counted handle is tracked by its other tables, not
        // as a plain struct: the written qualifier says so (an inferred one is a decision of
        // `infer_qualifiers`, not visible to a walk over the AST).
        let lock_like = ty.as_ref().is_some_and(|t| {
            let mut cur = t.without_mut();
            let mut found = false;
            while let Type::Qualified(inner, q) = cur {
                // Any written qualifier: such a local is tracked through its qualified `var_types` entry.
                if !matches!(q, OwnerQual::Borrow | OwnerQual::BorrowMut | OwnerQual::BorrowOption | OwnerQual::BorrowOptionMut) { found = true; }
                cur = inner.without_mut();
            }
            found
        });
        let claim = ty.as_ref().and_then(named)
            .filter(|n| d.known(n) && !no_claim.contains(name) && !by_name_no_claim.contains(name) && !lock_like)
            .filter(|n| !d.traits.contains(n) || d.types.contains(n) || param_names.contains(name.as_str()));
        locals.insert((scope.to_string(), name.clone()), claim);
    }
    // Then every variable and field expression gets its answer.
    let mut on_expr = |e: &mut Expr| {
        let what = match &e.kind { ExprKind::Var(n) => format!("v:{n}"), ExprKind::Field(_, n) => format!("f:{n}"), _ => return };
        let _ = receiver_type_name;
        // `Shape.Unit`: a variant path, not a value.
        if let ExprKind::Field(inner, _) = &e.kind {
            if matches!(&inner.kind, ExprKind::Var(n) if d.types.contains(n) && !env.contains_key(n)) { return; }
        }
        if let ExprKind::Var(v) = &e.kind { if no_claim.contains(v) { return; } }
        let name = type_of(e, &env, owner, d).as_ref().and_then(named)
            .filter(|n| matches!(e.kind, ExprKind::Field(..)) || d.known(n))
            // A trait name answers for a parameter declared with it; a loop element or a local of a trait
            // type is not a struct to the emitter.
            .filter(|n| match &e.kind {
                ExprKind::Var(v) if d.traits.contains(n) && !d.types.contains(n) => param_names.contains(v.as_str()),
                _ => true,
            });
        if let Some(name) = &name {
            out.insert((scope.to_string(), e.line, e.col, what), Some(name.clone()));
        }
        // A bare field name in a method is `self.<name>`: the emitter asks about that field access, an
        // expression it builds on the same token.
        if let (ExprKind::Var(v), Some(name)) = (&e.kind, &name) {
            if !env.contains_key(v) && owner.is_some_and(|o| d.fields.get(o).is_some_and(|fs| fs.iter().any(|(n, _)| n == v))) {
                out.insert((scope.to_string(), e.line, e.col, format!("f:{v}")), Some(name.clone()));
            }
        }
    };
    let mut on_stmt = |_: &mut Stmt| {};
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut stmts);
}

/// The struct-name answers a standalone walk over `program` can give, keyed like the recorded ones.
/// It makes a claim only when declarations say it: an annotated parameter or `let`, a constructor, a
/// declared return type, `self`, a field of a resolved struct, an element of a typed collection.
pub(crate) fn build_struct_names(program: &Program, imported: &[Item]) -> (HashMap<Key, Option<String>>, Locals) {
    let mut d = Decls { fields: HashMap::new(), types: Default::default(), methods: HashMap::new(), fns: HashMap::new(), variants: HashMap::new(), generics: Default::default(), traits: Default::default(), enums: Default::default(), enum_variants: HashMap::new() };
    seed_builtins(&mut d);
    collect_decls(imported, &mut d);
    // What the file itself declares comes last, so it wins over an import of the same name.
    collect_decls(&program.items, &mut d);
    let mut out = HashMap::new();
    let mut locals: Locals = HashMap::new();
    fn go(items: &[Item], d: &Decls, out: &mut HashMap<Key, Option<String>>, locals: &mut Locals) {
        for item in items {
            match item {
                Item::Fn(f) if f.qualifier.is_none() => walk_body(&format!("::{}", f.name), None, &f.params, &f.body, d, out, locals),
                Item::Struct(s) => {
                    for m in &s.methods { walk_body(&format!("{}::{}", s.name, m.name), Some(&s.name), &m.params, &m.body, d, out, locals) }
                    // Inits, setters and conversions are not emitted through `emit_fn`: their scope is the owner's.
                    let scope = format!("{}::", s.name);
                    for i in &s.inits {
                        let params: Vec<Param> = i.params.iter().map(|p| Param {
                            name: p.name.clone(), ty: p.ty.clone(), mutable: false, rebindable: false, var_mut: false,
                            owned: false, variadic: false, default: None, line: 0, col: 0,
                        }).collect();
                        walk_body(&scope, Some(&s.name), &params, &i.body, d, out, locals);
                    }
                    for st in &s.setters {
                        let p = Param { name: st.param_name.clone(), ty: Some(st.param_ty.clone()), mutable: false, rebindable: false, var_mut: false, owned: false, variadic: false, default: None, line: 0, col: 0 };
                        walk_body(&scope, Some(&s.name), &[p], &st.body, d, out, locals);
                    }
                    for c in &s.conversions { walk_body(&scope, Some(&s.name), &[], &c.body, d, out, locals); }
                }
                Item::Trait(t) => for m in &t.defaults { walk_body(&format!("::{}", m.name), Some(&t.name), &m.params, &m.body, d, out, locals) },
                Item::Enum(e) => for m in &e.methods { walk_body(&format!("{}::{}", e.name, m.name), Some(&e.name), &m.params, &m.body, d, out, locals) },
                Item::Ext(e) => for m in &e.methods { walk_body(&format!("{}::{}", e.type_name, m.name), Some(&e.type_name), &m.params, &m.body, d, out, locals) },
                Item::Mod(m) => go(&m.items, d, out, locals),
                _ => {}
            }
        }
    }
    go(&program.items, &d, &mut out, &mut locals);
    // The file's own statements are one body (scope ""), the top-level `let`s its locals.
    let top: Vec<Stmt> = program.items.iter().filter_map(|i| match i {
        Item::Stmt(s) => Some(s.clone()),
        Item::Let(l) => Some(Stmt::Let(l.clone())),
        _ => None,
    }).collect();
    if !top.is_empty() { walk_body("", None, &[], &top, &d, &mut out, &mut locals); }
    (out, locals)
}

/// How a standalone table compares with the recorded one.
#[derive(Default, Debug)]
pub(crate) struct Comparison {
    pub agree: usize,
    pub missing: usize,
    pub disagree: usize,
    pub over_claim: usize,
    pub undecided: usize,
    pub by_rule: usize,
}

pub(crate) fn compare(recorded: &TypeTable, built: &HashMap<Key, Option<String>>) -> Comparison {
    let mut c = Comparison::default();
    for (key, answer) in &recorded.struct_names {
        let Ok(rec) = answer else { c.undecided += 1; continue };
        // `self` of a method is its owner: asked about an expression the emitter builds itself while
        // emitting (an implicit `self.field`), so it is not in any AST to be found by a walk.
        if key.3 == "v:self" && rec.as_deref() == key.0.split("::").next().filter(|o| !o.is_empty()) {
            c.by_rule += 1;
            continue;
        }
        match (rec, built.get(key)) {
            (Some(r), Some(Some(b))) if r == b => c.agree += 1,
            (Some(r), Some(Some(b))) => { c.disagree += 1; if std::env::var_os("BORING_TYPE_TABLE_WHY").is_some() { eprintln!("DISAGREE {:?}: emitter {r} built {b}", key); } }
            (Some(r), _) => { c.missing += 1; if std::env::var_os("BORING_TYPE_TABLE_WHY").is_some() { eprintln!("MISSING {:?}: emitter {r}", key); } }
            (None, Some(Some(b))) => { c.over_claim += 1; if std::env::var_os("BORING_TYPE_TABLE_WHY").is_some() { eprintln!("OVERCLAIM {:?}: emitter none built {b}", key); } }
            (None, _) => c.agree += 1,
        }
    }
    c
}

// ─── The emitter's by-name reads of the table of locals ─────────────────────────────────────

impl super::Transpiler {
    /// The struct (or enum) a local is, by name: what `var_struct_types` holds. In `walk` mode the walk's
    /// table of locals answers where it names the local (the body binds it once); `check` mode asks the
    /// emitter and logs a difference. Every read of the emitter's table by variable name goes through here.
    pub(crate) fn local_struct(&self, v: &str) -> Option<String> {
        use TypeTableMode::*;
        let own = || self.var_struct_types.get(v).cloned();
        // The type is the walk's, the representation is a decision (`infer_qualifiers`): a local decided to
        // live behind a lock or a counted handle is tracked by the other tables, not as a plain struct.
        let decided_lock = || matches!(self.inferred_qualifiers.get(v),
            Some(OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Atomic | OwnerQual::Shared | OwnerQual::ActorTask | OwnerQual::GuardTask));
        let walked = || -> Option<String> {
            if decided_lock() { return None; }
            self.built_locals.borrow().get(&(self.type_scope.clone(), v.to_string())).cloned().flatten()
        };
        match self.type_table_mode {
            Off | Record | Replay => own(),
            Walk => match walked() {
                Some(n) => { self.type_table.borrow_mut().local_hits += 1; Some(n) }
                None => { self.type_table.borrow_mut().local_misses += 1; own() }
            },
            Check => {
                let answer = own();
                if let Some(claim) = walked() {
                    if answer.as_deref() != Some(claim.as_str()) {
                        self.type_table.borrow_mut().local_conflicts += 1;
                        if let Some(path) = std::env::var_os("BORING_TYPE_TABLE_LOG") {
                            use std::io::Write;
                            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                                let _ = writeln!(f, "LOCALMISMATCH ({:?}, {:?}) walk {:?} emitter {:?}", self.type_scope, v, claim, answer);
                            }
                        }
                    } else {
                        self.type_table.borrow_mut().local_hits += 1;
                    }
                }
                answer
            }
        }
    }

    /// Whether the emitter tracks `v` as a struct local (`var_struct_types.contains_key`).
    pub(crate) fn is_struct_local(&self, v: &str) -> bool {
        match self.type_table_mode {
            TypeTableMode::Walk | TypeTableMode::Check => self.local_struct(v).is_some(),
            _ => self.var_struct_types.contains_key(v),
        }
    }
}

impl super::Transpiler {
    /// The qualifier `infer_qualifiers` decided for a local or parameter of the body being emitted. Every read
    /// of the emitter's per-body `inferred_qualifiers` goes through here; in `walk` mode the table filled by
    /// the speculative runs before emission answers where it has the name, `check` mode logs a difference.
    pub(crate) fn decided_qual(&self, v: &str) -> Option<OwnerQual> {
        use TypeTableMode::*;
        let own = || self.inferred_qualifiers.get(v).cloned();
        // `Inline` is the default: the emitter's map holds no entry for it (and acts differently on an
        // explicit one: an annotated `let`), so only a decided non-default qualifier is read from the table.
        let tabled = || self.decision_table.borrow().get(&(self.type_scope.clone(), v.to_string())).cloned()
            .filter(|q| !matches!(q, OwnerQual::Inline));
        match self.type_table_mode {
            Off | Record | Replay => own(),
            Walk => match tabled() {
                Some(q) => { self.type_table.borrow_mut().decision_hits += 1; Some(q) }
                None => own(),
            },
            Check => {
                let answer = own();
                if let Some(q) = tabled() {
                    if answer.as_ref() != Some(&q) {
                        self.type_table.borrow_mut().decision_conflicts += 1;
                        if let Some(path) = std::env::var_os("BORING_TYPE_TABLE_LOG") {
                            use std::io::Write;
                            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                                let _ = writeln!(f, "DECISIONMISMATCH ({:?}, {:?}) table {:?} emitter {:?}", self.type_scope, v, q, answer);
                            }
                        }
                    } else {
                        self.type_table.borrow_mut().decision_hits += 1;
                    }
                }
                answer
            }
        }
    }
}

// ─── `.bri`: patterns follow the instance of their subject ───────────────────────────────────

/// After the monomorphizer, `match slot: Slot.Full(p)` still names the generic enum it dropped while `slot` is a
/// `Slot_actor`: the transpiler reads the instance off the subject's type, a reader of the `.bri` cannot. Each
/// variant pattern of a subject whose type is an instance (`G_..`) of the enum the pattern names is renamed to the
/// instance (`Slot_actor.Full(p)`). `generics` are the names of the enums the monomorphizer dropped.
pub(crate) fn rewrite_variant_patterns(program: &mut Program, generics: &std::collections::HashSet<String>) {
    if generics.is_empty() { return; }
    let mut d = Decls { fields: HashMap::new(), types: Default::default(), methods: HashMap::new(), fns: HashMap::new(), variants: HashMap::new(), generics: Default::default(), traits: Default::default(), enums: Default::default(), enum_variants: HashMap::new() };
    seed_builtins(&mut d);
    collect_decls(&program.items, &mut d);
    fn instance_of(name: &str, generics: &std::collections::HashSet<String>) -> Option<String> {
        generics.iter().filter(|g| name.len() > g.len() && name.starts_with(g.as_str()) && name.as_bytes()[g.len()] == b'_')
            .max_by_key(|g| g.len()).cloned()
    }
    fn rename(p: &mut Pattern, instance: &str, base: &str) {
        match p {
            Pattern::Variant(name, subs) => {
                // `Slot::Full`, `Slot<inline>::Full` (the parser's spelling) or the bare variant.
                let (prefix, bare) = match name.rsplit_once("::") { Some((p, b)) => (Some(p.to_string()), b.to_string()), None => (None, name.clone()) };
                let prefix_base = prefix.as_deref().map(|p| p.split('<').next().unwrap_or(p).to_string());
                if prefix_base.as_deref().is_none_or(|e| e == base) { *name = format!("{instance}::{bare}"); }
                for s in subs { rename(s, instance, base); }
            }
            Pattern::Some(i) => rename(i, instance, base),
            Pattern::Tuple(ps) => for s in ps { rename(s, instance, base); },
            _ => {}
        }
    }
    fn body(params: &[Param], stmts: &mut [Stmt], owner: Option<&str>, d: &Decls, generics: &std::collections::HashSet<String>) {
        let mut env: Env = HashMap::new();
        for p in params { bind(&mut env, &p.name, p.ty.clone()); }
        // Bindings in source order, then the matches (flow-insensitive, like the walk).
        let mut snapshot = stmts.to_vec();
        {
            let env_cell = std::cell::RefCell::new(&mut env);
            let mut on_expr = |_: &mut Expr| {};
            let mut on_stmt = |s: &mut Stmt| match s {
                Stmt::Let(l) => {
                    let mut env = env_cell.borrow_mut();
                    let ty = l.ty.clone().or_else(|| l.value.as_ref().and_then(|v| type_of(v, &env, owner, d)));
                    bind(&mut env, &l.name, ty);
                }
                Stmt::For(f) => {
                    let mut env = env_cell.borrow_mut();
                    let elem = if f.vars.len() == 1 && f.key.is_none() {
                        type_of(&f.iterable, &env, owner, d).and_then(|t| super::emit_methods::indexed_elem_type(&t).cloned())
                    } else { None };
                    for (i, v) in f.vars.iter().enumerate() { bind(&mut env, v, if i == 0 { elem.clone() } else { None }); }
                }
                _ => {}
            };
            Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut snapshot);
        }
        let mut on_expr = |_: &mut Expr| {};
        let mut on_stmt = |s: &mut Stmt| {
            if let Stmt::Match(m) = s {
                let Some(inst) = type_of(&m.subject, &env, owner, d).as_ref().and_then(named) else { return };
                let Some(base) = instance_of(&inst, generics) else { return };
                for arm in &mut m.arms { for p in &mut arm.patterns { rename(p, &inst, &base); } }
            }
        };
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(stmts);
    }
    fn go(items: &mut [Item], d: &Decls, generics: &std::collections::HashSet<String>) {
        for item in items.iter_mut() {
            match item {
                Item::Fn(f) if f.qualifier.is_none() => body(&f.params, &mut f.body, None, d, generics),
                Item::Struct(s) => { let o = s.name.clone(); for m in &mut s.methods { body(&m.params, &mut m.body, Some(&o), d, generics) } }
                Item::Enum(e) => { let o = e.name.clone(); for m in &mut e.methods { body(&m.params, &mut m.body, Some(&o), d, generics) } }
                Item::Ext(e) => { let o = e.type_name.clone(); for m in &mut e.methods { body(&m.params, &mut m.body, Some(&o), d, generics) } }
                Item::Mod(m) => go(&mut m.items, d, generics),
                _ => {}
            }
        }
    }
    go(&mut program.items, &d, generics);
    // The file's own statements.
    let mut top: Vec<Stmt> = program.items.iter().filter_map(|i| match i { Item::Stmt(s) => Some(s.clone()), Item::Let(l) => Some(Stmt::Let(l.clone())), _ => None }).collect();
    body(&[], &mut top, None, &d, generics);
    let mut it = top.into_iter();
    for item in program.items.iter_mut() {
        match item {
            Item::Stmt(dst) => if let Some(s) = it.next() { *dst = s },
            Item::Let(dst) => if let Some(Stmt::Let(l)) = it.next() { *dst = l },
            _ => {}
        }
    }
}
