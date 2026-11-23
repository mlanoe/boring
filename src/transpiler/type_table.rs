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
    /// `var_types` reads (`local_type`): the walk's declared type agrees / differs / is the only one.
    pub vt_hits: usize,
    pub vt_conflicts: usize,
    pub vt_walk_only: usize,
    /// Same type, different representation (a decision the walk does not make).
    pub vt_repr: usize,
    /// `var_scalar_types` reads (`local_scalar_type`).
    pub sc_hits: usize,
    pub sc_conflicts: usize,
    pub sc_walk_only: usize,
    /// `var_std_types` reads (`local_std_type`), and how many found an entry.
    pub std_reads: usize,
    pub std_reads_known: usize,
    pub std_hits: usize,
    /// `var_init_exprs` reads that found an entry (`local_init_expr`): agreeing with the walk, differing, the body binding
    /// the name more than once, the body not binding it at all.
    pub init_reads: usize,
    pub init_hits: usize,
    pub init_conflicts: usize,
    pub init_unsure: usize,
    pub init_stray: usize,
    /// Last-use sites per function: table agrees / differs with the emitter's own, and answers given by the table.
    pub lu_hits: usize,
    pub lu_conflicts: usize,
    pub lu_table_answers: usize,
    /// `is_last_use_move`'s representation test: emitter's sets agree / differ with the decisions.
    pub rep_hits: usize,
    pub rep_conflicts: usize,
    /// Walk mode: representation tests answered from the decisions.
    pub rep_decided: usize,
    /// Reads where the representation forbade the move (an actor, a lock, a parameter): the edge the test must cover.
    pub rep_blocked: usize,
    /// `var_mutex_types` reads (`is_mutex_local`): the emitter's answer equals / differs from the decisions'.
    pub lock_hits: usize,
    pub lock_conflicts: usize,
    /// `var_rwlock_types` reads (`is_rwlock_local`).
    pub rw_hits: usize,
    pub rw_conflicts: usize,
    /// Reads of either lock set that found the local held: the side of the check that is not vacuous.
    pub lock_held: usize,
    pub rw_held: usize,
    /// Walk mode: stale initialisers the walk withheld.
    pub init_dropped: usize,
    pub std_conflicts: usize,
    pub std_walk_only: usize,
    /// Walk mode: declared types of top-level locals the walk answered.
    pub vt_walk_answers: usize,
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

/// `(scope, local name)` -> declared type of the local (`None`: bound more than once to different things):
/// what the emitter's `var_types` holds, as the walk sees it.
pub(crate) type LocalTypes = HashMap<(String, String), Option<Type>>;

/// `(scope, name)` -> position of the initialiser of the `let` that binds the name (`None`: bound more than once, or by
/// something other than a `let` with a value). An absent entry is a name the body does not bind at all.
/// One binding event of a name in a body, by position: where it starts, where its region ends (`None`: the rest of the
/// body), the type it gives the name, and whether it is a `with` (the name stands for the unlocked value there).
#[derive(Clone, Debug)]
pub(crate) struct Binder {
    pub pos: (usize, usize),
    pub end: Option<(usize, usize)>,
    pub ty: Option<Type>,
    pub unlocked: bool,
}

/// `(scope, name)` -> the binding events of the name in the body, in source order.
pub(crate) type Binders = HashMap<(String, String), Vec<Binder>>;

pub(crate) type LocalInits = HashMap<(String, String), Option<(usize, usize)>>;

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
        // `dict.get(k)` of a `{K=V}`: `V?`. A dict's own members come before a user method of the same name.
        ExprKind::MethodCall(recv, m, args) if m == "get" && args.len() == 1
            && matches!(type_of(recv, env, owner, d).as_ref().map(|t| t.without_mut()), Some(Type::Dict(..))) => {
            match type_of(recv, env, owner, d)?.without_mut() {
                Type::Dict(_, v) => Some(Type::Optional(v.clone())),
                _ => None,
            }
        }
        ExprKind::OptionalMethodCall(recv, m, args) if m == "get" && args.len() == 1 => {
            let inner_opt = type_of(recv, env, owner, d)?;
            let inner_ty = match inner_opt.without_mut() {
                Type::Optional(inner) => inner.without_mut(),
                other => other,
            };
            match inner_ty {
                Type::Dict(_, v) => Some(Type::Optional(v.clone())),
                _ => None,
            }
        }
        ExprKind::OptionalMethodCall(recv, m, _) => {
            let inner_opt = type_of(recv, env, owner, d)?;
            let inner_ty = match inner_opt.without_mut() {
                Type::Optional(inner) => inner.without_mut(),
                other => other,
            };
            let o = struct_of(inner_ty)?;
            let ret = d.methods.get(&(o, m.clone()))?.clone()?;
            match ret {
                Type::Optional(_) => Some(ret),
                other => Some(Type::Optional(Box::new(other))),
            }
        }
        ExprKind::OptionalField(recv, f) => {
            let inner_opt = type_of(recv, env, owner, d)?;
            let inner_ty = match inner_opt.without_mut() {
                Type::Optional(inner) => inner.without_mut(),
                other => other,
            };
            let o = struct_of(inner_ty)?;
            let field_ty = d.fields.get(&o)?.iter().find(|(name, _)| name == f).map(|(_, t)| t.clone())?;
            match field_ty {
                Type::Optional(_) => Some(field_ty),
                other => Some(Type::Optional(Box::new(other))),
            }
        }
        ExprKind::MethodCall(recv, m, _) => {
            let o = struct_of(&type_of(recv, env, owner, d)?)?;
            d.methods.get(&(o, m.clone()))?.clone()
        }
        _ => None,
    }
}

/// The std type of an expression by the emitter's own table of std returns (`infer_std_type`): a `try?` over a std
/// call, a call on a std type (a declared local, an earlier std binding, or a bare std type name), a tuple element or
/// a `Child` stream; unknown as soon as a link is. A Boring struct, enum or trait of the same name makes it the
/// user's own, so it is no std type.
fn std_type_of(e: &Expr, std_env: &Env, env: &Env, annotated: &std::collections::HashSet<String>, d: &Decls) -> Option<Type> {
    let ty = match &e.kind {
        ExprKind::Var(v) => {
            let t = if annotated.contains(v) { env.get(v)?.clone()? } else { std_env.get(v)?.clone()? };
            t.without_mut().clone()
        }
        ExprKind::TryElse(inner, dflt) if matches!(dflt.kind, ExprKind::Nil) => {
            Type::Optional(Box::new(std_type_of(inner, std_env, env, annotated, d)?))
        }
        ExprKind::MethodCall(recv, method, _) => {
            let recv_ty = match &recv.kind {
                ExprKind::Var(v) if !env.contains_key(v) && !std_env.contains_key(v)
                    && v.chars().next().is_some_and(|c| c.is_ascii_uppercase()) => v.clone(),
                _ => match std_type_of(recv, std_env, env, annotated, d)? { Type::Named(n) => n, _ => return None },
            };
            if d.known(&recv_ty) { return None; }
            super::Transpiler::std_call_return_type(&recv_ty, method)?
        }
        ExprKind::Field(obj, field) => match std_type_of(obj, std_env, env, annotated, d)? {
            Type::Tuple(elems) => elems.get(field.parse::<usize>().ok()?).cloned()?,
            Type::Named(n) if n == "Child" => match field.as_str() {
                "stdin" => Type::Optional(Box::new(Type::Named("ChildStdin".into()))),
                "stdout" => Type::Optional(Box::new(Type::Named("ChildStdout".into()))),
                "stderr" => Type::Optional(Box::new(Type::Named("ChildStderr".into()))),
                _ => return None,
            },
            _ => return None,
        },
        _ => return None,
    };
    let ok = match ty.without_mut() { Type::Named(n) | Type::Generic(n, _) => !d.known(n), _ => true };
    ok.then_some(ty)
}

/// `Enum.Variant`, `Enum::Variant`, `Enum<elem'actor>.Variant` or a bare `Variant`: the enum (generic arguments dropped) and
/// the variant.
fn split_variant(name: &str) -> (Option<String>, String) {
    let mut flat = String::new();
    let mut depth = 0usize;
    for c in name.chars() {
        match c { '<' => depth += 1, '>' => depth = depth.saturating_sub(1), _ if depth == 0 => flat.push(c), _ => {} }
    }
    let flat = flat.replace("::", ".");
    match flat.rsplit_once('.') {
        Some((e, v)) => (Some(e.to_string()), v.to_string()),
        None => (None, flat),
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
            let (prefix, bare) = split_variant(name);
            let bare = bare.as_str();
            let enum_named = prefix.or_else(|| ty.and_then(named));
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

fn walk_body(scope: &str, owner: Option<&str>, params: &[Param], body: &[Stmt], d: &Decls, tables: &mut WalkTables) {
    let WalkTables { out, locals, types, declared, scalar, std, inits, binders } = tables;
    use super::emit_methods::receiver_type_name;
    let mut env: Env = HashMap::new();
    // Locals declared with an enum type: the emitter does not answer its struct question for them (it does
    // for one built from a value, `let c = Color.Red`), so the walk does not either.
    let mut no_claim: std::collections::HashSet<String> = std::collections::HashSet::new();
    let param_names: std::collections::HashSet<&str> = params.iter().map(|p| p.name.as_str()).collect();
    // Shapes the emitter does not track in its by-name table of struct locals while its by-position question still
    // answers them (a plain cast, an alias of a local): only the table of locals leaves them out.
    let mut by_name_no_claim: std::collections::HashSet<String> = std::collections::HashSet::new();
    // `Visitor v = A()`: the struct question answers the struct built, the declared type stays the trait.
    // Overload-resolution type of an unannotated local bound from a literal or a built-in method result, or of a
    // loop variable over a typed array (the emitter's `var_scalar_types`).
    let mut scalar_env: Env = HashMap::new();
    // Std type of a local bound from a std call chain (`var_std_types`).
    let mut std_env: Env = HashMap::new();
    // Position of the initialiser of each `let` (the emitter's `var_init_exprs`, which keeps the latest one by name).
    let mut init_pos: HashMap<String, Option<(usize, usize)>> = HashMap::new();
    let mut declared_trait: HashMap<String, Type> = HashMap::new();
    // Names bound with a written type (parameters and annotated `let`s): the only locals the emitter keeps a declared type for.
    let mut annotated: std::collections::HashSet<String> = std::collections::HashSet::new();
    for p in params { if p.ty.is_some() { annotated.insert(p.name.clone()); } }
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
        let declared_trait_cell = std::cell::RefCell::new(&mut declared_trait);
        let annotated_cell = std::cell::RefCell::new(&mut annotated);
        let scalar_cell = std::cell::RefCell::new(&mut scalar_env);
        let std_cell = std::cell::RefCell::new(&mut std_env);
        let init_cell = std::cell::RefCell::new(&mut init_pos);
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
                if let GuardCond::Clauses(cs) = &g.cond {
                    bind_clauses(cs, &mut env_cell.borrow_mut(), owner, d);
                    for c in cs {
                        if let CondClause::Let(n, e) = c {
                            let st = match std_type_of(e, &std_cell.borrow(), &env_cell.borrow(), &annotated_cell.borrow(), d) { Some(Type::Optional(i)) => Some(*i), _ => None };
                            bind(&mut std_cell.borrow_mut(), n, st);
                        }
                    }
                }
            }
            Stmt::Let(l) => {
                let mut env = env_cell.borrow_mut();
                let from_value = l.value.as_ref().and_then(|v| type_of(v, &env, owner, d));
                let ty = match &l.ty {
                    // `Visitor v = A()`: the emitter keeps the struct the value is built from.
                    Some(t) if named(t).is_some_and(|n| d.traits.contains(&n)) && from_value.as_ref().and_then(named).is_some_and(|n| d.known(&n)) => {
                        declared_trait_cell.borrow_mut().insert(l.name.clone(), t.clone());
                        from_value
                    }
                    Some(t) => Some(t.clone()),
                    None => from_value,
                };
                if l.ty.is_some() { annotated_cell.borrow_mut().insert(l.name.clone()); }
                {
                    let pos = l.value.as_ref().map(|v| (v.line, v.col));
                    let mut ip = init_cell.borrow_mut();
                    match ip.get(&l.name) {
                        Some(prev) if *prev != pos => { ip.insert(l.name.clone(), None); }
                        Some(_) => {}
                        None => { ip.insert(l.name.clone(), pos); }
                    }
                }
                if let Some(v) = &l.value {
                    let st = std_type_of(v, &std_cell.borrow(), &env, &annotated_cell.borrow(), d);
                    bind(&mut std_cell.borrow_mut(), &l.name, st);
                } else { bind(&mut std_cell.borrow_mut(), &l.name, None); }
                {
                    use super::helpers::{builtin_method_result_type, infer_overload_expr_type};
                    let st = if l.ty.is_some() { None } else { l.value.as_ref().and_then(|v| match &v.kind {
                        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) | ExprKind::StringInterp(_) => {
                            let (no_vars, no_fields) = (HashMap::new(), HashMap::new());
                            infer_overload_expr_type(v, &no_vars, &no_vars, &no_fields)
                        }
                        ExprKind::MethodCall(recv, m, _) => type_of(recv, &env, owner, d).and_then(|rt| builtin_method_result_type(&rt, m)),
                        _ => None,
                    }) };
                    bind(&mut scalar_cell.borrow_mut(), &l.name, st);
                }
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
            Stmt::LetDestructure(l) => {
                let slots: Option<Vec<(usize, usize)>> = match &l.value.kind {
                    ExprKind::Tuple(es) if es.len() == l.bindings.len() => Some(es.iter().map(|e| (e.line, e.col)).collect()),
                    _ => None,
                };
                let mut env = env_cell.borrow_mut();
                for (i, b) in l.bindings.iter().enumerate() {
                    if b.name == "_" { continue; }
                    bind(&mut env, &b.name, None);
                    let pos = slots.as_ref().map(|s| s[i]);
                    let mut ip = init_cell.borrow_mut();
                    match ip.get(&b.name) {
                        Some(prev) if *prev != pos => { ip.insert(b.name.clone(), None); }
                        Some(_) => {}
                        None => { ip.insert(b.name.clone(), pos); }
                    }
                }
            }
            Stmt::For(f) => {
                let mut env = env_cell.borrow_mut();
                let elem = match (f.vars.len(), f.key.is_none()) {
                    (1, true) => type_of(&f.iterable, &env, owner, d)
                        .and_then(|t| super::emit_methods::indexed_elem_type(&t).cloned()),
                    _ => None,
                };
                for (i, v) in f.vars.iter().enumerate() { bind(&mut env, v, if i == 0 { elem.clone() } else { None }); }
                {
                    let st = if f.vars.len() == 1 { type_of(&f.iterable, &env, owner, d).and_then(|t| super::helpers::loop_element_type(&t)) } else { None };
                    let mut sc = scalar_cell.borrow_mut();
                    for (i, v) in f.vars.iter().enumerate() { bind(&mut sc, v, if i == 0 { st.clone() } else { None }); }
                }
            }
            _ => {}
        };
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut stmts);
    }
    // Binding events by position (for the lock check): `let`s, the names of a match arm's patterns for the arm's region,
    // and `with` names (unlocked inside the body). The environment is the finished one of this body.
    {
        fn max_pos(stmts: &[Stmt], exprs: &[&Expr]) -> (usize, usize) {
            let best = std::cell::Cell::new((0usize, 0usize));
            let mut on_expr = |e: &mut Expr| { if (e.line, e.col) > best.get() { best.set((e.line, e.col)); } };
            let mut on_stmt = |s: &mut Stmt| { let p = s.line_col(); if p > best.get() { best.set(p); } };
            let mut w = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
            let mut owned = stmts.to_vec();
            w.stmts(&mut owned);
            for e in exprs { let mut c = (*e).clone(); w.expr(&mut c); }
            best.get()
        }
        let mut found: Vec<(String, Binder)> = Vec::new();
        for p in params {
            found.push((p.name.clone(), Binder { pos: (0, 0), end: None, ty: p.ty.clone(), unlocked: false }));
        }
        let mut snap = body.to_vec();
        {
            let found_cell = std::cell::RefCell::new(&mut found);
            let arms = |m: &MatchStmt, found: &mut Vec<(String, Binder)>| {
                let subject = type_of(&m.subject, &env, owner, d);
                for arm in &m.arms {
                    let mut tmp: Env = HashMap::new();
                    for pat in &arm.patterns { bind_pattern(pat, subject.as_ref(), &mut tmp, d); }
                    let end = match &arm.body {
                        MatchBody::Expr(e) => max_pos(&[], &[e]),
                        MatchBody::Block(b) => max_pos(b, &[]),
                    };
                    for (name, ty) in tmp {
                        found.push((name, Binder { pos: (arm.line, arm.col), end: Some(end), ty, unlocked: false }));
                    }
                }
            };
            let mut on_expr = |e: &mut Expr| { if let ExprKind::Match(m) = &e.kind { arms(m, &mut found_cell.borrow_mut()); } };
            let mut on_stmt = |s: &mut Stmt| match s {
                Stmt::Match(m) => arms(m, &mut found_cell.borrow_mut()),
                Stmt::Let(l) => {
                    let ty = l.ty.clone().or_else(|| l.value.as_ref().and_then(|v| type_of(v, &env, owner, d)));
                    found_cell.borrow_mut().push((l.name.clone(), Binder { pos: (l.line, l.col), end: None, ty, unlocked: false }));
                }
                Stmt::With(w) => {
                    let end = max_pos(&w.body, &[]);
                    for n in &w.names {
                        // The header asks the lock state at the statement's own position: the body starts after it.
                        found_cell.borrow_mut().push((n.clone(), Binder { pos: (w.line, w.col + 1), end: Some(end), ty: None, unlocked: true }));
                    }
                }
                _ => {}
            };
            Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut snap);
        }
        for (name, b) in found {
            binders.entry((scope.to_string(), name)).or_default().push(b);
        }
    }
    // The body was walked: a name it does not bind is known not to be bound (the sentinel `""`).
    inits.entry((scope.to_string(), String::new())).or_insert(None);
    // Every name the body binds; its initialiser when one `let` with a value binds it.
    for name in env.keys() {
        let pos = init_pos.get(name).copied().flatten();
        let key = (scope.to_string(), name.clone());
        match inits.get(&key) {
            Some(prev) if *prev != pos => { inits.insert(key, None); }
            Some(_) => {}
            None => { inits.insert(key, pos); }
        }
    }
    for (name, ty) in &std_env {
        if ty.is_none() { continue; }
        let key = (scope.to_string(), name.clone());
        match std.get(&key) {
            Some(prev) if prev != ty => { std.insert(key, None); }
            Some(_) => {}
            None => { std.insert(key, ty.clone()); }
        }
    }
    for (name, ty) in &scalar_env {
        if ty.is_none() { continue; }
        let key = (scope.to_string(), name.clone());
        match scalar.get(&key) {
            Some(prev) if prev != ty => { scalar.insert(key, None); }
            Some(_) => {}
            None => { scalar.insert(key, ty.clone()); }
        }
    }
    // The struct each local of this body is, by name (what the emitter's `var_struct_types` holds): `None`
    // when the name is bound to different things in the body.
    for (name, ty) in &env {
        // A name two bodies share (overloads, which have one scope) with different types is bound to none.
        let ty_here = match (ty, declared_trait.get(name)) { (Some(_), Some(t)) => Some(t.clone()), _ => ty.clone() };
        let key = (scope.to_string(), name.clone());
        let put = |m: &mut LocalTypes| match m.get(&key) {
            Some(prev) if *prev != ty_here => { m.insert(key.clone(), None); }
            Some(_) => {}
            None => { m.insert(key.clone(), ty_here.clone()); }
        };
        put(types);
        if annotated.contains(name) { put(declared); }
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

/// What one walk over a program produces, one table per question the transpiler asks.
#[derive(Default)]
pub(crate) struct WalkTables {
    /// Struct type of each position (`Key`), `None` where the walk claims it has none.
    pub(crate) out: HashMap<Key, Option<String>>,
    pub(crate) locals: Locals,
    pub(crate) types: LocalTypes,
    pub(crate) declared: LocalTypes,
    pub(crate) scalar: LocalTypes,
    pub(crate) std: LocalTypes,
    pub(crate) inits: LocalInits,
    pub(crate) binders: Binders,
}

/// The struct-name answers a standalone walk over `program` can give, keyed like the recorded ones.
/// It makes a claim only when declarations say it: an annotated parameter or `let`, a constructor, a
/// declared return type, `self`, a field of a resolved struct, an element of a typed collection.
pub(crate) fn build_struct_names(program: &Program, imported: &[Item]) -> WalkTables {
    let mut d = Decls { fields: HashMap::new(), types: Default::default(), methods: HashMap::new(), fns: HashMap::new(), variants: HashMap::new(), generics: Default::default(), traits: Default::default(), enums: Default::default(), enum_variants: HashMap::new() };
    seed_builtins(&mut d);
    collect_decls(imported, &mut d);
    // What the file itself declares comes last, so it wins over an import of the same name.
    collect_decls(&program.items, &mut d);
    let mut tables = WalkTables::default();
    fn go(items: &[Item], d: &Decls, tables: &mut WalkTables) {
        for item in items {
            match item {
                Item::Fn(f) if f.qualifier.is_none() => walk_body(&format!("::{}", f.name), None, &f.params, &f.body, d, tables),
                Item::Struct(s) => {
                    for m in &s.methods { walk_body(&format!("{}::{}", s.name, m.name), Some(&s.name), &m.params, &m.body, d, tables) }
                    // Inits, setters and conversions are not emitted through `emit_fn`: their scope is the owner's.
                    let scope = format!("{}::", s.name);
                    for i in &s.inits {
                        let params: Vec<Param> = i.params.iter().map(|p| Param {
                            name: p.name.clone(), ty: p.ty.clone(), mutable: false, rebindable: false, var_mut: false,
                            owned: false, variadic: false, default: None, line: 0, col: 0,
                        }).collect();
                        walk_body(&scope, Some(&s.name), &params, &i.body, d, tables);
                    }
                    for st in &s.setters {
                        let p = Param { name: st.param_name.clone(), ty: Some(st.param_ty.clone()), mutable: false, rebindable: false, var_mut: false, owned: false, variadic: false, default: None, line: 0, col: 0 };
                        walk_body(&scope, Some(&s.name), &[p], &st.body, d, tables);
                    }
                    for c in &s.conversions { walk_body(&scope, Some(&s.name), &[], &c.body, d, tables); }
                }
                Item::Trait(t) => for m in &t.defaults { walk_body(&format!("::{}", m.name), Some(&t.name), &m.params, &m.body, d, tables) },
                Item::Enum(e) => for m in &e.methods { walk_body(&format!("{}::{}", e.name, m.name), Some(&e.name), &m.params, &m.body, d, tables) },
                Item::Ext(e) => for m in &e.methods { walk_body(&format!("{}::{}", e.type_name, m.name), Some(&e.type_name), &m.params, &m.body, d, tables) },
                Item::Mod(m) => go(&m.items, d, tables),
                _ => {}
            }
        }
    }
    go(&program.items, &d, &mut tables);
    // The file's own statements are one body (scope ""), the top-level `let`s its locals.
    let top: Vec<Stmt> = program.items.iter().filter_map(|i| match i {
        Item::Stmt(s) => Some(s.clone()),
        Item::Let(l) => Some(Stmt::Let(l.clone())),
        _ => None,
    }).collect();
    if !top.is_empty() { walk_body("", None, &[], &top, &d, &mut tables); }
    tables
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

/// The type with every written or decided representation (`'shared`, `'owned`, a borrow, `mut`, ...) removed.
fn bare(t: &Type) -> Type {
    match canon(t) {
        Type::Qualified(inner, _) | Type::Mut(inner) => bare(&inner),
        Type::Array(i) => Type::Array(Box::new(bare(&i))),
        Type::Optional(i) => Type::Optional(Box::new(bare(&i))),
        other => other,
    }
}

/// A type as the walk and the emitter both mean it: builtin names spelled as `Named` become their variant, and the
/// `'inline` qualifier, which is the default representation the emitter adds to a bare struct, is dropped (a
/// representation decision, `decided_qual`'s business, not part of the type).
fn canon(t: &Type) -> Type {
    match t {
        Type::Qualified(inner, OwnerQual::Inline) => canon(inner),
        Type::Named(n) => match n.as_str() {
            "int" => Type::Int, "uint" => Type::Uint, "uint8" | "byte" => Type::Uint8, "int8" => Type::Int8,
            "int16" => Type::Int16, "int32" => Type::Int32, "int64" => Type::Int64, "int128" => Type::Int128,
            "uint16" => Type::Uint16, "uint32" => Type::Uint32, "uint64" => Type::Uint64, "uint128" => Type::Uint128,
            "float16" | "f16" => Type::Float16, "float32" | "f32" => Type::Float32, "float64" | "float" => Type::Float64,
            "string" => Type::Str, "bool" => Type::Bool,
            _ => t.clone(),
        },
        Type::Qualified(inner, q) => Type::Qualified(Box::new(canon(inner)), q.clone()),
        Type::Array(i) => Type::Array(Box::new(canon(i))),
        Type::Optional(i) => Type::Optional(Box::new(canon(i))),
        Type::Mut(i) => Type::Mut(Box::new(canon(i))),
        Type::LabeledArray(i, axes) => Type::LabeledArray(Box::new(canon(i)), axes.clone()),
        _ => t.clone(),
    }
}

impl super::Transpiler {
    /// The declared type of a local or parameter of the body being emitted: what `var_types` holds. Every read of
    /// the emitter's table by variable name goes through here. The emitter's table answers; in `check` mode the
    /// walk's declared type for the same `(scope, name)` is compared and a difference is logged
    /// (`TYPEVARMISMATCH`), the gate before the walk can answer instead.
    pub(crate) fn local_type(&self, v: &str) -> Option<&Type> {
        let own = self.var_types.get(v);
        // The emitter keeps no declared type for the locals of the file's own statements (it has `global_var_types`
        // for some); the walk does, and answers where the emitter has nothing.
        if own.is_none() && self.type_table_mode == TypeTableMode::Walk && self.type_scope.is_empty() && self.self_type.is_none() {
            let t = self.walk_top_types.get(v).and_then(|t| t.as_ref());
            if t.is_some() { self.type_table.borrow_mut().vt_walk_answers += 1; }
            return t;
        }
        if self.type_table_mode == TypeTableMode::Check {
            let walked = self.built_local_types.borrow().get(&(self.type_scope.clone(), v.to_string())).cloned().flatten();
            if let Some(w) = walked {
                match own {
                    Some(o) if canon(o) == canon(&w) => self.type_table.borrow_mut().vt_hits += 1,
                    Some(o) if bare(o) == bare(&w) => self.type_table.borrow_mut().vt_repr += 1,
                    Some(o) => {
                        self.type_table.borrow_mut().vt_conflicts += 1;
                        self.log_check(format!("TYPEVARMISMATCH ({:?}, {:?}) walk {:?} emitter {:?}", self.type_scope, v, w, o));
                    }
                    None => {
                        self.type_table.borrow_mut().vt_walk_only += 1;
                        self.log_check(format!("TYPEVARWALKONLY ({:?}, {:?}) walk {:?}", self.type_scope, v, w));
                    }
                }
            }
        }
        own
    }

    /// The overload-resolution type of a literal-initialised local, or of a loop variable over a typed array: what
    /// `var_scalar_types` holds. The emitter answers; `check` compares the walk's claim (`TYPESCALARMISMATCH`).
    pub(crate) fn local_scalar_type(&self, v: &str) -> Option<Type> {
        let own = self.var_scalar_types.get(v).cloned();
        let walked = || self.built_scalar_types.borrow().get(&(self.type_scope.clone(), v.to_string())).cloned().flatten();
        match self.type_table_mode {
            TypeTableMode::Check => {
                if let Some(w) = walked() {
                    match &own {
                        Some(o) if canon(o) == canon(&w) => self.type_table.borrow_mut().sc_hits += 1,
                        Some(o) => {
                            self.type_table.borrow_mut().sc_conflicts += 1;
                            self.log_check(format!("TYPESCALARMISMATCH ({:?}, {:?}) walk {:?} emitter {:?}", self.type_scope, v, w, o));
                        }
                        None => {
                            self.type_table.borrow_mut().sc_walk_only += 1;
                            self.log_check(format!("TYPESCALARWALKONLY ({:?}, {:?}) walk {:?}", self.type_scope, v, w));
                        }
                    }
                }
                own
            }
            _ => own,
        }
    }

    /// The std type (`TcpStream`, `Child`, `(TcpStream, SocketAddr)?`, ...) of a local bound from a std call chain, or
    /// the shared handle of an `if let`/`guard let` over an actor dict: what `var_std_types` holds. The emitter answers;
    /// `check` counts the reads and compares any walk claim.
    pub(crate) fn local_std_type(&self, v: &str) -> Option<&Type> {
        let own = self.var_std_types.get(v);
        if self.type_table_mode == TypeTableMode::Check {
            self.type_table.borrow_mut().std_reads += 1;
            if own.is_some() { self.type_table.borrow_mut().std_reads_known += 1; }
            if let Some(w) = self.built_std_types.borrow().get(&(self.type_scope.clone(), v.to_string())).cloned().flatten() {
                match own {
                    Some(o) if canon(o) == canon(&w) => self.type_table.borrow_mut().std_hits += 1,
                    Some(o) => {
                        self.type_table.borrow_mut().std_conflicts += 1;
                        self.log_check(format!("TYPESTDMISMATCH ({:?}, {:?}) walk {:?} emitter {:?}", self.type_scope, v, w, o));
                    }
                    None => {
                        self.type_table.borrow_mut().std_walk_only += 1;
                        self.log_check(format!("TYPESTDWALKONLY ({:?}, {:?}) walk {:?}", self.type_scope, v, w));
                    }
                }
            }
        }
        own
    }

    /// The initialiser the emitter recorded for the local `v` (`var_init_exprs`): flat by name, the latest `let` of that
    /// name, never reset between bodies. `check` compares it with the walk's per-body table and logs, per read:
    /// `TYPEINITMISMATCH` (the body's one `let` of the name has another initialiser), `TYPEINITUNSURE` (the body binds
    /// the name more than once, or not by a `let`: the emitter answers with the latest one) and `TYPEINITSTRAY` (the
    /// body does not bind the name at all: the entry comes from elsewhere).
    pub(crate) fn local_init_expr(&self, v: &str) -> Option<&Expr> {
        let own = self.var_init_exprs.get(v);
        // A name the body being emitted does not bind (a bare field of `self`, a name another body bound) has no
        // initialiser here: the emitter's table is flat by name and never reset, so it would answer with a stale one.
        if own.is_some() && self.type_table_mode == TypeTableMode::Walk {
            let table = self.built_inits.borrow();
            let scope = &self.type_scope;
            if table.contains_key(&(scope.clone(), String::new())) && !table.contains_key(&(scope.clone(), v.to_string())) {
                drop(table);
                self.type_table.borrow_mut().init_dropped += 1;
                return None;
            }
        }
        if self.type_table_mode == TypeTableMode::Check {
            self.type_table.borrow_mut().init_reads += 1;
            if let Some(o) = own {
                let walked = self.built_inits.borrow().get(&(self.type_scope.clone(), v.to_string())).copied();
                match walked {
                    Some(Some(pos)) if pos == (o.line, o.col) => self.type_table.borrow_mut().init_hits += 1,
                    Some(Some(pos)) => {
                        self.type_table.borrow_mut().init_conflicts += 1;
                        self.log_check(format!("TYPEINITMISMATCH ({:?}, {:?}) walk {:?} emitter {:?}", self.type_scope, v, pos, (o.line, o.col)));
                    }
                    Some(None) => {
                        self.type_table.borrow_mut().init_unsure += 1;
                        self.log_check(format!("TYPEINITUNSURE ({:?}, {:?}) emitter {:?}", self.type_scope, v, (o.line, o.col)));
                    }
                    None => {
                        self.type_table.borrow_mut().init_stray += 1;
                        self.log_check(format!("TYPEINITSTRAY ({:?}, {:?}) emitter {:?}", self.type_scope, v, (o.line, o.col)));
                    }
                }
            }
        }
        own
    }

    /// The last-use sites of the body of `f` (`last_use.rs`): which by-value reads of a local move instead of clone.
    /// Computed from the body alone, so the decisions pass has already made it before any emission
    /// (`last_use_table`, keyed by scope). The emitter computes it again here; `check` compares the two
    /// (`LASTUSEMISMATCH`), walk mode reads the table where the scope has one body.
    pub(crate) fn last_use_for(&self, f: &FnDecl) -> crate::transpiler::last_use::LastUseSites {
        let own = || crate::transpiler::last_use::last_use_sites(&f.params, &f.body);
        let scope = self.type_scope.clone();
        let tabled = self.last_use_table.borrow().get(&scope).cloned().flatten();
        match (self.type_table_mode, tabled) {
            (TypeTableMode::Walk, Some(t)) => { self.type_table.borrow_mut().lu_table_answers += 1; t }
            (TypeTableMode::Check, Some(t)) => {
                let o = own();
                if o == t { self.type_table.borrow_mut().lu_hits += 1; } else {
                    self.type_table.borrow_mut().lu_conflicts += 1;
                    self.log_check(format!("LASTUSEMISMATCH ({:?}) table {} sites emitter {} sites", scope, t.len(), o.len()));
                }
                o
            }
            _ => own(),
        }
    }

    /// Whether `v` is held as a plain owned value, not behind a lock, reference, or counted handle:
    /// the representation half of `is_last_use_move` (a `Var` the analysis found last-used may only
    /// be moved when it is). The emitter answers from its sets (`arc_vars`, `var_mutex_types`,
    /// `var_rwlock_types`, `var_primitive_params`, `fn_current_params`, `inferred_qualifiers`,
    /// `ref_bound_vars`); the decisions pass says the same through `decision_table` (the qualifier
    /// chosen for the local/param) and the signature. `check` compares the two per read (`LASTUSEREPMISMATCH`).
    pub(crate) fn last_use_representation_ok(&self, v: &str) -> bool {
        let param_ty = self.fn_current_params.get(v);
        let param_explicit_non_owned = param_ty.is_some_and(|t| matches!(
            t.without_mut(),
            Type::Qualified(
                _,
                OwnerQual::Borrow
                    | OwnerQual::BorrowMut
                    | OwnerQual::BorrowOwned
                    | OwnerQual::BorrowOption
                    | OwnerQual::BorrowOptionMut
                    | OwnerQual::BorrowWeak
                    | OwnerQual::BorrowShared
                    | OwnerQual::Actor
                    | OwnerQual::ActorTask
                    | OwnerQual::Guard
                    | OwnerQual::GuardTask
                    | OwnerQual::Observed
                    | OwnerQual::Shared
                    | OwnerQual::Weak
                    | OwnerQual::Atomic
            )
        ));
        let common_blocked = self.arc_vars.contains(v)
            || self.var_mutex_types.contains(v)
            || self.var_rwlock_types.contains(v)
            || self.var_primitive_params.contains(v)
            || self.fn_current_params_lent.contains(v)
            || self.fn_current_params_rebind_only.contains(v)
            || self.ref_bound_vars.contains(v)
            || self.dyn_borrowed_vars.borrow().contains(v)
            || param_explicit_non_owned;
        let own = !common_blocked
            && (param_ty.is_none()
                || matches!(
                    self.inferred_qualifiers.get(v),
                    None | Some(OwnerQual::Inline | OwnerQual::Owned)
                ));
        if !matches!(self.type_table_mode, TypeTableMode::Check | TypeTableMode::Walk) { return own; }
        let decided = self.decision_table.borrow().get(&(self.type_scope.clone(), v.to_string())).cloned();
        let by_decision = !common_blocked
            && if param_ty.is_some() {
                matches!(decided, None | Some(OwnerQual::Inline | OwnerQual::Owned))
            } else {
                !matches!(decided, Some(OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Atomic | OwnerQual::Shared | OwnerQual::ActorTask | OwnerQual::GuardTask))
            };
        if self.type_table_mode == TypeTableMode::Check {
            if !own { self.type_table.borrow_mut().rep_blocked += 1; }
            if by_decision == own { self.type_table.borrow_mut().rep_hits += 1; } else {
                self.type_table.borrow_mut().rep_conflicts += 1;
                self.log_check(format!("LASTUSEREPMISMATCH ({:?}, {:?}) emitter {} decisions {} (arc {} mutex {} prim {} param {} decided {:?})",
                    self.type_scope, v, own, by_decision, self.arc_vars.contains(v), self.var_mutex_types.contains(v),
                    self.var_primitive_params.contains(v), self.fn_current_params.contains_key(v), decided));
            }
        }
        if self.type_table_mode == TypeTableMode::Walk {
            // A move needs both to agree: where they differ the answer is the always-correct one (clone), the
            // rule `last_use.rs` itself follows ("every doubt resolves to clone").
            self.type_table.borrow_mut().rep_decided += 1;
            return own && by_decision;
        }
        own
    }

    /// Whether the local `v` is a `T'actor` held behind a mutex (or `RefCell`): what `var_mutex_types` holds, and what makes
    /// a method call or field access on it take the lock. Every read of the set by variable name goes through here.
    /// The emitter answers; `check` compares the answer the decisions give (the qualifier `infer_qualifiers` chose for
    /// the local, else the outermost qualifier of its declared type in the walk, which covers a written type, a
    /// parameter and the result of a call) and logs a difference (`LOCKMISSING`: the emitter holds a lock the walk does
    /// not see; `LOCKEXTRA`: the walk says actor, the emitter does not hold one).
    pub(crate) fn is_mutex_local(&self, v: &str) -> bool {
        let own = self.var_mutex_types.contains(v);
        if self.type_table_mode == TypeTableMode::Check { self.check_lock(v, own, OwnerQual::Actor, "LOCK"); }
        own
    }

    /// Whether the local `v` is a `T'guard` held behind a read-write lock: what `var_rwlock_types` holds. Same check as
    /// `is_mutex_local` with `'guard` as the qualifier (`RWLOCKMISSING` / `RWLOCKEXTRA`).
    pub(crate) fn is_rwlock_local(&self, v: &str) -> bool {
        let own = self.var_rwlock_types.contains(v);
        if self.type_table_mode == TypeTableMode::Check { self.check_lock(v, own, OwnerQual::Guard, "RWLOCK"); }
        own
    }

    /// The comparison behind `is_mutex_local` and `is_rwlock_local`: `qual` is the qualifier that makes a local this kind of
    /// lock (`'actor`, `'guard`); `prefix` names the log lines and the counters.
    fn check_lock(&self, v: &str, own: bool, qual: OwnerQual, prefix: &str) {
        let key = (self.type_scope.clone(), v.to_string());
        let decided = self.decision_table.borrow().get(&key).cloned();
        let outer = |t: &Type| -> Option<OwnerQual> { match t.without_mut() { Type::Qualified(_, q) => Some(q.clone()), _ => None } };
        let walked_ty = self.built_local_types.borrow().get(&key).cloned();
        let declared = walked_ty.clone().flatten().and_then(|t| outer(&t));
        // A written qualifier wins over what the decisions pass chose for an unqualified local; a candidate set
        // (`'new`, a qualifier group) is resolved by the decision.
        // What the decisions pass chose counts as a lock only for a value that has fields to route through one: an
        // inferred qualifier on a scalar (what a `with` over a plain `var int` makes of it) is a no-op block, not a lock.
        let scalar = |t: &Option<Type>| t.as_ref().is_some_and(|t| super::Transpiler::is_copy_type(&canon(t.without_mut())));
        let decided_lock = |t: &Option<Type>| decided.as_ref() == Some(&qual) && !scalar(t);
        let by_name = match &declared { Some(OwnerQual::Union(_)) | None => decided_lock(&walked_ty.clone().flatten()), Some(q) => *q == qual };
        // By position: the latest binding event of the name that is in force where the emitter has reached (a `let`, an
        // arm's pattern for the arm, a `with` for its body). Where the name has events and none is in force yet, it is
        // not bound: a read before the binding (a pre-scan of the body) is not a lock.
        let pos = self.cur_pos.get();
        let events = self.built_binders.borrow().get(&key).cloned();
        let by_pos = events.as_ref().and_then(|evs| {
            evs.iter().filter(|b| b.pos <= pos && b.end.is_none_or(|e| pos <= e)).max_by_key(|b| b.pos).cloned()
        });
        let walk = match (&by_pos, &events) {
            (Some(b), _) if b.unlocked => false,
            (Some(b), _) => match b.ty.as_ref().and_then(outer) {
                Some(OwnerQual::Union(_)) | None => decided_lock(&b.ty),
                Some(q) => q == qual,
            },
            (None, Some(_)) => false,
            (None, None) => by_name,
        };
        let rw = prefix == "RWLOCK";
        let mut tt = self.type_table.borrow_mut();
        if own { if rw { tt.rw_held += 1 } else { tt.lock_held += 1 } }
        if walk == own { if rw { tt.rw_hits += 1 } else { tt.lock_hits += 1 } } else {
            if rw { tt.rw_conflicts += 1 } else { tt.lock_conflicts += 1 }
            drop(tt);
            let kind = if own { format!("{prefix}MISSING") } else { format!("{prefix}EXTRA") };
            if std::env::var_os("BORING_TYPE_TABLE_BACKTRACE").is_some() { eprintln!("{kind} {v} {}", std::backtrace::Backtrace::force_capture()); }
            self.log_check(format!("{kind} ({:?}, {:?}) decided {:?} declared {:?} walk type {:?} at {:?} binder {:?}", self.type_scope, v, decided, declared, walked_ty, pos, by_pos));
        }
    }

    /// Whether the emitter tracks a declared type for `v` (`var_types.contains_key`).
    pub(crate) fn has_local_type(&self, v: &str) -> bool {
        self.local_type(v).is_some()
    }

    fn log_check(&self, line: String) {
        if let Some(path) = std::env::var_os("BORING_TYPE_TABLE_LOG") {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(f, "{line}");
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
