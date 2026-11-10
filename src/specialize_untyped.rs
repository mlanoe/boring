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

//! SPIKE of docs/design-notes/typed-specialisation.md (phase 1): a free function whose
//! untyped parameter has a method called on it (`def bump(a): a.inc()`) cannot be a Rust
//! generic, because what `a.inc()` emits depends on the qualifier of the argument (`'actor`
//! takes the lock, `'owned` does not). Each call is therefore typed from its *annotated*
//! argument (`let Counter'actor c = ...`), the function is cloned with the concrete type
//! (`bump_Counter_actor`), the call is rewritten, and the generic original is dropped.
//!
//! Deliberately narrow: top-level free functions only (no `mod`, no methods), positional
//! arguments only, an argument must be a plain variable with an annotated `let`/parameter type.
//! An argument whose type is not known here is an error, never a silent fallback. The
//! substitution engine is the monomorphizer's own (`substitute_fn_decl`).
//!
//! `BORING_SHOW_MONO=1` prints the specialised headers to stderr (a throwaway stand-in for the
//! `.bri` printer of the design note).

use crate::ast::*;
use crate::ast_walk::Walker;
use crate::desugar_array_block::describe_type;
use crate::infer_param_generics::{fresh_name, unbounded_use};
use crate::parser::ParseError;
use crate::transpiler::monomorphize::{mangle_type, substitute_fn_decl};
use std::collections::{HashMap, HashSet};

const MAX_INSTANCES: usize = 64;

fn err(msg: String, line: usize, col: usize) -> ParseError {
    ParseError::Generic { msg, line, col, len: 1 }
}

/// Indices of the parameters declared without a type.
fn implicit_indices(f: &FnDecl) -> Vec<usize> {
    f.params.iter().enumerate()
        .filter(|(_, p)| matches!(&p.ty, Some(Type::TypeParam(t)) if *t == fresh_name(&p.name)))
        .map(|(i, _)| i)
        .collect()
}

/// The unbounded uses of an untyped parameter that typed specialisation takes over: the ones
/// whose meaning depends on the type of the argument (a method call, a field read or write, an
/// index). The others (`a + b` of two untyped parameters, ...) stay errors.
pub(crate) fn takes_over(what: &str) -> bool {
    what.starts_with("the method call") || what.starts_with("the field access") || what.starts_with("indexing")
}

/// A function this pass takes over: an untyped parameter used in a way that needs its type.
fn needs_specialisation(f: &FnDecl) -> bool {
    if f.qualifier.is_some() || f.task || f.stream || f.is_native { return false; }
    let idx = implicit_indices(f);
    if idx.is_empty() { return false; }
    if !f.slot_quals.is_empty() { return true; }
    let names: Vec<String> = idx.iter().map(|&i| f.params[i].name.clone()).collect();
    matches!(unbounded_use(&names, &f.body, &[]), Some(u) if takes_over(&u.what))
}

/// `f` passes one of its own untyped parameters, as a plain variable, to an untyped parameter of
/// a function (`g(a)`) or method (`o.g(a)`) already taken over. `free` and `methods` map the names
/// taken over to their declarations.
fn passes_param_to(f: &FnDecl, free: &HashMap<String, &FnDecl>, methods: &HashMap<String, &FnDecl>) -> bool {
    let own: HashSet<&str> = implicit_indices(f).iter().map(|&i| f.params[i].name.as_str()).collect();
    let mut hit = false;
    let mut body = f.body.clone();
    let mut on_expr = |e: &mut Expr| {
        let (decl, args) = match &e.kind {
            ExprKind::Call(callee, args) => match &callee.kind {
                ExprKind::Var(g) => match free.get(g) { Some(d) => (*d, args), None => return },
                _ => return,
            },
            ExprKind::MethodCall(_, g, args) => match methods.get(g) { Some(d) => (*d, args), None => return },
            _ => return,
        };
        for i in implicit_indices(decl) {
            if let Some(Arg { value: Expr { kind: ExprKind::Var(v), .. }, .. }) = args.get(i) {
                if own.contains(v.as_str()) { hit = true; }
            }
        }
    };
    let mut on_stmt = |_: &mut Stmt| {};
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
    hit
}

/// The qualifier written directly on the base type of `t` (`Counter'actor` and `Counter'actor'weak`
/// are both `'actor`); `None` when no qualifier is written.
fn base_qual(t: &Type) -> Option<&OwnerQual> {
    match t {
        Type::Mut(i) => base_qual(i),
        Type::Qualified(inner, q) => match &**inner {
            Type::Qualified(..) => base_qual(inner),
            _ => Some(q),
        },
        _ => None,
    }
}

/// `'actor`, with its tick.
fn qual_text(q: &OwnerQual) -> String {
    let name = OwnerQual::chain_name(std::slice::from_ref(q)).unwrap_or_else(|| format!("{q:?}"));
    format!("'{}", name.trim_start_matches('\''))
}

/// `T'owned a` on an explicit generic `<T>`: `(parameter index, name, qualifier)`.
fn explicit_slots(f: &FnDecl) -> Vec<(usize, String, OwnerQual)> {
    f.params.iter().enumerate().filter_map(|(i, p)| {
        let t = p.ty.as_ref()?.without_mut();
        match t {
            Type::Qualified(inner, q) => match &**inner {
                // A qualifier *parameter* (`T'q`, from `<T'q>`) is not a written qualifier to compare.
                Type::TypeParam(n) if f.type_params.contains(n) && !n.starts_with("T_") && !matches!(q, OwnerQual::Param(_)) => Some((i, p.name.clone(), q.clone())),
                _ => None,
            },
            _ => None,
        }
    }).collect()
}

/// A written qualifier on an explicit generic slot (`def bump<T>(T'owned a)`) refuses a call whose
/// argument is a variable written with another qualifier. Nothing is specialised: the call stays on
/// the generic path, this only reports the mismatch the generic path would not see.
fn check_explicit_slots(program: &Program) -> Result<(), ParseError> {
    let mut slots: HashMap<String, Vec<(usize, String, OwnerQual)>> = HashMap::new();
    let mut count: HashMap<String, usize> = HashMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item { *count.entry(f.name.clone()).or_default() += 1; }
    }
    for item in &program.items {
        if let Item::Fn(f) = item {
            let sl = explicit_slots(f);
            if !sl.is_empty() && f.qualifier.is_none() && count.get(&f.name) == Some(&1) { slots.insert(f.name.clone(), sl); }
        }
    }
    if slots.is_empty() { return Ok(()); }
    for item in &program.items {
        let Item::Fn(f) = item else { continue };
        let (env, ambiguous) = local_types(&f.params, &f.body);
        let mut failure: Option<ParseError> = None;
        let mut body = f.body.clone();
        let mut on_expr = |e: &mut Expr| {
            if failure.is_some() { return; }
            let ExprKind::Call(callee, args) = &e.kind else { return };
            let ExprKind::Var(name) = &callee.kind else { return };
            let Some(list) = slots.get(name) else { return };
            for (i, pname, want) in list {
                let arg = args.iter().find(|a| a.label.as_deref() == Some(pname.as_str()))
                    .or_else(|| args.get(*i).filter(|a| a.label.is_none()));
                let Some(Arg { value: Expr { kind: ExprKind::Var(v), line, col, .. }, .. }) = arg else { continue };
                if ambiguous.contains(v) { continue; }
                let Some(have) = env.get(v).and_then(base_qual) else { continue };
                if matches!(have, OwnerQual::Param(_)) { continue; }
                if have != want {
                    failure = Some(err(
                        format!("the parameter `{pname}` of `{name}` is declared `{}`-qualified, but `{v}` is written `{}`", qual_text(want), qual_text(have)),
                        *line, *col));
                    return;
                }
            }
        };
        let mut on_stmt = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
        if let Some(e) = failure { return Err(e); }
    }
    Ok(())
}

/// A generic struct or function: its type parameters, and the slots they appear in
/// (`(field or parameter name, position, type parameter)`).
type GenericDecl = (Vec<String>, Vec<(String, usize, String)>);

/// The type parameter named by a slot type (`T`, `mut T`), if it is one of `params`.
fn slot_param<'a>(t: &'a Type, params: &[String]) -> Option<&'a str> {
    match t.without_mut() {
        Type::TypeParam(n) | Type::Named(n) if params.iter().any(|p| p == n) => Some(n.as_str()),
        _ => None,
    }
}

/// `Entry(c)` / `show(c)` on a generic struct or function with plain type parameters, where an
/// argument is a variable *written* with a qualifier (`let Counter'actor c`): the type argument is
/// that written type, so the call becomes `Entry<Counter'actor>(c)` and the turbofish
/// monomorphizer specialises it (what `e.value.inc()` emits depends on the qualifier, a generic
/// body cannot take the lock). Only when every type parameter is determined by a variable with a
/// written type and at least one carries a qualifier; any other call is left untouched, on the
/// ordinary generic path.
fn infer_qualified_generic_args(program: &mut Program) {
    // name -> (type parameters, slots: (field/parameter name, position, type parameter))
    let mut decls: HashMap<String, GenericDecl> = HashMap::new();
    let mut count: HashMap<String, usize> = HashMap::new();
    for item in &program.items {
        match item {
            Item::Fn(f) => *count.entry(f.name.clone()).or_default() += 1,
            Item::Struct(s) => *count.entry(s.name.clone()).or_default() += 1,
            _ => {}
        }
    }
    let plain = |tp: &[String]| !tp.is_empty() && tp.iter().all(|p| !p.starts_with('^') && !p.starts_with('$') && !p.starts_with("T_"));
    for item in &program.items {
        match item {
            Item::Struct(s) if plain(&s.type_params) && s.inits.is_empty() && count.get(&s.name) == Some(&1) => {
                let slots = s.fields.iter().enumerate()
                    .filter_map(|(i, f)| slot_param(&f.ty, &s.type_params).map(|t| (f.name.clone(), i, t.to_string())))
                    .collect();
                decls.insert(s.name.clone(), (s.type_params.clone(), slots));
            }
            Item::Fn(f) if plain(&f.type_params) && f.qualifier.is_none() && count.get(&f.name) == Some(&1) => {
                let slots = f.params.iter().enumerate()
                    .filter_map(|(i, p)| p.ty.as_ref().and_then(|t| slot_param(t, &f.type_params)).map(|t| (p.name.clone(), i, t.to_string())))
                    .collect();
                decls.insert(f.name.clone(), (f.type_params.clone(), slots));
            }
            _ => {}
        }
    }
    if decls.is_empty() { return; }
    for item in program.items.iter_mut() {
        let Item::Fn(f) = item else { continue };
        let (env, ambiguous) = local_types(&f.params, &f.body);
        let mut on_expr = |e: &mut Expr| {
            let ExprKind::Call(callee, args) = &e.kind else { return };
            let ExprKind::Var(name) = &callee.kind else { return };
            let Some((tparams, slots)) = decls.get(name) else { return };
            let mut chosen: HashMap<&str, Type> = HashMap::new();
            let mut qualified = false;
            for (sname, pos, tp) in slots {
                let arg = args.iter().find(|a| a.label.as_deref() == Some(sname.as_str()))
                    .or_else(|| args.get(*pos).filter(|a| a.label.is_none()));
                let Some(Arg { value: Expr { kind: ExprKind::Var(v), .. }, .. }) = arg else { continue };
                if ambiguous.contains(v) { continue; }
                let Some(t) = env.get(v) else { continue };
                if matches!(base_qual(t), Some(OwnerQual::Param(_))) { continue; }
                if base_qual(t).is_some() { qualified = true; }
                chosen.entry(tp.as_str()).or_insert_with(|| t.clone().into_without_mut());
            }
            if !qualified || tparams.iter().any(|p| !chosen.contains_key(p.as_str())) { return; }
            let types: Vec<Type> = tparams.iter().map(|p| chosen[p.as_str()].clone()).collect();
            if types.iter().any(|t| slug(t).is_none()) { return; }
            let (callee, args) = (callee.clone(), args.clone());
            e.kind = ExprKind::GenericCall(callee, types, args);
        };
        let mut on_stmt = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut f.body);
    }
}

fn slug(t: &Type) -> Option<String> {
    match t {
        Type::Mut(i) => slug(i).map(|s| format!("mut_{s}")),
        Type::Qualified(i, q) => {
            let base = slug(i)?;
            Some(format!("{base}_{}", OwnerQual::chain_mangle(std::slice::from_ref(q))?))
        }
        other => mangle_type(other),
    }
}

/// Annotated types visible in a body: the typed parameters and the annotated `let`s. A name
/// bound more than once with anything but one and the same written type (a shadowing `let`, an
/// unannotated rebinding, a `for` variable) is *ambiguous*: it is reported, never guessed.
fn local_types(params: &[Param], body: &[Stmt]) -> (HashMap<String, Type>, HashSet<String>) {
    let mut bindings: HashMap<String, Vec<Option<Type>>> = HashMap::new();
    for p in params {
        let t = p.ty.clone().filter(|t| !matches!(t, Type::TypeParam(_)));
        bindings.entry(p.name.clone()).or_default().push(t);
    }
    let mut body = body.to_vec();
    let mut on_expr = |_: &mut Expr| {};
    let mut on_stmt = |s: &mut Stmt| match s {
        Stmt::Let(l) => bindings.entry(l.name.clone()).or_default().push(l.ty.clone()),
        Stmt::For(fs) => for v in &fs.vars { bindings.entry(v.clone()).or_default().push(None); },
        _ => {}
    };
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
    let mut env = HashMap::new();
    let mut ambiguous = HashSet::new();
    for (name, list) in bindings {
        let first = list[0].clone();
        let same = list.iter().all(|t| match (t, &first) {
            (Some(a), Some(b)) => format!("{a:?}") == format!("{b:?}"),
            _ => false,
        });
        match (list.len(), first) {
            (_, Some(t)) if same => { env.insert(name, t); }
            (1, None) => {}
            _ => { ambiguous.insert(name); }
        }
    }
    (env, ambiguous)
}

thread_local! {
    /// Where each specialised copy comes from, by its name, for the `.bri` (`# bump_Counter: copy of ...`).
    static ORIGINS: std::cell::RefCell<HashMap<String, String>> = std::cell::RefCell::new(HashMap::new());
}

/// Forgets the origins recorded so far (one `.bri` is printed per parse).
pub fn clear_origins() {
    ORIGINS.with(|o| o.borrow_mut().clear());
}

/// The comment describing where the specialised copy `name` comes from.
pub fn origin_of(name: &str) -> Option<String> {
    ORIGINS.with(|o| o.borrow().get(name).cloned())
}

struct Instance {
    at: (usize, usize),
    base: String,
    mangled: String,
    /// `(parameter index, concrete type)`
    types: Vec<(usize, Type)>,
    /// A method (the struct or enum that declares it), not a free function.
    method_of: Option<String>,
}

/// The taken-over functions and methods, by name.
struct Candidates {
    free: HashMap<String, FnDecl>,
    /// method name -> (declaring struct or enum, declaration)
    methods: HashMap<String, (String, FnDecl)>,
}

impl Candidates {
    fn is_empty(&self) -> bool { self.free.is_empty() && self.methods.is_empty() }
}

/// The concrete type of each untyped parameter of `decl` for one call, from the annotated
/// variables given as arguments.
fn call_types(
    name: &str,
    decl: &FnDecl,
    args: &[Arg],
    env: &HashMap<String, Type>,
    ambiguous: &HashSet<String>,
    at: (usize, usize),
) -> Result<Vec<(usize, Type)>, ParseError> {
    let mut types: Vec<(usize, Type)> = Vec::new();
    for i in implicit_indices(decl) {
        let pname = &decl.params[i].name;
        // The argument of this parameter: the one labelled with its name, else the positional one.
        let arg = args.iter().find(|a| a.label.as_deref() == Some(pname.as_str()))
            .or_else(|| args.get(i).filter(|a| a.label.is_none()));
        let Some(arg) = arg else {
            return Err(err(format!("cannot specialise `{name}`: no argument for the untyped parameter `{pname}`"), at.0, at.1));
        };
        if arg.spread {
            return Err(err(format!("cannot specialise `{name}`: a spread argument cannot give the type of `{pname}`"), arg.value.line, arg.value.col));
        }
        if let ExprKind::Var(v) = &arg.value.kind {
            if ambiguous.contains(v) {
                return Err(err(
                    format!("cannot specialise `{name}` for this call: `{v}` is bound more than once in this function with different (or unwritten) types, so the type of the argument given to `{pname}` is ambiguous; use a variable bound once, with its type written"),
                    arg.value.line, arg.value.col));
            }
        }
        // The `mut` permission of the argument's own binding is not part of the
        // parameter's type: the parameter keeps the permission it was declared with.
        let ty = match &arg.value.kind {
            ExprKind::Var(v) => env.get(v).cloned().map(Type::into_without_mut),
            _ => None,
        };
        match ty {
            Some(t) if slug(&t).is_some() => {
                let t = match decl.slot_quals.iter().find(|(n, _)| n == pname) {
                    None => t,
                    Some((_, want)) => match base_qual(&t) {
                        Some(have) if have != want => {
                            return Err(err(
                                format!("the parameter `{pname}{}` of `{name}` expects that qualifier, but this argument is written `{}`", qual_text(want), qual_text(have)),
                                arg.value.line, arg.value.col));
                        }
                        Some(_) => t,
                        None => Type::Qualified(Box::new(t), want.clone()),
                    },
                };
                types.push((i, t));
            }
            _ => {
                return Err(err(
                    format!("cannot specialise `{name}` for this call: the type of the argument given to the untyped parameter `{pname}` is not known here; pass a variable whose type is written (`let Counter'actor c = ...`)"),
                    arg.value.line, arg.value.col));
            }
        }
    }
    Ok(types)
}

/// Rewrites the calls of `candidates` found in a body (`params` are the parameters of the
/// function around it, none at the top level), recording the instances they need. A free
/// function named in `shadowed` is declared again where this body lives: its calls are left alone.
fn rewrite_calls(
    params: &[Param],
    body: &mut [Stmt],
    candidates: &Candidates,
    shadowed: &HashSet<String>,
    instances: &mut Vec<Instance>,
    seen: &mut HashSet<String>,
) -> Result<(), ParseError> {
    let (env, ambiguous) = local_types(params, body);
    let mut failure: Option<ParseError> = None;
    let mut new_instances: Vec<Instance> = Vec::new();
    {
        let mut on_expr = |e: &mut Expr| {
            if failure.is_some() { return; }
            let at = (e.line, e.col);
            // (name, declaration, owner of a method)
            let (name, decl, owner) = match &e.kind {
                ExprKind::Call(callee, _) => match &callee.kind {
                    ExprKind::Var(n) if !shadowed.contains(n) => match candidates.free.get(n) {
                        Some(d) => (n.clone(), d, None),
                        None => return,
                    },
                    _ => return,
                },
                ExprKind::MethodCall(_, n, _) => match candidates.methods.get(n) {
                    Some((o, d)) => (n.clone(), d, Some(o.clone())),
                    None => return,
                },
                _ => return,
            };
            let args = match &e.kind {
                ExprKind::Call(_, a) | ExprKind::MethodCall(_, _, a) => a.clone(),
                _ => return,
            };
            let types = match call_types(&name, decl, &args, &env, &ambiguous, at) {
                Ok(t) => t,
                Err(x) => { failure = Some(x); return; }
            };
            let key: Vec<String> = types.iter().map(|(_, t)| slug(t).unwrap_or_default()).collect();
            let mangled = format!("{name}_{}", key.join("_"));
            match &mut e.kind {
                ExprKind::Call(callee, _) => { if let ExprKind::Var(n) = &mut callee.kind { *n = mangled.clone(); } }
                ExprKind::MethodCall(_, n, _) => *n = mangled.clone(),
                _ => {}
            }
            let seen_key = match &owner { Some(o) => format!("{o}::{mangled}"), None => mangled.clone() };
            if seen.insert(seen_key) {
                new_instances.push(Instance { at, base: name, mangled, types, method_of: owner });
            }
        };
        let mut on_stmt = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
    }
    if let Some(e) = failure { return Err(e); }
    instances.extend(new_instances);
    Ok(())
}

fn show(f: &FnDecl) {
    let params: Vec<String> = f.params.iter()
        .map(|p| format!("{} {}", p.ty.as_ref().map(describe_type).unwrap_or_default(), p.name))
        .collect();
    eprintln!("# specialised: def {}({})", f.name, params.join(", "));
}

/// Where a function declaration sits.
#[derive(Clone, Copy, PartialEq)]
enum Site { Free, Method, Other }

/// Calls `f` on every function declaration of `items` with its site, whether it sits inside a
/// `mod`, and the names of the free functions declared next to it in that `mod` (empty at the top
/// level).
/// Callback of [`for_each_fn`]: the function, its site, whether it sits in a `mod`, its siblings.
type FnVisitor<'a> = dyn FnMut(&mut FnDecl, Site, bool, &HashSet<String>) + 'a;

fn for_each_fn(items: &mut [Item], in_mod: bool, siblings: &HashSet<String>, f: &mut FnVisitor<'_>) {
    for item in items.iter_mut() {
        match item {
            Item::Fn(d) => f(d, Site::Free, in_mod, siblings),
            Item::Struct(s) => for m in &mut s.methods { f(m, Site::Method, in_mod, siblings) },
            Item::Enum(e) => for m in &mut e.methods { f(m, Site::Method, in_mod, siblings) },
            Item::Ext(e) => {
                let plain = e.type_params.is_empty() && e.type_args.is_empty();
                for m in &mut e.methods { f(m, if plain { Site::Method } else { Site::Other }, in_mod, siblings) }
            }
            Item::Trait(t) => for m in &mut t.defaults { f(m, Site::Other, in_mod, siblings) },
            Item::Kernel(k) => for m in &mut k.methods { f(m, Site::Other, in_mod, siblings) },
            Item::Mod(m) => {
                let names: HashSet<String> = m.items.iter().filter_map(|i| match i { Item::Fn(d) => Some(d.name.clone()), _ => None }).collect();
                for_each_fn(&mut m.items, true, &names, f);
            }
            _ => {}
        }
    }
}

/// How many free functions of each name there are, in the file and in every `mod` (a `mod`'s items
/// are reached by their plain name from anywhere, so a name is unique across all of them or not at all).
fn free_name_counts(items: &[Item], out: &mut HashMap<String, usize>) {
    for item in items {
        match item {
            Item::Fn(f) => *out.entry(f.name.clone()).or_default() += 1,
            Item::Mod(m) => free_name_counts(&m.items, out),
            _ => {}
        }
    }
}

fn collect_free<'a>(items: &'a [Item], count: &HashMap<String, usize>, out: &mut Vec<&'a FnDecl>) {
    for item in items {
        match item {
            Item::Fn(f) if count.get(&f.name) == Some(&1)
                && f.qualifier.is_none() && !f.task && !f.stream && !f.is_native
                && !implicit_indices(f).is_empty() => out.push(f),
            Item::Mod(m) => collect_free(&m.items, count, out),
            _ => {}
        }
    }
}

fn collect_methods<'a>(items: &'a [Item], mcount: &HashMap<String, usize>, out: &mut Vec<(String, &'a FnDecl)>) {
    for item in items {
        let (owner, methods): (&str, &Vec<FnDecl>) = match item {
            Item::Struct(s) => (s.name.as_str(), &s.methods),
            Item::Enum(e) => (e.name.as_str(), &e.methods),
            // An `ext` of a plain type (not one of a generic's instances).
            Item::Ext(e) if e.type_params.is_empty() && e.type_args.is_empty() => (e.type_name.as_str(), &e.methods),
            Item::Mod(m) => { collect_methods(&m.items, mcount, out); continue; }
            _ => continue,
        };
        for m in methods {
            if mcount.get(&m.name) == Some(&1) && m.qualifier.is_none() && !m.task && !m.stream && !m.is_native
                && !implicit_indices(m).is_empty() {
                out.push((owner.to_string(), m));
            }
        }
    }
}

/// `items` with each taken-over function and method replaced by its instances, `mod`s included.
fn replace_originals(items: Vec<Item>, candidates: &Candidates, built: &[(String, Option<String>, FnDecl)], top: bool) -> Vec<Item> {
    let mut out: Vec<Item> = Vec::new();
    for item in items {
        match item {
            Item::Fn(f) if candidates.free.contains_key(&f.name) => {
                let mut any = false;
                for (b, o, spec) in built {
                    if o.is_none() && *b == f.name { any = true; out.push(Item::Fn(spec.clone())); }
                }
                if !any || f.is_pub {
                    let mut f = f;
                    if f.is_pub && !is_specialise_only(&f) {
                        f.attrs.push(Attr { name: SPECIALISE_ONLY.into(), args: vec![], line: f.line, col: f.col });
                    }
                    out.push(Item::Fn(f));
                }
            }
            Item::Struct(mut s) => {
                s.methods = swap_methods(std::mem::take(&mut s.methods), &s.name, candidates, built, top);
                out.push(Item::Struct(s));
            }
            Item::Enum(mut e) => {
                e.methods = swap_methods(std::mem::take(&mut e.methods), &e.name, candidates, built, top);
                out.push(Item::Enum(e));
            }
            Item::Ext(mut e) if e.type_params.is_empty() && e.type_args.is_empty() => {
                e.methods = swap_methods(std::mem::take(&mut e.methods), &e.type_name, candidates, built, false);
                out.push(Item::Ext(e));
            }
            Item::Mod(mut m) => {
                m.items = replace_originals(std::mem::take(&mut m.items), candidates, built, false);
                out.push(Item::Mod(m));
            }
            other => out.push(other),
        }
    }
    out
}

/// How many declarations of each name there are among all methods (struct, enum, `ext`, trait
/// defaults and required signatures, kernels): a method name is specialised only when it is unique.
fn method_name_counts(items: &[Item], out: &mut HashMap<String, usize>) {
    let mut bump = |n: &str| *out.entry(n.to_string()).or_default() += 1;
    for item in items {
        match item {
            Item::Struct(s) => for m in &s.methods { bump(&m.name) },
            Item::Enum(e) => for m in &e.methods { bump(&m.name) },
            Item::Ext(e) => for m in &e.methods { bump(&m.name) },
            Item::Trait(t) => {
                for m in &t.defaults { bump(&m.name) }
                for m in &t.signatures { bump(&m.name) }
            }
            Item::Kernel(k) => for m in &k.methods { bump(&m.name) },
            _ => {}
        }
    }
    for item in items {
        if let Item::Mod(m) = item { method_name_counts(&m.items, out); }
    }
}

/// Every function or method with an untyped parameter used in a way that needs its type
/// (`takes_over`) that the pass does not specialise keeps the error it would have had without it:
/// one declared inside a `mod`, in an `ext`, a trait or a kernel, or whose name is declared more
/// than once. Without this, the parser's relaxed check would let it through to rustc.
fn refuse_unhandled(program: &mut Program, free: &HashSet<String>, methods: &HashSet<String>) -> Result<(), ParseError> {
    let mut free_count: HashMap<String, usize> = HashMap::new();
    free_name_counts(&program.items, &mut free_count);
    let mut result: Result<(), ParseError> = Ok(());
    for_each_fn(&mut program.items, false, &HashSet::new(), &mut |f, site, in_mod, _| {
        if result.is_err() { return; }
        if f.qualifier.is_some() || f.task || f.stream || f.is_native { return; }
        let idx = implicit_indices(f);
        if idx.is_empty() { return; }
        let names: Vec<String> = idx.iter().map(|&i| f.params[i].name.clone()).collect();
        let Some(u) = unbounded_use(&names, &f.body, &[]) else { return };
        if !takes_over(&u.what) { return; }
        let _ = in_mod;
        let handled = match site {
            Site::Free => free_count.get(&f.name) == Some(&1) && free.contains(&f.name),
            Site::Method => methods.contains(&f.name),
            Site::Other => false,
        };
        if !handled { result = Err(crate::infer_param_generics::unbounded_error(u, "parameter")); }
    });
    result
}

pub fn specialize_method_users(program: &mut Program) -> Result<(), ParseError> {
    check_explicit_slots(program)?;
    infer_qualified_generic_args(program);
    let mut count: HashMap<String, usize> = HashMap::new();
    free_name_counts(&program.items, &mut count);
    let mut mcount: HashMap<String, usize> = HashMap::new();
    method_name_counts(&program.items, &mut mcount);
    let mut eligible_free: Vec<&FnDecl> = Vec::new();
    collect_free(&program.items, &count, &mut eligible_free);
    let mut eligible_methods: Vec<(String, &FnDecl)> = Vec::new();
    collect_methods(&program.items, &mcount, &mut eligible_methods);
    // A function or method is taken over when it uses a method on an untyped parameter, or passes
    // one of its own untyped parameters to one that is (a fixpoint, like the bound propagation).
    let mut names: HashSet<String> = eligible_free.iter().filter(|f| needs_specialisation(f)).map(|f| f.name.clone()).collect();
    let mut mnames: HashSet<String> = eligible_methods.iter().filter(|(_, f)| needs_specialisation(f)).map(|(_, f)| f.name.clone()).collect();
    loop {
        let free_now: HashMap<String, &FnDecl> = eligible_free.iter().filter(|f| names.contains(&f.name)).map(|f| (f.name.clone(), *f)).collect();
        let meth_now: HashMap<String, &FnDecl> = eligible_methods.iter().filter(|(_, f)| mnames.contains(&f.name)).map(|(_, f)| (f.name.clone(), *f)).collect();
        let mut grew = false;
        for f in &eligible_free {
            if !names.contains(&f.name) && passes_param_to(f, &free_now, &meth_now) {
                names.insert(f.name.clone());
                grew = true;
            }
        }
        for (_, f) in &eligible_methods {
            if !mnames.contains(&f.name) && passes_param_to(f, &free_now, &meth_now) {
                mnames.insert(f.name.clone());
                grew = true;
            }
        }
        if !grew { break; }
    }
    let candidates = Candidates {
        free: eligible_free.iter().filter(|f| names.contains(&f.name)).map(|f| (f.name.clone(), (*f).clone())).collect(),
        methods: eligible_methods.iter().filter(|(_, f)| mnames.contains(&f.name)).map(|(o, f)| (f.name.clone(), (o.clone(), (*f).clone()))).collect(),
    };
    refuse_unhandled(program, &names, &mnames)?;
    if candidates.is_empty() { return Ok(()); }

    let mut instances: Vec<Instance> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    rewrite_program(program, &candidates, &mut instances, &mut seen)?;
    let built = build_instances(&candidates, &mut instances, &mut seen, false)?;

    // A candidate nobody calls keeps its original error.
    let all: Vec<(String, bool, &FnDecl)> = candidates.free.iter().map(|(n, d)| (n.clone(), false, d))
        .chain(candidates.methods.iter().map(|(n, (_, d))| (n.clone(), true, d))).collect();
    // `pub` free functions, in the file or in a `mod` (its items are reached by their plain name).
    fn pub_fns(items: &[Item], out: &mut HashSet<String>) {
        for i in items {
            match i {
                Item::Fn(f) if f.is_pub => { out.insert(f.name.clone()); }
                Item::Mod(m) => pub_fns(&m.items, out),
                _ => {}
            }
        }
    }
    let mut top_pub: HashSet<String> = HashSet::new();
    pub_fns(&program.items, &mut top_pub);
    fn pub_methods(items: &[Item], out: &mut HashSet<String>) {
        for i in items {
            match i {
                Item::Struct(s) => out.extend(s.methods.iter().filter(|m| m.is_pub).map(|m| m.name.clone())),
                Item::Enum(e) => out.extend(e.methods.iter().filter(|m| m.is_pub).map(|m| m.name.clone())),
                Item::Mod(m) => pub_methods(&m.items, out),
                _ => {}
            }
        }
    }
    let mut top_pub_methods: HashSet<String> = HashSet::new();
    pub_methods(&program.items, &mut top_pub_methods);
    for (name, is_method, decl) in all {
        // A `pub` function may be called from another file: it stays, marked, whether or not this
        // file calls it.
        if !is_method && top_pub.contains(&name) { continue; }
        if is_method && top_pub_methods.contains(&name) { continue; }
        if !built.iter().any(|(b, o, _)| *b == name && o.is_some() == is_method) {
            let names: Vec<String> = implicit_indices(decl).iter().map(|&i| decl.params[i].name.clone()).collect();
            if let Some(u) = unbounded_use(&names, &decl.body, &[]) {
                if takes_over(&u.what) {
                    return Err(crate::infer_param_generics::unbounded_error(u, "parameter"));
                }
            }
            // Only a qualifier slot made it a candidate: nothing to specialise, nothing to refuse.
            if !decl.slot_quals.is_empty() { continue; }
            return Err(err(
                format!("cannot specialise `{name}`: it passes an untyped parameter to a function that needs its type, and is never called with an argument whose type is written"),
                decl.line, decl.col));
        }
    }

    // Replace each generic original by its instances, where it was declared.
    program.items = replace_originals(std::mem::take(&mut program.items), &candidates, &built, true);
    Ok(())
}

/// Rewrites, in the whole program, the calls of `candidates` (a candidate itself is rewritten
/// through its instances).
fn rewrite_program(program: &mut Program, candidates: &Candidates, instances: &mut Vec<Instance>, seen: &mut HashSet<String>) -> Result<(), ParseError> {
    // Calls from every ordinary function and method.
    let mut failure: Option<ParseError> = None;
    for_each_fn(&mut program.items, false, &HashSet::new(), &mut |f, site, in_mod, siblings| {
        if failure.is_some() { return; }
        let _ = in_mod;
        let is_candidate = match site {
            Site::Free => candidates.free.contains_key(&f.name),
            Site::Method => candidates.methods.contains_key(&f.name),
            Site::Other => false,
        };
        if is_candidate { return; }
        if let Err(e) = rewrite_calls(&f.params, &mut f.body, candidates, siblings, instances, seen) { failure = Some(e); }
    });
    if let Some(e) = failure { return Err(e); }
    // The file's own statements: one body, the top-level `let`s its locals.
    let mut slots: Vec<usize> = Vec::new();
    let mut stmts: Vec<Stmt> = Vec::new();
    for (i, item) in program.items.iter().enumerate() {
        match item {
            Item::Stmt(s) => { slots.push(i); stmts.push(s.clone()); }
            Item::Let(l) => { slots.push(i); stmts.push(Stmt::Let(l.clone())); }
            _ => {}
        }
    }
    rewrite_calls(&[], &mut stmts, candidates, &HashSet::new(), instances, seen)?;
    for (i, s) in slots.into_iter().zip(stmts) {
        match (&mut program.items[i], s) {
            (Item::Stmt(dst), s) => *dst = s,
            (Item::Let(dst), Stmt::Let(l)) => *dst = l,
            _ => {}
        }
    }
    Ok(())
}

/// The instances asked for, built from the candidates; building one may ask for more (an instance
/// calls other candidates with its own now-typed parameters).
fn build_instances(candidates: &Candidates, instances: &mut Vec<Instance>, seen: &mut HashSet<String>, imported: bool) -> Result<Vec<(String, Option<String>, FnDecl)>, ParseError> {
    let mut built: Vec<(String, Option<String>, FnDecl)> = Vec::new();
    let mut next = 0;
    while next < instances.len() {
        if instances.len() > MAX_INSTANCES {
            return Err(err(format!("too many specialisations of untyped-parameter functions (more than {MAX_INSTANCES})"), 1, 1));
        }
        let (base, mangled, types, owner, at) = {
            let i = &instances[next];
            (i.base.clone(), i.mangled.clone(), i.types.clone(), i.method_of.clone(), i.at)
        };
        next += 1;
        let decl = match &owner {
            Some(_) => &candidates.methods[&base].1,
            None => &candidates.free[&base],
        };
        let mut subst: HashMap<String, Type> = HashMap::new();
        for (idx, t) in &types {
            subst.insert(fresh_name(&decl.params[*idx].name), t.clone());
        }
        if decl.type_params.iter().any(|p| !subst.contains_key(p)) {
            return Err(err(format!("cannot specialise `{base}`: it also declares explicit type parameters"), decl.line, decl.col));
        }
        let mut spec = substitute_fn_decl(decl, &subst, &mangled);
        {
            let args: Vec<String> = types.iter().map(|(i, t)| format!("{} = {}", decl.params[*i].name, crate::bri::type_text(t).unwrap_or_else(|| describe_type(t)))).collect();
            let from = match (&owner, imported) {
                (Some(o), _) => format!("{o}.{base} (line {})", decl.line),
                (None, true) => format!("{base} of an imported file (line {})", decl.line),
                (None, false) => format!("{base} (line {})", decl.line),
            };
            let note = format!("# copy of {from} for {}, first called at line {}", args.join(", "), at.0);
            ORIGINS.with(|o| o.borrow_mut().insert(mangled.clone(), note));
        }
        spec.slot_quals.clear();
        spec.attrs.retain(|a| a.name != SPECIALISE_ONLY);
        let params = spec.params.clone();
        rewrite_calls(&params, &mut spec.body, candidates, &HashSet::new(), instances, seen)?;
        if std::env::var_os("BORING_SHOW_MONO").is_some() { show(&spec); }
        built.push((base, owner, spec));
    }
    Ok(built)
}

/// The type name written for a variable (`Counter`, `Counter'actor`, `mut Counter`).
fn written_type_name(t: &Type) -> Option<String> {
    let mut cur = t.without_mut();
    while let Type::Qualified(inner, _) = cur { cur = inner.without_mut(); }
    if let Type::Named(n) = cur { Some(n.clone()) } else { None }
}

/// `shadowed`: imported marked methods (name -> owner) whose name this file also declares for a method.
/// A call `r.name(..)` where `r` is written with the imported owner's type cannot tell which method is
/// meant: an error naming both, instead of a missing method in the generated Rust.
fn check_shadowed_calls(program: &mut Program, shadowed: &HashMap<String, String>) -> Result<(), ParseError> {
    if shadowed.is_empty() { return Ok(()); }
    fn scan(params: &[Param], body: &mut [Stmt], shadowed: &HashMap<String, String>) -> Option<ParseError> {
        let (env, _) = local_types(params, body);
        let mut found: Option<ParseError> = None;
        let mut on_expr = |e: &mut Expr| {
            if found.is_some() { return; }
            let ExprKind::MethodCall(recv, name, _) = &e.kind else { return };
            let (Some(owner), ExprKind::Var(v)) = (shadowed.get(name), &recv.kind) else { return };
            if env.get(v).and_then(written_type_name).as_deref() == Some(owner.as_str()) {
                found = Some(err(
                    format!("`{name}` of `{owner}` takes an untyped parameter, so it is specialised by the calling file, but this file also declares a method named `{name}`: the two cannot be told apart here; rename the local method"),
                    e.line, e.col));
            }
        };
        let mut on_stmt = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
        found
    }
    let mut failure: Option<ParseError> = None;
    for_each_fn(&mut program.items, false, &HashSet::new(), &mut |f, _, _, _| {
        if failure.is_some() { return; }
        let params = f.params.clone();
        failure = scan(&params, &mut f.body, shadowed);
    });
    if let Some(e) = failure { return Err(e); }
    let mut stmts: Vec<Stmt> = program.items.iter().filter_map(|i| match i {
        Item::Stmt(s) => Some(s.clone()),
        Item::Let(l) => Some(Stmt::Let(l.clone())),
        _ => None,
    }).collect();
    match scan(&[], &mut stmts, shadowed) { Some(e) => Err(e), None => Ok(()) }
}

/// Marks a `pub` function that exists to be specialised by its callers, in this file or another:
/// the original is kept (a caller's file copies it) but never emitted as Rust.
pub const SPECIALISE_ONLY: &str = "specialise_only";

pub fn is_specialise_only(f: &FnDecl) -> bool {
    f.attrs.iter().any(|a| a.name == SPECIALISE_ONLY)
}

/// Specialises, in `program`, the calls of marked functions of the files it imports: each needed
/// instance is a copy of the imported function with the argument types written, added to this
/// file as a private function (the types it names come in through the same `use`).
pub fn specialize_imported(program: &mut Program, imported: &[Item]) -> Result<(), ParseError> {
    let mut local: HashMap<String, usize> = HashMap::new();
    free_name_counts(&program.items, &mut local);
    let mut found: HashMap<String, Vec<&FnDecl>> = HashMap::new();
    fn marked<'a>(items: &'a [Item], local: &HashMap<String, usize>, out: &mut HashMap<String, Vec<&'a FnDecl>>) {
        for item in items {
            match item {
                Item::Fn(f) if is_specialise_only(f) && !local.contains_key(&f.name) => out.entry(f.name.clone()).or_default().push(f),
                Item::Mod(m) => marked(&m.items, local, out),
                _ => {}
            }
        }
    }
    marked(imported, &local, &mut found);
    // A name imported from two files is ambiguous: neither is taken.
    let free: HashMap<String, FnDecl> = found.into_iter()
        .filter(|(_, v)| v.len() == 1)
        .map(|(n, v)| (n, v[0].clone()))
        .collect();
    // The marked methods of imported structs and enums; a name a local method also has is not taken.
    let mut local_methods: HashMap<String, usize> = HashMap::new();
    method_name_counts(&program.items, &mut local_methods);
    let mut found_methods: HashMap<String, Vec<(String, &FnDecl)>> = HashMap::new();
    fn marked_methods<'a>(items: &'a [Item], local: &HashMap<String, usize>, out: &mut HashMap<String, Vec<(String, &'a FnDecl)>>) {
        for item in items {
            let (owner, methods): (&str, &Vec<FnDecl>) = match item {
                Item::Struct(s) => (s.name.as_str(), &s.methods),
                Item::Enum(e) => (e.name.as_str(), &e.methods),
                Item::Mod(m) => { marked_methods(&m.items, local, out); continue; }
                _ => continue,
            };
            for m in methods {
                if is_specialise_only(m) && !local.contains_key(&m.name) {
                    out.entry(m.name.clone()).or_default().push((owner.to_string(), m));
                }
            }
        }
    }
    marked_methods(imported, &local_methods, &mut found_methods);
    // A marked imported method this file also declares a method of the same name for: a call on a
    // receiver written with the imported type is ambiguous (the local method hides it), never silently
    // left for rustc.
    let mut shadowed: HashMap<String, String> = HashMap::new();
    fn shadowed_methods(items: &[Item], local: &HashMap<String, usize>, out: &mut HashMap<String, String>) {
        for item in items {
            let (owner, methods): (&str, &Vec<FnDecl>) = match item {
                Item::Struct(s) => (s.name.as_str(), &s.methods),
                Item::Enum(e) => (e.name.as_str(), &e.methods),
                Item::Mod(m) => { shadowed_methods(&m.items, local, out); continue; }
                _ => continue,
            };
            for m in methods {
                if is_specialise_only(m) && local.contains_key(&m.name) { out.insert(m.name.clone(), owner.to_string()); }
            }
        }
    }
    shadowed_methods(imported, &local_methods, &mut shadowed);
    check_shadowed_calls(program, &shadowed)?;
    let methods: HashMap<String, (String, FnDecl)> = found_methods.into_iter()
        .filter(|(_, v)| v.len() == 1)
        .map(|(n, v)| (n, (v[0].0.clone(), v[0].1.clone())))
        .collect();
    if free.is_empty() && methods.is_empty() { return Ok(()); }
    let candidates = Candidates { free, methods };
    let mut instances: Vec<Instance> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    rewrite_program(program, &candidates, &mut instances, &mut seen)?;
    let built = build_instances(&candidates, &mut instances, &mut seen, true)?;
    // The copies of an imported method go in an `ext` of its type, in this file.
    let mut exts: Vec<(String, Vec<FnDecl>)> = Vec::new();
    for (_, owner, mut spec) in built {
        spec.is_pub = false;
        match owner {
            None => program.items.push(Item::Fn(spec)),
            Some(o) => match exts.iter_mut().find(|(n, _)| *n == o) {
                Some((_, v)) => v.push(spec),
                None => exts.push((o, vec![spec])),
            },
        }
    }
    for (type_name, methods) in exts {
        let (line, col) = (methods[0].line, methods[0].col);
        program.items.push(Item::Ext(ExtDecl {
            type_name, type_args: Vec::new(), type_params: Vec::new(), where_clause: Vec::new(),
            traits: Vec::new(), methods, setters: Vec::new(), conversions: Vec::new(), assoc_type_defs: Vec::new(),
            line, col,
        }));
    }
    Ok(())
}

/// `methods` of `owner` with each taken-over method replaced by its instances.
fn swap_methods(methods: Vec<FnDecl>, owner: &str, candidates: &Candidates, built: &[(String, Option<String>, FnDecl)], top: bool) -> Vec<FnDecl> {
    let mut out = Vec::new();
    for m in methods {
        let taken = candidates.methods.get(&m.name).is_some_and(|(o, _)| o == owner);
        if !taken { out.push(m); continue; }
        let mut any = false;
        for (b, o, spec) in built {
            if *b == m.name && o.as_deref() == Some(owner) { any = true; out.push(spec.clone()); }
        }
        // A `pub` method of a top-level type may be called from another file: it stays, marked.
        let _ = top;
        if !any || m.is_pub {
            let mut m = m;
            if m.is_pub && !is_specialise_only(&m) {
                m.attrs.push(Attr { name: SPECIALISE_ONLY.into(), args: vec![], line: m.line, col: m.col });
            }
            out.push(m);
        }
    }
    out
}
