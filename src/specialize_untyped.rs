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

//! SPIKE of docs/design-notes/typed-monomorphization-draft.md (phase 1): a free function whose
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
/// a function in `names`.
fn passes_param_to(f: &FnDecl, all: &[&FnDecl], names: &HashSet<String>) -> bool {
    let own: HashSet<&str> = implicit_indices(f).iter().map(|&i| f.params[i].name.as_str()).collect();
    let mut hit = false;
    let mut body = f.body.clone();
    let mut on_expr = |e: &mut Expr| {
        let ExprKind::Call(callee, args) = &e.kind else { return };
        let ExprKind::Var(g) = &callee.kind else { return };
        if !names.contains(g) { return; }
        let Some(decl) = all.iter().find(|d| d.name == *g) else { return };
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
        let (env, ambiguous) = local_types(f);
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
        let (env, ambiguous) = local_types(f);
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

/// Annotated types visible in a function: its typed parameters and its annotated `let`s. A name
/// bound more than once with anything but one and the same written type (a shadowing `let`, an
/// unannotated rebinding, a `for` variable) is *ambiguous*: it is reported, never guessed.
fn local_types(f: &FnDecl) -> (HashMap<String, Type>, HashSet<String>) {
    let mut bindings: HashMap<String, Vec<Option<Type>>> = HashMap::new();
    for p in &f.params {
        let t = p.ty.clone().filter(|t| !matches!(t, Type::TypeParam(_)));
        bindings.entry(p.name.clone()).or_default().push(t);
    }
    let mut body = f.body.clone();
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

struct Instance {
    base: String,
    mangled: String,
    /// `(parameter index, concrete type)`
    types: Vec<(usize, Type)>,
}

/// Rewrites the calls of `candidates` found in `f`'s body, recording the instances they need.
fn rewrite_calls(
    f: &mut FnDecl,
    candidates: &HashMap<String, FnDecl>,
    instances: &mut Vec<Instance>,
    seen: &mut HashSet<String>,
) -> Result<(), ParseError> {
    let (env, ambiguous) = local_types(f);
    let mut failure: Option<ParseError> = None;
    let mut new_instances: Vec<Instance> = Vec::new();
    {
        let mut on_expr = |e: &mut Expr| {
            if failure.is_some() { return; }
            let ExprKind::Call(callee, args) = &mut e.kind else { return };
            let ExprKind::Var(name) = &callee.kind else { return };
            let Some(decl) = candidates.get(name) else { return };
            let name = name.clone();
            let mut types: Vec<(usize, Type)> = Vec::new();
            for i in implicit_indices(decl) {
                let pname = &decl.params[i].name;
                // The argument of this parameter: the one labelled with its name, else the positional one.
                let arg = args.iter().find(|a| a.label.as_deref() == Some(pname.as_str()))
                    .or_else(|| args.get(i).filter(|a| a.label.is_none()));
                let Some(arg) = arg else {
                    failure = Some(err(format!("cannot specialise `{name}`: no argument for the untyped parameter `{pname}`"), e.line, e.col));
                    return;
                };
                if arg.spread {
                    failure = Some(err(format!("cannot specialise `{name}`: a spread argument cannot give the type of `{pname}`"), arg.value.line, arg.value.col));
                    return;
                }
                if let ExprKind::Var(v) = &arg.value.kind {
                    if ambiguous.contains(v) {
                        failure = Some(err(
                            format!("cannot specialise `{name}` for this call: `{v}` is bound more than once in this function with different (or unwritten) types, so the type of the argument given to `{pname}` is ambiguous; use a variable bound once, with its type written"),
                            arg.value.line, arg.value.col));
                        return;
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
                                    failure = Some(err(
                                        format!("the parameter `{pname}{}` of `{name}` expects that qualifier, but this argument is written `{}`", qual_text(want), qual_text(have)),
                                        arg.value.line, arg.value.col));
                                    return;
                                }
                                Some(_) => t,
                                None => Type::Qualified(Box::new(t), want.clone()),
                            },
                        };
                        types.push((i, t));
                    }
                    _ => {
                        failure = Some(err(
                            format!("cannot specialise `{name}` for this call: the type of the argument given to the untyped parameter `{pname}` is not known here; pass a variable whose type is written (`let Counter'actor c = ...`)"),
                            arg.value.line, arg.value.col));
                        return;
                    }
                }
            }
            let key: Vec<String> = types.iter().map(|(_, t)| slug(t).unwrap_or_default()).collect();
            let mangled = format!("{name}_{}", key.join("_"));
            if let ExprKind::Var(n) = &mut callee.kind { *n = mangled.clone(); }
            if seen.insert(mangled.clone()) {
                new_instances.push(Instance { base: name, mangled, types });
            }
        };
        let mut on_stmt = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut f.body);
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

/// Every function with an untyped parameter used in a way that needs its type (`takes_over`) that the
/// pass does not specialise keeps the error it would have had without it: one declared inside a `mod`,
/// or whose name is declared more than once (overloads). Without this, the parser's relaxed check
/// would let it through to rustc.
fn refuse_unhandled(items: &[Item], handled: &HashSet<String>, in_mod: bool) -> Result<(), ParseError> {
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for item in items {
        if let Item::Fn(f) = item { *seen.entry(f.name.as_str()).or_default() += 1; }
    }
    for item in items {
        match item {
            Item::Fn(f) if f.qualifier.is_none() && !f.task && !f.stream && !f.is_native => {
                let idx = implicit_indices(f);
                if idx.is_empty() { continue; }
                let names: Vec<String> = idx.iter().map(|&i| f.params[i].name.clone()).collect();
                let Some(u) = unbounded_use(&names, &f.body, &[]) else { continue };
                if !takes_over(&u.what) { continue; }
                let handled_here = !in_mod && seen.get(f.name.as_str()) == Some(&1) && handled.contains(&f.name);
                if !handled_here { return Err(crate::infer_param_generics::unbounded_error(u, "parameter")); }
            }
            Item::Mod(m) => refuse_unhandled(&m.items, handled, true)?,
            _ => {}
        }
    }
    Ok(())
}

pub fn specialize_method_users(program: &mut Program) -> Result<(), ParseError> {
    check_explicit_slots(program)?;
    infer_qualified_generic_args(program);
    let mut count: HashMap<String, usize> = HashMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item { *count.entry(f.name.clone()).or_default() += 1; }
    }
    let eligible: Vec<&FnDecl> = program.items.iter()
        .filter_map(|it| match it {
            Item::Fn(f) if count.get(&f.name) == Some(&1)
                && f.qualifier.is_none() && !f.task && !f.stream && !f.is_native
                && !implicit_indices(f).is_empty() => Some(f),
            _ => None,
        })
        .collect();
    // A function is taken over when it calls a method on an untyped parameter, or passes one of
    // its own untyped parameters to a function that is (a fixpoint, like the bound propagation).
    let mut names: HashSet<String> = eligible.iter().filter(|f| needs_specialisation(f)).map(|f| f.name.clone()).collect();
    loop {
        let mut grew = false;
        for f in &eligible {
            if !names.contains(&f.name) && passes_param_to(f, &eligible, &names) {
                names.insert(f.name.clone());
                grew = true;
            }
        }
        if !grew { break; }
    }
    let candidates: HashMap<String, FnDecl> = eligible.iter()
        .filter(|f| names.contains(&f.name))
        .map(|f| (f.name.clone(), (*f).clone()))
        .collect();
    refuse_unhandled(&program.items, &names, false)?;
    if candidates.is_empty() { return Ok(()); }

    let mut instances: Vec<Instance> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Calls from every ordinary function (a candidate itself is rewritten through its instances).
    for item in program.items.iter_mut() {
        if let Item::Fn(f) = item {
            if candidates.contains_key(&f.name) { continue; }
            rewrite_calls(f, &candidates, &mut instances, &mut seen)?;
        }
    }

    // Instances may call other candidates with their own now-typed parameters.
    let mut built: Vec<(String, FnDecl)> = Vec::new();
    let mut next = 0;
    while next < instances.len() {
        if instances.len() > MAX_INSTANCES {
            return Err(err(format!("too many specialisations of untyped-parameter functions (more than {MAX_INSTANCES})"), 1, 1));
        }
        let (base, mangled, types) = {
            let i = &instances[next];
            (i.base.clone(), i.mangled.clone(), i.types.clone())
        };
        next += 1;
        let decl = &candidates[&base];
        let mut subst: HashMap<String, Type> = HashMap::new();
        for (idx, t) in &types {
            subst.insert(fresh_name(&decl.params[*idx].name), t.clone());
        }
        if decl.type_params.iter().any(|p| !subst.contains_key(p)) {
            return Err(err(format!("cannot specialise `{base}`: it also declares explicit type parameters"), decl.line, decl.col));
        }
        let mut spec = substitute_fn_decl(decl, &subst, &mangled);
        spec.slot_quals.clear();
        rewrite_calls(&mut spec, &candidates, &mut instances, &mut seen)?;
        if std::env::var_os("BORING_SHOW_MONO").is_some() { show(&spec); }
        built.push((base, spec));
    }

    // A candidate nobody calls keeps its original error.
    for (name, decl) in &candidates {
        if !built.iter().any(|(b, _)| b == name) {
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

    // Replace each generic original by its instances.
    let mut out: Vec<Item> = Vec::new();
    for item in std::mem::take(&mut program.items) {
        match item {
            Item::Fn(f) if candidates.contains_key(&f.name) => {
                let mut any = false;
                for (b, spec) in &built {
                    if *b == f.name { any = true; out.push(Item::Fn(spec.clone())); }
                }
                if !any { out.push(Item::Fn(f)); }
            }
            other => out.push(other),
        }
    }
    program.items = out;
    Ok(())
}
