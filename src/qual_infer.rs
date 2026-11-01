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

//! Inference of qualifier arguments at construction (docs/design-notes/qualifier-generic-params.md,
//! section 8): `Board(cells = [a, b])` is `Board<elem'actor>` when the construction itself
//! determines the parameter `elem` of `struct Board<'elem>`.
//!
//! Evidence, in this order (two that disagree are an error, never a guess):
//! 1. explicit arguments (`Board<elem'actor>(..)`): untouched, this pass never sees them as bare;
//! 2. the expected type (`let Board<elem'actor> b = Board()`, a field, a parameter or a return
//!    type of a known `Board<..>` type);
//! 3. the *written* qualifier of the arguments that feed a `T'elem` slot (`T'elem`, `[mut T'elem]`,
//!    `{K = mut V'elem}`): a declared `Point'actor` binding, parameter or field. A qualifier that
//!    is itself inferred (`let p = Point(1, 2)`) or a fresh literal is no evidence;
//! 4. otherwise the construction stays bare and `check_bare_qual_uses` reports "cannot infer".
//!
//! The pass rewrites a resolved construction to the specialised instance's name (the exact
//! rewrite an explicit `<..>` gets from `monomorphize::rewrite_qual_program`) and reports the
//! instance so it is specialised. It is a function of one file's AST plus the declarations of the
//! files it `use`s, so `boring run` (diagnostics only, through the parser) and `boring build`
//! (rewrite, through the transpiler) reach the same answer.

use crate::ast::*;
use crate::ast_walk::Walker;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

static QUAL_SEEN: AtomicBool = AtomicBool::new(false);

/// A declaration with a qualifier parameter was read somewhere in this process; until then the
/// transpiler does not pay for a declaration table per file.
pub(crate) fn note_qual_decl() { QUAL_SEEN.store(true, Ordering::Relaxed); }
pub(crate) fn qual_decls_seen() -> bool { QUAL_SEEN.load(Ordering::Relaxed) }

/// `(line, col)` of the constructed type's name -> error message.
pub(crate) type Diags = HashMap<(usize, usize), String>;

/// Instances found: `(declaration name, positional arguments)`.
pub(crate) type Resolved = Vec<(String, Vec<Type>)>;

// ─── Declaration table ──────────────────────────────────────────────────────────

type Params = Vec<(Option<String>, Type)>;

enum Ctor {
    Struct { fields: Params, inits: Vec<Params> },
    Enum(HashMap<String, Params>),
}

struct QParam {
    name: String,
    bound: Vec<QualChain>,
    default: Option<QualChain>,
}

struct QualDecl {
    /// One entry per declared parameter: `Some((name, bound))` for a qualifier parameter.
    params: Vec<Option<QParam>>,
    pure: bool,
}

#[derive(Default)]
struct Decls {
    /// `None` when declared more than once (ambiguous by name).
    ctors: HashMap<String, Option<Ctor>>,
    quals: HashMap<String, QualDecl>,
    fns: HashMap<String, Option<Params>>,
}

fn add_unique<V>(map: &mut HashMap<String, Option<V>>, name: &str, v: V) {
    match map.get(name) {
        None => { map.insert(name.to_string(), Some(v)); }
        Some(_) => { map.insert(name.to_string(), None); }
    }
}

impl Decls {
    fn collect(&mut self, items: &[Item]) {
        for it in items {
            match it {
                Item::Struct(s) => {
                    let fields = s.fields.iter().map(|f| (Some(f.name.clone()), f.ty.clone())).collect();
                    let inits = s.inits.iter().map(|i| {
                        i.params.iter().filter_map(|p| p.ty.clone().map(|t| (Some(p.name.clone()), t))).collect::<Params>()
                    }).collect();
                    add_unique(&mut self.ctors, &s.name, Ctor::Struct { fields, inits });
                    self.add_qual(&s.name, &s.type_params);
                }
                Item::Enum(e) => {
                    let variants = e.variants.iter().map(|v| {
                        (v.name.clone(), v.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect::<Params>())
                    }).collect();
                    add_unique(&mut self.ctors, &e.name, Ctor::Enum(variants));
                    self.add_qual(&e.name, &e.type_params);
                }
                Item::Fn(f) => {
                    let params = f.params.iter().filter_map(|p| p.ty.clone().map(|t| (Some(p.name.clone()), t))).collect::<Params>();
                    // A parameter without a type would shift positions: treat as unknown.
                    let complete = f.params.iter().all(|p| p.ty.is_some());
                    if complete { add_unique(&mut self.fns, &f.name, params); } else { self.fns.insert(f.name.clone(), None); }
                }
                Item::Mod(m) => self.collect(&m.items),
                _ => {}
            }
        }
    }

    fn add_qual(&mut self, name: &str, type_params: &[String]) {
        if !type_params.iter().any(|p| qual_param_name(p).is_some()) { return; }
        note_qual_decl();
        let params: Vec<_> = type_params.iter()
            .map(|p| qual_param_decl(p).map(|(n, b)| QParam { name: n.to_string(), bound: b, default: qual_param_default(p) }))
            .collect();
        let pure = params.iter().all(Option::is_some);
        // Declared twice: not resolvable by name.
        if self.quals.contains_key(name) { self.quals.remove(name); self.ctors.insert(name.to_string(), None); return; }
        self.quals.insert(name.to_string(), QualDecl { params, pure });
    }

    fn of_program(program: &Program) -> Self {
        let mut d = Self::default();
        d.collect(&program.items);
        if let Some((dir, deps)) = crate::infer_actor_params::import_context() {
            let mut visited = HashSet::new();
            crate::infer_actor_params::visit_imported_files(&program.items, &dir, &deps, &mut visited, &mut |file| {
                d.collect(&file.items);
            });
        }
        d
    }

    fn struct_fields(&self, name: &str) -> Vec<(String, Type)> {
        match self.ctors.get(name) {
            Some(Some(Ctor::Struct { fields, .. })) => fields.iter().filter_map(|(n, t)| n.clone().map(|n| (n, t.clone()))).collect(),
            _ => Vec::new(),
        }
    }
}

// ─── Declared types of names ────────────────────────────────────────────────────

/// Types written on the bindings, parameters and fields visible in one function body. A name
/// declared with two different types, or without a written type, is unknown (`None`): it gives
/// no evidence.
#[derive(Default, Clone)]
struct Declared(HashMap<String, Option<Type>>);

impl Declared {
    fn declare(&mut self, name: &str, ty: Option<&Type>) {
        match self.0.get(name) {
            Some(Some(old)) if Some(old) == ty => {}
            Some(_) => { self.0.insert(name.to_string(), None); }
            None => { self.0.insert(name.to_string(), ty.cloned()); }
        }
    }

    fn get(&self, name: &str) -> Option<&Type> {
        self.0.get(name).and_then(|t| t.as_ref())
    }

    fn of_expr(&self, e: &Expr) -> Option<&Type> {
        match &e.kind {
            ExprKind::Var(n) => self.get(n),
            ExprKind::Field(r, n) if matches!(&r.kind, ExprKind::Var(s) if s == "self") => self.get(n),
            _ => None,
        }
    }
}

fn collect_declared(declared: &mut Declared, params: &[(String, Option<&Type>)], body: &mut [Stmt]) {
    for (n, t) in params { declared.declare(n, *t); }
    let cell = RefCell::new(std::mem::take(declared));
    {
        let mut on_expr = |_: &mut Expr| {};
        let mut on_stmt = |s: &mut Stmt| match s {
            Stmt::Let(l) => cell.borrow_mut().declare(&l.name, l.ty.as_ref()),
            Stmt::LetDestructure(l) => { for b in &l.bindings { cell.borrow_mut().declare(&b.name, None); } }
            Stmt::For(f) => { for v in &f.vars { cell.borrow_mut().declare(v, None); } }
            _ => {}
        };
        let mut w = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        w.stmts(body);
    }
    *declared = cell.into_inner();
}

// ─── Evidence from written qualifiers ───────────────────────────────────────────

/// `(parameter, qualifier, where it was written)`.
type Evidence = Vec<(String, QualChain, String)>;

fn peel(mut t: &Type) -> &Type {
    while let Type::Mut(i) | Type::Optional(i) = t { t = i; }
    t
}

/// The qualifier chain written on a type (`Point'actor'observed` → `[Actor, Observed]`), when it
/// is one a qualifier parameter may take; a borrow, a union or an invalid chain gives none.
fn written_chain(ty: &Type) -> Option<QualChain> {
    let mut chain = Vec::new();
    let mut t = ty;
    while let Type::Qualified(inner, q) = t { chain.push(q.clone()); t = inner; }
    chain.reverse();
    OwnerQual::chain_name(&chain)?;
    OwnerQual::check_chain(&chain).ok()?;
    Some(chain)
}

/// The declared type of an argument against the parameter slot type `decl`.
fn from_ty(decl: &Type, arg: &Type, desc: &str, out: &mut Evidence) {
    let (d, a) = (peel(decl), peel(arg));
    match (d, a) {
        (Type::Qualified(_, OwnerQual::Param(p)), Type::Qualified(..)) => {
            if let Some(chain) = written_chain(a) { out.push((p.clone(), chain, desc.to_string())); }
        }
        (Type::Array(di), Type::Array(ai)) | (Type::Set(di), Type::Set(ai)) => from_ty(di, ai, desc, out),
        (Type::Dict(dk, dv), Type::Dict(ak, av)) => { from_ty(dk, ak, desc, out); from_ty(dv, av, desc, out); }
        (Type::Tuple(ds), Type::Tuple(az)) => { for (x, y) in ds.iter().zip(az) { from_ty(x, y, desc, out); } }
        _ => {}
    }
}

/// An argument expression against the parameter slot type `decl`: a declared name, or a literal
/// whose elements are declared names.
fn from_expr(decl: &Type, e: &Expr, declared: &Declared, out: &mut Evidence) {
    let d = peel(decl);
    match &e.kind {
        ExprKind::Var(_) | ExprKind::Field(..) => {
            if let Some(t) = declared.of_expr(e) {
                let desc = match &e.kind {
                    ExprKind::Var(n) => format!("declared on `{n}`"),
                    ExprKind::Field(_, n) => format!("declared on `self.{n}`"),
                    _ => unreachable!(),
                };
                from_ty(d, t, &desc, out);
            }
        }
        ExprKind::Array(items) | ExprKind::Set(items) => match d {
            Type::Array(di) | Type::Set(di) => { for it in items { from_expr(di, it, declared, out); } }
            _ => {}
        },
        ExprKind::Dict(pairs) => {
            if let Type::Dict(dk, dv) = d {
                for (k, v) in pairs { from_expr(dk, k, declared, out); from_expr(dv, v, declared, out); }
            }
        }
        ExprKind::Tuple(items) => {
            if let Type::Tuple(ds) = d { for (x, it) in ds.iter().zip(items) { from_expr(x, it, declared, out); } }
        }
        _ => {}
    }
}

// ─── Construction shapes ────────────────────────────────────────────────────────

enum Shape<'a> {
    /// `Board(..)`
    Struct(&'a [Arg]),
    /// `Slot.Full(..)`: an enum variant (or a type-level method).
    Member(&'a str, &'a [Arg]),
    /// `Slot.Empty`
    Bare,
}

/// `(constructed name, position of that name, shape)` of an expression written as a
/// construction of a named type.
fn view(e: &Expr) -> Option<(&str, (usize, usize), Shape<'_>)> {
    fn var(x: &Expr) -> Option<(&str, (usize, usize))> {
        match &x.kind { ExprKind::Var(n) => Some((n.as_str(), (x.line, x.col))), _ => None }
    }
    match &e.kind {
        ExprKind::Call(c, args) => match &c.kind {
            ExprKind::Var(n) => Some((n, (c.line, c.col), Shape::Struct(args))),
            ExprKind::Field(r, m) => var(r).map(|(n, p)| (n, p, Shape::Member(m, args))),
            _ => None,
        },
        ExprKind::MethodCall(r, m, args) => var(r).map(|(n, p)| (n, p, Shape::Member(m, args))),
        ExprKind::Field(r, _) => var(r).map(|(n, p)| (n, p, Shape::Bare)),
        _ => None,
    }
}

/// The parameters a constructor-shaped call binds its arguments to.
fn ctor_params<'a>(decls: &'a Decls, base: &str, shape: &Shape<'_>) -> Option<&'a Params> {
    match (decls.ctors.get(base)?.as_ref()?, shape) {
        (Ctor::Struct { fields, inits }, Shape::Struct(args)) => {
            if inits.is_empty() { return Some(fields); }
            let fits = |ps: &&Params| {
                let pos = args.iter().filter(|a| a.label.is_none() && !a.spread && !a.default_rest).count();
                let labels_ok = args.iter().filter_map(|a| a.label.as_ref()).all(|l| ps.iter().any(|(n, _)| n.as_ref() == Some(l)));
                labels_ok && pos <= ps.len()
            };
            let mut it = inits.iter().filter(fits);
            let first = it.next()?;
            if it.next().is_some() { None } else { Some(first) }
        }
        (Ctor::Enum(variants), Shape::Member(v, _)) => variants.get(*v),
        _ => None,
    }
}

fn shape_args<'a>(shape: &Shape<'a>) -> &'a [Arg] {
    match shape { Shape::Struct(a) | Shape::Member(_, a) => a, Shape::Bare => &[] }
}

/// Pairs each argument with the parameter type it binds to.
fn bind_args<'a>(args: &'a [Arg], params: &'a Params) -> Vec<(&'a Arg, &'a Type)> {
    let mut out = Vec::new();
    let mut pos = 0;
    for a in args {
        if a.spread || a.default_rest { continue; }
        match &a.label {
            Some(l) => {
                if let Some((_, t)) = params.iter().find(|(n, _)| n.as_ref() == Some(l)) { out.push((a, t)); }
            }
            None => {
                if let Some((_, t)) = params.get(pos) { out.push((a, t)); }
                pos += 1;
            }
        }
    }
    out
}

// ─── The pass ───────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Out {
    resolved: Resolved,
    diags: Diags,
}

struct Ctx<'a> {
    decls: &'a Decls,
    out: RefCell<Out>,
}

fn qual_args_of(ty: &Type) -> Option<(&str, &[Type])> {
    let mut t = ty;
    while let Type::Mut(i) | Type::Qualified(i, _) | Type::Optional(i) = t { t = i; }
    match t {
        Type::Generic(base, args) if args.iter().any(|a| matches!(a, Type::QualArg(_))) => Some((base, args)),
        _ => None,
    }
}

fn peel_all(mut t: &Type) -> &Type {
    while let Type::Mut(i) | Type::Qualified(i, _) | Type::Optional(i) = t { t = i; }
    t
}

impl Ctx<'_> {
    /// `e` is being stored where `expected` is known: a bare construction of the expected `Board<..>`
    /// takes that instance; an array literal passes the element type down.
    fn resolve_expected(&self, e: &mut Expr, expected: &Type, declared: &Declared) {
        match peel_all(expected) {
            Type::Array(inner) => {
                if let ExprKind::Array(items) = &mut e.kind {
                    for it in items { self.resolve_expected(it, inner, declared); }
                }
            }
            _ => {
                if let Some((base, args)) = qual_args_of(expected) {
                    self.resolve_site(e, Some((base, args)), declared);
                } else {
                    // A bare expected type names no instance: the construction stands on its own
                    // evidence, then its default (a field default is reached only from here).
                    self.resolve_site(e, None, declared);
                }
            }
        }
    }

    /// Resolves the construction `e` if it is a bare construction of a qualifier-generic type.
    fn resolve_site(&self, e: &mut Expr, expected: Option<(&str, &[Type])>, declared: &Declared) {
        let Some((base, pos, shape)) = view(e) else { return };
        let Some(qd) = self.decls.quals.get(base) else { return };
        if let Some((eb, _)) = expected { if eb != base { return; } }
        let expected_args = expected.map(|(_, a)| a);
        let base = base.to_string();
        match self.infer(&base, qd, &shape, expected_args, declared) {
            Ok(Some(args)) => {
                if let Some(m) = crate::transpiler::monomorphize::mangled_name(&base, &args) {
                    crate::transpiler::monomorphize::rename_ctor(e, &base, &m);
                    self.out.borrow_mut().resolved.push((base, args));
                }
            }
            Ok(None) => {}
            Err(msg) => { self.out.borrow_mut().diags.insert(pos, msg); }
        }
    }

    fn infer(&self, base: &str, qd: &QualDecl, shape: &Shape<'_>, expected: Option<&[Type]>, declared: &Declared) -> Result<Option<Vec<Type>>, String> {
        // The expected type already names every argument (a declaration that mixes type and
        // qualifier parameters can only be resolved this way: type arguments are not inferred).
        let expected = expected.filter(|a| a.len() == qd.params.len());
        let mut ev = Evidence::new();
        if qd.pure {
            if let Some(params) = ctor_params(self.decls, base, shape) {
                for (arg, ty) in bind_args(shape_args(shape), params) {
                    from_expr(ty, &arg.value, declared, &mut ev);
                }
            }
        }
        if !qd.pure && expected.is_none() { return Ok(None); }
        let mut args = Vec::with_capacity(qd.params.len());
        for (i, p) in qd.params.iter().enumerate() {
            let Some(QParam { name, bound, default }) = p else {
                // A type parameter: only the expected type can supply it.
                match expected.and_then(|a| a.get(i)) { Some(t) => { args.push(t.clone()); continue; } None => return Ok(None) }
            };
            let shown = |c: &QualChain| OwnerQual::chain_name(c).unwrap_or_else(|| "?".into());
            let from_expected = match expected.and_then(|a| a.get(i)) {
                Some(Type::QualArg(q)) => Some(q.clone()),
                _ => None,
            };
            let mine: Vec<&(String, QualChain, String)> = ev.iter().filter(|(n, _, _)| n == name).collect();
            let mut chosen: Option<(QualChain, String)> = from_expected.map(|q| (q, "the expected type".to_string()));
            for (_, q, desc) in mine {
                match &chosen {
                    None => chosen = Some((q.clone(), desc.clone())),
                    Some((c, cdesc)) if c != q => {
                        return Err(format!(
                            "conflicting qualifiers for `{name}` of `{base}`: `'{}` ({cdesc}) and `'{}` ({desc}); write `{base}<{name}'...>` to choose",
                            shown(c), shown(q)));
                    }
                    _ => {}
                }
            }
            match chosen {
                Some((q, desc)) => {
                    if !bound.contains(&q) {
                        let allowed: Vec<String> = bound.iter().map(&shown).collect();
                        return Err(format!(
                            "qualifier `'{}` ({desc}) is outside the bound of `{name}` in `{base}` (allowed: {})",
                            shown(&q), allowed.join(", ")));
                    }
                    args.push(Type::QualArg(q));
                }
                // The declared default comes last: it never overrides evidence, and is no evidence.
                None => match default {
                    Some(d) => args.push(Type::QualArg(d.clone())),
                    None => return Ok(None),
                },
            }
        }
        Ok(Some(args))
    }

    /// The walk over one function body (or one top-level statement).
    fn run_body(&self, declared: &Declared, ret: Option<&Type>, body: &mut [Stmt]) {
        // A tail expression takes the function's return type.
        if let (Some(ret), Some(Stmt::Expr(e))) = (ret, body.last_mut()) { self.resolve_expected(e, ret, declared); }
        let mut on_stmt = |s: &mut Stmt| match s {
            Stmt::Let(l) => {
                if let (Some(t), Some(v)) = (&l.ty, &mut l.value) { self.resolve_expected(v, t, declared); }
            }
            Stmt::Return(r) => {
                if let (Some(t), Some(v)) = (ret, &mut r.value) { self.resolve_expected(v, t, declared); }
            }
            _ => {}
        };
        let mut on_expr = |e: &mut Expr| match &mut e.kind {
            ExprKind::Assign(l, r) => {
                if let Some(t) = declared.of_expr(l) { let t = t.clone(); self.resolve_expected(r, &t, declared); }
            }
            ExprKind::Call(..) | ExprKind::MethodCall(..) => {
                // Arguments take the parameter types of the callee (a function, a struct or an
                // enum variant), then the call itself may be a construction.
                let bound: Vec<(usize, Type)> = {
                    let params = match &e.kind {
                        ExprKind::Call(c, args) => match &c.kind {
                            ExprKind::Var(n) => self.callee_params(n, args),
                            _ => None,
                        }.or_else(|| view(e).and_then(|(b, _, sh)| ctor_params(self.decls, b, &sh))),
                        _ => view(e).and_then(|(b, _, sh)| ctor_params(self.decls, b, &sh)),
                    };
                    match (params, shape_args_of(e)) {
                        (Some(ps), Some(args)) => {
                            bind_args(args, ps).into_iter()
                                .filter_map(|(a, t)| args.iter().position(|x| std::ptr::eq(x, a)).map(|i| (i, t.clone())))
                                .collect()
                        }
                        _ => Vec::new(),
                    }
                };
                self.resolve_site(e, None, declared);
                let args = match &mut e.kind {
                    ExprKind::Call(_, a) | ExprKind::MethodCall(_, _, a) => Some(a),
                    _ => None,
                };
                if let Some(args) = args {
                    for (i, t) in bound { if let Some(a) = args.get_mut(i) { self.resolve_expected(&mut a.value, &t, declared); } }
                }
            }
            _ => {}
        };
        let mut w = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        w.stmts(body);
    }

    fn callee_params(&self, name: &str, args: &[Arg]) -> Option<&Params> {
        match self.decls.fns.get(name) {
            Some(Some(p)) => Some(p),
            Some(None) => None,
            None => ctor_params(self.decls, name, &Shape::Struct(args)),
        }
    }

    fn run_fn(&self, f: &mut FnDecl, fields: &[(String, Type)], globals: &Declared) {
        let mut declared = globals.clone();
        for (n, t) in fields { declared.declare(n, Some(t)); }
        // Locals and parameters shadow fields and globals instead of conflicting with them.
        for p in &f.params { declared.0.remove(&p.name); }
        let mut locals = Declared::default();
        let params: Vec<(String, Option<&Type>)> = f.params.iter().map(|p| (p.name.clone(), p.ty.as_ref())).collect();
        collect_declared(&mut locals, &params, &mut f.body);
        for (n, t) in locals.0 { declared.0.insert(n, t); }
        let ret = f.return_ty.clone();
        self.run_body(&declared, ret.as_ref(), &mut f.body);
    }

    fn run_items(&self, items: &mut [Item], globals: &Declared) {
        for item in items {
            match item {
                Item::Fn(f) => self.run_fn(f, &[], globals),
                Item::Struct(s) => {
                    let fields: Vec<(String, Type)> = s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect();
                    for f in &mut s.fields {
                        if let Some(d) = &mut f.default { let t = f.ty.clone(); self.resolve_expected(d, &t, globals); }
                    }
                    for m in &mut s.methods { self.run_fn(m, &fields, globals); }
                    for i in &mut s.inits {
                        let mut declared = globals.clone();
                        for (n, t) in &fields { declared.declare(n, Some(t)); }
                        for p in &i.params { declared.0.remove(&p.name); }
                        let mut locals = Declared::default();
                        let params: Vec<(String, Option<&Type>)> = i.params.iter().map(|p| (p.name.clone(), p.ty.as_ref())).collect();
                        collect_declared(&mut locals, &params, &mut i.body);
                        for (n, t) in locals.0 { declared.0.insert(n, t); }
                        self.run_body(&declared, None, &mut i.body);
                    }
                }
                Item::Enum(e) => { for m in &mut e.methods { self.run_fn(m, &[], globals); } }
                Item::Ext(e) => {
                    let fields = self.decls.struct_fields(&e.type_name);
                    for m in &mut e.methods { self.run_fn(m, &fields, globals); }
                }
                Item::Mod(m) => self.run_items(&mut m.items, globals),
                Item::Let(l) => {
                    if let (Some(t), Some(v)) = (&l.ty, &mut l.value) { self.resolve_expected(v, t, globals); }
                }
                Item::Stmt(s) => self.run_body(globals, None, std::slice::from_mut(s)),
                _ => {}
            }
        }
    }
}

/// Arguments of a call-shaped expression.
fn shape_args_of(e: &Expr) -> Option<&[Arg]> {
    match &e.kind {
        ExprKind::Call(_, a) | ExprKind::MethodCall(_, _, a) => Some(a),
        _ => None,
    }
}

/// Resolves every construction of a qualifier-generic type in `program` that its context
/// determines, rewriting it to the specialised instance's name. Returns the instances found and,
/// for a construction that is determined *wrongly* (conflicting or out-of-bound evidence), the
/// message to report at the position of the type's name. A construction nothing determines is left
/// untouched.
pub(crate) fn infer_qual_ctors(program: &mut Program) -> (Resolved, Diags) {
    let decls = Decls::of_program(program);
    if decls.quals.is_empty() { return (Vec::new(), Diags::new()); }
    // Top-level bindings are visible from every function.
    let mut globals = Declared::default();
    for it in &program.items {
        if let Item::Let(l) = it { globals.declare(&l.name, l.ty.as_ref()); }
    }
    let ctx = Ctx { decls: &decls, out: RefCell::new(Out::default()) };
    ctx.run_items(&mut program.items, &globals);
    let out = ctx.out.into_inner();
    (out.resolved, out.diags)
}

/// Declarations whose parameters are all qualifier parameters with a default: a bare use of the
/// name in type position means this instance (`Board` → `[QualArg(inline)]`).
pub(crate) fn default_instances(program: &Program) -> HashMap<String, Vec<Type>> {
    let decls = Decls::of_program(program);
    let mut out = HashMap::new();
    for (name, qd) in &decls.quals {
        let defaults: Option<Vec<Type>> = qd.params.iter()
            .map(|p| p.as_ref().and_then(|p| p.default.clone()).map(Type::QualArg))
            .collect();
        if let Some(d) = defaults { out.insert(name.clone(), d); }
    }
    out
}
