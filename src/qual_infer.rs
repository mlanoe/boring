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

//! Inference of qualifier arguments at construction (docs/book.md "Qualifier parameters",
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

/// A function with qualifier parameters (`int count<'q>(Board<elem'q> b)`).
struct QFn {
    params: Params,
    /// One entry per declared generic parameter: `Some` for a qualifier parameter.
    qparams: Vec<Option<QParam>>,
    /// Every generic parameter is a qualifier parameter (type arguments are not inferred).
    pure: bool,
    /// The declared spelling of each generic parameter, and whether every type parameter is tied
    /// to a qualifier parameter by `<T'q>` (then the call determines the type arguments too).
    raw: Vec<String>,
    linked: bool,
    ret: Option<Type>,
}

struct QualDecl {
    /// One entry per declared parameter: `Some((name, bound))` for a qualifier parameter.
    params: Vec<Option<QParam>>,
    pure: bool,
    /// The declared spelling of each parameter (`T`, `^q@T`, ...).
    raw: Vec<String>,
    /// Every type parameter is tied to a qualifier parameter by `<T'q>`: the construction can then
    /// determine the type arguments too (docs/book.md, "Qualified generics").
    linked: bool,
}

#[derive(Default)]
struct Decls {
    /// `None` when declared more than once (ambiguous by name).
    ctors: HashMap<String, Option<Ctor>>,
    quals: HashMap<String, QualDecl>,
    fns: HashMap<String, Option<Params>>,
    /// Qualifier-generic functions; `None` when declared more than once.
    qfns: HashMap<String, Option<QFn>>,
    /// Methods with a qualifier parameter of their own, by method name: `(owner, method)`.
    qmethods: HashMap<String, Vec<(String, QFn)>>,
    /// How many methods (of any owner, generic or not) carry each name: a receiver whose type is
    /// unknown can only be resolved when the name is unique.
    method_names: HashMap<String, usize>,
    /// Parameters of the methods that declare no qualifier parameter of their own, by
    /// `"Owner::method"`; `None` when the name is declared twice for one owner (ambiguous).
    methods: HashMap<String, Option<Params>>,
    /// Written return types of the free functions, `None` when ambiguous or unwritten.
    fn_rets: HashMap<String, Option<Type>>,
}

/// The qualifier-generic view of a function or method; `None` when it has no qualifier parameter
/// or a parameter without a written type (positions would shift).
fn qfn_of(f: &FnDecl) -> Option<QFn> {
    if !f.type_params.iter().any(|p| qual_param_name(p).is_some()) { return None; }
    if !f.params.iter().all(|p| p.ty.is_some()) { return None; }
    note_qual_decl();
    let qparams: Vec<_> = f.type_params.iter()
        .map(|p| qual_param_decl(p).map(|(n, b)| QParam { name: n.to_string(), bound: b, default: qual_param_default(p) }))
        .collect();
    let pure = qparams.iter().all(Option::is_some);
    let params = f.params.iter().filter_map(|p| p.ty.clone().map(|t| (Some(p.name.clone()), t))).collect::<Params>();
    let linked = f.type_params.iter().all(|p| {
        qual_param_name(p).is_some() || f.type_params.iter().any(|o| qual_linked_to(o, p))
    });
    Some(QFn { params, qparams, pure, raw: f.type_params.clone(), linked, ret: f.return_ty.clone() })
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
                    self.add_methods(&s.name, &s.methods);
                }
                Item::Enum(e) => {
                    let variants = e.variants.iter().map(|v| {
                        (v.name.clone(), v.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect::<Params>())
                    }).collect();
                    add_unique(&mut self.ctors, &e.name, Ctor::Enum(variants));
                    self.add_qual(&e.name, &e.type_params);
                    self.add_methods(&e.name, &e.methods);
                }
                Item::Fn(f) => {
                    let params = f.params.iter().filter_map(|p| p.ty.clone().map(|t| (Some(p.name.clone()), t))).collect::<Params>();
                    // A parameter without a type would shift positions: treat as unknown.
                    let complete = f.params.iter().all(|p| p.ty.is_some());
                    if complete { add_unique(&mut self.fns, &f.name, params); } else { self.fns.insert(f.name.clone(), None); }
                    match (self.fn_rets.contains_key(&f.name), &f.return_ty) {
                        (false, Some(t)) => { self.fn_rets.insert(f.name.clone(), Some(t.clone())); }
                        _ => { self.fn_rets.insert(f.name.clone(), None); }
                    }
                    if f.qualifier.is_none() {
                        match qfn_of(f) {
                            Some(qf) => add_unique(&mut self.qfns, &f.name, qf),
                            None if f.type_params.iter().any(|p| qual_param_name(p).is_some()) => { self.qfns.insert(f.name.clone(), None); }
                            None => {}
                        }
                    }
                }
                Item::Ext(e) => self.add_methods(&e.type_name, &e.methods),
                Item::Mod(m) => self.collect(&m.items),
                _ => {}
            }
        }
    }

    fn add_methods(&mut self, owner: &str, methods: &[FnDecl]) {
        for m in methods {
            *self.method_names.entry(m.name.clone()).or_default() += 1;
            if let Some(qf) = qfn_of(m) { self.qmethods.entry(m.name.clone()).or_default().push((owner.to_string(), qf)); continue; }
            // A parameter without a written type would shift positions: the method is unknown.
            if m.params.iter().all(|p| p.ty.is_some()) {
                let params = m.params.iter().filter_map(|p| p.ty.clone().map(|t| (Some(p.name.clone()), t))).collect::<Params>();
                add_unique(&mut self.methods, &format!("{owner}::{}", m.name), params);
            } else {
                self.methods.insert(format!("{owner}::{}", m.name), None);
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
        let linked = type_params.iter().all(|p| {
            qual_param_name(p).is_some() || type_params.iter().any(|o| qual_linked_to(o, p))
        });
        // Declared twice: not resolvable by name.
        if self.quals.contains_key(name) { self.quals.remove(name); self.ctors.insert(name.to_string(), None); return; }
        self.quals.insert(name.to_string(), QualDecl { params, pure, raw: type_params.to_vec(), linked });
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

/// Inside a qualifier-generic function body the parameter of the function is itself a written
/// qualifier: `Point'q` is the chain `[Param(q)]`, to be passed on (`g<r'q>(..)`).
fn pass_chain(ty: &Type) -> Option<QualChain> {
    match ty {
        Type::Qualified(_, OwnerQual::Param(p)) => Some(vec![OwnerQual::Param(p.clone())]),
        _ => None,
    }
}

/// The declared type of an argument against the parameter slot type `decl`: `from_ty`; with `pass`, a parameter of the enclosing function is evidence too (a call from the
/// body of a qualifier-generic function to another one).
fn from_ty_in(decl: &Type, arg: &Type, desc: &str, pass: bool, out: &mut Evidence) {
    let (d, a) = (peel(decl), peel(arg));
    match (d, a) {
        (Type::Qualified(_, OwnerQual::Param(p)), Type::Qualified(..)) => {
            if let Some(chain) = written_chain(a).or_else(|| pass.then(|| pass_chain(a)).flatten()) { out.push((p.clone(), chain, desc.to_string())); }
        }
        // `Board<elem'cells>` slot (an enclosing parameter passed on) against an argument declared
        // `Board<elem'actor>`: the written argument gives `cells`.
        (Type::Generic(db, dargs), Type::Generic(ab, aargs)) if db == ab && dargs.len() == aargs.len() => {
            for (x, y) in dargs.iter().zip(aargs) {
                if let (Type::QualArg(dq), Type::QualArg(aq)) = (x, y) {
                    if let Some((p, suffix)) = OwnerQual::pass_through(dq) {
                        // The written argument is the value of `p` plus the suffix added on the way.
                        let value: Option<QualChain> = if suffix.is_empty() { Some(aq.clone()) }
                            else { aq.strip_suffix(suffix).filter(|v| !v.is_empty()).map(<[OwnerQual]>::to_vec) };
                        if let Some(value) = value {
                            if pass || OwnerQual::pass_through(&value).is_none() { out.push((p.to_string(), value, desc.to_string())); }
                        }
                    }
                }
            }
        }
        (Type::Array(di), Type::Array(ai)) | (Type::Set(di), Type::Set(ai)) => from_ty_in(di, ai, desc, pass, out),
        (Type::Dict(dk, dv), Type::Dict(ak, av)) => { from_ty_in(dk, ak, desc, pass, out); from_ty_in(dv, av, desc, pass, out); }
        (Type::Tuple(ds), Type::Tuple(az)) => { for (x, y) in ds.iter().zip(az) { from_ty_in(x, y, desc, pass, out); } }
        _ => {}
    }
}

/// An argument expression against the parameter slot type `decl`: a declared name, or a literal
/// whose elements are declared names.
fn from_expr(decl: &Type, e: &Expr, declared: &Declared, out: &mut Evidence) { from_expr_in(decl, e, declared, false, out) }

fn from_expr_in(decl: &Type, e: &Expr, declared: &Declared, pass: bool, out: &mut Evidence) {
    let d = peel(decl);
    match &e.kind {
        ExprKind::Var(_) | ExprKind::Field(..) => {
            if let Some(t) = declared.of_expr(e) {
                let desc = match &e.kind {
                    ExprKind::Var(n) => format!("declared on `{n}`"),
                    ExprKind::Field(_, n) => format!("declared on `self.{n}`"),
                    _ => unreachable!(),
                };
                from_ty_in(d, t, &desc, pass, out);
            }
        }
        // `Board<elem'actor>()` passed where a `Board<elem'cells>` is expected.
        ExprKind::GenericCall(callee, targs, _) => {
            if let (ExprKind::Var(b), Type::Generic(db, dargs)) = (&callee.kind, d) {
                if b == db && dargs.len() == targs.len() {
                    from_ty_in(d, &Type::Generic(b.clone(), targs.clone()), &format!("written on the `{b}<..>` argument"), pass, out);
                }
            }
        }
        ExprKind::Array(items) | ExprKind::Set(items) => match d {
            Type::Array(di) | Type::Set(di) => { for it in items { from_expr_in(di, it, declared, pass, out); } }
            _ => {}
        },
        ExprKind::Dict(pairs) => {
            if let Type::Dict(dk, dv) = d {
                for (k, v) in pairs { from_expr_in(dk, k, declared, pass, out); from_expr_in(dv, v, declared, pass, out); }
            }
        }
        ExprKind::Tuple(items) => {
            if let Type::Tuple(ds) = d { for (x, it) in ds.iter().zip(items) { from_expr_in(x, it, declared, pass, out); } }
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
    /// The struct, enum or `ext` target whose methods are being walked (the type of `self`).
    owner: RefCell<Option<String>>,
    /// `return` statements that belong to a closure or a nested function of the body being
    /// walked, with the return type that one declares (`None`: unknown, so no expected type).
    /// Keyed by the statement's address; a `return` absent from the table returns from the
    /// function itself.
    inner_returns: RefCell<HashMap<*const Stmt, Option<Type>>>,
}

/// A declared type split into its base (qualifiers and permissions stripped) and written chain;
/// only a named or generic base is a type argument.
fn split_ty(t: &Type) -> Option<(Type, Option<QualChain>)> {
    let written = written_chain(peel(t));
    let mut b = peel(t);
    while let Type::Qualified(i, _) = b { b = peel(i); }
    match b {
        Type::Named(_) | Type::Generic(..) => Some((b.clone(), written)),
        _ => None,
    }
}

/// The type parameter a slot is exactly (`T`, `T'q`, `T'inline`), with the qualifier written on it.
fn direct_param<'a>(slot: &'a Type, raw: &[String]) -> Option<(&'a str, Option<OwnerQual>)> {
    let (inner, q) = match peel(slot) {
        Type::Qualified(i, q) => (peel(i), Some(q.clone())),
        t => (t, None),
    };
    let (Type::Named(t) | Type::TypeParam(t)) = inner else { return None };
    raw.iter().any(|p| p == t).then_some((t.as_str(), q))
}

/// Records `arg_base` as the type of the parameter `slot` is, after the no-silent-conversion check.
fn push_direct(base: &str, slot: &Type, raw: &[String], arg_base: Type, written: Option<QualChain>, desc: String, out: &mut Vec<(String, Type, String)>) -> Result<(), String> {
    let Some((t, slot_q)) = direct_param(slot, raw) else { return Ok(()) };
    if let (Some(sq), Some(w)) = (&slot_q, &written) {
        if !matches!(sq, OwnerQual::Param(_)) && w.as_slice() != std::slice::from_ref(sq) {
            let shown = |c: &[OwnerQual]| OwnerQual::chain_name(c).unwrap_or_else(|| "?".into());
            return Err(format!(
                "`{t}'{}` slot of `{base}` is given an argument written `'{}` ({desc}); no silent conversion, write the argument with `'{}`",
                shown(std::slice::from_ref(sq)), shown(w), shown(std::slice::from_ref(sq))));
        }
    }
    out.push((t.to_string(), arg_base, desc));
    Ok(())
}

/// A slot type against a declared argument type, walked together: `[T'q]` against `[Counter'actor]`,
/// `Box<T'q>` against `Box<Counter'actor>`.
fn type_from_ty(base: &str, raw: &[String], slot: &Type, arg: &Type, desc: &str, out: &mut Vec<(String, Type, String)>) -> Result<(), String> {
    let (s, a) = (peel(slot), peel(arg));
    if direct_param(s, raw).is_some() {
        if let Some((b, written)) = split_ty(a) { return push_direct(base, s, raw, b, written, desc.to_string(), out); }
        return Ok(());
    }
    match (s, a) {
        (Type::Array(x), Type::Array(y)) | (Type::Set(x), Type::Set(y)) => type_from_ty(base, raw, x, y, desc, out),
        (Type::Dict(k1, v1), Type::Dict(k2, v2)) => {
            type_from_ty(base, raw, k1, k2, desc, out)?;
            type_from_ty(base, raw, v1, v2, desc, out)
        }
        (Type::Tuple(xs), Type::Tuple(ys)) => {
            for (x, y) in xs.iter().zip(ys) { type_from_ty(base, raw, x, y, desc, out)?; }
            Ok(())
        }
        (Type::Generic(b1, xs), Type::Generic(b2, ys)) if b1 == b2 && xs.len() == ys.len() => {
            for (x, y) in xs.iter().zip(ys) { type_from_ty(base, raw, x, y, desc, out)?; }
            Ok(())
        }
        (Type::Qualified(i, _), a) => type_from_ty(base, raw, i, a, desc, out),
        _ => Ok(()),
    }
}

fn show_ty(t: &Type) -> String {
    match t { Type::Named(n) | Type::TypeParam(n) => n.clone(), other => format!("{other:?}") }
}

fn has_param(t: &Type) -> bool {
    match t {
        Type::QualArg(q) => q.iter().any(|o| matches!(o, OwnerQual::Param(_))),
        Type::Qualified(i, q) => matches!(q, OwnerQual::Param(_)) || has_param(i),
        Type::Generic(_, args) | Type::Tuple(args) => args.iter().any(has_param),
        Type::Array(i) | Type::Set(i) | Type::Optional(i) | Type::Mut(i) => has_param(i),
        Type::Dict(k, v) => has_param(k) || has_param(v),
        _ => false,
    }
}

fn qual_args_of(ty: &Type) -> Option<(&str, &[Type])> {
    let mut t = ty;
    while let Type::Mut(i) | Type::Qualified(i, _) | Type::Optional(i) = t { t = i; }
    match t {
        Type::Generic(base, args) if args.iter().any(|a| matches!(a, Type::QualArg(_))) => Some((base, args)),
        _ => None,
    }
}

/// `elem'cells` of a declaration of `'cells`: an enclosing parameter passed on, not yet resolved.
fn is_pass_through(t: &Type) -> bool {
    matches!(t, Type::QualArg(q) if OwnerQual::pass_through(q).is_some())
}

/// The substitution `parameter name -> its argument` of a resolved instance of `qd`.
fn inst_subst(qd: &QualDecl, args: &[Type]) -> HashMap<String, QualChain> {
    qd.params.iter().zip(args).filter_map(|(p, a)| match (p, a) {
        (Some(p), Type::QualArg(q)) => Some((p.name.clone(), q.clone())),
        _ => None,
    }).collect()
}

/// `t` with the parameters of the enclosing declaration replaced by `subst` (a pass-through
/// argument `elem'cells` and a use `Point'cells`).
fn subst_pass(t: &Type, subst: &HashMap<String, QualChain>) -> Type {
    let go = |x: &Type| Box::new(subst_pass(x, subst));
    match t {
        Type::QualArg(q) => match OwnerQual::pass_through(q) {
            Some((p, suffix)) => subst.get(p).map_or_else(|| t.clone(), |c| Type::QualArg(OwnerQual::with_suffix(c, suffix))),
            None => t.clone(),
        },
        Type::Qualified(inner, OwnerQual::Param(p)) => match subst.get(p) {
            Some(c) => OwnerQual::apply_chain(subst_pass(inner, subst), c),
            None => t.clone(),
        },
        Type::Qualified(inner, q) => Type::Qualified(go(inner), q.clone()),
        Type::Generic(n, args) => Type::Generic(n.clone(), args.iter().map(|a| subst_pass(a, subst)).collect()),
        Type::Array(i) => Type::Array(go(i)),
        Type::Set(i) => Type::Set(go(i)),
        Type::Optional(i) => Type::Optional(go(i)),
        Type::Mut(i) => Type::Mut(go(i)),
        Type::Dict(k, v) => Type::Dict(go(k), go(v)),
        Type::Tuple(xs) => Type::Tuple(xs.iter().map(|x| subst_pass(x, subst)).collect()),
        _ => t.clone(),
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
                self.resolve_fn_call(e, Some(expected), declared);
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
        // The expected type passes an enclosing parameter on (`Board<elem'cells>` in a field of
        // `Game<'cells>`): a bare construction never takes it implicitly, it is written out.
        if let Some(a) = expected_args {
            if let Some(i) = a.iter().position(is_pass_through) {
                let label = qd.params.get(i).and_then(|p| p.as_ref()).map_or("?", |p| p.name.as_str());
                let Some(Type::QualArg(q)) = a.get(i) else { return };
                let OwnerQual::Param(p) = &q[0] else { return };
                self.out.borrow_mut().diags.insert(pos, format!(
                    "`{base}` here takes its own default for `{label}`, not `{p}` of the enclosing declaration; write `{base}<{label}'{p}>` to pass it on"));
                return;
            }
        }
        match self.infer(&base, qd, &shape, expected_args, declared) {
            Ok(Some(args)) => {
                if let Some(m) = crate::transpiler::monomorphize::mangled_name(&base, &args) {
                    // What each argument is expected to be, now that the instance is known
                    // (`Game<cells'actor>` makes `board` a `Board<elem'actor>`).
                    let subst = inst_subst(qd, &args);
                    let pushed: Vec<(usize, Type)> = match ctor_params(self.decls, &base, &shape) {
                        Some(ps) => {
                            let call_args = shape_args(&shape);
                            bind_args(call_args, ps).into_iter()
                                .filter_map(|(a, t)| call_args.iter().position(|x| std::ptr::eq(x, a)).map(|i| (i, subst_pass(t, &subst))))
                                .collect()
                        }
                        None => Vec::new(),
                    };
                    crate::transpiler::monomorphize::rename_ctor(e, &base, &m);
                    self.out.borrow_mut().resolved.push((base, args));
                    let call_args = match &mut e.kind {
                        ExprKind::Call(_, a) | ExprKind::MethodCall(_, _, a) => Some(a),
                        _ => None,
                    };
                    if let Some(call_args) = call_args {
                        for (i, t) in pushed { if let Some(a) = call_args.get_mut(i) { self.resolve_expected(&mut a.value, &t, declared); } }
                    }
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
        // Type arguments of a `<T'q>` declaration, from the arguments feeding a `T` / `T'q` slot.
        let mut tev: Vec<(String, Type, String)> = Vec::new();
        let inferable = qd.pure || qd.linked;
        if inferable {
            if let Some(params) = ctor_params(self.decls, base, shape) {
                for (arg, ty) in bind_args(shape_args(shape), params) {
                    from_expr(ty, &arg.value, declared, &mut ev);
                    if qd.linked {
                        self.type_evidence(base, &qd.raw, ty, &arg.value, declared, &mut tev)?;
                    }
                }
            }
        }
        if !inferable && expected.is_none() { return Ok(None); }
        let mut args = Vec::with_capacity(qd.params.len());
        for (i, p) in qd.params.iter().enumerate() {
            let Some(QParam { name, bound, default }) = p else {
                // A type parameter: the expected type, else (`<T'q>`) the type of the arguments.
                if let Some(t) = expected.and_then(|a| a.get(i)) { args.push(t.clone()); continue; }
                let tname = qd.raw[i].as_str();
                let mut found: Option<&(String, Type, String)> = None;
                for e in tev.iter().filter(|(n, _, _)| n == tname) {
                    match found {
                        Some((_, t, d)) if *t != e.1 => return Err(format!(
                            "conflicting types for `{tname}` of `{base}`: `{}` ({d}) and `{}` ({}); write `{base}<{tname} = ...>` to choose",
                            show_ty(t), show_ty(&e.1), e.2)),
                        Some(_) => {}
                        None => found = Some(e),
                    }
                }
                match found { Some((_, t, _)) => { args.push(t.clone()); continue; } None => return Ok(None) }
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

    /// Evidence for the type arguments of a `<T'q>` declaration from one argument feeding `slot`.
    /// The slot is read by shape: `T`, `T'q`, `T'inline`, and `T` inside an array, set, dict,
    /// tuple, optional or generic (`[T'q]`, `Box<T'q>`), against a literal's elements or against
    /// the written type of a declared name. A slot that fixes its own qualifier (`T'inline`) given
    /// an argument written with another one is an error (no silent conversion).
    fn type_evidence(&self, base: &str, raw: &[String], slot: &Type, arg: &Expr, declared: &Declared, out: &mut Vec<(String, Type, String)>) -> Result<(), String> {
        let s = peel(slot);
        match (&arg.kind, s) {
            (ExprKind::Array(items) | ExprKind::Set(items), Type::Array(di) | Type::Set(di)) => {
                for it in items { self.type_evidence(base, raw, di, it, declared, out)?; }
                return Ok(());
            }
            (ExprKind::Dict(pairs), Type::Dict(dk, dv)) => {
                for (k, v) in pairs {
                    self.type_evidence(base, raw, dk, k, declared, out)?;
                    self.type_evidence(base, raw, dv, v, declared, out)?;
                }
                return Ok(());
            }
            (ExprKind::Tuple(items), Type::Tuple(ds)) => {
                for (d, it) in ds.iter().zip(items) { self.type_evidence(base, raw, d, it, declared, out)?; }
                return Ok(());
            }
            _ => {}
        }
        if direct_param(s, raw).is_some() {
            if let Some((arg_base, written, desc)) = self.arg_type(arg, declared) {
                return push_direct(base, s, raw, arg_base, written, desc, out);
            }
            return Ok(());
        }
        // A declared name against a compound slot: walk the two types together.
        if matches!(&arg.kind, ExprKind::Var(_) | ExprKind::Field(..)) {
            if let Some(t) = declared.of_expr(arg) {
                let desc = match &arg.kind {
                    ExprKind::Var(n) => format!("declared on `{n}`"),
                    ExprKind::Field(_, n) => format!("declared on `self.{n}`"),
                    _ => unreachable!(),
                };
                type_from_ty(base, raw, s, t, &desc, out)?;
            }
        }
        Ok(())
    }

    /// `(base type, written qualifier chain, description)` of an argument expression: a declared
    /// name (its written type) or a construction of a known, non-generic struct.
    fn arg_type(&self, e: &Expr, declared: &Declared) -> Option<(Type, Option<QualChain>, String)> {
        match &e.kind {
            ExprKind::Var(_) | ExprKind::Field(..) => {
                let t = declared.of_expr(e)?;
                let desc = match &e.kind {
                    ExprKind::Var(n) => format!("declared on `{n}`"),
                    ExprKind::Field(_, n) => format!("declared on `self.{n}`"),
                    _ => return None,
                };
                let (b, written) = split_ty(t)?;
                Some((b, written, desc))
            }
            ExprKind::Call(c, _) => match &c.kind {
                ExprKind::Var(n) if matches!(self.decls.ctors.get(n.as_str()), Some(Some(Ctor::Struct { .. }))) && !self.decls.quals.contains_key(n.as_str()) =>
                    Some((Type::Named(n.clone()), None, format!("the `{n}(..)` argument"))),
                _ => None,
            },
            _ => None,
        }
    }

    /// The walk over one function body (or one top-level statement).
    fn run_body(&self, declared: &Declared, ret: Option<&Type>, body: &mut [Stmt]) {
        // A tail expression takes the function's return type.
        if let (Some(ret), Some(Stmt::Expr(e))) = (ret, body.last_mut()) { self.resolve_expected(e, ret, declared); }
        self.inner_returns.borrow_mut().clear();
        self.collect_inner_returns(body);
        let mut on_stmt = |s: &mut Stmt| {
            let key = s as *const Stmt;
            match s {
                Stmt::Let(l) => {
                    if let (Some(t), Some(v)) = (&l.ty, &mut l.value) { self.resolve_expected(v, t, declared); }
                }
                Stmt::Match(m) => self.resolve_patterns(m, declared),
                Stmt::Return(r) => {
                    // A `return` in a closure or a nested function is read against that one's own
                    // return type, never the enclosing function's.
                    let want = match self.inner_returns.borrow().get(&key) { Some(t) => t.clone(), None => ret.cloned() };
                    if let (Some(t), Some(v)) = (want, &mut r.value) { self.resolve_expected(v, &t, declared); }
                }
                _ => {}
            }
        };
        let mut on_expr = |e: &mut Expr| match &mut e.kind {
            ExprKind::Match(m) => self.resolve_patterns(m, declared),
            // `Slot.Empty`: a bare construction of an enum variant with no payload.
            ExprKind::Field(..) => self.resolve_site(e, None, declared),
            ExprKind::Assign(l, r) => {
                if let Some(t) = declared.of_expr(l) { let t = t.clone(); self.resolve_expected(r, &t, declared); }
            }
            // `Game<cells'actor>(board = Board())`: the written arguments fix the instance, so the
            // arguments take the parameter types of that instance.
            ExprKind::GenericCall(callee, targs, args) => {
                if matches!(&callee.kind, ExprKind::Field(..)) { self.push_fn_arg_types(e, declared); return; }
                let ExprKind::Var(n) = &callee.kind else { return };
                if self.decls.qfns.contains_key(n.as_str()) { self.push_fn_arg_types(e, declared); return; }
                let Some(qd) = self.decls.quals.get(n.as_str()) else { return };
                let subst = inst_subst(qd, targs);
                if subst.is_empty() { return; }
                let Some(ps) = ctor_params(self.decls, n, &Shape::Struct(args)) else { return };
                let pushed: Vec<(usize, Type)> = bind_args(args, ps).into_iter()
                    .filter_map(|(a, t)| args.iter().position(|x| std::ptr::eq(x, a)).map(|i| (i, subst_pass(t, &subst))))
                    .collect();
                for (i, t) in pushed { if let Some(a) = args.get_mut(i) { self.resolve_expected(&mut a.value, &t, declared); } }
            }
            ExprKind::Call(..) | ExprKind::MethodCall(..) => {
                self.resolve_fn_call(e, None, declared);
                if matches!(e.kind, ExprKind::GenericCall(..)) { return; }
                // Arguments take the parameter types of the callee (a function, a struct or an
                // enum variant), then the call itself may be a construction.
                let bound: Vec<(usize, Type)> = {
                    let params = match &e.kind {
                        ExprKind::Call(c, args) => match &c.kind {
                            ExprKind::Var(n) => self.callee_params(n, args).cloned(),
                            _ => None,
                        }.or_else(|| view(e).and_then(|(b, _, sh)| ctor_params(self.decls, b, &sh).cloned())),
                        ExprKind::MethodCall(recv, m, _) => view(e).and_then(|(b, _, sh)| ctor_params(self.decls, b, &sh).cloned())
                            .or_else(|| self.method_params(recv, m, declared)),
                        _ => view(e).and_then(|(b, _, sh)| ctor_params(self.decls, b, &sh).cloned()),
                    };
                    match (params.as_ref(), shape_args_of(e)) {
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

    /// The written type of a match subject: a declared name or a call of a function that declares
    /// its return type.
    fn subject_type(&self, e: &Expr, declared: &Declared) -> Option<Type> {
        if let Some(t) = declared.of_expr(e) { return Some(t.clone()); }
        match &e.kind {
            ExprKind::Call(c, _) => match &c.kind {
                ExprKind::Var(n) => self.decls.fn_rets.get(n.as_str()).cloned().flatten(),
                _ => None,
            },
            _ => None,
        }
    }

    /// Qualified variant patterns of a qualifier-generic enum (`Slot.Full(p)`, `Slot<elem'actor>.Full(p)`)
    /// name the instance they match: the arguments written in the pattern, else the written type
    /// of the subject (its arguments, or the defaults of a bare name). The pattern becomes the
    /// instance's name. A pattern that nothing here determines is left alone: the emitter reads
    /// the instance of the match subject then.
    fn resolve_patterns(&self, m: &mut MatchStmt, declared: &Declared) {
        let subject = self.subject_type(&m.subject, declared);
        let (sline, scol) = (m.subject.line, m.subject.col);
        for arm in &mut m.arms {
            let pos = (arm.line, arm.col);
            for p in &mut arm.patterns { self.resolve_pattern(p, subject.as_ref(), pos, (sline, scol)); }
        }
    }

    fn resolve_pattern(&self, p: &mut Pattern, subject: Option<&Type>, pos: (usize, usize), spos: (usize, usize)) {
        let diag = |msg: String| { self.out.borrow_mut().diags.insert(pos, msg); };
        match p {
            Pattern::Some(inner) => self.resolve_pattern(inner, None, pos, spos),
            Pattern::Tuple(subs) => { for x in subs { self.resolve_pattern(x, None, pos, spos); } }
            Pattern::Variant(name, subs) => {
                for x in subs.iter_mut() { self.resolve_pattern(x, None, pos, spos); }
                let owned = name.clone();
                let Some((prefix, variant)) = owned.split_once("::") else { return };
                let (base, explicit) = match prefix.split_once('<') {
                    Some((b, rest)) => (b, Some(rest.trim_end_matches('>'))),
                    None => (prefix, None),
                };
                let Some(qd) = self.decls.quals.get(base) else { return };
                if !matches!(self.decls.ctors.get(base), Some(Some(Ctor::Enum(_)))) { return; }
                // What the subject's written type says, when it is concrete.
                let from_subject: Option<Vec<Type>> = match subject.map(peel_all) {
                    Some(Type::Generic(b, a)) if b == base && a.len() == qd.params.len()
                        && a.iter().all(|x| matches!(x, Type::QualArg(q) if !q.iter().any(|o| matches!(o, OwnerQual::Param(_))))) => Some(a.clone()),
                    Some(Type::Named(b)) if b == base => qd.params.iter()
                        .map(|p| p.as_ref().and_then(|p| p.default.clone()).map(Type::QualArg)).collect(),
                    _ => None,
                };
                let args: Vec<Type> = match explicit {
                    Some(list) => {
                        let chains: Vec<Option<QualChain>> = list.split(',').map(OwnerQual::chain_from_name).collect();
                        if chains.len() != qd.params.len() || chains.iter().any(Option::is_none) {
                            diag(format!("the pattern gives {} qualifier argument(s) to `{base}`, which declares {}", chains.len(), qd.params.len()));
                            return;
                        }
                        let args: Vec<Type> = chains.into_iter().flatten().map(Type::QualArg).collect();
                        for (a, q) in args.iter().zip(&qd.params) {
                            let (Type::QualArg(c), Some(q)) = (a, q) else { continue };
                            if !q.bound.contains(c) {
                                let allowed: Vec<String> = q.bound.iter().filter_map(|c| OwnerQual::chain_name(c)).collect();
                                diag(format!("qualifier `'{}` is outside the bound of `{}` in `{base}` (allowed: {})",
                                    OwnerQual::chain_name(c).unwrap_or_default(), q.name, allowed.join(", ")));
                                return;
                            }
                        }
                        if let Some(subj) = &from_subject {
                            if *subj != args {
                                let show = |v: &[Type]| v.iter().filter_map(|t| match t { Type::QualArg(c) => OwnerQual::chain_name(c).map(|n| format!("'{n}")), _ => None }).collect::<Vec<_>>().join(", ");
                                self.out.borrow_mut().diags.insert(spos, format!(
                                    "the pattern names the instance `{base}<{}>` but the subject is declared `{base}<{}>`", show(&args), show(subj)));
                                return;
                            }
                        }
                        args
                    }
                    None => match from_subject { Some(a) => a, None => return },
                };
                if let Some(m) = crate::transpiler::monomorphize::mangled_name(base, &args) {
                    *name = format!("{m}::{variant}");
                    self.out.borrow_mut().resolved.push((base.to_string(), args));
                }
            }
            _ => {}
        }
    }

    /// Records which `return` statements of `body` sit inside a closure or a nested function.
    fn collect_inner_returns(&self, body: &mut [Stmt]) {
        fn mark(table: &RefCell<HashMap<*const Stmt, Option<Type>>>, stmts: &mut [Stmt], ty: Option<Type>) {
            let mut on_expr = |_: &mut Expr| {};
            let mut on_stmt = |s: &mut Stmt| {
                if matches!(s, Stmt::Return(_)) { table.borrow_mut().insert(s as *const Stmt, ty.clone()); }
            };
            let mut w = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
            w.stmts(stmts);
        }
        // Outer scopes are visited first, so an inner closure's own entries replace theirs.
        let table = &self.inner_returns;
        let mut on_expr = |e: &mut Expr| {
            if let ExprKind::Closure(_, ret, body, _, _) = &mut e.kind {
                match body {
                    ClosureBody::Block(b) => mark(table, b, ret.clone()),
                    ClosureBody::Expr(x) => {
                        let mut stmt = [Stmt::Expr(std::mem::replace(&mut **x, Expr { kind: ExprKind::Void, line: 0, col: 0, len: 0 }))];
                        mark(table, &mut stmt, ret.clone());
                        let [Stmt::Expr(back)] = stmt else { unreachable!() };
                        **x = back;
                    }
                }
            }
        };
        let mut on_stmt = |s: &mut Stmt| {
            if let Stmt::Fn(f) = s { let ret = f.return_ty.clone(); mark(table, &mut f.body, ret); }
        };
        let mut w = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        w.stmts(body);
    }

    /// The arguments of a call to the qualifier-generic `qf` (a function or a method): resolved from
    /// the evidence (the expected type against the return type, then the written qualifiers of the
    /// arguments, then the declared defaults). Evidence that disagrees or leaves the bound, or a
    /// parameter nothing determines, is reported at `pos` and gives `None`.
    fn qcall_args(&self, qf: &QFn, name: &str, pos: (usize, usize), args: &[Arg], expected: Option<&Type>, declared: &Declared) -> Option<Vec<Type>> {
        let shown = |c: &QualChain| OwnerQual::chain_name(c).unwrap_or_else(|| match c.as_slice() {
            [OwnerQual::Param(p)] => p.clone(),
            _ => "?".into(),
        });
        let fail = |msg: String| { self.out.borrow_mut().diags.insert(pos, msg); None };
        if !qf.pure && !qf.linked {
            return fail(format!("`{name}` also declares type parameters, which are not inferred: write its generic arguments, `{name}<...>`"));
        }
        let mut ev = Evidence::new();
        if let (Some(t), Some(ret)) = (expected, &qf.ret) { from_ty_in(ret, t, "the expected type", true, &mut ev); }
        let mut tev: Vec<(String, Type, String)> = Vec::new();
        for (arg, ty) in bind_args(args, &qf.params) {
            from_expr_in(ty, &arg.value, declared, true, &mut ev);
            if qf.linked && !qf.pure {
                if let Err(m) = self.type_evidence(name, &qf.raw, ty, &arg.value, declared, &mut tev) { return fail(m); }
            }
        }
        let mut targs: Vec<Type> = Vec::with_capacity(qf.qparams.len());
        let mut missing: Vec<&str> = Vec::new();
        for (i, p) in qf.qparams.iter().enumerate() {
            let Some(QParam { name: q, bound, default }) = p else {
                let tname = qf.raw[i].as_str();
                let mut found: Option<&(String, Type, String)> = None;
                for e in tev.iter().filter(|(n, _, _)| n == tname) {
                    match found {
                        Some((_, t, d)) if *t != e.1 => return fail(format!(
                            "conflicting types for `{tname}` of `{name}`: `{}` ({d}) and `{}` ({}); write `{name}<{tname} = ...>` to choose",
                            show_ty(t), show_ty(&e.1), e.2)),
                        Some(_) => {}
                        None => found = Some(e),
                    }
                }
                match found {
                    Some((_, t, _)) => targs.push(t.clone()),
                    None => return fail(format!("cannot infer type parameter `{tname}` of `{name}`, write `{name}<{tname} = ...>`")),
                }
                continue;
            };
            let mut chosen: Option<(QualChain, String)> = None;
            for (_, c, desc) in ev.iter().filter(|(n, _, _)| n == q) {
                match &chosen {
                    None => chosen = Some((c.clone(), desc.clone())),
                    Some((x, xdesc)) if x != c => {
                        return fail(format!(
                            "conflicting qualifiers for `{q}` of `{name}`: `'{}` ({xdesc}) and `'{}` ({desc}); write `{name}<{q}'...>` to choose",
                            shown(x), shown(c)));
                    }
                    _ => {}
                }
            }
            match chosen {
                Some((c, desc)) => {
                    // A parameter of the enclosing function is checked where it is declared.
                    if OwnerQual::pass_through(&c).is_none() && !bound.contains(&c) {
                        let allowed: Vec<String> = bound.iter().map(&shown).collect();
                        return fail(format!(
                            "qualifier `'{}` ({desc}) is outside the bound of `{q}` in `{name}` (allowed: {})",
                            shown(&c), allowed.join(", ")));
                    }
                    targs.push(Type::QualArg(c));
                }
                None => match default {
                    Some(d) => targs.push(Type::QualArg(d.clone())),
                    None => missing.push(q.as_str()),
                },
            }
        }
        if !missing.is_empty() {
            let (plural, quoted, write) = (
                if missing.len() == 1 { "" } else { "s" },
                missing.iter().map(|q| format!("`{q}`")).collect::<Vec<_>>().join(", "),
                qf.qparams.iter().flatten().map(|p| format!("{}'...", p.name)).collect::<Vec<_>>().join(", "),
            );
            return fail(format!("cannot infer qualifier parameter{plural} {quoted} of `{name}`, write `{name}<{write}>`"));
        }
        Some(targs)
    }

    /// A call `count(a)` to a qualifier-generic function becomes `count<q'actor>(a)`; so does a
    /// method call `g.merge(b)` to a method that declares a parameter of its own.
    fn resolve_fn_call(&self, e: &mut Expr, expected: Option<&Type>, declared: &Declared) {
        match &e.kind {
            ExprKind::Call(callee, args) => {
                let ExprKind::Var(name) = &callee.kind else { return };
                let Some(Some(qf)) = self.decls.qfns.get(name.as_str()) else { return };
                let Some(targs) = self.qcall_args(qf, name, (callee.line, callee.col), args, expected, declared) else { return };
                let name = name.clone();
                let ExprKind::Call(callee, args) = std::mem::replace(&mut e.kind, ExprKind::Void) else { return };
                if targs.iter().all(|t| !is_pass_through(t)) { self.out.borrow_mut().resolved.push((name, targs.clone())); }
                e.kind = ExprKind::GenericCall(callee, targs, args);
                self.push_fn_arg_types(e, declared);
            }
            ExprKind::MethodCall(recv, m, args) => {
                let Some(cands) = self.decls.qmethods.get(m.as_str()) else { return };
                // The span of a method call starts at its dot: the name follows it.
                let pos = (e.line, e.col + 1);
                let recv_base = self.receiver_base(recv, declared);
                let qf = match &recv_base {
                    Some(b) => match cands.iter().find(|(o, _)| o == b) { Some((_, qf)) => qf, None => return },
                    None if cands.len() == 1 && self.decls.method_names.get(m.as_str()) == Some(&1) => &cands[0].1,
                    None => {
                        self.out.borrow_mut().diags.insert(pos, format!(
                            "cannot infer qualifier parameter of method `{m}`: the type of its receiver is not written, write `{m}<q'...>`"));
                        return;
                    }
                };
                let Some(targs) = self.qcall_args(qf, m, pos, args, expected, declared) else { return };
                let m = m.clone();
                let ExprKind::MethodCall(recv, _, args) = std::mem::replace(&mut e.kind, ExprKind::Void) else { return };
                if targs.iter().all(|t| !is_pass_through(t)) { self.out.borrow_mut().resolved.push((format!(".{m}"), targs.clone())); }
                let callee = Expr { kind: ExprKind::Field(recv, m), line: e.line, col: e.col, len: e.len };
                e.kind = ExprKind::GenericCall(Box::new(callee), targs, args);
                self.push_fn_arg_types(e, declared);
            }
            _ => {}
        }
    }

    /// The declaration name of the type of a method receiver, when it is written.
    fn receiver_base(&self, recv: &Expr, declared: &Declared) -> Option<String> {
        if matches!(&recv.kind, ExprKind::Var(n) if n == "self") { return self.owner.borrow().clone(); }
        let mut t = declared.of_expr(recv)?;
        while let Type::Mut(i) | Type::Qualified(i, _) | Type::Optional(i) = t { t = i; }
        match t {
            Type::Generic(b, _) | Type::Named(b) => Some(b.clone()),
            _ => None,
        }
    }

    /// `f<q'actor>(Board())`: the arguments take the parameter types of that instance.
    fn push_fn_arg_types(&self, e: &mut Expr, declared: &Declared) {
        let ExprKind::GenericCall(callee, targs, args) = &mut e.kind else { return };
        let qf: &QFn = match &callee.kind {
            ExprKind::Var(n) => match self.decls.qfns.get(n.as_str()) { Some(Some(q)) => q, _ => return },
            ExprKind::Field(recv, m) => {
                let Some(cands) = self.decls.qmethods.get(m.as_str()) else { return };
                match self.receiver_base(recv, declared) {
                    Some(b) => match cands.iter().find(|(o, _)| *o == b) { Some((_, q)) => q, None => return },
                    None if cands.len() == 1 => &cands[0].1,
                    None => return,
                }
            }
            _ => return,
        };
        let subst: HashMap<String, QualChain> = qf.qparams.iter().zip(targs.iter()).filter_map(|(p, a)| match (p, a) {
            (Some(p), Type::QualArg(q)) => Some((p.name.clone(), q.clone())),
            _ => None,
        }).collect();
        if subst.is_empty() { return; }
        let pushed: Vec<(usize, Type)> = bind_args(args, &qf.params).into_iter()
            .filter_map(|(a, t)| args.iter().position(|x| std::ptr::eq(x, a)).map(|i| (i, subst_pass(t, &subst))))
            .collect();
        for (i, t) in pushed { if let Some(a) = args.get_mut(i) { self.resolve_expected(&mut a.value, &t, declared); } }
    }

    /// The parameters of the method `recv.m(..)` when the method is certain: the written type of
    /// the receiver (or `self`) names its declaration, which declares exactly one method of that
    /// name. The qualifier arguments of the receiver's type are substituted into the parameter
    /// types (`Game<cells'actor>` makes a `Board<elem'cells>` parameter a `Board<elem'actor>`); a
    /// slot that still names a parameter of the declaration is left out, it proves nothing.
    fn method_params(&self, recv: &Expr, m: &str, declared: &Declared) -> Option<Params> {
        let base = self.receiver_base(recv, declared)?;
        let Some(Some(params)) = self.decls.methods.get(&format!("{base}::{m}")) else { return None };
        let mut subst = HashMap::new();
        if let Some(qd) = self.decls.quals.get(&base) {
            let written = if matches!(&recv.kind, ExprKind::Var(n) if n == "self") { None } else { declared.of_expr(recv) };
            let args: Option<Vec<Type>> = match written.map(peel_all) {
                Some(Type::Generic(_, a)) => Some(a.clone()),
                // A bare name is its default instance, when every parameter has one.
                Some(Type::Named(_)) => qd.params.iter()
                    .map(|p| p.as_ref().and_then(|p| p.default.clone()).map(Type::QualArg)).collect(),
                _ => None,
            };
            if let Some(args) = args { subst = inst_subst(qd, &args); }
        }
        // Positions must not shift: an unresolved slot becomes a type that names no instance.
        Some(params.iter().map(|(n, t)| {
            let t = subst_pass(t, &subst);
            (n.clone(), if has_param(&t) { Type::Named("_".into()) } else { t })
        }).collect())
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
                    *self.owner.borrow_mut() = Some(s.name.clone());
                    for m in &mut s.methods { self.run_fn(m, &fields, globals); }
                    *self.owner.borrow_mut() = None;
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
                Item::Enum(e) => {
                    *self.owner.borrow_mut() = Some(e.name.clone());
                    for m in &mut e.methods { self.run_fn(m, &[], globals); }
                    *self.owner.borrow_mut() = None;
                }
                Item::Ext(e) => {
                    let fields = self.decls.struct_fields(&e.type_name);
                    *self.owner.borrow_mut() = Some(e.type_name.clone());
                    for m in &mut e.methods { self.run_fn(m, &fields, globals); }
                    *self.owner.borrow_mut() = None;
                }
                Item::Mod(m) => self.run_items(&mut m.items, globals),
                Item::Let(l) => {
                    if let (Some(t), Some(v)) = (&l.ty, &mut l.value) { self.resolve_expected(v, t, globals); }
                    // The initialiser is an ordinary expression too: calls inside it are resolved.
                    if let Some(v) = l.value.take() {
                        let mut stmt = [Stmt::Expr(v)];
                        self.run_body(globals, None, &mut stmt);
                        let [Stmt::Expr(v)] = stmt else { unreachable!() };
                        l.value = Some(v);
                    }
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
    if decls.quals.is_empty() && decls.qfns.is_empty() { return (Vec::new(), Diags::new()); }
    // Top-level bindings are visible from every function.
    let mut globals = Declared::default();
    for it in &program.items {
        if let Item::Let(l) = it { globals.declare(&l.name, l.ty.as_ref()); }
    }
    let ctx = Ctx { decls: &decls, out: RefCell::new(Out::default()), owner: RefCell::new(None), inner_returns: RefCell::new(HashMap::new()) };
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
