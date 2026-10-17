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

//! Plain-data structs inside device code (kernel bodies and the free functions they call),
//! lowered source-to-source by scalar replacement, so every GPU backend — Metal, wgpu, CUDA,
//! ROCm and simd — inherits struct support without defining struct types in four shader
//! languages (and without layout/padding concerns):
//!
//! * a struct local `p: P` becomes one local per leaf field, `p__a`, `p__n__x` (nested structs
//!   are flattened); `p.a` / `p.n.x` read and write those locals;
//! * `P(a = 1, b = 2.0)` (labeled, positional or defaulted), a copy `q = p`, a field chain and
//!   a call of a struct-returning function all assign leaf by leaf, as does `p = <struct>`;
//! * a free function with a struct parameter and/or return gets device clones: each struct
//!   parameter is spread into its leaves (`f__dev`), and a struct return becomes one clone
//!   per leaf (`f__dev__a`, `f__dev__n__x`). Call sites in device code are rewritten to the
//!   clones; the original stays for host code. Like optional returns, a struct-returning call
//!   is evaluated once per leaf, so it must be free of side effects;
//! * an optional leaf (`int? a`) stays an optional local, which `desugar_gpu_optional` lowers
//!   next on the backends without an `Option`.
//!
//! Eligible structs have only scalar / `bool` / optional-scalar / nested-eligible-struct fields,
//! no type parameters and no custom `init`. Struct methods, arrays inside structs and structs
//! in kernel buffers are not lowered (left as written).

use crate::ast::*;
use crate::ast_walk::Walker;
use std::collections::{HashMap, HashSet};

/// Returns `program` with device-side structs scalar-replaced.
pub fn desugar(program: &Program) -> Program {
    let mut program = program.clone();
    let structs = eligible_structs(&program);
    if structs.is_empty() {
        return program;
    }
    let mut cx = Cx { structs, fns: HashMap::new(), env: HashMap::new() };
    for item in &program.items {
        if let Item::Fn(f) = item {
            if f.qualifier.is_none() && !f.task {
                let params: Vec<Option<String>> = f.params.iter().map(|p| cx.struct_name(p.ty.as_ref())).collect();
                let ret = cx.struct_name(f.return_ty.as_ref());
                if params.iter().any(Option::is_some) || ret.is_some() {
                    cx.fns.insert(f.name.clone(), Sig { params, ret });
                }
            }
        }
    }
    let mut clones = Vec::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if let Some(sig) = cx.fns.get(&f.name) {
                clones.extend(cx.device_clones(f, sig));
            }
        }
    }
    let mut touched = false;
    for item in &mut program.items {
        if let Item::Kernel(k) = item {
            for m in &mut k.methods {
                let mut local = Cx { env: HashMap::new(), ..cx.clone() };
                m.body = block(std::mem::take(&mut m.body), &mut local);
                rewrite_exprs(&mut m.body, &local);
                touched = true;
            }
        }
    }
    let _ = touched;
    program.items.extend(clones.into_iter().map(Item::Fn));
    program
}

#[derive(Clone)]
struct Sig {
    /// Struct name of each struct-typed parameter.
    params: Vec<Option<String>>,
    /// Struct name of a struct return.
    ret: Option<String>,
}

#[derive(Clone)]
struct StructInfo {
    fields: Vec<(String, Type, Option<Expr>)>,
}

#[derive(Clone)]
struct Cx {
    structs: HashMap<String, StructInfo>,
    fns: HashMap<String, Sig>,
    /// Struct-typed locals / parameters → struct name.
    env: HashMap<String, String>,
}

fn unwrap_ty(t: &Type) -> &Type {
    match t {
        Type::Mut(x) | Type::Qualified(x, _) => unwrap_ty(x),
        _ => t,
    }
}

fn is_scalar(t: &Type) -> bool {
    matches!(
        unwrap_ty(t),
        Type::Int | Type::Uint | Type::Uint8 | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64
            | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Float32 | Type::Float64 | Type::Bool
    ) || matches!(unwrap_ty(t), Type::Named(n) if matches!(n.as_str(),
        "int" | "uint" | "float" | "float32" | "float64" | "int8" | "int16" | "int32" | "int64"
        | "uint8" | "uint16" | "uint32" | "uint64" | "bool"))
}

/// Structs whose fields are all scalar, optional-scalar, or another eligible struct.
fn eligible_structs(program: &Program) -> HashMap<String, StructInfo> {
    let decls: HashMap<&str, &StructDecl> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Struct(s) if !s.is_view && !s.is_native && s.type_params.is_empty() && s.inits.is_empty() => {
                Some((s.name.as_str(), s))
            }
            _ => None,
        })
        .collect();
    let mut ok: HashSet<&str> = HashSet::new();
    loop {
        let before = ok.len();
        for (name, s) in &decls {
            if ok.contains(name) {
                continue;
            }
            let fine = s.fields.iter().all(|f| match unwrap_ty(&f.ty) {
                Type::Optional(inner) => is_scalar(inner),
                Type::Named(n) | Type::TypeParam(n) if !is_scalar(&f.ty) => ok.contains(n.as_str()),
                _ => is_scalar(&f.ty),
            });
                    if fine {
                ok.insert(name);
            }
        }
        if ok.len() == before {
            break;
        }
    }
    ok.into_iter()
        .map(|n| {
            let s = decls[n];
            (
                n.to_string(),
                StructInfo { fields: s.fields.iter().map(|f| (f.name.clone(), f.ty.clone(), f.default.clone())).collect() },
            )
        })
        .collect()
}

fn mk(kind: ExprKind, at: &Expr) -> Expr {
    Expr { kind, line: at.line, col: at.col, len: at.len }
}

fn var(name: &str, at: &Expr) -> Expr {
    mk(ExprKind::Var(name.to_string()), at)
}

fn zero(ty: &Type, at: &Expr) -> Expr {
    mk(
        match unwrap_ty(ty) {
            Type::Bool => ExprKind::Bool(false),
            Type::Float32 | Type::Float64 => ExprKind::Float(0.0),
            Type::Named(n) if matches!(n.as_str(), "float" | "float32" | "float64") => ExprKind::Float(0.0),
            _ => ExprKind::Int(0),
        },
        at,
    )
}

fn flat(root: &str, path: &[String]) -> String {
    if path.is_empty() { root.to_string() } else { format!("{root}__{}", path.join("__")) }
}

fn stmt_let(binding: BindingKind, name: &str, ty: Type, value: Expr, line: usize, col: usize) -> Stmt {
    Stmt::Let(LetStmt {
        binding,
        is_pub: false,
        is_static: false,
        name: name.to_string(),
        ty: Some(ty),
        var_mut: false,
        value: Some(value),
        is_lazy: false,
        line,
        col,
    })
}

fn assign(target: Expr, value: Expr) -> Stmt {
    let at = target.clone();
    Stmt::Expr(mk(ExprKind::Assign(Box::new(target), Box::new(value)), &at))
}

impl Cx {
    fn struct_name(&self, t: Option<&Type>) -> Option<String> {
        match t.map(unwrap_ty) {
            // A one-letter struct name (`V`) parses as a type parameter.
            Some(Type::Named(n) | Type::TypeParam(n)) if self.structs.contains_key(n) => Some(n.clone()),
            _ => None,
        }
    }

    /// Leaf fields of struct `s`, as (path, type).
    fn leaves(&self, s: &str) -> Vec<(Vec<String>, Type)> {
        let mut out = Vec::new();
        for (name, ty, _) in &self.structs[s].fields {
            match self.struct_name(Some(ty)) {
                Some(inner) => {
                    for (mut path, t) in self.leaves(&inner) {
                        path.insert(0, name.clone());
                        out.push((path, t));
                    }
                }
                None => out.push((vec![name.clone()], ty.clone())),
            }
        }
        out
    }

    /// `(root variable, field path)` of a `Var` / `Field` chain.
    fn path_of(e: &Expr) -> Option<(String, Vec<String>)> {
        match &e.kind {
            ExprKind::Var(v) => Some((v.clone(), Vec::new())),
            ExprKind::Field(obj, f) => {
                let (root, mut path) = Self::path_of(obj)?;
                path.push(f.clone());
                Some((root, path))
            }
            _ => None,
        }
    }

    /// Struct name of an expression, when it is one.
    fn struct_of(&self, e: &Expr) -> Option<String> {
        match &e.kind {
            ExprKind::Call(c, _) => match &c.kind {
                ExprKind::Var(n) if self.structs.contains_key(n) => Some(n.clone()),
                ExprKind::Var(f) => self.fns.get(f).and_then(|s| s.ret.clone()),
                _ => None,
            },
            ExprKind::Var(_) | ExprKind::Field(..) => {
                let (root, path) = Self::path_of(e)?;
                let mut s = self.env.get(&root)?.clone();
                for field in &path {
                    let ty = self.structs[&s].fields.iter().find(|(n, _, _)| n == field)?.1.clone();
                    s = self.struct_name(Some(&ty))?;
                }
                Some(s)
            }
            _ => None,
        }
    }

    /// One value per leaf of `s` (in `leaves` order) for the struct-valued expression `e`.
    fn field_values(&self, e: &Expr, s: &str) -> Option<Vec<Expr>> {
        match &e.kind {
            ExprKind::Call(c, args) => {
                let ExprKind::Var(name) = &c.kind else { return None };
                if name == s {
                    let info = &self.structs[s];
                    let mut out = Vec::new();
                    for (index, (fname, fty, default)) in info.fields.iter().enumerate() {
                        let given = args
                            .iter()
                            .find(|a| a.label.as_deref() == Some(fname.as_str()))
                            .or_else(|| args.iter().filter(|a| a.label.is_none()).nth(index));
                        let value = given.map(|a| a.value.clone()).or_else(|| default.clone())?;
                        match self.struct_name(Some(fty)) {
                            Some(inner) => out.extend(self.field_values(&value, &inner)?),
                            None => out.push(value),
                        }
                    }
                    return Some(out);
                }
                let sig = self.fns.get(name)?;
                if sig.ret.as_deref() != Some(s) {
                    return None;
                }
                let lowered = self.lower_args(name, args);
                Some(
                    self.leaves(s)
                        .iter()
                        .map(|(path, _)| {
                            mk(
                                ExprKind::Call(Box::new(var(&format!("{name}__dev__{}", path.join("__")), e)), lowered.clone()),
                                e,
                            )
                        })
                        .collect(),
                )
            }
            ExprKind::Var(_) | ExprKind::Field(..) => {
                if self.struct_of(e).as_deref() != Some(s) {
                    return None;
                }
                let (root, path) = Self::path_of(e)?;
                Some(self.leaves(s).iter().map(|(p, _)| {
                    let mut full = path.clone();
                    full.extend(p.iter().cloned());
                    var(&flat(&root, &full), e)
                }).collect())
            }
            _ => None,
        }
    }

    /// Arguments of a call to `f` for its device clone: struct arguments spread into leaves.
    fn lower_args(&self, f: &str, args: &[Arg]) -> Vec<Arg> {
        let Some(sig) = self.fns.get(f) else { return args.to_vec() };
        let mut out = Vec::new();
        for (index, arg) in args.iter().enumerate() {
            let spread = sig.params.get(index).and_then(|p| p.as_ref()).and_then(|s| self.field_values(&arg.value, s));
            match spread {
                Some(values) => out.extend(values.into_iter().map(|v| Arg { label: None, value: v, spread: false, default_rest: false })),
                None => out.push(arg.clone()),
            }
        }
        out
    }

    /// Statements declaring the leaves of `name: s`, initialised from `value`.
    fn declare(&self, binding: &BindingKind, name: &str, s: &str, value: &Expr, line: usize, col: usize) -> Option<Vec<Stmt>> {
        let leaves = self.leaves(s);
        if let ExprKind::If(i) = &value.kind {
            let mut out = Vec::new();
            for (path, ty) in &leaves {
                let init = match unwrap_ty(ty) {
                    Type::Optional(_) => mk(ExprKind::Nil, value),
                    t => zero(t, value),
                };
                out.push(stmt_let(BindingKind::Var, &flat(name, path), ty.clone(), init, line, col));
            }
            out.extend(self.store(name, &[], s, value)?);
            return Some(out);
        }
        let values = self.field_values(value, s)?;
        Some(
            leaves
                .iter()
                .zip(values)
                .map(|((path, ty), v)| stmt_let(binding.clone(), &flat(name, path), ty.clone(), v, line, col))
                .collect(),
        )
    }

    /// Statements assigning the struct-valued `value` into `root.path` (a struct of type `s`).
    fn store(&self, root: &str, path: &[String], s: &str, value: &Expr) -> Option<Vec<Stmt>> {
        if let ExprKind::If(i) = &value.kind {
            let lower = |body: &Vec<Stmt>| -> Option<Vec<Stmt>> {
                let mut out = body.clone();
                if let Some(Stmt::Expr(last)) = out.pop() {
                    out.extend(self.store(root, path, s, &last)?);
                }
                Some(out)
            };
            let mut branches = Vec::new();
            for (c, b) in &i.branches {
                branches.push((c.clone(), lower(b)?));
            }
            let else_body = match &i.else_body {
                Some(b) => Some(lower(b)?),
                None => None,
            };
            return Some(vec![Stmt::If(IfStmt { branches, else_body, line: i.line, col: i.col })]);
        }
        let values = self.field_values(value, s)?;
        Some(
            self.leaves(s)
                .iter()
                .zip(values)
                .map(|((p, _), v)| {
                    let mut full = path.to_vec();
                    full.extend(p.iter().cloned());
                    assign(var(&flat(root, &full), value), v)
                })
                .collect(),
        )
    }

    /// The device clone(s) of `f`: struct parameters spread, a struct return split per leaf.
    fn device_clones(&self, f: &FnDecl, sig: &Sig) -> Vec<FnDecl> {
        let mut seeded = self.clone();
        seeded.env.clear();
        let mut params = Vec::new();
        for (p, s) in f.params.iter().zip(&sig.params) {
            match s {
                Some(s) => {
                    seeded.env.insert(p.name.clone(), s.clone());
                    for (path, ty) in self.leaves(s) {
                        params.push(Param {
                            name: flat(&p.name, &path),
                            ty: Some(ty),
                            mutable: false,
                            // Not `var`: a rebindable parameter is a `&mut` out-parameter on the host side.
                            rebindable: false,
                            var_mut: false,
                            default: None,
                            ..p.clone()
                        });
                    }
                }
                None => params.push(p.clone()),
            }
        }
        let build = |name: String, return_ty: Option<Type>, leaf: Option<(&str, usize)>| {
            let mut cx = seeded.clone();
            let mut body = block(f.body.clone(), &mut cx);
            if let Some((s, index)) = leaf {
                map_returns(&mut body, &cx, s, index, true);
            }
            rewrite_exprs(&mut body, &cx);
            FnDecl { name, params: params.clone(), return_ty, body, qualifier: None, ..f.clone() }
        };
        match &sig.ret {
            Some(s) => self
                .leaves(s)
                .into_iter()
                .enumerate()
                .map(|(index, (path, ty))| build(format!("{}__dev__{}", f.name, path.join("__")), Some(ty), Some((s.as_str(), index))))
                .collect(),
            None => vec![build(format!("{}__dev", f.name), f.return_ty.clone(), None)],
        }
    }
}

/// Rewrites `return e` (and the tail value) of a struct-returning function to leaf `index`.
fn map_returns(body: &mut [Stmt], cx: &Cx, s: &str, index: usize, tail: bool) {
    let map = |e: &Expr| -> Expr {
        cx.field_values(e, s).and_then(|mut v| (index < v.len()).then(|| v.swap_remove(index))).unwrap_or_else(|| e.clone())
    };
    let last = body.len().saturating_sub(1);
    for (i, stmt) in body.iter_mut().enumerate() {
        let is_tail = tail && i == last;
        match stmt {
            Stmt::Return(r) => {
                if let Some(v) = &r.value {
                    r.value = Some(map(v));
                }
            }
            Stmt::Expr(e) if is_tail => *e = map(e),
            Stmt::If(i) => {
                for (_, b) in &mut i.branches {
                    map_returns(b, cx, s, index, is_tail);
                }
                if let Some(b) = &mut i.else_body {
                    map_returns(b, cx, s, index, is_tail);
                }
            }
            Stmt::While(w) => map_returns(&mut w.body, cx, s, index, false),
            Stmt::DoWhile(w) => map_returns(&mut w.body, cx, s, index, false),
            Stmt::Loop(l) => map_returns(&mut l.body, cx, s, index, false),
            Stmt::For(f) => map_returns(&mut f.body, cx, s, index, false),
            _ => {}
        }
    }
}

fn block(stmts: Vec<Stmt>, cx: &mut Cx) -> Vec<Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        match s {
            Stmt::Let(l) => {
                let declared = cx.struct_name(l.ty.as_ref());
                let sname = declared.or_else(|| l.value.as_ref().and_then(|v| cx.struct_of(v)));
                let lowered = match (&sname, &l.value) {
                    (Some(s), Some(v)) => cx.declare(&l.binding, &l.name, s, v, l.line, l.col),
                    _ => None,
                };
                match (sname, lowered) {
                    (Some(s), Some(stmts)) => {
                        cx.env.insert(l.name.clone(), s);
                        out.extend(stmts);
                    }
                    _ => out.push(Stmt::Let(l)),
                }
            }
            Stmt::Expr(Expr { kind: ExprKind::Assign(lhs, rhs), line, col, len }) => {
                let lowered = cx.struct_of(&lhs).and_then(|s| {
                    let (root, path) = Cx::path_of(&lhs)?;
                    cx.store(&root, &path, &s, &rhs)
                });
                match lowered {
                    Some(stmts) => out.extend(stmts),
                    None => out.push(Stmt::Expr(Expr { kind: ExprKind::Assign(lhs, rhs), line, col, len })),
                }
            }
            Stmt::If(mut i) => {
                for (_, b) in &mut i.branches {
                    *b = block(std::mem::take(b), cx);
                }
                i.else_body = i.else_body.take().map(|b| block(b, cx));
                out.push(Stmt::If(i));
            }
            Stmt::IfLet(mut i) => {
                i.then_body = block(std::mem::take(&mut i.then_body), cx);
                for b in &mut i.elif_branches {
                    b.body = block(std::mem::take(&mut b.body), cx);
                }
                i.else_body = i.else_body.take().map(|b| block(b, cx));
                out.push(Stmt::IfLet(i));
            }
            Stmt::While(mut w) => {
                w.body = block(std::mem::take(&mut w.body), cx);
                out.push(Stmt::While(w));
            }
            Stmt::DoWhile(mut w) => {
                w.body = block(std::mem::take(&mut w.body), cx);
                out.push(Stmt::DoWhile(w));
            }
            Stmt::Loop(mut l) => {
                l.body = block(std::mem::take(&mut l.body), cx);
                out.push(Stmt::Loop(l));
            }
            Stmt::For(mut f) => {
                f.body = block(std::mem::take(&mut f.body), cx);
                out.push(Stmt::For(f));
            }
            other => out.push(other),
        }
    }
    out
}

/// Expression-level rewrites: leaf field reads (`p.a`, `p.n.x` → `p__a`, `p__n__x`) and calls of
/// struct-parameter functions (renamed to their device clone, struct arguments spread).
fn rewrite_exprs(body: &mut [Stmt], cx: &Cx) {
    Walker {
        expr: &mut |e| {
            let replacement: Option<Expr> = match &e.kind {
                ExprKind::Field(..) => match Cx::path_of(e) {
                    Some((root, path)) if cx.env.contains_key(&root) && cx.struct_of(e).is_none() => {
                        Some(var(&flat(&root, &path), e))
                    }
                    _ => None,
                },
                ExprKind::Call(callee, args) => match &callee.kind {
                    ExprKind::Var(f) if cx.fns.get(f).is_some_and(|s| s.ret.is_none()) => Some(mk(
                        ExprKind::Call(Box::new(var(&format!("{f}__dev"), callee)), cx.lower_args(f, args)),
                        e,
                    )),
                    _ => None,
                },
                _ => None,
            };
            if let Some(new) = replacement {
                *e = new;
            }
        },
        stmt: &mut |_| {},
    }
    .stmts(body);
}
