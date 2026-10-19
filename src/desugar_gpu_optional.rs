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

//! Optional (`T?`) values inside device code for the Metal / wgpu / CUDA / ROCm backends,
//! which have no `Option` type of their own. The pass lowers them, source-to-source, to plain
//! values so every backend inherits the support unchanged (simd has a real `Option` and does
//! not use this pass):
//!
//! * a local `m: T?` becomes a presence flag `m__has: bool` plus a value `m: T`;
//! * a fixed array `[T?, N]` becomes `arr__has: [int32, N]` (0/1) plus `arr: [T, N]`;
//! * a free function with a `T?` parameter and/or return gets a device clone `f__dev`: each
//!   optional parameter `a` is split into `a__has: int32` (0/1) + `a: T`, and an optional
//!   return into two functions, `f__dev` (the value) and `f__dev_has` (the 0/1 flag). Call
//!   sites in device code are rewritten to the clones; the original stays for host code.
//!
//! Supported forms, with the host's meaning: `let`/`var T? m = nil | <T value> | some(..) |
//! <optional local> | <optional call> | <optional array element> | if c: .. else: nil`,
//! assignment to `m` / `arr[i]`, `if let`/`elif let`/`else`, `m else d`, `== nil`, `!= nil`,
//! `return`/tail of an optional-returning function, and passing any of those as an argument.
//! The initializer of a `T?` local, and the optional types themselves, must be written.
//! An optional-valued call in an `if let`/`else` is evaluated once per use of its flag/value
//! (the callee runs twice), so it must be free of side effects.
//!
//! Optional struct fields are handled by `desugar_gpu_struct` (which runs first and leaves
//! them as optional locals). Not lowered: optional kernel buffer fields, `[T?]` buffers.

use crate::ast::*;
use crate::ast_walk::Walker;
use std::collections::{HashMap, HashSet};

/// Returns `program` with device-side optionals lowered.
pub fn desugar(program: &Program) -> Program {
    let mut program = program.clone();
    let mut fns: HashMap<String, Sig> = HashMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if f.qualifier.is_none() && !f.task {
                let params: Vec<Option<Type>> = f.params.iter().map(|p| optional_inner(p.ty.as_ref())).collect();
                let ret = optional_inner(f.return_ty.as_ref());
                if params.iter().any(Option::is_some) || ret.is_some() {
                    fns.insert(f.name.clone(), Sig { params, ret });
                }
            }
        }
    }
    let cx = Cx { fns, ..Cx::default() };
    // As with struct scalar replacement, host-only functions must keep their original
    // optional representation. Besides producing dead code, lowering every optional
    // function creates ill-typed Rust twins whose bodies still contain host-only forms
    // such as `None` and enum payloads.
    let (reach_fns, _) = crate::desugar_gpu_struct::device_reachable(&program);
    let mut clones = Vec::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if let Some(sig) = cx.fns.get(&f.name).filter(|_| reach_fns.contains(&f.name)) {
                clones.extend(cx.device_clones(f, sig));
            }
        }
    }
    for item in &mut program.items {
        if let Item::Kernel(k) = item {
            for m in &mut k.methods {
                let mut local = cx.clone();
                m.body = block(std::mem::take(&mut m.body), &mut local);
                rewrite_exprs(&mut m.body, &local);
            }
        }
    }
    program.items.extend(clones.into_iter().map(Item::Fn));
    program
}

#[derive(Clone)]
struct Sig {
    /// `Some(T)` for each `T?` parameter.
    params: Vec<Option<Type>>,
    /// `Some(T)` for a `T?` return.
    ret: Option<Type>,
}

#[derive(Clone, Default)]
struct Cx {
    /// Optional locals (and parameters) → payload type.
    opts: HashMap<String, Type>,
    /// Optional locals whose flag is an int32 (parameters), not a bool.
    int_flag: HashSet<String>,
    /// Optional fixed arrays → element payload type.
    arrs: HashMap<String, Type>,
    fns: HashMap<String, Sig>,
}

fn unwrap_ty(t: &Type) -> &Type {
    match t {
        Type::Mut(x) | Type::Qualified(x, _) => unwrap_ty(x),
        _ => t,
    }
}

fn optional_inner(t: Option<&Type>) -> Option<Type> {
    match t.map(unwrap_ty) {
        Some(Type::Optional(inner)) => Some((**inner).clone()),
        _ => None,
    }
}

fn has_name(name: &str) -> String {
    format!("{name}__has")
}

fn mk(kind: ExprKind, at: &Expr) -> Expr {
    Expr { kind, line: at.line, col: at.col, len: at.len }
}

fn var(name: &str, at: &Expr) -> Expr {
    mk(ExprKind::Var(name.to_string()), at)
}

fn int(n: i64, at: &Expr) -> Expr {
    mk(ExprKind::Int(n), at)
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

fn ternary(cond: Expr, a: Expr, b: Expr, at: &Expr) -> Expr {
    mk(
        ExprKind::If(Box::new(IfStmt {
            branches: vec![(cond, vec![Stmt::Expr(a)])],
            else_body: Some(vec![Stmt::Expr(b)]),
            line: at.line,
            col: at.col,
        })),
        at,
    )
}

/// A 0/1 int32 from a boolean expression.
fn flag_int(h: Expr) -> Expr {
    match h.kind {
        ExprKind::Bool(true) => int(1, &h),
        ExprKind::Bool(false) => int(0, &h),
        _ => {
            let at = h.clone();
            ternary(h, int(1, &at), int(0, &at), &at)
        }
    }
}

fn not_zero(e: Expr) -> Expr {
    let at = e.clone();
    mk(ExprKind::BinOp(BinOp::NotEq, Box::new(e), Box::new(int(0, &at))), &at)
}

fn stmt_let(name: &str, ty: Option<Type>, value: Expr, line: usize, col: usize) -> Stmt {
    Stmt::Let(LetStmt {
        binding: BindingKind::Var,
        is_pub: false,
        is_static: false,
        name: name.to_string(),
        ty,
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

fn is_some_call(e: &Expr) -> Option<&Expr> {
    match &e.kind {
        ExprKind::Call(c, args) if args.len() == 1 && matches!(&c.kind, ExprKind::Var(n) if n == "some") => {
            Some(&args[0].value)
        }
        _ => None,
    }
}

fn plain_arg(value: Expr) -> Arg {
    Arg { label: None, value, spread: false, default_rest: false }
}

/// The single expression of a one-statement body, if it is one.
fn single_expr(body: &[Stmt]) -> Option<&Expr> {
    match body {
        [Stmt::Expr(e)] => Some(e),
        _ => None,
    }
}

impl Cx {
    /// Presence of the optional local `m` as a boolean expression.
    fn has_local(&self, m: &str, at: &Expr) -> Expr {
        if self.int_flag.contains(m) {
            not_zero(var(&has_name(m), at))
        } else {
            var(&has_name(m), at)
        }
    }

    fn set_flag(&self, m: &str, has: Expr, at: &Expr) -> Stmt {
        let target = var(&has_name(m), at);
        assign(target, if self.int_flag.contains(m) { flag_int(has) } else { has })
    }

    /// `(has, value, payload type)` for an expression that is optional by construction:
    /// an optional local, a call of an optional-returning function, an element of an
    /// optional array, or an `if` expression with such a branch (or `nil`).
    fn parts_opt(&self, e: &Expr) -> Option<(Expr, Expr, Type)> {
        match &e.kind {
            ExprKind::Var(m) => self.opts.get(m).map(|t| (self.has_local(m, e), e.clone(), t.clone())),
            ExprKind::Call(callee, args) => {
                let ExprKind::Var(f) = &callee.kind else { return None };
                let inner = self.fns.get(f)?.ret.clone()?;
                let lowered = self.lower_args(f, args);
                let call = |name: String| {
                    mk(ExprKind::Call(Box::new(var(&name, e)), lowered.clone()), e)
                };
                Some((not_zero(call(format!("{f}__dev_has"))), call(format!("{f}__dev")), inner))
            }
            ExprKind::Index(base, idx) => {
                let ExprKind::Var(arr) = &base.kind else { return None };
                let inner = self.arrs.get(arr)?.clone();
                let at = |name: &str| mk(ExprKind::Index(Box::new(var(name, e)), idx.clone()), e);
                Some((not_zero(at(&has_name(arr))), at(arr), inner))
            }
            ExprKind::If(i) => {
                let inner = i
                    .branches
                    .iter()
                    .map(|(_, b)| b)
                    .chain(i.else_body.iter())
                    .filter_map(|b| single_expr(b).and_then(|x| self.parts_opt(x)))
                    .map(|(_, _, t)| t)
                    .next()?;
                let (has, val) = self.if_parts(i, &inner, e)?;
                Some((has, val, inner))
            }
            _ => None,
        }
    }

    /// `if` expression over single-expression branches → (has, value) ternaries.
    fn if_parts(&self, i: &IfStmt, inner: &Type, at: &Expr) -> Option<(Expr, Expr)> {
        let mut has_branches = Vec::new();
        let mut val_branches = Vec::new();
        for (c, b) in &i.branches {
            let (h, v) = self.parts(single_expr(b)?, inner);
            has_branches.push((c.clone(), vec![Stmt::Expr(h)]));
            val_branches.push((c.clone(), vec![Stmt::Expr(v)]));
        }
        let (eh, ev) = self.parts(single_expr(i.else_body.as_ref()?)?, inner);
        let build = |branches, last| {
            mk(
                ExprKind::If(Box::new(IfStmt { branches, else_body: Some(vec![Stmt::Expr(last)]), line: at.line, col: at.col })),
                at,
            )
        };
        Some((build(has_branches, eh), build(val_branches, ev)))
    }

    /// `(has, value)` for any expression stored into a `T?` destination: `nil`, `some(x)`,
    /// an optional-valued expression, or a plain `T` value.
    fn parts(&self, e: &Expr, inner: &Type) -> (Expr, Expr) {
        if let ExprKind::Nil = e.kind {
            return (mk(ExprKind::Bool(false), e), zero(inner, e));
        }
        if let Some((h, v, _)) = self.parts_opt(e) {
            return (h, v);
        }
        let plain = is_some_call(e).unwrap_or(e);
        (mk(ExprKind::Bool(true), e), plain.clone())
    }

    /// Arguments of a call to `f` for its device clone: each optional parameter becomes
    /// `(has as 0/1, value)`.
    fn lower_args(&self, f: &str, args: &[Arg]) -> Vec<Arg> {
        let Some(sig) = self.fns.get(f) else { return args.to_vec() };
        let mut out = Vec::new();
        for (index, arg) in args.iter().enumerate() {
            match sig.params.get(index).and_then(|p| p.as_ref()) {
                Some(inner) if arg.label.is_none() => {
                    let (h, v) = self.parts(&arg.value, inner);
                    out.push(plain_arg(flag_int(h)));
                    out.push(plain_arg(v));
                }
                _ => out.push(arg.clone()),
            }
        }
        out
    }

    /// Statements storing the optional-valued `value` into the optional local `name`.
    fn store(&self, name: &str, inner: &Type, value: &Expr) -> Vec<Stmt> {
        if let ExprKind::If(i) = &value.kind {
            let lower = |body: &Vec<Stmt>| -> Vec<Stmt> {
                let mut out = body.clone();
                if let Some(Stmt::Expr(last)) = out.pop() {
                    out.extend(self.store(name, inner, &last));
                }
                out
            };
            return vec![Stmt::If(IfStmt {
                branches: i.branches.iter().map(|(c, b)| (c.clone(), lower(b))).collect(),
                else_body: i.else_body.as_ref().map(lower),
                line: i.line,
                col: i.col,
            })];
        }
        let (h, v) = self.parts(value, inner);
        if matches!(value.kind, ExprKind::Nil) {
            return vec![self.set_flag(name, h, value)];
        }
        vec![self.set_flag(name, h, value), assign(var(name, value), v)]
    }

    /// The device clone(s) of `f`: parameters split, optional return split in two.
    fn device_clones(&self, f: &FnDecl, sig: &Sig) -> Vec<FnDecl> {
        let mut seeded = self.clone();
        let mut params = Vec::new();
        for (p, opt) in f.params.iter().zip(&sig.params) {
            match opt {
                Some(inner) => {
                    seeded.opts.insert(p.name.clone(), inner.clone());
                    seeded.int_flag.insert(p.name.clone());
                    params.push(Param { name: has_name(&p.name), ty: Some(Type::Int32), mutable: false, rebindable: true, var_mut: false, ..p.clone() });
                    params.push(Param { ty: Some(inner.clone()), rebindable: true, ..p.clone() });
                }
                None => params.push(p.clone()),
            }
        }
        let build = |name: String, return_ty: Option<Type>, mode: Option<(bool, &Type)>| {
            let mut cx = seeded.clone();
            let mut body = block(f.body.clone(), &mut cx);
            if let Some((has_mode, inner)) = mode {
                map_returns(&mut body, &cx, inner, has_mode, true);
            }
            rewrite_exprs(&mut body, &cx);
            FnDecl { name, params: params.clone(), return_ty, body, qualifier: None, ..f.clone() }
        };
        match &sig.ret {
            Some(inner) => vec![
                build(format!("{}__dev", f.name), Some(inner.clone()), Some((false, inner))),
                build(format!("{}__dev_has", f.name), Some(Type::Int32), Some((true, inner))),
            ],
            None => vec![build(format!("{}__dev", f.name), f.return_ty.clone(), None)],
        }
    }
}

/// Rewrites `return e` (and the implicit tail value) of an optional-returning function to the
/// value part (`has_mode == false`) or the 0/1 flag part (`true`).
fn map_returns(body: &mut [Stmt], cx: &Cx, inner: &Type, has_mode: bool, tail: bool) {
    let map = |e: &Expr| -> Expr {
        let (h, v) = cx.parts(e, inner);
        if has_mode { flag_int(h) } else { v }
    };
    let last = body.len().saturating_sub(1);
    for (index, stmt) in body.iter_mut().enumerate() {
        let is_tail = tail && index == last;
        match stmt {
            Stmt::Return(r) => {
                if let Some(v) = &r.value {
                    r.value = Some(map(v));
                }
            }
            Stmt::Expr(e) if is_tail => *e = map(e),
            Stmt::If(i) => {
                for (_, b) in &mut i.branches {
                    map_returns(b, cx, inner, has_mode, is_tail);
                }
                if let Some(b) = &mut i.else_body {
                    map_returns(b, cx, inner, has_mode, is_tail);
                }
            }
            Stmt::While(w) => map_returns(&mut w.body, cx, inner, has_mode, false),
            Stmt::DoWhile(w) => map_returns(&mut w.body, cx, inner, has_mode, false),
            Stmt::Loop(l) => map_returns(&mut l.body, cx, inner, has_mode, false),
            Stmt::For(f) => map_returns(&mut f.body, cx, inner, has_mode, false),
            _ => {}
        }
    }
}

fn block(stmts: Vec<Stmt>, cx: &mut Cx) -> Vec<Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        match s {
            Stmt::Let(l) => match l.ty.as_ref().map(unwrap_ty) {
                Some(Type::Optional(inner)) => {
                    let inner = (**inner).clone();
                    let at = l.value.clone().unwrap_or(Expr { kind: ExprKind::Nil, line: l.line, col: l.col, len: 0 });
                    // The flag's type is inferred: wgpu stores a declared `bool` as `u32`.
                    out.push(stmt_let(&has_name(&l.name), None, mk(ExprKind::Bool(false), &at), l.line, l.col));
                    out.push(stmt_let(&l.name, Some(inner.clone()), zero(&inner, &at), l.line, l.col));
                    cx.opts.insert(l.name.clone(), inner.clone());
                    cx.int_flag.remove(&l.name);
                    if let Some(v) = &l.value {
                        out.extend(cx.store(&l.name, &inner, v));
                    }
                }
                Some(Type::ArrayN(elem, n)) if matches!(unwrap_ty(elem), Type::Optional(_)) => {
                    let Type::Optional(inner) = unwrap_ty(elem) else { unreachable!() };
                    let inner = (**inner).clone();
                    let init = l.value.as_ref().and_then(|v| lower_array_init(cx, v, &inner));
                    match (&l.value, init) {
                        (Some(_), None) => out.push(Stmt::Let(l)),
                        (_, init) => {
                            let at = Expr { kind: ExprKind::Nil, line: l.line, col: l.col, len: 0 };
                            let (flags, values) = init.unwrap_or_else(|| {
                                (
                                    mk(ExprKind::ArrayFill { value: Box::new(int(0, &at)), count: Box::new(int(*n as i64, &at)) }, &at),
                                    mk(ExprKind::ArrayFill { value: Box::new(zero(&inner, &at)), count: Box::new(int(*n as i64, &at)) }, &at),
                                )
                            });
                            out.push(stmt_let(&has_name(&l.name), Some(Type::ArrayN(Box::new(Type::Int32), *n)), flags, l.line, l.col));
                            out.push(stmt_let(&l.name, Some(Type::ArrayN(Box::new(inner.clone()), *n)), values, l.line, l.col));
                            cx.arrs.insert(l.name.clone(), inner);
                        }
                    }
                }
                _ => out.push(Stmt::Let(l)),
            },
            Stmt::Expr(Expr { kind: ExprKind::Assign(lhs, rhs), line, col, len }) => {
                let handled = match &lhs.kind {
                    ExprKind::Var(name) if cx.opts.contains_key(name) => {
                        let inner = cx.opts[name].clone();
                        Some(cx.store(name, &inner, &rhs))
                    }
                    ExprKind::Index(base, idx) => match &base.kind {
                        ExprKind::Var(arr) if cx.arrs.contains_key(arr) => {
                            let inner = cx.arrs[arr].clone();
                            let (h, v) = cx.parts(&rhs, &inner);
                            let at = |name: &str| mk(ExprKind::Index(Box::new(var(name, &lhs)), idx.clone()), &lhs);
                            Some(vec![assign(at(&has_name(arr)), flag_int(h)), assign(at(arr), v)])
                        }
                        _ => None,
                    },
                    _ => None,
                };
                match handled {
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
            Stmt::IfLet(i) => out.extend(lower_if_let(i, cx)),
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

/// `[a, nil, c]` / `[nil for ..<n]` initializers of a `[T?, N]` array → (flag array, value array).
fn lower_array_init(cx: &Cx, value: &Expr, inner: &Type) -> Option<(Expr, Expr)> {
    match &value.kind {
        ExprKind::Array(elems) => {
            let (flags, values): (Vec<Expr>, Vec<Expr>) = elems
                .iter()
                .map(|e| {
                    let (h, v) = cx.parts(e, inner);
                    (flag_int(h), v)
                })
                .unzip();
            Some((mk(ExprKind::Array(flags), value), mk(ExprKind::Array(values), value)))
        }
        ExprKind::ArrayFill { value: fill, count } => {
            let (h, v) = cx.parts(fill, inner);
            Some((
                mk(ExprKind::ArrayFill { value: Box::new(flag_int(h)), count: count.clone() }, value),
                mk(ExprKind::ArrayFill { value: Box::new(v), count: count.clone() }, value),
            ))
        }
        _ => None,
    }
}

/// `if let v = <optional> [, cond] ... elif ... else ...` becomes a plain `if/else` on the
/// presence flags; each binding is declared inside its branch. Clause lists it cannot lower
/// (patterns, expressions that are not optional-valued) are left unchanged.
fn lower_if_let(i: IfLetStmt, cx: &mut Cx) -> Vec<Stmt> {
    let lowerable = |clauses: &[CondClause], cx: &Cx| {
        clauses.iter().all(|c| match c {
            CondClause::Let(_, e) => cx.parts_opt(e).is_some(),
            CondClause::LetPat(..) => false,
            CondClause::Expr(_) => true,
        })
    };
    let all = lowerable(&i.clauses, cx) && i.elif_branches.iter().all(|b| lowerable(&b.clauses, cx));
    if !all {
        let mut i = i;
        i.then_body = block(std::mem::take(&mut i.then_body), cx);
        for b in &mut i.elif_branches {
            b.body = block(std::mem::take(&mut b.body), cx);
        }
        i.else_body = i.else_body.take().map(|b| block(b, cx));
        return vec![Stmt::IfLet(i)];
    }
    let branch = |clauses: Vec<CondClause>, body: Vec<Stmt>, cx: &mut Cx| {
        let mut cond: Option<Expr> = None;
        // (binding, optional source expression)
        let mut binds: Vec<(String, Expr, Type)> = Vec::new();
        for c in clauses {
            let term = match c {
                CondClause::Let(v, e) => {
                    let (has, val, inner) = cx.parts_opt(&e).unwrap();
                    binds.push((v, val, inner));
                    has
                }
                CondClause::Expr(mut e) => {
                    // Later boolean clauses see a binding as the optional's own value.
                    let renames: HashMap<String, Expr> = binds.iter().map(|(v, val, _)| (v.clone(), val.clone())).collect();
                    Walker {
                        expr: &mut |x| {
                            if let ExprKind::Var(n) = &x.kind {
                                if let Some(val) = renames.get(n) {
                                    *x = val.clone();
                                }
                            }
                        },
                        stmt: &mut |_| {},
                    }
                    .expr(&mut e);
                    e
                }
                CondClause::LetPat(..) => unreachable!(),
            };
            cond = Some(match cond {
                None => term,
                Some(prev) => {
                    let at = term.clone();
                    mk(ExprKind::BinOp(BinOp::And, Box::new(prev), Box::new(term)), &at)
                }
            });
        }
        let mut stmts = Vec::new();
        for (v, val, inner) in &binds {
            stmts.push(stmt_let(v, Some(inner.clone()), val.clone(), i.line, i.col));
        }
        stmts.extend(body);
        (cond.unwrap_or(Expr { kind: ExprKind::Bool(true), line: i.line, col: i.col, len: 0 }), block(stmts, cx))
    };
    let mut branches = vec![branch(i.clauses.clone(), i.then_body.clone(), cx)];
    for b in i.elif_branches.clone() {
        branches.push(branch(b.clauses, b.body, cx));
    }
    let else_body = i.else_body.clone().map(|b| block(b, cx));
    vec![Stmt::If(IfStmt { branches, else_body, line: i.line, col: i.col })]
}

/// Expression-level rewrites: `x else d`, `x == nil`, `x != nil` over optional-valued `x`,
/// and calls of optional-parameter functions (renamed to their device clone, arguments split).
fn rewrite_exprs(body: &mut [Stmt], cx: &Cx) {
    Walker {
        expr: &mut |e| {
            let replacement: Option<Expr> = match &e.kind {
                ExprKind::Else(l, r) => cx.parts_opt(l).map(|(has, val, _)| ternary(has, val, (**r).clone(), e)),
                ExprKind::BinOp(op @ (BinOp::Eq | BinOp::NotEq | BinOp::Is | BinOp::IsNot), l, r) => {
                    let opt = match (&l.kind, &r.kind) {
                        (_, ExprKind::Nil) => cx.parts_opt(l),
                        (ExprKind::Nil, _) => cx.parts_opt(r),
                        _ => None,
                    };
                    opt.map(|(has, _, _)| {
                        if matches!(op, BinOp::Eq | BinOp::Is) {
                            mk(ExprKind::UnaryOp(UnaryOp::Not, Box::new(has.clone())), &has)
                        } else {
                            has
                        }
                    })
                }
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
