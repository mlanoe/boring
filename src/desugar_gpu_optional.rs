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

//! Optional (`T?`) locals inside kernel bodies for the Metal / wgpu / CUDA / ROCm device
//! backends, which have no `Option` type of their own. Each optional local `m: T?` is
//! lowered, as a source-to-source pass, into two ordinary locals — a presence flag
//! `m__has: bool` and a value `m: T` — so every backend inherits the support unchanged
//! (simd has a real `Option` and does not use this pass).
//!
//! Supported on a kernel method's locals, with the same meaning as on the host:
//! * `let`/`var T? m = nil | <T value> | some(<T value>) | <optional local> | if c: .. else: nil`
//!   (the type must be written; its value type is the zero-initialised payload)
//! * `m = <same forms>`
//! * `if let v = m [, cond]:` / `elif let ...` / `else:` over optional locals
//! * `m else d`, `m == nil`, `m != nil`
//!
//! Optional parameters, returns, struct fields and arrays are not lowered (the backends
//! reject or ignore them as before).

use crate::ast::*;
use crate::ast_walk::Walker;
use std::collections::HashMap;

/// Returns `program` with every kernel method's optional locals lowered.
pub fn desugar(program: &Program) -> Program {
    let mut program = program.clone();
    for item in &mut program.items {
        if let Item::Kernel(k) = item {
            for m in &mut k.methods {
                let mut opts = HashMap::new();
                m.body = block(std::mem::take(&mut m.body), &mut opts);
                if !opts.is_empty() {
                    rewrite_exprs(&mut m.body, &opts);
                }
            }
        }
    }
    program
}

fn unwrap_ty(t: &Type) -> &Type {
    match t {
        Type::Mut(x) | Type::Qualified(x, _) => unwrap_ty(x),
        _ => t,
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

fn assign(name: &str, value: Expr, at: &Expr) -> Stmt {
    Stmt::Expr(mk(ExprKind::Assign(Box::new(var(name, at)), Box::new(value)), at))
}

fn is_some_call(e: &Expr) -> Option<&Expr> {
    match &e.kind {
        ExprKind::Call(c, args) if args.len() == 1 && matches!(&c.kind, ExprKind::Var(n) if n == "some") => {
            Some(&args[0].value)
        }
        _ => None,
    }
}

/// Statements storing the optional-valued `value` into the optional local `name`.
fn store(name: &str, value: &Expr, opts: &HashMap<String, Type>) -> Vec<Stmt> {
    let flag = has_name(name);
    match &value.kind {
        ExprKind::Nil => vec![assign(&flag, mk(ExprKind::Bool(false), value), value)],
        ExprKind::Var(other) if opts.contains_key(other) => vec![
            assign(&flag, var(&has_name(other), value), value),
            assign(name, var(other, value), value),
        ],
        ExprKind::If(i) => {
            let lower = |body: &Vec<Stmt>| -> Vec<Stmt> {
                let mut out = body.clone();
                if let Some(Stmt::Expr(last)) = out.pop() {
                    out.extend(store(name, &last, opts));
                }
                out
            };
            vec![Stmt::If(IfStmt {
                branches: i.branches.iter().map(|(c, b)| (c.clone(), lower(b))).collect(),
                else_body: i.else_body.as_ref().map(lower),
                line: i.line,
                col: i.col,
            })]
        }
        _ => {
            let plain = is_some_call(value).unwrap_or(value);
            vec![
                assign(&flag, mk(ExprKind::Bool(true), value), value),
                assign(name, plain.clone(), value),
            ]
        }
    }
}

fn optional_local(e: &Expr, opts: &HashMap<String, Type>) -> Option<String> {
    match &e.kind {
        ExprKind::Var(n) if opts.contains_key(n) => Some(n.clone()),
        _ => None,
    }
}

fn block(stmts: Vec<Stmt>, opts: &mut HashMap<String, Type>) -> Vec<Stmt> {
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
                    opts.insert(l.name.clone(), inner);
                    if let Some(v) = &l.value {
                        out.extend(store(&l.name, v, opts));
                    }
                }
                _ => out.push(Stmt::Let(l)),
            },
            Stmt::Expr(Expr { kind: ExprKind::Assign(lhs, rhs), line, col, len }) => {
                match optional_local(&lhs, opts) {
                    Some(name) => out.extend(store(&name, &rhs, opts)),
                    None => out.push(Stmt::Expr(Expr { kind: ExprKind::Assign(lhs, rhs), line, col, len })),
                }
            }
            Stmt::If(mut i) => {
                for (_, b) in &mut i.branches {
                    *b = block(std::mem::take(b), opts);
                }
                i.else_body = i.else_body.take().map(|b| block(b, opts));
                out.push(Stmt::If(i));
            }
            Stmt::IfLet(i) => out.extend(lower_if_let(i, opts)),
            Stmt::While(mut w) => {
                w.body = block(std::mem::take(&mut w.body), opts);
                out.push(Stmt::While(w));
            }
            Stmt::DoWhile(mut w) => {
                w.body = block(std::mem::take(&mut w.body), opts);
                out.push(Stmt::DoWhile(w));
            }
            Stmt::Loop(mut l) => {
                l.body = block(std::mem::take(&mut l.body), opts);
                out.push(Stmt::Loop(l));
            }
            Stmt::For(mut f) => {
                f.body = block(std::mem::take(&mut f.body), opts);
                out.push(Stmt::For(f));
            }
            other => out.push(other),
        }
    }
    out
}

/// `if let v = m [, cond] ... elif ... else ...` over optional locals becomes a plain
/// `if/else` on the presence flags; each binding is declared inside its branch.
/// Clause lists it cannot lower (patterns, non-local optionals) are left unchanged.
fn lower_if_let(i: IfLetStmt, opts: &mut HashMap<String, Type>) -> Vec<Stmt> {
    let lowerable = |clauses: &[CondClause], opts: &HashMap<String, Type>| {
        clauses.iter().all(|c| match c {
            CondClause::Let(_, e) => optional_local(e, opts).is_some(),
            CondClause::LetPat(..) => false,
            CondClause::Expr(_) => true,
        })
    };
    let all = lowerable(&i.clauses, opts) && i.elif_branches.iter().all(|b| lowerable(&b.clauses, opts));
    if !all {
        let mut i = i;
        i.then_body = block(std::mem::take(&mut i.then_body), opts);
        for b in &mut i.elif_branches {
            b.body = block(std::mem::take(&mut b.body), opts);
        }
        i.else_body = i.else_body.take().map(|b| block(b, opts));
        return vec![Stmt::IfLet(i)];
    }
    let branch = |clauses: Vec<CondClause>, body: Vec<Stmt>, opts: &mut HashMap<String, Type>| {
        // Bindings `let v = m` read `m` itself, so each is an ordinary copy at the top of the
        // body; later boolean clauses that mention a binding are evaluated against `m`.
        let mut cond: Option<Expr> = None;
        let mut binds: Vec<(String, String)> = Vec::new();
        for c in clauses {
            let term = match c {
                CondClause::Let(v, e) => {
                    let m = optional_local(&e, opts).unwrap();
                    binds.push((v, m.clone()));
                    var(&has_name(&m), &e)
                }
                CondClause::Expr(mut e) => {
                    // Replace uses of a binding with the optional local it names.
                    let renames: HashMap<String, String> = binds.iter().cloned().collect();
                    Walker {
                        expr: &mut |x| {
                            if let ExprKind::Var(n) = &x.kind {
                                if let Some(m) = renames.get(n) {
                                    x.kind = ExprKind::Var(m.clone());
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
        for (v, m) in &binds {
            let inner = opts[m].clone();
            let at = Expr { kind: ExprKind::Nil, line: i.line, col: i.col, len: 0 };
            stmts.push(stmt_let(v, Some(inner), var(m, &at), i.line, i.col));
        }
        stmts.extend(body);
        (cond.unwrap_or(Expr { kind: ExprKind::Bool(true), line: i.line, col: i.col, len: 0 }), block(stmts, opts))
    };
    let mut branches = Vec::new();
    branches.push(branch(i.clauses.clone(), i.then_body.clone(), opts));
    for b in i.elif_branches.clone() {
        branches.push(branch(b.clauses, b.body, opts));
    }
    let else_body = i.else_body.clone().map(|b| block(b, opts));
    vec![Stmt::If(IfStmt { branches, else_body, line: i.line, col: i.col })]
}

/// Expression-level rewrites for optional locals: `m else d`, `m == nil`, `m != nil`.
fn rewrite_exprs(body: &mut [Stmt], opts: &HashMap<String, Type>) {
    Walker {
        expr: &mut |e| {
            let replacement = match &e.kind {
                ExprKind::Else(l, r) => optional_local(l, opts).map(|m| {
                    ExprKind::If(Box::new(IfStmt {
                        branches: vec![(var(&has_name(&m), l), vec![Stmt::Expr((**l).clone())])],
                        else_body: Some(vec![Stmt::Expr((**r).clone())]),
                        line: e.line,
                        col: e.col,
                    }))
                }),
                ExprKind::BinOp(op @ (BinOp::Eq | BinOp::NotEq | BinOp::Is | BinOp::IsNot), l, r) => {
                    let (m, at) = match (&l.kind, &r.kind) {
                        (_, ExprKind::Nil) => (optional_local(l, opts), l),
                        (ExprKind::Nil, _) => (optional_local(r, opts), r),
                        _ => (None, l),
                    };
                    m.map(|m| {
                        let flag = var(&has_name(&m), at);
                        if matches!(op, BinOp::Eq | BinOp::Is) {
                            ExprKind::UnaryOp(UnaryOp::Not, Box::new(flag))
                        } else {
                            flag.kind
                        }
                    })
                }
                _ => None,
            };
            if let Some(kind) = replacement {
                e.kind = kind;
            }
        },
        stmt: &mut |_| {},
    }
    .stmts(body);
}
