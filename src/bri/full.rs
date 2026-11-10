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

//! `full` level of the `.bri`: the ownership qualifiers the transpiler's inference chose
//! (`--show-qualifiers`, CPU half) written onto the AST before it is printed.
//!
//! A decision is `(function, name, line, col)` -> qualifier, as recorded by
//! `Transpiler::record_cpu_decisions`; it is applied to the parameter or `let` it names in the
//! parsed program (positions are those of the source, which the later desugarings keep).
//! A binding that already carries a qualifier is left alone (a written qualifier is evidence,
//! an inferred one is not: the file must not turn the second into the first by itself).

use crate::ast::*;

/// One inferred qualifier of a parameter or `let`.
#[derive(Debug, Clone)]
pub struct QualDecision {
    /// Name of the function the binding lives in.
    pub scope: String,
    pub name: String,
    pub line: usize,
    pub col: usize,
    pub is_param: bool,
    /// The decision does not say whether it names a parameter or a `let` (GPU residency): the
    /// position finds the binding.
    pub either_kind: bool,
    pub qual: OwnerQual,
}

/// Result of applying decisions: how many were written, and how many could not be.
#[derive(Debug, Default, Clone)]
pub struct Applied {
    pub written: usize,
    /// `(scope, name, why)` of each decision left out.
    pub skipped: Vec<(String, String, &'static str)>,
}

/// Writes the decisions onto `program`.
pub fn apply_qualifiers(program: &mut Program, decisions: &[QualDecision]) -> Applied {
    let mut applied = Applied::default();
    let mut used = vec![false; decisions.len()];
    for_each_fn(&mut program.items, &mut |f: &mut FnDecl| {
        for (i, d) in decisions.iter().enumerate() {
            if d.scope != f.name { continue; }
            if d.is_param || d.either_kind {
                if let Some(p) = f.params.iter_mut().find(|p| p.name == d.name && p.line == d.line && p.col == d.col) {
                    used[i] = true;
                    match set_param_qual(p, &d.qual) {
                        Ok(()) => applied.written += 1,
                        Err(why) => applied.skipped.push((d.scope.clone(), d.name.clone(), why)),
                    }
                }
            }
        }
        // Lets of this function body (nested `fn` bodies are entered by the walker; their own
        // decisions carry their own scope and are applied when that function is visited).
        let scope = f.name.clone();
        let mut on_expr = |_: &mut Expr| {};
        let mut on_stmt = |s: &mut Stmt| {
            let Stmt::Let(l) = s else { return };
            for (i, d) in decisions.iter().enumerate() {
                if (d.is_param && !d.either_kind) || d.scope != scope || d.name != l.name || d.line != l.line || d.col != l.col { continue; }
                used[i] = true;
                match set_let_qual(l, &d.qual) {
                    Ok(()) => applied.written += 1,
                    Err(why) => applied.skipped.push((d.scope.clone(), d.name.clone(), why)),
                }
            }
        };
        let mut walker = crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        walker.stmts(&mut f.body);
    });
    // The file's own statements (and those of a `mod`) have no function around them.
    top_level_lets(&mut program.items, &mut |l: &mut LetStmt| {
        for (i, d) in decisions.iter().enumerate() {
            if (d.is_param && !d.either_kind) || d.scope != "<top level>" || d.name != l.name || d.line != l.line || d.col != l.col { continue; }
            used[i] = true;
            match set_let_qual(l, &d.qual) {
                Ok(()) => applied.written += 1,
                Err(why) => applied.skipped.push((d.scope.clone(), d.name.clone(), why)),
            }
        }
    });
    for (i, d) in decisions.iter().enumerate() {
        if !used[i] {
            applied.skipped.push((d.scope.clone(), d.name.clone(), "binding not found in the parsed program"));
        }
    }
    applied
}

fn top_level_lets(items: &mut [Item], f: &mut dyn FnMut(&mut LetStmt)) {
    for item in items.iter_mut() {
        match item {
            Item::Let(l) => f(l),
            Item::Stmt(Stmt::Let(l)) => f(l),
            Item::Mod(m) => top_level_lets(&mut m.items, f),
            _ => {}
        }
    }
}

fn writable(q: &OwnerQual) -> bool {
    matches!(q, OwnerQual::GpuGlobal | OwnerQual::GpuUnified | OwnerQual::Owned | OwnerQual::Actor | OwnerQual::ActorTask | OwnerQual::Guard | OwnerQual::GuardTask
        | OwnerQual::Atomic | OwnerQual::Shared | OwnerQual::Inline)
}

fn already_qualified(t: &Type) -> bool {
    matches!(t.without_mut(), Type::Qualified(..))
}

fn set_let_qual(l: &mut LetStmt, q: &OwnerQual) -> Result<(), &'static str> {
    // A `mut` binding the transpiler lifted to a lock handle by usage (stored in an actor member,
    // lent to an `'actor` parameter): written down, the same word makes the binding an actor from
    // the start and the arguments are passed another way.
    if matches!(q, OwnerQual::Actor) && l.binding == BindingKind::Mut { return Err("an actor by usage is not an actor by declaration"); }
    if !writable(q) { return Err("qualifier has no `'` spelling on a `let`"); }
    // A local built in place is `inline` already: spelling it adds nothing, and on an enum value
    // or an annotated generic it would pin a type the reader inferred differently.
    if matches!(q, OwnerQual::Inline) { return Err("`inline` is what a local is by default"); }
    // Boxing an oversized value is the transpiler's call; `'owned` written on the `let` also
    // changes how it is lent and returned (`&*big` vs `&big`, `b` vs `b.clone()`).
    if matches!(q, OwnerQual::Owned) { return Err("an oversized local is boxed by the transpiler, not by declaration"); }
    // A scalar a closure captures is lifted by the closure analysis; naming `shared` on it by hand
    // takes it through another path.
    if matches!(q, OwnerQual::Shared) && l.ty.as_ref().is_none_or(|t| t.without_mut().is_scalar_value())
        && l.value.as_ref().is_some_and(|v| matches!(v.kind, ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Bool(_) | ExprKind::Float(_)))
    { return Err("a captured scalar is handled by the closure analysis"); }
    match &l.ty {
        // `let x'q = value`: the parser fills the base type in from a constructor call or a scalar
        // literal only; with any other value the `_` stays and the binding would be typed `_'q`,
        // which a later assignment rejects. Nothing is written then (the type is not known here).
        None if !base_type_inferable(l.value.as_ref()) => return Err("type of the value is not known to the printer"),
        None => l.ty = Some(Type::Qualified(Box::new(Type::Named("_".into())), q.clone())),
        Some(t) if already_qualified(t) => return Err("already written"),
        Some(Type::Mut(inner)) => l.ty = Some(Type::Mut(Box::new(Type::Qualified(inner.clone(), q.clone())))),
        Some(t) => l.ty = Some(Type::Qualified(Box::new(t.clone()), q.clone())),
    }
    Ok(())
}

fn set_param_qual(p: &mut Param, q: &OwnerQual) -> Result<(), &'static str> {
    let ok = writable(q) || matches!(q, OwnerQual::Borrow | OwnerQual::BorrowMut);
    if !ok { return Err("qualifier has no spelling on a parameter"); }
    if matches!(q, OwnerQual::Actor) && p.mutable { return Err("an actor by usage is not an actor by declaration"); }
    match &p.ty {
        None => Err("untyped parameter"),
        Some(Type::TypeParam(_)) => Err("implicit type parameter"),
        Some(t) if already_qualified(t) => Err("already written"),
        // `mut` already is the mutable borrow; an `&` after it would read as a shared one.
        // A parameter is a borrow unless it says otherwise (`mut` for the mutable one): writing
        // `&` is a constraint, not a report, and it changes how the argument is passed.
        Some(_) if matches!(q, OwnerQual::Inline) => Err("`inline` is what a by-value parameter is by default"),
        Some(_) if matches!(q, OwnerQual::Borrow | OwnerQual::BorrowMut) => Err("a parameter is a borrow by default"),
        Some(t) => { p.ty = Some(Type::Qualified(Box::new(t.clone()), q.clone())); Ok(()) }
    }
}

/// Calls `f` on every function declaration of `items` (free functions, methods, `ext`, trait
/// defaults, kernels, nested modules), outermost first.
fn for_each_fn(items: &mut [Item], f: &mut dyn FnMut(&mut FnDecl)) {
    for item in items.iter_mut() {
        match item {
            Item::Fn(d) => f(d),
            Item::Struct(s) => for m in &mut s.methods { f(m) },
            Item::Enum(e) => for m in &mut e.methods { f(m) },
            Item::Ext(e) => for m in &mut e.methods { f(m) },
            Item::Trait(t) => for m in &mut t.defaults { f(m) },
            Item::Kernel(k) => for m in &mut k.methods { f(m) },
            Item::Mod(m) => for_each_fn(&mut m.items, f),
            _ => {}
        }
    }
}

/// Whether `let x'q = value` lets the parser resolve the base type from `value`.
fn base_type_inferable(value: Option<&Expr>) -> bool {
    fn scalar(e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Bool(_) | ExprKind::Float(_) => true,
            ExprKind::UnaryOp(_, inner) => scalar(inner),
            _ => false,
        }
    }
    let Some(v) = value else { return false };
    if scalar(v) { return true; }
    matches!(&v.kind, ExprKind::Call(callee, _)
        if matches!(&callee.kind, ExprKind::Var(n) if n.chars().next().is_some_and(char::is_uppercase)))
}
