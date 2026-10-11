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

//! Generic mutable AST walker for source-to-source passes. Visits every statement and
//! expression in pre-order (the callback runs on a node *before* its children, so a callback
//! that replaces a node has its replacement's children visited too). Nested `fn` bodies and
//! closures are entered; nested struct/enum/mod/alias declarations are not (a pass that cares
//! about them handles them itself).

use crate::ast::*;

pub struct Walker<'a> {
    pub expr: &'a mut dyn FnMut(&mut Expr),
    pub stmt: &'a mut dyn FnMut(&mut Stmt),
}

impl Walker<'_> {
    pub fn stmts(&mut self, stmts: &mut [Stmt]) {
        for s in stmts { self.stmt(s); }
    }

    fn args(&mut self, args: &mut [Arg]) {
        for a in args { self.expr(&mut a.value); }
    }

    fn clause(&mut self, c: &mut CondClause) {
        match c {
            CondClause::Let(_, e) | CondClause::LetPat(_, e) | CondClause::Expr(e) => self.expr(e),
        }
    }

    pub fn stmt(&mut self, stmt: &mut Stmt) {
        (self.stmt)(stmt);
        match stmt {
            Stmt::Let(l) => { if let Some(v) = &mut l.value { self.expr(v); } }
            Stmt::LetDestructure(l) => self.expr(&mut l.value),
            Stmt::Return(r) => { if let Some(v) = &mut r.value { self.expr(v); } }
            Stmt::Break(_, v) => { if let Some(v) = v { self.expr(v); } }
            Stmt::Throw(t) => { if let Some(v) = &mut t.value { self.expr(v); } }
            Stmt::If(i) => self.if_stmt(i),
            Stmt::IfLet(i) => {
                for c in &mut i.clauses { self.clause(c); }
                self.stmts(&mut i.then_body);
                for b in &mut i.elif_branches {
                    for c in &mut b.clauses { self.clause(c); }
                    self.stmts(&mut b.body);
                }
                if let Some(e) = &mut i.else_body { self.stmts(e); }
            }
            Stmt::Match(m) => self.match_stmt(m),
            Stmt::While(w) => { self.expr(&mut w.condition); self.stmts(&mut w.body); }
            Stmt::WhileLet(w) => { self.expr(&mut w.value); self.stmts(&mut w.body); }
            Stmt::DoWhile(d) => { self.stmts(&mut d.body); self.expr(&mut d.condition); }
            Stmt::Loop(l) => self.stmts(&mut l.body),
            Stmt::Wait(e, _) | Stmt::Yield(e, _) | Stmt::Expr(e) => self.expr(e),
            Stmt::For(f) => {
                self.expr(&mut f.iterable);
                if let Some(k) = &mut f.key { self.expr(k); }
                self.stmts(&mut f.body);
            }
            Stmt::Guard(g) => {
                match &mut g.cond {
                    GuardCond::Expr(e) => self.expr(e),
                    GuardCond::Clauses(cs) => { for c in cs { self.clause(c); } }
                }
                self.stmts(&mut g.else_body);
            }
            Stmt::Try(t) => {
                self.stmts(&mut t.body);
                for c in &mut t.catch_clauses { self.stmts(&mut c.body); }
            }
            Stmt::Defer(b) => self.stmts(b),
            Stmt::Fn(f) => self.stmts(&mut f.body),
            Stmt::KernelBlock(k) => self.stmts(&mut k.body),
            Stmt::With(w) => self.stmts(&mut w.body),
            Stmt::Continue(_) | Stmt::Comment(_) | Stmt::Struct(_) | Stmt::Enum(_)
            | Stmt::Mod(_) | Stmt::Alias(_) => {}
        }
    }

    fn if_stmt(&mut self, i: &mut IfStmt) {
        for (c, b) in &mut i.branches { self.expr(c); self.stmts(b); }
        if let Some(e) = &mut i.else_body { self.stmts(e); }
    }

    fn match_stmt(&mut self, m: &mut MatchStmt) {
        self.expr(&mut m.subject);
        for arm in &mut m.arms {
            if let Some(g) = &mut arm.guard { self.expr(g); }
            match &mut arm.body {
                MatchBody::Expr(e) => self.expr(e),
                MatchBody::Block(b) => self.stmts(b),
            }
        }
    }

    pub fn expr(&mut self, e: &mut Expr) {
        (self.expr)(e);
        match &mut e.kind {
            ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Str(_)
            | ExprKind::Bool(_) | ExprKind::Nil | ExprKind::Void | ExprKind::Var(_)
            | ExprKind::DotIdent(_) => {}
            ExprKind::StringInterp(segs) => {
                for seg in segs {
                    match seg {
                        StringSegment::Lit(_) => {}
                        StringSegment::Expr(x) | StringSegment::FormattedExpr(x, _) => self.expr(x),
                    }
                }
            }
            ExprKind::BinOp(_, l, r) | ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r)
            | ExprKind::TryElse(l, r) | ExprKind::Else(l, r) | ExprKind::TaskWithTimeout(l, r) => {
                self.expr(l);
                self.expr(r);
            }
            ExprKind::UnaryOp(_, x) | ExprKind::Field(x, _) | ExprKind::RelabelCast(x, _)
            | ExprKind::OptionalField(x, _) | ExprKind::Task(x) | ExprKind::Cast(x, _) => self.expr(x),
            ExprKind::Index(a, b) => { self.expr(a); self.expr(b); }
            ExprKind::LabeledIndex(x, args) => { self.expr(x); self.args(args); }
            ExprKind::Call(c, args) | ExprKind::GenericCall(c, _, args) => { self.expr(c); self.args(args); }
            ExprKind::TrailingArrayBlock { callee, args, body, .. } => {
                self.expr(callee);
                self.args(args);
                self.stmts(body);
            }
            ExprKind::MethodCall(o, _, args) | ExprKind::Pipe(o, _, args)
            | ExprKind::OptionalMethodCall(o, _, args) => { self.expr(o); self.args(args); }
            ExprKind::New { arena, ctor } => {
                if let Some(a) = arena { self.expr(a); }
                self.expr(ctor);
            }
            ExprKind::KernelLaunch { config, kernel } => {
                for x in [&mut config.block, &mut config.grid, &mut config.after].into_iter().flatten() {
                    self.expr(x);
                }
                self.expr(kernel);
            }
            ExprKind::TryElseBlock(a, b) => { self.stmts(a); self.stmts(b); }
            ExprKind::Array(items) | ExprKind::Tuple(items) | ExprKind::Set(items)
            | ExprKind::JoinAll(items) | ExprKind::MacroCall { args: items, .. } => {
                for x in items { self.expr(x); }
            }
            ExprKind::ArrayFill { value, count } => { self.expr(value); self.expr(count); }
            ExprKind::ArrayAlloc { count } => self.expr(count),
            ExprKind::ArrayComp { expr, count, .. } => { self.expr(expr); self.expr(count); }
            ExprKind::ArrayCompIter { expr, iter, .. } => { self.expr(expr); self.expr(iter); }
            ExprKind::LabeledArrayComp { expr, clauses } => {
                self.expr(expr);
                for (_, c) in clauses { self.expr(c); }
            }
            ExprKind::Dict(pairs) => { for (k, v) in pairs { self.expr(k); self.expr(v); } }
            ExprKind::Range { start, end, .. } => { self.expr(start); self.expr(end); }
            ExprKind::SliceRange { start, end, .. } => {
                if let Some(s) = start { self.expr(s); }
                if let Some(x) = end { self.expr(x); }
            }
            ExprKind::Closure(params, _, body, _, _) => {
                for p in params { if let Some(d) = &mut p.default { self.expr(d); } }
                match body {
                    ClosureBody::Expr(x) => self.expr(x),
                    ClosureBody::Block(b) => self.stmts(b),
                }
            }
            ExprKind::If(i) => self.if_stmt(i),
            ExprKind::Match(m) => self.match_stmt(m),
            ExprKind::Block(b) | ExprKind::Do(b) => self.stmts(b),
            ExprKind::Loop(l) => self.stmts(&mut l.body),
        }
    }
}
