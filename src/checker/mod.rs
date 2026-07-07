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

// Semantic checker — runs after parsing, before interpretation or transpilation.
//
// Current checks:
//   1. Immutability: assignment to a `let` or `lazy` binding.
//   2. Qualifier constraint: `mut 'shared` is always an error.
//   3. `lazy` misuse: `lazy` binding assigned via `=` after declaration
//      (the correct operator is `?=`).

use std::collections::HashMap;
use crate::ast::*;

// ─── Public interface ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CheckError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct CheckWarning {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

pub struct CheckResult {
    pub errors:   Vec<CheckError>,
    pub warnings: Vec<CheckWarning>,
}

pub fn check(program: &Program) -> CheckResult {
    let mut checker = Checker::new();
    checker.check_program(program);
    CheckResult { errors: checker.errors, warnings: checker.warnings }
}

// ─── Internal ─────────────────────────────────────────────────────────────────

/// One variable entry in the scope stack.
#[derive(Clone)]
struct Binding {
    kind: BindingKind,
    #[allow(dead_code)]
    line: usize,
    #[allow(dead_code)]
    col:  usize,
}

struct Checker {
    errors:   Vec<CheckError>,
    warnings: Vec<CheckWarning>,
    /// Stack of scopes; each scope maps a name to its binding info.
    scopes:   Vec<HashMap<String, Binding>>,
}

impl Checker {
    fn new() -> Self {
        Checker { errors: Vec::new(), warnings: Vec::new(), scopes: vec![HashMap::new()] }
    }

    // ── Scope helpers ─────────────────────────────────────────────────────────

    fn push_scope(&mut self) { self.scopes.push(HashMap::new()); }

    fn pop_scope(&mut self) { self.scopes.pop(); }

    fn define(&mut self, name: &str, kind: BindingKind, line: usize, col: usize) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), Binding { kind, line, col });
        }
    }

    fn lookup(&self, name: &str) -> Option<&Binding> {
        for scope in self.scopes.iter().rev() {
            if let Some(b) = scope.get(name) { return Some(b); }
        }
        None
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    fn error(&mut self, msg: impl Into<String>, line: usize, col: usize) {
        self.errors.push(CheckError { message: msg.into(), line, col });
    }

    #[allow(dead_code)]
    fn warning(&mut self, msg: impl Into<String>, line: usize, col: usize) {
        self.warnings.push(CheckWarning { message: msg.into(), line, col });
    }

    // ── Qualifier constraint: `mut 'shared` ───────────────────────────────────

    fn check_qualifier_constraint(&mut self, binding: &BindingKind, ty: &Option<Type>, line: usize, col: usize) {
        if !matches!(binding, BindingKind::Mut) { return; }
        let Some(ty) = ty else { return };
        if self.type_has_shared(ty) {
            self.error(
                "cannot combine `mut` with `'shared`: shared references are immutable by design; use `'actor` for interior mutability",
                line, col,
            );
        }
    }

    fn type_has_shared(&self, ty: &Type) -> bool {
        match ty {
            Type::Qualified(_, OwnerQual::Shared) => true,
            Type::Qualified(inner, _) => self.type_has_shared(inner),
            Type::Optional(inner) | Type::Array(inner) | Type::Dyn(inner) | Type::Impl(inner) => {
                self.type_has_shared(inner)
            }
            _ => false,
        }
    }

    // ── Top-level ─────────────────────────────────────────────────────────────

    fn check_program(&mut self, program: &Program) {
        for item in &program.items { self.check_item(item); }
    }

    fn check_item(&mut self, item: &Item) {
        match item {
            Item::Let(s)    => self.check_let_stmt(s),
            Item::Fn(f)     => self.check_fn(f),
            Item::Struct(s) => self.check_struct(s),
            Item::Enum(e)   => self.check_enum(e),
            Item::Ext(e)    => self.check_ext(e),
            Item::Mod(m)    => { for i in &m.items { self.check_item(i); } }
            Item::Stmt(s)   => self.check_stmt(s),
            Item::Use(_) | Item::Alias(_) | Item::Trait(_) | Item::Kernel(_) => {}
        }
    }

    // ── Struct / enum / ext ───────────────────────────────────────────────────

    fn check_struct(&mut self, s: &StructDecl) {
        for m in &s.methods { self.check_fn(m); }
        for m in &s.type_methods {
            self.push_scope();
            for p in &m.params { self.define(&p.name, param_binding(p), p.line, p.col); }
            for stmt in &m.body { self.check_stmt(stmt); }
            self.pop_scope();
        }
    }

    fn check_enum(&mut self, e: &EnumDecl) {
        for m in &e.methods { self.check_fn(m); }
    }

    fn check_ext(&mut self, e: &ExtDecl) {
        for m in &e.methods { self.check_fn(m); }
    }

    // ── Functions ─────────────────────────────────────────────────────────────

    fn check_fn(&mut self, f: &FnDecl) {
        self.push_scope();
        for p in &f.params {
            if p.mutable {
                self.check_qualifier_constraint(&BindingKind::Mut, &p.ty, p.line, p.col);
            }
            self.define(&p.name, param_binding(p), p.line, p.col);
        }
        for stmt in &f.body { self.check_stmt(stmt); }
        self.pop_scope();
    }

    // ── Statements ────────────────────────────────────────────────────────────

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(s) => self.check_let_stmt(s),
            Stmt::LetDestructure(s) => {
                self.check_expr(&s.value);
                for b in &s.bindings {
                    if b.name != "_" {
                        self.define(&b.name, s.binding.clone(), s.line, s.col);
                    }
                }
            }
            Stmt::Expr(e)      => self.check_expr(e),
            Stmt::Return(r)    => { if let Some(v) = &r.value { self.check_expr(v); } }
            Stmt::Throw(t)     => { if let Some(v) = &t.value { self.check_expr(v); } }
            Stmt::If(s)        => self.check_if(s),
            Stmt::IfLet(s)     => self.check_if_let(s),
            Stmt::While(s)     => {
                self.check_expr(&s.condition);
                self.check_block(&s.body);
            }
            Stmt::WhileLet(s)  => {
                self.check_expr(&s.value);
                self.push_scope();
                self.define(&s.name, BindingKind::Let, s.line, s.col);
                self.check_block_in_current_scope(&s.body);
                self.pop_scope();
            }
            Stmt::DoWhile(s)   => {
                self.check_block(&s.body);
                self.check_expr(&s.condition);
            }
            Stmt::Loop(s)      => self.check_block(&s.body),
            Stmt::For(s)       => {
                self.check_expr(&s.iterable);
                self.push_scope();
                for v in &s.vars { self.define(v, BindingKind::Let, s.line, s.col); }
                self.check_block_in_current_scope(&s.body);
                self.pop_scope();
            }
            Stmt::Match(s)     => self.check_match_stmt(s),
            Stmt::Guard(s)     => {
                match &s.cond {
                    GuardCond::Expr(e)      => self.check_expr(e),
                    GuardCond::Clauses(cs)  => self.check_cond_clauses(cs),
                }
                self.check_block(&s.else_body);
            }
            Stmt::Try(s)       => {
                self.check_block(&s.body);
                for clause in &s.catch_clauses { self.check_block(&clause.body); }
            }
            Stmt::Defer(body)  => self.check_block(body),
            Stmt::Yield(e, _)  => self.check_expr(e),
            Stmt::Wait(e, _)   => self.check_expr(e),
            Stmt::Break(_, v)  => { if let Some(e) = v { self.check_expr(e); } }
            Stmt::Fn(f)        => self.check_fn(f),
            Stmt::Struct(s)    => self.check_struct(s),
            Stmt::Enum(e)      => self.check_enum(e),
            Stmt::Mod(m)       => { for i in &m.items { self.check_item(i); } }
            Stmt::Continue(_) | Stmt::Alias(_) | Stmt::Comment(_) => {}
            Stmt::KernelBlock(s) => { for stmt in &s.body { self.check_stmt(stmt); } }
        }
    }

    fn check_let_stmt(&mut self, s: &LetStmt) {
        self.check_qualifier_constraint(&s.binding, &s.ty, s.line, s.col);
        if let Some(v) = &s.value { self.check_expr(v); }
        self.define(&s.name, s.binding.clone(), s.line, s.col);
    }

    fn check_if(&mut self, s: &IfStmt) {
        for (cond, body) in &s.branches {
            self.check_expr(cond);
            self.check_block(body);
        }
        if let Some(body) = &s.else_body { self.check_block(body); }
    }

    fn check_if_let(&mut self, s: &IfLetStmt) {
        self.push_scope();
        self.check_cond_clauses(&s.clauses);
        self.check_block_in_current_scope(&s.then_body);
        self.pop_scope();
        for branch in &s.elif_branches {
            self.push_scope();
            self.check_cond_clauses(&branch.clauses);
            self.check_block_in_current_scope(&branch.body);
            self.pop_scope();
        }
        if let Some(body) = &s.else_body { self.check_block(body); }
    }

    fn check_cond_clauses(&mut self, clauses: &[CondClause]) {
        for clause in clauses {
            match clause {
                CondClause::Expr(e)       => self.check_expr(e),
                CondClause::Let(name, e)  => {
                    self.check_expr(e);
                    self.define(name, BindingKind::Let, 0, 0);
                }
                CondClause::LetPat(_, e)  => self.check_expr(e),
            }
        }
    }

    fn check_match_stmt(&mut self, s: &MatchStmt) {
        self.check_expr(&s.subject);
        for arm in &s.arms {
            if let Some(g) = &arm.guard { self.check_expr(g); }
            self.push_scope();
            for pat in &arm.patterns {
                bind_in_pattern(pat, arm.line, arm.col, &mut |name, line, col| {
                    self.define(name, BindingKind::Let, line, col);
                });
            }
            match &arm.body {
                MatchBody::Expr(e)      => self.check_expr(e),
                MatchBody::Block(stmts) => self.check_block_in_current_scope(stmts),
            }
            self.pop_scope();
        }
    }

    // ── Block helpers ─────────────────────────────────────────────────────────

    fn check_block(&mut self, stmts: &[Stmt]) {
        self.push_scope();
        for stmt in stmts { self.check_stmt(stmt); }
        self.pop_scope();
    }

    fn check_block_in_current_scope(&mut self, stmts: &[Stmt]) {
        for stmt in stmts { self.check_stmt(stmt); }
    }

    // ── Expressions ───────────────────────────────────────────────────────────

    fn check_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            // ── Assignment — the core immutability check ──────────────────
            ExprKind::Assign(lhs, rhs) => {
                self.check_expr(rhs);
                self.check_assign_target(lhs, expr.line, expr.col);
                // Still recurse into lhs for nested expressions (e.g. index sub-expr).
                match &lhs.kind {
                    ExprKind::Var(_) => {}
                    _ => self.check_expr(lhs),
                }
            }
            ExprKind::QuestionAssign(lhs, rhs) => {
                // `?=` is always legal (lazy initialisation / nil-coalescing).
                self.check_expr(rhs);
                self.check_expr(lhs);
            }

            // ── Recurse into sub-expressions ──────────────────────────────
            ExprKind::BinOp(_, l, r) => { self.check_expr(l); self.check_expr(r); }
            ExprKind::UnaryOp(_, e)  => self.check_expr(e),
            ExprKind::Field(e, _)    => self.check_expr(e),
            ExprKind::OptionalField(e, _) => self.check_expr(e),
            ExprKind::Index(obj, idx) => { self.check_expr(obj); self.check_expr(idx); }
            ExprKind::Call(callee, args) => {
                self.check_expr(callee);
                for a in args { self.check_expr(&a.value); }
            }
            ExprKind::MethodCall(recv, _, args) | ExprKind::OptionalMethodCall(recv, _, args) => {
                self.check_expr(recv);
                for a in args { self.check_expr(&a.value); }
            }
            ExprKind::GenericCall(callee, _, args) => {
                self.check_expr(callee);
                for a in args { self.check_expr(&a.value); }
            }
            ExprKind::Pipe(lhs, _, args) => {
                self.check_expr(lhs);
                for a in args { self.check_expr(&a.value); }
            }
            ExprKind::New { ctor, arena } => {
                self.check_expr(ctor);
                if let Some(a) = arena { self.check_expr(a); }
            }
            ExprKind::Cast(e, _)  => self.check_expr(e),
            ExprKind::Else(e, d)  => { self.check_expr(e); self.check_expr(d); }
            ExprKind::TryElse(e, d) => { self.check_expr(e); self.check_expr(d); }
            ExprKind::TryElseBlock(body, els) => {
                self.check_block(body);
                self.check_block(els);
            }
            ExprKind::Array(elems) => { for e in elems { self.check_expr(e); } }
            ExprKind::ArrayFill { value, count } => {
                self.check_expr(value); self.check_expr(count);
            }
            ExprKind::ArrayAlloc { count } => { self.check_expr(count); }
            ExprKind::ArrayComp { expr, var, count } => {
                self.check_expr(count);
                self.push_scope();
                self.define(var, BindingKind::Let, 0, 0);
                self.check_expr(expr);
                self.pop_scope();
            }
            ExprKind::ArrayCompIter { expr, var, iter } => {
                self.check_expr(iter);
                self.push_scope();
                self.define(var, BindingKind::Let, 0, 0);
                self.check_expr(expr);
                self.pop_scope();
            }
            ExprKind::Tuple(elems) => { for e in elems { self.check_expr(e); } }
            ExprKind::Dict(pairs)  => {
                for (k, v) in pairs { self.check_expr(k); self.check_expr(v); }
            }
            ExprKind::Set(elems)   => { for e in elems { self.check_expr(e); } }
            ExprKind::Range { start, end, .. } => { self.check_expr(start); self.check_expr(end); }
            ExprKind::StringInterp(segs) => {
                for seg in segs {
                    if let StringSegment::Expr(e) | StringSegment::FormattedExpr(e, _) = seg {
                        self.check_expr(e);
                    }
                }
            }
            ExprKind::If(s)    => self.check_if(s),
            ExprKind::Match(s) => self.check_match_stmt(s),
            ExprKind::Block(stmts) | ExprKind::Do(stmts) => self.check_block(stmts),
            ExprKind::Loop(s)  => self.check_block(&s.body),
            ExprKind::Task(e)  => self.check_expr(e),
            ExprKind::TaskWithTimeout(dur, e) => { self.check_expr(dur); self.check_expr(e); }
            ExprKind::JoinAll(exprs) => { for e in exprs { self.check_expr(e); } }
            ExprKind::KernelLaunch { kernel, config } => {
                self.check_expr(kernel);
                if let Some(b) = &config.block { self.check_expr(b); }
                if let Some(g) = &config.grid  { self.check_expr(g); }
            }
            ExprKind::Closure(params, _, body, _, _) => {
                self.push_scope();
                for p in params { self.define(&p.name, param_binding(p), p.line, p.col); }
                match body {
                    ClosureBody::Expr(e)      => self.check_expr(e),
                    ClosureBody::Block(stmts) => self.check_block_in_current_scope(stmts),
                }
                self.pop_scope();
            }
            ExprKind::MacroCall { args, .. } => {
                for a in args { self.check_expr(a); }
            }

            // Leaves — nothing to recurse into.
            ExprKind::Var(_) | ExprKind::Int(_) | ExprKind::Float(_)
            | ExprKind::Str(_) | ExprKind::Bool(_) | ExprKind::Nil
            | ExprKind::Void | ExprKind::DotIdent(_) => {}
        }
    }

    // ── Immutability check on assignment targets ───────────────────────────────

    fn check_assign_target(&mut self, lhs: &Expr, assign_line: usize, assign_col: usize) {
        if let ExprKind::Var(name) = &lhs.kind {
            // `_` is the discard wildcard — never an error as assignment target.
            if name == "_" { return; }
            if let Some(binding) = self.lookup(name) {
                match binding.kind {
                    BindingKind::Let => {
                        self.error(
                            format!("cannot assign to `{name}`: declared as `let` (immutable)"),
                            assign_line, assign_col,
                        );
                    }
                    BindingKind::Lazy => {
                        self.error(
                            format!("cannot assign to `{name}` with `=`: `lazy` bindings are written with `?=`"),
                            assign_line, assign_col,
                        );
                    }
                    BindingKind::Mut | BindingKind::Var => {}
                }
            }
            // Unknown variable — undefined-var check belongs to the interpreter/transpiler.
        }
        // Field and index targets are not checked here: mutability of those
        // requires type information not yet available at this pass.
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn param_binding(p: &Param) -> BindingKind {
    if p.rebindable { BindingKind::Var }
    else if p.mutable { BindingKind::Mut }
    else { BindingKind::Let }
}

fn bind_in_pattern(pat: &Pattern, line: usize, col: usize, f: &mut impl FnMut(&str, usize, usize)) {
    match pat {
        Pattern::Bind(name)       => f(name, line, col),
        Pattern::Some(inner)      => bind_in_pattern(inner, line, col, f),
        Pattern::Variant(_, sub)  => { for p in sub { bind_in_pattern(p, line, col, f); } }
        Pattern::Tuple(sub)       => { for p in sub { bind_in_pattern(p, line, col, f); } }
        Pattern::Wildcard | Pattern::None | Pattern::Lit(_) => {}
    }
}
