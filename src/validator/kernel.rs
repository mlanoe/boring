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

//! Kernel validation pass — rejects Boring constructs that are incompatible
//! with Rust-for-Linux (no FPU, no Rc, no panic, …).

use crate::ast::{
    AliasDecl, EnumDecl, Expr, ExprKind, ExtDecl, FnDecl, Item, LitPattern, MatchBody, ModDecl,
    OwnerQual, Pattern, Program, SetDecl, Stmt, StructDecl, TraitDecl, Type, TypeMethod,
};
use super::{DiagLevel, KernelDiagnostic};

// ─── Float math built-ins ────────────────────────────────────────────────────

/// Names of floating-point math functions that are disallowed in kernel context.
const FLOAT_MATH_FNS: &[&str] = &[
    "sqrt", "cbrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "exp2", "ln", "log", "log2", "log10",
    "pow", "hypot", "floor", "ceil", "round", "trunc", "fract", "abs",
    "signum", "copysign",
];

// ─── Validator ───────────────────────────────────────────────────────────────

/// Kernel validation visitor.  Accumulates diagnostics as it walks the AST.
pub struct KernelValidator {
    diags: Vec<KernelDiagnostic>,
}

impl KernelValidator {
    pub fn new() -> Self {
        KernelValidator { diags: Vec::new() }
    }

    // ── Public entry point ───────────────────────────────────────────────────

    pub fn run(mut self, program: &Program) -> Vec<KernelDiagnostic> {
        for item in &program.items {
            self.check_item(item);
        }
        self.diags
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn error(&mut self, line: usize, message: impl Into<String>) {
        self.diags.push(KernelDiagnostic {
            level: DiagLevel::Error,
            line,
            message: message.into(),
        });
    }

    fn warn(&mut self, line: usize, message: impl Into<String>) {
        self.diags.push(KernelDiagnostic {
            level: DiagLevel::Warning,
            line,
            message: message.into(),
        });
    }

    // ── Type walking ─────────────────────────────────────────────────────────

    fn check_type(&mut self, ty: &Type, line: usize) {
        match ty {
            // Rule 1 — float type
            Type::Float => {
                self.error(line, "float is not allowed in kernel context — FPU is disabled");
            }
            // Rule 6 — T'shared in kernel context always maps to Arc<T> (no Rc in kernel)
            Type::Qualified(inner, OwnerQual::Shared) => {
                self.warn(
                    line,
                    "T'shared maps to Arc<T> in kernel context (Rc unavailable — single-thread mode ignored)",
                );
                self.check_type(inner, line);
            }
            // Recurse into compound types
            Type::Optional(inner) | Type::Array(inner) | Type::Dyn(inner) | Type::Impl(inner) => {
                self.check_type(inner, line);
            }
            Type::Dict(k, v) => {
                self.check_type(k, line);
                self.check_type(v, line);
            }
            Type::Set(inner) => self.check_type(inner, line),
            Type::Tuple(elems) => {
                for e in elems {
                    self.check_type(e, line);
                }
            }
            Type::Fn(ret, params, _, _, _) => {
                if let Some(r) = ret {
                    self.check_type(r, line);
                }
                for p in params {
                    self.check_type(p, line);
                }
            }
            Type::Qualified(inner, _) => self.check_type(inner, line),
            Type::Generic(_, args) => {
                for a in args {
                    self.check_type(a, line);
                }
            }
            Type::AssocOf(base, _) => self.check_type(base, line),
            // Leaf types with nothing to recurse into
            _ => {}
        }
    }

    // ── Expression walking ───────────────────────────────────────────────────

    fn check_expr(&mut self, expr: &Expr) {
        let line = expr.line;
        match &expr.kind {
            // Rule 1 — float literal
            ExprKind::Float(_) => {
                self.error(line, "float is not allowed in kernel context — FPU is disabled");
            }

            // Rule 5 — floating-point math functions
            ExprKind::Call(callee, args) => {
                if let ExprKind::Var(name) = &callee.kind {
                    if FLOAT_MATH_FNS.contains(&name.as_str()) {
                        self.error(
                            line,
                            "floating-point math is not allowed in kernel context",
                        );
                    }
                    // Rule 2 — panic(...)
                    if name == "panic" {
                        self.error(
                            line,
                            "panic is not allowed in kernel context — use throws/Result instead",
                        );
                    }
                }
                // Rule 7 — channel without explicit capacity
                // channel<T>() or channel() with no args → no capacity specified
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "channel" && args.is_empty() {
                        self.warn(
                            line,
                            "channel capacity defaults to 2 in kernel context — consider specifying it explicitly",
                        );
                    }
                }
                self.check_expr(callee);
                for arg in args {
                    self.check_expr(&arg.value);
                }
            }

            ExprKind::GenericCall(callee, type_args, args) => {
                if let ExprKind::Var(name) = &callee.kind {
                    if FLOAT_MATH_FNS.contains(&name.as_str()) {
                        self.error(
                            line,
                            "floating-point math is not allowed in kernel context",
                        );
                    }
                    // Rule 7 — channel<T>() with no capacity arg
                    if name == "channel" && args.is_empty() {
                        self.warn(
                            line,
                            "channel capacity defaults to 2 in kernel context — consider specifying it explicitly",
                        );
                    }
                }
                self.check_expr(callee);
                for ty in type_args {
                    self.check_type(ty, line);
                }
                for arg in args {
                    self.check_expr(&arg.value);
                }
            }

            // Method call — check for Math.sqrt(...) style
            ExprKind::MethodCall(receiver, method, args) => {
                if let ExprKind::Var(obj) = &receiver.kind {
                    if obj == "Math" && FLOAT_MATH_FNS.contains(&method.as_str()) {
                        self.error(
                            line,
                            "floating-point math is not allowed in kernel context",
                        );
                    }
                }
                self.check_expr(receiver);
                for arg in args {
                    self.check_expr(&arg.value);
                }
            }

            // Cast — check target type
            ExprKind::Cast(inner, ty) => {
                self.check_expr(inner);
                self.check_type(ty, line);
            }

            // Recurse into all other expression kinds
            ExprKind::BinOp(_, lhs, rhs) => {
                self.check_expr(lhs);
                self.check_expr(rhs);
            }
            ExprKind::UnaryOp(_, inner) => self.check_expr(inner),
            ExprKind::Assign(lhs, rhs) | ExprKind::QuestionAssign(lhs, rhs) => {
                self.check_expr(lhs);
                self.check_expr(rhs);
            }
            ExprKind::Field(obj, _) | ExprKind::OptionalField(obj, _) => self.check_expr(obj),
            ExprKind::Index(obj, idx) => {
                self.check_expr(obj);
                self.check_expr(idx);
            }
            ExprKind::OptionalMethodCall(receiver, _, args) => {
                self.check_expr(receiver);
                for arg in args {
                    self.check_expr(&arg.value);
                }
            }
            ExprKind::Pipe(lhs, _, args) => {
                self.check_expr(lhs);
                for arg in args {
                    self.check_expr(&arg.value);
                }
            }
            ExprKind::TryElse(e, def) => {
                self.check_expr(e);
                self.check_expr(def);
            }
            ExprKind::TryElseBlock(body, else_body) => {
                for s in body {
                    self.check_stmt(s);
                }
                for s in else_body {
                    self.check_stmt(s);
                }
            }
            ExprKind::Array(items) | ExprKind::Set(items) | ExprKind::Tuple(items) => {
                for i in items {
                    self.check_expr(i);
                }
            }
            ExprKind::ArrayFill { value, count } => {
                self.check_expr(value); self.check_expr(count);
            }
            ExprKind::ArrayAlloc { count } => { self.check_expr(count); }
            ExprKind::ArrayComp { expr, count, .. } => {
                self.check_expr(expr); self.check_expr(count);
            }
            ExprKind::ArrayCompIter { expr, iter, .. } => {
                self.check_expr(expr); self.check_expr(iter);
            }
            ExprKind::Dict(pairs) => {
                for (k, v) in pairs {
                    self.check_expr(k);
                    self.check_expr(v);
                }
            }
            ExprKind::Range { start, end, .. } => {
                self.check_expr(start);
                self.check_expr(end);
            }
            ExprKind::Else(a, b) => {
                self.check_expr(a);
                self.check_expr(b);
            }
            ExprKind::Closure(params, ret_ty, body, _, _) => {
                for p in params {
                    if let Some(ty) = &p.ty {
                        self.check_type(ty, p.line);
                    }
                    if let Some(def) = &p.default {
                        self.check_expr(def);
                    }
                }
                if let Some(ty) = ret_ty {
                    self.check_type(ty, line);
                }
                match body {
                    crate::ast::ClosureBody::Expr(e) => self.check_expr(e),
                    crate::ast::ClosureBody::Block(stmts) => {
                        for s in stmts {
                            self.check_stmt(s);
                        }
                    }
                }
            }
            ExprKind::If(if_stmt) => {
                for (cond, body) in &if_stmt.branches {
                    self.check_expr(cond);
                    for s in body {
                        self.check_stmt(s);
                    }
                }
                if let Some(else_body) = &if_stmt.else_body {
                    for s in else_body {
                        self.check_stmt(s);
                    }
                }
            }
            ExprKind::Match(match_stmt) => {
                self.check_expr(&match_stmt.subject);
                for arm in &match_stmt.arms {
                    self.check_patterns(&arm.patterns, line);
                    if let Some(guard) = &arm.guard {
                        self.check_expr(guard);
                    }
                    match &arm.body {
                        MatchBody::Expr(e) => self.check_expr(e),
                        MatchBody::Block(stmts) => {
                            for s in stmts {
                                self.check_stmt(s);
                            }
                        }
                    }
                }
            }
            ExprKind::Block(stmts) | ExprKind::Do(stmts) => {
                for s in stmts {
                    self.check_stmt(s);
                }
            }
            ExprKind::Loop(loop_stmt) => {
                for s in &loop_stmt.body {
                    self.check_stmt(s);
                }
            }
            ExprKind::Task(inner) | ExprKind::TaskWithTimeout(inner, _) => {
                self.check_expr(inner);
                if let ExprKind::TaskWithTimeout(_, timeout) = &expr.kind {
                    self.check_expr(timeout);
                }
            }
            ExprKind::JoinAll(exprs) => {
                for e in exprs {
                    self.check_expr(e);
                }
            }
            ExprKind::MacroCall { args, .. } => {
                for a in args {
                    self.check_expr(a);
                }
            }
            ExprKind::StringInterp(segments) => {
                for seg in segments {
                    if let crate::ast::StringSegment::Expr(e)
                    | crate::ast::StringSegment::FormattedExpr(e, _) = seg
                    {
                        self.check_expr(e);
                    }
                }
            }
            ExprKind::New { arena, ctor } => {
                if let Some(a) = arena { self.check_expr(a); }
                self.check_expr(ctor);
            }

            ExprKind::KernelLaunch { config, kernel } => {
                if let Some(e) = &config.block { self.check_expr(e); }
                if let Some(e) = &config.grid  { self.check_expr(e); }
                if let Some(e) = &config.after { self.check_expr(e); }
                self.check_expr(kernel);
            }

            ExprKind::SliceRange { start, end, .. } => {
                if let Some(s) = start { self.check_expr(s); }
                if let Some(e) = end   { self.check_expr(e); }
            }

            // Leaf kinds — nothing to recurse into
            ExprKind::Int(_)
            | ExprKind::Str(_)
            | ExprKind::Bool(_)
            | ExprKind::Nil
            | ExprKind::Void
            | ExprKind::Var(_)
            | ExprKind::DotIdent(_) => {}
        }
    }

    // ── Pattern walking ──────────────────────────────────────────────────────

    fn check_patterns(&mut self, patterns: &[Pattern], line: usize) {
        for pat in patterns {
            self.check_pattern(pat, line);
        }
    }

    fn check_pattern(&mut self, pat: &Pattern, line: usize) {
        match pat {
            // Rule 1 — float literal in pattern
            Pattern::Lit(LitPattern::Float(_)) => {
                self.error(line, "float is not allowed in kernel context — FPU is disabled");
            }
            Pattern::Variant(_, inner) | Pattern::Tuple(inner) => {
                self.check_patterns(inner, line);
            }
            Pattern::Some(inner) => self.check_pattern(inner, line),
            _ => {}
        }
    }

    // ── Statement walking ────────────────────────────────────────────────────

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(let_stmt) => {
                if let Some(ty) = &let_stmt.ty {
                    self.check_type(ty, let_stmt.line);
                }
                if let Some(val) = &let_stmt.value {
                    self.check_expr(val);
                }
            }
            Stmt::LetDestructure(d) => {
                for binding in &d.bindings {
                    if let Some(ty) = &binding.ty {
                        self.check_type(ty, d.line);
                    }
                }
                self.check_expr(&d.value);
            }
            Stmt::Return(r) => {
                if let Some(v) = &r.value {
                    self.check_expr(v);
                }
            }
            Stmt::Break(_, val) => {
                if let Some(v) = val {
                    self.check_expr(v);
                }
            }
            Stmt::Continue(_) => {}
            Stmt::Throw(t) => {
                if let Some(v) = &t.value {
                    self.check_expr(v);
                }
            }
            Stmt::If(if_stmt) => {
                for (cond, body) in &if_stmt.branches {
                    self.check_expr(cond);
                    for s in body {
                        self.check_stmt(s);
                    }
                }
                if let Some(else_body) = &if_stmt.else_body {
                    for s in else_body {
                        self.check_stmt(s);
                    }
                }
            }
            Stmt::IfLet(if_let) => {
                for clause in &if_let.clauses {
                    match clause {
                        crate::ast::CondClause::Expr(e) => self.check_expr(e),
                        crate::ast::CondClause::Let(_, e) => self.check_expr(e),
                        crate::ast::CondClause::LetPat(pat, e) => {
                            self.check_pattern(pat, if_let.line);
                            self.check_expr(e);
                        }
                    }
                }
                for s in &if_let.then_body {
                    self.check_stmt(s);
                }
                for branch in &if_let.elif_branches {
                    for clause in &branch.clauses {
                        match clause {
                            crate::ast::CondClause::Expr(e) => self.check_expr(e),
                            crate::ast::CondClause::Let(_, e) => self.check_expr(e),
                            crate::ast::CondClause::LetPat(pat, e) => {
                                self.check_pattern(pat, if_let.line);
                                self.check_expr(e);
                            }
                        }
                    }
                    for s in &branch.body {
                        self.check_stmt(s);
                    }
                }
                if let Some(else_body) = &if_let.else_body {
                    for s in else_body {
                        self.check_stmt(s);
                    }
                }
            }
            Stmt::Match(m) => {
                self.check_expr(&m.subject);
                for arm in &m.arms {
                    self.check_patterns(&arm.patterns, m.line);
                    if let Some(guard) = &arm.guard {
                        self.check_expr(guard);
                    }
                    match &arm.body {
                        MatchBody::Expr(e) => self.check_expr(e),
                        MatchBody::Block(stmts) => {
                            for s in stmts {
                                self.check_stmt(s);
                            }
                        }
                    }
                }
            }
            Stmt::While(w) => {
                self.check_expr(&w.condition);
                for s in &w.body {
                    self.check_stmt(s);
                }
            }
            Stmt::WhileLet(w) => {
                if let Some(pat) = &w.pattern {
                    self.check_pattern(pat, w.line);
                }
                self.check_expr(&w.value);
                for s in &w.body {
                    self.check_stmt(s);
                }
            }
            Stmt::DoWhile(d) => {
                for s in &d.body {
                    self.check_stmt(s);
                }
                self.check_expr(&d.condition);
            }
            Stmt::Loop(l) => {
                for s in &l.body {
                    self.check_stmt(s);
                }
            }
            Stmt::Wait(e, _) => self.check_expr(e),
            Stmt::For(f) => {
                self.check_expr(&f.iterable);
                for s in &f.body {
                    self.check_stmt(s);
                }
            }
            Stmt::Guard(g) => {
                match &g.cond {
                    crate::ast::GuardCond::Expr(e) => self.check_expr(e),
                    crate::ast::GuardCond::Clauses(clauses) => {
                        for clause in clauses {
                            match clause {
                                crate::ast::CondClause::Expr(e) => self.check_expr(e),
                                crate::ast::CondClause::Let(_, e) => self.check_expr(e),
                                crate::ast::CondClause::LetPat(pat, e) => {
                                    self.check_pattern(pat, g.line);
                                    self.check_expr(e);
                                }
                            }
                        }
                    }
                }
                for s in &g.else_body {
                    self.check_stmt(s);
                }
            }
            Stmt::Try(try_stmt) => {
                for s in &try_stmt.body {
                    self.check_stmt(s);
                }
                for clause in &try_stmt.catch_clauses {
                    // Rule 4 — catch clause warning (kernel error type not yet defined)
                    for ty_name in &clause.types {
                        self.warn(
                            clause.line,
                            format!(
                                "catch {ty_name}: kernel error stdlib is not yet defined — \
                                 ensure {ty_name} is declared as `type {ty_name} as kernel.error.Error(...)`"
                            ),
                        );
                    }
                    for s in &clause.body {
                        self.check_stmt(s);
                    }
                }
            }
            Stmt::Defer(stmts) => {
                for s in stmts {
                    self.check_stmt(s);
                }
            }
            Stmt::Expr(e) => self.check_expr(e),
            Stmt::Fn(fn_decl) => self.check_fn(fn_decl),
            Stmt::Struct(s) => self.check_struct(s),
            Stmt::Enum(e) => self.check_enum(e),
            Stmt::Mod(m) => self.check_mod(m),
            Stmt::Alias(a) => self.check_alias(a),
            Stmt::Yield(e, _) => self.check_expr(e),
            Stmt::Comment(_) => {}
            Stmt::KernelBlock(s) => { for stmt in &s.body { self.check_stmt(stmt); } }
        }
    }

    // ── Function / method checking ───────────────────────────────────────────

    fn check_fn(&mut self, fn_decl: &FnDecl) {
        let line = fn_decl.line;

        // Check return type
        if let Some(ty) = &fn_decl.return_ty {
            self.check_type(ty, line);
        }

        // Check throws type
        if let Some(ty) = &fn_decl.throws_ty {
            self.check_type(ty, line);
        }

        // Rule 8 — stream def without explicit capacity (stream<N> not specified)
        if fn_decl.stream && fn_decl.stream_capacity.is_none() {
            self.warn(
                line,
                "stream capacity defaults to 2 in kernel context — consider specifying it explicitly with stream<N>",
            );
        }

        // Rule 3 — task def method on self with wrong qualifier
        if fn_decl.task && fn_decl.mutating {
            // A method on self has the first param named "self"
            if let Some(self_param) = fn_decl.params.first() {
                if self_param.name == "self" {
                    if let Some(ty) = &self_param.ty {
                        if !is_task_actor_or_guard(ty) {
                            self.error(
                                line,
                                "task method on self requires 'task, 'actor, or 'guard qualifier",
                            );
                        }
                    }
                    // If no type annotation on self, we cannot infer — emit warning
                    else {
                        // No type annotation — skip, the transpiler will handle it
                    }
                }
            }
        }

        // Check parameter types and defaults
        for param in &fn_decl.params {
            if let Some(ty) = &param.ty {
                self.check_type(ty, param.line);
            }
            if let Some(def) = &param.default {
                self.check_expr(def);
            }
        }

        // Check body
        for stmt in &fn_decl.body {
            self.check_stmt(stmt);
        }
    }

    fn check_type_method(&mut self, tm: &TypeMethod) {
        let line = tm.line;
        if let Some(ty) = &tm.return_ty {
            self.check_type(ty, line);
        }
        for param in &tm.params {
            if let Some(ty) = &param.ty {
                self.check_type(ty, param.line);
            }
            if let Some(def) = &param.default {
                self.check_expr(def);
            }
        }
        for stmt in &tm.body {
            self.check_stmt(stmt);
        }
    }

    fn check_set_decl(&mut self, sd: &SetDecl) {
        self.check_type(&sd.param_ty, sd.line);
        for s in &sd.body {
            self.check_stmt(s);
        }
    }

    // ── Top-level item checking ───────────────────────────────────────────────

    fn check_struct(&mut self, s: &StructDecl) {
        let line = s.line;
        for field in &s.fields {
            self.check_type(&field.ty, field.line);
            if let Some(def) = &field.default {
                self.check_expr(def);
            }
        }
        for init in &s.inits {
            for p in &init.params {
                if let Some(ty) = &p.ty {
                    self.check_type(ty, p.line);
                }
                if let Some(def) = &p.default {
                    self.check_expr(def);
                }
            }
            for s in &init.body {
                self.check_stmt(s);
            }
        }
        for method in &s.methods {
            self.check_fn(method);
        }
        for tm in &s.type_methods {
            self.check_type_method(tm);
        }
        for tv in &s.type_vars {
            if let Some(ty) = &tv.ty {
                self.check_type(ty, line);
            }
            self.check_expr(&tv.default);
        }
        for sd in &s.setters {
            self.check_set_decl(sd);
        }
        for conv in &s.conversions {
            if let Some(ty) = conv.ty.inner_type() {
                self.check_type(ty, conv.line);
            } else {
                self.check_type(&conv.ty, conv.line);
            }
            for s in &conv.body {
                self.check_stmt(s);
            }
        }
        for assoc in &s.assoc_type_defs {
            self.check_type(&assoc.ty, line);
        }
    }

    fn check_enum(&mut self, e: &EnumDecl) {
        let line = e.line;
        for variant in &e.variants {
            for field in &variant.fields {
                self.check_type(&field.ty, line);
            }
        }
        for method in &e.methods {
            self.check_fn(method);
        }
        for sd in &e.setters {
            self.check_set_decl(sd);
        }
        for conv in &e.conversions {
            self.check_type(&conv.ty, conv.line);
            for s in &conv.body {
                self.check_stmt(s);
            }
        }
    }

    fn check_trait(&mut self, t: &TraitDecl) {
        let line = t.line;
        for sig in &t.signatures {
            if let Some(ty) = &sig.return_ty {
                self.check_type(ty, line);
            }
            for p in &sig.params {
                if let Some(ty) = &p.ty {
                    self.check_type(ty, p.line);
                }
            }
        }
        for def in &t.defaults {
            self.check_fn(def);
        }
        for assoc in &t.assoc_types {
            if let Some(c) = &assoc.constraint {
                self.check_type(c, line);
            }
        }
    }

    fn check_ext(&mut self, ext: &ExtDecl) {
        let line = ext.line;
        for ty in &ext.type_args {
            self.check_type(ty, line);
        }
        for method in &ext.methods {
            self.check_fn(method);
        }
        for sd in &ext.setters {
            self.check_set_decl(sd);
        }
        for conv in &ext.conversions {
            self.check_type(&conv.ty, conv.line);
            for s in &conv.body {
                self.check_stmt(s);
            }
        }
        for assoc in &ext.assoc_type_defs {
            self.check_type(&assoc.ty, line);
        }
    }

    fn check_mod(&mut self, m: &ModDecl) {
        for item in &m.items {
            self.check_item(item);
        }
    }

    fn check_alias(&mut self, a: &AliasDecl) {
        self.check_type(&a.ty, a.line);
    }

    fn check_item(&mut self, item: &Item) {
        match item {
            Item::Fn(f) => self.check_fn(f),
            Item::Struct(s) => self.check_struct(s),
            Item::Enum(e) => self.check_enum(e),
            Item::Trait(t) => self.check_trait(t),
            Item::Ext(e) => self.check_ext(e),
            Item::Mod(m) => self.check_mod(m),
            Item::Let(let_stmt) => {
                if let Some(ty) = &let_stmt.ty {
                    self.check_type(ty, let_stmt.line);
                }
                if let Some(val) = &let_stmt.value {
                    self.check_expr(val);
                }
            }
            Item::Alias(a) => self.check_alias(a),
            Item::Stmt(s) => self.check_stmt(s),
            Item::Use(_) => {}
            Item::Kernel(k) => self.check_kernel(k),
        }
    }

    /// GPU kernel struct validation.
    ///
    /// Rule (Item 8): `'shared` and `'local` fields are block-/thread-local device
    /// memory and cannot be written from a host-side `init` constructor.  Any
    /// assignment in an `init` body whose LHS names such a field is an error, no
    /// matter how deeply it is nested inside `if`/`for`/`match`/etc.
    fn check_kernel(&mut self, k: &crate::ast::KernelDecl) {
        for field in &k.fields {
            self.check_type(&field.ty, field.line);
        }
        for init in &k.inits {
            for param in &init.params {
                if let Some(ty) = &param.ty {
                    self.check_type(ty, param.line);
                }
                if let Some(def) = &param.default {
                    self.check_expr(def);
                }
            }
            for stmt in &init.body {
                self.check_kernel_init_stmt(stmt, k);
                self.check_stmt(stmt);
            }
        }
        for method in &k.methods {
            self.check_fn(method);
        }
    }

    /// Recursively walk an `init` body statement (descending into `if`/`for`/`while`/
    /// `match`/`try`/etc. bodies) looking for assignments to `'shared`/`'local` fields.
    fn check_kernel_init_stmt(&mut self, stmt: &Stmt, k: &crate::ast::KernelDecl) {
        use crate::ast::GpuQual;
        let check_assign_target = |this: &mut Self, lhs: &Expr, line: usize| {
            // LHS may be `field` or `field[idx]` or `self.field`.
            let target = match &lhs.kind {
                ExprKind::Var(n) => Some(n.clone()),
                ExprKind::Index(base, _) => {
                    if let ExprKind::Var(n) = &base.kind { Some(n.clone()) } else { None }
                }
                ExprKind::Field(_, f) => Some(f.clone()),
                _ => None,
            };
            if let Some(name) = target {
                if let Some(field) = k.fields.iter().find(|f| f.name == name) {
                    if matches!(field.qual, GpuQual::Sync | GpuQual::Local) {
                        let kind = if matches!(field.qual, GpuQual::Sync) { "'shared" } else { "'local" };
                        this.error(
                            line,
                            format!(
                                "cannot assign to {kind} field '{name}' in an init constructor — \
                                 {kind} memory is device-local and not accessible from the host"
                            ),
                        );
                    }
                }
            }
        };
        match stmt {
            Stmt::Expr(e) => {
                if let ExprKind::Assign(lhs, _) = &e.kind {
                    check_assign_target(self, lhs, e.line);
                }
            }
            Stmt::If(if_stmt) => {
                for (_, body) in &if_stmt.branches {
                    for s in body { self.check_kernel_init_stmt(s, k); }
                }
                if let Some(else_body) = &if_stmt.else_body {
                    for s in else_body { self.check_kernel_init_stmt(s, k); }
                }
            }
            Stmt::IfLet(if_let) => {
                for s in &if_let.then_body { self.check_kernel_init_stmt(s, k); }
                for branch in &if_let.elif_branches {
                    for s in &branch.body { self.check_kernel_init_stmt(s, k); }
                }
                if let Some(else_body) = &if_let.else_body {
                    for s in else_body { self.check_kernel_init_stmt(s, k); }
                }
            }
            Stmt::Match(m) => {
                for arm in &m.arms {
                    if let MatchBody::Block(stmts) = &arm.body {
                        for s in stmts { self.check_kernel_init_stmt(s, k); }
                    }
                }
            }
            Stmt::While(w) => { for s in &w.body { self.check_kernel_init_stmt(s, k); } }
            Stmt::WhileLet(w) => { for s in &w.body { self.check_kernel_init_stmt(s, k); } }
            Stmt::DoWhile(d) => { for s in &d.body { self.check_kernel_init_stmt(s, k); } }
            Stmt::Loop(l) => { for s in &l.body { self.check_kernel_init_stmt(s, k); } }
            Stmt::For(f) => { for s in &f.body { self.check_kernel_init_stmt(s, k); } }
            Stmt::Guard(g) => { for s in &g.else_body { self.check_kernel_init_stmt(s, k); } }
            Stmt::Try(try_stmt) => {
                for s in &try_stmt.body { self.check_kernel_init_stmt(s, k); }
                for clause in &try_stmt.catch_clauses {
                    for s in &clause.body { self.check_kernel_init_stmt(s, k); }
                }
            }
            Stmt::Defer(stmts) => { for s in stmts { self.check_kernel_init_stmt(s, k); } }
            _ => {}
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Returns `true` if the type carries a `'task`, `'actor`, or `'guard` qualifier
/// (possibly nested inside borrows or optional wrappers).
fn is_task_actor_or_guard(ty: &Type) -> bool {
    match ty {
        Type::Qualified(_, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard) => true,
        Type::Qualified(inner, _) => is_task_actor_or_guard(inner),
        Type::Optional(inner) => is_task_actor_or_guard(inner),
        _ => false,
    }
}

// ─── Extension trait for Type ─────────────────────────────────────────────────

/// Small helper to extract the inner type of a conversion target, avoiding
/// double-reporting on the same type in conversion blocks.
trait TypeInner {
    fn inner_type(&self) -> Option<&Type>;
}

impl TypeInner for Type {
    fn inner_type(&self) -> Option<&Type> {
        match self {
            Type::Qualified(inner, _) => Some(inner),
            _ => None,
        }
    }
}
