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

use super::*;
use super::Transpiler;
use super::helpers::*;

impl Transpiler {
    pub(crate) fn emit_if(&mut self, s: &IfStmt, is_last: bool) {
        // When this if/else is the last statement in a value-returning function,
        // use emit_body so branch tails are returned without semicolons.
        // use_value_body: emit branch tails as Rust tail expressions (non-throws).
        // use_throws_tail: in a throws function, emit branch tails as `return Ok(expr)`.
        let use_value_body = is_last && !self.fn_returns_void && (!self.in_throws || self.suppress_ok_wrap);
        let use_throws_tail = is_last && self.in_throws && !self.suppress_ok_wrap
            && !self.fn_returns_void && s.else_body.is_some();
        for (i, (cond, body)) in s.branches.iter().enumerate() {
            let kw = if i == 0 { "if" } else { "} else if" };
            let cond_s = self.emit_expr(cond);
            self.line(&format!("{} {} {{", kw, cond_s));
            self.indent += 1;
            if use_value_body || use_throws_tail {
                self.emit_body(body);
            } else {
                // Use emit_loop_body so nested if-branches never get Ok()/Ok(()) wrapping.
                // Ok-wrapping is only correct at the top-level function body.
                self.emit_loop_body(body);
            }
            self.indent -= 1;
        }
        if let Some(else_body) = &s.else_body {
            self.line("} else {");
            self.indent += 1;
            if use_value_body || use_throws_tail {
                self.emit_body(else_body);
            } else {
                self.emit_loop_body(else_body);
            }
            self.indent -= 1;
        }
        self.line("}");
    }

    pub(crate) fn emit_if_let(&mut self, s: &IfLetStmt, is_last: bool) {
        // Track if-let bindings as known locals; also track actor-typed bindings.
        // Covers the initial clauses as well as every `elif let` branch's clauses.
        for clause in s.clauses.iter().chain(s.elif_branches.iter().flat_map(|b| b.clauses.iter())) {
            match clause {
                CondClause::Let(name, expr) => {
                    self.known_local_vars.insert(name.clone());
                    // If the expression is an optional actor field, track the binding as managed.
                    let is_actor = self.expr_yields_actor(expr);
                    if is_actor {
                        match self.config.threading {
                            crate::transpiler::ThreadingMode::Single => { self.managed_refcell_vars.insert(name.clone()); }
                            crate::transpiler::ThreadingMode::Multi  => { self.managed_mutex_vars.insert(name.clone()); }
                        }
                        // Also track the inner struct type so method return types can be inferred.
                        // e.g. `if let p = self.parent:` where parent: Env'actor? → var_struct_types["p"] = "Env"
                        let struct_ty = match &expr.kind {
                            crate::ast::ExprKind::Field(obj, field_name) => {
                                let sn = match &obj.kind {
                                    crate::ast::ExprKind::Var(v) if v.as_str() == "self" => self.self_type.clone(),
                                    crate::ast::ExprKind::Var(v) => self.var_struct_types.get(v.as_str()).cloned(),
                                    _ => None,
                                };
                                sn.and_then(|sn| self.struct_fields.get(sn.as_str()))
                                    .and_then(|fs| fs.iter().find(|(n, _)| n == field_name))
                                    .and_then(|(_, ty)| match ty {
                                        crate::ast::Type::Optional(inner) => match inner.as_ref() {
                                            crate::ast::Type::Qualified(inner2, _) => match inner2.as_ref() {
                                                crate::ast::Type::Named(n) => Some(n.clone()),
                                                _ => None,
                                            },
                                            crate::ast::Type::Named(n) => Some(n.clone()),
                                            _ => None,
                                        },
                                        crate::ast::Type::Qualified(inner, _) => match inner.as_ref() {
                                            crate::ast::Type::Named(n) => Some(n.clone()),
                                            _ => None,
                                        },
                                        _ => None,
                                    })
                            }
                            _ => None,
                        };
                        if let Some(sty) = struct_ty {
                            self.var_struct_types.insert(name.clone(), sty);
                        }
                    }
                }
                CondClause::LetPat(pat, _) => { Self::collect_pattern_binds(pat, &mut self.known_local_vars); }
                CondClause::Expr(_) => {}
            }
        }
        // When this if-let is the last statement in a value-returning function,
        // use emit_body so the branch tail is returned without a semicolon.
        let use_value_body = is_last && !self.fn_returns_void && (!self.in_throws || self.suppress_ok_wrap);
        // Multi-clause: emit as a chain of let-else or if-let
        let cond_s = self.emit_cond_clauses(&s.clauses);
        self.line(&format!("if {} {{", cond_s));
        self.indent += 1;
        if use_value_body {
            self.emit_body(&s.then_body);
        } else {
            self.emit_loop_body(&s.then_body);
        }
        self.indent -= 1;
        for branch in &s.elif_branches {
            let elif_cond_s = self.emit_cond_clauses(&branch.clauses);
            self.line(&format!("}} else if {} {{", elif_cond_s));
            self.indent += 1;
            if use_value_body {
                self.emit_body(&branch.body);
            } else {
                self.emit_loop_body(&branch.body);
            }
            self.indent -= 1;
        }
        if let Some(else_body) = &s.else_body {
            self.line("} else {");
            self.indent += 1;
            if use_value_body {
                self.emit_body(else_body);
            } else {
                self.emit_loop_body(else_body);
            }
            self.indent -= 1;
        }
        self.line("}");
    }

    pub(crate) fn emit_cond_clauses(&self, clauses: &[CondClause]) -> String {
        clauses.iter().map(|c| match c {
            CondClause::Let(name, expr) => {
                // Narrowing numeric `as` cast scrutinee needs its own checked,
                // Option-producing codegen — the plain
                // `emit_expr` path for a numeric-to-integer cast emits an unconditional
                // infallible `(src as dst)`, which doesn't type-check against the
                // `Some(...)` pattern this function always emits below.
                if let Some(checked) = self.try_emit_checked_int_cast_as_option(expr) {
                    return format!("let Some({}) = {}", name, checked);
                }
                let expr_s = self.emit_expr(expr);
                // Auto-clone: actor fields and general field accesses must be cloned
                // to avoid moving out of the struct.
                let expr_s = if self.expr_yields_actor(expr)
                    || (matches!(&expr.kind, ExprKind::Field(..))
                        && !expr_s.ends_with(".clone()")
                        && !expr_s.starts_with('&')
                        && !expr_s.starts_with("Arc::")
                        && !expr_s.starts_with("Rc::")
                        && !expr_s.starts_with("{ let __g"))
                {
                    format!("{}.clone()", expr_s)
                } else {
                    expr_s
                };
                format!("let Some({}) = {}", name, expr_s)
            }
            CondClause::LetPat(pat, expr) => {
                format!("let {} = {}", self.emit_pattern(pat), self.emit_expr(expr))
            }
            CondClause::Expr(e) => self.emit_expr(e),
        }).collect::<Vec<_>>().join(" && ")
    }

    /// Return the set of variable names from `bound` that are mutated (index-assigned,
    /// directly assigned, or have a `def`/mutating method called through them) anywhere in
    /// `stmts`. Used to decide which pattern bindings need `mut` on the Rust side.
    ///
    /// `bound_structs` maps a bound name to its struct type name (only populated when that
    /// type is known — e.g. a variant field typed `mut Point`, see
    /// `docs/mut-type-modifier.md`) — needed to resolve a method call's `def`/`req` status
    /// via `method_is_req_or_task`. Names bound to a non-struct or unresolved type never
    /// need `mut` for this reason (they can still need it via direct assignment above).
    ///
    /// `subject_is_self` disables ONLY the method-call-triggered detection: matching bare
    /// `self` always matches a reference (`&Self`/`&mut Self`), which puts every bound name
    /// in Rust's implicit `ref`/`ref mut` binding mode — an explicit `mut` modifier there is
    /// a hard compile error ("cannot mutably bind by value within an implicitly-borrowing
    /// pattern"), and it's also unneeded: a field bound that way is already `&mut T` inside
    /// a `&mut self` method (see `emit_top.rs`'s `is_enum_self`), so `p.def_method()` works
    /// on the bare binding with no annotation at all. A non-`self` owned local still needs
    /// the promotion — there the bound field is an owned value moved out of the match.
    fn collect_mutated_bindings(&self, bound: &[String], bound_structs: &std::collections::HashMap<String, String>, subject_is_self: bool, stmts: &[Stmt]) -> std::collections::HashSet<String> {
        fn expr_mutated(t: &Transpiler, bound: &[String], bound_structs: &std::collections::HashMap<String, String>, subject_is_self: bool, e: &Expr, out: &mut std::collections::HashSet<String>) {
            match &e.kind {
                ExprKind::Assign(target, rhs) => {
                    // Direct assign: `name = val` — the bound var itself is mutated.
                    if let ExprKind::Var(v) = &target.kind {
                        if bound.contains(v) { out.insert(v.clone()); }
                    }
                    // Index-assign: `name[idx] = val` — the collection is mutated.
                    if let ExprKind::Index(obj, _) = &target.kind {
                        if let ExprKind::Var(v) = &obj.kind {
                            if bound.contains(v) { out.insert(v.clone()); }
                        }
                    }
                    expr_mutated(t, bound, bound_structs, subject_is_self, rhs, out);
                }
                ExprKind::MethodCall(obj, method, args) => {
                    // `p.def_method()` on a bound name whose struct type's method is
                    // mutating — needs a `mut` Rust binding to call through (except when
                    // matching `self` — see this function's doc).
                    if !subject_is_self {
                        if let ExprKind::Var(v) = &obj.kind {
                            if bound.contains(v) {
                                if let Some(struct_name) = bound_structs.get(v) {
                                    if !t.method_is_req_or_task(struct_name, method) {
                                        out.insert(v.clone());
                                    }
                                }
                            }
                        }
                    }
                    expr_mutated(t, bound, bound_structs, subject_is_self, obj, out);
                    for a in args { expr_mutated(t, bound, bound_structs, subject_is_self, &a.value, out); }
                }
                ExprKind::BinOp(_, l, r) => {
                    expr_mutated(t, bound, bound_structs, subject_is_self, l, out); expr_mutated(t, bound, bound_structs, subject_is_self, r, out);
                }
                ExprKind::Call(callee, args) => {
                    expr_mutated(t, bound, bound_structs, subject_is_self, callee, out);
                    for a in args { expr_mutated(t, bound, bound_structs, subject_is_self, &a.value, out); }
                }
                ExprKind::If(s) => {
                    for (_, b) in &s.branches {
                        for st in b { stmt_mutated(t, bound, bound_structs, subject_is_self, st, out); }
                    }
                    if let Some(eb) = &s.else_body {
                        for st in eb { stmt_mutated(t, bound, bound_structs, subject_is_self, st, out); }
                    }
                }
                _ => {}
            }
        }
        fn stmt_mutated(t: &Transpiler, bound: &[String], bound_structs: &std::collections::HashMap<String, String>, subject_is_self: bool, s: &Stmt, out: &mut std::collections::HashSet<String>) {
            match s {
                Stmt::Expr(e) => expr_mutated(t, bound, bound_structs, subject_is_self, e, out),
                Stmt::Let(l) => { if let Some(v) = &l.value { expr_mutated(t, bound, bound_structs, subject_is_self, v, out); } }
                Stmt::For(f) => {
                    for st in &f.body { stmt_mutated(t, bound, bound_structs, subject_is_self, st, out); }
                }
                Stmt::While(w) => {
                    for st in &w.body { stmt_mutated(t, bound, bound_structs, subject_is_self, st, out); }
                }
                Stmt::Match(m) => {
                    for arm in &m.arms {
                        match &arm.body {
                            MatchBody::Block(stmts) => { for st in stmts { stmt_mutated(t, bound, bound_structs, subject_is_self, st, out); } }
                            MatchBody::Expr(e) => expr_mutated(t, bound, bound_structs, subject_is_self, e, out),
                        }
                    }
                }
                _ => {}
            }
        }
        let mut out = std::collections::HashSet::new();
        for s in stmts { stmt_mutated(self, bound, bound_structs, subject_is_self, s, &mut out); }
        out
    }

    /// Recursively collects every string-typed plain-variable name that is indexed
    /// (`name[idx]`, non-constant `idx`) anywhere in `stmts`. `string[i]` transpiles to
    /// `.chars().nth(i)` (Rust can't O(1)-index a UTF-8 `str` by char position) -- fine
    /// for one-off access, but a sequential scan (`while i < s.length: ... s[i] ...`,
    /// the idiomatic way to hand-parse a string in Boring) turns into O(n^2), since
    /// every access re-walks the string from byte 0. Callers use this set to decide
    /// which string bindings are worth materializing as a `Vec<char>` once (see
    /// `emit_let`'s `__strchars_*` shadow) so indexed access becomes O(1).
    ///
    /// "String-typed" is intentionally narrow -- only names whose type is knowable
    /// without full inference (an explicit `string`/`str` parameter or `let`
    /// annotation, or a `let` with a string-literal initializer) -- because an
    /// `Index` site alone can't distinguish `s[i]` (string) from `arr[i]` (array/Vec):
    /// both parse to the same `ExprKind::Index(Var, _)` shape, and array elements are
    /// never `char`, so wrongly caching one as `Vec<char>` would be a type error at
    /// best. Under-detection here only costs a missed optimization (the access stays
    /// correct, just slow) -- it is never relied on to prove a variable safe to cache;
    /// callers separately restrict the optimization to `let` bindings and non-`var`
    /// parameters, which the checker already guarantees can't be reassigned, so the
    /// cached `Vec<char>` can never go stale.
    pub(crate) fn collect_str_index_targets(params: &[Param], stmts: &[Stmt]) -> std::collections::HashSet<String> {
        fn is_const_index(e: &Expr) -> bool {
            matches!(e.kind, ExprKind::Int(_))
        }
        // Pass 1: which names are knowable, without inference, to be string-typed?
        fn collect_string_typed(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
            for s in stmts {
                match s {
                    Stmt::Let(l) => {
                        let is_string_annotated = l.ty.as_ref().is_some_and(Transpiler::is_string_type);
                        let is_string_literal = l.ty.is_none()
                            && matches!(l.value.as_ref().map(|v| &v.kind), Some(ExprKind::Str(_)) | Some(ExprKind::StringInterp(_)));
                        if is_string_annotated || is_string_literal {
                            out.insert(l.name.clone());
                        }
                    }
                    Stmt::If(s) => {
                        for (_, body) in &s.branches { collect_string_typed(body, out); }
                        if let Some(eb) = &s.else_body { collect_string_typed(eb, out); }
                    }
                    Stmt::IfLet(s) => {
                        collect_string_typed(&s.then_body, out);
                        for branch in &s.elif_branches { collect_string_typed(&branch.body, out); }
                        if let Some(eb) = &s.else_body { collect_string_typed(eb, out); }
                    }
                    Stmt::Match(m) => {
                        for arm in &m.arms {
                            if let MatchBody::Block(b) = &arm.body { collect_string_typed(b, out); }
                        }
                    }
                    Stmt::While(w) => collect_string_typed(&w.body, out),
                    Stmt::WhileLet(w) => collect_string_typed(&w.body, out),
                    Stmt::DoWhile(d) => collect_string_typed(&d.body, out),
                    Stmt::Loop(l) => collect_string_typed(&l.body, out),
                    Stmt::For(f) => collect_string_typed(&f.body, out),
                    Stmt::Guard(g) => collect_string_typed(&g.else_body, out),
                    Stmt::Try(t) => {
                        collect_string_typed(&t.body, out);
                        for cc in &t.catch_clauses { collect_string_typed(&cc.body, out); }
                    }
                    Stmt::Defer(body) => collect_string_typed(body, out),
                    Stmt::KernelBlock(k) => collect_string_typed(&k.body, out),
                    Stmt::With(w) => collect_string_typed(&w.body, out),
                    _ => {}
                }
            }
        }
        let mut string_typed = std::collections::HashSet::new();
        for p in params {
            if p.ty.as_ref().is_some_and(Transpiler::is_string_type) {
                string_typed.insert(p.name.clone());
            }
        }
        collect_string_typed(stmts, &mut string_typed);
        fn walk_expr(e: &Expr, string_typed: &std::collections::HashSet<String>, out: &mut std::collections::HashSet<String>) {
            match &e.kind {
                ExprKind::Index(obj, idx) => {
                    if !is_const_index(idx) {
                        if let ExprKind::Var(v) = &obj.kind {
                            if string_typed.contains(v.as_str()) { out.insert(v.clone()); }
                        }
                    }
                    walk_expr(obj, string_typed, out);
                    walk_expr(idx, string_typed, out);
                }
                ExprKind::BinOp(_, l, r) => { walk_expr(l, string_typed, out); walk_expr(r, string_typed, out); }
                ExprKind::UnaryOp(_, x) => walk_expr(x, string_typed, out),
                ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r) => {
                    walk_expr(l, string_typed, out); walk_expr(r, string_typed, out);
                }
                ExprKind::Field(obj, _) => walk_expr(obj, string_typed, out),
                ExprKind::Call(callee, args) => {
                    walk_expr(callee, string_typed, out);
                    for a in args { walk_expr(&a.value, string_typed, out); }
                }
                ExprKind::MethodCall(obj, _, args) => {
                    walk_expr(obj, string_typed, out);
                    for a in args { walk_expr(&a.value, string_typed, out); }
                }
                ExprKind::GenericCall(callee, _, args) => {
                    walk_expr(callee, string_typed, out);
                    for a in args { walk_expr(&a.value, string_typed, out); }
                }
                ExprKind::Pipe(obj, _, args) => {
                    walk_expr(obj, string_typed, out);
                    for a in args { walk_expr(&a.value, string_typed, out); }
                }
                ExprKind::New { arena, ctor } => {
                    if let Some(a) = arena { walk_expr(a, string_typed, out); }
                    walk_expr(ctor, string_typed, out);
                }
                ExprKind::KernelLaunch { kernel, .. } => walk_expr(kernel, string_typed, out),
                ExprKind::TryElse(a, b) => { walk_expr(a, string_typed, out); walk_expr(b, string_typed, out); }
                ExprKind::TryElseBlock(body, else_body) => {
                    for s in body { walk_stmt(s, string_typed, out); }
                    for s in else_body { walk_stmt(s, string_typed, out); }
                }
                ExprKind::Array(items) | ExprKind::Tuple(items) | ExprKind::Set(items) => {
                    for it in items { walk_expr(it, string_typed, out); }
                }
                ExprKind::ArrayFill { value, count } => {
                    walk_expr(value, string_typed, out); walk_expr(count, string_typed, out);
                }
                ExprKind::ArrayAlloc { count } => walk_expr(count, string_typed, out),
                ExprKind::ArrayComp { expr, count, .. } => {
                    walk_expr(expr, string_typed, out); walk_expr(count, string_typed, out);
                }
                ExprKind::ArrayCompIter { expr, iter, .. } => {
                    walk_expr(expr, string_typed, out); walk_expr(iter, string_typed, out);
                }
                ExprKind::Dict(pairs) => {
                    for (k, v) in pairs { walk_expr(k, string_typed, out); walk_expr(v, string_typed, out); }
                }
                ExprKind::Range { start, end, .. } => {
                    walk_expr(start, string_typed, out); walk_expr(end, string_typed, out);
                }
                ExprKind::SliceRange { start, end, .. } => {
                    if let Some(s) = start { walk_expr(s, string_typed, out); }
                    if let Some(e) = end { walk_expr(e, string_typed, out); }
                }
                ExprKind::Cast(x, _) => walk_expr(x, string_typed, out),
                ExprKind::StringInterp(segs) => {
                    for seg in segs {
                        match seg {
                            StringSegment::Expr(e) | StringSegment::FormattedExpr(e, _) => walk_expr(e, string_typed, out),
                            StringSegment::Lit(_) => {}
                        }
                    }
                }
                _ => {}
            }
        }
        fn walk_cond_clause(c: &CondClause, string_typed: &std::collections::HashSet<String>, out: &mut std::collections::HashSet<String>) {
            match c {
                CondClause::Let(_, e) => walk_expr(e, string_typed, out),
                CondClause::LetPat(_, e) => walk_expr(e, string_typed, out),
                CondClause::Expr(e) => walk_expr(e, string_typed, out),
            }
        }
        fn walk_stmt(s: &Stmt, string_typed: &std::collections::HashSet<String>, out: &mut std::collections::HashSet<String>) {
            match s {
                Stmt::Let(l) => { if let Some(v) = &l.value { walk_expr(v, string_typed, out); } }
                Stmt::LetDestructure(l) => walk_expr(&l.value, string_typed, out),
                Stmt::Return(r) => { if let Some(e) = &r.value { walk_expr(e, string_typed, out); } }
                Stmt::Throw(t) => { if let Some(e) = &t.value { walk_expr(e, string_typed, out); } }
                Stmt::Break(_, Some(e)) => walk_expr(e, string_typed, out),
                Stmt::If(s) => {
                    for (cond, body) in &s.branches {
                        walk_expr(cond, string_typed, out);
                        for st in body { walk_stmt(st, string_typed, out); }
                    }
                    if let Some(eb) = &s.else_body { for st in eb { walk_stmt(st, string_typed, out); } }
                }
                Stmt::IfLet(s) => {
                    for c in &s.clauses { walk_cond_clause(c, string_typed, out); }
                    for st in &s.then_body { walk_stmt(st, string_typed, out); }
                    for branch in &s.elif_branches {
                        for c in &branch.clauses { walk_cond_clause(c, string_typed, out); }
                        for st in &branch.body { walk_stmt(st, string_typed, out); }
                    }
                    if let Some(eb) = &s.else_body { for st in eb { walk_stmt(st, string_typed, out); } }
                }
                Stmt::Match(m) => {
                    walk_expr(&m.subject, string_typed, out);
                    for arm in &m.arms {
                        if let Some(g) = &arm.guard { walk_expr(g, string_typed, out); }
                        match &arm.body {
                            MatchBody::Block(stmts) => for st in stmts { walk_stmt(st, string_typed, out); },
                            MatchBody::Expr(e) => walk_expr(e, string_typed, out),
                        }
                    }
                }
                Stmt::While(w) => {
                    walk_expr(&w.condition, string_typed, out);
                    for st in &w.body { walk_stmt(st, string_typed, out); }
                }
                Stmt::WhileLet(w) => {
                    walk_expr(&w.value, string_typed, out);
                    for st in &w.body { walk_stmt(st, string_typed, out); }
                }
                Stmt::DoWhile(d) => {
                    for st in &d.body { walk_stmt(st, string_typed, out); }
                    walk_expr(&d.condition, string_typed, out);
                }
                Stmt::Loop(l) => { for st in &l.body { walk_stmt(st, string_typed, out); } }
                Stmt::Wait(e, _) => walk_expr(e, string_typed, out),
                Stmt::For(f) => {
                    walk_expr(&f.iterable, string_typed, out);
                    for st in &f.body { walk_stmt(st, string_typed, out); }
                }
                Stmt::Guard(g) => {
                    match &g.cond {
                        GuardCond::Expr(e) => walk_expr(e, string_typed, out),
                        GuardCond::Clauses(cs) => for c in cs { walk_cond_clause(c, string_typed, out); },
                    }
                    for st in &g.else_body { walk_stmt(st, string_typed, out); }
                }
                Stmt::Try(t) => {
                    for st in &t.body { walk_stmt(st, string_typed, out); }
                    for cc in &t.catch_clauses { for st in &cc.body { walk_stmt(st, string_typed, out); } }
                }
                Stmt::Defer(body) => { for st in body { walk_stmt(st, string_typed, out); } }
                Stmt::Expr(e) => walk_expr(e, string_typed, out),
                Stmt::KernelBlock(k) => { for st in &k.body { walk_stmt(st, string_typed, out); } }
                Stmt::With(w) => { for st in &w.body { walk_stmt(st, string_typed, out); } }
                Stmt::Yield(e, _) => walk_expr(e, string_typed, out),
                _ => {}
            }
        }
        let mut out = std::collections::HashSet::new();
        for s in stmts { walk_stmt(s, &string_typed, &mut out); }
        out
    }

    /// Collect all `Bind(name)` leaves from a pattern into a set (for known_local_vars tracking).
    pub(crate) fn collect_pattern_binds(pat: &Pattern, out: &mut std::collections::HashSet<String>) {
        match pat {
            Pattern::Bind(name) => { out.insert(name.clone()); }
            Pattern::Some(inner) => Self::collect_pattern_binds(inner, out),
            Pattern::Variant(_, subs) | Pattern::Tuple(subs) => {
                for s in subs { Self::collect_pattern_binds(s, out); }
            }
            _ => {}
        }
    }

    pub(crate) fn emit_match(&mut self, s: &MatchStmt, is_last: bool) {
        // ── Special case: `match error:` with qualified enum-variant patterns ────────
        //
        // In the interpreter, `error` is the original thrown value (e.g. MyError::NotFound).
        // In the transpiler, `error` is Box<dyn Error> — direct match arms don't compile.
        //
        // When subject is `error` and the first non-wildcard arm pattern is a qualified
        // `Enum::Variant`, downcast error via BoringError before matching:
        //
        //   let __boring_error_typed = error.downcast_ref::<BoringError>()
        //     .and_then(|be| if let BoringError::Other(tid, inner) = be {
        //         if *tid == TypeId::of::<AppError>() { inner.as_any().downcast_ref::<AppError>() }
        //         else { None }
        //     } else { None });
        //   match __boring_error_typed {
        //     Some(AppError::NotFound) => body,    ← variant arms
        //     None => fallback,                    ← wildcard arm
        //   }
        //
        // This rewrite only applies when `error` is still the untyped `Box<dyn Error>`
        // from an untyped/generic catch. A typed `catch TypeName:` clause (for an enum
        // TypeName) has already downcast `error` to a concrete `&TypeName` reference
        // before this body is emitted (see emit_flow.rs's typed-catch dispatch) — in
        // that case `error_var_is_concrete_enum` is set and we must fall through to the
        // plain match below, since `error` has no `.downcast_ref::<BoringError>()` method.
        if let ExprKind::Var(vname) = &s.subject.kind {
            if vname == "error" && !self.error_var_is_concrete_enum {
                // Find enum type from first qualified variant pattern (name contains "::")
                let enum_type: Option<String> = s.arms.iter().find_map(|arm| {
                    arm.patterns.iter().find_map(|p| {
                        if let Pattern::Variant(name, _) = p {
                            name.split_once("::").map(|(et, _)| et.to_string())
                        } else { None }
                    })
                });

                if let Some(ref enum_ty) = enum_type {
                    // Emit the BoringError downcast to get Option<&EnumType>
                    self.line(&format!(
                        "let __boring_error_typed = error.downcast_ref::<BoringError>()\
                         .and_then(|__be| if let BoringError::Other(__tid, __boring_inner) = __be \
                         {{ if *__tid == std::any::TypeId::of::<{}>() \
                         {{ (**__boring_inner).as_any().downcast_ref::<{}>() }} \
                         else {{ None }} }} else {{ None }});",
                        enum_ty, enum_ty
                    ));

                    // Build a modified MatchStmt replacing the subject with __boring_error_typed
                    // and wrapping variant patterns in Some(…), wildcard → None.
                    let arms_transformed: Vec<MatchArm> = s.arms.iter().map(|arm| {
                        let new_pats: Vec<Pattern> = arm.patterns.iter().map(|p| match p {
                            Pattern::Wildcard => Pattern::Variant("None".into(), vec![]),
                            Pattern::Bind(_) => Pattern::Variant("None".into(), vec![]),
                            Pattern::Variant(name, subs) => {
                                // Rewrite qualified pattern: "AppError::NotFound" → Some(AppError::NotFound)
                                let inner = if subs.is_empty() {
                                    format!("Some({})", name)
                                } else {
                                    let sub_s: Vec<String> = subs.iter().map(|sp| match sp {
                                        Pattern::Bind(b) => b.clone(),
                                        Pattern::Wildcard => "_".into(),
                                        _ => "_".into(),
                                    }).collect();
                                    format!("Some({}({}))", name, sub_s.join(", "))
                                };
                                Pattern::Lit(LitPattern::Str(inner)) // use Str as a passthrough literal
                            }
                            other => other.clone(),
                        }).collect();
                        MatchArm { patterns: new_pats, guard: arm.guard.clone(), body: arm.body.clone(), line: arm.line, col: 0 }
                    }).collect();

                    // Emit the transformed match against __boring_error_typed
                    let use_value_body = is_last && !self.fn_returns_void;
                    self.line("match __boring_error_typed {");
                    self.indent += 1;
                    for arm in &arms_transformed {
                        for pat in &arm.patterns {
                            let pat_s = match pat {
                                Pattern::Lit(LitPattern::Str(s)) => s.clone(), // passthrough literal
                                Pattern::Variant(n, _) => n.clone(),
                                Pattern::Wildcard => "_".into(),
                                _ => "_".into(),
                            };
                            if let Some(guard) = &arm.guard {
                                self.line(&format!("{} if {} => {{", pat_s, self.emit_expr(guard)));
                            } else {
                                self.line(&format!("{} => {{", pat_s));
                            }
                        }
                        self.indent += 1;
                        match &arm.body {
                            MatchBody::Expr(e) => {
                                if use_value_body {
                                    let val = self.emit_expr_owned(e);
                                    self.line(&val);
                                } else {
                                    let val = self.emit_expr_owned(e);
                                    self.line(&format!("{};", val));
                                }
                            }
                            MatchBody::Block(stmts) => {
                                if use_value_body {
                                    self.emit_body(stmts);
                                } else {
                                    self.emit_loop_body(stmts);
                                }
                            }
                        }
                        self.indent -= 1;
                        self.line("}");
                    }
                    self.indent -= 1;
                    self.line("}");
                    return;
                }
            }
        }

        // When the whole match is the last stmt in a value-returning function,
        // each arm body also returns its value — use emit_body instead of emit_loop_body.
        // For throws functions, emit_body produces `return Ok(expr)` for arm tails.
        let use_value_body = is_last && !self.fn_returns_void;

        // Detect if the match subject is a function type parameter being matched against
        // concrete struct types. Rust cannot match `S` (a type param) against struct patterns,
        // so we transform this into a std::any::Any downcast chain.
        // The subject variable name (e.g. `s`) may differ from the type parameter name (e.g. `S`),
        // so we look up the variable's type in var_types and check if *that* is a type param.
        let subj_is_type_param = if let ExprKind::Var(vname) = &s.subject.kind {
            if self.current_fn_type_params.contains(vname.as_str()) {
                // Variable name itself is a type param (unusual but possible)
                true
            } else {
                // Check if the variable's declared type is a type parameter.
                // Param types can be Type::Named("S") or Type::TypeParam("S").
                let ty_name = self.var_types.get(vname.as_str()).and_then(|t| match t {
                    Type::Named(n) => Some(n.as_str()),
                    Type::TypeParam(n) => Some(n.as_str()),
                    _ => None,
                });
                ty_name.map(|n| self.current_fn_type_params.contains(n)).unwrap_or(false)
            }
        } else {
            false
        };
        let has_struct_arm = subj_is_type_param && s.arms.iter().any(|arm| {
            arm.patterns.iter().any(|p| {
                if let Pattern::Variant(name, _) = p {
                    self.struct_fields.contains_key(name.as_str())
                } else {
                    false
                }
            })
        });

        if has_struct_arm {
            // Emit an Any-downcast chain instead of a match expression.
            let subj_var = if let ExprKind::Var(vname) = &s.subject.kind {
                vname.clone()
            } else {
                self.emit_expr(&s.subject)
            };
            let any_var = format!("__any_{}", subj_var);
            self.line(&format!("let {} = &{} as &dyn std::any::Any;", any_var, subj_var));
            let mut first = true;
            for arm in &s.arms {
                // Check if this arm's pattern is a struct type check.
                let struct_name = arm.patterns.iter().find_map(|p| {
                    if let Pattern::Variant(name, _) = p {
                        if self.struct_fields.contains_key(name.as_str()) {
                            return Some(name.clone());
                        }
                    }
                    None
                });
                let is_wildcard = arm.patterns.iter().any(|p| matches!(p, Pattern::Wildcard));

                let kw = if first { "if" } else { "} else if" };
                if let Some(sname) = struct_name {
                    self.line(&format!("{} {}.is::<{}>() {{", kw, any_var, sname));
                    self.indent += 1;
                    // Emit arm body.
                    match &arm.body {
                        MatchBody::Expr(e) => {
                            let expr_s = self.emit_expr(e);
                            if use_value_body {
                                self.line(&expr_s);
                            } else {
                                self.line(&format!("{};", expr_s));
                            }
                        }
                        MatchBody::Block(stmts) => {
                            if use_value_body {
                                self.emit_body(stmts);
                            } else {
                                self.emit_loop_body(stmts);
                            }
                        }
                    }
                    self.indent -= 1;
                    first = false;
                } else if is_wildcard {
                    // Wildcard arm becomes the else branch.
                    self.line("} else {");
                    self.indent += 1;
                    match &arm.body {
                        MatchBody::Expr(e) => {
                            let expr_s = self.emit_expr(e);
                            if use_value_body {
                                self.line(&expr_s);
                            } else {
                                self.line(&format!("{};", expr_s));
                            }
                        }
                        MatchBody::Block(stmts) => {
                            if use_value_body {
                                self.emit_body(stmts);
                            } else {
                                self.emit_loop_body(stmts);
                            }
                        }
                    }
                    self.indent -= 1;
                }
            }
            self.line("}");
            return;
        }

        // Try to infer which enum type the match subject has, so patterns can be
        // qualified with the correct enum name when multiple enums share variant names.
        let inferred_enum = self.infer_match_enum(&s.subject, &s.arms);
        let prev_match_enum = self.match_subject_enum.clone();
        self.match_subject_enum = inferred_enum;

        let subj_raw = self.emit_expr(&s.subject);
        // Auto-clone: field accesses used as match subject would be moved out of the struct.
        let subj_raw = if matches!(&s.subject.kind, ExprKind::Field(..))
            && !subj_raw.ends_with(".clone()")
            && !subj_raw.starts_with('&')
            && !subj_raw.starts_with("Arc::")
            && !subj_raw.starts_with("Rc::")
            && !subj_raw.starts_with("{ let __g")
        {
            format!("{}.clone()", subj_raw)
        } else {
            subj_raw
        };
        // If the match subject is Rc<T> or Arc<T>, dereference it so patterns can match T.
        let subj_ty = if let ExprKind::Var(vname) = &s.subject.kind {
            self.var_types.get(vname.as_str()).cloned()
        } else {
            None
        };
        let is_smart_ptr = matches!(&subj_ty, Some(Type::Qualified(_, OwnerQual::Shared | OwnerQual::Owned | OwnerQual::New)));
        // Shared/actor params are passed as &Rc<T> / &Arc<T> — need double deref to reach T.
        let is_shared_ref_param = if let ExprKind::Var(vname) = &s.subject.kind {
            self.shared_ref_params.contains(vname.as_str())
        } else { false };
        // Arc<str> cannot be matched against &str patterns — emit `match &*var {` to coerce.
        let is_arc_str = matches!(&subj_ty, Some(Type::Str))
            || s.arms.iter().any(|arm| arm.patterns.iter().any(|p| matches!(p, Pattern::Lit(LitPattern::Str(_)))));
        let subj = if is_smart_ptr && is_shared_ref_param {
            format!("(**{})", subj_raw)
        } else if is_smart_ptr {
            format!("(*{})", subj_raw)
        } else if is_arc_str {
            format!("&*{}", subj_raw)
        } else {
            subj_raw
        };
        // When the match subject is `T?` (Option<T>), bare enum-variant patterns like
        // `Signal::BreakSignal(_)` must be wrapped in `Some(...)` to satisfy Rust's type checker.
        let subj_is_optional = matches!(&subj_ty, Some(Type::Optional(_)))
            || if let ExprKind::Var(vname) = &s.subject.kind {
                self.optional_vars.contains(vname.as_str())
            } else { false };
        let arms_for_emit: Option<Vec<MatchArm>> = if subj_is_optional {
            let has_variant_arm = s.arms.iter().any(|arm| arm.patterns.iter().any(|p| {
                matches!(p, Pattern::Variant(_, _) | Pattern::Bind(_))
                    && !matches!(p, Pattern::None)
            }));
            if has_variant_arm {
                Some(s.arms.iter().map(|arm| {
                    let new_pats: Vec<Pattern> = arm.patterns.iter().map(|p| match p {
                        Pattern::Lit(LitPattern::Nil) | Pattern::None => p.clone(),
                        Pattern::Wildcard => p.clone(),
                        Pattern::Bind(n) if n == "_" => p.clone(),
                        other => Pattern::Some(Box::new(other.clone())),
                    }).collect();
                    MatchArm { patterns: new_pats, guard: arm.guard.clone(), body: arm.body.clone(), line: arm.line, col: 0 }
                }).collect())
            } else {
                None
            }
        } else {
            None
        };
        let arms_ref: &[MatchArm] = arms_for_emit.as_deref().unwrap_or(&s.arms);

        // When the match subject holds a Mutex guard (e.g. `p.lock().unwrap().peek()`),
        // the guard lives for the entire match expression. If any match arm calls a function
        // that tries to lock the same Mutex, it deadlocks (std::sync::Mutex is not reentrant).
        // Fix: wrap in a block `{ expr }` so the guard is dropped before the arms execute.
        let subj = if subj.contains("lock().unwrap()") || subj.contains("borrow_mut()") || subj.contains("borrow()") {
            format!("{{ {} }}", subj)
        } else {
            subj
        };
        // Auto-clone the match subject when:
        // - it is a local optional var (optional_vars) with binding arms, AND
        // - there are non-wildcard binding arms that would partially move the subject.
        // This avoids E0382 "use of partially moved value" when the subject is also
        // used elsewhere after the match (e.g. as a return value in an else branch).
        let subj = if let ExprKind::Var(vname) = &s.subject.kind {
            let is_optional_var = self.optional_vars.contains(vname.as_str());
            let has_binding_arm = arms_ref.iter().any(|arm| arm.patterns.iter().any(|p| {
                matches!(p, Pattern::Variant(_, subs) if !subs.is_empty())
                    || matches!(p, Pattern::Some(inner) if matches!(inner.as_ref(), Pattern::Variant(_, subs) if !subs.is_empty()))
            }));
            if is_optional_var && has_binding_arm {
                format!("{}.clone()", vname)
            } else {
                subj
            }
        } else {
            subj
        };
        // Bare `self` is always a reference in Rust (&Self/&mut Self) — matching it
        // triggers match ergonomics (implicit `ref`/`ref mut` binding mode), under which
        // an explicit `mut` binding modifier on a sub-pattern is a hard compile error
        // ("cannot mutably bind by value within an implicitly-borrowing pattern"). It's
        // also unnecessary there: a variant field bound this way is already `&mut T`
        // (inside a `&mut self` method — see `enum_has_mut_field`/`is_enum_self` in
        // emit_top.rs), so calling a `def` method straight through it needs no `mut` at
        // all. `emit_match_arm` uses this to skip its method-call-triggered `mut`
        // promotion in exactly that one case; a plain owned local (not `self`, not a
        // by-reference parameter) still needs it, since there the bound field is an owned
        // value moved out of the match, and Rust requires `let mut` to take `&mut` of it.
        let subject_is_self = subj == "self";
        self.line(&format!("match {} {{", subj));
        self.indent += 1;
        for arm in arms_ref {
            self.emit_match_arm(arm, use_value_body, subject_is_self);
        }
        self.indent -= 1;
        self.line("}");

        self.match_subject_enum = prev_match_enum;
    }

    /// Infer the enum name for a match subject so we can qualify variant patterns.
    pub(crate) fn infer_match_enum(&self, subject: &Expr, arms: &[crate::ast::MatchArm]) -> Option<String> {
        // Strategy 1: subject is a named variable with a known type (from var_types).
        if let ExprKind::Var(vname) = &subject.kind {
            if let Some(ty) = self.var_types.get(vname.as_str()) {
                // Helper: extract the base type name, unwrapping smart-pointer qualifiers.
                let tname = match ty {
                    Type::Named(n) => Some(n.as_str()),
                    // MOpt'auto → Rc<MOpt>: extract "MOpt"
                    Type::Qualified(inner, _) => match inner.as_ref() {
                        Type::Named(n) => Some(n.as_str()),
                        _ => None,
                    },
                    // Optional(Named(n)) — unwrap the Option to get the inner enum name.
                    Type::Optional(inner) => match inner.as_ref() {
                        Type::Named(n) => Some(n.as_str()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(tname) = tname {
                    if self.enum_variant_fields.keys().any(|k| k.starts_with(&format!("{}::", tname))) {
                        return Some(tname.to_string());
                    }
                }
            }
        }

        // Strategy 1b: subject is a struct field access — explicit (`self.signal`,
        // `obj.field`) or implicit (`signal` inside a method, meaning `self.signal`).
        // Strategy 1 above only looks at `var_types`, which never has an entry for a
        // field name (only for actual local bindings), so a match on a field whose enum
        // type has variant names shared with another enum previously fell straight
        // through to Strategy 4's ambiguous-intersection check and then to emit_pattern's
        // "last enum registered wins" fallback. resolve_expr_struct_type already knows
        // how to walk field chains and the same implicit-self-field rule emit_expr uses.
        if let Some(tname) = self.resolve_expr_struct_type(subject) {
            if self.enum_variant_fields.keys().any(|k| k.starts_with(&format!("{}::", tname))) {
                return Some(tname);
            }
        }

        // Strategy 2: look at the function parameter type for the subject variable.
        if let ExprKind::Var(vname) = &subject.kind {
            // fn_sigs is keyed by function name, not useful here directly.
            // Check if it's `self` — use self_type.
            if vname == "self" {
                // Could be matching on self in an enum method, but skip for now.
            }
        }

        // Strategy 3: look for function call that returns an enum type.
        if let ExprKind::Call(callee, _) = &subject.kind {
            if let ExprKind::Var(fn_name) = &callee.kind {
                if let Some(Type::Named(tname)) = self.fn_return_types.get(fn_name.as_str()) {
                    if self.enum_variant_fields.keys().any(|k| k.starts_with(&format!("{}::", tname))) {
                        return Some(tname.clone());
                    }
                }
            }
        }

        // Strategy 4: if all arm variant patterns uniquely belong to one enum, use it.
        // Collect all variant names from the arm patterns.
        let mut variant_names: Vec<String> = Vec::new();
        for arm in arms {
            for pat in &arm.patterns {
                collect_pattern_variants(pat, &mut variant_names);
            }
        }

        if variant_names.is_empty() {
            return None;
        }

        // Find enum(s) that contain all variant names.
        // Build a set of candidate enums.
        let mut candidates: Option<std::collections::HashSet<String>> = None;
        for vname in &variant_names {
            let enums_with_variant: std::collections::HashSet<String> = self.enum_variant_fields
                .keys()
                .filter(|k| k.ends_with(&format!("::{}", vname)))
                .filter_map(|k| k.split("::").next().map(|s| s.to_string()))
                .collect();
            if enums_with_variant.is_empty() {
                continue;
            }
            candidates = Some(match candidates.take() {
                None => enums_with_variant,
                Some(prev) => prev.intersection(&enums_with_variant).cloned().collect(),
            });
        }

        match candidates {
            Some(set) if set.len() == 1 => set.into_iter().next(),
            _ => None,
        }
    }

    pub(crate) fn emit_match_arm(&mut self, arm: &MatchArm, use_value_body: bool, subject_is_self: bool) {
        // Collect bound variable names first so we can detect mutations in the arm body.
        let mut bound: Vec<String> = Vec::new();
        for p in &arm.patterns {
            Self::collect_pattern_bindings(p, &mut bound);
        }
        // Infer types for match-arm bound variables from enum variant field types, ahead of
        // mutation detection below — resolving a bound name's struct type (e.g. a variant
        // field declared `mut Point`, docs/mut-type-modifier.md) is needed there to tell a
        // `def` method call apart from a `req` one.
        // e.g. `Value.Int(a)` → var_types["a"] = Type::Int; `Value.Float(f)` → Type::Float64.
        let mut bound_types: Vec<(String, Type)> = Vec::new();
        for p in &arm.patterns {
            Self::collect_pattern_var_types(p, &self.enum_variant_field_types, self.match_subject_enum.as_deref(), &mut bound_types);
        }
        let mut bound_struct_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for (name, ty) in &bound_types {
            if let Type::Named(struct_name) = ty.without_mut() {
                if self.struct_fields.contains_key(struct_name.as_str()) {
                    bound_struct_map.insert(name.clone(), struct_name.clone());
                }
            }
        }
        // Detect which bound vars are mutated (index-assigned, directly assigned, or have a
        // `def` method called through them) in the body.
        let mutated = match &arm.body {
            MatchBody::Block(stmts) => self.collect_mutated_bindings(&bound, &bound_struct_map, subject_is_self, stmts),
            MatchBody::Expr(_) => std::collections::HashSet::new(),
        };
        // Detect nested Box<T> variant sub-patterns and rewrite them to temp bindings.
        // e.g. `Wrap(A(n))` where Wrap's field is Box<NInner> → `Wrap(__boring_b0)` + nested match.
        let subj_enum_ref2 = self.match_subject_enum.as_deref();
        let mut box_counter = 0usize;
        let mut all_nested: Vec<(String, Pattern)> = vec![];
        let rewritten_pats: Vec<Pattern> = arm.patterns.iter().map(|p| {
            let (rp, nested) = Self::rewrite_boxed_nested_variant(
                p, &self.enum_variant_field_types, subj_enum_ref2, &mut box_counter);
            all_nested.extend(nested);
            rp
        }).collect();
        let effective_pats: &[Pattern] = if all_nested.is_empty() { &arm.patterns } else { &rewritten_pats };

        // Emit patterns, promoting mutated bindings to `mut name`.
        let pats: Vec<String> = effective_pats.iter().map(|p| {
            self.emit_pattern_with_mut(p, &mutated)
        }).collect();
        let pat_s = pats.join(" | ");
        // Register all bound variables from this arm's patterns in known_local_vars so that
        // field accesses like `s.name` on pattern-bound vars are not treated as module paths.
        // Must happen BEFORE the guard is emitted just below (`x if x < 0:` needs `x` to
        // already read as a known local when emitting the guard condition — otherwise a
        // bare guard reference to a pattern-bound name that happens to also be a promoted
        // top-level scalar `const` gets misread as that const instead of the binding,
        // confirmed via examples/hello.br's `match n: ... x if x < 0: "negative"`, which
        // wrongly emitted `x if X < 0` once top-level scalar consts started being
        // uppercased on every target, not just GPU ones).
        for b in &bound {
            self.known_local_vars.insert(b.clone());
        }
        let guard = arm.guard.as_ref().map(|g| format!(" if {}", self.emit_expr(g))).unwrap_or_default();
        let mut bound_structs: Vec<String> = Vec::new();
        let mut bound_optionals: Vec<String> = Vec::new();
        // Content-mutation bookkeeping for struct-typed bound names — mirrors `let`'s
        // unconditional `mut_checked_local_vars` + conditional `content_mutable_local_vars`
        // (emit_let.rs:880-888): once a bound name's struct type is known, it's always
        // "checked" (so `emit_method_call_fallback`'s def-on-non-mut diagnostic can fire for
        // an un-qualified variant field, exactly like it already does for a plain struct
        // field one level down — emit_methods.rs's "One level down" comment), and separately
        // "content-mutable" only when the variant field's declared type actually grants it
        // (`mut Type`, docs/mut-type-modifier.md). Both scoped to this arm body and removed
        // afterward, like `bound_structs`.
        let mut bound_mut_checked: Vec<String> = Vec::new();
        for (name, ty) in &bound_types {
            self.var_types.insert(name.clone(), ty.clone());
            if Self::is_string_type(ty) {
                self.string_vars.insert(name.clone());
            }
            // Register Optional-typed pattern vars so they aren't double-wrapped in Some().
            if matches!(ty, Type::Optional(_)) {
                self.optional_vars.insert(name.clone());
                bound_optionals.push(name.clone());
            }
            // Register struct/enum-typed pattern vars so field accesses aren't mistaken for
            // JoinHandle, and so method dispatch recognizes an enum-typed variant field the
            // same way a struct-typed one already was (`is_known_user_type` covers both) —
            // e.g. `match wrap: W(b): b.position()` where `b`'s variant field type is itself
            // an enum with a `position()` method.
            if let Type::Named(struct_name) = ty.without_mut() {
                if self.is_known_user_type(struct_name.as_str()) {
                    self.var_struct_types.insert(name.clone(), struct_name.clone());
                    bound_structs.push(name.clone());
                    self.mut_checked_local_vars.insert(name.clone());
                    bound_mut_checked.push(name.clone());
                    if ty.grants_mut() {
                        self.content_mutable_local_vars.insert(name.clone());
                    }
                }
            }
        }
        // Register actor-typed pattern vars so field/method accesses get .borrow() wrapping.
        // These are scoped to this arm body and removed afterward.
        let mut bound_actors: Vec<String> = Vec::new();
        for (name, ty) in &bound_types {
            if matches!(ty.without_mut(), Type::Qualified(_, crate::ast::OwnerQual::Actor)) {
                bound_actors.push(name.clone());
                // Track actor-bound vars in var_mutex_types so emit_let_value knows they are
                // already Arc<Mutex<T>> and avoids double-wrapping them.
                self.var_mutex_types.insert(name.clone());
                match self.config.threading {
                    crate::transpiler::ThreadingMode::Single => { self.managed_refcell_vars.insert(name.clone()); }
                    crate::transpiler::ThreadingMode::Multi  => { self.managed_mutex_vars.insert(name.clone()); }
                }
            }
        }
        // Collect boxed bindings — pattern vars bound from recursive (Box<T>) fields.
        // We emit `let x = *x;` at the top of the arm body to auto-unbox them.
        let mut boxed_bindings: Vec<String> = Vec::new();
        let subj_enum_ref = self.match_subject_enum.as_deref();
        for p in &arm.patterns {
            Self::collect_boxed_bindings(p, &self.enum_variant_field_types, &self.recursive_fields, subj_enum_ref, &mut boxed_bindings);
        }
        // When there are nested Box<T> variant sub-patterns, wrap the arm in nested matches.
        // e.g. `Wrap(__boring_b0)` with nested [("__boring_b0", Pattern::Variant("A", [...]))]
        // emits: `Wrap(__boring_b0) => match __boring_b0.as_ref() { A(n) => body, _ => unreachable!() }`
        if !all_nested.is_empty() {
            self.line(&format!("{}{} => {{", pat_s, guard));
            self.indent += 1;
            // Emit nested matches, innermost first — each wraps the next.
            // For simplicity with one level of nesting, just emit one nested match.
            // Record the bound vars from the nested pattern so emit_body can see them.
            for (tmp_var, inner_pat) in &all_nested {
                // Save/restore match_subject_enum for inner context.
                let inner_enum = if let Pattern::Variant(vname, _) = inner_pat {
                    // Look up which enum this variant belongs to.
                    self.enum_variants.get(vname.as_str()).cloned()
                        .or_else(|| {
                            self.enum_variant_fields.keys()
                                .find(|k| k.ends_with(&format!("::{}", vname)))
                                .and_then(|k| k.split("::").next().map(|s| s.to_string()))
                        })
                } else { None };
                let prev_subj = self.match_subject_enum.clone();
                self.match_subject_enum = inner_enum;
                let inner_pat_s = self.emit_pattern(inner_pat);
                // Collect inner bindings to register in var_types.
                let mut inner_bound: Vec<String> = Vec::new();
                Self::collect_pattern_bindings(inner_pat, &mut inner_bound);
                for b in &inner_bound { self.known_local_vars.insert(b.clone()); }
                let mut inner_bound_types: Vec<(String, Type)> = Vec::new();
                Self::collect_pattern_var_types(inner_pat, &self.enum_variant_field_types, self.match_subject_enum.as_deref(), &mut inner_bound_types);
                for (n, t) in &inner_bound_types { self.var_types.insert(n.clone(), t.clone()); }
                self.line(&format!("match {}.as_ref() {{", tmp_var));
                self.indent += 1;
                // Body of inner arm — temporarily disable in_throws so we don't emit Ok(...).
                let prev_throws = self.in_throws;
                self.in_throws = false;
                let body_s = match &arm.body {
                    MatchBody::Expr(e) => self.emit_expr(e),
                    MatchBody::Block(stmts) => {
                        // For block bodies we need a block expression.
                        let mut sub = self.make_sub();
                        sub.fn_return_ty = self.fn_return_ty.clone();
                        sub.fn_returns_void = self.fn_returns_void;
                        // Copy inner bound vars into sub.
                        for (n, t) in &inner_bound_types { sub.var_types.insert(n.clone(), t.clone()); }
                        sub.emit_body(stmts);
                        sub.out.trim_end_matches('\n').to_string()
                    }
                };
                self.in_throws = prev_throws;
                for b in &boxed_bindings { self.line(&format!("let {} = *{};", b, b)); }
                self.line(&format!("{} => {},", inner_pat_s, body_s));
                self.line("_ => unreachable!(),");
                self.indent -= 1;
                self.line("}");
                self.match_subject_enum = prev_subj;
                for b in &inner_bound { self.known_local_vars.remove(b.as_str()); }
                for (n, _) in &inner_bound_types { self.var_types.remove(n.as_str()); }
            }
            self.indent -= 1;
            self.line("}");
        } else {
        match &arm.body {
            MatchBody::Expr(e) => {
                let expr_s = self.emit_expr(e);
                // If the enclosing function returns Option<T>, wrap non-optional arm
                // expressions in Some() so the caller doesn't need explicit `some(...)`.
                let expr_s = if matches!(&self.fn_return_ty, Some(Type::Optional(_)))
                    && !self.is_option_expr(e)
                    && expr_s != "None"
                    && !expr_s.starts_with("Some(")
                    && !matches!(&e.kind, ExprKind::Nil)
                    && !matches!(&e.kind, ExprKind::Var(v) if self.optional_vars.contains(v.as_str()))
                    && !matches!(&e.kind, ExprKind::Var(v) if self.var_types.get(v.as_str())
                        .map(|t| matches!(t, Type::Optional(_))).unwrap_or(false))
                    && !matches!(&e.kind, ExprKind::Call(callee, _)
                        if matches!(&callee.kind, ExprKind::Var(fn_name)
                            if self.fn_return_types.get(fn_name.as_str())
                                .map(|t| matches!(t, Type::Optional(_))).unwrap_or(false)))
                {
                    format!("Some({})", expr_s)
                } else {
                    expr_s
                };
                if boxed_bindings.is_empty() {
                    self.line(&format!("{}{} => {},", pat_s, guard, expr_s));
                } else {
                    // Can't emit let statements in expr position; wrap in a block.
                    self.line(&format!("{}{} => {{", pat_s, guard));
                    self.indent += 1;
                    for b in &boxed_bindings { self.line(&format!("let {} = *{};", b, b)); }
                    self.line(&expr_s);
                    self.indent -= 1;
                    self.line("}");
                }
            }
            MatchBody::Block(stmts) => {
                self.line(&format!("{}{} => {{", pat_s, guard));
                self.indent += 1;
                for b in &boxed_bindings { self.line(&format!("let {} = *{};", b, b)); }
                if use_value_body {
                    self.emit_body(stmts);
                } else {
                    self.emit_loop_body(stmts);
                }
                self.indent -= 1;
                self.line("}");
            }
        }
        } // end else !all_nested.is_empty()
        for b in &bound {
            self.known_local_vars.remove(b.as_str());
        }
        for (name, _) in &bound_types {
            self.var_types.remove(name.as_str());
            self.string_vars.remove(name.as_str());
        }
        for name in &bound_structs {
            self.var_struct_types.remove(name.as_str());
        }
        for name in &bound_mut_checked {
            self.mut_checked_local_vars.remove(name.as_str());
            self.content_mutable_local_vars.remove(name.as_str());
        }
        for name in &bound_optionals {
            self.optional_vars.remove(name.as_str());
        }
        // Remove actor tracking added for this arm (scoped to the arm body).
        for name in &bound_actors {
            self.var_mutex_types.remove(name.as_str());
            self.managed_refcell_vars.remove(name.as_str());
            self.managed_mutex_vars.remove(name.as_str());
        }
    }

    /// Rewrite a match pattern that contains nested variant sub-patterns inside Box<T> fields.
    ///
    /// When a variant field type is `Box<T>` (OwnerQual::Owned/New) and the sub-pattern is
    /// itself a non-trivial Variant (not Bind/Wildcard), Rust cannot match directly. We
    /// replace the sub-pattern with a temp binding `__boring_bN` and return the (binding, original)
    /// pairs so the caller can emit a nested `match __boring_bN.as_ref() { inner => body }`.
    fn rewrite_boxed_nested_variant(
        pat: &Pattern,
        enum_variant_field_types: &std::collections::HashMap<String, Vec<Type>>,
        subject_enum: Option<&str>,
        counter: &mut usize,
    ) -> (Pattern, Vec<(String, Pattern)>) {
        let Pattern::Variant(name, fields) = pat else {
            return (pat.clone(), vec![]);
        };
        // Resolve the enum::variant key (same logic as collect_boxed_bindings).
        let field_tys_key = if enum_variant_field_types.contains_key(name.as_str()) {
            Some(name.as_str().to_string())
        } else if !name.contains("::") {
            let suffix = format!("::{}", name);
            let subject_key = subject_enum.and_then(|en| {
                let k = format!("{}::{}", en, name);
                if enum_variant_field_types.contains_key(k.as_str()) { Some(k) } else { None }
            });
            subject_key.or_else(|| {
                enum_variant_field_types.iter()
                    .filter(|(k, _)| k.ends_with(&suffix))
                    .max_by_key(|(_, v)| v.len())
                    .map(|(k, _)| k.clone())
            })
        } else { None };
        let Some(key) = field_tys_key else {
            return (pat.clone(), vec![]);
        };
        let field_types = &enum_variant_field_types[&key];
        let mut new_fields = fields.clone();
        let mut nested: Vec<(String, Pattern)> = vec![];
        for (i, sub_pat) in fields.iter().enumerate() {
            let is_box = field_types.get(i)
                .map(|t| matches!(t, Type::Qualified(_, OwnerQual::Owned | OwnerQual::New)))
                .unwrap_or(false);
            let is_nested_variant = matches!(sub_pat, Pattern::Variant(_, _));
            if is_box && is_nested_variant {
                let tmp = format!("__boring_b{}", *counter);
                *counter += 1;
                new_fields[i] = Pattern::Bind(tmp.clone());
                nested.push((tmp, sub_pat.clone()));
            }
        }
        (Pattern::Variant(name.clone(), new_fields), nested)
    }

    /// Infer types for pattern-bound variables using enum_variant_field_types.
    /// e.g. Pattern::Variant("Value", "Int", [Bind("a")]) → ("a", Type::Int)
    /// Collect names of pattern bindings that are stored as Box<T> in the enum variant.
    /// These must be auto-unboxed (`let x = *x;`) at the start of the match arm body.
    /// `subject_enum` is the enum name of the match subject, used to prefer the right variant.
    fn collect_boxed_bindings(
        pat: &Pattern,
        enum_variant_field_types: &std::collections::HashMap<String, Vec<Type>>,
        recursive_fields: &std::collections::HashSet<String>,
        subject_enum: Option<&str>,
        out: &mut Vec<String>,
    ) {
        match pat {
            Pattern::Variant(qual_name, fields) => {
                // Try to find the enum variant key "Enum::Variant".
                // Prefer the match subject enum if available to avoid picking the wrong variant.
                let field_tys_key = if enum_variant_field_types.contains_key(qual_name.as_str()) {
                    Some(qual_name.as_str().to_string())
                } else if !qual_name.contains("::") {
                    let suffix = format!("::{}", qual_name);
                    // First, prefer the subject enum's variant if it exists.
                    let subject_key = subject_enum.and_then(|en| {
                        let k = format!("{}::{}", en, qual_name);
                        if enum_variant_field_types.contains_key(k.as_str()) { Some(k) } else { None }
                    });
                    subject_key.or_else(|| {
                        enum_variant_field_types.iter()
                            .filter(|(k, _)| k.ends_with(&suffix))
                            .max_by_key(|(_, v)| v.len())
                            .map(|(k, _)| k.clone())
                    })
                } else { None };
                if let Some(key) = field_tys_key {
                    let parts: Vec<&str> = key.splitn(2, "::").collect();
                    let (enum_name, variant_name) = if parts.len() == 2 { (parts[0], parts[1]) } else { return };
                    for (i, f) in fields.iter().enumerate() {
                        let rec_key = format!("{}::{}::{}", enum_name, variant_name, i);
                        if recursive_fields.contains(&rec_key) {
                            if let Pattern::Bind(name) = f {
                                out.push(name.clone());
                            }
                        } else {
                            Self::collect_boxed_bindings(f, enum_variant_field_types, recursive_fields, subject_enum, out);
                        }
                    }
                }
            }
            Pattern::Some(inner) => Self::collect_boxed_bindings(inner, enum_variant_field_types, recursive_fields, subject_enum, out),
            Pattern::Tuple(fields) => {
                for f in fields { Self::collect_boxed_bindings(f, enum_variant_field_types, recursive_fields, subject_enum, out); }
            }
            _ => {}
        }
    }

    fn collect_pattern_var_types(
        pat: &Pattern,
        enum_variant_field_types: &std::collections::HashMap<String, Vec<Type>>,
        subject_enum: Option<&str>,
        out: &mut Vec<(String, Type)>,
    ) {
        match pat {
            Pattern::Variant(qual_name, fields) => {
                // qual_name may be "Enum::Variant" or just "Variant"; try both.
                // Also try all enums whose variant matches the unqualified name.
                let field_tys = enum_variant_field_types.get(qual_name.as_str())
                    .cloned()
                    .or_else(|| {
                        if !qual_name.contains("::") {
                            let suffix = format!("::{}", qual_name);
                            // Prefer the subject enum's variant if known, to avoid ambiguity
                            // (e.g. Stmt::Fn vs Value::Fn when matching on a Stmt).
                            if let Some(en) = subject_enum {
                                let subject_key = format!("{}::{}", en, qual_name);
                                if let Some(tys) = enum_variant_field_types.get(&subject_key) {
                                    return Some(tys.clone());
                                }
                            }
                            // Fallback: pick the variant with the most fields.
                            enum_variant_field_types.iter()
                                .filter(|(k, _)| k.ends_with(&suffix))
                                .max_by_key(|(_, v)| v.len())
                                .map(|(_, v)| v.clone())
                        } else { None }
                    })
                    .unwrap_or_default();
                for (i, f) in fields.iter().enumerate() {
                    if let Pattern::Bind(name) = f {
                        if let Some(ty) = field_tys.get(i) {
                            out.push((name.clone(), ty.clone()));
                        }
                    } else {
                        Self::collect_pattern_var_types(f, enum_variant_field_types, subject_enum, out);
                    }
                }
            }
            Pattern::Some(inner) => Self::collect_pattern_var_types(inner, enum_variant_field_types, subject_enum, out),
            Pattern::Tuple(fields) => {
                for f in fields { Self::collect_pattern_var_types(f, enum_variant_field_types, subject_enum, out); }
            }
            _ => {}
        }
    }

    fn collect_pattern_bindings(pat: &Pattern, out: &mut Vec<String>) {
        match pat {
            Pattern::Bind(n) => out.push(n.clone()),
            Pattern::Variant(_, fields) => {
                for f in fields { Self::collect_pattern_bindings(f, out); }
            }
            Pattern::Some(inner) => Self::collect_pattern_bindings(inner, out),
            Pattern::Tuple(fields) => {
                for f in fields { Self::collect_pattern_bindings(f, out); }
            }
            _ => {}
        }
    }

    /// Like `emit_pattern` but adds `mut` before each mutated binding name in the output string.
    /// Uses post-processing to avoid re-implementing enum qualification logic.
    pub(crate) fn emit_pattern_with_mut(&self, pat: &Pattern, mutated: &std::collections::HashSet<String>) -> String {
        let s = self.emit_pattern(pat);
        if mutated.is_empty() { return s; }
        let mut result = s;
        for name in mutated {
            // Insert `mut ` before the name in the common positional contexts within a pattern string.
            // Handles: `(name)`, `(name, `, `, name)`, `, name, `.
            let n = name.as_str();
            result = result
                .replace(&format!("({})", n), &format!("(mut {})", n))
                .replace(&format!("({}, ", n), &format!("(mut {}, ", n))
                .replace(&format!(", {})", n), &format!(", mut {})", n))
                .replace(&format!(", {}, ", n), &format!(", mut {}, ", n));
            // Also handle the case where the pattern IS just the name (top-level Bind).
            if result == *name {
                result = format!("mut {}", name);
            }
        }
        result
    }

    pub(crate) fn emit_pattern(&self, pat: &Pattern) -> String {
        match pat {
            Pattern::Wildcard   => "_".into(),
            Pattern::Bind(n)    => n.clone(),
            Pattern::None       => {
                // If the match subject is a user-defined enum with a `None` variant, qualify it.
                if let Some(enum_name) = &self.match_subject_enum {
                    let key = format!("{}::None", enum_name);
                    if self.enum_variant_fields.contains_key(&key) {
                        return format!("{}::None", enum_name);
                    }
                }
                "None".into()
            }
            Pattern::Some(inner) => {
                // If the match subject is a user-defined enum with a `Some` variant,
                // qualify the pattern as `EnumName::Some(...)`.
                let inner_s = self.emit_pattern(inner);
                let wildcard = matches!(inner.as_ref(), Pattern::Wildcard);
                if let Some(enum_name) = &self.match_subject_enum {
                    let key = format!("{}::Some", enum_name);
                    if self.enum_variant_fields.contains_key(&key) {
                        if wildcard {
                            return format!("{}::Some(_)", enum_name);
                        } else {
                            return format!("{}::Some({})", enum_name, inner_s);
                        }
                    }
                }
                // Boring Optional — standard Rust Option pattern
                if wildcard { "Some(_)".into() } else { format!("Some({})", inner_s) }
            }
            Pattern::Lit(lit)   => match lit {
                LitPattern::Int(n)  => n.to_string(),
                LitPattern::Float(f) => {
                    let s = format!("{}", f);
                    if s.contains('.') || s.contains('e') || s.contains('E') { s }
                    else { format!("{}.0", s) }
                }
                LitPattern::Str(s)  => format!("\"{}\"", crate::transpiler::helpers::escape_str(s)),
                LitPattern::Bool(b) => b.to_string(),
                LitPattern::Nil     => "None".into(),
            },
            Pattern::Variant(name, fields) => {
                // Pre-qualified pattern `Enum::Variant` (written as `Enum.Variant` in Boring) —
                // emit verbatim, skip all lookup / re-qualification logic.
                if name.contains("::") {
                    return if fields.is_empty() {
                        name.clone()
                    } else {
                        let fs: Vec<String> = fields.iter().map(|p| self.emit_pattern(p)).collect();
                        format!("{}({})", name, fs.join(", "))
                    };
                }
                // `Some(x)` and `None` — treat as Option patterns ONLY when the match
                // subject is not a user-defined enum that has a real `Some`/`None` variant.
                let subject_has_variant = self.match_subject_enum.as_deref().map(|en| {
                    self.enum_variant_fields.contains_key(&format!("{}::{}", en, name))
                }).unwrap_or(false);
                if !subject_has_variant {
                    if (name == "Some" || name == "some") && fields.len() == 1 {
                        return format!("Some({})", self.emit_pattern(&fields[0]));
                    }
                    if (name == "None" || name == "none") && fields.is_empty() {
                        return "None".into();
                    }
                }
                // If the name is a variant of the match subject enum, prefer the variant
                // interpretation — even if a struct with the same name exists.
                let is_subject_variant = self.match_subject_enum.as_deref().map(|en| {
                    self.enum_variant_fields.contains_key(&format!("{}::{}", en, name))
                }).unwrap_or(false);

                // Check if this is a struct (not an enum variant) — only when not a subject variant.
                if !is_subject_variant {
                if let Some(field_names) = self.struct_fields.get(name.as_str()) {
                    // Struct destructuring: `Point2D { x, y }` or `Point2D { x: a, y: _ }`
                    if fields.is_empty() {
                        // Type-check-only pattern: `Point2D { .. }`
                        return format!("{} {{ .. }}", name);
                    }
                    let pairs: Vec<String> = fields.iter().enumerate().map(|(i, pat)| {
                        let fname = field_names.get(i).map(|(n, _)| n.as_str()).unwrap_or("_");
                        match pat {
                            Pattern::Wildcard => format!("{}: _", fname),
                            Pattern::Bind(b) if b == fname => fname.to_string(),
                            Pattern::Bind(b) => format!("{}: {}", fname, b),
                            _ => format!("{}: {}", fname, self.emit_pattern(pat)),
                        }
                    }).collect();
                    return format!("{} {{ {} }}", name, pairs.join(", "));
                }
                } // end !is_subject_variant
                // Qualify with enum name: `Num(v)` → `ExprEnum::Num(v)`
                // Priority: 1) use the inferred enum from match subject (match_subject_enum)
                //           2) disambiguate by field count (tuple vs unit variant)
                //           3) fall back to last-registered enum_variants entry
                let qualified = if let Some(subject_enum) = &self.match_subject_enum {
                    // Use the inferred enum from the match context.
                    let key = format!("{}::{}", subject_enum, name);
                    if self.enum_variant_fields.contains_key(&key) {
                        format!("{}::{}", subject_enum, name)
                    } else if let Some(enum_name) = self.enum_variants.get(name) {
                        format!("{}::{}", enum_name, name)
                    } else {
                        name.clone()
                    }
                } else if let Some(enum_name) = self.enum_variants.get(name) {
                    if !fields.is_empty() {
                        // Check if the mapped enum's variant actually has fields.
                        let mapped_key = format!("{}::{}", enum_name, name);
                        let mapped_has_fields = self.enum_variant_fields.get(&mapped_key)
                            .map(|f| !f.is_empty())
                            .unwrap_or(false);
                        if !mapped_has_fields {
                            // Look for another enum that has this variant with fields.
                            let alt = self.enum_variant_fields.iter()
                                .find(|(k, v)| {
                                    k.ends_with(&format!("::{}", name)) && !v.is_empty()
                                })
                                .and_then(|(k, _)| k.split("::").next().map(|s| s.to_string()));
                            if let Some(alt_enum) = alt {
                                format!("{}::{}", alt_enum, name)
                            } else {
                                format!("{}::{}", enum_name, name)
                            }
                        } else {
                            format!("{}::{}", enum_name, name)
                        }
                    } else {
                        format!("{}::{}", enum_name, name)
                    }
                } else {
                    name.clone()
                };
                if fields.is_empty() {
                    // Check if the qualified variant is actually a tuple variant (has fields in Rust).
                    // If so, emit `Variant(..)` to match all fields (Boring unit pattern = wildcard).
                    let variant_has_fields = self.enum_variant_fields
                        .get(&qualified)
                        .map(|f| !f.is_empty())
                        .unwrap_or(false);
                    if variant_has_fields {
                        format!("{} (..)", qualified)
                    } else {
                        qualified
                    }
                } else {
                    let fs: Vec<String> = fields.iter().map(|p| self.emit_pattern(p)).collect();
                    format!("{}({})", qualified, fs.join(", "))
                }
            }
            Pattern::Tuple(elems) => {
                let es: Vec<String> = elems.iter().map(|p| self.emit_pattern(p)).collect();
                format!("({})", es.join(", "))
            }
        }
    }

}
