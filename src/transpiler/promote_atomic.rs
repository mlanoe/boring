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

//! Automatic `'actor`/`'guard` → `'atomic` promotion (Part 2 of the `'atomic` design —
//! see `docs/qualifiers.md`'s `'atomic` section).
//!
//! This is architecturally distinct from Part 1's `'atomic` qualifier itself: it runs
//! as its own analysis/rewrite pass, strictly AFTER ordinary qualifier resolution has
//! already committed a local binding to `'actor`/`'guard` — never as another candidate
//! competing in `infer_qualifiers.rs`'s priority-ordered fallback (`'atomic` already
//! never wins that fallback on its own, since `'actor` always precedes it — see
//! `resolve_fallback`'s doc comment — so a promotion that only ever fired through that
//! shared candidate list would never fire at all; this pass is why it fires anyway).
//!
//! Promotes a local (never a struct field, never a parameter, never a return value)
//! `'actor`/`'guard`-qualified scalar binding to the `'atomic` representation
//! (`var_atomic_types`, replacing its `var_mutex_types`/`var_rwlock_types` membership)
//! only when ALL FOUR of the following hold, checked within the same function-local
//! analysis scope `with`'s own mutation scan already uses (recursing into
//! `if`/`while`/`for`/`match`/`loop`/`do-while`/`guard`/`try`/`defer`/closures nested in
//! the same function, never into a called function's own body — see
//! `ast::with_block_mutates`'s doc comment for the precedent this leans on):
//!
//! 1. The underlying type is one of the atomic-representable scalars from Part 1
//!    (`Type::is_atomic_eligible_scalar`) — never a struct, never a float.
//! 2. The binding never escapes the analyzed local scope: never returned, never
//!    assigned into a struct field, never captured by a `task`/closure, never passed
//!    as an argument to a function/method call (its own recognized single-op methods
//!    excepted — see point 3).
//! 3. Every individual access decomposes 1:1 into a single atomic primitive per Part
//!    1's operation mapping: a bare read, `x = <value not referencing x>`,
//!    `x = x + <value not referencing x>` / `x = x - <value not referencing x>`
//!    (`fetch_add`/`fetch_sub` — genuinely a single atomic instruction), or
//!    `x.swap(<value not referencing x>)`. Any other shape referencing `x` on an
//!    assignment's RHS (`x = x * 2`, `x = f(x)`, `x = x + x`, ...) would require
//!    decomposing into a separate load then store under the atomic representation —
//!    NOT equivalent to the original lock-protected read-modify-write, so it blocks
//!    promotion rather than firing an unsound rewrite.
//! 4. The binding is never used inside a `with` block anywhere within the same local
//!    analysis scope (same non-whole-program boundary as `with`'s own existing scan —
//!    a known, accepted, already-documented limitation, not a new one).
//!
//! Purely an optimization: never changes observable program behavior, only the
//! underlying representation. False negatives (missing a safe promotion) are
//! acceptable; false positives (an unsound promotion) are not — every construct this
//! walker doesn't specifically recognize as safe is treated conservatively as
//! escaping/unsafe, blocking promotion rather than guessing.
//!
//! Deferred (see docs/qualifiers.md's `'atomic` section): cross-function whole-program
//! promotion (a promoted variable passed to another Boring function/method is always
//! treated as escaping, exactly like `with`'s own local-only scan never opens a
//! called function's body), and compare-and-swap pattern detection
//! (`if x == a: x = b` stays on `Mutex`/`RwLock`, never auto-promoted).

use super::Transpiler;
use crate::ast::{
    CondClause, Expr, ExprKind, GuardCond, MatchBody, OwnerQual, Stmt, Type,
};
use super::helpers::collect_var_names;

impl Transpiler {
    /// Pre-pass, run once per function body alongside `infer_qualifiers` (before any
    /// statement is emitted) — populates `self.promoted_atomic_vars` with the names
    /// that `try_emit_qualified_let`'s `'actor`/`'guard` branches should emit as
    /// `'atomic` instead.
    pub(crate) fn scan_atomic_promotions(&mut self, stmts: &[Stmt]) {
        self.promoted_atomic_vars.clear();
        let mut candidates: Vec<String> = Vec::new();
        collect_actor_guard_scalar_lets(stmts, &mut candidates);
        for name in candidates {
            let mut escapes = false;
            let mut bad_op = false;
            stmts_use_name(stmts, &name, &mut escapes, &mut bad_op);
            if !escapes && !bad_op {
                self.promoted_atomic_vars.insert(name);
            }
        }
    }
}

/// Criterion 1 + candidate collection: every local `let`/`var`/`mut` binding whose
/// declared type is `Qualified(inner, Actor | Guard)` with an atomic-eligible scalar
/// `inner`. Recurses into the same constructs `with_block_mutates` does — never into
/// a nested `Fn`/`Struct`/`Enum`/`Mod` (new scope, new callable, signature only).
fn collect_actor_guard_scalar_lets(stmts: &[Stmt], out: &mut Vec<String>) {
    for stmt in stmts {
        match stmt {
            Stmt::Let(s) => {
                if let Some(ty) = &s.ty {
                    if let Type::Qualified(inner, OwnerQual::Actor | OwnerQual::Guard) = ty.without_mut() {
                        if inner.is_atomic_eligible_scalar() {
                            out.push(s.name.clone());
                        }
                    }
                }
            }
            Stmt::If(s) => {
                for (_, body) in &s.branches { collect_actor_guard_scalar_lets(body, out); }
                if let Some(eb) = &s.else_body { collect_actor_guard_scalar_lets(eb, out); }
            }
            Stmt::IfLet(s) => {
                collect_actor_guard_scalar_lets(&s.then_body, out);
                for b in &s.elif_branches { collect_actor_guard_scalar_lets(&b.body, out); }
                if let Some(eb) = &s.else_body { collect_actor_guard_scalar_lets(eb, out); }
            }
            Stmt::While(s) => collect_actor_guard_scalar_lets(&s.body, out),
            Stmt::WhileLet(s) => collect_actor_guard_scalar_lets(&s.body, out),
            Stmt::DoWhile(s) => collect_actor_guard_scalar_lets(&s.body, out),
            Stmt::Loop(s) => collect_actor_guard_scalar_lets(&s.body, out),
            Stmt::For(s) => collect_actor_guard_scalar_lets(&s.body, out),
            Stmt::Guard(s) => collect_actor_guard_scalar_lets(&s.else_body, out),
            Stmt::Try(s) => {
                collect_actor_guard_scalar_lets(&s.body, out);
                for c in &s.catch_clauses { collect_actor_guard_scalar_lets(&c.body, out); }
            }
            Stmt::Defer(body) => collect_actor_guard_scalar_lets(body, out),
            Stmt::With(w) => collect_actor_guard_scalar_lets(&w.body, out),
            Stmt::Match(s) => {
                for arm in &s.arms {
                    if let MatchBody::Block(body) = &arm.body {
                        collect_actor_guard_scalar_lets(body, out);
                    }
                }
            }
            _ => {}
        }
    }
}

/// True if `name` is referenced anywhere in `e` (bare-read shorthand for the many
/// call sites below that don't need finer-grained classification).
fn expr_refs(e: &Expr, name: &str) -> bool {
    collect_var_names(e).iter().any(|n| n == name)
}

/// Criteria 2 + 3, statement-level: walks the same local scope `collect_actor_guard_scalar_lets`
/// does (see its doc), setting `escapes` on any use this pass doesn't positively prove
/// safe, and `bad_op` on an assignment whose shape isn't one of Part 1's single atomic
/// primitives. Criterion 4 (`with` usage) is folded in here too: `Stmt::With` sets
/// `escapes` directly when `name` is one of its own scoped names.
fn stmts_use_name(stmts: &[Stmt], name: &str, escapes: &mut bool, bad_op: &mut bool) {
    for stmt in stmts { stmt_use_name(stmt, name, escapes, bad_op); }
}

fn stmt_use_name(stmt: &Stmt, name: &str, escapes: &mut bool, bad_op: &mut bool) {
    match stmt {
        Stmt::Let(s) => {
            if let Some(v) = &s.value { expr_use_name(v, name, escapes, bad_op); }
        }
        Stmt::LetDestructure(s) => expr_use_name(&s.value, name, escapes, bad_op),
        Stmt::Return(r) => {
            // A returned value can never reference the candidate — the whole point of
            // the promotion is that the binding stays wholly local.
            if let Some(v) = &r.value {
                if expr_refs(v, name) { *escapes = true; }
                expr_use_name(v, name, escapes, bad_op);
            }
        }
        Stmt::Break(_, Some(v)) => {
            if expr_refs(v, name) { *escapes = true; }
            expr_use_name(v, name, escapes, bad_op);
        }
        Stmt::Break(_, None) | Stmt::Continue(_) | Stmt::Comment(_) => {}
        Stmt::Throw(t) => {
            if let Some(v) = &t.value {
                if expr_refs(v, name) { *escapes = true; }
                expr_use_name(v, name, escapes, bad_op);
            }
        }
        Stmt::If(s) => {
            for (cond, body) in &s.branches {
                expr_use_name(cond, name, escapes, bad_op);
                stmts_use_name(body, name, escapes, bad_op);
            }
            if let Some(eb) = &s.else_body { stmts_use_name(eb, name, escapes, bad_op); }
        }
        Stmt::IfLet(s) => {
            for c in &s.clauses { cond_clause_use_name(c, name, escapes, bad_op); }
            stmts_use_name(&s.then_body, name, escapes, bad_op);
            for b in &s.elif_branches {
                for c in &b.clauses { cond_clause_use_name(c, name, escapes, bad_op); }
                stmts_use_name(&b.body, name, escapes, bad_op);
            }
            if let Some(eb) = &s.else_body { stmts_use_name(eb, name, escapes, bad_op); }
        }
        Stmt::Match(s) => {
            expr_use_name(&s.subject, name, escapes, bad_op);
            for arm in &s.arms {
                if let Some(g) = &arm.guard { expr_use_name(g, name, escapes, bad_op); }
                match &arm.body {
                    MatchBody::Expr(e) => expr_use_name(e, name, escapes, bad_op),
                    MatchBody::Block(b) => stmts_use_name(b, name, escapes, bad_op),
                }
            }
        }
        Stmt::While(s) => {
            expr_use_name(&s.condition, name, escapes, bad_op);
            stmts_use_name(&s.body, name, escapes, bad_op);
        }
        Stmt::WhileLet(s) => {
            if expr_refs(&s.value, name) { *escapes = true; }
            expr_use_name(&s.value, name, escapes, bad_op);
            stmts_use_name(&s.body, name, escapes, bad_op);
        }
        Stmt::DoWhile(s) => {
            stmts_use_name(&s.body, name, escapes, bad_op);
            expr_use_name(&s.condition, name, escapes, bad_op);
        }
        Stmt::Loop(s) => stmts_use_name(&s.body, name, escapes, bad_op),
        Stmt::Wait(e, _) | Stmt::Yield(e, _) => expr_use_name(e, name, escapes, bad_op),
        Stmt::For(s) => {
            if expr_refs(&s.iterable, name) { *escapes = true; } // iterated-over — not a single atomic access
            expr_use_name(&s.iterable, name, escapes, bad_op);
            stmts_use_name(&s.body, name, escapes, bad_op);
        }
        Stmt::Guard(s) => {
            match &s.cond {
                GuardCond::Expr(e) => expr_use_name(e, name, escapes, bad_op),
                GuardCond::Clauses(cs) => for c in cs { cond_clause_use_name(c, name, escapes, bad_op); },
            }
            stmts_use_name(&s.else_body, name, escapes, bad_op);
        }
        Stmt::Try(s) => {
            stmts_use_name(&s.body, name, escapes, bad_op);
            for c in &s.catch_clauses { stmts_use_name(&c.body, name, escapes, bad_op); }
        }
        Stmt::Defer(body) => stmts_use_name(body, name, escapes, bad_op),
        Stmt::Expr(e) => expr_use_name(e, name, escapes, bad_op),
        // Criterion 4: `name` is never allowed inside any `with` block, own-target or
        // not — an enclosing `with othername:` block still only holds `othername`'s
        // lock/residency, so a plain read/write of our candidate inside it is
        // otherwise ordinary and handled by the recursion; but if `name` is itself
        // one of this block's own scoped names, atomics have no lock/guard to hold
        // across it (mirrors the checker's own with-incompatibility rejection).
        Stmt::With(w) => {
            if w.names.iter().any(|n| n == name) { *escapes = true; }
            stmts_use_name(&w.body, name, escapes, bad_op);
        }
        // New scope / new callable — signature only, never the body (mirrors
        // `ast::with_block_mutates`'s identical rule). Boring has no implicit
        // capture into a nested `fn`, so nothing further to check here.
        Stmt::Fn(_) | Stmt::Struct(_) | Stmt::Enum(_) | Stmt::Mod(_) | Stmt::Alias(_) => {}
        // GPU kernel blocks are a wholly separate codegen path with their own
        // qualifier space ('gpu'unified/'gpu'global/...) — out of scope for
        // `'atomic` entirely (Part 1's checker never allows a kernel-context
        // qualifier to combine with `'atomic`). Conservatively treated as opaque:
        // not recursed into, so a real reference to `name` inside one would be a
        // missed escape (a false negative on the "should we promote" question would
        // actually risk soundness here) — mitigated by declining to promote whenever
        // this function contains a kernel block at all.
        Stmt::KernelBlock(_) => { *escapes = true; }
    }
}

fn cond_clause_use_name(c: &CondClause, name: &str, escapes: &mut bool, bad_op: &mut bool) {
    match c {
        CondClause::Let(_, e) | CondClause::LetPat(_, e) | CondClause::Expr(e) => {
            expr_use_name(e, name, escapes, bad_op);
        }
    }
}

/// Criteria 2 + 3, expression-level. `e` is being evaluated in an ordinary
/// (non-escaping) context unless a specific sub-position below says otherwise.
fn expr_use_name(e: &Expr, name: &str, escapes: &mut bool, bad_op: &mut bool) {
    match &e.kind {
        // The one recognized read/write shape: `name = <rhs>`.
        ExprKind::Assign(target, value) => {
            if matches!(&target.kind, ExprKind::Var(v) if v == name) {
                check_assign_shape(value, name, escapes, bad_op);
                // Still walk the RHS for *other* references (e.g. `name = other + name`
                // is caught by check_assign_shape as bad_op; still recurse in case the
                // RHS itself contains an escaping sub-expression unrelated to `name`).
                expr_use_name(value, name, escapes, bad_op);
                return;
            }
            // `name` used as (part of) an assignment TARGET other than a bare `Var`
            // match above (e.g. an index target `arr[name] = v`) is an ordinary read;
            // `obj.field = <value containing name>` hands `name`'s value into a
            // struct field — an escape.
            if let ExprKind::Field(_, _) = &target.kind {
                if expr_refs(value, name) { *escapes = true; }
            }
            expr_use_name(target, name, escapes, bad_op);
            expr_use_name(value, name, escapes, bad_op);
        }
        ExprKind::QuestionAssign(target, rhs) => {
            expr_use_name(target, name, escapes, bad_op);
            expr_use_name(rhs, name, escapes, bad_op);
        }
        // `name.swap(v)` — the one recognized method call. Any other method call on
        // `name` (there are none in Boring's builtin scalar surface besides this) or
        // `name` appearing as an ARGUMENT (not receiver) elsewhere is an escape — the
        // callee's body isn't part of this local analysis.
        ExprKind::MethodCall(obj, method, args) => {
            let recv_is_name = matches!(&obj.kind, ExprKind::Var(v) if v == name);
            if recv_is_name {
                if method != "swap" || args.len() != 1 || expr_refs(&args[0].value, name) {
                    *escapes = true;
                }
                for a in args { expr_use_name(&a.value, name, escapes, bad_op); }
            } else {
                expr_use_name(obj, name, escapes, bad_op);
                for a in args {
                    if expr_refs(&a.value, name) { *escapes = true; }
                    expr_use_name(&a.value, name, escapes, bad_op);
                }
            }
        }
        ExprKind::Call(callee, args) | ExprKind::GenericCall(callee, _, args) | ExprKind::Pipe(callee, _, args) => {
            expr_use_name(callee, name, escapes, bad_op);
            // Builtin formatting/logging calls (`print`, `println`, `format`, ...) only
            // ever read their arguments to format output — never store them, hand them
            // to concurrent code, or otherwise let them outlive the call. An ordinary
            // read, same as a bare `Var(name)` elsewhere, not an escape. Any other call
            // (including a Boring-defined function/method) hands `name`'s value to a
            // body this local analysis doesn't open, per criterion 2 — an escape.
            let is_builtin_format_call = matches!(&callee.kind, ExprKind::Var(fn_name)
                if matches!(fn_name.as_str(),
                    "print" | "println" | "eprint" | "eprintln" | "format"
                    | "write" | "error" | "warn" | "info" | "debug" | "trace"));
            for a in args {
                if !is_builtin_format_call && expr_refs(&a.value, name) { *escapes = true; }
                expr_use_name(&a.value, name, escapes, bad_op);
            }
        }
        ExprKind::OptionalMethodCall(obj, _, args) => {
            if expr_refs(obj, name) { *escapes = true; }
            expr_use_name(obj, name, escapes, bad_op);
            for a in args {
                if expr_refs(&a.value, name) { *escapes = true; }
                expr_use_name(&a.value, name, escapes, bad_op);
            }
        }
        // Captured by a task/closure — conservatively always an escape when `name`
        // appears anywhere inside, since the captured value may be handed to
        // genuinely concurrent code this local analysis can't see.
        ExprKind::Task(inner) if expr_refs(inner, name) => { *escapes = true; }
        ExprKind::Task(_) => {}
        ExprKind::TaskWithTimeout(dur, inner) if expr_refs(dur, name) || expr_refs(inner, name) => { *escapes = true; }
        ExprKind::TaskWithTimeout(..) => {}
        // Captured by a closure — conservatively always an escape when `name` appears
        // anywhere inside its body (`collect_var_names` already recurses into a
        // closure's body for us), same rationale as `Task`/`TaskWithTimeout` above.
        ExprKind::Closure(..) if expr_refs(e, name) => { *escapes = true; }
        ExprKind::Closure(..) => {}
        ExprKind::Field(obj, _) => expr_use_name(obj, name, escapes, bad_op),
        ExprKind::Index(obj, idx) => { expr_use_name(obj, name, escapes, bad_op); expr_use_name(idx, name, escapes, bad_op); }
        ExprKind::LabeledIndex(obj, args) => {
            expr_use_name(obj, name, escapes, bad_op);
            for a in args { expr_use_name(&a.value, name, escapes, bad_op); }
        }
        ExprKind::BinOp(_, l, r) => { expr_use_name(l, name, escapes, bad_op); expr_use_name(r, name, escapes, bad_op); }
        ExprKind::UnaryOp(_, inner) => expr_use_name(inner, name, escapes, bad_op),
        ExprKind::If(if_stmt) => {
            for (cond, body) in &if_stmt.branches {
                expr_use_name(cond, name, escapes, bad_op);
                stmts_use_name(body, name, escapes, bad_op);
            }
            if let Some(eb) = &if_stmt.else_body { stmts_use_name(eb, name, escapes, bad_op); }
        }
        ExprKind::TryElse(inner, default) => {
            expr_use_name(inner, name, escapes, bad_op);
            expr_use_name(default, name, escapes, bad_op);
        }
        ExprKind::StringInterp(segs) => {
            for seg in segs {
                match seg {
                    crate::ast::StringSegment::Expr(e2) | crate::ast::StringSegment::FormattedExpr(e2, _) => {
                        expr_use_name(e2, name, escapes, bad_op);
                    }
                    crate::ast::StringSegment::Lit(_) => {}
                }
            }
        }
        ExprKind::New { ctor, .. } => expr_use_name(ctor, name, escapes, bad_op),
        // Leaves and anything else not specifically walked: no sub-expressions of
        // interest (literals, bare `Var` reads not covered above, etc.) — a bare
        // `Var(name)` read reaching here is an ordinary, single `.load()`-mapped
        // access, not flagged.
        _ => {}
    }
}

/// Validates the RHS shape of `name = <value>` against Part 1's operation mapping.
/// Sets `bad_op` (blocking promotion) for anything other than:
/// - a value not referencing `name` at all → plain `store`
/// - `name + <value not referencing name>` / `name - <value not referencing name>`
///   → `fetch_add`/`fetch_sub`, a genuinely single atomic instruction
fn check_assign_shape(value: &Expr, name: &str, _escapes: &mut bool, bad_op: &mut bool) {
    if !expr_refs(value, name) {
        return; // plain store — safe regardless of representation
    }
    if let ExprKind::BinOp(op, l, r) = &value.kind {
        if matches!(op, crate::ast::BinOp::Add | crate::ast::BinOp::Sub)
            && matches!(&l.kind, ExprKind::Var(v) if v == name) && !expr_refs(r, name) {
            return; // fetch_add / fetch_sub — safe
        }
    }
    // Anything else referencing `name` on the RHS (`name * 2`, `f(name)`, `name + name`,
    // ...) would need a separate load then store under the atomic representation —
    // not equivalent to the original lock-protected read-modify-write.
    *bad_op = true;
}
