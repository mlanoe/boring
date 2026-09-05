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

//! Boring-side monomorphization (V1): specializes a generic struct/free-function
//! declaration into a concrete, non-generic clone whenever it is constructed/called
//! through an explicit turbofish (`ExprKind::GenericCall`) whose type arguments are
//! already fully concrete. This coexists with — does not replace — the existing
//! "emit real Rust generics, let rustc monomorphize" path: only the specific call
//! sites this module recognizes get rewritten; every other (inferred, no-turbofish)
//! generic struct/fn use keeps flowing through the unmodified `impl<T: Clone> ...`
//! emission.
//!
//! Scope: turbofish call sites anywhere in the reachable file graph (the entry file
//! plus every file reached transitively via `use`, including `[deps]` cross-project
//! `use`s) are specialized against a generic struct/fn declared in ANY reachable
//! file — not just the same file as the call site. This is done without a third
//! re-parse per file: `deep_pre_scan` (`emit_top.rs`) already visits every reachable
//! file exactly once, so the global generic-decl registry and the raw turbofish
//! call-site candidate list are both folded into that existing walk (see
//! `Transpiler::global_generic_structs`/`global_generic_fns`/
//! `pending_generic_call_candidates`/`global_instantiations`). A turbofish call
//! naming a generic struct/fn that isn't found ANYWHERE in the reachable graph is
//! left untouched and falls through to the existing generic-emission path unchanged.
//!
//! Generic-method extension: a method with its own type param, separate from
//! its enclosing declaration's own (`obj.method<T>(...)`/`obj?.method<T>(...)`),
//! is monomorphized the same way regardless of whether it's declared on a
//! `StructDecl`, an `ExtDecl` (`ext TypeName: def m<U>(...): ...`), or an
//! `EnumDecl` — see `MethodOwnerKind`/`global_generic_methods`. Resolution is
//! name-only and spans all three kinds together: a method name owned by more
//! than one struct/ext/enum anywhere in the reachable graph is ambiguous and
//! safely falls back to ordinary (unspecialized) generic Rust for every call
//! site sharing that name, exactly like the zero-match case.
//!
//! Naming style (mangled specialized names, e.g. `Pair_int_string`) mirrors the
//! existing wgpu kernel const-generic specializer (`transpiler::wgpu::mod`'s
//! `monomorphised_name`/`monomorphise`), which this module is NOT part of and does
//! not touch — GPU targets (wgpu/cuda/metal/rocm) keep their own, separate
//! const-generic-only specialization mechanism untouched.

use super::*;
use std::collections::HashMap;

// ─── Type substitution ─────────────────────────────────────────────────────────

/// Walks the same variant shape as `emit_struct::type_mentions_type_params` (the
/// canonical list of `Type` variants that recurse through a type-parameter
/// reference), returning a new `Type` with every `Type::Named(n)`/`Type::TypeParam(n)`
/// for which `subst` has an entry replaced by the substituted concrete type.
/// Any variant not in that recursing list (primitives, `ConstInt`, `SelfAssoc`, ...)
/// is cloned through unchanged — there is nothing in it that could mention a type
/// parameter.
pub(crate) fn substitute_type(ty: &Type, subst: &HashMap<String, Type>) -> Type {
    match ty {
        Type::Named(n) => subst.get(n).cloned().unwrap_or_else(|| ty.clone()),
        Type::TypeParam(n) => subst.get(n).cloned().unwrap_or_else(|| ty.clone()),
        Type::Optional(inner) => Type::Optional(Box::new(substitute_type(inner, subst))),
        Type::Array(inner) => Type::Array(Box::new(substitute_type(inner, subst))),
        Type::ArrayN(inner, n) => Type::ArrayN(Box::new(substitute_type(inner, subst)), *n),
        Type::ArrayNExpr(inner, ce) => Type::ArrayNExpr(Box::new(substitute_type(inner, subst)), ce.clone()),
        Type::LabeledArray(inner, axes) => Type::LabeledArray(Box::new(substitute_type(inner, subst)), axes.clone()),
        Type::Tuple(elems) => Type::Tuple(elems.iter().map(|t| substitute_type(t, subst)).collect()),
        Type::Dict(k, v) => Type::Dict(Box::new(substitute_type(k, subst)), Box::new(substitute_type(v, subst))),
        Type::Set(inner) => Type::Set(Box::new(substitute_type(inner, subst))),
        Type::Fn(ret, params, throws, task, req) => Type::Fn(
            ret.as_ref().map(|r| Box::new(substitute_type(r, subst))),
            params.iter().map(|t| substitute_type(t, subst)).collect(),
            *throws, *task, *req,
        ),
        Type::Qualified(inner, q) => Type::Qualified(Box::new(substitute_type(inner, subst)), q.clone()),
        Type::Dyn(inner) => Type::Dyn(Box::new(substitute_type(inner, subst))),
        Type::Impl(inner) => Type::Impl(Box::new(substitute_type(inner, subst))),
        Type::Generic(name, args) => Type::Generic(name.clone(), args.iter().map(|t| substitute_type(t, subst)).collect()),
        Type::AssocOf(base, name) => Type::AssocOf(Box::new(substitute_type(base, subst)), name.clone()),
        Type::Mut(inner) => Type::Mut(Box::new(substitute_type(inner, subst))),
        // Int/Uint/.../Str/Bool/Nil/Void/Never/ConstInt/SelfAssoc — no nested Type, no
        // possible type-param reference of their own. Cloned through unchanged.
        other => other.clone(),
    }
}

fn substitute_type_opt(ty: &Option<Type>, subst: &HashMap<String, Type>) -> Option<Type> {
    ty.as_ref().map(|t| substitute_type(t, subst))
}

// ─── AST body walker: substitute every `Type` annotation nested in a body ──────

fn substitute_types_in_param(p: &mut Param, subst: &HashMap<String, Type>) {
    p.ty = substitute_type_opt(&p.ty, subst);
    if let Some(d) = &mut p.default { substitute_types_in_expr(d, subst); }
}

fn substitute_types_in_arg(a: &mut Arg, subst: &HashMap<String, Type>) {
    substitute_types_in_expr(&mut a.value, subst);
}

fn substitute_types_in_args(args: &mut [Arg], subst: &HashMap<String, Type>) {
    for a in args { substitute_types_in_arg(a, subst); }
}

fn substitute_types_in_cond_clause(c: &mut CondClause, subst: &HashMap<String, Type>) {
    match c {
        CondClause::Let(_, e) => substitute_types_in_expr(e, subst),
        CondClause::LetPat(_, e) => substitute_types_in_expr(e, subst),
        CondClause::Expr(e) => substitute_types_in_expr(e, subst),
    }
}

fn substitute_types_in_kernel_config(cfg: &mut KernelConfig, subst: &HashMap<String, Type>) {
    if let Some(e) = &mut cfg.block { substitute_types_in_expr(e, subst); }
    if let Some(e) = &mut cfg.grid { substitute_types_in_expr(e, subst); }
    if let Some(e) = &mut cfg.after { substitute_types_in_expr(e, subst); }
}

pub(crate) fn substitute_types_in_stmts(stmts: &mut [Stmt], subst: &HashMap<String, Type>) {
    for s in stmts { substitute_types_in_stmt(s, subst); }
}

pub(crate) fn substitute_types_in_stmt(stmt: &mut Stmt, subst: &HashMap<String, Type>) {
    match stmt {
        Stmt::Let(l) => {
            l.ty = substitute_type_opt(&l.ty, subst);
            if let Some(v) = &mut l.value { substitute_types_in_expr(v, subst); }
        }
        Stmt::LetDestructure(l) => {
            for b in &mut l.bindings {
                b.ty = substitute_type_opt(&b.ty, subst);
            }
            substitute_types_in_expr(&mut l.value, subst);
        }
        Stmt::Return(r) => { if let Some(v) = &mut r.value { substitute_types_in_expr(v, subst); } }
        Stmt::Break(_, v) => { if let Some(v) = v { substitute_types_in_expr(v, subst); } }
        Stmt::Continue(_) => {}
        Stmt::Throw(t) => { if let Some(v) = &mut t.value { substitute_types_in_expr(v, subst); } }
        Stmt::If(i) => {
            for (cond, body) in &mut i.branches {
                substitute_types_in_expr(cond, subst);
                substitute_types_in_stmts(body, subst);
            }
            if let Some(e) = &mut i.else_body { substitute_types_in_stmts(e, subst); }
        }
        Stmt::IfLet(i) => {
            for c in &mut i.clauses { substitute_types_in_cond_clause(c, subst); }
            substitute_types_in_stmts(&mut i.then_body, subst);
            for branch in &mut i.elif_branches {
                for c in &mut branch.clauses { substitute_types_in_cond_clause(c, subst); }
                substitute_types_in_stmts(&mut branch.body, subst);
            }
            if let Some(e) = &mut i.else_body { substitute_types_in_stmts(e, subst); }
        }
        Stmt::Match(m) => {
            substitute_types_in_expr(&mut m.subject, subst);
            for arm in &mut m.arms {
                if let Some(g) = &mut arm.guard { substitute_types_in_expr(g, subst); }
                match &mut arm.body {
                    MatchBody::Expr(e) => substitute_types_in_expr(e, subst),
                    MatchBody::Block(b) => substitute_types_in_stmts(b, subst),
                }
            }
        }
        Stmt::While(w) => {
            substitute_types_in_expr(&mut w.condition, subst);
            substitute_types_in_stmts(&mut w.body, subst);
        }
        Stmt::WhileLet(w) => {
            substitute_types_in_expr(&mut w.value, subst);
            substitute_types_in_stmts(&mut w.body, subst);
        }
        Stmt::DoWhile(d) => {
            substitute_types_in_stmts(&mut d.body, subst);
            substitute_types_in_expr(&mut d.condition, subst);
        }
        Stmt::Loop(l) => substitute_types_in_stmts(&mut l.body, subst),
        Stmt::Wait(e, _) => substitute_types_in_expr(e, subst),
        Stmt::For(f) => {
            substitute_types_in_expr(&mut f.iterable, subst);
            substitute_types_in_stmts(&mut f.body, subst);
        }
        Stmt::Guard(g) => {
            match &mut g.cond {
                GuardCond::Expr(e) => substitute_types_in_expr(e, subst),
                GuardCond::Clauses(cs) => { for c in cs { substitute_types_in_cond_clause(c, subst); } }
            }
            substitute_types_in_stmts(&mut g.else_body, subst);
        }
        Stmt::Try(t) => {
            substitute_types_in_stmts(&mut t.body, subst);
            for c in &mut t.catch_clauses { substitute_types_in_stmts(&mut c.body, subst); }
        }
        Stmt::Defer(body) => substitute_types_in_stmts(body, subst),
        Stmt::Expr(e) => substitute_types_in_expr(e, subst),
        Stmt::Fn(f) => substitute_types_in_fn_body(f, subst),
        Stmt::Struct(s) => substitute_types_in_struct_body(s, subst),
        Stmt::Enum(e) => substitute_types_in_enum_body(e, subst),
        Stmt::Mod(m) => { for item in &mut m.items { substitute_types_in_item(item, subst); } }
        Stmt::Alias(a) => { a.ty = substitute_type(&a.ty, subst); }
        Stmt::Yield(e, _) => substitute_types_in_expr(e, subst),
        Stmt::Comment(_) => {}
        Stmt::KernelBlock(k) => substitute_types_in_stmts(&mut k.body, subst),
        Stmt::With(w) => substitute_types_in_stmts(&mut w.body, subst),
    }
}

/// Substitutes a nested (non-top-level) `FnDecl`'s own param/return types and body,
/// WITHOUT clearing its `type_params`/renaming it — this is for a `def`/closure-like
/// nested declaration found inside a generic body, not the top-level specialization
/// entry point (see `substitute_fn_decl` for that).
fn substitute_types_in_fn_body(f: &mut FnDecl, subst: &HashMap<String, Type>) {
    for p in &mut f.params { substitute_types_in_param(p, subst); }
    f.return_ty = substitute_type_opt(&f.return_ty, subst);
    f.throws_ty = substitute_type_opt(&f.throws_ty, subst);
    substitute_types_in_stmts(&mut f.body, subst);
    // A nested/method `FnDecl` (e.g. a struct method) may carry its OWN
    // `type_params`, auto-inferred at parse time from a bare single-letter type
    // reference in its signature (`def push(T item)` inside `struct Stack<T>` gets
    // `push.type_params == ["T"]` even though `push` was never written with an
    // explicit `<T>` — see src/parser/parse_fn.rs's "Auto-infer type params from
    // single-uppercase-letter types" comment) — that's really just the method
    // borrowing the ENCLOSING generic struct's own type parameter, not introducing
    // a new one of its own. Once a name has been substituted to a concrete type
    // here (`subst` has an entry for it), the method must stop re-declaring it as
    // its own generic parameter — `emit_top::compute_fn_type_params_str` only
    // suppresses a method's own `type_params` entry when it matches the ENCLOSING
    // `impl<...>` block's type params, which are empty for a specialized
    // (non-generic) struct/enum; left un-filtered, this would emit an invalid
    // stray `fn push<T: Clone>(&mut self, item: isize)` (concrete param type, but a
    // still-generic, now-unconstrained `<T>`) on the specialized clone.
    f.type_params.retain(|p| !subst.contains_key(p));
}

fn substitute_types_in_struct_body(s: &mut StructDecl, subst: &HashMap<String, Type>) {
    for f in &mut s.fields {
        f.ty = substitute_type(&f.ty, subst);
        if let Some(d) = &mut f.default { substitute_types_in_expr(d, subst); }
    }
    for init in &mut s.inits {
        for p in &mut init.params {
            p.ty = substitute_type_opt(&p.ty, subst);
            if let Some(d) = &mut p.default { substitute_types_in_expr(d, subst); }
        }
        substitute_types_in_stmts(&mut init.body, subst);
    }
    for m in &mut s.methods { substitute_types_in_fn_body(m, subst); }
    for conv in &mut s.conversions {
        conv.ty = substitute_type(&conv.ty, subst);
        substitute_types_in_stmts(&mut conv.body, subst);
    }
    for set in &mut s.setters {
        set.param_ty = substitute_type(&set.param_ty, subst);
        substitute_types_in_stmts(&mut set.body, subst);
    }
    for tm in &mut s.type_methods {
        for p in &mut tm.params { substitute_types_in_param(p, subst); }
        tm.return_ty = substitute_type_opt(&tm.return_ty, subst);
        tm.throws_ty = substitute_type_opt(&tm.throws_ty, subst);
        substitute_types_in_stmts(&mut tm.body, subst);
    }
    for tv in &mut s.type_vars {
        tv.ty = substitute_type_opt(&tv.ty, subst);
        substitute_types_in_expr(&mut tv.default, subst);
    }
}

fn substitute_types_in_enum_body(e: &mut EnumDecl, subst: &HashMap<String, Type>) {
    for v in &mut e.variants {
        for f in &mut v.fields { f.ty = substitute_type(&f.ty, subst); }
    }
    for m in &mut e.methods { substitute_types_in_fn_body(m, subst); }
    for conv in &mut e.conversions {
        conv.ty = substitute_type(&conv.ty, subst);
        substitute_types_in_stmts(&mut conv.body, subst);
    }
    for set in &mut e.setters {
        set.param_ty = substitute_type(&set.param_ty, subst);
        substitute_types_in_stmts(&mut set.body, subst);
    }
}

fn substitute_types_in_item(item: &mut Item, subst: &HashMap<String, Type>) {
    match item {
        Item::Fn(f) => substitute_types_in_fn_body(f, subst),
        Item::Struct(s) => substitute_types_in_struct_body(s, subst),
        Item::Enum(e) => substitute_types_in_enum_body(e, subst),
        Item::Ext(ext) => {
            for m in &mut ext.methods { substitute_types_in_fn_body(m, subst); }
            for conv in &mut ext.conversions {
                conv.ty = substitute_type(&conv.ty, subst);
                substitute_types_in_stmts(&mut conv.body, subst);
            }
            for set in &mut ext.setters {
                set.param_ty = substitute_type(&set.param_ty, subst);
                substitute_types_in_stmts(&mut set.body, subst);
            }
        }
        Item::Mod(m) => { for it in &mut m.items { substitute_types_in_item(it, subst); } }
        Item::Let(l) => { if let Some(v) = &mut l.value { substitute_types_in_expr(v, subst); } }
        Item::Alias(a) => { a.ty = substitute_type(&a.ty, subst); }
        Item::Stmt(s) => substitute_types_in_stmt(s, subst),
        Item::Use(_) | Item::Trait(_) | Item::Kernel(_) => {}
    }
}

fn substitute_types_in_string_segments(segs: &mut [StringSegment], subst: &HashMap<String, Type>) {
    for seg in segs {
        match seg {
            StringSegment::Lit(_) => {}
            StringSegment::Expr(e) => substitute_types_in_expr(e, subst),
            StringSegment::FormattedExpr(e, _) => substitute_types_in_expr(e, subst),
        }
    }
}

pub(crate) fn substitute_types_in_expr(expr: &mut Expr, subst: &HashMap<String, Type>) {
    match &mut expr.kind {
        ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Str(_)
            | ExprKind::Bool(_) | ExprKind::Nil | ExprKind::Void | ExprKind::Var(_)
            | ExprKind::DotIdent(_) => {}
        ExprKind::StringInterp(segs) => substitute_types_in_string_segments(segs, subst),
        ExprKind::BinOp(_, l, r) => { substitute_types_in_expr(l, subst); substitute_types_in_expr(r, subst); }
        ExprKind::UnaryOp(_, e) => substitute_types_in_expr(e, subst),
        ExprKind::Assign(l, r) => { substitute_types_in_expr(l, subst); substitute_types_in_expr(r, subst); }
        ExprKind::QuestionAssign(l, r) => { substitute_types_in_expr(l, subst); substitute_types_in_expr(r, subst); }
        ExprKind::Field(e, _) => substitute_types_in_expr(e, subst),
        ExprKind::Index(e, i) => { substitute_types_in_expr(e, subst); substitute_types_in_expr(i, subst); }
        ExprKind::LabeledIndex(e, args) => { substitute_types_in_expr(e, subst); substitute_types_in_args(args, subst); }
        ExprKind::Call(callee, args) => { substitute_types_in_expr(callee, subst); substitute_types_in_args(args, subst); }
        ExprKind::MethodCall(obj, _, args) => { substitute_types_in_expr(obj, subst); substitute_types_in_args(args, subst); }
        ExprKind::GenericCall(callee, type_args, args) => {
            substitute_types_in_expr(callee, subst);
            for t in type_args.iter_mut() { *t = substitute_type(t, subst); }
            substitute_types_in_args(args, subst);
        }
        ExprKind::Pipe(lhs, _, args) => { substitute_types_in_expr(lhs, subst); substitute_types_in_args(args, subst); }
        ExprKind::New { arena, ctor } => {
            if let Some(a) = arena { substitute_types_in_expr(a, subst); }
            substitute_types_in_expr(ctor, subst);
        }
        ExprKind::KernelLaunch { config, kernel } => {
            substitute_types_in_kernel_config(config, subst);
            substitute_types_in_expr(kernel, subst);
        }
        ExprKind::TryElse(a, b) => { substitute_types_in_expr(a, subst); substitute_types_in_expr(b, subst); }
        ExprKind::TryElseBlock(body, else_body) => {
            substitute_types_in_stmts(body, subst);
            substitute_types_in_stmts(else_body, subst);
        }
        ExprKind::Array(items) => { for e in items { substitute_types_in_expr(e, subst); } }
        ExprKind::ArrayFill { value, count } => { substitute_types_in_expr(value, subst); substitute_types_in_expr(count, subst); }
        ExprKind::ArrayAlloc { count } => substitute_types_in_expr(count, subst),
        ExprKind::ArrayComp { expr: e, var: _, count } => { substitute_types_in_expr(e, subst); substitute_types_in_expr(count, subst); }
        ExprKind::ArrayCompIter { expr: e, var: _, iter } => { substitute_types_in_expr(e, subst); substitute_types_in_expr(iter, subst); }
        ExprKind::LabeledArrayComp { expr: e, clauses } => {
            substitute_types_in_expr(e, subst);
            for (_, c) in clauses { substitute_types_in_expr(c, subst); }
        }
        ExprKind::Tuple(items) => { for e in items { substitute_types_in_expr(e, subst); } }
        ExprKind::Dict(pairs) => { for (k, v) in pairs { substitute_types_in_expr(k, subst); substitute_types_in_expr(v, subst); } }
        ExprKind::Set(items) => { for e in items { substitute_types_in_expr(e, subst); } }
        ExprKind::Range { start, end, inclusive: _ } => { substitute_types_in_expr(start, subst); substitute_types_in_expr(end, subst); }
        ExprKind::SliceRange { start, end, inclusive: _ } => {
            if let Some(s) = start { substitute_types_in_expr(s, subst); }
            if let Some(e) = end { substitute_types_in_expr(e, subst); }
        }
        ExprKind::Cast(e, ty) => { substitute_types_in_expr(e, subst); *ty = substitute_type(ty, subst); }
        ExprKind::RelabelCast(e, _) => substitute_types_in_expr(e, subst),
        ExprKind::Else(a, b) => { substitute_types_in_expr(a, subst); substitute_types_in_expr(b, subst); }
        ExprKind::OptionalField(e, _) => substitute_types_in_expr(e, subst),
        ExprKind::OptionalMethodCall(obj, _, args) => { substitute_types_in_expr(obj, subst); substitute_types_in_args(args, subst); }
        ExprKind::Closure(params, ret, body, _, _) => {
            for p in params { substitute_types_in_param(p, subst); }
            *ret = substitute_type_opt(ret, subst);
            match body {
                ClosureBody::Expr(e) => substitute_types_in_expr(e, subst),
                ClosureBody::Block(b) => substitute_types_in_stmts(b, subst),
            }
        }
        ExprKind::If(if_stmt) => {
            for (cond, body) in &mut if_stmt.branches {
                substitute_types_in_expr(cond, subst);
                substitute_types_in_stmts(body, subst);
            }
            if let Some(e) = &mut if_stmt.else_body { substitute_types_in_stmts(e, subst); }
        }
        ExprKind::Match(match_stmt) => {
            substitute_types_in_expr(&mut match_stmt.subject, subst);
            for arm in &mut match_stmt.arms {
                if let Some(g) = &mut arm.guard { substitute_types_in_expr(g, subst); }
                match &mut arm.body {
                    MatchBody::Expr(e) => substitute_types_in_expr(e, subst),
                    MatchBody::Block(b) => substitute_types_in_stmts(b, subst),
                }
            }
        }
        ExprKind::Block(stmts) => substitute_types_in_stmts(stmts, subst),
        ExprKind::Do(stmts) => substitute_types_in_stmts(stmts, subst),
        ExprKind::Loop(l) => substitute_types_in_stmts(&mut l.body, subst),
        ExprKind::Task(e) => substitute_types_in_expr(e, subst),
        ExprKind::TaskWithTimeout(a, b) => { substitute_types_in_expr(a, subst); substitute_types_in_expr(b, subst); }
        ExprKind::JoinAll(items) => { for e in items { substitute_types_in_expr(e, subst); } }
        ExprKind::MacroCall { name: _, args } => { for e in args { substitute_types_in_expr(e, subst); } }
    }
}

// ─── Specialized-clone construction ────────────────────────────────────────────

/// Non-const-generic (ordinary type) parameter names out of a `type_params` list —
/// const-generic params are encoded as `"$name:ty"` (see wgpu's `build_subst`) and
/// are intentionally left alone here: this feature only ever substitutes ordinary
/// type parameters, never wgpu's separate const-generic concept.
fn ordinary_type_param_names(type_params: &[String]) -> Vec<&str> {
    type_params.iter()
        .filter(|p| !p.starts_with('$'))
        .map(|p| p.as_str())
        .collect()
}

/// Builds `decl.type_params`(ordinary only) → `type_args` (positionally zipped,
/// skipping any `$`-prefixed const-generic param) substitution map.
fn build_type_subst(type_params: &[String], type_args: &[Type]) -> HashMap<String, Type> {
    let mut subst = HashMap::new();
    let mut arg_iter = type_args.iter();
    for p in type_params {
        if p.starts_with('$') { continue; }
        if let Some(arg) = arg_iter.next() {
            subst.insert(p.clone(), arg.clone());
        }
    }
    subst
}

/// Clone `decl`, rename to `new_name`, clear `type_params`/`where_clause`, and
/// substitute every nested `Type` (fields, methods, inits, conversions, setters,
/// type-level members) per `subst`.
pub(crate) fn substitute_struct_decl(decl: &StructDecl, subst: &HashMap<String, Type>, new_name: &str) -> StructDecl {
    let mut out = decl.clone();
    out.name = new_name.to_string();
    out.type_params = Vec::new();
    out.where_clause = Vec::new();
    substitute_types_in_struct_body(&mut out, subst);
    out
}

/// Clone `decl`, rename to `new_name`, clear `type_params`/`where_clause`, and
/// substitute every nested `Type` (params, return type, throws type, body).
pub(crate) fn substitute_fn_decl(decl: &FnDecl, subst: &HashMap<String, Type>, new_name: &str) -> FnDecl {
    let mut out = decl.clone();
    out.name = new_name.to_string();
    out.type_params = Vec::new();
    out.where_clause = Vec::new();
    substitute_types_in_fn_body(&mut out, subst);
    out
}

// ─── Mangled naming ─────────────────────────────────────────────────────────────

/// Returns a Rust-identifier-safe name fragment for `ty`, or `None` when `ty` isn't
/// one of the shapes this V1 mangler confidently knows how to name (nested
/// generics-of-generics, tuples, dicts, function types, ...) — callers must bail
/// out of specializing the whole instantiation rather than emit a colliding/unsafe
/// partial name.
pub(crate) fn mangle_type(ty: &Type) -> Option<String> {
    match ty {
        Type::Int => Some("int".to_string()),
        Type::Uint => Some("uint".to_string()),
        Type::Uint8 => Some("uint8".to_string()),
        Type::Int8 => Some("int8".to_string()),
        Type::Int16 => Some("int16".to_string()),
        Type::Int32 => Some("int32".to_string()),
        Type::Int64 => Some("int64".to_string()),
        Type::Int128 => Some("int128".to_string()),
        Type::Uint16 => Some("uint16".to_string()),
        Type::Uint32 => Some("uint32".to_string()),
        Type::Uint64 => Some("uint64".to_string()),
        Type::Uint128 => Some("uint128".to_string()),
        Type::Float32 => Some("float32".to_string()),
        Type::Float64 => Some("float".to_string()),
        Type::Str => Some("string".to_string()),
        Type::Bool => Some("bool".to_string()),
        Type::Named(n) => match n.as_str() {
            "int" | "uint" | "float" | "float32" | "float64" | "bool" | "string" => Some(n.clone()),
            _ => Some(n.clone()),
        },
        Type::Optional(inner) => mangle_type(inner).map(|s| format!("opt_{}", s)),
        Type::Array(inner) => mangle_type(inner).map(|s| format!("arr_{}", s)),
        // `mut Type` — a Boring-only permission wrapper with no distinct Rust type
        // (see `Type::Mut`'s doc); mangles as its inner type, prefixed, so
        // `Container<mut Point>` and `Container<Point>` still get distinct names
        // (the specialized copy is otherwise identical either way in V1 — see
        // docs/book.md's known gap this module doesn't yet close, "mut Type" section
        // in the final report).
        Type::Mut(inner) => mangle_type(inner).map(|s| format!("mut_{}", s)),
        _ => None,
    }
}

/// Composes `mangle_type` over every type argument into a single specialized name,
/// e.g. `Pair` + `[int, string]` → `Some("Pair_int_string")`. Bails (`None`) if any
/// argument itself doesn't mangle — no partial/fallback name is ever produced.
pub(crate) fn mangled_name(base: &str, args: &[Type]) -> Option<String> {
    let mut parts = Vec::with_capacity(args.len());
    for a in args {
        parts.push(mangle_type(a)?);
    }
    if parts.is_empty() {
        return None;
    }
    Some(format!("{}_{}", base, parts.join("_")))
}

// ─── Instantiation collection ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub(crate) struct Instantiation {
    pub base_name: String,
    pub type_args: Vec<Type>,
    pub mangled: String,
    pub is_struct: bool,
}

/// Keyed by `base_name`, deduplicated by mangled name (the practical equality key —
/// two distinct `Type`s that mangle to the same string would already be a mangler
/// bug, so this is robust without requiring a full structural `Type` comparison).
pub(crate) type InstantiationMap = HashMap<String, Vec<Instantiation>>;

fn instantiation_map_contains(map: &InstantiationMap, base_name: &str, mangled: &str) -> bool {
    map.get(base_name).map(|v| v.iter().any(|i| i.mangled == mangled)).unwrap_or(false)
}

fn instantiation_map_insert(map: &mut InstantiationMap, inst: Instantiation) {
    if instantiation_map_contains(map, &inst.base_name, &inst.mangled) { return; }
    map.entry(inst.base_name.clone()).or_default().push(inst);
}

/// True when `ty` contains no unresolved type parameter — neither a bare
/// `Type::TypeParam`, nor a `Type::Named(n)` where `n` is itself one of the
/// currently-in-scope generic type parameter names of whatever generic
/// function/struct the call site is lexically inside (`enclosing_type_params`).
fn is_concrete_type(ty: &Type, enclosing_type_params: &[String]) -> bool {
    match ty {
        Type::TypeParam(_) => false,
        Type::Named(n) => !enclosing_type_params.iter().any(|p| p == n),
        Type::Optional(inner) | Type::Array(inner) | Type::ArrayN(inner, _)
            | Type::ArrayNExpr(inner, _) | Type::Set(inner) | Type::Qualified(inner, _)
            | Type::Dyn(inner) | Type::Impl(inner) | Type::Mut(inner) =>
            is_concrete_type(inner, enclosing_type_params),
        Type::Dict(k, v) => is_concrete_type(k, enclosing_type_params) && is_concrete_type(v, enclosing_type_params),
        Type::Tuple(elems) => elems.iter().all(|t| is_concrete_type(t, enclosing_type_params)),
        Type::Generic(_, args) => args.iter().all(|t| is_concrete_type(t, enclosing_type_params)),
        Type::AssocOf(base, _) => is_concrete_type(base, enclosing_type_params),
        Type::Fn(ret, params, ..) =>
            ret.as_ref().map(|r| is_concrete_type(r, enclosing_type_params)).unwrap_or(true)
                && params.iter().all(|t| is_concrete_type(t, enclosing_type_params)),
        _ => true,
    }
}

/// A raw, unfiltered turbofish call-site sighting — recorded regardless of whether
/// `name` is currently known to resolve to a generic struct/fn (that isn't decided
/// until every reachable file has been visited and the global decl registries are
/// complete; see this module's top-level doc comment). `enclosing_type_params` is
/// the stack of type-parameter names lexically in scope at this call site (from any
/// generic function/struct body it's nested inside) — needed later to tell a
/// turbofish argument that is still an outer generic context's own type parameter
/// (not concrete, must not be specialized) apart from one that has become fully
/// concrete inside a (potentially nested) monomorphic caller.
#[derive(Debug, Clone)]
pub(crate) struct CandidateCall {
    pub name: String,
    pub type_args: Vec<Type>,
    pub enclosing_type_params: Vec<String>,
}

/// Same as `CandidateCall`, for a turbofish METHOD call site
/// (`obj.method<T>(...)`, parsed as `GenericCall(Field(receiver, method), ...)`).
/// The receiver expression itself is deliberately not recorded here — resolution
/// never tracks the receiver's resolved struct type (see this module's top-level
/// doc comment and `Transpiler::global_generic_methods`); only the bare method
/// name and type args matter for the lookup. The receiver stays available at the
/// AST node itself for the later rewrite pass, which doesn't need this struct.
#[derive(Debug, Clone)]
pub(crate) struct MethodCandidateCall {
    pub method_name: String,
    pub type_args: Vec<Type>,
    pub enclosing_type_params: Vec<String>,
}

/// Both flavors of raw candidate found by one walk of `collect_candidate_calls`.
#[derive(Debug, Clone, Default)]
pub(crate) struct CollectedCandidates {
    pub free: Vec<CandidateCall>,
    pub methods: Vec<MethodCandidateCall>,
}

struct CollectCtx {
    candidates: Vec<CandidateCall>,
    method_candidates: Vec<MethodCandidateCall>,
}

/// Walks EVERY function/method body reachable from `program` (top-level `Item::Fn`,
/// every `StructDecl`/`EnumDecl` method, `ExtDecl` methods, top-level `Item::Stmt`,
/// nested `Item::Mod`) recursively through every `Stmt`/`Expr`, threading a stack of
/// "current enclosing type_params" as it descends into generic function/struct
/// bodies, and records every turbofish (`ExprKind::GenericCall`) call site found as
/// a raw `CandidateCall` — unconditionally, with no filtering against any known
/// generic-decl registry (that happens later, once, in `build_instantiation_map`,
/// after every reachable file has contributed its own candidates via
/// `deep_pre_scan`).
pub(crate) fn collect_candidate_calls(program: &Program) -> CollectedCandidates {
    let mut ctx = CollectCtx { candidates: Vec::new(), method_candidates: Vec::new() };
    let empty: Vec<String> = Vec::new();
    for item in &program.items {
        collect_in_item(item, &empty, &mut ctx);
    }
    CollectedCandidates { free: ctx.candidates, methods: ctx.method_candidates }
}

fn collect_in_item(item: &Item, enclosing: &[String], ctx: &mut CollectCtx) {
    match item {
        Item::Fn(f) => collect_in_fn(f, enclosing, ctx),
        Item::Struct(s) => collect_in_struct(s, enclosing, ctx),
        Item::Enum(e) => collect_in_enum(e, enclosing, ctx),
        Item::Ext(ext) => {
            for m in &ext.methods { collect_in_fn(m, enclosing, ctx); }
            for conv in &ext.conversions { collect_in_stmts(&conv.body, enclosing, ctx); }
            for set in &ext.setters { collect_in_stmts(&set.body, enclosing, ctx); }
        }
        Item::Mod(m) => { for it in &m.items { collect_in_item(it, enclosing, ctx); } }
        Item::Let(l) => { if let Some(v) = &l.value { collect_in_expr(v, enclosing, ctx); } }
        Item::Stmt(s) => collect_in_stmt(s, enclosing, ctx),
        Item::Use(_) | Item::Trait(_) | Item::Kernel(_) | Item::Alias(_) => {}
    }
}

fn collect_in_fn(f: &FnDecl, enclosing: &[String], ctx: &mut CollectCtx) {
    let inner: Vec<String>;
    let scope: &[String] = if f.type_params.is_empty() {
        enclosing
    } else {
        inner = enclosing.iter().cloned().chain(f.type_params.iter().cloned()).collect();
        &inner
    };
    collect_in_stmts(&f.body, scope, ctx);
}

fn collect_in_struct(s: &StructDecl, enclosing: &[String], ctx: &mut CollectCtx) {
    let inner: Vec<String>;
    let scope: &[String] = if s.type_params.is_empty() {
        enclosing
    } else {
        inner = enclosing.iter().cloned().chain(s.type_params.iter().cloned()).collect();
        &inner
    };
    for m in &s.methods { collect_in_fn(m, scope, ctx); }
    for init in &s.inits { collect_in_stmts(&init.body, scope, ctx); }
    for conv in &s.conversions { collect_in_stmts(&conv.body, scope, ctx); }
    for set in &s.setters { collect_in_stmts(&set.body, scope, ctx); }
    for tm in &s.type_methods { collect_in_stmts(&tm.body, scope, ctx); }
    for tv in &s.type_vars { collect_in_expr(&tv.default, scope, ctx); }
}

fn collect_in_enum(e: &EnumDecl, enclosing: &[String], ctx: &mut CollectCtx) {
    let inner: Vec<String>;
    let scope: &[String] = if e.type_params.is_empty() {
        enclosing
    } else {
        inner = enclosing.iter().cloned().chain(e.type_params.iter().cloned()).collect();
        &inner
    };
    for m in &e.methods { collect_in_fn(m, scope, ctx); }
    for conv in &e.conversions { collect_in_stmts(&conv.body, scope, ctx); }
    for set in &e.setters { collect_in_stmts(&set.body, scope, ctx); }
}

fn collect_in_stmts(stmts: &[Stmt], enclosing: &[String], ctx: &mut CollectCtx) {
    for s in stmts { collect_in_stmt(s, enclosing, ctx); }
}

fn collect_in_cond_clause(c: &CondClause, enclosing: &[String], ctx: &mut CollectCtx) {
    match c {
        CondClause::Let(_, e) | CondClause::LetPat(_, e) | CondClause::Expr(e) => collect_in_expr(e, enclosing, ctx),
    }
}

fn collect_in_stmt(stmt: &Stmt, enclosing: &[String], ctx: &mut CollectCtx) {
    match stmt {
        Stmt::Let(l) => { if let Some(v) = &l.value { collect_in_expr(v, enclosing, ctx); } }
        Stmt::LetDestructure(l) => collect_in_expr(&l.value, enclosing, ctx),
        Stmt::Return(r) => { if let Some(v) = &r.value { collect_in_expr(v, enclosing, ctx); } }
        Stmt::Break(_, v) => { if let Some(v) = v { collect_in_expr(v, enclosing, ctx); } }
        Stmt::Continue(_) => {}
        Stmt::Throw(t) => { if let Some(v) = &t.value { collect_in_expr(v, enclosing, ctx); } }
        Stmt::If(i) => {
            for (cond, body) in &i.branches { collect_in_expr(cond, enclosing, ctx); collect_in_stmts(body, enclosing, ctx); }
            if let Some(e) = &i.else_body { collect_in_stmts(e, enclosing, ctx); }
        }
        Stmt::IfLet(i) => {
            for c in &i.clauses { collect_in_cond_clause(c, enclosing, ctx); }
            collect_in_stmts(&i.then_body, enclosing, ctx);
            for branch in &i.elif_branches {
                for c in &branch.clauses { collect_in_cond_clause(c, enclosing, ctx); }
                collect_in_stmts(&branch.body, enclosing, ctx);
            }
            if let Some(e) = &i.else_body { collect_in_stmts(e, enclosing, ctx); }
        }
        Stmt::Match(m) => {
            collect_in_expr(&m.subject, enclosing, ctx);
            for arm in &m.arms {
                if let Some(g) = &arm.guard { collect_in_expr(g, enclosing, ctx); }
                match &arm.body {
                    MatchBody::Expr(e) => collect_in_expr(e, enclosing, ctx),
                    MatchBody::Block(b) => collect_in_stmts(b, enclosing, ctx),
                }
            }
        }
        Stmt::While(w) => { collect_in_expr(&w.condition, enclosing, ctx); collect_in_stmts(&w.body, enclosing, ctx); }
        Stmt::WhileLet(w) => { collect_in_expr(&w.value, enclosing, ctx); collect_in_stmts(&w.body, enclosing, ctx); }
        Stmt::DoWhile(d) => { collect_in_stmts(&d.body, enclosing, ctx); collect_in_expr(&d.condition, enclosing, ctx); }
        Stmt::Loop(l) => collect_in_stmts(&l.body, enclosing, ctx),
        Stmt::Wait(e, _) => collect_in_expr(e, enclosing, ctx),
        Stmt::For(f) => { collect_in_expr(&f.iterable, enclosing, ctx); collect_in_stmts(&f.body, enclosing, ctx); }
        Stmt::Guard(g) => {
            match &g.cond {
                GuardCond::Expr(e) => collect_in_expr(e, enclosing, ctx),
                GuardCond::Clauses(cs) => { for c in cs { collect_in_cond_clause(c, enclosing, ctx); } }
            }
            collect_in_stmts(&g.else_body, enclosing, ctx);
        }
        Stmt::Try(t) => {
            collect_in_stmts(&t.body, enclosing, ctx);
            for c in &t.catch_clauses { collect_in_stmts(&c.body, enclosing, ctx); }
        }
        Stmt::Defer(body) => collect_in_stmts(body, enclosing, ctx),
        Stmt::Expr(e) => collect_in_expr(e, enclosing, ctx),
        Stmt::Fn(f) => collect_in_fn(f, enclosing, ctx),
        Stmt::Struct(s) => collect_in_struct(s, enclosing, ctx),
        Stmt::Enum(e) => collect_in_enum(e, enclosing, ctx),
        Stmt::Mod(m) => { for it in &m.items { collect_in_item(it, enclosing, ctx); } }
        Stmt::Alias(_) => {}
        Stmt::Yield(e, _) => collect_in_expr(e, enclosing, ctx),
        Stmt::Comment(_) => {}
        Stmt::KernelBlock(k) => collect_in_stmts(&k.body, enclosing, ctx),
        Stmt::With(w) => collect_in_stmts(&w.body, enclosing, ctx),
    }
}

fn collect_in_args(args: &[Arg], enclosing: &[String], ctx: &mut CollectCtx) {
    for a in args { collect_in_expr(&a.value, enclosing, ctx); }
}

fn collect_in_expr(expr: &Expr, enclosing: &[String], ctx: &mut CollectCtx) {
    match &expr.kind {
        ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Str(_)
            | ExprKind::Bool(_) | ExprKind::Nil | ExprKind::Void | ExprKind::Var(_)
            | ExprKind::DotIdent(_) => {}
        ExprKind::StringInterp(segs) => {
            for seg in segs {
                match seg {
                    StringSegment::Lit(_) => {}
                    StringSegment::Expr(e) | StringSegment::FormattedExpr(e, _) => collect_in_expr(e, enclosing, ctx),
                }
            }
        }
        ExprKind::BinOp(_, l, r) => { collect_in_expr(l, enclosing, ctx); collect_in_expr(r, enclosing, ctx); }
        ExprKind::UnaryOp(_, e) => collect_in_expr(e, enclosing, ctx),
        ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r) => { collect_in_expr(l, enclosing, ctx); collect_in_expr(r, enclosing, ctx); }
        ExprKind::Field(e, _) => collect_in_expr(e, enclosing, ctx),
        ExprKind::Index(e, i) => { collect_in_expr(e, enclosing, ctx); collect_in_expr(i, enclosing, ctx); }
        ExprKind::LabeledIndex(e, args) => { collect_in_expr(e, enclosing, ctx); collect_in_args(args, enclosing, ctx); }
        ExprKind::Call(callee, args) => { collect_in_expr(callee, enclosing, ctx); collect_in_args(args, enclosing, ctx); }
        ExprKind::MethodCall(obj, _, args) => { collect_in_expr(obj, enclosing, ctx); collect_in_args(args, enclosing, ctx); }
        ExprKind::GenericCall(callee, type_args, args) => {
            collect_in_expr(callee, enclosing, ctx);
            collect_in_args(args, enclosing, ctx);
            record_candidate(callee, type_args, enclosing, ctx);
        }
        ExprKind::Pipe(lhs, _, args) => { collect_in_expr(lhs, enclosing, ctx); collect_in_args(args, enclosing, ctx); }
        ExprKind::New { arena, ctor } => {
            if let Some(a) = arena { collect_in_expr(a, enclosing, ctx); }
            collect_in_expr(ctor, enclosing, ctx);
        }
        ExprKind::KernelLaunch { config, kernel } => {
            if let Some(e) = &config.block { collect_in_expr(e, enclosing, ctx); }
            if let Some(e) = &config.grid { collect_in_expr(e, enclosing, ctx); }
            if let Some(e) = &config.after { collect_in_expr(e, enclosing, ctx); }
            collect_in_expr(kernel, enclosing, ctx);
        }
        ExprKind::TryElse(a, b) => { collect_in_expr(a, enclosing, ctx); collect_in_expr(b, enclosing, ctx); }
        ExprKind::TryElseBlock(body, else_body) => {
            collect_in_stmts(body, enclosing, ctx);
            collect_in_stmts(else_body, enclosing, ctx);
        }
        ExprKind::Array(items) | ExprKind::Tuple(items) | ExprKind::Set(items) => {
            for e in items { collect_in_expr(e, enclosing, ctx); }
        }
        ExprKind::ArrayFill { value, count } => { collect_in_expr(value, enclosing, ctx); collect_in_expr(count, enclosing, ctx); }
        ExprKind::ArrayAlloc { count } => collect_in_expr(count, enclosing, ctx),
        ExprKind::ArrayComp { expr: e, var: _, count } => { collect_in_expr(e, enclosing, ctx); collect_in_expr(count, enclosing, ctx); }
        ExprKind::ArrayCompIter { expr: e, var: _, iter } => { collect_in_expr(e, enclosing, ctx); collect_in_expr(iter, enclosing, ctx); }
        ExprKind::LabeledArrayComp { expr: e, clauses } => {
            collect_in_expr(e, enclosing, ctx);
            for (_, c) in clauses { collect_in_expr(c, enclosing, ctx); }
        }
        ExprKind::Dict(pairs) => { for (k, v) in pairs { collect_in_expr(k, enclosing, ctx); collect_in_expr(v, enclosing, ctx); } }
        ExprKind::Range { start, end, .. } => { collect_in_expr(start, enclosing, ctx); collect_in_expr(end, enclosing, ctx); }
        ExprKind::SliceRange { start, end, .. } => {
            if let Some(s) = start { collect_in_expr(s, enclosing, ctx); }
            if let Some(e) = end { collect_in_expr(e, enclosing, ctx); }
        }
        ExprKind::Cast(e, _) => collect_in_expr(e, enclosing, ctx),
        ExprKind::RelabelCast(e, _) => collect_in_expr(e, enclosing, ctx),
        ExprKind::Else(a, b) => { collect_in_expr(a, enclosing, ctx); collect_in_expr(b, enclosing, ctx); }
        ExprKind::OptionalField(e, _) => collect_in_expr(e, enclosing, ctx),
        ExprKind::OptionalMethodCall(obj, _, args) => { collect_in_expr(obj, enclosing, ctx); collect_in_args(args, enclosing, ctx); }
        ExprKind::Closure(_, _, body, _, _) => match body {
            ClosureBody::Expr(e) => collect_in_expr(e, enclosing, ctx),
            ClosureBody::Block(b) => collect_in_stmts(b, enclosing, ctx),
        },
        ExprKind::If(if_stmt) => {
            for (cond, body) in &if_stmt.branches { collect_in_expr(cond, enclosing, ctx); collect_in_stmts(body, enclosing, ctx); }
            if let Some(e) = &if_stmt.else_body { collect_in_stmts(e, enclosing, ctx); }
        }
        ExprKind::Match(match_stmt) => {
            collect_in_expr(&match_stmt.subject, enclosing, ctx);
            for arm in &match_stmt.arms {
                if let Some(g) = &arm.guard { collect_in_expr(g, enclosing, ctx); }
                match &arm.body {
                    MatchBody::Expr(e) => collect_in_expr(e, enclosing, ctx),
                    MatchBody::Block(b) => collect_in_stmts(b, enclosing, ctx),
                }
            }
        }
        ExprKind::Block(stmts) | ExprKind::Do(stmts) => collect_in_stmts(stmts, enclosing, ctx),
        ExprKind::Loop(l) => collect_in_stmts(&l.body, enclosing, ctx),
        ExprKind::Task(e) => collect_in_expr(e, enclosing, ctx),
        ExprKind::TaskWithTimeout(a, b) => { collect_in_expr(a, enclosing, ctx); collect_in_expr(b, enclosing, ctx); }
        ExprKind::JoinAll(items) => { for e in items { collect_in_expr(e, enclosing, ctx); } }
        ExprKind::MacroCall { name: _, args } => { for e in args { collect_in_expr(e, enclosing, ctx); } }
    }
}

fn record_candidate(callee: &Expr, type_args: &[Type], enclosing: &[String], ctx: &mut CollectCtx) {
    if type_args.is_empty() { return; }
    match &callee.kind {
        ExprKind::Var(name) => {
            ctx.candidates.push(CandidateCall {
                name: name.clone(),
                type_args: type_args.to_vec(),
                enclosing_type_params: enclosing.to_vec(),
            });
        }
        // `obj.method<T>(...)` — generic method call (V1 extension). The receiver
        // isn't recorded (see `MethodCandidateCall`'s doc comment); only the bare
        // method name and type args matter for resolution.
        //
        // `obj?.method<T>(...)` — optional-chained generic method call. Same
        // registry, same name-only resolution: an `OptionalField` callee is
        // recorded as a method candidate exactly like a plain `Field` callee.
        ExprKind::Field(_, method) | ExprKind::OptionalField(_, method) => {
            ctx.method_candidates.push(MethodCandidateCall {
                method_name: method.clone(),
                type_args: type_args.to_vec(),
                enclosing_type_params: enclosing.to_vec(),
            });
        }
        _ => {}
    }
}

/// Filters `candidates` (gathered from every reachable file by `collect_candidate_calls`,
/// folded into `deep_pre_scan`'s walk) against the now-complete
/// `generic_structs`/`generic_fns` registries (also global, populated across the
/// same reachable file graph) to build the final, deduped, mangled instantiation
/// map. This is the same concreteness-check + mangling + dedup logic the old
/// same-file-only `try_record_instantiation` used to apply inline during the AST
/// walk — now applied as a separate pass over the flat candidate list instead,
/// since the walk (gathering) and the filtering (needs every file's decls to be
/// known first) can no longer happen in the same single-file pass once a candidate
/// may reference a decl from another file.
pub(crate) fn build_instantiation_map(
    candidates: &[CandidateCall],
    generic_structs: &HashMap<String, StructDecl>,
    generic_fns: &HashMap<String, FnDecl>,
) -> InstantiationMap {
    let mut map = InstantiationMap::new();
    for cand in candidates {
        let (is_struct, type_params) = if let Some(s) = generic_structs.get(cand.name.as_str()) {
            (true, &s.type_params)
        } else if let Some(f) = generic_fns.get(cand.name.as_str()) {
            (false, &f.type_params)
        } else {
            continue;
        };
        if !cand.type_args.iter().all(|t| is_concrete_type(t, &cand.enclosing_type_params)) { continue; }
        let Some(mangled) = mangled_name(&cand.name, &cand.type_args) else { continue; };
        // Sanity: the number of ordinary (non-const-generic) type params must match
        // the number of concrete type args supplied, or the zip in `build_type_subst`
        // would silently under/over-substitute. Skip (don't specialize) on a
        // mismatch rather than emit a broken clone.
        if ordinary_type_param_names(type_params).len() != cand.type_args.len() { continue; }
        instantiation_map_insert(&mut map, Instantiation {
            base_name: cand.name.clone(),
            type_args: cand.type_args.clone(),
            mangled,
            is_struct,
        });
    }
    map
}

/// Which kind of declaration a resolved generic method (`MethodInstantiation`)
/// was found on — a `StructDecl`, an `ExtDecl` (`ext TypeName: def m<U>(...): ...`),
/// or an `EnumDecl`. Resolution itself is name-only across all three kinds (see
/// `build_method_instantiation_map`'s doc comment) — this tag is only consulted
/// afterward, at method-append time in `monomorphize_program`, to find the right
/// container item (matched by kind AND owner name) to attach the specialized
/// clone to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MethodOwnerKind {
    Struct,
    Ext,
    Enum,
}

/// A resolved, specializable generic-method turbofish call site
/// (`obj.method<T>(...)`). Unlike `Instantiation`, this also carries the
/// declaring struct/ext/enum's name and kind — needed at method-append time (in
/// `monomorphize_program`) to find the right declaration to attach the specialized
/// clone to, since resolution itself never inspects the receiver's type (see
/// `MethodCandidateCall`'s doc comment).
#[derive(Debug, Clone)]
pub(crate) struct MethodInstantiation {
    pub owner_name: String,
    pub owner_kind: MethodOwnerKind,
    pub type_args: Vec<Type>,
    pub mangled: String,
}

/// Keyed by bare method name, deduplicated by mangled name — mirrors
/// `InstantiationMap`.
pub(crate) type MethodInstantiationMap = HashMap<String, Vec<MethodInstantiation>>;

fn method_instantiation_map_contains(map: &MethodInstantiationMap, method_name: &str, mangled: &str) -> bool {
    map.get(method_name).map(|v| v.iter().any(|i| i.mangled == mangled)).unwrap_or(false)
}

fn method_instantiation_map_insert(map: &mut MethodInstantiationMap, method_name: &str, inst: MethodInstantiation) {
    if method_instantiation_map_contains(map, method_name, &inst.mangled) { return; }
    map.entry(method_name.to_string()).or_default().push(inst);
}

/// Method-call analogue of `build_instantiation_map`. Resolution here is
/// deliberately name-only (no receiver-type tracking — see this module's
/// top-level doc comment and `Transpiler::global_generic_methods`): a method
/// name owned by more than one struct/ext/enum in the whole reachable graph —
/// regardless of which of the three kinds each owner is — is AMBIGUOUS and is
/// skipped entirely (zero entries emitted for it) rather than guessing — the
/// call site then safely falls through to the pre-existing `emit_generic_call`
/// fallback, emitting ordinary (unspecialized) generic Rust.
pub(crate) fn build_method_instantiation_map(
    candidates: &[MethodCandidateCall],
    generic_methods: &HashMap<String, Vec<(String, MethodOwnerKind, Vec<String>, FnDecl)>>,
) -> MethodInstantiationMap {
    let mut map = MethodInstantiationMap::new();
    for cand in candidates {
        let Some(owners) = generic_methods.get(cand.method_name.as_str()) else { continue };
        // Ambiguous (declared with its own type param on more than one
        // struct/ext/enum anywhere in the reachable graph, combined across all
        // three kinds) — skip, safe no-op fallback.
        if owners.len() != 1 { continue; }
        let (owner_name, owner_kind, own_type_params, _decl) = &owners[0];
        if !cand.type_args.iter().all(|t| is_concrete_type(t, &cand.enclosing_type_params)) { continue; }
        let Some(mangled) = mangled_name(&cand.method_name, &cand.type_args) else { continue; };
        // Sanity: the method's own (non-const-generic) type param count must match
        // the turbofish arg count, or the zip in `build_type_subst` would silently
        // under/over-substitute. Skip (don't specialize) on a mismatch.
        if ordinary_type_param_names(own_type_params).len() != cand.type_args.len() { continue; }
        method_instantiation_map_insert(&mut map, &cand.method_name, MethodInstantiation {
            owner_name: owner_name.clone(),
            owner_kind: *owner_kind,
            type_args: cand.type_args.clone(),
            mangled,
        });
    }
    map
}

// ─── Call-site rewriting ────────────────────────────────────────────────────────

/// Re-walks `program` mutably and rewrites every `ExprKind::GenericCall` whose
/// `(base_name, type_args)` matches an entry in `instantiations` into a plain
/// `ExprKind::Call(Var(mangled_name), args)` — turning the turbofish call into an
/// ordinary call on the specialized (non-generic) identifier, so it flows through
/// the existing, unmodified `emit_call`/`emit_constructor` path with zero turbofish
/// left for the emitter to deal with.
pub(crate) fn rewrite_call_sites(program: &mut Program, instantiations: &InstantiationMap, method_instantiations: &MethodInstantiationMap) {
    for item in &mut program.items {
        rewrite_in_item(item, instantiations, method_instantiations);
    }
}

fn rewrite_in_item(item: &mut Item, instantiations: &InstantiationMap, method_instantiations: &MethodInstantiationMap) {
    match item {
        Item::Fn(f) => rewrite_in_stmts(&mut f.body, instantiations, method_instantiations),
        Item::Struct(s) => rewrite_in_struct(s, instantiations, method_instantiations),
        Item::Enum(e) => rewrite_in_enum(e, instantiations, method_instantiations),
        Item::Ext(ext) => {
            for m in &mut ext.methods { rewrite_in_stmts(&mut m.body, instantiations, method_instantiations); }
            for conv in &mut ext.conversions { rewrite_in_stmts(&mut conv.body, instantiations, method_instantiations); }
            for set in &mut ext.setters { rewrite_in_stmts(&mut set.body, instantiations, method_instantiations); }
        }
        Item::Mod(m) => { for it in &mut m.items { rewrite_in_item(it, instantiations, method_instantiations); } }
        Item::Let(l) => { if let Some(v) = &mut l.value { rewrite_in_expr(v, instantiations, method_instantiations); } }
        Item::Stmt(s) => rewrite_in_stmt(s, instantiations, method_instantiations),
        Item::Use(_) | Item::Trait(_) | Item::Kernel(_) | Item::Alias(_) => {}
    }
}

fn rewrite_in_struct(s: &mut StructDecl, instantiations: &InstantiationMap, method_instantiations: &MethodInstantiationMap) {
    for m in &mut s.methods { rewrite_in_stmts(&mut m.body, instantiations, method_instantiations); }
    for init in &mut s.inits { rewrite_in_stmts(&mut init.body, instantiations, method_instantiations); }
    for conv in &mut s.conversions { rewrite_in_stmts(&mut conv.body, instantiations, method_instantiations); }
    for set in &mut s.setters { rewrite_in_stmts(&mut set.body, instantiations, method_instantiations); }
    for tm in &mut s.type_methods { rewrite_in_stmts(&mut tm.body, instantiations, method_instantiations); }
    for tv in &mut s.type_vars { rewrite_in_expr(&mut tv.default, instantiations, method_instantiations); }
    for f in &mut s.fields { if let Some(d) = &mut f.default { rewrite_in_expr(d, instantiations, method_instantiations); } }
}

fn rewrite_in_enum(e: &mut EnumDecl, instantiations: &InstantiationMap, method_instantiations: &MethodInstantiationMap) {
    for m in &mut e.methods { rewrite_in_stmts(&mut m.body, instantiations, method_instantiations); }
    for conv in &mut e.conversions { rewrite_in_stmts(&mut conv.body, instantiations, method_instantiations); }
    for set in &mut e.setters { rewrite_in_stmts(&mut set.body, instantiations, method_instantiations); }
}

fn rewrite_in_stmts(stmts: &mut [Stmt], instantiations: &InstantiationMap, method_instantiations: &MethodInstantiationMap) {
    for s in stmts { rewrite_in_stmt(s, instantiations, method_instantiations); }
}

fn rewrite_in_cond_clause(c: &mut CondClause, instantiations: &InstantiationMap, method_instantiations: &MethodInstantiationMap) {
    match c {
        CondClause::Let(_, e) | CondClause::LetPat(_, e) | CondClause::Expr(e) => rewrite_in_expr(e, instantiations, method_instantiations),
    }
}

fn rewrite_in_stmt(stmt: &mut Stmt, instantiations: &InstantiationMap, method_instantiations: &MethodInstantiationMap) {
    match stmt {
        Stmt::Let(l) => { if let Some(v) = &mut l.value { rewrite_in_expr(v, instantiations, method_instantiations); } }
        Stmt::LetDestructure(l) => rewrite_in_expr(&mut l.value, instantiations, method_instantiations),
        Stmt::Return(r) => { if let Some(v) = &mut r.value { rewrite_in_expr(v, instantiations, method_instantiations); } }
        Stmt::Break(_, v) => { if let Some(v) = v { rewrite_in_expr(v, instantiations, method_instantiations); } }
        Stmt::Continue(_) => {}
        Stmt::Throw(t) => { if let Some(v) = &mut t.value { rewrite_in_expr(v, instantiations, method_instantiations); } }
        Stmt::If(i) => {
            for (cond, body) in &mut i.branches { rewrite_in_expr(cond, instantiations, method_instantiations); rewrite_in_stmts(body, instantiations, method_instantiations); }
            if let Some(e) = &mut i.else_body { rewrite_in_stmts(e, instantiations, method_instantiations); }
        }
        Stmt::IfLet(i) => {
            for c in &mut i.clauses { rewrite_in_cond_clause(c, instantiations, method_instantiations); }
            rewrite_in_stmts(&mut i.then_body, instantiations, method_instantiations);
            for branch in &mut i.elif_branches {
                for c in &mut branch.clauses { rewrite_in_cond_clause(c, instantiations, method_instantiations); }
                rewrite_in_stmts(&mut branch.body, instantiations, method_instantiations);
            }
            if let Some(e) = &mut i.else_body { rewrite_in_stmts(e, instantiations, method_instantiations); }
        }
        Stmt::Match(m) => {
            rewrite_in_expr(&mut m.subject, instantiations, method_instantiations);
            for arm in &mut m.arms {
                if let Some(g) = &mut arm.guard { rewrite_in_expr(g, instantiations, method_instantiations); }
                match &mut arm.body {
                    MatchBody::Expr(e) => rewrite_in_expr(e, instantiations, method_instantiations),
                    MatchBody::Block(b) => rewrite_in_stmts(b, instantiations, method_instantiations),
                }
            }
        }
        Stmt::While(w) => { rewrite_in_expr(&mut w.condition, instantiations, method_instantiations); rewrite_in_stmts(&mut w.body, instantiations, method_instantiations); }
        Stmt::WhileLet(w) => { rewrite_in_expr(&mut w.value, instantiations, method_instantiations); rewrite_in_stmts(&mut w.body, instantiations, method_instantiations); }
        Stmt::DoWhile(d) => { rewrite_in_stmts(&mut d.body, instantiations, method_instantiations); rewrite_in_expr(&mut d.condition, instantiations, method_instantiations); }
        Stmt::Loop(l) => rewrite_in_stmts(&mut l.body, instantiations, method_instantiations),
        Stmt::Wait(e, _) => rewrite_in_expr(e, instantiations, method_instantiations),
        Stmt::For(f) => { rewrite_in_expr(&mut f.iterable, instantiations, method_instantiations); rewrite_in_stmts(&mut f.body, instantiations, method_instantiations); }
        Stmt::Guard(g) => {
            match &mut g.cond {
                GuardCond::Expr(e) => rewrite_in_expr(e, instantiations, method_instantiations),
                GuardCond::Clauses(cs) => { for c in cs { rewrite_in_cond_clause(c, instantiations, method_instantiations); } }
            }
            rewrite_in_stmts(&mut g.else_body, instantiations, method_instantiations);
        }
        Stmt::Try(t) => {
            rewrite_in_stmts(&mut t.body, instantiations, method_instantiations);
            for c in &mut t.catch_clauses { rewrite_in_stmts(&mut c.body, instantiations, method_instantiations); }
        }
        Stmt::Defer(body) => rewrite_in_stmts(body, instantiations, method_instantiations),
        Stmt::Expr(e) => rewrite_in_expr(e, instantiations, method_instantiations),
        Stmt::Fn(f) => rewrite_in_stmts(&mut f.body, instantiations, method_instantiations),
        Stmt::Struct(s) => rewrite_in_struct(s, instantiations, method_instantiations),
        Stmt::Enum(e) => rewrite_in_enum(e, instantiations, method_instantiations),
        Stmt::Mod(m) => { for it in &mut m.items { rewrite_in_item(it, instantiations, method_instantiations); } }
        Stmt::Alias(_) => {}
        Stmt::Yield(e, _) => rewrite_in_expr(e, instantiations, method_instantiations),
        Stmt::Comment(_) => {}
        Stmt::KernelBlock(k) => rewrite_in_stmts(&mut k.body, instantiations, method_instantiations),
        Stmt::With(w) => rewrite_in_stmts(&mut w.body, instantiations, method_instantiations),
    }
}

fn rewrite_in_args(args: &mut [Arg], instantiations: &InstantiationMap, method_instantiations: &MethodInstantiationMap) {
    for a in args { rewrite_in_expr(&mut a.value, instantiations, method_instantiations); }
}

fn rewrite_in_expr(expr: &mut Expr, instantiations: &InstantiationMap, method_instantiations: &MethodInstantiationMap) {
    // Recurse first so a nested GenericCall inside this node's sub-expressions
    // (args, callee, ...) gets rewritten regardless of what happens at this level.
    match &mut expr.kind {
        ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Str(_)
            | ExprKind::Bool(_) | ExprKind::Nil | ExprKind::Void | ExprKind::Var(_)
            | ExprKind::DotIdent(_) => {}
        ExprKind::StringInterp(segs) => {
            for seg in segs {
                match seg {
                    StringSegment::Lit(_) => {}
                    StringSegment::Expr(e) | StringSegment::FormattedExpr(e, _) => rewrite_in_expr(e, instantiations, method_instantiations),
                }
            }
        }
        ExprKind::BinOp(_, l, r) => { rewrite_in_expr(l, instantiations, method_instantiations); rewrite_in_expr(r, instantiations, method_instantiations); }
        ExprKind::UnaryOp(_, e) => rewrite_in_expr(e, instantiations, method_instantiations),
        ExprKind::Assign(l, r) | ExprKind::QuestionAssign(l, r) => { rewrite_in_expr(l, instantiations, method_instantiations); rewrite_in_expr(r, instantiations, method_instantiations); }
        ExprKind::Field(e, _) => rewrite_in_expr(e, instantiations, method_instantiations),
        ExprKind::Index(e, i) => { rewrite_in_expr(e, instantiations, method_instantiations); rewrite_in_expr(i, instantiations, method_instantiations); }
        ExprKind::LabeledIndex(e, args) => { rewrite_in_expr(e, instantiations, method_instantiations); rewrite_in_args(args, instantiations, method_instantiations); }
        ExprKind::Call(callee, args) => { rewrite_in_expr(callee, instantiations, method_instantiations); rewrite_in_args(args, instantiations, method_instantiations); }
        ExprKind::MethodCall(obj, _, args) => { rewrite_in_expr(obj, instantiations, method_instantiations); rewrite_in_args(args, instantiations, method_instantiations); }
        ExprKind::GenericCall(callee, _type_args, args) => {
            rewrite_in_expr(callee, instantiations, method_instantiations);
            rewrite_in_args(args, instantiations, method_instantiations);
        }
        ExprKind::Pipe(lhs, _, args) => { rewrite_in_expr(lhs, instantiations, method_instantiations); rewrite_in_args(args, instantiations, method_instantiations); }
        ExprKind::New { arena, ctor } => {
            if let Some(a) = arena { rewrite_in_expr(a, instantiations, method_instantiations); }
            rewrite_in_expr(ctor, instantiations, method_instantiations);
        }
        ExprKind::KernelLaunch { kernel, .. } => rewrite_in_expr(kernel, instantiations, method_instantiations),
        ExprKind::TryElse(a, b) => { rewrite_in_expr(a, instantiations, method_instantiations); rewrite_in_expr(b, instantiations, method_instantiations); }
        ExprKind::TryElseBlock(body, else_body) => {
            rewrite_in_stmts(body, instantiations, method_instantiations);
            rewrite_in_stmts(else_body, instantiations, method_instantiations);
        }
        ExprKind::Array(items) | ExprKind::Tuple(items) | ExprKind::Set(items) => {
            for e in items { rewrite_in_expr(e, instantiations, method_instantiations); }
        }
        ExprKind::ArrayFill { value, count } => { rewrite_in_expr(value, instantiations, method_instantiations); rewrite_in_expr(count, instantiations, method_instantiations); }
        ExprKind::ArrayAlloc { count } => rewrite_in_expr(count, instantiations, method_instantiations),
        ExprKind::ArrayComp { expr: e, var: _, count } => { rewrite_in_expr(e, instantiations, method_instantiations); rewrite_in_expr(count, instantiations, method_instantiations); }
        ExprKind::ArrayCompIter { expr: e, var: _, iter } => { rewrite_in_expr(e, instantiations, method_instantiations); rewrite_in_expr(iter, instantiations, method_instantiations); }
        ExprKind::LabeledArrayComp { expr: e, clauses } => {
            rewrite_in_expr(e, instantiations, method_instantiations);
            for (_, c) in clauses { rewrite_in_expr(c, instantiations, method_instantiations); }
        }
        ExprKind::Dict(pairs) => { for (k, v) in pairs { rewrite_in_expr(k, instantiations, method_instantiations); rewrite_in_expr(v, instantiations, method_instantiations); } }
        ExprKind::Range { start, end, .. } => { rewrite_in_expr(start, instantiations, method_instantiations); rewrite_in_expr(end, instantiations, method_instantiations); }
        ExprKind::SliceRange { start, end, .. } => {
            if let Some(s) = start { rewrite_in_expr(s, instantiations, method_instantiations); }
            if let Some(e) = end { rewrite_in_expr(e, instantiations, method_instantiations); }
        }
        ExprKind::Cast(e, _) => rewrite_in_expr(e, instantiations, method_instantiations),
        ExprKind::RelabelCast(e, _) => rewrite_in_expr(e, instantiations, method_instantiations),
        ExprKind::Else(a, b) => { rewrite_in_expr(a, instantiations, method_instantiations); rewrite_in_expr(b, instantiations, method_instantiations); }
        ExprKind::OptionalField(e, _) => rewrite_in_expr(e, instantiations, method_instantiations),
        ExprKind::OptionalMethodCall(obj, _, args) => { rewrite_in_expr(obj, instantiations, method_instantiations); rewrite_in_args(args, instantiations, method_instantiations); }
        ExprKind::Closure(_, _, body, _, _) => match body {
            ClosureBody::Expr(e) => rewrite_in_expr(e, instantiations, method_instantiations),
            ClosureBody::Block(b) => rewrite_in_stmts(b, instantiations, method_instantiations),
        },
        ExprKind::If(if_stmt) => {
            for (cond, body) in &mut if_stmt.branches { rewrite_in_expr(cond, instantiations, method_instantiations); rewrite_in_stmts(body, instantiations, method_instantiations); }
            if let Some(e) = &mut if_stmt.else_body { rewrite_in_stmts(e, instantiations, method_instantiations); }
        }
        ExprKind::Match(match_stmt) => {
            rewrite_in_expr(&mut match_stmt.subject, instantiations, method_instantiations);
            for arm in &mut match_stmt.arms {
                if let Some(g) = &mut arm.guard { rewrite_in_expr(g, instantiations, method_instantiations); }
                match &mut arm.body {
                    MatchBody::Expr(e) => rewrite_in_expr(e, instantiations, method_instantiations),
                    MatchBody::Block(b) => rewrite_in_stmts(b, instantiations, method_instantiations),
                }
            }
        }
        ExprKind::Block(stmts) | ExprKind::Do(stmts) => rewrite_in_stmts(stmts, instantiations, method_instantiations),
        ExprKind::Loop(l) => rewrite_in_stmts(&mut l.body, instantiations, method_instantiations),
        ExprKind::Task(e) => rewrite_in_expr(e, instantiations, method_instantiations),
        ExprKind::TaskWithTimeout(a, b) => { rewrite_in_expr(a, instantiations, method_instantiations); rewrite_in_expr(b, instantiations, method_instantiations); }
        ExprKind::JoinAll(items) => { for e in items { rewrite_in_expr(e, instantiations, method_instantiations); } }
        ExprKind::MacroCall { name: _, args } => { for e in args { rewrite_in_expr(e, instantiations, method_instantiations); } }
    }

    // Now, having recursed into every child first, check whether THIS node itself is
    // a turbofish call site that matches a known instantiation.
    if let ExprKind::GenericCall(callee, type_args, _) = &expr.kind {
        match &callee.kind {
            ExprKind::Var(name) => {
                if let Some(variants) = instantiations.get(name.as_str()) {
                    if let Some(inst) = variants.iter().find(|i| i.type_args == *type_args) {
                        let mangled = inst.mangled.clone();
                        let line = expr.line;
                        let col = expr.col;
                        let len = expr.len;
                        // Take ownership of the existing node to move out `args` without cloning.
                        let old = std::mem::replace(expr, Expr { kind: ExprKind::Void, line, col, len });
                        let ExprKind::GenericCall(old_callee, _, args) = old.kind else { unreachable!() };
                        let new_callee = Expr { kind: ExprKind::Var(mangled), line: old_callee.line, col: old_callee.col, len: old_callee.len };
                        expr.kind = ExprKind::Call(Box::new(new_callee), args);
                    }
                }
            }
            // `obj.method<T>(...)` — generic method call (V1 extension). Rewrite
            // into a plain `ExprKind::MethodCall(receiver, mangled_name, args)`,
            // preserving `receiver` as-is (unlike the free-fn/struct case above,
            // the receiver expression must be kept — there is no bare identifier
            // to swap in its place).
            ExprKind::Field(_, method) => {
                if let Some(variants) = method_instantiations.get(method.as_str()) {
                    if let Some(inst) = variants.iter().find(|i| i.type_args == *type_args) {
                        let mangled = inst.mangled.clone();
                        let line = expr.line;
                        let col = expr.col;
                        let len = expr.len;
                        let old = std::mem::replace(expr, Expr { kind: ExprKind::Void, line, col, len });
                        let ExprKind::GenericCall(old_callee, _, args) = old.kind else { unreachable!() };
                        let ExprKind::Field(receiver, _) = old_callee.kind else { unreachable!() };
                        expr.kind = ExprKind::MethodCall(receiver, mangled, args);
                    }
                }
            }
            // `obj?.method<T>(...)` — same rewrite as the plain `.method<T>(...)`
            // case above, but preserving the optional/short-circuit semantics: the
            // resolved candidate rewrites into `OptionalMethodCall(receiver,
            // mangled_name, args)` rather than a plain `MethodCall`.
            ExprKind::OptionalField(_, method) => {
                if let Some(variants) = method_instantiations.get(method.as_str()) {
                    if let Some(inst) = variants.iter().find(|i| i.type_args == *type_args) {
                        let mangled = inst.mangled.clone();
                        let line = expr.line;
                        let col = expr.col;
                        let len = expr.len;
                        let old = std::mem::replace(expr, Expr { kind: ExprKind::Void, line, col, len });
                        let ExprKind::GenericCall(old_callee, _, args) = old.kind else { unreachable!() };
                        let ExprKind::OptionalField(receiver, _) = old_callee.kind else { unreachable!() };
                        expr.kind = ExprKind::OptionalMethodCall(receiver, mangled, args);
                    }
                }
            }
            _ => {}
        }
    }
}

// ─── Pipeline hook ──────────────────────────────────────────────────────────────

/// Cheap top-level-only scan (no recursion into bodies) for whether `program`
/// declares any generic struct/free-fn at all — used to skip the owned clone +
/// full walk in `emit_program`'s common case (no generics in this file).
pub(crate) fn program_declares_generics(program: &Program) -> bool {
    program.items.iter().any(|item| match item {
        // A struct triggers the gate either by being itself generic, or by owning
        // at least one method with its own separate type param (see
        // `Transpiler::global_generic_methods`'s doc comment) — the declaring file
        // must always be monomorphized so the specialized method clone can be
        // attached, even when the only turbofish call site naming it lives in a
        // different file.
        Item::Struct(s) => !s.type_params.is_empty()
            || s.methods.iter().any(|m| m.type_params.iter().any(|p| !s.type_params.contains(p))),
        Item::Fn(f) => !f.type_params.is_empty() && f.qualifier.is_none(),
        // An `ext`/`enum` is never itself whole-item-specialized (see this
        // module's doc comment), so only an own-generic METHOD gates it — the
        // declaring file must still be monomorphized so the specialized method
        // clone can be appended, even when the only turbofish call site naming
        // it lives in a different file. Mirrors the struct arm's method check.
        Item::Ext(e) => e.methods.iter().any(|m| m.type_params.iter().any(|p| !e.type_params.contains(p))),
        Item::Enum(en) => en.methods.iter().any(|m| m.type_params.iter().any(|p| !en.type_params.contains(p))),
        _ => false,
    })
}

/// True when `program` contains at least one turbofish (`ExprKind::GenericCall`)
/// site anywhere in its reachable bodies — cheap syntactic check, doesn't need to
/// know yet whether the callee resolves to a locally- or cross-file-declared
/// generic. Widens `emit_program`'s monomorphization gate (alongside
/// `program_declares_generics`) so a file that merely CALLS a cross-file generic
/// (but declares zero generics of its own) still goes through
/// `monomorphize_program` and gets its call sites rewritten against
/// `self.global_instantiations`.
pub(crate) fn program_contains_generic_call(program: &Program) -> bool {
    let cands = collect_candidate_calls(program);
    !cands.free.is_empty() || !cands.methods.is_empty()
}

impl Transpiler {
    /// Runs the full monomorphization pipeline over one file's own `Program`, in
    /// place: for every base name THIS file declares that has ≥1 entry in
    /// `self.global_instantiations` (the cross-file-complete map, already computed
    /// once up front — see `emit_program`'s call site and `emit_top::deep_pre_scan`),
    /// build a specialized non-generic clone and append it as a new top-level item;
    /// then ALWAYS rewrite this file's own call sites against
    /// `self.global_instantiations`, regardless of what this file itself declares —
    /// a file that only CALLS a cross-file generic (declares none itself) still gets
    /// its turbofish sites rewritten to plain calls on the specialized name.
    ///
    /// Called once per file, on that file's own freshly-parsed `Program`, before any
    /// of its items are emitted. Each specialization is emitted exactly once — only
    /// by the file that declares its base — avoiding duplicate-definition (E0428)
    /// even when multiple other files call the same instantiation; cross-file
    /// visibility works with zero prefixing because the emitted Rust is one flat,
    /// concatenated namespace (see this module's top-level doc comment).
    pub(crate) fn monomorphize_program(&mut self, program: &mut Program) {
        // Which base names does THIS file declare? (Local, cheap, name-only scan —
        // the actual decl clones already live in `self.global_generic_structs`/
        // `global_generic_fns`, populated once for the whole reachable graph.)
        let mut local_generic_struct_names: Vec<String> = Vec::new();
        let mut local_generic_fn_names: Vec<String> = Vec::new();
        for item in &program.items {
            match item {
                Item::Struct(s) if !s.type_params.is_empty() => local_generic_struct_names.push(s.name.clone()),
                // Free functions only — a `def Type.method()`-style qualified top-level
                // `Item::Fn` is excluded (`qualifier.is_some()`); ordinary methods live
                // inside `StructDecl.methods`, never as a top-level `Item::Fn`, and a
                // turbofish call site can only ever name a bare identifier anyway
                // (`record_candidate` requires `ExprKind::Var`), so this exclusion is
                // defensive rather than load-bearing.
                Item::Fn(f) if !f.type_params.is_empty() && f.qualifier.is_none() => local_generic_fn_names.push(f.name.clone()),
                _ => {}
            }
        }

        // Generic-method monomorphization (`obj.method<T>(...)`): for every
        // struct/ext/enum DECLARED in this file (generic or not — a non-generic
        // struct with an own-generic method, like a bare `struct Box:` with a
        // `def push<U>(...)` method, is in scope here too, and likewise for a
        // non-generic `ext`/`enum`), append a specialized clone of each
        // own-generic method matched by a resolved instantiation in
        // `self.global_method_instantiations`. Deliberately runs BEFORE the
        // struct-level specialization loop right below, so a newly-appended method
        // clone is already part of the struct's method list by the time that loop
        // clones the struct (from `self.global_generic_structs`'s stored copy,
        // mutated in lockstep here — NOT from `program.items`) into every concrete
        // struct instantiation. `ext`/`enum` items are never whole-item cloned by
        // any existing pass, so no analogous mirroring is needed for them — see
        // this module's doc comment for the fuller rationale. See
        // `Transpiler::global_generic_methods`'s doc comment for the resolution
        // rules (ambiguous method names, across all three kinds, are skipped).
        for item in program.items.iter_mut() {
            match item {
                Item::Struct(s) => {
                    let struct_name = s.name.clone();
                    let mut new_methods: Vec<FnDecl> = Vec::new();
                    for m in &s.methods {
                        let own_type_params: Vec<String> = m.type_params.iter()
                            .filter(|p| !s.type_params.contains(p))
                            .cloned()
                            .collect();
                        if own_type_params.is_empty() { continue; }
                        let Some(insts) = self.global_method_instantiations.get(&m.name) else { continue };
                        for inst in insts {
                            if inst.owner_kind != MethodOwnerKind::Struct || inst.owner_name != struct_name { continue; }
                            let subst = build_type_subst(&own_type_params, &inst.type_args);
                            let spec = substitute_fn_decl(m, &subst, &inst.mangled);
                            new_methods.push(spec);
                        }
                    }
                    if new_methods.is_empty() { continue; }
                    s.methods.extend(new_methods.clone());
                    // Mirror into `self.global_generic_structs`'s stored copy (only
                    // present when this struct is itself generic) so the
                    // struct-level specialization loop below — which clones FROM
                    // that stored copy, not from `program.items` — carries the new
                    // method(s) into every concrete struct clone too.
                    if let Some(stored) = self.global_generic_structs.get_mut(&struct_name) {
                        for m in &new_methods {
                            if !stored.methods.iter().any(|sm| sm.name == m.name) {
                                stored.methods.push(m.clone());
                            }
                        }
                    }
                }
                Item::Ext(e) => {
                    let owner_name = e.type_name.clone();
                    let mut new_methods: Vec<FnDecl> = Vec::new();
                    for m in &e.methods {
                        let own_type_params: Vec<String> = m.type_params.iter()
                            .filter(|p| !e.type_params.contains(p))
                            .cloned()
                            .collect();
                        if own_type_params.is_empty() { continue; }
                        let Some(insts) = self.global_method_instantiations.get(&m.name) else { continue };
                        for inst in insts {
                            if inst.owner_kind != MethodOwnerKind::Ext || inst.owner_name != owner_name { continue; }
                            let subst = build_type_subst(&own_type_params, &inst.type_args);
                            let spec = substitute_fn_decl(m, &subst, &inst.mangled);
                            new_methods.push(spec);
                        }
                    }
                    if !new_methods.is_empty() { e.methods.extend(new_methods); }
                }
                Item::Enum(en) => {
                    let owner_name = en.name.clone();
                    let mut new_methods: Vec<FnDecl> = Vec::new();
                    for m in &en.methods {
                        let own_type_params: Vec<String> = m.type_params.iter()
                            .filter(|p| !en.type_params.contains(p))
                            .cloned()
                            .collect();
                        if own_type_params.is_empty() { continue; }
                        let Some(insts) = self.global_method_instantiations.get(&m.name) else { continue };
                        for inst in insts {
                            if inst.owner_kind != MethodOwnerKind::Enum || inst.owner_name != owner_name { continue; }
                            let subst = build_type_subst(&own_type_params, &inst.type_args);
                            let spec = substitute_fn_decl(m, &subst, &inst.mangled);
                            new_methods.push(spec);
                        }
                    }
                    if !new_methods.is_empty() { en.methods.extend(new_methods); }
                }
                _ => {}
            }
        }

        for name in &local_generic_struct_names {
            let Some(insts) = self.global_instantiations.get(name) else { continue };
            let Some(decl) = self.global_generic_structs.get(name) else { continue };
            let decl = decl.clone();
            let insts = insts.clone();
            for inst in insts {
                if !inst.is_struct { continue; }
                let subst = build_type_subst(&decl.type_params, &inst.type_args);
                let spec = substitute_struct_decl(&decl, &subst, &inst.mangled);
                program.items.push(Item::Struct(spec));
                self.struct_has_specialization.insert(name.clone());
            }
        }
        for name in &local_generic_fn_names {
            let Some(insts) = self.global_instantiations.get(name) else { continue };
            let Some(decl) = self.global_generic_fns.get(name) else { continue };
            let decl = decl.clone();
            let insts = insts.clone();
            for inst in insts {
                if inst.is_struct { continue; }
                let subst = build_type_subst(&decl.type_params, &inst.type_args);
                let spec = substitute_fn_decl(&decl, &subst, &inst.mangled);
                program.items.push(Item::Fn(spec));
            }
        }

        rewrite_call_sites(program, &self.global_instantiations, &self.global_method_instantiations);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subst_of(pairs: &[(&str, Type)]) -> HashMap<String, Type> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn substitute_type_struct_field() {
        // A struct field typed `T` substitutes directly to the concrete arg.
        let ty = Type::TypeParam("T".to_string());
        let subst = subst_of(&[("T", Type::Int)]);
        assert_eq!(substitute_type(&ty, &subst), Type::Int);
    }

    #[test]
    fn substitute_type_nested_optional_array() {
        // `[T?]` (Array<Optional<TypeParam T>>) → `[int?]` once T = int.
        let ty = Type::Array(Box::new(Type::Optional(Box::new(Type::TypeParam("T".to_string())))));
        let subst = subst_of(&[("T", Type::Int)]);
        let expected = Type::Array(Box::new(Type::Optional(Box::new(Type::Int))));
        assert_eq!(substitute_type(&ty, &subst), expected);
    }

    #[test]
    fn substitute_type_named_multiletter_param() {
        // Multi-letter type param names parse as `Type::Named`, not `Type::TypeParam`
        // (src/parser/parse_type.rs) — substitution must match both variants.
        let ty = Type::Named("Value".to_string());
        let subst = subst_of(&[("Value", Type::Str)]);
        assert_eq!(substitute_type(&ty, &subst), Type::Str);
    }

    #[test]
    fn substitute_type_unrelated_named_untouched() {
        let ty = Type::Named("OtherStruct".to_string());
        let subst = subst_of(&[("T", Type::Int)]);
        assert_eq!(substitute_type(&ty, &subst), ty);
    }

    #[test]
    fn mangled_name_two_args() {
        assert_eq!(
            mangled_name("Pair", &[Type::Int, Type::Str]),
            Some("Pair_int_string".to_string())
        );
    }

    #[test]
    fn mangled_name_bails_on_unmangleable_arg() {
        // A tuple type arg isn't in the V1 mangler's known-shape list.
        assert_eq!(mangled_name("Wrapper", &[Type::Tuple(vec![Type::Int, Type::Int])]), None);
    }

    #[test]
    fn mangled_name_mut_qualified_arg() {
        assert_eq!(
            mangled_name("Container", &[Type::Mut(Box::new(Type::Named("Point".to_string())))]),
            Some("Container_mut_Point".to_string())
        );
    }
}
