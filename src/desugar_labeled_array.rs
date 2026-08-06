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

//! Desugars `Type::LabeledArray` (docs/array-multidim-types.md) — the
//! labeled multi-dimensional array feature that replaced `Image`/`Volume`
//! (deleted once whisper-boring, the only real consumer, was migrated off
//! them). Runs once, on the whole `Program`, right after parsing — every
//! consumer downstream (checker, interpreter, all four transpiler backends)
//! only ever sees plain `Index`/`Var` expressions and ordinary
//! flat-array-typed fields.
//!
//! This pass runs over **the whole program** — free functions, struct/kernel
//! methods, top-level `let`s — not just kernel fields, matching the design
//! doc's CPU-and-GPU ambition.
//!
//! ## Scope (v1) — see docs/array-multidim-types.md
//!
//! Handled:
//! - Kernel fields, dynamic shape: flat buffer + positional shadow fields
//!   (`__name_axis0`, `__name_axis1`, ...), exactly like `Image<T>`/
//!   `Volume<T>`'s existing treatment, just positionally named instead of
//!   `w`/`h`/`d` (labels are arbitrary user text here, so a positional
//!   scheme avoids downstream code needing to remember spelling/order).
//! - Kernel fields, fixed shape: **untouched** — left as `Type::LabeledArray`
//!   for the interpreter's GPU path (`eval_gpu.rs`, stage 5) and all four
//!   transpiler backends (stage 6) to lower directly, mirroring how
//!   `Image<T,C,R>`'s fixed shape is never desugared either.
//! - Plain `let`-declared locals (free functions, methods, top-level) with
//!   an explicit `LabeledArray` type annotation: same flat-buffer-plus-
//!   shadow-bindings treatment for dynamic shape; fixed-shape `.at`-
//!   equivalent/`.size(.axis)` calls fold directly into `Index`/a literal at
//!   this pass, since the axis sizes are already compile-time-known.
//! - `[f(w,h) for w in ..W for h in ..H]` (`ExprKind::LabeledArrayComp`) —
//!   lowered to an `ArrayAlloc` + nested `for` loops (innermost = axis 1 =
//!   `clauses[0]`, per its own doc comment) writing into the flat buffer.
//! - `.reshape(...)` / `.flatten()` — identity at the value level (the
//!   underlying buffer is unchanged either way); `.reshape(...)`'s *shape*
//!   is captured by synthesizing shadow-axis bindings from its named
//!   arguments, but only when it appears directly as a `let`'s initializer
//!   (see `extract_shadow_values`) — the one place this pass has an
//!   annotation to synthesize shadows *for*.
//! - `img as [line = width, column = height]` (`ExprKind::RelabelCast`) —
//!   always desugars to a plain passthrough of its inner expression,
//!   unconditionally, wherever it appears. Shape is threaded positionally
//!   (axis order, not label text), so a same-axis-order relabel has no
//!   runtime effect — all of `as [...]`'s real work already happened in the
//!   checker (`check_relabel_cast`).
//!
//! Explicitly deferred (documented, not silently dropped — see the
//! implementation plan, stage 4):
//! - Ordinary (non-kernel) struct fields — not needed by any real `.br` file
//!   today; would reuse this exact machinery once needed.
//! - A dynamic-shape labeled array as an arbitrary function *parameter* —
//!   needs call-site shadow-argument threading, a capability `Image`/
//!   `Volume` never needed either (its dynamic form was only ever a kernel
//!   field). Not exercised by any real `.br` file today.
//! - `LabeledIndex`/`.size(.axis)` on an object this pass can't trace back to
//!   a `let`-tracked declaration (e.g. flowing through an untracked function
//!   parameter or return value) — best-effort only: left unresolved rather
//!   than guessed, matching the checker's own "never a false positive, just
//!   possibly a missed check" limitation for the same reason
//!   (`Checker::static_labeled_array_type`).

use crate::ast::*;
use std::collections::HashMap;

/// What's known about one `let`-tracked name: its element type, its axis
/// list (all-dynamic or all-fixed, per D1 — see `LabeledAxis`), and — for a
/// dynamic-shape name only — the positional shadow-binding names already
/// synthesized for it (`None` for fixed shape: no shadows needed, sizes are
/// already compile-time constants baked directly into rewritten expressions).
#[derive(Clone)]
struct LabeledInfo {
    // Not read by this pass yet — kept for symmetry with `axes` and for
    // future use (e.g. zero-init synthesis for a fixed-shape CPU-side
    // declaration with no initializer, not yet needed by any real `.br` file).
    #[allow(dead_code)]
    elem: Type,
    axes: Vec<LabeledAxis>,
    shadow_names: Option<Vec<String>>,
}

impl LabeledInfo {
    fn axis_size_expr(&self, i: usize, line: usize, col: usize) -> Expr {
        match &self.shadow_names {
            Some(names) => Expr { kind: ExprKind::Var(names[i].clone()), line, col, len: 0 },
            None => {
                let ConstExpr(boxed) = self.axes[i].size.as_ref()
                    .expect("fixed-shape LabeledInfo has Some(size) on every axis (D1)");
                (**boxed).clone()
            }
        }
    }
}

/// name -> what's known about it, built incrementally while walking a
/// statement/item sequence top-to-bottom (a name is only resolvable in
/// expressions *after* its own declaration — matching ordinary lexical
/// scoping closely enough for this pass's best-effort purposes).
type ArrayScope = HashMap<String, LabeledInfo>;

/// The synthesized shadow-binding name for axis `i` of a name — positional,
/// not label-text-based (see this module's doc comment for why).
fn shadow_axis_name(name: &str, i: usize) -> String {
    format!("__{name}_axis{i}")
}

pub fn desugar_labeled_array(mut program: Program) -> Program {
    let scope = ArrayScope::new();
    program.items = desugar_items(program.items, &scope);
    program
}

// ─── Item-level walk (top-level program, and `mod` bodies) ─────────────────

fn desugar_items(items: Vec<Item>, outer: &ArrayScope) -> Vec<Item> {
    let mut scope = outer.clone();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.extend(desugar_item(item, &mut scope));
    }
    out
}

fn desugar_item(item: Item, scope: &mut ArrayScope) -> Vec<Item> {
    match item {
        Item::Let(s) => desugar_let(s, scope).into_iter().map(Item::Let).collect(),
        Item::Fn(mut f) => {
            f.body = desugar_body(f.body, &ArrayScope::new());
            vec![Item::Fn(f)]
        }
        Item::Struct(mut s) => {
            for m in s.methods.iter_mut() {
                let body = std::mem::take(&mut m.body);
                m.body = desugar_body(body, &ArrayScope::new());
            }
            vec![Item::Struct(s)]
        }
        Item::Kernel(mut k) => {
            desugar_kernel_decl(&mut k);
            vec![Item::Kernel(k)]
        }
        Item::Mod(mut m) => {
            m.items = desugar_items(m.items, scope);
            vec![Item::Mod(m)]
        }
        Item::Stmt(stmt) => desugar_stmt(stmt, scope).into_iter().map(Item::Stmt).collect(),
        other => vec![other],
    }
}

// ─── Kernel fields (dynamic-shape only — fixed-shape is stage 5/6's job) ───

fn desugar_kernel_decl(decl: &mut KernelDecl) {
    let mut dynamic_scope: ArrayScope = HashMap::new();
    let mut new_fields = Vec::with_capacity(decl.fields.len());
    for field in decl.fields.drain(..) {
        if let Type::LabeledArray(elem, axes) = &field.ty {
            let is_dynamic = axes.iter().all(|a| a.size.is_none());
            if is_dynamic {
                let shadow_names: Vec<String> =
                    (0..axes.len()).map(|i| shadow_axis_name(&field.name, i)).collect();
                dynamic_scope.insert(
                    field.name.clone(),
                    LabeledInfo { elem: (**elem).clone(), axes: axes.clone(), shadow_names: Some(shadow_names.clone()) },
                );
                new_fields.push(KernelFieldDecl {
                    name: field.name,
                    binding: field.binding,
                    qual: field.qual,
                    ty: Type::Array(elem.clone()),
                    default: field.default,
                    line: field.line,
                    col: field.col,
                });
                for name in shadow_names {
                    new_fields.push(KernelFieldDecl {
                        name,
                        binding: FieldBinding::Let,
                        qual: GpuQual::Const,
                        // Type::Int, not Uint: a shadow value commonly comes
                        // from an `int`-typed init/for-loop-bound source
                        // (e.g. `nf`, `nfreq` in a real kernel constructor
                        // call) — matching that exactly avoids a type
                        // mismatch at every level (interpreter, transpiled
                        // Rust, *and* `for i in 0..field.size(.axis)`, whose
                        // range requires both bounds to be Int — confirmed
                        // via a real interpreter crash otherwise: "range
                        // requires Int bounds, got Int and Uint"). Sizes are
                        // never negative in practice, so Int vs Uint has no
                        // real consequence beyond the type itself matching
                        // everywhere.
                        ty: Type::Int,
                        default: None,
                        line: field.line,
                        col: field.col,
                    });
                }
                continue;
            }
            // Fixed-shape: leave untouched here — see this module's doc comment.
        }
        new_fields.push(field);
    }
    decl.fields = new_fields;

    if dynamic_scope.is_empty() {
        return;
    }
    for init in &mut decl.inits {
        init.body = desugar_body(std::mem::take(&mut init.body), &dynamic_scope);
    }
    for method in &mut decl.methods {
        method.body = desugar_body(std::mem::take(&mut method.body), &dynamic_scope);
    }
}

// ─── `let` handling (shared between top-level items and statement bodies) ──

/// If `value` is the recognized initializer for a dynamic-shape labeled
/// array — a chained-for comprehension, or a `.reshape(...)` call — returns
/// each axis's size expression, in the *declared* axis order (`axes`'
/// order), regardless of what order the comprehension's `for` clauses or
/// the `.reshape(...)` call's named arguments were written in. `None` for
/// anything else (deferred initialization, or an initializer this pass
/// doesn't recognize) — the declaration is still tracked in scope, just
/// without shadow bindings, so later indexing/`.size()` on it is left
/// unresolved rather than guessed (see this module's doc comment).
fn extract_shadow_values(value: &Expr, axes: &[LabeledAxis]) -> Option<Vec<Expr>> {
    match &value.kind {
        ExprKind::LabeledArrayComp { clauses, .. } if clauses.len() == axes.len() => {
            Some(clauses.iter().map(|(_, count)| (**count).clone()).collect())
        }
        ExprKind::MethodCall(_, method, args) if method == "reshape" => {
            let mut out = Vec::with_capacity(axes.len());
            for axis in axes {
                let arg = args.iter().find(|a| a.label.as_deref() == Some(axis.label.as_str()))?;
                out.push(arg.value.clone());
            }
            Some(out)
        }
        _ => None,
    }
}

/// Placeholder element type for a `let` whose `LabeledArray` shape is
/// *inferred* (no explicit annotation) — this pass has no general type
/// inference, so the real element type is genuinely unknown here. Harmless:
/// `LabeledInfo.elem` isn't read by anything in this pass today (see its
/// `#[allow(dead_code)]`).
fn unknown_elem() -> Type {
    Type::Named("_".to_string())
}

/// Determines a `let`'s labeled-array shape and, when it needs new shadow
/// bindings, their values — from the explicit annotation if present,
/// otherwise inferred directly from the initializer's own shape, so type
/// inference works with **no annotation at all**, matching the design doc's
/// own `let a = [f(w,h) for w in ..W for h in ..H]` example:
/// - an explicit `[T, ...]` annotation (dynamic or fixed);
/// - a chained-for comprehension — axis labels = the clause variable names;
/// - a `.reshape(width = W, height = H)` call — axis labels = its named
///   arguments;
/// - `other as [line = width, column = height]` — axis labels = the
///   mapping's target labels, *reusing* `other`'s own already-synthesized
///   shadow bindings (reordered/renamed per the mapping) rather than
///   synthesizing new ones — `RelabelCast` has no runtime effect (it
///   desugars to a plain passthrough), so there's nothing new to compute.
///
/// Returns `None` when nothing here recognizes the `let` as a labeled array
/// at all (an ordinary `let`, or an initializer shape this pass doesn't
/// track — left alone, not guessed).
fn infer_let_labeled_info(s: &LetStmt, scope: &ArrayScope) -> Option<(LabeledInfo, Option<Vec<Expr>>)> {
    if let Some(Type::LabeledArray(elem, axes)) = &s.ty {
        let is_dynamic = axes.iter().all(|a| a.size.is_none());
        let shadow_values = if is_dynamic { s.value.as_ref().and_then(|v| extract_shadow_values(v, axes)) } else { None };
        let shadow_names = if is_dynamic {
            Some((0..axes.len()).map(|i| shadow_axis_name(&s.name, i)).collect::<Vec<_>>())
        } else {
            None
        };
        return Some((LabeledInfo { elem: (**elem).clone(), axes: axes.clone(), shadow_names }, shadow_values));
    }

    let value = s.value.as_ref()?;
    match &value.kind {
        ExprKind::LabeledArrayComp { clauses, .. } => {
            let axes: Vec<LabeledAxis> = clauses.iter()
                .map(|(name, _)| LabeledAxis { label: name.clone(), size: None })
                .collect();
            let shadow_values = extract_shadow_values(value, &axes);
            let shadow_names = Some((0..axes.len()).map(|i| shadow_axis_name(&s.name, i)).collect());
            Some((LabeledInfo { elem: unknown_elem(), axes, shadow_names }, shadow_values))
        }
        ExprKind::MethodCall(_, method, args) if method == "reshape" => {
            let labels: Vec<String> = args.iter().filter_map(|a| a.label.clone()).collect();
            if labels.is_empty() { return None; }
            let axes: Vec<LabeledAxis> = labels.into_iter().map(|label| LabeledAxis { label, size: None }).collect();
            let shadow_values = extract_shadow_values(value, &axes);
            let shadow_names = Some((0..axes.len()).map(|i| shadow_axis_name(&s.name, i)).collect());
            Some((LabeledInfo { elem: unknown_elem(), axes, shadow_names }, shadow_values))
        }
        ExprKind::RelabelCast(inner, pairs) => {
            let ExprKind::Var(name) = &inner.kind else { return None };
            let inner_info = scope.get(name)?;
            let mut new_axes = Vec::with_capacity(pairs.len());
            let mut new_shadow_names = inner_info.shadow_names.as_ref().map(|_| Vec::with_capacity(pairs.len()));
            for (target, source) in pairs {
                let src_i = inner_info.axes.iter().position(|a| &a.label == source)?;
                new_axes.push(LabeledAxis { label: target.clone(), size: inner_info.axes[src_i].size.clone() });
                if let Some(dst) = &mut new_shadow_names {
                    dst.push(inner_info.shadow_names.as_ref().unwrap()[src_i].clone());
                }
            }
            // `None` for shadow_values: no NEW shadow lets to synthesize —
            // `new_shadow_names` above already points at `inner`'s existing
            // ones, just reordered/renamed.
            Some((LabeledInfo { elem: inner_info.elem.clone(), axes: new_axes, shadow_names: new_shadow_names }, None))
        }
        _ => None,
    }
}

/// Desugars one `let`, returning the (possibly-rewritten) original binding
/// plus any synthesized shadow-axis bindings, in declaration order — the
/// caller splices all of them in, back to back, right where the original
/// `let` was.
fn desugar_let(mut s: LetStmt, scope: &mut ArrayScope) -> Vec<LetStmt> {
    let line = s.line;
    let col = s.col;

    // Read the annotation/initializer shape *before* desugaring the
    // initializer — desugaring rewrites a LabeledArrayComp into its lowered
    // Block form (and a RelabelCast into a bare passthrough), losing the
    // very shape `infer_let_labeled_info` needs to read.
    let labeled = infer_let_labeled_info(&s, scope);

    s.value = s.value.map(|v| desugar_expr(v, scope));

    let Some((info, shadow_values)) = labeled else {
        return vec![s];
    };
    scope.insert(s.name.clone(), info.clone());
    let mut out = vec![s];
    if let (Some(names), Some(values)) = (&info.shadow_names, shadow_values) {
        for (name, value) in names.iter().zip(values.into_iter()) {
            out.push(LetStmt {
                binding: BindingKind::Let,
                var_mut: false,
                is_pub: false,
                is_static: false,
                name: name.clone(),
                // Type::Int — see the kernel-field shadow declaration's
                // identical note in desugar_kernel_decl. Explicit `as int`:
                // the source expression's own type isn't guaranteed (an
                // `int` OR a `uint` param both appear in real code) — the
                // cast normalizes either into a definite Int value, which
                // both `for i in 0..field.size(.axis)`'s range (requires
                // Int on both sides) and the shadow field's own declared
                // type need unconditionally, regardless of the source.
                ty: Some(Type::Int),
                value: Some(Expr {
                    kind: ExprKind::Cast(Box::new(desugar_expr(value, scope)), Type::Int),
                    line, col, len: 0,
                }),
                is_lazy: false,
                line,
                col,
            });
        }
    }
    out
}

/// Handles a bare expression-statement that *re-assigns* an already-tracked
/// dynamic-shape name from a recognized shape (a `.reshape(...)` call — the
/// only recognized shape here, since a chained-for comprehension is
/// vanishingly unlikely to appear as a bare reassignment RHS): expands into
/// the buffer assignment plus one assignment per shadow binding, exactly
/// mirroring `desugar_image_volume`'s construction-sugar expansion for
/// `field = Image(data, w, h)` inside a kernel `init()`.
///
/// This is the mechanism a kernel field's `init()` uses to construct a
/// dynamic-shape field — kernel fields are pre-declared (as
/// `KernelFieldDecl`s, never a `let`), so unlike a local, there's no `let`
/// initializer for `infer_let_labeled_info`/`extract_shadow_values` to run
/// against; the shape only shows up later, as a plain assignment inside
/// `init()` (`src = s.reshape(width = w, height = h)`). Falls through to
/// ordinary single-statement desugaring for everything else, including a
/// re-assignment whose RHS isn't a recognized shape (left unresolved rather
/// than guessed, same as this module's other best-effort limitations).
fn desugar_reassign_stmt(e: Expr, scope: &mut ArrayScope) -> Vec<Stmt> {
    let extracted = if let ExprKind::Assign(lhs, rhs) = &e.kind {
        if let ExprKind::Var(name) = &lhs.kind {
            scope.get(name.as_str()).cloned().and_then(|info| {
                info.shadow_names.clone().and_then(|shadow_names| {
                    extract_shadow_values(rhs, &info.axes).map(|values| (shadow_names, values))
                })
            })
        } else {
            None
        }
    } else {
        None
    };

    let Some((shadow_names, values)) = extracted else {
        return vec![Stmt::Expr(desugar_expr(e, scope))];
    };

    let Expr { kind, line, col, len: _ } = e;
    let ExprKind::Assign(lhs, rhs) = kind else { unreachable!("matched above") };
    let ExprKind::Var(target_name) = &lhs.kind else { unreachable!("matched above: extracted is only Some for a Var lhs") };
    let target_name = target_name.clone();

    let mut out: Vec<Stmt> = match *rhs {
        // See `labeled_comp_fill_stmts`'s doc: spliced in as plain
        // statements writing directly into the field, not a `Block`
        // expression — some backends' kernel-constructor codegen doesn't
        // support a `Block` used as an rvalue.
        Expr { kind: ExprKind::LabeledArrayComp { expr, clauses }, .. } => {
            labeled_comp_fill_stmts(&target_name, *expr, clauses, scope, line, col)
        }
        other_expr => {
            let rhs_d = desugar_expr(other_expr, scope);
            vec![Stmt::Expr(Expr { kind: ExprKind::Assign(lhs, Box::new(rhs_d)), line, col, len: 0 })]
        }
    };
    for (shadow_name, value) in shadow_names.iter().zip(values.into_iter()) {
        // Explicit `as int` — see desugar_let's identical note: the source
        // expression's type isn't guaranteed (int or uint both appear in
        // real kernel `init`/method params), and the shadow field's own
        // Type::Int declaration plus `for i in 0..field.size(.axis)`'s
        // range (requires Int on both sides) both need a definite Int
        // value regardless of the source.
        let value_d = Expr {
            kind: ExprKind::Cast(Box::new(desugar_expr(value, scope)), Type::Int),
            line, col, len: 0,
        };
        out.push(Stmt::Expr(Expr {
            kind: ExprKind::Assign(
                Box::new(Expr { kind: ExprKind::Var(shadow_name.clone()), line, col, len: 0 }),
                Box::new(value_d),
            ),
            line, col, len: 0,
        }));
    }
    out
}

// ─── Statement-body walk (free functions, methods, blocks) ─────────────────

fn desugar_body(stmts: Vec<Stmt>, outer: &ArrayScope) -> Vec<Stmt> {
    let mut scope = outer.clone();
    let mut out = Vec::with_capacity(stmts.len());
    for stmt in stmts {
        out.extend(desugar_stmt(stmt, &mut scope));
    }
    out
}

fn desugar_cond_clause(c: CondClause, scope: &mut ArrayScope) -> CondClause {
    match c {
        CondClause::Let(name, e) => CondClause::Let(name, desugar_expr(e, scope)),
        CondClause::LetPat(p, e) => CondClause::LetPat(p, desugar_expr(e, scope)),
        CondClause::Expr(e) => CondClause::Expr(desugar_expr(e, scope)),
    }
}

fn desugar_stmt(stmt: Stmt, scope: &mut ArrayScope) -> Vec<Stmt> {
    match stmt {
        Stmt::Let(s) => desugar_let(s, scope).into_iter().map(Stmt::Let).collect(),
        Stmt::LetDestructure(mut s) => {
            s.value = desugar_expr(s.value, scope);
            vec![Stmt::LetDestructure(s)]
        }
        Stmt::Return(mut r) => {
            r.value = r.value.map(|v| desugar_expr(v, scope));
            vec![Stmt::Return(r)]
        }
        Stmt::Break(line, val) => vec![Stmt::Break(line, val.map(|v| desugar_expr(v, scope)))],
        Stmt::Throw(mut t) => {
            t.value = t.value.map(|v| desugar_expr(v, scope));
            vec![Stmt::Throw(t)]
        }
        Stmt::If(mut i) => {
            i.branches = i.branches.into_iter()
                .map(|(c, b)| (desugar_expr(c, scope), desugar_body(b, scope)))
                .collect();
            i.else_body = i.else_body.map(|b| desugar_body(b, scope));
            vec![Stmt::If(i)]
        }
        Stmt::IfLet(mut i) => {
            i.clauses = i.clauses.into_iter().map(|c| desugar_cond_clause(c, scope)).collect();
            i.then_body = desugar_body(i.then_body, scope);
            i.elif_branches = i.elif_branches.into_iter().map(|mut b| {
                b.clauses = b.clauses.into_iter().map(|c| desugar_cond_clause(c, scope)).collect();
                b.body = desugar_body(b.body, scope);
                b
            }).collect();
            i.else_body = i.else_body.map(|b| desugar_body(b, scope));
            vec![Stmt::IfLet(i)]
        }
        Stmt::Match(mut m) => {
            m.subject = desugar_expr(m.subject, scope);
            m.arms = m.arms.into_iter().map(|mut arm| {
                arm.guard = arm.guard.map(|g| desugar_expr(g, scope));
                arm.body = match arm.body {
                    MatchBody::Expr(e) => MatchBody::Expr(desugar_expr(e, scope)),
                    MatchBody::Block(b) => MatchBody::Block(desugar_body(b, scope)),
                };
                arm
            }).collect();
            vec![Stmt::Match(m)]
        }
        Stmt::While(mut w) => {
            w.condition = desugar_expr(w.condition, scope);
            w.body = desugar_body(w.body, scope);
            vec![Stmt::While(w)]
        }
        Stmt::WhileLet(mut w) => {
            w.body = desugar_body(w.body, scope);
            vec![Stmt::WhileLet(w)]
        }
        Stmt::DoWhile(mut d) => {
            d.body = desugar_body(d.body, scope);
            d.condition = desugar_expr(d.condition, scope);
            vec![Stmt::DoWhile(d)]
        }
        Stmt::Loop(mut l) => {
            l.body = desugar_body(l.body, scope);
            vec![Stmt::Loop(l)]
        }
        Stmt::Wait(e, line) => vec![Stmt::Wait(desugar_expr(e, scope), line)],
        Stmt::For(mut f) => {
            f.iterable = desugar_expr(f.iterable, scope);
            f.body = desugar_body(f.body, scope);
            vec![Stmt::For(f)]
        }
        Stmt::Guard(mut g) => {
            // Mirrors desugar_image_volume's own scope note: the condition
            // itself isn't rewritten, only the else body.
            g.else_body = desugar_body(g.else_body, scope);
            vec![Stmt::Guard(g)]
        }
        Stmt::Try(mut t) => {
            t.body = desugar_body(t.body, scope);
            t.catch_clauses = t.catch_clauses.into_iter().map(|mut c| {
                c.body = desugar_body(c.body, scope);
                c
            }).collect();
            vec![Stmt::Try(t)]
        }
        Stmt::Defer(body) => vec![Stmt::Defer(desugar_body(body, scope))],
        Stmt::Expr(e) => desugar_reassign_stmt(e, scope),
        Stmt::Yield(e, line) => vec![Stmt::Yield(desugar_expr(e, scope), line)],
        Stmt::Fn(mut f) => {
            f.body = desugar_body(f.body, &ArrayScope::new());
            vec![Stmt::Fn(f)]
        }
        Stmt::Struct(mut s) => {
            for m in s.methods.iter_mut() {
                let body = std::mem::take(&mut m.body);
                m.body = desugar_body(body, &ArrayScope::new());
            }
            vec![Stmt::Struct(s)]
        }
        Stmt::Mod(mut m) => {
            m.items = desugar_items(m.items, &ArrayScope::new());
            vec![Stmt::Mod(m)]
        }
        Stmt::With(mut w) => {
            w.body = desugar_body(w.body, scope);
            vec![Stmt::With(w)]
        }
        Stmt::KernelBlock(mut k) => {
            k.body = desugar_body(k.body, scope);
            vec![Stmt::KernelBlock(k)]
        }
        // Continue/Enum/Alias/Comment — nothing to rewrite.
        other => vec![other],
    }
}

// ─── Expression walk ────────────────────────────────────────────────────────

/// Row-major flat offset for `args` (a `LabeledIndex`'s labeled arguments,
/// already desugared) into `info` — `None` if any label in `args` doesn't
/// match one of `info`'s axes (left unresolved rather than guessed; the
/// checker is the place a genuine mismatch gets reported, not this pass).
fn labeled_index_offset(args: &[Arg], info: &LabeledInfo, line: usize, col: usize) -> Option<Expr> {
    let mut flat: Option<Expr> = None;
    for (i, axis) in info.axes.iter().enumerate() {
        let arg = args.iter().find(|a| a.label.as_deref() == Some(axis.label.as_str()))?;
        let idx_expr = arg.value.clone();
        let term = if i == 0 {
            idx_expr
        } else {
            let stride = (0..i).map(|j| info.axis_size_expr(j, line, col))
                .reduce(|acc, v| Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(acc), Box::new(v)), line, col, len: 0 })
                .expect("i >= 1 implies a non-empty stride range");
            Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(idx_expr), Box::new(stride)), line, col, len: 0 }
        };
        flat = Some(match flat {
            None => term,
            Some(prev) => Expr { kind: ExprKind::BinOp(BinOp::Add, Box::new(prev), Box::new(term)), line, col, len: 0 },
        });
    }
    flat
}

/// `.size(.axis)`'s resolved value for a known name — the axis's shadow
/// `Var` (dynamic shape) or its literal size expression (fixed shape).
/// `None` if `axis_label` doesn't match any of `info`'s axes.
fn resolve_size_call(info: &LabeledInfo, axis_label: &str, line: usize, col: usize) -> Option<Expr> {
    let i = info.axes.iter().position(|a| a.label == axis_label)?;
    Some(info.axis_size_expr(i, line, col))
}

fn desugar_arg(a: Arg, scope: &ArrayScope) -> Arg {
    Arg { label: a.label, value: desugar_expr(a.value, scope), spread: a.spread }
}

fn desugar_expr(e: Expr, scope: &ArrayScope) -> Expr {
    let Expr { kind, line, col, len } = e;
    let kind = match kind {
        // Always a passthrough — see this module's doc comment. Unconditional:
        // by the time anything downstream (checker aside) sees the program,
        // RelabelCast must not exist anymore.
        ExprKind::RelabelCast(inner, _) => return desugar_expr(*inner, scope),

        ExprKind::LabeledIndex(obj, args) => {
            let obj_d = desugar_expr(*obj, scope);
            let args_d: Vec<Arg> = args.into_iter().map(|a| desugar_arg(a, scope)).collect();
            if let ExprKind::Var(name) = &obj_d.kind {
                if let Some(info) = scope.get(name) {
                    if let Some(offset) = labeled_index_offset(&args_d, info, line, col) {
                        return Expr { kind: ExprKind::Index(Box::new(obj_d), Box::new(offset)), line, col, len };
                    }
                }
            }
            // Unresolvable here (see this module's doc comment) — leave the
            // node as-is; it's a compiler-internal-error panic downstream if
            // actually reached at eval/codegen time, not a silent miscompile.
            ExprKind::LabeledIndex(Box::new(obj_d), args_d)
        }

        ExprKind::MethodCall(obj, method, args) => {
            let obj_d = desugar_expr(*obj, scope);
            // `.reshape(...)`/`.flatten()` — identity at the value level.
            // Best-effort name-based recognition (no type checker has run
            // yet) — same accepted risk as Image/Volume's own bare-name
            // recognition elsewhere in this compiler.
            if method == "reshape" || method == "flatten" {
                return obj_d;
            }
            if method == "size" {
                if let ExprKind::Var(name) = &obj_d.kind {
                    if let Some(info) = scope.get(name) {
                        if let [arg] = args.as_slice() {
                            if let ExprKind::DotIdent(axis) = &arg.value.kind {
                                if let Some(resolved) = resolve_size_call(info, axis, line, col) {
                                    return resolved;
                                }
                            }
                        }
                    }
                }
            }
            let args_d = args.into_iter().map(|a| desugar_arg(a, scope)).collect();
            ExprKind::MethodCall(Box::new(obj_d), method, args_d)
        }
        ExprKind::OptionalMethodCall(obj, method, args) => {
            let obj_d = desugar_expr(*obj, scope);
            let args_d = args.into_iter().map(|a| desugar_arg(a, scope)).collect();
            ExprKind::OptionalMethodCall(Box::new(obj_d), method, args_d)
        }

        ExprKind::LabeledArrayComp { expr, clauses } => {
            return desugar_labeled_comp(*expr, clauses, scope, line, col, len);
        }

        ExprKind::BinOp(op, l, r) => ExprKind::BinOp(op, Box::new(desugar_expr(*l, scope)), Box::new(desugar_expr(*r, scope))),
        ExprKind::UnaryOp(op, v) => ExprKind::UnaryOp(op, Box::new(desugar_expr(*v, scope))),
        ExprKind::Assign(l, r) => ExprKind::Assign(Box::new(desugar_expr(*l, scope)), Box::new(desugar_expr(*r, scope))),
        ExprKind::QuestionAssign(l, r) => ExprKind::QuestionAssign(Box::new(desugar_expr(*l, scope)), Box::new(desugar_expr(*r, scope))),
        ExprKind::Field(o, name) => ExprKind::Field(Box::new(desugar_expr(*o, scope)), name),
        ExprKind::OptionalField(o, name) => ExprKind::OptionalField(Box::new(desugar_expr(*o, scope)), name),
        ExprKind::Index(a, i) => ExprKind::Index(Box::new(desugar_expr(*a, scope)), Box::new(desugar_expr(*i, scope))),
        ExprKind::Call(callee, args) => ExprKind::Call(
            Box::new(desugar_expr(*callee, scope)),
            args.into_iter().map(|a| desugar_arg(a, scope)).collect(),
        ),
        ExprKind::GenericCall(callee, tys, args) => ExprKind::GenericCall(
            Box::new(desugar_expr(*callee, scope)),
            tys,
            args.into_iter().map(|a| desugar_arg(a, scope)).collect(),
        ),
        ExprKind::Pipe(l, name, args) => ExprKind::Pipe(
            Box::new(desugar_expr(*l, scope)),
            name,
            args.into_iter().map(|a| desugar_arg(a, scope)).collect(),
        ),
        ExprKind::New { arena, ctor } => ExprKind::New {
            arena: arena.map(|a| Box::new(desugar_expr(*a, scope))),
            ctor: Box::new(desugar_expr(*ctor, scope)),
        },
        ExprKind::KernelLaunch { mut config, kernel } => {
            config.block = config.block.take().map(|e| desugar_expr(e, scope));
            config.grid = config.grid.take().map(|e| desugar_expr(e, scope));
            config.after = config.after.take().map(|e| desugar_expr(e, scope));
            ExprKind::KernelLaunch { config, kernel: Box::new(desugar_expr(*kernel, scope)) }
        }
        ExprKind::TryElse(a, b) => ExprKind::TryElse(Box::new(desugar_expr(*a, scope)), Box::new(desugar_expr(*b, scope))),
        ExprKind::TryElseBlock(body, els) => ExprKind::TryElseBlock(desugar_body(body, scope), desugar_body(els, scope)),
        ExprKind::Array(elems) => ExprKind::Array(elems.into_iter().map(|x| desugar_expr(x, scope)).collect()),
        ExprKind::ArrayFill { value, count } => ExprKind::ArrayFill {
            value: Box::new(desugar_expr(*value, scope)),
            count: Box::new(desugar_expr(*count, scope)),
        },
        ExprKind::ArrayAlloc { count } => ExprKind::ArrayAlloc { count: Box::new(desugar_expr(*count, scope)) },
        ExprKind::ArrayComp { expr, var, count } => ExprKind::ArrayComp {
            expr: Box::new(desugar_expr(*expr, scope)), var, count: Box::new(desugar_expr(*count, scope)),
        },
        ExprKind::ArrayCompIter { expr, var, iter } => ExprKind::ArrayCompIter {
            expr: Box::new(desugar_expr(*expr, scope)), var, iter: Box::new(desugar_expr(*iter, scope)),
        },
        ExprKind::Tuple(xs) => ExprKind::Tuple(xs.into_iter().map(|x| desugar_expr(x, scope)).collect()),
        ExprKind::Dict(pairs) => ExprKind::Dict(
            pairs.into_iter().map(|(k, v)| (desugar_expr(k, scope), desugar_expr(v, scope))).collect(),
        ),
        ExprKind::Set(elems) => ExprKind::Set(elems.into_iter().map(|x| desugar_expr(x, scope)).collect()),
        ExprKind::Range { start, end, inclusive } => ExprKind::Range {
            start: Box::new(desugar_expr(*start, scope)), end: Box::new(desugar_expr(*end, scope)), inclusive,
        },
        ExprKind::SliceRange { start, end, inclusive } => ExprKind::SliceRange {
            start: start.map(|s| Box::new(desugar_expr(*s, scope))),
            end: end.map(|e| Box::new(desugar_expr(*e, scope))),
            inclusive,
        },
        ExprKind::Cast(inner, ty) => ExprKind::Cast(Box::new(desugar_expr(*inner, scope)), ty),
        ExprKind::Else(a, b) => ExprKind::Else(Box::new(desugar_expr(*a, scope)), Box::new(desugar_expr(*b, scope))),
        ExprKind::Closure(params, ret, body, throws, task) => {
            let body = match body {
                ClosureBody::Expr(e) => ClosureBody::Expr(Box::new(desugar_expr(*e, scope))),
                ClosureBody::Block(b) => ClosureBody::Block(desugar_body(b, scope)),
            };
            ExprKind::Closure(params, ret, body, throws, task)
        }
        ExprKind::If(mut i) => {
            i.branches = i.branches.into_iter()
                .map(|(c, b)| (desugar_expr(c, scope), desugar_body(b, scope)))
                .collect();
            i.else_body = i.else_body.map(|b| desugar_body(b, scope));
            ExprKind::If(i)
        }
        ExprKind::Match(mut m) => {
            m.subject = desugar_expr(m.subject, scope);
            m.arms = m.arms.into_iter().map(|mut arm| {
                arm.guard = arm.guard.map(|g| desugar_expr(g, scope));
                arm.body = match arm.body {
                    MatchBody::Expr(e) => MatchBody::Expr(desugar_expr(e, scope)),
                    MatchBody::Block(b) => MatchBody::Block(desugar_body(b, scope)),
                };
                arm
            }).collect();
            ExprKind::Match(m)
        }
        ExprKind::Block(stmts) => ExprKind::Block(desugar_body(stmts, scope)),
        ExprKind::Do(stmts) => ExprKind::Do(desugar_body(stmts, scope)),
        ExprKind::Loop(mut l) => {
            l.body = desugar_body(l.body, scope);
            ExprKind::Loop(l)
        }
        ExprKind::Task(e) => ExprKind::Task(Box::new(desugar_expr(*e, scope))),
        ExprKind::TaskWithTimeout(a, b) => ExprKind::TaskWithTimeout(Box::new(desugar_expr(*a, scope)), Box::new(desugar_expr(*b, scope))),
        ExprKind::JoinAll(exprs) => ExprKind::JoinAll(exprs.into_iter().map(|e| desugar_expr(e, scope)).collect()),
        ExprKind::MacroCall { name, args } => ExprKind::MacroCall {
            name, args: args.into_iter().map(|e| desugar_expr(e, scope)).collect(),
        },
        ExprKind::StringInterp(segs) => ExprKind::StringInterp(segs.into_iter().map(|seg| match seg {
            StringSegment::Expr(e) => StringSegment::Expr(Box::new(desugar_expr(*e, scope))),
            StringSegment::FormattedExpr(e, f) => StringSegment::FormattedExpr(Box::new(desugar_expr(*e, scope)), f),
            other @ StringSegment::Lit(_) => other,
        }).collect()),

        // Leaves: Int/Float/Str/Bool/Nil/Void/Var/DotIdent — nothing to
        // recurse into.
        other => other,
    };
    Expr { kind, line, col, len }
}

/// Builds the alloc-plus-nested-for-loop statements for a chained-for
/// comprehension, writing directly into `target_name` — no synthesized
/// temp, no `Block` wrapper. `clauses[0]` (axis 1) is the **innermost**
/// loop (fastest-varying, matching row-major storage — see
/// `ExprKind::LabeledArrayComp`'s own doc comment, D2), regardless of the
/// order the `for` clauses were written in.
///
/// Split out from `desugar_labeled_comp` specifically so
/// `desugar_reassign_stmt` can splice these in as plain top-level
/// statements assigning into an existing name (a kernel field) instead of
/// wrapping them in a `Block` expression — some backends' kernel-
/// constructor codegen has its own, more limited expression emitter that
/// doesn't support a `Block` used as an rvalue (confirmed via a real
/// `cargo build` failure targeting `--target cuda`, migrating
/// `whisper-boring`'s `power = Image(nfreq, nf)`-style zero-alloc
/// construction sugar to `power = [0.0 for freq = nfreq, frame = nf]`).
fn labeled_comp_fill_stmts(
    target_name: &str,
    expr: Expr,
    clauses: Vec<(String, Box<Expr>)>,
    scope: &ArrayScope,
    line: usize, col: usize,
) -> Vec<Stmt> {
    let counts_d: Vec<Expr> = clauses.iter().map(|(_, c)| desugar_expr((**c).clone(), scope)).collect();
    let total = counts_d.iter().cloned()
        .reduce(|a, b| Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(a), Box::new(b)), line, col, len: 0 })
        .expect("LabeledArrayComp always has >= 2 clauses");

    let target_var = || Expr { kind: ExprKind::Var(target_name.to_string()), line, col, len: 0 };

    // Fast path: `expr` doesn't reference any clause variable — a true
    // constant fill (guaranteed by construction for the `[value for label =
    // count, ...]` shape-only fill shorthand, whose labels are never bound
    // as usable variables in `value` in the first place — see that
    // syntax's own doc comment). Lower straight to the existing ArrayFill
    // node instead of a per-element nested-loop write.
    //
    // This isn't just an optimization: a per-element loop writing
    // `target[i] = value` is flatly wrong when `target_name` is a GPU
    // host-side buffer handle (CudaSlice/DeviceBuffer/Buffer) rather than a
    // plain indexable Vec — those types aren't directly writable element-
    // by-element from host code. ArrayFill is already correctly handled
    // everywhere a dynamic `[T]'unified` field's zero/fill-value
    // construction already is. Confirmed via a real `cargo build --target
    // cuda` failure otherwise (`img[...] = 0.0` on a `CudaSlice<f64>`,
    // which doesn't implement `IndexMut`).
    let referenced = crate::transpiler::helpers::collect_var_names(&expr);
    if clauses.iter().all(|(var_name, _)| !referenced.contains(var_name)) {
        let fill = Expr {
            kind: ExprKind::ArrayFill { value: Box::new(desugar_expr(expr, scope)), count: Box::new(total) },
            line, col, len: 0,
        };
        return vec![Stmt::Expr(Expr {
            kind: ExprKind::Assign(Box::new(target_var()), Box::new(fill)),
            line, col, len: 0,
        })];
    }

    let alloc = Stmt::Expr(Expr {
        kind: ExprKind::Assign(
            Box::new(target_var()),
            Box::new(Expr { kind: ExprKind::ArrayAlloc { count: Box::new(total) }, line, col, len: 0 }),
        ),
        line, col, len: 0,
    });

    // flat_index = clauses[0].var + clauses[1].var*counts[0] + clauses[2].var*(counts[0]*counts[1]) + ...
    let mut flat: Option<Expr> = None;
    for (i, (var_name, _)) in clauses.iter().enumerate() {
        let var_expr = Expr { kind: ExprKind::Var(var_name.clone()), line, col, len: 0 };
        let term = if i == 0 {
            var_expr
        } else {
            let stride = counts_d[0..i].iter().cloned()
                .reduce(|a, b| Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(a), Box::new(b)), line, col, len: 0 })
                .expect("i >= 1 implies a non-empty stride range");
            Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(var_expr), Box::new(stride)), line, col, len: 0 }
        };
        flat = Some(match flat {
            None => term,
            Some(prev) => Expr { kind: ExprKind::BinOp(BinOp::Add, Box::new(prev), Box::new(term)), line, col, len: 0 },
        });
    }
    let flat_index = flat.expect("LabeledArrayComp always has >= 2 clauses");

    let assign = Stmt::Expr(Expr {
        kind: ExprKind::Assign(
            Box::new(Expr {
                kind: ExprKind::Index(Box::new(target_var()), Box::new(flat_index)),
                line, col, len: 0,
            }),
            Box::new(desugar_expr(expr, scope)),
        ),
        line, col, len: 0,
    });

    // Wrap innermost-out: clauses[0]'s loop wraps `assign` directly (so it's
    // innermost); each subsequent clause wraps the previous result.
    let mut body = vec![assign];
    for (i, (var_name, _)) in clauses.iter().enumerate() {
        body = vec![Stmt::For(ForStmt {
            vars: vec![var_name.clone()],
            iterable: Expr {
                kind: ExprKind::Range {
                    start: Box::new(Expr { kind: ExprKind::Int(0), line, col, len: 0 }),
                    end: Box::new(counts_d[i].clone()),
                    inclusive: false,
                },
                line, col, len: 0,
            },
            body,
            line, col,
        })];
    }

    let mut stmts = vec![alloc];
    stmts.extend(body);
    stmts
}

/// Lowers `[expr for clauses[0].0 in ..clauses[0].1 for clauses[1].0 in
/// ..clauses[1].1 ...]` to a `Block` wrapping `labeled_comp_fill_stmts`'
/// alloc-plus-loop statements against a synthesized temp, then yielding it
/// (a `Block`'s last statement, if an expression statement, is its value).
/// Used for every EXPRESSION-position occurrence (e.g. `let a = [comp]`);
/// see `labeled_comp_fill_stmts`'s doc for why a kernel-field reassignment
/// (`desugar_reassign_stmt`) needs the unwrapped statements instead.
fn desugar_labeled_comp(
    expr: Expr,
    clauses: Vec<(String, Box<Expr>)>,
    scope: &ArrayScope,
    line: usize, col: usize, len: usize,
) -> Expr {
    let tmp = format!("__comp_{line}_{col}");
    let mut stmts = vec![Stmt::Let(LetStmt {
        binding: BindingKind::Var, var_mut: false, is_pub: false, is_static: false,
        name: tmp.clone(), ty: None, value: None, is_lazy: false, line, col,
    })];
    stmts.extend(labeled_comp_fill_stmts(&tmp, expr, clauses, scope, line, col));
    stmts.push(Stmt::Expr(Expr { kind: ExprKind::Var(tmp), line, col, len: 0 }));
    Expr { kind: ExprKind::Block(stmts), line, col, len }
}

#[cfg(test)]
mod tests;
