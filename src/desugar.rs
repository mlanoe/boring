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

//! AST desugaring passes that run once, right after parsing, before any
//! checker/interpreter/transpiler/validator ever sees the program — every
//! consumer downstream sees an already-simplified AST and needs no new
//! per-target machinery to support what a pass here handles.

use crate::ast::*;
use std::collections::HashMap;

/// Phase 3 of docs/image-volume-types.md's dynamic-shape extension.
///
/// Rewrites every dynamic-shape `Image<T>`/`Volume<T>` kernel field (recognized
/// via `Type::as_image_volume` returning an empty `dims` slice — see its doc
/// comment in `ast/mod.rs`) into a plain flat-buffer field plus 2 (`Image`) or
/// 3 (`Volume`) synthesized `uint` "shadow" fields carrying width/height/depth
/// at runtime — exactly the pattern `TransposeKernel`
/// (`whisper-boring/src/math_gpu.br`) already hand-rolls with separate
/// `rows`/`cols` fields (see docs/image-volume-types.md's "Problem statement"),
/// generated here instead of hand-typed by every kernel author.
///
/// Also rewrites the `field = Image(data, w, h)` / `field = Volume(data, x, y,
/// z)` construction sugar — an assignment to the *original* field name, RHS a
/// call to the bare name `"Image"`/`"Volume"`, neither of which is ever a real
/// bound value or function anywhere in the language — into plain assignments:
/// one to the desugared buffer field, one per shadow field, positionally.
///
/// Runs once, on the whole `Program`, before checker/interpreter/transpiler/
/// validator ever see it (every real entry point in `main.rs` calls this right
/// after a successful parse) — every consumer downstream only ever sees
/// ordinary `[T]` + `uint` kernel fields it already fully supports; no new
/// per-target machinery needed for construction. (`.at()`/`.width()`/
/// `.height()`/`.depth()` calls and grid inference on a desugared field are
/// NOT handled by this pass — that's Phase 5/6's job; a dynamic-shape field
/// can be constructed after this pass but not yet indexed via `.at()`.)
///
/// Fixed-shape `Image<T,C,R>`/`Volume<T,X,Y,Z>` fields (non-empty dims) and
/// malformed ones (wrong arity, non-`ConstInt` dims) are untouched here — this
/// pass only ever matches an empty `dims` slice; everything else still flows
/// through Phase 1/2's existing recognition and validation unchanged.
pub fn desugar_image_volume(mut program: Program) -> Program {
    for item in &mut program.items {
        desugar_item(item);
    }
    program
}

fn desugar_item(item: &mut Item) {
    match item {
        Item::Kernel(decl) => desugar_kernel_decl(decl),
        Item::Mod(m) => {
            for i in &mut m.items {
                desugar_item(i);
            }
        }
        _ => {}
    }
}

/// If `ty` is a dynamic-shape `Image`/`Volume` (empty dims — see
/// `Type::as_image_volume`'s doc comment), returns the cloned element type and
/// the shadow-axis suffixes to synthesize (`["w","h"]` for `Image`, `["w","h",
/// "d"]` for `Volume`). `None` for anything else — fixed-shape, malformed, or
/// not an `Image`/`Volume` at all.
fn dynamic_image_volume_axes(ty: &Type) -> Option<(Type, &'static [&'static str])> {
    let (elem, dims) = ty.as_image_volume()?;
    if !dims.is_empty() {
        return None;
    }
    let is_volume = matches!(ty, Type::Generic(n, _) if n == "Volume");
    let axes: &'static [&'static str] = if is_volume { &["w", "h", "d"] } else { &["w", "h"] };
    Some((elem.clone(), axes))
}

/// The synthesized shadow field name for a given buffer field name + axis —
/// e.g. `shadow_field_name("img", "w")` -> `"__img_w"`. Reserved-prefix scheme:
/// a leading `__` is not currently rejected at the lexer level (see
/// docs/image-volume-types.md's "Open questions — resolved" §2, which flags
/// this as a follow-up hardening, not yet implemented), so this is a
/// convention, not yet an enforced guarantee against a same-named user field.
fn shadow_field_name(field_name: &str, axis: &str) -> String {
    format!("__{field_name}_{axis}")
}

fn desugar_kernel_decl(decl: &mut KernelDecl) {
    // Original field name -> its shadow field names, in declared order (w, h[, d]).
    let mut shadow_names: HashMap<String, Vec<String>> = HashMap::new();
    // Original field name -> (element type, shadow field names) — the construction
    // sugar additionally needs the element type, to build a zero/`fill =` literal
    // when no data buffer is supplied (see `build_construction_assigns`).
    let mut field_shapes: FieldShapes = HashMap::new();

    let mut new_fields = Vec::with_capacity(decl.fields.len());
    for field in decl.fields.drain(..) {
        match dynamic_image_volume_axes(&field.ty) {
            Some((elem, axes)) => {
                let names: Vec<String> =
                    axes.iter().map(|ax| shadow_field_name(&field.name, ax)).collect();
                shadow_names.insert(field.name.clone(), names.clone());
                field_shapes.insert(field.name.clone(), (elem.clone(), names.clone()));

                // The buffer field: same name/binding/qualifier, now a plain
                // dynamic flat array — identical shape to what a hand-written
                // `[T]` field of the same qualifier already gets everywhere.
                new_fields.push(KernelFieldDecl {
                    name: field.name,
                    binding: field.binding,
                    qual: field.qual,
                    ty: Type::Array(Box::new(elem)),
                    default: field.default,
                    line: field.line,
                    col: field.col,
                });
                // Shadow fields: `let uint`, `'const` — read-only for the
                // kernel's lifetime, matching `TransposeKernel`'s plain
                // `let int rows`/`let int cols` (see this module's doc comment).
                for name in names {
                    new_fields.push(KernelFieldDecl {
                        name,
                        binding: FieldBinding::Let,
                        qual: GpuQual::Const,
                        ty: Type::Uint,
                        default: None,
                        line: field.line,
                        col: field.col,
                    });
                }
            }
            None => new_fields.push(field),
        }
    }
    decl.fields = new_fields;

    if shadow_names.is_empty() {
        return; // No dynamic-shape fields on this kernel — nothing else to rewrite.
    }

    // Order matters: lower `.at()`/`.width()`/`.height()`/`.depth()` calls
    // FIRST, over the whole body — this also rewrites any such call nested
    // inside a construction-sugar call's own arguments (e.g. `dst =
    // Image([0.0 for ..src.width()*src.height()], src.width(),
    // src.height())`, sizing `dst` from `src`'s own shape). Only then split
    // the (now fully-lowered) construction-sugar assignment into plain
    // per-field assignments — that pass only needs to recognize the
    // statement's own top-level shape, not worry about what's nested inside
    // its argument expressions anymore.
    for init in &mut decl.inits {
        let body = lower_at_width_height_stmts(std::mem::take(&mut init.body), &shadow_names);
        init.body = desugar_stmts(body, &field_shapes);
    }
    for method in &mut decl.methods {
        let body = lower_at_width_height_stmts(std::mem::take(&mut method.body), &shadow_names);
        method.body = desugar_stmts(body, &field_shapes);
    }
}

// ── `.at()`/`.width()`/`.height()`/`.depth()` lowering (Phase 5) ────────────
//
// Mirrors `interpreter::eval_gpu::lower_image_volume_methods`'s walk (same
// scope: every statement/expression form realistic inside a kernel method
// body — GuardCond, Match-as-expr, Block/Do exprs, and closures are left
// unrewritten, same as that pass, for the same reason: none are meaningful
// inside a kernel body today). The key difference from that pass: this one
// runs here, once, for every real target (interpreter AND all four
// transpiler backends alike — see this module's top-level doc comment),
// producing shadow-*field* references (`Var("__img_w")`) instead of
// compile-time `Int` literals, since a dynamic shape's dimensions are only
// known at runtime. Because this runs before any of those consumers ever see
// the program, none of them need their own dynamic-shape-aware `.at()`
// lowering — they only ever see the plain `Index`/`Var` expressions left
// behind, exactly like the fixed-shape interpreter path already does for its
// own (compile-time-constant) case.

type ShadowFieldNames = HashMap<String, Vec<String>>;

fn lower_at_width_height_stmts(stmts: Vec<Stmt>, shadows: &ShadowFieldNames) -> Vec<Stmt> {
    stmts.into_iter().map(|s| lower_at_width_height_stmt(s, shadows)).collect()
}

fn lower_at_width_height_stmt(stmt: Stmt, shadows: &ShadowFieldNames) -> Stmt {
    use crate::ast::MatchBody;
    match stmt {
        Stmt::Let(mut s) => {
            s.value = s.value.map(|v| lower_at_width_height_expr(v, shadows));
            Stmt::Let(s)
        }
        Stmt::LetDestructure(mut s) => {
            s.value = lower_at_width_height_expr(s.value, shadows);
            Stmt::LetDestructure(s)
        }
        Stmt::Return(mut r) => {
            r.value = r.value.map(|v| lower_at_width_height_expr(v, shadows));
            Stmt::Return(r)
        }
        Stmt::Break(line, val) => Stmt::Break(line, val.map(|v| lower_at_width_height_expr(v, shadows))),
        Stmt::Throw(mut t) => {
            t.value = t.value.map(|v| lower_at_width_height_expr(v, shadows));
            Stmt::Throw(t)
        }
        Stmt::If(mut i) => {
            i.branches = i.branches.into_iter()
                .map(|(c, b)| (lower_at_width_height_expr(c, shadows), lower_at_width_height_stmts(b, shadows)))
                .collect();
            i.else_body = i.else_body.map(|b| lower_at_width_height_stmts(b, shadows));
            Stmt::If(i)
        }
        Stmt::IfLet(mut i) => {
            i.clauses = i.clauses.into_iter().map(|c| lower_at_width_height_cond_clause(c, shadows)).collect();
            i.then_body = lower_at_width_height_stmts(i.then_body, shadows);
            i.elif_branches = i.elif_branches.into_iter().map(|mut b| {
                b.clauses = b.clauses.into_iter().map(|c| lower_at_width_height_cond_clause(c, shadows)).collect();
                b.body = lower_at_width_height_stmts(b.body, shadows);
                b
            }).collect();
            i.else_body = i.else_body.map(|b| lower_at_width_height_stmts(b, shadows));
            Stmt::IfLet(i)
        }
        Stmt::Match(mut m) => {
            m.subject = lower_at_width_height_expr(m.subject, shadows);
            m.arms = m.arms.into_iter().map(|mut arm| {
                arm.guard = arm.guard.map(|g| lower_at_width_height_expr(g, shadows));
                arm.body = match arm.body {
                    MatchBody::Expr(e) => MatchBody::Expr(lower_at_width_height_expr(e, shadows)),
                    MatchBody::Block(b) => MatchBody::Block(lower_at_width_height_stmts(b, shadows)),
                };
                arm
            }).collect();
            Stmt::Match(m)
        }
        Stmt::While(mut w) => {
            w.condition = lower_at_width_height_expr(w.condition, shadows);
            w.body = lower_at_width_height_stmts(w.body, shadows);
            Stmt::While(w)
        }
        Stmt::WhileLet(mut w) => {
            w.body = lower_at_width_height_stmts(w.body, shadows);
            Stmt::WhileLet(w)
        }
        Stmt::DoWhile(mut d) => {
            d.body = lower_at_width_height_stmts(d.body, shadows);
            d.condition = lower_at_width_height_expr(d.condition, shadows);
            Stmt::DoWhile(d)
        }
        Stmt::Loop(mut l) => {
            l.body = lower_at_width_height_stmts(l.body, shadows);
            Stmt::Loop(l)
        }
        Stmt::Wait(e, line) => Stmt::Wait(lower_at_width_height_expr(e, shadows), line),
        Stmt::For(mut f) => {
            f.iterable = lower_at_width_height_expr(f.iterable, shadows);
            f.body = lower_at_width_height_stmts(f.body, shadows);
            Stmt::For(f)
        }
        Stmt::Guard(mut g) => {
            g.else_body = lower_at_width_height_stmts(g.else_body, shadows);
            Stmt::Guard(g)
        }
        Stmt::Try(mut t) => {
            t.body = lower_at_width_height_stmts(t.body, shadows);
            t.catch_clauses = t.catch_clauses.into_iter().map(|mut c| {
                c.body = lower_at_width_height_stmts(c.body, shadows);
                c
            }).collect();
            Stmt::Try(t)
        }
        Stmt::Defer(body) => Stmt::Defer(lower_at_width_height_stmts(body, shadows)),
        Stmt::Expr(e) => Stmt::Expr(lower_at_width_height_expr(e, shadows)),
        Stmt::Yield(e, line) => Stmt::Yield(lower_at_width_height_expr(e, shadows), line),
        // Fn/Struct/Enum/Mod/Alias/Comment/KernelBlock/With/Continue — no
        // sub-expressions relevant to `.at(...)` rewriting; left unchanged.
        other => other,
    }
}

fn lower_at_width_height_cond_clause(c: CondClause, shadows: &ShadowFieldNames) -> CondClause {
    match c {
        CondClause::Let(name, e) => CondClause::Let(name, lower_at_width_height_expr(e, shadows)),
        CondClause::LetPat(p, e) => CondClause::LetPat(p, lower_at_width_height_expr(e, shadows)),
        CondClause::Expr(e) => CondClause::Expr(lower_at_width_height_expr(e, shadows)),
    }
}

fn lower_at_width_height_arg(a: Arg, shadows: &ShadowFieldNames) -> Arg {
    let mut a = a;
    a.value = lower_at_width_height_expr(a.value, shadows);
    a
}

fn lower_at_width_height_expr(e: Expr, shadows: &ShadowFieldNames) -> Expr {
    let Expr { kind, line, col, len } = e;
    let kind = match kind {
        ExprKind::MethodCall(obj, method, args) => {
            let field_shadows = match &obj.kind {
                ExprKind::Var(name) => shadows.get(name),
                _ => None,
            };
            match field_shadows {
                Some(field_shadows) => {
                    match rewrite_dynamic_shape_call(&obj, &method, args, field_shadows, shadows, line, col, len) {
                        Ok(rewritten) => return rewritten,
                        Err(args_back) => ExprKind::MethodCall(
                            Box::new(lower_at_width_height_expr(*obj, shadows)),
                            method,
                            args_back.into_iter().map(|a| lower_at_width_height_arg(a, shadows)).collect(),
                        ),
                    }
                }
                None => ExprKind::MethodCall(
                    Box::new(lower_at_width_height_expr(*obj, shadows)),
                    method,
                    args.into_iter().map(|a| lower_at_width_height_arg(a, shadows)).collect(),
                ),
            }
        }
        ExprKind::BinOp(op, l, r) => ExprKind::BinOp(op, Box::new(lower_at_width_height_expr(*l, shadows)), Box::new(lower_at_width_height_expr(*r, shadows))),
        ExprKind::UnaryOp(op, v) => ExprKind::UnaryOp(op, Box::new(lower_at_width_height_expr(*v, shadows))),
        ExprKind::Assign(l, r) => ExprKind::Assign(Box::new(lower_at_width_height_expr(*l, shadows)), Box::new(lower_at_width_height_expr(*r, shadows))),
        ExprKind::QuestionAssign(l, r) => ExprKind::QuestionAssign(Box::new(lower_at_width_height_expr(*l, shadows)), Box::new(lower_at_width_height_expr(*r, shadows))),
        ExprKind::Field(o, name) => ExprKind::Field(Box::new(lower_at_width_height_expr(*o, shadows)), name),
        ExprKind::Index(a, i) => ExprKind::Index(Box::new(lower_at_width_height_expr(*a, shadows)), Box::new(lower_at_width_height_expr(*i, shadows))),
        ExprKind::Call(callee, args) => ExprKind::Call(Box::new(lower_at_width_height_expr(*callee, shadows)), args.into_iter().map(|a| lower_at_width_height_arg(a, shadows)).collect()),
        ExprKind::GenericCall(callee, tys, args) => ExprKind::GenericCall(Box::new(lower_at_width_height_expr(*callee, shadows)), tys, args.into_iter().map(|a| lower_at_width_height_arg(a, shadows)).collect()),
        ExprKind::Pipe(l, name, args) => ExprKind::Pipe(Box::new(lower_at_width_height_expr(*l, shadows)), name, args.into_iter().map(|a| lower_at_width_height_arg(a, shadows)).collect()),
        ExprKind::TryElse(a, b) => ExprKind::TryElse(Box::new(lower_at_width_height_expr(*a, shadows)), Box::new(lower_at_width_height_expr(*b, shadows))),
        ExprKind::Array(elems) => ExprKind::Array(elems.into_iter().map(|x| lower_at_width_height_expr(x, shadows)).collect()),
        ExprKind::ArrayFill { value, count } => ExprKind::ArrayFill { value: Box::new(lower_at_width_height_expr(*value, shadows)), count: Box::new(lower_at_width_height_expr(*count, shadows)) },
        ExprKind::ArrayAlloc { count } => ExprKind::ArrayAlloc { count: Box::new(lower_at_width_height_expr(*count, shadows)) },
        ExprKind::ArrayComp { expr, var, count } => ExprKind::ArrayComp { expr: Box::new(lower_at_width_height_expr(*expr, shadows)), var, count: Box::new(lower_at_width_height_expr(*count, shadows)) },
        ExprKind::ArrayCompIter { expr, var, iter } => ExprKind::ArrayCompIter { expr: Box::new(lower_at_width_height_expr(*expr, shadows)), var, iter: Box::new(lower_at_width_height_expr(*iter, shadows)) },
        ExprKind::Tuple(xs) => ExprKind::Tuple(xs.into_iter().map(|x| lower_at_width_height_expr(x, shadows)).collect()),
        ExprKind::Range { start, end, inclusive } => ExprKind::Range { start: Box::new(lower_at_width_height_expr(*start, shadows)), end: Box::new(lower_at_width_height_expr(*end, shadows)), inclusive },
        ExprKind::Cast(inner, ty) => ExprKind::Cast(Box::new(lower_at_width_height_expr(*inner, shadows)), ty),
        ExprKind::Else(a, b) => ExprKind::Else(Box::new(lower_at_width_height_expr(*a, shadows)), Box::new(lower_at_width_height_expr(*b, shadows))),
        ExprKind::OptionalField(o, name) => ExprKind::OptionalField(Box::new(lower_at_width_height_expr(*o, shadows)), name),
        ExprKind::OptionalMethodCall(o, name, args) => ExprKind::OptionalMethodCall(Box::new(lower_at_width_height_expr(*o, shadows)), name, args.into_iter().map(|a| lower_at_width_height_arg(a, shadows)).collect()),
        ExprKind::If(mut i) => {
            i.branches = i.branches.into_iter()
                .map(|(c, b)| (lower_at_width_height_expr(c, shadows), lower_at_width_height_stmts(b, shadows)))
                .collect();
            i.else_body = i.else_body.map(|b| lower_at_width_height_stmts(b, shadows));
            ExprKind::If(i)
        }
        // TryElseBlock/Match/Block/Do/Closure/New/KernelLaunch/Dict/Set/
        // StringInterp/SliceRange/DotIdent — not realistic inside a kernel
        // body's numeric hot path (see this pass's doc comment above).
        other => other,
    };
    Expr { kind, line, col, len }
}

/// If `method` is `.at`/`.width`/`.height`/`.depth` on a dynamic-shape
/// `Image`/`Volume` field (`field_shadows` = that field's own shadow-field
/// names, in `[w, h[, d]]` order), returns `Ok` with the lowered `Var`/`Index`
/// expression. Otherwise returns `Err(args)`, handing the (unmodified) args
/// back to the caller so it can fall back to generic recursion, keeping the
/// original `MethodCall`. `all_shadows` is threaded through so `.at(...)`'s
/// index arguments get lowered too, in case one of them itself calls
/// `.width()`/`.height()` on some (possibly different) dynamic field — see
/// this pass's ordering note in `desugar_kernel_decl`.
#[allow(clippy::too_many_arguments)]
fn rewrite_dynamic_shape_call(
    obj: &Expr, method: &str, args: Vec<Arg>, field_shadows: &[String], all_shadows: &ShadowFieldNames,
    line: usize, col: usize, len: usize,
) -> Result<Expr, Vec<Arg>> {
    use crate::ast::BinOp;
    let var_expr = |name: &str| Expr { kind: ExprKind::Var(name.to_string()), line, col, len };
    let dim = match method {
        "width"  => 0,
        "height" => 1,
        "depth"  => 2,
        "at" => usize::MAX, // sentinel: handled separately below
        _ => return Err(args),
    };
    if dim != usize::MAX {
        return match field_shadows.get(dim) {
            Some(s) => Ok(var_expr(s)),
            None => Err(args), // e.g. `.depth()` on an `Image` (only 2 shadows)
        };
    }
    // .at(a0, a1, ...) → a0 + a1*shadows[0] + a2*(shadows[0]*shadows[1]) + ... —
    // row-major, same formula as `transpiler::helpers::image_volume_at_index`,
    // but multiplying by shadow-field *variables* instead of `ConstInt` literals.
    // `.take(field_shadows.len())` caps a too-long arg list instead of indexing
    // out of bounds — matches the fixed-shape lowering's own looseness on a
    // wrong-arity `.at()` call (see `eval_gpu.rs`'s `rewrite_image_volume_call`).
    let arg_exprs: Vec<Expr> = args.into_iter()
        .map(|a| lower_at_width_height_expr(a.value, all_shadows))
        .take(field_shadows.len())
        .collect();
    let mut flat: Option<Expr> = None;
    for (i, a) in arg_exprs.into_iter().enumerate() {
        let term = if i == 0 {
            a
        } else {
            let stride = field_shadows[0..i].iter()
                .map(|s| var_expr(s))
                .reduce(|acc, v| Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(acc), Box::new(v)), line, col, len })
                .expect("i >= 1 implies field_shadows[0..i] is non-empty");
            Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(a), Box::new(stride)), line, col, len }
        };
        flat = Some(match flat {
            None => term,
            Some(prev) => Expr { kind: ExprKind::BinOp(BinOp::Add, Box::new(prev), Box::new(term)), line, col, len },
        });
    }
    let Some(flat) = flat else { return Err(vec![]) }; // `.at()` with zero args — nothing sensible to build
    Ok(Expr { kind: ExprKind::Index(Box::new(obj.clone()), Box::new(flat)), line, col, len })
}

/// Field name -> (element type, shadow field names) — see `build_construction_assigns`.
type FieldShapes = HashMap<String, (Type, Vec<String>)>;

fn desugar_stmts(stmts: Vec<Stmt>, field_shapes: &FieldShapes) -> Vec<Stmt> {
    stmts.into_iter().flat_map(|s| desugar_stmt(s, field_shapes)).collect()
}

/// Rewrites one statement, possibly into several — the construction-sugar
/// case expands `field = Image(...)` into one assignment per buffer/shadow
/// field (see `build_construction_assigns` for the three recognized forms).
/// Any other statement shape (including `if`/`for`/nested blocks that might
/// contain the sugar deeper inside) passes through unchanged: the sugar is
/// only ever meaningful as a direct top-level assignment RHS — see this
/// module's doc comment — so there is nothing to recurse into here.
fn desugar_stmt(stmt: Stmt, field_shapes: &FieldShapes) -> Vec<Stmt> {
    let Stmt::Expr(expr) = &stmt else { return vec![stmt] };
    let ExprKind::Assign(lhs, rhs) = &expr.kind else { return vec![stmt] };
    let ExprKind::Var(field_name) = &lhs.kind else { return vec![stmt] };
    let Some((elem_ty, shadows)) = field_shapes.get(field_name) else { return vec![stmt] };
    let ExprKind::Call(callee, args) = &rhs.kind else { return vec![stmt] };
    let ExprKind::Var(callee_name) = &callee.kind else { return vec![stmt] };
    if callee_name != "Image" && callee_name != "Volume" {
        return vec![stmt];
    }
    match build_construction_assigns(field_name, elem_ty, shadows, args, expr.line, expr.col) {
        Some(assigns) => assigns,
        // Arity matches neither recognized form (see below) — leave the
        // statement unrewritten rather than guess; it'll fail downstream the
        // same way any other genuinely-malformed call already does ("Image"
        // is never a real bound name), no worse than before this pass existed.
        None => vec![stmt],
    }
}

/// Three recognized forms for `field = Image(...)` / `field = Volume(...)`,
/// dispatched by positional arity (a trailing `fill = ...` labeled arg, if
/// present, is set aside first and doesn't count towards it):
///
/// 1. `Image(data, w, h)` (`positional.len() == shadows.len() + 1`) — wraps
///    an existing buffer the caller already built; `field` gets `data`
///    directly, no allocation happens here.
/// 2. `Image(w, h)` (`positional.len() == shadows.len()`, no `fill =`) —
///    allocates a new buffer of length `w * h`, zero-filled — the dynamic-shape
///    equivalent of a fixed-shape `Image<T,C,R>` field needing no `init()`
///    assignment at all.
/// 3. `Image(w, h, fill = value)` (same arity as 2, `fill =` present) —
///    same allocation, filled with `value` instead of zero — e.g. to skip a
///    redundant zero-fill immediately overwritten by every kernel thread, or
///    to pick a non-zero starting value.
///
/// `Volume` follows the same three forms one axis further. Returns `None` for
/// any other positional arity — not a recognized form; see `desugar_stmt`.
fn build_construction_assigns(
    field_name: &str,
    elem_ty: &Type,
    shadows: &[String],
    args: &[Arg],
    line: usize,
    col: usize,
) -> Option<Vec<Stmt>> {
    let mk_assign = |lhs_name: String, rhs: Expr| -> Stmt {
        let lhs = Expr { kind: ExprKind::Var(lhs_name), line, col, len: 0 };
        Stmt::Expr(Expr { kind: ExprKind::Assign(Box::new(lhs), Box::new(rhs)), line, col, len: 0 })
    };

    let mut fill_arg: Option<&Arg> = None;
    let mut positional: Vec<&Arg> = Vec::with_capacity(args.len());
    for a in args {
        if fill_arg.is_none() && a.label.as_deref() == Some("fill") {
            fill_arg = Some(a);
        } else {
            positional.push(a);
        }
    }

    let mut out = Vec::with_capacity(1 + shadows.len());

    if positional.len() == shadows.len() + 1 {
        // Form 1: `Image(data, w, h)` — wrap the caller's own buffer as-is.
        out.push(mk_assign(field_name.to_string(), positional[0].value.clone()));
        for (shadow, a) in shadows.iter().zip(&positional[1..]) {
            out.push(mk_assign(shadow.clone(), a.value.clone()));
        }
    } else if positional.len() == shadows.len() {
        // Forms 2/3: `Image(w, h)` or `Image(w, h, fill = value)` — no data
        // buffer supplied; allocate one here, sized from the dims themselves.
        let count = dims_product_expr(&positional, line, col);
        let fill_value = match fill_arg {
            Some(a) => a.value.clone(),
            None => zero_expr_for_type(elem_ty, line, col),
        };
        let data_expr = Expr {
            kind: ExprKind::ArrayFill { value: Box::new(fill_value), count: Box::new(count) },
            line, col, len: 0,
        };
        out.push(mk_assign(field_name.to_string(), data_expr));
        for (shadow, a) in shadows.iter().zip(positional.iter()) {
            out.push(mk_assign(shadow.clone(), a.value.clone()));
        }
    } else {
        return None;
    }
    Some(out)
}

/// `positional[0] * positional[1] * ...` — the total element count for a
/// zero/`fill =`-allocated buffer (form 2/3 above). Always at least 2 args in
/// practice (`Image` has 2 shadow axes, `Volume` 3), but written generally.
fn dims_product_expr(dims: &[&Arg], line: usize, col: usize) -> Expr {
    dims.iter().map(|a| a.value.clone())
        .reduce(|acc, v| Expr { kind: ExprKind::BinOp(BinOp::Mul, Box::new(acc), Box::new(v)), line, col, len: 0 })
        .expect("build_construction_assigns only calls this with shadows.len() >= 1 dims")
}

/// The zero-value literal for `ty`, as a Boring AST expression — used to fill
/// a `[<zero> for ..count]` buffer when `Image(w, h)`/`Volume(x, y, z)`
/// allocates without a `fill =` value. Mirrors
/// `interpreter::eval_gpu::zero_value`'s per-type mapping, but produces an AST
/// literal (`ExprKind`) rather than a runtime `Value` — this runs at desugar
/// time, before either the interpreter or any transpiler backend exists to
/// evaluate one. `float`/`bool` as a *generic type argument* (i.e.
/// `Image<float>`'s own `T`) parse as `Type::Named("float"/"bool")`, not
/// `Type::Float`/`Type::Bool` — both spellings are handled here (see this
/// crate's other `Type::as_image_volume` callers for the same distinction).
fn zero_expr_for_type(ty: &Type, line: usize, col: usize) -> Expr {
    let kind = match ty {
        Type::Float => ExprKind::Float(0.0),
        Type::Bool => ExprKind::Bool(false),
        Type::Named(n) if n == "float" => ExprKind::Float(0.0),
        Type::Named(n) if n == "bool" => ExprKind::Bool(false),
        // Every int/uint width (signed, unsigned, and all sized variants), plus
        // anything unrecognized, defaults to a bare `0` — the same fallback
        // `zero_value` uses for its own catch-all arm, coerced by the
        // surrounding `[T]` context the same way any other bare `0` literal
        // already is.
        _ => ExprKind::Int(0),
    };
    Expr { kind, line, col, len: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser::parse;
    use crate::interpreter::Value;

    fn desugared(src: &str) -> Program {
        let tokens = lex(src).expect("lex error");
        let program = parse(tokens).expect("parse error");
        desugar_image_volume(program)
    }

    fn only_kernel(program: &Program) -> &KernelDecl {
        program.items.iter().find_map(|i| match i {
            Item::Kernel(k) => Some(k),
            _ => None,
        }).expect("expected exactly one kernel decl")
    }

    #[test]
    fn dynamic_image_field_becomes_buffer_plus_two_shadow_fields() {
        let src = r#"
kernel Tile:
    mut Image<float>'unified img
    def ():
        img[0] = 1.0
"#;
        let program = desugared(src);
        let k = only_kernel(&program);
        let names: Vec<&str> = k.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["img", "__img_w", "__img_h"]);
        // `float` as a generic type argument parses to `Type::Named("float")`, not
        // `Type::Float` (a pre-existing parser quirk, not introduced by this pass) —
        // exactly what a hand-written `[float]'unified` field's element type already
        // is, which is the invariant that matters here: the desugared buffer field
        // must be structurally identical to a hand-written one of the same element type.
        assert!(matches!(&k.fields[0].ty, Type::Array(inner) if matches!(&**inner, Type::Named(n) if n == "float")),
            "unexpected buffer field type: {:?}", k.fields[0].ty);
        assert!(matches!(k.fields[1].ty, Type::Uint));
        assert!(matches!(k.fields[1].binding, FieldBinding::Let));
        assert!(matches!(k.fields[1].qual, GpuQual::Const));
        assert!(matches!(k.fields[2].ty, Type::Uint));
    }

    #[test]
    fn dynamic_volume_field_becomes_buffer_plus_three_shadow_fields() {
        let src = r#"
kernel Cube:
    mut Volume<float>'unified vol
    def ():
        vol[0] = 1.0
"#;
        let program = desugared(src);
        let k = only_kernel(&program);
        let names: Vec<&str> = k.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["vol", "__vol_w", "__vol_h", "__vol_d"]);
    }

    #[test]
    fn fixed_shape_field_is_untouched() {
        let src = r#"
kernel Tile:
    mut Image<float, 16, 16>'actor tile
    def ():
        tile.at(0, 0) = 1.0
"#;
        let program = desugared(src);
        let k = only_kernel(&program);
        assert_eq!(k.fields.len(), 1);
        assert_eq!(k.fields[0].name, "tile");
        assert!(matches!(&k.fields[0].ty, Type::Generic(n, _) if n == "Image"));
    }

    #[test]
    fn construction_sugar_expands_to_plain_assignments() {
        let src = r#"
kernel Tile:
    let [float]'global   src
    mut Image<float>'unified dst
    init([float]'global s, int w, int h):
        src = s
        dst = Image([0.0 for ..w * h], w, h)
    def ():
        dst[0] = src[0]
"#;
        let program = desugared(src);
        let k = only_kernel(&program);
        let init = &k.inits[0];
        // src = s ; dst = [...] ; __dst_w = w ; __dst_h = h  (4 statements, sugar expanded to 3)
        assert_eq!(init.body.len(), 4);
        let names_assigned: Vec<&str> = init.body.iter().map(|s| {
            let Stmt::Expr(e) = s else { panic!("expected Stmt::Expr") };
            let ExprKind::Assign(lhs, _) = &e.kind else { panic!("expected Assign") };
            let ExprKind::Var(name) = &lhs.kind else { panic!("expected Var lhs") };
            name.as_str()
        }).collect();
        assert_eq!(names_assigned, vec!["src", "dst", "__dst_w", "__dst_h"]);
        // The second statement's RHS is the plain array-fill expr (`[0.0 for
        // ..w*h]`), not a leftover `Image(...)` call — the sugar call itself
        // must be gone.
        let Stmt::Expr(dst_assign) = &init.body[1] else { unreachable!() };
        let ExprKind::Assign(_, rhs) = &dst_assign.kind else { unreachable!() };
        assert!(matches!(rhs.kind, ExprKind::ArrayFill { .. }), "expected the array-fill expr, got {:?}", rhs.kind);
    }

    #[test]
    fn no_data_constructor_zero_allocates() {
        // `Image(w, h)` — no data buffer supplied — allocates `[0.0 for ..w*h]`
        // itself, the dynamic-shape equivalent of a fixed-shape field needing
        // no `init()` assignment at all.
        let src = r#"
kernel Tile:
    mut Image<float>'unified dst
    init(int w, int h):
        dst = Image(w, h)
    def ():
        dst[0] = 0.0
"#;
        let program = desugared(src);
        let init = &only_kernel(&program).inits[0];
        assert_eq!(init.body.len(), 3, "expected dst=[...] ; __dst_w=w ; __dst_h=h");
        let Stmt::Expr(dst_assign) = &init.body[0] else { unreachable!() };
        let ExprKind::Assign(lhs, rhs) = &dst_assign.kind else { unreachable!() };
        assert!(matches!(&lhs.kind, ExprKind::Var(n) if n == "dst"));
        let ExprKind::ArrayFill { value, count } = &rhs.kind else {
            panic!("expected an array-fill expr, got {:?}", rhs.kind)
        };
        assert!(matches!(value.kind, ExprKind::Float(f) if f == 0.0), "expected a zero fill value, got {:?}", value.kind);
        assert!(matches!(count.kind, ExprKind::BinOp(BinOp::Mul, ..)), "expected w*h, got {:?}", count.kind);

        let Stmt::Expr(w_assign) = &init.body[1] else { unreachable!() };
        let ExprKind::Assign(lhs, rhs) = &w_assign.kind else { unreachable!() };
        assert!(matches!(&lhs.kind, ExprKind::Var(n) if n == "__dst_w"));
        assert!(matches!(&rhs.kind, ExprKind::Var(n) if n == "w"));
    }

    #[test]
    fn no_data_constructor_zero_allocates_int_field_as_bare_zero_literal() {
        // Element type `int` (not `float`) — the zero literal must be `Int(0)`,
        // not `Float(0.0)`.
        let src = r#"
kernel Tile:
    mut Image<int>'unified dst
    init(int w, int h):
        dst = Image(w, h)
    def ():
        dst[0] = 0
"#;
        let program = desugared(src);
        let init = &only_kernel(&program).inits[0];
        let Stmt::Expr(dst_assign) = &init.body[0] else { unreachable!() };
        let ExprKind::Assign(_, rhs) = &dst_assign.kind else { unreachable!() };
        let ExprKind::ArrayFill { value, .. } = &rhs.kind else { panic!("expected an array-fill expr") };
        assert!(matches!(value.kind, ExprKind::Int(0)), "expected a zero fill value, got {:?}", value.kind);
    }

    #[test]
    fn fill_labeled_constructor_uses_the_given_value_instead_of_zero() {
        // `Image(w, h, fill = 7.0)` — allocates, but fills with 7.0, not 0.0.
        let src = r#"
kernel Tile:
    mut Image<float>'unified dst
    init(int w, int h):
        dst = Image(w, h, fill = 7.0)
    def ():
        dst[0] = 0.0
"#;
        let program = desugared(src);
        let init = &only_kernel(&program).inits[0];
        assert_eq!(init.body.len(), 3);
        let Stmt::Expr(dst_assign) = &init.body[0] else { unreachable!() };
        let ExprKind::Assign(_, rhs) = &dst_assign.kind else { unreachable!() };
        let ExprKind::ArrayFill { value, .. } = &rhs.kind else { panic!("expected an array-fill expr") };
        assert!(matches!(value.kind, ExprKind::Float(f) if f == 7.0), "expected the fill=7.0 value, got {:?}", value.kind);
    }

    #[test]
    fn volume_no_data_constructor_zero_allocates_with_three_axis_product() {
        let src = r#"
kernel Cube:
    mut Volume<float>'unified vol
    init(int x, int y, int z):
        vol = Volume(x, y, z)
    def ():
        vol[0] = 0.0
"#;
        let program = desugared(src);
        let init = &only_kernel(&program).inits[0];
        assert_eq!(init.body.len(), 4, "expected vol=[...] ; __vol_w=x ; __vol_h=y ; __vol_d=z");
        let Stmt::Expr(vol_assign) = &init.body[0] else { unreachable!() };
        let ExprKind::Assign(_, rhs) = &vol_assign.kind else { unreachable!() };
        let ExprKind::ArrayFill { count, .. } = &rhs.kind else { panic!("expected an array-fill expr") };
        // x * y * z — a nested BinOp::Mul product of all three axes.
        assert!(matches!(count.kind, ExprKind::BinOp(BinOp::Mul, ..)), "expected x*y*z, got {:?}", count.kind);
    }

    #[test]
    fn malformed_arity_is_left_unrewritten() {
        // Neither `shadows.len()` (2) nor `shadows.len()+1` (3) — not a
        // recognized form; the statement must survive untouched rather than
        // being guessed at (it'll fail downstream the normal way, same as
        // any other call to an undefined name).
        let src = r#"
kernel Tile:
    mut Image<float>'unified dst
    init(int a, int b, int c, int d):
        dst = Image(a, b, c, d)
    def ():
        dst[0] = 0.0
"#;
        let program = desugared(src);
        let init = &only_kernel(&program).inits[0];
        assert_eq!(init.body.len(), 1, "malformed call should not be split");
        let Stmt::Expr(e) = &init.body[0] else { unreachable!() };
        let ExprKind::Assign(_, rhs) = &e.kind else { unreachable!() };
        assert!(matches!(&rhs.kind, ExprKind::Call(callee, _) if matches!(&callee.kind, ExprKind::Var(n) if n == "Image")),
            "expected the original unrewritten Image(...) call, got {:?}", rhs.kind);
    }

    #[test]
    fn end_to_end_zero_alloc_and_fill_constructors_through_the_real_pipeline() {
        let src = r#"
kernel TwoImages:
    mut Image<float>'unified zeroed
    mut Image<float>'unified filled

    init(int w, int h):
        zeroed = Image(w, h)
        filled = Image(w, h, fill = 9.0)

    def ():
        pass

let width = 3
let height = 2
mut k = TwoImages(width, height)
kernel:
    k(block = (16, 16), grid = (1, 1))
let z = k.zeroed
let f = k.filled
with z:
    with f:
        for v in z:
            print "{v}"
        for v in f:
            print "{v}"
"#;
        let tokens = lex(src).expect("lex error");
        let program = parse(tokens).expect("parse error");
        let program = desugar_image_volume(program);

        let check_errs: Vec<String> = crate::checker::check(&program).errors.into_iter().map(|e| e.message).collect();
        assert!(check_errs.is_empty(), "expected no checker errors post-desugar, got {check_errs:?}");

        let mut interp = crate::interpreter::Interpreter::new();
        interp.exec_program(&program).expect("runtime error");
        let zeroed = interp.global.borrow().get("z").unwrap_or(Value::Nil);
        let filled = interp.global.borrow().get("f").unwrap_or(Value::Nil);
        assert_eq!(zeroed, Value::Array(vec![Value::Float(0.0); 6].into()));
        assert_eq!(filled, Value::Array(vec![Value::Float(9.0); 6].into()));
    }

    // ── `.at()`/`.width()`/`.height()`/`.depth()` lowering (Phase 5) ────────

    fn body_exprs(program: &Program) -> Vec<Expr> {
        only_kernel(program).methods[0].body.iter().map(|s| {
            let Stmt::Expr(e) = s else { panic!("expected Stmt::Expr, got {s:?}") };
            e.clone()
        }).collect()
    }

    #[test]
    fn width_and_height_calls_lower_to_shadow_field_vars() {
        let src = r#"
kernel Tile:
    mut Image<float>'unified img
    def ():
        let w = img.width()
        let h = img.height()
"#;
        let program = desugared(src);
        let stmts = &only_kernel(&program).methods[0].body;
        for (stmt, expected) in stmts.iter().zip(["__img_w", "__img_h"]) {
            let Stmt::Let(s) = stmt else { panic!("expected Stmt::Let, got {stmt:?}") };
            let value = s.value.as_ref().expect("let binding should have a value");
            assert!(matches!(&value.kind, ExprKind::Var(n) if n == expected),
                "expected Var({expected:?}), got {:?}", value.kind);
        }
    }

    #[test]
    fn depth_on_a_2_axis_image_is_left_unrewritten() {
        // `Image` only has 2 shadow fields (w, h) — `.depth()` has nothing to
        // rewrite to, so the call must survive as-is rather than being
        // silently dropped or panicking.
        let src = r#"
kernel Tile:
    mut Image<float>'unified img
    def ():
        img.depth()
"#;
        let program = desugared(src);
        let exprs = body_exprs(&program);
        assert!(matches!(&exprs[0].kind, ExprKind::MethodCall(obj, m, _)
            if m == "depth" && matches!(&obj.kind, ExprKind::Var(n) if n == "img")));
    }

    #[test]
    fn image_at_call_lowers_to_row_major_index_over_shadow_width() {
        // `img.at(x, y)` -> `img[x + y * __img_w]` — row-major, matching
        // `transpiler::helpers::image_volume_at_index`'s formula for the
        // fixed-shape case, but with a shadow-field `Var` stride instead of a
        // `ConstInt` literal.
        let src = r#"
kernel Tile:
    mut Image<float>'unified img
    def ():
        img.at(x, y)
"#;
        let program = desugared(src);
        let exprs = body_exprs(&program);
        let ExprKind::Index(obj, idx) = &exprs[0].kind else {
            panic!("expected Index, got {:?}", exprs[0].kind)
        };
        assert!(matches!(&obj.kind, ExprKind::Var(n) if n == "img"));
        let ExprKind::BinOp(BinOp::Add, lhs, rhs) = &idx.kind else {
            panic!("expected `x + y*__img_w`, got {:?}", idx.kind)
        };
        assert!(matches!(&lhs.kind, ExprKind::Var(n) if n == "x"));
        let ExprKind::BinOp(BinOp::Mul, y, w) = &rhs.kind else {
            panic!("expected `y * __img_w`, got {:?}", rhs.kind)
        };
        assert!(matches!(&y.kind, ExprKind::Var(n) if n == "y"));
        assert!(matches!(&w.kind, ExprKind::Var(n) if n == "__img_w"));
    }

    #[test]
    fn volume_at_call_includes_width_times_height_stride_for_the_third_axis() {
        // `vol.at(x, y, z)` -> `vol[x + y*__vol_w + z*(__vol_w*__vol_h)]`.
        let src = r#"
kernel Cube:
    mut Volume<float>'unified vol
    def ():
        vol.at(x, y, z)
"#;
        let program = desugared(src);
        let exprs = body_exprs(&program);
        let ExprKind::Index(_, idx) = &exprs[0].kind else {
            panic!("expected Index, got {:?}", exprs[0].kind)
        };
        // Outermost is `(x + y*w) + z*(w*h)`.
        let ExprKind::BinOp(BinOp::Add, xy_term, z_term) = &idx.kind else {
            panic!("expected an outer Add, got {:?}", idx.kind)
        };
        assert!(matches!(&xy_term.kind, ExprKind::BinOp(BinOp::Add, ..)), "expected `x + y*w` on the left, got {:?}", xy_term.kind);
        let ExprKind::BinOp(BinOp::Mul, z, wh) = &z_term.kind else {
            panic!("expected `z * (w*h)`, got {:?}", z_term.kind)
        };
        assert!(matches!(&z.kind, ExprKind::Var(n) if n == "z"));
        let ExprKind::BinOp(BinOp::Mul, w, h) = &wh.kind else {
            panic!("expected `__vol_w * __vol_h`, got {:?}", wh.kind)
        };
        assert!(matches!(&w.kind, ExprKind::Var(n) if n == "__vol_w"));
        assert!(matches!(&h.kind, ExprKind::Var(n) if n == "__vol_h"));
    }

    #[test]
    fn at_as_an_assignment_target_lowers_too() {
        // `img.at(x, y) = v` — `.at()` on the LHS of an assignment must lower
        // exactly like the read case, since `Index` is itself a valid
        // assignment target (plain array index-assignment already works).
        let src = r#"
kernel Tile:
    mut Image<float>'unified img
    def ():
        img.at(0, 0) = 1.0
"#;
        let program = desugared(src);
        let exprs = body_exprs(&program);
        let ExprKind::Assign(lhs, _) = &exprs[0].kind else {
            panic!("expected Assign, got {:?}", exprs[0].kind)
        };
        assert!(matches!(&lhs.kind, ExprKind::Index(..)), "expected the LHS to be lowered to Index, got {:?}", lhs.kind);
    }

    #[test]
    fn end_to_end_construct_write_and_read_via_at_and_width() {
        // Full pipeline (lex -> parse -> desugar -> checker::check -> interpret),
        // exercising construction sugar (Phase 3) AND `.at()`/`.width()`/
        // `.height()` (Phase 5) together on a genuinely runtime-shaped buffer.
        let src = r#"
kernel Fill:
    mut Image<float>'unified img

    init([float]'unified data, int w, int h):
        img = Image(data, w, h)

    def ():
        let x = gpu.thread.x
        let y = gpu.thread.y
        img.at(x, y) = img.at(x, y) + float(img.width()) + float(img.height())

let data = [0.0, 0.0, 0.0, 0.0]
mut k = Fill(data, 2, 2)
kernel:
    k(block = (2, 2))
let _result = k.img
"#;
        let tokens = lex(src).expect("lex error");
        let program = parse(tokens).expect("parse error");
        let program = desugar_image_volume(program);

        let check_errs: Vec<String> = crate::checker::check(&program).errors.into_iter().map(|e| e.message).collect();
        assert!(check_errs.is_empty(), "expected no checker errors post-desugar, got {check_errs:?}");

        let mut interp = crate::interpreter::Interpreter::new();
        interp.exec_program(&program).expect("runtime error");
        let result = interp.global.borrow().get("_result").unwrap_or(Value::Nil);
        // Every cell starts at 0.0, then gets width(2) + height(2) added once = 4.0.
        assert_eq!(
            result,
            Value::Array(vec![Value::Float(4.0), Value::Float(4.0), Value::Float(4.0), Value::Float(4.0)].into())
        );
    }

    #[test]
    fn end_to_end_through_the_real_pipeline_order_run_file_uses() {
        // Mirrors `main::run_file`'s exact order: lex -> parse -> desugar ->
        // checker::check -> interpret. Confirms the checker's Phase 2 rejection
        // no longer fires post-desugar (it never sees `Image<T>` at all once
        // this pass has rewritten it away), and that the desugared field
        // behaves as a genuinely usable, constructible `[T]'unified` buffer —
        // `.at()`/`.width()` are NOT exercised here, since lowering those is
        // Phase 5's job, not Phase 3's.
        let src = r#"
kernel Tile:
    mut Image<float>'unified img

    init([float]'unified data, int w, int h):
        img = Image(data, w, h)

    def ():
        let i = gpu.thread.x
        img[i] = img[i] + 1.0

let data = [1.0, 2.0, 3.0, 4.0]
mut k = Tile(data, 2, 2)
kernel:
    k(block = 4)
let _result = k.img
"#;
        let tokens = lex(src).expect("lex error");
        let program = parse(tokens).expect("parse error");
        let program = desugar_image_volume(program);

        let check_errs: Vec<String> = crate::checker::check(&program).errors.into_iter().map(|e| e.message).collect();
        assert!(check_errs.is_empty(), "expected no checker errors post-desugar, got {check_errs:?}");

        let mut interp = crate::interpreter::Interpreter::new();
        interp.exec_program(&program).expect("runtime error");
        let result = interp.global.borrow().get("_result").unwrap_or(Value::Nil);
        assert_eq!(
            result,
            Value::Array(vec![Value::Float(2.0), Value::Float(3.0), Value::Float(4.0), Value::Float(5.0)].into())
        );
    }

    #[test]
    fn kernel_with_no_dynamic_fields_is_unaffected() {
        let src = r#"
kernel Plain:
    mut [float]'unified out
    def ():
        out[0] = 1.0
"#;
        let program = desugared(src);
        let k = only_kernel(&program);
        assert_eq!(k.fields.len(), 1);
        assert_eq!(k.fields[0].name, "out");
    }
}
