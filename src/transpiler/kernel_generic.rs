// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Const-generic kernel monomorphisation as a program-to-program pass, shared by the
// Metal, CUDA and ROCm targets (wgpu predates it and specialises inside its own
// `resolve_effective_kernels`, from top-level `let`s only).
//
// `kernel Name<int D>: ...` plus every `Name<0>(...)` / `Name<1>(...)` construction
// found ANYWHERE in the program (top level, function bodies, struct/ext methods) becomes
// one ordinary non-generic kernel per unique argument list (`Name_0`, `Name_1`), with
// every use of the const parameter in the kernel (field types, defaults, init/method
// bodies) replaced by the literal. The construction sites are rewritten to plain
// `Name_0(...)` calls, so everything downstream (device emitter, host emitter, general
// splice) only ever sees monomorphic kernels, and the device compiler folds
// `if D == 0` branches like any other constant condition. A generic kernel with no
// instantiation is dropped (nothing is emitted for it).
//
// Only `int`/`uint` (`$N:i64` / `$N:usize`) parameters are supported; a type parameter
// or an argument that is not an integer literal is a transpile error. Type-generic structs and
// helper functions are `device_generic.rs`'s (they run before this pass).

use std::collections::HashMap;

use crate::ast::*;
use crate::ast_walk::Walker;
use crate::transpiler::TranspileError;

/// `Blur` + `[3, 1]` → `"Blur_3_1"`; a negative value is spelled `n<abs>`.
fn mangled(name: &str, args: &[i64]) -> String {
    let parts: Vec<String> = args.iter().map(|a| if *a < 0 { format!("n{}", -a) } else { a.to_string() }).collect();
    format!("{}_{}", name, parts.join("_"))
}

fn const_param_name(p: &str) -> Option<&str> {
    p.strip_prefix('$').and_then(|r| r.split_once(':')).map(|(n, _)| n)
}

pub(crate) fn monomorphise_generic_kernels(program: &Program) -> (Program, Vec<TranspileError>) {
    let generics: HashMap<String, KernelDecl> = program.items.iter().filter_map(|i| match i {
        Item::Kernel(k) if !k.type_params.is_empty() => Some((k.name.clone(), k.clone())),
        _ => None,
    }).collect();
    if generics.is_empty() {
        return (program.clone(), Vec::new());
    }

    let mut out = program.clone();
    let mut errors = Vec::new();
    // generic kernel name → unique argument lists, in first-seen order
    let mut insts: HashMap<String, Vec<Vec<i64>>> = HashMap::new();

    {
        let mut on_expr = |e: &mut Expr| {
            let ExprKind::GenericCall(callee, type_args, args) = &mut e.kind else { return };
            let ExprKind::Var(name) = &callee.kind else { return };
            let Some(decl) = generics.get(name) else { return };
            let mut vals = Vec::new();
            for t in type_args.iter() {
                match t {
                    Type::ConstInt(n) => vals.push(*n),
                    _ => {
                        errors.push(TranspileError::at(
                            format!("generic kernel `{}`: every argument must be an integer literal -- a type argument (`kernel K<T>`) or a non-literal value is not supported, kernels take `int`/`uint` const parameters only; keep the kernel concrete and put the generic code in a helper function or struct called with explicit type arguments (`f<float>(..)`)", name),
                            e.line, e.col));
                        return;
                    }
                }
            }
            if vals.len() != decl.type_params.len() || decl.type_params.iter().any(|p| const_param_name(p).is_none()) {
                errors.push(TranspileError::at(
                    format!("generic kernel `{}` takes {} parameter(s) and only `int`/`uint` const parameters with literal arguments are supported: a type-generic kernel (`kernel {}<T>`), a `float`/`bool` const parameter or a wrong argument count is rejected; keep the kernel concrete and put the generic code in a helper function or struct called with explicit type arguments (`f<float>(..)`)", name, decl.type_params.len(), name),
                    e.line, e.col));
                return;
            }
            let list = insts.entry(name.clone()).or_default();
            if !list.contains(&vals) { list.push(vals.clone()); }
            let new_name = mangled(name, &vals);
            let taken = std::mem::take(args);
            let line = callee.line; let col = callee.col;
            *e = Expr {
                kind: ExprKind::Call(Box::new(Expr { kind: ExprKind::Var(new_name), line, col, len: 0 }), taken),
                line: e.line, col: e.col, len: e.len,
            };
        };
        let mut on_stmt = |_: &mut Stmt| {};
        let mut w = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        for item in &mut out.items { walk_item(item, &mut w); }
    }

    let mut items = Vec::with_capacity(out.items.len());
    for item in std::mem::take(&mut out.items) {
        match item {
            Item::Kernel(k) if generics.contains_key(&k.name) => {
                for vals in insts.get(&k.name).cloned().unwrap_or_default() {
                    items.push(Item::Kernel(specialise(&k, &vals)));
                }
            }
            other => items.push(other),
        }
    }
    out.items = items;
    (out, errors)
}

pub(crate) fn walk_item(item: &mut Item, w: &mut Walker) {
    match item {
        Item::Fn(f) => w.stmts(&mut f.body),
        Item::Struct(s) => {
            for m in &mut s.methods { w.stmts(&mut m.body); }
            for i in &mut s.inits { w.stmts(&mut i.body); }
        }
        Item::Ext(x) => for m in &mut x.methods { w.stmts(&mut m.body); },
        Item::Enum(e) => for m in &mut e.methods { w.stmts(&mut m.body); },
        Item::Mod(m) => for i in &mut m.items { walk_item(i, w); },
        Item::Let(l) => if let Some(v) = &mut l.value { w.expr(v); },
        Item::Stmt(s) => w.stmt(s),
        _ => {}
    }
}

fn specialise(decl: &KernelDecl, vals: &[i64]) -> KernelDecl {
    let subst: HashMap<String, i64> = decl.type_params.iter().zip(vals)
        .filter_map(|(p, v)| const_param_name(p).map(|n| (n.to_string(), *v))).collect();
    let mut k = decl.clone();
    k.name = mangled(&decl.name, vals);
    k.type_params.clear();
    k.where_clause.clear();

    let ty = |t: &Type| crate::transpiler::wgpu::monomorphise_type(t, &subst);
    let mut on_expr = |e: &mut Expr| {
        if let ExprKind::Var(n) = &e.kind {
            if let Some(v) = subst.get(n) { e.kind = ExprKind::Int(*v); }
        }
    };
    let mut on_stmt = |s: &mut Stmt| {
        if let Stmt::Let(l) = s {
            if let Some(t) = &l.ty { l.ty = Some(ty(t)); }
        }
    };
    let mut w = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
    for f in &mut k.fields {
        f.ty = ty(&f.ty);
        if let Some(d) = &mut f.default { w.expr(d); }
    }
    for init in &mut k.inits {
        for p in &mut init.params { if let Some(t) = &p.ty { p.ty = Some(ty(t)); } }
        w.stmts(&mut init.body);
    }
    for m in &mut k.methods {
        for p in &mut m.params { if let Some(t) = &p.ty { p.ty = Some(ty(t)); } }
        w.stmts(&mut m.body);
    }
    k
}
