// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Boring → wgpu transpiler.
//
// Entry point: `transpile_wgpu(program, stem, version)` returns a `WgpuOutput`
// containing:
//   - `host_rs`    — Rust source for the host binary (uses `wgpu` crate)
//   - `device_wgsl` — WGSL source for the compute shaders (shaders/main.wgsl)
//   - `cargo_toml` — Generated Cargo.toml (no build.rs — naga compiles WGSL at runtime)
//
// Usage:
//   boring build --target wgpu main.br
//   → creates main_wgpu/
//         src/main.rs          (host_rs)
//         shaders/main.wgsl    (device_wgsl)
//         Cargo.toml

use std::collections::HashMap;
use crate::ast::{Program, Item, KernelDecl, KernelFieldDecl, Type, Expr, ExprKind, BinOp, UnaryOp};

mod device;
mod host;

// ─── Public output type ───────────────────────────────────────────────────────

pub struct WgpuOutput {
    /// Rust host source (src/main.rs).
    pub host_rs: String,
    /// WGSL device source (shaders/main.wgsl).
    pub device_wgsl: String,
    /// Names of all `kernel` struct declarations found in the program.
    pub kernel_names: Vec<String>,
    /// Generated Cargo.toml content (no build.rs — wgpu/naga compile WGSL at runtime).
    pub cargo_toml: String,
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub fn transpile_wgpu(program: &Program, stem: &str, version: &str) -> WgpuOutput {
    let kernel_names: Vec<String> = program.items.iter().filter_map(|item| {
        if let Item::Kernel(decl) = item { Some(decl.name.clone()) } else { None }
    }).collect();

    // Resolve generic kernel instantiations → monomorphised KernelDecls.
    // Non-generic kernels pass through unchanged; generic ones are specialised
    // once per unique set of concrete type arguments found in the program.
    let effective_kernels = resolve_effective_kernels(program);

    let has_screen = program.items.iter().any(|item| {
        if let Item::Let(s) = item {
            if let Some(val) = &s.value {
                if let ExprKind::Call(callee, _) = &val.kind {
                    if let ExprKind::Var(name) = &callee.kind {
                        return name == "Screen";
                    }
                }
            }
        }
        false
    });

    let device_wgsl = device::emit_device_wgsl(program, &effective_kernels);
    let host_rs     = host::emit_host_rs(program, &kernel_names, &effective_kernels);
    let cargo_toml  = emit_cargo_toml(stem, version, has_screen);

    WgpuOutput { host_rs, device_wgsl, kernel_names, cargo_toml }
}

// ─── Monomorphisation ─────────────────────────────────────────────────────────

/// For each kernel declaration, return the list of concrete (monomorphised) decls to emit.
/// Non-generic kernels → one entry (unchanged).
/// Generic kernels → one entry per unique instantiation found in the program.
pub(super) fn resolve_effective_kernels(program: &Program) -> Vec<KernelDecl> {
    let mut result = Vec::new();
    for item in &program.items {
        if let Item::Kernel(decl) = item {
            if decl.type_params.is_empty() {
                result.push(decl.clone());
            } else {
                // Collect all concrete arg lists for this kernel.
                let mut seen: Vec<Vec<i64>> = Vec::new();
                for inst in collect_instantiations(program, &decl.name) {
                    if !seen.contains(&inst) {
                        seen.push(inst.clone());
                        let subst = build_subst(decl, &inst);
                        result.push(monomorphise(decl, &inst, &subst));
                    }
                }
            }
        }
    }
    result
}

/// Scan every top-level `let`/`var` statement for `Name<arg, ...>()` calls
/// that match the given kernel name; return each unique concrete arg list.
fn collect_instantiations(program: &Program, kernel_name: &str) -> Vec<Vec<i64>> {
    let mut found = Vec::new();
    for item in &program.items {
        let val_expr = match item {
            Item::Let(s) => s.value.as_ref(),
            Item::Stmt(crate::ast::Stmt::Let(s)) => s.value.as_ref(),
            _ => None,
        };
        if let Some(expr) = val_expr {
            if let ExprKind::GenericCall(callee, type_args, _) = &expr.kind {
                if let ExprKind::Var(name) = &callee.kind {
                    if name == kernel_name {
                        let args: Vec<i64> = type_args.iter().map(|t| match t {
                            Type::ConstInt(n) => *n,
                            _ => 0,
                        }).collect();
                        found.push(args);
                    }
                }
            }
        }
    }
    found
}

/// Build a `name → concrete_value` substitution map from a kernel's `type_params`
/// and a matching list of concrete integer arguments.
fn build_subst(decl: &KernelDecl, args: &[i64]) -> HashMap<String, i64> {
    decl.type_params.iter().enumerate().filter_map(|(i, param)| {
        // Const-generic params are encoded as `"$N:i64"` or `"$N:usize"`.
        if let Some(rest) = param.strip_prefix('$') {
            if let Some((name, _ty)) = rest.split_once(':') {
                let val = args.get(i).copied().unwrap_or(0);
                return Some((name.to_string(), val));
            }
        }
        None
    }).collect()
}

/// Compute the monomorphised name for a generic kernel given concrete args.
/// `Blur` + `[3]` → `"Blur_3"`, `GameOfLife` + `[64, 64]` → `"GameOfLife_64_64"`.
/// Non-generic kernels (empty `type_params`) keep their original name.
pub(super) fn monomorphised_name(decl: &KernelDecl, args: &[i64]) -> String {
    if decl.type_params.is_empty() || args.is_empty() {
        return decl.name.clone();
    }
    let suffix: Vec<String> = args.iter().map(|n| n.to_string()).collect();
    format!("{}_{}", decl.name, suffix.join("_"))
}

/// Clone a kernel decl, substituting every `ArrayNExpr` with a concrete `ArrayN`.
/// The resulting decl has `type_params = []` — it is fully specialised,
/// and its `name` is the monomorphised name (e.g. `"Blur_3"`).
fn monomorphise(decl: &KernelDecl, args: &[i64], subst: &HashMap<String, i64>) -> KernelDecl {
    let fields = decl.fields.iter().map(|f| KernelFieldDecl {
        ty: monomorphise_type(&f.ty, subst),
        ..f.clone()
    }).collect();
    KernelDecl {
        name: monomorphised_name(decl, args),
        fields,
        type_params: vec![],
        where_clause: vec![],
        ..decl.clone()
    }
}

fn monomorphise_type(ty: &Type, subst: &HashMap<String, i64>) -> Type {
    match ty {
        Type::ArrayNExpr(inner, ce) => {
            let n = eval_const_expr(&ce.0, subst).unwrap_or(0) as usize;
            Type::ArrayN(Box::new(monomorphise_type(inner, subst)), n)
        }
        Type::Array(inner) => Type::Array(Box::new(monomorphise_type(inner, subst))),
        Type::ArrayN(inner, n) => Type::ArrayN(Box::new(monomorphise_type(inner, subst)), *n),
        other => other.clone(),
    }
}

/// Evaluate a compile-time arithmetic expression given a substitution map.
/// Supports: integer literals, const param references, +, -, *, /.
pub(super) fn eval_const_expr(expr: &Expr, subst: &HashMap<String, i64>) -> Option<i64> {
    match &expr.kind {
        ExprKind::Int(n) => Some(*n),
        ExprKind::Var(name) => subst.get(name).copied(),
        ExprKind::BinOp(op, l, r) => {
            let lv = eval_const_expr(l, subst)?;
            let rv = eval_const_expr(r, subst)?;
            match op {
                BinOp::Add => Some(lv + rv),
                BinOp::Sub => Some(lv - rv),
                BinOp::Mul => Some(lv * rv),
                BinOp::Div if rv != 0 => Some(lv / rv),
                BinOp::Rem if rv != 0 => Some(lv % rv),
                _ => None,
            }
        }
        ExprKind::UnaryOp(UnaryOp::Neg, inner) => {
            eval_const_expr(inner, subst).map(|v| -v)
        }
        _ => None,
    }
}

// ─── Cargo.toml generation ────────────────────────────────────────────────────

fn emit_cargo_toml(stem: &str, version: &str, has_screen: bool) -> String {
    // No build.rs needed: WGSL is compiled at runtime via wgpu/naga.
    // No external GPU toolkit required — works on DX12, Vulkan, and Metal.
    let extra_deps = if has_screen {
        "winit = \"0.30\"\n"
    } else {
        ""
    };
    format!(
        r#"[package]
name = "{stem}"
version = "{version}"
edition = "2024"

[[bin]]
name = "{stem}"
path = "src/main.rs"

[dependencies]
wgpu = "22"
bytemuck = {{ version = "1", features = ["derive"] }}
pollster = "0.3"
{extra_deps}"#,
        stem = stem,
        version = version,
        extra_deps = extra_deps,
    )
}
