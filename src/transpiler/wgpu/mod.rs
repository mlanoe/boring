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
use crate::ast::{Program, Item, KernelDecl, KernelFieldDecl, Type, Expr, ExprKind, BinOp, UnaryOp, ConstExpr, LabeledAxis};

mod device;
mod host;

// ─── Public output type ───────────────────────────────────────────────────────

pub struct WgpuOutput {
    /// Rust host source (src/main.rs).
    pub host_rs: String,
    /// WGSL device source (shaders/main.wgsl). Uses the real-subgroup
    /// `gpu.warp.*` mapping when any kernel uses it (see `device::WarpMode`);
    /// identical to any prior single-module output otherwise.
    pub device_wgsl: String,
    /// Second WGSL module (shaders/main_emulated.wgsl) using the
    /// shared-memory-emulated `gpu.warp.*` mapping, for adapters lacking
    /// `wgpu::Features::SUBGROUP`. `Some(..)` only when some kernel uses
    /// `gpu.warp.*`; `None` otherwise (no second file is written).
    pub device_wgsl_emulated: Option<String>,
    /// Names of all `kernel` struct declarations found in the program.
    pub kernel_names: Vec<String>,
    /// Generated Cargo.toml content (no build.rs — wgpu/naga compile WGSL at runtime).
    pub cargo_toml: String,
    /// Errors accumulated while transpiling non-kernel code through the general pass
    /// (see `transpile_wgpu`'s `general_out`) — e.g. an unsupported kernel constructor
    /// call, or top-level `task`/stream usage. Callers must check this and report
    /// instead of writing out `host_rs`/`device_wgsl`, exactly like the std target's
    /// own `TranspileOutput::errors` (see `main.rs`'s `report_transpile_errors`) --
    /// otherwise these errors are silently dropped instead of surfaced.
    pub errors: Vec<crate::transpiler::TranspileError>,
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

    let (device_wgsl, device_wgsl_emulated) = device::emit_device_wgsl(program, &effective_kernels);

    // Non-kernel code (regular fn/struct/enum/dict logic, including the user's own
    // `def main()`, if any) is transpiled by the SAME general pipeline the std/Rust
    // target uses, with `gpu_kernels` set so it special-cases kernel construction,
    // `kernel:` dispatch, and kernel 'unified-field reads wherever they appear --
    // including inside ordinary function bodies (see `transpiler::emit_kernel`).
    // The general pipeline's own `fn main()`/`#[tokio::main]` handling only
    // triggers on a function literally named "main", so rename the boring
    // program's entry point before feeding it through -- this project's own
    // `fn main()`/`async fn async_main()` (below) is the real entry point,
    // and calls the renamed function once the GPU device/queue are ready.
    let renamed_program = crate::transpiler::rename_top_level_main(program, "boring_main");
    let general_config = crate::transpiler::TranspileConfig {
        gpu_kernels: effective_kernels.clone(),
        is_gpu_target: true,
        // A `Screen`-using program's top-level kernel/Screen construction, dispatch, and
        // the render loop itself are entirely owned by `host::emit_screen_main` (kernel
        // instances become `__App` struct fields; see that function) -- the general pass
        // must leave that untouched rather than trying to reconstruct it a second time.
        gpu_top_level_handled_by_host: has_screen,
        ..crate::transpiler::TranspileConfig::default()
    };
    let general_out = crate::transpiler::transpile_with_config(&renamed_program, general_config);
    // `boring_main` exists either because the user's own `def main():` was renamed to it
    // above (visible as an `Item::Fn` in `renamed_program`), or because the general pass
    // synthesized one from bare top-level statements/non-const `let`s (invisible here --
    // it exists only in `general_out.code` -- so `general_out.gpu_main_emitted` is the
    // only way to know; see `emit_program_items`). Shared with cuda/metal, which need the
    // identical derivation -- see `crate::transpiler::detect_boring_main`.
    let (has_boring_main, boring_main_throws) = crate::transpiler::detect_boring_main(&renamed_program, &general_out);

    let host_rs = host::emit_host_rs(program, &kernel_names, &effective_kernels, &general_out.code, has_boring_main, boring_main_throws, device_wgsl_emulated.is_some());
    let cargo_toml  = emit_cargo_toml(stem, version, has_screen);

    WgpuOutput { host_rs, device_wgsl, device_wgsl_emulated, kernel_names, cargo_toml, errors: general_out.errors }
}

/// Does this kernel's body use `gpu.warp.*` anywhere? WGSL subgroup builtins
/// need an explicit `enable subgroups;` module directive and are only valid
/// when the adapter has `wgpu::Features::SUBGROUP` — unlike CUDA/Metal, that
/// makes their kernel-parameter/directive emission conditional rather than
/// unconditional (see `device::emit_device_wgsl` and `host::emit_host_rs`).
/// Reuses the interpreter's own AST walker (`stmts_use_gpu_warp`), which needs
/// the identical detection to decide when a kernel dispatch requires the
/// real-OS-thread barrier path — rather than re-implementing the same
/// exhaustive `Stmt`/`ExprKind` walk a second time here.
pub(super) fn kernel_uses_gpu_warp(decl: &KernelDecl) -> bool {
    decl.methods.iter().any(|m| crate::interpreter::eval_gpu::stmts_use_gpu_warp(&m.body))
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
        // LabeledArray sibling: substitute each fixed axis's const-generic
        // expression down to a literal ConstInt-equivalent, same as
        // ArrayNExpr -> ArrayN above (docs/array-multidim-proposal.md).
        // Dynamic axes (`size: None`) pass through unchanged — nothing to
        // substitute.
        Type::LabeledArray(inner, axes) => {
            let new_axes = axes.iter().map(|a| {
                let new_size = a.size.as_ref().map(|ConstExpr(boxed)| {
                    let n = eval_const_expr(boxed, subst).unwrap_or(0);
                    ConstExpr(Box::new(Expr { kind: ExprKind::Int(n), line: 0, col: 0, len: 0 }))
                });
                LabeledAxis { label: a.label.clone(), size: new_size }
            }).collect();
            Type::LabeledArray(Box::new(monomorphise_type(inner, subst)), new_axes)
        }
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
