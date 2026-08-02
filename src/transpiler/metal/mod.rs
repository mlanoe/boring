// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Boring → Metal transpiler.
//
// Entry point: `transpile_metal(program)` returns a `MetalOutput` containing:
//   - `host_rs`    — Rust source for the host binary (uses `metal` crate)
//   - `device_msl` — MSL source for the device kernels (kernels/main.metal)
//   - `cargo_toml` — Generated Cargo.toml (no build.rs needed — runtime MSL compilation)
//
// Usage:
//   boring build --target metal main.br
//   → creates main_metal/
//         src/main.rs          (host_rs)
//         kernels/main.metal   (device_msl)
//         Cargo.toml
//
// ── Architecture: general-pipeline splice + kernel-touching carve-out ─────────
//
// Mirrors `cuda::mod`'s identical architecture exactly — see that module's doc
// comment for the full rationale (kernel-touching functions/top-level code stay
// on this backend's own custom emitter, since the general pipeline's kernel-aware
// codegen is wgpu-shaped; everything else is transpiled by the general pipeline
// for full language-feature correctness). A `Screen`-using program keeps its
// entire top-level/main-building on this backend's own existing Screen-aware
// driver (mirrors wgpu's own `has_screen` carve-out) rather than attempting the
// splice for it.
//
// Known gap: this backend's own kernel-launch/struct emission (`host.rs`'s
// `emit_kernel_new`/`emit_boring_launch`/etc, kept unchanged per the same "don't
// touch what works" rule as CUDA) was NOT independently verified against the
// real `metal` crate's actual API the way CUDA's was against real cudarc 0.19.8
// (no macOS toolchain available here) — see the caller's validation report for
// exactly what check was performed instead.

use crate::ast::{Program, Item, ExprKind};

mod device;
mod host;

// ─── Public output type ───────────────────────────────────────────────────────

pub struct MetalOutput {
    /// Rust host source (src/main.rs).
    pub host_rs: String,
    /// MSL device source (kernels/main.metal).
    pub device_msl: String,
    /// Names of all `kernel` struct declarations found in the program.
    pub kernel_names: Vec<String>,
    /// Generated Cargo.toml content (no build.rs — Metal compiler is built into macOS).
    pub cargo_toml: String,
    /// Errors accumulated while transpiling non-kernel-touching code through the
    /// general pass. See `cuda::CudaOutput::errors`'s identical field.
    pub errors: Vec<crate::transpiler::TranspileError>,
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub fn transpile_metal(program: &Program, stem: &str, version: &str) -> MetalOutput {
    let kernel_names: Vec<String> = program.items.iter().filter_map(|item| {
        if let Item::Kernel(decl) = item { Some(decl.name.clone()) } else { None }
    }).collect();
    let kernel_names_set: std::collections::HashSet<String> = kernel_names.iter().cloned().collect();

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

    let device_msl = device::emit_device_msl(program);

    // See this module's doc comment for the full splice architecture.
    let kernel_touching = crate::transpiler::kernel_touching_fn_names(program, &kernel_names_set);
    let kernel_touching_structs = crate::transpiler::kernel_touching_struct_names(program, &kernel_names_set);
    if !kernel_touching_structs.is_empty() {
        eprintln!(
            "warning: struct(s) {:?} have a kernel-touching method -- the Metal backend's \
             general-pipeline splice doesn't support this combination (see metal::mod's doc \
             comment); these structs will keep the old by-value parameter behavior instead \
             of the fix applied to every other struct/function.",
            kernel_touching_structs
        );
    }

    // A `Screen` program keeps its ENTIRE top-level on this backend's own
    // existing driver (mirrors wgpu's own `has_screen` carve-out) -- treated
    // the same way as `top_level_kernel_touching` for the general pass's
    // `gpu_top_level_handled_by_host` flag.
    let top_level_kernel_touching = has_screen || crate::transpiler::top_level_touches_kernel(program, &kernel_names_set);

    let renamed_program = crate::transpiler::rename_top_level_main(program, "boring_main");
    let general_config = crate::transpiler::TranspileConfig {
        gpu_kernels: Vec::new(),
        is_gpu_target: true,
        gpu_top_level_handled_by_host: top_level_kernel_touching,
        ..crate::transpiler::TranspileConfig::default()
    };
    let general_out = crate::transpiler::transpile_with_config(&renamed_program, general_config);
    let (has_boring_main, boring_main_throws) = crate::transpiler::detect_boring_main(&renamed_program, &general_out);

    let strip_names: std::collections::HashSet<String> = kernel_touching.iter().map(|n| {
        if n == "main" { "boring_main".to_string() } else { n.clone() }
    }).collect();
    // Unlike `cuda::mod` (whose own prelude also imports `Arc`, causing a
    // duplicate with the general-spliced `use std::sync::Arc;`), this backend's
    // own prelude uses only fully-qualified `std::sync::Arc<...>` paths (no bare
    // `use`), so the general pass's own import doesn't need stripping here.
    let general_code = crate::transpiler::strip_top_level_fns(&general_out.code, &strip_names);

    let host_rs = host::emit_host_rs(
        program, &kernel_names, &kernel_touching, &general_code,
        has_boring_main, boring_main_throws, top_level_kernel_touching,
    );
    let cargo_toml = emit_cargo_toml(stem, version, has_screen);

    MetalOutput { host_rs, device_msl, kernel_names, cargo_toml, errors: general_out.errors }
}

// ─── Cargo.toml generation ────────────────────────────────────────────────────

fn emit_cargo_toml(stem: &str, version: &str, has_screen: bool) -> String {
    // No build.rs needed: MSL is compiled at runtime via newLibraryWithSource.
    // The Metal compiler is built into macOS — no external toolchain required.
    // `objc` is unconditional (not just when `Screen` is present): every
    // program's `__boring_metal_flush` reads the real `NSError` off a failed
    // command buffer via `objc::msg_send!` to classify the failure, not just
    // the display path.
    let extra_deps = if has_screen {
        "winit = \"0.28\"\nobjc = \"0.2\"\ncore-graphics = \"0.23\"\n"
    } else {
        "objc = \"0.2\"\n"
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
metal = "0.29"
{extra_deps}"#,
        stem = stem,
        version = version,
        extra_deps = extra_deps,
    )
}
