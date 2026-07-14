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

use crate::ast::{Program, Item, ExprKind};

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

    let device_wgsl = device::emit_device_wgsl(program);
    let host_rs     = host::emit_host_rs(program, &kernel_names);
    let cargo_toml  = emit_cargo_toml(stem, version, has_screen);

    WgpuOutput { host_rs, device_wgsl, kernel_names, cargo_toml }
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
