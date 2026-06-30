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

use crate::ast::{Program, Item};

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
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub fn transpile_metal(program: &Program, stem: &str, version: &str) -> MetalOutput {
    let kernel_names: Vec<String> = program.items.iter().filter_map(|item| {
        if let Item::Kernel(decl) = item { Some(decl.name.clone()) } else { None }
    }).collect();

    let device_msl = device::emit_device_msl(program);
    let host_rs    = host::emit_host_rs(program, &kernel_names);
    let cargo_toml = emit_cargo_toml(stem, version);

    MetalOutput { host_rs, device_msl, kernel_names, cargo_toml }
}

// ─── Cargo.toml generation ────────────────────────────────────────────────────

fn emit_cargo_toml(stem: &str, version: &str) -> String {
    // No build.rs needed: MSL is compiled at runtime via newLibraryWithSource.
    // The Metal compiler is built into macOS — no external toolchain required.
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
"#,
        stem = stem,
        version = version,
    )
}
