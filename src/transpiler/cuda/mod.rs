// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Boring → CUDA transpiler.
//
// Entry point: `transpile_cuda(program)` returns a `CudaOutput` containing:
//   - `host_rs`   — Rust source for the host binary (uses `cudarc` crate)
//   - `device_cu` — CUDA C source for the device kernels
//   - `kernel_names` — list of kernel struct names (used to populate the Cargo project)
//
// Usage:
//   boring build --target cuda main.br
//   → creates main_cuda/
//         src/main.rs         (host_rs)
//         kernels/main.cu     (device_cu)
//         build.rs
//         Cargo.toml
//
// ── Architecture: general-pipeline splice + kernel-touching carve-out ─────────
//
// `host::emit_host_rs`'s own hand-rolled `expr()`/`emit_stmt()`/`emit_fn()` used
// to re-implement Rust codegen for the ENTIRE program from scratch, independent
// of the general (std/wgpu-shared) transpiler pipeline in `transpiler::mod.rs`.
// That duplication was the root cause of a long tail of bugs found in this
// backend (closures/`.map()`, slice-range indexing, dict get/insert, `guard`/
// `throw`, tuple-destructure, `match`, enum-variant dot access, and, most
// systemically, ignoring boring's own by-ref parameter-passing contract —
// `CLAUDE.md`: "Structs, enums, arrays, dicts, sets — always passed by
// reference" — emitting every array/struct param BY VALUE instead, a real
// `E0382`/`E0308` the moment such a value is used twice or passed to a callee
// that (correctly) expects a reference).
//
// `wgpu::transpile_wgpu` avoids all of this by never re-implementing general
// Rust codegen at all: it runs the SAME general pipeline
// (`transpiler::transpile_with_config`) the plain `boring build` (std/tokio)
// target uses, with `TranspileConfig::is_gpu_target` set so that pipeline's
// OWN kernel-aware codegen (`transpiler::emit_kernel.rs`) special-cases kernel
// construction/`kernel:` dispatch/`'unified` field reads wherever they appear,
// even inside ordinary function bodies. That kernel-aware codegen is written
// entirely in terms of wgpu's own simple, INFALLIBLE kernel API (`Kernel::new
// (device, queue)`, `.dispatch(gx,gy,gz)`, `.copy_x_to_device/to_host()`) —
// which is what makes reusing it safe for wgpu, but UNSAFE to reuse verbatim
// for cuda/metal, whose real kernel dispatch is FALLIBLE and richer (cudarc's
// `Kernel::new(ctx, ...) -> Result<Self, _>`, `__boring_launch(block, grid,
// after, priority) -> Result<KernelHandle<Self>, _>` with real CUDA-stream
// dependency ordering — a feature with no wgpu equivalent at all). Passing
// `gpu_kernels` (populated) to the general pipeline for THIS backend would
// silently emit wgpu-shaped construction/dispatch calls that don't exist on
// this backend's own kernel structs.
//
// The fix implemented here: split the program into two disjoint categories.
//   - "Kernel-touching" functions (`transpiler::kernel_touching_fn_names` —
//     directly construct a `kernel` instance or contain a `kernel: ...` block;
//     e.g. math_gpu.br's `transpose_gpu`/`linear_gpu`) keep using THIS
//     backend's own custom emitter (`host::emit_fn` etc.), which knows the
//     real cudarc API. This is a small, fixed set (kernel authors' own
//     wrapper functions) — not, e.g., every function that merely CALLS one.
//   - Every other item (the overwhelming majority of any real program — plain
//     math/string/collection logic, struct methods, top-level statements) is
//     transpiled by the general pipeline, run on the WHOLE (unfiltered)
//     program so every function's signature/`throws`-ness is known for
//     correct by-ref/`?`-propagation call-site codegen — then
//     `transpiler::strip_top_level_fns` discards just the (necessarily wrong,
//     wgpu-shaped) rendered BODY text for each kernel-touching function name,
//     leaving every call site TO it (now correctly by-ref-coerced) intact.
//     This backend's own custom emitter supplies the real replacement body,
//     itself using a matching by-ref signature (`host::is_ref_worthy_type`) so
//     cross-calls in either direction type-check.
//
// Known gap: a kernel-touching STRUCT METHOD (as opposed to a free function)
// isn't supported by this split — see `kernel_touching_struct_names`'s doc.
// Not exercised by this codebase's own `.br` corpus (every kernel-touching
// construct there is a free function); detected and reported via `eprintln!`
// rather than silently mishandled if it ever occurs.

use crate::ast::{Program, Item};

mod device;
mod host;

// ─── Public output type ───────────────────────────────────────────────────────

pub struct CudaOutput {
    /// Rust host source (src/main.rs).
    pub host_rs: String,
    /// CUDA C device source (kernels/main.cu).
    pub device_cu: String,
    /// Names of all `kernel` struct declarations found in the program.
    pub kernel_names: Vec<String>,
    /// Generated build.rs content.
    pub build_rs: String,
    /// Generated Cargo.toml content.
    pub cargo_toml: String,
    /// Errors accumulated while transpiling non-kernel-touching code through
    /// the general pass (see `transpile_cuda`'s `general_out`) -- callers must
    /// check this and report instead of writing out `host_rs`/`device_cu`,
    /// exactly like the wgpu target's own `WgpuOutput::errors`.
    pub errors: Vec<crate::transpiler::TranspileError>,
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub fn transpile_cuda(program: &Program, stem: &str, version: &str) -> CudaOutput {
    // Collect all kernel names.
    let kernel_names: Vec<String> = program.items.iter().filter_map(|item| {
        if let Item::Kernel(decl) = item { Some(decl.name.clone()) } else { None }
    }).collect();
    let kernel_names_set: std::collections::HashSet<String> = kernel_names.iter().cloned().collect();

    let device_cu = device::emit_device_cu(program);

    // See this module's doc comment for the full splice architecture.
    let kernel_touching = crate::transpiler::kernel_touching_fn_names(program, &kernel_names_set);
    let kernel_touching_structs = crate::transpiler::kernel_touching_struct_names(program, &kernel_names_set);
    if !kernel_touching_structs.is_empty() {
        eprintln!(
            "warning: struct(s) {:?} have a kernel-touching method -- the CUDA backend's \
             general-pipeline splice doesn't support this combination (see cuda::mod's doc \
             comment); these structs will keep the old by-value parameter behavior instead \
             of the fix applied to every other struct/function.",
            kernel_touching_structs
        );
    }

    // Bare top-level kernel construction/dispatch (no enclosing `def main():` —
    // e.g. `examples/vector_add_gpu.br`'s `mut k = VectorAdd(...); kernel:
    // k(block=256)`, both top-level statements) can't be folded into a
    // general-pipeline-synthesized `boring_main` either, for the same reason a
    // kernel-touching FUNCTION can't be general-spliced (see this module's doc
    // comment) — so when it's present, top-level statement/let handling stays
    // entirely on this backend's own custom emitter (its pre-splice behavior,
    // which already worked for this case), and `gpu_top_level_handled_by_host`
    // tells the general pass to leave top-level alone rather than trying to
    // fold it into `boring_main` itself.
    let top_level_kernel_touching = crate::transpiler::top_level_touches_kernel(program, &kernel_names_set);

    let renamed_program = crate::transpiler::rename_top_level_main(program, "boring_main");
    let general_config = crate::transpiler::TranspileConfig {
        gpu_kernels: Vec::new(),
        is_gpu_target: true,
        gpu_top_level_handled_by_host: top_level_kernel_touching,
        ..crate::transpiler::TranspileConfig::default()
    };
    let general_out = crate::transpiler::transpile_with_config(&renamed_program, general_config);
    let (has_boring_main, boring_main_throws) = crate::transpiler::detect_boring_main(&renamed_program, &general_out);

    // A kernel-touching function keeps its original name in `general_out.code`
    // UNLESS it's the user's own `main`, which `rename_top_level_main` above
    // already renamed to `boring_main` for the general pass's benefit -- strip
    // that name instead so `host::emit_program`'s matching rename lines up.
    let strip_names: std::collections::HashSet<String> = kernel_touching.iter().map(|n| {
        if n == "main" { "boring_main".to_string() } else { n.clone() }
    }).collect();
    let general_code = crate::transpiler::strip_top_level_fns(&general_out.code, &strip_names);
    // The general pipeline unconditionally emits its own `use std::sync::Arc;`
    // whenever the "plain" code needs it (e.g. `Arc<str>` strings) -- this
    // backend's own prelude (`host::emit_prelude`) ALSO imports `Arc` (needed
    // for `Arc<CudaContext>` etc. in the kernel-struct code that comes before
    // the splice), so both together are a duplicate-import E0252, confirmed via
    // a real `cargo check`. Drop the general pass's copy; the prelude's covers
    // the whole file regardless of where either `use` line physically sits.
    let general_code = general_code.replace("use std::sync::Arc;\n", "");

    let host_rs = host::emit_host_rs(
        program, &kernel_names, &kernel_touching, &general_code,
        has_boring_main, boring_main_throws, top_level_kernel_touching,
    );
    let build_rs  = emit_build_rs();
    let cargo_toml = emit_cargo_toml(stem, version);

    CudaOutput { host_rs, device_cu, kernel_names, build_rs, cargo_toml, errors: general_out.errors }
}

// ─── build.rs generation ─────────────────────────────────────────────────────

fn emit_build_rs() -> String {
    r#"// Generated by boring build --target cuda.
// Compiles kernels/main.cu to PTX via nvcc, then makes the path available
// to main.rs through the BORING_PTX_PATH env var at compile time.

use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=kernels/main.cu");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let ptx_path = PathBuf::from(&out_dir).join("main.ptx");

    let status = Command::new("nvcc")
        .args([
            "--ptx",
            "-O2",
            "--output-file", ptx_path.to_str().unwrap(),
            "kernels/main.cu",
        ])
        .status()
        .expect("nvcc not found — install the CUDA toolkit (https://developer.nvidia.com/cuda-downloads)");

    if !status.success() {
        panic!("nvcc failed to compile kernels/main.cu");
    }

    println!("cargo:rustc-env=BORING_PTX_PATH={}", ptx_path.display());
}
"#.into()
}

// ─── Cargo.toml generation ────────────────────────────────────────────────────

fn emit_cargo_toml(stem: &str, version: &str) -> String {
    format!(
        r#"[package]
name = "{stem}"
version = "{version}"
edition = "2024"

[[bin]]
name = "{stem}"
path = "src/main.rs"

[dependencies]
cudarc = {{ version = "0.19", features = ["driver", "nvrtc"] }}
"#,
        stem = stem,
        version = version,
    )
}
