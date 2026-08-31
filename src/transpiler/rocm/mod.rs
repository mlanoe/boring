// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Boring → ROCm/HIP transpiler.
//
// Entry point: `transpile_rocm(program)` returns a `RocmOutput` containing:
//   - `host_rs`   — Rust source for the host binary (hand-rolled HIP FFI, see `host.rs`)
//   - `device_hip` — HIP C++ source for the device kernels
//   - `kernel_names` — list of kernel struct names (used to populate the Cargo project)
//
// Usage:
//   boring build --target rocm main.br
//   → creates main_rocm/
//         src/main.rs          (host_rs)
//         kernels/main.hip     (device_hip)
//         build.rs
//         Cargo.toml
//
// ── Architecture: general-pipeline splice + kernel-touching carve-out ─────────
//
// Mirrors `cuda::mod`'s identical architecture exactly — see that module's doc
// comment for the full rationale (kernel-touching functions/top-level code stay
// on this backend's own custom emitter, since the general pipeline's kernel-aware
// codegen is wgpu-shaped; everything else is transpiled by the general pipeline
// for full language-feature correctness).
//
// ── Why not just reuse cuda's host.rs verbatim? ────────────────────────────────
//
// HIP's device-side C++ (kernel qualifiers, `threadIdx`/`blockIdx`, atomics,
// `__shared__`, `printf`) is source-compatible with CUDA C by design (that's
// HIP's whole purpose), so `device.rs` really is close to a straight clone of
// `cuda::device` — see that module's doc comment.
//
// The HOST side is a different story: there is no mature, widely-used safe
// Rust crate for ROCm/HIP analogous to `cudarc` (the crate `cuda::host` is
// built on). Rather than depend on an unverified/unmaintained third-party
// binding, this backend hand-rolls a small `extern "C"` FFI layer directly in
// the generated `host_rs` (linking `libamdhip64`, ROCm's stable, documented C
// API — `hipModuleLoadData`/`hipModuleLaunchKernel`/`hipMemcpy*`/etc., all
// long-stable HIP runtime entry points) plus a safe wrapper around it
// (`HipContext`/`HipStream`/`HipModule`/`DeviceBuffer<T>`/`LaunchBuilder`) that
// deliberately mirrors cudarc's own method names/shapes (`alloc_zeros`,
// `clone_htod`, `clone_dtoh`, `load_function`, `launch_builder`/`.arg`/
// `.launch`, `new_stream_with_priority`, ...). That's what makes it possible to
// carry over `cuda::host`'s ~1300 lines of Boring-AST→Rust statement/expression
// codegen (`emit_fn`/`emit_stmt`/`expr`/pattern-matching/dict handling/GPU-
// residency tracking — all entirely target-agnostic logic) with only
// mechanical type-name substitutions (`CudaContext`→`HipContext`,
// `CudaSlice<T>`→`DeviceBuffer<T>`, etc.), instead of re-deriving that whole
// pipeline from scratch for a fourth GPU target.
//
// `hipDeviceAttribute_t`'s numeric enum values are NOT guaranteed ABI-stable
// across ROCm releases (unlike CUDA's driver enum, which cudarc depends on
// directly), so `warpSize()`/`maxThreads()`/`maxSharedMem()` can't safely
// hardcode an attribute ID the way `name()`/`totalMem()`/`computeCapability()`
// use dedicated stable API calls (`hipDeviceGetName`/`hipDeviceTotalMem`/
// `hipMemGetInfo`/`hipDeviceComputeCapability`). Instead, `emit_build_rs`
// below compiles and runs a tiny C probe against whatever ROCm headers are
// actually installed on the build machine, and bakes the *real* enum values
// it reads back into a generated `OUT_DIR/boring_hip_attrs.rs` -- see that
// function's doc comment. This sidesteps the ABI question entirely instead of
// guessing a value from a specific ROCm version's header.
//
// Known gap (shared with `cuda`/`metal`): a kernel-touching STRUCT METHOD (as
// opposed to a free function) isn't supported by the general-pipeline splice —
// see `cuda::mod`'s identical doc note.
//
// Not independently verified against a real ROCm toolchain/AMD GPU (none
// available in this dev environment) — mirrors the same caveat `metal::mod`
// documents for lacking a macOS toolchain.

use crate::ast::{Program, Item, ExprKind};

mod device;
mod host;

// ─── Public output type ───────────────────────────────────────────────────────

pub struct RocmOutput {
    /// Rust host source (src/main.rs).
    pub host_rs: String,
    /// HIP C++ device source (kernels/main.hip).
    pub device_hip: String,
    /// Names of all `kernel` struct declarations found in the program.
    pub kernel_names: Vec<String>,
    /// Generated build.rs content.
    pub build_rs: String,
    /// Generated Cargo.toml content.
    pub cargo_toml: String,
    /// Errors accumulated while transpiling non-kernel-touching code through
    /// the general pass — see `cuda::CudaOutput::errors`'s identical field.
    pub errors: Vec<crate::transpiler::TranspileError>,
}

// ─── Public entry point ───────────────────────────────────────────────────────

pub fn transpile_rocm(program: &Program, stem: &str, version: &str) -> RocmOutput {
    // Collect all kernel names.
    let kernel_names: Vec<String> = program.items.iter().filter_map(|item| {
        if let Item::Kernel(decl) = item { Some(decl.name.clone()) } else { None }
    }).collect();
    let kernel_names_set: std::collections::HashSet<String> = kernel_names.iter().cloned().collect();

    // See metal::mod's identical `has_screen` pre-pass -- a top-level
    // `let screen = Screen(...)` marks this as a display program, which
    // gets the same top-level-handled-by-host treatment as bare top-level
    // kernel dispatch (see `top_level_kernel_touching` below).
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

    let device_hip = device::emit_device_hip(program);

    // See this module's doc comment for the full splice architecture.
    let kernel_touching = crate::transpiler::kernel_touching_fn_names(program, &kernel_names_set);
    let kernel_touching_structs = crate::transpiler::kernel_touching_struct_names(program, &kernel_names_set);
    if !kernel_touching_structs.is_empty() {
        eprintln!(
            "warning: struct(s) {:?} have a kernel-touching method -- the ROCm backend's \
             general-pipeline splice doesn't support this combination (see rocm::mod's doc \
             comment); these structs will keep the old by-value parameter behavior instead \
             of the fix applied to every other struct/function.",
            kernel_touching_structs
        );
    }

    // Bare top-level kernel construction/dispatch — see `cuda::mod`'s identical
    // doc comment for why this can't go through the general-pipeline splice.
    // A `Screen` program gets the same treatment (mirrors metal/wgpu's own
    // `has_screen` carve-out) -- it keeps its ENTIRE top-level on this
    // backend's own existing driver too.
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
    let general_code = crate::transpiler::strip_top_level_fns(&general_out.code, &strip_names);
    // This backend's own prelude (`host::emit_prelude`) needs `Arc` for
    // `Arc<HipContext>`/`Arc<HipStream>` -- same duplicate-import fix as
    // `cuda::mod` (see its identical comment); the general pass's own
    // `use std::sync::Arc;` would otherwise collide (E0252).
    let general_code = general_code.replace("use std::sync::Arc;\n", "");

    let host_rs = host::emit_host_rs(
        program, &kernel_names, &kernel_touching, &general_code,
        has_boring_main, boring_main_throws, top_level_kernel_touching,
    );
    let build_rs  = emit_build_rs();
    let cargo_toml = emit_cargo_toml(stem, version, has_screen);

    RocmOutput { host_rs, device_hip, kernel_names, build_rs, cargo_toml, errors: general_out.errors }
}

// ─── build.rs generation ─────────────────────────────────────────────────────

// In addition to compiling kernels/main.hip, the generated build.rs compiles
// and runs a tiny C probe (`__boring_hip_attr_probe.c`) against whatever
// ROCm headers are installed on THIS build machine, and prints the real,
// locally-correct numeric values of the three `hipDeviceAttribute_t` members
// backing `GPU().warpSize()`/`.maxThreads()`/`.maxSharedMem()`. Those values
// get written to `OUT_DIR/boring_hip_attrs.rs`, which `host.rs`'s prelude
// `include!()`s -- see rocm::mod's doc comment for why this, rather than a
// hardcoded constant, is the only safe way to resolve those three properties
// without depending on a specific ROCm version's enum layout. If the probe
// can't be compiled/run (e.g. a `hip/hip_runtime_api.h` too old or missing),
// the build does not fail: the three constants fall back to a `-1` sentinel,
// and the corresponding `HipContext` methods report a clean runtime error
// instead (same fallback as before this probe existed).
fn emit_build_rs() -> String {
    r#"// Generated by boring build --target rocm.
//
// Compiles kernels/main.hip to a loadable HIP code object via hipcc --genco,
// makes the path available to main.rs through the BORING_HIP_CO_PATH env var,
// wires up linking against libamdhip64 (ROCm's HIP runtime shared lib), and
// probes the local ROCm install's hip_runtime_api.h for the real
// hipDeviceAttribute_t values behind GPU().warpSize()/.maxThreads()/
// .maxSharedMem() (see rocm::mod's doc comment for why).

use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=kernels/main.hip");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    println!("cargo:rustc-link-search=native={}/lib", rocm_path);
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let co_path = PathBuf::from(&out_dir).join("main.hipfb");

    // `--genco` ("generate code object") produces a loadable fat-binary code
    // object, the HIP analogue of `nvcc --ptx` -- loaded at runtime via
    // `hipModuleLoadData` (see host.rs's `boring_gpu_init`). Set
    // BORING_ROCM_ARCH (e.g. "gfx1100") to target a specific GPU architecture;
    // left unset, hipcc uses its own default detection.
    let mut cmd = Command::new("hipcc");
    cmd.args(["--genco", "-O2", "-o", co_path.to_str().unwrap(), "kernels/main.hip"]);
    if let Ok(arch) = std::env::var("BORING_ROCM_ARCH") {
        cmd.arg(format!("--offload-arch={}", arch));
    }

    let status = cmd.status()
        .expect("hipcc not found — install the ROCm toolkit (https://rocm.docs.amd.com)");

    if !status.success() {
        panic!("hipcc failed to compile kernels/main.hip");
    }

    println!("cargo:rustc-env=BORING_HIP_CO_PATH={}", co_path.display());

    probe_hip_device_attributes(&out_dir);
}

// Compiles and runs a tiny host-only C program against this machine's own
// `hip/hip_runtime_api.h` to read the *actual* installed values of
// `hipDeviceAttributeWarpSize`/`hipDeviceAttributeMaxThreadsPerBlock`/
// `hipDeviceAttributeSharedMemPerBlock` -- these are NOT hardcoded anywhere
// in boring itself, because the numeric values of `hipDeviceAttribute_t`
// are not guaranteed stable across ROCm releases. Reading them from this
// build's own header is the only way to get a value guaranteed to match
// what the locally installed `libamdhip64` actually expects.
fn probe_hip_device_attributes(out_dir: &str) {
    let probe_c = PathBuf::from(out_dir).join("__boring_hip_attr_probe.c");
    let probe_bin = PathBuf::from(out_dir).join("__boring_hip_attr_probe");
    let attrs_rs = PathBuf::from(out_dir).join("boring_hip_attrs.rs");

    let fallback = || {
        std::fs::write(&attrs_rs, "\
pub(crate) const BORING_HIP_ATTR_WARP_SIZE: i32 = -1;\n\
pub(crate) const BORING_HIP_ATTR_MAX_THREADS_PER_BLOCK: i32 = -1;\n\
pub(crate) const BORING_HIP_ATTR_SHARED_MEM_PER_BLOCK: i32 = -1;\n\
").expect("failed to write boring_hip_attrs.rs fallback");
    };

    if std::fs::write(&probe_c, "\
#include <hip/hip_runtime_api.h>\n\
#include <stdio.h>\n\
int main(void) {\n\
    printf(\"%d %d %d\\n\",\n\
           (int)hipDeviceAttributeWarpSize,\n\
           (int)hipDeviceAttributeMaxThreadsPerBlock,\n\
           (int)hipDeviceAttributeSharedMemPerBlock);\n\
    return 0;\n\
}\n\
").is_err() {
        fallback();
        return;
    }

    let compiled = Command::new("hipcc")
        .args(["-o", probe_bin.to_str().unwrap(), probe_c.to_str().unwrap()])
        .status();
    if !matches!(compiled, Ok(s) if s.success()) {
        eprintln!("cargo:warning=boring: could not compile the hipDeviceAttribute_t probe -- \
                    GPU().warpSize()/.maxThreads()/.maxSharedMem() will report a runtime error");
        fallback();
        return;
    }

    let output = Command::new(&probe_bin).output();
    let parsed = output.ok().and_then(|o| {
        if !o.status.success() { return None; }
        let text = String::from_utf8_lossy(&o.stdout);
        let mut nums = text.split_whitespace().filter_map(|s| s.parse::<i32>().ok());
        Some((nums.next()?, nums.next()?, nums.next()?))
    });

    match parsed {
        Some((warp, max_threads, shared_mem)) => {
            std::fs::write(&attrs_rs, format!("\
pub(crate) const BORING_HIP_ATTR_WARP_SIZE: i32 = {warp};\n\
pub(crate) const BORING_HIP_ATTR_MAX_THREADS_PER_BLOCK: i32 = {max_threads};\n\
pub(crate) const BORING_HIP_ATTR_SHARED_MEM_PER_BLOCK: i32 = {shared_mem};\n\
")).expect("failed to write boring_hip_attrs.rs");
        }
        None => {
            eprintln!("cargo:warning=boring: could not run the hipDeviceAttribute_t probe -- \
                        GPU().warpSize()/.maxThreads()/.maxSharedMem() will report a runtime error");
            fallback();
        }
    }
}
"#.into()
}

// ─── Cargo.toml generation ────────────────────────────────────────────────────

fn emit_cargo_toml(stem: &str, version: &str, has_screen: bool) -> String {
    // `winit`/`softbuffer` are only needed for a `Screen` (display) program --
    // see `host::emit_screen_setup`'s doc comment for why this specific pair
    // (winit 0.28's `run_return` API + the matching raw-window-handle 0.5
    // softbuffer release) was chosen. A compute-only program stays free of
    // any graphics-crate dependency, same as today.
    let extra_deps = if has_screen { "winit = \"0.28\"\nsoftbuffer = \"0.3\"\n" } else { "" };
    format!(
        r#"[package]
name = "{stem}"
version = "{version}"
edition = "2024"

[[bin]]
name = "{stem}"
path = "src/main.rs"

# No external GPU crate dependency -- see this module's doc comment for why:
# host.rs hand-rolls the HIP FFI/safe-wrapper layer directly, linked against
# libamdhip64 via build.rs. `winit`/`softbuffer` below are the sole exception,
# added only for a `Screen` (display) program -- they're a CPU-side windowing/
# present layer, not a GPU compute crate.
[dependencies]
{extra_deps}"#,
        stem = stem,
        version = version,
        extra_deps = extra_deps,
    )
}
