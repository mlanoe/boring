// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `boring build` on the plain `std` target (no `--target`
// flag) used to silently transpile a program that declares a `kernel` struct
// and emit Rust referencing GPU-only runtime machinery (`BoringGpuArg<T>`,
// `__boring_gpu_copy_d2h`/`__boring_gpu_device`/`__boring_gpu_queue`, the
// kernel struct type itself) — none of which the `std` target's generated
// project ever defines (that machinery only exists in the cuda/rocm/metal/wgpu
// backends' own `host.rs`). The result was `cargo build` failing on the
// generated project with 200+ raw `E0425`/`E0433` "cannot find function/type"
// errors, instead of one clear diagnostic from `boring` itself.
//
// Fix: `transpiler::transpile_with_config` now rejects any `kernel` struct
// declaration up front, before generating any Rust, whenever
// `TranspileConfig::is_gpu_target` is false (see `first_kernel_decl`'s doc in
// `src/transpiler/mod.rs`).
//
// Run with:
//   cargo test --test gpu_kernel_std_target

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const KERNEL_SRC: &str = "\
kernel Scale:
    let [float]'unified data
    let float factor = 2.0

    def ():
        data[thread.x] = data[thread.x] * factor

def main():
    mut k = Scale(data=[1.0, 2.0, 3.0], factor=2.0)
    kernel:
        k(block=3)
    print \"{k.data}\"
";

fn write_case(test_name: &str) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("gpu_kernel_std_target")
        .join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br_file = tmp.join("main.br");
    fs::write(&br_file, KERNEL_SRC).unwrap();
    br_file
}

/// Plain `boring build main.br` (std target, full project-emitting mode) must
/// fail fast with the new diagnostic and must NOT leave a generated
/// `main_rust*` project directory behind.
#[test]
fn kernel_on_std_target_rejected_by_plain_build() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let br_file = write_case("plain_build");

    let result = Command::new(bin)
        .arg("build")
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {e}"));

    assert!(
        !result.status.success(),
        "expected `boring build` to reject a `kernel` struct on the std target, \
         but it succeeded"
    );

    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("kernel `Scale` requires a GPU target"),
        "expected stderr to name the offending kernel and explain the fix, got:\n{stderr}"
    );
    assert!(
        stderr.contains("--target cuda|rocm|metal|wgpu"),
        "expected stderr to hint at the fix (--target flag), got:\n{stderr}"
    );

    // No generated Rust project should have been left behind — the check must
    // run before any codegen/file-writing happens.
    let generated_dir = br_file.parent().unwrap().join("main_rust");
    assert!(
        !generated_dir.exists(),
        "expected no generated project directory on rejection, but {generated_dir:?} exists"
    );
}

/// Same rejection must apply to `--emit-rust` (prints Rust to stdout instead
/// of writing a project) — this is the other main entry point into
/// `transpiler::transpile_with_config` for the std target.
#[test]
fn kernel_on_std_target_rejected_by_emit_rust() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let br_file = write_case("emit_rust");

    let result = Command::new(bin)
        .arg("build")
        .arg("--emit-rust")
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {e}"));

    assert!(
        !result.status.success(),
        "expected `boring build --emit-rust` to reject a `kernel` struct on the \
         std target, but it succeeded — generated source:\n{}",
        String::from_utf8_lossy(&result.stdout)
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("kernel `Scale` requires a GPU target"),
        "expected stderr to name the offending kernel, got:\n{stderr}"
    );
}

/// Sanity check / no-regression: the exact same source must still transpile
/// cleanly on a real GPU target (metal), which sets `is_gpu_target: true` and
/// is therefore unaffected by the new std-target check.
#[test]
fn kernel_on_metal_target_still_accepted() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let br_file = write_case("metal_target");

    let result = Command::new(bin)
        .args(["build", "--target", "metal"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {e}"));

    assert!(
        result.status.success(),
        "expected `boring build --target metal` to still accept a `kernel` \
         struct, but it failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
