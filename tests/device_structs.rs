// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Plain-data structs in device code (locals, nested structs, struct parameters and returns,
// optional fields) on every GPU backend. `src/desugar_gpu_struct.rs` scalar-replaces them, so
// the backends need no struct definitions. Fixture: `tests/cases/device_structs.br`; the
// expected value per thread i is 16 i + 1225 (worked out in the fixture's arithmetic).
//
// Run with: cargo test --test device_structs

use std::path::{Path, PathBuf};
use std::process::Command;

const FIXTURE: &str = "tests/cases/device_structs.br";
const EXPECTED: &str = "1225 1241 1257 1273";

fn generate(target: &str, name: &str) -> PathBuf {
    generate_from(FIXTURE, target, name)
}

fn generate_from(fixture: &str, target: &str, name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("device_structs_{name}_{target}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(fixture, dir.join("main.br")).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target, "main.br"])
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(out.status.success(), "{target}: {}", String::from_utf8_lossy(&out.stderr));
    dir
}

fn run(dir: &Path, crate_dir: &str, release: bool) -> (String, String, bool) {
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--quiet"]);
    if release {
        cmd.arg("--release");
    }
    let out = cmd
        .arg("--manifest-path")
        .arg(dir.join(crate_dir).join("Cargo.toml"))
        .arg("--target-dir")
        .arg(dir.join("target"))
        .output()
        .unwrap();
    (
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

#[test]
fn device_backends_scalar_replace_structs() {
    for (target, kernel_file) in [
        ("metal", "main_metal/kernels/main.metal"),
        ("wgpu", "main_wgpu/shaders/main.wgsl"),
        ("cuda", "main_cuda/kernels/main.cu"),
        ("rocm", "main_rocm/kernels/main.hip"),
    ] {
        let dir = generate(target, "shape");
        let source = std::fs::read_to_string(dir.join(kernel_file)).unwrap();
        assert!(!source.contains("unsupported"), "{target}: {source}");
        for needle in ["b__pos__y", "add__dev__x", "heavier__dev__tag", "c__tag__has"] {
            assert!(source.contains(needle), "{target}: missing {needle}: {source}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn simd_kernels_with_structs_run_correctly() {
    let dir = generate("simd", "run");
    let (stdout, stderr, ok) = run(&dir, "main_simd", true);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(ok, "{stderr}");
    assert_eq!(stdout, EXPECTED);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_kernels_with_structs_run_correctly() {
    let dir = generate("metal", "run");
    let (stdout, stderr, ok) = run(&dir, "main_metal", false);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(ok, "{stderr}");
    assert_eq!(stdout, EXPECTED);
}

// Struct methods: `req` methods (with a struct parameter / return) and a `def` that updates
// `self`. Per thread i: a = (i + 10, 12) after `shift`, so 4 i + 114.
const METHODS_FIXTURE: &str = "tests/cases/device_struct_methods.br";
const METHODS_EXPECTED: &str = "114 118 122 126";

#[test]
fn device_backends_lower_struct_methods() {
    for (target, kernel_file) in [
        ("metal", "main_metal/kernels/main.metal"),
        ("wgpu", "main_wgpu/shaders/main.wgsl"),
        ("cuda", "main_cuda/kernels/main.cu"),
        ("rocm", "main_rocm/kernels/main.hip"),
    ] {
        let dir = generate_from(METHODS_FIXTURE, target, "methods_shape");
        let source = std::fs::read_to_string(dir.join(kernel_file)).unwrap();
        assert!(!source.contains("unsupported"), "{target}: {source}");
        for needle in ["Vec2__dot__dev", "Vec2__shift__dev__x", "Vec2__plus__dev__y"] {
            assert!(source.contains(needle), "{target}: missing {needle}: {source}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn struct_methods_run_correctly_on_simd() {
    let dir = generate_from(METHODS_FIXTURE, "simd", "methods_run");
    let (stdout, stderr, ok) = run(&dir, "main_simd", true);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(ok, "{stderr}");
    assert_eq!(stdout, METHODS_EXPECTED);
}

#[cfg(target_os = "macos")]
#[test]
fn struct_methods_run_correctly_on_metal() {
    let dir = generate_from(METHODS_FIXTURE, "metal", "methods_run");
    let (stdout, stderr, ok) = run(&dir, "main_metal", false);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(ok, "{stderr}");
    assert_eq!(stdout, METHODS_EXPECTED);
}
