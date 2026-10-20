// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression tests for two kernel gaps found writing boring-ui's rasterizer:
//
// 1. A device helper declared inside a `kernel` struct (`def float32 twice(..)`) and
//    called bare (`twice(x)`) from `def ()` worked under `boring run` but failed on
//    every compiled GPU/SIMD target (simd: "cannot find function `twice`"; wgpu: helper
//    emitted as `Dbl_twice` but called as `twice`; cuda/rocm/metal: undefined `twice`).
//    Fixed by `desugar_gpu_struct::qualify_kernel_helper_calls`.
// 2. The interpreter's kernel simulation rejected a float literal / unannotated
//    `var x = 0.0` mixed with float32 values (`max(0.0, x32)`, `sum += x32 * y32`) that the
//    compiled targets accept. Fixed by `Interpreter::kernel_thread`.
//
// Run with:
//   cargo test --test kernel_helper_and_float_promotion

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const HELPER_SRC: &str = "\
kernel Dbl:
    mut [float32]'unified f

    init([float32]'unified data):
        f = data

    def float32 twice(float32 x):
        x * 2.0

    def float32 quad(float32 x):
        twice(twice(x))

    def ():
        let i = .thread.x
        f[i] = twice(f[i]) + quad(1.0)

def main():
    let [float32] d = [1.0, 2.0, 3.0]
    mut k = Dbl(d)
    kernel:
        k(block = 3)
    print \"{k.f[0]} {k.f[1]} {k.f[2]}\"
";

const FLOAT_SRC: &str = "\
kernel Fl:
    mut [float32]'unified f
    mut [float32]'unified g

    init([float32]'unified a, [float32]'unified b):
        f = a
        g = b

    def ():
        let i = .thread.x
        var sum = 0.0
        sum += f[i] * g[i]
        g[i] = max(0.0, sum) + min(f[i], 3.0)

def main():
    let [float32] a = [1.0, 2.0, 5.0]
    let [float32] b = [1.0, -2.0, 1.0]
    mut k = Fl(a, b)
    kernel:
        k(block = 3)
    print \"{k.g[0]} {k.g[1]} {k.g[2]}\"
";

fn write_case(group: &str, src: &str) -> PathBuf {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("kernel_helper_and_float_promotion")
        .join(group);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("main.br");
    fs::write(&br, src).unwrap();
    br
}

fn run(br: &PathBuf) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(br).output().unwrap();
    assert!(out.status.success(), "boring run failed:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn emit(br: &PathBuf, target: &str) {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target, "--emit-rust"])
        .arg(br)
        .output()
        .unwrap();
    assert!(out.status.success(), "[{target}] build failed:\n{}", String::from_utf8_lossy(&out.stderr));
}

fn read(br: &PathBuf, rel: &str) -> String {
    fs::read_to_string(br.parent().unwrap().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

#[test]
fn kernel_helper_bare_call_interpreter() {
    let br = write_case("helper_run", HELPER_SRC);
    // twice(f)+quad(1) = 2f + 4
    assert_eq!(run(&br), "6 8 10");
}

#[test]
fn kernel_helper_bare_call_simd() {
    let br = write_case("helper_simd", HELPER_SRC);
    emit(&br, "simd");
    let src = read(&br, "main_simd/src/main.rs");
    assert!(src.contains("self.twice(self.f["), "bare helper call must be qualified:\n{src}");
    assert!(src.contains("self.twice(self.twice(x))") || src.contains("self.twice(self.twice("), "nested helper call");
}

#[test]
fn kernel_helper_bare_call_wgpu() {
    let br = write_case("helper_wgpu", HELPER_SRC);
    emit(&br, "wgpu");
    let wgsl = read(&br, "main_wgpu/shaders/main.wgsl");
    assert!(wgsl.contains("fn Dbl_twice("), "{wgsl}");
    assert!(wgsl.contains("Dbl_twice(dbl_f["), "call must target the emitted helper:\n{wgsl}");
    assert!(!wgsl.contains(" twice("), "unqualified helper call left in WGSL:\n{wgsl}");
}

#[test]
fn kernel_helper_bare_call_cuda_rocm_metal() {
    for (target, file) in [
        ("cuda", "main_cuda/kernels/main.cu"),
        ("rocm", "main_rocm/kernels/main.hip"),
        ("metal", "main_metal/kernels/main.metal"),
    ] {
        let br = write_case(&format!("helper_{target}"), HELPER_SRC);
        emit(&br, target);
        let src = read(&br, file);
        assert!(src.contains("Dbl_twice(f, f["), "[{target}] call must be qualified:\n{src}");
        assert!(!src.contains(" twice("), "[{target}] unqualified helper call left:\n{src}");
    }
}

#[test]
fn kernel_float_literal_promotes_to_float32_in_interpreter() {
    let br = write_case("float_run", FLOAT_SRC);
    // i0: sum=1 -> max(0,1)+min(1,3)=2; i1: sum=-4 -> 0+2=2; i2: sum=5 -> 5+3=8
    assert_eq!(run(&br), "2 2 8");
}

#[test]
fn kernel_float_literal_mixing_compiles_on_gpu_targets() {
    for target in ["simd", "wgpu"] {
        let br = write_case(&format!("float_{target}"), FLOAT_SRC);
        emit(&br, target);
    }
}
