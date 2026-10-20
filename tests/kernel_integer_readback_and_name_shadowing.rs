// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression tests for two kernel-target defects found writing boring-ui's kernel rasterizer:
//
// 1. Reading an integer kernel field (`[uint]`, `[int]`) back on the host was mistyped: the
//    read-back converted through `i64` while the host's own `uint`/`int` are `usize`/`isize`
//    (`let [uint] p = k.out` -> E0308), and an indexed read cast (`k.out[i] as uint`) fell
//    through to the string-parse path (`.trim().parse::<usize>()`, E0599). `[float32]` worked.
// 2. `--target wgpu` swept every identifier read in a kernel (not only call callees) into the
//    "reachable host functions" set, so a kernel-local `fade` array next to a host `fade(...)`
//    function emitted the host function (returning a struct with a string) into the WGSL.
//
// Run with:
//   cargo test --test kernel_integer_readback_and_name_shadowing

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const READBACK: &str = r#"
kernel K:
    mut [uint]'unified out
    mut [int]'unified signed

    init(int n):
        out = [0 as uint for ..<n]
        signed = [0 for ..<n]

    def ():
        out[.thread.x] = (.thread.x * 3) as uint
        signed[.thread.x] = .thread.x - 4

def main() throws:
    mut k = K(8)
    kernel:
        k(block = 8, grid = 1)
    let [uint] pixels = k.out
    mut [uint] copy = []
    for i in 0..<8:
        copy.push(k.out[i] as uint)
    let [int] s = k.signed
    print "{pixels[3]} {copy[7]} {s[0]}"
"#;

const SHADOWING: &str = r#"
struct Out:
    pub string name
    pub float32 v

def Out fade(float32 t):
    Out(name = "x", v = t)

def float32 helper(float32 t):
    t * 2.0

kernel K:
    mut [float32]'unified out

    init(int n):
        out = [0.0 as float32 for ..<n]

    def ():
        var [float32, 5] fade = [0.0, 1.0, 2.0, 3.0, 4.0]
        out[.thread.x] = fade[2] + helper(1.0)

def main() throws:
    mut k = K(8)
    kernel:
        k(block = 8, grid = 1)
    let f = fade(2.0)
    print "{k.out[3]} {f.name}"
"#;

fn build(test_name: &str, target: &str, src: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("kernel_integer_readback_and_name_shadowing")
        .join(format!("{test_name}_{target}"));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("main.br");
    fs::write(&input, src).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target])
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    root
}

fn assert_readback_types(rs: &str) {
    assert!(
        rs.contains("let pixels: Vec<usize> = k.copy_out_to_host().iter().map(|&x| x as usize).collect::<Vec<usize>>()"),
        "{rs}"
    );
    assert!(rs.contains("let s: Vec<isize> = k.copy_signed_to_host().iter().map(|&x| x as isize)"), "{rs}");
    assert!(!rs.contains(".trim().parse::<usize>()"), "{rs}");
    assert!(rs.contains("[(i) as usize].clone() as usize)"), "{rs}");
}

#[test]
fn simd_integer_kernel_field_reads_back_with_host_element_type() {
    let root = build("readback", "simd", READBACK);
    let rs = fs::read_to_string(root.join("main_simd/src/main.rs")).unwrap();
    assert_readback_types(&rs);
    let check = Command::new("cargo")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(root.join("main_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(check.status.success(), "{}", String::from_utf8_lossy(&check.stderr));
}

#[test]
fn wgpu_integer_kernel_field_reads_back_with_host_element_type() {
    let root = build("readback", "wgpu", READBACK);
    let rs = fs::read_to_string(root.join("main_wgpu/src/main.rs")).unwrap();
    assert_readback_types(&rs);
}

#[test]
fn wgpu_kernel_local_shadows_same_named_host_function() {
    let root = build("shadowing", "wgpu", SHADOWING);
    let wgsl = fs::read_to_string(root.join("main_wgpu/shaders/main.wgsl")).unwrap();
    assert!(!wgsl.contains("fn fade"), "{wgsl}");
    assert!(!wgsl.contains("Out("), "{wgsl}");
    // a host function the kernel really calls is still emitted
    assert!(wgsl.contains("fn helper("), "{wgsl}");
    assert!(wgsl.contains("var fade: array<f32, 5>"), "{wgsl}");
}
