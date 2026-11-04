// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `after =` is an ordering guarantee: in a `kernel:` block, a dispatch observes the writes of the
// dispatches it lists (a single kernel or a `[k1, k2]` list), on every target. Concurrency between
// independent dispatches is NOT promised, so these tests only check results, never overlap.
//
// What runs where:
//   * `boring run` (CPU simulation) is the reference output.
//   * wgpu and Metal: the generated project is built AND run on this machine's GPU, and its output
//     must equal the reference (Metal also with `BORING_METAL_BATCH=3` / `64`, which encode several
//     launches into one command buffer).
//   * CUDA and ROCm: generation only (no such hardware here) -- the emitted launch calls must pass
//     the listed kernels' streams, in program order.
//
// Run with: tools/test.sh --test after_ordering

mod common;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Kernel structs own their buffers, so data can only flow from one dispatch to the next through
/// the same instance: each `Step` dispatch computes `out = out * 2 + i + 1`, which is not
/// commutative with itself or with a missing/misordered dispatch. `ka` runs 3 times, `kb` twice.
const SRC: &str = r#"
kernel Step:
    mut [float32]'unified out
    init(int n):
        out = [0.0 for ..<n]
    def ():
        let i = .thread.x
        out[i] = out[i] * 2.0 + (i as float32) + 1.0

mut ka = Step(8)
mut kb = Step(8)
kernel:
    ka(block = 8, grid = 1)
    kb(block = 8, grid = 1)
    ka(block = 8, grid = 1, after = ka)
    ka(block = 8, grid = 1, after = [ka, kb])
    kb(block = 8, grid = 1, after = ka)
print "{ka.out[0]} {ka.out[3]} {ka.out[7]} {kb.out[0]} {kb.out[3]} {kb.out[7]}"
"#;

/// Hand-computed: ka[i] after 3 steps = 7(i+1), kb[i] after 2 steps = 3(i+1).
const EXPECTED: &str = "7 28 56 3 12 24";

fn scratch(test: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("after_ordering").join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn boring(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(args)
        .arg(dir.join("test.br"))
        .current_dir(dir)
        .output()
        .unwrap()
}

fn build(test: &str, target: &str) -> (PathBuf, PathBuf) {
    let dir = scratch(test);
    fs::write(dir.join("test.br"), SRC).unwrap();
    let out = boring(&dir, &["build", "--target", target]);
    assert!(out.status.success(), "[{test}] build --target {target} failed:\n{}",
        String::from_utf8_lossy(&out.stderr));
    let proj = dir.join(format!("test_{target}"));
    (dir, proj)
}

fn run_project(shared: &str, proj: &Path, batch: Option<&str>) -> String {
    let mut c = common::SharedCargo::new(shared)
        .args(["run", "--quiet", "--manifest-path"])
        .arg(proj.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never");
    if let Some(b) = batch {
        c = c.env("BORING_METAL_BATCH", b);
    }
    let out = c.output().unwrap();
    assert!(out.status.success(), "run failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn cpu_simulation_reference() {
    let dir = scratch("cpu");
    fs::write(dir.join("test.br"), SRC).unwrap();
    let out = boring(&dir, &["run"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), EXPECTED);
}

/// Before the fix, a `kernel:` block with several dispatch statements emitted `ka(8);` /
/// `Ok(kb(8, ka))` on wgpu (E0618, no dispatch at all); only a one-statement block worked.
#[test]
fn wgpu_multi_dispatch_block_emits_every_dispatch_in_order() {
    let (_, proj) = build("wgpu_codegen", "wgpu");
    let rs = fs::read_to_string(proj.join("src/main.rs")).unwrap();
    let calls: Vec<&str> = rs.lines().map(str::trim)
        .filter(|l| l.starts_with("ka.dispatch(") || l.starts_with("kb.dispatch(")).collect();
    let who: Vec<&str> = calls.iter().map(|l| &l[..2]).collect();
    assert_eq!(who, ["ka", "kb", "ka", "ka", "kb"], "{rs}");
    assert!(!rs.contains("ka(8)"), "{rs}");
}

#[test]
fn wgpu_after_orders_dispatches() {
    let (_, proj) = build("wgpu_run", "wgpu");
    assert_eq!(run_project("after_ordering", &proj, None), EXPECTED);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_after_orders_dispatches_at_every_batch_size() {
    let (_, proj) = build("metal_run", "metal");
    for batch in [None, Some("1"), Some("3"), Some("64")] {
        assert_eq!(run_project("after_ordering", &proj, batch), EXPECTED, "BORING_METAL_BATCH={batch:?}");
    }
}

/// Generation only: no CUDA hardware here. Each launch must carry the streams it waits for.
#[test]
fn cuda_after_passes_listed_streams_in_order() {
    let (_, proj) = build("cuda_codegen", "cuda");
    assert_launch_streams(&fs::read_to_string(proj.join("src/main.rs")).unwrap());
}

/// Generation only: no ROCm hardware here.
#[test]
fn rocm_after_passes_listed_streams_in_order() {
    let (_, proj) = build("rocm_codegen", "rocm");
    assert_launch_streams(&fs::read_to_string(proj.join("src/main.rs")).unwrap());
}

fn assert_launch_streams(rs: &str) {
    let launches: Vec<&str> = rs.lines().map(str::trim)
        .filter(|l| l.starts_with("ka = ka.__boring_launch(") || l.starts_with("kb = kb.__boring_launch("))
        .collect();
    let after = |l: &str| {
        let s = l.split("), &[").nth(1).unwrap_or("").to_string();
        s.split("], ").next().unwrap_or("").to_string()
    };
    let got: Vec<(String, String)> = launches.iter().map(|l| (l[..2].to_string(), after(l))).collect();
    let want = [
        ("ka", ""), ("kb", ""), ("ka", "&ka.__stream"),
        ("ka", "&ka.__stream, &kb.__stream"), ("kb", "&ka.__stream"),
    ].map(|(a, b)| (a.to_string(), b.to_string()));
    assert_eq!(got, want, "{rs}");
}
