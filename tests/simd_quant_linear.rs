// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Runtime checks of the quantized `kernel.tensors.linear` on the CPU SIMD target.
//
// The two real-GPU fixtures (`metal_quant_prefill_bias.br`: Q8_0/Q4_K/Q6_K and
// `metal_tiled_quant_prefill.br`: Q4_0/Q5_0/IQ4_NL/Q3_K/Q2_K) are plain Boring programs, so they
// also build for `--target simd`. Each is compiled once, then run under different runtime
// configurations. Three schedules exist and are compared pairwise:
//   reference  BORING_SIMD_QUANT_REFERENCE=1  per-element decoder, horizontal sum per block
//   float      (default)                      block dequantizers, 16 lanes kept across blocks
//   int8       BORING_SIMD_QUANT_INT8=1       decode (m == 1) with int8-rounded activations
//
//   cargo test --test simd_quant_linear

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn build(root: &Path, fixture: &str) -> PathBuf {
    fs::create_dir_all(root).unwrap();
    fs::write(root.join("prefill.br"), fixture).unwrap();
    let transpile = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"]).arg(root.join("prefill.br")).output().unwrap();
    assert!(transpile.status.success(), "{}", String::from_utf8_lossy(&transpile.stderr));
    let cargo = Command::new("cargo")
        .args(["build", "--release", "--quiet", "--manifest-path"])
        .arg(root.join("prefill_simd/Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output().unwrap();
    assert!(cargo.status.success(), "{}", String::from_utf8_lossy(&cargo.stderr));
    root.join("prefill_simd/target/release/prefill-simd")
}

fn run(exe: &Path, format: &str, m: usize, n: usize, k: usize, bias: bool, env: &[(&str, &str)]) -> Vec<f32> {
    let mut command = Command::new(exe);
    command.args([format, &m.to_string(), &n.to_string(), &k.to_string(), if bias { "bias" } else { "no-bias" }]);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command.output().unwrap();
    assert!(output.status.success(), "{format} {m}x{n}x{k}: {}", String::from_utf8_lossy(&output.stderr));
    // The AVX2 integer kernel verifies itself before first use and reports a disagreement on stderr.
    assert!(!String::from_utf8_lossy(&output.stderr).contains("self-test"), "{}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.strip_prefix("value=").map(|value| value.parse().unwrap()))
        .collect()
}

/// Q4_0/Q5_0/IQ4_NL/Q8_0 need k % 32 == 0, the K-quants k % 256 == 0. `m == 1` is the decode shape
/// (columns split across workers); n values are not multiples of the worker count.
fn shapes(format: &str) -> Vec<(usize, usize, usize)> {
    let k_quant = format.ends_with("_k");
    vec![(1, 77, 512), (1, 1000, 1024), (3, 33, 256), (37, 129, if k_quant { 512 } else { 480 }), (130, 65, 256)]
}

const REFERENCE: [(&str, &str); 2] = [("BORING_SIMD_THREADS", "1"), ("BORING_SIMD_QUANT_REFERENCE", "1")];
const FLOAT_ONE_WORKER: [(&str, &str); 1] = [("BORING_SIMD_THREADS", "1")];
/// A zero threshold sends even the tiniest shape through the worker queue.
const FLOAT_PARALLEL: [(&str, &str); 1] = [("BORING_SIMD_QUANT_THRESHOLD", "0")];

/// Runs every shape of `formats` under `baseline` and `candidate` environments and requires each
/// output to agree within `tolerance` times the largest baseline magnitude.
fn compare(exe: &Path, formats: &[&str], baseline: &[(&str, &str)], candidate: &[(&str, &str)], label: &str, tolerance: f32) {
    compare_shapes(exe, formats, baseline, candidate, label, tolerance, |_| true);
}

/// `keep(m)` selects the shapes by row count: `m == 1` is the decode shape.
fn compare_shapes(exe: &Path, formats: &[&str], baseline: &[(&str, &str)], candidate: &[(&str, &str)], label: &str, tolerance: f32, keep: fn(usize) -> bool) {
    for &format in formats {
        for (m, n, k) in shapes(format).into_iter().filter(|&(m, _, _)| keep(m)) {
            for bias in [false, true] {
                let reference = run(exe, format, m, n, k, bias, baseline);
                assert_eq!(reference.len(), m * n, "{format} {m}x{n}x{k}");
                let scale = reference.iter().copied().map(f32::abs).fold(0.0, f32::max);
                assert!(scale.is_finite() && scale > 0.0, "{format} {m}x{n}x{k}: degenerate reference");
                let actual = run(exe, format, m, n, k, bias, candidate);
                assert_eq!(actual.len(), reference.len(), "{label} {format} {m}x{n}x{k}");
                for (index, (&a, &r)) in actual.iter().zip(&reference).enumerate() {
                    assert!((a - r).abs() <= tolerance * scale,
                        "{label} {format} {m}x{n}x{k} bias={bias} output {index} (row {}, col {}): got {a}, baseline {r} (tolerance {tolerance} x {scale})",
                        index / n, index % n);
                }
            }
        }
    }
}

const FIXTURE_LARGE: &str = include_str!("fixtures/metal_quant_prefill_bias.br");
const FIXTURE_SMALL: &str = include_str!("fixtures/metal_tiled_quant_prefill.br");

fn build_both(name: &str) -> [(PathBuf, &'static [&'static str]); 2] {
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&base);
    [
        (build(&base.join("large"), FIXTURE_LARGE), &["q8_0", "q4_k", "q6_k"]),
        (build(&base.join("small"), FIXTURE_SMALL), &["q4_0", "q5_0", "iq4_nl", "q3_k", "q2_k"]),
    ]
}

#[test]
fn float_schedule_is_deterministic_across_workers_and_close_to_the_reference_order() {
    for (exe, formats) in build_both("simd_quant_linear_float") {
        // The original decoder is bit-identical whatever the worker count.
        compare(&exe, formats, &REFERENCE, &[("BORING_SIMD_QUANT_REFERENCE", "1"), ("BORING_SIMD_QUANT_THRESHOLD", "0")],
            "reference, parallel", 0.0);
        // Keeping the sixteen lanes across blocks reassociates the sums: relative 1e-5 of the output scale.
        compare(&exe, formats, &REFERENCE, &FLOAT_ONE_WORKER, "float vs reference", 1.0e-5);
        // But the float schedule itself never depends on the number of workers.
        compare(&exe, formats, &FLOAT_ONE_WORKER, &FLOAT_PARALLEL, "float, parallel vs one worker", 0.0);
    }
}

const INT8: [(&str, &str); 2] = [("BORING_SIMD_THREADS", "1"), ("BORING_SIMD_QUANT_INT8", "1")];
const INT8_ACTIVATIONS_ONLY: [(&str, &str); 2] = [("BORING_SIMD_THREADS", "1"), ("BORING_SIMD_QUANT_INT8", "activations")];

/// Declared accuracy of the opt-in int8 decode: activations are rounded to int8 per 32-block
/// (step = max|x| / 127), which the integer kernels cannot recover. Against the float schedule the
/// largest observed error on these fixtures is 4.9e-2 of max|y| (Q4_K, where the minimum term
/// amplifies the rounding); 6e-2 is the declared bound. A wrong layout or sign is off by ~1x.
const INT8_DECLARED_TOLERANCE: f32 = 6.0e-2;

#[test]
fn int8_kernels_are_exact_on_rounded_activations() {
    for (exe, formats) in build_both("simd_quant_linear_int8_exact") {
        // Same rounded activations, integer vs float arithmetic: only float summation order differs.
        compare_shapes(&exe, formats, &INT8_ACTIVATIONS_ONLY, &INT8, "int8 kernels vs float on rounded activations", 2.0e-5, |m| m == 1);
    }
}

#[test]
fn int8_decode_stays_within_its_declared_tolerance_of_the_float_schedule() {
    for (exe, formats) in build_both("simd_quant_linear_int8_bound") {
        compare_shapes(&exe, formats, &FLOAT_ONE_WORKER, &INT8, "int8 vs float", INT8_DECLARED_TOLERANCE, |m| m == 1);
        // Only the single-row decode changes: every other shape is bit-identical to the float schedule.
        compare_shapes(&exe, formats, &FLOAT_ONE_WORKER, &INT8, "int8 leaves prefill alone", 0.0, |m| m > 1);
        // The int8 schedule does not depend on the number of workers either.
        compare_shapes(&exe, formats, &INT8, &[("BORING_SIMD_QUANT_INT8", "1"), ("BORING_SIMD_QUANT_THRESHOLD", "0")],
            "int8, parallel vs one worker", 0.0, |_| true);
    }
}

/// The 256-bit AVX2 integer dot could not be run where it was written; on any x86-64 machine with
/// AVX2 this compares it with the 128-bit SSSE3 kernel (`BORING_SIMD_QUANT_NO_AVX2=1`) bit for bit,
/// since both are exact integer arithmetic followed by identical float operations. Elsewhere it is a no-op.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_int8_kernel_is_bit_identical_to_the_ssse3_kernel_when_available() {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("skipped: this CPU has no AVX2");
        return;
    }
    for (exe, formats) in build_both("simd_quant_linear_avx2") {
        let avx2 = [("BORING_SIMD_THREADS", "1"), ("BORING_SIMD_QUANT_INT8", "1")];
        let ssse3 = [("BORING_SIMD_THREADS", "1"), ("BORING_SIMD_QUANT_INT8", "1"), ("BORING_SIMD_QUANT_NO_AVX2", "1")];
        compare_shapes(&exe, formats, &ssse3, &avx2, "avx2 vs ssse3", 0.0, |m| m == 1);
    }
}
