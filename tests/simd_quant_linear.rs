// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Runtime checks of the quantized `kernel.tensors.linear` on the CPU SIMD target.
//
// The two real-GPU fixtures (`metal_quant_prefill_bias.br`: Q8_0/Q4_K/Q6_K and
// `metal_tiled_quant_prefill.br`: Q4_0/Q5_0/IQ4_NL/Q3_K/Q2_K) are plain Boring programs, so they
// also build for `--target simd`. Each is compiled once, then run under different runtime
// configurations and compared with the reference: one worker and the original per-element
// dequantization (`BORING_SIMD_THREADS=1 BORING_SIMD_QUANT_REFERENCE=1`).
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

fn compare_with_reference(exe: &Path, formats: &[&str], env: &[(&str, &str)], label: &str, tolerance: f32) {
    for &format in formats {
        for (m, n, k) in shapes(format) {
            for bias in [false, true] {
                let reference = run(exe, format, m, n, k, bias,
                    &[("BORING_SIMD_THREADS", "1"), ("BORING_SIMD_QUANT_REFERENCE", "1")]);
                assert_eq!(reference.len(), m * n, "{format} {m}x{n}x{k}");
                let scale = reference.iter().copied().map(f32::abs).fold(0.0, f32::max);
                assert!(scale.is_finite() && scale > 0.0, "{format} {m}x{n}x{k}: degenerate reference");
                let actual = run(exe, format, m, n, k, bias, env);
                assert_eq!(actual.len(), reference.len(), "{label} {format} {m}x{n}x{k}");
                for (index, (&a, &r)) in actual.iter().zip(&reference).enumerate() {
                    // The block dequantizers reproduce the reference weights operation for operation
                    // and the parallel path computes the same cells in the same order, so the
                    // tolerance is zero; a schedule that reorders sums would pass a relative bound.
                    assert!((a - r).abs() <= tolerance * scale,
                        "{label} {format} {m}x{n}x{k} bias={bias} output {index} (row {}, col {}): got {a}, reference {r}",
                        index / n, index % n);
                }
            }
        }
    }
}

const FIXTURE_LARGE: &str = include_str!("fixtures/metal_quant_prefill_bias.br");
const FIXTURE_SMALL: &str = include_str!("fixtures/metal_tiled_quant_prefill.br");

#[test]
fn block_dequantizers_and_parallel_workers_are_bit_identical_to_the_reference_path() {
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_quant_linear");
    let _ = fs::remove_dir_all(&base);
    let large = build(&base.join("large"), FIXTURE_LARGE);
    let small = build(&base.join("small"), FIXTURE_SMALL);
    // A zero threshold sends even the tiniest shape through the worker queue.
    let parallel = [("BORING_SIMD_QUANT_THRESHOLD", "0")];
    let single = [("BORING_SIMD_THREADS", "1")];
    for (exe, formats) in [(&large, &["q8_0", "q4_k", "q6_k"][..]), (&small, &["q4_0", "q5_0", "iq4_nl", "q3_k", "q2_k"][..])] {
        // New dequantizers on one worker, then on forced workers.
        compare_with_reference(exe, formats, &single, "block dequant, one worker", 0.0);
        compare_with_reference(exe, formats, &parallel, "block dequant, parallel", 0.0);
    }
}
