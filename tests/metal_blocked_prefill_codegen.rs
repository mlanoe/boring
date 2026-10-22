// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Codegen checks for the Metal SIMD-group prefill of Q4_0 / Q5_0 / IQ4_NL / Q3_K / Q2_K (formats
// without a dedicated native kernel). Real-GPU accuracy is covered by `metal_quant_prefill_accuracy`.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn metal_codegen(name: &str, format: &str, config: Option<&str>) -> (String, String) {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("metal_blocked_prefill_codegen").join(name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let src = format!(
        "req [float32]'kernel'unified compute([float32]'kernel'global a, [uint8]'kernel'global weights, [float32]'kernel'global bias, int m, int n, int k) throws:\n    mut [float32]'kernel'unified y = [0.0 as float32 for ..<m * n]\n    kernel.tensors.linear(a, weights, bias, y, m = m, n = n, k = k, format = \"{format}\")\n    y\n"
    );
    fs::write(tmp.join("test.br"), src).unwrap();
    if let Some(config) = config { fs::write(tmp.join("boring.toml"), config).unwrap(); }
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "metal"]).arg(tmp.join("test.br")).output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    (
        fs::read_to_string(tmp.join("test_metal/kernels/main.metal")).unwrap(),
        fs::read_to_string(tmp.join("test_metal/src/main.rs")).unwrap(),
    )
}

#[test]
fn non_native_formats_prefill_with_simdgroup_matrices_and_keep_warp_decode() {
    for (format, block_bytes) in [("q4_0", "* 18;"), ("q5_0", "* 22;"), ("iq4_nl", "* 18;"), ("q3_k", "* 110;"), ("q2_k", "* 84;")] {
        let (msl, host) = metal_codegen(&format!("{format}_default"), format, None);
        assert!(msl.contains("simdgroup_multiply_accumulate"), "{format}: {msl}");
        assert!(msl.contains(block_bytes), "{format}: block stride {block_bytes} expected");
        assert!(!msl.contains("aTile"), "{format} must not declare portable tiles");
        assert!(!msl.contains("__boring_metal_blocked_prefill"), "{format}: marker must be expanded");
        // Decode (m == 1) keeps the SIMD-group reduction kernel.
        assert!(msl.contains("simd_sum") || msl.contains("simd_shuffle"), "{format}: warp decode expected");
        // 64x64 output tiles for prefill.
        assert!(host.contains("+ 63) / 64"), "{format}: {host}");
    }
}

#[test]
fn small_block_format_specifics_are_emitted() {
    let (iq4, _) = metal_codegen("iq4_table", "iq4_nl", None);
    assert!(iq4.contains("-127, -104, -83, -65"), "{iq4}");
    let (q5, _) = metal_codegen("q5_high_bits", "q5_0", None);
    assert!(q5.contains("bp_qh"), "{q5}");
}

#[test]
fn scalar_prefill_override_disables_the_simdgroup_kernel() {
    let (msl, _) = metal_codegen("q4_scalar", "q4_0", Some("[tensor.linear.prefill]\nalgorithm = \"scalar\"\n"));
    assert!(!msl.contains("simdgroup_multiply_accumulate"), "{msl}");
}
