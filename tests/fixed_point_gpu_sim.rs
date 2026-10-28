// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
//  simulation of a fixed-point kernel agrees with the semantics the
// CUDA / HIP lowering emits (tests/cuda_codegen.rs):  truncates the 2W-bit
// product,  shifts the dividend first, untyped literals are exact.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn fixed_point_kernel_simulation_matches_gpu_integer_semantics() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fixed_point_gpu_sim");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, r#"let N = 4
mut [fixed16_16]'kernel'unified x = [0.0 as fixed16_16 for ..<N]
var [fixed16_16]'kernel'unified y = [1.0 as fixed16_16 for ..<N]
for i in 0..<N:
    x[i] = (i as fixed16_16) + 0.5

kernel Fx:
    let fixed16_16 alpha
    let [fixed16_16]'unified x
    mut [fixed16_16]'unified y
    init(fixed16_16 a, [fixed16_16]'unified xs, [fixed16_16]'unified ys):
        alpha = a
        x = xs
        y = ys
    def ():
        let i = .thread.x
        y[i] = alpha * x[i] + y[i] / 3 - 0.25

mut k = Fx(2.5, x, y)
kernel:
    k(grid = 1, block = N)
for i in 0..<N:
    print "{k.y[i]}"
"#).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&br).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    // y = alpha * x + y / 3 - 0.25 on raw 16.16 integers, e.g. i = 0:
    // 2.5 * 0.5 = 81920, 65536 / 3 = 21845 (truncated), - 16384 => 87381 / 65536
    let got: Vec<&str> = stdout.lines().collect();
    assert_eq!(got, ["1.3333282470703125", "3.8333282470703125", "6.3333282470703125", "8.8333282470703125"], "{stdout}");
}
