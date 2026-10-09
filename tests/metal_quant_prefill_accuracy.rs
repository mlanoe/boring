// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later

#[cfg(target_os = "macos")]
mod macos {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn build(root: &Path, config: &str) -> PathBuf {
        fs::write(root.join("boring.toml"), config).unwrap();
        let source = root.join("prefill.br");
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", "metal"])
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let manifest = root.join("prefill_metal/Cargo.toml");
        let cargo = Command::new("cargo")
            .args(["build", "--release", "--quiet", "--manifest-path"])
            .arg(&manifest)
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap();
        assert!(
            cargo.status.success(),
            "{}",
            String::from_utf8_lossy(&cargo.stderr)
        );
        root.join("prefill_metal/target/release/prefill")
    }

    fn run(executable: &Path, format: &str, m: usize, n: usize, k: usize, with_bias: bool) -> Vec<f32> {
        let output = Command::new(executable)
            .args([
                format,
                &m.to_string(),
                &n.to_string(),
                &k.to_string(),
                if with_bias { "bias" } else { "no-bias" },
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                line.strip_prefix("value=")
                    .map(|value| value.parse().unwrap())
            })
            .collect()
    }

    /// Builds `config` in its own directory under `base`.  Every configuration
    /// needs a private Cargo project: `build` always emits to
    /// `<dir>/prefill_metal/target/release/prefill`, so sharing a directory
    /// would let a later build overwrite an earlier executable.
    fn build_in(base: &Path, name: &str, config: &str) -> PathBuf {
        let root = base.join(name);
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("prefill.br"),
            include_str!("fixtures/metal_quant_prefill_bias.br"),
        )
        .unwrap();
        build(&root, config)
    }

    /// Q8_0 needs k % 32 == 0, the K-quants need k % 256 == 0.  Every list
    /// has an odd n (odd Q6_K columns start at a 2-mod-4 byte address) and
    /// m, n that are not multiples of the 64/128-row, 64-column tiles.
    fn shapes(precision: &str, format: &str) -> Vec<(usize, usize, usize)> {
        let k_quant = format != "q8_0";
        if precision == "float" {
            vec![
                (613, 896, 4864),
                (613, 4864, if k_quant { 1024 } else { 896 }),
                (37, 77, 512),
                (8, 33, 256),
                (130, 129, 256),
            ]
        } else {
            vec![(193, 160, 1024), (131, 75, 512), (9, 65, 256)]
        }
    }

    #[test]
    fn native_quantized_prefill_precision_is_bounded_against_scalar() {
        let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("metal_quant_prefill_accuracy");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        let scalar_exe = build_in(&base, "scalar", "[tensor.linear.prefill]\nalgorithm = \"scalar\"\n");
        let configs = [
            (
                "float",
                Some(1.0e-5f32),
                "[tensor.linear.prefill]\nalgorithm = \"native\"\nprecision = \"float\"\n",
            ),
            (
                "half",
                None,
                "[tensor.linear.prefill]\nalgorithm = \"native\"\nprecision = \"half\"\n",
            ),
        ];
        for (precision, tolerance, config) in configs {
            let native_exe = build_in(&base, precision, config);
            // A shared executable makes this test compare native with itself.
            assert_ne!(
                fs::canonicalize(&scalar_exe).unwrap(),
                fs::canonicalize(&native_exe).unwrap(),
                "scalar and {precision} native builds must be different executables"
            );
            for format in ["q8_0", "q4_k", "q6_k"] {
                for (m, n, k) in shapes(precision, format) {
                    for with_bias in [false, true] {
                        let scalar = run(&scalar_exe, format, m, n, k, with_bias);
                        assert_eq!(scalar.len(), m * n);
                        let scale = scalar.iter().copied().map(f32::abs).fold(0.0, f32::max);
                        for launch in 0..2 {
                            let actual = run(&native_exe, format, m, n, k, with_bias);
                            assert_eq!(actual.len(), scalar.len());
                            let mut worst = 0.0f32;
                            for (index, (&a, &s)) in actual.iter().zip(scalar.iter()).enumerate() {
                                let error = (a - s).abs();
                                worst = worst.max(error);
                                if let Some(tolerance) = tolerance {
                                    // Without bias native and scalar agree bit for bit, so the
                                    // strict per-element bound holds.  The scalar kernel seeds its
                                    // accumulator with the bias while native adds it afterwards,
                                    // which shifts every rounding step by an ulp of the partial
                                    // sum (up to ~3e-7 of the output scale at k = 4864); bound
                                    // that case by the output scale.  A wrong column is off by
                                    // ~1x the scale, five orders of magnitude above either bound.
                                    let bound = if with_bias { tolerance * scale } else { tolerance * (s.abs() + 1.0) };
                                    assert!(error <= bound,
                                        "{format} {precision} {m}x{n}x{k} bias={with_bias} launch {launch}, output {index} (row {}, col {}): native={a}, scalar={s}, error={error}",
                                        index / n, index % n);
                                }
                            }
                            assert!(
                                precision != "half" || worst <= 1.0e-3 * scale,
                                "{format} half {m}x{n}x{k} bias={with_bias} launch {launch}: worst error {worst}, scalar scale {scale}"
                            );
                            eprintln!(
                                "{format} {precision} {m}x{n}x{k} bias={with_bias} launch {launch}: worst absolute error {worst}"
                            );
                        }
                    }
                }
            }
        }
    }
}
