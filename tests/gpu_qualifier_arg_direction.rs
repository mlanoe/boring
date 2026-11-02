// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: passing a `[T]'kernel'unified` argument to a `[T]'kernel'global`
// parameter is accepted (unified is a superset: host + device visible), but the
// reverse — a device-only `'kernel'global` value to a `'kernel'unified` parameter —
// is a checker error on every GPU target (on wgpu a `'global` buffer lacks `COPY_SRC`,
// so the callee's host readback would fail validation at run time).
//
// Run with:
//   cargo test --test gpu_qualifier_arg_direction

use std::process::Command;

fn build(target: &str, file: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("boring_qualdir_{}_{}", target, std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .current_dir(&dir)
        .arg("build").args(["--target", target])
        .arg(std::env::current_dir().unwrap().join(file))
        .output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn unified_argument_to_global_parameter_is_accepted() {
    for target in ["metal", "wgpu"] {
        let (ok, err) = build(target, "tests/cases/gpu_unified_arg_to_global_param.br");
        assert!(ok, "{target}: {err}");
    }
}

#[test]
fn global_argument_to_unified_parameter_is_rejected() {
    for target in ["metal", "wgpu"] {
        let (ok, err) = build(target, "tests/cases/error_gpu_global_arg_to_unified_param.br");
        assert!(!ok, "{target} succeeded");
        assert!(err.contains("expects `'kernel'unified` (host-visible), got a device-only `'kernel'global` value"), "{target}: {err}");
    }
}
