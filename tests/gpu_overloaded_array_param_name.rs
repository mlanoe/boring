// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: the GPU array-param borrow pass keyed per-position flags by function name, so
// two functions sharing a name with different parameter counts made it index past the entry
// (`index out of bounds: the len is 1 but the index is 1`). Overloaded names must stay by value.
//
// Run with:
//   cargo test --test gpu_overloaded_array_param_name

use std::process::Command;

#[test]
fn overloaded_function_names_do_not_panic_the_borrow_pass() {
    for target in ["metal", "wgpu"] {
        let dir = std::env::temp_dir().join(format!("boring_overload_{}_{}", target, std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let out = Command::new(env!("CARGO_BIN_EXE_boring"))
            .current_dir(&dir)
            .arg("build").args(["--target", target])
            .arg(std::env::current_dir().unwrap().join("tests/cases/gpu_overloaded_array_param_name.br"))
            .output().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success() && !err.contains("panicked"), "{target}: {err}");
    }
}
