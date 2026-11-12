// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a GPU-resident local bound in one `if` branch leaked into a same-named
// host-typed local bound in the sibling `else` branch, so `weights[k]` there was rejected with
// "`weights` is GPU-resident and cannot be indexed directly outside a `with weights:` block".
//
// Run with:
//   cargo test --test gpu_resident_branch_scope

use std::process::Command;

#[test]
fn resident_local_does_not_leak_into_sibling_branch() {
    for target in ["metal", "wgpu"] {
        let dir = std::env::temp_dir().join(format!("boring_branchscope_{}_{}", target, std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let out = Command::new(env!("CARGO_BIN_EXE_boring"))
            .current_dir(&dir)
            .arg("build").args(["--target", target])
            .arg(std::env::current_dir().unwrap().join("tests/cases/gpu_resident_branch_scope.br"))
            .output().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{target}: {err}");
    }
}
