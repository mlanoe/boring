// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: the speculative qualifier pre-inference ran per file during the deep
// pre-scan, before a *later* `use` in the entry file (`use big` after `use values`) had
// registered the struct that file's functions take. It then saw `Big` params as unknown, left
// the callee signature owned, and callers passed owned values to a callee the real pass emitted
// as `&Big` (rustc E0308 on the generated code). The pass now runs after the whole scan.
//
// Run with:
//   cargo test --test pre_infer_late_use

use std::process::Command;

#[test]
fn caller_borrows_param_of_struct_registered_by_later_use() {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--emit-rust", "tests/cases/pre_infer_late_use/main.br"])
        .output()
        .expect("failed to invoke boring");
    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let rust = String::from_utf8_lossy(&out.stdout);
    assert!(rust.contains("fn tier(a: &Big, b: &Big, c: &Big, d: &Big)"), "{rust}");
    assert!(rust.contains("tier(&p, &q, &r, &s)"), "{rust}");
}
