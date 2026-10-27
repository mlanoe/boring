// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a string-method pattern argument that already carries an explicit
// `.asRef()` (`hay.contains(needle.asRef())`, written before such receivers were typed as
// strings) was coerced a second time, `needle.as_ref().as_ref()`, which rustc cannot infer
// (E0283). The explicit coercion is now left alone; a plain `needle` still gets one `.as_ref()`.
//
// Run with:
//   cargo test --test string_pattern_explicit_as_ref

use std::process::Command;

#[test]
fn explicit_as_ref_is_not_doubled() {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--emit-rust", "tests/cases/string_pattern_explicit_as_ref.br"])
        .output()
        .expect("failed to invoke boring");
    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let rust = String::from_utf8_lossy(&out.stdout);
    assert!(!rust.contains("as_ref().as_ref()"), "{rust}");
    assert_eq!(rust.matches("hay.contains(needle.as_ref())").count(), 2, "{rust}");
}
