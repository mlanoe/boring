// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression: a regular string with an unbalanced `{{` (e.g. `"{{"`) used to make
// the `$` preprocessing scan treat it as an open interpolation hole, swallowing
// the rest of the file so a later `$"..."` failed with "unexpected character '$'".
//
// Run with:
//   cargo test --test dollar_after_unbalanced_brace

use std::path::Path;
use std::process::Command;

const CASE: &str = "tests/cases/dollar_after_unbalanced_brace.br";

fn boring(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(args)
        .arg(Path::new(CASE))
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn run_lexes_dollar_string_after_unbalanced_double_brace() {
    let (ok, stdout, stderr) = boring(&["run"]);
    assert!(ok, "{stderr}");
    let expected = std::fs::read_to_string(Path::new(CASE).with_extension("expected")).unwrap();
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        expected.lines().collect::<Vec<_>>(),
        "{stdout}"
    );
}

#[test]
fn build_emits_rust_for_dollar_string_after_unbalanced_double_brace() {
    let (ok, stdout, stderr) = boring(&["build", "--emit-rust"]);
    assert!(ok, "{stderr}");
    assert!(stdout.contains("\"{\""), "{stdout}");
}
