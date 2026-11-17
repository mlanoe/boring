// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a call missing a required (non-defaulted) argument used to
// be only a checker *warning*. `boring check` (which did not run the checker
// at all) exited 0, `boring run` warned and ran anyway, and `boring build`
// emitted Rust that rustc rejected with E0061 — buried under thousands of
// warnings in a real CI log (boring-self-interpreter, a dropped 4th argument
// of `byvalue_store(...)`). All three commands must now fail up front with
// the checker's diagnostic.
//
// Run with:
//   cargo test --test missing_required_argument

use std::process::Command;

const CASE: &str = "tests/cases/error_missing_required_argument.br";
const MSG: &str = "missing required argument 'c' in call to 'f'";

fn assert_rejected(args: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(args)
        .arg(CASE)
        .output()
        .expect("failed to invoke boring");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "`boring {}` must fail, stderr:\n{stderr}", args.join(" "));
    assert!(
        stderr.contains(&format!("error: {MSG}")),
        "`boring {}` must report `error: {MSG}`, stderr:\n{stderr}",
        args.join(" ")
    );
}

#[test]
fn check_rejects_missing_required_argument() {
    assert_rejected(&["check"]);
}

#[test]
fn run_rejects_missing_required_argument() {
    assert_rejected(&["run"]);
}

#[test]
fn build_rejects_missing_required_argument() {
    assert_rejected(&["build", "--emit-rust"]);
}
