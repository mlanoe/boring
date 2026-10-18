// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `try h.wait else 0`: `.wait` on a task handle has no value (unit), so a value default
// can't stand in for it — the transpiled Rust can't type the two arms, and `boring run`
// would yield `nil` on success. The checker rejects the literal-default form for both
// `boring run` and `boring build`.
//
// Run with:
//   cargo test --test try_else_wait_value_fails

use std::process::Command;

fn run_boring(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(args)
        .output()
        .expect("failed to invoke boring");
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn try_wait_else_value_is_a_checker_error() {
    let case = "tests/cases/error_try_else_wait_value.br";
    for args in [vec!["run", case], vec!["build", case, "--emit-rust"]] {
        let (ok, stderr) = run_boring(&args);
        assert!(!ok, "`boring {}` must fail on `try k.wait else 0`", args[0]);
        assert!(
            stderr.contains("`.wait` yields no value (unit)"),
            "`boring {}`: expected the .wait diagnostic, got:\n{}", args[0], stderr
        );
    }
}

#[test]
fn try_value_and_try_optional_wait_still_accepted() {
    // The documented alternatives keep working.
    let (ok, stderr) = run_boring(&["run", "tests/cases/spawn_throws_wait.br"]);
    assert!(ok, "spawn_throws_wait must still run, stderr:\n{}", stderr);
}
