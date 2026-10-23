// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a bare `x = v` for a never-declared `x` used to define it in
// `boring run` and emit undeclared Rust (E0425) in `boring build`. Both now fail
// with the same diagnostic.
//
// Run with:
//   cargo test --test assign_undeclared_fails

use std::path::Path;
use std::process::Command;

fn stderr_of(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(args)
        .arg(Path::new("tests/cases/error_assign_undeclared.br"))
        .args(if args[0] == "build" { vec!["--emit-rust"] } else { vec![] })
        .output()
        .unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn run_rejects_undeclared_assignment() {
    let (ok, err) = stderr_of(&["run"]);
    assert!(!ok);
    assert!(err.contains("cannot assign to undeclared variable 'start'"), "{err}");
}

#[test]
fn build_rejects_undeclared_assignment() {
    let (ok, err) = stderr_of(&["build"]);
    assert!(!ok);
    assert!(err.contains("cannot assign to undeclared variable `start`"), "{err}");
}
