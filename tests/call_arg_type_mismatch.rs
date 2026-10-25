// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a call argument whose type is positively known and differs
// from the declared parameter type is a checker error in `run` and `build`
// (previously only rustc caught it, after `boring build`).
//
// Run with:
//   cargo test --test call_arg_type_mismatch

use std::path::Path;
use std::process::Command;

fn run_mode(mode: &str) -> (bool, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_boring"));
    cmd.arg(mode).arg(Path::new("tests/cases/error_call_arg_type_mismatch.br"));
    if mode == "build" { cmd.arg("--emit-rust"); }
    let out = cmd.output().unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn run_and_build_reject_swapped_arguments() {
    for mode in ["run", "build"] {
        let (ok, err) = run_mode(mode);
        assert!(!ok, "{mode} succeeded");
        assert!(err.contains("argument 1 of selectionOf (parameter 'rt') expects Runtime, got Handle"), "{mode}: {err}");
        assert!(err.contains("argument 2 of selectionOf (parameter 'h') expects Handle, got Runtime"), "{mode}: {err}");
    }
}

#[test]
fn run_rejects_swapped_arguments_to_an_imported_function() {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .current_dir("tests/cases/fixtures/arg_type_import")
        .args(["run", "main.br"])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(err.contains("argument 1 of selectionOf (parameter 'rt') expects Runtime, got Handle"), "{err}");
}
