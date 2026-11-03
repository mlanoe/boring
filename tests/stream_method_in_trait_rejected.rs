// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `stream` is accepted on struct/enum/ext methods but not (yet) on trait method signatures.
// That used to fail with a bare "expected Colon, got Newline"; it must now be a clear
// diagnostic under both `boring run` and `boring build`.
//
// Run with:
//   cargo test --test stream_method_in_trait_rejected

use std::path::Path;
use std::process::Command;

fn check(args: &[&str]) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case = Path::new("tests/cases/error_stream_method_in_trait.br");
    let out = Command::new(bin)
        .args(args)
        .arg(case)
        .args(if args[0] == "build" { &["--emit-rust"][..] } else { &[][..] })
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(!out.status.success(), "`boring {}` must reject a `stream` trait method", args[0]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked at"), "must be a clean diagnostic:\n{}", stderr);
    assert!(
        stderr.contains("`stream` methods cannot be declared in a trait yet"),
        "unexpected diagnostic:\n{}", stderr
    );
}

#[test]
fn run_rejects_stream_trait_method() { check(&["run"]); }

#[test]
fn build_rejects_stream_trait_method() { check(&["build"]); }
