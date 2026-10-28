// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `$"..."` / `$"""..."""` strings: `$name` and `${expr}` interpolate, `{` and `}` are
// literal, `$$` is a literal `$`. Checked in `boring run` (interpreter) and
// `boring build --emit-rust` (transpiler).
//
// Run with:
//   cargo test --test dollar_string

use std::path::Path;
use std::process::Command;

const CASE: &str = "tests/cases/dollar_string.br";

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
fn run_interpolates_with_dollar_and_keeps_braces_literal() {
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
fn build_turns_dollar_holes_into_format_args() {
    let (ok, stdout, stderr) = boring(&["build", "--emit-rust"]);
    assert!(ok, "{stderr}");
    // `$name`/`${expr}` become real format arguments; literal braces stay escaped.
    assert!(stdout.contains("hello {}, next year {}"), "{stdout}");
    assert!(stdout.contains("{{\\\"id\\\": \\\"{}\\\""), "{stdout}");
    assert!(stdout.contains("cost $5 and $1 and {{braces}} and end$"), "{stdout}");
}
