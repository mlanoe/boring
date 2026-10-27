// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `'''...'''` raw string literal: no interpolation, no escape processing. Checked
// in `boring run` (interpreter) and `boring build --emit-rust` (transpiler).
//
// Run with:
//   cargo test --test raw_apostrophe_string

use std::path::Path;
use std::process::Command;

const CASE: &str = "tests/cases/raw_apostrophe_string.br";

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
fn run_keeps_quotes_braces_and_backslashes_literal() {
    let (ok, stdout, stderr) = boring(&["run"]);
    assert!(ok, "{stderr}");
    // The JSON text keeps its unescaped quotes/braces, the path
    // `C:\temp\n {not_a_hole}` is 22 chars (backslashes and `{...}` untouched),
    // and the multiline body is dedented like `"""`: `SELECT * FROM t` (15) + `\n` +
    // `  WHERE data->>'k' = "v"` (24) = 40, closing `'''` at column 0 so no trailing `\n`.
    // The expected output is the case's `.expected` file, shared with the
    // self-hosted interpreter's functional suite.
    let expected = std::fs::read_to_string(Path::new(CASE).with_extension("expected")).unwrap();
    assert_eq!(
        stdout.lines().collect::<Vec<_>>(),
        expected.lines().collect::<Vec<_>>(),
        "{stdout}"
    );
}

#[test]
fn build_emits_the_literal_text_unchanged() {
    let (ok, stdout, stderr) = boring(&["build", "--emit-rust"]);
    assert!(ok, "{stderr}");
    assert!(
        stdout.contains(r#"{\"name\": \"Alice\", \"age\": 30, \"addr\": {\"city\": \"Paris\"}}"#),
        "{stdout}"
    );
    assert!(stdout.contains(r"C:\\temp\\n {not_a_hole}"), "{stdout}");
}
