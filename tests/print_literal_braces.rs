// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `print` of a literal containing braces must re-escape them in the emitted
// `println!` format string (`{{}}` -> `println!("{{}}")`, not `println!("{}")`).
//
// Run with:
//   tools/test.sh --test print_literal_braces

use std::path::Path;
use std::process::Command;

const CASE: &str = "tests/cases/print_literal_braces.br";

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
fn run_prints_literal_braces() {
    let (ok, stdout, stderr) = boring(&["run"]);
    assert!(ok, "{stderr}");
    assert_eq!(stdout.lines().collect::<Vec<_>>(), ["{}", "a {x} b", "v=5 {lit}", "{}"], "{stdout}");
}

#[test]
fn build_reescapes_braces_in_format_string() {
    let (ok, stdout, stderr) = boring(&["build", "--emit-rust"]);
    assert!(ok, "{stderr}");
    assert!(stdout.contains(r#"println!("{{}}")"#), "{stdout}");
    assert!(stdout.contains(r#"println!("a {{x}} b")"#), "{stdout}");
    assert!(stdout.contains(r#"println!("v={} {{lit}}", x)"#), "{stdout}");
    assert!(!stdout.contains(r#"println!("{}")"#), "{stdout}");
}
