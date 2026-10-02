// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a bare call to a sibling struct method inside a method body (`one(n)` where
// `one` is declared in the same struct) was emitted unchanged -- `one(n.clone());` -- which
// fails `cargo build` with rustc E0425 ("cannot find function `one`"), a Rust error invisible
// from the Boring source. docs/book.md ("Implicit `self`") documents that implicit `self`
// covers fields only and a sibling method needs `self.`, so `boring build` now rejects the
// bare call with a Boring diagnostic that names the fix.
//
// Run with:
//   cargo test --test bare_sibling_method_call_build_fails

use std::path::Path;
use std::process::Command;

#[test]
fn bare_sibling_method_call_fails_with_boring_diagnostic() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/error_bare_sibling_method_call.br");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to reject the bare sibling-method call, \
         but it exited successfully and emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "must fail with a clean diagnostic, not a Rust panic — actual stderr:\n{}",
        stderr
    );
    let expected = "cannot call method 'one' without a receiver inside struct 'Src'";
    assert!(
        stderr.contains(expected) && stderr.contains("self.one(...)"),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        expected, stderr
    );
}
