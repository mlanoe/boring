// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `pre_scan` (src/transpiler/mod.rs) called
// `self.emit_expr_owned(s.value.as_ref().unwrap())` for every top-level
// mutable `var` item, unconditionally — but `LetStmt.value` is `None` for a
// deferred-init declaration with no `= expr` (`var int x`), a case the
// parser explicitly supports and the checker never rejects. The local
// (in-function) equivalent of this same code path, `emit_let.rs`, already
// guards `s.value.is_none()` before touching it; `pre_scan` (which runs
// first, on top-level items) did not, and `boring build` panicked with
// `unwrap()` on `None` instead of emitting a plain deferred-init `let mut x: T;`.
//
// Run with:
//   cargo test --test top_level_var_no_initializer

use std::path::Path;
use std::process::Command;

#[test]
fn top_level_var_with_no_initializer_transpiles_cleanly() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/top_level_var_no_initializer.br");

    let emit = Command::new(bin)
        .arg("build")
        .arg("--emit-rust")
        .arg(case_br)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    let stderr = String::from_utf8_lossy(&emit.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "must not panic on a top-level `var` with no initializer — actual stderr:\n{}",
        stderr
    );
    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to succeed on a top-level `var` with \
         no initializer, but it failed:\n{}",
        stderr
    );

    let code = String::from_utf8_lossy(&emit.stdout).into_owned();
    assert!(
        code.contains("let mut x: isize;"),
        "expected a plain deferred-init `let mut x: isize;`, got:\n{}",
        code
    );
}
