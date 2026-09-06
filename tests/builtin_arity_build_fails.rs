// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `emit_builtin_call` (src/transpiler/emit_expr.rs) indexed
// `args[0]`/`args[1]`/`args[2]` unconditionally for dozens of builtins (`len`,
// `assert_eq`, `sum`, `clamp`, `atan2`, `log`, `pow`, `ord`, `chr`, `exit`,
// `drop`, `json`, every math function, every numeric cast, ...) with no
// arity check anywhere — neither the parser nor the checker validates
// builtin call arity. Calling one with too few arguments (`len()`) panicked
// the transpiler itself with a raw Rust index-out-of-bounds instead of
// failing `boring build` with a clean diagnostic.
//
// `emit_call` now checks `builtin_min_arity(name)` before ever delegating to
// `emit_builtin_call`, pushing a proper `push_error` (which makes `boring
// build` exit non-zero per `main.rs`'s `if !out.errors.is_empty()`) instead
// of letting the match/guard indexing panic.
//
// Run with:
//   cargo test --test builtin_arity_build_fails

use std::path::Path;
use std::process::Command;

#[test]
fn len_with_no_args_fails_cleanly_not_a_panic() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/error_builtin_arity_len.br");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to fail on `len()` with no arguments, \
         but it exited successfully and emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "must fail with a clean diagnostic, not a Rust panic — actual stderr:\n{}",
        stderr
    );
    let expected = "`len` expects at least 1 argument";
    assert!(
        stderr.contains(expected),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        expected, stderr
    );
}
