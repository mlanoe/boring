// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: calling a `def` (mutating) method on a `T'owned` (or a `T'new`
// resolved to `Owned`/`Inline`, or a plain `T'inline`) parameter that was NOT
// declared `mut`/`var` used to raise no Boring-level diagnostic at all —
// `boring build --emit-rust` "succeeded" and emitted `fn bump(c: Box<Counter>)`
// (missing the `mut`), which then failed `cargo build`/`rustc` with a confusing
// E0596 ("cannot borrow `*c` as mutable, as `c` is not declared as mutable").
//
// Root cause: the existing "not declared `mut`" parameter diagnostic
// (`emit_methods.rs`'s `emit_method_call_fallback`, and the matching field-assign
// diagnostic in `emit_expr.rs`) only resolved the receiver's struct name for a
// bare `Type::Named` parameter — a `Type::Qualified(_, OwnerQual::Owned)` (or
// `Inline`, or a `Union`/`'new` resolved to one of those) parameter type never
// matched, so the struct name resolved to `None` and the whole diagnostic was
// silently skipped. `Type::Named`, `Owned`, and `Inline` params are all emitted
// as a *direct* Rust value (`T`/`Box<T>`, no `&`) by `emit_top.rs`'s `emit_param`,
// which only adds `mut` when the Boring source explicitly wrote `mut`/`var` on
// the parameter — so all three need the same Boring-level enforcement.
//
// This test covers the exact repro from the bug report (`Counter'owned c`).
// See `tests/cases/error_owned_param_def_call_without_mut.br`.
//
// Run with:
//   cargo test --test owned_param_mut_def_call_build_fails

use std::path::Path;
use std::process::Command;

#[test]
fn owned_param_def_call_without_mut_fails_boring_build() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/error_owned_param_def_call_without_mut.br");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to fail on a `def` method call through a \
         non-`mut` `'owned` parameter, but it exited successfully and emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    let expected = "`c` is not declared `mut` — cannot call `def` method `.inc()` on an \
                     immutable binding; fix: declare the parameter as `mut Counter c`";
    assert!(
        stderr.contains(expected),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        expected, stderr
    );
}

// The `mut`-declared form of the same program (the confirmed workaround) must
// still transpile cleanly — this diagnostic must not become a false positive
// once the parameter is properly declared `mut`.
#[test]
fn owned_param_def_call_with_mut_still_succeeds() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/ok_owned_param_def_call_with_mut.br");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to succeed for a `mut`-declared `'owned` \
         parameter, but it failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let stdout = String::from_utf8_lossy(&emit.stdout);
    assert!(
        stdout.contains("fn bump(mut c: Box<Counter>)"),
        "expected emitted Rust to declare `mut c: Box<Counter>`, got:\n{}",
        stdout
    );
}
