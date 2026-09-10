// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: passing an already-moved `'owned` local variable into a
// second struct-constructor call used to raise no Boring-level diagnostic at
// all. `boring build --emit-rust` "succeeded" and emitted Rust that moves the
// same `Box<T>` twice, failing only at the later `cargo build`/`rustc` step
// with a raw E0382 pointing at generated code the user never wrote. See
// `src/checker/mod.rs`'s "Use-after-move: committed-`'owned` struct-
// constructor arguments" section for the full design (what is and isn't
// caught — deliberately narrower than "every `'owned` parameter", since a
// plain function call to an `'owned` parameter is NOT a move — see
// `tests/cases/owned_call_arg_no_double_box.br`).
//
// Run with:
//   cargo test --test owned_ctor_use_after_move_build_fails

use std::path::Path;
use std::process::Command;

fn emit_rust(case_name: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases").join(format!("{case_name}.br"));
    Command::new(bin)
        .arg("build")
        .arg(&case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e))
}

const EXPECTED_MSG: &str = "was already moved (passed to an owned parameter) here; \
                             a value can only be moved once";

// The exact repro from the bug report: an explicit `init(...)` body storing
// its argument into an `'owned` field, called twice with the same variable.
#[test]
fn ctor_reuse_across_statements_fails_boring_build() {
    let emit = emit_rust("error_owned_ctor_use_after_move");

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to fail on a struct-constructor \
         use-after-move, but it exited successfully and emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    assert!(
        stderr.contains(EXPECTED_MSG),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        EXPECTED_MSG, stderr
    );
    assert!(
        stderr.contains("`ac`"),
        "expected stderr to name the moved variable `ac`:\n{}",
        stderr
    );
}

// Same gap, but for a struct with no explicit `init` at all (the implicit,
// fully-positional constructor) — must be caught too, not just the
// explicit-`init`-body case.
#[test]
fn default_ctor_reuse_across_statements_fails_boring_build() {
    let emit = emit_rust("error_owned_default_ctor_use_after_move");

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to fail on a use-after-move \
         through the implicit (no-`init`) constructor, but it exited \
         successfully and emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    assert!(
        stderr.contains(EXPECTED_MSG),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        EXPECTED_MSG, stderr
    );
}

// Same variable passed to two different `'owned` params in ONE call
// (`Pair(ac, ac)`) — must be caught immediately, not just across statements.
#[test]
fn double_move_in_same_call_fails_boring_build() {
    let emit = emit_rust("error_owned_double_move_same_call");

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to fail on a double-owned-move \
         within a single constructor call, but it exited successfully and \
         emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    assert!(
        stderr.contains(EXPECTED_MSG),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        EXPECTED_MSG, stderr
    );
}

// False-positive guard: two DISTINCT `'owned` variables, each moved into its
// own constructor call, must never be flagged.
#[test]
fn distinct_vars_into_two_ctors_still_succeeds() {
    let emit = emit_rust("ok_owned_distinct_vars_two_ctors");

    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to succeed for two distinct \
         `'owned` variables each moved into their own constructor, but it \
         failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
}

// False-positive guard: reusing an `'owned` variable after passing it to a
// PLAIN FUNCTION call (not a constructor) must never be flagged — the
// transpiler clones the box at that call site instead of moving it, so the
// variable stays legitimately usable afterward. This is the existing,
// already-passing `tests/cases/owned_call_arg_no_double_box.br` regression
// test (see `tests/transpile.rs`); re-asserted here, colocated with the new
// check that could very easily have broken it (an earlier draft of this
// check did exactly that, treating every `'owned` parameter as a move).
#[test]
fn plain_function_call_reuse_still_succeeds() {
    let emit = emit_rust("owned_call_arg_no_double_box");

    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to succeed when reusing an \
         `'owned` variable after a plain (non-constructor) function call, \
         but it failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
}
