// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `try_emit_dict_method`'s "contains" | "containsKey" | "has"
// arm (src/transpiler/emit_methods.rs) indexed `args[0]` unconditionally — the
// only arm in that match with no `args.len()` guard, unlike `get`, `set`/`put`,
// and `remove`, which all check first. `dict.contains()` with no key argument
// panicked the transpiler itself with a raw Rust index-out-of-bounds instead
// of falling through to the generic method-call path.
//
// The arm now requires `!args.is_empty()`, same convention as `remove`. A
// zero-arg call falls through to the generic fallback instead, which degrades
// to emitting `contains_key()` with no argument — invalid Rust that `cargo
// build` would separately reject, but the *transpiler* itself no longer
// panics. This test only asserts the latter (no panic); it does not attempt
// the (expected-to-fail) downstream `cargo build`.
//
// Run with:
//   cargo test --test dict_method_arity_no_panic

use std::path::Path;
use std::process::Command;

#[test]
fn dict_contains_with_no_args_does_not_panic_the_transpiler() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/error_dict_contains_no_args.br");

    let emit = Command::new(bin)
        .arg("build")
        .arg("--emit-rust")
        .arg(case_br)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    let stderr = String::from_utf8_lossy(&emit.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "dict.contains() with no arguments must not panic the transpiler — \
         actual stderr:\n{}",
        stderr
    );
}
