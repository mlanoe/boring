// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `emit_args_coerced` (src/transpiler/emit_methods.rs)
// sliced `args[i..]` where `i` is the callee's declared variadic parameter
// position, once the emission loop reaches it. A call site passing fewer
// arguments than that position (never validated by the checker) makes `i`
// exceed `args.len()`, and the raw slice panicked the transpiler with a raw
// Rust index-out-of-bounds instead of degrading gracefully.
//
// `greet(string name, string tags...)` called as `greet()` (zero arguments,
// variadic position 1) used to panic; the slice is now clamped
// (`args.get(i.min(args.len())..)`), so this no longer panics the compiler
// itself — the resulting emitted Rust is still invalid for a genuinely
// wrong call (a missing required argument), which real `cargo build` would
// separately reject; this test only asserts the transpiler doesn't crash.
//
// Run with:
//   cargo test --test variadic_arity_no_panic

use std::path::Path;
use std::process::Command;

#[test]
fn variadic_call_with_too_few_args_does_not_panic_the_transpiler() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/error_variadic_too_few_args.br");

    let emit = Command::new(bin)
        .arg("build")
        .arg("--emit-rust")
        .arg(case_br)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    let stderr = String::from_utf8_lossy(&emit.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "greet() with too few arguments for a variadic parameter must not \
         panic the transpiler — actual stderr:\n{}",
        stderr
    );
}
