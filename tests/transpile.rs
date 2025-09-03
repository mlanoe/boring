// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Transpiler consistency tests.
//
// Each test:
//   1. Runs `boring --emit-rust <case>.br`  → generates a Cargo project
//   2. Runs `cargo run` on the generated project
//   3. Compares stdout to the same `<case>.expected` used by the interpreter tests
//
// Run with:
//   cargo test --test transpile
//
// The first run compiles all generated Rust projects from scratch (slow).
// Subsequent runs reuse the cargo build cache (fast, < 2s per test).

use std::path::Path;
use std::process::Command;

fn run_transpile_case(name: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_dir = Path::new("tests/cases");
    let br_file       = case_dir.join(format!("{}.br",       name));
    let expected_file = case_dir.join(format!("{}.expected", name));
    let rust_dir      = case_dir.join(format!("{}_rust",     name));

    // ── Step 1: emit Rust ─────────────────────────────────────────────────────
    let emit = Command::new(bin)
        .arg("--emit-rust")
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{}] failed to invoke boring: {}", name, e));

    assert!(
        emit.status.success(),
        "[{}] boring --emit-rust failed:\n{}",
        name,
        String::from_utf8_lossy(&emit.stderr)
    );

    // ── Step 2: cargo run ─────────────────────────────────────────────────────
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(rust_dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("[{}] failed to invoke cargo: {}", name, e));

    assert!(
        run.status.success(),
        "[{}] cargo run failed:\n--- stderr ---\n{}",
        name,
        String::from_utf8_lossy(&run.stderr)
    );

    // ── Step 3: compare output ────────────────────────────────────────────────
    // Normalise line endings so tests pass on Windows (CRLF → LF)
    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    let expected = std::fs::read_to_string(&expected_file)
        .unwrap_or_else(|_| panic!("[{}] missing expected file: {}", name, expected_file.display()))
        .replace("\r\n", "\n");

    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "[{}] interpreter / transpiler output mismatch\n--- expected (interpreter) ---\n{}\n--- actual (transpiler) ---\n{}",
        name,
        expected.trim_end(),
        actual.trim_end()
    );
}

macro_rules! transpile_test {
    ($name:ident) => {
        #[test]
        fn $name() {
            run_transpile_case(stringify!($name));
        }
    };
}

// ── Tests ────────────────────────────────────────────────────────────────────
// One entry per interpreter test that has a .expected file.
// Error/rejection tests are excluded (they test compile-time checks, not output).

transpile_test!(basics);
transpile_test!(strings);
transpile_test!(control_flow);
transpile_test!(match_stmt);
transpile_test!(functions);
transpile_test!(closures);
transpile_test!(structs);
transpile_test!(collections);
transpile_test!(error_handling);
transpile_test!(protocols);
transpile_test!(optionals);
transpile_test!(enums);
transpile_test!(newtypes);
transpile_test!(guard);
transpile_test!(generics);
transpile_test!(operators);
transpile_test!(macros);
transpile_test!(defer);
transpile_test!(do_block);
transpile_test!(tuples);
transpile_test!(format);
transpile_test!(loops);
transpile_test!(traits);
transpile_test!(numeric);
transpile_test!(modules);
transpile_test!(ownership);
transpile_test!(tasks);
transpile_test!(channels);
transpile_test!(streams);
transpile_test!(let_pattern);
transpile_test!(result_compat);
transpile_test!(multi_catch);
transpile_test!(implicit_self);
transpile_test!(shadowing);
transpile_test!(struct_spread);
transpile_test!(tuple_string);
transpile_test!(array_pop_remove);
transpile_test!(transpiler_coerce);
transpile_test!(string_len_chars);
transpile_test!(mixed_modulo);
transpile_test!(range_unary);
transpile_test!(closure_colon);
transpile_test!(collections2);
// Note: nil_assign (type inference for nil variables), pattern_some (Some/None on non-Option),
// and closure_break (break inside closure) are interpreter-only tests — not added here.
