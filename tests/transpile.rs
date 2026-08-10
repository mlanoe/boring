// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Transpiler consistency tests.
//
// Each test:
//   1. Runs `boring build <case>.br --mode <m> --threading <t> --output-dir <dir>`
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


fn run_transpile_case_with_config(name: &str, mode_str: &str, threading_str: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_dir = Path::new("tests/cases");
    let br_file       = case_dir.join(format!("{}.br", name));
    let expected_file = case_dir.join(format!("{}.expected", name));
    let rust_dir_name = format!("{}_{}_{}_rust", name, mode_str, threading_str);
    let rust_dir      = case_dir.join(&rust_dir_name);

    // ── Step 1: emit Rust ─────────────────────────────────────────────────────
    let emit = Command::new(bin)
        .arg("build").arg(&br_file)
        .arg("--mode").arg(mode_str)
        .arg("--threading").arg(threading_str)
        .arg("--output-dir").arg(&rust_dir)
        .output()
        .unwrap_or_else(|e| panic!("[{}@{}+{}] failed to invoke boring: {}", name, mode_str, threading_str, e));

    assert!(
        emit.status.success(),
        "[{}@{}+{}] boring build failed:\n{}",
        name, mode_str, threading_str,
        String::from_utf8_lossy(&emit.stderr)
    );

    // ── Step 2: cargo run ─────────────────────────────────────────────────────
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(rust_dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("[{}@{}+{}] failed to invoke cargo: {}", name, mode_str, threading_str, e));

    assert!(
        run.status.success(),
        "[{}@{}+{}] cargo run failed:\n--- stderr ---\n{}",
        name, mode_str, threading_str,
        String::from_utf8_lossy(&run.stderr)
    );

    // ── Step 3: compare output ────────────────────────────────────────────────
    let actual   = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    let expected = std::fs::read_to_string(&expected_file)
        .unwrap_or_else(|_| panic!("[{}] missing expected file: {}", name, expected_file.display()))
        .replace("\r\n", "\n");

    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "[{}@{}+{}] output mismatch\n--- expected ---\n{}\n--- actual ---\n{}",
        name, mode_str, threading_str,
        expected.trim_end(),
        actual.trim_end()
    );
}

// Keep the old single-config runner for backwards compatibility during migration.
fn run_transpile_case(name: &str) {
    run_transpile_case_with_config(name, "strict", "multi");
}

macro_rules! transpile_test {
    ($name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn strict_multi()   { run_transpile_case_with_config(stringify!($name), "strict",  "multi"); }
            #[test]
            fn strict_single()  { run_transpile_case_with_config(stringify!($name), "strict",  "single"); }
            #[test]
            fn managed_multi()  { run_transpile_case_with_config(stringify!($name), "managed", "multi"); }
            #[test]
            fn managed_single() { run_transpile_case_with_config(stringify!($name), "managed", "single"); }
        }
    };
    // Variant for tests that should be ignored in single-thread mode (e.g. LocalSet issues).
    ($name:ident, ignore_single) => {
        mod $name {
            use super::*;
            #[test]
            fn strict_multi()   { run_transpile_case_with_config(stringify!($name), "strict",  "multi"); }
            #[test]
            #[ignore = "single-thread LocalSet not yet supported"]
            fn strict_single()  { run_transpile_case_with_config(stringify!($name), "strict",  "single"); }
            #[test]
            fn managed_multi()  { run_transpile_case_with_config(stringify!($name), "managed", "multi"); }
            #[test]
            #[ignore = "single-thread LocalSet not yet supported"]
            fn managed_single() { run_transpile_case_with_config(stringify!($name), "managed", "single"); }
        }
    };
    // Variant for tests that use T' in complex patterns that managed mode cannot handle yet
    // (constructor call sites are not updated to emit Arc::new(Mutex::new(...)) instead of Box::new).
    ($name:ident, ignore_managed) => {
        mod $name {
            use super::*;
            #[test]
            fn strict_multi()   { run_transpile_case_with_config(stringify!($name), "strict",  "multi"); }
            #[test]
            fn strict_single()  { run_transpile_case_with_config(stringify!($name), "strict",  "single"); }
            #[test]
            #[ignore = "managed mode T' call sites not yet fully supported"]
            fn managed_multi()  { run_transpile_case_with_config(stringify!($name), "managed", "multi"); }
            #[test]
            #[ignore = "managed mode T' call sites not yet fully supported"]
            fn managed_single() { run_transpile_case_with_config(stringify!($name), "managed", "single"); }
        }
    };
    // Variant for tests that have both single-thread and managed mode issues.
    ($name:ident, ignore_single_managed) => {
        mod $name {
            use super::*;
            #[test]
            fn strict_multi()   { run_transpile_case_with_config(stringify!($name), "strict",  "multi"); }
            #[test]
            #[ignore = "single-thread mode has known issues with Rc/Arc mixing in weak refs"]
            fn strict_single()  { run_transpile_case_with_config(stringify!($name), "strict",  "single"); }
            #[test]
            #[ignore = "managed mode T' call sites not yet fully supported"]
            fn managed_multi()  { run_transpile_case_with_config(stringify!($name), "managed", "multi"); }
            #[test]
            #[ignore = "managed mode T' call sites not yet fully supported"]
            fn managed_single() { run_transpile_case_with_config(stringify!($name), "managed", "single"); }
        }
    };
    // Variant for tests that fail in both single-thread mode variants (strict_single and managed_single).
    ($name:ident, ignore_single_all) => {
        mod $name {
            use super::*;
            #[test]
            fn strict_multi()   { run_transpile_case_with_config(stringify!($name), "strict",  "multi"); }
            #[test]
            #[ignore = "single-thread mode not yet supported for this test"]
            fn strict_single()  { run_transpile_case_with_config(stringify!($name), "strict",  "single"); }
            #[test]
            fn managed_multi()  { run_transpile_case_with_config(stringify!($name), "managed", "multi"); }
            #[test]
            #[ignore = "single-thread mode not yet supported for this test"]
            fn managed_single() { run_transpile_case_with_config(stringify!($name), "managed", "single"); }
        }
    };
}

// Silence dead_code warning for the old helper (used via the macro).
#[allow(dead_code)]
fn _keep_run_transpile_case(name: &str) { run_transpile_case(name); }

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
transpile_test!(with_stmt);
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
transpile_test!(uint_int_cross_eq);
transpile_test!(float_width_cross_eq);
transpile_test!(scalar_catch);
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
transpile_test!(default_rest);
transpile_test!(tuple_string);
transpile_test!(tuple_methods);
transpile_test!(tuple_map);
transpile_test!(array_pop_remove);
transpile_test!(transpiler_coerce);
transpile_test!(string_len_chars);
transpile_test!(mixed_modulo);
transpile_test!(range_unary);
transpile_test!(closure_colon);
transpile_test!(collections2);
transpile_test!(join_handle);
transpile_test!(select);
transpile_test!(auto_ref_infer);
transpile_test!(qualifiers_actor);
transpile_test!(pipe);
transpile_test!(inline_match);
transpile_test!(supertraits);
transpile_test!(type_cast);
transpile_test!(ref_identity);
transpile_test!(mut_scalar);
transpile_test!(int_float_literal_compare);
transpile_test!(float32_math_builtins);
// Top-level `let` constants referenced from a free function, a struct method, AND an
// enum method — regression test for the transpiler silently dropping the `const`
// declaration for a module-scope `let` whenever nothing but a function/method body
// referenced it (the reference compiled fine under `boring run`, since the tree-walk
// interpreter tracks globals directly, but `boring build`'s emitted Rust failed with
// E0425 "cannot find value" -- this case's real value is the `cargo run` compile this
// harness performs, not just the interpreter comparison `tests/run.rs` also runs here).
transpile_test!(top_level_const);
// `.pointee` — explicit dereference for opaque/external Rust types (Deref/DerefMut),
// e.g. Bevy's `Single<T>`/`Mut<T>`. Real `Box<T>` stands in for the foreign type here
// so this stays a self-contained transpile+cargo-build case (no extra Cargo
// dependency) — transpiler-only by nature (the interpreter has no runtime concept of
// an external Rust value to deref), so this is NOT in tests/run.rs.
transpile_test!(pointee);
// `pub let` at module scope must emit `pub const` (private `let` stays a private
// `const`) -- this single-crate run only proves the generated code still compiles
// and runs correctly for both; it can't observe cross-module visibility on its own,
// since the generated project is a single binary crate with no sibling module
// importing it. `pub_module_const.rs` is the test that actually exercises that
// (a hand-written sibling file in a real two-file crate, built with `cargo build`).
transpile_test!(pub_top_level_const);
// Note: nil_assign (type inference for nil variables), pattern_some (Some/None on non-Option),
// and closure_break (break inside closure) are interpreter-only tests — not added here.
