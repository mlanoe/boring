// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Functional test suite for the Boring-in-Boring interpreter.
//
// Each test runs the `.br` case against all 4 transpiled binaries:
//   strict+multi, strict+single, managed+multi, managed+single.
//
// Prerequisite: run `cargo test --test interpreter_build` at least once to
// compile all four interpreter binaries before running these tests.
//
// Run with:
//   cargo test --test interpreter_functional

use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MODES: &[(&str, &str, &str)] = &[
    ("strict",  "multi",  "main_rust"),
    ("strict",  "single", "main_rust_single"),
    ("managed", "multi",  "main_rust_managed"),
    ("managed", "single", "main_rust_managed_single"),
];

fn find_bin_in(rust_dir: &str) -> PathBuf {
    let base = Path::new("boring/interpreter").join(rust_dir).join("target");
    let name = format!("main{}", std::env::consts::EXE_SUFFIX);
    // Scan one level deep for a target-triple subdirectory (e.g. x86_64-pc-windows-msvc).
    if let Ok(entries) = std::fs::read_dir(&base) {
        for entry in entries.flatten() {
            let candidate = entry.path().join("debug").join(&name);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    base.join("debug").join(&name)
}

fn run_case_with_bin(name: &str, bin: &Path, label: &str) {
    assert!(
        bin.exists(),
        "[{}@{}] binary not found at {} — run `cargo test --test interpreter_build` first",
        name, label, bin.display()
    );

    let case_dir = Path::new("tests/cases");
    let br_file = case_dir.join(format!("{}.br", name));
    let expected_file = case_dir.join(format!("{}.expected", name));

    let source = std::fs::read(&br_file)
        .unwrap_or_else(|_| panic!("cannot read {}", br_file.display()));

    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("[{}@{}] failed to spawn: {}", name, label, e));

    child.stdin.take().unwrap().write_all(&source).unwrap();

    let out = child.wait_with_output()
        .unwrap_or_else(|e| panic!("[{}@{}] wait failed: {}", name, label, e));

    assert!(
        out.status.success(),
        "[{}@{}] interpreter exited with error:\n{}",
        name, label,
        String::from_utf8_lossy(&out.stderr)
    );

    let actual = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
    let expected = std::fs::read_to_string(&expected_file)
        .unwrap_or_else(|_| panic!("missing expected file: {}", expected_file.display()))
        .replace("\r\n", "\n");

    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "[{}@{}] output mismatch\n--- expected ---\n{}\n--- actual ---\n{}",
        name, label,
        expected.trim_end(),
        actual.trim_end(),
    );
}

fn run_case(name: &str) {
    for (mode, threading, rust_dir) in MODES {
        let label = format!("{}+{}", mode, threading);
        let bin = find_bin_in(rust_dir);
        run_case_with_bin(name, &bin, &label);
    }
}


macro_rules! itest {
    ($name:ident) => {
        #[test]
        fn $name() {
            run_case(stringify!($name));
        }
    };
}

itest!(basics);
itest!(strings);
// `.slice()` negative-index clamping — deliberately NOT added to the
// `strings`/`collections` fixtures above: those are shared verbatim with
// `tests/transpile.rs` (the main compiler's own transpile+build+run suite),
// whose native `.slice()` builtin doesn't clamp negative indices at all (a
// separate, out-of-scope gap) and would fail to even compile on a negative
// literal. Dedicated cases keep this self-hosted-interpreter-only fix
// (boring/interpreter/methods.br) isolated from that other suite.
itest!(string_slice_clamp);
itest!(array_slice_clamp);
itest!(control_flow);
itest!(match_stmt);
itest!(functions);
itest!(closures);
itest!(structs);
itest!(collections);
itest!(error_handling);
itest!(tasks);
itest!(channels);
itest!(protocols);
itest!(optionals);
itest!(enums);
itest!(streams);
itest!(newtypes);
itest!(guard);
itest!(with_stmt);
itest!(generics);
itest!(operators);
itest!(method_overloading);
itest!(free_fn_overloading);
itest!(macros);
itest!(defer);
itest!(do_block);
itest!(tuples);
itest!(tuple_methods);
itest!(tuple_map);
itest!(inline_loops);
itest!(format);
itest!(loops);
itest!(traits);
itest!(numeric);
itest!(float_width_cross_eq);
itest!(scalar_catch);
itest!(modules);
itest!(ownership);
itest!(let_pattern);
itest!(result_compat);
itest!(multi_catch);
itest!(implicit_self);
itest!(shadowing);
itest!(struct_spread);
itest!(default_rest);
itest!(tuple_string);
itest!(array_pop_remove);
itest!(closure_break);
itest!(pattern_some);
itest!(string_len_chars);
itest!(mixed_modulo);
itest!(range_unary);
itest!(closure_colon);
itest!(for_destructure);
itest!(numeric_separators);
itest!(fn_shorthand);
itest!(camel_to_snake);
itest!(lazy);
itest!(array_comprehension);
itest!(callable_struct);
itest!(fixed_array);
itest!(labeled_array);
itest!(collections2);
itest!(triple_string);
itest!(pipe);
itest!(supertraits);
itest!(type_cast);
itest!(inline_match);
itest!(ref_identity);
itest!(qualifiers_actor);
itest!(join_handle);
itest!(select);
itest!(task_timeout);
itest!(error_match);
itest!(try_else_block);
itest!(nil_assign);
itest!(transpiler_coerce);

// ─── Real, on-disk `use` module resolution ─────────────────────────────────
//
// Every case above is piped in over stdin as a single in-memory "file" with
// no real path (see run_case_with_bin) — there's no entry-file directory for
// `use`'s relative sibling-file resolution (exec_use in
// boring/interpreter/stdlib.br) to resolve against. These cases instead
// invoke the compiled interpreter binary with a real file argument, from a
// fixture directory under tests/cases/ that has actual sibling `.br` files.

fn run_file_case_with_bin(dir: &str, entry: &str, bin: &Path, label: &str) -> std::process::Output {
    assert!(
        bin.exists(),
        "[{}@{}] binary not found at {} — run `cargo test --test interpreter_build` first",
        dir, label, bin.display()
    );
    let entry_path = Path::new("tests/cases").join(dir).join(entry);
    Command::new(bin)
        .arg(&entry_path)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap_or_else(|e| panic!("[{}@{}] failed to spawn: {}", dir, label, e))
}

/// Runs `dir/entry` against all 4 interpreter binaries and checks stdout
/// against `dir/expected_name`. Mirrors `run_case_with_bin`, but via a real
/// file argument instead of stdin.
fn run_file_case_ok(dir: &str, entry: &str, expected_name: &str) {
    for (mode, threading, rust_dir) in MODES {
        let label = format!("{}+{}", mode, threading);
        let bin = find_bin_in(rust_dir);
        let out = run_file_case_with_bin(dir, entry, &bin, &label);

        assert!(
            out.status.success(),
            "[{}@{}] interpreter exited with error:\n{}",
            dir, label,
            String::from_utf8_lossy(&out.stderr)
        );

        let actual = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
        let expected_file = Path::new("tests/cases").join(dir).join(expected_name);
        let expected = std::fs::read_to_string(&expected_file)
            .unwrap_or_else(|_| panic!("missing expected file: {}", expected_file.display()))
            .replace("\r\n", "\n");

        assert_eq!(
            actual.trim_end(),
            expected.trim_end(),
            "[{}@{}] output mismatch\n--- expected ---\n{}\n--- actual ---\n{}",
            dir, label,
            expected.trim_end(),
            actual.trim_end(),
        );
    }
}

/// Runs `dir/entry` against all 4 interpreter binaries and asserts it fails
/// fast (nonzero exit) with `expected_stderr_substr` somewhere in stderr —
/// for `use` forms that are deliberately unsupported (see
/// `use_boring_stdlib_unsupported` below).
fn run_file_case_err(dir: &str, entry: &str, expected_stderr_substr: &str) {
    for (mode, threading, rust_dir) in MODES {
        let label = format!("{}+{}", mode, threading);
        let bin = find_bin_in(rust_dir);
        let out = run_file_case_with_bin(dir, entry, &bin, &label);

        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !out.status.success(),
            "[{}@{}] expected a failure, but the interpreter exited successfully (stdout:\n{})",
            dir, label,
            String::from_utf8_lossy(&out.stdout)
        );
        assert!(
            stderr.contains(expected_stderr_substr),
            "[{}@{}] stderr did not contain {:?}:\n{}",
            dir, label, expected_stderr_substr, stderr
        );
    }
}

#[test]
fn use_modules() {
    run_file_case_ok("use_modules", "main.br", "main.expected");
}

#[test]
fn use_boring_stdlib_unsupported() {
    run_file_case_err("use_boring_stdlib_unsupported", "main.br", "boring.*");
}

// Regression test for `parse_hex_str`/`parse_bin_str`/`parse_oct_str`
// (boring/interpreter/lexer.br) accumulating a hex/octal/binary literal
// straight into an `int` (isize, 64-bit) with no overflow check at all — a
// 17+ hex digit literal (e.g. `0xFFFFFFFFFFFFFFFF`, u64::MAX) silently
// wrapped in release or panicked on the raw `acc * 16 + digit` overflow in
// debug. Must now fail lexing fast with a clear error naming the literal,
// same as the main compiler's own lexer already does for this class of
// literal (src/lexer/mod.rs's `LexError::IntegerOverflow`).
#[test]
fn numeric_literal_overflow() {
    run_file_case_err("numeric_literal_overflow", "main.br", "too large");
}

// Regression test for the self-hosted parser's recursion-depth guard
// (`boring/interpreter/parser_core.br`'s `parser_depth`/`parser_depth_inc`/
// `parser_depth_dec`, and `parser_exprstmt.br`'s `parse_block`/`parse_or`/
// `parse_unary`/`parser_core.br`'s `parse_pattern`). Before this fix, the
// depth counter was a dead stub (`parser_depth` always returned `0`,
// `_inc`/`_dec` were no-ops) — the sole guard site that used it
// (`parse_not`, for `not not not …` chains) never actually tripped, and
// every other recursive path (nested blocks, parens, patterns) had no guard
// at all. A deeply nested program used to crash this compiled interpreter
// with a native stack overflow (SIGABRT) instead of a clean parse error.
//
// Generates the source programmatically (500 levels of nested `if true:`
// blocks) rather than committing a huge static fixture file — same approach
// as the Rust-hosted parser's own analogous regression tests
// (`src/parser/tests_recursion_depth.rs`).
#[test]
fn deeply_nested_blocks_produce_a_clean_error_not_a_crash() {
    let mut src = String::new();
    const N: usize = 500;
    for i in 0..N {
        src.push_str(&"    ".repeat(i));
        src.push_str("if true:\n");
    }
    src.push_str(&"    ".repeat(N));
    src.push_str("print 1\n");

    let tmp = Path::new("tests/cases").join("deep_nested_blocks_generated.br");
    std::fs::write(&tmp, &src).expect("failed to write generated fixture");

    for (mode, threading, rust_dir) in MODES {
        let label = format!("{}+{}", mode, threading);
        let bin = find_bin_in(rust_dir);
        assert!(
            bin.exists(),
            "[deep_nested_blocks@{}] binary not found at {} — run `cargo test --test interpreter_build` first",
            label, bin.display()
        );

        let out = Command::new(&bin)
            .arg(&tmp)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap_or_else(|e| panic!("[deep_nested_blocks@{}] failed to spawn: {}", label, e));

        // The exact failure mode we're guarding against is a native crash
        // (SIGABRT from a stack overflow) rather than a normal nonzero exit —
        // `ExitStatus::code()` is `None` on Unix when the process was killed
        // by a signal, which is exactly the crash this test must rule out.
        assert!(
            out.status.code().is_some(),
            "[deep_nested_blocks@{}] interpreter crashed (terminated by signal, status: {}) \
             instead of returning a clean parse error — stderr:\n{}",
            label, out.status, String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.status.success(),
            "[deep_nested_blocks@{}] expected a clean parse-error failure, but the interpreter \
             exited successfully (stdout:\n{})",
            label, String::from_utf8_lossy(&out.stdout)
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("nested too deeply"),
            "[deep_nested_blocks@{}] expected a 'nested too deeply' parse error, got:\n{}",
            label, stderr
        );
    }

    let _ = std::fs::remove_file(&tmp);
}
