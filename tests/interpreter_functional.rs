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
itest!(modules);
itest!(ownership);
itest!(let_pattern);
itest!(result_compat);
itest!(multi_catch);
itest!(implicit_self);
itest!(shadowing);
itest!(struct_spread);
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
