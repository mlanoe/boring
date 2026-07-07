// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Functional test suite for the Boring-in-Boring interpreter.
//
// Each test:
//   1. Pipes a `tests/cases/<name>.br` file to the compiled interpreter binary
//      (boring/interpreter/main_rust/target/debug/main).
//   2. Compares stdout against `tests/cases/<name>.expected`.
//
// Prerequisite: run `cargo test --test interpreter_build` at least once to
// compile the interpreter binary before running these tests.
//
// Run with:
//   cargo test --test interpreter_functional

use std::io::Write as IoWrite;
use std::path::Path;
use std::process::{Command, Stdio};

const INTERP_BIN: &str = "boring/interpreter/main_rust/target/debug/main";

fn run_case(name: &str) {
    let bin = Path::new(INTERP_BIN);
    assert!(
        bin.exists(),
        "interpreter binary not found at {}  — run `cargo test --test interpreter_build` first",
        bin.display()
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
        .unwrap_or_else(|e| panic!("[{}] failed to spawn interpreter: {}", name, e));

    child.stdin.take().unwrap().write_all(&source).unwrap();

    let out = child.wait_with_output()
        .unwrap_or_else(|e| panic!("[{}] wait failed: {}", name, e));

    assert!(
        out.status.success(),
        "[{}] interpreter exited with error:\n{}",
        name,
        String::from_utf8_lossy(&out.stderr)
    );

    let actual = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
    let expected = std::fs::read_to_string(&expected_file)
        .unwrap_or_else(|_| panic!("missing expected file: {}", expected_file.display()))
        .replace("\r\n", "\n");

    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "[{}] output mismatch\n--- expected ---\n{}\n--- actual ---\n{}",
        name,
        expected.trim_end(),
        actual.trim_end()
    );
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
