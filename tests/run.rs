use std::path::Path;
use std::process::Command;

fn run_case(name: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_dir = Path::new("tests/cases");
    let br_file = case_dir.join(format!("{}.br", name));
    let expected_file = case_dir.join(format!("{}.expected", name));

    let output = Command::new(bin)
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("failed to run boring: {}", e));

    assert!(
        output.status.success(),
        "test '{}' exited with error:\n{}",
        name,
        String::from_utf8_lossy(&output.stderr)
    );

    // Normalise line endings so tests pass on Windows (CRLF → LF)
    let actual = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
    let expected = std::fs::read_to_string(&expected_file)
        .unwrap_or_else(|_| panic!("missing expected file: {}", expected_file.display()))
        .replace("\r\n", "\n");

    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "test '{}' output mismatch\n--- expected ---\n{}\n--- actual ---\n{}",
        name,
        expected.trim_end(),
        actual.trim_end()
    );
}

fn run_error_case(name: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_dir = Path::new("tests/cases");
    let br_file = case_dir.join(format!("{}.br", name));
    let error_file = case_dir.join(format!("{}.error", name));

    let output = Command::new(bin)
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("failed to run boring: {}", e));

    assert!(
        !output.status.success(),
        "test '{}' expected to fail but exited successfully",
        name
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    let expected = std::fs::read_to_string(&error_file)
        .unwrap_or_else(|_| panic!("missing error file: {}", error_file.display()));

    let expected = expected.trim();
    assert!(
        stderr.contains(expected),
        "test '{}': stderr mismatch\n--- expected to contain ---\n{}\n--- actual stderr ---\n{}",
        name,
        expected,
        stderr
    );
}

macro_rules! interp_test {
    ($name:ident) => {
        #[test]
        fn $name() {
            run_case(stringify!($name));
        }
    };
}

macro_rules! error_test {
    ($name:ident) => {
        #[test]
        fn $name() {
            run_error_case(stringify!($name));
        }
    };
}

// ── Interpreter tests ────────────────────────────────────────────────────────

interp_test!(basics);
interp_test!(strings);
interp_test!(control_flow);
interp_test!(match_stmt);
interp_test!(functions);
interp_test!(closures);
interp_test!(structs);
interp_test!(collections);
interp_test!(error_handling);
interp_test!(tasks);
interp_test!(channels);
interp_test!(protocols);
interp_test!(optionals);
interp_test!(enums);
interp_test!(streams);
interp_test!(newtypes);
interp_test!(guard);
interp_test!(generics);
interp_test!(operators);
interp_test!(macros);
interp_test!(defer);
interp_test!(do_block);
interp_test!(tuples);
interp_test!(format);
interp_test!(loops);
interp_test!(traits);
interp_test!(numeric);
interp_test!(modules);
interp_test!(ownership);
interp_test!(let_pattern);
interp_test!(result_compat);
interp_test!(multi_catch);
interp_test!(implicit_self);
interp_test!(nil_assign);
interp_test!(shadowing);
interp_test!(struct_spread);
interp_test!(tuple_string);
interp_test!(array_pop_remove);
interp_test!(closure_break);
interp_test!(transpiler_coerce);
interp_test!(pattern_some);
interp_test!(string_len_chars);
interp_test!(mixed_modulo);
interp_test!(range_unary);
interp_test!(closure_colon);

// ── Error / rejection tests ──────────────────────────────────────────────────

error_test!(error_undefined_var);
error_test!(error_uncaught_throw);
error_test!(error_move_source);
error_test!(error_immutable_param);
error_test!(error_conformance_missing);
error_test!(error_must_use);
