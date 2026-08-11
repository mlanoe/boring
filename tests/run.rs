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

/// Exact diagnostic test: normalises the file path in `-->` lines to `<file>`
/// then compares the full stderr against the `.error` snapshot.
fn run_error_case_exact(name: &str) {
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

    let path_str = br_file.to_string_lossy();
    let raw = String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n");
    let actual = raw.replace(path_str.as_ref(), "<file>").trim_end().to_string();

    let expected = std::fs::read_to_string(&error_file)
        .unwrap_or_else(|_| panic!("missing error file: {}", error_file.display()))
        .replace("\r\n", "\n")
        .trim_end()
        .to_string();

    assert_eq!(
        actual,
        expected,
        "test '{}': diagnostic mismatch\n--- expected ---\n{}\n--- actual ---\n{}",
        name,
        expected,
        actual
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

macro_rules! error_test_exact {
    ($name:ident) => {
        #[test]
        fn $name() {
            run_error_case_exact(stringify!($name));
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
interp_test!(with_stmt);
interp_test!(generics);
interp_test!(operators);
interp_test!(macros);
interp_test!(defer);
interp_test!(do_block);
interp_test!(tuples);
interp_test!(tuple_methods);
interp_test!(tuple_map);
interp_test!(inline_loops);
interp_test!(format);
interp_test!(loops);
interp_test!(traits);
interp_test!(numeric);
interp_test!(float_width_cross_eq);
interp_test!(scalar_catch);
interp_test!(modules);
interp_test!(ownership);
interp_test!(let_pattern);
interp_test!(result_compat);
interp_test!(multi_catch);
interp_test!(implicit_self);
interp_test!(nil_assign);
interp_test!(shadowing);
interp_test!(struct_spread);
interp_test!(default_rest);
interp_test!(tuple_string);
interp_test!(array_pop_remove);
interp_test!(closure_break);
interp_test!(transpiler_coerce);
interp_test!(pattern_some);
interp_test!(string_len_chars);
interp_test!(mixed_modulo);
interp_test!(range_unary);
interp_test!(closure_colon);
interp_test!(for_destructure);
interp_test!(numeric_separators);
interp_test!(task_timeout);
interp_test!(try_else_block);
interp_test!(error_match);
interp_test!(fn_shorthand);
interp_test!(camel_to_snake);
interp_test!(qualifiers_actor);
interp_test!(lazy);
interp_test!(array_comprehension);
interp_test!(array_comp_iter);
interp_test!(callable_struct);
interp_test!(fixed_array);
interp_test!(auto_ref_infer);
interp_test!(collections2);
interp_test!(join_handle);
interp_test!(select);
interp_test!(triple_string);
interp_test!(pipe);
interp_test!(inline_match);
interp_test!(supertraits);
interp_test!(type_cast);
interp_test!(ref_identity);
interp_test!(mut_scalar);
interp_test!(int_float_literal_compare);
interp_test!(float32_math_builtins);
interp_test!(top_level_const);
interp_test!(pub_top_level_const);
interp_test!(pub_top_level_const_unused);
// A struct field literally named `count` -- must never be hijacked by the
// `.length`/`.count` collection-length builtin. The interpreter's `get_field`
// was never affected (its Array/Set/Dict shortcuts are gated by value type,
// never reached for a struct/Object); this is here to lock that in.
interp_test!(struct_count_field);

// ── Error / rejection tests ──────────────────────────────────────────────────

error_test_exact!(error_undefined_var);
error_test_exact!(error_uncaught_throw);
error_test_exact!(error_move_source);
error_test_exact!(error_immutable_param);
error_test_exact!(error_immutable_let);
error_test_exact!(error_immutable_loop_var);
error_test_exact!(error_mut_shared);
error_test_exact!(error_lazy_assign);
error_test_exact!(error_lazy_regular_assign);
error_test_exact!(error_conformance_missing);
error_test_exact!(error_must_use);
error_test_exact!(error_array_comp_nonzero);
error_test!(error_array_comp_iter_non_array);
error_test!(error_default_rest_spread_conflict);
error_test!(error_default_rest_positional_conflict);
