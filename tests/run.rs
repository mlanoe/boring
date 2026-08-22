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
interp_test!(guard_let_mut_field_escape);
interp_test!(strings);
interp_test!(control_flow);
interp_test!(match_stmt);
interp_test!(functions);
interp_test!(closures);
interp_test!(structs);
interp_test!(collections);
interp_test!(error_handling);
interp_test!(throws_untyped_enum_catch);
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
// `use boring.collections` — the first-party stdlib mechanism (docs/cross-
// project-code-sharing-gap.md's stdlib work), backed by src/stdlib_embed.rs.
// See tests/cases/boring_stdlib_collections.br's own doc comment. Paired
// with tests/transpile.rs's registration below for the `boring build` side.
interp_test!(boring_stdlib_collections);

// Named cross-project dependency (docs/cross-project-code-sharing-gap.md's [deps]
// work): `use numlib.big_uint.*` resolves against tests/cases/fixtures/dep_numlib
// via tests/cases/cross_project_dep/boring.toml's own `[deps]` section. Doesn't fit
// `interp_test!`'s flat `tests/cases/<name>.br` convention (this is a small project
// with its own boring.toml, like `transpile_project_test!` cases) -- `run_file`
// discovers that boring.toml itself via `find_project_root`, independent of the test
// runner's cwd, so no `current_dir` juggling is needed here unlike the
// `transpile_project_test!` counterpart below (project-mode `boring build` reads
// `./boring.toml` from cwd). Paired with tests/transpile.rs's registration for the
// `boring build` + real `cargo run` side.
#[test]
fn cross_project_dep() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let br_file = Path::new("tests/cases/cross_project_dep/src/main.br");

    let output = Command::new(bin)
        .arg(br_file)
        .output()
        .unwrap_or_else(|e| panic!("failed to run boring: {}", e));

    assert!(
        output.status.success(),
        "cross_project_dep exited with error:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let actual = String::from_utf8_lossy(&output.stdout).replace("\r\n", "\n");
    assert_eq!(actual.trim_end(), "21\n42");
}

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
interp_test!(optional_pop_tail_call);
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
interp_test!(int_literal_overflow_cast);
interp_test!(builtin_error_enum);
interp_test!(multi_variant_catch_dispatch);
interp_test!(type_def_typed_throws);
interp_test!(type_method_throws_untyped);
interp_test!(ref_identity);
interp_test!(mut_scalar);
interp_test!(int_float_literal_compare);
interp_test!(float32_math_builtins);
interp_test!(float32_struct_method_math);
// Same gap as float32_struct_method_math above, but for a plain `let`-bound local
// variable computed from an unannotated arithmetic expression, in an ordinary
// (non-method) function — see tests/cases/float32_local_var_math.br's doc comment.
interp_test!(float32_local_var_math);
interp_test!(top_level_const);
interp_test!(pub_top_level_const);
interp_test!(pub_top_level_const_unused);
// `pub let`/used-elsewhere `let` STRING constants -- unlike the scalar cases just above,
// promotion to a Rust module-level item requires a dedicated `top_level_let_is_string_literal`
// check (a plain scalar-literal check never matches a string). This run only proves the
// interpreter (entirely unaffected by transpiler promotion) evaluates it correctly -- unlike
// `pub_top_level_const_unused` above, every constant here HAS an in-file consumer
// (`is_add`/`is_sub`/`reveal_private` each read one across a function boundary), so
// `transpile.rs`'s single-crate `transpile_test!` run is what actually proves promotion: if
// the string stayed a `fn main()` local, those calls would fail to compile (E0425) rather
// than just produce a wrong runtime value.
interp_test!(pub_top_level_string_const);
// A struct field literally named `count` -- must never be hijacked by the
// `.length`/`.count` collection-length builtin. The interpreter's `get_field`
// was never affected (its Array/Set/Dict shortcuts are gated by value type,
// never reached for a struct/Object); this is here to lock that in.
interp_test!(struct_count_field);
// Dict `[key]` indexing (read + write) with a string key, as a function
// parameter and as an implicit-self struct field. The interpreter (this
// test) never caught the underlying bugs -- both were only visible in
// `--emit-rust` output (see tests/transpile.rs's registration for the real
// regression coverage) -- kept here anyway so the runtime *values* stay
// pinned down too. See tests/cases/dict_string_key_index.br's own doc comment.
interp_test!(dict_string_key_index);
// A `var StructType` parameter must mutate the caller's variable directly
// (docs/CLAUDE.md: "changes are visible at the call site"), not a throwaway
// clone of it. The interpreter (this test) never caught this either -- it
// was a `--emit-rust`-only codegen bug (see tests/transpile.rs). See
// tests/cases/var_struct_param_mutation.br's own doc comment.
interp_test!(var_struct_param_mutation);

// `type def`/`type req` must
// parse inside an `enum` body, not just a `struct` body. Now registered in
// both tests/run.rs and tests/transpile.rs (see tests/cases/enum_type_def.br's
// own doc comment) -- the transpiler codegen gap for this is fixed too.
interp_test!(enum_type_def);
// An enum's type-level method's
// `throws Type:` clause must be wrapped in Result<T, E> by the transpiler,
// the same as a struct's already is (item 4). `boring run` side of this
// pairs with tests/transpile.rs's registration below.
interp_test!(enum_type_def_throws);

// Most dangerous of the four: a
// silent wrong-value bug, not a compile error: a two-branch
// `if let x = expr: A else: B` used as a nested (non-tail) expression used
// to always evaluate to `B`, and the same shape as a function's direct tail
// statement used to fail with an unrelated "return value discarded" error.
interp_test!(if_let_expr_nested);

// Unary `-` on a genuinely
// int64/int128-tagged value (as opposed to the generic untyped `Value::Int`)
// used to fail with "cannot negate Int64"/"cannot negate Int128".
interp_test!(tagged_int_negate);

// The interpreter's move-checker
// used to treat int64/int128-tagged scalars as non-Copy, wrongly raising
// "use of moved value" on a plain `let t = n` reuse, unlike the generic
// untyped `int`.
interp_test!(tagged_int_copy);
// Vec::first()/last() and HashMap::get(k) return Option<&T> in Rust; Boring's
// first()/last()/get() are documented as owned `T?`. Regression test for the
// transpiler double-wrapping an already-Option-shaped result in Some(...), and
// for `.cloned()` insertion producing owned values. Interpreter-side (this file)
// is the semantic baseline; `transpile_test!` below exercises the actual bug.
interp_test!(option_owned_methods);

// ── Error / rejection tests ──────────────────────────────────────────────────

// `use boring.<module>` for an unrecognized module name is a hard error,
// unlike the generic filesystem `use` loader's silent no-op (which assumes
// a native Rust module — not applicable to `boring`, never a real crate).
error_test!(error_unknown_boring_stdlib_module);
error_test_exact!(error_undefined_var);
error_test_exact!(error_uncaught_throw);
error_test_exact!(error_move_source);
error_test_exact!(error_immutable_param);
error_test_exact!(error_immutable_let);
error_test_exact!(error_immutable_guard_let_field);
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
