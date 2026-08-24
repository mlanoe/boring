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
//   cargo test --test transpile -- --test-threads=1
//
// The first run compiles all generated Rust projects from scratch (slow).
// Subsequent runs reuse the cargo build cache (fast, < 2s per test).
//
// IMPORTANT: always pass `--test-threads=1`. Every generated project's inner `cargo
// run`/`cargo build` is pointed at one shared build directory (see `shared_target_dir`
// below) to avoid duplicating tokio/serde's build output per fixture. Cargo's own
// locking on that shared dir serializes concurrent builds anyway, so the default
// multi-threaded test runner buys no real parallelism -- it just piles up N test
// threads all queued on the same build lock. Measured: with the default thread count
// this queuing degraded into an effective multi-hour stall (one thread parked over 60s
// on a single build, the whole run ultimately killed after 10+ hours); run serially,
// the full 416-test suite passes in ~5 minutes.

use std::path::{Path, PathBuf};
use std::process::Command;

// Every generated fixture project used to get its own `target/` dir (each one pulling in
// tokio and/or serde as real dependencies, ~100-300MB apiece) -- across 400+ tests that
// blew past 40GB on a full cold run and once filled the disk outright. Point every inner
// `cargo run`/`cargo build` spawned by this suite at one shared build directory instead;
// different generated packages have distinct names so they coexist fine there, and cargo's
// own locking serializes concurrent builds against it. This must NOT be set on the *outer*
// `cargo test` process (only on these inner spawned commands) -- the outer cargo already
// holds a lock on its own target dir, so pointing both at the same path risks a deadlock.
fn shared_target_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("transpile-cases")
}

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
        .env("CARGO_TARGET_DIR", shared_target_dir())
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

// ── Project-mode cases (need a real external Cargo dependency) ─────────────────────────
//
// `run_transpile_case_with_config` above always runs `boring build <file.br> --output-dir
// <dir>` (single-file mode), which never reads a `boring.toml` -- only the no-file
// "project mode" path (`boring build` with the working directory set to a project
// containing one) merges a `[dependencies]` section into the generated Cargo.toml (see
// `main.rs`'s `build_project_with_config` vs. `emit_rust_to_dir`'s other call sites,
// which always pass `extra_deps: &[]`). A case that needs a real external crate --
// tests/cases/ext_const_promotion, exercising a top-level `let` whose initializer calls
// into an external/opaque type (tests/cases/fixtures/ext_geom stands in for something
// like `bevy`/`glam`) -- has to go through that path instead.
fn run_transpile_project_case(name: &str, mode_str: &str, threading_str: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let project_dir = Path::new("tests/cases").join(name);
    let expected_file = Path::new("tests/cases").join(format!("{}.expected", name));

    // Mirrors `rust_dir_name_full` (src/main.rs) for `main = "src/main.br"` projects: the
    // generated Cargo project lands at `<project_dir>/src/main_rust[_managed][_single]`.
    let rust_dir_name = match (mode_str, threading_str) {
        ("strict",  "multi")  => "main_rust",
        ("strict",  "single") => "main_rust_single",
        ("managed", "multi")  => "main_rust_managed",
        ("managed", "single") => "main_rust_managed_single",
        _ => panic!("[{}] unknown mode/threading combo: {}+{}", name, mode_str, threading_str),
    };
    let rust_dir = project_dir.join("src").join(rust_dir_name);

    // ── Step 1: emit Rust (project mode: no file arg, cwd = the project dir) ───────────
    let emit = Command::new(bin)
        .arg("build")
        .arg("--mode").arg(mode_str)
        .arg("--threading").arg(threading_str)
        .current_dir(&project_dir)
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
        .env("CARGO_TARGET_DIR", shared_target_dir())
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

macro_rules! transpile_project_test {
    ($name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn strict_multi()   { run_transpile_project_case(stringify!($name), "strict",  "multi"); }
            #[test]
            fn strict_single()  { run_transpile_project_case(stringify!($name), "strict",  "single"); }
            #[test]
            fn managed_multi()  { run_transpile_project_case(stringify!($name), "managed", "multi"); }
            #[test]
            fn managed_single() { run_transpile_project_case(stringify!($name), "managed", "single"); }
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
transpile_test!(throws_untyped_enum_catch);
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
// `use boring.collections` — the first-party stdlib mechanism (docs/cross-
// project-code-sharing-gap.md's stdlib work), backed by src/stdlib_embed.rs.
// See tests/cases/boring_stdlib_collections.br's own doc comment. Paired
// with tests/run.rs's registration above for the `boring run` side.
transpile_test!(boring_stdlib_collections);
// Named cross-project dependency (docs/cross-project-code-sharing-gap.md's [deps]
// work): `use numlib.big_uint.*` resolves against tests/cases/fixtures/dep_numlib via
// tests/cases/cross_project_dep/boring.toml's own `[deps]` section. Project mode (its
// own boring.toml, not a flat tests/cases/<name>.br file) so a real `cargo run`
// exercises the generated Rust — same reason `ext_tuple_construct` needs project
// mode. Paired with tests/run.rs's `cross_project_dep` test for the `boring run` side.
transpile_project_test!(cross_project_dep);
transpile_test!(ownership);
transpile_test!(tasks);
transpile_test!(channels);
transpile_test!(streams);
transpile_test!(let_pattern);
transpile_test!(result_compat);
transpile_test!(multi_catch);
transpile_test!(implicit_self);
transpile_test!(shadowing);
transpile_test!(for_destructure);
transpile_test!(struct_spread);
transpile_test!(default_rest);
transpile_test!(tuple_string);
transpile_test!(tuple_methods);
transpile_test!(tuple_map);
transpile_test!(array_pop_remove);
transpile_test!(optional_pop_tail_call);
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
transpile_test!(int_literal_overflow_cast);
transpile_test!(builtin_error_enum);
transpile_test!(typed_catch_match_error);
// A type-level method's (`type def`/`type req`/`type set`) `throws` was previously
// ignored entirely by `emit_type_method` (src/transpiler/emit_struct.rs) -- no
// `Result<T, E>` wrapping, so `throw` fell through to a bare `panic!(...)` under
// `boring build` even though `boring run` (the interpreter) already handled it
// correctly. Untyped `throws:` and typed `throws Type:` both exercised: typed_throws
// additionally needs its thrown enum registered as a typed error (Display/BoringVal
// impls) the same way a regular throwing function's typed throws_ty already is.
transpile_test!(type_def_typed_throws);
transpile_test!(type_method_throws_untyped);
// An enum's
// `type_methods` were silently dropped from codegen entirely -- `enum Foo { A(isize) }`
// with no `impl Foo { fn make() ... }` block, even though the call site (`Foo::make()`)
// was still emitted. Fixed in `emit_enum` (src/transpiler/emit_struct.rs), reusing the
// same `emit_type_method` helper the struct side already used. `enum_type_def` is the
// exact untyped repro from the doc; `enum_type_def_throws` additionally covers a typed
// `throws Type:` clause on an enum type method, mirroring `type_def_typed_throws` (item 4).
transpile_test!(enum_type_def);
transpile_test!(enum_type_def_throws);
transpile_test!(ref_identity);
transpile_test!(mut_scalar);
transpile_test!(int_float_literal_compare);
transpile_test!(float32_math_builtins);
transpile_test!(float32_struct_method_math);
// Same gap as float32_struct_method_math above, but for a plain `let`-bound local
// variable computed from an unannotated arithmetic expression, in an ordinary
// (non-method) function — see tests/cases/float32_local_var_math.br's doc comment.
transpile_test!(float32_local_var_math);
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
// Same caveat as `pub_top_level_const` above -- this single-crate run only proves
// `pub let SCOREBOARD_FONT_SIZE`/`SCOREBOARD_TEXT_PADDING` (zero in-file consumers,
// the actual bug-report shape) still compile and run inside their own crate. It
// can't observe promotion-vs-not on its own, since nothing in this file needs to
// see them either way -- that's exactly `pub_module_const.rs`'s
// `pub_let_with_no_in_file_consumer_is_visible_across_modules` test's job: a real
// two-file crate where a hand-written sibling module reads the constant, which
// only compiles if it was promoted to a real `pub const` despite never being
// referenced anywhere in the `.br` file itself.
transpile_test!(pub_top_level_const_unused);
// `pub let`/used-elsewhere `let` STRING constants (`top_level_let_is_string_literal`,
// src/transpiler/mod.rs) -- the scalar cases above never exercised this path at all, since
// `top_level_let_is_const_safe` never matches a string. Unlike `pub_top_level_const_unused`
// (zero in-file consumers, needs `pub_module_const.rs`'s cross-module `cargo build` to prove
// anything), every constant here IS read from another function in the same file
// (`is_add`/`is_sub`/`reveal_private`), so this single-crate `cargo run` already proves
// promotion on its own: an un-promoted string `let` stays a `fn main()` local, and
// `block_opcode_is(opcode, OP_ADD)` (etc.) would fail to compile with E0425 "cannot find
// value" rather than silently producing a wrong runtime value. Also covers the explicit-
// type-annotation sibling (`pub let string OP_SUB = ...`) and confirms a private-but-
// used `let` (`PRIVATE_USED`) promotes too, not just `pub` ones -- run across all four
// mode/threading combinations catches both the `Sync`-required-for-`static` `Arc<str>`-
// vs-`Rc<str>` mismatch under `--threading single` (see `emit_expr_owned`'s
// `global_string_const_names` arm, src/transpiler/emit_top.rs) and the invalid-syntax
// failure the bug report was originally about.
transpile_test!(pub_top_level_string_const);
// A top-level `let` whose initializer calls into an external/opaque type -- one hand-
// verified as a `const fn` (`Duration.from_secs`/`from_millis`, see
// `Transpiler::KNOWN_EXTERNAL_CONST_FNS`) -- promotes as a plain `const`, same as a
// scalar literal. Companion to `ext_const_promotion` below, which covers the `static`
// fallback for a constructor NOT on that list.
transpile_test!(const_promotion_known_fn);
// A struct field literally named `count` used to be hijacked by the
// `.length`/`.count` -> `.len() as isize` collection-length builtin in BOTH
// `emit_expr_field` and its owned-context twin `emit_expr_owned` (their
// `is_user_field` guard resolved the receiver's struct type only for the
// literal `self` receiver, so a function parameter, a local `let`, and a
// field-of-field chain all fell through to the builtin) -- `Thing` here has
// no `.len()` method at all, so this used to fail to compile outright; this
// case's real value is exactly that real `cargo run` compile, same rationale
// as `top_level_const` above.
transpile_test!(struct_count_field);
// String-interpolating a struct field whose type is an array (`[int]`,
// `[string]`, and one level of field chaining) emitted a bare
// `println!("{}", p.scores)` on a `Vec<isize>`/`Vec<Rc<str>>` field -- these
// have no `Display` impl, so it failed to compile (E0277) even though the
// exact same array in a local variable was already wrapped in the
// transpiler's `BoringFmt` shim. This case's real value is the `cargo run`
// compile, same rationale as `struct_count_field` above.
transpile_test!(struct_field_array_interp);
// Dict `[key]` indexing (read via `else`, write via `=`) with a non-integer
// (string) key always cast the key `as usize` -- invalid for `Arc<str>` --
// whenever the dict-typed receiver wasn't recognized as a dict: `dict_vars`
// was only ever populated from local `let`/`var` declarations, never from
// function parameters, and the struct-field Dict check matched `Type::Dict(..)`
// directly, missing every `mut`/`var mut` dict field (wrapped in `Type::Mut(..)`
// at parse time). A second, separate bug dropped the `self.` prefix entirely
// on `table[id] = v` when `table` was an implicit-self struct field (the
// assignment-target codegen path never checked `self_type`/`struct_fields`
// the way the read path already did) -- neither compiles as real Rust, which
// only `cargo run` (this test), not `boring run`, catches. See tests/cases/
// dict_string_key_index.br's own doc comment.
transpile_test!(dict_string_key_index);
// A `var StructType` parameter (docs/CLAUDE.md: "passes `&mut T`; changes are
// visible at the call site") actually cloned the argument before taking the
// `&mut` reference -- `&mut v.clone()` borrows a throwaway temporary, so the
// callee's mutation never reached the real caller variable. Only reachable
// via a real compile+run (the interpreter's object model has no such
// clone-before-mutate step to get wrong). See tests/cases/
// var_struct_param_mutation.br's own doc comment.
transpile_test!(var_struct_param_mutation);
// A user-declared method/field whose name collides with a builtin Rust Iterator/Vec
// adapter (`position`, `count`) must dispatch to the user's own declaration, on both
// a struct AND an enum -- see the file's own doc comment for the full writeup of the
// bug this guards against (an enum receiver was invisible to the "is this a real user
// type" check, so `map_method`/`map_field` fired unconditionally regardless of what the
// enum actually declared).
transpile_test!(builtin_name_user_members);
transpile_test!(implicit_self_length_nontail);
transpile_test!(throws_method_name_collision);
transpile_test!(narrowing_cast_if_let);
transpile_test!(try_else_nil_if_let);
// `try`/`try?` used to only be recognized as a prefix inside `parse_else_expr`,
// one precedence level above where `guard let x = EXPR`/`if let x = EXPR`
// clauses parse their RHS (`parse_or`) -- so a bare (unparenthesized)
// `guard let x = try? foo() else ...` was a parse error. See
// tests/cases/try_prefix_in_cond_clause_noparen.br's own doc comment.
transpile_test!(try_prefix_in_cond_clause_noparen);
// Vec::first()/last() and HashMap::get(k) return Option<&T> in Rust; Boring's
// first()/last()/get() are documented (book.md) as owned `T?`. This is the real
// value of this case: `cargo run` on the emitted Rust only compiles if `.cloned()`
// was inserted AND the already-Option-shaped result wasn't double-wrapped in
// Some(...) (which fails to type-check against `Option<isize>`).
transpile_test!(option_owned_methods);
// Note: nil_assign (type inference for nil variables), pattern_some (Some/None on non-Option),
// and closure_break (break inside closure) are interpreter-only tests — not added here.

// Three related bugs, all only reachable with a genuinely external (non-Boring) type --
// see tests/cases/ext_const_promotion/src/main.br's own doc comment for the full
// writeup, and tests/cases/fixtures/ext_geom for the stand-in external crate:
//   1. A top-level `let` calling into an external type's (non-const-fn) constructor used
//      to stay a `fn main()` local instead of being promoted to module scope, dropped
//      out of scope for every other function that referenced it (E0425 in real Rust,
//      even though `boring run`'s interpreter resolved it fine).
//   2. Field access on such a promoted value (`PADDLE_SIZE.x`) used to mis-emit as a
//      type-level path lookup (`PADDLE_SIZE::x`) instead of a real field access.
//   3. Using such a promoted value at a struct-literal field VALUE position
//      (`Sprite(color = PADDLE_COLOR)`) -- unlike a `.field` read or method-call
//      receiver, this needs the exact type `T`, not `LazyLock<T>` (Rust never
//      auto-derefs an owned value there). Covers both the fix (a hand-verified
//      `const fn` call promotes straight to `const`, already exactly `T`) and the
//      documented escape valve for a call that stays `static ... LazyLock<T>`
//      (explicit `.pointee`).
// Needs project mode (a real `boring.toml` + `[dependencies]`), not the single-file
// `run_transpile_case_with_config` every other case above uses — see
// `run_transpile_project_case`'s doc for why.
transpile_project_test!(ext_const_promotion);

// Companion to `ext_const_promotion` just above, covering a different initializer
// shape into the same const-promotion decision: a top-level `let` calling into an
// external *enum's* tuple variant via the dot-shorthand (`FontSize.Px(33.0)`) rather
// than an external struct's associated function (`Color.srgb(...)`). Boring parses
// both shapes identically (`MethodCall(Var(Type), method, args)`), but the const-vs-
// static decision used to only recognize the hand-verified-const-fn allowlist
// (`Transpiler::KNOWN_EXTERNAL_CONST_FNS`), so an enum tuple-variant construction
// always fell back to `static ... LazyLock<T>` -- valid Rust at a `.field` read, but a
// real `cargo build` E0308 at a struct-literal field-VALUE position (the position
// `ext_enum_const_promotion/src/main.br`'s `make_label` actually exercises), same
// failure mode as `ext_const_promotion`'s `PADDLE_COLOR`/`PADDLE_SIZE` before their own
// fix. See `Transpiler::is_external_enum_variant_construction` (src/transpiler/mod.rs)
// for the fix -- unlike `KNOWN_EXTERNAL_CONST_FNS`, this needs no allowlist at all,
// since constructing an enum variant is unconditionally const-evaluable in Rust
// regardless of which enum/variant. Needs project mode for the same reason as
// `ext_const_promotion` (a real `[dependencies]` path crate).
transpile_project_test!(ext_enum_const_promotion);

// Bare `Type(args)` construction of a genuinely external/opaque tuple struct with no
// inherent `new()` (e.g. bevy's `Mesh2d`/`MeshMaterial2d`) used to unconditionally
// rewrite to `Type::new(args)`, which doesn't compile (E0599) -- see tests/cases/
// ext_tuple_construct/src/main.br's own doc comment, tests/cases/fixtures/ext_tuple for
// the stand-in external crate, and `Transpiler::KNOWN_EXTERNAL_TUPLE_STRUCTS`'s doc in
// src/transpiler/mod.rs for the fix. Needs project mode for the same reason as
// `ext_const_promotion` just above (a real `[dependencies]` path crate).
transpile_project_test!(ext_tuple_construct);

// task_791a91d0: the field-resolution guard `struct_count_field` added (a real struct
// field named `count`/`length` must win over the `.len() as isize` builtin) only
// resolved the receiver's struct type for a *plain* `Named` parameter -- a Bevy system
// parameter declared through the `Res<T>`/`ResMut<T>` deref-transparent wrapper (the
// realistic, common shape for a Bevy resource read) still fell through to the builtin
// on the same field name. See tests/cases/ext_res_field/src/main.br's own doc comment,
// tests/cases/fixtures/ext_res for the stand-in `Res`/`ResMut` crate, and
// `TRANSPARENT_WRAPPER_GENERICS`'s doc in src/transpiler/emit_methods.rs for the fix.
// Needs project mode for the same reason as `ext_const_promotion` above (a real
// `[dependencies]` path crate) -- and specifically a real `cargo run`, not just
// `boring run`: the interpreter resolves field access dynamically and never hits this
// codegen-only bug.
transpile_project_test!(ext_res_field);

// `TextColor` specifically (bevy::prelude::TextColor, bevy_text 0.19) -- the concrete
// case that motivated auditing/extending `Transpiler::KNOWN_EXTERNAL_TUPLE_STRUCTS`
// beyond `Mesh2d`/`MeshMaterial2d` above; see tests/cases/text_color_construct/src/
// main.br's own doc comment. Reuses the `ext_tuple` fixture crate (extended with a
// stand-in `TextColor`), same reason as `ext_tuple_construct`.
transpile_project_test!(text_color_construct);

// `ClearColor` specifically (bevy::prelude::ClearColor, bevy_camera 0.19) -- found
// migrating `breakout-boring/src/lib.rs`'s `BACKGROUND_COLOR` constant to a genuine
// Boring `Startup` system; third instance of the same `KNOWN_EXTERNAL_TUPLE_STRUCTS`
// gap after `Mesh2d`/`MeshMaterial2d` and `TextColor` above. See tests/cases/
// clear_color_construct/src/main.br's own doc comment. Reuses the `ext_tuple` fixture
// crate (extended with a stand-in `ClearColor`), same reason as `ext_tuple_construct`.
transpile_project_test!(clear_color_construct);

// `boring.toml [external_fns]`'s built-in `KNOWN_EXTERNAL_FN_BORROWS` supplement
// (src/transpiler/mod.rs) -- `std::mem::swap`/`replace`/`take` argument-borrow
// whitelisting, plus the `expr_is_path_receiver` fix for `mem.take(t)` being
// misidentified as the generic Vec/Iterator `.take(n)` adapter. See tests/cases/
// mem_borrow_builtins.br's own doc comment. Zero external Cargo dependency (std::mem
// only), so single-file mode is enough -- no boring.toml/[external_fns] needed here,
// since this exercises the compiler's *built-in* table, not a project-declared one.
transpile_test!(mem_borrow_builtins);

// `@derive(Serialize, Deserialize)`/`fromJson<T>()`/`json()` -- see tests/cases/
// json_serde_rename.br's own doc comment for the two real bugs this pins (a bare,
// as-documented `@derive(Serialize, Deserialize)` never compiled at all: no serde
// import was ever emitted, and the auto Display impl assumed Debug unconditionally)
// plus the field-level `@serde(rename = "...")` addition. Now registered in
// tests/run.rs too -- both `fromJson` and `json(v)` are real on the interpreter side
// (src/interpreter/json.rs), so this case's second line agrees between backends too.
transpile_test!(json_serde_rename);

// docs/try-wrap-double-handling-bug.md -- `try? EXPR` used to double-handle builtins
// (`fromJson<T>(s)`, `fs.read`/`fs.readLines`/`fs.readBytes`) that already do their own
// Result->Option/panic handling in a plain context: `.ok()` got appended a second time
// (didn't compile), `fs.read`'s inner `.unwrap()` panicked on a real read failure
// instead of yielding `None`, and a `try? EXPR` tail-returned from a `T?` function got
// an extra `Some(...)` wrapped around its already-`Option<T>` value. See
// tests/cases/try_wrap_double_handling.br's own doc comment for the full breakdown.
// Now registered in tests/run.rs too (it never used `json()`, only `fromJson`, so a
// real interpreter-side `fromJson` makes both backends agree byte-for-byte).
transpile_test!(try_wrap_double_handling);

// docs/interpreter-untagged-enum-fromjson-mismatch.md -- `boring run` and `boring build`
// printed completely different things for `fromJson<T>` on a `@serde(untagged)` enum,
// because the interpreter's `fromJson` was a no-op stub returning its input string and
// its enum `Display` formatted payloads with Display where the compiled program uses
// derived `Debug`. Unlike json_serde_rename/try_wrap_double_handling above, these three
// ARE registered in tests/run.rs too, against these same .expected files -- pinning the
// same bytes for both backends is precisely the regression test for that divergence.
// See each fixture's own doc comment for what it covers:
//   json_untagged_enum   -- the doc's repro (untagged enum, every JSON shape)
//   json_serde_shapes    -- struct rename/rename_all + externally-tagged enums
//   enum_derive_no_debug -- the enum auto-`Display` missing the struct path's
//                           `will_have_debug` gate: `@derive(Clone, Serialize,
//                           Deserialize)` on an enum emitted `write!(f, "{:?}", self)`
//                           for a non-`Debug` type and never compiled.
transpile_test!(json_untagged_enum);
transpile_test!(json_serde_shapes);
transpile_test!(enum_derive_no_debug);

// docs/self-field-loop-match-borrow-bug.md -- a `for`/`match` directly over a bare
// struct field (implicit `self.field`) inside a `req`/`def` method generated Rust
// that consumed the field by value even though the method only holds a borrowed
// `&self`/`&mut self`: `for s in items:` over a `[T]` field emitted
// `self.items.into_iter()` (E0507), `match status:` over an enum field bound arms
// by value out of the same borrow, and `for k, v in scores:` over a `{K=V}` field
// mis-transpiled through the array-style auto-enumerate path
// (`.enumerate().map(|(i, v)| (i as isize, v))`), binding the loop var to the
// whole `(k, v)` pair instead of just the value. See tests/cases/
// self_field_loop_match_borrow.br's own doc comment. Also registered in
// tests/run.rs -- the interpreter's output is the semantic baseline this pins
// the compiled output against.
transpile_test!(self_field_loop_match_borrow);
