// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression tests for the `'static` qualifier (docs/qualifiers.md's `'static`
// section):
// constant global instances with no Rc/Arc refcounting, `&'static T` in
// generated Rust. Covers the three authorized construction sites (top
// level, `main`-scope, and `type let`'s implicit path), the provenance
// gate (both at a `let`'s own initializer and at call-argument sites),
// `mut`/`'weak` rejection, the `Sync` check under `--threading single`,
// and the generic-struct field rejection.
//
// Run with:
//   cargo test --test static_qualifier

use std::path::Path;
use std::process::Command;

/// Runs `boring build --emit-rust` (optionally with extra flags) on a
/// `tests/cases/*.br` file and asserts it fails with the given stderr
/// substring — the checker/transpiler-level rejection tests.
fn assert_build_rejected(case_name: &str, extra_args: &[&str], expected_stderr_substring: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases").join(case_name);

    let output = Command::new(bin)
        .arg("build")
        .arg(&case_br)
        .arg("--emit-rust")
        .args(extra_args)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    assert!(
        !output.status.success(),
        "expected `boring build --emit-rust{}` on {} to fail, but it succeeded — \
         generated source:\n{}",
        if extra_args.is_empty() { String::new() } else { format!(" {}", extra_args.join(" ")) },
        case_name,
        String::from_utf8_lossy(&output.stdout)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(expected_stderr_substring),
        "expected stderr for {} to contain:\n{}\n--- actual stderr ---\n{}",
        case_name, expected_stderr_substring, stderr
    );
}

/// Runs `boring build --emit-rust` (optionally with extra flags) on a
/// `tests/cases/*.br` file, asserts it succeeds, then builds and runs the
/// generated Rust in a scratch Cargo project and asserts its stdout —
/// the full end-to-end positive tests. `dir_suffix` keeps parallel test
/// runs (e.g. the same case under different `--threading` flags) from
/// colliding on the same scratch directory.
fn assert_build_and_run(case_name: &str, extra_args: &[&str], dir_suffix: &str, expected_stdout: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases").join(case_name);
    let dir = Path::new("tests/cases").join(format!(
        "{}_{}_rust",
        case_br.file_stem().unwrap().to_str().unwrap(),
        dir_suffix
    ));
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let emit = Command::new(bin)
        .arg("build")
        .arg(&case_br)
        .arg("--emit-rust")
        .args(extra_args)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust{}` on {} to succeed, but it failed:\n{}",
        if extra_args.is_empty() { String::new() } else { format!(" {}", extra_args.join(" ")) },
        case_name,
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"static_qualifier_check_{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            dir_suffix
        ),
    )
    .expect("failed to write Cargo.toml");

    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {}", e));
    assert!(
        run.status.success(),
        "expected {} to build and run, but it failed:\n--- stderr ---\n{}\n--- generated source ---\n{}",
        case_name,
        String::from_utf8_lossy(&run.stderr),
        generated,
    );

    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(
        actual.trim_end(),
        expected_stdout,
        "unexpected stdout from {}",
        case_name
    );

    // Clean up so repeated runs don't accumulate disk usage.
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Positive: construction + usage at the three authorized sites ──────────

#[test]
fn top_level_and_main_scope_construction_build_and_run() {
    assert_build_and_run(
        "static_top_level_and_main.br",
        &[],
        "default",
        "direct field: 42\nvalue: 42\nvalue: 99",
    );
}

#[test]
fn top_level_and_main_scope_construction_build_and_run_single_threaded() {
    assert_build_and_run(
        "static_top_level_and_main.br",
        &["--threading", "single"],
        "single",
        "direct field: 42\nvalue: 42\nvalue: 99",
    );
}

#[test]
fn type_let_non_scalar_with_init_body_build_and_run() {
    // The specific bug this feature exists to fix: a non-scalar `type let`
    // field whose constructor has a real `init` body previously emitted
    // `const NAME: Point = Point::new(...);` — invalid Rust (E0015).
    assert_build_and_run("static_type_let.br", &[], "default", "1, 2");
}

// ── Positive: Sync check passes under --threading multi ───────────────────

#[test]
fn nested_shared_under_static_builds_and_runs_multi_threaded() {
    assert_build_and_run("static_sync.br", &["--threading", "multi"], "multi", "1");
}

// ── Positive: generic struct, field independent of T (Case A) ─────────────

#[test]
fn generic_struct_static_field_independent_of_type_param_build_and_run() {
    assert_build_and_run("static_generic_ok.br", &[], "default", "1");
}

// ── Negative: provenance gate ───────────────────────────────────────────────

#[test]
fn construction_outside_authorized_site_is_rejected() {
    assert_build_rejected(
        "static_provenance_bad.br",
        &[],
        "cannot construct a 'static instance here",
    );
}

#[test]
fn passing_non_static_value_to_static_param_is_rejected() {
    assert_build_rejected(
        "static_call_arg_provenance_bad.br",
        &[],
        "cannot pass a non-'static value where 'static is expected",
    );
}

// ── Negative: mut / 'weak ───────────────────────────────────────────────────

#[test]
fn mut_static_is_rejected() {
    assert_build_rejected(
        "static_mut_bad.br",
        &[],
        "cannot combine `mut` with `'static`",
    );
}

#[test]
fn static_weak_is_rejected() {
    assert_build_rejected(
        "static_weak_bad.br",
        &[],
        "'static'weak is invalid",
    );
}

// ── Negative: Sync check under --threading single ──────────────────────────

#[test]
fn nested_shared_under_static_is_rejected_single_threaded() {
    assert_build_rejected(
        "static_sync.br",
        &["--threading", "single"],
        "cannot be 'static under --threading single",
    );
}

// ── Negative: generic struct, field depends on T (Case B) ──────────────────

#[test]
fn generic_struct_static_field_depending_on_type_param_is_rejected() {
    assert_build_rejected(
        "static_generic_bad.br",
        &[],
        "cannot depend on Wrapper's own generic type parameter",
    );
}
