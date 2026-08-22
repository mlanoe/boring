// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a `guard let` field-mutation bug: a `guard let b =
// ...` binding has no `mut`/`var mut` spelling (see `ast::CondClause::Let`
// and `parser::parse_stmt::parse_cond_clause`, which only ever consumes a
// bare `let`), so it should be exactly as immutable as a plain `let b =
// ...` — writing to one of its fields (`b.field = x`) must be rejected by
// `boring build` with a clear Boring diagnostic, the same way it already is
// for a classic `let`.
//
// Root cause: `emit_flow.rs`'s `emit_guard` registered the binding in
// `known_local_vars` but never in `mut_checked_local_vars` (unlike
// `emit_let.rs`'s unconditional insert for a plain `let_stmt`), so
// `emit_expr.rs`'s field-write diagnostic (`known_local_vars.contains(v) &&
// !content_mutable_local_vars.contains(v) && mut_checked_local_vars.
// contains(v)`) silently short-circuited on the last condition and the
// transpiler emitted invalid Rust (`let Some(b) = ... else { ... }; b.field
// = ...` on a non-`mut` Rust binding) that only failed downstream at `cargo
// build` with E0594 — instead of a clear Boring-level error out of `boring
// build` itself.
//
// `boring run` (the interpreter) already got this right independently — see
// `interpreter/mod.rs`'s `Env::define`, which is the same call used for a
// plain `let` and a `guard let` binding alike — so this test pins the
// transpiler side specifically. The interpreter side is covered by
// `tests/cases/error_immutable_guard_let_field.br`/`.error` via
// `tests/run.rs`'s `error_test_exact!`.
//
// Run with:
//   cargo test --test guard_let_immutable_field

use std::path::Path;
use std::process::Command;

#[test]
fn guard_let_field_write_is_rejected_by_boring_build() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/error_immutable_guard_let_field.br");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to reject a field write through a \
         non-mut `guard let` binding, but it succeeded — generated source:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    let expected = "`b` is not declared `mut` — cannot assign to field `.opCode` on a non-mut binding";
    assert!(
        stderr.contains(expected),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        expected, stderr
    );
}

#[test]
fn guard_let_field_write_via_var_mut_escape_hatch_still_compiles() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/guard_let_mut_field_escape.br");
    let dir = Path::new("tests/cases/guard_let_mut_field_escape_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    // ── boring build --emit-rust must still succeed for the escape hatch ────
    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "expected the `var mut` escape hatch (rebinding a `guard let` result \
         into a fresh `var mut` local) to still transpile cleanly, but \
         `boring build --emit-rust` failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    // ── And the generated Rust must actually build and run ──────────────────
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"guard_let_mut_field_escape_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
        "expected the `var mut` escape-hatch case to build and run, but it \
         failed:\n--- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&run.stderr),
        generated,
    );

    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(
        actual.trim_end(),
        "changed",
        "unexpected stdout from the escape-hatch case"
    );

    // Clean up the generated build dir so repeated runs don't accumulate disk
    // usage (target/ dirs in particular).
    let _ = std::fs::remove_dir_all(dir);
}
