// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for an `if let`/`elif let` field-mutation bug — the same one
// `guard let` had until it was fixed (see `tests/guard_let_immutable_field.rs`).
// An `if let b = ...`/`elif let b = ...` binding has no `mut`/`var mut`
// spelling (see `ast::CondClause::Let`, which carries no `BindingKind`), so it
// should be exactly as immutable as a plain `let b = ...` — writing to one of
// its fields (`b.field = x`) must be rejected by `boring build` with a clear
// Boring diagnostic, the same way it already is for a classic `let` and for
// `guard let`.
//
// Root cause: `emit_match.rs`'s `emit_if_let` registered the binding in
// `known_local_vars` but never in `mut_checked_local_vars` (unlike
// `emit_let.rs`'s unconditional insert for a plain `let_stmt`, and unlike
// `emit_flow.rs`'s `emit_guard` after its own fix), so `emit_expr.rs`'s
// field-write diagnostic silently short-circuited and the transpiler emitted
// invalid Rust (`if let Some(b) = ... { b.field = ...; }` on a non-`mut`
// binding) that only failed downstream at `cargo build` with E0594, instead of
// a clear Boring-level error out of `boring build` itself.
//
// Extra wrinkle versus `guard let`: an `if let`/`elif let` binding is
// block-scoped (unlike `guard let`, which lives to the end of the enclosing
// function), so the fix also has to remove the binding from
// `mut_checked_local_vars` and friends once the block is done, mirroring
// `emit_match_arm`'s existing scope-exit cleanup for match-arm pattern
// bindings — otherwise a later, unrelated local reusing the same name would
// wrongly inherit the earlier binding's tracking. `if_let_field_write_scope_
// exit_reuse_still_compiles` below pins that half of the fix specifically.
//
// `boring run` (the interpreter) already got this right independently — see
// `interpreter/mod.rs`'s `Env::define`, the same call used for a plain `let`
// and an `if let` binding alike — so this test pins the transpiler side
// specifically. The interpreter side is covered by
// `tests/cases/error_immutable_if_let_field.br`/`.error` via
// `tests/run.rs`'s `error_test_exact!`.
//
// Run with:
//   cargo test --test if_let_immutable_field

use std::path::Path;
use std::process::Command;

fn boring_build_emit_rust(case_br: &Path) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e))
}

/// Build+run a generated Rust source in a scratch Cargo project under `dir`,
/// asserting it compiles and returning its trimmed stdout. Cleans `dir` up on
/// success so repeated runs don't accumulate `target/` disk usage.
fn cargo_run_generated(dir: &Path, crate_name: &str, generated: &str) -> String {
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");
    std::fs::write(dir.join("src/main.rs"), generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            crate_name
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
        "expected the generated Rust to build and run, but it failed:\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&run.stderr),
        generated,
    );

    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    let _ = std::fs::remove_dir_all(dir);
    actual.trim_end().to_string()
}

#[test]
fn if_let_field_write_is_rejected_by_boring_build() {
    let case_br = Path::new("tests/cases/error_immutable_if_let_field.br");
    let emit = boring_build_emit_rust(case_br);

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to reject a field write through a \
         non-mut `if let` binding, but it succeeded — generated source:\n{}",
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
fn if_let_field_write_via_var_mut_escape_hatch_still_compiles() {
    let case_br = Path::new("tests/cases/if_let_mut_field_escape.br");
    let emit = boring_build_emit_rust(case_br);
    assert!(
        emit.status.success(),
        "expected the `var mut` escape hatch (rebinding an `if let` result \
         into a fresh `var mut` local) to still transpile cleanly, but \
         `boring build --emit-rust` failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    let dir = Path::new("tests/cases/if_let_mut_field_escape_rust");
    let actual = cargo_run_generated(dir, "if_let_mut_field_escape_check", &generated);
    assert_eq!(actual, "changed", "unexpected stdout from the escape-hatch case");
}

/// Pins the block-scoping half of the fix: an `if let b = ...:` binding must
/// stop being tracked as `mut`-checked once its block ends, so a later,
/// unrelated `var mut Block b = ...` local reusing the same name isn't
/// spuriously affected (and, before the fix existed, so the checked-ness
/// itself didn't leak past the block either).
#[test]
fn if_let_field_write_scope_exit_reuse_still_compiles() {
    let case_br = Path::new("tests/cases/if_let_scope_exit_reuse.br");
    let emit = boring_build_emit_rust(case_br);
    assert!(
        emit.status.success(),
        "expected a later, unrelated `var mut Block b = ...` local reusing an \
         `if let` binding's name to transpile cleanly, but \
         `boring build --emit-rust` failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    let dir = Path::new("tests/cases/if_let_scope_exit_reuse_rust");
    let actual = cargo_run_generated(dir, "if_let_scope_exit_reuse_check", &generated);
    assert_eq!(actual, "x\nz", "unexpected stdout from the scope-exit-reuse case");
}
