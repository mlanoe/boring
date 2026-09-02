// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a local variable/parameter named after a Rust reserved
// word but a valid Boring identifier (e.g. `ref`, `move`) was correctly
// escaped to `r#ref`/`r#move` at *usage* sites (`emit_methods.rs`'s
// `map_builtin_var`, which routes every plain-variable read through
// `helpers::escape_rust_keyword`) but NOT at the *declaration* site —
// `emit_let.rs`'s `emit_let` (and its `'actor`/`'guard`/managed/lazy/task
// sub-paths) and `emit_top.rs`'s `emit_param` interpolated the raw Boring
// name straight into the generated `let`/`fn` signature text, bypassing the
// helper entirely. This produced invalid Rust: `let ref = 5;` (missing
// `r#`) immediately followed by correctly-escaped `println!("{}", r#ref);`
// — a name mismatch the generated code could never compile with.
//
// `boring run` (the interpreter) is unaffected — it never emits Rust source
// text, so this is a transpiler-only bug. Fixed by routing every
// declaration-site name (let/mut bindings across all `emit_let.rs`
// sub-paths, and function parameters in `emit_top.rs::emit_param`) through
// the same `escape_rust_keyword` helper already used at usage sites.
//
// Run with:
//   cargo test --test reserved_word_let_binding

use std::path::Path;
use std::process::Command;

#[test]
fn reserved_word_bindings_and_params_compile_and_run() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/reserved_word_let_binding.br");
    let dir = Path::new("tests/cases/reserved_word_let_binding_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    // ── boring build --emit-rust must succeed ────────────────────────────
    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to accept {}, but it failed:\n{}",
        case_br.display(),
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    // The bug's exact signature: a bare, un-escaped `let ref`/`let mut move`
    // declaration alongside (already-correct) escaped usages. Assert the
    // declaration sites are escaped too, for both the `let` binding and the
    // `mut` binding, and for the `ref`-named function parameter.
    assert!(
        generated.contains("let r#ref = 5;"),
        "expected the `let ref = 5` declaration to emit as `let r#ref = 5;`, \
         but the generated source is:\n{}",
        generated
    );
    assert!(
        generated.contains("let mut r#move"),
        "expected the `mut move = Counter(0)` declaration to emit as \
         `let mut r#move: ...`, but the generated source is:\n{}",
        generated
    );
    assert!(
        generated.contains("fn bump(r#ref:"),
        "expected the `ref`-named parameter to emit as `fn bump(r#ref: ...)`, \
         but the generated source is:\n{}",
        generated
    );
    assert!(
        !generated.contains("let ref ="),
        "found an un-escaped `let ref =` declaration — the bug this test \
         pins — in the generated source:\n{}",
        generated
    );

    // ── And the generated Rust must actually build and run ───────────────
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"reserved_word_let_binding_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
    assert_eq!(
        actual.trim_end(),
        "5 2\nparam 5",
        "unexpected stdout for reserved-word-named bindings/parameter"
    );

    // Clean up the generated build dir so repeated runs don't accumulate
    // disk usage (target/ dirs in particular).
    let _ = std::fs::remove_dir_all(dir);
}
