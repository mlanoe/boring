// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for `emit_setter` (src/transpiler/emit_struct.rs): a
// setter whose own parameter is named the same as the field it assigns (the
// idiomatic `set balance(float balance): self.balance = balance`) used to
// transpile to a silent no-op. Two compounding bugs, both fixed together
// here (see tests/cases/setter_param_shadows_field.br's own doc comment and
// `emit_setter`/`in_instance_setter`'s doc comments for the full story):
//   1. `emit_setter` never registered the setter's own parameter in
//      `known_local_vars`, so the RHS `balance` in `self.balance = balance`
//      was wrongly resolved as an implicit `self.balance` read.
//   2. Independently, `emit_expr_assign`'s instance-setter dispatch had no
//      guard against recursing into the very setter whose body it was
//      already emitting — `self.balance = ...` (once bug 1 is fixed and the
//      RHS resolves correctly) is itself a `Field` assignment target that
//      matches the registered setter for `balance`, so without a guard it
//      re-dispatches to `self.set_balance(balance)` from inside
//      `set_balance` itself: infinite recursion, a stack overflow at
//      runtime, not merely a no-op.
//
// This test transpiles, compiles, and actually runs the generated Rust —
// the only way to observe either bug (both are silent at `boring build`
// time: bug 1 alone produces valid Rust that just doesn't work; bug 2 only
// crashes at runtime, not at compile time).
//
// Run with:
//   cargo test --test setter_param_shadows_field

use std::path::Path;
use std::process::Command;

#[test]
fn setter_with_param_named_like_field_actually_changes_the_value() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/setter_param_shadows_field.br");
    let dir = Path::new("tests/cases/setter_param_shadows_field_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to succeed, but it failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"setter_param_shadows_field_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
        "expected the generated Rust to build and run (a hang/crash here means the \
         infinite-recursion setter-dispatch bug is back), but it failed:\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&run.stderr),
        generated,
    );

    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(
        actual.trim_end(),
        "42",
        "setter did not actually change `balance` — got: {}",
        actual
    );

    let _ = std::fs::remove_dir_all(dir);
}
