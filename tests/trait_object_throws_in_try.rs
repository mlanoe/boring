// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A `throws` method called through a trait object (`o.handler.onKey(ev)`, receiver typed as the
// trait) inside `try:` must propagate its error with `?` -- statement form and value form
// (`let r = o.handler.onKey(ev)`). Before the fix the transpiler registered `throws` only for
// concrete struct methods, so the trait-typed receiver emitted `...onKey(&mut ev);` with the
// `Result` dropped and the `catch` never ran under `boring build` (the interpreter was right).
//
// Run with:
//   cargo test --test trait_object_throws_in_try

use std::path::Path;
use std::process::Command;

const CASE: &str = "tests/cases/trait_object_throws_in_try.br";
const EXPECTED: &str = "ok true\ncaught observer failed\nok2 true\ncaught2 observer failed\n";

#[test]
fn interpreter_catches_trait_object_throw() {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run").arg(CASE)
        .output().expect("failed to invoke boring");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout), EXPECTED);
}

#[test]
fn transpiler_propagates_trait_object_throw() {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("build").arg(CASE).arg("--emit-rust")
        .output().expect("failed to invoke boring");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("o.handler.clone().onKey(&mut ev)?;"),
        "statement-form trait-object throwing call must end in `?`:\n{}", generated
    );
    assert!(
        generated.contains("let r = o.handler.clone().onKey(&mut ev)?;"),
        "value-form trait-object throwing call must end in `?`:\n{}", generated
    );
}

#[test]
fn built_binary_catches_trait_object_throw() {
    let dir = std::env::temp_dir().join(format!("boring_trait_throws_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(Path::new(CASE), dir.join("main.br")).unwrap();
    let build = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("build").arg("main.br").current_dir(&dir)
        .output().expect("failed to invoke boring");
    assert!(build.status.success(), "{}", String::from_utf8_lossy(&build.stderr));
    let run = Command::new("cargo")
        .args(["run", "-q"]).current_dir(dir.join("main_rust"))
        .output().expect("failed to invoke cargo");
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout), EXPECTED);
    let _ = std::fs::remove_dir_all(&dir);
}
