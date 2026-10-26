// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression: `b.f = a.f` with both fields `'actor` used to be transpiled to
// `b.f = Arc::new(Mutex::new(a.f))` (rustc E0308). It must alias instead.

use std::path::Path;
use std::process::Command;

const CASE: &str = "tests/cases/actor_field_alias_assign.br";

#[test]
fn actor_field_alias_assign_is_not_rewrapped() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let out = Command::new(bin).args(["build", CASE, "--emit-rust"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(generated.contains("memo.memory = rt.lm.clone();"), "got:\n{}", generated);
    assert!(!generated.contains("Mutex::new(rt.lm)"), "got:\n{}", generated);
}

#[test]
fn actor_field_alias_assign_compiles_and_runs() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("actor_field_alias_assign_test_scratch");
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let emit = Command::new(bin).args(["build", CASE, "--emit-rust"]).output().unwrap();
    assert!(emit.status.success());
    std::fs::write(dir.join("src/main.rs"), &emit.stdout).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"actor_field_alias_assign_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout), "5\n7\n7\n");
}
