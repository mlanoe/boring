// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `list[cursor[c]] = v` (assignment target whose index is
// itself an indexed read) emitted the outer index without `as usize`, so
// rustc failed with E0277 ("[usize] cannot be indexed by isize").
// `emit_expr_assign`'s index cast only covered Int/Var/BinOp/Field indices.
//
// Run with: cargo test --test nested_index_assign_target

use std::path::Path;
use std::process::Command;

#[test]
fn nested_index_assignment_target_casts_outer_index() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = Path::new("tests/cases/nested_index_assign_target_rust");
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let emit = Command::new(bin)
        .args(["build", "tests/cases/nested_index_assign_target.br", "--emit-rust"])
        .output()
        .unwrap();
    assert!(emit.status.success(), "{}", String::from_utf8_lossy(&emit.stderr));
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();
    std::fs::write(dir.join("src/main.rs"), &generated).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"nested_index_assign_target_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    let build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(
        build.status.success(),
        "{}\n--- generated ---\n{}",
        String::from_utf8_lossy(&build.stderr),
        generated
    );
    let run = Command::new(dir.join("target/debug/nested_index_assign_target_check"))
        .output()
        .unwrap();
    assert!(run.status.success());
    let out = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(out.trim(), "[0, 1, 7, 0]");
}
