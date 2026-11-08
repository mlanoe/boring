// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: an indexed write whose index reads the same array on a `mut` array
// parameter (`f[f.length - 1] = v`) used to transpile to `f[(f.len() ...) as usize] = v`,
// which fails the borrow checker (E0502: `f` is a `&mut Vec`, so no two-phase borrow).
// `emit_expr_assign` now evaluates such indices into temporaries first. Covers plain,
// compound (`+=`) and nested writes, and compares `boring build` + `cargo run` with `boring run`.
//
// Run with: cargo test --test index_assign_self_reading_index

use std::path::Path;
use std::process::Command;

#[test]
fn self_reading_index_assignment_builds_and_matches_interpreter() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cases/index_assign_self_reading_index.br");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/index_assign_self_reading_index_scratch");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(&case, dir.join("case.br")).unwrap();

    let run = Command::new(bin).arg("run").arg(&case).output().unwrap();
    assert!(run.status.success(), "boring run failed: {}", String::from_utf8_lossy(&run.stderr));
    let expected = String::from_utf8_lossy(&run.stdout).to_string();
    assert_eq!(expected.trim(), "12 3\n7 0 7\n4 14 9");

    let build = Command::new(bin).current_dir(&dir).arg("build").arg("case.br").output().unwrap();
    assert!(build.status.success(), "boring build failed: {}", String::from_utf8_lossy(&build.stderr));
    let out = Command::new("cargo").current_dir(dir.join("case_rust")).args(["run", "--quiet"]).output().unwrap();
    assert!(out.status.success(), "generated project failed to build/run:\n{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout), expected);
}
