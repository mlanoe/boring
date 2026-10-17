// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `Body(pos = b.pos, ...)` where `b` is a borrowed struct
// parameter and `pos` has a one-letter struct type (`V`) emitted `b.pos` with
// no `.clone()` (E0507, cannot move out of a shared reference). One-letter
// names parse as `Type::TypeParam`, which `field_read_needs_clone` did not
// treat as a declared struct/enum. Also checks `boring run` parity.
//
// Run with: cargo test --test struct_field_clone_one_letter

use std::path::Path;
use std::process::Command;

#[test]
fn one_letter_struct_field_read_is_cloned() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/struct_field_clone_one_letter.br");
    let dir = Path::new("tests/cases/struct_field_clone_one_letter_rust");
    std::fs::create_dir_all(dir.join("src")).expect("create src dir");

    let emit = Command::new(bin).arg("build").arg(case_br).arg("--emit-rust").output().unwrap();
    assert!(emit.status.success(), "emit-rust failed:\n{}", String::from_utf8_lossy(&emit.stderr));
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();
    assert!(generated.contains("pos: b.pos.clone()"), "missing .clone():\n{}", generated);

    std::fs::write(dir.join("src/main.rs"), &generated).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"struct_field_clone_one_letter_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    ).unwrap();
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&run.stdout).trim().to_string();
    let err = String::from_utf8_lossy(&run.stderr).into_owned();
    let _ = std::fs::remove_dir_all(dir);
    assert!(run.status.success(), "cargo run failed:\n{}\n{}", err, generated);
    assert_eq!(out, "8 1 2");

    let interp = Command::new(bin).arg("run").arg(case_br).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&interp.stdout).trim(), "8 1 2");
}
