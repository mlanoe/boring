// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `_ = arr.pop()` on `[T]` where T is a plain user struct
// (no `Default`) used to transpile to `arr.pop().unwrap_or_default()`, which
// fails to compile (E0277: `T: Default` not satisfied). A discarded pop must
// emit the raw `Option<T>` (`let _ = arr.pop();`).
//
// Run with:
//   cargo test --test discard_pop_no_default

use std::path::Path;
use std::process::Command;

#[test]
fn discarded_pop_needs_no_default() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/discard_pop_no_default.br");
    let dir = Path::new("tests/cases/discard_pop_no_default_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .expect("failed to invoke boring");
    assert!(emit.status.success(), "emit failed:\n{}", String::from_utf8_lossy(&emit.stderr));
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();
    assert!(!generated.contains("pop().unwrap_or_default()"), "{generated}");

    std::fs::write(dir.join("src/main.rs"), &generated).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"discard_pop_no_default_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .expect("failed to invoke cargo");
    assert!(
        run.status.success(),
        "generated Rust failed to build/run:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim_end(), "1");
    let _ = std::fs::remove_dir_all(dir);
}
