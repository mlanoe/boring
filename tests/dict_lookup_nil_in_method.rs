// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression: `let hit = dict[key]` followed by `hit != nil` (and `dict[key] != nil`) in a struct method keeps the lookup an Option.
//
// Run with:
//   cargo test --test dict_lookup_nil_in_method

use std::path::Path;
use std::process::Command;

#[test]
fn dict_lookup_nil_in_method() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/dict_lookup_nil_in_method.br");
    let dir = Path::new("tests/cases/dict_lookup_nil_in_method_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "boring build --emit-rust failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    assert!(
        generated.contains("let hit: Option<Size> = self.rows.get(&*key).cloned();"),
        "lookup must stay an Option:\n{}",
        generated
    );

    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"dict_lookup_nil_in_method_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
        "generated Rust failed to build/run:\n--- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&run.stderr),
        generated,
    );
    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(actual.trim_end(), "true false\nfalse");

    let _ = std::fs::remove_dir_all(dir);
}
