// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `match v:` over an owned enum local/param, where a binding arm
// moves the payload out of `v` and an arm body mentions `v` again (`return v`),
// used to emit `match v { .. }` and fail rustc with E0382 (borrow of partially
// moved value). The subject is now cloned when an arm mentions it.
//
// Run with:
//   cargo test --test match_subject_reused_in_arm

use std::path::Path;
use std::process::Command;

#[test]
fn match_subject_reused_in_arm_compiles() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/match_subject_reused_in_arm.br");
    let dir = Path::new("tests/cases/match_subject_reused_in_arm_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    // ── boring build --emit-rust must succeed (checker already permits the
    // `def` call through the dict's `mut`-qualified value type) ─────────────
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

    // ── And the generated Rust must actually build and run — this is the ────
    // part that used to silently drop the mutation (no build/runtime error
    // of any kind, just the wrong printed value).
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"match_subject_reused_in_arm_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
        "1",
        "unexpected stdout — the dict-value mutations were not observed \
         (this is exactly the silent-correctness bug this test pins: no \
         build or runtime error, just the wrong value)"
    );

    // Clean up the generated build dir so repeated runs don't accumulate
    // disk usage (target/ dirs in particular).
    let _ = std::fs::remove_dir_all(dir);
}
