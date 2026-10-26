// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression: a dict field of the struct held by an 'actor field (memo.memory.placed[k]) must be read/written through the lock (get/insert), not indexed as an array nor accessed on the Arc<Mutex<..>> handle.
//
// Run with:
//   cargo test --test actor_field_dict_lookup

use std::path::Path;
use std::process::Command;

#[test]
fn actor_field_dict_lookup() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/actor_field_dict_lookup.br");
    let dir = Path::new("tests/cases/actor_field_dict_lookup_rust");
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
        generated.contains("memo.memory.lock().unwrap().placed.insert("),
        "insert must go through the lock:\n{}",
        generated
    );
    assert!(generated.contains("__g.placed.get(__k).cloned()"), "lookup must use get:\n{}", generated);

    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"actor_field_dict_lookup_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
    assert_eq!(actual.trim_end(), "1 2\nnone");

    let _ = std::fs::remove_dir_all(dir);
}
