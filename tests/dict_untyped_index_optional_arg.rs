// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression: untyped `let x = dict[key]` passed to a `V?` parameter, returned as `V?`,
// or assigned to a `V?` slot stays an Option lookup and does not panic when the key is absent.
//
// Run with:
//   cargo test --test dict_untyped_index_optional_arg

use std::path::Path;
use std::process::Command;

#[test]
fn dict_untyped_index_optional_arg() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/dict_untyped_index_optional_arg.br");
    let dir = Path::new("tests/cases/dict_untyped_index_optional_arg_rust");
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
        generated.contains("let h1: Option<EditHistory> = rt.histories.get(&1).cloned();"),
        "lookup passed to fn param V? must stay an Option:\n{}",
        generated
    );
    assert!(
        generated.contains("let m1: Option<EditHistory> = rt.histories.get(&1).cloned();"),
        "lookup passed to method param V? must stay an Option:\n{}",
        generated
    );
    assert!(
        generated.contains("let priorHistory: Option<EditHistory> = rt.histories.get(&handle).cloned();"),
        "lookup returned as V? or assigned to V? must stay an Option:\n{}",
        generated
    );
    assert!(
        !generated.contains("rt.histories.get(&1).cloned().expect(\"dict key not found\")"),
        "lookup must not panic on missing key:\n{}",
        generated
    );

    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"dict_untyped_index_optional_arg_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
    let expected = std::fs::read_to_string("tests/cases/dict_untyped_index_optional_arg.expected")
        .expect("failed to read expected")
        .replace("\r\n", "\n");
    assert_eq!(actual.trim_end(), expected.trim_end());

    let _ = std::fs::remove_dir_all(dir);
}
