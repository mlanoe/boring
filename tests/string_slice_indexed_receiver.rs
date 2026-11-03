// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `String.slice(from, to)` on an indexed receiver of an untyped
// string-array literal (`let months = ["January", "February"]; months[m - 1].slice(0, 3)`,
// bare, bound to a local, or inside a string interpolation) was emitted as an *array*
// slice (`[a..b].iter().cloned().collect::<Vec<_>>()`) because `resolve_expr_type` had no
// case for a string literal, so `["a", "b"]` resolved to no type at all. `cargo build`
// failed with E0599 (no method `iter` on `str`) while `boring run` printed `Jan`.
//
// Run with:
//   cargo test --test string_slice_indexed_receiver

use std::path::Path;
use std::process::Command;

#[test]
fn slice_on_indexed_string_array_element_is_a_string_slice() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/string_slice_indexed_receiver.br");
    let dir = Path::new("tests/cases/string_slice_indexed_receiver_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let interp = Command::new(bin).arg("run").arg(case_br).output().expect("boring run");
    assert!(interp.status.success(), "boring run failed:\n{}", String::from_utf8_lossy(&interp.stderr));
    let expected = String::from_utf8_lossy(&interp.stdout).into_owned();
    assert_eq!(expected, "Jan\nJan\nJan\nJan\nAlp\n");

    let emit = Command::new(bin).arg("build").arg(case_br).arg("--emit-rust").output().expect("boring build");
    assert!(emit.status.success(), "boring build --emit-rust failed:\n{}", String::from_utf8_lossy(&emit.stderr));
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();
    assert!(
        !generated.contains(") as usize].iter().cloned().collect"),
        "string slice emitted as an array slice:\n{}",
        generated
    );

    std::fs::write(dir.join("src/main.rs"), &generated).expect("write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"string_slice_indexed_receiver_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("write Cargo.toml");

    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .expect("cargo run");
    assert!(
        run.status.success(),
        "generated Rust failed to build/run:\n{}\n--- source ---\n{}",
        String::from_utf8_lossy(&run.stderr),
        generated
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout), expected);

    let _ = std::fs::remove_dir_all(dir);
}
