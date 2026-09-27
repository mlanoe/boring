// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: string index-*assignment* (`s[i] = "x"`, `s[a..<b] =
// "..."`) used to compile cleanly under a plain `boring build` (no GPU
// target at all -- the general/std pipeline, not the Metal/CUDA/ROCm
// host-code emitters fixed separately for the *read* path in commit
// 79b3193) and then fail `cargo build` with a confusing error, because
// `emit_expr.rs::emit_expr_assign`'s array-index LHS codegen (just below the
// dict-subscript-assignment handling) had no string-vs-array distinction and
// happily emitted `Vec`-style `s[(i) as usize] = Arc::<str>::from("H")`
// codegen -- neither `Arc<str>`/`Rc<str>` (boring's own `string`
// representation) nor plain `str`/`String` implement `IndexMut`, so there is
// no way to assign a single character (or range) into a Rust string in
// place at all.
//
// docs/book.md already documents the design intent plainly: "strings are
// immutable; every method returns a new value" -- so rather than teaching
// the transpiler to silently rebuild the whole string (surprising codegen
// for something that reads like an O(1) mutation), `emit_expr_assign` now
// rejects both shapes with a clear `push_error` diagnostic BEFORE emitting
// any Rust at all, which makes `boring build` itself fail with a pointed
// message -- see `main.rs`'s `if !out.errors.is_empty() { ...
// process::exit(1) }` checks, which run before cargo is ever invoked.
//
// `string_index_assign_ok_case_still_builds_and_runs` is the accompanying
// control: plain (non-string) index-assignment into an array and a dict
// must keep working exactly as before -- the new check
// (`index_receiver_is_string`) only recognizes an actual `string` receiver.
//
// Run with:
//   cargo test --test string_index_assign_build_fails

use std::path::Path;
use std::process::Command;

fn run_boring_build(case_br: &Path) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e))
}

#[test]
fn single_index_string_assignment_fails_boring_build() {
    let case_br = Path::new("tests/cases/error_string_index_assign.br");
    let emit = run_boring_build(case_br);

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to reject `s[0] = \"H\"` on a \
         string local, but it exited successfully and emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    let expected = "cannot assign into a `string` via index";
    assert!(
        stderr.contains(expected),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        expected, stderr
    );
    assert!(
        !stderr.contains("via index range"),
        "single-index target must not be reported as a range -- actual stderr:\n{}",
        stderr
    );
}

#[test]
fn range_index_string_assignment_fails_boring_build() {
    let case_br = Path::new("tests/cases/error_string_index_assign_range.br");
    let emit = run_boring_build(case_br);

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to reject `s[0..<2] = \"HE\"` on \
         a string local, but it exited successfully and emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    let expected = "cannot assign into a `string` via index range";
    assert!(
        stderr.contains(expected),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        expected, stderr
    );
}

#[test]
fn string_index_assign_ok_case_still_builds_and_runs() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/string_index_assign_ok.br");
    let dir = Path::new("tests/cases/string_index_assign_ok_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "boring build --emit-rust failed for plain array/dict index-assignment:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"string_index_assign_ok_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
        "expected the generated Rust for plain array/dict index-assignment to \
         compile and run, but it failed:\n--- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&run.stderr),
        generated,
    );

    let stdout = String::from_utf8_lossy(&run.stdout);
    assert_eq!(
        stdout.trim(),
        "99\n42",
        "unexpected program output:\n{}",
        stdout
    );

    let _ = std::fs::remove_dir_all(dir);
}
