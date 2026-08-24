// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for `boring build --emit-rust <file>` (file passed explicitly)
// skipping the project's own `boring.toml` `[external_types]`/`[derives]`/
// `[external_fns]` supplement -- a merge `boring build --emit-rust` (no file arg,
// relying on `boring.toml`'s `main` field) already applied. The two invocations
// must produce the *same* transpile config for the same source file; before the
// fix they diverged whenever the project declared `[external_types]`.
//
// `tests/cases/ext_types_toml_project/` declares `ExtOpaque` (see
// `tests/cases/fixtures/ext_tuple/src/lib.rs`) via `[external_types] tuple_structs =
// ["ExtOpaque"]` -- `ExtOpaque` is deliberately NOT in the compiler's built-in
// `Transpiler::KNOWN_EXTERNAL_TUPLE_STRUCTS` list, so only that `boring.toml` entry
// makes bare `ExtOpaque(5)` construction transpile as a literal tuple-struct call
// instead of the nonexistent `ExtOpaque::new(5)` (which doesn't compile, E0599).
//
// Run with:
//   cargo test --test emit_rust_explicit_path

use std::path::Path;
use std::process::Command;

fn run_emit_rust(project_dir: &Path, file_arg: Option<&str>) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    let mut cmd = Command::new(bin);
    cmd.current_dir(project_dir).arg("build").arg("--emit-rust");
    if let Some(file) = file_arg {
        cmd.arg(file);
    }
    cmd.output().unwrap_or_else(|e| panic!("failed to invoke boring: {}", e))
}

/// `boring build --emit-rust src/main.br` (explicit file arg) must apply the same
/// `[external_types]` supplement as `boring build --emit-rust` (no file arg) --
/// both must emit the literal tuple-struct form `ExtOpaque(5)`, never the invalid
/// `ExtOpaque::new(5)`.
#[test]
fn explicit_file_arg_applies_boring_toml_external_types() {
    let project_dir = Path::new("tests/cases/ext_types_toml_project");

    let explicit = run_emit_rust(project_dir, Some("src/main.br"));
    assert!(
        explicit.status.success(),
        "boring build --emit-rust src/main.br failed:\n{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    let explicit_code = String::from_utf8_lossy(&explicit.stdout).into_owned();

    let implicit = run_emit_rust(project_dir, None);
    assert!(
        implicit.status.success(),
        "boring build --emit-rust (no file arg) failed:\n{}",
        String::from_utf8_lossy(&implicit.stderr)
    );
    let implicit_code = String::from_utf8_lossy(&implicit.stdout).into_owned();

    assert!(
        !explicit_code.contains("ExtOpaque::new("),
        "explicit-path --emit-rust wrongly rewrote `ExtOpaque(5)` to `ExtOpaque::new(...)` \
         (boring.toml's [external_types] supplement was not applied):\n{}",
        explicit_code
    );
    assert!(
        explicit_code.contains("ExtOpaque(5)"),
        "expected explicit-path --emit-rust to emit the literal tuple-struct form \
         `ExtOpaque(5)`, got:\n{}",
        explicit_code
    );

    assert_eq!(
        explicit_code, implicit_code,
        "boring build --emit-rust <file> and boring build --emit-rust (no file arg) \
         produced different Rust for the same source file"
    );
}
