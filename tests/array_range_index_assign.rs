// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: range-index *assignment* into a non-string collection
// (`arr[a..<b] = [...]`) used to crash the whole `boring build` process with
// a real Rust `panic!`, not a clean compile error.
//
// Root cause: `emit_expr_assign`'s array-index LHS codegen block
// (`src/transpiler/emit_expr.rs`) called `self.emit_expr(idx_expr)`
// unconditionally on the assignment target's index expression. That's fine
// for a plain index (`arr[i] = v`), but when the target is a range
// (`arr[0..<2] = ...`) `idx_expr.kind` is a bare `ExprKind::SliceRange` —
// `emit_expr`'s top-level dispatch has no arm for that (it's only ever meant
// to be unwrapped from *inside* `emit_expr_index`'s own dedicated
// slice-range handling), so it hit that dispatch's panic arm ("SliceRange
// cannot appear outside an index expression").
//
// Fixed by adding a `SliceRange`-aware branch to `emit_expr_assign`, ahead of
// the generic array-index LHS codegen, that lowers to `Vec::splice(range,
// replacement)` — chosen (new design surface, nothing pre-existing in
// docs/book.md) to mirror Python's `list[a:b] = [...]` semantics: the
// replacement need not be the same length as the range (can grow or shrink
// the array), unlike Rust's own `IndexMut`-based range assignment (which
// requires the exact same length and panics otherwise).
//
// Fixture: `tests/cases/array_range_index_assign.br` covers three shapes in
// one file so a future change can't silently regress any of them —
// same-length replacement (`same_length`), a replacement longer than the
// range (`grow` — the shape only `.splice()`, not direct range-index
// assignment, can do), and a struct-field target with an inclusive range
// (`Holder::self_field_inclusive`, implicit `self`, `..=`).
//
// This test emits the Boring functions via `--emit-rust` (raw Rust source,
// no Boring-generated Cargo project — same technique as
// `tests/dict_chained_index_assign.rs`), then does a real `cargo build` +
// runs the compiled binary and checks its stdout against the three arrays'
// actual expected contents after the range assignment.
//
// Run with:
//   cargo test --test array_range_index_assign

use std::path::Path;
use std::process::Command;

#[test]
fn range_index_assignment_into_array_uses_splice_not_a_panic() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/array_range_index_assign.br");
    let dir = Path::new("tests/cases/array_range_index_assign_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "boring build --emit-rust failed (should never panic internally):\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    // ── Codegen-shape assertions (exact string checks, no compiler needed) ──

    assert!(
        generated.contains(".splice((0) as usize..(2) as usize, vec![9, 9])"),
        "expected the same-length exclusive-range assignment to lower to \
         `Vec::splice`, but it didn't — generated source:\n{}",
        generated
    );
    assert!(
        generated.contains(".splice((1) as usize..(3) as usize, vec![8, 8, 8, 8])"),
        "expected the growing exclusive-range assignment to lower to \
         `Vec::splice`, but it didn't — generated source:\n{}",
        generated
    );
    assert!(
        generated.contains(".splice((1) as usize..=(2) as usize, vec![9])"),
        "expected the struct-field inclusive-range assignment to lower to \
         `Vec::splice` with an inclusive Rust range, but it didn't — \
         generated source:\n{}",
        generated
    );

    // ── Real `cargo build` + run, checking actual array contents ────────────
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"array_range_index_assign_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("failed to write Cargo.toml");

    let manifest_path = dir.join("Cargo.toml");
    let build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(&manifest_path)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo build: {}", e));
    assert!(
        build.status.success(),
        "expected the generated Rust to compile, but `cargo build` failed:\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&build.stderr),
        generated,
    );

    let exe_name = format!("array_range_index_assign_check{}", std::env::consts::EXE_SUFFIX);
    let exe_path = dir.join("target/debug").join(&exe_name);
    let run = Command::new(&exe_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run compiled binary: {}", e));
    assert!(
        run.status.success(),
        "expected the generated program to exit successfully, but it failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");

    // same_length: [1,2,3,4] with [0..<2] = [9,9]  → [9,9,3,4]
    // grow:        [1,2,3,4,5] with [1..<3] = [8,8,8,8] → [1,8,8,8,8,4,5]
    // self_field_inclusive: [1,2,3,4,5] with [1..=2] = [9] → [1,9,4,5]
    let expected = "9\n9\n3\n4\n1\n8\n8\n8\n8\n4\n5\n1\n9\n4\n5\n";
    assert_eq!(
        stdout, expected,
        "unexpected array contents after range-index assignment — got:\n{}",
        stdout
    );

    let _ = std::fs::remove_dir_all(dir);
}
