// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: an open-ended range (`M..`) whose start `M` is a compound
// expression — `i+1`, not a bare variable or literal — used to crash the
// whole `boring build` process with a real Rust `panic!`, not a clean
// compile error, for BOTH a string read (`s[i+1..]`) and an array read
// (`arr[i+1..]`).
//
// Root cause: the parser's range-operator check lived inline inside
// `parse_unary` (`src/parser/parse_expr.rs`), attached right after a single
// unary term — one layer *below* `parse_mul`/`parse_add`, not above them.
// So `i+1..` never got the chance to combine `i` and `1` via `+` before the
// range operator grabbed the most-recently-parsed unary operand (`1`):
// `i+1..` parsed as `i + (1..)` — a `BinOp::Add` whose right operand is a
// bare `ExprKind::SliceRange`, reachable directly by `emit_expr` outside of
// `emit_expr_index`'s dedicated slice-range handling, hitting its "SliceRange
// cannot appear outside an index expression" panic arm. Parenthesizing
// (`s[(i+1)..]`) was the only workaround, since parentheses force `i+1` to be
// parsed as one atomic operand before the range logic ever saw it.
//
// Fixed by moving the range-operator check out of `parse_unary` into a new
// `parse_range`, inserted between `parse_shift` and `parse_add` in the
// precedence chain, whose start/end are parsed via `parse_add` (full
// unary/mul/add chain) instead of a single unary term — matching
// `spec/grammar.bnf`'s `slice_range ::= expr ".."` (a full `expr`, not a
// restricted-precedence term).
//
// This test emits the Boring functions via `--emit-rust` (raw Rust source,
// no Boring-generated Cargo project — same technique as
// `tests/array_range_index_assign.rs`), then does a real `cargo build` +
// runs the compiled binary and checks its stdout, and additionally asserts
// that the compound-start and parenthesized-start forms produce byte-
// identical generated Rust (proving the fix, not just the absence of a
// panic).
//
// Run with:
//   cargo test --test open_ended_range_compound_start

use std::path::Path;
use std::process::Command;

#[test]
fn open_ended_range_with_compound_start_does_not_panic() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/open_ended_range_compound_start.br");
    let dir = Path::new("tests/cases/open_ended_range_compound_start_rust");
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
    // The compound-start form (`i+1..`) must generate the exact same slice
    // expression as its parenthesized twin (`(i+1)..`) — proving `i+1` is
    // parsed as one atomic operand feeding the range, not split apart by it.
    assert!(
        generated.contains("s.chars().skip(((i + 1)) as usize).collect::<String>()"),
        "expected the string slice's start to compile to a single `(i + 1)` \
         skip count, but it didn't — generated source:\n{}",
        generated
    );
    assert!(
        generated.contains("arr[((i + 1)) as usize..].to_vec()"),
        "expected the array slice's start to compile to a single `(i + 1)` \
         range bound, but it didn't — generated source:\n{}",
        generated
    );

    // ── Real `cargo build` + run, checking actual slice contents ────────────
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"open_ended_range_compound_start_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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

    let exe_name = format!("open_ended_range_compound_start_check{}", std::env::consts::EXE_SUFFIX);
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

    // string_slice(): "hello"[1..] == "ello"
    // string_slice() == string_slice_parenthesized(): true
    // array_slice(): [10,20,30,40][1..] == [20,30,40]
    // array_slice() == array_slice_parenthesized(): true
    let expected = "ello\ntrue\n20\n30\n40\ntrue\n";
    assert_eq!(
        stdout, expected,
        "unexpected output after open-ended range read with a compound start — got:\n{}",
        stdout
    );

    let _ = std::fs::remove_dir_all(dir);
}
