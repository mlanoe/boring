// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `emit_expr_index`'s char-safe string range-slice codegen
// (the `is_str` branch under `ExprKind::SliceRange` in emit_expr.rs) emitted
// `Arc::from({collected}.as_str())` / `Rc::from(...)` with no `::<str>`
// turbofish, unlike the sibling single-index string case (`s[i]`, just below
// in the same function) which already had it. `Arc<T, A = Global>` carries a
// generic allocator parameter; a plain `let x = s[a..<b]` still compiles
// because rustc can infer it from later usage, but when the slice expression
// is one leaf of a string concatenation chain (`a + b + c`) it gets flattened
// by `collect_string_parts`/`emit_expr_raw_string` (emit_stmt.rs) directly
// into a single `format!(...)` call's argument list — there rustc has
// nothing to infer the allocator from and fails with E0283 ("type
// annotations needed ... cannot satisfy `_: Allocator`").
//
// Fixed by adding the missing `::<str>` turbofish so the range-slice branch
// matches the already-correct single-index branch exactly.
//
// tests/cases/string_slice_concat_build.br exercises the exact shape that
// exposed the bug: `s[..<i] + "H" + s[(i+1)..]`. This test:
//   1. Pins the codegen shape directly (string check for the turbofish).
//   2. Does a real `cargo build` against the generated Rust — the bug's
//      actual failure mode is a `cargo build` error, not anything `boring
//      build` itself detects (it exits 0 either way).
//
// Same technique as tests/dict_chained_index_assign.rs.
//
// Run with:
//   cargo test --test string_slice_concat_turbofish

use std::path::Path;
use std::process::Command;

#[test]
fn string_range_slice_in_concat_chain_has_turbofish_and_compiles() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/string_slice_concat_build.br");
    let dir = Path::new("tests/cases/string_slice_concat_build_rust");
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

    // ── Codegen-shape assertion (exact string check, no compiler needed) ──
    assert!(
        generated.contains("::<str>::from("),
        "expected the range-slice-in-concat codegen to include the `::<str>` \
         turbofish on `Arc::from`/`Rc::from` (needed because the slice is \
         flattened into a `format!(...)` argument list where rustc can't \
         infer the allocator type parameter), but it didn't — generated \
         source:\n{}",
        generated
    );

    // ── Real `cargo build` — this is the bug's actual failure mode (E0283) ──
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"string_slice_concat_build_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("failed to write Cargo.toml");

    let build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {}", e));

    assert!(
        build.status.success(),
        "expected the generated Rust to compile, but `cargo build` failed:\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&build.stderr),
        generated,
    );

    let _ = std::fs::remove_dir_all(dir);
}
