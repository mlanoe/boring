// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `let existing = dict[key] else default` (a dict-index
// nil-coalescing binding) with NO explicit type annotation was left out of
// `dict_vars` tracking in `emit_let.rs`'s `track_let_metadata`. The generated
// Rust for that shape (`outer.get(&key).cloned().unwrap_or_else(...)`, from
// `emit_expr.rs`'s `ExprKind::Else` handling) doesn't start with `HashMap::`
// or contain `.collect::<HashMap`, the only two string-based checks that
// populated `dict_vars` for an *untyped* local. That left a later
// `for k, val in existing:` unable to prove `existing` was dict-shaped, so
// `emit_loop.rs`'s `iterable_yields_tuples` fell through to `false` and the
// two-loop-var auto-enumerate rewrite kicked in, treating `existing` as a
// `Vec` (`existing.iter().cloned().enumerate().map(|(i, v)| (i as isize, v))`)
// — wrong pair shape (index, value) instead of (key, value), and fails to
// compile against the real `HashMap<Arc<str>, Arc<str>>`.
//
// Fixed by teaching `track_let_metadata` to recognize the same
// `Else(Index(dict_obj, _), _)` AST shape (via `expr_is_dict`) that
// `emit_expr.rs` already uses to *emit* the `.get().cloned().unwrap_or_else()`
// code, and use it to also populate `dict_vars` — matching what an explicit
// `{K=V}` type annotation already did via `matches!(&s.ty, Some(Type::Dict(..)))`.
//
// Fixture: `tests/cases/dict_index_else_untyped_for.br` declares both shapes
// in one file so a future change can't silently regress either — see that
// file's own header comment.
//
// This test emits the Boring functions via `--emit-rust` (raw Rust source,
// no Boring-generated Cargo project — same technique as
// `tests/for_loop_mut_tuple.rs`) and:
//   1. String checks on the generated source pin the exact codegen shape for
//      both functions (catches the bug directly, no compiler needed).
//   2. A real `cargo build` (no external stub needed — `HashMap` is real
//      std) catches the bug's actual failure mode too: the array
//      auto-enumerate rewrite over a `HashMap` doesn't type-check as
//      `(k, val)` pairs.
//
// Run with:
//   cargo test --test dict_index_else_untyped_for

use std::path::Path;
use std::process::Command;

#[test]
fn untyped_dict_index_else_binding_keeps_dict_style_for_loop() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/dict_index_else_untyped_for.br");
    let dir = Path::new("tests/cases/dict_index_else_untyped_for_rust");
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

    // ── Codegen-shape assertions (exact string checks, no compiler needed) ──

    let untyped_start = generated
        .find("fn demo_untyped")
        .expect("demo_untyped not found in generated source");
    let untyped_end = generated[untyped_start..]
        .find("fn demo_typed")
        .map(|off| untyped_start + off)
        .unwrap_or(generated.len());
    let untyped_body = &generated[untyped_start..untyped_end];

    assert!(
        untyped_body.contains("existing.into_iter()"),
        "expected the untyped `let existing = outer[key] else {{=}}` binding's \
         `for` loop to transpile to plain dict `.into_iter()`, but it didn't — \
         generated function:\n{}",
        untyped_body
    );
    assert!(
        !untyped_body.contains("enumerate"),
        "the dict-shaped `existing` loop must never get the array \
         auto-enumerate rewrite (a HashMap has no `.iter()` yielding \
         (usize, Item) pairs) — generated function:\n{}",
        untyped_body
    );

    let typed_body = &generated[generated
        .find("fn demo_typed")
        .expect("demo_typed not found in generated source")..];
    assert!(
        typed_body.contains("existing.into_iter()"),
        "expected the explicitly-typed `let {{string=string}} existing = ...` \
         control case to also keep the plain dict `.into_iter()` — generated \
         function:\n{}",
        typed_body
    );

    // ── Real `cargo build` against a real `std::collections::HashMap` ──────
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"dict_index_else_untyped_for_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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

    // Clean up the generated build dir so repeated runs don't accumulate disk
    // usage (target/ dirs in particular).
    let _ = std::fs::remove_dir_all(dir);
}
