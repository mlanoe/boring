// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for docs/self-field-loop-match-borrow-bug.md: a `for`/`match`
// directly over a bare struct field (implicit `self.field`, no explicit `self.`
// prefix) inside a `req`/`def` method generated Rust that tried to *consume* the
// field by value even though the method only holds a borrowed `&self`/`&mut self`.
//
// Root cause: the for-loop/match codegen paths (`emit_loop.rs`'s `emit_for` and
// `emit_match.rs`'s `emit_match`) already special-cased the *explicit* `self.field`
// spelling (`ExprKind::Field(Var("self"), _)`) to avoid moving out of the borrow —
// but a bare field reference parses as a plain `ExprKind::Var(name)`, syntactically
// indistinguishable from a real local variable at that level. Neither codegen path
// recognized that shape as "this resolves to an implicit `self.field` access", so it
// fell through to the same treatment as an ordinary owned local:
//   - `for s in items:`        -> `self.items.into_iter()`         (E0507)
//   - `match status:`          -> arms bound `self.status` by value (E0507-shaped)
//   - `for id, score in scores:` (a `{K=V}` field) -> mis-detected as "not already
//     tuple-shaped" (the same bare-Var blind spot also broke the dict-shape check
//     `iterable_yields_tuples` consults) and wrongly routed through the array
//     auto-enumerate rewrite (`.enumerate().map(|(i, v)| (i as isize, v))`), binding
//     the loop variable to the whole `(k, v)` pair instead of just the value.
//
// Fix: `resolve_self_field_type`/`bare_self_field_type` (src/transpiler/emit_methods.rs)
// resolve a bare `Var` back to its implicit field the same way `emit_expr`'s own
// `ExprKind::Var` arm already does when lowering to Rust, so both codegen paths (and
// `iterable_yields_tuples`'s dict-shape check) see through the bare spelling exactly
// like they already did for the explicit one.
//
// This test emits the Boring functions via `--emit-rust` (raw Rust source, no
// Boring-generated Cargo project — same technique as `tests/for_loop_mut_tuple.rs`)
// and asserts the exact codegen shape directly (no compiler needed), then also runs
// the fixture through a real `cargo build` for end-to-end confirmation — the same
// fixture is registered in tests/transpile.rs/tests/run.rs for the full
// `boring build` + `cargo run` + interpreter-parity path.
//
// Run with:
//   cargo test --test self_field_loop_match_borrow

use std::path::Path;
use std::process::Command;

#[test]
fn bare_self_field_for_loop_and_match_stay_borrow_safe() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/self_field_loop_match_borrow.br");

    let emit = Command::new(bin)
        .arg("build").arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "boring build --emit-rust failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    // Isolate each method's body so the assertions below can't accidentally match
    // codegen belonging to a different method.
    let body_of = |fn_sig: &str, next_marker: &str| -> String {
        let start = generated.find(fn_sig)
            .unwrap_or_else(|| panic!("`{}` not found in generated source:\n{}", fn_sig, generated));
        let rest = &generated[start..];
        let end = rest.find(next_marker).unwrap_or(rest.len());
        rest[..end].to_string()
    };

    // ── Repro 1: `for s in items:` over a bare `[string]` field ──────────────
    let first_match_body = body_of("fn first_match", "fn describe");
    assert!(
        first_match_body.contains("self.items.iter().cloned()"),
        "expected the bare-field `[string]` for-loop to borrow via \
         `.iter().cloned()`, not move out of `&self` — generated function:\n{}",
        first_match_body
    );
    assert!(
        !first_match_body.contains("self.items.into_iter()"),
        "the bare-field for-loop must never move `self.items` out of `&self` via \
         `.into_iter()` (E0507) — generated function:\n{}",
        first_match_body
    );

    // ── Match over a bare enum field ──────────────────────────────────────────
    let describe_body = body_of("fn describe", "fn highest");
    assert!(
        describe_body.contains("self.status.clone()"),
        "expected `match status:` over a borrowed enum field to clone the subject \
         before matching by value — generated function:\n{}",
        describe_body
    );

    // ── Repro 2: `for id, score in scores:` over a bare `{K=V}` field ─────────
    let highest_body = body_of("fn highest", "\n}\n");
    assert!(
        highest_body.contains("self.scores.clone().into_iter()"),
        "expected the bare-field `{{K=V}}` 2-var destructuring to iterate an owned \
         clone, not the borrowed field directly — generated function:\n{}",
        highest_body
    );
    assert!(
        !highest_body.contains("enumerate"),
        "a `{{K=V}}` dict field must never go through the array-style \
         `.enumerate().map(|(i, v)| (i as isize, v))` auto-enumerate rewrite — a \
         dict value never needs a reconstructed index — generated function:\n{}",
        highest_body
    );

    // ── Real `boring build` + `cargo build`, end to end ───────────────────────
    let dir = Path::new("tests/cases/self_field_loop_match_borrow_unit_rust");
    let build_emit = Command::new(bin)
        .arg("build").arg(case_br)
        .arg("--mode").arg("strict")
        .arg("--threading").arg("multi")
        .arg("--output-dir").arg(dir)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring build: {}", e));
    assert!(
        build_emit.status.success(),
        "boring build failed:\n{}",
        String::from_utf8_lossy(&build_emit.stderr)
    );

    let cargo_build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {}", e));
    assert!(
        cargo_build.status.success(),
        "expected the generated project to compile, but `cargo build` \
         failed:\n--- stderr ---\n{}",
        String::from_utf8_lossy(&cargo_build.stderr),
    );

    let _ = std::fs::remove_dir_all(dir);
}
