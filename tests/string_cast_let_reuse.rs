// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `let msg = "hello" as string` — an untyped local bound to
// an explicit `as string` cast on a string literal — was left out of
// `string_vars` tracking in `emit_let.rs`'s `track_let_metadata`.
//
// That block's `is_immutable_string_lit` check only matched a *bare* literal
// initializer (`ExprKind::Str(_) | ExprKind::StringInterp(_)`). An explicit
// `as string` cast wraps the same literal in `ExprKind::Cast(..)`, which
// didn't match, so the binding was never added to `string_vars` at all —
// unlike the bare-literal form (`let msg = "hello"`, no cast), which already
// gets tracked and clone-inserted correctly.
//
// Left untracked, neither call site of `tok.encode(msg)` got a `.clone()`
// inserted — `error[E0382]: use of moved value: msg` at `cargo build`,
// even though the identical pattern without the `as string` cast already
// compiled cleanly. This is the same class of bug as
// `tests/method_call_string_return_reuse.rs` (a different untyped-`let`
// initializer shape falling through the same tracking logic uncovered), but
// a distinct trigger — that fix didn't cover this one.
//
// Fixed by extending `is_immutable_string_lit` in `emit_let.rs` to also match
// `ExprKind::Cast(_, dst_ty)` when `dst_ty` is a string type.
//
// This test emits the Boring function via `--emit-rust` (raw Rust source, no
// Boring-generated Cargo project — same technique as
// `tests/method_call_string_return_reuse.rs`) and:
//   1. A string check on the generated source pins the exact codegen shape
//      (catches the bug directly, no compiler needed).
//   2. A real `cargo build` (no external stub needed) catches the bug's
//      actual failure mode too: `cargo build` on the generated project must
//      succeed, not fail with E0382.
//
// Run with:
//   cargo test --test string_cast_let_reuse

use std::path::Path;
use std::process::Command;

#[test]
fn string_cast_let_is_cloned_on_later_reuse() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/string_cast_let_reuse.br");
    let dir = Path::new("tests/cases/string_cast_let_reuse_rust");
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

    let main_start = generated
        .find("fn main")
        .expect("fn main not found in generated source");
    let main_body = &generated[main_start..];

    assert!(
        main_body.contains("tok.encode(msg.clone())"),
        "expected the `as string`-cast `msg` binding to be `.clone()`d at \
         (at least) the first call site, since it is reused at a second one \
         afterward — generated function:\n{}",
        main_body
    );

    // ── Real `cargo build` — catches the bug's actual failure mode (E0382) ──
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"string_cast_let_reuse_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
