// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `let x = struct_val.dict_field.clone()` — a plain
// `.clone()` method call on a dict-typed field access, with NO explicit type
// annotation on `x` — was left out of `dict_vars` tracking in
// `emit_let.rs`'s `track_let_metadata`. The `dict_vars` population there only
// recognized an explicit `Type::Dict` annotation, a `HashMap::`-prefixed RHS,
// a `.collect::<HashMap` RHS, or the `dict[key] else default` heuristic shape
// — a bare `.clone()` call on a dict-typed field never matched any of those
// string/AST checks, since the emitted Rust for that shape starts with the
// receiver's own text (e.g. `h.table.clone()`), not one of the literal
// prefixes.
//
// That left a later `for k, v in x:` (2-variable destructuring) unable to
// prove `x` was dict-shaped, so `emit_loop.rs`'s tuple-yielding check fell
// through to `false` and the array-style auto-enumerate rewrite kicked in —
// `x.iter().cloned().enumerate().map(|(i, v)| (i as isize, v))` — which
// doesn't type-check against a real `HashMap<Arc<str>, Arc<str>>` (E0271:
// expected an iterator yielding `&_`, got `(&Arc<str>, &Arc<str>)`; E0599:
// no `enumerate` on `Cloned<...>`).
//
// Fixed by teaching `track_let_metadata` to recognize the
// `MethodCall(recv, "clone", _)` AST shape and resolve `recv`'s own declared
// type via `resolve_expr_declared_type` (the same helper the neighboring
// `is_dict_index_else` heuristic already calls) — if that resolves to
// `Type::Dict(..)`, the local is registered in `dict_vars` just like an
// explicit `{K=V}` annotation already did.
//
// Fixture: `tests/cases/dict_field_clone_untyped_for.br` declares both
// shapes (untyped local vs. the documented explicit-annotation workaround)
// in one file so a future change can't silently regress either — see that
// file's own header comment.
//
// This test emits the Boring functions via `--emit-rust` (raw Rust source,
// no Boring-generated Cargo project — same technique as
// `tests/dict_index_else_untyped_for.rs`) and:
//   1. String checks on the generated source pin the exact codegen shape for
//      both functions (catches the bug directly, no compiler needed).
//   2. A real `cargo build` (no external stub needed — `HashMap` is real
//      std) catches the bug's actual failure mode too: the array
//      auto-enumerate rewrite over a `HashMap` doesn't type-check as
//      `(k, v)` pairs.
//
// Run with:
//   cargo test --test dict_field_clone_untyped_for

use std::path::Path;
use std::process::Command;

#[test]
fn untyped_dict_field_clone_binding_keeps_dict_style_for_loop() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/dict_field_clone_untyped_for.br");
    let dir = Path::new("tests/cases/dict_field_clone_untyped_for_rust");
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
        untyped_body.contains("instance_types.into_iter()")
            || untyped_body.contains("instance_types.iter()"),
        "expected the untyped `let instance_types = h.table.clone()` binding's \
         `for` loop to transpile to plain dict iteration, but it didn't — \
         generated function:\n{}",
        untyped_body
    );
    assert!(
        !untyped_body.contains("enumerate"),
        "the dict-shaped `instance_types` loop must never get the array \
         auto-enumerate rewrite (a HashMap has no `.iter()` yielding \
         (usize, Item) pairs) — generated function:\n{}",
        untyped_body
    );

    let typed_start = generated
        .find("fn demo_typed")
        .expect("demo_typed not found in generated source");
    let typed_end = generated[typed_start..]
        .find("fn main")
        .map(|off| typed_start + off)
        .unwrap_or(generated.len());
    let typed_body = &generated[typed_start..typed_end];
    assert!(
        typed_body.contains("instance_types.into_iter()")
            || typed_body.contains("instance_types.iter()"),
        "expected the explicitly-typed `let {{string=string}} instance_types = ...` \
         control case to also keep plain dict iteration — generated \
         function:\n{}",
        typed_body
    );
    assert!(
        !typed_body.contains("enumerate"),
        "the explicitly-typed control case must never get the array \
         auto-enumerate rewrite either — generated function:\n{}",
        typed_body
    );

    // ── Real `cargo build` against a real `std::collections::HashMap` ──────
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"dict_field_clone_untyped_for_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
