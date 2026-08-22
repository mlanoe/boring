// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a `for (a, b) in <iterable>:` codegen bug: the
// transpiler's two-loop-var auto-enumerate detection (`emit_loop.rs`'s
// `iterable_yields_tuples`) only recognized a dict or an array-of-tuples as
// already tuple-shaped. Any *other* type fell through to "not proven
// tuple-shaped" → "auto-enumerate" → an injected `.enumerate().map(|(i, v)|
// (i as isize, v))`.
//
// That's wrong for an external generic type whose sole type argument is
// itself a tuple — e.g. a Bevy-ECS-style `Query<(mut Transform&,
// ScratchSprite&)>` parameter, where the "tuple" is a component-destructuring
// pattern, not an index/value pair. `bevy::ecs::query::Query<T>` has no
// `.iter()` returning `(usize, Item)` pairs, so the auto-enumerate rewrite
// produced code that doesn't compile against the real type. It should have
// stayed a plain `.into_iter()` — exactly like the dict/array-of-tuples cases
// already correctly do.
//
// Root cause: `iterable_yields_tuples` matched only `Type::Dict(_, _)` and
// `Type::Array(Type::Tuple(_))` by hand. It's fixed to reuse
// `Type::tuple_slot_mut_flags` (already used a few lines below to decide
// which destructured loop variable needs a Rust `mut` prefix) as the general
// "is this type's item shape a tuple" probe — it already unwraps
// `Array`/`Optional`/`Mut`/`Qualified`, and a one-arg `Generic` type, so
// `Holder<(mut Point&, Sprite&)>` (this test's bevy-free stand-in for
// `Query<...>`) is now correctly recognized as tuple-shaped too.
//
// Fixture: `tests/cases/for_loop_mut_tuple.br` declares both loop shapes in
// one file so a future fix to one can't silently regress the other:
//   - `sync_from_holder`: `for point, sprite in query:` over a
//     `Holder<(mut Point&, Sprite&)>` param — must stay `.into_iter()`.
//   - `sync_from_array`: `for i, v in pts:` over a real Boring `[int]` — must
//     keep the auto-enumerate rewrite that motivated the feature.
//
// This test emits the Boring functions via `--emit-rust` (raw Rust source,
// no Boring-generated Cargo project — same technique as
// `tests/external_enum_variant.rs`) and prepends a hand-written stand-in for
// the "external" `Holder<T>` generic (a plain `Vec<T>`-backed `IntoIterator`)
// into a single-file binary crate, so:
//   1. String checks on the generated source pin the exact codegen shape for
//      both loops (catches the bug directly, no compiler needed).
//   2. A real `cargo build` against the stand-in `Holder` catches the bug's
//      actual failure mode too: `Holder<T>` (like `Query<T>`) has no
//      `.iter().cloned().enumerate()...` — that rewrite doesn't type-check.
//
// Run with:
//   cargo test --test for_loop_mut_tuple

use std::path::Path;
use std::process::Command;

/// Hand-written stand-in for an external generic ECS-query-shaped container
/// (e.g. `bevy::ecs::query::Query<T>`) — Boring never parses this
/// declaration, exactly mirroring how a real `use bevy.prelude.*` import
/// behaves (the type just exists at Rust compile time, with no Boring-side
/// registration). Only `IntoIterator` is implemented, on purpose: a
/// `.iter().cloned().enumerate()...`-shaped rewrite must NOT type-check
/// against it.
const HOLDER_STUB: &str = r#"
#[allow(dead_code)]
pub struct Holder<T> {
    pub items: Vec<T>,
}

impl<T> IntoIterator for Holder<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}
"#;

#[test]
fn query_shaped_tuple_stays_into_iter_array_keeps_auto_enumerate() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/for_loop_mut_tuple.br");
    let dir = Path::new("tests/cases/for_loop_mut_tuple_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

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

    // ── Codegen-shape assertions (exact string checks, no compiler needed) ──

    // The `Holder<(mut Point&, Sprite&)>` loop must stay a plain `.into_iter()`.
    assert!(
        generated.contains("for (mut point, sprite) in query.into_iter() {"),
        "expected the Query-shaped `Holder` loop to transpile to a plain \
         `.into_iter()`, but it didn't — generated source:\n{}",
        generated
    );

    // Isolate `sync_from_holder`'s body and confirm no `.enumerate()` rewrite
    // snuck in anywhere within it (belt-and-suspenders on top of the exact
    // string check above).
    let holder_fn_start = generated.find("fn sync_from_holder").expect("sync_from_holder not found in generated source");
    let holder_fn_body = &generated[holder_fn_start..];
    let holder_fn_end = holder_fn_body.find("fn sync_from_array").unwrap_or(holder_fn_body.len());
    let holder_fn_body = &holder_fn_body[..holder_fn_end];
    assert!(
        !holder_fn_body.contains("enumerate"),
        "the Query-shaped `Holder` loop must never get the array auto-enumerate \
         rewrite (Query has no `.iter()` yielding (usize, Item) pairs) — \
         generated function:\n{}",
        holder_fn_body
    );

    // The real `[int]` loop must keep the auto-enumerate rewrite this feature
    // exists for — regression coverage for the *other* direction, so a future
    // fix to the Query case can't silently break plain-array indexing.
    assert!(
        generated.contains("for (i, v) in pts.iter().cloned().enumerate().map(|(i, v)| (i as isize, v)) {"),
        "expected the plain `[int]` loop to keep the `.enumerate()` \
         auto-enumerate rewrite, but it didn't — generated source:\n{}",
        generated
    );

    // ── Real `cargo build` against the stand-in `Holder<T>` ──────────────────
    // Combine into a single file so the stub is in scope for the generated
    // code without needing any cross-module `use`/path wiring.
    let combined = format!("{}\n{}\n", HOLDER_STUB, generated);
    std::fs::write(dir.join("src/main.rs"), &combined).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"for_loop_mut_tuple_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    ).expect("failed to write Cargo.toml");

    let build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {}", e));

    assert!(
        build.status.success(),
        "expected the `Holder<(mut Point&, Sprite&)>` loop to transpile to valid \
         Rust against a real (stand-in) external generic type, but `cargo build` \
         failed:\n--- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&build.stderr),
        combined,
    );

    // Clean up the generated build dir so repeated runs don't accumulate disk
    // usage (target/ dirs in particular).
    let _ = std::fs::remove_dir_all(dir);
}
