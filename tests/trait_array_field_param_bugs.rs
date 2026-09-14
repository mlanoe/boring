// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression tests for three transpiler bugs around a struct field or function/method
// parameter typed as a trait-object array (`[Trait]`, docs/book.md "Traits as types" ->
// `Vec<Box<dyn Trait>>`). All three are pre-existing on `main`, independent of the
// trailing-array-block-sugar feature (which merely made `[dyn Trait]`-typed slots more
// common to reach for) -- found while implementing that feature, but the underlying bugs
// live in `src/transpiler/emit_struct.rs` (struct derive computation) and
// `src/transpiler/emit_loop.rs` (`emit_for`), unrelated to array-block desugaring itself.
//
// 1. A struct with a `[Trait]` field got an unconditional auto-default
//    `#[derive(Debug, Clone, PartialEq)]` -- `Box<dyn Trait>` (what `[Trait]` transpiles
//    to) implements neither `Clone` nor `PartialEq` (no object-safe blanket impl), so this
//    never compiled. Fix: `emit_struct.rs`'s existing `has_non_clone_field` check (already
//    carved out for atomic fields, e.g. `AtomicUsize` -- see its own `NON_CLONE_TYPES`
//    list) now also recognizes a `[Trait]`-typed field and skips `Clone`/`PartialEq` the
//    same way, in both `emit_struct.rs` and its mirrored bookkeeping in `mod.rs`'s
//    `pre_scan_struct_item` (`struct_derives_clone`, used by the Introspect feature).
//
// 2. `for x in some_struct_value.field:` where `field` is `[Trait]`-typed mis-emitted
//    `.iter().cloned()` (via the generic `is_borrowed_collection_field` array/set/dict
//    handling in `emit_for`) -- a hard compile error (E0277), since `Box<dyn Trait>` isn't
//    `Clone`. Fix: a new `is_trait_array_field` check, carved out ahead of the generic
//    array/set/dict branch, borrows instead (`&{iter}`), matching how a *local* `[Trait]`
//    variable was already correctly handled.
//
// 3. `for x in trait_array_param:` where `trait_array_param` is a `[Trait]`-typed
//    function/method parameter mis-emitted a doubled `&` borrow (`&shapes` where `shapes`
//    is already `&Vec<Box<dyn Trait>>` in the generated signature -- arrays are always
//    passed by reference -- producing `&&Vec<Box<dyn Trait>>`). Fix: the existing
//    local-variable trait-array borrow branch in `emit_for` now checks
//    `fn_current_params` and skips the extra `&` for a parameter (already a reference),
//    only adding it for a genuine local `let`/`var` binding (an owned `Vec<...>`).
//
// 4. Once bug 1 stopped auto-deriving `Clone`/`PartialEq` for a `[Trait]` field, the
//    auto-default's remaining `Debug` still broke the same way: `Box<dyn Trait>: Debug`
//    only holds when `Trait` itself requires `Debug`, and there was no way to even
//    declare that in Boring -- a bare `trait Foo as Debug:` supertrait emitted `trait Foo:
//    Debug` in the generated Rust, which fails to resolve (`Debug`/`Hash` are NOT in the
//    2021 prelude for plain name resolution, unlike the rest of the derive whitelist --
//    they resolve fine as *derive macro* names, which are always in scope, but not as an
//    ordinary trait-bound identifier: confirmed via `cargo build`, "expected trait, found
//    derive macro `Debug`"). Fix, two parts:
//      - `emit_trait` (`emit_struct.rs`) now fully qualifies `Debug`/`Hash` supertrait
//        names (`std::fmt::Debug`/`std::hash::Hash` -- see `Transpiler::
//        qualify_supertrait_name`/`NON_PRELUDE_TRAIT_PATHS` in `mod.rs`) so `as Debug`
//        actually compiles.
//      - `emit_struct.rs`'s derive computation now also drops `Debug` from the
//        auto-default for a struct with a `[Trait]` field whose trait does NOT
//        (transitively, via `Transpiler::trait_requires_debug`, walking the new
//        `trait_parents` map) require `Debug` -- so the derive line degrades to nothing
//        at all (safe, matches the documented `@derive()` escape hatch) instead of
//        emitting a `#[derive(Debug)]` that can't compile. `Debug` is kept exactly when
//        the field's trait declares (or transitively inherits) a `Debug` supertrait.
//
// `tests/cases/trait_array_field_param_bugs.br` has no explicit `@derive(...)` anywhere
// (auto-default path) and its `Drawable` trait does NOT require `Debug`, so it's used for
// direct codegen-shape assertions via `--emit-rust` (no `cargo build` -- the struct ends up
// with no derive attribute at all, by design; see bug 4 above).
//
// `tests/cases/trait_array_field_param_bugs_build.br` is the same shape but with an
// explicit `@derive()` on the struct (book.md's documented escape hatch: suppresses the
// auto-default entirely) so it can be run through a real `boring build` + `cargo build` +
// `cargo run` end to end, proving bugs 2 and 3 actually fixed (not just codegen-shape-
// approximated).
//
// `tests/cases/trait_array_field_debug_supertrait.br` has a trait that DOES declare `as
// Debug`, proving bug 4's other half: `Debug` is correctly kept (and the struct really
// compiles and runs) when the field's trait requires it.
//
// Run with:
//   cargo test --test trait_array_field_param_bugs

use std::path::Path;
use std::process::Command;

#[test]
fn trait_array_field_and_param_codegen_shape() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/trait_array_field_param_bugs.br");

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

    let body_of = |fn_sig: &str, next_marker: &str| -> String {
        let start = generated.find(fn_sig)
            .unwrap_or_else(|| panic!("`{}` not found in generated source:\n{}", fn_sig, generated));
        let rest = &generated[start..];
        let end = rest.find(next_marker).unwrap_or(rest.len());
        rest[..end].to_string()
    };

    // ── Bug 1 (+4): auto-default derive on a struct with a `[Trait]` field whose
    // trait does NOT require Debug -- must end up with NO derive attribute at all (not
    // just Clone/PartialEq dropped: Debug goes too, since `Drawable` here isn't `Debug`-
    // bounded -- see `trait_array_field_debug_supertrait.br` for the case where it is).
    let scene_start = generated.find("struct Scene {")
        .unwrap_or_else(|| panic!("`struct Scene` not found in generated source:\n{}", generated));
    // The immediately preceding non-blank line must not be a `#[derive(...)]` attached to
    // Scene -- look only at the tail end of the text right before `struct Scene {` (a few
    // dozen bytes is enough to catch an adjacent attribute line without also matching a
    // *different*, unrelated struct's derive line further back).
    let preceding = &generated[scene_start.saturating_sub(64)..scene_start];
    assert!(
        !preceding.contains("#[derive("),
        "a struct whose only `[Trait]` field's trait does NOT require Debug must get NO \
         derive attribute at all (Box<dyn Trait> implements neither Clone, PartialEq, nor \
         Debug here) -- got, just before `struct Scene {{`:\n{}", preceding
    );

    // ── Bug 2: `for s in scene.shapes:` (field access) must borrow, not clone ─
    let main_body = body_of("fn main", "\n}\n");
    assert!(
        main_body.contains("for s in &scene.shapes"),
        "iterating a `[Trait]` struct field must borrow (`&scene.shapes`), got:\n{}",
        main_body
    );
    assert!(
        !main_body.contains(".iter().cloned()"),
        "iterating a `[Trait]` struct field must not use `.iter().cloned()` \
         (Box<dyn Trait> isn't Clone), got:\n{}", main_body
    );

    // ── Bug 3: `for s in shapes:` over a `[Trait]`-typed parameter must not double-borrow ─
    let param_body = body_of("fn describeParam", "\n}\n");
    assert!(
        param_body.contains("shapes: &Vec<Box<dyn Drawable>>"),
        "expected the `[Trait]` parameter to be a single reference, got:\n{}", param_body
    );
    assert!(
        param_body.contains("for s in shapes {"),
        "a `[Trait]`-typed parameter is already `&Vec<Box<dyn Trait>>` -- iterating it \
         must not add another `&`, got:\n{}", param_body
    );
    assert!(
        !param_body.contains("for s in &shapes"),
        "a `[Trait]`-typed parameter must not be double-borrowed (`&&Vec<...>`), got:\n{}",
        param_body
    );
}

#[test]
fn trait_array_field_and_param_build_and_run() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/trait_array_field_param_bugs_build.br");
    let dir = Path::new("tests/cases/trait_array_field_param_bugs_build_unit_rust");

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

    let cargo_run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo run: {}", e));
    assert!(
        cargo_run.status.success(),
        "expected the generated project to run, but `cargo run` failed:\n--- stderr ---\n{}",
        String::from_utf8_lossy(&cargo_run.stderr),
    );
    let stdout = String::from_utf8_lossy(&cargo_run.stdout).replace("\r\n", "\n");
    assert_eq!(
        stdout.trim_end(),
        "circle\nsquare\ncircle\nsquare",
        "unexpected program output:\n{}", stdout
    );

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn trait_array_field_debug_supertrait_is_kept_and_compiles() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/trait_array_field_debug_supertrait.br");

    // ── Codegen shape: the supertrait bound must be fully qualified, and Scene's
    // auto-default derive must keep Debug (unlike the sibling fixture whose trait
    // doesn't require it) ──────────────────────────────────────────────────────
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
    assert!(
        generated.contains("trait Drawable: std::fmt::Debug {"),
        "a bare `Debug` supertrait must be fully qualified to `std::fmt::Debug` \
         (unqualified `Debug` fails to resolve as a trait bound -- it's not in the 2021 \
         prelude for plain name resolution, only for derive-macro invocation), got:\n{}",
        generated
    );
    let scene_start = generated.find("struct Scene {")
        .unwrap_or_else(|| panic!("`struct Scene` not found in generated source:\n{}", generated));
    let preceding = &generated[scene_start.saturating_sub(64)..scene_start];
    assert!(
        preceding.contains("#[derive(Debug)]"),
        "Scene's `[Trait]` field's trait requires Debug, so Scene must still derive it, \
         got, just before `struct Scene {{`:\n{}", preceding
    );

    // ── Real end-to-end compile + run ─────────────────────────────────────────
    let dir = Path::new("tests/cases/trait_array_field_debug_supertrait_unit_rust");
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

    let cargo_run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo run: {}", e));
    assert!(
        cargo_run.status.success(),
        "expected the generated project to run, but `cargo run` failed:\n--- stderr ---\n{}",
        String::from_utf8_lossy(&cargo_run.stderr),
    );
    let stdout = String::from_utf8_lossy(&cargo_run.stdout).replace("\r\n", "\n");
    assert_eq!(
        stdout.trim_end(),
        "circle\nsquare",
        "unexpected program output:\n{}", stdout
    );

    let _ = std::fs::remove_dir_all(dir);
}
