// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for the dict-value-mutation bug documented in
// docs/mut-type-modifier.md's "Known implementation bugs": for a dict
// declared `{K = mut V}`, calling a `def` (mutating) method through a
// fetched value (`d[k].method()`) transpiled to
// `d.get(k).cloned().expect(...).method()` — mutating a throwaway clone of
// the fetched value, never the actual map entry. `boring build --emit-rust`
// accepted the source with no diagnostic and the generated Rust compiled
// and ran with no error, but the mutation was silently lost: `boring run`
// (the interpreter) printed the mutated value, while the compiled binary
// printed the original, unmutated one.
//
// Root cause: `emit_expr.rs`'s `emit_expr_index` had two dict-indexing
// branches (a bare dict var, and a `self.field` dict-typed struct field)
// that always emitted `.get(key).cloned().expect(...)`, never consulting
// `self.in_lhs_assign` — unlike the plain array-indexing branch a few lines
// below, which already switches to a clone-free `arr[idx]` when
// `in_lhs_assign` is set (the same flag `emit_method_call_fallback` sets
// before emitting an `Index` method-call receiver, precisely to avoid this
// class of bug for arrays). Fixed by making both dict branches honor
// `in_lhs_assign` the same way, switching to `(*d.get_mut(key).expect(...))`
// — the leading `*` turns the `&mut V` into a real lvalue place, so it works
// both as a method-call receiver (`(*d.get_mut(k)...).method()`) and as a
// compound-assignment target (`(*d.get_mut(k)...) += rhs`).
//
// `boring run` (the interpreter) already got this right independently, so
// this test pins the transpiler side specifically.
//
// Run with:
//   cargo test --test dict_mut_value_binding

use std::path::Path;
use std::process::Command;

#[test]
fn dict_value_mutation_compiles_and_is_observed() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/dict_mut_value_binding.br");
    let dir = Path::new("tests/cases/dict_mut_value_binding_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    // ── boring build --emit-rust must succeed (checker already permits the
    // `def` call through the dict's `mut`-qualified value type) ─────────────
    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to accept {}, but it failed:\n{}",
        case_br.display(),
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    // The bug's signature: `.inc()` called through a throwaway
    // `.get(key).cloned()` instead of a real mutable place via `.get_mut`.
    // Assert the fix is actually present in the generated source, not just
    // that the final stdout happens to match (belt and suspenders — a future
    // refactor could reintroduce the clone while still getting the *last*
    // mutation to "work" by accident on a single-key test).
    assert!(
        generated.contains("get_mut(") && !generated.contains(".cloned().expect(\"dict key not found\").inc()"),
        "expected the `.inc()` method-call receiver to go through a real \
         mutable place (`get_mut`), not a throwaway `.get(...).cloned()`, \
         but the generated source is:\n{}",
        generated
    );

    // ── And the generated Rust must actually build and run — this is the ────
    // part that used to silently drop the mutation (no build/runtime error
    // of any kind, just the wrong printed value).
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"dict_mut_value_binding_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("failed to write Cargo.toml");

    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {}", e));

    assert!(
        run.status.success(),
        "expected the generated Rust to build and run, but it failed:\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&run.stderr),
        generated,
    );

    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(
        actual.trim_end(),
        "2 11",
        "unexpected stdout — the dict-value mutations were not observed \
         (this is exactly the silent-correctness bug this test pins: no \
         build or runtime error, just the wrong value)"
    );

    // Clean up the generated build dir so repeated runs don't accumulate
    // disk usage (target/ dirs in particular).
    let _ = std::fs::remove_dir_all(dir);
}
