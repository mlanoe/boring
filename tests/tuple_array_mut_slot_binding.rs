// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a real transpiler codegen gap: owned `mut Type`
// inside a tuple slot (`(mut Point, string) t`) or an array element
// (`[mut Point] arr`) is correctly permission-checked by the checker/
// interpreter (a `def` call through the mut slot/element is allowed — see
// `Type::grants_mut`/`Type::tuple_slot_mut_flags`/`Type::index_element_type`),
// and `boring build --emit-rust` accepts the source with no diagnostic. But
// Rust has no per-tuple-slot or per-array-element `mut` — the *whole* Rust
// binding must be `mut` for `t.0.move_to(...)` / `arr[0].move_to(...)` to
// compile, and the transpiler wasn't emitting that: it produced
// `let t: (Point, Arc<str>) = ...` / `let arr: Vec<Point> = ...` (missing
// `mut`), which only failed downstream at `cargo build` with E0596 — despite
// `boring build` itself reporting success.
//
// Root cause: `emit_let.rs`'s `emit_let` decided the Rust `let`/`let mut`
// keyword solely from `s.binding.is_mutable()` (the Boring-level `let`/`mut`/
// `var` binding keyword), never consulting the *type*'s own nested `mut`
// slots. Fixed by adding `Type::nested_slot_grants_mut` (`src/ast/mod.rs`) —
// true for a tuple with any `mut`-qualified slot or an array/`ArrayN` with a
// `mut`-qualified element type — and forcing `let mut` whenever it's set,
// exactly mirroring how a plain `mut`-qualified struct binding already gets
// `let mut` today. This is a "transpiler honesty" invariant — the checker's
// per-slot permission tracking must be backed by a Rust binding that's
// actually `mut` wherever Rust itself has no per-slot equivalent — finally
// implemented as codegen rather than just an unenforced assumption.
//
// Deliberately scoped to tuples/arrays only — dict value mutation
// (`{K = mut V}`) has a separate, worse bug (silently mutates a throwaway
// clone, never the map entry) that this fix does not address; see the same
// "Known implementation bugs" section.
//
// Run with:
//   cargo test --test tuple_array_mut_slot_binding

use std::path::Path;
use std::process::Command;

fn build_run_and_check(case_name: &str, br_relpath: &str, expected_stdout: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new(br_relpath);
    let dir_name = format!("tests/cases/{}_rust", case_name);
    let dir = Path::new(&dir_name);
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    // ── boring build --emit-rust must succeed (checker already allows this) ─
    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to accept {} (the checker already \
         permits the `def` call through the mut slot/element), but it failed:\n{}",
        br_relpath,
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    // ── And the generated Rust must actually build and run — this is the ────
    // part that used to fail with E0596 before the fix.
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{}_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            case_name
        ),
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
        "expected the generated Rust for {} to build and run (previously \
         failed with E0596, \"cannot borrow ... as mutable\", because the \
         `let`/`let mut` binding didn't reflect the type's nested `mut` slot), \
         but it failed:\n--- stderr ---\n{}\n--- generated source ---\n{}",
        br_relpath,
        String::from_utf8_lossy(&run.stderr),
        generated,
    );

    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(
        actual.trim_end(),
        expected_stdout,
        "unexpected stdout from {}",
        br_relpath
    );

    // Clean up the generated build dir so repeated runs don't accumulate disk
    // usage (target/ dirs in particular).
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn tuple_slot_mut_binding_compiles_and_mutation_is_observed() {
    build_run_and_check(
        "tuple_mut_slot_binding",
        "tests/cases/tuple_mut_slot_binding.br",
        "1 2 label",
    );
}

#[test]
fn array_element_mut_binding_compiles_and_mutation_is_observed() {
    build_run_and_check(
        "array_mut_element_binding",
        "tests/cases/array_mut_element_binding.br",
        "1 11",
    );
}
