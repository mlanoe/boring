// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: commit 27e8432 ("fix(transpiler): track dict_vars for
// untyped dict[key] else default lets") taught `emit_let.rs`'s
// `track_let_metadata` to recognize `let existing = dict[key] else default`
// as dict-shaped via an AST-shape check (`expr_is_dict(dict_obj)`) — meant to
// cover the dict-of-*dict* case (tests/cases/dict_index_else_untyped_for.br).
//
// That check only proves the *container* being indexed (`dict_obj`, e.g.
// `table` in `table[key]`) is itself a dict — it never checked the dict's own
// VALUE type. A dict-of-*array* field (`{K=[V]}`) is just as
// `expr_is_dict`-true as a dict-of-dict field, so `table[key] else []`
// (yielding a plain `[V]`, not a dict) was wrongly added to `dict_vars` too.
// Every later index into the result (`xs[idx]`) then emitted broken
// dict-style `xs.get(&idx).cloned().expect(...)` codegen instead of ordinary
// Vec/slice indexing — `&idx` doesn't even type-check as a `Vec` index
// (`error[E0277]: the type [T] cannot be indexed by &isize`, slice indices
// must be `usize`). An explicit `[T]` type annotation on the local did NOT
// protect against this either — the syntactic-shape heuristic was overriding
// even an unambiguous, explicit array annotation.
//
// Fixed by additionally resolving `dict_obj`'s own declared type and checking
// that its *value* type (`index_element_type()`) is itself `Type::Dict(..)`
// — the actual dict-of-dict shape this tracking exists for — before trusting
// the heuristic, plus an explicit-array/set-annotation override as
// defense-in-depth.
//
// Fixture: `tests/cases/dict_index_else_array_value.br` declares four shapes
// in one file (two dict-of-array locals — untyped and explicitly `[int]`
// typed, a dict-of-array *struct field* read inside a method — the actual
// reported shape, and a dict-of-dict control case) so a future change can't
// silently regress any one of them while fixing another.
//
// This test emits the Boring functions via `--emit-rust` (raw Rust source,
// no Boring-generated Cargo project — same technique as
// `tests/dict_index_else_untyped_for.rs`) and:
//   1. String checks on the generated source pin the exact codegen shape for
//      each function (catches the bug directly, no compiler needed).
//   2. A real `cargo build` (no external stub needed — `HashMap`/`Vec` are
//      real std) catches the bug's actual failure mode too: dict-style
//      `.get(&idx)` over a `Vec<T>` doesn't type-check.
//
// Run with:
//   cargo test --test dict_index_else_array_value

use std::path::Path;
use std::process::Command;

#[test]
fn dict_of_array_index_else_binding_keeps_array_style_indexing() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/dict_index_else_array_value.br");
    let dir = Path::new("tests/cases/dict_index_else_array_value_rust");
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

    let fn_body = |start_marker: &str, end_markers: &[&str]| -> String {
        let start = generated
            .find(start_marker)
            .unwrap_or_else(|| panic!("{} not found in generated source", start_marker));
        let end = end_markers
            .iter()
            .filter_map(|m| generated[start..].find(m).map(|off| start + off))
            .min()
            .unwrap_or(generated.len());
        generated[start..end].to_string()
    };

    let untyped_local = fn_body("fn demo_untyped_local", &["fn demo_typed_local"]);
    assert!(
        untyped_local.contains("xs[(1) as usize]"),
        "expected the untyped dict-of-array `let xs = table[key] else []` \
         binding to index `xs` as a plain Vec, but it didn't — generated \
         function:\n{}",
        untyped_local
    );
    assert!(
        !untyped_local.contains("xs.get(&"),
        "the dict-of-array `xs` must never get dict-style `.get(&idx)` \
         indexing (a Vec index must be `usize`, not `&isize`) — generated \
         function:\n{}",
        untyped_local
    );

    let typed_local = fn_body("fn demo_typed_local", &["impl Wrapper", "struct Wrapper"]);
    assert!(
        typed_local.contains("xs[(1) as usize]"),
        "expected the explicitly `[int]`-typed dict-of-array `xs` binding to \
         also keep plain Vec indexing — an explicit array annotation must \
         win over the syntactic-shape heuristic — generated function:\n{}",
        typed_local
    );
    assert!(
        !typed_local.contains("xs.get(&"),
        "the explicitly `[int]`-typed `xs` must never get dict-style \
         `.get(&idx)` indexing — generated function:\n{}",
        typed_local
    );

    let wrapper_impl = fn_body("impl Wrapper", &["fn demo_dict_of_dict_still_works"]);
    assert!(
        wrapper_impl.contains("xs[(idx) as usize]") || wrapper_impl.contains("xs[idx"),
        "expected `Wrapper::get`'s dict-of-array *field* read \
         (`table[key] else []`) to index the result as a plain Vec — the \
         actual reported regression shape — generated impl:\n{}",
        wrapper_impl
    );
    assert!(
        !wrapper_impl.contains(".get(&idx)") && !wrapper_impl.contains(".get(&*idx)"),
        "`Wrapper::get`'s dict-of-array field read must never index its \
         result with dict-style `.get(&idx)` — generated impl:\n{}",
        wrapper_impl
    );

    // ── Control case: the original dict-of-dict fix must keep working ──────
    let dict_of_dict = fn_body("fn demo_dict_of_dict_still_works", &[]);
    assert!(
        dict_of_dict.contains("inner.get(\"a\")") || dict_of_dict.contains("inner.get(&"),
        "expected the dict-of-dict control case's nested dict read to keep \
         dict-style `.get(...)` indexing — generated function:\n{}",
        dict_of_dict
    );

    // ── Real `cargo build` against real `std::collections::HashMap`/`Vec` ──
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"dict_index_else_array_value_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
