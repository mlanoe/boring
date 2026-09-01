// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: index-*assignment* into a dict-of-dicts (`{K1={K2=V}}`)
// via a chained double index, `local_table[k1][k2] = v`, with no
// intermediate local. The assignment-target `ExprKind::Index(dict_obj, key)`
// arm in `emit_expr.rs`'s `emit_expr_assign` only recognized `dict_obj` shaped
// as `Var` (a plain dict-typed local) or `Field` (`obj.field`/implicit-self
// `field`) — never `Index` itself, i.e. `dict_obj` being `local_table[k1]`,
// the outer index of the chain. That left every double-index assignment
// falling through to the generic array-index LHS codegen further down,
// producing `local_table.get(&k1).cloned().expect("dict key not found")
// [(k2) as usize].clone() = v.clone()` — a numeric-cast array index into a
// `HashMap` (doesn't type-check) assigning into the *result of a `.clone()`
// call* (not even a valid Rust assignment target either way).
//
// Fixed by:
//   - Teaching `expr_is_dict` (`emit_methods.rs`) to recognize a chained
//     `Index(obj, _)` as dict-shaped when `obj`'s declared type has a `Dict`
//     element/value type (via a new `Index` arm on
//     `resolve_expr_declared_type`, which resolves an index read's type as
//     one level of `index_element_type()` off its base — recursing through
//     `Var`/`Field`/`Index` so a chain resolves too).
//   - Adding a chained-dict branch to `emit_expr_assign`'s dict-subscript
//     handling: when `dict_obj` is itself `Index(..)` and `expr_is_dict`
//     confirms it, emit `dict_obj` under `in_lhs_assign` (reusing
//     `emit_expr_index`'s existing `.get_mut(..).expect("dict key not
//     found")` place-expression path — the same one `d[k].method()`/
//     `d[k] += rhs` already use) and `.insert()` into the resulting place.
//
// This mirrors the tree-walk interpreter's `methods.rs::assign`'s recursive
// `ExprKind::Index` semantics for the same source: it reads the outer key
// (erroring if absent, no auto-vivify of an empty inner dict), mutates a
// pairs copy, and reassigns it back — i.e. the outer key must already exist.
//
// Fixture: `tests/cases/dict_chained_index_assign.br` declares two shapes in
// one struct so a future change can't silently regress either — see that
// file's own header comment. `set_via_local` (copy into an intermediate local,
// mutate, write back) was already fixed by the untyped-`else`-binding
// `dict_vars` tracking fix (see `dict_index_else_untyped_for.rs`) and is
// pinned here as the control case; `set_direct` (the chained double index,
// no intermediate local) is the shape this test actually targets.
//
// This test emits the Boring struct via `--emit-rust` (raw Rust source, no
// Boring-generated Cargo project — same technique as `for_loop_mut_tuple.rs`
// and `dict_index_else_untyped_for.rs`) and:
//   1. String checks on the generated source pin the exact codegen shape for
//      `set_direct` (catches the bug directly, no compiler needed).
//   2. A real `cargo build` (no external stub needed — `HashMap` is real
//      std) catches the bug's actual failure mode too: the broken array-index
//      LHS assignment doesn't type-check, and isn't a valid Rust assignment
//      target in the first place.
//
// Run with:
//   cargo test --test dict_chained_index_assign

use std::path::Path;
use std::process::Command;

#[test]
fn chained_double_index_dict_assignment_uses_get_mut_insert() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/dict_chained_index_assign.br");
    let dir = Path::new("tests/cases/dict_chained_index_assign_rust");
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

    let direct_start = generated
        .find("fn set_direct")
        .expect("set_direct not found in generated source");
    let direct_body = &generated[direct_start..];

    assert!(
        direct_body.contains(".get_mut(") && direct_body.contains(".insert("),
        "expected the chained `local_table[k1][k2] = v` assignment (no \
         intermediate local) to transpile to a `.get_mut(..)` place \
         expression followed by `.insert(..)` on the inner dict, but it \
         didn't — generated function:\n{}",
        direct_body
    );
    assert!(
        !direct_body.contains("as usize"),
        "the chained dict-of-dicts assignment must never get the array-index \
         LHS rewrite (a HashMap value has no numeric-cast index) — generated \
         function:\n{}",
        direct_body
    );

    let local_start = generated
        .find("fn set_via_local")
        .expect("set_via_local not found in generated source");
    let local_end = direct_start.max(local_start);
    let local_body = &generated[local_start..local_end];
    assert!(
        local_body.contains("by_var.insert("),
        "expected the intermediate-local control case (`by_var = ... else {{=}}`, \
         then `by_var[id] = v`) to keep transpiling to plain dict `.insert(..)` \
         — generated function:\n{}",
        local_body
    );

    // ── Real `cargo build` against a real `std::collections::HashMap` ──────
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"dict_chained_index_assign_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
