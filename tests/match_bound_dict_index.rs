// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `emit_match_arm` (src/transpiler/emit_match.rs) registers
// a match-arm bound name's type into `var_types`/`string_vars`/
// `var_struct_types` etc. (the loop building `bound_types`), but never into
// `dict_vars`. `expr_is_dict` (src/transpiler/emit_methods.rs) — used
// throughout emit_expr.rs to decide dict-style `.get(&key)` codegen vs
// array-style `.get((idx) as usize)` codegen for `expr[key]` — checks
// `dict_vars` for a bare `Var`. `let`/`var` locals and function parameters
// already get added to `dict_vars` (emit_let.rs / emit_top.rs's
// `pre_seed_...` helpers) — a match-arm-bound name never did.
//
// Net effect: a dict-typed enum variant field (`{K=V}`), once bound by
// `match Wrapper.Dict(fields): fields[key]`, was transpiled as if `fields`
// were an array — casting the (non-numeric) key to `usize` and calling
// `.get()` as if on a `Vec`. `boring run` (the tree-walk interpreter) was
// unaffected — this is transpiler-only, and hits every backend that shares
// this general codegen path (confirmed here for the default Rust/std
// target; the bug was originally found via `--target metal`, which shares
// the same `emit_match_arm`/`expr_is_dict` codegen).
//
// Fixed by registering a match-bound name into `dict_vars` (and removing it
// again once the arm body is done, mirroring `bound_structs`/
// `bound_optionals`'s existing scoped-cleanup pattern) whenever its inferred
// type is `Type::Dict(..)`.
//
// Fixture: `tests/cases/match_bound_dict_index.br` declares four shapes in
// one file so a future change can't silently regress any one of them while
// fixing another — see the fixture's own header comment for the shape list.
//
// This test emits the Boring functions via `--emit-rust` (raw Rust source,
// no Boring-generated Cargo project — same technique as
// `tests/dict_index_else_array_value.rs`) and:
//   1. String checks on the generated source pin the exact codegen shape for
//      each function (catches the bug directly, no compiler needed).
//   2. A real `cargo build` (no external stub needed — `HashMap` is real
//      std) catches the bug's actual failure mode too: `fields.get((key) as
//      usize)` doesn't type-check when `key` is an `Arc<str>`.
//
// Run with:
//   cargo test --test match_bound_dict_index

use std::path::Path;
use std::process::Command;

#[test]
fn match_bound_dict_uses_hashmap_get_not_array_index() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/match_bound_dict_index.br");
    let dir = Path::new("tests/cases/match_bound_dict_index_rust");
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

    // The actual reported bug: a match-bound dict indexed directly in the
    // arm body must get HashMap-style `.get(&key)`, never `(key) as usize`.
    let get_direct = fn_body("fn get_direct", &["struct Holder", "impl Holder"]);
    assert!(
        get_direct.contains("fields.get(&*key)") || get_direct.contains("fields.get(&(key"),
        "expected the match-bound dict `fields` to be indexed with HashMap-style \
         `.get(&key)`, but it wasn't — generated function:\n{}",
        get_direct
    );
    assert!(
        !get_direct.contains("as usize") && !get_direct.contains("fields.get(("),
        "the match-bound dict `fields` must never get array-style `(key) as usize` \
         indexing — a non-numeric string key doesn't even type-check as `usize` — \
         generated function:\n{}",
        get_direct
    );

    // Controls: these were never broken — a regression here would mean the fix
    // for the match-bound case broke an already-working shape.
    let get_via_field = fn_body("fn get_via_field", &["fn get_via_param"]);
    assert!(
        get_via_field.contains("h.fields.get(&"),
        "expected the struct-field dict read to keep HashMap-style `.get(&key)` \
         indexing — generated function:\n{}",
        get_via_field
    );

    let get_via_param = fn_body("fn get_via_param", &["fn unwrap"]);
    assert!(
        get_via_param.contains("fields.get(&"),
        "expected the plain-parameter dict read to keep HashMap-style `.get(&key)` \
         indexing — generated function:\n{}",
        get_via_param
    );

    let get_via_fn = fn_body("fn get_via_fn", &["fn main"]);
    assert!(
        get_via_fn.contains(".get(&"),
        "expected the documented workaround (route the match-bound dict through a \
         function call, then index the return value) to keep working — generated \
         function:\n{}",
        get_via_fn
    );

    // ── Real `cargo build` against real `std::collections::HashMap` ──
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"match_bound_dict_index_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
