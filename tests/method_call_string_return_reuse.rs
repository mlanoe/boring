// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `let reply = tok.decode(ids)` — an untyped local bound to
// a user STRUCT METHOD call returning `string` — was left out of
// `string_vars` tracking in `emit_let.rs`'s untyped-`let` inference block.
//
// That block (the `ExprKind::MethodCall(recv, method, _)` arm reading
// `struct_method_return_types`) only handled a `string`-returning method's
// return type by falling into its `_ => {}` catch-all — unlike the sibling
// `ExprKind::Call` (free-function) arm just above it, which has an
// unconditional `Type::Named(_) | Array(_) | Dict(..) | Set(_) =>
// var_types.insert(...)` fallback that happens to also cover `Type::Named
// ("string")`. So a free function's `string` return got auto-clone-tracked
// (via that generic `var_types` fallback), while a struct method's `string`
// return got no tracking under either `string_vars` or `var_types` at all.
//
// Left untracked, `emit_expr_owned`'s `Var` arm (which decides whether a
// later reuse of a moved-from local needs `.clone()`) never matched `reply`,
// so `arr.push(reply)` emitted a bare move immediately followed by another
// use (`print reply`) — `error[E0382]: borrow of moved value` at `cargo
// build`, even though the exact same pattern through a free function already
// compiled cleanly.
//
// Fixed by adding an explicit `is_string_type` arm (inserting into
// `string_vars`, matching what an explicit `let string reply = ...`
// annotation already does) plus the same generic `Type::Named(_) |
// Array(_) | Dict(..) | Set(_) => var_types.insert(...)` fallback the
// free-function arm already had, to the struct-method-call arm.
//
// Fixture: `tests/cases/method_call_string_return_reuse.br` exercises both
// the struct-method-call shape (the bug) and the free-function shape
// (already-working control case, kept here so a future change can't
// silently regress it) in one file.
//
// This test emits the Boring functions via `--emit-rust` (raw Rust source,
// no Boring-generated Cargo project — same technique as
// `tests/dict_field_clone_untyped_for.rs`) and:
//   1. String checks on the generated source pin the exact codegen shape for
//      both functions (catches the bug directly, no compiler needed).
//   2. A real `cargo build` (no external stub needed) catches the bug's
//      actual failure mode too: `cargo build` on the generated project must
//      succeed, not fail with E0382.
//
// Run with:
//   cargo test --test method_call_string_return_reuse

use std::path::Path;
use std::process::Command;

#[test]
fn struct_method_string_return_is_cloned_on_later_reuse() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/method_call_string_return_reuse.br");
    let dir = Path::new("tests/cases/method_call_string_return_reuse_rust");
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

    let method_start = generated
        .find("fn demo_method_call")
        .expect("demo_method_call not found in generated source");
    let method_end = generated[method_start..]
        .find("fn demo_free_function")
        .map(|off| method_start + off)
        .unwrap_or(generated.len());
    let method_body = &generated[method_start..method_end];

    assert!(
        method_body.contains("arr.push(reply.clone())"),
        "expected the struct-method-call-derived `reply` binding to be \
         `.clone()`d before the later `.push()` move (it is used again \
         afterward by `print reply`) — generated function:\n{}",
        method_body
    );

    let free_start = generated
        .find("fn demo_free_function")
        .expect("demo_free_function not found in generated source");
    let free_end = generated[free_start..]
        .find("fn main")
        .map(|off| free_start + off)
        .unwrap_or(generated.len());
    let free_body = &generated[free_start..free_end];

    assert!(
        free_body.contains("arr.push(reply.clone())"),
        "the free-function-derived `reply` binding (already-working control \
         case) must keep getting `.clone()`d too — generated function:\n{}",
        free_body
    );

    // ── Real `cargo build` — catches the bug's actual failure mode (E0382) ──
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"method_call_string_return_reuse_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
