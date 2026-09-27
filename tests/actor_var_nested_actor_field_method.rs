// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a method-call codegen bug in
// `try_emit_actor_field_method` (`src/transpiler/emit_methods.rs`):
// `outer.actor_field.method(args)`, where `actor_field`'s declared type is
// `T'actor`/`T'actor'task`, when the OUTER receiver variable is *itself*
// actor-qualified (its own Rust type is `Arc<Mutex<Struct>>`/
// `Rc<RefCell<Struct>>`, not a plain `&Struct`).
//
// Root cause: the function built `obj_s` as
// `format!("{}.{}", self.emit_expr(inner_obj), field_name)` unconditionally.
// `self.emit_expr(inner_obj)` deliberately returns the *bare handle name* for
// a struct-typed actor/guard variable (see emit_expr.rs's `var_lock_scalar`
// doc comment — field-write/method-call paths are supposed to route the
// lock themselves), so for an actor-qualified outer variable this produced
// `outer.counter` verbatim instead of first unlocking `outer` — invalid Rust,
// since `Arc<Mutex<Outer>>` has no field named `counter`. Confirmed against
// the real bug this was extracted from: `boring/interpreter/stdlib.br`'s
// `Interpreter` struct, where every method call reached through
// `interp.current_env.<method>(...)` / `interp.global_env.<method>(...)`
// (`interp` being inferred `'actor`) failed identically with a real rustc
// E0609 ("no field `current_env`/`global_env` on `&Arc<Mutex<Interpreter>>`") —
// see `tests/interpreter_build.rs` and CHANGELOG.md's `[Unreleased]` entry.
//
// Fixed by resolving the correct lock/borrow access for the outer variable
// (`var_mutex_types`/`var_mutex_task_types` via `mutex_var_read`, or
// `managed_mutex_vars`/`managed_refcell_vars`) before appending `.field_name`,
// mirroring the `MUTATING_COLLECTION_METHODS` branch just below in the same
// file, which already did this correctly for the same shape of receiver.
//
// Run with:
//   cargo test --test actor_var_nested_actor_field_method

use std::path::Path;
use std::process::Command;

#[test]
fn method_call_on_actor_field_through_an_actor_qualified_outer_var_locks_the_outer_var_first() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/actor_var_nested_actor_field_method.br");
    let dir = Path::new("tests/cases/actor_var_nested_actor_field_method_rust");
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

    // ── Codegen-shape assertion (exact string check, no compiler needed) ────
    assert!(
        generated.contains("outer.lock().unwrap().counter.lock().unwrap().inc();"),
        "expected the method call on the nested actor field to lock the outer \
         actor-qualified variable BEFORE reaching the field, but it didn't — \
         generated source:\n{}",
        generated
    );

    // ── Real `cargo build` + run, checking the actual mutated value ─────────
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"actor_var_nested_actor_field_method_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
        "expected the generated Rust to build and run, but it failed (this is the \
         original bug — a real rustc E0609 'no field `counter`' on the actor-qualified \
         outer variable):\n--- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&run.stderr),
        generated,
    );

    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(
        actual.trim_end(),
        "2",
        "expected two `.inc()` calls through the outer actor var to be reflected \
         in the nested actor field's value"
    );

    let _ = std::fs::remove_dir_all(dir);
}
