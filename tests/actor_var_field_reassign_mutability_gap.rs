// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a checker gap in `emit_expr_assign`'s "Mutex field
// write" / "RwLock field write" fast paths (`src/transpiler/emit_expr.rs`):
// `w.field = v`, where `w` itself is an actor/guard-qualified variable, used
// to `return` its fast-path codegen unconditionally — before ever reaching
// the "assigning to a non-reassignable (`let`/`mut`, not `var`/`var mut`)
// struct field" diagnostic further down. That diagnostic exists specifically
// to reject reassigning a field whose own declaration doesn't grant `var`/
// `var mut`, mirroring `boring run`'s interpreter — but for an actor/guard-
// qualified `w`, it was silently skipped, so `w.field = v` against a bare
// (`let`/`mut`) field compiled with no diagnostic at all.
//
// Found while fixing a real instance of this exact shape in
// `boring/interpreter/stdlib.br`: its `Interpreter` struct declared
// `Env'actor current_env` with no `var` (unlike its sibling `global_env`,
// which correctly has one), yet `exec.br`/`eval.br` reassign it throughout —
// an illegal-per-`docs/book.md` reassignment that should have been rejected
// at compile time (see CHANGELOG.md's `[Unreleased]` entry for the full
// story — `current_env` turned out to genuinely need `var`, so the real fix
// there was adding it, but the checker gap that let the illegal reassignment
// through unnoticed is a separate, general bug fixed here).
//
// Fixed by adding a shared `check_field_reassignable_via_var` helper, called
// from both fast paths before they return, as well as from the ordinary
// (non-actor-qualified-outer-var) diagnostic site it was extracted from.
//
// Run with:
//   cargo test --test actor_var_field_reassign_mutability_gap

use std::path::Path;
use std::process::Command;

fn emit_rust(src: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = tempfile_dir();
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, src).expect("failed to write fixture .br file");

    Command::new(bin)
        .arg("build")
        .arg(&br_file)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e))
}

// Each test gets its own scratch subdirectory under target/ so parallel test
// threads never race on the same file path.
fn tempfile_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("actor_var_field_reassign_mutability_gap_test_scratch")
        .join(format!("{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
    dir
}

#[test]
fn reassigning_a_bare_field_through_an_actor_qualified_outer_var_is_rejected() {
    let src = "\
struct Env:
    var int value = 0

struct Interpreter:
    Env'actor current_env

def swap(Interpreter'actor interp, Env'actor new_env):
    interp.current_env = new_env

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        !out.status.success(),
        "expected `boring build --emit-rust` to reject reassigning a bare \
         (non-`var`) field through an actor-qualified outer variable, but it \
         succeeded:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not reassignable"),
        "expected a 'not reassignable' diagnostic, got:\n{}",
        stderr
    );
}

#[test]
fn reassigning_a_bare_field_through_a_guard_qualified_outer_var_is_rejected() {
    let src = "\
struct Env:
    var int value = 0

struct Interpreter:
    Env'guard current_env

def swap(Interpreter'guard interp, Env'guard new_env):
    interp.current_env = new_env

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        !out.status.success(),
        "expected `boring build --emit-rust` to reject reassigning a bare \
         (non-`var`) field through a guard-qualified outer variable, but it \
         succeeded:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not reassignable"),
        "expected a 'not reassignable' diagnostic, got:\n{}",
        stderr
    );
}

// Regression guard: a `var`-qualified field through the same actor-qualified
// outer var must still compile — the fix must not over-reject.
#[test]
fn reassigning_a_var_field_through_an_actor_qualified_outer_var_still_compiles() {
    let src = "\
struct Env:
    var int value = 0

struct Interpreter:
    var Env'actor current_env

def swap(Interpreter'actor interp, Env'actor new_env):
    interp.current_env = new_env

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected a `var`-qualified field to still be reassignable through an \
         actor-qualified outer variable, but `boring build --emit-rust` failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
