// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression tests for the "implicit self doesn't route through the lock" bug:
// Boring's own "bare field access inside a method resolves to `self.field`
// automatically" convention (CLAUDE.md's "implicit self", docs/book.md's
// "implicit-self" section) is a source-level convention only — it was never an
// AST rewrite. Every shape-based receiver recognizer in
// src/transpiler/emit_methods.rs (`try_emit_mutex_method`/`try_emit_rwlock_method`/
// `try_emit_actor_field_method`/`resolve_observed_field_receiver` and the `.value`
// field-read path via `resolve_observed_value_base`) pattern-matches the literal
// parsed shape `ExprKind::Field(Box::new(ExprKind::Var("self")), field)` — a shape
// that only exists when the source text itself wrote `self.field`. A bare
// `field.method()`/`field.value.property` inside the declaring struct's own method
// parses to a plain `ExprKind::Var("field")` (or `Field(Var("field"), "value")`)
// instead, so none of those recognizers ever fired for it: it fell through to the
// generic fallback, which emits an unlocked, unnotified call/read straight on the
// field's real Rust wrapper type (`Rc<RefCell<T>>`/`Arc<Mutex<T>>`/
// `BoringObserved<T>`) — normally a hard `cargo build` E0599/E0609 — and (more
// seriously) skipped `check_field_def_call_mut_gate` entirely, so calling a `def`
// method on a field that isn't declared `mut` compiled with zero Boring-level
// diagnostic, unlike the identical `self.field.method()` spelling.
//
// Fixed by `Transpiler::normalize_implicit_self_field` (emit_methods.rs): it
// rewrites a bare implicit-self-field receiver into the equivalent explicit
// `Field(Var("self"), field)` `Expr` shape up front (in `emit_method_call`, and at
// the two `.value`-hop resolution points in `try_emit_observed_field_method`/
// `resolve_observed_value_base`), so the rest of the dispatch chain — and its
// mut-gating — can no longer tell the bare and explicit forms apart. These tests
// mirror tests/observed_qualifier.rs's `self.field`-scoped mut-gating/dispatch
// tests one-for-one, but with the struct's own method calling/reading the field
// bare instead of through `self.`.
//
// Run with:
//   cargo test --test implicit_self_field_dispatch

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
        .join("implicit_self_field_dispatch_test_scratch")
        .join(format!("{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
    dir
}

// ── (a) plain 'actor field — bare method call dispatches through the lock ──────

#[test]
fn bare_actor_field_method_call_dispatches_through_the_lock() {
    let src = "struct Counter:\n    var int value = 0\n    def inc(): value += 1\n\nstruct Holder:\n    mut Counter'actor c = Counter(0)\n    def bump(): c.inc()\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected a bare 'actor field method call to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("self.c.lock().unwrap().inc()"),
        "expected the bare field method call to dispatch through the Mutex lock (not an unlocked call on the field's own Arc<Mutex<_>> type), got:\n{}", generated
    );
}

// ── (b) plain 'guard field — bare method call dispatches through the lock ──────

#[test]
fn bare_guard_field_method_call_dispatches_through_the_lock() {
    let src = "struct Counter:\n    var int value = 0\n    def inc(): value += 1\n\nstruct Holder:\n    mut Counter'guard c = Counter(0)\n    def bump(): c.inc()\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected a bare 'guard field method call to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("self.c.write().unwrap().inc()"),
        "expected the bare field method call to dispatch through the RwLock write lock, got:\n{}", generated
    );
}

// ── (c) 'observed field — bare method call AND bare `.value` field read ────────

#[test]
fn bare_observed_field_method_call_dispatches_through_lock_and_notifies() {
    let src = "struct Counter:\n    var int value = 0\n    def inc(): value += 1\n\nstruct Holder:\n    mut Counter'actor'observed c = Counter(0)\n    def bump(): c.inc()\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected a bare 'actor'observed field method call to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("self.c.value.lock().unwrap().inc()"),
        "expected the bare field method call to dispatch through the lock, got:\n{}", generated
    );
    assert!(
        generated.contains("__boring_notify"),
        "expected the bare field method call to still notify subscribers afterward, got:\n{}", generated
    );
}

#[test]
fn bare_observed_field_value_read_dispatches_through_the_lock() {
    let src = "struct Inner:\n    var string name = \"\"\n\nstruct Outer:\n    var mut Inner'actor'observed model = Inner()\n    def touch():\n        print \"{model.value.name}\"\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected a bare `.value` field read to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("self.model.value.lock().unwrap().name"),
        "expected the bare `.value` field read to route through the Mutex lock (not a bogus direct field access on BoringObserved<Arc<Mutex<Inner>>>), got:\n{}", generated
    );
}

// ── (d) checker-rejection parity: the missing-`mut` gate must fire for the bare
// form exactly like it already does for the explicit `self.field` spelling ─────

#[test]
fn bare_non_mut_actor_field_method_call_is_rejected() {
    let src = "struct Counter:\n    var int value = 0\n    def inc(): value += 1\n\nstruct Holder:\n    Counter'actor c = Counter(0)\n    def bump(): c.inc()\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a non-mut 'actor field's bare def-method call to be rejected, got:\n{}", String::from_utf8_lossy(&out.stdout));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`c` is not declared `mut`") && stderr.contains("non-mut field"),
        "expected the field-scoped mut-gating diagnostic to fire for the bare form exactly like it does for `self.c.inc()`, got:\n{}", stderr
    );
}

#[test]
fn bare_non_mut_guard_field_method_call_is_rejected() {
    let src = "struct Counter:\n    var int value = 0\n    def inc(): value += 1\n\nstruct Holder:\n    Counter'guard c = Counter(0)\n    def bump(): c.inc()\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a non-mut 'guard field's bare def-method call to be rejected, got:\n{}", String::from_utf8_lossy(&out.stdout));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`c` is not declared `mut`") && stderr.contains("non-mut field"),
        "expected the field-scoped mut-gating diagnostic to fire for the bare form, got:\n{}", stderr
    );
}

#[test]
fn bare_non_mut_observed_field_method_call_is_rejected() {
    let src = "struct Counter:\n    var int value = 0\n    def inc(): value += 1\n\nstruct Holder:\n    Counter'actor'observed c = Counter(0)\n    def bump(): c.inc()\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a non-mut 'actor'observed field's bare def-method call to be rejected, got:\n{}", String::from_utf8_lossy(&out.stdout));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`c` is not declared `mut`") && stderr.contains("non-mut field"),
        "expected the field-scoped mut-gating diagnostic to fire for the bare observed form too, got:\n{}", stderr
    );
}

// A mut/var-mut field must keep compiling — this fix must not regress the
// already-correct permissive case (the positive tests above already cover this
// implicitly, but this makes the "still succeeds" claim explicit per qualifier).
#[test]
fn mut_bare_actor_and_guard_field_method_calls_still_succeed() {
    for qual in ["'actor", "'guard"] {
        let src = format!("struct Counter:\n    var int value = 0\n    def inc(): value += 1\n\nstruct Holder:\n    mut Counter{qual} c = Counter(0)\n    def bump(): c.inc()\n\ndef main():\n    print \"ok\"\n");
        let out = emit_rust(&src);
        assert!(out.status.success(), "expected a mut {qual} field's bare def-method call to still succeed, got:\n{}", String::from_utf8_lossy(&out.stderr));
    }
}

// A non-mutating `req` method through a bare non-mut field must NOT be rejected —
// the gate only applies to `def` (mutating) methods.
#[test]
fn bare_req_method_through_non_mut_actor_field_is_not_rejected() {
    let src = "struct Counter:\n    var int value = 0\n    req int current(): value\n\nstruct Holder:\n    Counter'actor c = Counter(0)\n    def bump():\n        let n = c.current()\n        print \"{n}\"\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected a bare req method through a non-mut field to be permitted, got:\n{}", String::from_utf8_lossy(&out.stderr));
}
