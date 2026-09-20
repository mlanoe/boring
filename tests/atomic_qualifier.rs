// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Checker and codegen tests for the `'atomic` ownership qualifier (see
// docs/qualifiers.md's `'atomic` section for the full design).
//
// Behavioral (stdout-equivalence) tests for `'atomic` live in tests/transpile.rs
// as ordinary `transpile_test!` cases (`atomic_explicit_ops`,
// `atomic_promotion_actor`, `atomic_promotion_suppressed_with`) — those confirm
// the *observable* output is correct/unchanged. This file instead inspects the
// actual generated Rust text (`boring build --emit-rust`) to confirm the
// *representation* is what the design says it should be: the checker rejects
// what it should reject, and the automatic `'actor`/`'guard` → `'atomic`
// promotion pass fires exactly when its four criteria hold and not otherwise.
//
// Run with:
//   cargo test --test atomic_qualifier

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
        .join("atomic_qualifier_test_scratch")
        .join(format!("{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
    dir
}

// ── Checker: type-compatibility gate ────────────────────────────────────────────

#[test]
fn float_atomic_is_rejected() {
    let out = emit_rust("def main():\n    var x'atomic = 1.5\n    print \"{x}\"\n");
    assert!(!out.status.success(), "expected float 'atomic to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot combine `'atomic` with"),
        "expected the atomic-incompatibility error, got:\n{}", stderr
    );
    assert!(stderr.contains("float64"), "expected the error to name the offending type, got:\n{}", stderr);
}

#[test]
fn struct_atomic_is_rejected() {
    let src = "struct Point:\n    int x\n    int y\n\ndef main():\n    var p'atomic = Point(1, 2)\n    print \"{p.x}\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected struct 'atomic to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot combine `'atomic` with `Point`"),
        "expected the atomic-incompatibility error naming Point, got:\n{}", stderr
    );
}

#[test]
fn int128_atomic_is_rejected() {
    // No `AtomicI128` in stable `std` — rejected same as a float, even though
    // `int128` is otherwise a perfectly ordinary integer type.
    let out = emit_rust("def main():\n    var int128'atomic x = 1\n    print \"{x}\"\n");
    assert!(!out.status.success(), "expected int128 'atomic to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot combine `'atomic` with"),
        "expected the atomic-incompatibility error, got:\n{}", stderr
    );
}

#[test]
fn scalar_atomic_types_are_accepted() {
    for (ty, lit) in [
        ("int", "5"), ("uint", "5"), ("bool", "true"),
        ("int8", "1"), ("int16", "1"), ("int32", "1"), ("int64", "1"),
        ("uint8", "1"), ("uint16", "1"), ("uint32", "1"), ("uint64", "1"),
    ] {
        // Type-position form: `var Type'atomic name = value`.
        let src = format!("def main():\n    var {ty}'atomic v = {lit}\n    print \"{{v}}\"\n", ty = ty, lit = lit);
        let out = emit_rust(&src);
        assert!(
            out.status.success(),
            "expected {ty}'atomic to be accepted, got:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

// ── Checker: `with`-block incompatibility ───────────────────────────────────────

#[test]
fn with_block_on_atomic_binding_is_rejected() {
    let src = "def main():\n    var counter'atomic = 0\n    with counter:\n        counter += 5\n    print \"{counter}\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected `with` on an 'atomic binding to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot use `with counter:`") && stderr.contains("'atomic"),
        "expected the with-incompatibility error, got:\n{}", stderr
    );
}

// ── Fallback-chain inertness (source-level corroboration) ───────────────────────
//
// The precise unit-level proof (`resolve_fallback([Actor, Guard, Atomic]) == Actor`)
// lives in `src/transpiler/infer_qualifiers.rs`'s own `#[cfg(test)] mod tests` —
// `resolve_fallback` is a private function, only reachable from an in-crate test.
// This test corroborates the same claim from the outside: a plain `'actor`
// annotation must never be silently reinterpreted as `'atomic` by the transpiler
// on its own (only the promotion pass, a distinct mechanism with its own four
// criteria, ever rewrites `'actor` to the atomic representation).
#[test]
fn bare_actor_scalar_is_never_atomic_without_promotion_eligibility() {
    // Escapes via return → promotion criterion 2 fails → must stay Mutex-backed,
    // never atomic, even though the underlying type is atomic-eligible.
    let src = "def int'actor make():\n    var counter'actor = 0\n    counter += 5\n    return counter\n\ndef main():\n    let c = make()\n    print \"done\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to build:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(generated.contains("Mutex<"), "expected the escaping 'actor local to stay Mutex-backed, got:\n{}", generated);
    assert!(!generated.contains("Atomic"), "expected no atomic representation for an escaping binding, got:\n{}", generated);
}

// ── Promotion pass: positive and negative cases (codegen inspection) ────────────

#[test]
fn promotion_fires_for_local_scalar_actor() {
    let src = "def main():\n    var counter'actor = 0\n    counter += 5\n    counter -= 2\n    print \"{counter}\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to build:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("AtomicIsize") && generated.contains("fetch_add") && generated.contains("fetch_sub"),
        "expected the promotion pass to fire (Arc<AtomicIsize>, fetch_add/fetch_sub), got:\n{}", generated
    );
    assert!(!generated.contains("Mutex<"), "expected no leftover Mutex representation, got:\n{}", generated);
}

#[test]
fn promotion_suppressed_for_return_escape() {
    let src = "def int'actor make():\n    var counter'actor = 0\n    counter += 5\n    return counter\n\ndef main():\n    let c = make()\n    print \"done\"\n";
    let out = emit_rust(src);
    assert!(out.status.success());
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(generated.contains("Mutex<"), "expected Mutex representation for a returned binding, got:\n{}", generated);
    assert!(!generated.contains("Atomic"), "expected promotion NOT to fire for a returned binding, got:\n{}", generated);
}

#[test]
fn promotion_suppressed_for_with_block_usage() {
    let src = "def main():\n    var counter'actor = 0\n    with counter:\n        counter += 5\n        counter -= 2\n    print \"{counter}\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to build:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(generated.contains("Mutex<"), "expected Mutex representation for a with-block-used binding, got:\n{}", generated);
    assert!(!generated.contains("Atomic"), "expected promotion NOT to fire when used inside a `with` block, got:\n{}", generated);
}

#[test]
fn promotion_suppressed_for_non_atomic_op_shape() {
    // `counter = counter * 2` is not a recognized single atomic primitive (no
    // `fetch_mul`) — criterion 3 fails, promotion must not fire.
    let src = "def main():\n    var counter'actor = 1\n    counter = counter * 2\n    print \"{counter}\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to build:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(generated.contains("Mutex<"), "expected Mutex representation for a non-single-op assignment, got:\n{}", generated);
    assert!(!generated.contains("Atomic"), "expected promotion NOT to fire for `x = x * 2`, got:\n{}", generated);
}

#[test]
fn promotion_suppressed_for_closure_capture() {
    let src = "def main():\n    var counter'actor = 0\n    let f = () :\n        counter += 1\n    f()\n    print \"{counter}\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to build:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(generated.contains("Mutex<"), "expected Mutex representation for a closure-captured binding, got:\n{}", generated);
    assert!(!generated.contains("Atomic"), "expected promotion NOT to fire for a closure-captured binding, got:\n{}", generated);
}

#[test]
fn promotion_fires_for_local_scalar_guard_too() {
    // `'guard`-sourced promotion is at least as safe as `'actor`-sourced — same
    // four criteria, no extra restriction (docs/qualifiers.md's `'atomic` section).
    let src = "def main():\n    var counter'guard = 0\n    counter += 5\n    print \"{counter}\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to build:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("AtomicIsize") && generated.contains("fetch_add"),
        "expected the promotion pass to fire from a 'guard source too, got:\n{}", generated
    );
    assert!(!generated.contains("RwLock<"), "expected no leftover RwLock representation, got:\n{}", generated);
}

// ── Multi vs single threading emission ──────────────────────────────────────────

#[test]
fn explicit_atomic_emits_cell_single_threaded() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = tempfile_dir();
    let br_file = dir.join("main.br");
    // `mut`, not `var` — see the binding-permission enforcement tests below:
    // a bare `var'atomic` scalar is rebind-only, `+=` needs `mut`/`var mut`.
    std::fs::write(&br_file, "def main():\n    mut counter'atomic = 0\n    counter += 5\n    print \"{counter}\"\n").unwrap();
    let out = Command::new(bin)
        .arg("build").arg(&br_file).arg("--emit-rust").arg("--threading").arg("single")
        .output().unwrap();
    assert!(out.status.success(), "expected this program to build:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(generated.contains("Rc<std::cell::Cell<isize>>"), "expected the single-thread Cell<T> collapse, got:\n{}", generated);
    assert!(!generated.contains("Atomic"), "single-thread mode must never emit a real atomic type, got:\n{}", generated);
}

// ── Binding permission: `let`/`mut`/`var`/`var mut` on an explicit `'atomic` ────
//
// `'atomic` is structurally in the same family as `'actor`/`'guard` (this file's
// own header comment, docs/qualifiers.md's `'atomic` section) — a bare `let` must
// be read-only, and (unlike `'actor`/`'guard`) bare `var` grants nothing extra
// either, since a scalar `'atomic` binding has no separate rebind-the-pointer
// operation: every one of `x = n`/`x += n`/`x -= n`/`x.swap(n)` is content-
// mutation through the shared lock-free cell. See `src/checker/mod.rs`'s
// `check_assign_target` (assignment-shaped ops) and
// `src/transpiler/emit_methods.rs`'s `try_emit_atomic_method` (`.swap()`).

#[test]
fn atomic_bare_let_rejects_compound_assign() {
    let out = emit_rust("def main():\n    let counter'atomic = 0\n    counter += 5\n    print \"{counter}\"\n");
    assert!(!out.status.success(), "expected a bare `let` 'atomic compound-assign to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not declared `mut`") && stderr.contains("'atomic"),
        "expected a clear mut-permission error naming 'atomic, got:\n{}", stderr
    );
}

#[test]
fn atomic_bare_let_rejects_plain_store() {
    let out = emit_rust("def main():\n    let counter'atomic = 0\n    counter = 10\n    print \"{counter}\"\n");
    assert!(!out.status.success(), "expected a bare `let` 'atomic plain store to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not declared `mut`") && stderr.contains("'atomic"), "got:\n{}", stderr);
}

#[test]
fn atomic_bare_let_rejects_swap() {
    let out = emit_rust("def main():\n    let counter'atomic = 0\n    let old = counter.swap(100)\n    print \"{old}\"\n");
    assert!(!out.status.success(), "expected a bare `let` 'atomic `.swap()` to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not declared `mut`") && stderr.contains(".swap()") && stderr.contains("'atomic"),
        "expected a clear mut-permission error naming 'atomic and `.swap()`, got:\n{}", stderr
    );
}

#[test]
fn atomic_bare_var_alone_also_rejects_compound_assign() {
    // The one real structural difference from `'actor`/`'guard`: bare `var` is
    // rebind-only for those (never content-mutable), but a scalar `'atomic` has
    // no separate rebind operation at all, so `var` alone grants no more than
    // `let` — it must ALSO be rejected, not merely fall back to `'actor`/`'guard`'s
    // "rebind-only" semantics.
    let out = emit_rust("def main():\n    var counter'atomic = 0\n    counter += 5\n    print \"{counter}\"\n");
    assert!(!out.status.success(), "expected a bare `var` (no `mut`) 'atomic compound-assign to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not declared `mut`") && stderr.contains("'atomic"), "got:\n{}", stderr);
}

#[test]
fn atomic_bare_var_alone_also_rejects_swap() {
    let out = emit_rust("def main():\n    var counter'atomic = 0\n    let old = counter.swap(100)\n    print \"{old}\"\n");
    assert!(!out.status.success(), "expected a bare `var` (no `mut`) 'atomic `.swap()` to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not declared `mut`") && stderr.contains(".swap()"), "got:\n{}", stderr);
}

#[test]
fn atomic_mut_binding_permits_mutation() {
    let src = "def main():\n    mut counter'atomic = 0\n    counter += 5\n    counter -= 1\n    counter = 10\n    let old = counter.swap(100)\n    print \"{counter} {old}\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected `mut x'atomic` to permit every mutating op, got:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(generated.contains("fetch_add") && generated.contains("fetch_sub") && generated.contains(".store(") && generated.contains(".swap("),
        "expected the full 'atomic operation mapping to be emitted, got:\n{}", generated);
}

#[test]
fn atomic_var_mut_binding_permits_mutation() {
    let src = "def main():\n    var mut counter'atomic = 0\n    counter += 5\n    let old = counter.swap(100)\n    print \"{counter} {old}\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected `var mut x'atomic` to permit every mutating op, got:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(generated.contains("fetch_add") && generated.contains(".swap("), "got:\n{}", generated);
}

// ── Regression: the `'actor`/`'guard` → `'atomic` promotion pass must still ────
// ── carry forward a mutation-permitting binding after the fix above ───────────
//
// Before this fix, a scalar `'actor`/`'guard` local's own compound-assign/`.swap()`
// was gated only by the GENERIC rebind-permission rule (`var` sufficed, `mut`
// alone was actually rejected — see `check_assign_target`'s `BindingKind::Mut`
// arm, "mut is never rebindable") — never by a `'atomic`-style content-mutation
// rule. The promotion pass (`promote_atomic.rs`) rewrites such a local's
// *representation* to `'atomic` (`Arc<AtomicIsize>`/fetch_add/swap) without ever
// re-running the checker against the new representation — so if the new
// `'atomic` mut-gating added above had been applied indiscriminately to every
// name in `var_atomic_types` (rather than skipping `promoted_atomic_vars`), a
// promoted-but-not-`mut`-declared `'actor`/`'guard` local like the one below
// would have started failing to compile — a real regression this test guards
// against. See `try_emit_atomic_method`'s doc comment for the exact exemption.
//
// This is the codegen-inspection half; the true end-to-end compile-and-run half
// (real `boring build` → `cargo run` → stdout comparison) is
// `tests/cases/atomic_promotion_binding_permission_regression.br`, wired up via
// `transpile_test!` in `tests/transpile.rs` — see that fixture's own doc comment.
#[test]
fn promotion_still_permits_mutation_on_a_bare_var_actor_source_after_binding_permission_fix() {
    // Deliberately `var counter'actor` — bare `var`, no `mut`/`var mut` — exactly
    // the shape every pre-existing promotion fixture/test in this file already
    // uses (`promotion_fires_for_local_scalar_actor` above, `tests/cases/
    // atomic_promotion_actor.br`). Promotion criteria all hold (atomic-eligible
    // scalar, never escapes, every access is a recognized single primitive,
    // never used in `with`), so this must build successfully post-fix, covering
    // both a compound-assign AND a `.swap()` on the same promoted name.
    let src = "def main():\n    var counter'actor = 0\n    counter += 5\n    counter -= 2\n    let old = counter.swap(100)\n    print \"{counter} {old}\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected the promoted program to build:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("AtomicIsize") && generated.contains("fetch_add") && generated.contains("fetch_sub") && generated.contains(".swap("),
        "expected the promotion pass to still fire for both compound-assign and `.swap()`, got:\n{}", generated
    );
    assert!(!generated.contains("Mutex<"), "expected no leftover Mutex representation, got:\n{}", generated);
}
