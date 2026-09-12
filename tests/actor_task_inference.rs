// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Codegen tests for the refined `'actor'task`/`'guard'task` inference (see
// docs/qualifiers.md's "Inferring `'actor'task`/`'guard'task` from task-method calls").
//
// These inspect the actual generated Rust text (`boring build --emit-rust`) to confirm
// which concrete lock type (`std::sync::Mutex` vs `tokio::sync::Mutex`) a given parameter's
// signature resolves to — the same style already used by `tests/atomic_qualifier.rs`. A full
// `cargo build` round-trip is deliberately NOT used here: whether the emitted `run`/`with`
// bodies of an *inferred* (not explicitly annotated) 'actor'/'actor'task parameter correctly
// lock/unlock is a separate, pre-existing codegen gap (`seed_param_locals` in emit_top.rs only
// reads a parameter's own declared `Type`, never `self.inferred_qualifiers`) that predates and
// is independent of this file's inference-only changes — confirmed by reproducing the same
// `cargo build` failure on an unmodified checkout using docs/qualifiers.md's own pre-existing
// canonical example. What this file verifies is strictly the INFERENCE DECISION (which lock
// type gets chosen), which is fully determined by the signature text alone.
//
// Run with:
//   cargo test --test actor_task_inference

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
        .join("actor_task_inference_test_scratch")
        .join(format!("{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
    dir
}

/// Returns the single source line declaring `fn_name`'s signature (`fn X(...)` or
/// `async fn X(...)`), panicking with the full generated text if it isn't found —
/// makes a failing assertion's mismatch obvious instead of silently matching nothing.
fn signature_line<'a>(generated: &'a str, fn_name: &str) -> &'a str {
    let needle_async = format!("async fn {}(", fn_name);
    let needle_plain = format!("fn {}(", fn_name);
    generated
        .lines()
        .find(|l| l.contains(&needle_async) || (l.trim_start().starts_with("fn ") && l.contains(&needle_plain)))
        .unwrap_or_else(|| panic!("no signature line found for `{}` in:\n{}", fn_name, generated))
}

fn assert_plain_actor(generated: &str, fn_name: &str) {
    let line = signature_line(generated, fn_name);
    assert!(
        line.contains("std::sync::Mutex") && !line.contains("tokio::sync::Mutex"),
        "expected `{}` to stay plain `'actor` (std::sync::Mutex), got:\n{}",
        fn_name, line
    );
}

fn assert_actor_task(generated: &str, fn_name: &str) {
    let line = signature_line(generated, fn_name);
    assert!(
        line.contains("tokio::sync::Mutex"),
        "expected `{}` to infer `'actor'task` (tokio::sync::Mutex), got:\n{}",
        fn_name, line
    );
}

// ── Regression: the existing "called-method-is-itself-task" heuristic ──────────
//
// Unchanged behavior — docs/qualifiers.md's own canonical example. Must keep resolving
// to 'actor'task exactly as before this file's changes.

#[test]
fn task_method_call_still_infers_actor_task() {
    let src = "\
struct Counter:
    var int value = 0

    task def inc():
        value += 1

def void run(mut Counter c):
    task c.inc()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert_actor_task(&generated, "run");
}

// ── The gap this refinement closes: a `with` block holding an unrelated await ───
//
// `c.inc()` is a plain `def`, not `task` — the OLD heuristic alone would see no signal
// and leave `c` on the plain sync variant, requiring an explicit `'actor'task` annotation
// (exactly the case docs/qualifiers.md used to describe as needing one). The new local
// live-range analysis instead finds the unrelated `wait(...)` inside the same `with`-held
// span and upgrades `c` on its own.

#[test]
fn with_block_holding_unrelated_wait_infers_actor_task() {
    let src = "\
struct Counter:
    var int value = 0

    def inc():
        value += 1

def void run(mut Counter c):
    task:
        with c:
            c.inc()
            wait(Duration.fromMillis(10))

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert_actor_task(&generated, "run");
}

// ── Cross-function propagation via fn_sigs ──────────────────────────────────────
//
// `holdAndWait` awaits while holding its own parameter via a `with` block (no task-method
// call involved) — the same local live-range signal as the test above, but this time on a
// `task def` function's own top-level body (no nested `task:`/closure capture needed at
// all, since the whole function is already async). `caller` never itself awaits or opens
// a `with` block; it only calls `holdAndWait(c)` directly. It must still pick up the
// 'actor'task requirement transitively through `fn_sigs`, purely because it is calling a
// function whose own signature now demands the async lock.

#[test]
fn cross_function_propagation_picks_up_task_requirement() {
    let src = "\
struct Counter:
    var int value = 0

    def inc():
        value += 1

task def void holdAndWait(mut Counter c):
    with c:
        c.inc()
        wait(Duration.fromMillis(10))

task def void caller(mut Counter c):
    holdAndWait(c)

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert_actor_task(&generated, "holdAndWait");
    assert_actor_task(&generated, "caller");
}

// ── The regression the design discussion was most worried about ────────────────
//
// A value passed through several purely-synchronous call layers — none of them ever
// await while holding it — must NEVER be silently promoted to the heavier tokio lock just
// because it escapes its defining scope. `'actor'/'guard' values are used specifically
// *because* they're shared, so "escapes into another function" is the ordinary case, not
// a rare edge case — direction 1 (upgrade whenever an await can't be ruled out) would have
// promoted every one of `bump`/`bumpTwice`/`bumpThrice` here. `Counter` is registered as an
// actor-source type (via `makeCounter`'s return type) purely so these bare parameters have
// a reason to resolve to `'actor` at all, rather than a plain `&mut Counter` auto-ref
// borrow — the promotion-to-'task question this test is about only arises once `'actor`
// is already in play.
#[test]
fn multi_function_always_sync_value_is_never_promoted_to_task() {
    let src = "\
struct Counter:
    var int value = 0
    def inc():
        value += 1
    req int get():
        value

def Counter'actor makeCounter():
    Counter()

def bump(mut Counter c):
    c.inc()

def bumpTwice(mut Counter c):
    bump(c)
    bump(c)

def bumpThrice(mut Counter c):
    bumpTwice(c)
    bump(c)

def main():
    mut c = makeCounter()
    bumpThrice(c)
    print \"{c.get()}\"
";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    for f in ["makeCounter", "bump", "bumpTwice", "bumpThrice"] {
        assert_plain_actor(&generated, f);
    }
    // Belt-and-suspenders: the tokio lock type must not appear anywhere in this program at all.
    assert!(
        !generated.contains("tokio::sync::Mutex"),
        "expected no tokio::sync::Mutex anywhere in a program that never awaits while \
         holding its actor value, got:\n{}", generated
    );
}

// ── The propagation mechanism's known, documented limitation ───────────────────
//
// `caller` calls `mid`, which calls `deep` (the function that actually awaits while
// holding its parameter via a `with` block). All three are declared in this file order.
// `fn_sigs` propagation is itself file-ordered (see docs/qualifiers.md): by the time
// `caller`'s own body is really emitted, `mid`'s signature has not yet been updated with
// the fact that `deep` — declared after `mid` — requires the async lock (that update only
// lands when `mid` is REALLY emitted, which is after `caller`). This is the documented,
// accepted residual gap, not a bug: `caller` must stay on the plain sync variant rather
// than erroring out or guessing async — the fix, when a developer actually hits this, is
// to annotate the parameter explicitly, not for the compiler to guess in the safe-but-
// expensive direction on its own. (`mid` itself, one hop closer to `deep`, does happen to
// pick the requirement up here — the speculative `pre_infer_fn_qualifiers` pass gives it
// one full pass of head start over `caller` — but that is an implementation detail of
// *how far* the propagation reaches today, not something this test depends on.)
#[test]
fn forward_reference_two_hops_away_stays_conservative() {
    let src = "\
struct Counter:
    var int value = 0
    def inc():
        value += 1

def caller(mut Counter c):
    mid(c)

def mid(mut Counter c):
    deep(c)

task def void deep(mut Counter c):
    with c:
        c.inc()
        wait(Duration.fromMillis(10))

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected this program to transpile without error (the residual gap must never \
         surface as a compile error) — stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let generated = String::from_utf8_lossy(&out.stdout);
    // `caller` never becomes an actor-source-typed value here (nothing forces `Counter`
    // into that role the way `multi_function_always_sync_value_is_never_promoted_to_task`
    // does), so it may resolve to a plain `&mut Counter` auto-ref borrow rather than an
    // explicit `'actor` — either is fine; what must never happen is the async lock.
    let caller_line = signature_line(&generated, "caller");
    assert!(
        !caller_line.contains("tokio::sync::Mutex"),
        "expected `caller` to stay conservative (no async lock), got:\n{}", caller_line
    );
    assert_actor_task(&generated, "deep");
}
