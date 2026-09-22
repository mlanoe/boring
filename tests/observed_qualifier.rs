// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Checker and codegen tests for the `'observed` composable ownership-qualifier
// suffix (see docs/book.md's "'observed" section for the full design: the
// composition table, `.value` field access, `subscribe()`/`Subscription`, and the
// `'shared'observed` rejection).
//
// Behavioral (stdout-equivalence) tests for the four legal compositions, `.value`
// read-only access never notifying, `Subscription`'s `Drop` genuinely unsubscribing,
// and multiple subscribers all firing on one write live in tests/transpile.rs as an
// ordinary `transpile_test!` case (`observed_qualifier`) — those confirm the
// *observable* behavior is correct via a real `cargo build`+run. This file instead
// inspects the actual generated Rust text (`boring build --emit-rust`, fast, no
// `cargo build`) to confirm: the checker rejects `'shared'observed`, the `mut`/`var`
// binding regression documented in the book falls out for free from the existing
// `'actor`/`'guard` row, and bare `'observed` qualifier inference resolves to the
// spec'd defaults.
//
// Run with:
//   cargo test --test observed_qualifier

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
        .join("observed_qualifier_test_scratch")
        .join(format!("{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
    dir
}

// ── Checker: `'shared'observed` compile-error rejection (test category 2) ──────

#[test]
fn shared_observed_is_rejected() {
    let src = "struct Counter:\n    var int value = 0\n\ndef main():\n    let Counter'shared'observed c = Counter(0)\n    print \"{c.value}\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected 'shared'observed to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot combine `'observed` with `'shared`"),
        "expected the observed-compatibility error, got:\n{}", stderr
    );
}

#[test]
fn actor_guard_inline_owned_observed_are_accepted() {
    // Direct call (`c.inc()`, no `.value`) is the primary path — 'actor/'guard
    // already dispatch method calls transparently elsewhere in Boring
    // (docs/book.md §21), and 'observed follows the same convention.
    for base in ["inline", "owned", "actor", "guard"] {
        let src = format!(
            "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n\ndef main():\n    mut Counter'{base}'observed c = Counter(0)\n    c.inc()\n    print \"ok\"\n",
            base = base
        );
        let out = emit_rust(&src);
        assert!(
            out.status.success(),
            "expected 'observed to compose with '{base} to be accepted, got:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

// ── `subscribe()`'s callback signature: `fn (T'observed) callback` ─────────────
//
// The fix this session implements (see the branch's report): `subscribe()`'s
// callback now takes exactly one parameter — a reference to the same observed
// struct (`value` + `subscribers`), supplied fresh at each notification — instead
// of a zero-argument closure that forced a subscriber held long-term to separately
// capture (by reference) the observed object itself, a real Rust lifetime problem
// when `subscribe()` is called from inside a method of the type that owns the
// observed field. These checker/codegen-level tests confirm the generated Rust
// shape directly; the full read-`.value`-off-the-parameter behavioral proof (for
// both an 'actor'observed case where the old capture pattern happened to still work,
// and an 'inline'observed case where it never could have) lives in
// tests/cases/observed_qualifier.br (run via tests/transpile.rs).

#[test]
fn subscribe_callback_takes_one_observed_ref_param() {
    let src = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n    req int current():\n        value\n\ndef main():\n    mut Counter'actor'observed c = Counter(0)\n    let sub = c.subscribe((obj):\n        print \"{obj.value.current()}\"\n    )\n    c.inc()\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected a one-parameter subscribe() callback to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("fn subscribe(&self, callback: impl FnMut(&BoringObserved<V>)"),
        "expected subscribe()'s generated signature to take a &BoringObserved<V> parameter, got:\n{}", generated
    );
    assert!(
        generated.contains("cb(self)"),
        "expected __boring_notify to pass `self` into each stored callback, got:\n{}", generated
    );
}

#[test]
fn subscribe_callback_wrong_arity_is_rejected() {
    // Zero parameters (the OLD signature, before this session's fix) must now be
    // rejected with a clear diagnostic rather than silently emitting a callback that
    // can never satisfy `subscribe()`'s real (one-parameter) signature.
    let src = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n\ndef main():\n    mut Counter'actor'observed c = Counter(0)\n    let sub = c.subscribe(():\n        print \"changed\"\n    )\n    c.inc()\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a zero-parameter subscribe() callback to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("callback must take exactly one parameter"),
        "expected the callback-arity diagnostic, got:\n{}", stderr
    );
}

// ── Binding × qualifier regression (test category 6) ────────────────────────────
//
// `'actor'observed`/`'guard'observed` fall into the *existing* `'actor`/`'guard` row
// of docs/book.md §21's binding table: `var` alone is rebind-only and does not
// unlock a `def` call; `mut`/`var mut` does. No new checker rule was written for
// this — it falls out for free from `crate::ast::binding_grants_mut` (already
// qualifier-shape-agnostic) plus `try_emit_observed_method_direct`'s mut-gating,
// which mirrors `try_emit_mutex_method`/`try_emit_rwlock_method`'s existing rule.
// This test is the regression proving that claim, not assuming it. Checked on the
// direct-call path (the primary one) and the `.value` escape hatch alike — both
// gate on the same binding rule, only whether a *successful* call also notifies
// differs between them.

#[test]
fn var_alone_does_not_unlock_direct_def_call() {
    let src = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n\ndef main():\n    var Counter'actor'observed c = Counter(0)\n    c.inc()\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected `var` alone (no `mut`) to reject a direct def call");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("is not declared `mut`") && stderr.contains(".inc()"),
        "expected the not-declared-mut diagnostic, got:\n{}", stderr
    );
}

#[test]
fn var_alone_does_not_unlock_def_call_through_value_escape_hatch() {
    let src = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n\ndef main():\n    var Counter'actor'observed c = Counter(0)\n    c.value.inc()\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected `var` alone (no `mut`) to reject a def call through the `.value` escape hatch too");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("is not declared `mut`") && stderr.contains(".inc()") && stderr.contains(".value"),
        "expected the not-declared-mut diagnostic naming `.value`, got:\n{}", stderr
    );
}

#[test]
fn var_mut_does_unlock_direct_def_call() {
    let src = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n    req int current():\n        value\n\ndef main():\n    var mut Counter'actor'observed c = Counter(0)\n    c.inc()\n    print \"{c.current()}\"\n";
    let out = emit_rust(&src);
    assert!(
        out.status.success(),
        "expected `var mut` to unlock a direct def call, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn plain_mut_unlocks_direct_def_call() {
    // `mut` (bare, no `var`) also grants content mutation — the fixed-binding half
    // of the same row, distinct from the two rebind-axis cases above.
    let src = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n    req int current():\n        value\n\ndef main():\n    mut Counter'guard'observed c = Counter(0)\n    c.inc()\n    print \"{c.current()}\"\n";
    let out = emit_rust(&src);
    assert!(
        out.status.success(),
        "expected plain `mut` to unlock a direct def call, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ── Bare `'observed` qualifier inference (test category 7) ──────────────────────
//
// Judgment call (see this session's report for the full writeup): bare `'observed`
// is seeded into the *same* bare-struct candidate-elimination pipeline
// (`infer_qualifiers.rs`) as an ordinary unqualified struct local, with `'shared`
// excluded from the candidate set up front (never a legal `'observed` resolution).
// A genuine multi-owner usage signal narrows the candidate set to `{Actor, Guard}`
// (or a singleton) *before* the fallback chain even runs, so it's chosen directly;
// absent such a signal, the untouched existing fallback (size check, then `'owned`
// before `'actor` in the ordered chain) reproduces "resolve 'inline'observed vs
// 'owned'observed via the size threshold" exactly, with no new resolution logic.

#[test]
fn bare_observed_resolves_actor_on_multi_owner_signal() {
    // `spawn_actor(c.value)` demands `Counter'actor` at that parameter position —
    // the same "call site demanding a qualifier" signal an ordinary bare struct local
    // already uses to infer `'actor` today (docs/book.md §30). `boring build
    // --emit-rust` only checks the transpiler's *emitted representation* here (not a
    // full `cargo build`) — passing a `.value`-computed expression as a call argument
    // into a qualifier-demanding parameter is a separate, pre-existing-style gap in
    // the general argument-coercion pipeline (it double-wraps, since the coercer has
    // no static-type knowledge of an arbitrary `.value` expression — the same class of
    // gap as the documented "string literal as external call argument" one in
    // CLAUDE.md) flagged in this session's report, not fixed here; it does not affect
    // the inference decision itself, which is what this test checks.
    let src = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n\ndef spawn_actor(Counter'actor c):\n    print \"spawned\"\n\ndef main():\n    mut Counter'observed c = Counter(0)\n    spawn_actor(c.value)\n    c.value.inc()\n    print \"done\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("BoringObserved<Arc<std::sync::Mutex<Counter>>>"),
        "expected bare 'observed to resolve to 'actor'observed on a multi-owner signal, got:\n{}", generated
    );
}

#[test]
fn bare_observed_resolves_inline_for_small_struct_no_signal() {
    let src = "struct Counter:\n    var int value = 0\n\ndef main():\n    mut Counter'observed c = Counter(0)\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("let mut c: BoringObserved<Counter>"),
        "expected bare 'observed to resolve to 'inline'observed for a small struct with no narrowing signal, got:\n{}", generated
    );
}

// ── Struct fields / parameters / return types (this session's extension) ───────
//
// Full behavioral coverage (construct + subscribe + mutate-via-a-method-reached-
// through-the-field + confirm notification, for a field, a constructor parameter,
// and a return type) lives in tests/cases/observed_qualifier.br (run via
// tests/transpile.rs, a real cargo build+run). These tests instead inspect the
// generated Rust text directly — checker rejections and the bare-`'observed`
// default-to-`'actor'observed` policy for these three new positions.
//
// Judgment call (see this session's report): a bare `'observed` field/parameter/
// return type defaults straight to `'actor'observed`, unlike a bare *local*
// binding (which runs the full usage-based candidate-elimination pipeline just
// above). A field/param/return has no equally narrow, single-body usage signal
// to analyze at its own declaration site — deferred as a simpler, safer default
// for this extension rather than building three new cross-body/cross-call-site
// inference passes.

#[test]
fn shared_observed_is_rejected_on_a_struct_field() {
    let src = "struct Counter:\n    var int value = 0\n\nstruct Holder:\n    mut Counter'shared'observed c = Counter(0)\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected 'shared'observed to be rejected on a struct field");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot combine `'observed` with `'shared`"),
        "expected the observed-compatibility error for a field, got:\n{}", stderr
    );
}

#[test]
fn shared_observed_is_rejected_on_a_parameter() {
    let src = "struct Counter:\n    var int value = 0\n\ndef useCounter(Counter'shared'observed c):\n    print \"ok\"\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected 'shared'observed to be rejected on a parameter");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot combine `'observed` with `'shared`"),
        "expected the observed-compatibility error for a parameter, got:\n{}", stderr
    );
}

#[test]
fn shared_observed_is_rejected_on_a_return_type() {
    let src = "struct Counter:\n    var int value = 0\n\ndef Counter'shared'observed makeCounter():\n    Counter(0)\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected 'shared'observed to be rejected on a return type");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot combine `'observed` with `'shared`"),
        "expected the observed-compatibility error for a return type, got:\n{}", stderr
    );
}

#[test]
fn explicit_actor_observed_struct_field_renders_boring_observed() {
    let src = "struct Counter:\n    var int value = 0\n\nstruct Holder:\n    mut Counter'actor'observed c = Counter(0)\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("c: BoringObserved<Arc<std::sync::Mutex<Counter>>>"),
        "expected an explicit 'actor'observed field to render BoringObserved<Arc<Mutex<Counter>>>, got:\n{}", generated
    );
}

#[test]
fn bare_observed_struct_field_defaults_to_actor() {
    let src = "struct Counter:\n    var int value = 0\n\nstruct Holder:\n    mut Counter'observed c = Counter(0)\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("c: BoringObserved<Arc<std::sync::Mutex<Counter>>>"),
        "expected a bare 'observed field to default to 'actor'observed, got:\n{}", generated
    );
    assert!(
        generated.contains("BoringObserved::new(Arc::new(std::sync::Mutex::new(Counter { value: 0 })))"),
        "expected the field's default-value construction to also wrap it, got:\n{}", generated
    );
}

#[test]
fn bare_observed_parameter_defaults_to_actor() {
    let src = "struct Counter:\n    var int value = 0\n\ndef useCounter(Counter'observed c):\n    print \"ok\"\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("fn useCounter(c: &BoringObserved<Arc<std::sync::Mutex<Counter>>>)"),
        "expected a bare 'observed parameter to default to a by-reference 'actor'observed, got:\n{}", generated
    );
}

#[test]
fn bare_observed_return_type_defaults_to_actor() {
    let src = "struct Counter:\n    var int value = 0\n\ndef Counter'observed makeCounter():\n    Counter(0)\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("fn makeCounter() -> BoringObserved<Arc<std::sync::Mutex<Counter>>>"),
        "expected a bare 'observed return type to default to 'actor'observed, got:\n{}", generated
    );
    assert!(
        generated.contains("BoringObserved::new(Arc::new(std::sync::Mutex::new(Counter { value: 0 })))"),
        "expected the tail constructor call to be wrapped to match, got:\n{}", generated
    );
}

// ── Auto-derive interaction: `BoringObserved<V>` needs its own Clone/Debug/
// PartialEq/Default (test category: Clone-skip regression) ─────────────────────
//
// Judgment call (see this session's report): rather than adding a new recursive
// "does the wrapped struct actually implement Clone" check to `emit_struct.rs`'s
// auto-derive decision (`has_non_clone_field` and friends) — Boring has no such
// recursive check for an ordinary NON-observed nested struct field either, a
// pre-existing, orthogonal gap this extension doesn't need to fix to close its
// own — `BoringObserved<V>` gets its own manual `Clone`/`Debug`/`PartialEq`/
// `Default` impls, each bounded only on `V` (`emit_observed_derived_impls`,
// `mod.rs`). Without these, ANY struct with an `'observed` field of ANY base
// broke its own auto-derived `Debug`/`Clone`/`PartialEq` outright (no impl
// existed on `BoringObserved<V>` at all, for any `V`) — this is the real,
// guaranteed regression risk this test guards, not merely the narrower
// "`'inline'observed`/`'owned'observed`'s `V` might not be `Clone`" case, which
// doesn't need a new checker rule at all: an actually-non-Clone `V` simply fails
// to satisfy `BoringObserved<V>`'s own `impl<V: Clone> Clone` bound, the same
// unremarkable way any other generic field's non-Clone type would.
#[test]
fn struct_with_inline_observed_field_gets_working_clone_derive() {
    let src = "struct Position:\n    var float x = 0.0\n\nstruct Canvas:\n    mut Position'inline'observed origin = Position()\n\ndef main():\n    mut c = Canvas()\n    let c2 = c.clone()\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected a struct with an 'inline'observed field to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("#[derive(Debug, Clone, PartialEq)]\nstruct Canvas {"),
        "expected Canvas to still get Debug/Clone/PartialEq despite its 'inline'observed field, got:\n{}", generated
    );
    assert!(
        generated.contains("impl<V: Clone> Clone for BoringObserved<V>")
            && generated.contains("impl<V: std::fmt::Debug> std::fmt::Debug for BoringObserved<V>")
            && generated.contains("impl<V: PartialEq> PartialEq for BoringObserved<V>"),
        "expected BoringObserved<V>'s own manual Clone/Debug/PartialEq impls to be emitted, got:\n{}", generated
    );
}

#[test]
fn struct_with_actor_observed_field_skips_partial_eq_in_sync_multi_mode() {
    // Mirrors the pre-existing `has_sync_mutex_field` exclusion for a plain (non-
    // observed) 'actor field in non-async multi-thread mode (Arc<Mutex<T>> has no
    // PartialEq) — 'actor'observed has the exact same problem one level down
    // (BoringObserved<Arc<Mutex<T>>>'s manual PartialEq needs V: PartialEq).
    let src = "struct Counter:\n    var int value = 0\n\nstruct Holder:\n    mut Counter'actor'observed c = Counter(0)\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("#[derive(Debug, Clone)]\nstruct Holder {"),
        "expected Holder to get Debug/Clone but NOT PartialEq (Arc<Mutex<T>> has none), got:\n{}", generated
    );
}

// ── Field-level mut/var permission parity (test category: "mut/var field-
// permission regression") ───────────────────────────────────────────────────────
//
// Finding (see this session's report): calling a `def` method through a struct
// field (`self.field.method()`) does NOT currently push a "not declared mut"
// diagnostic for ANY qualifier — not `'actor`/`'guard` alone, and not
// `'actor'observed`/`'guard'observed` either. This is a pre-existing, general gap
// (flagged separately as its own follow-up task, not fixed here) — the analogous
// LOCAL BINDING check (`var_alone_does_not_unlock_direct_def_call` above) DOES
// fire correctly, because that diagnostic lives in `observed_call_expr`, gated on
// `known_local_vars`/`mut_checked_local_vars` (populated only for local bindings),
// and neither `try_emit_mutex_method`/`try_emit_rwlock_method`'s own `self.field`
// branch nor this session's new `try_emit_observed_field_method_direct` add an
// equivalent field-scoped check. This test is therefore a PARITY regression, not
// a rejection: an `'actor'observed` field must compile identically to a plain
// (non-observed) `'actor` field in this same shape — i.e., this extension must
// not make the pre-existing gap any worse, and must not spuriously start
// rejecting a call the un-observed case already accepts either.
//
#[test]
fn non_mut_observed_field_method_call_has_same_permissiveness_as_plain_actor_field() {
    let observed_src = "struct Counter:\n    var int value = 0\n    def inc(): value += 1\n\nstruct Holder:\n    Counter'actor'observed c = Counter(0)\n    def bump(): self.c.inc()\n\ndef main():\n    print \"ok\"\n";
    let plain_src = "struct Counter:\n    var int value = 0\n    def inc(): value += 1\n\nstruct Holder:\n    Counter'actor c = Counter(0)\n    def bump(): self.c.inc()\n\ndef main():\n    print \"ok\"\n";
    let observed_out = emit_rust(observed_src);
    let plain_out = emit_rust(plain_src);
    assert_eq!(
        observed_out.status.success(), plain_out.status.success(),
        "expected a non-mut 'actor'observed field's def-method call through the field to succeed/fail exactly like the non-observed 'actor case (parity, not a new regression) — observed: {}, plain: {}",
        observed_out.status.success(), plain_out.status.success()
    );
}

// A `mut`-declared 'observed field DOES correctly unlock the direct transparent
// call + notification — this is the actual, positive requirement (mirrors the
// local-binding `plain_mut_unlocks_direct_def_call` test above, at field scope).
#[test]
fn mut_observed_field_unlocks_direct_def_call_and_notifies() {
    let src = "struct Counter:\n    var int value = 0\n    def inc(): value += 1\n    req int current(): value\n\nstruct Holder:\n    mut Counter'actor'observed c = Counter(0)\n    def bump():\n        let sub = self.c.subscribe((obj):\n            print \"{obj.value.current()}\"\n        )\n        self.c.inc()\n\ndef main():\n    print \"ok\"\n";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected a mut 'actor'observed field to unlock a direct def call + notify through self.field, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("__boring_notify"),
        "expected the field-scoped direct call to still emit the notify step, got:\n{}", generated
    );
}

#[test]
fn bare_observed_resolves_owned_for_oversized_struct_no_signal() {
    // 33 `int` fields (isize, 8 bytes each on this platform) = 264 bytes, over the
    // default 256-byte `--inline-auto-bytes` threshold.
    let fields: String = (0..33).map(|i| format!("    var int f{} = 0\n", i)).collect();
    let ctor_args: String = (0..33).map(|_| "0".to_string()).collect::<Vec<_>>().join(", ");
    let src = format!(
        "struct Big:\n{fields}\ndef main():\n    mut Big'observed c = Big({ctor_args})\n    print \"ok\"\n",
        fields = fields, ctor_args = ctor_args
    );
    let out = emit_rust(&src);
    assert!(out.status.success(), "expected this program to transpile:\n{}", String::from_utf8_lossy(&out.stderr));
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("let mut c: BoringObserved<Box<Big>>"),
        "expected bare 'observed to resolve to 'owned'observed for an oversized struct with no narrowing signal, got:\n{}", generated
    );
}
