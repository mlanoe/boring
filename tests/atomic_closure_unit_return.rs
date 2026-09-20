// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a closure literal whose body's last statement is a bare
// `'atomic` compound-assign (`counter += 1`) used to transpile to a single
// trailing Rust expression with no `;` (`|| { counter.fetch_add(1, ..) }`),
// because `emit_expr_closure`'s `Block` arm (src/transpiler/emit_expr.rs)
// unconditionally routes the last statement through `emit_stmt_inline`
// (src/transpiler/emit_methods.rs), which had no notion of a "void" tail
// statement at all — unlike `emit_stmt` (used for ordinary `def` function
// bodies), which always appends `;` to the last statement of a function with
// no declared return type. `fetch_add`'s Rust return value (the *previous*
// atomic value) then became the closure's own inferred return type
// (`isize`, not `()`), which compiles fine as long as nothing constrains the
// closure to a concrete return type — but fails with E0308 ("expected `()`,
// found `isize`") the moment the closure is passed somewhere that demands
// exactly `impl FnMut()`/`impl Fn()`.
//
// Fixed in `emit_stmt_inline`'s `Stmt::Expr` arm: an `Assign`/`QuestionAssign`
// expression (Boring's own compound-assign, plain-store, and `?=` all desugar
// to one of these two `ExprKind`s at parse time) is always statement-like in
// Boring's semantics — never a value the caller reads — so it now always gets
// a trailing `;`, regardless of position in the block.
//
// This is checked two ways:
//   1. Directly on the generated Rust text (the transpiler-level fix).
//   2. By splicing the actual emitted closure into a real `impl FnMut()`
//      call site and compiling it with `rustc` — proving the fix holds up
//      against the exact trait bound the bug report was filed against,
//      without going through `'observed.subscribe()`'s own pre-existing
//      workaround (`try_emit_observed_subscribe` in emit_methods.rs, which
//      forces unit regardless of this fix and must keep doing so).
//
// A comparison case for an `'actor`-backed (Mutex-based) capture is included
// too: that lowering already wrapped its last statement in a block ending in
// `;` before this fix (see `emit_expr_assign`'s Mutex/RwLock field-write
// arms), so it should compile into the same `impl FnMut()` call site with or
// without this change — a guard against the fix accidentally narrowing to
// only the `'atomic` case.
//
// Run with:
//   cargo test --test atomic_closure_unit_return

use std::path::Path;
use std::process::Command;

fn emit_rust(src: &str, dir: &Path) -> String {
    let bin = env!("CARGO_BIN_EXE_boring");
    std::fs::create_dir_all(dir).expect("failed to create scratch dir");
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, src).expect("failed to write fixture .br file");

    let emit = Command::new(bin)
        .arg("build").arg(&br_file).arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to succeed, but it failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    String::from_utf8_lossy(&emit.stdout).into_owned()
}

/// Splices a `takes_fn_mut(impl FnMut())` call site into `generated` in place
/// of its sole `f();` invocation, appends the helper function, writes the
/// result to `<dir>/gen.rs`, and compiles it standalone with `rustc`. Panics
/// (with the full generated source attached) on any failure, including a
/// wrong occurrence count for `f();` — this test must know exactly which
/// call it's replacing, not silently replace zero or several.
fn compile_through_fn_mut_bound(generated: &str, dir: &Path) {
    let occurrences = generated.matches("f();").count();
    assert_eq!(
        occurrences, 1,
        "expected exactly one `f();` call to splice into an `impl FnMut()` \
         call site, found {}:\n{}",
        occurrences, generated
    );
    let spliced = generated.replacen("f();", "takes_fn_mut(f);", 1);
    let spliced = format!(
        "{}\nfn takes_fn_mut(mut cb: impl FnMut()) {{ cb(); }}\n",
        spliced
    );

    std::fs::create_dir_all(dir).expect("failed to create scratch dir");
    let rs_path = dir.join("gen.rs");
    std::fs::write(&rs_path, &spliced).expect("failed to write gen.rs");
    let bin_path = dir.join("gen_bin");

    let rustc = Command::new("rustc")
        .arg("--edition").arg("2021")
        .arg(&rs_path)
        .arg("-o").arg(&bin_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke rustc: {}", e));
    assert!(
        rustc.status.success(),
        "expected the closure spliced into an `impl FnMut()` call site to \
         compile, but it failed:\n--- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&rustc.stderr),
        spliced,
    );

    let run = Command::new(&bin_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run compiled binary: {}", e));
    assert!(run.status.success(), "expected the compiled binary to run successfully");
    let stdout = String::from_utf8_lossy(&run.stdout).trim_end().to_string();
    assert_eq!(stdout, "1", "expected the counter to have been incremented exactly once, got: {}", stdout);
}

#[test]
fn atomic_closure_trailing_compound_assign_is_unit_returning() {
    let src = "def main():\n    var mut counter'atomic = 0\n    let f = ():\n        counter += 1\n    f()\n    print \"{counter}\"\n";
    let dir = Path::new("target/atomic_closure_unit_return_test_scratch/atomic");
    let generated = emit_rust(src, dir);

    // Transpiler-level check: the fetch_add call must be followed by `;`
    // before the closure block's closing brace, not left as a bare tail
    // expression.
    assert!(
        generated.contains("fetch_add(1, std::sync::atomic::Ordering::SeqCst); }"),
        "expected the closure's trailing 'atomic compound-assign to end in `;` \
         (forcing the block to `()`), got:\n{}", generated
    );

    // End-to-end: the exact same emitted closure must satisfy a real
    // `impl FnMut()` bound.
    compile_through_fn_mut_bound(&generated, dir);

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn actor_closure_trailing_compound_assign_is_already_unit_returning() {
    // Comparison case (docs above): an `'actor`-backed capture already wraps
    // its last statement in a `{ ...; }` block, so this must keep compiling
    // into the same `impl FnMut()` call site regardless of this fix.
    let src = "def main():\n    var counter'actor = 0\n    let f = ():\n        counter += 1\n    f()\n    print \"{counter}\"\n";
    let dir = Path::new("target/atomic_closure_unit_return_test_scratch/actor");
    let generated = emit_rust(src, dir);

    assert!(generated.contains("Mutex<"), "expected the Mutex representation for an 'actor capture, got:\n{}", generated);

    compile_through_fn_mut_bound(&generated, dir);

    let _ = std::fs::remove_dir_all(dir);
}
