// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Kernel (Rust-for-Linux, no_std) codegen snapshot tests.
//
// These tests verify the text emitted by `boring build --linux`
// without requiring the real Linux kernel build system. Each test:
//   1. Writes a Boring source snippet to a temp file.
//   2. Invokes `boring build --linux <file>`.
//   3. Reads the generated src/lib.rs.
//   4. Asserts that the generated text contains the expected patterns.
//
// Run with:
//   cargo test --test kernel_codegen

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Invoke `boring build --linux <file>` and return the generated
/// src/lib.rs text.
///
/// boring names the output directory `<stem>_linux` next to the source
/// file, so we place the source in a dedicated temp dir and read from there.
fn kernel_codegen(test_name: &str, src: &str) -> String {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("kernel_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // Source file named "test.br" → boring creates "test_linux/" next to it.
    let br_file    = tmp.join("test.br");
    let kernel_dir = tmp.join("test_linux");
    fs::write(&br_file, src).unwrap();

    let result = Command::new(bin)
        .args(["build", "--linux"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{test_name}] failed to invoke boring: {e}"));

    assert!(
        result.status.success(),
        "[{test_name}] boring build --linux failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );

    fs::read_to_string(kernel_dir.join("src/lib.rs")).unwrap_or_default()
}

#[test]
fn linux_backend_is_a_standalone_flag_not_a_compute_target() {
    let bin = env!("CARGO_BIN_EXE_boring");

    for obsolete_target in ["kernel", "linux"] {
        let result = Command::new(bin)
            .args(["build", "--target", obsolete_target])
            .output()
            .unwrap_or_else(|e| panic!("failed to invoke boring: {e}"));

        assert!(
            !result.status.success(),
            "--target {obsolete_target} must not remain accepted"
        );
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains(&format!("unknown target '{obsolete_target}'")),
            "expected an unknown-target diagnostic for --target {obsolete_target}; got:\n{stderr}"
        );
        assert!(
            stderr.contains("use --linux for Rust-for-Linux"),
            "expected the diagnostic to point to --linux; got:\n{stderr}"
        );
    }
}

// ─── while-let self-shorthand loop ──────────────────────────────────────────
//
// Regression for the gap found while porting ecb2690/3a42aa1 (the
// standard-target `while let v:` self-shorthand fix) to the Linux target:
// the kernel transpiler's `Stmt::WhileLet` arm had none of the standard
// transpiler's self-shorthand handling, so it would naively emit
// `while let Some(v) = v { ...; v = next; }` — shadow-colliding the outer
// `Option`-typed `v` with the loop-pattern's own unwrapped `v`.

#[test]
fn while_let_self_shorthand_reassignment_uses_distinct_outer_binding() {
    let rs = kernel_codegen("self_shorthand_reassign", r#"
def int? next_val(int c):
    if c <= 0: return nil
    return c - 1

def void run():
    var int? countdown = next_val(3)
    while let countdown:
        countdown = next_val(countdown)
"#);
    // The outer binding must get a distinct mangled name — the loop-pattern's
    // own unwrapped binding keeps the original name unshadowed-by-codegen.
    assert!(rs.contains("let mut __wl_countdown = countdown;"),
        "expected outer binding to be renamed to __wl_countdown;\ngot:\n{rs}");
    assert!(rs.contains("while let Some(countdown) = __wl_countdown {"),
        "expected loop head to match on the renamed outer binding;\ngot:\n{rs}");
    // The body's reassignment must target the renamed outer binding, not the
    // loop-pattern's own (shadowed) unwrapped `countdown`.
    assert!(rs.contains("__wl_countdown = next_val(countdown);"),
        "expected body reassignment to be redirected to __wl_countdown;\ngot:\n{rs}");
    // Post-loop sync writes the final (`None`) state back to the real binding.
    assert!(rs.contains("countdown = __wl_countdown;"),
        "expected post-loop sync back into countdown;\ngot:\n{rs}");
}

#[test]
fn while_let_self_shorthand_break_refills_outer_binding() {
    let rs = kernel_codegen("self_shorthand_break", r#"
def int? next_val(int c):
    if c <= 0: return nil
    return c - 1

def void run():
    var int? countdown = next_val(5)
    while let countdown:
        if countdown == 2:
            break
        countdown = next_val(countdown)
"#);
    // Reaching `break` must refill the outer binding from the loop-local
    // unwrapped value first, or the post-loop sync reads it back out while
    // still in the moved-from state the `while let Some(name) = outer` match
    // left it in on that iteration (Rust E0382).
    assert!(rs.contains("__wl_countdown = Some(countdown);\n") || rs.contains("__wl_countdown = Some(countdown);"),
        "expected break to refill __wl_countdown before exiting;\ngot:\n{rs}");
    let break_pos = rs.find("break;").expect("expected a break statement");
    let refill_pos = rs.find("__wl_countdown = Some(countdown);").expect("expected refill before break");
    assert!(refill_pos < break_pos,
        "expected the outer-binding refill to precede `break;`;\ngot:\n{rs}");
    assert!(rs.contains("countdown = __wl_countdown;"),
        "expected post-loop sync back into countdown;\ngot:\n{rs}");
}

#[test]
fn while_let_self_shorthand_nested_loop_break_is_unaffected() {
    let rs = kernel_codegen("self_shorthand_nested_break", r#"
def int? next_val(int c):
    if c <= 0: return nil
    return c - 1

def void run():
    var int? countdown = next_val(5)
    while let countdown:
        for i in 0..<10:
            if i == 3:
                break
        countdown = next_val(countdown)
"#);
    // A `break` belonging to the nested `for` loop must never be mistaken
    // for exiting the enclosing while-let-shorthand loop — no refill of the
    // outer binding should be emitted around it.
    let for_pos = rs.find("for i in").expect("expected a for loop");
    let after_for = &rs[for_pos..];
    assert!(!after_for.contains("__wl_countdown = Some(countdown);"),
        "the nested for-loop's break must not trigger the outer-binding refill;\ngot:\n{rs}");
}
