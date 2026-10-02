// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: passing a place of the receiver's own object (`self.bump(inner)`,
// `self.bump(self.inner)`, `self.fill(buf)`, `h.bump(h.inner)`) to a `mut` parameter of a method
// of that receiver is accepted by the interpreter but was emitted as `self.bump(&mut self.inner)`
// -- two overlapping `&mut` borrows, rustc E0499 in generated code with no hint at the Boring
// source. The semantic checker now rejects it for `boring run` and `boring build` alike with a
// diagnostic naming the fix. A call on a different receiver (disjoint fields / distinct locals)
// is valid Rust and keeps compiling (tests/cases/lend_disjoint_receiver_ok.br, transpile.rs).
//
// Run with:
//   cargo test --test lend_own_field_build_fails

use std::path::Path;
use std::process::Command;

const CASES: &[(&str, &str, &str)] = &[
    // (case, argument as named in the diagnostic, receiver as named in the diagnostic)
    ("error_lend_own_field_to_self_method", "`inner`", "`self.bump()`"),
    ("error_lend_own_field_explicit_self", "`self.inner`", "`self.bump()`"),
    ("error_lend_own_field_collection", "`buf`", "`self.fill()`"),
    ("error_lend_own_field_through_local", "`h.inner`", "`h.bump()`"),
];

fn check_rejected(args: &[&str], case_br: &Path, arg: &str, call: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let out = Command::new(bin)
        .args(args)
        .arg(case_br)
        .args(if args.contains(&"build") { vec!["--emit-rust"] } else { vec![] })
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "expected `boring {}` to reject {}, but it exited successfully:\n{}",
        args.join(" "), case_br.display(), String::from_utf8_lossy(&out.stdout)
    );
    assert!(!stderr.contains("panicked at"), "must fail with a clean diagnostic — stderr:\n{}", stderr);
    let expected = format!("cannot lend {} to the `mut` parameter", arg);
    assert!(
        stderr.contains(&expected) && stderr.contains(call) && stderr.contains("E0499")
            && stderr.contains("copy it into a local first"),
        "expected stderr to contain `{}` ... {} and name the fix\n--- actual stderr ---\n{}",
        expected, call, stderr
    );
}

#[test]
fn lending_a_field_of_the_receiver_fails_in_build_and_run() {
    for (case, arg, call) in CASES {
        let case_br = Path::new("tests/cases").join(format!("{}.br", case));
        check_rejected(&["build"], &case_br, arg, call);
        check_rejected(&["run"], &case_br, arg, call);
    }
}

#[test]
fn lending_to_a_different_receiver_is_accepted() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/lend_disjoint_receiver_ok.br");
    let out = Command::new(bin)
        .arg("build").arg(case_br).arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        out.status.success(),
        "disjoint-receiver lending must keep compiling — stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
