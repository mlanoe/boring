// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: returning an `if let` binding (or a `T&` param) borrowed from a struct that
// holds a mutable-trait object (`Box<dyn Trait>`, trait with a `def` method) can't be cloned nor
// moved out of the borrow. It used to emit `Some(x.clone())` -> rustc E0308 with no hint at the
// Boring source; `boring build` now rejects it with a diagnostic naming the fixes. All-`req`
// traits are `'shared` automatically and keep compiling (dyn_return_borrowed_field_req_trait),
// as does the explicit `Trait'shared` spelling (dyn_return_borrowed_field_shared).
//
// Run with:
//   cargo test --test dyn_return_borrowed_build_fails

use std::path::Path;
use std::process::Command;

#[test]
fn returning_a_borrowed_trait_object_is_rejected_by_build() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case = Path::new("tests/cases/error_dyn_return_borrowed_field.br");
    let out = Command::new(bin)
        .args(["build"]).arg(case).arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "expected rejection, got success:\n{}", String::from_utf8_lossy(&out.stdout));
    assert!(!stderr.contains("panicked at"), "must fail cleanly:\n{}", stderr);
    assert!(
        stderr.contains("cannot return `x`") && stderr.contains("Trait'shared") && stderr.contains("all-`req`"),
        "diagnostic must name the value and the fixes:\n{}", stderr
    );
}
