// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `.clone()` on a bare (unqualified, non-`mut`) struct parameter is read-only
// and must not trigger "parameter is immutable but `clone` is a `def` method" (auto-ref
// inference used to classify it as a mutating `def` because it is not in `struct_req_methods`).
// A genuinely mutating `def` method on such a parameter must still be rejected.
//
// Run with:
//   cargo test --test param_clone_read_only

use std::path::Path;
use std::process::Command;

fn emit(case: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("build")
        .arg(Path::new(case))
        .arg("--emit-rust")
        .output()
        .expect("failed to invoke boring")
}

#[test]
fn clone_on_bare_param_compiles() {
    let out = emit("tests/cases/param_clone_read_only.br");
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Read-only for every inference run, so the param is lent consistently (`&Big`) instead of
    // being owned in the signature but treated as a borrow in the body (E0308 at call sites).
    let rust = String::from_utf8_lossy(&out.stdout);
    assert!(rust.contains("fn usek(d: &Big)"), "{rust}");
}

#[test]
fn mutating_def_on_bare_param_still_errors() {
    let out = emit("tests/cases/error_param_def_method_needs_mut.br");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("is immutable but `bump` is a `def` method"), "stderr:\n{stderr}");
}
