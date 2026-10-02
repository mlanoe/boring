// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A bare `var` binding is only *rebindable*; mutating the contents of a built-in collection
// (`[T]`, `{K=V}`, `{T}`) needs `mut` or `var mut` (docs/book.md, "Variables and Mutability").
// The rule lives in the semantic checker, so `boring run` and `boring build --emit-rust` must
// both reject it — before this check, `v.push(2)` on a `var [int] v` compiled and ran.
//
// See `tests/cases/error_var_collection_content_mutation.br`.
//
// Run with:
//   cargo test --test var_collection_content_mut_build_fails

use std::path::Path;
use std::process::Command;

const CASE_ERR: &str = "tests/cases/error_var_collection_content_mutation.br";
const CASE_OK: &str = "tests/cases/var_mut_collection_ok.br";

#[test]
fn var_collection_content_mutation_fails_boring_build() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let emit = Command::new(bin)
        .arg("build").arg(Path::new(CASE_ERR)).arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to reject content mutation of `var` collections, \
         but it succeeded and emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );
    let stderr = String::from_utf8_lossy(&emit.stderr);
    for expected in [
        "`b` is `var` (rebindable only, not content-mutable) — cannot call `.push()` on it; fix: declare it `mut` or `var mut`",
        "`b` is `var` (rebindable only, not content-mutable) — cannot assign to its elements",
        "`v` is `var` (rebindable only, not content-mutable) — cannot call `.push()` on it",
        "`d` is `var` (rebindable only, not content-mutable) — cannot assign to its elements",
        "`s` is `var` (rebindable only, not content-mutable) — cannot call `.add()` on it",
    ] {
        assert!(stderr.contains(expected), "expected stderr to contain:\n{}\n--- actual stderr ---\n{}", expected, stderr);
    }
    // The permitted forms in the same file (`mut` param, `var mut`/`mut` locals) are not flagged.
    for unexpected in ["`w` is", "`m` is", "`a_mut_param`"] {
        assert!(!stderr.contains(unexpected), "unexpected diagnostic `{}` in:\n{}", unexpected, stderr);
    }
}

#[test]
fn mut_and_var_mut_collections_still_transpile() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let emit = Command::new(bin)
        .arg("build").arg(Path::new(CASE_OK)).arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "`var mut`/`mut` collection mutation must still build; stderr:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
}
