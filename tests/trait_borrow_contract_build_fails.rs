// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A bare-trait parameter of a trait method is passed by reference (`&dyn Trait`) unless the
// trait writes `Trait'owned` (docs/qualifiers.md, "Design: borrow inference for bare-trait
// parameters", phase 4). An implementer that consumes such a parameter, or that declares it with
// a different type than the trait, is a Boring-level diagnostic naming the fix — not a confusing
// rustc signature mismatch.
//
// Run with:
//   cargo test --test trait_borrow_contract_build_fails

use std::path::Path;
use std::process::Command;

fn build_error(case: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("build")
        .arg(Path::new(case))
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        !out.status.success(),
        "expected `boring build --emit-rust {}` to fail, but it emitted:\n{}",
        case, String::from_utf8_lossy(&out.stdout)
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn implementer_consuming_a_borrowed_trait_param_is_diagnosed() {
    let stderr = build_error("tests/cases/error_trait_contract_consumed.br");
    assert!(
        stderr.contains("`b` is consumed in `A.visit`") && stderr.contains("Builder'owned b"),
        "expected the consumed-parameter diagnostic with the `'owned` fix, got:\n{}", stderr
    );
}

#[test]
fn implementer_param_not_matching_the_trait_contract_is_diagnosed() {
    let stderr = build_error("tests/cases/error_trait_contract_mismatch.br");
    assert!(
        stderr.contains("must be a bare trait type to match trait `Visitor`"),
        "expected the contract-mismatch diagnostic, got:\n{}", stderr
    );
}
