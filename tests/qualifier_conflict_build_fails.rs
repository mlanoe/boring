// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `infer_qualifiers.rs` used to report a real, detected
// qualifier conflict (a local variable whose candidate set is narrowed to
// empty by two mutually-exclusive usage signals) via `eprintln!` instead of
// `self.push_error(...)`. Since only `self.errors` (populated by
// `push_error`) makes `boring build` fail (see `main.rs`'s
// `if !out.errors.is_empty() { ... process::exit(1) }`), the conflict text
// was printed to stderr but the build still exited 0 and emitted valid,
// running Rust with no qualifier wrapper on the offending variable at all
// (`emit_top.rs::emit_param` falls back to the bare type when no qualifier
// was inferred) — a silently-broken build, not a caught error.
//
// tests/cases/error_qualifier_conflict_inline_actor.br's `c` is constrained
// to `{Actor, Guard}` (it holds the result of a call to a function declared
// to return `Counter'actor`) and then passed to a function whose parameter is
// declared `Counter'inline` (compatible set: `{Inline}` only) — the
// intersection is empty, the exact "0 candidates remaining" conflict this
// bug swallowed.
//
// Run with:
//   cargo test --test qualifier_conflict_build_fails

use std::path::Path;
use std::process::Command;

#[test]
fn qualifier_conflict_fails_boring_build() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/error_qualifier_conflict_inline_actor.br");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));

    assert!(
        !emit.status.success(),
        "expected `boring build --emit-rust` to fail on a real, detected qualifier \
         conflict ('inline vs 'actor), but it exited successfully and emitted:\n{}",
        String::from_utf8_lossy(&emit.stdout)
    );

    let stderr = String::from_utf8_lossy(&emit.stderr);
    let expected = "`c` has no valid qualifier — usage constraints are incompatible";
    assert!(
        stderr.contains(expected),
        "expected stderr to contain:\n{}\n--- actual stderr ---\n{}",
        expected, stderr
    );
}
