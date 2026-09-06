// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of Boring.
// Boring is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// See the LICENSE file at the project root for the full text.

//! Regression tests for the parser's recursion-depth guard.
//!
//! `MAX_EXPR_DEPTH` (`crate::parser::MAX_EXPR_DEPTH`, currently 200) is meant to
//! bound recursive-descent parsing against a crafted input that recurses deep
//! enough to overflow the Rust call stack. Several real recursive paths used to
//! bypass the guard entirely (or never had one): no-paren closure bodies, named
//! call arguments, nested types, nested blocks/statements, and nested patterns.
//! Each test below generates an input, programmatically, that nests far past
//! the limit and asserts parsing fails with a clean `ParseError` — not a
//! process crash (stack overflow) and not a silent success.

use crate::lexer::lex;
use crate::parser::parse;

/// Depth used by every test here: comfortably past `MAX_EXPR_DEPTH` (200) so
/// the guard is guaranteed to trip, but small enough to keep the test fast.
const N: usize = 2000;

/// Runs `parse` on a worker thread with a large stack, mirroring the 256 MB
/// stack `main.rs` spawns for every real `boring` invocation (see its
/// `STACK_SIZE` comment). `cargo test`'s own per-test thread stack is far
/// smaller (a few MB), which is plenty for the depth guard itself but not
/// for the ~200 legitimate recursion levels it allows through before
/// tripping — without this, the test would overflow the *test harness*
/// thread's stack while still under the guard's own limit, which would be a
/// false failure unrelated to the bug being guarded against.
fn assert_rejected_cleanly(src: &str) {
    const STACK_SIZE: usize = 256 * 1024 * 1024; // 256 MB, matches main.rs
    let src = src.to_string();
    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(move || {
            let tokens = lex(&src).expect("lex should succeed — depth is a parser concern, not a lexer one");
            parse(tokens)
        })
        .expect("failed to spawn worker thread");
    match handle.join().expect("parser thread panicked instead of returning a clean error") {
        Ok(_) => panic!("expected a clean parse error for deeply nested input, got Ok"),
        Err(e) => {
            let msg = e.msg();
            assert!(
                msg.contains("nested too deeply"),
                "expected a 'nested too deeply' parse error, got: {msg}"
            );
        }
    }
}

/// `parse_closure_body` (no-paren closure `ident: body`) calls `self.parse_or()`
/// directly, bypassing `parse_expr`'s old chokepoint entirely. A chain of
/// nested single-param closures used to recurse unbounded.
#[test]
fn closure_body_chain_is_bounded() {
    let src = format!("let f = {}0", "a: ".repeat(N));
    assert_rejected_cleanly(&src);
}

/// `parse_arg`'s named-argument value (`ident= expr`) also calls
/// `self.parse_or()` directly. Nested calls each passing a named argument
/// recurse through this exact bypassed path.
#[test]
fn named_arg_chain_is_bounded() {
    let src = format!("let v = {}0{}", "g(x= ".repeat(N), ")".repeat(N));
    assert_rejected_cleanly(&src);
}

/// `parse_type` had no depth guard at all — deeply nested array types
/// (`[[[...Int...]]]`) used to recurse unbounded. Uses a `let` type
/// annotation (matching the audit's own repro) rather than a function
/// parameter: `parse_param`'s speculative type-then-fallback-to-bare-name
/// logic deliberately swallows a `parse_type` error to retry as an untyped
/// name, which would mask the specific "nested too deeply" message (while
/// still failing cleanly, just with a different message) — `parse_let_stmt`
/// calls `parse_type` directly with no such fallback.
#[test]
fn type_nesting_is_bounded() {
    let src = format!("let {}int{} x = []\n", "[".repeat(N), "]".repeat(N));
    assert_rejected_cleanly(&src);
}

/// `parse_block`/`parse_stmt` had no depth guard — deeply nested `if` blocks
/// used to recurse unbounded (confirmed by the audit to SIGABRT in real runs).
#[test]
fn block_nesting_is_bounded() {
    let mut src = String::new();
    for i in 0..N {
        src.push_str(&"    ".repeat(i));
        src.push_str("if true:\n");
    }
    src.push_str(&"    ".repeat(N));
    src.push_str("0\n");
    assert_rejected_cleanly(&src);
}

/// `parse_pattern` had no depth guard — deeply nested tuple/variant patterns
/// in a `match` arm used to recurse unbounded.
#[test]
fn pattern_nesting_is_bounded() {
    let src = format!("match x:\n    {}y{}: 1\n", "(".repeat(N), ")".repeat(N));
    assert_rejected_cleanly(&src);
}

/// `resolve_interp` used to instantiate a fresh `Parser` (starting back at
/// `depth: 0`) for every string-interpolation hole, discarding the enclosing
/// parser's recursion-depth counter entirely. A string interpolated inside
/// another interpolated string — nested thousands of levels deep — could
/// therefore recurse well past `MAX_EXPR_DEPTH` (in fact straight through the
/// real call stack) without the guard ever tripping, since each sub-parser
/// only ever saw itself at depth 1.
#[test]
fn nested_string_interpolation_is_bounded() {
    // Builds `"{"{"{...1...}"}"}"`, N levels deep. Parsing the outermost string
    // literal recurses into a fresh sub-parser per hole (one level per wrap);
    // the fix makes each sub-parser inherit the enclosing depth so the guard
    // trips instead of recursing unbounded.
    let mut nested = "1".to_string();
    for _ in 0..N {
        nested = format!("\"{{{}}}\"", nested);
    }
    let src = format!("let x = {}\n", nested);
    assert_rejected_cleanly(&src);
}
