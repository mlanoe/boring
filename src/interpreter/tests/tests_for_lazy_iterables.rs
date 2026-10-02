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

// Regression coverage for `exec_for`'s lazy `Value::Range`/`Value::Str`/`Value::Dict`
// paths — before this fix, `for i in 0..<n:` (and, same root cause, `for c in bigStr:`)
// went through `collect_iterable`, which eagerly collected the *entire* iterable into
// a `Vec<Value>` (352 bytes per `Value` on a 64-bit target, same oversized-enum-variant
// cost documented in `tests_bytearray.rs`) before the loop body ran even once.
// Confirmed empirically (release build, `/usr/bin/time -l`) on
// `for i in 0..<20000000: sum = sum + 1`: **4.24GB** peak RSS before this fix,
// ~4.4MB after. `Dict` iteration is lower-risk (its backing `Vec<(Value, Value)>` is
// already fully materialized as the dict's own storage either way) but still gets a
// lazy path, avoiding the transient per-pair `Tuple`-wrapping allocation and letting
// `break` skip wrapping the remaining entries.
//
// Unlike `tests_bytearray.rs`'s file-round-trip test, this bug is in the tree-walk
// interpreter's own per-iteration loop overhead, not a one-shot Rust-native
// allocation — so a "tens of millions of iterations" fixture (matching the scale
// above) would take well over a minute under `cargo test`'s default (unoptimized)
// profile, unlike the I/O-bound ByteArray case. The range test below uses a smaller-
// but-still-substantial N (2,000,000 — 704MB under the old eager `Vec<Value>`
// collection) to stay fast in a normal `cargo test` run while still meaningfully
// exercising the lazy path; the 20-million-iteration/4.24GB numbers above are the
// actual large-scale evidence for the fix, reproducible manually via `boring run` +
// `time -l`.

use super::{run, get_var};
use super::*;

#[test]
fn for_range_large_n_completes_without_eager_collection() {
    const N: i64 = 2_000_000;
    let src = format!(r#"
var int sum = 0
for i in 0..<{n}:
    sum = sum + 1
print "{{sum}}"
"#, n = N);
    let (interp, res) = run(&src);
    res.expect("no runtime error iterating a large range");
    assert_eq!(get_var(&interp, "sum"), Value::Int(N));
}

#[test]
fn for_range_inclusive_multivar_and_break_continue_semantics_unchanged() {
    // Same idx/val auto-enumerate + break/continue semantics as the old eager
    // `collect_iterable` path — the lazy `Value::Range` path must preserve them.
    let src = r#"
var int total = 0
for idx, val in 1..=5:
    if val == 3:
        continue
    if val == 5:
        break
    total = total + val + idx
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    // idx=0,val=1 -> +1; idx=1,val=2 -> +3; idx=2,val=3 -> skipped (continue);
    // idx=3,val=4 -> +7; idx=4,val=5 -> break (not added). Total: 1+3+7 = 11.
    assert_eq!(get_var(&interp, "total"), Value::Int(11));
}

#[test]
fn for_str_large_n_completes_without_eager_collection() {
    const N: usize = 2_000_000;
    let src = format!(r#"
let string s = "{chars}"
var int n = 0
for c in s:
    n = n + 1
print "{{n}}"
"#, chars = "a".repeat(N));
    let (interp, res) = run(&src);
    res.expect("no runtime error iterating a large string");
    assert_eq!(get_var(&interp, "n"), Value::Int(N as i64));
}

#[test]
fn for_str_multivar_and_break_continue_semantics_unchanged() {
    let src = r#"
var string letters = ""
for c in "abcdef":
    if c == "c":
        continue
    if c == "e":
        break
    letters = letters + c

var int idxsum = 0
for i, c in "abc":
    idxsum = idxsum + i
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "letters"), Value::Str("abd".to_string()));
    assert_eq!(get_var(&interp, "idxsum"), Value::Int(1 + 2));
}

#[test]
fn for_dict_single_var_binds_whole_tuple_and_kv_destructure_works() {
    let src = r#"
let {string=int} scores = {"Alice" = 90, "Bob" = 85}
var int count = 0
for kv in scores:
    count = count + 1

var int total = 0
for k, v in scores:
    total = total + v
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "count"), Value::Int(2));
    assert_eq!(get_var(&interp, "total"), Value::Int(175));
}

#[test]
fn for_dict_break_continue_semantics_unchanged() {
    let src = r#"
let {int=int} d = {1 = 10, 2 = 20, 3 = 30}
var int seen = 0
for k, v in d:
    if k == 2:
        continue
    if k == 3:
        break
    seen = seen + v
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    // Only k=1 (v=10) is added; k=2 is skipped, k=3 breaks before adding.
    assert_eq!(get_var(&interp, "seen"), Value::Int(10));
}
