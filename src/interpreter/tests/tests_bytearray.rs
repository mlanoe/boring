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

// Regression coverage for `Value::ByteArray` — the packed `[uint8]` representation
// that replaced one `Value::Uint8` (352 bytes on a 64-bit target under `boring
// run`, dominated by the interpreter's largest `Value` variants) per byte with a
// real `Vec<u8>`. See `Value::ByteArray`'s own doc comment in `interpreter/mod.rs`.
//
// A tiny fixture wouldn't catch this class of bug at all — the bug is purely
// about the constant-factor memory multiplier at scale, invisible on a 3-element
// array — so `bytearray_fs_readbytes_stays_packed_at_scale` below builds a real
// tens-of-millions-of-elements file and checks both correctness and (via a
// white-box match on the resulting `Value`) that it actually stayed packed.

use super::{run, get_var};
use super::*;
use std::io::Write;

#[test]
fn bytearray_fs_readbytes_stays_packed_at_scale() {
    // "tens of millions of elements", per the regression-test ask this fixes —
    // large enough that the old one-`Value`-per-byte representation (352 bytes
    // each) would need several GB just for this one array.
    const N: usize = 20_000_000;

    let mut path = std::env::temp_dir();
    path.push(format!("boring_bytearray_regression_{}.bin", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).expect("create temp file");
        // Deterministic, non-uniform content so the index/slice checks below
        // aren't trivially satisfied by an all-zero buffer.
        let buf: Vec<u8> = (0..N).map(|i| (i % 256) as u8).collect();
        f.write_all(&buf).expect("write temp file");
    }
    // `Display`-escape backslashes so a Windows temp path survives round-tripping
    // through a Boring string literal.
    let path_str = path.to_str().expect("non-utf8 temp path").replace('\\', "\\\\");

    let src = format!(r#"
let [uint8] bytes = fs.readBytes("{path}")
let _len = bytes.length
let _first = bytes[0]
let _mid = bytes[10000000]
let _last = bytes[bytes.length - 1]
let _slice = bytes[5..<9]
let _sum_check = bytes[300] as int
"#, path = path_str);

    let (interp, res) = run(&src);
    std::fs::remove_file(&path).ok();
    res.expect("no runtime error reading/indexing/slicing a large [uint8] array");

    assert_eq!(get_var(&interp, "_len"), Value::Int(N as i64));
    assert_eq!(get_var(&interp, "_first"), Value::Uint8(0));
    assert_eq!(get_var(&interp, "_mid"), Value::Uint8((10_000_000i64 % 256) as u8));
    assert_eq!(get_var(&interp, "_last"), Value::Uint8(((N - 1) % 256) as u8));
    assert_eq!(get_var(&interp, "_sum_check"), Value::Int(300 % 256));

    // The actual regression guard: confirm the array is still the packed
    // representation, not one `Value` per byte.
    match get_var(&interp, "bytes") {
        Value::ByteArray(bytes) => assert_eq!(bytes.len(), N),
        other => panic!(
            "expected fs.readBytes()'s [uint8] result to stay a packed Value::ByteArray, got {}",
            other.type_name(),
        ),
    }
    let sliced = get_var(&interp, "_slice");
    match &sliced {
        Value::ByteArray(s) => assert_eq!(s.as_slice(), &[5u8, 6, 7, 8]),
        other => panic!("expected a slice of a ByteArray to stay a ByteArray, got {}", other.type_name()),
    }
}

#[test]
fn bytearray_uint8_typed_literal_and_comprehension_pack() {
    let src = r#"
let [uint8] lit = [1, 2, 3]
let [uint8] comp = [uint8(i % 5) for i in 0..<4]
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert!(matches!(get_var(&interp, "lit"), Value::ByteArray(_)),
        "a `[uint8]`-typed array literal should be coerced into a packed ByteArray");
    assert!(matches!(get_var(&interp, "comp"), Value::ByteArray(_)),
        "a `[uint8]`-typed comprehension result should be coerced into a packed ByteArray");
}

#[test]
fn bytearray_push_index_assign_and_equality() {
    // Pushed/assigned values are explicitly cast (`uint8(...)`) — a bare integer
    // literal argument stays an untyped `Value::Int` (method-call args aren't
    // coerced to the receiver's declared element type), which would legitimately
    // degrade the array to a mixed `Value::Array`, same as it always has for
    // `push`'s existing permissiveness. That degrade path is intentional (see
    // `call_bytearray_method`'s `push` arm) and not what this test is checking.
    let src = r#"
var [uint8] buf = []
buf.push(uint8(10))
buf.push(uint8(20))
buf.push(uint8(30))
buf[1] = uint8(99)
let _len = buf.length
let _eq_true = (buf == [uint8(10), uint8(99), uint8(30)])
let _eq_false = (buf == [uint8(1), uint8(2), uint8(3)])
"#;
    let (interp, res) = run(src);
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_len"), Value::Int(3));
    assert_eq!(get_var(&interp, "_eq_true"), Value::Bool(true));
    assert_eq!(get_var(&interp, "_eq_false"), Value::Bool(false));
    assert!(matches!(get_var(&interp, "buf"), Value::ByteArray(_)),
        "push/index-assign on a [uint8] array should keep it a packed ByteArray");
}

#[test]
fn bytearray_write_bytes_roundtrip() {
    let mut path = std::env::temp_dir();
    path.push(format!("boring_bytearray_roundtrip_{}.bin", std::process::id()));
    let path_str = path.to_str().expect("non-utf8 temp path").replace('\\', "\\\\");

    let src = format!(r#"
let [uint8] original = [1, 2, 3, 250, 251, 252]
fs.writeBytes("{path}", original)
let roundtrip = fs.readBytes("{path}")
let _matches = (roundtrip == original)
"#, path = path_str);

    let (interp, res) = run(&src);
    std::fs::remove_file(&path).ok();
    res.expect("no runtime error");
    assert_eq!(get_var(&interp, "_matches"), Value::Bool(true));
}
