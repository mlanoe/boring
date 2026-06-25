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

use super::{run, get_var};
use super::*;

// ─── kernel struct declaration ───────────────────────────────────────────────

#[test]
fn test_kernel_decl_registers_in_env() {
    let src = r#"
kernel Scale:
    mut [float]'unified buf
    let int'unified     n

    init([float]'unified data, int size):
        buf = data
        n   = size
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = interp.global.borrow().get("Scale");
    assert!(matches!(val, Some(Value::KernelStruct { .. })), "Scale should be a KernelStruct");
}

// ─── kernel instantiation ────────────────────────────────────────────────────

#[test]
fn test_kernel_instantiation() {
    let src = r#"
kernel Scale:
    mut [float]'unified buf
    let int'unified     n

    init([float]'unified data, int size):
        buf = data
        n   = size

let data = [1.0, 2.0, 3.0]
let _result = Scale(data, 3)
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    assert!(matches!(val, Value::Object(_)), "Scale() should return an Object");
    if let Value::Object(inner) = val {
        let fields = inner.borrow().fields.clone();
        let n = fields.iter().find(|(k, _)| k == "n").map(|(_, v)| v.clone());
        assert_eq!(n, Some(Value::Int(3)));
    }
}

// ─── kernel launch and KernelHandle.wait ────────────────────────────────────

#[test]
fn test_kernel_launch_returns_handle() {
    let src = r#"
kernel Identity:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let _tid = gpu.thread.x

let data = [10.0, 20.0]
mut k = Identity(data)
let _result = kernel(block = 2) k
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    assert!(matches!(val, Value::KernelHandle { .. }), "kernel(...) should return KernelHandle");
}

#[test]
fn test_kernel_handle_wait_returns_object() {
    let src = r#"
kernel Identity:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let _tid = gpu.thread.x

let data = [10.0, 20.0]
mut k = Identity(data)
let h = kernel(block = 2) k
let _result = h.wait
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    assert!(matches!(val, Value::Object(_)), ".wait should return the kernel Object");
}

#[test]
fn test_kernel_handle_done_returns_true() {
    let src = r#"
kernel Identity:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let _tid = gpu.thread.x

mut k = Identity([1.0])
let h = kernel(block = 1) k
let _result = h.done()
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Bool(true));
}

// ─── kernel simulation — element-wise operation ──────────────────────────────

#[test]
fn test_kernel_scale_elementwise() {
    let src = r#"
kernel Scale:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0, 3.0, 4.0]
mut k = Scale(data)
mut k = kernel(block = 4) k |> .wait
let _result = k.buf
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    assert_eq!(
        val,
        Value::Array(vec![
            Value::Float(2.0),
            Value::Float(4.0),
            Value::Float(6.0),
            Value::Float(8.0),
        ]),
        "each element should be doubled"
    );
}

#[test]
fn test_kernel_scale_three_elements() {
    let src = r#"
kernel Scale3:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 3.0

let data = [2.0, 4.0, 6.0]
mut k = Scale3(data)
mut k = kernel(block = 3) k |> .wait
let _result = k.buf
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    assert_eq!(
        val,
        Value::Array(vec![
            Value::Float(6.0),
            Value::Float(12.0),
            Value::Float(18.0),
        ])
    );
}

// ─── gpu.thread.x builtin ────────────────────────────────────────────────────

#[test]
fn test_gpu_thread_x_builtin() {
    let src = r#"
kernel RecordTid:
    mut [int]'unified tids

    init(int n):
        tids = [0, 0, 0]

    def ():
        let tid = gpu.thread.x
        tids[tid] = tid

mut k = RecordTid(3)
mut k = kernel(block = 3) k |> .wait
let _result = k.tids
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    assert_eq!(
        val,
        Value::Array(vec![Value::Int(0), Value::Int(1), Value::Int(2)]),
        "tids[i] should equal i (thread index)"
    );
}

// ─── kernel field access after wait ─────────────────────────────────────────

#[test]
fn test_kernel_field_access_after_wait() {
    let src = r#"
kernel AddOne:
    mut [float]'unified arr
    let int'unified n

    init(int size, [float]'unified data):
        n   = size
        arr = data

    def ():
        let tid = gpu.thread.x
        if tid < n:
            arr[tid] = arr[tid] + 1.0

let data = [0.0, 1.0, 2.0]
mut k = AddOne(3, data)
mut k = kernel(block = 3) k |> .wait
let _a0 = k.arr[0]
let _a1 = k.arr[1]
let _a2 = k.arr[2]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_a0"), Value::Float(1.0));
    assert_eq!(get_var(&interp, "_a1"), Value::Float(2.0));
    assert_eq!(get_var(&interp, "_a2"), Value::Float(3.0));
}

// ─── kernel method called from entry point ───────────────────────────────────

#[test]
fn test_kernel_helper_method() {
    let src = r#"
kernel WithHelper:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def apply(int i):
        buf[i] = buf[i] * 10.0

    def ():
        let tid = gpu.thread.x
        apply(tid)

let data = [1.0, 2.0]
mut k = WithHelper(data)
mut k = kernel(block = 2) k |> .wait
let _result = k.buf
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    assert_eq!(
        val,
        Value::Array(vec![Value::Float(10.0), Value::Float(20.0)])
    );
}

// ─── sync is a no-op ────────────────────────────────────────────────────────

#[test]
fn test_sync_is_noop() {
    let src = r#"
kernel WithSync:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] + 1.0
        sync
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0, 3.0]
mut k = WithSync(data)
mut k = kernel(block = 3) k |> .wait
let _result = k.buf
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
    assert_eq!(
        val,
        Value::Array(vec![Value::Float(4.0), Value::Float(6.0), Value::Float(8.0)])
    );
}

// ─── pipe |> .wait syntax ───────────────────────────────────────────────────

#[test]
fn test_pipe_dot_wait_syntax() {
    let src = r#"
kernel PipeTest:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] + 100.0

let data = [1.0, 2.0]
mut k = PipeTest(data)
mut k = kernel(block = 2) k |> .wait
let _result = k.buf[0]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Float(101.0));
}

// ─── end-to-end: element-wise multiply (dot product inputs) ─────────────────

#[test]
fn test_elementwise_multiply_full_program() {
    let src = r#"
kernel Multiply:
    mut [float]'unified a
    mut [float]'unified b
    mut [float]'unified out

    init([float]'unified xs, [float]'unified ys):
        a   = xs
        b   = ys
        out = [0.0, 0.0, 0.0, 0.0]

    def ():
        let tid = gpu.thread.x
        out[tid] = a[tid] * b[tid]

let xs = [1.0, 2.0, 3.0, 4.0]
let ys = [4.0, 3.0, 2.0, 1.0]
mut k = Multiply(xs, ys)
mut k = kernel(block = 4) k |> .wait
let _r0 = k.out[0]
let _r1 = k.out[1]
let _r2 = k.out[2]
let _r3 = k.out[3]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    // 1*4=4, 2*3=6, 3*2=6, 4*1=4
    assert_eq!(get_var(&interp, "_r0"), Value::Float(4.0));
    assert_eq!(get_var(&interp, "_r1"), Value::Float(6.0));
    assert_eq!(get_var(&interp, "_r2"), Value::Float(6.0));
    assert_eq!(get_var(&interp, "_r3"), Value::Float(4.0));
}

// ─── end-to-end: two-kernel pipeline (scale then shift) ─────────────────────

#[test]
fn test_two_kernel_pipeline() {
    let src = r#"
kernel Scale:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

kernel Shift:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] + 10.0

let data = [1.0, 2.0, 3.0]
mut k1 = Scale(data)
mut k1 = kernel(block = 3) k1 |> .wait
let scaled = k1.buf

mut k2 = Shift(scaled)
mut k2 = kernel(block = 3) k2 |> .wait
let _r0 = k2.buf[0]
let _r1 = k2.buf[1]
let _r2 = k2.buf[2]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    // scale: [2, 4, 6], shift: [12, 14, 16]
    assert_eq!(get_var(&interp, "_r0"), Value::Float(12.0));
    assert_eq!(get_var(&interp, "_r1"), Value::Float(14.0));
    assert_eq!(get_var(&interp, "_r2"), Value::Float(16.0));
}

// ─── memory qualifiers ───────────────────────────────────────────────────────

// 'global — device-global writeable buffer (simulation: behaves like 'unified)
#[test]
fn test_gpu_global_qualifier() {
    let src = r#"
kernel GlobalScale:
    mut [float]'global buf

    init([float]'global data):
        buf = data

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 3.0

let data = [1.0, 2.0, 3.0]
mut k = GlobalScale(data)
mut k = kernel(block = 3) k |> .wait
let _result = k.buf
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(
        get_var(&interp, "_result"),
        Value::Array(vec![Value::Float(3.0), Value::Float(6.0), Value::Float(9.0)])
    );
}

// 'shared — block-shared SRAM; used as a per-block lookup table
#[test]
fn test_gpu_shared_qualifier() {
    let src = r#"
kernel SharedWeight:
    mut [float]'unified  out
    let [float]'shared   weights

    init([float]'unified data, [float]'shared w):
        out     = data
        weights = w

    def ():
        let tid = gpu.thread.x
        out[tid] = out[tid] * weights[0]

let data    = [1.0, 2.0, 4.0]
let weights = [5.0]
mut k = SharedWeight(data, weights)
mut k = kernel(block = 3) k |> .wait
let _result = k.out
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(
        get_var(&interp, "_result"),
        Value::Array(vec![Value::Float(5.0), Value::Float(10.0), Value::Float(20.0)])
    );
}

// 'local — per-thread scratch; simulation runs sequentially so the last
//           thread's scratch value is visible in the field after launch.
#[test]
fn test_gpu_local_qualifier() {
    let src = r#"
kernel LocalScratch:
    mut [float]'unified out
    mut float'local     scratch

    init(int n):
        out     = [0.0, 0.0, 0.0]
        scratch = 0.0

    def ():
        let tid = gpu.thread.x
        scratch = float(tid) * 10.0
        out[tid] = scratch

mut k = LocalScratch(3)
mut k = kernel(block = 3) k |> .wait
let _result = k.out
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    // Each thread writes its own scratch then stores it — result is thread-local
    assert_eq!(
        get_var(&interp, "_result"),
        Value::Array(vec![Value::Float(0.0), Value::Float(10.0), Value::Float(20.0)])
    );
}

// 'const — read-only constant memory; all threads read the same value
#[test]
fn test_gpu_const_qualifier() {
    let src = r#"
kernel ConstScale:
    mut [float]'unified buf
    let float'const     factor

    init([float]'unified data, float f):
        buf    = data
        factor = f

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * factor

let data = [2.0, 4.0, 8.0]
mut k = ConstScale(data, 0.5)
mut k = kernel(block = 3) k |> .wait
let _result = k.buf
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(
        get_var(&interp, "_result"),
        Value::Array(vec![Value::Float(1.0), Value::Float(2.0), Value::Float(4.0)])
    );
}
