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

// ─── kernel: block — basic execution ────────────────────────────────────────

#[test]
fn test_kernel_block_runs_entry_point() {
    let src = r#"
kernel Identity:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 1.0

let data = [10.0, 20.0]
mut k = Identity(data)
kernel:
    k(block = 2)
let _result = k.buf[0]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Float(10.0));
}

#[test]
fn test_kernel_block_writeback() {
    let src = r#"
kernel Identity:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] + 1.0

let data = [10.0, 20.0]
mut k = Identity(data)
kernel:
    k(block = 2)
let _r0 = k.buf[0]
let _r1 = k.buf[1]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_r0"), Value::Float(11.0));
    assert_eq!(get_var(&interp, "_r1"), Value::Float(21.0));
}

#[test]
fn test_kernel_block_single_element() {
    let src = r#"
kernel Identity:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        buf[0] = buf[0] * 2.0

mut k = Identity([5.0])
kernel:
    k(block = 1)
let _result = k.buf[0]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Float(10.0));
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
kernel:
    k(block = 4)
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
kernel:
    k(block = 3)
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
kernel:
    k(block = 3)
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

// ─── kernel field access after launch ────────────────────────────────────────

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
kernel:
    k(block = 3)
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
kernel:
    k(block = 2)
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
kernel:
    k(block = 3)
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

// ─── end-to-end: element-wise multiply ──────────────────────────────────────

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
kernel:
    k(block = 4)
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
kernel:
    k1(block = 3)
let scaled = k1.buf

mut k2 = Shift(scaled)
kernel:
    k2(block = 3)
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
kernel:
    k(block = 3)
let _result = k.buf
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(
        get_var(&interp, "_result"),
        Value::Array(vec![Value::Float(3.0), Value::Float(6.0), Value::Float(9.0)])
    );
}

#[test]
fn test_gpu_shared_qualifier() {
    let src = r#"
kernel SharedWeight:
    mut [float]'unified  out
    let [float]'sync     weights

    init([float]'unified data, [float]'sync w):
        out     = data
        weights = w

    def ():
        let tid = gpu.thread.x
        out[tid] = out[tid] * weights[0]

let data    = [1.0, 2.0, 4.0]
let weights = [5.0]
mut k = SharedWeight(data, weights)
kernel:
    k(block = 3)
let _result = k.out
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(
        get_var(&interp, "_result"),
        Value::Array(vec![Value::Float(5.0), Value::Float(10.0), Value::Float(20.0)])
    );
}

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
kernel:
    k(block = 3)
let _result = k.out
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(
        get_var(&interp, "_result"),
        Value::Array(vec![Value::Float(0.0), Value::Float(10.0), Value::Float(20.0)])
    );
}

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
kernel:
    k(block = 3)
let _result = k.buf
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(
        get_var(&interp, "_result"),
        Value::Array(vec![Value::Float(1.0), Value::Float(2.0), Value::Float(4.0)])
    );
}

// ─── Screen / Dimension built-ins ────────────────────────────────────────────

#[test]
fn test_dimension_constructor() {
    let src = r#"
let d = Dimension(1024, 768)
let _w = d.width
let _h = d.height
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_w"), Value::Uint(1024));
    assert_eq!(get_var(&interp, "_h"), Value::Uint(768));
}

#[test]
fn test_screen_constructor_positional() {
    let src = r#"
let s = Screen(800, 600)
let _w = s.width
let _h = s.height
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_w"), Value::Uint(800));
    assert_eq!(get_var(&interp, "_h"), Value::Uint(600));
}

#[test]
fn test_screen_constructor_with_dimension() {
    let src = r#"
let s = Screen(Dimension(640, 480), title = "test")
let d = s.dimension
let _w = d.width
let _h = d.height
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_w"), Value::Uint(640));
    assert_eq!(get_var(&interp, "_h"), Value::Uint(480));
}

#[test]
fn test_screen_present_increments_frame() {
    let src = r#"
let screen = Screen(4, 4)
let pixels = [0 for ..16]
screen.present(pixels)
screen.present(pixels)
let _f = screen.frame
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_f"), Value::Uint(2));
}

#[test]
fn test_screen_key_false_by_default() {
    let src = r#"
let screen = Screen(4, 4)
let _pressed = screen.key("q")
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_pressed"), Value::Bool(false));
}

#[test]
fn test_screen_resized_false_initially() {
    let src = r#"
let screen = Screen(4, 4)
let _r = screen.resized
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_r"), Value::Bool(false));
}

// ─── 'surface qualifier + kernel: block ──────────────────────────────────────

#[test]
fn test_surface_qualifier_kernel_field() {
    // The kernel body uses Dimension field access (Object.field) which adds stack frames.
    // Spawn a thread with a larger stack to avoid overflow in test mode (debug build).
    let result = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let src = r#"
kernel Fill:
    mut [uint]'surface pixels
    var Dimension dim

    init(Dimension d):
        pixels = [0 for ..d.width * d.height]
        dim = d

    def ():
        let col = gpu.block.x * gpu.block_dim.x + gpu.thread.x
        let row = gpu.block.y * gpu.block_dim.y + gpu.thread.y
        if col < dim.width and row < dim.height:
            pixels[row * dim.width + col] = 0xFF0000FF

let screen = Screen(Dimension(1, 1))
var k = Fill(screen.dimension)
kernel:
    k(block = (1, 1))
let _p = k.pixels[0]
"#;
            let (interp, result) = run(src);
            result.expect("runtime error");
            // 0xFF0000FF = 4278190335 — stored as Int literal in Boring
            assert_eq!(get_var(&interp, "_p"), Value::Int(4278190335u64 as i64));
        })
        .unwrap()
        .join();
    result.expect("thread panicked");
}

#[test]
fn test_kernel_block_executes_body() {
    let src = r#"
kernel Counter:
    mut [uint]'surface pixels
    var Dimension dim

    init(Dimension d):
        pixels = [0 for ..d.width * d.height]
        dim = d

    def ():
        let col = gpu.block.x * gpu.block_dim.x + gpu.thread.x
        let row = gpu.block.y * gpu.block_dim.y + gpu.thread.y
        if col < dim.width and row < dim.height:
            pixels[row * dim.width + col] = 42

let screen = Screen(Dimension(1, 1))
var k = Counter(screen.dimension)

kernel:
    k(block = (1, 1))
    screen.present(k.pixels)

let _frame = screen.frame
let _px    = k.pixels[0]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_frame"), Value::Uint(1));
    assert_eq!(get_var(&interp, "_px"),    Value::Int(42));
}

#[test]
fn test_kernel_block_loop_break() {
    let src = r#"
kernel Fill:
    mut [uint]'surface pixels
    var Dimension dim
    var int iters

    init(Dimension d):
        pixels = [0 for ..d.width * d.height]
        dim = d
        iters = 0

    def ():
        pixels[0] = uint(iters)

let screen = Screen(Dimension(1, 1))
var k = Fill(screen.dimension)

kernel:
    loop:
        k.iters += 1
        k(block = (1, 1))
        screen.present(k.pixels)
        if k.iters >= 4:
            break

let _frames = screen.frame
let _iters  = k.iters
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_frames"), Value::Uint(4));
    assert_eq!(get_var(&interp, "_iters"),  Value::Int(4));
}

#[test]
fn test_kernel_init_overload_dispatch() {
    let src = r#"
kernel Zoom:
    mut [uint]'surface pixels
    var Dimension dim
    var float zoom

    init(Dimension d, float zoom):
        pixels = [0 for ..d.width * d.height]
        dim    = d
        zoom   = zoom

    init(Zoom prev, Dimension d):
        pixels = [0 for ..d.width * d.height]
        dim    = d
        zoom   = prev.zoom

    def ():
        pixels[0] = 0

var k1 = Zoom(Dimension(2, 2), 3.5)
var k2 = Zoom(k1, Dimension(4, 4))
let _z1 = k1.zoom
let _z2 = k2.zoom
let _w2 = k2.dim.width
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_z1"), Value::Float(3.5));
    assert_eq!(get_var(&interp, "_z2"), Value::Float(3.5));
    assert_eq!(get_var(&interp, "_w2"), Value::Uint(4));
}
