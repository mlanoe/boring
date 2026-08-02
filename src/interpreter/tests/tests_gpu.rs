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
        ].into()),
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
        ].into())
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
        Value::Array(vec![Value::Int(0), Value::Int(1), Value::Int(2)].into()),
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
        Value::Array(vec![Value::Float(10.0), Value::Float(20.0)].into())
    );
}

// ─── unassigned fixed-size 'actor fields default to zero, not Nil ───────────

#[test]
fn test_unassigned_actor_fixed_array_defaults_to_zero() {
    // Per gpu-module.md's "no init() assignment needed": a `[T, N]'actor` field
    // the kernel's `init()` never touches must still be a real (zero-filled)
    // array inside the kernel body, matching what every real GPU target does
    // (WGSL `var<workgroup>` / CUDA `__shared__` are hardware zero-initialized)
    // — not `Nil`, which would hard-error on the first `tile[i] = ...`.
    let src = r#"
kernel Tile:
    mut [float]'unified   input
    mut [float]'unified   out
    mut [float, 4]'actor  tile

    init([float]'unified data):
        input = data
        out   = [0.0, 0.0, 0.0, 0.0]

    def ():
        let tid = gpu.thread.x
        out[tid] = tile[tid] + input[tid]

let data = [1.0, 2.0, 3.0, 4.0]
mut k = Tile(data)
kernel:
    k(block = 4)
let _result = k.out
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    // tile[tid] defaults to 0.0, so out == input unchanged.
    assert_eq!(
        val,
        Value::Array(vec![Value::Float(1.0), Value::Float(2.0), Value::Float(3.0), Value::Float(4.0)].into())
    );
}

// ─── sync is a no-op (for kernels with no `'actor` field) ───────────────────
//
// `sync` is a REAL cross-thread barrier when the kernel has an `'actor` field
// (see `test_sync_barrier_cross_thread_visibility` below) — `kernel_barrier`
// is only populated on kernels that declare one. A kernel with none, like
// this one, never gets a barrier at all, so `sync` really is still a no-op
// here — this test is about that narrower case, not `sync` in general.

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
        Value::Array(vec![Value::Float(4.0), Value::Float(6.0), Value::Float(8.0)].into())
    );
}

// ─── real cross-thread `'actor` barrier visibility ──────────────────────────

#[test]
fn test_sync_barrier_cross_thread_visibility() {
    // Every thread writes its own tile slot, then (after a real barrier) reads
    // slot 0 — written by a DIFFERENT thread. Only correct if `sync` actually
    // blocks every thread in the block until all writes have landed, and if
    // `tile` reads observe the shared (not per-thread-private) backing array.
    let src = r#"
kernel Broadcast:
    mut [float]'unified    out
    mut [float, 256]'actor tile

    init([float]'unified o):
        out = o

    def ():
        let tid = gpu.thread.x
        tile[tid] = float(tid) + 1.0
        sync
        out[tid] = tile[0]

var [float] data = []
for i in 0..256:
    data.push(0.0)
mut k = Broadcast(data)
kernel:
    k(block = 256)
let _out0 = k.out[0]
let _out255 = k.out[255]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_out0"), Value::Float(1.0));
    assert_eq!(get_var(&interp, "_out255"), Value::Float(1.0));
}

// ─── explicit `grid =` must be honored, not silently re-inferred ───────────

#[test]
fn test_explicit_grid_not_overridden_by_length_inference() {
    // A dispatch that gives BOTH `block =` and `grid =` explicitly used to be
    // routed through the "`k(block = N)` short-hand" path regardless, which
    // ignores `grid` entirely and infers it from the longest array-typed
    // field instead. Here the `'global` input `big` is deliberately longer
    // than `block * the-grid-that-should-be-used`, so a silently-inferred
    // grid would dispatch MORE threads than intended and corrupt `count`
    // (every extra thread increments it once more than expected).
    let src = r#"
kernel MarkThreads:
    let [float]'global  big
    mut [float]'unified marks

    init([float]'global b, [float]'unified m):
        big = b
        marks = m

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        marks[i] = 1.0

var [float] big = []
for i in 0..1000:
    big.push(0.0)
var [float] marks = []
for i in 0..1024:
    marks.push(0.0)
mut k = MarkThreads(big, marks)
kernel:
    k(block = 256, grid = 1)
let _mark_255 = k.marks[255]
let _mark_256 = k.marks[256]
let _mark_1023 = k.marks[1023]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    // block=256, grid=1 (explicit) -> exactly 256 threads (indices 0..255
    // marked). Length-inference from `big` (1000 elements) would instead give
    // ceil(1000/256)=4 blocks — 1024 threads, marking indices up to 1023 too.
    assert_eq!(get_var(&interp, "_mark_255"), Value::Float(1.0));
    assert_eq!(get_var(&interp, "_mark_256"), Value::Float(0.0));
    assert_eq!(get_var(&interp, "_mark_1023"), Value::Float(0.0));
}

// ─── 3D block/grid dispatch: gpu.thread.z / gpu.block.z are real ───────────

#[test]
fn test_3d_block_and_grid_dispatch() {
    let src = r#"
kernel Sum3D:
    mut [float]'unified out

    init([float]'unified o):
        out = o

    def ():
        let idx = gpu.thread.z + gpu.block.z * gpu.block_dim.z
        out[idx] = float(idx) * 10.0

var [float] data = []
for i in 0..4:
    data.push(0.0)
mut k = Sum3D(data)
kernel:
    k(block = (1, 1, 2), grid = (1, 1, 2))
let _r0 = k.out[0]
let _r1 = k.out[1]
let _r2 = k.out[2]
let _r3 = k.out[3]
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_r0"), Value::Float(0.0));
    assert_eq!(get_var(&interp, "_r1"), Value::Float(10.0));
    assert_eq!(get_var(&interp, "_r2"), Value::Float(20.0));
    assert_eq!(get_var(&interp, "_r3"), Value::Float(30.0));
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
        Value::Array(vec![Value::Float(3.0), Value::Float(6.0), Value::Float(9.0)].into())
    );
}

#[test]
fn test_gpu_shared_qualifier() {
    let src = r#"
kernel SharedWeight:
    mut [float]'unified  out
    let [float]'actor    weights

    init([float]'unified data, [float]'actor w):
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
        Value::Array(vec![Value::Float(5.0), Value::Float(10.0), Value::Float(20.0)].into())
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
        Value::Array(vec![Value::Float(0.0), Value::Float(10.0), Value::Float(20.0)].into())
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
        Value::Array(vec![Value::Float(1.0), Value::Float(2.0), Value::Float(4.0)].into())
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

// ─── MatMul kernel — Whisper attention building block ────────────────────────
//
// Computes out = a * b where:
//   a is [rows × inner] stored row-major in a flat [float]
//   b is [inner × cols] stored row-major in a flat [float]
//   out is [rows × cols] stored row-major in a flat [float]
//
// Each GPU thread computes one output element: out[row, col].
// In simulation mode threads run sequentially; block = rows * cols.

#[test]
fn test_kernel_matmul_2x2() {
    // a = [[1, 2], [3, 4]]  b = [[5, 6], [7, 8]]
    // expected = [[19, 22], [43, 50]]
    let src_str = r#"
kernel MatMul:
    let [float]'global  a
    let [float]'global  b
    mut [float]'unified out
    let int rows
    let int cols
    let int inner

    init([float]'global a, [float]'global b, [float]'unified out, int rows, int cols, int inner):
        a     = a
        b     = b
        out   = out
        rows  = rows
        cols  = cols
        inner = inner

    def ():
        let tid = gpu.thread.x
        let row = tid / cols
        let col = tid % cols
        if row < rows and col < cols:
            var float acc = 0.0
            for k in 0..inner:
                acc += a[row * inner + k] * b[k * cols + col]
            out[row * cols + col] = acc

let a   = [1.0, 2.0, 3.0, 4.0]
let b   = [5.0, 6.0, 7.0, 8.0]
var out = [0.0, 0.0, 0.0, 0.0]
mut k = MatMul(a, b, out, 2, 2, 2)
kernel:
    k(block = 4)
let _r00 = k.out[0]
let _r01 = k.out[1]
let _r10 = k.out[2]
let _r11 = k.out[3]
"#;
    let src_str = src_str.to_string();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let (interp, result) = run(&src_str);
            result.expect("runtime error");
            assert_eq!(get_var(&interp, "_r00"), Value::Float(19.0));
            assert_eq!(get_var(&interp, "_r01"), Value::Float(22.0));
            assert_eq!(get_var(&interp, "_r10"), Value::Float(43.0));
            assert_eq!(get_var(&interp, "_r11"), Value::Float(50.0));
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn test_kernel_matmul_2x3() {
    // a = [[1, 2, 3], [4, 5, 6]]  (2×3)
    // b = [[7, 8], [9, 10], [11, 12]]  (3×2)
    // expected = [[58, 64], [139, 154]]
    let src_str = r#"
kernel MatMul:
    let [float]'global  a
    let [float]'global  b
    mut [float]'unified out
    let int rows
    let int cols
    let int inner

    init([float]'global a, [float]'global b, [float]'unified out, int rows, int cols, int inner):
        a     = a
        b     = b
        out   = out
        rows  = rows
        cols  = cols
        inner = inner

    def ():
        let tid = gpu.thread.x
        let row = tid / cols
        let col = tid % cols
        if row < rows and col < cols:
            var float acc = 0.0
            for k in 0..inner:
                acc += a[row * inner + k] * b[k * cols + col]
            out[row * cols + col] = acc

let a   = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
let b   = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
var out = [0.0, 0.0, 0.0, 0.0]
mut k = MatMul(a, b, out, 2, 2, 3)
kernel:
    k(block = 4)
let _r00 = k.out[0]
let _r01 = k.out[1]
let _r10 = k.out[2]
let _r11 = k.out[3]
"#;
    let src_str = src_str.to_string();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let (interp, result) = run(&src_str);
            result.expect("runtime error");
            assert_eq!(get_var(&interp, "_r00"), Value::Float(58.0));
            assert_eq!(get_var(&interp, "_r01"), Value::Float(64.0));
            assert_eq!(get_var(&interp, "_r10"), Value::Float(139.0));
            assert_eq!(get_var(&interp, "_r11"), Value::Float(154.0));
        })
        .unwrap()
        .join()
        .unwrap();
}

// ─── Softmax kernel — Whisper attention scores ───────────────────────────────
//
// GPU softmax in two phases:
//   Phase 1 (CPU): compute max_val and sum_exp — reductions over the full vector.
//   Phase 2 (GPU): each thread computes out[i] = exp(x[i] - max_val) / sum_exp.
//
// This matches the pattern used in production (cuDNN, whisper.cpp) for
// short sequences where the reduction cost on CPU is negligible.

#[test]
fn test_kernel_softmax_uniform() {
    // All inputs equal → all outputs equal to 1/n
    let src_str = r#"
kernel Softmax:
    let [float]'global  x
    mut [float]'unified out
    let int n
    let float max_val
    let float sum_exp

    init([float]'global x, [float]'unified out, int n, float max_val, float sum_exp):
        x       = x
        out     = out
        n       = n
        max_val = max_val
        sum_exp = sum_exp

    def ():
        let i = gpu.thread.x
        if i < n:
            out[i] = exp(x[i] - max_val) / sum_exp

let x       = [2.0, 2.0, 2.0, 2.0]
var out     = [0.0, 0.0, 0.0, 0.0]
let max_val = 2.0
let sum_exp = 4.0
mut k = Softmax(x, out, 4, max_val, sum_exp)
kernel:
    k(block = 4)
let _r0 = k.out[0]
let _r1 = k.out[1]
let _r2 = k.out[2]
let _r3 = k.out[3]
"#;
    let src_str = src_str.to_string();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let (interp, result) = run(&src_str);
            result.expect("runtime error");
            let check = |v: Value| {
                if let Value::Float(f) = v { (f - 0.25).abs() < 1e-9 }
                else { false }
            };
            assert!(check(get_var(&interp, "_r0")), "expected 0.25");
            assert!(check(get_var(&interp, "_r1")), "expected 0.25");
            assert!(check(get_var(&interp, "_r2")), "expected 0.25");
            assert!(check(get_var(&interp, "_r3")), "expected 0.25");
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn test_kernel_softmax_peaked() {
    // x = [0.0, 0.0, 10.0, 0.0] — almost all mass on index 2
    let src_str = r#"
kernel Softmax:
    let [float]'global  x
    mut [float]'unified out
    let int n
    let float max_val
    let float sum_exp

    init([float]'global x, [float]'unified out, int n, float max_val, float sum_exp):
        x       = x
        out     = out
        n       = n
        max_val = max_val
        sum_exp = sum_exp

    def ():
        let i = gpu.thread.x
        if i < n:
            out[i] = exp(x[i] - max_val) / sum_exp

let x       = [0.0, 0.0, 10.0, 0.0]
var out     = [0.0, 0.0, 0.0, 0.0]
let max_val = 10.0
let e10     = exp(-10.0)
let sum_exp = 3.0 * e10 + 1.0
mut k = Softmax(x, out, 4, max_val, sum_exp)
kernel:
    k(block = 4)
let _peak  = k.out[2]
let _other = k.out[0]
"#;
    let src_str = src_str.to_string();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let (interp, result) = run(&src_str);
            result.expect("runtime error");
            if let Value::Float(peak) = get_var(&interp, "_peak") {
                assert!(peak > 0.999, "peak should be near 1, got {}", peak);
            } else { panic!("expected float"); }
            if let Value::Float(other) = get_var(&interp, "_other") {
                assert!(other < 0.001, "other should be near 0, got {}", other);
            } else { panic!("expected float"); }
        })
        .unwrap()
        .join()
        .unwrap();
}

// ─── Scaled dot-product attention ────────────────────────────────────────────
//
// attention(Q, K, V) = softmax(Q × Kᵀ / sqrt(d_k)) × V
//
// Three kernels chained with CPU reductions between phases 1 and 2:
//   1. ScaledQKt  — Q × Kᵀ * scale  (GPU)
//   2. RowSoftmax — row-wise softmax (CPU reductions + GPU element-wise)
//   3. MatMul     — weights × V      (GPU)
//
// Test: seq_len=2, d_k=2, d_v=2, scale=1.0
//   Q=[[1,1],[1,1]]  K=[[1,0],[0,1]]  V=[[1,2],[3,4]]
//   scores = [[1,1],[1,1]]  →  weights = [[0.5,0.5],[0.5,0.5]]
//   output = [[2,3],[2,3]]

#[test]
fn test_kernel_scaled_dot_product_attention() {
    let src_str = r#"
kernel ScaledQKt:
    let [float]'global  q
    let [float]'global  k
    mut [float]'unified out
    let int seq_len
    let int d_k
    let float scale

    init([float]'global q, [float]'global k, [float]'unified out, int seq_len, int d_k, float scale):
        q       = q
        k       = k
        out     = out
        seq_len = seq_len
        d_k     = d_k
        scale   = scale

    def ():
        let tid = gpu.thread.x
        let row = tid / seq_len
        let col = tid % seq_len
        if row < seq_len and col < seq_len:
            var float acc = 0.0
            for i in 0..d_k:
                acc += q[row * d_k + i] * k[col * d_k + i]
            out[row * seq_len + col] = acc * scale

kernel RowSoftmax:
    let [float]'global  scores
    mut [float]'unified weights
    let [float]'global  max_vals
    let [float]'global  sum_exps
    let int seq_len

    init([float]'global scores, [float]'unified weights, [float]'global max_vals, [float]'global sum_exps, int seq_len):
        scores   = scores
        weights  = weights
        max_vals = max_vals
        sum_exps = sum_exps
        seq_len  = seq_len

    def ():
        let tid = gpu.thread.x
        let row = tid / seq_len
        let col = tid % seq_len
        if row < seq_len and col < seq_len:
            weights[row * seq_len + col] = exp(scores[row * seq_len + col] - max_vals[row]) / sum_exps[row]

kernel MatMul:
    let [float]'global  a
    let [float]'global  b
    mut [float]'unified out
    let int rows
    let int cols
    let int inner

    init([float]'global a, [float]'global b, [float]'unified out, int rows, int cols, int inner):
        a     = a
        b     = b
        out   = out
        rows  = rows
        cols  = cols
        inner = inner

    def ():
        let tid = gpu.thread.x
        let row = tid / cols
        let col = tid % cols
        if row < rows and col < cols:
            var float acc = 0.0
            for k in 0..inner:
                acc += a[row * inner + k] * b[k * cols + col]
            out[row * cols + col] = acc

let seq_len = 2
let d_k     = 2
let d_v     = 2
let scale   = 1.0
let q = [1.0, 1.0, 1.0, 1.0]
let k = [1.0, 0.0, 0.0, 1.0]
let v = [1.0, 2.0, 3.0, 4.0]

var scores = [0.0, 0.0, 0.0, 0.0]
mut qkt = ScaledQKt(q, k, scores, seq_len, d_k, scale)
kernel:
    qkt(block = 4)

let max_vals = [max([qkt.out[r * seq_len + c] for c in 0..seq_len]) for r in 0..seq_len]
let sum_exps = [sum([exp(qkt.out[r * seq_len + c] - max_vals[r]) for c in 0..seq_len]) for r in 0..seq_len]

var weights = [0.0, 0.0, 0.0, 0.0]
mut sm = RowSoftmax(qkt.out, weights, max_vals, sum_exps, seq_len)
kernel:
    sm(block = 4)

var attn_out = [0.0, 0.0, 0.0, 0.0]
mut mm = MatMul(sm.weights, v, attn_out, seq_len, d_v, seq_len)
kernel:
    mm(block = 4)

let _o00 = mm.out[0]
let _o01 = mm.out[1]
let _o10 = mm.out[2]
let _o11 = mm.out[3]
"#;
    let src_str = src_str.to_string();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let (interp, result) = run(&src_str);
            result.expect("runtime error");
            assert_eq!(get_var(&interp, "_o00"), Value::Float(2.0));
            assert_eq!(get_var(&interp, "_o01"), Value::Float(3.0));
            assert_eq!(get_var(&interp, "_o10"), Value::Float(2.0));
            assert_eq!(get_var(&interp, "_o11"), Value::Float(3.0));
        })
        .unwrap()
        .join()
        .unwrap();
}

// ─── Layer norm kernel — Whisper residual stream ──────────────────────────────
//
// y[i] = weight[i] * (x[i] - mean) / sqrt(variance + eps) + bias[i]
//
// CPU: compute mean and variance (reductions).
// GPU: normalize and affine-transform element-wise.

#[test]
fn test_kernel_layer_norm_identity_weights() {
    // weight=1 bias=0 → out = (x - mean) / sqrt(variance + eps)
    // x=[1,2,3,4] mean=2.5 variance=1.25
    let src_str = r#"
kernel LayerNorm:
    let [float]'global  x
    let [float]'global  weight
    let [float]'global  bias
    mut [float]'unified out
    let int n
    let float mean
    let float variance
    let float eps

    init([float]'global x, [float]'global weight, [float]'global bias, [float]'unified out, int n, float mean, float variance, float eps):
        x        = x
        weight   = weight
        bias     = bias
        out      = out
        n        = n
        mean     = mean
        variance = variance
        eps      = eps

    def ():
        let i = gpu.thread.x
        if i < n:
            let norm = (x[i] - mean) / sqrt(variance + eps)
            out[i] = weight[i] * norm + bias[i]

let x      = [1.0, 2.0, 3.0, 4.0]
let weight = [1.0, 1.0, 1.0, 1.0]
let bias   = [0.0, 0.0, 0.0, 0.0]
let n      = 4
let mean     = sum(x) / float(n)
let variance = sum([(v - mean) * (v - mean) for v in x]) / float(n)
let eps      = 0.00001
var out = [0.0, 0.0, 0.0, 0.0]
mut k = LayerNorm(x, weight, bias, out, n, mean, variance, eps)
kernel:
    k(block = 4)
let _o0 = k.out[0]
let _o1 = k.out[1]
let _o2 = k.out[2]
let _o3 = k.out[3]
"#;
    let src_str = src_str.to_string();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let (interp, result) = run(&src_str);
            result.expect("runtime error");
            let expected = [-1.3416354199689269_f64, -0.447211806656309,
                             0.447211806656309,   1.3416354199689269];
            for (i, &exp) in expected.iter().enumerate() {
                let name = ["_o0", "_o1", "_o2", "_o3"][i];
                if let Value::Float(got) = get_var(&interp, name) {
                    assert!((got - exp).abs() < 1e-9, "{}: expected {}, got {}", name, exp, got);
                } else { panic!("{}: expected float", name); }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn test_kernel_layer_norm_with_affine() {
    // weight=[2,2,2,2] bias=[1,1,1,1] → out = 2*norm + 1
    let src_str = r#"
kernel LayerNorm:
    let [float]'global  x
    let [float]'global  weight
    let [float]'global  bias
    mut [float]'unified out
    let int n
    let float mean
    let float variance
    let float eps

    init([float]'global x, [float]'global weight, [float]'global bias, [float]'unified out, int n, float mean, float variance, float eps):
        x        = x
        weight   = weight
        bias     = bias
        out      = out
        n        = n
        mean     = mean
        variance = variance
        eps      = eps

    def ():
        let i = gpu.thread.x
        if i < n:
            let norm = (x[i] - mean) / sqrt(variance + eps)
            out[i] = weight[i] * norm + bias[i]

let x      = [1.0, 2.0, 3.0, 4.0]
let weight = [2.0, 2.0, 2.0, 2.0]
let bias   = [1.0, 1.0, 1.0, 1.0]
let n      = 4
let mean     = sum(x) / float(n)
let variance = sum([(v - mean) * (v - mean) for v in x]) / float(n)
let eps      = 0.00001
var out = [0.0, 0.0, 0.0, 0.0]
mut k = LayerNorm(x, weight, bias, out, n, mean, variance, eps)
kernel:
    k(block = 4)
let _o0 = k.out[0]
let _o3 = k.out[3]
"#;
    let src_str = src_str.to_string();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let (interp, result) = run(&src_str);
            result.expect("runtime error");
            if let Value::Float(o0) = get_var(&interp, "_o0") {
                assert!((o0 - (-1.6832708399378538)).abs() < 1e-9, "o0={}", o0);
            } else { panic!("expected float"); }
            if let Value::Float(o3) = get_var(&interp, "_o3") {
                assert!((o3 - 3.6832708399378538).abs() < 1e-9, "o3={}", o3);
            } else { panic!("expected float"); }
        })
        .unwrap()
        .join()
        .unwrap();
}

// ─── GELU kernel — Whisper FFN activation ────────────────────────────────────
//
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
//
// gelu() is a free Boring function called from inside the kernel entry point.
// This tests that free functions are accessible from kernel def () bodies.

#[test]
fn test_kernel_gelu() {
    let src_str = r#"
float gelu(float x):
    let c = 0.7978845608028654
    0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)))

kernel GeluKernel:
    let [float]'global  x
    mut [float]'unified out
    let int n

    init([float]'global x, [float]'unified out, int n):
        x   = x
        out = out
        n   = n

    def ():
        let i = gpu.thread.x
        if i < n:
            out[i] = gelu(x[i])

let x = [0.0, 1.0, -1.0, 2.0]
var out = [0.0, 0.0, 0.0, 0.0]
mut k = GeluKernel(x, out, 4)
kernel:
    k(block = 4)
let _o0 = k.out[0]
let _o1 = k.out[1]
let _o2 = k.out[2]
let _o3 = k.out[3]
"#;
    let src_str = src_str.to_string();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let (interp, result) = run(&src_str);
            result.expect("runtime error");
            // gelu(0) = 0
            assert_eq!(get_var(&interp, "_o0"), Value::Float(0.0));
            // gelu(1) ≈ 0.8412
            if let Value::Float(v) = get_var(&interp, "_o1") {
                assert!((v - 0.8411919906082768).abs() < 1e-9, "gelu(1)={}", v);
            } else { panic!("expected float"); }
            // gelu(-1) ≈ -0.1588  (antisymmetric around 0)
            if let Value::Float(v) = get_var(&interp, "_o2") {
                assert!((v - (-0.15880800939172324)).abs() < 1e-9, "gelu(-1)={}", v);
            } else { panic!("expected float"); }
            // gelu(2) ≈ 1.9546
            if let Value::Float(v) = get_var(&interp, "_o3") {
                assert!((v - 1.954597694087775).abs() < 1e-9, "gelu(2)={}", v);
            } else { panic!("expected float"); }
        })
        .unwrap()
        .join()
        .unwrap();
}

// ─── FFN kernel — Whisper feed-forward network ────────────────────────────────
//
// FFN(x) = W2 · GELU(W1 · x + b1) + b2
//
// Three kernels chained:
//   1. LinearBias      — W1·x + b1
//   2. GeluActivation  — element-wise GELU
//   3. LinearBias      — W2·h + b2
//
// Test: W1=W2=identity, b1=b2=0 → FFN(x) = GELU(x) element-wise.

#[test]
fn test_kernel_ffn() {
    let src_str = r#"
float gelu(float x):
    let c = 0.7978845608028654
    0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)))

kernel LinearBias:
    let [float]'global  x
    let [float]'global  w
    let [float]'global  b
    mut [float]'unified out
    let int rows
    let int cols
    let int inner

    init([float]'global x, [float]'global w, [float]'global b, [float]'unified out, int rows, int cols, int inner):
        x     = x
        w     = w
        b     = b
        out   = out
        rows  = rows
        cols  = cols
        inner = inner

    def ():
        let tid = gpu.thread.x
        let row = tid / cols
        let col = tid % cols
        if row < rows and col < cols:
            var float acc = 0.0
            for k in 0..inner:
                acc += x[row * inner + k] * w[k * cols + col]
            out[row * cols + col] = acc + b[col]

kernel GeluActivation:
    let [float]'global  x
    mut [float]'unified out
    let int n

    init([float]'global x, [float]'unified out, int n):
        x   = x
        out = out
        n   = n

    def ():
        let i = gpu.thread.x
        if i < n:
            out[i] = gelu(x[i])

let seq_len = 2
let d_model = 2
let d_ff    = 2
let x  = [1.0, 2.0, 3.0, 4.0]
let w1 = [1.0, 0.0, 0.0, 1.0]
let b1 = [0.0, 0.0]
let w2 = [1.0, 0.0, 0.0, 1.0]
let b2 = [0.0, 0.0]

var h = [0.0, 0.0, 0.0, 0.0]
mut lin1 = LinearBias(x, w1, b1, h, seq_len, d_ff, d_model)
kernel:
    lin1(block = 4)

var h2 = [0.0, 0.0, 0.0, 0.0]
mut act = GeluActivation(lin1.out, h2, 4)
kernel:
    act(block = 4)

var ffn_out = [0.0, 0.0, 0.0, 0.0]
mut lin2 = LinearBias(act.out, w2, b2, ffn_out, seq_len, d_model, d_ff)
kernel:
    lin2(block = 4)

let _o0 = lin2.out[0]
let _o1 = lin2.out[1]
let _o2 = lin2.out[2]
let _o3 = lin2.out[3]
"#;
    let src_str = src_str.to_string();
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let (interp, result) = run(&src_str);
            result.expect("runtime error");
            let expected = [
                0.8411919906082768_f64,
                1.954597694087775,
                2.996362607918227,
                3.9999297540518075,
            ];
            for (i, &exp) in expected.iter().enumerate() {
                let name = ["_o0", "_o1", "_o2", "_o3"][i];
                if let Value::Float(got) = get_var(&interp, name) {
                    assert!((got - exp).abs() < 1e-9, "{}: expected {}, got {}", name, exp, got);
                } else { panic!("{}: expected float", name); }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
