// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Metal codegen snapshot tests.
//
// These tests verify the text emitted by `boring build --target metal` without
// requiring a real Metal GPU or macOS.  Each test:
//   1. Writes a Boring source snippet to a temp file.
//   2. Invokes `boring build --target metal <file>`.
//   3. Reads the generated kernels/main.metal and src/main.rs.
//   4. Asserts that the generated text contains the expected patterns.
//
// Run with:
//   cargo test --test metal_codegen

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Invoke `boring build --target metal <file>` and return the generated
/// (kernels/main.metal, src/main.rs) text pair.
///
/// boring names the output directory `<stem>_metal` next to the source file,
/// so we place the source in a dedicated temp dir and read from there.
fn metal_codegen(test_name: &str, src: &str) -> (String, String) {
    let (msl, rs, _toml) = run_metal(test_name, src);
    (msl, rs)
}

fn cargo_toml(test_name: &str, src: &str) -> String {
    let (_, _, toml) = run_metal(test_name, src);
    toml
}

fn run_metal(test_name: &str, src: &str) -> (String, String, String) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("metal_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // Source file named "test.br" → boring creates "test_metal/" next to it.
    let br_file   = tmp.join("test.br");
    let metal_dir = tmp.join("test_metal");
    fs::write(&br_file, src).unwrap();

    let result = Command::new(bin)
        .args(["build", "--target", "metal"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{test_name}] failed to invoke boring: {e}"));

    assert!(
        result.status.success(),
        "[{test_name}] boring build --target metal failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let read = |rel: &str| fs::read_to_string(metal_dir.join(rel)).unwrap_or_default();
    (
        read("kernels/main.metal"),
        read("src/main.rs"),
        read("Cargo.toml"),
    )
}

// ─── MSL header ──────────────────────────────────────────────────────────────

#[test]
fn msl_header_includes_metal_stdlib() {
    let (msl, _) = metal_codegen("msl_header", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("#include <metal_stdlib>"),
        "expected #include <metal_stdlib>;\ngot:\n{msl}");
    assert!(msl.contains("using namespace metal;"),
        "expected using namespace metal;\ngot:\n{msl}");
}

// ─── device — kernel signature ───────────────────────────────────────────────

#[test]
fn device_unified_field_becomes_device_ptr_buffer() {
    let (msl, _) = metal_codegen("unified_ptr", r#"
kernel Scale:
    mut [float32]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("device float* buf [[buffer(0)]]"),
        "expected device float* buf [[buffer(0)]];\ngot:\n{msl}");
}

#[test]
fn device_global_field_becomes_device_ptr_buffer() {
    let (msl, _) = metal_codegen("global_ptr", r#"
kernel G:
    mut [float32]'global buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] + 1.0
"#);
    assert!(msl.contains("device float* buf [[buffer(0)]]"),
        "expected device float* buf [[buffer(0)]] for 'global;\ngot:\n{msl}");
}

#[test]
fn device_entry_point_has_kernel_attribute() {
    let (msl, _) = metal_codegen("kernel_attr", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("kernel void Scale_kernel("),
        "expected 'kernel void Scale_kernel(' entry point;\ngot:\n{msl}");
}

#[test]
fn device_const_scalar_becomes_constant_ptr_with_deref() {
    let (msl, _) = metal_codegen("const_scalar", r#"
kernel C:
    mut [float32]'unified buf
    let float32'const     factor
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * factor
"#);
    assert!(msl.contains("constant float* __factor [[buffer("),
        "expected constant float* __factor [[buffer(N)]];\ngot:\n{msl}");
    assert!(msl.contains("const float factor = *__factor;"),
        "expected deref of __factor into const local;\ngot:\n{msl}");
}

#[test]
fn device_shared_dynamic_becomes_threadgroup_ptr() {
    let (msl, _) = metal_codegen("shared_dynamic", r#"
kernel S:
    mut [float32]'unified out
    let [float32]'actor  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(msl.contains("threadgroup float* scratch [[threadgroup(0)]]"),
        "expected threadgroup pointer param for dynamic 'actor;\ngot:\n{msl}");
    // dynamic 'actor must NOT appear as a device buffer param
    assert!(!msl.contains("device float* scratch"),
        "dynamic 'actor must not appear as device buffer;\ngot:\n{msl}");
}

#[test]
fn device_shared_static_declared_in_body() {
    let (msl, _) = metal_codegen("shared_static", r#"
kernel S:
    mut [float32]'unified out
    let [float32, 32]'actor tile
    def ():
        let tid = gpu.thread.x
        out[tid] = tile[0]
"#);
    assert!(msl.contains("threadgroup float tile[32];"),
        "expected threadgroup T name[N] for static 'actor;\ngot:\n{msl}");
    // static 'actor must not appear as a threadgroup param
    assert!(!msl.contains("tile [[threadgroup("),
        "static 'actor must not appear as threadgroup param;\ngot:\n{msl}");
}

#[test]
fn device_local_fixed_array_declared_in_body() {
    let (msl, _) = metal_codegen("local_array", r#"
kernel L:
    mut [float32]'unified out
    let [float32, 8]'local tmp
    def ():
        let tid = gpu.thread.x
        out[tid] = tmp[0]
"#);
    assert!(msl.contains("float tmp[8];"),
        "expected fixed-size local array in body;\ngot:\n{msl}");
}

// ─── device — built-in position parameters ───────────────────────────────────

#[test]
fn device_builtin_position_params_present() {
    let (msl, _) = metal_codegen("builtin_params", r#"
kernel B:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("[[thread_position_in_threadgroup]]"),
        "expected [[thread_position_in_threadgroup]];\ngot:\n{msl}");
    assert!(msl.contains("[[threadgroup_position_in_grid]]"),
        "expected [[threadgroup_position_in_grid]];\ngot:\n{msl}");
    assert!(msl.contains("[[threads_per_threadgroup]]"),
        "expected [[threads_per_threadgroup]];\ngot:\n{msl}");
    assert!(msl.contains("[[threadgroups_per_grid]]"),
        "expected [[threadgroups_per_grid]];\ngot:\n{msl}");
}

#[test]
fn device_gpu_warp_builtins_map_correctly() {
    let (msl, _) = metal_codegen("gpu_warp_builtins", r#"
kernel W:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        let lane = gpu.warp.lane
        let size = gpu.warp.size
        gpu.warp.sync()
        let a = gpu.warp.shuffle_down(buf[tid], 1)
        let b = gpu.warp.shuffle_up(buf[tid], 1)
        let c = gpu.warp.shuffle_xor(buf[tid], 1)
        let d = gpu.warp.shuffle(buf[tid], 0)
        buf[tid] = a + b + c + d + lane + size
"#);
    assert!(msl.contains("[[thread_index_in_simdgroup]]"),
        "expected [[thread_index_in_simdgroup]];\ngot:\n{msl}");
    assert!(msl.contains("[[threads_per_simdgroup]]"),
        "expected [[threads_per_simdgroup]];\ngot:\n{msl}");
    assert!(msl.contains("__simd_lane_id"), "expected __simd_lane_id;\ngot:\n{msl}");
    assert!(msl.contains("__simd_size"), "expected __simd_size;\ngot:\n{msl}");
    assert!(msl.contains("simdgroup_barrier(mem_flags::mem_none)"),
        "expected simdgroup_barrier;\ngot:\n{msl}");
    assert!(msl.contains("simd_shuffle_down("), "expected simd_shuffle_down;\ngot:\n{msl}");
    assert!(msl.contains("simd_shuffle_up("), "expected simd_shuffle_up;\ngot:\n{msl}");
    assert!(msl.contains("simd_shuffle_xor("), "expected simd_shuffle_xor;\ngot:\n{msl}");
    assert!(msl.contains("simd_shuffle("), "expected simd_shuffle;\ngot:\n{msl}");
    // `buf`'s element type is `float` (MSL `float`, simdgroup-valid) — none of the
    // four shuffle calls above need the int32 round-trip cast added below.
    assert!(!msl.contains("int32_t"), "float shuffle should stay uncast;\ngot:\n{msl}");
}

/// Regression test for a real bug found via `perso/boring-llm`: Metal's
/// `simd_shuffle*` template family rejects 64-bit integers
/// (`__is_valid_simdgroup_type<long>` is false), but Boring's default `int`
/// maps to MSL `int64_t` — so `gpu.warp.shuffle` on an `int` local compiled
/// fine through `boring build` but panicked at first kernel dispatch with
/// "no matching function for call to 'simd_shuffle'" (confirmed on real
/// Apple Silicon hardware). Fixed by shuffling through a narrower
/// `int32_t` view and casting back.
#[test]
fn device_gpu_warp_shuffle_int_operand_gets_int32_roundtrip() {
    let (msl, _) = metal_codegen("gpu_warp_shuffle_int", r#"
kernel W:
    mut [float]'unified out
    def ():
        let tid = gpu.thread.x
        var int v = 0
        v = gpu.warp.shuffle(v, 0)
        var int w = 0
        w = gpu.warp.shuffle_down(w, 1)
        out[tid] = v as float32 + w as float32
"#);
    assert!(
        msl.contains("(int64_t)(simd_shuffle((int32_t)(v_), 0))")
            || msl.contains("(int64_t)(simd_shuffle((int32_t)(v), 0))"),
        "expected int32-round-tripped simd_shuffle for `int` operand;\ngot:\n{msl}"
    );
    assert!(
        msl.contains("(int64_t)(simd_shuffle_down((int32_t)(w")
        , "expected int32-round-tripped simd_shuffle_down for `int` operand;\ngot:\n{msl}"
    );
}

#[test]
fn device_gpu_thread_x_maps_to_thread_pos() {
    let (msl, _) = metal_codegen("gpu_thread_x", r#"
kernel B:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("__thread_pos"),
        "expected __thread_pos for gpu.thread;\ngot:\n{msl}");
    assert!(msl.contains("__thread_pos.x"),
        "expected __thread_pos.x for gpu.thread.x;\ngot:\n{msl}");
}

#[test]
fn device_gpu_block_dim_maps_correctly() {
    let (msl, _) = metal_codegen("gpu_block_dim", r#"
kernel B:
    mut [float]'unified buf
    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] = buf[i] * 2.0
"#);
    assert!(msl.contains("__block_pos.x"),
        "expected __block_pos.x for gpu.block.x;\ngot:\n{msl}");
    assert!(msl.contains("__block_dim.x"),
        "expected __block_dim.x for gpu.block_dim.x;\ngot:\n{msl}");
}

// ─── device — 'actor'global atomics ──────────────────────────────────────────

#[test]
fn device_actor_global_compound_assign_uses_atomic_fetch_add() {
    let (msl, _) = metal_codegen("atomic_add", r#"
kernel A:
    mut [int]'actor'global counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(msl.contains("atomic_fetch_add_explicit"),
        "expected atomic_fetch_add_explicit for 'actor'global += ;\ngot:\n{msl}");
    assert!(msl.contains("memory_order_relaxed"),
        "expected memory_order_relaxed;\ngot:\n{msl}");
}

// Regression test: `try_atomic_assign`/`try_atomic_method_call` used to cast
// every atomic op through `(device atomic_long*)` unconditionally (8 bytes),
// regardless of the field's real element type — a 4-byte `int32`/`uint32`
// field got the same 8-byte cast, corrupting adjacent GPU memory or producing
// a wrong numeric result (the bit pattern of a neighboring element folded
// into the same "atomic word"). Every existing atomic test before this one
// used a bare `[int]` field (64-bit — `int64_t`, the one case the old
// hardcoded `atomic_long` was actually width-correct for), so none of them
// exercised the 32-bit path at all. See `atomic_msl_cast`'s own doc comment.
#[test]
fn device_actor_global_int32_atomic_uses_32bit_intrinsic_not_atomic_long() {
    let (msl, _) = metal_codegen("atomic_add_int32", r#"
kernel A:
    mut [int32]'actor'global counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(msl.contains("(device atomic_int*)&counts[0]"),
        "expected a 4-byte `atomic_int` cast for an `int32` element, not the \
         field-width-blind `atomic_long` (8 bytes);\ngot:\n{msl}");
    assert!(!msl.contains("atomic_long"),
        "must not fall back to the 64-bit `atomic_long` cast for a 32-bit element;\ngot:\n{msl}");
    assert!(msl.contains("device int* counts [[buffer(0)]]"),
        "expected a 4-byte `int*` buffer param for an `int32` field, not `int64_t*`;\ngot:\n{msl}");
}

#[test]
fn device_actor_global_uint32_atomic_uses_32bit_intrinsic_not_atomic_long() {
    let (msl, _) = metal_codegen("atomic_add_uint32", r#"
kernel A:
    mut [uint32]'actor'global counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(msl.contains("(device atomic_uint*)&counts[0]"),
        "expected a 4-byte `atomic_uint` cast for a `uint32` element, not the \
         field-width-blind `atomic_long` (8 bytes);\ngot:\n{msl}");
    assert!(!msl.contains("atomic_long"),
        "must not fall back to the 64-bit `atomic_long` cast for a 32-bit element;\ngot:\n{msl}");
}

#[test]
fn device_actor_global_field_has_device_ptr_param() {
    let (msl, _) = metal_codegen("atomic_ptr_param", r#"
kernel A:
    mut [int]'actor'global counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(msl.contains("device int64_t* counts [[buffer(0)]]"),
        "expected device int64_t* counts [[buffer(0)]] for 'actor'global;\ngot:\n{msl}");
}

// ─── device/host — 'actor'unified atomics on host+device DRAM ───────────────

#[test]
fn device_actor_unified_compound_assign_uses_atomic_fetch_add() {
    let (msl, rs) = metal_codegen("actor_unified_add", r#"
kernel A:
    mut [int]'actor'unified counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(msl.contains("atomic_fetch_add_explicit"),
        "expected atomic_fetch_add_explicit for 'actor'unified += ;\ngot:\n{msl}");
    assert!(msl.contains("device int64_t* counts [[buffer(0)]]"),
        "expected device int64_t* counts [[buffer(0)]] for 'actor'unified;\ngot:\n{msl}");
    // Unlike 'actor'global, 'actor'unified is host-visible — it must get the same
    // D2H read accessor 'unified fields get.
    assert!(rs.contains("fn read_counts(&self)"),
        "expected a host-side read_counts() accessor for 'actor'unified;\ngot:\n{rs}");
}

// ─── device — sync barrier ────────────────────────────────────────────────────

#[test]
fn device_sync_emits_threadgroup_barrier() {
    let (msl, _) = metal_codegen("sync_barrier", r#"
kernel S:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = 1.0
        sync
        buf[tid] = buf[tid] + 1.0
"#);
    // Metal uses `threadgroup_barrier` for threadgroup memory sync.
    // The sync comment is emitted by the kernel backend as a Stmt::Comment("sync").
    assert!(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "expected threadgroup_barrier for sync;\ngot:\n{msl}");
}

// ─── host — struct and Metal plumbing ────────────────────────────────────────

#[test]
fn host_prelude_includes_metal_crate() {
    let (_, rs) = metal_codegen("host_prelude", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("use metal::*;"),
        "expected 'use metal::*;' in host prelude;\ngot:\n{rs}");
    assert!(rs.contains("include_str!(\"../kernels/main.metal\")"),
        "expected include_str! for BORING_MSL;\ngot:\n{rs}");
}

#[test]
fn host_pipeline_init_compiles_msl_and_gets_function() {
    let (_, rs) = metal_codegen("host_pipeline", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("new_library_with_source(BORING_MSL"),
        "expected new_library_with_source;\ngot:\n{rs}");
    assert!(rs.contains("\"Scale_kernel\""),
        "expected get_function(\"Scale_kernel\");\ngot:\n{rs}");
    assert!(rs.contains("new_compute_pipeline_state_with_function"),
        "expected new_compute_pipeline_state_with_function;\ngot:\n{rs}");
}

#[test]
fn host_unified_field_is_metal_buffer() {
    let (_, rs) = metal_codegen("host_buffer_field", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("buf: Buffer,"),
        "expected 'buf: Buffer,' for 'unified field;\ngot:\n{rs}");
}

#[test]
fn host_shared_field_absent_from_rust_struct() {
    let (_, rs) = metal_codegen("host_shared_absent", r#"
kernel S:
    mut [float]'unified out
    let [float]'actor  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(!rs.contains("scratch: Buffer"),
        "'actor field must not appear as Buffer in Rust struct;\ngot:\n{rs}");
}

#[test]
fn host_init_uploads_array_via_new_buffer_with_data() {
    // The upload doesn't happen inside `Scale::new` itself -- `data`'s param
    // is already a `Buffer` there; the upload happens at the CONSTRUCTOR
    // CALL SITE instead (`emit_kernel_ctor_args`). A source snippet with no
    // such call site (as this test previously had) can never emit
    // `new_buffer_with_data` anywhere regardless of whether the codegen is
    // correct -- this was a stale test from before that refactor.
    let (_, rs) = metal_codegen("host_htod", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Scale(data)
"#);
    assert!(rs.contains("new_buffer_with_data"),
        "expected new_buffer_with_data at the Scale(data) constructor call site;\ngot:\n{rs}");
    // `data` is a plain host Vec<f64> (the general pipeline's float
    // convention) but Metal buffers are always f32 (MSL has no native f64) --
    // missing this cast doesn't fail to compile, it silently copies half the
    // intended bytes (mem::size_of::<f32>() against actual f64 data),
    // confirmed by inspecting the generated Rust directly before this fix.
    assert!(rs.contains("as f32"),
        "expected an explicit f64->f32 cast before uploading the host array;\ngot:\n{rs}");
}

#[test]
fn host_new_with_arena_uploads_array_via_new_buffer_with_data() {
    // Same upload requirement as the plain `Scale(data)` call site above,
    // but through the arena-qualified `new(g) Scale(data)` constructor path
    // -- this used to skip the whole buffer-upload dance and pass `data`
    // straight through as a bare `Vec<f64>` where `Scale::new` expects a
    // `Buffer`, a real type mismatch confirmed via cargo check (mirrors the
    // identical bug fixed in cuda::host's `new(g) Scale(data)` handling).
    let (_, rs) = metal_codegen("host_htod_arena", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let g0 = GPU(0)
let data = [1.0, 2.0]
let k = new(g0) Scale(data)
"#);
    assert!(rs.contains("new_buffer_with_data"),
        "expected new_buffer_with_data at the new(g0) Scale(data) call site;\ngot:\n{rs}");
    assert!(rs.contains("as f32"),
        "expected an explicit f64->f32 cast before uploading the host array;\ngot:\n{rs}");
}

#[test]
fn host_fn_float32_param_passed_directly_to_kernel_ctor_is_vec_f32_not_f64() {
    // Regression test: a `pub req [T]'gpu'unified` host function whose
    // `[float32]` parameter is passed DIRECTLY to a kernel struct's own
    // constructor call (`AddKernel(a, n)`) used to have that parameter's
    // Rust element type wrongly forced to `f64` (`is_float_array_param`
    // didn't distinguish `float32` from bare `float`/`float64`, and the
    // general pipeline's own convention for `float32` is already `f32` --
    // there was never an f64/f32 mismatch to bridge for it in the first
    // place). This produced `a: &Vec<f64>` in the generated signature, a
    // real E0308 (`expected &Vec<f64>, found &Vec<f32>`) at every call site
    // passing a genuine `[float32]`-typed Boring value -- confirmed via a
    // real cross-compile `cargo build` before this fix, and via a real
    // Metal run producing the correct output (`2 3 4` for inputs `1 2 3`)
    // after it.
    let (_, rs) = metal_codegen("host_fn_float32_direct_kernel_ctor", r#"
kernel AddKernel:
    let [float32]'global a
    mut [float32]'unified out
    let int n

    init([float32]'global ai, int ni):
        a = ai
        n = ni
        out = [0.0 for i in 0..<ni]

    def ():
        let i = gpu.thread.x
        if i < n:
            out[i] = a[i] + 1.0

pub req [float32]'gpu'unified add_gpu([float32] a, int n) throws:
    mut k = AddKernel(a, n)
    kernel:
        k(block = n)
    k.out
"#);
    assert!(rs.contains("pub fn add_gpu(a: &Vec<f32>, n: isize)"),
        "expected the [float32] param to render as &Vec<f32>, not &Vec<f64>;\ngot:\n{rs}");
    assert!(!rs.contains("a: &Vec<f64>") && !rs.contains("a: Vec<f64>"),
        "the [float32] param must never be declared as a Vec<f64>;\ngot:\n{rs}");
}

#[test]
fn host_fn_chained_resident_call_arg_no_stray_ref() {
    // Regression test: chaining the result of one `pub req [T]'gpu'unified`
    // host function directly into another such function's non-first `[T]`
    // argument, via a single intermediate `let` binding (`let stage1 =
    // add_one_gpu(a, 4); let stage2 = add_one_gpu(stage1, 4)`), used to emit
    // an extraneous leading `&` around the whole `BoringGpuArg<T>` ->
    // `Vec<T>` materializing match-expression at the second call site --
    // `emit_args_coerced`'s `resident_call_vars` branch always hardcoded
    // `&(match &stage1 { ... })` without ever checking whether the callee's
    // declared parameter at that position was actually `T&` (Borrow-
    // qualified). A plain by-value `[float32]'global` param (as here) must
    // receive the materialized `Vec<f32>` itself, not `&Vec<f32>` -- this was
    // a genuine E0308 (`expected Vec<f32>, found &Vec<f32>`) on a real
    // `cargo build`, confirmed fixed by a real Metal run producing the
    // correct output (`3 4 5 6` for inputs `1 2 3 4` through two chained
    // +1.0 kernel dispatches) after this fix.
    let (_, rs) = metal_codegen("host_fn_chained_resident_call_arg", r#"
kernel AddOneKernel:
    let [float32]'global x
    mut [float32]'unified out
    let int n

    init([float32]'global xi, int ni):
        x = xi
        n = ni
        out = [0.0 for i in 0..<ni]

    def ():
        let cell = gpu.thread.x
        if cell < n:
            out[cell] = x[cell] + 1.0

pub req [float32]'gpu'unified add_one_gpu([float32]'global x, int n) throws:
    mut k = AddOneKernel(x, n)
    kernel:
        k(block = 256, grid = 1)
    k.out

[float32] passthrough([float32] x):
    x

def main() throws:
    let a = [1.0, 2.0, 3.0, 4.0]
    let stage1 = add_one_gpu(a, 4)
    let stage2 = add_one_gpu(stage1, 4)
    let result = passthrough(stage2)
    print "{result[0]}"
"#);
    assert!(rs.contains("add_one_gpu(stage1.clone(), 4)"),
        "expected the resident GPU argument to remain resident across the chained call;\ngot:\n{rs}");
    assert!(rs.contains("passthrough(&(match &stage2 {"),
        "expected materialization only at the ordinary host-function boundary;\ngot:\n{rs}");
}

// ─── host — __boring_launch ───────────────────────────────────────────────────

#[test]
fn host_boring_launch_signature() {
    let (_, rs) = metal_codegen("host_launch_sig", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("fn __boring_launch(&mut self, block_dim: (u32,u32,u32), grid_dim: Option<(u32,u32,u32)>"),
        "expected __boring_launch with Option grid_dim;\ngot:\n{rs}");
}

#[test]
fn host_boring_launch_dispatches_thread_groups() {
    let (_, rs) = metal_codegen("host_dispatch", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("dispatch_thread_groups("),
        "expected dispatch_thread_groups in __boring_launch;\ngot:\n{rs}");
    assert!(rs.contains("MTLSize"),
        "expected MTLSize for grid/block dims;\ngot:\n{rs}");
}

#[test]
fn host_boring_launch_waits_for_completion() {
    let (_, rs) = metal_codegen("host_wait", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("wait_until_completed()"),
        "expected wait_until_completed() for Metal synchronous launch;\ngot:\n{rs}");
}

#[test]
fn host_auto_grid_sizing_from_first_array_len() {
    let (_, rs) = metal_codegen("auto_grid", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("(n + block_dim.0 - 1) / block_dim.0"),
        "expected auto grid ceil-div expression;\ngot:\n{rs}");
    assert!(rs.contains("grid_dim: Option<(u32,u32,u32)>"),
        "expected Option grid_dim for auto-grid kernel;\ngot:\n{rs}");
}

#[test]
fn host_dynamic_shared_sets_threadgroup_memory_length() {
    let (_, rs) = metal_codegen("threadgroup_mem", r#"
kernel D:
    mut [float]'unified out
    let [float]'actor  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(rs.contains("set_threadgroup_memory_length("),
        "expected set_threadgroup_memory_length for dynamic 'actor;\ngot:\n{rs}");
}

#[test]
fn host_encoder_sets_buffer_for_unified_field() {
    let (_, rs) = metal_codegen("set_buffer", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("set_buffer(0, Some(&self.buf)"),
        "expected set_buffer(0, Some(&self.buf));\ngot:\n{rs}");
}

#[test]
fn host_const_scalar_uses_set_bytes() {
    let (_, rs) = metal_codegen("const_set_bytes", r#"
kernel C:
    mut [float]'unified buf
    let float'const     factor
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * factor
"#);
    assert!(rs.contains("set_bytes("),
        "expected set_bytes for 'const scalar field;\ngot:\n{rs}");
    assert!(rs.contains("&self.factor"),
        "expected &self.factor in set_bytes;\ngot:\n{rs}");
}

// ─── host — read accessor ─────────────────────────────────────────────────────

#[test]
fn host_read_accessor_generated_for_unified_array() {
    let (_, rs) = metal_codegen("read_accessor", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("fn read_buf("),
        "expected read_buf accessor for 'unified array;\ngot:\n{rs}");
    assert!(rs.contains("from_raw_parts"),
        "expected unsafe slice in read accessor;\ngot:\n{rs}");
}

#[test]
fn host_gpu_failure_surfaces_as_a_real_error_not_silent_wrong_behavior() {
    // Before this fix, `__boring_metal_flush` only called `wait_until_completed()`
    // and never inspected the command buffer's own status -- a GPU-side failure
    // (invalid threadgroup size, out-of-bounds access, device removal, ...)
    // completed with `status() == Error` and nobody looked, so `read_buf()`
    // happily read back whatever garbage/zeroed memory was left, reporting
    // success regardless. Confirmed via a real `cargo check` against the real
    // `metal` crate that this whole chain (flush -> read_<field> -> call site)
    // compiles end-to-end.
    let (_, rs) = metal_codegen("gpu_failure_surfaces", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Scale(data)
kernel:
    k(block = 2)
print "{k.buf[0]}"
"#);
    assert!(rs.contains("fn __boring_metal_flush() -> Result<(), Box<dyn std::error::Error + Send + Sync>>"),
        "expected __boring_metal_flush to return a real Result;\ngot:\n{rs}");
    assert!(rs.contains("buf.status() == MTLCommandBufferStatus::Error"),
        "expected __boring_metal_flush to check the command buffer's completion status;\ngot:\n{rs}");
    assert!(rs.contains("fn read_buf(&self) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>>"),
        "expected read_buf to propagate a real Result instead of silently returning garbage on failure;\ngot:\n{rs}");
    assert!(rs.contains("__boring_metal_flush()?;"),
        "expected read_buf to propagate __boring_metal_flush's error via ?;\ngot:\n{rs}");
    assert!(rs.contains("k.read_buf()?[0 as usize]"),
        "expected the k.buf[0] read call site to propagate via ? into main()'s own Result;\ngot:\n{rs}");
}

#[test]
fn dtod_ctor_arg_uses_real_device_to_device_copy_not_an_objc_retain() {
    // `Scale(k1.buf)` used to pass `k1.buf.clone()` straight through -- but
    // `Buffer::clone()` in the real `metal` crate is just an ObjC `retain`
    // (a reference-count bump, confirmed against the crate's
    // `foreign_type!`-generated impl), NOT a content copy. k1 and k2 ended up
    // sharing the exact same underlying `MTLBuffer`: dispatching k1 again
    // afterward would silently change k2's "own" buffer too, with no compile
    // error (unlike the analogous bug in cuda::host/rocm::host, a real
    // E0382 the Rust compiler catches). `__boring_metal_buffer_copy`
    // allocates a fresh buffer and memcpy's into it instead.
    let (_, rs) = metal_codegen("dtod_candidate", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k1 = Scale(data)
kernel:
    k1(block = 2)
mut k2 = Scale(k1.buf)
kernel:
    k1(block = 2)
    k2(block = 2)
print "{k1.buf[0]}"
"#);
    assert!(rs.contains("fn __boring_metal_buffer_copy(dev: &Device, buf: &Buffer) -> Result<Buffer, Box<dyn std::error::Error + Send + Sync>>"),
        "expected a real buffer-copy helper (new buffer + memcpy);\ngot:\n{rs}");
    assert!(rs.contains("std::ptr::copy_nonoverlapping"),
        "expected the copy helper to actually copy buffer contents;\ngot:\n{rs}");
    assert!(rs.contains("Scale::new(boring_metal_device(), __boring_metal_buffer_copy(&boring_metal_device(), &k1.buf)?)"),
        "expected the k2 constructor call to use the real copy helper, not a bare Buffer::clone() retain;\ngot:\n{rs}");
}

#[test]
fn read_only_kernel_input_reuses_explicitly_resident_local_buffer() {
    let (_, rs) = metal_codegen("resident_read_only_input", r#"
kernel Produce:
    mut [float]'unified out
    init([float]'unified initial):
        out = initial
    def ():
        out[gpu.thread.x] = 1.0

kernel Consume:
    let [float]'global input
    mut [float]'unified out
    init([float]'global value, [float]'unified result):
        input = value
        out = result
    def ():
        out[gpu.thread.x] = input[gpu.thread.x]

req [float]'gpu'unified produce() throws:
    mut p = Produce([0.0, 0.0])
    kernel:
        p(block = 2)
    p.out

def main() throws:
    let [float]'gpu'unified value = produce()
    mut c = Consume(value, [0.0, 0.0])
    kernel:
        c(block = 2)
"#);
    assert!(rs.contains("BoringGpuArg::Resident(buf, _) => buf.clone()"),
        "read-only Metal input should retain the resident Buffer without copying its contents:\n{rs}");
}

// ─── host — struct `count`/`length` field & method shadowing ─────────────────
//
// A plain (non-kernel) struct's own methods are only ever custom-emitted by
// this backend's `HostEmitter` (as opposed to the already-correct general/std
// pipeline splice) for a `Screen`-driven program — see `metal::mod`'s doc
// comment. That's the one path that exercises the `self.count`/`self.count()`
// guard added to `host.rs`'s `expr()` (`Field` and `MethodCall` cases): before
// the fix, a real user field or method literally named `count`/`length`
// (Boring's builtin array-length shortcut) was unconditionally rewritten to
// `.len() as isize` even on `self`, producing `self.len() as isize` — nonsense
// Rust, since `self` has no such method. Mirrors the identical, already-fixed
// guard in the general transpiler (`emit_expr.rs`'s `emit_expr_field` /
// `emit_top.rs`'s `emit_expr_owned`).

#[test]
fn screen_struct_self_count_field_not_shadowed_by_len_builtin() {
    let (_, rs) = metal_codegen("screen_self_count_field", r#"
let width = 4
let height = 4
let screen = Screen(Dimension(width, height), title = "Test")

struct Counter:
    int count
    req int get():
        self.count

kernel Noop:
    mut [uint]'surface pixels
    init():
        pixels = [0 for ..<width * height]
    def ():
        let tid = gpu.thread.x
        pixels[tid] = 0

var k = Noop()
let c = Counter(5)
print "{c.get()}"

kernel:
    loop:
        k(block = (4, 4))
        screen.present(k.pixels)
        break
"#);
    assert!(rs.contains("fn get(&self) -> isize {\n        self.count\n    }"),
        "expected `self.count` to read the real declared field, not the `.length`/`.count` \
         array-length builtin;\ngot:\n{rs}");
    assert!(!rs.contains("self.len() as isize"),
        "`self.count` must not be shadowed by the `.len() as isize` builtin shortcut;\ngot:\n{rs}");
}

#[test]
fn screen_struct_self_count_method_not_shadowed_by_len_builtin() {
    let (_, rs) = metal_codegen("screen_self_count_method", r#"
let width = 4
let height = 4
let screen = Screen(Dimension(width, height), title = "Test")

struct Ledger:
    int total
    req int count():
        self.total
    req int double_count():
        self.count() * 2

kernel Noop:
    mut [uint]'surface pixels
    init():
        pixels = [0 for ..<width * height]
    def ():
        let tid = gpu.thread.x
        pixels[tid] = 0

var k = Noop()
let ledger = Ledger(10)
print "{ledger.double_count()}"

kernel:
    loop:
        k(block = (4, 4))
        screen.present(k.pixels)
        break
"#);
    assert!(rs.contains("(self.count() * 2)"),
        "expected `self.count()` to call the real declared method, not the `.length`/`.count` \
         array-length builtin;\ngot:\n{rs}");
    assert!(!rs.contains("self.len() as isize"),
        "`self.count()` must not be shadowed by the `.len() as isize` builtin shortcut;\ngot:\n{rs}");
}

// ─── infrastructure — Cargo.toml ─────────────────────────────────────────────

#[test]
fn cargo_toml_depends_on_metal_crate() {
    let toml = cargo_toml("infra_cargo_toml", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(toml.contains("metal"),
        "Cargo.toml must depend on the metal crate;\ngot:\n{toml}");
}

// ─── saxpy example ────────────────────────────────────────────────────────────

#[test]
fn example_saxpy_metal() {
    let src = std::fs::read_to_string("examples/saxpy.br").expect("examples/saxpy.br not found");
    let (msl, rs) = metal_codegen("saxpy_example", &src);

    // MSL kernel
    assert!(msl.contains("kernel void Saxpy_kernel("), "missing Saxpy_kernel;\ngot:\n{msl}");
    assert!(msl.contains("device float* y [[buffer("), "missing y buffer param;\ngot:\n{msl}");

    // Host struct
    assert!(rs.contains("struct Saxpy"),  "missing struct Saxpy;\ngot:\n{rs}");
    assert!(rs.contains("buf: Buffer,") || rs.contains("y: Buffer,"),
        "missing Buffer field in Saxpy;\ngot:\n{rs}");
    assert!(rs.contains("new_library_with_source(BORING_MSL"),
        "missing MSL compile step;\ngot:\n{rs}");
}

// ─── KernelHandle must_use ─────────────────────────────────────────────────────

#[test]
fn kernel_handle_is_must_use() {
    // Dropping a `KernelHandle<T>` without `.wait`/`.inner` used to compile
    // silently -- `#[must_use]` turns that into a compiler warning instead.
    let (_, rs) = metal_codegen("kernel_handle_must_use", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(
        rs.contains("#[must_use = \"a KernelHandle must be waited on (.wait/.inner) or the launch may not be synchronized\"]\nstruct KernelHandle<T>"),
        "expected #[must_use] directly above struct KernelHandle<T>;\ngot:\n{rs}"
    );
}

// ─── GPU error classification ─────────────────────────────────────────────────

#[test]
fn command_buffer_failure_extracts_real_nserror_code() {
    // Previously `status() == MTLCommandBufferStatus::Error` was the only
    // check -- `{:?}` on the status enum just printed the literal word
    // "Error", no indication of the actual cause. `CommandBufferRef`
    // implements `objc::Message` (confirmed against real metal 0.29 source
    // -- the crate's own generated Debug impl relies on this same fact to
    // call `debugDescription`), so the real NSError and its `code` are one
    // `msg_send![..., error]` away, classified against Apple's own
    // MTLCommandBufferError codes. `objc` is now an unconditional
    // dependency (previously only added when `Screen` was present) since
    // every program's `__boring_metal_flush` needs it, not just the display
    // path. Verified to compile clean via a real cargo check against real
    // metal 0.29 + objc 0.2, for both a plain compute program and a real
    // Screen-using example (`examples/plasma_metal.br`) -- confirming no
    // duplicate `extern crate objc;`.
    let (_, rs) = metal_codegen("metal_error_classify", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("#[macro_use] extern crate objc;"),
        "expected objc to be an unconditional dependency, not gated on Screen;\ngot:\n{rs}");
    assert!(rs.contains("let buf_ref: &CommandBufferRef = &buf;"),
        "expected a &CommandBufferRef borrow for the msg_send call;\ngot:\n{rs}");
    assert!(rs.contains("objc::msg_send![buf_ref, error]"),
        "expected the real NSError to be fetched via msg_send![.., error];\ngot:\n{rs}");
    assert!(rs.contains("8  => \"out of memory\","),
        "expected MTLCommandBufferError code 8 classified as out of memory;\ngot:\n{rs}");
    assert!(rs.contains("3  => \"page fault (illegal memory access)\","),
        "expected MTLCommandBufferError code 3 classified as illegal memory access;\ngot:\n{rs}");
}

// ─── atomic min/max/swap/cas ───────────────────────────────────────────────────

#[test]
fn device_atomic_method_calls_map_to_msl_intrinsics() {
    // min/max/swap map directly onto MSL's atomic_fetch_min/max_explicit and
    // atomic_exchange_explicit, which already return the previous value like
    // their CUDA/HIP equivalents. cas is a real shape mismatch: MSL's
    // atomic_compare_exchange_weak_explicit takes a pointer to the expected
    // value and returns a bool, not the previous value directly -- bridged
    // via a GNU/Clang statement-expression (`({ ... })`, supported by
    // Metal's Clang-based compiler) so it's still usable as one expression.
    let (msl, _) = metal_codegen("atomic_methods", r#"
kernel Histogram:
    mut [int]'actor'global counts
    init([int]'actor'global data):
        counts = data
    def ():
        let bucket = gpu.thread.x
        counts[bucket].min(5)
        counts[bucket].max(5)
        let old_swap = counts[bucket].swap(0)
        let old_cas = counts[bucket].cas(0, 1)
"#);
    assert!(msl.contains("atomic_fetch_min_explicit((device atomic_long*)&counts[bucket], (long)(5), memory_order_relaxed)"),
        "expected atomic_fetch_min_explicit;\ngot:\n{msl}");
    assert!(msl.contains("atomic_fetch_max_explicit((device atomic_long*)&counts[bucket], (long)(5), memory_order_relaxed)"),
        "expected atomic_fetch_max_explicit;\ngot:\n{msl}");
    assert!(msl.contains("atomic_exchange_explicit((device atomic_long*)&counts[bucket], (long)(0), memory_order_relaxed)"),
        "expected atomic_exchange_explicit;\ngot:\n{msl}");
    assert!(msl.contains("atomic_compare_exchange_weak_explicit((device atomic_long*)&counts[bucket], &__boring_cas_exp, (long)(1), memory_order_relaxed, memory_order_relaxed); __boring_cas_exp; }"),
        "expected atomic_compare_exchange_weak_explicit bridged via a statement-expression;\ngot:\n{msl}");
}

// ─── Labeled multi-dimensional arrays (docs/array-multidim-types.md) ───────

#[test]
fn device_labeled_index_lowers_to_row_major_index() {
    let (msl, _) = metal_codegen("labeled_at", r#"
kernel Img:
    mut [float, width = 4, height = 4]'unified img
    init([float, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(msl.contains("img[c + r * 4]"),
        "expected [width=c,height=r] to lower to row-major c + r*width;\ngot:\n{msl}");
}

#[test]
fn device_labeled_array_field_becomes_device_buffer_param() {
    let (msl, _) = metal_codegen("labeled_ptr_param", r#"
kernel Img:
    mut [float32, width = 4, height = 4]'unified img
    init([float32, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = 0.0
"#);
    assert!(msl.contains("device float* img [[buffer(0)]]"),
        "expected a LabeledArray field to become a device buffer param, same as [T]'unified;\ngot:\n{msl}");
}

#[test]
fn host_labeled_array_field_infers_2d_grid() {
    let (_, rs) = metal_codegen("labeled_2d_grid", r#"
kernel Img:
    mut [float, width = 16, height = 32]'unified img
    init([float, width = 16, height = 32]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(rs.contains("((16 + block_dim.0 - 1) / block_dim.0)"),
        "expected grid.x inferred from width=16;\ngot:\n{rs}");
    assert!(rs.contains("((32 + block_dim.1 - 1) / block_dim.1)"),
        "expected grid.y inferred from height=32;\ngot:\n{rs}");
}

#[test]
fn host_dynamic_labeled_array_field_infers_2d_grid_from_shadow_fields() {
    let (_, rs) = metal_codegen("dynamic_labeled_2d_grid", r#"
kernel Img:
    mut [float, width, height]'unified img
    init([float]'unified data, uint w, uint h):
        img = data.reshape(width = w, height = h)
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(rs.contains("self.__img_axis0"), "expected grid.x inferred from the __img_axis0 shadow field;\ngot:\n{rs}");
    assert!(rs.contains("self.__img_axis1"), "expected grid.y inferred from the __img_axis1 shadow field;\ngot:\n{rs}");
    // No negative "doesn't fall back to 1D" assertion: `self.img.length()`
    // legitimately appears elsewhere, in the `read_img()` accessor.
}

#[test]
fn host_labeled_array_field_is_metal_buffer() {
    let (_, rs) = metal_codegen("labeled_buffer_field", r#"
kernel Img:
    mut [float, width = 4, height = 4]'unified img
    init([float, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(rs.contains("img: Buffer,"),
        "expected bare 'unified LabeledArray field to become a Buffer host field, same as [T]'unified;\ngot:\n{rs}");
}

#[test]
fn device_labeled_array_3_axis_lowers_to_row_major_index() {
    let (msl, _) = metal_codegen("labeled_3_axis", r#"
kernel Vol:
    mut [float, x = 4, y = 4, z = 4]'unified vol
    init([float, x = 4, y = 4, z = 4]'unified data):
        vol = data
    def ():
        let tx = gpu.thread.x
        let ty = gpu.thread.y
        let tz = gpu.thread.z
        vol[x = tx, y = ty, z = tz] = vol[x = tx, y = ty, z = tz] * 2.0
"#);
    assert!(msl.contains("vol[tx + ty * 4 + tz * 16]"),
        "expected [x,y,z] to lower to row-major x + y*4 + z*(4*4);\ngot:\n{msl}");
}

#[test]
fn device_shared_labeled_array_becomes_fixed_threadgroup_decl() {
    let (msl, _) = metal_codegen("shared_labeled", r#"
kernel Tile:
    mut [float32]'unified out
    let [float32, width = 4, height = 4]'actor tile
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        out[0] = tile[width = c, height = r]
"#);
    assert!(msl.contains("threadgroup float tile[16];"),
        "expected fixed threadgroup decl sized width*height, declared in the kernel body;\ngot:\n{msl}");
    assert!(!msl.contains("tile [[threadgroup("),
        "static 'actor LabeledArray must not appear as a threadgroup param (that's the dynamic-array path);\ngot:\n{msl}");
}

// ─── .min/.max/.swap/.cas without 'actor — plain, non-atomic fallback ─────────

#[test]
fn atomic_method_calls_degrade_to_plain_read_modify_write_without_actor() {
    // Mirrors the identical CUDA/ROCm fix -- see cuda_codegen.rs's own doc.
    // Metal's compiler is Clang-based, same GNU statement-expression support.
    let (msl, _) = metal_codegen("plain_atomic_methods", r#"
kernel Scale:
    mut [int]'unified buf
    init([int]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        let m = buf[tid].min(5)
        let x = buf[tid].max(5)
        let s = buf[tid].swap(0)
        let c = buf[tid].cas(0, 1)
"#);
    assert!(msl.contains("({ auto __old = buf[tid]; buf[tid] = min(buf[tid], (5)); __old; })"),
        "expected plain min via a GNU statement-expression;\ngot:\n{msl}");
    assert!(msl.contains("({ auto __old = buf[tid]; buf[tid] = max(buf[tid], (5)); __old; })"),
        "expected plain max via a GNU statement-expression;\ngot:\n{msl}");
    assert!(msl.contains("({ auto __old = buf[tid]; buf[tid] = (0); __old; })"),
        "expected plain swap via a GNU statement-expression;\ngot:\n{msl}");
    assert!(msl.contains("({ auto __old = buf[tid]; if (__old == (0)) buf[tid] = (1); __old; })"),
        "expected plain cas via a GNU statement-expression;\ngot:\n{msl}");
}

#[test]
fn device_auto_sync_barrier_found_when_loop_nested_inside_top_level_if() {
    // No explicit `sync` here — this relies on the auto-inserted write-phase barrier
    // (`first_loop_index` in transpiler/helpers.rs). The accumulation loop that reads
    // `shared` cross-thread is nested inside a top-level `if`, not a bare top-level
    // `for`/`while` sibling — `first_loop_index` used to only match a bare top-level
    // loop statement directly, so this shape was invisible to it and no barrier at
    // all was emitted before the loop, a real cross-thread race on shared memory.
    let (msl, _) = metal_codegen("auto_sync_nested_if", r#"
kernel Reduce:
    let [int, 4]'actor shared

    def ():
        let tid = gpu.thread.x
        if tid == 0:
            shared[0] = 0
        if true:
            for i in 0..<4:
                shared[i] = shared[i] + 1
"#);
    assert!(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "expected an auto-inserted threadgroup_barrier before the loop nested inside \
         the top-level `if`, even with no explicit `sync`;\ngot:\n{msl}");
}

// ─── kernel-touching struct method — hard error, not a silent eprintln! ────────

#[test]
fn kernel_touching_struct_method_is_a_hard_build_error() {
    // A struct method that touches a kernel (constructs one / dispatches one) isn't
    // supported by the general-pipeline splice (see this module's doc comment and
    // `kernel_touching_struct_names`) -- this used to just `eprintln!` a warning and
    // fall through to codegen that's documented as the historical cause of real
    // `E0382`/`E0308` build failures in the *generated* Rust. `boring build` itself
    // must now fail with a clear diagnostic instead of silently writing out a
    // project doomed to fail downstream in `cargo build`.
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("metal_codegen").join("kernel_touching_struct_method");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let br_file = tmp.join("test.br");
    fs::write(&br_file, r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

struct Runner:
    def run():
        let data = [1.0, 2.0]
        mut k = Scale(data)
"#).unwrap();

    let result = Command::new(bin)
        .args(["build", "--target", "metal"])
        .arg(&br_file)
        .output()
        .unwrap();

    assert!(
        !result.status.success(),
        "boring build must fail (exit non-zero) for a kernel-touching struct method \
         instead of silently succeeding with a stderr-only warning"
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        stderr.contains("Runner"),
        "error output should name the offending struct `Runner`;\ngot:\n{stderr}"
    );
    assert!(
        !tmp.join("test_metal").join("src").join("main.rs").exists(),
        "boring build should not write out a generated project when it reports \
         this as a hard error"
    );
}

// ─── const-generic kernel field (`kernel Foo<int W, int H>:`) — regression ──
//
// A `'const`-qualified array field whose size is a const-generic expression
// (`[float, W * H]'const`, `Type::ArrayNExpr`) — rather than a literal
// (`[float, 3]`, `Type::ArrayN`) — used to fall through every "is this an
// array field" check in this backend's host codegen (all of which only
// recognized `Type::Array`/`Type::ArrayN`), landing in the scalar fallback
// and producing `weights: ()` (a unit-type struct field!) plus a matching
// `let weights: () = w;` init assignment — a guaranteed E0308 the moment the
// generated Rust was compiled. Separately, the turbofish CONSTRUCTION itself
// (`Blur<3, 1>(...)`) fell through this file's `expr()`'s `/* expr */`
// catch-all (no `ExprKind::GenericCall` arm existed), and the later
// `blur(block = 256)` dispatch wasn't recognized as a kernel launch either
// (`resolve_kernel_type` had no `GenericCall` arm, so `blur` never got
// registered in `var_kernel_type`). All found while verifying
// `linguist/samples/gpu.br` for the 0.9.7 release; see `rust_type`'s
// `Type::ArrayNExpr` arm and this backend's two `resolve_kernel_type`/
// `expr()` `GenericCall` arms for the fix.

#[test]
fn const_generic_array_field_becomes_buffer_not_unit() {
    let (_msl, rs, _toml) = run_metal("const_generic_field", r#"
kernel Blur<int W, int H>:
    let [float, W * H]'const weights
    let [float]'global        input
    mut [float]'global        output

    init([float] w, [float] inp, [float] out):
        weights = w
        input   = inp
        output  = out

    def ():
        let i = gpu.thread.x
        output[i] = weights[0] * input[i]

let w = [0.5]
let pixels = [1.0, 2.0]
mut result = [0.0, 0.0]
mut blur = Blur<1, 1>(w, pixels, result)
kernel:
    blur(block = 2)
"#);
    assert!(
        rs.contains("weights: Buffer,"),
        "expected the const-generic-sized 'const array field to become a \
         Buffer struct field (matching plain `[T, N]'const`), not `()`;\ngot:\n{rs}"
    );
    assert!(
        !rs.contains("weights: (),") && !rs.contains("let weights: () = "),
        "the field/init-assignment must not fall back to the unit type;\ngot:\n{rs}"
    );
    assert!(
        rs.contains("let mut blur = Blur::new("),
        "expected the turbofish construction `Blur<1, 1>(...)` to emit a plain \
         `Blur::new(...)` call (type args erased, same as a non-generic kernel), \
         not this file's `/* expr */` catch-all;\ngot:\n{rs}"
    );
    assert!(
        rs.contains("blur.__boring_launch("),
        "expected `blur(block = 2)` to be recognized as a kernel launch \
         (`blur.__boring_launch(...)`), not a bogus ordinary function call;\ngot:\n{rs}"
    );
}

// ─── kernel/free-function tail expression → explicit `return` ────────────────
//
// Regression tests for a silent-correctness bug: a device function/method's
// implicit tail expression (the last statement, with no explicit `return`)
// used to be emitted as a bare, discarded statement instead of `return <expr>;`
// — compiles cleanly, runs, and produces silently wrong results (the caller
// always got the pre-call value back). MSL requires an explicit `return` for
// a non-void function, unlike Rust's own implicit-tail-return convention.

#[test]
fn device_kernel_helper_method_tail_expression_emits_return() {
    let (msl, _) = metal_codegen("kernel_helper_tail_return", r#"
kernel AddOneF32:
    mut [float32]'unified data

    def float32 helper(float32 x):
        x + 1.0

    def ():
        let i = gpu.thread.x
        data[i] = self.helper(data[i])
"#);
    assert!(
        msl.contains("return (x + 1.0);") || msl.contains("return x + 1.0;"),
        "expected the kernel helper method's tail expression to be emitted as \
         an explicit `return ...;`, not a discarded bare statement;\ngot:\n{msl}"
    );
    // A trimmed-line-equality check (not a plain substring check) — the bad
    // bare form `(x + 1.0);` is itself a substring of the good `return (x +
    // 1.0);` line, so a naive `!msl.contains(...)` would always incorrectly
    // pass once the fix's own `return ` prefix is present.
    let has_bad_bare_stmt = msl.lines().any(|l| {
        let t = l.trim();
        t == "(x + 1.0);" || t == "x + 1.0;"
    });
    assert!(
        !has_bad_bare_stmt,
        "the old discarded bare-statement form must not still be present \
         alongside the `return` (that would mean the tail statement was \
         duplicated, not fixed);\ngot:\n{msl}"
    );
}

#[test]
fn device_free_function_tail_expression_emits_return() {
    let (msl, _) = metal_codegen("free_fn_tail_return", r#"
def float32 addOne(float32 x):
    x + 1.0

kernel AddOneF32:
    mut [float32]'unified data

    def ():
        let i = gpu.thread.x
        data[i] = addOne(data[i])
"#);
    assert!(
        msl.contains("return (x + 1.0);") || msl.contains("return x + 1.0;"),
        "expected the free function's tail expression to be emitted as an \
         explicit `return ...;`, not a discarded bare statement;\ngot:\n{msl}"
    );
    let has_bad_bare_stmt = msl.lines().any(|l| {
        let t = l.trim();
        t == "(x + 1.0);" || t == "x + 1.0;"
    });
    assert!(
        !has_bad_bare_stmt,
        "the old discarded bare-statement form must not still be present \
         alongside the `return`;\ngot:\n{msl}"
    );
}

// ─── MSL reserved-word identifier collisions ──────────────────────────────────
//
// An ordinary Boring identifier (no special meaning in the language) can
// collide with one of MSL's own builtin scalar type names — `half` (16-bit
// float) is the real-world case that motivated this: a kernel field or local
// named `half` (e.g. `half = d_head / 2` in a RoPE positional-encoding
// kernel) used to be emitted verbatim, producing a confusing MSL *parse*
// error at runtime (`newLibraryWithSource`) rather than a Boring-level error
// — `boring build --target metal` itself always reported success. Fixed by
// `msl_safe_ident` (`src/transpiler/metal/device.rs`): any identifier that
// collides with an MSL reserved word gets a trailing underscore, applied
// consistently at both its declaration and every reference. Verified
// end-to-end against a real Metal compiler (not just these snapshot
// assertions): `boring build --target metal` + `cargo build` + running the
// resulting binary on real Apple Silicon hardware, both before this fix
// (reproducing the exact MSL compile error from the bug report) and after
// (produces the expected output).

#[test]
fn device_field_named_half_is_mangled_not_left_colliding_with_msl_builtin_type() {
    let (msl, _) = metal_codegen("field_named_half", r#"
kernel HalfKernel:
    let [float32]'unified x
    mut [float32]'unified out
    let int'const         half

    init([float32]'unified xs, int h):
        x    = xs
        half = h
        out  = [0.0 for ..<8]

    def ():
        let cell = gpu.thread.x
        if cell < half:
            out[cell] = x[cell]
"#);
    // Declaration and every use must agree on the same mangled name — a bare,
    // unmangled `half` declaration is exactly the collision this test guards
    // against (MSL parses it as its own builtin `half` type, not a variable).
    assert!(msl.contains("const int64_t half_ = *__half;"),
        "expected the deref'd local to be declared as `half_`;\ngot:\n{msl}");
    assert!(msl.contains("cell < half_"),
        "expected the read of the field inside the kernel body to use the \
         same mangled name `half_` as its declaration;\ngot:\n{msl}");
}

#[test]
fn device_local_let_named_half_is_mangled() {
    let (msl, _) = metal_codegen("local_let_named_half", r#"
kernel LocalHalf:
    let int                n
    mut [float32]'unified  out

    init(int nn):
        n   = nn
        out = [0.0 for ..<nn]

    def ():
        let half = n / 2
        let tid = gpu.thread.x
        if tid < half:
            out[tid] = 1.0
"#);
    assert!(!msl.contains("int64_t half ="),
        "a local `let half = ...` must not be emitted as a bare `half` \
         identifier (collides with MSL's builtin `half` type);\ngot:\n{msl}");
    assert!(msl.contains("half_ ="),
        "expected the local to be mangled to `half_`;\ngot:\n{msl}");
    assert!(msl.contains("tid < half_"),
        "expected the later read of the local to use the same mangled name \
         as its declaration;\ngot:\n{msl}");
}

#[test]
fn device_for_loop_var_named_half_is_mangled() {
    let (msl, _) = metal_codegen("for_loop_var_named_half", r#"
kernel LoopHalf:
    mut [float32]'unified out

    def ():
        for half in 0..<4:
            out[half] = 1.0
"#);
    assert!(!msl.contains("int64_t half ="),
        "a for-loop variable named `half` must not be emitted verbatim \
         (collides with MSL's builtin `half` type);\ngot:\n{msl}");
    assert!(msl.contains("int64_t half_ ="),
        "expected the loop variable to be mangled to `half_`;\ngot:\n{msl}");
    assert!(msl.contains("out[half_]"),
        "expected the loop body's reference to the loop variable to use the \
         same mangled name as its declaration;\ngot:\n{msl}");
}

// ─── Free function with a GPU-qualified array parameter — MSL address space ──
//
// A free (non-kernel, non-method) function whose parameter type carries a GPU
// array qualifier (`[T]'global`, `'unified`, `'const`, ...), called from inside a
// kernel's `def()` body, used to transpile its parameter as a bare pointer with no
// MSL address-space qualifier (`uchar* w_packed` instead of `device const uchar*
// w_packed`). A kernel STRUCT FIELD of the identical qualifier already got the
// right address space (`buffer_field_params`); only a free function's OWN
// parameter list fell through `msl_type`'s generic `Type::Qualified(inner, _) =>
// msl_type(inner)` arm, which unconditionally discards the qualifier.
//
// This is invisible to both `boring build --target metal` (reports success
// regardless) and `cargo build`/`cargo run` on the generated Rust (compiles and
// links fine — the Rust host code has no idea what's inside the MSL string
// literal it embeds via `include_str!`). It only surfaces when the Metal API
// actually compiles the embedded MSL source at process *runtime*
// (`new_library_with_source`), with an opaque MSL error nowhere near the Boring
// source: `error: pointer type must have explicit address space qualifier`.
// Reproduced against a real fused-dequantization-style helper (read quantized
// bytes out of a `'global` array from a free function, called from a kernel).
//
// Fixed by `msl_free_fn_param_type` (`src/transpiler/metal/device.rs`), used by
// both `emit_free_device_fn` (free functions) and `emit_device_fn` (a kernel
// method's own extra, non-field parameters — identical bug shape). It also
// matches the `const` a kernel field of the same qualifier gets from a `let`
// binding (`buffer_field_params`) — omitting that second-order fix still
// compiles the MSL cleanly right up until a caller passes a `let`-bound (already
// `device const T*`) field into the free function, which MSL then rejects for
// discarding `const` — a second runtime-only shader-compile failure with the
// exact same invisible-to-`cargo-build` failure mode as the missing address
// space itself.

#[test]
fn device_free_fn_gpu_array_param_gets_address_space_and_constness() {
    let (msl, _) = metal_codegen("free_fn_global_array_param", r#"
float32 dequant_at([uint8]'global w_packed, int idx):
    let b = w_packed[idx]
    (b as float32) * 2.0

kernel Dequant:
    let [uint8]'global w_packed
    mut [float32]'unified out

    init([uint8] w, [float32] o):
        w_packed = w
        out = o

    def ():
        let tid = gpu.thread.x
        out[tid] = dequant_at(w_packed, tid)
"#);
    assert!(
        msl.contains("inline float dequant_at(device const uchar* w_packed, int64_t idx)"),
        "expected the free function's `[uint8]'global` parameter to be emitted \
         with the `device` address space AND `const` (matching the `let`-bound \
         kernel field of the same qualifier, `device const uchar* w_packed \
         [[buffer(0)]]`) -- a bare `uchar* w_packed` compiles fine as Rust/MSL \
         *text* but fails real Metal shader compilation at process runtime with \
         \"pointer type must have explicit address space qualifier\", or (once \
         only the address space is fixed but not constness) \"would lose const \
         qualifier\";\ngot:\n{msl}"
    );
}

// Real end-to-end verification of the above fix: `boring build --target metal` +
// `cargo build` + actually running the resulting binary, which dispatches
// `Dequant`'s kernel and reads back GPU-written results. This is the only way to
// catch this bug class at all -- it is invisible to both `boring build` (always
// reports success) and `cargo build`/`cargo run` on the generated Rust (the buggy
// MSL is just an opaque string literal to Rust) up until the Metal API itself
// compiles that embedded MSL source at runtime. Confirmed this test fails with
// the exact MSL compiler error from the bug report against the pre-fix code, and
// produces the correct GPU-computed values after.
#[test]
fn real_gpu_dispatch_free_fn_with_global_array_param() {
    let test_name = "free_fn_global_array_param_dispatch";
    let (_msl, _rs, _toml) = run_metal(test_name, r#"
float32 dequant_at([uint8]'global w_packed, int idx):
    let b = w_packed[idx]
    (b as float32) * 2.0

kernel Dequant:
    let [uint8]'global w_packed
    mut [float32]'unified out

    init([uint8] w, [float32] o):
        w_packed = w
        out = o

    def ():
        let tid = gpu.thread.x
        out[tid] = dequant_at(w_packed, tid)

let w = [1, 2, 3, 4]
mut result = [0.0, 0.0, 0.0, 0.0]

mut k = Dequant(w, result)
kernel:
    k(block = 4)

print "out[0] = {k.out[0]}"
print "out[1] = {k.out[1]}"
print "out[2] = {k.out[2]}"
print "out[3] = {k.out[3]}"
"#);

    let manifest = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("metal_codegen").join(test_name).join("test_metal").join("Cargo.toml");
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated Metal project to build AND run to completion \
         against a real Metal GPU (dispatching a kernel whose body calls a free \
         function taking a `[uint8]'global` parameter), but it failed:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    let expected = "out[0] = 2\nout[1] = 4\nout[2] = 6\nout[3] = 8";
    assert_eq!(
        stdout.trim_end(), expected,
        "expected the GPU-computed dequantized values (w[i] * 2.0);\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
}

// ─── host — string → numeric parse cast width ─────────────────────────────────

// `(s as float32)` — the documented optional-parse cast form (same shape as
// `(s as int)`) — used to transpile with the parsed value hardcoded to `f64`
// regardless of the cast's actual target width, under EVERY target including
// this one: `s.trim().parse::<f64>().ok()` instead of `::<f32>()`. Harmless as
// long as the f64 value is only ever printed/compared, but a hard `E0308`
// "expected f32, found f64" as soon as it flows into an `f32`-typed slot (a
// function return type, a struct field, ...) -- exactly the shape a real
// program hits parsing a numeric CLI argument or config value declared
// `float32` throughout (matching GPU-facing buffer element types). No real
// Metal GPU needed here: the buggy/fixed code lives entirely in the shared
// host-side cast codegen, never touches device/kernel code.
#[test]
fn host_string_to_float32_cast_uses_f32_not_f64() {
    let (_, rs) = metal_codegen("string_to_float32_cast", r#"
float32 parse_float_arg(string name, string s) throws:
    guard let f = (s as float32) else throw "invalid {name} value: '{s}'"
    f
"#);
    assert!(
        rs.contains("parse::<f32>()"),
        "expected `(s as float32)` to parse as f32, not a hardcoded f64;\ngot:\n{rs}"
    );
    assert!(
        !rs.contains("parse::<f64>()"),
        "found a leftover hardcoded f64 parse for a float32 cast target;\ngot:\n{rs}"
    );
}

// Real end-to-end verification of the above fix: `boring build --target metal` +
// `cargo build` + running the resulting binary. This is the only way to catch
// the compile-time symptom at all -- `boring build` always reports success, and
// a codegen snapshot test alone wouldn't prove the generated Rust actually
// type-checks. Confirmed this test fails to compile (`E0308: expected f32,
// found f64`) against the pre-fix code, and prints the correctly-parsed value
// after.
#[test]
fn real_metal_target_string_to_float32_cast_compiles_and_runs() {
    let test_name = "string_to_float32_cast_run";
    let (_msl, _rs, _toml) = run_metal(test_name, r#"
float32 parse_float_arg(string name, string s) throws:
    guard let f = (s as float32) else throw "invalid {name} value: '{s}'"
    f

let v = parse_float_arg("x", "2.25")
print "{v}"
"#);

    let manifest = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("metal_codegen").join(test_name).join("test_metal").join("Cargo.toml");
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected `parse_float_arg`'s generated Rust to compile and run cleanly \
         under `--target metal` (no real GPU touched -- this function never \
         dispatches a kernel), but it failed:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert_eq!(
        stdout.trim_end(), "2.25",
        "expected the correctly string-parsed float32 value;\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
}

// ─── host — string indexing/slicing in a kernel-touching function ─────────────

// `s[i]` (single-char index) and `s[a..<b]` (range slice) on a `string` local
// previously compiled correctly under a plain (non-GPU) `boring build`, but
// failed `cargo build` under `--target metal` (`E0277: the type str cannot be
// indexed by usize` / a `.to_vec()` call on `str`, which doesn't exist) as
// soon as the ENCLOSING function is "kernel-touching" (constructs/dispatches a
// kernel, or takes a kernel-typed param) -- see `metal::host.rs`'s own custom
// `expr()`/`ExprKind::Index` case, which (unlike `emit_expr.rs`'s
// `emit_expr_index` used by the general-pipeline splice for every OTHER
// function) had no string-vs-array distinction at all and always emitted
// Vec-style `[i as usize]`/`[range].to_vec()`. A non-kernel-touching helper
// function doing the exact same string indexing was NEVER affected (it's
// spliced from the general pipeline, which already had correct char-safe
// codegen) -- this test's `main` is deliberately the one doing both the
// string indexing AND the kernel dispatch, so it's forced through the buggy
// custom emitter. Real end-to-end verification (real Metal GPU dispatch),
// same rationale as `real_metal_target_string_to_float32_cast_compiles_and_runs`
// above -- a codegen snapshot alone wouldn't prove the generated Rust actually
// compiles.
#[test]
fn real_metal_target_string_indexing_and_slicing_in_kernel_touching_fn_compiles_and_runs() {
    let test_name = "string_indexing_kernel_touching_fn";
    let (_msl, _rs, _toml) = run_metal(test_name, r#"
kernel NoopKernel:
    mut [float32]'unified out
    init():
        out = [0.0]
    def ():
        out[0] = 1.0

def main() throws:
    let s = "hello world"
    let c = s[1]
    let sub = s[0..<5]
    print "{c}"
    print "{sub}"
    mut k = NoopKernel()
    kernel:
        k(block = 1)
    print "{k.out[0]}"
"#);

    let manifest = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("metal_codegen").join(test_name).join("test_metal").join("Cargo.toml");
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected `main`'s string indexing/slicing to compile and run cleanly \
         under `--target metal` even though `main` also dispatches a real \
         kernel (which is what routes it through the backend's own custom \
         emitter instead of the general pipeline), but it failed:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    let expected = "e\nhello\n1";
    assert_eq!(
        stdout.trim_end(), expected,
        "expected the char-safe single-index result (\"e\"), the char-safe \
         range-slice result (\"hello\"), and the kernel's own dispatched \
         output (\"1\");\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
}

// ─── host — kernel-less program still gets a real `fn main()` ────────────────

// `rename_top_level_main` always renames the user's `fn main()` to
// `boring_main` before this backend's own host emitter runs (see
// `metal::mod`'s `transpile_metal`) so the general pipeline's kernel-aware
// codegen never collides with a function literally named `main`. Every other
// target (wgpu, cuda) then unconditionally emits a real `fn main()` wrapper
// that calls `boring_main()` -- but this backend's own `emit_program` used to
// gate the entire `fn main()` block behind `self.screen_var.is_some() ||
// top_level_kernel_touching || !kernel_names.is_empty() || program.items...
// Stmt/Let`, with no `has_boring_main` case at all. A program with no
// `Screen`, no `kernel` declarations, and no bare top-level statement/let --
// i.e. a plain `def main():` and nothing else -- satisfied none of those, so
// `fn main()` (and the call to `boring_main()`) was silently omitted
// entirely, leaving `boring_main` defined but never called and the crate
// missing an entry point (`error[E0601]: main function not found`). Confirmed
// this test fails to compile with exactly that error against the pre-fix
// code.
#[test]
fn real_metal_target_kernel_less_program_gets_main_and_runs() {
    let test_name = "kernel_less_program_gets_main";
    let (_msl, rs, _toml) = run_metal(test_name, r#"
def main() throws:
    print "hi"
"#);

    assert!(
        rs.contains("fn main("),
        "expected a real `fn main()` wrapper to be emitted even though this \
         program has no Screen, no kernel declarations, and no bare \
         top-level statement/let;\ngot:\n{rs}"
    );
    assert!(
        rs.contains("boring_main()"),
        "expected the generated `fn main()` to actually call `boring_main()`;\ngot:\n{rs}"
    );

    let manifest = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("metal_codegen").join(test_name).join("test_metal").join("Cargo.toml");
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated Metal project for a kernel-less program to \
         compile AND run to completion (no real GPU touched -- this program \
         never constructs or dispatches a kernel), but it failed:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert_eq!(
        stdout.trim_end(), "hi",
        "expected `boring_main()` to actually run;\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
}

// ─── device — multi-branch if-expression inside a kernel body ────────────────

// A kernel `def()` body's `if`/`elif`/.../`else` *expression* (assigned to a
// local, unlike an `if` used as a statement) used to lower to a single
// ternary that only ever checked the FIRST condition, unconditionally using
// the `else` branch's value otherwise -- every `elif` branch's condition
// AND value were silently dropped from the generated MSL entirely, not
// merely miscompiled. Two elif branches is the minimum that actually proves
// nesting: one elif could (in principle, though it wasn't the case here)
// still be papered over by a checker or interpreter path; three branches
// beyond the first `if` leaves no doubt every one of them is independently
// reachable in the emitted ternary chain.
#[test]
fn device_multi_branch_if_expression_emits_full_chain_not_first_condition_only() {
    let (msl, _rs) = metal_codegen("multi_branch_if_expr", r#"
kernel FourWay:
    let [int]'global inp
    mut [int]'unified out
    let int n

    init([int]'global i, int nn):
        inp = i
        n = nn

    def ():
        let idx = gpu.thread.x
        if idx < n:
            let v = inp[idx]
            let r = if v == 0:
                100
            elif v == 1:
                200
            elif v == 2:
                300
            else:
                400
            out[idx] = r
"#);
    assert!(
        msl.contains("((v == 0) ? 100 : ((v == 1) ? 200 : ((v == 2) ? 300 : 400)))"),
        "expected all four branches (if + 2 elif + else) to be nested in the \
         generated ternary chain, not collapsed to just the first condition \
         with the else value as an unconditional fallback (e.g. \
         `((v == 0) ? 100 : 400)`, silently dropping both elif branches);\n\
         got:\n{msl}"
    );
}

// Real end-to-end verification of the above fix: `boring build --target
// metal` + `cargo build` + actually running the resulting binary, which
// dispatches a kernel whose body evaluates a 4-way if/elif/elif/else
// expression per-thread and writes the result back to a GPU buffer. This is
// the only way to catch this bug class for certain -- the buggy generated
// MSL compiles and runs successfully, it just silently computes the wrong
// value for every thread that should have taken an elif branch (confirmed:
// pre-fix this printed "100 400 400 400" instead of "100 200 300 400").
#[test]
fn real_gpu_multi_branch_if_expression_computes_every_branch_correctly() {
    let test_name = "multi_branch_if_expr_dispatch";
    let (_msl, _rs, _toml) = run_metal(test_name, r#"
kernel FourWay:
    let [int]'global inp
    mut [int]'unified out
    let int n

    init([int]'global i, int nn):
        inp = i
        n = nn
        out = [0 for ..<nn]

    def ():
        let idx = gpu.thread.x
        if idx < n:
            let v = inp[idx]
            let r = if v == 0:
                100
            elif v == 1:
                200
            elif v == 2:
                300
            else:
                400
            out[idx] = r

def main() throws:
    let inp = [0, 1, 2, 3]
    mut k = FourWay(inp, 4)
    kernel:
        k(block = 4, grid = 1)
    print "{k.out[0]} {k.out[1]} {k.out[2]} {k.out[3]}"
"#);

    let manifest = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("metal_codegen").join(test_name).join("test_metal").join("Cargo.toml");
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated Metal project (a per-thread 4-way \
         if/elif/elif/else kernel body) to build AND run to completion \
         against a real Metal GPU, but it failed:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    let expected = "100 200 300 400";
    assert_eq!(
        stdout.trim_end(), expected,
        "expected every branch (if + both elif + else) to compute its own \
         value for the matching thread, not fall through to the else value \
         for any elif-matching thread;\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
}

#[test]
fn device_gpu_warp_builtins_map_correctly_camel_case() {
    let (msl, _) = metal_codegen("gpu_warp_builtins_camel_case", r#"
kernel W:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        let lane = gpu.warp.lane
        let size = gpu.warp.size
        gpu.warp.sync()
        let a = gpu.warp.shuffleDown(buf[tid], 1)
        let b = gpu.warp.shuffleUp(buf[tid], 1)
        let c = gpu.warp.shuffleXor(buf[tid], 1)
        let d = gpu.warp.shuffle(buf[tid], 0)
        buf[tid] = a + b + c + d + lane + size
"#);
    assert!(msl.contains("[[thread_index_in_simdgroup]]"),
        "expected [[thread_index_in_simdgroup]];\ngot:\n{msl}");
    assert!(msl.contains("[[threads_per_simdgroup]]"),
        "expected [[threads_per_simdgroup]];\ngot:\n{msl}");
    assert!(msl.contains("__simd_lane_id"), "expected __simd_lane_id;\ngot:\n{msl}");
    assert!(msl.contains("__simd_size"), "expected __simd_size;\ngot:\n{msl}");
    assert!(msl.contains("simdgroup_barrier(mem_flags::mem_none)"),
        "expected simdgroup_barrier;\ngot:\n{msl}");
    assert!(msl.contains("simd_shuffle_down("), "expected simd_shuffle_down;\ngot:\n{msl}");
    assert!(msl.contains("simd_shuffle_up("), "expected simd_shuffle_up;\ngot:\n{msl}");
    assert!(msl.contains("simd_shuffle_xor("), "expected simd_shuffle_xor;\ngot:\n{msl}");
    assert!(msl.contains("simd_shuffle("), "expected simd_shuffle;\ngot:\n{msl}");
}

#[test]
fn device_gpu_block_dim_maps_correctly_camel_case() {
    let (msl, _) = metal_codegen("gpu_block_dim_camel_case", r#"
kernel B:
    mut [float]'unified buf
    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.blockDim.x
        buf[i] = buf[i] * 2.0
"#);
    assert!(msl.contains("__block_pos.x"),
        "expected __block_pos.x for gpu.block.x;\ngot:\n{msl}");
    assert!(msl.contains("__block_dim.x"),
        "expected __block_dim.x for gpu.blockDim.x;\ngot:\n{msl}");
}

// ─── device — 'actor'global atomics ──────────────────────────────────────────
