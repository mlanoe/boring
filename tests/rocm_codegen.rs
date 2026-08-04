// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// ROCm/HIP codegen snapshot tests.
//
// These tests verify the text emitted by `boring build --target rocm` without
// requiring hipcc or a real AMD GPU. Each test:
//   1. Writes a Boring source snippet to a temp file.
//   2. Invokes `boring build --target rocm <file>`.
//   3. Reads the generated kernels/main.hip and src/main.rs.
//   4. Asserts that the generated text contains the expected patterns.
//
// Mirrors tests/cuda_codegen.rs -- rocm/device.rs and rocm/host.rs are
// near-verbatim clones of the cuda emitters (HIP C++ is source-compatible
// with CUDA C, and host.rs hand-rolls a HIP FFI wrapper shaped like cudarc's
// own API), so most assertions differ only in type/function names
// (DeviceBuffer vs CudaSlice, HipContext vs CudaContext, hipcc vs nvcc, ...).
//
// Run with:
//   cargo test --test rocm_codegen

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Invoke `boring build --target rocm <file>` and return the generated
/// (kernels/main.hip, src/main.rs) text pair.
///
/// boring names the output directory `<stem>_rocm` next to the source file,
/// so we place the source in a dedicated temp dir and read from there.
fn rocm_codegen(test_name: &str, src: &str) -> (String, String) {
    let (hip, rs, _, _) = run_rocm(test_name, src);
    (hip, rs)
}

fn build_rs_and_toml(test_name: &str, src: &str) -> (String, String) {
    let (_, _, build, toml) = run_rocm(test_name, src);
    (build, toml)
}

fn run_rocm(test_name: &str, src: &str) -> (String, String, String, String) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("rocm_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // Source file named "test.br" → boring creates "test_rocm/" next to it.
    let br_file  = tmp.join("test.br");
    let rocm_dir = tmp.join("test_rocm");
    fs::write(&br_file, src).unwrap();

    let result = Command::new(bin)
        .args(["build", "--target", "rocm"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{test_name}] failed to invoke boring: {e}"));

    assert!(
        result.status.success(),
        "[{test_name}] boring build --target rocm failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let read = |rel: &str| fs::read_to_string(rocm_dir.join(rel)).unwrap_or_default();
    (
        read("kernels/main.hip"),
        read("src/main.rs"),
        read("build.rs"),
        read("Cargo.toml"),
    )
}

// ─── device — kernel signature ───────────────────────────────────────────────

#[test]
fn device_unified_field_becomes_pointer_param() {
    let (hip, _) = rocm_codegen("unified_ptr", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(hip.contains("__global__ void Scale_kernel(double* buf)"),
        "expected pointer param for 'unified;\ngot:\n{hip}");
}

#[test]
fn device_global_field_becomes_pointer_param() {
    let (hip, _) = rocm_codegen("global_ptr", r#"
kernel G:
    mut [float]'global buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] + 1.0
"#);
    assert!(hip.contains("__global__ void G_kernel(double* buf)"),
        "expected pointer param for 'global;\ngot:\n{hip}");
}

#[test]
fn device_const_scalar_becomes_value_param() {
    let (hip, _) = rocm_codegen("const_scalar", r#"
kernel C:
    mut [float]'unified buf
    let float'const     factor
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * factor
"#);
    assert!(hip.contains("const double factor"),
        "expected const scalar param;\ngot:\n{hip}");
}

#[test]
fn device_shared_static_becomes_shared_decl_not_param() {
    // 'actor with a fixed-size literal init — device emitter declares __shared__
    // inside the kernel body rather than as a pointer parameter.
    let (hip, _) = rocm_codegen("shared_static", r#"
kernel S:
    mut [float]'unified out
    let [float]'actor scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[tid]
"#);
    assert!(hip.contains("__shared__") || hip.contains("extern __shared__"),
        "expected __shared__ declaration;\ngot:\n{hip}");
    assert!(!hip.contains("scratch* scratch"),
        "__shared__ field must not appear as a kernel parameter;\ngot:\n{hip}");
}

#[test]
fn device_shared_dynamic_becomes_extern_shared() {
    let (hip, _) = rocm_codegen("shared_dynamic", r#"
kernel D:
    mut [float]'unified out
    let [float]'actor  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(hip.contains("extern __shared__"),
        "expected extern __shared__ for dynamic 'actor Array;\ngot:\n{hip}");
    assert!(!hip.contains("scratch* scratch"),
        "dynamic 'actor must not appear as a kernel parameter;\ngot:\n{hip}");
}

#[test]
fn device_gpu_builtins_map_correctly() {
    let (hip, _) = rocm_codegen("gpu_builtins", r#"
kernel B:
    mut [float]'unified buf
    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] = buf[i] * 2.0
"#);
    assert!(hip.contains("threadIdx.x"), "expected threadIdx.x;\ngot:\n{hip}");
    assert!(hip.contains("blockIdx.x"),  "expected blockIdx.x;\ngot:\n{hip}");
    assert!(hip.contains("blockDim.x"),  "expected blockDim.x;\ngot:\n{hip}");
}

#[test]
fn device_gpu_warp_builtins_map_correctly() {
    let (hip, _) = rocm_codegen("gpu_warp_builtins", r#"
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
    assert!(hip.contains("warpSize"), "expected warpSize;\ngot:\n{hip}");
    assert!(hip.contains("% warpSize"), "expected lane linearization mod warpSize;\ngot:\n{hip}");
    assert!(hip.contains("__syncwarp(0xffffffff)"), "expected __syncwarp;\ngot:\n{hip}");
    assert!(hip.contains("__shfl_down_sync(0xffffffff,"), "expected __shfl_down_sync;\ngot:\n{hip}");
    assert!(hip.contains("__shfl_up_sync(0xffffffff,"), "expected __shfl_up_sync;\ngot:\n{hip}");
    assert!(hip.contains("__shfl_xor_sync(0xffffffff,"), "expected __shfl_xor_sync;\ngot:\n{hip}");
    assert!(hip.contains("__shfl_sync(0xffffffff,"), "expected __shfl_sync;\ngot:\n{hip}");
}

// ─── host — struct and constructor ───────────────────────────────────────────

#[test]
fn host_unified_field_becomes_device_buffer() {
    let (_, rs) = rocm_codegen("host_device_buffer", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("buf: DeviceBuffer<f64>"),
        "expected DeviceBuffer field;\ngot:\n{rs}");
}

#[test]
fn host_init_uploads_unified_via_htod() {
    let (_, rs) = rocm_codegen("host_htod", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("clone_htod"),
        "expected clone_htod in constructor;\ngot:\n{rs}");
}

#[test]
fn host_array_fill_becomes_alloc_zeros() {
    let (_, rs) = rocm_codegen("host_alloc_zeros", r#"
kernel Z:
    mut [float]'unified buf
    init(int n):
        buf = [0.0, 0.0, 0.0, 0.0]
    def ():
        let tid = gpu.thread.x
        buf[tid] = 0.0
"#);
    assert!(rs.contains("clone_htod"),
        "expected literal array upload;\ngot:\n{rs}");
}

#[test]
fn host_kernel_new_inlines_top_level_scalar() {
    // Same fix as cuda::host's top_level_scalars: a top-level scalar `let`
    // referenced inside a kernel's `init(...)` body must be inlined as its
    // literal value, since `init`'s Rust `fn new` is its own impl block,
    // scope-wise separate from `fn main()`.
    let (_, rs) = rocm_codegen("host_kernel_new_scalar", r#"
let n = 1000

kernel VectorAdd:
    let [int]'global  a
    mut [int]'unified result

    init([int]'global input_a):
        a = input_a
        result = [0 for ..n]

    def ():
        let i = gpu.thread.x
        result[i] = a[i]

var host_a = [i for i in 0..n]
var k = VectorAdd(host_a)
kernel:
    k(block = 256)
"#);
    assert!(rs.contains("alloc_zeros::<isize>(1000 as usize)"),
        "expected the top-level scalar `n` inlined as its literal value inside `VectorAdd::new`;\ngot:\n{rs}");
    let new_fn = rs.split("fn new(").nth(1).unwrap_or("");
    assert!(!new_fn.contains("(n as usize)"),
        "`VectorAdd::new` must not reference bare `n` -- it isn't in scope there;\ngot:\n{rs}");
}

#[test]
fn host_shared_field_absent_from_rust_struct() {
    let (_, rs) = rocm_codegen("host_shared_absent", r#"
kernel S:
    mut [float]'unified out
    let [float]'actor  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(!rs.contains("scratch: DeviceBuffer"),
        "'actor field must not appear as DeviceBuffer in Rust struct;\ngot:\n{rs}");
}

// ─── host — __boring_launch ───────────────────────────────────────────────────

#[test]
fn host_boring_launch_has_no_smem_param() {
    let (_, rs) = rocm_codegen("launch_no_smem_param", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    // grid_dim is Option here because Scale's first field is a 'unified array (auto grid sizing).
    assert!(rs.contains("fn __boring_launch(mut self, block_dim: (u32,u32,u32), grid_dim: Option<(u32,u32,u32)>, after: &[&Arc<HipStream>], priority: i32)"),
        "expected __boring_launch signature with after and priority params;\ngot:\n{rs}");
}

#[test]
fn host_no_dynamic_shared_smem_is_zero() {
    let (_, rs) = rocm_codegen("smem_zero", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("let smem_bytes: u32 = 0u32"),
        "expected smem_bytes = 0 for no dynamic shared field;\ngot:\n{rs}");
}

#[test]
fn host_dynamic_shared_smem_uses_block_x_times_elem_size() {
    let (_, rs) = rocm_codegen("smem_dynamic", r#"
kernel D:
    mut [float]'unified out
    let [float]'actor  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(rs.contains("block_dim.0 as usize * 8"),
        "expected block_dim.0 * sizeof(f64)=8 for dynamic 'actor;\ngot:\n{rs}");
}

#[test]
fn host_unified_arg_passed_as_mut_ref_in_launch() {
    let (_, rs) = rocm_codegen("launch_mut_ref", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("(&mut self.buf)"),
        "expected &mut self.buf for 'unified launch arg;\ngot:\n{rs}");
}

#[test]
fn host_const_arg_passed_as_immutable_ref_in_launch() {
    let (_, rs) = rocm_codegen("launch_const_ref", r#"
kernel C:
    mut [float]'unified buf
    let float'const     factor
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * factor
"#);
    assert!(rs.contains("(&self.factor)"),
        "expected &self.factor for 'const launch arg;\ngot:\n{rs}");
}

// ─── host — kernel call site ─────────────────────────────────────────────────

#[test]
fn host_kernel_launch_calls_boring_launch_without_smem() {
    let (_, rs) = rocm_codegen("call_site", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Scale(data)
kernel:
    k(block = 2)
"#);
    // Auto grid sizing: Scale's first field is a 'unified array, so grid is None when omitted.
    assert!(rs.contains("__boring_launch((2 as u32, 1, 1), None, &[], 0i32)?"),
        "expected __boring_launch((block_dim), None, &[], 0i32) with auto grid;\ngot:\n{rs}");
}

// ─── infrastructure ───────────────────────────────────────────────────────────

#[test]
fn build_rs_invokes_hipcc_and_sets_co_path() {
    let (build, _) = build_rs_and_toml("infra_build_rs", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(build.contains("hipcc"), "build.rs must invoke hipcc;\ngot:\n{build}");
    assert!(build.contains("BORING_HIP_CO_PATH"),
        "build.rs must set BORING_HIP_CO_PATH;\ngot:\n{build}");
}

#[test]
fn cargo_toml_has_no_external_gpu_crate_dependency() {
    // Unlike cuda (cudarc), rocm hand-rolls its own FFI/safe-wrapper layer in
    // host.rs (no mature safe Rust HIP crate exists) -- see rocm/mod.rs's
    // emit_cargo_toml doc comment. Cargo.toml should therefore declare no
    // GPU crate under [dependencies].
    let (_, toml) = build_rs_and_toml("infra_cargo_toml", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(toml.contains("[dependencies]"),
        "Cargo.toml must have a [dependencies] section;\ngot:\n{toml}");
    let deps = toml.split("[dependencies]").nth(1).unwrap_or("");
    assert!(deps.trim().is_empty(),
        "rocm Cargo.toml should declare no external GPU crate;\ngot:\n{toml}");
}

// ─── 2D block / grid launch ───────────────────────────────────────────────────

#[test]
fn host_2d_block_tuple_becomes_dim3() {
    let (_, rs) = rocm_codegen("launch_2d_block", r#"
kernel Grid2D:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Grid2D(data)
kernel:
    k(block = (16, 16))
"#);
    assert!(rs.contains("(16 as u32, 16 as u32, 1)"),
        "expected (16, 16, 1) dim3 for 2D block tuple;\ngot:\n{rs}");
}

#[test]
fn host_3d_block_tuple_becomes_dim3() {
    let (_, rs) = rocm_codegen("launch_3d_block", r#"
kernel Grid3D:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Grid3D(data)
kernel:
    k(block = (8, 8, 4))
"#);
    assert!(rs.contains("(8 as u32, 8 as u32, 4 as u32)"),
        "expected (8, 8, 4) dim3 for 3D block tuple;\ngot:\n{rs}");
}

// ─── automatic grid sizing ─────────────────────────────────────────────────────

#[test]
fn host_auto_grid_sizing_computed_from_first_array_len() {
    let (_, rs) = rocm_codegen("auto_grid", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Scale(data)
kernel:
    k(block = 2)
"#);
    assert!(rs.contains("((n + block_dim.0 - 1) / block_dim.0, 1, 1)"),
        "expected auto grid calculation;\ngot:\n{rs}");
    assert!(rs.contains("grid_dim: Option<(u32,u32,u32)>"),
        "expected optional grid_dim param;\ngot:\n{rs}");
    // Without `grid =`, the call site passes None.
    assert!(rs.contains("__boring_launch((2 as u32, 1, 1), None, &[], 0i32)"),
        "expected None grid arg at call site;\ngot:\n{rs}");
}

// ─── [T, N]'local fixed-size local arrays ──────────────────────────────────────

#[test]
fn device_local_array_declared_in_body() {
    let (hip, _) = rocm_codegen("local_array", r#"
kernel L:
    mut [float]'unified out
    let [float, 8]'local tmp
    def ():
        let tid = gpu.thread.x
        out[tid] = tmp[0]
"#);
    assert!(hip.contains("double tmp[8];"),
        "expected fixed-size local array declaration;\ngot:\n{hip}");
}

// ─── atomics via 'actor'global ──────────────────────────────────────────────────

#[test]
fn device_actor_global_uses_atomic_add() {
    let (hip, _) = rocm_codegen("atomic_add", r#"
kernel A:
    mut [int]'actor'global counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(hip.contains("atomicAdd(&counts[0]"),
        "expected atomicAdd for 'actor'global compound assign;\ngot:\n{hip}");
    assert!(hip.contains("int64_t* counts"),
        "expected 'actor'global field as pointer param;\ngot:\n{hip}");
}

// ─── 'actor'unified — atomics on host+device DRAM ──────────────────────────────

#[test]
fn device_actor_unified_uses_atomic_add() {
    let (hip, rs) = rocm_codegen("actor_unified_add", r#"
kernel A:
    mut [int]'actor'unified counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(hip.contains("atomicAdd(&counts[0]"),
        "expected atomicAdd for 'actor'unified compound assign;\ngot:\n{hip}");
    assert!(hip.contains("int64_t* counts"),
        "expected 'actor'unified field as pointer param;\ngot:\n{hip}");
    assert!(rs.contains("fn read_counts(&self)"),
        "expected a host-side read_counts() accessor for 'actor'unified;\ngot:\n{rs}");
}

// ─── print in kernel → printf ───────────────────────────────────────────────────

#[test]
fn device_print_emits_printf() {
    let (hip, _) = rocm_codegen("device_printf", r#"
kernel P:
    mut [int]'unified buf
    def ():
        let int x = 5
        print "val = {x}"
"#);
    assert!(hip.contains("printf(\"val = %lld\\n\", x);"),
        "expected printf with %lld for int;\ngot:\n{hip}");
}

// ─── GPU as built-in type ─────────────────────────────────────────────────────

#[test]
fn gpu_n_emits_boring_gpu_ctx_n() {
    let (_, rs) = rocm_codegen("gpu_n", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let g = GPU(0)
"#);
    assert!(rs.contains("boring_gpu_ctx_n(0 as usize)?"),
        "expected boring_gpu_ctx_n(0) for GPU(0);\ngot:\n{rs}");
}

#[test]
fn gpu_all_emits_device_iterator() {
    let (_, rs) = rocm_codegen("gpu_all", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let devs = GPU.all()
"#);
    assert!(rs.contains("HipContext::device_count"),
        "expected HipContext::device_count in GPU.all() expansion;\ngot:\n{rs}");
}

// ─── Multi-GPU: new(g) Scale(data) ─────────────────────────────────────────────

#[test]
fn new_with_arena_emits_scale_new_with_device() {
    let (_, rs) = rocm_codegen("new_arena", r#"
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
    assert!(rs.contains("Scale::new("),
        "expected Scale::new( for new(g0) Scale(data);\ngot:\n{rs}");
    // The arena expression (g0) must appear as the first arg to Scale::new.
    assert!(rs.contains("Scale::new(g0,") || rs.contains("Scale::new(g0 ,"),
        "expected g0 as first arg to Scale::new;\ngot:\n{rs}");
    // `data` is a plain host Vec<f64> at this point -- the arena-qualified
    // constructor call must upload it via clone_htod exactly like the plain
    // `Scale(data)` call site does (see `emit_kernel_ctor_args`), not pass it
    // straight through as a bare Vec where Scale::new expects a
    // DeviceBuffer<f64> (mirrors the identical bug fixed in cuda::host's
    // `new(g) Scale(data)` handling).
    assert!(rs.contains("clone_htod"),
        "expected clone_htod at the new(g0) Scale(data) call site;\ngot:\n{rs}");
}

// ─── sequential kernel launches in kernel: block ───────────────────────────────

#[test]
fn sequential_launches_in_kernel_block() {
    let (_, rs) = rocm_codegen("sequential_launches", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k1 = Scale(data)
mut k2 = Scale(data)
kernel:
    k1(block = 2)
    k2(block = 2)
"#);
    // Both launches should appear; sequential ordering is guaranteed by the block.
    assert!(rs.contains("k1.__boring_launch("),
        "expected k1.__boring_launch in kernel: block;\ngot:\n{rs}");
    assert!(rs.contains("k2.__boring_launch("),
        "expected k2.__boring_launch in kernel: block;\ngot:\n{rs}");
    let pos_k1 = rs.find("k1.__boring_launch(").unwrap();
    let pos_k2 = rs.find("k2.__boring_launch(").unwrap();
    assert!(pos_k1 < pos_k2, "k1 launch should appear before k2 launch;\ngot:\n{rs}");
}

#[test]
fn no_after_passes_empty_slice() {
    let (_, rs) = rocm_codegen("no_after_dep", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Scale(data)
kernel:
    k(block = 2)
"#);
    // Without `after =`, the call site passes &[] and priority defaults to 0.
    assert!(rs.contains("__boring_launch((2 as u32, 1, 1), None, &[], 0i32)"),
        "expected &[] as after arg and 0i32 priority when not specified;\ngot:\n{rs}");
}

#[test]
fn after_dependency_references_the_real_stream_field_name() {
    // `after = ka` used to generate `&ka.stream` -- the kernel struct's real
    // field is `__stream` (double-underscore; mirrors the identical bug fixed
    // in cuda::host -- `KernelHandle<T>`'s own field really is named `stream`,
    // a different type entirely, which is presumably how this went unnoticed).
    // This exact combination (an `after =` dependency on another kernel struct
    // variable) had no prior test coverage at all.
    let (_, rs) = rocm_codegen("after_dep_stream_field", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut ka = Scale(data)
mut kb = Scale(data)
kernel:
    ka(block = 2)
    kb(block = 2, after = ka)
"#);
    assert!(rs.contains("&[&ka.__stream]"),
        "expected the after = dependency to reference ka.__stream, not ka.stream;\ngot:\n{rs}");
    assert!(!rs.contains("&ka.stream]"),
        "must not reference the nonexistent ka.stream field;\ngot:\n{rs}");
}

#[test]
fn dtod_ctor_arg_uses_real_device_to_device_copy_not_a_bare_move() {
    // Mirrors the identical fix in cuda::host: `Scale(k1.buf)` used to move
    // `k1.buf` straight into the new kernel's constructor -- correct only if
    // `k1` is never used again, which this test deliberately violates
    // (`k1` is dispatched again and read afterward). `.clone()` here is a
    // real device-to-device copy (`DeviceBuffer::clone()`'s own
    // `hipMemcpyDtoDAsync`, see that impl's doc comment), not a host round
    // trip, so `k1` stays independently usable.
    let (_, rs) = rocm_codegen("dtod_candidate", r#"
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
    assert!(rs.contains("Scale::new(boring_gpu_ctx(), k1.buf.clone())"),
        "expected a real device-to-device .clone(), not a bare move, so k1 stays usable afterward, and not a D2H+H2D round trip;\ngot:\n{rs}");
}

// ─── GPU device properties ────────────────────────────────────────────────────

#[test]
fn host_gpu_name_property() {
    let (_, rs) = rocm_codegen("gpu_name", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let g = GPU(0)
let _name = g.name()
"#);
    assert!(rs.contains(".name()?"),
        "expected .name()? for GPU name property;\ngot:\n{rs}");
}

#[test]
fn host_gpu_total_mem_property() {
    let (_, rs) = rocm_codegen("gpu_total_mem", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let g = GPU(0)
let _mem = g.totalMem()
"#);
    assert!(rs.contains(".total_mem()?"),
        "expected .total_mem()? for GPU total_mem property;\ngot:\n{rs}");
}

#[test]
fn host_gpu_compute_capability_property() {
    let (_, rs) = rocm_codegen("gpu_compute", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let g = GPU(0)
let _cc = g.computeCapability()
"#);
    assert!(rs.contains(".compute_capability()?"),
        "expected .compute_capability()? for GPU compute property;\ngot:\n{rs}");
}

// ─── GPU().warpSize()/maxThreads()/maxSharedMem() via build-time header probe ──

#[test]
fn gpu_device_attribute_queries_use_probed_constants() {
    // hipDeviceAttribute_t enum values aren't ABI-stable across ROCm
    // versions, so rather than hardcode an attribute ID, build.rs compiles
    // and runs a tiny C probe against the locally installed
    // hip_runtime_api.h and bakes the real values into
    // OUT_DIR/boring_hip_attrs.rs (see rocm/mod.rs's
    // probe_hip_device_attributes). HipContext::warp_size/max_threads/
    // max_shared_mem read those generated constants via hipDeviceGetAttribute
    // instead of a guessed literal; if the probe couldn't run, the constants
    // fall back to -1 and get_attribute() reports a clean error.
    let (_, rs) = rocm_codegen("warp_size_probe", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let g = GPU(0)
let _ws = g.warpSize()
let _mt = g.maxThreads()
let _sm = g.maxSharedMem()
"#);
    assert!(rs.contains(".warp_size()?"),
        "expected .warp_size()? call site for GPU().warpSize();\ngot:\n{rs}");
    assert!(rs.contains(".max_threads()?"),
        "expected .max_threads()? call site for GPU().maxThreads();\ngot:\n{rs}");
    assert!(rs.contains(".max_shared_mem()?"),
        "expected .max_shared_mem()? call site for GPU().maxSharedMem();\ngot:\n{rs}");
    assert!(rs.contains("include!(concat!(env!(\"OUT_DIR\"), \"/boring_hip_attrs.rs\"))"),
        "expected the prelude to include! the build-time-probed attribute constants;\ngot:\n{rs}");
    assert!(rs.contains("self.get_attribute(BORING_HIP_ATTR_WARP_SIZE"),
        "expected warp_size() to read the probed BORING_HIP_ATTR_WARP_SIZE constant;\ngot:\n{rs}");
    assert!(rs.contains("hipDeviceGetAttribute"),
        "expected hipDeviceGetAttribute in the FFI declarations;\ngot:\n{rs}");
}

#[test]
fn build_rs_probes_hip_device_attributes() {
    let (build, _) = build_rs_and_toml("infra_attr_probe", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(build.contains("hipDeviceAttributeWarpSize"),
        "build.rs must probe hipDeviceAttributeWarpSize from the local hip_runtime_api.h;\ngot:\n{build}");
    assert!(build.contains("hipDeviceAttributeMaxThreadsPerBlock"),
        "build.rs must probe hipDeviceAttributeMaxThreadsPerBlock;\ngot:\n{build}");
    assert!(build.contains("hipDeviceAttributeSharedMemPerBlock"),
        "build.rs must probe hipDeviceAttributeSharedMemPerBlock;\ngot:\n{build}");
    assert!(build.contains("boring_hip_attrs.rs"),
        "build.rs must write the probed values to boring_hip_attrs.rs;\ngot:\n{build}");
}

// ─── saxpy example ────────────────────────────────────────────────────────────

#[test]
fn example_saxpy() {
    let src = std::fs::read_to_string("examples/saxpy.br").expect("examples/saxpy.br not found");
    let (hip, rs) = rocm_codegen("saxpy_example", &src);

    // Device kernel
    assert!(hip.contains("__global__ void Saxpy_kernel("), "missing Saxpy_kernel;\ngot:\n{hip}");
    assert!(hip.contains("const double alpha"),            "missing const scalar alpha;\ngot:\n{hip}");
    assert!(hip.contains("const double* x"),              "missing x param;\ngot:\n{hip}");
    assert!(hip.contains("double* y"),                    "missing y param;\ngot:\n{hip}");
    assert!(hip.contains("y[i] = ((alpha * x[i]) + y[i])"), "missing saxpy body;\ngot:\n{hip}");

    // Host struct
    assert!(rs.contains("struct Saxpy"),          "missing struct Saxpy;\ngot:\n{rs}");
    assert!(rs.contains("alpha: f64"),                "missing alpha field;\ngot:\n{rs}");
    assert!(rs.contains("x: DeviceBuffer<f64>"),      "missing x field;\ngot:\n{rs}");
    assert!(rs.contains("y: DeviceBuffer<f64>"),      "missing y field;\ngot:\n{rs}");

    // Host main: print, float cast, enumerate loop
    assert!(rs.contains("println!("),                      "print not translated to println!;\ngot:\n{rs}");
    assert!(rs.contains("as f64)"),                        "float() cast not translated;\ngot:\n{rs}");
    assert!(rs.contains(".iter().enumerate()"),            "for i,v not translated to enumerate;\ngot:\n{rs}");
}

// ─── KernelHandle must_use ─────────────────────────────────────────────────────

#[test]
fn kernel_handle_is_must_use() {
    // Dropping a `KernelHandle<T>` without `.wait`/`.inner` used to compile
    // silently -- `#[must_use]` turns that into a compiler warning instead.
    let (_, rs) = rocm_codegen("kernel_handle_must_use", r#"
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

// ─── explicit `grid =` override ────────────────────────────────────────────────

#[test]
fn explicit_grid_arg_is_not_silently_dropped() {
    // Same bug as the identical fix in cuda::host: `k(block=.., grid=..)`
    // ignored any explicit `grid` label and always passed `None`, silently
    // overriding the caller's 2D/3D grid with 1D-length inference (or
    // `(1,1,1)` with no auto-grid field). `grid` must now be threaded
    // through as `Some((gx, gy, gz))`.
    let (_, rs) = rocm_codegen("explicit_grid", r#"
kernel Scale2D:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0, 3.0, 4.0]
mut k = Scale2D(data)
kernel:
    k(block = (16, 16, 1), grid = (4, 4, 1))
"#);
    assert!(
        rs.contains("k.__boring_launch((16 as u32, 16 as u32, 1 as u32), Some((4 as u32, 4 as u32, 1 as u32)), &[], 0i32)"),
        "explicit grid=(4,4,1) must reach __boring_launch as Some((4,4,1)), not None;\ngot:\n{rs}"
    );
}

// ─── GPU error classification ─────────────────────────────────────────────────

#[test]
fn hip_error_display_includes_classified_category() {
    // Same rationale as the identical CUDA fix: HipError::from_code already
    // calls hipGetErrorString (a real message), but gave no way to tell
    // failure classes apart at a glance, and a real catch-by-variant isn't
    // reachable on this backend (no Stmt::Try support, same gap as `with`).
    // Improves HipError's own Display directly instead of touching every
    // call site -- the numeric codes mirror CUDA's CUresult values (HIP
    // mirrors the CUDA driver API by design), NOT independently verified
    // against a real ROCm install (none available in this dev environment)
    // -- same caveat this backend's docs already carry elsewhere. Verified
    // to compile clean via a real cargo check with a stubbed build step.
    let (_, rs) = rocm_codegen("hip_error_classify", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("fn category(&self) -> &'static str"),
        "expected HipError::category();\ngot:\n{rs}");
    assert!(rs.contains("2   => \"GPU out of memory\","),
        "expected HIP code 2 classified as GPU out of memory;\ngot:\n{rs}");
    assert!(rs.contains("write!(f, \"{}: HIP error {}: {}\", self.category(), self.code, self.message)"),
        "expected Display to prefix the existing message with the classified category;\ngot:\n{rs}");
    // Mirrors the identical CUDA_ERROR_INVALID_VALUE case -- the real
    // rejection an oversized `block =` hits at `hipModuleLaunchKernel`.
    // Boring deliberately does not duplicate this check in the validator
    // or interpreter; it defers entirely to this real runtime classification.
    assert!(rs.contains("1   => \"GPU launch configuration invalid (e.g. block size exceeds device limits)\","),
        "expected HIP code 1 classified as an invalid launch config, covering an oversized block size;\ngot:\n{rs}");
}

// ─── atomic min/max/swap/cas ───────────────────────────────────────────────────

#[test]
fn device_atomic_method_calls_map_to_hip_intrinsics() {
    // Mirrors the identical CUDA fix -- HIP's atomicMin/Max/Exch/CAS have the
    // same names and signatures (HIP is designed as a near-1:1 source-level
    // match for the CUDA driver API).
    let (hip, _) = rocm_codegen("atomic_methods", r#"
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
    assert!(hip.contains("atomicMin(&counts[bucket], 5);"), "expected atomicMin;\ngot:\n{hip}");
    assert!(hip.contains("atomicMax(&counts[bucket], 5);"), "expected atomicMax;\ngot:\n{hip}");
    assert!(hip.contains("atomicExch(&counts[bucket], 0)"), "expected atomicExch;\ngot:\n{hip}");
    assert!(hip.contains("atomicCAS(&counts[bucket], 0, 1)"), "expected atomicCAS;\ngot:\n{hip}");
}

// ─── Labeled multi-dimensional arrays (docs/array-multidim-types.md) ───────
// Mirrors tests/cuda_codegen.rs's own LabeledArray section exactly, with
// `DeviceBuffer<T>` in place of `CudaSlice<T>` for host field types.

#[test]
fn device_labeled_index_lowers_to_row_major_index() {
    let (hip, _) = rocm_codegen("labeled_at", r#"
kernel Img:
    mut [float, width = 4, height = 4]'unified img
    init([float, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(hip.contains("img[c + r * 4]"),
        "expected [width=c,height=r] to lower to row-major c + r*width;\ngot:\n{hip}");
}

#[test]
fn device_labeled_size_lowers_to_literals() {
    let (hip, _) = rocm_codegen("labeled_width_height", r#"
kernel Img:
    mut [float, width = 4, height = 8]'unified img
    init([float, width = 4, height = 8]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        if c < img.size(.width) and r < img.size(.height):
            img[width = c, height = r] = 0.0
"#);
    assert!(hip.contains("c < 4"), "expected .size(.width) to lower to the literal 4;\ngot:\n{hip}");
    assert!(hip.contains("r < 8"), "expected .size(.height) to lower to the literal 8;\ngot:\n{hip}");
}

#[test]
fn device_labeled_array_3_axis_lowers_to_row_major_index() {
    let (hip, _) = rocm_codegen("labeled_3_axis", r#"
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
    assert!(hip.contains("vol[tx + ty * 4 + tz * 16]"),
        "expected [x,y,z] to lower to row-major x + y*4 + z*(4*4);\ngot:\n{hip}");
}

#[test]
fn device_shared_labeled_array_becomes_fixed_shared_decl() {
    let (hip, _) = rocm_codegen("shared_labeled", r#"
kernel Tile:
    mut [float]'unified out
    let [float, width = 4, height = 4]'actor tile
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        out[0] = tile[width = c, height = r]
"#);
    assert!(hip.contains("__shared__ double tile[16];"),
        "expected fixed __shared__ decl sized width*height, not extern __shared__;\ngot:\n{hip}");
}

#[test]
fn kernel_field_labeled_array_bare_unified_is_valid() {
    let (hip, rs) = rocm_codegen("labeled_bare_unified", r#"
kernel Img:
    mut [float, width = 4, height = 4]'unified img
    init([float, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(hip.contains("__global__ void Img_kernel(double* img)"),
        "expected bare 'unified LabeledArray field to compile to a pointer param;\ngot:\n{hip}");
    assert!(rs.contains("img: DeviceBuffer<f64>"),
        "expected bare 'unified LabeledArray field to become a DeviceBuffer host field, same as [T]'unified;\ngot:\n{rs}");
}

#[test]
fn host_labeled_array_field_infers_2d_grid() {
    let (_, rs) = rocm_codegen("labeled_2d_grid", r#"
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
    let (_, rs) = rocm_codegen("dynamic_labeled_2d_grid", r#"
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
    assert!(!rs.contains("self.img.len()"), "should not fall back to 1D length-based grid inference;\ngot:\n{rs}");
}

// ─── .min/.max/.swap/.cas without 'actor — plain, non-atomic fallback ─────────

#[test]
fn atomic_method_calls_degrade_to_plain_read_modify_write_without_actor() {
    // Mirrors the identical CUDA fix -- see that test's own doc.
    let (hip, _) = rocm_codegen("plain_atomic_methods", r#"
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
    assert!(hip.contains("({ auto __old = buf[tid]; buf[tid] = min(buf[tid], (5)); __old; })"),
        "expected plain min via a GNU statement-expression;\ngot:\n{hip}");
    assert!(hip.contains("({ auto __old = buf[tid]; buf[tid] = max(buf[tid], (5)); __old; })"),
        "expected plain max via a GNU statement-expression;\ngot:\n{hip}");
    assert!(hip.contains("({ auto __old = buf[tid]; buf[tid] = (0); __old; })"),
        "expected plain swap via a GNU statement-expression;\ngot:\n{hip}");
    assert!(hip.contains("({ auto __old = buf[tid]; if (__old == (0)) buf[tid] = (1); __old; })"),
        "expected plain cas via a GNU statement-expression;\ngot:\n{hip}");
}
