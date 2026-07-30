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
    // 'sync with a fixed-size literal init — device emitter declares __shared__
    // inside the kernel body rather than as a pointer parameter.
    let (hip, _) = rocm_codegen("shared_static", r#"
kernel S:
    mut [float]'unified out
    let [float]'sync scratch
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
    let [float]'sync  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(hip.contains("extern __shared__"),
        "expected extern __shared__ for dynamic 'sync Array;\ngot:\n{hip}");
    assert!(!hip.contains("scratch* scratch"),
        "dynamic 'sync must not appear as a kernel parameter;\ngot:\n{hip}");
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
    let [float]'sync  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(!rs.contains("scratch: DeviceBuffer"),
        "'sync field must not appear as DeviceBuffer in Rust struct;\ngot:\n{rs}");
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
    let [float]'sync  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(rs.contains("block_dim.0 as usize * 8"),
        "expected block_dim.0 * sizeof(f64)=8 for dynamic 'sync;\ngot:\n{rs}");
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
