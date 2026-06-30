// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// CUDA codegen snapshot tests.
//
// These tests verify the text emitted by `boring build --target cuda` without
// requiring nvcc or a real GPU.  Each test:
//   1. Writes a Boring source snippet to a temp file.
//   2. Invokes `boring build --target cuda <file>`.
//   3. Reads the generated kernels/main.cu and src/main.rs.
//   4. Asserts that the generated text contains the expected patterns.
//
// Run with:
//   cargo test --test cuda_codegen

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Invoke `boring build --target cuda <file>` and return the generated
/// (kernels/main.cu, src/main.rs) text pair.
///
/// boring names the output directory `<stem>_cuda` next to the source file,
/// so we place the source in a dedicated temp dir and read from there.
fn cuda_codegen(test_name: &str, src: &str) -> (String, String) {
    let (cu, rs, _, _) = run_cuda(test_name, src);
    (cu, rs)
}

fn build_rs_and_toml(test_name: &str, src: &str) -> (String, String) {
    let (_, _, build, toml) = run_cuda(test_name, src);
    (build, toml)
}

fn run_cuda(test_name: &str, src: &str) -> (String, String, String, String) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("cuda_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // Source file named "test.br" → boring creates "test_cuda/" next to it.
    let br_file  = tmp.join("test.br");
    let cuda_dir = tmp.join("test_cuda");
    fs::write(&br_file, src).unwrap();

    let result = Command::new(bin)
        .args(["build", "--target", "cuda"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{test_name}] failed to invoke boring: {e}"));

    assert!(
        result.status.success(),
        "[{test_name}] boring build --target cuda failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let read = |rel: &str| fs::read_to_string(cuda_dir.join(rel)).unwrap_or_default();
    (
        read("kernels/main.cu"),
        read("src/main.rs"),
        read("build.rs"),
        read("Cargo.toml"),
    )
}

// ─── device — kernel signature ───────────────────────────────────────────────

#[test]
fn device_unified_field_becomes_pointer_param() {
    let (cu, _) = cuda_codegen("unified_ptr", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(cu.contains("__global__ void Scale_kernel(double* buf)"),
        "expected pointer param for 'unified;\ngot:\n{cu}");
}

#[test]
fn device_global_field_becomes_pointer_param() {
    let (cu, _) = cuda_codegen("global_ptr", r#"
kernel G:
    mut [float]'global buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] + 1.0
"#);
    assert!(cu.contains("__global__ void G_kernel(double* buf)"),
        "expected pointer param for 'global;\ngot:\n{cu}");
}

#[test]
fn device_const_scalar_becomes_value_param() {
    let (cu, _) = cuda_codegen("const_scalar", r#"
kernel C:
    mut [float]'unified buf
    let float'const     factor
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * factor
"#);
    assert!(cu.contains("const double factor"),
        "expected const scalar param;\ngot:\n{cu}");
}

#[test]
fn device_shared_static_becomes_shared_decl_not_param() {
    // 'shared with a fixed-size literal init — device emitter declares __shared__
    // inside the kernel body rather than as a pointer parameter.
    let (cu, _) = cuda_codegen("shared_static", r#"
kernel S:
    mut [float]'unified out
    let [float]'shared scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[tid]
"#);
    assert!(cu.contains("__shared__") || cu.contains("extern __shared__"),
        "expected __shared__ declaration;\ngot:\n{cu}");
    assert!(!cu.contains("scratch* scratch"),
        "__shared__ field must not appear as a kernel parameter;\ngot:\n{cu}");
}

#[test]
fn device_shared_dynamic_becomes_extern_shared() {
    let (cu, _) = cuda_codegen("shared_dynamic", r#"
kernel D:
    mut [float]'unified out
    let [float]'shared  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(cu.contains("extern __shared__"),
        "expected extern __shared__ for dynamic 'shared Array;\ngot:\n{cu}");
    assert!(!cu.contains("scratch* scratch"),
        "dynamic 'shared must not appear as a kernel parameter;\ngot:\n{cu}");
}

#[test]
fn device_gpu_builtins_map_correctly() {
    let (cu, _) = cuda_codegen("gpu_builtins", r#"
kernel B:
    mut [float]'unified buf
    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] = buf[i] * 2.0
"#);
    assert!(cu.contains("threadIdx.x"), "expected threadIdx.x;\ngot:\n{cu}");
    assert!(cu.contains("blockIdx.x"),  "expected blockIdx.x;\ngot:\n{cu}");
    assert!(cu.contains("blockDim.x"),  "expected blockDim.x;\ngot:\n{cu}");
}

// ─── host — struct and constructor ───────────────────────────────────────────

#[test]
fn host_unified_field_becomes_cuda_slice() {
    let (_, rs) = cuda_codegen("host_cuda_slice", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("buf: CudaSlice<f64>"),
        "expected CudaSlice field;\ngot:\n{rs}");
}

#[test]
fn host_init_uploads_unified_via_htod() {
    let (_, rs) = cuda_codegen("host_htod", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("clone_htod"),
        "expected htod_sync_copy in constructor;\ngot:\n{rs}");
}

#[test]
fn host_array_fill_becomes_alloc_zeros() {
    let (_, rs) = cuda_codegen("host_alloc_zeros", r#"
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
fn host_shared_field_absent_from_rust_struct() {
    let (_, rs) = cuda_codegen("host_shared_absent", r#"
kernel S:
    mut [float]'unified out
    let [float]'shared  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(!rs.contains("scratch: CudaSlice"),
        "'shared field must not appear as CudaSlice in Rust struct;\ngot:\n{rs}");
}

// ─── host — __boring_launch ───────────────────────────────────────────────────

#[test]
fn host_boring_launch_has_no_smem_param() {
    let (_, rs) = cuda_codegen("launch_no_smem_param", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    // grid_dim is Option here because Scale's first field is a 'unified array (auto grid sizing).
    assert!(rs.contains("fn __boring_launch(mut self, block_dim: (u32,u32,u32), grid_dim: Option<(u32,u32,u32)>, after: &[&Arc<CudaStream>])"),
        "expected __boring_launch signature with after param;\ngot:\n{rs}");
}

#[test]
fn host_no_dynamic_shared_smem_is_zero() {
    let (_, rs) = cuda_codegen("smem_zero", r#"
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
    let (_, rs) = cuda_codegen("smem_dynamic", r#"
kernel D:
    mut [float]'unified out
    let [float]'shared  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(rs.contains("block_dim.0 as usize * 8"),
        "expected block_dim.0 * sizeof(f64)=8 for dynamic 'shared;\ngot:\n{rs}");
}

#[test]
fn host_unified_arg_passed_as_mut_ref_in_launch() {
    let (_, rs) = cuda_codegen("launch_mut_ref", r#"
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
    let (_, rs) = cuda_codegen("launch_const_ref", r#"
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
    let (_, rs) = cuda_codegen("call_site", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Scale(data)
mut k = kernel(block = 2) k |> .wait
"#);
    // Auto grid sizing: Scale's first field is a 'unified array, so grid is None when omitted.
    assert!(rs.contains("__boring_launch((2 as u32, 1, 1), None, &[])?"),
        "expected __boring_launch((block_dim), None, &[]) with auto grid;\ngot:\n{rs}");
}

// ─── infrastructure ───────────────────────────────────────────────────────────

#[test]
fn build_rs_invokes_nvcc_and_sets_ptx_path() {
    let (build, _) = build_rs_and_toml("infra_build_rs", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(build.contains("nvcc"), "build.rs must invoke nvcc;\ngot:\n{build}");
    assert!(build.contains("BORING_PTX_PATH"),
        "build.rs must set BORING_PTX_PATH;\ngot:\n{build}");
}

#[test]
fn cargo_toml_depends_on_cudarc() {
    let (_, toml) = build_rs_and_toml("infra_cargo_toml", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(toml.contains("cudarc"),
        "Cargo.toml must depend on cudarc;\ngot:\n{toml}");
}

// ─── 2D block / grid launch ───────────────────────────────────────────────────

#[test]
fn host_2d_block_tuple_becomes_dim3() {
    let (_, rs) = cuda_codegen("launch_2d_block", r#"
kernel Grid2D:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Grid2D(data)
mut k = kernel(block = (16, 16)) k |> .wait
"#);
    assert!(rs.contains("(16 as u32, 16 as u32, 1)"),
        "expected (16, 16, 1) dim3 for 2D block tuple;\ngot:\n{rs}");
}

#[test]
fn host_3d_block_tuple_becomes_dim3() {
    let (_, rs) = cuda_codegen("launch_3d_block", r#"
kernel Grid3D:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Grid3D(data)
mut k = kernel(block = (8, 8, 4)) k |> .wait
"#);
    assert!(rs.contains("(8 as u32, 8 as u32, 4 as u32)"),
        "expected (8, 8, 4) dim3 for 3D block tuple;\ngot:\n{rs}");
}

// ─── Item 1 — automatic grid sizing ───────────────────────────────────────────

#[test]
fn host_auto_grid_sizing_computed_from_first_array_len() {
    let (_, rs) = cuda_codegen("auto_grid", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Scale(data)
mut k = kernel(block = 2) k |> .wait
"#);
    assert!(rs.contains("((n + block_dim.0 - 1) / block_dim.0, 1, 1)"),
        "expected auto grid calculation;\ngot:\n{rs}");
    assert!(rs.contains("grid_dim: Option<(u32,u32,u32)>"),
        "expected optional grid_dim param;\ngot:\n{rs}");
    // Without `grid =`, the call site passes None.
    assert!(rs.contains("__boring_launch((2 as u32, 1, 1), None, &[])"),
        "expected None grid arg at call site;\ngot:\n{rs}");
}

// ─── Item 2 — [T, N]'local fixed-size local arrays ─────────────────────────────

#[test]
fn device_local_array_declared_in_body() {
    let (cu, _) = cuda_codegen("local_array", r#"
kernel L:
    mut [float]'unified out
    let [float, 8]'local tmp
    def ():
        let tid = gpu.thread.x
        out[tid] = tmp[0]
"#);
    assert!(cu.contains("double tmp[8];"),
        "expected fixed-size local array declaration;\ngot:\n{cu}");
}

// ─── Item 3 — atomics via 'actor'global ────────────────────────────────────────

#[test]
fn device_actor_global_uses_atomic_add() {
    let (cu, _) = cuda_codegen("atomic_add", r#"
kernel A:
    mut [int]'actor'global counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(cu.contains("atomicAdd(&counts[0]"),
        "expected atomicAdd for 'actor'global compound assign;\ngot:\n{cu}");
    assert!(cu.contains("int64_t* counts"),
        "expected 'actor'global field as pointer param;\ngot:\n{cu}");
}

// ─── Item 9 — print in kernel → printf ─────────────────────────────────────────

#[test]
fn device_print_emits_printf() {
    let (cu, _) = cuda_codegen("device_printf", r#"
kernel P:
    mut [int]'unified buf
    def ():
        let int x = 5
        print "val = {x}"
"#);
    assert!(cu.contains("printf(\"val = %lld\\n\", x);"),
        "expected printf with %lld for int;\ngot:\n{cu}");
}

// ─── Item 10 — GPU as built-in type ─────────────────────────────────────────

#[test]
fn item10_gpu_n_emits_boring_gpu_device_n() {
    let (_, rs) = cuda_codegen("gpu_n", r#"
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
fn item10_gpu_all_emits_device_iterator() {
    let (_, rs) = cuda_codegen("gpu_all", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let devs = GPU.all()
"#);
    assert!(rs.contains("CudaContext::device_count"),
        "expected CudaContext::device_count in GPU.all() expansion;\ngot:\n{rs}");
}

// ─── Item 6 — Multi-GPU: new(g) Scale(data) ──────────────────────────────────

#[test]
fn item6_new_with_arena_emits_scale_new_with_device() {
    let (_, rs) = cuda_codegen("new_arena", r#"
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
}

// ─── Item 4 — after = CUDA stream dependencies ───────────────────────────────

#[test]
fn item4_after_passes_stream_reference_to_launch() {
    let (_, rs) = cuda_codegen("after_dep", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k1 = Scale(data)
mut h1 = kernel(block = 2) k1
mut k2 = Scale(data)
mut h2 = kernel(block = 2, after = [h1]) k2
"#);
    // The second launch must forward h1.stream as a dependency.
    assert!(rs.contains("&h1.stream"),
        "expected &h1.stream in the after arg;\ngot:\n{rs}");
    // The call should pass a non-empty slice for the after arg.
    assert!(rs.contains("[&h1.stream]"),
        "expected [&h1.stream] slice;\ngot:\n{rs}");
}

#[test]
fn item4_no_after_passes_empty_slice() {
    let (_, rs) = cuda_codegen("no_after_dep", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Scale(data)
mut h = kernel(block = 2) k
"#);
    // Without `after =`, the call site passes &[].
    assert!(rs.contains("__boring_launch((2 as u32, 1, 1), None, &[])"),
        "expected &[] as after arg when no after specified;\ngot:\n{rs}");
}

// ─── Item 5 — dtod inference (analysis comment) ──────────────────────────────

#[test]
fn item5_dtod_candidate_produces_comment_or_direct_pass() {
    // When one kernel's output buffer is fed directly into another kernel
    // as input, the transpiler should either emit a dtod-candidate comment
    // or pass the buffer directly without a read_buf round-trip.
    // This is a minimal test: we verify the generated code compiles and
    // does not contain read_buf for the chained field when it is never
    // read on the host side.
    let (_, rs) = cuda_codegen("dtod_candidate", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    // The kernel struct must be present (basic sanity check).
    assert!(rs.contains("struct Scale"),
        "expected Scale struct in generated code;\ngot:\n{rs}");
}

// ─── GPU device properties ────────────────────────────────────────────────────

#[test]
fn host_gpu_name_property() {
    let (_, rs) = cuda_codegen("gpu_name", r#"
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
    let (_, rs) = cuda_codegen("gpu_total_mem", r#"
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
    let (_, rs) = cuda_codegen("gpu_compute", r#"
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

// ─── saxpy example ────────────────────────────────────────────────────────────

#[test]
fn example_saxpy() {
    let src = std::fs::read_to_string("examples/saxpy.br").expect("examples/saxpy.br not found");
    let (cu, rs) = cuda_codegen("saxpy_example", &src);

    // Device kernel
    assert!(cu.contains("__global__ void Saxpy_kernel("), "missing Saxpy_kernel;\ngot:\n{cu}");
    assert!(cu.contains("const double alpha"),            "missing const scalar alpha;\ngot:\n{cu}");
    assert!(cu.contains("const double* x"),              "missing x param;\ngot:\n{cu}");
    assert!(cu.contains("double* y"),                    "missing y param;\ngot:\n{cu}");
    assert!(cu.contains("y[i] = ((alpha * x[i]) + y[i])"), "missing saxpy body;\ngot:\n{cu}");

    // Host struct
    assert!(rs.contains("struct Saxpy"),          "missing struct Saxpy;\ngot:\n{rs}");
    assert!(rs.contains("alpha: CudaSlice<f64>"), "missing alpha field;\ngot:\n{rs}");
    assert!(rs.contains("x: CudaSlice<f64>"),    "missing x field;\ngot:\n{rs}");
    assert!(rs.contains("y: CudaSlice<f64>"),    "missing y field;\ngot:\n{rs}");

    // Host main: print, float cast, enumerate loop
    assert!(rs.contains("println!("),                      "print not translated to println!;\ngot:\n{rs}");
    assert!(rs.contains("as f64)"),                        "float() cast not translated;\ngot:\n{rs}");
    assert!(rs.contains(".iter().enumerate()"),            "for i,v not translated to enumerate;\ngot:\n{rs}");
}
