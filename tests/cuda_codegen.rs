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
    // 'actor with a fixed-size literal init — device emitter declares __shared__
    // inside the kernel body rather than as a pointer parameter.
    let (cu, _) = cuda_codegen("shared_static", r#"
kernel S:
    mut [float]'unified out
    let [float]'actor scratch
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
    let [float]'actor  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(cu.contains("extern __shared__"),
        "expected extern __shared__ for dynamic 'actor Array;\ngot:\n{cu}");
    assert!(!cu.contains("scratch* scratch"),
        "dynamic 'actor must not appear as a kernel parameter;\ngot:\n{cu}");
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

#[test]
fn device_gpu_warp_builtins_map_correctly() {
    let (cu, _) = cuda_codegen("gpu_warp_builtins", r#"
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
    assert!(cu.contains("warpSize"), "expected warpSize;\ngot:\n{cu}");
    assert!(cu.contains("% warpSize"), "expected lane linearization mod warpSize;\ngot:\n{cu}");
    assert!(cu.contains("__syncwarp(0xffffffff)"), "expected __syncwarp;\ngot:\n{cu}");
    assert!(cu.contains("__shfl_down_sync(0xffffffff,"), "expected __shfl_down_sync;\ngot:\n{cu}");
    assert!(cu.contains("__shfl_up_sync(0xffffffff,"), "expected __shfl_up_sync;\ngot:\n{cu}");
    assert!(cu.contains("__shfl_xor_sync(0xffffffff,"), "expected __shfl_xor_sync;\ngot:\n{cu}");
    assert!(cu.contains("__shfl_sync(0xffffffff,"), "expected __shfl_sync;\ngot:\n{cu}");
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
    // The htod upload doesn't happen inside `Scale::new` itself anymore --
    // `Scale::new`'s `data` param is already a `CudaSlice<f64>` (see
    // `host_unified_field_becomes_cuda_slice`); the upload happens at the
    // CONSTRUCTOR CALL SITE instead (`emit_kernel_ctor_args`), converting the
    // host `Vec` argument before it reaches `Scale::new`. A source snippet
    // with no such call site (as this test previously had) can never emit
    // `clone_htod` anywhere, regardless of whether the codegen is correct --
    // this was a stale test from before that refactor; a real call site is
    // needed to exercise the assertion at all.
    let (_, rs) = cuda_codegen("host_htod", r#"
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
    assert!(rs.contains("clone_htod"),
        "expected clone_htod at the Scale(data) constructor call site;\ngot:\n{rs}");
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
fn host_kernel_new_inlines_top_level_scalar() {
    // A top-level scalar `let` referenced inside a kernel's `init(...)` body
    // (here via `[0 for ..n]`) used to emit a bare `n` identifier in the
    // generated `impl VectorAdd { fn new(...) }` -- that constructor is its
    // own Rust fn/impl block, textually and scope-wise separate from
    // `fn main()`, which is the only place `n` actually becomes a Rust local
    // (see `top_level_scalars`'s doc in `cuda::host`). A real E0425
    // ("cannot find value `n` in this scope"), confirmed via `cargo check`.
    let (_, rs) = cuda_codegen("host_kernel_new_scalar", r#"
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
    let (_, rs) = cuda_codegen("host_shared_absent", r#"
kernel S:
    mut [float]'unified out
    let [float]'actor  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(!rs.contains("scratch: CudaSlice"),
        "'actor field must not appear as CudaSlice in Rust struct;\ngot:\n{rs}");
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
    assert!(rs.contains("fn __boring_launch(mut self, block_dim: (u32,u32,u32), grid_dim: Option<(u32,u32,u32)>, after: &[&Arc<CudaStream>], priority: i32)"),
        "expected __boring_launch signature with after and priority params;\ngot:\n{rs}");
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
// TODO: these tests use kernel: block syntax; the CUDA transpiler needs to handle
// Stmt::KernelBlock before they can pass (transpiler currently emits nothing for it).

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
kernel:
    k(block = 2)
"#);
    // Auto grid sizing: Scale's first field is a 'unified array, so grid is None when omitted.
    assert!(rs.contains("__boring_launch((2 as u32, 1, 1), None, &[], 0i32)?"),
        "expected __boring_launch((block_dim), None, &[], 0i32) with auto grid;\ngot:\n{rs}");
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
kernel:
    k(block = (16, 16))
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
kernel:
    k(block = (8, 8, 4))
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

// ─── 'actor'unified — atomics on host+device DRAM ──────────────────────────────

#[test]
fn device_actor_unified_uses_atomic_add() {
    let (cu, rs) = cuda_codegen("actor_unified_add", r#"
kernel A:
    mut [int]'actor'unified counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(cu.contains("atomicAdd(&counts[0]"),
        "expected atomicAdd for 'actor'unified compound assign;\ngot:\n{cu}");
    assert!(cu.contains("int64_t* counts"),
        "expected 'actor'unified field as pointer param;\ngot:\n{cu}");
    // Unlike 'actor'global, 'actor'unified is host-visible — it must get the same
    // D2H read accessor 'unified fields get.
    assert!(rs.contains("fn read_counts(&self)"),
        "expected a host-side read_counts() accessor for 'actor'unified;\ngot:\n{rs}");
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
    // `data` is a plain host Vec<f64> at this point -- the arena-qualified
    // constructor call must upload it via clone_htod exactly like the plain
    // `Scale(data)` call site does (see `emit_kernel_ctor_args`), not pass it
    // straight through as a bare Vec where Scale::new expects a CudaSlice<f64>.
    // This was a real E0308 ("expected CudaSlice<f64>, found Vec<{float}>"),
    // confirmed via cargo check against real cudarc 0.19.8, before the `New`
    // arm of `expr()` was fixed to route through `emit_kernel_ctor_args`
    // instead of emitting each arg raw.
    assert!(rs.contains("clone_htod"),
        "expected clone_htod at the new(g0) Scale(data) call site;\ngot:\n{rs}");
}

// ─── Item 4 — sequential kernel launches in kernel: block ───────────────────

#[test]
fn item4_sequential_launches_in_kernel_block() {
    let (_, rs) = cuda_codegen("sequential_launches", r#"
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
    // field is `__stream` (double-underscore; `KernelHandle<T>`'s OWN field
    // really is named `stream`, a different type entirely, which is presumably
    // how this went unnoticed). A real E0609 ("no field `stream` on type
    // `Scale`, did you mean `__stream`"), confirmed via `cargo check` against
    // real cudarc 0.19.8 -- this exact combination (an `after =` dependency on
    // another kernel struct variable) had no prior test coverage at all.
    let (_, rs) = cuda_codegen("after_dep_stream_field", r#"
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
fn multi_device_contexts_attempt_peer_access() {
    // Cross-device `after =` needs `cuStreamWaitEvent` (what `stream.join`
    // uses under the hood) to work across two different CUDA contexts, which
    // itself requires peer access enabled between them first -- confirmed via
    // NVIDIA's own docs. `boring_gpu_init`/`boring_gpu_ctx_n` now register
    // every context they create and attempt bidirectional peer access with
    // every other one already known, checking real hardware capability via
    // `cuDeviceCanAccessPeer` first (skipping silently if unsupported) rather
    // than requiring a `--peer-access` opt-in flag (see docs/cuda-module.md's
    // "Multi-device" section).
    let (_, rs) = cuda_codegen("multi_device_peer_access", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

let g0 = GPU(0)
let g1 = GPU(1)
let data = [1.0, 2.0]
mut ka = new(g0) Scale(data)
mut kb = new(g1) Scale(data)
kernel:
    ka(block = 2)
    kb(block = 2, after = ka)
"#);
    assert!(rs.contains("__boring_gpu_enable_peer_access(&ctx)?;"),
        "expected boring_gpu_init/boring_gpu_ctx_n to register+enable peer access for every context;\ngot:\n{rs}");
    assert!(rs.contains("cuDeviceCanAccessPeer"),
        "expected a real hardware-capability check before enabling peer access;\ngot:\n{rs}");
    assert!(rs.contains("cuCtxEnablePeerAccess"),
        "expected the actual peer-access-enabling call;\ngot:\n{rs}");
}

// ─── Item 5 — dtod inference (analysis comment) ──────────────────────────────

#[test]
fn item5_dtod_candidate_produces_comment_or_direct_pass() {
    // When one kernel's output buffer is fed directly into another kernel
    // as input, the transpiler passes the buffer via `.clone()` -- a real
    // device-to-device copy (`CudaSlice::clone()`'s own `clone_dtod`,
    // confirmed against real cudarc 0.19.8 source), NOT a host round-trip
    // and NOT a bare move. A bare move used to be this optimization's actual
    // behavior, but it applied unconditionally regardless of whether the
    // source kernel variable (`k1`) is used again afterward -- a real E0382
    // ("use of partially moved value") confirmed via `cargo check` the
    // moment `k1` is read or dispatched again later. `.clone()` is correct
    // in every case and still far cheaper than a full D2H+H2D round trip.
    let (_, rs) = cuda_codegen("dtod_candidate", r#"
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
    // A scalar `'const` field (`alpha` has no array type) is a plain kernel-launch
    // parameter, not a device buffer -- `CudaSlice<f64>` here was a real E0308,
    // confirmed via a `cargo check` against real cudarc 0.19.8 (the constructor
    // assigns it a bare `f64`, not a `CudaSlice`). See `host_field_type`'s fix.
    assert!(rs.contains("alpha: f64"),           "missing alpha field;\ngot:\n{rs}");
    assert!(rs.contains("x: CudaSlice<f64>"),    "missing x field;\ngot:\n{rs}");
    assert!(rs.contains("y: CudaSlice<f64>"),    "missing y field;\ngot:\n{rs}");

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
    let (_, rs) = cuda_codegen("kernel_handle_must_use", r#"
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
    // A kernel with a 1D auto-grid-capable field (`buf`) makes `grid_dim` an
    // `Option<(u32,u32,u32)>` parameter -- but `k(block=.., grid=..)` used to
    // ignore any explicit `grid` label entirely and always pass `None`,
    // silently overriding the caller's 2D/3D grid with 1D-length inference
    // (or `(1,1,1)` when the kernel has no auto-grid field at all). Confirmed
    // via a real generated project: the emitted call dropped `grid = (4, 4, 1)`
    // and passed `None` instead. `grid` must now be threaded through as
    // `Some((gx, gy, gz))`, matching the Metal backend's identical handling.
    let (_, rs) = cuda_codegen("explicit_grid", r#"
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
fn kernel_launch_and_sync_errors_are_classified_by_curesult() {
    // cudarc's own `DriverError` Display already calls
    // `cuGetErrorName`/`cuGetErrorString` (real message present already),
    // but gave no way to tell failure classes apart at a glance. A real
    // `catch`-by-variant isn't reachable here -- confirmed cuda::host has no
    // `Stmt::Try` handling at all (unlike the general/wgpu-shared pipeline),
    // the same prerequisite gap already blocking `with` there
    // (scoped-access-blocks.md) -- so this only classifies the message via
    // `__boring_cuda_classify_error`, applied at kernel launch and both
    // stream-sync points. Verified to compile clean via a real `cargo check`
    // against real cudarc 0.19.8 (with the `cuda-12080` feature, no real
    // CUDA toolkit needed for type-checking).
    let (_, rs) = cuda_codegen("gpu_error_classify", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("fn __boring_cuda_classify_error(e: cudarc::driver::DriverError)"),
        "expected the __boring_cuda_classify_error helper;\ngot:\n{rs}");
    assert!(rs.contains("CUresult::CUDA_ERROR_OUT_OF_MEMORY => \"GPU out of memory\""),
        "expected CUDA_ERROR_OUT_OF_MEMORY classified as GPU out of memory;\ngot:\n{rs}");
    assert!(rs.contains("unsafe { launcher.launch(cfg) }.map_err(__boring_cuda_classify_error)?;"),
        "expected the kernel launch call to classify its error;\ngot:\n{rs}");
    assert!(rs.contains("self.stream.synchronize().map_err(__boring_cuda_classify_error)?;"),
        "expected KernelHandle::wait's sync call to classify its error;\ngot:\n{rs}");
    // An oversized `block =` is the real, common case CUDA_ERROR_INVALID_VALUE
    // covers -- `cuLaunchKernel` rejects it at the driver level. Boring
    // deliberately does not duplicate this check in the validator or
    // interpreter; it defers entirely to this real runtime classification.
    assert!(rs.contains("CUresult::CUDA_ERROR_INVALID_VALUE => \"GPU launch configuration invalid (e.g. block size exceeds device limits)\""),
        "expected CUDA_ERROR_INVALID_VALUE classified as an invalid launch config, covering an oversized block size;\ngot:\n{rs}");
}

// ─── atomic min/max/swap/cas ───────────────────────────────────────────────────

#[test]
fn device_atomic_method_calls_map_to_cuda_intrinsics() {
    // `arr[i].min/max/swap/cas(...)` on an `'actor'global` field -- unlike
    // `+= -= &= |= ^=` (a statement-only compound-assign desugar via
    // try_atomic_assign), these are handled in expression position since
    // atomicMin/Max/Exch/CAS all return the previous value in real CUDA C.
    let (cu, _) = cuda_codegen("atomic_methods", r#"
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
    assert!(cu.contains("atomicMin(&counts[bucket], 5);"), "expected atomicMin;\ngot:\n{cu}");
    assert!(cu.contains("atomicMax(&counts[bucket], 5);"), "expected atomicMax;\ngot:\n{cu}");
    assert!(cu.contains("atomicExch(&counts[bucket], 0)"), "expected atomicExch;\ngot:\n{cu}");
    assert!(cu.contains("atomicCAS(&counts[bucket], 0, 1)"), "expected atomicCAS;\ngot:\n{cu}");
}

// ─── Labeled multi-dimensional arrays (docs/array-multidim-types.md) ───────

#[test]
fn device_labeled_index_lowers_to_row_major_index() {
    let (cu, _) = cuda_codegen("labeled_at", r#"
kernel Img:
    mut [float, width = 4, height = 4]'unified img
    init([float, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(cu.contains("img[c + r * 4]"),
        "expected [width=c,height=r] to lower to row-major c + r*width;\ngot:\n{cu}");
}

#[test]
fn device_labeled_axis_property_lowers_to_literals() {
    let (cu, _) = cuda_codegen("labeled_width_height", r#"
kernel Img:
    mut [float, width = 4, height = 8]'unified img
    init([float, width = 4, height = 8]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        if c < img.width and r < img.height:
            img[width = c, height = r] = 0.0
"#);
    assert!(cu.contains("c < 4"), "expected img.width to lower to the literal 4;\ngot:\n{cu}");
    assert!(cu.contains("r < 8"), "expected img.height to lower to the literal 8;\ngot:\n{cu}");
}

#[test]
fn device_labeled_array_field_becomes_pointer_param() {
    let (cu, _) = cuda_codegen("labeled_ptr_param", r#"
kernel Img:
    mut [float, width = 4, height = 4]'unified img
    init([float, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = 0.0
"#);
    assert!(cu.contains("__global__ void Img_kernel(double* img)"),
        "expected a LabeledArray field to become a flat pointer param, same as [T]'unified;\ngot:\n{cu}");
}

#[test]
fn device_labeled_array_3_axis_lowers_to_row_major_index() {
    let (cu, _) = cuda_codegen("labeled_3_axis", r#"
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
    assert!(cu.contains("vol[tx + ty * 4 + tz * 16]"),
        "expected [x,y,z] to lower to row-major x + y*4 + z*(4*4);\ngot:\n{cu}");
}

#[test]
fn device_shared_labeled_array_becomes_fixed_shared_decl() {
    let (cu, _) = cuda_codegen("shared_labeled", r#"
kernel Tile:
    mut [float]'unified out
    let [float, width = 4, height = 4]'actor tile
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        out[0] = tile[width = c, height = r]
"#);
    assert!(cu.contains("__shared__ double tile[16];"),
        "expected fixed __shared__ decl sized width*height, not extern __shared__;\ngot:\n{cu}");
}

#[test]
fn kernel_field_labeled_array_actor_global_is_valid() {
    let (cu, _) = cuda_codegen("labeled_actor_global", r#"
kernel Hist:
    mut [int, width = 4, height = 4]'actor'global hist
    init([int, width = 4, height = 4]'actor'global data):
        hist = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        hist[width = c, height = r] = hist[width = c, height = r] + 1
"#);
    assert!(cu.contains("hist"), "expected a LabeledArray'actor'global field to compile through codegen;\ngot:\n{cu}");
}

#[test]
fn kernel_field_labeled_array_bare_unified_is_valid() {
    let (cu, rs) = cuda_codegen("labeled_bare_unified", r#"
kernel Img:
    mut [float, width = 4, height = 4]'unified img
    init([float, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(cu.contains("__global__ void Img_kernel(double* img)"),
        "expected bare 'unified LabeledArray field to compile to a pointer param;\ngot:\n{cu}");
    assert!(rs.contains("img: CudaSlice<f64>"),
        "expected bare 'unified LabeledArray field to become a CudaSlice host field, same as [T]'unified;\ngot:\n{rs}");
}

#[test]
fn host_labeled_array_field_infers_2d_grid() {
    let (_, rs) = cuda_codegen("labeled_2d_grid", r#"
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
fn host_labeled_array_3_axis_infers_3d_grid() {
    let (_, rs) = cuda_codegen("labeled_3d_grid", r#"
kernel Vol:
    mut [float, x = 8, y = 16, z = 32]'unified vol
    init([float, x = 8, y = 16, z = 32]'unified data):
        vol = data
    def ():
        let tx = gpu.thread.x
        let ty = gpu.thread.y
        let tz = gpu.thread.z
        vol[x = tx, y = ty, z = tz] = vol[x = tx, y = ty, z = tz] * 2.0
"#);
    assert!(rs.contains("((8 + block_dim.0 - 1) / block_dim.0)"), "expected grid.x from x=8;\ngot:\n{rs}");
    assert!(rs.contains("((16 + block_dim.1 - 1) / block_dim.1)"), "expected grid.y from y=16;\ngot:\n{rs}");
    assert!(rs.contains("((32 + block_dim.2 - 1) / block_dim.2)"), "expected grid.z from z=32;\ngot:\n{rs}");
}

// ─── Dynamic-shape LabeledArray: grid inference from shadow fields ─────────

#[test]
fn host_dynamic_labeled_array_field_infers_2d_grid_from_shadow_fields() {
    let (_, rs) = cuda_codegen("dynamic_labeled_2d_grid", r#"
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

#[test]
fn host_dynamic_labeled_array_3_axis_infers_3d_grid_from_shadow_fields() {
    let (_, rs) = cuda_codegen("dynamic_labeled_3d_grid", r#"
kernel Vol:
    mut [float, x, y, z]'unified vol
    init([float]'unified data, uint xn, uint yn, uint zn):
        vol = data.reshape(x = xn, y = yn, z = zn)
    def ():
        let tx = gpu.thread.x
        let ty = gpu.thread.y
        let tz = gpu.thread.z
        vol[x = tx, y = ty, z = tz] = vol[x = tx, y = ty, z = tz] * 2.0
"#);
    assert!(rs.contains("self.__vol_axis0"), "expected grid.x from the __vol_axis0 shadow field;\ngot:\n{rs}");
    assert!(rs.contains("self.__vol_axis1"), "expected grid.y from the __vol_axis1 shadow field;\ngot:\n{rs}");
    assert!(rs.contains("self.__vol_axis2"), "expected grid.z from the __vol_axis2 shadow field;\ngot:\n{rs}");
    assert!(!rs.contains("self.vol.len()"), "should not fall back to 1D length-based grid inference;\ngot:\n{rs}");
}

// ─── .min/.max/.swap/.cas without 'actor — plain, non-atomic fallback ─────────

#[test]
fn atomic_method_calls_degrade_to_plain_read_modify_write_without_actor() {
    // `.min/.max/.swap/.cas` work on *any* indexed element, not just an
    // `'actor'global`/`'actor'unified` one -- matching `+=`/`-=`/etc.'s
    // existing degrade-to-plain-arithmetic behavior off a non-actor field,
    // rather than erroring or (worse) silently doing nothing. Bridged via a
    // GNU statement-expression (`({ ... })`) since there's no atomic
    // intrinsic to lean on for the "return the previous value" contract.
    let (cu, _) = cuda_codegen("plain_atomic_methods", r#"
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
    assert!(cu.contains("({ auto __old = buf[tid]; buf[tid] = min(buf[tid], (5)); __old; })"),
        "expected plain min via a GNU statement-expression;\ngot:\n{cu}");
    assert!(cu.contains("({ auto __old = buf[tid]; buf[tid] = max(buf[tid], (5)); __old; })"),
        "expected plain max via a GNU statement-expression;\ngot:\n{cu}");
    assert!(cu.contains("({ auto __old = buf[tid]; buf[tid] = (0); __old; })"),
        "expected plain swap via a GNU statement-expression;\ngot:\n{cu}");
    assert!(cu.contains("({ auto __old = buf[tid]; if (__old == (0)) buf[tid] = (1); __old; })"),
        "expected plain cas via a GNU statement-expression;\ngot:\n{cu}");
}
