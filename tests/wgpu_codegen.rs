// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// wgpu codegen snapshot tests.
//
// These tests verify the text emitted by `boring build --target wgpu` without
// requiring a real GPU.  Each test:
//   1. Writes a Boring source snippet to a temp file.
//   2. Invokes `boring build --target wgpu <file>`.
//   3. Reads the generated shaders/main.wgsl and src/main.rs.
//   4. Asserts that the generated text contains the expected patterns.
//
// Run with:
//   cargo test --test wgpu_codegen

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn run_wgpu(test_name: &str, src: &str) -> (String, String, String, String) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // Source file named "test.br" → boring creates "test_wgpu/" next to it.
    let br_file  = tmp.join("test.br");
    let wgpu_dir = tmp.join("test_wgpu");
    fs::write(&br_file, src).unwrap();

    let result = Command::new(bin)
        .args(["build", "--target", "wgpu"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{test_name}] failed to invoke boring: {e}"));

    assert!(
        result.status.success(),
        "[{test_name}] boring build --target wgpu failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let read = |rel: &str| fs::read_to_string(wgpu_dir.join(rel)).unwrap_or_default();
    (
        read("shaders/main.wgsl"),
        read("shaders/main_emulated.wgsl"),
        read("src/main.rs"),
        read("Cargo.toml"),
    )
}

fn wgpu_codegen(test_name: &str, src: &str) -> (String, String) {
    let (wgsl, _emulated, rs, _toml) = run_wgpu(test_name, src);
    (wgsl, rs)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn test_simple_vector_add() {
    let src = r#"
kernel VecAdd:
    mut [float]'unified a
    mut [float]'unified b
    mut [float]'unified c
    let int n

    def ():
        let i = gpu.block.x * gpu.block_dim.x + gpu.thread.x
        if i < n:
            c[i] = a[i] + b[i]
"#;
    let (wgsl, rs) = wgpu_codegen("vector_add", src);

    // WGSL device side.
    assert!(wgsl.contains("@group(0) @binding(0)"), "missing binding 0");
    assert!(wgsl.contains("@group(0) @binding(1)"), "missing binding 1");
    assert!(wgsl.contains("@group(0) @binding(2)"), "missing binding 2");
    assert!(wgsl.contains("var<storage"), "missing storage qualifier");
    assert!(wgsl.contains("@compute @workgroup_size("), "missing workgroup_size");
    assert!(wgsl.contains("@builtin(local_invocation_id)"), "missing local_invocation_id builtin");
    assert!(wgsl.contains("@builtin(workgroup_id)"), "missing workgroup_id builtin");
    assert!(wgsl.contains("VecAdd_main"), "missing entry fn name");

    // Host side.
    assert!(rs.contains("wgpu::BufferUsages::STORAGE"), "missing STORAGE usage");
    assert!(rs.contains("wgpu::BufferUsages::COPY_SRC"), "missing COPY_SRC on storage buffer");
    assert!(rs.contains("dispatch_workgroups"), "missing dispatch");
    assert!(rs.contains("queue.submit"), "missing queue submit");
    assert!(rs.contains("device.poll"), "missing device poll");
    assert!(rs.contains("bytemuck"), "missing bytemuck import");
}

#[test]
fn test_kernel_dispatch_surfaces_validation_errors_instead_of_silent_failure() {
    // Before this fix, `dispatch()` returned `()` and no error scope existed
    // anywhere in the generated code -- a validation failure (e.g. a rejected
    // workgroup count) was never observed by anything Boring generated. Confirmed
    // via a real `cargo check` against the real `wgpu`/`pollster` crates that this
    // whole chain (dispatch -> kernel: block call site -> boring_main()'s own
    // Result) compiles end-to-end.
    let src = r#"
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
"#;
    let (_wgsl, rs) = wgpu_codegen("dispatch_error_scope", src);

    assert!(rs.contains("fn dispatch(&self, gx: u32, gy: u32, gz: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>>"),
        "expected dispatch() to return a real Result;\ngot:\n{rs}");
    assert!(rs.contains("push_error_scope(wgpu::ErrorFilter::Validation)"),
        "expected dispatch() to open a validation error scope before encoding;\ngot:\n{rs}");
    assert!(rs.contains("pollster::block_on(self.device.pop_error_scope())"),
        "expected dispatch() to check the error scope after submit;\ngot:\n{rs}");
    assert!(rs.contains("k.dispatch((") && rs.contains(")?;"),
        "expected the kernel: block's dispatch call site to propagate the error via ?;\ngot:\n{rs}");
    assert!(rs.contains("fn boring_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>>"),
        "expected the synthesized boring_main() to be Result-returning so dispatch()'s ? has somewhere to go;\ngot:\n{rs}");
}

#[test]
fn test_scalar_uniform() {
    let src = r#"
kernel Scale:
    mut [float32]'unified data
    let float32 alpha

    def ():
        let i = gpu.block.x * gpu.block_dim.x + gpu.thread.x
        data[i] = data[i] * alpha
"#;
    let (wgsl, rs) = wgpu_codegen("scalar_uniform", src);

    assert!(wgsl.contains("struct ScaleParams"), "missing params struct in WGSL");
    assert!(wgsl.contains("alpha: f32"), "missing alpha field in params");
    assert!(wgsl.contains("var<uniform> scale_params"), "missing uniform binding");

    assert!(rs.contains("struct ScaleParams"), "missing params struct in Rust");
    assert!(rs.contains("queue.write_buffer"), "missing params upload");
}

#[test]
fn test_sync_barrier_fixed_array() {
    let src = r#"
kernel Tile:
    let [float32, 256]'actor tile
    mut [float32]'unified data

    def ():
        let i = gpu.block.x * gpu.block_dim.x + gpu.thread.x
        tile[gpu.thread.x] = data[i]
        sync
        data[i] = tile[gpu.thread.x]
"#;
    let (wgsl, _rs) = wgpu_codegen("sync_barrier", src);

    assert!(wgsl.contains("var<workgroup>"), "missing workgroup var");
    assert!(wgsl.contains("array<f32, 256>"), "missing fixed-size array type");
    assert!(wgsl.contains("workgroupBarrier()"), "missing explicit barrier");

    // `var<workgroup>` is only legal at WGSL module scope — naga rejects it as a
    // statement inside a function body ("expected identifier, found '<'"). Make
    // sure the declaration appears before the entry point's `@compute` annotation
    // (i.e. outside the function), not after it (i.e. inside the function body).
    let workgroup_pos = wgsl.find("var<workgroup>").expect("workgroup var present");
    let entry_pos = wgsl.find("@compute @workgroup_size(").expect("entry point present");
    assert!(
        workgroup_pos < entry_pos,
        "var<workgroup> must be declared at module scope, before the @compute entry point — \
         found it after, which means it was emitted inside the function body"
    );
}

#[test]
fn test_actor_global_atomic() {
    let src = r#"
kernel Histogram:
    mut [int]'actor'global counts
    mut [int]'unified data

    def ():
        let i = gpu.block.x * gpu.block_dim.x + gpu.thread.x
        counts[data[i]] += 1
"#;
    let (wgsl, rs) = wgpu_codegen("actor_global", src);

    assert!(wgsl.contains("atomic<i32>"), "missing atomic type in WGSL");
    assert!(wgsl.contains("atomicAdd"), "missing atomicAdd");
    assert!(rs.contains("COPY_SRC"), "actor global should have COPY_SRC");
}

#[test]
fn test_actor_unified_atomic() {
    let src = r#"
kernel Histogram:
    mut [int]'actor'unified counts
    mut [int]'unified       data

    def ():
        let i = gpu.block.x * gpu.block_dim.x + gpu.thread.x
        counts[data[i]] += 1
"#;
    let (wgsl, rs) = wgpu_codegen("actor_unified", src);

    assert!(wgsl.contains("atomic<i32>"), "missing atomic type in WGSL");
    assert!(wgsl.contains("atomicAdd"), "missing atomicAdd");
    // Same storage-only usage as 'actor'global/'unified — MAP_READ/MAP_WRITE is
    // never combined with the atomic<T> storage buffer itself; host access goes
    // through the staging-buffer copy path instead (see
    // `copy_counts_to_host`/`copy_counts_to_device`), which is what sidesteps the
    // open question of whether WGSL even allows that combination.
    let usage_line = "wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,";
    assert!(rs.contains(usage_line),
        "expected counts_buf's own creation to request only STORAGE|COPY_SRC|COPY_DST \
         (no MAP_READ/MAP_WRITE on the atomic<T> buffer itself);\ngot:\n{rs}");
    // Unlike 'actor'global, 'actor'unified is host-visible — it must get the
    // same read-back/upload accessors 'unified fields get.
    assert!(rs.contains("fn copy_counts_to_host"),
        "expected a host-side copy_counts_to_host() accessor for 'actor'unified;\ngot:\n{rs}");
    assert!(rs.contains("fn copy_counts_to_device"),
        "expected a host-side copy_counts_to_device() accessor for 'actor'unified;\ngot:\n{rs}");
}

#[test]
fn test_gpu_builtins_mapped() {
    let src = r#"
kernel Builtins:
    mut [int]'unified out

    def ():
        let tx = gpu.thread.x
        let bx = gpu.block.x
        let bdx = gpu.block_dim.x
        let gdx = gpu.grid_dim.x
        out[0] = tx + bx + bdx + gdx
"#;
    let (wgsl, _rs) = wgpu_codegen("builtins", src);

    assert!(wgsl.contains("local_invocation_id"), "gpu.thread.x → local_invocation_id");
    assert!(wgsl.contains("workgroup_id"), "gpu.block.x → workgroup_id");
    assert!(wgsl.contains("let bp_bdim = vec3<u32>("), "gpu.block_dim.x → derived from block sizes");
    assert!(wgsl.contains("num_workgroups"), "gpu.grid_dim.x → num_workgroups");
}

#[test]
fn test_gpu_warp_builtins_real_subgroup_path() {
    let src = r#"
kernel WarpBuiltins:
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
        buf[tid] = a + b + c + d + f32(lane) + f32(size)
"#;
    let (wgsl, emulated, _rs, _toml) = run_wgpu("warp_builtins_real", src);

    assert!(wgsl.contains("enable subgroups;"), "expected enable subgroups;\ngot:\n{wgsl}");
    assert!(wgsl.contains("@builtin(subgroup_size)"), "expected @builtin(subgroup_size);\ngot:\n{wgsl}");
    assert!(wgsl.contains("@builtin(subgroup_invocation_id)"), "expected @builtin(subgroup_invocation_id);\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupBarrier()"), "expected subgroupBarrier();\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffleDown("), "expected subgroupShuffleDown;\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffleUp("), "expected subgroupShuffleUp;\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffleXor("), "expected subgroupShuffleXor;\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffle("), "expected subgroupShuffle;\ngot:\n{wgsl}");

    // The emulated fallback module must exist alongside the real one whenever
    // `gpu.warp.*` is used, and never uses the subgroup extension.
    assert!(!emulated.is_empty(), "expected shaders/main_emulated.wgsl to be written");
    assert!(!emulated.contains("enable subgroups;"), "emulated module must not enable subgroups;\ngot:\n{emulated}");
}

#[test]
fn test_gpu_warp_shuffle_emulated_fallback_shape() {
    let src = r#"
kernel WarpEmulated:
    mut [float32]'unified buf

    def ():
        let tid = gpu.thread.x
        gpu.warp.sync()
        let shuffled = gpu.warp.shuffle_down(buf[tid], 1)
        buf[tid] = shuffled
"#;
    let (_wgsl, emulated, _rs, _toml) = run_wgpu("warp_shuffle_emulated", src);

    assert!(emulated.contains("var<workgroup> bp_warp_scratch_f32"),
        "expected a f32 workgroup scratch buffer;\ngot:\n{emulated}");
    assert!(emulated.contains("workgroupBarrier()"), "expected workgroupBarrier();\ngot:\n{emulated}");
    assert!(emulated.contains("@builtin(local_invocation_index)"),
        "expected @builtin(local_invocation_index);\ngot:\n{emulated}");
    assert!(emulated.contains("let bp_wsize: u32 = 32u;"), "expected fixed 32-lane fallback constant;\ngot:\n{emulated}");
    assert!(emulated.contains("select("), "expected a select() for the warp-boundary clamp;\ngot:\n{emulated}");
}

#[test]
fn test_gpu_warp_not_used_leaves_output_unchanged() {
    let src = r#"
kernel Plain:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#;
    let (wgsl, emulated, _rs, _toml) = run_wgpu("warp_not_used", src);
    assert!(!wgsl.contains("enable subgroups;"), "no gpu.warp.* usage should never enable subgroups;\ngot:\n{wgsl}");
    assert!(emulated.is_empty(), "no gpu.warp.* usage should not emit an emulated shader file");
}

#[test]
fn test_cargo_toml_deps() {
    let src = r#"
kernel Empty:
    mut [float]'unified data
    def ():
        data[0] = 1.0
"#;
    let (_wgsl, _emulated, _rs, toml) = run_wgpu("cargo_toml", src);

    assert!(toml.contains("wgpu = \"22\""), "missing wgpu dep");
    assert!(toml.contains("bytemuck"), "missing bytemuck dep");
    assert!(toml.contains("pollster"), "missing pollster dep");
    assert!(!toml.contains("winit"), "winit should not be present for compute-only");
}

#[test]
fn test_type_narrowing_int_to_i32() {
    let src = r#"
kernel Narrow:
    mut [int]'unified buf

    def ():
        let x: int = 42
        buf[0] = x
"#;
    let (wgsl, _rs) = wgpu_codegen("narrowing", src);

    assert!(wgsl.contains("i32"), "int fields should narrow to i32 in WGSL");
    assert!(!wgsl.contains("i64"), "i64 must not appear in WGSL");
}

#[test]
fn test_global_buffer_d2h_helper() {
    let src = r#"
kernel Compute:
    mut [float]'global result

    def ():
        result[0] = 1.0
"#;
    let (_wgsl, rs) = wgpu_codegen("global_buffer", src);

    assert!(rs.contains("__boring_gpu_copy_d2h"), "missing D2H staging helper");
    assert!(rs.contains("__boring_gpu_copy_h2d"), "missing H2D staging helper");
    assert!(rs.contains("MAP_READ | wgpu::BufferUsages::COPY_DST"), "staging D2H usages");
}

#[test]
fn test_d2h_staging_buffer_pool_reuse() {
    // Same source as test_global_buffer_d2h_helper -- this test asserts on the
    // *shape* of __boring_gpu_copy_d2h's generated body: repeated readbacks of
    // the same size must reuse a pooled staging buffer instead of allocating a
    // fresh one every call. There's no real GPU in this test harness (these are
    // codegen-shape snapshot tests, see the module doc comment), so we verify
    // the pooling logic is structurally present rather than exercising it at
    // runtime against real device readbacks.
    let src = r#"
kernel Compute:
    mut [float]'global result

    def ():
        result[0] = 1.0
"#;
    let (_wgsl, rs) = wgpu_codegen("d2h_staging_pool", src);

    assert!(rs.contains("thread_local!"), "missing thread_local staging pool declaration:\n{rs}");
    assert!(rs.contains("__BORING_STAGING_POOL"), "missing staging pool storage:\n{rs}");

    // The pool lookup (by exact size match) must happen before falling back to
    // `device.create_buffer` -- i.e. create_buffer is reached only on a pool miss.
    let copy_d2h_start = rs.find("fn __boring_gpu_copy_d2h").expect("missing __boring_gpu_copy_d2h fn");
    let copy_d2h_body = &rs[copy_d2h_start..];
    let pool_lookup_pos = copy_d2h_body.find("pool.iter().position").expect("missing pool lookup by size");
    let create_buffer_pos = copy_d2h_body.find("device.create_buffer").expect("missing create_buffer fallback");
    assert!(pool_lookup_pos < create_buffer_pos,
        "pool lookup must be attempted before falling back to device.create_buffer:\n{copy_d2h_body}");

    // The staging buffer must be unmapped, then returned to the pool -- not dropped.
    let unmap_pos = copy_d2h_body.find("staging.unmap()").expect("missing staging.unmap()");
    let pool_push_pos = copy_d2h_body.find("pool.borrow_mut().push(staging)").expect("missing pool push-back of staging buffer");
    assert!(unmap_pos < pool_push_pos,
        "staging buffer must be fully unmapped before being returned to the pool:\n{copy_d2h_body}");
}

#[test]
fn test_screen_present_and_key() {
    // Minimal game-of-life-style program with Screen + kernel + render loop.
    let src = r#"
kernel Step:
    mut [int]'actor cells_in
    mut [int]'actor cells_out
    let int w

    def ():
        cells_out[0] = cells_in[0]

kernel Render:
    mut [int]'actor pixels
    let int w

    def ():
        pixels[0] = 0

let w = 800
let h = 600
let screen = Screen(Dimension(w, h), title = "Test")
var step = Step(Dimension(w, h))
var render = Render(Dimension(w, h))

kernel:
    loop:
        step(block = (16, 16))
        render(block = (16, 16))
        screen.present(render.pixels)
        if screen.key("\x1B"):
            break
"#;
    let (_wgsl, rs) = wgpu_codegen("screen_present", src);

    assert!(rs.contains("use winit::application::ApplicationHandler"), "missing ApplicationHandler import");
    assert!(rs.contains("use winit::event::{WindowEvent, ElementState}"), "missing winit event imports");
    assert!(rs.contains("use winit::keyboard::{Key, NamedKey}"), "missing NamedKey import");
    assert!(rs.contains("use winit::window::{Window, WindowAttributes, WindowId}"), "missing WindowAttributes import");
    assert!(rs.contains("fn resumed(&mut self, event_loop: &ActiveEventLoop)"), "missing resumed method");
    assert!(rs.contains("fn window_event(&mut self, event_loop: &ActiveEventLoop"), "missing window_event method");
    assert!(rs.contains("event_loop.create_window(WindowAttributes::default()"), "window created via create_window");
    assert!(rs.contains("fn __boring_present_buffer("), "missing present buffer helper");
    assert!(rs.contains("__boring_present_buffer(&self.device, &self.queue, self.surface.as_ref().unwrap()"), "present call");
    assert!(rs.contains("if self.__keys.contains(\"Escape\") { event_loop.exit(); }"), "Escape key exits");
    assert!(rs.contains("surface.get_capabilities(&self.adapter)"), "adapter used for surface caps");
    assert!(rs.contains("surface.configure(&self.device,"), "surface configured");
    assert!(rs.contains("EventLoop::new()"), "event loop created");
    assert!(rs.contains("event_loop.run_app(&mut app)"), "run_app used");
    assert!(rs.contains("NamedKey::Escape"), "Escape named key in key handler");
}

// ── `with` GPU-residency materialization (docs/scoped-access-blocks.md) ────────
//
// `let py'gpu'unified = k.y` followed by `with py:` should read the kernel field
// back exactly once (`k.copy_y_to_host()`), regardless of how many times the
// block's body indexes `py` — the actual bug this exists to fix, confirmed against
// `examples/vector_add_gpu.br`'s `for i in 0..n: print k.result[i]`, which today
// re-reads the whole buffer on every loop iteration with no `with` available.

#[test]
fn test_with_gpu_resident_read_only_single_readback() {
    let src = r#"
kernel Saxpy:
    let float alpha
    let [float]'unified x
    mut [float]'unified y

    init(float a, [float]'unified xs, [float]'unified ys):
        alpha = a
        x = xs
        y = ys

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        y[i] = alpha * x[i] + y[i]

var [float] hx = [0.0, 1.0]
var [float] hy = [1.0, 1.0]
mut k = Saxpy(2.0, hx, hy)
kernel:
    k(block = 2)

let [float]'gpu'unified py = k.y
with py:
    for i in 0..2:
        print "{py[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_read", src);

    // Exactly one readback CALL (`k.copy_y_to_host()`), bound before the loop — not
    // one per iteration. `copy_y_to_host` alone also matches the method's own `fn`
    // definition, so count the call form specifically.
    assert_eq!(rs.matches("k.copy_y_to_host()").count(), 1, "expected exactly one copy_y_to_host call:\n{rs}");
    assert!(rs.contains("let py = k.copy_y_to_host()"), "missing single materializing readback:\n{rs}");
    // Read-only block: no write-back targeting `py` (the constructor's own initial
    // upload, `k.copy_y_to_device(&hy...)`, is unrelated and expected), and the
    // alias binding isn't `mut`.
    assert!(!rs.contains("k.copy_y_to_device(&py"), "read-only with-block should not write back:\n{rs}");
    assert!(!rs.contains("let mut py"), "read-only alias should not be `mut`:\n{rs}");
    // No leftover placeholder pointer type for the plain host arrays.
    assert!(!rs.contains("*mut Vec"), "gpu'unified/gpu'global should emit a plain Vec, not a pointer:\n{rs}");
}

#[test]
fn test_with_gpu_resident_write_back_on_mutation() {
    let src = r#"
kernel Saxpy:
    let float alpha
    let [float]'unified x
    mut [float]'unified y

    init(float a, [float]'unified xs, [float]'unified ys):
        alpha = a
        x = xs
        y = ys

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        y[i] = alpha * x[i] + y[i]

var [float] hx = [0.0, 1.0]
var [float] hy = [1.0, 1.0]
mut k = Saxpy(2.0, hx, hy)
kernel:
    k(block = 2)

let [float]'gpu'unified py = k.y
with py:
    py[0] = 0.0
"#;
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_write", src);

    assert!(rs.contains("let mut py = k.copy_y_to_host()"), "mutating block needs a `mut` alias:\n{rs}");
    assert!(rs.contains("k.copy_y_to_device(&py"), "write-back should target the kernel field:\n{rs}");
    // The constructor's own initial upload (`k.copy_y_to_device(&hy...)`) is a
    // separate, expected call — only the with-block's write-back targets `py`.
    assert_eq!(rs.matches("k.copy_y_to_device(&py").count(), 1, "expected exactly one write-back call targeting py:\n{rs}");
}

#[test]
fn test_with_gpu_resident_infers_qualifier_without_annotation() {
    // Same as test_with_gpu_resident_read_only_single_readback, but `py` has no
    // explicit 'gpu'unified annotation at all — the qualifier is inferred from `k.y`
    // being a 'unified array field on a tracked kernel instance.
    let src = r#"
kernel Saxpy:
    let float alpha
    let [float]'unified x
    mut [float]'unified y

    init(float a, [float]'unified xs, [float]'unified ys):
        alpha = a
        x = xs
        y = ys

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        y[i] = alpha * x[i] + y[i]

var [float] hx = [0.0, 1.0]
var [float] hy = [1.0, 1.0]
mut k = Saxpy(2.0, hx, hy)
kernel:
    k(block = 2)

let py = k.y
with py:
    for i in 0..2:
        print "{py[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_inferred", src);

    assert_eq!(rs.matches("k.copy_y_to_host()").count(), 1, "expected exactly one copy_y_to_host call:\n{rs}");
    assert!(rs.contains("let py = k.copy_y_to_host()"), "missing single materializing readback:\n{rs}");
    assert!(!rs.contains("*mut Vec"), "inferred qualifier should emit a plain Vec, not a pointer:\n{rs}");
}

// ── Interprocedural residency: `with` surviving a function-call boundary ──────
//
// The intra-procedural tests above cover a kernel instance and its field read living
// in the *same* scope. These tests cover the actual motivating case
// (docs/scoped-access-blocks.md): a free function returning a `'gpu'unified`-typed
// value, chained into a second call, with only the *final* consumer paying a host
// round-trip — the shape of whisper-boring's `linear_gpu` -> `gelu_gpu` -> `linear_gpu`.

const SCALE_GPU_KERNEL: &str = r#"
kernel Saxpy:
    let float alpha
    let [float]'unified x
    mut [float]'unified y

    init(float a, [float]'unified xs, [float]'unified ys):
        alpha = a
        x = xs
        y = ys

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        y[i] = alpha * x[i] + y[i]
"#;

#[test]
fn test_with_gpu_resident_call_chain_no_intermediate_roundtrip() {
    // `scale_gpu` wraps kernel construction+dispatch+field-read behind a function
    // boundary with an explicit `'gpu'unified` return type — exactly the shape a
    // real kernel-launcher wrapper (`linear_gpu`, etc.) uses. Called twice in a
    // chain: the second call's argument is the first call's still-resident return
    // value, and only the final `with` pays a real device->host transfer.
    let src = format!(r#"{SCALE_GPU_KERNEL}
req [float]'gpu'unified scale_gpu([float] xv, float factor):
    var [float] zero = [0.0, 0.0]
    mut k = Saxpy(factor, xv, zero)
    kernel:
        k(block = 2)
    k.y

var [float] ha = [1.0, 2.0]
let [float]'gpu'unified fc = scale_gpu(ha, 2.0)
let [float]'gpu'unified fc2 = scale_gpu(fc, 3.0)
with fc2:
    print "{{fc2[0]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_call_chain", &src);

    // Signature: dual-typed param (the only use of `xv` is as a kernel-constructor
    // argument at a 'unified field position), resident return type.
    assert!(rs.contains("fn scale_gpu(xv: BoringGpuArg<f64>, factor: f64) -> BoringGpuArg<f64>"),
        "expected dual-typed param + resident return signature:\n{rs}");

    // Tail expression returns the buffer directly -- no download.
    assert!(rs.contains("BoringGpuArg::Resident(std::sync::Arc::clone(&k.y_buf)"),
        "tail expression should return a Resident handle, not a download:\n{rs}");

    // Kernel-construction consumes `xv` via the dual-mode branch, not an
    // unconditional upload.
    assert!(rs.contains("match &xv {"), "constructor argument for `xv` should branch on BoringGpuArg:\n{rs}");
    assert!(rs.contains("k.x_buf = __boring_gpu_copy_d2d(&__boring_gpu_device(), &__boring_gpu_queue(), buf);"), "resident branch should copy the buffer device-to-device, not alias it:\n{rs}");
    assert!(rs.contains("k.rebuild_bind_group();"), "resident branch should rebuild the bind group:\n{rs}");

    // Call sites: `ha` (a plain host array) is wrapped; `fc` (already resident) is
    // passed straight through, not re-wrapped as a host upload.
    assert!(rs.contains("scale_gpu(BoringGpuArg::Host(ha.clone())"), "plain host argument should wrap as BoringGpuArg::Host:\n{rs}");
    assert!(rs.contains("scale_gpu(fc.clone()"), "already-resident argument should pass straight through:\n{rs}");
    assert!(!rs.contains("BoringGpuArg::Host(fc"), "resident value should not be re-wrapped as a host upload:\n{rs}");

    // The whole point: no kernel-field download (`copy_y_to_host`) happens anywhere
    // in the chain -- only the final `with fc2:` materializes, via the free d2h
    // helper directly on the retained buffer (no live kernel instance to call
    // `copy_y_to_host` on at that point).
    assert_eq!(rs.matches("copy_y_to_host()").count(), 0, "no kernel-field download should occur anywhere in the chain:\n{rs}");
    assert!(rs.contains("__boring_gpu_copy_d2h::<f32>(&__boring_gpu_device(), &__boring_gpu_queue(), buf)"),
        "final `with` should materialize via the free d2h helper on the raw buffer:\n{rs}");
    // Read-only `with` block -- no write-back for `fc2` specifically. (The bare
    // `__boring_gpu_copy_h2d` helper still appears elsewhere in the file -- it's
    // also what `copy_x_to_device`/`copy_y_to_device` call internally for the
    // kernel's own H2D uploads, unrelated to this `with` block.)
    assert!(!rs.contains("__fc2_buf"), "read-only with-block should not capture a buffer handle for write-back:\n{rs}");
}

#[test]
fn test_with_gpu_resident_call_infers_qualifier_without_annotation() {
    // Same shape as the chain test above, but neither `fc` nor `fc2` has an explicit
    // `'gpu'unified` annotation -- inferred from `scale_gpu`'s own declared return
    // type, mirroring the same-scope `let py = k.y` inference precedent.
    let src = format!(r#"{SCALE_GPU_KERNEL}
req [float]'gpu'unified scale_gpu([float] xv, float factor):
    var [float] zero = [0.0, 0.0]
    mut k = Saxpy(factor, xv, zero)
    kernel:
        k(block = 2)
    k.y

var [float] ha = [1.0, 2.0]
let fc = scale_gpu(ha, 2.0)
let fc2 = scale_gpu(fc, 3.0)
with fc2:
    print "{{fc2[0]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_call_chain_inferred", &src);

    assert!(rs.contains("fn scale_gpu(xv: BoringGpuArg<f64>, factor: f64) -> BoringGpuArg<f64>"),
        "expected dual-typed param + resident return signature:\n{rs}");
    assert!(rs.contains("scale_gpu(fc.clone()"), "already-resident argument should pass straight through even without an explicit annotation:\n{rs}");
    assert_eq!(rs.matches("copy_y_to_host()").count(), 0, "no kernel-field download should occur anywhere in the chain:\n{rs}");
}

// ── Regression tests: consuming a resident value, not just returning one ──────
//
// The tests above all cover the *return* side of interprocedural residency —
// `scale_gpu`'s own kernel dispatches with a literal block size, never indexing or
// sizing off its dual-typed param. Real kernel-launcher wrappers (whisper-boring's
// `linear_gpu`/`gelu_gpu`/etc.) size their dispatch block off the very array they
// pass to the kernel constructor (`k(block = x.length)`), and real pipelines chain
// kernels directly in one scope as often as across a function boundary. These three
// mirror that consuming shape exactly.

const SCALE_ONE_ARG_KERNEL: &str = r#"
kernel Scale:
    let float factor
    let [float]'unified x
    mut [float]'unified y

    init(float f, [float]'unified xs):
        factor = f
        x = xs
        y = [0.0 for ..xs.length]

    def ():
        let i = gpu.thread.x
        y[i] = x[i] * factor
"#;

#[test]
fn test_with_gpu_resident_call_param_used_for_dispatch_size() {
    // `x` is used both as the kernel-constructor argument AND to size the dispatch
    // block (`x.length`) -- a second, non-constructor use that must NOT disqualify
    // the exclusive-ctor-arg scan (`ast::scan_var_call_arg_uses`), since a dual-typed
    // `BoringGpuArg<T>` can answer `.length` without ever materializing.
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'gpu'unified scale_gpu([float] x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let fc = scale_gpu(a, 2.0)
    let fc2 = scale_gpu(fc, 3.0)
    with fc2:
        for i in 0..3:
            print "{{fc2[i]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_param_dispatch_size", &src);

    assert!(rs.contains("fn scale_gpu(x: BoringGpuArg<f64>, factor: f64) -> BoringGpuArg<f64>"),
        "x.length use should not disqualify x from the dual-typed param treatment:\n{rs}");
    assert!(rs.contains("match &x {"), "constructor argument for `x` should branch on BoringGpuArg:\n{rs}");
    assert!(rs.contains("k.x_buf = __boring_gpu_copy_d2d(&__boring_gpu_device(), &__boring_gpu_queue(), buf);"), "resident branch should copy the buffer device-to-device, not alias it:\n{rs}");
    assert!(rs.contains("(x.len()) as usize"), "x.length should compile via BoringGpuArg::len(), not a bare field access:\n{rs}");
    assert!(!rs.contains("x::length") && !rs.contains("x::count"), "x.length must not be emitted as a module path:\n{rs}");

    // Chain: plain host array wraps, already-resident value passes straight through.
    assert!(rs.contains("scale_gpu(BoringGpuArg::Host(a.clone())"), "plain host argument should wrap as BoringGpuArg::Host:\n{rs}");
    assert!(rs.contains("scale_gpu(fc.clone()"), "already-resident argument should pass straight through:\n{rs}");
    assert!(!rs.contains("scale_gpu(&fc"), "the by-ref array-argument convention must not apply to a dual-typed param:\n{rs}");
}

#[test]
fn test_with_gpu_resident_call_param_explicit_annotation_used_for_dispatch_size() {
    // Same shape as above, but `x` carries an explicit `'gpu'unified` annotation --
    // the annotation must not survive into the emitted parameter type (it should
    // still collapse to the same dual-typed `BoringGpuArg<T>` signature, matching the
    // return-type case), and the same `x.length` use must not disqualify it either.
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'gpu'unified scale_gpu([float]'gpu'unified x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let fc = scale_gpu(a, 2.0)
    let fc2 = scale_gpu(fc, 3.0)
    with fc2:
        for i in 0..3:
            print "{{fc2[i]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_param_annotated_dispatch_size", &src);

    assert!(rs.contains("fn scale_gpu(x: BoringGpuArg<f64>, factor: f64) -> BoringGpuArg<f64>"),
        "an explicit 'gpu'unified annotation on the param should collapse to BoringGpuArg<T>, not a plain Vec:\n{rs}");
    assert!(rs.contains("match &x {"), "constructor argument for `x` should branch on BoringGpuArg:\n{rs}");
    assert!(rs.contains("scale_gpu(fc.clone()"), "already-resident argument should pass straight through:\n{rs}");
}

#[test]
fn test_kernel_constructor_consumes_resident_local_no_function_boundary() {
    // No function boundary at all: `k1.y` is aliased to `fc` (`gpu_resident_vars` --
    // a pure compile-time alias with no Rust binding) and then used directly as
    // `k2`'s constructor argument. This isolates the constructor-argument-consumption
    // gap from the fn-parameter dual-typing above -- `fc` never has a Rust identifier
    // to type `BoringGpuArg<T>` in the first place, so the fix must reach into
    // `gpu_resident_vars` directly rather than going through that enum at all.
    let src = r#"
kernel Scale:
    let float factor
    let [float]'unified x
    mut [float]'unified y

    init(float f, [float]'unified xs):
        factor = f
        x = xs
        y = [0.0 for ..xs.length]

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    mut k1 = Scale(2.0, a)
    kernel:
        k1(block = 3)
    let fc = k1.y

    mut k2 = Scale(3.0, fc)
    kernel:
        k2(block = 3)
    let fc2 = k2.y

    with fc2:
        for i in 0..3:
            print "{fc2[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("kernel_ctor_consumes_resident_local", src);

    // The second kernel gets its own device-to-device copy of the first
    // kernel's buffer -- no host round-trip, no dangling reference to a `fc`
    // Rust binding that never exists, and (unlike a bare `Arc::clone`, which
    // would silently alias the same `wgpu::Buffer` between k1 and k2) still
    // correct if `k1` were dispatched again afterward.
    assert!(rs.contains("k2.x_buf = __boring_gpu_copy_d2d(&__boring_gpu_device(), &__boring_gpu_queue(), &k1.y_buf);"),
        "second kernel's x field should get a real device-to-device copy of the first kernel's y buffer:\n{rs}");
    assert!(rs.contains("k2.rebuild_bind_group();"), "buffer aliasing should rebuild the bind group:\n{rs}");
    assert!(!rs.contains("k2.copy_x_to_device"), "no host upload should happen for a resident-aliased argument:\n{rs}");
    assert!(!rs.contains("&fc") && !rs.contains("(fc)") && !rs.contains("fc.iter()"),
        "`fc` has no Rust binding at all -- it must never appear as a bare identifier:\n{rs}");

    // `Scale`'s own `y = [0.0 for ..xs.length]` zero-fill, for k2, must size off the
    // aliased buffer's own length -- not the nonexistent `xs` init-param identifier
    // (the `xs::length` bug) and not a stale reference to `fc`.
    assert!(rs.contains("k2.copy_y_to_device(&vec![(0) as f32; ((k1.y_buf.size() as usize / std::mem::size_of::<f32>())) as usize]);"),
        "k2's output zero-fill should size off k1's buffer directly:\n{rs}");
    assert!(!rs.contains("xs::length") && !rs.contains("xs::count"), "init-param length must not be emitted as a module path:\n{rs}");
}

// ── Transitive parameter propagation: a wrapper function forwarding to another
// Boring function (not a raw kernel constructor) qualifies too, any number of
// call-graph hops deep — see `Checker::collect_gpu_arg_params`'s fixed point.

#[test]
fn test_fn_gpu_arg_param_transitive_two_hop_wrapper() {
    // `wrap_scale_gpu` forwards its own parameter straight into `scale_gpu` — not a
    // raw kernel constructor — so it only qualifies via the *transitive* fixed point,
    // one call-graph hop beyond the base case `scale_gpu` itself uses. Exercises the
    // actual gap this fix closes: a caller passing an already-resident value into the
    // wrapper (confirmed against a real `cargo check` failure before this fix:
    // `BoringGpuArg::Host(xv.clone())` passed where `scale_gpu` expects
    // `BoringGpuArg<f64>` directly).
    let src = format!(r#"{SCALE_GPU_KERNEL}
req [float]'gpu'unified scale_gpu([float] xv, float factor):
    var [float] zero = [0.0, 0.0]
    mut k = Saxpy(factor, xv, zero)
    kernel:
        k(block = 2)
    k.y

req [float]'gpu'unified wrap_scale_gpu([float] xv, float factor):
    scale_gpu(xv, factor)

var [float] ha = [1.0, 2.0]
let [float]'gpu'unified fc = scale_gpu(ha, 2.0)
let [float]'gpu'unified fc2 = wrap_scale_gpu(fc, 3.0)
with fc2:
    print "{{fc2[0]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("fn_gpu_arg_param_transitive_wrapper", &src);

    // The wrapper's own parameter should be dual-typed too, transitively.
    assert!(rs.contains("fn wrap_scale_gpu(xv: BoringGpuArg<f64>, factor: f64) -> BoringGpuArg<f64>"),
        "wrapper's forwarded parameter should qualify transitively:\n{rs}");
    // Forwarding `xv` into `scale_gpu(xv, factor)` inside the wrapper must pass the
    // enum straight through, not re-wrap it as a host upload.
    assert!(rs.contains("scale_gpu(xv.clone()"), "forwarded resident parameter should pass straight through:\n{rs}");
    assert!(!rs.contains("BoringGpuArg::Host(xv"), "forwarded resident parameter must not be re-wrapped as a host upload:\n{rs}");
    // Call site: an already-resident value (`fc`) passed into the wrapper passes
    // straight through too.
    assert!(rs.contains("wrap_scale_gpu(fc.clone()"), "already-resident argument into the wrapper should pass straight through:\n{rs}");
    assert_eq!(rs.matches("copy_y_to_host()").count(), 0, "no kernel-field download should occur anywhere in the chain:\n{rs}");
}

#[test]
fn test_fn_gpu_arg_param_disqualified_when_any_use_is_not_qualifying() {
    // `x` is used in TWO call positions inside `mixed_use`: one at a genuinely
    // qualifying position (`scale_gpu`'s own dual-typed param, transitively valid),
    // and one at a plain, non-qualifying function (`plain_use`, an ordinary host
    // consumer). The "exclusively qualifying, everywhere in the body" rule must still
    // hold under the transitive fixed point — a single disqualifying use anywhere
    // disqualifies the whole parameter, even though another use of the same
    // parameter would, on its own, have qualified.
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'gpu'unified scale_gpu([float] x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

def plain_use([float] x, float factor):
    print "{{x[0]}}"

req [float]'gpu'unified mixed_use([float] x, float factor):
    plain_use(x, factor)
    scale_gpu(x, factor)

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let fc = mixed_use(a, 2.0)
    with fc:
        print "{{fc[0]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("fn_gpu_arg_param_disqualified_mixed_use", &src);

    assert!(!rs.contains("fn mixed_use(x: BoringGpuArg<f64>"),
        "a parameter with any non-qualifying use anywhere must not dual-type, even transitively:\n{rs}");
}

// ── Tuple-return residency chaining (`mha_step_gpu`-style: a function returning
// `([float]'gpu'unified, [float], ...)`, chaining whichever tail-tuple elements are
// themselves resident while leaving genuinely host-side elements alone) ──────────

#[test]
fn test_gpu_resident_tuple_return_chains_with_explicit_opt_in() {
    // `tuple_fn` returns a tuple whose first element is itself GPU-resident (chained
    // from a kernel-wrapper call) and whose second element is a genuinely host-side
    // array. The tail expression is a bare tuple literal `(doubled, side)` — the case
    // `try_emit_gpu_resident_tuple_return` (emit_kernel.rs) exists for. At the call
    // site, the destructured binding `r` carries an *explicit* `'gpu'unified` opt-in
    // annotation -- unlike the single-value interprocedural case, a resident tuple
    // position stays resident only when asked to (see `emit_resident_tuple_destructure`'s
    // doc for why the default has to run the other way for tuples).
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'gpu'unified scale_gpu([float] x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

req ([float]'gpu'unified, [float]) tuple_fn([float] x, float factor):
    let [float]'gpu'unified doubled = scale_gpu(x, factor)
    let [float] side = [1.0, 2.0]
    (doubled, side)

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let ([float]'gpu'unified r, [float] s) = tuple_fn(a, 2.0)
    with r:
        print "{{r[0]}}"
    print "{{s.length}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("gpu_resident_tuple_return", &src);

    // Return type: position 0 collapses to BoringGpuArg<T>, position 1 stays Vec<T>.
    assert!(rs.contains("fn tuple_fn(x: BoringGpuArg<f64>, factor: f64) -> (BoringGpuArg<f64>, Vec<f64>)"),
        "tuple return type should substitute BoringGpuArg<T> only at the resident position:\n{rs}");
    // Tail expression: element 0 (already a resident local) passes through as a
    // clone, no download; element 1 emits normally.
    assert!(rs.contains("(doubled.clone(), side.clone())"),
        "resident tuple element should pass through as a clone, not a download:\n{rs}");
    assert_eq!(rs.matches("copy_y_to_host()").count(), 0, "no kernel-field download should occur inside tuple_fn:\n{rs}");
    // Destructure at the call site: `r` opted in explicitly, so it binds straight
    // from the call with no materialization; `s` is an ordinary Vec<f64>.
    assert!(rs.contains("let (r, s) = tuple_fn(BoringGpuArg::Host(a.clone()), 2.0);"),
        "opted-in destructure should bind straight from the call, no extra materialization:\n{rs}");
}

#[test]
fn test_gpu_resident_tuple_return_destructure_materializes_by_default() {
    // Same shape as above, but the destructure has NO annotation at all on `r` —
    // the default for a resident tuple position, since tuple destructuring predates
    // this residency feature everywhere in real code (every existing unannotated
    // `let (a, b, c) = some_tuple_fn(...)` already assumes a plain, immediately
    // usable value — see `Checker::check_let_destructure`'s doc for the real
    // `cargo check` failure an opt-*out* default caused against `test_math_gpu.br`).
    // Materializes right at the destructure through a temp binding, since the call
    // must run exactly once (no re-invoking `tuple_fn` to materialize a second time).
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'gpu'unified scale_gpu([float] x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

req ([float]'gpu'unified, [float]) tuple_fn([float] x, float factor):
    let [float]'gpu'unified doubled = scale_gpu(x, factor)
    let [float] side = [1.0, 2.0]
    (doubled, side)

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let (r, s) = tuple_fn(a, 2.0)
    print "{{r[0]}}"
    print "{{s.length}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("gpu_resident_tuple_return_default_materialize", &src);

    // The call still runs exactly once, into per-position temp bindings.
    assert_eq!(rs.matches("tuple_fn(BoringGpuArg::Host(a.clone())").count(), 1,
        "tuple_fn should be called exactly once, not re-invoked to materialize a second time:\n{rs}");
    // The unannotated resident position is materialized via a temp binding, not left as a raw enum.
    assert!(rs.contains("BoringGpuArg::Resident(buf, _) => __boring_gpu_copy_d2h::<f32>(&__boring_gpu_device(), &__boring_gpu_queue(), &buf)"),
        "unannotated resident position should materialize through the free d2h helper by default:\n{rs}");
    assert!(rs.contains("let r ="), "default-materialized binding should still end up bound to the plain name `r`:\n{rs}");
}

// ── `GPU` introspection (portable between the interpreter's simulation and
// --target wgpu — see examples/saxpy.br's `GPU(0)`/`.name()`/`.totalMem()`) ────

#[test]
fn test_gpu_device_handle_and_properties() {
    let src = r#"
let g = GPU(0)
print g.name()
print g.totalMem()
print g.freeMem()
print g.computeCapability()
print g.warpSize()
print g.maxThreads()
print g.maxSharedMem()
print g.index()
"#;
    let (_wgsl, rs) = wgpu_codegen("gpu_device_properties", src);

    assert!(rs.contains("let g = ((0) as usize);"), "GPU(0) should emit a plain usize:\n{rs}");
    assert!(rs.contains("__boring_gpu_name()"), "missing .name() rewrite:\n{rs}");
    assert!(rs.contains("__boring_gpu_total_mem()"), "missing .totalMem() rewrite:\n{rs}");
    assert!(rs.contains("__boring_gpu_free_mem()"), "missing .freeMem() rewrite:\n{rs}");
    assert!(rs.contains("__boring_gpu_compute_capability()"), "missing .computeCapability() rewrite:\n{rs}");
    assert!(rs.contains("__boring_gpu_warp_size()"), "missing .warpSize() rewrite:\n{rs}");
    assert!(rs.contains("__boring_gpu_max_threads()"), "missing .maxThreads() rewrite:\n{rs}");
    assert!(rs.contains("__boring_gpu_max_shared_mem()"), "missing .maxSharedMem() rewrite:\n{rs}");
    assert!(rs.contains("(g as i64)"), "missing .index() rewrite:\n{rs}");
    // Backing globals/helpers must actually be emitted.
    assert!(rs.contains("static __BORING_GPU_ADAPTER"), "missing adapter global:\n{rs}");
    assert!(rs.contains("fn __boring_gpu_name() -> String { __boring_gpu_adapter().get_info().name }"), "missing name() helper body:\n{rs}");
    assert!(rs.contains("let _ = __BORING_GPU_ADAPTER.set("), "adapter global is never populated:\n{rs}");
}

#[test]
fn test_gpu_all_returns_single_device_and_loop_var_gets_properties() {
    let src = r#"
for g in GPU.all():
    print g.name()
    print g.index()
"#;
    let (_wgsl, rs) = wgpu_codegen("gpu_all_loop", src);

    assert!(rs.contains("for g in vec![0usize].into_iter()"), "GPU.all() should be a single-element usize vec:\n{rs}");
    assert!(rs.contains("__boring_gpu_name()"), "loop var should get the .name() rewrite:\n{rs}");
    assert!(rs.contains("(g as i64)"), "loop var should get the .index() rewrite:\n{rs}");
}

// ─── typed GpuError ─────────────────────────────────────────────────────────

#[test]
fn dispatch_pushes_outofmemory_and_validation_scopes_and_wraps_typed_gpu_error() {
    // Previously only `ErrorFilter::Validation` was pushed, and any error
    // collapsed into one generic formatted-string message -- no way to tell
    // an out-of-memory failure apart from a rejected launch config, and no
    // way for Boring source to `catch` a specific cause at all. `wgpu`
    // already exposes `ErrorFilter::OutOfMemory` unused (confirmed against
    // real wgpu 22.1.0 source); this wires it up alongside Validation and
    // classifies each into a typed `GpuError` variant wrapped in
    // `BoringError::Other`, exactly the same mechanism `throws CalcError`
    // already uses (`book.md`), so `catch GpuError.OutOfMemory:` genuinely
    // dispatches -- verified end to end via a real `cargo check` against
    // real wgpu.
    let (_, rs) = wgpu_codegen("gpu_error_scopes", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);"),
        "expected an OutOfMemory error scope pushed alongside Validation;\ngot:\n{rs}");
    assert!(rs.contains("self.device.push_error_scope(wgpu::ErrorFilter::Validation);"),
        "expected the existing Validation error scope to still be pushed;\ngot:\n{rs}");
    assert!(rs.contains("BoringError::Other(std::any::TypeId::of::<GpuError>(), Box::new(GpuError::LaunchError)"),
        "expected the Validation-scope error to classify as GpuError::LaunchError, typed via BoringError::Other;\ngot:\n{rs}");
    assert!(rs.contains("BoringError::Other(std::any::TypeId::of::<GpuError>(), Box::new(GpuError::OutOfMemory)"),
        "expected the OutOfMemory-scope error to classify as GpuError::OutOfMemory, typed via BoringError::Other;\ngot:\n{rs}");
    assert!(rs.contains("enum GpuError") && rs.contains("OutOfMemory,") && rs.contains("DeviceLost,"),
        "expected the built-in GpuError enum (all 6 documented variants) in the generated prelude;\ngot:\n{rs}");
}

#[test]
fn catch_gpu_error_by_variant_downcasts_correctly() {
    // End-to-end: a Boring `catch GpuError.OutOfMemory:` inside a `try:`
    // wrapping a `kernel:` dispatch must lower to a real BoringError
    // downcast + variant match, the same codegen shape `catch CalcError.X:`
    // already produces for a user-declared enum (book.md:6565) -- confirmed
    // this compiles clean via a real `cargo check` against real wgpu.
    let (_, rs) = wgpu_codegen("gpu_error_catch", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0

def run() throws:
    let data = [1.0, 2.0]
    mut k = Scale(data)
    try:
        kernel:
            k(block = 2)
    catch GpuError.OutOfMemory:
        print "out of memory"
    catch GpuError.LaunchError:
        print "launch error"
"#);
    assert!(rs.contains("__tid == std::any::TypeId::of::<GpuError>()"),
        "expected a TypeId-gated downcast for GpuError;\ngot:\n{rs}");
    assert!(rs.contains(".downcast_ref::<GpuError>()"),
        "expected a downcast_ref::<GpuError> call at the catch site;\ngot:\n{rs}");
    assert!(rs.contains("GpuError::OutOfMemory =>") && rs.contains("GpuError::LaunchError =>"),
        "expected both catch arms to match on the specific GpuError variant;\ngot:\n{rs}");
}

#[test]
fn test_kernel_output_field_plain_array_literal_sized_correctly() {
    // `out`'s init-body assignment is a plain bracketed literal (`ExprKind::Array`),
    // not the `[value for ..count]` fill (`ExprKind::ArrayFill`) that
    // `kernel_output_fill_map` used to be the only pattern recognized for. Before the
    // fix, this field's buffer was never covered by that map at all, so it stayed at
    // `new()`'s placeholder size (one `f32`, `4u64` bytes -- see wgpu::host's
    // `emit_kernel_new`) instead of the 8 elements the literal actually declares --
    // a real "index out of bounds: the len is 1 but the index is 1" panic on readback,
    // confirmed via a real `cargo run` against the generated project.
    let src = r#"
kernel PlainInit:
    mut [float]'unified out
    let [float]'unified vals
    let int n

    init([float]'unified data, int size):
        vals = data
        n    = size
        out  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def ():
        let tid = gpu.thread.x
        out[tid] = vals[tid] * 2.0

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
mut k = PlainInit(data, 8)
kernel:
    k(block = 8)
let result = k.out
with result:
    for i in 0..8:
        print "{i}: {result[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("kernel_output_field_plain_array_literal", src);

    // All 8 literal elements must be uploaded verbatim -- sizing `out_buf` to 8
    // `f32`s (32 bytes) via `copy_out_to_device`'s own `data.len()`-based resize,
    // not left at the constructor's placeholder allocation.
    assert!(
        rs.contains("k.copy_out_to_device(&vec![(0) as f32, (0) as f32, (0) as f32, (0) as f32, (0) as f32, (0) as f32, (0) as f32, (0) as f32]);"),
        "expected the 8-element literal to be uploaded verbatim via copy_out_to_device;\ngot:\n{rs}"
    );
}

// ─── atomic pointer indexing: `u32(...)`, not `... as u32` ────────────────────

#[test]
fn atomic_pointer_index_uses_wgsl_cast_not_rust_cast() {
    // Real, pre-existing bug, found while verifying the new atomic method
    // calls below against real naga (not just `cargo check`, which only
    // validates the Rust host side, never the WGSL a wgpu-target program
    // actually runs): the atomic-pointer helper shared by `try_atomic_assign`
    // and `try_atomic_method_call` used to emit `&buf[i as u32]` -- `as` is
    // Rust cast syntax, not valid inside a WGSL expression at all. A real
    // `naga::front::wgsl::parse_str` on the generated shader failed with
    // "expected ']', found 'as'" for a plain `counts[bucket] += 1`, meaning
    // every atomic op emitted through this path (`+= -= &= |= ^=`, and now
    // min/max/swap/cas) was unparseable WGSL until this fix -- undetected
    // because nothing in this test suite had run generated WGSL through a
    // real WGSL parser before. WGSL casts are `u32(x)`, matching the
    // (already-correct) plain `ExprKind::Index` case elsewhere in this file.
    let (wgsl, _) = wgpu_codegen("atomic_pointer_cast", r#"
kernel Histogram:
    mut [int]'actor'global counts
    init([int]'actor'global data):
        counts = data
    def ():
        let bucket = gpu.thread.x
        counts[bucket] += 1
"#);
    assert!(wgsl.contains("atomicAdd(&histogram_counts[u32(bucket)], 1);"),
        "expected u32(bucket), not 'bucket as u32' (invalid WGSL);\ngot:\n{wgsl}");
    assert!(!wgsl.contains("as u32]"),
        "must not contain the invalid 'expr as u32]' cast anywhere;\ngot:\n{wgsl}");
}

// ─── atomic min/max/swap/cas ───────────────────────────────────────────────────

#[test]
fn device_atomic_method_calls_map_to_wgsl_intrinsics() {
    // min/max/swap map directly onto WGSL's atomicMin/atomicMax/atomicExchange,
    // which already return the previous value. cas doesn't:
    // atomicCompareExchangeWeak returns a struct ({old_value, exchanged}), not
    // a bare value -- `.old_value` field access on the call result gives
    // exactly the previous value, matching every other backend's contract.
    // Verified to both parse and validate against real naga 22.1.0
    // (naga::front::wgsl::parse_str + naga::valid::Validator).
    let (wgsl, _) = wgpu_codegen("atomic_methods", r#"
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
    assert!(wgsl.contains("atomicMin(&histogram_counts[u32(bucket)], 5);"), "expected atomicMin;\ngot:\n{wgsl}");
    assert!(wgsl.contains("atomicMax(&histogram_counts[u32(bucket)], 5);"), "expected atomicMax;\ngot:\n{wgsl}");
    assert!(wgsl.contains("atomicExchange(&histogram_counts[u32(bucket)], 0)"), "expected atomicExchange;\ngot:\n{wgsl}");
    assert!(wgsl.contains("atomicCompareExchangeWeak(&histogram_counts[u32(bucket)], 0, 1).old_value"),
        "expected atomicCompareExchangeWeak(...).old_value;\ngot:\n{wgsl}");
}

#[test]
fn host_device_installs_on_uncaptured_error_handler() {
    // Pipeline creation (`emit_kernel_new`, inside a `PIPELINE.get_or_init` closure
    // that can't itself return a Result) isn't wrapped in an explicit error scope,
    // unlike dispatch() and shader-module creation -- an oversized fixed-'actor
    // field would otherwise panic via wgpu's default uncaptured-error
    // handler instead of being reported. Fixed by installing a non-panicking
    // handler once at device-creation time.
    let (_wgsl, rs) = wgpu_codegen("uncaptured_error_handler", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("device.on_uncaptured_error(Box::new(|e| eprintln!(\"boring: uncaptured GPU error: {}\", e)));"),
        "expected a non-panicking on_uncaptured_error handler installed at device-creation time;\ngot:\n{rs}");
}

// ─── Labeled multi-dimensional arrays (docs/array-multidim-types.md) ───────
// Note the `img_img` naming (the field's WGSL storage-buffer global gets a
// `{kernel_name_lowercased}_{field}` prefix) — this backend renames
// storage-buffer variables regardless of which syntax declared them.

#[test]
fn device_labeled_index_lowers_to_row_major_index() {
    let (wgsl, _) = wgpu_codegen("labeled_at", r#"
kernel Img:
    mut [float, width = 4, height = 4]'unified img
    init([float, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(wgsl.contains("img_img[u32(c + r * 4)]"),
        "expected [width=c,height=r] to lower to row-major c + r*width, with the u32 index cast this backend's Index already uses;\ngot:\n{wgsl}");
}

#[test]
fn device_labeled_axis_property_lowers_to_literals() {
    let (wgsl, _) = wgpu_codegen("labeled_width_height", r#"
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
    assert!(wgsl.contains("c < 4"), "expected img.width to lower to the literal 4;\ngot:\n{wgsl}");
    assert!(wgsl.contains("r < 8"), "expected img.height to lower to the literal 8;\ngot:\n{wgsl}");
}

#[test]
fn device_labeled_array_field_becomes_storage_buffer() {
    let (wgsl, _) = wgpu_codegen("labeled_storage_buffer", r#"
kernel Img:
    mut [float32, width = 4, height = 4]'unified img
    init([float32, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = 0.0
"#);
    assert!(wgsl.contains("var<storage, read_write> img_img: array<f32>;"),
        "expected a LabeledArray field to become a flat storage buffer, same as [T]'unified;\ngot:\n{wgsl}");
}

#[test]
fn host_labeled_array_field_dispatch_infers_2d_grid() {
    let src = r#"
kernel Img:
    mut [float, width = 16, height = 32]'unified img
    init([float, width = 16, height = 32]'unified data):
        img = data
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0

let data = [0.0]
mut k = Img(data)
kernel:
    k(block = (8, 8, 1))
"#;
    let (_wgsl, rs) = wgpu_codegen("labeled_2d_grid", src);
    assert!(rs.contains("k.dispatch((((16 + (8) - 1) / (8))) as u32, (((32 + (8) - 1) / (8))) as u32, (1) as u32)?;"),
        "expected the kernel: block with no explicit grid= to default gx/gy from width/height and the block= size;\ngot:\n{rs}");
}

#[test]
fn host_dynamic_labeled_array_field_dispatch_infers_2d_grid_from_shadow_fields() {
    let src = r#"
kernel Img:
    mut [float, width, height]'unified img
    init([float]'unified data, uint w, uint h):
        img = data.reshape(width = w, height = h)
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0

let data = [0.0]
mut k = Img(data, 16, 32)
kernel:
    k(block = (8, 8, 1))
"#;
    let (_wgsl, rs) = wgpu_codegen("dynamic_labeled_2d_grid", src);
    assert!(rs.contains("k.__img_axis0"), "expected grid.x inferred from the __img_axis0 shadow field;\ngot:\n{rs}");
    assert!(rs.contains("k.__img_axis1"), "expected grid.y inferred from the __img_axis1 shadow field;\ngot:\n{rs}");
}

#[test]
fn device_shared_labeled_array_becomes_workgroup_decl() {
    let (wgsl, _) = wgpu_codegen("shared_labeled", r#"
kernel Tile:
    mut [float32]'unified out
    let [float32, width = 4, height = 4]'actor tile
    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        out[0] = tile[width = c, height = r]
"#);
    assert!(wgsl.contains("var<workgroup> tile: array<f32, 16>;"),
        "expected a module-scope var<workgroup> declaration sized width*height;\ngot:\n{wgsl}");
}

// ─── .min/.max/.swap/.cas without 'actor — plain, non-atomic fallback ─────────

#[test]
fn atomic_method_calls_degrade_to_plain_two_statement_form_without_actor() {
    // WGSL has no statement-expression (unlike CUDA/HIP/Metal's `({ ... })`),
    // so the plain (non-atomic) fallback needs two real statements: bind the
    // let-name to the current value, then perform the update -- rather than
    // erroring or silently doing nothing off a non-actor field, matching
    // `+=`/`-=`/etc.'s existing degrade-to-plain-arithmetic behavior.
    // Verified to both parse and validate against real naga 22.1.0.
    let (wgsl, _) = wgpu_codegen("plain_atomic_methods", r#"
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
    assert!(wgsl.contains("let m = scale_buf[u32(tid)];\n    scale_buf[u32(tid)] = min(m, 5);"),
        "expected plain min as two WGSL statements;\ngot:\n{wgsl}");
    assert!(wgsl.contains("let x = scale_buf[u32(tid)];\n    scale_buf[u32(tid)] = max(x, 5);"),
        "expected plain max as two WGSL statements;\ngot:\n{wgsl}");
    assert!(wgsl.contains("let s = scale_buf[u32(tid)];\n    scale_buf[u32(tid)] = 0;"),
        "expected plain swap as two WGSL statements;\ngot:\n{wgsl}");
    assert!(wgsl.contains("let c = scale_buf[u32(tid)];\n    if (c == 0) {\n        scale_buf[u32(tid)] = 1;\n    }"),
        "expected plain cas as an if-guarded WGSL statement;\ngot:\n{wgsl}");
}

#[test]
fn atomic_method_call_discarded_uses_synthetic_name_not_reserved_underscore() {
    // Two real bugs found while verifying this against real WGSL, neither
    // just "unverified" but genuinely wrong: (1) `_ = buf[i].min(v)` used to
    // try reading `_` back to compute the update -- WGSL's `_` is a
    // write-only phony discard target, confirmed via a real naga parse
    // ("no definition in scope for identifier: '_'") when read; (2) the
    // first synthetic name tried (`__boring_discard_0`) hit WGSL's reserved
    // `__` identifier prefix (naga: "Identifier starts with a reserved
    // prefix"), the same constraint this project already documents
    // elsewhere for `__params`. Fixed by declaring a fresh `bp_discard_N`
    // `let` (matching the existing shuffle-hoist temp-naming convention)
    // instead of assigning to `_` directly. Verified to both parse and
    // validate against real naga 22.1.0.
    let (wgsl, _) = wgpu_codegen("plain_atomic_discard", r#"
kernel Scale:
    mut [int]'unified buf
    init([int]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        _ = buf[tid].min(9)
"#);
    assert!(!wgsl.contains("_ ="), "must never assign to or read back WGSL's phony `_` target;\ngot:\n{wgsl}");
    assert!(!wgsl.contains("__boring"), "must never use a WGSL-reserved `__` identifier prefix;\ngot:\n{wgsl}");
    assert!(wgsl.contains("let bp_discard_0 = scale_buf[u32(tid)];\n    scale_buf[u32(tid)] = min(bp_discard_0, 9);"),
        "expected a synthetic bp_discard_N temp instead of `_`;\ngot:\n{wgsl}");
}

#[test]
fn atomic_method_call_in_nested_position_is_a_visible_marker_not_silent_wrong_wgsl() {
    // Non-atomic min/max/swap/cas only has a real (correct) codegen path when
    // the whole call is the direct RHS of a `let`/assignment -- WGSL can't
    // express "read old, mutate, yield old" as a single expression. Buried
    // inside a larger expression, this used to either silently fall through
    // to the *unrelated* pre-existing scalar `.min`/`.max` builtin (a pure,
    // non-mutating comparison -- never touches the buffer) or, for
    // `.swap`/`.cas`, emit genuinely invalid WGSL (confirmed via a real naga
    // parse: "no definition in scope for identifier: 'swap'"). Now emits a
    // visible, unambiguous marker instead of guessing.
    let (wgsl, _) = wgpu_codegen("plain_atomic_nested", r#"
kernel Scale:
    mut [int]'unified buf
    init([int]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        let x = buf[tid].min(5) + 1
"#);
    assert!(wgsl.contains("/* unsupported here:") && wgsl.contains(".min(...) needs 'actor'global/'actor'unified"),
        "expected a visible unsupported-position marker, not silently wrong WGSL;\ngot:\n{wgsl}");
}

// ─── Host-side codegen for `Type::LabeledArray` fields (regressions found ───
// while regenerating examples/{matrix_mul_gpu,vector_add_gpu,plasma_metal}_wgpu
// for 0.9.5 — see CHANGELOG's "Known Issues"/"Fixed" entries). The device
// (WGSL) side already recognized `LabeledArray` fields correctly (the tests
// above); these check the *host* Rust side, which a fixed-shape multi-dim
// array field ('global) fell straight through as if it weren't a buffer
// field at all.

#[test]
fn host_labeled_array_field_gets_real_buffer_not_dropped_or_cast_to_i64() {
    let src = r#"
kernel Img:
    let [float32, width = 4, height = 4]'global a
    mut [float32, width = 4, height = 4]'unified c

    init([float32, width = 4, height = 4]'global input_a):
        a = input_a

    def ():
        let col = gpu.thread.x
        let row = gpu.thread.y
        c[width = col, height = row] = a[width = col, height = row] * 2.0

var [float32] data = [float32(i) for i in 0..16]
mut k = Img(data.reshape(width = 4, height = 4))
kernel:
    k(block = (4, 4))
"#;
    let (_wgsl, rs) = wgpu_codegen("labeled_field_host_buffer", src);
    assert!(rs.contains("a_buf: std::sync::Arc<wgpu::Buffer>,"),
        "a `[T, width=.., height=..]'global` field must still get a host struct buffer field;\ngot:\n{rs}");
    assert!(rs.contains("k.copy_a_to_device(&data.iter().map(|&x| x as f32).collect::<Vec<f32>>());"),
        "the constructor argument must be uploaded via copy_a_to_device, not cast straight to a scalar;\ngot:\n{rs}");
    assert!(!rs.contains("k.a = "),
        "must not fall back to a bare (wrongly-typed) field assignment for a LabeledArray buffer field;\ngot:\n{rs}");
}

#[test]
fn host_array_comprehension_loop_var_is_isize_not_i64() {
    // `int`/`uint` transpile to `isize`/`usize` as of this release (previously
    // `i64`/`u64`) -- this comprehension's implicit loop var was the one
    // codegen path that never followed, producing a `Vec<i64>` that didn't
    // match an explicitly `[int]`-typed (`Vec<isize>`) binding.
    let src = r#"
kernel Dummy:
    mut [int]'unified out
    init([int]'unified data):
        out = data
    def ():
        let tid = gpu.thread.x
        out[tid] = out[tid] * 2

var [int] host = [i for i in 0..8]
mut k = Dummy(host)
kernel:
    k(block = 8)
"#;
    let (_wgsl, rs) = wgpu_codegen("array_comp_isize", src);
    assert!(rs.contains("let mut host: Vec<isize>") && rs.contains("let i = __boring_i as isize; i "),
        "expected the comprehension's loop var cast `as isize`, matching the `Vec<isize>` binding it initializes;\ngot:\n{rs}");
    assert!(!rs.contains("as i64; i "),
        "must not cast the comprehension loop var to i64 (stale pre-isize-migration codegen);\ngot:\n{rs}");
}

#[test]
fn host_kernel_output_fill_count_resolves_promoted_top_level_const() {
    // `result = [0 for ..n]` inside `init()` refers to a top-level `let n =
    // ...`, not an init parameter -- `substitute_and_emit`'s fallback used to
    // reproduce the boring-source name verbatim, but a GPU-target top-level
    // scalar `let` is promoted to an uppercased Rust `const`
    // (`gpu_top_level_const_names`), leaving a dangling lowercase reference.
    let src = r#"
let n = 4

kernel Filler:
    let [int]'global a
    mut [int]'unified out

    init([int]'global input_a):
        a = input_a
        out = [0 for ..n]

    def ():
        let i = gpu.thread.x
        if i < n:
            out[i] = a[i]

var [int] host_a = [i for i in 0..n]
mut k = Filler(host_a)
kernel:
    k(block = 4)
"#;
    let (_wgsl, rs) = wgpu_codegen("kernel_output_fill_const_promoted", src);
    assert!(rs.contains("k.copy_out_to_device(&vec![(0) as i32; (N) as usize]);"),
        "expected the fill count to reference the uppercased promoted const N;\ngot:\n{rs}");
    assert!(!rs.contains("(n) as usize"),
        "must not leave a dangling lowercase reference to the pre-promotion name;\ngot:\n{rs}");
}

#[test]
fn host_bare_float_scalar_field_assign_casts_to_f64_not_f32() {
    // `float(expr)` is a pure alias of `float64`, not its own type (see
    // CLAUDE.md/host_scalar_type) -- `var float t` gets an `f64` host struct
    // field, so `k.t = float(screen.time)` must cast to `f64`, not join
    // `float32(expr)`'s `as f32` narrowing.
    let src = r#"
let width = 4
let height = 4
let screen = Screen(Dimension(width, height), title = "test")

kernel T:
    mut [uint]'surface pixels
    let Dimension dim
    var float t

    init(Dimension d):
        pixels = [0 for ..d.width * d.height]
        dim = d
        t = 0.0

    def ():
        pass

var mut k = T(Dimension(width, height))
kernel:
    loop:
        k.t = float(screen.time)
        k(block = (4, 4))
        break
"#;
    let (_wgsl, rs) = wgpu_codegen("float_alias_scalar_field", src);
    assert!(rs.contains("k.t = (") && rs.contains("__start_time.elapsed().as_secs_f32() as f64);"),
        "expected bare float(...) to cast to f64, matching `var float t`'s f64 host field;\ngot:\n{rs}");
    assert!(!rs.contains("__start_time.elapsed().as_secs_f32() as f32);"),
        "must not narrow a bare float(...) scalar-field assignment to f32;\ngot:\n{rs}");
}
