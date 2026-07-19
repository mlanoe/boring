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

fn run_wgpu(test_name: &str, src: &str) -> (String, String, String) {
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
        read("src/main.rs"),
        read("Cargo.toml"),
    )
}

fn wgpu_codegen(test_name: &str, src: &str) -> (String, String) {
    let (wgsl, rs, _toml) = run_wgpu(test_name, src);
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
fn test_scalar_uniform() {
    let src = r#"
kernel Scale:
    mut [float]'unified data
    let float alpha

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
    let [float, 256]'sync tile
    mut [float]'unified data

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
fn test_cargo_toml_deps() {
    let src = r#"
kernel Empty:
    mut [float]'unified data
    def ():
        data[0] = 1.0
"#;
    let (_wgsl, _rs, toml) = run_wgpu("cargo_toml", src);

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
