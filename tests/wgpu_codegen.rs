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
