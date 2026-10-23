// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A `kernel: loop:` render loop can replace a kernel's array fields between frames
// (`k.colors = palette(screen.frame)`, `k.f = enc.f`), read `screen.frame`, and a Screen
// can redraw on demand (`Screen(..., redraw = "demand")` + `screen.requestRedraw()`).
// See docs/gpu-display.md ("Updating arrays between frames", "On-demand redraw").
//
// wgpu / metal / simd generated crates are `cargo check`ed (no window is opened);
// cuda / rocm have no toolchain here, so only their generated host text is asserted.

mod common;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = r#"
let width = 8
let height = 8
let screen = Screen(Dimension(width, height), title = "test"REDRAW)

struct Enc:
    [uint] f
    int total

Enc encode(int frame):
    # the array length changes from frame to frame
    Enc([uint(frame), 1, 2, 3][..<(2 + frame % 3)], 3 + frame)

kernel Fill:
    mut [uint]'surface pixels
    let [uint]'global colors
    let Dimension dim
    var int total

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        colors = [0xFF0000FF]
        dim = d
        total = 0

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < dim.width and row < dim.height:
            pixels[row * dim.width + col] = colors[0] + uint(total)

var mut k = Fill(Dimension(width, height))
kernel:
    loop:
        let enc = encode(screen.frame)
        k.colors = enc.f
        k.total = enc.total
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key_pressed(" "):
            screen.requestRedraw()
        if screen.key("\x1B"):
            break
"#;

fn source(on_demand: bool) -> String {
    PROGRAM.replace("REDRAW", if on_demand { ", redraw = \"demand\"" } else { "" })
}

/// Builds `src` for `target` and returns `(project dir, generated host rs)`.
fn build(target: &str, name: &str, src: &str) -> (PathBuf, String) {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_arrays").join(format!("{name}_{target}"));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target])
        .arg(&br)
        .output()
        .unwrap();
    assert!(out.status.success(), "[{name}/{target}] boring build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let project = tmp.join(format!("test_{target}"));
    let rs = fs::read_to_string(project.join("src/main.rs")).unwrap();
    (project, rs)
}

/// `cargo check` the generated crate; one shared target dir per backend so the heavy
/// dependencies are compiled once.
fn cargo_check(project: &std::path::Path, target: &str) {
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("screen_render_loop_arrays_{target}_target"));
    let out = common::SharedCargo::new("screen_render_loop_arrays")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "[{target}] generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn wgpu_render_loop_replaces_array_fields_and_reads_frame() {
    let (project, rs) = build("wgpu", "continuous", &source(false));
    assert!(rs.contains("self.k.copy_colors_to_device(&__v.iter().map(|&x| x as u32).collect::<Vec<u32>>())"), "{rs}");
    assert!(rs.contains("let enc = encode(self.__frame);"), "{rs}");
    assert!(rs.contains("self.k.total = (enc.total) as i32;"), "{rs}");
    assert!(!rs.contains("self.k.colors ="), "array field must not be assigned directly:\n{rs}");
    // Continuous mode: every frame schedules the next one.
    assert!(rs.contains("self.__frame += 1;"), "{rs}");
    assert!(rs.contains("if let Some(w) = &self.window { w.request_redraw(); }"), "{rs}");
    assert!(!rs.contains("ControlFlow::Wait"), "{rs}");
    cargo_check(&project, "wgpu");
}

#[test]
fn wgpu_on_demand_redraw_waits_for_events() {
    let (project, rs) = build("wgpu", "demand", &source(true));
    assert!(rs.contains("event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);"), "{rs}");
    assert!(rs.contains("WindowEvent::KeyboardInput { .. } | WindowEvent::CursorMoved { .. }"), "{rs}");
    assert!(rs.contains("self.__redraw = true;"), "screen.requestRedraw():\n{rs}");
    assert!(rs.contains("if std::mem::take(&mut self.__redraw)"), "{rs}");
    cargo_check(&project, "wgpu");
}

#[test]
fn metal_render_loop_replaces_array_fields_and_reads_frame() {
    let (project, rs) = build("metal", "continuous", &source(false));
    assert!(rs.contains("k.colors = { let __v = &(enc.f); boring_screen_upload(&k.__device"), "{rs}");
    assert!(rs.contains("boring_screen_upload(&k.__device"), "{rs}");
    assert!(rs.contains("fn encode("), "plain host fns reach a Screen program:\n{rs}");
    assert!(rs.contains("ControlFlow::Poll"), "{rs}");
    assert!(!rs.contains("boring_dirty = false"), "{rs}");
    if cfg!(target_os = "macos") {
        cargo_check(&project, "metal");
    }
}

#[test]
fn metal_on_demand_redraw_waits_for_events() {
    let (project, rs) = build("metal", "demand", &source(true));
    assert!(rs.contains("*__boring_cf = winit::event_loop::ControlFlow::Wait;"), "{rs}");
    assert!(rs.contains("if !boring_dirty { return; }"), "{rs}");
    assert!(rs.contains("boring_dirty = true;"), "screen.requestRedraw():\n{rs}");
    if cfg!(target_os = "macos") {
        cargo_check(&project, "metal");
    }
}

#[test]
fn simd_render_loop_replaces_array_fields_and_reads_frame() {
    let (project, rs) = build("simd", "continuous", &source(false));
    assert!(rs.contains("k.copy_colors_to_device(&__v.iter().map(|&x| x as u32).collect::<Vec<u32>>())"), "{rs}");
    assert!(rs.contains("fn __boring_redraw_on_demand() -> bool { false }"), "{rs}");
    cargo_check(&project, "simd");
}

#[test]
fn simd_on_demand_redraw_sleeps_in_present() {
    let (project, rs) = build("simd", "demand", &source(true));
    assert!(rs.contains("fn __boring_redraw_on_demand() -> bool { true }"), "{rs}");
    assert!(rs.contains("screen.request_redraw()"), "{rs}");
    assert!(rs.contains("dirty_cv.wait(dirty)"), "{rs}");
    cargo_check(&project, "simd");
}

#[test]
fn cuda_and_rocm_render_loops_replace_array_fields() {
    for (target, htod) in [("cuda", "clone_htod::<usize, _>"), ("rocm", "clone_htod::<usize>")] {
        let (_, rs) = build(target, "continuous", &source(false));
        assert!(rs.contains("k.as_mut().unwrap().colors = Arc::new(boring_new_stream_with_priority(&k.as_ref().unwrap().__ctx, 0)"), "[{target}] {rs}");
        assert!(rs.contains(htod), "[{target}] {rs}");
        assert!(rs.contains("let enc = encode((boring_frame as isize));"), "[{target}] {rs}");
        assert!(rs.contains("ControlFlow::Poll"), "[{target}] {rs}");

        let (_, rs) = build(target, "demand", &source(true));
        assert!(rs.contains("ControlFlow::Wait"), "[{target}] {rs}");
        assert!(rs.contains("if !boring_dirty { return; }"), "[{target}] {rs}");
        assert!(rs.contains("boring_dirty = true;"), "[{target}] {rs}");
    }
}
