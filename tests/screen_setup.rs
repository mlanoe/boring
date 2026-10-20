// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A `Screen` program's top level is ordinary code: kernel constructors take any expression and
// run the kernel's `init`, plain `let`s (a display list, a runtime...) are built before the
// loop, the loop body can bind locals and call ordinary functions, and module-level constants
// are real `const`s the functions can read.
//
// wgpu generated crates are `cargo check`ed (no window is opened); cuda / rocm have no toolchain
// here, so only their generated host text is asserted.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = r#"
struct Frame:
    [uint] f
    int total

pub let float scale = 2.0
pub let LIMIT = 3

def Frame makeFrame(int n):
    Frame([uint(n), 1, 2, 3], 3 + n)

def [uint] palette(int n):
    [0xFF0000FF, uint(n)]

def float half(float v):
    v * scale

def int limit():
    LIMIT + 1

def int total(Frame fr):
    fr.total + limit()

let width = 8
let height = 8
let screen = Screen(Dimension(width, height), title = "test")

# not a kernel constructor, not a literal: built before the loop, read by it
let base = makeFrame(LIMIT)
let seeds = palette(total(base))

kernel Fill:
    mut [uint]'surface pixels
    let [uint]'global colors
    let int w
    let int h
    var int total

    init([uint] c, int width, int height, int t):
        pixels = [0 for ..<width * height]
        colors = c
        w = width
        h = height
        total = t

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < w and row < h:
            pixels[row * w + col] = colors[0] + uint(total)

var mut k = Fill(seeds, width, height, total(base))
kernel:
    loop:
        let n = limit()
        let fr = makeFrame(n)
        if total(fr) > 1000 and total(base) > 0:
            break
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

fn build(target: &str) -> (PathBuf, String) {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_setup").join(target);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, PROGRAM).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target])
        .arg(&br)
        .output()
        .unwrap();
    assert!(out.status.success(), "[{target}] boring build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let project = tmp.join(format!("test_{target}"));
    let rs = fs::read_to_string(project.join("src/main.rs")).unwrap();
    (project, rs)
}

#[test]
fn wgpu_screen_program_runs_kernel_init_and_top_level_code() {
    let (project, rs) = build("wgpu");
    // the constructor and its `init` run, arguments evaluated once, before the window opens
    assert!(rs.contains("fn __boring_screen_setup() -> Result<(Frame, Fill)"), "{rs}");
    assert!(rs.contains("let mut k = Fill::new(__boring_gpu_device(), __boring_gpu_queue());"), "{rs}");
    assert!(rs.contains("k.copy_colors_to_device("), "init must upload the array argument:\n{rs}");
    assert!(rs.contains("k.copy_pixels_to_device("), "{rs}");
    assert!(rs.contains("k.total = "), "{rs}");
    assert!(!rs.contains("let k = Fill::new(std::sync::Arc::clone(&device)"), "kernel built without init:\n{rs}");
    assert!(rs.contains("match __boring_screen_setup()"), "{rs}");
    // the plain top-level `let`s are ordinary statements of the setup function
    assert!(rs.contains("let base = makeFrame(LIMIT);"), "{rs}");
    assert!(rs.contains("let seeds = palette("), "{rs}");
    // module-level constants are items, not locals of `main`
    assert!(rs.contains("pub const SCALE: f64 = 2.0;"), "{rs}");
    assert!(rs.contains("pub const LIMIT: isize = 3;"), "{rs}");
    assert!(!rs.contains("let LIMIT"), "{rs}");
    assert!(!rs.contains("scale: "), "unused imported constants must not become __App fields:\n{rs}");
    // the render loop binds locals and calls ordinary functions
    assert!(rs.contains("let n = limit();"), "{rs}");
    assert!(rs.contains("let fr = makeFrame(n);"), "{rs}");
    assert!(rs.contains("total(&fr)"), "arguments follow the callee's signature:\n{rs}");
    assert!(rs.contains("total(&self.base)"), "setup values live on __App:\n{rs}");
    cargo_check(&project);
}

fn cargo_check(project: &PathBuf) {
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_setup_wgpu_target");
    let out = Command::new("cargo")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn cuda_and_rocm_screen_programs_keep_module_level_constants() {
    for target in ["cuda", "rocm"] {
        let (_, rs) = build(target);
        assert!(rs.contains("pub const SCALE: f64 = 2.0;"), "[{target}] {rs}");
        assert!(rs.contains("pub const LIMIT: isize = 3;"), "[{target}] {rs}");
        assert!(rs.contains("fn limit()"), "[{target}] {rs}");
    }
}
