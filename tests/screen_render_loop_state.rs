// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What a Screen `kernel: loop:` body can do with the program's own top-level state (wgpu):
//   * assign a top-level `var` (`dirty = true`) -- it is an `__App` field;
//   * use `screen.key_events` / `key_up_events` as a value (`.length`, as a function argument);
//   * mix `screen.time` (f32) with `as float` (f64) and pass it to a `float` parameter;
//   * call a method of a top-level struct value, arguments passed the way its signature takes them.
// See docs/gpu-display.md ("State between frames"). The generated crates are `cargo check`ed.

mod common;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = r#"
trait Builder:
    req int val()

struct Root as Builder:
    int v
    req int val():
        v

struct Rt:
    var int count = 0

struct Pace:
    var int n = 0
    def int frame(mut Rt rt, Builder host, int w):
        rt.count += w
        n += host.val()
        n + rt.count

def int keyCount([string] keys):
    keys.length

def float ms(float t):
    t * 1000.0

let width = 8
let height = 8
let screen = Screen(Dimension(width, height), title = "test")

var dirty = false
var ticks = 0
let Root host = Root(5)
mut Rt rt = Rt()
mut Pace pace = Pace()

kernel Fill:
    mut [uint]'surface pixels
    let Dimension dim
    var int total

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        dim = d
        total = 0

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < dim.width and row < dim.height:
            pixels[row * dim.width + col] = uint(total)

var mut k = Fill(Dimension(width, height))
kernel:
    loop:
        if screen.key_events.length > 0:
            dirty = true
        ticks += keyCount(screen.key_up_events)
        var local = 1
        local = local + 1
        let t = (screen.time as float) * 1000.0
        let r = pace.frame(rt, host, 2)
        if dirty:
            k.total = r + ticks + local + (ms(screen.time) + t) as int
            dirty = false
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

fn build(src: &str, name: &str) -> (PathBuf, Result<String, String>) {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state").join(name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", "wgpu"]).arg(&br).output().unwrap();
    let project = tmp.join("test_wgpu");
    if !out.status.success() {
        return (project, Err(String::from_utf8_lossy(&out.stderr).into_owned()));
    }
    (project.clone(), Ok(fs::read_to_string(project.join("src/main.rs")).unwrap()))
}

#[test]
fn wgpu_render_loop_state_events_time_and_methods() {
    let (project, rs) = build(PROGRAM, "state");
    let rs = rs.unwrap_or_else(|e| panic!("boring build failed:\n{e}"));
    assert!(rs.contains("self.dirty = true;"), "{rs}");
    assert!(rs.contains("self.ticks = (self.ticks + keyCount(&"), "{rs}");
    assert!(rs.contains("self.__input.key_repeat.len() as isize"), "{rs}");
    assert!(rs.contains("self.pace.frame(&mut self.rt, &self.host, 2)"), "{rs}");
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state_wgpu_target");
    let out = common::SharedCargo::new("screen_render_loop_state")
        .args(["check", "--offline", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn wgpu_render_loop_supports_float16_expressions() {
    let src = r#"
let screen = Screen(Dimension(8, 8), title = "half")
kernel Fill:
    mut [uint]'surface pixels
    var float16 value

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        value = float16(0.0)

    def ():
        pixels[.thread.x] = uint(value as float32)

var mut k = Fill(Dimension(8, 8))
kernel:
    loop:
        let half = float16(1.5)
        let sum = half + 2.0
        let mixed = sum + float32(1.0)
        k.value = mixed as float16
        k(block = 8)
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;
    let (project, rs) = build(src, "float16");
    let rs = rs.unwrap_or_else(|e| panic!("boring build failed:\n{e}"));
    assert!(rs.contains("half::f16::from_f32"), "float16 construction/cast was not lowered:\n{rs}");
    assert!(rs.contains(".to_f32()"), "mixed float16/float32 arithmetic was not promoted:\n{rs}");
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_float16_wgpu_target");
    let out = common::SharedCargo::new("screen_render_loop_float16")
        .args(["check", "--offline", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn wgpu_assigning_an_unknown_variable_in_the_loop_names_the_fix() {
    let src = PROGRAM.replace("var ticks = 0\n", "").replace("ticks += keyCount(screen.key_up_events)", "mystery = 3");
    let (_, rs) = build(&src, "unknown");
    let err = rs.expect_err("an assignment to an undeclared variable must be rejected");
    assert!(err.contains("the assigned variable must be a top-level `var`"), "{err}");
}

/// `screen.key_events.length` in an expression on the other native-window hosts.
#[test]
fn metal_cuda_and_rocm_key_events_in_an_expression() {
    let src = r#"
let screen = Screen(Dimension(8, 8), title = "test")
kernel Fill:
    mut [uint]'surface pixels
    var int total

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        total = 0

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        pixels[col] = uint(total)

var mut k = Fill(Dimension(8, 8))
kernel:
    loop:
        if screen.key_events.length > 0:
            k.total = 3
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;
    for target in ["metal", "cuda", "rocm"] {
        let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state").join(format!("keys_{target}"));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        let br = tmp.join("test.br");
        fs::write(&br, src).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", target]).arg(&br).output().unwrap();
        assert!(out.status.success(), "[{target}] {}", String::from_utf8_lossy(&out.stderr));
        let rs = fs::read_to_string(tmp.join(format!("test_{target}/src/main.rs"))).unwrap();
        assert!(rs.contains("boring_input.key_repeat.clone().len() as isize"), "[{target}] {rs}");
    }
}

/// simd: a top-level `var` the loop assigns is a local of the generated `boring_main`, not a `const`.
#[test]
fn simd_render_loop_assigns_top_level_vars() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state").join("simd_state");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, PROGRAM).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", "simd"]).arg(&br).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let project = tmp.join("test_simd");
    let rs = fs::read_to_string(project.join("src/main.rs")).unwrap();
    assert!(rs.contains("let mut dirty = false;") && rs.contains("dirty = true;"), "{rs}");
    assert!(!rs.contains("const DIRTY") && !rs.contains("const TICKS"), "{rs}");
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state_simd_target");
    let out = common::SharedCargo::new("screen_render_loop_state")
        .args(["check", "--offline", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}

/// Metal: the top-level struct values and `var`s the loop reads are built before the window opens
/// (by the general pass's `__boring_screen_setup`), and a method call on one passes each argument
/// the way its signature takes it. (`key_up_events` as an argument and `screen.time` mixed with `as float` included.)
#[test]
fn metal_render_loop_state_values_and_methods() {
    let src = PROGRAM;
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state").join("metal_state");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", "metal"]).arg(&br).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let project = tmp.join("test_metal");
    let rs = fs::read_to_string(project.join("src/main.rs")).unwrap();
    assert!(rs.contains("let (mut dirty, mut ticks, host, mut rt, mut pace) = __boring_screen_setup()?;"), "{rs}");
    assert!(rs.contains("ticks = (ticks + keyCount(&(boring_input.key_released.iter().map("), "{rs}");
    assert!(rs.contains("pace.frame(&mut rt, &host, 2)"), "{rs}");
    if cfg!(target_os = "macos") {
        let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state_metal_target");
        let out = common::SharedCargo::new("screen_render_loop_state_metal")
            .args(["check", "--offline", "--quiet", "--manifest-path"])
            .arg(project.join("Cargo.toml"))
            .arg("--target-dir")
            .arg(shared)
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap();
        assert!(out.status.success(), "generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
    }
}

/// ROCm: same as Metal -- the top-level struct values and `var`s are built by `__boring_screen_setup`
/// and bound before the window opens. No HIP toolchain here, so only the generated text is asserted.
#[test]
fn rocm_render_loop_state_values_and_methods() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state").join("rocm_state");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, PROGRAM).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", "rocm"]).arg(&br).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let rs = fs::read_to_string(tmp.join("test_rocm/src/main.rs")).unwrap();
    assert!(rs.contains("let (mut dirty, mut ticks, host, mut rt, mut pace) = __boring_screen_setup()?;"), "{rs}");
    assert!(rs.contains("dirty = true;"), "{rs}");
    assert!(rs.contains("ticks = (ticks + keyCount(&(boring_input.key_released.iter().map("), "{rs}");
    assert!(rs.contains("pace.frame(&mut rt, &host, 2)"), "{rs}");
}

#[test]
fn wgpu_render_loop_local_var_assigned_from_fn_call_in_if() {
    // `var flag = false` then `flag = f(...)` inside an `if` is a plain local assignment, not a
    // kernel rebuild (`self.flag = ...; self.__reblit()` -> E0609 no field `flag` on `__App`).
    let src = r#"
def bool syncIt(int n):
    n > 0

let screen = Screen(Dimension(8, 8), title = "t")
kernel Fill:
    mut [uint]'surface pixels
    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
    def ():
        pixels[.thread.x] = uint(1)
var mut k = Fill(Dimension(8, 8))
kernel:
    loop:
        var flag = false
        if screen.focus_gained:
            flag = syncIt(2)
        let other = flag or screen.resized
        k(block = (4, 4))
        if other:
            screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;
    let (project, rs) = build(src, "local_var_fn_assign");
    let rs = rs.unwrap_or_else(|e| panic!("boring build failed:\n{e}"));
    assert!(rs.contains("flag = syncIt(2);"), "{rs}");
    assert!(!rs.contains("self.flag"), "{rs}");
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state_wgpu_target");
    let out = common::SharedCargo::new("screen_render_loop_state")
        .args(["check", "--offline", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}

// `float(screen.time)` is an f64 (`screen.time` itself is an f32), so it mixes with a `float` field.
#[test]
fn wgpu_float_of_screen_time_is_f64() {
    let src = r#"
struct Pace:
    var float nextPoll = 0.0

let screen = Screen(Dimension(8, 8), title = "t")
mut Pace pace = Pace()
kernel Fill:
    mut [uint]'surface pixels
    var int total

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        total = 0

    def ():
        pixels[.thread.x] = uint(total)

var mut k = Fill(Dimension(8, 8))
kernel:
    loop:
        if float(screen.time) >= pace.nextPoll:
            pace.nextPoll = float(screen.time) + 2.0
            k.total = 1
        let small = float32(screen.time)
        k.total += int(small)
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;
    let (project, rs) = build(src, "float_time");
    let rs = rs.unwrap_or_else(|e| panic!("boring build failed:\n{e}"));
    assert!(rs.contains("as_secs_f32() as f64)"), "{rs}");
    assert!(rs.contains("as_secs_f32() as f32)"), "{rs}");
    assert!(rs.contains("self.k.total = (((self.k.total as isize) +"), "kernel field read in a compound assignment: {rs}");
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_state_wgpu_target");
    let out = common::SharedCargo::new("screen_render_loop_state")
        .args(["check", "--offline", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}
