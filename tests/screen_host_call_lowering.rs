// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Lowering of ordinary-function calls in a wgpu `Screen` render loop, and of `mut` struct
// constructors in the setup code (`fn __boring_screen_setup`):
//  - `mut rt = Runtime()` is an owned `let mut rt = Runtime { .. }`, not a `&mut` borrow;
//  - a free function named like a method of some struct (`release`) is called with the
//    arguments its own signature wants, not the same-named method's;
//  - `screen.mouse.scroll_x as float` converts the member (here `f32`) to the parameter type.
// The generated crate is `cargo check`ed (no window is opened).

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = r#"
trait Builder:
    req int id()

struct Root as Builder:
    req int id(): 7

struct Runtime:
    var int downs = 0
    def start(Builder host):
        downs = host.id()

struct Pool:
    var int free = 0
    def release(int n):
        free += n

def void press(mut Runtime rt, Builder host, int x, int y):
    rt.downs += x + y + host.id()

def void release(mut Runtime rt, Builder host, int x, int y):
    rt.downs -= x + y + host.id()

def void wheel(mut Runtime rt, Builder host, int x, int y, float dx, float dy):
    rt.downs += x + y + int(dx + dy)

def int peek(Runtime rt):
    rt.downs

def int bump(mut Pool pool):
    pool.release(1)
    pool.free

# the last function declared has a parameter named like the setup's `mut rt`
def int tail(Runtime rt, int w):
    rt.downs + w

let width = 8
let height = 8
let screen = Screen(Dimension(width, height), title = "test")

let host = Root()
mut rt = Runtime()
rt.start(host)
mut pool = Pool()
let first = bump(pool)

kernel Fill:
    mut [uint]'surface pixels
    let int w

    init(int width, int height):
        pixels = [0 for ..<width * height]
        w = width

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < w:
            pixels[row * w + col] = 1

var mut k = Fill(width, height)
kernel:
    loop:
        if screen.mouse_pressed("left"):
            press(rt, host, screen.mouse.x, screen.mouse.y)
        if screen.mouse_released("left"):
            release(rt, host, screen.mouse.x, screen.mouse.y)
        if screen.mouse.scroll_x != 0.0 or screen.mouse.scroll_y != 0.0:
            wheel(rt, host, screen.mouse.x, screen.mouse.y, screen.mouse.scroll_x as float, screen.mouse.scroll_y as float)
        if peek(rt) > 100:
            break
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

#[test]
fn screen_loop_calls_follow_the_free_function_signature() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_host_call_lowering");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, PROGRAM).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "wgpu"])
        .arg(&br)
        .output()
        .unwrap();
    assert!(out.status.success(), "boring build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    let project = tmp.join("test_wgpu");
    let rs = fs::read_to_string(project.join("src/main.rs")).unwrap();

    // (1) owned, not a borrow
    assert!(rs.contains("let mut rt = Runtime {") || rs.contains("let mut rt: Runtime = Runtime {"), "{rs}");
    assert!(!rs.contains("let mut rt: &"), "{rs}");
    assert!(!rs.contains("rt: &mut Runtime = &mut Runtime"), "{rs}");
    // (2) `release` is the free function, not `Pool::release`
    assert!(rs.contains("release(&mut self.rt, Box::new(self.host.clone()), self.__mouse_x, self.__mouse_y)"), "{rs}");
    // (3) casts of members are converted, never an unsupported expression
    assert!(!rs.contains("/* unsupported expr */"), "{rs}");
    assert!(rs.contains("self.__input.scroll_x as f64"), "{rs}");

    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_setup_wgpu_target");
    let check = Command::new("cargo")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(check.status.success(), "generated crate does not compile:\n{}", String::from_utf8_lossy(&check.stderr));
}
