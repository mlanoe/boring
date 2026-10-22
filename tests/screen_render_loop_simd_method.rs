// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// simd: a Screen `kernel: loop:` calling a method of a top-level struct value takes its arguments the
// way a free function does (`&mut rt`, `&host`), never `&mut *rt` (E0614). See docs/gpu-display.md.

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

let width = 8
let height = 8
let screen = Screen(Dimension(width, height), title = "test")

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
        let r = pace.frame(rt, host, 2)
        k.total = r
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

#[test]
fn simd_render_loop_state_method_call() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_simd_method");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, PROGRAM).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", "simd"]).arg(&br).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let project = tmp.join("test_simd");
    let rs = fs::read_to_string(project.join("src/main.rs")).unwrap();
    assert!(rs.contains("pace.frame(&mut rt, &host, 2)"), "{rs}");
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("screen_render_loop_simd_method_target");
    let out = common::SharedCargo::new("screen_render_loop_simd_method")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}
