// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Overloaded free functions (`size(int)` / `size(string)`) called from a `Screen` render loop, and from
// plain code, resolve to the overload the argument's type selects -- a literal-initialised local
// included -- on `simd`, `wgpu` and the standard target.

mod common;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const OVERLOADS: &str = r#"
def int size(int x):
    x + 1

def int size(string s):
    s.length

"#;

const LOOP: &str = r#"
let width = 8
let height = 8
let screen = Screen(Dimension(width, height), title = "test")

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
        let cluster = "hello"
        let n = 3
        k.total = size(cluster) + size(n)
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;

const PLAIN: &str = r#"
def main():
    let cluster = "hello"
    let n = 3
    print "{size(cluster) + size(n)}"
"#;

fn build(name: &str, program: &str, target: Option<&str>) -> (PathBuf, String) {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, program).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_boring"));
    cmd.arg("build");
    if let Some(t) = target { cmd.args(["--target", t]); }
    let out = cmd.arg(&br).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let dir = tmp.join(match target { Some(t) => format!("test_{t}"), None => "test_rust".into() });
    let rs = fs::read_to_string(dir.join("src/main.rs")).unwrap();
    (dir, rs)
}

#[test]
fn simd_render_loop_overload_by_local_type() {
    let (project, rs) = build("overload_render_loop_simd", &format!("{OVERLOADS}{LOOP}"), Some("simd"));
    assert!(rs.contains("size__string(cluster") && rs.contains("size__int(n)"), "{rs}");
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("overload_render_loop_simd_target");
    let out = common::SharedCargo::new("overload_render_loop_simd")
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
fn wgpu_render_loop_overload_by_local_type() {
    let (_, rs) = build("overload_render_loop_wgpu", &format!("{OVERLOADS}{LOOP}"), Some("wgpu"));
    assert!(rs.contains("size__string(cluster)") && rs.contains("size__int(n)"), "{rs}");
}

#[test]
fn plain_overload_by_local_type() {
    let (_, rs) = build("overload_plain", &format!("{OVERLOADS}{PLAIN}"), None);
    assert!(rs.contains("size__string(cluster") && rs.contains("size__int(n)"), "{rs}");
}
