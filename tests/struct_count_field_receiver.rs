// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A user struct field/method literally named `count`/`length`, read on a *local* value
// (not only `self`), must win over the array-length builtin (`.len() as isize`) on every
// GPU/host backend. `.count`/`.length` on real arrays keeps the builtin.

mod common;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = r#"
let width = 8
let height = 8
let screen = Screen(Dimension(width, height), title = "test")

struct Enc:
    [uint] f
    int count
    int length
    req int count_twice():
        count * 2

struct Ledger:
    int total
    req int count():
        total
    req int length():
        total + 1

Enc encode(int frame):
    Enc([uint(frame), 1, 2], 3 + frame, 4)

Ledger ledger():
    Ledger(7)

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
        let typed Enc other = encode(0)
        let arr = [1, 2, 3]
        k.colors = enc.f
        k.total = enc.count + enc.length + other.count + arr.count + arr.length
        let led = ledger()
        k.total = led.count() + led.length() + encode(1).count
        if screen.key("\x1B"):
            break
"#;

fn build(target: &str) -> (PathBuf, String) {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("struct_count_field_receiver").join(target);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br = tmp.join("test.br");
    fs::write(&br, PROGRAM.replace("let typed Enc other", "let Enc other")).unwrap();
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

fn cargo_check(project: &std::path::Path, target: &str) {
    let shared = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("struct_count_field_receiver_{target}_target"));
    let out = common::SharedCargo::new("struct_count_field_receiver")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(shared)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "[{target}] generated crate does not compile:\n{}", String::from_utf8_lossy(&out.stderr));
}

fn assert_field_wins(target: &str, rs: &str) {
    for bad in ["enc.len()", "other.len()", "encode(1).len()", "encode(1_isize).len()"] {
        assert!(!rs.contains(bad), "[{target}] `{bad}` — struct field shadowed by the array-length builtin:\n{rs}");
    }
    assert!(rs.contains("enc.count"), "[{target}] expected real `enc.count`:\n{rs}");
    assert!(rs.contains("enc.length"), "[{target}] expected real `enc.length`:\n{rs}");
    assert!(rs.contains("other.count"), "[{target}] expected annotated-local `other.count`:\n{rs}");
    assert!(!rs.contains("led.len()"), "[{target}] `led.count()` shadowed:\n{rs}");
    assert!(rs.contains("led.count()"), "[{target}] expected real `led.count()`:\n{rs}");
    assert!(rs.contains("arr.len()"), "[{target}] real array `.count` must keep the builtin:\n{rs}");
}

#[test]
fn metal_struct_field_count_on_local_not_shadowed() {
    let (project, rs) = build("metal");
    assert_field_wins("metal", &rs);
    if cfg!(target_os = "macos") {
        cargo_check(&project, "metal");
    }
}

#[test]
fn cuda_struct_field_count_on_local_not_shadowed() {
    let (_, rs) = build("cuda");
    assert_field_wins("cuda", &rs);
}

#[test]
fn rocm_struct_field_count_on_local_not_shadowed() {
    let (_, rs) = build("rocm");
    assert_field_wins("rocm", &rs);
}

#[test]
fn simd_struct_field_count_on_local_not_shadowed() {
    let (project, rs) = build("simd");
    assert_field_wins("simd", &rs);
    cargo_check(&project, "simd");
}

#[test]
fn wgpu_struct_field_count_on_local_not_shadowed() {
    let (project, rs) = build("wgpu");
    assert!(!rs.contains("unsupported expr"), "[wgpu] unsupported host expr:\n{rs}");
    for bad in ["enc.len()", "other.len()", "led.len()", "encode(1).len()"] {
        assert!(!rs.contains(bad), "[wgpu] `{bad}` — struct member shadowed by the builtin:\n{rs}");
    }
    assert!(rs.contains("enc.count") && rs.contains("other.count"), "{rs}");
    assert!(rs.contains("led.count()"), "{rs}");
    assert!(rs.contains("(arr.len() as isize)"), "real array `.count` must keep the builtin:\n{rs}");
    cargo_check(&project, "wgpu");
}
