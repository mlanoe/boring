// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A user struct built inside a kernel-touching function (here `main`, which calls a free
// function holding a `kernel.tensors.matmul`) is emitted by the cuda/metal/rocm host
// emitter, not the general pass. It used to come out as a call of the struct name
// (`Layer(vec![..], vec![..])`, rustc E0423) -- labels dropped, resident-array fields
// unwrapped. Transpile-only tests missed it, so the Metal crate is compiled and run too.

mod common;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const LAYER: &str = r#"
struct Layer:
    let [float32, k = 4, m = 4]'kernel'global w
    mut [float32, n = 4, m = 4]'kernel'unified out

struct Cfg:
    let int a
    let int b = 2

def run([float32, n = 4, k = 4]'kernel'global x, [float32, k = 4, m = 4]'kernel'global w, mut [float32, n = 4, m = 4]'kernel'unified y) throws:
    kernel.tensors.matmul(x, w, y)

def main() throws:
    let [float32, n = 4, k = 4]'kernel'global x = [1.0 as float32 for ..<16]
    mut [float32, n = 4, m = 4]'kernel'unified y = [0.0 as float32 for ..<16]
"#;

fn programs() -> Vec<(&'static str, String)> {
    let tail = "    let c = Cfg(a = 1)\n    let d = Cfg(3, 4)\n    run(x, [2.0 as float32 for ..<16], y)\n    with y:\n        print \"done {y[0]} {c.a + d.b}\"\n";
    vec![
        ("positional", format!("{LAYER}    let l = Layer([2.0 as float32 for ..<16], [0.0 as float32 for ..<16])\n{tail}")),
        ("named", format!("{LAYER}    let l = Layer(out = [0.0 as float32 for ..<16], w = [2.0 as float32 for ..<16])\n{tail}")),
    ]
}

fn build(tag: &str, name: &str, target: &str, src: &str) -> (PathBuf, String) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gpu_host_struct_ctor").join(format!("{tag}_{name}_{target}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let br = dir.join("main.br");
    fs::write(&br, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", target]).arg(&br).output().unwrap();
    assert!(out.status.success(), "{name}/{target}: {}", String::from_utf8_lossy(&out.stderr));
    let crate_dir = dir.join(format!("main_{target}"));
    let rs = fs::read_to_string(crate_dir.join("src/main.rs")).unwrap();
    (crate_dir, rs)
}

fn assert_literals(name: &str, target: &str, rs: &str) {
    assert!(!rs.contains("Layer(") && !rs.contains("Cfg("), "{name}/{target}: struct built by a call:\n{rs}");
    assert!(rs.contains("Layer { ") && rs.contains("BoringKernelArg::Host("), "{name}/{target}: no Layer literal");
    assert!(rs.contains("Cfg { a: 1, b: 2 }") && rs.contains("Cfg { a: 3, b: 4 }"), "{name}/{target}: Cfg literals");
}

#[test]
fn struct_literal_in_kernel_touching_main_on_every_host_emitter() {
    for (name, src) in programs() {
        for target in ["cuda", "rocm", "metal"] {
            let (_, rs) = build("text", name, target, &src);
            assert_literals(name, target, &rs);
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_crate_with_struct_literal_compiles_and_runs() {
    for (name, src) in programs() {
        let (crate_dir, _) = build("run", name, "metal", &src);
        let run = common::SharedCargo::new("gpu_host_struct_ctor")
            .args(["run", "--quiet", "--manifest-path"])
            .arg(crate_dir.join("Cargo.toml"))
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&run.stderr);
        if stderr.contains("no Metal device found") {
            // The crate compiled (the former failure); only the GPU launch is unavailable.
            eprintln!("no Metal device available, skipping the run");
            continue;
        }
        assert!(run.status.success(), "{name}: {stderr}");
        assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "done 8 5", "{name}");
    }
}
