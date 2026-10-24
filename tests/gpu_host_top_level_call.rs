// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A bare call, at the top level (no `def main`), to a free function that touches a kernel
// must keep the top level on the cuda/metal/rocm host emitter: that emitter wraps the
// arguments in `BoringKernelArg::Host`. It used to be left to the general pass, which
// passed plain `Vec<T>` (rustc E0308) -- the call's shape (bare, `let r = ...`, nested in
// `if`/`for`) must not matter. Transpile-only tests missed it, so the Metal crate is
// compiled and run too.

mod common;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const HEAD: &str = r#"
def run([float32, n = 4, k = 4]'kernel'global x, [float32, k = 4, m = 4]'kernel'global w, mut [float32, n = 4, m = 4]'kernel'unified y) throws:
    kernel.tensors.matmul(x, w, y)

let [float32, n = 4, k = 4]'kernel'global x = [1.0 as float32 for ..<16]
mut [float32, n = 4, m = 4]'kernel'unified y = [0.0 as float32 for ..<16]
"#;

const CALL: &str = "run(x, [2.0 as float32 for ..<16], y)";

fn programs() -> Vec<(&'static str, String)> {
    vec![
        ("bare", format!("{HEAD}{CALL}\nprint \"done\"\n")),
        ("let", format!("{HEAD}let r = {CALL}\nprint \"done\"\n")),
        ("if", format!("{HEAD}if true:\n    {CALL}\nprint \"done\"\n")),
        ("for", format!("{HEAD}for i in 0..<2:\n    {CALL}\nprint \"done\"\n")),
    ]
}

fn build(tag: &str, name: &str, target: &str, src: &str) -> (PathBuf, String) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gpu_host_top_level_call").join(format!("{tag}_{name}_{target}"));
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

#[test]
fn top_level_call_wraps_arguments_on_every_host_emitter() {
    for (name, src) in programs() {
        for target in ["cuda", "rocm", "metal"] {
            let (_, rs) = build("text", name, target, &src);
            assert!(rs.contains("run(BoringKernelArg::Host("), "{name}/{target}: argument not wrapped:\n{rs}");
            assert!(!rs.contains("run(x.clone()"), "{name}/{target}: call emitted by the general pass:\n{rs}");
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_crate_with_top_level_call_compiles_and_runs() {
    for (name, src) in programs() {
        let (crate_dir, _) = build("run", name, "metal", &src);
        let run = common::SharedCargo::new("gpu_host_top_level_call")
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
        assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "done", "{name}");
    }
}
