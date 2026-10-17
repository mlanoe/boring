// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `some` is implicit: a `T?` destination (`let`, assignment, argument, return, struct
// field, `[T?]` element) accepts a plain `T`, and `nil` is `None`. This runs one fixture
// (`tests/cases/implicit_some.br`) through the interpreter and checks the generated host
// Rust for std and every `--target` backend; the std output was additionally built and run
// with cargo (and for simd/metal/wgpu) when this was written.
//
// Run with: cargo test --test implicit_some

use std::path::Path;
use std::process::Command;

const EXPECTED: &str = "c1 5\nc2 9\nc3 3 0\nc4 4\nc5 3 -1\nc6 7\nc6b 3 0\nc7 1 0 1 2\nc8 1 0 3\nc9 x 5\nc9 none\nc10 5 -1 8\n";
const FIXTURE: &str = "tests/cases/implicit_some.br";

fn boring(args: &[&str]) -> (String, String, bool) {
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(args).output().unwrap();
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
        out.status.success(),
    )
}

#[test]
fn interpreter_accepts_plain_values_for_optional_destinations() {
    let (stdout, stderr, ok) = boring(&["run", FIXTURE]);
    assert!(ok, "{stderr}");
    assert_eq!(stdout, EXPECTED);
}

#[test]
fn std_codegen_wraps_plain_values_and_array_elements() {
    let (rust, stderr, ok) = boring(&["build", FIXTURE, "--emit-rust"]);
    assert!(ok, "{stderr}");
    assert!(rust.contains("let x: Option<isize> = Some(5);"), "{rust}");
    assert!(rust.contains("f(Some(3))"), "{rust}");
    assert!(rust.contains("vec![Some(1), None, Some(3)]"), "{rust}");
    // A `nil` element coalesces to the default as well (`[T?]` is flattened).
    assert!(rust.contains(".get((0) as usize).cloned().flatten().unwrap_or_else(|| 0)"), "{rust}");
}

#[test]
fn every_backend_emits_the_same_host_code() {
    let (std_rust, _, ok) = boring(&["build", FIXTURE, "--emit-rust"]);
    assert!(ok);
    let std_run = fn_run(&std_rust);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("implicit_some");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(FIXTURE, dir.join("implicit_some.br")).unwrap();
    for target in ["simd", "metal", "wgpu", "cuda", "rocm"] {
        let out = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target, "implicit_some.br"])
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(out.status.success(), "{target}: {}", String::from_utf8_lossy(&out.stderr));
        let main = std::fs::read_to_string(dir.join(format!("implicit_some_{target}/src/main.rs"))).unwrap();
        assert_eq!(fn_run(&main), std_run, "host code differs on --target {target}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The body of `fn run`, from its signature to the closing brace at column 0.
fn fn_run(source: &str) -> String {
    let start = source.find("fn run(").expect("fn run");
    let rest = &source[start..];
    rest[..rest.find("\n}\n").expect("end of fn run")].to_string()
}

#[test]
fn explicit_some_still_works_in_the_interpreter_and_std_codegen() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let file = dir.join("explicit_some.br");
    std::fs::write(
        &file,
        "let int? x = some(5)\nvar int? v = nil\nv = some(4)\nprint \"{x else 0} {v else 0}\"\n",
    )
    .unwrap();
    let path = file.to_str().unwrap();
    let (stdout, stderr, ok) = boring(&["run", path]);
    assert!(ok, "{stderr}");
    assert_eq!(stdout, "5 4\n");
    let (rust, stderr, ok) = boring(&["build", path, "--emit-rust"]);
    assert!(ok, "{stderr}");
    assert!(rust.contains("Some(5)"), "{rust}");
    let _ = std::fs::remove_file(&file);
}
