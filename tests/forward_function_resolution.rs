// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Free-function declarations are module-scoped. In particular, top-level
// initialization must resolve a function declared later in the same module,
// just as the transpiler's declaration pre-scan already does.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const SOURCE: &str = r#"
let answer = later()

def int later():
    return 42

print(answer)
"#;

fn fixture_dir(test_name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("forward_function_resolution")
        .join(test_name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn run_resolves_later_function_during_top_level_initialization() {
    let dir = fixture_dir("run");
    let source = dir.join("test.br");
    fs::write(&source, SOURCE).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(&source)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "boring run rejected a same-module forward function reference:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "42");
}

#[test]
fn metal_build_accepts_the_same_forward_function_graph() {
    let dir = fixture_dir("metal");
    let source = dir.join("test.br");
    fs::write(&source, SOURCE).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "metal"])
        .arg(&source)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "boring build --target metal rejected the forward function graph:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(dir.join("test_metal/src/main.rs").is_file());
    assert!(dir.join("test_metal/kernels/main.metal").is_file());
}
