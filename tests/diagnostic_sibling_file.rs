// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a diagnostic for code that lives in a sibling file reached through a bare
// `use <name>` must name that sibling file, with its own line and source excerpt -- not the entry
// file. The files of a project are merged into one item list before the checker/transpiler run,
// and the error line used to be resolved against the entry file (right line number, wrong file
// and excerpt).
//
// Run with:
//   cargo test --test diagnostic_sibling_file

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Entry file: long enough that the sibling's error line also exists in it, with different text
/// (so a wrong-file excerpt cannot be mistaken for the right one).
const MAIN: &str = "use model\n\ndef main():\n    print \"entry-a\"\n    print \"entry-b\"\n    print \"entry-c\"\n    print \"entry-d\"\n    print \"entry-e\"\n    print \"entry-f\"\n    print \"entry-g\"\n    print \"entry-h\"\n    print \"entry-i\"\n    print \"entry-j\"\n    print \"entry-k\"\n    print \"entry-l\"\n";

fn project(name: &str, model: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("diagnostic_sibling_file").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(dir.join("boring.toml"), "[project]\nname = \"sibling_diag\"\n").unwrap();
    fs::write(dir.join("src/main.br"), MAIN).unwrap();
    fs::write(dir.join("src/model.br"), model).unwrap();
    dir
}

/// Builds the project's `src/main.br` and asserts it fails with `message` reported at
/// `src/model.br:<line>` and the excerpt `excerpt`, never against the entry file.
fn assert_reported_in_sibling(name: &str, model: &str, extra: &[&str], message: &str, line: usize, excerpt: &str) {
    let dir = project(name, model);
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("build")
        .args(extra)
        .arg("--emit-rust")
        .arg("src/main.br")
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(!out.status.success(), "[{name}] expected a failing build");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains(message), "[{name}] missing `{message}` in:\n{stderr}");
    assert!(
        stderr.contains(&format!("model.br:{line}")),
        "[{name}] expected the location `model.br:{line}` in:\n{stderr}"
    );
    assert!(!stderr.contains("main.br"), "[{name}] diagnostic wrongly names the entry file:\n{stderr}");
    assert!(
        stderr.contains(&format!("{line} | {excerpt}")),
        "[{name}] expected the excerpt `{excerpt}` of the sibling file in:\n{stderr}"
    );
    assert!(!stderr.contains("entry-"), "[{name}] excerpt taken from the entry file:\n{stderr}");
    assert!(!stderr.contains("1048"), "[{name}] internal line tag leaked into:\n{stderr}");
}

// Line 11 of the sibling, same class of error as the real-world report (found after the merge).
const GPU_RESIDENT: &str = "pub req [float32]'kernel'unified scale_gpu([float32]'kernel'global x, int n) throws:\n    kernel(block = 32, grid = 1):\n        mut [float32]'unified out = [0.0 as float32 for ..<n]\n        let i = .thread.x\n        if i < n:\n            out[i] = x[i] * 2.0 as float32\n        out\n\ndef float32 bad([float32]'kernel'global x) throws:\n    let y = scale_gpu(x, x.length)\n    x[0]\n";

#[test]
fn gpu_resident_index_error_names_the_sibling_file() {
    for target in ["metal", "wgpu", "cuda"] {
        assert_reported_in_sibling(
            &format!("gpu_resident_{target}"),
            GPU_RESIDENT,
            &["--target", target],
            "is GPU-resident and cannot be indexed directly",
            11,
            "    x[0]",
        );
    }
}

#[test]
fn parse_error_names_the_sibling_file() {
    let model = "def int one():\n    1\n\ndef bad(:\n    print \"x\"\n";
    for (name, extra) in [("parse_std", vec![]), ("parse_metal", vec!["--target", "metal"])] {
        assert_reported_in_sibling(name, model, &extra, "expected identifier", 4, "def bad(:");
    }
}

#[test]
fn transpile_error_names_the_sibling_file() {
    let model = "def main2():\n    start = 1\n    print \"{start}\"\n";
    for (name, extra) in [("assign_std", vec![]), ("assign_metal", vec!["--target", "metal"])] {
        assert_reported_in_sibling(name, model, &extra, "cannot assign to undeclared variable `start`", 2, "    start = 1");
    }
}

#[test]
fn bare_sibling_method_call_error_names_the_sibling_file() {
    let model = "struct S:\n    int v\n    def int one():\n        1\n    def int two():\n        one()\n";
    for (name, extra) in [("method_std", vec![]), ("method_metal", vec!["--target", "metal"])] {
        assert_reported_in_sibling(name, model, &extra, "cannot call method 'one' without a receiver", 6, "        one()");
    }
}

#[test]
fn entry_file_errors_keep_naming_the_entry_file() {
    let dir = project("entry_error", "def int one():\n    1\n");
    fs::write(dir.join("src/main.br"), "use model\n\ndef main():\n    start = 1\n    print \"{start}\"\n").unwrap();
    for extra in [vec![], vec!["--target", "metal"]] {
        let out = Command::new(env!("CARGO_BIN_EXE_boring"))
            .arg("build").args(&extra).arg("--emit-rust").arg("src/main.br")
            .current_dir(&dir).output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success());
        assert!(stderr.contains("main.br:4"), "{stderr}");
        assert!(stderr.contains("4 |     start = 1"), "{stderr}");
        assert!(!stderr.contains("model.br"), "{stderr}");
    }
}
