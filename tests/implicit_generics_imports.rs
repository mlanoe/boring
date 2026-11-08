// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Implicit generics, phase 4 step 2: the bounds an untyped parameter needs flow through a call
// into a function of ANOTHER file (a bare sibling `use`, or a multi-segment `use` of a `pub`
// function), not only into one of the same file. Without it `def show(x): label(x)` was rejected
// by rustc with the `Display` bound of `label(y)` named.
//
// Run with:
//   tools/test.sh --test implicit_generics_imports

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("implicit_generics_imports").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    for (file, src) in files {
        let p = dir.join(file);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, src).unwrap();
    }
    dir
}

fn boring(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boring")).current_dir(dir).args(args).output().unwrap()
}

fn text(b: &[u8]) -> String { String::from_utf8_lossy(b).into_owned() }

fn cargo_run(dir: &Path, mode: &str, threading: &str) -> String {
    let out_dir = dir.join(format!("out_{mode}_{threading}"));
    let out = boring(dir, &["build", "main.br", "--mode", mode, "--threading", threading,
                            "--output-dir", out_dir.to_str().unwrap()]);
    assert!(out.status.success(), "[{mode}+{threading}] build failed:\n{}", text(&out.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"]).arg(out_dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .env("CARGO_TARGET_DIR", Path::new(env!("CARGO_TARGET_TMPDIR")).join("implicit_generics_imports_target").join(dir.file_name().unwrap()))
        .env("CARGO_INCREMENTAL", "0")
        .output().unwrap();
    assert!(run.status.success(), "[{mode}+{threading}] cargo run failed:\n{}", text(&run.stderr));
    text(&run.stdout)
}

const UTIL: &str = "def label(y):\n    print \"item {y}\"\n\ndef grow(n):\n    return n + 1\n\ndef twice(n):\n    return n + n\n";
const MAIN: &str = "use util\n\ndef show(x):\n    label(x)\n\ndef fold(x):\n    _ = twice(x)\n    print \"folded\"\n\ndef main():\n    show(5)\n    show(\"s\")\n    fold(2)\n    fold(1.5)\n";

#[test]
fn bounds_flow_from_a_sibling_file_in_every_config() {
    let dir = project("sibling", &[("util.br", UTIL), ("main.br", MAIN)]);
    for (mode, threading) in [("strict", "multi"), ("managed", "single")] {
        assert_eq!(cargo_run(&dir, mode, threading), "item 5\nitem s\nfolded\nfolded\n", "[{mode}+{threading}]");
    }
}

#[test]
fn bounds_flow_from_a_pub_function_of_a_selective_use() {
    let util = UTIL.replace("def label", "pub def label");
    let main = "use util.label\n\ndef show(x):\n    label(x)\n\ndef main():\n    show(5)\n";
    let dir = project("qualified", &[("util.br", &util), ("main.br", main)]);
    assert_eq!(cargo_run(&dir, "strict", "multi"), "item 5\n");
}
