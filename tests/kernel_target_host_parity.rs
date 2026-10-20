// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression tests: `--target simd|wgpu` must build the same host code the std target does.
//
//  1. The generated Cargo.toml carries the `unicode-*` crates the grapheme / display-width string
//     methods lower to, and `boring.toml`'s `[dependencies]` (both were missing).
//  2. Two same-named free functions (overloads, e.g. `describe(View)` / `describe(Command)` once
//     merged into one program) used to leave the call's result typed by whichever overload was
//     registered last, so `let d = describe(3)` then `d.props["k"] else ("?")` was emitted as an
//     array index (`d.props.get(("k") as usize)`) instead of a dict lookup.
//
// Run with:
//   cargo test --test kernel_target_host_parity

use std::{fs, path::PathBuf, process::Command};

const KERNEL: &str = r#"
mut [float32]'kernel'unified x = [0.0 as float32 for ..<8]
kernel Inc:
    mut [float32]'unified x
    init([float32]'unified a):
        x = a
    def ():
        let i = .thread.x
        x[i] = x[i] + 1.0
mut k = Inc(x)
kernel:
    k(block = 8, grid = 1)
"#;

const HOST: &str = r#"
struct Desc:
    string kind
    {string=string} props = {=}

struct Cmd:
    int x

Desc describe(int v):
    Desc(kind = "k")

string describe(Cmd c):
    "cmd"

def main():
    let s = "héllo"
    print "{s.graphemeCount} {s.displayWidth}"
    let d = describe(3)
    print "{d.props["what"] else ("?")}"
"#;

fn build(target: &str, name: &str, with_toml: bool) -> String {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("kernel_parity_{name}_{target}"));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("main.br"), format!("{HOST}\n{KERNEL}")).unwrap();
    if with_toml {
        fs::write(
            root.join("boring.toml"),
            "[project]\nname = \"parity\"\nversion = \"0.1.0\"\n\n[dependencies]\nfastrand = \"2\"\n",
        )
        .unwrap();
    }
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target])
        .arg(root.join("main.br"))
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let project = root.join(format!("main_{target}"));
    let cargo = fs::read_to_string(project.join("Cargo.toml")).unwrap();
    let main = fs::read_to_string(project.join("src/main.rs")).unwrap();
    assert!(
        main.contains("unicode_segmentation::"),
        "test program should lower graphemeCount to unicode_segmentation"
    );
    assert!(
        !main.contains(".get((\"what\") as usize)"),
        "overloaded call result mis-typed: dict index emitted as array index"
    );
    assert!(main.contains(".get(\"what\")"), "dict lookup not emitted");
    cargo
}

#[test]
fn simd_manifest_has_unicode_and_boring_toml_dependencies() {
    let cargo = build("simd", "toml", true);
    assert!(cargo.contains("unicode-segmentation"), "{cargo}");
    assert!(cargo.contains("unicode-width"), "{cargo}");
    assert!(cargo.contains("fastrand = \"2\""), "{cargo}");
    assert_eq!(cargo.matches("[dependencies]").count(), 1, "{cargo}");
}

#[test]
fn wgpu_manifest_has_unicode_and_boring_toml_dependencies() {
    let cargo = build("wgpu", "toml", true);
    assert!(cargo.contains("unicode-segmentation"), "{cargo}");
    assert!(cargo.contains("unicode-width"), "{cargo}");
    assert!(cargo.contains("fastrand = \"2\""), "{cargo}");
    assert_eq!(cargo.matches("[dependencies]").count(), 1, "{cargo}");
}

#[test]
fn simd_manifest_has_unicode_without_boring_toml() {
    let cargo = build("simd", "notoml", false);
    assert!(cargo.contains("unicode-segmentation"), "{cargo}");
    assert!(!cargo.contains("fastrand"), "{cargo}");
}
