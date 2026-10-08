// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: two sibling `.br` modules of one project that both `use` the same
// external Rust item (`use std.fs.File`) used to emit `use std::fs::File;` once per
// module, and since every module of a project lands in ONE Rust namespace (`include!`d
// into `src/main.rs` for the std target, concatenated into a single `main.rs` for the
// metal/wgpu/... targets and for `--emit-rust`), the generated project failed with
// rustc E0252 ("the name `File` is defined multiple times"). `Transpiler::emit_use`
// now dedupes external imports on their fully-qualified name (see
// `emitted_external_uses` in src/transpiler/mod.rs).
//
// Run with:
//   cargo test --test duplicate_external_use_sibling_modules

use std::path::{Path, PathBuf};
use std::process::Command;

const MODULE_A: &str = "\
use std.fs.File
use std.os.unix.fs.FileExt

pub def open_a(string path) throws:
    let f = try? File.open(path)
    guard let f else throw \"cannot open\"
    print \"a ok\"
";

const MODULE_B: &str = "\
use std.fs.File
use std.os.unix.fs.FileExt

pub def open_b(string path) throws:
    let f = try? File.open(path)
    guard let f else throw \"cannot open\"
    print \"b ok\"
";

const MAIN: &str = "\
use a
use b

def main() throws:
    open_a(\"/dev/null\")
    open_b(\"/dev/null\")
";

/// Fresh project directory under cargo's per-target scratch dir, seeded with `files`.
fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("failed to create project dir");
    for (file, src) in files {
        std::fs::write(dir.join(file), src).expect("failed to write fixture");
    }
    dir
}

fn boring(dir: &Path, args: &[&str]) -> std::process::Output {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {e}"));
    assert!(
        out.status.success(),
        "boring {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

/// Compiles `rs` standalone with rustc (no cargo, no dependencies) and returns the
/// binary's stdout. `include!`s inside `rs` resolve relative to it.
fn rustc_and_run(rs: &Path, out_dir: &Path) -> String {
    let bin = out_dir.join("gen_bin");
    let rustc = Command::new("rustc")
        .args(["--edition", "2021"])
        .arg(rs)
        .arg("-o").arg(&bin)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke rustc: {e}"));
    assert!(
        rustc.status.success(),
        "rustc failed on {} (duplicate `use` across sibling modules?):\n{}",
        rs.display(),
        String::from_utf8_lossy(&rustc.stderr)
    );
    let run = Command::new(&bin).output().expect("failed to run compiled binary");
    assert!(run.status.success(), "binary failed:\n{}", String::from_utf8_lossy(&run.stderr));
    String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n")
}

fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

/// std target: `src/a.rs` + `src/b.rs` `include!`d into `src/main.rs`.
#[test]
fn std_target_sibling_modules_share_external_use() {
    let dir = project("dup_use_std", &[("a.br", MODULE_A), ("b.br", MODULE_B), ("main.br", MAIN)]);
    boring(&dir, &["build", "main.br"]);

    let out = rustc_and_run(&dir.join("main_rust/src/main.rs"), &dir);
    assert_eq!(out, "a ok\nb ok\n");
}

/// `--emit-rust`: modules are inlined into the single stdout stream.
#[test]
fn emit_rust_sibling_modules_share_external_use() {
    let dir = project("dup_use_emit_rust", &[("a.br", MODULE_A), ("b.br", MODULE_B), ("main.br", MAIN)]);
    let emitted = boring(&dir, &["build", "main.br", "--emit-rust"]).stdout;
    let rs = dir.join("gen.rs");
    std::fs::write(&rs, &emitted).unwrap();

    assert_eq!(count(&String::from_utf8_lossy(&emitted), "use std::fs::File;"), 1);
    assert_eq!(rustc_and_run(&rs, &dir), "a ok\nb ok\n");
}

/// GPU targets flatten every module into one `main.rs`; each import must appear once.
/// Compiling that file needs the target's GPU crates, so assert on the generated text
/// (the E0252 is purely a function of the duplicated line).
#[test]
fn gpu_targets_flattened_main_has_each_external_use_once() {
    for (target, out_dir) in [("metal", "main_metal"), ("wgpu", "main_wgpu")] {
        let dir = project(
            &format!("dup_use_{target}"),
            &[("a.br", MODULE_A), ("b.br", MODULE_B), ("main.br", MAIN)],
        );
        boring(&dir, &["build", "main.br", "--target", target]);
        let main_rs = std::fs::read_to_string(dir.join(out_dir).join("src/main.rs")).unwrap();
        assert_eq!(count(&main_rs, "use std::fs::File;"), 1, "{target}:\n{main_rs}");
        assert_eq!(count(&main_rs, "use std::os::unix::fs::FileExt;"), 1, "{target}:\n{main_rs}");
    }
}

/// Item lists overlapping only partially: the later import must keep what is new
/// (`Write`) and drop what was already imported (`Read`), and an entirely redundant
/// list must emit nothing.
#[test]
fn partially_overlapping_item_lists_import_only_new_names() {
    let a = "\
use std.io.Read

pub def a_fn():
    print \"a\"
";
    let b = "\
use std.io.Read, Write

pub def b_fn():
    print \"b\"
";
    let c = "\
use std.io.Read, Write

pub def c_fn():
    print \"c\"
";
    let main = "\
use a
use b
use c

def main():
    a_fn()
    b_fn()
    c_fn()
";
    let dir = project("dup_use_partial", &[("a.br", a), ("b.br", b), ("c.br", c), ("main.br", main)]);
    let emitted = boring(&dir, &["build", "main.br", "--emit-rust"]).stdout;
    let text = String::from_utf8_lossy(&emitted);

    assert_eq!(count(&text, "use std::io::Read;"), 1, "{text}");
    assert_eq!(count(&text, "use std::io::Write;"), 1, "{text}");
    assert_eq!(count(&text, "use std::io::{Read, Write};"), 0, "{text}");

    let rs = dir.join("gen.rs");
    std::fs::write(&rs, &emitted).unwrap();
    assert_eq!(rustc_and_run(&rs, &dir), "a\nb\nc\n");
}
