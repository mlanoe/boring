// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a Boring `string` (`Rc<str>`/`Arc<str>`) handed to a `std::fs` API
// bounded on `AsRef<Path>` was only lowered to `&str` in one shape — a `string`
// *parameter* passed to an associated function (`File::open((&*path))`). Equivalent
// programs failed rustc with E0277 ("the trait bound `Arc<str>: AsRef<Path>` is not
// satisfied"):
//   * a local bound to a call returning `string` (`let p = name_of(1)`), because the
//     binding was never tracked as a string, so `File::create(p)` passed it by value;
//   * a free function imported with `use std.fs.create_dir_all`, which has no Boring
//     declaration and so went through the generic `Arc::clone(&dir)` argument path.
// Every string argument is now lowered to `&*s` whatever the argument's shape and
// whatever the callee's (associated function or imported free function).
//
// Run with:
//   cargo test --test external_fs_path_args

use std::path::{Path, PathBuf};
use std::process::Command;

fn project(name: &str, src: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("failed to create project dir");
    std::fs::write(dir.join("main.br"), src).expect("failed to write fixture");
    dir
}

fn boring_emit_rust(dir: &Path) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .current_dir(dir)
        .args(["build", "main.br", "--emit-rust"])
        .output()
        .expect("failed to invoke boring");
    assert!(out.status.success(), "boring build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Compiles `rs` standalone with rustc and returns the binary's stdout.
fn rustc_and_run(rs: &Path, out_dir: &Path) -> String {
    let bin = out_dir.join("gen_bin");
    let rustc = Command::new("rustc")
        .args(["--edition", "2021"])
        .arg(rs)
        .arg("-o").arg(&bin)
        .output()
        .expect("failed to invoke rustc");
    assert!(
        rustc.status.success(),
        "rustc failed on {} (string passed by value to an AsRef<Path> API?):\n{}",
        rs.display(),
        String::from_utf8_lossy(&rustc.stderr)
    );
    let run = Command::new(&bin).output().expect("failed to run compiled binary");
    assert!(run.status.success(), "binary failed:\n{}", String::from_utf8_lossy(&run.stderr));
    String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n")
}

fn build_and_run(name: &str, src: &str) -> (String, String) {
    let dir = project(name, src);
    let rust = boring_emit_rust(&dir);
    let rs = dir.join("gen.rs");
    std::fs::write(&rs, &rust).unwrap();
    let out = rustc_and_run(&rs, &dir);
    (rust, out)
}

/// The exact shapes from the bug report: parameter (already worked), local bound to a
/// `string`-returning call (File::create), and an imported free function (create_dir_all).
#[test]
fn string_args_to_fs_apis_are_lowered_to_str_in_every_shape() {
    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("fs_path_args_shapes_data");
    let _ = std::fs::remove_dir_all(&base);
    let base = base.to_string_lossy().replace('\\', "/");
    let src = format!("\
use std.fs.File
use std.fs.create_dir_all

string name_of(int i):
    \"{base}/rep1_{{i}}.bin\"

def make_file() throws:
    let path = name_of(1)
    let f = try? File.create(path)
    guard let f else throw \"cannot create\"
    print \"created\"

def make_dir(string dir) throws:
    let made = try? create_dir_all(dir)
    guard let made else throw \"cannot mkdir\"
    print \"mkdir\"

def open_it(string path) throws:
    let f = try? File.open(path)
    guard let f else throw \"cannot open\"
    print \"opened\"

def main() throws:
    make_dir(\"{base}\")
    make_file()
    open_it(\"{base}/rep1_1.bin\")
");
    let (rust, out) = build_and_run("fs_path_args_shapes", &src);
    assert_eq!(out, "mkdir\ncreated\nopened\n", "{rust}");
    assert!(rust.contains("File::create((&*path))"), "{rust}");
    assert!(rust.contains("create_dir_all((&*dir))"), "{rust}");
    assert!(!rust.contains("create_dir_all(Arc::clone"), "{rust}");
    assert!(!rust.contains("create_dir_all(Rc::clone"), "{rust}");
}

/// Every flavour of string expression, against associated functions and free functions
/// alike (`remove_file`, `rename`, `read_dir`, `File::open/create`).
#[test]
fn string_expression_flavours_and_callee_flavours() {
    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("fs_path_args_flavours_data");
    let _ = std::fs::remove_dir_all(&base);
    let base = base.to_string_lossy().replace('\\', "/");
    let src = format!("\
use std.fs.File
use std.fs.create_dir_all
use std.fs.remove_file
use std.fs.rename
use std.fs.read_dir

struct Cfg:
    string root

string joined(string dir, string leaf):
    dir + \"/\" + leaf

string tmp_name(string dir, int i):
    \"{{dir}}/f{{i}}.tmp\"

def run(Cfg cfg, string dir) throws:
    # free function, string parameter
    let a = try? create_dir_all(dir)
    guard let a else throw \"mkdir\"
    # associated function, local inferred from a call to a string-returning function
    let one = tmp_name(dir, 1)
    let f1 = try? File.create(one)
    guard let f1 else throw \"create one\"
    # local inferred from interpolation and from concatenation
    let two = \"{{dir}}/f2.tmp\"
    let f2 = try? File.create(two)
    guard let f2 else throw \"create two\"
    let three = dir + \"/f3.tmp\"
    let f3 = try? File.create(three)
    guard let f3 else throw \"create three\"
    # call result passed directly, nested call result, struct field
    let f4 = try? File.create(joined(dir, \"f4.tmp\"))
    guard let f4 else throw \"create four\"
    let f5 = try? File.open(tmp_name(dir, 1))
    guard let f5 else throw \"open one\"
    let r0 = try? read_dir(cfg.root)
    guard let r0 else throw \"read_dir root\"
    # free functions with locals and call results
    let moved = tmp_name(dir, 9)
    let r1 = try? rename(one, moved)
    guard let r1 else throw \"rename\"
    let r2 = try? remove_file(moved)
    guard let r2 else throw \"remove moved\"
    let r3 = try? remove_file(joined(dir, \"f4.tmp\"))
    guard let r3 else throw \"remove four\"
    let r4 = try? remove_file(two)
    guard let r4 else throw \"remove two\"
    let r5 = try? remove_file(three)
    guard let r5 else throw \"remove three\"
    print \"done\"

def main() throws:
    run(Cfg(\"{base}\"), \"{base}\")
");
    let (rust, out) = build_and_run("fs_path_args_flavours", &src);
    assert_eq!(out, "done\n", "{rust}");
}

/// The free-function lowering only applies to functions Boring has no declaration for: a
/// Boring-declared function with a `string` parameter still receives the owned string.
#[test]
fn declared_boring_functions_keep_owned_string_arguments() {
    let src = "\
string shout(string s):
    s + \"!\"

def main() throws:
    let w = \"hi\"
    let x = shout(w)
    print x
";
    let (rust, out) = build_and_run("fs_path_args_declared_untouched", src);
    assert_eq!(out, "hi!\n", "{rust}");
    assert!(!rust.contains("shout((&*"), "{rust}");
}
