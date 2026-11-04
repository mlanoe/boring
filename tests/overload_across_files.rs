// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Overloads of one free function / one type's methods (struct body and `ext` blocks) declared in
// DIFFERENT files of a project resolve by argument type, like same-file ones, under `boring run`
// and `boring build`. Before, the entry file's own declaration hid the imported same-named one, so
// `describe(1)` failed with "argument 1 of describe (parameter 's') expects string, got int".
// An overload pair that is ambiguous (same parameter types, or a default making a shorter call
// match both) stays an error in both modes, never a silent pick of one file's version.
//
// Run with:
//   tools/test.sh --test overload_across_files

mod common;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("overload_across_files").join(name);
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

/// `boring run main.br`: stdout on success.
fn run_ok(dir: &Path) -> String {
    let out = boring(dir, &["run", "main.br"]);
    assert!(out.status.success(), "run failed:\n{}", text(&out.stderr));
    text(&out.stdout)
}

/// `boring build main.br`, then compile and run the generated project: stdout.
fn build_ok(dir: &Path, name: &str) -> String {
    let out = boring(dir, &["build", "main.br"]);
    assert!(out.status.success(), "build failed:\n{}", text(&out.stderr));
    // Every generated package is called `main`: give each its own name so the shared target
    // directory never runs another test's executable.
    let manifest = dir.join("main_rust/Cargo.toml");
    let toml = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(&manifest, toml.replacen("name = \"main\"", &format!("name = \"ovl_{name}\""), 1)).unwrap();
    let out = common::SharedCargo::new("overload_across_files")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("main_rust/Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "{name}: generated project failed:\n{}", text(&out.stderr));
    text(&out.stdout)
}

/// Both `run` and `build` must reject the project with a message containing `needle`.
fn assert_rejected(dir: &Path, needle: &str) {
    for args in [&["run", "main.br"][..], &["build", "main.br"][..]] {
        let out = boring(dir, args);
        assert!(!out.status.success(), "{args:?} accepted an ambiguous overload:\n{}", text(&out.stdout));
        let err = text(&out.stderr);
        assert!(err.contains(needle), "{args:?}: expected `{needle}` in:\n{err}");
    }
}

const A_INT: &str = "string describe(int n):\n    \"int {n}\"\n";
const MAIN_STR: &str = "use a\n\nstring describe(string s):\n    \"str {s}\"\n\n\
                        def main():\n    print describe(1)\n    print describe(\"x\")\n";

#[test]
fn free_fn_overloads_in_imported_and_entry_file() {
    let dir = project("free", &[("a.br", A_INT), ("main.br", MAIN_STR)]);
    assert_eq!(run_ok(&dir), "int 1\nstr x\n");
    assert_eq!(build_ok(&dir, "free"), "int 1\nstr x\n");
}

#[test]
fn free_fn_overloads_in_three_files() {
    let b = "use a\n\nstring describe(string s):\n    \"str {s}\"\n";
    let main = "use a\nuse b\n\nstring describe(float f):\n    \"float {f}\"\n\n\
                def main():\n    print describe(1)\n    print describe(\"x\")\n    print describe(2.5)\n";
    let dir = project("three", &[("a.br", A_INT), ("b.br", b), ("main.br", main)]);
    assert_eq!(run_ok(&dir), "int 1\nstr x\nfloat 2.5\n");
    assert_eq!(build_ok(&dir, "three"), "int 1\nstr x\nfloat 2.5\n");
}

#[test]
fn free_fn_overloads_in_two_imported_files() {
    let b = "string describe(string s):\n    \"str {s}\"\n";
    let main = "use a\nuse b\n\ndef main():\n    print describe(1)\n    print describe(\"x\")\n";
    let dir = project("imported_only", &[("a.br", A_INT), ("b.br", b), ("main.br", main)]);
    assert_eq!(run_ok(&dir), "int 1\nstr x\n");
    assert_eq!(build_ok(&dir, "imported_only"), "int 1\nstr x\n");
}

const A_BOX: &str = "struct Box:\n    var int v = 0\n    req string show(int n):\n        \"box-int {n}\"\n";
const MAIN_EXT: &str = "use a\n\next Box:\n    req string show(string s):\n        \"box-str {s}\"\n\n\
                        def main():\n    let b = Box(1)\n    print b.show(1)\n    print b.show(\"x\")\n";

#[test]
fn ext_method_overload_of_an_imported_struct_method() {
    let dir = project("ext", &[("a.br", A_BOX), ("main.br", MAIN_EXT)]);
    assert_eq!(run_ok(&dir), "box-int 1\nbox-str x\n");
    assert_eq!(build_ok(&dir, "ext"), "box-int 1\nbox-str x\n");
}

#[test]
fn ext_methods_overloaded_across_two_imported_files() {
    let a = "struct Box:\n    var int v = 0\n";
    let b = "use a\n\next Box:\n    req string show(int n):\n        \"box-int {n}\"\n";
    let c = "use a\n\next Box:\n    req string show(string s):\n        \"box-str {s}\"\n";
    let main = "use a\nuse b\nuse c\n\ndef main():\n    let x = Box(1)\n    print x.show(1)\n    print x.show(\"x\")\n";
    let dir = project("ext_two", &[("a.br", a), ("b.br", b), ("c.br", c), ("main.br", main)]);
    assert_eq!(run_ok(&dir), "box-int 1\nbox-str x\n");
    assert_eq!(build_ok(&dir, "ext_two"), "box-int 1\nbox-str x\n");
}

#[test]
fn same_file_struct_and_ext_overload_keeps_both_methods() {
    let main = "struct Box:\n    var int v = 0\n    req string show(int n):\n        \"box-int {n}\"\n\n\
                ext Box:\n    req string show(string s):\n        \"box-str {s}\"\n\n\
                def main():\n    let b = Box(1)\n    print b.show(1)\n    print b.show(\"x\")\n";
    let dir = project("same_file_ext", &[("main.br", main)]);
    assert_eq!(run_ok(&dir), "box-int 1\nbox-str x\n");
    assert_eq!(build_ok(&dir, "same_file_ext"), "box-int 1\nbox-str x\n");
}

#[test]
fn identical_signature_across_files_is_an_error() {
    let main = "use a\n\nstring describe(int n):\n    \"main {n}\"\n\ndef main():\n    print describe(1)\n";
    let dir = project("dup_entry", &[("a.br", A_INT), ("main.br", main)]);
    assert_rejected(&dir, "ambiguous overload for 'describe'");
}

#[test]
fn identical_signature_in_two_imported_files_is_an_error() {
    let b = "string describe(int m):\n    \"b {m}\"\n";
    let main = "use a\nuse b\n\ndef main():\n    print describe(1)\n";
    let dir = project("dup_imported", &[("a.br", A_INT), ("b.br", b), ("main.br", main)]);
    assert_rejected(&dir, "ambiguous overload for 'describe'");
}

#[test]
fn default_parameter_making_a_call_ambiguous_across_files_is_an_error() {
    let a = "string describe(int n, string s = \"d\"):\n    \"a {n}{s}\"\n";
    let main = "use a\n\nstring describe(int n):\n    \"main {n}\"\n\ndef main():\n    print describe(1)\n";
    let dir = project("dup_default", &[("a.br", a), ("main.br", main)]);
    assert_rejected(&dir, "ambiguous overload for 'describe'");
}

#[test]
fn identical_ext_signature_across_files_is_an_error() {
    let main = "use a\n\next Box:\n    req string show(int n):\n        \"ext {n}\"\n\n\
                def main():\n    let b = Box(1)\n    print b.show(1)\n";
    let dir = project("dup_ext", &[("a.br", A_BOX), ("main.br", main)]);
    assert_rejected(&dir, "ambiguous overload for method 'Box::show'");
}

#[test]
fn overloads_across_a_deps_project() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("overload_across_files").join("deps");
    let _ = std::fs::remove_dir_all(&dir);
    let app_src = dir.join("app/src");
    let lib_src = dir.join("lib/src");
    std::fs::create_dir_all(&app_src).unwrap();
    std::fs::create_dir_all(&lib_src).unwrap();
    std::fs::write(dir.join("app/boring.toml"),
        "[project]\nname = \"app\"\nversion = \"0.1.0\"\nmain = \"src/main.br\"\n\n[deps]\nlib = \"../lib\"\n").unwrap();
    // Items of another project must be `pub`.
    std::fs::write(lib_src.join("a.br"), format!("pub {A_INT}")).unwrap();
    let main = "use lib.a.*\n\nstring describe(string s):\n    \"str {s}\"\n\n\
                def main():\n    print describe(1)\n    print describe(\"x\")\n";
    std::fs::write(app_src.join("main.br"), main).unwrap();
    let app = dir.join("app");
    let main_br = app_src.join("main.br");
    let main_br = main_br.to_str().unwrap();
    let out = boring(&app, &["run", main_br]);
    assert!(out.status.success(), "run failed:\n{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), "int 1\nstr x\n");
    let out = boring(&app, &["build", main_br, "--emit-rust"]);
    assert!(out.status.success(), "build failed:\n{}", text(&out.stderr));
    let rs = text(&out.stdout);
    assert!(rs.contains("fn describe__int(") && rs.contains("fn describe__string("), "{rs}");
}

/// A trait whose methods all have defaults has no required method, so every struct satisfies it
/// structurally. An overload set `f(Shape)` + `f(Point)` must still pick `f(Point)` for a `Point`
/// (and the trait variant for a real implementor), not whichever variant was bound first.
#[test]
fn trait_overload_does_not_swallow_unrelated_struct() {
    let dir = project("trait_overload", &[
        ("shape.br", "pub trait Shape:\n    req string kind(): \"shape\"\n\npub struct Circle as Shape:\n    float r\n\npub string describe(Shape s):\n    \"shape \" + s.kind()\n"),
        ("point.br", "pub struct Point:\n    int x\n\npub string describe(Point p):\n    \"point {p.x}\"\n"),
        ("main.br", "use shape\nuse point\n\nprint describe(Circle(r = 1.0))\nprint describe(Point(x = 3))\n"),
    ]);
    assert_eq!(run_ok(&dir), "shape shape\npoint 3\n");
}
