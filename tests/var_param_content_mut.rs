// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for docs/book.md's "`mut` vs `var` on a struct parameter" table: the
// "rebind caller" and "mutate content" columns are independent permissions. A bare
// `var T p` parameter used to also grant content mutation (`p.bump()`, `p.x = 5`) in both
// `boring run` and `boring build`, because `Param.mutable` is `true` for `var` and `mut`
// alike and the interpreter / transpiler never looked at `rebindable`/`var_mut`.
//
// Run with:
//   cargo test --test var_param_content_mut

use std::path::Path;
use std::process::{Command, Output};

const PRELUDE: &str = "struct Point:\n    var int x = 0\n    def bump(): self.x += 1\n\n";

fn write_case(name: &str, body: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("boring_var_param_content_mut");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let file = dir.join(format!("{name}.br"));
    std::fs::write(&file, format!("{PRELUDE}{body}")).expect("write case");
    file
}

fn run(file: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(file).output().expect("boring run")
}

fn emit(file: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boring")).arg("build").arg(file).arg("--emit-rust").output().expect("boring build")
}

/// Both backends must reject the program, run with the interpreter's wording and build with
/// the transpiler's.
fn assert_rejected(name: &str, body: &str, run_fragment: &str, build_fragment: &str) {
    let file = write_case(name, body);
    let (r, b) = (run(&file), emit(&file));
    assert!(!r.status.success(), "{name}: boring run must fail");
    assert!(!b.status.success(), "{name}: boring build must fail");
    let (re, be) = (String::from_utf8_lossy(&r.stderr), String::from_utf8_lossy(&b.stderr));
    assert!(re.contains(run_fragment), "{name}: expected `{run_fragment}` from run, got:\n{re}");
    assert!(be.contains(build_fragment), "{name}: expected `{build_fragment}` from build, got:\n{be}");
}

#[test]
fn var_param_def_call_is_rejected() {
    assert_rejected(
        "var_def_call",
        "def viaVar(var Point p): p.bump()\n\ndef main():\n    var Point a = Point()\n    viaVar(a)\n",
        "cannot call mutating method 'bump' on non-mut binding 'p'",
        "`p` is not declared `mut` — cannot call `def` method `.bump()` on an immutable binding; fix: declare the parameter as `mut Point p` or `var mut Point p`",
    );
}

#[test]
fn var_param_field_write_is_rejected() {
    assert_rejected(
        "var_field_write",
        "def viaVar(var Point p): p.x = 5\n\ndef main():\n    var Point a = Point()\n    viaVar(a)\n",
        "cannot mutate field 'x' on non-mut binding 'p'",
        "`p` is not declared `mut` — cannot assign to field `.x` on an immutable binding; fix: declare the parameter as `mut Point p` or `var mut Point p`",
    );
}

#[test]
fn var_param_in_struct_method_is_rejected() {
    assert_rejected(
        "var_method_param",
        "struct Holder:\n    var int n = 0\n    def f(var Point p): p.bump()\n\ndef main():\n    var Point a = Point()\n    Holder().f(a)\n",
        "cannot call mutating method 'bump' on non-mut binding 'p'",
        "`p` is not declared `mut` — cannot call `def` method `.bump()` on an immutable binding",
    );
}

#[test]
fn mut_param_cannot_rebind() {
    let file = write_case("mut_rebind", "def m(mut Point p): p = Point(1)\n\ndef main():\n    mut Point a = Point()\n    m(a)\n");
    for out in [run(&file), emit(&file)] {
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("cannot assign to `p`: declared as `mut` (fixed)"));
    }
}

#[test]
fn mut_var_and_var_mut_params_behave_per_the_table() {
    // `mut` mutates the caller's content, a bare `var` rebinds (and reads) the caller's
    // variable, `var mut` does both.
    let body = "\
def mutate(mut Point p):
    p.bump()
    p.x += 10

def reseat(var Point p):
    print p.x
    p = Point(100)

def both(var mut Point p):
    p.bump()
    p = Point(p.x + 1000)

def main():
    var mut Point a = Point()
    mutate(a)
    print a.x
    reseat(a)
    print a.x
    both(a)
    print a.x
";
    let file = write_case("table_ok", body);
    let r = run(&file);
    assert!(r.status.success(), "boring run failed:\n{}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(String::from_utf8_lossy(&r.stdout).trim(), "11\n11\n100\n1101");
    let b = emit(&file);
    assert!(b.status.success(), "boring build failed:\n{}", String::from_utf8_lossy(&b.stderr));
}
