//! Diagnostics for binary fixed-point types (docs/fixed-point-types.md): invalid
//! names, out-of-range literals, strict mixing, `mut` on a scalar. The positive
//! behaviour lives in `tests/cases/fixed_point_*.br` (shared by all builds).

use std::{fs, path::PathBuf, process::Command};

fn run(name: &str, src: &str, subcommand: &[&str]) -> (bool, String) {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fixed_point_errors").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let file = dir.join("main.br");
    fs::write(&file, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(subcommand).arg(&file).current_dir(&dir).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

fn fails_run(name: &str, src: &str, needle: &str) {
    let (ok, text) = run(name, src, &["run"]);
    assert!(!ok && text.contains(needle), "{name}: expected `{needle}`, got: {text}");
}

#[test]
fn out_of_range_literal() {
    fails_run("range", "let fixed8_8 w = 200.5\nprint \"{w}\"\n", "literal out of range for fixed8_8");
}

#[test]
fn different_formats_do_not_mix() {
    fails_run("mix", "let fixed16_16 a = 1.5\nlet fixed8_24 b = 0.5\nprint \"{a + b}\"\n",
        "cannot mix fixed16_16 and fixed8_24");
}

#[test]
fn fixed_does_not_mix_with_int_or_float_variables() {
    fails_run("int", "let fixed16_16 a = 1.5\nlet n = 3\nprint \"{a * n}\"\n", "does not mix");
    fails_run("float", "let fixed16_16 a = 1.5\nlet float64 f = 2.0\nlet fixed16_16 c = f\nprint \"{c}\"\n",
        "convert explicitly with `as fixed16_16`");
}

#[test]
fn invalid_format_names_are_rejected() {
    fails_run("width", "let fixed10_10 a = 1.5\n", "must be one of 8, 16, 32, 64 or 128");
    fails_run("sign", "let fixed0_16 a = 0.5\n", "at least 1 integer bit");
}

#[test]
fn mut_on_a_fixed_scalar_is_an_error() {
    fails_run("mut", "mut fixed16_16 x = 1.5\nprint \"{x}\"\n", "cannot mark a scalar as `mut`");
}

#[test]
fn overflow_and_division_by_zero() {
    fails_run("ovf", "let fixed16_16 a = 32767.0\nprint \"{a * 2}\"\n", "fixed-point overflow");
    fails_run("div0", "let fixed16_16 a = 1.5\nprint \"{a / 0}\"\n", "division by zero");
}

#[test]
fn build_rejects_128_bit_formats_cleanly() {
    let (ok, text) = run("w128", "let fixed64_64 a = 1.5\nprint \"{a}\"\n", &["build", "--emit-rust"]);
    assert!(!ok && text.contains("128-bit fixed-point formats"), "{text}");
}
