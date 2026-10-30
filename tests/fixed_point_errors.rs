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
fn unsigned_128_bit_formats_are_rejected_with_a_reason() {
    fails_run("u128", "let ufixed64_64 a = 1.5\nprint \"{a}\"\n", "unsigned 128-bit fixed-point formats are not supported");
    fails_run("frac127", "let fixed1_127 a = 0.5\nprint \"{a}\"\n", "at most 126 fractional bits");
}

#[test]
fn wide_math_and_overflow_errors() {
    fails_run("wmath", "let fixed64_64 a = 4.0\nprint \"{a.sqrt()}\"\n", "not supported for 128-bit fixed-point formats");
    fails_run("wovf", "let fixed64_64 a = fixed64_64.max\nprint \"{a * 2.0}\"\n", "fixed-point overflow");
}

#[test]
fn math_domain_and_overflow_errors() {
    fails_run("sqrtneg", "let fixed16_16 a = -1.0\nprint \"{a.sqrt()}\"\n", "square root of a negative value");
    fails_run("ln0", "let fixed16_16 a = 0.0\nprint \"{a.ln()}\"\n", "logarithm of a non-positive value");
    fails_run("expovf", "let fixed16_16 a = 20.0\nprint \"{a.exp()}\"\n", "fixed-point overflow in fixed16_16.exp()");
    fails_run("lnneg_unsigned", "let ufixed8_8 a = 0.5\nprint \"{a.ln()}\"\n", "fixed-point overflow in ufixed8_8.ln()");
}

#[test]
fn float_and_int_variables_need_as_at_every_binding_site() {
    let hint = "convert explicitly with `as fixed16_16`";
    let head = "struct P:\n    var fixed16_16 v = 0.0\n    def set(fixed16_16 x):\n        v = x\n";
    fails_run("arg", &format!("fixed16_16 id(fixed16_16 a):\n    a\nlet float64 f = 1.5\nprint \"{{id(f)}}\"\n"), hint);
    fails_run("ctor", &format!("{head}let float64 f = 1.5\nprint \"{{P(v = f).v}}\"\n"), hint);
    fails_run("method", &format!("{head}mut p = P()\nlet int n = 2\np.set(n)\n"), hint);
    fails_run("field", &format!("{head}mut p = P()\nlet float64 f = 1.5\np.v = f\n"), hint);
    fails_run("reassign", "var fixed16_16 g = 0.0\nlet float64 f = 1.5\ng = f\n", hint);
    fails_run("return", "fixed16_16 r(float64 f):\n    f\nprint \"{r(1.5)}\"\n", hint);
}

#[test]
fn literals_still_adopt_the_fixed_format_everywhere() {
    let src = "struct P:\n    var fixed16_16 v = 0.0\n    def set(fixed16_16 x):\n        v = x\n\
fixed16_16 a(fixed16_16 x):\n    1.5\nfixed16_16 b(bool c):\n    if c:\n        return 2\n    0.25\n\
mut p = P(v = 0.5)\np.set(2)\np.v = 3.25\nvar fixed16_16 g = 0.0\ng = -1.5\n\
print \"{p.v} {a(1)} {b(true)} {b(false)} {g}\"\n";
    let (ok, text) = run("literals_ok", src, &["run"]);
    assert!(ok && text.trim() == "3.25 1.5 2 0.25 -1.5", "got: {text}");
}

#[test]
fn fixed_type_names_cannot_be_declared() {
    let needle = "reserved fixed-point type name";
    fails_run("struct_name", "struct u8_8:\n    int x\n", needle);
    fails_run("enum_name", "enum fixed16_16:\n    A\n", needle);
    fails_run("trait_name", "trait ufixed8_8:\n    req int f()\n", needle);
    fails_run("type_name", "type i16_16 as int\n", needle);
}
