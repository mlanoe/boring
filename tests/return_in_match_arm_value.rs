// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: an early `return` nested in an `if` inside a block-bodied
// `match` arm that is used as a value (`let x = match ...`) must exit the
// enclosing function. The interpreter's `eval_block_as_expr` used to absorb
// `Signal::Return` into the arm's value, so `bias` was bound to the returned
// string and the next statement failed with `cannot compare String and Float`.
// Only function/method/closure bodies (`eval_body_as_expr`) are return
// boundaries. The transpiler already emitted correct Rust for this shape, so
// the same program is also checked to pass `boring build --emit-rust`.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const SOURCE: &str = r#"
enum V:
    I(int)
    B(bool)
    U

string check(V value):
    let bias = match value:
        I(n): n as float
        B(enabled):
            if enabled: return "boolean may only be false"
            -100.0
        _: 0.0
    if bias < -100.0 or bias > 100.0:
        return "range"
    "ok"

def main():
    print check(V.I(5))
    print check(V.B(false))
    print check(V.B(true))
    print check(V.U)
"#;

fn write_fixture(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("return_in_match_arm_value").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let source = dir.join("test.br");
    fs::write(&source, SOURCE).unwrap();
    source
}

#[test]
fn run_return_in_match_arm_exits_function() {
    let source = write_fixture("run");
    let output = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&source).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "ok\nok\nboolean may only be false\nok"
    );
}

#[test]
fn emit_rust_accepts_return_in_match_arm() {
    let source = write_fixture("emit");
    let output = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--emit-rust"]).arg(&source).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let rust = String::from_utf8_lossy(&output.stdout);
    assert!(rust.contains("return Arc::<str>::from(\"boolean may only be false\")"), "{rust}");
}
