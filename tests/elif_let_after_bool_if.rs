// Integration test for `elif let` after a boolean `if` condition,
// verifying both the interpreter (`boring`) and the Rust transpiler (`boring build --emit-rust`).

use std::path::Path;
use std::process::Command;

const EXPECTED_STDOUT: &str = "\
zero_mode
has_val: 42
empty
zero
neg
big: 20
small: 5
ninety_nine
other
zero
val: 7
none
zero
val: 9
none";

fn cargo_run_generated(dir: &Path, crate_name: &str, generated: &str) -> String {
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");
    std::fs::write(dir.join("src/main.rs"), generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
            crate_name
        ),
    )
    .expect("failed to write Cargo.toml");

    let out = Command::new("cargo")
        .arg("run")
        .arg("--quiet")
        .current_dir(dir)
        .output()
        .unwrap_or_else(|e| panic!("failed to run cargo: {}", e));

    assert!(
        out.status.success(),
        "generated Rust failed to compile/run:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n").trim_end().to_string();
    let _ = std::fs::remove_dir_all(dir);
    stdout
}

#[test]
fn elif_let_after_bool_if_interpreter() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let out = Command::new(bin)
        .arg("tests/elif_let_after_bool_if.br")
        .output()
        .unwrap_or_else(|e| panic!("failed to run boring: {}", e));

    assert!(
        out.status.success(),
        "interpreter exited with error:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let actual = String::from_utf8_lossy(&out.stdout).replace("\r\n", "\n");
    assert_eq!(actual.trim_end(), EXPECTED_STDOUT);
}

#[test]
fn elif_let_after_bool_if_transpiler() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let out = Command::new(bin)
        .arg("build")
        .arg("tests/elif_let_after_bool_if.br")
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to run boring build: {}", e));

    assert!(
        out.status.success(),
        "boring build --emit-rust failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let generated = String::from_utf8_lossy(&out.stdout);
    let scratch = std::env::temp_dir().join("boring_elif_let_after_bool_if_test");
    let _ = std::fs::remove_dir_all(&scratch);
    let actual = cargo_run_generated(&scratch, "elif_let_after_bool_if_test", &generated);
    assert_eq!(actual, EXPECTED_STDOUT);
}
