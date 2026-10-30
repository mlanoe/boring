// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The transpiler must never silently drop source code: a construct it cannot translate is a
// compile-time `error:`, never a `/* ... */` placeholder in otherwise-compiling output.
// (`/* complex stmt */`, `/* unsupported ... */`, `/* kernel launch */`, ... — see
// `PLACEHOLDER_MARKERS` in `src/transpiler/helpers.rs`.)
//
// Run with:
//   cargo test --test no_silent_placeholders

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Keep in sync with `PLACEHOLDER_MARKERS` in `src/transpiler/helpers.rs`.
const MARKERS: &[&str] = &[
    "/* complex stmt */", "/* kernel launch */", "/* missing arg */", "/* unknown */",
    "/* unsupported", "/* expr */", "/* op */", "/*op*/", "/* unreachable catch",
    " -- unsupported */",
];

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("no_silent_placeholders").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn build(name: &str, src: &str, args: &[&str]) -> Output {
    let dir = scratch(name);
    let br = dir.join("test.br");
    fs::write(&br, src).unwrap();
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("build").args(args).arg(&br)
        .output().expect("failed to run boring")
}

fn assert_rejected(out: &Output, needle: &str) {
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "expected a transpile error, but the build succeeded:\n{stderr}");
    assert!(stderr.contains("error:") && stderr.contains(needle),
        "expected an `error:` mentioning {needle:?}, got:\n{stderr}");
}

/// Global safety net: no `tests/cases/*.br` program transpiles to output containing a placeholder
/// (programs that fail to build for their own reasons, e.g. `error_*.br`, are skipped — but a
/// build failing *because of* a placeholder is still caught through its stderr).
#[test]
fn no_test_case_emits_a_placeholder() {
    let cases = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("cases");
    let mut offenders = Vec::new();
    let mut checked = 0;
    let mut entries: Vec<_> = fs::read_dir(&cases).unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "br")).collect();
    entries.sort();
    for path in entries {
        let out = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--emit-rust"]).arg(&path)
            .current_dir(&cases)
            .output().expect("failed to run boring");
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        if out.status.success() { checked += 1; }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let is_negative_case = name.starts_with("error_");
        if is_negative_case { continue; }
        if let Some(m) = MARKERS.iter().find(|m| text.contains(**m)) {
            offenders.push(format!("{name}: contains `{m}`"));
        }
        if text.contains("left the placeholder") {
            offenders.push(format!("{name}: placeholder backstop fired"));
        }
    }
    assert!(checked > 50, "only {checked} cases transpiled — the sweep is not exercising the corpus");
    assert!(offenders.is_empty(), "placeholders in transpiler output:\n{}", offenders.join("\n"));
}

// ── Regression cases: each construct below used to leave a placeholder in the output ─────────

const FOR_IN_ARRAY_KERNEL: &str = r#"
kernel Foo:
    let [float32]'global a
    mut [float32]'unified out
    init([float32]'global ai, int n):
        a = ai
        out = [0.0 for ..<n]
    def ():
        var acc = 0.0
        for v in a:
            acc = acc + v
        out[.thread.x] = acc
pub req [float32]'kernel'unified build([float32]'global x, int n) throws:
    mut k = Foo(x, n)
    k.out
"#;

const METHOD_ON_FIELD_KERNEL: &str = r#"
kernel Foo:
    let [float32]'global a
    mut [float32]'unified out
    init([float32]'global ai, int n):
        a = ai
        out = [0.0 for ..<n]
    def ():
        let t = a.first()
        out[.thread.x] = t
pub req [float32]'kernel'unified build([float32]'global x, int n) throws:
    mut k = Foo(x, n)
    k.out
"#;

#[test]
fn gpu_device_for_over_array_is_an_error() {
    for target in ["cuda", "rocm", "metal"] {
        let out = build(&format!("for_in_{target}"), FOR_IN_ARRAY_KERNEL, &["--target", target]);
        assert_rejected(&out, "only supports a range");
    }
}

#[test]
fn gpu_device_method_on_non_self_receiver_is_an_error() {
    for target in ["cuda", "rocm", "metal"] {
        let out = build(&format!("method_{target}"), METHOD_ON_FIELD_KERNEL, &["--target", target]);
        assert_rejected(&out, "is not supported in");
    }
}

#[test]
fn linux_kernel_closure_with_for_loop_is_an_error() {
    // The std target transpiles this (tests/cases/closure_block_for_loop.br); the kernel-module
    // emitter used to write `/* complex stmt */` and drop the loop.
    let src = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cases/closure_block_for_loop.br")).unwrap();
    let out = build("linux_closure_for", &src, &["--linux"]);
    assert_rejected(&out, "inside a closure/expression block");
}
