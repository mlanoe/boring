// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `text.chars()` passed straight to an external callee must be the real
// `str::chars()` iterator (`Item = char`), not the builtin's `Vec<Arc<str>>` of one-scalar
// strings, and a `for a, b in <imported external fn>(..)` loop must destructure the tuple
// items instead of auto-enumerating. Both shapes appear in
// `swash::text::analyze<I: IntoIterator + Clone>(chars) where I::Item: Borrow<char>`
// (UAX #14/#29 boundaries). Needs `swash` in the local cargo cache (built `--offline`).
//
// Run with:
//   cargo test --test external_chars_iterator

use std::path::Path;
use std::process::Command;

#[test]
fn chars_to_external_fn_and_tuple_items_compile_and_run() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("external_chars_iterator");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("boring.toml"),
        "[project]\nname = \"extchars\"\nversion = \"0.1.0\"\nmain = \"main.br\"\n\n[dependencies]\nswash = \"0.2.10\"\n",
    )
    .unwrap();
    let src = [
        "use swash.text.analyze",
        "use swash.text.cluster.Boundary",
        "",
        "def main():",
        "    let text = \"ab cd\\nef\"",
        "    var n = 0",
        "    for props, boundary in analyze(text.chars()):",
        "        if boundary == Boundary.Mandatory:",
        "            n += 1",
        "    for item in analyze(text.chars()):",
        "        if item.1 == Boundary.Mandatory:",
        "            n += 10",
        "    print n",
        "",
    ]
    .join("\n");
    std::fs::write(dir.join("main.br"), src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .current_dir(&dir)
        .args(["build", "main.br", "--offline"])
        .output()
        .expect("failed to invoke boring");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let proj = dir.join("main_rust");
    let rust = std::fs::read_to_string(proj.join("src/main.rs")).unwrap();
    assert!(rust.contains("analyze((&*text).chars())"), "{rust}");
    assert!(!rust.contains("analyze((&*text).chars()).into_iter().enumerate()"), "{rust}");

    let run = Command::new("cargo")
        .current_dir(&proj)
        .args(["run", "--offline", "--quiet"])
        .output()
        .expect("failed to invoke cargo");
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "11");
}
