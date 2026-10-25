// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: indexing an array with a cast of an external struct's field
// (`a[r.start as int]`, `r` a `swash` `SourceRange` with `u32` fields) used to emit
// `a[(r.start as isize)]`, which rustc rejects. Needs `swash` in the local cargo
// cache (built `--offline`).
//
// Run with:
//   cargo test --test external_field_cast_index

use std::path::Path;
use std::process::Command;

#[test]
fn cast_external_field_index_compiles_and_runs() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("external_field_cast_index");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("boring.toml"),
        "[project]\nname = \"extcastidx\"\nversion = \"0.1.0\"\nmain = \"main.br\"\n\n[dependencies]\nswash = \"0.2.10\"\n",
    )
    .unwrap();
    let src = [
        "use swash.text.cluster.SourceRange",
        "",
        "def main():",
        "    mut [int] a = [10, 20, 30, 40]",
        "    let r = SourceRange(start = 2, end = 3)",
        "    print a[r.start as int]",
        "    a[r.end as int] = 7",
        "    a[r.start as int] += 1",
        "    print a",
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
    let run = Command::new("cargo")
        .current_dir(dir.join("main_rust"))
        .args(["run", "--offline", "--quiet"])
        .output()
        .expect("failed to run cargo");
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "30\n[10, 20, 31, 7]");
}
