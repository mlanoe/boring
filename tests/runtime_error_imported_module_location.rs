// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a runtime error raised inside an imported module was reported against the
// *importing* file (with the module's line number, and an empty source excerpt). It must name
// the module's own file and show its source line.
//
// Run with:
//   cargo test --test runtime_error_imported_module_location

use std::process::Command;

#[test]
fn runtime_error_in_imported_module_names_that_module() {
    let dir = std::env::temp_dir().join(format!("boring_rt_err_loc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Module: the failing line is deep, past the end of the entry file.
    let mut helper = String::from("def int bad(int n):\n");
    for _ in 0..10 {
        helper.push_str("    let int pad = n\n");
    }
    helper.push_str("    n / 0\n");
    std::fs::write(dir.join("helper.br"), helper).unwrap();
    std::fs::write(dir.join("main.br"), "use helper\n\ndef main():\n    print bad(1)\n").unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(dir.join("main.br"))
        .output()
        .expect("failed to invoke boring");
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(stderr.contains("helper.br:12"), "wrong file in diagnostic:\n{stderr}");
    assert!(!stderr.contains("main.br:12"), "{stderr}");
    assert!(stderr.contains("n / 0"), "source line missing:\n{stderr}");
}
