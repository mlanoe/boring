// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A top-level array `let` read from a function body is promoted to a module-scope
// `static` (uppercased) instead of staying a `main()` local invisible to the function.
//
// Run with: cargo test --test top_level_array_statics

use std::process::Command;

const SRC: &str = "let a = [1, 2, 3]\nlet [float] f = [0.5, 1.5]\nint total():\n    a[0] + a[2]\nfloat half():\n    f[1]\nprint \"{total()} {half()}\"\n";

#[test]
fn top_level_arrays_are_visible_inside_functions() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("top_level_array_statics");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("main.br");
    std::fs::write(&src, SRC).unwrap();
    let build = Command::new(env!("CARGO_BIN_EXE_boring")).arg("build").arg(&src).current_dir(&dir).output().unwrap();
    assert!(build.status.success(), "{}", String::from_utf8_lossy(&build.stderr));
    let rs = std::fs::read_to_string(dir.join("main_rust/src/main.rs")).unwrap();
    assert!(rs.contains("static A: std::sync::LazyLock<Vec<isize>>"), "{rs}");
    assert!(rs.contains("static F: std::sync::LazyLock<Vec<f64>>"), "{rs}");
    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&src).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim_end(), "4 1.5");
}
