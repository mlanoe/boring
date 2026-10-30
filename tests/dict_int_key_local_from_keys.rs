// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `let id = ks[0]` where `ks = d.keys()` of a `{int=float}` dict left `id`
// untyped, so `d[id]` fell into `emit_dict_key_borrow`'s `Arc<str>` fallback and emitted
// `d.get(&*id)` — rustc E0614 (`isize` cannot be dereferenced). `resolve_expr_type` now types
// `dict.keys()`/`.values()` as arrays of the key/value type, so the local resolves to `int`.
//
// Run with: cargo test --test dict_int_key_local_from_keys

use std::path::Path;
use std::process::Command;

#[test]
fn int_key_local_from_keys_indexes_without_deref() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let emit = Command::new(bin)
        .args(["build", "tests/cases/dict_int_key_local_from_keys.br", "--emit-rust"])
        .output()
        .expect("failed to invoke boring");
    assert!(emit.status.success(), "{}", String::from_utf8_lossy(&emit.stderr));
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();
    assert!(!generated.contains("get(&*id)"), "int key must not be dereferenced:\n{generated}");
    assert!(generated.contains("get(&id)"), "expected `get(&id)`:\n{generated}");

    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("dict_int_key_local_from_keys");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("main.rs");
    std::fs::write(&src, &generated).unwrap();
    let out = Command::new("rustc")
        .args(["--edition", "2021", "--crate-type", "bin", "-o"])
        .arg(dir.join("main_bin"))
        .arg(&src)
        .output()
        .expect("failed to invoke rustc");
    assert!(out.status.success(), "rustc failed:\n{}\n{generated}", String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_dir_all(dir);
}
