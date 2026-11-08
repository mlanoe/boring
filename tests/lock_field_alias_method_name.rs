// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression: `mut m = h.memory` (a lock-handle field declared in ANOTHER file) is an alias of
// the handle; a camelCase `def` called on `m` must keep its Boring name (`lookUp`), not be
// emitted as the snake_case `look_up` (E0599 on `MutexGuard<Memory>`).
//
// Run with:
//   tools/test.sh --test lock_field_alias_method_name

mod common;
use common::SharedCargo;
use std::path::Path;
use std::process::Command;

const MEMORY: &str = "pub struct Memory:\n    var mut {string=float} rows = {=}\n    var int hits = 0\n    def float lookUp(string key):\n        hits += 1\n        rows[key] else (0.0)\n\npub struct Holder:\n    pub mut Memory'actor memory = Memory()\n";
const MAIN: &str = "use memory\n\ndef float readIt(Holder h, string key):\n    mut m = h.memory\n    m.lookUp(key)\n\nlet h = Holder()\nprint \"value: {readIt(h, \"a\")}\"\n";

#[test]
fn alias_of_cross_file_lock_field_keeps_method_name() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("lock_field_alias_method_name");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("memory.br"), MEMORY).unwrap();
    std::fs::write(dir.join("main.br"), MAIN).unwrap();
    let bin = env!("CARGO_BIN_EXE_boring");

    let run = Command::new(bin).current_dir(&dir).args(["run", "main.br"]).output().unwrap();
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "value: 0");

    let build = Command::new(bin).current_dir(&dir).args(["build", "main.br"]).output().unwrap();
    assert!(build.status.success(), "build failed:\n{}", String::from_utf8_lossy(&build.stderr));
    let out = SharedCargo::new("lock_field_alias_method_name")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("main_rust/Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output().unwrap();
    assert!(out.status.success(), "cargo failed:\n{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "value: 0");
}
