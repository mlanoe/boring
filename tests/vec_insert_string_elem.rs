// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: `ids.insert(i, target_id)` on an untyped `[]` local filled with strings fell
// into the external-call fallback, which passes a string argument as a borrowed `(&*s)` (`&str`)
// — rustc E0308 against `Vec<Arc<str>>::insert`. A Vec `insert(index, value)` now takes its value
// like `push` does: an owned, cloned element.
//
// Run with:
//   cargo test --test vec_insert_string_elem

use std::process::Command;

#[test]
fn vec_insert_passes_owned_string_element() {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--emit-rust", "tests/cases/vec_insert_string_elem.br"])
        .output()
        .expect("failed to invoke boring");
    assert!(out.status.success(), "stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    let rust = String::from_utf8_lossy(&out.stdout);
    assert!(rust.contains("ids.insert((1 as usize), target_id.clone())"), "{rust}");
    assert!(!rust.contains("(&*target_id)"), "{rust}");
}
