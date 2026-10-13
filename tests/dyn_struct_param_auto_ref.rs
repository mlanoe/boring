// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a bare struct parameter whose struct has dynamic fields
// (arrays/dicts/sets) used to be emitted by value (`s: Dyn`) with callers
// `.clone()`ing it, because `infer_qualifiers` only made fixed-size
// (`type_sizes`) structs eligible for auto-ref inference. The docs promise
// structs are always passed by reference with no `&` written.
//
// Run with:
//   cargo test --test dyn_struct_param_auto_ref

use std::path::Path;
use std::process::Command;

#[test]
fn dynamic_field_struct_param_is_auto_borrowed() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let emit = Command::new(bin)
        .arg("build")
        .arg(Path::new("tests/cases/dyn_struct_param_auto_ref.br"))
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "boring build --emit-rust failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    assert!(generated.contains("fn small_n(s: &Small)"), "{}", generated);
    assert!(generated.contains("fn dyn_n(s: &Dyn)"), "{}", generated);
    assert!(generated.contains("fn dyn_len(s: &Dyn)"), "{}", generated);

    let main_start = generated.find("fn main").expect("fn main not found");
    let main_body = &generated[main_start..];
    assert!(
        !main_body.contains(".clone()"),
        "call sites must not clone the borrowed struct:\n{}",
        main_body
    );
}
