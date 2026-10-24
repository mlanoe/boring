// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A receiver typed as an external (non-user) type: `.map(variable)` is a plain method call
// (not an iterator adaptor), and a method name that a user `trait` also declares is still
// camel→snake converted.

use std::process::Command;

fn emit(name: &str, src: &str) -> String {
    let dir = std::env::temp_dir().join(format!("boring_ext_recv_{}_{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("a.br");
    std::fs::write(&file, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--emit-rust"]).arg(&file).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_dir_all(&dir);
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn external_typed_receiver_map_with_variable_is_plain_call() {
    let rs = emit("map", "def uint32 lookup(Charmap charmap, uint32 code):\n    charmap.map(code)\n\ndef main():\n    print \"x\"\n");
    assert!(rs.contains("charmap.map(code)"), "{rs}");
    assert!(!rs.contains("iter().cloned().map(code)"), "{rs}");
}

#[test]
fn external_typed_receiver_trait_named_method_is_snake_case() {
    let rs = emit("trait", "trait Dir:\n    req bool isRtl()\n\ndef bool f(Lvl lv):\n    lv.isRtl()\n\ndef main():\n    print \"x\"\n");
    assert!(rs.contains("lv.is_rtl()"), "{rs}");
}
