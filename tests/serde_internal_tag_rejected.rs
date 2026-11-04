// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// An internally tagged enum (`@serde(tag = "t")`, no `content`) writes the tag inside the
// payload's own JSON object, so serde can only represent field-less variants and
// one-field variants holding a struct or a map. `Rect(float, float)` used to fail
// `cargo build` with serde's derive error (`cannot be used with tuple variants`), and
// `Circle(float)` compiled but failed at runtime in `json(v)`. The checker now rejects
// both with the same clear message for `boring run` and `boring build`.
//
// Run with:
//   cargo test --test serde_internal_tag_rejected

use std::process::Command;

fn stderr_of(sub: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(sub)
        .arg("tests/cases/error_serde_internal_tag.br")
        .args(if sub[0] == "build" { &["--emit-rust"][..] } else { &[][..] })
        .output()
        .expect("failed to invoke boring");
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn run_and_build_reject_unrepresentable_internal_tag_variants() {
    for sub in [&["run"][..], &["build"][..]] {
        let (ok, err) = stderr_of(sub);
        assert!(!ok, "`boring {}` must fail; stderr:\n{}", sub[0], err);
        assert!(!err.contains("panicked at"), "{}: not a panic:\n{}", sub[0], err);
        assert!(
            err.contains("variant `Rect` cannot use the internally tagged form")
                && err.contains("it has more than one field"),
            "{}: missing Rect diagnostic:\n{}", sub[0], err
        );
        assert!(
            err.contains("variant `Circle` cannot use the internally tagged form")
                && err.contains("not a struct or a `{string=T}` dict"),
            "{}: missing Circle diagnostic:\n{}", sub[0], err
        );
        assert!(!err.contains("variant `Dot`"), "{}: Dot is valid:\n{}", sub[0], err);
    }
}
