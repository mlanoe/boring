// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A non-`task` method containing `for x in <async stream>:` used to emit `.await` inside a
// plain `fn` (rustc E0728, invisible from the Boring source). `boring build` must reject it
// with a diagnostic that names the fix (`task`).
//
// Run with:
//   cargo test --test method_iterates_async_stream_build_fails

use std::process::Command;

#[test]
fn method_iterating_async_stream_must_be_task() {
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "tests/cases/error_method_iterates_async_stream.br", "--emit-rust"])
        .output()
        .expect("failed to invoke boring");
    assert!(!out.status.success(), "expected a build error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked at"), "must be a clean diagnostic:\n{}", stderr);
    assert!(
        stderr.contains("method 'sumTicks' iterates an async stream") && stderr.contains("`task`"),
        "unexpected diagnostic:\n{}", stderr
    );
}
