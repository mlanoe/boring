// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Build-compilation tests for the boring-written interpreter
// (boring/interpreter/).
//
// Each test:
//   1. Runs `boring build [--threading <t>]` from boring/interpreter/
//   2. Runs `cargo build` on the generated Rust project
//   3. Asserts zero errors (no expected output — the interpreter reads stdin)
//
// Run with:
//   cargo test --test interpreter_build
//
// The first run compiles the generated Rust project from scratch (slow ~30s).
// Subsequent runs reuse the cargo build cache (fast).

use std::path::Path;
use std::process::Command;

fn build_interpreter(threading: &str) {
    let bin         = env!("CARGO_BIN_EXE_boring");
    let project_dir = Path::new("boring/interpreter");

    // ── Step 1: transpile ─────────────────────────────────────────────────────
    let mut cmd = Command::new(bin);
    cmd.arg("build").current_dir(project_dir);
    if threading != "multi" {
        cmd.arg("--threading").arg(threading);
    }
    let emit = cmd
        .output()
        .unwrap_or_else(|e| panic!("[interpreter@{}] failed to invoke boring: {}", threading, e));

    assert!(
        emit.status.success(),
        "[interpreter@{}] boring build failed:\n{}",
        threading,
        String::from_utf8_lossy(&emit.stderr)
    );

    // ── Step 2: cargo build ───────────────────────────────────────────────────
    let rust_dir = if threading == "multi" {
        project_dir.join("main_rust")
    } else {
        project_dir.join(format!("main_rust_{}", threading))
    };

    let build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(rust_dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("[interpreter@{}] failed to invoke cargo build: {}", threading, e));

    assert!(
        build.status.success(),
        "[interpreter@{}] cargo build failed:\n--- stderr ---\n{}",
        threading,
        String::from_utf8_lossy(&build.stderr)
    );
}

#[test]
fn interpreter_transpiles_single_thread() {
    build_interpreter("single");
}

#[test]
fn interpreter_transpiles_multi_thread() {
    build_interpreter("multi");
}
