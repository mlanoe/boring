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

/// Return the host triple (e.g. `x86_64-unknown-linux-gnu`) so that `cargo build
/// --target <triple>` ignores any `[build] target` override in `.cargo/config.toml`.
fn host_triple() -> String {
    let out = Command::new("rustc")
        .args(["-vV"])
        .output()
        .expect("rustc not found");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find(|l| l.starts_with("host:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("x86_64-unknown-linux-gnu")
        .to_string()
}

fn build_interpreter(mode: &str, threading: &str) {
    let label       = format!("{}+{}", mode, threading);
    let bin         = env!("CARGO_BIN_EXE_boring");
    let project_dir = Path::new("boring/interpreter");

    // ── Step 1: transpile ─────────────────────────────────────────────────────
    let mut cmd = Command::new(bin);
    cmd.arg("build").current_dir(project_dir);
    if threading != "multi" {
        cmd.arg("--threading").arg(threading);
    }
    if mode != "strict" {
        cmd.arg("--mode").arg(mode);
    }
    let emit = cmd
        .output()
        .unwrap_or_else(|e| panic!("[interpreter@{}] failed to invoke boring: {}", label, e));

    assert!(
        emit.status.success(),
        "[interpreter@{}] boring build failed:\n{}",
        label,
        String::from_utf8_lossy(&emit.stderr)
    );

    // ── Step 2: cargo build ───────────────────────────────────────────────────
    let rust_dir = match (mode, threading) {
        ("strict",  "multi")  => project_dir.join("main_rust"),
        ("strict",  _)        => project_dir.join(format!("main_rust_{}", threading)),
        (_,         "multi")  => project_dir.join(format!("main_rust_{}", mode)),
        _                     => project_dir.join(format!("main_rust_{}_{}", mode, threading)),
    };

    let triple = host_triple();
    let build = Command::new("cargo")
        .args(["build", "--quiet", "--target", &triple, "--manifest-path"])
        .arg(rust_dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("[interpreter@{}] failed to invoke cargo build: {}", label, e));

    assert!(
        build.status.success(),
        "[interpreter@{}] cargo build failed:\n--- stderr ---\n{}",
        label,
        String::from_utf8_lossy(&build.stderr)
    );
}

#[test] fn interpreter_strict_multi()          { build_interpreter("strict",  "multi");  }
#[test] fn interpreter_strict_single()         { build_interpreter("strict",  "single"); }
#[test] fn interpreter_managed_multi()         { build_interpreter("managed", "multi");  }
#[test] fn interpreter_managed_single()        { build_interpreter("managed", "single"); }
