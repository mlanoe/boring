// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a genuinely *external* enum's tuple-variant construction
// via dot-shorthand (`FontSize.Px(33.0)`) being camelCase→snake_cased into a
// nonexistent method name (`FontSize::px(33.0)`) instead of emitted verbatim
// (`FontSize::Px(33.0)`).
//
// Boring never parses the external crate's source, so an enum like `FontSize`
// (real shape: `bevy_text::FontSize { Px(f32), Vw(f32), Vh(f32), VMin(f32), ... }`,
// bevy_text-0.19.0/src/text.rs:487) is never registered in the transpiler's
// `enum_variant_fields` map the way a locally-declared Boring `enum` would be --
// which is exactly why a fixture using a local `enum FontSize: Px(float)` would
// NOT reproduce this bug: a local declaration resolves through the
// (already-correct) registered-variant path, not the unregistered/external
// fallback this test targets.
//
// `tests/transpile.rs` alone can't catch this either -- it builds a single
// binary crate that never references a real external type, so the bug's
// actual failure mode (`no function or associated item named 'px' found`)
// never triggers. This test instead emits the Boring function via
// `--emit-rust` (raw Rust source, no Boring-generated Cargo project) and
// prepends a hand-written stand-in for the "external" `FontSize` enum --
// exactly as if it came from a real dependency -- into a single-file binary
// crate, so a real `cargo build` either succeeds (`Px` preserved) or fails
// with the bug's real compile error (`px` doesn't exist).
//
// Run with:
//   cargo test --test external_enum_variant

use std::path::Path;
use std::process::Command;

/// Hand-written stand-in for the real external enum (bevy_text::FontSize) --
/// Boring never parses this declaration, exactly mirroring how a real
/// `use bevy.prelude.*` import behaves (the type just exists at Rust
/// compile time, with no Boring-side registration).
const FONT_SIZE_STUB: &str = r#"
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum FontSize {
    Px(f32),
    Vw(f32),
    Vh(f32),
    VMin(f32),
}
"#;

#[test]
fn external_tuple_variant_dot_call_preserves_case() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/external_enum_tuple_variant.br");
    let dir = Path::new("tests/cases/external_enum_tuple_variant_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let emit = Command::new(bin)
        .arg("build").arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "boring build --emit-rust failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout);

    // Combine into a single file so the stub enum is in scope for the
    // generated code without needing any cross-module `use`/path wiring --
    // Rust item resolution within one file doesn't depend on declaration order.
    let combined = format!("{}\n{}\n", FONT_SIZE_STUB, generated);
    std::fs::write(dir.join("src/main.rs"), &combined).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"external_enum_tuple_variant_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    ).expect("failed to write Cargo.toml");

    let build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {}", e));

    assert!(
        build.status.success(),
        "expected `FontSize.Px(33.0)` to transpile to valid Rust against a real \
         external `FontSize` enum, but `cargo build` failed:\n--- stderr ---\n{}\n\
         --- generated source ---\n{}",
        String::from_utf8_lossy(&build.stderr),
        combined,
    );
}
