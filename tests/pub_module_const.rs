// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a `pub let` at module scope emitting a *private* Rust
// `const` regardless of the `pub` keyword in the source. `tests/transpile.rs`'s
// `pub_top_level_const` case proves the generated code still runs correctly, but
// that alone can't catch this bug: it builds a single binary crate, so nothing
// outside the crate ever tries to reach the constant and the missing `pub` never
// bites. The real motivating shape (see `bevy-boring`/`breakout-boring`) is a
// `.br` file's `boring build --emit-rust` output wired into a hand-written
// `lib.rs` as `pub mod boring_gen;`, specifically so a *sibling* file (tests,
// other modules) can import from it -- exactly the E0603 "constant is private"
// failure this test reproduces against a real `cargo build` of a two-file crate.
//
// Run with:
//   cargo test --test pub_module_const

use std::path::{Path, PathBuf};
use std::process::Command;

/// Emits `<case>.br` to a bare Rust module (`boring build --emit-rust`, no
/// Cargo project) and writes it to `<dir>/src/gen.rs`.
fn emit_gen_module(case_br: &Path, dir: &Path) {
    let bin = env!("CARGO_BIN_EXE_boring");
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
    std::fs::write(dir.join("src/gen.rs"), emit.stdout).expect("failed to write gen.rs");
}

fn write_crate(dir: &Path, name: &str, lib_rs: &str) {
    std::fs::create_dir_all(dir).expect("failed to create crate dir");
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
            name
        ),
    ).expect("failed to write Cargo.toml");
    std::fs::write(dir.join("src/lib.rs"), lib_rs).expect("failed to write lib.rs");
}

fn cargo_build(manifest: &Path) -> std::process::Output {
    Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {}", e))
}

/// `pub let float32 LEFT_WALL` must emit `pub const LEFT_WALL: f32`, reachable
/// from a hand-written sibling module the same way `pub struct`/`pub def` already are.
#[test]
fn pub_let_const_is_visible_across_modules() {
    let case_br = Path::new("tests/cases/pub_top_level_const.br");
    let dir: PathBuf = Path::new("tests/cases/pub_top_level_const_cross_module_rust").join("pub_ok");
    emit_gen_module(case_br, &dir);
    write_crate(
        &dir,
        "pub_ok",
        r#"
pub mod gen;

// Hand-written sibling code, exactly the `pub mod boring_gen;` shape
// `bevy-boring`/`breakout-boring` use to share a Boring-authored constant
// with hand-written Rust (e.g. from `tests/`).
pub fn use_left_wall() -> f32 {
    gen::LEFT_WALL
}
"#,
    );

    let out = cargo_build(&dir.join("Cargo.toml"));
    assert!(
        out.status.success(),
        "expected cross-module access to `pub let LEFT_WALL` to compile, but it failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `pub let` promotion must not depend on whether anything *inside the same file*
/// references the constant -- `SCOREBOARD_FONT_SIZE`/`SCOREBOARD_TEXT_PADDING` in
/// `pub_top_level_const_unused.br` have zero in-file consumers (no function,
/// struct/enum method reads them anywhere), which is the actual bug-report shape:
/// constants moved into a `.br` file purely so a hand-written Rust sibling module
/// can read them. Before the fix, promotion was gated entirely on an in-file
/// "referenced elsewhere" scan, so a `pub` constant with no in-file reader stayed
/// a local inside the synthesized `fn main()` -- invisible to any other module,
/// and `pub` on a local binding isn't even valid Rust in the first place. This
/// test's `emit_gen_module` step alone would have caught that second half (a
/// compile error emitting the bare module), but only the cross-module `cargo
/// build` below proves the constant was actually promoted to a *real* module-level
/// item, as opposed to just not crashing.
#[test]
fn pub_let_with_no_in_file_consumer_is_visible_across_modules() {
    let case_br = Path::new("tests/cases/pub_top_level_const_unused.br");
    let dir: PathBuf = Path::new("tests/cases/pub_top_level_const_unused_cross_module_rust").join("pub_ok");
    emit_gen_module(case_br, &dir);
    write_crate(
        &dir,
        "pub_ok",
        r#"
pub mod gen;

// Hand-written sibling code reading constants that are never referenced
// anywhere inside the `.br` file itself -- the only thing that marks them as
// public API is the `pub` keyword on their declaration.
pub fn use_scoreboard_constants() -> f32 {
    gen::SCOREBOARD_FONT_SIZE + gen::SCOREBOARD_TEXT_PADDING
}
"#,
    );

    let out = cargo_build(&dir.join("Cargo.toml"));
    assert!(
        out.status.success(),
        "expected cross-module access to unreferenced `pub let` constants to compile, but it failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Control case for `pub_let_with_no_in_file_consumer_is_visible_across_modules`:
/// `UNUSED_PRIVATE` (bare `let`, zero in-file consumers, same file) must stay
/// un-promoted entirely -- a private, unused constant has no reason to exist as
/// a real Rust item. Unlike `bare_let_const_stays_private_across_modules` below
/// (whose `PRIVATE_SPEED` control IS promoted, just as a private `const`, so the
/// sibling module hits E0603 "private"), this one was never emitted at module
/// scope at all -- the identifier doesn't exist there, so the failure mode is
/// "cannot find value" (E0425)/unresolved-name, not a privacy error.
#[test]
fn unused_bare_let_is_not_promoted_across_modules() {
    let case_br = Path::new("tests/cases/pub_top_level_const_unused.br");
    let dir: PathBuf = Path::new("tests/cases/pub_top_level_const_unused_cross_module_rust").join("priv_blocked");
    emit_gen_module(case_br, &dir);
    write_crate(
        &dir,
        "priv_blocked",
        r#"
pub mod gen;

pub fn use_unused_private() -> f32 {
    gen::UNUSED_PRIVATE
}
"#,
    );

    let out = cargo_build(&dir.join("Cargo.toml"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "expected cross-module access to the never-promoted `UNUSED_PRIVATE` const to fail to compile, but it succeeded"
    );
    assert!(
        stderr.contains("cannot find") || stderr.contains("E0425") || stderr.contains("unresolved"),
        "expected a name-resolution error referencing `UNUSED_PRIVATE`, got:\n{}",
        stderr
    );
}

/// A bare (non-`pub`) top-level `let` must keep emitting a *private* `const` --
/// this is the control case proving the fix didn't just make everything `pub`.
#[test]
fn bare_let_const_stays_private_across_modules() {
    let case_br = Path::new("tests/cases/pub_top_level_const.br");
    let dir: PathBuf = Path::new("tests/cases/pub_top_level_const_cross_module_rust").join("priv_blocked");
    emit_gen_module(case_br, &dir);
    write_crate(
        &dir,
        "priv_blocked",
        r#"
pub mod gen;

pub fn use_private_speed() -> f32 {
    gen::PRIVATE_SPEED
}
"#,
    );

    let out = cargo_build(&dir.join("Cargo.toml"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "expected cross-module access to the private `PRIVATE_SPEED` const to fail to compile, but it succeeded"
    );
    assert!(
        stderr.contains("private") || stderr.contains("E0603"),
        "expected a privacy error (E0603) referencing `PRIVATE_SPEED`, got:\n{}",
        stderr
    );
}
