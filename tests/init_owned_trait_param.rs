// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Codegen-shape half of the `init_owned_trait_param` regression (the real
// end-to-end compile+run half lives in `tests/transpile.rs`'s
// `transpile_test!(init_owned_trait_param, ignore_managed)`, backed by
// `tests/cases/init_owned_trait_param.br`/`.expected`).
//
// Pins the exact generated Rust shape for an explicit body-`init`'s
// `Trait'owned` parameter — the shape `desugar_inject.rs` synthesizes for
// `@inject`'s own `'owned` transient-dependency case (docs/design-notes/
// boring-di-draft.md §2, §4) — so a future change can't silently regress
// either half of the original bug:
//   1. The field/param type itself must be `Box<dyn Greeter>`, not
//      doubly-boxed as `Box<Box<dyn Greeter>>` (`emit_type`'s
//      `Type::Qualified(_, OwnerQual::Owned)` arm double-wrapped a trait
//      name, which already self-boxes to `Box<dyn Trait>` on its own).
//   2. The default expression's call site (`freshGreeter()`, returning bare
//      `impl Greeter` static dispatch) must be wrapped in `Box::new(...)`
//      to match — `struct_init_defaults` rendered it via the qualifier-blind
//      `emit_expr` instead of the qualifier-aware `emit_let_value`, so the
//      wrap never happened.
//
// Run with:
//   cargo test --test init_owned_trait_param

use std::path::Path;
use std::process::Command;

fn emit_rust(src: &str) -> String {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("init_owned_trait_param_codegen");
    std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, src).expect("failed to write fixture .br file");

    let emit = Command::new(bin)
        .arg("build")
        .arg(&br_file)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "boring build --emit-rust failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    String::from_utf8_lossy(&emit.stdout).into_owned()
}

const SRC: &str = r#"
trait Greeter:
    req string greet()

struct EnglishGreeter as Greeter:
    req string greet(): "hello"

Greeter freshGreeter():
    EnglishGreeter()

struct Greeting:
    Greeter'owned greeter

    init(Greeter'owned greeter = freshGreeter()):
        self.greeter = greeter

    req string say():
        self.greeter.greet()

def main():
    let g = Greeting()
    print g.say()
"#;

#[test]
fn owned_trait_init_param_is_single_boxed_and_default_is_wrapped() {
    let generated = emit_rust(SRC);

    // ── Bug 1: the field and the `new()` param must both be a single
    // `Box<dyn Greeter>` — never `Box<Box<dyn Greeter>>`. ────────────────────
    assert!(
        !generated.contains("Box<Box<dyn Greeter>>"),
        "field/param type must not be doubly boxed — generated source:\n{}",
        generated
    );
    assert!(
        generated.contains("pub greeter: Box<dyn Greeter>,"),
        "expected a single-boxed `Box<dyn Greeter>` field — generated source:\n{}",
        generated
    );
    assert!(
        generated.contains("pub fn new(greeter: Box<dyn Greeter>) -> Self {"),
        "expected `new()`'s param to be a single-boxed `Box<dyn Greeter>` — generated source:\n{}",
        generated
    );

    // ── Bug 2: the omitted-arg call site must wrap the default expression in
    // `Box::new(...)` to match that param type. ─────────────────────────────
    assert!(
        generated.contains("Greeting::new(Box::new(freshGreeter()))"),
        "expected the default-arg call site to box `freshGreeter()`'s result — generated source:\n{}",
        generated
    );
}
