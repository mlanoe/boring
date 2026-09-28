// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Codegen-shape half of the `init_bare_trait_param` regression (the real
// end-to-end compile+run half lives in `tests/transpile.rs`'s
// `transpile_test!(init_bare_trait_param)`, backed by
// `tests/cases/init_bare_trait_param.br`/`.expected`).
//
// Sibling to `tests/init_owned_trait_param.rs`, but for a BARE (unqualified)
// trait-typed init param instead of an explicitly `'owned`-qualified one —
// `Greeter greeter = greeterProvider()`, no qualifier written at all. This is
// exactly the shape `desugar_inject.rs` needs for a bare `@inject` field
// matched against a non-`@singleton` ("transient") provider (see its own doc
// comment and `tests/dependency_injection.rs`'s header, "explicit field
// qualifier required (no bare-field inference yet)").
//
// Unlike the `'owned` case, a bare trait-typed field's representation isn't
// decided by chapter-30's per-function usage-based inference (that pipeline
// is for genuine struct-typed fields) — it's a fixed rule: a trait object is
// unsized, so a bare trait-typed field is *always* `Box<dyn Trait>`
// (`emit_field_type`'s "Priority 4 (dyn Trait) still applies" arm). But
// `struct_init_defaults` (`src/transpiler/mod.rs`) only recognized an
// EXPLICITLY `'owned`/`'new`-qualified param as needing the qualifier-aware
// `emit_let_value` rendering — a bare trait-typed param fell through to the
// qualifier-blind `emit_expr`, so `greeterProvider()` (returning bare `impl
// Greeter`) was never wrapped in `Box::new(...)` to match the field/param's
// own `Box<dyn Greeter>` type, and the generated Rust failed to compile
// (E0308: expected `Box<dyn Greeter>`, found opaque type).
//
// Run with:
//   cargo test --test init_bare_trait_param

use std::path::Path;
use std::process::Command;

fn emit_rust(src: &str) -> String {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("init_bare_trait_param_codegen");
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

Greeter greeterProvider():
    EnglishGreeter()

struct Greeting:
    Greeter greeter

    init(Greeter greeter = greeterProvider()):
        self.greeter = greeter

    req string say():
        self.greeter.greet()

def main():
    let g = Greeting()
    print g.say()
"#;

#[test]
fn bare_trait_init_param_default_is_wrapped() {
    let generated = emit_rust(SRC);

    // The field and `new()`'s param must be `Box<dyn Greeter>` (unconditional
    // for a bare trait-typed field — no double-boxing risk here since there's
    // no explicit `'owned` qualifier for `emit_type` to additionally wrap).
    assert!(
        generated.contains("pub greeter: Box<dyn Greeter>,"),
        "expected a `Box<dyn Greeter>` field — generated source:\n{}",
        generated
    );
    assert!(
        generated.contains("pub fn new(greeter: Box<dyn Greeter>) -> Self {"),
        "expected `new()`'s param to be `Box<dyn Greeter>` — generated source:\n{}",
        generated
    );

    // The omitted-arg call site must wrap the default expression in
    // `Box::new(...)` to match that param type.
    assert!(
        generated.contains("Greeting::new(Box::new(greeterProvider()))"),
        "expected the default-arg call site to box `greeterProvider()`'s result — generated source:\n{}",
        generated
    );
}
