// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: nested map chained reads (PROP-01).
// Verifies that chained reads on maps (`local_table.get(k1)?.get(k2)`)
// emit `.get().and_then(...)` without cloning intermediate maps.

use std::path::Path;
use std::process::Command;

#[test]
fn chained_nested_map_get_uses_and_then_without_cloning_inner_map() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/codegen/nested_map_get.br");
    let dir = Path::new("tests/cases/nested_map_get_test_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "boring build --emit-rust failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    // ── Codegen-shape assertions ──

    // 1. Double lookup: local_table.get(k1)?.get(k2)
    let get_val_start = generated
        .find("fn get_val")
        .expect("get_val not found in generated source");
    let get_val_body = &generated[get_val_start..];

    assert!(
        get_val_body.contains(".local_table.get(&*k1).and_then(|__v| __v.get(&*k2)).cloned()"),
        "expected double map lookup to transpile to `.get(&*k1).and_then(|__v| __v.get(&*k2)).cloned()`, but got:\n{}",
        &get_val_body[..get_val_body.find('}').unwrap_or(200)]
    );

    // Ensure no redundant Some(...) wrapping
    assert!(
        !get_val_body[..get_val_body.find('}').unwrap_or(200)].contains("Some("),
        "get_val must not wrap the already-Option result in Some(...)"
    );

    // 2. Chained get with else default: local_table.get(k1)?.get(k2) else def_val
    let get_or_default_start = generated
        .find("fn get_or_default")
        .expect("get_or_default not found in generated source");
    let get_or_default_body = &generated[get_or_default_start..];

    assert!(
        get_or_default_body.contains(".unwrap_or_else("),
        "expected else branch to emit `.unwrap_or_else(...)`"
    );

    // 3. Triple lookup: deep.get(k1)?.get(k2)?.get(k3)
    let get_3_start = generated
        .find("fn get_3")
        .expect("get_3 not found in generated source");
    let get_3_body = &generated[get_3_start..];

    assert!(
        get_3_body.contains("deep.get(&*k1).and_then(|__v| __v.get(&*k2)).and_then(|__v| __v.get(&*k3)).cloned()"),
        "expected triple map lookup to chain two and_then calls without intermediate clones"
    );

    // ── Real `cargo build` verification ──
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"nested_map_get_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("failed to write Cargo.toml");

    let build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {}", e));

    assert!(
        build.status.success(),
        "expected the generated Rust to compile, but `cargo build` failed:\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&build.stderr),
        generated,
    );

    let _ = std::fs::remove_dir_all(dir);
}
