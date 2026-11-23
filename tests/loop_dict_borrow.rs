// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: non-consuming reference iteration for maps (PROP-02).
// Verifies that `for k, v in my_dict:` does not emit `.clone().into_iter()`
// when the loop body only reads `k` and `v`.

use std::path::Path;
use std::process::Command;

#[test]
fn loop_dict_borrow_avoids_clone_into_iter() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/loop_dict_borrow.br");
    let dir = Path::new("tests/cases/loop_dict_borrow_test_rust");
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

    // 1. Struct field dict: `for id, b in blocks:` -> `.iter()`, not `.clone().into_iter()`
    let field_start = generated
        .find("fn count_tops_field")
        .expect("count_tops_field not found in generated source");
    let field_end = generated[field_start..].find("\n    }").unwrap_or(400);
    let field_fn = &generated[field_start..field_start + field_end];
    assert!(
        field_fn.contains("self.blocks.iter()"),
        "expected `self.blocks.iter()`, got:\n{}",
        field_fn
    );
    assert!(
        !field_fn.contains(".clone().into_iter()"),
        "must not emit `.clone().into_iter()` for read-only dict loop:\n{}",
        field_fn
    );

    // 2. Param dict: `for k, v in my_dict:` -> `my_dict.iter()`
    let param_start = generated
        .find("fn count_tops_param")
        .expect("count_tops_param not found in generated source");
    let param_end = generated[param_start..].find("\n}").unwrap_or(400);
    let param_fn = &generated[param_start..param_start + param_end];
    assert!(
        param_fn.contains("my_dict.iter()"),
        "expected `my_dict.iter()`, got:\n{}",
        param_fn
    );
    assert!(
        !param_fn.contains(".clone().into_iter()"),
        "must not emit `.clone().into_iter()`:\n{}",
        param_fn
    );

    // 3. Local dict: `for k, v in local_dict:` -> `local_dict.iter()` (non-consuming)
    let local_start = generated
        .find("fn count_tops_local")
        .expect("count_tops_local not found in generated source");
    let local_end = generated[local_start..].find("\n}").unwrap_or(400);
    let local_fn = &generated[local_start..local_start + local_end];
    assert!(
        local_fn.contains("local_dict.iter()"),
        "expected `local_dict.iter()`, got:\n{}",
        local_fn
    );

    // ── Real `cargo build` verification ──
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"loop_dict_borrow_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
