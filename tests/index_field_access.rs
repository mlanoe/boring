// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: index + field fusion (PROP-02).
// Verifies that `records[i].status` emits `records[(i) as usize].status.clone()`
// rather than `records[(i) as usize].clone().status`, and that `by_id[k].status`
// projects `.status.clone()` without cloning the entire `Record`.

use std::path::Path;
use std::process::Command;

#[test]
fn index_field_access_avoids_whole_element_clone() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/codegen/index_field_access.br");
    let dir = Path::new("tests/cases/index_field_access_test_rust");
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

    // 1. Array index + primitive field: `records[i].status` -> `records[(i) as usize].status` (Copy, no clone)
    let get_status_start = generated
        .find("fn get_status")
        .expect("get_status not found in generated source");
    let get_status_body = &generated[get_status_start..];
    let get_status_fn = &get_status_body[..get_status_body.find('}').unwrap_or(200)];
    assert!(
        get_status_fn.contains("self.records[(i) as usize].status;"),
        "expected `self.records[(i) as usize].status;` (Copy, no clone), got:\n{}",
        get_status_fn
    );
    assert!(
        !get_status_fn.contains("self.records[(i) as usize].clone().status"),
        "must not clone the whole Record before projecting .status:\n{}",
        get_status_fn
    );

    // 1b. Array index + non-Copy field: `records[i].name` -> `records[(i) as usize].name.clone()`
    let get_name_start = generated
        .find("fn get_name")
        .expect("get_name not found in generated source");
    let get_name_body = &generated[get_name_start..];
    let get_name_fn = &get_name_body[..get_name_body.find('}').unwrap_or(200)];
    assert!(
        get_name_fn.contains("self.records[(i) as usize].name.clone()"),
        "expected `self.records[(i) as usize].name.clone()`, got:\n{}",
        get_name_fn
    );
    assert!(
        !get_name_fn.contains("self.records[(i) as usize].clone().name"),
        "must not clone the whole Record before projecting .name:\n{}",
        get_name_fn
    );

    // 2. Dict index + primitive field: `by_id[k].status` -> `.expect("dict key not found").status`
    let get_dict_start = generated
        .find("fn get_dict_status")
        .expect("get_dict_status not found in generated source");
    let get_dict_body = &generated[get_dict_start..];
    let get_dict_fn = &get_dict_body[..get_dict_body.find('}').unwrap_or(200)];
    assert!(
        get_dict_fn.contains(".expect(\"dict key not found\").status;"),
        "expected dict index + field to project .status without clone, got:\n{}",
        get_dict_fn
    );
    assert!(
        !get_dict_fn.contains(".cloned().expect(\"dict key not found\").status"),
        "must not clone the whole Record out of the dict:\n{}",
        get_dict_fn
    );

    // ── Real `cargo build` verification ──
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"index_field_access_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
