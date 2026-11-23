// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: automatic `&T` borrow inference for read-only parameters (PROP-06).
// Verifies that read-only, non-escaping struct/collection/enum parameters infer `&T`
// and call sites pass `&arg` without `.clone()`, while escaping or mutating parameters
// preserve by-value / `&mut T` semantics and compile cleanly.

use std::path::Path;
use std::process::Command;

fn fn_slice<'a>(src: &'a str, name: &str) -> &'a str {
    let needle = format!("fn {}(", name);
    let start = src
        .find(&needle)
        .unwrap_or_else(|| panic!("{} not found in generated source", needle));
    let rest = &src[start..];
    let mut depth = 0usize;
    let mut started = false;
    for (i, ch) in rest.char_indices() {
        if ch == '{' {
            depth += 1;
            started = true;
        } else if ch == '}' {
            depth -= 1;
            if started && depth == 0 {
                return &rest[..=i];
            }
        }
    }
    rest
}

#[test]
fn auto_param_borrow_infers_ref_for_readonly_and_preserves_escapes() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/codegen/auto_param_borrow.br");
    let dir = Path::new("tests/cases/auto_param_borrow_test_rust");
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

    // Read-only, non-escaping parameters infer `&T`.
    assert!(
        generated.contains("fn inspect_len(d: &BigData)"),
        "expected inspect_len to take &BigData:\n{}",
        fn_slice(&generated, "inspect_len")
    );
    assert!(
        generated.contains("fn relay_inspect(d: &BigData)"),
        "expected relay_inspect to take &BigData:\n{}",
        fn_slice(&generated, "relay_inspect")
    );
    assert!(
        generated.contains("fn sum_items(xs: &Vec<isize>)"),
        "expected sum_items to take &Vec<isize>:\n{}",
        fn_slice(&generated, "sum_items")
    );
    assert!(
        generated.contains("fn has_key(m: &HashMap<Arc<str>, isize>"),
        "expected has_key to take &HashMap:\n{}",
        fn_slice(&generated, "has_key")
    );
    assert!(
        generated.contains("fn tree_noop(t: &Tree)"),
        "expected tree_noop to take &Tree:\n{}",
        fn_slice(&generated, "tree_noop")
    );

    // Escaping / mutating parameters do NOT infer `&T`.
    assert!(
        generated.contains("fn passthrough(d: BigData)"),
        "expected passthrough to take BigData by value:\n{}",
        fn_slice(&generated, "passthrough")
    );
    assert!(
        generated.contains("fn passthrough_alias(d: BigData)"),
        "expected passthrough_alias to take BigData by value:\n{}",
        fn_slice(&generated, "passthrough_alias")
    );
    assert!(
        generated.contains("fn in_arr(d: BigData)"),
        "expected in_arr to take BigData by value:\n{}",
        fn_slice(&generated, "in_arr")
    );
    assert!(
        generated.contains("fn wrap(d: BigData)"),
        "expected wrap to take BigData by value:\n{}",
        fn_slice(&generated, "wrap")
    );
    assert!(
        generated.contains("d: &mut BigData"),
        "expected add_item to take &mut BigData:\n{}",
        fn_slice(&generated, "add_item")
    );
    assert!(
        generated.contains("fn tree_count(t: Tree)"),
        "expected tree_count to take Tree by value:\n{}",
        fn_slice(&generated, "tree_count")
    );

    // Call sites in main pass `&arg` for read-only calls without cloning.
    let main_body = fn_slice(&generated, "main");
    assert!(
        main_body.contains("inspect_len(&b)"),
        "expected inspect_len(&b) in main:\n{}",
        main_body
    );
    assert!(
        main_body.contains("relay_inspect(&b)"),
        "expected relay_inspect(&b) in main:\n{}",
        main_body
    );
    assert!(
        main_body.contains("has_key(&m"),
        "expected has_key(&m, ...) in main:\n{}",
        main_body
    );
    assert!(
        main_body.contains("tree_noop(&t)"),
        "expected tree_noop(&t) in main:\n{}",
        main_body
    );

    // Real `cargo build` verification.
    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"auto_param_borrow_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
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
