// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for Proposition 5:
// Last-use move elision on match / if-let pattern bindings and owned by-value parameters.

use std::path::Path;
use std::process::Command;

fn fn_body<'a>(src: &'a str, name: &str) -> &'a str {
    let start = src
        .find(&format!("fn {name}"))
        .unwrap_or_else(|| panic!("{name} not found in generated source:\n{src}"));
    let rest = &src[start..];
    // Find the matching closing brace of the function.
    let mut depth = 0usize;
    let mut end = rest.len();
    for (i, b) in rest.bytes().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = i + 1;
                    break;
                }
            }
            _ => {}
        }
    }
    &rest[..end]
}

#[test]
fn last_use_pattern_binding_and_owned_param_avoid_redundant_clones() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/codegen/last_use_pattern_binding.br");

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

    // 1. Owned match pattern binding: `consume(items)` without `.clone()`
    let owned_body = fn_body(&generated, "process_owned");
    assert!(
        owned_body.contains("consume(items)"),
        "process_owned must move `items` without .clone(), got:\n{owned_body}"
    );
    assert!(
        !owned_body.contains("items.clone()"),
        "process_owned must not clone `items`, got:\n{owned_body}"
    );

    // 2. Borrowed match subject: `items` is `&Vec<i64>`, so `consume(items.clone())` is required
    let borrowed_body = fn_body(&generated, "process_borrowed");
    assert!(
        borrowed_body.contains("consume(items.clone())"),
        "process_borrowed must clone `&Vec<i64>` `items`, got:\n{borrowed_body}"
    );

    // 3. Reused pattern binding: `items` is read again later, so first call must clone
    let reused_body = fn_body(&generated, "process_reused");
    assert!(
        reused_body.contains("consume(items.clone())"),
        "process_reused must clone `items` when read again later, got:\n{reused_body}"
    );

    // 4. Same-statement borrow + owned use: must keep `.clone()` to avoid E0505
    let same_stmt_body = fn_body(&generated, "process_same_stmt");
    assert!(
        same_stmt_body.contains("Holder { xs: items.clone() }"),
        "process_same_stmt must clone `items` when shared in same statement, got:\n{same_stmt_body}"
    );

    // 5. Owned `if let` binding: `consume(items)` without `.clone()`
    let if_let_body = fn_body(&generated, "process_if_let");
    assert!(
        if_let_body.contains("consume(items)"),
        "process_if_let must move `items` without .clone(), got:\n{if_let_body}"
    );
    assert!(
        !if_let_body.contains("items.clone()"),
        "process_if_let must not clone `items`, got:\n{if_let_body}"
    );

    // 6. Owned by-value parameter: `consume(xs)` without `.clone()`
    let forward_body = fn_body(&generated, "forward_owned");
    assert!(
        forward_body.contains("consume(xs)"),
        "forward_owned must move `xs` without .clone(), got:\n{forward_body}"
    );
    assert!(
        !forward_body.contains("xs.clone()"),
        "forward_owned must not clone `xs`, got:\n{forward_body}"
    );
}
