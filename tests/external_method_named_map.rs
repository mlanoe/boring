// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test: a plain method named `map` (or `filter`) on an external/unknown receiver,
// taking a non-callable argument (`swash::Charmap::map(codepoint)`), must stay a plain
// method call instead of being lowered to `.iter().cloned().map(..).collect::<Vec<_>>()`.
//
// Run with:
//   cargo test --test external_method_named_map

use std::process::Command;

fn emit_rust(src: &str, name: &str) -> String {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("main.br"), src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .current_dir(&dir)
        .args(["build", "main.br", "--emit-rust"])
        .output()
        .expect("failed to invoke boring");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn external_map_with_scalar_arg_is_plain_method_call() {
    let rust = emit_rust(
        "use swash.FontRef\n\ndef void main():\n    let font = FontRef.from_index(0 as uint32)\n    let g = font.charmap().map(65 as uint32)\n    print(g)\n",
        "external_method_named_map",
    );
    assert!(rust.contains("font.charmap().map((65i64 as u32))"), "{}", rust);
    assert!(!rust.contains("iter().cloned().map"), "{}", rust);
}

#[test]
fn array_map_with_closure_still_lowers_to_iterator_chain() {
    let rust = emit_rust(
        "def void main():\n    let [int] a = [1, 2]\n    let b = a.map((x): x + 1)\n    print(b)\n",
        "array_map_closure_unchanged",
    );
    assert!(rust.contains("iter().cloned().map"), "{}", rust);
}

#[test]
fn external_callback_closure_param_makes_boring_fn_param_borrowed() {
    let rust = emit_rust(
        "use swash.shape.ShapeContext\nuse swash.shape.cluster.GlyphCluster\n\ndef void collect(GlyphCluster cluster, var mut [int] out):\n    out.push(cluster.len as int)\n\ndef void main():\n    var mut out = [0]\n    let shaper = ShapeContext.new()\n    shaper.shapeWith((cluster): collect(cluster, out))\n",
        "external_callback_closure_borrow",
    );
    assert!(rust.contains("fn collect(cluster: &GlyphCluster"), "{}", rust);
}
