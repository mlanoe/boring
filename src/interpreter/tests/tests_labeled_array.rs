// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// This file is part of Boring.
// Boring is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// See the LICENSE file at the project root for the full text.

// Labeled multi-dimensional arrays (docs/array-multidim-proposal.md) —
// interpreter GPU-kernel lowering (`lower_labeled_array_methods`,
// eval_gpu.rs), the LabeledArray sibling of tests_gpu.rs's Image/Volume
// `.at()`/`.width()`/`.height()` tests just above this module's own
// declaration.
//
// Fixed-shape tests use plain `run()` directly, same as the Image/Volume
// precedent — that lowering is entirely self-contained inside
// `exec_kernel_decl`, no pre-pass desugar dependency. Dynamic-shape tests
// need `desugar_labeled_array` run first (it injects the shadow fields
// `lower_labeled_array_methods` never has to — that field is already a
// plain buffer by the time the interpreter sees it), so those use their own
// small full-pipeline helper instead of `run()`.

use super::{run, get_var};
use super::*;
use crate::desugar_labeled_array::desugar_labeled_array;

fn run_with_labeled_array_desugar(src: &str) -> Value {
    let tokens = crate::lexer::lex(src).expect("lex error");
    let program = crate::parser::parse(tokens).expect("parse error");
    let program = desugar_labeled_array(program);
    let mut interp = Interpreter::new();
    interp.exec_program(&program).expect("runtime error");
    let val = interp.global.borrow().get("_result").unwrap_or(Value::Nil);
    val
}

#[test]
fn labeled_index_and_size_lower_correctly_on_fixed_shape_field() {
    // Mirrors test_image_at_width_height_lower_correctly (tests_gpu.rs) —
    // same shape, LabeledArray syntax instead of Image<T,C,R>.
    let src = r#"
kernel Tile:
    mut [float]'unified                     out
    mut [float, width = 4, height = 4]'actor tile

    init([float]'unified data):
        out = data

    def ():
        tile[width = 0, height = 0] = 1.0
        tile[width = 1, height = 2] = 5.0
        out[0] = tile[width = 0, height = 0]
        out[1] = tile[width = 1, height = 2]
        out[2] = float(tile.size(.width))
        out[3] = float(tile.size(.height))

let data = [0.0, 0.0, 0.0, 0.0]
mut k = Tile(data)
kernel:
    k(block = 1)
let _result = k.out
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    let val = get_var(&interp, "_result");
    assert_eq!(
        val,
        Value::Array(vec![Value::Float(1.0), Value::Float(5.0), Value::Float(4.0), Value::Float(4.0)].into())
    );
}

#[test]
fn labeled_index_order_free_at_use_site() {
    // a[height=.., width=..] must address the same element as
    // a[width=.., height=..] — order is free at the use site (design doc).
    let src = r#"
kernel Tile:
    mut [float]'unified                     out
    mut [float, width = 4, height = 4]'actor tile

    init([float]'unified data):
        out = data

    def ():
        tile[height = 2, width = 1] = 9.0
        out[0] = tile[width = 1, height = 2]

let data = [0.0]
mut k = Tile(data)
kernel:
    k(block = 1)
let _result = k.out
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Array(vec![Value::Float(9.0)].into()));
}

#[test]
fn three_axis_fixed_shape_row_major_offset_is_correct() {
    let src = r#"
kernel Cube:
    mut [float]'unified                                out
    mut [float, x = 2, y = 3, z = 4]'actor              vol

    init([float]'unified data):
        out = data

    def ():
        vol[x = 1, y = 2, z = 3] = 7.0
        # row-major: 1 + 2*2 + 3*(2*3) = 1 + 4 + 18 = 23
        out[0] = vol[x = 1, y = 2, z = 3]

let data = [0.0]
mut k = Cube(data)
kernel:
    k(block = 1)
let _result = k.out
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    assert_eq!(get_var(&interp, "_result"), Value::Array(vec![Value::Float(7.0)].into()));
}

#[test]
fn unassigned_fixed_shape_actor_field_defaults_to_zero() {
    // Parity with test_unassigned_actor_fixed_array_defaults_to_zero — a
    // fixed-shape 'actor field the kernel's init() never touches must still
    // be a real, zero-filled buffer, not Nil.
    let src = r#"
kernel Tile:
    mut [float]'unified                     input
    mut [float]'unified                     out
    mut [float, width = 2, height = 2]'actor tile

    init([float]'unified data):
        input = data
        out   = [0.0, 0.0, 0.0, 0.0]

    def ():
        let tid = gpu.thread.x
        out[tid] = tile[width = tid, height = 0] + input[tid]

let data = [1.0, 2.0, 3.0, 4.0]
mut k = Tile(data)
kernel:
    k(block = 2)
let _result = k.out
"#;
    let (interp, result) = run(src);
    result.expect("runtime error");
    // tile is zero everywhere it was never written, so out == input.
    assert_eq!(get_var(&interp, "_result"), Value::Array(vec![Value::Float(1.0), Value::Float(2.0), Value::Float(0.0), Value::Float(0.0)].into()));
}

#[test]
fn dynamic_shape_kernel_field_end_to_end_through_full_pipeline() {
    // Full pipeline test: desugar_labeled_array injects the shadow fields and
    // lowers LabeledIndex/.size() for the DYNAMIC-shape case (unlike the
    // fixed-shape tests above, this one can't rely on exec_kernel_decl's
    // self-contained lowering alone — the shadow fields have to already
    // exist on the KernelDecl by the time the kernel is even constructed).
    let src = r#"
kernel Grid:
    let [float, width, height]'global  src
    mut  [float]'unified               out

    init([float]'global s, uint w, uint h):
        src = s.reshape(width = w, height = h)
        out = [0.0 for ..(w * h)]

    def ():
        let tid = gpu.thread.x
        let w = src.size(.width)
        let row = tid / w
        let col = tid % w
        out[tid] = src[width = col, height = row] * 2.0

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
mut k = Grid(data, 3, 2)
kernel:
    k(block = 6)
let _result = k.out
"#;
    assert_eq!(
        run_with_labeled_array_desugar(src),
        Value::Array(vec![
            Value::Float(2.0), Value::Float(4.0), Value::Float(6.0),
            Value::Float(8.0), Value::Float(10.0), Value::Float(12.0),
        ].into())
    );
}

#[test]
fn dynamic_shape_kernel_field_in_a_use_imported_module_works() {
    // Regression pin for a real bug found migrating whisper-boring's
    // audio_gpu.br: `run_file`/`parse_and_merge_program` desugar the ENTRY
    // file, but a `use`-imported module is parsed and executed through the
    // interpreter's own separate runtime loader (`Interpreter::exec_use`,
    // interpreter/mod.rs) — which was never updated to also run
    // `desugar_labeled_array`. A dynamic-shape kernel field defined in the
    // imported module therefore reached eval_gpu.rs's *fixed*-shape-only
    // lowering still holding an un-desugared Type::LabeledArray, panicking
    // ("fixed-shape kernel field: every axis has Some(size) by construction")
    // the moment the module was merely `use`d — no construction or dispatch
    // even required to trigger it.
    let dir = std::env::temp_dir().join(format!(
        "boring_use_import_test_{}_{}",
        std::process::id(),
        line!(),
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    std::fs::write(dir.join("grid_kernel.br"), r#"
kernel Grid:
    let [float, width, height]'global src

    init([float]'global s, uint w, uint h):
        src = s.reshape(width = w, height = h)

    def ():
        let w = src.size(.width)
        let tid = gpu.thread.x
        if tid < w:
            pass
"#).expect("write module file");

    let entry_src = "use grid_kernel\n";
    let tokens = crate::lexer::lex(entry_src).expect("lex error");
    let program = crate::parser::parse(tokens).expect("parse error");
    let program = crate::desugar_labeled_array::desugar_labeled_array(program);

    let mut interp = Interpreter::new();
    interp.add_search_path(dir.clone());
    interp.exec_program(&program).expect("runtime error — see this test's own doc comment");

    let _ = std::fs::remove_dir_all(&dir);
}
