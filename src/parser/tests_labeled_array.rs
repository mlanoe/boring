// Parser tests for labeled multi-dimensional arrays
// (docs/array-multidim-proposal.md). Covers the four grammar additions:
// type declaration, labeled indexing, chained-for comprehension, and the
// `as [...]` cross-label mapping — plus regression checks proving the
// pre-existing `[T]` / `[T, N]` / `[T, <expr>]` / single-`for` comprehension
// forms are completely unaffected.

use crate::ast::{self, ExprKind};

fn parse_program(src: &str) -> ast::Program {
    let tokens = crate::lexer::lex(src).expect("lex");
    crate::parser::parse(tokens).expect("parse")
}

fn parse_err(src: &str) -> String {
    let tokens = crate::lexer::lex(src).expect("lex");
    match crate::parser::parse(tokens) {
        Ok(p) => panic!("expected parse error, got program: {:?}", p),
        Err(e) => format!("{:?}", e),
    }
}

fn let_ty(program: &ast::Program) -> &ast::Type {
    match &program.items[0] {
        ast::Item::Let(l) => l.ty.as_ref().expect("expected type annotation"),
        other => panic!("expected top-level let, got {:?}", other),
    }
}

fn let_value(program: &ast::Program) -> &ast::Expr {
    match &program.items[0] {
        ast::Item::Let(l) => l.value.as_ref().expect("expected initializer"),
        other => panic!("expected top-level let, got {:?}", other),
    }
}

// ─── Type declaration ────────────────────────────────────────────────────

#[test]
fn dynamic_labeled_array_type() {
    let program = parse_program("let [float, width, height] a");
    match let_ty(&program) {
        ast::Type::LabeledArray(elem, axes) => {
            // `float` is a lowercase alias — parses as Named("float"), resolved
            // to Type::Float64 later (see book.md's "Common types" table).
            assert!(matches!(&**elem, ast::Type::Named(n) if n == "float"));
            assert_eq!(axes.len(), 2);
            assert_eq!(axes[0].label, "width");
            assert!(axes[0].size.is_none());
            assert_eq!(axes[1].label, "height");
            assert!(axes[1].size.is_none());
        }
        other => panic!("expected LabeledArray, got {:?}", other),
    }
}

#[test]
fn fixed_labeled_array_type() {
    let program = parse_program("let [float, width = 16, height = 16] a");
    match let_ty(&program) {
        ast::Type::LabeledArray(elem, axes) => {
            assert!(matches!(&**elem, ast::Type::Named(n) if n == "float"));
            assert_eq!(axes.len(), 2);
            assert_eq!(axes[0].label, "width");
            assert!(matches!(&axes[0].size, Some(ast::ConstExpr(e)) if matches!(e.kind, ExprKind::Int(16))));
            assert_eq!(axes[1].label, "height");
            assert!(matches!(&axes[1].size, Some(ast::ConstExpr(e)) if matches!(e.kind, ExprKind::Int(16))));
        }
        other => panic!("expected LabeledArray, got {:?}", other),
    }
}

#[test]
fn three_axis_labeled_array_type() {
    let program = parse_program("let [float, width, height, depth] a");
    match let_ty(&program) {
        ast::Type::LabeledArray(_, axes) => {
            assert_eq!(axes.iter().map(|a| a.label.as_str()).collect::<Vec<_>>(), ["width", "height", "depth"]);
        }
        other => panic!("expected LabeledArray, got {:?}", other),
    }
}

#[test]
fn fixed_axis_size_can_be_a_const_generic_expression() {
    // Reuses the same const_expr machinery as ArrayNExpr (`[T, W * H]`).
    let program = parse_program("kernel K:\n    let [float, width = W, height = H * 2] a\n    def ():\n        pass");
    if let ast::Item::Kernel(k) = &program.items[0] {
        match &k.fields[0].ty {
            ast::Type::LabeledArray(_, axes) => {
                assert!(matches!(&axes[0].size, Some(ast::ConstExpr(e)) if matches!(&e.kind, ExprKind::Var(n) if n == "W")));
                assert!(matches!(&axes[1].size, Some(ast::ConstExpr(e)) if matches!(e.kind, ExprKind::BinOp(..))));
            }
            other => panic!("expected LabeledArray field, got {:?}", other),
        }
    } else {
        panic!("expected kernel item");
    }
}

#[test]
fn qualifier_wraps_the_whole_labeled_array_type() {
    // Qualifier placement after the closing bracket falls out for free from
    // parse_type_qualifier already running once after parse_type_base returns
    // — see docs/array-multidim-proposal.md, "Qualifiers".
    let program = parse_program("kernel K:\n    let [float, width, height]'global a\n    def ():\n        pass");
    if let ast::Item::Kernel(k) = &program.items[0] {
        // parse_kernel_field strips the qualifier into `qual`/`ty` — the field's
        // `ty` should be the bare LabeledArray, `qual` should reflect 'global.
        assert!(matches!(&k.fields[0].ty, ast::Type::LabeledArray(_, _)));
        assert_eq!(k.fields[0].qual, ast::GpuQual::Global);
    } else {
        panic!("expected kernel item");
    }
}

#[test]
fn kernel_field_qualifier_inferred_from_binding_when_unqualified() {
    let program = parse_program(
        "kernel K:\n    let [float, width, height] a\n    mut [float, width, height] b\n    def ():\n        pass"
    );
    if let ast::Item::Kernel(k) = &program.items[0] {
        assert_eq!(k.fields[0].qual, ast::GpuQual::Const); // let -> Const
        assert_eq!(k.fields[1].qual, ast::GpuQual::Local); // mut -> Local
    } else {
        panic!("expected kernel item");
    }
}

// ─── Regression: legacy `[T, N]` / `[T, <expr>]` unaffected ─────────────

#[test]
fn legacy_array_n_literal_unaffected() {
    let program = parse_program("let [float, 4] a");
    assert!(matches!(let_ty(&program), ast::Type::ArrayN(elem, 4) if matches!(&**elem, ast::Type::Named(n) if n == "float")));
}

#[test]
fn legacy_const_generic_reference_unaffected() {
    // A single bare identifier after the comma keeps meaning "reference to an
    // existing const generic param" — never reinterpreted as a 1-axis labeled
    // array (LabeledArray always has 2+ axes).
    let program = parse_program("let [float, N] a");
    match let_ty(&program) {
        ast::Type::ArrayNExpr(elem, ast::ConstExpr(e)) => {
            assert!(matches!(&**elem, ast::Type::Named(n) if n == "float"));
            assert!(matches!(e.kind, ExprKind::Var(ref n) if n == "N"));
        }
        other => panic!("expected ArrayNExpr(_, Var(N)), got {:?}", other),
    }
}

#[test]
fn legacy_const_generic_arithmetic_expression_unaffected() {
    let program = parse_program("let [float, N + 1] a");
    assert!(matches!(let_ty(&program), ast::Type::ArrayNExpr(_, ast::ConstExpr(e)) if matches!(e.kind, ExprKind::BinOp(..))));
}

// ─── Type declaration: malformed input ──────────────────────────────────

#[test]
fn mixed_fixed_and_dynamic_axes_is_a_parse_error() {
    let msg = parse_err("let [float, width, height = 16] a");
    assert!(msg.contains("all dynamic") || msg.contains("all fixed"), "unexpected message: {msg}");
}

#[test]
fn single_fixed_axis_is_a_parse_error_not_silently_reinterpreted() {
    // Must NOT roll back and let `width = 16` be silently parsed as an
    // assignment expression under the legacy ArrayNExpr path.
    let msg = parse_err("let [float, width = 16] a");
    assert!(msg.contains("at least 2 axes"), "unexpected message: {msg}");
}

#[test]
fn single_dynamic_axis_with_trailing_comma_is_a_parse_error() {
    let msg = parse_err("let [float, width,] a");
    assert!(msg.contains("at least 2 axes"), "unexpected message: {msg}");
}

// ─── Labeled indexing ────────────────────────────────────────────────────

#[test]
fn labeled_index_parses_order_free() {
    let program = parse_program("let v = a[width = w, height = h]");
    match &let_value(&program).kind {
        ExprKind::LabeledIndex(obj, args) => {
            assert!(matches!(&obj.kind, ExprKind::Var(n) if n == "a"));
            assert_eq!(args.len(), 2);
            assert_eq!(args[0].label.as_deref(), Some("width"));
            assert_eq!(args[1].label.as_deref(), Some("height"));
        }
        other => panic!("expected LabeledIndex, got {:?}", other),
    }
}

#[test]
fn labeled_index_reversed_order_still_parses() {
    // Order-free at the use site — no positional requirement (see design doc).
    let program = parse_program("let v = a[height = h, width = w]");
    match &let_value(&program).kind {
        ExprKind::LabeledIndex(_, args) => {
            assert_eq!(args[0].label.as_deref(), Some("height"));
            assert_eq!(args[1].label.as_deref(), Some("width"));
        }
        other => panic!("expected LabeledIndex, got {:?}", other),
    }
}

#[test]
fn plain_index_unaffected() {
    let program = parse_program("let v = a[i]");
    assert!(matches!(&let_value(&program).kind, ExprKind::Index(..)));
}

#[test]
fn equality_inside_index_not_misdetected_as_labeled() {
    // `a[i == j]` must stay a plain Index, not a labeled index — guarded by
    // the same is_double_eq check parse_arg uses for labeled call args.
    let program = parse_program("let v = a[i == j]");
    match &let_value(&program).kind {
        ExprKind::Index(_, idx) => assert!(matches!(idx.kind, ExprKind::BinOp(ast::BinOp::Eq, ..))),
        other => panic!("expected Index, got {:?}", other),
    }
}

#[test]
fn slice_range_index_unaffected() {
    let program = parse_program("let v = a[..n]");
    match &let_value(&program).kind {
        ExprKind::Index(_, idx) => assert!(matches!(idx.kind, ExprKind::SliceRange { .. })),
        other => panic!("expected Index with SliceRange, got {:?}", other),
    }
}

// ─── Chained-for comprehension ───────────────────────────────────────────

#[test]
fn chained_for_comprehension_two_axes() {
    let program = parse_program("let a = [ f(width, height) for width in ..W for height in ..H ]");
    match &let_value(&program).kind {
        ExprKind::LabeledArrayComp { clauses, .. } => {
            assert_eq!(clauses.len(), 2);
            assert_eq!(clauses[0].0, "width");
            assert_eq!(clauses[1].0, "height");
        }
        other => panic!("expected LabeledArrayComp, got {:?}", other),
    }
}

#[test]
fn chained_for_comprehension_three_axes() {
    let program = parse_program(
        "let a = [ f(width, height, depth) for width in ..W for height in ..H for depth in ..D ]"
    );
    match &let_value(&program).kind {
        ExprKind::LabeledArrayComp { clauses, .. } => {
            assert_eq!(clauses.iter().map(|(v, _)| v.as_str()).collect::<Vec<_>>(), ["width", "height", "depth"]);
        }
        other => panic!("expected LabeledArrayComp, got {:?}", other),
    }
}

#[test]
fn single_for_comprehension_unaffected() {
    // A single `for` clause must keep producing plain ArrayComp, not
    // LabeledArrayComp — no behavior change for today's 1D comprehensions.
    let program = parse_program("let a = [ i * i for i in ..5 ]");
    match &let_value(&program).kind {
        ExprKind::ArrayComp { var, .. } => assert_eq!(var, "i"),
        other => panic!("expected ArrayComp, got {:?}", other),
    }
}

#[test]
fn single_for_over_collection_unaffected() {
    let program = parse_program("let a = [ x * 2 for x in xs ]");
    assert!(matches!(&let_value(&program).kind, ExprKind::ArrayCompIter { .. }));
}

#[test]
fn fill_comprehension_unaffected() {
    let program = parse_program("let a = [ 0 for ..5 ]");
    assert!(matches!(&let_value(&program).kind, ExprKind::ArrayFill { .. }));
}

// ─── Fill shorthand (no bound variable) ──────────────────────────────────
//
// [value for n] / [value for width = w, height = h] — pure parser sugar
// over the existing ArrayFill/LabeledArrayComp nodes, no bound loop
// variables introduced. See docs/array-multidim-proposal.md.

#[test]
fn bare_fill_count_without_dots() {
    let program = parse_program("let a = [ 0.0 for n ]");
    match &let_value(&program).kind {
        ExprKind::ArrayFill { value, count } => {
            assert!(matches!(value.kind, ExprKind::Float(f) if f == 0.0));
            assert!(matches!(&count.kind, ExprKind::Var(n) if n == "n"));
        }
        other => panic!("expected ArrayFill, got {:?}", other),
    }
}

#[test]
fn bare_fill_count_accepts_an_arbitrary_expression() {
    let program = parse_program("let a = [ 0.0 for n * 2 ]");
    match &let_value(&program).kind {
        ExprKind::ArrayFill { count, .. } => {
            assert!(matches!(count.kind, ExprKind::BinOp(ast::BinOp::Mul, ..)));
        }
        other => panic!("expected ArrayFill, got {:?}", other),
    }
}

#[test]
fn dotted_and_bare_fill_counts_are_still_both_accepted() {
    // `..n` (existing) and a bare `n` (new) must both keep working — purely
    // additive, not a replacement.
    let dotted = parse_program("let a = [ 0.0 for ..n ]");
    let bare = parse_program("let b = [ 0.0 for n ]");
    assert!(matches!(&let_value(&dotted).kind, ExprKind::ArrayFill { .. }));
    assert!(matches!(&let_value(&bare).kind, ExprKind::ArrayFill { .. }));
}

#[test]
fn labeled_shape_fill_two_axes() {
    let program = parse_program("let a = [ 0.0 for width = w, height = h ]");
    match &let_value(&program).kind {
        ExprKind::LabeledArrayComp { expr, clauses } => {
            assert!(matches!(expr.kind, ExprKind::Float(f) if f == 0.0));
            assert_eq!(clauses.len(), 2);
            assert_eq!(clauses[0].0, "width");
            assert!(matches!(&clauses[0].1.kind, ExprKind::Var(n) if n == "w"));
            assert_eq!(clauses[1].0, "height");
            assert!(matches!(&clauses[1].1.kind, ExprKind::Var(n) if n == "h"));
        }
        other => panic!("expected LabeledArrayComp, got {:?}", other),
    }
}

#[test]
fn labeled_shape_fill_three_axes() {
    let program = parse_program("let a = [ 0.0 for x = 2, y = 3, z = 4 ]");
    match &let_value(&program).kind {
        ExprKind::LabeledArrayComp { clauses, .. } => {
            assert_eq!(clauses.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(), ["x", "y", "z"]);
        }
        other => panic!("expected LabeledArrayComp, got {:?}", other),
    }
}

#[test]
fn two_element_array_literal_unaffected_by_fill_shorthand() {
    // The exact collision this shorthand had to avoid: `[0.0, n]` (comma,
    // no `for`) must keep meaning a plain 2-element array literal.
    let program = parse_program("let a = [ 0.0, 2 ]");
    match &let_value(&program).kind {
        ExprKind::Array(elems) => {
            assert_eq!(elems.len(), 2);
            assert!(matches!(elems[0].kind, ExprKind::Float(f) if f == 0.0));
            assert!(matches!(elems[1].kind, ExprKind::Int(2)));
        }
        other => panic!("expected a plain Array literal, got {:?}", other),
    }
}

// ─── `as [...]` cross-label mapping ──────────────────────────────────────

#[test]
fn relabel_cast_parses_mapping_pairs() {
    let program = parse_program("let b = img as [line = width, column = height]");
    match &let_value(&program).kind {
        ExprKind::RelabelCast(obj, pairs) => {
            assert!(matches!(&obj.kind, ExprKind::Var(n) if n == "img"));
            assert_eq!(pairs, &[("line".to_string(), "width".to_string()), ("column".to_string(), "height".to_string())]);
        }
        other => panic!("expected RelabelCast, got {:?}", other),
    }
}

#[test]
fn plain_type_cast_unaffected() {
    let program = parse_program("let b = x as int");
    assert!(matches!(&let_value(&program).kind, ExprKind::Cast(_, ast::Type::Named(n)) if n == "int"));
}

#[test]
fn cast_to_labeled_array_type_is_a_real_cast_not_relabel() {
    // Starts with a *type* token (`float`), not `label =` — must stay Cast.
    let program = parse_program("let b = x as [float, width, height]");
    match &let_value(&program).kind {
        ExprKind::Cast(_, ast::Type::LabeledArray(_, axes)) => assert_eq!(axes.len(), 2),
        other => panic!("expected Cast to LabeledArray, got {:?}", other),
    }
}

#[test]
fn single_axis_relabel_cast_is_a_parse_error_not_silently_reinterpreted() {
    // Regression test: the `Type::LabeledArray` type-annotation form has
    // rejected a single axis at parse time from the start (see
    // `single_fixed_axis_is_a_parse_error_not_silently_reinterpreted` above),
    // but the `as [...]` relabel-cast expression form had no equivalent
    // check at all — `a as [x = width]` silently parsed as a one-pair
    // RelabelCast instead of failing loudly, an inconsistency with the type
    // form despite both representing the same "at least 2 axes" concept.
    let msg = parse_err("let b = img as [line = width]");
    assert!(msg.contains("at least 2 axes"), "unexpected message: {msg}");
}
