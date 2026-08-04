use super::*;
use crate::lexer::lex;
use crate::parser::parse;
use crate::interpreter::{Interpreter, Value};

fn desugared(src: &str) -> Program {
    let tokens = lex(src).expect("lex error");
    let program = parse(tokens).expect("parse error");
    desugar_labeled_array(program)
}

fn only_kernel(program: &Program) -> &KernelDecl {
    program.items.iter().find_map(|i| match i {
        Item::Kernel(k) => Some(k),
        _ => None,
    }).expect("expected exactly one kernel decl")
}

/// Full pipeline for an execution test: lex -> parse -> desugar_labeled_array
/// -> interpret. Deliberately NOT `interpreter::tests::run_src` — that
/// helper skips every desugar pass entirely (fine for ordinary language
/// tests, since desugar_image_volume/desugar_labeled_array are no-ops on a
/// program that doesn't use either feature — but exactly the thing under
/// test here).
fn run_desugared(src: &str) -> Value {
    let program = desugared(src);
    let mut interp = Interpreter::new();
    interp.exec_program(&program).expect("runtime error");
    let val = interp.global.borrow().get("_result").unwrap_or(Value::Nil);
    val
}

// ─── Kernel fields (dynamic shape) ──────────────────────────────────────────

#[test]
fn dynamic_kernel_field_becomes_buffer_plus_positional_shadow_fields() {
    let src = r#"
kernel Grid:
    mut [float, width, height]'unified img
    def ():
        img[0] = 1.0
"#;
    let program = desugared(src);
    let k = only_kernel(&program);
    let names: Vec<&str> = k.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, vec!["img", "__img_axis0", "__img_axis1"]);
    assert!(matches!(&k.fields[0].ty, Type::Array(inner) if matches!(&**inner, Type::Named(n) if n == "float")));
    // Type::Int, not Uint — see desugar_kernel_decl's own note (matches a
    // shadow value's typical `int`-typed source, and the Int/Int bounds a
    // `for i in 0..field.size(.axis)` range requires).
    assert!(matches!(k.fields[1].ty, Type::Int));
    assert!(matches!(k.fields[1].binding, FieldBinding::Let));
    assert!(matches!(k.fields[1].qual, GpuQual::Const));
    assert!(matches!(k.fields[2].ty, Type::Int));
}

#[test]
fn fixed_shape_kernel_field_is_untouched() {
    let src = r#"
kernel Tile:
    mut [float, width = 16, height = 16]'actor tile
    def ():
        pass
"#;
    let program = desugared(src);
    let k = only_kernel(&program);
    assert_eq!(k.fields.len(), 1);
    assert_eq!(k.fields[0].name, "tile");
    assert!(matches!(&k.fields[0].ty, Type::LabeledArray(_, axes) if axes.len() == 2));
}

#[test]
fn labeled_index_in_kernel_method_lowers_to_row_major_index() {
    let src = r#"
kernel Grid:
    mut [float, width, height]'unified img
    def ():
        img[width = 2, height = 3] = 1.0
"#;
    let program = desugared(src);
    let k = only_kernel(&program);
    let Stmt::Expr(e) = &k.methods[0].body[0] else { panic!("expected Stmt::Expr") };
    let ExprKind::Assign(lhs, _) = &e.kind else { panic!("expected Assign") };
    // 2 + 3 * __img_axis0  (width=axis0=fastest-varying, so its own index has
    // no stride factor; height=axis1 is multiplied by axis0's shadow size).
    let ExprKind::Index(obj, offset) = &lhs.kind else { panic!("expected Index, got {:?}", lhs.kind) };
    assert!(matches!(&obj.kind, ExprKind::Var(n) if n == "img"));
    let ExprKind::BinOp(BinOp::Add, left, right) = &offset.kind else { panic!("expected Add, got {:?}", offset.kind) };
    assert!(matches!(left.kind, ExprKind::Int(2)));
    let ExprKind::BinOp(BinOp::Mul, factor, stride) = &right.kind else { panic!("expected Mul, got {:?}", right.kind) };
    assert!(matches!(factor.kind, ExprKind::Int(3)));
    assert!(matches!(&stride.kind, ExprKind::Var(n) if n == "__img_axis0"));
}

#[test]
fn size_call_in_kernel_method_resolves_to_shadow_var() {
    let src = r#"
kernel Grid:
    mut [float, width, height]'unified img
    def ():
        let uint w = img.size(.width)
"#;
    let program = desugared(src);
    let k = only_kernel(&program);
    let Stmt::Let(l) = &k.methods[0].body[0] else { panic!("expected Stmt::Let") };
    let value = l.value.as_ref().expect("expected initializer");
    assert!(matches!(&value.kind, ExprKind::Var(n) if n == "__img_axis0"));
}

// ─── `let`-declared locals ───────────────────────────────────────────────────

#[test]
fn chained_for_comprehension_lowers_to_alloc_plus_nested_loops() {
    let src = r#"
let [float, width, height] a = [width for width in ..2 for height in ..3]
"#;
    let program = desugared(src);
    let Item::Let(s) = &program.items[0] else { panic!("expected Item::Let") };
    let value = s.value.as_ref().expect("initializer");
    let ExprKind::Block(stmts) = &value.kind else { panic!("expected Block, got {:?}", value.kind) };
    // declare tmp (deferred) ; tmp = alloc ; for height in ..3: for width in
    // ..2: buf[...] = width ; buf — the declare/alloc split (rather than a
    // single `let tmp = alloc`) is deliberate: it lets `labeled_comp_fill_
    // stmts` share the exact same alloc+loop shape with the
    // kernel-field-reassignment path, which assigns into an EXISTING name
    // (no `let` involved) — see that function's own doc comment.
    assert_eq!(stmts.len(), 4, "expected declare, alloc, outer-for, final value: {stmts:?}");
    assert!(matches!(&stmts[0], Stmt::Let(l) if l.value.is_none()), "expected a deferred-init declare, got {:?}", stmts[0]);
    let Stmt::Expr(alloc) = &stmts[1] else { panic!("expected alloc Expr, got {:?}", stmts[1]) };
    let ExprKind::Assign(_, alloc_rhs) = &alloc.kind else { panic!("expected Assign") };
    assert!(matches!(alloc_rhs.kind, ExprKind::ArrayAlloc { .. }));
    // Outer loop is clauses[1] (height) — clauses[0] (width) must be innermost per D2.
    let Stmt::For(outer) = &stmts[2] else { panic!("expected outer For, got {:?}", stmts[2]) };
    assert_eq!(outer.vars, vec!["height".to_string()]);
    let Stmt::For(inner) = &outer.body[0] else { panic!("expected inner For, got {:?}", outer.body[0]) };
    assert_eq!(inner.vars, vec!["width".to_string()]);
    // Also confirms the top-level `let a` itself is now tracked with 2 dynamic
    // shadow lets spliced in right after it.
    assert_eq!(program.items.len(), 3, "expected let a, __a_axis0, __a_axis1");
    // Shadow-let values are wrapped in an explicit `as int` (Type::Int, not
    // Uint — see desugar_let's own doc note) — unwrap the Cast to check
    // the literal.
    let unwrap_int = |e: &Expr| -> i64 {
        let ExprKind::Cast(inner, ty) = &e.kind else { panic!("expected an explicit Cast, got {:?}", e.kind) };
        assert_eq!(*ty, Type::Int);
        let ExprKind::Int(n) = inner.kind else { panic!("expected an Int literal inside the cast, got {:?}", inner.kind) };
        n
    };
    let Item::Let(shadow0) = &program.items[1] else { panic!("expected shadow let") };
    assert_eq!(shadow0.name, "__a_axis0");
    assert_eq!(shadow0.ty, Some(Type::Int));
    assert_eq!(unwrap_int(shadow0.value.as_ref().unwrap()), 2);
    let Item::Let(shadow1) = &program.items[2] else { panic!("expected shadow let") };
    assert_eq!(shadow1.name, "__a_axis1");
    assert_eq!(shadow1.ty, Some(Type::Int));
    assert_eq!(unwrap_int(shadow1.value.as_ref().unwrap()), 3);
}

#[test]
fn labeled_array_type_inferred_with_no_annotation() {
    // No `[float, width, height]` annotation at all — shape must still be
    // inferred from the comprehension itself (design doc's own example).
    let src = "let a = [width for width in ..2 for height in ..3]";
    let program = desugared(src);
    assert_eq!(program.items.len(), 3, "expected let a, __a_axis0, __a_axis1 (inferred)");
}

#[test]
fn relabel_cast_reuses_source_shadow_bindings_no_new_lets() {
    let src = r#"
let [float, width, height] a = [width for width in ..2 for height in ..3]
let b = a as [line = width, column = height]
"#;
    let program = desugared(src);
    // 3 items for `a` (let + 2 shadows) + exactly 1 for `b` (RelabelCast
    // desugars to a bare passthrough — no new shadow lets synthesized).
    assert_eq!(program.items.len(), 4, "expected no new shadow lets for b: {:#?}", program.items);
    let Item::Let(b) = &program.items[3] else { panic!("expected let b") };
    assert!(matches!(&b.value.as_ref().unwrap().kind, ExprKind::Var(n) if n == "a"), "RelabelCast should desugar to a bare passthrough");
}

#[test]
fn relabeled_indexing_resolves_using_source_shadow_bindings() {
    let src = r#"
let [float, width, height] a = [width + height * 10.0 for width in ..2 for height in ..3]
let b = a as [line = width, column = height]
let _result = b[line = 1, column = 2]
"#;
    // a[width=1,height=2] = 1 + 2*10 = 21.0, and b addresses the same buffer.
    assert_eq!(run_desugared(src), Value::Float(21.0));
}

// ─── End-to-end execution (real row-major math + D2 ordering) ─────────────

#[test]
fn comprehension_then_labeled_index_round_trips_row_major() {
    let src = r#"
let [float, width, height] a = [width + height * 10.0 for width in ..3 for height in ..4]
let _result = a[width = 2, height = 3]
"#;
    assert_eq!(run_desugared(src), Value::Float(32.0));
}

#[test]
fn size_call_on_local_returns_correct_axis_value() {
    let src = r#"
let [float, width, height] a = [0.0 for width in ..3 for height in ..4]
let _result = a.size(.height)
"#;
    // Value::Int, not Uint — dynamic-shape shadow bindings are Type::Int
    // (see desugar_kernel_decl's note: matches a shadow value's typical
    // int/uint-mixed source uniformly, and the Int/Int bounds a `for i in
    // 0..field.size(.axis)` range requires).
    assert_eq!(run_desugared(src), Value::Int(4));
}

#[test]
fn reshape_threads_shadow_values_through_labeled_indexing() {
    let src = r#"
let flat = [i for i in ..6]
let a = flat.reshape(width = 2, height = 3)
let _result = a[width = 1, height = 2]
"#;
    // row-major: width fastest-varying -> flat index = 1 + 2*2 = 5
    assert_eq!(run_desugared(src), Value::Int(5));
}

#[test]
fn fixed_shape_local_labeled_index_resolves_at_desugar_time() {
    let src = r#"
let [float, width = 3, height = 4] a = [width + height * 10.0 for width in ..3 for height in ..4]
let _result = a[width = 2, height = 3]
"#;
    assert_eq!(run_desugared(src), Value::Float(32.0));
}

// ─── Kernel field construction sugar: `field = flat.reshape(...)` ─────────
//
// Unlike a local `let`, a kernel field is pre-declared (KernelFieldDecl, not
// a `let`) — its shape only shows up later, as a plain re-assignment inside
// `init()`. Mirrors desugar_image_volume's `field = Image(data, w, h)`
// construction-sugar expansion, one axis assignment per shadow field.

#[test]
fn reassigning_a_dynamic_kernel_field_via_reshape_expands_to_shadow_assigns() {
    let src = r#"
kernel Grid:
    let [float, width, height]'global src
    def (uint w, uint h):
        src = src.reshape(width = w, height = h)
"#;
    let program = desugared(src);
    let k = only_kernel(&program);
    assert_eq!(k.methods[0].body.len(), 3, "expected src=..., __src_axis0=w, __src_axis1=h: {:#?}", k.methods[0].body);
    let names_assigned: Vec<&str> = k.methods[0].body.iter().map(|s| {
        let Stmt::Expr(e) = s else { panic!("expected Stmt::Expr") };
        let ExprKind::Assign(lhs, _) = &e.kind else { panic!("expected Assign") };
        let ExprKind::Var(name) = &lhs.kind else { panic!("expected Var lhs") };
        name.as_str()
    }).collect();
    assert_eq!(names_assigned, vec!["src", "__src_axis0", "__src_axis1"]);
    let Stmt::Expr(shadow0) = &k.methods[0].body[1] else { unreachable!() };
    let ExprKind::Assign(_, rhs) = &shadow0.kind else { unreachable!() };
    // Wrapped in an explicit `as int` — the param here is `uint`-typed
    // (init params are `int` or `uint` in real code), so this also pins
    // the cast normalizing a uint-typed source down to a definite Int, not
    // just the (also-tested) int-typed case elsewhere.
    let ExprKind::Cast(inner, ty) = &rhs.kind else { panic!("expected an explicit Cast, got {:?}", rhs.kind) };
    assert!(matches!(&inner.kind, ExprKind::Var(n) if n == "w"));
    assert_eq!(*ty, Type::Int);
}

#[test]
fn reassigning_with_an_unrecognized_rhs_shape_is_left_unresolved() {
    // No `.reshape(...)` — this pass can't determine axis values, so the
    // reassignment passes through as a single plain statement rather than
    // guessing. (Downstream `.size()`/indexing on `src` after this point
    // would simply stay unresolved too — a documented best-effort limit,
    // not a silent miscompile.)
    let src = r#"
kernel Grid:
    let [float, width, height]'global src
    def ():
        src = [0.0, 0.0]
"#;
    let program = desugared(src);
    let k = only_kernel(&program);
    assert_eq!(k.methods[0].body.len(), 1);
}

// ─── Fill shorthand end-to-end: [value for n] / [value for w=.., h=..] ─────
// Pure parser sugar over ArrayFill/LabeledArrayComp (see
// src/parser/tests_labeled_array.rs for the structural parser tests) — these
// verify the *values* actually come out right once the whole pipeline runs,
// not just that the right AST node gets built.

#[test]
fn bare_1d_fill_shorthand_produces_correct_values() {
    let src = r#"
let n = 4
let a = [7.0 for n]
let _result = a[2]
"#;
    assert_eq!(run_desugared(src), Value::Float(7.0));
}

#[test]
fn labeled_shape_fill_shorthand_round_trips_row_major() {
    let src = r#"
let a = [9.0 for width = 3, height = 4]
let _result = a[width = 2, height = 3]
"#;
    // The fill value is constant, so any in-range index must read it back —
    // this also exercises the shape (width=3, height=4) actually being
    // threaded through correctly (shadow bindings synthesized from the
    // clause counts, same as the general chained-for comprehension).
    assert_eq!(run_desugared(src), Value::Float(9.0));
}

#[test]
fn labeled_shape_fill_shorthand_infers_correct_axis_sizes() {
    let src = r#"
let a = [0.0 for width = 3, height = 4]
let _result = a.size(.height)
"#;
    assert_eq!(run_desugared(src), Value::Int(4));
}
