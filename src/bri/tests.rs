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

use super::*;

fn parse(src: &str) -> Program {
    let tokens = crate::lexer::lex_all(src).expect("lex");
    crate::parser::parse(tokens).unwrap_or_else(|e| panic!("parse: {}", e.msg()))
}

fn print(src: &str) -> String {
    print_program(&parse(src)).unwrap_or_else(|e| panic!("{e}"))
}

/// print(parse(x)) is a fixpoint: the output parses back to something that prints identically.
fn assert_fixpoint(src: &str) -> String {
    let once = print(src);
    let twice = print(&once);
    assert_eq!(once, twice, "not a fixpoint:\n{once}");
    once
}

#[test]
fn qualifiers_are_written_in_full() {
    let out = assert_fixpoint(
        "struct Counter:\n    var int n = 0\n\
         def f(Counter'actor a, Counter'owned b, Counter'shared'weak c, Counter& d, var Counter& e):\n    pass\n",
    );
    // `var Counter&` is the mutable borrow: the keyword says it, the type stays `Counter&`
    for want in ["Counter'actor a", "Counter'owned b", "Counter'shared'weak c", "Counter& d", "var Counter& e"] {
        assert!(out.contains(want), "{want} missing from:\n{out}");
    }
}

#[test]
fn mut_binding_keywords_round_trip() {
    let out = assert_fixpoint(
        "struct C:\n    var int n = 0\n    def inc(): n += 1\n\
         def g():\n    let a = C()\n    mut b = C()\n    var c = C()\n    var mut d = C()\n    b.inc()\n    d.inc()\n",
    );
    for want in ["let a = C()", "mut b = C()", "var c = C()", "var mut d = C()"] {
        assert!(out.contains(want), "{want} missing from:\n{out}");
    }
}

#[test]
fn implicit_generics_are_written_untyped_with_their_bounds_as_a_comment() {
    let out = assert_fixpoint("def show(x):\n    print \"{x}\"\n");
    assert!(out.contains("def show(x):  # inferred: T_x as Display"), "{out}");
    // the comment is a trailing one: the lexer drops it, so it never reaches the generated Rust
    assert!(!out.contains("<T_x"), "{out}");
}

#[test]
fn compound_assignment_is_folded_back() {
    let out = assert_fixpoint("def f():\n    var x = 1\n    x += 2\n    x = x * 3\n    x <<= 1\n");
    assert!(out.contains("x += 2"), "{out}");
    assert!(out.contains("x *= 3"), "{out}");
    assert!(out.contains("x <<= 1"), "{out}");
}

#[test]
fn precedence_is_kept_with_minimal_parentheses() {
    let out = assert_fixpoint("def f(int a, int b, int c):\n    print(a + b * c)\n    print((a + b) * c)\n    print(not (a == b))\n    print(-(a + b))\n");
    assert!(out.contains("a + b * c"), "{out}");
    assert!(out.contains("(a + b) * c"), "{out}");
    assert!(out.contains("not a == b"), "{out}");
    assert!(out.contains("-(a + b)"), "{out}");
}

#[test]
fn cast_receivers_and_comparison_operands_are_parenthesised() {
    let out = assert_fixpoint("def f(float x):\n    print((x as int).abs())\n    print((x as int) < 3)\n");
    assert!(out.contains("(x as int).abs()"), "{out}");
    assert!(out.contains("(x as int) < 3"), "{out}");
}

#[test]
fn strings_escape_braces_quotes_and_newlines() {
    let out = assert_fixpoint("def f(int n):\n    print \"a {{b}} \\\"q\\\" {n:.2f}\\n\"\n");
    assert!(out.contains("{{b}}"), "{out}");
    assert!(out.contains("{n:.2f}"), "{out}");
    assert!(out.contains("\\n"), "{out}");
}

#[test]
fn control_flow_and_patterns_round_trip() {
    assert_fixpoint(
        "enum Shape:\n    Circle(float r)\n    Rect(float w, float h)\n\
         float area(Shape s):\n    match s:\n        Circle(r): 3.0 * r * r\n        Rect(w, h): w * h\n\
         def loops():\n    for i in 0..<3:\n        if i == 1:\n            continue\n        elif i == 2:\n            break\n        else:\n            print i\n    var k = 0\n    while k < 3:\n        k += 1\n",
    );
}

#[test]
fn unknown_constructs_are_errors_naming_them() {
    // a slice range is only an index
    let bad = Expr { kind: ExprKind::SliceRange { start: None, end: None, inclusive: false }, line: 7, col: 0, len: 0 };
    let program = Program { items: vec![Item::Stmt(Stmt::Expr(bad))] };
    let err = print_program(&program).unwrap_err();
    assert!(err.what.contains("slice range"), "{err}");
    assert_eq!(err.line, 7);
    // a qualifier argument has no label outside its declaration
    let ty = Type::Generic("Board".into(), vec![Type::QualArg(vec![OwnerQual::Actor])]);
    let let_stmt = LetStmt {
        binding: BindingKind::Let, is_pub: false, is_static: false, name: "b".into(), ty: Some(ty), var_mut: false,
        value: None, is_lazy: false, is_resource: false, line: 1, col: 0,
    };
    let err = print_program(&Program { items: vec![Item::Let(let_stmt)] }).unwrap_err();
    assert!(err.what.contains("qualifier argument of `Board`"), "{err}");
}

#[test]
fn qualifier_generic_declarations_label_their_arguments() {
    let out = assert_fixpoint(
        "struct Board<'elem>:\n    var [int] cells = []\n\
         int count(Board<elem'actor> b):\n    return 0\n",
    );
    assert!(out.contains("Board<elem'actor> b"), "{out}");
}
