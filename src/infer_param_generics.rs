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

//! Phase 2 of implicit generics (`docs/design-notes/implicit-generics-draft.md`): a parameter
//! written without a type is a fresh type parameter of its function.
//!
//! ```text
//! def show(x):                    def show<T_x>(T_x x) where T_x as Display:
//!     print "value {x}"     ==>       print "value {x}"
//! ```
//!
//! The parser calls [`fresh_name`] while reading the signature and [`bounds`] once the body is
//! known. A type parameter has no operations until it is bounded, so the bounds are read off the
//! *direct* uses of the parameter in the body:
//!
//! - printed or interpolated (`print x`, `"{x}"`) → `Display`;
//! - `x == x2` / `x != x2` → `PartialEq`, `x < x2` and friends → `PartialOrd`;
//! - `x + x`, `x - x`, ... with the same parameter on both sides → `Add<Output = T>` (and so on) and
//!   `Copy`, since the operator consumes both operands.
//!
//! Anything else (passing `x` to another function, `x + 1`, a method call) adds no bound: the
//! generated Rust is then rejected by rustc with the missing bound named, and writing the
//! parameter's type is the way out. Bounds that follow a call into another generic function are
//! phase 4.

use crate::ast::*;
use crate::ast_walk::Walker;
use std::collections::BTreeSet;

/// Name of the type parameter standing for the untyped parameter `param`.
pub(crate) fn fresh_name(param: &str) -> String {
    format!("T_{param}")
}

/// `(type parameter, bound)` pairs for the untyped parameters `params` (their *parameter* names)
/// of a function with the given `body`, in the shape of `FnDecl::where_clause`. Also used for the
/// untyped fields of a struct (phase 3), read from its methods, where `x` and `self.x` both count.
pub(crate) fn bounds(params: &[String], body: &[Stmt]) -> Vec<(String, String)> {
    let mut found: BTreeSet<(String, String)> = BTreeSet::new();
    let mut body = body.to_vec();
    let is_param = |e: &Expr| -> Option<String> {
        match &e.kind {
            ExprKind::Var(n) if params.contains(n) => Some(n.clone()),
            // `self.field` inside a method of the struct that declares the untyped field.
            ExprKind::Field(base, n) if params.contains(n) && matches!(&base.kind, ExprKind::Var(s) if s == "self") => Some(n.clone()),
            _ => None,
        }
    };
    {
        let mut on_expr = |e: &mut Expr| match &e.kind {
            ExprKind::StringInterp(segs) => {
                for seg in segs {
                    let inner = match seg {
                        StringSegment::Expr(x) | StringSegment::FormattedExpr(x, _) => x,
                        StringSegment::Lit(_) => continue,
                    };
                    if let Some(n) = is_param(inner) {
                        found.insert((fresh_name(&n), "Display".into()));
                    }
                }
            }
            ExprKind::Call(callee, args) if matches!(&callee.kind, ExprKind::Var(f) if f == "print" || f == "println") => {
                for a in args {
                    if let Some(n) = is_param(&a.value) {
                        found.insert((fresh_name(&n), "Display".into()));
                    }
                }
            }
            ExprKind::BinOp(op, l, r) => {
                let (Some(a), Some(b)) = (is_param(l), is_param(r)) else { return };
                let t = fresh_name(&a);
                let same = a == b;
                let rhs = fresh_name(&b);
                match op {
                    BinOp::Eq | BinOp::NotEq => {
                        found.insert((t, if same { "PartialEq".into() } else { format!("PartialEq<{rhs}>") }));
                    }
                    BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                        found.insert((t, if same { "PartialOrd".into() } else { format!("PartialOrd<{rhs}>") }));
                    }
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem if same => {
                        let tr = match op {
                            BinOp::Add => "Add", BinOp::Sub => "Sub", BinOp::Mul => "Mul",
                            BinOp::Div => "Div", _ => "Rem",
                        };
                        found.insert((t.clone(), format!("std::ops::{tr}<Output = {t}>")));
                        found.insert((t, "Copy".into()));
                    }
                    _ => {}
                }
            }
            _ => {}
        };
        let mut on_stmt = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
    }
    found.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    use crate::parser;

    fn first_fn(src: &str) -> FnDecl {
        let tokens = lex(src).expect("lex");
        let program = parser::parse(tokens).expect("parse");
        program.items.iter().find_map(|i| match i {
            Item::Fn(f) => Some(f.clone()),
            _ => None,
        }).expect("a function")
    }

    #[test]
    fn untyped_parameter_becomes_a_type_parameter() {
        let f = first_fn("def keep(x):\n    return x\n");
        assert_eq!(f.type_params, vec!["T_x".to_string()]);
        assert_eq!(f.params[0].ty, Some(Type::TypeParam("T_x".into())));
        assert_eq!(f.return_ty, Some(Type::TypeParam("T_x".into())));
    }

    #[test]
    fn typed_and_untyped_parameters_mix() {
        let f = first_fn("def pair(int n, x):\n    return x\n");
        assert_eq!(f.type_params, vec!["T_x".to_string()]);
        assert_eq!(f.params[0].ty, Some(Type::Named("int".into())));
    }

    #[test]
    fn interpolation_and_print_add_display() {
        let f = first_fn("def show(x, y):\n    print \"v {x}\"\n    print y\n");
        assert!(f.where_clause.contains(&("T_x".into(), "Display".into())));
        assert!(f.where_clause.contains(&("T_y".into(), "Display".into())));
    }

    #[test]
    fn comparison_and_arithmetic_bounds() {
        let f = first_fn("def f(a, b):\n    if a < b:\n        print \"x\"\n    return a + a\n");
        assert!(f.where_clause.contains(&("T_a".into(), "PartialOrd<T_b>".into())));
        assert!(f.where_clause.contains(&("T_a".into(), "std::ops::Add<Output = T_a>".into())));
        assert!(f.where_clause.contains(&("T_a".into(), "Copy".into())));
    }

    #[test]
    fn unused_parameter_gets_no_bound() {
        let f = first_fn("def ignore(x):\n    print \"hi\"\n");
        assert!(f.where_clause.is_empty());
    }

    fn first_struct(src: &str) -> StructDecl {
        let tokens = lex(src).expect("lex");
        let program = parser::parse(tokens).expect("parse");
        program.items.iter().find_map(|i| match i {
            Item::Struct(s) => Some(s.clone()),
            _ => None,
        }).expect("a struct")
    }

    #[test]
    fn untyped_field_becomes_a_struct_type_parameter() {
        let s = first_struct("struct Response:\n    status\n    string text\n    let payload\n");
        assert_eq!(s.type_params, vec!["T_status".to_string(), "T_payload".to_string()]);
        assert_eq!(s.fields[0].ty, Type::TypeParam("T_status".into()));
        assert_eq!(s.fields[1].ty, Type::Named("string".into()));
        assert_eq!(s.fields[2].ty, Type::TypeParam("T_payload".into()));
    }

    #[test]
    fn untyped_field_is_bounded_by_its_method_uses() {
        let s = first_struct("struct B:\n    v\n    w\n\n    req string show():\n        return \"{v}\"\n\n    req bool same():\n        return self.w == self.w\n");
        assert!(s.where_clause.contains(&("T_v".into(), "Display".into())));
        assert!(s.where_clause.contains(&("T_w".into(), "PartialEq".into())));
    }

    #[test]
    fn untyped_field_with_a_default_still_needs_a_type() {
        let tokens = lex("struct B:\n    retries = 0\n").expect("lex");
        let e = parser::parse(tokens).err().expect("rejected").to_string();
        assert!(e.contains("default value but no type"), "{e}");
    }

    #[test]
    fn typed_fields_are_untouched() {
        let s = first_struct("struct P:\n    int x\n    float y\n");
        assert!(s.type_params.is_empty());
    }

    #[test]
    fn methods_still_require_typed_parameters() {
        let tokens = lex("struct S:\n    int n\n\ndef S.get(x):\n    return n\n").expect("lex");
        assert!(parser::parse(tokens).is_err());
    }
}
