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

//! Phase 2 of implicit generics (docs/book.md, "Implicit generics: design notes"): a parameter
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
//! - `x > 0`, `x + 1` with a numeric literal on the right → `PartialOrd<isize>`, `Add<isize, Output = T>` + `Copy`
//!   (`float` literals: `f64`).
//!
//! Passing `x` as is to another top-level function carries that function's bounds over
//! ([`propagate_bounds`]). Anything else (`x + 1`, a method call, a call this pass cannot see)
//! adds no bound: the generated Rust is then rejected by rustc with the missing bound named, and
//! writing the parameter's type is the way out.

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
            ExprKind::MethodCall(recv, m, a) if m == "len" && a.is_empty() => {
                if let Some(n) = is_param(recv) {
                    found.insert((fresh_name(&n), "BoringLen".into()));
                }
            }
            ExprKind::MethodCall(recv, m, _) if m == "to_string" => {
                if let Some(n) = is_param(recv) {
                    found.insert((fresh_name(&n), "Display".into()));
                }
            }
            ExprKind::BinOp(op, l, r) => {
                // `x > 0`, `x + 1`: a literal on the right fixes the operand type (`isize`/`f64`).
                if let (Some(a), Some(lit)) = (term_of(l, &is_param), literal_type(r)) {
                    let t = fresh_name(&a);
                    match op {
                        BinOp::Eq | BinOp::NotEq => { found.insert((t, format!("PartialEq<{lit}>"))); }
                        BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => { found.insert((t, format!("PartialOrd<{lit}>"))); }
                        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                            let tr = match op {
                                BinOp::Add => "Add", BinOp::Sub => "Sub", BinOp::Mul => "Mul",
                                BinOp::Div => "Div", _ => "Rem",
                            };
                            found.insert((t.clone(), format!("std::ops::{tr}<{lit}, Output = {t}>")));
                            found.insert((t, "Copy".into()));
                        }
                        _ => {}
                    }
                    return;
                }
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

/// The Rust type a numeric literal operand stands for (`int` is `isize`, `float` is `f64`).
fn literal_type(e: &Expr) -> Option<&'static str> {
    match &e.kind {
        ExprKind::Int(_) => Some("isize"),
        ExprKind::Float(_) => Some("f64"),
        _ => None,
    }
}

/// The untyped parameter an operand is made of: the parameter itself, or arithmetic that keeps its
/// type (`x + x`, `x * 2`: the `Output = T` bounds make the result a `T` again).
fn term_of(e: &Expr, is_param: &dyn Fn(&Expr) -> Option<String>) -> Option<String> {
    if let Some(n) = is_param(e) { return Some(n); }
    match &e.kind {
        ExprKind::BinOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem, l, r) => {
            let a = term_of(l, is_param)?;
            if literal_type(r).is_some() || term_of(r, is_param).as_deref() == Some(a.as_str()) { Some(a) } else { None }
        }
        _ => None,
    }
}

/// A use of an untyped parameter or field that no inferred bound can allow (D5 of the design
/// note): reported as a Boring error naming the use, instead of the rustc failure on the
/// generated code.
pub(crate) struct Unbounded {
    pub name: String,
    pub what: String,
    pub line: usize,
    pub col: usize,
    pub len: usize,
}

fn op_symbol(op: &BinOp) -> Option<&'static str> {
    Some(match op {
        BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/", BinOp::Rem => "%",
        BinOp::Eq => "==", BinOp::NotEq => "!=", BinOp::Lt => "<", BinOp::Gt => ">", BinOp::LtEq => "<=", BinOp::GtEq => ">=",
        _ => return None,
    })
}

/// Whether `name` is bound again somewhere in `body` (a `let`, a loop variable, a pattern, a
/// closure parameter), in which case a bare `name` may not be the parameter: no error is raised.
fn rebinds(body: &[Stmt], name: &str) -> bool {
    let dbg = format!("{body:?}");
    dbg.contains(&format!("name: \"{name}\"")) || dbg.contains(&format!("Bind(\"{name}\")"))
        || dbg.match_indices("vars: [").any(|(i, _)| {
            let rest = &dbg[i..];
            let end = rest.find(']').unwrap_or(rest.len());
            rest[..end].contains(&format!("\"{name}\""))
        })
}

/// The first use in `body` of one of the untyped `params` that needs a bound the pass cannot
/// infer: a method call (`x.len()`; `clone` and `to_string` are not reported), a field read, an
/// index, or an operator whose other operand is not the same parameter (`x + 1`, `x == 3`,
/// `x + y` for two parameters). `shadowed` lists names a bare use cannot refer to (the
/// parameters of a method whose struct field has the same name).
pub(crate) fn unbounded_use(params: &[String], body: &[Stmt], shadowed: &[String]) -> Option<Unbounded> {
    let live: Vec<String> = params.iter()
        .filter(|p| !shadowed.contains(p) && !rebinds(body, p))
        .cloned()
        .collect();
    if live.is_empty() {
        return None;
    }
    // The untyped parameter an expression is made of: itself, or arithmetic of that parameter alone.
    fn param_of(e: &Expr, live: &[String]) -> Option<String> {
        match &e.kind {
            ExprKind::Var(n) if live.contains(n) => Some(n.clone()),
            ExprKind::Field(b, n) if live.contains(n) && matches!(&b.kind, ExprKind::Var(s) if s == "self") => Some(n.clone()),
            ExprKind::BinOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem, l, r) => {
                let a = param_of(l, live)?;
                if literal_type(r).is_some() { return Some(a); }
                let b = param_of(r, live)?;
                (a == b).then_some(a)
            }
            _ => None,
        }
    }
    let mut found: Option<Unbounded> = None;
    let mut body = body.to_vec();
    {
        let mut on_expr = |e: &mut Expr| {
            if found.is_some() { return; }
            let mut report = |name: String, what: String, at: &Expr| {
                found = Some(Unbounded { name, what, line: at.line, col: at.col, len: at.len.max(1) });
            };
            match &e.kind {
                ExprKind::MethodCall(recv, m, a) if m != "clone" && m != "to_string" && !(m == "len" && a.is_empty()) => {
                    if let Some(n) = param_of(recv, &live) {
                        report(n, format!("the method call `.{m}()`"), e);
                    }
                }
                ExprKind::Field(recv, f) => {
                    if let Some(n) = param_of(recv, &live) {
                        report(n, format!("the field access `.{f}`"), e);
                    }
                }
                ExprKind::Index(recv, _) => {
                    if let Some(n) = param_of(recv, &live) {
                        report(n, "indexing `[...]`".into(), e);
                    }
                }
                ExprKind::BinOp(op, l, r) => {
                    let Some(sym) = op_symbol(op) else { return };
                    let (a, b) = (param_of(l, &live), param_of(r, &live));
                    let arithmetic = !matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq);
                    // `x > 0`, `x + 1`: allowed (literal bound); the literal on the left is not.
                    if a.is_some() && b.is_none() && literal_type(r).is_some() { return; }
                    if a.is_none() && b.is_some() && literal_type(l).is_some() {
                        let n = b.clone().unwrap();
                        report(n.clone(), format!("the `{sym}` operator with a literal on its left (write the parameter first: `{n} {sym} literal`)"), e);
                        return;
                    }
                    match (a, b) {
                        (Some(n), None) | (None, Some(n)) => report(n, format!("the `{sym}` operator with a value that is not itself"), e),
                        (Some(a), Some(b)) if a != b && arithmetic => report(a.clone(), format!("the `{sym}` operator with `{b}`, another untyped parameter (a separate type; to share one type write `T f(T {a}, T {b})`)"), e),
                        _ => {}
                    }
                }
                _ => {}
            }
        };
        let mut on_stmt = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
    }
    found
}

/// The parse error for an [`Unbounded`] use of an untyped `kind` (`"parameter"` / `"field"`).
pub(crate) fn unbounded_error(u: Unbounded, kind: &str) -> crate::parser::ParseError {
    crate::parser::ParseError::Generic {
        msg: format!(
            "cannot use the untyped {kind} `{n}` with {w}: an untyped {kind} can be printed, compared with another untyped {kind} and combined with itself by arithmetic, nothing more — write its type",
            n = u.name, w = u.what
        ),
        line: u.line, col: u.col, len: u.len,
    }
}

/// D8: replaces a bare mention of the struct `name` inside `ty` with `name<args>` (its own
/// parameters), so that `Node? next` in `struct Node` is `Node<T_value>?`. A mention that already
/// carries arguments is left alone.
pub(crate) fn share_self_params(ty: &mut Type, name: &str, args: &[Type]) {
    match ty {
        Type::Named(n) if n == name => *ty = Type::Generic(name.to_string(), args.to_vec()),
        Type::Generic(_, a) | Type::Tuple(a) => {
            for t in a { share_self_params(t, name, args); }
        }
        Type::Optional(i) | Type::Array(i) | Type::Set(i) | Type::Mut(i) | Type::Qualified(i, _) => share_self_params(i, name, args),
        Type::Dict(k, v) => {
            share_self_params(k, name, args);
            share_self_params(v, name, args);
        }
        _ => {}
    }
}

/// Bound inference through calls (phase 4): `def show(x): label(x)` where `label(y)` needs
/// `T_y: Display` means `x` needs it too. A fixpoint over the top-level functions of the file:
/// when an untyped parameter of `f` is passed *as is* in position `i` of a call to another
/// function `g` of the same scope (the file, or one `mod`) whose `i`th parameter is an implicit type parameter with bounds, the
/// bounds are copied onto `f`'s parameter. Bounds that name another parameter
/// (`PartialOrd<T_b>`) are not carried over; one that names only the callee's own parameter
/// (`Add<Output = T_a>`) is carried with the caller's parameter in its place: they tie the callee's
/// parameters together in a way the call site does not necessarily reproduce.
pub(crate) fn propagate_bounds(program: &mut Program) {
    propagate_in(&mut program.items);
}

/// `bound` with the identifier `from` replaced by `to` (whole identifiers only).
fn rename_token(bound: &str, from: &str, to: &str) -> String {
    let mut out = String::new();
    let mut tok = String::new();
    let flush = |tok: &mut String, out: &mut String| {
        out.push_str(if tok == from { to } else { tok });
        tok.clear();
    };
    for c in bound.chars() {
        if c.is_alphanumeric() || c == '_' { tok.push(c); } else { flush(&mut tok, &mut out); out.push(c); }
    }
    flush(&mut tok, &mut out);
    out
}

/// One scope: the functions of a file, or of one `mod` (each `mod` resolves calls on its own).
/// `pub(crate)` so `infer_return_types_with_imports` can run it over imported + local items.
pub(crate) fn propagate_in(items: &mut [Item]) {
    for item in items.iter_mut() {
        if let Item::Mod(m) = item {
            propagate_in(&mut m.items);
        }
    }
    for _ in 0..8 {
        // callee name -> per position: (type parameter, its simple bounds). A name declared more than
        // once (an overload, resolved by argument type) is ambiguous here and is left out.
        let mut declared: std::collections::HashMap<&str, usize> = Default::default();
        for item in items.iter() {
            if let Item::Fn(g) = item { *declared.entry(g.name.as_str()).or_default() += 1; }
        }
        let mut callees: std::collections::HashMap<String, Vec<Option<(String, Vec<String>)>>> = Default::default();
        for item in items.iter() {
            let Item::Fn(g) = item else { continue };
            if g.is_native || g.task || g.stream || declared.get(g.name.as_str()).copied().unwrap_or(0) != 1 { continue; }
            // A bound that names a type parameter of `g` (`Add<Output = T>`) means something only
            // inside `g`: not carried over.
            let own_other = |b: &str, n: &str| b.split(|c: char| !(c.is_alphanumeric() || c == '_')).any(|tok| tok != n && g.type_params.iter().any(|p| p == tok));
            let per_param = g.params.iter().map(|p| match &p.ty {
                // An implicit (`T_x`) or a written (`T`) type parameter of `g`.
                Some(Type::TypeParam(n)) if g.type_params.contains(n) => Some((
                    n.clone(),
                    g.where_clause.iter().filter(|(t, b)| t == n && !own_other(b, n)).map(|(_, b)| b.clone()).collect(),
                )),
                _ => None,
            }).collect();
            callees.insert(g.name.clone(), per_param);
        }
        let mut changed = false;
        for item in items.iter_mut() {
            let Item::Fn(f) = item else { continue };
            if f.is_native || f.task || f.stream { continue; }
            // parameter name -> its implicit type parameter
            let own: std::collections::HashMap<String, String> = f.params.iter().filter_map(|p| match &p.ty {
                Some(Type::TypeParam(n)) if n.starts_with("T_") && f.type_params.contains(n) => Some((p.name.clone(), n.clone())),
                _ => None,
            }).collect();
            if own.is_empty() { continue; }
            let mut found: Vec<(String, String)> = Vec::new();
            let mut body = f.body.clone();
            {
                let mut on_expr = |e: &mut Expr| {
                    let ExprKind::Call(callee, args) = &e.kind else { return };
                    let ExprKind::Var(g) = &callee.kind else { return };
                    let Some(slots) = callees.get(g) else { return };
                    for (i, a) in args.iter().enumerate() {
                        if a.label.is_some() { continue; }
                        let ExprKind::Var(v) = &a.value.kind else { continue };
                        let (Some(t), Some(Some((_, bounds)))) = (own.get(v), slots.get(i)) else { continue };
                        for b in bounds {
                            // A bound naming only the callee's own parameter (`Add<Output = T_n>`) is
                            // carried over with the caller's parameter in its place.
                            let renamed = rename_token(b, &slots[i].as_ref().unwrap().0, t);
                            found.push((t.clone(), renamed));
                        }
                    }
                };
                let mut on_stmt = |_: &mut Stmt| {};
                Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
            }
            for pair in found {
                if !f.where_clause.contains(&pair) {
                    f.where_clause.push(pair);
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }
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
    fn bounds_flow_through_a_call_into_another_untyped_function() {
        let src = "def label(y):\n    print \"item {y}\"\n\ndef show(x):\n    label(x)\n\ndef outer(z):\n    show(z)\n";
        let program = parser::parse(lex(src).expect("lex")).expect("parse");
        let wc = |name: &str| program.items.iter().find_map(|i| match i {
            Item::Fn(f) if f.name == name => Some(f.where_clause.clone()),
            _ => None,
        }).unwrap();
        assert!(wc("show").contains(&("T_x".into(), "Display".into())));
        // Two hops: `outer` passes `z` to `show`, which got its bound from `label`.
        assert!(wc("outer").contains(&("T_z".into(), "Display".into())));
    }

    #[test]
    fn bounds_flow_inside_a_mod_but_not_across_scopes() {
        let src = "mod m:\n    def label(y):\n        print \"{y}\"\n\n    def show(x):\n        label(x)\n\ndef label(y):\n    print y\n";
        let program = parser::parse(lex(src).expect("lex")).expect("parse");
        let Some(Item::Mod(m)) = program.items.iter().find(|i| matches!(i, Item::Mod(_))) else { panic!("mod") };
        let show = m.items.iter().find_map(|i| match i { Item::Fn(f) if f.name == "show" => Some(f), _ => None }).unwrap();
        assert!(show.where_clause.contains(&("T_x".into(), "Display".into())));
    }

    fn where_of(src: &str, name: &str) -> Vec<(String, String)> {
        let program = parser::parse(lex(src).expect("lex")).expect("parse");
        program.items.iter().find_map(|i| match i {
            Item::Fn(f) if f.name == name => Some(f.where_clause.clone()),
            _ => None,
        }).expect("function")
    }

    #[test]
    fn bounds_flow_through_a_written_generic_function() {
        let src = "string describe<T as Display>(T v):\n    \"{v}\"\n\ndef relay(x):\n    describe(x)\n";
        assert!(where_of(src, "relay").contains(&("T_x".into(), "Display".into())));
        // Several written bounds all flow.
        let src = "string both<T as Display + Eq>(T v):\n    \"{v}\"\n\ndef relay(x):\n    both(x)\n";
        let w = where_of(src, "relay");
        assert!(w.contains(&("T_x".into(), "Display".into())) && w.contains(&("T_x".into(), "Eq".into())), "{w:?}");
        // A bound that names another parameter of the callee is not carried over.
        let src = "def less(a, b):\n    print a < b\n\ndef relay(x, y):\n    less(x, y)\n";
        assert!(where_of(src, "relay").is_empty());
    }

    #[test]
    fn an_overloaded_callee_propagates_nothing() {
        let src = "def label(x):\n    print \"{x}\"\n\ndef label(x, y):\n    print \"{x}{y}\"\n\ndef relay(z):\n    label(z)\n";
        assert!(where_of(src, "relay").is_empty());
    }

    #[test]
    fn a_literal_or_labelled_argument_adds_no_bound() {
        let src = "def label(y):\n    print \"{y}\"\n\ndef show(x):\n    label(1)\n    label(y = x)\n";
        let program = parser::parse(lex(src).expect("lex")).expect("parse");
        let f = program.items.iter().find_map(|i| match i { Item::Fn(f) if f.name == "show" => Some(f), _ => None }).unwrap();
        assert!(f.where_clause.is_empty());
    }

    fn rejects(src: &str) -> Option<String> {
        parser::parse(lex(src).expect("lex")).err().map(|e| e.to_string())
    }

    #[test]
    fn a_use_that_needs_a_bound_is_a_boring_error() {
        for (src, what) in [
            ("def f(x):\n    print x.push(1)\n", ".push()"),
            ("def f(x):\n    print x.len(1)\n", ".len()"),
            ("def f(x):\n    print 1 + x\n", "literal on its left"),
            ("def f(x):\n    print 0 < x\n", "literal on its left"),
            ("def f(x):\n    print x + \"a\"\n", "`+` operator"),
            ("def f(x):\n    print x == y\n", "`==` operator"),
            ("def f(x):\n    print x.name\n", "field access `.name`"),
            ("def f(x):\n    print x[0]\n", "indexing"),
            ("def f(a, b):\n    print a * b\n", "another untyped parameter"),
        ] {
            let e = rejects(src).unwrap_or_else(|| panic!("should be rejected: {src}"));
            assert!(e.contains("cannot use the untyped parameter"), "{src}: {e}");
            assert!(e.contains(what), "{src}: {e}");
        }
    }

    #[test]
    fn a_literal_operand_adds_a_typed_bound() {
        let w = where_of("def f(n, x):\n    print n > 0\n    print x + 1.5\n", "f");
        assert!(w.contains(&("T_n".into(), "PartialOrd<isize>".into())), "{w:?}");
        assert!(w.contains(&("T_x".into(), "std::ops::Add<f64, Output = T_x>".into())), "{w:?}");
        assert!(w.contains(&("T_x".into(), "Copy".into())), "{w:?}");
    }

    #[test]
    fn len_adds_the_boring_len_bound_and_types_the_return() {
        let f = first_fn("def size(x):\n    return x.len()\n");
        assert!(f.where_clause.contains(&("T_x".into(), "BoringLen".into())), "{:?}", f.where_clause);
        assert_eq!(f.return_ty, Some(Type::Named("uint".into())));
    }

    #[test]
    fn own_output_bounds_flow_through_calls() {
        let w = where_of("def twice(n):\n    return n + n\n\ndef relay(x):\n    _ = twice(x)\n", "relay");
        assert!(w.contains(&("T_x".into(), "std::ops::Add<Output = T_x>".into())), "{w:?}");
    }

    #[test]
    fn supported_uses_and_rebound_names_are_not_reported() {
        for src in [
            "def f(x):\n    print x + x + x\n",
            "def f(n):\n    print n > 0\n    print n == 1\n    print n + 1\n    print n * 2 - 3 > 0.5\n",
            "def f(a, b):\n    print a < b\n    print a == b\n",
            "def f(x):\n    print x.to_string()\n    let y = x.clone()\n    print y\n",
            // `x` is a different thing inside the closure / after the `let`.
            "def f(x):\n    print x\n    let r = [1, 2].map((x): x + 1)\n    print r\n",
            "def f(x):\n    print x\n    let n = 2\n    for x in [1, 2]:\n        print x + n\n",
        ] {
            assert!(rejects(src).is_none(), "should be accepted: {src}: {:?}", rejects(src));
        }
    }

    #[test]
    fn an_untyped_field_used_beyond_its_bounds_is_an_error() {
        let e = rejects("struct B:\n    items\n\n    req int count():\n        return items.push(1)\n").expect("rejected");
        assert!(e.contains("untyped field `items`") && e.contains(".push()"), "{e}");
        // A method parameter of the same name is not the field.
        assert!(rejects("struct B:\n    items\n\n    req int count([int] items):\n        return items.len()\n").is_none());
        // Printing and comparing the field is fine.
        assert!(rejects("struct B:\n    v\n\n    req bool same():\n        return self.v == self.v\n").is_none());
    }

    #[test]
    fn untyped_view_fields() {
        let s = first_struct("view Badge:\n    bind var label\n    note\n    state taps = 0\n    state ratio = 1.5\n    state title = \"x\"\n    state flag = true\n    state neg = -2\n\n    body():\n        Text(\"{label}\")\n");
        // `bind` and a plain prop are generic; `state` takes the type of its literal default.
        assert_eq!(s.type_params, vec!["T_label".to_string(), "T_note".to_string()]);
        let ty = |n: &str| s.fields.iter().find(|f| f.name == n).unwrap().ty.clone();
        assert_eq!(ty("label"), Type::TypeParam("T_label".into()));
        assert_eq!(ty("taps"), Type::Named("int".into()));
        assert_eq!(ty("ratio"), Type::Named("float".into()));
        assert_eq!(ty("title"), Type::Named("string".into()));
        assert_eq!(ty("flag"), Type::Named("bool".into()));
        assert_eq!(ty("neg"), Type::Named("int".into()));
    }

    #[test]
    fn untyped_view_fields_that_cannot_be_typed_are_errors() {
        let e = |src: &str| rejects(src).expect("should be rejected");
        assert!(e("view V:\n    env theme\n\n    body():\n        Text(\"x\")\n").contains("`env theme` needs a written type"));
        assert!(e("view V:\n    state taps\n\n    body():\n        Text(\"x\")\n").contains("`state taps` has no type and no default"));
        assert!(e("view V:\n    state taps = make()\n\n    body():\n        Text(\"x\")\n").contains("cannot infer the type of `state taps`"));
        // Written types keep working.
        assert!(rejects("view V:\n    state int taps = 0\n    env string theme\n\n    body():\n        Text(\"x\")\n").is_none());
    }

    #[test]
    fn a_struct_naming_itself_shares_its_parameters() {
        let s = first_struct("struct Node:\n    value\n    Node? next\n\n    req Node? follow():\n        return next\n");
        let own = Type::Generic("Node".into(), vec![Type::TypeParam("T_value".into())]);
        assert_eq!(s.fields[1].ty, Type::Optional(Box::new(own.clone())));
        assert_eq!(s.methods[0].return_ty, Some(Type::Optional(Box::new(own))));
        // A mention that already carries arguments keeps them.
        let s = first_struct("struct Node:\n    value\n    Node<int>? next\n");
        let written = Type::Generic("Node".into(), vec![Type::Named("int".into())]);
        assert_eq!(s.fields[1].ty, Type::Optional(Box::new(written)));
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
    fn untyped_field_with_a_default_keeps_it() {
        let s = first_struct("struct B:\n    string url\n    retries = 0\n");
        assert_eq!(s.type_params, vec!["T_retries".to_string()]);
        assert!(s.fields[1].default.is_some());
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
