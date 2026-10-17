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

//! Method-form view modifiers (docs/design-notes/boring-ui-draft.md, "Layout and modifier
//! composition"): `Text("x").padding(10)` is sugar for the ordered wrapper
//! `Padding(insets = 10, children = [Text("x")])`, and `a.f().g()` wraps `a` in `F`, then that in `G`.
//!
//! A *modifier* is any `view` that has a `[View] children` field (the facades in `boring.ui` are
//! exactly that): its method name is the view's name with a lowercase first letter (`Padding` →
//! `padding`, `OnMount` → `onMount`) and its method arguments are its other fields in declaration
//! order (positional arguments take the field's name; labeled ones keep their label).
//!
//! There is no type inference here, so a call is only rewritten when its receiver is *syntactically*
//! a view: a construction of a known view (`Text(..)`, `Frame(..)`) or an already-rewritten
//! modifier. A method of the same name on any other receiver is left alone, so a user struct's own
//! `padding()` is never captured.

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::ast_walk::Walker;

/// `modifier method name -> (view name, its input fields in order)`.
type Modifiers = HashMap<String, (String, Vec<String>)>;

fn lower_camel(name: &str) -> String {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) => c.to_lowercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

fn is_view_array(t: &Type) -> bool {
    match t {
        Type::Mut(inner) => is_view_array(inner),
        Type::Array(inner) => matches!(&**inner, Type::Named(n) if n == "View"),
        _ => false,
    }
}

fn collect(items: &[Item], mods: &mut Modifiers, views: &mut HashSet<String>) {
    for item in items {
        match item {
            Item::Struct(s) if s.is_view => {
                views.insert(s.name.clone());
                if s.fields.iter().any(|f| f.name == "children" && is_view_array(&f.ty)) {
                    let inputs = s.fields.iter()
                        .filter(|f| f.name != "children" && f.view_kind.is_none())
                        .map(|f| f.name.clone())
                        .collect();
                    mods.insert(lower_camel(&s.name), (s.name.clone(), inputs));
                }
            }
            Item::Mod(m) => collect(&m.items, mods, views),
            // `use boring.<module>`: the embedded stdlib is not merged into the program.
            Item::Use(u) if u.path.first().map(String::as_str) == Some("boring") => {
                let module = u.path.get(1).map(String::as_str).unwrap_or("");
                if let Some(src) = crate::stdlib_embed::lookup(module) {
                    if let Ok(tokens) = crate::lexer::lex(src) {
                        if let Ok(p) = crate::parser::parse(tokens) {
                            collect(&p.items, mods, views);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn is_view_value(e: &Expr, views: &HashSet<String>) -> bool {
    matches!(&e.kind, ExprKind::Call(callee, _)
        if matches!(&callee.kind, ExprKind::Var(n) if views.contains(n)))
}

/// One rewriting step on `e` if it is a modifier call on a view value. Returns whether it changed.
fn rewrite(e: &mut Expr, mods: &Modifiers, views: &HashSet<String>) -> bool {
    let ExprKind::MethodCall(recv, method, args) = &mut e.kind else { return false };
    let Some((view, inputs)) = mods.get(method.as_str()) else { return false };
    if !is_view_value(recv, views) { return false; }
    let (line, col, len) = (e.line, e.col, e.len);
    let mut call_args: Vec<Arg> = Vec::with_capacity(args.len() + 1);
    let mut positional = 0;
    for a in args.drain(..) {
        let label = match a.label {
            Some(l) => Some(l),
            None if a.spread || a.default_rest => None,
            None => {
                let l = inputs.get(positional).cloned();
                positional += 1;
                l
            }
        };
        call_args.push(Arg { label, value: a.value, spread: a.spread, default_rest: a.default_rest });
    }
    let receiver = std::mem::replace(&mut **recv, Expr { kind: ExprKind::Nil, line, col, len });
    let child = Expr { line: receiver.line, col: receiver.col, len: receiver.len,
        kind: ExprKind::Array(vec![receiver]) };
    call_args.push(Arg { label: Some("children".to_string()), value: child, spread: false, default_rest: false });
    e.kind = ExprKind::Call(
        Box::new(Expr { kind: ExprKind::Var(view.clone()), line, col, len }),
        call_args,
    );
    true
}

fn rewrite_body(body: &mut [Stmt], mods: &Modifiers, views: &HashSet<String>) {
    // The walker is pre-order, so in `a.f().g()` the outer call is seen while its receiver is
    // still a method call; repeat until nothing changes (one pass per link of the longest chain).
    loop {
        let mut changed = false;
        let mut on_expr = |e: &mut Expr| { changed |= rewrite(e, mods, views); };
        let mut on_stmt = |_: &mut Stmt| {};
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
        if !changed { break; }
    }
}

/// Calls `f` on every statement list in `items`: function bodies, struct/view methods, inits,
/// top-level `let`/statements (a top-level `let` is handed over as a one-statement list).
pub fn walk_bodies(items: &mut [Item], f: &mut dyn FnMut(&mut [Stmt])) {
    for item in items {
        match item {
            Item::Fn(func) => f(&mut func.body),
            Item::Struct(s) => {
                for m in s.methods.iter_mut() { f(&mut m.body); }
                for m in s.type_methods.iter_mut() { f(&mut m.body); }
                for i in s.inits.iter_mut() { f(&mut i.body); }
            }
            Item::Mod(m) => walk_bodies(&mut m.items, f),
            Item::Let(l) => {
                let mut s = Stmt::Let(l.clone());
                f(std::slice::from_mut(&mut s));
                if let Stmt::Let(done) = s { *l = done; }
            }
            Item::Stmt(s) => f(std::slice::from_mut(s)),
            _ => {}
        }
    }
}

fn rewrite_items(items: &mut [Item], mods: &Modifiers, views: &HashSet<String>) {
    walk_bodies(items, &mut |body| rewrite_body(body, mods, views));
}

/// Rewrites every method-form modifier call in `program` (see the module doc).
pub fn desugar_view_modifiers(items: &mut [Item]) {
    let mut mods = Modifiers::new();
    let mut views = HashSet::new();
    collect(items, &mut mods, &mut views);
    if mods.is_empty() { return; }
    rewrite_items(items, &mods, &views);
}
