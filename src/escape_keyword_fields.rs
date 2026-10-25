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

//! Transpiler-only pass: a struct field whose name is a Rust keyword (`final`, `move`, ...)
//! is renamed to its escaped spelling (`r#final`) everywhere it denotes that field — the
//! declaration, `obj.field` accesses, labeled constructor arguments, and bare uses inside the
//! struct's own methods — so every emitter writes a valid Rust identifier without each one
//! escaping field names at every site. Never applied to `boring run` (the interpreter keeps
//! the source spelling, which `toJson`/introspection expose).

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::ast_walk::Walker;
use crate::transpiler::helpers::escape_rust_keyword;

fn escaped(name: &str) -> Option<String> {
    let e = escape_rust_keyword(name);
    (e != name).then_some(e)
}

/// Keyword-named fields per struct name, recursing into `mod` blocks.
fn collect(items: &[Item], out: &mut HashMap<String, HashSet<String>>) {
    for item in items {
        match item {
            Item::Struct(s) => {
                let names: HashSet<String> = s.fields.iter()
                    .map(|f| f.name.clone())
                    .chain(s.inits.iter().filter(|i| i.body.is_empty()).flat_map(|i| i.params.iter().map(|p| p.name.clone())))
                    .filter(|n| escaped(n).is_some())
                    .collect();
                if !names.is_empty() { out.insert(s.name.clone(), names); }
            }
            Item::Mod(m) => collect(&m.items, out),
            _ => {}
        }
    }
}

fn rewrite_body(body: &mut [Stmt], all: &HashSet<String>, own: &HashSet<String>, shadowed: &HashSet<String>, structs: &HashMap<String, HashSet<String>>) {
    let mut on_expr = |e: &mut Expr| match &mut e.kind {
        ExprKind::Field(_, n) | ExprKind::OptionalField(_, n) if all.contains(n.as_str()) => {
            *n = escape_rust_keyword(n);
        }
        ExprKind::Var(n) if own.contains(n.as_str()) && !shadowed.contains(n.as_str()) => {
            *n = escape_rust_keyword(n);
        }
        ExprKind::Call(callee, args) => {
            if let ExprKind::Var(c) = &callee.kind {
                if let Some(fields) = structs.get(c.as_str()) {
                    for a in args.iter_mut() {
                        if let Some(l) = &mut a.label {
                            if fields.contains(l.as_str()) { *l = escape_rust_keyword(l); }
                        }
                    }
                }
            }
        }
        _ => {}
    };
    let mut on_stmt = |_: &mut Stmt| {};
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
}

fn rewrite_items(items: &mut [Item], all: &HashSet<String>, structs: &HashMap<String, HashSet<String>>) {
    let none = HashSet::new();
    for item in items {
        match item {
            Item::Fn(f) => rewrite_body(&mut f.body, all, &none, &none, structs),
            Item::Struct(s) => {
                let own = structs.get(&s.name).cloned().unwrap_or_default();
                for f in &mut s.fields {
                    if let Some(e) = escaped(&f.name) { f.name = e; }
                }
                for init in &mut s.inits {
                    if init.body.is_empty() {
                        for p in &mut init.params { if let Some(e) = escaped(&p.name) { p.name = e; } }
                    } else {
                        let shadowed: HashSet<String> = init.params.iter().map(|p| p.name.clone()).collect();
                        rewrite_body(&mut init.body, all, &own, &shadowed, structs);
                    }
                }
                for m in &mut s.methods {
                    let shadowed: HashSet<String> = m.params.iter().map(|p| p.name.clone()).collect();
                    rewrite_body(&mut m.body, all, &own, &shadowed, structs);
                }
            }
            Item::Mod(m) => rewrite_items(&mut m.items, all, structs),
            Item::Stmt(st) => {
                let mut one = [st.clone()];
                rewrite_body(&mut one, all, &none, &none, structs);
                let [r] = one;
                *st = r;
            }
            Item::Let(l) => {
                if let Some(v) = &mut l.value {
                    let mut one = [Stmt::Expr(v.clone())];
                    rewrite_body(&mut one, all, &none, &none, structs);
                    let [Stmt::Expr(r)] = one else { continue };
                    *v = r;
                }
            }
            _ => {}
        }
    }
}

pub fn escape_keyword_fields(mut program: Program) -> Program {
    let mut structs = HashMap::new();
    collect(&program.items, &mut structs);
    if structs.is_empty() { return program; }
    let all: HashSet<String> = structs.values().flatten().cloned().collect();
    rewrite_items(&mut program.items, &all, &structs);
    program
}
