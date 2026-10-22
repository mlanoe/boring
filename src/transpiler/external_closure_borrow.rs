// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
//! Closure parameters of an *external* callback (`shaper.shapeWith((cluster): collect(cluster, out))`,
//! swash's `shape_with(impl FnMut(&GlyphCluster))`) are borrowed by the Rust API, but an untyped
//! Boring closure parameter carries no type to say so. When such a parameter is passed bare to a
//! top-level Boring function whose matching parameter is declared as an external named type by
//! value (`GlyphCluster cluster`), that parameter is rewritten to `GlyphCluster&` — exactly the
//! manual workaround — so the generated call type-checks (`expected GlyphCluster, found &GlyphCluster`).

use crate::ast::*;
use std::collections::{HashMap, HashSet};

/// Iterator/collection-style methods whose closures receive owned elements, not borrows.
const COLLECTION_CLOSURE_METHODS: &[&str] = &[
    "map", "filter", "any", "all", "find", "forEach", "for_each", "flatMap", "flat_map",
    "filterMap", "filter_map", "takeWhile", "skipWhile", "partition", "position", "count",
    "reduce", "fold", "sort", "sortBy", "sorted", "sortedBy", "subscribe", "min", "max",
    "minBy", "maxBy", "retain", "each", "then", "else",
];

fn collect_names(items: &[Item], fns: &mut HashMap<String, Vec<Option<String>>>, user_types: &mut HashSet<String>) {
    for it in items {
        match it {
            Item::Fn(f) if f.qualifier.is_none() => {
                fns.insert(f.name.clone(), f.params.iter().map(|p| match &p.ty {
                    Some(Type::Named(n)) if !p.mutable && !p.rebindable => Some(n.clone()),
                    _ => None,
                }).collect());
            }
            Item::Struct(s) => { user_types.insert(s.name.clone()); }
            Item::Enum(e) => { user_types.insert(e.name.clone()); }
            Item::Trait(t) => { user_types.insert(t.name.clone()); }
            Item::Alias(a) => { user_types.insert(a.name.clone()); }
            Item::Mod(m) => collect_names(&m.items, fns, user_types),
            _ => {}
        }
    }
}

fn scan_body(stmts: &[Stmt], fns: &HashMap<String, Vec<Option<String>>>, user_types: &HashSet<String>, found: &mut HashSet<(String, usize)>) {
    walk_stmts(stmts, &mut |n| {
        let WalkNode::Expr(e) = n else { return };
        let ExprKind::MethodCall(_, m, args) = &e.kind else { return };
        if COLLECTION_CLOSURE_METHODS.contains(&m.as_str()) || fns.contains_key(m.as_str()) { return; }
        for a in args {
            let ExprKind::Closure(params, _, body, _, _) = &a.value.kind else { continue };
            let names: Vec<&str> = params.iter().filter(|p| p.ty.is_none()).map(|p| p.name.as_str()).collect();
            if names.is_empty() { continue; }
            let mut visit = |n: WalkNode| {
                let WalkNode::Expr(c) = n else { return };
                let ExprKind::Call(callee, cargs) = &c.kind else { return };
                let ExprKind::Var(f) = &callee.kind else { return };
                let Some(sig) = fns.get(f.as_str()) else { return };
                for (i, ca) in cargs.iter().enumerate() {
                    if let (ExprKind::Var(v), Some(Some(ty))) = (&ca.value.kind, sig.get(i)) {
                        if names.contains(&v.as_str()) && !user_types.contains(ty) && !is_builtin_name(ty) {
                            found.insert((f.clone(), i));
                        }
                    }
                }
            };
            match body {
                ClosureBody::Expr(e) => walk_expr(e, &mut visit),
                ClosureBody::Block(b) => walk_stmts(b, &mut visit),
            }
        }
    });
}

fn is_builtin_name(n: &str) -> bool {
    matches!(n, "String" | "Vec" | "HashMap" | "HashSet" | "Option" | "Box" | "Rc" | "Arc")
}

fn scan_items(items: &[Item], fns: &HashMap<String, Vec<Option<String>>>, user_types: &HashSet<String>, found: &mut HashSet<(String, usize)>) {
    for it in items {
        match it {
            Item::Fn(f) => scan_body(&f.body, fns, user_types, found),
            Item::Struct(s) => for m in &s.methods { scan_body(&m.body, fns, user_types, found); },
            Item::Ext(x) => for m in &x.methods { scan_body(&m.body, fns, user_types, found); },
            Item::Mod(m) => scan_items(&m.items, fns, user_types, found),
            Item::Stmt(s) => scan_body(std::slice::from_ref(s), fns, user_types, found),
            _ => {}
        }
    }
}

fn apply(items: &mut [Item], found: &HashSet<(String, usize)>) {
    for it in items {
        match it {
            Item::Fn(f) if f.qualifier.is_none() => {
                for (i, p) in f.params.iter_mut().enumerate() {
                    if found.contains(&(f.name.clone(), i)) {
                        if let Some(t) = p.ty.take() {
                            p.ty = Some(Type::Qualified(Box::new(t), OwnerQual::Borrow));
                        }
                    }
                }
            }
            Item::Mod(m) => apply(&mut m.items, found),
            _ => {}
        }
    }
}

/// Returns a rewritten copy of `program` when any parameter needs to become a borrow.
pub(crate) fn infer(program: &Program) -> Option<Program> {
    let mut fns = HashMap::new();
    let mut user_types = HashSet::new();
    collect_names(&program.items, &mut fns, &mut user_types);
    let mut found = HashSet::new();
    scan_items(&program.items, &fns, &user_types, &mut found);
    if found.is_empty() { return None; }
    let mut p = program.clone();
    apply(&mut p.items, &found);
    Some(p)
}
