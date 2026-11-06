// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later

//! Struct field qualifier inference over the whole project.
//!
//! `infer_struct_field_qualifiers` (run while a file is pre-scanned) only sees the struct's own
//! methods and the `ext` blocks of the same file, and runs before the files it `use`s have been
//! scanned. This pass runs once every reachable file is registered and its function signatures
//! refined: for each struct with unqualified struct-typed fields it re-runs the same constraint
//! elimination over *every* body of every file, where an access `x.field` on a local `x` of the
//! struct's type (a parameter, a `let` with a declared type, a constructor call, an alias, a
//! `a.b.field` chain over declared fields) counts exactly like `self.field` does in the struct's
//! own methods. The intersection of constraints is commutative, so the result does not depend on
//! the order the files were visited; a field whose constraints cannot all be met is an error.

use super::infer_qualifiers::all_qualifiers;
use super::Transpiler;
use crate::ast::{Expr, ExprKind, Item, OwnerQual, Program, Stmt, StructDecl, Type};
use std::collections::{HashMap, HashSet};

/// Placeholder receiver the rewrite uses for "a value of the scanned struct", turned into
/// `self` once the real `self` of the body has been renamed out of the way.
const MARK: &str = "\u{1}recv";

struct ScanBody {
    /// `Type.method` / function name, for diagnostics.
    label: String,
    /// The type `self` denotes (struct, enum or `ext` target), `None` for a free function.
    owner: Option<String>,
    params: Vec<(String, Type)>,
    stmts: Vec<Stmt>,
}

fn collect_items<'a>(items: &'a [Item], structs: &mut Vec<&'a StructDecl>, bodies: &mut Vec<ScanBody>, top: &mut Vec<Stmt>) {
    let fn_body = |owner: Option<&str>, f: &crate::ast::FnDecl| ScanBody {
        label: match owner { Some(o) => format!("{o}.{}", f.name), None => f.name.clone() },
        owner: owner.map(str::to_string),
        params: f.params.iter().filter_map(|p| p.ty.clone().map(|t| (p.name.clone(), t))).collect(),
        stmts: f.body.clone(),
    };
    let set_body = |owner: &str, s: &crate::ast::SetDecl| ScanBody {
        label: format!("{owner}.{}", s.name),
        owner: Some(owner.to_string()),
        params: vec![(s.param_name.clone(), s.param_ty.clone())],
        stmts: s.body.clone(),
    };
    for item in items {
        match item {
            Item::Fn(f) => bodies.push(fn_body(None, f)),
            Item::Struct(s) => {
                structs.push(s);
                bodies.extend(s.methods.iter().map(|m| fn_body(Some(&s.name), m)));
                bodies.extend(s.setters.iter().map(|m| set_body(&s.name, m)));
            }
            Item::Enum(e) => bodies.extend(e.methods.iter().map(|m| fn_body(Some(&e.name), m))),
            Item::Ext(e) => {
                bodies.extend(e.methods.iter().map(|m| fn_body(Some(&e.type_name), m)));
                bodies.extend(e.setters.iter().map(|m| set_body(&e.type_name, m)));
            }
            Item::Mod(m) => collect_items(&m.items, structs, bodies, top),
            Item::Stmt(s) => top.push(s.clone()),
            Item::Let(l) => top.push(Stmt::Let(l.clone())),
            _ => {}
        }
    }
}

impl Transpiler {
    /// Returns `true` when it changed the qualifier of at least one field.
    pub(crate) fn infer_cross_file_field_qualifiers(&mut self, programs: &[Program]) -> bool {
        let mut structs: Vec<&StructDecl> = Vec::new();
        let mut bodies: Vec<ScanBody> = Vec::new();
        for p in programs {
            let mut p_top = Vec::new();
            collect_items(&p.items, &mut structs, &mut bodies, &mut p_top);
            if !p_top.is_empty() {
                bodies.push(ScanBody { label: "top-level code".into(), owner: None, params: vec![], stmts: p_top });
            }
        }
        let mut seen = HashSet::new();
        structs.retain(|s| seen.insert(s.name.clone()));
        // Deterministic order whatever the file visit order was.
        structs.sort_by(|a, b| a.name.cmp(&b.name));
        bodies.sort_by(|a, b| a.label.cmp(&b.label));

        let mut changed = false;
        for s in structs {
            let targets = Self::field_inference_targets(s);
            if targets.is_empty() { continue; }
            let empty_alias = HashMap::new();
            let fresh = || -> HashMap<String, Vec<OwnerQual>> {
                targets.keys().map(|n| (n.clone(), all_qualifiers())).collect()
            };
            self.task_method_call_fields.clear();
            let mut candidates = fresh();
            let mut contributions: Vec<(String, HashMap<String, Vec<OwnerQual>>)> = Vec::new();
            let mut touched = false;
            for body in &bodies {
                let Some((stmts, locals)) = self.rewrite_for_field_scan(body, &s.name, &targets) else { continue };
                touched = true;
                self.walk_stmts_for_field_qualifiers(&stmts, &s.name, &targets, &locals, &empty_alias, &mut candidates);
                // What this body alone allows, for the conflict diagnostic.
                let mut alone = fresh();
                self.walk_stmts_for_field_qualifiers(&stmts, &s.name, &targets, &locals, &empty_alias, &mut alone);
                contributions.push((body.label.clone(), alone));
            }
            if !touched { continue; }
            for (field, remaining) in &candidates {
                if !remaining.is_empty() { continue; }
                let all = all_qualifiers().len();
                let mut who: Vec<String> = contributions.iter()
                    .filter(|(_, c)| c.get(field).is_some_and(|q| q.len() < all))
                    .map(|(l, c)| format!("`{l}` ({})", describe(&c[field])))
                    .collect();
                who.sort();
                who.dedup();
                self.push_error(s.line, s.col, format!(
                    "conflicting qualifier requirements for field `{}.{field}`: {} — no single qualifier satisfies them all\n  \
                     fix: write the qualifier on the field (`{}'actor {field}`) and adapt the users that disagree",
                    s.name, who.join(", "), targets[field],
                ));
            }
            let resolved = self.resolve_field_candidates(&targets, &candidates);
            for field in targets.keys() {
                let key = format!("{}::{field}", s.name);
                let before = self.field_registry_state(&key);
                let new = resolved.iter().find(|(f, _)| f == field).map(|(_, q)| q.clone());
                // A field the global view cannot decide (empty set: reported above) keeps what the
                // per-file pass chose.
                let Some(q) = new else { continue };
                self.struct_mutex_fields.remove(&key);
                self.struct_mutex_task_fields.remove(&key);
                self.struct_rwlock_fields.remove(&key);
                self.struct_rwlock_task_fields.remove(&key);
                self.register_field_qualifier(key.clone(), &q);
                if self.field_registry_state(&key) != before { changed = true; }
            }
        }
        changed
    }

    /// A clone of `body` where every `x.field` whose receiver `x` has the scanned struct's type
    /// reads `self.field` (and the body's own `self` is renamed away), or `None` when the body
    /// never reaches a field of the struct. Also returns the body's local types, for the
    /// rewritten body (`self` is the scanned struct there, `__self` the body's own owner).
    fn rewrite_for_field_scan(
        &self,
        body: &ScanBody,
        struct_name: &str,
        targets: &HashMap<String, String>,
    ) -> Option<(Vec<Stmt>, HashMap<String, String>)> {
        let mut stmts = body.stmts.clone();
        // Inside the struct's own methods a bare `field` is `self.field`.
        if body.owner.as_deref() == Some(struct_name) {
            let fields: HashSet<&str> = targets.keys().map(String::as_str).collect();
            let names: Vec<String> = body.params.iter().map(|(n, _)| n.clone()).collect();
            super::infer_field_scan::rewrite_bare_fields(&mut stmts, &fields, &names);
        }
        let mut locals = self.body_local_types(body.owner.as_deref(), &body.params, &stmts);
        let no_alias = HashMap::new();
        let mut hit = false;
        {
            let mut on_stmt = |_: &mut Stmt| {};
            let mut on_expr = |e: &mut Expr| {
                let (line, col, len) = (e.line, e.col, e.len);
                if let ExprKind::Field(obj, f) = &mut e.kind {
                    if targets.contains_key(f.as_str())
                        && !matches!(&obj.kind, ExprKind::Var(v) if v == MARK)
                        && self.method_receiver_type(obj, &locals, &no_alias).as_deref() == Some(struct_name)
                    {
                        **obj = Expr { kind: ExprKind::Var(MARK.into()), line, col, len };
                        hit = true;
                    }
                }
            };
            crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut stmts);
        }
        if !hit { return None; }
        let mut on_stmt = |_: &mut Stmt| {};
        let mut on_expr = |e: &mut Expr| {
            if let ExprKind::Var(v) = &mut e.kind {
                if v == "self" { *v = "__self".into(); } else if v == MARK { *v = "self".into(); }
            }
        };
        crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut stmts);
        // In the rewritten body `self` is the scanned struct, `__self` the original receiver.
        locals.remove("self");
        if let Some(owner) = &body.owner { locals.insert("__self".into(), owner.clone()); }
        Some((stmts, locals))
    }
}

fn describe(q: &[OwnerQual]) -> String {
    let names: Vec<&str> = q.iter().map(|q| super::infer_qualifiers::qual_name(q)).collect();
    if names.is_empty() { "nothing".into() } else { format!("allows only '{}", names.join("/'")) }
}
