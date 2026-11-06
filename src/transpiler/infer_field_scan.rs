// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later

//! Body preparation for the struct-field qualifier walk (`walk_stmts_for_field_qualifiers`):
//! the walk only understands `self.field`, so a method body is first normalized so that a bare
//! `field` (which the language resolves to `self.field`, see "Implicit self" in the book) reads
//! `self.field` unless a parameter or a local of the same name shadows it, and the types of the
//! body's locals are collected so a call `recv.method(self.field)` can be resolved to its
//! declaration.

use super::Transpiler;
use crate::ast::*;
use std::collections::{HashMap, HashSet};

fn pattern_names(p: &Pattern, out: &mut HashSet<String>) {
    match p {
        Pattern::Bind(n) => { out.insert(n.clone()); }
        Pattern::Variant(_, ps) | Pattern::Tuple(ps) => ps.iter().for_each(|p| pattern_names(p, out)),
        Pattern::Some(p) => pattern_names(p, out),
        Pattern::Wildcard | Pattern::Lit(_) | Pattern::None => {}
    }
}

fn clause_names(c: &CondClause, out: &mut HashSet<String>) {
    match c {
        CondClause::Let(n, _) => { out.insert(n.clone()); }
        CondClause::LetPat(p, _) => pattern_names(p, out),
        CondClause::Expr(_) => {}
    }
}

fn match_names(m: &MatchStmt, out: &mut HashSet<String>) {
    for arm in &m.arms { arm.patterns.iter().for_each(|p| pattern_names(p, out)); }
}

/// Every name a body binds anywhere (parameters, `let`s, loop variables, patterns, closure
/// parameters, comprehension variables). Deliberately over-approximate: a name found here is
/// never rewritten to `self.name`, which at worst misses an inference.
pub(super) fn bound_names(params: &[String], stmts: &[Stmt]) -> HashSet<String> {
    let mut out: HashSet<String> = params.iter().cloned().collect();
    let mut stmts = stmts.to_vec();
    let out_cell = std::cell::RefCell::new(&mut out);
    let mut on_stmt = |st: &mut Stmt| {
        let mut out = out_cell.borrow_mut();
        match st {
            Stmt::Let(l) => { out.insert(l.name.clone()); }
            Stmt::LetDestructure(l) => out.extend(l.bindings.iter().map(|b| b.name.clone())),
            Stmt::For(f) => out.extend(f.vars.iter().cloned()),
            Stmt::WhileLet(w) => {
                out.insert(w.name.clone());
                if let Some(p) = &w.pattern { pattern_names(p, &mut out); }
            }
            Stmt::IfLet(i) => {
                i.clauses.iter().for_each(|c| clause_names(c, &mut out));
                i.elif_branches.iter().for_each(|b| b.clauses.iter().for_each(|c| clause_names(c, &mut out)));
            }
            Stmt::Guard(g) => {
                if let GuardCond::Clauses(cs) = &g.cond { cs.iter().for_each(|c| clause_names(c, &mut out)); }
            }
            Stmt::Match(m) => match_names(m, &mut out),
            Stmt::Fn(f) => {
                out.insert(f.name.clone());
                out.extend(f.params.iter().map(|p| p.name.clone()));
            }
            _ => {}
        }
    };
    let mut on_expr = |e: &mut Expr| {
        let mut out = out_cell.borrow_mut();
        match &e.kind {
            ExprKind::Closure(ps, ..) => out.extend(ps.iter().map(|p| p.name.clone())),
            ExprKind::Match(m) => match_names(m, &mut out),
            ExprKind::ArrayComp { var, .. } | ExprKind::ArrayCompIter { var, .. } => { out.insert(var.clone()); }
            ExprKind::LabeledArrayComp { clauses, .. } => out.extend(clauses.iter().map(|(v, _)| v.clone())),
            ExprKind::TryElseBlock(..) => { out.insert("error".into()); }
            _ => {}
        }
    };
    crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut stmts);
    out
}

/// Rewrite every bare `field` of `fields` (a field of the struct whose method this body is)
/// into `self.field`, unless the body binds that name itself.
pub(super) fn rewrite_bare_fields(stmts: &mut [Stmt], fields: &HashSet<&str>, params: &[String]) {
    if fields.is_empty() { return; }
    let bound = bound_names(params, stmts);
    let mut on_stmt = |_: &mut Stmt| {};
    let mut on_expr = |e: &mut Expr| {
        let ExprKind::Var(name) = &e.kind else { return };
        if !fields.contains(name.as_str()) || bound.contains(name.as_str()) { return; }
        let (line, col, len) = (e.line, e.col, e.len);
        let recv = Expr { kind: ExprKind::Var("self".into()), line, col, len };
        e.kind = ExprKind::Field(Box::new(recv), name.clone());
    };
    crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(stmts);
}

/// The struct a local of type `ty` holds (`Service`), or `[Service]` for an array of it.
/// Arrays carry the brackets so `for s in arr` and `arr[i]` can be typed from them.
fn type_tag(ty: &Type) -> Option<String> {
    match ty.without_mut() {
        Type::Array(inner) => named_of(inner).map(|n| format!("[{n}]")),
        Type::Qualified(inner, _) => type_tag(inner),
        other => named_of(other).map(str::to_string),
    }
}

fn named_of(ty: &Type) -> Option<&str> {
    match ty.without_mut() {
        Type::Named(n) => Some(n.as_str()),
        Type::Qualified(inner, _) => named_of(inner),
        _ => None,
    }
}

impl Transpiler {
    /// The struct type of each local of a body (parameters, `let`s with a declared type or a
    /// resolvable initializer); a name bound to two different types, or to something untyped,
    /// is left out. `self` maps to `owner`.
    pub(super) fn body_local_types(
        &self,
        owner: Option<&str>,
        params: &[(String, Type)],
        stmts: &[Stmt],
    ) -> HashMap<String, String> {
        let mut locals: HashMap<String, String> = HashMap::new();
        let mut ambiguous: HashSet<String> = HashSet::new();
        let bind = |locals: &mut HashMap<String, String>, ambiguous: &mut HashSet<String>, name: &str, ty: Option<String>| {
            match ty {
                Some(t) => match locals.get(name) {
                    Some(prev) if *prev != t => { ambiguous.insert(name.to_string()); }
                    _ => { locals.insert(name.to_string(), t); }
                },
                // Rebound to something we cannot type: no claim is safe.
                None => { ambiguous.insert(name.to_string()); }
            }
        };
        if let Some(owner) = owner { locals.insert("self".into(), owner.to_string()); }
        for (n, t) in params {
            bind(&mut locals, &mut ambiguous, n, type_tag(t));
        }
        let no_alias = HashMap::new();
        let mut stmts = stmts.to_vec();
        let locals_cell = std::cell::RefCell::new((&mut locals, &mut ambiguous));
        let mut on_stmt = |st: &mut Stmt| {
            if let Stmt::For(f) = st {
                // `for s in services`: `s` is a `Service` when `services` is a `[Service]`.
                if let [var] = f.vars.as_slice() {
                    let mut g = locals_cell.borrow_mut();
                    let elem = self.method_receiver_type(&f.iterable, &*g.0, &no_alias)
                        .and_then(|t| t.strip_prefix('[').and_then(|t| t.strip_suffix(']')).map(str::to_string));
                    let (loc, amb) = &mut *g;
                    bind(loc, amb, var, elem);
                }
            }
            if let Stmt::Let(l) = st {
                let mut g = locals_cell.borrow_mut();
                let ty = match &l.ty {
                    Some(t) => type_tag(t),
                    None => l.value.as_ref().and_then(|v| self.method_receiver_type(v, &*g.0, &no_alias)),
                };
                let (loc, amb) = &mut *g;
                bind(loc, amb, &l.name, ty);
            }
        };
        let mut on_expr = |_: &mut Expr| {};
        crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut stmts);
        for n in &ambiguous { locals.remove(n); }
        locals
    }

    /// Run the field walk over one method/setter body of `struct_name` itself (same-file
    /// inference): bare fields are normalized to `self.field` first.
    pub(super) fn walk_own_body_for_field_qualifiers(
        &mut self,
        params: &[(String, Type)],
        body: &[Stmt],
        struct_name: &str,
        target_fields: &HashMap<String, String>,
        alias_of: &HashMap<String, String>,
        candidates: &mut HashMap<String, Vec<crate::ast::OwnerQual>>,
    ) {
        let mut stmts = body.to_vec();
        let field_set: HashSet<&str> = target_fields.keys().map(String::as_str).collect();
        let names: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
        rewrite_bare_fields(&mut stmts, &field_set, &names);
        substitute_field_aliases(&mut stmts, &field_set);
        let locals = self.body_local_types(Some(struct_name), params, &stmts);
        self.walk_stmts_for_field_qualifiers(&stmts, struct_name, target_fields, &locals, alias_of, candidates);
    }
}

/// `mut c = svc.stats` (a `let` whose value is `self.field`, in a body already rewritten so the
/// scanned struct is `self`) makes `c` another name of the field for the rest of the body:
/// every later `c` is replaced by `self.field`, so a demand on `c` is a demand on the field.
/// Only a name bound exactly once in the body is substituted (a rebound or shadowed `c` could
/// hold something else).
pub(super) fn substitute_field_aliases(stmts: &mut Vec<Stmt>, fields: &HashSet<&str>) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut candidates: HashMap<String, String> = HashMap::new();
    {
        let mut copy = stmts.clone();
        let mut on_expr = |_: &mut Expr| {};
        let mut on_stmt = |st: &mut Stmt| {
            if let Stmt::Let(l) = st {
                *counts.entry(l.name.clone()).or_default() += 1;
                if let Some(Expr { kind: ExprKind::Field(obj, f), .. }) = &l.value {
                    if matches!(&obj.kind, ExprKind::Var(v) if v == "self") && fields.contains(f.as_str()) {
                        candidates.insert(l.name.clone(), f.clone());
                    }
                }
            }
        };
        crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut copy);
    }
    candidates.retain(|n, _| counts.get(n) == Some(&1));
    if candidates.is_empty() { return; }
    // Other binders (loop variables, patterns, closure parameters) of the same name disqualify it.
    let mut others: HashMap<String, usize> = HashMap::new();
    {
        let mut copy = stmts.clone();
        let cell = std::cell::RefCell::new(&mut others);
        let mut on_stmt = |st: &mut Stmt| match st {
            Stmt::For(f) => { for v in &f.vars { *cell.borrow_mut().entry(v.clone()).or_default() += 1; } }
            Stmt::LetDestructure(l) => { for b in &l.bindings { *cell.borrow_mut().entry(b.name.clone()).or_default() += 1; } }
            _ => {}
        };
        let mut on_expr = |e: &mut Expr| {
            if let ExprKind::Closure(ps, ..) = &e.kind {
                for p in ps { *cell.borrow_mut().entry(p.name.clone()).or_default() += 1; }
            }
        };
        crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut copy);
    }
    candidates.retain(|n, _| !others.contains_key(n));
    let mut on_stmt = |_: &mut Stmt| {};
    let mut on_expr = |e: &mut Expr| {
        let ExprKind::Var(name) = &e.kind else { return };
        let Some(f) = candidates.get(name) else { return };
        let (line, col, len) = (e.line, e.col, e.len);
        let recv = Expr { kind: ExprKind::Var("self".into()), line, col, len };
        e.kind = ExprKind::Field(Box::new(recv), f.clone());
    };
    crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(stmts);
}
