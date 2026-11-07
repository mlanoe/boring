// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later

//! A struct field placed in an array/dict/tuple/set literal is copied (value semantics). When the
//! element is then handed to a parameter that demands a shared handle (`'actor`/`'guard`, with or
//! without `'task`), the callee's mutations land on the copy and are silently lost, because the
//! field itself was never asked to be a handle. Reject it with the two ways out; a literal that is
//! only read (sum, print) is an ordinary by-value copy and stays valid. A field that already is a
//! lock handle is reported where the literal is emitted (`emit_expr.rs`).

use super::infer_field_scan::bound_names;
use super::Transpiler;
use crate::ast::*;
use std::collections::HashSet;

fn literal_items(e: &Expr) -> Vec<&Expr> {
    match &e.kind {
        ExprKind::Array(es) | ExprKind::Tuple(es) | ExprKind::Set(es) => es.iter().collect(),
        ExprKind::Dict(pairs) => pairs.iter().map(|(_, v)| v).collect(),
        _ => Vec::new(),
    }
}

fn demands_handle(ty: &Type) -> bool {
    matches!(ty.without_mut(), Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask | OwnerQual::Guard | OwnerQual::GuardTask))
}

impl Transpiler {
    /// A lock-handle field placed in a literal with no declared element type: the `run`
    /// interpreter copies it, so the build must not silently share it (same message for arrays,
    /// dicts and tuples).
    pub(crate) fn reject_lock_handle_copy(&self, e: &Expr, what: &str, example: &str) {
        if self.lock_handle_clone_any(e).is_none() { return; }
        self.push_error(e.line, e.col, format!(
            "`{}` is a shared lock handle ('actor/'guard) placed in {what} with no \
             declared element type: its elements would be copies. Declare the element type \
             with the qualifier (e.g. {example}) to share the handle",
            Self::expr_path_text(e)
        ));
    }

    /// A field read of a struct-typed field that is not a lock handle (a plain by-value copy).
    fn bare_struct_field(&self, e: &Expr) -> bool {
        let ExprKind::Field(inner, f) = &e.kind else { return false };
        if self.lock_handle_clone_any(e).is_some() { return false; }
        let Some(sn) = self.resolve_expr_struct_type(inner) else { return false };
        let Some(fields) = self.struct_fields.get(&sn) else { return false };
        fields.iter().any(|(n, t)| n == f
            && matches!(t.without_mut(), Type::Named(tn) if self.struct_fields.contains_key(tn)))
    }

    /// Whether `stmts` hand an expression satisfying `is_elem` to a parameter demanding a handle.
    fn passes_elem_to_handle_param(&self, is_elem: &dyn Fn(&Expr) -> bool, stmts: &[Stmt], ctx: &[Stmt]) -> bool {
        let mut found = false;
        // Receiver types of the body's locals (parameters, typed or constructed `let`s).
        let params: Vec<(String, Type)> = self.fn_current_params.iter().map(|(n, t)| (n.clone(), t.clone())).collect();
        let locals = self.body_local_types(None, &params, ctx);
        let no_alias = std::collections::HashMap::new();
        let mut body = stmts.to_vec();
        let mut on_stmt = |_: &mut Stmt| {};
        let mut on_expr = |e: &mut Expr| {
            match &e.kind {
                ExprKind::Call(callee, args) => {
                    let ExprKind::Var(f) = &callee.kind else { return };
                    let Some(pts) = self.fn_sigs.get(f.as_str()) else { return };
                    for (i, a) in args.iter().enumerate() {
                        if is_elem(&a.value) && pts.get(i).is_some_and(demands_handle) { found = true; }
                    }
                }
                // A task capturing an element: the field would have to be a shared handle too
                // (same rule as `constrain_task_field_captures` for `self.field`).
                ExprKind::Task(inner) | ExprKind::TaskWithTimeout(_, inner) => {
                    let mut captured = false;
                    let mut body = vec![Stmt::Expr((**inner).clone())];
                    let mut on_stmt = |_: &mut Stmt| {};
                    let mut on_expr = |x: &mut Expr| { if is_elem(x) { captured = true; } };
                    crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
                    if captured { found = true; }
                }
                ExprKind::MethodCall(obj, m, args) => {
                    // The receiver's struct when its local is typed; otherwise the method by name
                    // when exactly one struct declares it.
                    let suffix = format!("::{m}");
                    let ds = match self.method_receiver_type(obj, &locals, &no_alias) {
                        Some(sn) => self.struct_method_overload_decls.get(&format!("{sn}::{m}")),
                        None => {
                            let mut cands = self.struct_method_overload_decls.iter().filter(|(k, _)| k.ends_with(&suffix));
                            match (cands.next(), cands.next()) { (Some((_, ds)), None) => Some(ds), _ => None }
                        }
                    };
                    let Some(ds) = ds.filter(|d| d.len() == 1) else { return };
                    for (i, a) in args.iter().enumerate() {
                        if is_elem(&a.value) && ds[0].params.get(i).and_then(|p| p.ty.as_ref()).is_some_and(demands_handle) { found = true; }
                    }
                }
                _ => {}
            }
        };
        crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
        found
    }

    fn reject_copied_field(&self, items: &[&Expr]) {
        for it in items.iter().filter(|it| self.bare_struct_field(it)) {
            let path = Self::expr_path_text(it);
            let ExprKind::Field(inner, f) = &it.kind else { continue };
            let sn = self.resolve_expr_struct_type(inner).unwrap_or_default();
            let ty = self.struct_fields.get(&sn)
                .and_then(|fs| fs.iter().find(|(n, _)| n == f))
                .map(|(_, t)| self.emit_type(t.without_mut()))
                .unwrap_or_default();
            self.push_error(it.line, it.col, format!(
                "`{path}` is copied into a literal, but its elements need a shared handle (passed to an 'actor/'guard parameter or captured by a task), \
                 so the callee's mutations would be lost on the copy. Write the qualifier on the field \
                 (`{ty}'actor {f}`) to share it, or keep the handle in a local (`mut c = {path}`) \
                 and pass that"
            ));
        }
    }

    /// Syntactic pre-pass, run with a function body's statements before emission: records (by
    /// the literal's position) every array/dict/tuple/set literal that holds a field read and
    /// whose elements are handed to a handle-demanding parameter. The field's type is only known
    /// at emission, so `check_copied_field_literal` finishes the job there.
    ///
    /// A `let`'s elements are looked for in the statements after it in its own block (its scope),
    /// provided nothing there rebinds its name; a `for` variable, in the loop body.
    pub(crate) fn scan_copied_literal_demands(&self, stmts: &[Stmt]) {
        let mut copy = stmts.to_vec();
        let mut lets: Vec<*const Stmt> = Vec::new();
        let mut fors: Vec<*const Stmt> = Vec::new();
        {
            let mut on_expr = |_: &mut Expr| {};
            let mut on_stmt = |st: &mut Stmt| match st {
                Stmt::Let(_) => lets.push(st as *const Stmt),
                Stmt::For(_) => fors.push(st as *const Stmt),
                _ => {}
            };
            crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut copy);
        }
        let has_field = |e: &Expr| literal_items(e).iter().any(|i| matches!(i.kind, ExprKind::Field(..)));
        let hit = |lit: &Expr| { self.copied_literal_demand.borrow_mut().insert((lit.line, lit.col)); };
        for p in fors {
            // SAFETY: points into `copy`, alive and unmodified since the walk.
            let Stmt::For(f) = (unsafe { &*p }) else { continue };
            let [v] = f.vars.as_slice() else { continue };
            if !has_field(&f.iterable) || bound_names(&[], &f.body).contains(v) { continue; }
            if self.passes_elem_to_handle_param(&|e| matches!(&e.kind, ExprKind::Var(n) if n == v), &f.body, stmts) {
                hit(&f.iterable);
            }
        }
        for p in lets {
            let Stmt::Let(l) = (unsafe { &*p }) else { continue };
            let Some(val) = &l.value else { continue };
            if !has_field(val) { continue; }
            if let Some(t) = &l.ty {
                match t.without_mut() {
                    Type::Array(e) | Type::Set(e) if demands_handle(e) => continue,
                    Type::Dict(_, v) if demands_handle(v) => continue,
                    Type::Tuple(ts) if ts.iter().any(demands_handle) => continue,
                    _ => {}
                }
            }
            // Scope: the statements after the let in its own block; a let inside an expression
            // block (not reachable through statements) falls back to the whole body, which is only
            // sound while the name is bound once there.
            let name = l.name.as_str();
            let rest: Vec<Stmt> = match rest_after(&copy, p) {
                Some(r) => r.to_vec(),
                None if count_let_binders(&copy, name) == 1 => copy.clone(),
                None => continue,
            };
            if bound_names(&[], &rest).contains(name) { continue; }
            // Loop variables over the collection are elements too.
            let mut elems: HashSet<String> = HashSet::new();
            let mut rc = rest.clone();
            let mut on_expr = |_: &mut Expr| {};
            let mut on_stmt = |s: &mut Stmt| {
                if let Stmt::For(f) = s {
                    if matches!(&f.iterable.kind, ExprKind::Var(n) if n == name) {
                        if let [v] = f.vars.as_slice() { elems.insert(v.clone()); }
                    }
                }
            };
            crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut rc);
            let is_elem = |e: &Expr| match &e.kind {
                ExprKind::Var(n) => elems.contains(n),
                ExprKind::Index(o, _) | ExprKind::Field(o, _) => matches!(&o.kind, ExprKind::Var(n) if n == name),
                _ => false,
            };
            if self.passes_elem_to_handle_param(&is_elem, &rest, stmts) { hit(val); }
        }
    }

    /// Called when a `let`/`for` is emitted: rejects its literal when the pre-pass found a handle
    /// demand on its elements and one of them copies a plain struct field.
    pub(crate) fn check_copied_field_literal(&self, literal: &Expr) {
        if !self.copied_literal_demand.borrow().contains(&(literal.line, literal.col)) { return; }
        self.reject_copied_field(&literal_items(literal));
    }
}

/// Number of `let`s of `name` in `stmts` (any depth).
fn count_let_binders(stmts: &[Stmt], name: &str) -> usize {
    let mut n = 0;
    let mut copy = stmts.to_vec();
    let mut on_expr = |_: &mut Expr| {};
    let mut on_stmt = |s: &mut Stmt| { if matches!(s, Stmt::Let(l) if l.name == name) { n += 1; } };
    crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut copy);
    n
}

/// The blocks directly nested in a statement.
fn child_blocks(s: &Stmt) -> Vec<&[Stmt]> {
    match s {
        Stmt::If(i) => i.branches.iter().map(|(_, b)| b.as_slice()).chain(i.else_body.as_deref()).collect(),
        Stmt::IfLet(i) => std::iter::once(i.then_body.as_slice())
            .chain(i.elif_branches.iter().map(|b| b.body.as_slice()))
            .chain(i.else_body.as_deref()).collect(),
        Stmt::Match(m) => m.arms.iter().filter_map(|a| match &a.body { MatchBody::Block(b) => Some(b.as_slice()), _ => None }).collect(),
        Stmt::While(w) => vec![&w.body],
        Stmt::WhileLet(w) => vec![&w.body],
        Stmt::DoWhile(d) => vec![&d.body],
        Stmt::Loop(l) => vec![&l.body],
        Stmt::For(f) => vec![&f.body],
        Stmt::Guard(g) => vec![&g.else_body],
        Stmt::Try(t) => std::iter::once(t.body.as_slice()).chain(t.catch_clauses.iter().map(|c| c.body.as_slice())).collect(),
        Stmt::With(w) => vec![&w.body],
        Stmt::Fn(f) => vec![&f.body],
        _ => Vec::new(),
    }
}

/// The statements following `target` in the block that holds it.
fn rest_after<'a>(block: &'a [Stmt], target: *const Stmt) -> Option<&'a [Stmt]> {
    for (i, s) in block.iter().enumerate() {
        if std::ptr::eq(s, target) { return Some(&block[i + 1..]); }
        for b in child_blocks(s) {
            if let Some(r) = rest_after(b, target) { return Some(r); }
        }
    }
    None
}
