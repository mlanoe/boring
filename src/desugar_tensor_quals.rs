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

//! Residency inference for `kernel.tensors.*` host operands.
//!
//! Same model as the ownership-qualifier inference (`transpiler::infer_qualifiers`): every
//! unqualified labeled-array binding or parameter starts with the candidate set
//! `{'kernel'global, 'kernel'unified}`, constraints narrow it, bindings that flow into one
//! another share their fate, and a default resolves what is left. It runs as an AST pass
//! before the checker, interpreter and transpiler (they need the qualified type, and the
//! ownership engine only runs inside the transpiler), so all three see exactly the type an
//! explicit annotation would have produced.
//!
//! * **Participation.** Only bindings connected to a host `kernel.tensors.matmul/mma/linear`
//!   operand — directly, or through the links below — are touched.
//! * **Links.** `let y = x` and `f(x)` (a free function declared in the same unit, the
//!   argument being a plain variable) put the two bindings in one group, in both directions.
//!   A parameter has one qualifier, so every caller of it agrees.
//! * **Constraints** (the only things that really decide): an explicit
//!   `'kernel'global`/`'kernel'unified` on a callee parameter that receives the array, or on the
//!   return type of a function returning it, demands that qualifier. `'unified` wins over
//!   `'global` if a group is demanded both ways.
//! * **Default**, when nothing is demanded — a policy, not a correctness rule (host code may
//!   read either qualifier directly, and `with` works for both; they differ in cost only):
//!   a group holding a content-mutable (`mut`) binding or parameter → `'kernel'unified`,
//!   otherwise `'kernel'global`. A read-only input needs no host view; a mutable array is
//!   written, and usually read back, by the host. Spell the qualifier out to override.
//! * **Warning.** A `mut` binding in a participating group that is demonstrably never written
//!   (not a tensor destination, no assignment, not passed on, not returned) gets a warning:
//!   it only pushes its group to `'kernel'unified`.
//!
//! * **Methods.** `o.m(x)` links `x` to the parameter of struct method `m`, and `self.m(x)`
//!   inside a method does too. Methods are resolved by name: only a name declared exactly once
//!   among struct methods, extensions and trait defaults is followed, and a receiver evidently
//!   of another type (an annotated or constructed local) is ruled out. A written bare `'kernel`
//!   is accepted on a struct method's array parameters, return type and `let`s (not on `ext`
//!   methods or struct fields).
//! * **Top-level bindings in functions.** A top-level array used inside a free function body
//!   is read like a use at the top level (tensor operand, `f(x)`, `let y = x`), unless the
//!   function declares the same name. Methods may mean a field of that name, so a top-level
//!   mention there stays out of sight.
//!
//! Explicit qualifiers are never touched. Scope: `let`/`mut`/`var` declared directly in a
//! function or method body (or at the top level / in a `mod`), parameters of free functions
//! and struct methods. A name declared twice in one scope is left alone, as are calls through
//! closures, type methods and computed arguments.

use crate::ast::*;
use crate::ast_walk::Walker;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

pub fn infer_tensor_operand_quals(mut program: Program) -> Program {
    reject_unsupported_markers(&program.items);
    infer_unit(&mut program.items);
    program
}

thread_local! {
    /// Errors raised by the pass (a written `'kernel` somewhere it cannot be resolved);
    /// `checker::check` drains them into its own error list.
    static ERRORS: RefCell<Vec<crate::errors::SourceError>> = const { RefCell::new(Vec::new()) };
}

/// Drains the errors raised by earlier runs of the pass on this thread.
pub fn take_errors() -> Vec<crate::errors::SourceError> {
    ERRORS.with(|e| std::mem::take(&mut *e.borrow_mut()))
}

fn marker_error(what: &str, line: usize, col: usize) {
    ERRORS.with(|e| e.borrow_mut().push(crate::errors::SourceError::at(
        format!(
            "bare `'kernel` is not supported on {what}: it applies to array-typed `let`s, parameters and \
             return types of free functions — write `'kernel'global` or `'kernel'unified` here"
        ),
        line, col,
    )));
}

/// A written `'kernel` the pass will not resolve (so it must not reach the later stages).
fn reject_unsupported_markers(items: &[Item]) {
    fn fn_decl(f: &FnDecl, what: &str) {
        for p in &f.params {
            if has_marker(p.ty.as_ref()) { marker_error(what, p.line, p.col); }
        }
        if has_marker(f.return_ty.as_ref()) { marker_error(what, f.line, f.col); }
    }
    fn free_fn(f: &FnDecl) {
        for p in &f.params {
            if has_marker(p.ty.as_ref()) && !is_node_type(p.ty.as_ref()) { marker_error("a non-array parameter", p.line, p.col); }
        }
        if has_marker(f.return_ty.as_ref()) && !is_node_type(f.return_ty.as_ref()) {
            marker_error("a non-array return type", f.line, f.col);
        }
        for l in all_lets(&f.body) {
            if has_marker(l.ty.as_ref()) && !is_node_type(l.ty.as_ref()) { marker_error("a non-array binding", l.line, l.col); }
        }
    }
    for item in items {
        match item {
            Item::Let(l) if has_marker(l.ty.as_ref()) && !is_node_type(l.ty.as_ref()) => {
                marker_error("a non-array binding", l.line, l.col);
            }
            Item::Fn(f) => free_fn(f),
            Item::Mod(m) => reject_unsupported_markers(&m.items),
            Item::Struct(st) => {
                for f in &st.fields {
                    if has_marker(Some(&f.ty)) { marker_error("a struct field", f.line, f.col); }
                }
                // Struct methods take part in the inference like free functions.
                for m in &st.methods { free_fn(m); }
            }
            Item::Ext(e) => {
                for m in &e.methods { fn_decl(m, "a method"); }
            }
            _ => {}
        }
    }
}

/// One inferred qualifier, as reported by `--show-qualifiers`.
pub struct Decision {
    pub scope: String,
    pub name: String,
    pub line: usize,
    pub col: usize,
    pub qual: &'static str,
    pub why: String,
}

fn qual_text(q: &OwnerQual) -> &'static str {
    match q {
        OwnerQual::GpuUnified => "'kernel'unified",
        _ => "'kernel'global",
    }
}

thread_local! {
    /// Every qualifier the pass inferred on this thread, drained by `take_decisions`.
    static DECISIONS: RefCell<Vec<Decision>> = const { RefCell::new(Vec::new()) };

    /// Warnings raised by the pass, handed to the checker's own warning list
    /// (`checker::check` drains them) — the pass runs before the checker and returns a
    /// plain `Program`.
    static WARNINGS: RefCell<Vec<crate::errors::SourceError>> = const { RefCell::new(Vec::new()) };
}

/// Drains the warnings raised by earlier runs of the pass on this thread.
pub fn take_warnings() -> Vec<crate::errors::SourceError> {
    WARNINGS.with(|w| std::mem::take(&mut *w.borrow_mut()))
}

/// Drains the qualifiers inferred by earlier runs of the pass on this thread.
pub fn take_decisions() -> Vec<Decision> {
    DECISIONS.with(|d| std::mem::take(&mut *d.borrow_mut()))
}

const TOP: usize = usize::MAX;

/// What a function's `i`-th parameter looks like to a caller.
#[derive(Clone)]
enum ParamKind {
    /// Unqualified labeled array: a node in the inference graph. The flag: content-mutable.
    Node(usize, bool),
    /// Explicit `'kernel'global`/`'kernel'unified`: a demand on the argument.
    Force(OwnerQual, bool),
    Other,
}

/// A free function's parameters, as a caller sees them.
struct Sig {
    names: Vec<String>,
    kinds: Vec<ParamKind>,
}

impl Sig {
    /// The parameter an argument binds to: by label when it has one, by position otherwise.
    /// `(index, kind)`.
    fn param_for(&self, pos: usize, label: Option<&str>) -> Option<(usize, &ParamKind)> {
        let idx = match label {
            Some(l) => self.names.iter().position(|n| n == l)?,
            None => pos,
        };
        self.kinds.get(idx).map(|k| (idx, k))
    }
}

/// A struct method as a caller sees it.
struct MethodSig {
    owner: String,
    sig: Sig,
}

/// Every `let` of `stmts`, nested blocks (`if`, loops, `match`, closures...) included.
fn all_lets(stmts: &[Stmt]) -> Vec<LetStmt> {
    let mut stmts = stmts.to_vec();
    let mut out = Vec::new();
    {
        let mut on_expr = |_: &mut Expr| {};
        let mut on_stmt = |s: &mut Stmt| {
            if let Stmt::Let(l) = s { out.push(l.clone()); }
        };
        let mut walker = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        walker.stmts(&mut stmts);
    }
    out
}

#[derive(Default)]
struct Facts {
    operand: bool,
    /// Demanded qualifier, with a human-readable origin (for `--show-qualifiers`).
    forced: Vec<(OwnerQual, String)>,
    unions: Vec<usize>,
    /// Some use of this array is (or may be) a write: destination, passed to a mutable
    /// parameter, aliased, returned, or any use the pass does not model.
    maybe_written: bool,
}

#[derive(Clone)]
struct Scope {
    /// Enclosing function ("<top level>" for the file's own statements).
    fn_name: String,
    /// The struct a method belongs to; `None` for a free function and the top level.
    owner: Option<String>,
    /// Every parameter and `let` name the body declares, candidate or not: a top-level
    /// binding with the same name is shadowed there.
    declared: HashSet<String>,
    /// Struct type of the locals whose type is evident from an annotation or a constructor
    /// call (`None`: declared twice with different types), to resolve method calls.
    var_types: HashMap<String, Option<String>>,
    stmts: Vec<Stmt>,
    /// Qualifier demanded by the enclosing function's return type, if it is resident.
    ret_force: Option<OwnerQual>,
    /// Node standing for a bare-`'kernel` return type: whatever the function returns joins it.
    ret_node: Option<usize>,
    /// `(name, node)` for every candidate declared directly in this scope.
    names: Vec<(String, usize)>,
}

/// Where a node was declared, for the unneeded-`mut` warning.
struct NodeInfo {
    /// Written as a bare `'kernel`: takes part in the inference even with no tensor call.
    marker: bool,
    scope: String,
    name: String,
    line: usize,
    col: usize,
    mutable: bool,
}

fn infer_unit(items: &mut [Item]) {
    // Nested modules are independent units: they resolve calls against their own functions.
    for item in items.iter_mut() {
        if let Item::Mod(m) = item { infer_unit(&mut m.items); }
    }

    // --- nodes -------------------------------------------------------------------------
    let mut infos: Vec<NodeInfo> = Vec::new();
    let mut let_nodes: HashMap<(usize, String), usize> = HashMap::new();
    let mut param_nodes: HashMap<(usize, usize), usize> = HashMap::new();
    let mut ret_nodes: HashMap<usize, usize> = HashMap::new();
    let mut scope_of_fn: HashMap<usize, usize> = HashMap::new();
    let mut fn_index: HashMap<String, usize> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();
    for (i, item) in items.iter().enumerate() {
        if let Item::Fn(f) = item {
            if fn_index.insert(f.name.clone(), i).is_some() { ambiguous.insert(f.name.clone()); }
        }
    }
    for name in &ambiguous { fn_index.remove(name); }

    let mut scopes: Vec<Scope> = Vec::new();
    let mut top = Scope {
        fn_name: "<top level>".to_string(), owner: None, declared: HashSet::new(),
        var_types: HashMap::new(), stmts: Vec::new(), ret_force: None, ret_node: None, names: Vec::new(),
    };
    let mut top_seen: HashMap<String, usize> = HashMap::new();
    for item in items.iter() {
        match item {
            Item::Let(l) => {
                *top_seen.entry(l.name.clone()).or_default() += 1;
                if is_candidate(l) {
                    let id = infos.len();
                    infos.push(NodeInfo { marker: has_marker(l.ty.as_ref()), scope: "<top level>".to_string(), name: l.name.clone(), line: l.line, col: l.col, mutable: let_is_mutable(l) });
                    let_nodes.insert((TOP, l.name.clone()), id);
                    top.names.push((l.name.clone(), id));
                }
                // Top-level `let`s are items, but their initializers are host code like any
                // other statement: walk them too (aliases, uses in an initializer).
                top.stmts.push(Stmt::Let(l.clone()));
            }
            Item::Stmt(s) => {
                top.stmts.push(s.clone());
                // Lets nested in a top-level `if`/loop block.
                for l in all_lets(std::slice::from_ref(s)) {
                    *top_seen.entry(l.name.clone()).or_default() += 1;
                    if is_candidate(&l) {
                        let id = infos.len();
                        infos.push(NodeInfo { marker: has_marker(l.ty.as_ref()), scope: "<top level>".to_string(), name: l.name.clone(), line: l.line, col: l.col, mutable: let_is_mutable(&l) });
                        let_nodes.insert((TOP, l.name.clone()), id);
                        top.names.push((l.name.clone(), id));
                    }
                }
            }
            _ => {}
        }
    }
    // A name declared twice in the scope is not tracked; a written `'kernel` on one cannot
    // be left unresolved.
    for (name, id) in &top.names {
        if top_seen.get(name) != Some(&1) && infos[*id].marker {
            marker_error("a name declared twice in its scope", infos[*id].line, infos[*id].col);
        }
    }
    top.names.retain(|(n, _)| top_seen.get(n) == Some(&1));
    top.var_types = collect_var_types(&[], &top.stmts);
    scopes.push(top);

    // Free functions are keyed by their item index; methods get ids after the items, so one
    // key space covers both. `(id, item index, method index)`.
    let mut fn_like: Vec<(usize, usize, Option<usize>)> = Vec::new();
    let mut method_fid: HashMap<(usize, usize), usize> = HashMap::new();
    for (i, item) in items.iter().enumerate() {
        match item {
            Item::Fn(_) => fn_like.push((i, i, None)),
            Item::Struct(st) => {
                for mi in 0..st.methods.len() {
                    let fid = items.len() + method_fid.len();
                    method_fid.insert((i, mi), fid);
                    fn_like.push((fid, i, Some(mi)));
                }
            }
            _ => {}
        }
    }

    for &(i, item_idx, method_idx) in &fn_like {
        let (f, owner) = match (&items[item_idx], method_idx) {
            (Item::Fn(f), None) => (f, None),
            (Item::Struct(st), Some(mi)) => (&st.methods[mi], Some(st.name.clone())),
            _ => continue,
        };
        let label = match &owner { Some(o) => format!("{o}.{}", f.name), None => f.name.clone() };
        let ret_node = if has_marker(f.return_ty.as_ref()) {
            let id = infos.len();
            infos.push(NodeInfo { marker: true, scope: label.clone(), name: "<return>".to_string(), line: f.line, col: f.col, mutable: false });
            ret_nodes.insert(i, id);
            Some(id)
        } else {
            None
        };
        let mut scope = Scope {
            fn_name: label.clone(),
            owner: owner.clone(),
            declared: HashSet::new(),
            var_types: collect_var_types(&f.params, &f.body),
            ret_node,
            stmts: f.body.clone(),
            ret_force: f.return_ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned()),
            names: Vec::new(),
        };
        let mut seen: HashMap<String, usize> = HashMap::new();
        for p in &f.params { *seen.entry(p.name.clone()).or_default() += 1; }
        let lets = all_lets(&f.body);
        for l in &lets { *seen.entry(l.name.clone()).or_default() += 1; }
        scope.declared = seen.keys().cloned().collect();
        for (pi, p) in f.params.iter().enumerate() {
            if !p.variadic && p.default.is_none() && is_node_type(p.ty.as_ref())
                && seen.get(&p.name) == Some(&1)
            {
                let id = infos.len();
                infos.push(NodeInfo { marker: has_marker(p.ty.as_ref()), scope: label.clone(), name: p.name.clone(), line: p.line, col: p.col, mutable: param_is_mutable(p) });
                param_nodes.insert((i, pi), id);
                scope.names.push((p.name.clone(), id));
            }
        }
        for l in &lets {
            if !is_candidate(l) { continue; }
            if seen.get(&l.name) == Some(&1) {
                let id = infos.len();
                infos.push(NodeInfo { marker: has_marker(l.ty.as_ref()), scope: label.clone(), name: l.name.clone(), line: l.line, col: l.col, mutable: let_is_mutable(l) });
                let_nodes.insert((i, l.name.clone()), id);
                scope.names.push((l.name.clone(), id));
            } else if has_marker(l.ty.as_ref()) {
                marker_error("a name declared twice in its scope", l.line, l.col);
            }
        }
        for (pi, p) in f.params.iter().enumerate() {
            if has_marker(p.ty.as_ref()) && !param_nodes.contains_key(&(i, pi)) && is_node_type(p.ty.as_ref()) {
                marker_error("a name declared twice in its scope", p.line, p.col);
            }
        }
        if method_idx.is_none() { scope_of_fn.insert(i, scopes.len()); }
        scopes.push(scope);
    }
    if infos.is_empty() { return; }

    let sig_of = |fid: usize, f: &FnDecl| -> Sig {
        let kinds = f.params.iter().enumerate().map(|(pi, p)| {
            if let Some(&id) = param_nodes.get(&(fid, pi)) {
                ParamKind::Node(id, param_is_mutable(p))
            } else if let Some(q) = p.ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned()) {
                ParamKind::Force(q, param_is_mutable(p))
            } else {
                ParamKind::Other
            }
        }).collect();
        Sig { names: f.params.iter().map(|p| p.name.clone()).collect(), kinds }
    };
    let sigs: HashMap<String, Sig> = fn_index.iter().map(|(name, &i)| {
        let Item::Fn(f) = &items[i] else { unreachable!() };
        (name.clone(), sig_of(i, f))
    }).collect();
    // Methods are resolved by name: only a name no other method, extension or trait default
    // declares is followed, and `receiver_ok` rules out a receiver evidently of another type.
    let msigs: HashMap<String, MethodSig> = {
        let mut declared: HashMap<String, usize> = HashMap::new();
        for item in items.iter() {
            match item {
                Item::Struct(st) => for m in &st.methods { *declared.entry(m.name.clone()).or_default() += 1; },
                Item::Ext(e) => for m in &e.methods { *declared.entry(m.name.clone()).or_default() += 1; },
                Item::Trait(t) => for m in &t.defaults { *declared.entry(m.name.clone()).or_default() += 1; },
                _ => {}
            }
        }
        let mut out = HashMap::new();
        for (&(item_idx, mi), &fid) in &method_fid {
            let Item::Struct(st) = &items[item_idx] else { continue };
            let m = &st.methods[mi];
            if declared.get(&m.name) == Some(&1) {
                out.insert(m.name.clone(), MethodSig { owner: st.name.clone(), sig: sig_of(fid, m) });
            }
        }
        out
    };

    // --- returns --------------------------------------------------------------------------
    // A function whose returned array lives on the GPU can return it resident (no copy back
    // between chained calls) — but a resident value is opaque outside `with`, so this is only
    // done when it provably breaks no caller: see `auto_return_calls`.
    let mut ret_notes: HashMap<usize, String> = HashMap::new();
    let mut auto_fns: Vec<(String, usize)> = fn_index.iter().map(|(n, &i)| (n.clone(), i)).collect();
    auto_fns.sort_by_key(|(_, i)| *i);
    for (fname, fi) in auto_fns {
        let Item::Fn(f) = &items[fi] else { continue };
        let si = scope_of_fn[&fi];
        if !auto_return_shape(f, &scopes[si]) { continue; }
        let Some(calls) = auto_return_calls(&fname, &scopes, items, &sigs) else { continue };
        let id = infos.len();
        infos.push(NodeInfo { marker: false, scope: f.name.clone(), name: "<return>".to_string(), line: f.line, col: f.col, mutable: false });
        ret_nodes.insert(fi, id);
        scopes[si].ret_node = Some(id);
        ret_notes.insert(id, format!(
            "return type made resident: all {calls} call site(s) of `{fname}` keep the result on the GPU or take it into a typed binding"
        ));
    }
    let node_count = infos.len();

    // --- constraints and links ---------------------------------------------------------
    let mut facts: Vec<Facts> = (0..node_count).map(|_| Facts::default()).collect();
    // An explicit `'kernel` takes part in the inference even with no tensor call in sight.
    for (id, info) in infos.iter().enumerate() { facts[id].operand = info.marker; }
    for scope in &scopes {
        for (name, id) in &scope.names {
            let f = usage_of(scope, name, *id, &sigs, &msigs);
            let slot = &mut facts[*id];
            slot.operand |= f.operand;
            slot.maybe_written |= f.maybe_written;
            slot.forced.extend(f.forced);
            slot.unions.extend(f.unions);
        }
    }
    // A top-level binding is visible from every free-function body: read its uses there like
    // those of the top level itself. A function that declares the same name shadows it (the
    // mention is not this binding), and a method may mean a field of that name: both stay
    // out of sight, so the binding counts as possibly written.
    let top_names = scopes[0].names.clone();
    for (name, id) in &top_names {
        let mut out_of_sight = items.iter().any(|item| other_item_mentions(item, name));
        for scope in scopes.iter().skip(1) {
            if scope.owner.is_some() || !mentions(&scope.stmts, name) { continue; }
            if scope.declared.contains(name) { out_of_sight = true; continue; }
            let mut seen_from = scope.clone();
            seen_from.names.extend(top_names.iter().filter(|(n, _)| !scope.declared.contains(n)).cloned());
            let f = usage_of(&seen_from, name, *id, &sigs, &msigs);
            let slot = &mut facts[*id];
            slot.operand |= f.operand;
            slot.maybe_written |= f.maybe_written;
            slot.forced.extend(f.forced);
            slot.unions.extend(f.unions);
        }
        if out_of_sight { facts[*id].maybe_written = true; }
    }

    // --- groups, resolution ------------------------------------------------------------
    let mut parent: Vec<usize> = (0..node_count).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root { root = parent[root]; }
        let mut cur = x;
        while parent[cur] != root { let next = parent[cur]; parent[cur] = root; cur = next; }
        root
    }
    for (id, fact) in facts.iter().enumerate().take(node_count) {
        for &other in &fact.unions {
            let (a, b) = (find(&mut parent, id), find(&mut parent, other));
            if a != b { parent[a] = b; }
        }
    }
    #[derive(Default)]
    struct Group { operand: bool, members: Vec<usize>, forced: Vec<(OwnerQual, String)> }
    let mut groups: HashMap<usize, Group> = HashMap::new();
    for (id, fact) in facts.iter().enumerate().take(node_count) {
        let root = find(&mut parent, id);
        let g = groups.entry(root).or_default();
        g.operand |= fact.operand;
        g.members.push(id);
        g.forced.extend(fact.forced.iter().cloned());
    }
    let label = |id: usize| format!("{}::{}", infos[id].scope, infos[id].name);
    // The decision and the sentence that explains it (for `--show-qualifiers`).
    let resolve = |g: &Group| -> Option<(OwnerQual, String)> {
        if !g.operand { return None; }
        let demanded = |q: &OwnerQual| g.forced.iter().find(|(f, _)| f == q).map(|(_, why)| why.clone());
        let mutable = g.members.iter().copied().find(|&m| infos[m].mutable);
        Some(if let Some(why) = demanded(&OwnerQual::GpuUnified) {
            let conflict = demanded(&OwnerQual::GpuGlobal)
                .map(|other| format!(" (also demanded 'kernel'global by {other}; 'kernel'unified wins)"))
                .unwrap_or_default();
            (OwnerQual::GpuUnified, format!("demanded by {why}{conflict}"))
        } else if let Some(why) = demanded(&OwnerQual::GpuGlobal) {
            (OwnerQual::GpuGlobal, format!("demanded by {why}"))
        } else if let Some(m) = mutable {
            (OwnerQual::GpuUnified, format!("default: `{}` in its group is `mut`", label(m)))
        } else {
            (OwnerQual::GpuGlobal, "default: no explicit demand and no `mut` in its group".to_string())
        })
    };

    let mut decided: HashMap<usize, OwnerQual> = HashMap::new();
    for (id, info) in infos.iter().enumerate().take(node_count) {
        let root = find(&mut parent, id);
        let Some(g) = groups.get(&root) else { continue };
        let Some((q, why)) = resolve(g) else { continue };
        decided.insert(id, q.clone());
        let shared: Vec<String> = g.members.iter().copied().filter(|&m| m != id).map(label).collect();
        DECISIONS.with(|d| d.borrow_mut().push(Decision {
            scope: info.scope.clone(),
            name: info.name.clone(),
            line: info.line,
            col: info.col,
            qual: qual_text(&q),
            why: {
                let why = match ret_notes.get(&id) { Some(note) => format!("{note}; {why}"), None => why };
                if shared.is_empty() { why } else { format!("{why}; shares its qualifier with {}", shared.join(", ")) }
            },
        }));
    }

    // --- warnings ----------------------------------------------------------------------
    // A `mut` that changes nothing but the default: it is in a participating group, and
    // every use of the array is one the pass fully understands as read-only.
    for (id, info) in infos.iter().enumerate().take(node_count) {
        if decided.contains_key(&id) && info.mutable && !facts[id].maybe_written {
            WARNINGS.with(|w| w.borrow_mut().push(crate::errors::SourceError::at(
                format!(
                    "`{}` is declared `mut` but is never written: use `let` (a `mut` array is inferred \
                     'kernel'unified, a `let` one 'kernel'global), or spell the qualifier out",
                    info.name
                ),
                info.line, info.col,
            )));
        }
    }

    // --- apply -------------------------------------------------------------------------
    for (i, item) in items.iter_mut().enumerate() {
        match item {
            Item::Let(l) => {
                if let Some(q) = let_nodes.get(&(TOP, l.name.clone())).and_then(|id| decided.get(id)) {
                    requalify(&mut l.ty, q.clone());
                }
            }
            Item::Stmt(st) => requalify_lets(std::slice::from_mut(st), TOP, &let_nodes, &decided),
            Item::Fn(f) => {
                if let Some(q) = ret_nodes.get(&i).and_then(|id| decided.get(id)) {
                    requalify(&mut f.return_ty, q.clone());
                }
                for (pi, p) in f.params.iter_mut().enumerate() {
                    if let Some(q) = param_nodes.get(&(i, pi)).and_then(|id| decided.get(id)) {
                        requalify(&mut p.ty, q.clone());
                    }
                }
                requalify_lets(&mut f.body, i, &let_nodes, &decided);
            }
            Item::Struct(st) => {
                for (mi, m) in st.methods.iter_mut().enumerate() {
                    let fid = method_fid[&(i, mi)];
                    if let Some(q) = ret_nodes.get(&fid).and_then(|id| decided.get(id)) {
                        requalify(&mut m.return_ty, q.clone());
                    }
                    for (pi, p) in m.params.iter_mut().enumerate() {
                        if let Some(q) = param_nodes.get(&(fid, pi)).and_then(|id| decided.get(id)) {
                            requalify(&mut p.ty, q.clone());
                        }
                    }
                    requalify_lets(&mut m.body, fid, &let_nodes, &decided);
                }
            }
            _ => {}
        }
    }
}

/// A written bare `'kernel`: GPU-resident, `'global`/`'unified` left to this pass.
fn has_marker(ty: Option<&Type>) -> bool {
    fn go(t: &Type) -> bool {
        match t {
            Type::Mut(inner) => go(inner),
            Type::Qualified(_, q) => q.is_kernel_marker(),
            _ => false,
        }
    }
    ty.is_some_and(go)
}

/// An array-typed declaration the pass resolves: unqualified, or marked with `'kernel`.
fn is_node_type(ty: Option<&Type>) -> bool {
    if has_marker(ty) {
        fn inner_is_array(t: &Type) -> bool {
            match t {
                Type::Mut(inner) => inner_is_array(inner),
                Type::Qualified(inner, _) => matches!(inner.without_mut(), Type::LabeledArray(..) | Type::Array(_)),
                _ => false,
            }
        }
        return ty.is_some_and(inner_is_array);
    }
    is_unqualified_array(ty)
}

/// A labeled array `[T, k = 2, m = 3]` or a plain dynamic array `[T]` (the shape-less
/// `kernel.tensors.linear(.., m = .., n = .., k = ..)` operands), with no qualifier yet.
/// Applies the decided qualifiers to every `let` of `stmts` (nested blocks included).
fn requalify_lets(
    stmts: &mut [Stmt],
    scope: usize,
    let_nodes: &HashMap<(usize, String), usize>,
    decided: &HashMap<usize, OwnerQual>,
) {
    let mut on_expr = |_: &mut Expr| {};
    let mut on_stmt = |s: &mut Stmt| {
        if let Stmt::Let(l) = s {
            if let Some(q) = let_nodes.get(&(scope, l.name.clone())).and_then(|id| decided.get(id)) {
                requalify(&mut l.ty, q.clone());
            }
        }
    };
    let mut walker = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
    walker.stmts(stmts);
}

fn is_unqualified_array(ty: Option<&Type>) -> bool {
    matches!(ty, Some(t) if matches!(t.without_mut(), Type::LabeledArray(..) | Type::Array(_)))
}

fn is_candidate(l: &LetStmt) -> bool {
    !l.is_static && !l.is_lazy && is_node_type(l.ty.as_ref())
}

/// Content-mutable: `mut` / `var mut`. A bare `var` is rebindable only.
fn let_is_mutable(l: &LetStmt) -> bool {
    l.ty.as_ref().is_some_and(Type::is_mut) || l.var_mut
}

/// `mutable` is set for both `var` and `mut` parameters; `rebindable` marks the `var` one.
fn param_is_mutable(p: &Param) -> bool {
    p.ty.as_ref().is_some_and(Type::is_mut) || (p.mutable && !p.rebindable) || p.var_mut
}

fn requalify(ty: &mut Option<Type>, qual: OwnerQual) {
    if let Some(t) = ty.take() { *ty = Some(qualify(t, qual)); }
}

/// Replaces a written `'kernel` with the resolved qualifier.
fn replace_marker(ty: Type, qual: OwnerQual) -> Type {
    match ty {
        Type::Mut(inner) => Type::Mut(Box::new(replace_marker(*inner, qual))),
        Type::Qualified(inner, q) if q.is_kernel_marker() => Type::Qualified(inner, qual),
        other => other,
    }
}

/// Wraps the array type in `qual`, inside any outer `mut` — the same shape the parser gives
/// an explicit `mut [..]'kernel'global`.
fn qualify(ty: Type, qual: OwnerQual) -> Type {
    if has_marker(Some(&ty)) { return replace_marker(ty, qual); }
    match ty {
        Type::Mut(inner) => Type::Mut(Box::new(qualify(*inner, qual))),
        other => Type::Qualified(Box::new(other), qual),
    }
}

fn tensor_call(e: &Expr) -> Option<&[Arg]> {
    let ExprKind::MethodCall(recv, method, args) = &e.kind else { return None };
    if !matches!(method.as_str(), "matmul" | "mma" | "linear") { return None; }
    let ExprKind::Field(base, ns) = &recv.kind else { return None };
    // The parser spells `kernel.tensors` as `gpu.tensor`.
    if ns != "tensor" || !matches!(&base.kind, ExprKind::Var(n) if n == "gpu") { return None; }
    Some(args)
}

fn usage_of(scope: &Scope, name: &str, node: usize, sigs: &HashMap<String, Sig>, msigs: &HashMap<String, MethodSig>) -> Facts {
    // Both walker callbacks update the same counters.
    let facts = RefCell::new(Facts::default());
    let total = RefCell::new(0usize);
    let accounted = RefCell::new(0usize);
    let mut stmts = scope.stmts.clone();
    // The tail expression of a function with a resident return type: the array is handed
    // back as the function's own result.
    if let (Some(q), Some(Stmt::Expr(tail))) = (&scope.ret_force, stmts.last()) {
        if matches!(&tail.kind, ExprKind::Var(n) if n == name) {
            *accounted.borrow_mut() += 1;
            let mut f = facts.borrow_mut();
            f.maybe_written = true;
            f.forced.push((q.clone(), format!("the explicit return type of `{}`", scope.fn_name)));
        }
    }
    if let (Some(r), Some(Stmt::Expr(tail))) = (scope.ret_node, stmts.last()) {
        if matches!(&tail.kind, ExprKind::Var(n) if n == name) {
            *accounted.borrow_mut() += 1;
            let mut f = facts.borrow_mut();
            f.maybe_written = true;
            f.unions.push(r);
        }
    }
    {
        let ret_node = scope.ret_node;
        let ret_force = scope.ret_force.clone();
        let scope_fn = scope.fn_name.clone();
        let mut on_expr = |e: &mut Expr| {
            if matches!(&e.kind, ExprKind::Var(n) if n == name) { *total.borrow_mut() += 1; }
            if let Some(args) = tensor_call(e) {
                let operand_count = args.iter().position(|a| a.label.is_some()).unwrap_or(args.len());
                for (idx, arg) in args[..operand_count].iter().enumerate() {
                    if matches!(&arg.value.kind, ExprKind::Var(n) if n == name) {
                        let mut f = facts.borrow_mut();
                        f.operand = true;
                        // The last positional operand is the destination.
                        if idx + 1 == operand_count { f.maybe_written = true; }
                        *accounted.borrow_mut() += 1;
                    }
                }
            }
            // A free function (`f(x)`) or a method (`o.m(x)`) the pass has a signature for.
            let target: Option<(&Sig, String, &[Arg])> = match &e.kind {
                ExprKind::Call(callee, args) => match &callee.kind {
                    ExprKind::Var(fname) => sigs.get(fname).map(|sig| (sig, fname.clone(), args.as_slice())),
                    _ => None,
                },
                ExprKind::MethodCall(recv, mname, args) if tensor_call(e).is_none() => {
                    msigs.get(mname)
                        .filter(|m| receiver_ok(recv, &m.owner, scope))
                        .map(|m| (&m.sig, format!("{}.{mname}", m.owner), args.as_slice()))
                }
                _ => None,
            };
            let Some((sig, fname, args)) = target else { return };
            for (i, arg) in args.iter().enumerate() {
                if arg.spread || arg.default_rest { continue; }
                if !matches!(&arg.value.kind, ExprKind::Var(n) if n == name) { continue; }
                let Some((pi, kind)) = sig.param_for(i, arg.label.as_deref()) else { continue };
                match kind {
                    ParamKind::Node(other, mutable) => {
                        *accounted.borrow_mut() += 1;
                        let mut f = facts.borrow_mut();
                        f.maybe_written |= *mutable;
                        f.unions.push(*other);
                    }
                    ParamKind::Force(q, mutable) => {
                        // Handing the array to a function that wants it resident is
                        // enough to make it participate, tensor operand or not.
                        *accounted.borrow_mut() += 1;
                        let mut f = facts.borrow_mut();
                        f.operand = true;
                        f.maybe_written |= *mutable;
                        f.forced.push((q.clone(), format!("parameter {} of `{}` is explicitly {}", pi + 1, fname, qual_text(q))));
                    }
                    ParamKind::Other => {}
                }
            }
        };
        let names = &scope.names;
        let mut on_stmt = |s: &mut Stmt| match s {
            // `let y = x`: one group.
            Stmt::Let(l) => {
                if matches!(l.value.as_ref().map(|v| &v.kind), Some(ExprKind::Var(n)) if n == name) {
                    if let Some((_, other)) = names.iter().find(|(n, _)| *n == l.name) {
                        if *other != node {
                            *accounted.borrow_mut() += 1;
                            let mut f = facts.borrow_mut();
                            f.maybe_written = true;
                            f.unions.push(*other);
                        }
                    }
                }
            }
            Stmt::Return(r) => {
                if let (Some(node), Some(v)) = (ret_node, &r.value) {
                    if matches!(&v.kind, ExprKind::Var(n) if n == name) {
                        *accounted.borrow_mut() += 1;
                        let mut f = facts.borrow_mut();
                        f.maybe_written = true;
                        f.unions.push(node);
                    }
                }
                if let (Some(q), Some(v)) = (&ret_force, &r.value) {
                    if matches!(&v.kind, ExprKind::Var(n) if n == name) {
                        *accounted.borrow_mut() += 1;
                        let mut f = facts.borrow_mut();
                        f.maybe_written = true;
                        f.forced.push((q.clone(), format!("the explicit return type of `{}`", scope_fn)));
                    }
                }
            }
            _ => {}
        };
        let mut walker = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        walker.stmts(&mut stmts);
    }
    let mut facts = facts.into_inner();
    // A use the pass does not model (an assignment, a method call, a read, a `with` body...)
    // may be a write: only a fully understood array is reported as never written.
    if total.into_inner() > accounted.into_inner() { facts.maybe_written = true; }
    facts
}

/// Could `recv` be an instance of struct `owner`? Only a receiver evidently of another type
/// (an annotated or constructed local, a `self` of another struct) is ruled out.
fn receiver_ok(recv: &Expr, owner: &str, scope: &Scope) -> bool {
    match &recv.kind {
        ExprKind::Var(v) if v == "self" => scope.owner.as_deref() == Some(owner),
        ExprKind::Var(v) => !matches!(scope.var_types.get(v), Some(Some(t)) if t != owner),
        ExprKind::Call(callee, _) => match &callee.kind {
            ExprKind::Var(n) if n.chars().next().is_some_and(char::is_uppercase) => n == owner,
            _ => true,
        },
        _ => true,
    }
}

/// The struct name a type annotation names; the empty string for a type no struct method can
/// be called on (arrays, strings, numbers...); `None` when unknown.
fn type_owner(ty: &Type) -> Option<String> {
    match ty {
        Type::Mut(inner) | Type::Optional(inner) | Type::Qualified(inner, _) => type_owner(inner),
        Type::Named(n) => Some(n.clone()),
        Type::Array(_) | Type::ArrayN(..) | Type::LabeledArray(..) | Type::Dict(..) | Type::Set(_)
        | Type::Tuple(_) | Type::Str | Type::Int | Type::Uint | Type::Float32 | Type::Float64
        | Type::Bool => Some(String::new()),
        _ => None,
    }
}

/// The type of each local whose struct is evident: from its annotation, else from a
/// constructor-call initializer (`Op()`). A name declared with two different types maps to `None`.
fn collect_var_types(params: &[Param], stmts: &[Stmt]) -> HashMap<String, Option<String>> {
    let mut out: HashMap<String, Option<String>> = HashMap::new();
    let mut note = |name: &str, ty: Option<String>| {
        match out.get(name) {
            Some(prev) if *prev != ty => { out.insert(name.to_string(), None); }
            Some(_) => {}
            None => { out.insert(name.to_string(), ty); }
        }
    };
    for p in params { note(&p.name, p.ty.as_ref().and_then(type_owner)); }
    for l in all_lets(stmts) {
        let ty = l.ty.as_ref().and_then(type_owner).or_else(|| match l.value.as_ref().map(|v| &v.kind) {
            Some(ExprKind::Call(callee, _)) => match &callee.kind {
                ExprKind::Var(n) if n.chars().next().is_some_and(char::is_uppercase) => Some(n.clone()),
                _ => None,
            },
            _ => None,
        });
        note(&l.name, ty);
    }
    out
}

/// Free function whose single, plain-array return is a variable of its own: the shape a
/// resident return needs (every returned value is a tracked array, so none is a host copy).
fn auto_return_shape(f: &FnDecl, scope: &Scope) -> bool {
    if f.is_pub || f.task || f.is_native || !f.type_params.is_empty() { return false; }
    if !is_unqualified_array(f.return_ty.as_ref()) || scope.ret_node.is_some() || scope.ret_force.is_some() {
        return false;
    }
    let tracked = |e: &Expr| matches!(&e.kind, ExprKind::Var(n) if scope.names.iter().any(|(m, _)| m == n));
    let Some(Stmt::Expr(tail)) = scope.stmts.last() else { return false };
    if !tracked(tail) { return false; }
    let mut stmts = scope.stmts.clone();
    let mut ok = true;
    {
        let mut on_expr = |_: &mut Expr| {};
        let mut on_stmt = |s: &mut Stmt| {
            if let Stmt::Return(r) = s {
                if !matches!(&r.value, Some(v) if tracked(v)) { ok = false; }
            }
        };
        let mut walker = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        walker.stmts(&mut stmts);
    }
    ok
}

fn is_call_to(e: &Expr, fname: &str) -> bool {
    matches!(&e.kind, ExprKind::Call(callee, _) if matches!(&callee.kind, ExprKind::Var(n) if n == fname))
}

fn count_var(stmts: &[Stmt], name: &str) -> usize {
    let mut stmts = stmts.to_vec();
    let mut n = 0usize;
    {
        let mut on_expr = |e: &mut Expr| {
            if matches!(&e.kind, ExprKind::Var(v) if v == name) { n += 1; }
        };
        let mut on_stmt = |_: &mut Stmt| {};
        let mut walker = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        walker.stmts(&mut stmts);
    }
    n
}

fn expr_mentions(e: &Expr, name: &str) -> bool {
    count_var(&[Stmt::Expr(e.clone())], name) > 0
}

/// Does any item the pass does not analyze as a scope (methods, initializers, field defaults,
/// nested modules...) mention `name`?
fn other_item_mentions(item: &Item, name: &str) -> bool {
    match item {
        Item::Struct(st) => {
            st.methods.iter().any(|m| mentions(&m.body, name))
                || st.inits.iter().any(|i| mentions(&i.body, name))
                || st.setters.iter().any(|d| mentions(&d.body, name))
                || st.conversions.iter().any(|d| mentions(&d.body, name))
                || st.type_methods.iter().any(|d| mentions(&d.body, name))
                || st.fields.iter().any(|f| f.default.as_ref().is_some_and(|e| expr_mentions(e, name)))
                || st.type_vars.iter().any(|v| expr_mentions(&v.default, name))
        }
        Item::Ext(e) => {
            e.methods.iter().any(|m| mentions(&m.body, name))
                || e.setters.iter().any(|d| mentions(&d.body, name))
                || e.conversions.iter().any(|d| mentions(&d.body, name))
        }
        Item::Trait(t) => t.defaults.iter().any(|m| mentions(&m.body, name)),
        Item::Mod(m) => m.items.iter().any(|i| {
            other_item_mentions(i, name)
                || match i {
                    Item::Fn(f) => mentions(&f.body, name),
                    Item::Let(l) => l.value.as_ref().is_some_and(|e| expr_mentions(e, name)),
                    Item::Stmt(s) => mentions(std::slice::from_ref(s), name),
                    _ => false,
                }
        }),
        _ => false,
    }
}

/// `Some(number of call sites)` when every mention of `fname` in the program is a call that
/// stays correct if the function starts returning a resident array: the result is discarded,
/// bound to an explicitly typed binding (a plain annotation materializes a host copy), or bound
/// to an untyped `let`/`mut` whose every use keeps it on the GPU (a tensor operand, an argument
/// to a function taking a GPU array, a `with` block). Any other mention — an argument
/// expression, a method, a function value, a read outside `with` — makes it `None`, as does
/// having no call to look at.
fn auto_return_calls(
    fname: &str,
    scopes: &[Scope],
    items: &[Item],
    sigs: &HashMap<String, Sig>,
) -> Option<usize> {
    let mut mentions_total = 0usize;
    let mut safe_calls = 0usize;
    for (si, scope) in scopes.iter().enumerate() {
        mentions_total += count_var(&scope.stmts, fname);
        for stmt in &scope.stmts {
            match stmt {
                Stmt::Expr(e) if is_call_to(e, fname) => safe_calls += 1,
                Stmt::Let(l)
                    if l.value.as_ref().is_some_and(|v| is_call_to(v, fname))
                        && (l.ty.is_some()
                            || (matches!(l.binding, BindingKind::Let | BindingKind::Mut)
                                && result_uses_safe(scope, si == 0, &l.name, items, sigs))) =>
                {
                    safe_calls += 1;
                }
                _ => {}
            }
        }
    }
    let elsewhere = items.iter().any(|item| other_item_mentions(item, fname));
    (!elsewhere && safe_calls >= 1 && mentions_total == safe_calls).then_some(safe_calls)
}

/// Every use of the untyped binding `rname` (declared once in `scope`) keeps a resident value
/// resident.
fn result_uses_safe(scope: &Scope, is_top: bool, rname: &str, items: &[Item], sigs: &HashMap<String, Sig>) -> bool {
    let declared = scope.stmts.iter().filter(|s| matches!(s, Stmt::Let(l) if l.name == rname)).count();
    if declared != 1 { return false; }
    // A top-level binding is visible from every function body and method.
    if is_top && items.iter().any(|it| other_item_mentions(it, rname) || matches!(it, Item::Fn(f) if mentions(&f.body, rname))) {
        return false;
    }
    let total = RefCell::new(0usize);
    let accounted = RefCell::new(0usize);
    let mut stmts = scope.stmts.clone();
    {
        let mut on_expr = |e: &mut Expr| {
            if matches!(&e.kind, ExprKind::Var(v) if v == rname) { *total.borrow_mut() += 1; }
            if let Some(args) = tensor_call(e) {
                let operand_count = args.iter().position(|a| a.label.is_some()).unwrap_or(args.len());
                *accounted.borrow_mut() += args[..operand_count].iter()
                    .filter(|a| matches!(&a.value.kind, ExprKind::Var(v) if v == rname)).count();
            }
            if let ExprKind::Call(callee, args) = &e.kind {
                if let ExprKind::Var(f) = &callee.kind {
                    if let Some(sig) = sigs.get(f) {
                        for (i, arg) in args.iter().enumerate() {
                            if arg.spread || arg.default_rest { continue; }
                            if matches!(&arg.value.kind, ExprKind::Var(v) if v == rname)
                                && matches!(sig.param_for(i, arg.label.as_deref()), Some((_, ParamKind::Node(..) | ParamKind::Force(..))))
                            {
                                *accounted.borrow_mut() += 1;
                            }
                        }
                    }
                }
            }
        };
        // Reads inside `with rname:` are the sanctioned way to look at a resident value.
        let mut on_stmt = |s: &mut Stmt| {
            if let Stmt::With(w) = s {
                if w.names.iter().any(|n| n == rname) { w.body.clear(); }
            }
        };
        let mut walker = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        walker.stmts(&mut stmts);
    }
    total.into_inner() == accounted.into_inner()
}

fn mentions(body: &[Stmt], name: &str) -> bool {
    let mut stmts = body.to_vec();
    let mut found = false;
    {
        let mut on_expr = |e: &mut Expr| {
            if matches!(&e.kind, ExprKind::Var(n) if n == name) { found = true; }
        };
        let mut on_stmt = |_: &mut Stmt| {};
        let mut walker = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
        walker.stmts(&mut stmts);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lexer, parser};

    fn parse(src: &str) -> Program {
        parser::parse(lexer::lex(src).unwrap()).unwrap()
    }

    fn run(src: &str) -> Program {
        infer_tensor_operand_quals(parse(src))
    }

    /// Runs the pass and returns the warnings it raised.
    fn warnings_of(src: &str) -> Vec<String> {
        take_warnings();
        run(src);
        take_warnings().into_iter().map(|w| w.message).collect()
    }

    /// Types of every top-level `let` / `mut` after the pass, by name.
    fn types(src: &str) -> Vec<(String, Type)> {
        run(src).items.into_iter().filter_map(|i| match i {
            Item::Let(l) => l.ty.map(|t| (l.name, t)),
            _ => None,
        }).collect()
    }

    fn explicit_types(src: &str) -> Vec<(String, Type)> {
        parse(src).items.into_iter().filter_map(|i| match i {
            Item::Let(l) => l.ty.map(|t| (l.name, t)),
            _ => None,
        }).collect()
    }

    fn qual_of(types: &[(String, Type)], name: &str) -> Option<OwnerQual> {
        let (_, t) = types.iter().find(|(n, _)| n == name).unwrap();
        t.without_mut().gpu_resident_qual().cloned()
    }

    fn fn_quals(program: &Program, fname: &str) -> (Vec<Option<OwnerQual>>, Vec<Option<OwnerQual>>) {
        let f = program.items.iter().find_map(|i| match i {
            Item::Fn(f) if f.name == fname => Some(f),
            _ => None,
        }).unwrap();
        let params = f.params.iter().map(|p| p.ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned())).collect();
        let lets = f.body.iter().filter_map(|s| match s {
            Stmt::Let(l) => Some(l.ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned())),
            _ => None,
        }).collect();
        (params, lets)
    }

    const G: Option<OwnerQual> = Some(OwnerQual::GpuGlobal);
    const U: Option<OwnerQual> = Some(OwnerQual::GpuUnified);

    const EXPLICIT: &str = "let [float32, k = 4, m = 4]'kernel'global a = [1.0 as float32 for ..<16]\nlet [float32, n = 4, k = 4]'kernel'global b = [1.0 as float32 for ..<16]\nmut [float32, n = 4, m = 4]'kernel'unified c = [0.0 as float32 for ..<16]\nkernel.tensors.matmul(a, b, c)\nwith c:\n    print \"{c[0]}\"\n";
    const INFERRED: &str = "let [float32, k = 4, m = 4] a = [1.0 as float32 for ..<16]\nlet [float32, n = 4, k = 4] b = [1.0 as float32 for ..<16]\nmut [float32, n = 4, m = 4] c = [0.0 as float32 for ..<16]\nkernel.tensors.matmul(a, b, c)\nwith c:\n    print \"{c[0]}\"\n";

    #[test]
    fn inferred_matches_explicit_annotation() {
        // `ConstExpr` never compares equal, so compare the rendered types.
        assert_eq!(format!("{:?}", types(INFERRED)), format!("{:?}", explicit_types(EXPLICIT)));
    }

    #[test]
    fn explicit_qualifier_is_untouched() {
        assert_eq!(format!("{:?}", types(EXPLICIT)), format!("{:?}", explicit_types(EXPLICIT)));
    }

    #[test]
    fn let_defaults_to_global_and_mut_to_unified() {
        let got = types(INFERRED);
        assert_eq!((qual_of(&got, "a"), qual_of(&got, "b"), qual_of(&got, "c")), (G, G, U));
    }

    #[test]
    fn host_reads_do_not_change_the_choice() {
        let src = INFERRED.to_string() + "print \"{a[0]}\"\nwith a:\n    print \"{a[1]}\"\n";
        let got = types(&src);
        assert_eq!((qual_of(&got, "a"), qual_of(&got, "c")), (G, U));
    }

    #[test]
    fn explicit_qualifier_beats_the_mut_default() {
        let src = INFERRED.replace("mut [float32, n = 4, m = 4] c", "mut [float32, n = 4, m = 4]'kernel'global c");
        assert_eq!(qual_of(&types(&src), "c"), G);
    }

    #[test]
    fn array_unrelated_to_tensor_calls_is_untouched() {
        let src = "let [float32, n = 2, m = 2] x = [0.0 as float32 for ..<4]\nprint \"{x[0]}\"\n";
        assert_eq!(format!("{:?}", types(src)), format!("{:?}", explicit_types(src)));
    }

    #[test]
    fn binding_inside_function_body_is_inferred() {
        let src = "def work() throws:\n    let [float32, k = 2, m = 2] a = [0.0 for ..<4]\n    let [float32, n = 2, k = 2] b = [0.0 for ..<4]\n    mut [float32, n = 2, m = 2] c = [0.0 for ..<4]\n    kernel.tensors.matmul(a, b, c)\n";
        assert_eq!(fn_quals(&run(src), "work").1, vec![G, G, U]);
    }

    const CALLEE: &str = "def mm([float32, k = 2, m = 2] x, [float32, n = 2, k = 2] y, mut [float32, n = 2, m = 2] z) throws:\n    kernel.tensors.matmul(x, y, z)\n";
    const BODY: &str = "    let [float32, k = 2, m = 2] a = [0.0 for ..<4]\n    let [float32, n = 2, k = 2] b = [0.0 for ..<4]\n    mut [float32, n = 2, m = 2] c = [0.0 for ..<4]\n    mm(a, b, c)\n";

    #[test]
    fn qualifiers_propagate_through_a_called_function() {
        let src = format!("{CALLEE}def work() throws:\n{BODY}    with c:\n        print \"{{c[0]}}\"\n");
        let program = run(&src);
        assert_eq!(fn_quals(&program, "mm").0, vec![G, G, U]);
        assert_eq!(fn_quals(&program, "work").1, vec![G, G, U]);
    }

    #[test]
    fn a_mut_caller_array_reaches_the_callee_and_every_other_caller() {
        // `one` hands a `mut` array to `mm`'s read-only `x`: that group becomes unified, and
        // `two`'s plain `let` array, sharing the same parameter, follows.
        let one = "def one() throws:\n    mut [float32, k = 2, m = 2] a = [0.0 for ..<4]\n    let [float32, n = 2, k = 2] b = [0.0 for ..<4]\n    mut [float32, n = 2, m = 2] c = [0.0 for ..<4]\n    mm(a, b, c)\n";
        let two = format!("def two() throws:\n{BODY}");
        let program = run(&format!("{CALLEE}{one}{two}"));
        assert_eq!(fn_quals(&program, "mm").0, vec![U, G, U]);
        assert_eq!(fn_quals(&program, "one").1, vec![U, G, U]);
        assert_eq!(fn_quals(&program, "two").1, vec![U, G, U]);
    }

    #[test]
    fn explicit_callee_parameter_is_a_demand_on_the_caller() {
        // `c` is `mut` (default unified) but the callee explicitly wants 'global.
        let src = format!("def mm([float32, k = 2, m = 2]'kernel'unified x, [float32, n = 2, k = 2]'kernel'global y, mut [float32, n = 2, m = 2]'kernel'global z) throws:\n    kernel.tensors.matmul(x, y, z)\ndef work() throws:\n{BODY}");
        assert_eq!(fn_quals(&run(&src), "work").1, vec![U, G, G]);
    }

    #[test]
    fn returned_array_takes_the_return_type_qualifier() {
        let src = "req [float32, n = 2, m = 2]'kernel'global go([float32, k = 2, m = 2] x, [float32, n = 2, k = 2] y) throws:\n    mut [float32, n = 2, m = 2] z = [0.0 for ..<4]\n    kernel.tensors.matmul(x, y, z)\n    z\n";
        assert_eq!(fn_quals(&run(src), "go").1, vec![G]);
    }

    const SHAPES: &str = "[float32, k = 2, m = 2] x, [float32, n = 2, k = 2] y, mut [float32, n = 2, m = 2] z";

    #[test]
    fn qualifiers_propagate_through_a_chain_of_calls() {
        // outer -> inner -> kernel.tensors: only `inner` touches a tensor call.
        let body = BODY.replace("mm(a, b, c)", "outer(a, b, c)");
        let src = format!("def inner({SHAPES}) throws:\n    kernel.tensors.matmul(x, y, z)\ndef outer({SHAPES}) throws:\n    inner(x, y, z)\ndef work() throws:\n{body}");
        let program = run(&src);
        assert_eq!(fn_quals(&program, "inner").0, vec![G, G, U]);
        assert_eq!(fn_quals(&program, "outer").0, vec![G, G, U]);
        assert_eq!(fn_quals(&program, "work").1, vec![G, G, U]);
    }

    #[test]
    fn a_mut_at_the_top_of_a_chain_reaches_the_bottom() {
        let top = "mut [float32, k = 2, m = 2] x, [float32, n = 2, k = 2] y, mut [float32, n = 2, m = 2] z";
        let src = format!("def inner({SHAPES}) throws:\n    kernel.tensors.matmul(x, y, z)\ndef outer({top}) throws:\n    inner(x, y, z)\n");
        let program = run(&src);
        assert_eq!(fn_quals(&program, "inner").0, vec![U, G, U]);
        assert_eq!(fn_quals(&program, "outer").0, vec![U, G, U]);
    }

    #[test]
    fn a_function_never_connected_to_a_tensor_call_is_left_alone() {
        let src = format!("def other({SHAPES}) throws:\n    print \"{{x[0]}}\"\n");
        assert_eq!(fn_quals(&run(&src), "other").0, vec![None, None, None]);
    }

    #[test]
    fn alias_shares_the_group() {
        // `d` aliases the `mut` array `c`, so it follows `c` to 'kernel'unified.
        let src = INFERRED.replace("with c:", "let [float32, n = 4, m = 4] d = c\nwith c:");
        let got = types(&src);
        assert_eq!((qual_of(&got, "c"), qual_of(&got, "d")), (U, U));
    }

    const LINEAR: &str = "let [float32] x = [1.0 as float32 for ..<6]\nlet [float32] w = [1.0 as float32 for ..<6]\nmut [float32] y = [0.0 as float32 for ..<4]\nkernel.tensors.linear(x, w, y, m = 2, n = 2, k = 3)\n";

    #[test]
    fn plain_arrays_of_a_dynamic_linear_call_are_inferred() {
        let got = types(LINEAR);
        assert_eq!((qual_of(&got, "x"), qual_of(&got, "w"), qual_of(&got, "y")), (G, G, U));
    }

    #[test]
    fn plain_array_unrelated_to_tensor_calls_is_untouched() {
        let src = "let [float32] x = [1.0 as float32 for ..<6]\nprint \"{x[0]}\"\n";
        assert_eq!(format!("{:?}", types(src)), format!("{:?}", explicit_types(src)));
    }

    #[test]
    fn plain_array_parameters_propagate_like_labeled_ones() {
        let src = "def lin([float32] x, [float32] w, mut [float32] y) throws:\n    kernel.tensors.linear(x, w, y, m = 2, n = 2, k = 3)\ndef work() throws:\n    let [float32] a = [1.0 as float32 for ..<6]\n    let [float32] b = [1.0 as float32 for ..<6]\n    mut [float32] c = [0.0 as float32 for ..<4]\n    lin(a, b, c)\n";
        let program = run(src);
        assert_eq!(fn_quals(&program, "lin").0, vec![G, G, U]);
        assert_eq!(fn_quals(&program, "work").1, vec![G, G, U]);
    }

    #[test]
    fn decisions_explain_each_qualifier() {
        take_decisions();
        run(INFERRED);
        let d = take_decisions();
        let by_name = |n: &str| d.iter().find(|x| x.name == n).unwrap();
        assert_eq!((by_name("a").qual, by_name("c").qual), ("'kernel'global", "'kernel'unified"));
        assert!(by_name("a").why.contains("no `mut`"), "{}", by_name("a").why);
        assert!(by_name("c").why.contains("`mut`"), "{}", by_name("c").why);
        assert_eq!((by_name("a").line, by_name("c").line), (1, 3));
    }

    #[test]
    fn decisions_name_the_explicit_demand_and_the_group() {
        let src = format!("def mm([float32, k = 2, m = 2]'kernel'unified x, [float32, n = 2, k = 2]'kernel'global y, mut [float32, n = 2, m = 2]'kernel'global z) throws:\n    kernel.tensors.matmul(x, y, z)\ndef work() throws:\n{BODY}");
        take_decisions();
        run(&src);
        let d = take_decisions();
        let a = d.iter().find(|x| x.name == "a").unwrap();
        assert_eq!(a.scope, "work");
        assert_eq!(a.qual, "'kernel'unified");
        assert!(a.why.contains("parameter 1 of `mm` is explicitly 'kernel'unified"), "{}", a.why);
    }

    #[test]
    fn decisions_report_a_shared_group() {
        let src = format!("{CALLEE}def work() throws:\n{BODY}");
        take_decisions();
        run(&src);
        let d = take_decisions();
        let a = d.iter().find(|x| x.scope == "work" && x.name == "a").unwrap();
        assert!(a.why.contains("shares its qualifier with mm::x"), "{}", a.why);
    }

    const MARKED: &str = "let [float32, k = 4, m = 4]'kernel a = [1.0 as float32 for ..<16]\nlet [float32, n = 4, k = 4]'kernel b = [1.0 as float32 for ..<16]\nmut [float32, n = 4, m = 4]'kernel c = [0.0 as float32 for ..<16]\nkernel.tensors.matmul(a, b, c)\nwith c:\n    print \"{c[0]}\"\n";

    #[test]
    fn bare_kernel_resolves_like_an_unqualified_array() {
        // `'kernel` says "GPU", the pass picks the placement: same result as no qualifier.
        assert_eq!(format!("{:?}", types(MARKED)), format!("{:?}", types(INFERRED)));
        assert_eq!(format!("{:?}", types(MARKED)), format!("{:?}", explicit_types(EXPLICIT)));
    }

    #[test]
    fn bare_kernel_takes_part_even_without_a_tensor_call() {
        // An unqualified array that is not a tensor operand is left alone; a `'kernel` one is not.
        let src = "def hold([float32, k = 2, m = 2]'kernel x, mut [float32, k = 2, m = 2]'kernel y) throws:\n    print \"{x[0]}\"\n";
        assert_eq!(fn_quals(&run(src), "hold").0, vec![G, U]);
    }

    #[test]
    fn bare_kernel_joins_the_group_of_an_explicit_demand() {
        let src = format!("def mm([float32, k = 2, m = 2]'kernel'unified x, [float32, n = 2, k = 2]'kernel'global y, mut [float32, n = 2, m = 2]'kernel'global z) throws:\n    kernel.tensors.matmul(x, y, z)\ndef work() throws:\n{}", BODY.replace("let [float32, k = 2, m = 2] a", "let [float32, k = 2, m = 2]'kernel a"));
        assert_eq!(fn_quals(&run(&src), "work").1, vec![U, G, G]);
    }

    #[test]
    fn bare_kernel_return_type_follows_what_is_returned() {
        let src = "req [float32, n = 2, m = 2]'kernel go([float32, k = 2, m = 2] x, [float32, n = 2, k = 2] y) throws:\n    mut [float32, n = 2, m = 2] z = [0.0 for ..<4]\n    kernel.tensors.matmul(x, y, z)\n    z\n";
        let program = run(src);
        let f = program.items.iter().find_map(|i| match i { Item::Fn(f) if f.name == "go" => Some(f), _ => None }).unwrap();
        assert_eq!(f.return_ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned()), U);
        // `z` and the return share a group: same qualifier, no separate decision.
        assert_eq!(fn_quals(&program, "go").1, vec![U]);
    }

    #[test]
    fn bare_kernel_return_of_a_let_only_group_is_global() {
        let src = "req [float32, n = 2, m = 2]'kernel pass([float32, n = 2, m = 2]'kernel x) throws:\n    x\n";
        let program = run(src);
        let f = program.items.iter().find_map(|i| match i { Item::Fn(f) if f.name == "pass" => Some(f), _ => None }).unwrap();
        assert_eq!(f.return_ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned()), G);
        assert_eq!(fn_quals(&program, "pass").0, vec![G]);
    }

    fn errors_of(src: &str) -> Vec<String> {
        take_errors();
        run(src);
        take_errors().into_iter().map(|e| e.message).collect()
    }

    #[test]
    fn bare_kernel_where_it_cannot_be_resolved_is_an_error() {
        for (src, what) in [
            ("struct S:\n    let [float32, k = 2, m = 2]'kernel a\n", "a struct field"),
            ("let float'kernel x = 1.0\n", "a non-array binding"),
            ("def f(float'kernel x) throws:\n    print \"{x}\"\n", "a non-array parameter"),
        ] {
            let got = errors_of(src);
            assert!(got.iter().any(|e| e.contains(what)), "{src}: {got:?}");
        }
    }

    #[test]
    fn the_checker_reports_an_unresolvable_bare_kernel() {
        take_errors();
        let program = run("let float'kernel x = 1.0\n");
        let result = crate::checker::check(&program);
        assert!(result.errors.iter().any(|e| e.message.contains("bare `'kernel`")));
    }

    // --- option C: automatic resident returns -------------------------------------------
    const PRODUCT: &str = "req [float32, n = 2, m = 2] product([float32, k = 2, m = 2] x, [float32, n = 2, k = 2] y) throws:\n    mut [float32, n = 2, m = 2] z = [0.0 for ..<4]\n    kernel.tensors.matmul(x, y, z)\n    z\n";
    const ARGS: &str = "    let [float32, k = 2, m = 2] a = [0.0 for ..<4]\n    let [float32, n = 2, k = 2] b = [0.0 for ..<4]\n";

    fn return_qual(program: &Program, fname: &str) -> Option<OwnerQual> {
        let f = program.items.iter().find_map(|i| match i { Item::Fn(f) if f.name == fname => Some(f), _ => None }).unwrap();
        f.return_ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned())
    }

    fn with_caller(body: &str) -> String {
        format!("{PRODUCT}def work() throws:\n{ARGS}{body}")
    }

    #[test]
    fn a_resident_return_is_inferred_when_callers_only_use_it_through_with() {
        let program = run(&with_caller("    let r = product(a, b)\n    with r:\n        print \"{r[0]}\"\n"));
        assert_eq!(return_qual(&program, "product"), U);
    }

    #[test]
    fn a_resident_return_is_inferred_when_the_result_feeds_a_tensor_call() {
        let src = with_caller("    let r = product(a, b)\n    mut [float32, n = 2, m = 2] out = [0.0 for ..<4]\n    kernel.tensors.matmul(r, b, out)\n");
        assert_eq!(return_qual(&run(&src), "product"), U);
    }

    #[test]
    fn a_resident_return_is_inferred_for_typed_and_discarding_callers() {
        let typed = with_caller("    let [float32, n = 2, m = 2] r = product(a, b)\n    print \"{r[0]}\"\n");
        assert_eq!(return_qual(&run(&typed), "product"), U);
        let discard = with_caller("    product(a, b)\n");
        assert_eq!(return_qual(&run(&discard), "product"), U);
    }

    #[test]
    fn no_resident_return_when_a_caller_reads_the_result_directly() {
        let program = run(&with_caller("    let r = product(a, b)\n    print \"{r[0]}\"\n"));
        assert_eq!(return_qual(&program, "product"), None);
    }

    #[test]
    fn no_resident_return_when_the_call_is_an_expression_argument_or_a_function_value() {
        for body in ["    print \"{product(a, b)[0]}\"\n", "    let f = product\n    f(a, b)\n"] {
            assert_eq!(return_qual(&run(&with_caller(body)), "product"), None, "{body}");
        }
    }

    #[test]
    fn no_resident_return_when_one_of_several_callers_is_unsafe() {
        let src = format!("{}def other() throws:\n{ARGS}    let r = product(a, b)\n    print \"{{r[0]}}\"\n", with_caller("    let r = product(a, b)\n    with r:\n        print \"{r[0]}\"\n"));
        assert_eq!(return_qual(&run(&src), "product"), None);
    }

    #[test]
    fn no_resident_return_when_a_method_calls_the_function() {
        let src = format!("{}struct S:\n    req run([float32, k = 2, m = 2] a, [float32, n = 2, k = 2] b):\n        let r = product(a, b)\n        with r:\n            print \"{{r[0]}}\"\n", with_caller("    let r = product(a, b)\n    with r:\n        print \"{r[0]}\"\n"));
        assert_eq!(return_qual(&run(&src), "product"), None);
    }

    #[test]
    fn no_resident_return_for_pub_functions_or_functions_nobody_calls() {
        let public = with_caller("    let r = product(a, b)\n    with r:\n        print \"{r[0]}\"\n").replace("req [float32, n = 2, m = 2] product", "pub req [float32, n = 2, m = 2] product");
        assert_eq!(return_qual(&run(&public), "product"), None);
        assert_eq!(return_qual(&run(PRODUCT), "product"), None);
    }

    #[test]
    fn no_resident_return_when_the_returned_value_is_not_a_tracked_array() {
        let src = "req [float32, n = 2, m = 2] product([float32, k = 2, m = 2] x, [float32, n = 2, k = 2] y) throws:\n    mut [float32, n = 2, m = 2] z = [0.0 for ..<4]\n    kernel.tensors.matmul(x, y, z)\n    [0.0 as float32 for ..<4]\ndef work() throws:\n    let [float32, k = 2, m = 2] a = [0.0 for ..<4]\n    let [float32, n = 2, k = 2] b = [0.0 for ..<4]\n    let r = product(a, b)\n    with r:\n        print \"{r[0]}\"\n";
        assert_eq!(return_qual(&run(src), "product"), None);
    }

    #[test]
    fn no_resident_return_for_an_array_that_never_touches_the_gpu() {
        let src = "req [float32, n = 2, m = 2] plain([float32, n = 2, m = 2] x) throws:\n    x\ndef work() throws:\n    let [float32, n = 2, m = 2] a = [0.0 for ..<4]\n    let r = plain(a)\n    with r:\n        print \"{r[0]}\"\n";
        assert_eq!(return_qual(&run(src), "plain"), None);
    }

    #[test]
    fn the_decision_explains_an_automatic_resident_return() {
        take_decisions();
        run(&with_caller("    let r = product(a, b)\n    with r:\n        print \"{r[0]}\"\n"));
        let d = take_decisions();
        let ret = d.iter().find(|x| x.name == "<return>").unwrap();
        assert_eq!(ret.qual, "'kernel'unified");
        assert!(ret.why.contains("return type made resident") && ret.why.contains("1 call site"), "{}", ret.why);
    }

    // --- reach: nested blocks, labeled arguments ------------------------------------------
    /// All `let` types of a function body, nested blocks included, in source order.
    fn nested_quals(program: &Program, fname: &str) -> Vec<(String, Option<OwnerQual>)> {
        let f = program.items.iter().find_map(|i| match i { Item::Fn(f) if f.name == fname => Some(f), _ => None }).unwrap();
        all_lets(&f.body).into_iter()
            .map(|l| (l.name, l.ty.and_then(|t| t.without_mut().gpu_resident_qual().cloned())))
            .collect()
    }

    #[test]
    fn lets_in_nested_blocks_are_inferred() {
        let src = "def work(bool go) throws:\n    if go:\n        let [float32, k = 2, m = 2] a = [0.0 for ..<4]\n        let [float32, n = 2, k = 2] b = [0.0 for ..<4]\n        mut [float32, n = 2, m = 2] c = [0.0 for ..<4]\n        kernel.tensors.matmul(a, b, c)\n";
        let got = nested_quals(&run(src), "work");
        assert_eq!(got, vec![("a".to_string(), G), ("b".to_string(), G), ("c".to_string(), U)]);
    }

    #[test]
    fn lets_in_a_top_level_block_are_inferred() {
        let src = "for i in 0..<2:\n    let [float32, k = 2, m = 2] a = [0.0 for ..<4]\n    let [float32, n = 2, k = 2] b = [0.0 for ..<4]\n    mut [float32, n = 2, m = 2] c = [0.0 for ..<4]\n    kernel.tensors.matmul(a, b, c)\n";
        let program = run(src);
        let Item::Stmt(st) = &program.items[0] else { panic!("expected a top-level statement") };
        let got: Vec<_> = all_lets(std::slice::from_ref(st)).into_iter()
            .map(|l| l.ty.and_then(|t| t.without_mut().gpu_resident_qual().cloned())).collect();
        assert_eq!(got, vec![G, G, U]);
    }

    #[test]
    fn a_name_declared_in_two_blocks_is_left_alone() {
        let block = |c: &str| format!("    if go:\n        let [float32, k = 2, m = 2] a = [0.0 for ..<4]\n        let [float32, n = 2, k = 2] b = [0.0 for ..<4]\n        mut [float32, n = 2, m = 2] {c} = [0.0 for ..<4]\n        kernel.tensors.matmul(a, b, {c})\n");
        let src = format!("def work(bool go) throws:\n{}{}", block("c"), block("d"));
        let got = nested_quals(&run(&src), "work");
        assert!(got.iter().all(|(n, q)| q.is_none() || n == "c" || n == "d"), "{got:?}");
        assert!(got.iter().filter(|(n, _)| n == "a" || n == "b").all(|(_, q)| q.is_none()), "{got:?}");
    }

    #[test]
    fn a_bare_kernel_on_a_name_declared_twice_is_an_error() {
        let src = "def work(bool go) throws:\n    if go:\n        let [float32, k = 2, m = 2]'kernel a = [0.0 for ..<4]\n    if go:\n        let [float32, k = 2, m = 2]'kernel a = [0.0 for ..<4]\n";
        let got = errors_of(src);
        assert!(got.iter().any(|e| e.contains("declared twice")), "{got:?}");
    }

    #[test]
    fn labeled_arguments_bind_by_name_not_position() {
        // Out of order on purpose: z first, then x, then y.
        let body = BODY.replace("mm(a, b, c)", "mm(z = c, x = a, y = b)");
        let src = format!("{CALLEE}def work() throws:\n{body}");
        let program = run(&src);
        assert_eq!(fn_quals(&program, "mm").0, vec![G, G, U]);
        assert_eq!(fn_quals(&program, "work").1, vec![G, G, U]);
    }

    #[test]
    fn a_labeled_argument_meets_an_explicit_demand_on_its_parameter() {
        let callee = "def mm([float32, k = 2, m = 2]'kernel'unified x, [float32, n = 2, k = 2]'kernel'global y, mut [float32, n = 2, m = 2]'kernel'global z) throws:\n    kernel.tensors.matmul(x, y, z)\n";
        let body = BODY.replace("mm(a, b, c)", "mm(y = b, x = a, z = c)");
        let src = format!("{callee}def work() throws:\n{body}");
        assert_eq!(fn_quals(&run(&src), "work").1, vec![U, G, G]);
    }

    #[test]
    fn no_warning_when_mut_is_written() {
        // `c` is the destination.
        assert!(warnings_of(INFERRED).is_empty());
    }

    #[test]
    fn no_warning_when_mut_array_is_assigned() {
        let src = INFERRED
            .replace("let [float32, k = 4, m = 4] a = [1.0 as float32 for ..<16]", "mut [float32, k = 4, m = 4] a = [1.0 as float32 for ..<16]\na[0] = 2.0 as float32");
        assert!(warnings_of(&src).is_empty(), "{:?}", warnings_of(&src));
    }

    #[test]
    fn warns_when_mut_input_is_never_written() {
        let src = INFERRED.replace("let [float32, k = 4, m = 4] a", "mut [float32, k = 4, m = 4] a");
        let got = warnings_of(&src);
        assert_eq!(got.len(), 1, "{got:?}");
        assert!(got[0].contains("`a`") && got[0].contains("never written"), "{got:?}");
    }

    #[test]
    fn explicit_qualifier_is_not_warned_about() {
        let src = EXPLICIT.replace("let [float32, k = 4, m = 4]'kernel'global a", "mut [float32, k = 4, m = 4]'kernel'global a");
        assert!(warnings_of(&src).is_empty());
    }

    #[test]
    fn the_checker_reports_the_warning() {
        let src = INFERRED.replace("let [float32, k = 4, m = 4] a", "mut [float32, k = 4, m = 4] a");
        take_warnings();
        let program = run(&src);
        let result = crate::checker::check(&program);
        assert!(
            result.warnings.iter().any(|w| w.message.contains("never written")),
            "{:?}", result.warnings.iter().map(|w| &w.message).collect::<Vec<_>>()
        );
    }
    // --- methods and top-level bindings used inside functions ----------------------------

    /// Qualifiers of `Struct.method`'s parameters and body `let`s.
    fn method_quals(program: &Program, sname: &str, mname: &str) -> (Vec<Option<OwnerQual>>, Vec<Option<OwnerQual>>) {
        let m = program.items.iter().find_map(|i| match i {
            Item::Struct(st) if st.name == sname => st.methods.iter().find(|m| m.name == mname),
            _ => None,
        }).unwrap();
        let params = m.params.iter().map(|p| p.ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned())).collect();
        let lets = m.body.iter().filter_map(|s| match s {
            Stmt::Let(l) => Some(l.ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned())),
            _ => None,
        }).collect();
        (params, lets)
    }

    const METHOD: &str = "struct Op:\n    def go([float32, k = 2, m = 2] x, [float32, n = 2, k = 2] y, mut [float32, n = 2, m = 2] z) throws:\n        kernel.tensors.matmul(x, y, z)\n";

    #[test]
    fn a_method_body_tensor_call_is_inferred() {
        let program = run(METHOD);
        assert_eq!(method_quals(&program, "Op", "go").0, vec![G, G, U]);
    }

    #[test]
    fn qualifiers_propagate_through_a_method_call() {
        let src = format!("{METHOD}def work() throws:\n    mut o = Op()\n{BODY_METHOD}    with c:\n        print \"{{c[0]}}\"\n");
        let program = run(&src);
        assert_eq!(method_quals(&program, "Op", "go").0, vec![G, G, U]);
        assert_eq!(fn_quals(&program, "work").1[1..], [G, G, U]);
    }

    const BODY_METHOD: &str = "    let [float32, k = 2, m = 2] a = [0.0 for ..<4]\n    let [float32, n = 2, k = 2] b = [0.0 for ..<4]\n    mut [float32, n = 2, m = 2] c = [0.0 for ..<4]\n    o.go(a, b, c)\n";

    #[test]
    fn a_method_is_followed_through_self() {
        let src = "struct Op:\n    def inner([float32, k = 2, m = 2] x, [float32, n = 2, k = 2] y, mut [float32, n = 2, m = 2] z) throws:\n        kernel.tensors.matmul(x, y, z)\n    def outer([float32, k = 2, m = 2] p, [float32, n = 2, k = 2] q, mut [float32, n = 2, m = 2] r) throws:\n        self.inner(p, q, r)\n";
        let program = run(src);
        assert_eq!(method_quals(&program, "Op", "outer").0, vec![G, G, U]);
    }

    #[test]
    fn an_ambiguous_method_name_is_not_followed() {
        let src = format!("{METHOD}struct Other:\n    def go(int x):\n        print \"{{x}}\"\ndef work() throws:\n    mut o = Op()\n{BODY_METHOD}");
        let got = fn_quals(&run(&src), "work").1;
        assert_eq!(got[1..], [G, G, U].map(|_| None::<OwnerQual>));
    }

    #[test]
    fn a_receiver_of_another_struct_is_not_followed() {
        let src = format!("{METHOD}def work() throws:\n    let int o = 1\n{BODY_METHOD}");
        let got = fn_quals(&run(&src), "work").1;
        assert!(got[1..].iter().all(Option::is_none), "{got:?}");
    }

    const TOP: &str = "let [float32, k = 2, m = 2] a = [0.0 for ..<4]\nlet [float32, n = 2, k = 2] b = [0.0 for ..<4]\nmut [float32, n = 2, m = 2] c = [0.0 for ..<4]\n";

    #[test]
    fn a_top_level_binding_used_in_a_function_tensor_call_is_inferred() {
        let src = format!("{TOP}def run() throws:\n    kernel.tensors.matmul(a, b, c)\nrun()\n");
        let got = types(&src);
        assert_eq!((qual_of(&got, "a"), qual_of(&got, "b"), qual_of(&got, "c")), (G, G, U));
    }

    #[test]
    fn a_top_level_binding_passed_on_from_a_function_joins_the_callee_group() {
        let src = format!("{CALLEE}{TOP}def run() throws:\n    mm(a, b, c)\nrun()\n");
        let got = types(&src);
        assert_eq!((qual_of(&got, "a"), qual_of(&got, "b"), qual_of(&got, "c")), (G, G, U));
    }

    #[test]
    fn a_function_declaring_the_same_name_shadows_the_top_level_binding() {
        let src = format!("{TOP}def run() throws:\n    let [float32, k = 2, m = 2] a = [1.0 for ..<4]\n    kernel.tensors.matmul(a, b, c)\nrun()\n");
        let got = types(&src);
        assert_eq!(qual_of(&got, "a"), None);
    }

    #[test]
    fn a_top_level_use_in_a_method_still_counts_as_a_possible_write() {
        let src = format!("{TOP}struct S:\n    def go() throws:\n        print \"{{c[0]}}\"\nkernel.tensors.matmul(a, b, c)\n");
        assert!(warnings_of(&src).is_empty());
    }
    #[test]
    fn bare_kernel_on_a_method_resolves_like_on_a_free_function() {
        let src = "struct Op:\n    def go([float32, k = 2, m = 2]'kernel x, [float32, n = 2, k = 2]'kernel y, mut [float32, n = 2, m = 2]'kernel z) throws:\n        kernel.tensors.matmul(x, y, z)\n";
        assert!(errors_of(src).is_empty());
        assert_eq!(method_quals(&run(src), "Op", "go").0, vec![G, G, U]);
    }

    #[test]
    fn bare_kernel_on_a_method_return_follows_what_is_returned() {
        let src = "struct Op:\n    def [float32, n = 2, m = 2]'kernel go([float32, k = 2, m = 2]'kernel x, [float32, n = 2, k = 2]'kernel y) throws:\n        mut [float32, n = 2, m = 2] z = [0.0 as float32 for ..<4]\n        kernel.tensors.matmul(x, y, z)\n        z\n";
        assert!(errors_of(src).is_empty());
        let program = run(src);
        let m = program.items.iter().find_map(|i| match i {
            Item::Struct(st) => st.methods.first(),
            _ => None,
        }).unwrap();
        assert_eq!(m.return_ty.as_ref().and_then(|t| t.without_mut().gpu_resident_qual().cloned()), U);
    }

    #[test]
    fn bare_kernel_on_a_non_array_method_parameter_is_an_error() {
        let got = errors_of("struct Op:\n    def go(float'kernel x):\n        print \"{x}\"\n");
        assert!(got.iter().any(|e| e.contains("a non-array parameter")), "{got:?}");
    }

    #[test]
    fn bare_kernel_on_an_extension_method_is_still_an_error() {
        let got = errors_of("ext Vec:\n    def go([float32, k = 2, m = 2]'kernel x):\n        print \"{x}\"\n");
        assert!(got.iter().any(|e| e.contains("a method")), "{got:?}");
    }
}
