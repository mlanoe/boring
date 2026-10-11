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

//! Lowers `view` declarations to ordinary structs (docs/design-notes/boring-ui-draft.md, "State
//! slots and binding handles"). Runs once on the merged `Program`, after `view_perm`, so the
//! checker, interpreter and transpilers only ever see plain structs.
//!
//! ```text
//! view Badge:                        struct Badge__State:           # mounted-state layout
//!     string title                       var int count = 0          # `state` fields, declaration order
//!     state int count = 0        ==>
//!                                    struct Badge:                  # ephemeral description
//!     def bump():                        string title               # inputs only
//!         count += 1                     def bump(mut Badge__State __st):
//!     body():                                __st.count += 1
//!         Text("{title}{count}")         body(mut Badge__State __st):
//!                                            Text("{title}{__st.count}")
//! ```
//!
//! The description (`Badge`) holds inputs only; the persistent values live in `Badge__State`,
//! one instance per mounted node, seeded once at mount by `Badge__State()` (the declaration's
//! default seed). Every method and `body()` takes the mounted state as an explicit trailing
//! `__st` parameter; bare reads/writes of a `state` field and `self.field` are rewritten to
//! `__st.field`, every method becomes `req` (the description itself is never written), and `self.method(...)` calls forward `__st`. Callback closures capture `__st`
//! like any other local, which is the "callback `self` as mounted-view context" of the draft.
//!
//! ## Slice scope (what is NOT lowered yet)
//!
//! - **`bind` / `env` fields** and **native views** are left as `view` declarations;
//!   `boring build` still reports "cannot be transpiled yet" for them.
//! - **Parent-supplied seeds** (`Counter(count = expensive())`): a `state` field is no longer a
//!   constructor argument, so that is an ordinary "unknown field" error today.
//! - **Invalidation**: there is no write-site invalidation call. The runtime (layer 2) treats any
//!   callback that received `__st` as having possibly written it, which is conservative but
//!   correct; precise per-slot invalidation comes with the runtime's slot API.
//! - A `state` field needs an initializer (there is no default seed otherwise), and a local,
//!   parameter or loop variable may not shadow a `state` field name inside the view.

use std::collections::HashSet;

use crate::ast::*;
use crate::ast_walk::Walker;
use crate::parser::ParseError;

const ST: &str = "__st";

fn err(msg: String, line: usize, col: usize) -> ParseError {
    ParseError::Generic { msg, line, col, len: 1 }
}

/// A `mut <State> __st` parameter, built by parsing a one-line signature so its exact
/// permission encoding always matches what the parser produces for source-written code.
fn state_param(state_ty: &str, line: usize, col: usize) -> Param {
    let src = format!("def f(mut {state_ty} {ST}):\n    pass\n");
    let tokens = crate::lexer::lex(&src).expect("generated signature lexes");
    let program = crate::parser::parse(tokens).expect("generated signature parses");
    for item in program.items {
        if let Item::Fn(mut f) = item {
            let mut p = f.params.remove(0);
            p.line = line;
            p.col = col;
            return p;
        }
    }
    unreachable!("generated signature has one fn")
}

fn lowerable(s: &StructDecl) -> bool {
    s.is_view
        && !s.methods.iter().any(|m| m.is_native)
        && !s.fields.iter().any(|f| matches!(f.view_kind, Some(ViewFieldKind::Bind | ViewFieldKind::Env)))
}

fn lower_items(items: Vec<Item>) -> Result<Vec<Item>, ParseError> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            Item::Struct(s) if lowerable(&s) => {
                let (state, view) = lower_view(s)?;
                if let Some(state) = state { out.push(Item::Struct(state)); }
                out.push(Item::Struct(view));
            }
            Item::Mod(mut m) => { m.items = lower_items(m.items)?; out.push(Item::Mod(m)); }
            other => out.push(other),
        }
    }
    Ok(out)
}

fn lower_view(mut s: StructDecl) -> Result<(Option<StructDecl>, StructDecl), ParseError> {
    let state_name = format!("{}__State", s.name);
    let state_fields: Vec<FieldDecl> = s.fields.iter()
        .filter(|f| f.view_kind == Some(ViewFieldKind::State))
        .cloned()
        .collect();
    for f in &state_fields {
        if f.default.is_none() {
            return Err(err(
                format!("`state {}` needs an initializer: its seed is the only source of its first value", f.name),
                f.line, f.col,
            ));
        }
    }
    let names: HashSet<String> = state_fields.iter().map(|f| f.name.clone()).collect();

    let state_decl = (!state_fields.is_empty()).then(|| {
        let fields = state_fields.iter().map(|f| {
            let mut f = f.clone();
            f.view_kind = None;
            f.explicit_perm = false;
            f.is_pub = true;
            f
        }).collect();
        StructDecl {
            name: state_name.clone(),
            is_pub: s.is_pub,
            is_view: false,
            is_native: false,
            protocols: vec![],
            fields,
            inits: vec![],
            methods: vec![],
            conversions: vec![],
            type_params: vec![],
            where_clause: vec![],
            setters: vec![],
            type_methods: vec![],
            type_vars: vec![],
            assoc_type_defs: vec![],
            attrs: vec![],
            line: s.line,
            col: s.col,
        }
    });

    s.fields.retain(|f| f.view_kind.is_none());
    s.is_view = false;

    if state_decl.is_none() {
        return Ok((None, s));
    }

    let own_methods: HashSet<String> = s.methods.iter().map(|m| m.name.clone()).collect();
    let mut failure: Option<ParseError> = None;
    for m in &mut s.methods {
        for p in &m.params {
            if names.contains(&p.name) {
                failure.get_or_insert_with(|| err(
                    format!("parameter `{}` shadows `state {}` of view '{}'", p.name, p.name, s.name),
                    m.line, m.col,
                ));
            }
        }
        m.params.push(state_param(&state_name, m.line, m.col));
        // The description is never mutated (inputs are immutable, state lives in `__st`), so
        // even a method that writes state only needs `&self`.
        m.mutating = false;
        rewrite_body(&mut m.body, &names, &own_methods, &s.name, &mut failure);
    }
    match failure {
        Some(e) => Err(e),
        None => Ok((state_decl, s)),
    }
}

fn rewrite_body(
    body: &mut [Stmt],
    names: &HashSet<String>,
    own_methods: &HashSet<String>,
    view: &str,
    failure: &mut Option<ParseError>,
) {
    // Pass 1 — on the untouched body: a binding that reuses a `state` name would be rewritten
    // into `__st.name` by pass 2, silently changing what it refers to.
    let mut shadow: Vec<(String, usize, usize)> = Vec::new();
    {
        let shadow_cell = std::cell::RefCell::new(&mut shadow);
        let mut on_expr = |e: &mut Expr| match &e.kind {
            ExprKind::Closure(params, ..) => {
                for p in params.iter().filter(|p| names.contains(&p.name)) {
                    shadow_cell.borrow_mut().push((p.name.clone(), p.line, p.col));
                }
            }
            ExprKind::ArrayComp { var, .. } | ExprKind::ArrayCompIter { var, .. } if names.contains(var) => {
                shadow_cell.borrow_mut().push((var.clone(), e.line, e.col));
            }
            _ => {}
        };
        let mut on_stmt = |st: &mut Stmt| match st {
            Stmt::Let(l) if names.contains(&l.name) => {
                shadow_cell.borrow_mut().push((l.name.clone(), l.line, l.col));
            }
            Stmt::LetDestructure(l) => {
                for b in l.bindings.iter().filter(|b| names.contains(&b.name)) {
                    shadow_cell.borrow_mut().push((b.name.clone(), l.line, l.col));
                }
            }
            Stmt::For(f) => {
                for v in f.vars.iter().filter(|v| names.contains(*v)) {
                    shadow_cell.borrow_mut().push((v.clone(), f.line, f.col));
                }
            }
            _ => {}
        };
        Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(body);
    }
    if let Some((n, line, col)) = shadow.into_iter().next() {
        failure.get_or_insert_with(|| err(
            format!("local `{n}` shadows `state {n}` of view '{view}' — rename it"),
            line, col,
        ));
        return;
    }

    // Pass 2 — rewrite state accesses and forward `__st` through own-method calls.
    let st_var = |line, col, len| Expr { kind: ExprKind::Var(ST.to_string()), line, col, len };
    let mut on_expr = |e: &mut Expr| {
        let (line, col, len) = (e.line, e.col, e.len);
        match &mut e.kind {
            ExprKind::Var(n) if names.contains(n) => {
                let field = n.clone();
                e.kind = ExprKind::Field(Box::new(st_var(line, col, len)), field);
            }
            ExprKind::Field(obj, f) if names.contains(f)
                && matches!(&obj.kind, ExprKind::Var(n) if n == "self") =>
            {
                obj.kind = ExprKind::Var(ST.to_string());
            }
            ExprKind::MethodCall(obj, m, args) if own_methods.contains(m)
                && matches!(&obj.kind, ExprKind::Var(n) if n == "self") =>
            {
                args.push(Arg { label: None, value: st_var(line, col, len), spread: false, default_rest: false });
            }
            _ => {}
        }
    };
    let mut noop = |_: &mut Stmt| {};
    Walker { expr: &mut on_expr, stmt: &mut noop }.stmts(body);
}

/// Lowers every non-native `view` that uses only inputs and `state` fields.
pub fn lower_views(mut program: Program) -> Result<Program, ParseError> {
    if let Some(e) = crate::checker::check_view_declarations(&program).into_iter().next() {
        return Err(err(e.message, e.line, e.col));
    }
    program.items = lower_items(program.items)?;
    Ok(program)
}
