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

//! Struct fields as `kernel.tensors.matmul/mma/linear` host operands.
//!
//! The host-dispatch machinery (checker, interpreter, CPU/SIMD emission, the four GPU host
//! synthesizers) only knows plain variables, whose declared type carries the residency. Rather
//! than teach each of them about fields, this early AST pass (it runs first inside
//! `desugar_tensor_quals::infer_tensor_operand_quals`, so every consumer sees its output)
//! rewrites a call inside a struct method whose operand is `self.w`, or a bare `w` naming a
//! field the method does not shadow, into the local form they already handle:
//!
//! ```text
//! let <field type> __tf_w_0 = self.w          # alias, typed like the field
//! mut <field type> __tf_out_1 = self.out
//! kernel.tensors.matmul(x, __tf_w_0, __tf_out_1)
//! self.out = __tf_out_1                        # destination only: write-back
//! ```
//!
//! * **Mutation.** A destination field *is* visible to the caller afterwards: the call replaces
//!   the array and the replacement is stored back into the field. The field must be `mut`
//!   (content-mutable; the write-back is exempt from the "not reassignable" rule a plain
//!   `self.out = ..` obeys, as the call mutates the field's content, not its binding), and the
//!   method must be `def`.
//! * **Several methods / calls.** Each call gets its own alias, so the field stays the single
//!   source of truth; the aliases of one field are tied to one residency by the inference
//!   (`desugar_tensor_quals`), which then writes the result back onto the field declaration
//!   when it was left unqualified or written with a bare `'kernel`.
//! * **Same field as input and destination** is the existing "destination must not alias an
//!   input" error (one alias per distinct field per call).
//! * Aliases are `__tf_`-prefixed; the interpreter recognises the write-back by that prefix.

use crate::ast::*;
use std::collections::{HashMap, HashSet};

/// The alias prefix; also how the interpreter spots the write-back of a destination field.
pub const ALIAS_PREFIX: &str = "__tf_";

#[derive(Default)]
pub struct FieldAliases {
    /// alias name -> (struct, field)
    pub by_alias: HashMap<String, (String, String)>,
    /// every (struct, field) that got at least one alias
    pub used: HashSet<(String, String)>,
    pub errors: Vec<crate::errors::SourceError>,
}

pub fn lower_field_operands(items: &mut [Item]) -> FieldAliases {
    let mut out = FieldAliases::default();
    let mut counter = 0usize;
    lower_items(items, &mut counter, &mut out);
    out
}

fn lower_items(items: &mut [Item], counter: &mut usize, out: &mut FieldAliases) {
    for item in items.iter_mut() {
        match item {
            Item::Mod(m) => lower_items(&mut m.items, counter, out),
            Item::Struct(st) => lower_struct(st, counter, out),
            _ => {}
        }
    }
}

fn is_array_like(ty: &Type) -> bool {
    let mut t = ty.without_mut();
    while let Type::Qualified(inner, _) = t { t = inner.without_mut(); }
    matches!(t, Type::LabeledArray(..) | Type::Array(_))
}

struct Ctx<'a> {
    owner: &'a str,
    fields: &'a HashMap<String, Type>,
    declared: HashSet<String>,
    mutating: bool,
}

fn lower_struct(st: &mut StructDecl, counter: &mut usize, out: &mut FieldAliases) {
    let fields: HashMap<String, Type> = st.fields.iter()
        .filter(|f| is_array_like(&f.ty))
        .map(|f| (f.name.clone(), f.ty.clone()))
        .collect();
    if fields.is_empty() { return; }
    let owner = st.name.clone();
    for m in &mut st.methods {
        let mut declared: HashSet<String> = m.params.iter().map(|p| p.name.clone()).collect();
        let mut body = m.body.clone();
        {
            let mut on_expr = |_: &mut Expr| {};
            let mut on_stmt = |s: &mut Stmt| {
                if let Stmt::Let(l) = s { declared.insert(l.name.clone()); }
            };
            crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut body);
        }
        let ctx = Ctx { owner: &owner, fields: &fields, declared, mutating: m.mutating };
        rewrite_block(&mut m.body, &ctx, counter, out);
    }
}

fn rewrite_block(stmts: &mut Vec<Stmt>, ctx: &Ctx, counter: &mut usize, out: &mut FieldAliases) {
    let old = std::mem::take(stmts);
    for mut stmt in old {
        match &mut stmt {
            Stmt::If(i) => {
                for (_, b) in &mut i.branches { rewrite_block(b, ctx, counter, out); }
                if let Some(e) = &mut i.else_body { rewrite_block(e, ctx, counter, out); }
            }
            Stmt::For(f) => rewrite_block(&mut f.body, ctx, counter, out),
            Stmt::While(w) => rewrite_block(&mut w.body, ctx, counter, out),
            Stmt::DoWhile(d) => rewrite_block(&mut d.body, ctx, counter, out),
            Stmt::Loop(l) => rewrite_block(&mut l.body, ctx, counter, out),
            Stmt::With(w) => rewrite_block(&mut w.body, ctx, counter, out),
            _ => {}
        }
        let Stmt::Expr(call) = &mut stmt else { stmts.push(stmt); continue };
        let Some((pre, post)) = rewrite_call(call, ctx, counter, out) else { stmts.push(stmt); continue };
        stmts.extend(pre);
        stmts.push(stmt);
        stmts.extend(post);
    }
}

fn field_of(e: &Expr, ctx: &Ctx) -> Option<String> {
    match &e.kind {
        ExprKind::Var(n) if ctx.fields.contains_key(n) && !ctx.declared.contains(n) => Some(n.clone()),
        ExprKind::Field(base, f)
            if matches!(&base.kind, ExprKind::Var(s) if s == "self") && ctx.fields.contains_key(f) => Some(f.clone()),
        _ => None,
    }
}

fn rewrite_call(call: &mut Expr, ctx: &Ctx, counter: &mut usize, out: &mut FieldAliases)
    -> Option<(Vec<Stmt>, Vec<Stmt>)>
{
    let (line, col, len) = (call.line, call.col, call.len);
    let ExprKind::MethodCall(recv, method, args) = &mut call.kind else { return None };
    if !matches!(method.as_str(), "matmul" | "mma" | "linear") { return None; }
    let ExprKind::Field(base, ns) = &recv.kind else { return None };
    if ns != "tensor" || !matches!(&base.kind, ExprKind::Var(n) if n == "gpu") { return None; }
    let operand_count = args.iter().position(|a| a.label.is_some()).unwrap_or(args.len());
    let dest_index = operand_count.checked_sub(1)?;
    let mut aliases: HashMap<String, String> = HashMap::new();
    let (mut pre, mut post) = (Vec::new(), Vec::new());
    for index in 0..operand_count {
        if args[index].spread || args[index].default_rest { continue; }
        let Some(field) = field_of(&args[index].value, ctx) else { continue };
        let is_dest = index == dest_index;
        let field_ty = &ctx.fields[&field];
        let alias = match aliases.get(&field) {
            Some(a) => a.clone(),
            None => {
                let alias = format!("{ALIAS_PREFIX}{field}_{counter}");
                *counter += 1;
                aliases.insert(field.clone(), alias.clone());
                out.by_alias.insert(alias.clone(), (ctx.owner.to_string(), field.clone()));
                out.used.insert((ctx.owner.to_string(), field.clone()));
                pre.push(alias_let(&alias, &field, field_ty, line, col, len));
                alias
            }
        };
        if is_dest {
            if !field_ty.is_mut() {
                out.errors.push(crate::errors::SourceError::new(
                    format!("tensor destination `{field}` is a struct field and must be declared `mut` (content-mutable) to receive the result"),
                    line, col, len,
                ));
            }
            if !ctx.mutating {
                out.errors.push(crate::errors::SourceError::new(
                    format!("tensor destination `{field}` is a struct field the call writes back into: the method must be `def`, not `req`"),
                    line, col, len,
                ));
            }
            post.push(write_back(&field, &alias, line, col, len));
        }
        args[index].value = Expr { kind: ExprKind::Var(alias), line, col, len };
    }
    if pre.is_empty() { None } else { Some((pre, post)) }
}

fn alias_let(alias: &str, field: &str, ty: &Type, line: usize, col: usize, len: usize) -> Stmt {
    let template = crate::parser::parse(crate::lexer::lex("let __tf_template = 0\n").expect("template lexes"))
        .expect("template parses");
    let Some(Item::Let(mut l)) = template.items.into_iter().next() else { unreachable!("template is a let") };
    l.name = alias.to_string();
    l.binding = if ty.is_mut() { BindingKind::Mut } else { BindingKind::Let };
    l.ty = Some(ty.clone());
    l.line = line;
    l.col = col;
    l.value = Some(Expr {
        kind: ExprKind::Field(Box::new(Expr { kind: ExprKind::Var("self".into()), line, col, len }), field.to_string()),
        line, col, len,
    });
    Stmt::Let(l)
}

fn write_back(field: &str, alias: &str, line: usize, col: usize, len: usize) -> Stmt {
    let var = |n: &str| Expr { kind: ExprKind::Var(n.into()), line, col, len };
    Stmt::Expr(Expr {
        kind: ExprKind::Assign(
            Box::new(Expr { kind: ExprKind::Field(Box::new(var("self")), field.to_string()), line, col, len }),
            Box::new(var(alias)),
        ),
        line, col, len,
    })
}
