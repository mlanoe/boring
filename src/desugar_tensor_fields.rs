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
use crate::desugar_tensor_quals::{all_lets, collect_var_types, type_owner};
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
    /// Methods generated for calls taking `obj.field` operands outside a method: `(struct, method)`.
    pending: Vec<(String, FnDecl)>,
}

pub fn lower_field_operands(items: &mut Vec<Item>) -> FieldAliases {
    let mut out = FieldAliases::default();
    let mut counter = 0usize;
    lower_items(items, &mut counter, &mut out);
    out
}

/// What the pass knows about the unit it rewrites.
struct Tables {
    /// struct -> its array-like fields
    structs: HashMap<String, HashMap<String, Type>>,
    /// free function -> the struct (or `""`) its declared return type names
    fn_returns: HashMap<String, String>,
}

fn lower_items(items: &mut Vec<Item>, counter: &mut usize, out: &mut FieldAliases) {
    for item in items.iter_mut() {
        if let Item::Mod(m) = item {
            lower_items(&mut m.items, counter, out);
        }
    }
    let mut tables = Tables {
        structs: HashMap::new(),
        fn_returns: HashMap::new(),
    };
    for item in items.iter() {
        match item {
            Item::Struct(st) => {
                let fields: HashMap<String, Type> = st
                    .fields
                    .iter()
                    .filter(|f| is_array_like(&f.ty))
                    .map(|f| (f.name.clone(), f.ty.clone()))
                    .collect();
                if !fields.is_empty() {
                    tables.structs.insert(st.name.clone(), fields);
                }
            }
            Item::Fn(f) => {
                if let Some(t) = f.return_ty.as_ref().and_then(type_owner) {
                    tables.fn_returns.insert(f.name.clone(), t);
                }
            }
            _ => {}
        }
    }
    if tables.structs.is_empty() {
        return;
    }

    for item in items.iter_mut() {
        match item {
            Item::Fn(f) => lower_fn(f, None, &tables, counter, out),
            Item::Struct(st) => {
                let owner = st.name.clone();
                for m in &mut st.methods {
                    lower_fn(m, Some(&owner), &tables, counter, out);
                }
            }
            _ => {}
        }
    }

    // The top level: statements run in `main`, where `obj.field` operands can appear too.
    let top: Vec<Stmt> = items
        .iter()
        .filter_map(|i| match i {
            Item::Let(l) => Some(Stmt::Let(l.clone())),
            Item::Stmt(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let ctx = Ctx {
        owner: None,
        declared: HashSet::new(),
        mutating: true,
        var_types: collect_var_types(&[], &top, &tables.fn_returns),
        var_mut: mutability(&[], &top),
        types: declared_types(&[], &top),
        tables: &tables,
    };
    let old = std::mem::take(items);
    for item in old {
        match item {
            Item::Stmt(s) => {
                let mut one = vec![s];
                rewrite_block(&mut one, &ctx, counter, out);
                // A top-level alias `let` stays a top-level item (what the GPU host synthesizers
                // read declared operand types from); the call and its write-back stay statements.
                items.extend(one.into_iter().map(|s| match s {
                    Stmt::Let(l) => Item::Let(l),
                    other => Item::Stmt(other),
                }));
            }
            other => items.push(other),
        }
    }

    // The methods generated for calls outside any method: appended to their struct, then lowered
    // like any other method (their `self.field` operands become aliases).
    for (owner, method) in std::mem::take(&mut out.pending) {
        for item in items.iter_mut() {
            let Item::Struct(st) = item else { continue };
            if st.name != owner {
                continue;
            }
            st.methods.push(method);
            let last = st.methods.len() - 1;
            lower_fn(&mut st.methods[last], Some(&owner), &tables, counter, out);
            break;
        }
    }
}

fn is_array_like(ty: &Type) -> bool {
    let mut t = ty.without_mut();
    while let Type::Qualified(inner, _) = t { t = inner.without_mut(); }
    matches!(t, Type::LabeledArray(..) | Type::Array(_))
}

/// name -> content-mutable, for the parameters and `let`s of a scope (a name declared twice with
/// different mutability is treated as mutable: the error is only for a name known immutable).
fn mutability(params: &[Param], stmts: &[Stmt]) -> HashMap<String, bool> {
    let mut out: HashMap<String, bool> = HashMap::new();
    let mut note = |name: &str, m: bool| {
        *out.entry(name.to_string()).or_insert(false) |= m;
    };
    for p in params {
        note(
            &p.name,
            p.ty.as_ref().is_some_and(Type::is_mut) || (p.mutable && !p.rebindable) || p.var_mut,
        );
    }
    for l in all_lets(stmts) {
        note(
            &l.name,
            l.ty.as_ref().is_some_and(Type::is_mut)
                || l.var_mut
                || matches!(l.binding, BindingKind::Mut),
        );
    }
    out
}

/// name -> explicit declared type, for the parameters and `let`s of a scope.
fn declared_types(params: &[Param], stmts: &[Stmt]) -> HashMap<String, Type> {
    let mut out = HashMap::new();
    for p in params {
        if let Some(t) = &p.ty {
            out.insert(p.name.clone(), t.clone());
        }
    }
    for l in all_lets(stmts) {
        if let Some(t) = &l.ty {
            out.insert(l.name.clone(), t.clone());
        }
    }
    out
}

struct Ctx<'a> {
    /// The struct whose method is rewritten; `None` in a free function and at the top level.
    owner: Option<&'a str>,
    declared: HashSet<String>,
    mutating: bool,
    /// Struct of each local whose type is evident, and whether its binding is content-mutable.
    var_types: HashMap<String, Option<String>>,
    var_mut: HashMap<String, bool>,
    /// Declared (explicit) type of each parameter and `let` of the scope.
    types: HashMap<String, Type>,
    tables: &'a Tables,
}

impl Ctx<'_> {
    /// The array-like fields of the struct whose method is rewritten.
    fn own_fields(&self) -> Option<&HashMap<String, Type>> {
        self.owner.and_then(|o| self.tables.structs.get(o))
    }
}

fn lower_fn(
    f: &mut FnDecl,
    owner: Option<&str>,
    tables: &Tables,
    counter: &mut usize,
    out: &mut FieldAliases,
) {
    let mut declared: HashSet<String> = f.params.iter().map(|p| p.name.clone()).collect();
    let mut body = f.body.clone();
    {
        let mut on_expr = |_: &mut Expr| {};
        let mut on_stmt = |s: &mut Stmt| {
            if let Stmt::Let(l) = s {
                declared.insert(l.name.clone());
            }
        };
        crate::ast_walk::Walker {
            expr: &mut on_expr,
            stmt: &mut on_stmt,
        }
        .stmts(&mut body);
    }
    let ctx = Ctx {
        owner,
        declared,
        mutating: f.mutating,
        var_types: collect_var_types(&f.params, &f.body, &tables.fn_returns),
        var_mut: mutability(&f.params, &f.body),
        types: declared_types(&f.params, &f.body),
        tables,
    };
    rewrite_block(&mut f.body, &ctx, counter, out);
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

/// A struct field used as a host tensor operand.
struct Operand {
    /// The expression the field is read from: `self`, or a local (`layer`).
    base: Expr,
    /// `self` or the local's name: part of the alias key, and the error text.
    base_name: String,
    owner: String,
    field: String,
    ty: Type,
}

fn field_of(e: &Expr, ctx: &Ctx) -> Option<Operand> {
    let self_base = |e: &Expr| Expr {
        kind: ExprKind::Var("self".into()),
        line: e.line,
        col: e.col,
        len: e.len,
    };
    let method_field = |f: &str| -> Option<(String, Type)> {
        let owner = ctx.owner?;
        ctx.own_fields()?
            .get(f)
            .map(|t| (owner.to_string(), t.clone()))
    };
    match &e.kind {
        // A bare `w` naming a field the method does not shadow.
        ExprKind::Var(n) if !ctx.declared.contains(n) => {
            let (owner, ty) = method_field(n)?;
            Some(Operand {
                base: self_base(e),
                base_name: "self".into(),
                owner,
                field: n.clone(),
                ty,
            })
        }
        ExprKind::Field(base, f) => {
            let ExprKind::Var(b) = &base.kind else {
                return None;
            };
            if b == "self" {
                let (owner, ty) = method_field(f)?;
                return Some(Operand {
                    base: self_base(e),
                    base_name: "self".into(),
                    owner,
                    field: f.clone(),
                    ty,
                });
            }
            // `layer.w`: a local (or parameter) whose struct is evident.
            let owner = ctx.var_types.get(b)?.clone().filter(|t| !t.is_empty())?;
            let ty = ctx.tables.structs.get(&owner)?.get(f)?.clone();
            Some(Operand {
                base: (**base).clone(),
                base_name: b.clone(),
                owner,
                field: f.clone(),
                ty,
            })
        }
        _ => None,
    }
}

fn rewrite_call(call: &mut Expr, ctx: &Ctx, counter: &mut usize, out: &mut FieldAliases)
    -> Option<(Vec<Stmt>, Vec<Stmt>)>
{
    match lower_outside(call, ctx, counter, out) {
        Some(true) => return Some((Vec::new(), Vec::new())),
        // The error is recorded; the call is dropped so the checker does not add a misleading
        // second one about the operands it could not read.
        Some(false) => {
            call.kind = ExprKind::Bool(true);
            return Some((Vec::new(), Vec::new()));
        }
        None => {}
    }
    let (line, col, len) = (call.line, call.col, call.len);
    let ExprKind::MethodCall(recv, method, args) = &mut call.kind else { return None };
    if !matches!(method.as_str(), "matmul" | "mma" | "linear") { return None; }
    let ExprKind::Field(base, ns) = &recv.kind else { return None };
    if ns != "tensor" || !matches!(&base.kind, ExprKind::Var(n) if n == "gpu") { return None; }
    let operand_count = args.iter().position(|a| a.label.is_some()).unwrap_or(args.len());
    let dest_index = operand_count.checked_sub(1)?;
    let mut aliases: HashMap<(String, String), String> = HashMap::new();
    let (mut pre, mut post) = (Vec::new(), Vec::new());
    for index in 0..operand_count {
        if args[index].spread || args[index].default_rest {
            continue;
        }
        let Some(op) = field_of(&args[index].value, ctx) else {
            continue;
        };
        // `obj.field` operands were handled by `lower_outside`; one left here could not be.
        if op.base_name != "self" {
            continue;
        }
        let is_dest = index == dest_index;
        let key = (op.base_name.clone(), op.field.clone());
        let alias = match aliases.get(&key) {
            Some(a) => a.clone(),
            None => {
                let alias = format!("{ALIAS_PREFIX}{}_{counter}", op.field);
                *counter += 1;
                aliases.insert(key, alias.clone());
                out.by_alias
                    .insert(alias.clone(), (op.owner.clone(), op.field.clone()));
                out.used.insert((op.owner.clone(), op.field.clone()));
                pre.push(alias_let(&alias, &op, line, col, len));
                alias
            }
        };
        if is_dest {
            let shown = if op.base_name == "self" {
                op.field.clone()
            } else {
                format!("{}.{}", op.base_name, op.field)
            };
            if !op.ty.is_mut() {
                out.errors.push(crate::errors::SourceError::new(
                    format!("tensor destination `{shown}` is a struct field and must be declared `mut` (content-mutable) to receive the result"),
                    line, col, len,
                ));
            }
            if op.base_name == "self" {
                if !ctx.mutating {
                    out.errors.push(crate::errors::SourceError::new(
                        format!("tensor destination `{shown}` is a struct field the call writes back into: the method must be `def`, not `req`"),
                        line, col, len,
                    ));
                }
            } else if ctx.var_mut.get(&op.base_name) == Some(&false) {
                out.errors.push(crate::errors::SourceError::new(
                    format!("tensor destination `{shown}` is a struct field the call writes back into: `{}` must be declared `mut` (or `var mut`)", op.base_name),
                    line, col, len,
                ));
            }
            post.push(write_back(&op, &alias, line, col, len));
        }
        args[index].value = Expr { kind: ExprKind::Var(alias), line, col, len };
    }
    if pre.is_empty() { None } else { Some((pre, post)) }
}

fn alias_let(alias: &str, op: &Operand, line: usize, col: usize, len: usize) -> Stmt {
    let template =
        crate::parser::parse(crate::lexer::lex("let __tf_template = 0\n").expect("template lexes"))
            .expect("template parses");
    let Some(Item::Let(mut l)) = template.items.into_iter().next() else {
        unreachable!("template is a let")
    };
    l.name = alias.to_string();
    l.binding = if op.ty.is_mut() {
        BindingKind::Mut
    } else {
        BindingKind::Let
    };
    l.ty = Some(op.ty.clone());
    l.line = line;
    l.col = col;
    l.value = Some(Expr {
        kind: ExprKind::Field(
            Box::new(Expr {
                line,
                col,
                len,
                ..op.base.clone()
            }),
            op.field.clone(),
        ),
        line,
        col,
        len,
    });
    Stmt::Let(l)
}

fn write_back(op: &Operand, alias: &str, line: usize, col: usize, len: usize) -> Stmt {
    let var = |n: &str| Expr {
        kind: ExprKind::Var(n.into()),
        line,
        col,
        len,
    };
    Stmt::Expr(Expr {
        kind: ExprKind::Assign(
            Box::new(Expr {
                kind: ExprKind::Field(
                    Box::new(Expr {
                        line,
                        col,
                        len,
                        ..op.base.clone()
                    }),
                    op.field.clone(),
                ),
                line,
                col,
                len,
            }),
            Box::new(var(alias)),
        ),
        line, col, len,
    })
}

fn is_literal(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) => true,
        ExprKind::UnaryOp(_, inner) => is_literal(inner),
        _ => false,
    }
}

/// A call with an `obj.field` operand outside a method of its struct: the call moves into a
/// generated method of that struct (`__tf_call_N`), where the operands are `self.field`s — the
/// form every backend already handles — and is replaced by `obj.__tf_call_N(..)`. The other
/// operands (plain variables) and any non-literal labeled argument become parameters of it.
/// Returns `true` when the call was replaced; on an error it records it and leaves the call alone.
fn lower_outside(call: &mut Expr, ctx: &Ctx, counter: &mut usize, out: &mut FieldAliases) -> Option<bool> {
    let (line, col, len) = (call.line, call.col, call.len);
    let ExprKind::MethodCall(recv, method, args) = &call.kind else {
        return None;
    };
    if !matches!(method.as_str(), "matmul" | "mma" | "linear") {
        return None;
    }
    let ExprKind::Field(base, ns) = &recv.kind else {
        return None;
    };
    if ns != "tensor" || !matches!(&base.kind, ExprKind::Var(n) if n == "gpu") {
        return None;
    }
    let operand_count = args.iter().position(|a| a.label.is_some()).unwrap_or(args.len());
    let Some(dest_index) = operand_count.checked_sub(1) else {
        return None;
    };
    let ops: Vec<Option<Operand>> = (0..operand_count)
        .map(|i| {
            if args[i].spread || args[i].default_rest {
                None
            } else {
                field_of(&args[i].value, ctx)
            }
        })
        .collect();
    if !ops.iter().flatten().any(|o| o.base_name != "self") {
        return None;
    }
    let mut error = |msg: String| {
        out.errors.push(crate::errors::SourceError::new(msg, line, col, len));
        Some(false)
    };
    if ops.iter().flatten().any(|o| o.base_name == "self") {
        return error("a tensor call cannot mix fields of `self` with fields of another object: pass them as parameters".into());
    }
    let first = ops.iter().flatten().next().expect("one outside operand");
    if ops.iter().flatten().any(|o| o.base_name != first.base_name) {
        return error("a tensor call cannot take fields of two different objects: pass them as parameters".into());
    }
    let (base_name, owner, base_expr) = (first.base_name.clone(), first.owner.clone(), first.base.clone());

    // The destination: a field is written back (so it must be `mut`, on a `mut` object).
    let dest_is_field = ops[dest_index].is_some();
    if let Some(dest) = &ops[dest_index] {
        let shown = format!("{base_name}.{}", dest.field);
        if !dest.ty.is_mut() {
            return error(format!("tensor destination `{shown}` is a struct field and must be declared `mut` (content-mutable) to receive the result"));
        }
        if ctx.var_mut.get(&base_name) == Some(&false) {
            return error(format!("tensor destination `{shown}` is a struct field the call writes back into: `{base_name}` must be declared `mut` (or `var mut`)"));
        }
    }

    let template = crate::parser::parse(
        crate::lexer::lex("struct __tf_template:\n    def __tf_m(int __tf_p):\n        print 0\n").expect("template lexes"),
    )
    .expect("template parses");
    let Some(Item::Struct(st)) = template.items.into_iter().next() else {
        unreachable!("template is a struct")
    };
    let mut method_decl = st.methods.into_iter().next().expect("template method");
    let param_template = method_decl.params[0].clone();
    let arg_template = args[0].clone();

    let var = |n: &str| Expr { kind: ExprKind::Var(n.into()), line, col, len };
    let new_param = |name: &str, ty: Type, mutable: bool| Param {
        name: name.into(),
        ty: Some(ty),
        mutable,
        rebindable: false,
        var_mut: false,
        owned: false,
        variadic: false,
        default: None,
        line,
        col,
        ..param_template.clone()
    };
    let new_arg = |value: Expr| Arg {
        label: None,
        value,
        spread: false,
        default_rest: false,
        ..arg_template.clone()
    };

    let mut params: Vec<Param> = Vec::new();
    let mut call_args: Vec<Arg> = Vec::new();
    let mut body_args: Vec<Arg> = args.clone();
    for i in 0..operand_count {
        if let Some(op) = &ops[i] {
            body_args[i].value = Expr {
                kind: ExprKind::Field(Box::new(var("self")), op.field.clone()),
                line,
                col,
                len,
            };
            continue;
        }
        let ExprKind::Var(v) = &args[i].value.kind else {
            return error("a tensor operand must be a variable or a field of one object".into());
        };
        let Some(ty) = ctx.types.get(v) else {
            return error(format!("tensor operand `{v}` needs an explicit array type to be passed alongside a struct field"));
        };
        let name = format!("__tf_a{i}");
        let is_dest = i == dest_index;
        let ty = ty.without_mut().clone();
        params.push(new_param(&name, if is_dest { Type::Mut(Box::new(ty)) } else { ty }, is_dest));
        call_args.push(new_arg(args[i].value.clone()));
        body_args[i].value = var(&name);
    }
    for j in operand_count..args.len() {
        if is_literal(&args[j].value) {
            continue;
        }
        let label = args[j].label.clone().unwrap_or_default();
        let name = format!("__tf_l_{label}");
        params.push(new_param(&name, Type::Int, false));
        call_args.push(new_arg(args[j].value.clone()));
        body_args[j].value = var(&name);
    }

    let method_name = format!("__tf_call_{counter}");
    *counter += 1;
    method_decl.name = method_name.clone();
    method_decl.params = params;
    method_decl.throws = true;
    method_decl.mutating = dest_is_field;
    method_decl.line = line;
    method_decl.col = col;
    method_decl.body = vec![Stmt::Expr(Expr {
        kind: ExprKind::MethodCall(recv.clone(), method.clone(), body_args),
        line,
        col,
        len,
    })];
    out.pending.push((owner, method_decl));
    call.kind = ExprKind::MethodCall(Box::new(Expr { line, col, len, ..base_expr }), method_name, call_args);
    Some(true)
}
