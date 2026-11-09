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

//! `.bri` printer: expressions.
//!
//! An expression prints to a `String` whose first line carries no padding (the caller pads it)
//! and whose following lines (blocks of closures, `if`/`match` expressions, ...) carry their own
//! absolute padding, taken from `Printer::indent` (the indentation of the statement being printed).

use super::*;

pub(super) fn quote_str(s: &str) -> String {
    let mut out = String::from("\"");
    out.push_str(&escape_lit(s));
    out.push('"');
    out
}

fn escape_lit(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '{' => out.push_str("{{"),
            '}' => out.push_str("}}"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\0' => out.push_str("\\0"),
            c => out.push(c),
        }
    }
    out
}

pub(super) fn float_text(f: f64) -> R<String> {
    if !f.is_finite() {
        return unsupported("non-finite float literal", 0);
    }
    let s = format!("{f:?}");
    Ok(s)
}

/// The precedence level an expression sits at when printed without parentheses.
fn level_of(e: &Expr) -> u8 {
    match &e.kind {
        ExprKind::BinOp(op, _, _) => binop_info(op).1,
        ExprKind::UnaryOp(UnaryOp::Not, _) => P_NOT,
        ExprKind::UnaryOp(_, _) => P_UNARY,
        ExprKind::Int(n) if *n < 0 => P_UNARY,
        ExprKind::Float(f) if *f < 0.0 || (*f == 0.0 && f.is_sign_negative()) => P_UNARY,
        ExprKind::Range { .. } => P_RANGE,
        ExprKind::Pipe(..) => P_PIPE,
        ExprKind::Cast(..) | ExprKind::RelabelCast(..) | ExprKind::Field(..) | ExprKind::Index(..)
        | ExprKind::LabeledIndex(..) | ExprKind::Call(..) | ExprKind::MethodCall(..)
        | ExprKind::GenericCall(..) | ExprKind::OptionalField(..) | ExprKind::OptionalMethodCall(..) => P_POSTFIX,
        ExprKind::Else(..) | ExprKind::TryElse(..) | ExprKind::TryElseBlock(..)
        | ExprKind::Assign(..) | ExprKind::QuestionAssign(..) => P_ELSE,
        // Primaries that extend to the right: fine wherever `parse_or` is, parenthesised as an operand.
        ExprKind::Closure(..) | ExprKind::If(..) | ExprKind::Match(..) | ExprKind::Block(..) | ExprKind::Do(..)
        | ExprKind::Loop(..) | ExprKind::Task(..) | ExprKind::TaskWithTimeout(..) | ExprKind::KernelLaunch { .. }
        | ExprKind::TrailingArrayBlock { .. } => P_OR,
        _ => P_ATOM,
    }
}

impl Printer {
    /// An expression statement (an assignment is only legal here).
    pub(super) fn expr_stmt(&mut self, e: &Expr) -> R<String> {
        match &e.kind {
            ExprKind::Assign(lhs, rhs) => {
                let l = self.expr(lhs, P_POSTFIX)?;
                // `x = x op n` was written `x op= n`: fold it back (same AST).
                if let ExprKind::BinOp(op, a, b) = &rhs.kind {
                    let foldable = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
                        | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr);
                    if foldable && self.expr(a, P_POSTFIX).ok().as_deref() == Some(l.as_str()) && level_of(a) >= P_POSTFIX {
                        let sym = binop_info(op).0;
                        let r = self.expr(b, P_OR)?;
                        return Ok(format!("{l} {sym}= {r}"));
                    }
                }
                let r = self.expr(rhs, P_OR)?;
                Ok(format!("{l} = {r}"))
            }
            ExprKind::QuestionAssign(lhs, rhs) => {
                let l = self.expr(lhs, P_POSTFIX)?;
                let r = self.expr(rhs, P_OR)?;
                Ok(format!("{l} ?= {r}"))
            }
            // A statement starting with `if`/`match`/`loop`/`do` is the statement form, not an
            // expression statement: parenthesise the expression.
            ExprKind::If(_) | ExprKind::Match(_) | ExprKind::Loop(_) | ExprKind::Do(_) => {
                Ok(format!("({})", self.expr_raw(e)?))
            }
            _ => self.expr(e, P_ELSE),
        }
    }

    /// A left operand: like `expr`, and a construct that extends to the right (`if`, a closure, ...)
    /// is parenthesised, or it would swallow the operator.
    fn left(&mut self, e: &Expr, min: u8) -> R<String> {
        if matches!(e.kind, ExprKind::Closure(..) | ExprKind::If(..) | ExprKind::Match(..) | ExprKind::Block(..)
            | ExprKind::Do(..) | ExprKind::Loop(..) | ExprKind::Task(..) | ExprKind::TaskWithTimeout(..)
            | ExprKind::KernelLaunch { .. } | ExprKind::TrailingArrayBlock { .. })
        {
            return Ok(format!("({})", self.expr_raw(e)?));
        }
        self.expr(e, min)
    }

    /// Prints `e`, parenthesised when it binds looser than `min`.
    pub(super) fn expr(&mut self, e: &Expr, min: u8) -> R<String> {
        let text = self.expr_raw(e)?;
        if level_of(e) < min { Ok(format!("({text})")) } else { Ok(text) }
    }

    fn args(&mut self, args: &[Arg]) -> R<String> {
        let mut parts: Vec<String> = Vec::new();
        for a in args {
            parts.push(if a.spread {
                format!("..{}", self.expr(&a.value, P_OR)?)
            } else if a.default_rest {
                "_".to_string()
            } else if let Some(l) = &a.label {
                format!("{l} = {}", self.expr(&a.value, P_OR)?)
            } else {
                self.expr(&a.value, P_ELSE)?
            });
        }
        // An argument that spans lines (a closure with a block body) ends with its last
        // statement; the next argument / the closing paren go on their own line at the
        // statement's indentation.
        let pad = self.pad();
        let mut out = String::new();
        for (i, p) in parts.iter().enumerate() {
            if i > 0 {
                if parts[i - 1].contains('\n') { out.push('\n'); out.push_str(&pad); out.push_str(", "); } else { out.push_str(", "); }
            }
            out.push_str(p);
        }
        if parts.last().is_some_and(|p| p.contains('\n')) {
            out.push('\n');
            out.push_str(&pad);
        }
        Ok(out)
    }

    fn type_args(&self, callee: &Expr, types: &[Type]) -> R<String> {
        let labels = match &callee.kind {
            ExprKind::Var(n) => self.qual_labels.get(n),
            _ => None,
        };
        let mut parts = Vec::new();
        for (i, t) in types.iter().enumerate() {
            match t {
                Type::QualArg(chain) => {
                    let label = labels.and_then(|l| l.get(i)).and_then(|l| l.clone());
                    let Some(label) = label else {
                        return unsupported("qualifier argument of a call whose declaration is not in this file", callee.line);
                    };
                    parts.push(self.qual_arg_text(&label, chain)?);
                }
                other => parts.push(self.ty(other)?),
            }
        }
        Ok(format!("<{}>", parts.join(", ")))
    }

    fn block_text(&mut self, stmts: &[Stmt]) -> R<String> {
        let mut sub = self.sub();
        sub.indent = self.indent + 1;
        if stmts.is_empty() { sub.line("pass"); } else { for s in stmts { sub.stmt(s)?; } }
        let mut t = sub.out;
        if t.ends_with('\n') { t.pop(); }
        Ok(t)
    }

    /// `match` as an expression: the one-line `match x with pat: e, ...` form when every arm is an
    /// expression, the block form otherwise.
    fn match_expr_text(&mut self, m: &MatchStmt) -> R<String> {
        let all_expr = m.arms.iter().all(|a| matches!(a.body, MatchBody::Expr(_)));
        if all_expr && !m.arms.is_empty() {
            let mut out = format!("match {} with ", self.expr(&m.subject, P_ELSE)?);
            let mut arms = Vec::new();
            let mut ok = true;
            for arm in &m.arms {
                let pats: R<Vec<String>> = arm.patterns.iter().map(|p| self.pattern(p)).collect();
                let mut head = pats?.join(" | ");
                if let Some(g) = &arm.guard { let _ = write!(head, " if {}", self.expr(g, P_ELSE)?); }
                let MatchBody::Expr(e) = &arm.body else { unreachable!() };
                let body = self.expr(e, P_OR)?;
                if body.contains('\n') || head.contains('\n') { ok = false; break; }
                arms.push(format!("{head}: {body}"));
            }
            if ok {
                out.push_str(&arms.join(", "));
                return Ok(out);
            }
        }
        self.match_text(m)
    }

    /// `if`/`elif`/`else` as an expression: inline when every branch is one expression.
    fn if_expr(&mut self, i: &IfStmt) -> R<String> {
        let single = |b: &[Stmt]| -> Option<Expr> {
            match b {
                [Stmt::Expr(e)] if !matches!(e.kind, ExprKind::Assign(..) | ExprKind::QuestionAssign(..)) => Some(e.clone()),
                _ => None,
            }
        };
        let inline_ok = i.branches.iter().all(|(_, b)| single(b).is_some())
            && i.else_body.as_deref().map(|b| single(b).is_some()).unwrap_or(true);
        if inline_ok {
            let mut out = String::new();
            for (n, (c, b)) in i.branches.iter().enumerate() {
                let ct = self.expr(c, P_ELSE)?;
                let bt = self.expr(&single(b).unwrap(), P_OR)?;
                if bt.contains('\n') || ct.contains('\n') { return self.if_expr_block(i); }
                if n > 0 { out.push(' '); }
                let _ = write!(out, "{} {ct}: {bt}", if n == 0 { "if" } else { "elif" });
            }
            if let Some(b) = &i.else_body {
                let bt = self.expr(&single(b).unwrap(), P_OR)?;
                if bt.contains('\n') { return self.if_expr_block(i); }
                let _ = write!(out, " else: {bt}");
            }
            return Ok(out);
        }
        self.if_expr_block(i)
    }

    fn if_expr_block(&mut self, i: &IfStmt) -> R<String> {
        let pad = self.pad();
        let mut out = String::new();
        for (n, (c, b)) in i.branches.iter().enumerate() {
            let ct = self.expr(c, P_ELSE)?;
            if n > 0 { out.push('\n'); out.push_str(&pad); }
            let _ = write!(out, "{} {ct}:\n{}", if n == 0 { "if" } else { "elif" }, self.block_text(b)?);
        }
        if let Some(b) = &i.else_body {
            let _ = write!(out, "\n{pad}else:\n{}", self.block_text(b)?);
        }
        Ok(out)
    }

    fn expr_raw(&mut self, e: &Expr) -> R<String> {
        Ok(match &e.kind {
            ExprKind::Int(n) => n.to_string(),
            ExprKind::UInt64(n) => n.to_string(),
            ExprKind::Float(f) => float_text(*f)?,
            ExprKind::Str(s) => quote_str(s),
            ExprKind::StringInterp(segs) => {
                let mut out = String::from("\"");
                for seg in segs {
                    match seg {
                        StringSegment::Lit(s) => out.push_str(&escape_lit(s)),
                        StringSegment::Expr(x) => {
                            let t = self.expr(x, P_ELSE)?;
                            let _ = write!(out, "{{{t}}}");
                        }
                        StringSegment::FormattedExpr(x, fmt) => {
                            let t = self.expr(x, P_ELSE)?;
                            let _ = write!(out, "{{{t}:{fmt}}}");
                        }
                    }
                }
                out.push('"');
                out
            }
            ExprKind::Bool(b) => b.to_string(),
            ExprKind::Nil => "nil".into(),
            ExprKind::Void => "void".into(),
            ExprKind::Var(n) => n.clone(),
            ExprKind::BinOp(op, a, b) => {
                let (sym, lvl) = binop_info(op);
                let cmp = lvl == P_CMP;
                let mut left_min = if cmp { lvl + 1 } else if lvl == P_RANGE { P_ADD } else { lvl };
                let right_min = if lvl == P_RANGE { P_ADD } else { lvl + 1 };
                if matches!(op, BinOp::Lt | BinOp::Gt | BinOp::Shl | BinOp::Shr) && matches!(a.kind, ExprKind::Cast(..)) {
                    left_min = P_UNARY.max(left_min) + 2; // force parentheses: `x as T < y` reads as generics
                }
                let l = self.left(a, left_min.min(P_ATOM + 1))?;
                let r = self.expr(b, right_min)?;
                format!("{l} {sym} {r}")
            }
            ExprKind::UnaryOp(op, x) => match op {
                UnaryOp::Neg => format!("-{}", self.expr(x, P_POSTFIX)?),
                UnaryOp::BitNot => format!("~{}", self.expr(x, P_POSTFIX)?),
                UnaryOp::Not => format!("not {}", self.expr(x, P_NOT)?),
            },
            ExprKind::Assign(..) | ExprKind::QuestionAssign(..) => self.expr_stmt(e)?,
            // `.thread` / `.tensors`: the parser spells the implicit kernel namespaces `gpu.<name>`.
            ExprKind::Field(x, n) if matches!(&x.kind, ExprKind::Var(g) if g == "gpu") => {
                format!(".{}", if n == "tensor" { "tensors" } else { n.as_str() })
            }
            ExprKind::Field(x, n) => format!("{}.{}", self.receiver(x)?, n),
            ExprKind::Index(x, i) => {
                let r = self.receiver(x)?;
                format!("{r}[{}]", self.index_text(i)?)
            }
            ExprKind::LabeledIndex(x, args) => {
                let r = self.receiver(x)?;
                let mut parts = Vec::new();
                for a in args {
                    let Some(l) = &a.label else { return unsupported("labeled index argument without label", e.line) };
                    parts.push(format!("{l} = {}", self.expr(&a.value, P_ELSE)?));
                }
                format!("{r}[{}]", parts.join(", "))
            }
            ExprKind::Call(callee, args) => {
                let c = self.receiver(callee)?;
                format!("{c}({})", self.args(args)?)
            }
            ExprKind::MethodCall(recv, name, args) => {
                format!("{}.{}({})", self.receiver(recv)?, name, self.args(args)?)
            }
            ExprKind::GenericCall(callee, types, args) => {
                let c = match &callee.kind {
                    ExprKind::Field(r, n) => format!("{}.{}", self.receiver(r)?, n),
                    _ => self.expr(callee, P_POSTFIX)?,
                };
                format!("{c}{}({})", self.type_args(callee, types)?, self.args(args)?)
            }
            ExprKind::Pipe(lhs, name, args) => {
                let l = self.left(lhs, P_PIPE)?;
                format!("{l} |> {name}({})", self.args(args)?)
            }
            ExprKind::KernelLaunch { config, kernel } => {
                let mut parts = Vec::new();
                if let Some(b) = &config.block { parts.push(format!("block = {}", self.expr(b, P_OR)?)); }
                if let Some(g) = &config.grid { parts.push(format!("grid = {}", self.expr(g, P_OR)?)); }
                if let Some(a) = &config.after { parts.push(format!("after = {}", self.expr(a, P_OR)?)); }
                if let Some(p) = &config.priority { parts.push(format!("priority = {}", quote_str(p))); }
                let body = match &kernel.kind {
                    ExprKind::Block(stmts) => self.block_text(stmts)?,
                    _ => return unsupported("kernel launch whose body is not a block", e.line),
                };
                format!("kernel({}):\n{body}", parts.join(", "))
            }
            ExprKind::TryElse(x, d) => {
                if matches!(d.kind, ExprKind::Nil) {
                    format!("try? {}", self.expr(x, P_PIPE)?)
                } else {
                    format!("try {} else {}", self.expr(x, P_PIPE)?, self.expr(d, P_OR)?)
                }
            }
            ExprKind::TryElseBlock(body, els) => {
                let pad = self.pad();
                let single = |s: &[Stmt]| -> Option<Expr> {
                    match s {
                        [Stmt::Expr(x)] if !matches!(x.kind, ExprKind::Assign(..) | ExprKind::QuestionAssign(..)) => Some(x.clone()),
                        _ => None,
                    }
                };
                let mut head = None;
                if let Some(x) = single(body) {
                    let t = self.expr(&x, P_PIPE)?;
                    if !t.contains('\n') { head = Some(format!("try {t} ")); }
                }
                let head = match head {
                    Some(h) => h,
                    None => format!("try:\n{}\n{pad}", self.block_text(body)?),
                };
                let mut tail = None;
                if let Some(x) = single(els) {
                    let t = self.expr(&x, P_OR)?;
                    if !t.contains('\n') { tail = Some(format!("else {t}")); }
                }
                let tail = match tail {
                    Some(t) => t,
                    None => format!("else:\n{}", self.block_text(els)?),
                };
                format!("{head}{tail}")
            }
            ExprKind::Array(es) => {
                let mut parts = Vec::new();
                for x in es { parts.push(self.expr(x, P_ELSE)?); }
                format!("[{}]", parts.join(", "))
            }
            ExprKind::ArrayFill { value, count } => {
                format!("[{} for ..<{}]", self.expr(value, P_ELSE)?, self.expr(count, P_ADD)?)
            }
            ExprKind::ArrayAlloc { count } => format!("[..<{}]", self.expr(count, P_ADD)?),
            ExprKind::ArrayComp { expr, var, count } => {
                format!("[{} for {var} in ..<{}]", self.expr(expr, P_ELSE)?, self.expr(count, P_ADD)?)
            }
            ExprKind::ArrayCompIter { expr, var, iter } => {
                format!("[{} for {var} in {}]", self.expr(expr, P_ELSE)?, self.expr(iter, P_ELSE)?)
            }
            ExprKind::LabeledArrayComp { expr, clauses } => {
                let mut out = format!("[{}", self.expr(expr, P_ELSE)?);
                for (v, c) in clauses {
                    let _ = write!(out, " for {v} in ..<{}", self.expr(c, P_ADD)?);
                }
                out.push(']');
                out
            }
            ExprKind::Tuple(es) => {
                let mut parts = Vec::new();
                for x in es { parts.push(self.expr(x, P_ELSE)?); }
                if parts.len() == 1 { format!("({},)", parts[0]) } else { format!("({})", parts.join(", ")) }
            }
            ExprKind::Dict(kvs) => {
                if kvs.is_empty() {
                    "{=}".into()
                } else {
                    let mut parts = Vec::new();
                    for (k, v) in kvs {
                        parts.push(format!("{} = {}", self.expr(k, P_PIPE)?, self.expr(v, P_ELSE)?));
                    }
                    format!("{{{}}}", parts.join(", "))
                }
            }
            ExprKind::Set(es) => {
                let mut parts = Vec::new();
                for x in es { parts.push(self.expr(x, P_ELSE)?); }
                format!("{{{}}}", parts.join(", "))
            }
            ExprKind::DotIdent(n) => format!(".{n}"),
            ExprKind::Range { start, end, inclusive } => {
                format!("{}{}{}", self.expr(start, P_ADD)?, if *inclusive { "..=" } else { "..<" }, self.expr(end, P_ADD)?)
            }
            ExprKind::SliceRange { .. } => return unsupported("slice range outside an index", e.line),
            ExprKind::Cast(x, t) => {
                let base = match &x.kind {
                    ExprKind::Cast(..) | ExprKind::RelabelCast(..) => format!("({})", self.expr_raw(x)?),
                    _ => self.expr(x, P_POSTFIX)?,
                };
                format!("{base} as {}", self.ty(t)?)
            }
            ExprKind::RelabelCast(x, pairs) => {
                let ps: Vec<String> = pairs.iter().map(|(a, b)| format!("{a} = {b}")).collect();
                format!("{} as [{}]", self.receiver(x)?, ps.join(", "))
            }
            ExprKind::Else(a, b) => {
                let l = self.left(a, P_PIPE)?;
                let r = match &b.kind {
                    ExprKind::Block(stmts) => format!(":\n{}", self.block_text(stmts)?),
                    _ => format!(" {}", self.expr(b, P_OR)?),
                };
                format!("{l} else{r}")
            }
            ExprKind::OptionalField(x, n) => format!("{}?.{}", self.receiver(x)?, n),
            ExprKind::OptionalMethodCall(x, n, args) => format!("{}?.{}({})", self.receiver(x)?, n, self.args(args)?),
            ExprKind::Closure(params, ret, body, throws, task) => {
                if ret.is_some() {
                    return unsupported("closure with an explicit return type", e.line);
                }
                let ps = self.params(params, &[])?;
                let mut head = format!("({ps})");
                if *task { head.push_str(" task"); }
                if *throws { head.push_str(" throws"); }
                match body {
                    ClosureBody::Expr(x) => format!("{head}: {}", self.expr(x, P_OR)?),
                    ClosureBody::Block(stmts) => format!("{head}:\n{}", self.block_text(stmts)?),
                }
            }
            ExprKind::If(i) => self.if_expr(i)?,
            ExprKind::Match(m) => self.match_expr_text(m)?,
            ExprKind::Block(_) => return unsupported("block expression outside `else`/`task`/`kernel` position", e.line),
            ExprKind::Do(stmts) => format!("do:\n{}", self.block_text(stmts)?),
            ExprKind::Loop(l) => format!("loop:\n{}", self.block_text(&l.body)?),
            ExprKind::Task(x) => match &x.kind {
                ExprKind::Block(stmts) => format!("task:\n{}", self.block_text(stmts)?),
                _ => format!("task {}", self.expr(x, P_OR)?),
            },
            ExprKind::TaskWithTimeout(d, body) => {
                let dt = self.expr(d, P_ELSE)?;
                match &body.kind {
                    ExprKind::Block(stmts) => format!("task({dt}):\n{}", self.block_text(stmts)?),
                    _ => format!("task({dt}): {}", self.expr(body, P_OR)?),
                }
            }
            ExprKind::JoinAll(es) => {
                let mut parts = Vec::new();
                for x in es { parts.push(self.expr(x, P_ELSE)?); }
                format!("join({})", parts.join(", "))
            }
            ExprKind::MacroCall { name, args } => {
                let mut parts = Vec::new();
                for x in args { parts.push(self.expr(x, P_ELSE)?); }
                format!("{name}!({})", parts.join(", "))
            }
            ExprKind::TrailingArrayBlock { callee, args, has_parens, body } => {
                let c = self.expr(callee, P_POSTFIX)?;
                let head = if *has_parens { format!("{c}({})", self.args(args)?) } else { c };
                format!("{head}:\n{}", self.block_text(body)?)
            }
        })
    }

    /// The receiver of a `.field` / `.method()`: a numeric literal needs parentheses.
    fn receiver(&mut self, x: &Expr) -> R<String> {
        // A cast target is a type: `x as T.f()` would read `T.f` as an associated type.
        if matches!(x.kind, ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Cast(..) | ExprKind::RelabelCast(..)) {
            return Ok(format!("({})", self.expr_raw(x)?));
        }
        self.expr(x, P_POSTFIX)
    }

    fn index_text(&mut self, i: &Expr) -> R<String> {
        match &i.kind {
            ExprKind::SliceRange { start, end, inclusive } => {
                let s = match start { Some(x) => Some(self.expr(x, P_ADD)?), None => None };
                let en = match end { Some(x) => Some(self.expr(x, P_ADD)?), None => None };
                let op = if *inclusive { "..=" } else { "..<" };
                Ok(match (s, en) {
                    (None, None) => "..".to_string(),
                    (Some(s), None) => if *inclusive { format!("{s}..=") } else { format!("{s}..") },
                    (None, Some(e)) => format!("{op}{e}"),
                    (Some(s), Some(e)) => format!("{s}{op}{e}"),
                })
            }
            _ => self.expr(i, P_ELSE),
        }
    }
}
