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

//! `dispatch NAME in [v1, v2, ...]:` — compile-time specialisation ladder. See docs/book.md,
//! "Compile-time dispatch".
//!
//! Const generic arguments must be literals (`Kernel<8>`), so instantiating a const-generic
//! kernel from a runtime value used to need a hand-written `if x == 8: ... if x == 6: ...`
//! ladder. This statement is pure syntax sugar expanded **in the parser**: the body is cloned
//! once per listed value with `NAME` replaced by that integer literal (in expressions and in
//! generic-argument / type positions), and the clones are chained as `if NAME == v1: ...
//! elif NAME == v2: ... else: <fallback>`. Everything downstream (checker, interpreter,
//! every GPU backend, `req` rules, monomorphiser) only ever sees the ordinary ladder, so each
//! arm still instantiates its own specialised kernel.
//!
//! `dispatch` is a contextual keyword: it is only a statement when followed by `IDENT in`, so
//! an existing variable or function named `dispatch` is unaffected.

use super::*;
use crate::ast_walk::Walker;

fn subst_type(t: &mut Type, name: &str, v: i64) {
    match t {
        Type::Named(n) | Type::TypeParam(n) if n == name => *t = Type::ConstInt(v),
        Type::Optional(i) | Type::Array(i) | Type::Set(i) | Type::Dyn(i) | Type::Impl(i)
        | Type::Mut(i) | Type::Qualified(i, _) | Type::ArrayN(i, _) | Type::ArrayNExpr(i, _) => {
            subst_type(i, name, v)
        }
        Type::Generic(_, args) | Type::Tuple(args) => {
            for a in args { subst_type(a, name, v); }
        }
        Type::Dict(k, val) => { subst_type(k, name, v); subst_type(val, name, v); }
        _ => {}
    }
}

/// Clone `body` with every use of `name` replaced by the literal `v`.
fn specialise(body: &[Stmt], name: &str, v: i64) -> Vec<Stmt> {
    let mut out = body.to_vec();
    let mut on_expr = |e: &mut Expr| match &mut e.kind {
        ExprKind::Var(n) if n == name => e.kind = ExprKind::Int(v),
        ExprKind::GenericCall(_, targs, _) => {
            for t in targs { subst_type(t, name, v); }
        }
        _ => {}
    };
    let mut on_stmt = |s: &mut Stmt| {
        if let Stmt::Let(l) = s {
            if let Some(t) = &mut l.ty { subst_type(t, name, v); }
        }
    };
    Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut out);
    out
}

impl Parser {
    /// True at `dispatch IDENT in` — the only shape that makes `dispatch` a keyword.
    pub(crate) fn is_dispatch_stmt(&self) -> bool {
        matches!(self.peek(), TokenKind::Ident(s) if s == "dispatch")
            && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Ident(_)))
            && matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind), Some(TokenKind::In))
    }

    pub(crate) fn parse_dispatch_stmt(&mut self) -> Result<Stmt, ParseError> {
        let line = self.line();
        let col = self.col();
        self.advance(); // `dispatch`
        let (nline, ncol, nlen) = (self.line(), self.col(), self.tok_len());
        let name = self.expect_ident()?;
        self.expect(&TokenKind::In)?;
        self.expect(&TokenKind::LBracket)?;
        let mut values: Vec<i64> = Vec::new();
        while !self.check(&TokenKind::RBracket) {
            let (vl, vc, vn) = (self.line(), self.col(), self.tok_len());
            let neg = self.eat(&TokenKind::Minus);
            let TokenKind::Int(n) = self.peek().clone() else {
                return Err(ParseError::Generic {
                    line: vl, col: vc, len: vn,
                    msg: "`dispatch` values must be integer literals".into(),
                });
            };
            self.advance();
            let n = if neg { -n } else { n };
            if values.contains(&n) {
                return Err(ParseError::Generic {
                    line: vl, col: vc, len: vn,
                    msg: format!("duplicate `dispatch` value {n}"),
                });
            }
            values.push(n);
            if !self.eat(&TokenKind::Comma) { break; }
        }
        self.expect(&TokenKind::RBracket)?;
        if values.is_empty() {
            return Err(ParseError::Generic {
                line, col, len: nlen,
                msg: "`dispatch` needs at least one value".into(),
            });
        }
        self.expect(&TokenKind::Colon)?;
        self.expect_newline()?;
        let body = self.parse_block()?;
        self.skip_newlines();
        let user_else = if self.check(&TokenKind::Else) {
            self.advance();
            Some(self.parse_else_body_stmts()?)
        } else {
            None
        };

        let var = |l, c| Expr { kind: ExprKind::Var(name.clone()), line: l, col: c, len: nlen };
        let branches = values.iter().map(|&v| {
            let cond = Expr {
                kind: ExprKind::BinOp(
                    BinOp::Eq,
                    Box::new(var(nline, ncol)),
                    Box::new(Expr { kind: ExprKind::Int(v), line: nline, col: ncol, len: nlen }),
                ),
                line: nline, col: ncol, len: nlen,
            };
            (cond, specialise(&body, &name, v))
        }).collect();
        let else_body = user_else.unwrap_or_else(|| {
            let msg = Expr {
                kind: ExprKind::StringInterp(vec![
                    StringSegment::Lit(format!("dispatch: unsupported {name} = ")),
                    StringSegment::Expr(Box::new(var(nline, ncol))),
                    StringSegment::Lit(format!(" (expected one of {values:?})")),
                ]),
                line, col, len: nlen,
            };
            vec![Stmt::Throw(ThrowStmt { value: Some(msg), line, col })]
        });
        Ok(Stmt::If(IfStmt { branches, else_body: Some(else_body), line, col }))
    }
}
