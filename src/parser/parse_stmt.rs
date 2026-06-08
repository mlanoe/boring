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

use super::*;
use crate::ast::*;
use crate::lexer::TokenKind;

impl Parser {
    // ─── Statements ─────────────────────────────────────────────────────────

    pub(crate) fn parse_block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        // Skip comment tokens (and newlines) that may appear before the indent
        while matches!(self.peek(), TokenKind::Comment(_) | TokenKind::Newline | TokenKind::Semicolon) {
            self.advance();
        }
        self.expect(&TokenKind::Indent)?;
        let stmts = self.parse_stmts_until_dedent()?;
        self.eat(&TokenKind::Dedent);
        Ok(stmts)
    }

    pub(crate) fn parse_stmts_until_dedent(&mut self) -> Result<Vec<Stmt>, ParseError> {
        let mut stmts = Vec::new();
        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof)
                || self.check(&TokenKind::RParen)
                || self.check(&TokenKind::RBracket)
                || self.check(&TokenKind::RBrace)
            {
                break;
            }
            stmts.push(self.parse_stmt()?);
        }
        Ok(stmts)
    }

    pub(crate) fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek().clone() {
            TokenKind::Comment(text) => {
                self.advance();
                while self.is_newline() { self.advance(); }
                Ok(Stmt::Comment(text))
            }
            TokenKind::Let | TokenKind::Var | TokenKind::Static => {
                if self.is_let_destructure() {
                    let line = self.line();
                    let _is_static = self.eat(&TokenKind::Static);
                    let mutable = matches!(self.peek(), TokenKind::Var);
                    self.advance(); // consume let/var
                    Ok(Stmt::LetDestructure(self.parse_let_destructure(mutable, line)?))
                } else {
                    Ok(Stmt::Let(self.parse_let_stmt()?))
                }
            }
            TokenKind::Return => Ok(Stmt::Return(self.parse_return_stmt()?)),
            TokenKind::Yield => {
                let line = self.line();
                self.advance(); // consume `yield`
                let expr = self.parse_expr()?;
                self.expect_newline_soft();
                Ok(Stmt::Yield(expr, line))
            }
            TokenKind::Break => {
                let line = self.line();
                self.advance();
                // `break expr` — optional value; newline/dedent/eof = no value
                let value = if self.is_newline()
                    || self.check(&TokenKind::Dedent)
                    || self.check(&TokenKind::Eof)
                {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                self.expect_newline_soft();
                Ok(Stmt::Break(line, value))
            }
            TokenKind::Continue => {
                let line = self.line();
                self.advance();
                self.expect_newline()?;
                Ok(Stmt::Continue(line))
            }
            TokenKind::Throw => Ok(Stmt::Throw(self.parse_throw_stmt()?)),
            TokenKind::If => {
                // `if let …` → multi-clause if-let; `if expr:` → regular if
                if self.tokens.get(self.pos + 1).map(|t| &t.kind) == Some(&TokenKind::Let) {
                    Ok(Stmt::IfLet(self.parse_if_let_stmt()?))
                } else {
                    Ok(Stmt::If(self.parse_if_stmt()?))
                }
            }
            TokenKind::Match => Ok(Stmt::Match(self.parse_match_stmt()?)),
            TokenKind::While => {
                // `while let x = expr:` → while-let (like Rust's `while let Some(x) = expr`)
                if self.tokens.get(self.pos + 1).map(|t| &t.kind) == Some(&TokenKind::Let) {
                    Ok(Stmt::WhileLet(self.parse_while_let_stmt()?))
                } else {
                    Ok(Stmt::While(self.parse_while_stmt()?))
                }
            }
            TokenKind::Do => {
                let line = self.line();
                self.advance(); // consume 'do'
                self.expect(&TokenKind::Colon)?;
                if !self.is_newline() && !self.check(&TokenKind::Eof) {
                    // Inline form: `do: stmt [while cond]`
                    let body = self.parse_inline_stmts()?;
                    if self.check(&TokenKind::While) {
                        self.advance(); // consume 'while'
                        let condition = self.parse_expr()?;
                        self.expect_newline_soft();
                        Ok(Stmt::DoWhile(DoWhileStmt { body, condition, line }))
                    } else {
                        self.expect_newline_soft();
                        Ok(Stmt::Expr(Expr { kind: ExprKind::Do(body), line }))
                    }
                } else {
                    self.expect_newline()?;
                    let body = self.parse_block()?;
                    self.skip_newlines();
                    if self.check(&TokenKind::While) {
                        // do: ... while cond  →  do-while loop
                        self.advance(); // consume 'while'
                        let condition = self.parse_expr()?;
                        self.expect_newline()?;
                        Ok(Stmt::DoWhile(DoWhileStmt { body, condition, line }))
                    } else {
                        // do: ...  →  scoped block expression (no while)
                        Ok(Stmt::Expr(Expr { kind: ExprKind::Do(body), line }))
                    }
                }
            }
            TokenKind::Loop => Ok(Stmt::Loop(self.parse_loop_stmt()?)),
            TokenKind::Wait => {
                let line = self.line();
                self.advance(); // consume `wait`
                let dur = self.parse_expr()?;
                self.expect_newline_soft();
                Ok(Stmt::Wait(dur, line))
            }
            TokenKind::For => Ok(Stmt::For(self.parse_for_stmt()?)),
            TokenKind::Guard => Ok(Stmt::Guard(self.parse_guard_stmt()?)),
            TokenKind::Try => {
                // `try:` block form → TryStmt
                // `try expr else …` inline form → expression statement
                let next = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                if matches!(next, Some(TokenKind::Colon)) {
                    Ok(Stmt::Try(self.parse_try_stmt()?))
                } else {
                    // Inline: parse as expression statement (handles assignment too)
                    let _line = self.line();
                    let expr = self.parse_expr()?;
                    self.expect_newline_soft();
                    Ok(Stmt::Expr(expr))
                }
            }
            TokenKind::Defer => Ok(Stmt::Defer(self.parse_defer_stmt()?)),
            TokenKind::Def => Ok(Stmt::Fn(self.parse_fn_decl(false, true)?)),
            TokenKind::Req => Ok(Stmt::Fn(self.parse_fn_decl(false, false)?)),
            TokenKind::Struct => Ok(Stmt::Struct(self.parse_struct_decl(false)?)),
            TokenKind::Enum   => Ok(Stmt::Enum(self.parse_enum_decl(false)?)),
            TokenKind::Mod    => Ok(Stmt::Mod(self.parse_mod_decl(false)?)),
            TokenKind::Use => {
                // `use Name as Type` — local type alias
                let after_ident = self.tokens.get(self.pos + 2).map(|t| &t.kind);
                if matches!(after_ident, Some(TokenKind::As)) {
                    Ok(Stmt::Alias(self.parse_alias_decl()?))
                } else {
                    Err(ParseError::Generic {
                        line: self.line(),
                        msg: "use inside a function body must be a type alias: `use Name as Type`".into(),
                    })
                }
            }
            TokenKind::Pub => {
                self.advance();
                // `pub static let/var` or `pub let/var` → static variable
                if matches!(self.peek(), TokenKind::Static | TokenKind::Let | TokenKind::Var) {
                    Ok(Stmt::Let(self.parse_let_stmt_pub(true)?))
                } else if self.check(&TokenKind::Req) {
                    Ok(Stmt::Fn(self.parse_fn_decl(true, false)?))
                } else if self.check(&TokenKind::Struct) {
                    Ok(Stmt::Struct(self.parse_struct_decl(true)?))
                } else if self.check(&TokenKind::Enum) {
                    Ok(Stmt::Enum(self.parse_enum_decl(true)?))
                } else {
                    Ok(Stmt::Fn(self.parse_fn_decl(true, true)?))
                }
            }
            TokenKind::Task => {
                // `task def …` / `task req …` / `task RetType …` — function declaration
                match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
                    Some(TokenKind::Def) => return Ok(Stmt::Fn(self.parse_fn_decl(false, true)?)),
                    Some(TokenKind::Req) => return Ok(Stmt::Fn(self.parse_fn_decl(false, false)?)),
                    _ if self.is_task_fn_shorthand() => return Ok(Stmt::Fn(self.parse_fn_decl(false, true)?)),
                    _ => {}
                }
                // Otherwise: `task expr` — spawn expression
                let task_expr = self.parse_task_expr()?;
                if self.is_newline() || self.check(&TokenKind::Eof) {
                    self.skip_newlines();
                }
                Ok(Stmt::Expr(task_expr))
            }
            _ => {
                let line = self.line();
                // Parse lhs (no assignment — parse_expr doesn't produce Assign nodes)
                let lhs = self.parse_expr()?;

                // Simple assignment: `lhs = rhs`
                if self.eat(&TokenKind::Eq) {
                    let rhs = self.parse_else_expr()?;
                    self.expect_newline()?;
                    return Ok(Stmt::Expr(Expr { kind: ExprKind::Assign(Box::new(lhs), Box::new(rhs)), line }));
                }
                // Compound assignment: desugar `lhs op= rhs` → `lhs = lhs op rhs`
                let compound_op = match self.peek() {
                    TokenKind::PlusEq      => Some(BinOp::Add),
                    TokenKind::MinusEq     => Some(BinOp::Sub),
                    TokenKind::StarEq      => Some(BinOp::Mul),
                    TokenKind::SlashEq     => Some(BinOp::Div),
                    TokenKind::PercentEq   => Some(BinOp::Rem),
                    TokenKind::AmpersandEq => Some(BinOp::BitAnd),
                    TokenKind::PipeEq      => Some(BinOp::BitOr),
                    TokenKind::CaretEq     => Some(BinOp::BitXor),
                    _ => None,
                };
                if let Some(op) = compound_op {
                    self.advance();
                    let rhs = self.parse_else_expr()?;
                    let binop = Expr { kind: ExprKind::BinOp(op, Box::new(lhs.clone()), Box::new(rhs)), line };
                    self.expect_newline()?;
                    return Ok(Stmt::Expr(Expr { kind: ExprKind::Assign(Box::new(lhs), Box::new(binop)), line }));
                }
                // Nil-coalescing assignment: `lhs ?= rhs` → `lhs = lhs else rhs`
                if self.eat(&TokenKind::QuestionEq) {
                    let rhs = self.parse_else_expr()?;
                    let else_expr = Expr { kind: ExprKind::Else(Box::new(lhs.clone()), Box::new(rhs)), line };
                    self.expect_newline()?;
                    return Ok(Stmt::Expr(Expr { kind: ExprKind::Assign(Box::new(lhs), Box::new(else_expr)), line }));
                }

                // Command-style call: `print "hello"` or `print "{}", expr, expr2`
                // Only at statement level, only when the base expression is a bare Var.
                // Accepts multiple comma-separated arguments without parentheses.
                let expr = if let ExprKind::Var(_) = &lhs.kind {
                    if self.peek_starts_expr() {
                        let mut args = Vec::new();
                        loop {
                            let arg = self.parse_or()?;
                            args.push(Arg { label: None, value: arg , spread: false});
                            if !self.eat(&TokenKind::Comma) { break; }
                        }
                        Expr { kind: ExprKind::Call(Box::new(lhs), args), line }
                    } else {
                        lhs
                    }
                } else {
                    lhs
                };
                self.expect_newline()?;
                Ok(Stmt::Expr(expr))
            }
        }
    }

    pub(crate) fn parse_let_stmt(&mut self) -> Result<LetStmt, ParseError> {
        self.parse_let_stmt_pub(false)
    }

    pub(crate) fn parse_let_stmt_pub(&mut self, is_pub: bool) -> Result<LetStmt, ParseError> {
        let line = self.line();
        // optional `static` before `let` / `var`
        let is_static = self.eat(&TokenKind::Static);
        let mutable = match self.peek() {
            TokenKind::Var => { self.advance(); true }
            _ => { self.advance(); false } // let
        };
        // `let name = value`         — no type annotation, borrow by default
        // `let name' = value`        — no type annotation, move (tick without qualifier)
        // `let type name = value`    — explicit type annotation (boring convention)
        let (name, ty, _is_move) = if self.is_type_start_before_ident() {
            let base = self.parse_type()?;
            let ty = self.parse_type_qualifier(base);
            // `var T&` → absorb mutability into the borrow type
            let ty = if mutable { Self::apply_var_to_borrow(ty) } else { ty };
            let name = self.expect_ident()?;
            // `var T name'qualifier = value` — qualifier after the name applies to the type
            let ty = if self.check(&TokenKind::Tick) {
                let next_kind = self.tokens.get(self.pos + 1).map(|t| t.kind.clone());
                if matches!(next_kind, Some(TokenKind::Ident(_)) | Some(TokenKind::Task) | Some(TokenKind::Guard)) {
                    self.parse_type_qualifier(ty)
                } else {
                    ty
                }
            } else {
                ty
            };
            (name, Some(ty), false)
        } else {
            let name = self.expect_ident()?;
            if self.check(&TokenKind::Tick) {
                let next_kind = self.tokens.get(self.pos + 1).map(|t| t.kind.clone());
                match &next_kind {
                    // `name' = value` → move marker (tick immediately before `=`)
                    Some(TokenKind::Eq) => {
                        self.advance(); // consume tick
                        self.expect(&TokenKind::Eq)?;
                        let value = self.parse_expr()?;
                        self.expect_newline_soft();
                        return Ok(LetStmt { mutable, is_pub, is_static, name, ty: None, value: Some(value), is_move: true, line });
                    }
                    // `name'qualifier = Ctor(...)` → qualifier on variable, type inferred from RHS
                    Some(TokenKind::Ident(_)) | Some(TokenKind::Task) | Some(TokenKind::Guard) => {
                        // Do NOT advance here — parse_type_qualifier consumes the tick + qualifier itself.
                        let placeholder = Type::Named("_".to_string());
                        let qualified = self.parse_type_qualifier(placeholder);
                        self.expect(&TokenKind::Eq)?;
                        let value = self.parse_expr()?;
                        self.expect_newline_soft();
                        // Infer the real base type from the constructor call in the RHS.
                        let ty = if let ExprKind::Call(callee, _) = &value.kind {
                            if let ExprKind::Var(type_name) = &callee.kind {
                                if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                                    // Replace the `_` placeholder with the actual constructor type.
                                    let base = Type::Named(type_name.clone());
                                    Some(Self::replace_type_base(qualified, base))
                                } else { Some(qualified) }
                            } else { Some(qualified) }
                        } else { Some(qualified) };
                        return Ok(LetStmt { mutable, is_pub, is_static, name, ty, value: Some(value), is_move: false, line });
                    }
                    _ => {}
                }
            }
            (name, None, false)
        };
        // `let v` / `var v` — deferred initialisation (no `= expr`).
        if !self.check(&TokenKind::Eq) {
            self.expect_newline_soft();
            return Ok(LetStmt { mutable, is_pub, is_static, name, ty, value: None, is_move: false, line });
        }
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;
        self.expect_newline_soft();
        Ok(LetStmt { mutable, is_pub, is_static, name, ty, value: Some(value), is_move: false, line })
    }

    /// Parse the binding list of a destructuring `let`.
    /// Handles both the parenthesised form `(a, b)` and the bare form `a, b`.
    /// Called after the `let`/`var` keyword has already been consumed.
    pub(crate) fn parse_let_destructure(&mut self, mutable: bool, line: usize) -> Result<LetDestructureStmt, ParseError> {
        let parens = self.eat(&TokenKind::LParen);
        if parens { self.skip_newlines_and_indent(); }
        let mut bindings = Vec::new();
        loop {
            if parens && self.check(&TokenKind::RParen) { break; }
            if !parens && (self.check(&TokenKind::Eq) || self.check(&TokenKind::Eof)) { break; }
            // Each slot: optional type + name, or just `_`
            let (name, ty) = if self.check(&TokenKind::Ident(String::from("_"))) {
                self.advance();
                ("_".to_string(), None)
            } else if self.is_type_start_before_ident() {
                let ty = self.parse_type()?;
                let name = self.expect_ident()?;
                (name, Some(ty))
            } else {
                (self.expect_ident()?, None)
            };
            bindings.push(DestructureBinding { name, ty });
            if !self.eat(&TokenKind::Comma) { break; }
            self.skip_newlines_and_indent();
        }
        if parens { self.skip_newlines_and_indent(); self.expect(&TokenKind::RParen)?; }
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;
        self.expect_newline_soft();
        Ok(LetDestructureStmt { mutable, bindings, value, line })
    }

    pub(crate) fn parse_return_stmt(&mut self) -> Result<ReturnStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Return)?;
        let value = if self.is_newline() || self.check(&TokenKind::Eof) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect_newline()?;
        Ok(ReturnStmt { value, line })
    }

    pub(crate) fn parse_throw_stmt(&mut self) -> Result<ThrowStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Throw)?;
        let value = if self.is_newline() || self.check(&TokenKind::Eof) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect_newline()?;
        Ok(ThrowStmt { value, line })
    }

    pub(crate) fn parse_if_stmt(&mut self) -> Result<IfStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::If)?;
        let saved = self.allow_noparen_closure;
        self.allow_noparen_closure = false; self.allow_trailing_closure = false;
        let cond = self.parse_expr()?;
        self.allow_noparen_closure = saved; self.allow_trailing_closure = true;
        self.expect(&TokenKind::Colon)?;

        // Check if the body is inline (no newline) or a block
        if !self.is_newline() && !self.check(&TokenKind::Eof) {
            // Inline then-statement: `if c: stmt [; stmt]* [elif c: stmt]* [else [:] stmt]`
            let then_stmts = self.parse_inline_stmts_if_body()?;
            let mut branches = vec![(cond, then_stmts)];
            let mut else_body = None;
            // Same-line elif/else
            loop {
                if self.check(&TokenKind::Elif) {
                    self.advance();
                    let saved = self.allow_noparen_closure;
                    self.allow_noparen_closure = false; self.allow_trailing_closure = false;
                    let elif_cond = self.parse_expr()?;
                    self.allow_noparen_closure = saved; self.allow_trailing_closure = true;
                    self.expect(&TokenKind::Colon)?;
                    let elif_stmts = self.parse_inline_stmts_if_body()?;
                    branches.push((elif_cond, elif_stmts));
                } else if self.check(&TokenKind::Else) {
                    self.advance();
                    else_body = Some(self.parse_else_body_stmts()?);
                    break;
                } else {
                    break;
                }
            }
            // If no else was found on the same line, check next line(s) for elif/else
            if else_body.is_none() {
                self.expect_newline()?;
                loop {
                    self.skip_newlines();
                    if self.check(&TokenKind::Elif) {
                        self.advance();
                        let saved = self.allow_noparen_closure;
                        self.allow_noparen_closure = false; self.allow_trailing_closure = false;
                        let elif_cond = self.parse_expr()?;
                        self.allow_noparen_closure = saved; self.allow_trailing_closure = true;
                        self.expect(&TokenKind::Colon)?;
                        let elif_body = if !self.is_newline() && !self.check(&TokenKind::Eof) {
                            self.parse_inline_stmts_if_body()?
                        } else {
                            self.expect_newline()?;
                            self.parse_block()?
                        };
                        branches.push((elif_cond, elif_body));
                    } else if self.check(&TokenKind::Else) {
                        self.advance();
                        else_body = Some(self.parse_else_body_stmts()?);
                        break;
                    } else {
                        break;
                    }
                }
            } else if self.is_newline() || self.check(&TokenKind::Eof) {
                self.expect_newline_soft();
            }
            return Ok(IfStmt { branches, else_body, line });
        }

        self.expect_newline()?;
        let body = self.parse_block()?;
        let mut branches = vec![(cond, body)];
        let mut else_body = None;

        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Elif) {
                self.advance();
                let saved_elif = self.allow_noparen_closure;
                self.allow_noparen_closure = false; self.allow_trailing_closure = false;
                let elif_cond = self.parse_expr()?;
                self.allow_noparen_closure = saved_elif; self.allow_trailing_closure = true;
                self.expect(&TokenKind::Colon)?;
                let elif_body = if !self.is_newline() && !self.check(&TokenKind::Eof) {
                    self.parse_inline_stmts_if_body()?
                } else {
                    self.expect_newline()?;
                    self.parse_block()?
                };
                branches.push((elif_cond, elif_body));
            } else if self.check(&TokenKind::Else) {
                self.advance();
                else_body = Some(self.parse_else_body_stmts()?);
                break;
            } else {
                break;
            }
        }

        Ok(IfStmt { branches, else_body, line })
    }

    /// Parse one condition clause: `let name = expr` or a boolean expression.
    /// Called inside `if let` and `guard let` comma-separated clause lists.
    pub(crate) fn parse_cond_clause(&mut self) -> Result<CondClause, ParseError> {
        if self.eat(&TokenKind::Let) {
            // `let Some(x) = expr` / `let Ok(v) = expr` / `let Variant(a, b) = expr`
            // Detect: next token is Ident and the one after is `(`
            let is_pattern = matches!(self.peek().clone(), TokenKind::Ident(_))
                && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::LParen));
            if is_pattern {
                let pat = self.parse_pattern()?;
                self.expect(&TokenKind::Eq)?;
                let expr = self.parse_or()?;
                Ok(CondClause::LetPat(pat, expr))
            } else {
                let name_line = self.line();
                let name = self.expect_ident()?;
                // Swift-style shorthand: `if let v:` ≡ `if let v = v:`
                if !self.check(&TokenKind::Eq) {
                    let expr = Expr { kind: ExprKind::Var(name.clone()), line: name_line };
                    return Ok(CondClause::Let(name, expr));
                }
                self.expect(&TokenKind::Eq)?;
                let expr = self.parse_or()?;
                Ok(CondClause::Let(name, expr))
            }
        } else {
            Ok(CondClause::Expr(self.parse_or()?))
        }
    }

    /// Parse a comma-separated list of `CondClause`s until `stop` token.
    /// Does not consume the stop token.
    pub(crate) fn parse_cond_clauses(&mut self, stop: &TokenKind) -> Result<Vec<CondClause>, ParseError> {
        let saved_np = self.allow_noparen_closure;
        let saved_tc = self.allow_trailing_closure;
        self.allow_noparen_closure = false;
        self.allow_trailing_closure = false;
        let mut clauses = vec![self.parse_cond_clause()?];
        while self.eat(&TokenKind::Comma) {
            clauses.push(self.parse_cond_clause()?);
        }
        self.allow_noparen_closure = saved_np;
        self.allow_trailing_closure = saved_tc;
        if !self.check(stop) {
            return Err(ParseError::Generic {
                line: self.line(),
                msg: format!("expected {:?} after condition clauses, got {:?}", stop, self.peek()),
            });
        }
        Ok(clauses)
    }

    pub(crate) fn parse_if_let_stmt(&mut self) -> Result<IfLetStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::If)?;
        // Parse one or more comma-separated clauses until `:`
        let clauses = self.parse_cond_clauses(&TokenKind::Colon)?;
        self.expect(&TokenKind::Colon)?;

        if !self.is_newline() && !self.check(&TokenKind::Eof) {
            // Inline: `if let n = expr: body [else body]`
            let then_body = self.parse_inline_stmts()?;
            let mut else_body = None;
            if self.check(&TokenKind::Else) {
                self.advance();
                self.eat(&TokenKind::Colon);
                else_body = Some(self.parse_inline_stmts()?);
            }
            self.expect_newline_soft();
            return Ok(IfLetStmt { clauses, then_body, else_body, line });
        }

        self.expect_newline()?;
        let then_body = self.parse_block()?;
        let mut else_body = None;
        self.skip_newlines();
        if self.check(&TokenKind::Else) {
            self.advance();
            self.eat(&TokenKind::Colon);
            else_body = Some(self.parse_else_body_stmts()?);
        }
        Ok(IfLetStmt { clauses, then_body, else_body, line })
    }

    pub(crate) fn parse_match_stmt(&mut self) -> Result<MatchStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Match)?;
        let saved = self.allow_noparen_closure;
        self.allow_noparen_closure = false; self.allow_trailing_closure = false;
        let subject = self.parse_expr()?;
        self.allow_noparen_closure = saved; self.allow_trailing_closure = true;
        // Inline form: `match expr with Pat1: val1, Pat2: val2, _: val3`
        if self.check(&TokenKind::With) {
            self.advance();
            let arms = self.parse_inline_match_arms()?;
            return Ok(MatchStmt { subject, arms, line });
        }
        self.expect(&TokenKind::Colon)?;
        self.expect_newline()?;
        self.expect(&TokenKind::Indent)?;
        let mut arms = Vec::new();
        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof)
                || self.check(&TokenKind::RParen)
                || self.check(&TokenKind::RBracket)
                || self.check(&TokenKind::RBrace)
            { break; }
            arms.push(self.parse_match_arm()?);
        }
        self.eat(&TokenKind::Dedent);
        Ok(MatchStmt { subject, arms, line })
    }

    /// Parse inline match arms: `Pat: val, Pat: val, ...`
    /// Used after `match expr with`. Arms separated by `,`.
    /// Arm bodies are parsed with parse_or() — no trailing comma ambiguity since bodies
    /// are simple expressions and `,` is not a binary operator in Boring.
    pub(crate) fn parse_inline_match_arms(&mut self) -> Result<Vec<MatchArm>, ParseError> {
        let mut arms = Vec::new();
        loop {
            let line = self.line();
            let pat = self.parse_pattern()?;
            self.expect(&TokenKind::Colon)?;
            let body_expr = self.parse_or()?;
            let body = MatchBody::Block(vec![Stmt::Expr(body_expr)]);
            arms.push(MatchArm { patterns: vec![pat], guard: None, body, line });
            if !self.eat(&TokenKind::Comma) { break; }
            self.skip_newlines();
        }
        Ok(arms)
    }

    pub(crate) fn parse_match_arm(&mut self) -> Result<MatchArm, ParseError> {
        let line = self.line();
        // Each alternative is a (possibly bare-tuple) pattern; alternatives separated by `|`.
        let mut patterns = vec![self.parse_pattern_or_tuple()?];
        while self.eat(&TokenKind::Pipe) {
            patterns.push(self.parse_pattern_or_tuple()?);
        }
        // Optional guard: `pattern if cond:`
        // Disable no-paren closure so `ident: body` is not mistaken for a closure.
        let guard = if self.eat(&TokenKind::If) {
            let saved = self.allow_noparen_closure;
            self.allow_noparen_closure = false;
            let g = self.parse_or()?;
            self.allow_noparen_closure = saved;
            Some(g)
        } else {
            None
        };
        self.expect(&TokenKind::Colon)?;

        // Body: either inline stmt (on same line) or a block
        let body = if self.is_newline() {
            self.advance();
            let stmts = self.parse_block()?;
            MatchBody::Block(stmts)
        } else {
            // Disable no-paren closures in arm bodies: `_: expr` after a block-consuming
            // inline stmt (like `if`) would otherwise mistake the next arm's `_:` for a
            // trailing closure appended to the expression.
            let saved_npc = self.allow_noparen_closure;
            self.allow_noparen_closure = false;
            let stmts = self.parse_inline_stmts()?;
            self.allow_noparen_closure = saved_npc;
            self.expect_newline_soft();
            MatchBody::Block(stmts)
        };

        Ok(MatchArm { patterns, guard, body, line })
    }

    /// Parse a pattern that may be a bare tuple: `a, b, c` (no parens required).
    /// Used at the top level of a match arm where `,` is not an OR separator.
    pub(crate) fn parse_pattern_or_tuple(&mut self) -> Result<Pattern, ParseError> {
        let first = self.parse_pattern()?;
        if self.check(&TokenKind::Comma) {
            let mut elems = vec![first];
            while self.eat(&TokenKind::Comma) {
                self.skip_newlines_and_indent();
                elems.push(self.parse_pattern()?);
            }
            Ok(Pattern::Tuple(elems))
        } else {
            Ok(first)
        }
    }

    pub(crate) fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        // Tuple pattern: `(pat, pat, ...)`
        if self.check(&TokenKind::LParen) {
            self.advance();
            self.skip_newlines_and_indent();
            let mut elems = Vec::new();
            while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                elems.push(self.parse_pattern()?);
                if !self.eat(&TokenKind::Comma) { break; }
                self.skip_newlines_and_indent();
            }
            self.skip_newlines_and_indent();
            self.expect(&TokenKind::RParen)?;
            return Ok(Pattern::Tuple(elems));
        }

        match self.peek().clone() {
            TokenKind::Ident(s) if s == "_" => {
                self.advance();
                Ok(Pattern::Wildcard)
            }
            TokenKind::Ident(s) => {
                self.advance();
                if s == "nil" {
                    Ok(Pattern::Lit(LitPattern::Nil))
                } else if s == "some" || s == "Some" {
                    // `Some(pat)` — non-nil Optional with inner binding
                    // `Some`      — bare: matches any non-nil value (wildcard inner)
                    if self.check(&TokenKind::LParen) {
                        self.advance();
                        let inner = self.parse_pattern()?;
                        self.expect(&TokenKind::RParen)?;
                        Ok(Pattern::Some(Box::new(inner)))
                    } else {
                        Ok(Pattern::Some(Box::new(Pattern::Wildcard)))
                    }
                } else if s == "none" || s == "None" {
                    Ok(Pattern::None)
                } else if self.check(&TokenKind::LParen) {
                    // Generic variant with payload fields
                    self.advance();
                    let mut sub_pats = Vec::new();
                    self.skip_newlines_and_indent();
                    while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                        sub_pats.push(self.parse_pattern()?);
                        if !self.eat(&TokenKind::Comma) { break; }
                        self.skip_newlines_and_indent();
                    }
                    self.skip_newlines_and_indent();
                    self.expect(&TokenKind::RParen)?;
                    Ok(Pattern::Variant(s, sub_pats))
                } else if s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && self.check(&TokenKind::Dot)
                    && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind),
                                Some(TokenKind::Ident(v)) if v.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
                {
                    // Qualified variant pattern: `Error.Expired` or `Status.Ok(v)`
                    // Stored internally as `Enum::Variant` so emit_pattern emits it verbatim.
                    self.advance(); // consume `.`
                    let variant = self.expect_ident()?;
                    let sub_pats = if self.check(&TokenKind::LParen) {
                        self.advance();
                        let mut sp = Vec::new();
                        self.skip_newlines_and_indent();
                        while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                            sp.push(self.parse_pattern()?);
                            if !self.eat(&TokenKind::Comma) { break; }
                            self.skip_newlines_and_indent();
                        }
                        self.skip_newlines_and_indent();
                        self.expect(&TokenKind::RParen)?;
                        sp
                    } else { vec![] };
                    Ok(Pattern::Variant(format!("{}::{}", s, variant), sub_pats))
                } else {
                    // Could be a variant name or a binding
                    // If starts uppercase => variant, else => bind
                    if s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        Ok(Pattern::Variant(s, vec![]))
                    } else {
                        Ok(Pattern::Bind(s))
                    }
                }
            }
            TokenKind::Int(n) => {
                let n = n;
                self.advance();
                Ok(Pattern::Lit(LitPattern::Int(n)))
            }
            TokenKind::Float(f) => {
                let f = f;
                self.advance();
                Ok(Pattern::Lit(LitPattern::Float(f)))
            }
            TokenKind::Str(s) => {
                let s = s.clone();
                self.advance();
                Ok(Pattern::Lit(LitPattern::Str(s)))
            }
            TokenKind::Bool(b) => {
                let b = b;
                self.advance();
                Ok(Pattern::Lit(LitPattern::Bool(b)))
            }
            TokenKind::Nil => {
                self.advance();
                Ok(Pattern::Lit(LitPattern::Nil))
            }
            _ => Err(ParseError::Generic {
                line: self.line(),
                msg: format!("expected pattern, got {:?}", self.peek()),
            }),
        }
    }

    /// `while let name = expr:` — while-let loop, binds the result of `expr` each iteration.
    pub(crate) fn parse_while_let_stmt(&mut self) -> Result<WhileLetStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::While)?;
        self.expect(&TokenKind::Let)?;
        // `while let Some(x) = expr:` — detect pattern form
        let is_pattern = matches!(self.peek().clone(), TokenKind::Ident(_))
            && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::LParen));
        let (name, pattern) = if is_pattern {
            let pat = self.parse_pattern()?;
            ("".to_string(), Some(pat))
        } else {
            (self.expect_ident()?, None)
        };
        // Swift-style shorthand: `while let v:` ≡ `while let v = v:`
        let saved = self.allow_noparen_closure;
        let value = if !is_pattern && !self.check(&TokenKind::Eq) {
            Expr { kind: ExprKind::Var(name.clone()), line }
        } else {
            self.expect(&TokenKind::Eq)?;
            self.allow_noparen_closure = false; self.allow_trailing_closure = false;
            let v = self.parse_expr()?;
            self.allow_noparen_closure = saved; self.allow_trailing_closure = true;
            v
        };
        self.expect(&TokenKind::Colon)?;
        let body = if !self.is_newline() && !self.check(&TokenKind::Eof) {
            let stmts = self.parse_inline_stmts()?;
            self.expect_newline_soft();
            stmts
        } else {
            self.expect_newline()?;
            self.parse_block()?
        };
        Ok(WhileLetStmt { name, pattern, value, body, line })
    }

    pub(crate) fn parse_while_stmt(&mut self) -> Result<WhileStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::While)?;
        let saved = self.allow_noparen_closure;
        self.allow_noparen_closure = false; self.allow_trailing_closure = false;
        let condition = self.parse_expr()?;
        self.allow_noparen_closure = saved; self.allow_trailing_closure = true;
        self.expect(&TokenKind::Colon)?;
        let body = if !self.is_newline() && !self.check(&TokenKind::Eof) {
            let stmts = self.parse_inline_stmts()?;
            self.expect_newline_soft();
            stmts
        } else {
            self.expect_newline()?;
            self.parse_block()?
        };
        Ok(WhileStmt { condition, body, line })
    }

    pub(crate) fn parse_loop_stmt(&mut self) -> Result<LoopStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Loop)?;
        self.expect(&TokenKind::Colon)?;
        let body = if !self.is_newline() && !self.check(&TokenKind::Eof) {
            let stmts = self.parse_inline_stmts()?;
            self.expect_newline_soft();
            stmts
        } else {
            self.expect_newline()?;
            self.parse_block()?
        };
        Ok(LoopStmt { body, line })
    }

    /// Returns true if the current position starts a `for var [, var] in` binding form.
    /// Scans forward past comma-separated idents; returns true only if `in` follows.
    pub(crate) fn is_for_with_vars(&self) -> bool {
        if !matches!(self.peek(), TokenKind::Ident(_)) { return false; }
        let mut i = self.pos;
        loop {
            match self.tokens.get(i).map(|t| &t.kind) {
                Some(TokenKind::Ident(_)) => { i += 1; }
                Some(TokenKind::Comma)    => { i += 1; }
                Some(TokenKind::In)       => return true,
                _                         => return false,
            }
        }
    }

    pub(crate) fn parse_for_stmt(&mut self) -> Result<ForStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::For)?;

        let saved = self.allow_noparen_closure;
        self.allow_noparen_closure = false; self.allow_trailing_closure = false;

        let (vars, iterable) = if self.is_for_with_vars() {
            // `for i in expr:` or `for i, j in expr:` — explicit variable(s)
            let mut vars = vec![self.expect_ident()?];
            while self.eat(&TokenKind::Comma) {
                vars.push(self.expect_ident()?);
            }
            self.expect(&TokenKind::In)?;
            let iterable = self.parse_expr()?;
            (vars, iterable)
        } else {
            // `for expr:` — no variable, implicit `_` (e.g. `for 1..<8:`)
            let iterable = self.parse_expr()?;
            (vec!["_".to_string()], iterable)
        };

        self.allow_noparen_closure = saved; self.allow_trailing_closure = true;
        self.expect(&TokenKind::Colon)?;
        let body = if !self.is_newline() && !self.check(&TokenKind::Eof) {
            let stmts = self.parse_inline_stmts()?;
            self.expect_newline_soft();
            stmts
        } else {
            self.expect_newline()?;
            self.parse_block()?
        };
        Ok(ForStmt { vars, iterable, body, line })
    }

    pub(crate) fn parse_guard_stmt(&mut self) -> Result<GuardStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Guard)?;

        // `guard let …` or `guard let …, …` → Clauses
        // `guard boolexpr else` → plain Expr (single clause, no let)
        let cond = if self.check(&TokenKind::Let) {
            let clauses = self.parse_cond_clauses(&TokenKind::Else)?;
            GuardCond::Clauses(clauses)
        } else {
            let saved = self.allow_noparen_closure;
            self.allow_noparen_closure = false; self.allow_trailing_closure = false;
            let expr = self.parse_or()?;
            self.allow_noparen_closure = saved; self.allow_trailing_closure = true;
            GuardCond::Expr(expr)
        };

        self.expect(&TokenKind::Else)?;
        let else_body = self.parse_else_body_stmts()?;
        Ok(GuardStmt { cond, else_body, line })
    }

    /// After `else`: optionally consume `:`, then parse a block (if next is Newline)
    /// or a single statement inline.
    pub(crate) fn parse_else_body_stmts(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.eat(&TokenKind::Colon);
        if self.is_newline() {
            self.expect_newline()?;
            self.parse_block()
        } else {
            let stmts = self.parse_inline_stmts()?;
            self.expect_newline_soft();
            Ok(stmts)
        }
    }

    /// After `else`: optionally consume `:`, then parse a block-as-expr (if next is Newline)
    /// or a single expression inline.
    pub(crate) fn parse_else_body_expr(&mut self) -> Result<Expr, ParseError> {
        let line = self.line();
        self.eat(&TokenKind::Colon);
        if self.is_newline() {
            self.expect_newline()?;
            let stmts = self.parse_block()?;
            Ok(Expr { kind: ExprKind::Block(stmts), line })
        } else {
            self.parse_or()
        }
    }

    pub(crate) fn parse_try_stmt(&mut self) -> Result<TryStmt, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Try)?;
        self.expect(&TokenKind::Colon)?;
        self.expect_newline()?;
        let body = self.parse_block()?;
        let mut catch_clauses = Vec::new();

        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Catch) {
                let catch_line = self.line();
                self.advance();
                let mut types = Vec::new();
                let mut variant: Option<String> = None;
                // Optional type list — supports `catch Error.Variant:` for variant dispatch.
                if matches!(self.peek(), TokenKind::Ident(_)) {
                    let first_type = self.expect_ident()?;
                    // Check for `.Variant` suffix: `catch Error.Expired:`
                    if self.eat(&TokenKind::Dot) {
                        variant = Some(self.expect_ident()?);
                        types.push(first_type);
                        // No comma-separated multi-type when using variant syntax.
                    } else {
                        types.push(first_type);
                        while self.eat(&TokenKind::Comma) {
                            types.push(self.expect_ident()?);
                        }
                    }
                }
                self.expect(&TokenKind::Colon)?;
                self.expect_newline()?;
                let catch_body = self.parse_block()?;
                catch_clauses.push(CatchClause { types, variant, body: catch_body, line: catch_line });
            } else {
                break;
            }
        }

        Ok(TryStmt { body, catch_clauses, line })
    }


    pub(crate) fn parse_defer_stmt(&mut self) -> Result<Vec<Stmt>, ParseError> {
        self.expect(&TokenKind::Defer)?;
        // Three forms:
        //   defer:          — multiline block (colon + newline + indent)
        //   defer: expr     — inline single expression
        //   defer expr      — no colon, single expression (command-style)
        if self.check(&TokenKind::Colon) {
            self.advance(); // consume `:`
            if self.is_newline() {
                // Multiline block
                self.advance();
                Ok(self.parse_block()?)
            } else {
                // Inline: `defer: stmt [; stmt]*`
                let stmts = self.parse_inline_stmts()?;
                self.expect_newline_soft();
                Ok(stmts)
            }
        } else {
            // No colon: `defer stmt` — parse as inline statement (may be assignment)
            let stmt = self.parse_inline_stmt()?;
            self.expect_newline_soft();
            Ok(vec![stmt])
        }
    }
}
