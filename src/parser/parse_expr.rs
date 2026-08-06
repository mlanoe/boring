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
use crate::lexer::{lex, TokenKind, RawInterpPart};

impl Parser {
    // ─── Expressions ────────────────────────────────────────────────────────

    pub(crate) fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        // Assignment is a statement only — expressions never produce Assign nodes
        //
        // Every recursive sub-expression (parenthesized groups, array/tuple elements,
        // call arguments, ...) re-enters here, so guarding this single chokepoint bounds
        // the whole expression-parsing recursion (deeply nested `((((...))))`,
        // `[[[[...]]]]`, etc.) against a stack-overflow crash, the same way the `not`-chain
        // guard in `parse_not` bounds unary-`not` recursion.
        let line = self.line();
        let col = self.col();
        self.depth += 1;
        if self.depth > crate::parser::MAX_EXPR_DEPTH {
            self.depth -= 1;
            return Err(ParseError::Generic {
                line, col,
                msg: format!("expression nested too deeply (limit: {})", crate::parser::MAX_EXPR_DEPTH), len: self.tok_len(),
            });
        }
        let result = self.parse_else_expr();
        self.depth -= 1;
        result
    }

    /// Parse a single statement for inline body positions (match arms, if-let inline,
    /// setter body, defer inline) where no trailing newline is consumed.
    /// Handles assignment (`lhs = rhs`) and compound assignment (`lhs op= rhs`).
    pub(crate) fn parse_inline_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.parse_inline_stmt_impl(false)
    }

    /// Like `parse_inline_stmt` but uses `parse_or` instead of `parse_else_expr` for
    /// expressions, so that a bare `else` token is NOT consumed as nil-coalescing.
    /// Used for inline if/elif bodies where `else` must remain available for the else-branch.
    pub(crate) fn parse_inline_if_body(&mut self) -> Result<Stmt, ParseError> {
        self.parse_inline_stmt_impl(true)
    }

    /// Parse one or more statements separated by `;` on the same line.
    /// Stops at a real Newline or EOF.
    pub(crate) fn parse_inline_stmts(&mut self) -> Result<Vec<Stmt>, ParseError> {
        let mut stmts = vec![self.parse_inline_stmt()?];
        while self.check(&TokenKind::Semicolon) {
            self.advance();
            if self.is_newline() || self.check(&TokenKind::Eof) { break; }
            stmts.push(self.parse_inline_stmt()?);
        }
        Ok(stmts)
    }

    /// Like `parse_inline_stmts` but stops before `else`/`elif` (for if-body parsing).
    pub(crate) fn parse_inline_stmts_if_body(&mut self) -> Result<Vec<Stmt>, ParseError> {
        let mut stmts = vec![self.parse_inline_if_body()?];
        while self.check(&TokenKind::Semicolon) {
            self.advance();
            if self.is_newline() || self.check(&TokenKind::Eof)
                || self.check(&TokenKind::Else) || self.check(&TokenKind::Elif) { break; }
            stmts.push(self.parse_inline_if_body()?);
        }
        Ok(stmts)
    }

    pub(crate) fn parse_inline_stmt_impl(&mut self, stop_at_else: bool) -> Result<Stmt, ParseError> {
        let line = self.line();
        let col = self.col();
        // Allow let/mut/var/static/lazy bindings in inline positions (e.g. match arm bodies)
        if matches!(self.peek(), TokenKind::Let | TokenKind::Mut | TokenKind::Var | TokenKind::Static | TokenKind::Lazy) {
            if self.is_let_destructure() {
                let _is_static = self.eat(&TokenKind::Static);
                let (binding, var_mut) = self.consume_binding_keyword();
                let saved = self.in_inline_context;
                self.in_inline_context = true;
                let result = self.parse_let_destructure(binding, var_mut, line, col);
                self.in_inline_context = saved;
                return Ok(Stmt::LetDestructure(result?));
            } else {
                let saved = self.in_inline_context;
                self.in_inline_context = true;
                let result = self.parse_let_stmt();
                self.in_inline_context = saved;
                return Ok(Stmt::Let(result?));
            }
        }
        // Allow control-flow keywords in inline positions
        if self.check(&TokenKind::Return) {
            self.advance();
            let value = if self.is_newline() || self.check(&TokenKind::Eof) {
                None
            } else {
                Some(self.parse_or()?)
            };
            return Ok(Stmt::Return(ReturnStmt { value, line, col }));
        }
        if self.check(&TokenKind::Throw) {
            let line = self.line();
            let col = self.col();
            self.advance();
            let value = if self.is_newline() || self.check(&TokenKind::Eof) {
                None
            } else {
                Some(self.parse_or()?)
            };
            return Ok(Stmt::Throw(ThrowStmt { value, line, col }));
        }
        if self.check(&TokenKind::Break) {
            self.advance();
            let value = if self.is_newline() || self.check(&TokenKind::Eof) {
                None
            } else {
                Some(self.parse_or()?)
            };
            return Ok(Stmt::Break(line, value));
        }
        if self.check(&TokenKind::Continue) {
            self.advance();
            return Ok(Stmt::Continue(line));
        }
        let lhs = if stop_at_else { self.parse_or()? } else { self.parse_else_expr()? };
        // Simple assignment — use parse_or (not parse_else_expr) to avoid consuming
        // the `else` of an enclosing if-let as nil-coalescing
        if self.eat(&TokenKind::Eq) {
            let rhs = self.parse_or()?;
            return Ok(Stmt::Expr(Expr { kind: ExprKind::Assign(Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()}));
        }
        // Compound assignment
        let op = match self.peek() {
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
        if let Some(op) = op {
            self.advance();
            let rhs = self.parse_or()?;
            let binop = Expr { kind: ExprKind::BinOp(op, Box::new(lhs.clone()), Box::new(rhs)), line, col, len: self.tok_len()};
            return Ok(Stmt::Expr(Expr { kind: ExprKind::Assign(Box::new(lhs), Box::new(binop)), line, col, len: self.tok_len()}));
        }
        // Write-once / nil-coalescing assignment: `lhs ?= rhs`
        if self.eat(&TokenKind::QuestionEq) {
            let rhs = self.parse_or()?;
            return Ok(Stmt::Expr(Expr { kind: ExprKind::QuestionAssign(Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()}));
        }
        // Command-style call: `print "hello"` or `foo arg1, arg2`
        // Same logic as in parse_stmt, but here we don't consume the trailing newline
        // (that's the caller's responsibility in inline contexts).
        let expr = if let ExprKind::Var(_) = &lhs.kind {
            if self.peek_starts_expr() {
                let mut args = Vec::new();
                loop {
                    let arg = self.parse_or()?;
                    args.push(Arg { label: None, value: arg , spread: false});
                    if !self.eat(&TokenKind::Comma) { break; }
                }
                Expr { kind: ExprKind::Call(Box::new(lhs), args), line, col, len: self.tok_len()}
            } else {
                lhs
            }
        } else {
            lhs
        };
        Ok(Stmt::Expr(expr))
    }

    pub(crate) fn parse_else_expr(&mut self) -> Result<Expr, ParseError> {
        // try expr else default  OR  expr else default
        let line = self.line();
        let col = self.col();
        if self.check(&TokenKind::Try) {
            self.advance();

            // `try: block else ...` — multi-line try body.
            // Detected when the token after `try` is `:` followed by a newline.
            if self.check(&TokenKind::Colon) && (self.is_newline2() || self.check2(&TokenKind::Eof)) {
                self.advance(); // consume `:`
                self.expect_newline()?;
                let try_stmts = self.parse_block()?;
                // Expect `else` followed by either an inline expr or an indented block.
                // Both forms bind `error` in the else scope.
                self.expect(&TokenKind::Else)?;
                let else_stmts = self.parse_else_body_stmts()?;
                return Ok(Expr {
                    kind: ExprKind::TryElseBlock(try_stmts, else_stmts),
                    line, col, len: self.tok_len(), });
            }

            // `try? expr` — shorthand for `try expr else nil` (Result → Option).
            // Only form that does NOT bind `error` (there is no else body).
            if self.eat(&TokenKind::Question) {
                let inner = self.parse_pipe()?;
                return Ok(Expr {
                    kind: ExprKind::TryElse(
                        Box::new(inner),
                        Box::new(Expr { kind: ExprKind::Nil, line, col , len: self.tok_len()}),
                    ),
                    line, col, len: self.tok_len(), });
            }

            // `try expr else ...` — inline try expression with an else branch.
            // Fold into TryElseBlock so that `error` is always bound in the else scope,
            // regardless of whether the else body is inline or a block.
            //
            // We use parse_else_body_expr (not parse_else_body_stmts) so that the
            // inline form `try f() else 0` does NOT consume the trailing newline that
            // belongs to the enclosing statement.  The block form `try f() else: block`
            // returns ExprKind::Block whose stmts we unwrap directly.
            let inner = self.parse_pipe()?;
            if self.check(&TokenKind::Else) {
                self.advance();
                let else_expr = self.parse_else_body_expr()?;
                let else_stmts = match else_expr.kind {
                    ExprKind::Block(stmts) => stmts,
                    other => vec![Stmt::Expr(Expr { kind: other, line: else_expr.line, col, len: self.tok_len()})],
                };
                return Ok(Expr {
                    kind: ExprKind::TryElseBlock(
                        vec![Stmt::Expr(inner)],
                        else_stmts,
                    ),
                    line, col, len: self.tok_len(), });
            }
            // bare `try expr` without else — the `try` keyword has no effect here, which
            // is almost certainly a mistake. Return a parse error rather than silently
            // discarding the `try` wrapper. Guide the user toward a valid form:
            //   try f() else default    — expression with fallback
            //   try f() else: block     — expression with block fallback
            //   try? f()                — convert Result to Option (nil on error)
            //   try: … catch: …         — statement with typed catch
            return Err(ParseError::Generic {
                line, col,
                msg: "'try expr' requires an else clause or '?' — use:\n  \
                     try f() else default     (expression with fallback)\n  \
                     try? f()                 (nil on error)\n  \
                     try: … catch: …          (statement with catch)".to_string(), len: self.tok_len(),
            });
        }
        let expr = self.parse_pipe()?;
        if self.check(&TokenKind::Else) {
            self.advance();
            let default = self.parse_else_body_expr()?;
            let line = expr.line;
            return Ok(Expr {
                kind: ExprKind::Else(Box::new(expr), Box::new(default)),
                line, col, len: self.tok_len(), });
        }
        Ok(expr)
    }

    pub(crate) fn parse_pipe(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_or()?;
        let mut indent_depth: i32 = 0;
        loop {
            // Support multi-line pipe chains: eat newline+indent before `|>`
            // so the user can write:
            //   numbers
            //       |> filter(n: n > 0)
            //       |> map(n: n * 2)
            let skipped = if self.check(&TokenKind::Newline) {
                self.peek_pipe_after_newlines()
            } else {
                None
            };
            if let Some((offset, indents)) = skipped {
                self.skip_to_offset(offset);
                indent_depth += indents;
            }
            if !self.check(&TokenKind::PipeArrow) {
                break;
            }
            let line = self.line();
            let col = self.col();
            self.advance(); // consume `|>`
            // RHS: either `.method` (method pipe) or `func` (function pipe).
            // After a dot, allow keyword-named methods like `.wait`, `.done`.
            let (name, method_pipe) = if self.eat(&TokenKind::Dot) {
                (self.expect_ident_or_keyword()?, true)
            } else {
                (self.expect_ident()?, false)
            };
            // Optional argument list
            let args = if self.check(&TokenKind::LParen) {
                self.parse_call_args()?
            } else {
                vec![]
            };
            if method_pipe {
                // `lhs |> .method(args)` → `lhs.method(args)` — emit as MethodCall
                lhs = Expr { kind: ExprKind::MethodCall(Box::new(lhs), name, args), line, col, len: self.tok_len()};
            } else {
                lhs = Expr { kind: ExprKind::Pipe(Box::new(lhs), name, args), line, col, len: self.tok_len()};
            }
        }
        // Consume Dedents corresponding to any Indents we skipped over
        if indent_depth > 0 {
            let saved = self.pos;
            self.skip_newlines();
            let mut consumed = 0;
            while consumed < indent_depth && self.check(&TokenKind::Dedent) {
                self.advance();
                consumed += 1;
            }
            if consumed < indent_depth {
                self.pos = saved;
            }
        }
        Ok(lhs)
    }

    /// Peek ahead past newlines and indents to see if a `|>` token follows.
    /// Returns `(pipe_arrow_offset, indents_consumed)` or None.
    pub(crate) fn peek_pipe_after_newlines(&self) -> Option<(usize, i32)> {
        let mut i = self.pos;
        let mut indents: i32 = 0;
        while i < self.tokens.len() {
            match &self.tokens[i].kind {
                TokenKind::Newline => i += 1,
                TokenKind::Indent  => { indents += 1; i += 1; }
                // Do NOT cross Dedent — that would chain past a block boundary
                TokenKind::PipeArrow => return Some((i, indents)),
                _ => return None,
            }
        }
        None
    }

    pub(crate) fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_and()?;
        while self.check(&TokenKind::Or) {
            let line = self.line();
            let col = self.col();
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Expr { kind: ExprKind::BinOp(BinOp::Or, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
        }
        Ok(lhs)
    }

    pub(crate) fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_not()?;
        while self.check(&TokenKind::And) {
            let line = self.line();
            let col = self.col();
            self.advance();
            let rhs = self.parse_not()?;
            lhs = Expr { kind: ExprKind::BinOp(BinOp::And, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
        }
        Ok(lhs)
    }

    pub(crate) fn parse_not(&mut self) -> Result<Expr, ParseError> {
        if self.check(&TokenKind::Not) {
            let line = self.line();
            let col = self.col();
            self.advance();
            // Guard: `not not not …` chains recurse directly, one frame per `not`.
            // Each `not` contributes ~15 Rust frames; at MAX_EXPR_DEPTH the stack is
            // MAX * 15 frames deep which is safe within the 8 MB thread stack.
            self.depth += 1;
            if self.depth > crate::parser::MAX_EXPR_DEPTH {
                self.depth -= 1;
                return Err(ParseError::Generic {
                    line, col,
                    msg: format!("expression nested too deeply (limit: {})", crate::parser::MAX_EXPR_DEPTH), len: self.tok_len(),
                });
            }
            let result = self.parse_not();
            self.depth -= 1;
            return Ok(Expr { kind: ExprKind::UnaryOp(UnaryOp::Not, Box::new(result?)), line, col, len: self.tok_len()});
        }
        self.parse_comparison()
    }

    pub(crate) fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitor()?;
        loop {
            // Handle `is` and `is not` specially (need to peek ahead for `not`)
            if self.check(&TokenKind::Is) {
                let line = self.line();
                let col = self.col();
                self.advance(); // consume `is`
                let op = if self.check(&TokenKind::Not) {
                    self.advance(); // consume `not`
                    BinOp::IsNot
                } else {
                    BinOp::Is
                };
                let rhs = self.parse_bitor()?;
                lhs = Expr { kind: ExprKind::BinOp(op, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
                continue;
            }
            let op = match self.peek() {
                TokenKind::EqEqEq => BinOp::RefEq,
                TokenKind::EqEq   => BinOp::Eq,
                TokenKind::BangEq => BinOp::NotEq,
                TokenKind::LtEq   => BinOp::LtEq,
                TokenKind::GtEq   => BinOp::GtEq,
                // `<` is comparison only when NOT followed by another `<` (which would be shift)
                TokenKind::Lt if !self.check2(&TokenKind::Lt) => BinOp::Lt,
                // `>` is comparison only when NOT followed by another `>` (which would be shift)
                TokenKind::Gt if !self.check2(&TokenKind::Gt) => BinOp::Gt,
                _ => break,
            };
            let line = self.line();
            let col = self.col();
            self.advance();
            let rhs = self.parse_bitor()?;
            lhs = Expr { kind: ExprKind::BinOp(op, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
        }
        Ok(lhs)
    }

    pub(crate) fn parse_bitor(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitxor()?;
        while self.check(&TokenKind::Pipe) {
            let line = self.line();
            let col = self.col();
            self.advance();
            let rhs = self.parse_bitxor()?;
            lhs = Expr { kind: ExprKind::BinOp(BinOp::BitOr, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
        }
        Ok(lhs)
    }

    pub(crate) fn parse_bitxor(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_bitand()?;
        while self.check(&TokenKind::Caret) {
            let line = self.line();
            let col = self.col();
            self.advance();
            let rhs = self.parse_bitand()?;
            lhs = Expr { kind: ExprKind::BinOp(BinOp::BitXor, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
        }
        Ok(lhs)
    }

    pub(crate) fn parse_bitand(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_shift()?;
        while self.check(&TokenKind::Ampersand) {
            let line = self.line();
            let col = self.col();
            self.advance();
            let rhs = self.parse_shift()?;
            lhs = Expr { kind: ExprKind::BinOp(BinOp::BitAnd, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
        }
        Ok(lhs)
    }

    pub(crate) fn parse_shift(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_add()?;
        loop {
            // `<<`: two consecutive Lt tokens
            if self.check(&TokenKind::Lt) && self.check2(&TokenKind::Lt) {
                let line = self.line();
                let col = self.col();
                self.advance(); self.advance();
                let rhs = self.parse_add()?;
                lhs = Expr { kind: ExprKind::BinOp(BinOp::Shl, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
            // `>>`: two consecutive Gt tokens
            } else if self.check(&TokenKind::Gt) && self.check2(&TokenKind::Gt) {
                let line = self.line();
                let col = self.col();
                self.advance(); self.advance();
                let rhs = self.parse_add()?;
                lhs = Expr { kind: ExprKind::BinOp(BinOp::Shr, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    pub(crate) fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus  => BinOp::Add,
                TokenKind::Minus => BinOp::Sub,
                _ => break,
            };
            let line = self.line();
            let col = self.col();
            self.advance();
            let rhs = self.parse_mul()?;
            lhs = Expr { kind: ExprKind::BinOp(op, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
        }
        Ok(lhs)
    }

    pub(crate) fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star    => BinOp::Mul,
                TokenKind::Slash   => BinOp::Div,
                TokenKind::Percent => BinOp::Rem,
                _ => break,
            };
            let line = self.line();
            let col = self.col();
            self.advance();
            let rhs = self.parse_unary()?;
            lhs = Expr { kind: ExprKind::BinOp(op, Box::new(lhs), Box::new(rhs)), line, col, len: self.tok_len()};
        }
        Ok(lhs)
    }

    /// Unary (`-`/`!`/`~`) binds tighter than range (`..`/`..=`), matching Rust:
    /// `-1..2` is `(-1)..2`, not `-(1..2)`. So parse the full unary chain first
    /// via `parse_unary_no_range`, then attach a trailing range at this level —
    /// one layer above unary, below `parse_mul`.
    pub(crate) fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        let line = self.line();
        let col = self.col();
        let start = self.parse_unary_no_range()?;
        match self.peek() {
            TokenKind::DotDot | TokenKind::DotDotEq => {
                let inclusive = self.peek() == &TokenKind::DotDotEq;
                self.advance();
                // `M..` inside `[M..]` — open-ended slice (next token is `]`)
                if self.check(&TokenKind::RBracket) {
                    Ok(Expr {
                        kind: ExprKind::SliceRange { start: Some(Box::new(start)), end: None, inclusive },
                        line, col, len: self.tok_len(),
                    })
                } else {
                    let end = self.parse_unary_no_range()?;
                    Ok(Expr {
                        kind: ExprKind::Range { start: Box::new(start), end: Box::new(end), inclusive },
                        line, col, len: self.tok_len(),
                    })
                }
            }
            _ => Ok(start),
        }
    }

    fn parse_unary_no_range(&mut self) -> Result<Expr, ParseError> {
        let line = self.line();
        let col = self.col();
        match self.peek().clone() {
            TokenKind::Minus | TokenKind::Bang | TokenKind::Tilde => {
                let op = match self.peek() {
                    TokenKind::Minus => UnaryOp::Neg,
                    TokenKind::Bang => UnaryOp::Not,
                    _ => UnaryOp::BitNot,
                };
                self.advance();
                // Guard: `----x` / `!!!!x` / `~~~~x` chains recurse directly, one frame
                // per operator — same bound as the `not`-chain guard in `parse_not`.
                self.depth += 1;
                if self.depth > crate::parser::MAX_EXPR_DEPTH {
                    self.depth -= 1;
                    return Err(ParseError::Generic {
                        line, col,
                        msg: format!("expression nested too deeply (limit: {})", crate::parser::MAX_EXPR_DEPTH), len: self.tok_len(),
                    });
                }
                let expr = self.parse_unary_no_range();
                self.depth -= 1;
                Ok(Expr { kind: ExprKind::UnaryOp(op, Box::new(expr?)), line, col, len: self.tok_len()})
            }
            TokenKind::New => {
                self.advance();
                let arena = if matches!(self.peek(), TokenKind::LParen) {
                    self.advance(); // consume '('
                    let expr = self.parse_expr()?;
                    self.expect(&TokenKind::RParen)?;
                    Some(Box::new(expr))
                } else {
                    None
                };
                let ctor = self.parse_postfix_top_level()?;
                Ok(Expr { kind: ExprKind::New { arena, ctor: Box::new(ctor) }, line, col, len: self.tok_len()})
            }
            _ => self.parse_postfix_top_level(),
        }
    }

    /// Like parse_postfix but allows chaining across Newline+Indent for trailing closures.
    /// Call this from statement-level parsing where chain continuation is valid.
    pub(crate) fn parse_postfix_top_level(&mut self) -> Result<Expr, ParseError> {
        self.parse_postfix_inner(true)
    }

    pub(crate) fn parse_postfix_inner(&mut self, allow_chain_continuation: bool) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        // Track how many Indent tokens we consumed for chain continuation so we
        // can consume the matching Dedent tokens after the chain ends.
        let mut continuation_indent_depth: i32 = 0;
        // Whether we've seen at least one trailing closure (enables chain continuation)
        let mut seen_trailing_closure = false;
        loop {
            // Allow chaining across newlines when the next non-whitespace token after
            // newlines/indents is a `.` (method/field chain continuation).
            // This supports inline trailing closures like:
            //   [1,2,3].filter (x): x > 1
            //           .map (x): x * 2
            // Chain continuation is only enabled when:
            // 1. We're at the top-level postfix (allow_chain_continuation = true), AND
            // 2. We've already parsed at least one trailing closure
            if self.check(&TokenKind::Newline) && allow_chain_continuation && seen_trailing_closure {
                if let Some((dot_offset, indents_consumed)) = self.peek_dot_after_newlines_and_indents() {
                    // Consume the newlines and any indent/dedent up to the dot
                    self.skip_to_offset(dot_offset);
                    continuation_indent_depth += indents_consumed;
                } else {
                    break;
                }
            } else if self.is_newline() {
                break;
            }
            let line = self.line();
            let col = self.col();
            match self.peek().clone() {
                TokenKind::Dot => {
                    self.advance();
                    // Support tuple index access: `t.0`, `t.1`, etc.
                    // Also allow keywords as field/method names (e.g. `future.wait`, `list.join()`).
                    let field = if let TokenKind::Int(n) = self.peek().clone() {
                        self.advance();
                        n.to_string()
                    } else {
                        self.expect_ident_or_keyword()?
                    };
                    // Check for trailing closure FIRST (before regular call args),
                    // since `(x):` looks like an LParen but is actually a closure param list.
                    if self.peek_is_trailing_closure() {
                        // Method call with only a trailing closure (no parentheses for regular args)
                        let args = self.parse_trailing_closure(vec![])?;
                        seen_trailing_closure = true;
                        expr = Expr { kind: ExprKind::MethodCall(Box::new(expr), field, args), line, col, len: self.tok_len()};
                    } else if self.check(&TokenKind::LParen) {
                        let mut args = self.parse_call_args()?;
                        // Check for trailing closure after the argument list
                        if self.peek_is_trailing_closure() {
                            args = self.parse_trailing_closure(args)?;
                            seen_trailing_closure = true;
                        } else if self.peek_is_trailing_closure_no_paren() {
                            args = self.parse_trailing_closure_no_paren(args)?;
                            seen_trailing_closure = true;
                        } else if self.allow_trailing_closure && self.check(&TokenKind::Colon) {
                            // Bare `:` after call args = zero-arg trailing body.
                            // e.g. `timeout(Duration.fromSecs(5)): fetch(url)`
                            args = self.parse_trailing_body(args)?;
                            seen_trailing_closure = true;
                        } else if self.peek_is_trailing_body_no_colon() {
                            // No-colon trailing body: `f(args) expr` — same-line, no separator.
                            // e.g. `timeout(Duration.fromSecs(5)) fetch(url)`
                            args = self.parse_trailing_body_no_colon(args)?;
                            seen_trailing_closure = true;
                        }
                        expr = Expr { kind: ExprKind::MethodCall(Box::new(expr), field, args), line, col, len: self.tok_len()};
                    } else if self.peek_is_trailing_closure_no_paren() {
                        // Method call with only a no-paren trailing closure: `expr.method x: body`
                        let args = self.parse_trailing_closure_no_paren(vec![])?;
                        seen_trailing_closure = true;
                        expr = Expr { kind: ExprKind::MethodCall(Box::new(expr), field, args), line, col, len: self.tok_len()};
                    } else {
                        expr = Expr { kind: ExprKind::Field(Box::new(expr), field), line, col, len: self.tok_len()};
                    }
                }
                TokenKind::QuestionDot => {
                    // Optional chaining: ?.field or ?.method(args)
                    self.advance();
                    let field = self.expect_ident_or_keyword()?;
                    if self.peek_is_trailing_closure() {
                        let args = self.parse_trailing_closure(vec![])?;
                        seen_trailing_closure = true;
                        expr = Expr { kind: ExprKind::OptionalMethodCall(Box::new(expr), field, args), line, col, len: self.tok_len()};
                    } else if self.check(&TokenKind::LParen) {
                        let mut args = self.parse_call_args()?;
                        if self.peek_is_trailing_closure() {
                            args = self.parse_trailing_closure(args)?;
                            seen_trailing_closure = true;
                        }
                        expr = Expr { kind: ExprKind::OptionalMethodCall(Box::new(expr), field, args), line, col, len: self.tok_len()};
                    } else {
                        expr = Expr { kind: ExprKind::OptionalField(Box::new(expr), field), line, col, len: self.tok_len()};
                    }
                }
                TokenKind::LBracket => {
                    self.advance();
                    // Labeled multi-dim indexing: a[width = w, height = h, ...] —
                    // mandatory labels, order-free at the use site (see
                    // docs/array-multidim-proposal.md, "Indexing"). Same `Ident`
                    // followed by `=` (not `==`) lookahead already used to detect
                    // labeled call arguments in `parse_arg`, so `a[i]`/`a[i == j]`/
                    // `a[..n]` etc. are never misdetected as this form.
                    let is_labeled_index = matches!(self.peek(), TokenKind::Ident(_))
                        && self.check2(&TokenKind::Eq)
                        && !matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind), Some(TokenKind::Eq));
                    if is_labeled_index {
                        let mut args = vec![self.parse_arg()?];
                        while self.eat(&TokenKind::Comma) {
                            if self.check(&TokenKind::RBracket) { break; } // trailing comma
                            args.push(self.parse_arg()?);
                        }
                        self.expect(&TokenKind::RBracket)?;
                        expr = Expr { kind: ExprKind::LabeledIndex(Box::new(expr), args), line, col, len: self.tok_len() };
                        continue;
                    }
                    // Slice with no start: `a[..N]`, `a[..=N]`, `a[..]`
                    let idx = if self.check(&TokenKind::DotDot) || self.check(&TokenKind::DotDotEq) {
                        let inclusive = self.peek() == &TokenKind::DotDotEq;
                        self.advance();
                        let end = if self.check(&TokenKind::RBracket) {
                            None
                        } else {
                            Some(Box::new(self.parse_expr()?))
                        };
                        Expr { kind: ExprKind::SliceRange { start: None, end, inclusive }, line, col, len: self.tok_len() }
                    } else {
                        let inner = self.parse_expr()?;
                        // `parse_expr` consumed `M..N` → Range; convert to SliceRange.
                        // `M..` with no end is caught in the DotDot postfix arm below.
                        match inner.kind {
                            ExprKind::Range { start, end, inclusive } =>
                                Expr { kind: ExprKind::SliceRange { start: Some(start), end: Some(end), inclusive }, line, col, len: self.tok_len() },
                            ExprKind::SliceRange { .. } => inner, // M.. already produced by postfix
                            _ => inner,
                        }
                    };
                    self.expect(&TokenKind::RBracket)?;
                    expr = Expr { kind: ExprKind::Index(Box::new(expr), Box::new(idx)), line, col, len: self.tok_len()};
                }
                TokenKind::LParen => {
                    // Check for trailing closure FIRST before trying to parse as regular call
                    if self.peek_is_trailing_closure() {
                        let args = self.parse_trailing_closure(vec![])?;
                        seen_trailing_closure = true;
                        expr = Expr { kind: ExprKind::Call(Box::new(expr), args), line, col, len: self.tok_len()};
                    } else {
                        let mut args = self.parse_call_args()?;
                        // Check for trailing closure after the argument list
                        if self.peek_is_trailing_closure() {
                            args = self.parse_trailing_closure(args)?;
                            seen_trailing_closure = true;
                        } else if self.allow_trailing_closure && self.check(&TokenKind::Colon) {
                            // Bare `:` after call args = zero-arg trailing body.
                            // e.g. `timeout(Duration.fromSecs(5)): fetch(url)`
                            args = self.parse_trailing_body(args)?;
                            seen_trailing_closure = true;
                        } else if self.peek_is_trailing_body_no_colon() {
                            // No-colon trailing body: `f(args) expr` — same-line, no separator.
                            // e.g. `timeout(Duration.fromSecs(5)) fetch(url)`
                            args = self.parse_trailing_body_no_colon(args)?;
                            seen_trailing_closure = true;
                        }
                        expr = Expr { kind: ExprKind::Call(Box::new(expr), args), line, col, len: self.tok_len()};
                    }
                }
                TokenKind::As => {
                    self.advance();
                    // Cross-label array mapping: `img as [line = width, column = height]`
                    // (docs/array-multidim-proposal.md, "Cross-label compatibility") — the
                    // bracket contents are a `target_label = source_label` mapping table,
                    // not a type (`parse_type()` can't parse this: `line` alone parses as
                    // `Type::Named("line")`, then hits `=` and fails) — so it needs its own
                    // lookahead and its own node, not `Cast`. A real cast to a labeled-array
                    // type (`x as [float, width, height]`) starts with a *type* token
                    // (`float`, not `label =`), so this never misfires on that.
                    let is_relabel = self.check(&TokenKind::LBracket)
                        && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Ident(_)))
                        && matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind), Some(TokenKind::Eq));
                    if is_relabel {
                        self.advance(); // consume `[`
                        let mut pairs = vec![self.parse_relabel_pair()?];
                        while self.eat(&TokenKind::Comma) {
                            if self.check(&TokenKind::RBracket) { break; } // trailing comma
                            pairs.push(self.parse_relabel_pair()?);
                        }
                        self.expect(&TokenKind::RBracket)?;
                        expr = Expr { kind: ExprKind::RelabelCast(Box::new(expr), pairs), line, col, len: self.tok_len() };
                    } else {
                        let ty = self.parse_type()?;
                        expr = Expr { kind: ExprKind::Cast(Box::new(expr), ty), line, col, len: self.tok_len()};
                    }
                }
                TokenKind::Ident(_) if self.peek_is_trailing_closure_no_paren() => {
                    // `expr x: body` — single-param trailing closure without parens
                    let args = self.parse_trailing_closure_no_paren(vec![])?;
                    seen_trailing_closure = true;
                    expr = Expr {
                        kind: ExprKind::Call(Box::new(expr), args),
                        line, col, len: self.tok_len(), };
                }
                _ => break,
            }
        }
        // Consume any Dedent tokens that were produced by indent levels we skipped
        // for chain continuation. We may need to skip a Newline first.
        if continuation_indent_depth > 0 {
            // Skip to the Dedents: skip Newline, then eat matching Dedents
            let saved_pos = self.pos;
            self.skip_newlines();
            let mut consumed = 0;
            while consumed < continuation_indent_depth && self.check(&TokenKind::Dedent) {
                self.advance();
                consumed += 1;
            }
            // If we didn't find all the expected Dedents, restore position
            if consumed < continuation_indent_depth {
                self.pos = saved_pos;
            }
        }
        Ok(expr)
    }

    /// Peek ahead past Newline/Indent tokens and return `(dot_position, indents_consumed)`
    /// if the first non-whitespace, non-Indent token is a Dot, otherwise return None.
    /// Only crosses Newline and Indent tokens (not Dedent) to allow visual-alignment chaining.
    pub(crate) fn peek_dot_after_newlines_and_indents(&self) -> Option<(usize, i32)> {
        let mut i = self.pos;
        let mut indents_consumed: i32 = 0;
        while i < self.tokens.len() {
            match &self.tokens[i].kind {
                TokenKind::Newline => i += 1,
                TokenKind::Indent => { indents_consumed += 1; i += 1; }
                // Do NOT cross Dedent — that would chain past a block boundary
                TokenKind::Dot => return Some((i, indents_consumed)),
                _ => return None,
            }
        }
        None
    }

    /// Advance `self.pos` to the given absolute offset.
    pub(crate) fn skip_to_offset(&mut self, offset: usize) {
        self.pos = offset;
    }

    /// Returns true if the current token can start a primary expression (used for command-style calls).
    /// Excludes newline, EOF, dedent, and tokens that would be ambiguous in statement position.
    pub(crate) fn peek_starts_expr(&self) -> bool {
        matches!(
            self.tokens.get(self.pos).map(|t| &t.kind),
            Some(TokenKind::Str(_))
            | Some(TokenKind::StringInterp(_))
            | Some(TokenKind::Int(_))
            | Some(TokenKind::Float(_))
            | Some(TokenKind::Bool(_))
            | Some(TokenKind::Nil)
            | Some(TokenKind::LParen)
            | Some(TokenKind::LBracket)
            | Some(TokenKind::Ident(_))
        )
    }

    /// Returns true if the current position looks like a no-paren single-param trailing closure: `Ident ':'`.
    pub(crate) fn peek_is_trailing_closure_no_paren(&self) -> bool {
        self.allow_noparen_closure
            && matches!(self.tokens.get(self.pos).map(|t| &t.kind), Some(TokenKind::Ident(_)))
            && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Colon))
    }

    /// Returns true if the current token position starts a trailing closure: `(params):`.
    ///
    /// Scans ahead to find the matching `)` and checks if the next token is `:`.
    /// Crucially, also verifies that the content between `()` looks like a closure
    /// *parameter list* (identifiers, commas, type annotations) and NOT like function
    /// *call arguments* (expressions with `.`, `=`, operators, literals, nested calls).
    ///
    /// This prevents false positives such as:
    ///   `timeout(Duration.fromSecs(5)):` — call arg, not closure params  (contains `.`)
    ///   `timeout(duration = dur):` — labeled call arg, not closure params (contains `=`)
    ///   `filter(x > 0):` — expression arg, not closure params            (contains `>`)
    pub(crate) fn peek_is_trailing_closure(&self) -> bool {
        if !self.allow_trailing_closure {
            return false;
        }
        if !matches!(self.tokens.get(self.pos).map(|t| &t.kind), Some(TokenKind::LParen)) {
            return false;
        }
        let mut depth = 0usize;
        let mut i = self.pos;
        while i < self.tokens.len() {
            match &self.tokens[i].kind {
                TokenKind::LParen => {
                    if depth > 0 {
                        // Nested `(` inside the outer `()` means a function call argument
                        // like `f(getTimer())` — this is call args, not closure params.
                        return false;
                    }
                    depth += 1;
                }
                TokenKind::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        // Verify next token is `:`
                        return matches!(
                            self.tokens.get(i + 1).map(|t| &t.kind),
                            Some(TokenKind::Colon)
                        );
                    }
                }
                // ── Tokens that mean "this is a call argument, not a closure param" ──
                //
                // Field/method access and labeled arguments:
                TokenKind::Dot | TokenKind::Eq => return false,
                // Arithmetic / comparison / logical operators:
                TokenKind::Plus | TokenKind::Minus | TokenKind::Star | TokenKind::Slash
                | TokenKind::Percent | TokenKind::PlusEq | TokenKind::MinusEq
                | TokenKind::StarEq | TokenKind::SlashEq | TokenKind::PercentEq
                | TokenKind::EqEq | TokenKind::BangEq | TokenKind::EqEqEq
                | TokenKind::Lt | TokenKind::Gt | TokenKind::LtEq | TokenKind::GtEq
                | TokenKind::And | TokenKind::Or
                | TokenKind::Ampersand | TokenKind::AmpersandEq
                | TokenKind::Pipe | TokenKind::PipeEq
                | TokenKind::Caret | TokenKind::CaretEq => return false,
                // Range operators:
                TokenKind::DotDot | TokenKind::DotDotEq | TokenKind::DotDotDot => return false,
                // Literal values (can never be a param name):
                TokenKind::Int(_) | TokenKind::Float(_) | TokenKind::Bool(_)
                | TokenKind::Nil => return false,
                // String / interpolated string literal:
                TokenKind::Str(_) | TokenKind::StringInterp(_) => return false,
                // Collection / dict literals:
                TokenKind::LBracket | TokenKind::LBrace => return false,
                TokenKind::Newline | TokenKind::Semicolon | TokenKind::Eof => break,
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// Parse optional `throws`/`task` modifiers before a closure `:`.
    /// Either order is accepted; explicit flags are OR'd with inferred ones after the body is parsed.
    pub(crate) fn parse_closure_modifiers(&mut self) -> (bool, bool) {
        let mut throws = self.eat(&TokenKind::Throws);
        let mut task   = self.eat(&TokenKind::Task);
        if !throws { throws = self.eat(&TokenKind::Throws); }
        if !task   { task   = self.eat(&TokenKind::Task);   }
        (throws, task)
    }

    /// Infer `throws` and `task` flags from a closure/function body by scanning for
    /// `throw` statements and `task` expressions respectively.
    pub(crate) fn infer_throws_task(body: &ClosureBody) -> (bool, bool) {
        match body {
            ClosureBody::Expr(e) => {
                let (t, k) = Self::scan_expr_throws_task(e);
                (t, k)
            }
            ClosureBody::Block(stmts) => Self::scan_stmts_throws_task(stmts),
        }
    }

    pub(crate) fn scan_stmts_throws_task(stmts: &[Stmt]) -> (bool, bool) {
        let mut throws = false;
        let mut task = false;
        for s in stmts {
            let (t, k) = Self::scan_stmt_throws_task(s);
            throws |= t;
            task |= k;
        }
        (throws, task)
    }

    pub(crate) fn scan_stmt_throws_task(s: &Stmt) -> (bool, bool) {
        match s {
            Stmt::Throw(_) => (true, false),
            Stmt::Expr(e) => Self::scan_expr_throws_task(e),
            Stmt::Let(l) => l.value.as_ref().map(Self::scan_expr_throws_task).unwrap_or((false, false)),
            Stmt::Return(r) => r.value.as_ref().map(Self::scan_expr_throws_task).unwrap_or((false, false)),
            Stmt::If(i) => {
                let mut t = false; let mut k = false;
                for (cond, body) in &i.branches {
                    let (ct, ck) = Self::scan_expr_throws_task(cond); t |= ct; k |= ck;
                    let (bt, bk) = Self::scan_stmts_throws_task(body); t |= bt; k |= bk;
                }
                if let Some(e) = &i.else_body { let (bt, bk) = Self::scan_stmts_throws_task(e); t |= bt; k |= bk; }
                (t, k)
            }
            Stmt::While(w) => {
                let (ct, ck) = Self::scan_expr_throws_task(&w.condition);
                let (bt, bk) = Self::scan_stmts_throws_task(&w.body);
                (ct | bt, ck | bk)
            }
            Stmt::For(f) => Self::scan_stmts_throws_task(&f.body),
            Stmt::Try(_) => (true, false),  // try block may catch throws, but the body throws
            _ => (false, false),
        }
    }

    pub(crate) fn scan_expr_throws_task(e: &Expr) -> (bool, bool) {
        match &e.kind {
            ExprKind::Task(_) => (false, true),
            ExprKind::Call(callee, args) => {
                let (mut t, mut k) = Self::scan_expr_throws_task(callee);
                for a in args { let (at, ak) = Self::scan_expr_throws_task(&a.value); t |= at; k |= ak; }
                (t, k)
            }
            ExprKind::MethodCall(obj, _, args) => {
                let (mut t, mut k) = Self::scan_expr_throws_task(obj);
                for a in args { let (at, ak) = Self::scan_expr_throws_task(&a.value); t |= at; k |= ak; }
                (t, k)
            }
            ExprKind::BinOp(_, l, r) => {
                let (lt, lk) = Self::scan_expr_throws_task(l);
                let (rt, rk) = Self::scan_expr_throws_task(r);
                (lt | rt, lk | rk)
            }
            ExprKind::Block(stmts) | ExprKind::Do(stmts) => Self::scan_stmts_throws_task(stmts),
            ExprKind::MacroCall { args, .. } => {
                let (mut t, mut k) = (false, false);
                for a in args { let (at, ak) = Self::scan_expr_throws_task(a); t |= at; k |= ak; }
                (t, k)
            }
            ExprKind::If(i) => {
                let mut t = false; let mut k = false;
                for (cond, body) in &i.branches {
                    let (ct, ck) = Self::scan_expr_throws_task(cond); t |= ct; k |= ck;
                    let (bt, bk) = Self::scan_stmts_throws_task(body); t |= bt; k |= bk;
                }
                (t, k)
            }
            _ => (false, false),
        }
    }

    /// Parse a trailing closure `(params): body`, append the resulting Closure arg to
    /// `existing_args`, and return the updated arg list.
    /// After parsing a multiline closure body, checks that `.` (chaining) does not follow.
    pub(crate) fn parse_trailing_closure(&mut self, mut existing_args: Vec<Arg>) -> Result<Vec<Arg>, ParseError> {
        let line = self.line();
        let col = self.col();
        // Parse the closure parameter list: `(param, param, ...)`
        let params = self.parse_closure_params()?;
        let (ex_throws, ex_task) = self.parse_closure_modifiers();
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_closure_body()?;

        // After a multiline trailing closure, chaining with `.` is forbidden
        if matches!(body, ClosureBody::Block(_)) {
            // Skip newlines and dedents to find what comes next
            let mut i = self.pos;
            while i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Newline | TokenKind::Dedent) {
                i += 1;
            }
            if i < self.tokens.len() && self.tokens[i].kind == TokenKind::Dot {
                return Err(ParseError::Generic {
                    line, col,
                    msg: "multiline trailing closure cannot be chained — use parentheses: f((x):\n    body).next()".into(), len: self.tok_len(),
                });
            }
        }

        let (inf_throws, inf_task) = Self::infer_throws_task(&body);
        let closure_expr = Expr {
            kind: ExprKind::Closure(params, None, body, ex_throws | inf_throws, ex_task | inf_task),
            line, col, len: self.tok_len(), };
        existing_args.push(Arg { label: None, value: closure_expr , spread: false});
        Ok(existing_args)
    }

    /// Parse a bare trailing body `: body` (zero-arg, no param list at all).
    ///
    /// Used when a function call is followed directly by `:` without `(params)`,
    /// e.g. `timeout(Duration.fromSecs(5)): fetch(url)`.
    /// The body is wrapped in a zero-argument closure and appended to `existing_args`.
    pub(crate) fn parse_trailing_body(&mut self, mut existing_args: Vec<Arg>) -> Result<Vec<Arg>, ParseError> {
        let line = self.line();
        let col = self.col();
        self.expect(&TokenKind::Colon)?;
        let body = self.parse_closure_body()?;
        // Multiline body cannot be chained
        if matches!(body, ClosureBody::Block(_)) {
            let mut i = self.pos;
            while i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Newline | TokenKind::Dedent) {
                i += 1;
            }
            if i < self.tokens.len() && self.tokens[i].kind == TokenKind::Dot {
                return Err(ParseError::Generic {
                    line, col,
                    msg: "multiline trailing body cannot be chained — use parentheses: f(args, ():\n    body).next()".into(), len: self.tok_len(),
                });
            }
        }
        let (inf_throws, inf_task) = Self::infer_throws_task(&body);
        let closure_expr = Expr {
            kind: ExprKind::Closure(vec![], None, body, inf_throws, inf_task),
            line, col, len: self.tok_len(), };
        existing_args.push(Arg { label: None, value: closure_expr, spread: false });
        Ok(existing_args)
    }

    /// Parse a no-colon trailing body `expr` (zero-arg, no separator at all).
    ///
    /// Handles the command-style form `f(args) expr` where `expr` is on the same
    /// line — analogous to how `task(dur) body` works without a `:`.
    /// e.g. `timeout(Duration.fromSecs(5)) fetch(url)`
    ///   → equivalent to `timeout(Duration.fromSecs(5), (): fetch(url))`
    pub(crate) fn parse_trailing_body_no_colon(&mut self, mut existing_args: Vec<Arg>) -> Result<Vec<Arg>, ParseError> {
        let line = self.line();
        let col = self.col();
        let body_expr = self.parse_or()?;
        let (inf_throws, inf_task) = Self::infer_throws_task(&ClosureBody::Expr(Box::new(body_expr.clone())));
        let closure_expr = Expr {
            kind: ExprKind::Closure(vec![], None, ClosureBody::Expr(Box::new(body_expr)), inf_throws, inf_task),
            line, col, len: self.tok_len(), };
        existing_args.push(Arg { label: None, value: closure_expr, spread: false });
        Ok(existing_args)
    }

    /// Returns true if the current position starts a no-colon trailing body:
    /// an identifier on the same line (typically a function call), not followed by `:`
    /// (which would be the no-paren closure param form `ident: body`).
    ///
    /// Restricted to `Ident` only to avoid false positives with postfix operators:
    /// - `LBracket` is subscript access (`arr[0]`), not a trailing body
    /// - `LParen`   is a chained call or grouping, not a trailing body
    /// - Literals   are not meaningful trailing bodies
    pub(crate) fn peek_is_trailing_body_no_colon(&self) -> bool {
        if !self.allow_trailing_closure { return false; }
        // Must be an Ident (variable or function call start) — not LBracket, LParen, literals
        if !matches!(self.tokens.get(self.pos).map(|t| &t.kind), Some(TokenKind::Ident(_))) {
            return false;
        }
        // If Ident is followed by `:`, that's the no-paren closure param form — let it win
        if self.peek_is_trailing_closure_no_paren() { return false; }
        true
    }

    /// Parse a trailing closure `x: body` (no-paren single-param form), append the resulting
    /// Closure arg to `existing_args`, and return the updated arg list.
    pub(crate) fn parse_trailing_closure_no_paren(&mut self, mut existing_args: Vec<Arg>) -> Result<Vec<Arg>, ParseError> {
        let line = self.line();
        let col = self.col();
        let name = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        let param = Param { name, ty: None, mutable: false, rebindable: false, owned: false, variadic: false, default: None, line, col };
        let body = self.parse_closure_body()?;
        // Check: multiline trailing closure cannot be chained
        if matches!(body, ClosureBody::Block(_)) && matches!(self.peek(), TokenKind::Dot) {
            return Err(ParseError::Generic {
                line: self.line(), col: self.col(),
                msg: "multiline trailing closure cannot be chained — use parentheses: f((x):\n    body).next()".into(), len: self.tok_len(),
            });
        }
        let (throws, task) = Self::infer_throws_task(&body);
        let closure_expr = Expr {
            kind: ExprKind::Closure(vec![param], None, body, throws, task),
            line, col, len: self.tok_len(), };
        existing_args.push(Arg { label: None, value: closure_expr , spread: false});
        Ok(existing_args)
    }

    /// Parse closure parameter list `(param, param, ...)` — the parens are consumed here.
    pub(crate) fn parse_closure_params(&mut self) -> Result<Vec<Param>, ParseError> {
        self.expect(&TokenKind::LParen)?;
        self.skip_newlines_and_indent(); // allow `(\n    param,` multi-line form
        let mut params = Vec::new();

        if self.check(&TokenKind::RParen) {
            self.advance();
            return Ok(params);
        }

        // Detect typed closure params (same logic as peek_is_typed_closure but consuming)
        // We try to parse params; they may be typed or untyped.
        // Strategy: parse as param (which handles both `name` and `Type name` forms),
        // collecting until `)`.
        // For untyped params like `(x, y)`, parse_param handles `x` → name only.
        // For typed params like `(Int x)`, parse_param handles the type-before-name form.
        while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
            // Check if this looks like a typed param or untyped
            if self.is_type_start_before_ident() {
                params.push(self.parse_param()?);
            } else {
                // Untyped param: just an identifier
                let line = self.line();
                let col = self.col();
                let name = self.expect_ident()?;
                params.push(Param { name, ty: None, mutable: false, rebindable: false, owned: false, variadic: false, default: None, line, col });
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
            self.skip_newlines_and_indent(); // allow newline + indent between params
        }
        self.skip_newlines_and_indent(); // allow newline before `)`
        self.expect(&TokenKind::RParen)?;
        Ok(params)
    }

    pub(crate) fn parse_call_args(&mut self) -> Result<Vec<Arg>, ParseError> {
        self.expect(&TokenKind::LParen)?;
        self.skip_newlines_and_indent(); // allow `(\n    arg,` multi-line form
        // Inside explicit parentheses, `(params): body` closures are always valid,
        // regardless of the outer allow_trailing_closure setting.
        let saved_tc = self.allow_trailing_closure;
        self.allow_trailing_closure = true;
        let mut args = Vec::new();
        while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
            args.push(self.parse_arg()?);
            if !self.eat(&TokenKind::Comma) { break; }
            self.skip_newlines_and_indent(); // allow newline + indent between args
        }
        self.skip_newlines_and_indent(); // allow newline before `)`
        self.expect(&TokenKind::RParen)?;
        self.allow_trailing_closure = saved_tc;
        Ok(args)
    }

    /// One `target_label = source_label` pair inside `as [...]` (cross-label
    /// array mapping). See the `TokenKind::As` postfix arm above.
    fn parse_relabel_pair(&mut self) -> Result<(String, String), ParseError> {
        let target = self.expect_ident()?;
        self.expect(&TokenKind::Eq)?;
        let source = self.expect_ident()?;
        Ok((target, source))
    }

    pub(crate) fn parse_arg(&mut self) -> Result<Arg, ParseError> {
        // Spread arg: `..expr` — copy all fields from the given struct value.
        if self.eat(&TokenKind::DotDot) {
            let value = self.parse_or()?;
            return Ok(Arg { label: None, value, spread: true });
        }
        // Labeled arg: `ident= expr` or `ident: expr`
        // Detected by `Ident` followed immediately by `Eq` or `Colon`.
        if matches!(self.peek(), TokenKind::Ident(_))
            && self.check2(&TokenKind::Eq) {
                // Make sure it is not `ident ==` (equality check)
                let after_eq_pos = self.pos + 2;
                let is_double_eq = after_eq_pos < self.tokens.len()
                    && self.tokens[after_eq_pos].kind == TokenKind::Eq;
                if !is_double_eq {
                    let label = self.expect_ident()?;
                    self.advance(); // consume `=`
                    let value = self.parse_or()?; // stop before comma
                    return Ok(Arg { label: Some(label), value, spread: false });
                }
            }
            // Note: `ident: expr` is NOT parsed as a labeled arg here because it is
            // indistinguishable from a no-paren closure `x: body`. Use `ident= expr`
            // for labeled arguments instead. The `ident:` form is parsed as a
            // no-paren closure in parse_primary when allow_noparen_closure is true.
        let value = self.parse_expr()?;
        Ok(Arg { label: None, value, spread: false })
    }

    pub(crate) fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        let line = self.line();
        let col = self.col();
        match self.peek().clone() {
            TokenKind::Int(n) => {
                self.advance();
                Ok(Expr { kind: ExprKind::Int(n), line, col, len: self.tok_len()})
            }
            TokenKind::Float(f) => {
                self.advance();
                Ok(Expr { kind: ExprKind::Float(f), line, col, len: self.tok_len()})
            }
            TokenKind::Str(s) => {
                self.advance();
                Ok(Expr { kind: ExprKind::Str(s), line, col, len: self.tok_len()})
            }
            TokenKind::StringInterp(parts) => {
                let parts = parts.clone();
                self.advance();
                let segments = self.resolve_interp(parts, line)?;
                Ok(Expr { kind: ExprKind::StringInterp(segments), line, col, len: self.tok_len()})
            }
            TokenKind::Bool(b) => {
                self.advance();
                Ok(Expr { kind: ExprKind::Bool(b), line, col, len: self.tok_len()})
            }
            TokenKind::Nil => {
                self.advance();
                Ok(Expr { kind: ExprKind::Nil, line, col, len: self.tok_len()})
            }
            TokenKind::Void => {
                self.advance();
                Ok(Expr { kind: ExprKind::Void, line, col, len: self.tok_len()})
            }
            TokenKind::SelfKw => {
                self.advance();
                Ok(Expr { kind: ExprKind::Var("self".to_string()), line, col, len: self.tok_len()})
            }
            TokenKind::Ident(s) => {
                let name = s.clone();
                // Macro call: `name!(...)`, `name![...]`, `name!{...}`
                // Lookahead: tokens[pos+1] == Bang AND tokens[pos+2] is a delimiter.
                {
                    let is_bang  = matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Bang));
                    let is_delim = matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind),
                                           Some(TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace));
                    if is_bang && is_delim {
                        self.advance(); // consume Ident
                        self.advance(); // consume Bang
                        let args = self.parse_macro_args()?;
                        return Ok(Expr { kind: ExprKind::MacroCall { name, args }, line, col, len: self.tok_len()});
                    }
                }
                // Generic call: `name<T1, T2>(args)`
                // Lookahead: check for `< Type, ... > (` without ambiguity with `<` operator.
                if matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Lt))
                    && self.is_generic_call_ahead(self.pos + 1)
                {
                    self.advance(); // consume Ident
                    self.advance(); // consume '<'
                    let mut type_args: Vec<Type> = Vec::new();
                    type_args.push(self.parse_type()?);
                    while self.eat(&TokenKind::Comma) {
                        if self.check(&TokenKind::Gt) { break; }
                        type_args.push(self.parse_type()?);
                    }
                    self.expect(&TokenKind::Gt)?;
                    let callee = Expr { kind: ExprKind::Var(name), line, col, len: self.tok_len()};
                    let args = if self.check(&TokenKind::LParen) {
                        self.parse_call_args()?
                    } else {
                        vec![]
                    };
                    return Ok(Expr { kind: ExprKind::GenericCall(Box::new(callee), type_args, args), line, col, len: self.tok_len()});
                }
                // Single-param closure without parens: `x: body`
                // Only when allow_noparen_closure is set — disabled in condition contexts
                // where `:` is a body separator (if, while, for, match, if let, guard)
                if self.allow_noparen_closure
                    && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Colon))
                {
                    self.advance(); // consume ident
                    self.advance(); // consume ':'
                    let param = Param { name, ty: None, mutable: false, rebindable: false, owned: false, variadic: false, default: None, line, col };
                    let body = self.parse_closure_body()?;
                    let (throws, task) = Self::infer_throws_task(&body);
                    return Ok(Expr { kind: ExprKind::Closure(vec![param], None, body, throws, task), line, col, len: self.tok_len()});
                }
                self.advance();
                Ok(Expr { kind: ExprKind::Var(name), line, col, len: self.tok_len()})
            }
            TokenKind::Colon => {
                // Closure shorthand: `:expr` → `(__x): __x.expr`
                //   `:name`        → `(__x): __x.name`           (field access)
                //   `:method()`    → `(__x): __x.method()`       (method call)
                //   `:a.b`         → `(__x): __x.a.b`            (chained field)
                //   `:a.b()`       → `(__x): __x.a.b()`          (chained method)
                //   `:length > 3`  → `(__x): __x.length > 3`     (binary op continuation)
                //   `:name == "x"` → `(__x): __x.name == "x"`
                self.advance(); // consume ':'
                let param = Param {
                    name: "__x".to_string(),
                    ty: None, mutable: false, rebindable: false, owned: false, variadic: false,
                    default: None, line, col: 0,
                };
                let base = Expr { kind: ExprKind::Var("__x".to_string()), line, col, len: self.tok_len()};
                // First member: ident [ ( args ) ]
                let member = self.expect_ident()?;
                let mut acc = if self.check(&TokenKind::LParen) {
                    let args = self.parse_call_args()?;
                    Expr { kind: ExprKind::MethodCall(Box::new(base), member, args), line, col, len: self.tok_len()}
                } else {
                    Expr { kind: ExprKind::Field(Box::new(base), member), line, col, len: self.tok_len()}
                };
                // Optional trailing .field / .method() chain
                while self.check(&TokenKind::Dot) {
                    self.advance(); // consume '.'
                    let next = self.expect_ident()?;
                    acc = if self.check(&TokenKind::LParen) {
                        let args = self.parse_call_args()?;
                        Expr { kind: ExprKind::MethodCall(Box::new(acc), next, args), line, col, len: self.tok_len()}
                    } else {
                        Expr { kind: ExprKind::Field(Box::new(acc), next), line, col, len: self.tok_len()}
                    };
                }
                // Optional binary-operator continuation: `:length > 3`, `:count != 0`, etc.
                let body_expr = {
                    let op = match self.peek() {
                        TokenKind::EqEqEq => Some(BinOp::RefEq),
                        TokenKind::EqEq   => Some(BinOp::Eq),
                        TokenKind::BangEq => Some(BinOp::NotEq),
                        TokenKind::LtEq   => Some(BinOp::LtEq),
                        TokenKind::GtEq   => Some(BinOp::GtEq),
                        TokenKind::Lt if !self.check2(&TokenKind::Lt) => Some(BinOp::Lt),
                        TokenKind::Gt if !self.check2(&TokenKind::Gt) => Some(BinOp::Gt),
                        TokenKind::Plus    => Some(BinOp::Add),
                        TokenKind::Minus   => Some(BinOp::Sub),
                        TokenKind::Star    => Some(BinOp::Mul),
                        TokenKind::Slash   => Some(BinOp::Div),
                        TokenKind::Percent => Some(BinOp::Rem),
                        _ => None,
                    };
                    if let Some(op) = op {
                        self.advance(); // consume operator
                        let rhs = self.parse_or()?;
                        Expr { kind: ExprKind::BinOp(op, Box::new(acc), Box::new(rhs)), line, col, len: self.tok_len()}
                    } else {
                        acc
                    }
                };
                let body = ClosureBody::Expr(Box::new(body_expr));
                let (throws, task) = Self::infer_throws_task(&body);
                Ok(Expr { kind: ExprKind::Closure(vec![param], None, body, throws, task), line, col, len: self.tok_len()})
            }
            TokenKind::Dot => {
                // Dot-prefix enum shorthand: `.Red`
                self.advance();
                let name = self.expect_ident()?;
                Ok(Expr { kind: ExprKind::DotIdent(name), line, col, len: self.tok_len()})
            }
            TokenKind::LParen => {
                self.advance();
                self.skip_newlines_and_indent(); // allow `(\n    param,` multi-line form
                if self.check(&TokenKind::RParen) {
                    // Empty tuple OR empty-param closure `(): expr`
                    self.advance();
                    let (ex_throws, ex_task) = self.parse_closure_modifiers();
                    if self.check(&TokenKind::Colon) {
                        self.advance();
                        let body = self.parse_closure_body()?;
                        let (inf_throws, inf_task) = Self::infer_throws_task(&body);
                        return Ok(Expr { kind: ExprKind::Closure(vec![], None, body, ex_throws | inf_throws, ex_task | inf_task), line, col, len: self.tok_len()});
                    }
                    return Ok(Expr { kind: ExprKind::Tuple(vec![]), line, col, len: self.tok_len()});
                }
                // Detect typed closure params: `(Type name, ...)` or `(var Type name, ...)`
                // Heuristic: if first token is type-start AND second is an ident → typed params
                if self.peek_is_typed_closure() {
                    let mut params = vec![self.parse_param()?];
                    while self.eat(&TokenKind::Comma) {
                        self.skip_newlines_and_indent(); // allow newline between params
                        if self.check(&TokenKind::RParen) { break; }
                        params.push(self.parse_param()?);
                    }
                    self.skip_newlines_and_indent(); // allow newline before `)`
                    self.expect(&TokenKind::RParen)?;
                    let (ex_throws, ex_task) = self.parse_closure_modifiers();
                    self.expect(&TokenKind::Colon)?;
                    let body = self.parse_closure_body()?;
                    let (inf_throws, inf_task) = Self::infer_throws_task(&body);
                    return Ok(Expr { kind: ExprKind::Closure(params, None, body, ex_throws | inf_throws, ex_task | inf_task), line, col, len: self.tok_len()});
                }
                // Untyped: parse as expression(s), then decide closure vs tuple vs grouping
                let expr = self.parse_expr()?;
                if self.check(&TokenKind::Comma) {
                    let mut elems = vec![expr];
                    while self.eat(&TokenKind::Comma) {
                        self.skip_newlines_and_indent(); // allow newline between params
                        if self.check(&TokenKind::RParen) { break; }
                        elems.push(self.parse_expr()?);
                    }
                    self.skip_newlines_and_indent(); // allow newline before `)`
                    self.expect(&TokenKind::RParen)?;
                    let (ex_throws, ex_task) = self.parse_closure_modifiers();
                    if self.allow_trailing_closure && self.check(&TokenKind::Colon) {
                        self.advance();
                        let params = elems.iter().map(|e| expr_to_param(e, line, col)).collect::<Vec<_>>();
                        let body = self.parse_closure_body()?;
                        let (inf_throws, inf_task) = Self::infer_throws_task(&body);
                        return Ok(Expr { kind: ExprKind::Closure(params, None, body, ex_throws | inf_throws, ex_task | inf_task), line, col, len: self.tok_len()});
                    }
                    Ok(Expr { kind: ExprKind::Tuple(elems), line, col, len: self.tok_len()})
                } else {
                    self.skip_newlines_and_indent(); // allow newline before `)`
                    self.expect(&TokenKind::RParen)?;
                    let (ex_throws, ex_task) = self.parse_closure_modifiers();
                    if self.allow_trailing_closure && self.check(&TokenKind::Colon) {
                        self.advance();
                        let params = vec![expr_to_param(&expr, line, col)];
                        let body = self.parse_closure_body()?;
                        let (inf_throws, inf_task) = Self::infer_throws_task(&body);
                        return Ok(Expr { kind: ExprKind::Closure(params, None, body, ex_throws | inf_throws, ex_task | inf_task), line, col, len: self.tok_len()});
                    }
                    Ok(expr)
                }
            }
            TokenKind::LBracket => {
                self.advance();
                self.skip_newlines_and_indent();
                // `[..n]` — allocate array of length n without initialization
                if self.check(&TokenKind::DotDot) {
                    return self.parse_array_alloc(line, col);
                }
                // Comprehension forms: `[v for ..n]` or `[f(i) for i in ..n]`
                if !self.check(&TokenKind::RBracket) && !self.check(&TokenKind::Eof) {
                    // Parse first expression (value or computed expr)
                    let first = self.parse_expr()?;
                    if self.eat(&TokenKind::For) {
                        // `[value for width = w, height = h]` — labeled shape-only fill
                        // (docs/array-multidim-proposal.md). Checked FIRST, before the
                        // `IDENT in ...` comprehension case below: distinguished from it
                        // by `=` immediately following the identifier instead of `in`.
                        // The labels are NOT bound as usable variables in `value` — purely
                        // descriptive of shape, unlike the bound chained-for comprehension.
                        // Desugars identically to `[value for width in ..w for height in
                        // ..h]` by construction (same `LabeledArrayComp` node, using the
                        // label text directly as the loop variable name) — keeps this a
                        // pure parser-level convenience, no new desugar/interpreter/
                        // transpiler work: a name that's never referenced in `value` is
                        // simply never looked up.
                        if matches!(self.peek(), TokenKind::Ident(_)) && self.check2(&TokenKind::Eq) {
                            let mut clauses = Vec::new();
                            loop {
                                let label = self.expect_ident()?;
                                self.expect(&TokenKind::Eq)?;
                                let count = Box::new(self.parse_or()?);
                                clauses.push((label, count));
                                if !self.eat(&TokenKind::Comma) { break; }
                                self.skip_newlines_and_indent();
                            }
                            self.skip_newlines_and_indent();
                            self.expect(&TokenKind::RBracket)?;
                            let kind = ExprKind::LabeledArrayComp { expr: Box::new(first), clauses };
                            return Ok(Expr { kind, line, col, len: self.tok_len() });
                        }
                        // `[v for i in ..n]` or `[f(x) for x in collection]` — computed form
                        let kind = if matches!(self.peek(), TokenKind::Ident(_)) && self.check2(&TokenKind::In) {
                            let var = self.expect_ident()?;
                            self.expect(&TokenKind::In)?;
                            if self.check(&TokenKind::DotDot) {
                                // `[f(i) for i in ..n]` — range form
                                let count = self.parse_comprehension_count(line, col)?;
                                ExprKind::ArrayComp { expr: Box::new(first), var, count: Box::new(count) }
                            } else {
                                // Parse the source — could be `0..n` (range) or a collection expr
                                let source = self.parse_or()?;
                                match source.kind {
                                    ExprKind::Range { ref start, ref end, inclusive: false }
                                        if matches!(start.kind, ExprKind::Int(0)) =>
                                    {
                                        // `[f(i) for i in 0..n]` — treat as range form
                                        ExprKind::ArrayComp { expr: Box::new(first), var, count: end.clone() }
                                    }
                                    ExprKind::Range { inclusive: true, .. } => {
                                        return Err(ParseError::Generic {
                                            msg: "array comprehension does not accept inclusive range (`..=`)".to_string(),
                                            line, col, len: self.tok_len(),
                                        });
                                    }
                                    ExprKind::Range { .. } => {
                                        return Err(ParseError::Generic {
                                            msg: "array comprehension range must start at 0 — use `..n` or `0..n`".to_string(),
                                            line, col, len: self.tok_len(),
                                        });
                                    }
                                    _ => {
                                        // `[f(x) for x in collection]` — iter form
                                        ExprKind::ArrayCompIter { expr: Box::new(first), var, iter: Box::new(source) }
                                    }
                                }
                            }
                        } else {
                            // `[v for ..n]` or `[v for n]` — fill form. Unlike the bound
                            // comprehension above, a bare count with no `..`/`0..` wrapper
                            // is also accepted here — there's no loop variable to justify
                            // requiring explicit range syntax (see `parse_fill_count`'s doc).
                            let count = self.parse_fill_count(line, col)?;
                            ExprKind::ArrayFill { value: Box::new(first), count: Box::new(count) }
                        };
                        // Chained `for`: [f(w,h) for w in ..W for h in ..H] — a labeled
                        // multi-dim comprehension (docs/array-multidim-proposal.md). Only
                        // the range form (`ArrayComp`) chains; a collection-iteration or
                        // fill clause stays single-axis, unchanged. `clauses[0]` is axis 1
                        // (fastest-varying in storage) — declaration order, NOT syntactic
                        // nesting order (see LabeledArrayComp's doc comment, D2).
                        if let ExprKind::ArrayComp { expr: comp_expr, var: first_var, count: first_count } = kind {
                            let mut clauses = vec![(first_var, first_count)];
                            while self.eat(&TokenKind::For) {
                                let var = self.expect_ident()?;
                                self.expect(&TokenKind::In)?;
                                let count = Box::new(self.parse_comprehension_count(line, col)?);
                                clauses.push((var, count));
                            }
                            self.skip_newlines_and_indent();
                            self.expect(&TokenKind::RBracket)?;
                            let kind = if clauses.len() == 1 {
                                let (var, count) = clauses.into_iter().next().unwrap();
                                ExprKind::ArrayComp { expr: comp_expr, var, count }
                            } else {
                                ExprKind::LabeledArrayComp { expr: comp_expr, clauses }
                            };
                            return Ok(Expr { kind, line, col, len: self.tok_len() });
                        }
                        self.skip_newlines_and_indent();
                        self.expect(&TokenKind::RBracket)?;
                        return Ok(Expr { kind, line, col, len: self.tok_len()});
                    }
                    // Regular array literal — continue collecting elements
                    let mut elems = vec![first];
                    if self.eat(&TokenKind::Comma) {
                        self.skip_newlines_and_indent();
                        while !self.check(&TokenKind::RBracket) && !self.check(&TokenKind::Eof) {
                            elems.push(self.parse_expr()?);
                            if !self.eat(&TokenKind::Comma) { break; }
                            self.skip_newlines_and_indent();
                        }
                    }
                    self.skip_newlines_and_indent();
                    self.expect(&TokenKind::RBracket)?;
                    return Ok(Expr { kind: ExprKind::Array(elems), line, col, len: self.tok_len()});
                }
                self.skip_newlines_and_indent();
                self.expect(&TokenKind::RBracket)?;
                Ok(Expr { kind: ExprKind::Array(vec![]), line, col, len: self.tok_len()})
            }
            TokenKind::LBrace => {
                self.parse_brace_expr(line, col)
            }
            TokenKind::If => {
                let if_stmt = self.parse_if_stmt()?;
                Ok(Expr { kind: ExprKind::If(Box::new(if_stmt)), line, col, len: self.tok_len()})
            }
            TokenKind::Match => {
                let match_stmt = self.parse_match_stmt()?;
                Ok(Expr { kind: ExprKind::Match(Box::new(match_stmt)), line, col, len: self.tok_len()})
            }
            TokenKind::Do => {
                // `do:` in expression context — always a scoped block, never do-while
                self.advance(); // consume 'do'
                self.expect(&TokenKind::Colon)?;
                self.expect_newline()?;
                let stmts = self.parse_block()?;
                Ok(Expr { kind: ExprKind::Do(stmts), line, col, len: self.tok_len()})
            }
            TokenKind::Loop => {
                // `loop:` as an expression — evaluates to the `break value`
                let s = self.parse_loop_stmt()?;
                Ok(Expr { kind: ExprKind::Loop(s), line, col, len: self.tok_len()})
            }
            TokenKind::Task => {
                self.parse_task_expr()
            }
            TokenKind::Join => {
                let line = self.line();
                let col = self.col();
                self.advance(); // consume `join`
                self.expect(&TokenKind::LParen)?;
                let mut exprs = Vec::new();
                while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                    exprs.push(self.parse_expr()?);
                    if !self.eat(&TokenKind::Comma) { break; }
                }
                self.expect(&TokenKind::RParen)?;
                Ok(Expr { kind: ExprKind::JoinAll(exprs), line, col, len: self.tok_len()})
            }
            // Soft keywords — also usable as function names / variable references in expressions.
            TokenKind::Get => {
                self.advance();
                Ok(Expr { kind: ExprKind::Var("get".to_string()), line, col, len: self.tok_len()})
            }
            TokenKind::Set => {
                self.advance();
                Ok(Expr { kind: ExprKind::Var("set".to_string()), line, col, len: self.tok_len()})
            }
            // `wait(dur)` — function-call form of the `wait dur` statement.
            // Allows passing `wait` as a callable and using it in closures.
            TokenKind::Wait => {
                self.advance();
                Ok(Expr { kind: ExprKind::Var("wait".to_string()), line, col, len: self.tok_len()})
            }
            _ => Err(ParseError::Generic {
                line, col,
                msg: format!("unexpected token in expression: {:?}", self.peek()), len: self.tok_len(),
            }),
        }
    }

    /// Parse the argument list of a macro call.
    /// Accepts any of `(...)`, `[...]`, `{...}` as delimiter.
    /// Content: comma-separated expressions (trailing comma allowed).
    pub(crate) fn parse_macro_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        let (open, close) = match self.peek().clone() {
            TokenKind::LParen   => (TokenKind::LParen,   TokenKind::RParen),
            TokenKind::LBracket => (TokenKind::LBracket, TokenKind::RBracket),
            TokenKind::LBrace   => (TokenKind::LBrace,   TokenKind::RBrace),
            other => return Err(ParseError::Generic {
                line: self.line(), col: self.col(),
                msg: format!("expected '(', '[', or '{{' after '!' in macro call, got {:?}", other), len: self.tok_len(),
            }),
        };
        self.expect(&open)?;
        self.skip_newlines_and_indent();
        let mut args = Vec::new();
        while !self.check(&close) && !self.check(&TokenKind::Eof) {
            args.push(self.parse_expr()?);
            if !self.eat(&TokenKind::Comma) { break; }
            self.skip_newlines_and_indent();
        }
        self.skip_newlines_and_indent();
        self.expect(&close)?;
        Ok(args)
    }

    /// Parse `{...}` — could be dict `{k: v, ...}` or set `{v, ...}`
    pub(crate) fn parse_brace_expr(&mut self, line: usize, col: usize) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LBrace)?;
        self.skip_newlines_and_indent(); // allow `{\n    ...` multi-line form
        if self.check(&TokenKind::RBrace) {
            self.advance();
            // {} = empty set; {=} = empty dict
            return Ok(Expr { kind: ExprKind::Set(vec![]), line, col, len: self.tok_len()});
        }

        // {=} = also empty dict (alternate syntax)
        if self.check(&TokenKind::Eq) {
            self.advance();
            self.expect(&TokenKind::RBrace)?;
            return Ok(Expr { kind: ExprKind::Dict(vec![]), line, col, len: self.tok_len()});
        }

        // Distinguish dict from set: dict uses `=` as key-value separator
        // Dict: {key = value, ...}   Set: {elem, ...}
        let first_expr = self.parse_expr()?;
        if self.eat(&TokenKind::Eq) {
            // Dict
            let first_val = self.parse_expr()?;
            let mut pairs = vec![(first_expr, first_val)];
            while self.eat(&TokenKind::Comma) {
                self.skip_newlines_and_indent(); // allow newline between pairs
                if self.check(&TokenKind::RBrace) { break; }
                let k = self.parse_expr()?;
                self.expect(&TokenKind::Eq)?;
                let v = self.parse_expr()?;
                pairs.push((k, v));
            }
            self.skip_newlines_and_indent(); // allow newline before `}`
            self.expect(&TokenKind::RBrace)?;
            Ok(Expr { kind: ExprKind::Dict(pairs), line, col, len: self.tok_len()})
        } else {
            // Set
            let mut elems = vec![first_expr];
            while self.eat(&TokenKind::Comma) {
                self.skip_newlines_and_indent(); // allow newline between elements
                if self.check(&TokenKind::RBrace) { break; }
                elems.push(self.parse_expr()?);
            }
            self.skip_newlines_and_indent(); // allow newline before `}`
            self.expect(&TokenKind::RBrace)?;
            Ok(Expr { kind: ExprKind::Set(elems), line, col, len: self.tok_len()})
        }
    }

    pub(crate) fn resolve_interp(&mut self, parts: Vec<RawInterpPart>, line: usize) -> Result<Vec<StringSegment>, ParseError> {
        let mut segments = Vec::new();
        for part in parts {
            match part {
                RawInterpPart::Lit(s) => segments.push(StringSegment::Lit(s)),
                RawInterpPart::Hole(code) => {
                    if code.trim().is_empty() {
                        segments.push(StringSegment::Lit("{}".to_string()));
                    } else {
                        let hole_tokens = lex(&code).map_err(ParseError::Lex)?;
                        let mut sub_parser = Parser::new(hole_tokens);
                        sub_parser.skip_newlines();
                        let expr = sub_parser.parse_expr().map_err(|e| ParseError::Generic { line, col: 0, msg: format!("in string interpolation: {}", e), len: 1 })?;
                        segments.push(StringSegment::Expr(Box::new(expr)));
                    }
                }
                RawInterpPart::HoleFormatted(code, fmt) => {
                    if code.trim().is_empty() {
                        segments.push(StringSegment::Lit(format!("{{:{}}}", fmt)));
                    } else {
                        let hole_tokens = lex(&code).map_err(ParseError::Lex)?;
                        let mut sub_parser = Parser::new(hole_tokens);
                        sub_parser.skip_newlines();
                        let expr = sub_parser.parse_expr().map_err(|e| ParseError::Generic { line, col: 0, msg: format!("in string interpolation: {}", e), len: 1 })?;
                        segments.push(StringSegment::FormattedExpr(Box::new(expr), fmt));
                    }
                }
            }
        }
        Ok(segments)
    }

    // Returns true if the current position looks like typed closure params:
    // `(Type name` or `(var Type name` — detect without consuming tokens.
    pub(crate) fn peek_is_typed_closure(&self) -> bool {
        // pos is just after the opening `(` was consumed
        let t0 = self.tokens.get(self.pos).map(|t| &t.kind);
        let t1 = self.tokens.get(self.pos + 1).map(|t| &t.kind);
        // `var` prefix
        let (type_tok, name_tok) = if matches!(t0, Some(TokenKind::Var)) {
            (self.tokens.get(self.pos + 1).map(|t| &t.kind),
             self.tokens.get(self.pos + 2).map(|t| &t.kind))
        } else {
            (t0, t1)
        };
        // type must be an uppercase-starting ident, a known alias, or a primitive keyword
        let type_is_named = match type_tok {
            Some(TokenKind::Ident(s)) => Self::is_type_name(s),
            Some(TokenKind::LBracket | TokenKind::LBrace) => true,
            _ => false,
        };
        let name_is_ident = matches!(name_tok, Some(TokenKind::Ident(_)));
        type_is_named && name_is_ident
    }

    /// Parse the body of a closure after the `:` has been consumed.
    pub(crate) fn parse_closure_body(&mut self) -> Result<ClosureBody, ParseError> {
        if self.is_newline() {
            self.expect_newline()?;
            let stmts = self.parse_block()?;
            check_no_return(&stmts, "closure")?;
            Ok(ClosureBody::Block(stmts))
        } else {
            // Inline expression — forbid `return`
            if self.check(&TokenKind::Return) {
                return Err(ParseError::Generic {
                    line: self.line(), col: self.col(),
                    msg: "last expression (no 'return' allowed in closure)".to_string(), len: self.tok_len(),
                });
            }
            let expr = self.parse_or()?;
            Ok(ClosureBody::Expr(Box::new(expr)))
        }
    }

    pub(crate) fn parse_task_expr(&mut self) -> Result<Expr, ParseError> {
        let line = self.line();
        let col = self.col();
        self.expect(&TokenKind::Task)?;

        // ── task(duration): body  OR  task(timeout = duration): body ─────────
        // Detected when task is immediately followed by `(` without a `:` first.
        if self.check(&TokenKind::LParen) {
            self.advance(); // consume `(`
            // Accept both labeled `timeout = expr` and bare `expr`
            let dur_expr = if self.check(&TokenKind::Ident("timeout".to_string()))
                && self.check2(&TokenKind::Eq)
            {
                self.advance(); // consume `timeout`
                self.advance(); // consume `=`
                self.parse_expr()?
            } else {
                self.parse_expr()?
            };
            self.expect(&TokenKind::RParen)?;
            // Now parse the body — same `:` + block/inline logic as plain task
            self.eat(&TokenKind::Colon);
            let body = if self.is_newline() || self.check(&TokenKind::Eof) {
                self.expect_newline()?;
                let stmts = self.parse_block()?;
                check_no_return(&stmts, "task block")?;
                Expr { kind: ExprKind::Block(stmts), line, col, len: self.tok_len()}
            } else {
                self.parse_or()?
            };
            return Ok(Expr {
                kind: ExprKind::TaskWithTimeout(Box::new(dur_expr), Box::new(body)),
                line, col, len: self.tok_len(), });
        }

        // Optionally consume ':'
        self.eat(&TokenKind::Colon);
        // If next token is Newline => parse block form
        let inner = if self.is_newline() {
            self.expect_newline()?;
            let stmts = self.parse_block()?;
            check_no_return(&stmts, "task block")?;
            Expr { kind: ExprKind::Block(stmts), line, col, len: self.tok_len()}
        } else {
            let expr = self.parse_or()?;
            // Command-style: `task print "..."` → Task(Call(print, ["..."]))
            if let ExprKind::Var(_) = &expr.kind {
                if self.peek_starts_expr() {
                    let arg_line = expr.line;
                    let arg = self.parse_expr()?;
                    Expr {
                        kind: ExprKind::Call(Box::new(expr), vec![Arg { label: None, value: arg , spread: false}]),
                        line: arg_line, col, len: self.tok_len(), }
                } else {
                    expr
                }
            } else {
                expr
            }
        };
        Ok(Expr { kind: ExprKind::Task(Box::new(inner)), line, col, len: self.tok_len()})
    }

    /// Parse the count expression in an array comprehension (`[v for ..n]` or `[f(i) for i in ..n]`).
    /// Accepts `..n` (sugar for `0..n`) and `0..n`. Rejects any non-zero start.
    #[inline(never)]
    fn parse_array_alloc(&mut self, line: usize, col: usize) -> Result<Expr, ParseError> {
        self.advance(); // consume `..`
        let count = self.parse_expr()?;
        self.skip_newlines_and_indent();
        self.expect(&TokenKind::RBracket)?;
        Ok(Expr { kind: ExprKind::ArrayAlloc { count: Box::new(count) }, line, col, len: self.tok_len() })
    }

    fn parse_comprehension_count(&mut self, line: usize, col: usize) -> Result<Expr, ParseError> {
        if self.eat(&TokenKind::DotDot) {
            // `..n` — implicit start 0
            return self.parse_or();
        }
        // Parse the full expression — `parse_or` will consume `0..n` as a Range node.
        let expr = self.parse_or()?;
        match expr.kind {
            ExprKind::Range { ref start, ref end, inclusive: false } => {
                if matches!(start.kind, ExprKind::Int(0)) {
                    Ok(*end.clone())
                } else {
                    Err(ParseError::Generic {
                        msg: "array comprehension range must start at 0 — use `..n` or `0..n`".to_string(),
                        line, col, len: self.tok_len(),
                    })
                }
            }
            ExprKind::Range { inclusive: true, .. } => Err(ParseError::Generic {
                msg: "array comprehension does not accept inclusive range (`..=`)".to_string(),
                line, col, len: self.tok_len(),
            }),
            _ => Err(ParseError::Generic {
                msg: "expected `..n` or `0..n` in array comprehension".to_string(),
                line, col, len: self.tok_len(),
            }),
        }
    }

    /// Count for the two "no bound variable" fill forms
    /// (docs/array-multidim-proposal.md): `[value for n]` and, via the
    /// labeled shape branch above, `label = n`. Mirrors
    /// `parse_comprehension_count` (`..n` / `0..n`) but additionally accepts
    /// a bare expression with no range wrapper at all — unlike the bound
    /// comprehension forms, there's no loop variable here to justify
    /// requiring explicit range syntax; `[0.0 for n]` and `[0.0 for ..n]`
    /// both mean "fill n elements with 0.0".
    fn parse_fill_count(&mut self, line: usize, col: usize) -> Result<Expr, ParseError> {
        if self.eat(&TokenKind::DotDot) {
            return self.parse_or();
        }
        let expr = self.parse_or()?;
        match expr.kind {
            ExprKind::Range { ref start, ref end, inclusive: false } => {
                if matches!(start.kind, ExprKind::Int(0)) {
                    Ok(*end.clone())
                } else {
                    Err(ParseError::Generic {
                        msg: "array fill range must start at 0 — use `..n`, `0..n`, or a bare `n`".to_string(),
                        line, col, len: self.tok_len(),
                    })
                }
            }
            ExprKind::Range { inclusive: true, .. } => Err(ParseError::Generic {
                msg: "array fill does not accept inclusive range (`..=`)".to_string(),
                line, col, len: self.tok_len(),
            }),
            _ => Ok(expr),
        }
    }
}
