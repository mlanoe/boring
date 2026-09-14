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

//! Parses the **trailing array-block sugar** — see docs/book.md, "Trailing
//! array-block sugar", and `ExprKind::TrailingArrayBlock`'s own doc comment.
//!
//! ## The grammar-collision this works around
//!
//! Boring already has a *no-paren closure* shorthand (`x: body`, parsed in
//! `parse_expr.rs`'s `parse_primary`): a bare identifier immediately followed
//! by `:` is a single-param closure literal whose parameter is that
//! identifier. That rule's grammar — `Ident Colon <body>` — is *exactly* the
//! shape the trailing array-block sugar wants for its bare (no-parens) form
//! (`Column:` followed by an indented block). The two features cannot both
//! claim the same token sequence by grammar alone.
//!
//! Earlier revisions of this feature resolved the collision by requiring the
//! callee to be spelled with an uppercase-led (constructor-style) name. That
//! was fragile on two counts: PascalCase-for-types is a convention this
//! compiler does not otherwise enforce as a semantic rule (nothing in
//! `checker`/`validator` rejects a lowercase struct name or an uppercase
//! function name — see `is_type_name`'s own doc comment: "looks like a type
//! name", a parser-only spelling heuristic, not a checked property), and it
//! wrongly forced *every* uppercase-led `Ident: <block>` into the array-block
//! interpretation even when that identifier's real last parameter was
//! `Fn(...)`-typed (a legitimate constructor wanting an ordinary trailing
//! closure), misrouting it into a compile error instead.
//!
//! The resolution used now: **parse a single, deliberately generic
//! `ExprKind::TrailingArrayBlock` node whenever the ambiguous shape appears —
//! regardless of the callee's spelling — and let `desugar_array_block`
//! decide what it means once it can look the callee up.** That pass has (and
//! needs) the very information the parser lacks: a signature table mapping
//! callee names to their declared last-parameter type. Concretely, the
//! block's body is parsed with the fully general statement grammar
//! (`Parser::parse_block`, the exact same routine any closure/function body
//! already uses), not a restricted "one collectible item per line" grammar —
//! reinterpreting a generic `Vec<Stmt>` after the fact (as a `[dyn Trait]`
//! collect-builder, an ordinary trailing closure, or a closure literal) is
//! cheap; re-parsing under a different grammar once resolution is known is
//! not an option (the tokens are already consumed).
//!
//! See `desugar_array_block`'s module doc comment for the full decision
//! table this now resolves to, and this crate's final report / commit
//! message for the full write-up of why a parse-time, spelling-based
//! disambiguation was rejected in favor of this scope-resolution approach.

use super::*;
use crate::ast::*;
use crate::lexer::TokenKind;

impl Parser {
    /// True when the current position is a bare (no-parens) trailing
    /// array-block head: an `Ident` directly followed by `:` then a real
    /// indented block (a same-line inline body is never this sugar — that
    /// shape stays the pre-existing closure-shorthand/collect-inapplicable
    /// meaning unconditionally, per docs/book.md's scoping rule 4).
    ///
    /// No longer gated on the callee's casing (see this module's doc
    /// comment) — every bare `Ident:` immediately followed by a newline and
    /// an indented block parses into the same provisional node; resolution
    /// happens later, in `desugar_array_block`.
    ///
    /// Gated on `allow_noparen_closure` exactly like the no-paren closure
    /// shorthand it defers to: that flag is already turned off while parsing
    /// a condition whose own trailing `:` belongs to the enclosing statement
    /// (`if`/`elif`/`while`/`for`/`guard ... else`) — without the same guard
    /// here, `if Foo:` would have this rule swallow the `if` statement's own
    /// colon-plus-block as if `Foo` were the array call's callee.
    pub(crate) fn peek_is_bare_block_array_call(&self) -> bool {
        self.allow_noparen_closure
            && matches!(self.tokens.get(self.pos).map(|t| &t.kind), Some(TokenKind::Ident(_)))
            && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Colon))
            && matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind), Some(TokenKind::Newline))
    }

    /// True when the current position is `(` and, skipping forward to its
    /// matching `)` (by paren-depth counting only — deliberately ignorant of
    /// what's inside, exactly like an ordinary call-argument list may span
    /// multiple lines), what follows is `:` then a newline. Used to detect
    /// the parenthesized-args form of the trailing array-block sugar
    /// (`Column(spacing = 8):`) before committing to parse the parens as
    /// either closure params or call args.
    ///
    /// Gated on `allow_trailing_closure` exactly like `peek_is_trailing_closure`
    /// (the pre-existing explicit-param trailing-closure detector this shares a
    /// call site with): that flag is already turned off while parsing a
    /// condition/subject whose own trailing `:` belongs to the enclosing
    /// statement (`if`/`elif`/`while`/`for`/`match`/`guard ... else`) — without
    /// the same guard here, e.g. `match parse(s):` (a `match` whose *subject* is
    /// a call, immediately followed by the `match` statement's own `:` and
    /// indented arms) would have this rule swallow the whole arm list as if
    /// `parse` were this sugar's callee and the arms were its array/closure
    /// body. This one call site used to be implicitly protected by the old
    /// uppercase-led casing gate (a condition/subject expression essentially
    /// never called an uppercase-led constructor in practice) — removing that
    /// gate (see this module's doc comment) means this needs its own, explicit
    /// guard instead.
    pub(crate) fn peek_is_block_array_call_after_parens(&self) -> bool {
        if !self.allow_trailing_closure {
            return false;
        }
        let mut depth = 0i32;
        let mut i = self.pos;
        loop {
            match self.tokens.get(i).map(|t| &t.kind) {
                Some(TokenKind::LParen) => { depth += 1; i += 1; }
                Some(TokenKind::RParen) => {
                    depth -= 1;
                    i += 1;
                    if depth == 0 { break; }
                }
                Some(TokenKind::Eof) | None => return false,
                Some(_) => { i += 1; }
            }
        }
        matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::Colon))
            && matches!(self.tokens.get(i + 1).map(|t| &t.kind), Some(TokenKind::Newline))
    }

    /// Parses the `:` + indented block tail of a trailing array-block call,
    /// given its already-parsed callee and (possibly empty) parenthesized
    /// argument list. The `:` itself must still be the current token (not
    /// yet consumed) when this is called. `has_parens` records whether an
    /// explicit (possibly empty) `(...)` argument list preceded the colon —
    /// see `ExprKind::TrailingArrayBlock`'s doc comment for why that matters
    /// to resolution later.
    ///
    /// The body is parsed with the fully general statement grammar
    /// (`parse_block`) — the same one any closure/function body already
    /// uses — deliberately not a restricted grammar, since this node's
    /// eventual meaning (collect into an array / ordinary trailing closure /
    /// closure literal) isn't known until `desugar_array_block` resolves the
    /// callee.
    pub(crate) fn parse_array_block_tail(
        &mut self,
        callee: Expr,
        args: Vec<Arg>,
        has_parens: bool,
        line: usize,
        col: usize,
    ) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::Colon)?;
        self.expect_newline()?;
        let body = self.parse_block()?;
        check_no_return(&body, "closure")?;
        // No chaining after a multiline trailing array-block — the exact
        // same parsing-ambiguity reason a multiline trailing closure can't
        // be chained either (see `parse_trailing_closure`). Any "modifier"
        // calls must be ordinary labeled arguments before the `:` instead.
        let mut i = self.pos;
        while i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Newline | TokenKind::Dedent) {
            i += 1;
        }
        if i < self.tokens.len() && self.tokens[i].kind == TokenKind::Dot {
            return Err(ParseError::Generic {
                line, col, len: self.tok_len(),
                msg: "trailing array-block cannot be chained — pass modifier arguments as \
                      labeled arguments before the colon instead, e.g. `Column(spacing = 8):`".into(),
            });
        }
        Ok(Expr {
            kind: ExprKind::TrailingArrayBlock { callee: Box::new(callee), args, has_parens, body },
            line, col, len: self.span_len(line, col),
        })
    }
}
