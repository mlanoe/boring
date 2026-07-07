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

mod parse_fn;
mod parse_stmt;
mod parse_expr;
mod parse_type;

use crate::ast::*;
use crate::lexer::{LexError, Token, TokenKind};
use thiserror::Error;

use parse_type::{check_no_return, expr_to_param, resolve_assoc_in_fn, resolve_assoc_in_sig};

/// Internal helper: result of parsing one `type …` member inside a struct body.
enum TypeMemberKind {
    Method(TypeMethod),
    Var(TypeVar),
}

/// Internal helper: abstract signature vs default implementation in a trait body.
enum Either<L, R> { Left(L), Right(R) }

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("line {line}:{col}: {msg}")]
    Generic { line: usize, col: usize, len: usize, msg: String },
    #[error("lex error: {0}")]
    Lex(#[from] LexError),
}

impl ParseError {
    pub fn line(&self) -> usize {
        match self {
            ParseError::Generic { line, .. } => *line,
            ParseError::Lex(e) => e.line(),
        }
    }

    pub fn col(&self) -> usize {
        match self {
            ParseError::Generic { col, .. } => *col,
            ParseError::Lex(_) => 0,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ParseError::Generic { len, .. } => *len,
            ParseError::Lex(_) => 1,
        }
    }

    pub fn msg(&self) -> String {
        match self {
            ParseError::Generic { msg, .. } => msg.clone(),
            ParseError::Lex(e) => e.msg(),
        }
    }
}

pub fn parse(tokens: Vec<Token>) -> Result<Program, ParseError> {
    let mut p = Parser::new(tokens);
    p.parse_program()
}

/// Maximum `not` chain depth. Each `not` creates ~15 Rust stack frames in the
/// recursive-descent parser; at 200 that's ~3000 frames (~6 MB in debug mode),
/// safely within the 8 MB thread stack set in main.rs.
const MAX_EXPR_DEPTH: usize = 200;

struct Parser {
    pub(crate) tokens: Vec<Token>,
    pub(crate) pos: usize,
    /// Recursion depth counter — incremented on every `parse_expr` entry,
    /// decremented on exit.  Prevents stack overflow from crafted inputs.
    pub(crate) depth: usize,
    /// When false, `Ident ":"` is NOT parsed as a no-paren closure.
    /// Must be disabled in condition/value positions (if, while, for, if let, guard, match)
    /// where `:` is a body separator rather than a closure intro.
    pub(crate) allow_noparen_closure: bool,
    /// Must be disabled in condition/iterable positions where `):`  would be
    /// misread as a trailing-closure intro instead of call + body separator.
    pub(crate) allow_trailing_closure: bool,
    /// When true, `expect_newline_soft` will not consume a `;` token, leaving it
    /// for `parse_inline_stmts` to use as a statement separator.
    pub(crate) in_inline_context: bool,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0, depth: 0, allow_noparen_closure: true, allow_trailing_closure: true, in_inline_context: false }
    }

    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn line(&self) -> usize {
        self.tokens[self.pos].line
    }

    fn col(&self) -> usize {
        self.tokens[self.pos].col
    }

    fn tok_len(&self) -> usize {
        self.tokens[self.pos].len
    }

    fn advance(&mut self) -> &Token {
        let t = &self.tokens[self.pos];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn check(&self, kind: &TokenKind) -> bool {
        self.peek() == kind
    }

    /// Look at the token after the current one (for two-token detection like `<<` and `>>`).
    fn check2(&self, kind: &TokenKind) -> bool {
        let next_pos = self.pos + 1;
        if next_pos < self.tokens.len() {
            &self.tokens[next_pos].kind == kind
        } else {
            false
        }
    }

    fn is_newline2(&self) -> bool {
        let next_pos = self.pos + 1;
        if next_pos < self.tokens.len() {
            matches!(self.tokens[next_pos].kind, TokenKind::Newline | TokenKind::Semicolon)
        } else {
            false
        }
    }

    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.peek() == kind {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<&Token, ParseError> {
        if self.peek() == kind {
            Ok(self.advance())
        } else {
            Err(ParseError::Generic {
                line: self.line(), col: self.col(),
                                msg: format!("expected {:?}, got {:?}", kind, self.peek()), len: self.tok_len(),
            })
        }
    }

    fn is_newline(&self) -> bool {
        matches!(self.peek(), TokenKind::Newline | TokenKind::Semicolon)
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), TokenKind::Newline | TokenKind::Semicolon | TokenKind::Comment(_)) {
            self.advance();
        }
    }

    /// Like `skip_newlines` but also skips INDENT/DEDENT tokens.
    /// Used inside `(…)` parameter lists to allow multi-line declarations.
    fn skip_newlines_and_indent(&mut self) {
        while matches!(self.peek(), TokenKind::Newline | TokenKind::Semicolon | TokenKind::Indent | TokenKind::Dedent) {
            self.advance();
        }
    }

    fn expect_newline(&mut self) -> Result<(), ParseError> {
        if self.is_newline() || self.check(&TokenKind::Eof) {
            self.skip_newlines();
            Ok(())
        } else {
            Err(ParseError::Generic {
                line: self.line(), col: self.col(),
                                msg: format!("expected newline, got {:?}", self.peek()), len: self.tok_len(),
            })
        }
    }

    /// Like expect_newline, but also succeeds when already past the newline
    /// (e.g. after parsing a block-form expression that consumed trailing whitespace).
    fn expect_newline_soft(&mut self) {
        // In inline context (e.g. match arm inline body), don't consume `;` — it's
        // the statement separator that `parse_inline_stmts` needs to see.
        if self.in_inline_context && self.check(&TokenKind::Semicolon) {
            return;
        }
        if self.is_newline() || self.check(&TokenKind::Eof) {
            self.skip_newlines();
        }
        // else: newline was already consumed (e.g. by a block form) — just proceed
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        self.skip_newlines_and_indent(); // also skip Dedent at top level (from multi-line parens)
        while !self.check(&TokenKind::Eof) {
            let item = self.parse_item(false)?;
            items.push(item);
            self.skip_newlines_and_indent();
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self, is_pub: bool) -> Result<Item, ParseError> {
        match self.peek().clone() {
            TokenKind::Comment(text) => {
                self.advance();
                while self.is_newline() { self.advance(); }
                Ok(Item::Stmt(Stmt::Comment(text)))
            }
            TokenKind::Pub => {
                self.advance();
                self.parse_item(true)
            }
            TokenKind::At => {
                // Attributes: parse them and then dispatch to the annotated item
                let attrs = self.parse_attrs();
                match self.peek().clone() {
                    TokenKind::Def => Ok(Item::Fn(self.parse_fn_decl_with_attrs(is_pub, true, attrs)?)),
                    TokenKind::Req => Ok(Item::Fn(self.parse_fn_decl_with_attrs(is_pub, false, attrs)?)),
                    TokenKind::Struct => Ok(Item::Struct(self.parse_struct_decl_with_attrs(is_pub, attrs)?)),
                    TokenKind::Enum => Ok(Item::Enum(self.parse_enum_decl_with_attrs(is_pub, attrs)?)),
                    _ => {
                        // Attrs on unsupported items: parse item normally (attrs are discarded)
                        self.parse_item(is_pub)
                    }
                }
            }
            TokenKind::Use => {
                // `use Ident as Type`         → type alias
                // `use Ident<T> as Type`       → generic type alias
                // `use Ident.path`             → module import
                // Detect alias form: `use Ident [< ... >] as …`
                // Scan past optional `<…>` to find `As`.
                let is_alias = {
                    let mut i = self.pos + 2; // skip `use` and the identifier
                    // skip optional `<…>` type params
                    if matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::Lt)) {
                        i += 1;
                        let mut depth = 1usize;
                        while i < self.tokens.len() && depth > 0 {
                            match &self.tokens[i].kind {
                                TokenKind::Lt => { depth += 1; i += 1; }
                                TokenKind::Gt => { depth -= 1; i += 1; }
                                _ => { i += 1; }
                            }
                        }
                    }
                    matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::As))
                };
                if is_alias {
                    Ok(Item::Alias(self.parse_alias_decl()?))
                } else {
                    Ok(Item::Use(self.parse_use_decl()?))
                }
            }
            TokenKind::Def => Ok(Item::Fn(self.parse_fn_decl(is_pub, true)?)),
            TokenKind::Req => Ok(Item::Fn(self.parse_fn_decl(is_pub, false)?)),
            TokenKind::Task => {
                // `task def …` / `task req …` / `task RetType …` / `task name():` — function declaration
                match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
                    Some(TokenKind::Def) => Ok(Item::Fn(self.parse_fn_decl(is_pub, true)?)),
                    Some(TokenKind::Req) => Ok(Item::Fn(self.parse_fn_decl(is_pub, false)?)),
                    _ if self.is_task_fn_shorthand() => Ok(Item::Fn(self.parse_fn_decl(is_pub, true)?)),
                    // `task name():` — void shorthand, disambiguated by `:` after params
                    _ if self.is_task_void_fn_decl() => Ok(Item::Fn(self.parse_fn_decl(is_pub, true)?)),
                    _ => Ok(Item::Stmt(self.parse_stmt()?)),
                }
            }
            TokenKind::Stream => {
                // `stream def …` / `stream<N> def …` / `stream req …` / shorthand
                // Skip optional `<N>` to find the real next keyword.
                let lookahead = {
                    let next1 = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                    if matches!(next1, Some(TokenKind::Lt)) {
                        // stream<N> — skip `< Int >` (3 tokens)
                        self.tokens.get(self.pos + 4).map(|t| &t.kind)
                    } else {
                        next1
                    }
                };
                match lookahead {
                    Some(TokenKind::Def) => Ok(Item::Fn(self.parse_fn_decl(is_pub, true)?)),
                    Some(TokenKind::Req) => Ok(Item::Fn(self.parse_fn_decl(is_pub, false)?)),
                    _ if self.is_stream_fn_shorthand() => Ok(Item::Fn(self.parse_fn_decl(is_pub, true)?)),
                    _ => Ok(Item::Stmt(self.parse_stmt()?)),
                }
            }
            TokenKind::Kernel => {
                // `kernel Name:` → GPU kernel struct declaration.
                // `kernel Name(` / `kernel:` / `kernel expr` → kernel block — handled in parse_stmt.
                let next  = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                let next2 = self.tokens.get(self.pos + 2).map(|t| &t.kind);
                if matches!(next, Some(TokenKind::Ident(_)))
                    && matches!(next2, Some(TokenKind::Colon))
                {
                    Ok(Item::Kernel(self.parse_kernel_decl(is_pub)?))
                } else {
                    Ok(Item::Stmt(self.parse_stmt()?))
                }
            }
            TokenKind::Struct => Ok(Item::Struct(self.parse_struct_decl(is_pub)?)),
            TokenKind::Enum => Ok(Item::Enum(self.parse_enum_decl(is_pub)?)),
            TokenKind::Trait => Ok(Item::Trait(self.parse_trait_decl()?)),
            TokenKind::Ext   => Ok(Item::Ext(self.parse_ext_decl()?)),
            TokenKind::Mod   => Ok(Item::Mod(self.parse_mod_decl(is_pub)?)),
            TokenKind::Type  => {
                // `type Name as InnerType` — newtype wrapper declaration.
                // Distinguish from `type Name = T` (assoc type def) by checking for `as` at pos+2.
                let after_ident = self.tokens.get(self.pos + 2).map(|t| &t.kind);
                if matches!(after_ident, Some(TokenKind::As)) {
                    Ok(Item::Alias(self.parse_newtype_decl()?))
                } else {
                    Ok(Item::Stmt(self.parse_stmt()?))
                }
            }
            TokenKind::Static | TokenKind::Let | TokenKind::Mut | TokenKind::Var => {
                if self.is_let_destructure() {
                    let line = self.line();
                    let col = self.col();
                    let _is_static = self.eat(&TokenKind::Static);
                    let binding = match self.peek() {
                        TokenKind::Mut => BindingKind::Mut,
                        TokenKind::Var => BindingKind::Var,
                        _ => BindingKind::Let,
                    };
                    self.advance(); // consume let/mut/var
                    Ok(Item::Stmt(Stmt::LetDestructure(self.parse_let_destructure(binding, line, col)?)))
                } else {
                    Ok(Item::Let(self.parse_let_stmt_pub(is_pub)?))
                }
            }
            // Shorthand function declaration without `def`/`req`:
            //   `RetType name(params):` at the top level.
            //
            // Detects: [type-start-token] [ident] `(` … `)` … `:`
            // `req` is still explicit — bare `RetType name()` defaults to `def`.
            _ if self.is_fn_decl_shorthand() => Ok(Item::Fn(self.parse_fn_decl(is_pub, true)?)),
            _ => Ok(Item::Stmt(self.parse_stmt()?)),
        }
    }

    /// Returns true when the token stream looks like a shorthand function declaration:
    ///   `RetType funcName(` — a return type followed by a lowercase identifier and `(`
    ///
    /// This covers top-level bare functions without `def`/`req`:
    ///   `string greet(string name):`
    ///   `int add(int a, int b):`
    ///   `[string] words():`
    ///
    /// The heuristic: try to parse a type at the current position (speculatively),
    /// then check that what follows is an identifier (function name) and then `(`.
    /// On failure the position is reset — purely look-ahead, no side effects.
    fn is_fn_decl_shorthand(&self) -> bool {
        // Quick guard: the current token must look like the start of a type.
        let starts_type = match self.peek() {
            TokenKind::Ident(s) => Self::is_type_name(s),
            TokenKind::Void => true,
            TokenKind::LBracket | TokenKind::LBrace | TokenKind::LParen => true,
            // `<Trait>` impl shorthand: `<` Ident `>` followed by function name
            TokenKind::Lt => {
                matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Ident(_)))
                && matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind), Some(TokenKind::Gt))
            }
            _ => false,
        };
        if !starts_type { return false; }

        // Speculatively scan past the type to see what follows.
        // We mirror what try_parse_return_type_prefix does: scan forward over
        // a type expression (brackets, angle brackets, qualifiers, optionals).
        let mut i = self.pos;
        // Skip the type tokens by counting nesting depth.
        let n = self.tokens.len();
        let mut depth = 0i32;
        loop {
            if i >= n { return false; }
            match &self.tokens[i].kind {
                TokenKind::LParen | TokenKind::LBracket | TokenKind::LBrace | TokenKind::Lt => {
                    depth += 1; i += 1;
                }
                TokenKind::RParen | TokenKind::RBracket | TokenKind::RBrace | TokenKind::Gt => {
                    depth -= 1; i += 1;
                    if depth < 0 { return false; }
                }
                TokenKind::Tick => {
                    // ownership qualifier 'xxx
                    i += 1;
                    if i < n && matches!(&self.tokens[i].kind, TokenKind::Ident(_)) {
                        i += 1;
                    }
                }
                TokenKind::Question => { i += 1; } // optional T?
                TokenKind::Ident(_) if depth > 0 => { i += 1; } // inside brackets
                TokenKind::Ident(s) if depth == 0 => {
                    if Self::is_type_name(s) {
                        i += 1; // part of the type
                    } else {
                        // Lowercase non-type ident at depth 0 → this is the function name
                        // Check: function name followed by `(`
                        let next = self.tokens.get(i + 1).map(|t| &t.kind);
                        return matches!(next,
                            Some(TokenKind::LParen)
                            | Some(TokenKind::Lt)      // generic: name<T>(
                        );
                    }
                }
                TokenKind::Comma if depth > 0 => { i += 1; }
                _ if depth > 0 => { i += 1; }
                _ => {
                    // End of type at depth 0 — what follows?
                    let next = self.tokens.get(i).map(|t| &t.kind);
                    // If next is a lowercase ident followed by `(`, it's a fn name
                    if let Some(TokenKind::Ident(name)) = next {
                        if !Self::is_type_name(name) {
                            let after = self.tokens.get(i + 1).map(|t| &t.kind);
                            return matches!(after,
                                Some(TokenKind::LParen) | Some(TokenKind::Lt)
                            );
                        }
                    }
                    return false;
                }
            }
        }
    }

    fn parse_use_decl(&mut self) -> Result<UseDecl, ParseError> {
        let line = self.line();
        let col = self.col();
        self.expect(&TokenKind::Use)?;

        // Parse all dot-separated segments.
        let mut path = vec![self.expect_ident()?];
        while self.eat(&TokenKind::Dot) {
            if self.eat(&TokenKind::Star) {
                // use a.b.c.*  — glob import
                self.expect_newline()?;
                return Ok(UseDecl { path, glob: true, items: vec![], line, col });
            }
            path.push(self.expect_ident()?);
        }

        // Single-segment path (`use a`) — whole-module edge case.
        if path.len() == 1 {
            self.expect_newline()?;
            return Ok(UseDecl { path, glob: false, items: vec![], line, col });
        }

        // The last segment is always the first item.
        // use a.b.c.X     → path=["a","b","c"], items=["X"]
        // use a.b.c.X, Y  → path=["a","b","c"], items=["X","Y"]
        let first_item = path.pop().unwrap();
        let mut items = vec![first_item];
        while self.eat(&TokenKind::Comma) {
            items.push(self.expect_ident()?);
        }
        self.expect_newline()?;
        Ok(UseDecl { path, glob: false, items, line, col })
    }

    /// Returns true if `s` looks like a type name: uppercase-starting OR a known lowercase alias.
    fn is_type_name(s: &str) -> bool {
        s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            || matches!(s,
                "int" | "uint" | "float" | "bool" | "string" | "str"
                | "void" | "never"
                | "i8" | "i16" | "i32" | "i64"
                | "u8" | "u16" | "u32" | "u64" | "usize"
                | "isize" | "f32" | "f64"
            )
    }

    /// Returns true when `pos` is at `task` and the token after looks like the
    /// start of a return type (not `def`/`req`), indicating the shorthand form
    /// `task RetType name(` instead of the explicit `task def RetType name(`.
    fn is_task_fn_shorthand(&self) -> bool {
        if !matches!(self.peek(), TokenKind::Task) { return false; }
        match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
            Some(TokenKind::Def) | Some(TokenKind::Req) => false,
            Some(TokenKind::Ident(s)) => Self::is_type_name(s),
            // keyword types that can appear as return types
            Some(TokenKind::Void) => true,
            Some(TokenKind::LBracket) | Some(TokenKind::LBrace) => true, // [T] / {T} return type
            Some(TokenKind::LParen) => {
                // `task (T, U) f():` — tuple return type → function declaration.
                // `task(dur):` — timeout expression → NOT a function declaration.
                //
                // Disambiguate by scanning past the closing `)`:
                //   • followed by an identifier  → tuple return type + function name → fn decl
                //   • followed by `:` or newline → task-with-timeout expression
                let mut depth = 0usize;
                let mut i = self.pos + 1;
                while i < self.tokens.len() {
                    match &self.tokens[i].kind {
                        TokenKind::LParen => { depth += 1; i += 1; }
                        TokenKind::RParen => {
                            if depth == 1 {
                                // Peek at what follows the closing paren
                                let after = self.tokens.get(i + 1).map(|t| &t.kind);
                                return matches!(after,
                                    Some(TokenKind::Ident(_))
                                    | Some(TokenKind::Set)
                                    | Some(TokenKind::Wait)
                                    | Some(TokenKind::Join)
                                );
                            }
                            depth -= 1; i += 1;
                        }
                        TokenKind::Newline | TokenKind::Semicolon | TokenKind::Eof => break,
                        _ => { i += 1; }
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Returns true when `pos` is at `task` followed by a lowercase function name and
    /// a parameter list that is eventually followed by `:` (with optional `throws`).
    /// This disambiguates `task foo():` (void function declaration) from `task foo()`
    /// (async call expression) at top-level and mod-body context.
    fn is_task_void_fn_decl(&self) -> bool {
        if !matches!(self.peek(), TokenKind::Task) { return false; }
        // Next token must be a lowercase ident that is NOT a known type name
        let name_tok = self.tokens.get(self.pos + 1).map(|t| &t.kind);
        match name_tok {
            Some(TokenKind::Ident(s)) if !Self::is_type_name(s) => {}
            _ => return false,
        }
        // Scan forward past the `(...)` parameter list looking for `:` or `throws`
        let mut i = self.pos + 2; // skip `task` and name
        // skip optional type params `<...>`
        if matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::Lt)) {
            let mut depth = 0i32;
            while i < self.tokens.len() {
                match &self.tokens[i].kind {
                    TokenKind::Lt => { depth += 1; i += 1; }
                    TokenKind::Gt => { depth -= 1; i += 1; if depth <= 0 { break; } }
                    _ => { i += 1; }
                }
            }
        }
        // expect `(`
        if !matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::LParen)) { return false; }
        // scan past balanced `(...)` allowing nested parens
        let mut depth = 0i32;
        while i < self.tokens.len() {
            match &self.tokens[i].kind {
                TokenKind::LParen => { depth += 1; i += 1; }
                TokenKind::RParen => { depth -= 1; i += 1; if depth <= 0 { break; } }
                _ => { i += 1; }
            }
        }
        // after `)`: optional `throws T`, then must find `:`
        if matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::Throws)) {
            i += 1; // skip `throws`
            // skip optional type (ident or path)
            while i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Ident(_) | TokenKind::Dot) {
                i += 1;
            }
        }
        matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::Colon))
    }

    /// Returns true when `pos` is at `stream` and the next token is a return type
    /// (not `def`/`req`), indicating shorthand `stream RetType name(…):`.
    fn is_stream_fn_shorthand(&self) -> bool {
        if !matches!(self.peek(), TokenKind::Stream) { return false; }
        match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
            Some(TokenKind::Def) | Some(TokenKind::Req) => false,
            Some(TokenKind::Ident(s)) => Self::is_type_name(s),
            Some(TokenKind::Void) => true,
            Some(TokenKind::LBracket) | Some(TokenKind::LBrace) => true,
            Some(TokenKind::LParen) => true,
            _ => false,
        }
    }

    /// Returns true if the token at `lt_pos` starts a `<Types>(`  generic-call pattern.
    /// Scans forward matching `<` / `>` nesting; succeeds if the matching `>` is followed by `(`.
    fn is_generic_call_ahead(&self, lt_pos: usize) -> bool {
        let mut depth = 0i32;
        let mut i = lt_pos;
        while i < self.tokens.len() {
            match &self.tokens[i].kind {
                TokenKind::Lt => { depth += 1; i += 1; }
                TokenKind::Gt => {
                    depth -= 1;
                    if depth == 0 {
                        // The token right after the closing `>` must be `(`
                        return matches!(self.tokens.get(i + 1).map(|t| &t.kind), Some(TokenKind::LParen));
                    }
                    i += 1;
                }
                // Tokens that cannot appear inside a type argument list: bail out
                TokenKind::Newline | TokenKind::Semicolon | TokenKind::Indent | TokenKind::Dedent
                | TokenKind::Eq | TokenKind::EqEq | TokenKind::BangEq
                | TokenKind::LtEq | TokenKind::GtEq | TokenKind::PipeArrow => return false,
                _ => { i += 1; }
            }
        }
        false
    }

    /// Parse the optional error type after `throws`.
    ///
    /// Handles both simple names (`throws CalcError`) and module-qualified paths
    /// (`throws io.Error`, `throws std.io.Error`). The dots are converted to `::` so
    /// the transpiler emits valid Rust paths without any further transformation.
    fn parse_throws_type(&mut self) -> Result<Option<Type>, ParseError> {
        let s = match self.peek() {
            TokenKind::Ident(s) => s.clone(),
            _ => return Ok(None),
        };
        if Self::is_type_name(&s) {
            // Simple/primitive type name — use the normal type parser (handles generics etc.)
            let ty = self.parse_type()?;
            let ty = self.parse_type_qualifier(ty)?;
            return Ok(Some(ty));
        }
        // Module-qualified path: lowercase module segment followed by at least one dot.
        // e.g.  io.Error  →  io::Error
        //       std.io.Error  →  std::io::Error
        if self.check2(&TokenKind::Dot) {
            let mut segments = vec![self.expect_ident()?];
            while self.check(&TokenKind::Dot)
                && matches!(
                    self.tokens.get(self.pos + 1).map(|t| &t.kind),
                    Some(TokenKind::Ident(_))
                )
            {
                self.advance(); // consume '.'
                segments.push(self.expect_ident()?);
            }
            let path = segments.join("::");
            return Ok(Some(Type::Named(path)));
        }
        Ok(None)
    }

    /// Parse one type argument inside `<…>` at a *use site* (e.g. `Foo<&a, T, U as Clone>`).
    fn parse_generic_type_arg(&mut self) -> Result<Type, ParseError> {
        // Bare lifetime: `&a` where `a` is a single lowercase letter.
        if self.check(&TokenKind::Ampersand) {
            let next_is_lifetime = matches!(
                self.tokens.get(self.pos + 1).map(|t| &t.kind),
                Some(TokenKind::Ident(s))
                    if s.len() == 1
                        && s.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
            );
            if next_is_lifetime {
                self.advance(); // consume `&`
                let lt = self.expect_ident()?;
                return Ok(Type::Named(format!("'{}", lt)));
            }
        }
        // Const generic arg at use site: `uint N`, `int N`, `bool N`.
        // Encoded as `Type::Named("$N:usize")` for the transpiler.
        {
            let is_const_type = matches!(self.peek(), TokenKind::Ident(s) if matches!(s.as_str(), "uint" | "int" | "bool"));
            if is_const_type {
                let next_is_ident = matches!(
                    self.tokens.get(self.pos + 1).map(|t| &t.kind),
                    Some(TokenKind::Ident(_))
                );
                if next_is_ident {
                    let type_kw = if let TokenKind::Ident(s) = self.peek().clone() { self.advance(); s } else { unreachable!() };
                    let rust_ty = match type_kw.as_str() {
                        "uint" => "usize",
                        "int"  => "i64",
                        "bool" => "bool",
                        _      => "usize",
                    };
                    let name = self.expect_ident()?;
                    return Ok(Type::Named(format!("${}:{}", name, rust_ty)));
                }
            }
        }
        // Regular type (with optional ownership qualifier, `?`, etc.)
        let ty = self.parse_type()?;
        // Silently consume optional `as Trait + AnotherTrait …` constraint.
        if self.eat(&TokenKind::As) {
            if let TokenKind::Ident(_) = self.peek().clone() {
                self.advance(); // consume first trait name
                while self.eat(&TokenKind::Plus) {
                    if let TokenKind::Ident(_) = self.peek().clone() {
                        self.advance(); // consume additional trait names
                    } else {
                        break;
                    }
                }
            }
        }
        Ok(ty)
    }

    fn try_parse_return_type_prefix(&mut self) -> Option<Type> {
        let is_type_start = match self.peek() {
            TokenKind::Ident(s) => {
                Self::is_type_name(s)
            }
            // `void` keyword as return type: `def void foo():`
            TokenKind::Void => {
                let next_pos = self.pos + 1;
                next_pos < self.tokens.len()
                    && matches!(&self.tokens[next_pos].kind, TokenKind::Ident(_) | TokenKind::Set)
            }
            TokenKind::LParen => true,
            TokenKind::LBracket => true,
            TokenKind::LBrace => true,
            // `<Trait>` impl-shorthand: `<` Ident `>` followed by a param name
            TokenKind::Lt => {
                matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Ident(_)))
                && matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind), Some(TokenKind::Gt))
            }
            _ => false,
        };

        if is_type_start {
            let saved = self.pos;
            let line = self.line();
            let _col = self.col();
            // First attempt: full parse_type (handles `T?`, `[T]`, fn types, etc.).
            if let Ok(ty) = self.parse_type() {
                if let Ok(ty) = self.parse_type_qualifier(ty) {
                    if self.keyword_as_ident_str(self.peek()).is_some() || self.check(&TokenKind::LParen) {
                        return Some(ty);
                    }
                }
            }
            // Fallback: parse_type may have consumed `int ()` as a fn-type signature.
            // For the anonymous call operator `def int ():`, the `()` is the method
            // signature, not part of the return type. Re-try with parse_type_base so
            // that only `int` is consumed, leaving `(` for parse_params().
            self.pos = saved;
            if let Ok(ty) = self.parse_type_base(line) {
                if let Ok(ty) = self.parse_type_qualifier(ty) {
                    if self.keyword_as_ident_str(self.peek()).is_some() || self.check(&TokenKind::LParen) {
                        return Some(ty);
                    }
                }
            }
            self.pos = saved;
        }
        None
    }

    /// When `var` precedes a borrow type, upgrade the borrow to a mutable borrow.
    /// `var T&` → `BorrowMut`; `var T&auto` / `var T&task` are left unchanged
    /// (mutating an Rc/Arc borrow makes no sense).
    fn apply_var_to_borrow(ty: Type) -> Type {
        match ty {
            Type::Qualified(inner, OwnerQual::Borrow) =>
                Type::Qualified(inner, OwnerQual::BorrowMut),
            // `mut T?&` → &mut Option<T>
            Type::Qualified(inner, OwnerQual::BorrowOption) =>
                Type::Qualified(inner, OwnerQual::BorrowOptionMut),
            // `mut T&a` → &'a mut T  (encoded as Lifetime wrapping BorrowMut)
            Type::Qualified(inner, OwnerQual::Lifetime(lt)) =>
                Type::Qualified(
                    Box::new(Type::Qualified(inner, OwnerQual::BorrowMut)),
                    OwnerQual::Lifetime(lt),
                ),
            other => other,
        }
    }

    /// Returns true when the token at `name_pos` is the start of an associated type
    /// definition `Name = T` or `Name<params> = T` (GAT form).
    /// `name_pos` points to the `Ident` token that is the type name.
    fn is_assoc_type_def(&self, name_pos: usize) -> bool {
        // Must start with an identifier
        if !matches!(self.tokens.get(name_pos).map(|t| &t.kind), Some(TokenKind::Ident(_))) {
            return false;
        }
        let mut i = name_pos + 1;
        // Skip optional `<...>` GAT params
        if matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::Lt)) {
            i += 1;
            let mut depth = 1usize;
            while i < self.tokens.len() && depth > 0 {
                match &self.tokens[i].kind {
                    TokenKind::Lt => depth += 1,
                    TokenKind::Gt => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
        }
        // Must be followed by `=`
        matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::Eq))
    }

    /// True when the upcoming `let`/`var` statement is a destructuring.
    /// Triggers on: `let (` or `let ident ,` (bare tuple form).
    fn is_let_destructure(&self) -> bool {
        // skip optional `static`
        let base = self.pos + if matches!(self.peek(), TokenKind::Static) { 1 } else { 0 };
        // base points to let/var; base+1 is the first binding token
        let after_kw = base + 1;
        match self.tokens.get(after_kw).map(|t| &t.kind) {
            Some(TokenKind::LParen) => {
                let mut i = after_kw + 1;
                let mut depth = 1usize;
                while i < self.tokens.len() && depth > 0 {
                    match &self.tokens[i].kind {
                        TokenKind::LParen => { depth += 1; i += 1; }
                        TokenKind::RParen => { depth -= 1; i += 1; }
                        _ => { i += 1; }
                    }
                }
                !matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::Ident(_)))
            }
            Some(TokenKind::Ident(_)) => {
                let after_first = after_kw + 1;
                if matches!(self.tokens.get(after_first).map(|t| &t.kind), Some(TokenKind::Comma)) {
                    return true;
                }
                if matches!(self.tokens.get(after_first).map(|t| &t.kind), Some(TokenKind::Ident(_))) {
                    return matches!(self.tokens.get(after_first + 1).map(|t| &t.kind), Some(TokenKind::Comma));
                }
                false
            }
            Some(t) if self.kind_is_type_start(t) => {
                let mut i = after_kw + 1;
                // If the type starts with `[` or `{`, we're already inside the bracket — start at depth 1.
                let mut depth = match self.tokens.get(after_kw).map(|t| &t.kind) {
                    Some(TokenKind::LBracket | TokenKind::LBrace) => 1i32,
                    _ => 0i32,
                };
                while i < self.tokens.len() {
                    match &self.tokens[i].kind {
                        TokenKind::Lt => { depth += 1; i += 1; }
                        TokenKind::Gt => { depth -= 1; i += 1; if depth < 0 { break; } }
                        TokenKind::LBracket => { depth += 1; i += 1; }
                        TokenKind::RBracket => { depth -= 1; i += 1; if depth < 0 { break; } }
                        TokenKind::Question => { i += 1; break; }
                        TokenKind::Ident(_) if depth == 0 => {
                            return matches!(self.tokens.get(i + 1).map(|t| &t.kind), Some(TokenKind::Comma));
                        }
                        _ if depth == 0 => break,
                        _ => { i += 1; }
                    }
                }
                let _ = i;
                false
            }
            _ => false,
        }
    }

    fn is_type_start_before_ident(&self) -> bool {
        // Returns true if current token starts a type that is followed by an ident (param name).
        match self.peek() {
            TokenKind::Req | TokenKind::Def | TokenKind::Task => return true,
            TokenKind::Ident(s) => {
                if !Self::is_type_name(s) { return false; }
                let mut i = self.pos + 1;
                // skip optional generic type args: `Foo<T, U>`
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Lt) {
                    i += 1;
                    let mut depth = 1usize;
                    while i < self.tokens.len() && depth > 0 {
                        match &self.tokens[i].kind {
                            TokenKind::Lt => depth += 1,
                            TokenKind::Gt => depth -= 1,
                            _ => {}
                        }
                        i += 1;
                    }
                }
                // skip optional `.AssocType` — `LinkedList.Index name`
                if i < self.tokens.len()
                    && matches!(self.tokens[i].kind, TokenKind::Dot)
                    && matches!(self.tokens.get(i + 1).map(|t| &t.kind), Some(TokenKind::Ident(_)))
                {
                    i += 2;
                }
                // skip optional `?`
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Question) { i += 1; }
                // skip optional tick + qualifier keyword
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Tick) {
                    i += 1;
                    let mut qual_is_auto_or_shared = false;
                    if let Some(tok) = self.tokens.get(i) {
                        match &tok.kind {
                            TokenKind::Ident(q) if q == "auto"   => { i += 1; qual_is_auto_or_shared = true; }
                            TokenKind::Ident(q) if matches!(q.as_str(), "const" | "stack" | "heap") => { i += 1; }
                            // `'actor` may be followed by `'task` → `'actor'task`
                            TokenKind::Ident(q) if q == "actor" => {
                                i += 1;
                                if i < self.tokens.len()
                                    && matches!(self.tokens[i].kind, TokenKind::Tick)
                                    && matches!(self.tokens.get(i + 1).map(|t| &t.kind), Some(TokenKind::Task))
                                {
                                    i += 2; // consume `'task`
                                    qual_is_auto_or_shared = true;
                                }
                            }
                            // `'guard` may be followed by `'task` → `'guard'task`
                            TokenKind::Guard => {
                                i += 1;
                                qual_is_auto_or_shared = true;
                                if i < self.tokens.len()
                                    && matches!(self.tokens[i].kind, TokenKind::Tick)
                                    && matches!(self.tokens.get(i + 1).map(|t| &t.kind), Some(TokenKind::Task))
                                {
                                    i += 2;
                                }
                            }
                            TokenKind::Ident(q) if q == "weak" => { i += 1; }
                            TokenKind::Task => { i += 1; qual_is_auto_or_shared = true; }
                            TokenKind::New => { i += 1; }
                            _ => {}
                        }
                    }
                    if qual_is_auto_or_shared
                        && i < self.tokens.len()
                        && matches!(self.tokens[i].kind, TokenKind::Tick)
                        && matches!(self.tokens.get(i + 1).map(|t| &t.kind),
                                    Some(TokenKind::Ident(q)) if q == "weak")
                    {
                        i += 2;
                    }
                    if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Question) { i += 1; }
                    if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Ampersand) {
                        i += 1;
                        if let Some(t) = self.tokens.get(i) {
                            if let TokenKind::Ident(q) = &t.kind {
                                if q.len() == 1
                                    && q.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
                                    && matches!(self.tokens.get(i + 1).map(|t| &t.kind), Some(TokenKind::Ident(_)))
                                {
                                    i += 1;
                                }
                            }
                        }
                    }
                }
                // skip `&` qualifier in any form
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Ampersand) {
                    if let Some(tok) = self.tokens.get(i + 1) {
                        match &tok.kind {
                            TokenKind::Ident(q) if q == "auto"   => { i += 2; }
                            TokenKind::Task => { i += 2; }
                            TokenKind::Ident(q)
                                if q.len() == 1
                                    && q.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false) =>
                            {
                                if matches!(self.tokens.get(i + 2).map(|t| &t.kind), Some(TokenKind::Ident(_))) {
                                    i += 2;
                                } else {
                                    i += 1;
                                }
                            }
                            TokenKind::Ident(_) => { i += 1; }
                            _ => {}
                        }
                    }
                }
                // skip optional function-type suffix
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::LParen) {
                    i += 1;
                    let mut depth = 1usize;
                    while i < self.tokens.len() && depth > 0 {
                        match &self.tokens[i].kind {
                            TokenKind::LParen => depth += 1,
                            TokenKind::RParen => depth -= 1,
                            _ => {}
                        }
                        i += 1;
                    }
                    while i < self.tokens.len()
                        && matches!(self.tokens[i].kind, TokenKind::Throws | TokenKind::Task)
                    {
                        i += 1;
                    }
                }
                // skip optional `...`
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::DotDotDot) { i += 1; }
                i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Ident(_))
            }
            // Array type `[...]` or Set/Dict type `{...}` before a param name
            TokenKind::LBracket | TokenKind::LBrace => {
                let open = &self.peek().clone();
                let close = if matches!(open, TokenKind::LBracket) { &TokenKind::RBracket } else { &TokenKind::RBrace };
                let mut i = self.pos + 1;
                let mut depth = 1usize;
                while i < self.tokens.len() && depth > 0 {
                    if &self.tokens[i].kind == open  { depth += 1; }
                    if &self.tokens[i].kind == close { depth -= 1; }
                    i += 1;
                }
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Question) { i += 1; }
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Tick) {
                    i += 1;
                    let mut qual_is_auto_or_shared = false;
                    if let Some(tok) = self.tokens.get(i) {
                        match &tok.kind {
                            TokenKind::Ident(q) if q == "auto"   => { i += 1; qual_is_auto_or_shared = true; }
                            TokenKind::Ident(q) if matches!(q.as_str(), "const" | "stack" | "heap" | "actor") => { i += 1; }
                            TokenKind::Guard => { i += 1; qual_is_auto_or_shared = true; }
                            TokenKind::Ident(q) if q == "weak" => { i += 1; }
                            TokenKind::Task => { i += 1; qual_is_auto_or_shared = true; }
                            TokenKind::New => { i += 1; }
                            _ => {}
                        }
                    }
                    if qual_is_auto_or_shared
                        && i < self.tokens.len()
                        && matches!(self.tokens[i].kind, TokenKind::Tick)
                        && matches!(self.tokens.get(i + 1).map(|t| &t.kind),
                                    Some(TokenKind::Ident(q)) if q == "weak")
                    {
                        i += 2;
                    }
                    if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Question) { i += 1; }
                    if qual_is_auto_or_shared
                        && i < self.tokens.len()
                        && matches!(self.tokens[i].kind, TokenKind::Ampersand)
                    {
                        i += 1;
                    }
                }
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Ampersand) {
                    if let Some(tok) = self.tokens.get(i + 1) {
                        match &tok.kind {
                            TokenKind::Ident(q) if q == "auto"   => { i += 2; }
                            TokenKind::Task => { i += 2; }
                            TokenKind::Ident(q)
                                if q.len() == 1
                                    && q.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false) =>
                            { i += 2; }
                            TokenKind::Ident(_) => { i += 1; }
                            _ => {}
                        }
                    }
                }
                // skip optional function-type suffix
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::LParen) {
                    i += 1;
                    let mut depth = 1usize;
                    while i < self.tokens.len() && depth > 0 {
                        match &self.tokens[i].kind {
                            TokenKind::LParen => depth += 1,
                            TokenKind::RParen => depth -= 1,
                            _ => {}
                        }
                        i += 1;
                    }
                    while i < self.tokens.len()
                        && matches!(self.tokens[i].kind, TokenKind::Throws | TokenKind::Task)
                    {
                        i += 1;
                    }
                }
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::DotDotDot) { i += 1; }
                i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Ident(_))
            }
            // Tuple type `(T1, T2, ...)` before a param/variable name.
            TokenKind::LParen => {
                let mut i = self.pos + 1;
                let mut depth = 1usize;
                while i < self.tokens.len() && depth > 0 {
                    match &self.tokens[i].kind {
                        TokenKind::LParen => { depth += 1; i += 1; }
                        TokenKind::RParen => { depth -= 1; i += 1; }
                        _ => { i += 1; }
                    }
                }
                matches!(self.tokens.get(i).map(|t| &t.kind), Some(TokenKind::Ident(_)))
            }
            // `<Trait>` impl-shorthand: `< Ident >` where Ident is a PascalCase type name.
            // Must verify Gt follows, then an Ident (param name).
            TokenKind::Lt => {
                matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Ident(_)))
                && matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind), Some(TokenKind::Gt))
                && matches!(self.tokens.get(self.pos + 3).map(|t| &t.kind), Some(TokenKind::Ident(_)))
            }
            _ => false,
        }
    }

    /// Parse `<T>`, `<T as Trait>`, `<T as (Trait1, Trait2)>`, `<T, U as Comparable>`, etc.
    fn parse_type_params(&mut self) -> (Vec<String>, Vec<(String, String)>) {
        if !self.eat(&TokenKind::Lt) { return (vec![], vec![]); }
        let mut params = Vec::new();
        let mut where_clause = Vec::new();
        self.skip_newlines_and_indent();
        loop {
            if self.check(&TokenKind::Ampersand) {
                let next_is_lifetime = matches!(
                    self.tokens.get(self.pos + 1).map(|t| &t.kind),
                    Some(TokenKind::Ident(s))
                        if s.len() == 1
                            && s.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
                );
                if next_is_lifetime {
                    self.advance();
                    if let TokenKind::Ident(lt) = self.peek().clone() {
                        self.advance();
                        params.push(format!("'{}", lt));
                    }
                    if !self.eat(&TokenKind::Comma) { break; }
                    self.skip_newlines_and_indent();
                    continue;
                }
            }
            // Const generic parameter: `uint N`, `int N`, or `bool N`.
            // Encoded as `"$N:usize"` (etc.) in the params vec for the transpiler.
            {
                let is_const_type = matches!(self.peek(), TokenKind::Ident(s) if matches!(s.as_str(), "uint" | "int" | "bool"));
                if is_const_type {
                    let next_is_ident = matches!(
                        self.tokens.get(self.pos + 1).map(|t| &t.kind),
                        Some(TokenKind::Ident(_))
                    );
                    if next_is_ident {
                        let type_kw = if let TokenKind::Ident(s) = self.peek().clone() { self.advance(); s } else { unreachable!() };
                        let rust_ty = match type_kw.as_str() {
                            "uint"  => "usize",
                            "int"   => "i64",
                            "bool"  => "bool",
                            _       => "usize",
                        };
                        let param_name = if let TokenKind::Ident(s) = self.peek().clone() { self.advance(); s } else { unreachable!() };
                        params.push(format!("${}:{}", param_name, rust_ty));
                        if !self.eat(&TokenKind::Comma) { break; }
                        self.skip_newlines_and_indent();
                        continue;
                    }
                }
            }
            let name = match self.peek().clone() {
                TokenKind::Ident(s) if s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) => {
                    self.advance(); s
                }
                _ => break,
            };
            params.push(name.clone());

            if self.eat(&TokenKind::As) {
                if let TokenKind::Ident(trait_name) = self.peek().clone() {
                    self.advance();
                    where_clause.push((name.clone(), trait_name));
                    while self.eat(&TokenKind::Plus) {
                        if let TokenKind::Ident(trait_name) = self.peek().clone() {
                            self.advance();
                            where_clause.push((name.clone(), trait_name));
                        } else {
                            break;
                        }
                    }
                }
            }

            if !self.eat(&TokenKind::Comma) { break; }
            self.skip_newlines_and_indent();
        }
        self.skip_newlines_and_indent();
        let _ = self.expect(&TokenKind::Gt);
        (params, where_clause)
    }

    // ─── GPU kernel declaration ───────────────────────────────────────────────

    fn parse_kernel_decl(&mut self, is_pub: bool) -> Result<KernelDecl, ParseError> {
        let line = self.line();
        let col = self.col();
        self.expect(&TokenKind::Kernel)?;
        let name = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        self.expect_newline()?;
        self.expect(&TokenKind::Indent)?;

        let mut fields: Vec<KernelFieldDecl> = Vec::new();
        let mut inits: Vec<InitDecl> = Vec::new();
        let mut methods: Vec<FnDecl> = Vec::new();

        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof) {
                break;
            }
            match self.peek().clone() {
                TokenKind::Init => {
                    inits.push(self.parse_init_decl()?);
                }
                TokenKind::Def => {
                    methods.push(self.parse_fn_decl(false, true)?);
                }
                TokenKind::Req => {
                    methods.push(self.parse_fn_decl(false, false)?);
                }
                TokenKind::Let | TokenKind::Var => {
                    fields.push(self.parse_kernel_field()?);
                }
                TokenKind::Mut => {
                    fields.push(self.parse_kernel_field()?);
                }
                _ => break,
            }
        }
        self.eat(&TokenKind::Dedent);
        Ok(KernelDecl { name, is_pub, fields, inits, methods, line, col })
    }

    /// Parse a kernel field: `let [float]'unified input` or `mut [float]'shared tile`
    ///
    /// The `parse_type()` call consumes the tick `'` but leaves the qualifier ident
    /// unconsumed (because `"unified"` etc. are not in the standard qualifier table).
    /// We then pick up the remaining ident to determine the GPU qualifier.
    fn parse_kernel_field(&mut self) -> Result<KernelFieldDecl, ParseError> {
        let (line, col) = (self.line(), self.col());
        let binding = match self.peek().clone() {
            TokenKind::Let => { self.advance(); FieldBinding::Let }
            TokenKind::Mut => { self.advance(); FieldBinding::Mut }
            TokenKind::Var => { self.advance(); FieldBinding::Var }
            _ => return Err(ParseError::Generic {
                msg: "kernel field must start with let, mut, or var".into(),
                line, col, len: 1,
            }),
        };

        // Parse the type — e.g. `[float]'unified`.
        // parse_type() produces Qualified(inner, GpuUnified/GpuGlobal/...) for GPU qualifiers.
        let parsed_ty = self.parse_type()?;

        // Extract the GPU qualifier and base type.
        // Note: `'shared` maps to OwnerQual::Shared (Arc/Rc) in the global qualifier table;
        // in kernel context it means block SRAM — we accept both spellings.
        let (qual, base_ty) = match parsed_ty {
            Type::Qualified(inner, OwnerQual::GpuUnified | OwnerQual::GpuGlobal) if matches!(*inner, Type::ArrayN(_, _)) => {
                return Err(ParseError::Generic {
                    msg: "fixed-size arrays cannot use 'unified or 'global — the size is implicit from the init parameter; use '[T]'unified or '[T]'global instead".into(),
                    line, col, len: 1,
                });
            }
            Type::Qualified(inner, OwnerQual::GpuUnified) => (GpuQual::Unified, *inner),
            Type::Qualified(inner, OwnerQual::GpuGlobal)  => (GpuQual::Global,  *inner),
            Type::Qualified(inner, OwnerQual::GpuSync)    => (GpuQual::Sync,    *inner),
            Type::Qualified(inner, OwnerQual::Shared) => {
                return Err(ParseError::Generic {
                    msg: "'shared is not valid inside a kernel — use 'sync for block SRAM (auto-barrier) or 'global/'unified for device memory".into(),
                    line, col, len: 1,
                });
            }
            Type::Qualified(inner, OwnerQual::GpuLocal) => {
                // '[T]'local is not representable on GPU — unsized thread-local arrays don't exist.
                if matches!(*inner, Type::Array(_)) {
                    return Err(ParseError::Generic {
                        msg: "'local does not support dynamic arrays — use a fixed-size '[T, N]'local or choose 'unified/'global".into(),
                        line, col, len: 1,
                    });
                }
                (GpuQual::Local, *inner)
            }
            Type::Qualified(inner, OwnerQual::GpuConst) if matches!(*inner, Type::Array(_)) => {
                return Err(ParseError::Generic {
                    msg: "'const does not support dynamic arrays — use a fixed-size '[T, N]'const for lookup tables".into(),
                    line, col, len: 1,
                });
            }
            Type::Qualified(inner, OwnerQual::GpuConst)   => (GpuQual::Const,   *inner),
            Type::Qualified(inner, OwnerQual::GpuActorGlobal) => (GpuQual::ActorGlobal, *inner),
            Type::Qualified(inner, OwnerQual::GpuSurface) => {
                // `'surface` is only valid on `[uint]` — pixel buffer.
                if !matches!(*inner, Type::Named(ref n) if n == "uint") {
                    if !matches!(*inner, Type::Array(_)) {
                        return Err(ParseError::Generic {
                            msg: "'surface requires a '[uint]' element type — pixel buffers hold 32-bit RGBA values".into(),
                            line, col, len: 1,
                        });
                    }
                    // Allow [uint] (Type::Array(Box<Type::Named("uint")>)) too
                }
                (GpuQual::Surface, *inner)
            }
            // Unqualified scalar → infer from binding:
            //   let scalar  → 'const  (read-only constant cache)
            //   mut/var scalar → 'local (mutable thread-private register)
            unqualified if !matches!(unqualified, Type::Array(_) | Type::ArrayN(_, _)) => {
                let qual = match binding {
                    FieldBinding::Let => GpuQual::Const,
                    FieldBinding::Mut | FieldBinding::Var => GpuQual::Local,
                };
                (qual, unqualified)
            }
            // Unqualified fixed array [T, N]: infer from binding.
            //   let [T, N] → 'const (read-only lookup table in constant cache)
            //   mut/var [T, N] → 'local (thread-private stack array)
            // Dynamic [T] stays an error: 'unified vs 'global is a semantic choice.
            Type::ArrayN(_, _) => {
                let qual = match binding {
                    FieldBinding::Let => GpuQual::Const,
                    FieldBinding::Mut | FieldBinding::Var => GpuQual::Local,
                };
                (qual, parsed_ty)
            }
            _ => return Err(ParseError::Generic {
                msg: "kernel array field must have an explicit GPU memory qualifier ('unified, 'global, 'shared, or 'local)".into(),
                line, col, len: 1,
            }),
        };

        let name = self.expect_ident()?;
        self.expect_newline_soft();
        Ok(KernelFieldDecl { name, binding, qual, ty: base_ty, line, col })
    }

    /// Parse a `kernel(params) expr` launch expression.
    /// Called from expression parsing when `kernel` is followed by `(`.
    fn parse_struct_decl(&mut self, is_pub: bool) -> Result<StructDecl, ParseError> {
        self.parse_struct_decl_with_attrs(is_pub, vec![])
    }

    fn parse_struct_decl_with_attrs(&mut self, is_pub: bool, attrs: Vec<Attr>) -> Result<StructDecl, ParseError> {
        let line = self.line();
        let col = self.col();
        self.expect(&TokenKind::Struct)?;
        let name = self.expect_ident()?;
        let (type_params, where_clause) = self.parse_type_params();
        let mut protocols = Vec::new();
        if self.eat(&TokenKind::As) {
            let proto = self.expect_ident()?;
            protocols.push(proto);
            while self.eat(&TokenKind::Comma) {
                self.skip_newlines_and_indent();
                protocols.push(self.expect_ident()?);
            }
        }
        self.expect(&TokenKind::Colon)?;
        if self.check(&TokenKind::Pass) {
            self.advance();
            self.expect_newline_soft();
            return Ok(StructDecl { name, is_pub, is_native: false, protocols, fields: vec![], inits: vec![], methods: vec![], conversions: vec![], type_params, where_clause, setters: vec![], type_methods: vec![], type_vars: vec![], assoc_type_defs: vec![], attrs, line, col });
        }
        if self.check(&TokenKind::Native) {
            self.advance();
            self.expect_newline_soft();
            return Ok(StructDecl { name, is_pub, is_native: true, protocols, fields: vec![], inits: vec![], methods: vec![], conversions: vec![], type_params, where_clause, setters: vec![], type_methods: vec![], type_vars: vec![], assoc_type_defs: vec![], attrs, line, col });
        }
        self.expect_newline()?;
        self.expect(&TokenKind::Indent)?;
        self.parse_struct_body(name, is_pub, protocols, type_params, where_clause, attrs, line, col)
    }

    fn parse_struct_body(&mut self, name: String, is_pub: bool, protocols: Vec<String>, type_params: Vec<String>, where_clause: Vec<(String, String)>, attrs: Vec<Attr>, line: usize, col: usize) -> Result<StructDecl, ParseError> {

        let mut fields = Vec::new();
        let mut inits = Vec::new();
        let mut methods = Vec::new();
        let mut conversions = Vec::new();
        let mut setters = Vec::new();
        let mut type_methods = Vec::new();
        let mut type_vars = Vec::new();
        let mut assoc_type_defs = Vec::new();

        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof) {
                break;
            }
            match self.peek().clone() {
                TokenKind::Init => {
                    inits.push(self.parse_init_decl()?);
                }
                TokenKind::Def => {
                    let m = self.parse_fn_decl(false, true)?;
                    methods.push(m);
                }
                TokenKind::Req => {
                    let m = self.parse_fn_decl(false, false)?;
                    methods.push(m);
                }
                TokenKind::Task => {
                    match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
                        Some(TokenKind::Def) => methods.push(self.parse_fn_decl(false, true)?),
                        Some(TokenKind::Req) => methods.push(self.parse_fn_decl(false, false)?),
                        _ if self.is_task_fn_shorthand() => methods.push(self.parse_fn_decl(false, true)?),
                        // Void shorthand in body context: `task name(...)` — unambiguous (not an expression)
                        Some(TokenKind::Ident(_)) => methods.push(self.parse_fn_decl(false, true)?),
                        _ => break,
                    }
                }
                TokenKind::Set => {
                    setters.push(self.parse_set_decl(false)?);
                }
                TokenKind::Type => {
                    let is_pub_type = false;
                    // `type Name = T` — assoc type def
                    let after_type = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                    if matches!(after_type, Some(TokenKind::Ident(_)))
                        && self.is_assoc_type_def(self.pos + 1)
                    {
                        let def_line = self.line();
                        self.advance(); // consume `type`
                        let assoc_name = self.expect_ident()?;
                        if self.check(&TokenKind::Lt) { let _ = self.parse_type_params(); }
                        self.advance(); // consume `=`
                        let ty = self.parse_type()?;
                        self.expect_newline_soft();
                        assoc_type_defs.push(AssocTypeDef { name: assoc_name, ty, line: def_line , col: 0 });
                    } else {
                        match self.parse_type_member(is_pub_type)? {
                            TypeMemberKind::Method(m) => type_methods.push(m),
                            TypeMemberKind::Var(v) => type_vars.push(v),
                        }
                    }
                }
                TokenKind::Pub => {
                    let next = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                    if matches!(next, Some(TokenKind::Def)) {
                        self.advance();
                        methods.push(self.parse_fn_decl(true, true)?);
                    } else if matches!(next, Some(TokenKind::Req)) {
                        self.advance();
                        methods.push(self.parse_fn_decl(true, false)?);
                    } else if matches!(next, Some(TokenKind::Task)) {
                        self.advance(); // consume `pub`
                        match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
                            Some(TokenKind::Def) => methods.push(self.parse_fn_decl(true, true)?),
                            Some(TokenKind::Req) => methods.push(self.parse_fn_decl(true, false)?),
                            _ if self.is_task_fn_shorthand() => methods.push(self.parse_fn_decl(true, true)?),
                            Some(TokenKind::Ident(_)) => methods.push(self.parse_fn_decl(true, true)?),
                            _ => break,
                        }
                    } else if matches!(next, Some(TokenKind::Set)) {
                        self.advance();
                        setters.push(self.parse_set_decl(true)?);
                    } else if matches!(next, Some(TokenKind::As)) {
                        conversions.push(self.parse_as_decl()?);
                    } else if matches!(next, Some(TokenKind::Type)) {
                        self.advance(); // consume `pub`
                        match self.parse_type_member(true)? {
                            TypeMemberKind::Method(m) => type_methods.push(m),
                            TypeMemberKind::Var(v) => type_vars.push(v),
                        }
                    } else if matches!(next, Some(TokenKind::Init)) {
                        inits.push(self.parse_init_decl()?);
                    } else {
                        // Try to parse as field declaration (starts with `pub` already consumed above? No.)
                        // Actually pub is NOT consumed here yet — parse_field_decl handles the `pub` keyword
                        fields.push(self.parse_field_decl()?);
                    }
                }
                TokenKind::As => {
                    conversions.push(self.parse_as_decl()?);
                }
                _ => {
                    fields.push(self.parse_field_decl()?);
                }
            }
        }
        self.eat(&TokenKind::Dedent);
        Ok(StructDecl { name, is_pub, is_native: false, protocols, fields, inits, methods, conversions, type_params, where_clause, setters, type_methods, type_vars, assoc_type_defs, attrs, line, col })
    }

    fn parse_ext_decl(&mut self) -> Result<ExtDecl, ParseError> {
        let line = self.line();
        let col = self.col();
        self.expect(&TokenKind::Ext)?;
        let type_name = self.expect_ident()?;

        // Optional `<T as Bound, ...>` generic arguments
        let mut type_args: Vec<Type> = Vec::new();
        let mut type_params: Vec<String> = Vec::new();
        let mut where_clause: Vec<(String, String)> = Vec::new();
        if self.eat(&TokenKind::Lt) {
            loop {
                let is_type_param = matches!(self.peek(), TokenKind::Ident(n) if {
                    let first = n.chars().next().unwrap_or('a');
                    first.is_uppercase()
                }) && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::As));

                if is_type_param {
                    let param_name = self.expect_ident()?;
                    self.expect(&TokenKind::As)?;
                    let bound = self.expect_ident()?;
                    where_clause.push((param_name.clone(), bound));
                    while self.eat(&TokenKind::Plus) {
                        let extra_bound = self.expect_ident()?;
                        where_clause.push((param_name.clone(), extra_bound));
                    }
                    type_params.push(param_name.clone());
                    type_args.push(Type::TypeParam(param_name));
                } else {
                    let ty = self.parse_type()?;
                    type_args.push(ty);
                }

                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::Gt)?;
        }

        // Optional `as Trait1, Trait2` conformance list
        let mut traits = Vec::new();
        if self.eat(&TokenKind::As) {
            traits.push(self.expect_ident()?);
            while self.eat(&TokenKind::Comma) {
                self.skip_newlines_and_indent();
                traits.push(self.expect_ident()?);
            }
        }
        self.expect(&TokenKind::Colon)?;
        self.expect_newline()?;
        self.expect(&TokenKind::Indent)?;

        let mut methods = Vec::new();
        let mut setters = Vec::new();
        let mut conversions = Vec::new();
        let mut assoc_type_defs: Vec<AssocTypeDef> = Vec::new();

        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof) {
                break;
            }
            match self.peek().clone() {
                TokenKind::Def => methods.push(self.parse_fn_decl(false, true)?),
                TokenKind::Req => methods.push(self.parse_fn_decl(false, false)?),
                TokenKind::Task => {
                    match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
                        Some(TokenKind::Def) => methods.push(self.parse_fn_decl(false, true)?),
                        Some(TokenKind::Req) => methods.push(self.parse_fn_decl(false, false)?),
                        _ if self.is_task_fn_shorthand() => methods.push(self.parse_fn_decl(false, true)?),
                        Some(TokenKind::Ident(_)) => methods.push(self.parse_fn_decl(false, true)?),
                        _ => break,
                    }
                }
                TokenKind::Set => setters.push(self.parse_set_decl(false)?),
                TokenKind::Type => {
                    let after_type = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                    if matches!(after_type, Some(TokenKind::Ident(_)))
                        && self.is_assoc_type_def(self.pos + 1)
                    {
                        let def_line = self.line();
                        self.advance(); // consume `type`
                        let assoc_name = self.expect_ident()?;
                        if self.check(&TokenKind::Lt) {
                            let _ = self.parse_type_params();
                        }
                        self.advance(); // consume `=`
                        let ty = self.parse_type()?;
                        self.expect_newline_soft();
                        assoc_type_defs.push(AssocTypeDef { name: assoc_name, ty, line: def_line , col: 0 });
                    } else {
                        return Err(ParseError::Generic {
                            msg: "unexpected 'type' in ext body (only 'type Name = T' is allowed)".into(),
                            line: self.line(), col: self.col(), len: self.tok_len(),
                        });
                    }
                }
                TokenKind::Pub => {
                    let next = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                    if matches!(next, Some(TokenKind::Def)) {
                        self.advance();
                        methods.push(self.parse_fn_decl(true, true)?);
                    } else if matches!(next, Some(TokenKind::Req)) {
                        self.advance();
                        methods.push(self.parse_fn_decl(true, false)?);
                    } else if matches!(next, Some(TokenKind::Task)) {
                        self.advance(); // consume `pub`
                        match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
                            Some(TokenKind::Def) => methods.push(self.parse_fn_decl(true, true)?),
                            Some(TokenKind::Req) => methods.push(self.parse_fn_decl(true, false)?),
                            _ if self.is_task_fn_shorthand() => methods.push(self.parse_fn_decl(true, true)?),
                            _ => return Err(ParseError::Generic {
                                msg: "expected 'def', 'req', or return type after 'pub task' in ext body".into(),
                                line: self.line(), col: self.col(), len: self.tok_len(),
                            }),
                        }
                    } else if matches!(next, Some(TokenKind::Set)) {
                        self.advance();
                        setters.push(self.parse_set_decl(true)?);
                    } else if matches!(next, Some(TokenKind::As)) {
                        conversions.push(self.parse_as_decl()?);
                    } else {
                        return Err(ParseError::Generic {
                            msg: "expected 'def', 'req', 'task', 'set', or 'as' after 'pub' in ext body".into(),
                            line: self.line(), col: self.col(), len: self.tok_len(),
                        });
                    }
                }
                TokenKind::As => {
                    conversions.push(self.parse_as_decl()?);
                }
                _ => {
                    return Err(ParseError::Generic {
                        msg: format!("unexpected token in ext body: {:?}", self.peek()),
                        line: self.line(), col: self.col(), len: self.tok_len(),
                    });
                }
            }
        }
        self.eat(&TokenKind::Dedent);

        Ok(ExtDecl { type_name, type_args, type_params, where_clause, traits, methods, setters, conversions, assoc_type_defs, line, col })
    }

    fn parse_enum_decl(&mut self, is_pub: bool) -> Result<EnumDecl, ParseError> {
        self.parse_enum_decl_with_attrs(is_pub, vec![])
    }

    fn parse_enum_decl_with_attrs(&mut self, is_pub: bool, attrs: Vec<Attr>) -> Result<EnumDecl, ParseError> {
        let line = self.line();
        let col = self.col();
        self.expect(&TokenKind::Enum)?;
        let name = self.expect_ident()?;
        let (type_params, _) = self.parse_type_params();
        let mut protocols = Vec::new();
        if self.eat(&TokenKind::As) {
            protocols.push(self.expect_ident()?);
            while self.eat(&TokenKind::Comma) {
                self.skip_newlines_and_indent();
                protocols.push(self.expect_ident()?);
            }
        }
        self.expect(&TokenKind::Colon)?;
        if self.check(&TokenKind::Pass) {
            self.advance();
            self.expect_newline_soft();
            return Ok(EnumDecl { name, is_pub, is_native: false, type_params, protocols, variants: vec![], methods: vec![], setters: vec![], conversions: vec![], attrs, line, col });
        }
        if self.check(&TokenKind::Native) {
            self.advance();
            self.expect_newline_soft();
            return Ok(EnumDecl { name, is_pub, is_native: true, type_params, protocols, variants: vec![], methods: vec![], setters: vec![], conversions: vec![], attrs, line, col });
        }
        self.expect_newline()?;
        self.expect(&TokenKind::Indent)?;
        let mut variants = Vec::new();
        let mut methods = Vec::new();
        let mut setters = Vec::new();
        let mut conversions = Vec::new();
        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof) { break; }
            match self.peek().clone() {
                TokenKind::Def => {
                    methods.push(self.parse_fn_decl(false, true)?);
                }
                TokenKind::Req => {
                    methods.push(self.parse_fn_decl(false, false)?);
                }
                TokenKind::Task => {
                    match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
                        Some(TokenKind::Def) => methods.push(self.parse_fn_decl(false, true)?),
                        Some(TokenKind::Req) => methods.push(self.parse_fn_decl(false, false)?),
                        _ if self.is_task_fn_shorthand() => methods.push(self.parse_fn_decl(false, true)?),
                        // Void shorthand in body context: `task name(...)` — unambiguous
                        Some(TokenKind::Ident(_)) => methods.push(self.parse_fn_decl(false, true)?),
                        _ => break,
                    }
                }
                TokenKind::Set => {
                    setters.push(self.parse_set_decl(false)?);
                }
                TokenKind::Pub => {
                    let next = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                    if matches!(next, Some(TokenKind::Def)) {
                        self.advance(); // consume `pub`
                        methods.push(self.parse_fn_decl(true, true)?);
                    } else if matches!(next, Some(TokenKind::Req)) {
                        self.advance(); // consume `pub`
                        methods.push(self.parse_fn_decl(true, false)?);
                    } else if matches!(next, Some(TokenKind::Task)) {
                        self.advance(); // consume `pub`
                        match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
                            Some(TokenKind::Def) => methods.push(self.parse_fn_decl(true, true)?),
                            Some(TokenKind::Req) => methods.push(self.parse_fn_decl(true, false)?),
                            _ if self.is_task_fn_shorthand() => methods.push(self.parse_fn_decl(true, true)?),
                            // Void shorthand in body context: `task name(...)` — unambiguous
                            Some(TokenKind::Ident(_)) => methods.push(self.parse_fn_decl(true, true)?),
                            _ => break,
                        }
                    } else if matches!(next, Some(TokenKind::Set)) {
                        self.advance(); // consume `pub`
                        setters.push(self.parse_set_decl(true)?);
                    } else if matches!(next, Some(TokenKind::As)) {
                        // parse_as_decl eats `pub` itself
                        conversions.push(self.parse_as_decl()?);
                    } else {
                        return Err(ParseError::Generic {
                            msg: "expected 'def', 'req', 'set', or 'as' after 'pub' in enum body".into(),
                            line: self.line(), col: self.col(), len: self.tok_len(),
                        });
                    }
                }
                TokenKind::As => {
                    conversions.push(self.parse_as_decl()?);
                }
                _ => {
                    variants.push(self.parse_enum_variant()?);
                }
            }
        }
        self.eat(&TokenKind::Dedent);
        Ok(EnumDecl { name, is_pub, is_native: false, type_params, protocols, variants, methods, setters, conversions, attrs, line, col })
    }

    fn parse_mod_decl(&mut self, _is_pub: bool) -> Result<ModDecl, ParseError> {
        let line = self.line();
        let col = self.col();
        self.expect(&TokenKind::Mod)?;
        let name = self.expect_ident()?;
        self.expect(&TokenKind::Colon)?;
        self.expect_newline()?;
        self.expect(&TokenKind::Indent)?;
        let mut items = Vec::new();
        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof) { break; }
            items.push(self.parse_item(false)?);
        }
        self.eat(&TokenKind::Dedent);
        Ok(ModDecl { name, items, line, col })
    }

    fn parse_enum_variant(&mut self) -> Result<EnumVariant, ParseError> {
        let line = self.line();
        let col = self.col();
        // Optional per-variant attributes: `@error("message")`, `@doc("...")`, etc.
        let attrs = if self.check(&TokenKind::At) { self.parse_attrs() } else { vec![] };
        let name = self.expect_ident_or_keyword()?;
        let mut fields = Vec::new();
        if self.eat(&TokenKind::LParen) {
            while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                let ty = self.parse_type()?;
                let name = if matches!(self.peek(), TokenKind::Ident(_)) {
                    Some(self.expect_ident()?)
                } else {
                    None
                };
                fields.push(VariantField { name, ty });
                if !self.eat(&TokenKind::Comma) { break; }
            }
            self.expect(&TokenKind::RParen)?;
        }
        self.expect_newline()?;
        Ok(EnumVariant { name, fields, attrs, line, col })
    }

    fn parse_trait_decl(&mut self) -> Result<TraitDecl, ParseError> {
        let line = self.line();
        let col = self.col();
        self.expect(&TokenKind::Trait)?;
        let name = self.expect_ident()?;
        let (type_params, _) = self.parse_type_params();
        let mut parents = Vec::new();
        // `trait B as A:` or `trait B as A, C:` — supertrait(s) via `as`
        if self.eat(&TokenKind::As) {
            parents.push(self.expect_ident()?);
            while self.eat(&TokenKind::Comma) {
                self.skip_newlines_and_indent();
                parents.push(self.expect_ident()?);
            }
            self.expect(&TokenKind::Colon)?;
        } else {
            self.expect(&TokenKind::Colon)?;
        }
        self.expect_newline()?;
        self.expect(&TokenKind::Indent)?;
        let mut signatures = Vec::new();
        let mut defaults = Vec::new();
        let mut type_signatures = Vec::new();
        let mut assoc_types = Vec::new();
        loop {
            self.skip_newlines();
            if self.check(&TokenKind::Dedent) || self.check(&TokenKind::Eof) { break; }
            if self.check(&TokenKind::Def) || self.check(&TokenKind::Req)
                || (self.check(&TokenKind::Task)
                    && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind),
                                Some(TokenKind::Def | TokenKind::Req | TokenKind::Ident(_)
                                    | TokenKind::Void | TokenKind::LBracket | TokenKind::LBrace
                                    | TokenKind::LParen)))
            {
                let decl_or_sig = self.parse_fn_signature_or_default()?;
                match decl_or_sig {
                    Either::Left(sig)  => signatures.push(sig),
                    Either::Right(decl) => defaults.push(decl),
                }
            } else if self.check(&TokenKind::Type) {
                let after_type = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                if matches!(after_type, Some(TokenKind::Ident(_))) {
                    let decl_line = self.line();
                    self.advance(); // consume `type`
                    let assoc_name = self.expect_ident()?;
                    let (assoc_type_params, _) = self.parse_type_params();
                    let constraint = if self.eat(&TokenKind::As) {
                        Some(self.parse_type()?)
                    } else {
                        None
                    };
                    self.expect_newline_soft();
                    assoc_types.push(AssocTypeDecl { name: assoc_name, constraint, type_params: assoc_type_params, line: decl_line , col: 0 });
                } else {
                    self.advance(); // consume `type`
                    let mutating = if self.check(&TokenKind::Def) {
                        self.advance(); true
                    } else if self.check(&TokenKind::Req) {
                        self.advance(); false
                    } else {
                        break;
                    };
                    let return_ty = self.try_parse_return_type_prefix();
                    let sig_name = self.expect_ident()?;
                    let (type_params_sig, _) = self.parse_type_params();
                    let params = self.parse_params()?;
                    let mut throws = self.eat(&TokenKind::Throws);
                    let mut task   = self.eat(&TokenKind::Task);
                    if !throws { throws = self.eat(&TokenKind::Throws); }
                    if !task   { task   = self.eat(&TokenKind::Task);   }
                    self.expect_newline_soft();
                    type_signatures.push(FnSignature {
                        name: sig_name, params, return_ty, throws, task, stream: false, mutating,
                        return_mutable: false, type_params: type_params_sig, line, col: 0,
                    });
                }
            } else {
                break;
            }
        }
        self.eat(&TokenKind::Dedent);
        let assoc_names: Vec<String> = assoc_types.iter().map(|a| a.name.clone()).collect();
        let signatures = signatures.into_iter()
            .map(|sig| resolve_assoc_in_sig(sig, &assoc_names))
            .collect();
        let defaults = defaults.into_iter()
            .map(|d| resolve_assoc_in_fn(d, &assoc_names))
            .collect();
        Ok(TraitDecl { name, parents, signatures, defaults, type_signatures, type_params, assoc_types, line, col })
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(s) => {
                self.advance();
                Ok(s)
            }
            TokenKind::Get => {
                self.advance();
                Ok("get".to_string())
            }
            TokenKind::Set => {
                self.advance();
                Ok("set".to_string())
            }
            other => Err(ParseError::Generic {
                line: self.line(), col: self.col(),
                msg: format!("expected identifier, got {:?}", other), len: self.tok_len(),
            }),
        }
    }

    /// Like `expect_ident`, but also accepts any keyword token as an identifier.
    /// Used in contexts where keywords are valid names:
    ///   - field / method access after `.` (e.g. `future.wait`, `obj.join()`)
    ///   - function / method name in `def`/`req`/`stream` declarations
    ///   - labeled argument names
    fn expect_ident_or_keyword(&mut self) -> Result<String, ParseError> {
        let name = self.keyword_as_ident_str(self.peek());
        if let Some(s) = name {
            self.advance();
            Ok(s)
        } else {
            Err(ParseError::Generic {
                line: self.line(), col: self.col(),
                msg: format!("expected identifier, got {:?}", self.peek()), len: self.tok_len(),
            })
        }
    }

    /// If `tok` is an identifier token (including soft-keyword forms), return its
    /// string representation.  Returns `None` for structural tokens that cannot
    /// serve as identifiers (punctuation, literals, etc.).
    fn keyword_as_ident_str(&self, tok: &TokenKind) -> Option<String> {
        match tok {
            TokenKind::Ident(s) => Some(s.clone()),
            // All keywords allowed as field/method names (e.g. enum variants, method names).
            TokenKind::Let       => Some("Let".into()),
            TokenKind::Var       => Some("Var".into()),
            TokenKind::Def       => Some("def".into()),
            TokenKind::Return    => Some("return".into()),
            TokenKind::If        => Some("If".into()),
            TokenKind::Elif      => Some("Elif".into()),
            TokenKind::Else      => Some("Else".into()),
            TokenKind::Match     => Some("Match".into()),
            TokenKind::While     => Some("While".into()),
            TokenKind::Do        => Some("do".into()),
            TokenKind::Loop      => Some("loop".into()),
            TokenKind::Wait      => Some("wait".into()),
            TokenKind::For       => Some("For".into()),
            TokenKind::In        => Some("in".into()),
            TokenKind::Break     => Some("Break".into()),
            TokenKind::Continue  => Some("Continue".into()),
            TokenKind::Struct    => Some("Struct".into()),
            TokenKind::Enum      => Some("Enum".into()),
            TokenKind::Trait     => Some("Trait".into()),
            TokenKind::Use       => Some("Use".into()),
            TokenKind::Ext       => Some("ext".into()),
            TokenKind::As        => Some("As".into()),
            TokenKind::And       => Some("And".into()),
            TokenKind::Or        => Some("Or".into()),
            TokenKind::Not       => Some("Not".into()),
            TokenKind::Is        => Some("Is".into()),
            TokenKind::SelfKw    => Some("SelfKw".into()),
            TokenKind::Throw     => Some("throw".into()),
            TokenKind::Throws    => Some("Throws".into()),
            TokenKind::Try       => Some("try".into()),
            TokenKind::Catch     => Some("catch".into()),
            TokenKind::Defer     => Some("Defer".into()),
            TokenKind::Void      => Some("Void".into()),
            TokenKind::Pub       => Some("pub".into()),
            TokenKind::Guard     => Some("guard".into()),
            TokenKind::Task      => Some("task".into()),
            TokenKind::Join      => Some("join".into()),
            TokenKind::Stream    => Some("stream".into()),
            TokenKind::Yield     => Some("yield".into()),
            TokenKind::Static    => Some("static".into()),
            TokenKind::Type      => Some("type".into()),
            TokenKind::Req       => Some("req".into()),
            TokenKind::Transient => Some("transient".into()),
            TokenKind::With      => Some("With".into()),
            TokenKind::Get       => Some("get".into()),
            TokenKind::Set       => Some("set".into()),
            TokenKind::Init      => Some("init".into()),
            TokenKind::Pass      => Some("pass".into()),
            TokenKind::Native    => Some("native".into()),
            TokenKind::Mod       => Some("mod".into()),
            TokenKind::New       => Some("new".into()),
            TokenKind::Bool(b)   => Some(if *b { "True".into() } else { "False".into() }),
            TokenKind::Nil       => Some("Nil".into()),
            _ => None,
        }
    }
}

// ─── Const generic helpers ───────────────────────────────────────────────────

/// Recursively scan a type for implicit generic params and add them to `type_params`:
/// - `Type::Named("$N:usize")` (const-encoded) → adds `"$N:usize"`
/// - `Type::TypeParam("T")` (single uppercase letter) → adds `"T"`
///
/// Supports the inline shorthand: `Matrix<T, uint N>` in a param type auto-adds
/// both `"T"` and `"$N:usize"` to the enclosing function's type_params list.
pub(crate) fn collect_const_params_from_type(ty: &crate::ast::Type, type_params: &mut Vec<String>) {
    use crate::ast::Type;
    match ty {
        Type::Named(n) if n.starts_with('$') => {
            if !type_params.contains(n) { type_params.push(n.clone()); }
        }
        Type::TypeParam(n) => {
            if !type_params.contains(n) { type_params.push(n.clone()); }
        }
        Type::Generic(_, args) => {
            for arg in args { collect_const_params_from_type(arg, type_params); }
        }
        Type::Array(inner) | Type::Set(inner) | Type::Optional(inner)
        | Type::Dyn(inner) | Type::Impl(inner) | Type::Qualified(inner, _) => {
            collect_const_params_from_type(inner, type_params);
        }
        Type::Dict(k, v) => {
            collect_const_params_from_type(k, type_params);
            collect_const_params_from_type(v, type_params);
        }
        Type::Tuple(elems) => {
            for e in elems { collect_const_params_from_type(e, type_params); }
        }
        _ => {}
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
