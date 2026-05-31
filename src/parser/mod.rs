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
use crate::lexer::{lex, LexError, RawInterpPart, Token, TokenKind};
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
    #[error("line {line}: {msg}")]
    Generic { line: usize, msg: String },
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
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0, depth: 0, allow_noparen_closure: true, allow_trailing_closure: true }
    }

    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn peek_token(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn line(&self) -> usize {
        self.tokens[self.pos].line
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
                line: self.line(),
                msg: format!("expected {:?}, got {:?}", kind, self.peek()),
            })
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), TokenKind::Newline | TokenKind::Comment(_)) {
            self.advance();
        }
    }

    /// Like `skip_newlines` but also skips INDENT/DEDENT tokens.
    /// Used inside `(…)` parameter lists to allow multi-line declarations.
    fn skip_newlines_and_indent(&mut self) {
        while matches!(self.peek(), TokenKind::Newline | TokenKind::Indent | TokenKind::Dedent) {
            self.advance();
        }
    }

    fn expect_newline(&mut self) -> Result<(), ParseError> {
        if self.check(&TokenKind::Newline) || self.check(&TokenKind::Eof) {
            self.skip_newlines();
            Ok(())
        } else {
            Err(ParseError::Generic {
                line: self.line(),
                msg: format!("expected newline, got {:?}", self.peek()),
            })
        }
    }

    /// Like expect_newline, but also succeeds when already past the newline
    /// (e.g. after parsing a block-form expression that consumed trailing whitespace).
    fn expect_newline_soft(&mut self) {
        if self.check(&TokenKind::Newline) || self.check(&TokenKind::Eof) {
            self.skip_newlines();
        }
        // else: newline was already consumed (e.g. by a block form) — just proceed
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        self.skip_newlines();
        while !self.check(&TokenKind::Eof) {
            let item = self.parse_item(false)?;
            items.push(item);
            self.skip_newlines();
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self, is_pub: bool) -> Result<Item, ParseError> {
        match self.peek().clone() {
            TokenKind::Comment(text) => {
                self.advance();
                self.eat(&TokenKind::Newline);
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
                // `stream def …` / `stream req …` / `stream RetType name(…):` — stream function
                match self.tokens.get(self.pos + 1).map(|t| &t.kind) {
                    Some(TokenKind::Def) => Ok(Item::Fn(self.parse_fn_decl(is_pub, true)?)),
                    Some(TokenKind::Req) => Ok(Item::Fn(self.parse_fn_decl(is_pub, false)?)),
                    _ if self.is_stream_fn_shorthand() => Ok(Item::Fn(self.parse_fn_decl(is_pub, true)?)),
                    _ => Ok(Item::Stmt(self.parse_stmt()?)),
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
            TokenKind::Static | TokenKind::Let | TokenKind::Var => {
                if self.is_let_destructure() {
                    let line = self.line();
                    let _is_static = self.eat(&TokenKind::Static);
                    let mutable = matches!(self.peek(), TokenKind::Var);
                    self.advance(); // consume let/var
                    Ok(Item::Stmt(Stmt::LetDestructure(self.parse_let_destructure(mutable, line)?)))
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
            TokenKind::Void | TokenKind::Any => true,
            TokenKind::LBracket | TokenKind::LBrace | TokenKind::LParen => true,
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
        self.expect(&TokenKind::Use)?;

        // Parse all dot-separated segments.
        let mut path = vec![self.expect_ident()?];
        while self.eat(&TokenKind::Dot) {
            if self.eat(&TokenKind::Star) {
                // use a.b.c.*  — glob import
                self.expect_newline()?;
                return Ok(UseDecl { path, glob: true, items: vec![], line });
            }
            path.push(self.expect_ident()?);
        }

        // Single-segment path (`use a`) — whole-module edge case.
        if path.len() == 1 {
            self.expect_newline()?;
            return Ok(UseDecl { path, glob: false, items: vec![], line });
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
        Ok(UseDecl { path, glob: false, items, line })
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
            Some(TokenKind::Void) | Some(TokenKind::Any) => true,
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
                        TokenKind::Newline | TokenKind::Eof => break,
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
            Some(TokenKind::Void) | Some(TokenKind::Any) => true,
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
                TokenKind::Newline | TokenKind::Indent | TokenKind::Dedent
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
            let ty = self.parse_type_qualifier(ty);
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
            TokenKind::Any => true,
            TokenKind::LParen => true,
            TokenKind::LBracket => true,
            TokenKind::LBrace => true,
            _ => false,
        };

        if is_type_start {
            let saved = self.pos;
            if let Ok(ty) = self.parse_type() {
                let ty = self.parse_type_qualifier(ty);
                // Accept any token that can be used as an identifier (including soft keywords
                // like `join`, `wait`, etc.) as the function / field name following the type.
                if self.keyword_as_ident_str(self.peek()).is_some() {
                    return Some(ty);
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
                let mut depth = 0i32;
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
                            TokenKind::Ident(q) if matches!(q.as_str(), "copy" | "const" | "stack" | "heap" | "actor" | "wguard") => { i += 1; }
                            TokenKind::Guard => { i += 1; qual_is_auto_or_shared = true; }
                            TokenKind::Ident(q) if q == "weak" => { i += 1; }
                            TokenKind::Task => { i += 1; qual_is_auto_or_shared = true; }
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
            // `any Trait name`
            TokenKind::Any => {
                let mut i = self.pos + 1;
                if i < self.tokens.len() && matches!(self.tokens[i].kind, TokenKind::Ident(_)) {
                    i += 1;
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
                }
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
                            TokenKind::Ident(q) if matches!(q.as_str(), "copy" | "const" | "stack" | "heap" | "actor" | "wguard") => { i += 1; }
                            TokenKind::Guard => { i += 1; qual_is_auto_or_shared = true; }
                            TokenKind::Ident(q) if q == "weak" => { i += 1; }
                            TokenKind::Task => { i += 1; qual_is_auto_or_shared = true; }
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
            _ => false,
        }
    }

    /// Parse `<T>`, `<T as Trait>`, `<T as (Trait1, Trait2)>`, `<T, U as Comparable>`, etc.
    fn parse_type_params(&mut self) -> (Vec<String>, Vec<(String, String)>) {
        if !self.eat(&TokenKind::Lt) { return (vec![], vec![]); }
        let mut params = Vec::new();
        let mut where_clause = Vec::new();
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
        }
        let _ = self.expect(&TokenKind::Gt);
        (params, where_clause)
    }

    fn parse_struct_decl(&mut self, is_pub: bool) -> Result<StructDecl, ParseError> {
        self.parse_struct_decl_with_attrs(is_pub, vec![])
    }

    fn parse_struct_decl_with_attrs(&mut self, is_pub: bool, attrs: Vec<Attr>) -> Result<StructDecl, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Struct)?;
        let name = self.expect_ident()?;
        let (type_params, where_clause) = self.parse_type_params();
        let mut protocols = Vec::new();
        if self.eat(&TokenKind::As) {
            let proto = self.expect_ident()?;
            protocols.push(proto);
            while self.eat(&TokenKind::Comma) {
                protocols.push(self.expect_ident()?);
            }
        }
        self.expect(&TokenKind::Colon)?;
        if self.check(&TokenKind::Pass) {
            self.advance();
            self.expect_newline_soft();
            return Ok(StructDecl { name, is_pub, is_native: false, protocols, fields: vec![], inits: vec![], methods: vec![], conversions: vec![], type_params, where_clause, setters: vec![], type_methods: vec![], type_vars: vec![], assoc_type_defs: vec![], attrs, line });
        }
        if self.check(&TokenKind::Native) {
            self.advance();
            self.expect_newline_soft();
            return Ok(StructDecl { name, is_pub, is_native: true, protocols, fields: vec![], inits: vec![], methods: vec![], conversions: vec![], type_params, where_clause, setters: vec![], type_methods: vec![], type_vars: vec![], assoc_type_defs: vec![], attrs, line });
        }
        self.expect_newline()?;
        self.expect(&TokenKind::Indent)?;
        self.parse_struct_body(name, is_pub, protocols, type_params, where_clause, attrs, line)
    }

    fn parse_struct_body(&mut self, name: String, is_pub: bool, protocols: Vec<String>, type_params: Vec<String>, where_clause: Vec<(String, String)>, attrs: Vec<Attr>, line: usize) -> Result<StructDecl, ParseError> {

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
                        assoc_type_defs.push(AssocTypeDef { name: assoc_name, ty, line: def_line });
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
        Ok(StructDecl { name, is_pub, is_native: false, protocols, fields, inits, methods, conversions, type_params, where_clause, setters, type_methods, type_vars, assoc_type_defs, attrs, line })
    }

    fn parse_ext_decl(&mut self) -> Result<ExtDecl, ParseError> {
        let line = self.line();
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
                        assoc_type_defs.push(AssocTypeDef { name: assoc_name, ty, line: def_line });
                    } else {
                        return Err(ParseError::Generic {
                            msg: "unexpected 'type' in ext body (only 'type Name = T' is allowed)".into(),
                            line: self.line(),
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
                                line: self.line(),
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
                            line: self.line(),
                        });
                    }
                }
                TokenKind::As => {
                    conversions.push(self.parse_as_decl()?);
                }
                _ => {
                    return Err(ParseError::Generic {
                        msg: format!("unexpected token in ext body: {:?}", self.peek()),
                        line: self.line(),
                    });
                }
            }
        }
        self.eat(&TokenKind::Dedent);

        Ok(ExtDecl { type_name, type_args, type_params, where_clause, traits, methods, setters, conversions, assoc_type_defs, line })
    }

    fn parse_enum_decl(&mut self, is_pub: bool) -> Result<EnumDecl, ParseError> {
        self.parse_enum_decl_with_attrs(is_pub, vec![])
    }

    fn parse_enum_decl_with_attrs(&mut self, is_pub: bool, attrs: Vec<Attr>) -> Result<EnumDecl, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Enum)?;
        let name = self.expect_ident()?;
        let (type_params, _) = self.parse_type_params();
        let mut protocols = Vec::new();
        if self.eat(&TokenKind::As) {
            protocols.push(self.expect_ident()?);
            while self.eat(&TokenKind::Comma) {
                protocols.push(self.expect_ident()?);
            }
        }
        self.expect(&TokenKind::Colon)?;
        if self.check(&TokenKind::Pass) {
            self.advance();
            self.expect_newline_soft();
            return Ok(EnumDecl { name, is_pub, is_native: false, type_params, protocols, variants: vec![], methods: vec![], setters: vec![], conversions: vec![], attrs, line });
        }
        if self.check(&TokenKind::Native) {
            self.advance();
            self.expect_newline_soft();
            return Ok(EnumDecl { name, is_pub, is_native: true, type_params, protocols, variants: vec![], methods: vec![], setters: vec![], conversions: vec![], attrs, line });
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
                            line: self.line(),
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
        Ok(EnumDecl { name, is_pub, is_native: false, type_params, protocols, variants, methods, setters, conversions, attrs, line })
    }

    fn parse_mod_decl(&mut self, _is_pub: bool) -> Result<ModDecl, ParseError> {
        let line = self.line();
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
        Ok(ModDecl { name, items, line })
    }

    fn parse_enum_variant(&mut self) -> Result<EnumVariant, ParseError> {
        let line = self.line();
        // Optional per-variant attributes: `@error("message")`, `@doc("...")`, etc.
        let attrs = if self.check(&TokenKind::At) { self.parse_attrs() } else { vec![] };
        let name = self.expect_ident()?;
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
        Ok(EnumVariant { name, fields, attrs, line })
    }

    fn parse_trait_decl(&mut self) -> Result<TraitDecl, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Trait)?;
        let name = self.expect_ident()?;
        let (type_params, _) = self.parse_type_params();
        let mut parents = Vec::new();
        if matches!(self.peek(), TokenKind::Colon) {
            let after_colon = self.tokens.get(self.pos + 1).map(|t| &t.kind);
            if matches!(after_colon, Some(TokenKind::Ident(_))) {
                self.advance(); // eat ':'
                parents.push(self.expect_ident()?);
                while self.eat(&TokenKind::Comma) {
                    parents.push(self.expect_ident()?);
                }
                self.expect(&TokenKind::Colon)?; // body colon
            } else {
                self.expect(&TokenKind::Colon)?;
            }
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
                    assoc_types.push(AssocTypeDecl { name: assoc_name, constraint, type_params: assoc_type_params, line: decl_line });
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
                        type_params: type_params_sig, line,
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
        Ok(TraitDecl { name, parents, signatures, defaults, type_signatures, type_params, assoc_types, line })
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
                line: self.line(),
                msg: format!("expected identifier, got {:?}", other),
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
                line: self.line(),
                msg: format!("expected identifier, got {:?}", self.peek()),
            })
        }
    }

    /// If `tok` is an identifier token (including soft-keyword forms), return its
    /// string representation.  Returns `None` for structural tokens that cannot
    /// serve as identifiers (punctuation, literals, etc.).
    fn keyword_as_ident_str(&self, tok: &TokenKind) -> Option<String> {
        match tok {
            TokenKind::Ident(s) => Some(s.clone()),
            // Keywords that are commonly used as method/field/function names:
            TokenKind::Get       => Some("get".into()),
            TokenKind::Set       => Some("set".into()),
            TokenKind::Wait      => Some("wait".into()),
            TokenKind::Join      => Some("join".into()),
            TokenKind::Type      => Some("type".into()),
            TokenKind::Init      => Some("init".into()),
            TokenKind::Pass      => Some("pass".into()),
            TokenKind::Stream    => Some("stream".into()),
            TokenKind::Yield     => Some("yield".into()),
            TokenKind::Select    => Some("select".into()),
            TokenKind::Loop      => Some("loop".into()),
            TokenKind::Do        => Some("do".into()),
            TokenKind::Mod       => Some("mod".into()),
            TokenKind::Any       => Some("any".into()),
            TokenKind::Static    => Some("static".into()),
            TokenKind::Native    => Some("native".into()),
            TokenKind::Transient => Some("transient".into()),
            TokenKind::Req       => Some("req".into()),
            TokenKind::Def       => Some("def".into()),
            TokenKind::Ext       => Some("ext".into()),
            TokenKind::Task      => Some("task".into()),
            TokenKind::In        => Some("in".into()),
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
        | Type::Any(inner) | Type::Qualified(inner, _) => {
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
