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
use crate::lexer::{Token, TokenKind, RawInterpPart};

impl Parser {
    pub(crate) fn parse_fn_decl(&mut self, is_pub: bool, mutating: bool) -> Result<FnDecl, ParseError> {
        let line = self.line();
        // `task`/`stream` may come BEFORE the keyword: `task def f():`, `stream def f():`
        // Shorthand: `task RetType f():` or `stream RetType f():` — `def` is implicit.
        let prefix_task   = self.eat(&TokenKind::Task);
        let prefix_stream = if !prefix_task { self.eat(&TokenKind::Stream) } else { false };
        // Consume `def` or `req` — skipped in the shorthand forms:
        //   • `task RetType f():` / `stream RetType f():` — task/stream prefix present
        //   • `RetType f():` — bare return type without any keyword (top-level shorthand)
        let bare_shorthand = !prefix_task && !prefix_stream
            && !self.check(&TokenKind::Def)
            && !self.check(&TokenKind::Req);
        let shorthand = bare_shorthand
            || ((prefix_task || prefix_stream)
                && !self.check(&TokenKind::Def)
                && !self.check(&TokenKind::Req));
        if !shorthand {
            if mutating {
                self.expect(&TokenKind::Def)?;
            } else {
                self.expect(&TokenKind::Req)?;
            }
        }

        // Return type is optional: `def foo()` defaults to void, `def int foo()` is explicit.
        // Only closures (`let f = (x): x * 2`) may omit parameter types.
        let (return_ty, qualifier, name) = self.parse_fn_head()?;
        // Default to void when no return type is specified.
        let return_ty = return_ty.or(Some(Type::Void));
        let (mut type_params, where_clause) = self.parse_type_params();

        let params = self.parse_params()?;
        // All parameters in a `def`/`req` declaration must have explicit type annotations.
        // Unannotated params produce invalid Rust when using --emit-rust.
        for p in &params {
            if p.ty.is_none() && !p.variadic {
                return Err(ParseError::Generic {
                    msg: format!(
                        "parameter '{}' in 'def {}' has no type annotation — add a type (e.g. 'int {}')",
                        p.name, name, p.name
                    ),
                    line: p.line,
                });
            }
        }
        // Auto-collect implicit generic/const params from param types and return type
        // when no explicit `<...>` was written. E.g. `def T get(Matrix<T, uint N> m):`
        // automatically gets type_params = ["T", "$N:usize"].
        if type_params.is_empty() {
            if let Some(ret) = &return_ty {
                crate::parser::collect_const_params_from_type(ret, &mut type_params);
            }
            for p in &params {
                if let Some(ty) = &p.ty {
                    crate::parser::collect_const_params_from_type(ty, &mut type_params);
                }
            }
        }
        let task   = prefix_task;
        let stream = prefix_stream;
        let mut throws = self.eat(&TokenKind::Throws);
        let mut throws_ty: Option<Type> = None;
        if throws {
            throws_ty = self.parse_throws_type()?;
        }
        self.expect(&TokenKind::Colon)?;
        // Support block body, inline body, `pass` (empty body), and `native` (runtime-implemented)
        let (body, is_native) = if self.check(&TokenKind::Pass) {
            self.advance();
            self.expect_newline_soft();
            (vec![], false)
        } else if self.check(&TokenKind::Native) {
            self.advance();
            self.expect_newline_soft();
            (vec![], true)
        } else if self.check(&TokenKind::Newline) || self.check(&TokenKind::Eof) {
            self.expect_newline()?;
            (self.parse_block()?, false)
        } else {
            let expr_line = self.line();
            // Use parse_inline_stmt so that assignments like `self.x = val` are valid.
            // Wrap plain (non-assignment) expressions in an implicit Return;
            // assignments and explicit control-flow keep their own statement semantics.
            let stmt = self.parse_inline_stmt()?;
            self.expect_newline_soft();
            let s = match stmt {
                Stmt::Expr(ref e) if !matches!(e.kind, ExprKind::Assign(..)) => {
                    vec![Stmt::Return(ReturnStmt { value: Some(e.clone()), line: expr_line })]
                }
                other => vec![other],
            };
            (s, false)
        };

        // ── Auto-infer type params from single-uppercase-letter types in the signature ──
        // If no explicit `<...>` was written, collect every TypeParam that appears in
        // the return type or parameter types so the interpreter can bind them.
        if type_params.is_empty() {
            if let Some(ref ret) = return_ty {
                Self::collect_type_params_from_ty(ret, &mut type_params);
            }
            for param in &params {
                if let Some(ref ty) = param.ty {
                    Self::collect_type_params_from_ty(ty, &mut type_params);
                }
            }
        }

        Ok(FnDecl {
            name,
            qualifier,
            params,
            return_ty,
            body,
            is_pub,
            throws,
            throws_ty,
            task,
            stream,
            mutating,
            is_native,
            type_params,
            where_clause,
            attrs: vec![],
            line,
        })
    }

    pub(crate) fn parse_fn_decl_with_attrs(&mut self, is_pub: bool, mutating: bool, attrs: Vec<Attr>) -> Result<FnDecl, ParseError> {
        let mut decl = self.parse_fn_decl(is_pub, mutating)?;
        decl.attrs = attrs;
        Ok(decl)
    }

    /// Recursively collect all `TypeParam` names from a type into `out`, preserving
    /// first-occurrence order and avoiding duplicates.
    pub(crate) fn collect_type_params_from_ty(ty: &Type, out: &mut Vec<String>) {
        match ty {
            Type::TypeParam(name) => {
                if !out.contains(name) { out.push(name.clone()); }
            }
            Type::Optional(inner) | Type::Array(inner) | Type::Set(inner) => {
                Self::collect_type_params_from_ty(inner, out);
            }
            Type::Qualified(inner, _) => Self::collect_type_params_from_ty(inner, out),
            Type::Dict(k, v) => {
                Self::collect_type_params_from_ty(k, out);
                Self::collect_type_params_from_ty(v, out);
            }
            Type::Tuple(types) => {
                for t in types { Self::collect_type_params_from_ty(t, out); }
            }
            Type::Fn(ret, param_types, _, _, _) => {
                if let Some(r) = ret { Self::collect_type_params_from_ty(r, out); }
                for p in param_types { Self::collect_type_params_from_ty(p, out); }
            }
            Type::Generic(_, args) => {
                for a in args { Self::collect_type_params_from_ty(a, out); }
            }
            _ => {}
        }
    }

    /// Parses `[ReturnType] [Qualifier.]Name` after `def`
    pub(crate) fn parse_fn_head(&mut self) -> Result<(Option<Type>, Option<String>, String), ParseError> {
        // Check if we have a return type annotation before the name.
        // Heuristic: if the current token is a type keyword or uppercase ident,
        // and the *next* non-whitespace token is an ident or ident.ident pattern,
        // treat current as return type.
        let maybe_ty = self.try_parse_return_type_prefix();

        // Now parse qualifier.name or just name.
        // Use expect_ident_or_keyword so reserved words (e.g. `join`, `wait`) are
        // valid function / method names.
        let first_ident = self.expect_ident_or_keyword()?;
        if self.eat(&TokenKind::Dot) {
            let method_name = self.expect_ident_or_keyword()?;
            Ok((maybe_ty, Some(first_ident), method_name))
        } else {
            Ok((maybe_ty, None, first_ident))
        }
    }

    pub(crate) fn parse_params(&mut self) -> Result<Vec<Param>, ParseError> {
        self.expect(&TokenKind::LParen)?;
        self.skip_newlines_and_indent(); // allow `(\n    param,` multi-line form
        let mut params = Vec::new();
        while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
            params.push(self.parse_param()?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
            self.skip_newlines_and_indent(); // allow newline + indent between params
        }
        self.skip_newlines_and_indent(); // allow newline before `)`
        self.expect(&TokenKind::RParen)?;
        Ok(params)
    }

    pub(crate) fn parse_param(&mut self) -> Result<Param, ParseError> {
        let line = self.line();
        let mutable = self.eat(&TokenKind::Var);
        let mut variadic = false;

        // Support `Type[qual][...]? name` (Swift-style) or bare `name` parameter syntax.
        let (name, ty, owned) = if self.is_type_start_before_ident() {
            let saved = self.pos;
            if let Ok(t) = self.parse_type() {
                // Apply ownership qualifier: `Dog'`, `Dog'copy`, etc.
                let t = self.parse_type_qualifier(t);
                // `var T&` → mutable borrow: absorb `mutable` into the type qualifier.
                let t = if mutable {
                    Self::apply_var_to_borrow(t)
                } else {
                    t
                };
                let is_owned = matches!(&t, Type::Qualified(_, OwnerQual::Owned));

                // Variadic marker sits between type and name: `int... args`
                variadic = self.eat(&TokenKind::DotDotDot);

                if matches!(self.peek(), TokenKind::Ident(_)) {
                    let name = self.expect_ident()?;
                    // Also accept `Type name...` (marker after name)
                    if !variadic { variadic = self.eat(&TokenKind::DotDotDot); }
                    // `Type name (Params) throws? task?` — function-typed parameter
                    let final_ty = if matches!(self.peek(), TokenKind::LParen) {
                        self.advance();
                        let mut fn_params = Vec::new();
                        if !matches!(self.peek(), TokenKind::RParen) {
                            fn_params.push(self.parse_type()?);
                            while self.eat(&TokenKind::Comma) {
                                if matches!(self.peek(), TokenKind::RParen) { break; }
                                fn_params.push(self.parse_type()?);
                            }
                        }
                        self.expect(&TokenKind::RParen)?;
                        let fn_task   = self.eat(&TokenKind::Task);
                        let fn_throws = self.eat(&TokenKind::Throws);
                        // Wrap the return type in a Fn type; qualifier was on return type
                        Type::Fn(Some(Box::new(t)), fn_params, fn_throws, fn_task, false)
                    } else {
                        t
                    };
                    (name, Some(final_ty), is_owned)
                } else {
                    self.pos = saved;
                    variadic = false;
                    let n = self.expect_ident()?;
                    (n, None, false)
                }
            } else {
                self.pos = saved;
                let n = self.expect_ident()?;
                (n, None, false)
            }
        } else {
            let n = self.expect_ident()?;
            // `name(Params)` — function-typed parameter with inferred return type
            let ty = if matches!(self.peek(), TokenKind::LParen) {
                self.advance();
                let mut fn_params = Vec::new();
                if !matches!(self.peek(), TokenKind::RParen) {
                    fn_params.push(self.parse_type()?);
                    while self.eat(&TokenKind::Comma) {
                        if matches!(self.peek(), TokenKind::RParen) { break; }
                        fn_params.push(self.parse_type()?);
                    }
                }
                self.expect(&TokenKind::RParen)?;
                let fn_task   = self.eat(&TokenKind::Task);
                let fn_throws = self.eat(&TokenKind::Throws);
                Some(Type::Fn(None, fn_params, fn_throws, fn_task, false))
            } else {
                None
            };
            (n, ty, false)
        };

        // Default value: `= expr`  (only when not part of a labeled-arg check)
        let default = if self.eat(&TokenKind::Eq) {
            Some(self.parse_or()?)  // parse_or avoids eating a trailing comma
        } else {
            None
        };

        Ok(Param { name, ty, mutable, owned, variadic, default, line })
    }

    pub(crate) fn parse_set_decl(&mut self, is_pub: bool) -> Result<SetDecl, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Set)?;
        let name = self.expect_ident()?;
        self.expect(&TokenKind::LParen)?;
        let param_ty = self.parse_type()?;
        let param_ty = self.parse_type_qualifier(param_ty);
        let param_name = self.expect_ident()?;
        self.expect(&TokenKind::RParen)?;
        let mut throws = self.eat(&TokenKind::Throws);
        let mut task   = self.eat(&TokenKind::Task);
        if !throws { throws = self.eat(&TokenKind::Throws); }
        if !task   { task   = self.eat(&TokenKind::Task);   }
        self.expect(&TokenKind::Colon)?;
        let body = if self.check(&TokenKind::Newline) || self.check(&TokenKind::Eof) {
            self.expect_newline()?;
            self.parse_block()?
        } else {
            let stmt = self.parse_inline_stmt()?;
            self.expect_newline_soft();
            vec![stmt]
        };
        Ok(SetDecl { name, param_name, param_ty, is_pub, throws, task, body, line })
    }

    /// Parse a method/variable body block or inline statement.
    pub(crate) fn parse_method_body(&mut self) -> Result<Vec<Stmt>, ParseError> {
        if self.check(&TokenKind::Newline) || self.check(&TokenKind::Eof) {
            self.expect_newline()?;
            self.parse_block()
        } else {
            let stmt = self.parse_inline_stmt()?;
            self.expect_newline_soft();
            Ok(vec![stmt])
        }
    }

    /// Parse `[pub] type def/req/set/var/let …` inside a struct body.
    /// `is_pub` is already consumed by the caller; `type` token is NOT yet consumed.
    pub(crate) fn parse_type_member(&mut self, is_pub: bool) -> Result<TypeMemberKind, ParseError> {
        use crate::ast::{TypeMethod, TypeMethodKind, TypeVar};
        let line = self.line();
        self.expect(&TokenKind::Type)?;
        match self.peek().clone() {
            // ── type var / type let ───────────────────────────────────────────
            TokenKind::Var | TokenKind::Let => {
                let mutable = matches!(self.peek(), TokenKind::Var);
                self.advance(); // consume var / let
                // Optional explicit type before name: `type var int count = 0`
                let ty = self.try_parse_return_type_prefix();
                let name = self.expect_ident()?;
                self.expect(&TokenKind::Eq)?;
                let default = self.parse_expr()?;
                self.expect_newline_soft();
                Ok(TypeMemberKind::Var(TypeVar { name, ty, default, is_pub, mutable, line }))
            }
            // ── type set name(T v): body ──────────────────────────────────────
            TokenKind::Set => {
                self.advance();
                let name = self.expect_ident()?;
                self.expect(&TokenKind::LParen)?;
                let param_ty = self.parse_type()?;
                let param_ty = self.parse_type_qualifier(param_ty);
                let param_name = self.expect_ident()?;
                self.expect(&TokenKind::RParen)?;
                let mut throws = self.eat(&TokenKind::Throws);
                let mut task   = self.eat(&TokenKind::Task);
                if !throws { throws = self.eat(&TokenKind::Throws); }
                if !task   { task   = self.eat(&TokenKind::Task);   }
                self.expect(&TokenKind::Colon)?;
                let body = self.parse_method_body()?;
                let param = crate::ast::Param {
                    name: param_name, ty: Some(param_ty),
                    mutable: false, owned: false, variadic: false, default: None, line,
                };
                Ok(TypeMemberKind::Method(TypeMethod {
                    kind: TypeMethodKind::Set, name, params: vec![param],
                    return_ty: None, body, is_pub, throws, task, line,
                }))
            }
            // ── type def / type req: [RetTy] name(params): body ──────────────
            TokenKind::Def | TokenKind::Req => {
                let kind = if matches!(self.peek(), TokenKind::Def) {
                    TypeMethodKind::Def
                } else {
                    TypeMethodKind::Req
                };
                self.advance();
                let return_ty = self.try_parse_return_type_prefix();
                let name = self.expect_ident()?;
                let params = self.parse_params()?;
                let mut throws = self.eat(&TokenKind::Throws);
                let mut task   = self.eat(&TokenKind::Task);
                if !throws { throws = self.eat(&TokenKind::Throws); }
                if !task   { task   = self.eat(&TokenKind::Task);   }
                self.expect(&TokenKind::Colon)?;
                let body = self.parse_method_body()?;
                Ok(TypeMemberKind::Method(TypeMethod {
                    kind, name, params, return_ty, body, is_pub, throws, task, line,
                }))
            }
            other => Err(ParseError::Generic {
                msg: format!("expected 'def', 'req', 'set', 'var', or 'let' after 'type', got {:?}", other),
                line,
            }),
        }
    }

    pub(crate) fn parse_as_decl(&mut self) -> Result<AsDecl, ParseError> {
        let line = self.line();
        let is_pub = self.eat(&TokenKind::Pub);
        self.expect(&TokenKind::As)?;
        let ty = self.parse_type()?;
        let throws = self.eat(&TokenKind::Throws);
        let task   = self.eat(&TokenKind::Task);
        self.expect(&TokenKind::Colon)?;
        let body = if self.check(&TokenKind::Newline) || self.check(&TokenKind::Eof) {
            self.expect_newline()?;
            self.parse_block()?
        } else {
            // Inline: `as String: expr`
            let expr_line = self.line();
            let expr = self.parse_expr()?;
            self.expect_newline_soft();
            vec![Stmt::Expr(expr)]
        };
        Ok(AsDecl { is_pub, ty, throws, task, body, line })
    }

    pub(crate) fn parse_field_decl(&mut self) -> Result<FieldDecl, ParseError> {
        let line = self.line();
        let explicit_pub = self.eat(&TokenKind::Pub);
        // `transient` implies `var` (mutable)
        let transient = self.eat(&TokenKind::Transient);
        let (is_pub, mutable) = if transient {
            // transient → mutable (implicitly var), private unless pub was explicit
            (explicit_pub, true)
        } else if self.eat(&TokenKind::Let) {
            // explicit `let` → private unless `pub` was explicit
            (explicit_pub, false)
        } else if self.eat(&TokenKind::Var) {
            // explicit `var` → private unless `pub` was explicit
            (explicit_pub, true)
        } else {
            // no keyword → implicit `pub let`
            (true, false)
        };
        let ty = self.parse_type()?;
        let name = self.expect_ident()?;
        let default = if self.eat(&TokenKind::Eq) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect_newline()?;
        Ok(FieldDecl { name, is_pub, mutable, transient, ty, default, line })
    }

    pub(crate) fn parse_init_decl(&mut self) -> Result<InitDecl, ParseError> {
        use crate::ast::{InitDecl, InitParam};
        let line = self.line();
        self.expect(&TokenKind::Init)?;
        self.expect(&TokenKind::LParen)?;

        let mut params = Vec::new();
        while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
            let pline = self.line();

            // `pub var?` or `var` or nothing
            let is_pub = self.eat(&TokenKind::Pub);
            let mutable = self.eat(&TokenKind::Var);

            // Parse optional type + name.
            // Strategy: if current ident is followed by `,` `)` or `=`, it's untyped.
            let (ty, name) = if matches!(self.peek(), TokenKind::Ident(_)) {
                let next2 = self.tokens.get(self.pos + 1).map(|t| &t.kind);
                if matches!(next2, Some(TokenKind::Comma) | Some(TokenKind::RParen) | Some(TokenKind::Eq)) {
                    let n = self.expect_ident()?;
                    (None, n)
                } else {
                    let ty = self.parse_type()?;
                    let n = self.expect_ident()?;
                    (Some(ty), n)
                }
            } else {
                let ty = self.parse_type()?;
                let n = self.expect_ident()?;
                (Some(ty), n)
            };

            let default = if self.eat(&TokenKind::Eq) {
                Some(self.parse_expr()?)
            } else {
                None
            };

            params.push(InitParam { is_pub, mutable, name, ty, default, line: pline });

            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;

        // No body — all params declare fields
        if self.check(&TokenKind::Newline) || self.check(&TokenKind::Eof) {
            self.eat(&TokenKind::Newline);
            return Ok(InitDecl { params, body: vec![], line });
        }

        // With body — all params are local
        self.expect(&TokenKind::Colon)?;
        let body = if self.check(&TokenKind::Newline) || self.check(&TokenKind::Eof) {
            self.expect_newline()?;
            self.parse_block()?
        } else {
            let stmt = self.parse_inline_stmt()?;
            self.expect_newline_soft();
            vec![stmt]
        };
        Ok(InitDecl { params, body, line })
    }

    /// `use Name as Type` — pure type alias (transparent, same Rust type)
    pub(crate) fn parse_alias_decl(&mut self) -> Result<AliasDecl, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Use)?;
        let name = self.expect_ident()?;
        self.expect(&TokenKind::As)?;
        let ty = self.parse_type()?;
        // Apply qualifier to the entire type if a tick follows
        let ty = self.parse_type_qualifier(ty);
        self.expect_newline()?;
        Ok(AliasDecl { name, ty, newtype: false, line })
    }

    /// `type Name as InnerType` — newtype wrapper (distinct Rust struct around InnerType)
    pub(crate) fn parse_newtype_decl(&mut self) -> Result<AliasDecl, ParseError> {
        let line = self.line();
        self.expect(&TokenKind::Type)?;
        let name = self.expect_ident()?;
        self.expect(&TokenKind::As)?;
        let ty = self.parse_type()?;
        self.expect_newline()?;
        Ok(AliasDecl { name, ty, newtype: true, line })
    }

    // ─── Attribute parsing ───────────────────────────────────────────────────

    pub(crate) fn parse_attrs(&mut self) -> Vec<Attr> {
        let mut attrs = Vec::new();
        while self.check(&TokenKind::At) {
            let line = self.line();
            self.advance(); // consume @
            let name = match self.expect_ident() {
                Ok(n) => n,
                Err(_) => break,
            };
            let mut args = Vec::new();
            if self.eat(&TokenKind::LParen) {
                // Parenthesised form: @derive(thiserror::Error, Debug)
                while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                    let arg = self.collect_attr_arg();
                    args.push(arg);
                    if !self.eat(&TokenKind::Comma) { break; }
                }
                let _ = self.expect(&TokenKind::RParen);
            } else if !matches!(self.peek(), TokenKind::Newline | TokenKind::Eof | TokenKind::At | TokenKind::Indent | TokenKind::Dedent) {
                // Paren-free form: @error "msg"  or  @derive thiserror::Error, Debug
                loop {
                    let arg = self.collect_attr_arg();
                    if !arg.is_empty() { args.push(arg); }
                    if !self.eat(&TokenKind::Comma) { break; }
                    if matches!(self.peek(), TokenKind::Newline | TokenKind::Eof) { break; }
                }
            }
            // Skip optional newline between attributes
            self.eat(&TokenKind::Newline);
            attrs.push(Attr { name, args, line });
        }
        attrs
    }

    pub(crate) fn collect_attr_arg(&mut self) -> String {
        let mut parts = Vec::new();
        loop {
            match self.peek().clone() {
                TokenKind::Comma | TokenKind::RParen | TokenKind::Eof | TokenKind::Newline => break,
                TokenKind::Ident(s) => { parts.push(s.clone()); self.advance(); }
                TokenKind::Eq => { parts.push("=".to_string()); self.advance(); }
                TokenKind::Str(s) => { parts.push(format!("\"{}\"", s)); self.advance(); }
                TokenKind::Int(n) => { parts.push(n.to_string()); self.advance(); }
                // Preserve `::` path separators (e.g. `thiserror::Error`, `std::fmt::Display`)
                TokenKind::Colon => { parts.push(":".to_string()); self.advance(); }
                // Reconstruct interpolated strings verbatim so that `@error("not found: {0}")`
                // passes `"not found: {0}"` to Rust rather than losing the `{0}` hole.
                TokenKind::StringInterp(segs) => {
                    let mut raw = String::from("\"");
                    for seg in &segs {
                        match seg {
                            RawInterpPart::Lit(s) => raw.push_str(s),
                            RawInterpPart::Hole(expr) => { raw.push('{'); raw.push_str(expr); raw.push('}'); }
                            RawInterpPart::HoleFormatted(expr, fmt) => {
                                raw.push('{'); raw.push_str(expr); raw.push(':'); raw.push_str(fmt); raw.push('}');
                            }
                        }
                    }
                    raw.push('"');
                    parts.push(raw);
                    self.advance();
                }
                _ => { self.advance(); }
            }
        }
        parts.join("")
    }

    pub(crate) fn parse_fn_signature(&mut self) -> Result<FnSignature, ParseError> {
        let line = self.line();
        let mutating = if self.check(&TokenKind::Req) {
            self.advance();
            false
        } else {
            self.expect(&TokenKind::Def)?;
            true
        };
        let (return_ty, _qualifier, name) = self.parse_fn_head()?;
        let (type_params, _) = self.parse_type_params();
        let params = self.parse_params()?;
        let throws = self.eat(&TokenKind::Throws);
        let task   = self.eat(&TokenKind::Task);
        self.expect_newline()?;
        Ok(FnSignature { name, params, return_ty, throws, task, stream: false, mutating, type_params, line })
    }

    /// Parse a trait member that is either an abstract signature or a default implementation.
    /// Returns `Left(FnSignature)` for abstract, `Right(FnDecl)` for default (has a body).
    pub(crate) fn parse_fn_signature_or_default(&mut self) -> Result<Either<FnSignature, FnDecl>, ParseError> {
        let line = self.line();
        // `task` may come BEFORE `def`/`req`: `task req f():` (preferred) or after params (legacy)
        let prefix_task = self.eat(&TokenKind::Task);
        // Shorthand: `task RetType name(` — `def` is implicit, same as in parse_fn_decl.
        let shorthand = prefix_task
            && !self.check(&TokenKind::Def)
            && !self.check(&TokenKind::Req);
        let mutating = if shorthand {
            true // shorthand defaults to `def` (mutating)
        } else if self.check(&TokenKind::Req) {
            self.advance(); false
        } else {
            self.expect(&TokenKind::Def)?; true
        };
        let (return_ty, _qualifier, name) = self.parse_fn_head()?;
        let (type_params, _) = self.parse_type_params();
        let params = self.parse_params()?;
        let task = prefix_task;
        let mut throws = self.eat(&TokenKind::Throws);
        let mut throws_ty: Option<Type> = None;
        if throws {
            throws_ty = self.parse_throws_type()?;
        }
        if !throws { throws = self.eat(&TokenKind::Throws); }
        // Colon → default body; newline only → abstract signature
        if self.eat(&TokenKind::Colon) {
            let body = if self.check(&TokenKind::Newline) || self.check(&TokenKind::Eof) {
                self.expect_newline()?;
                self.parse_block()?
            } else {
                let expr_line = self.line();
                let stmt = self.parse_inline_stmt()?;
                self.expect_newline_soft();
                match stmt {
                    Stmt::Expr(ref e) if !matches!(e.kind, ExprKind::Assign(..)) =>
                        vec![Stmt::Return(ReturnStmt { value: Some(e.clone()), line: expr_line })],
                    other => vec![other],
                }
            };
            Ok(Either::Right(FnDecl {
                name, qualifier: None, params, return_ty, body,
                is_pub: false, throws, task, stream: false, mutating,
                is_native: false, throws_ty,
                type_params, where_clause: vec![], attrs: vec![], line,
            }))
        } else {
            self.expect_newline()?;
            Ok(Either::Left(FnSignature { name, params, return_ty, throws, task, stream: false, mutating, type_params, line }))
        }
    }
}
