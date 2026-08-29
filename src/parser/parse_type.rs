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
    // ─── Types ──────────────────────────────────────────────────────────────

    pub(crate) fn kind_is_type_start(&self, kind: &TokenKind) -> bool {
        matches!(kind,
            TokenKind::Ident(_)
            | TokenKind::LBracket
            | TokenKind::LBrace
            | TokenKind::LParen
            | TokenKind::Void
        )
    }

    pub(crate) fn parse_type(&mut self) -> Result<Type, ParseError> {
        let line = self.line();
        let _col = self.col();

        // `mut Type` / `mut Type&` — a "mut"? prefix is grammar-legal on any
        // `type` (see `spec/grammar.bnf`'s `type` production), nested
        // anywhere: tuple slot, generic argument, array element, dict
        // value. The checker (not the parser) restricts which POSITIONS
        // actually grant the permission. The let_stmt/destructure/field_decl
        // *statement-level* leading `mut` keyword is consumed by their own
        // dedicated code before ever calling into `parse_type`, so a `mut`
        // reaching here always means this nested, in-type-position prefix.
        if self.check(&TokenKind::Mut) {
            let next = self.tokens.get(self.pos + 1).map(|t| &t.kind);
            if next.map(|k| self.kind_is_type_start(k)).unwrap_or(false) {
                self.advance(); // consume `mut`
                let inner = self.parse_type()?;
                return Ok(Self::wrap_type_mut(inner));
            }
        }

        // Prefix modifiers for function types: `req`, `def`, `task`, or combinations.
        // `req int (int)` → Fn (pure), `def int (int)` / `int (int)` → FnMut (default),
        // `task int (int)` → async FnMut, `req task int (int)` → async Fn, etc.
        let prefix_req  = self.eat(&TokenKind::Req);
        let prefix_def  = if !prefix_req { self.eat(&TokenKind::Def) } else { false };
        let prefix_task = self.eat(&TokenKind::Task);
        let _ = prefix_def; // `def` is the default — consumed to allow explicit annotation

        if prefix_req || prefix_task || prefix_def {
            let ret_ty = self.parse_type_base(line)?;
            self.expect(&TokenKind::LParen)?;
            let mut params = Vec::new();
            if !matches!(self.peek(), TokenKind::RParen) {
                params.push(self.parse_type()?);
                while self.eat(&TokenKind::Comma) {
                    if matches!(self.peek(), TokenKind::RParen) { break; }
                    params.push(self.parse_type()?);
                }
            }
            self.expect(&TokenKind::RParen)?;
            let fn_throws = self.eat(&TokenKind::Throws);
            return Ok(Type::Fn(Some(Box::new(ret_ty)), params, fn_throws, prefix_task, prefix_req));
        }

        let base_ty = self.parse_type_base(line)?;

        // Check if next token is LParen — could be a function type: `ReturnType (ParamTypes)`
        let ty = if matches!(self.peek(), TokenKind::LParen) {
            let next_after_lparen = self.tokens.get(self.pos + 1).map(|t| &t.kind);
            let is_fn_type = matches!(next_after_lparen, Some(TokenKind::RParen))
                || next_after_lparen.map(|k| self.kind_is_type_start(k)).unwrap_or(false);
            if is_fn_type {
                self.advance(); // consume LParen
                let mut params = Vec::new();
                if !matches!(self.peek(), TokenKind::RParen) {
                    params.push(self.parse_type()?);
                    while self.eat(&TokenKind::Comma) {
                        if matches!(self.peek(), TokenKind::RParen) { break; }
                        params.push(self.parse_type()?);
                    }
                }
                self.expect(&TokenKind::RParen)?;
                let fn_task   = self.eat(&TokenKind::Task);
                let fn_throws = self.eat(&TokenKind::Throws);
                // No prefix → def (default, not pure)
                Type::Fn(Some(Box::new(base_ty)), params, fn_throws, fn_task, false)
            } else {
                base_ty
            }
        } else {
            base_ty
        };

        // Apply ownership qualifier: `Dog'rc`, `[Int]'static`, etc.
        let ty = self.parse_type_qualifier(ty)?;

        // Optional suffix `?`, or optional borrow `?&` → &Option<T>
        if self.eat(&TokenKind::Question) {
            if self.check(&TokenKind::Ampersand) {
                self.advance(); // consume `&`
                Ok(Type::Qualified(Box::new(ty), OwnerQual::BorrowOption))
            } else {
                Ok(Type::Optional(Box::new(ty)))
            }
        } else {
            Ok(ty)
        }
    }

    /// Speculatively parses a comma-separated labeled-axis list for the new
    /// `[T, width, height]` / `[T, width = W, height = H]` form
    /// (docs/array-multidim-proposal.md), called with `self.pos` right at the
    /// first token after the comma following the array's element type.
    ///
    /// `Ok(None)` means "not this form after all" — `self.pos` is restored to
    /// where it started, so the caller's existing `[T, N]` (int literal) /
    /// `[T, <expr>]` (`ArrayNExpr`) parsing runs completely unchanged. This
    /// happens when the first item is a lone identifier immediately followed
    /// by `]` (the pre-existing meaning of a bare `[T, N]` — a reference to a
    /// const generic param, unchanged) or by anything other than `=`/`,`
    /// (the start of a larger arithmetic expression, e.g. `[T, N + 1]`).
    ///
    /// Once a second comma-separated item or an `=` is seen, the input is
    /// unambiguously the new form — from that point on, a malformed list
    /// (fewer than 2 axes, or mixing fixed and dynamic axes — D1) is reported
    /// as a hard parse error rather than rolled back, since falling through to
    /// `parse_expr()` on e.g. `width = 16` would risk silently parsing it as
    /// an assignment expression instead of failing loudly.
    fn try_parse_labeled_axes(&mut self) -> Result<Option<Vec<LabeledAxis>>, ParseError> {
        let saved = self.pos;
        let first_label = match self.peek().clone() {
            TokenKind::Ident(s) => s,
            _ => return Ok(None),
        };
        self.advance(); // consume the identifier

        if self.eat(&TokenKind::Eq) {
            let size_expr = self.parse_expr()?;
            let mut axes = vec![LabeledAxis { label: first_label, size: Some(ConstExpr(Box::new(size_expr))) }];
            if self.check(&TokenKind::RBracket) {
                return Err(ParseError::Generic {
                    line: self.line(), col: self.col(),
                    msg: "labeled array types need at least 2 axes — add another \
                          `label = size`, or remove the label to use a plain [T, N] array"
                        .to_string(),
                    len: self.tok_len(),
                });
            }
            self.expect(&TokenKind::Comma)?;
            self.parse_labeled_axes_rest(&mut axes, true)?;
            Ok(Some(axes))
        } else if self.check(&TokenKind::Comma) {
            self.advance(); // consume comma
            if self.check(&TokenKind::RBracket) {
                return Err(ParseError::Generic {
                    line: self.line(), col: self.col(),
                    msg: "labeled array types need at least 2 axes — add another axis \
                          label, or remove the trailing comma to use a plain [T, N] array"
                        .to_string(),
                    len: self.tok_len(),
                });
            }
            let mut axes = vec![LabeledAxis { label: first_label, size: None }];
            self.parse_labeled_axes_rest(&mut axes, false)?;
            Ok(Some(axes))
        } else {
            // Lone identifier followed by `]` (legacy const-generic reference) or by
            // an operator (start of a larger legacy expression) — roll back.
            self.pos = saved;
            Ok(None)
        }
    }

    /// Parses the remaining comma-separated axis entries after the first one
    /// (already pushed by the caller), up to (not including) the closing `]`.
    /// `expect_fixed` pins whether every remaining entry must have `= expr`
    /// (fixed-shape list) or must not (dynamic-shape list) — D1: an axis list
    /// is never a mix of the two, checked entry by entry as they're parsed.
    /// A single trailing comma before `]` is allowed, matching `arg_list`.
    fn parse_labeled_axes_rest(&mut self, axes: &mut Vec<LabeledAxis>, expect_fixed: bool) -> Result<(), ParseError> {
        loop {
            let label = self.expect_ident()?;
            let has_eq = self.eat(&TokenKind::Eq);
            if has_eq != expect_fixed {
                return Err(ParseError::Generic {
                    line: self.line(), col: self.col(),
                    msg: "labeled array axes must be all dynamic ([T, a, b]) or all \
                          fixed ([T, a = A, b = B]) — mixing the two is not supported"
                        .to_string(),
                    len: self.tok_len(),
                });
            }
            let size = if has_eq { Some(ConstExpr(Box::new(self.parse_expr()?))) } else { None };
            axes.push(LabeledAxis { label, size });
            if !self.eat(&TokenKind::Comma) { break; }
            if self.check(&TokenKind::RBracket) { break; } // trailing comma
        }
        Ok(())
    }

    pub(crate) fn parse_type_base(&mut self, line: usize) -> Result<Type, ParseError> {
        // `<Trait>` — impl Trait shorthand (static dispatch).
        if self.check(&TokenKind::Lt) {
            let is_impl_shorthand = matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind),
                Some(TokenKind::Ident(_)))
                && matches!(self.tokens.get(self.pos + 2).map(|t| &t.kind),
                    Some(TokenKind::Gt));
            if is_impl_shorthand {
                self.advance(); // consume `<`
                let inner = self.parse_type()?;
                self.expect(&TokenKind::Gt)?;
                return Ok(Type::Impl(Box::new(inner)));
            }
        }
        match self.peek().clone() {
            TokenKind::Ident(s) => {
                let s = s.clone();
                self.advance();
                let base = match s.as_str() {
                    "Int"     => Type::Int,
                    "Uint"    => Type::Uint,
                    "Uint8"   => Type::Uint8,
                    "Int8"    => Type::Int8,
                    "Int16"   => Type::Int16,
                    "Int32"   => Type::Int32,
                    "Int64"   => Type::Int64,
                    "Int128"  => Type::Int128,
                    "Uint16"  => Type::Uint16,
                    "Uint32"  => Type::Uint32,
                    "Uint64"  => Type::Uint64,
                    "Uint128" => Type::Uint128,
                    "Float"   => Type::Float64,
                    "Float32" => Type::Float32,
                    "Float64" => Type::Float64,
                    "String" => Type::Str,
                    "Bool"   => Type::Bool,
                    "Nil"    => Type::Nil,
                    "Never"  => Type::Never,
                    "Self" => {
                        if self.check(&TokenKind::Dot) {
                            self.advance(); // consume "."
                            let assoc_name = self.expect_ident()?;
                            return Ok(Type::SelfAssoc(assoc_name));
                        }
                        Type::Named("Self".to_string())
                    }
                    _ if s.len() == 1 && s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) => {
                        Type::TypeParam(s.clone())
                    }
                    _ => Type::Named(s.clone()),
                };
                // Generic type args: Foo<T, U> — supports the same extended syntax as
                // struct/fn type-param declarations: lifetime args `&a` and optional
                // `T as Trait` bounds (bounds are silently ignored at use sites).
                let built = if self.eat(&TokenKind::Lt) {
                    let mut args = Vec::new();
                    args.push(self.parse_generic_type_arg()?);
                    while self.eat(&TokenKind::Comma) {
                        if self.check(&TokenKind::Gt) { break; }
                        args.push(self.parse_generic_type_arg()?);
                    }
                    self.expect(&TokenKind::Gt)?;
                    let name = match &base {
                        Type::Named(n) => n.clone(),
                        _ => s.clone(),
                    };
                    Type::Generic(name, args)
                } else {
                    base
                };
                // Associated type access: `LinkedList.Index` or `Tree<T>.Node`
                // Only on Named/Generic — not on TypeParam (single uppercase letter).
                if matches!(built, Type::Named(_) | Type::Generic(_, _))
                    && self.check(&TokenKind::Dot)
                    && matches!(
                        self.tokens.get(self.pos + 1).map(|t| &t.kind),
                        Some(TokenKind::Ident(_))
                    )
                {
                    self.advance(); // consume "."
                    let assoc_name = self.expect_ident()?;
                    return Ok(Type::AssocOf(Box::new(built), assoc_name));
                }
                Ok(built)
            }
            TokenKind::Void => {
                self.advance();
                Ok(Type::Void)
            }
            TokenKind::LBracket => {
                self.advance();
                let inner = self.parse_type()?;
                let inner = self.parse_type_qualifier(inner)?;
                if self.eat(&TokenKind::Comma) {
                    // New labeled multi-dim form: [T, width, height] / [T, width=W, height=H]
                    // (docs/array-multidim-proposal.md). Only attempted when the next token
                    // is a bare identifier — try_parse_labeled_axes itself rolls back
                    // (returns Ok(None)) and leaves `self.pos` unchanged whenever the input
                    // turns out to be the legacy single-item form after all (`[T, N]` /
                    // `[T, N + 1]`), so the Int-literal and parse_expr() fallbacks below run
                    // completely unchanged in that case.
                    if matches!(self.peek(), TokenKind::Ident(_)) {
                        if let Some(axes) = self.try_parse_labeled_axes()? {
                            self.expect(&TokenKind::RBracket)?;
                            return Ok(Type::LabeledArray(Box::new(inner), axes));
                        }
                    }
                    // Size can be an integer literal OR an expression over const generic params.
                    if let TokenKind::Int(n) = self.peek().clone() {
                        // Peek ahead: if the next token after the int is `]`, it's a plain literal.
                        if matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::RBracket)) {
                            self.advance(); // consume int
                            self.expect(&TokenKind::RBracket)?;
                            return Ok(Type::ArrayN(Box::new(inner), n as usize));
                        }
                    }
                    // Otherwise parse as a full expression (handles `W * H`, `N + 1`, etc.).
                    let size_expr = self.parse_expr()?;
                    self.expect(&TokenKind::RBracket)?;
                    return Ok(Type::ArrayNExpr(Box::new(inner), ConstExpr(Box::new(size_expr))));
                }
                self.expect(&TokenKind::RBracket)?;
                Ok(Type::Array(Box::new(inner)))
            }
            // Integer literal in type-argument position: e.g. `GameOfLife<64, 64>`.
            TokenKind::Int(n) => {
                self.advance();
                Ok(Type::ConstInt(n))
            }
            TokenKind::LParen => {
                self.advance();
                let mut types = Vec::new();
                while !self.check(&TokenKind::RParen) && !self.check(&TokenKind::Eof) {
                    types.push(self.parse_type()?);
                    if !self.eat(&TokenKind::Comma) { break; }
                }
                self.expect(&TokenKind::RParen)?;
                Ok(Type::Tuple(types))
            }
            TokenKind::LBrace => {
                self.advance();
                let first = self.parse_type()?;
                if self.eat(&TokenKind::Eq) {
                    let val = self.parse_type()?;
                    let val = self.parse_type_qualifier(val)?;
                    self.expect(&TokenKind::RBrace)?;
                    Ok(Type::Dict(Box::new(first), Box::new(val)))
                } else {
                    let first = self.parse_type_qualifier(first)?;
                    self.expect(&TokenKind::RBrace)?;
                    Ok(Type::Set(Box::new(first)))
                }
            }
            _ => Err(ParseError::Generic {
                line, col: self.col(),
                msg: format!("expected type, got {:?}", self.peek()), len: self.tok_len(),
            }),
        }
    }

    /// Replace the innermost base type in a qualified type with `new_base`.
    /// Used when we parse `let name'qual = Ctor()` — the placeholder `_` is replaced
    /// by the actual constructor type once we see the RHS.
    pub(crate) fn replace_type_base(ty: Type, new_base: Type) -> Type {
        match ty {
            Type::Qualified(inner, qual) => {
                Type::Qualified(Box::new(Self::replace_type_base(*inner, new_base)), qual)
            }
            Type::Named(ref n) if n == "_" => new_base,
            other => other,
        }
    }

    /// `Dog'owned`   → Qualified(Dog, Owned)
    /// `Dog'new`     → Qualified(Dog, Union([Owned, Shared, Actor, Guard])) — replaces bare tick
    /// `Dog'copy`    → Qualified(Dog, Copy)
    /// `Dog'const`   → Qualified(Dog, Const)
    /// `Dog'shared`  → Qualified(Dog, Shared) — Arc<T> (multi) / Rc<T> (single), threading-aware
    /// `Dog'weak`    → Qualified(Dog, Weak)   — Weak<T>, non-owning ref
    pub(crate) fn parse_type_qualifier(&mut self, ty: Type) -> Result<Type, ParseError> {
        if self.check(&TokenKind::Ampersand) {
            // Peek at the token after `&` without consuming anything yet.
            let next = self.tokens.get(self.pos + 1).map(|t| t.kind.clone());
            match next {
                Some(TokenKind::Ident(ref s)) if s == "weak" => {
                    self.advance(); self.advance();
                    return Ok(Type::Qualified(Box::new(ty), OwnerQual::BorrowWeak));
                }
                // `T&a name` — single lowercase letter is a lifetime only when another
                // identifier follows it (the actual parameter name). If the letter is the
                // last ident before `,` / `)` / `:`, it is the param name → bare borrow.
                Some(TokenKind::Ident(ref s))
                    if s.len() == 1
                        && s.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false) =>
                {
                    // pos+0 = `&`, pos+1 = letter; check pos+2 for another ident (param name)
                    let token_after_letter = self.tokens.get(self.pos + 2).map(|t| &t.kind);
                    if matches!(token_after_letter, Some(TokenKind::Ident(_))) {
                        // True lifetime: `T& a name` — consume `&` and the letter
                        self.advance(); // consume `&`
                        let lt = if let TokenKind::Ident(s) = self.peek().clone() { s } else { unreachable!() };
                        self.advance(); // consume lifetime letter
                        return Ok(Type::Qualified(Box::new(ty), OwnerQual::Lifetime(lt)));
                    } else {
                        // Bare borrow: `T& c` — `c` is the param name, only consume `&`
                        self.advance(); // consume `&`
                        return Ok(Type::Qualified(Box::new(ty), OwnerQual::Borrow));
                    }
                }
                // Bare `&` followed by an identifier (the param / field name) — generic borrow.
                // Used with type aliases: `use Node as Tree'shared; def walk(Node& n)`.
                // The transpiler resolves the alias to determine &Rc<T> vs &Arc<T>.
                Some(TokenKind::Ident(_)) => {
                    self.advance(); // consume `&`
                    return Ok(Type::Qualified(Box::new(ty), OwnerQual::Borrow));
                }
                // `&` with nothing to disambiguate against because there is no
                // per-element name in this position at all — a tuple slot or
                // generic argument, e.g. `(Position&, mut Velocity&)`,
                // `Query<Position&, mut Velocity&>` (docs/book.md
                // §2's Bevy-ECS motivating case). Every other `&`-parsing arm
                // above assumes a trailing name (`T& name`, the param/field/
                // let_stmt convention) — there is none to assume here, so a
                // bare borrow is the only possible reading.
                Some(TokenKind::Comma | TokenKind::RParen | TokenKind::RBracket
                    | TokenKind::RBrace | TokenKind::Gt) | None => {
                    self.advance(); // consume `&`
                    return Ok(Type::Qualified(Box::new(ty), OwnerQual::Borrow));
                }
                _ => {} // not a borrow qualifier — fall through
            }
        }
        if !self.eat(&TokenKind::Tick) {
            return Ok(ty);
        }
        let qual = match self.peek().clone() {
            TokenKind::Ident(ref s) => match s.as_str() {
                "shared" => { self.advance(); OwnerQual::Shared }
                "weak"   => { self.advance(); OwnerQual::Weak }
                "inline" => { self.advance(); OwnerQual::Inline }
                "owned"  => { self.advance(); OwnerQual::Owned }
                // GPU memory qualifiers (kernel-context and host-context).
                "unified" => { self.advance(); OwnerQual::GpuUnified }
                "global"  => { self.advance(); OwnerQual::GpuGlobal }
                "local"   => { self.advance(); OwnerQual::GpuLocal }
                "const"   => { self.advance(); OwnerQual::GpuConst }
                "gpu"    => {
                    self.advance();
                    // `T'gpu'unified`, `T'gpu'global` — host-side GPU qualifiers.
                    // `'const` is deliberately excluded: it has no host access (like `'local`),
                    // so a host-context binding could never be read from or written to.
                    if self.eat(&TokenKind::Tick) {
                        match self.peek().clone() {
                            TokenKind::Ident(ref s) => match s.as_str() {
                                "unified" => { self.advance(); OwnerQual::GpuUnified }
                                "global"  => { self.advance(); OwnerQual::GpuGlobal }
                                _ => return Err(ParseError::Generic {
                                    msg: "expected 'unified or 'global after 'gpu".into(),
                                    line: self.line(), col: self.col(), len: self.tok_len(),
                                }),
                            },
                            _ => return Err(ParseError::Generic {
                                msg: "expected 'unified or 'global after 'gpu".into(),
                                line: self.line(), col: self.col(), len: self.tok_len(),
                            }),
                        }
                    } else {
                        return Err(ParseError::Generic {
                            msg: "'gpu must be followed by 'unified or 'global".into(),
                            line: self.line(), col: self.col(), len: self.tok_len(),
                        });
                    }
                }
                "surface" => { self.advance(); OwnerQual::GpuSurface }
                "actor"  => {
                    self.advance();
                    // `T'actor'task` → ActorTask  /  `T'actor'global` → GpuActorGlobal  /
                    // `T'actor'unified` → GpuActorUnified  /  bare `T'actor` → Actor (CPU-side;
                    // in kernel-struct field position this is reinterpreted as block-shared
                    // memory by parse_kernel_field, replacing the old 'sync spelling).
                    // `T'actor'weak` is deliberately left un-consumed here — it's handled
                    // by the generic Shared|Actor|Guard chained-`'weak` logic below, which
                    // expects to see the tick itself still unconsumed.
                    let next_is_weak = matches!(
                        self.tokens.get(self.pos + 1).map(|t| &t.kind),
                        Some(TokenKind::Ident(s)) if s == "weak"
                    );
                    if !next_is_weak && self.eat(&TokenKind::Tick) {
                        if matches!(self.peek(), TokenKind::Task) { self.advance(); OwnerQual::ActorTask }
                        else if matches!(self.peek(), TokenKind::Ident(ref s) if s == "global") { self.advance(); OwnerQual::GpuActorGlobal }
                        else if matches!(self.peek(), TokenKind::Ident(ref s) if s == "unified") { self.advance(); OwnerQual::GpuActorUnified }
                        else { return Err(ParseError::Generic { msg: "expected 'task, 'global, or 'unified after 'actor'".into(), line: self.line(), col: self.col(), len: self.tok_len() }); }
                    } else {
                        OwnerQual::Actor
                    }
                }
                // Named qualifier groups — sugar for qualifier unions.
                "one"  => { self.advance(); OwnerQual::Union(vec![OwnerQual::Inline, OwnerQual::Owned]) }
                "many" => { self.advance(); OwnerQual::Union(vec![OwnerQual::Shared, OwnerQual::Actor, OwnerQual::Guard]) }
                "mut"  => { self.advance(); OwnerQual::Union(vec![OwnerQual::Inline, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::Guard]) }
                "req"  => { self.advance(); OwnerQual::Union(vec![OwnerQual::Shared]) }
                // Bare tick with no recognized qualifier word after it is no longer
                // supported — `T'` used to silently default to `'owned` (don't consume
                // the following ident, e.g. the variable name in `let BigData' backup`).
                // Write the qualifier explicitly; `'new` covers what bare tick used to mean
                // ("any indirection, inferred" — see OwnerQual::Union's doc comment).
                _ => return Err(ParseError::Generic {
                    msg: "expected an ownership qualifier after ' (e.g. 'owned, 'shared, 'actor, 'guard, 'inline, 'new, 'weak, 'copy, 'const) — bare tick is no longer supported, use 'new for \"any indirection, inferred\"".into(),
                    line: self.line(), col: self.col(), len: self.tok_len(),
                }),
            },
            // `T'new` — `new` is a keyword, not an ident: candidate-set pseudo-qualifier,
            // "any indirection, inferred" ('inline excluded from the candidate set) —
            // replaces the old bare-tick `T'` spelling. NOT a caller-facing acceptance
            // group like 'one/'many/'mut/'req (see OwnerQual::Union's doc comment) —
            // it must keep narrowing by usage on local variables too, not just parameters.
            TokenKind::New => { self.advance(); OwnerQual::Union(OwnerQual::NEW_MEMBERS.to_vec()) }
            // `T'guard` — `guard` is a reserved keyword, not an ident: Arc<std::sync::RwLock<T>>
            TokenKind::Guard => {
                self.advance();
                // `T'guard'task` → GuardTask. `T'guard'weak` is deliberately left
                // un-consumed here — handled by the generic chained-`'weak` logic below.
                let next_is_weak = matches!(
                    self.tokens.get(self.pos + 1).map(|t| &t.kind),
                    Some(TokenKind::Ident(s)) if s == "weak"
                );
                if !next_is_weak && self.eat(&TokenKind::Tick) {
                    if matches!(self.peek(), TokenKind::Task) { self.advance(); OwnerQual::GuardTask }
                    else { return Err(ParseError::Generic { msg: "expected 'task after 'guard'".into(), line: self.line(), col: self.col(), len: self.tok_len() }); }
                } else {
                    OwnerQual::Guard
                }
            }
            // `T'task` — alias for `T'actor'task`: Arc<tokio::sync::Mutex<T>>
            TokenKind::Task => { self.advance(); OwnerQual::ActorTask }
            // Truly bare tick — nothing recognizable follows `'` at all. Same rule as
            // the ident catch-all above: no longer supported, `'new` is the replacement.
            _ => return Err(ParseError::Generic {
                msg: "expected an ownership qualifier after ' (e.g. 'owned, 'shared, 'actor, 'guard, 'inline, 'new, 'weak, 'copy, 'const) — bare tick is no longer supported, use 'new for \"any indirection, inferred\"".into(),
                line: self.line(), col: self.col(), len: self.tok_len(),
            }),
        };
        // `T'inline|owned|actor` — qualifier union (pipe-separated list).
        // A Union qualifier is a Boring-level constraint; emits as a plain generic in Rust.
        // Also handles the case where the first qual was already a Union (named group).
        let qual = if matches!(self.peek(), TokenKind::Pipe) && !matches!(qual, OwnerQual::Union(_)) {
            // Start a union from the single qualifier already parsed.
            let mut members = vec![qual];
            while self.eat(&TokenKind::Pipe) {
                let member = match self.peek().clone() {
                    TokenKind::Ident(ref s) => match s.as_str() {
                        "inline" => { self.advance(); OwnerQual::Inline }
                        "owned"  => { self.advance(); OwnerQual::Owned }
                        "shared" => { self.advance(); OwnerQual::Shared }
                        "actor"  => { self.advance(); OwnerQual::Actor }
                        _        => break,
                    },
                    TokenKind::Guard => { self.advance(); OwnerQual::Guard }
                    _ => break,
                };
                members.push(member);
            }
            OwnerQual::Union(members)
        } else {
            qual
        };
        // Postfix `&` after a qualifier.
        // `T'shared&`, `T'actor&`, `T'guard&`, `T'owned&` are removed — auto-ref handles
        // 'shared/'actor/'guard in parameter position, and Counter& is the universal borrow.
        if self.check(&TokenKind::Ampersand) {
            // Check for lifetime immediately after `&`: `T'qual&a name`
            let next_is_lifetime = matches!(
                self.tokens.get(self.pos + 1),
                Some(t) if matches!(&t.kind,
                    TokenKind::Ident(s)
                        if s.len() == 1
                            && s.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false))
            ) && matches!(
                self.tokens.get(self.pos + 2).map(|t| &t.kind),
                Some(TokenKind::Ident(_))
            );
            if next_is_lifetime {
                self.advance(); // consume `&`
                let lt = if let TokenKind::Ident(s) = self.peek().clone() { s } else { unreachable!() };
                self.advance(); // consume lifetime letter
                let inner = Type::Qualified(Box::new(ty), qual);
                return Ok(Type::Qualified(Box::new(inner), OwnerQual::Lifetime(lt)));
            }
            if qual == OwnerQual::Weak {
                self.advance();
                return Ok(Type::Qualified(Box::new(ty), OwnerQual::BorrowWeak));
            }
        }
        let qualified = Type::Qualified(Box::new(ty), qual.clone());
        // `T'shared'weak`, `T'actor'weak` — weak ref on any ref-counted type.
        // `'weak` is a second-level qualifier: Qualified(Qualified(T, Shared|Actor|Guard), Weak).
        if matches!(qual, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard)
            && self.check(&TokenKind::Tick) {
                let after_tick = self.tokens.get(self.pos + 1).map(|t| t.kind.clone());
                if matches!(after_tick, Some(TokenKind::Ident(ref s)) if s == "weak") {
                    self.advance(); // consume `'`
                    self.advance(); // consume `weak`
                    return Ok(Type::Qualified(Box::new(qualified), OwnerQual::Weak));
                }
            }
        Ok(qualified)
    }
}

// ─── Standalone helpers ──────────────────────────────────────────────────────

/// Convert an expression (from a closure param list) to a Param node.
/// Handles `Var("name")` → unnamed param, `Type name` patterns are already
/// handled upstream; here we just treat simple var expressions as param names.
pub(crate) fn expr_to_param(expr: &Expr, line: usize, col: usize) -> Param {
    let name = match &expr.kind {
        ExprKind::Var(n) => n.clone(),
        _ => "_".to_string(),
    };
    Param { name, ty: None, mutable: false, rebindable: false, owned: false, variadic: false, default: None, line, col }
}

pub(crate) fn check_no_return(stmts: &[Stmt], context: &str) -> Result<(), ParseError> {
    for stmt in stmts {
        if let Stmt::Return(ret) = stmt {
            return Err(ParseError::Generic {
                line: ret.line, col: 0,
                msg: format!("last expression (no 'return' allowed in {})", context), len: 1,
            });
        }
    }
    Ok(())
}

// ─── Associated type helpers ─────────────────────────────────────────────────

pub(crate) fn resolve_assoc_in_type(ty: Type, names: &[String]) -> Type {
    match ty {
        Type::Named(ref s) if names.contains(s) => Type::SelfAssoc(s.clone()),
        Type::Optional(inner) => Type::Optional(Box::new(resolve_assoc_in_type(*inner, names))),
        Type::Array(inner) => Type::Array(Box::new(resolve_assoc_in_type(*inner, names))),
        Type::Tuple(elems) => Type::Tuple(elems.into_iter().map(|t| resolve_assoc_in_type(t, names)).collect()),
        Type::Dict(k, v) => Type::Dict(Box::new(resolve_assoc_in_type(*k, names)), Box::new(resolve_assoc_in_type(*v, names))),
        Type::Set(inner) => Type::Set(Box::new(resolve_assoc_in_type(*inner, names))),
        Type::Qualified(inner, q) => Type::Qualified(Box::new(resolve_assoc_in_type(*inner, names)), q),
        Type::Fn(ret, params, throws, task, req) => Type::Fn(
            ret.map(|r| Box::new(resolve_assoc_in_type(*r, names))),
            params.into_iter().map(|t| resolve_assoc_in_type(t, names)).collect(),
            throws, task, req,
        ),
        Type::Generic(name, args) => Type::Generic(name, args.into_iter().map(|t| resolve_assoc_in_type(t, names)).collect()),
        Type::Dyn(inner) => Type::Dyn(Box::new(resolve_assoc_in_type(*inner, names))),
        Type::Impl(inner) => Type::Impl(Box::new(resolve_assoc_in_type(*inner, names))),
        other => other,
    }
}

pub(crate) fn resolve_assoc_in_sig(sig: FnSignature, names: &[String]) -> FnSignature {
    let return_ty = sig.return_ty.map(|t| resolve_assoc_in_type(t, names));
    let params = sig.params.into_iter().map(|p| Param {
        ty: p.ty.map(|t| resolve_assoc_in_type(t, names)), ..p
    }).collect();
    FnSignature { return_ty, params, ..sig }
}

pub(crate) fn resolve_assoc_in_fn(decl: FnDecl, names: &[String]) -> FnDecl {
    let return_ty = decl.return_ty.map(|t| resolve_assoc_in_type(t, names));
    let params = decl.params.into_iter().map(|p| Param {
        ty: p.ty.map(|t| resolve_assoc_in_type(t, names)), ..p
    }).collect();
    FnDecl { return_ty, params, ..decl }
}
