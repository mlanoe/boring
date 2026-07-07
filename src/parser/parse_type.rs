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
                    "Int"    => Type::Int,
                    "Uint"   => Type::Uint,
                    "Float"  => Type::Float,
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
                    let n = match self.peek().clone() {
                        TokenKind::Int(n) if n >= 0 => { self.advance(); n as usize }
                        _ => return Err(ParseError::Generic {
                            line: self.line(), col: self.col(),
                            msg: "expected integer literal for fixed-size array length".into(), len: self.tok_len(),
                        }),
                    };
                    self.expect(&TokenKind::RBracket)?;
                    return Ok(Type::ArrayN(Box::new(inner), n));
                }
                self.expect(&TokenKind::RBracket)?;
                Ok(Type::Array(Box::new(inner)))
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

    /// `Dog'`        → Qualified(Dog, Owned)
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
                "stack"  => { self.advance(); OwnerQual::Stack }
                "heap"   => { self.advance(); OwnerQual::Owned }
                "new"    => { self.advance(); OwnerQual::New }
                // GPU memory qualifiers (kernel-context and host-context).
                "unified" => { self.advance(); OwnerQual::GpuUnified }
                "global"  => { self.advance(); OwnerQual::GpuGlobal }
                "sync"    => { self.advance(); OwnerQual::GpuSync }
                "local"   => { self.advance(); OwnerQual::GpuLocal }
                "const"   => { self.advance(); OwnerQual::GpuConst }
                "gpu"    => {
                    self.advance();
                    // `T'gpu'unified`, `T'gpu'global`, `T'gpu'const` — host-side GPU qualifiers.
                    if self.eat(&TokenKind::Tick) {
                        match self.peek().clone() {
                            TokenKind::Ident(ref s) => match s.as_str() {
                                "unified" => { self.advance(); OwnerQual::GpuUnified }
                                "global"  => { self.advance(); OwnerQual::GpuGlobal }
                                "const"   => { self.advance(); OwnerQual::GpuConst }
                                _ => return Err(ParseError::Generic {
                                    msg: "expected 'unified, 'global, or 'const after 'gpu".into(),
                                    line: self.line(), col: self.col(), len: self.tok_len(),
                                }),
                            },
                            _ => return Err(ParseError::Generic {
                                msg: "expected 'unified, 'global, or 'const after 'gpu".into(),
                                line: self.line(), col: self.col(), len: self.tok_len(),
                            }),
                        }
                    } else {
                        return Err(ParseError::Generic {
                            msg: "'gpu must be followed by 'unified, 'global, or 'const".into(),
                            line: self.line(), col: self.col(), len: self.tok_len(),
                        });
                    }
                }
                "actor"  => {
                    self.advance();
                    // `T'actor'task` → ActorTask  /  `T'actor'global` → GpuActorGlobal
                    if self.eat(&TokenKind::Tick) {
                        if matches!(self.peek(), TokenKind::Task) { self.advance(); OwnerQual::ActorTask }
                        else if matches!(self.peek(), TokenKind::Ident(ref s) if s == "global") { self.advance(); OwnerQual::GpuActorGlobal }
                        else { return Err(ParseError::Generic { msg: "expected 'task or 'global after 'actor'".into(), line: self.line(), col: self.col(), len: self.tok_len() }); }
                    } else {
                        OwnerQual::Actor
                    }
                }
                // Named qualifier groups — sugar for qualifier unions.
                "one"  => { self.advance(); OwnerQual::Union(vec![OwnerQual::Stack, OwnerQual::Owned]) }
                "many" => { self.advance(); OwnerQual::Union(vec![OwnerQual::Shared, OwnerQual::Actor, OwnerQual::Guard]) }
                "mut"  => { self.advance(); OwnerQual::Union(vec![OwnerQual::Stack, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::Guard]) }
                "req"  => { self.advance(); OwnerQual::Union(vec![OwnerQual::Shared]) }
                _ => OwnerQual::Owned,  // unknown word → bare owned (don't consume it)
            },
            // `T'sync` — `sync` is a keyword, not an ident: block SRAM with auto-barrier
            TokenKind::Sync => { self.advance(); OwnerQual::GpuSync }
            // `T'new` — `new` is a keyword, not an ident: pseudo-qualifier "infer excluding 'stack"
            TokenKind::New => { self.advance(); OwnerQual::New }
            // `T'guard` — `guard` is a reserved keyword, not an ident: Arc<std::sync::RwLock<T>>
            TokenKind::Guard => {
                self.advance();
                // `T'guard'task` → GuardTask
                if self.eat(&TokenKind::Tick) {
                    if matches!(self.peek(), TokenKind::Task) { self.advance(); OwnerQual::GuardTask }
                    else { return Err(ParseError::Generic { msg: "expected 'task after 'guard'".into(), line: self.line(), col: self.col(), len: self.tok_len() }); }
                } else {
                    OwnerQual::Guard
                }
            }
            // `T'task` — alias for `T'actor'task`: Arc<tokio::sync::Mutex<T>>
            TokenKind::Task => { self.advance(); OwnerQual::ActorTask }
            _ => OwnerQual::Owned,
        };
        // `T'stack|heap|actor` — qualifier union (pipe-separated list).
        // A Union qualifier is a Boring-level constraint; emits as a plain generic in Rust.
        // Also handles the case where the first qual was already a Union (named group).
        let qual = if matches!(self.peek(), TokenKind::Pipe) && !matches!(qual, OwnerQual::Union(_)) {
            // Start a union from the single qualifier already parsed.
            let mut members = vec![qual];
            while self.eat(&TokenKind::Pipe) {
                let member = match self.peek().clone() {
                    TokenKind::Ident(ref s) => match s.as_str() {
                        "stack"  => { self.advance(); OwnerQual::Stack }
                        "heap"   => { self.advance(); OwnerQual::Owned }
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
        // `T'shared&`, `T'actor&`, `T'guard&`, `T'heap&` are removed — auto-ref handles
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
            match qual {
                OwnerQual::Weak => {
                    self.advance();
                    return Ok(Type::Qualified(Box::new(ty), OwnerQual::BorrowWeak));
                }
                _ => {}
            }
        }
        let qualified = Type::Qualified(Box::new(ty), qual.clone());
        // `T'shared'weak`, `T'actor'weak` — weak ref on any ref-counted type.
        // `'weak` is a second-level qualifier: Qualified(Qualified(T, Shared|Actor|Guard), Weak).
        if matches!(qual, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard) {
            if self.check(&TokenKind::Tick) {
                let after_tick = self.tokens.get(self.pos + 1).map(|t| t.kind.clone());
                if matches!(after_tick, Some(TokenKind::Ident(ref s)) if s == "weak") {
                    self.advance(); // consume `'`
                    self.advance(); // consume `weak`
                    return Ok(Type::Qualified(Box::new(qualified), OwnerQual::Weak));
                }
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
