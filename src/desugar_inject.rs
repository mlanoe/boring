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

//! Resolves `@inject` (docs/design-notes/boring-di-draft.md §1) against the
//! `@provide` registry and desugars it away — runs once, on the whole
//! `Program`, right after `desugar_array_block` (same pipeline slot, for the
//! same reason: every consumer downstream — checker, transpiler — only ever
//! sees a plain, ordinary `init` it already knows how to handle).
//!
//! ## What this pass actually is
//!
//! Same shape as `desugar_array_block`: this is simultaneously the
//! "resolution" step (building the `(base type) -> provider` registry this
//! design calls for, and matching each `@inject` field against it) and the
//! "desugar" step (rewriting the match away into an ordinary, `init`-based
//! defaulted constructor argument — "the same omit-and-fall-back call-site
//! mechanic Boring already implements for default parameters", per the design
//! doc's §1). A struct with an `@inject` field is never passed through to the
//! checker/transpiler with that attribute still meaningful — either this pass
//! synthesizes a real `init(...)` for it (one parameter per field, the
//! `@inject` field's own parameter defaulting to a call to its resolved
//! provider — verified against a real `Circle(radius = computeDefault())`-
//! style fixture to reuse Boring's *existing* labeled-argument-with-defaults
//! call-site machinery unmodified, `emit_expr.rs`'s
//! `try_emit_labeled_init_call`), or this pass returns a hard `ParseError`
//! (reusing that type purely for its existing `line`/`col`/`len`/`msg`
//! accessors and `main.rs`'s existing `report_error` plumbing, exactly like
//! `desugar_array_block` already does for its own resolution errors).
//!
//! ## Scope of this first implementation (see the design doc's Open
//! Questions/"Before implementation begins" for what's deliberately deferred)
//!
//! - **Same-project (in practice: same parsed `Program`) only** — no `[deps]`
//!   cross-project resolution yet. A provider is visible here exactly when
//!   it's a top-level (or one-level-nested-in-`mod`) `@provide` function in
//!   this same `Program`.
//! - **No `id`/`env`** — the registry key is the base type name alone; two
//!   `@provide` functions for the same type are always ambiguous, with no way
//!   to disambiguate yet.
//! - **`@inject` fields must write an explicit qualifier** — bare-field
//!   qualifier inference (copying a `@singleton` provider's qualifier
//!   verbatim, or running chapter 30's usual inference against a transient
//!   one) needs the whole-program collection-pass-before-layout-is-fixed
//!   machinery the design doc's §2 describes; not attempted yet.
//! - **A struct with an `@inject` field can't also declare its own `init`**
//!   yet — keeps this first cut to the common case (no hand-written
//!   constructor at all) rather than also merging synthesized and
//!   user-written parameter lists.
//!
//! None of this is `@inject`-site-vs-`@provide`-site cross-project or `id`/
//! `env` machinery the checker itself would need to know about later —
//! widening any of the above only ever changes *this* pass's resolution
//! logic, not what a resolved `@inject` field desugars into.

use crate::ast::*;
use crate::parser::ParseError;
use std::collections::HashMap;

/// What `@inject` needs to know about one `@provide` function.
struct Provider {
    /// The provider function's own name — called with no arguments at every
    /// `@inject` site that resolves against it (docs/design-notes/
    /// boring-di-draft.md §1: providers are always zero-arg by construction,
    /// since none of their own parameters could ever be supplied at an
    /// `@inject` call site).
    fn_name: String,
    is_singleton: bool,
    /// The provider's own declared return type, `mut`-stripped — compared
    /// structurally against an explicit `@inject` field's own type when the
    /// provider is `@singleton` (§2: the field must match exactly, since
    /// there's only one physical representation in play).
    return_ty: Type,
    line: usize,
}

type Registry = HashMap<String, Provider>;

fn err(line: usize, col: usize, msg: String) -> ParseError {
    ParseError::Generic { line, col, len: 1, msg }
}

/// Strips `Type::Mut` and any number of `Type::Qualified` layers down to the
/// innermost `Type::Named` — the "base type" `@inject`/`@provide` key on.
fn base_type_name(ty: &Type) -> Option<&str> {
    match ty {
        Type::Named(n) => Some(n.as_str()),
        Type::Qualified(inner, _) => base_type_name(inner),
        Type::Mut(inner) => base_type_name(inner),
        _ => None,
    }
}

/// The outermost ownership qualifier actually carried by a type (skipping a
/// `mut` wrapper, if present) — `None` for a bare/scalar type. For a
/// composed suffix like `T'actor'observed`, this is `Observed` (the outer
/// layer), matching how `boring-ui-draft.md` treats the suffix as what
/// actually decides `@inject`/`@provide` acceptance (docs/design-notes/
/// boring-di-draft.md §2).
fn outer_qualifier(ty: &Type) -> Option<&OwnerQual> {
    match ty {
        Type::Qualified(_, q) => Some(q),
        Type::Mut(inner) => outer_qualifier(inner),
        _ => None,
    }
}

/// §2's accepted-qualifier gate for the `@inject`-site's own field type:
/// `'shared`/`'actor`/`'guard`/`'observed`(-suffixed)/`'static` always legal;
/// `'owned` only when the matched provider isn't `@singleton`; anything else
/// (bare/scalar, `'inline`, `'weak`-suffixed) is a hard error.
fn check_field_qualifier_accepted(
    field: &FieldDecl,
    provider: &Provider,
) -> Result<(), ParseError> {
    let ty = field.ty.without_mut();
    let Some(q) = outer_qualifier(ty) else {
        return Err(err(field.line, field.col, format!(
            "`@inject` field `{}` has no qualifier — `@inject` needs one of `'shared`/`'actor`/\
             `'guard`/`'observed`/`'static`, or `'owned` for a non-`@singleton` provider \
             (docs/design-notes/boring-di-draft.md §2); a bare scalar or plain struct type has \
             nothing for dependency injection to abstract",
            field.name,
        )));
    };
    match q {
        OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Static
        | OwnerQual::Observed => Ok(()),
        OwnerQual::Owned => {
            if provider.is_singleton {
                Err(err(field.line, field.col, format!(
                    "`@inject` field `{}` cannot be `'owned` — the matched provider (`{}`) is \
                     `@singleton`, and `'owned` (`Box<T>`) is exclusive by definition and cannot \
                     be referenced by more than one consumer (docs/design-notes/boring-di-draft.md §2)",
                    field.name, provider.fn_name,
                )))
            } else {
                Ok(())
            }
        }
        OwnerQual::Inline => Err(err(field.line, field.col, format!(
            "`@inject` field `{}` cannot be `'inline` — `@inject` almost always keys on a trait, \
             and a bare trait type is unsized (`dyn Trait` has no `'inline`/no-indirection \
             representation); use `'shared`/`'actor`/`'guard`/`'owned` instead \
             (docs/design-notes/boring-di-draft.md §2)",
            field.name,
        ))),
        OwnerQual::Weak => Err(err(field.line, field.col, format!(
            "`@inject` field `{}` cannot be a `'weak` reference — a weak reference can vanish, \
             and an injected dependency needs to guarantee it stays alive \
             (docs/design-notes/boring-di-draft.md §2)",
            field.name,
        ))),
        _ => Err(err(field.line, field.col, format!(
            "`@inject` field `{}`'s qualifier is not legal here (docs/design-notes/\
             boring-di-draft.md §2) — use `'shared`/`'actor`/`'guard`/`'observed`/`'static`, or \
             `'owned` for a non-`@singleton` provider",
            field.name,
        ))),
    }
}

/// §2: against a `@singleton` provider there's exactly one physical
/// representation in play, so an explicit field's own qualifier must match
/// the provider's return type exactly.
fn check_singleton_qualifier_match(field: &FieldDecl, provider: &Provider) -> Result<(), ParseError> {
    if !provider.is_singleton {
        return Ok(());
    }
    let field_ty = field.ty.without_mut();
    if field_ty != &provider.return_ty {
        return Err(err(field.line, field.col, format!(
            "`{}` is declared `{:?}`, but the only visible provider (`{}`) is `@singleton` and \
             returns `{:?}` — a `@singleton` provider's return type must be matched exactly, \
             since there's only one physical instance in play \
             (docs/design-notes/boring-di-draft.md §2)",
            field.name, field_ty, provider.fn_name, provider.return_ty,
        )));
    }
    Ok(())
}

fn collect_providers(items: &[Item], reg: &mut Registry) -> Result<(), ParseError> {
    for item in items {
        match item {
            Item::Fn(f) if f.attrs.iter().any(|a| a.name == "provide") => {
                let Some(ret_ty) = &f.return_ty else { continue };
                let Some(base) = base_type_name(ret_ty) else { continue };
                let is_singleton = f.attrs.iter().any(|a| a.name == "singleton");
                let entry = Provider {
                    fn_name: f.name.clone(),
                    is_singleton,
                    return_ty: ret_ty.without_mut().clone(),
                    line: f.line,
                };
                if let Some(existing) = reg.get(base) {
                    return Err(err(f.line, f.col, format!(
                        "ambiguous provider for `{}` — `@provide` functions `{}` (line {}) and \
                         `{}` (line {}) both provide this type, with no `id`/`env` to \
                         disambiguate (docs/design-notes/boring-di-draft.md, \"Ambiguity UX\")",
                        base, existing.fn_name, existing.line, f.name, f.line,
                    )));
                }
                reg.insert(base.to_string(), entry);
            }
            Item::Mod(m) => collect_providers(&m.items, reg)?,
            _ => {}
        }
    }
    Ok(())
}

/// Builds the `init` this struct needs so its `@inject` field(s) become
/// omittable, defaulted constructor arguments — one parameter per field (in
/// declaration order), each assigned straight into `self.<field>` in the
/// body (the exact shape `docs/book.md`'s own `Circle`-with-defaulted-param
/// example uses), so this reuses `emit_init`'s already-correct body-init
/// codegen and `emit_expr.rs`'s `try_emit_labeled_init_call` call-site
/// defaulting verbatim — no new transpiler machinery.
fn synthesize_init(s: &StructDecl, reg: &Registry) -> Result<InitDecl, ParseError> {
    let mut params = Vec::with_capacity(s.fields.len());
    let mut body = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let is_inject = f.attrs.iter().any(|a| a.name == "inject");
        let default = if is_inject {
            let Some(base) = base_type_name(&f.ty) else {
                return Err(err(f.line, f.col, format!(
                    "`@inject` field `{}` has no recognizable base type to resolve a provider \
                     against", f.name,
                )));
            };
            let Some(provider) = reg.get(base) else {
                return Err(err(f.line, f.col, format!(
                    "no `@provide` found for type `{}` (needed by `@inject` field `{}`) — declare \
                     a `pub @provide` function returning `{}` somewhere in this project \
                     (docs/design-notes/boring-di-draft.md §3)",
                    base, f.name, base,
                )));
            };
            check_field_qualifier_accepted(f, provider)?;
            check_singleton_qualifier_match(f, provider)?;
            Some(Expr {
                kind: ExprKind::Call(
                    Box::new(Expr { kind: ExprKind::Var(provider.fn_name.clone()), line: f.line, col: f.col, len: 0 }),
                    Vec::new(),
                ),
                line: f.line,
                col: f.col,
                len: 0,
            })
        } else {
            f.default.clone()
        };
        params.push(InitParam {
            is_pub: false,
            mutable: false,
            name: f.name.clone(),
            ty: Some(f.ty.clone()),
            default,
            line: f.line,
            col: f.col,
        });
        body.push(Stmt::Expr(Expr {
            kind: ExprKind::Assign(
                Box::new(Expr {
                    kind: ExprKind::Field(
                        Box::new(Expr { kind: ExprKind::Var("self".to_string()), line: f.line, col: f.col, len: 0 }),
                        f.name.clone(),
                    ),
                    line: f.line, col: f.col, len: 0,
                }),
                Box::new(Expr { kind: ExprKind::Var(f.name.clone()), line: f.line, col: f.col, len: 0 }),
            ),
            line: f.line, col: f.col, len: 0,
        }));
    }
    Ok(InitDecl { params, body, line: s.line, col: s.col })
}

fn desugar_struct(mut s: StructDecl, reg: &Registry) -> Result<StructDecl, ParseError> {
    let has_inject = s.fields.iter().any(|f| f.attrs.iter().any(|a| a.name == "inject"));
    if !has_inject {
        return Ok(s);
    }
    if !s.inits.is_empty() {
        return Err(err(s.line, s.col, format!(
            "`@inject` is not yet supported on `{}`, which already declares its own `init` — \
             remove the custom `init` (this struct's fields don't need one, `@inject` synthesizes \
             the constructor) or drop `@inject` for now",
            s.name,
        )));
    }
    let init = synthesize_init(&s, reg)?;
    s.inits.push(init);
    Ok(s)
}

fn desugar_items(items: Vec<Item>, reg: &Registry) -> Result<Vec<Item>, ParseError> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(match item {
            Item::Struct(s) => Item::Struct(desugar_struct(s, reg)?),
            Item::Mod(mut m) => { m.items = desugar_items(m.items, reg)?; Item::Mod(m) }
            other => other,
        });
    }
    Ok(out)
}

pub fn desugar_inject(mut program: Program) -> Result<Program, ParseError> {
    let mut reg = Registry::new();
    collect_providers(&program.items, &mut reg)?;
    program.items = desugar_items(program.items, &reg)?;
    Ok(program)
}
