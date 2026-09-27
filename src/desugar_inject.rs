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
//! "resolution" step (building the `(base type, id) -> provider` registry
//! this design calls for, and matching each `@inject` field against it) and
//! the "desugar" step (rewriting the match away into an ordinary, `init`-based
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
//! ## Scope of this implementation (see the design doc's Open Questions/
//! "Before implementation begins" for what's deliberately still deferred)
//!
//! - **Same-project (in practice: same parsed `Program`) only** — no `[deps]`
//!   cross-project resolution yet. A provider is visible here exactly when
//!   it's a top-level (or one-level-nested-in-`mod`) `@provide` function in
//!   this same `Program`.
//! - **`id`/`env` (§5-§6) are implemented** — see `attr_kv`/`resolve_provider`
//!   below. `env` is read once, up front, from a `--env <value>` CLI flag
//!   (`main.rs`'s `current_env_flag`) — a self-contained Boring-CLI concern,
//!   no dependency on Cargo/Rust build profiles.
//! - **Bare `@inject` fields are supported, but only against a `@singleton`
//!   provider** (§2's "single most likely real case" — `synthesize_init`
//!   copies the provider's return type onto the field verbatim, skipping
//!   chapter 30 inference entirely; sound here because `collect_providers`
//!   always finishes building the whole registry before any struct is
//!   processed, so the whole-program-collection-pass-before-layout-is-fixed
//!   ordering concern §2 originally raised is satisfied by construction). A
//!   bare field against a *transient* provider still requires an explicit
//!   qualifier — not a design limitation, a real `boring build`-specific gap:
//!   chapter 30 inference runs later, per function, at transpile time, but
//!   the default expression substituted at an omitting call site is rendered
//!   once, up front, when the struct's `init` is registered — before
//!   inference has decided the field's actual representation, so it never
//!   gets the wrap (`Box::new(...)`/`Arc::new(...)`) inference later
//!   requires. Confirmed via a real `cargo build` failure; `boring run` has
//!   no such ordering problem (the interpreter evaluates the default
//!   expression fresh, with no static wrapping step to get out of sync), but
//!   this is rejected either way to keep both backends behaving identically.
//! - **A struct with an `@inject` field can't also declare its own `init`**
//!   yet — keeps this first cut to the common case (no hand-written
//!   constructor at all) rather than also merging synthesized and
//!   user-written parameter lists.
//! - **`'static` is not yet in the accepted set** — blocked on the
//!   `docs/book.md` §21 amendment the design doc's §2 calls for (a fourth
//!   legal `'static`-construction site, inside a `@provide`-attributed
//!   function body); not attempted yet.
//! - **No `[deps]` cross-project resolution, no cycle detection.**
//!
//! None of this is `@inject`-site-vs-`@provide`-site cross-project machinery
//! the checker itself would need to know about later — widening any of the
//! above only ever changes *this* pass's resolution logic, not what a
//! resolved `@inject` field desugars into.

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
    /// `@provide(env = "...")` (§6) — `None` for a plain, unconditional
    /// provider. Filtered against the current build's `--env` value at
    /// resolution time; an env-matching provider outranks a plain one for the
    /// same `(base type, id)` key.
    env: Option<String>,
    line: usize,
}

/// Keyed by `(base type, id)` — `id` is `None` for a bare, unnamed binding
/// (§5). Each bucket holds every `@provide` candidate for that key, at most
/// one per distinct `env` value (including at most one with `env: None`) —
/// `collect_providers` rejects two providers sharing both key *and* `env` as
/// unconditionally ambiguous; `resolve_provider` picks among the rest by the
/// current build's `env`.
type Registry = HashMap<(String, Option<String>), Vec<Provider>>;

fn err(line: usize, col: usize, msg: String) -> ParseError {
    ParseError::Generic { line, col, len: 1, msg }
}

/// Parses one named argument's value out of an attribute's raw arg list
/// (`Attr::args`, e.g. `@provide(id = "primary", env = "test")` parses to
/// `["id=\"primary\"", "env=\"test\""]` — one already-comma-split string per
/// argument, `key=value` with no surrounding whitespace, confirmed against
/// the actual parser output rather than assumed). Strips the value's
/// surrounding string-literal quotes, if any — `id`/`env` are always written
/// as string literals (§5's "guardrail, non-negotiable: must be a
/// compile-time string literal on both sides" — trivially true here, since
/// `Attr::args` only ever holds literal source text in the first place, never
/// a re-evaluatable expression).
fn attr_kv<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    for arg in args {
        if let Some((k, v)) = arg.split_once('=') {
            if k == key {
                return Some(v.trim().trim_matches('"'));
            }
        }
    }
    None
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
/// `'shared`/`'actor`/`'guard`/`'observed`(-suffixed) always legal; `'owned`
/// only when the matched provider isn't `@singleton`; anything else
/// (bare/scalar, `'inline`, `'weak`-suffixed) is a hard error. `'static` is
/// not yet accepted — see this file's module doc.
fn check_field_qualifier_accepted(
    field: &FieldDecl,
    provider: &Provider,
) -> Result<(), ParseError> {
    let ty = field.ty.without_mut();
    let Some(q) = outer_qualifier(ty) else {
        return Err(err(field.line, field.col, format!(
            "`@inject` field `{}` has no qualifier — `@inject` needs one of `'shared`/`'actor`/\
             `'guard`/`'observed`, or `'owned` for a non-`@singleton` provider \
             (docs/design-notes/boring-di-draft.md §2); a bare scalar or plain struct type has \
             nothing for dependency injection to abstract",
            field.name,
        )));
    };
    match q {
        OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Observed => Ok(()),
        OwnerQual::Static => Err(err(field.line, field.col, format!(
            "`@inject` field `{}` cannot be `'static` yet — this needs a `docs/book.md` §21 \
             amendment (a fourth legal `'static`-construction site, inside a `@provide`-attributed \
             function body) not implemented yet (docs/design-notes/boring-di-draft.md §2)",
            field.name,
        ))),
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
             boring-di-draft.md §2) — use `'shared`/`'actor`/`'guard`/`'observed`, or `'owned` \
             for a non-`@singleton` provider",
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
                let provide_attr = f.attrs.iter().find(|a| a.name == "provide")
                    .expect("guarded by this match arm's own guard");
                let is_singleton = f.attrs.iter().any(|a| a.name == "singleton");
                let id = attr_kv(&provide_attr.args, "id").map(str::to_string);
                let env = attr_kv(&provide_attr.args, "env").map(str::to_string);
                let entry = Provider {
                    fn_name: f.name.clone(),
                    is_singleton,
                    return_ty: ret_ty.without_mut().clone(),
                    env: env.clone(),
                    line: f.line,
                };
                let key = (base.to_string(), id.clone());
                let bucket = reg.entry(key).or_default();
                if let Some(existing) = bucket.iter().find(|p| p.env == env) {
                    let id_desc = id.as_deref().map(|i| format!(" id = \"{}\",", i)).unwrap_or_default();
                    let env_desc = env.as_deref().map(|e| format!(" env = \"{}\"", e)).unwrap_or_else(|| "no env".to_string());
                    return Err(err(f.line, f.col, format!(
                        "ambiguous provider for `{}` ({}{}) — `@provide` functions `{}` (line {}) \
                         and `{}` (line {}) both provide this exact binding, with no further `id`/\
                         `env` to disambiguate (docs/design-notes/boring-di-draft.md, \"Ambiguity UX\")",
                        base, id_desc, env_desc, existing.fn_name, existing.line, f.name, f.line,
                    )));
                }
                bucket.push(entry);
            }
            Item::Mod(m) => collect_providers(&m.items, reg)?,
            _ => {}
        }
    }
    Ok(())
}

/// Resolves `(base, id)` against the registry for the current build's `env`
/// (§6): an env-matching candidate outranks a plain (`env: None`) one for the
/// same key; a plain candidate is the fallback when nothing matches the
/// current `env` (or no `--env` was given at all). Two same-rank candidates
/// can never both survive to this point — `collect_providers` already
/// rejects two providers sharing both key and `env` value as unconditionally
/// ambiguous — so this only ever picks among *different* `env` values.
fn resolve_provider<'a>(
    reg: &'a Registry,
    base: &str,
    id: Option<&str>,
    current_env: Option<&str>,
    field: &FieldDecl,
) -> Result<&'a Provider, ParseError> {
    let key = (base.to_string(), id.map(str::to_string));
    let candidates = reg.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
    if let Some(env) = current_env {
        if let Some(p) = candidates.iter().find(|p| p.env.as_deref() == Some(env)) {
            return Ok(p);
        }
    }
    if let Some(p) = candidates.iter().find(|p| p.env.is_none()) {
        return Ok(p);
    }
    let id_desc = id.map(|i| format!(" (id = \"{}\")", i)).unwrap_or_default();
    let env_desc = current_env.map(|e| format!(" for env \"{}\"", e)).unwrap_or_default();
    Err(err(field.line, field.col, format!(
        "no `@provide` found for type `{}`{}{} (needed by `@inject` field `{}`) — declare a \
         `pub @provide` function returning `{}` somewhere in this project \
         (docs/design-notes/boring-di-draft.md §3)",
        base, id_desc, env_desc, field.name, base,
    )))
}

/// Builds the `init` this struct needs so its `@inject` field(s) become
/// omittable, defaulted constructor arguments — one parameter per field (in
/// declaration order), each assigned straight into `self.<field>` in the
/// body (the exact shape `docs/book.md`'s own `Circle`-with-defaulted-param
/// example uses), so this reuses `emit_init`'s already-correct body-init
/// codegen and `emit_expr.rs`'s `try_emit_labeled_init_call` call-site
/// defaulting verbatim — no new transpiler machinery.
///
/// Takes `s` mutably: a **bare** `@inject` field (no qualifier written)
/// matched against a `@singleton` provider has its type rewritten in place to
/// the provider's own return type, copied verbatim (§2 — "the field simply
/// takes the provider's qualifier verbatim, since that's the one and only
/// representation the shared value actually has"). This is exactly the
/// whole-program-registry-before-any-struct-layout ordering the design doc's
/// §2 worried a bare field would need — already satisfied for free here,
/// since `collect_providers` always finishes before this function is ever
/// called (`desugar_inject`'s two-pass structure). A bare field matched
/// against a *transient* provider is left untouched — no fixed
/// representation to copy, so it runs chapter 30's ordinary usage-based
/// inference exactly like any other unqualified struct field, same as if
/// `@inject` had never been written at all.
fn synthesize_init(s: &mut StructDecl, reg: &Registry, current_env: Option<&str>) -> Result<InitDecl, ParseError> {
    let mut params = Vec::with_capacity(s.fields.len());
    let mut body = Vec::with_capacity(s.fields.len());
    for f in &mut s.fields {
        let inject_attr = f.attrs.iter().find(|a| a.name == "inject");
        let default = if let Some(inject_attr) = inject_attr {
            let is_bare = outer_qualifier(f.ty.without_mut()).is_none();
            let id = attr_kv(&inject_attr.args, "id").map(str::to_string);
            let base = {
                let Some(base) = base_type_name(&f.ty) else {
                    return Err(err(f.line, f.col, format!(
                        "`@inject` field `{}` has no recognizable base type to resolve a provider \
                         against", f.name,
                    )));
                };
                base.to_string()
            };
            let provider = resolve_provider(reg, &base, id.as_deref(), current_env, f)?;
            if is_bare && provider.is_singleton {
                // §2: bare field, `@singleton` provider — copy verbatim, skip chapter 30
                // inference entirely (not just narrow it). Runs the same acceptance/
                // match checks afterward, against the now-copied type — this is also
                // what correctly rejects a bare field copying a provider whose own
                // return type has nothing for DI to abstract (e.g. a `@singleton`
                // function that just happens to return a bare scalar, legal on its own
                // — §4 — but meaningless as something to `@inject`).
                f.ty = provider.return_ty.clone();
                check_field_qualifier_accepted(f, provider)?;
                check_singleton_qualifier_match(f, provider)?;
            } else if !is_bare {
                check_field_qualifier_accepted(f, provider)?;
                check_singleton_qualifier_match(f, provider)?;
            } else {
                // Bare field, transient provider — §2 says this should run chapter 30's
                // ordinary usage-based inference chain unmodified. In principle, yes; in
                // this implementation, no, not yet: chapter 30 inference runs later, at
                // transpile time, per function — but the default expression substituted at
                // an omitting call site is rendered once, up front, when this struct's
                // `init` is first registered (`struct_init_defaults`,
                // `src/transpiler/mod.rs`), before inference has decided anything. By the
                // time inference picks (say) `'owned` for this bare field, the plain
                // `providerFn()` default text is already fixed with no `Box::new(...)`
                // wrap — confirmed via a real `cargo build` failure (E0308: expected
                // `Box<dyn Trait>`, found the provider's own bare return type). `boring run`
                // has no such ordering problem at all (the interpreter evaluates the
                // default expression fresh, at construction time, with no static wrapping
                // step to get out of sync) — so this is a `boring build`-only gap, and
                // rejecting it here keeps both backends honest rather than shipping a
                // combination that silently works on one and not the other. Write an
                // explicit qualifier for now; only the `@singleton` case above is affected
                // by the ordering concern the design doc's §2 originally worried about.
                return Err(err(f.line, f.col, format!(
                    "`@inject` field `{}` needs an explicit qualifier for now when its provider \
                     (`{}`) isn't `@singleton` — bare-field inference against a transient \
                     provider isn't supported yet (a `boring build`-specific default-expression-\
                     wrapping gap, not part of the design itself); write e.g. `{}'shared` instead",
                    f.name, provider.fn_name, base,
                )));
            }
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

fn desugar_struct(mut s: StructDecl, reg: &Registry, current_env: Option<&str>) -> Result<StructDecl, ParseError> {
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
    let init = synthesize_init(&mut s, reg, current_env)?;
    s.inits.push(init);
    Ok(s)
}

fn desugar_items(items: Vec<Item>, reg: &Registry, current_env: Option<&str>) -> Result<Vec<Item>, ParseError> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(match item {
            Item::Struct(s) => Item::Struct(desugar_struct(s, reg, current_env)?),
            Item::Mod(mut m) => { m.items = desugar_items(m.items, reg, current_env)?; Item::Mod(m) }
            other => other,
        });
    }
    Ok(out)
}

pub fn desugar_inject(mut program: Program, current_env: Option<&str>) -> Result<Program, ParseError> {
    let mut reg = Registry::new();
    collect_providers(&program.items, &mut reg)?;
    program.items = desugar_items(program.items, &reg, current_env)?;
    Ok(program)
}
