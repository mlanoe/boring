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
//! - **Same-project only** — no `[deps]` cross-project resolution yet. A
//!   provider is visible here when it's a top-level (or
//!   one-level-nested-in-`mod`) `@provide` function in the entry `Program`
//!   itself, *or* in any same-project sibling file transitively reachable
//!   from it via a bare `use <name>` (`walk_same_project_uses`) — this only
//!   widens the *provider registry* (and trait-name set, for bare-field
//!   detection), read-only: the sibling file's own content is parsed purely to
//!   look for `@provide`/`trait` declarations, never rewritten or merged back.
//!   **A struct declared only in a sibling file, with its own `@inject`
//!   field, is not covered** — that field never gets desugared at all,
//!   because neither backend's own `use`-loading (`inline_boring_use` in the
//!   transpiler, `exec_use` in the interpreter) invokes this pass on a file it
//!   loads; only the file actually handed to `desugar_inject` up front (the
//!   entry file) ever has its own structs desugared. Closing that gap means
//!   hooking this pass into both of those call sites directly — a
//!   substantially bigger architectural change than the read-only registry
//!   widening implemented here, deliberately not attempted in this slice.
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
//! - **`'static` is accepted** (§2) — an explicit `@inject` field only; never
//!   copied verbatim onto a bare one, even from a `@singleton` provider (§2's
//!   own carve-out — see the dedicated check in `synthesize_init`). The
//!   `docs/book.md` §21 "fourth legal construction site" amendment §2
//!   originally called for turned out to be unnecessary in practice: no
//!   existing checker rule actually gates a function's own return-type
//!   provenance or a defaulted-parameter's default-value provenance today
//!   (`check_static_provenance`/`check_static_arg_provenance` only cover a
//!   `let` statement's initializer and a call argument, respectively) — so
//!   there was no site-authorization list to extend at all for either a
//!   `@provide` function's tail expression or `desugar_inject`'s own
//!   synthesized default. Also fixed a real, unrelated parser bug found while
//!   testing this: the bare (no `def`/`req`) return-type-first function
//!   shorthand (`Config'static loadConfig(): ...`) failed to parse at all,
//!   for any of `'static`/`'guard`/`'task`/`'req` specifically — `is_fn_decl_shorthand`'s
//!   scanner only knew how to skip a qualifier name written as a generic
//!   `Ident`, not one of these four reserved-keyword-tokenized qualifiers
//!   (`parser/mod.rs`).
//! - **Cycle detection is implemented** (§7) — `detect_cycles`, best-effort (see
//!   `Provider::target_struct`'s doc for the one accepted limitation).
//! - **No `[deps]` cross-project resolution.**
//!
//! None of this is `@inject`-site-vs-`@provide`-site cross-project machinery
//! the checker itself would need to know about later — widening any of the
//! above only ever changes *this* pass's resolution logic, not what a
//! resolved `@inject` field desugars into.

use crate::ast::*;
use crate::parser::ParseError;
use std::collections::{HashMap, HashSet};

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
    /// The concrete struct this provider's body directly constructs, when its
    /// body is the simple, common shape every worked example in the design
    /// doc actually uses — a bare tail-expression or explicit `return` that's
    /// itself a constructor call (`RealNetworkClient()`, `PostgresDatabase(host
    /// = ...)`) — used only for cycle detection (§7): if that struct itself has
    /// `@inject` fields, their own resolved providers become this provider's
    /// graph edges. `None` for anything else (an arbitrary expression, a
    /// multi-statement body ending some other way) — cycle detection simply
    /// can't see through the edge in that case and treats it as a dead end,
    /// same as a provider that constructs nothing `@inject`-relevant at all.
    /// This is a deliberate, accepted limitation (see this file's module doc)
    /// — a real static analysis that's sound-by-omission (never a false
    /// "cycle detected"), not a sound-by-construction whole-program dataflow
    /// pass.
    target_struct: Option<String>,
}

/// Best-effort: does this function's body end in a bare constructor call
/// (`Type(args...)` or `Type(args...) as`-free equivalent), either as the tail
/// expression or an explicit `return`? See `Provider::target_struct`'s doc for
/// why this doesn't need to be exhaustive.
fn provider_target_struct(f: &FnDecl) -> Option<String> {
    let tail = f.body.last()?;
    let expr = match tail {
        Stmt::Expr(e) => Some(e),
        Stmt::Return(r) => r.value.as_ref(),
        _ => None,
    }?;
    match &expr.kind {
        ExprKind::Call(callee, _) => match &callee.kind {
            ExprKind::Var(name) => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
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
/// `'shared`/`'actor`/`'guard`/`'observed`(-suffixed)/`'static` always legal;
/// `'owned` only when the matched provider isn't `@singleton`; anything else
/// (bare/scalar, `'inline`, `'weak`-suffixed) is a hard error. `'static`'s
/// extra "must be written explicitly, never bare-copied" rule is checked
/// separately, by `synthesize_init`, before this function ever runs on an
/// (already-explicit-by-then) field.
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
        OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Observed
        | OwnerQual::Static => Ok(()),
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
                    target_struct: provider_target_struct(f),
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

/// Every trait name declared in this `Program` (recursing into `mod`, same
/// scope as `collect_providers` above) — lets `synthesize_init` tell a bare
/// `@inject` field's base type apart from an ordinary struct/generic type
/// without needing any transpiler state (this pass runs well before the
/// transpiler ever sees the program). See `synthesize_init`'s "Bare field,
/// transient provider" arm for why the distinction matters: a trait is the
/// one case where a bare field's Rust representation is a fixed, unconditional
/// rule (`Box<dyn Trait>` — a trait object is unsized, so there's no other
/// option) rather than something chapter 30's per-function usage-based
/// inference decides later.
fn collect_trait_names(items: &[Item], names: &mut HashSet<String>) {
    for item in items {
        match item {
            Item::Trait(t) => { names.insert(t.name.clone()); }
            Item::Mod(m) => collect_trait_names(&m.items, names),
            _ => {}
        }
    }
}

/// Widens `collect_providers`/`collect_trait_names` to also see a **same-project
/// sibling file**, reached the same way a bare `use <name>` already does at
/// every other point in the pipeline (`source_dir`-relative — mirrors
/// `emit_top.rs`'s `emit_use`, minus the `[deps]`/`boring.<module>` special
/// cases, which this simply never matches: a `[deps]` name or a stdlib module
/// resolves to a path that doesn't exist under `source_dir`, so the `exists()`-
/// equivalent check below naturally — not by any explicit exclusion — limits
/// this to same-project files only, exactly this pass's intended scope for now
/// (see this file's module doc: no `[deps]` cross-project resolution yet).
///
/// Read-only and best-effort: a sibling file that fails to read/lex/parse is
/// silently skipped here rather than reported — any *real* syntax error in it
/// still surfaces moments later, with a proper diagnostic, when the checker/
/// transpiler/interpreter independently (and always) re-parses it themselves.
/// This pass only ever *adds* candidates a bare `use` already makes reachable;
/// it never changes what's a compile error elsewhere.
///
/// **What this does not do** (a real, accepted limitation — see this file's
/// module doc): a *struct* declared only in a sibling file, with its own
/// `@inject` field, never gets that field desugared at all by visiting it this
/// way — this function only ever reads a sibling file to grow the provider/
/// trait-name registry, it never rewrites or returns the sibling `Program`
/// itself. Making that direction work would mean hooking this whole pass into
/// `inline_boring_use` (`transpiler/emit_top.rs`) and `exec_use`
/// (`interpreter/mod.rs`) — the actual places each backend independently loads
/// a `use`d file from disk — a materially bigger change than widening a
/// read-only registry scan, deliberately not attempted in this first slice.
fn walk_same_project_uses(
    items: &[Item],
    source_dir: &std::path::Path,
    reg: &mut Registry,
    trait_names: &mut HashSet<String>,
    visited: &mut HashSet<std::path::PathBuf>,
) -> Result<(), ParseError> {
    for item in items {
        match item {
            Item::Use(u) if !u.path.is_empty() => {
                let rel: std::path::PathBuf = u.path.iter().collect();
                let candidate = source_dir.join(rel).with_extension("br");
                let Ok(candidate) = candidate.canonicalize() else { continue };
                if !visited.insert(candidate.clone()) { continue; }
                let Ok(source) = std::fs::read_to_string(&candidate) else { continue };
                let Ok(tokens) = crate::lexer::lex_all(&source) else { continue };
                let Ok(sibling) = crate::parser::parse(tokens) else { continue };
                // Errors from a sibling file (ambiguous providers within *it*) are real
                // and worth surfacing — unlike read/lex/parse failure, an ambiguity here
                // is this pass's own, well-defined diagnostic, not something the
                // checker/transpiler would otherwise catch on their own independent
                // re-parse (they don't run `desugar_inject` on a `use`d file at all, per
                // this function's own doc above).
                collect_providers(&sibling.items, reg)?;
                collect_trait_names(&sibling.items, trait_names);
                walk_same_project_uses(&sibling.items, source_dir, reg, trait_names, visited)?;
            }
            Item::Mod(m) => walk_same_project_uses(&m.items, source_dir, reg, trait_names, visited)?,
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
/// against a *transient* provider is, in the general case, left untouched —
/// no fixed representation to copy, so it runs chapter 30's ordinary
/// usage-based inference exactly like any other unqualified struct field,
/// same as if `@inject` had never been written at all. That general case is
/// still rejected outright (see the "Bare field, transient provider" arm
/// below) — a real `boring build` gap, not a design limitation. A bare field
/// whose base type is a *trait* (`trait_names`) is the one exception: a trait
/// object is unsized, so its bare representation is always `Box<dyn Trait>`
/// regardless of usage, a fixed rule rather than something chapter 30 decides
/// — so that combination is accepted and left bare, same as the general case
/// says it should be, just without needing to wait on inference at all.
fn synthesize_init(
    s: &mut StructDecl,
    reg: &Registry,
    current_env: Option<&str>,
    trait_names: &HashSet<String>,
) -> Result<InitDecl, ParseError> {
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
            if is_bare && outer_qualifier(&provider.return_ty) == Some(&OwnerQual::Static) {
                // §2: `'static` is the one deliberate exception to "just copy the
                // provider" — not because of the general "'static is never inferred"
                // rule (nothing is *constructed* at an `@inject` site, ever, so that
                // risk doesn't apply here), but because `'static` uniquely carries two
                // sharp, easy-to-miss consequences (the §6 test-override escape hatch
                // breaks; a fourth legal construction site is needed) that deserve to be
                // visible at the `@inject` site itself, especially given the provider
                // can live in a different project entirely. Bare `NetworkClient client`
                // silently inheriting `'static`-ness from a distant provider would hide
                // that cost from a reader standing at the field declaration.
                return Err(err(f.line, f.col, format!(
                    "`@inject` field `{}` must write `'static` explicitly — the only visible \
                     provider (`{}`) returns `{}'static`, and unlike every other accepted \
                     qualifier, `'static` is never copied onto a bare field silently \
                     (docs/design-notes/boring-di-draft.md §2)",
                    f.name, provider.fn_name, base,
                )));
            }
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
            } else if trait_names.contains(&base) && outer_qualifier(&provider.return_ty).is_none() {
                // Bare field, transient provider, TRAIT base type, and — critically —
                // the provider itself returns that trait BARE too (no `'shared`/`'owned`/
                // etc. of its own). This is the one case where the general ordering gap
                // below doesn't apply. A trait object is unsized, so a bare trait-typed
                // field's Rust representation is always `Box<dyn Trait>`, unconditionally
                // (`emit_field_type`'s "Priority 4 (dyn Trait) still applies" arm) — not
                // something chapter 30's per-function usage-based inference decides, so
                // there's nothing for the up-front `struct_init_defaults` rendering
                // (`src/transpiler/mod.rs`) to race against; it now recognizes a bare
                // trait-typed param as needing the qualifier-aware `emit_let_value`
                // (which already `Box::new(...)`-wraps a bare trait-typed default) the
                // same way an explicit `'owned`/`'new` param always did — confirmed via a
                // real `cargo build` + `cargo run` round trip.
                //
                // The `provider.return_ty` guard matters independently of that ordering
                // fix: a bare field is *always* `Box<dyn Trait>`, so the provider call
                // must produce something a single `Box::new(...)` can turn into that —
                // true only when the provider itself returns the trait bare (a concrete,
                // by-value/static-dispatch return). A `'shared`/`'actor`/`'guard`
                // provider already returns `Arc<dyn Trait>`/`Arc<Mutex<dyn Trait>>`/etc.,
                // which `Box::new(...)` can't turn into `Box<dyn Trait>` (E0308, not an
                // ordering bug) — and an `'owned`/`'new` provider already returns
                // `Box<dyn Trait>` itself, which `box_if_trait_typed` doesn't recognize
                // from a bare `Call(fn_name, ..)` the way it does a variable of the exact
                // trait type, so it would double-box. Both stay on the rejection path
                // below, same as before this fix — write an explicit qualifier there.
            } else {
                // Bare field, transient provider, non-trait (plain struct/generic) base
                // type — §2 says this should run chapter 30's ordinary usage-based
                // inference chain unmodified. In principle, yes; in this implementation,
                // no, not yet: chapter 30 inference runs later, at transpile time, per
                // function — but the default expression substituted at an omitting call
                // site is rendered once, up front, when this struct's `init` is first
                // registered (`struct_init_defaults`, `src/transpiler/mod.rs`), before
                // inference has decided anything. By the time inference picks (say)
                // `'owned` for this bare field, the plain `providerFn()` default text is
                // already fixed with no `Box::new(...)` wrap — confirmed via a real
                // `cargo build` failure (E0308: expected `Box<dyn Trait>`, found the
                // provider's own bare return type). `boring run` has no such ordering
                // problem at all (the interpreter evaluates the default expression
                // fresh, at construction time, with no static wrapping step to get out
                // of sync) — so this is a `boring build`-only gap, and rejecting it here
                // keeps both backends honest rather than shipping a combination that
                // silently works on one and not the other. Unlike the trait case above,
                // there's no fixed rule to fall back on here — the field's eventual
                // representation genuinely isn't knowable this early. Write an explicit
                // qualifier for now; only the `@singleton` case above is affected by the
                // ordering concern the design doc's §2 originally worried about.
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

fn desugar_struct(
    mut s: StructDecl,
    reg: &Registry,
    current_env: Option<&str>,
    trait_names: &HashSet<String>,
) -> Result<StructDecl, ParseError> {
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
    let init = synthesize_init(&mut s, reg, current_env, trait_names)?;
    s.inits.push(init);
    Ok(s)
}

fn desugar_items(
    items: Vec<Item>,
    reg: &Registry,
    current_env: Option<&str>,
    trait_names: &HashSet<String>,
) -> Result<Vec<Item>, ParseError> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(match item {
            Item::Struct(s) => Item::Struct(desugar_struct(s, reg, current_env, trait_names)?),
            Item::Mod(mut m) => { m.items = desugar_items(m.items, reg, current_env, trait_names)?; Item::Mod(m) }
            other => other,
        });
    }
    Ok(out)
}

/// One struct's own `@inject` dependencies — `(base type, id, field line/col)`
/// per field, recursed into `mod` the same as `collect_providers`/
/// `collect_trait_names`. Bare fields are included too (keyed on their own
/// base type, `id`) — cycle detection doesn't care whether a field ends up
/// bare or explicit, only what type it ultimately needs a provider for.
fn collect_struct_inject_deps(
    items: &[Item],
    out: &mut HashMap<String, Vec<(String, Option<String>, usize, usize)>>,
) {
    for item in items {
        match item {
            Item::Struct(s) => {
                let deps: Vec<_> = s.fields.iter().filter_map(|f| {
                    let inject_attr = f.attrs.iter().find(|a| a.name == "inject")?;
                    let base = base_type_name(&f.ty)?.to_string();
                    let id = attr_kv(&inject_attr.args, "id").map(str::to_string);
                    Some((base, id, f.line, f.col))
                }).collect();
                if !deps.is_empty() {
                    out.insert(s.name.clone(), deps);
                }
            }
            Item::Mod(m) => collect_struct_inject_deps(&m.items, out),
            _ => {}
        }
    }
}

/// §7: "A cycle... is a compile error, reported as the full chain, not just
/// 'cycle detected'." Builds a directed graph over provider *functions*
/// (nodes = `fn_name`) — an edge `P -> Q` means "resolving `P` (i.e. calling
/// the struct it constructs) transitively needs `Q` too" — and runs a
/// straightforward DFS with an explicit path stack, reporting the first back-
/// edge found as the full chain from where it re-enters itself. See
/// `Provider::target_struct`'s doc for the real, accepted limitation this
/// inherits: an edge only exists where a provider's body is simple enough
/// (§ above) for this pass to see through it at all — a genuine cycle hidden
/// behind a provider whose body does anything more elaborate than a bare
/// constructor call silently isn't caught. Never a false positive, only a
/// possible false negative — same "sound-by-omission" posture as the rest of
/// this file's best-effort static checks.
fn detect_cycles(
    reg: &Registry,
    struct_deps: &HashMap<String, Vec<(String, Option<String>, usize, usize)>>,
) -> Result<(), ParseError> {
    // Flatten the registry into one lookup by provider name (fn_name is
    // globally unique — `collect_providers` already rejects two providers
    // sharing a full key, and distinct keys always have distinct fn_names in
    // any program that doesn't itself have a duplicate top-level fn name,
    // already a separate compile error elsewhere).
    let by_name: HashMap<&str, &Provider> = reg.values()
        .flat_map(|bucket| bucket.iter())
        .map(|p| (p.fn_name.as_str(), p))
        .collect();

    // Edges: provider name -> every provider it transitively needs. Resolved
    // ignoring `env`/`current_env` entirely and fanning out over *every*
    // candidate for a given `(base, id)` key — deliberately conservative, the
    // same "same-project exhaustive" spirit `collect_providers`'s own
    // ambiguity check already uses: a cycle that only manifests for one
    // particular `--env` value is still a real cycle worth catching, not
    // something to silently defer to whichever build happens to hit it.
    let edges = |p: &Provider| -> Vec<String> {
        let Some(target) = &p.target_struct else { return Vec::new() };
        let Some(deps) = struct_deps.get(target) else { return Vec::new() };
        deps.iter()
            .filter_map(|(base, id, _, _)| reg.get(&(base.clone(), id.clone())))
            .flat_map(|bucket| bucket.iter().map(|q| q.fn_name.clone()))
            .collect()
    };

    let mut state: HashMap<&str, u8> = HashMap::new(); // 0=unvisited, 1=on stack, 2=done
    let mut path: Vec<&str> = Vec::new();

    fn visit<'a>(
        name: &'a str,
        by_name: &HashMap<&'a str, &'a Provider>,
        edges: &dyn Fn(&Provider) -> Vec<String>,
        state: &mut HashMap<&'a str, u8>,
        path: &mut Vec<&'a str>,
    ) -> Result<(), ParseError> {
        match state.get(name).copied().unwrap_or(0) {
            2 => return Ok(()),
            1 => {
                // Back-edge found — `name` is already on the current path. Report
                // the full chain from its first occurrence back to itself.
                let start = path.iter().position(|n| *n == name).unwrap_or(0);
                let chain: Vec<&str> = path[start..].iter().copied().chain(std::iter::once(name)).collect();
                let provider = by_name.get(name).expect("cycle node must be a known provider");
                return Err(err(provider.line, 0, format!(
                    "cycle detected among `@provide` providers: {} — each one transitively needs \
                     the next, with no way to ever finish constructing any of them \
                     (docs/design-notes/boring-di-draft.md §7)",
                    chain.join(" -> "),
                )));
            }
            _ => {}
        }
        let Some(provider) = by_name.get(name) else { return Ok(()) };
        state.insert(name, 1);
        path.push(name);
        for next in edges(provider) {
            // Leak the owned String into the same lifetime as everything else in
            // `by_name`'s keys would require unsafe or an arena; simplest correct
            // fix is to look the callee up by value each time instead of trying
            // to thread a borrowed `&str` through — re-borrow via `by_name`.
            if let Some(next_key) = by_name.keys().find(|k| **k == next.as_str()) {
                visit(next_key, by_name, edges, state, path)?;
            }
        }
        path.pop();
        state.insert(name, 2);
        Ok(())
    }

    for name in by_name.keys().copied().collect::<Vec<_>>() {
        visit(name, &by_name, &edges, &mut state, &mut path)?;
    }
    Ok(())
}

/// `source_dir`: the entry file's own directory (`main.rs`'s callers all have
/// this — the file they just parsed `program` from) — lets `collect_providers`/
/// `collect_trait_names` widen to same-project sibling files reached via a
/// bare `use <name>` (`walk_same_project_uses`, this file's own doc there for
/// the full scope/limitation). `None` (e.g. no real file on disk — a REPL-style
/// caller, if one ever exists) simply skips that widening; every other pass
/// still runs, scoped to `program` alone, exactly as before this parameter
/// existed.
pub fn desugar_inject(
    mut program: Program,
    current_env: Option<&str>,
    source_dir: Option<&std::path::Path>,
) -> Result<Program, ParseError> {
    let mut reg = Registry::new();
    collect_providers(&program.items, &mut reg)?;
    let mut trait_names = HashSet::new();
    collect_trait_names(&program.items, &mut trait_names);
    if let Some(source_dir) = source_dir {
        let mut visited = HashSet::new();
        walk_same_project_uses(&program.items, source_dir, &mut reg, &mut trait_names, &mut visited)?;
    }
    let mut struct_deps = HashMap::new();
    collect_struct_inject_deps(&program.items, &mut struct_deps);
    detect_cycles(&reg, &struct_deps)?;
    program.items = desugar_items(program.items, &reg, current_env, &trait_names)?;
    Ok(program)
}
