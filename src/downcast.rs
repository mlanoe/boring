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

//! Checked downcast of a trait object: `n as A` where `n`'s static type is a bare trait and `A`
//! is a struct/enum implementing it. The result is `A?` — the value when the dynamic type is
//! `A`, `nil` otherwise (docs/book.md, "Downcasting a trait object").
//!
//! This module holds what the checker and the transpiler share: which names are traits, which
//! types implement which trait (through `as` headers, `ext ... as`, and supertraits), how a
//! static type is recognised as a trait object, and the pre-scan that tells the transpiler
//! which traits need the `BoringAnyCast` supertrait.

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::ast_walk::Walker;

/// How a trait object is held, which decides how the downcast reaches the `dyn Trait` inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraitHandle {
    /// `Trait`, `Trait'owned`, `Dyn(Trait)`, borrows: `Box<dyn Trait>` / `&dyn Trait`.
    Plain,
    /// `Trait'shared`: `Arc<dyn Trait>` / `Rc<dyn Trait>`.
    Shared,
    /// `Trait'actor`: `Arc<Mutex<dyn Trait>>` / `Rc<RefCell<dyn Trait>>`.
    Actor,
    /// Any other qualifier (`'guard`, `'weak`, `'atomic`, ...): not supported, rejected clearly.
    Unsupported,
}

/// Declared traits and, per struct/enum, the traits it names in its `as` header.
#[derive(Debug, Default, Clone)]
pub struct TypeTraits {
    /// Trait name -> its direct supertraits.
    pub parents: HashMap<String, Vec<String>>,
    /// Struct/enum name -> traits it implements directly (header `as` list plus `ext T as ...`).
    pub protocols: HashMap<String, Vec<String>>,
}

impl TypeTraits {
    pub fn collect(items: &[Item]) -> Self {
        let mut tt = TypeTraits::default();
        tt.add_items(items);
        tt
    }

    pub fn add_items(&mut self, items: &[Item]) {
        for item in items {
            match item {
                Item::Trait(t) => { self.parents.insert(t.name.clone(), t.parents.clone()); }
                Item::Struct(s) => {
                    self.protocols.entry(s.name.clone()).or_default().extend(s.protocols.iter().cloned());
                }
                Item::Enum(e) => {
                    self.protocols.entry(e.name.clone()).or_default().extend(e.protocols.iter().cloned());
                }
                Item::Ext(e) => {
                    self.protocols.entry(e.type_name.clone()).or_default().extend(e.traits.iter().cloned());
                }
                Item::Mod(m) => self.add_items(&m.items),
                _ => {}
            }
        }
    }

    pub fn is_trait(&self, name: &str) -> bool {
        self.parents.contains_key(name)
    }

    /// A user struct/enum (anything that can appear as a header `as` implementor).
    pub fn is_concrete(&self, name: &str) -> bool {
        self.protocols.contains_key(name) && !self.parents.contains_key(name)
    }

    /// `ty` implements `tr` directly or through a supertrait of something it implements.
    pub fn implements(&self, ty: &str, tr: &str) -> bool {
        let Some(direct) = self.protocols.get(ty) else { return false };
        let mut seen = HashSet::new();
        let mut stack: Vec<&str> = direct.iter().map(String::as_str).collect();
        while let Some(t) = stack.pop() {
            if t == tr { return true; }
            if !seen.insert(t) { continue; }
            if let Some(ps) = self.parents.get(t) {
                stack.extend(ps.iter().map(String::as_str));
            }
        }
        false
    }

    /// Every trait `ty` implements, supertraits included.
    pub fn all_traits_of(&self, ty: &str) -> HashSet<String> {
        let mut out = HashSet::new();
        let Some(direct) = self.protocols.get(ty) else { return out };
        let mut stack: Vec<&str> = direct.iter().map(String::as_str).collect();
        while let Some(t) = stack.pop() {
            if !out.insert(t.to_string()) { continue; }
            if let Some(ps) = self.parents.get(t) {
                stack.extend(ps.iter().map(String::as_str));
            }
        }
        out
    }
}

/// If `ty` is a trait object — a bare declared trait name, possibly behind `mut`, a borrow,
/// `'owned`/`'inline`, `Dyn(..)`, `'shared` or `'actor` — returns the trait and how it is held.
pub fn trait_object_source(ty: &Type, is_trait: &dyn Fn(&str) -> bool) -> Option<(String, TraitHandle)> {
    fn go(ty: &Type, is_trait: &dyn Fn(&str) -> bool, handle: TraitHandle) -> Option<(String, TraitHandle)> {
        match ty {
            Type::Named(n) if is_trait(n) => Some((n.clone(), handle)),
            Type::Mut(inner) | Type::Dyn(inner) => go(inner, is_trait, handle),
            Type::Qualified(inner, q) => {
                let h = match q {
                    OwnerQual::Shared => TraitHandle::Shared,
                    OwnerQual::Actor => TraitHandle::Actor,
                    OwnerQual::Guard | OwnerQual::GuardTask | OwnerQual::ActorTask
                    | OwnerQual::Weak | OwnerQual::Atomic => TraitHandle::Unsupported,
                    _ => handle,
                };
                go(inner, is_trait, h)
            }
            _ => None,
        }
    }
    go(ty, is_trait, TraitHandle::Plain)
}

/// The struct/enum name a cast targets (`as A`, `as A?`), if it names one.
pub fn cast_target_name(ty: &Type) -> Option<&str> {
    match ty {
        // A single-letter struct name (`A`) parses as `TypeParam`.
        Type::Named(n) | Type::TypeParam(n) => Some(n),
        Type::Optional(inner) | Type::Mut(inner) | Type::Qualified(inner, _) => cast_target_name(inner),
        _ => None,
    }
}

/// Names that appear as the target of an `as` cast anywhere in `items` (fn/method/init/setter
/// bodies, trait defaults, nested modules, top-level statements).
fn cast_targets(items: &[Item]) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut items = items.to_vec();
    walk_items(&mut items, &mut out);
    out
}

fn walk_items(items: &mut [Item], out: &mut HashSet<String>) {
    let mut on_expr = |e: &mut Expr| {
        if let ExprKind::Cast(_, ty) = &e.kind {
            if let Some(n) = cast_target_name(ty) { out.insert(n.to_string()); }
        }
    };
    let mut on_stmt = |_: &mut Stmt| {};
    let mut w = Walker { expr: &mut on_expr, stmt: &mut on_stmt };
    fn fns(w: &mut Walker, fs: &mut [FnDecl]) {
        for f in fs { w.stmts(&mut f.body); }
    }
    let mut nested: Vec<&mut ModDecl> = Vec::new();
    for item in items.iter_mut() {
        match item {
            Item::Fn(f) => w.stmts(&mut f.body),
            Item::Struct(s) => {
                fns(&mut w, &mut s.methods);
                for i in &mut s.inits { w.stmts(&mut i.body); }
                for t in &mut s.type_methods { w.stmts(&mut t.body); }
                for t in &mut s.setters { w.stmts(&mut t.body); }
            }
            Item::Enum(e) => {
                fns(&mut w, &mut e.methods);
                for t in &mut e.type_methods { w.stmts(&mut t.body); }
                for t in &mut e.setters { w.stmts(&mut t.body); }
            }
            Item::Trait(t) => fns(&mut w, &mut t.defaults),
            Item::Ext(e) => fns(&mut w, &mut e.methods),
            Item::Let(l) => { if let Some(v) = &mut l.value { w.expr(v); } }
            Item::Stmt(s) => w.stmt(s),
            Item::Mod(m) => nested.push(m),
            _ => {}
        }
    }
    drop(w);
    for m in nested { walk_items(&mut m.items, out); }
}

/// Traits that must carry the `BoringAnyCast` supertrait: every trait implemented by a
/// struct/enum that some `as` cast targets. Over-approximate on purpose — the cast's static
/// source type is only known later, per expression, and an unused supertrait is harmless.
pub fn traits_needing_any(items: &[Item]) -> HashSet<String> {
    let tt = TypeTraits::collect(items);
    let mut out = HashSet::new();
    for target in cast_targets(items) {
        if tt.is_concrete(&target) {
            out.extend(tt.all_traits_of(&target));
        }
    }
    out
}
