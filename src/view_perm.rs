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

//! Default-permission resolution for `view` `state`/`bind` fields
//! (docs/design-notes/boring-ui-draft.md, "Implicit and explicit binding permissions").
//!
//! The parser only records whether `let`/`var`/`mut` was written (`FieldDecl::explicit_perm`);
//! the default depends on the value's *category*, which needs the program's declaration table.
//! This pass runs once on the merged `Program` (same pipeline slot as `desugar_inject`) and
//! rewrites each implicit-permission field into the exact `mutable` / `Type::Mut` encoding the
//! explicit spelling would have produced, so the checker, interpreter and a future lowering only
//! ever see one representation. `explicit_perm` is left `false` so tools can still tell the
//! source spelling apart.
//!
//! The classification is closed and predictable — never inferred from how the body uses the field:
//!
//! - primitives, strings, enums, newtypes, tuples, function values and every optional → `var`
//! - structs, arrays, dictionaries and sets → `mut`
//! - trait objects, generic parameters and opaque/external types → error, an explicit
//!   permission is required
//!
//! An `'observed` field is outside this sugar (it follows the `'actor`/`'guard` table) and is
//! left untouched; `env` fields are always read-only and never rewritten.

use std::collections::{HashMap, HashSet};

use crate::ast::*;
use crate::parser::ParseError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Category {
    /// Change operation is replacement → `var`.
    Value,
    /// Change operation is in-place mutation → `mut`.
    Mutable,
    /// The checker can't classify the mutation surface.
    Unclassifiable,
}

struct Decls {
    structs: HashSet<String>,
    enums: HashSet<String>,
    aliases: HashMap<String, (Type, bool)>,
}

impl Decls {
    fn collect(items: &[Item], out: &mut Decls) {
        for item in items {
            match item {
                Item::Struct(s) if !s.is_view => { out.structs.insert(s.name.clone()); }
                Item::Enum(e) => { out.enums.insert(e.name.clone()); }
                Item::Alias(a) => { out.aliases.insert(a.name.clone(), (a.ty.clone(), a.newtype)); }
                Item::Mod(m) => Self::collect(&m.items, out),
                _ => {}
            }
        }
    }

    fn classify(&self, ty: &Type, depth: usize) -> Category {
        use Category::*;
        if depth > 16 {
            return Unclassifiable;
        }
        match ty {
            Type::Int | Type::Uint | Type::Uint8 | Type::Int8 | Type::Int16 | Type::Int32
            | Type::Int64 | Type::Int128 | Type::Uint16 | Type::Uint32 | Type::Uint64
            | Type::Uint128 | Type::Float32 | Type::Float64 | Type::Str | Type::Bool
            | Type::Nil | Type::Void | Type::Never | Type::ConstInt(_)
            | Type::Tuple(_) | Type::Optional(_) | Type::Fn(..) => Value,
            Type::Array(_) | Type::ArrayN(..) | Type::ArrayNExpr(..) | Type::LabeledArray(..)
            | Type::Dict(..) | Type::Set(_) => Mutable,
            Type::Mut(inner) => self.classify(inner, depth + 1),
            Type::Qualified(inner, q) => match q {
                // `'shared` has no interior mutability: replacement is the only change.
                OwnerQual::Shared => Value,
                _ => self.classify(inner, depth + 1),
            },
            Type::Named(n) | Type::Generic(n, _) => {
                if is_primitive_name(n) || self.enums.contains(n) {
                    Value
                } else if self.structs.contains(n) {
                    Mutable
                } else if let Some((aliased, newtype)) = self.aliases.get(n) {
                    if *newtype { Value } else { self.classify(aliased, depth + 1) }
                } else {
                    Unclassifiable
                }
            }
            Type::Dyn(_) | Type::Impl(_) | Type::TypeParam(_) | Type::SelfAssoc(_)
            | Type::AssocOf(..) => Unclassifiable,
        }
    }
}

/// Primitive type names the parser leaves as `Type::Named` (`int`, `float`, `string`, ...).
fn is_primitive_name(n: &str) -> bool {
    matches!(
        n,
        "int" | "uint" | "float" | "bool" | "string" | "byte" | "char"
            | "int8" | "int16" | "int32" | "int64" | "int128"
            | "uint8" | "uint16" | "uint32" | "uint64" | "uint128"
            | "float32" | "float64"
    )
}

fn has_observed(ty: &Type) -> bool {
    match ty {
        Type::Qualified(inner, q) => matches!(q, OwnerQual::Observed) || has_observed(inner),
        Type::Mut(inner) => has_observed(inner),
        _ => false,
    }
}

/// Wraps a type in `Type::Mut`, like `Parser::wrap_type_mut` does for an explicit `mut`.
fn wrap_mut(ty: Type) -> Type {
    match ty {
        Type::Qualified(inner, OwnerQual::Borrow) => Type::Qualified(inner, OwnerQual::BorrowMut),
        Type::Qualified(inner, OwnerQual::BorrowOption) => Type::Qualified(inner, OwnerQual::BorrowOptionMut),
        already @ Type::Mut(_) => already,
        other => Type::Mut(Box::new(other)),
    }
}

fn resolve_items(items: &mut [Item], decls: &Decls) -> Result<(), ParseError> {
    for item in items {
        match item {
            Item::Struct(s) if s.is_view => resolve_view(s, decls)?,
            Item::Mod(m) => resolve_items(&mut m.items, decls)?,
            _ => {}
        }
    }
    Ok(())
}

fn resolve_view(s: &mut StructDecl, decls: &Decls) -> Result<(), ParseError> {
    for f in &mut s.fields {
        let kind = match f.view_kind {
            Some(k @ (ViewFieldKind::State | ViewFieldKind::Bind)) => k,
            _ => continue,
        };
        if f.explicit_perm || has_observed(&f.ty) {
            continue;
        }
        match decls.classify(&f.ty, 0) {
            Category::Value => f.mutable = true,
            Category::Mutable => f.ty = wrap_mut(f.ty.clone()),
            Category::Unclassifiable => {
                let word = if kind == ViewFieldKind::State { "state" } else { "bind" };
                return Err(ParseError::Generic {
                    msg: format!(
                        "`{word} {}` holds a trait object, generic or opaque value whose mutation surface \
                         can't be classified — write `{word} var`, `{word} mut` or `{word} var mut` explicitly",
                        f.name
                    ),
                    line: f.line, col: f.col, len: 1,
                });
            }
        }
    }
    Ok(())
}

/// Resolves the default permission of every implicit `state`/`bind` field in the program.
pub fn resolve_view_permissions(mut program: Program) -> Result<Program, ParseError> {
    let mut decls = Decls { structs: HashSet::new(), enums: HashSet::new(), aliases: HashMap::new() };
    Decls::collect(&program.items, &mut decls);
    resolve_items(&mut program.items, &decls)?;
    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lexer::lex, parser::parse};

    fn resolved(src: &str) -> Result<Program, ParseError> {
        resolve_view_permissions(parse(lex(src).unwrap()).unwrap())
    }

    /// (mutable, is_mut) of the named field of the first view.
    fn perm(p: &Program, field: &str) -> (bool, bool) {
        for item in &p.items {
            if let Item::Struct(s) = item {
                if s.is_view {
                    let f = s.fields.iter().find(|f| f.name == field).unwrap();
                    return (f.mutable, f.ty.is_mut());
                }
            }
        }
        panic!("no view")
    }

    const PRELUDE: &str = "struct Profile:\n    var string name = \"\"\nenum Field:\n    A\n    B\ntype Token as string\nuse Names as [string]\n";

    #[test]
    fn value_like_types_default_to_var() {
        let src = format!("{PRELUDE}view V:\n    state int n = 0\n    state string s = \"\"\n    state Field f = Field.A\n    state Field? o = nil\n    state Token t = Token(\"x\")\n    state (int, int) p = (1, 2)\n    bind string name\n");
        let p = resolved(&src).unwrap();
        for name in ["n", "s", "f", "o", "t", "p", "name"] {
            assert_eq!(perm(&p, name), (true, false), "{name}");
        }
    }

    #[test]
    fn structs_and_collections_default_to_mut() {
        let src = format!("{PRELUDE}view V:\n    state Profile pr = Profile()\n    state [int] xs = []\n    state {{string=int}} d = {{=}}\n    state {{int}} st = {{}}\n    state Names ns = []\n    bind Profile owner\n");
        let p = resolved(&src).unwrap();
        for name in ["pr", "xs", "d", "st", "ns", "owner"] {
            assert_eq!(perm(&p, name), (false, true), "{name}");
        }
    }

    #[test]
    fn explicit_permission_wins() {
        let src = format!("{PRELUDE}view V:\n    state var Profile pr = Profile()\n    state let int n = 1\n    state var mut Profile q = Profile()\n");
        let p = resolved(&src).unwrap();
        assert_eq!(perm(&p, "pr"), (true, false));
        assert_eq!(perm(&p, "n"), (false, false));
        assert_eq!(perm(&p, "q"), (true, true));
    }

    #[test]
    fn env_and_observed_are_left_alone() {
        let src = format!("{PRELUDE}view V:\n    env Profile theme\n    Profile'actor'observed shared\n");
        let p = resolved(&src).unwrap();
        assert_eq!(perm(&p, "theme"), (false, false));
    }

    #[test]
    fn unclassifiable_requires_explicit_permission() {
        let err = resolved("view V:\n    state Mystery m = Mystery()\n").unwrap_err();
        assert!(err.msg().contains("can't be classified"), "{}", err.msg());
        assert!(resolved("view V:\n    state var Mystery m = Mystery()\n").is_ok());
    }
}
