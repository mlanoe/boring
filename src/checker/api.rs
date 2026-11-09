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

//! `@api` — marks a `pub` function, struct or enum as part of the surface other
//! projects rely on (docs/book.md, "Implicit generics: design notes", D1).
//!
//! What it enforces:
//! - It requires `pub` (like `@provide`).
//! - Its arguments are `since = "<free text>"` and `deprecated = "<free text>"`; both
//!   are documentation only (never parsed, never emitted to Rust), anything else is an
//!   error.
//! - The **signature must be written out in full**: every parameter typed, no qualifier
//!   parameter introduced implicitly by its first occurrence, no `type let`/`type var`
//!   whose type is only inferred from its value. On a `struct`/`enum` this covers every
//!   `pub` member (fields, methods, `type` members, setters, conversions): the attribute
//!   is not repeated on each one. Non-`pub` members stay free.
//! - A warning when an `@api` signature mentions a type declared in the same program
//!   that is not itself `@api` (the equivalent of Rust's private-in-public lint): the API
//!   would depend on a type free to change. A type declared elsewhere (an imported
//!   file) is not known here and is never flagged.
//!
//! The parser lets a `def` parameter and a struct field omit its type (it writes the placeholder
//! `T_<name>`, see `infer_param_generics`); under `@api` that omission is an error, so an inferred
//! type never becomes part of a public signature. The return type is checked too: an `@api`
//! function with no written return type whose body says `return <value>` is an error
//! (`infer_return_types` skips `@api` items, so the type is never silently written back).

use super::*;

/// Keys `@api(...)` accepts; both values are free text.
const API_KEYS: &[&str] = &["since", "deprecated"];

/// True when `ty` is the placeholder the implicit-generics passes write for an omitted type
/// (`T_<name>`, see `infer_param_generics`): the user wrote no type there.
fn is_implicit_param(ty: &Type, name: &str) -> bool {
    matches!(ty, Type::TypeParam(n) if n.strip_prefix("T_") == Some(name))
}

fn api_attr(attrs: &[Attr]) -> Option<&Attr> {
    attrs.iter().find(|a| a.name == "api")
}

pub(super) fn has_api(attrs: &[Attr]) -> bool {
    api_attr(attrs).is_some()
}

/// Every user-visible type name written in `ty`, for the exposure lint.
fn named_types(ty: &Type, out: &mut Vec<String>) {
    match ty {
        Type::Named(n) => out.push(n.clone()),
        Type::Generic(n, args) => {
            out.push(n.clone());
            for a in args {
                named_types(a, out);
            }
        }
        Type::Optional(i) | Type::Array(i) | Type::Set(i) | Type::Qualified(i, _) => named_types(i, out),
        Type::ArrayN(i, _) => named_types(i, out),
        Type::Dict(k, v) => {
            named_types(k, out);
            named_types(v, out);
        }
        Type::Tuple(es) => {
            for e in es {
                named_types(e, out);
            }
        }
        Type::Fn(ret, params, ..) => {
            if let Some(r) = ret {
                named_types(r, out);
            }
            for p in params {
                named_types(p, out);
            }
        }
        _ => {}
    }
}

/// Splits one raw `@api` argument (`since="1.0"`, spaces already dropped by the parser)
/// into `(key, free text)`; `Err` carries what is wrong with it.
fn parse_api_arg(arg: &str) -> Result<(&str, &str), String> {
    let Some((key, value)) = arg.split_once('=') else {
        return Err(format!(
            "`@api` takes `since = \"...\"` and `deprecated = \"...\"`, got `{arg}`"
        ));
    };
    let key = key.trim();
    if !API_KEYS.contains(&key) {
        return Err(format!(
            "unknown `@api` argument `{key}` (expected `since` or `deprecated`)"
        ));
    }
    let value = value.trim();
    if value.len() < 2 || !value.starts_with('"') || !value.ends_with('"') {
        return Err(format!(
            "`@api` argument `{key}` must be a string literal (free text), e.g. `{key} = \"...\"`"
        ));
    }
    Ok((key, &value[1..value.len() - 1]))
}

impl Checker {
    /// Records whether a top-level struct/enum is `@api`, for the exposure lint.
    pub(super) fn note_api_type(&mut self, name: &str, attrs: &[Attr]) {
        self.api_types.insert(name.to_string(), has_api(attrs));
    }

    /// `pub` requirement and argument validation, shared by every kind of item.
    fn check_api_attr(&mut self, attr: &Attr, is_pub: bool, what: &str) {
        if !is_pub {
            self.error(
                format!(
                    "`@api` requires `pub` — {what} is not visible to other projects, so it \
                     cannot be part of an API (see `@api` in docs/book.md)"
                ),
                attr.line,
                attr.col,
            );
        }
        let mut seen: Vec<&str> = Vec::new();
        for arg in &attr.args {
            match parse_api_arg(arg) {
                Ok((key, _)) => {
                    if seen.contains(&key) {
                        self.error(format!("`@api` argument `{key}` is given more than once"), attr.line, attr.col);
                    }
                    seen.push(key);
                }
                Err(msg) => self.error(msg, attr.line, attr.col),
            }
        }
    }

    /// Warns for each type written in `ty` that this program declares without `@api`.
    fn check_api_exposure(&mut self, owner: &str, ty: &Type, line: usize, col: usize) {
        let mut names = Vec::new();
        named_types(ty, &mut names);
        names.sort();
        names.dedup();
        for n in names {
            if self.api_types.get(&n) == Some(&false) {
                self.warning(
                    format!(
                        "`@api` item `{owner}` exposes `{n}` in its signature, but `{n}` is not \
                         `@api`: the public API would depend on a type that is free to change"
                    ),
                    line,
                    col,
                );
            }
        }
    }

    /// A function-like signature under `@api`: parameters typed, qualifier parameters declared,
    /// no hidden exposure.
    #[allow(clippy::too_many_arguments)]
    fn check_api_sig(
        &mut self,
        owner: &str,
        params: &[Param],
        return_ty: &Option<Type>,
        body: &[Stmt],
        implicit_quals: &[(String, usize)],
        line: usize,
        col: usize,
    ) {
        for p in params {
            match &p.ty {
                Some(t) if is_implicit_param(t, &p.name) => self.error(
                    format!(
                        "parameter `{}` of `{owner}` has no type: an `@api` signature must be written out in full",
                        p.name
                    ),
                    p.line,
                    p.col,
                ),
                None if !p.variadic => self.error(
                    format!(
                        "parameter `{}` of `{owner}` has no type: an `@api` signature must be written out in full",
                        p.name
                    ),
                    p.line,
                    p.col,
                ),
                Some(t) => self.check_api_exposure(owner, t, p.line, p.col),
                None => {}
            }
        }
        match return_ty {
            // The parser gives an omitted return type `Some(Void)`. `void` (written or not) with a
            // `return <value>` in the body means the type was left out, and the implicit
            // return-type inference deliberately skips `@api` items.
            None | Some(Type::Void)
                if crate::infer_return_types::valued_returns(body).is_some_and(|r| !r.is_empty()) =>
            {
                self.error(
                    format!(
                        "`{owner}` returns a value but has no written return type: an `@api` \
                         signature must be written out in full"
                    ),
                    line,
                    col,
                );
            }
            Some(t) => self.check_api_exposure(owner, t, line, col),
            None => {}
        }
        for (q, _) in implicit_quals {
            self.error(
                format!(
                    "qualifier parameter `{q}` of `{owner}` is introduced implicitly by its first use: \
                     an `@api` signature must declare it, write `<'{q}>`"
                ),
                line,
                col,
            );
        }
    }

    /// Top-level `@api` function (a method of an `@api` type goes through `check_api_struct`/`_enum`).
    pub(super) fn check_api_fn(&mut self, f: &FnDecl) {
        if self.kernel_dispatch_only {
            return;
        }
        let Some(attr) = api_attr(&f.attrs) else { return };
        let what = format!("function `{}`", f.name);
        self.check_api_attr(attr, f.is_pub, &what);
        self.check_api_sig(&f.name, &f.params, &f.return_ty, &f.body, &f.implicit_quals, f.line, f.col);
    }

    /// The members an `@api` type covers: every `pub` method and `type` method.
    fn check_api_members(&mut self, owner: &str, methods: &[FnDecl], type_methods: &[TypeMethod]) {
        for m in methods.iter().filter(|m| m.is_pub) {
            let name = format!("{owner}.{}", m.name);
            self.check_api_sig(&name, &m.params, &m.return_ty, &m.body, &m.implicit_quals, m.line, m.col);
        }
        for m in type_methods.iter().filter(|m| m.is_pub) {
            let name = format!("{owner}.{}", m.name);
            self.check_api_sig(&name, &m.params, &m.return_ty, &m.body, &[], m.line, m.col);
        }
    }

    pub(super) fn check_api_struct(&mut self, s: &StructDecl) {
        if self.kernel_dispatch_only {
            return;
        }
        let Some(attr) = api_attr(&s.attrs) else { return };
        self.check_api_attr(attr, s.is_pub, &format!("`{}`", s.name));
        for f in s.fields.iter().filter(|f| f.is_pub) {
            if is_implicit_param(&f.ty, &f.name) {
                self.error(
                    format!(
                        "field `{}.{}` has no type: an `@api` signature must be written out in full \
                         (an inferred field type would silently become part of the API)",
                        s.name, f.name
                    ),
                    f.line,
                    f.col,
                );
            }
            self.check_api_exposure(&format!("{}.{}", s.name, f.name), &f.ty, f.line, f.col);
        }
        self.check_api_members(&s.name, &s.methods, &s.type_methods);
        for tv in s.type_vars.iter().filter(|tv| tv.is_pub) {
            let name = format!("{}.{}", s.name, tv.name);
            match &tv.ty {
                None => self.error(
                    format!(
                        "`{name}` has no written type: an `@api` signature must be written out in full \
                         (its type is only inferred from its value)"
                    ),
                    tv.line,
                    tv.col,
                ),
                Some(t) => self.check_api_exposure(&name, t, tv.line, tv.col),
            }
        }
        for sd in s.setters.iter().filter(|sd| sd.is_pub) {
            self.check_api_exposure(&format!("{}.{}", s.name, sd.name), &sd.param_ty, sd.line, sd.col);
        }
        for c in s.conversions.iter().filter(|c| c.is_pub) {
            self.check_api_exposure(&format!("{} as", s.name), &c.ty, c.line, c.col);
        }
    }

    pub(super) fn check_api_enum(&mut self, e: &EnumDecl) {
        if self.kernel_dispatch_only {
            return;
        }
        let Some(attr) = api_attr(&e.attrs) else { return };
        self.check_api_attr(attr, e.is_pub, &format!("`{}`", e.name));
        // A variant is as public as its enum.
        for v in &e.variants {
            for f in &v.fields {
                self.check_api_exposure(&format!("{}.{}", e.name, v.name), &f.ty, v.line, v.col);
            }
        }
        self.check_api_members(&e.name, &e.methods, &e.type_methods);
        for sd in e.setters.iter().filter(|sd| sd.is_pub) {
            self.check_api_exposure(&format!("{}.{}", e.name, sd.name), &sd.param_ty, sd.line, sd.col);
        }
        for c in e.conversions.iter().filter(|c| c.is_pub) {
            self.check_api_exposure(&format!("{} as", e.name), &c.ty, c.line, c.col);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::*;
    use crate::lexer::lex;
    use crate::parser::parse;

    fn program(src: &str) -> Program {
        parse(lex(src).expect("lex error")).expect("parse error")
    }

    fn errors(p: &Program) -> Vec<String> {
        super::super::check(p).errors.into_iter().map(|e| e.message).collect()
    }

    fn warnings(p: &Program) -> Vec<String> {
        super::super::check(p).warnings.into_iter().map(|w| w.message).collect()
    }

    fn any(msgs: &[String], needle: &str) -> bool {
        msgs.iter().any(|m| m.contains(needle))
    }

    #[test]
    fn pub_api_function_is_accepted() {
        let p = program("@api(since = \"1.0\")\npub def int add(int a, int b):\n    return a + b\n");
        assert!(errors(&p).is_empty(), "unexpected errors: {:?}", errors(&p));
    }

    #[test]
    fn bare_api_is_accepted() {
        let p = program("@api\npub def int one():\n    return 1\n");
        assert!(errors(&p).is_empty(), "unexpected errors: {:?}", errors(&p));
    }

    #[test]
    fn api_requires_pub() {
        let p = program("@api\ndef int one():\n    return 1\n");
        assert!(any(&errors(&p), "`@api` requires `pub`"), "got {:?}", errors(&p));
    }

    #[test]
    fn unknown_argument_is_rejected() {
        let p = program("@api(version = \"1.0\")\npub def int one():\n    return 1\n");
        assert!(any(&errors(&p), "unknown `@api` argument `version`"), "got {:?}", errors(&p));
    }

    #[test]
    fn since_must_be_a_string_literal() {
        let p = program("@api(since = 1)\npub def int one():\n    return 1\n");
        assert!(any(&errors(&p), "must be a string literal"), "got {:?}", errors(&p));
    }

    #[test]
    fn duplicate_argument_is_rejected() {
        let p = program("@api(since = \"1\", since = \"2\")\npub def int one():\n    return 1\n");
        assert!(any(&errors(&p), "more than once"), "got {:?}", errors(&p));
    }

    #[test]
    fn api_function_returning_a_value_needs_a_written_return_type() {
        let p = program("@api\npub def one():\n    return 1\n");
        assert!(any(&errors(&p), "returns a value but has no written return type"), "got {:?}", errors(&p));
    }

    #[test]
    fn api_function_without_a_returned_value_is_void_and_fine() {
        let p = program("@api\npub def hello():\n    print 1\n");
        assert!(errors(&p).is_empty(), "unexpected errors: {:?}", errors(&p));
    }

    #[test]
    fn non_api_function_returning_a_value_gets_its_type_inferred_not_an_error() {
        let p = program("pub def one():\n    return 1\n");
        assert!(errors(&p).is_empty(), "unexpected errors: {:?}", errors(&p));
        match &p.items[0] {
            Item::Fn(f) => assert_eq!(f.return_ty, Some(Type::Named("int".into())), "expected an inferred return type"),
            other => panic!("expected a fn, got {other:?}"),
        }
    }

    #[test]
    fn untyped_parameter_is_rejected_once_the_parser_allows_it() {
        // The parser still requires a type; simulate the relaxed parser by clearing it.
        let mut p = program("@api\npub def int twice(int x):\n    return x * 2\n");
        match &mut p.items[0] {
            Item::Fn(f) => f.params[0].ty = None,
            other => panic!("expected a fn, got {other:?}"),
        }
        assert!(any(&errors(&p), "parameter `x` of `twice` has no type"), "got {:?}", errors(&p));
    }

    #[test]
    fn implicit_field_and_parameter_are_rejected_under_api() {
        let p = program("@api\npub struct R:\n    pub let status\n    pub let int n\n    let hidden\n");
        let errs = errors(&p);
        assert!(any(&errs, "field `R.status` has no type"), "got {errs:?}");
        assert!(!any(&errs, "R.n") && !any(&errs, "R.hidden"), "got {errs:?}");
        let p = program("@api\npub def show(x):\n    print \"{x}\"\n");
        assert!(any(&errors(&p), "parameter `x` of `show` has no type"), "got {:?}", errors(&p));
        let p = program("pub struct R:\n    pub let status\n");
        assert!(!any(&errors(&p), "has no type"), "got {:?}", errors(&p));
    }

    #[test]
    fn untyped_parameter_is_fine_without_api() {
        let mut p = program("pub def int twice(int x):\n    return x * 2\n");
        match &mut p.items[0] {
            Item::Fn(f) => f.params[0].ty = None,
            other => panic!("expected a fn, got {other:?}"),
        }
        assert!(!any(&errors(&p), "has no type"), "got {:?}", errors(&p));
    }

    #[test]
    fn implicit_qualifier_parameter_is_rejected() {
        let mut p = program("@api\npub def int one():\n    return 1\n");
        match &mut p.items[0] {
            Item::Fn(f) => f.implicit_quals.push(("q".into(), 2)),
            other => panic!("expected a fn, got {other:?}"),
        }
        assert!(any(&errors(&p), "declare it, write `<'q>`"), "got {:?}", errors(&p));
    }

    #[test]
    fn api_on_a_type_covers_pub_members_only() {
        let mut p = program(
            "@api\npub struct Counter:\n    var int n = 0\n    pub def int get():\n        return n\n    def int hidden():\n        return n\n",
        );
        match &mut p.items[0] {
            Item::Struct(s) => {
                for m in &mut s.methods {
                    m.params.push(Param {
                        name: "x".into(),
                        ty: None,
                        mutable: false,
                        rebindable: false,
                        var_mut: false,
                        owned: false,
                        variadic: false,
                        default: None,
                        line: 1,
                        col: 1,
                    });
                }
            }
            other => panic!("expected a struct, got {other:?}"),
        }
        let errs = errors(&p);
        let hits: Vec<_> = errs.iter().filter(|m| m.contains("has no type")).collect();
        assert_eq!(hits.len(), 1, "only the pub method is covered, got {errs:?}");
        assert!(hits[0].contains("Counter.get"), "got {errs:?}");
    }

    #[test]
    fn pub_type_member_with_inferred_type_is_rejected() {
        let mut p = program("@api\npub struct S:\n    type let N = 5\n");
        match &mut p.items[0] {
            Item::Struct(s) => {
                let tv = s.type_vars.first_mut().expect("a type let");
                tv.is_pub = true;
                tv.ty = None;
            }
            other => panic!("expected a struct, got {other:?}"),
        }
        assert!(any(&errors(&p), "`S.N` has no written type"), "got {:?}", errors(&p));
    }

    #[test]
    fn exposing_a_non_api_type_warns() {
        let p = program(
            "struct Inner:\n    int v = 0\n\n@api\npub def Inner make():\n    return Inner()\n",
        );
        assert!(
            any(&warnings(&p), "exposes `Inner` in its signature"),
            "got {:?}",
            warnings(&p)
        );
    }

    #[test]
    fn exposing_an_api_type_does_not_warn() {
        let p = program(
            "@api\npub struct Inner:\n    int v = 0\n\n@api\npub def Inner make():\n    return Inner()\n",
        );
        assert!(!any(&warnings(&p), "exposes"), "got {:?}", warnings(&p));
    }
}
