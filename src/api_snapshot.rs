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

//! `boring api <file.br>` / `boring api --check <snapshot> <file.br>`
//! (D1, see docs/book.md, "Implicit generics: design notes").
//!
//! `@api` guarantees the *form* of a signature (it is written out in full); this module
//! guarantees *stability* by comparing the signatures of every `@api` item with a snapshot taken
//! earlier. A snapshot is plain text, one `key<TAB>signature` line per public entry (`#` lines are
//! comments), so it diffs well in version control.
//!
//! Comparison rules, deliberately conservative:
//! - an entry of the snapshot that is gone, or whose signature differs, is **breaking**;
//! - a new entry is **compatible** (reported, never fails the check);
//! - `since`/`deprecated` are documentation and take no part in it.

use crate::ast::*;
use crate::show_types::describe_type;
use std::collections::BTreeMap;

pub const HEADER: &str = "# boring api snapshot (one `key<TAB>signature` per line; `boring api --check` compares against it)";

fn is_api(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| a.name == "api")
}

fn generics(params: &[String]) -> String {
    if params.is_empty() { String::new() } else { format!("<{}>", params.join(", ")) }
}

fn ty(t: &Option<Type>) -> String {
    match t {
        Some(t) => describe_type(t),
        None => "?".into(),
    }
}

fn params_text(params: &[Param]) -> String {
    params
        .iter()
        .map(|p| {
            let mut s = String::new();
            if p.var_mut {
                s.push_str("var mut ")
            } else if p.rebindable {
                s.push_str("var ")
            } else if p.mutable {
                s.push_str("mut ")
            }
            s.push_str(&ty(&p.ty));
            if p.variadic {
                s.push_str("...")
            }
            s.push(' ');
            s.push_str(&p.name);
            if p.default.is_some() {
                s.push_str(" = ...")
            }
            s
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn fn_sig(kw: &str, f: &FnDecl) -> String {
    let ret = match &f.return_ty {
        Some(t) if !matches!(t, Type::Void) => format!(" -> {}{}", if f.return_mutable { "mut " } else { "" }, describe_type(t)),
        _ => String::new(),
    };
    let throws = if f.throws { " throws" } else { "" };
    format!("{kw} {}{}({}){ret}{throws}", f.name, generics(&f.type_params), params_text(&f.params))
}

fn type_method_sig(m: &TypeMethod) -> String {
    let kw = match m.kind {
        TypeMethodKind::Def => "type def",
        TypeMethodKind::Req => "type req",
        TypeMethodKind::Set => "type set",
    };
    let ret = match &m.return_ty {
        Some(t) if !matches!(t, Type::Void) => format!(" -> {}", describe_type(t)),
        _ => String::new(),
    };
    format!("{kw} {}({}){ret}{}", m.name, params_text(&m.params), if m.throws { " throws" } else { "" })
}

fn members(owner: &str, methods: &[FnDecl], type_methods: &[TypeMethod], out: &mut BTreeMap<String, String>) {
    for m in methods.iter().filter(|m| m.is_pub) {
        let kw = if m.mutating { "def" } else { "req" };
        out.insert(format!("{owner}.{}", m.name), fn_sig(kw, m));
    }
    for m in type_methods.iter().filter(|m| m.is_pub) {
        out.insert(format!("{owner}::{}", m.name), type_method_sig(m));
    }
}

/// Every public entry of every `@api` item of `program`, keyed and sorted.
pub fn collect(program: &Program) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for item in &program.items {
        match item {
            Item::Fn(f) if f.is_pub && is_api(&f.attrs) => {
                out.insert(f.name.clone(), fn_sig("def", f));
            }
            Item::Struct(s) if s.is_pub && is_api(&s.attrs) => {
                let kind = if s.is_view { "view" } else { "struct" };
                out.insert(s.name.clone(), format!("{kind} {}{}", s.name, generics(&s.type_params)));
                for f in s.fields.iter().filter(|f| f.is_pub) {
                    let m = if f.mutable { "var " } else { "" };
                    out.insert(format!("{}.{}", s.name, f.name), format!("field {m}{} {}", describe_type(&f.ty), f.name));
                }
                members(&s.name, &s.methods, &s.type_methods, &mut out);
                for tv in s.type_vars.iter().filter(|tv| tv.is_pub) {
                    let kw = if tv.mutable { "type var" } else { "type let" };
                    out.insert(format!("{}::{}", s.name, tv.name), format!("{kw} {} {}", ty(&tv.ty), tv.name));
                }
            }
            Item::Enum(e) if e.is_pub && is_api(&e.attrs) => {
                out.insert(e.name.clone(), format!("enum {}{}", e.name, generics(&e.type_params)));
                for v in &e.variants {
                    let fields: Vec<String> = v
                        .fields
                        .iter()
                        .map(|f| match &f.name {
                            Some(n) => format!("{} {n}", describe_type(&f.ty)),
                            None => describe_type(&f.ty),
                        })
                        .collect();
                    out.insert(format!("{}.{}", e.name, v.name), format!("variant {}({})", v.name, fields.join(", ")));
                }
                members(&e.name, &e.methods, &e.type_methods, &mut out);
            }
            _ => {}
        }
    }
    out
}

/// The snapshot text for `program`.
pub fn render(program: &Program) -> String {
    let mut s = String::from(HEADER);
    s.push('\n');
    for (k, v) in collect(program) {
        s.push_str(&format!("{k}\t{v}\n"));
    }
    s
}

/// Parses a snapshot back; `Err` names the first malformed line.
pub fn parse_snapshot(text: &str) -> Result<BTreeMap<String, String>, String> {
    let mut m = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('\t') else {
            return Err(format!("line {}: expected `key<TAB>signature`, got `{line}`", i + 1));
        };
        m.insert(k.to_string(), v.to_string());
    }
    Ok(m)
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct Diff {
    pub removed: Vec<(String, String)>,
    pub changed: Vec<(String, String, String)>,
    pub added: Vec<(String, String)>,
}

impl Diff {
    pub fn breaking(&self) -> bool {
        !self.removed.is_empty() || !self.changed.is_empty()
    }
}

pub fn diff(old: &BTreeMap<String, String>, new: &BTreeMap<String, String>) -> Diff {
    let mut d = Diff::default();
    for (k, o) in old {
        match new.get(k) {
            None => d.removed.push((k.clone(), o.clone())),
            Some(n) if n != o => d.changed.push((k.clone(), o.clone(), n.clone())),
            _ => {}
        }
    }
    for (k, n) in new {
        if !old.contains_key(k) {
            d.added.push((k.clone(), n.clone()));
        }
    }
    d
}

/// Human report of a diff, breaking changes first.
pub fn report(d: &Diff) -> String {
    let mut s = String::new();
    for (k, o) in &d.removed {
        s.push_str(&format!("breaking: `{k}` was removed (was `{o}`)\n"));
    }
    for (k, o, n) in &d.changed {
        s.push_str(&format!("breaking: `{k}` changed\n    was: {o}\n    now: {n}\n"));
    }
    for (k, n) in &d.added {
        s.push_str(&format!("added:    `{k}` (`{n}`)\n"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(src: &str) -> BTreeMap<String, String> {
        collect(&crate::parser::parse(crate::lexer::lex(src).expect("lex")).expect("parse"))
    }

    const V1: &str = "@api\npub struct Counter:\n    pub var int n = 0\n    pub def inc(int by):\n        n += by\n    pub req int get():\n        n\n    def hidden():\n        n += 1\n\n@api(since = \"1.0\")\npub int twice(int x):\n    x * 2\n\npub int unmarked(int x):\n    x\n";

    #[test]
    fn only_pub_members_of_api_items_are_listed() {
        let s = snap(V1);
        assert!(
            s.contains_key("Counter") && s.contains_key("Counter.n") && s.contains_key("Counter.inc") && s.contains_key("Counter.get") && s.contains_key("twice"),
            "{s:?}"
        );
        assert!(!s.contains_key("Counter.hidden") && !s.contains_key("unmarked"), "{s:?}");
        assert_eq!(s["twice"], "def twice(int x) -> int");
        assert_eq!(s["Counter.get"], "req get() -> int");
    }

    #[test]
    fn snapshot_roundtrips_and_an_identical_program_has_no_diff() {
        let p = crate::parser::parse(crate::lexer::lex(V1).unwrap()).unwrap();
        let old = parse_snapshot(&render(&p)).unwrap();
        assert_eq!(old, collect(&p));
        assert_eq!(diff(&old, &collect(&p)), Diff::default());
    }

    #[test]
    fn removal_and_signature_change_are_breaking_addition_is_not() {
        let old = snap(V1);
        let new = snap(
            &(V1.replace("pub int twice(int x):", "pub float twice(float x):").replace("    pub req int get():\n        n\n", "")
                + "\n@api\npub int thrice(int x):\n    x * 3\n"),
        );
        let d = diff(&old, &new);
        assert!(d.breaking());
        assert_eq!(d.removed.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(), vec!["Counter.get"]);
        assert_eq!(d.changed.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(), vec!["twice"]);
        assert_eq!(d.added.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(), vec!["thrice"]);
        let only_added = diff(&old, &snap(&(V1.to_string() + "\n@api\npub int thrice(int x):\n    x * 3\n")));
        assert!(!only_added.breaking() && only_added.added.len() == 1);
    }

    #[test]
    fn since_does_not_take_part() {
        let a = snap(V1);
        let b = snap(&V1.replace("since = \"1.0\"", "since = \"2.0\", deprecated = \"use thrice\""));
        assert_eq!(a, b);
    }

    #[test]
    fn enum_variants_are_listed() {
        let s = snap("@api\npub enum Shape:\n    Circle(float r)\n    Rect(float, float)\n");
        assert_eq!(s["Shape.Circle"], "variant Circle(float r)");
        assert_eq!(s["Shape.Rect"], "variant Rect(float, float)");
    }

    #[test]
    fn malformed_snapshot_line_is_named() {
        assert!(parse_snapshot("# c\nnotabs\n").unwrap_err().contains("line 2"));
    }
}
