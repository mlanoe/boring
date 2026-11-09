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

//! `boring check --show-types <file.br>` (D7, see docs/book.md, "Implicit generics: design notes"):
//! prints the headers the implicit-generics passes arrived at, so that a signature nobody wrote
//! can still be read. Only declarations that carry an implicit type parameter (`T_<name>`, one per
//! untyped parameter or field) are listed; a function whose return type was inferred shows it.

use crate::ast::*;
use crate::desugar_array_block::describe_type as describe_leaf;

/// A qualifier parameter as the user wrote it: the internal `q.A` of `<A'q, B'q>` is `q`.
fn written_name(q: &str) -> &str { q.split('.').next().unwrap_or(q) }

/// D10: the user never wrote `T_<name>`, so it is shown as `<name>`, the parameter or field it comes
/// from. Only a `T_` at the start of an identifier is rewritten.
fn unplace(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        let boundary = i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_');
        if boundary && b[i] == 'T' && b.get(i + 1) == Some(&'_') && b.get(i + 2).is_some_and(|c| c.is_alphabetic() || *c == '_') {
            i += 2;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// True when `ty` is the placeholder written for an omitted type on `name` itself.
fn is_own_placeholder(ty: &Type, name: &str) -> bool {
    matches!(ty, Type::TypeParam(n) if n.strip_prefix("T_") == Some(name))
}

/// `describe_type`, with the qualifier of a qualified type shown (`T'q`, `T'inline`,
/// `T'actor'observed`) instead of elided, through arrays, sets, dicts, tuples, optionals and generics.
pub(crate) fn describe_type(t: &Type) -> String {
    unplace(&describe_raw(t))
}

fn describe_raw(t: &Type) -> String {
    match t {
        Type::Qualified(..) => {
            let mut chain = Vec::new();
            let mut base = t;
            while let Type::Qualified(i, q) = base { chain.push(q.clone()); base = i; }
            chain.reverse();
            let quals: String = match chain.as_slice() {
                [OwnerQual::Param(p), rest @ ..] => {
                    let tail = if rest.is_empty() { String::new() } else { OwnerQual::chain_name(rest).map(|n| format!("'{n}")).unwrap_or_default() };
                    format!("'{}{tail}", written_name(p))
                }
                _ => match OwnerQual::chain_name(&chain) { Some(n) => format!("'{n}"), None => "'...".into() },
            };
            format!("{}{quals}", describe_type(base))
        }
        Type::Array(i) => format!("[{}]", describe_type(i)),
        Type::Set(i) => format!("{{{}}}", describe_type(i)),
        Type::Optional(i) => format!("{}?", describe_type(i)),
        Type::Dict(k, v) => format!("{{{}={}}}", describe_type(k), describe_type(v)),
        Type::Tuple(ts) => format!("({})", ts.iter().map(describe_type).collect::<Vec<_>>().join(", ")),
        Type::Generic(n, args) => format!("{n}<{}>", args.iter().map(describe_type).collect::<Vec<_>>().join(", ")),
        other => describe_leaf(other),
    }
}

fn implicit(t: &Type) -> bool {
    match t {
        Type::TypeParam(n) => n.starts_with("T_"),
        Type::Generic(_, a) | Type::Tuple(a) => a.iter().any(implicit),
        Type::Optional(i) | Type::Array(i) | Type::Set(i) => implicit(i),
        Type::Dict(k, v) => implicit(k) || implicit(v),
        _ => false,
    }
}

/// `["T", "^q@T", "U"]` -> `T'q, U`: the `<T'q>` sugar written back as it was declared.
fn generics_text(params: &[String]) -> String {
    unplace(&generics_raw(params))
}

fn generics_raw(params: &[String]) -> String {
    params.iter().filter_map(|p| {
        if !qual_param_links(p).is_empty() { return None; }
        if let Some(q) = params.iter().find_map(|o| qual_linked_to(o, p).then(|| qual_param_name(o)).flatten()) {
            return Some(format!("{p}'{}", written_name(q)));
        }
        Some(match qual_param_name(p) { Some(q) => format!("'{}", written_name(q)), None => p.clone() })
    }).collect::<Vec<_>>().join(", ")
}

fn qualified(params: &[String]) -> bool { params.iter().any(|p| !qual_param_links(p).is_empty()) }

fn where_text(clause: &[(String, String)]) -> String {
    if clause.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = clause.iter().map(|(t, b)| unplace(&format!("{t}: {b}"))).collect();
    format!(" where {}", parts.join(", "))
}

fn fn_line(f: &FnDecl, prefix: &str) -> Option<String> {
    let touched = qualified(&f.type_params) || f.params.iter().any(|p| p.ty.as_ref().is_some_and(implicit));
    if !touched {
        return None;
    }
    let generics = if f.type_params.is_empty() { String::new() } else { format!("<{}>", generics_text(&f.type_params)) };
    let params: Vec<String> = f.params.iter().map(|p| match &p.ty {
        Some(t) if is_own_placeholder(t, &p.name) => p.name.clone(),
        Some(t) => format!("{} {}", describe_type(t), p.name),
        None => p.name.clone(),
    }).collect();
    let ret = match &f.return_ty {
        Some(t) if !matches!(t, Type::Void) => format!(" -> {}", describe_type(t)),
        _ => String::new(),
    };
    Some(format!("{prefix}def {}{generics}({}){ret}{}", f.name, params.join(", "), where_text(&f.where_clause)))
}

fn collect(items: &[Item], prefix: &str, out: &mut Vec<String>) {
    for item in items {
        match item {
            Item::Fn(f) => out.extend(fn_line(f, prefix)),
            Item::Struct(s) if qualified(&s.type_params) || s.fields.iter().any(|f| implicit(&f.ty)) => {
                out.push(format!("{prefix}struct {}<{}>{}", s.name, generics_text(&s.type_params), where_text(&s.where_clause)));
                for f in &s.fields {
                    if is_own_placeholder(&f.ty, &f.name) {
                        out.push(format!("{prefix}    {}", f.name));
                    } else {
                        out.push(format!("{prefix}    {} {}", describe_type(&f.ty), f.name));
                    }
                }
            }
            Item::Mod(m) => collect(&m.items, &format!("{prefix}{}.", m.name), out),
            _ => {}
        }
    }
}

/// One line per declaration with an implicit type parameter (a struct is followed by its fields),
/// then, under `uses:`, each construction of such a struct with the types it was fixed to
/// (`3:12  Response<status = int, payload = string>`).
pub fn render(program: &Program) -> String {
    render_with_imports(program, &[])
}

/// [`render`] for a file whose `use`d files' items are known: a construction of an imported struct
/// is listed under `uses:` too (its own declaration belongs to the file that declares it).
pub fn render_with_imports(program: &Program, imported: &[Item]) -> String {
    let mut out = Vec::new();
    collect(&program.items, "", &mut out);
    let uses = crate::infer_return_types::instantiations_with_imports(program, imported);
    if !uses.is_empty() {
        out.push("uses:".to_string());
        out.extend(uses.iter().map(|u| format!("  {}:{}  {}", u.line, u.col, u.text)));
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn show(src: &str) -> String {
        render(&crate::parser::parse(crate::lexer::lex(src).expect("lex")).expect("parse"))
    }

    #[test]
    fn lists_implicit_headers_with_inferred_returns_and_bounds() {
        let out = show("def keep(x):\n    return x\n\nstruct R:\n    status\n    string text\n\nint plain(int n):\n    n\n");
        assert!(out.contains("def keep<x>(x) -> x"), "{out}");
        assert!(out.contains("struct R<status>"), "{out}");
        assert!(out.contains("    status\n") || out.contains("    status"), "{out}");
        assert!(!out.contains("T_"), "no generated name is shown: {out}");
        assert!(!out.contains("plain"), "{out}");
    }

    #[test]
    fn each_construction_shows_the_types_it_was_fixed_to() {
        let out = show("struct Resp:\n    let status\n    let payload\n\ndef a():\n    return Resp(status = 1, payload = \"x\")\n\ndef b():\n    return Resp(status = 2, payload = [1, 2])\n\nlet r = Resp(status = 3.5, payload = true)\n");
        assert!(out.contains("uses:"), "{out}");
        assert!(out.contains("  6:16  Resp<status = int, payload = string>"), "{out}");
        assert!(out.contains("  9:16  Resp<status = int, payload = [int]>"), "{out}");
        assert!(out.contains("  11:13  Resp<status = float, payload = bool>"), "{out}");
    }

    #[test]
    fn a_parameter_fixed_by_a_default_is_marked() {
        let out = show("struct Request:\n    let url\n    let retries = 0\n\nlet a = Request(url = \"/s\")\nlet b = Request(url = \"/s\", retries = 1.5)\n");
        assert!(out.contains("Request<url = string, retries = int (default)>"), "{out}");
        assert!(out.contains("Request<url = string, retries = float>"), "{out}");
    }

    #[test]
    fn untypable_constructions_and_plain_structs_are_not_listed() {
        let out = show("struct P:\n    int x\n\nstruct Resp:\n    let payload\n\ndef f(int n):\n    let p = P(x = 1)\n    return n\n");
        assert!(!out.contains("uses:"), "{out}");
    }

    #[test]
    fn qualified_generics_are_shown_as_declared() {
        let out = show("struct Entry<T'q>:\n    let T'q res\n    let T'inline fallback\n\nint peek<T'q>(T'q x):\n    1\n");
        assert!(out.contains("struct Entry<T'q>"), "{out}");
        assert!(out.contains("def peek<T'q>(T'q x) -> int"), "{out}");
        assert!(out.contains("T'q res") && out.contains("T'inline fallback"), "{out}");
    }

    #[test]
    fn qualifiers_are_shown_through_collections_and_per_type() {
        let out = show("struct Group<T'q>:\n    var [T'q] items\n    var {string = T'q} named\n    var (T'q, int) pair\n    var T'q? maybe\n\nstruct Duo<A'q, B'q>:\n    var A'q first\n    var B'q second\n");
        assert!(out.contains("[T'q] items") && out.contains("{string=T'q} named") && out.contains("(T'q, int) pair") && out.contains("T'q? maybe"), "{out}");
        assert!(out.contains("struct Duo<A'q, B'q>") && out.contains("A'q first") && out.contains("B'q second"), "{out}");
    }
}
