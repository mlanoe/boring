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

//! Labelled generic arguments (`Pair<K = int, V = string>`, `Grid<N = 4>`;
//! docs/design-notes/labelled-generic-args.md).
//!
//! Every generic argument list (constructor/call, method turbofish, type position, `ext`
//! header) is read by `parse_generic_arg_list`, which accepts positional arguments followed
//! by `Label = value` ones and returns plain positional `Vec<Type>`: everything downstream
//! (`ExprKind::GenericCall`, `Type::Generic`, checker, interpreter, monomorphizer, emitters)
//! only ever sees positional arguments.
//!
//! A label names a parameter of the callee's declaration, which may live in another file, so
//! `parse` reads a file that contains labels twice: a first parse in *lenient* mode (no
//! declaration table: labels are dropped, values kept in written order) only to collect the
//! declarations of the file and of the files it `use`s, then the real parse with the table,
//! which resolves each labelled list and reports its errors with positions. A file without
//! any label is parsed once and never pays for this.
use super::*;
use crate::ast::*;
use crate::lexer::TokenKind;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Declared generic parameters (`"T"`, `"$N:usize"`, `"'a"`), by declaration name.
#[derive(Debug, Default)]
pub(crate) struct GenericDecls {
    /// struct / enum / trait / alias name -> its parameter lists (several when declared twice).
    types: HashMap<String, Vec<Vec<String>>>,
    fns: HashMap<String, Vec<Vec<String>>>,
    /// method name (`m`) and type-qualified method name (`Type.m`).
    methods: HashMap<String, Vec<Vec<String>>>,
}

impl GenericDecls {
    fn add(map: &mut HashMap<String, Vec<Vec<String>>>, name: &str, params: &[String]) {
        let lists = map.entry(name.to_string()).or_default();
        if !lists.iter().any(|l| l == params) { lists.push(params.to_vec()); }
    }

    fn add_methods(&mut self, owner: &str, methods: &[FnDecl]) {
        for m in methods {
            Self::add(&mut self.methods, &m.name, &m.type_params);
            Self::add(&mut self.methods, &format!("{owner}.{}", m.name), &m.type_params);
        }
    }

    fn collect(&mut self, items: &[Item]) {
        for it in items {
            match it {
                Item::Struct(s) => {
                    Self::add(&mut self.types, &s.name, &s.type_params);
                    self.add_methods(&s.name, &s.methods);
                }
                Item::Enum(e) => {
                    Self::add(&mut self.types, &e.name, &e.type_params);
                    self.add_methods(&e.name, &e.methods);
                }
                Item::Trait(t) => Self::add(&mut self.types, &t.name, &t.type_params),
                Item::Alias(a) => Self::add(&mut self.types, &a.name, &a.type_params),
                Item::Fn(f) => Self::add(&mut self.fns, &f.name, &f.type_params),
                Item::Ext(e) => self.add_methods(&e.type_name, &e.methods),
                Item::Mod(m) => self.collect(&m.items),
                _ => {}
            }
        }
    }

    /// Declarations of `program` and of every file it reaches through `use`.
    pub(crate) fn of_program(program: &Program) -> Self {
        let mut decls = Self::default();
        decls.collect(&program.items);
        if let Some((dir, deps)) = crate::infer_actor_params::import_context() {
            let mut visited = HashSet::new();
            crate::infer_actor_params::visit_imported_files(&program.items, &dir, &deps, &mut visited, &mut |file| {
                decls.collect(&file.items);
            });
        }
        decls
    }
}

/// What a generic argument list is attached to, to find the declaration its labels name.
pub(crate) enum ArgTarget<'a> {
    /// struct / enum / alias / trait in a type position, a constructor call, an `ext` header.
    Type(&'a str),
    /// A free function turbofish `f<T = int>(..)`.
    Fn(&'a str),
    /// A method turbofish `obj.m<T = int>(..)`; the receiver's type when it is a bare type name.
    Method(Option<&'a str>, &'a str),
}

/// One parsed element of a generic argument list.
pub(crate) struct GArg {
    pub label: Option<String>,
    pub ty: Type,
    pub line: usize,
    pub col: usize,
    pub len: usize,
}

fn err(line: usize, col: usize, len: usize, msg: String) -> ParseError {
    ParseError::Generic { msg, line, col, len }
}

/// Parameter name of a declared generic parameter: `T`, `$N:usize` -> `N`; a lifetime has none.
fn param_label(p: &str) -> Option<&str> {
    if p.starts_with('\'') { return None; }
    match p.strip_prefix('$') {
        Some(rest) => rest.split(':').next(),
        None => Some(p),
    }
}

impl Parser {
    /// `Ident =` (not `==`) with a type-parameter-shaped label (declared parameter names start
    /// uppercase): the start of a labelled argument at the current token.
    pub(crate) fn at_generic_label(&self) -> bool {
        matches!(self.peek(), TokenKind::Ident(s) if s.starts_with(|c: char| c.is_uppercase()))
            && self.check2(&TokenKind::Eq)
    }

    /// Parses `arg, arg, Label = arg, ...>` after the opening `<` (already consumed) up to and
    /// including the closing `>`; `elem` parses one value. Returns the arguments in declaration
    /// order. A list without labels is returned as written, with no validation (unchanged).
    pub(crate) fn parse_generic_arg_list(
        &mut self,
        target: ArgTarget,
        elem: fn(&mut Parser) -> Result<Type, ParseError>,
    ) -> Result<Vec<Type>, ParseError> {
        let lt = &self.tokens[self.pos.saturating_sub(1)];
        let (lt_line, lt_col) = (lt.line, lt.col);
        let mut items = Vec::new();
        loop {
            items.push(self.parse_generic_arg(elem)?);
            if !self.eat(&TokenKind::Comma) { break; }
            if self.check(&TokenKind::Gt) { break; }
        }
        self.expect(&TokenKind::Gt)?;
        self.resolve_generic_args(target, items, lt_line, lt_col)
    }

    pub(crate) fn parse_generic_arg(
        &mut self,
        elem: fn(&mut Parser) -> Result<Type, ParseError>,
    ) -> Result<GArg, ParseError> {
        let (line, col, len) = (self.line(), self.col(), self.tok_len());
        let label = if self.at_generic_label() {
            let TokenKind::Ident(name) = self.peek().clone() else { unreachable!() };
            self.advance(); // label
            self.advance(); // `=`
            Some(name)
        } else {
            None
        };
        let ty = elem(self)?;
        Ok(GArg { label, ty, line, col, len })
    }

    /// Normalises a parsed list to positional arguments; see the module doc.
    pub(crate) fn resolve_generic_args(
        &mut self,
        target: ArgTarget,
        items: Vec<GArg>,
        lt_line: usize,
        lt_col: usize,
    ) -> Result<Vec<Type>, ParseError> {
        if items.iter().all(|a| a.label.is_none()) {
            return Ok(items.into_iter().map(|a| a.ty).collect());
        }
        self.saw_labelled_generics = true;
        let Some(decls) = self.generic_decls.clone() else {
            // Lenient first parse: only the declarations matter, labels are dropped.
            return Ok(items.into_iter().map(|a| a.ty).collect());
        };
        let (what, lists) = match &target {
            ArgTarget::Type(n) => (format!("`{n}`"), decls.types.get(*n)),
            ArgTarget::Fn(n) => (format!("function `{n}`"), decls.fns.get(*n)),
            ArgTarget::Method(recv, n) => {
                let qualified = recv.and_then(|r| decls.methods.get(&format!("{r}.{n}")));
                (format!("method `{n}`"), qualified.or_else(|| decls.methods.get(*n)))
            }
        };
        let (first_line, first_col, first_len) = {
            let a = items.iter().find(|a| a.label.is_some()).expect("a label is present");
            (a.line, a.col, a.len)
        };
        let Some(lists) = lists else {
            return Err(err(first_line, first_col, first_len, format!(
                "labelled generic arguments need the declaration of {what} in scope (in this file or a file it `use`s) to name its parameters")));
        };
        if lists.len() > 1 {
            return Err(err(first_line, first_col, first_len, format!(
                "{what} is declared with different generic parameters; labelled generic arguments are ambiguous")));
        }
        let params = &lists[0];
        let mut slots: Vec<Option<Type>> = vec![None; params.len()];
        let mut seen_label = false;
        let mut positional = 0usize;
        for a in items {
            match a.label {
                None => {
                    if seen_label {
                        return Err(err(a.line, a.col, a.len, "positional generic argument after a labelled one".into()));
                    }
                    if positional >= params.len() {
                        return Err(err(a.line, a.col, a.len, format!(
                            "{what} takes {} generic argument(s), got more", params.len())));
                    }
                    slots[positional] = Some(a.ty);
                    positional += 1;
                }
                Some(label) => {
                    seen_label = true;
                    let Some(i) = params.iter().position(|p| param_label(p) == Some(label.as_str())) else {
                        let known: Vec<&str> = params.iter().filter_map(|p| param_label(p)).collect();
                        return Err(err(a.line, a.col, a.len, format!(
                            "unknown generic parameter `{label}` of {what} (parameters: {})", known.join(", "))));
                    };
                    if slots[i].is_some() {
                        return Err(err(a.line, a.col, a.len, format!(
                            "generic parameter `{label}` of {what} is given more than once")));
                    }
                    slots[i] = Some(a.ty);
                }
            }
        }
        let mut out = Vec::with_capacity(slots.len());
        for (i, slot) in slots.into_iter().enumerate() {
            match slot {
                Some(t) => out.push(t),
                None => {
                    let name = param_label(&params[i]).map(str::to_string).unwrap_or_else(|| params[i].clone());
                    return Err(err(lt_line, lt_col, 1, format!(
                        "missing generic argument `{name}` of {what}")));
                }
            }
        }
        Ok(out)
    }
}

pub(crate) fn decls_rc(program: &Program) -> Rc<GenericDecls> {
    Rc::new(GenericDecls::of_program(program))
}
