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

/// Parameter name of a declared generic parameter: `T`, `$N:usize` -> `N`, `^elem:..` -> `elem`;
/// a lifetime has none.
fn param_label(p: &str) -> Option<&str> {
    if p.starts_with('\'') { return None; }
    if let Some(n) = qual_param_name(p) { return Some(n); }
    match p.strip_prefix('$') {
        Some(rest) => rest.split(':').next(),
        None => Some(p),
    }
}

/// `label'qual` as written in an argument list: a qualified bare name. Whether it is a
/// qualifier-parameter argument or an ordinary qualified type (`Container<T'actor>`) is decided
/// against the callee's declaration.
pub(crate) fn qual_arg_shape(ty: &Type) -> Option<(&str, QualChain)> {
    let mut chain = Vec::new();
    let mut t = ty;
    while let Type::Qualified(inner, q) = t {
        chain.push(q.clone());
        t = inner;
    }
    chain.reverse();
    match t {
        Type::Named(n) | Type::TypeParam(n) if !chain.is_empty() => Some((n.as_str(), chain)),
        _ => None,
    }
}

/// Words that are qualifiers, so cannot label a qualifier parameter (`'actor` would be ambiguous
/// with the qualifier it names).
pub(crate) const QUALIFIER_WORDS: &[&str] = &[
    "inline", "owned", "shared", "actor", "guard", "new", "weak", "atomic", "static", "task",
    "one", "many", "mut", "req", "kernel", "unified", "global", "local", "const", "surface",
    "observed", "copy",
];

impl Parser {
    /// One qualifier chain after its leading tick (already consumed): a base word and glued
    /// suffix ticks (`actor'observed`, `inline`). Returns the word list as a chain; groups are
    /// handled by the caller before this is reached.
    fn parse_qual_chain(&mut self, what: &str) -> Result<QualChain, ParseError> {
        let mut words: Vec<String> = Vec::new();
        loop {
            let (l, c, n) = (self.line(), self.col(), self.tok_len());
            let word = match self.peek().clone() {
                TokenKind::Ident(w) => w,
                TokenKind::Guard => "guard".to_string(),
                TokenKind::Task => "task".to_string(),
                _ => return Err(err(l, c, n, format!("expected a qualifier {what}"))),
            };
            self.advance();
            words.push(word);
            // A glued suffix: `'actor'observed`.
            let suffix = self.check(&TokenKind::Tick) && matches!(
                self.tokens.get(self.pos + 1).map(|t| &t.kind),
                Some(TokenKind::Ident(s)) if s == "observed" || s == "weak" || s == "task"
            ) || self.check(&TokenKind::Tick) && matches!(self.tokens.get(self.pos + 1).map(|t| &t.kind), Some(TokenKind::Task));
            if !suffix { break; }
            self.advance(); // tick
        }
        let (l, c, n) = (self.line(), self.col(), self.tok_len());
        let joined = words.join("'");
        let chain = OwnerQual::chain_from_name(&joined).ok_or_else(|| err(l, c, n, format!(
            "`'{joined}` cannot be a qualifier {what} (use inline, owned, shared, actor, guard, optionally followed by 'observed or 'weak)")))?;
        OwnerQual::check_chain(&chain).map_err(|m| err(l, c, n, format!("`'{joined}`: {m}")))?;
        Ok(chain)
    }

    /// A qualifier parameter declaration at a `'` inside a declaration's `<...>`:
    /// `'elem`, `'elem as 'new`, `'elem as 'inline|owned`, `'elem = 'inline`,
    /// `'elem as 'actor|'actor'observed = 'actor'observed`. Encoded as
    /// `^elem[:members][=default]` (`ast::qual_param_decl`); members and the default are chains.
    /// Declaration errors are reported at the label.
    pub(crate) fn parse_qual_param_decl(&mut self) -> Result<String, ParseError> {
        self.expect(&TokenKind::Tick)?;
        let (line, col, len) = (self.line(), self.col(), self.tok_len());
        let TokenKind::Ident(name) = self.peek().clone() else {
            return Err(err(line, col, len, "expected a qualifier parameter name after `'` (e.g. `'elem`)".into()));
        };
        self.advance();
        if QUALIFIER_WORDS.contains(&name.as_str()) {
            return Err(err(line, col, len, format!(
                "`'{name}` is a qualifier, it cannot name a qualifier parameter")));
        }
        let mut bound: Option<Vec<QualChain>> = None;
        if self.eat(&TokenKind::As) {
            let mut members: Vec<QualChain> = Vec::new();
            self.expect(&TokenKind::Tick)?;
            loop {
                let (l, c, n) = (self.line(), self.col(), self.tok_len());
                let group: Option<Vec<OwnerQual>> = match self.peek() {
                    TokenKind::Ident(w) if w == "new" => Some(OwnerQual::NEW_MEMBERS.to_vec()),
                    TokenKind::Ident(w) if w == "one" => Some(vec![OwnerQual::Inline, OwnerQual::Owned]),
                    TokenKind::Ident(w) if w == "many" => Some(vec![OwnerQual::Shared, OwnerQual::Actor, OwnerQual::Guard]),
                    _ => None,
                };
                let set: Vec<QualChain> = match group {
                    Some(g) => { self.advance(); g.into_iter().map(|q| vec![q]).collect() }
                    None => {
                        let word = match self.peek().clone() {
                            TokenKind::Ident(w) => w,
                            TokenKind::Guard => "guard".to_string(),
                            TokenKind::Task => "task".to_string(),
                            _ => return Err(err(l, c, n, "expected a qualifier in the bound of a qualifier parameter".into())),
                        };
                        if OwnerQual::chain_from_name(&word).is_none() || matches!(word.as_str(), "task" | "weak" | "observed") {
                            self.advance();
                            return Err(err(l, c, n, format!(
                                "`'{word}` cannot bound a qualifier parameter (use inline, owned, shared, actor, guard, new, one, many)")));
                        }
                        vec![self.parse_qual_chain("in the bound of a qualifier parameter")?]
                    }
                };
                for q in set { if !members.contains(&q) { members.push(q); } }
                if !self.eat(&TokenKind::Pipe) { break; }
                self.eat(&TokenKind::Tick);
            }
            bound = Some(members);
        }
        let mut out = format!("^{name}");
        if let Some(members) = &bound {
            let names: Vec<String> = members.iter().filter_map(|m| OwnerQual::chain_name(m)).collect();
            out.push_str(&format!(":{}", names.join(",")));
        }
        if self.eat(&TokenKind::Eq) {
            let (l, c, n) = (self.line(), self.col(), self.tok_len());
            self.expect(&TokenKind::Tick)?;
            let chain = self.parse_qual_chain("as the default of a qualifier parameter")?;
            let members: Vec<QualChain> = match &bound {
                Some(m) => m.clone(),
                None => OwnerQual::BARE_MEMBERS.iter().map(|q| vec![q.clone()]).collect(),
            };
            if !members.contains(&chain) {
                let allowed: Vec<String> = members.iter().filter_map(|m| OwnerQual::chain_name(m)).collect();
                return Err(err(l, c, n, format!(
                    "default `'{}` of qualifier parameter `{name}` is outside its bound (allowed: {})",
                    OwnerQual::chain_name(&chain).unwrap_or_default(), allowed.join(", "))));
            }
            out.push_str(&format!("={}", OwnerQual::chain_name(&chain).unwrap_or_default()));
        }
        Ok(out)
    }

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
        let maybe_qual = items.iter().any(|a| a.label.is_none() && matches!(a.ty, Type::Qualified(..)));
        if !maybe_qual && items.iter().all(|a| a.label.is_none()) {
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
        // `label'qual` is a qualifier-parameter argument only when the callee declares a
        // qualifier parameter of that name; otherwise it is the ordinary qualified type it was.
        let mut items = items;
        if let Some(lists) = lists {
            if lists.len() == 1 {
                for a in items.iter_mut().filter(|a| a.label.is_none()) {
                    let Some((n, q)) = qual_arg_shape(&a.ty) else { continue };
                    if lists[0].iter().any(|p| qual_param_name(p) == Some(n)) {
                        let n = n.to_string();
                        a.label = Some(n);
                        a.ty = Type::QualArg(q);
                    }
                }
            }
        }
        // A callee with qualifier parameters reads every argument (a stray positional one is an
        // error, not an ordinary type argument).
        let callee_has_qual = lists.is_some_and(|l| l.len() == 1 && l[0].iter().any(|p| qual_param_name(p).is_some()));
        let Some(first) = items.iter().find(|a| a.label.is_some()).or_else(|| items.first().filter(|_| callee_has_qual)) else {
            return Ok(items.into_iter().map(|a| a.ty).collect());
        };
        let (first_line, first_col, first_len) = (first.line, first.col, first.len);
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
                    if let Some(q) = qual_param_name(&params[positional]) {
                        if let Some((n, _)) = qual_arg_shape(&a.ty) {
                            let known: Vec<&str> = params.iter().filter_map(|p| qual_param_name(p)).collect();
                            return Err(err(a.line, a.col, a.len, format!(
                                "unknown qualifier parameter `{n}` of {what} (qualifier parameters: {})", known.join(", "))));
                        }
                        return Err(err(a.line, a.col, a.len, format!(
                            "qualifier parameter `{q}` of {what} has no positional form, write `{q}'<qualifier>`")));
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
                    match (qual_param_decl(&params[i]), &a.ty) {
                        // `Board<elem'cells>`: the enclosing declaration's `'cells` is passed on.
                        (Some((_, bound)), Type::QualArg(q)) if matches!(q.as_slice(), [OwnerQual::Param(_)]) => {
                            let [OwnerQual::Param(p)] = q.as_slice() else { unreachable!() };
                            let mine = self.qual_bounds.get(p).cloned().unwrap_or_default();
                            if let Some(out) = mine.iter().find(|m| !bound.contains(m)) {
                                let shown = |c: &QualChain| OwnerQual::chain_name(c).unwrap_or_else(|| "?".into());
                                let allowed: Vec<String> = bound.iter().map(shown).collect();
                                let given: Vec<String> = mine.iter().map(shown).collect();
                                return Err(err(a.line, a.col, a.len, format!(
                                    "the bound of `{p}` (allowed: {}) is not within the bound of `{label}` in {what}: `'{}` is outside it (allowed: {}); narrow it with `'{p} as ...`",
                                    given.join(", "), shown(out), allowed.join(", "))));
                            }
                        }
                        (Some((_, bound)), Type::QualArg(q)) => {
                            let Some(shown) = OwnerQual::chain_name(q) else {
                                return Err(err(a.line, a.col, a.len, format!(
                                    "qualifier argument `{label}` of {what} must be one of 'inline, 'owned, 'shared, 'actor, 'guard, optionally followed by 'observed or 'weak")));
                            };
                            if let Err(m) = OwnerQual::check_chain(q) {
                                return Err(err(a.line, a.col, a.len, format!(
                                    "qualifier argument `{label}'{shown}` of {what}: {m}")));
                            }
                            if !bound.contains(q) {
                                let allowed: Vec<String> = bound.iter().filter_map(|m| OwnerQual::chain_name(m)).collect();
                                return Err(err(a.line, a.col, a.len, format!(
                                    "qualifier `'{shown}` is outside the bound of `{label}` in {what} (allowed: {})",
                                    allowed.join(", "))));
                            }
                        }
                        (Some(_), _) => return Err(err(a.line, a.col, a.len, format!(
                            "`{label}` is a qualifier parameter of {what}: write `{label}'<qualifier>`"))),
                        (None, Type::QualArg(_)) => return Err(err(a.line, a.col, a.len, format!(
                            "`{label}` is not a qualifier parameter of {what}"))),
                        _ => {}
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

/// A bare use of a declaration with qualifier parameters (`Board`, `Board()`, `Slot.Full(x)`)
/// is an error: every use names its qualifier arguments (docs/design-notes/qualifier-generic-params.md).
/// Declarations come from the file and the files it `use`s.
pub(crate) fn check_bare_qual_uses(program: &Program) -> Result<(), ParseError> {
    if !program.items.iter().any(|i| matches!(i, Item::Use(_)) || crate::transpiler::monomorphize::is_qual_generic_item(i)) {
        return Ok(());
    }
    let decls = GenericDecls::of_program(program);
    let mut qual: HashMap<String, Vec<String>> = HashMap::new();
    for (name, lists) in &decls.types {
        if let [list] = lists.as_slice() {
            let names: Vec<String> = list.iter().filter_map(|p| qual_param_name(p)).map(str::to_string).collect();
            if !names.is_empty() { qual.insert(name.clone(), names); }
        }
    }
    let has_qual_fn = decls.fns.values().any(|lists| lists.iter().any(|l| l.iter().any(|p| qual_param_name(p).is_some())));
    if qual.is_empty() && !has_qual_fn { return Ok(()); }
    let names = qual.keys().cloned().collect();
    let (hits, diags) = crate::transpiler::monomorphize::find_bare_qual_uses(program, names);
    // A construction whose evidence was wrong (conflict, bound) says so, even when another rule
    // (the expected type on a `let`) still names an instance for it; any other bare use is one
    // nothing determines. The first in source order is reported.
    let first_hit = hits.into_iter().next();
    let first_diag = diags.iter().min_by_key(|(pos, _)| **pos);
    match (first_hit, first_diag) {
        (hit, Some((&(line, col), msg))) if hit.as_ref().is_none_or(|(_, l, c)| (line, col) <= (*l, *c)) => {
            Err(err(line, col, diag_name_len(msg).unwrap_or(1), msg.clone()))
        }
        (Some((name, line, col)), _) => {
            let list = &decls.types[&name][0];
            let missing = &qual[&name];
            let plural = if missing.len() == 1 { "" } else { "s" };
            let quoted = missing.iter().map(|q| format!("`{q}`")).collect::<Vec<_>>().join(", ");
            let write = list.iter().map(|p| match qual_param_name(p) {
                Some(q) => format!("{q}'..."),
                None => format!("{} = ...", param_label(p).unwrap_or(p)),
            }).collect::<Vec<_>>().join(", ");
            Err(err(line, col, name.len(), format!(
                "cannot infer qualifier parameter{plural} {quoted} of `{name}`, write `{name}<{write}>`")))
        }
        _ => Ok(()),
    }
}

/// The declaration name a construction diagnostic is about: the backquoted word after `in ` or `of `.
fn diag_name_len(msg: &str) -> Option<usize> {
    let rest = msg.split(" in `").nth(1).or_else(|| msg.split(" of `").nth(1))?;
    rest.split('`').next().map(str::len)
}

pub(crate) fn decls_rc(program: &Program) -> Rc<GenericDecls> {
    Rc::new(GenericDecls::of_program(program))
}
