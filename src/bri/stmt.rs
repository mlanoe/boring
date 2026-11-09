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

//! `.bri` printer: statements and patterns.

use super::*;
use super::expr::{float_text, quote_str};

impl Printer {
    pub(super) fn stmt(&mut self, s: &Stmt) -> R<()> {
        match s {
            Stmt::Let(l) => self.let_stmt(l),
            Stmt::LetDestructure(d) => self.let_destructure(d),
            Stmt::Return(r) => {
                match &r.value {
                    None => self.line("return"),
                    Some(e) => {
                        let t = self.expr(e, P_ELSE)?;
                        self.line(&format!("return {t}"));
                    }
                }
                Ok(())
            }
            Stmt::Break(_, v) => {
                match v {
                    None => self.line("break"),
                    Some(e) => {
                        let t = self.expr(e, P_OR)?;
                        self.line(&format!("break {t}"));
                    }
                }
                Ok(())
            }
            Stmt::Continue(_) => { self.line("continue"); Ok(()) }
            Stmt::Throw(t) => {
                match &t.value {
                    None => self.line("throw"),
                    Some(e) => {
                        let x = self.expr(e, P_OR)?;
                        self.line(&format!("throw {x}"));
                    }
                }
                Ok(())
            }
            Stmt::If(i) => self.if_stmt(i),
            Stmt::IfLet(i) => self.if_let_stmt(i),
            Stmt::Match(m) => {
                let text = self.match_text(m)?;
                self.line(&text);
                Ok(())
            }
            Stmt::While(w) => {
                let c = self.expr(&w.condition, P_ELSE)?;
                self.line(&format!("while {c}:"));
                self.block_inner(&w.body)
            }
            Stmt::WhileLet(w) => {
                let v = self.expr(&w.value, P_ELSE)?;
                match &w.pattern {
                    Some(p) => {
                        let pt = self.pattern(p)?;
                        self.line(&format!("while let {pt} = {v}:"));
                    }
                    None => self.line(&format!("while let {} = {v}:", w.name)),
                }
                self.block_inner(&w.body)
            }
            Stmt::DoWhile(d) => {
                self.line("do:");
                self.block_inner(&d.body)?;
                let c = self.expr(&d.condition, P_ELSE)?;
                self.line(&format!("while {c}"));
                Ok(())
            }
            Stmt::Loop(l) => {
                self.line("loop:");
                self.block_inner(&l.body)
            }
            Stmt::Wait(e, _) => {
                let t = self.expr(e, P_ELSE)?;
                self.line(&format!("wait {t}"));
                Ok(())
            }
            Stmt::For(f) => {
                let it = self.expr(&f.iterable, P_ELSE)?;
                let mut head = format!("for {} in {it}", f.vars.join(", "));
                if let Some(k) = &f.key {
                    let kt = self.expr(k, P_ELSE)?;
                    let _ = write!(head, " with {kt}");
                }
                self.line(&format!("{head}:"));
                self.block_inner(&f.body)
            }
            Stmt::Guard(g) => {
                let cond = match &g.cond {
                    GuardCond::Expr(e) => self.expr(e, P_ELSE)?,
                    GuardCond::Clauses(cs) => self.clauses(cs)?,
                };
                self.line(&format!("guard {cond} else:"));
                self.block_inner(&g.else_body)
            }
            Stmt::Try(t) => {
                self.line("try:");
                self.block_inner(&t.body)?;
                for c in &t.catch_clauses {
                    let mut head = String::from("catch");
                    if !c.types.is_empty() {
                        head.push(' ');
                        head.push_str(&c.types.join(", "));
                        if let Some(v) = &c.variant { let _ = write!(head, ".{v}"); }
                    }
                    self.line(&format!("{head}:"));
                    self.block_inner(&c.body)?;
                }
                Ok(())
            }
            Stmt::Defer(body) => {
                self.line("defer:");
                self.block_inner(body)
            }
            Stmt::Expr(e) => {
                let t = self.expr_stmt(e)?;
                self.line(&t);
                Ok(())
            }
            Stmt::Fn(f) => self.fn_decl(f, &[]),
            Stmt::Struct(s) => self.struct_decl(s),
            Stmt::Enum(e) => self.enum_decl(e),
            Stmt::Mod(m) => self.item(&Item::Mod(m.clone())),
            Stmt::Alias(a) => self.alias_decl(a),
            Stmt::Yield(e, _) => {
                let t = self.expr(e, P_ELSE)?;
                self.line(&format!("yield {t}"));
                Ok(())
            }
            Stmt::Comment(c) => {
                // `pass` and `sync` are parsed to comment markers (no-ops for the runtime, `sync` is
                // also read by the kernel backends); write them back as the keywords.
                if c == "pass" || c == "sync" {
                    self.line(c);
                } else if c.trim().is_empty() {
                    self.line("#");
                } else {
                    self.line(&format!("# {}", c.trim_start()));
                }
                Ok(())
            }
            Stmt::KernelBlock(k) => {
                self.line("kernel:");
                self.block_inner(&k.body)
            }
            Stmt::With(w) => {
                self.line(&format!("with {}:", w.names.join(", ")));
                self.block_inner(&w.body)
            }
        }
    }

    pub(super) fn let_stmt(&mut self, l: &LetStmt) -> R<()> {
        let mut s = String::new();
        if l.is_pub { s.push_str("pub "); }
        if l.is_static { s.push_str("static "); }
        let wrap_mut;
        if l.is_lazy || matches!(l.binding, BindingKind::Lazy) {
            s.push_str("lazy ");
            wrap_mut = false;
        } else {
            match l.binding {
                BindingKind::Let => { s.push_str("let "); wrap_mut = false; }
                BindingKind::Mut => { s.push_str("mut "); wrap_mut = true; }
                BindingKind::Var => {
                    s.push_str("var ");
                    if l.var_mut { s.push_str("mut "); }
                    wrap_mut = l.var_mut;
                }
                BindingKind::Lazy => unreachable!(),
            }
        }
        match &l.ty {
            // `let x'actor = f()`: the qualifier written on the name, base type left to inference.
            Some(t) if placeholder_base(t.without_mut()) => {
                s.push_str(&l.name);
                s.push_str(&self.placeholder_quals(t.without_mut())?);
            }
            Some(t) => {
                let tt = if wrap_mut { self.ty_no_leading_mut(t)? } else { self.ty(t)? };
                s.push_str(&tt);
                s.push(' ');
                s.push_str(&l.name);
            }
            None => s.push_str(&l.name),
        }
        if let Some(v) = &l.value {
            let t = self.expr(v, P_ELSE)?;
            s.push_str(" = ");
            s.push_str(&t);
        }
        self.line(&s);
        Ok(())
    }

    /// The `'q'r` suffix of a type whose base is the `_` placeholder.
    fn placeholder_quals(&self, t: &Type) -> R<String> {
        match t {
            Type::Qualified(inner, q) => {
                let base = self.placeholder_quals(inner)?;
                Ok(format!("{base}'{}", self.qual_word(q)?))
            }
            _ => Ok(String::new()),
        }
    }

    fn let_destructure(&mut self, d: &LetDestructureStmt) -> R<()> {
        let mut s = String::new();
        let group = match d.binding {
            BindingKind::Let => "let",
            BindingKind::Mut => "mut",
            BindingKind::Var => if d.var_mut { "var mut" } else { "var" },
            BindingKind::Lazy => return unsupported("`lazy` destructure", d.line),
        };
        s.push_str(group);
        s.push_str(" (");
        let mut slots = Vec::new();
        for b in &d.bindings {
            if b.name == "_" {
                slots.push("_".to_string());
                continue;
            }
            let slot_kw = match b.binding {
                BindingKind::Let => "let",
                BindingKind::Mut => "mut",
                BindingKind::Var => if b.var_mut { "var mut" } else { "var" },
                BindingKind::Lazy => return unsupported("`lazy` destructure slot", d.line),
            };
            let wrap = matches!(b.binding, BindingKind::Mut) || (matches!(b.binding, BindingKind::Var) && b.var_mut);
            let mut one = String::new();
            if slot_kw != group { one.push_str(slot_kw); one.push(' '); }
            match &b.ty {
                Some(t) => {
                    let tt = if wrap { self.ty_no_leading_mut(t)? } else { self.ty(t)? };
                    let _ = write!(one, "{tt} {}", b.name);
                }
                None => one.push_str(&b.name),
            }
            slots.push(one);
        }
        s.push_str(&slots.join(", "));
        s.push_str(") = ");
        let v = self.expr(&d.value, P_ELSE)?;
        s.push_str(&v);
        self.line(&s);
        Ok(())
    }

    fn if_stmt(&mut self, i: &IfStmt) -> R<()> {
        for (n, (cond, body)) in i.branches.iter().enumerate() {
            let c = self.expr(cond, P_ELSE)?;
            self.line(&format!("{} {c}:", if n == 0 { "if" } else { "elif" }));
            self.block_inner(body)?;
        }
        if let Some(e) = &i.else_body {
            self.line("else:");
            self.block_inner(e)?;
        }
        Ok(())
    }

    pub(super) fn clauses(&mut self, cs: &[CondClause]) -> R<String> {
        let mut out = Vec::new();
        for c in cs {
            out.push(match c {
                CondClause::Let(name, e) => format!("let {name} = {}", self.expr(e, P_OR)?),
                CondClause::LetPat(p, e) => format!("let {} = {}", self.pattern(p)?, self.expr(e, P_OR)?),
                CondClause::Expr(e) => self.expr(e, P_OR)?,
            });
        }
        Ok(out.join(", "))
    }

    fn if_let_stmt(&mut self, i: &IfLetStmt) -> R<()> {
        let c = self.clauses(&i.clauses)?;
        self.line(&format!("if {c}:"));
        self.block_inner(&i.then_body)?;
        for b in &i.elif_branches {
            let c = self.clauses(&b.clauses)?;
            self.line(&format!("elif {c}:"));
            self.block_inner(&b.body)?;
        }
        if let Some(e) = &i.else_body {
            self.line("else:");
            self.block_inner(e)?;
        }
        Ok(())
    }

    /// The `match` statement as text: the head line, then the arms at one more indentation. The
    /// first line carries no padding (the caller pads it), the others do.
    pub(super) fn match_text(&mut self, m: &MatchStmt) -> R<String> {
        let subject = self.expr(&m.subject, P_ELSE)?;
        let mut out = format!("match {subject}:\n");
        for arm in &m.arms {
            let pats: R<Vec<String>> = arm.patterns.iter().map(|p| self.pattern(p)).collect();
            let mut head = pats?.join(" | ");
            if let Some(g) = &arm.guard {
                let gt = self.expr(g, P_ELSE)?;
                let _ = write!(head, " if {gt}");
            }
            let pad = "    ".repeat(self.indent + 1);
            match &arm.body {
                MatchBody::Expr(e) => {
                    let mut sub = self.sub();
                    sub.indent = self.indent + 1;
                    let t = sub.expr(e, P_ELSE)?;
                    let _ = writeln!(out, "{pad}{head}: {t}");
                }
                MatchBody::Block(stmts) => {
                    let _ = writeln!(out, "{pad}{head}:");
                    let mut sub = self.sub();
                    sub.indent = self.indent + 2;
                    if stmts.is_empty() { sub.line("pass"); } else { for s in stmts { sub.stmt(s)?; } }
                    out.push_str(&sub.out);
                }
            }
        }
        if out.ends_with('\n') { out.pop(); }
        Ok(out)
    }

    pub(super) fn pattern(&self, p: &Pattern) -> R<String> {
        Ok(match p {
            Pattern::Wildcard => "_".into(),
            Pattern::Bind(n) => n.clone(),
            Pattern::Variant(name, subs) => {
                let shown = if let Some((base, rest)) = name.split_once('<') {
                    // `Slot<inline>::Full`: the qualifier arguments of a qualifier-generic enum.
                    let (args, variant) = rest.split_once(">::").unwrap_or((rest, ""));
                    let labels: Vec<String> = self.qual_labels.get(base).map(|l| l.iter().flatten().cloned().collect()).unwrap_or_default();
                    let chains: Vec<&str> = args.split(',').collect();
                    if labels.len() != chains.len() {
                        return unsupported(format!("pattern naming a qualifier-generic instance (`{name}`) whose declaration is not in this file"), 0);
                    }
                    let parts: Vec<String> = labels.iter().zip(&chains).map(|(l, c)| format!("{l}'{c}")).collect();
                    format!("{base}<{}>.{variant}", parts.join(", "))
                } else {
                    name.replace("::", ".")
                };
                if subs.is_empty() {
                    shown
                } else {
                    let ps: R<Vec<String>> = subs.iter().map(|s| self.pattern(s)).collect();
                    format!("{}({})", shown, ps?.join(", "))
                }
            }
            Pattern::Lit(l) => match l {
                LitPattern::Int(n) => n.to_string(),
                LitPattern::Float(f) => float_text(*f)?,
                LitPattern::Str(s) => quote_str(s),
                LitPattern::Bool(b) => b.to_string(),
                LitPattern::Nil => "nil".into(),
            },
            Pattern::None => "None".into(),
            Pattern::Some(inner) => match &**inner {
                Pattern::Wildcard => "Some".into(),
                other => format!("Some({})", self.pattern(other)?),
            },
            Pattern::Tuple(ps) => {
                let parts: R<Vec<String>> = ps.iter().map(|s| self.pattern(s)).collect();
                format!("({})", parts?.join(", "))
            }
        })
    }
}

/// A type written `_'q` (the `let x'q = ...` placeholder the parser could not resolve).
fn placeholder_base(t: &Type) -> bool {
    match t {
        Type::Qualified(inner, _) => placeholder_base(inner),
        Type::Named(n) => n == "_",
        _ => false,
    }
}
