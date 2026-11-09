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

//! `.bri` ("Boring inferred") printer: a complete AST -> Boring source printer.
//!
//! The output is valid, re-parsable Boring written for a developer to read: what the compiler
//! decided (inferred return types, implicit generics written as explicit type parameters with
//! their bounds, specialised copies) is spelled out. See `docs/design-notes/typed-monomorphization-draft.md`
//! sections 3.6 and 11.
//!
//! The printer never guesses: a construct it does not know is an explicit error naming it
//! (`BriError`), never a wrong output. It is *not* `desugar_array_block::describe_type` (which
//! elides qualifiers); every qualifier of the AST is written.

use crate::ast::*;
use std::collections::HashMap;
use std::fmt::Write as _;

/// A construct the printer cannot write faithfully.
#[derive(Debug, Clone)]
pub struct BriError {
    pub what: String,
    pub line: usize,
}

impl std::fmt::Display for BriError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line > 0 {
            write!(f, ".bri printer: {} (line {})", self.what, self.line)
        } else {
            write!(f, ".bri printer: {}", self.what)
        }
    }
}

type R<T> = Result<T, BriError>;

fn unsupported<T>(what: impl Into<String>, line: usize) -> R<T> {
    Err(BriError { what: what.into(), line })
}

/// How much the `.bri` shows (draft section 3.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BriMode {
    /// The AST right after specialisation, nothing else added.
    Mono,
    /// Additionally every inferred qualifier written out.
    Full,
}

/// Prints `program` (as the parser leaves it, so after `specialize_untyped`) at the given level.
/// `mono` prints the AST as is; `full` first writes `decisions`, the qualifiers the transpiler's
/// inference chose (`full::apply_qualifiers`).
pub fn print_program_mode(program: &Program, mode: BriMode, decisions: &[full::QualDecision]) -> Result<String, BriError> {
    match mode {
        BriMode::Mono => print_program(program),
        BriMode::Full => {
            let mut p = program.clone();
            full::apply_qualifiers(&mut p, decisions);
            print_program(&p)
        }
    }
}

/// Prints a whole program. `Err` names the first construct the printer does not know.
pub fn print_program(program: &Program) -> Result<String, BriError> {
    let mut p = Printer::new(program);
    p.items(&program.items, true)?;
    Ok(p.out)
}

// ─── Precedence (matches `parser/parse_expr.rs`, low -> high) ────────────────────────────────

const P_ELSE: u8 = 0; // `a else b`, `try ...`, closures, if/match/blocks (extend to the right)
const P_PIPE: u8 = 1;
const P_OR: u8 = 2;
const P_AND: u8 = 3;
const P_NOT: u8 = 4;
const P_CMP: u8 = 5;
const P_BITOR: u8 = 6;
const P_BITXOR: u8 = 7;
const P_BITAND: u8 = 8;
const P_SHIFT: u8 = 9;
const P_RANGE: u8 = 10;
const P_ADD: u8 = 11;
const P_MUL: u8 = 12;
const P_UNARY: u8 = 13;
const P_POSTFIX: u8 = 14;
const P_ATOM: u8 = 15;

fn binop_info(op: &BinOp) -> (&'static str, u8) {
    match op {
        BinOp::Add => ("+", P_ADD),
        BinOp::Sub => ("-", P_ADD),
        BinOp::Mul => ("*", P_MUL),
        BinOp::Div => ("/", P_MUL),
        BinOp::Rem => ("%", P_MUL),
        BinOp::Eq => ("==", P_CMP),
        BinOp::NotEq => ("!=", P_CMP),
        BinOp::RefEq => ("===", P_CMP),
        BinOp::Lt => ("<", P_CMP),
        BinOp::Gt => (">", P_CMP),
        BinOp::LtEq => ("<=", P_CMP),
        BinOp::GtEq => (">=", P_CMP),
        BinOp::And => ("and", P_AND),
        BinOp::Or => ("or", P_OR),
        BinOp::BitAnd => ("&", P_BITAND),
        BinOp::BitOr => ("|", P_BITOR),
        BinOp::BitXor => ("^", P_BITXOR),
        BinOp::Shl => ("<<", P_SHIFT),
        BinOp::Shr => (">>", P_SHIFT),
        BinOp::Is => ("is", P_CMP),
        BinOp::IsNot => ("is not", P_CMP),
    }
}

struct Printer {
    out: String,
    indent: usize,
    /// Declaration labels of qualifier parameters, by declaration name: `Board` -> `['elem']`
    /// (positional), to write a `QualArg` as `elem'actor`.
    qual_labels: HashMap<String, Vec<Option<String>>>,
}

impl Printer {
    fn new(program: &Program) -> Printer {
        let mut qual_labels = HashMap::new();
        collect_qual_labels(&program.items, &mut qual_labels);
        Printer { out: String::new(), indent: 0, qual_labels }
    }

    fn pad(&self) -> String {
        "    ".repeat(self.indent)
    }

    fn line(&mut self, text: &str) {
        let pad = self.pad();
        self.out.push_str(&pad);
        self.out.push_str(text);
        self.out.push('\n');
    }

    // ─── Items ───────────────────────────────────────────────────────────────

    fn items(&mut self, items: &[Item], top: bool) -> R<()> {
        let mut prev_decl = false;
        for (i, item) in items.iter().enumerate() {
            let is_decl = matches!(item, Item::Fn(_) | Item::Struct(_) | Item::Enum(_) | Item::Trait(_)
                | Item::Ext(_) | Item::Mod(_) | Item::Kernel(_));
            if i > 0 && (is_decl || prev_decl) && top {
                self.out.push('\n');
            }
            prev_decl = is_decl;
            self.item(item)?;
        }
        Ok(())
    }

    fn item(&mut self, item: &Item) -> R<()> {
        match item {
            Item::Use(u) => self.use_decl(u),
            Item::Fn(f) => self.fn_decl(f, &[]),
            Item::Struct(s) => self.struct_decl(s),
            Item::Enum(e) => self.enum_decl(e),
            Item::Trait(t) => self.trait_decl(t),
            Item::Ext(e) => self.ext_decl(e),
            Item::Mod(m) => {
                self.line(&format!("mod {}:", m.name));
                self.indent += 1;
                if m.items.is_empty() { self.line("pass"); } else { self.items(&m.items, false)?; }
                self.indent -= 1;
                Ok(())
            }
            Item::Let(l) => self.let_stmt(l),
            Item::Alias(a) => self.alias_decl(a),
            Item::Kernel(k) => self.kernel_decl(k),
            Item::Stmt(s) => self.stmt(s),
        }
    }

    fn use_decl(&mut self, u: &UseDecl) -> R<()> {
        let path = u.path.join(".");
        if u.glob {
            self.line(&format!("use {path}.*"));
        } else if u.items.is_empty() {
            self.line(&format!("use {path}"));
        } else {
            self.line(&format!("use {path}.{}", u.items.join(", ")));
        }
        Ok(())
    }

    fn alias_decl(&mut self, a: &AliasDecl) -> R<()> {
        let ty = self.ty(&a.ty)?;
        if a.newtype {
            self.line(&format!("type {} as {}", a.name, ty));
        } else if a.type_params.is_empty() {
            self.line(&format!("use {} as {}", a.name, ty));
        } else {
            let tp = self.type_params(&a.type_params, &[], &[], &[])?;
            self.line(&format!("use {}{} as {}", a.name, tp, ty));
        }
        Ok(())
    }

    fn attrs(&mut self, attrs: &[Attr]) {
        for a in attrs {
            if a.args.is_empty() {
                self.line(&format!("@{}", a.name));
            } else {
                self.line(&format!("@{}({})", a.name, a.args.join(", ")));
            }
        }
    }

    /// `<T as A + B, 'q, uint N>` from the encoded `type_params` and `where_clause`.
    /// `skip` names enclosing-declaration parameters not to redeclare; `implicit_quals` are
    /// qualifier parameters introduced by their first use in a signature.
    fn type_params(
        &self,
        params: &[String],
        where_clause: &[(String, String)],
        skip: &[String],
        implicit_quals: &[(String, usize)],
    ) -> R<String> {
        let mut parts: Vec<String> = Vec::new();
        for p in params {
            if skip.contains(p) {
                continue;
            }
            if let Some(rest) = p.strip_prefix('^') {
                let name = qual_param_name(p).unwrap_or(rest);
                if implicit_quals.iter().any(|(n, _)| n == name) {
                    continue;
                }
                parts.push(self.qual_param_text(p)?);
            } else if let Some(rest) = p.strip_prefix('$') {
                let Some((name, rust_ty)) = rest.split_once(':') else {
                    return unsupported(format!("const generic parameter `{p}`"), 0);
                };
                let kw = match rust_ty {
                    "usize" => "uint",
                    "isize" | "i64" => "int",
                    "bool" => "bool",
                    other => return unsupported(format!("const generic of type `{other}`"), 0),
                };
                parts.push(format!("{kw} {name}"));
            } else if let Some(lt) = p.strip_prefix('\'') {
                parts.push(format!("&{lt}"));
            } else {
                let bounds: Vec<&str> = where_clause.iter().filter(|(n, _)| n == p).map(|(_, b)| b.as_str()).collect();
                if bounds.is_empty() {
                    parts.push(p.clone());
                } else {
                    parts.push(format!("{} as {}", p, bounds.join(" + ")));
                }
            }
        }
        if parts.is_empty() { Ok(String::new()) } else { Ok(format!("<{}>", parts.join(", "))) }
    }

    fn qual_param_text(&self, p: &str) -> R<String> {
        let Some((name, bound)) = qual_param_decl(p) else {
            return unsupported(format!("qualifier parameter `{p}`"), 0);
        };
        let mut s = format!("'{name}");
        let bare: Vec<QualChain> = OwnerQual::BARE_MEMBERS.iter().map(|q| vec![q.clone()]).collect();
        if bound != bare {
            let names: Option<Vec<String>> = bound.iter().map(|c| OwnerQual::chain_name(c)).collect();
            let Some(names) = names else { return unsupported(format!("qualifier bound of `{p}`"), 0) };
            s.push_str(&format!(" as '{}", names.join("|")));
        }
        if let Some(d) = qual_param_default(p) {
            let Some(n) = OwnerQual::chain_name(&d) else { return unsupported(format!("qualifier default of `{p}`"), 0) };
            s.push_str(&format!(" = '{n}"));
        }
        Ok(s)
    }

    fn fn_decl(&mut self, f: &FnDecl, enclosing: &[String]) -> R<()> {
        self.attrs(&f.attrs);
        let implicit = implicit_param_types(f);
        let note = if implicit.is_empty() { None } else { Some(self.implicit_comment(&implicit, &f.where_clause)) };
        let head = self.fn_head(f, enclosing)?;
        self.fn_body_after(&head, &f.body, f.is_native, note)
    }

    /// `# inferred: T_n as PartialOrd<isize> + Copy`: the bounds the compiler derived for implicit
    /// type parameters (not Boring syntax, so a comment; the untyped parameter re-derives them).
    fn implicit_comment(&self, fresh: &[String], where_clause: &[(String, String)]) -> String {
        let mut parts = Vec::new();
        for t in fresh {
            let bounds: Vec<&str> = where_clause.iter().filter(|(n, _)| n == t).map(|(_, b)| b.as_str()).collect();
            if bounds.is_empty() { parts.push(t.clone()); } else { parts.push(format!("{} as {}", t, bounds.join(" + "))); }
        }
        format!("# inferred: {}", parts.join(", "))
    }

    fn fn_head(&self, f: &FnDecl, enclosing: &[String]) -> R<String> {
        let implicit = implicit_param_types(f);
        let mut enclosing: Vec<String> = enclosing.to_vec();
        enclosing.extend(implicit.iter().cloned());
        let enclosing = &enclosing[..];
        let wc: Vec<(String, String)> = f.where_clause.iter().filter(|(n, _)| !implicit.contains(n)).cloned().collect();
        let mut s = String::new();
        if f.is_pub { s.push_str("pub "); }
        if f.task { s.push_str("task "); }
        if f.stream {
            s.push_str("stream");
            if let Some(n) = f.stream_capacity { let _ = write!(s, "<{n}>"); }
            s.push(' ');
        }
        s.push_str(if f.mutating { "def " } else { "req " });
        if f.return_mutable { s.push_str("mut "); }
        match &f.return_ty {
            // `def Worker.run()` would read `Worker` as the return type of `run`: say `void`.
            None | Some(Type::Void) if f.qualifier.is_none() => {}
            None => {}
            Some(t) => { s.push_str(&self.ty(t)?); s.push(' '); }
        }
        if let Some(q) = &f.qualifier {
            s.push_str(q);
            s.push('.');
        }
        s.push_str(&f.name);
        s.push_str(&self.type_params(&f.type_params, &wc, enclosing, &f.implicit_quals)?);
        s.push('(');
        let mut ps = Vec::new();
        for p in &f.params {
            let untyped = matches!(&p.ty, Some(Type::TypeParam(t)) if implicit.contains(t));
            let bare = p.ty.as_ref().and_then(|t| bare_struct_param(t, &p.name, &f.type_params))
                .filter(|(_, names)| names.iter().all(|n| implicit.contains(n)));
            if untyped {
                let mut q = p.clone();
                q.ty = None;
                ps.push(self.param(&q, &f.slot_quals)?);
            } else if let Some((bare_ty, _)) = bare {
                let mut q = p.clone();
                q.ty = Some(bare_ty);
                ps.push(self.param(&q, &f.slot_quals)?);
            } else {
                ps.push(self.param(p, &f.slot_quals)?);
            }
        }
        s.push_str(&ps.join(", "));
        s.push(')');
        if f.throws {
            s.push_str(" throws");
            if let Some(t) = &f.throws_ty { s.push(' '); s.push_str(&self.ty(t)?); }
        }
        Ok(s)
    }

    /// `head:` followed by the block, or ` native` / `pass`.
    ///
    /// `note` is a trailing comment on the header line (the lexer drops it: a full-line comment
    /// would be kept as a statement and reach the generated Rust).
    fn fn_body_after(&mut self, head: &str, body: &[Stmt], native: bool, note: Option<String>) -> R<()> {
        let note = note.map(|n| format!("  {n}")).unwrap_or_default();
        if native {
            self.line(&format!("{head}: native{note}"));
            return Ok(());
        }
        if body.is_empty() {
            // `: pass` on the header line is an empty body; a block with `pass` is a `pass` statement.
            self.line(&format!("{head}: pass{note}"));
            return Ok(());
        }
        self.line(&format!("{head}:{note}"));
        self.block_inner(body)
    }

    fn block_inner(&mut self, body: &[Stmt]) -> R<()> {
        self.indent += 1;
        if body.is_empty() {
            self.line("pass");
        } else {
            for s in body { self.stmt(s)?; }
            if body.iter().all(|s| matches!(s, Stmt::Comment(c) if c != "pass" && c != "sync")) { self.line("pass"); }
        }
        self.indent -= 1;
        Ok(())
    }

    fn params(&self, params: &[Param], slot_quals: &[(String, OwnerQual)]) -> R<String> {
        let mut out = Vec::new();
        for p in params {
            out.push(self.param(p, slot_quals)?);
        }
        Ok(out.join(", "))
    }

    fn param(&self, p: &Param, slot_quals: &[(String, OwnerQual)]) -> R<String> {
        let mut s = String::new();
        if p.rebindable {
            s.push_str("var ");
            if p.var_mut { s.push_str("mut "); }
        } else if p.mutable {
            s.push_str("mut ");
        }
        match &p.ty {
            None => {
                s.push_str(&p.name);
                if let Some((_, q)) = slot_quals.iter().find(|(n, _)| *n == p.name) {
                    s.push_str(&self.qual_suffix(q)?);
                }
            }
            Some(Type::Fn(None, ps, throws, task, false)) => {
                // `name(Params)`: function-typed parameter with an inferred return type
                let _ = write!(s, "{}(", p.name);
                let tys: R<Vec<String>> = ps.iter().map(|t| self.ty(t)).collect();
                s.push_str(&tys?.join(", "));
                s.push(')');
                if *task { s.push_str(" task"); }
                if *throws { s.push_str(" throws"); }
            }
            Some(t) => {
                // The keyword already carries a `mut` borrow (`var T&`, `mut T&`): write the plain
                // borrow so `mut mut` never appears.
                let tt = if p.mutable { self.ty_no_leading_mut(t)? } else { self.ty(t)? };
                s.push_str(&tt);
                if p.variadic { s.push_str("..."); }
                s.push(' ');
                s.push_str(&p.name);
            }
        }
        if let Some(d) = &p.default {
            let mut sub = self.sub();
            s.push_str(" = ");
            s.push_str(&sub.expr(d, P_OR)?);
        }
        Ok(s)
    }

    /// A printer for a nested expression on a scratch buffer (same labels, indent 0).
    fn sub(&self) -> Printer {
        Printer { out: String::new(), indent: self.indent, qual_labels: self.qual_labels.clone() }
    }

    fn qual_suffix(&self, q: &OwnerQual) -> R<String> {
        Ok(format!("'{}", self.qual_word(q)?))
    }

    // ─── Struct / enum / trait / ext / kernel ────────────────────────────────

    fn struct_decl(&mut self, s: &StructDecl) -> R<()> {
        self.attrs(&s.attrs);
        let fresh: Vec<String> = s.fields.iter().filter(|f| is_fresh_field_type(&f.ty, &f.name))
            .map(|f| crate::infer_param_generics::fresh_name(&f.name))
            .filter(|t| s.type_params.contains(t)).collect();
        let note = if fresh.is_empty() { String::new() } else { format!("  {}", self.implicit_comment(&fresh, &s.where_clause)) };
        let wc: Vec<(String, String)> = s.where_clause.iter().filter(|(n, _)| !fresh.contains(n)).cloned().collect();
        let mut head = String::new();
        if s.is_pub { head.push_str("pub "); }
        head.push_str(if s.is_view { "view " } else { "struct " });
        head.push_str(&s.name);
        head.push_str(&self.type_params(&s.type_params, &wc, &fresh, &[])?);
        if !s.protocols.is_empty() {
            head.push_str(" as ");
            head.push_str(&s.protocols.join(", "));
        }
        if s.is_native {
            self.line(&format!("{head}: native"));
            return Ok(());
        }
        let empty = s.fields.is_empty() && s.inits.is_empty() && s.methods.is_empty() && s.conversions.is_empty()
            && s.setters.is_empty() && s.type_methods.is_empty() && s.type_vars.is_empty() && s.assoc_type_defs.is_empty();
        if empty {
            self.line(&format!("{head}: pass{note}"));
            return Ok(());
        }
        self.line(&format!("{head}:{note}"));
        self.indent += 1;
        for tv in &s.type_vars { self.type_var(tv)?; }
        for a in &s.assoc_type_defs {
            let t = self.ty(&a.ty)?;
            self.line(&format!("type {} = {}", a.name, t));
        }
        for f in &s.fields { self.field_decl(f, &fresh)?; }
        for i in &s.inits { self.init_decl(i)?; }
        for m in &s.methods { self.fn_decl(m, &s.type_params)?; }
        for st in &s.setters { self.setter(st)?; }
        for tm in &s.type_methods { self.type_method(tm)?; }
        for c in &s.conversions { self.conversion(c)?; }
        self.indent -= 1;
        Ok(())
    }

    fn type_var(&mut self, tv: &TypeVar) -> R<()> {
        let mut s = String::new();
        if tv.is_pub { s.push_str("pub "); }
        s.push_str(if tv.mutable { "type var " } else { "type let " });
        if let Some(t) = &tv.ty { s.push_str(&self.ty(t)?); s.push(' '); }
        s.push_str(&tv.name);
        s.push_str(" = ");
        let mut sub = self.sub();
        s.push_str(&sub.expr(&tv.default, P_ELSE)?);
        self.line(&s);
        Ok(())
    }

    fn field_decl(&mut self, f: &FieldDecl, fresh: &[String]) -> R<()> {
        self.attrs(&f.attrs);
        let untyped = is_fresh_field_type(&f.ty, &f.name)
            && fresh.contains(&crate::infer_param_generics::fresh_name(&f.name));
        let mut s = String::new();
        if let Some(k) = f.view_kind {
            if f.is_pub { s.push_str("pub "); }
            s.push_str(match k { ViewFieldKind::State => "state ", ViewFieldKind::Bind => "bind ", ViewFieldKind::Env => "env " });
            let mut ty_text = None;
            if f.explicit_perm {
                let has_mut = f.ty.grants_mut();
                s.push_str(match (f.mutable, has_mut) {
                    (false, false) => "let ",
                    (false, true) => "mut ",
                    (true, false) => "var ",
                    (true, true) => "var mut ",
                });
                if has_mut { ty_text = Some(self.ty_no_leading_mut(&f.ty)?); }
            }
            let ty_text = match ty_text { Some(t) => t, None => self.ty(&f.ty)? };
            if !untyped { s.push_str(&ty_text); }
        } else if f.transient {
            if f.is_pub { s.push_str("pub "); }
            s.push_str("transient ");
            if !untyped { s.push_str(&self.ty(&f.ty)?); }
        } else {
            let has_mut = f.ty.grants_mut();
            let kw = match (f.mutable, has_mut) {
                (false, false) => if f.is_pub { "" } else { "let " },
                (false, true) => "mut ",
                (true, false) => "var ",
                (true, true) => "var mut ",
            };
            if f.is_pub && !kw.is_empty() { s.push_str("pub "); }
            s.push_str(kw);
            if !untyped {
                if has_mut { s.push_str(&self.ty_no_leading_mut(&f.ty)?); } else { s.push_str(&self.ty(&f.ty)?); }
            }
        }
        if !untyped || (!s.is_empty() && !s.ends_with(' ')) { s.push(' '); }
        s.push_str(&f.name);
        if let Some(d) = &f.default {
            s.push_str(" = ");
            let mut sub = self.sub();
            s.push_str(&sub.expr(d, P_ELSE)?);
        }
        self.line(&s);
        Ok(())
    }

    fn init_decl(&mut self, i: &InitDecl) -> R<()> {
        let mut ps = Vec::new();
        for p in &i.params {
            let mut s = String::new();
            if p.is_pub { s.push_str("pub "); }
            if p.mutable { s.push_str("var "); }
            if let Some(t) = &p.ty { s.push_str(&self.ty(t)?); s.push(' '); }
            s.push_str(&p.name);
            if let Some(d) = &p.default {
                s.push_str(" = ");
                let mut sub = self.sub();
                s.push_str(&sub.expr(d, P_OR)?);
            }
            ps.push(s);
        }
        let head = format!("init({})", ps.join(", "));
        if i.body.is_empty() {
            self.line(&head);
        } else {
            self.line(&format!("{head}:"));
            self.block_inner(&i.body)?;
        }
        Ok(())
    }

    fn setter(&mut self, s: &SetDecl) -> R<()> {
        let mut head = String::new();
        if s.is_pub { head.push_str("pub "); }
        let _ = write!(head, "set {}({} {})", s.name, self.ty(&s.param_ty)?, s.param_name);
        if s.throws { head.push_str(" throws"); }
        if s.task { head.push_str(" task"); }
        self.line(&format!("{head}:"));
        self.block_inner(&s.body)
    }

    fn type_method(&mut self, m: &TypeMethod) -> R<()> {
        let mut head = String::new();
        if m.is_pub { head.push_str("pub "); }
        match m.kind {
            TypeMethodKind::Set => {
                let p = m.params.first().ok_or_else(|| BriError { what: "`type set` without parameter".into(), line: m.line })?;
                let t = match &p.ty { Some(t) => self.ty(t)?, None => return unsupported("`type set` parameter without type", m.line) };
                let _ = write!(head, "type set {}({} {})", m.name, t, p.name);
            }
            _ => {
                head.push_str(if matches!(m.kind, TypeMethodKind::Def) { "type def " } else { "type req " });
                if let Some(t) = &m.return_ty {
                    if !matches!(t, Type::Void) { head.push_str(&self.ty(t)?); head.push(' '); }
                }
                let _ = write!(head, "{}({})", m.name, self.params(&m.params, &[])?);
            }
        }
        if m.throws {
            head.push_str(" throws");
            if let Some(t) = &m.throws_ty { head.push(' '); head.push_str(&self.ty(t)?); }
        }
        if m.task { head.push_str(" task"); }
        self.line(&format!("{head}:"));
        self.block_inner(&m.body)
    }

    fn conversion(&mut self, c: &AsDecl) -> R<()> {
        let mut head = String::new();
        if c.is_pub { head.push_str("pub "); }
        let _ = write!(head, "as {}", self.ty(&c.ty)?);
        if c.throws { head.push_str(" throws"); }
        if c.task { head.push_str(" task"); }
        self.line(&format!("{head}:"));
        self.block_inner(&c.body)
    }

    fn enum_decl(&mut self, e: &EnumDecl) -> R<()> {
        self.attrs(&e.attrs);
        let mut head = String::new();
        if e.is_pub { head.push_str("pub "); }
        head.push_str("enum ");
        head.push_str(&e.name);
        head.push_str(&self.type_params(&e.type_params, &[], &[], &[])?);
        if !e.protocols.is_empty() {
            head.push_str(" as ");
            head.push_str(&e.protocols.join(", "));
        }
        if e.is_native {
            self.line(&format!("{head}: native"));
            return Ok(());
        }
        if e.variants.is_empty() && e.methods.is_empty() && e.setters.is_empty() && e.conversions.is_empty() && e.type_methods.is_empty() {
            self.line(&format!("{head}: pass"));
            return Ok(());
        }
        self.line(&format!("{head}:"));
        self.indent += 1;
        for v in &e.variants {
            self.attrs(&v.attrs);
            if v.fields.is_empty() {
                self.line(&v.name);
            } else {
                let mut fs = Vec::new();
                for f in &v.fields {
                    let t = self.ty(&f.ty)?;
                    fs.push(match &f.name { Some(n) => format!("{t} {n}"), None => t });
                }
                self.line(&format!("{}({})", v.name, fs.join(", ")));
            }
        }
        for m in &e.methods { self.fn_decl(m, &e.type_params)?; }
        for st in &e.setters { self.setter(st)?; }
        for tm in &e.type_methods { self.type_method(tm)?; }
        for c in &e.conversions { self.conversion(c)?; }
        self.indent -= 1;
        Ok(())
    }

    fn trait_decl(&mut self, t: &TraitDecl) -> R<()> {
        let mut head = format!("trait {}", t.name);
        head.push_str(&self.type_params(&t.type_params, &[], &[], &[])?);
        if !t.parents.is_empty() {
            head.push_str(" as ");
            head.push_str(&t.parents.join(", "));
        }
        self.line(&format!("{head}:"));
        self.indent += 1;
        let mut any = false;
        for a in &t.assoc_types {
            let mut s = format!("type {}", a.name);
            if !a.type_params.is_empty() {
                return unsupported("associated type with generic parameters", a.line);
            }
            if let Some(c) = &a.constraint { let _ = write!(s, " as {}", self.ty(c)?); }
            self.line(&s);
            any = true;
        }
        for sig in &t.signatures { self.signature(sig, false)?; any = true; }
        for sig in &t.type_signatures { self.signature(sig, true)?; any = true; }
        for d in &t.defaults { self.fn_decl(d, &t.type_params)?; any = true; }
        if !any { self.line("pass"); }
        self.indent -= 1;
        Ok(())
    }

    fn signature(&mut self, sig: &FnSignature, type_level: bool) -> R<()> {
        let mut s = String::new();
        if sig.task { s.push_str("task "); }
        if sig.stream { s.push_str("stream "); }
        if type_level { s.push_str("type "); }
        s.push_str(if sig.mutating { "def " } else { "req " });
        if sig.return_mutable { s.push_str("mut "); }
        match &sig.return_ty {
            None | Some(Type::Void) => {}
            Some(t) => { s.push_str(&self.ty(t)?); s.push(' '); }
        }
        s.push_str(&sig.name);
        s.push_str(&self.type_params(&sig.type_params, &[], &[], &[])?);
        let _ = write!(s, "({})", self.params(&sig.params, &[])?);
        if sig.throws { s.push_str(" throws"); }
        self.line(&s);
        Ok(())
    }

    fn ext_decl(&mut self, e: &ExtDecl) -> R<()> {
        let mut head = format!("ext {}", e.type_name);
        if !e.type_args.is_empty() {
            let mut args = Vec::new();
            for a in &e.type_args {
                match a {
                    Type::TypeParam(n) if e.type_params.contains(n) => {
                        let bounds: Vec<&str> = e.where_clause.iter().filter(|(p, _)| p == n).map(|(_, b)| b.as_str()).collect();
                        args.push(if bounds.is_empty() { n.clone() } else { format!("{} as {}", n, bounds.join(" + ")) });
                    }
                    Type::QualArg(c) => match c.as_slice() {
                        [OwnerQual::Param(n)] if e.type_params.iter().any(|p| qual_param_name(p) == Some(n.as_str())) => {
                            let decl = e.type_params.iter().find(|p| qual_param_name(p) == Some(n.as_str())).unwrap();
                            args.push(self.qual_param_text(decl)?);
                        }
                        chain => {
                            let idx = e.type_args.iter().position(|x| std::ptr::eq(x, a)).unwrap_or(0);
                            let label = self.qual_labels.get(&e.type_name).and_then(|l| l.get(idx)).and_then(|l| l.clone());
                            let Some(label) = label else {
                                return unsupported(format!("qualifier argument of `{}` whose declaration is not in this file", e.type_name), e.line);
                            };
                            args.push(self.qual_arg_text(&label, chain)?);
                        }
                    },
                    other => args.push(self.ty(other)?),
                }
            }
            let _ = write!(head, "<{}>", args.join(", "));
        }
        if !e.traits.is_empty() {
            head.push_str(" as ");
            head.push_str(&e.traits.join(", "));
        }
        self.line(&format!("{head}:"));
        self.indent += 1;
        let empty = e.methods.is_empty() && e.setters.is_empty() && e.conversions.is_empty() && e.assoc_type_defs.is_empty();
        if empty { self.line("pass"); }
        for a in &e.assoc_type_defs {
            let t = self.ty(&a.ty)?;
            self.line(&format!("type {} = {}", a.name, t));
        }
        for m in &e.methods { self.fn_decl(m, &e.type_params)?; }
        for st in &e.setters { self.setter(st)?; }
        for c in &e.conversions { self.conversion(c)?; }
        self.indent -= 1;
        Ok(())
    }

    fn kernel_decl(&mut self, k: &KernelDecl) -> R<()> {
        if let Some(c) = k.prefer { self.line(&format!("@prefer(.{})", c.name())); }
        if let Some(c) = k.only { self.line(&format!("@only(.{})", c.name())); }
        let mut head = String::new();
        if k.is_pub { head.push_str("pub "); }
        let _ = write!(head, "kernel {}", k.name);
        head.push_str(&self.type_params(&k.type_params, &k.where_clause, &[], &[])?);
        self.line(&format!("{head}:"));
        self.indent += 1;
        for f in &k.fields {
            let kw = match f.binding { FieldBinding::Let => "let", FieldBinding::Mut => "mut", FieldBinding::Var => "var" };
            let mut s = format!("{kw} {}", self.kernel_field_ty(&f.ty, &f.qual)?);
            let _ = write!(s, " {}", f.name);
            if let Some(d) = &f.default {
                s.push_str(" = ");
                let mut sub = self.sub();
                s.push_str(&sub.expr(d, P_ELSE)?);
            }
            self.line(&s);
        }
        for i in &k.inits { self.init_decl(i)?; }
        for m in &k.methods { self.fn_decl(m, &k.type_params)?; }
        if k.fields.is_empty() && k.inits.is_empty() && k.methods.is_empty() { self.line("pass"); }
        self.indent -= 1;
        Ok(())
    }

    /// `[float]'global` etc.: the field type with its kernel-memory qualifier.
    fn kernel_field_ty(&self, ty: &Type, qual: &GpuQual) -> R<String> {
        let t = self.ty(ty)?;
        let q = match qual {
            GpuQual::Unified => "unified",
            GpuQual::Global => "global",
            GpuQual::Actor => "actor",
            GpuQual::Local => "local",
            GpuQual::Const => "const",
            GpuQual::ActorGlobal => "actor'global",
            GpuQual::ActorUnified => "actor'unified",
            GpuQual::Surface => "surface",
        };
        Ok(format!("{t}'{q}"))
    }

    // ─── Types ───────────────────────────────────────────────────────────────

    /// Like `ty`, but writes `T&` for a mutable borrow and strips a leading `mut`: used where a
    /// binding keyword (`mut`, `var mut`) has already said it.
    fn ty_no_leading_mut(&self, t: &Type) -> R<String> {
        match t {
            Type::Mut(inner) => self.ty(inner),
            Type::Qualified(inner, OwnerQual::BorrowMut) => Ok(format!("{}&", self.ty(inner)?)),
            Type::Qualified(inner, OwnerQual::BorrowOptionMut) => Ok(format!("{}?&", self.ty(inner)?)),
            other => self.ty(other),
        }
    }

    fn ty(&self, t: &Type) -> R<String> {
        Ok(match t {
            Type::Int => "int".into(),
            Type::Uint => "uint".into(),
            Type::Uint8 => "uint8".into(),
            Type::Int8 => "int8".into(),
            Type::Int16 => "int16".into(),
            Type::Int32 => "int32".into(),
            Type::Int64 => "int64".into(),
            Type::Int128 => "int128".into(),
            Type::Uint16 => "uint16".into(),
            Type::Uint32 => "uint32".into(),
            Type::Uint64 => "uint64".into(),
            Type::Uint128 => "uint128".into(),
            Type::Float16 => "float16".into(),
            Type::Fixed(f) => f.name(),
            Type::Float32 => "float32".into(),
            Type::Float64 => "float".into(),
            Type::Str => "string".into(),
            Type::Bool => "bool".into(),
            Type::Nil => "nil".into(),
            Type::Void => "void".into(),
            Type::Never => "never".into(),
            Type::Named(n) => n.clone(),
            Type::TypeParam(n) => n.clone(),
            Type::Optional(inner) => match &**inner {
                Type::Fn(..) => format!("({})?", self.ty(inner)?),
                Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut | OwnerQual::Lifetime(_)) => {
                    return unsupported("optional of a borrow type (`T&?`)", 0)
                }
                _ => format!("{}?", self.ty(inner)?),
            },
            Type::Array(inner) => format!("[{}]", self.ty(inner)?),
            Type::ArrayN(inner, n) => format!("[{}, {}]", self.ty(inner)?, n),
            Type::ArrayNExpr(inner, size) => {
                let mut sub = self.sub();
                format!("[{}, {}]", self.ty(inner)?, sub.expr(&size.0, P_ELSE)?)
            }
            Type::LabeledArray(inner, axes) => {
                let mut parts = Vec::new();
                for a in axes {
                    match &a.size {
                        None => parts.push(a.label.clone()),
                        Some(sz) => {
                            let mut sub = self.sub();
                            parts.push(format!("{} = {}", a.label, sub.expr(&sz.0, P_ELSE)?));
                        }
                    }
                }
                format!("[{}, {}]", self.ty(inner)?, parts.join(", "))
            }
            Type::ConstInt(n) => n.to_string(),
            Type::Tuple(ts) => {
                let parts: R<Vec<String>> = ts.iter().map(|t| self.ty(t)).collect();
                let parts = parts?;
                if parts.len() == 1 { format!("({},)", parts[0]) } else { format!("({})", parts.join(", ")) }
            }
            Type::Dict(k, v) => format!("{{{} = {}}}", self.ty(k)?, self.ty(v)?),
            Type::Set(inner) => format!("{{{}}}", self.ty(inner)?),
            Type::Fn(ret, params, throws, task, req) => {
                let ps: R<Vec<String>> = params.iter().map(|t| self.ty(t)).collect();
                let ps = ps?.join(", ");
                let ret_text = match ret { Some(r) => self.ty(r)?, None => "Fn".to_string() };
                if *req {
                    format!("req {}{}({}){}", if *task { "task " } else { "" }, ret_text, ps, if *throws { " throws" } else { "" })
                } else {
                    format!("{}({}){}{}", ret_text, ps, if *task { " task" } else { "" }, if *throws { " throws" } else { "" })
                }
            }
            Type::Qualified(inner, q) => self.qualified(inner, q)?,
            Type::Dyn(inner) => self.ty(inner)?,
            Type::Impl(inner) => format!("<{}>", self.ty(inner)?),
            Type::Generic(name, args) => {
                let labels = self.qual_labels.get(name);
                let mut parts = Vec::new();
                for (i, a) in args.iter().enumerate() {
                    match a {
                        Type::QualArg(chain) => {
                            let label = labels.and_then(|l| l.get(i)).and_then(|l| l.clone());
                            let Some(label) = label else {
                                return unsupported(format!("qualifier argument of `{name}` whose declaration is not in this file"), 0);
                            };
                            parts.push(self.qual_arg_text(&label, chain)?);
                        }
                        other => parts.push(self.ty(other)?),
                    }
                }
                format!("{}<{}>", name, parts.join(", "))
            }
            Type::QualArg(_) => return unsupported("qualifier argument outside a generic argument list", 0),
            Type::SelfAssoc(n) => format!("Self.{n}"),
            Type::AssocOf(base, n) => format!("{}.{}", self.ty(base)?, n),
            Type::Mut(inner) => format!("mut {}", self.ty(inner)?),
        })
    }

    fn qual_arg_text(&self, label: &str, chain: &[OwnerQual]) -> R<String> {
        let mut s = label.to_string();
        for q in chain {
            s.push('\'');
            s.push_str(&self.qual_word(q)?);
        }
        Ok(s)
    }

    /// The word after the tick for a plain (non-borrow) qualifier.
    fn qual_word(&self, q: &OwnerQual) -> R<String> {
        Ok(match q {
            OwnerQual::Owned => "owned".into(),
            OwnerQual::Actor => "actor".into(),
            OwnerQual::ActorTask => "actor'task".into(),
            OwnerQual::Guard => "guard".into(),
            OwnerQual::GuardTask => "guard'task".into(),
            OwnerQual::Atomic => "atomic".into(),
            OwnerQual::Shared => "shared".into(),
            OwnerQual::Weak => "weak".into(),
            OwnerQual::Inline => "inline".into(),
            OwnerQual::Static => "static".into(),
            OwnerQual::GpuUnified => "unified".into(),
            OwnerQual::GpuGlobal => "global".into(),
            OwnerQual::GpuLocal => "local".into(),
            OwnerQual::GpuConst => "const".into(),
            OwnerQual::GpuActorGlobal => "actor'global".into(),
            OwnerQual::GpuActorUnified => "actor'unified".into(),
            OwnerQual::GpuSurface => "surface".into(),
            OwnerQual::Observed => "observed".into(),
            OwnerQual::Param(n) => n.split('.').next().unwrap_or(n).to_string(),
            OwnerQual::Union(members) => {
                if q.is_new() { "new".into() }
                else if q.is_kernel_marker() { "kernel".into() }
                else if members.as_slice() == [OwnerQual::Inline, OwnerQual::Owned] { "one".into() }
                else if members.as_slice() == [OwnerQual::Shared, OwnerQual::Actor, OwnerQual::Guard] { "many".into() }
                else if members.as_slice() == [OwnerQual::Inline, OwnerQual::Owned, OwnerQual::Actor, OwnerQual::Guard] { "mut".into() }
                else if members.as_slice() == [OwnerQual::Shared, OwnerQual::Static] { "req".into() }
                else {
                    let mut words = Vec::new();
                    for m in members {
                        words.push(match m {
                            OwnerQual::Inline => "inline",
                            OwnerQual::Owned => "owned",
                            OwnerQual::Shared => "shared",
                            OwnerQual::Actor => "actor",
                            OwnerQual::Atomic => "atomic",
                            OwnerQual::Guard => "guard",
                            _ => return unsupported("qualifier union with an unwritable member", 0),
                        });
                    }
                    words.join("|")
                }
            }
            other => return unsupported(format!("qualifier {other:?} written as a tick qualifier"), 0),
        })
    }

    fn qualified(&self, inner: &Type, q: &OwnerQual) -> R<String> {
        Ok(match q {
            OwnerQual::Borrow => format!("{}&", self.ty(inner)?),
            OwnerQual::BorrowMut => format!("mut {}&", self.ty(inner)?),
            OwnerQual::BorrowOption => format!("{}?&", self.ty(inner)?),
            OwnerQual::BorrowOptionMut => format!("mut {}?&", self.ty(inner)?),
            OwnerQual::BorrowWeak => format!("{}&weak", self.ty(inner)?),
            OwnerQual::Lifetime(l) => match inner {
                // `mut T&a`
                Type::Qualified(base, OwnerQual::BorrowMut) => format!("mut {}&{}", self.ty(base)?, l),
                // `T'qual&a`
                other => format!("{}&{}", self.ty(other)?, l),
            },
            OwnerQual::BorrowShared | OwnerQual::BorrowOwned => {
                return unsupported("internal borrow qualifier (`BorrowShared`/`BorrowOwned`) in a type", 0)
            }
            other => format!("{}'{}", self.ty(inner)?, self.qual_word(other)?),
        })
    }
}

/// The fresh type parameters of the untyped parameters of `f` (`T_<name>`): printed back as
/// untyped parameters, which re-derive them and their bounds.
fn implicit_param_types(f: &FnDecl) -> Vec<String> {
    if f.qualifier.is_some() || f.task || f.stream {
        return Vec::new();
    }
    let mut out = Vec::new();
    for p in &f.params {
        if p.variadic || p.default.is_some() {
            continue;
        }
        if let Some(Type::TypeParam(t)) = &p.ty {
            if *t == crate::infer_param_generics::fresh_name(&p.name) && f.type_params.contains(t) {
                out.push(t.clone());
            }
        } else if let Some(ty) = &p.ty {
            if let Some((_, names)) = bare_struct_param(ty, &p.name, &f.type_params) {
                out.extend(names);
            }
        }
    }
    out
}

/// A parameter typed `Response<T_r__status, T_r__payload>` by `desugar_struct_params` (written
/// `Response r`): the bare type and the implicit parameters it stands for.
fn bare_struct_param(ty: &Type, pname: &str, type_params: &[String]) -> Option<(Type, Vec<String>)> {
    match ty {
        Type::Mut(i) => bare_struct_param(i, pname, type_params).map(|(t, n)| (Type::Mut(Box::new(t)), n)),
        Type::Qualified(i, q) => bare_struct_param(i, pname, type_params).map(|(t, n)| (Type::Qualified(Box::new(t), q.clone()), n)),
        Type::Generic(name, args) if !args.is_empty() => {
            let prefix = format!("T_{pname}__");
            let names: Vec<String> = args.iter().filter_map(|a| match a {
                Type::TypeParam(n) if n.starts_with(&prefix) && type_params.contains(n) => Some(n.clone()),
                _ => None,
            }).collect();
            if names.len() == args.len() { Some((Type::Named(name.clone()), names)) } else { None }
        }
        _ => None,
    }
}

/// Whether `ty` is the fresh type parameter of the untyped field `name` (possibly under `mut`).
fn is_fresh_field_type(ty: &Type, name: &str) -> bool {
    let fresh = crate::infer_param_generics::fresh_name(name);
    match ty {
        Type::TypeParam(n) => *n == fresh,
        Type::Mut(inner) => matches!(&**inner, Type::TypeParam(n) if *n == fresh),
        _ => false,
    }
}

fn collect_qual_labels(items: &[Item], out: &mut HashMap<String, Vec<Option<String>>>) {
    fn labels(params: &[String]) -> Vec<Option<String>> {
        // Positional slots of a generic argument list: every parameter in order; a qualifier
        // parameter has a label, the others none.
        params.iter().map(|p| qual_param_name(p).map(|s| s.to_string())).collect()
    }
    for item in items {
        match item {
            Item::Struct(s) if !s.type_params.is_empty() => { out.insert(s.name.clone(), labels(&s.type_params)); }
            Item::Enum(e) if !e.type_params.is_empty() => { out.insert(e.name.clone(), labels(&e.type_params)); }
            Item::Fn(f) if !f.type_params.is_empty() => { out.insert(f.name.clone(), labels(&f.type_params)); }
            Item::Mod(m) => collect_qual_labels(&m.items, out),
            _ => {}
        }
    }
}

mod stmt;
mod expr;
pub mod full;
#[cfg(test)]
mod tests;
