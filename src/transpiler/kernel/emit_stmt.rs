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

// Statement emission for the kernel transpiler.

use crate::ast::{Stmt, IfStmt, ForStmt, WhileStmt};
use super::helpers::KernelTranspiler;

impl KernelTranspiler {
    /// Emit a statement (non-last in a block — always adds semicolon for expressions).
    pub(super) fn emit_stmt(&mut self, stmt: &Stmt) {
        self.emit_stmt_impl(stmt, false);
    }

    /// Emit the last statement in a non-throws function body (no semicolon for expression).
    pub(super) fn emit_stmt_last(&mut self, stmt: &Stmt) {
        self.emit_stmt_impl(stmt, true);
    }

    /// Emit the last statement in a task body function (wraps bare expression in `Ok(...)`).
    pub(super) fn emit_stmt_last_ok(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr(e) => {
                let s = self.emit_expr(e);
                self.line(&format!("Ok({})", s));
            }
            Stmt::Return(r) => {
                match &r.value {
                    None    => self.line("return Ok(());"),
                    Some(v) => {
                        let s = self.emit_expr(v);
                        self.line(&format!("return Ok({});", s));
                    }
                }
            }
            _ => self.emit_stmt(stmt),
        }
    }

    fn emit_stmt_impl(&mut self, stmt: &Stmt, is_last: bool) {
        match stmt {
            Stmt::Let(s) => {
                use crate::ast::ExprKind;
                if let Some(val) = &s.value {
                    // Track `let tx = broadcast<T>(cap)` (single-binding sender).
                    let is_broadcast_call = matches!(&val.kind,
                        ExprKind::GenericCall(callee, _, _)
                        if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"))
                        || matches!(&val.kind,
                        ExprKind::Call(callee, _)
                        if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"));
                    if is_broadcast_call && s.name != "_" {
                        self.broadcast_senders.insert(s.name.clone());
                    }
                    // Track `let rx = tx.subscribe()` so rx.recv() is dispatched correctly.
                    if let ExprKind::MethodCall(obj, method, _) = &val.kind {
                        if method == "subscribe" {
                            if let ExprKind::Var(tx_name) = &obj.kind {
                                if self.broadcast_senders.contains(tx_name.as_str()) {
                                    self.broadcast_receivers.insert(s.name.clone());
                                }
                            }
                        }
                    }
                }
                let kw = if s.mutable { "let mut" } else { "let" };
                match &s.value {
                    None => {
                        if let Some(ty) = &s.ty {
                            self.line(&format!("{} {}: {};", kw, s.name, self.emit_type(ty)));
                        } else {
                            self.line(&format!("{} {};", kw, s.name));
                        }
                    }
                    Some(val) => {
                        let val_s = self.emit_expr(val);
                        if let Some(ty) = &s.ty {
                            let ty_s = self.emit_type(ty);
                            self.line(&format!("{} {}: {} = {};", kw, s.name, ty_s, val_s));
                        } else {
                            self.line(&format!("{} {} = {};", kw, s.name, val_s));
                        }
                    }
                }
            }

            Stmt::Return(r) => {
                match &r.value {
                    None    => self.line("return;"),
                    Some(v) => {
                        let s = self.emit_expr(v);
                        self.line(&format!("return {};", s));
                    }
                }
            }

            Stmt::Break(_, val) => {
                match val {
                    Some(e) => {
                        let s = self.emit_expr(e);
                        self.line(&format!("break {};", s));
                    }
                    None => self.line("break;"),
                }
            }

            Stmt::Continue(_) => self.line("continue;"),

            Stmt::If(s) => self.emit_if(s, is_last),

            Stmt::While(s) => self.emit_while(s),

            Stmt::For(s) => self.emit_for(s),

            Stmt::Expr(e) => {
                if is_last {
                    // Last expression in a non-throws function: no semicolon
                    let s = self.emit_expr(e);
                    self.line(&s);
                } else {
                    let s = self.emit_expr(e);
                    self.line(&format!("{};", s));
                }
            }

            Stmt::Fn(f) => {
                self.emit_fn(f, None);
                self.blank();
            }

            Stmt::Struct(s) => {
                self.emit_struct(s);
                self.blank();
            }

            Stmt::Enum(e) => {
                self.emit_enum(e);
                self.blank();
            }

            Stmt::Comment(text) => {
                self.line(&format!("// {}", text));
            }

            Stmt::LetDestructure(s) => {
                use crate::ast::ExprKind;
                // Detect `let tx, rx = oneshot<T>()` and register the variables.
                let is_oneshot = matches!(&s.value.kind,
                    ExprKind::GenericCall(callee, _, _)
                    if matches!(&callee.kind, ExprKind::Var(n) if n == "oneshot"))
                    || matches!(&s.value.kind,
                    ExprKind::Call(callee, _)
                    if matches!(&callee.kind, ExprKind::Var(n) if n == "oneshot"));
                if is_oneshot {
                    if let (Some(sender), Some(receiver)) = (s.bindings.get(0), s.bindings.get(1)) {
                        if sender.name != "_" { self.oneshot_senders.insert(sender.name.clone()); }
                        if receiver.name != "_" { self.oneshot_receivers.insert(receiver.name.clone()); }
                    }
                }
                // Detect `let tx, rx = broadcast<T>(cap)` and register the variables.
                let is_broadcast = matches!(&s.value.kind,
                    ExprKind::GenericCall(callee, _, _)
                    if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"))
                    || matches!(&s.value.kind,
                    ExprKind::Call(callee, _)
                    if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"));
                if is_broadcast {
                    if let Some(sender) = s.bindings.get(0) {
                        if sender.name != "_" { self.broadcast_senders.insert(sender.name.clone()); }
                    }
                    if let Some(receiver) = s.bindings.get(1) {
                        if receiver.name != "_" { self.broadcast_receivers.insert(receiver.name.clone()); }
                    }
                }
                // Detect `let tx, rx = watch<T>(initial)` and register the variables.
                let is_watch = matches!(&s.value.kind,
                    ExprKind::GenericCall(callee, _, _)
                    if matches!(&callee.kind, ExprKind::Var(n) if n == "watch"))
                    || matches!(&s.value.kind,
                    ExprKind::Call(callee, _)
                    if matches!(&callee.kind, ExprKind::Var(n) if n == "watch"));
                if is_watch {
                    if let (Some(sender), Some(receiver)) = (s.bindings.get(0), s.bindings.get(1)) {
                        if sender.name != "_" { self.watch_senders.insert(sender.name.clone()); }
                        if receiver.name != "_" { self.watch_receivers.insert(receiver.name.clone()); }
                    }
                }
                // watch receiver must be `mut` (calls `recv(&mut self)`); oneshot is consumed once.
                let names: Vec<String> = if is_watch && s.bindings.len() == 2 {
                    vec![
                        s.bindings[0].name.clone(),
                        format!("mut {}", s.bindings[1].name),
                    ]
                } else {
                    s.bindings.iter().map(|b| b.name.clone()).collect()
                };
                let val_s = self.emit_expr(&s.value);
                self.line(&format!("let ({}) = {};", names.join(", "), val_s));
            }

            Stmt::Throw(t) => {
                match &t.value {
                    None    => self.line("return Err(kernel::error::code::EINVAL.into());"),
                    Some(v) => {
                        let s = self.emit_expr(v);
                        self.line(&format!("return Err({}.into());", s));
                    }
                }
            }

            // Loops
            Stmt::Loop(s) => {
                self.line("loop {");
                self.indent += 1;
                for stmt in &s.body { self.emit_stmt(stmt); }
                self.indent -= 1;
                self.line("}");
            }

            Stmt::DoWhile(s) => {
                self.line("loop {");
                self.indent += 1;
                for stmt in &s.body { self.emit_stmt(stmt); }
                let cond_s = self.emit_expr(&s.condition);
                self.line(&format!("if !({}) {{ break; }}", cond_s));
                self.indent -= 1;
                self.line("}");
            }

            Stmt::WhileLet(s) => {
                let val_s = self.emit_expr(&s.value);
                self.line(&format!("while let Some({}) = {} {{", s.name, val_s));
                self.indent += 1;
                for stmt in &s.body { self.emit_stmt(stmt); }
                self.indent -= 1;
                self.line("}");
            }

            Stmt::Alias(a) => {
                let ty_s = self.emit_type(&a.ty);
                self.line(&format!("type {} = {};", a.name, ty_s));
            }

            Stmt::Mod(m) => {
                for item in &m.items {
                    self.emit_item(item);
                    self.blank();
                }
            }

            // Unsupported in kernel
            Stmt::Wait(_, _) => {
                self.line("// TODO: kernel wait (no async)");
            }
            Stmt::Try(s) => {
                // Simple try: emit body, ignore catches (kernel uses Result)
                for stmt in &s.body { self.emit_stmt(stmt); }
            }
            Stmt::Guard(s) => {
                // guard cond else: body
                let cond_s = match &s.cond {
                    crate::ast::GuardCond::Expr(e) => self.emit_expr(e),
                    crate::ast::GuardCond::Clauses(_) => "/* TODO: kernel guard let */".into(),
                };
                self.line(&format!("if !({}) {{", cond_s));
                self.indent += 1;
                for stmt in &s.else_body { self.emit_stmt(stmt); }
                self.indent -= 1;
                self.line("}");
            }
            Stmt::IfLet(_) => self.line("// TODO: kernel if let"),
            Stmt::Match(s) => self.emit_match_stmt(s),
            Stmt::Defer(_) => self.line("// TODO: kernel defer"),
            Stmt::Yield(e, _) => {
                let s = self.emit_expr(e);
                if self.in_iter_stream {
                    self.line(&format!("__items.push({});", s));
                } else if self.in_stream_body {
                    self.line(&format!("this.tx.send({});", s));
                } else {
                    self.line(&format!("yield {};", s));
                }
            }
        }
    }

    fn emit_if(&mut self, s: &IfStmt, is_last: bool) {
        for (i, (cond, body)) in s.branches.iter().enumerate() {
            let cond_s = self.emit_expr(cond);
            if i == 0 {
                self.line(&format!("if {} {{", cond_s));
            } else {
                self.line(&format!("}} else if {} {{", cond_s));
            }
            self.indent += 1;
            let len = body.len();
            for (j, stmt) in body.iter().enumerate() {
                if is_last && j + 1 == len {
                    self.emit_stmt_last(stmt);
                } else {
                    self.emit_stmt(stmt);
                }
            }
            self.indent -= 1;
        }
        if let Some(else_body) = &s.else_body {
            self.line("} else {");
            self.indent += 1;
            let len = else_body.len();
            for (j, stmt) in else_body.iter().enumerate() {
                if is_last && j + 1 == len {
                    self.emit_stmt_last(stmt);
                } else {
                    self.emit_stmt(stmt);
                }
            }
            self.indent -= 1;
        }
        self.line("}");
    }

    fn emit_while(&mut self, s: &WhileStmt) {
        let cond_s = self.emit_expr(&s.condition);
        self.line(&format!("while {} {{", cond_s));
        self.indent += 1;
        for stmt in &s.body { self.emit_stmt(stmt); }
        self.indent -= 1;
        self.line("}");
    }

    fn emit_for(&mut self, s: &ForStmt) {
        // `for msg in rx` on a broadcast receiver → `loop { let msg = rx.recv(); body }`
        use crate::ast::ExprKind;
        if let ExprKind::Var(rx_name) = &s.iterable.kind {
            if self.broadcast_receivers.contains(rx_name.as_str()) {
                let vars_s = if s.vars.is_empty() {
                    "_".into()
                } else {
                    s.vars.join(", ")
                };
                self.line("loop {");
                self.indent += 1;
                self.line(&format!("let {} = {}.recv();", vars_s, rx_name));
                for stmt in &s.body { self.emit_stmt(stmt); }
                self.indent -= 1;
                self.line("}");
                return;
            }
        }
        let iter_s = self.emit_expr(&s.iterable);
        let vars_s = if s.vars.len() == 1 {
            s.vars[0].clone()
        } else {
            format!("({})", s.vars.join(", "))
        };
        self.line(&format!("for {} in {} {{", vars_s, iter_s));
        self.indent += 1;
        for stmt in &s.body { self.emit_stmt(stmt); }
        self.indent -= 1;
        self.line("}");
    }

    fn emit_match_stmt(&mut self, s: &crate::ast::MatchStmt) {
        let subj = self.emit_expr(&s.subject);
        self.line(&format!("match {} {{", subj));
        self.indent += 1;
        for arm in &s.arms {
            let pats: Vec<String> = arm.patterns.iter().map(|p| self.emit_pattern(p)).collect();
            let guard_s = arm.guard.as_ref()
                .map(|g| format!(" if {}", self.emit_expr(g)))
                .unwrap_or_default();
            let pat_s = format!("{}{}", pats.join(" | "), guard_s);
            match &arm.body {
                crate::ast::MatchBody::Expr(e) => {
                    let es = self.emit_expr(e);
                    self.line(&format!("{} => {},", pat_s, es));
                }
                crate::ast::MatchBody::Block(stmts) => {
                    self.line(&format!("{} => {{", pat_s));
                    self.indent += 1;
                    for stmt in stmts { self.emit_stmt(stmt); }
                    self.indent -= 1;
                    self.line("},");
                }
            }
        }
        self.indent -= 1;
        self.line("}");
    }

    fn emit_pattern(&self, pat: &crate::ast::Pattern) -> String {
        use crate::ast::Pattern;
        match pat {
            Pattern::Wildcard      => "_".into(),
            Pattern::Bind(name)    => name.clone(),
            Pattern::None          => "None".into(),
            Pattern::Some(inner)   => format!("Some({})", self.emit_pattern(inner)),
            Pattern::Variant(name, fields) => {
                if fields.is_empty() {
                    name.clone()
                } else {
                    let fs: Vec<String> = fields.iter().map(|f| self.emit_pattern(f)).collect();
                    format!("{}({})", name, fs.join(", "))
                }
            }
            Pattern::Tuple(elems) => {
                let es: Vec<String> = elems.iter().map(|e| self.emit_pattern(e)).collect();
                format!("({})", es.join(", "))
            }
            Pattern::Lit(lit) => {
                use crate::ast::LitPattern;
                match lit {
                    LitPattern::Int(n)  => n.to_string(),
                    LitPattern::Float(f) => format!("/* float forbidden */ {}", f),
                    LitPattern::Str(s)  => format!("\"{}\"", s),
                    LitPattern::Bool(b) => b.to_string(),
                    LitPattern::Nil     => "None".into(),
                }
            }
        }
    }
}
