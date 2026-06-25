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

// Top-level item emission for the kernel transpiler.

use crate::ast::{Item, FnDecl, StructDecl, EnumDecl};
use super::helpers::KernelTranspiler;

impl KernelTranspiler {
    pub(super) fn emit_item(&mut self, item: &Item) {
        match item {
            Item::Fn(f)     => self.emit_fn(f, None),
            Item::Struct(s) => self.emit_struct(s),
            Item::Enum(e)   => self.emit_enum(e),
            Item::Use(_)    => { /* use declarations are not emitted for kernel targets */ }
            Item::Alias(a)  => {
                let rust_ty = self.emit_type(&a.ty);
                self.line(&format!("type {} = {};", a.name, rust_ty));
            }
            Item::Let(s)    => {
                // Top-level let/var becomes a static or const
                if let Some(val) = &s.value {
                    let val_s = self.emit_expr(val);
                    if let Some(ty) = &s.ty {
                        let ty_s = self.emit_type(ty);
                        if s.binding.is_mutable() {
                            self.line(&format!("static mut {}: {} = {};", s.name, ty_s, val_s));
                        } else {
                            self.line(&format!("static {}: {} = {};", s.name, ty_s, val_s));
                        }
                    } else if s.binding.is_mutable() {
                        self.line(&format!("// TODO: kernel top-level var {} = {};", s.name, val_s));
                    } else {
                        self.line(&format!("// TODO: kernel top-level let {} = {};", s.name, val_s));
                    }
                }
            }
            Item::Stmt(stmt) => self.emit_stmt(stmt),
            Item::Mod(m) => {
                for item in &m.items {
                    self.emit_item(item);
                    self.blank();
                }
            }
            Item::Trait(_) | Item::Ext(_) => {
                self.line("// TODO: kernel trait/ext");
            }
            Item::Kernel(_) => { /* GPU kernel struct inside Linux kernel module — not supported */ }
        }
    }

    /// Emit a `task def` function as a Work item + KernelFuture wrapper.
    ///
    /// Generates four pieces:
    ///   1. `struct XxxWork { params..., result, done_cond, work }`
    ///   2. `impl kernel::workqueue::Work<XxxWork> for XxxWork { fn run(...) }`
    ///   3. wrapper `fn xxx(params) -> KernelFuture<T>`
    ///   4. body `fn xxx_body(params) -> Result<T, kernel::error::Error>`
    pub(super) fn emit_task_fn(&mut self, f: &FnDecl, self_ty: Option<&str>) {
        self.has_task_fns = true;

        // Capitalise first letter for the struct name: fetch_page → FetchPage
        let struct_name = {
            let camel: String = f.name.split('_')
                .map(|s| {
                    let mut c = s.chars();
                    match c.next() {
                        None => String::new(),
                        Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                    }
                })
                .collect();
            format!("{}Work", camel)
        };

        let ret_ty = f.return_ty.as_ref()
            .map(|t| self.emit_type(t))
            .unwrap_or_else(|| "()".to_string());

        // ── 1. Work item struct ───────────────────────────────────────────────
        self.line(&format!("struct {} {{", struct_name));
        self.indent += 1;

        // Receiver field for instance methods (`task def Foo.method(...)`)
        if let Some(receiver_ty) = self_ty {
            self.line(&format!("receiver: Arc<{}>,", receiver_ty));
        }

        // Parameters
        for p in &f.params {
            if let Some(ty) = &p.ty {
                let ty_s = self.emit_type(ty);
                self.line(&format!("{}: {},", p.name, ty_s));
            }
        }
        // Result + synchronisation
        self.line(&format!(
            "result: Arc<kernel::sync::Mutex<Option<Result<{}, kernel::error::Error>>>>,",
            ret_ty
        ));
        self.line("done_cond: Arc<kernel::sync::CondVar>,");
        // Self-referential Work field
        self.line(&format!(
            "work: kernel::workqueue::Work<{}>,",
            struct_name
        ));
        self.indent -= 1;
        self.line("}");
        self.blank();

        // ── 2. impl Work ─────────────────────────────────────────────────────
        self.line(&format!(
            "impl kernel::workqueue::Work<{sn}> for {sn} {{",
            sn = struct_name
        ));
        self.indent += 1;
        self.line(&format!("fn run(this: Arc<Self>) {{"));
        self.indent += 1;

        // Build the call to the _body function
        let body_args: Vec<String> = f.params.iter().map(|p| {
            format!("this.{}", p.name)
        }).collect();

        if let Some(_receiver_ty) = self_ty {
            self.line(&format!(
                "let r = Arc::clone(&this.receiver).{}_body({});",
                f.name,
                body_args.join(", ")
            ));
        } else {
            self.line(&format!(
                "let r = {}_body({});",
                f.name,
                body_args.join(", ")
            ));
        }
        self.line("*this.result.lock() = Some(r);");
        self.line("this.done_cond.notify_all();");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();

        // ── 3. Wrapper fn ────────────────────────────────────────────────────
        let vis = if f.is_pub { "pub " } else { "" };

        let params_s: Vec<String> = f.params.iter().map(|p| {
            let ty_s = p.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_default();
            format!("{}: {}", p.name, ty_s)
        }).collect();

        let wrapper_params = if let Some(receiver_ty) = self_ty {
            let self_ref = if f.mutating { "&mut self" } else { "&self" };
            let base = format!("receiver: Arc<{}>", receiver_ty);
            if params_s.is_empty() {
                format!("{}, {}", self_ref, base)
            } else {
                format!("{}, {}, {}", self_ref, base, params_s.join(", "))
            }
        } else {
            params_s.join(", ")
        };

        self.line(&format!(
            "{}fn {}({}) -> KernelFuture<{}> {{",
            vis, f.name, wrapper_params, ret_ty
        ));
        self.indent += 1;
        self.line(&format!(
            "let result = Arc::new(kernel::sync::Mutex::new(None::<Result<{}, kernel::error::Error>>));",
            ret_ty
        ));
        self.line("let done_cond = Arc::new(kernel::sync::CondVar::new());");

        // Build struct initialiser
        self.line(&format!("let work = Arc::new({} {{", struct_name));
        self.indent += 1;
        if self_ty.is_some() {
            self.line("receiver,");
        }
        for p in &f.params {
            self.line(&format!("{},", p.name));
        }
        self.line("result: Arc::clone(&result),");
        self.line("done_cond: Arc::clone(&done_cond),");
        self.line(&format!("work: kernel::workqueue::Work::new(),"));
        self.indent -= 1;
        self.line("});");

        self.line("kernel::workqueue::system().enqueue(Arc::clone(&work));");
        self.line("KernelFuture { result, done_cond }");
        self.indent -= 1;
        self.line("}");
        self.blank();

        // ── 4. Body fn ───────────────────────────────────────────────────────
        // The body function has the real logic and always returns Result<T, Error>
        let body_params_s: Vec<String> = f.params.iter().map(|p| {
            let ty_s = p.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_default();
            format!("{}: {}", p.name, ty_s)
        }).collect();

        let body_all_params = if let Some(receiver_ty) = self_ty {
            let self_ref = if f.mutating { "&mut self" } else { "&self" };
            if body_params_s.is_empty() {
                format!("{}", self_ref)
            } else {
                format!("{}, _receiver: Arc<{}>, {}", self_ref, receiver_ty, body_params_s.join(", "))
            }
        } else {
            body_params_s.join(", ")
        };

        self.line(&format!(
            "fn {}_body({}) -> Result<{}, kernel::error::Error> {{",
            f.name, body_all_params, ret_ty
        ));
        self.indent += 1;

        let body_len = f.body.len();
        for (i, stmt) in f.body.iter().enumerate() {
            let is_last = i + 1 == body_len;
            if is_last {
                // Wrap last expression in Ok(...)
                self.emit_stmt_last_ok(stmt);
            } else {
                self.emit_stmt(stmt);
            }
        }

        self.indent -= 1;
        self.line("}");
    }

    /// Emit a `stream def` function as a Work item + KernelReceiver<T, N> wrapper.
    ///
    /// Emit a purely-sequential `stream def` as `impl Iterator<Item = T>` (no workqueue).
    /// `yield expr` → `__items.push(expr)`, body runs eagerly, returns `vec.into_iter()`.
    pub(super) fn emit_iter_stream_fn(&mut self, f: &FnDecl, self_ty: Option<&str>) {
        let vis = if f.is_pub { "pub " } else { "" };
        let item_ty = f.return_ty.as_ref()
            .map(|t| self.emit_type(t))
            .unwrap_or_else(|| "()".to_string());

        let params_s: Vec<String> = f.params.iter().map(|p| {
            let ty_s = p.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_default();
            format!("{}: {}", p.name, ty_s)
        }).collect();
        let all_params = match self_ty {
            Some(_) => {
                let self_ref = if f.mutating { "&mut self" } else { "&self" };
                if params_s.is_empty() { self_ref.to_string() }
                else { format!("{}, {}", self_ref, params_s.join(", ")) }
            }
            None => params_s.join(", "),
        };

        self.line(&format!(
            "{}fn {}({}) -> impl Iterator<Item = {}> {{",
            vis, f.name, all_params, item_ty
        ));
        self.indent += 1;
        self.line(&format!("let mut __items: kernel::prelude::Vec<{}> = kernel::prelude::Vec::new();", item_ty));

        self.in_iter_stream = true;
        for stmt in &f.body { self.emit_stmt(stmt); }
        self.in_iter_stream = false;

        self.line("__items.into_iter()");
        self.indent -= 1;
        self.line("}");
    }

    /// Generates three pieces:
    ///   1. `struct XxxWork { params..., tx: KernelSender<T, N> }`
    ///   2. `impl kernel::workqueue::Work<XxxWork> { fn run(...) }` with `yield` → `this.tx.send(...)`
    ///   3. `fn xxx(params) -> KernelReceiver<T, N>` that creates the channel, enqueues the work, returns rx
    pub(super) fn emit_stream_fn(&mut self, f: &FnDecl, self_ty: Option<&str>) {
        self.has_stream = true;

        // Default capacity — the parser doesn't store stream<N> capacity yet
        let capacity: usize = f.stream_capacity.unwrap_or(2);

        // Capitalise first letter for the struct name: words → WordsWork
        let struct_name = {
            let camel: String = f.name.split('_')
                .map(|s| {
                    let mut c = s.chars();
                    match c.next() {
                        None => String::new(),
                        Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                    }
                })
                .collect();
            format!("{}Work", camel)
        };

        let item_ty = f.return_ty.as_ref()
            .map(|t| self.emit_type(t))
            .unwrap_or_else(|| "()".to_string());

        // ── 1. Work item struct ───────────────────────────────────────────────
        self.line(&format!("struct {} {{", struct_name));
        self.indent += 1;

        // Receiver field for instance methods
        if let Some(receiver_ty) = self_ty {
            self.line(&format!("receiver: Arc<{}>,", receiver_ty));
        }

        // Parameters
        for p in &f.params {
            if let Some(ty) = &p.ty {
                let ty_s = self.emit_type(ty);
                self.line(&format!("{}: {},", p.name, ty_s));
            }
        }
        // Sender end of the internal channel
        self.line(&format!(
            "tx: KernelSender<{}, {}>,",
            item_ty, capacity
        ));
        // Self-referential Work field
        self.line(&format!(
            "work: kernel::workqueue::Work<{}>,",
            struct_name
        ));
        self.indent -= 1;
        self.line("}");
        self.blank();

        // ── 2. impl Work ─────────────────────────────────────────────────────
        self.line(&format!(
            "impl kernel::workqueue::Work<{sn}> for {sn} {{",
            sn = struct_name
        ));
        self.indent += 1;
        self.line("fn run(this: Arc<Self>) {");
        self.indent += 1;

        // Emit body with yield → this.tx.send(...)
        self.in_stream_body = true;
        for stmt in &f.body {
            self.emit_stmt(stmt);
        }
        self.in_stream_body = false;
        // tx is dropped here when run() returns → signals end of stream to receiver

        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();

        // ── 3. Public function returning KernelReceiver<T, N> ────────────────
        let vis = if f.is_pub { "pub " } else { "" };

        let params_s: Vec<String> = f.params.iter().map(|p| {
            let ty_s = p.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_default();
            format!("{}: {}", p.name, ty_s)
        }).collect();

        let all_params = if let Some(receiver_ty) = self_ty {
            let self_ref = if f.mutating { "&mut self" } else { "&self" };
            let base = format!("receiver: Arc<{}>", receiver_ty);
            if params_s.is_empty() {
                format!("{}, {}", self_ref, base)
            } else {
                format!("{}, {}, {}", self_ref, base, params_s.join(", "))
            }
        } else {
            params_s.join(", ")
        };

        self.line(&format!(
            "{}fn {}({}) -> KernelReceiver<{}, {}> {{",
            vis, f.name, all_params, item_ty, capacity
        ));
        self.indent += 1;
        self.line(&format!(
            "let (tx, rx) = kernel_channel::<{}, {}>();",
            item_ty, capacity
        ));

        // Build struct initialiser
        self.line(&format!("let work = Arc::new({} {{", struct_name));
        self.indent += 1;
        if self_ty.is_some() {
            self.line("receiver,");
        }
        for p in &f.params {
            self.line(&format!("{},", p.name));
        }
        self.line("tx,");
        self.line(&format!("work: kernel::workqueue::Work::new(),"));
        self.indent -= 1;
        self.line("});");

        self.line("kernel::workqueue::system().enqueue(Arc::clone(&work));");
        self.line("rx");
        self.indent -= 1;
        self.line("}");
    }

    /// Emit a function declaration.
    pub(super) fn emit_fn(&mut self, f: &FnDecl, self_ty: Option<&str>) {
        if f.is_native { return; }

        // Delegate task functions to the specialized emitter
        if f.task {
            self.emit_task_fn(f, self_ty);
            return;
        }

        // Delegate stream functions to the specialized emitter
        if f.stream {
            let empty = std::collections::HashSet::new();
            if crate::transpiler::helpers::body_is_sequential(&f.body, &empty) {
                self.emit_iter_stream_fn(f, self_ty);
            } else {
                self.emit_stream_fn(f, self_ty);
            }
            return;
        }

        // Attributes
        for attr in &f.attrs {
            let args_s = if attr.args.is_empty() {
                String::new()
            } else {
                format!("({})", attr.args.join(", "))
            };
            self.line(&format!("#[{}{}]", attr.name, args_s));
        }

        let vis = if f.is_pub { "pub " } else { "" };

        // Parameters
        let params_s: Vec<String> = f.params.iter().map(|p| {
            let name = if p.mutable { format!("mut {}", p.name) } else { p.name.clone() };
            match &p.ty {
                Some(ty) => format!("{}: {}", name, self.emit_type(ty)),
                None => name,
            }
        }).collect();

        let all_params = match self_ty {
            Some(_) => {
                let self_s = if f.mutating { "&mut self" } else { "&self" };
                if params_s.is_empty() {
                    self_s.to_string()
                } else {
                    format!("{}, {}", self_s, params_s.join(", "))
                }
            }
            None => params_s.join(", "),
        };

        // Return type
        let ret_ty = if f.throws {
            let base = f.return_ty.as_ref()
                .map(|t| self.emit_type(t))
                .unwrap_or_else(|| "()".to_string());
            format!("Result<{}, kernel::error::Error>", base)
        } else {
            f.return_ty.as_ref()
                .map(|t| self.emit_type(t))
                .unwrap_or_else(|| "()".to_string())
        };

        // Type params (simple: just names with Clone bound)
        let type_params = if f.type_params.is_empty() {
            String::new()
        } else {
            let bounded: Vec<String> = f.type_params.iter()
                .map(|p| if p.starts_with('\'') { p.clone() } else { format!("{}: Clone", p) })
                .collect();
            format!("<{}>", bounded.join(", "))
        };

        let sig = format!("{}fn {}{}({}) -> {}", vis, f.name, type_params, all_params, ret_ty);

        if f.body.is_empty() {
            self.line(&format!("{} {{}}", sig));
            return;
        }

        self.line(&format!("{} {{", sig));
        self.indent += 1;

        // Emit function body
        let body_len = f.body.len();
        for (i, stmt) in f.body.iter().enumerate() {
            let is_last = i + 1 == body_len;
            if is_last && !f.throws {
                self.emit_stmt_last(stmt);
            } else {
                self.emit_stmt(stmt);
            }
        }

        // For throws functions, add implicit Ok(()) if the body doesn't end with a return/expr
        if f.throws {
            let last = f.body.last();
            let needs_ok = match last {
                Some(crate::ast::Stmt::Return(_)) | Some(crate::ast::Stmt::Expr(_)) => false,
                _ => true,
            };
            if needs_ok { self.line("Ok(())"); }
        }

        self.indent -= 1;
        self.line("}");
    }

    /// Emit a struct declaration with all its fields.
    pub(super) fn emit_struct(&mut self, s: &StructDecl) {
        if s.is_native { return; }

        let vis = if s.is_pub { "pub " } else { "" };

        // Type params
        let type_params = if s.type_params.is_empty() {
            String::new()
        } else {
            format!("<{}>", s.type_params.join(", "))
        };

        self.line(&format!("{}struct {}{} {{", vis, s.name, type_params));
        self.indent += 1;

        for field in &s.fields {
            let fvis = if field.is_pub { "pub " } else { "" };
            let ty_s = self.emit_type(&field.ty);
            self.line(&format!("{}{}: {},", fvis, field.name, ty_s));
        }

        self.indent -= 1;
        self.line("}");

        // Emit methods if any
        if !s.methods.is_empty() {
            self.blank();
            self.line(&format!("impl{} {} {{", type_params, s.name));
            self.indent += 1;
            for method in &s.methods {
                self.emit_fn(method, Some(&s.name));
                self.blank();
            }
            self.indent -= 1;
            self.line("}");
        }
    }

    /// Emit an enum declaration with all its variants.
    pub(super) fn emit_enum(&mut self, e: &EnumDecl) {
        if e.is_native { return; }

        let vis = if e.is_pub { "pub " } else { "" };

        let type_params = if e.type_params.is_empty() {
            String::new()
        } else {
            format!("<{}>", e.type_params.join(", "))
        };

        self.line(&format!("{}enum {}{} {{", vis, e.name, type_params));
        self.indent += 1;

        for variant in &e.variants {
            if variant.fields.is_empty() {
                self.line(&format!("{},", variant.name));
            } else {
                // Check if fields are named
                let has_names = variant.fields.iter().any(|f| f.name.is_some());
                if has_names {
                    self.line(&format!("{} {{", variant.name));
                    self.indent += 1;
                    for field in &variant.fields {
                        let ty_s = self.emit_type(&field.ty);
                        if let Some(name) = &field.name {
                            self.line(&format!("{}: {},", name, ty_s));
                        } else {
                            self.line(&format!("{},", ty_s));
                        }
                    }
                    self.indent -= 1;
                    self.line("},");
                } else {
                    let fields_s: Vec<String> = variant.fields.iter()
                        .map(|f| self.emit_type(&f.ty))
                        .collect();
                    self.line(&format!("{}({}),", variant.name, fields_s.join(", ")));
                }
            }
        }

        self.indent -= 1;
        self.line("}");

        // Emit enum methods if any
        if !e.methods.is_empty() {
            self.blank();
            self.line(&format!("impl{} {} {{", type_params, e.name));
            self.indent += 1;
            for method in &e.methods {
                self.emit_fn(method, Some(&e.name));
                self.blank();
            }
            self.indent -= 1;
            self.line("}");
        }
    }
}
