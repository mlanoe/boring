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

use super::*;
use super::Transpiler;
use super::helpers::*;

impl Transpiler {
    pub(crate) fn emit_while(&mut self, s: &WhileStmt) {
        let cond = self.emit_expr(&s.condition);
        self.line(&format!("while {} {{", cond));
        self.indent += 1;
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
    }

    pub(crate) fn emit_while_let(&mut self, s: &WhileLetStmt) {
        let val = self.emit_expr(&s.value);
        if let Some(pat) = &s.pattern {
            // `while let Some(x) = expr:` — explicit pattern form
            Self::collect_pattern_binds(pat, &mut self.known_local_vars);
            let pat_s = self.emit_pattern(pat);
            self.line(&format!("while let {} = {} {{", pat_s, val));
        } else {
            // `while let name = expr:` — implicit Some unwrap
            self.known_local_vars.insert(s.name.clone());
            self.line(&format!("while let Some({}) = {} {{", s.name, val));
        }
        self.indent += 1;
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
    }

    pub(crate) fn emit_do_while(&mut self, s: &DoWhileStmt) {
        self.line("loop {");
        self.indent += 1;
        self.emit_loop_body(&s.body);
        let cond = self.emit_expr(&s.condition);
        self.line(&format!("if !({}) {{ break; }}", cond));
        self.indent -= 1;
        self.line("}");
    }

    pub(crate) fn emit_loop(&mut self, s: &LoopStmt) {
        self.line("loop {");
        self.indent += 1;
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
    }

    pub(crate) fn emit_for(&mut self, s: &ForStmt) {
        // Loop variables are scoped to the for body: save outer set first so
        // that after the loop the loop vars are not visible to subsequent code.
        let saved_locals = self.known_local_vars.clone();
        // Track loop variables as known locals so field accesses inside the
        // loop body use `.` not `::`.
        for v in &s.vars {
            self.known_local_vars.insert(v.clone());
        }
        // Track variables that iterate over `.chars()` — they are Rust `char` and need
        // `.to_string()` conversion when used as dict keys (HashMap<Arc<str>, V>).
        if let ExprKind::MethodCall(_, method, _) = &s.iterable.kind {
            if method == "chars" {
                for v in &s.vars {
                    self.chars_vars.insert(v.clone());
                }
            }
        }
        // Track the array element type when iterating over a known `[T]` iterable
        // (`self.blocks`, a local `[T]` var, etc.) so the loop var inside the body resolves
        // correctly: struct elements get owned-arg cloning / qualified throws lookup instead
        // of bare-name heuristics, and primitive elements (e.g. `[int]`) are known non-string
        // so dict-key emission doesn't wrongly treat them as `Arc<str>`.
        if s.vars.len() == 1 {
            let elem_ty: Option<Type> = match &s.iterable.kind {
                ExprKind::Field(inner, field) => self.resolve_expr_struct_type(inner).and_then(|owner_ty| {
                    self.struct_fields.get(owner_ty.as_str())
                        .and_then(|fields| fields.iter().find(|(fname, _)| fname == field))
                        .and_then(|(_, fty)| match fty {
                            Type::Array(elem) => Some(elem.as_ref().clone()),
                            _ => None,
                        })
                }),
                ExprKind::Var(v) => self.var_types.get(v.as_str())
                    .or_else(|| self.fn_current_params.get(v.as_str()))
                    .and_then(|ty| match ty {
                        Type::Array(elem) => Some(elem.as_ref().clone()),
                        _ => None,
                    }),
                _ => None,
            };
            match &elem_ty {
                Some(Type::Named(n)) if self.struct_fields.contains_key(n.as_str()) => {
                    self.var_struct_types.insert(s.vars[0].clone(), n.clone());
                }
                // Primitive element types: `int`/`uint`/`float`/`bool` parse as `Type::Named`
                // (lowercase source syntax), not the bare `Type::Int` etc. builtin variants —
                // match both forms.
                Some(ty @ (Type::Int | Type::Uint | Type::Uint8 | Type::Float32 | Type::Float64 | Type::Bool
                    | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                    | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128)) => {
                    self.var_types.insert(s.vars[0].clone(), ty.clone());
                }
                Some(Type::Named(n)) if matches!(n.as_str(),
                    "int" | "uint" | "uint8" | "float" | "float32" | "float64" | "bool"
                    | "int8" | "int16" | "int32" | "int64" | "int128"
                    | "uint16" | "uint32" | "uint64" | "uint128") => {
                    let canonical = match n.as_str() {
                        "int" => Type::Int,
                        "uint" => Type::Uint,
                        "uint8" => Type::Uint8,
                        "int8" => Type::Int8,
                        "int16" => Type::Int16,
                        "int32" => Type::Int32,
                        "int64" => Type::Int64,
                        "int128" => Type::Int128,
                        "uint16" => Type::Uint16,
                        "uint32" => Type::Uint32,
                        "uint64" => Type::Uint64,
                        "uint128" => Type::Uint128,
                        "float32" => Type::Float32,
                        "float" | "float64" => Type::Float64,
                        _ => Type::Bool,
                    };
                    self.var_types.insert(s.vars[0].clone(), canonical);
                }
                _ => {}
            }
        }
        // Track loop variables from Vec<Arc<str>> iterables so string methods dispatch correctly.
        let iterable_is_str_vec = match &s.iterable.kind {
            ExprKind::Var(v) => self.str_vec_vars.contains(v.as_str()),
            ExprKind::MethodCall(_, m, _) => m == "split",
            _ => false,
        };
        if iterable_is_str_vec {
            for v in &s.vars {
                self.string_arc_vars.insert(v.clone());
                self.string_vars.insert(v.clone());
            }
        }

        // Detect `for item in stream_fn(args):` — iterator or async stream consumer.
        let stream_fn_name: Option<(String, bool)> = match &s.iterable.kind {
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(name) = &callee.kind {
                    if self.stream_iter_fns.contains(name.as_str()) {
                        Some((name.clone(), true))
                    } else if self.stream_fns.contains(name.as_str()) {
                        Some((name.clone(), false))
                    } else { None }
                } else { None }
            }
            _ => None,
        };
        if let Some((ref fn_name, is_iter)) = stream_fn_name {
            if is_iter {
                return self.emit_for_iter_stream(s);
            }
            return self.emit_for_stream(s, fn_name);
        }

        // Detect `for item in rx:` — channel receiver iteration.
        if let ExprKind::Var(rx_name) = &s.iterable.kind {
            if self.channel_receivers.contains(rx_name.as_str()) {
                return self.emit_for_channel(s, rx_name.clone());
            }
            // broadcast: while let Ok(msg) = rx.recv().await { body }
            if self.broadcast_receivers.contains(rx_name.as_str()) {
                return self.emit_for_broadcast(s, rx_name.clone());
            }
            // watch: while rx.changed().await.is_ok() { let msg = rx.borrow().clone(); body }
            if self.watch_receivers.contains(rx_name.as_str()) {
                return self.emit_for_watch(s, rx_name.clone());
            }
        }

        // Detect `for item in actor_var:` — actor (Arc<Mutex<Vec<T>>>) iteration.
        // Lock the mutex, iterate with `.iter().cloned()` to keep items as owned values.
        if let ExprKind::Var(actor_name) = &s.iterable.kind {
            let is_multi = matches!(self.config.threading, crate::transpiler::ThreadingMode::Multi);
            if (self.var_mutex_types.contains(actor_name.as_str()) || self.var_mutex_task_types.contains(actor_name.as_str()))
                && (self.in_async || is_multi)
            {
                let vars = if s.vars.is_empty() {
                    "_".into()
                } else {
                    s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
                };
                let guard = format!("__guard_{}", actor_name);
                let lock_expr = self.mutex_var_write(actor_name, actor_name);
                self.line(&format!("let {} = {};", guard, lock_expr));
                self.line(&format!("for {} in {}.iter().cloned() {{", vars, guard));
                self.indent += 1;
                self.emit_loop_body(&s.body);
                self.indent -= 1;
                self.line("}");
                self.known_local_vars = saved_locals;
                return;
            }
        }
        // Detect `for item in guard_var:` — guard (Arc<RwLock<Vec<T>>>) iteration.
        // Acquire a read lock, iterate with `.iter().cloned()`.
        if let ExprKind::Var(guard_name) = &s.iterable.kind {
            let is_multi = matches!(self.config.threading, crate::transpiler::ThreadingMode::Multi);
            if self.var_rwlock_types.contains(guard_name.as_str())
                && (self.in_async || is_multi)
            {
                let vars = if s.vars.is_empty() {
                    "_".into()
                } else {
                    s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
                };
                let rguard = format!("__rguard_{}", guard_name);
                let read_expr = if self.use_async_actors() && self.in_async {
                    format!("{}.read().await", guard_name)
                } else {
                    format!("{}.read().unwrap()", guard_name)
                };
                self.line(&format!("let {} = {};", rguard, read_expr));
                self.line(&format!("for {} in {}.iter().cloned() {{", vars, rguard));
                self.indent += 1;
                self.emit_loop_body(&s.body);
                self.indent -= 1;
                self.line("}");
                self.known_local_vars = saved_locals;
                return;
            }
        }

        // Custom iterator protocol: struct with `def T? next():`.
        // `for x in obj:` → `{ let mut __iter = obj; while let Some(x) = __iter.next() { body } }`
        // Detect from var_struct_types (variable with known struct type) or from the call return type.
        let iterable_struct_type: Option<String> = match &s.iterable.kind {
            ExprKind::Var(v) => self.var_struct_types.get(v.as_str()).cloned(),
            ExprKind::Call(callee, _) => match &callee.kind {
                ExprKind::Var(fn_name) => self.fn_return_types.get(fn_name.as_str())
                    .and_then(|t| if let Type::Named(n) = t { Some(n.clone()) } else { None }),
                _ => None,
            },
            _ => None,
        };
        if let Some(ref struct_ty) = iterable_struct_type {
            if self.iterable_structs.contains(struct_ty.as_str()) {
                let iter_s = self.emit_expr(&s.iterable);
                let vars = if s.vars.is_empty() {
                    "_".into()
                } else {
                    s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
                };
                let pat = if s.vars.len() == 1 { vars.clone() } else { format!("({})", vars) };
                self.line(&format!("let mut __iter = {};", iter_s));
                self.line(&format!("while let Some({}) = __iter.next() {{", pat));
                self.indent += 1;
                self.emit_loop_body(&s.body);
                self.indent -= 1;
                self.line("}");
                self.known_local_vars = saved_locals;
                return;
            }
        }

        let iter = self.emit_expr(&s.iterable);
        // Escape Rust keywords that might be used as loop variables (e.g. `fn`).
        let vars = if s.vars.is_empty() {
            "_".into()
        } else {
            s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
        };
        // Range expressions iterate directly.
        // `self.field` (struct field) uses `.iter().cloned()` to avoid moving out of self.
        // All other expressions (local vars, method call results) use `.into_iter()` —
        // this handles non-Clone types like JoinHandle and is safe for owned collections.
        // For `for x in actor_var.field:` or `for k, v in actor_var.field:`,
        // the field access moves out of the MutexGuard. Clone the field first.
        if let ExprKind::Field(obj, field_name) = &s.iterable.kind {
            if let ExprKind::Var(v) = &obj.kind {
                if self.var_mutex_types.contains(v.as_str()) || self.managed_mutex_vars.contains(v.as_str()) {
                    let access = self.mutex_var_read(v, v);
                    let cloned_field = format!("{}.{}.clone()", access, field_name);
                    let tmp = format!("__iter_{}", v);
                    self.line(&format!("let {} = {};", tmp, cloned_field));
                    let pat = if s.vars.len() > 1 { format!("({})", vars) } else { vars };
                    self.line(&format!("for {} in {}.into_iter() {{", pat, tmp));
                    self.indent += 1;
                    self.emit_loop_body(&s.body);
                    self.indent -= 1;
                    self.line("}");
                    self.known_local_vars = saved_locals;
                    return;
                }
            }
        }
        let iter_expr = match &s.iterable.kind {
            ExprKind::Range { .. } => iter,
            ExprKind::Field(obj, _) if matches!(&obj.kind, ExprKind::Var(v) if v == "self") => {
                format!("{}.iter().cloned()", iter)
            }
            // Local variable iteration: use iter().cloned() so the variable is not moved
            // and can be reused after the loop. into_iter() would consume the collection.
            // Exception: multi-var (dict/tuple) iteration needs into_iter() to get owned pairs.
            ExprKind::Var(v) if self.known_local_vars.contains(v.as_str()) && s.vars.len() <= 1 => {
                format!("{}.iter().cloned()", iter)
            }
            _ => format!("{}.into_iter()", iter),
        };
        // Tuple destructuring: `for k, v in dict:` → `for (k, v) in dict { ... }`
        let pat = if s.vars.len() > 1 { format!("({})", vars) } else { vars };
        self.line(&format!("for {} in {} {{", pat, iter_expr));
        self.indent += 1;
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
        // Restore: loop vars and any let-bindings inside the body are not
        // visible after the for loop (emit_loop_body already saved/restored
        // its own inner lets; here we restore the loop var additions).
        self.known_local_vars = saved_locals;
    }

    /// Emit a `for item in iter_stream_fn(args):` as a plain Rust `for` loop.
    /// The callee returns `impl Iterator<Item = T>` — no `.await`, no pinning.
    pub(crate) fn emit_for_iter_stream(&mut self, s: &ForStmt) {
        let iter_expr = self.emit_expr(&s.iterable);
        let vars = if s.vars.is_empty() {
            "_".into()
        } else {
            s.vars.iter().map(|v| crate::transpiler::helpers::escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
        };
        self.line(&format!("for {} in {} {{", vars, iter_expr));
        self.indent += 1;
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
    }

    /// Emit a `for item in stream_fn(args):` loop as a pinned stream consumer.
    ///
    /// ```rust
    /// {
    ///     use tokio_stream::StreamExt;
    ///     let mut __stream_N = std::pin::pin!(stream_fn(args));
    ///     while let Some(item) = __stream_N.next().await { body }
    /// }
    /// ```
    /// For `throws` streams the item is `Result<T, E>`; we unwrap with `?`.
    pub(crate) fn emit_for_stream(&mut self, s: &ForStmt, fn_name: &str) {
        let stream_expr = self.emit_expr(&s.iterable);
        let throws = self.stream_throws_fns.contains(fn_name);
        let vars = if s.vars.is_empty() {
            "_".into()
        } else {
            s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
        };
        // Unique stream var name to avoid shadowing in nested stream loops
        let stream_var = format!("__stream_{}", s.iterable.line);

        self.line("{");
        self.indent += 1;
        self.line("use tokio_stream::StreamExt;");
        self.line(&format!("let mut {} = std::pin::pin!({});", stream_var, stream_expr));
        if throws {
            self.line(&format!("while let Some(__res) = {}.next().await {{", stream_var));
            self.indent += 1;
            self.line(&format!("let {} = __res?;", vars));
        } else {
            self.line(&format!("while let Some({}) = {}.next().await {{", vars, stream_var));
            self.indent += 1;
        }
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// Emit `for item in rx:` as a `while let Some(item) = rx.recv().await { body }`.
    pub(crate) fn emit_for_channel(&mut self, s: &ForStmt, rx_name: String) {
        let vars = if s.vars.is_empty() {
            "_".into()
        } else {
            s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
        };
        self.line(&format!("while let Some({}) = {}.recv().await {{", vars, rx_name));
        self.indent += 1;
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
    }

    pub(crate) fn emit_for_broadcast(&mut self, s: &ForStmt, rx_name: String) {
        let vars = if s.vars.is_empty() {
            "_".into()
        } else {
            s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
        };
        // In single-thread mode, LocalBroadcastReceiver::recv() returns T (no Result).
        if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
            self.line("loop {");
            self.indent += 1;
            self.line(&format!("let {} = {}.recv().await;", vars, rx_name));
            self.emit_loop_body(&s.body);
            self.indent -= 1;
            self.line("}");
        } else {
            self.line(&format!("while let Ok({}) = {}.recv().await {{", vars, rx_name));
            self.indent += 1;
            self.emit_loop_body(&s.body);
            self.indent -= 1;
            self.line("}");
        }
    }

    pub(crate) fn emit_for_watch(&mut self, s: &ForStmt, rx_name: String) {
        let vars = if s.vars.is_empty() {
            "_".into()
        } else {
            s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
        };
        self.line(&format!("while {}.changed().await.is_ok() {{", rx_name));
        self.indent += 1;
        self.line(&format!("let {} = {}.borrow().clone();", vars, rx_name));
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
    }

}
