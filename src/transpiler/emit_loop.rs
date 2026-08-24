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
    /// Best-effort resolution of an expression's static Boring type, for the
    /// two shapes relevant to for-loop auto-enumerate detection: a plain
    /// local variable/parameter, and a struct field access. Returns `None`
    /// when the type can't be determined from local bookkeeping (e.g. a
    /// method-call result) — callers must treat that as "unknown", not
    /// "definitely not a dict/tuple-array".
    fn resolve_iterable_type(&self, expr: &Expr) -> Option<Type> {
        match &expr.kind {
            // `.or_else(bare_self_field_type)`: a bare identifier naming one of the
            // struct's own fields (implicit `self.field`) isn't in `var_types`/
            // `fn_current_params` at all — without this fallback a `{K=V}` dict field
            // referenced bare was invisible to the dict-shape check below, so a 2-var
            // `for k, v in dict_field:` wrongly fell through to the array auto-enumerate
            // path. See docs/self-field-loop-match-borrow-bug.md (repro 2).
            ExprKind::Var(v) => self.var_types.get(v.as_str())
                .or_else(|| self.fn_current_params.get(v.as_str()))
                .cloned()
                .or_else(|| self.bare_self_field_type(v)),
            ExprKind::Field(inner, field) => {
                let owner_ty = self.resolve_expr_struct_type(inner)?;
                self.struct_fields.get(owner_ty.as_str())?
                    .iter().find(|(fname, _)| fname == field)
                    .map(|(_, fty)| fty.clone())
            }
            _ => None,
        }
    }

    /// Shape check on an expression's own syntax only (no variable lookups):
    /// `Some(true)`/`Some(false)` when the expression is itself one of the
    /// telltale tuple-or-not shapes, `None` when it's some other kind of
    /// expression entirely (a method call, a field access, ...) and the
    /// caller needs to fall back to type-based resolution instead.
    fn literal_shape_is_tuples(&self, e: &Expr) -> Option<bool> {
        match &e.kind {
            // `.enumerate()`/`.zip(...)` already produce tuples.
            ExprKind::MethodCall(_, method, _) if method == "enumerate" || method == "zip" => Some(true),
            // Array literal: tuple-shaped iff its elements are tuple literals,
            // e.g. `[(1, "one"), (2, "two")]` vs. `[10, 20, 30]`.
            ExprKind::Array(elems) => Some(elems.first().is_some_and(|el| matches!(el.kind, ExprKind::Tuple(_)))),
            // Dict literal: `{"a" = 1}` — HashMap iteration already yields (K, V).
            ExprKind::Dict(_) => Some(true),
            _ => None,
        }
    }

    /// True when a `for <a>, <b> in <iterable>:` iterable is already
    /// tuple-shaped (a dict, an array of tuples, or an external generic
    /// container — e.g. a Bevy-ECS-style `Query<(mut Transform&, Sprite&)>`
    /// parameter — whose sole type argument is itself a tuple) — i.e. the
    /// two loop variables should destructure each item directly, per
    /// docs/book.md's "`for` with index — auto-enumerate" rule. False (or
    /// unknown, which callers treat the same as false) means the two-var
    /// form is the auto-enumerate shorthand (`for i, v in arr:` ≡
    /// `for i, v in arr.enumerate():`) and needs an explicit `.enumerate()`
    /// injected, since — unlike the tree-walk interpreter, which decides
    /// per item at runtime — the transpiler has to decide once, statically.
    fn iterable_yields_tuples(&self, iterable: &Expr) -> bool {
        if let Some(shape) = self.literal_shape_is_tuples(iterable) {
            return shape;
        }
        // A local variable: check the dict tracking set (populated even for
        // unannotated `var scores = {...}` bindings — see emit_let.rs), then
        // fall back to its recorded initializer's own literal shape (covers
        // `let pairs = [(1, "one"), ...]` with no type annotation, which
        // `var_types`/`resolve_iterable_type` below can't see at all).
        if let ExprKind::Var(v) = &iterable.kind {
            if self.dict_vars.contains(v.as_str()) { return true; }
            if let Some(shape) = self.var_init_exprs.get(v.as_str())
                .and_then(|init| self.literal_shape_is_tuples(init))
            {
                return shape;
            }
        }
        // Last resort: an explicit type annotation (local var, param, or
        // struct field — the latter is how the kernel-field case, e.g.
        // `for i, v in k.y:` where `y` is a declared `[float32]'unified`, gets
        // proven non-tuple without ever touching `var_types`/`dict_vars`).
        //
        // `tuple_slot_mut_flags` is reused here (not just for `mut`-prefix
        // emission below) as the general "does this type's item shape have a
        // tuple at its core" probe: it already unwraps `Array`/`Optional`/
        // `Mut`/`Qualified`, and — critically — a one-arg `Generic` type
        // (`Query<(mut Transform&, Sprite&)>`), which a plain
        // `Type::Array(Tuple)` match can't see at all. Without this, any
        // external generic whose "tuple" is actually a component-destructuring
        // pattern (not an index/value pair) fell through to `false` and got
        // wrongly `.enumerate()`-wrapped, since `Query`/`Res`-style types have
        // no `.iter()` returning `(usize, Item)` pairs — see the regression
        // this guards, `tests/for_loop_mut_tuple.rs`.
        let resolved = self.resolve_iterable_type(iterable);
        matches!(resolved, Some(Type::Dict(_, _)))
            || resolved.as_ref().is_some_and(|ty| ty.tuple_slot_mut_flags().is_some())
    }

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
                // `is_known_user_type` (not `struct_fields` alone) so iterating a `[SomeEnum]`
                // array (`for w in walls: w.position()`) also gets the loop var registered for
                // method dispatch — matches the struct case exactly, enums just have no fields.
                Some(Type::Named(n)) if self.is_known_user_type(n.as_str()) => {
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
        // `for i, v in arr:` auto-enumerate shorthand (docs/book.md's "`for` with
        // index" rule): two loop vars over an iterable that isn't already
        // tuple-shaped (a dict, or an array of tuples) bind vars[0] = index,
        // vars[1] = element, same as `arr.enumerate()` written out explicitly.
        // The interpreter already implements this dynamically (exec_for); the
        // transpiler has to decide statically since Rust has no such thing as
        // "iterate a Vec<T> as index+value only when T happens not to be a tuple".
        let needs_auto_enumerate = s.vars.len() == 2 && !self.iterable_yields_tuples(&s.iterable);
        // A field only borrowed through `&self`/`&mut self` — explicit `self.field` or
        // the far more common implicit bare `field` spelling — is never safe to move out
        // of directly. See docs/self-field-loop-match-borrow-bug.md: `.into_iter()` moves
        // out of `&self`, and (for a `{K=V}` dict) `.iter().cloned()` doesn't even compile
        // — `Iterator::cloned` only applies when the item is a plain `&T`, not a HashMap's
        // `(&K, &V)` pair — so dict fields get their own owned-pair path below.
        let self_field_ty = self.resolve_self_field_type(&s.iterable).map(|t| t.without_mut().clone());
        let is_borrowed_collection_field = self_field_ty.as_ref()
            .is_some_and(|t| matches!(t, Type::Dict(..) | Type::Array(_) | Type::Set(_)));
        let iter_expr = match &s.iterable.kind {
            ExprKind::Range { .. } => iter,
            _ if is_borrowed_collection_field => match &self_field_ty {
                // Single var over a dict binds the key only (docs/book.md's "`for` over a
                // dict" rule) — `.keys().cloned()` matches that arity directly.
                Some(Type::Dict(_, _)) if s.vars.len() <= 1 => format!("{}.keys().cloned()", iter),
                // 2-var `for k, v in dict_field:` needs owned `(K, V)` pairs; clone the
                // whole map first (mirrors the manual `.clone()` workaround this bug
                // forced from Boring source) rather than `.iter()` + per-slot `.clone()`.
                Some(Type::Dict(_, _)) => format!("{}.clone().into_iter()", iter),
                // Array/Set field: `.iter().cloned()` borrows without consuming, same as
                // the local-variable case just below.
                _ => format!("{}.iter().cloned()", iter),
            },
            // Local variable iteration: use iter().cloned() so the variable is not moved
            // and can be reused after the loop. into_iter() would consume the collection.
            // Exception: multi-var (dict/tuple) iteration needs into_iter() to get owned pairs
            // — except the auto-enumerate case just above, which behaves like the single-var
            // case (the source array itself isn't already tuple-shaped, so it's just as
            // reusable afterward as the plain `for v in arr:` form).
            // Exception: a loop variable tracked as a task/JoinHandle var (the common
            // `for future in futures: future.wait` idiom — `futures` was built by
            // pushing `task_vars`/`join_handle_vars`-tracked values) holds
            // `tokio::task::JoinHandle<T>`, which isn't `Clone`; `.iter().cloned()`
            // is a hard compile error there (E0277). `into_iter()` is always safe for
            // this idiom since the array is never reused after awaiting every handle.
            ExprKind::Var(v) if self.known_local_vars.contains(v.as_str())
                && (s.vars.len() <= 1 || needs_auto_enumerate)
                && !s.vars.first().is_some_and(|lv| {
                    self.task_vars.contains(lv.as_str()) || self.join_handle_vars.contains(lv.as_str())
                }) =>
            {
                format!("{}.iter().cloned()", iter)
            }
            _ => format!("{}.into_iter()", iter),
        };
        // Inject the index: `<base>.enumerate()` yields `(usize, T)`, cast to
        // `isize` to match bare `int` (see CLAUDE.md's scalar-type table) so the
        // bound index composes with ordinary `int` arithmetic in the loop body.
        let iter_expr = if needs_auto_enumerate {
            format!("{}.enumerate().map(|(i, v)| (i as isize, v))", iter_expr)
        } else {
            iter_expr
        };
        // Tuple destructuring: `for k, v in dict:` → `for (k, v) in dict { ... }`.
        // When the iterable's static type is tuple-shaped with `mut`-tagged
        // slots (e.g. a Bevy-ECS-style `Query<(mut Position&, Velocity&)>`
        // parameter), prefix the corresponding loop variable(s) with `mut` —
        // required for the item's `Mut<T>`-style wrapper (or any type whose
        // field-mutation needs a mutable binding to reborrow through) to be
        // usable in the loop body; see docs/mut-type-modifier.md's open
        // "propagating a mut-qualified type argument through generic
        // instantiation" note.
        let pat = if s.vars.len() > 1 {
            let mut_flags: Option<Vec<bool>> = match &s.iterable.kind {
                ExprKind::Var(v) => self.var_types.get(v.as_str())
                    .or_else(|| self.fn_current_params.get(v.as_str()))
                    .and_then(Type::tuple_slot_mut_flags),
                _ => None,
            };
            let parts: Vec<String> = s.vars.iter().enumerate().map(|(i, v)| {
                let name = escape_rust_keyword(v);
                let is_mut = mut_flags.as_ref().and_then(|flags| flags.get(i)).copied().unwrap_or(false);
                if is_mut { format!("mut {}", name) } else { name }
            }).collect();
            format!("({})", parts.join(", "))
        } else {
            vars
        };
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
