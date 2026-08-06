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
    /// `T'actor` → `Arc<Mutex<T>>`/`Rc<RefCell<T>>`, `T'guard` → `Arc<RwLock<T>>`, and
    /// managed-mode `T'` (`OwnerQual::Owned`) over a user type — locking wrapper bindings
    /// that need their own constructor call and lock-guard shadow, not a plain `let`.
    /// Returns `true` if it fully emitted the binding (caller must return immediately).
    fn try_emit_qualified_let(&mut self, s: &LetStmt, ty: &Type, s_value: &Expr) -> bool {
        // T'actor → Arc<Mutex<T>> (multi) or Rc<RefCell<T>> (single).
        // All field reads/writes and method calls on this variable will go through the lock/borrow.
        // Works with both `let` and `var` — the actor qualifier alone triggers mutex semantics.
        if Self::is_mutex_binding(s.binding.is_mutable(), ty) {
            if let Some(inner) = Self::mutex_inner(ty) {
                let is_task = Self::is_mutex_task_binding(s.binding.is_mutable(), ty);
                let mutex_ty = if is_task { self.emit_actor_task_type(inner) } else { self.emit_actor_type(inner) };
                let raw_val = self.emit_let_value(Some(inner), s_value);
                let init = if is_task { self.emit_actor_task_new(&raw_val) } else { self.emit_actor_new(&raw_val) };
                if is_task {
                    self.var_mutex_task_types.insert(s.name.clone());
                } else {
                    self.var_mutex_types.insert(s.name.clone());
                }
                self.arc_vars.insert(s.name.clone());
                if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                    self.rc_vars.insert(s.name.clone());
                }
                if let ExprKind::Call(callee, _) = &s_value.kind {
                    if let ExprKind::Var(type_name) = &callee.kind {
                        if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                            && self.struct_fields.contains_key(type_name.as_str())
                        {
                            self.var_struct_types.insert(s.name.clone(), type_name.clone());
                        }
                    }
                }
                let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
                self.line(&format!("{} {}: {} = {};", kw, s.name, mutex_ty, init));
                return true;
            }
        }
        // T'guard / T'guard'task → Arc<RwLock<T>> (multi) or Rc<RefCell<T>> (single).
        if Self::is_rwlock_binding(s.binding.is_mutable(), ty) {
            if let Some(inner) = Self::rwlock_inner(ty) {
                let is_task = Self::is_rwlock_task_binding(s.binding.is_mutable(), ty);
                let rwlock_ty = if is_task { self.emit_guard_task_type(inner) } else { self.emit_guard_type(inner) };
                let raw_val = self.emit_let_value(Some(inner), s_value);
                let init = if is_task {
                    self.emit_guard_task_new(&raw_val)
                } else {
                    self.emit_guard_new(&raw_val)
                };
                if is_task {
                    self.var_rwlock_task_types.insert(s.name.clone());
                } else {
                    self.var_rwlock_types.insert(s.name.clone());
                }
                self.arc_vars.insert(s.name.clone());
                if let ExprKind::Call(callee, _) = &s_value.kind {
                    if let ExprKind::Var(type_name) = &callee.kind {
                        if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                            && self.struct_fields.contains_key(type_name.as_str())
                        {
                            self.var_struct_types.insert(s.name.clone(), type_name.clone());
                        }
                    }
                }
                let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
                self.line(&format!("{} {}: {} = {};", kw, s.name, rwlock_ty, init));
                return true;
            }
        }
        // Managed mode T' (OwnerQual::Owned) over a user type:
        // multi → Arc<std::sync::Mutex<T>>, single → RefCell<T>.
        // Track the variable so field/method access emits correct locking.
        if self.is_managed_owned_user(ty) {
            if let Type::Qualified(inner, OwnerQual::Owned) = ty.without_mut() {
                let managed_ty = self.emit_managed_actor(inner);
                let raw_val = self.emit_let_value(Some(inner.as_ref()), s_value);
                let init = self.wrap_managed(&raw_val);
                let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
                match self.config.threading {
                    crate::transpiler::ThreadingMode::Multi => {
                        self.managed_mutex_vars.insert(s.name.clone());
                        self.arc_vars.insert(s.name.clone());
                    }
                    crate::transpiler::ThreadingMode::Single => {
                        self.managed_refcell_vars.insert(s.name.clone());
                    }
                }
                self.line(&format!("{} {}: {} = {};", kw, s.name, managed_ty, init));
                // Emit a lock guard so multi-field accesses in a single expression
                // don't deadlock (two separate .lock().unwrap() on the same Mutex).
                if matches!(self.config.threading, crate::transpiler::ThreadingMode::Multi) {
                    let shadow = format!("__{}_mg", s.name);
                    self.line(&format!("let mut {} = {}.lock().unwrap();", shadow, s.name));
                    self.managed_mutex_vars.remove(&s.name);
                    self.managed_param_shadows.insert(s.name.clone(), shadow);
                }
                return true;
            }
        }
        false
    }

    /// Computes the emitted type annotation (e.g. `": Foo"`, possibly empty) and value
    /// expression string for a `let`/`var` binding's RHS, plus whether it's a mutable
    /// string binding (literal or `string`-typed) — those need `Arc<str>`/`Rc<str>`, not
    /// `&str`, so they can be reassigned. Read-only: emits nothing, mutates nothing.
    fn compute_let_ty_and_value(&self, s: &LetStmt, s_value: &Expr) -> (String, String, bool, bool) {
        // Mutable string bindings must be Arc<str> (not &str) so they can be reassigned
        let is_mutable_string_lit = s.binding.is_mutable() && s.ty.is_none()
            && matches!(&s_value.kind, ExprKind::Str(_) | ExprKind::StringInterp(_));
        let is_mutable_string_ty = s.binding.is_mutable()
            && matches!(&s.ty, Some(Type::Named(n)) if n == "string" || n == "str")
            && matches!(&s_value.kind, ExprKind::Str(_) | ExprKind::StringInterp(_));
        if is_mutable_string_lit || is_mutable_string_ty {
            let str_ty_annotation = match self.config.threading {
                crate::transpiler::ThreadingMode::Single => ": Rc<str>",
                crate::transpiler::ThreadingMode::Multi  => ": Arc<str>",
            };
            return (str_ty_annotation.to_string(), self.emit_expr_owned(s_value), is_mutable_string_lit, is_mutable_string_ty);
        }
        let val = self.emit_let_value(s.ty.as_ref(), s_value);
        // Auto-clone: field accesses can't be moved out of a struct in Rust.
        // When the RHS of a let is a field access and the type is non-Copy, add .clone()
        // unless emit_let_value already produced a fresh owned value.
        let val = if matches!(&s_value.kind, ExprKind::Field(..))
            && !val.ends_with(".clone()")
            && !val.starts_with('&')
            && !val.starts_with("Arc::")
            && !val.starts_with("Rc::")
            && !val.starts_with("{ let __g")
            && !matches!(s.ty.as_ref(), Some(Type::Int | Type::Uint | Type::Uint8 | Type::Float | Type::Bool
                | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128))
        {
            format!("{}.clone()", val)
        } else {
            val
        };
        // Inferred T'weak binding (bare `d'weak`, no compound qualifier): if the value
        // is Arc::downgrade(...), the annotation must be std::sync::Weak (not rc::Weak).
        // Compound forms like `Resource'task'weak` are handled correctly by emit_type.
        let ty = if let Some(ty) = s.ty.as_ref() {
            let is_bare_weak = matches!(ty,
                Type::Qualified(inner, OwnerQual::Weak)
                if !matches!(inner.as_ref(), Type::Qualified(_, _)));
            if is_bare_weak && val.starts_with("Arc::downgrade(") {
                ": std::sync::Weak<_>".to_string()
            } else {
                format!(": {}", self.emit_type(ty))
            }
        } else if matches!(&s_value.kind, ExprKind::Nil) {
            // `let x = nil` — Rust can't infer the type of `None`; add `Option<()>`.
            ": Option<()>".to_string()
        } else if val == "None" {
            // Cast that produces None (e.g. `42 as bool`) — add type annotation.
            ": Option<()>".to_string()
        } else if let Some(inferred_qual) = self.inferred_qualifiers.get(&s.name).cloned() {
            // Priority 5: use-site qualifier inference — apply the inferred qualifier.
            // Handles bare T, T', T?, and T'? initialisers.
            let type_name_opt = match &s_value.kind {
                // some(Counter(0)) — must come before the generic Call arm
                ExprKind::Call(callee, args)
                    if matches!(&callee.kind, ExprKind::Var(n) if n.as_str() == "some") =>
                {
                    if let Some(arg) = args.first() {
                        if let ExprKind::Call(inner, _) = &arg.value.kind {
                            if let ExprKind::Var(n) = &inner.kind {
                                if n.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                                    Some((n.clone(), true))
                                } else { None }
                            } else { None }
                        } else { None }
                    } else { None }
                }
                // Counter(0)
                ExprKind::Call(callee, _) => {
                    if let ExprKind::Var(n) = &callee.kind {
                        if n.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                            Some((n.clone(), false))
                        } else { None }
                    } else { None }
                }
                _ => None,
            };
            if let Some((type_name, is_optional)) = type_name_opt {
                let base_ty = crate::ast::Type::Named(type_name);
                let declared_ty = if is_optional || matches!(&s.ty, Some(crate::ast::Type::Optional(_))) {
                    crate::ast::Type::Optional(Box::new(base_ty))
                } else {
                    base_ty
                };
                let qualified_ty = crate::transpiler::infer_qualifiers::apply_inferred_qual(
                    &declared_ty, inferred_qual,
                );
                format!(": {}", self.emit_type(&qualified_ty))
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        (ty, val, is_mutable_string_lit, is_mutable_string_ty)
    }

    /// Classifies a `let`/`var` binding after its value/type-annotation strings have been
    /// computed: string/collection/tuple/index/task/enum/newtype/Arc/managed-mode tracking
    /// sets that later field access, method dispatch, and cloning decisions all key off of.
    /// Pure bookkeeping — with one exception: a cancellable spawned task fn emits its own
    /// `let` (cancel token + spawn) and returns `true`, telling the caller to return early
    /// instead of falling through to `emit_let`'s normal final-emission code.
    fn track_let_metadata(&mut self, s: &LetStmt, s_value: &Expr, val: &str, is_mutable_string_lit: bool, is_mutable_string_ty: bool) -> bool {
        // Track mutable Arc<str> vars for read_line / clear() special-casing
        if is_mutable_string_lit || is_mutable_string_ty {
            self.string_arc_vars.insert(s.name.clone());
            self.string_vars.insert(s.name.clone());
            // A redeclaration of a var as string overrides any prior array/collection tracking.
            self.vec_vars.remove(s.name.as_str());
            self.collection_vars.remove(s.name.as_str());
        }
        // Also track immutable string literal vars so string methods (parseInt, indexOf, slice…)
        // can dispatch correctly even without an explicit type annotation.
        let is_immutable_string_lit = !s.binding.is_mutable() && s.ty.is_none()
            && matches!(&s_value.kind, ExprKind::Str(_) | ExprKind::StringInterp(_));
        // readLine() returns Option<Arc<str>> — don't track as plain string var (it's optional).
        let is_readline_call = false;
        if is_immutable_string_lit || is_readline_call {
            self.string_vars.insert(s.name.clone());
            self.vec_vars.remove(s.name.as_str());
            self.collection_vars.remove(s.name.as_str());
        }
        // Track variables that hold collections (for {:?} formatting later)
        if looks_like_collection(val) || is_collection_type(s.ty.as_ref()) {
            self.collection_vars.insert(s.name.clone());
        }
        // Track variables that unambiguously hold a Vec<T> (not HashMap/HashSet, not scalars from reduce).
        // Only consider expressions that END as a Vec — this excludes reduce/fold chains that
        // contain intermediate .collect::<Vec<_>>() but terminate as a scalar.
        if (expr_ends_as_vec(val) && !looks_like_map_or_set(val))
            || matches!(&s.ty, Some(Type::Array(_)))
        {
            self.vec_vars.insert(s.name.clone());
        }
        // Track Vec<Arc<str>> variables: assigned from split/chars or declared as [string].
        let is_str_array_ty = matches!(&s.ty, Some(Type::Array(inner))
            if matches!(inner.as_ref(), Type::Str)
            || matches!(inner.as_ref(), Type::Named(n) if n == "string" || n == "str"));
        let is_split_or_chars = matches!(&s_value.kind,
            ExprKind::MethodCall(_, m, _) if m == "split" || m == "chars");
        if is_str_array_ty || is_split_or_chars {
            self.str_vec_vars.insert(s.name.clone());
        }
        // Track HashSet variables for `remove(&v)` and `add`→`insert` dispatch.
        if matches!(&s.ty, Some(Type::Set(_)))
            || val.starts_with("HashSet::")
            || (val.starts_with("HashSet::from(") || val.contains(".collect::<HashSet"))
        {
            self.set_vars.insert(s.name.clone());
        }
        // Track HashMap/dict variables for `.get()`/`.insert()` subscript dispatch.
        if matches!(&s.ty, Some(Type::Dict(..)))
            || val.starts_with("HashMap::")
            || val.contains(".collect::<HashMap")
        {
            self.dict_vars.insert(s.name.clone());
        }
        // Managed mode inference: if no explicit type annotation and the expression result
        // is inferred to be a managed-mode wrapped type (Arc<Mutex<T>> or RefCell<T>),
        // track the variable for correct field/method call-site transforms.
        if s.ty.is_none() && self.infers_as_managed(s_value) {
            match self.config.threading {
                crate::transpiler::ThreadingMode::Multi => {
                    self.managed_mutex_vars.insert(s.name.clone());
                    self.arc_vars.insert(s.name.clone());
                }
                crate::transpiler::ThreadingMode::Single => {
                    self.managed_refcell_vars.insert(s.name.clone());
                }
            }
        }
        // Track tuple variables for method dispatch (length, isEmpty, first, last).
        if let ExprKind::Tuple(elems) = &s_value.kind {
            self.tuple_vars.insert(s.name.clone(), elems.len());
        } else if matches!(&s.ty, Some(Type::Tuple(elems)) if !elems.is_empty()) {
            if let Some(Type::Tuple(elems)) = &s.ty {
                self.tuple_vars.insert(s.name.clone(), elems.len());
            }
        }
        // Track variables that hold an opaque collection index (from firstIndex/nextIndex).
        if matches!(&s_value.kind,
            ExprKind::MethodCall(_, m, _) if m == "firstIndex" || m == "nextIndex")
        {
            self.index_vars.insert(s.name.clone());
        }
        // Track variables that hold a std::time::Instant (for sleep_until/timeout_at dispatch).
        if expr_is_instant(s_value, &self.instant_vars.clone()) {
            self.instant_vars.insert(s.name.clone());
        }
        // task(dur): body — always a throws JoinHandle (timeout fires → Elapsed error via ?)
        if let ExprKind::TaskWithTimeout(..) = &s_value.kind {
            self.task_vars.insert(s.name.clone());
            self.join_handle_vars.insert(s.name.clone());
            self.throws_join_handle_vars.insert(s.name.clone());
        }
        // Track variables that hold a spawned future (task expr) — .value → .await.unwrap()
        if let ExprKind::Task(inner) = &s_value.kind {
            self.task_vars.insert(s.name.clone());
            self.join_handle_vars.insert(s.name.clone());
            // If the spawned function is `throws`, the JoinHandle wraps Result<T, BoringError>.
            // Track these separately so `.value` / `.wait` emit the correct double-unwrap.
            let spawned_fn_throws = match &inner.kind {
                ExprKind::Call(callee, _) => match &callee.kind {
                    ExprKind::Var(fn_name) => self.fn_throws.contains(fn_name.as_str()),
                    _ => false,
                },
                ExprKind::MethodCall(_, method, _) => self.fn_throws.contains(method.as_str()),
                _ => false,
            };
            if spawned_fn_throws {
                self.throws_join_handle_vars.insert(s.name.clone());
            }
            // If spawning a cancellable task fn, emit the cancel token before the binding.
            if let ExprKind::Call(callee, call_args) = &inner.kind {
                if let ExprKind::Var(fn_name) = &callee.kind {
                    if self.cancellable_task_fns.contains(fn_name.as_str()) {
                        self.uses_tokio_util.set(true);
                        let cancel_var = format!("__cancel_{}", s.name);
                        self.cancel_token_vars.insert(s.name.clone(), cancel_var.clone());
                        // Emit: let __cancel_NAME = tokio_util::sync::CancellationToken::new();
                        self.line(&format!(
                            "let {} = tokio_util::sync::CancellationToken::new();",
                            cancel_var
                        ));
                        // Emit the Arc captures for any arc vars
                        let captured = collect_var_names(inner);
                        let arc_captures: Vec<String> = captured.iter()
                            .filter(|v| self.arc_vars.contains(*v))
                            .cloned()
                            .collect();
                        // Build call args with cancel token cloned first
                        let args_s: Vec<String> = call_args.iter().map(|a| self.emit_expr(&a.value)).collect();
                        let all_args = if args_s.is_empty() {
                            format!("{}.clone()", cancel_var)
                        } else {
                            format!("{}.clone(), {}", cancel_var, args_s.join(", "))
                        };
                        let call_s = format!("{fn_name}({all_args}).await");
                        let inner_s = format!("{{ {} }}", call_s);
                        let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
                        let spawn_fn = match self.config.threading {
                            crate::transpiler::ThreadingMode::Single => "tokio::task::spawn_local",
                            crate::transpiler::ThreadingMode::Multi  => "tokio::spawn",
                        };
                        let spawn_s = if arc_captures.is_empty() {
                            format!("{}(async move {})", spawn_fn, inner_s)
                        } else {
                            let clones: String = arc_captures.iter()
                                .map(|v| {
                                    if self.rc_vars.contains(v.as_str()) {
                                        format!("let {} = Rc::clone(&{});", v, v)
                                    } else {
                                        format!("let {} = Arc::clone(&{});", v, v)
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join(" ");
                            format!("{}({{ {} async move {} }})", spawn_fn, clones, inner_s)
                        };
                        self.line(&format!("{} {} = {};", kw, s.name, spawn_s));
                        return true;
                    }
                }
            }
        }
        // Track variables bound to user struct constructors for getter dispatch on non-self receivers.
        // Also handle type method calls: `let c2 = Counter2.zero()` → c2 is Counter2.
        if let ExprKind::MethodCall(callee_obj, _, _) = &s_value.kind {
            if let ExprKind::Var(type_name) = &callee_obj.kind {
                if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && self.struct_fields.contains_key(type_name.as_str()) {
                        self.var_struct_types.insert(s.name.clone(), type_name.clone());
                    }
            }
        }
        if let ExprKind::Call(callee, _) = &s_value.kind {
            if let ExprKind::Var(type_name) = &callee.kind {
                if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    if self.struct_fields.contains_key(type_name.as_str()) {
                        self.var_struct_types.insert(s.name.clone(), type_name.clone());
                    }
                    // Track newtype vars: `let id = UserId(42)` → id is a UserId.
                    if self.newtype_types.contains(type_name.as_str()) {
                        self.var_newtype_type.insert(s.name.clone(), type_name.clone());
                    }
                }
                // If the callee is a function with an Optional return type, mark the var as optional.
                if s.ty.is_none() {
                    if let Some(ret_ty) = self.fn_return_types.get(type_name.as_str()).cloned() {
                        match &ret_ty {
                            Type::Optional(_) => { self.optional_vars.insert(s.name.clone()); }
                            // Track function calls returning a named struct type so field access
                            // Optional detection works (prevents double-wrapping in struct literals).
                            Type::Named(n) if self.struct_fields.contains_key(n.as_str()) => {
                                self.var_struct_types.insert(s.name.clone(), n.clone());
                            }
                            // Track all Named return types (including enums) in var_types so
                            // auto-clone can detect non-Copy variables at call sites.
                            Type::Named(_) | Type::Array(_) | Type::Dict(..) | Type::Set(_) => {
                                self.var_types.insert(s.name.clone(), ret_ty.clone());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        // Track element type for `let x = arr[i]` when arr has a known Array type.
        // e.g. `let key = args[0]` where `args: [Value]` → var_types["key"] = Named("Value").
        if s.ty.is_none() {
            if let ExprKind::Index(arr_expr, _) = &s_value.kind {
                if let ExprKind::Var(arr_name) = &arr_expr.kind {
                    let elem_ty = self.fn_current_params.get(arr_name.as_str())
                        .or_else(|| self.var_types.get(arr_name.as_str()))
                        .and_then(|t| if let Type::Array(elem) = t { Some(elem.as_ref().clone()) } else { None });
                    if let Some(elem_ty) = elem_ty {
                        self.var_types.insert(s.name.clone(), elem_ty);
                    }
                }
            }
        }
        // When value is nil (None), the var is always optional.
        if matches!(&s_value.kind, ExprKind::Nil) {
            self.optional_vars.insert(s.name.clone());
        }
        // Propagate optional through `.clone()` — `let x = opt_var.clone()` keeps x optional.
        if s.ty.is_none() {
            if let ExprKind::MethodCall(recv, method, _) = &s_value.kind {
                if method == "clone" {
                    if let ExprKind::Var(src) = &recv.kind {
                        if self.optional_vars.contains(src.as_str()) {
                            self.optional_vars.insert(s.name.clone());
                        }
                    }
                }
            }
        }
        // If-expression or match-expression with nil/some branches already produces Option<T>.
        if s.ty.is_none() {
            fn body_ends_optional_sv(body: &[Stmt], fn_return_types: &std::collections::HashMap<String, crate::ast::Type>) -> bool {
                match body.last() {
                    Some(Stmt::Expr(e)) => matches!(&e.kind, ExprKind::Nil)
                        || matches!(&e.kind, ExprKind::Call(callee, _)
                            if matches!(&callee.kind, ExprKind::Var(v) if v == "some"))
                        // Call to a function returning Optional
                        || matches!(&e.kind, ExprKind::Call(callee, _)
                            if matches!(&callee.kind, ExprKind::Var(fn_name)
                                if fn_return_types.get(fn_name.as_str())
                                    .map(|t| matches!(t, crate::ast::Type::Optional(_))).unwrap_or(false))),
                    _ => false,
                }
            }
            if let ExprKind::If(if_stmt) = &s_value.kind {
                let is_opt = if_stmt.branches.iter().any(|(_, b)| body_ends_optional_sv(b, &self.fn_return_types))
                    || if_stmt.else_body.as_ref().map(|b| body_ends_optional_sv(b, &self.fn_return_types)).unwrap_or(false);
                if is_opt { self.optional_vars.insert(s.name.clone()); }
            }
            // Match-expression with a nil arm also produces Option<T>.
            if let ExprKind::Match(match_stmt) = &s_value.kind {
                let is_opt = match_stmt.arms.iter().any(|arm| {
                    match &arm.body {
                        crate::ast::MatchBody::Block(stmts) => body_ends_optional_sv(stmts, &self.fn_return_types),
                        crate::ast::MatchBody::Expr(e) => matches!(&e.kind, ExprKind::Nil),
                    }
                });
                if is_opt { self.optional_vars.insert(s.name.clone()); }
            }
        }
        // Optional chaining produces Option<T> — mark the variable as optional.
        if s.ty.is_none() && matches!(&s_value.kind,
            ExprKind::OptionalField(..) | ExprKind::OptionalMethodCall(..))
        {
            self.optional_vars.insert(s.name.clone());
        }
        // When value is a string-to-numeric cast (returns Option<T> with .ok()), mark as optional.
        // Also mark int/float-to-bool as optional (always returns None in Boring).
        if s.ty.is_none() {
            if let ExprKind::Cast(src_expr, dst_ty) = &s_value.kind {
                let src_is_str = matches!(&src_expr.kind, ExprKind::Str(_) | ExprKind::StringInterp(_))
                    || matches!(&src_expr.kind, ExprKind::Var(v) if self.string_vars.contains(v.as_str()));
                let dst_is_numeric = matches!(dst_ty, Type::Int | Type::Uint | Type::Uint8 | Type::Float
                        | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                        | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128)
                    || matches!(dst_ty, Type::Named(n) if matches!(n.as_str(),
                        "int" | "uint" | "uint8" | "float"
                        | "int8" | "int16" | "int32" | "int64" | "int128"
                        | "uint16" | "uint32" | "uint64" | "uint128"
                        | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
                        | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"));
                let src_is_numeric = matches!(&src_expr.kind, ExprKind::Int(_) | ExprKind::Float(_));
                let dst_is_bool = matches!(dst_ty, Type::Bool)
                    || matches!(dst_ty, Type::Named(n) if n == "bool");
                if (src_is_str && dst_is_numeric) || (src_is_numeric && dst_is_bool) {
                    self.optional_vars.insert(s.name.clone());
                }
                // Track as numeric optional so `else "string"` coalescing uses map_or_else.
                // Only for string→numeric casts (which produce Option<i64/f64>), not numeric→bool (None).
                if src_is_str && dst_is_numeric {
                    self.optional_numeric_vars.insert(s.name.clone());
                }
                // numeric→bool casts always produce None — mark for direct-default coalescing.
                if src_is_numeric && dst_is_bool {
                    self.always_none_vars.insert(s.name.clone());
                }
            }
        }
        // Track enum type for variables initialized from enum constructors.
        // `let c = Color.Green` or `let c = Color::Green(...)` → var_types["c"] = Named("Color")
        // This is used for match subject enum inference.
        if s.ty.is_none() {
            let inferred_enum = match &s_value.kind {
                ExprKind::Field(obj, variant) => {
                    if let ExprKind::Var(type_name) = &obj.kind {
                        let key = format!("{}::{}", type_name, variant);
                        if self.enum_variant_fields.contains_key(&key) {
                            Some(type_name.clone())
                        } else { None }
                    } else { None }
                }
                ExprKind::Call(callee, _) => {
                    // `Color.Green(x, y)` → `ExprKind::MethodCall(Color, "Green", [x,y])`
                    // handled separately below
                    if let ExprKind::Field(obj, variant) = &callee.kind {
                        if let ExprKind::Var(type_name) = &obj.kind {
                            let key = format!("{}::{}", type_name, variant);
                            if self.enum_variant_fields.contains_key(&key) {
                                Some(type_name.clone())
                            } else { None }
                        } else { None }
                    } else { None }
                }
                ExprKind::MethodCall(obj, variant, _) => {
                    if let ExprKind::Var(type_name) = &obj.kind {
                        let key = format!("{}::{}", type_name, variant);
                        if self.enum_variant_fields.contains_key(&key) {
                            Some(type_name.clone())
                        } else { None }
                    } else { None }
                }
                _ => None,
            };
            if let Some(enum_name) = inferred_enum {
                self.var_types.insert(s.name.clone(), Type::Named(enum_name));
            }
        }
        // Track newtype vars from explicit type annotation: `let id: UserId = ...`
        if let Some(Type::Named(ty_name)) = &s.ty {
            if self.newtype_types.contains(ty_name.as_str()) {
                self.var_newtype_type.insert(s.name.clone(), ty_name.clone());
            }
        }
        // Track Arc<T> variables (string, T'shared, T'actor, T'guard) — must be cloned before
        // being moved into an `async move {}` block so the outer binding stays valid.
        if let Some(ty) = &s.ty {
            // Resolve named type aliases (e.g. `use ONode as OTree'shared`) before classifying.
            let resolved_ty: Option<&Type> = if let Type::Named(n) = ty {
                self.non_fn_type_aliases.get(n.as_str()).map(|t| t as &Type)
            } else {
                None
            };
            let effective_ty = resolved_ty.unwrap_or(ty);
            if Self::is_string_type(effective_ty) || Self::is_arc_qualified(effective_ty) || Self::is_rc_qualified(effective_ty) {
                self.arc_vars.insert(s.name.clone());
                // In single-thread mode, T'shared → Rc<T>; mark for Rc::clone (not Arc::clone).
                if Self::is_rc_qualified(effective_ty) && matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                    self.rc_vars.insert(s.name.clone());
                }
            }
            // Track T'weak variables — already Weak<T>, must not be downgraded again.
            if Self::is_weak_qualified(ty) {
                self.weak_vars.insert(s.name.clone());
            }
            // Track Optional-typed variables so they are never double-wrapped in Some().
            if matches!(ty, Type::Optional(_)) {
                self.optional_vars.insert(s.name.clone());
            }
            // Track managed-mode T' (non-optional) variables — field/method access needs locking.
            if self.is_managed_owned_user(ty) {
                match self.config.threading {
                    crate::transpiler::ThreadingMode::Multi  => { self.managed_mutex_vars.insert(s.name.clone()); }
                    crate::transpiler::ThreadingMode::Single => { self.managed_refcell_vars.insert(s.name.clone()); }
                }
            }
            // Track managed-mode T'? (optional) variables — optional-chain access needs locking.
            if let Type::Optional(inner) = ty {
                if self.is_managed_owned_user(inner.as_ref()) {
                    match self.config.threading {
                        crate::transpiler::ThreadingMode::Multi  => { self.managed_mutex_vars.insert(s.name.clone()); }
                        crate::transpiler::ThreadingMode::Single => { self.managed_refcell_vars.insert(s.name.clone()); }
                    }
                }
            }
            // Track var type for match subject enum inference.
            self.var_types.insert(s.name.clone(), ty.clone());
            // Track string vars for string concatenation detection.
            if Self::is_string_type(ty) {
                self.string_vars.insert(s.name.clone());
            }
        }
        // Infer type of unannotated vars from field access on actor variables.
        // e.g. `let env = interp.global_env` where interp: Interpreter'actor and global_env: Env'actor
        // Without this, overload resolution can't distinguish `env_define(env,…)` overloads.
        if s.ty.is_none() {
            // `let sub = make_parser(...)` where make_parser returns Parser'actor — infer actor type.
            if matches!(&s_value.kind, ExprKind::Call(_, _) | ExprKind::MethodCall(_, _, _)) {
                let fn_name = match &s_value.kind {
                    ExprKind::Call(callee, _) => {
                        if let ExprKind::Var(n) = &callee.kind { Some(n.clone()) } else { None }
                    }
                    _ => None,
                };
                if let Some(fname) = fn_name {
                    if let Some(ret_ty) = self.fn_return_types.get(fname.as_str()).cloned() {
                        let is_actor = Self::is_mutex_binding(false, &ret_ty) || Self::is_rwlock_binding(false, &ret_ty);
                        if is_actor {
                            if Self::is_mutex_task_binding(false, &ret_ty) {
                                self.var_mutex_task_types.insert(s.name.clone());
                            } else if Self::is_rwlock_task_binding(false, &ret_ty) {
                                self.var_rwlock_task_types.insert(s.name.clone());
                            } else if Self::is_mutex_binding(false, &ret_ty) {
                                self.var_mutex_types.insert(s.name.clone());
                            } else {
                                self.var_rwlock_types.insert(s.name.clone());
                            }
                            self.arc_vars.insert(s.name.clone());
                            if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                                self.rc_vars.insert(s.name.clone());
                            }
                        }
                    }
                }
            }
            // `var f = Foo(...) as Foo'actor` — register f as actor/rc var (no explicit ty annotation).
            if let ExprKind::Cast(_, dst_ty) = &s_value.kind {
                let is_actor = Self::is_mutex_binding(false, dst_ty) || Self::is_rwlock_binding(false, dst_ty);
                let is_rc_like = Self::is_rc_qualified(dst_ty);
                if is_actor {
                    if Self::is_mutex_task_binding(false, dst_ty) {
                        self.var_mutex_task_types.insert(s.name.clone());
                    } else if Self::is_rwlock_task_binding(false, dst_ty) {
                        self.var_rwlock_task_types.insert(s.name.clone());
                    } else if Self::is_mutex_binding(false, dst_ty) {
                        self.var_mutex_types.insert(s.name.clone());
                    } else {
                        self.var_rwlock_types.insert(s.name.clone());
                    }
                    self.arc_vars.insert(s.name.clone());
                    if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        self.rc_vars.insert(s.name.clone());
                    }
                } else if is_rc_like {
                    self.arc_vars.insert(s.name.clone());
                    if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        self.rc_vars.insert(s.name.clone());
                    }
                }
            }
            // Propagate string_vars when `let x = str_var.trim()` / similar string methods.
            if let ExprKind::MethodCall(recv_expr, method_name, _) = &s_value.kind {
                // String-only methods: always return a string regardless of receiver tracking.
                const STRING_ONLY_METHODS: &[&str] = &[
                    "trim", "trimStart", "trimEnd", "toUpperCase", "toLowerCase",
                    "upper", "lower", "replace", "replaceAll",
                ];
                // Mixed methods: only return string when receiver is tracked as string.
                const STRING_CONDITIONAL_METHODS: &[&str] = &["slice"];
                let recv_is_str = match &recv_expr.kind {
                    ExprKind::Var(v) => self.string_vars.contains(v.as_str()) || self.string_arc_vars.contains(v.as_str()),
                    ExprKind::Str(_) | ExprKind::StringInterp(_) => true,
                    _ => false,
                };
                if STRING_ONLY_METHODS.contains(&method_name.as_str()) || (STRING_CONDITIONAL_METHODS.contains(&method_name.as_str()) && recv_is_str) {
                    self.string_vars.insert(s.name.clone());
                } else if method_name == "clone" {
                    // clone() on a string var or a string field access → result is also a string
                    let is_str_field = if let ExprKind::Field(obj, field_name) = &recv_expr.kind {
                        if let ExprKind::Var(v) = &obj.kind {
                            let struct_name = self.var_types.get(v.as_str())
                                .and_then(|t| match t {
                                    Type::Named(n) => Some(n.as_str()),
                                    Type::Qualified(inner, _) => if let Type::Named(n) = inner.as_ref() { Some(n.as_str()) } else { None },
                                    _ => None,
                                })
                                .or_else(|| self.var_struct_types.get(v.as_str()).map(|s| s.as_str()));
                            struct_name.and_then(|sn| self.struct_fields.get(sn))
                                .and_then(|fields| fields.iter().find(|(fname, _)| fname == field_name))
                                .map(|(_, fty)| Self::is_string_type(fty))
                                .unwrap_or(false)
                        } else { false }
                    } else { false };
                    if recv_is_str || is_str_field {
                        self.string_vars.insert(s.name.clone());
                    }
                }
            }
            // Propagate actor/rc type when `let g = f` where f is already an actor var.
            if let ExprKind::Var(src) = &s_value.kind {
                if self.var_mutex_types.contains(src.as_str()) {
                    self.var_mutex_types.insert(s.name.clone());
                    self.arc_vars.insert(s.name.clone());
                    if self.rc_vars.contains(src.as_str()) {
                        self.rc_vars.insert(s.name.clone());
                    }
                } else if self.var_mutex_task_types.contains(src.as_str()) {
                    self.var_mutex_task_types.insert(s.name.clone());
                    self.arc_vars.insert(s.name.clone());
                    if self.rc_vars.contains(src.as_str()) {
                        self.rc_vars.insert(s.name.clone());
                    }
                } else if self.var_rwlock_types.contains(src.as_str()) {
                    self.var_rwlock_types.insert(s.name.clone());
                    self.arc_vars.insert(s.name.clone());
                    if self.rc_vars.contains(src.as_str()) {
                        self.rc_vars.insert(s.name.clone());
                    }
                } else if self.var_rwlock_task_types.contains(src.as_str()) {
                    self.var_rwlock_task_types.insert(s.name.clone());
                    self.arc_vars.insert(s.name.clone());
                    if self.rc_vars.contains(src.as_str()) {
                        self.rc_vars.insert(s.name.clone());
                    }
                } else if self.rc_vars.contains(src.as_str()) {
                    self.rc_vars.insert(s.name.clone());
                    self.arc_vars.insert(s.name.clone());
                }
                if self.string_vars.contains(src.as_str()) {
                    self.string_vars.insert(s.name.clone());
                }
                if let Some(ty) = self.var_types.get(src.as_str()).cloned() {
                    self.var_types.insert(s.name.clone(), ty);
                }
            }
            if let ExprKind::Field(obj_expr, field_name) = &s_value.kind {
                if let ExprKind::Var(v) = &obj_expr.kind {
                    if self.var_mutex_types.contains(v.as_str()) {
                        let struct_ty_name = self.var_struct_types.get(v.as_str())
                            .cloned()
                            .or_else(|| self.var_types.get(v.as_str()).and_then(|t| match t {
                                Type::Named(n) => Some(n.clone()),
                                Type::Qualified(inner, _) => if let Type::Named(n) = inner.as_ref() { Some(n.clone()) } else { None },
                                _ => None,
                            }));
                        if let Some(ty_name) = struct_ty_name {
                            if let Some(fields) = self.struct_fields.get(ty_name.as_str()).cloned() {
                                if let Some((_, field_ty)) = fields.iter().find(|(fname, _)| fname == field_name) {
                                    let field_ty = field_ty.clone();
                                    if Self::is_mutex_binding(false, &field_ty) || Self::is_rwlock_binding(false, &field_ty) {
                                        if Self::is_mutex_task_binding(false, &field_ty) {
                                            self.var_mutex_task_types.insert(s.name.clone());
                                        } else if Self::is_rwlock_task_binding(false, &field_ty) {
                                            self.var_rwlock_task_types.insert(s.name.clone());
                                        } else if Self::is_mutex_binding(false, &field_ty) {
                                            self.var_mutex_types.insert(s.name.clone());
                                        } else {
                                            self.var_rwlock_types.insert(s.name.clone());
                                        }
                                        self.arc_vars.insert(s.name.clone());
                                        if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                                            self.rc_vars.insert(s.name.clone());
                                        }
                                    }
                                    self.var_types.insert(s.name.clone(), field_ty);
                                }
                            }
                        }
                    }
                    // Track string vars when initialized from a string field of any struct.
                    let struct_ty_name = self.var_struct_types.get(v.as_str())
                        .cloned()
                        .or_else(|| self.var_struct_type.get(v.as_str()).cloned())
                        .or_else(|| self.var_types.get(v.as_str()).and_then(|t| match t {
                            Type::Named(n) => Some(n.clone()),
                            Type::Qualified(inner, _) => if let Type::Named(n) = inner.as_ref() { Some(n.clone()) } else { None },
                            _ => None,
                        }));
                    if let Some(ty_name) = struct_ty_name {
                        let field_ty_opt = self.struct_fields.get(ty_name.as_str())
                            .and_then(|fields| fields.iter().find(|(fname, _)| fname == field_name))
                            .map(|(_, fty)| fty.clone());
                        if let Some(field_ty) = field_ty_opt {
                            if Self::is_string_type(&field_ty) {
                                self.string_vars.insert(s.name.clone());
                            }
                            // Propagate actor/guard qualifier from a plain struct's field to the local binding.
                            if Self::is_mutex_binding(false, &field_ty) || Self::is_rwlock_binding(false, &field_ty) {
                                if Self::is_mutex_task_binding(false, &field_ty) {
                                    self.var_mutex_task_types.insert(s.name.clone());
                                } else if Self::is_rwlock_task_binding(false, &field_ty) {
                                    self.var_rwlock_task_types.insert(s.name.clone());
                                } else if Self::is_mutex_binding(false, &field_ty) {
                                    self.var_mutex_types.insert(s.name.clone());
                                } else {
                                    self.var_rwlock_types.insert(s.name.clone());
                                }
                                self.arc_vars.insert(s.name.clone());
                                if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                                    self.rc_vars.insert(s.name.clone());
                                }
                            }
                            self.var_types.insert(s.name.clone(), field_ty);
                        }
                    }
                }
            }
        }
        false
    }

    pub(crate) fn emit_let(&mut self, s: &LetStmt, _is_last: bool) {
        // GPU targets only (see emit_kernel.rs): a `let`/`mut`/`var` initialized from a
        // `kernel Name(...)` constructor needs GPU-specific codegen, not the plain
        // tuple-constructor call the rest of this function would otherwise emit.
        if self.try_emit_kernel_let(s) {
            self.known_local_vars.insert(s.name.clone());
            return;
        }
        if self.try_emit_gpu_resident_let(s) {
            return;
        }
        if self.try_emit_gpu_resident_call_let(s) {
            return;
        }
        if self.try_emit_gpu_device_let(s) {
            self.known_local_vars.insert(s.name.clone());
            return;
        }
        // Validate `mut` qualifier combinations.
        if s.binding == BindingKind::Mut {
            if let Some(ty) = &s.ty {
                if matches!(Self::unwrap_qual(ty), OwnerQual::Shared) {
                    self.push_error(s.line, s.col,
                        "`mut` is not allowed with the `'shared` qualifier — \
                         use `'actor` for interior mutability instead"
                    );
                }
            }
        }
        // Track every declared local variable so that field/method access can distinguish
        // instance variables (use `.`) from type/module paths (use `::`).
        self.known_local_vars.insert(s.name.clone());
        // Track binding kind for parameter-passing checks (let ≤ mut ≤ var hierarchy).
        match s.binding {
            BindingKind::Let => {
                self.immutable_local_vars.insert(s.name.clone());
                self.mut_local_vars.remove(&s.name);
            }
            BindingKind::Mut => {
                self.immutable_local_vars.remove(&s.name);
                self.mut_local_vars.insert(s.name.clone());
            }
            _ => {
                // var / lazy — rebindable, not in either set
                self.immutable_local_vars.remove(&s.name);
                self.mut_local_vars.remove(&s.name);
            }
        }
        // Content-mutation permission (`def` calls, field writes) — independent
        // of the rebind-axis tracking above; see `content_mutable_local_vars`'s
        // doc and `crate::ast::binding_grants_mut`.
        if crate::ast::binding_grants_mut(&s.binding, s.var_mut, s.ty.as_ref()) {
            self.content_mutable_local_vars.insert(s.name.clone());
        } else {
            self.content_mutable_local_vars.remove(&s.name);
        }
        self.mut_checked_local_vars.insert(s.name.clone());
        // `lazy T name` — deferred write-once binding backed by OnceCell<T>.
        // `lazy` vars must NOT have an initializer; the value is provided later via `?=`.
        if s.binding == BindingKind::Lazy {
            self.lazy_vars.insert(s.name.clone());
            if let Some(ty) = &s.ty {
                self.lazy_var_types.insert(s.name.clone(), ty.clone());
                let inner_ty = self.emit_type(ty);
                let once_cell = format!("std::cell::OnceCell::<{}>::new()", inner_ty);
                self.line(&format!("let {} = {};", s.name, once_cell));
            } else {
                // No type annotation — emit without the turbofish
                self.line(&format!("let {} = std::cell::OnceCell::new();", s.name));
            }
            return;
        }
        // Track `let tx = broadcast<T>(cap)` (single-binding sender).
        if let Some(val) = &s.value {
            let is_broadcast_call = matches!(&val.kind,
                ExprKind::GenericCall(callee, _, _)
                if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"))
                || matches!(&val.kind,
                ExprKind::Call(callee, _)
                if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"));
            if is_broadcast_call && s.name != "_" {
                self.broadcast_senders.insert(s.name.clone());
            }
        }
        // Track `let rx = tx.subscribe()` so rx is recognized as a broadcast receiver.
        if let Some(val) = &s.value {
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
        // Shadowing: clear previous struct-type tracking so a re-declared variable with a
        // different type (e.g. `let d = Doubler()` then `let d'weak = c`) doesn't inherit
        // the old struct type and incorrectly suppress `.await.unwrap()` on `.value`.
        self.var_struct_types.remove(&s.name);
        // `let v` / `var v` — deferred initialisation: emit `let v;` and let Rust
        // enforce definite assignment via its own control-flow analysis.
        if s.value.is_none() {
            let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
            if let Some(ty) = &s.ty {
                self.line(&format!("{} {}: {};", kw, s.name, self.emit_type(ty)));
            } else {
                self.line(&format!("{} {};", kw, s.name));
            }
            return;
        }
        let s_value = s.value.as_ref().expect("invariant: Let statement without type annotation must have an initializer value");
        // T'actor / T'guard / managed-mode T' bindings — locking wrapper types that are
        // emitted entirely differently from a plain `let`, and fully handle their own
        // tracking + `self.line(...)` emission.
        if let Some(ty) = &s.ty {
            if self.try_emit_qualified_let(s, ty, s_value) {
                return;
            }
        }
        // Infer actor/refcell type for local let bindings from function return type (no explicit annotation).
        // E.g. `let child = new_env(...)` where `new_env` returns `Env'actor` — add child to managed_refcell_vars.
        if s.ty.is_none() {
            let ret_ty = match &s_value.kind {
                ExprKind::Call(callee, _) => {
                    if let ExprKind::Var(fn_name) = &callee.kind {
                        self.fn_return_types.get(fn_name.as_str()).cloned()
                    } else { None }
                }
                _ => None,
            };
            if let Some(Type::Qualified(_, crate::ast::OwnerQual::Actor)) = ret_ty {
                match self.config.threading {
                    crate::transpiler::ThreadingMode::Multi => {
                        self.managed_mutex_vars.insert(s.name.clone());
                        // Fresh Arc from a function return: no pre-lock (may be moved, can't deadlock).
                        self.managed_mutex_fn_return_vars.insert(s.name.clone());
                    }
                    crate::transpiler::ThreadingMode::Single => {
                        self.managed_refcell_vars.insert(s.name.clone());
                    }
                }
            }
        }
        let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
        let vis = if s.is_pub { "pub " } else { "" };
        let (ty, val, is_mutable_string_lit, is_mutable_string_ty) = self.compute_let_ty_and_value(s, s_value);
        if self.track_let_metadata(s, s_value, &val, is_mutable_string_lit, is_mutable_string_ty) {
            return;
        }
        // Reference-identity wrapping: if this variable is used in `x is y` comparisons
        // (which require pointer equality), wrap the value in `Rc::new(...)` so that:
        //   - assignment from another Rc variable (`let cdb = cda`) → `cda.clone()` shares pointer
        //   - constructing a new object → `Rc::new(CDog { ... })` gives unique pointer
        // Then `Rc::ptr_eq(&cdb, &cda)` correctly returns true/false.
        if self.rc_identity_vars.contains(&s.name) && !s.is_static {
            // If value is a struct constructor call, wrap in Rc::new.
            let is_struct_ctor = if let ExprKind::Call(callee, _) = &s_value.kind {
                if let ExprKind::Var(type_name) = &callee.kind {
                    self.struct_fields.contains_key(type_name.as_str())
                } else { false }
            } else { false };
            // If value is a simple variable reference to another rc_identity var, clone as Rc.
            let is_rc_var_ref = if let ExprKind::Var(vname) = &s_value.kind {
                self.rc_identity_vars.contains(vname.as_str())
            } else { false };

            if is_struct_ctor {
                let rc_val = format!("Rc::new({})", val);
                self.line(&format!("{}{} {}{} = {};", vis, kw, s.name, ty, rc_val));
                self.var_types.insert(s.name.clone(), Type::Named(format!("Rc<{}>", val)));
                return;
            } else if is_rc_var_ref {
                // Clone the Rc (shares pointer), not a deep clone.
                let src_var = if let ExprKind::Var(v) = &s_value.kind { v.clone() } else { val.clone() };
                let rc_val = format!("{}.clone()", src_var);
                self.line(&format!("{}{} {}{} = {};", vis, kw, s.name, ty, rc_val));
                self.var_types.insert(s.name.clone(), Type::Named("Rc<ref>".to_string()));
                return;
            }
        }
        if s.is_static {
            self.line(&format!("{}static {}: {} = {};", vis, s.name, ty.trim_start_matches(": ").trim(), val));
        } else {
            self.line(&format!("{}{} {}{} = {};", vis, kw, s.name, ty, val));
            // Emit a lock guard shadow for managed mutex locals in multi-thread mode to
            // avoid deadlock when multiple fields are accessed in the same expression
            // (two separate .lock().unwrap() calls hold the guard simultaneously).
            if matches!(self.config.threading, crate::transpiler::ThreadingMode::Multi)
                && self.managed_mutex_vars.contains(&s.name)
                && !self.managed_param_shadows.contains_key(&s.name)
                && !self.optional_vars.contains(&s.name)
                && !self.managed_mutex_fn_return_vars.contains(&s.name)
            {
                let shadow = format!("__{}_mg", s.name);
                self.line(&format!("let mut {} = {}.lock().unwrap();", shadow, s.name));
                self.managed_mutex_vars.remove(&s.name);
                self.managed_param_shadows.insert(s.name.clone(), shadow);
            }
        }
    }

    /// True when `v` is a function parameter whose bare `[T]`/`{K=V}`/`{T}` type was
    /// auto-ref inferred to `&Vec<T>`/`&HashMap<K,V>`/`&HashSet<T>` (see the free-function
    /// array/dict/set eligibility in `infer_qualifiers.rs`). Reading such a param into an
    /// owned context (a fresh local, a struct field, …) needs an explicit `.clone()` —
    /// Rust would otherwise infer the destination's type as the reference itself.
    fn is_borrowed_collection_param(&self, v: &str) -> bool {
        matches!(
            self.fn_current_params.get(v),
            Some(Type::Array(_) | Type::Dict(..) | Type::Set(_))
        ) && matches!(
            self.inferred_qualifiers.get(v),
            Some(OwnerQual::Borrow) | Some(OwnerQual::BorrowMut)
        )
    }

    /// `let T? name = value` — wraps a non-nil value in `Some(...)`, unless it's already
    /// known to produce `Option<T>` (an if-expression with nil/some branches, a throws call,
    /// a function/method/field known to return Optional, etc.), in which case it's passed
    /// through unwrapped to avoid double-wrapping.
    fn emit_let_value_optional(&self, inner: &Type, value: &Expr, is_option_cast: bool) -> String {
        // If-expression with mixed branches (some nil, some non-optional): emit via a
        // sub-transpiler that has fn_return_ty = Optional(inner) so each branch
        // independently wraps non-nil values in Some() and nil → None.
        if matches!(&value.kind, ExprKind::If(_)) {
            let mut sub = self.make_sub();
            sub.fn_return_ty = Some(Type::Optional(Box::new(inner.clone())));
            sub.suppress_ok_wrap = true;
            return sub.emit_expr(value);
        }
        if is_option_cast {
            // Cast to a numeric type with an Optional declared type: emit `.ok()` so
            // the result is Option<T>, matching the annotation.
            // e.g. `let int? v = s as int` → `s.trim().parse::<isize>().ok()`
            if let ExprKind::Cast(src, cast_ty) = &value.kind {
                let src_s = self.emit_expr(src);
                let cast_dst = self.emit_type(cast_ty);
                let parse_ty = if matches!(cast_ty, Type::Float) || matches!(cast_ty, Type::Named(n) if n == "float") {
                    Some("f64".to_string())
                } else if crate::transpiler::helpers::is_specific_numeric_type(&cast_dst) && cast_dst != "f32" && cast_dst != "f64" {
                    Some(cast_dst)
                } else {
                    None
                };
                if let Some(pt) = parse_ty {
                    return format!("{}.trim().parse::<{}>().ok()", src_s, pt);
                }
            }
            return self.emit_expr(value);
        }
        // Wrap non-nil value in Some(...)
        let inner_val = if Self::is_string_type(inner) {
            self.emit_expr_owned(value)
        } else if Self::is_weak_qualified(inner) {
            // T'weak? field: downgrade Rc to Weak, unless already Weak.
            let e = self.emit_expr(value);
            if self.weak_vars.contains(e.as_str()) || e.starts_with("Rc::downgrade(") {
                e
            } else {
                format!("Rc::downgrade(&{})", e)
            }
        } else {
            self.emit_expr(value)
        };
        // Detect expressions that already produce Option<T>:
        // • starts with "Some(" or equals "None"
        // • var already in optional_vars
        // • method calls known to return Option (indexOf, parseInt, parseFloat, string indexOf/find)
        //   whose emitted form ends with ".ok()" or ".map(|i| i as isize)"
        let already_opt = inner_val.starts_with("Some(") || inner_val == "None"
            || matches!(&value.kind, ExprKind::Var(v) if self.optional_vars.contains(v.as_str())
                || self.var_types.get(v.as_str()).map(|t| matches!(t, Type::Optional(_))).unwrap_or(false))
            || inner_val.ends_with(".ok()")
            || inner_val.ends_with(".map(|i| i as isize)")
            // A throws-propagated call (ending in `?`) in an Optional declared context
            // is already Option<T> — the throws function returns Result<Option<T>>.
            || (inner_val.ends_with("?") && matches!(&value.kind, ExprKind::Call(_, _)))
            // Free-function call whose declared return type is Option<T>
            || matches!(&value.kind, ExprKind::Call(callee, _)
                if matches!(&callee.kind, ExprKind::Var(fn_name)
                    if self.fn_return_types.get(fn_name.as_str())
                        .map(|t| matches!(t, Type::Optional(_))).unwrap_or(false)))
            // Method call where the struct method return type is Optional
            || matches!(&value.kind, ExprKind::MethodCall(recv, method, _)
                if matches!(&recv.kind, ExprKind::Var(v)
                    if self.var_struct_types.get(v.as_str()).map(|sty| {
                        self.struct_method_return_types
                            .get(&format!("{}::{}", sty, method))
                            .map(|t| matches!(t, Type::Optional(_)))
                            .unwrap_or(false)
                    }).unwrap_or(false)))
            // Field access on a known struct where the field type is Optional
            || matches!(&value.kind, ExprKind::Field(obj, field_name) if {
                let sn = match &obj.kind {
                    ExprKind::Var(v) if v == "self" => self.self_type.clone(),
                    ExprKind::Var(v) => self.var_struct_types.get(v.as_str()).cloned(),
                    _ => None,
                };
                sn.and_then(|sn| self.struct_fields.get(sn.as_str()))
                    .and_then(|fs| fs.iter().find(|(n, _)| n == field_name))
                    .map(|(_, ty)| matches!(ty, Type::Optional(_)))
                    .unwrap_or(false)
            })
            // If-expression whose branches already produce Option (nil/some/method branches)
            || matches!(&value.kind, ExprKind::If(if_stmt) if {
                fn body_ends_optional(body: &[Stmt]) -> bool {
                    match body.last() {
                        Some(Stmt::Expr(e)) => matches!(&e.kind, ExprKind::Nil)
                            || matches!(&e.kind, ExprKind::Call(callee, _)
                                if matches!(&callee.kind, ExprKind::Var(v) if v == "some"))
                            // Method call on a variable — likely returns Optional in Optional ctx.
                            || matches!(&e.kind, ExprKind::MethodCall(_, _, _)),
                        _ => false,
                    }
                }
                if_stmt.branches.iter().any(|(_, b)| body_ends_optional(b))
                    || if_stmt.else_body.as_ref().map(|b| body_ends_optional(b)).unwrap_or(false)
            });
        if already_opt { return inner_val; }
        // `T'? (Box<T>?)` or managed-mode `T'?`: wrap the value appropriately.
        let wrapped = if matches!(inner, Type::Qualified(_, OwnerQual::Owned | OwnerQual::New)) {
            // Managed mode: wrap in Arc<std::sync::Mutex<T>> or RefCell<T>
            if self.is_managed_owned_user(inner) {
                if inner_val.starts_with("Arc::new(std::sync::Mutex::new(")
                    || inner_val.starts_with("RefCell::new(")
                {
                    inner_val
                } else {
                    self.wrap_managed(&inner_val)
                }
            } else {
                // Strict mode: wrap in Box::new(...)
                if inner_val.starts_with("Box::new(") { inner_val }
                else { format!("Box::new({})", inner_val) }
            }
        } else {
            inner_val
        };
        format!("Some({})", wrapped)
    }

    /// `let T'task/'actor/'guard name = value` (any Arc-qualified type) → `Arc<T>`, or
    /// `Arc<Mutex<T>>`/`Arc<RwLock<T>>` for actor/guard (`Rc<RefCell<T>>`/`Rc<T>` in
    /// single-thread mode). Clones an existing Arc/Rc var instead of moving it; unboxes
    /// a `'heap` source with `*` before wrapping; `.clone()`s a `'stack` source so the
    /// original binding stays valid.
    fn emit_let_value_arc_qualified(&self, t: &Type, value: &Expr) -> String {
        let is_actor = Self::is_mutex_binding(false, t);
        let is_guard = Self::is_rwlock_binding(false, t);
        let is_actor_or_guard = is_actor || is_guard;
        let is_single = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single);
        let is_heap = self.arg_is_heap_var(value);
        // In single-thread mode, T'actor/T'guard = Rc<RefCell<T>> — use Rc::clone.
        if is_actor_or_guard && is_single {
            let inner = self.emit_expr(value);
            let is_existing_rc = inner.starts_with("Rc::clone(") || inner.starts_with("Rc::new(")
                || matches!(&value.kind, ExprKind::Var(v)
                    if self.var_mutex_types.contains(v.as_str())
                    || self.var_rwlock_types.contains(v.as_str())
                    || self.rc_vars.contains(v.as_str()))
                // MethodCall(obj, "clone", []) on a known rc_var — already an Rc<RefCell<T>>
                || matches!(&value.kind, ExprKind::MethodCall(obj, m, args)
                    if m == "clone" && args.is_empty()
                    && matches!(&obj.kind, ExprKind::Var(v)
                        if self.var_mutex_types.contains(v.as_str())
                        || self.var_rwlock_types.contains(v.as_str())
                        || self.rc_vars.contains(v.as_str())));
            return if is_existing_rc {
                if inner.starts_with("Rc::") { inner }
                else { format!("Rc::clone(&{})", inner.trim_end_matches(".clone()")) }
            } else if is_heap {
                format!("Rc::new(RefCell::new(*{}))", inner)
            } else {
                format!("Rc::new(RefCell::new({}))", inner)
            };
        }
        let inner = self.emit_expr(value);
        // Already wrapped in Arc::new/clone — pass through.
        if inner.starts_with("Arc::new(") || inner.starts_with("Arc::clone(") {
            return inner;
        }
        // If the value is an existing Arc variable, clone it instead of moving.
        let is_existing_arc =
            (is_actor && matches!(&value.kind, ExprKind::Var(v) if self.var_mutex_types.contains(v.as_str())))
            || (is_guard && matches!(&value.kind, ExprKind::Var(v) if self.var_rwlock_types.contains(v.as_str())))
            || matches!(&value.kind, ExprKind::Var(v) if self.arc_vars.contains(v.as_str()))
            // MethodCall(obj, "clone", []) on a known arc_var — already Arc<Mutex<T>>
            || matches!(&value.kind, ExprKind::MethodCall(obj, m, args)
                if m == "clone" && args.is_empty()
                && matches!(&obj.kind, ExprKind::Var(v)
                    if self.var_mutex_types.contains(v.as_str()) || self.arc_vars.contains(v.as_str())))
            // Field access on a struct where the field is actor/Arc typed (e.g. interp.global_env)
            || matches!(&value.kind, ExprKind::Field(obj, field_name)
                if matches!(&obj.kind, ExprKind::Var(v) if {
                    let sn = self.var_struct_types.get(v.as_str())
                        .cloned()
                        .or_else(|| self.var_types.get(v.as_str()).and_then(|t| match t {
                            Type::Named(n) => Some(n.clone()),
                            _ => None,
                        }));
                    sn.and_then(|sn| self.struct_fields.get(sn.as_str()))
                        .and_then(|fields| fields.iter().find(|(fname, _)| fname == field_name))
                        .map(|(_, fty)| Self::is_arc_qualified(fty) || Self::is_rc_qualified(fty)
                            || Self::is_mutex_binding(false, fty) || Self::is_rwlock_binding(false, fty))
                        .unwrap_or(false)
                }));
        if is_existing_arc {
            format!("Arc::clone(&{})", inner)
        } else if is_heap {
            // Unbox before wrapping in the appropriate lock type.
            if is_actor {
                self.emit_actor_new(&format!("*{}", inner))
            } else if is_guard {
                self.emit_guard_new(&format!("*{}", inner))
            } else {
                format!("Arc::new(*{})", inner)
            }
        } else if matches!(&value.kind, ExprKind::Var(_)) {
            // 'stack source: wrap with lock + .clone() to preserve the original binding.
            if is_actor {
                self.emit_actor_new(&format!("{}.clone()", inner))
            } else if is_guard {
                self.emit_guard_new(&format!("{}.clone()", inner))
            } else {
                format!("Arc::new({}.clone())", inner)
            }
        } else {
            format!("Arc::new({})", inner)
        }
    }

    pub(crate) fn emit_let_value(&self, declared_ty: Option<&Type>, value: &Expr) -> String {
        // Implicit Arc::clone for auto-ref parameters assigned to an owned context.
        // e.g. `counter = c` where `c: Counter'actor` (emitted as &Arc<Mutex<Counter>>)
        // and `counter` expects an owned Arc<Mutex<Counter>>.
        // Note: T'actor/'shared/'guard params are now by-value (owned clones at call site).
        // The regular emit_let_value coercion paths below handle Rc::clone/Arc::clone correctly.
        // Resolve named type aliases through non_fn_type_aliases before dispatching.
        // e.g. `use Pt as LPoint'` makes `Pt` an alias for `Box<LPoint>`;
        // when calling `describe(p)` where describe expects `Pt`, we must Box::new() the arg.
        let declared_ty = if let Some(Type::Named(n)) = declared_ty {
            self.non_fn_type_aliases.get(n.as_str()).or(declared_ty)
        } else {
            declared_ty
        };
        // Fixed-size array: `[val for N]` or `[val; N]` with declared type `[T, N]` → `[val; N]`
        if let Some(Type::ArrayN(elem_ty, n)) = declared_ty {
            match &value.kind {
                ExprKind::ArrayFill { value: fill_val, .. } => {
                    let v = self.emit_let_value(Some(elem_ty), fill_val);
                    return format!("[{}; {}]", v, n);
                }
                ExprKind::Array(elems) => {
                    let es: Vec<String> = elems.iter().map(|e| self.emit_let_value(Some(elem_ty), e)).collect();
                    return format!("[{}]", es.join(", "));
                }
                _ => {}
            }
        }
        // Context-aware DotIdent: `.Variant` with a known Named enum type → `EnumType::Variant`.
        // This ensures `.South` resolves to `Direction::South` (not a later enum with same variant).
        // Also handles qualified types (e.g. Direction'stack inferred by cross-fn propagation).
        if let ExprKind::DotIdent(variant) = &value.kind {
            if let Some(Type::Named(enum_type)) = declared_ty {
                let enum_rust = normalize_type_name(enum_type, self.use_rc_str());
                return format!("{}::{}", enum_rust, variant);
            }
            if let Some(Type::Qualified(inner, qual)) = declared_ty {
                if let Type::Named(enum_type) = inner.as_ref() {
                    let enum_rust = normalize_type_name(enum_type, self.use_rc_str());
                    if matches!(qual, OwnerQual::Owned | OwnerQual::New) {
                        return format!("Box::new({}::{})", enum_rust, variant);
                    } else {
                        return format!("{}::{}", enum_rust, variant);
                    }
                }
            }
        }
        // Context-aware static method call: `.fromSecs(1)` with type hint `Duration`
        //   → `Duration::from_secs(1)`.
        // Pattern: Call(DotIdent(method), args) + declared_ty = Named(TypeName).
        // camel_to_snake applied so Boring `.fromSecs` → Rust `from_secs`.
        if let ExprKind::Call(callee, dot_args) = &value.kind {
            if let ExprKind::DotIdent(method) = &callee.kind {
                if let Some(Type::Named(type_name)) = declared_ty {
                    let rust_type = normalize_type_name(type_name, self.use_rc_str());
                    let rust_method = camel_to_snake(method);
                    let vals: Vec<String> = dot_args.iter()
                        .map(|a| self.emit_expr(&a.value))
                        .collect();
                    return format!("{}::{}({})", rust_type, rust_method, vals.join(", "));
                }
            }
        }
        let is_nil = matches!(value.kind, ExprKind::Nil);
        // A Cast to a numeric type (or directly to Optional) already returns Option<T>
        let is_option_cast = matches!(&value.kind, ExprKind::Cast(_, ty)
            if matches!(ty, Type::Int | Type::Uint | Type::Uint8 | Type::Float
                | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128
                | Type::Named(_) | Type::Optional(_)));
        match declared_ty {
            Some(Type::Optional(inner)) if !is_nil => self.emit_let_value_optional(inner, value, is_option_cast),
            Some(t) if Self::is_str_ref_type(t) => {
                // str param (&str): literals are already &str; variables need &* to
                // coerce Arc<str> → &String → &str via Rust deref coercions.
                match &value.kind {
                    ExprKind::Str(_) => self.emit_expr(value),
                    _ => format!("&*{}", self.emit_expr(value)),
                }
            }
            // [string] / [str] param: coerce each array element to Arc<str>.
            Some(Type::Array(elem_ty)) if Self::is_string_type(elem_ty) || Self::is_str_ref_type(elem_ty) => {
                match &value.kind {
                    ExprKind::Array(elems) => {
                        let es: Vec<String> = elems.iter()
                            .map(|e| self.emit_let_value(Some(elem_ty), e))
                            .collect();
                        format!("vec![{}]", es.join(", "))
                    }
                    ExprKind::Var(v) if self.is_borrowed_collection_param(v) => {
                        format!("{}.clone()", self.emit_expr(value))
                    }
                    _ => self.emit_expr(value),
                }
            }
            // {string} Set field: emit typed HashSet::new() for empty set literals.
            Some(Type::Dict(..)) => {
                // Empty set literal `{}` parsed as ExprKind::Set — coerce to HashMap::new().
                match &value.kind {
                    ExprKind::Set(elems) if elems.is_empty() => "HashMap::new()".to_string(),
                    _ => self.emit_expr_owned(value),
                }
            }
            Some(Type::Set(elem_ty)) if Self::is_string_type(elem_ty) || Self::is_str_ref_type(elem_ty) => {
                match &value.kind {
                    ExprKind::Set(elems) if elems.is_empty() => "HashSet::<Arc<str>>::new()".to_string(),
                    _ => self.emit_expr(value),
                }
            }
            // T'borrow (e.g. Task& or MyStruct&): pass by reference.
            Some(Type::Qualified(_, OwnerQual::Borrow)) => {
                let s = self.emit_expr(value);
                if s.starts_with('&') { s } else { format!("&{}", s) }
            }
            // T&shared (e.g. OCounter&shared) → &Arc<T> (multi) or &Rc<T> (single): pass reference.
            Some(Type::Qualified(_, OwnerQual::BorrowShared)) => {
                // Use the raw variable name (not .clone()) so the borrow is valid.
                if let ExprKind::Var(v) = &value.kind {
                    format!("&{}", v)
                } else {
                    let s = self.emit_expr(value);
                    if s.starts_with('&') { s } else { format!("&{}", s) }
                }
            }
            // 'actor / 'guard / task variants: callee receives &Arc<Mutex<T>> / &Arc<RwLock<T>>.
            // Three cases for the arg variable v:
            //   1. Owned Arc local (in var_mutex_types / var_rwlock_types, NOT a param) → &v
            //   2. &Arc param being forwarded (inferred as actor, or explicit actor param) → v
            //   3. Plain non-actor value → wrap in Arc::new(Mutex::new(v)) and borrow: &Arc::new(...)
            Some(Type::Qualified(_, OwnerQual::Actor)) => {
                if let ExprKind::Var(v) = &value.kind {
                    let is_owned_actor = self.var_mutex_types.contains(v.as_str())
                        && !self.fn_current_params.contains_key(v.as_str());
                    let is_borrowed_actor = self.var_mutex_types.contains(v.as_str())
                        && self.fn_current_params.contains_key(v.as_str())
                        || matches!(self.inferred_qualifiers.get(v.as_str()), Some(crate::ast::OwnerQual::Actor | crate::ast::OwnerQual::ActorTask));
                    if is_owned_actor { format!("&{}", v) }
                    else if is_borrowed_actor { v.to_string() }
                    else { let inner = self.emit_expr(value); format!("&{}", self.emit_actor_new(&inner)) }
                } else if let ExprKind::MethodCall(recv, method, _) = &value.kind {
                    // `actor_var.clone()` already produces Arc<Mutex<T>> — don't double-wrap.
                    if method == "clone" {
                        if let ExprKind::Var(v) = &recv.kind {
                            if self.var_mutex_types.contains(v.as_str()) {
                                return format!("{}.clone()", v);
                            }
                        }
                    }
                    let inner = self.emit_expr(value);
                    format!("&{}", self.emit_actor_new(&inner))
                } else {
                    let inner = self.emit_expr(value);
                    if inner.starts_with('&') {
                        inner
                    } else if inner.starts_with("Arc::") || inner.starts_with("Rc::") {
                        // Already an owned Arc/Rc (e.g. field access on actor or Arc::clone call)
                        // — just borrow it, don't double-wrap.
                        format!("&{}", inner)
                    } else {
                        format!("&{}", self.emit_actor_new(&inner))
                    }
                }
            }
            Some(Type::Qualified(_, OwnerQual::ActorTask)) => {
                if let ExprKind::Var(v) = &value.kind {
                    let is_owned = self.var_mutex_task_types.contains(v.as_str())
                        && !self.fn_current_params.contains_key(v.as_str());
                    let is_borrowed = self.var_mutex_task_types.contains(v.as_str())
                        && self.fn_current_params.contains_key(v.as_str())
                        || matches!(self.inferred_qualifiers.get(v.as_str()), Some(crate::ast::OwnerQual::Actor | crate::ast::OwnerQual::ActorTask));
                    if is_owned { format!("&{}", v) }
                    else if is_borrowed { v.to_string() }
                    else { let inner = self.emit_expr(value); format!("&{}", self.emit_actor_task_new(&inner)) }
                } else {
                    let inner = self.emit_expr(value);
                    if inner.starts_with('&') { inner } else { format!("&{}", self.emit_actor_task_new(&inner)) }
                }
            }
            Some(Type::Qualified(_, OwnerQual::Guard)) => {
                if let ExprKind::Var(v) = &value.kind {
                    let is_owned = self.var_rwlock_types.contains(v.as_str())
                        && !self.fn_current_params.contains_key(v.as_str());
                    let is_borrowed = self.var_rwlock_types.contains(v.as_str())
                        && self.fn_current_params.contains_key(v.as_str())
                        || matches!(self.inferred_qualifiers.get(v.as_str()), Some(crate::ast::OwnerQual::Guard | crate::ast::OwnerQual::GuardTask));
                    if is_owned { format!("&{}", v) }
                    else if is_borrowed { v.to_string() }
                    else { let inner = self.emit_expr(value); format!("&{}", self.emit_guard_new(&inner)) }
                } else {
                    let inner = self.emit_expr(value);
                    if inner.starts_with('&') { inner } else { format!("&{}", self.emit_guard_new(&inner)) }
                }
            }
            Some(Type::Qualified(_, OwnerQual::GuardTask)) => {
                if let ExprKind::Var(v) = &value.kind {
                    let is_owned = self.var_rwlock_task_types.contains(v.as_str())
                        && !self.fn_current_params.contains_key(v.as_str());
                    let is_borrowed = self.var_rwlock_task_types.contains(v.as_str())
                        && self.fn_current_params.contains_key(v.as_str())
                        || matches!(self.inferred_qualifiers.get(v.as_str()), Some(crate::ast::OwnerQual::Guard | crate::ast::OwnerQual::GuardTask));
                    if is_owned { format!("&{}", v) }
                    else if is_borrowed { v.to_string() }
                    else { let inner = self.emit_expr(value); format!("&{}", self.emit_guard_task_new(&inner)) }
                } else {
                    let inner = self.emit_expr(value);
                    if inner.starts_with('&') { inner } else { format!("&{}", self.emit_guard_task_new(&inner)) }
                }
            }
            // T& mutable borrow (var T&): pass &mut reference.
            Some(Type::Qualified(_, OwnerQual::BorrowMut)) => {
                if let ExprKind::Var(v) = &value.kind {
                    format!("&mut {}", v)
                } else {
                    let s = self.emit_expr(value);
                    if s.starts_with("&mut ") { s } else { format!("&mut {}", s) }
                }
            }
            Some(t) if Self::is_string_type(t) => {
                let s = self.emit_expr_owned(value);
                // emit_expr_owned may return &str for index/method results not handled specially.
                // Ensure the result is Rc/Arc<str>; if not, wrap with Rc/Arc::<str>::from(x.to_string()).
                if s.starts_with("Arc::") || s.starts_with("Rc::") {
                    s
                } else if matches!(&value.kind,
                    ExprKind::Var(v) if self.arc_vars.contains(v.as_str())
                        || self.string_arc_vars.contains(v.as_str()))
                {
                    // Known Rc/Arc<str> variable — clone the pointer efficiently
                    format!("{}.clone()", s)
                } else {
                    self.str_from_expr(&format!("{}.to_string()", s))
                }
            }
            // T'shared → Arc<T> (multi) or Rc<T> (single): wrap accordingly.
            // 'stack source: wrap with .clone() to avoid moving the original binding.
            // 'heap source (Box<T>): dereference with * to move out of the box before wrapping.
            Some(Type::Qualified(_, OwnerQual::Shared)) => {
                let inner = self.emit_expr(value);
                let is_heap = self.arg_is_heap_var(value);
                match self.config.threading {
                    crate::transpiler::ThreadingMode::Single => {
                        let already_rc_expr = inner.starts_with("Rc::new(") || inner.starts_with("Rc::clone(");
                        let is_existing_shared_var = matches!(&value.kind, ExprKind::Var(v)
                                if matches!(self.var_types.get(v.as_str()),
                                    Some(Type::Qualified(_, OwnerQual::Shared))));
                        if already_rc_expr {
                            inner
                        } else if is_existing_shared_var || matches!(&value.kind, ExprKind::Var(v) if self.rc_vars.contains(v.as_str())) {
                            format!("Rc::clone(&{})", inner)
                        } else if is_heap {
                            format!("Rc::new(*{})", inner)
                        } else if matches!(&value.kind, ExprKind::Var(_)) {
                            format!("Rc::new({}.clone())", inner)
                        } else {
                            format!("Rc::new({})", inner)
                        }
                    }
                    crate::transpiler::ThreadingMode::Multi => {
                        if inner.starts_with("Arc::new(") || inner.starts_with("Arc::clone(") {
                            return inner;
                        }
                        let is_existing_arc = matches!(&value.kind, ExprKind::Var(v) if self.arc_vars.contains(v.as_str()));
                        if is_existing_arc {
                            format!("Arc::clone(&{})", inner)
                        } else if is_heap {
                            format!("Arc::new(*{})", inner)
                        } else if matches!(&value.kind, ExprKind::Var(_)) {
                            format!("Arc::new({}.clone())", inner)
                        } else {
                            format!("Arc::new({})", inner)
                        }
                    }
                }
            }
            // T'weak → Weak<T>: downgrade from Rc or Arc, unless already Weak.
            Some(t) if Self::is_weak_qualified(t) => {
                let inner = self.emit_expr(value);
                // Don't double-downgrade a variable already declared as T'weak.
                if self.weak_vars.contains(inner.as_str())
                    || inner.starts_with("Rc::downgrade(")
                    || inner.starts_with("Arc::downgrade(")
                {
                    return inner;
                }
                // Use Arc::downgrade only in multi-thread mode for compound-weak types,
                // or when the RHS variable is known to be an Arc.
                // In single-thread mode, `T'shared` uses Rc, so all weak refs use Rc::downgrade.
                let use_arc = matches!(self.config.threading, crate::transpiler::ThreadingMode::Multi)
                    && (Self::is_arc_weak(t) || self.arc_vars.contains(inner.as_str()));
                if use_arc {
                    format!("Arc::downgrade(&{})", inner)
                } else {
                    format!("Rc::downgrade(&{})", inner)
                }
            }
            // T'task / T'actor → Arc<T> (or Arc<Mutex<T>> for actor, Rc<RefCell<T>> in single).
            // 'stack source: wrap contents with .clone().
            // 'heap source (Box<T>): dereference with * to move out of the box before wrapping.
            Some(t) if Self::is_arc_qualified(t) => self.emit_let_value_arc_qualified(t, value),
            // Tuple type: coerce each element to its declared slot type.
            // `let (int, string) t = (0, "hello")` → `let t: (i64, Arc<str>) = (0, Arc::from("hello".to_string()))`
            Some(Type::Tuple(elem_tys)) => {
                if let ExprKind::Tuple(elems) = &value.kind {
                    let parts: Vec<String> = elems.iter().enumerate().map(|(i, e)| {
                        let slot_ty = elem_tys.get(i);
                        self.emit_let_value(slot_ty, e)
                    }).collect();
                    format!("({})", parts.join(", "))
                } else {
                    self.emit_expr(value)
                }
            }
            // T'owned (Box<T> in strict, Arc<Mutex<T>>/RefCell<T> in managed): wrap accordingly.
            Some(ty @ Type::Qualified(_, OwnerQual::Owned | OwnerQual::New)) => {
                let inner = self.emit_expr(value);
                if self.is_managed_owned_user(ty) {
                    if inner.starts_with("Arc::new(std::sync::Mutex::new(")
                        || inner.starts_with("RefCell::new(")
                    {
                        inner
                    } else {
                        self.wrap_managed(&inner)
                    }
                } else if inner.starts_with("Box::new(") {
                    inner
                } else {
                    format!("Box::new({})", inner)
                }
            }
            _ => self.emit_let_value_fallback(declared_ty, value),
        }
    }

    /// Fallback for `let`/`var` bindings whose declared type didn't match any of the
    /// ownership-qualifier cases above (plain structs, enums, primitives, borrowed
    /// collection params): auto-clones `'shared`/`'actor`/`'guard` vars (Rc/Arc refcount
    /// bump, not a deep copy), struct/enum vars, borrowed collection params, and string
    /// struct fields — everywhere else Rust would otherwise move out from under the binding.
    fn emit_let_value_fallback(&self, declared_ty: Option<&Type>, value: &Expr) -> String {
        let s = self.emit_expr(value);
        // Implicit clone for 'shared / 'actor / 'guard: cloning an Rc/Arc is just a
        // refcount increment, not a deep copy — assignment is always an alias.
        // 'stack and 'heap are owned types; move is the default there.
        if let ExprKind::Var(v) = &value.kind {
            if self.arc_vars.contains(v.as_str()) && !s.ends_with(".clone()") {
                return if self.rc_vars.contains(v.as_str()) {
                    format!("Rc::clone(&{})", v)
                } else {
                    format!("Arc::clone(&{})", v)
                };
            }
        }
        // If the value is a variable that holds a struct (non-Arc) type, clone it
        // to avoid a move. In Boring, assignment always copies; Rust structs need .clone().
        // Exception: `var` params are `&mut T` — don't clone, pass the reference directly.
        if let ExprKind::Var(v) = &value.kind {
            if self.var_struct_types.contains_key(v.as_str())
                && !self.arc_vars.contains(v.as_str())
                && !self.var_mutex_types.contains(v.as_str())
                && !self.var_primitive_params.contains(v.as_str())
                && !s.ends_with(".clone()")
            {
                return format!("{}.clone()", s);
            }
            // If the param type is a user-defined enum, clone to avoid moves.
            // Enum values are Clone but not Copy; re-use in loops requires cloning.
            if let Some(Type::Named(type_name)) = declared_ty {
                let is_user_enum = self.enum_variant_fields.keys()
                    .any(|k| k.starts_with(&format!("{}::", type_name)));
                if is_user_enum && !s.ends_with(".clone()") {
                    return format!("{}.clone()", s);
                }
            }
            // Auto-ref array/dict/set param (bare [T]/{K=V}/{T} inferred to &Vec<T>/
            // &HashMap<K,V>/&HashSet<T>): assigning it to a fresh local needs a deep
            // clone, same reasoning as the struct case above — otherwise Rust infers
            // the local's type as the reference itself, and any later mutation
            // (`.push()`, index-assign, etc.) fails to borrow it as mutable.
            if self.is_borrowed_collection_param(v) && !s.ends_with(".clone()") {
                return format!("{}.clone()", s);
            }
        }
        // If the value is a field access on a local struct variable, and the field type
        // is `string` (Rc<str>/Arc<str>), add .clone() to avoid a partial struct move.
        // In Boring, string is a reference-counted type — cloning is cheap and required.
        if let ExprKind::Field(obj, field_name) = &value.kind {
            if let ExprKind::Var(obj_var) = &obj.kind {
                let struct_type_name = self.var_struct_types.get(obj_var.as_str())
                    .or_else(|| self.var_struct_type.get(obj_var.as_str()));
                let field_is_string = struct_type_name
                    .and_then(|sn| self.struct_fields.get(sn.as_str()))
                    .and_then(|fs| fs.iter().find(|(n, _)| n == field_name.as_str()))
                    .map(|(_, ty)| Self::is_string_type(ty))
                    .unwrap_or(false);
                if field_is_string && !s.ends_with(".clone()") {
                    return format!("{}.clone()", s);
                }
            }
        }
        s
    }

    pub(crate) fn emit_let_destructure(&mut self, s: &LetDestructureStmt) {
        // Track all bound names as known locals, and each slot's own
        // content-mutation permission — independent per-slot, may differ from
        // the statement's overall `binding` (docs/mut-type-modifier.md §4).
        for b in &s.bindings {
            if b.name != "_" {
                self.known_local_vars.insert(b.name.clone());
                if crate::ast::binding_grants_mut(&b.binding, b.var_mut, b.ty.as_ref()) {
                    self.content_mutable_local_vars.insert(b.name.clone());
                } else {
                    self.content_mutable_local_vars.remove(&b.name);
                }
                self.mut_checked_local_vars.insert(b.name.clone());
            }
        }
        // Interprocedural GPU residency, tuple case (mirrors
        // `emit_kernel::try_emit_gpu_resident_call_let` for the single-value case):
        // when `s.value` calls a `fn_returns_resident_tuple` function, its Rust
        // signature returns e.g. `(BoringGpuArg<f64>, Vec<f64>, Vec<f64>)` regardless
        // of what this call site wants — handle that before any of the ordinary
        // destructure logic below (channel/oneshot/broadcast detection etc. all
        // assume a plain value and would mis-handle a `BoringGpuArg<T>` element).
        if let ExprKind::Call(callee, _) = &s.value.kind {
            if let ExprKind::Var(fn_name) = &callee.kind {
                if let Some(flags) = self.fn_returns_resident_tuple.get(fn_name.as_str()).cloned() {
                    self.emit_resident_tuple_destructure(s, &flags);
                    return;
                }
            }
        }
        // `let [a, b] = join [f1, f2]` — parallel JoinHandle await
        if let ExprKind::JoinAll(handles) = &s.value.kind {
            let n = handles.len();
            // Emit: let (__jh0, __jh1, ...) = tokio::join!(f1, f2, ...);
            let tmp_vars: Vec<String> = (0..n).map(|i| format!("__jh{}", i)).collect();
            let handle_exprs: Vec<String> = handles.iter().map(|e| self.emit_expr(e)).collect();
            self.line(&format!("let ({}) = tokio::join!({});",
                tmp_vars.join(", "),
                handle_exprs.join(", ")));
            // Emit: let a = __jh0.unwrap(); let b = __jh1.unwrap();
            // Note: tokio::join! already resolves the futures — results are Result<T, JoinError>,
            // NOT JoinHandles — so no extra `.await` needed here.
            let unwrap_or_q = if self.in_throws || self.in_try_body { "?" } else { ".unwrap()" };
            for (i, binding) in s.bindings.iter().enumerate() {
                if binding.name == "_" { continue; }
                let tmp = &tmp_vars[i];
                let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
                self.line(&format!("{} {} = {}{};", kw, binding.name, tmp, unwrap_or_q));
            }
            return;
        }
        // Detect `let tx, rx = channel<T>(n)` or `let T tx, rx = channel(n)`.
        // Must be done before building bindings so we can suppress type annotations on LHS.
        let is_channel_generic = matches!(&s.value.kind, ExprKind::GenericCall(callee, _, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "channel"));
        let is_channel_typed = !is_channel_generic && matches!(&s.value.kind,
            ExprKind::Call(callee, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "channel"));
        let is_channel = is_channel_generic || is_channel_typed;
        if is_channel {
            if let (Some(sender), Some(receiver)) = (s.bindings.first(), s.bindings.get(1)) {
                if sender.name != "_" { self.channel_senders.insert(sender.name.clone()); }
                if receiver.name != "_" {
                    self.channel_receivers.insert(receiver.name.clone());
                    // Track whether the channel element type is `string` so that
                    // values received from it are known to be Arc<str>.
                    let is_string_elem = match &s.value.kind {
                        ExprKind::GenericCall(_, type_args, _) => type_args.first()
                            .map(|t| matches!(t, Type::Named(n) if n == "string" || n == "String"))
                            .unwrap_or(false),
                        _ => s.bindings.first()
                            .and_then(|b| b.ty.as_ref())
                            .map(|t| matches!(t, Type::Named(n) if n == "string" || n == "String"))
                            .unwrap_or(false),
                    };
                    if is_string_elem {
                        self.string_channel_receivers.insert(receiver.name.clone());
                    }
                }
                self.has_streams = true;
            }
        }
        // Detect `let tx, rx = oneshot<T>()`.
        let is_oneshot = matches!(&s.value.kind, ExprKind::GenericCall(callee, _, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "oneshot"))
            || matches!(&s.value.kind, ExprKind::Call(callee, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "oneshot"));
        if is_oneshot {
            if let (Some(sender), Some(receiver)) = (s.bindings.first(), s.bindings.get(1)) {
                if sender.name != "_" { self.oneshot_senders.insert(sender.name.clone()); }
                if receiver.name != "_" { self.oneshot_receivers.insert(receiver.name.clone()); }
                self.has_streams = true;
            }
        }
        // Detect `let tx, rx = broadcast<T>(cap)`.
        let is_broadcast = matches!(&s.value.kind, ExprKind::GenericCall(callee, _, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"))
            || matches!(&s.value.kind, ExprKind::Call(callee, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"));
        if is_broadcast {
            if let (Some(sender), Some(receiver)) = (s.bindings.first(), s.bindings.get(1)) {
                if sender.name != "_" { self.broadcast_senders.insert(sender.name.clone()); }
                if receiver.name != "_" { self.broadcast_receivers.insert(receiver.name.clone()); }
                self.has_streams = true;
            }
        }
        // Detect `let tx, rx = watch<T>(initial)`.
        let is_watch = matches!(&s.value.kind, ExprKind::GenericCall(callee, _, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "watch"))
            || matches!(&s.value.kind, ExprKind::Call(callee, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "watch"));
        if is_watch {
            if let (Some(sender), Some(receiver)) = (s.bindings.first(), s.bindings.get(1)) {
                if sender.name != "_" { self.watch_senders.insert(sender.name.clone()); }
                if receiver.name != "_" { self.watch_receivers.insert(receiver.name.clone()); }
                self.has_streams = true;
            }
        }
        // For `let T tx, rx = channel(n)`, emit with explicit type from the binding annotation.
        let val = if is_channel_typed {
            let item_ty = s.bindings.first()
                .and_then(|b| b.ty.as_ref())
                .map(|t| self.emit_type(t))
                .unwrap_or_else(|| "_".to_string());
            let cap = if let ExprKind::Call(_, args) = &s.value.kind {
                args.first().map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "0".to_string())
            } else { "0".to_string() };
            let channel_mod = match self.config.threading {
                crate::transpiler::ThreadingMode::Single => {
                    self.uses_local_channel.set(true);
                    "local_channel::mpsc"
                }
                crate::transpiler::ThreadingMode::Multi  => "tokio::sync::mpsc",
            };
            // local_channel::mpsc::channel() is unbounded — no capacity argument.
            if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                format!("{}::channel::<{}>()", channel_mod, item_ty)
            } else {
                format!("{}::channel::<{}>({})", channel_mod, item_ty, cap)
            }
        } else {
            self.emit_expr(&s.value)
        };
        // Build binding patterns.
        // Rust does not support per-slot type annotations in tuple destructure patterns
        // (`let (a: T, b: U) = ...` is invalid).  Type information is only used on the
        // RHS (e.g. channel::<T>()) — drop it from the LHS and let type inference work.
        let bindings: Vec<String> = s.bindings.iter().map(|b| {
            if b.name == "_" { "_".into() }
            else {
                let mut_kw = if s.binding.is_mutable() { "mut " } else { "" };
                format!("{}{}", mut_kw, b.name)
            }
        }).collect();
        // Channel/broadcast/watch receiver must be `mut`; oneshot receiver is consumed once (no mut).
        let bindings_s = if (is_channel || is_broadcast || is_watch) && bindings.len() == 2 {
            format!("{}, mut {}", bindings[0], bindings[1])
        } else {
            bindings.join(", ")
        };
        self.line(&format!("let ({}) = {};", bindings_s, val));
        // Track optional_vars for tuple destructure: if the RHS function returns a Tuple,
        // mark bindings whose element type is Optional so they aren't double-wrapped in Some().
        if let ExprKind::Call(callee, _) = &s.value.kind {
            if let ExprKind::Var(fn_name) = &callee.kind {
                if let Some(Type::Tuple(elem_tys)) = self.fn_return_types.get(fn_name.as_str()).cloned() {
                    for (i, binding) in s.bindings.iter().enumerate() {
                        if binding.name == "_" { continue; }
                        if let Some(ty) = elem_tys.get(i) {
                            if matches!(ty, Type::Optional(_)) {
                                self.optional_vars.insert(binding.name.clone());
                            }
                        }
                    }
                }
            }
        }
        // Also handle if-expression RHS: scan tuple branches to detect Optional fields.
        // e.g. `let (a, b) = if cond: (x, nil) elif ...: (x, some(y)) else: (x, nil)`
        // → `b` is Optional.
        if let ExprKind::If(if_stmt) = &s.value.kind {
            fn tuple_elem_is_optional(body: &[Stmt], idx: usize) -> bool {
                match body.last() {
                    Some(Stmt::Expr(e)) => match &e.kind {
                        ExprKind::Tuple(elems) => {
                            if let Some(elem) = elems.get(idx) {
                                matches!(&elem.kind, ExprKind::Nil)
                                    || matches!(&elem.kind, ExprKind::Call(callee, _)
                                        if matches!(&callee.kind, ExprKind::Var(v) if v == "some"))
                            } else { false }
                        }
                        _ => false,
                    },
                    _ => false,
                }
            }
            for (i, binding) in s.bindings.iter().enumerate() {
                if binding.name == "_" { continue; }
                let is_opt = if_stmt.branches.iter().any(|(_, b)| tuple_elem_is_optional(b, i))
                    || if_stmt.else_body.as_ref().map(|b| tuple_elem_is_optional(b, i)).unwrap_or(false);
                if is_opt {
                    self.optional_vars.insert(binding.name.clone());
                }
            }
        }
        // Track optional_vars from explicit binding type annotations (e.g. `let (string a, Type? b) = v`).
        for binding in s.bindings.iter() {
            if binding.name == "_" { continue; }
            if let Some(ty) = &binding.ty {
                if matches!(ty, Type::Optional(_)) {
                    self.optional_vars.insert(binding.name.clone());
                }
            }
        }
    }

    /// Destructure of a call to a `fn_returns_resident_tuple` function (see
    /// `emit_let_destructure`'s dispatch to this). A binding at a resident tuple
    /// position stays chained (registered in `resident_call_vars`) only with an
    /// *explicit* `'gpu'unified`/`'gpu'global` opt-in annotation on that one binding
    /// — the opposite default from the single-value interprocedural case (which
    /// stays resident with *no* annotation). Tuple destructuring predates this
    /// residency feature everywhere in real code, so the default has to be
    /// "materialize right here through a temp binding", matching what every
    /// existing unannotated `let (a, b, c) = some_tuple_fn(...)` already assumed —
    /// see `Checker::check_let_destructure`'s doc for the real `cargo check`
    /// failure (`test_math_gpu.br`) that an opt-*out* default caused. The call
    /// itself must run exactly once, so this can't reuse `materialize_resident_call`
    /// (which re-embeds the whole call expression per element); it destructures
    /// once into per-position temps, then downloads only the ones that didn't opt in.
    fn emit_resident_tuple_destructure(&mut self, s: &LetDestructureStmt, flags: &[bool]) {
        let elem_tys: Vec<Type> = match &s.value.kind {
            ExprKind::Call(callee, _) => match &callee.kind {
                ExprKind::Var(fn_name) => match self.fn_return_types.get(fn_name.as_str()) {
                    Some(Type::Tuple(tys)) => tys.clone(),
                    _ => Vec::new(),
                },
                _ => Vec::new(),
            },
            _ => Vec::new(),
        };
        let val = self.emit_expr(&s.value);
        let needs_materialize: Vec<bool> = s.bindings.iter().enumerate().map(|(i, b)| {
            flags.get(i).copied().unwrap_or(false)
                && !b.ty.as_ref().map(|t| t.gpu_resident_qual().is_some()).unwrap_or(false)
        }).collect();
        let names: Vec<String> = s.bindings.iter().enumerate().map(|(i, b)| {
            if b.name == "_" { "_".to_string() }
            else if needs_materialize[i] { format!("__resid_tmp_{}", i) }
            else {
                let mut_kw = if s.binding.is_mutable() { "mut " } else { "" };
                format!("{}{}", mut_kw, b.name)
            }
        }).collect();
        self.line(&format!("let ({}) = {};", names.join(", "), val));
        for (i, b) in s.bindings.iter().enumerate() {
            if b.name == "_" { continue; }
            if needs_materialize[i] {
                let elem_ty = elem_tys.get(i).cloned().unwrap_or_else(|| Type::Array(Box::new(Type::Float)));
                let inner_ty = match &elem_ty {
                    Type::Qualified(inner, _) => super::emit_kernel::array_inner_type(inner),
                    other => super::emit_kernel::array_inner_type(other),
                };
                let host_ty = super::emit_kernel::kernel_host_element_type(&inner_ty);
                let device_ty = super::emit_kernel::kernel_host_scalar_type(&inner_ty);
                let tmp = format!("__resid_tmp_{}", i);
                let mut_kw = if s.binding.is_mutable() { "mut " } else { "" };
                self.line(&format!(
                    "let {mut_kw}{name} = match {tmp} {{ BoringGpuArg::Resident(buf, _) => __boring_gpu_copy_d2h::<{device_ty}>(&__boring_gpu_device(), &__boring_gpu_queue(), &buf).iter().map(|&x| x as {host_ty}).collect::<Vec<{host_ty}>>(), BoringGpuArg::Host(v) => v }};",
                    name = b.name
                ));
            } else if flags.get(i).copied().unwrap_or(false) {
                let elem_ty = elem_tys.get(i).cloned().unwrap_or_else(|| Type::Array(Box::new(Type::Float)));
                self.resident_call_vars.insert(b.name.clone(), elem_ty);
            }
        }
    }
}
