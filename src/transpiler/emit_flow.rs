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
    pub(crate) fn emit_return(&mut self, s: &ReturnStmt) {
        match &s.value {
            // Bare `return` in a throws function must return Ok(()) to satisfy Result type.
            None => if self.in_throws {
                self.line("return Ok(());")
            } else {
                self.line("return;")
            },
            Some(e) => {
                let is_optional_return = matches!(&self.fn_return_ty, Some(Type::Optional(_)));
                // Check if the declared return type is a known trait → wrap value in Box::new().
                let _is_trait_return = matches!(&self.fn_return_ty, Some(Type::Named(n))
                    if self.trait_method_names.contains_key(n.as_str()));
                let val = if is_optional_return && is_bare_pop_call(e) {
                    // Bare `return arr.pop()` with a declared `T?` return: pass
                    // `Vec::pop()`'s `Option<T>` straight through raw — skip
                    // map_method's default `.unwrap_or_default()` and don't
                    // re-wrap in `Some(...)` below (see emit_stmt.rs's tail-return
                    // twin of this check, and map_method's `want_raw_option` doc).
                    self.want_raw_option_pop.set(true);
                    let raw = self.emit_expr_owned(e);
                    self.want_raw_option_pop.set(false);
                    raw
                } else if is_optional_return && !self.is_option_expr(e) {
                    let inner = self.emit_expr_owned(e);
                    // Check if the return expression is already an Option.
                    let already_opt = inner.starts_with("Some(") || inner == "None"
                        || is_try_optional(e)
                        || matches!(&e.kind, ExprKind::Var(v) if self.optional_vars.contains(v.as_str()))
                        || matches!(&e.kind, ExprKind::Var(v) if self.var_types.get(v.as_str())
                            .map(|t| matches!(t, Type::Optional(_))).unwrap_or(false))
                        || matches!(&e.kind, ExprKind::Call(callee, _)
                            if matches!(&callee.kind, ExprKind::Var(fn_name)
                                if self.fn_return_types.get(fn_name.as_str())
                                    .map(|t| matches!(t, Type::Optional(_))).unwrap_or(false)))
                        || (inner.ends_with("?") && matches!(&e.kind, ExprKind::Call(_, _)))
                        || matches!(&e.kind, ExprKind::MethodCall(recv, method, _)
                            if matches!(&recv.kind, ExprKind::Var(v)
                                if self.var_struct_types.get(v.as_str()).map(|sty| {
                                    self.struct_method_return_types
                                        .get(&format!("{}::{}", sty, method))
                                        .map(|t| matches!(t, Type::Optional(_)))
                                        .unwrap_or(false)
                                }).unwrap_or(false)));
                    if already_opt { inner } else { format!("Some({})", inner) }
                } else if matches!(&self.fn_return_ty, Some(Type::Array(_)))
                    && !self.in_req_fn
                    && self.self_type.is_some()
                    && matches!(&e.kind, ExprKind::Field(obj, _) if matches!(&obj.kind, ExprKind::Var(v) if v == "self"))
                {
                    // `def` method returning `self.field` where field is Vec<T>: moving out of
                    // `&mut self` is rejected by Rust. Use std::mem::take to drain the field.
                    if let ExprKind::Field(_, field_name) = &e.kind {
                        format!("std::mem::take(&mut self.{})", field_name)
                    } else {
                        self.emit_let_value(self.fn_return_ty.as_ref(), e)
                    }
                } else if matches!(&self.fn_return_ty,
                    Some(Type::Tuple(_)) | Some(Type::Array(_)) | Some(Type::Dict(_, _)))
                {
                    // Tuple/Array/Dict return: use emit_let_value for per-element
                    // coercion (e.g. string literals → Arc<str> in the right slots).
                    self.emit_let_value(self.fn_return_ty.as_ref(), e)
                } else if let Some(ret_ty) = &self.fn_return_ty.clone() {
                    // Managed mode T' return: wrap in Arc<Mutex<T>> or RefCell<T>
                    if self.is_managed_owned_user(ret_ty) {
                        let inner = self.emit_expr_owned(e);
                        if inner.starts_with("Arc::new(std::sync::Mutex::new(")
                            || inner.starts_with("RefCell::new(")
                        {
                            inner
                        } else {
                            self.wrap_managed(&inner)
                        }
                    } else if matches!(ret_ty, Type::Qualified(_, OwnerQual::Owned | OwnerQual::New)) {
                        // Strict mode T'new / T' return: wrap in Box::new().
                        let inner = self.emit_expr_owned(e);
                        if inner.starts_with("Box::new(") { inner } else { format!("Box::new({})", inner) }
                    } else if matches!(&e.kind, ExprKind::DotIdent(_)) {
                        // `return .Left` — resolve the enum shorthand against the function's
                        // own declared return type instead of falling through to
                        // emit_expr_owned's flat "last enum registered wins" DotIdent lookup,
                        // which can silently pick the wrong enum when variant names collide.
                        self.emit_let_value(Some(ret_ty), e)
                    } else {
                        self.emit_expr_owned(e)
                    }
                } else {
                    self.emit_expr_owned(e)
                };
                // For trait return types the signature already uses `-> impl Trait`
                // (static dispatch), so the concrete value must be returned as-is — no
                // boxing.  Boxing would produce `Box<ConcreteType>` which does NOT
                // automatically implement `Trait` unless there is an explicit blanket impl.
                if self.in_throws {
                    self.line(&format!("return Ok({});", val));
                } else {
                    self.line(&format!("return {};", val));
                }
            }
        }
    }

    pub(crate) fn emit_throw(&mut self, s: &ThrowStmt) {
        // In an async try body (`async { }.await`), Rust infers the async block's return type
        // from the *first* `return Err(...)` it sees — before it reaches `Ok(())` at the end.
        // If that expression is `Box::new(BoringError::...)`, the inferred type becomes
        // `Result<_, Box<BoringError>>`.  The `?` operator then fails because it needs
        // `From<Box<dyn Error>>` for `Box<BoringError>`, which is not implemented.
        //
        // Fix: when inside an async try body, add `as Box<dyn std::error::Error + Send + Sync>` INSIDE the
        // `Err(...)` argument so the async block return type is forced to `Result<_, Box<dyn Error>>`.
        let dyn_cast = if self.in_try_body && self.in_async {
            " as Box<dyn std::error::Error + Send + Sync>"
        } else {
            ""
        };
        // Helper: build `return Err(BOX_EXPR[dyn_cast]);`
        // The dyn_cast (` as Box<dyn std::error::Error + Send + Sync>`) goes after the Box::new(...) call
        // so Rust unifies the async block return type as Result<_, Box<dyn Error>>.
        let make_err = |box_expr: &str| -> String {
            format!("return Err({}{});", box_expr, dyn_cast)
        };
        // If not in a throws function, generate panic! instead of Err — the function
        // signature is `-> T` not `-> Result<T, _>`, so Err would be a type error.
        let not_throws = !self.in_throws && !self.in_try_body;
        if not_throws {
            match &s.value {
                None => { self.line("panic!(\"throw\");"); return; }
                Some(e) => match &e.kind {
                    ExprKind::Str(msg) => {
                        self.line(&format!("panic!(\"{}\");", escape_str(msg)));
                        return;
                    }
                    ExprKind::StringInterp(segs) => {
                        let s = self.emit_interp(segs);
                        self.line(&format!("panic!(\"{{}}\", {});", s));
                        return;
                    }
                    _ => {
                        let msg = self.emit_expr(e);
                        self.line(&format!("panic!(\"{{}}\", {});", msg));
                        return;
                    }
                }
            }
        }
        match &s.value {
            // Bare `throw` inside a catch block — re-throw the original error as-is.
            // `error` is Box<dyn Error>, so just forward it.
            None => self.line(&format!("return Err(error){};", dyn_cast)),
            Some(e) => {
                match &e.kind {
                    ExprKind::Int(n) =>
                        self.line(&make_err(&format!("Box::new(BoringError::Int({}))", n))),
                    ExprKind::Float(f) => {
                        let fs = {
                            let s = format!("{}", f);
                            if s.contains('.') || s.contains('e') || s.contains('E') { s }
                            else { format!("{}.0", s) }
                        };
                        self.line(&make_err(&format!("Box::new(BoringError::Float({}))", fs)));
                    }
                    ExprKind::Bool(b) =>
                        self.line(&make_err(&format!("Box::new(BoringError::Bool({}))", b))),
                    // String literal: &'static str — zero allocation.
                    ExprKind::Str(s) =>
                        self.line(&make_err(&format!("Box::new(BoringError::Str(\"{}\"))", escape_str(s)))),
                    // String interpolation: dynamic — heap-allocate via Arc<str>.
                    ExprKind::StringInterp(segs) => {
                        let s = self.emit_interp(segs);
                        // emit_interp returns `"literal"` (&str) or `format!(...)` (String).
                        let arc_inner = if s.starts_with('"') {
                            format!("{}.to_string()", s)
                        } else {
                            s
                        };
                        self.line(&make_err(&format!("Box::new(BoringError::String({}::<str>::from({})))", self.str_ptr(), arc_inner)));
                    }
                    _ => {
                        // Fixed-width scalar throws (int8..int128, uint8..uint128,
                        // float32, float64) route through BoringError::Scalar
                        // (docs/float-width-types.md §7) — checked before the
                        // is_typed_error/String fallback below, which would otherwise
                        // stringify the value via BoringError::String, losing the type
                        // information `catch Int8:`/`catch Float32:` need to dispatch on.
                        let scalar_ctor = crate::transpiler::helpers::infer_overload_expr_type(
                            e, &self.var_types, &self.fn_return_types, &self.struct_fields,
                        ).as_ref().and_then(crate::transpiler::helpers::scalar_ctor_name);
                        if let Some(ctor) = scalar_ctor {
                            let val = self.emit_expr(e);
                            self.line(&make_err(&format!("Box::new(BoringError::{}({}))", ctor, val)));
                            return;
                        }
                        // Constructor / enum variant calls implement Error directly via the
                        // `typed_error_enums` path — use .into() for those.
                        // Everything else (variables, call results) wraps as BoringError::Str.
                        let is_typed_error = match &e.kind {
                            // Enum variant without args: `throw FileError.NotFound`
                            // Parsed as Field(Var("FileError"), "NotFound")
                            ExprKind::Field(base, _) => match &base.kind {
                                ExprKind::Var(enum_name) =>
                                    self.typed_error_enums.contains(enum_name.as_str()),
                                _ => false,
                            },
                            // Bare enum variant name: `throw NotFound` (if enum_variants resolves it)
                            ExprKind::Var(name) =>
                                self.enum_variants.get(name.as_str())
                                    .map(|enum_name| self.typed_error_enums.contains(enum_name.as_str()))
                                    .unwrap_or(false),
                            // Enum variant call with args: `throw FileError.Custom("msg")`
                            // Struct constructor call: `throw EMyError("msg")`
                            ExprKind::Call(func, _) => match &func.kind {
                                ExprKind::Field(base, _) => match &base.kind {
                                    ExprKind::Var(enum_name) =>
                                        self.typed_error_enums.contains(enum_name.as_str()),
                                    _ => false,
                                },
                                ExprKind::Var(name) =>
                                    self.typed_error_enums.contains(name.as_str())
                                    || self.enum_variants.get(name.as_str())
                                        .map(|enum_name| self.typed_error_enums.contains(enum_name.as_str()))
                                        .unwrap_or(false),
                                _ => false,
                            },
                            _ => false,
                        };
                        let val = self.emit_expr(e);
                        if is_typed_error {
                            // Determine the enum type name
                            let type_name = match &e.kind {
                                ExprKind::Field(base, _) => match &base.kind {
                                    ExprKind::Var(n) => n.clone(),
                                    _ => "Error".to_string(),
                                },
                                ExprKind::Var(name) =>
                                    self.enum_variants.get(name.as_str())
                                        .cloned()
                                        .unwrap_or_else(|| name.clone()),
                                ExprKind::Call(func, _) => match &func.kind {
                                    ExprKind::Field(base, _) => match &base.kind {
                                        ExprKind::Var(n) => n.clone(),
                                        _ => "Error".to_string(),
                                    },
                                    ExprKind::Var(n) => self.enum_variants.get(n.as_str())
                                        .cloned()
                                        .unwrap_or_else(|| n.clone()),
                                    _ => "Error".to_string(),
                                },
                                _ => "Error".to_string(),
                            };
                            self.line(&make_err(&format!(
                                "Box::new(BoringError::Other(std::any::TypeId::of::<{}>(), Box::new({}) as Box<dyn BoringVal + Send + Sync>))",
                                type_name, val
                            )));
                        } else {
                            self.line(&make_err(&format!(
                                "Box::new(BoringError::String({}::<str>::from(format!(\"{{}}\", {}))))",
                                self.str_ptr(), val
                            )));
                        }
                    }
                }
            }
        }
    }


    pub(crate) fn emit_guard(&mut self, s: &GuardStmt) {
        match &s.cond {
            GuardCond::Expr(e) => {
                let cond = self.emit_expr(e);
                self.line(&format!("if !({}) {{", cond));
                self.indent += 1;
                self.emit_body(&s.else_body);
                self.indent -= 1;
                self.line("}");
            }
            GuardCond::Clauses(clauses) => {
                // Use let-else where possible (Rust 1.65+)
                for clause in clauses {
                    match clause {
                        CondClause::Let(name, expr) => {
                            // Track the inner type of the optional — after the guard let,
                            // `name` has the unwrapped type (e.g. string? → string).
                            if let ExprKind::Var(src) = &expr.kind {
                                if let Some(Type::Optional(inner)) = self.var_types.get(src.as_str()).cloned() {
                                        self.var_types.insert(name.clone(), *inner.clone());
                                        if Self::is_string_type(&inner) {
                                            self.string_vars.insert(name.clone());
                                        }
                                        if matches!(*inner, Type::Optional(_)) {
                                            self.optional_vars.insert(name.clone());
                                        }
                                        // Propagate the struct type too (mirrors `emit_let.rs`'s
                                        // `var_struct_types` tracking) so a field write through
                                        // `name` can resolve the struct and be checked below —
                                        // otherwise `emit_expr.rs`'s field-write diagnostic falls
                                        // back to `var_types`, which already has `Type::Named`
                                        // from the `insert` above, so this is belt-and-suspenders.
                                        if let Type::Named(n) = inner.as_ref() {
                                            if self.is_known_user_type(n.as_str()) {
                                                self.var_struct_types.insert(name.clone(), n.clone());
                                            }
                                        }
                                }
                            } else if let ExprKind::Call(callee, _) = &expr.kind {
                                // `guard let b = make()` where `make()` returns `T?` — mirror the
                                // `Var` branch above using the callee's declared return type
                                // instead of a variable's tracked type.
                                if let ExprKind::Var(fn_name) = &callee.kind {
                                    if let Some(Type::Optional(inner)) = self.fn_return_types.get(fn_name.as_str()).cloned() {
                                        self.var_types.insert(name.clone(), *inner.clone());
                                        if Self::is_string_type(&inner) {
                                            self.string_vars.insert(name.clone());
                                        }
                                        if let Type::Named(n) = inner.as_ref() {
                                            if self.is_known_user_type(n.as_str()) {
                                                self.var_struct_types.insert(name.clone(), n.clone());
                                            }
                                        }
                                    }
                                }
                            }
                            // A `guard let` binding has no `mut`/`var mut` spelling — the parser's
                            // `parse_cond_clause` only ever consumes a bare `let` for `CondClause::
                            // Let` (see `ast::CondClause` doc) — so it is never content-mutable.
                            // Register it as *checked* so `emit_expr.rs`'s field-write and
                            // `emit_methods.rs`'s `def`-call diagnostics actually fire for it,
                            // instead of silently no-op'ing and letting invalid Rust reach rustc
                            // (E0594) further down the pipeline. Mirrors `emit_let.rs`'s
                            // unconditional `mut_checked_local_vars.insert` for a plain `let`.
                            self.content_mutable_local_vars.remove(name);
                            self.mut_checked_local_vars.insert(name.clone());
                            // Narrowing numeric `as` cast scrutinee (docs/known-issues-
                            // biguint-spike.md #11): see the identical check in
                            // `emit_cond_clauses` (emit_match.rs) for the full rationale --
                            // `guard let` has the exact same `let Some(name) = val else {...}`
                            // shape and needs the same checked, Option-producing codegen.
                            let val = self.try_emit_checked_int_cast_as_option(expr)
                                .unwrap_or_else(|| self.emit_expr(expr));
                            // Parenthesize unconditionally: Rust's `let-else` grammar rejects a
                            // block-like initializer ending directly in `}` right before `else`
                            // (ambiguous with the `else` binding to the initializer's own
                            // trailing `if`/`match`/block) — e.g. `readLine()` transpiles to an
                            // inline `{ ...; if ... { None } else { Some(...) } }` block, which
                            // hit exactly this without parens. Wrapping is valid Rust for any
                            // expression shape, so do it unconditionally rather than pattern-
                            // matching which shapes need it.
                            self.line(&format!("let Some({}) = ({}) else {{", name, val));
                            self.known_local_vars.insert(name.clone());
                            self.indent += 1;
                            self.emit_body(&s.else_body);
                            self.indent -= 1;
                            self.line("};");
                        }
                        CondClause::LetPat(pat, expr) => {
                            let pat_s = self.emit_pattern(pat);
                            let val = self.emit_expr(expr);
                            Self::collect_pattern_binds(pat, &mut self.known_local_vars);
                            // See the `CondClause::Let` arm just above for why this is
                            // unconditionally parenthesized.
                            self.line(&format!("let {} = ({}) else {{", pat_s, val));
                            self.indent += 1;
                            self.emit_body(&s.else_body);
                            self.indent -= 1;
                            self.line("};");
                        }
                        CondClause::Expr(e) => {
                            let cond = self.emit_expr(e);
                            self.line(&format!("if !({}) {{", cond));
                            self.indent += 1;
                            self.emit_body(&s.else_body);
                            self.indent -= 1;
                            self.line("}");
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn emit_try(&mut self, s: &TryStmt) {
        // try: { body } catch: { handler }
        // Emitted as a closure that propagates throws via `?`, then an if-let on the result.
        self.line("{");
        self.indent += 1;
        // In an async context, the try body may contain `.await` calls, so emit as
        // `async { ... }.await` instead of a synchronous closure.
        if self.in_async {
            self.line("let __try_result: Result<(), Box<dyn std::error::Error + Send + Sync>> = async {");
        } else {
            self.line("let __try_result: Result<(), Box<dyn std::error::Error + Send + Sync>> = (|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
        }
        self.indent += 1;
        let prev_throws   = self.in_throws;
        let prev_try_body = self.in_try_body;
        self.in_throws   = true;
        self.in_try_body = true;
        // All body statements are non-last (side-effect only); throws calls get `?`.
        for stmt in &s.body {
            self.emit_stmt(stmt, false);
        }
        self.in_throws   = prev_throws;
        self.in_try_body = prev_try_body;
        // In an async try body, explicitly annotate the Ok type so that Rust unifies all
        // `return Err(Box::new(BoringError::...))` returns with `Box<dyn std::error::Error + Send + Sync>`.
        // Without this, Rust may infer `Box<BoringError>` from the throw expressions and then
        // reject the `?` operators (which expect `Box<dyn std::error::Error + Send + Sync>`).
        if self.in_async {
            self.line("Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())");
        } else {
            self.line("Ok(())");
        }
        self.indent -= 1;
        if self.in_async {
            self.line("}.await;");
        } else {
            self.line("})();");
        }

        if s.catch_clauses.is_empty() {
            // No catch: re-propagate if we're already in a throws context
            self.line("__try_result?;");
        } else {
            // `error` is implicitly bound in the catch body
            self.known_local_vars.insert("error".to_string());

            // Determine whether any clause is typed.
            let has_typed = s.catch_clauses.iter().any(|c| !c.types.is_empty());
            let untyped_clause = s.catch_clauses.iter().find(|c| c.types.is_empty());
            let typed_clauses: Vec<_> = s.catch_clauses.iter().filter(|c| !c.types.is_empty()).collect();

            if !has_typed {
                // ── Simple untyped catch ──────────────────────────────────────
                // Bind `error` as Box<dyn Error> — the original thrown value.
                // String interpolation {error} works via Display.
                // match error: with typed enum variants works via match_pattern.
                // Re-throw (`throw` bare) just forwards `error` as-is.
                self.line("if let Err(__err) = __try_result {");
                self.indent += 1;
                self.line("let error = __err;");
                let clause = &s.catch_clauses[0];
                self.emit_loop_body(&clause.body);
                self.indent -= 1;
                self.line("}");
            } else {
                // ── Typed catch dispatch ─────────────────────────────────────
                // Primitive types (String/Int/Float/Bool) → BoringError prim arms.
                // Named error types (enums) → BoringError::Other guard arms.
                // Unmatched → untyped fallback body or propagate.
                //
                // Code shape:
                //   if let Err(__err) = __try_result {
                //     [Phase A: BoringError match → Option<Box<dyn Error>>]
                //     [if let Some(__err) = __err {]
                //       [Phase C: unmatched → return Err / untyped body]
                //     [}]
                //   }
                const PRIM_TYPES: &[&str] = &[
                    "String", "string", "cstring", "tstring",
                    "Int", "int", "Float", "float", "Bool", "bool",
                    "Uint", "uint", "Uint8", "uint8",
                    "Int8", "int8", "Int16", "int16", "Int32", "int32", "Int64", "int64", "Int128", "int128",
                    "Uint16", "uint16", "Uint32", "uint32", "Uint64", "uint64", "Uint128", "uint128",
                    // The two float widths route through BoringError::Scalar, same as the
                    // fixed-width ints above (docs/float-width-types.md §7) — `Float`/`float`
                    // (bare) above stays on its own BoringError::Float fast path, unaffected.
                    "Float32", "float32", "f32", "Float64", "float64", "f64",
                ];
                let prim_clauses: Vec<_> = typed_clauses.iter()
                    .filter(|c| c.types.iter().any(|t| PRIM_TYPES.contains(&t.as_str())))
                    .collect();
                let named_clauses: Vec<_> = typed_clauses.iter()
                    .filter(|c| c.types.iter().any(|t| !PRIM_TYPES.contains(&t.as_str())))
                    .collect();
                let untyped_body = untyped_clause.map(|uc| uc.body.clone());

                self.line("if let Err(__err) = __try_result {");
                self.indent += 1;

                // ── Phase A: BoringError (prims + named via Other) ──────────────
                if !prim_clauses.is_empty() || !named_clauses.is_empty() {
                    self.line("let __err: Option<Box<dyn std::error::Error + Send + Sync>> = match __err.downcast::<BoringError>() {");
                    self.indent += 1;
                    self.line("Ok(__bv) => match *__bv {");
                    self.indent += 1;
                    // Prim arms — String yields two arms (Str + String), others yield one.
                    for clause in &prim_clauses {
                        let body_stmts = clause.body.clone();
                        for ty_name in &clause.types {
                            if PRIM_TYPES.contains(&ty_name.as_str()) {
                                for (arm_pat, error_ty, error_bind) in boring_type_to_boring_val_arms(ty_name) {
                                    self.line(&format!("{} => {{", arm_pat));
                                    self.indent += 1;
                                    self.line(&format!("let error: {} = {};", error_ty, error_bind));
                                    self.emit_loop_body(&body_stmts);
                                    self.line("None");
                                    self.indent -= 1;
                                    self.line("}");
                                }
                            }
                        }
                    }
                    // Named arms via BoringError::Other guard.
                    // Split into plain catches (`catch Error:`) and variant catches (`catch Error.Expired:`).
                    let plain_named: Vec<_> = named_clauses.iter()
                        .filter(|c| c.variant.is_none())
                        .collect();
                    let variant_named: Vec<_> = named_clauses.iter()
                        .filter(|c| c.variant.is_some())
                        .collect();

                    // Plain named: for typed error enums bind `error` as `&TypeName` so that
                    // `match error: Error.Expired: …` works directly in the catch body.
                    // For non-enum types keep `error: Arc<str>` (string representation).
                    for clause in &plain_named {
                        let body_stmts = clause.body.clone();
                        for ty_name in &clause.types {
                            if !PRIM_TYPES.contains(&ty_name.as_str()) {
                                let is_enum = self.typed_error_enums.contains(ty_name.as_str());
                                self.line(&format!(
                                    "BoringError::Other(ref __tid, ref __boring_err) if *__tid == std::any::TypeId::of::<{}>() => {{",
                                    ty_name
                                ));
                                self.indent += 1;
                                if is_enum {
                                    // `error` is the typed enum reference: supports both
                                    // `{error}` (Display) and `match error: Variant: …` dispatch.
                                    self.line(&format!(
                                        "let error = (**__boring_err).as_any().downcast_ref::<{}>().unwrap();",
                                        ty_name
                                    ));
                                } else {
                                    self.line(&format!("let error: {p}<str> = {p}::<str>::from(__boring_err.to_string());", p = self.str_ptr()));
                                }
                                let prev_error_concrete = self.error_var_is_concrete_enum;
                                self.error_var_is_concrete_enum = is_enum;
                                self.emit_loop_body(&body_stmts);
                                self.error_var_is_concrete_enum = prev_error_concrete;
                                self.line("None");
                                self.indent -= 1;
                                self.line("}");
                            }
                        }
                    }

                    // Variant named: group by enum type, emit a single arm with inner match.
                    // Unhandled variants are re-thrown via `return Err(...)`.
                    {
                        // Build ordered groups: Vec<(type_name, Vec<(variant, body)>)>
                        #[allow(clippy::type_complexity)]
                        let mut variant_groups: Vec<(String, Vec<(String, Vec<crate::ast::Stmt>)>)> = Vec::new();
                        for clause in &variant_named {
                            let ty_name = clause.types.first().cloned().unwrap_or_default();
                            let variant_name = clause.variant.clone().unwrap_or_default();
                            let body = clause.body.clone();
                            if let Some(grp) = variant_groups.iter_mut().find(|(n, _)| n == &ty_name) {
                                grp.1.push((variant_name, body));
                            } else {
                                variant_groups.push((ty_name, vec![(variant_name, body)]));
                            }
                        }
                        // At this point self.in_throws reflects the enclosing function context
                        // (restored from prev_throws after the try body was emitted).
                        let rethrow_unhandled = if self.in_throws {
                            |ty: &str| format!(
                                "__unhandled => return Err(Box::new(BoringError::Other(std::any::TypeId::of::<{}>(), Box::new(__unhandled.clone()) as Box<dyn BoringVal + Send + Sync>)) as Box<dyn std::error::Error + Send + Sync>),",
                                ty
                            )
                        } else {
                            |ty: &str| format!(
                                "__unhandled => {{ eprintln!(\"[boring] unhandled {} variant: {{}}\", __unhandled); panic!(\"unhandled {} variant\"); }},",
                                ty, ty
                            )
                        };
                        for (ty_name, arms) in &variant_groups {
                            self.line(&format!(
                                "BoringError::Other(ref __tid, ref __boring_err) if *__tid == std::any::TypeId::of::<{}>() => {{",
                                ty_name
                            ));
                            self.indent += 1;
                            // Double-deref: __boring_err is &Box<dyn BoringVal + Sync + Send>,
                            // so (**__boring_err) gives dyn BoringVal — the vtable dispatch
                            // correctly reaches the concrete type's as_any(), not Box's.
                            self.line(&format!(
                                "match (**__boring_err).as_any().downcast_ref::<{}>().unwrap() {{",
                                ty_name
                            ));
                            self.indent += 1;
                            for (variant_name, body_stmts) in arms {
                                self.line(&format!("{}::{} => {{", ty_name, variant_name));
                                self.indent += 1;
                                self.emit_loop_body(body_stmts);
                                self.line("None");
                                self.indent -= 1;
                                self.line("}");
                            }
                            // Re-throw (or panic) for any variant not explicitly caught.
                            self.line(&rethrow_unhandled(ty_name));
                            self.indent -= 1;
                            self.line("}");
                            self.indent -= 1;
                            self.line("}");
                        }
                    }
                    self.line("__other => Some(Box::new(__other) as Box<dyn std::error::Error + Send + Sync>),");
                    self.indent -= 1;
                    self.line("},");
                    // ── Phase A′: direct downcast for non-BoringError errors ────────
                    // Errors from native Rust code (e.g. thiserror types propagated via ?)
                    // are NOT wrapped in BoringError. Try downcast<T> directly for each
                    // named catch type.  Pattern: chain via `let __orig = match downcast {
                    //   Ok(__e) => { body; None }  Err(__b) => Some(__b) }` so that each
                    // failed attempt hands the box to the next attempt untouched.
                    // Phase A' only applies to plain named catches (not variant catches,
                    // which always come through BoringError::Other and are handled above).
                    let plain_named_for_apost: Vec<_> = named_clauses.iter()
                        .filter(|c| c.variant.is_none())
                        .collect();
                    if !plain_named_for_apost.is_empty() {
                        self.line("Err(__orig) => {");
                        self.indent += 1;
                        self.line("let __orig: Option<Box<dyn std::error::Error + Send + Sync>> = Some(__orig);");
                        for clause in &plain_named_for_apost {
                            let body_stmts = clause.body.clone();
                            for ty_name in &clause.types {
                                if !PRIM_TYPES.contains(&ty_name.as_str()) {
                                    self.line("let __orig = if let Some(__b) = __orig {");
                                    self.indent += 1;
                                    self.line(&format!("match __b.downcast::<{}>() {{", ty_name));
                                    self.indent += 1;
                                    self.line("Ok(__e) => {");
                                    self.indent += 1;
                                    let is_enum = self.typed_error_enums.contains(ty_name.as_str());
                                    if is_enum {
                                        // `error` as typed ref → `match error: Variant:` works.
                                        self.line("let error = &*__e;");
                                    } else {
                                        self.line(&format!("let error: {p}<str> = {p}::<str>::from(__e.to_string());", p = self.str_ptr()));
                                    }
                                    let prev_error_concrete = self.error_var_is_concrete_enum;
                                    self.error_var_is_concrete_enum = is_enum;
                                    self.emit_loop_body(&body_stmts);
                                    self.error_var_is_concrete_enum = prev_error_concrete;
                                    self.line("None");
                                    self.indent -= 1;
                                    self.line("}");
                                    self.line("Err(__b) => Some(__b),");
                                    self.indent -= 1;
                                    self.line("}");
                                    self.indent -= 1;
                                    self.line("} else { None };");
                                }
                            }
                        }
                        self.line("__orig");
                        self.indent -= 1;
                        self.line("}");
                    } else {
                        self.line("Err(__orig) => Some(__orig),");
                    }
                    self.indent -= 1;
                    self.line("};");
                    // Everything from here runs only if the error was NOT handled by Phase A.
                    self.line("if let Some(__err) = __err {");
                    self.indent += 1;
                }

                // ── Phase C: Unmatched ────────────────────────────────────────
                // __err: Box<dyn Error> (inside Phase A wrapper when Phase A ran, otherwise original)
                if let Some(body_stmts) = &untyped_body {
                    // __err: Box<dyn Error>
                    self.line(&format!("{{ let error: {p}<str> = {p}::<str>::from(__err.to_string());", p = self.str_ptr()));
                    self.indent += 1;
                    self.emit_loop_body(body_stmts);
                    self.indent -= 1;
                    self.line("}");
                } else {
                    // __err: Box<dyn Error>, not matched — propagate if throws, panic otherwise.
                    if self.in_throws {
                        self.line("return Err(__err);");
                    } else {
                        self.line("eprintln!(\"[boring] unhandled error: {}\", __err);");
                        self.line("panic!(\"unhandled error\");");
                    }
                }

                // Close Phase A wrapper
                if !prim_clauses.is_empty() || !named_clauses.is_empty() {
                    self.indent -= 1;
                    self.line("}"); // end if let Some(__err) from Phase A
                }

                self.indent -= 1;
                self.line("}"); // end if let Err(__err)
            }
        }
        self.indent -= 1;
        self.line("}");
    }

    pub(crate) fn emit_defer(&mut self, _body: &[Stmt]) {
        // Defer is handled in emit_body: deferred stmts are collected and emitted
        // in LIFO order before the function's return value. This fallback should
        // never be reached for well-formed boring programs (defer only inside fns).
    }
}
