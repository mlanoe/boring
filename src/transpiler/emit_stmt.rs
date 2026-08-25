use super::*;
use super::Transpiler;
use super::helpers::*;

impl Transpiler {
    pub(crate) fn emit_stmt(&mut self, stmt: &Stmt, is_last: bool) {
        match stmt {
            Stmt::Let(s)            => { self.emit_let(s, false); self.maybe_emit_str_index_cache(&s.name); }
            Stmt::LetDestructure(s) => self.emit_let_destructure(s),
            Stmt::Return(s)         => self.emit_return(s),
            Stmt::Throw(s)          => self.emit_throw(s),
            Stmt::Break(_, val) => {
                match val {
                    Some(e) => self.line(&format!("break {};", self.emit_expr(e))),
                    None    => self.line("break;"),
                }
            }
            Stmt::Continue(_)       => self.line("continue;"),
            Stmt::If(s)             => self.emit_if(s, is_last),
            Stmt::IfLet(s)          => self.emit_if_let(s, is_last),
            Stmt::Match(s)          => self.emit_match(s, is_last),
            Stmt::While(s)          => self.emit_while(s),
            Stmt::WhileLet(s)       => self.emit_while_let(s),
            Stmt::DoWhile(s)        => self.emit_do_while(s),
            Stmt::Loop(s)           => self.emit_loop(s),
            Stmt::Wait(dur, _)      => {
                // Resolve leading-dot syntax: `.fromSecs(1)` → `Duration::from_secs(1)`
                // Also detect Instant vs Duration for sleep_until vs sleep dispatch.
                let is_instant = expr_is_instant(dur, &self.instant_vars);
                let type_prefix = if is_instant { "Instant" } else { "Duration" };
                let d = if let Some(resolved) = self.resolve_dot_with_type(dur, type_prefix) {
                    resolved
                } else {
                    let raw = self.emit_expr(dur);
                    // Strip .await suffix — wait takes a Duration, not a future.
                    if let Some(s) = raw.strip_suffix(".await?") { s.to_string() }
                    else if let Some(s) = raw.strip_suffix(".await") { s.to_string() }
                    else { raw }
                };
                if is_instant {
                    self.line(&format!("tokio::time::sleep_until({}).await;", d));
                } else {
                    self.line(&format!("tokio::time::sleep({}).await;", d));
                }
            }
            Stmt::For(s)            => {
                // `for g in GPU.all():` — track the loop var as a GPU-device handle
                // for the body's duration (see `gpu_device_vars`'s doc comment),
                // restored via a snapshot here rather than threading extra
                // bookkeeping through emit_for's several iterable-shape branches.
                let is_gpu_all = self.is_gpu_target && s.vars.len() == 1 && matches!(
                    &s.iterable.kind,
                    ExprKind::MethodCall(obj, method, _)
                        if method == "all" && matches!(&obj.kind, ExprKind::Var(v) if v == "GPU")
                );
                if is_gpu_all {
                    let saved_gpu_device_vars = self.gpu_device_vars.clone();
                    self.gpu_device_vars.insert(s.vars[0].clone());
                    self.emit_for(s);
                    self.gpu_device_vars = saved_gpu_device_vars;
                } else {
                    self.emit_for(s);
                }
            }
            Stmt::Guard(s)          => self.emit_guard(s),
            Stmt::Try(s)            => self.emit_try(s),
            Stmt::Defer(body)       => self.emit_defer(body),
            Stmt::Fn(f)             => {
                // If we're inside a function body and the nested `def` captures any outer
                // local variable, emit it as a closure `let name = move |params| { body }`
                // rather than a nested `fn name(params)` (which cannot capture from env).
                if self.indent > 0 && self.nested_fn_captures_outer(f) {
                    self.emit_nested_fn_as_closure(f);
                } else {
                    self.emit_fn(f, None);
                }
                self.blank();
            }
            Stmt::Struct(s)         => { self.emit_struct(s); self.blank(); }
            Stmt::Enum(e)           => {
                // Register variants so infer_match_enum can find them (overrides same-named globals).
                for v in &e.variants {
                    self.enum_variants.insert(v.name.clone(), e.name.clone());
                    let key = format!("{}::{}", e.name, v.name);
                    let field_names: Vec<Option<String>> = v.fields.iter().map(|f| f.name.clone()).collect();
                    let field_types: Vec<Type> = v.fields.iter().map(|f| match &f.ty {
                        Type::Qualified(inner, OwnerQual::Owned) => *inner.clone(),
                        other => other.clone(),
                    }).collect();
                    self.enum_variant_fields.insert(key.clone(), field_names);
                    self.enum_variant_field_types.insert(key, field_types);
                }
                self.emit_enum(e);
                self.blank();
            }
            Stmt::Mod(m)            => { self.emit_mod(m); self.blank(); }
            Stmt::Alias(a)          => self.emit_alias(a),
            Stmt::Yield(expr, _)    => {
                // Use emit_expr_owned so string literals are Arc<str>, not &str,
                // matching the stream's Item type.
                let val = self.emit_expr_owned(expr);
                if self.in_iter_stream {
                    self.line(&format!("__items.push({});", val));
                } else {
                    self.line(&format!("yield {};", val));
                }
            }
            Stmt::Comment(text)     => self.line(&format!("// {}", text)),
            Stmt::Expr(e)           => {
                if is_last && self.in_throws && !self.suppress_ok_wrap && self.fn_declared_void {
                    // Void throws function: emit the expression as a statement (for side-effects/?)
                    // then Ok(()). Skip nil entirely (no side-effect).
                    if !matches!(&e.kind, ExprKind::Nil) {
                        let s = format!("{};", self.emit_expr_owned(e));
                        self.line(&s);
                    }
                    self.line("Ok(())");
                } else if is_last && self.in_throws && !self.suppress_ok_wrap {
                    // Interprocedural GPU residency: every real kernel-launcher wrapper in
                    // practice declares `throws` (kernel dispatch can fail), so this branch —
                    // not the non-throws one below — is the one that actually needs the
                    // `try_emit_gpu_resident_return` check. Missing it here meant a `throws`
                    // function with a `'gpu'unified`/`'gpu'global` return type and a bare
                    // `k.field` tail expression still got an eager `Ok(k.copy_y_to_host()...)`
                    // download wrapped in `Ok(...)`, which doesn't even type-check against the
                    // function's own `Result<BoringGpuArg<T>, _>` return type — confirmed via a
                    // real `cargo check` against every `throws`-declared function in this file.
                    if let Some(resident) = self.try_emit_gpu_resident_return(e) {
                        self.line(&format!("Ok({resident})"));
                        return;
                    }
                    // Tuple counterpart: this function's return type is a resident
                    // tuple (`mha_step_gpu`-style) and the tail is a tuple literal.
                    if let Some(resident) = self.try_emit_gpu_resident_tuple_return(e) {
                        self.line(&format!("Ok({resident})"));
                        return;
                    }
                    // This function's own return type is NOT itself resident (the case
                    // above already handled that, via a raw pass-through that "just
                    // works" since both sides are `BoringGpuArg<T>`) — but the tail
                    // expression may still be a bare call to a `fn_returns_resident`
                    // function (e.g. `Ok(linear_gpu(normed, ...))` inside a plain
                    // `[float]`-returning function). Materialize instead of letting
                    // `Ok(...)` wrap a `BoringGpuArg<T>` where `Vec<T>` is expected.
                    if self.current_fn_returns_resident.is_none() {
                        if let Some(materialized) = self.try_materialize_resident_call(e) {
                            self.line(&format!("Ok({materialized})"));
                            return;
                        }
                    }
                    let s = format!("Ok({})", self.emit_expr_owned(e));
                    self.line(&s);
                } else if is_last && !self.fn_returns_void {
                    // Interprocedural GPU residency (docs/scoped-access-blocks.md): this
                    // function's declared return type is `'gpu'unified`/`'gpu'global` and
                    // the tail expression is a bare `k.field` read — emit
                    // `BoringGpuArg::Resident(...)` instead of the unconditional
                    // `copy_field_to_host()` download the normal expression emitter would
                    // otherwise produce via `try_emit_kernel_field_read`.
                    if let Some(resident) = self.try_emit_gpu_resident_return(e) {
                        self.line(&resident);
                        return;
                    }
                    // Tuple counterpart: this function's return type is a resident
                    // tuple (`mha_step_gpu`-style) and the tail is a tuple literal.
                    if let Some(resident) = self.try_emit_gpu_resident_tuple_return(e) {
                        self.line(&resident);
                        return;
                    }
                    // Same materializing fallback as the `throws` branch above, for a
                    // non-throws function whose tail is a bare call to a
                    // `fn_returns_resident` function without this function's own
                    // return type opting into residency.
                    if self.current_fn_returns_resident.is_none() {
                        if let Some(materialized) = self.try_materialize_resident_call(e) {
                            self.line(&materialized);
                            return;
                        }
                    }
                    // Expression return: no semicolon (value-returning function).
                    // If the function returns Option<T>, wrap non-nil, non-Some values in Some().
                    let is_optional_return = matches!(
                        &self.fn_return_ty,
                        Some(Type::Optional(_))
                    );
                    let s = if is_optional_return && is_bare_pop_call(e) {
                        // Bare `arr.pop()` tail call with a declared `T?` return: pass
                        // `Vec::pop()`'s `Option<T>` straight through raw (skip
                        // map_method's default `.unwrap_or_default()`, and don't
                        // re-wrap in `Some(...)` below — it's already Option-shaped).
                        self.want_raw_option_pop.set(true);
                        let raw = self.emit_expr_owned(e);
                        self.want_raw_option_pop.set(false);
                        raw
                    } else if is_optional_return && !self.is_option_expr(e) {
                        // Function returns Option<T>; expression is not already Option-typed.
                        // Wrap scalar/integer/variable values in Some() so Rust is happy.
                        // Use emit_expr_owned so string literals become Arc::<str>::from("...")
                        // rather than &str, which would mismatch Option<Arc<str>>.
                        let raw = self.emit_expr_owned(e);
                        let already_opt = raw == "None" || raw.starts_with("Some(")
                            || is_try_optional(e)
                            || matches!(&e.kind, ExprKind::Var(v)
                                if self.optional_vars.contains(v.as_str())
                                || self.var_types.get(v.as_str())
                                    .map(|t| matches!(t, Type::Optional(_))).unwrap_or(false))
                            || matches!(&e.kind, ExprKind::Call(callee, _)
                                if matches!(&callee.kind, ExprKind::Var(fn_name)
                                    if self.fn_return_types.get(fn_name.as_str())
                                        .map(|t| matches!(t, Type::Optional(_))).unwrap_or(false)))
                            || matches!(&e.kind, ExprKind::MethodCall(recv, method, _)
                                if matches!(&recv.kind, ExprKind::Var(v)
                                    if self.var_struct_types.get(v.as_str()).map(|sty| {
                                        self.struct_method_return_types
                                            .get(&format!("{}::{}", sty, method))
                                            .map(|t| matches!(t, Type::Optional(_)))
                                            .unwrap_or(false)
                                    }).unwrap_or(false)))
                            // Field read of a `T?`-declared field, or a call/method-call whose
                            // own declared return type is `T?` (any receiver shape, not just a
                            // bare var — see `expr_is_declared_optional`'s doc for why the
                            // MethodCall check above alone isn't enough, e.g. `items[0].as_str()`).
                            // See docs/option-return-double-some-wrap-bug.md.
                            || self.expr_is_declared_optional(e)
                            // If-expression whose branches already produce Option (nil/some/method)
                            || matches!(&e.kind, ExprKind::If(if_stmt) if {
                                fn branch_ends_optional(body: &[Stmt]) -> bool {
                                    match body.last() {
                                        Some(Stmt::Expr(e)) => matches!(&e.kind, ExprKind::Nil)
                                            || matches!(&e.kind, ExprKind::Call(callee, _)
                                                if matches!(&callee.kind, ExprKind::Var(v) if v == "some"))
                                            || matches!(&e.kind, ExprKind::MethodCall(_, _, _)),
                                        _ => false,
                                    }
                                }
                                if_stmt.branches.iter().any(|(_, b)| branch_ends_optional(b))
                                    || if_stmt.else_body.as_ref().map(|b| branch_ends_optional(b)).unwrap_or(false)
                            });
                        if already_opt {
                            raw
                        } else {
                            format!("Some({})", raw)
                        }
                    } else if matches!(&self.fn_return_ty,
                        Some(Type::Tuple(_)) | Some(Type::Array(_)) | Some(Type::Dict(_, _)))
                    {
                        // Tuple/Array/Dict return: use emit_let_value for per-element
                        // coercion (string literals → Arc<str> in the right slots).
                        self.emit_let_value(self.fn_return_ty.as_ref(), e)
                    } else {
                        self.emit_expr_owned(e)
                    };
                    self.line(&s);
                } else {
                    // Void function or non-last: always emit as a statement with semicolon.
                    // Skip nil (None) as a standalone statement — it's a no-op placeholder.
                    if !matches!(&e.kind, ExprKind::Nil) {
                        let s = self.emit_expr(e);
                        self.line(&format!("{};", s));
                    }
                }
            }
            Stmt::KernelBlock(s) => {
                // GPU targets only (see emit_kernel.rs): `k(block=.., grid=..)` dispatches
                // a tracked kernel variable instead of the naive body passthrough below.
                if let Some(code) = self.try_emit_kernel_dispatch(s) {
                    self.line(&code);
                    return;
                }
                self.line("// kernel: block");
                let body = s.body.clone();
                let last = body.len().saturating_sub(1);
                for (i, stmt) in body.iter().enumerate() {
                    self.emit_stmt(stmt, i == last);
                }
            }
            Stmt::With(s) => self.emit_with(s),
        }
    }

    /// `with <name> [, <name> ...]:` — scoped access block (docs/scoped-access-blocks.md).
    ///
    /// For `'actor`/`'actor'task`/`'guard`/`'guard'task` names (supported on every host
    /// target, including plain `boring build`, since these qualifiers already work there
    /// today): acquires the lock once, shadowing `name` with the guard for the block's
    /// duration, instead of the per-call `.lock()`/`.read()`/`.write()` that ordinary
    /// method/field codegen on `name` would otherwise emit on every access. Because a
    /// lock guard auto-derefs to the inner value, the shadowed binding lets the block's
    /// body be emitted with completely ordinary (plain-struct-receiver) codegen — so
    /// `name` is temporarily removed from the actor/guard tracking sets for exactly the
    /// duration of this block, then restored, rather than teaching every call site in
    /// emit_methods.rs/emit_expr.rs a new "suppressed" case.
    ///
    /// For a `'gpu'unified`/`'gpu'global` name registered in `gpu_resident_vars`
    /// (`let py'gpu'unified = k.y` — see `emit_kernel::try_emit_gpu_resident_let`):
    /// materializes it to a plain host `Vec` exactly once via `k.copy_y_to_host()`
    /// (the same conversion `emit_kernel::try_emit_kernel_field_read` already does
    /// for a bare `k.y`), regardless of how many times the block's body indexes it —
    /// which is the actual round-trip-per-access problem this whole feature exists to
    /// fix (`examples/vector_add_gpu.br`'s `for i in 0..n: print k.result[i]` reads
    /// the whole buffer back on *every* iteration today). Write-back (`copy_y_to_device`)
    /// happens once at block close, only if the body's own mutation scan finds an
    /// index-assignment into it.
    ///
    /// Unqualified names fall through unhandled here — the block's body is still
    /// emitted (see the shared `{ }` wrapper and `emit_loop_body` call below), just
    /// without any acquire/write-back codegen, which is the correct no-op degradation.
    /// See docs/scoped-access-blocks.md, "Cross-target behavior".
    pub(crate) fn emit_with(&mut self, s: &WithStmt) {
        self.line("{");
        self.indent += 1;

        struct Opened { name: String, was_mutex: bool, was_mutex_task: bool, was_rwlock: bool, was_rwlock_task: bool }
        let mut opened: Vec<Opened> = Vec::new();

        struct GpuOpened { name: String, kernel_var: String, field: String, is_write: bool, kernel_scalar_ty: &'static str }
        let mut gpu_opened: Vec<GpuOpened> = Vec::new();

        // Interprocedural counterpart to `GpuOpened`/`gpu_resident_vars` above: a
        // resident value returned across a function-call boundary (`resident_call_vars`,
        // `let fc = linear_gpu(...)`) has no kernel instance left to call
        // `copy_{field}_to_{host,device}` on — it was dropped when that function
        // returned. Materialization/write-back goes through the free d2h/h2d helpers
        // directly on the retained `Arc<wgpu::Buffer>` plus the global device/queue
        // accessors instead. See docs/scoped-access-blocks.md's interprocedural case.
        struct ResidentCallOpened { name: String, is_write: bool, device_ty: &'static str }
        let mut resident_call_opened: Vec<ResidentCallOpened> = Vec::new();

        let saved_locals = self.known_local_vars.clone();
        let saved_var_types = self.var_types.clone();

        for name in &s.names {
            if let Some(ty) = self.resident_call_vars.get(name.as_str()).cloned() {
                let inner_ty = match &ty {
                    Type::Qualified(inner, _) => super::emit_kernel::array_inner_type(inner),
                    other => super::emit_kernel::array_inner_type(other),
                };
                let host_ty = super::emit_kernel::kernel_host_element_type(&inner_ty);
                let device_ty = super::emit_kernel::kernel_host_scalar_type(&inner_ty);

                let mut is_var_param = |_: &str, _: usize| false;
                let mut is_mutating_method = |_: &str, _: &str| false;
                let is_write = crate::ast::with_block_mutates(&s.body, name, &mut is_var_param, &mut is_mutating_method);

                if is_write {
                    self.line(&format!(
                        "let __{name}_buf = match &{name} {{ BoringGpuArg::Resident(buf, _) => Some(std::sync::Arc::clone(buf)), BoringGpuArg::Host(_) => None }};"
                    ));
                }
                self.line(&format!(
                    "let {}{name} = match &{name} {{ BoringGpuArg::Resident(buf, _) => __boring_gpu_copy_d2h::<{device_ty}>(&__boring_gpu_device(), &__boring_gpu_queue(), buf).iter().map(|&x| x as {host_ty}).collect::<Vec<{host_ty}>>(), BoringGpuArg::Host(v) => v.clone() }};",
                    if is_write { "mut " } else { "" },
                ));
                self.var_types.insert(name.clone(), Type::Array(Box::new(inner_ty)));
                self.known_local_vars.insert(name.clone());
                self.with_open_names.insert(name.clone());
                resident_call_opened.push(ResidentCallOpened { name: name.clone(), is_write, device_ty });
                continue;
            }
            if let Some((kvar, field)) = self.gpu_resident_vars.get(name.as_str()).cloned() {
                let field_ty = self.kernel_vars.get(kvar.as_str())
                    .and_then(|kname| self.kernel_decls.get(kname))
                    .and_then(|decl| decl.fields.iter().find(|f| f.name == field))
                    .map(|f| f.ty.clone());
                // The checker only ever allows this shape when `kvar` is a real tracked
                // kernel var with a matching field (`Binding::resident_from_field` +
                // `try_emit_gpu_resident_let`'s own `kernel_vars` check) — but stay
                // defensive rather than panicking if that invariant is ever violated.
                let Some(field_ty) = field_ty else { continue };
                let inner_ty = super::emit_kernel::array_inner_type(&field_ty);
                let host_ty = super::emit_kernel::kernel_host_element_type(&inner_ty);
                let kernel_scalar_ty = super::emit_kernel::kernel_host_scalar_type(&inner_ty);

                // GPU arrays have no `def`/`req` methods to scan for — only a direct
                // index-assignment (`name[i] = v`) can mutate one, which the shared
                // scan already detects via its assignment-target check.
                let mut is_var_param = |_: &str, _: usize| false;
                let mut is_mutating_method = |_: &str, _: &str| false;
                let is_write = crate::ast::with_block_mutates(&s.body, name, &mut is_var_param, &mut is_mutating_method);

                self.line(&format!(
                    "let {}{} = {}.copy_{}_to_host().iter().map(|&x| x as {}).collect::<Vec<{}>>();",
                    if is_write { "mut " } else { "" }, name, kvar, field, host_ty, host_ty,
                ));
                self.var_types.insert(name.clone(), Type::Array(Box::new(inner_ty)));
                self.known_local_vars.insert(name.clone());
                self.with_open_names.insert(name.clone());
                gpu_opened.push(GpuOpened { name: name.clone(), kernel_var: kvar, field, is_write, kernel_scalar_ty });
                continue;
            }

            let is_mutex = self.var_mutex_types.contains(name.as_str());
            let is_mutex_task = self.var_mutex_task_types.contains(name.as_str());
            let is_rwlock = self.var_rwlock_types.contains(name.as_str());
            let is_rwlock_task = self.var_rwlock_task_types.contains(name.as_str());
            if !(is_mutex || is_mutex_task || is_rwlock || is_rwlock_task) {
                continue;
            }

            // Two-step hybrid access scan (ast::with_block_mutates) — signature-only
            // lookups, never opening a called function/method's body. `let`-bound names
            // never satisfy the scan in valid Boring code (mutating them would already
            // require a rejected `def` call), so no separate binding-kind check is needed.
            let struct_name = self.var_struct_types.get(name.as_str()).cloned();
            let req_methods = self.struct_req_methods.clone();
            let fn_var_params = self.fn_var_params.clone();
            let mut is_var_param = |fn_name: &str, idx: usize| {
                fn_var_params.get(fn_name).and_then(|v| v.get(idx)).copied().unwrap_or(false)
            };
            let mut is_mutating_method = |recv: &str, method: &str| {
                if recv != name.as_str() { return false; }
                match &struct_name {
                    Some(sn) => !req_methods.contains(&format!("{}::{}", sn, method)),
                    None => true, // unknown struct — conservative: assume mutating
                }
            };
            let is_write = crate::ast::with_block_mutates(&s.body, name, &mut is_var_param, &mut is_mutating_method);

            let guard_expr = if is_mutex || is_mutex_task {
                // Mutex has one mode regardless of read/write — only method-call
                // legality (def vs req) differs, per docs/scoped-access-blocks.md.
                self.mutex_var_write(name, name)
            } else if is_rwlock_task {
                if is_write { self.guard_task_write_guard(name) } else { self.guard_task_read_access(name) }
            } else if is_write {
                self.guard_write_guard(name)
            } else {
                self.guard_read_access(name)
            };
            let needs_mut = is_mutex || is_mutex_task || is_write;
            self.line(&format!("let {}{} = {};", if needs_mut { "mut " } else { "" }, name, guard_expr));

            opened.push(Opened { name: name.clone(), was_mutex: is_mutex, was_mutex_task: is_mutex_task, was_rwlock: is_rwlock, was_rwlock_task: is_rwlock_task });
            self.var_mutex_types.remove(name.as_str());
            self.var_mutex_task_types.remove(name.as_str());
            self.var_rwlock_types.remove(name.as_str());
            self.var_rwlock_task_types.remove(name.as_str());
            self.with_open_names.insert(name.clone());
        }

        self.emit_loop_body(&s.body);

        for o in opened.into_iter().rev() {
            if o.was_mutex { self.var_mutex_types.insert(o.name.clone()); }
            if o.was_mutex_task { self.var_mutex_task_types.insert(o.name.clone()); }
            if o.was_rwlock { self.var_rwlock_types.insert(o.name.clone()); }
            if o.was_rwlock_task { self.var_rwlock_task_types.insert(o.name.clone()); }
            self.with_open_names.remove(&o.name);
        }

        for o in &gpu_opened {
            if o.is_write {
                self.line(&format!(
                    "{}.copy_{}_to_device(&{}.iter().map(|&x| x as {}).collect::<Vec<{}>>());",
                    o.kernel_var, o.field, o.name, o.kernel_scalar_ty, o.kernel_scalar_ty,
                ));
            }
            self.with_open_names.remove(&o.name);
        }

        for o in &resident_call_opened {
            if o.is_write {
                self.line(&format!(
                    "if let Some(buf) = &__{name}_buf {{ __boring_gpu_copy_h2d(&__boring_gpu_device(), &__boring_gpu_queue(), bytemuck::cast_slice(&{name}.iter().map(|&x| x as {device_ty}).collect::<Vec<{device_ty}>>()), buf); }}",
                    name = o.name, device_ty = o.device_ty,
                ));
            }
            self.with_open_names.remove(&o.name);
        }
        // `name` was a pure block-local alias (no Rust binding predates this `with`) —
        // restore the outer scope's view exactly like a loop body would, so it doesn't
        // leak into surrounding code as a spuriously "already known" local/type.
        self.known_local_vars = saved_locals;
        self.var_types = saved_var_types;

        self.indent -= 1;
        self.line("}");
    }

    /// After a `let` binding is emitted, additionally materialize a `Vec<char>` shadow
    /// (`__strchars_<name>`) when `name` is both indexed elsewhere in the function
    /// (`str_index_cache_vars`, from `collect_str_index_targets`) and truly immutable
    /// (`immutable_local_vars` -- `let`-bound, never `mut`/`var`, so it can't go stale).
    /// See `collect_str_index_targets` for the full rationale.
    pub(crate) fn maybe_emit_str_index_cache(&mut self, name: &str) {
        if self.str_index_cache_vars.contains(name) && self.immutable_local_vars.contains(name) {
            self.line(&format!("let __strchars_{name}: Vec<char> = {name}.chars().collect();"));
        }
    }


    /// Returns true if a nested `def` function body captures any variable from the outer scope.
    pub(crate) fn nested_fn_captures_outer(&self, f: &FnDecl) -> bool {
        // Collect all param names (they are NOT captures).
        let param_names: std::collections::HashSet<&str> = f.params.iter()
            .map(|p| p.name.as_str())
            .collect();
        // Collect all variable references in the function body.
        let mut body_vars: Vec<String> = Vec::new();
        for stmt in &f.body {
            collect_vars_in_stmt(stmt, &mut body_vars);
        }
        // A variable is captured if it's in outer known_local_vars but not a param of this fn.
        body_vars.iter().any(|v| {
            !param_names.contains(v.as_str()) && self.known_local_vars.contains(v.as_str())
        })
    }

    /// Emit a nested `def` function that captures outer variables as a `let` closure.
    pub(crate) fn emit_nested_fn_as_closure(&mut self, f: &FnDecl) {
        // Build param list: `name: Type` or just `name` if untyped.
        let params: Vec<String> = f.params.iter().map(|p| {
            if let Some(ty) = &p.ty {
                format!("{}: {}", p.name, self.emit_type(ty))
            } else {
                p.name.clone()
            }
        }).collect();
        let params_str = params.join(", ");
        // Return type annotation.
        let ret_ty = f.return_ty.as_ref()
            .map(|t| format!(" -> {}", self.emit_type(t)))
            .unwrap_or_default();
        // Register this name as a local var so subsequent code can call it.
        self.known_local_vars.insert(f.name.clone());
        // Emit the closure.
        self.line(&format!("let {} = move |{}|{} {{", f.name, params_str, ret_ty));
        self.indent += 1;
        // Emit body — set up context like emit_fn does.
        let prev_in_throws = self.in_throws;
        let prev_fn_return_ty = self.fn_return_ty.clone();
        let prev_fn_returns_void = self.fn_returns_void;
        self.in_throws = f.throws;
        self.fn_return_ty = f.return_ty.clone();
        self.fn_returns_void = f.return_ty.is_none() || matches!(&f.return_ty, Some(Type::Void));
        let body_len = f.body.len();
        for (i, stmt) in f.body.iter().enumerate() {
            self.emit_stmt(stmt, i + 1 == body_len);
        }
        self.in_throws = prev_in_throws;
        self.fn_return_ty = prev_fn_return_ty;
        self.fn_returns_void = prev_fn_returns_void;
        self.indent -= 1;
        self.line("};");
    }

    pub(crate) fn is_string_type(ty: &Type) -> bool {
        let ty = ty.without_mut();
        matches!(ty, Type::Str)
            || matches!(ty, Type::Named(n) if n == "string")
            || matches!(ty, Type::Qualified(inner, _) if matches!(**inner, Type::Str))
    }

    /// Returns true for types that map to `Arc<Mutex<T>>` or `Arc<RwLock<T>>` in Rust.
    /// Does NOT include `Shared` — handled separately because `Shared` is threading-aware.
    pub(crate) fn is_arc_qualified(ty: &Type) -> bool {
        matches!(ty.without_mut(), Type::Qualified(_, OwnerQual::Actor | OwnerQual::Guard))
    }

    /// Returns true if `value` is a variable whose qualifier is 'owned (Box<T>).
    /// Used at call sites to emit *x dereference instead of x.clone() when wrapping in Rc/Arc.
    pub(crate) fn arg_is_heap_var(&self, value: &Expr) -> bool {
        let ExprKind::Var(v) = &value.kind else { return false };
        if let Some(q) = self.inferred_qualifiers.get(v.as_str()) {
            return q.is_owned_or_new();
        }
        if let Some(ty) = self.var_types.get(v.as_str()) {
            return matches!(ty.without_mut(), Type::Qualified(_, q) if q.is_owned_or_new());
        }
        false
    }

    /// Returns true for `T'shared` (Arc<T> multi or Rc<T> single).
    pub(crate) fn is_rc_qualified(ty: &Type) -> bool {
        matches!(ty.without_mut(), Type::Qualified(_, OwnerQual::Shared))
    }

    /// Returns true when `ty` is an anonymous `T'` (OwnerQual::Owned) and the inner type
    /// is a user-defined struct/enum. In managed mode this resolves to Arc<std::sync::Mutex<T>>
    /// (multi) or RefCell<T> (single) rather than Box<T>.
    pub(crate) fn is_managed_owned_user(&self, ty: &Type) -> bool {
        crate::transpiler::Transpiler::is_managed_user_owned(
            &self.config, &self.user_types, &self.unit_enums, ty)
    }

/// Wrap a raw constructor value in the managed-mode wrapper.
    /// Multi: `Arc::new(std::sync::Mutex::new(val))`
    /// Single: `RefCell::new(val)`
    pub(crate) fn wrap_managed(&self, val: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi =>
                format!("Arc::new(std::sync::Mutex::new({}))", val),
            crate::transpiler::ThreadingMode::Single =>
                format!("RefCell::new({})", val),
        }
    }

    /// Infer whether an expression's result type is a managed-mode owned user type.
    /// Returns true when the result would be `Arc<Mutex<T>>` (multi) or `RefCell<T>` (single).
    /// Used to auto-track untyped `let` bindings in managed mode.
    pub(crate) fn infers_as_managed(&self, expr: &Expr) -> bool {
        use crate::ast::ExprKind;
        use crate::ast::BinOp;
        if self.config.mode != crate::transpiler::TranspileMode::Managed { return false; }
        match &expr.kind {
            // Direct constructor call of a user type with T' return: if the result is used
            // in an untyped binding and the called type is a user type, check if it's the
            // top-level binding — but we can't know without explicit type, so skip plain calls.
            ExprKind::BinOp(op, l, _r) => {
                // a op b where op is an arithmetic/comparison operator dispatched to a struct method
                let mname = match op {
                    BinOp::Add => "add", BinOp::Sub => "sub", BinOp::Mul => "mul",
                    BinOp::Div => "div", BinOp::Rem => "rem",
                    BinOp::Eq => "eq", BinOp::NotEq => "ne",
                    BinOp::Lt => "lt", BinOp::LtEq => "le", BinOp::Gt => "gt", BinOp::GtEq => "ge",
                    _ => return false,
                };
                // Determine struct type of left operand
                let struct_ty = if let ExprKind::Var(v) = &l.kind {
                    self.var_struct_types.get(v.as_str()).cloned()
                        .or({
                            // Also handle plain (non-tracked) vars whose type we know from constructor
                            None
                        })
                } else { None };
                if let Some(sty) = struct_ty {
                    let key = format!("{}::{}", sty, mname);
                    if let Some(ret_ty) = self.struct_method_return_types.get(&key) {
                        return self.is_managed_owned_user(ret_ty);
                    }
                }
                false
            }
            ExprKind::MethodCall(obj, method, _args) => {
                let struct_ty = if let ExprKind::Var(v) = &obj.kind {
                    self.var_struct_types.get(v.as_str()).cloned()
                } else { None };
                if let Some(sty) = struct_ty {
                    let key = format!("{}::{}", sty, method);
                    if let Some(ret_ty) = self.struct_method_return_types.get(&key) {
                        return self.is_managed_owned_user(ret_ty);
                    }
                }
                false
            }
            ExprKind::UnaryOp(crate::ast::UnaryOp::Neg, operand) => {
                // `-a` where `a` is a struct with `neg()` method returning managed type
                let struct_ty = if let ExprKind::Var(v) = &operand.kind {
                    self.var_struct_types.get(v.as_str()).cloned()
                } else { None };
                if let Some(sty) = struct_ty {
                    let key = format!("{}::neg", sty);
                    if let Some(ret_ty) = self.struct_method_return_types.get(&key) {
                        return self.is_managed_owned_user(ret_ty);
                    }
                }
                false
            }
            ExprKind::Call(callee, _args) => {
                if let ExprKind::Var(fn_name) = &callee.kind {
                    // Function returning managed type
                    if let Some(ret_ty) = self.fn_return_types.get(fn_name.as_str()) {
                        if self.is_managed_owned_user(ret_ty) { return true; }
                    }
                    // Non-function type alias constructor: `AP` → `APoint'`
                    if let Some(alias_ty) = self.non_fn_type_aliases.get(fn_name.as_str()) {
                        if self.is_managed_owned_user(alias_ty) { return true; }
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Returns true for `T'weak` (any form) which maps to `Weak<T>` in Rust.
    pub(crate) fn is_weak_qualified(ty: &Type) -> bool {
        matches!(ty.without_mut(), Type::Qualified(_, OwnerQual::Weak))
    }

    /// Returns true for `T'shared'weak` / `T'actor'weak` — weak ref to an Arc-backed type.
    /// These require `Arc::downgrade` and `std::sync::Weak<T>` in Rust.
    pub(crate) fn is_arc_weak(ty: &Type) -> bool {
        matches!(ty.without_mut(),
            Type::Qualified(inner, OwnerQual::Weak)
            if matches!(inner.as_ref(), Type::Qualified(_, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard))
        )
    }

    pub(crate) fn is_str_ref_type(ty: &Type) -> bool {
        let ty = ty.without_mut();
        matches!(ty, Type::Named(n) if n == "str")
            || matches!(ty, Type::Qualified(inner, OwnerQual::Inline) if matches!(**inner, Type::Str))
    }

    /// Returns true if the expression produces a string (Arc<str>) value.
    pub(crate) fn is_string_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Str(_) | ExprKind::StringInterp(_) => true,
            ExprKind::Var(v) => self.string_vars.contains(v.as_str())
                || self.string_arc_vars.contains(v.as_str()),
            ExprKind::BinOp(BinOp::Add, l, r) => self.is_string_expr(l) || self.is_string_expr(r),
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(name) = &callee.kind {
                    // readLine() now returns Option<Arc<str>>, not a bare string.
                    matches!(name.as_str(), "str" | "string")
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// True when `expr` is positively known to be a `[float]`/`[float32]`/`[float64]`
    /// array — used to pick the right turbofish type for `.sum()` (see emit_methods.rs),
    /// returning which width ("f32"/"f64") so `.sum::<T>()` is correct for a `[float32]`
    /// receiver instead of always assuming 64-bit (docs/float-width-types.md —
    /// float32/float64 are distinct runtime types, `Sum<f32>` isn't implemented for
    /// `f64` or vice versa). Conservatively `None` (not "definitely an int array") for
    /// anything not confidently identifiable, since the caller's default (`i64`) is the
    /// historically-safe choice for those cases.
    pub(crate) fn float_array_elem_ty(&self, expr: &Expr) -> Option<&'static str> {
        fn float_width(t: &Type) -> Option<&'static str> {
            match t {
                Type::Float64 => Some("f64"),
                Type::Float32 => Some("f32"),
                Type::Named(n) => match n.as_str() {
                    "float" | "float64" | "f64" => Some("f64"),
                    "float32" | "f32" => Some("f32"),
                    _ => None,
                },
                _ => None,
            }
        }
        match &expr.kind {
            ExprKind::Var(v) => match self.var_types.get(v.as_str()) {
                Some(Type::Array(inner)) => float_width(inner),
                _ => None,
            },
            ExprKind::Array(elems) => elems.first().and_then(|e| match &e.kind {
                ExprKind::Float(_) => Some("f64"),
                _ => None,
            }),
            _ => None,
        }
    }

    /// Collect all leaf parts of a string concatenation chain `a + b + c` into a flat Vec.
    /// Each part is emitted as a raw expression string (no Arc::new wrapping).
    pub(crate) fn collect_string_parts(&self, expr: &Expr, parts: &mut Vec<String>) {
        if let ExprKind::BinOp(BinOp::Add, l, r) = &expr.kind {
            if self.is_string_expr(l) || self.is_string_expr(r) {
                self.collect_string_parts(l, parts);
                self.collect_string_parts(r, parts);
                return;
            }
        }
        // Leaf: emit as a raw expression (not wrapped in Arc::new)
        parts.push(self.emit_expr_raw_string(expr));
    }

    /// Emit a string expression as a raw value (unwrapped Arc content or literal without quotes).
    pub(crate) fn emit_expr_raw_string(&self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Str(s) => format!("\"{}\"", escape_str(s)),
            ExprKind::StringInterp(segs) => self.emit_interp(segs),
            _ => self.emit_expr(expr),
        }
    }

    /// Returns (left_rust_type, right_rust_type) if both expressions have known
    /// specific numeric types (i8/i16/i32/i64/u8/u16/u32/u64/f32/f64/isize/usize).
    /// Used for numeric type coercion in arithmetic expressions.
    pub(crate) fn get_numeric_types(&self, l: &Expr, r: &Expr) -> Option<(String, String)> {
        let l_ty = self.get_expr_rust_type(l)?;
        let r_ty = self.get_expr_rust_type(r)?;
        if is_specific_numeric_type(&l_ty) && is_specific_numeric_type(&r_ty) {
            Some((l_ty, r_ty))
        } else {
            None
        }
    }

    /// Returns the Rust type string for a simple expression (Var with known type, or a
    /// binary op result of same-type numeric ops).
    pub(crate) fn get_expr_rust_type(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Var(v) => {
                if let Some(ty) = self.var_types.get(v.as_str()) {
                    let rust_ty = match ty {
                        Type::Named(n) => normalize_type_name(n, self.use_rc_str()),
                        _ => self.emit_type(ty),
                    };
                    if is_specific_numeric_type(&rust_ty) {
                        return Some(rust_ty);
                    }
                }
                None
            }
            ExprKind::BinOp(_, l, r) => {
                // If both sides have the same type, the result has that type.
                let lt = self.get_expr_rust_type(l)?;
                let rt = self.get_expr_rust_type(r)?;
                if lt == rt { Some(lt) } else { Some(wider_numeric_type(&lt, &rt)) }
            }
            _ => None,
        }
    }



    // ── Expressions ───────────────────────────────────────────────────────────

}
