use super::*;
use super::Transpiler;
use super::helpers::*;

/// Bevy system-param/deref wrapper generics that transparently carry a real value of
/// their first type argument (`Res<T>`, `ResMut<T>`, `Mut<T>`) or of a filtered slot
/// where the first type argument is still the value type (`Single<T, Filter>`) — all
/// implement `Deref`/`DerefMut` straight through to `T`, so `res.count` at the Rust
/// level is a field access on `T` via auto-deref, not on the wrapper itself. Deliberately
/// a fixed whitelist, not "peel any `Type::Generic`'s first argument": an arbitrary
/// user-defined generic struct (`Container<Score>`) is NOT deref-transparent to its type
/// argument, and treating `container.count` as `Score`'s field would be wrong — Container
/// has its own, unrelated fields. Only wrappers Boring/Bevy itself guarantees are
/// deref-transparent belong here.
const TRANSPARENT_WRAPPER_GENERICS: &[&str] = &["Res", "ResMut", "Single", "Mut"];

/// If `ty` is one of `TRANSPARENT_WRAPPER_GENERICS`, returns the `Type::Named` name
/// carried by its first type argument (stripping one level of `mut`/borrow-qualifier,
/// e.g. `Single<mut Transform&, With<Paddle>>`'s first argument). `None` for anything
/// else, including a wrapper whose first argument isn't (or doesn't resolve to) a bare
/// named type.
fn transparent_wrapper_inner_name(ty: &Type) -> Option<&str> {
    let Type::Generic(name, args) = ty else { return None };
    if !TRANSPARENT_WRAPPER_GENERICS.contains(&name.as_str()) { return None }
    match args.first()?.without_mut() {
        Type::Named(n) => Some(n.as_str()),
        Type::Qualified(inner, OwnerQual::Borrow | OwnerQual::BorrowMut
            | OwnerQual::BorrowOption | OwnerQual::BorrowOptionMut) => {
            if let Type::Named(n) = inner.as_ref() { Some(n.as_str()) } else { None }
        }
        _ => None,
    }
}

impl Transpiler {
    /// Resolve the user struct type name of an expression, walking `self`/local-var roots
    /// and field chains (`self.encoder`, `a.b.c`). Also resolves bare implicit-self-field
    /// names (`signal` inside a method meaning `self.signal`) the same way `emit_expr`'s
    /// `ExprKind::Var` arm lowers them — otherwise callers relying on this helper see a
    /// field access as an unresolvable local, even though `emit_expr` will emit `self.<name>`
    /// for the exact same expression a few lines later. Returns None for anything else
    /// (locals of non-struct type, method-call results, etc.) — callers should treat that
    /// as "not a known user struct".
    ///
    /// Also sees through `Res<T>`/`ResMut<T>`/`Single<T, F>`/`Mut<T>` (see
    /// `TRANSPARENT_WRAPPER_GENERICS`) to `T` — a Bevy system parameter declared
    /// `Res<Score> score` resolves to `Score`, not to the unresolvable wrapper name
    /// `Res`, so `score.count` finds `Score`'s real `count` field instead of falling
    /// through to the `.count`/`.length` builtin.
    pub(crate) fn resolve_expr_struct_type(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Var(v) if v == "self" => self.self_type.clone(),
            ExprKind::Var(v) => self.var_struct_types.get(v.as_str()).cloned().or_else(|| {
                if let Some(inner) = self.var_types.get(v.as_str())
                    .and_then(transparent_wrapper_inner_name)
                {
                    return Some(inner.to_string());
                }
                if self.known_local_vars.contains(v.as_str()) { return None; }
                let struct_name = self.self_type.as_ref()?;
                let (_, fty) = self.struct_fields.get(struct_name.as_str())?
                    .iter().find(|(fname, _)| fname == v)?;
                match fty {
                    Type::Named(n) => Some(n.clone()),
                    _ => None,
                }
            }),
            ExprKind::Field(inner, field) => {
                let inner_ty = self.resolve_expr_struct_type(inner)?;
                self.struct_fields.get(inner_ty.as_str())?
                    .iter()
                    .find(|(fname, _)| fname == field)
                    .and_then(|(_, fty)| match fty {
                        Type::Named(n) => Some(n.clone()),
                        _ => None,
                    })
            }
            // `arr[i]` — resolve the array's declared element type (not `arr`'s own
            // type): `self.blocks[i]` needs `self.blocks`'s field type to be
            // `Type::Array(Named("DecoderBlock"))` to know indexing it yields a
            // DecoderBlock, one level further than the Field case above.
            ExprKind::Index(inner, _) => {
                let elem_ty = match &inner.kind {
                    ExprKind::Field(obj, field) => {
                        let owner_ty = self.resolve_expr_struct_type(obj)?;
                        self.struct_fields.get(owner_ty.as_str())?
                            .iter()
                            .find(|(fname, _)| fname == field)
                            .map(|(_, fty)| fty.clone())
                    }
                    ExprKind::Var(v) => self.var_types.get(v.as_str()).cloned(),
                    _ => None,
                }?;
                match elem_ty {
                    Type::Array(elem) => match elem.as_ref() {
                        Type::Named(n) => Some(n.clone()),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Is `name` a real user-declared type (struct or enum)? The shared "is this receiver a
    /// user type with its own members, or should builtin Iterator/collection name tables
    /// apply?" check — used by both method-call dispatch (`is_user_struct_receiver` below)
    /// and field-access dispatch (`map_field` call sites in emit_expr.rs/emit_top.rs) so a
    /// user method/field named `position`, `count`, `find`, etc. always wins over the
    /// same-named builtin adapter, on structs AND enums alike. Before this existed, every
    /// such check consulted `struct_fields` alone, so an enum receiver (which is never a key
    /// in `struct_fields`) silently fell through to the builtin tables no matter what it
    /// actually declared.
    pub(crate) fn is_known_user_type(&self, name: &str) -> bool {
        self.struct_fields.contains_key(name) || self.all_enum_types.contains(name)
    }

    /// Resolves whether `obj`'s receiver type is a known user struct/enum — i.e. whether the
    /// call should dispatch through its own declared members rather than any of the
    /// Vec/HashMap/HashSet/String/Iterator-name-only special cases (`try_emit_array_special_method`
    /// and `emit_method_call_fallback`'s `is_user_struct_receiver`). Shared between the two so a
    /// user struct/enum method named `min`/`max`/`sum`/`count`/`indexOf`/`position`/etc. is never
    /// hijacked by a same-named builtin adapter that runs earlier in the dispatch chain and, unlike
    /// `try_emit_dict_method`/`try_emit_set_method`/`try_emit_string_method`, used to have no
    /// receiver-type guard of its own at all — it fired on method name (and sometimes arg shape)
    /// alone, for any receiver.
    pub(crate) fn expr_receiver_is_known_user_type(&self, obj: &Expr) -> bool {
        match &obj.kind {
            ExprKind::Var(v) => {
                if v == "self" {
                    // Inside a method body, self_type names the current struct/enum.
                    self.self_type.as_deref()
                        .map(|t| self.is_known_user_type(t))
                        .unwrap_or(false)
                } else {
                    // `var_struct_types` is the primary source (also gates the
                    // mut/content-mutation diagnostics), but it's only reliably populated for
                    // *struct*-typed vars in some binding shapes -- an enum bound via a bare
                    // unit-variant reference (`let b = Bar.B`, parsed as `ExprKind::Field`, not
                    // `MethodCall` -- see `parse_postfix_inner`) is never routed through any of
                    // the `var_struct_types.insert` call sites, only through `var_types`
                    // (emit_let.rs's "Track enum type for variables initialized from enum
                    // constructors"). Fall back to `var_types` for exactly that case, mirroring
                    // the same struct-then-enum fallback emit_expr.rs's field-access `is_getter`
                    // check already uses.
                    let type_name = self.var_struct_types.get(v.as_str()).cloned()
                        .or_else(|| self.var_types.get(v.as_str()).and_then(|t| {
                            if let Type::Named(n) = t.without_mut() { Some(n.clone()) } else { None }
                        }));
                    type_name.map(|t| self.is_known_user_type(t.as_str())).unwrap_or(false)
                }
            }
            ExprKind::Field(..) | ExprKind::Index(..) => self.resolve_expr_struct_type(obj)
                .map(|t| self.is_known_user_type(t.as_str()))
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Returns true when an expression is likely to produce an `Option<T>` value,
    /// used to avoid wrapping Option-chain methods in `.iter().cloned().collect()`
    /// (`try_emit_option_chain_method`). Was a free function in helpers.rs; moved to a
    /// `&self` method (and given the `expr_receiver_is_known_user_type` guard below) once
    /// a user struct/enum method sharing one of these names (`and_then`, `map`, `filter`,
    /// ...) was found to be misidentified as an Option chain purely by name — e.g. a
    /// struct with its own `.and_then()` had `x.and_then(f).map(g)` treated as an Option
    /// chain regardless of `x`'s actual type.
    pub(crate) fn is_option_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::MethodCall(recv, method, _) | ExprKind::Pipe(recv, method, _) => {
                // A call whose own receiver is a known user struct/enum is never guessed to
                // be Option-shaped from name alone — its methods are real declared members,
                // not Option's, whatever they happen to be named.
                if self.expr_receiver_is_known_user_type(recv) { return false; }
                // Unambiguously Option-producing (not Vec methods):
                const OPTION_ONLY: &[&str] = &["and_then", "or_else", "flatten", "as_ref", "as_deref"];
                if OPTION_ONLY.contains(&method.as_str()) { return true; }
                // Ambiguous (exists on both Vec and Option): propagate — only treat as
                // Option if the receiver is itself Option-like.
                const MAYBE_OPTION: &[&str] = &[
                    "filter", "map", "next", "cloned", "copied",
                    "find", "first", "last", "get", "pop",
                ];
                if MAYBE_OPTION.contains(&method.as_str()) {
                    return self.is_option_expr(recv);
                }
                false
            }
            ExprKind::Var(name) => name.ends_with('?'),
            _ => false,
        }
    }

    /// Does `expected` name an enum that actually has a variant called `variant`? Used by
    /// overload resolution (here and in emit_expr.rs's free-function overload lookup) to
    /// judge a `.Variant` argument against a candidate's declared param type.
    /// `infer_overload_expr_type` has no case for `ExprKind::DotIdent` — it returns `None`,
    /// and the overload predicate's `None => true` then "optimistically" treats `.Variant`
    /// as compatible with *every* candidate, so the first one in declaration order wins
    /// regardless of whether its param type actually has that variant. This gives that
    /// predicate a real answer for the `.Variant` case instead of the optimistic default.
    pub(crate) fn dotident_matches_enum_type(&self, variant: &str, expected: &Type) -> bool {
        let base = match expected {
            Type::Named(n) => n.as_str(),
            Type::Qualified(inner, _) => match inner.as_ref() {
                Type::Named(n) => n.as_str(),
                _ => return false,
            },
            _ => return false,
        };
        self.enum_variant_fields.contains_key(&format!("{}::{}", base, variant))
    }

    /// Built-in `fs`/`GPU` module namespace calls, and property access on a tracked
    /// `GPU(n)` device handle — intercepted before any other method-call dispatch.
    fn try_emit_builtin_namespace_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        if let ExprKind::Var(v) = &obj.kind {
            if v == "fs" {
                return Some(self.emit_fs_call(method, args));
            }
            // `GPU.all()` — iterate all GPU devices. wgpu only has one real adapter
            // (see `gpu_device_vars`'s doc comment), so this is a single-element
            // array — matching the interpreter's own simulation-mode behavior.
            if v == "GPU" && method == "all" && self.is_gpu_target {
                return Some("vec![0usize]".to_string());
            }
        }
        // `g.name()`/`.totalMem()`/etc on a tracked `GPU(n)` handle (`let g = GPU(0)`,
        // or a `for g in GPU.all():` loop variable) — rewritten to the introspection
        // helpers `wgpu::host::emit_gpu_introspection_globals` emits. `.index()` is
        // just the stored usize itself; every other property has no per-device
        // notion on wgpu (one real adapter total), so the index is otherwise unused.
        if let ExprKind::Var(v) = &obj.kind {
            if self.gpu_device_vars.contains(v.as_str()) {
                return Some(match method {
                    "name"              => "__boring_gpu_name()".to_string(),
                    "totalMem"          => "__boring_gpu_total_mem()".to_string(),
                    "freeMem"           => "__boring_gpu_free_mem()".to_string(),
                    "computeCapability" => "__boring_gpu_compute_capability()".to_string(),
                    "warpSize"          => "__boring_gpu_warp_size()".to_string(),
                    "maxThreads"        => "__boring_gpu_max_threads()".to_string(),
                    "maxSharedMem"      => "__boring_gpu_max_shared_mem()".to_string(),
                    "index"             => format!("({} as i64)", v),
                    _ => format!("{}.{}({})", v, method, args.iter().map(|a| self.emit_expr(&a.value)).collect::<Vec<_>>().join(", ")),
                });
            }
        }
        None
    }

    /// `.clone()` pass-through, plus `Task`-cancellation method calls: `Task.cancelled()`
    /// (inside a cancellable `def` task fn), `v.cancel()` on a spawned join-handle var, and
    /// `future.done()` — a non-blocking poll on a `JoinHandle`.
    fn try_emit_clone_and_task_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        // .clone() — pass-through to Rust's Clone trait. Used in .br source when a value
        // needs to be used multiple times (Boring has reference semantics; Rust needs explicit clone).
        if method == "clone" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return Some(format!("{}.clone()", obj_s));
        }
        // Task.cancelled() → __task_cancel.cancelled() (inside a cancellable task def fn)
        if method == "cancelled" && args.is_empty() {
            if let ExprKind::Var(v) = &obj.kind {
                if v == "Task" {
                    self.uses_tokio_util.set(true);
                    return Some("__task_cancel.cancelled()".to_string());
                }
            }
        }
        // v.cancel() where v is a join_handle_var → use the cancel token
        if method == "cancel" && args.is_empty() {
            if let ExprKind::Var(v) = &obj.kind {
                if self.join_handle_vars.contains(v.as_str()) {
                    self.uses_tokio_util.set(true);
                    if let Some(cancel_var) = self.cancel_token_vars.get(v.as_str()) {
                        return Some(format!("{}.cancel()", cancel_var));
                    }
                    // Fallback: abort the JoinHandle
                    return Some(format!("{}.abort()", v));
                }
            }
        }
        // Future.done() — non-blocking poll on a JoinHandle
        if method == "done" && args.is_empty() {
            if let ExprKind::Var(v) = &obj.kind {
                if self.task_vars.contains(v.as_str()) || self.join_handle_vars.contains(v.as_str()) {
                    return Some(format!(
                        "tokio::time::timeout(std::time::Duration::ZERO, {}).await.is_ok()",
                        v
                    ));
                }
            }
            // Fallback below covers a JoinHandle/Future reached through a field
            // (`self.handle.done()`) or another expression not tracked in the var-level
            // sets above — but never a known user struct/enum with its own zero-arg
            // `done()` method (e.g. a `Job.done()` status check), which must dispatch
            // through its own declared method instead.
            if self.expr_receiver_is_known_user_type(obj) {
                return None;
            }
            let obj_s = self.emit_expr(obj);
            return Some(format!(
                "tokio::time::timeout(std::time::Duration::ZERO, {}).await.is_ok()",
                obj_s
            ));
        }
        None
    }

    /// Channel operations: sender `.send()` (mpsc/oneshot/watch/broadcast, each with its
    /// own sync-vs-async and error-handling shape), `.subscribe()` on a broadcast sender,
    /// and receiver `.recv()` (oneshot/broadcast/watch). `tx.clone()` needs no special
    /// case — Rust's derived `Clone` handles channel senders natively.
    fn try_emit_channel_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        // Channel sender: `tx.send(value)` → `tx.send(value).await.unwrap()`
        // Use emit_expr_owned so string literals are wrapped in Arc::from(...) to match the
        // channel's Arc<str> item type.
        if method == "send" {
            if let ExprKind::Var(var_name) = &obj.kind {
                if self.channel_senders.contains(var_name.as_str()) {
                    let val = args.first().map(|a| self.emit_expr_owned(&a.value)).unwrap_or_default();
                    // In single-thread mode, local_channel::mpsc::send() is synchronous (not async).
                    let is_single = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single);
                    return Some(if is_single {
                        if self.in_throws || self.in_try_body {
                            format!("{}.send({}).map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?", var_name, val)
                        } else {
                            format!("{}.send({}).expect(\"channel receiver dropped\")", var_name, val)
                        }
                    } else if self.in_throws || self.in_try_body {
                        format!("{}.send({}).await.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?", var_name, val)
                    } else {
                        format!("{}.send({}).await.expect(\"channel receiver dropped\")", var_name, val)
                    });
                }
                // oneshot/watch senders: non-async, swallow error with .ok()
                if self.oneshot_senders.contains(var_name.as_str())
                    || self.watch_senders.contains(var_name.as_str())
                {
                    let val = args.first().map(|a| self.emit_expr_owned(&a.value)).unwrap_or_default();
                    return Some(format!("{}.send({}).ok()", var_name, val));
                }
                // broadcast sender: single-thread LocalBroadcastSender::send() returns ();
                // multi-thread tokio::sync::broadcast::Sender::send() returns Result.
                if self.broadcast_senders.contains(var_name.as_str()) {
                    let val = args.first().map(|a| self.emit_expr_owned(&a.value)).unwrap_or_default();
                    return Some(if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        format!("{}.send({})", var_name, val)
                    } else {
                        format!("{}.send({}).ok()", var_name, val)
                    });
                }
            }
        }
        // broadcast tx.subscribe() → tx.subscribe()
        if method == "subscribe" {
            if let ExprKind::Var(var_name) = &obj.kind {
                if self.broadcast_senders.contains(var_name.as_str()) {
                    return Some(format!("{}.subscribe()", var_name));
                }
            }
        }
        // oneshot receiver: rx.recv() → rx.await.unwrap() (consumed once)
        if method == "recv" {
            if let ExprKind::Var(var_name) = &obj.kind {
                if self.oneshot_receivers.contains(var_name.as_str()) {
                    return Some(if self.in_throws || self.in_try_body {
                        format!("{}.await?", var_name)
                    } else {
                        format!("{}.await.expect(\"oneshot channel sender dropped\")", var_name)
                    });
                }
                // broadcast receiver: rx.recv() → rx.recv().await
                // In single-thread mode, LocalBroadcastReceiver::recv() returns T directly (no Result).
                if self.broadcast_receivers.contains(var_name.as_str()) {
                    if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        return Some(format!("{}.recv().await", var_name));
                    }
                    return Some(if self.in_throws || self.in_try_body {
                        format!("{}.recv().await.map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?", var_name)
                    } else {
                        format!("{}.recv().await.expect(\"broadcast channel error: sender dropped or lagged\")", var_name)
                    });
                }
                // watch receiver: rx.recv() → { rx.changed().await.ok(); rx.borrow().clone() }
                if self.watch_receivers.contains(var_name.as_str()) {
                    return Some(format!("{{ {}.changed().await.ok(); {}.borrow().clone() }}", var_name, var_name));
                }
            }
        }
        None
    }

    /// `TypeName.method(args)` — an enum variant constructor (`Color.Green(x, y)` →
    /// `Color::Green(x, y)`, uppercase-first method name), a registered type/static
    /// method (`Counter.zero()` → `Counter::zero()`), or the camelCase→snake_case
    /// fallback for external Rust types (`Duration.fromMillis(100)`). Returns `None`
    /// when `obj` isn't a type/module path — including a local var that happens to be
    /// uppercase, which must fall through to ordinary instance-method dispatch.
    fn try_emit_type_method_call(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        let ExprKind::Var(type_name) = &obj.kind else { return None };
        // A known local variable that happens to start with an uppercase letter
        // (e.g. `var Qh = []`), or a top-level `let` constant (`PADDLE_SIZE`,
        // conventionally UPPER_SNAKE_CASE) -- is still a value, not a type/module path --
        // must fall through to ordinary instance-method dispatch further down, not the
        // `TypeName::method(...)` treatment below. See `emit_expr.rs`'s matching guard on
        // the field-access side (`is_top_level_let_value`) for the same reasoning.
        let is_type = !self.known_local_vars.contains(type_name.as_str())
            && !self.user_top_level_names.contains(type_name.as_str())
            && type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
        if !is_type { return None; }
        let is_variant = method.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
        if is_variant {
            let key = format!("{}::{}", type_name, method);
            if self.enum_variant_fields.contains_key(&key) {
                // Enum variant: `EnumType.Variant(args)` → `EnumType::Variant(...)`
                // Use emit_let_value so string args are coerced to Arc<str>.
                let field_tys = self.enum_variant_field_types.get(&key).cloned().unwrap_or_default();
                let vals: Vec<String> = args.iter().enumerate().map(|(i, a)| {
                    let ty = field_tys.get(i);
                    let raw = self.emit_let_value(ty, &a.value);
                    // enum variant fields are owned — strip the leading `&` that
                    // emit_let_value adds for actor-typed function params, and add .clone()
                    // so the original Arc<Mutex<T>> variable can still be used afterward.
                    let raw = if matches!(ty,
                        Some(Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask | OwnerQual::Guard | OwnerQual::GuardTask))
                    ) {
                        if let Some(stripped) = raw.strip_prefix('&') {
                            format!("{}.clone()", stripped)
                        } else {
                            raw
                        }
                    } else {
                        raw
                    };
                    let rec_key = format!("{}::{}::{}", type_name, method, i);
                    if self.recursive_fields.contains(&rec_key) {
                        if matches!(ty, Some(Type::Optional(_))) {
                            format!("{}.map(Box::new)", raw)
                        } else {
                            format!("Box::new({})", raw)
                        }
                    } else {
                        raw
                    }
                }).collect();
                // When the user defines their own Result<T,E> enum, Rust can't infer
                // both type params from one variant's args. Add turbofish to disambiguate.
                let type_turbofish = if self.user_defines_result && type_name == "Result" {
                    if method == "Ok" { "::<_, ()>" } else { "::<(), _>" }
                } else { "" };
                return Some(format!("{}{}::{}({})", type_name, type_turbofish, method, vals.join(", ")));
            }
        }
        // Type method (lowercase): `Counter.zero()` → `Counter::zero()`
        // For type set methods: `Counter.count(v)` → `Counter::set_count(v)`
        if let Some(sigs) = self.struct_type_method_sigs.get(type_name.as_str()) {
            let (rust_name, is_setter) = sigs.get(method)
                .map(|kind| match kind {
                    TypeMethodKind::Set => (format!("set_{}", method), true),
                    _ => (method.to_string(), false),
                })
                .unwrap_or_else(|| (method.to_string(), false));
            let vals: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
            let _ = is_setter;
            return Some(format!("{}::{}({})", type_name, rust_name, vals.join(", ")));
        }
        // `is_variant` (computed above) is true but the type wasn't found in
        // `enum_variant_fields` -- this is a *genuinely external* enum's tuple
        // variant (`FontSize.Px(33.0)` → `FontSize::Px(33.0)`), one Boring never
        // parsed a declaration for (e.g. from a `use bevy.prelude.*`-style
        // import), so it has no entry to look up. A real Rust method name is
        // never capitalized, so a capitalized `method` here can only ever be a
        // variant (or associated constant) — emit it verbatim. Applying
        // camelCase→snake_case below would silently mangle it (`Px` → `px`,
        // a method that doesn't exist) and only ever produce a bad Rust name,
        // since a genuine external method call never reaches this branch
        // capitalized in the first place.
        if is_variant {
            let vals: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
            return Some(format!("{}::{}({})", type_name, method, vals.join(", ")));
        }
        // Fallback for any uppercase type not registered in the transpiler
        // (external types like Duration, File, Path, BufReader, etc.):
        // `Duration.fromMillis(100)` (Boring camelCase) → `Duration::from_millis(100)` (Rust)
        let rust_method = camel_to_snake(method);
        let vals: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
        let call = format!("{}::{}({})", type_name, rust_method, vals.join(", "));
        // Known async static methods need .await (+ ? in throws context) in async context
        const TOKIO_ASYNC_STATIC_TYPE: &[&str] = &["open", "create", "connect", "bind"];
        if self.in_async && TOKIO_ASYNC_STATIC_TYPE.iter().any(|&m| m == rust_method) {
            let awaited = format!("{}.await", call);
            // In throws/try context, propagate the Result error with `?`
            return Some(if self.in_throws || self.in_try_body {
                format!("{}?", awaited)
            } else {
                awaited
            });
        }
        Some(call)
    }

    /// `RwLock`-backed (`'guard`/`'guard'task`) method dispatch: a local var of type
    /// `T'guard` or a `self.field` of that type. `req` methods take a `.read()` lock,
    /// mutating (`def`) methods take `.write()`; `'guard'task` (tokio `RwLock`) awaits
    /// the lock acquisition in async context.
    fn try_emit_rwlock_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        // RwLock local var method: c.method(args) → c.read/write()[.await|.unwrap()].method(args)
        // req methods use .read(), def methods use .write().
        // 'guard → std::sync::RwLock (unwrap), 'guard'task → tokio::sync::RwLock (await in async ctx).
        if let ExprKind::Var(v) = &obj.kind {
            if self.var_rwlock_types.contains(v.as_str()) || self.var_rwlock_task_types.contains(v.as_str()) {
                let is_task = self.var_rwlock_task_types.contains(v.as_str());
                let (rust_method, extra_wrap) = map_method(method, args.len());
                let args_s: Vec<String> = args.iter().map(|a| self.emit_expr_owned(&a.value)).collect();
                let struct_name = self.resolve_receiver_type_name(v.as_str());
                let is_req = self.method_is_req_or_task(&struct_name, method);
                // `'guard` (RwLock) gets no exception from the general rule —
                // docs/mut-type-modifier.md §1: mutation goes through the lock
                // rather than `&mut self`, but Boring still gates it on the
                // binding, same as every other type. `var T'guard x` alone no
                // longer suffices; only `var mut T'guard x` does. Rust's own
                // type system wouldn't catch this (the RwLock always allows
                // `.write()` regardless of Boring's own `mut` bookkeeping).
                if !is_req && self.known_local_vars.contains(v.as_str())
                    && !self.content_mutable_local_vars.contains(v.as_str())
                    && self.mut_checked_local_vars.contains(v.as_str())
                    // Permissive when the receiver's struct type couldn't even
                    // be resolved — see the matching comment in
                    // `try_emit_mutex_method`. A struct always gets this check;
                    // an enum only when it has a `mut`-qualified variant field
                    // (`enum_has_mut_field`) — see this file's general fallback
                    // diagnostic for the non-actor/guard case.
                    && (self.struct_fields.contains_key(struct_name.as_str()) || self.enum_has_mut_field(struct_name.as_str()))
                {
                    self.push_error(obj.line, obj.col, format!("`{}` is not declared `mut` — cannot call `def` method `.{}()` on a non-mut binding; fix: declare it `mut {} {}` or `var mut {} {}`", v, method, struct_name, v, struct_name, v));
                }
                let guard = if is_req {
                    if is_task { self.guard_task_read_access(v) } else { self.guard_read_access(v) }
                } else {
                    if is_task { self.guard_task_write_guard(v) } else { self.guard_write_guard(v) }
                };
                let call = format!("{}.{}({})", guard, rust_method, args_s.join(", "));
                let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                const TOKIO_ASYNC_INSTANCE: &[&str] = &["recv", "send", "write_all", "read_line", "acquire", "flush"];
                let needs_await = self.instance_task_methods.contains(method)
                    || TOKIO_ASYNC_INSTANCE.contains(&method);
                return Some(if self.in_async && needs_await {
                    format!("{}.await", call)
                } else {
                    call
                });
            }
        }
        // RwLock struct field method: self.data.method() → self.data.read/write().await.method()
        if let ExprKind::Field(inner_obj, rwlock_field) = &obj.kind {
            if let ExprKind::Var(v) = &inner_obj.kind {
                if v == "self" {
                    let key = self.self_type.as_deref()
                        .map(|t| format!("{}::{}", t, rwlock_field));
                    if let Some(k) = key {
                        if self.struct_rwlock_fields.contains(&k) || self.struct_rwlock_task_fields.contains(&k) {
                            let (rust_method, extra_wrap) = map_method(method, args.len());
                            let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
                            let struct_type_name = self.self_type.as_deref().unwrap_or("");
                            let req_key = format!("{}::{}", struct_type_name, method);
                            let is_req = self.struct_req_methods.contains(&req_key);
                            let field_expr = format!("self.{}", rwlock_field);
                            let guard = if is_req {
                                self.rwlock_field_read(&k, &field_expr)
                            } else {
                                self.rwlock_field_write(&k, &field_expr)
                            };
                            let call = format!("{}.{}({})", guard, rust_method, args_s.join(", "));
                            let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                            return Some(if self.in_async && self.instance_task_methods.contains(method) {
                                format!("{}.await", call)
                            } else {
                                call
                            });
                        }
                    }
                }
            }
        }
        None
    }

    /// `Mutex`-backed (`'actor`/`'actor'task`) method dispatch: a local var or `self.field`
    /// of that type, plus managed-mode (implicit `Arc<Mutex<T>>`/`RefCell<T>`) vars. Locks
    /// via `.lock()`/`.borrow_mut()` (or `.borrow()` for `req` methods in single-thread
    /// mode), awaits the lock in async 'actor'task context, and propagates `throws` errors.
    fn try_emit_mutex_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        // Mutex local var method: w.method(args) → w.lock().await.method(args)
        if let ExprKind::Var(v) = &obj.kind {
            if self.var_mutex_types.contains(v.as_str()) || self.var_mutex_task_types.contains(v.as_str()) {
                // `'actor` gets no exception from the general rule either — see
                // the matching check/comment in `try_emit_rwlock_method`.
                // Params are excluded (unchanged, still-unenforced parameter
                // model — see `content_mutable_local_vars`'s doc); only a
                // genuine local binding is checked here.
                if !self.fn_current_params.contains_key(v.as_str())
                    && self.known_local_vars.contains(v.as_str())
                    && !self.content_mutable_local_vars.contains(v.as_str())
                    && self.mut_checked_local_vars.contains(v.as_str())
                {
                    let struct_name = self.resolve_receiver_type_name(v.as_str());
                    // If we can't even resolve the receiver's struct type (a
                    // real inference gap independent of this diagnostic — e.g.
                    // an inferred `let cur = interp.current_env` chained field
                    // read), we have no basis to call it mutating either;
                    // don't guess. Same permissive-on-unknown shape as the
                    // general local-binding check in `emit_method_call_fallback`
                    // (gated on `self.struct_fields.contains_key(sn)`). A struct
                    // always gets this check; an enum only when it has a
                    // `mut`-qualified variant field (`enum_has_mut_field`).
                    if (self.struct_fields.contains_key(struct_name.as_str()) || self.enum_has_mut_field(struct_name.as_str()))
                        && !self.method_is_req_or_task(&struct_name, method)
                    {
                        self.push_error(obj.line, obj.col, format!("`{}` is not declared `mut` — cannot call `def` method `.{}()` on a non-mut binding; fix: declare it `mut {} {}` or `var mut {} {}`", v, method, struct_name, v, struct_name, v));
                    }
                }
                let (mut rust_method, extra_wrap) = map_method(method, args.len());
                // `append(xs)` where xs is a collection → use `extend` instead of `push`
                // so that Vec<T> arguments are flattened into the actor collection.
                if rust_method == "push" && args.len() == 1 {
                    let arg_is_collection = match &args[0].value.kind {
                        ExprKind::Var(arg_v) => self.vec_vars.contains(arg_v.as_str()),
                        ExprKind::Array(_) => true,
                        _ => false,
                    };
                    if arg_is_collection { rust_method = "extend".into(); }
                }
                // Use emit_expr_owned so string interpolations/trim results get Arc<str> wrapping.
                let args_s: Vec<String> = args.iter().map(|a| self.emit_expr_owned(&a.value)).collect();
                if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                    // Single-thread: T'actor = Rc<RefCell<T>>.
                    // Use borrow() for req (read-only) methods, borrow_mut() for def methods.
                    let borrow_kind = {
                        let struct_name = self.fn_current_params.get(v.as_str())
                            .and_then(|ty| if let crate::ast::Type::Qualified(inner, _) = ty {
                                if let crate::ast::Type::Named(n) = inner.as_ref() { Some(n.clone()) } else { None }
                            } else { None });
                        let is_req = struct_name.is_some_and(|sn| {
                            self.struct_req_methods.contains(&format!("{}::{}", sn, method))
                        });
                        if is_req { "borrow" } else { "borrow_mut" }
                    };
                    let call = format!("{}.{}().{}({})", v, borrow_kind, rust_method, args_s.join(", "));
                    let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                    let throws = (self.in_throws || self.in_try_body)
                        && (self.fn_throws.contains(method) || self.struct_method_throws.contains(method));
                    return Some(if throws { format!("{}?", call) } else { call });
                }
                let guard_expr = self.mutex_var_write(v, v);
                let call = format!("{}.{}({})", guard_expr, rust_method, args_s.join(", "));
                let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                // Add .await for user task methods AND known tokio async instance methods (recv, etc.)
                const TOKIO_ASYNC_INSTANCE: &[&str] = &["recv", "send", "write_all", "read_line", "acquire", "flush"];
                let needs_await = self.instance_task_methods.contains(method)
                    || TOKIO_ASYNC_INSTANCE.contains(&method);
                let throws = (self.in_throws || self.in_try_body)
                    && (self.fn_throws.contains(method) || self.struct_method_throws.contains(method));
                let call = if throws { format!("{}?", call) } else { call };
                return Some(if self.in_async && needs_await {
                    format!("{}.await", call)
                } else {
                    call
                });
            }
        }
        // Managed-mode mutex var method: w.method(args) → w.lock().unwrap().method(args)
        // Uses std::sync::Mutex (synchronous), no .await needed.
        if let ExprKind::Var(v) = &obj.kind {
            if self.managed_mutex_vars.contains(v.as_str()) {
                let (rust_method, extra_wrap) = map_method(method, args.len());
                let args_s: Vec<String> = args.iter().map(|a| self.emit_expr_owned(&a.value)).collect();
                let call = format!("{}.lock().unwrap().{}({})", v, rust_method, args_s.join(", "));
                let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                return Some(call);
            }
            // Managed-mode RefCell var method: w.method(args) → w.borrow_mut().method(args)
            if self.managed_refcell_vars.contains(v.as_str()) {
                let (rust_method, extra_wrap) = map_method(method, args.len());
                let args_s: Vec<String> = args.iter().map(|a| self.emit_expr_owned(&a.value)).collect();
                let call = format!("{}.borrow_mut().{}({})", v, rust_method, args_s.join(", "));
                let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                return Some(call);
            }
        }
        // Mutex struct field method: self.worker.method(args) → self.worker.lock().await.method(args)
        if let ExprKind::Field(inner_obj, mutex_field) = &obj.kind {
            if let ExprKind::Var(v) = &inner_obj.kind {
                if v == "self" {
                    let key = self.self_type.as_deref()
                        .map(|t| format!("{}::{}", t, mutex_field));
                    if let Some(k) = key {
                        if self.struct_mutex_fields.contains(&k) || self.struct_mutex_task_fields.contains(&k) {
                            let (rust_method, extra_wrap) = map_method(method, args.len());
                            let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
                            let guard_expr = self.mutex_field_write(&k, &format!("self.{}", mutex_field));
                            let call = format!("{}.{}({})", guard_expr, rust_method, args_s.join(", "));
                            let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                            return Some(if self.in_async && self.instance_task_methods.contains(method) {
                                format!("{}.await", call)
                            } else {
                                call
                            });
                        }
                    }
                }
            }
        }
        None
    }

    /// Actor-qualified struct-field method dispatch: `outer.actor_field.method(args)` where
    /// `actor_field`'s declared type is `T'actor`/`T'actor'task` (either on `outer: struct`
    /// via `self.actor_field...` / `some_struct_var.actor_field...`, or on `outer: T'actor`
    /// itself for methods requiring write access to a nested field).
    fn try_emit_actor_field_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        // Actor-typed field method call: `outer.actor_field.method(args)`.
        // When `actor_field` has type `T'actor` (Rc<RefCell<T>> in single-thread),
        // emit `outer.actor_field.borrow_mut().method(args)`.
        if let ExprKind::Field(inner_obj, field_name) = &obj.kind {
            if let ExprKind::Var(v) = &inner_obj.kind {
                // Determine the struct type of the outer variable.
                let outer_struct = self.var_struct_types.get(v.as_str())
                    .or_else(|| self.var_struct_type.get(v.as_str()))
                    .cloned()
                    .or_else(|| {
                        if self.managed_refcell_vars.contains(v.as_str())
                            || self.managed_mutex_vars.contains(v.as_str())
                            || self.var_mutex_types.contains(v.as_str())
                            || self.var_mutex_task_types.contains(v.as_str())
                        {
                            self.var_types.get(v.as_str()).and_then(|t| {
                                match t {
                                    crate::ast::Type::Named(n) => Some(n.clone()),
                                    crate::ast::Type::Qualified(inner, _) => {
                                        if let crate::ast::Type::Named(n) = inner.as_ref() { Some(n.clone()) } else { None }
                                    }
                                    _ => None,
                                }
                            })
                        } else { None }
                    });
                if let Some(struct_name) = outer_struct {
                    // Look up the field's declared type.
                    let field_ty = self.struct_fields.get(struct_name.as_str())
                        .and_then(|fs| fs.iter().find(|(n, _)| n == field_name))
                        .map(|(_, ty)| ty.clone());
                    let is_actor_field = matches!(&field_ty,
                        Some(crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Actor | crate::ast::OwnerQual::ActorTask)));
                    if is_actor_field {
                        let (rust_method, extra_wrap) = map_method(method, args.len());
                        let args_s: Vec<String> = args.iter().map(|a| self.emit_expr_owned(&a.value)).collect();
                        let obj_s = self.emit_expr(obj);
                        let is_task_field = matches!(&field_ty,
                            Some(crate::ast::Type::Qualified(_, crate::ast::OwnerQual::ActorTask)));
                        let call = if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                            format!("{}.borrow_mut().{}({})", obj_s, rust_method, args_s.join(", "))
                        } else if is_task_field {
                            { let g = self.actor_task_write_guard(&obj_s); format!("{}.{}({})", g, rust_method, args_s.join(", ")) }
                        } else {
                            { let g = self.actor_write_guard(&obj_s); format!("{}.{}({})", g, rust_method, args_s.join(", ")) }
                        };
                        let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                        return Some(call);
                    }
                }
            }
        }
        // Actor local-var field mutating method: `actor_var.field.method(args)`.
        // When `actor_var` is `T'actor` (Rc<RefCell<T>> / Arc<Mutex<T>>), field reads use
        // `borrow()` / `lock().unwrap()`, but *mutating* method calls on those fields need
        // write access. Emit `actor_var.borrow_mut().field.method(args)` (single) or
        // `actor_var.lock().unwrap().field.method(args)` (multi).
        // Only applies to methods that require `&mut self` on the field — read-only methods
        // (get, len, contains, …) fall through to more specific handlers below.
        if MUTATING_COLLECTION_METHODS.contains(&method) {
            if let ExprKind::Field(inner_obj, field_name) = &obj.kind {
                if let ExprKind::Var(v) = &inner_obj.kind {
                    if self.var_mutex_types.contains(v.as_str()) || self.var_mutex_task_types.contains(v.as_str()) {
                        let (rust_method, extra_wrap) = map_method(method, args.len());
                        let args_s: Vec<String> = args.iter().map(|a| self.emit_expr_owned(&a.value)).collect();
                        let guard = self.mutex_var_write(v, v);
                        let call = format!("{}.{}.{}({})", guard, field_name, rust_method, args_s.join(", "));
                        let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                        return Some(call);
                    }
                }
            }
        }
        None
    }

    /// Special-cased method emits that don't fit any general category: `read_line` into
    /// a tracked mutable string var (Arc<str> is immutable, so round-trips through a
    /// temporary `String` buffer), `.clear()` on a mutable string var, and array
    /// `.filter(closure)` (rebinds the closure param to an owned clone, since
    /// `Iterator::filter` passes `&Item` but Boring closures expect an owned value).
    fn try_emit_special_case_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        // Special case: read_line(string_arc_var) — Arc<str> is immutable so we use a
        // temporary String buffer then assign back.
        // tokio BufReader (local var) → async (.await.unwrap_or(0))
        // std::io::stdin() (call result) → sync (.unwrap_or(0), no await)
        if method == "read_line" && !self.expr_receiver_is_known_user_type(obj) {
            if let Some(arg) = args.first() {
                if let ExprKind::Var(v) = &arg.value.kind {
                    if self.string_arc_vars.contains(v.as_str()) {
                        let obj_s = self.emit_expr(obj);
                        return Some(if matches!(&obj.kind, ExprKind::Var(_)) {
                            // async tokio reader
                            format!(
                                "{{ let mut __boring_buf = String::new(); \
                                 let __boring_n = {}.read_line(&mut __boring_buf).await.unwrap_or(0); \
                                 {} = Arc::from(__boring_buf); __boring_n }}",
                                obj_s, v
                            )
                        } else {
                            // sync std::io::Stdin
                            format!(
                                "{{ let mut __boring_buf = String::new(); \
                                 let __boring_n = {}.read_line(&mut __boring_buf).unwrap_or(0); \
                                 {} = Arc::from(__boring_buf); __boring_n }}",
                                obj_s, v
                            )
                        });
                    }
                }
            }
        }
        // Special case: string_arc_var.clear() — reassign to empty Arc<str>
        if method == "clear" {
            if let ExprKind::Var(v) = &obj.kind {
                if self.string_arc_vars.contains(v.as_str()) {
                    return Some(format!("{} = Arc::from(\"\")", v));
                }
            }
        }
        // Array filter: Rust's Iterator::filter passes &Item to the predicate, but Boring
        // closures expect owned values (e.g. `n: n > 0`).  Wrap the closure param with a
        // `.clone()` rebind so `n` inside the body is an owned T, not a &T.
        // Only applies to array filter — not Option::filter (handled below) and not
        // dict::filter (handled in the recv_is_dict block below with a 2-param closure).
        // Also never applies to a user struct/enum's own `filter(closure)` method — this
        // used to have no receiver-type check at all (unlike try_emit_dict_method/
        // try_emit_set_method/try_emit_array_special_method, each gated on their own
        // `expr_is_*`/`expr_receiver_is_known_user_type` check).
        if method == "filter" && !self.is_option_expr(obj) && !self.expr_is_dict(obj)
            && !self.expr_receiver_is_known_user_type(obj) {
            if let Some(first_arg) = args.first() {
                if let ExprKind::Closure(params, _, body, _, _) = &first_arg.value.kind {
                    if let Some(param) = params.first() {
                        let pname = &param.name;
                        let mut sub = self.make_sub();
                        for p in params { sub.known_local_vars.insert(p.name.clone()); }
                        let body_s = match body {
                            ClosureBody::Expr(e) => sub.emit_expr(e),
                            ClosureBody::Block(stmts) => {
                                let inner: Vec<String> = stmts.iter().map(|s| sub.emit_stmt_inline(s)).collect();
                                format!("{{ {} }}", inner.join(" "))
                            }
                        };
                        let obj_s = self.emit_expr(obj);
                        let closure_s = format!("|{}| {{ let {} = {}.clone(); {} }}", pname, pname, pname, body_s);
                        return Some(format!("{}.iter().cloned().filter({}).collect::<Vec<_>>()", obj_s, closure_s));
                    }
                }
            }
        }
        None
    }

    /// Emits a closure argument with its first (and optionally second) parameter rebound
    /// to an owned clone: iterator adapters like `.filter()`/`.any()` pass `&Item` to the
    /// predicate, but Boring closures expect an owned value. `extra_params > 0` produces a
    /// two-param tuple-destructuring closure (dict `map`/`filter` over `(k, v)`).
    fn emit_owned_closure(&self, arg: &Arg, extra_params: usize) -> Option<String> {
        if let ExprKind::Closure(params, _, body, _, _) = &arg.value.kind {
            if let Some(param) = params.first() {
                let pname = &param.name;
                let mut sub = self.make_sub();
                for p in params { sub.known_local_vars.insert(p.name.clone()); }
                let body_s = match body {
                    ClosureBody::Expr(e) => sub.emit_expr(e),
                    ClosureBody::Block(stmts) => {
                        let inner: Vec<String> = stmts.iter().map(|s| sub.emit_stmt_inline(s)).collect();
                        format!("{{ {} }}", inner.join(" "))
                    }
                };
                let closure_s = if extra_params == 0 {
                    format!("|{}| {{ let {} = {}.clone(); {} }}", pname, pname, pname, body_s)
                } else {
                    // two-param closure (e.g. dict map/filter with (k, v))
                    let p2 = params.get(1).map(|p| p.name.as_str()).unwrap_or("__v");
                    format!("|({}, {})| {{ let {} = {}.clone(); let {} = {}.clone(); {} }}", pname, p2, pname, pname, p2, p2, body_s)
                };
                return Some(closure_s);
            }
        }
        None
    }

    /// Tuple-specific methods (`length`/`count`/`isEmpty`/`first`/`last`/`map`/`all`/`any`).
    /// Tuples are heterogeneous, so only this limited set applies; arity is resolved from
    /// a literal receiver or from `tuple_vars`. `map`/`all`/`any` apply the closure to each
    /// slot independently via a scoped `let`-bind (not a single Rust closure — the slots
    /// can have different types), joined with `,`/`&&`/`||` respectively.
    fn try_emit_tuple_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        let tuple_arity: Option<usize> = match &obj.kind {
            ExprKind::Tuple(elems) => Some(elems.len()),
            ExprKind::Var(v) => self.tuple_vars.get(v.as_str()).copied(),
            _ => None,
        };
        let arity = tuple_arity?;
        let obj_s = self.emit_expr(obj);
        match method {
            "length" | "count" if args.is_empty() => {
                return Some(format!("{}", arity));
            }
            "isEmpty" if args.is_empty() => {
                return Some(format!("{}", arity == 0));
            }
            "first" if args.is_empty() && arity > 0 => {
                return Some(format!("{}.0", obj_s));
            }
            "last" if args.is_empty() && arity > 0 => {
                return Some(format!("{}.{}", obj_s, arity - 1));
            }
            "map" if args.len() == 1 && arity > 0 => {
                // Heterogeneous tuple map: apply the closure to each slot independently.
                // Result is a tuple of the same arity with each element transformed.
                // Each slot is emitted as `(|pname| body)(__boring_t.i.clone())` — a Rust
                // closure applied immediately. This lets Rust infer a distinct result type per
                // slot (heterogeneous-safe) and resolves `.value` / struct field accesses
                // correctly via Rust's own type inference rather than the sub-transpiler.
                if let ExprKind::Closure(params, _, body, _, task_flag) = &args[0].value.kind {
                    let pname = params.first().map(|p| p.name.as_str()).unwrap_or("__x");
                    let mut sub = self.make_sub();
                    sub.known_local_vars.insert(pname.to_string());
                    // Mark the closure param as a non-future local so that field accesses
                    // like `.value` are emitted as plain field access, not `.await.unwrap()`.
                    // An empty string as the struct type is enough to satisfy is_known_struct
                    // without incorrectly dispatching any struct method lookups.
                    sub.var_struct_types.insert(pname.to_string(), String::new());
                    if *task_flag { sub.in_async = true; }
                    let body_s = match body {
                        ClosureBody::Expr(e) => sub.emit_expr(e),
                        ClosureBody::Block(stmts) => {
                            let inner: Vec<String> = stmts.iter().map(|s| sub.emit_stmt_inline(s)).collect();
                            format!("{{ {} }}", inner.join(" "))
                        }
                    };
                    // Emit each slot as a scoped let-bind + body.
                    // Using `let pname = t.i.clone(); body` rather than an immediately-invoked
                    // closure so Rust can infer pname's type from the assignment RHS.
                    let slots: Vec<String> = (0..arity)
                        .map(|i| format!("{{ let {} = (__boring_t).{}.clone(); {} }}", pname, i, body_s))
                        .collect();
                    return Some(format!("{{ let __boring_t = {}; ({},) }}", obj_s, slots.join(", ")));
                }
            }
            "all" | "any" if args.len() == 1 && arity > 0 => {
                if let ExprKind::Closure(params, _, body, _, task_flag) = &args[0].value.kind {
                    let pname = params.first().map(|p| p.name.as_str()).unwrap_or("__x");
                    let mut sub = self.make_sub();
                    sub.known_local_vars.insert(pname.to_string());
                    sub.var_struct_types.insert(pname.to_string(), String::new());
                    if *task_flag { sub.in_async = true; }
                    let body_s = match body {
                        ClosureBody::Expr(e) => sub.emit_expr(e),
                        ClosureBody::Block(stmts) => {
                            let inner: Vec<String> = stmts.iter().map(|s| sub.emit_stmt_inline(s)).collect();
                            format!("{{ {} }}", inner.join(" "))
                        }
                    };
                    let op = if method == "all" { " && " } else { " || " };
                    // Wrap each slot in parens: `{ ... } || { ... }` would be parsed by
                    // Rust as an empty-params closure `|| { ... }` — use `({ ... })` instead.
                    let slots: Vec<String> = (0..arity)
                        .map(|i| format!("({{ let {} = (__boring_t).{}.clone(); {} }})", pname, i, body_s))
                        .collect();
                    return Some(format!("{{ let __boring_t = {}; {} }}", obj_s, slots.join(op)));
                }
            }
            _ => {}
        }
        None
    }

    /// Whether `obj` is confirmed to be a string receiver: in `string_vars`/`string_arc_vars`
    /// AND not also tracked as a collection (which would be a false positive), cross-checked
    /// against `var_types` — or a string/interpolation literal directly.
    fn expr_is_string_receiver(&self, obj: &Expr) -> bool {
        match &obj.kind {
            ExprKind::Var(v) => {
                // Must be in string_vars AND confirmed NOT a collection type.
                // string_vars can have false positives (e.g. when a var is also
                // tracked as a Vec), so we cross-check var_types.
                let in_string_set = self.string_arc_vars.contains(v.as_str())
                    || self.string_vars.contains(v.as_str());
                let not_collection = !self.vec_vars.contains(v.as_str())
                    && !self.collection_vars.contains(v.as_str());
                let type_confirms = {
                    let vt = self.var_types.get(v.as_str());
                    vt.is_none()
                    || matches!(vt, Some(Type::Str))
                    || matches!(vt, Some(Type::Named(n)) if n == "string" || n == "str")
                };
                in_string_set && not_collection && type_confirms
            }
            ExprKind::Str(_) | ExprKind::StringInterp(_) => true,
            _ => false,
        }
    }

    /// String-specific methods: `parseInt`/`parseFloat`, `chars`/`split` (threading-mode-aware
    /// `Rc<str>`/`Arc<str>` element type), `indexOf`/`slice`/`startsWith`/`endsWith`/`contains`/
    /// `replace`/`repeat`/`trim*`/`toUpper*`/`toLower*`. Coerces non-literal pattern args with
    /// `.as_ref()` since `Rc<str>`/`Arc<str>` don't implement `Pattern` directly.
    fn try_emit_string_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        if !self.expr_is_string_receiver(obj) { return None; }
        let obj_s = self.emit_expr(obj);
        match method {
            "parseInt" => {
                return Some(format!("{}.trim().parse::<isize>().ok()", obj_s));
            }
            "parseFloat" => {
                return Some(format!("{}.trim().parse::<f64>().ok()", obj_s));
            }
            "parseFloat32" => {
                return Some(format!("{}.trim().parse::<f32>().ok()", obj_s));
            }
            "chars" => {
                // Use threading-mode-aware string type (Rc<str> in single-thread, Arc<str> in multi).
                let str_ty = if self.use_rc_str() { "Rc::<str>" } else { "Arc::<str>" };
                let vec_ty = if self.use_rc_str() { "Rc<str>" } else { "Arc<str>" };
                return Some(format!(
                    "{}.chars().map(|c| {}::from(c.to_string())).collect::<Vec<{}>>()",
                    obj_s, str_ty, vec_ty
                ));
            }
            "indexOf" => {
                let sub_s = args.first().map(|a| self.emit_expr(&a.value))
                    .unwrap_or_else(|| "\"\"".to_string());
                // Use &* to coerce Arc<str>/Arc<str> argument to &str for str::find.
                return Some(format!("{}.find(&*{}).map(|i| i as isize)", obj_s, sub_s));
            }
            "slice" if args.len() >= 2 => {
                let start_s = self.emit_expr(&args[0].value);
                let end_s   = self.emit_expr(&args[1].value);
                // Bind start to a local so it is not double-evaluated (fix #23).
                // Wrap in Rc<str>/Arc<str> (threading-mode aware) so the result type is `string`.
                let is_single = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single);
                let str_ty = if is_single { "Rc::<str>" } else { "Arc::<str>" };
                return Some(format!(
                    "{{ let __start = ({}) as usize; {}::from({}.chars().skip(__start).take(({}) as usize - __start).collect::<String>().as_str()) }}",
                    start_s, str_ty, obj_s, end_s
                ));
            }
            // Pattern methods: Rc<str>/Arc<str> doesn't implement Pattern, need .as_ref().
            // For &str literals, no coercion needed (already &str which implements Pattern).
            "split" => {
                let sep_arg = args.first();
                let needs_coerce = sep_arg.map(|a| !matches!(&a.value.kind, ExprKind::Str(_))).unwrap_or(false);
                let sep_s = sep_arg.map(|a| self.emit_expr(&a.value))
                    .unwrap_or_else(|| "\"\"".to_string());
                let sep_s = if needs_coerce { format!("{}.as_ref()", sep_s) } else { sep_s };
                let is_single = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single);
                let str_ty = if is_single { "Rc::<str>" } else { "Arc::<str>" };
                return Some(format!(
                    "{}.split({}).map(|p| {}::from(p.to_string())).collect::<Vec<_>>()",
                    obj_s, sep_s, str_ty
                ));
            }
            "startsWith" | "hasPrefix" => {
                let arg = args.first();
                let needs_coerce = arg.map(|a| !matches!(&a.value.kind, ExprKind::Str(_))).unwrap_or(false);
                let arg_s = arg.map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "\"\"".to_string());
                let arg_s = if needs_coerce { format!("{}.as_ref()", arg_s) } else { arg_s };
                return Some(format!("{}.starts_with({})", obj_s, arg_s));
            }
            "endsWith" | "hasSuffix" => {
                let arg = args.first();
                let needs_coerce = arg.map(|a| !matches!(&a.value.kind, ExprKind::Str(_))).unwrap_or(false);
                let arg_s = arg.map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "\"\"".to_string());
                let arg_s = if needs_coerce { format!("{}.as_ref()", arg_s) } else { arg_s };
                return Some(format!("{}.ends_with({})", obj_s, arg_s));
            }
            "contains" => {
                let arg = args.first();
                let needs_coerce = arg.map(|a| !matches!(&a.value.kind, ExprKind::Str(_))).unwrap_or(false);
                let arg_s = arg.map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "\"\"".to_string());
                let arg_s = if needs_coerce { format!("{}.as_ref()", arg_s) } else { arg_s };
                return Some(format!("{}.contains({})", obj_s, arg_s));
            }
            "replace" | "replaceAll" => {
                let from_arg = args.first();
                let to_arg = args.get(1);
                let from_needs_coerce = from_arg.map(|a| !matches!(&a.value.kind, ExprKind::Str(_))).unwrap_or(false);
                let to_needs_coerce = to_arg.map(|a| !matches!(&a.value.kind, ExprKind::Str(_))).unwrap_or(false);
                let from_s = from_arg.map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "\"\"".to_string());
                let to_s = to_arg.map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "\"\"".to_string());
                let from_s = if from_needs_coerce { format!("{}.as_ref()", from_s) } else { from_s };
                let to_s = if to_needs_coerce { format!("{}.as_ref()", to_s) } else { to_s };
                let is_single = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single);
                let str_ty = if is_single { "Rc::<str>" } else { "Arc::<str>" };
                return Some(format!(
                    "{}::from({}.replace({}, {}).as_str())",
                    str_ty, obj_s, from_s, to_s
                ));
            }
            "repeat" if !args.is_empty() => {
                let n_s = self.emit_expr(&args[0].value);
                let is_single = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single);
                let str_ty = if is_single { "Rc::<str>" } else { "Arc::<str>" };
                return Some(format!("{}::from({}.repeat(({}) as usize).as_str())", str_ty, obj_s, n_s));
            }
            "trim" | "trimStart" | "trimEnd" => {
                let is_single = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single);
                let str_ty = if is_single { "Rc::<str>" } else { "Arc::<str>" };
                let rust_m = match method { "trimStart" => "trim_start", "trimEnd" => "trim_end", _ => "trim" };
                return Some(format!("{}::from({}.{}())", str_ty, obj_s, rust_m));
            }
            "toUpperCase" | "uppercased" | "upper" | "to_upper" | "toUpper" => {
                let is_single = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single);
                let str_ty = if is_single { "Rc::<str>" } else { "Arc::<str>" };
                return Some(format!("{}::from({}.to_uppercase().as_str())", str_ty, obj_s));
            }
            "toLowerCase" | "lowercased" | "lower" | "to_lower" | "toLower" => {
                let is_single = matches!(self.config.threading, crate::transpiler::ThreadingMode::Single);
                let str_ty = if is_single { "Rc::<str>" } else { "Arc::<str>" };
                return Some(format!("{}::from({}.to_lowercase().as_str())", str_ty, obj_s));
            }
            _ => {}
        }
        None
    }

    /// Dict (`HashMap`)-specific methods: `keys`/`values`/`get`(+default)/`set`|`put`/
    /// `contains`|`containsKey`|`has`/`remove`/`map`/`filter` (the latter two via
    /// `emit_owned_closure` with a 2-param `(k, v)` closure).
    fn try_emit_dict_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        if !self.expr_is_dict(obj) { return None; }
        let obj_s = self.emit_expr(obj);
        match method {
            "keys" => {
                return Some(format!("{}.keys().cloned().collect::<Vec<_>>()", obj_s));
            }
            "values" => {
                return Some(format!("{}.values().cloned().collect::<Vec<_>>()", obj_s));
            }
            "get" if args.len() >= 2 => {
                let key_s = self.emit_dict_key_borrow(&args[0].value);
                let def_s = self.emit_expr(&args[1].value);
                return Some(format!("{}.get({}).cloned().unwrap_or({})", obj_s, key_s, def_s));
            }
            "get" if args.len() == 1 => {
                let key_s = self.emit_dict_key_borrow(&args[0].value);
                return Some(format!("{}.get({}).cloned()", obj_s, key_s));
            }
            "set" | "put" if args.len() >= 2 => {
                let key_s = self.emit_expr_owned(&args[0].value);
                let val_s = self.emit_expr_owned(&args[1].value);
                return Some(format!("{}.insert({}, {})", obj_s, key_s, val_s));
            }
            "contains" | "containsKey" | "has" => {
                let key_s = self.emit_dict_key_borrow(&args[0].value);
                return Some(format!("{}.contains_key({})", obj_s, key_s));
            }
            "remove" if !args.is_empty() => {
                let key_s = self.emit_dict_key_borrow(&args[0].value);
                return Some(format!("{}.remove({})", obj_s, key_s));
            }
            "map" => {
                if let Some(first_arg) = args.first() {
                    if let ExprKind::Closure(params, _, body, _, _) = &first_arg.value.kind {
                        let kname = params.first().map(|p| p.name.as_str()).unwrap_or("k");
                        let vname = params.get(1).map(|p| p.name.as_str()).unwrap_or("v");
                        let mut sub = self.make_sub();
                        for p in params { sub.known_local_vars.insert(p.name.clone()); }
                        let body_s = match body {
                            ClosureBody::Expr(e) => sub.emit_expr(e),
                            ClosureBody::Block(stmts) => {
                                let inner: Vec<String> = stmts.iter().map(|s| sub.emit_stmt_inline(s)).collect();
                                format!("{{ {} }}", inner.join(" "))
                            }
                        };
                        return Some(format!(
                            "{}.iter().map(|({}, {})| {{ let {} = {}.clone(); let {} = {}.clone(); ({}.clone(), {}) }}).collect::<HashMap<_,_>>()",
                            obj_s, kname, vname, kname, kname, vname, vname, kname, body_s
                        ));
                    }
                }
            }
            "filter" => {
                if let Some(first_arg) = args.first() {
                    if let Some(closure_s) = self.emit_owned_closure(first_arg, 1) {
                        return Some(format!(
                            "{}.clone().into_iter().filter({}).collect::<HashMap<_,_>>()",
                            obj_s, closure_s
                        ));
                    }
                }
            }
            _ => {}
        }
        None
    }

    /// Set (`HashSet`)-specific methods: `toArray`, `union`/`intersection`/`difference`,
    /// `add`|`insert`, `remove`, `contains`.
    fn try_emit_set_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        if !self.expr_is_set(obj) { return None; }
        let obj_s = self.emit_expr(obj);
        match method {
            "toArray" => {
                return Some(format!("{}.iter().cloned().collect::<Vec<_>>()", obj_s));
            }
            "union" => {
                let other_s = args.first().map(|a| self.emit_expr(&a.value))
                    .unwrap_or_else(|| "Default::default()".to_string());
                return Some(format!("{}.union(&{}).cloned().collect::<HashSet<_>>()", obj_s, other_s));
            }
            "intersection" => {
                let other_s = args.first().map(|a| self.emit_expr(&a.value))
                    .unwrap_or_else(|| "Default::default()".to_string());
                return Some(format!("{}.intersection(&{}).cloned().collect::<HashSet<_>>()", obj_s, other_s));
            }
            "difference" => {
                let other_s = args.first().map(|a| self.emit_expr(&a.value))
                    .unwrap_or_else(|| "Default::default()".to_string());
                return Some(format!("{}.difference(&{}).cloned().collect::<HashSet<_>>()", obj_s, other_s));
            }
            "add" | "insert" if !args.is_empty() => {
                let val_s = self.emit_expr_owned(&args[0].value);
                return Some(format!("{}.insert({})", obj_s, val_s));
            }
            "remove" if !args.is_empty() => {
                let val_s = self.emit_expr_owned(&args[0].value);
                return Some(format!("{}.remove(&{})", obj_s, val_s));
            }
            "contains" if !args.is_empty() => {
                let val_s = self.emit_expr_owned(&args[0].value);
                return Some(format!("{}.contains(&{})", obj_s, val_s));
            }
            _ => {}
        }
        None
    }

    /// Array/iterator methods needing special emit beyond the generic `map_method` fallback
    /// (closures with owned-rebind, or block-expr construction): `future.value()`/`.wait()`,
    /// `joined`/`join`, `any`/`all`/`flatMap`/`count` (closure form), `sortBy`/`sortedBy`/
    /// `sorted`, `flat`, `zip`, `enumerate`, `slice`/`take`/`drop`, `min`/`max`/`sum`, `indexOf`.
    fn try_emit_array_special_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        // Unlike try_emit_dict_method/try_emit_set_method/try_emit_string_method (each gated on
        // `expr_is_dict`/`expr_is_set`/`expr_is_string_receiver`), the cases below used to have no
        // receiver-type guard at all — `min`/`max`/`sum`/`count`/`indexOf`/`any`/`all`/etc. fired
        // on method name alone. A user struct/enum method sharing one of those names must dispatch
        // through emit_method_call_fallback's own-member resolution instead; see
        // `expr_receiver_is_known_user_type`'s doc.
        if self.expr_receiver_is_known_user_type(obj) {
            return None;
        }
        // `future.value()` and `future.wait()` — method-call syntax for task results.
        // Equivalent to the field-access forms `future.value` / `future.wait`.
        // Delegate to emit_expr for the Field variant, which has the full
        // throws-aware .await.unwrap()? logic.
        if (method == "value" || method == "wait") && args.is_empty() {
            if let ExprKind::Var(v) = &obj.kind {
                if self.task_vars.contains(v.as_str())
                    || self.join_handle_vars.contains(v.as_str())
                {
                    // Delegate to emit_expr(Field(obj, method)) which has the full
                    // throws-aware .await.unwrap()? / .await.unwrap() logic.
                    let field_expr = crate::ast::Expr {
                        kind: crate::ast::ExprKind::Field(Box::new(obj.clone()), method.to_string()),
                        line: obj.line, col: obj.col, len: 0,
                    };
                    return Some(self.emit_expr(&field_expr));
                }
            }
        }

        // `joined(sep)` — join a Vec<Arc<str>> with a separator.
        // `Vec<Arc<str>>::join()` needs &str as separator (Arc<str> is emitted by the
        // standard arg-coercion path). We deref with &* to obtain &str.
        if method == "joined" || method == "join" {
            let obj_s = self.emit_expr(obj);
            let sep_s = args.first()
                .map(|a| self.emit_expr_owned(&a.value))
                .unwrap_or_else(|| self.str_from(""));
            return Some(format!("{}.iter().map(|__s| __s.as_ref()).collect::<Vec<&str>>().join(&*{})", obj_s, sep_s));
        }

        // `any(closure)` → iter().cloned().any(|x| { let x = x.clone(); body })
        if method == "any" {
            if let Some(first_arg) = args.first() {
                if let Some(closure_s) = self.emit_owned_closure(first_arg, 0) {
                    let obj_s = self.emit_expr(obj);
                    return Some(format!("{}.iter().cloned().any({})", obj_s, closure_s));
                }
            }
        }
        // `all(closure)` → iter().cloned().all(...)
        if method == "all" {
            if let Some(first_arg) = args.first() {
                if let Some(closure_s) = self.emit_owned_closure(first_arg, 0) {
                    let obj_s = self.emit_expr(obj);
                    return Some(format!("{}.iter().cloned().all({})", obj_s, closure_s));
                }
            }
        }
        // `flatMap(closure)` → iter().cloned().flat_map(...).collect::<Vec<_>>()
        if method == "flatMap" {
            if let Some(first_arg) = args.first() {
                if let Some(closure_s) = self.emit_owned_closure(first_arg, 0) {
                    let obj_s = self.emit_expr(obj);
                    return Some(format!("{}.iter().cloned().flat_map({}).collect::<Vec<_>>()", obj_s, closure_s));
                }
            }
        }
        // `count(closure?)` → with closure: filter+count; without: len() as i64
        if method == "count" {
            if let Some(first_arg) = args.first() {
                if let Some(closure_s) = self.emit_owned_closure(first_arg, 0) {
                    let obj_s = self.emit_expr(obj);
                    return Some(format!("{}.iter().cloned().filter({}).count() as i64", obj_s, closure_s));
                }
            }
            // No closure: fallthrough to map_method which maps "count" → "len" as i64.
        }
        // `sortBy(closure)` — in-place sort by key extractor (mutating).
        // Emits: obj.sort_by_key(|param| key_expr)
        if method == "sortBy" {
            if let Some(first_arg) = args.first() {
                if let ExprKind::Closure(params, _, body, _, _) = &first_arg.value.kind {
                    if let Some(param) = params.first() {
                        let pname = &param.name;
                        let mut sub = self.make_sub();
                        for p in params { sub.known_local_vars.insert(p.name.clone()); }
                        let body_s = match body {
                            ClosureBody::Expr(e) => sub.emit_expr(e),
                            ClosureBody::Block(stmts) => {
                                let inner: Vec<String> = stmts.iter().map(|s| sub.emit_stmt_inline(s)).collect();
                                format!("{{ {} }}", inner.join(" "))
                            }
                        };
                        let obj_s = self.emit_expr(obj);
                        return Some(format!(
                            "{}.sort_by(|{}, __boring_b| {{ let __boring_ka = {{ let {} = {}.clone(); {} }}; let __boring_kb = {{ let {} = __boring_b.clone(); {} }}; __boring_ka.partial_cmp(&__boring_kb).unwrap_or(std::cmp::Ordering::Equal) }})",
                            obj_s, pname, pname, pname, body_s, pname, body_s
                        ));
                    }
                }
            }
        }
        // `sortedBy(closure)` → { let mut __v = obj.clone(); __v.sort_by_key(...); __v.iter().cloned().collect::<Vec<_>>() }
        // The collect at the end makes looks_like_collection() return true so print wraps with BoringFmt.
        if method == "sortedBy" {
            if let Some(first_arg) = args.first() {
                if let ExprKind::Closure(params, _, body, _, _) = &first_arg.value.kind {
                    if let Some(param) = params.first() {
                        let pname = &param.name;
                        let mut sub = self.make_sub();
                        for p in params { sub.known_local_vars.insert(p.name.clone()); }
                        let body_s = match body {
                            ClosureBody::Expr(e) => sub.emit_expr(e),
                            ClosureBody::Block(stmts) => {
                                let inner: Vec<String> = stmts.iter().map(|s| sub.emit_stmt_inline(s)).collect();
                                format!("{{ {} }}", inner.join(" "))
                            }
                        };
                        let obj_s = self.emit_expr(obj);
                        return Some(format!(
                            "{{ let mut __boring_v = {}.clone(); __boring_v.sort_by_key(|{}| {{ let {} = {}.clone(); {} }}); __boring_v.iter().cloned().collect::<Vec<_>>() }}",
                            obj_s, pname, pname, pname, body_s
                        ));
                    }
                }
            }
        }
        // `sorted()` → { let mut __v = obj.clone(); __v.sort_by(...); __v.iter().cloned().collect::<Vec<_>>() }
        // The collect at the end makes looks_like_collection() return true so print wraps with BoringFmt.
        if method == "sorted" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return Some(format!(
                "{{ let mut __boring_v = {}.clone(); __boring_v.sort_by(|__a, __b| __a.partial_cmp(__b).unwrap_or(std::cmp::Ordering::Equal)); __boring_v.iter().cloned().collect::<Vec<_>>() }}",
                obj_s
            ));
        }
        // `flat()` → into_iter().flatten().collect::<Vec<_>>()
        if method == "flat" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return Some(format!("{}.into_iter().flatten().collect::<Vec<_>>()", obj_s));
        }
        // `zip(other)` → iter().cloned().zip(other.iter().cloned()).map(|(a,b)| (a,b)).collect::<Vec<_>>()
        if method == "zip" {
            if let Some(other_arg) = args.first() {
                let obj_s   = self.emit_expr(obj);
                let other_s = self.emit_expr(&other_arg.value);
                return Some(format!(
                    "{}.iter().cloned().zip({}.iter().cloned()).map(|(a,b)| (a,b)).collect::<Vec<_>>()",
                    obj_s, other_s
                ));
            }
        }
        // `enumerate()` → iter().cloned().enumerate().map(|(i,x)| (i as i64,x)).collect::<Vec<_>>()
        if method == "enumerate" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return Some(format!(
                "{}.iter().cloned().enumerate().map(|(i,x)| (i as i64,x)).collect::<Vec<_>>()",
                obj_s
            ));
        }
        // `slice(start, end)` on arrays → [start..end].iter().cloned().collect::<Vec<_>>()
        // Using .iter().cloned().collect instead of .to_vec() so looks_like_collection() detects it.
        if method == "slice" && args.len() >= 2 && !self.expr_is_string_receiver(obj) {
            let obj_s   = self.emit_expr(obj);
            let start_s = self.emit_expr(&args[0].value);
            let end_s   = self.emit_expr(&args[1].value);
            return Some(format!("{}[({}) as usize..({}) as usize].iter().cloned().collect::<Vec<_>>()", obj_s, start_s, end_s));
        }
        // `take(n)` → iter().cloned().take(n as usize).collect::<Vec<_>>()
        if method == "take" {
            if let Some(n_arg) = args.first() {
                let obj_s = self.emit_expr(obj);
                let n_s   = self.emit_expr(&n_arg.value);
                return Some(format!(
                    "{}.iter().cloned().take(({}) as usize).collect::<Vec<_>>()",
                    obj_s, n_s
                ));
            }
        }
        // `drop(n)` → iter().cloned().skip(n as usize).collect::<Vec<_>>()
        if method == "drop" {
            if let Some(n_arg) = args.first() {
                let obj_s = self.emit_expr(obj);
                let n_s   = self.emit_expr(&n_arg.value);
                return Some(format!(
                    "{}.iter().cloned().skip(({}) as usize).collect::<Vec<_>>()",
                    obj_s, n_s
                ));
            }
        }
        // `min()` → iter().cloned().min_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or_default()
        if method == "min" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return Some(format!(
                "{}.iter().cloned().min_by(|__a,__b| __a.partial_cmp(__b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or_default()",
                obj_s
            ));
        }
        // `max()` → iter().cloned().max_by(...)
        if method == "max" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return Some(format!(
                "{}.iter().cloned().max_by(|__a,__b| __a.partial_cmp(__b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or_default()",
                obj_s
            ));
        }
        // `sum()` → iter().cloned().sum::<T>() where T is the receiver's known element
        // type. A hardcoded `::<i64>()` broke summing a `[float]` (Sum<f64> isn't
        // implemented for i64); omitting the turbofish entirely instead broke callers
        // with no surrounding type context to infer from (e.g. `print arr.sum()`).
        // Default to i64 (the historical behavior) unless the receiver is positively
        // known to be a float array.
        if method == "sum" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            let elem_ty = self.float_array_elem_ty(obj).unwrap_or("i64");
            return Some(format!("{}.iter().cloned().sum::<{}>()", obj_s, elem_ty));
        }
        // `indexOf(val)` on arrays → iter().position(|x| *x == val).map(|i| i as i64)
        // Note: the map_method fallback maps indexOf → iter().position which returns Option<usize>;
        // we need Option<isize> to match interpreter semantics (indexOf returns `int`).
        if method == "indexOf" && !self.expr_is_string_receiver(obj) {
            if let Some(val_arg) = args.first() {
                let obj_s = self.emit_expr(obj);
                let val_s = self.emit_expr(&val_arg.value);
                return Some(format!(
                    "{}.iter().position(|__x| __x == &({})).map(|i| i as isize)",
                    obj_s, val_s
                ));
            }
        }
        None
    }

    /// Option-chain short-circuit: when the receiver is `Option`-like and the method is one
    /// that operates on `Option` directly (`map`, `filter`, `or`/`or_else`, `flatten`,
    /// `and_then`), emit it without the `iter`/`collect` wrapping the generic fallback would
    /// otherwise add for `Vec` receivers.
    fn try_emit_option_chain_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        const OPTION_CHAIN_METHODS: &[&str] = &["map", "filter", "or", "or_else", "flatten", "and_then"];
        if !(OPTION_CHAIN_METHODS.contains(&method) && self.is_option_expr(obj)) { return None; }
        let obj_s = self.emit_expr(obj);
        let closure_args: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
        let call = format!("{}.{}({})", obj_s, method, closure_args.join(", "));
        Some(if self.in_async && self.instance_task_methods.contains(method) {
            format!("{}.await", call)
        } else {
            call
        })
    }

    /// Float math methods (`sqrt`/`abs`/`floor`/trig/`pow`/`log`/`atan2`/`clamp`/etc.), mapped
    /// to their Rust `f64` equivalents. Only intercepts when the receiver is known to be a
    /// numeric primitive (literal, tracked numeric var, or numeric cast) — not a struct, so a
    /// user-defined method of the same name on a numeric-like struct still dispatches normally.
    fn try_emit_float_math_method(&self, obj: &Expr, method: &str, args: &[Arg]) -> Option<String> {
        let receiver_is_float = match &obj.kind {
            ExprKind::Float(_) => true,
            ExprKind::Int(_)   => true,
            ExprKind::Var(v) => matches!(
                self.var_types.get(v.as_str()),
                Some(crate::ast::Type::Float32) | Some(crate::ast::Type::Float64) | Some(crate::ast::Type::Int) | Some(crate::ast::Type::Uint)
                | Some(crate::ast::Type::Uint8)
                | Some(crate::ast::Type::Int8) | Some(crate::ast::Type::Int16) | Some(crate::ast::Type::Int32)
                | Some(crate::ast::Type::Int64) | Some(crate::ast::Type::Int128)
                | Some(crate::ast::Type::Uint16) | Some(crate::ast::Type::Uint32)
                | Some(crate::ast::Type::Uint64) | Some(crate::ast::Type::Uint128)
            ),
            ExprKind::Cast(_, ty) => matches!(*ty, crate::ast::Type::Float32 | crate::ast::Type::Float64 | crate::ast::Type::Int | crate::ast::Type::Uint
                | crate::ast::Type::Uint8
                | crate::ast::Type::Int8 | crate::ast::Type::Int16 | crate::ast::Type::Int32
                | crate::ast::Type::Int64 | crate::ast::Type::Int128
                | crate::ast::Type::Uint16 | crate::ast::Type::Uint32
                | crate::ast::Type::Uint64 | crate::ast::Type::Uint128),
            _ => false,
        };
        if !receiver_is_float { return None; }
        const FLOAT_UNARY_METHODS: &[(&str, &str)] = &[
            ("sqrt",  "sqrt"),  ("cbrt",   "cbrt"),  ("abs",   "abs"),
            ("floor", "floor"), ("ceil",   "ceil"),  ("round", "round"),
            ("exp",   "exp"),   ("exp2",   "exp2"),  ("ln",    "ln"),
            ("log2",  "log2"),  ("log10",  "log10"),
            ("sin",   "sin"),   ("cos",    "cos"),   ("tan",   "tan"),
            ("asin",  "asin"),  ("acos",   "acos"),  ("atan",  "atan"),
            ("sinh",  "sinh"),  ("cosh",   "cosh"),  ("tanh",  "tanh"),
            ("signum","signum"),("recip",  "recip"),
            ("toRadians","to_radians"), ("toDegrees","to_degrees"),
        ];
        if let Some(&(_, rust_name)) = FLOAT_UNARY_METHODS.iter().find(|&&(n, _)| n == method) {
            let obj_s = self.emit_expr(obj);
            return Some(format!("({} as f64).{}()", obj_s, rust_name));
        }
        if method == "pow" || method == "powf" {
            let obj_s = self.emit_expr(obj);
            let exp = args.first().map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "1.0".to_string());
            return Some(format!("({} as f64).powf({} as f64)", obj_s, exp));
        }
        if method == "log" {
            let obj_s = self.emit_expr(obj);
            let base = args.first().map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "std::f64::consts::E".to_string());
            return Some(format!("({} as f64).log({} as f64)", obj_s, base));
        }
        if method == "atan2" {
            let obj_s = self.emit_expr(obj);
            let other = args.first().map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "0.0f64".to_string());
            return Some(format!("({} as f64).atan2({} as f64)", obj_s, other));
        }
        if method == "clamp" && args.len() >= 2 {
            let obj_s = self.emit_expr(obj);
            let lo = self.emit_expr(&args[0].value);
            let hi = self.emit_expr(&args[1].value);
            return Some(format!("({} as f64).clamp({} as f64, {} as f64)", obj_s, lo, hi));
        }
        None
    }

    pub(crate) fn emit_method_call(&self, obj: &Expr, method: &str, args: &[Arg]) -> String {
        if let Some(r) = self.try_emit_builtin_namespace_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_clone_and_task_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_channel_method(obj, method, args) { return r; }

        if let Some(r) = self.try_emit_type_method_call(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_rwlock_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_mutex_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_actor_field_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_special_case_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_tuple_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_string_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_dict_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_set_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_array_special_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_option_chain_method(obj, method, args) { return r; }
        if let Some(r) = self.try_emit_float_math_method(obj, method, args) { return r; }

        self.emit_method_call_fallback(obj, method, args)
    }

    /// The generic method-call fallback, reached once every specialized dispatch above has
    /// declined: path receivers (`mpsc::channel`, `tokio::time::sleep`), the immutable-param
    /// `def`-method diagnostic, user-struct method-name/overload resolution, Boring→Rust
    /// method-name mapping (`map_method`), arg coercion (dict keys, usize-index methods,
    /// `Option<usize>` index-arg wrapping), and the final `.await`/`?` propagation.
    fn emit_method_call_fallback(&self, obj: &Expr, method: &str, args: &[Arg]) -> String {
        // A method-call receiver that's an array/dict index (`arr[i].method(...)`) must
        // be a genuine place expression, not a fresh clone of the element — cloning here
        // silently drops any mutation a `def` (mutating) method makes, since it would
        // then be mutating the throwaway clone instead of the actual array element (e.g.
        // `blocks[i].step(...)` updating a per-element KV cache). `in_lhs_assign` already
        // signals exactly this ("don't clone, give me the real place") for the assignment-
        // target case in emit_expr's Index arm — reuse it here for the same effect.
        let obj_s = if matches!(&obj.kind, ExprKind::Index(_, _)) {
            self.in_lhs_assign.set(true);
            let s = self.emit_expr(obj);
            self.in_lhs_assign.set(false);
            s
        } else {
            self.emit_expr(obj)
        };
        // Use `::` for module/type path receivers (not instance variable method calls).
        // Lowercase non-local vars: `mpsc.channel(32)` → `mpsc::channel(32)`
        // Cascaded paths: `tokio::time.sleep()` → `tokio::time::sleep()`,
        // but NOT call results: `Path::new(x).exists()` uses `.` (result is a value).
        let is_path_receiver = match &obj.kind {
            ExprKind::Var(v) => {
                // `self`, any known local variable that happens to start with an
                // uppercase letter (e.g. `var Qh = []`), and a top-level `let` constant
                // (`PADDLE_SIZE`) are still values, not type/module paths, and must
                // dispatch as `.method()` -- checked before the uppercase heuristic below.
                if v == "self" || self.known_local_vars.contains(v.as_str()) || self.user_top_level_names.contains(v.as_str()) { false }
                else if v.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) { true }
                else {
                    // Struct fields accessed without `self.` in method bodies are NOT path receivers.
                    let is_struct_field = self.self_type.as_ref()
                        .and_then(|sn| self.struct_fields.get(sn.as_str()))
                        .map(|fields| fields.iter().any(|(f, _)| f == v.as_str()))
                        .unwrap_or(false);
                    !is_struct_field
                }
            }
            _ => obj_s.contains("::") && !obj_s.ends_with(')') && !obj_s.ends_with('}'),
        };
        if is_path_receiver {
            // Normalize bare stdlib module names to their fully-qualified forms.
            // `io.stdin()` → `std::io::stdin()`;  `io::Error` already handled in normalize_type_name.
            let obj_qualified = match obj_s.as_str() {
                "io" => "std::io".to_string(),
                "fs" => "std::fs".to_string(),
                other => other.to_string(),
            };
            let rust_method_name = camel_to_snake(method);
            let vals: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
            let call = format!("{}::{}({})", obj_qualified, rust_method_name, vals.join(", "));
            // Known tokio async static methods (File::open, File::create, etc.) need .await
            const TOKIO_ASYNC_STATIC: &[&str] = &["open", "create", "connect", "bind"];

            if self.in_async && TOKIO_ASYNC_STATIC.iter().any(|&m| m == rust_method_name) {
                return format!("{}.await", call);
            }
            return call;
        }
        // Diagnostic: calling a `def` (mutating) method on a non-`mut` parameter is an error.
        // The developer must declare the parameter `mut T param` in Boring.
        if let ExprKind::Var(v) = &obj.kind {
            if v != "self"
                && self.fn_current_params.contains_key(v.as_str())
                && !self.fn_current_params_mut.contains(v.as_str())
                && !self.var_mutex_types.contains(v.as_str())
                && !self.var_mutex_task_types.contains(v.as_str())
                && !self.var_rwlock_types.contains(v.as_str())
                && !self.var_rwlock_task_types.contains(v.as_str())
            {
                // Check if the param's type is a known user struct and the method is `def` (not `req`).
                let struct_name = self.fn_current_params.get(v.as_str()).and_then(|ty| {
                    if let crate::ast::Type::Named(n) = ty { Some(n.clone()) } else { None }
                });
                if let Some(sn) = struct_name {
                    // A struct always gets this check (unchanged); an enum only when it has
                    // at least one `mut`-qualified variant field — see this function's other
                    // branch below, and `enum_has_mut_field`'s doc.
                    if self.struct_fields.contains_key(sn.as_str()) {
                        let req_key = format!("{}::{}", sn, method);
                        let is_req = self.struct_req_methods.contains(&req_key);
                        if !is_req {
                            let line = self.fn_current_param_lines.get(v.as_str()).copied().unwrap_or(0);
                            let col = self.fn_current_param_cols.get(v.as_str()).copied().unwrap_or(0);
                            self.push_error(line, col, format!("`{}` is not declared `mut` — cannot call `def` method `.{}()` on an immutable binding; fix: declare the parameter as `mut {} {}`", v, method, sn, v));
                        }
                    } else if self.enum_has_mut_field(sn.as_str()) && !self.method_is_req_or_task(sn.as_str(), method) {
                        let line = self.fn_current_param_lines.get(v.as_str()).copied().unwrap_or(0);
                        let col = self.fn_current_param_cols.get(v.as_str()).copied().unwrap_or(0);
                        self.push_error(line, col, format!("`{}` is not declared `mut` — cannot call `def` method `.{}()` on an immutable binding; fix: declare the parameter as `mut {} {}`", v, method, sn, v));
                    }
                }
            } else if v != "self"
                && !self.fn_current_params.contains_key(v.as_str())
                && self.known_local_vars.contains(v.as_str())
                && !self.content_mutable_local_vars.contains(v.as_str())
                && self.mut_checked_local_vars.contains(v.as_str())
                && !self.var_mutex_types.contains(v.as_str())
                && !self.var_mutex_task_types.contains(v.as_str())
                && !self.var_rwlock_types.contains(v.as_str())
                && !self.var_rwlock_task_types.contains(v.as_str())
            {
                // Same diagnostic as above, for a plain local binding rather than
                // a parameter — docs/mut-type-modifier.md's Implementation
                // checklist item 0: `var Point p` alone no longer grants
                // `p.inc()`; only `mut`/`var mut` does. Rust's own borrow
                // checker would NOT catch this on its own (the emitted binding
                // is `let mut` either way — see `BindingKind::is_mutable`'s
                // doc), so this has to be a Boring-level check.
                let struct_name = self.var_struct_types.get(v.as_str()).cloned()
                    .or_else(|| self.var_types.get(v.as_str()).and_then(|t| {
                        if let crate::ast::Type::Named(n) = t.without_mut() { Some(n.clone()) } else { None }
                    }));
                if let Some(sn) = struct_name {
                    // A struct always gets this check; an enum only when it has at least
                    // one `mut`-qualified variant field (`enum_has_mut_field`) — every
                    // other enum keeps `def` == `req` (docs/book.md's "Enum methods"),
                    // so gating those would reject calls that work fine today.
                    let is_gated_type = self.struct_fields.contains_key(sn.as_str())
                        || self.enum_has_mut_field(sn.as_str());
                    if is_gated_type && !self.method_is_req_or_task(sn.as_str(), method) {
                        self.push_error(obj.line, obj.col, format!("`{}` is not declared `mut` — cannot call `def` method `.{}()` on a non-mut binding; fix: declare it `mut {} {}` or `var mut {} {}`", v, method, sn, v, sn, v));
                    }
                }
            }
        }
        // One level down: `recv.field.method()` (including `self.field.method()`)
        // needs `field`'s own declared type to grant `mut` — the reassign/
        // content-mutate split docs/mut-type-modifier.md §3 gives struct
        // fields, independent of whatever `recv` itself allows (checked
        // above). `struct_fields` already carries each field's declared
        // `Type` (already `Type::Mut`-wrapped by the parser where
        // applicable), so no separate lookup of the struct's own AST is
        // needed. Only a bare `Var` owner is resolved (a deeper chain,
        // `a.b.field.method()`, is not — matches the interpreter's identical
        // scope in `eval_expr.rs`).
        if let ExprKind::Field(inner_obj, field_name) = &obj.kind {
            let owner_struct: Option<String> = match &inner_obj.kind {
                ExprKind::Var(v) if v == "self" => self.self_type.clone(),
                ExprKind::Var(v) => self.var_struct_types.get(v.as_str()).cloned()
                    .or_else(|| self.var_types.get(v.as_str()).and_then(|t| {
                        if let crate::ast::Type::Named(n) = t.without_mut() { Some(n.clone()) } else { None }
                    })),
                _ => None,
            };
            if let Some(owner) = owner_struct {
                let field_ty = self.struct_fields.get(owner.as_str())
                    .and_then(|fs| fs.iter().find(|(n, _)| n == field_name).map(|(_, t)| t.clone()));
                if let Some(field_ty) = field_ty {
                    if !field_ty.grants_mut() {
                        let field_struct_name = match field_ty.without_mut() {
                            crate::ast::Type::Named(n) => Some(n.clone()),
                            _ => None,
                        };
                        if let Some(fsn) = field_struct_name {
                            if self.struct_fields.contains_key(fsn.as_str())
                                && !self.method_is_req_or_task(fsn.as_str(), method)
                            {
                                self.push_error(obj.line, obj.col, format!(
                                    "cannot call mutating method `.{}()` on field `.{}` — its declared type doesn't grant content mutation; declare the field `mut {}`/`var mut {}`",
                                    method, field_name, field_name, field_name
                                ));
                            }
                        }
                    }
                }
            }
        }

        // One more level down: `arr[i].method()` (or `dict[k].method()`)
        // needs the collection's own declared *element*/*value* type to
        // grant `mut` — `[mut Point] arr` vs plain `[Point] arr`
        // (docs/mut-type-modifier.md §3). Only resolved for a bare `Var`
        // collection receiver with an explicit type in `var_types` — an
        // inferred-type collection isn't checked, matching the interpreter's
        // identical scope in `eval_expr.rs`.
        if let ExprKind::Index(inner_obj, _idx) = &obj.kind {
            if let ExprKind::Var(coll_name) = &inner_obj.kind {
                if let Some(coll_ty) = self.var_types.get(coll_name.as_str()) {
                    if let Some(elem_ty) = coll_ty.index_element_type() {
                        if !elem_ty.grants_mut() {
                            let elem_struct_name = match elem_ty.without_mut() {
                                crate::ast::Type::Named(n) => Some(n.clone()),
                                _ => None,
                            };
                            if let Some(esn) = elem_struct_name {
                                if self.struct_fields.contains_key(esn.as_str())
                                    && !self.method_is_req_or_task(esn.as_str(), method)
                                {
                                    self.push_error(obj.line, obj.col, format!(
                                        "cannot call mutating method `.{}()` on an element of `{}` — its declared element type doesn't grant content mutation; declare it `[mut {}]`/`{{K = mut {}}}`",
                                        method, coll_name, esn, esn
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // If the receiver is a known user struct, preserve method names (don't remap to Rust builtins).
        // Field chains (e.g. `self.encoder`, a struct field holding another struct instance)
        // are resolved the same way so args to `self.encoder.forward(mel)` get the owned/clone
        // treatment struct methods expect, not the bare emit_expr fallback.
        let is_user_struct_receiver = self.expr_receiver_is_known_user_type(obj);
        // Map boring method names → Rust equivalents
        // For HashSet vars, `add` maps to `insert` (not Vec::push).
        // For dict vars, `contains` maps to `contains_key` (HashMap has no `contains`).
        let receiver_is_set_var = matches!(&obj.kind, ExprKind::Var(v)
            if self.set_vars.contains(v.as_str()));
        let receiver_is_dict_var = matches!(&obj.kind, ExprKind::Var(v)
            if self.dict_vars.contains(v.as_str()));
        // Declared param types of the struct method actually being called, used below to
        // resolve `.Variant` enum shorthand args by the *parameter's* type rather than
        // falling back to the transpiler's flat "last enum registered wins" lookup — see
        // resolve_expr_struct_type / struct_method_overload_decls for the same struct-type
        // resolution used by overload matching just below.
        let mut struct_method_param_types: Vec<Option<Type>> = Vec::new();
        let (rust_method, extra_wrap) = if is_user_struct_receiver {
            // Check for overloaded struct methods — pick the best-matching overload.
            let struct_type_opt = self.resolve_expr_struct_type(obj);
            let chosen_decl = struct_type_opt.as_ref().and_then(|type_name| {
                let key = format!("{}::{}", type_name, method);
                let overloads = self.struct_method_overload_decls.get(&key)?;
                if self.overloaded_method_keys.contains(&key) {
                    // Find best matching overload by arity and inferred arg types
                    Some(overloads.iter().find(|decl| {
                        let min_a = decl.params.iter().filter(|p| p.default.is_none()).count();
                        let max_a = decl.params.len();
                        if args.len() < min_a || args.len() > max_a { return false; }
                        decl.params.iter().zip(args.iter()).all(|(p, a)| {
                            match &p.ty {
                                None => true,
                                Some(expected) => {
                                    if let ExprKind::DotIdent(variant) = &a.value.kind {
                                        self.dotident_matches_enum_type(variant, expected)
                                    } else {
                                        match infer_overload_expr_type(&a.value, &self.var_types, &self.fn_return_types, &self.struct_fields) {
                                            Some(actual) => types_compatible(expected, &actual),
                                            None => true,
                                        }
                                    }
                                }
                            }
                        })
                    }).unwrap_or_else(|| overloads.first().unwrap()))
                } else {
                    overloads.first()
                }
            });
            struct_method_param_types = chosen_decl
                .map(|decl| decl.params.iter().map(|p| p.ty.clone()).collect())
                .unwrap_or_default();
            let overloaded_name = chosen_decl.filter(|_| {
                struct_type_opt.as_ref()
                    .map(|t| self.overloaded_method_keys.contains(&format!("{}::{}", t, method)))
                    .unwrap_or(false)
            }).map(|decl| mangle_overload_name(method, &decl.params));
            (overloaded_name.unwrap_or_else(|| method.to_string()), None)
        } else if receiver_is_set_var && method == "add" {
            ("insert".to_string(), None)
        } else if receiver_is_dict_var && (method == "contains" || method == "containsKey" || method == "has") {
            ("contains_key".to_string(), None)
        } else if receiver_is_dict_var && (method == "set" || method == "put") {
            // dict.set(key, val) / dict.put(key, val) → dict.insert(key, val)
            ("insert".to_string(), None)
        } else {
            let (m, w) = map_method(method, args.len());
            (m, w)
        };
        // HashSet::contains needs a plain reference; HashMap::contains_key needs &String.
        let args_s: Vec<String> = if rust_method == "contains_key" && receiver_is_dict_var {
            // Use emit_dict_key_borrow so the key is &str (Arc<str> via Deref).
            args.iter().map(|a| self.emit_dict_key_borrow(&a.value)).collect()
        } else if receiver_is_dict_var && rust_method == "insert" {
            // dict.insert(key, val) — key must be owned Arc<str>, val is emit_expr_owned.
            args.iter().enumerate().map(|(i, a)| {
                if i == 0 { self.emit_dict_key_owned(&a.value) }
                else { self.emit_expr_owned(&a.value) }
            }).collect()
        } else if rust_method == "contains" || rust_method == "contains_key" {
            args.iter().map(|a| format!("&{}", self.emit_expr(&a.value))).collect()
        } else if is_user_struct_receiver {
            // User-defined struct methods: coerce string literals to Arc<str> to match
            // generated method signatures (Boring `string` params map to `Arc<str>`).
            // `.Variant` enum shorthand args are resolved against the method's own
            // declared param type (struct_method_param_types) instead of falling through
            // to emit_expr_owned's flat "last enum registered wins" DotIdent lookup —
            // this is what lets `obj.method(.Left)` pick the right enum when multiple
            // enums share a `Left` variant.
            args.iter().enumerate().map(|(i, a)| {
                if matches!(&a.value.kind, ExprKind::DotIdent(_)) {
                    if let Some(Some(param_ty)) = struct_method_param_types.get(i) {
                        return self.emit_let_value(Some(param_ty), &a.value);
                    }
                }
                self.emit_expr_owned(&a.value)
            }).collect()
        } else if (rust_method == "push" || rust_method == "extend") && {
            // Use emit_expr_owned for any vec push so non-Copy values (e.g. Value enum) are cloned.
            match &obj.kind {
                ExprKind::Var(v) => self.vec_vars.contains(v.as_str()) || self.str_vec_vars.contains(v.as_str()),
                _ => false,
            }
        } {
            args.iter().map(|a| self.emit_expr_owned(&a.value)).collect()
        } else {
            args.iter().map(|a| self.emit_expr(&a.value)).collect()
        };
        // Rust collection methods that take a positional usize index:
        // Boring's `uint` maps to u64, so cast the first argument to usize.
        // Exception: `remove` on HashMap/HashSet takes &K/&T (not a usize index).
        // We use self_type (the type being impl'd) to distinguish Vec::remove from
        // HashMap::remove / HashSet::remove.
        const USIZE_INDEX_METHODS: &[&str] = &[
            "nth", "insert", "split_at", "split_off",
            "drain", "truncate", "rotate_left", "rotate_right", "swap",
        ];
        // HashSet::insert takes T (not usize), so exclude set vars from usize coercion.
        // HashMap::insert takes (K, V) — also exclude dict vars from usize coercion.
        let receiver_is_set = matches!(&obj.kind, ExprKind::Var(v)
            if self.set_vars.contains(v.as_str()));
        let args_s: Vec<String> = if USIZE_INDEX_METHODS.contains(&rust_method.as_str()) && !args_s.is_empty() && !receiver_is_set && !receiver_is_dict_var {
            let mut v = args_s;
            v[0] = format!("({} as usize)", v[0]);
            v
        } else if rust_method == "remove" && !args_s.is_empty() {
            // HashMap::remove(&K) and HashSet::remove(&T) take a reference.
            // Vec::remove(usize) and all other cases (field-accessed vecs, etc.) take usize.
            let self_is_hash = matches!(self.self_type.as_deref(),
                Some("HashMap") | Some("HashSet"));
            let receiver_is_set = matches!(&obj.kind, ExprKind::Var(v)
                if self.set_vars.contains(v.as_str()));
            let mut v = args_s;
            if self_is_hash || receiver_is_set || receiver_is_dict_var {
                // For dict vars, the key must be a &String (same Borrow rule as .get())
                if receiver_is_dict_var {
                    // Re-emit the key as a borrow-compatible &String reference
                    if let Some(first_arg) = args.first() {
                        v[0] = self.emit_dict_key_borrow(&first_arg.value);
                    }
                } else {
                    v[0] = format!("&{}", v[0]);
                }
            } else {
                // Vec::remove(usize): support negative indices (Boring Python-style).
                // Emit a bounds-safe index expression: if i < 0 → len + i, else i.
                let recv_s = self.emit_expr(obj);
                v[0] = format!(
                    "{{ let __boring_idx = {} as i64; if __boring_idx < 0 {{ ({}.len() as i64 + __boring_idx) as usize }} else {{ __boring_idx as usize }} }}",
                    v[0], recv_s
                );
            }
            v
        } else {
            args_s
        };
        // `nextIndex(idx)` / `getAt(idx)` / `removeAt(idx)` — the arg `idx` is bound by
        // `while let Some(idx) = i` (usize), but these methods expect `Option<usize>`.
        // Wrap the first argument in `Some(...)` when it is not already an index_var.
        // `nextIndex(idx)` / `getAt(idx)` / `removeAt(idx)` — when the first arg is a plain
        // variable bound by `while let Some(idx) = i` (type usize), these Rust methods expect
        // `Option<usize>`. Wrap such non-index-var Var args in `Some(...)`.
        // Only applies when the arg is a Var that is NOT already in index_vars (which carry
        // Option<usize> directly). Non-Var expressions (method calls like `first_index()`)
        // already return Option<usize> and must not be wrapped.
        const INDEX_ARG_METHODS: &[&str] = &["next_index", "get_at", "remove_at"];
        let args_s: Vec<String> = if INDEX_ARG_METHODS.contains(&rust_method.as_str()) && !args_s.is_empty() {
            let needs_wrap = args.first()
                .map(|a| match &a.value.kind {
                    ExprKind::Var(v) => !self.index_vars.contains(v.as_str()),
                    _ => false, // method calls / literals already return Option<usize>
                })
                .unwrap_or(false);
            if needs_wrap {
                let raw_first = args_s[0].clone();
                let mut v = args_s;
                v[0] = format!("Some({})", raw_first);
                v
            } else {
                args_s
            }
        } else {
            args_s
        };
        let call = format!("{}.{}({})", obj_s, rust_method, args_s.join(", "));
        let call = if let Some(wrap) = extra_wrap {
            format!("{}{}", call, wrap)
        } else {
            call
        };
        // Add .await for task (async) instance methods when inside an async context.
        // Also await known tokio async I/O methods that aren't declared in Boring source.
        // These names are specific enough to tokio/async-std that sync conflicts are unlikely.
        const TOKIO_ASYNC_METHODS: &[&str] = &[
            "read_line", "write_all", "acquire", "recv",
        ];
        let is_tokio_async = TOKIO_ASYNC_METHODS.contains(&method);
        // When the receiver's struct type is known, check the qualified
        // "StructName::method" key ONLY -- otherwise a same-named-but-non-throwing
        // method on a different struct (e.g. EncoderBlock.forward vs
        // AudioEncoder.forward, both "forward") would incorrectly pick up a stray
        // `?` from the bare-name entry. Fall back to the bare-name check only when
        // the receiver's struct type can't be resolved here.
        let receiver_struct = match &obj.kind {
            ExprKind::Var(v) if v == "self" => self.self_type.clone(),
            ExprKind::Var(v) => self.var_struct_types.get(v.as_str())
                .or_else(|| self.var_struct_type.get(v.as_str()))
                .cloned(),
            _ => None,
        };
        let struct_throws = match &receiver_struct {
            Some(sn) => self.struct_method_throws.contains(&format!("{}::{}", sn, method)),
            None => self.struct_method_throws.contains(method),
        };
        let propagates_throw = (self.in_throws || self.in_try_body)
            && (self.fn_throws.contains(method) || struct_throws);
        if self.in_async && (self.instance_task_methods.contains(method) || is_tokio_async) {
            if propagates_throw { format!("{}.await?", call) } else { format!("{}.await", call) }
        } else if propagates_throw {
            format!("{}?", call)
        } else {
            call
        }
    }

    /// Resolve a leading-dot expression with a known type name.
    ///
    /// Returns `Some(String)` when `expr` is a dot-call or dot-ident:
    ///   `.fromSecs(5)` + `"Duration"` → `Some("Duration::from_secs(5)")`
    ///   `.Expired`     + `"Error"`    → `Some("Error::Expired")`
    ///
    /// Returns `None` for all other expressions — callers fall back to `emit_expr`.
    pub(crate) fn resolve_dot_with_type(&self, expr: &Expr, type_name: &str) -> Option<String> {
        match &expr.kind {
            // `.method(args)` → `TypeName::method(args)`
            ExprKind::Call(callee, dot_args) => {
                if let ExprKind::DotIdent(method) = &callee.kind {
                    let rust_method = camel_to_snake(method);
                    let vals: Vec<String> = dot_args.iter()
                        .map(|a| self.resolve_dot_with_type(&a.value, type_name)
                            .unwrap_or_else(|| self.emit_expr(&a.value)))
                        .collect();
                    return Some(format!("{}::{}({})", type_name, rust_method, vals.join(", ")));
                }
                None
            }
            // `.Variant` → `TypeName::Variant`
            ExprKind::DotIdent(variant) => {
                Some(format!("{}::{}", type_name, variant))
            }
            _ => None,
        }
    }

    /// Emit an expression with an optional type hint.
    ///
    /// When the hint is a `Named` type and the expression is a leading-dot call
    /// or ident (`.fromSecs(5)`, `.Expired`), the type prefix is prepended:
    ///   `.fromSecs(5)`  + `Duration`  →  `Duration::from_secs(5)`
    ///   `.Expired`      + `Error`     →  `Error::Expired`
    /// Falls back to `emit_expr` when no hint applies.
    pub(crate) fn emit_args(&self, args: &[Arg]) -> String {
        args.iter().map(|a| self.emit_expr_owned(&a.value)).collect::<Vec<_>>().join(", ")
    }

    /// Like emit_args but coerces non-nil values to Some(v) when the param is Optional,
    /// handles string-type params via emit_expr_owned, and fills missing args with defaults.
    /// Also reorders labeled (named) arguments to match the declared parameter order.
    pub(crate) fn emit_args_coerced(&self, fn_name: &str, args: &[Arg]) -> String {
        let sig = self.fn_sigs.get(fn_name).cloned().unwrap_or_default();
        let rebindable_flags = self.fn_rebindable.get(fn_name).cloned().unwrap_or_default();
        let mutable_flags = self.fn_mutable.get(fn_name).cloned().unwrap_or_default();
        let defaults = self.fn_defaults.get(fn_name).cloned().unwrap_or_default();
        let variadic_idx = self.fn_variadic.get(fn_name).copied();
        let param_names = self.fn_param_names.get(fn_name).cloned().unwrap_or_default();
        // If any arg has a label and we know the param names, reorder args to parameter order.
        let args: Vec<&Arg> = if !param_names.is_empty() && args.iter().any(|a| a.label.is_some()) {
            let mut ordered: Vec<Option<&Arg>> = vec![None; param_names.len()];
            let mut positional_idx = 0;
            for a in args.iter() {
                if let Some(label) = &a.label {
                    // Named arg: find its position in param_names.
                    if let Some(pos) = param_names.iter().position(|n| n == label) {
                        ordered[pos] = Some(a);
                    }
                } else {
                    // Positional arg: fill the next unoccupied slot.
                    while positional_idx < ordered.len() && ordered[positional_idx].is_some() {
                        positional_idx += 1;
                    }
                    if positional_idx < ordered.len() {
                        ordered[positional_idx] = Some(a);
                        positional_idx += 1;
                    }
                }
            }
            ordered.into_iter().flatten().collect()
        } else {
            args.iter().collect()
        };
        let n_params = sig.len().max(args.len());
        let mut result: Vec<String> = Vec::new();
        let mut i = 0;
        while i < n_params {
            if variadic_idx == Some(i) {
                // Collect all remaining args into vec![...]
                let elem_ty = sig.get(i);
                let elems: Vec<String> = args[i..].iter()
                    .map(|a| self.emit_let_value(elem_ty, &a.value))
                    .collect();
                result.push(format!("vec![{}]", elems.join(", ")));
                break;
            }
            if let Some(a) = args.get(i) {
                // Interprocedural GPU residency (docs/scoped-access-blocks.md): this
                // position is known, via the checker's bounded body scan (mirrored
                // here in `fn_gpu_arg_params`), to be consumed *only* as a kernel-
                // constructor argument in the callee — by construction its declared
                // type is always a plain, unqualified array, so none of the richer
                // per-position coercion below (weak-ref checks, Borrow handling,
                // hierarchy checks) can ever actually apply to it. Short-circuit
                // straight to the `BoringGpuArg<T>` wrap instead: pass an already-
                // resident value through directly (`resident_call_vars` — an
                // `Arc::clone` inside, no data copy), otherwise wrap the plain host
                // value as `BoringGpuArg::Host(...)` for upload as today. Caller keeps
                // ownership either way (`.clone()` on a bare variable), matching
                // Boring's normal by-value argument semantics.
                if self.fn_gpu_arg_params.get(fn_name).and_then(|flags| flags.get(i)).copied().unwrap_or(false) {
                    let wrapped = if let ExprKind::Var(vname) = &a.value.kind {
                        // Already a `BoringGpuArg<T>`-typed Rust binding either way:
                        // `resident_call_vars` (a same-function local bound to a
                        // `fn_returns_resident` call) or `current_fn_gpu_arg_param_names`
                        // (the *enclosing* function's own transitively-qualifying
                        // parameter, forwarded straight through — the two-hop wrapper
                        // case: `wrap_scale`'s `x` is itself `BoringGpuArg<T>`, so
                        // re-wrapping it in `Host(...)` on the way into `scale(x, n)`
                        // would be a type error, not just a missed optimization).
                        if self.resident_call_vars.contains_key(vname.as_str())
                            || self.current_fn_gpu_arg_param_names.contains(vname.as_str())
                        {
                            format!("{}.clone()", vname)
                        } else {
                            format!("BoringGpuArg::Host({}.clone())", vname)
                        }
                    } else {
                        // Any other expression shape (most commonly a struct field —
                        // `linear_gpu(x, self.mlp_fc_w, ...)`) needs the same clone the
                        // `Var` branch above gives a bare local: `self.mlp_fc_w` is
                        // behind `&self`/`&mut self`, and moving it into
                        // `BoringGpuArg::Host(...)` without cloning is a real E0507,
                        // confirmed against a real `cargo check` on whisper-boring's
                        // actual call sites (every weight-matrix argument is a struct
                        // field, never a bare local).
                        format!("BoringGpuArg::Host(({}).clone())", self.emit_expr(&a.value))
                    };
                    result.push(wrapped);
                    i += 1;
                    continue;
                }
                // Interprocedural GPU residency, the materializing counterpart to the
                // `fn_gpu_arg_params` branch above: this position does NOT want a
                // `BoringGpuArg<T>` (checked above and not taken), but the argument
                // itself is a bare `resident_call_vars` local (`let fc =
                // linear_gpu(...)?`) — a real `BoringGpuArg<T>` Rust value, not the
                // plain array the callee's signature declares. The generic by-ref
                // coercion below has no idea this Var's actual Rust type diverges
                // from its boring-level declared type, and would emit a bare `&fc`
                // — a genuine E0308 confirmed via a real `cargo check` on
                // whisper-boring's `mha_gpu` calling `attention_heads_gpu`/
                // `linear_gpu`/`gelu_gpu` (this exact bug is also present, unfixed,
                // in a plain `boring build`'s own output — pre-existing and
                // target-agnostic, not introduced by any GPU-backend work).
                // Materializes by reference (mirrors `emit_stmt.rs`'s identical
                // by-ref materialization for a reassignment target) so `fc` itself
                // stays usable afterward, matching boring's no-moves semantics.
                if let ExprKind::Var(vname) = &a.value.kind {
                    if let Some(ret_ty) = self.resident_call_vars.get(vname.as_str()).cloned() {
                        let inner_ty = match &ret_ty {
                            Type::Qualified(inner, _) => crate::transpiler::emit_kernel::array_inner_type(inner),
                            other => crate::transpiler::emit_kernel::array_inner_type(other),
                        };
                        let host_ty = crate::transpiler::emit_kernel::kernel_host_element_type(&inner_ty);
                        let device_ty = crate::transpiler::emit_kernel::kernel_host_scalar_type(&inner_ty);
                        let materialized = format!(
                            "match &{vname} {{ BoringGpuArg::Resident(buf, _) => __boring_gpu_copy_d2h::<{device_ty}>(&__boring_gpu_device(), &__boring_gpu_queue(), buf).iter().map(|&x| x as {host_ty}).collect::<Vec<{host_ty}>>(), BoringGpuArg::Host(v) => v.clone() }}"
                        );
                        result.push(format!("&({})", materialized));
                        i += 1;
                        continue;
                    }
                }
                let param_ty = sig.get(i);
                let param_rebindable = rebindable_flags.get(i).copied().unwrap_or(false);
                let param_mutable = mutable_flags.get(i).copied().unwrap_or(false);
                // When passing a `throws` function where a non-throws fn param is expected,
                // wrap it in a closure that unwraps the Result.
                let coerced = if let ExprKind::Var(fn_name) = &a.value.kind {
                    if self.fn_throws.contains(fn_name.as_str()) {
                        // Check if the expected param type is a non-throws fn type.
                        let param_is_nothrows_fn = match param_ty {
                            Some(Type::Fn(_, params, false, _, _)) => Some(params.clone()),
                            Some(Type::Named(n)) => {
                                if let Some(Type::Fn(_, params, false, _, _)) = self.fn_type_aliases.get(n.as_str()) {
                                    Some(params.clone())
                                } else { None }
                            }
                            _ => None,
                        };
                        if let Some(param_types) = param_is_nothrows_fn {
                            // Build closure: |p0, p1, ...| fn_name(p0, p1, ...).unwrap_or_default()
                            let param_names: Vec<String> = (0..param_types.len())
                                .map(|j| format!("__p{}", j)).collect();
                            let params_s = param_names.join(", ");
                            Some(format!("|{}| {fn_name}({}).unwrap_or_default()",
                                params_s, params_s, fn_name = fn_name))
                        } else { None }
                    } else { None }
                } else { None };
                // Error: 'weak argument passed to a non-weak parameter.
                // upgrade() returns Option — the transpiler cannot insert it implicitly.
                if let ExprKind::Var(vname) = &a.value.kind {
                    let arg_ty = self.fn_current_params.get(vname.as_str())
                        .or_else(|| self.var_types.get(vname.as_str()));
                    let arg_is_weak = matches!(arg_ty, Some(Type::Qualified(_, OwnerQual::Weak)));
                    let param_is_non_weak = matches!(param_ty,
                        Some(Type::Qualified(_, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard)) |
                        Some(Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut))
                    );
                    if arg_is_weak && param_is_non_weak {
                        self.push_error(a.value.line, a.value.col, format!("cannot pass `{}` (weak reference) to a non-weak parameter — weak references may be invalid. Call .upgrade() first and handle the Option.", vname));
                    }
                    // Hierarchy check: let < mut < var — caller cannot pass up the hierarchy.
                    // let → mut param: immutable argument passed to a mutable (non-rebindable) parameter.
                    if param_mutable && !param_rebindable && self.immutable_local_vars.contains(vname.as_str()) {
                        self.push_error(a.value.line, a.value.col, format!(
                            "cannot pass `{}` to a `mut` parameter — `{}` is immutable (`let` binding). Use `var` or `mut` instead.",
                            vname, vname
                        ));
                    }
                }
                // For Borrow/BorrowMut params, pass the inner type to emit_let_value so it does
                // not prepend a `&` — the Borrow branch below is responsible for adding `&`.
                let borrow_inner: Option<&Type> = match param_ty {
                    Some(Type::Qualified(inner, OwnerQual::Borrow | OwnerQual::BorrowMut)) => Some(inner.as_ref()),
                    _ => None,
                };
                // When borrow_inner is a named alias that resolves to a smart-pointer type
                // (e.g. `ONode& n` where `ONode = OTree'shared = Rc<OTree>`), the Rust param
                // is `&Rc<OTree>`, not `&OTree`. Detect this early and pass `&var` directly.
                let borrow_inner_is_smart_ptr = borrow_inner.map(|bi| {
                    if let Type::Named(n) = bi {
                        matches!(
                            self.non_fn_type_aliases.get(n.as_str()),
                            Some(Type::Qualified(_, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard
                                | OwnerQual::ActorTask | OwnerQual::GuardTask))
                        )
                    } else { false }
                }).unwrap_or(false);
                let emit_ty = borrow_inner.map(Some).unwrap_or(param_ty);
                let emitted = coerced.unwrap_or_else(|| self.emit_let_value(emit_ty, &a.value));
                // Strip a trailing `.clone()` added by emit_let_value for struct variables when
                // the param is Borrow/BorrowMut — the Borrow branch below provides the `&`, so
                // cloning just before taking a reference is wasteful.
                let emitted = if borrow_inner.is_some() {
                    emitted.strip_suffix(".clone()").map(str::to_owned).unwrap_or(emitted)
                } else {
                    emitted
                };
                // Auto-clone: boring has value semantics (no moves).
                // Field accesses can never be moved out of a struct in Rust; non-Copy variables
                // may be used again after the call. Add .clone() unless:
                //   - the param type is Copy (int/float/bool/uint/reference)
                //   - the emitted string already ends with .clone()
                //   - the emitted string starts with & (already a reference)
                //   - emit_let_value already produced a fresh owned value (Arc::, Rc::, Vec::,
                //     actor borrow block, etc.) — cloning further would be wrong/wasteful
                let emitted_needs_clone = !emitted.ends_with(".clone()")
                    && !emitted.starts_with('&')
                    && !emitted.starts_with("Arc::")
                    && !emitted.starts_with("Rc::")
                    && !emitted.starts_with("Vec::")
                    && !emitted.starts_with("{ let __g")
                    && !param_ty_is_copy(param_ty)
                    && !param_rebindable
                    // Borrow params take a reference — no clone needed, the Borrow branch adds `&`.
                    && !matches!(param_ty, Some(Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut)));
                let emitted = if emitted_needs_clone {
                    match &a.value.kind {
                        ExprKind::Field(..) => format!("{}.clone()", emitted),
                        ExprKind::Var(vname) if
                            self.var_struct_types.contains_key(vname.as_str())
                            || self.collection_vars.contains(vname.as_str())
                            || self.vec_vars.contains(vname.as_str())
                            || self.dict_vars.contains(vname.as_str())
                            || self.set_vars.contains(vname.as_str())
                            || self.string_arc_vars.contains(vname.as_str())
                            || self.tuple_vars.contains_key(vname.as_str())
                            // Function params with non-Copy types.
                            || matches!(
                                self.fn_current_params.get(vname.as_str()),
                                Some(Type::Named(_) | Type::Array(_) | Type::Dict(..) | Type::Set(_) | Type::Optional(_))
                            )
                            // Local variables whose type was tracked as a named/collection type.
                            || matches!(
                                self.var_types.get(vname.as_str()),
                                Some(Type::Named(_) | Type::Array(_) | Type::Dict(..) | Type::Set(_) | Type::Optional(_))
                            ) => {
                            format!("{}.clone()", emitted)
                        }
                        _ => emitted,
                    }
                } else {
                    emitted
                };
// Counter& coercion: parameter expects &T (Borrow/BorrowMut), caller may hold
                // any qualifier. Wrap the emitted argument with the appropriate deref.
                let emitted = if matches!(param_ty, Some(Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut))) {
                    let mutable = matches!(param_ty, Some(Type::Qualified(_, OwnerQual::BorrowMut)));
                    let ref_prefix = if mutable { "&mut " } else { "&" };
                    // borrow_inner is a smart-pointer alias (e.g. ONode = OTree'shared).
                    // The Rust param is &Rc<T>, so pass `&var` directly — no clone, no deref.
                    if borrow_inner_is_smart_ptr {
                        let base = if let ExprKind::Var(vname) = &a.value.kind {
                            vname.as_str().to_owned()
                        } else {
                            emitted.strip_prefix("Rc::clone(&")
                                .and_then(|s| s.strip_suffix(')'))
                                .or_else(|| emitted.strip_prefix("Arc::clone(&").and_then(|s| s.strip_suffix(')')))
                                .unwrap_or(&emitted)
                                .to_owned()
                        };
                        result.push(format!("{}{}", ref_prefix, base));
                        i += 1;
                        continue;
                    }
                    let _actor_placeholder;
                    let _guard_placeholder;
                    let _shared_placeholder;
                    let _borrow_placeholder;
                    let _borrow_mut_placeholder;
                    let arg_qual = if let ExprKind::Var(vname) = &a.value.kind {
                        if self.var_mutex_types.contains(vname.as_str()) {
                            _actor_placeholder = Type::Qualified(Box::new(Type::Named(String::new())), OwnerQual::Actor);
                            Some(&_actor_placeholder)
                        } else if self.var_rwlock_types.contains(vname.as_str()) {
                            _guard_placeholder = Type::Qualified(Box::new(Type::Named(String::new())), OwnerQual::Guard);
                            Some(&_guard_placeholder)
                        } else if self.arc_vars.contains(vname.as_str()) {
                            _shared_placeholder = Type::Qualified(Box::new(Type::Named(String::new())), OwnerQual::Shared);
                            Some(&_shared_placeholder)
                        } else if matches!(self.inferred_qualifiers.get(vname.as_str()), Some(OwnerQual::Borrow)) {
                            _borrow_placeholder = Type::Qualified(Box::new(Type::Named(String::new())), OwnerQual::Borrow);
                            Some(&_borrow_placeholder)
                        } else if matches!(self.inferred_qualifiers.get(vname.as_str()), Some(OwnerQual::BorrowMut)) {
                            _borrow_mut_placeholder = Type::Qualified(Box::new(Type::Named(String::new())), OwnerQual::BorrowMut);
                            Some(&_borrow_mut_placeholder)
                        } else {
                            self.fn_current_params.get(vname.as_str())
                                .or_else(|| self.var_types.get(vname.as_str()))
                        }
                    } else { None };
                    // Q4: 'shared → mut Counter& is a compile error — 'shared has no interior mutability.
                    if mutable && matches!(arg_qual, Some(Type::Qualified(_, OwnerQual::Shared))) {
                        if let ExprKind::Var(vname) = &a.value.kind {
                            self.push_error(a.value.line, a.value.col, format!("cannot pass `{}` ('shared) to `mut Counter&` — 'shared does not support mutable references. Use 'actor (Arc<Mutex<T>>) or 'guard (Arc<RwLock<T>>) instead.", vname));
                        }
                    }
                    match arg_qual {
                        Some(Type::Qualified(_, OwnerQual::Stack)) =>
                            format!("{}{}", ref_prefix, emitted),   // &val (T on stack, no deref needed)
                        Some(Type::Qualified(_, OwnerQual::Owned)) =>
                            format!("{}*{}", ref_prefix, emitted),  // &*box_val (Box<T> → T)
                        Some(Type::Qualified(_, OwnerQual::Shared)) =>
                            format!("{}*{}", ref_prefix, emitted), // &*rc (Rc<T>/Arc<T> → T, one deref)
                        Some(Type::Qualified(_, OwnerQual::Actor)) => {
                            // Q5: MutexGuard held across .await — reject the combination.
                            if self.task_fns.contains(fn_name) {
                                self.push_error(a.value.line, a.value.col, format!("cannot pass 'actor argument to `mut Counter&` in async function `{}` — holding a MutexGuard across .await makes the future !Send. Acquire the lock inside the callee body instead.", fn_name));
                            }
                            // Use the raw variable name (not the auto-ref'd form) for lock acquisition,
                            // so the guard lifetime is correct: `let __g = ac.borrow()` not `&ac.borrow()`.
                            let raw = if let ExprKind::Var(vname) = &a.value.kind {
                                vname.clone()
                            } else {
                                emitted.strip_prefix('&').unwrap_or(&emitted).to_string()
                            };
                            let lock_method = if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                                if mutable { "borrow_mut" } else { "borrow" }
                            } else { "lock" };
                            format!("{{ let __g = {}.{}(); {}*__g }}", raw, lock_method, ref_prefix)
                        }
                        Some(Type::Qualified(_, OwnerQual::Guard)) => {
                            // Q5: RwLockGuard held across .await — same issue.
                            if self.task_fns.contains(fn_name) {
                                self.push_error(a.value.line, a.value.col, format!("cannot pass 'guard argument to `mut Counter&` in async function `{}` — holding an RwLockGuard across .await makes the future !Send. Acquire the lock inside the callee body instead.", fn_name));
                            }
                            let raw = if let ExprKind::Var(vname) = &a.value.kind {
                                vname.clone()
                            } else {
                                emitted.strip_prefix('&').unwrap_or(&emitted).to_string()
                            };
                            let lock_method = if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                                if mutable { "borrow_mut" } else { "borrow" }
                            } else if mutable { "write" } else { "read" };
                            format!("{{ let __g = {}.{}(); {}*__g }}", raw, lock_method, ref_prefix)
                        }
                        // Arg is already a borrow (Counter&): pass through without adding another &.
                        Some(Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut)) => {
                            if let ExprKind::Var(vname) = &a.value.kind {
                                vname.clone()
                            } else {
                                emitted.strip_prefix('&').unwrap_or(&emitted).to_string()
                            }
                        }
                        _ => format!("{}{}", ref_prefix, emitted),
                    }
                // Qualified smart-pointer params ('shared/'actor/'guard/'weak) are passed by
                // owned value — emit_let_value already handles Rc::clone / Arc::clone.
                // Exception: trait object wrappers (Rc<dyn Trait> / Arc<dyn Trait>) also pass
                // by value (unsized coercion requires an owned value anyway).
                } else if is_auto_ref_param(param_ty) || is_unqualified_actor_source_param(param_ty, &self.actor_source_types) {
                    // Actor/Guard params (explicit or inferred) are declared as `&Arc<Mutex<T>>`.
                    // Pass by reference: strip the auto-added `.clone()` and prepend `&`.
                    let is_actor_or_guard = matches!(param_ty,
                        Some(Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask | OwnerQual::Guard | OwnerQual::GuardTask))
                    ) || is_unqualified_actor_source_param(param_ty, &self.actor_source_types);
                    if is_actor_or_guard {
                        if emitted.starts_with('&') {
                            // Already a reference — pass through.
                            emitted
                        } else if let Some(inner) = emitted.strip_prefix("Rc::clone(&").and_then(|s| s.strip_suffix(')'))
                            .or_else(|| emitted.strip_prefix("Arc::clone(&").and_then(|s| s.strip_suffix(')')))
                        {
                            // Rc::clone(&x) / Arc::clone(&x) → &x (borrow without refcount bump).
                            format!("&{}", inner)
                        } else if emitted.starts_with("Arc::") || emitted.starts_with("Rc::") {
                            // Other Rc::/Arc:: expressions (e.g. Rc::new(...)): take a reference.
                            format!("&{}", emitted)
                        } else {
                            let base = emitted.strip_suffix(".clone()").unwrap_or(&emitted);
                            format!("&{}", base)
                        }
                    } else {
                        emitted
                    }
                // var (rebindable) param: out-parameter, pass &mut for all stack/primitive types.
                } else if param_rebindable && !matches!(param_ty,
                    Some(Type::Qualified(_, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Weak))
                ) {
                    if let ExprKind::Var(vname) = &a.value.kind {
                        if self.immutable_local_vars.contains(vname.as_str()) {
                            self.push_error(a.value.line, a.value.col, format!(
                                "cannot pass `{}` to a `var` out-parameter — `{}` is immutable (`let` binding). Use `var` instead.",
                                vname, vname
                            ));
                        } else if self.mut_local_vars.contains(vname.as_str()) {
                            self.push_error(a.value.line, a.value.col, format!(
                                "cannot pass `{}` to a `var` out-parameter — `{}` is `mut` (non-rebindable). Use `var` instead.",
                                vname, vname
                            ));
                        }
                    }
                    format!("&mut {}", emitted)
                } else {
                    emitted
                };
                result.push(emitted);
            } else {
                result.push(
                    defaults.get(i).and_then(|d| d.clone())
                        .unwrap_or_else(|| "/* missing arg */".to_string())
                );
            }
            i += 1;
        }
        result.join(", ")
    }

    pub(crate) fn emit_macro(&self, name: &str, args: &[Expr]) -> String {
        // Most macros pass through; some need special handling
        match name {
            "println" | "print" | "eprintln" | "eprint" | "format" => {
                if let Some(first) = args.first() {
                    let rest: Vec<String> = args.iter().skip(1).map(|e| self.emit_expr(e)).collect();
                    // Literal string first arg: pass through as a raw Rust format string.
                    // The `{}` holes in it are Rust format placeholders — do NOT escape them.
                    let fmt_str = match &first.kind {
                        ExprKind::Str(s) => Some((escape_str_macro(s), Vec::new())),
                        ExprKind::StringInterp(segs) => {
                            // Boring interpolation: rebuild as Rust format string, along
                            // with the emitted expression for each `{}` / `{:spec}` hole.
                            Some(self.build_macro_format_string(segs))
                        }
                        _ => None,
                    };
                    if let Some((fmt, interp_args)) = fmt_str {
                        let all_args: Vec<String> = interp_args.into_iter().chain(rest).collect();
                        return if all_args.is_empty() {
                            format!("{}!(\"{}\")", name, fmt)
                        } else {
                            format!("{}!(\"{}\", {})", name, fmt, all_args.join(", "))
                        };
                    }
                }
                let args_s: Vec<String> = args.iter().map(|e| self.emit_expr(e)).collect();
                format!("{}!({})", name, args_s.join(", "))
            }
            "vec" => {
                let args_s: Vec<String> = args.iter().map(|e| self.emit_expr(e)).collect();
                format!("vec![{}]", args_s.join(", "))
            }
            "assert" | "assert_eq" | "assert_ne" | "panic" | "todo" | "unimplemented" | "unreachable" | "dbg" => {
                let args_s: Vec<String> = args.iter().map(|e| self.emit_expr(e)).collect();
                format!("{}!({})", name, args_s.join(", "))
            }
            _ => {
                let args_s: Vec<String> = args.iter().map(|e| self.emit_expr(e)).collect();
                format!("{}!({})", name, args_s.join(", "))
            }
        }
    }

    // ── String interpolation ──────────────────────────────────────────────────

    pub(crate) fn emit_interp(&self, segs: &[StringSegment]) -> String {
        let (fmt, args) = self.build_format_string(segs);
        let str_ty = match self.config.threading {
            crate::transpiler::ThreadingMode::Single => "Rc::<str>",
            crate::transpiler::ThreadingMode::Multi  => "Arc::<str>",
        };
        if args.is_empty() {
            format!("{}::from(\"{}\")", str_ty, fmt)
        } else {
            let fmt_call = format!("format!(\"{}\"{})", fmt,
                args.iter().map(|a| format!(", {}", a)).collect::<String>());
            format!("{}::from({}.as_str())", str_ty, fmt_call)
        }
    }

    /// Build (format_string, [arg_exprs]) from interpolation segments.
    pub(crate) fn build_format_string(&self, segs: &[StringSegment]) -> (String, Vec<String>) {
        let mut fmt = String::new();
        let mut args = Vec::new();
        for seg in segs {
            match seg {
                StringSegment::Lit(s) => {
                    for c in s.chars() {
                        match c {
                            '{' => fmt.push_str("{{"),
                            '}' => fmt.push_str("}}"),
                            '"' => fmt.push_str("\\\""),
                            '\\' => fmt.push_str("\\\\"),
                            '\n' => fmt.push_str("\\n"),
                            '\r' => fmt.push_str("\\r"),
                            '\t' => fmt.push_str("\\t"),
                            c   => fmt.push(c),
                        }
                    }
                }
                StringSegment::Expr(e) => {
                    let expr_s = self.emit_expr(e);
                    // Vec collections: wrap in BoringFmt for Display without debug quotes.
                    // HashMap/HashSet: keep {:?} (no Display impl).
                    let is_vec_var = matches!(&e.kind, ExprKind::Var(n) if self.vec_vars.contains(n.as_str()));
                    let is_col = looks_like_collection(&expr_s)
                        || matches!(&e.kind, ExprKind::Var(n) if self.collection_vars.contains(n))
                        || matches!(&e.kind, ExprKind::Array(_))
                        || self.expr_returns_collection(e);
                    let (expr_s, spec) = boring_vec_fmt(expr_s, is_col, is_vec_var);
                    fmt.push_str(spec);
                    args.push(expr_s);
                }
                StringSegment::FormattedExpr(e, spec) => {
                    let rust_spec = spec.trim_end_matches(['f', 'd', 's', 'g', 'G']);
                    fmt.push_str(&format!("{{:{}}}", rust_spec));
                    args.push(self.emit_expr(e));
                }
            }
        }
        (fmt, args)
    }

    /// Build the format string portion of a macro call (println!, format!, etc.),
    /// along with the emitted Rust expressions for each `{}` / `{:spec}` placeholder.
    /// Unlike build_format_string, literal `{}` segments are passed through as `{}`
    /// (Rust format placeholders) rather than being escaped to `{{}}`.
    pub(crate) fn build_macro_format_string(&self, segs: &[StringSegment]) -> (String, Vec<String>) {
        let mut fmt = String::new();
        let mut args = Vec::new();
        for seg in segs {
            match seg {
                StringSegment::Lit(s) => {
                    for c in s.chars() {
                        match c {
                            '"'  => fmt.push_str("\\\""),
                            '\\' => fmt.push_str("\\\\"),
                            '\n' => fmt.push_str("\\n"),
                            '\r' => fmt.push_str("\\r"),
                            '\t' => fmt.push_str("\\t"),
                            c    => fmt.push(c),
                        }
                    }
                }
                StringSegment::Expr(e) => {
                    // Boring interpolation inside a macro format string → keep as {}
                    fmt.push_str("{}");
                    args.push(self.emit_expr(e));
                }
                StringSegment::FormattedExpr(e, spec) => {
                    let rust_spec = spec.trim_end_matches(['f', 'd', 's', 'g', 'G']);
                    fmt.push_str(&format!("{{:{}}}", rust_spec));
                    args.push(self.emit_expr(e));
                }
            }
        }
        (fmt, args)
    }

    /// Build a Rust format string + combined arg list for positional print:
    ///   `print "x={}, y={}", a, b`  →  `("x={}, y={}", ["a", "b"])`
    ///
    /// Processes segments left-to-right:
    /// - `Lit("{}")` → `{}` in fmt, consumes next positional arg
    /// - `Expr(e)`   → `{}` in fmt, appends inline expr at its natural position
    /// - `Lit(s)`    → escapes `{`/`}` normally (not positional placeholders)
    pub(crate) fn build_positional_format(&self, segs: &[StringSegment], positional: &[String])
        -> (String, Vec<String>)
    {
        let mut fmt = String::new();
        let mut combined = Vec::new();
        let mut pos_idx = 0usize;
        for seg in segs {
            match seg {
                StringSegment::Lit(s) if s == "{}" => {
                    // Empty hole → positional placeholder
                    fmt.push_str("{}");
                    if pos_idx < positional.len() {
                        combined.push(positional[pos_idx].clone());
                        pos_idx += 1;
                    }
                }
                StringSegment::Lit(s) => {
                    for c in s.chars() {
                        match c {
                            '{' => fmt.push_str("{{"),
                            '}' => fmt.push_str("}}"),
                            '"'  => fmt.push_str("\\\""),
                            '\\' => fmt.push_str("\\\\"),
                            '\n' => fmt.push_str("\\n"),
                            '\r' => fmt.push_str("\\r"),
                            '\t' => fmt.push_str("\\t"),
                            c    => fmt.push(c),
                        }
                    }
                }
                StringSegment::Expr(e) => {
                    let expr_s = self.emit_expr(e);
                    let is_vec_var = matches!(&e.kind, ExprKind::Var(n) if self.vec_vars.contains(n.as_str()));
                    let is_col = looks_like_collection(&expr_s)
                        || matches!(&e.kind, ExprKind::Var(n) if self.collection_vars.contains(n))
                        || matches!(&e.kind, ExprKind::Array(_));
                    let (expr_s, spec) = boring_vec_fmt(expr_s, is_col, is_vec_var);
                    fmt.push_str(spec);
                    combined.push(expr_s);
                }
                StringSegment::FormattedExpr(e, spec) => {
                    let rust_spec = spec.trim_end_matches(['f', 'd', 's', 'g', 'G']);
                    fmt.push_str(&format!("{{:{}}}", rust_spec));
                    combined.push(self.emit_expr(e));
                }
            }
        }
        (fmt, combined)
    }

    // ── Receiver-type helpers ─────────────────────────────────────────────────

    /// Return true when `expr` produces a `HashMap` (dict) value.
    ///
    /// Covers four cases so that dict-specific methods work beyond simple
    /// local variables:
    ///   1. `Var(v)` that is tracked in `dict_vars`
    ///   2. A dict literal `{ k = v, … }`
    ///   3. A chained call whose result is still a dict:
    ///      `d.map(…)`, `d.filter(…)`, `d.set(…)`, `d.put(…)`, `d.remove(…)`
    ///   4. A function call whose declared return type is `Dict(…)` in
    ///      `fn_return_types`
    pub(crate) fn expr_is_dict(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Var(v) => {
                if self.dict_vars.contains(v.as_str()) { return true; }
                // Struct fields accessed as bare vars inside method bodies.
                self.self_type.as_ref()
                    .and_then(|sn| self.struct_fields.get(sn.as_str()))
                    .and_then(|fs| fs.iter().find(|(n, _)| n == v.as_str()))
                    .map(|(_, ty)| matches!(ty, crate::ast::Type::Dict(..)))
                    .unwrap_or(false)
            }
            ExprKind::Dict(_) => true,
            // Field access: look up the field type in struct_fields
            ExprKind::Field(obj, field_name) => {
                let struct_name = match &obj.kind {
                    ExprKind::Var(v) if v.as_str() == "self" => self.self_type.clone(),
                    ExprKind::Var(v) => self.var_struct_types.get(v.as_str())
                        .or_else(|| self.var_struct_type.get(v.as_str()))
                        .cloned(),
                    _ => None,
                };
                struct_name.and_then(|sn| self.struct_fields.get(sn.as_str()))
                    .and_then(|fs| fs.iter().find(|(n, _)| n == field_name))
                    .map(|(_, ty)| matches!(ty, crate::ast::Type::Dict(..)))
                    .unwrap_or(false)
            }
            // Chained dict → dict methods
            ExprKind::MethodCall(inner, m, _) => {
                const DICT_RETURNING: &[&str] = &["map", "filter", "set", "put", "remove"];
                DICT_RETURNING.contains(&m.as_str()) && self.expr_is_dict(inner)
            }
            // Function call whose return type is Dict(…)
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(fn_name) = &callee.kind {
                    matches!(self.fn_return_types.get(fn_name.as_str()), Some(Type::Dict(..)))
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Return true when `expr` yields an actor-qualified value (Rc<RefCell<T>> or Arc<Mutex<T>>).
    /// Used to detect if-let bindings that need to be tracked as managed vars.
    pub(crate) fn expr_yields_actor(&self, expr: &crate::ast::Expr) -> bool {
        use crate::ast::{ExprKind, Type, OwnerQual};
        match &expr.kind {
            ExprKind::Var(v) => {
                if let Some(ty) = self.var_types.get(v.as_str()) {
                    let ty = ty.without_mut();
                    return matches!(ty, Type::Qualified(_, OwnerQual::Actor))
                        || matches!(ty, Type::Optional(inner) if matches!(inner.without_mut(), Type::Qualified(_, OwnerQual::Actor)));
                }
                self.managed_refcell_vars.contains(v.as_str()) || self.managed_mutex_vars.contains(v.as_str())
            }
            ExprKind::Field(obj, field_name) => {
                let struct_name = match &obj.kind {
                    ExprKind::Var(v) if v.as_str() == "self" => self.self_type.clone(),
                    ExprKind::Var(v) => self.var_struct_types.get(v.as_str())
                        .or_else(|| self.var_struct_type.get(v.as_str()))
                        .cloned(),
                    _ => None,
                };
                struct_name.and_then(|sn| self.struct_fields.get(sn.as_str()))
                    .and_then(|fs| fs.iter().find(|(n, _)| n == field_name))
                    .map(|(_, ty)| {
                        let ty = ty.without_mut();
                        matches!(ty, Type::Qualified(_, OwnerQual::Actor))
                            || matches!(ty, Type::Optional(inner) if matches!(inner.without_mut(), Type::Qualified(_, OwnerQual::Actor)))
                    })
                    .unwrap_or(false)
            }
            _ => false,
        }
    }

    /// Return true when `expr` produces a `HashSet` value.
    ///
    /// Analogous to `expr_is_dict`:
    ///   1. `Var(v)` tracked in `set_vars`
    ///   2. A set literal `{ a, b, … }`
    ///   3. Chained set → set methods: `union`, `intersection`, `difference`,
    ///      `add`, `remove`
    ///   4. Function call returning `Set(…)`
    fn expr_is_set(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Var(v) => self.set_vars.contains(v.as_str()),
            ExprKind::Set(_) => true,
            // Field access: look up the field type in struct_fields
            ExprKind::Field(obj, field_name) => {
                let struct_name = match &obj.kind {
                    ExprKind::Var(v) if v.as_str() == "self" => self.self_type.clone(),
                    ExprKind::Var(v) => self.var_struct_types.get(v.as_str())
                        .or_else(|| self.var_struct_type.get(v.as_str()))
                        .cloned(),
                    _ => None,
                };
                struct_name.and_then(|sn| self.struct_fields.get(sn.as_str()))
                    .and_then(|fs| fs.iter().find(|(n, _)| n == field_name))
                    .map(|(_, ty)| matches!(ty, crate::ast::Type::Set(_)))
                    .unwrap_or(false)
            }
            ExprKind::MethodCall(inner, m, _) => {
                const SET_RETURNING: &[&str] = &["union", "intersection", "difference", "add", "remove"];
                SET_RETURNING.contains(&m.as_str()) && self.expr_is_set(inner)
            }
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(fn_name) = &callee.kind {
                    matches!(self.fn_return_types.get(fn_name.as_str()), Some(Type::Set(_)))
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    // ── Sub-transpiler ────────────────────────────────────────────────────────

    pub(crate) fn make_sub(&self) -> Transpiler {
        Transpiler {
            config: self.config.clone(),
            out: String::new(),
            errors: std::cell::RefCell::new(Vec::new()),
            warnings: std::cell::RefCell::new(Vec::new()),
            indent: self.indent,
            in_throws: self.in_throws,
            in_async: self.in_async,
            in_iter_stream: self.in_iter_stream,
            self_type: self.self_type.clone(),
            collection_vars: self.collection_vars.clone(),
            vec_vars: self.vec_vars.clone(),
            str_vec_vars: self.str_vec_vars.clone(),
            set_vars: self.set_vars.clone(),
            tuple_vars: self.tuple_vars.clone(),
            dict_vars: self.dict_vars.clone(),
            instant_vars: self.instant_vars.clone(),
            chars_vars: self.chars_vars.clone(),
            fn_return_types: self.fn_return_types.clone(),
            index_vars: self.index_vars.clone(),
            fn_sigs: self.fn_sigs.clone(),
            fn_rebindable: self.fn_rebindable.clone(),
            fn_mutable: self.fn_mutable.clone(),
            enum_variants: self.enum_variants.clone(),
            enum_variant_fields: self.enum_variant_fields.clone(),
            enum_variant_field_types: self.enum_variant_field_types.clone(),
            optional_vars: self.optional_vars.clone(),
            fn_defaults: self.fn_defaults.clone(),
            struct_fields: self.struct_fields.clone(),
            struct_field_defaults: self.struct_field_defaults.clone(),
            structs_needing_default: self.structs_needing_default.clone(),
            type_sizes: self.type_sizes.clone(),
            qualified_struct_types: self.qualified_struct_types.clone(),
            actor_source_types: self.actor_source_types.clone(),
            all_struct_types: self.all_struct_types.clone(),
            all_enum_types: self.all_enum_types.clone(),
            struct_assoc_types: self.struct_assoc_types.clone(),
            fn_throws: self.fn_throws.clone(),
            typed_error_enums: self.typed_error_enums.clone(),
            display_types: self.display_types.clone(),
            task_fns: self.task_fns.clone(),
            instance_task_methods: self.instance_task_methods.clone(),
            struct_task_methods: self.struct_task_methods.clone(),
            task_method_call_fields: std::collections::HashSet::new(),
            task_method_call_vars: std::collections::HashSet::new(),
            task_vars: self.task_vars.clone(),
            throws_fn_params: self.throws_fn_params.clone(),
            arc_vars: self.arc_vars.clone(),
            weak_vars: self.weak_vars.clone(),
            fn_variadic: self.fn_variadic.clone(),
            in_try_body: self.in_try_body,
            in_type_setter: self.in_type_setter,
            in_init_body: self.in_init_body,
            struct_type_var_names: self.struct_type_var_names.clone(),
            struct_type_mut_var_names: self.struct_type_mut_var_names.clone(),
            struct_type_method_sigs: self.struct_type_method_sigs.clone(),
            struct_getters: self.struct_getters.clone(),
            enum_field_getters: self.enum_field_getters.clone(),
            struct_setters: self.struct_setters.clone(),
            transient_fields: self.transient_fields.clone(),
            var_struct_types: self.var_struct_types.clone(),
            var_mutex_types: self.var_mutex_types.clone(),
            var_mutex_task_types: self.var_mutex_task_types.clone(),
            struct_mutex_fields: self.struct_mutex_fields.clone(),
            struct_mutex_task_fields: self.struct_mutex_task_fields.clone(),
            var_rwlock_types: self.var_rwlock_types.clone(),
            var_rwlock_task_types: self.var_rwlock_task_types.clone(),
            struct_rwlock_fields: self.struct_rwlock_fields.clone(),
            struct_rwlock_task_fields: self.struct_rwlock_task_fields.clone(),
            struct_req_methods: self.struct_req_methods.clone(),
            struct_protocols: self.struct_protocols.clone(),
            trait_default_mutating: self.trait_default_mutating.clone(),
            fn_var_params: self.fn_var_params.clone(),
            with_open_names: self.with_open_names.clone(),
            iterable_structs: self.iterable_structs.clone(),
            known_local_vars: self.known_local_vars.clone(),
            fn_returns_void: self.fn_returns_void,
            fn_declared_void: self.fn_declared_void,
            suppress_ok_wrap: false,
            trait_method_names: self.trait_method_names.clone(),
            user_conv_targets: self.user_conv_targets.clone(),
            string_arc_vars: self.string_arc_vars.clone(),
            impl_type_params: self.impl_type_params.clone(),
            fn_return_ty: self.fn_return_ty.clone(),
            newtype_types: self.newtype_types.clone(),
            newtype_inner: self.newtype_inner.clone(),
            var_newtype_type: self.var_newtype_type.clone(),
            stream_fns: self.stream_fns.clone(),
            stream_iter_fns: self.stream_iter_fns.clone(),
            stream_throws_fns: self.stream_throws_fns.clone(),
            has_streams: self.has_streams,
            channel_receivers: self.channel_receivers.clone(),
            string_channel_receivers: self.string_channel_receivers.clone(),
            channel_senders: self.channel_senders.clone(),
            oneshot_receivers: self.oneshot_receivers.clone(),
            oneshot_senders: self.oneshot_senders.clone(),
            broadcast_receivers: self.broadcast_receivers.clone(),
            broadcast_senders: self.broadcast_senders.clone(),
            watch_receivers: self.watch_receivers.clone(),
            watch_senders: self.watch_senders.clone(),
            join_handle_vars: self.join_handle_vars.clone(),
            throws_join_handle_vars: self.throws_join_handle_vars.clone(),
            trait_assoc_type_names: self.trait_assoc_type_names.clone(),
            inside_trait_impl: self.inside_trait_impl,
            match_subject_enum: self.match_subject_enum.clone(),
            var_types: self.var_types.clone(),
            string_vars: self.string_vars.clone(),
            str_index_cache_vars: self.str_index_cache_vars.clone(),
            struct_operator_methods: self.struct_operator_methods.clone(),
            struct_operator_param_types: self.struct_operator_param_types.clone(),
            var_struct_type: self.var_struct_type.clone(),
            fn_param_names: self.fn_param_names.clone(),
            user_defines_box: self.user_defines_box,
            user_defines_result: self.user_defines_result,
            struct_has_init_body: self.struct_has_init_body.clone(),
            struct_init_defaults: self.struct_init_defaults.clone(),
            global_var_types: self.global_var_types.clone(),
            global_var_inits: self.global_var_inits.clone(),
            global_vars_used_in_fns: self.global_vars_used_in_fns.clone(),
            global_lets_used_elsewhere: self.global_lets_used_elsewhere.clone(),
            optional_numeric_vars: self.optional_numeric_vars.clone(),
            always_none_vars: self.always_none_vars.clone(),
            fn_type_aliases: self.fn_type_aliases.clone(),
            non_fn_type_aliases: self.non_fn_type_aliases.clone(),
            current_fn_type_params: self.current_fn_type_params.clone(),
            struct_ext_method_overrides: self.struct_ext_method_overrides.clone(),
            rc_identity_vars: self.rc_identity_vars.clone(),
            boring_mod_names: self.boring_mod_names.clone(),
            // Share Rc so any set(true) in a sub is immediately visible in the parent —
            // fixes silent loss of feature flags when code using log/serde/etc. appears
            // inside try: blocks or other sub-transpiled contexts.
            uses_log: std::rc::Rc::clone(&self.uses_log),
            uses_thiserror: std::rc::Rc::clone(&self.uses_thiserror),
            uses_reqwest: self.uses_reqwest,
            current_trait_assoc_names: self.current_trait_assoc_names.clone(),
            cancellable_task_fns: self.cancellable_task_fns.clone(),
            cancel_token_vars: self.cancel_token_vars.clone(),
            in_cancellable_fn: self.in_cancellable_fn,
            uses_tokio_util: std::rc::Rc::clone(&self.uses_tokio_util),
            uses_serde: std::rc::Rc::clone(&self.uses_serde),
            fn_overload_decls: self.fn_overload_decls.clone(),
            overloaded_fn_names: self.overloaded_fn_names.clone(),
            struct_method_overload_decls: self.struct_method_overload_decls.clone(),
            overloaded_method_keys: self.overloaded_method_keys.clone(),
            arc_qualified_types: self.arc_qualified_types.clone(),
            unit_enums: self.unit_enums.clone(),
            recursive_fields: self.recursive_fields.clone(),
            user_types: self.user_types.clone(),
            uses_local_channel: std::rc::Rc::clone(&self.uses_local_channel),
            uses_local_broadcast: std::rc::Rc::clone(&self.uses_local_broadcast),
            rc_vars: self.rc_vars.clone(),
            shared_ref_params: self.shared_ref_params.clone(),
            var_primitive_params: self.var_primitive_params.clone(),
            managed_mutex_vars: self.managed_mutex_vars.clone(),
            managed_mutex_fn_return_vars: self.managed_mutex_fn_return_vars.clone(),
            in_lhs_assign: std::cell::Cell::new(false),
            managed_refcell_vars: self.managed_refcell_vars.clone(),
            managed_param_shadows: self.managed_param_shadows.clone(),
            struct_method_return_types: self.struct_method_return_types.clone(),
            struct_method_throws: self.struct_method_throws.clone(),
            inferred_qualifiers: self.inferred_qualifiers.clone(),
            infer_local_actor_vars: std::collections::HashSet::new(),
            source_dir: self.source_dir.clone(),
            loaded: self.loaded.clone(),
            prelude_emitted: self.prelude_emitted,
            emitted_fn_sigs: self.emitted_fn_sigs.clone(),
            fn_current_params: std::collections::HashMap::new(),
            fn_current_param_lines: std::collections::HashMap::new(),
            fn_current_param_cols: std::collections::HashMap::new(),
            fn_current_params_mut: std::collections::HashSet::new(),
            immutable_local_vars: self.immutable_local_vars.clone(),
            mut_local_vars: self.mut_local_vars.clone(),
            content_mutable_local_vars: self.content_mutable_local_vars.clone(),
            mut_checked_local_vars: self.mut_checked_local_vars.clone(),
            auto_ref_params: self.auto_ref_params.clone(),
            in_req_fn: self.in_req_fn,
            in_struct_method: self.in_struct_method,
            modules: Vec::new(),
            lazy_vars: self.lazy_vars.clone(),
            lazy_var_types: self.lazy_var_types.clone(),
            callable_structs: self.callable_structs.clone(),
            kernel_decls: self.kernel_decls.clone(),
            kernel_vars: self.kernel_vars.clone(),
            gpu_resident_vars: self.gpu_resident_vars.clone(),
            fn_returns_resident: self.fn_returns_resident.clone(),
            fn_gpu_arg_params: self.fn_gpu_arg_params.clone(),
            fn_returns_resident_tuple: self.fn_returns_resident_tuple.clone(),
            resident_call_vars: self.resident_call_vars.clone(),
            current_fn_returns_resident: self.current_fn_returns_resident.clone(),
            current_fn_returns_resident_tuple: self.current_fn_returns_resident_tuple.clone(),
            current_fn_gpu_arg_params: self.current_fn_gpu_arg_params.clone(),
            current_fn_gpu_arg_param_names: self.current_fn_gpu_arg_param_names.clone(),
            gpu_device_vars: self.gpu_device_vars.clone(),
            is_gpu_target: self.is_gpu_target,
            user_top_level_names: self.user_top_level_names.clone(),
            gpu_main_emitted: std::cell::Cell::new(false),
            gpu_top_level_const_names: self.gpu_top_level_const_names.clone(),
        }
    }

    // ── Inline statement emit (for closures) ──────────────────────────────────

    pub(crate) fn emit_stmt_inline(&self, stmt: &Stmt) -> String {
        match stmt {
            Stmt::Expr(e) => self.emit_expr(e),
            Stmt::Return(s) => match &s.value {
                Some(e) => format!("return {};", self.emit_expr(e)),
                None    => "return;".into(),
            },
            Stmt::Let(s) => {
                let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
                let ty = s.ty.as_ref().map(|t| format!(": {}", self.emit_type(t))).unwrap_or_default();
                format!("{} {}{} = {};", kw, s.name, ty, self.emit_expr(s.value.as_ref().expect("invariant: Let statement in expression context must have an initializer value")))
            }
            Stmt::If(s) => {
                // Emit if-expression inline using a sub-transpiler
                let mut sub = self.make_sub();
                sub.emit_if(s, false);
                sub.out.trim_end().to_string()
            }
            _ => "/* complex stmt */".into(),
        }
    }

    // ── Variable/builtin name mapping ─────────────────────────────────────────

    // ─── Built-in `fs` module ─────────────────────────────────────────────────
    //
    // `fs.read("path")` etc.  In async contexts emits tokio::fs; otherwise std::fs.
    // All fallible operations add `?` in throws/try_body, `.unwrap()` otherwise.

    pub(crate) fn emit_fs_call(&self, method: &str, args: &[Arg]) -> String {
        // Error-propagation suffix: ? in throws context, .unwrap() otherwise.
        let prop: &str = if self.in_throws || self.in_try_body { "?" } else { ".unwrap()" };

        // Await suffix: .await? or .await.unwrap() or nothing for sync.
        // We always emit tokio::fs in Boring — all programs run under tokio.
        let aw = if self.in_async {
            format!(".await{}", prop)
        } else {
            // Sync context: emit std::fs (blocking), still propagate errors.
            prop.to_string()
        };

        // Shorthand: emit the i-th arg value as an expression.
        let a = |i: usize| -> String {
            args.get(i).map(|a| self.emit_expr(&a.value)).unwrap_or_default()
        };

        // Path args go through functions bound on `impl AsRef<Path>` — `Rc<str>`/`Arc<str>`
        // (boring's `string`) only implements `AsRef<str>`, not `AsRef<Path>`, so a bare
        // string *variable* (unlike a `&'static str` literal) fails to compile. Deref-reborrow
        // to `&str` first, which does implement `AsRef<Path>` — harmless on literals too.
        let pth = |i: usize| -> String { format!("&*({})", a(i)) };

        // Which fs module to use (tokio for async, std for sync).
        let fs_mod = if self.in_async { "tokio::fs" } else { "std::fs" };

        match method {
            // fs.read("path") → Arc<str>
            "read" => {
                let path = pth(0);
                format!(
                    "{{ let __boring_s = {}::read_to_string({}){aw}; {}::<str>::from(__boring_s.as_str()) }}",
                    fs_mod, path, self.str_ptr(), aw = aw
                )
            }

            // fs.readLines("path") → Vec<Arc<str>>
            "readLines" => {
                let path = pth(0);
                format!(
                    "{{ let __boring_s = {}::read_to_string({}){aw}; __boring_s.lines().map(|l| {}::<str>::from(l)).collect::<Vec<{}<str>>>() }}",
                    fs_mod, path, self.str_ptr(), self.str_ptr(), aw = aw
                )
            }

            // fs.write("path", content)
            "write" => {
                let path    = pth(0);
                let content = a(1);
                // content is Arc<str>; Deref<Target=str> → .as_bytes() works.
                format!("{}::write({}, ({}).as_bytes()){aw}", fs_mod, path, content, aw = aw)
            }

            // fs.append("path", content)  — OpenOptions::append
            "append" => {
                let path    = pth(0);
                let content = a(1);
                if self.in_async {
                    format!(
                        "{{ use tokio::io::AsyncWriteExt as _; let mut __boring_f = tokio::fs::OpenOptions::new().append(true).create(true).open({}){aw}; __boring_f.write_all(({}).as_bytes()){aw} }}",
                        path, content, aw = aw
                    )
                } else {
                    format!(
                        "{{ use std::io::Write as _; std::fs::OpenOptions::new().append(true).create(true).open({}){prop}.write_all(({}).as_bytes()){prop} }}",
                        path, content, prop = prop
                    )
                }
            }

            // fs.exists("path") → bool  (never throws — uses is_ok())
            "exists" => {
                let path = pth(0);
                if self.in_async {
                    format!("tokio::fs::metadata({}).await.is_ok()", path)
                } else {
                    format!("std::path::Path::new({}).exists()", path)
                }
            }

            // fs.isDir("path") → bool
            "isDir" => {
                let path = pth(0);
                format!("std::path::Path::new({}).is_dir()", path)
            }

            // fs.isFile("path") → bool
            "isFile" => {
                let path = pth(0);
                format!("std::path::Path::new({}).is_file()", path)
            }

            // fs.mkdir("path")  — create_dir_all
            "mkdir" => {
                let path = pth(0);
                format!("{}::create_dir_all({}){aw}", fs_mod, path, aw = aw)
            }

            // fs.remove("path")  — remove file or directory tree
            "remove" => {
                let path = pth(0);
                // Smart remove: directory tree if it's a dir, single file otherwise.
                // Emit a block that tries remove_file, falls back to remove_dir_all.
                if self.in_async {
                    format!(
                        "{{ if std::path::Path::new({path}).is_dir() {{ tokio::fs::remove_dir_all({path}).await{prop} }} else {{ tokio::fs::remove_file({path}).await{prop} }} }}",
                        path = path, prop = prop
                    )
                } else {
                    format!(
                        "{{ if std::path::Path::new({path}).is_dir() {{ std::fs::remove_dir_all({path}){prop} }} else {{ std::fs::remove_file({path}){prop} }} }}",
                        path = path, prop = prop
                    )
                }
            }

            // fs.rename("old", "new") / fs.move(...)
            "rename" | "move" => {
                let from = pth(0);
                let to   = pth(1);
                format!("{}::rename({}, {}){aw}", fs_mod, from, to, aw = aw)
            }

            // fs.copy("src", "dst")
            "copy" => {
                let from = pth(0);
                let to   = pth(1);
                // std::fs::copy returns u64 (bytes); tokio::fs::copy too — discard it.
                format!("{{ let _ = {}::copy({}, {}){aw}; }}", fs_mod, from, to, aw = aw)
            }

            // fs.list("path") → Vec<Arc<str>> of entry names
            "list" => {
                let path = pth(0);
                let p = self.str_ptr();
                if self.in_async {
                    format!(
                        "{{ let mut __boring_dir = tokio::fs::read_dir({}){aw}; let mut __boring_entries: Vec<{p}<str>> = Vec::new(); while let Some(__boring_e) = __boring_dir.next_entry().await{prop} {{ __boring_entries.push({p}::<str>::from(__boring_e.file_name().to_string_lossy().as_ref())); }} __boring_entries }}",
                        path, aw = aw, prop = prop, p = p
                    )
                } else {
                    format!(
                        "{{ std::fs::read_dir({}){prop}.filter_map(|e| e.ok()).map(|e| {p}::<str>::from(e.file_name().to_string_lossy().as_ref())).collect::<Vec<{p}<str>>>() }}",
                        path, prop = prop, p = p
                    )
                }
            }

            // fs.readBytes("path") → Vec<u8>, matching boring's [uint8] directly —
            // no per-element widening needed since uint8 is a real 1-byte type.
            // `aw` already includes `.await` when async (see its definition above),
            // so no separate self.in_async branch is needed here.
            "readBytes" => {
                let path = pth(0);
                format!("{}::read({}){aw}", fs_mod, path, aw = aw)
            }

            // fs.writeBytes("path", bytes) — bytes is already Vec<u8> ([uint8]).
            "writeBytes" => {
                let path  = pth(0);
                let bytes = a(1);
                format!("{}::write({}, &{}){aw}", fs_mod, path, bytes, aw = aw)
            }

            // Fallback: emit as a generic path call
            other => {
                let vals: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
                format!("{}::{}({}){aw}", fs_mod, other, vals.join(", "), aw = aw)
            }
        }
    }

    pub(crate) fn map_builtin_var(&self, name: &str) -> String {
        match name {
            "PI"  => "PI".into(),
            "E"   => "E".into(),
            "TAU" => "TAU".into(),
            "INF" => "f64::INFINITY".into(),
            "NAN" => "f64::NAN".into(),
            "nil" => "None".into(),
            "self" => "self".into(),
            n => {
                // Global mutable var accessed anywhere (both functions and main):
                // emit as `NAME.lock().unwrap().clone()`.
                if self.global_vars_used_in_fns.contains(n) {
                    let static_name = n.to_uppercase();
                    return format!("{}.lock().unwrap_or_else(|e| e.into_inner()).clone()", static_name);
                }
                // GPU-target top-level scalar const, uppercased at emission (see
                // `emit_item`'s `Item::Let` case) to avoid colliding with a same-named fn
                // parameter elsewhere in the file. A genuine local of the same name in
                // THIS scope (e.g. a fn parameter actually called `width`) must still
                // shadow it, matching ordinary Rust scoping -- only rewrite when there
                // isn't one.
                if self.gpu_top_level_const_names.contains(n) && !self.known_local_vars.contains(n) {
                    return n.to_uppercase();
                }
                // Bare PascalCase enum variant not in known_local_vars — qualify it.
                // e.g. `Nil` (Value::Nil) or `Uninitialized` (Value::Uninitialized)
                if n.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && !self.known_local_vars.contains(n)
                    && !self.struct_fields.contains_key(n)
                {
                    if let Some(enum_type) = self.enum_variants.get(n) {
                        return format!("{}::{}", enum_type, n);
                    }
                }
                escape_rust_keyword(n)
            }
        }
    }
}


fn is_auto_ref_param(ty: Option<&Type>) -> bool {
    matches!(ty,
        Some(Type::Qualified(_, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard)) |
        Some(Type::Qualified(_, OwnerQual::Weak))
    )
}

/// Returns true if the param type is an unqualified Named type whose name is a known actor-source type.
/// These are emitted as `&Arc<Mutex<T>>` by qualifier inference, so call sites must add `&`.
fn is_unqualified_actor_source_param(ty: Option<&Type>, actor_source_types: &std::collections::HashSet<String>) -> bool {
    match ty {
        Some(Type::Named(n)) => actor_source_types.contains(n.as_str()),
        _ => false,
    }
}

/// Returns true when param_ty is a primitive Copy type that never needs .clone().
fn param_ty_is_copy(ty: Option<&Type>) -> bool {
    matches!(ty,
        Some(Type::Int | Type::Uint | Type::Uint8 | Type::Float32 | Type::Float64 | Type::Bool
            | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
            | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128) |
        Some(Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut))
    )
}
