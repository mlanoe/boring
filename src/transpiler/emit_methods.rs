use super::*;
use crate::ast::*;
use super::Transpiler;
use super::helpers::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

impl Transpiler {
    pub(crate) fn emit_method_call(&self, obj: &Expr, method: &str, args: &[Arg]) -> String {
        // Built-in `fs` module namespace — intercept before any other dispatch.
        if let ExprKind::Var(v) = &obj.kind {
            if v == "fs" {
                return self.emit_fs_call(method, args);
            }
        }

        // Task.cancelled() → __task_cancel.cancelled() (inside a cancellable task def fn)
        if method == "cancelled" && args.is_empty() {
            if let ExprKind::Var(v) = &obj.kind {
                if v == "Task" {
                    self.uses_tokio_util.set(true);
                    return "__task_cancel.cancelled()".to_string();
                }
            }
        }
        // v.cancel() where v is a join_handle_var → use the cancel token
        if method == "cancel" && args.is_empty() {
            if let ExprKind::Var(v) = &obj.kind {
                if self.join_handle_vars.contains(v.as_str()) {
                    self.uses_tokio_util.set(true);
                    if let Some(cancel_var) = self.cancel_token_vars.get(v.as_str()) {
                        return format!("{}.cancel()", cancel_var);
                    }
                    // Fallback: abort the JoinHandle
                    return format!("{}.abort()", v);
                }
            }
        }
        // Channel sender: `tx.send(value)` → `tx.send(value).await.unwrap()`
        // Use emit_expr_owned so string literals are wrapped in Arc::from(...) to match the
        // channel's Arc<str> item type.
        if method == "send" {
            if let ExprKind::Var(var_name) = &obj.kind {
                if self.channel_senders.contains(var_name.as_str()) {
                    let val = args.first().map(|a| self.emit_expr_owned(&a.value)).unwrap_or_default();
                    return format!("{}.send({}).await.unwrap()", var_name, val);
                }
                // oneshot/broadcast/watch senders: non-async, swallow error with .ok()
                if self.oneshot_senders.contains(var_name.as_str())
                    || self.broadcast_senders.contains(var_name.as_str())
                    || self.watch_senders.contains(var_name.as_str())
                {
                    let val = args.first().map(|a| self.emit_expr_owned(&a.value)).unwrap_or_default();
                    return format!("{}.send({}).ok()", var_name, val);
                }
            }
        }
        // broadcast tx.subscribe() → tx.subscribe()
        if method == "subscribe" {
            if let ExprKind::Var(var_name) = &obj.kind {
                if self.broadcast_senders.contains(var_name.as_str()) {
                    return format!("{}.subscribe()", var_name);
                }
            }
        }
        // oneshot receiver: rx.recv() → rx.await.unwrap() (consumed once)
        if method == "recv" {
            if let ExprKind::Var(var_name) = &obj.kind {
                if self.oneshot_receivers.contains(var_name.as_str()) {
                    return if self.in_throws || self.in_try_body {
                        format!("{}.await?", var_name)
                    } else {
                        format!("{}.await.unwrap()", var_name)
                    };
                }
                // broadcast receiver: rx.recv() → rx.recv().await.unwrap()
                if self.broadcast_receivers.contains(var_name.as_str()) {
                    return format!("{}.recv().await.unwrap()", var_name);
                }
                // watch receiver: rx.recv() → { rx.changed().await.ok(); rx.borrow().clone() }
                if self.watch_receivers.contains(var_name.as_str()) {
                    return format!("{{ {}.changed().await.ok(); {}.borrow().clone() }}", var_name, var_name);
                }
            }
        }
        // Channel sender clone: `tx.clone()` is pass-through — Rust handles it natively.

        // Detect `TypeName.method(args)` — type method or enum variant call.
        if let ExprKind::Var(type_name) = &obj.kind {
            let is_type = type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
            if is_type {
                let is_variant = method.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
                if is_variant {
                    let key = format!("{}::{}", type_name, method);
                    if self.enum_variant_fields.contains_key(&key) {
                        // Enum variant: `EnumType.Variant(args)` → `EnumType::Variant(...)`
                        // Use emit_let_value so string args are coerced to Arc<str>.
                        let field_tys = self.enum_variant_field_types.get(&key).cloned().unwrap_or_default();
                        let vals: Vec<String> = args.iter().enumerate().map(|(i, a)| {
                            let ty = field_tys.get(i);
                            self.emit_let_value(ty, &a.value)
                        }).collect();
                        // When the user defines their own Result<T,E> enum, Rust can't infer
                        // both type params from one variant's args. Add turbofish to disambiguate.
                        let type_turbofish = if self.user_defines_result && type_name == "Result" {
                            if method == "Ok" { "::<_, ()>" } else { "::<(), _>" }
                        } else { "" };
                        return format!("{}{}::{}({})", type_name, type_turbofish, method, vals.join(", "));
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
                    return format!("{}::{}({})", type_name, rust_name, vals.join(", "));
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
                    return if self.in_throws || self.in_try_body {
                        format!("{}?", awaited)
                    } else {
                        awaited
                    };
                }
                return call;
            }
        }
        // RwLock local var method: c.method(args) → c.read/write().await.method(args)
        // req methods use .read().await, def (mutating) methods use .write().await.
        if let ExprKind::Var(v) = &obj.kind {
            if self.var_rwlock_types.contains(v.as_str()) {
                let (rust_method, extra_wrap) = map_method(method, args.len());
                let args_s: Vec<String> = args.iter().map(|a| self.emit_expr_owned(&a.value)).collect();
                let struct_name = self.var_struct_types.get(v.as_str()).cloned().unwrap_or_default();
                let req_key = format!("{}::{}", struct_name, method);
                let lock = if self.struct_req_methods.contains(&req_key) { "read" } else { "write" };
                let call = format!("{}.{}().await.{}({})", v, lock, rust_method, args_s.join(", "));
                let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                const TOKIO_ASYNC_INSTANCE: &[&str] = &["recv", "send", "write_all", "read_line", "acquire", "flush"];
                let needs_await = self.instance_task_methods.contains(method)
                    || TOKIO_ASYNC_INSTANCE.iter().any(|&m| m == method);
                return if self.in_async && needs_await {
                    format!("{}.await", call)
                } else {
                    call
                };
            }
        }
        // RwLock struct field method: self.data.method() → self.data.read/write().await.method()
        if let ExprKind::Field(inner_obj, rwlock_field) = &obj.kind {
            if let ExprKind::Var(v) = &inner_obj.kind {
                if v == "self" {
                    let key = self.self_type.as_deref()
                        .map(|t| format!("{}::{}", t, rwlock_field));
                    if let Some(k) = key {
                        if self.struct_rwlock_fields.contains(&k) {
                            let (rust_method, extra_wrap) = map_method(method, args.len());
                            let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
                            // Determine lock kind from struct_req_methods
                            let struct_type_name = self.self_type.as_deref().unwrap_or("");
                            let req_key = format!("{}::{}", struct_type_name, method);
                            let lock = if self.struct_req_methods.contains(&req_key) { "read" } else { "write" };
                            let call = format!("self.{}.{}().await.{}({})", rwlock_field, lock, rust_method, args_s.join(", "));
                            let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                            return if self.in_async && self.instance_task_methods.contains(method) {
                                format!("{}.await", call)
                            } else {
                                call
                            };
                        }
                    }
                }
            }
        }
        // Mutex local var method: w.method(args) → w.lock().await.method(args)
        if let ExprKind::Var(v) = &obj.kind {
            if self.var_mutex_types.contains(v.as_str()) {
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
                let call = format!("{}.lock().await.{}({})", v, rust_method, args_s.join(", "));
                let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                // Add .await for user task methods AND known tokio async instance methods (recv, etc.)
                const TOKIO_ASYNC_INSTANCE: &[&str] = &["recv", "send", "write_all", "read_line", "acquire", "flush"];
                let needs_await = self.instance_task_methods.contains(method)
                    || TOKIO_ASYNC_INSTANCE.iter().any(|&m| m == method);
                return if self.in_async && needs_await {
                    format!("{}.await", call)
                } else {
                    call
                };
            }
        }
        // Mutex struct field method: self.worker.method(args) → self.worker.lock().await.method(args)
        if let ExprKind::Field(inner_obj, mutex_field) = &obj.kind {
            if let ExprKind::Var(v) = &inner_obj.kind {
                if v == "self" {
                    let key = self.self_type.as_deref()
                        .map(|t| format!("{}::{}", t, mutex_field));
                    if let Some(k) = key {
                        if self.struct_mutex_fields.contains(&k) {
                            let (rust_method, extra_wrap) = map_method(method, args.len());
                            let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
                            let call = format!("self.{}.lock().await.{}({})", mutex_field, rust_method, args_s.join(", "));
                            let call = if let Some(wrap) = extra_wrap { format!("{}{}", call, wrap) } else { call };
                            return if self.in_async && self.instance_task_methods.contains(method) {
                                format!("{}.await", call)
                            } else {
                                call
                            };
                        }
                    }
                }
            }
        }
        // Special case: read_line(string_arc_var) — Arc<str> is immutable so we use a
        // temporary String buffer then assign back.
        // tokio BufReader (local var) → async (.await.unwrap_or(0))
        // std::io::stdin() (call result) → sync (.unwrap_or(0), no await)
        if method == "read_line" {
            if let Some(arg) = args.first() {
                if let ExprKind::Var(v) = &arg.value.kind {
                    if self.string_arc_vars.contains(v.as_str()) {
                        let obj_s = self.emit_expr(obj);
                        return if matches!(&obj.kind, ExprKind::Var(_)) {
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
                        };
                    }
                }
            }
        }
        // Special case: string_arc_var.clear() — reassign to empty Arc<str>
        if method == "clear" {
            if let ExprKind::Var(v) = &obj.kind {
                if self.string_arc_vars.contains(v.as_str()) {
                    return format!("{} = Arc::from(\"\")", v);
                }
            }
        }
        // Array filter: Rust's Iterator::filter passes &Item to the predicate, but Boring
        // closures expect owned values (e.g. `n: n > 0`).  Wrap the closure param with a
        // `.clone()` rebind so `n` inside the body is an owned T, not a &T.
        // Only applies to array filter — not Option::filter (handled below) and not
        // dict::filter (handled in the recv_is_dict block below with a 2-param closure).
        if method == "filter" && !is_option_expr(obj) && !self.expr_is_dict(obj) {
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
                        return format!("{}.iter().cloned().filter({}).collect::<Vec<_>>()", obj_s, closure_s);
                    }
                }
            }
        }
        // ── New collection / string / dict / set methods ─────────────────────────

        // Helper: emit a closure arg with an owned rebind for its first parameter.
        // This is the same pattern used by filter above: the iterator passes a &T but
        // Boring closures expect an owned value, so we insert `let pname = pname.clone();`.
        let emit_owned_closure = |me: &Transpiler, arg: &Arg, extra_params: usize| -> Option<String> {
            if let ExprKind::Closure(params, _, body, _, _) = &arg.value.kind {
                if let Some(param) = params.first() {
                    let pname = &param.name;
                    let mut sub = me.make_sub();
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
        };

        // ── Detect string receiver ────────────────────────────────────────────
        let receiver_is_string = match &obj.kind {
            ExprKind::Var(v) => self.string_arc_vars.contains(v.as_str())
                             || self.string_vars.contains(v.as_str()),
            ExprKind::Str(_) | ExprKind::StringInterp(_) => true,
            _ => false,
        };

        // ── Detect set/dict receivers (used further down too) ─────────────────
        let recv_is_set = self.expr_is_set(obj);
        let recv_is_dict = self.expr_is_dict(obj);

        // ── String-specific methods ───────────────────────────────────────────
        if receiver_is_string {
            let obj_s = self.emit_expr(obj);
            match method {
                "parseInt" => {
                    return format!("{}.trim().parse::<i64>().ok()", obj_s);
                }
                "parseFloat" => {
                    return format!("{}.trim().parse::<f64>().ok()", obj_s);
                }
                "indexOf" => {
                    let sub_s = args.first().map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "\"\"".to_string());
                    // Use &* to coerce Arc<str>/Arc<str> argument to &str for str::find.
                    return format!("{}.find(&*{}).map(|i| i as i64)", obj_s, sub_s);
                }
                "slice" if args.len() >= 2 => {
                    let start_s = self.emit_expr(&args[0].value);
                    let end_s   = self.emit_expr(&args[1].value);
                    // Bind start to a local so it is not double-evaluated (fix #23).
                    // Wrap in Arc<str> so the result type is `string`, not `String`.
                    return format!(
                        "{{ let __start = ({}) as usize; Arc::<str>::from({}.chars().skip(__start).take(({}) as usize - __start).collect::<String>().as_str()) }}",
                        start_s, obj_s, end_s
                    );
                }
                _ => {}
            }
        }

        // ── Dict-specific methods ─────────────────────────────────────────────
        if recv_is_dict {
            let obj_s = self.emit_expr(obj);
            match method {
                "keys" => {
                    return format!("{}.keys().cloned().collect::<Vec<_>>()", obj_s);
                }
                "values" => {
                    return format!("{}.values().cloned().collect::<Vec<_>>()", obj_s);
                }
                "get" if args.len() >= 2 => {
                    let key_s = self.emit_dict_key_borrow(&args[0].value);
                    let def_s = self.emit_expr(&args[1].value);
                    return format!("{}.get({}).cloned().unwrap_or({})", obj_s, key_s, def_s);
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
                            return format!(
                                "{}.iter().map(|({}, {})| {{ let {} = {}.clone(); let {} = {}.clone(); ({}.clone(), {}) }}).collect::<HashMap<_,_>>()",
                                obj_s, kname, vname, kname, kname, vname, vname, kname, body_s
                            );
                        }
                    }
                }
                "filter" => {
                    if let Some(first_arg) = args.first() {
                        if let Some(closure_s) = emit_owned_closure(self, first_arg, 1) {
                            return format!(
                                "{}.clone().into_iter().filter({}).collect::<HashMap<_,_>>()",
                                obj_s, closure_s
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        // ── Set-specific methods ──────────────────────────────────────────────
        if recv_is_set {
            let obj_s = self.emit_expr(obj);
            match method {
                "toArray" => {
                    return format!("{}.iter().cloned().collect::<Vec<_>>()", obj_s);
                }
                "union" => {
                    let other_s = args.first().map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "Default::default()".to_string());
                    return format!("{}.union(&{}).cloned().collect::<HashSet<_>>()", obj_s, other_s);
                }
                "intersection" => {
                    let other_s = args.first().map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "Default::default()".to_string());
                    return format!("{}.intersection(&{}).cloned().collect::<HashSet<_>>()", obj_s, other_s);
                }
                "difference" => {
                    let other_s = args.first().map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "Default::default()".to_string());
                    return format!("{}.difference(&{}).cloned().collect::<HashSet<_>>()", obj_s, other_s);
                }
                _ => {}
            }
        }

        // ── Array methods that need special emit (closures / block exprs) ─────
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
                        line: obj.line,
                    };
                    return self.emit_expr(&field_expr);
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
                .unwrap_or_else(|| "Arc::<str>::from(\"\")".to_string());
            return format!("{}.iter().map(|__s| __s.as_ref()).collect::<Vec<&str>>().join(&*{})", obj_s, sep_s);
        }

        // `any(closure)` → iter().cloned().any(|x| { let x = x.clone(); body })
        if method == "any" {
            if let Some(first_arg) = args.first() {
                if let Some(closure_s) = emit_owned_closure(self, first_arg, 0) {
                    let obj_s = self.emit_expr(obj);
                    return format!("{}.iter().cloned().any({})", obj_s, closure_s);
                }
            }
        }
        // `all(closure)` → iter().cloned().all(...)
        if method == "all" {
            if let Some(first_arg) = args.first() {
                if let Some(closure_s) = emit_owned_closure(self, first_arg, 0) {
                    let obj_s = self.emit_expr(obj);
                    return format!("{}.iter().cloned().all({})", obj_s, closure_s);
                }
            }
        }
        // `flatMap(closure)` → iter().cloned().flat_map(...).collect::<Vec<_>>()
        if method == "flatMap" {
            if let Some(first_arg) = args.first() {
                if let Some(closure_s) = emit_owned_closure(self, first_arg, 0) {
                    let obj_s = self.emit_expr(obj);
                    return format!("{}.iter().cloned().flat_map({}).collect::<Vec<_>>()", obj_s, closure_s);
                }
            }
        }
        // `count(closure?)` → with closure: filter+count; without: len() as i64
        if method == "count" {
            if let Some(first_arg) = args.first() {
                if let Some(closure_s) = emit_owned_closure(self, first_arg, 0) {
                    let obj_s = self.emit_expr(obj);
                    return format!("{}.iter().cloned().filter({}).count() as i64", obj_s, closure_s);
                }
            }
            // No closure: fallthrough to map_method which maps "count" → "len" as i64.
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
                        return format!(
                            "{{ let mut __boring_v = {}.clone(); __boring_v.sort_by_key(|{}| {{ let {} = {}.clone(); {} }}); __boring_v.iter().cloned().collect::<Vec<_>>() }}",
                            obj_s, pname, pname, pname, body_s
                        );
                    }
                }
            }
        }
        // `sorted()` → { let mut __v = obj.clone(); __v.sort_by(...); __v.iter().cloned().collect::<Vec<_>>() }
        // The collect at the end makes looks_like_collection() return true so print wraps with BoringFmt.
        if method == "sorted" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return format!(
                "{{ let mut __boring_v = {}.clone(); __boring_v.sort_by(|__a, __b| __a.partial_cmp(__b).unwrap_or(std::cmp::Ordering::Equal)); __boring_v.iter().cloned().collect::<Vec<_>>() }}",
                obj_s
            );
        }
        // `flat()` → into_iter().flatten().collect::<Vec<_>>()
        if method == "flat" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return format!("{}.into_iter().flatten().collect::<Vec<_>>()", obj_s);
        }
        // `zip(other)` → iter().cloned().zip(other.iter().cloned()).map(|(a,b)| (a,b)).collect::<Vec<_>>()
        if method == "zip" {
            if let Some(other_arg) = args.first() {
                let obj_s   = self.emit_expr(obj);
                let other_s = self.emit_expr(&other_arg.value);
                return format!(
                    "{}.iter().cloned().zip({}.iter().cloned()).map(|(a,b)| (a,b)).collect::<Vec<_>>()",
                    obj_s, other_s
                );
            }
        }
        // `enumerate()` → iter().cloned().enumerate().map(|(i,x)| (i as i64,x)).collect::<Vec<_>>()
        if method == "enumerate" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return format!(
                "{}.iter().cloned().enumerate().map(|(i,x)| (i as i64,x)).collect::<Vec<_>>()",
                obj_s
            );
        }
        // `slice(start, end)` on arrays → [start..end].iter().cloned().collect::<Vec<_>>()
        // Using .iter().cloned().collect instead of .to_vec() so looks_like_collection() detects it.
        if method == "slice" && args.len() >= 2 && !receiver_is_string {
            let obj_s   = self.emit_expr(obj);
            let start_s = self.emit_expr(&args[0].value);
            let end_s   = self.emit_expr(&args[1].value);
            return format!("{}[({}) as usize..({}) as usize].iter().cloned().collect::<Vec<_>>()", obj_s, start_s, end_s);
        }
        // `take(n)` → iter().cloned().take(n as usize).collect::<Vec<_>>()
        if method == "take" {
            if let Some(n_arg) = args.first() {
                let obj_s = self.emit_expr(obj);
                let n_s   = self.emit_expr(&n_arg.value);
                return format!(
                    "{}.iter().cloned().take(({}) as usize).collect::<Vec<_>>()",
                    obj_s, n_s
                );
            }
        }
        // `drop(n)` → iter().cloned().skip(n as usize).collect::<Vec<_>>()
        if method == "drop" {
            if let Some(n_arg) = args.first() {
                let obj_s = self.emit_expr(obj);
                let n_s   = self.emit_expr(&n_arg.value);
                return format!(
                    "{}.iter().cloned().skip(({}) as usize).collect::<Vec<_>>()",
                    obj_s, n_s
                );
            }
        }
        // `min()` → iter().cloned().min_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or_default()
        if method == "min" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return format!(
                "{}.iter().cloned().min_by(|__a,__b| __a.partial_cmp(__b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or_default()",
                obj_s
            );
        }
        // `max()` → iter().cloned().max_by(...)
        if method == "max" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return format!(
                "{}.iter().cloned().max_by(|__a,__b| __a.partial_cmp(__b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or_default()",
                obj_s
            );
        }
        // `sum()` → iter().cloned().sum::<i64>()
        if method == "sum" && args.is_empty() {
            let obj_s = self.emit_expr(obj);
            return format!("{}.iter().cloned().sum::<i64>()", obj_s);
        }
        // `indexOf(val)` on arrays → iter().position(|x| *x == val).map(|i| i as i64)
        // Note: the map_method fallback maps indexOf → iter().position which returns Option<usize>;
        // we need Option<i64> to match interpreter semantics.
        if method == "indexOf" && !receiver_is_string {
            if let Some(val_arg) = args.first() {
                let obj_s = self.emit_expr(obj);
                let val_s = self.emit_expr(&val_arg.value);
                return format!(
                    "{}.iter().position(|__x| __x == &({})).map(|i| i as i64)",
                    obj_s, val_s
                );
            }
        }

        // Option-chain short-circuit: if the receiver is Option-like and the method is one
        // that operates on Option (map, filter, or_else, etc.), emit it directly without the
        // iter/collect wrapping that map_method would add for Vec receivers.
        const OPTION_CHAIN_METHODS: &[&str] = &["map", "filter", "or", "or_else", "flatten", "and_then"];
        if OPTION_CHAIN_METHODS.contains(&method) && is_option_expr(obj) {
            let obj_s = self.emit_expr(obj);
            let closure_args: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
            let call = format!("{}.{}({})", obj_s, method, closure_args.join(", "));
            return if self.in_async && self.instance_task_methods.contains(method) {
                format!("{}.await", call)
            } else {
                call
            };
        }
        let obj_s = self.emit_expr(obj);
        // Use `::` for module/type path receivers (not instance variable method calls).
        // Lowercase non-local vars: `mpsc.channel(32)` → `mpsc::channel(32)`
        // Cascaded paths: `tokio::time.sleep()` → `tokio::time::sleep()`,
        // but NOT call results: `Path::new(x).exists()` uses `.` (result is a value).
        let is_path_receiver = match &obj.kind {
            ExprKind::Var(v) => {
                if v == "self" { false }
                else if v.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) { true }
                else { !self.known_local_vars.contains(v.as_str()) }
            }
            _ => obj_s.contains("::") && !obj_s.ends_with(')'),
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
        // If the receiver is a known user struct, preserve method names (don't remap to Rust builtins).
        let is_user_struct_receiver = match &obj.kind {
            ExprKind::Var(v) => self.var_struct_types.get(v.as_str())
                .map(|t| self.struct_fields.contains_key(t.as_str()))
                .unwrap_or(false),
            _ => false,
        };
        // Map boring method names → Rust equivalents
        // For HashSet vars, `add` maps to `insert` (not Vec::push).
        // For dict vars, `contains` maps to `contains_key` (HashMap has no `contains`).
        let receiver_is_set_var = matches!(&obj.kind, ExprKind::Var(v)
            if self.set_vars.contains(v.as_str()));
        let receiver_is_dict_var = matches!(&obj.kind, ExprKind::Var(v)
            if self.dict_vars.contains(v.as_str()));
        let (rust_method, extra_wrap) = if is_user_struct_receiver {
            // Check for overloaded struct methods — pick the best-matching overload.
            let struct_type_opt = match &obj.kind {
                ExprKind::Var(v) => self.var_struct_types.get(v.as_str()).cloned(),
                _ => None,
            };
            let overloaded_name = struct_type_opt.and_then(|type_name| {
                let key = format!("{}::{}", type_name, method);
                if !self.overloaded_method_keys.contains(&key) { return None; }
                let overloads = self.struct_method_overload_decls.get(&key)?;
                // Find best matching overload by arity and inferred arg types
                let chosen = overloads.iter().find(|decl| {
                    let min_a = decl.params.iter().filter(|p| p.default.is_none()).count();
                    let max_a = decl.params.len();
                    if args.len() < min_a || args.len() > max_a { return false; }
                    decl.params.iter().zip(args.iter()).all(|(p, a)| {
                        match &p.ty {
                            None => true,
                            Some(expected) => {
                                match infer_overload_expr_type(&a.value, &self.var_types, &self.fn_return_types) {
                                    Some(actual) => types_compatible(expected, &actual),
                                    None => true,
                                }
                            }
                        }
                    })
                }).or_else(|| overloads.first());
                chosen.map(|decl| mangle_overload_name(method, &decl.params))
            });
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
        let is_tokio_async = TOKIO_ASYNC_METHODS.iter().any(|&m| m == method);
        if self.in_async && (self.instance_task_methods.contains(method) || is_tokio_async) {
            format!("{}.await", call)
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
    pub(crate) fn emit_with_type_hint(&self, expr: &Expr, hint: Option<&Type>) -> String {
        if let Some(ty) = hint {
            // Resolve the named type (including aliases)
            let type_name = match ty {
                Type::Named(n) => Some(normalize_type_name(n)),
                _ => None,
            };
            if let Some(rust_type) = type_name {
                // .method(args) → TypeName::method(args)
                if let ExprKind::Call(callee, call_args) = &expr.kind {
                    if let ExprKind::DotIdent(method) = &callee.kind {
                        let rust_method = camel_to_snake(method);
                        let vals: Vec<String> = call_args.iter()
                            .map(|a| self.emit_expr(&a.value))
                            .collect();
                        return format!("{}::{}({})", rust_type, rust_method, vals.join(", "));
                    }
                }
                // .Variant → TypeName::Variant
                if let ExprKind::DotIdent(variant) = &expr.kind {
                    return format!("{}::{}", rust_type, variant);
                }
            }
        }
        self.emit_expr(expr)
    }

    pub(crate) fn emit_args(&self, args: &[Arg]) -> String {
        args.iter().map(|a| self.emit_expr_owned(&a.value)).collect::<Vec<_>>().join(", ")
    }

    /// Like emit_args but coerces non-nil values to Some(v) when the param is Optional,
    /// handles string-type params via emit_expr_owned, and fills missing args with defaults.
    /// Also reorders labeled (named) arguments to match the declared parameter order.
    pub(crate) fn emit_args_coerced(&self, fn_name: &str, args: &[Arg]) -> String {
        let sig = self.fn_sigs.get(fn_name).cloned().unwrap_or_default();
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
                let param_ty = sig.get(i);
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
                result.push(coerced.unwrap_or_else(|| self.emit_let_value(param_ty, &a.value)));
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
                        ExprKind::Str(s) => Some(escape_str_macro(s)),
                        ExprKind::StringInterp(segs) => {
                            // Boring interpolation: rebuild as Rust format string.
                            // Literal `{}` segments pass through as `{}` (placeholder).
                            Some(self.build_macro_format_string(segs))
                        }
                        _ => None,
                    };
                    if let Some(fmt) = fmt_str {
                        return if rest.is_empty() {
                            format!("{}!(\"{}\")", name, fmt)
                        } else {
                            format!("{}!(\"{}\", {})", name, fmt, rest.join(", "))
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
        if args.is_empty() {
            format!("\"{}\"", fmt)
        } else {
            format!("format!(\"{}\"{})", fmt,
                args.iter().map(|a| format!(", {}", a)).collect::<String>())
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
                    fmt.push_str(&format!("{{:{}}}", spec));
                    args.push(self.emit_expr(e));
                }
            }
        }
        (fmt, args)
    }

    /// Build the format string portion of a macro call (println!, format!, etc.).
    /// Unlike build_format_string, literal `{}` segments are passed through as `{}`
    /// (Rust format placeholders) rather than being escaped to `{{}}`.
    pub(crate) fn build_macro_format_string(&self, segs: &[StringSegment]) -> String {
        let mut fmt = String::new();
        for seg in segs {
            match seg {
                StringSegment::Lit(s) => {
                    for c in s.chars() {
                        match c {
                            '"'  => fmt.push_str("\\\""),
                            '\\' => fmt.push_str("\\\\"),
                            '\n' => fmt.push_str("\\n"),
                            '\t' => fmt.push_str("\\t"),
                            c    => fmt.push(c),
                        }
                    }
                }
                StringSegment::Expr(_) => {
                    // Boring interpolation inside a macro format string → keep as {}
                    fmt.push_str("{}");
                }
                StringSegment::FormattedExpr(_, spec) => {
                    fmt.push_str(&format!("{{:{}}}", spec));
                }
            }
        }
        fmt
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
                    fmt.push_str(&format!("{{:{}}}", spec));
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
    fn expr_is_dict(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Var(v) => self.dict_vars.contains(v.as_str()),
            ExprKind::Dict(_) => true,
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
            out: String::new(),
            indent: self.indent,
            in_throws: self.in_throws,
            in_async: self.in_async,
            self_type: self.self_type.clone(),
            collection_vars: self.collection_vars.clone(),
            vec_vars: self.vec_vars.clone(),
            set_vars: self.set_vars.clone(),
            dict_vars: self.dict_vars.clone(),
            instant_vars: self.instant_vars.clone(),
            chars_vars: self.chars_vars.clone(),
            fn_return_types: self.fn_return_types.clone(),
            index_vars: self.index_vars.clone(),
            fn_sigs: self.fn_sigs.clone(),
            enum_variants: self.enum_variants.clone(),
            enum_variant_fields: self.enum_variant_fields.clone(),
            enum_variant_field_types: self.enum_variant_field_types.clone(),
            optional_vars: self.optional_vars.clone(),
            fn_defaults: self.fn_defaults.clone(),
            struct_fields: self.struct_fields.clone(),
            struct_assoc_types: self.struct_assoc_types.clone(),
            fn_throws: self.fn_throws.clone(),
            typed_error_enums: self.typed_error_enums.clone(),
            display_types: self.display_types.clone(),
            task_fns: self.task_fns.clone(),
            instance_task_methods: self.instance_task_methods.clone(),
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
            struct_setters: self.struct_setters.clone(),
            transient_fields: self.transient_fields.clone(),
            var_struct_types: self.var_struct_types.clone(),
            var_mutex_types: self.var_mutex_types.clone(),
            struct_mutex_fields: self.struct_mutex_fields.clone(),
            var_rwlock_types: self.var_rwlock_types.clone(),
            struct_rwlock_fields: self.struct_rwlock_fields.clone(),
            struct_req_methods: self.struct_req_methods.clone(),
            iterable_structs: self.iterable_structs.clone(),
            known_local_vars: self.known_local_vars.clone(),
            fn_returns_void: self.fn_returns_void,
            trait_method_names: self.trait_method_names.clone(),
            user_conv_targets: self.user_conv_targets.clone(),
            string_arc_vars: self.string_arc_vars.clone(),
            impl_type_params: self.impl_type_params.clone(),
            fn_return_ty: self.fn_return_ty.clone(),
            newtype_types: self.newtype_types.clone(),
            newtype_inner: self.newtype_inner.clone(),
            var_newtype_type: self.var_newtype_type.clone(),
            stream_fns: self.stream_fns.clone(),
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
                let kw = if s.mutable { "let mut" } else { "let" };
                let ty = s.ty.as_ref().map(|t| format!(": {}", self.emit_type(t))).unwrap_or_default();
                format!("{} {}{} = {};", kw, s.name, ty, self.emit_expr(&s.value))
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

        // Which fs module to use (tokio for async, std for sync).
        let fs_mod = if self.in_async { "tokio::fs" } else { "std::fs" };

        match method {
            // fs.read("path") → Arc<str>
            "read" => {
                let path = a(0);
                format!(
                    "{{ let __boring_s = {}::read_to_string({}){aw}; Arc::<str>::from(__boring_s.as_str()) }}",
                    fs_mod, path, aw = aw
                )
            }

            // fs.readLines("path") → Vec<Arc<str>>
            "readLines" => {
                let path = a(0);
                format!(
                    "{{ let __boring_s = {}::read_to_string({}){aw}; __boring_s.lines().map(|l| Arc::<str>::from(l)).collect::<Vec<Arc<str>>>() }}",
                    fs_mod, path, aw = aw
                )
            }

            // fs.write("path", content)
            "write" => {
                let path    = a(0);
                let content = a(1);
                // content is Arc<str>; Deref<Target=str> → .as_bytes() works.
                format!("{}::write({}, ({}).as_bytes()){aw}", fs_mod, path, content, aw = aw)
            }

            // fs.append("path", content)  — OpenOptions::append
            "append" => {
                let path    = a(0);
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
                let path = a(0);
                if self.in_async {
                    format!("tokio::fs::metadata({}).await.is_ok()", path)
                } else {
                    format!("std::path::Path::new({}).exists()", path)
                }
            }

            // fs.isDir("path") → bool
            "isDir" => {
                let path = a(0);
                format!("std::path::Path::new({}).is_dir()", path)
            }

            // fs.isFile("path") → bool
            "isFile" => {
                let path = a(0);
                format!("std::path::Path::new({}).is_file()", path)
            }

            // fs.mkdir("path")  — create_dir_all
            "mkdir" => {
                let path = a(0);
                format!("{}::create_dir_all({}){aw}", fs_mod, path, aw = aw)
            }

            // fs.remove("path")  — remove file or directory tree
            "remove" => {
                let path = a(0);
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
                let from = a(0);
                let to   = a(1);
                format!("{}::rename({}, {}){aw}", fs_mod, from, to, aw = aw)
            }

            // fs.copy("src", "dst")
            "copy" => {
                let from = a(0);
                let to   = a(1);
                // std::fs::copy returns u64 (bytes); tokio::fs::copy too — discard it.
                format!("{{ let _ = {}::copy({}, {}){aw}; }}", fs_mod, from, to, aw = aw)
            }

            // fs.list("path") → Vec<Arc<str>> of entry names
            "list" => {
                let path = a(0);
                if self.in_async {
                    format!(
                        "{{ let mut __boring_dir = tokio::fs::read_dir({}){aw}; let mut __boring_entries: Vec<Arc<str>> = Vec::new(); while let Some(__boring_e) = __boring_dir.next_entry().await{prop} {{ __boring_entries.push(Arc::<str>::from(__boring_e.file_name().to_string_lossy().as_ref())); }} __boring_entries }}",
                        path, aw = aw, prop = prop
                    )
                } else {
                    format!(
                        "{{ std::fs::read_dir({}){prop}.filter_map(|e| e.ok()).map(|e| Arc::<str>::from(e.file_name().to_string_lossy().as_ref())).collect::<Vec<Arc<str>>>() }}",
                        path, prop = prop
                    )
                }
            }

            // fs.readBytes("path") → Vec<u8>
            "readBytes" => {
                let path = a(0);
                format!("{}::read({}){aw}", fs_mod, path, aw = aw)
            }

            // fs.writeBytes("path", bytes)
            "writeBytes" => {
                let path  = a(0);
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
                    return format!("{}.lock().unwrap().clone()", static_name);
                }
                escape_rust_keyword(n)
            }
        }
    }
}
