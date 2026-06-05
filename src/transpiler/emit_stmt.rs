use super::*;
use super::Transpiler;
use super::helpers::*;

impl Transpiler {
    pub(crate) fn emit_stmt(&mut self, stmt: &Stmt, is_last: bool) {
        match stmt {
            Stmt::Let(s)            => self.emit_let(s, false),
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
            Stmt::For(s)            => self.emit_for(s),
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
                let s = format!("yield {};", self.emit_expr_owned(expr));
                self.line(&s);
            }
            Stmt::Select(s)         => self.emit_select(s),
            Stmt::Comment(text)     => self.line(&format!("// {}", text)),
            Stmt::Expr(e)           => {
                if is_last && self.in_throws {
                    let s = format!("Ok({})", self.emit_expr_owned(e));
                    self.line(&s);
                } else if is_last && !self.fn_returns_void {
                    // Expression return: no semicolon (value-returning function).
                    // If the function returns Option<T>, wrap non-nil, non-Some values in Some().
                    let is_optional_return = matches!(
                        &self.fn_return_ty,
                        Some(Type::Optional(_))
                    );
                    let s = if is_optional_return && !is_option_expr(e) {
                        // Function returns Option<T>; expression is not already Option-typed.
                        // Wrap scalar/integer/variable values in Some() so Rust is happy.
                        let raw = self.emit_expr(e);
                        if raw == "None" || raw.starts_with("Some(") {
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
                    let s = self.emit_expr(e);
                    self.line(&format!("{};", s));
                }
            }
        }
    }

    pub(crate) fn emit_let(&mut self, s: &LetStmt, _is_last: bool) {
        // Track every declared local variable so that field/method access can distinguish
        // instance variables (use `.`) from type/module paths (use `::`).
        self.known_local_vars.insert(s.name.clone());
        // Shadowing: clear previous struct-type tracking so a re-declared variable with a
        // different type (e.g. `let d = Doubler()` then `let d'weak = c`) doesn't inherit
        // the old struct type and incorrectly suppress `.await.unwrap()` on `.value`.
        self.var_struct_types.remove(&s.name);
        // `let v` / `var v` — deferred initialisation: emit `let v;` and let Rust
        // enforce definite assignment via its own control-flow analysis.
        if s.value.is_none() {
            let kw = if s.mutable { "let mut" } else { "let" };
            if let Some(ty) = &s.ty {
                self.line(&format!("{} {}: {};", kw, s.name, self.emit_type(ty)));
            } else {
                self.line(&format!("{} {};", kw, s.name));
            }
            return;
        }
        let s_value = s.value.as_ref().expect("invariant: Let statement without type annotation must have an initializer value");
        // T'actor → Arc<tokio::sync::Mutex<T>>.
        // All field reads/writes and method calls on this variable will go through .lock().await.
        // Works with both `let` and `var` — the actor qualifier alone triggers mutex semantics.
        if let Some(ty) = &s.ty {
            if Self::is_mutex_binding(s.mutable, ty) {
                if let Some(inner) = Self::mutex_inner(ty) {
                    let mutex_ty = self.emit_mutex_type(inner);
                    let raw_val = self.emit_let_value(Some(inner), s_value);
                    let init = format!("Arc::new(tokio::sync::Mutex::new({}))", raw_val);
                    self.var_mutex_types.insert(s.name.clone());
                    self.arc_vars.insert(s.name.clone());
                    // Track struct type for getter dispatch, same as the general path below.
                    if let ExprKind::Call(callee, _) = &s_value.kind {
                        if let ExprKind::Var(type_name) = &callee.kind {
                            if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                                && self.struct_fields.contains_key(type_name.as_str())
                            {
                                self.var_struct_types.insert(s.name.clone(), type_name.clone());
                            }
                        }
                    }
                    self.line(&format!("let mut {}: {} = {};", s.name, mutex_ty, init));
                    return;
                }
            }
            // T'guard → Arc<tokio::sync::RwLock<T>>.
            if Self::is_rwlock_binding(s.mutable, ty) {
                if let Some(inner) = Self::rwlock_inner(ty) {
                    let rwlock_ty = self.emit_rwlock_type(inner);
                    let raw_val = self.emit_let_value(Some(inner), s_value);
                    let init = format!("Arc::new(tokio::sync::RwLock::new({}))", raw_val);
                    self.var_rwlock_types.insert(s.name.clone());
                    self.arc_vars.insert(s.name.clone());
                    // Track struct type for getter dispatch.
                    if let ExprKind::Call(callee, _) = &s_value.kind {
                        if let ExprKind::Var(type_name) = &callee.kind {
                            if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                                && self.struct_fields.contains_key(type_name.as_str())
                            {
                                self.var_struct_types.insert(s.name.clone(), type_name.clone());
                            }
                        }
                    }
                    let kw = if s.mutable { "let mut" } else { "let" };
                    self.line(&format!("{} {}: {} = {};", kw, s.name, rwlock_ty, init));
                    return;
                }
            }
        }
        let kw = if s.mutable { "let mut" } else { "let" };
        let vis = if s.is_pub { "pub " } else { "" };
        // Mutable string bindings must be Arc<str> (not &str) so they can be reassigned
        let is_mutable_string_lit = s.mutable && s.ty.is_none()
            && matches!(&s_value.kind, ExprKind::Str(_) | ExprKind::StringInterp(_));
        let is_mutable_string_ty = s.mutable
            && matches!(&s.ty, Some(Type::Named(n)) if n == "string" || n == "str")
            && matches!(&s_value.kind, ExprKind::Str(_) | ExprKind::StringInterp(_));
        let (ty, val) = if is_mutable_string_lit || is_mutable_string_ty {
            (
                ": Arc<str>".to_string(),
                self.emit_expr_owned(s_value),
            )
        } else {
            let val = self.emit_let_value(s.ty.as_ref(), s_value);
            // Inferred T'weak binding (bare `d'weak`, no compound qualifier): if the value
            // is Arc::downgrade(...), the annotation must be std::sync::Weak (not rc::Weak).
            // Compound forms like `Resource'task'weak` are handled correctly by emit_type.
            let ty = if let Some(ty) = s.ty.as_ref() {
                let is_bare_weak = matches!(ty,
                    Type::Qualified(inner, OwnerQual::Weak)
                    if !matches!(inner.as_ref(), Type::Qualified(_, _)));
                if is_bare_weak && val.starts_with("Arc::downgrade(") {
                    format!(": std::sync::Weak<_>")
                } else {
                    format!(": {}", self.emit_type(ty))
                }
            } else if matches!(&s_value.kind, ExprKind::Nil) {
                // `let x = nil` — Rust can't infer the type of `None`; add `Option<()>`.
                ": Option<()>".to_string()
            } else if val == "None" {
                // Cast that produces None (e.g. `42 as bool`) — add type annotation.
                ": Option<()>".to_string()
            } else {
                String::new()
            };
            (ty, val)
        };
        // Track mutable Arc<str> vars for read_line / clear() special-casing
        if is_mutable_string_lit || is_mutable_string_ty {
            self.string_arc_vars.insert(s.name.clone());
            self.string_vars.insert(s.name.clone());
        }
        // Also track immutable string literal vars so string methods (parseInt, indexOf, slice…)
        // can dispatch correctly even without an explicit type annotation.
        let is_immutable_string_lit = !s.mutable && s.ty.is_none()
            && matches!(&s_value.kind, ExprKind::Str(_) | ExprKind::StringInterp(_));
        if is_immutable_string_lit {
            self.string_vars.insert(s.name.clone());
        }
        // Track variables that hold collections (for {:?} formatting later)
        if looks_like_collection(&val) || is_collection_type(s.ty.as_ref()) {
            self.collection_vars.insert(s.name.clone());
        }
        // Track variables that unambiguously hold a Vec<T> (not HashMap/HashSet, not scalars from reduce).
        // Only consider expressions that END as a Vec — this excludes reduce/fold chains that
        // contain intermediate .collect::<Vec<_>>() but terminate as a scalar.
        if (expr_ends_as_vec(&val) && !looks_like_map_or_set(&val))
            || matches!(&s.ty, Some(Type::Array(_)))
        {
            self.vec_vars.insert(s.name.clone());
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
                        let kw = if s.mutable { "let mut" } else { "let" };
                        let spawn_s = if arc_captures.is_empty() {
                            format!("tokio::spawn(async move {})", inner_s)
                        } else {
                            let clones: String = arc_captures.iter()
                                .map(|v| format!("let {} = Arc::clone(&{});", v, v))
                                .collect::<Vec<_>>()
                                .join(" ");
                            format!("tokio::spawn({{ {} async move {} }})", clones, inner_s)
                        };
                        self.line(&format!("{} {} = {};", kw, s.name, spawn_s));
                        return;
                    }
                }
            }
        }
        // Track variables bound to user struct constructors for getter dispatch on non-self receivers.
        // Also handle type method calls: `let c2 = Counter2.zero()` → c2 is Counter2.
        if let ExprKind::MethodCall(callee_obj, _, _) = &s_value.kind {
            if let ExprKind::Var(type_name) = &callee_obj.kind {
                if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    if self.struct_fields.contains_key(type_name.as_str()) {
                        self.var_struct_types.insert(s.name.clone(), type_name.clone());
                    }
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
                    if let Some(ret_ty) = self.fn_return_types.get(type_name.as_str()) {
                        if matches!(ret_ty, Type::Optional(_)) {
                            self.optional_vars.insert(s.name.clone());
                        }
                    }
                }
            }
        }
        // When value is nil (None), the var is always optional.
        if matches!(&s_value.kind, ExprKind::Nil) {
            self.optional_vars.insert(s.name.clone());
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
                let dst_is_numeric = matches!(dst_ty, Type::Int | Type::Uint | Type::Float)
                    || matches!(dst_ty, Type::Named(n) if matches!(n.as_str(), "int" | "uint" | "float"));
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
        // Track Arc<T> variables (string or T'task / T'actor) — must be cloned before
        // being moved into an `async move {}` block so the outer binding stays valid.
        // Note: T'auto (Rc<T>) is NOT task-safe so it never needs Arc::clone for tasks.
        if let Some(ty) = &s.ty {
            if Self::is_string_type(ty) || Self::is_arc_qualified(ty) {
                self.arc_vars.insert(s.name.clone());
            }
            // Track T'weak variables — already Weak<T>, must not be downgraded again.
            if Self::is_weak_qualified(ty) {
                self.weak_vars.insert(s.name.clone());
            }
            // Track Optional-typed variables so they are never double-wrapped in Some().
            if matches!(ty, Type::Optional(_)) {
                self.optional_vars.insert(s.name.clone());
            }
            // Track var type for match subject enum inference.
            self.var_types.insert(s.name.clone(), ty.clone());
            // Track string vars for string concatenation detection.
            if Self::is_string_type(ty) {
                self.string_vars.insert(s.name.clone());
            }
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
                self.var_types.insert(s.name.clone(), Type::Named(format!("Rc<ref>")));
                return;
            }
        }
        if s.is_static {
            self.line(&format!("{}static {}{} = {};", vis, s.name, ty, val));
        } else {
            self.line(&format!("{}{} {}{} = {};", vis, kw, s.name, ty, val));
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
        matches!(ty, Type::Str)
            || matches!(ty, Type::Named(n) if n == "string")
            || matches!(ty, Type::Qualified(inner, _) if matches!(**inner, Type::Str))
    }

    /// Returns true for types that map to `Arc<T>` (or `Arc<Mutex<T>>` / `Arc<RwLock<T>>`) in Rust.
    pub(crate) fn is_arc_qualified(ty: &Type) -> bool {
        matches!(ty, Type::Qualified(_, OwnerQual::Task | OwnerQual::Actor | OwnerQual::Guard))
    }

    /// Returns true for `T'auto` which maps to `Rc<T>` in Rust.
    pub(crate) fn is_rc_qualified(ty: &Type) -> bool {
        matches!(ty, Type::Qualified(_, OwnerQual::Auto))
    }

    /// Returns true for `T'weak` (any form) which maps to `Weak<T>` in Rust.
    pub(crate) fn is_weak_qualified(ty: &Type) -> bool {
        matches!(ty, Type::Qualified(_, OwnerQual::Weak))
    }

    /// Returns true for `T'task'weak` / `T'actor'weak` — weak ref to an Arc-backed type.
    /// These require `Arc::downgrade` and `std::sync::Weak<T>` in Rust.
    pub(crate) fn is_arc_weak(ty: &Type) -> bool {
        matches!(ty,
            Type::Qualified(inner, OwnerQual::Weak)
            if matches!(inner.as_ref(), Type::Qualified(_, OwnerQual::Task | OwnerQual::Actor | OwnerQual::Guard))
        )
    }

    pub(crate) fn is_str_ref_type(ty: &Type) -> bool {
        matches!(ty, Type::Named(n) if n == "str")
            || matches!(ty, Type::Qualified(inner, OwnerQual::Const) if matches!(**inner, Type::Str))
    }

    /// Returns true if the expression produces a string (Arc<str>) value.
    pub(crate) fn is_string_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Str(_) | ExprKind::StringInterp(_) => true,
            ExprKind::Var(v) => self.string_vars.contains(v.as_str())
                || self.string_arc_vars.contains(v.as_str()),
            ExprKind::BinOp(BinOp::Add, l, r) => self.is_string_expr(l) || self.is_string_expr(r),
            _ => false,
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
                        Type::Named(n) => normalize_type_name(n),
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

    pub(crate) fn emit_let_value(&self, declared_ty: Option<&Type>, value: &Expr) -> String {
        // Resolve named type aliases through non_fn_type_aliases before dispatching.
        // e.g. `use Pt as LPoint'` makes `Pt` an alias for `Box<LPoint>`;
        // when calling `describe(p)` where describe expects `Pt`, we must Box::new() the arg.
        let declared_ty = if let Some(Type::Named(n)) = declared_ty {
            self.non_fn_type_aliases.get(n.as_str()).or(declared_ty)
        } else {
            declared_ty
        };
        // Context-aware DotIdent: `.Variant` with a known Named enum type → `EnumType::Variant`.
        // This ensures `.South` resolves to `Direction::South` (not a later enum with same variant).
        if let ExprKind::DotIdent(variant) = &value.kind {
            if let Some(Type::Named(enum_type)) = declared_ty {
                let enum_rust = normalize_type_name(enum_type);
                return format!("{}::{}", enum_rust, variant);
            }
            if let Some(Type::Qualified(inner, OwnerQual::Owned)) = declared_ty {
                if let Type::Named(enum_type) = inner.as_ref() {
                    let enum_rust = normalize_type_name(enum_type);
                    return format!("Box::new({}::{})", enum_rust, variant);
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
                    let rust_type = normalize_type_name(type_name);
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
            if matches!(ty, Type::Int | Type::Uint | Type::Float | Type::Named(_) | Type::Optional(_)));
        match declared_ty {
            Some(Type::Optional(inner)) if !is_nil => {
                if is_option_cast {
                    // Cast to a numeric type with an Optional declared type: emit `.ok()` so
                    // the result is Option<T>, matching the annotation.
                    // e.g. `let int? v = s as int` → `s.trim().parse::<i64>().ok()`
                    if let ExprKind::Cast(src, cast_ty) = &value.kind {
                        let src_s = self.emit_expr(src);
                        let parse_ty = match cast_ty {
                            Type::Int                          => Some("i64"),
                            Type::Uint                         => Some("u64"),
                            Type::Float                        => Some("f64"),
                            Type::Named(n) if n == "int"       => Some("i64"),
                            Type::Named(n) if n == "uint"      => Some("u64"),
                            Type::Named(n) if n == "float"     => Some("f64"),
                            _                                  => None,
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
                //   whose emitted form ends with ".ok()" or ".map(|i| i as i64)"
                let already_opt = inner_val.starts_with("Some(") || inner_val == "None"
                    || matches!(&value.kind, ExprKind::Var(v) if self.optional_vars.contains(v.as_str()))
                    || inner_val.ends_with(".ok()")
                    || inner_val.ends_with(".map(|i| i as i64)");
                if already_opt { return inner_val; }
                // `T'? (Box<T>?)`: wrap the value in Box::new(...) before Some(...)
                let wrapped = if matches!(inner.as_ref(), Type::Qualified(_, OwnerQual::Owned)) {
                    if inner_val.starts_with("Box::new(") { inner_val }
                    else { format!("Box::new({})", inner_val) }
                } else {
                    inner_val
                };
                format!("Some({})", wrapped)
            }
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
                    _ => self.emit_expr(value),
                }
            }
            // T'borrow (e.g. Task& or MyStruct&): pass by reference.
            Some(Type::Qualified(_, OwnerQual::Borrow)) => {
                let s = self.emit_expr(value);
                if s.starts_with('&') { s } else { format!("&{}", s) }
            }
            // T&auto (e.g. OCounter&auto) → &Rc<T>: pass reference to the Rc.
            Some(Type::Qualified(_, OwnerQual::BorrowAuto)) => {
                // Use the raw variable name (not .clone()) so the borrow is valid.
                if let ExprKind::Var(v) = &value.kind {
                    format!("&{}", v)
                } else {
                    let s = self.emit_expr(value);
                    if s.starts_with('&') { s } else { format!("&{}", s) }
                }
            }
            // T&task (e.g. OCounter&task) → &Arc<T>: pass reference to the Arc.
            Some(Type::Qualified(_, OwnerQual::BorrowTask)) => {
                // Use the raw variable name (not .clone()) so the borrow is valid.
                if let ExprKind::Var(v) = &value.kind {
                    format!("&{}", v)
                } else {
                    let s = self.emit_expr(value);
                    if s.starts_with('&') { s } else { format!("&{}", s) }
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
                // Ensure the result is Arc<str>; if not, wrap with Arc::<str>::from(x.to_string()).
                if s.starts_with("Arc::") {
                    s
                } else if matches!(&value.kind,
                    ExprKind::Var(v) if self.arc_vars.contains(v.as_str())
                        || self.string_arc_vars.contains(v.as_str()))
                {
                    // Known Arc<str> variable — clone the Arc pointer efficiently
                    format!("{}.clone()", s)
                } else {
                    format!("Arc::<str>::from({}.to_string())", s)
                }
            }
            // T'auto → Rc<T>: wrap in Rc::new().
            Some(t) if Self::is_rc_qualified(t) => {
                let inner = self.emit_expr(value);
                // Don't double-wrap: if the value is already Rc (from a variable declared 'auto),
                // pass it through directly.
                let already_rc = inner.starts_with("Rc::new(")
                    || inner.starts_with("Rc::clone(")
                    || matches!(&value.kind, ExprKind::Var(v)
                        if matches!(self.var_types.get(v.as_str()),
                            Some(Type::Qualified(_, OwnerQual::Auto))));
                if already_rc {
                    inner
                } else {
                    format!("Rc::new({})", inner)
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
                // Use Arc::downgrade when: explicit `T'task'weak` / `T'actor'weak` type,
                // or the RHS variable is known to be an Arc.
                if Self::is_arc_weak(t) || self.arc_vars.contains(inner.as_str()) {
                    format!("Arc::downgrade(&{})", inner)
                } else {
                    format!("Rc::downgrade(&{})", inner)
                }
            }
            // [T]'const / Dict'const / Set'const → Arc<Collection>: wrap in Arc::new().
            Some(Type::Qualified(inner, OwnerQual::Const))
                if matches!(inner.as_ref(), Type::Array(_) | Type::Dict(_, _) | Type::Set(_)) => {
                let v = self.emit_expr(value);
                if v.starts_with("Arc::new(") { v } else { format!("Arc::new({})", v) }
            }
            // T'task / T'actor → Arc<T> (or Arc<Mutex<T>> for actor): wrap in Arc::new().
            Some(t) if Self::is_arc_qualified(t) => {
                let inner = self.emit_expr(value);
                // Already wrapped in Arc::new/clone — pass through.
                if inner.starts_with("Arc::new(") || inner.starts_with("Arc::clone(") {
                    return inner;
                }
                // If the value is an existing Arc variable, clone it instead of moving.
                let is_existing_arc =
                    (Self::is_mutex_binding(false, t) && matches!(&value.kind, ExprKind::Var(v) if self.var_mutex_types.contains(v.as_str())))
                    || matches!(&value.kind, ExprKind::Var(v) if self.arc_vars.contains(v.as_str()));
                if is_existing_arc {
                    format!("Arc::clone(&{})", inner)
                } else if matches!(&value.kind, ExprKind::Var(_)) {
                    // Clone the value when wrapping a variable in Arc::new so that the original
                    // binding is not moved — important when the same variable is passed multiple times.
                    format!("Arc::new({}.clone())", inner)
                } else {
                    format!("Arc::new({})", inner)
                }
            }
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
            // T'owned (Box<T>): wrap the expression in Box::new(...) unless already boxed.
            Some(Type::Qualified(_, OwnerQual::Owned)) => {
                let inner = self.emit_expr(value);
                if inner.starts_with("Box::new(") {
                    inner
                } else {
                    format!("Box::new({})", inner)
                }
            }
            _ => {
                let s = self.emit_expr(value);
                // If the value is a variable that holds a struct (non-Arc) type, clone it
                // to avoid a move. In Boring, assignment always copies; Rust structs need .clone().
                if let ExprKind::Var(v) = &value.kind {
                    if self.var_struct_types.contains_key(v.as_str())
                        && !self.arc_vars.contains(v.as_str())
                        && !self.var_mutex_types.contains(v.as_str())
                        && !s.ends_with(".clone()")
                    {
                        return format!("{}.clone()", s);
                    }
                }
                s
            }
        }
    }

    pub(crate) fn emit_let_destructure(&mut self, s: &LetDestructureStmt) {
        // Track all bound names as known locals.
        for b in &s.bindings {
            if b.name != "_" {
                self.known_local_vars.insert(b.name.clone());
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
                let kw = if s.mutable { "let mut" } else { "let" };
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
            if let (Some(sender), Some(receiver)) = (s.bindings.get(0), s.bindings.get(1)) {
                if sender.name != "_" { self.channel_senders.insert(sender.name.clone()); }
                if receiver.name != "_" {
                    self.channel_receivers.insert(receiver.name.clone());
                    // Track whether the channel element type is `string` so that
                    // values received from it are known to be Arc<str>.
                    let is_string_elem = match &s.value.kind {
                        ExprKind::GenericCall(_, type_args, _) => type_args.first()
                            .map(|t| matches!(t, Type::Named(n) if n == "string" || n == "String"))
                            .unwrap_or(false),
                        _ => s.bindings.get(0)
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
            if let (Some(sender), Some(receiver)) = (s.bindings.get(0), s.bindings.get(1)) {
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
            if let (Some(sender), Some(receiver)) = (s.bindings.get(0), s.bindings.get(1)) {
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
            if let (Some(sender), Some(receiver)) = (s.bindings.get(0), s.bindings.get(1)) {
                if sender.name != "_" { self.watch_senders.insert(sender.name.clone()); }
                if receiver.name != "_" { self.watch_receivers.insert(receiver.name.clone()); }
                self.has_streams = true;
            }
        }
        // For `let T tx, rx = channel(n)`, emit with explicit type from the binding annotation.
        let val = if is_channel_typed {
            let item_ty = s.bindings.get(0)
                .and_then(|b| b.ty.as_ref())
                .map(|t| self.emit_type(t))
                .unwrap_or_else(|| "_".to_string());
            let cap = if let ExprKind::Call(_, args) = &s.value.kind {
                args.first().map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "0".to_string())
            } else { "0".to_string() };
            format!("tokio::sync::mpsc::channel::<{}>({})", item_ty, cap)
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
                let mut_kw = if s.mutable { "mut " } else { "" };
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
    }

    // ─── select: ──────────────────────────────────────────────────────────────

    /// Emit `tokio::select! { arm1 => { ... } arm2 => { ... } }`.
    pub(crate) fn emit_select(&mut self, s: &SelectStmt) {
        self.line("tokio::select! {");
        self.indent += 1;
        for arm in &s.arms {
            self.emit_select_arm(arm);
        }
        self.indent -= 1;
        self.line("}");
    }

    pub(crate) fn emit_select_arm(&mut self, arm: &SelectArm) {
        // Strip `.await` / `.await?` from expression strings so the future is passed
        // un-polled to tokio::select! (same technique as for `timeout` args).
        let strip_await = |s: String| -> String {
            if let Some(s) = s.strip_suffix(".await?") { s.to_string() }
            else if let Some(s) = s.strip_suffix(".await") { s.to_string() }
            else { s }
        };

        if arm.is_cancelled {
            // `cancelled:` → `_ = __task_cancel.cancelled() => { body }`
            self.line("_ = __task_cancel.cancelled() => {");
        } else if arm.is_after {
            // `after dur:` → `_ = tokio::time::sleep(dur) => { body }`
            let raw = self.emit_expr(&arm.expr);
            let dur = strip_await(raw);
            self.line(&format!("_ = tokio::time::sleep({}) => {{", dur));
        } else {
            // `var = expr:` or `_ = expr:`
            // For channel receivers, wrap the pattern in `Some(var)` to match `recv()` return type.
            let raw_expr = match &arm.expr.kind {
                ExprKind::Task(inner) => self.emit_expr(inner),
                _ => {
                    let s = self.emit_expr(&arm.expr);
                    strip_await(s)
                }
            };
            let (pattern, lhs) = match &arm.var {
                None => ("_".to_string(), "_".to_string()),
                Some(var_name) => {
                    // If the expression receiver is a channel_receiver, wrap in Some(var)
                    let is_recv = matches!(&arm.expr.kind,
                        ExprKind::MethodCall(obj, m, _)
                        if m == "recv" && matches!(&obj.kind, ExprKind::Var(v) if self.channel_receivers.contains(v.as_str()))
                    );
                    // broadcast receiver: wrap in Ok(var)
                    let is_broadcast_recv = matches!(&arm.expr.kind,
                        ExprKind::MethodCall(obj, m, _)
                        if m == "recv" && matches!(&obj.kind, ExprKind::Var(v) if self.broadcast_receivers.contains(v.as_str()))
                    );
                    if is_broadcast_recv {
                        (format!("Ok({})", var_name), var_name.clone())
                    } else if is_recv {
                        // If the channel carries strings, track the bound var as Arc<str>
                        // so that `msg == "literal"` comparisons emit correctly.
                        let is_str_chan = matches!(&arm.expr.kind,
                            ExprKind::MethodCall(obj, _, _)
                            if matches!(&obj.kind, ExprKind::Var(v) if self.string_channel_receivers.contains(v.as_str()))
                        );
                        if is_str_chan {
                            self.string_arc_vars.insert(var_name.clone());
                            self.string_vars.insert(var_name.clone());
                        }
                        (format!("Some({})", var_name), var_name.clone())
                    } else {
                        (var_name.clone(), var_name.clone())
                    }
                }
            };
            let _ = lhs; // suppress unused warning; lhs is the bound name in body
            self.line(&format!("{} = {} => {{", pattern, raw_expr));
        }

        // Emit body indented inside the arm
        self.indent += 1;
        let body_len = arm.body.len();
        for (i, stmt) in arm.body.iter().enumerate() {
            self.emit_stmt(stmt, i + 1 == body_len);
        }
        self.indent -= 1;
        self.line("}");
    }

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
                let val = if is_optional_return && !is_option_expr(e) {
                    let inner = self.emit_expr_owned(e);
                    let already_opt = inner.starts_with("Some(") || inner == "None"
                        || matches!(&e.kind, ExprKind::Var(v) if self.optional_vars.contains(v.as_str()));
                    if already_opt { inner } else { format!("Some({})", inner) }
                } else if matches!(&self.fn_return_ty,
                    Some(Type::Tuple(_)) | Some(Type::Array(_)) | Some(Type::Dict(_, _)))
                {
                    // Tuple/Array/Dict return: use emit_let_value for per-element
                    // coercion (e.g. string literals → Arc<str> in the right slots).
                    self.emit_let_value(self.fn_return_ty.as_ref(), e)
                } else {
                    self.emit_expr_owned(e)
                };
                // For trait return types the signature already uses `-> impl Trait`
                // (static dispatch), so the concrete value must be returned as-is — no
                // boxing.  Boxing would produce `Box<ConcreteType>` which does NOT
                // automatically implement `Trait` unless there is an explicit blanket impl.
                let val = val;
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
                        self.line(&make_err(&format!("Box::new(BoringError::String(Arc::<str>::from({})))", arc_inner)));
                    }
                    _ => {
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
                                "Box::new(BoringError::String(Arc::<str>::from(format!(\"{{}}\", {}))))",
                                val
                            )));
                        }
                    }
                }
            }
        }
    }

    pub(crate) fn emit_if(&mut self, s: &IfStmt, is_last: bool) {
        // When this if/else is the last statement in a value-returning function,
        // use emit_body so branch tails are returned without semicolons.
        let use_value_body = is_last && !self.fn_returns_void && !self.in_throws;
        for (i, (cond, body)) in s.branches.iter().enumerate() {
            let kw = if i == 0 { "if" } else { "} else if" };
            let cond_s = self.emit_expr(cond);
            self.line(&format!("{} {} {{", kw, cond_s));
            self.indent += 1;
            if use_value_body {
                self.emit_body(body);
            } else {
                // Use emit_loop_body so nested if-branches never get Ok()/Ok(()) wrapping.
                // Ok-wrapping is only correct at the top-level function body.
                self.emit_loop_body(body);
            }
            self.indent -= 1;
        }
        if let Some(else_body) = &s.else_body {
            self.line("} else {");
            self.indent += 1;
            if use_value_body {
                self.emit_body(else_body);
            } else {
                self.emit_loop_body(else_body);
            }
            self.indent -= 1;
        }
        self.line("}");
    }

    pub(crate) fn emit_if_let(&mut self, s: &IfLetStmt, is_last: bool) {
        // Track if-let bindings as known locals.
        for clause in &s.clauses {
            match clause {
                CondClause::Let(name, _) => { self.known_local_vars.insert(name.clone()); }
                CondClause::LetPat(pat, _) => { Self::collect_pattern_binds(pat, &mut self.known_local_vars); }
                CondClause::Expr(_) => {}
            }
        }
        // When this if-let is the last statement in a value-returning function,
        // use emit_body so the branch tail is returned without a semicolon.
        let use_value_body = is_last && !self.fn_returns_void && !self.in_throws;
        // Multi-clause: emit as a chain of let-else or if-let
        let cond_s = self.emit_cond_clauses(&s.clauses);
        self.line(&format!("if {} {{", cond_s));
        self.indent += 1;
        if use_value_body {
            self.emit_body(&s.then_body);
        } else {
            self.emit_loop_body(&s.then_body);
        }
        self.indent -= 1;
        if let Some(else_body) = &s.else_body {
            self.line("} else {");
            self.indent += 1;
            if use_value_body {
                self.emit_body(else_body);
            } else {
                self.emit_loop_body(else_body);
            }
            self.indent -= 1;
        }
        self.line("}");
    }

    pub(crate) fn emit_cond_clauses(&self, clauses: &[CondClause]) -> String {
        clauses.iter().map(|c| match c {
            CondClause::Let(name, expr) => {
                format!("let Some({}) = {}", name, self.emit_expr(expr))
            }
            CondClause::LetPat(pat, expr) => {
                format!("let {} = {}", self.emit_pattern(pat), self.emit_expr(expr))
            }
            CondClause::Expr(e) => self.emit_expr(e),
        }).collect::<Vec<_>>().join(" && ")
    }

    /// Collect all `Bind(name)` leaves from a pattern into a set (for known_local_vars tracking).
    pub(crate) fn collect_pattern_binds(pat: &Pattern, out: &mut std::collections::HashSet<String>) {
        match pat {
            Pattern::Bind(name) => { out.insert(name.clone()); }
            Pattern::Some(inner) => Self::collect_pattern_binds(inner, out),
            Pattern::Variant(_, subs) | Pattern::Tuple(subs) => {
                for s in subs { Self::collect_pattern_binds(s, out); }
            }
            _ => {}
        }
    }

    pub(crate) fn emit_match(&mut self, s: &MatchStmt, is_last: bool) {
        // ── Special case: `match error:` with qualified enum-variant patterns ────────
        //
        // In the interpreter, `error` is the original thrown value (e.g. MyError::NotFound).
        // In the transpiler, `error` is Box<dyn Error> — direct match arms don't compile.
        //
        // When subject is `error` and the first non-wildcard arm pattern is a qualified
        // `Enum::Variant`, downcast error via BoringError before matching:
        //
        //   let __boring_error_typed = error.downcast_ref::<BoringError>()
        //     .and_then(|be| if let BoringError::Other(tid, inner) = be {
        //         if *tid == TypeId::of::<AppError>() { inner.as_any().downcast_ref::<AppError>() }
        //         else { None }
        //     } else { None });
        //   match __boring_error_typed {
        //     Some(AppError::NotFound) => body,    ← variant arms
        //     None => fallback,                    ← wildcard arm
        //   }
        if let ExprKind::Var(vname) = &s.subject.kind {
            if vname == "error" {
                // Find enum type from first qualified variant pattern (name contains "::")
                let enum_type: Option<String> = s.arms.iter().find_map(|arm| {
                    arm.patterns.iter().find_map(|p| {
                        if let Pattern::Variant(name, _) = p {
                            name.split_once("::").map(|(et, _)| et.to_string())
                        } else { None }
                    })
                });

                if let Some(ref enum_ty) = enum_type {
                    // Emit the BoringError downcast to get Option<&EnumType>
                    self.line(&format!(
                        "let __boring_error_typed = error.downcast_ref::<BoringError>()\
                         .and_then(|__be| if let BoringError::Other(__tid, __boring_inner) = __be \
                         {{ if *__tid == std::any::TypeId::of::<{}>() \
                         {{ (**__boring_inner).as_any().downcast_ref::<{}>() }} \
                         else {{ None }} }} else {{ None }});",
                        enum_ty, enum_ty
                    ));

                    // Build a modified MatchStmt replacing the subject with __boring_error_typed
                    // and wrapping variant patterns in Some(…), wildcard → None.
                    let arms_transformed: Vec<MatchArm> = s.arms.iter().map(|arm| {
                        let new_pats: Vec<Pattern> = arm.patterns.iter().map(|p| match p {
                            Pattern::Wildcard => Pattern::Variant("None".into(), vec![]),
                            Pattern::Bind(_) => Pattern::Variant("None".into(), vec![]),
                            Pattern::Variant(name, subs) => {
                                // Rewrite qualified pattern: "AppError::NotFound" → Some(AppError::NotFound)
                                let inner = if subs.is_empty() {
                                    format!("Some({})", name)
                                } else {
                                    let sub_s: Vec<String> = subs.iter().map(|sp| match sp {
                                        Pattern::Bind(b) => b.clone(),
                                        Pattern::Wildcard => "_".into(),
                                        _ => "_".into(),
                                    }).collect();
                                    format!("Some({}({}))", name, sub_s.join(", "))
                                };
                                Pattern::Lit(LitPattern::Str(inner)) // use Str as a passthrough literal
                            }
                            other => other.clone(),
                        }).collect();
                        MatchArm { patterns: new_pats, guard: arm.guard.clone(), body: arm.body.clone(), line: arm.line }
                    }).collect();

                    // Emit the transformed match against __boring_error_typed
                    let use_value_body = is_last && !self.fn_returns_void && !self.in_throws;
                    self.line("match __boring_error_typed {");
                    self.indent += 1;
                    for arm in &arms_transformed {
                        for pat in &arm.patterns {
                            let pat_s = match pat {
                                Pattern::Lit(LitPattern::Str(s)) => s.clone(), // passthrough literal
                                Pattern::Variant(n, _) => n.clone(),
                                Pattern::Wildcard => "_".into(),
                                _ => "_".into(),
                            };
                            if let Some(guard) = &arm.guard {
                                self.line(&format!("{} if {} => {{", pat_s, self.emit_expr(guard)));
                            } else {
                                self.line(&format!("{} => {{", pat_s));
                            }
                        }
                        self.indent += 1;
                        match &arm.body {
                            MatchBody::Expr(e) => {
                                if use_value_body {
                                    let val = self.emit_expr_owned(e);
                                    self.line(&val);
                                } else {
                                    let val = self.emit_expr_owned(e);
                                    self.line(&format!("{};", val));
                                }
                            }
                            MatchBody::Block(stmts) => {
                                if use_value_body {
                                    self.emit_body(stmts);
                                } else {
                                    self.emit_loop_body(stmts);
                                }
                            }
                        }
                        self.indent -= 1;
                        self.line("}");
                    }
                    self.indent -= 1;
                    self.line("}");
                    return;
                }
            }
        }

        // When the whole match is the last stmt in a value-returning function,
        // each arm body also returns its value — use emit_body instead of emit_loop_body.
        let use_value_body = is_last && !self.fn_returns_void && !self.in_throws;

        // Detect if the match subject is a function type parameter being matched against
        // concrete struct types. Rust cannot match `S` (a type param) against struct patterns,
        // so we transform this into a std::any::Any downcast chain.
        // The subject variable name (e.g. `s`) may differ from the type parameter name (e.g. `S`),
        // so we look up the variable's type in var_types and check if *that* is a type param.
        let subj_is_type_param = if let ExprKind::Var(vname) = &s.subject.kind {
            if self.current_fn_type_params.contains(vname.as_str()) {
                // Variable name itself is a type param (unusual but possible)
                true
            } else {
                // Check if the variable's declared type is a type parameter.
                // Param types can be Type::Named("S") or Type::TypeParam("S").
                let ty_name = self.var_types.get(vname.as_str()).and_then(|t| match t {
                    Type::Named(n) => Some(n.as_str()),
                    Type::TypeParam(n) => Some(n.as_str()),
                    _ => None,
                });
                ty_name.map(|n| self.current_fn_type_params.contains(n)).unwrap_or(false)
            }
        } else {
            false
        };
        let has_struct_arm = subj_is_type_param && s.arms.iter().any(|arm| {
            arm.patterns.iter().any(|p| {
                if let Pattern::Variant(name, _) = p {
                    self.struct_fields.contains_key(name.as_str())
                } else {
                    false
                }
            })
        });

        if has_struct_arm {
            // Emit an Any-downcast chain instead of a match expression.
            let subj_var = if let ExprKind::Var(vname) = &s.subject.kind {
                vname.clone()
            } else {
                self.emit_expr(&s.subject)
            };
            let any_var = format!("__any_{}", subj_var);
            self.line(&format!("let {} = &{} as &dyn std::any::Any;", any_var, subj_var));
            let mut first = true;
            for arm in &s.arms {
                // Check if this arm's pattern is a struct type check.
                let struct_name = arm.patterns.iter().find_map(|p| {
                    if let Pattern::Variant(name, _) = p {
                        if self.struct_fields.contains_key(name.as_str()) {
                            return Some(name.clone());
                        }
                    }
                    None
                });
                let is_wildcard = arm.patterns.iter().any(|p| matches!(p, Pattern::Wildcard));

                let kw = if first { "if" } else { "} else if" };
                if let Some(sname) = struct_name {
                    self.line(&format!("{} {}.is::<{}>() {{", kw, any_var, sname));
                    self.indent += 1;
                    // Emit arm body.
                    match &arm.body {
                        MatchBody::Expr(e) => {
                            let expr_s = self.emit_expr(e);
                            if use_value_body {
                                self.line(&expr_s);
                            } else {
                                self.line(&format!("{};", expr_s));
                            }
                        }
                        MatchBody::Block(stmts) => {
                            if use_value_body {
                                self.emit_body(stmts);
                            } else {
                                self.emit_loop_body(stmts);
                            }
                        }
                    }
                    self.indent -= 1;
                    first = false;
                } else if is_wildcard {
                    // Wildcard arm becomes the else branch.
                    self.line("} else {");
                    self.indent += 1;
                    match &arm.body {
                        MatchBody::Expr(e) => {
                            let expr_s = self.emit_expr(e);
                            if use_value_body {
                                self.line(&expr_s);
                            } else {
                                self.line(&format!("{};", expr_s));
                            }
                        }
                        MatchBody::Block(stmts) => {
                            if use_value_body {
                                self.emit_body(stmts);
                            } else {
                                self.emit_loop_body(stmts);
                            }
                        }
                    }
                    self.indent -= 1;
                }
            }
            self.line("}");
            return;
        }

        // Try to infer which enum type the match subject has, so patterns can be
        // qualified with the correct enum name when multiple enums share variant names.
        let inferred_enum = self.infer_match_enum(&s.subject, &s.arms);
        let prev_match_enum = self.match_subject_enum.clone();
        self.match_subject_enum = inferred_enum;

        let subj_raw = self.emit_expr(&s.subject);
        // If the match subject is Rc<T> or Arc<T>, dereference it so patterns can match T.
        let subj_ty = if let ExprKind::Var(vname) = &s.subject.kind {
            self.var_types.get(vname.as_str()).cloned()
        } else {
            None
        };
        let is_smart_ptr = matches!(&subj_ty, Some(Type::Qualified(_, OwnerQual::Auto | OwnerQual::Task | OwnerQual::Owned)));
        let subj = if is_smart_ptr {
            format!("(*{})", subj_raw)
        } else {
            subj_raw
        };
        self.line(&format!("match {} {{", subj));
        self.indent += 1;
        for arm in &s.arms {
            self.emit_match_arm(arm, use_value_body);
        }
        self.indent -= 1;
        self.line("}");

        self.match_subject_enum = prev_match_enum;
    }

    /// Infer the enum name for a match subject so we can qualify variant patterns.
    pub(crate) fn infer_match_enum(&self, subject: &Expr, arms: &[crate::ast::MatchArm]) -> Option<String> {
        // Strategy 1: subject is a named variable with a known type (from var_types).
        if let ExprKind::Var(vname) = &subject.kind {
            if let Some(ty) = self.var_types.get(vname.as_str()) {
                // Helper: extract the base type name, unwrapping smart-pointer qualifiers.
                let tname = match ty {
                    Type::Named(n) => Some(n.as_str()),
                    // MOpt'auto → Rc<MOpt>: extract "MOpt"
                    Type::Qualified(inner, _) => match inner.as_ref() {
                        Type::Named(n) => Some(n.as_str()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(tname) = tname {
                    if self.enum_variant_fields.keys().any(|k| k.starts_with(&format!("{}::", tname))) {
                        return Some(tname.to_string());
                    }
                }
            }
        }

        // Strategy 2: look at the function parameter type for the subject variable.
        if let ExprKind::Var(vname) = &subject.kind {
            // fn_sigs is keyed by function name, not useful here directly.
            // Check if it's `self` — use self_type.
            if vname == "self" {
                // Could be matching on self in an enum method, but skip for now.
            }
        }

        // Strategy 3: look for function call that returns an enum type.
        if let ExprKind::Call(callee, _) = &subject.kind {
            if let ExprKind::Var(fn_name) = &callee.kind {
                if let Some(ret_ty) = self.fn_return_types.get(fn_name.as_str()) {
                    if let Type::Named(tname) = ret_ty {
                        if self.enum_variant_fields.keys().any(|k| k.starts_with(&format!("{}::", tname))) {
                            return Some(tname.clone());
                        }
                    }
                }
            }
        }

        // Strategy 4: if all arm variant patterns uniquely belong to one enum, use it.
        // Collect all variant names from the arm patterns.
        let mut variant_names: Vec<String> = Vec::new();
        for arm in arms {
            for pat in &arm.patterns {
                collect_pattern_variants(pat, &mut variant_names);
            }
        }

        if variant_names.is_empty() {
            return None;
        }

        // Find enum(s) that contain all variant names.
        // Build a set of candidate enums.
        let mut candidates: Option<std::collections::HashSet<String>> = None;
        for vname in &variant_names {
            let enums_with_variant: std::collections::HashSet<String> = self.enum_variant_fields
                .keys()
                .filter(|k| k.ends_with(&format!("::{}", vname)))
                .filter_map(|k| k.split("::").next().map(|s| s.to_string()))
                .collect();
            if enums_with_variant.is_empty() {
                continue;
            }
            candidates = Some(match candidates.take() {
                None => enums_with_variant,
                Some(prev) => prev.intersection(&enums_with_variant).cloned().collect(),
            });
        }

        match candidates {
            Some(set) if set.len() == 1 => set.into_iter().next(),
            _ => None,
        }
    }

    pub(crate) fn emit_match_arm(&mut self, arm: &MatchArm, use_value_body: bool) {
        let pats: Vec<String> = arm.patterns.iter().map(|p| self.emit_pattern(p)).collect();
        let guard = arm.guard.as_ref().map(|g| format!(" if {}", self.emit_expr(g))).unwrap_or_default();
        let pat_s = pats.join(" | ");
        match &arm.body {
            MatchBody::Expr(e) => {
                let expr_s = self.emit_expr(e);
                self.line(&format!("{}{} => {},", pat_s, guard, expr_s));
            }
            MatchBody::Block(stmts) => {
                self.line(&format!("{}{} => {{", pat_s, guard));
                self.indent += 1;
                if use_value_body {
                    self.emit_body(stmts);
                } else {
                    self.emit_loop_body(stmts);
                }
                self.indent -= 1;
                self.line("}");
            }
        }
    }

    pub(crate) fn emit_pattern(&self, pat: &Pattern) -> String {
        match pat {
            Pattern::Wildcard   => "_".into(),
            Pattern::Bind(n)    => n.clone(),
            Pattern::None       => {
                // If the match subject is a user-defined enum with a `None` variant, qualify it.
                if let Some(enum_name) = &self.match_subject_enum {
                    let key = format!("{}::None", enum_name);
                    if self.enum_variant_fields.contains_key(&key) {
                        return format!("{}::None", enum_name);
                    }
                }
                "None".into()
            }
            Pattern::Some(inner) => {
                // If the match subject is a user-defined enum with a `Some` variant,
                // qualify the pattern as `EnumName::Some(...)`.
                let inner_s = self.emit_pattern(inner);
                let wildcard = matches!(inner.as_ref(), Pattern::Wildcard);
                if let Some(enum_name) = &self.match_subject_enum {
                    let key = format!("{}::Some", enum_name);
                    if self.enum_variant_fields.contains_key(&key) {
                        if wildcard {
                            return format!("{}::Some(_)", enum_name);
                        } else {
                            return format!("{}::Some({})", enum_name, inner_s);
                        }
                    }
                }
                // Boring Optional — standard Rust Option pattern
                if wildcard { "Some(_)".into() } else { format!("Some({})", inner_s) }
            }
            Pattern::Lit(lit)   => match lit {
                LitPattern::Int(n)  => n.to_string(),
                LitPattern::Float(f) => {
                    let s = format!("{}", f);
                    if s.contains('.') || s.contains('e') || s.contains('E') { s }
                    else { format!("{}.0", s) }
                }
                LitPattern::Str(s)  => format!("\"{}\"", s),
                LitPattern::Bool(b) => b.to_string(),
                LitPattern::Nil     => "None".into(),
            },
            Pattern::Variant(name, fields) => {
                // Pre-qualified pattern `Enum::Variant` (written as `Enum.Variant` in Boring) —
                // emit verbatim, skip all lookup / re-qualification logic.
                if name.contains("::") {
                    return if fields.is_empty() {
                        name.clone()
                    } else {
                        let fs: Vec<String> = fields.iter().map(|p| self.emit_pattern(p)).collect();
                        format!("{}({})", name, fs.join(", "))
                    };
                }
                // `Some(x)` and `None` — treat as Option patterns ONLY when the match
                // subject is not a user-defined enum that has a real `Some`/`None` variant.
                let subject_has_variant = self.match_subject_enum.as_deref().map(|en| {
                    self.enum_variant_fields.contains_key(&format!("{}::{}", en, name))
                }).unwrap_or(false);
                if !subject_has_variant {
                    if (name == "Some" || name == "some") && fields.len() == 1 {
                        return format!("Some({})", self.emit_pattern(&fields[0]));
                    }
                    if (name == "None" || name == "none") && fields.is_empty() {
                        return "None".into();
                    }
                }
                // Check if this is a struct (not an enum variant).
                if let Some(field_names) = self.struct_fields.get(name.as_str()) {
                    // Struct destructuring: `Point2D { x, y }` or `Point2D { x: a, y: _ }`
                    if fields.is_empty() {
                        // Type-check-only pattern: `Point2D { .. }`
                        return format!("{} {{ .. }}", name);
                    }
                    let pairs: Vec<String> = fields.iter().enumerate().map(|(i, pat)| {
                        let fname = field_names.get(i).map(|(n, _)| n.as_str()).unwrap_or("_");
                        match pat {
                            Pattern::Wildcard => format!("{}: _", fname),
                            Pattern::Bind(b) if b == fname => fname.to_string(),
                            Pattern::Bind(b) => format!("{}: {}", fname, b),
                            _ => format!("{}: {}", fname, self.emit_pattern(pat)),
                        }
                    }).collect();
                    return format!("{} {{ {} }}", name, pairs.join(", "));
                }
                // Qualify with enum name: `Num(v)` → `ExprEnum::Num(v)`
                // Priority: 1) use the inferred enum from match subject (match_subject_enum)
                //           2) disambiguate by field count (tuple vs unit variant)
                //           3) fall back to last-registered enum_variants entry
                let qualified = if let Some(subject_enum) = &self.match_subject_enum {
                    // Use the inferred enum from the match context.
                    let key = format!("{}::{}", subject_enum, name);
                    if self.enum_variant_fields.contains_key(&key) {
                        format!("{}::{}", subject_enum, name)
                    } else if let Some(enum_name) = self.enum_variants.get(name) {
                        format!("{}::{}", enum_name, name)
                    } else {
                        name.clone()
                    }
                } else if let Some(enum_name) = self.enum_variants.get(name) {
                    if !fields.is_empty() {
                        // Check if the mapped enum's variant actually has fields.
                        let mapped_key = format!("{}::{}", enum_name, name);
                        let mapped_has_fields = self.enum_variant_fields.get(&mapped_key)
                            .map(|f| !f.is_empty())
                            .unwrap_or(false);
                        if !mapped_has_fields {
                            // Look for another enum that has this variant with fields.
                            let alt = self.enum_variant_fields.iter()
                                .find(|(k, v)| {
                                    k.ends_with(&format!("::{}", name)) && !v.is_empty()
                                })
                                .and_then(|(k, _)| k.split("::").next().map(|s| s.to_string()));
                            if let Some(alt_enum) = alt {
                                format!("{}::{}", alt_enum, name)
                            } else {
                                format!("{}::{}", enum_name, name)
                            }
                        } else {
                            format!("{}::{}", enum_name, name)
                        }
                    } else {
                        format!("{}::{}", enum_name, name)
                    }
                } else {
                    name.clone()
                };
                if fields.is_empty() {
                    qualified
                } else {
                    let fs: Vec<String> = fields.iter().map(|p| self.emit_pattern(p)).collect();
                    format!("{}({})", qualified, fs.join(", "))
                }
            }
            Pattern::Tuple(elems) => {
                let es: Vec<String> = elems.iter().map(|p| self.emit_pattern(p)).collect();
                format!("({})", es.join(", "))
            }
        }
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

        // Detect `for item in stream_fn(args):` — emit a pinned stream consumer instead.
        let stream_fn_name: Option<String> = match &s.iterable.kind {
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(name) = &callee.kind {
                    if self.stream_fns.contains(name.as_str()) {
                        Some(name.clone())
                    } else { None }
                } else { None }
            }
            _ => None,
        };
        if let Some(ref fn_name) = stream_fn_name {
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
            if self.var_mutex_types.contains(actor_name.as_str()) && self.in_async {
                let vars = if s.vars.is_empty() {
                    "_".into()
                } else {
                    s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
                };
                let guard = format!("__guard_{}", actor_name);
                self.line(&format!("let {} = {}.lock().await;", guard, actor_name));
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
            if self.var_rwlock_types.contains(guard_name.as_str()) && self.in_async {
                let vars = if s.vars.is_empty() {
                    "_".into()
                } else {
                    s.vars.iter().map(|v| escape_rust_keyword(v)).collect::<Vec<_>>().join(", ")
                };
                let rguard = format!("__rguard_{}", guard_name);
                self.line(&format!("let {} = {}.read().await;", rguard, guard_name));
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
        let iter_expr = match &s.iterable.kind {
            ExprKind::Range { .. } => iter,
            ExprKind::Field(obj, _) if matches!(&obj.kind, ExprKind::Var(v) if v == "self") => {
                format!("{}.iter().cloned()", iter)
            }
            _ => format!("{}.into_iter()", iter),
        };
        self.line(&format!("for {} in {} {{", vars, iter_expr));
        self.indent += 1;
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
        // Restore: loop vars and any let-bindings inside the body are not
        // visible after the for loop (emit_loop_body already saved/restored
        // its own inner lets; here we restore the loop var additions).
        self.known_local_vars = saved_locals;
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
        self.line(&format!("while let Ok({}) = {}.recv().await {{", vars, rx_name));
        self.indent += 1;
        self.emit_loop_body(&s.body);
        self.indent -= 1;
        self.line("}");
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
                                if let Some(ty) = self.var_types.get(src.as_str()).cloned() {
                                    if let Type::Optional(inner) = ty {
                                        self.var_types.insert(name.clone(), *inner.clone());
                                        if Self::is_string_type(&inner) {
                                            self.string_vars.insert(name.clone());
                                        }
                                        if matches!(*inner, Type::Optional(_)) {
                                            self.optional_vars.insert(name.clone());
                                        }
                                    }
                                }
                            }
                            let val = self.emit_expr(expr);
                            self.line(&format!("let Some({}) = {} else {{", name, val));
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
                            self.line(&format!("let {} = {} else {{", pat_s, val));
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
                                for (arm_pat, error_bind) in boring_type_to_boring_val_arms(ty_name) {
                                    self.line(&format!("{} => {{", arm_pat));
                                    self.indent += 1;
                                    self.line(&format!("let error: Arc<str> = {};", error_bind));
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
                                    self.line("let error: Arc<str> = Arc::<str>::from(__boring_err.to_string());");
                                }
                                self.emit_loop_body(&body_stmts);
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
                                        self.line("let error: Arc<str> = Arc::<str>::from(__e.to_string());");
                                    }
                                    self.emit_loop_body(&body_stmts);
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
                    self.line("{ let error: Arc<str> = Arc::<str>::from(__err.to_string());");
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

    // ── Expressions ───────────────────────────────────────────────────────────

}
