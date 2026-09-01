use super::*;
use super::Transpiler;
use super::helpers::*;

/// Is `ty` one of Boring's known numeric scalar types (bare `int`/`uint`, every
/// fixed-width int alias, and both float widths) — by `Type` variant or by its
/// `Type::Named` string alias? Shared by `emit_expr_cast`'s numeric-source
/// checks: a bare numeric variable (`n as uint32`) and a numeric-element
/// array/dict index expression (`bytes[16] as uint32`, see
/// `src_is_numeric_index`) both need the exact same "is this already a number"
/// test so an `as` cast on either emits a plain `as T` widen, not the
/// string-parsing codegen (`.trim().parse::<T>()`) meant for `string` sources.
fn is_known_numeric_scalar_type(ty: &Type) -> bool {
    matches!(ty, Type::Int | Type::Uint | Type::Uint8 | Type::Float32 | Type::Float64
        | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
        | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128)
    || matches!(ty, Type::Named(n) if matches!(n.as_str(),
        "int" | "uint" | "uint8" | "float" | "float32" | "float64"
        | "int8" | "int16" | "int32" | "int64" | "int128"
        | "uint16" | "uint32" | "uint64" | "uint128"
        | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
        | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
        | "f32" | "f64"))
}

impl Transpiler {
    pub(crate) fn emit_expr(&self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Int(n)   => n.to_string(),
            // Oversized decimal literal (overflows `i64`, fits `u64`). Must
            // carry an explicit `u64` suffix: an unsuffixed Rust integer
            // literal this large would default-infer as `i32` and fail to
            // compile.
            ExprKind::UInt64(n) => format!("{}u64", n),
            ExprKind::Float(f) => {
                let s = format!("{}", f);
                if s.contains('.') || s.contains('e') || s.contains('E') { s }
                else { format!("{}.0", s) }
            }
            ExprKind::Str(s) => format!("\"{}\"", escape_str(s)),
            ExprKind::StringInterp(segs) => self.emit_interp(segs),
            ExprKind::Bool(b)  => b.to_string(),
            ExprKind::Nil      => "None".into(),
            ExprKind::Void     => "()".into(),
            ExprKind::Var(n)   => {
                // In init body, `self` is the local `__self` variable.
                if self.in_init_body && n == "self" {
                    return "__self".to_string();
                }
                // Implicit self: inside a struct method, a bare field name maps to `self.field`
                // only when it is NOT already declared as a local variable.
                if let Some(struct_name) = &self.self_type {
                    if !self.known_local_vars.contains(n.as_str()) {
                        if let Some(fields) = self.struct_fields.get(struct_name.as_str()) {
                            if let Some((_, fty)) = fields.iter().find(|(f, _)| f == n) {
                                let self_ref = if self.in_init_body { "__self" } else { "self" };
                                let field_s = escape_rust_keyword(n);
                                // Auto-clone non-Copy fields to avoid moving out of &self/&mut self.
                                // Only in read (RHS) context — not when the field is being assigned to.
                                // Do NOT clone collections (dict/array/set/vec) — they are mutated
                                // in-place via method calls on self; cloning would make mutations no-ops.
                                let is_primitive = matches!(fty, Type::Int | Type::Uint | Type::Float32 | Type::Float64 | Type::Bool)
                                    || matches!(fty, Type::Named(s) if matches!(s.as_str(), "int" | "uint" | "float" | "float32" | "float64" | "bool" | "usize" | "isize"));
                                // `.without_mut()` — a `var mut {K=V} field` / `mut [T] field`
                                // struct field declaration wraps its type in `Type::Mut(..)`
                                // (see `wrap_type_mut` in src/parser/mod.rs), so a direct
                                // `matches!(fty, Type::Dict(..) | ...)` would miss any
                                // mut/var-mut collection field and wrongly `.clone()` it here.
                                let is_collection = matches!(fty.without_mut(), Type::Dict(..) | Type::Array(_) | Type::Set(_));
                                let needs_clone = !is_primitive && !is_collection && !self.in_lhs_assign.get();
                                return if needs_clone {
                                    format!("{}.{}.clone()", self_ref, field_s)
                                } else {
                                    format!("{}.{}", self_ref, field_s)
                                };
                            }
                        }
                    }
                }
                // `var` primitive params are `&mut T` — auto-deref on use.
                if self.var_primitive_params.contains(n.as_str()) {
                    return format!("(*{})", n);
                }
                // `lazy` vars hold a `OnceCell<T>` — reads unwrap via `.get().expect(…)`.
                // For Copy types (int, float, bool) use `.copied()` to avoid a ref.
                if self.lazy_vars.contains(n.as_str()) {
                    let is_copy = self.lazy_var_types.get(n.as_str())
                        .map(|t| t.is_copy())
                        .unwrap_or(false);
                    return if is_copy {
                        format!("{}.get().copied().expect(\"{} used before lazy init\")", n, n)
                    } else {
                        format!("{}.get().expect(\"{} used before lazy init\")", n, n)
                    };
                }
                self.map_builtin_var(n)
            }

            ExprKind::BinOp(op, l, r) => self.emit_expr_binop(expr, op, l, r),
            ExprKind::UnaryOp(op, e) => {
                let s = self.emit_expr(e);
                // Struct unary neg dispatch: `-a` → `a.clone().neg()`
                if matches!(op, UnaryOp::Neg) {
                    if let ExprKind::Var(vname) = &e.kind {
                        if let Some(sty) = self.var_struct_types.get(vname.as_str()).cloned() {
                            let key = format!("{}::neg", sty);
                            if self.struct_operator_methods.contains(&key) {
                                return format!("{}.clone().neg()", s);
                            }
                        }
                    }
                }
                match op {
                    UnaryOp::Neg    => format!("(-{})", s),
                    UnaryOp::Not    => format!("(!{})", s),
                    UnaryOp::BitNot => format!("(!{})", s),
                }
            }
            ExprKind::Assign(target, value) => self.emit_expr_assign(target, value),
            ExprKind::QuestionAssign(target, rhs) => {
                // `w ?= expr`
                // For lazy vars: emit `w.get_or_init(|| expr)`
                // For optional vars: emit a nil-coalescing assignment block
                if let ExprKind::Var(var_name) = &target.kind {
                    if self.lazy_vars.contains(var_name.as_str()) {
                        let rhs_s = self.emit_expr_owned(rhs);
                        return format!("{}.get_or_init(|| {})", var_name, rhs_s);
                    }
                    // Non-lazy: nil-coalescing assignment `lhs = lhs else rhs`
                    let else_rhs = self.emit_expr_owned(rhs);
                    let lhs_s = self.emit_expr(target);
                    return format!("{{ if {lhs_s}.is_none() {{ {lhs_s} = Some({else_rhs}); }} }}",
                        lhs_s = lhs_s, else_rhs = else_rhs);
                }
                // Fallback: desugar as nil-coalescing
                let lhs_s = self.emit_expr(target);
                let rhs_s = self.emit_expr_owned(rhs);
                format!("{{ if {lhs_s}.is_none() {{ {lhs_s} = Some({rhs_s}); }} }}", lhs_s = lhs_s, rhs_s = rhs_s)
            }
            ExprKind::Field(obj, field) => self.emit_expr_field(obj, field),
            ExprKind::Index(obj, idx) => self.emit_expr_index(obj, idx),
            ExprKind::Call(callee, args) => self.emit_call(callee, args),
            ExprKind::MethodCall(obj, method, args) => self.emit_method_call(obj, method, args),
            ExprKind::Pipe(lhs, name, args) => self.emit_pipe(lhs, name, args),
            ExprKind::GenericCall(callee, type_args, args) => self.emit_generic_call(callee, type_args, args),

            ExprKind::New { ctor, .. } => {
                // Arena placement is not yet emitted — just emit the constructor call.
                // Qualifier wrapping is handled by the surrounding qualifier emission infrastructure.
                self.emit_expr(ctor)
            }

            ExprKind::KernelLaunch { .. } => {
                // GPU kernel launch transpilation — not yet implemented.
                "/* kernel launch */".to_string()
            }

            ExprKind::TryElse(e, default) => {
                // `try expr else default` — calls a throws/Result function and returns the Ok
                // value or the default on error.

                // `try? expr` desugars to TryElse(expr, Nil). Some builtins (`fromJson<T>(s)`,
                // `fs.read(path)`, …) already do their own Result→Option/panic handling in a
                // plain (non-throws) context — book.md documents `try? fromJson<T>(s)` as
                // exactly equivalent to the plain form. Letting the generic path below run
                // (emit the inner expr plain, then append another `.ok()`) double-handles
                // them: `.ok()` on an already-`Option` (fromJson) doesn't compile, and
                // `fs.read`'s plain form panics via `.unwrap()` instead of yielding `None`,
                // defeating the whole point of `try?`. See docs/try-wrap-double-handling-bug.md.
                // Recognize these up front and emit their dedicated `try?`-aware form directly,
                // instead of falling into the generic "emit plain, append .ok()" path.
                if matches!(default.kind, ExprKind::Nil) {
                    if let Some(code) = self.emit_try_optional_self_handling_builtin(e) {
                        return code;
                    }
                }

                // The inner expression must NOT get `?` propagation — TryElse handles the error
                // locally. Use a sub-transpiler with throws flags cleared.
                let mut sub = self.make_sub();
                sub.in_throws = false;
                sub.in_try_body = false;
                let inner = sub.emit_expr(e);
                // `try? expr` desugars to TryElse(expr, Nil) — emit the idiomatic `.ok()`
                // (Result<T,E> → Option<T>) rather than .unwrap_or_else(|_| None).
                if matches!(default.kind, ExprKind::Nil) {
                    return format!("{}.ok()", inner);
                }
                let default_s = self.emit_expr_owned(default);
                format!("{}.unwrap_or_else(|_| {})", inner, default_s)
            }

            ExprKind::TryElseBlock(try_stmts, else_stmts) => {
                // `try … else …` — try/else expression in all four body combinations.
                //
                // Sync context (not inside an async fn):
                //   { match (|| -> Result<_, Box<dyn Error + Send + Sync>> { … })() {
                //       Ok(__boring_v)  => __boring_v,
                //       Err(__boring_e) => { let error = …; <else body> } } }
                //
                // Async context (inside a task/async fn):
                //   { let __boring_r: Result<_, Box<dyn Error + Send + Sync>> =
                //       async { … }.await;
                //     match __boring_r {
                //       Ok(__boring_v)  => __boring_v,
                //       Err(__boring_e) => { let error = …; <else body> } } }
                //
                // The async form avoids the E0728 "await inside non-async closure" error
                // that arises when the try body contains task function calls (.await).
                let mut try_sub = self.make_sub();
                try_sub.in_throws = true;
                try_sub.fn_returns_void = false;
                try_sub.emit_body(try_stmts);

                let mut else_sub = self.make_sub();
                else_sub.in_throws = false;
                else_sub.fn_returns_void = false;
                else_sub.known_local_vars.insert("error".to_string());
                else_sub.emit_body(else_stmts);

                // `error` is bound as the original `Box<dyn Error>`, not as a string.
                // • `{error}` in string interpolation works — Box<dyn Error> implements Display.
                // • `match error:` with string patterns works (compare via Display string).
                // • For typed enum dispatch use `try … catch MyEnum:` which emits the
                //   appropriate downcast_ref automatically.
                if self.in_async {
                    format!(
                        "{{\nlet __boring_r: Result<_, Box<dyn std::error::Error + Send + Sync>> = async {{\n{}}}.await;\nmatch __boring_r {{\nOk(__boring_v) => __boring_v,\nErr(__boring_e) => {{\nlet error = __boring_e;\n{}}},\n}}\n}}",
                        try_sub.out,
                        else_sub.out,
                    )
                } else {
                    format!(
                        "{{\nmatch (|| -> Result<_, Box<dyn std::error::Error + Send + Sync>> {{\n{}}})() {{\nOk(__boring_v) => __boring_v,\nErr(__boring_e) => {{\nlet error = __boring_e;\n{}}},\n}}\n}}",
                        try_sub.out,
                        else_sub.out,
                    )
                }
            }

            ExprKind::Else(e, default) => {
                // `x as T else default` — cast with fallback: always use unwrap_or (never ?)
                if let ExprKind::Cast(inner, ty) = &e.kind {
                    let src = self.emit_expr(inner);
                    let dv = self.emit_expr_owned(default);
                    let dst_ty = self.emit_type(ty);
                    return match ty {
                        Type::Float64 => format!("{}.trim().parse::<f64>().unwrap_or({})", src, dv),
                        Type::Float32 => format!("{}.trim().parse::<f32>().unwrap_or({})", src, dv),
                        // Deref before comparing: `Arc<str>`/`Rc<str>` (boring `string`) has no
                        // `PartialEq<&str>` impl, only `PartialEq<Self>` — `&*(src) == "true"`
                        // derefs to a real `&str` first, which does compare against a `&'static
                        // str` literal directly (E0308 otherwise: "expected Arc<str>, found &str").
                        Type::Bool => format!("(&*({}) == \"true\")", src),
                        Type::Named(n) if n == "float" || n == "float64" || n == "f64" =>
                            format!("{}.trim().parse::<f64>().unwrap_or({})", src, dv),
                        Type::Named(n) if n == "float32" || n == "f32" =>
                            format!("{}.trim().parse::<f32>().unwrap_or({})", src, dv),
                        Type::Named(n) if n == "bool" => format!("(&*({}) == \"true\")", src),
                        // `f32`/`f64` used to be excluded here (routed to the generic
                        // `.unwrap_or()` fallback below, with no real parse-and-validate
                        // behavior at all) — now real types with their own arms above,
                        // so this guard is dead for them and only applies to genuinely
                        // "not a specific numeric type" targets.
                        _ if crate::transpiler::helpers::is_specific_numeric_type(&dst_ty) =>
                            format!("{}.trim().parse::<{}>().unwrap_or({})", src, dst_ty, dv),
                        _ => format!("{}.unwrap_or({})", self.emit_expr(e), dv),
                    };
                }
                // `dict[key] else default` — rebuild .get() directly to avoid double-unwrap
                // (ExprKind::Index for dict vars emits .unwrap() for bare access).
                // Uses expr_is_dict (not a plain-Var-only check) so `self.field[key] else d`
                // is recognized too, not just a bare dict variable.
                if let ExprKind::Index(dict_obj, key) = &e.kind {
                    if self.expr_is_dict(dict_obj) {
                        let dict_s = self.emit_expr(dict_obj);
                        let key_ref = self.emit_dict_key_borrow(key);
                        let dv = self.emit_expr_owned(default);
                        return format!("{}.get({}).cloned().unwrap_or_else(|| {})",
                            dict_s, key_ref, dv);
                    }
                }
                // `vec[i] else default` — Vec::get returns Option<&T>, use .cloned().unwrap_or_else.
                // Direct indexing would yield T (not Option<T>), so .unwrap_or_else would fail.
                if let ExprKind::Index(arr_obj, idx_expr) = &e.kind {
                    if !self.expr_is_dict(arr_obj) {
                        let arr_s = self.emit_expr(arr_obj);
                        let idx_s = self.emit_expr(idx_expr);
                        let dv = self.emit_expr_owned(default);
                        return format!("{}.get(({}) as usize).cloned().unwrap_or_else(|| {})",
                            arr_s, idx_s, dv);
                    }
                }
                // `expr else default` — nil coalescing / Option unwrap
                let e_s = self.emit_expr(e);
                let dv = self.emit_expr_owned(default);
                // When a numeric optional (Option<i64/f64>) is coalesced with a string default,
                // unwrap_or_else won't compile — use map_or_else to convert the value to string.
                let is_numeric_opt_var = matches!(&e.kind, ExprKind::Var(v)
                    if self.optional_numeric_vars.contains(v.as_str()));
                // When an always-None optional is coalesced with a string default, emit default directly.
                let is_always_none = matches!(&e.kind, ExprKind::Var(v)
                    if self.always_none_vars.contains(v.as_str()));
                let rc_ty = if self.use_rc_str() { "Rc" } else { "Arc" };
                let dv_is_str = dv.starts_with("Arc::new(") || dv.starts_with("Arc::<str>::from(")
                    || dv.starts_with("Rc::new(") || dv.starts_with("Rc::<str>::from(");
                if is_always_none && dv_is_str {
                    // This optional is always None — the result is always the default value.
                    dv
                } else if is_numeric_opt_var && dv_is_str {
                    format!("{}.as_ref().map_or_else(|| {}, |v| {rc_ty}::<str>::from(format!(\"{{}}\", v)))", e_s, dv)
                } else {
                    format!("{}.unwrap_or_else(|| {})", e_s, dv)
                }
            }

            ExprKind::Array(elems) => {
                // [TypeName] where TypeName is a known struct/enum → typed empty Vec::<T>::new()
                // This handles Boring's `[T]{}` typed-empty-array idiom (parser splits it into
                // `[T]` + `{}`, and the `{}` becomes a harmless HashSet::new() statement).
                if elems.len() == 1 {
                    if let ExprKind::Var(name) = &elems[0].kind {
                        if self.struct_fields.contains_key(name.as_str())
                            || self.enum_variant_fields.keys().any(|k| {
                                k.starts_with(name.as_str()) && k[name.len()..].starts_with("::")
                            })
                        {
                            return format!("Vec::<{}>::new()", name);
                        }
                    }
                }
                // If any element is a string literal, use emit_expr_owned for all
                // so the vec is typed Vec<Rc<str>> consistently.
                let has_str_lit = elems.iter().any(|e| matches!(&e.kind, ExprKind::Str(_) | ExprKind::StringInterp(_)));
                let es: Vec<String> = elems.iter().map(|e| {
                    if has_str_lit { self.emit_expr_owned(e) } else { self.emit_expr(e) }
                }).collect();
                format!("vec![{}]", es.join(", "))
            }
            ExprKind::ArrayFill { value, count } => {
                let v = self.emit_expr_owned(value);
                let n = self.emit_expr(count);
                format!("vec![{}; {} as usize]", v, n)
            }
            ExprKind::ArrayAlloc { count } => {
                let n = self.emit_expr(count);
                format!("vec![Default::default(); {} as usize]", n)
            }
            ExprKind::ArrayComp { expr, var, count } => {
                let n = self.emit_expr(count);
                let body = self.emit_expr(expr);
                // The comprehension's implicit loop var is a bare `int`, which
                // transpiles to `isize` (not `i64`) since this release — a stale
                // `as i64` here left it the one place that never followed, producing
                // a `Vec<isize>` (as declared on the `let`) vs `Vec<i64>` (as
                // collected here) mismatch anywhere the comprehension result binds
                // to an explicitly `int`-typed variable (vector_add_gpu.br's wgpu
                // regression).
                format!("(0..({} as usize)).map(|__boring_i| {{ let {} = __boring_i as isize; {} }}).collect::<Vec<_>>()", n, var, body)
            }
            ExprKind::ArrayCompIter { expr, var, iter } => {
                let it = self.emit_expr(iter);
                let body = self.emit_expr(expr);
                format!("{}.iter().map(|{}| {{ {} }}).collect::<Vec<_>>()", it, var, body)
            }
            ExprKind::Tuple(elems) => {
                // Use emit_expr_owned so string literals become Rc/Arc<str> in tuple slots.
                // Auto-clone non-Copy vars/fields so they remain usable after the tuple is built.
                let es: Vec<String> = elems.iter().map(|e| {
                    let s = self.emit_expr_owned(e);
                    if s.ends_with(".clone()") || s.starts_with('&') || s.starts_with("Arc::") || s.starts_with("Rc::") {
                        return s;
                    }
                    let needs_clone = match &e.kind {
                        ExprKind::Field(..) => true,
                        ExprKind::Var(vname) =>
                            self.var_struct_types.contains_key(vname.as_str())
                            || self.collection_vars.contains(vname.as_str())
                            || self.vec_vars.contains(vname.as_str())
                            || self.string_arc_vars.contains(vname.as_str())
                            || matches!(
                                self.fn_current_params.get(vname.as_str()).or_else(|| self.var_types.get(vname.as_str())),
                                Some(Type::Named(_) | Type::Array(_) | Type::Dict(..) | Type::Set(_))
                            ),
                        _ => false,
                    };
                    if needs_clone { format!("{}.clone()", s) } else { s }
                }).collect();
                format!("({})", es.join(", "))
            }
            ExprKind::Dict(pairs) => {
                if pairs.is_empty() {
                    "HashMap::new()".into()
                } else {
                    // Use emit_expr_owned for both keys and values so string literals
                    // become Arc<str> (string dicts are HashMap<Arc<str>, Arc<str>>).
                    let ps: Vec<String> = pairs.iter()
                        .map(|(k, v)| format!("({}, {})", self.emit_expr_owned(k), self.emit_expr_owned(v)))
                        .collect();
                    format!("HashMap::from([{}])", ps.join(", "))
                }
            }
            ExprKind::Set(elems) => {
                if elems.is_empty() {
                    // Provide a default element type so Rust can infer the HashSet type.
                    "HashSet::<isize>::new()".into()
                } else {
                    let es: Vec<String> = elems.iter().map(|e| self.emit_expr(e)).collect();
                    format!("HashSet::from([{}])", es.join(", "))
                }
            }

            ExprKind::DotIdent(name) => {
                // Enum variant shorthand: `.North` → `Direction::North`
                if let Some(enum_name) = self.enum_variants.get(name) {
                    format!("{}::{}", enum_name, name)
                } else {
                    name.clone() // unknown variant — emit bare name, will be caught by rustc
                }
            }
            ExprKind::Range { start, end, inclusive } => {
                let s = self.emit_expr(start);
                let e = self.emit_expr(end);
                if *inclusive { format!("({}..={})", s, e) } else { format!("({}..{})", s, e) }
            }
            ExprKind::Cast(e, ty) => self.emit_expr_cast(e, ty),

            ExprKind::OptionalField(obj, field) => {
                // Use .clone() so the result is Option<T> (owned), not Option<&T>.
                // This makes nil-coalescing (`?.field else default`) work with matching types.
                let obj_s = self.emit_expr(obj);
                let shadow_name = if let ExprKind::Var(v) = &obj.kind {
                    self.managed_param_shadows.get(v.as_str()).cloned()
                } else { None };
                let is_managed_mutex = shadow_name.is_none() && if let ExprKind::Var(v) = &obj.kind {
                    self.managed_mutex_vars.contains(v.as_str())
                } else { false };
                let is_managed_refcell = if let ExprKind::Var(v) = &obj.kind {
                    self.managed_refcell_vars.contains(v.as_str())
                } else { false };
                if let Some(shadow) = shadow_name {
                    format!("{}.as_ref().map(|__v| __v.{}.clone())", shadow, field)
                } else if is_managed_mutex {
                    // Arc<std::sync::Mutex<T>> — use .lock().unwrap()
                    format!("{}.as_ref().map(|__v| __v.lock().unwrap().{}.clone())", obj_s, field)
                } else if is_managed_refcell {
                    // RefCell<T> — use .borrow()
                    format!("{}.as_ref().map(|__v| __v.borrow().{}.clone())", obj_s, field)
                } else {
                    format!("{}.as_ref().map(|__v| __v.{}.clone())", obj_s, field)
                }
            }
            ExprKind::OptionalMethodCall(obj, method, args) => {
                // Use emit_expr_owned so string literals are coerced to Arc<str> (not &str).
                // Without this, opt?.push("hello") would pass &str where Arc<str> is expected.
                let args_s: Vec<String> = args.iter().map(|a| self.emit_expr_owned(&a.value)).collect();
                let obj_s = self.emit_expr(obj);
                let shadow_name_mc = if let ExprKind::Var(v) = &obj.kind {
                    self.managed_param_shadows.get(v.as_str()).cloned()
                } else { None };
                let is_managed_mutex = shadow_name_mc.is_none() && if let ExprKind::Var(v) = &obj.kind {
                    self.managed_mutex_vars.contains(v.as_str())
                } else { false };
                let is_managed_refcell = if let ExprKind::Var(v) = &obj.kind {
                    self.managed_refcell_vars.contains(v.as_str())
                } else { false };
                if let Some(shadow) = shadow_name_mc {
                    format!("{}.clone().map(|mut __v| __v.{}({}))", shadow, method, args_s.join(", "))
                } else if is_managed_mutex {
                    // Arc<std::sync::Mutex<T>> — use .lock().unwrap()
                    format!("{}.clone().map(|__v| __v.lock().unwrap().{}({}))", obj_s, method, args_s.join(", "))
                } else if is_managed_refcell {
                    // RefCell<T> — use .borrow_mut() for method calls
                    format!("{}.clone().map(|__v| __v.borrow_mut().{}({}))", obj_s, method, args_s.join(", "))
                } else {
                    // Use .clone().map(|mut __v| ...) so that &mut self methods can be called.
                    // Cloning Option<Box<T>> gives an owned value, and `mut __v` allows &mut deref.
                    format!("{}.clone().map(|mut __v| __v.{}({}))", obj_s, method, args_s.join(", "))
                }
            }

            ExprKind::Closure(params, _ret_ty, body, throws, task) =>
                self.emit_expr_closure(params, body, *throws, *task),

            ExprKind::If(s) => {
                // When the if-expression is used in an Optional context (fn_return_ty = Optional(T)),
                // OR when the if-expression itself has a nil branch (making it implicitly Optional),
                // non-nil branches must be wrapped in Some(...) and nil branches emit None.
                fn branch_has_nil(body: &[crate::ast::Stmt]) -> bool {
                    matches!(body.last(), Some(crate::ast::Stmt::Expr(e)) if matches!(&e.kind, ExprKind::Nil))
                }
                let has_nil_branch = s.branches.iter().any(|(_, b)| branch_has_nil(b))
                    || s.else_body.as_ref().map(|b| branch_has_nil(b)).unwrap_or(false);
                // Only apply Optional wrapping when there is an explicit nil branch.
                // The Stmt::Expr tail handler wraps the whole if/else in Some() when
                // needed — we must NOT apply fn_return_ty here or we double-wrap when
                // the if/else is used as an intermediate let-binding value.
                let optional_inner = if has_nil_branch {
                    Some(crate::ast::Type::Named("_".to_string()))
                } else {
                    None
                };
                let mut sub = self.make_sub();
                sub.fn_returns_void = false;
                sub.suppress_ok_wrap = true;
                sub.fn_return_ty = None; // prevent spurious Some() wrapping in branch bodies
                let emit_branch = |sub: &mut Self, body: &[crate::ast::Stmt]| {
                    if optional_inner.is_some() {
                        sub.emit_body_optional_last(body);
                    } else {
                        sub.emit_body(body);
                    }
                };
                for (i, (cond, body)) in s.branches.iter().enumerate() {
                    let kw = if i == 0 { "if" } else { "} else if" };
                    let cond_s = sub.emit_expr(cond);
                    sub.line(&format!("{} {} {{", kw, cond_s));
                    sub.indent += 1;
                    emit_branch(&mut sub, body);
                    sub.indent -= 1;
                }
                if let Some(else_body) = &s.else_body {
                    sub.line("} else {");
                    sub.indent += 1;
                    emit_branch(&mut sub, else_body);
                    sub.indent -= 1;
                }
                sub.line("}");
                format!("{{\n{}}}", sub.out)
            }
            ExprKind::Match(s) => {
                let mut sub = self.make_sub();
                // Match used as an expression — arms must return values, never add `;`.
                sub.fn_returns_void = false;
                sub.suppress_ok_wrap = true; // prevent Ok() wrapping; keep ?-propagation
                sub.fn_return_ty = None; // prevent spurious Some() wrapping in arm bodies
                sub.emit_match(s, true);
                sub.out.trim_end().to_string()
            }
            ExprKind::Block(stmts) => {
                let inner: Vec<String> = stmts.iter().map(|s| self.emit_stmt_inline(s)).collect();
                format!("{{ {} }}", inner.join(" "))
            }
            ExprKind::Do(stmts) => {
                // `do:` block — emit as a proper block using a sub-emitter so that
                // complex statements (for loops, if, etc.) are rendered correctly.
                // Do not inherit `in_throws` from the parent: the block's last expression
                // is not a Result — it's the block's value, not a function return.
                let mut sub = self.make_sub();
                sub.in_throws = false;
                sub.emit_body(stmts);
                format!("{{\n{}}}", sub.out)
            }
            ExprKind::Loop(s) => {
                // Use a sub-emitter so each statement in the body gets proper semicolons/formatting.
                let mut sub = self.make_sub();
                sub.emit_loop(s);
                sub.out.trim_end().to_string()
            }
            ExprKind::TaskWithTimeout(dur_expr, body_expr) => self.emit_expr_task_with_timeout(dur_expr, body_expr),

            ExprKind::Task(e) => self.emit_expr_task(expr, e),
            ExprKind::JoinAll(handles) => {
                // Standalone `join [f1, f2]` — emit tokio::join! directly
                let exprs: Vec<String> = handles.iter().map(|e| self.emit_expr(e)).collect();
                format!("tokio::join!({})", exprs.join(", "))
            }
            ExprKind::MacroCall { name, args } => self.emit_macro(name, args),
            ExprKind::SliceRange { .. } => {
                panic!("SliceRange cannot appear outside an index expression")
            }
            // Lowered away by the labeled-array desugar pass before codegen ever runs
            // (see docs/array-multidim-proposal.md) — reaching one here means that
            // pass was skipped, an internal compiler bug.
            ExprKind::LabeledIndex(..) | ExprKind::LabeledArrayComp { .. } | ExprKind::RelabelCast(..) => {
                panic!("labeled multi-dim array expression reached codegen without being desugared first")
            }
        }
    }

    /// `task(duration): body` — `tokio::time::timeout(dur, async move { body })`, spawned
    /// if inside an async context or emitted inline otherwise. The `Elapsed` error propagates
    /// via `?`, catchable by `try task(dur): body else: …` or a bare `catch:`.
    fn emit_expr_task_with_timeout(&self, dur_expr: &Expr, body_expr: &Expr) -> String {
        // Resolve leading-dot: `.fromSecs(5)` → `Duration::from_secs(5)`.
        let is_instant_dur = expr_is_instant(dur_expr, &self.instant_vars);
        let dur_type_prefix = if is_instant_dur { "Instant" } else { "Duration" };
        let dur_s = self.resolve_dot_with_type(dur_expr, dur_type_prefix)
            .unwrap_or_else(|| self.emit_expr(dur_expr));
        let captured = collect_var_names(body_expr);
        let arc_captures: Vec<&str> = captured.iter()
            .filter(|v| self.arc_vars.contains(*v))
            .map(String::as_str)
            .collect();

        let inner_s = if let ExprKind::Block(stmts) = &body_expr.kind {
            let mut sub = self.make_sub();
            sub.in_async = true;
            sub.in_throws = false;
            sub.emit_body(stmts);
            format!("{{\n{}}}", sub.out)
        } else {
            let mut sub = self.make_sub();
            sub.in_async = true;
            sub.in_throws = false;
            format!("{{ {} }}", sub.emit_expr(body_expr))
        };

        let clone_prefix = if arc_captures.is_empty() {
            String::new()
        } else {
            arc_captures.iter()
                .map(|v| {
                    if self.rc_vars.contains(*v) {
                        format!("let {} = Rc::clone(&{});", v, v)
                    } else {
                        format!("let {} = Arc::clone(&{});", v, v)
                    }
                })
                .collect::<Vec<_>>()
                .join(" ") + " "
        };

        // tokio::time::timeout wraps the body future; Elapsed propagates via ?
        let timeout_fn = if expr_is_instant(dur_expr, &self.instant_vars) {
            "timeout_at"
        } else {
            "timeout"
        };
        let timeout_future = format!(
            "{}async move {{ tokio::time::{}({}, async move {}).await? }}",
            clone_prefix, timeout_fn, dur_s, inner_s
        );

        // Spawn if inside an async context (produces a JoinHandle),
        // otherwise emit as an inline future expression.
        if self.in_async {
            // Mark the spawn as a throws JoinHandle so .value uses the ? unwrap
            let spawn_fn = match self.config.threading {
                crate::transpiler::ThreadingMode::Single => "tokio::task::spawn_local",
                crate::transpiler::ThreadingMode::Multi  => "tokio::spawn",
            };
            format!("{}({})", spawn_fn, timeout_future)
        } else {
            timeout_future
        }
    }

    /// `task expr` — auto-detects `tokio::spawn` (async body) vs. `tokio::task::spawn_blocking`
    /// (sync/CPU-bound body), cloning any captured `Arc<T>` vars into the spawned closure so the
    /// outer bindings remain valid. `expr` (the whole node) is only used for warning line/col.
    fn emit_expr_task(&self, expr: &Expr, e: &Expr) -> String {
        // Auto-detect whether to use tokio::spawn (async) or
        // tokio::task::spawn_blocking (sync/CPU-bound):
        //   task asyncFn(args)  — asyncFn ∈ task_fns  → tokio::spawn
        //   task syncFn(args)   — syncFn ∉ task_fns   → spawn_blocking
        //   task: { async body }                       → tokio::spawn
        //   task: { sync body }  (no await/task)       → spawn_blocking
        let blocking = is_blocking_spawn(e, &self.task_fns);

        // Arc<T> variables captured by the task body must be cloned so the outer
        // binding remains valid after the spawn (tokio::spawn moves its captures).
        let captured = collect_var_names(e);
        let arc_captures: Vec<&str> = captured.iter()
            .filter(|v| self.arc_vars.contains(*v))
            .map(String::as_str)
            .collect();

        // Build the inner body string.
        // For blocking tasks: no `async`, no `.await` on calls (in_async = false).
        // For async tasks:    standard async sub-emitter (in_async = true).
        let inner_s = if let ExprKind::Block(stmts) = &e.kind {
            let mut sub = self.make_sub();
            sub.in_async = !blocking;
            sub.in_throws = false;
            sub.emit_body(stmts);
            format!("{{\n{}}}", sub.out)
        } else {
            let mut sub = self.make_sub();
            sub.in_async = !blocking;
            sub.in_throws = false;
            format!("{{ {} }}", sub.emit_expr(e))
        };

        let clones: String = arc_captures.iter()
            .map(|v| {
                if self.rc_vars.contains(*v) {
                    format!("let {} = Rc::clone(&{});", v, v)
                } else {
                    format!("let {} = Arc::clone(&{});", v, v)
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        // !Send warning: spawn_local captures Rc vars in single-thread mode.
        if !blocking && matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
            for v in &arc_captures {
                if self.rc_vars.contains(*v) {
                    self.push_warning(expr.line, expr.col, format!("`spawn_local` captures `{}` which is Rc<T> (a !Send type); Rc values cannot be sent across task boundaries", v));
                }
            }
        }

        if blocking {
            // Synchronous closure — tokio::task::spawn_blocking(move || { body })
            let closure = if arc_captures.is_empty() {
                format!("move || {}", inner_s)
            } else {
                format!("{{ {} move || {} }}", clones, inner_s)
            };
            format!("tokio::task::spawn_blocking({})", closure)
        } else {
            // Asynchronous closure — spawn_local (single) or tokio::spawn (multi)
            let spawn_fn = match self.config.threading {
                crate::transpiler::ThreadingMode::Single => "tokio::task::spawn_local",
                crate::transpiler::ThreadingMode::Multi  => "tokio::spawn",
            };
            if arc_captures.is_empty() {
                if self.in_async {
                    format!("{}(async move {})", spawn_fn, inner_s)
                } else {
                    format!("async move {}", inner_s)
                }
            } else if self.in_async {
                format!("{}({{ {} async move {} }})", spawn_fn, clones, inner_s)
            } else {
                format!("{{ {} async move {} }}", clones, inner_s)
            }
        }
    }

    /// `(params): body` — plain closures and `task` closures (wrapped in `async move`,
    /// with Arc captures pre-cloned so an `FnMut` closure can be invoked more than once).
    fn emit_expr_closure(&self, params: &[Param], body: &ClosureBody, throws: bool, task: bool) -> String {
        let ps: Vec<String> = params.iter().map(|p| {
            let name = if p.mutable { format!("mut {}", p.name) } else { p.name.clone() };
            if let Some(ty) = &p.ty {
                format!("{}: {}", name, self.emit_type(ty))
            } else {
                name
            }
        }).collect();
        // Emit the closure body with params registered as known locals so that
        // `param.method()` doesn't get misread as a module path `param::method()`.
        let mut sub = self.make_sub();
        for p in params.iter() {
            sub.known_local_vars.insert(p.name.clone());
            // Remove any outer-scope var_struct_types entry for this param name.
            // Without this, a closure param `p` would inherit the type of an outer
            // variable named `p` (e.g. `p: Parrot`), causing field accesses like
            // `p.name` to be incorrectly emitted as getter calls `p.name()`.
            sub.var_struct_types.remove(&p.name);
        }
        // `task` closures: wrap body in `async move { ... }` so they return a Future.
        // `throws` closures: wrap return value in Ok(...).
        if task {
            // When the body is a bare `task: ...` expression whose body references Arc
            // variables from the outer scope, those Arcs would be moved by `async move`.
            // This is fine for a FnOnce closure but breaks FnMut (e.g. `.map(|x| ...)`)
            // because the same Arc can't be moved on every call.
            //
            // Fix: detect Arc captures in the task body and pre-clone them at the start
            // of the sync closure body so each invocation creates fresh owned clones
            // before the `async move` takes them.
            let param_names: std::collections::HashSet<&str> =
                params.iter().map(|p| p.name.as_str()).collect();
            // Collect Arc captures from the task body — works for both:
            //   ClosureBody::Expr(ExprKind::Task(inner))       — single-line `(x): task: expr`
            //   ClosureBody::Block([Stmt::Expr(ExprKind::Task(inner)), ...])  — multiline
            let task_inner_expr: Option<&Expr> = match body {
                ClosureBody::Expr(e) => {
                    if let ExprKind::Task(inner) = &e.kind { Some(inner.as_ref()) } else { None }
                }
                ClosureBody::Block(stmts) => {
                    // Last statement may be the task expression.
                    stmts.last().and_then(|s| match s {
                        Stmt::Expr(e) | Stmt::Return(ReturnStmt { value: Some(e), .. }) => {
                            if let ExprKind::Task(inner) = &e.kind { Some(inner.as_ref()) } else { None }
                        }
                        _ => None,
                    })
                }
            };
            let pre_clones: String = if let Some(inner) = task_inner_expr {
                let captured = collect_var_names(inner);
                let arc_caps: Vec<String> = captured.iter()
                    .filter(|v| {
                        (sub.arc_vars.contains(*v) || sub.string_arc_vars.contains(*v))
                            && !param_names.contains(v.as_str())
                    })
                    .map(|v| {
                        if sub.rc_vars.contains(v.as_str()) {
                            format!("let {} = Rc::clone(&{});", v, v)
                        } else {
                            format!("let {} = Arc::clone(&{});", v, v)
                        }
                    })
                    .collect();
                arc_caps.join(" ")
            } else {
                String::new()
            };

            let body_s = match body {
                ClosureBody::Expr(e) => {
                    let val = sub.emit_expr(e);
                    if throws { format!("Ok({})", val) } else { val }
                }
                ClosureBody::Block(stmts) => {
                    sub.fn_returns_void = false;
                    sub.in_throws = throws;
                    let n = stmts.len();
                    let inner: Vec<String> = stmts.iter().enumerate().map(|(i, s)| {
                        if i + 1 == n {
                            sub.emit_stmt_inline(s)
                        } else {
                            format!("{};", sub.emit_stmt_inline(s))
                        }
                    }).collect();
                    inner.join(" ")
                }
            };
            return if pre_clones.is_empty() {
                format!("|{}| async move {{ {} }}", ps.join(", "), body_s)
            } else {
                // Pre-clone Arcs in a sync wrapper block, then return the async future.
                format!("|{}| {{ {} async move {{ {} }} }}", ps.join(", "), pre_clones, body_s)
            };
        }
        match body {
            ClosureBody::Expr(e) => {
                let val = sub.emit_expr(e);
                if throws {
                    format!("|{}| Ok({})", ps.join(", "), val)
                } else {
                    format!("|{}| {}", ps.join(", "), val)
                }
            }
            ClosureBody::Block(stmts) => {
                // Closure blocks: the last statement should be a value expression.
                // Clear in_throws and fn_returns_void so if/match branches emit
                // values without Ok()-wrapping or trailing semicolons.
                sub.fn_returns_void = false;
                sub.in_throws = false;
                let n = stmts.len();
                let inner: Vec<String> = stmts.iter().enumerate().map(|(i, s)| {
                    if i + 1 == n {
                        // Last stmt: emit as value (if/match need is_last=true).
                        match s {
                            Stmt::If(if_s) => {
                                let prev_out = std::mem::take(&mut sub.out);
                                sub.emit_if(if_s, true);
                                let result = std::mem::replace(&mut sub.out, prev_out);
                                result.trim_end_matches('\n').to_string()
                            }
                            Stmt::Match(m_s) => {
                                let prev_out = std::mem::take(&mut sub.out);
                                sub.emit_match(m_s, true);
                                let result = std::mem::replace(&mut sub.out, prev_out);
                                result.trim_end_matches('\n').to_string()
                            }
                            _ => sub.emit_stmt_inline(s),
                        }
                    } else {
                        let v = sub.emit_stmt_inline(s);
                        format!("{};", v)
                    }
                }).collect();
                format!("|{}| {{ {} }}", ps.join(", "), inner.join(" "))
            }
        }
    }

    /// `e as ty` — user-defined `into_type()` conversions, newtype wrap/unwrap,
    /// Optional-type parse, numeric/bool/string coercions, and the `'actor` wrap.
    fn emit_expr_cast(&self, e: &Expr, ty: &Type) -> String {
        let src = self.emit_expr(e);
        let dst = self.emit_type(ty);
        // User-defined `as Type:` conversion → call the generated `into_type()` method.
        // Use the lowercased emitted type name for primitive types (float → f64, etc.)
        // as well as named types. Only apply if the source is a struct/enum variable —
        // do not transform string/numeric literal casts.
        let src_is_struct_or_enum = match &e.kind {
            ExprKind::Var(v) =>
                self.var_struct_types.contains_key(v.as_str())
                || self.var_struct_type.contains_key(v.as_str())
                || matches!(self.var_types.get(v.as_str()),
                    Some(Type::Named(n)) if self.struct_fields.contains_key(n.as_str())),
            ExprKind::Field(_, _) => true, // field access on struct
            _ => false,
        };
        let key = match ty {
            Type::Named(n) => Some(n.to_lowercase()),
            _ if src_is_struct_or_enum => Some(dst.to_lowercase()),
            _ => None,
        };
        // Never route `as string` through user_conv_targets — the Display impl's
        // Arc::<str>::from(x.to_string()) path handles it correctly without generating
        // a method name like `into_arc<string>` which is invalid Rust.
        let is_string_cast = matches!(ty, Type::Str)
            || matches!(ty, Type::Named(n) if n == "string" || n == "str");
        if !is_string_cast {
            if let Some(k) = key {
                // Try both the boring type name (e.g. "float") and the Rust type name (e.g. "f64").
                // user_conv_targets stores the lowercased Rust emit form, but the key from
                // Type::Named("float") is "float". Try the boring name first, then the emitted form.
                // Only apply user conversions when the source is a struct/enum instance —
                // don't call into_f64() on numeric expressions, only on struct variables/fields.
                if self.user_conv_targets.contains(k.as_str()) {
                    let method = format!("into_{}", k);
                    return format!("{}.{}()", src, method);
                } else if src_is_struct_or_enum {
                    let rust_key = dst.to_lowercase();
                    if k != rust_key && self.user_conv_targets.contains(rust_key.as_str()) {
                        let method = format!("into_{}", rust_key);
                        return format!("{}.{}()", src, method);
                    }
                }
            }
        }
        // Newtype unwrap: `id as uint` where `id` is a known newtype variable → `id.0`.
        // Works for let bindings and function parameters tracked in var_newtype_type.
        if let ExprKind::Var(v) = &e.kind {
            if let Some(nt_name) = self.var_newtype_type.get(v.as_str()) {
                if let Some(inner_rust) = self.newtype_inner.get(nt_name.as_str()) {
                    if *inner_rust == dst {
                        return format!("{}.0", src);
                    }
                }
            }
        }
        // Newtype construction: `42 as UserId` → `UserId(42)`.
        if let Type::Named(n) = ty {
            if self.newtype_types.contains(n.as_str()) {
                return format!("{}({})", n, src);
            }
        }
        // Cast to Optional type: `s as int?` → parse().ok(), not unwrap_or.
        if let Type::Optional(inner) = ty {
            let inner_dst = self.emit_type(inner);
            let parse_ty = if matches!(inner.as_ref(), Type::Float64) || matches!(inner.as_ref(), Type::Named(n) if n == "float" || n == "float64" || n == "f64") {
                Some("f64".to_string())
            } else if matches!(inner.as_ref(), Type::Float32) || matches!(inner.as_ref(), Type::Named(n) if n == "float32" || n == "f32") {
                Some("f32".to_string())
            } else if crate::transpiler::helpers::is_specific_numeric_type(&inner_dst) {
                Some(inner_dst)
            } else {
                None
            };
            return if let Some(pt) = parse_ty {
                format!("{}.trim().parse::<{}>().ok()", src, pt)
            } else {
                format!("{}.try_into().ok()", src)
            };
        }
        let src_is_numeric_lit = matches!(&e.kind, ExprKind::Int(_) | ExprKind::Float(_));
        // A decimal literal that overflows `i64` but fits `u64` (see
        // ExprKind::UInt64). `emit_expr` already suffixes it as
        // `NNNNu64`, so — unlike the `ExprKind::Int` literal branch below — it must NOT
        // get an additional `i64` suffix tacked on here.
        let src_is_big_uint_lit = matches!(&e.kind, ExprKind::UInt64(_));
        let _src_is_bool = matches!(&e.kind, ExprKind::Bool(_))
            || matches!(&e.kind, ExprKind::Var(v) if {
                // bool variable (rough heuristic: not in known numeric vars)
                let _ = v; false
            });
        let src_is_bool_lit = matches!(&e.kind, ExprKind::Bool(_));
        let src_is_numeric_var = matches!(&e.kind, ExprKind::Var(v)
            if !self.string_vars.contains(v.as_str()));
        // Covers both float32 and float64 — the emitted cast always uses `dst`
        // (`self.emit_type(ty)`, "f32" or "f64" as appropriate), never a hardcoded
        // "f64", so this one flag serves both widths correctly.
        let is_float_ty = matches!(ty, Type::Float64 | Type::Float32)
            || matches!(ty, Type::Named(n) if matches!(n.as_str(), "float" | "float64" | "float32" | "f32" | "f64"));
        // Every fixed-width/pointer-width integer target (isize/usize/u8/i8/../u128/i128)
        // resolves through `dst` (== self.emit_type(ty)), which already normalizes both
        // the dedicated Type variants and the lowercase `Type::Named` aliases — so this
        // check covers all 12 numeric kinds without hand-listing each one. Kept separate
        // from is_float_ty: floats are handled by the flag above instead, since two
        // distinct widths ("f32" vs "f64") both need to flow through unchanged as `dst`.
        let is_fixed_int_ty = crate::transpiler::helpers::is_specific_numeric_type(&dst)
            && dst != "f32" && dst != "f64";
        let is_bool_ty = matches!(ty, Type::Bool)
            || matches!(ty, Type::Named(n) if n == "bool");

        // Numeric computation (BinOp/Call/UnaryOp) → numeric cast: use `as T`, not .parse()
        // A `MethodCall` is only a numeric computation when the method itself doesn't
        // return a string — `id_input.trim() as uint` must still fall through to the
        // string→int `.trim().parse::<T>()` path further below (`STRING_RETURNING_METHODS`
        // below never exist on a numeric receiver, so excluding them here is unambiguous;
        // an actually-numeric method call like `arr.len() as uint` is unaffected).
        const STRING_RETURNING_METHODS: &[&str] = &[
            "trim", "trimStart", "trimEnd",
            "upper", "toUpper", "toUpperCase", "uppercased",
            "lower", "toLower", "toLowerCase", "lowercased",
            "replace",
        ];
        let is_string_returning_method_call = matches!(&e.kind,
            ExprKind::MethodCall(_, m, _) if STRING_RETURNING_METHODS.contains(&m.as_str()));
        let src_is_expr = !is_string_returning_method_call && matches!(&e.kind,
            ExprKind::BinOp(_, _, _) | ExprKind::Call(_, _) | ExprKind::UnaryOp(_, _)
            | ExprKind::MethodCall(_, _, _));
        if src_is_expr && is_float_ty {
            return format!("({} as {})", src, dst);
        }
        if src_is_expr && is_fixed_int_ty {
            return format!("({} as {})", src, dst);
        }
        // Known-numeric variable (tracked in var_types as Int/Float/Uint/...) → cast with `as T`
        let src_var_is_numeric = matches!(&e.kind, ExprKind::Var(v) if {
            self.var_types.get(v.as_str()).map(is_known_numeric_scalar_type).unwrap_or(false)
        });
        if src_var_is_numeric && is_float_ty {
            return format!("({} as {})", src, dst);
        }
        if src_var_is_numeric && is_fixed_int_ty {
            return format!("({} as {})", src, dst);
        }
        // Indexing into a known-numeric-element array/dict (`bytes[16] as uint32`,
        // `scores[k] as uint32`) → same numeric path as the bare-variable case above.
        // `ExprKind::Index` isn't a Var/BinOp/Call/UnaryOp/MethodCall, so without this
        // check it falls through every branch above and every branch below all the way
        // to the string-parsing fallback near the bottom of this function, which wrongly
        // emits `.trim().parse::<T>()` on an already-numeric element — that fails to
        // compile with `no method named 'trim' found for type 'u8'` (or whatever the
        // element type is) since `.trim()` only exists on strings.
        let src_is_numeric_index = matches!(&e.kind, ExprKind::Index(base, _) if match &base.kind {
            ExprKind::Var(v) => match self.var_types.get(v.as_str()) {
                Some(Type::Array(elem)) | Some(Type::ArrayN(elem, _)) | Some(Type::Dict(_, elem)) => {
                    let elem = match elem.as_ref() {
                        // `[mut uint8]`/`{K = mut V}` — unwrap the permission marker to
                        // get at the underlying numeric type.
                        Type::Mut(inner) => inner.as_ref(),
                        other => other,
                    };
                    is_known_numeric_scalar_type(elem)
                }
                _ => false,
            },
            _ => false,
        });
        if src_is_numeric_index && is_float_ty {
            return format!("({} as {})", src, dst);
        }
        if src_is_numeric_index && is_fixed_int_ty {
            return format!("({} as {})", src, dst);
        }

        // Oversized-decimal literal (already suffixed `u64` by emit_expr) → numeric target:
        // plain `as` cast, no extra literal suffix (it already carries one).
        if src_is_big_uint_lit && (is_float_ty || is_fixed_int_ty) {
            return format!("({} as {})", src, dst);
        }
        if src_is_big_uint_lit && is_bool_ty {
            return "None".into();
        }
        // bool → int: direct cast (true=1, false=0), always succeeds
        if src_is_bool_lit && is_fixed_int_ty {
            return format!("({} as {})", src, dst);
        }
        // int/float literal → float: use `as f64`, not .parse()
        if src_is_numeric_lit && is_float_ty {
            return format!("({} as {})", src, dst);
        }
        if src_is_numeric_lit && is_fixed_int_ty {
            // Suffix the literal (`300i64`, not bare `300`) before the `as` cast.
            // A *bare* integer literal cast directly to a narrower type (`300 as u8`)
            // trips rustc's `overflowing_literals` lint (deny-by-default) whenever the
            // literal provably doesn't fit — even though Boring's own cast semantics
            // want a normal truncating/wrapping runtime cast here, same as it would be
            // for any other numeric source. The explicit suffix makes the literal's own
            // type `i64` (every `ExprKind::Int` literal fits by construction —
            // anything bigger lexes as `ExprKind::UInt64` instead, handled by the
            // `src_is_big_uint_lit` branch above), so the subsequent `as` is an
            // ordinary (non-literal) cast.
            return format!("({}i64 as {})", src, dst);
        }
        // numeric literal → bool: always nil (int-to-bool not meaningful in Boring)
        if src_is_numeric_lit && is_bool_ty {
            return "None".into();
        }
        // Non-string (numeric var) → float/int: use `as T` cast, not .parse()
        if src_is_numeric_var && is_float_ty {
            return format!("({} as {})", src, dst);
        }
        if src_is_numeric_var && is_fixed_int_ty {
            return format!("({} as {})", src, dst);
        }
        // Non-string (numeric var) → bool: None (invalid cast)
        if src_is_numeric_var && is_bool_ty {
            return "None".into();
        }

        // T as T'actor → Rc::new(RefCell::new(src)) in single-thread mode,
        // Arc::new(Mutex::new(src)) or Arc::new(tokio::sync::Mutex::new(src)) in multi-thread mode.
        if matches!(ty, crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Actor)) {
            return self.emit_actor_new(&src);
        }

        // string → int/uint/float: use `?` only inside an explicit `try:` body,
        // never just because the enclosing function returns Result.
        // This keeps `"42" as int` producing Option<isize> in normal code.
        if is_fixed_int_ty || is_float_ty {
            let parse_ty = if is_float_ty { "f64" } else { dst.as_str() };
            return if self.in_try_body {
                format!("{}.trim().parse::<{}>()?", src, parse_ty)
            } else {
                format!("{}.trim().parse::<{}>().ok()", src, parse_ty)
            };
        }
        match ty {
            // string → bool: equality check. Deref first — see the identical `as bool
            // else default` branch above (`ExprKind::Else` arm) for why: `Arc<str>`/
            // `Rc<str>` has no `PartialEq<&str>` impl, only `PartialEq<Self>`.
            Type::Bool => format!("(&*({}) == \"true\")", src),
            Type::Named(n) if n == "bool" => format!("(&*({}) == \"true\")", src),
            // numeric/value → string
            Type::Str => self.str_from_expr(&format!("{}.to_string()", src)),
            Type::Named(n) if n == "string" => self.str_from_expr(&format!("{}.to_string()", src)),
            // everything else: primitive Rust cast
            _ => format!("({} as {})", src, dst),
        }
    }

    /// Is `expr` a source expression that already has its own correct
    /// Option-producing cast codegen in `emit_expr_cast` (string → `.parse().ok()`,
    /// bool → `None`) — i.e. NOT the plain-numeric case that `emit_expr_cast`
    /// emits as an unconditional infallible `(src as dst)` and that
    /// `try_emit_checked_int_cast_as_option` (below) needs to intercept instead.
    fn cast_source_is_stringy_or_bool(&self, inner: &Expr) -> bool {
        match &inner.kind {
            ExprKind::Str(_) | ExprKind::StringInterp(_) | ExprKind::Bool(_) => true,
            ExprKind::Var(v) => {
                self.string_vars.contains(v.as_str())
                    || matches!(self.var_types.get(v.as_str()), Some(Type::Str))
                    || matches!(self.var_types.get(v.as_str()), Some(Type::Named(n)) if n == "string" || n == "str")
            }
            _ => false,
        }
    }

    /// Checked-narrowing-cast-as-`Option` codegen for the scrutinee of an
    /// `if let`/`guard let` binding clause (see `emit_cond_clauses` and
    /// `emit_guard`'s `CondClause::Let` arms, the only two call sites).
    ///
    /// A two-branch `if let`/`guard let` clause always emits its pattern as
    /// `Some(name)` (both call sites hard-code that), which structurally
    /// requires the scrutinee to actually be an `Option<T>` at the Rust level.
    /// For most expression shapes that's already true (dict lookups, `T?`-
    /// returning calls, etc.) — but a numeric `as` cast used as a bare
    /// (non-`if let`) expression is deliberately emitted as an unconditional,
    /// infallible `(src as dst)` by `emit_expr_cast` (that infallible emission
    /// for a plain numeric-to-integer cast is pre-existing/unchanged
    /// behavior, not itself being "fixed" here), which doesn't type-check against the
    /// `Some(...)` pattern. This intercepts exactly that one shape — a numeric
    /// source cast to a fixed-width/pointer-width integer target — and emits
    /// Rust's standard `TryFrom` conversion instead, which is implemented for
    /// every pair of the twelve integer types (narrowing, widening, or
    /// same-width), so it isn't limited to the `int128 -> int64` pair the bug
    /// was originally found with.
    ///
    /// Returns `None` for every other shape (string/bool source, non-integer
    /// target, or not a cast at all) so the caller falls back to the ordinary
    /// `emit_expr`, which already produces correct `Option`-typed codegen for
    /// those (`.parse().ok()`, literal `None`, or whatever the non-cast
    /// expression's own codegen already does).
    pub(crate) fn try_emit_checked_int_cast_as_option(&self, expr: &Expr) -> Option<String> {
        let ExprKind::Cast(inner, ty) = &expr.kind else { return None };
        let dst = self.emit_type(ty);
        let is_integer_target = crate::transpiler::helpers::is_specific_numeric_type(&dst)
            && dst != "f32" && dst != "f64";
        if !is_integer_target || self.cast_source_is_stringy_or_bool(inner) {
            return None;
        }
        let src = self.emit_expr(inner);
        Some(format!("{}::try_from({}).ok()", dst, src))
    }

    /// `obj[idx]` — slice ranges, char-safe string indexing, opaque collection-index
    /// vars (`get_at`), dict/HashMap access (including `self.field[key]`), and plain
    /// array indexing (with `.clone()` unless this is itself an assignment target).
    fn emit_expr_index(&self, obj: &Expr, idx: &Expr) -> String {
        // Slice: a[M..N], a[..N], a[M..], a[..]  →  obj[M..N].to_vec()
        if let ExprKind::SliceRange { start, end, inclusive } = &idx.kind {
            // Detect whether the receiver is a string to emit a char-safe slice.
            let is_str = match &obj.kind {
                ExprKind::Str(_) => true,
                ExprKind::Var(v) =>
                    self.string_vars.contains(v.as_str())
                    || matches!(self.var_types.get(v.as_str()), Some(crate::ast::Type::Str))
                    || matches!(self.var_types.get(v.as_str()), Some(crate::ast::Type::Named(n)) if n == "string" || n == "str"),
                _ => false,
            };
            let obj_s = self.emit_expr(obj);
            // Cast an index expression to `usize` where needed.
            let cast_idx = |raw: String, e: &crate::ast::Expr| -> String {
                match &e.kind {
                    ExprKind::Int(_) | ExprKind::Var(_) | ExprKind::BinOp(..) | ExprKind::Field(..) =>
                        format!("({}) as usize", raw),
                    _ => raw,
                }
            };
            if is_str {
                // String slice via char indices → collect back to Arc<str>.
                // s[lo..hi]  →  Arc::from(&s.chars().skip(lo).take(hi-lo).collect::<String>())
                // s[lo..]    →  Arc::from(&s.chars().skip(lo).collect::<String>())
                let lo_s = start.as_deref().map(|e| cast_idx(self.emit_expr(e), e));
                let hi_s = end.as_deref().map(|e| cast_idx(self.emit_expr(e), e));
                let str_ptr = self.str_ptr();
                let collected = match (lo_s, hi_s) {
                    (Some(lo), Some(hi)) => {
                        if *inclusive {
                            format!("{obj_s}.chars().skip({lo}).take({hi}+1-{lo}).collect::<String>()")
                        } else {
                            format!("{obj_s}.chars().skip({lo}).take({hi}-{lo}).collect::<String>()")
                        }
                    }
                    (Some(lo), None)    => format!("{obj_s}.chars().skip({lo}).collect::<String>()"),
                    (None, Some(hi))    => {
                        if *inclusive {
                            format!("{obj_s}.chars().take({hi}+1).collect::<String>()")
                        } else {
                            format!("{obj_s}.chars().take({hi}).collect::<String>()")
                        }
                    }
                    (None, None)        => format!("{obj_s}.chars().collect::<String>()"),
                };
                return format!("{str_ptr}::from({collected}.as_str())");
            }
            let start_s = start.as_deref().map(|e| cast_idx(self.emit_expr(e), e));
            let end_s   = end.as_deref().map(|e| cast_idx(self.emit_expr(e), e));
            let dots = if *inclusive { "..=" } else { ".." };
            let range_s = match (start_s, end_s) {
                (Some(s), Some(e)) => format!("{s}{dots}{e}"),
                (Some(s), None)    => format!("{s}.."),
                (None,    Some(e)) => format!("{dots}{e}"),
                (None,    None)    => "..".to_string(),
            };
            return format!("{}[{}].to_vec()", obj_s, range_s);
        }
        // Single-index string access: s[i] → the i-th character, as a one-char
        // string (boring has no separate `char` type -- comparisons like
        // `s[i] == "x"` and `s[i] != Q` expect a string back). Rust can't index
        // `str`/`Arc<str>` by usize at all (UTF-8 isn't O(1) per-byte), so this
        // goes through `.chars().nth(i)` instead.
        let is_str = match &obj.kind {
            ExprKind::Str(_) => true,
            ExprKind::Var(v) =>
                self.string_vars.contains(v.as_str())
                || matches!(self.var_types.get(v.as_str()), Some(crate::ast::Type::Str))
                || matches!(self.var_types.get(v.as_str()), Some(crate::ast::Type::Named(n)) if n == "string" || n == "str"),
            _ => false,
        };
        if is_str {
            let raw = self.emit_expr(idx);
            let idx_s = match &idx.kind {
                ExprKind::Int(_) | ExprKind::Var(_) | ExprKind::BinOp(..) | ExprKind::Field(..) =>
                    format!("({}) as usize", raw),
                _ => raw,
            };
            let str_ptr = self.str_ptr();
            // A plain immutable binding that's indexed elsewhere in the function too
            // has a `__strchars_<name>: Vec<char>` shadow (see `maybe_emit_str_index_cache`
            // / the param-prologue in `emit_fn`) -- O(1) lookup instead of `.chars().nth(idx)`,
            // which is O(idx) and turns a sequential scan into O(n^2).
            if let ExprKind::Var(v) = &obj.kind {
                if self.str_index_cache_vars.contains(v.as_str()) && self.immutable_local_vars.contains(v.as_str()) {
                    return format!(
                        "{str_ptr}::<str>::from(__strchars_{v}[{idx_s}].to_string().as_str())"
                    );
                }
            }
            let obj_s = self.emit_expr(obj);
            return format!(
                "{str_ptr}::<str>::from({obj_s}.chars().nth({idx_s}).expect(\"string index out of bounds\").to_string().as_str())"
            );
        }
        // When the index is an opaque collection index var (Option<usize> from
        // firstIndex/nextIndex), use the get_at(Option<usize>) trait method.
        if let ExprKind::Var(v) = &idx.kind {
            if self.index_vars.contains(v.as_str()) {
                return format!("{}.get_at({})", self.emit_expr(obj), v);
            }
        }
        // Dict vars (HashMap): use .get().cloned().unwrap() for bare access.
        // When wrapped in `else` (ExprKind::Else), that handler rebuilds the get
        // directly as .unwrap_or_else() to avoid a double-unwrap.
        //
        // `expr_is_dict` (not a plain `dict_vars`-only check) so an implicit
        // self-field bare identifier (`table[key]` inside a method body, with
        // no `self.` prefix) is recognized too, not just a tracked local dict
        // var — `self.emit_expr(obj)` resolves that case to `self.table` on its
        // own. Before this used `dict_vars.contains(obj_var)` alone, a bare
        // struct-field dict access fell through to the generic numeric-index
        // codegen below (an `as usize` cast + raw `[]` indexing), which doesn't
        // type-check against a `HashMap` key at all — see
        // docs/dict-index-optional-return-bug.md.
        //
        // When this index expression is itself a place expression (`in_lhs_assign`
        // — set by `emit_method_call_fallback` for `d[k].method()`, and by the
        // compound-assign codegen for `d[k] += ...`), `.get(k).cloned()` must NOT
        // be used: it hands back a throwaway clone, so a `def` (mutating) method
        // called on it silently mutates the clone and never touches the actual
        // map entry (the map read afterwards still shows the old value, even
        // though `boring run` and `cargo build` both succeed with no error).
        // `.get_mut(k)` yields the real place instead; the leading `*` turns the
        // `&mut V` into an lvalue so both `(*d.get_mut(k)...).method()` and
        // `(*d.get_mut(k)...) += rhs` compile.
        if matches!(&obj.kind, ExprKind::Var(_)) && self.expr_is_dict(obj) {
            let obj_s = self.emit_expr(obj);
            let key_ref = self.emit_dict_key_borrow(idx);
            if self.in_lhs_assign.get() {
                return format!("(*{}.get_mut({}).expect(\"dict key not found\"))", obj_s, key_ref);
            }
            // A bare dict-index tail flowing directly into an Optional-typed
            // return/let (`want_raw_dict_get`, set by emit_stmt.rs/emit_let.rs):
            // pass `.get(...).cloned()`'s own `Option<V>` through raw instead of
            // `.expect(...)`-panicking on a missing key and having the caller
            // wrap it in a redundant `Some(...)`.
            if self.want_raw_dict_get.get() {
                return format!("{}.get({}).cloned()", obj_s, key_ref);
            }
            return format!("{}.get({}).cloned().expect(\"dict key not found\")", obj_s, key_ref);
        }
        // self.field[key] where field is a dict-type struct field (HashMap): use dict-style access.
        // Detect by checking if the index key is a string-typed var or the field type is Dict.
        if let ExprKind::Field(inner_obj, field_name) = &obj.kind {
            if let ExprKind::Var(v) = &inner_obj.kind {
                if v == "self" {
                    let is_dict_field = self.self_type.as_deref()
                        .and_then(|t| self.struct_fields.get(t))
                        .and_then(|fields| fields.iter().find(|(fname, _)| fname == field_name))
                        .map(|(_, fty)| matches!(fty.without_mut(), crate::ast::Type::Dict(..)))
                        .unwrap_or(false);
                    let idx_is_string = match &idx.kind {
                        ExprKind::Var(v) => {
                            let vt = self.var_types.get(v.as_str());
                            matches!(vt, Some(crate::ast::Type::Str))
                            || matches!(vt, Some(crate::ast::Type::Named(n)) if n == "string" || n == "str")
                            || self.string_vars.contains(v.as_str())
                        }
                        ExprKind::Str(_) => true,
                        _ => false,
                    };
                    if is_dict_field || idx_is_string {
                        let obj_s2 = self.emit_expr(obj);
                        let key_ref = self.emit_dict_key_borrow(idx);
                        // Same throwaway-clone hazard as the plain dict-var case above.
                        if self.in_lhs_assign.get() {
                            return format!("(*{}.get_mut({}).expect(\"dict key not found\"))", obj_s2, key_ref);
                        }
                        // See the `want_raw_dict_get` branch above (plain dict-var case).
                        if self.want_raw_dict_get.get() {
                            return format!("{}.get({}).cloned()", obj_s2, key_ref);
                        }
                        return format!("{}.get({}).cloned().expect(\"dict key not found\")", obj_s2, key_ref);
                    }
                }
            }
        }
        // For string literal keys (HashMap), use the string key directly (Arc<str>: Deref<Target=str>)
        let idx_s = if matches!(&idx.kind, ExprKind::Str(_)) {
            format!("&{}", self.emit_expr_owned(idx))
        } else {
            // Rust requires usize for slice indexing; cast integer expressions.
            let raw = self.emit_expr(idx);
            match &idx.kind {
                ExprKind::Int(_) | ExprKind::Var(_) | ExprKind::BinOp(..) | ExprKind::Field(..) => format!("({}) as usize", raw),
                _ => raw,
            }
        };
        // Add .clone() so generic T values can be moved out of collections --
        // except when this index expression is itself an assignment target
        // (e.g. `mel[i] /= 4.0`, `arr[i] = v`): `.clone()` produces a temporary,
        // not an lvalue, so `arr[i].clone() /= 4.0` fails to compile (E0067).
        if self.in_lhs_assign.get() {
            format!("{}[{}]", self.emit_expr(obj), idx_s)
        } else {
            format!("{}[{}].clone()", self.emit_expr(obj), idx_s)
        }
    }

    /// `l op r` — reference/identity equality (`===`, `is`/`is not`), string
    /// concatenation, numeric-width coercion, struct operator-method dispatch, and
    /// `Arc<str>` string-comparison wrapping, before falling back to a plain Rust
    /// binary expression. `expr` (the whole `BinOp` node, not just its parts) is
    /// needed by the string-concatenation case, which walks the full `+` chain.
    fn emit_expr_binop(&self, expr: &Expr, op: &BinOp, l: &Expr, r: &Expr) -> String {
        // Reference equality ===
        if matches!(op, BinOp::RefEq) {
            let ls = self.emit_expr(l);
            let rs = self.emit_expr(r);
            let ptr_eq_fn = match self.config.threading {
                crate::transpiler::ThreadingMode::Multi  => "Arc::ptr_eq",
                crate::transpiler::ThreadingMode::Single => "Rc::ptr_eq",
            };
            return format!("{}(&{}, &{})", ptr_eq_fn, ls, rs);
        }
        // `x is SomeType` / `x is not SomeType` — type/nil check
        if matches!(op, BinOp::Is | BinOp::IsNot) {
            let is_not = matches!(op, BinOp::IsNot);
            // `x is nil` / `x is not nil` — right side is Nil
            if matches!(r.kind, ExprKind::Nil) {
                // Check if left side is an optional variable
                let is_optional = matches!(&l.kind, ExprKind::Var(v) if
                    self.optional_vars.contains(v.as_str()));
                if is_optional {
                    let ls = self.emit_expr(l);
                    return if is_not {
                        format!("({} != None)", ls)
                    } else {
                        format!("({} == None)", ls)
                    };
                }
                // Left side is `None` literal — comparing nil to nil
                if matches!(l.kind, ExprKind::Nil) {
                    return if is_not { "false".to_string() } else { "true".to_string() };
                }
                // Non-optional value: `x is nil` is always false, `x is not nil` always true
                return if is_not { "true".to_string() } else { "false".to_string() };
            }
            // `x is y` — reference identity between Rc-wrapped struct variables
            if let (ExprKind::Var(lv), ExprKind::Var(rv)) = (&l.kind, &r.kind) {
                if self.rc_identity_vars.contains(lv.as_str())
                    && self.rc_identity_vars.contains(rv.as_str())
                {
                    return if is_not {
                        format!("(!Rc::ptr_eq(&{}, &{}))", lv, rv)
                    } else {
                        format!("(Rc::ptr_eq(&{}, &{}))", lv, rv)
                    };
                }
            }
            // `x is TypeName` — struct type check
            if let ExprKind::Var(type_name) = &r.kind {
                if self.struct_fields.contains_key(type_name.as_str()) {
                    let ls = self.emit_expr(l);
                    return if is_not {
                        format!("!matches!({}, {} {{ .. }})", ls, type_name)
                    } else {
                        format!("matches!({}, {} {{ .. }})", ls, type_name)
                    };
                }
                // Enum variant check — `x is EnumVariant` (unit variant)
                if let Some(enum_name) = self.enum_variants.get(type_name.as_str()) {
                    let ls = self.emit_expr(l);
                    let qualified = format!("{}::{}", enum_name, type_name);
                    return if is_not {
                        format!("!matches!({}, {})", ls, qualified)
                    } else {
                        format!("matches!({}, {})", ls, qualified)
                    };
                }
            }
        }
        // String concatenation: if either side is a string expression, emit as Arc::<str>::from(format!(...))
        // This handles: string literal, string interp, known string vars, and nested string +.
        if matches!(op, BinOp::Add) && (self.is_string_expr(l) || self.is_string_expr(r)) {
            // Flatten the whole chain into a single format! call.
            let mut parts: Vec<String> = Vec::new();
            self.collect_string_parts(expr, &mut parts);
            let fmt = parts.iter().map(|_| "{}").collect::<Vec<_>>().join("");
            return format!("{}::<str>::from(format!(\"{}\", {}))", self.str_ptr(), fmt, parts.join(", "));
        }
        // Integer overflow: `boring run`'s interpreter wraps on `+`/`*` overflow
        // (two's-complement wraparound, via explicit `wrapping_add`/`wrapping_mul`
        // per integer type — see eval_expr.rs), and on `-` too but ONLY for signed
        // types (unsigned Sub deliberately raises a catchable underflow error
        // instead — see `is_signed_integer_rust_type`'s doc comment). Plain Rust
        // `+`/`-`/`*` is *checked* arithmetic in a debug build (panics on overflow
        // instead of wrapping), so emitting the infix operator here would diverge
        // from the interpreter outside release builds. Intercept before the
        // width-coercion and catch-all paths below emit a plain operator for this
        // case. `Div`/`Rem` are deliberately excluded — both backends already panic
        // identically on division by zero, so there's no divergence to fix there.
        if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) {
            let wrap_on_sub_ok = |ty: &str| !matches!(op, BinOp::Sub) || is_signed_integer_rust_type(ty);
            if let Some(int_ty) = self.integer_binop_type(l, r).filter(|t| wrap_on_sub_ok(t)) {
                let wrap_method = match op {
                    BinOp::Add => "wrapping_add",
                    BinOp::Sub => "wrapping_sub",
                    BinOp::Mul => "wrapping_mul",
                    _ => unreachable!(),
                };
                // Cast to `int_ty` on any side whose type isn't *confirmed* to
                // already be exactly `int_ty` — not just the "known but
                // different width" case. Method-call syntax (`recv.wrapping_add`)
                // can't lean on Rust's usual literal/inference-driven defaulting
                // the way an infix `+` can: a receiver that's a bare untyped
                // int literal (`0.wrapping_sub(n)`) or an unannotated local
                // (`let mut sum = 0; sum = sum.wrapping_add(n)`) is otherwise
                // "ambiguous numeric type" (E0689) — rustc needs a method's
                // receiver type pinned down before it can even look the method
                // up, unlike operator-trait resolution. An explicit `as int_ty`
                // is always valid here (the checker already guarantees both
                // operands are compatible numeric types by the time we emit
                // this), and a same-type cast is a harmless no-op.
                let l_ty = self.get_expr_rust_type(l);
                let r_ty = self.get_expr_rust_type(r);
                let ls_raw = self.emit_expr(l);
                let rs_raw = self.emit_expr(r);
                let ls = if l_ty.as_deref() == Some(int_ty.as_str()) { ls_raw } else { format!("({} as {})", ls_raw, int_ty) };
                let rs = if r_ty.as_deref() == Some(int_ty.as_str()) { rs_raw } else { format!("({} as {})", rs_raw, int_ty) };
                return format!("{}.{}({})", ls, wrap_method, rs);
            }
        }
        // Numeric type coercion: when adding/subtracting/multiplying/comparing typed
        // numeric vars of different widths (i8 + i16, uint == int, etc.), cast both
        // to the wider type — Rust's `==`/`<`/etc. require identical operand types.
        // Also handle mixed float-literal/int-literal arithmetic: `7.5 % 2` → `7.5_f64 % 2_f64`.
        if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
            | BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq) {
            let l_is_float_lit = matches!(l.kind, ExprKind::Float(_));
            let r_is_float_lit = matches!(r.kind, ExprKind::Float(_));
            let l_is_int_lit   = matches!(l.kind, ExprKind::Int(_));
            let r_is_int_lit   = matches!(r.kind, ExprKind::Int(_));
            if (l_is_float_lit && r_is_int_lit) || (l_is_int_lit && r_is_float_lit) {
                let ls_raw = self.emit_expr(l);
                let rs_raw = self.emit_expr(r);
                let ls = if l_is_int_lit { format!("{}_f64", ls_raw) } else { ls_raw };
                let rs = if r_is_int_lit { format!("{}_f64", rs_raw) } else { rs_raw };
                return format!("({} {} {})", ls, binop_str(op), rs);
            }
            // Bare int literal against a float-*typed* (non-literal) operand, e.g.
            // `1 < b` / `1 + b` where `b: float`. Rust's numeric-literal inference
            // unifies an untyped int literal with another integer type or with an
            // untyped float literal, but never across the int/float kind boundary —
            // so without this, `1 < b` emits as `(1 < b)` and rustc rejects it
            // ("can't compare {integer} with f64" / "cannot add f64 to {integer}").
            // Suffix the literal with the other side's concrete float type instead.
            if l_is_int_lit != r_is_int_lit {
                let other = if l_is_int_lit { r } else { l };
                if let Some(other_ty) = self.get_expr_rust_type(other) {
                    if other_ty == "f32" || other_ty == "f64" {
                        let ls_raw = self.emit_expr(l);
                        let rs_raw = self.emit_expr(r);
                        let ls = if l_is_int_lit { format!("{}_{}", ls_raw, other_ty) } else { ls_raw };
                        let rs = if r_is_int_lit { format!("{}_{}", rs_raw, other_ty) } else { rs_raw };
                        return format!("({} {} {})", ls, binop_str(op), rs);
                    }
                }
            }
            if let Some((l_ty, r_ty)) = self.get_numeric_types(l, r) {
                if l_ty != r_ty {
                    let wider = wider_numeric_type(&l_ty, &r_ty);
                    let ls_raw = self.emit_expr(l);
                    let rs_raw = self.emit_expr(r);
                    let ls = if l_ty != wider { format!("({} as {})", ls_raw, wider) } else { ls_raw };
                    let rs = if r_ty != wider { format!("({} as {})", rs_raw, wider) } else { rs_raw };
                    return format!("({} {} {})", ls, binop_str(op), rs);
                }
            }
        }
        // Struct operator method dispatch: `a + b` → `a.clone().add(b.clone())`
        // when the left operand's struct type has an operator method registered.
        let method_name = match op {
            BinOp::Add   => Some("add"),
            BinOp::Sub   => Some("sub"),
            BinOp::Mul   => Some("mul"),
            BinOp::Div   => Some("div"),
            BinOp::Rem   => Some("rem"),
            BinOp::Eq    => Some("eq"),
            BinOp::NotEq => Some("ne"),
            BinOp::Lt    => Some("lt"),
            BinOp::LtEq  => Some("le"),
            BinOp::Gt    => Some("gt"),
            BinOp::GtEq  => Some("ge"),
            _ => None,
        };
        if let Some(mname) = method_name {
            // Determine struct type from left operand.
            let struct_ty = if let ExprKind::Var(vname) = &l.kind {
                self.var_struct_types.get(vname.as_str()).cloned()
            } else {
                None
            };
            if let Some(sty) = struct_ty {
                let key = format!("{}::{}", sty, mname);
                if self.struct_operator_methods.contains(&key) {
                    let ls = self.emit_expr(l);
                    // Look up param types to decide if rhs needs Box::new() wrapping.
                    let param_types = self.struct_operator_param_types.get(&key).cloned();
                    let rs_raw = self.emit_expr(r);
                    let rs = if let Some(ptypes) = param_types {
                        if let Some(pty) = ptypes.first() {
                            if matches!(pty, Type::Qualified(_, q) if q.is_owned_or_new()) {
                                // Need to clone before boxing to avoid moving `rs_raw`
                                // when it's used multiple times (e.g. e3 == e3).
                                let clone_expr = if rs_raw.ends_with(".clone()") {
                                    rs_raw.clone()
                                } else {
                                    format!("{}.clone()", rs_raw)
                                };
                                // In managed mode: wrap in Arc<Mutex<T>> or RefCell<T>.
                                if self.is_managed_owned_user(pty) {
                                    match self.config.threading {
                                        crate::transpiler::ThreadingMode::Multi =>
                                            format!("Arc::new(std::sync::Mutex::new({}))", clone_expr),
                                        crate::transpiler::ThreadingMode::Single =>
                                            format!("RefCell::new({})", clone_expr),
                                    }
                                } else {
                                    format!("Box::new({})", clone_expr)
                                }
                            } else {
                                rs_raw
                            }
                        } else {
                            rs_raw
                        }
                    } else {
                        rs_raw
                    };
                    return format!("{}.clone().{}({})", ls, mname, rs);
                }
            }
        }
        // Arc<str> equality: wrap any string literal in Arc::<str>::from(...) when
        // compared with a non-literal expression (which may be Arc<str>).
        // This ensures type compatibility without needing full type inference.
        if matches!(op, BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq) {
            let l_is_raw_lit = matches!(&l.kind, ExprKind::Str(_));
            let r_is_raw_lit = matches!(&r.kind, ExprKind::Str(_));
            // is_arc_str: non-literal side is known Arc<str>; literals alone don't count.
            let l_is_arc_str = !l_is_raw_lit && self.is_string_expr(l);
            let r_is_arc_str = !r_is_raw_lit && self.is_string_expr(r);
            // Check if either side is a &str (str-ref) typed variable — skip wrapping.
            let l_is_str_ref = matches!(&l.kind, ExprKind::Var(v)
                if self.var_types.get(v.as_str()).map(Self::is_str_ref_type).unwrap_or(false));
            let r_is_str_ref = matches!(&r.kind, ExprKind::Var(v)
                if self.var_types.get(v.as_str()).map(Self::is_str_ref_type).unwrap_or(false));
            // Wrap into Arc<str> when any side is a string literal or known Arc<str>,
            // unless a side is a &str variable (which rejects Arc<str> comparisons).
            let should_wrap = !l_is_str_ref && !r_is_str_ref
                && (l_is_raw_lit || r_is_raw_lit || l_is_arc_str || r_is_arc_str);
            if should_wrap {
                let ls_expr = self.emit_expr(l);
                let rs_expr = self.emit_expr(r);
                // For ordering ops: use &str deref so PartialOrd<str> kicks in.
                let ord_op = match op {
                    BinOp::Lt    => Some("< std::cmp::Ordering::Equal"),
                    BinOp::LtEq  => Some("!= std::cmp::Ordering::Greater"),
                    BinOp::Gt    => Some("> std::cmp::Ordering::Equal"),
                    BinOp::GtEq  => Some("!= std::cmp::Ordering::Less"),
                    _ => None,
                };
                if let Some(ord) = ord_op {
                    let l_deref = if l_is_raw_lit {
                        if let ExprKind::Str(s) = &l.kind { format!("\"{}\"", escape_str(s)) }
                        else { ls_expr.clone() }
                    } else { format!("(&*{})", ls_expr) };
                    let r_deref = if r_is_raw_lit {
                        if let ExprKind::Str(s) = &r.kind { format!("\"{}\"", escape_str(s)) }
                        else { rs_expr.clone() }
                    } else { format!("(&*{})", rs_expr) };
                    return format!("({}.cmp({}) {})", l_deref, r_deref, ord);
                }
                // For Eq/NotEq: wrap literals in Rc/Arc::<str>::from(...) for type compat.
                let ls = if l_is_raw_lit {
                    if let ExprKind::Str(s) = &l.kind {
                        self.str_from(&escape_str(s))
                    } else { ls_expr.clone() }
                } else { ls_expr };
                let rs = if r_is_raw_lit {
                    if let ExprKind::Str(s) = &r.kind {
                        self.str_from(&escape_str(s))
                    } else { rs_expr.clone() }
                } else { rs_expr };
                return format!("({} {} {})", ls, binop_str(op), rs);
            }
        }
        let ls = self.emit_expr(l);
        let rs = self.emit_expr(r);
        format!("({} {} {})", ls, binop_str(op), rs)
    }

    /// `obj.field` — by far the largest single case in `emit_expr`: task/JoinHandle
    /// `.value`/`.wait`/`.done`, type-level access (`Counter.MAX`), getter/enum-getter
    /// dispatch, mutex/rwlock/managed-mode field reads, transient fields, module-path
    /// vs. instance-field disambiguation, and the general mapped-field fallback.
    fn emit_expr_field(&self, obj: &Expr, field: &String) -> String {
        // `.pointee` — explicit dereference for opaque/external Rust values (e.g.
        // Bevy's `Single<T>`, `Mut<T>`) that implement `Deref`/`DerefMut` but are
        // not structs Boring itself manages. Pure syntactic pass-through:
        // `expr.pointee` emits Rust's `*expr` verbatim, in both read position
        // (`let x = expr.pointee`) and assignment-target position
        // (`expr.pointee = value`, handled by emit_expr_assign's fallback, which
        // calls back into emit_expr → here) — no interaction with Boring's own
        // ownership/mut-type-modifier system. A real struct field literally named
        // `pointee` still takes priority, so this only fires when the receiver's
        // type is unknown to Boring's checker (i.e. an opaque external type).
        if field == "pointee" {
            let has_real_field = self.resolve_expr_struct_type(obj)
                .map(|ty| self.struct_fields.get(ty.as_str())
                    .map(|fields| fields.iter().any(|(fname, _)| fname == "pointee"))
                    .unwrap_or(false))
                .unwrap_or(false);
            if !has_real_field {
                let obj_s = self.emit_expr(obj);
                return format!("*{}", obj_s);
            }
        }
        // GPU targets only (see emit_kernel.rs): reading a `'unified`/`'global`
        // array field on a tracked kernel variable reads back the GPU buffer
        // instead of a plain (host-uninitialized) field access.
        if let Some(code) = self.try_emit_kernel_field_read(obj, field) {
            return code;
        }
        // Special case: `(task expr).value` / `(task expr).wait` where the task body
        // captures non-Arc local variables.  We cannot safely `tokio::spawn(async move {})`
        // because that would move the variable — leaving the outer scope without it.
        // Solution: inline the async call instead of spawning.
        if field == "value" || field == "wait" {
            if let ExprKind::Task(inner_e) = &obj.kind {
                let captured = collect_var_names(inner_e);
                let has_non_arc_captures = captured.iter().any(|v| {
                    self.known_local_vars.contains(v.as_str())
                        && !self.arc_vars.contains(v.as_str())
                        && !self.string_arc_vars.contains(v.as_str())
                        && !self.weak_vars.contains(v.as_str())
                });
                if has_non_arc_captures {
                    // Inline: emit the inner expression (method call already gets .await
                    // appended by emit_expr for async methods).
                    let inner_s = self.emit_expr(inner_e);
                    return if field == "wait" {
                        format!("{{ let _ = {}; }}", inner_s)
                    } else {
                        inner_s
                    };
                }
            }
        }
        // Type-level access: `Counter.MAX` → `Counter::MAX`, `Counter.count` → `Counter::count()`
        if let ExprKind::Var(type_name) = &obj.kind {
            // A top-level `let` constant (`PADDLE_SIZE`, `WALL_COLOR`) is a VALUE, never a
            // type -- even though Boring constants conventionally use UPPER_SNAKE_CASE and
            // would otherwise trip the uppercase-heuristic branch below. `.field` on one of
            // these (`PADDLE_SIZE.x`) is a genuine instance field access, not a type-level
            // path lookup: it must fall through to the plain `obj.field` emission further
            // down, not return `PADDLE_SIZE::x` (invalid Rust -- there is no such associated
            // item). See `top_level_let_external_call`'s doc for the promotion this pairs
            // with -- a `let` promoted that way is exactly the shape that needs this.
            let is_top_level_let_value = self.user_top_level_names.contains(type_name.as_str());
            if !is_top_level_let_value && type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                let key = format!("{}::{}", type_name, field);
                if self.struct_type_var_names.contains(&key) {
                    // type let → associated const (UPPER_CASE in Rust)
                    return format!("{}::{}", type_name, field.to_uppercase());
                }
                if self.struct_type_mut_var_names.contains(&key) {
                    // type var → module-level static Mutex: read via lock(), recover from poisoning
                    return format!("*{}.lock().unwrap_or_else(|e| e.into_inner())", field.to_uppercase());
                }
                // Fieldless enum variant (no args): CalcError.DivByZero → CalcError::DivByZero
                if self.enum_variant_fields.contains_key(&key) {
                    return format!("{}::{}", type_name, field);
                }
                // Fallback for external PascalCase types (Ordering, Duration, etc.):
                // `Ordering.SeqCst` → `Ordering::SeqCst`
                if !self.known_local_vars.contains(type_name.as_str()) {
                    return format!("{}::{}", type_name, field);
                }
            }
            // oneshot rx.value → rx.await.unwrap() (receive the single value)
            if field == "value" && self.oneshot_receivers.contains(type_name.as_str()) {
                return if self.in_throws || self.in_try_body {
                    format!("{}.await?", type_name)
                } else {
                    format!("{}.await.unwrap()", type_name)
                };
            }
            // watch rx.value → current value without waiting
            if field == "value" && self.watch_receivers.contains(type_name.as_str()) {
                return format!("{}.borrow().clone()", type_name);
            }
            // future.value / future.wait on a spawned JoinHandle or async param.
            //
            // Three cases:
            //  (a) throws JoinHandle  — JoinHandle<Result<T,BoringError>>
            //      throws ctx : f.await.unwrap()?      — unwrap JoinError, propagate inner
            //      plain ctx  : f.await.unwrap().unwrap()  — panic on inner error
            //  (b) plain JoinHandle   — JoinHandle<T>
            //      always     : f.await.unwrap()       — just unwrap JoinError
            //  (c) async fn param     — impl Future<Output=T> (or Result<T,_>)
            //      throws ctx + value : f.await?
            //      otherwise          : f.await.unwrap()
            if field == "done" && self.task_vars.contains(type_name.as_str()) {
                return format!(
                    "tokio::time::timeout(std::time::Duration::ZERO, {}).await.is_ok()",
                    type_name
                );
            }
            if (field == "value" || field == "wait") && self.task_vars.contains(type_name.as_str()) {
                let in_throws_ctx = self.in_throws || self.in_try_body;
                let is_throws_handle = self.throws_join_handle_vars.contains(type_name.as_str());
                let is_join_handle   = self.join_handle_vars.contains(type_name.as_str());
                let p = self.str_ptr();
                return if field == "wait" {
                    if is_throws_handle && in_throws_ctx {
                        format!("{{ let _ = {}.await.map_err(|__e| Box::new(BoringError::String({}::from(__e.to_string()))) as Box<dyn std::error::Error + Send + Sync>)??.expect(\"unhandled task error\"); }}", type_name, p)
                    } else if is_throws_handle {
                        format!("{{ let _ = {}.await.expect(\"task panicked\").expect(\"unhandled task error\"); }}", type_name)
                    } else if in_throws_ctx {
                        format!("{{ {}.await.map_err(|__e| Box::new(BoringError::String({}::from(__e.to_string()))) as Box<dyn std::error::Error + Send + Sync>)?; }}", type_name, p)
                    } else {
                        format!("{{ let _ = {}.await; }}", type_name)
                    }
                } else {
                    // .value
                    if is_throws_handle {
                        if in_throws_ctx {
                            format!("{}.await.expect(\"task panicked\")?", type_name)
                        } else {
                            format!("{}.await.expect(\"task panicked\").expect(\"unhandled task error\")", type_name)
                        }
                    } else if is_join_handle {
                        format!("{}.await.expect(\"task panicked\")", type_name)
                    } else if in_throws_ctx {
                        format!("{}.await?", type_name)
                    } else {
                        format!("{}.await.expect(\"task panicked\")", type_name)
                    }
                };
            }
        }
        let obj_s = self.emit_expr(obj);
        // `.value` / `.wait` on a JoinHandle → `.await.unwrap()`.
        // Covers inline task expressions `(task ...).value` and loop vars `future.wait`
        // that aren't tracked in task_vars.
        // Future.done() — non-blocking poll: true if the JoinHandle is finished.
        if field == "done" {
            // Use try_join with zero timeout as a non-blocking poll.
            return format!(
                "tokio::time::timeout(std::time::Duration::ZERO, {}).await.is_ok()",
                obj_s
            );
        }
        if field == "value" || field == "wait" {
            // TaskWithTimeout: always a throws JoinHandle (wraps Result<T, Elapsed>).
            // Needs .await.unwrap()? in throws context to propagate Error.Expired,
            // or .await.unwrap().unwrap() otherwise (panics on Elapsed).
            if matches!(&obj.kind, ExprKind::TaskWithTimeout(..)) {
                let in_throws_ctx = self.in_throws || self.in_try_body;
                return if field == "wait" {
                    if in_throws_ctx {
                        format!("{{ let _ = {}.await.expect(\"task panicked\")?; }}", obj_s)
                    } else {
                        format!("{{ let _ = {}.await.expect(\"task panicked\").expect(\"unhandled task error\"); }}", obj_s)
                    }
                } else if in_throws_ctx {
                    format!("{}.await.expect(\"task panicked\")?", obj_s)
                } else {
                    format!("{}.await.expect(\"task panicked\").expect(\"unhandled task error\")", obj_s)
                };
            }

            let is_future = matches!(&obj.kind, ExprKind::Task(_))
                || obj_s.contains("tokio::spawn")
                || obj_s.contains("async move");
            if is_future {
                return if field == "wait" {
                    format!("{{ let _ = {}.await; }}", obj_s)
                } else {
                    format!("{}.await.expect(\"task panicked\")", obj_s)
                };
            }
            // Loop variable holding a JoinHandle: only treat as future if the var is
            // explicitly in task_vars (declared with a task expression).
            // Using var_struct_types, var_types, or struct_fields to distinguish
            // struct field access from JoinHandle avoids false positives on plain
            // structs with a "value" field (e.g. LetStmt, ReturnStmt, pair tuples).
            if let ExprKind::Var(v) = &obj.kind {
                let is_known_struct = self.var_struct_types.contains_key(v.as_str())
                    || self.struct_fields.contains_key(v.as_str())
                    || self.var_types.get(v.as_str()).map(|t| {
                        if let Type::Named(tn) = t { self.struct_fields.contains_key(tn.as_str()) } else { false }
                    }).unwrap_or(false);
                let is_task = self.task_vars.contains(v.as_str());
                if is_task
                    && v != "self"
                    && !self.var_mutex_types.contains(v.as_str())
                    && !is_known_struct
                {
                    return if field == "wait" {
                        format!("{{ let _ = {}.await; }}", obj_s)
                    } else {
                        format!("{}.await.expect(\"task panicked\")", obj_s)
                    };
                }
            }
        }
        // Check if this field access is a getter property (req method with no params).
        // (a) `self.field` where `self` is the current struct instance and `field` is a getter.
        // (b) `var.field` where `var` is any variable, and `field` is registered as a getter
        //     in any struct — cross-struct fallback for `let t = Temperature(); t.fahrenheit`.
        // Both guards require obj to be a plain Var (not a chained field access like `self.text`)
        // to avoid incorrectly treating built-in properties (`.length` on strings/arrays).
        let is_getter = if let ExprKind::Var(v) = &obj.kind {
            if v == "self" {
                self.self_type.as_deref()
                    .map(|t| self.struct_getters.contains(&format!("{}::{}", t, field)))
                    .unwrap_or(false)
            } else {
                let from_struct = self.var_struct_types.get(v.as_str())
                    .map(|type_name| self.struct_getters.contains(&format!("{}::{}", type_name, field)))
                    .unwrap_or(false);
                let from_enum = if !from_struct {
                    if let Some(Type::Named(type_name)) = self.var_types.get(v.as_str()) {
                        self.struct_getters.contains(&format!("{}::{}", type_name, field))
                    } else { false }
                } else { false };
                from_struct || from_enum
            }
        } else {
            false
        };
        // Enum field accessors return Option<T> — unwrap at callsite with a clear message.
        let is_enum_field_getter = if let ExprKind::Var(v) = &obj.kind {
            let type_name = if v == "self" {
                self.self_type.clone()
            } else {
                self.var_types.get(v.as_str()).and_then(|t| {
                    if let Type::Named(n) = t { Some(n.clone()) } else { None }
                })
            };
            type_name.map(|t| self.enum_field_getters.contains(&format!("{}::{}", t, field)))
                .unwrap_or(false)
        } else {
            false
        };
        if is_enum_field_getter {
            return format!("{}.{}().expect(\"field '{}' not available in this variant\")", obj_s, field, field);
        }
        if is_getter {
            return format!("{}.{}()", obj_s, field);
        }
        // Mutex var access: w.field → w.lock().await.field (multi) or w.borrow().field (single)
        if let ExprKind::Var(v) = &obj.kind {
            if self.var_mutex_types.contains(v.as_str()) || self.var_mutex_task_types.contains(v.as_str()) {
                let access = self.mutex_var_read(v, v);
                // If the field is itself an Rc/Arc<RefCell/Mutex<T>> (actor/guard),
                // we must clone it to avoid moving out of the borrow/lock guard.
                let field_is_shared = self.var_types.get(v.as_str())
                        .and_then(|t| match t { Type::Named(n) => Some(n.as_str()), Type::Qualified(inner, _) => if let Type::Named(n) = inner.as_ref() { Some(n.as_str()) } else { None }, _ => None })
                        .or_else(|| self.var_struct_types.get(v.as_str()).map(|s| s.as_str()))
                        .and_then(|tname| self.struct_fields.get(tname))
                        .and_then(|fields| fields.iter().find(|(fname, _)| fname == field))
                        .map(|(_, fty)| Self::is_arc_qualified(fty) || Self::is_rc_qualified(fty))
                        .unwrap_or(false);
                if field_is_shared {
                    return match self.config.threading {
                        crate::transpiler::ThreadingMode::Single =>
                            format!("Rc::clone(&{}.{})", access, field),
                        crate::transpiler::ThreadingMode::Multi =>
                            format!("Arc::clone(&{}.{})", access, field),
                    };
                }
                // Builtin pseudo-fields (`.length`/`.count`/`.isEmpty`) need the same
                // `map_field` remap the non-mutex path applies further down (`arr.length`
                // → `arr.len() as isize`) — this branch used to emit the Boring field
                // name verbatim, so `mutex_var.length` produced invalid Rust
                // (`MutexGuard<Vec<T>>` has no `length` field; confirmed via
                // examples/tokio.br's `all_users.length` on an `'actor'task` array).
                let mapped = map_field(field);
                let result = format!("{}.{}", access, mapped);
                return if mapped.contains(" as ") { format!("({})", result) } else { result };
            }
            // RwLock var access: w.field → w.read().unwrap().field (multi) or w.borrow().field (single)
            if self.var_rwlock_types.contains(v.as_str()) || self.var_rwlock_task_types.contains(v.as_str()) {
                let access = if self.var_rwlock_task_types.contains(v.as_str()) {
                    self.guard_task_read_access(v)
                } else {
                    self.guard_read_access(v)
                };
                return format!("{}.{}", access, field);
            }
            // Managed-mode mutex var (std::sync::Mutex, synchronous):
            // w.field → w.lock().unwrap().field
            // If the param has a shadow guard, use it directly (no re-locking).
            if let Some(shadow) = self.managed_param_shadows.get(v.as_str()) {
                return format!("{}.{}", shadow, field);
            }
            if self.managed_mutex_vars.contains(v.as_str()) {
                return format!("{}.lock().unwrap().{}", v, field);
            }
            // Managed-mode RefCell var (single-thread):
            // w.field → w.borrow().field
            if self.managed_refcell_vars.contains(v.as_str()) {
                return format!("{}.borrow().{}", v, field);
            }
        }
        // Mutex struct field: self.worker.field → self.worker.lock().await.field (multi) / self.worker.borrow().field (single)
        if let ExprKind::Field(inner_obj, mutex_field) = &obj.kind {
            if let ExprKind::Var(v) = &inner_obj.kind {
                if v == "self" {
                    let key = self.self_type.as_deref()
                        .map(|t| format!("{}::{}", t, mutex_field));
                    if let Some(k) = key {
                        if self.struct_mutex_fields.contains(&k) || self.struct_mutex_task_fields.contains(&k) {
                            return format!("{}.{}", self.mutex_field_read(&k, &format!("self.{}", mutex_field)), field);
                        }
                    }
                }
            }
        }
        // Transient field read: self.field → self.field.get() (Cell) or self.field.borrow().clone() (RefCell)
        if obj_s == "self" {
            let key = self.self_type.as_deref()
                .map(|t| format!("{}::{}", t, field));
            if let Some(k) = key {
                if let Some((is_copy, _, _)) = self.transient_fields.get(&k) {
                    return if *is_copy {
                        format!("self.{}.get()", field)
                    } else {
                        format!("self.{}.borrow().clone()", field)
                    };
                }
            }
        }
        // Determine if the receiver is a module/type path (use `::`) or instance (use `.`).
        // A receiver is a path when:
        //   (a) it is an uppercase Var (type name like `Ordering`, `Duration`, `File`)
        //   (b) it is a lowercase Var NOT in known_local_vars and NOT `self`
        //       (e.g. `mpsc`, `tokio` — module names imported but not declared as locals)
        //   (c) the emitted receiver already contains `::` (cascaded path: `tokio::time`)
        // A top-level `let` constant is never a path receiver regardless of case (see the
        // matching guard in the "Type-level access" block above, near the top of this
        // function, for why `PADDLE_SIZE.x` must stay `.x`).
        let is_path_receiver = match &obj.kind {
            ExprKind::Var(v) => {
                // Implicit-self field: a bare identifier that resolves to the current
                // struct's own field (not a real local var) is an *instance* receiver,
                // never a module/type path — even though it's absent from
                // known_local_vars (this
                // used to fall into the `else` arm below and get treated as an
                // imported-module-style path receiver, mis-emitting `self.limbs.length`
                // as `self.limbs::length`). Mirrors the implicit-self resolution at the
                // top of `emit_expr`'s own `ExprKind::Var` arm.
                let is_implicit_self_field = !self.known_local_vars.contains(v.as_str())
                    && self.self_type.as_deref()
                        .and_then(|t| self.struct_fields.get(t))
                        .map(|fields| fields.iter().any(|(fname, _)| fname == v))
                        .unwrap_or(false);
                if v == "self" || self.user_top_level_names.contains(v.as_str()) || is_implicit_self_field { false }
                else if v.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) { true }
                else { !self.known_local_vars.contains(v.as_str()) }
            }
            // A field access on another field/call is a path only if the receiver is also
            // a plain path (contains `::` but is NOT a call result ending with `)`)
            _ => obj_s.contains("::") && !obj_s.ends_with(')'),
        };
        if is_path_receiver {
            return format!("{}::{}", obj_s, field);
        }
        // Don't apply map_field to user-defined struct fields (e.g. a field named
        // `count` should not be remapped to `len()` on a user struct). This used to
        // only resolve the receiver's struct type for the literal `self` receiver
        // (`(v == "self").then_some(...)` short-circuited to `None`/`false` for
        // every other variable), so a plain parameter or local of struct type
        // (`t.count` where `t: Thing`) fell straight through to the builtin
        // `.len()` remap below. `resolve_expr_struct_type` resolves the receiver's
        // struct type uniformly -- `self`, any other `Var`, AND a field-of-field
        // chain (`self.item.count`) -- so this now covers every receiver shape,
        // not just `self`.
        let is_user_field = self.resolve_expr_struct_type(obj)
            .and_then(|t| self.struct_fields.get(t.as_str()))
            .map(|fields| fields.iter().any(|(fname, _)| fname == field))
            .unwrap_or(false);
        let mapped = if is_user_field { field.as_str() } else { map_field(field) };
        // If the accessed field is Arc-qualified (actor/guard/shared), add .clone()
        // so the value is not moved out of the struct — Arc::clone is cheap.
        let field_is_arc = if let ExprKind::Var(v) = &obj.kind {
            let struct_name = self.var_struct_types.get(v.as_str())
                .cloned()
                .or_else(|| self.var_types.get(v.as_str()).and_then(|t| match t {
                    Type::Named(n) => Some(n.clone()),
                    Type::Qualified(inner, _) => if let Type::Named(n) = inner.as_ref() { Some(n.clone()) } else { None },
                    _ => None,
                }));
            struct_name
                .and_then(|sn| self.struct_fields.get(sn.as_str()))
                .and_then(|fields| fields.iter().find(|(fname, _)| fname == field))
                .map(|(_, fty)| Self::is_arc_qualified(fty) || Self::is_rc_qualified(fty) || Self::is_string_type(fty))
                .unwrap_or(false)
        } else {
            false
        };
        let result = if field_is_arc && !self.in_lhs_assign.get() {
            format!("{}.{}.clone()", obj_s, mapped)
        } else {
            format!("{}.{}", obj_s, mapped)
        };
        // Wrap `x.len() as i64` etc. in parens to avoid Rust parsing `i64 <` as generic args.
        if mapped.contains(" as ") { format!("({})", result) } else { result }
    }

    /// `target = value` — setter/property dispatch (instance/type setters, transient
    /// fields, mutex/rwlock field writes), the immutable-param diagnostic, compound-
    /// assignment desugaring (`x += y`), dict/array-index writes, and the general
    /// fallback (GPU-resident materialization, actor auto-clone, Optional auto-wrap).
    fn emit_expr_assign(&self, target: &Expr, value: &Expr) -> String {
        // Global mutable var assignment: `logX = val` → `*LOGX.lock().unwrap() = val`.
        if let ExprKind::Var(var_name) = &target.kind {
            if self.global_vars_used_in_fns.contains(var_name.as_str()) {
                let static_name = var_name.to_uppercase();
                let val_s = self.emit_expr_owned(value);
                return format!("*{}.lock().unwrap_or_else(|e| e.into_inner()) = {}", static_name, val_s);
            }
            // `var` primitive param: `n = val` → `*n = val`.
            if self.var_primitive_params.contains(var_name.as_str()) {
                let val_s = self.emit_expr(value);
                return format!("*{} = {}", var_name, val_s);
            }
        }
        if let ExprKind::Field(obj, field) = &target.kind {
            // Instance setter property: `t.prop = v` → `t.set_prop(v)`.
            // Check if `field` is registered as a setter for any struct.
            let is_instance_setter = self.struct_setters.iter()
                .any(|k| k.ends_with(&format!("::{}", field)));
            if is_instance_setter {
                let obj_s = self.emit_expr(obj);
                let val_s = self.emit_expr_owned(value);
                return format!("{}.set_{}({})", obj_s, field, val_s);
            }
            // If assigning to a type var that has a type setter, call the setter function.
            if let ExprKind::Var(type_name) = &obj.kind {
                if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    let key = format!("{}::{}", type_name, field);
                    if self.struct_type_mut_var_names.contains(&key) {
                        // Invoke the type setter if one exists, unless already inside it
                        let has_setter = !self.in_type_setter
                            && self.struct_type_method_sigs.get(type_name.as_str())
                                .and_then(|m| m.get(field.as_str()))
                                .map(|k| matches!(k, TypeMethodKind::Set))
                                .unwrap_or(false);
                        if has_setter {
                            let val_s = self.emit_expr_owned(value);
                            return format!("{}::set_{}({})", type_name, field, val_s);
                        }
                    }
                }
            }
        }
        // Transient field write: self.field = v → self.field.set(v) (Cell) or *self.field.borrow_mut() = v (RefCell)
        // When the field type is Optional, the assigned value is coerced to Some(v).
        if let ExprKind::Field(obj, field) = &target.kind {
            if let ExprKind::Var(v) = &obj.kind {
                if v == "self" {
                    let key = self.self_type.as_deref()
                        .map(|t| format!("{}::{}", t, field));
                    if let Some(k) = key {
                        if let Some((is_copy, field_ty, _)) = self.transient_fields.get(&k) {
                            let is_copy = *is_copy;
                            // Wrap in Some() if field is Optional and value is not nil
                            let raw_val = self.emit_expr_owned(value);
                            let is_nil = matches!(value.kind, ExprKind::Nil);
                            let val_s = if !is_nil && matches!(field_ty, Type::Optional(_)) {
                                if raw_val.starts_with("Some(") || raw_val == "None" {
                                    raw_val
                                } else {
                                    format!("Some({})", raw_val)
                                }
                            } else {
                                raw_val
                            };
                            return if is_copy {
                                format!("self.{}.set({})", field, val_s)
                            } else {
                                format!("*self.{}.borrow_mut() = {}", field, val_s)
                            };
                        }
                    }
                }
            }
        }
        // Mutex field write: w.field = v
        if let ExprKind::Field(obj, field) = &target.kind {
            if let ExprKind::Var(v) = &obj.kind {
                if self.var_mutex_types.contains(v.as_str()) || self.var_mutex_task_types.contains(v.as_str()) {
                    let val_s = self.emit_expr_owned(value);
                    let guard = self.mutex_var_write(v, v);
                    return format!("{{ let mut __g = {}; __g.{} = {}; }}", guard, field, val_s);
                }
            }
            // self.worker.field = v
            if let ExprKind::Field(inner_obj, mutex_field) = &obj.kind {
                if let ExprKind::Var(v) = &inner_obj.kind {
                    if v == "self" {
                        let key = self.self_type.as_deref()
                            .map(|t| format!("{}::{}", t, mutex_field));
                        if let Some(k) = key {
                            if self.struct_mutex_fields.contains(&k) || self.struct_mutex_task_fields.contains(&k) {
                                let val_s = self.emit_expr_owned(value);
                                let guard = self.mutex_field_write(&k, &format!("self.{}", mutex_field));
                                return format!("{{ let mut __g = {}; __g.{} = {}; }}", guard, field, val_s);
                            }
                        }
                    }
                }
            }
        }
        // RwLock field write: c.field = v
        if let ExprKind::Field(obj, field) = &target.kind {
            if let ExprKind::Var(v) = &obj.kind {
                if self.var_rwlock_types.contains(v.as_str()) || self.var_rwlock_task_types.contains(v.as_str()) {
                    let val_s = self.emit_expr_owned(value);
                    let guard = if self.var_rwlock_task_types.contains(v.as_str()) {
                        self.guard_task_write_guard(v)
                    } else {
                        self.guard_write_guard(v)
                    };
                    return format!("{{ let mut __wg = {}; __wg.{} = {}; }}", guard, field, val_s);
                }
            }
            // self.data.field = v
            if let ExprKind::Field(inner_obj, rwlock_field) = &obj.kind {
                if let ExprKind::Var(v) = &inner_obj.kind {
                    if v == "self" {
                        let key = self.self_type.as_deref()
                            .map(|t| format!("{}::{}", t, rwlock_field));
                        if let Some(k) = key {
                            if self.struct_rwlock_fields.contains(&k) || self.struct_rwlock_task_fields.contains(&k) {
                                let val_s = self.emit_expr_owned(value);
                                let guard = self.rwlock_field_write(&k, &format!("self.{}", rwlock_field));
                                return format!("{{ let mut __wg = {}; __wg.{} = {}; }}", guard, field, val_s);
                            }
                        }
                    }
                }
            }
        }
        // Diagnostic: assigning to a field of a non-`mut` struct parameter.
        if let ExprKind::Field(obj, field) = &target.kind {
            if let ExprKind::Var(v) = &obj.kind {
                if v != "self"
                    && self.fn_current_params.contains_key(v.as_str())
                    && !self.fn_current_params_mut.contains(v.as_str())
                    && !self.var_mutex_types.contains(v.as_str())
                    && !self.var_mutex_task_types.contains(v.as_str())
                    && !self.var_rwlock_types.contains(v.as_str())
                    && !self.var_rwlock_task_types.contains(v.as_str())
                {
                    let struct_name = self.fn_current_params.get(v.as_str()).and_then(|ty| {
                        if let crate::ast::Type::Named(n) = ty { Some(n.clone()) } else { None }
                    });
                    if let Some(sn) = struct_name {
                        if self.struct_fields.contains_key(sn.as_str()) {
                            let line = self.fn_current_param_lines.get(v.as_str()).copied().unwrap_or(0);
                            let col = self.fn_current_param_cols.get(v.as_str()).copied().unwrap_or(0);
                            self.push_error(line, col, format!("`{}` is not declared `mut` — cannot assign to field `.{}` on an immutable binding; fix: declare the parameter as `mut {} {}`", v, field, sn, v));
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
                    // Same diagnostic as above, for a plain local binding rather
                    // than a parameter — see the matching comment in
                    // `emit_methods.rs`'s `emit_method_call_fallback`.
                    let struct_name = self.var_struct_types.get(v.as_str()).cloned()
                        .or_else(|| self.var_types.get(v.as_str()).and_then(|t| {
                            if let crate::ast::Type::Named(n) = t.without_mut() { Some(n.clone()) } else { None }
                        }));
                    if let Some(sn) = struct_name {
                        if self.struct_fields.contains_key(sn.as_str()) {
                            self.push_error(obj.line, obj.col, format!("`{}` is not declared `mut` — cannot assign to field `.{}` on a non-mut binding; fix: declare it `mut {} {}` or `var mut {} {}`", v, field, sn, v, sn, v));
                        }
                    }
                }
            } else if let ExprKind::Index(inner_obj, _idx) = &obj.kind {
                // `arr[i].field = v` — same permission as `arr[i].method()`
                // (see the matching check in `emit_methods.rs`'s
                // `emit_method_call_fallback`): the collection's own declared
                // element type must grant `mut`.
                if let ExprKind::Var(coll_name) = &inner_obj.kind {
                    if let Some(coll_ty) = self.var_types.get(coll_name.as_str()) {
                        if let Some(elem_ty) = coll_ty.index_element_type() {
                            if !elem_ty.grants_mut() {
                                self.push_error(obj.line, obj.col, format!(
                                    "cannot assign to field `.{}` on an element of `{}` — its declared element type doesn't grant content mutation; declare it `[mut T]`/`{{K = mut V}}`",
                                    field, coll_name
                                ));
                            }
                        }
                    }
                }
            }
        }
        // Diagnostic: assigning to a non-reassignable (`let`/`mut`, not
        // `var`/`var mut`) struct field from outside the struct's own
        // methods. Independent of the owner-mut checks just above (a field
        // can be legally *reachable* through a `mut`/`var mut` owner and
        // still not be reassignable, if the field's own declaration is
        // `let`/`mut` rather than `var`) — `boring run`'s interpreter
        // already rejects this (`methods.rs::assign`'s "cannot assign to
        // immutable field" check); this mirrors it here so `boring build`
        // doesn't silently transpile an illegal reassignment through a
        // `let`/`mut` field with no diagnostic at all (a real gap found
        // while finalizing docs/book.md). `self.field = v` is
        // untouched — every self-write path above already handles it
        // (transient/mutex/rwlock fields) or falls through unchecked,
        // matching today's behavior for in-method writes (mirrors
        // `methods.rs::assign`'s own `binding_name != "self"` exemption).
        if let ExprKind::Field(obj, field) = &target.kind {
            if let ExprKind::Var(v) = &obj.kind {
                if v != "self" {
                    let owner_struct = self.var_struct_types.get(v.as_str()).cloned()
                        .or_else(|| self.var_types.get(v.as_str()).and_then(|t| {
                            if let crate::ast::Type::Named(n) = t.without_mut() { Some(n.clone()) } else { None }
                        }))
                        .or_else(|| self.fn_current_params.get(v.as_str()).and_then(|ty| {
                            if let crate::ast::Type::Named(n) = ty { Some(n.clone()) } else { None }
                        }));
                    if let Some(sn) = owner_struct {
                        let key = format!("{}::{}", sn, field);
                        if self.struct_field_reassignable.get(&key) == Some(&false) {
                            self.push_error(obj.line, obj.col, format!(
                                "cannot assign to field `.{}` on `{}` — declared `let`/`mut` (not reassignable); use `var`/`var mut` on the field's own declaration to allow `.{} = ...`",
                                field, sn, field
                            ));
                        }
                    }
                }
            }
        }
        // Compound assignment: `x = x op rhs` → `x op= rhs` (idiomatic Rust).
        // Detected by matching BinOp(op, lhs_copy, rhs) where lhs_copy emits the same
        // string as target — safe because the parser already desugared `x op= rhs`.
        // Exception: string addition (`Arc<str>` does not implement `AddAssign`).
        if let ExprKind::BinOp(op, lhs_copy, rhs) = &value.kind {
            let is_string_add = matches!(op, BinOp::Add)
                && (self.is_string_expr(lhs_copy)
                    || self.is_string_expr(rhs)
                    || self.is_string_expr(target)
                    || matches!(&target.kind, ExprKind::Var(v)
                        if self.var_types.get(v.as_str())
                            .map(|t| matches!(t, Type::Named(n) if n == "string"))
                            .unwrap_or(false)));
            if !is_string_add {
                // Integer overflow (same rationale as emit_expr_binop's wrapping
                // branch): `x += y`/`-=`/`*=` must wrap on overflow like the
                // interpreter, but Rust has no `wrapping_add_assign` — desugar to a
                // full reassignment `x = x.wrapping_add(y)` instead of `x += y`.
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) {
                    let wrap_on_sub_ok = |ty: &str| !matches!(op, BinOp::Sub) || is_signed_integer_rust_type(ty);
                    if let Some(int_ty) = self.integer_binop_type(lhs_copy, rhs).filter(|t| wrap_on_sub_ok(t)) {
                        let wrap_method = match op {
                            BinOp::Add => "wrapping_add",
                            BinOp::Sub => "wrapping_sub",
                            BinOp::Mul => "wrapping_mul",
                            _ => unreachable!(),
                        };
                        self.in_lhs_assign.set(true);
                        let target_s = self.emit_expr(target);
                        let lhs_s    = self.emit_expr(lhs_copy);
                        self.in_lhs_assign.set(false);
                        if target_s == lhs_s {
                            // Same "cast unless confirmed already `int_ty`" rule as
                            // emit_expr_binop's wrapping branch (see its comment) —
                            // needed for the receiver too, e.g. an unannotated
                            // `let mut sum = 0` local reassigned as `sum += n`
                            // would otherwise leave `sum.wrapping_add(...)`
                            // ambiguous (E0689). The plain `target_s` (uncast)
                            // still has to be used as the assignment's own lvalue.
                            let lhs_ty = self.get_expr_rust_type(lhs_copy);
                            let recv_s = if lhs_ty.as_deref() == Some(int_ty.as_str()) { target_s.clone() } else { format!("({} as {})", target_s, int_ty) };
                            let rhs_ty  = self.get_expr_rust_type(rhs);
                            let rhs_raw = self.emit_expr_owned(rhs);
                            let rhs_s = if rhs_ty.as_deref() == Some(int_ty.as_str()) { rhs_raw } else { format!("({} as {})", rhs_raw, int_ty) };
                            return format!("{} = {}.{}({})", target_s, recv_s, wrap_method, rhs_s);
                        }
                    }
                }
                let compound_op = match op {
                    BinOp::Add => Some("+="),
                    BinOp::Sub => Some("-="),
                    BinOp::Mul => Some("*="),
                    BinOp::Div => Some("/="),
                    BinOp::Rem => Some("%="),
                    _ => None,
                };
                if let Some(op_str) = compound_op {
                    // Emit both sides in lhs-assign mode: `target` because it's the
                    // actual compound-assign target (e.g. `mel[i] /= 4.0` must not
                    // become `mel[i].clone() /= 4.0` -- clone() isn't an lvalue), and
                    // `lhs_copy` (the parser's desugared duplicate of the same target)
                    // to match, so the equality check below still lines up.
                    self.in_lhs_assign.set(true);
                    let target_s = self.emit_expr(target);
                    let lhs_s    = self.emit_expr(lhs_copy);
                    self.in_lhs_assign.set(false);
                    if target_s == lhs_s {
                        let rhs_s = self.emit_expr_owned(rhs);
                        return format!("{} {} {}", target_s, op_str, rhs_s);
                    }
                }
            }
        }
        // Dict subscript assignment: dict[key] = val → dict.insert(key_owned, val)
        if let ExprKind::Index(dict_obj, key) = &target.kind {
            if let ExprKind::Var(dict_name) = &dict_obj.kind {
                if self.dict_vars.contains(dict_name.as_str()) {
                    let key_owned = self.emit_dict_key_owned(key);
                    let val_s = self.emit_expr_owned(value);
                    return format!("{}.insert({}, {})", dict_name, key_owned, val_s);
                }
            }
            // obj.field[key] = val → obj.field.insert(key_owned, val)
            // `obj` may be `self` or any other struct-typed var/param reached from
            // outside the struct's own methods -- e.g. a `var Registry registry`
            // parameter on a free function assigning into `registry.table[key]`.
            // `resolve_struct_name` already special-cases `self` (matching the old
            // self-only check below) and resolves any other `Var`/`Field` chain via
            // `var_struct_types`/`var_types`, so one branch now covers both instead
            // of silently falling through to the generic read-expression path (which
            // used to emit an invalid `.clone()`-as-assignment-target for the
            // non-self case).
            if let ExprKind::Field(inner_obj, field_name) = &dict_obj.kind {
                let is_dict_field = self.resolve_struct_name(inner_obj)
                    .and_then(|t| self.struct_fields.get(t.as_str())
                        .and_then(|fields| fields.iter().find(|(n, _)| n == field_name))
                        .map(|(_, ty)| matches!(ty.without_mut(), Type::Dict(..))))
                    .unwrap_or(false);
                if is_dict_field {
                    let key_owned = self.emit_dict_key_owned(key);
                    let val_s = self.emit_expr_owned(value);
                    let obj_s = self.emit_expr(inner_obj);
                    return format!("{}.{}.insert({}, {})", obj_s, field_name, key_owned, val_s);
                }
            }
            // Implicit self: `table[key] = v` inside a struct method where `table` is
            // itself a dict-typed struct field (no explicit `self.` prefix) — same
            // implicit-self resolution the read path already does in the `ExprKind::Var`
            // arm of `emit_expr` above, which this assignment-target path never checked.
            if let ExprKind::Var(field_name) = &dict_obj.kind {
                if !self.known_local_vars.contains(field_name.as_str()) {
                    let is_dict_field = self.self_type.as_ref()
                        .and_then(|t| self.struct_fields.get(t))
                        .and_then(|fields| fields.iter().find(|(n, _)| n == field_name))
                        .map(|(_, ty)| matches!(ty.without_mut(), Type::Dict(..)))
                        .unwrap_or(false);
                    if is_dict_field {
                        let key_owned = self.emit_dict_key_owned(key);
                        let val_s = self.emit_expr_owned(value);
                        return format!("self.{}.insert({}, {})", field_name, key_owned, val_s);
                    }
                }
            }
            // Chained double-index assignment on a dict-of-dicts, with no
            // intermediate local: `local_table[k1][k2] = v`. Here `dict_obj`
            // (`local_table[k1]`) is itself an `Index` expr, not `Var`/`Field`,
            // so none of the branches above match and this used to fall through
            // to the generic array-index LHS codegen further down — `(k2) as usize`
            // array indexing, which doesn't type-check against a real `HashMap`
            // value. `expr_is_dict(dict_obj)` (now `Index`-aware, see its own doc
            // comment) recognizes `local_table[k1]` as dict-shaped via
            // `local_table`'s declared `{K1={K2=V}}` type; emitting `dict_obj`
            // itself under `in_lhs_assign` reuses `emit_expr_index`'s existing
            // `.get_mut(..).expect("dict key not found")` place-expression path
            // (the same one already used for `d[k].method()`/`d[k] += rhs`) to
            // get a mutable handle on the inner dict, then `.insert()`s into it.
            // Mirrors the interpreter's `assign`'s recursive `ExprKind::Index`
            // semantics: panics if the outer key is absent rather than silently
            // auto-vivifying an empty inner dict.
            if matches!(&dict_obj.kind, ExprKind::Index(..)) && self.expr_is_dict(dict_obj) {
                self.in_lhs_assign.set(true);
                let place_s = self.emit_expr(dict_obj);
                self.in_lhs_assign.set(false);
                let key_owned = self.emit_dict_key_owned(key);
                let val_s = self.emit_expr_owned(value);
                return format!("{}.insert({}, {})", place_s, key_owned, val_s);
            }
        }
        // emit_expr_owned wraps string literals in Arc::from; falls through for other types
        // For index LHS (arr[i] = v), emit without .clone() since we're writing not reading.
        let target_s = if let ExprKind::Index(arr_obj, idx_expr) = &target.kind {
            let raw_idx = self.emit_expr(idx_expr);
            let idx_s = match &idx_expr.kind {
                ExprKind::Int(_) | ExprKind::Var(_) | ExprKind::BinOp(..) | ExprKind::Field(..) => format!("({}) as usize", raw_idx),
                _ => raw_idx,
            };
            match &arr_obj.kind {
                // Implicit self: `items[i] = v` inside a struct method where `items` is
                // itself an array-typed (or any non-dict) struct field — mirrors the
                // dict implicit-self branch above; without this, the field name is
                // emitted bare (`items[i] = v`), which is not a local and doesn't
                // compile ("cannot find value `items`").
                ExprKind::Var(arr_name)
                    if !self.dict_vars.contains(arr_name.as_str())
                        && !self.known_local_vars.contains(arr_name.as_str())
                        && self.self_type.as_ref()
                            .and_then(|t| self.struct_fields.get(t))
                            .map(|fields| fields.iter().any(|(n, _)| n == arr_name))
                            .unwrap_or(false) =>
                {
                    format!("self.{}[{}]", arr_name, idx_s)
                }
                ExprKind::Var(arr_name) if !self.dict_vars.contains(arr_name.as_str()) => {
                    format!("{}[{}]", arr_name, idx_s)
                }
                // self.field[i] = v — struct field array element assignment
                ExprKind::Field(inner_obj, field_name)
                    if matches!(&inner_obj.kind, ExprKind::Var(v) if v == "self") =>
                {
                    format!("self.{}[{}]", field_name, idx_s)
                }
                _ => self.emit_expr(target),
            }
        } else {
            self.in_lhs_assign.set(true);
            let s = self.emit_expr(target);
            self.in_lhs_assign.set(false);
            s
        };
        // Interprocedural GPU residency: a plain (re-)assignment target (a
        // struct field like `self.cache_ca_k`, or an ordinary already-declared
        // local) has no way to opt into staying resident the way a fresh
        // `let` binding does — so a bare call to a `fn_returns_resident`
        // function on the RHS must always materialize here, unless the target
        // is itself a tracked resident local being reassigned (rare, but a
        // real "opt-in stays resident" case: don't force-materialize into it).
        let target_is_resident = matches!(&target.kind, ExprKind::Var(name)
            if self.resident_call_vars.contains_key(name.as_str()));
        let rhs_s = if target_is_resident {
            None
        } else {
            self.try_materialize_resident_call(value)
        }.unwrap_or_else(|| self.emit_expr_owned(value));
        // When assigning an actor (Arc<Mutex<T>>) variable into a struct field, auto-clone
        // so the original binding remains usable after the assignment. Arc::clone is cheap.
        let rhs_s = if matches!(&target.kind, ExprKind::Field(..))
            && matches!(&value.kind, ExprKind::Var(vn)
                if (self.var_mutex_types.contains(vn.as_str())
                    || self.var_mutex_task_types.contains(vn.as_str()))
                    && !rhs_s.ends_with(".clone()"))
        {
            format!("{}.clone()", rhs_s)
        } else {
            rhs_s
        };
        // When assigning to an optional var, wrap non-optional RHS in Some().
        // Needed for `var Expr? x = nil; x = throws_call()?` → `x = Some(throws_call()?)`.
        let rhs_s = if let ExprKind::Var(v) = &target.kind {
            let rhs_already_opt = rhs_s.starts_with("Some(")
                || rhs_s == "None"
                || is_try_optional(value)
                || matches!(&value.kind, ExprKind::Nil)
                || matches!(&value.kind, ExprKind::Var(vn)
                    if self.optional_vars.contains(vn.as_str())
                    || self.var_types.get(vn.as_str())
                        .map(|t| matches!(t, Type::Optional(_))).unwrap_or(false))
                // RHS is a call to a function that returns Optional — no extra Some() needed.
                || matches!(&value.kind, ExprKind::Call(callee, _)
                    if matches!(&callee.kind, ExprKind::Var(fn_name)
                        if self.fn_return_types.get(fn_name.as_str())
                            .map(|t| matches!(t, Type::Optional(_))).unwrap_or(false)))
                // RHS is a field read of a `T?`-declared field, or a method call whose own
                // declared return type is `T?` (any receiver shape) — e.g.
                // `id = items[1].as_str()` where `as_str()` already returns `string?`.
                // See docs/option-return-double-some-wrap-bug.md.
                || self.expr_is_declared_optional(value);
            if self.optional_vars.contains(v.as_str()) && !rhs_already_opt {
                format!("Some({})", rhs_s)
            } else {
                rhs_s
            }
        } else {
            rhs_s
        };
        format!("{} = {}", target_s, rhs_s)
    }

    /// `try? EXPR` — recognizes builtins that already do their own Result→Option/panic
    /// handling in a plain (non-throws) context (`fromJson<T>(s)`, `fs.read(path)`, and the
    /// other `fs.*` operations — see book.md's own tables), and emits their dedicated
    /// `try?`-aware form directly instead of letting `ExprKind::TryElse`'s generic path
    /// (emit plain, append another `.ok()`) double-handle them. Returns `None` for anything
    /// else, so the caller falls back to the generic handling. See
    /// docs/try-wrap-double-handling-bug.md for the bug this fixes.
    fn emit_try_optional_self_handling_builtin(&self, e: &Expr) -> Option<String> {
        match &e.kind {
            // `try? fromJson<T>(s)` — book.md documents this as identical to the plain
            // form (`serde_json::from_str::<T>(&s).ok()`). Force that plain emission via a
            // sub-transpiler with throws flags cleared, regardless of the ambient context
            // (`try?` may itself appear inside a `throws` function, where the ambient
            // `in_throws` would otherwise route `fromJson` to its `?`-suffixed form instead).
            ExprKind::GenericCall(callee, type_args, args) => {
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "fromJson" {
                        let mut sub = self.make_sub();
                        sub.in_throws = false;
                        sub.in_try_body = false;
                        return Some(sub.emit_generic_call(callee, type_args, args));
                    }
                }
                None
            }
            // `try? fs.read(path)` / `try? fs.write(...)` / etc. — route through the
            // dedicated `.ok()`-suffixed bare-Result form (see emit_methods.rs's
            // `emit_fs_call_try_optional`), which gracefully yields `None` on a real
            // failure instead of `fs.read`'s plain-context `.unwrap()` panicking.
            ExprKind::MethodCall(obj, method, args) => {
                if let ExprKind::Var(v) = &obj.kind {
                    if v == "fs" {
                        return Some(self.emit_fs_call_try_optional(method, args));
                    }
                }
                None
            }
            _ => None,
        }
    }

    pub(crate) fn emit_call(&self, callee: &Expr, args: &[Arg]) -> String {
        if let ExprKind::Var(name) = &callee.kind {
            // `GPU(n)` — a device handle. wgpu only ever has one real adapter (see
            // `wgpu::host::emit_gpu_introspection_globals`), so every index resolves
            // to it; this exists purely so `let g = GPU(0); print g.name()`-style
            // source (see examples/saxpy.br) is portable between the interpreter's
            // simulation and --target wgpu, matching how CUDA/Metal already support
            // multi-device `GPU(n)` for real. Must be checked before the `is_type`
            // constructor branch below — "GPU" is capitalized like a type name, but
            // no `struct GPU` exists to construct.
            if name == "GPU" && self.is_gpu_target {
                let idx = args.first().map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "0".into());
                return format!("(({}) as usize)", idx);
            }
            // Type constructors (PascalCase) — emit as struct literal or ::new()
            let is_type = name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
            if is_type {
                return self.emit_constructor(name, args);
            }
            // Built-in async primitives take priority over fn_sigs dispatch.
            // `wait` / `sleep` — emit as tokio::time::sleep (or sleep_until for Instant).
            // Must come before fn_sigs check because wait/timeout are now in fn_sigs
            // (so that DotIdent hints work), but they need to emit tokio:: paths, not
            // a plain `wait(...)` call.
            if (name == "sleep" || name == "wait") && args.len() == 1 && self.in_async {
                let is_instant = expr_is_instant(&args[0].value, &self.instant_vars);
                let type_prefix = if is_instant { "Instant" } else { "Duration" };
                // Resolve leading-dot static call: `.fromSecs(n)` → `Duration::from_secs(n)`
                let arg = self.resolve_dot_with_type(&args[0].value, type_prefix)
                    .unwrap_or_else(|| self.emit_expr(&args[0].value));
                return if is_instant {
                    format!("tokio::time::sleep_until({}).await", arg)
                } else {
                    format!("tokio::time::sleep({}).await", arg)
                };
            }
            // Overloaded function call — select the right mangled name based on arg types.
            if self.overloaded_fn_names.contains(name.as_str()) {
                let overloads = self.fn_overload_decls.get(name.as_str())
                    .cloned()
                    .unwrap_or_default();
                // Try to find a matching overload by type inference.
                let chosen = overloads.iter().find(|decl| {
                    if decl.params.len() != args.len() { return false; }
                    decl.params.iter().zip(args.iter()).all(|(param, arg)| {
                        match &param.ty {
                            None => true,
                            Some(expected_ty) => {
                                // `infer_overload_expr_type` has no case for `.Variant` —
                                // check it against this candidate's actual enum directly
                                // instead of falling into the optimistic-match default,
                                // which used to accept `.Variant` against every candidate.
                                if let ExprKind::DotIdent(variant) = &arg.value.kind {
                                    self.dotident_matches_enum_type(variant, expected_ty)
                                } else {
                                    let inferred = infer_overload_expr_type(
                                        &arg.value,
                                        &self.var_types,
                                        &self.fn_return_types,
                                        &self.struct_fields,
                                    );
                                    match inferred {
                                        Some(inferred_ty) => types_compatible(expected_ty, &inferred_ty),
                                        None => true, // can't determine — optimistically match
                                    }
                                }
                            }
                        }
                    })
                }).or_else(|| overloads.first());

                if let Some(decl) = chosen {
                    let mangled = mangle_overload_name(name, &decl.params);
                    let args_s = self.emit_args_coerced(&mangled, args);
                    let base = format!("{}({})", mangled, args_s);
                    let is_task = self.in_async && self.task_fns.contains(name.as_str());
                    let propagates = (self.in_try_body || self.in_throws) && self.fn_throws.contains(name.as_str());
                    return match (is_task, propagates) {
                        (true,  true)  => format!("{}.await?", base),
                        (true,  false) => format!("{}.await",  base),
                        (false, true)  => format!("{}?",       base),
                        (false, false) => base,
                    };
                }
            }
            // User-defined functions (and stdlib functions registered in fn_sigs).
            if self.fn_sigs.contains_key(name.as_str()) {
                let args_s = self.emit_args_coerced(name, args);
                let base = format!("{}({})", escape_rust_keyword(name), args_s);
                let is_task = self.in_async
                    && self.task_fns.contains(name.as_str())
                    && !self.stream_fns.contains(name.as_str());
                let propagates = (self.in_try_body || self.in_throws) && self.fn_throws.contains(name.as_str());
                // Correct ordering: async task calls must be `.await` then `?` (not `?` then `.await`).
                return match (is_task, propagates) {
                    (true,  true)  => format!("{}.await?", base),
                    (true,  false) => format!("{}.await",  base),
                    (false, true)  => format!("{}?",       base),
                    (false, false) => base,
                };
            }
            // Callable struct: `obj(args)` where `obj` is a struct instance with `def ()` / `req ()`.
            // Emit `obj.__call__(args)` — the anonymous method is named `__call__` in Rust.
            if let Some(struct_name) = self.var_struct_types.get(name.as_str()).cloned() {
                if self.callable_structs.contains(&struct_name) {
                    let args_s = args.iter().map(|a| self.emit_expr(&a.value)).collect::<Vec<_>>().join(", ");
                    return format!("{}.__call__({})", name, args_s);
                }
            }

            // Special case: `timeout(dur, future_expr)` — the second arg must be a future.
            // `tokio::time::timeout` takes `F: Future`, so the second arg should be the future
            // expression directly (e.g. `tokio::time::sleep(dur)` or a closure async block).
            // When the second arg is a `task expr` (Task node), emit the inner expression
            // directly as a future rather than spawning. This avoids the JoinHandle type mismatch.
            if name == "timeout" && args.len() == 2 && self.in_async {
                // Resolve leading-dot syntax for the duration/deadline argument:
                //   timeout(.fromSecs(5), …)  →  timeout(Duration::from_secs(5), …)
                let is_instant = expr_is_instant(&args[0].value, &self.instant_vars);
                let type_prefix = if is_instant { "Instant" } else { "Duration" };
                let dur = self.resolve_dot_with_type(&args[0].value, type_prefix)
                    .unwrap_or_else(|| self.emit_expr(&args[0].value));
                // For `task inner` args: emit the inner expression directly (already a Future).
                // For async method calls that end with `.await`, wrap in `async move { }` so
                // timeout receives a `Future<Output=T>` rather than the already-awaited value.
                // For plain futures (e.g. tokio::time::sleep): pass through as-is.
                let future_expr = {
                    // `timeout` needs a Future<Output=T>, not an already-awaited value.
                    // Three forms for the second argument:
                    //   task f(args)       — TaskExpr: emit inner expression directly as future
                    //   f                  — Callable<T> (task fn ref): call it as f() to get future
                    //   <already a future> — strip any trailing .await added by the expression emitter
                    let raw = match &args[1].value.kind {
                        ExprKind::Task(inner_e) => self.emit_expr(inner_e),
                        // Bare variable: check if it's a task function — call it to get the future
                        ExprKind::Var(fn_name)
                            if self.task_fns.contains(fn_name.as_str())
                               || self.fn_sigs.contains_key(fn_name.as_str()) =>
                        {
                            // If it's a known task_fn with no args: call it to produce the future
                            if self.task_fns.contains(fn_name.as_str()) {
                                format!("{}()", fn_name)
                            } else {
                                self.emit_expr(&args[1].value)
                            }
                        }
                        // Zero-arg trailing closure `(): body` or `(): fetch()` —
                        // unwrap the body and emit it directly as the future expression.
                        // This handles `timeout(dur): fetch()` → future is `fetch()`, not `|| fetch()`.
                        ExprKind::Closure(params, _, body, _, _) if params.is_empty() => {
                            match body {
                                ClosureBody::Expr(e) => self.emit_expr(e),
                                ClosureBody::Block(stmts) => {
                                    let mut sub = self.make_sub();
                                    sub.in_async = true;
                                    sub.in_throws = false;
                                    sub.emit_body(stmts);
                                    format!("async move {{{}}}", sub.out)
                                }
                            }
                        }
                        _ => self.emit_expr(&args[1].value),
                    };
                    if let Some(stripped) = raw.strip_suffix(".await") {
                        stripped.to_string()
                    } else if let Some(stripped) = raw.strip_suffix(".await?") {
                        stripped.to_string()
                    } else {
                        raw
                    }
                };
                // In a cancellable function: use select! to race future vs timer vs cancel.
                if self.in_cancellable_fn {
                    let timer_fn = if expr_is_instant(&args[0].value, &self.instant_vars) {
                        format!("tokio::time::sleep_until({})", dur)
                    } else {
                        format!("tokio::time::sleep({})", dur)
                    };
                    return if self.in_throws || self.in_try_body {
                        format!(
                            "{{ tokio::select! {{ __boring_r = ({}) => Ok(__boring_r), _ = {} => Err(Box::new(BoringError::Other(std::any::TypeId::of::<Error>(), Box::new(Error::Expired) as Box<dyn BoringVal + Send + Sync>))), _ = __task_cancel.cancelled() => Err(Box::new(BoringError::Other(std::any::TypeId::of::<Error>(), Box::new(Error::Cancelled) as Box<dyn BoringVal + Send + Sync>))), }} }}?",
                            future_expr, timer_fn
                        )
                    } else {
                        format!(
                            "{{ tokio::select! {{ __boring_r = ({}) => Some(__boring_r), _ = {} => None, _ = __task_cancel.cancelled() => None, }} }}",
                            future_expr, timer_fn
                        )
                    };
                }
                // Always add .await — TryElse clears in_throws/in_try_body to avoid adding `?`.
                let base = if expr_is_instant(&args[0].value, &self.instant_vars) {
                    format!("tokio::time::timeout_at({}, {}).await", dur, future_expr)
                } else {
                    format!("tokio::time::timeout({}, {}).await", dur, future_expr)
                };
                // In throws/try context, propagate Elapsed errors with `?`.
                return if self.in_throws || self.in_try_body {
                    format!("{}?", base)
                } else {
                    base
                };
            }
            // Task fn params: calling them produces a future that needs .await.
            // When the param type is also `throws` (returns Future<Output=Result<T,_>>),
            // add `?` in a throws / try context so errors propagate correctly.
            if self.in_async && self.task_vars.contains(name.as_str()) {
                let args_s = self.emit_args(args);
                let call_s = format!("{}({})", escape_rust_keyword(name), args_s);
                return if self.in_throws || self.in_try_body {
                    format!("{}.await?", call_s)
                } else {
                    format!("{}.await", call_s)
                };
            }
            // Non-task fn params declared as `throws` return Result — add `?` in throws context.
            if (self.in_throws || self.in_try_body) && self.throws_fn_params.contains(name.as_str()) {
                let args_s = self.emit_args(args);
                let call_s = format!("{}({})", escape_rust_keyword(name), args_s);
                return format!("{}?", call_s);
            }
            return self.emit_builtin_call(name, args);
        }
        // Enum variant constructor: Value.NativeFn(name, val) → Value::NativeFn(name, Box::new(val))
        // for recursive fields. Check if callee is Field(Var(EnumType), VariantName).
        if let ExprKind::Field(obj_expr, variant_name) = &callee.kind {
            if let ExprKind::Var(enum_type) = &obj_expr.kind {
                let key = format!("{}::{}", enum_type, variant_name);
                if let Some(field_types) = self.enum_variant_field_types.get(&key).cloned() {
                    let callee_s = format!("{}::{}", enum_type, variant_name);
                    let args_s: Vec<String> = args.iter().enumerate().map(|(i, a)| {
                        let raw = self.emit_let_value(field_types.get(i), &a.value);
                        // enum variant fields are owned — strip the leading `&` that
                        // emit_let_value adds for actor-typed function params.
                        let raw = if matches!(field_types.get(i),
                            Some(Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask | OwnerQual::Guard | OwnerQual::GuardTask))
                        ) {
                            raw.strip_prefix('&').unwrap_or(&raw).to_string()
                        } else {
                            raw
                        };
                        let rec_key = format!("{}::{}::{}", enum_type, variant_name, i);
                        if self.recursive_fields.contains(&rec_key) {
                            if matches!(field_types.get(i), Some(Type::Optional(_))) {
                                format!("{}.map(Box::new)", raw)
                            } else {
                                format!("Box::new({})", raw)
                            }
                        } else {
                            raw
                        }
                    }).collect();
                    return format!("{}({})", callee_s, args_s.join(", "));
                }
            }
        }
        let callee_s = self.emit_expr(callee);
        let args_s = self.emit_args(args);
        format!("{}({})", callee_s, args_s)
    }

    pub(crate) fn emit_generic_call(&self, callee: &Expr, type_args: &[Type], args: &[Arg]) -> String {
        if let ExprKind::Var(name) = &callee.kind {
            match name.as_str() {
                "channel" => {
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    // Capacity: second type arg (channel<T, 32>) or first call arg (channel<T>(cap)).
                    let cap = if type_args.len() >= 2 {
                        match &type_args[1] {
                            crate::ast::Type::Named(n) => n.clone(),
                            other => self.emit_type(other),
                        }
                    } else {
                        args.first()
                            .map(|a| self.emit_expr(&a.value))
                            .unwrap_or_else(|| "0".to_string())
                    };
                    let channel_mod = match self.config.threading {
                        crate::transpiler::ThreadingMode::Single => {
                            self.uses_local_channel.set(true);
                            "local_channel::mpsc"
                        }
                        crate::transpiler::ThreadingMode::Multi  => "tokio::sync::mpsc",
                    };
                    // local_channel::mpsc::channel() is unbounded — no capacity argument.
                    if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        return format!("{}::channel::<{}>()", channel_mod, ty);
                    }
                    return format!("{}::channel::<{}>({})", channel_mod, ty, cap);
                }
                "oneshot" => {
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    return format!("tokio::sync::oneshot::channel::<{}>()", ty);
                }
                "broadcast" => {
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        self.uses_local_broadcast.set(true);
                        return format!("local_broadcast::<{}>()", ty);
                    }
                    let cap = if type_args.len() >= 2 {
                        match &type_args[1] {
                            crate::ast::Type::Named(n) => n.clone(),
                            other => self.emit_type(other),
                        }
                    } else {
                        args.first()
                            .map(|a| self.emit_expr(&a.value))
                            .unwrap_or_else(|| "16".to_string())
                    };
                    return format!("tokio::sync::broadcast::channel::<{}>({})", ty, cap);
                }
                "watch" => {
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    let init = args.first()
                        .map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "Default::default()".to_string());
                    return format!("tokio::sync::watch::channel::<{}>({})", ty, init);
                }
                "timeout" => {
                    // timeout(dur, fut) — contextual:
                    //   cancellable fn  → select! racing future / sleep / cancel token
                    //   throws context  → .await?     (Elapsed propagated as error)
                    //   otherwise       → .await.ok() (returns T?)
                    let dur = args.first()
                        .map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "Duration::from_secs(0)".to_string());
                    let raw_fut = args.get(1)
                        .map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "async {}".to_string());
                    // Strip trailing .await so we pass a Future, not its value.
                    let fut = raw_fut.strip_suffix(".await")
                        .or_else(|| raw_fut.strip_suffix(".await?"))
                        .unwrap_or(&raw_fut)
                        .to_string();
                    if self.in_cancellable_fn {
                        return if self.in_throws || self.in_try_body {
                            format!(
                                "{{ tokio::select! {{ __boring_r = ({}) => Ok(__boring_r), _ = tokio::time::sleep({}) => Err(Box::new(BoringError::Other(std::any::TypeId::of::<Error>(), Box::new(Error::Expired) as Box<dyn BoringVal + Send + Sync>))), _ = __task_cancel.cancelled() => Err(Box::new(BoringError::Other(std::any::TypeId::of::<Error>(), Box::new(Error::Cancelled) as Box<dyn BoringVal + Send + Sync>))), }} }}?",
                                fut, dur
                            )
                        } else {
                            format!(
                                "{{ tokio::select! {{ __boring_r = ({}) => Some(__boring_r), _ = tokio::time::sleep({}) => None, _ = __task_cancel.cancelled() => None, }} }}",
                                fut, dur
                            )
                        };
                    }
                    let base = format!("tokio::time::timeout({}, {}).await", dur, fut);
                    return if self.in_throws || self.in_try_body {
                        format!("{}?", base)
                    } else {
                        format!("{}.ok()", base)
                    };
                }
                // from_json<T>(s) → serde_json::from_str::<T>(&s)
                // In a throws/try context propagates the error; otherwise wraps in .ok().
                "fromJson" => {
                    self.uses_serde.set(true);
                    let ty = type_args.first()
                        .map(|t| self.emit_type(t))
                        .unwrap_or_else(|| "_".to_string());
                    let s = args.first()
                        .map(|a| self.emit_expr(&a.value))
                        .unwrap_or_else(|| "\"\"".to_string());
                    let base = format!("serde_json::from_str::<{}>(&{})", ty, s);
                    return if self.in_throws || self.in_try_body {
                        format!("{}?", base)
                    } else {
                        format!("{}.ok()", base)
                    };
                }
                _ => {}
            }
            // A known user struct constructed with an explicit type argument, e.g.
            // `Stack<int>([])` — found wiring `boring.collections`'s `Stack<T>`/`Queue<T>`
            // (docs/cross-project-code-sharing-gap.md's stdlib work): without this check,
            // execution fell straight to the "regular call" fallback below, which ignores
            // struct-ness entirely and emits `Stack::<isize>(vec![])` — invalid Rust for a
            // named-field struct (E0423: "use struct literal syntax instead of calling").
            // `emit_constructor`/`emit_constructor_inner` already build the correct
            // `Name { field: value, ... }` / `Name::new(...)` form for a bare `Stack([])`
            // (no type args) — reuse that and just splice the turbofish onto its head.
            if self.struct_fields.contains_key(name.as_str()) {
                return self.emit_constructor_with_turbofish(name, type_args, args);
            }
            // Fallback: emit as a regular call, ignore type args
            let ty_args_s: Vec<String> = type_args.iter().map(|t| self.emit_type(t)).collect();
            let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
            format!("{}::<{}>({})", name, ty_args_s.join(", "), args_s.join(", "))
        } else {
            let callee_s = self.emit_expr(callee);
            let ty_args_s: Vec<String> = type_args.iter().map(|t| self.emit_type(t)).collect();
            let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
            format!("{}::<{}>({})", callee_s, ty_args_s.join(", "), args_s.join(", "))
        }
    }

    pub(crate) fn emit_pipe(&self, lhs: &Expr, name: &str, args: &[Arg]) -> String {
        // If the name is a known standalone function, insert lhs as first argument.
        // Otherwise treat it as a method call on lhs.
        if self.fn_sigs.contains_key(name) {
            // Route through the same per-parameter coercion as a normal call so lhs
            // (which may be an auto-ref'd array/dict/set/struct param) gets the
            // correct `&`/`&mut`/`.clone()` treatment instead of being emitted raw.
            let mut all_args: Vec<Arg> = Vec::with_capacity(args.len() + 1);
            all_args.push(Arg { label: None, value: lhs.clone(), spread: false, default_rest: false });
            all_args.extend_from_slice(args);
            let all_args = self.emit_args_coerced(name, &all_args);
            let base = format!("{}({})", escape_rust_keyword(name), all_args);
            let is_task = self.in_async && self.task_fns.contains(name);
            let propagates = (self.in_try_body || self.in_throws) && self.fn_throws.contains(name);
            match (is_task, propagates) {
                (true,  true)  => format!("{}.await?", base),
                (true,  false) => format!("{}.await",  base),
                (false, true)  => format!("{}?",       base),
                (false, false) => base,
            }
        } else {
            // Method call: delegate directly to emit_method_call with the real lhs expr.
            self.emit_method_call(lhs, name, args)
        }
    }

    pub(crate) fn emit_constructor(&self, name: &str, args: &[Arg]) -> String {
        let result = self.emit_constructor_inner(name, args);
        // Check if the current function returns T'owned/T'new in managed mode → wrap in managed actor.
        if let Some(Type::Qualified(inner, q)) = &self.fn_return_ty {
            if q.is_owned_or_new() && matches!(inner.as_ref(), Type::Named(n) if n == name) {
                if self.is_managed_owned_user(self.fn_return_ty.as_ref().unwrap()) {
                    return match self.config.threading {
                        crate::transpiler::ThreadingMode::Multi =>
                            format!("Arc::new(std::sync::Mutex::new({}))", result),
                        crate::transpiler::ThreadingMode::Single =>
                            format!("RefCell::new({})", result),
                    };
                }
                // Strict mode: wrap in Box<T>
                return format!("Box::new({})", result);
            }
        }
        result
    }

    /// `emit_constructor` variant for a struct constructed with explicit generic type
    /// arguments (`Stack<int>([])`, as opposed to inferred `Stack([])`). Delegates to the
    /// normal constructor logic for the actual `Name { field: value }` / `Name::new(...)`
    /// shape, then splices `::<T1, T2>` onto the type name — Rust accepts a turbofish on a
    /// struct-literal path (`Foo::<i32> { x: 0 }`) the same as on a call. This is also
    /// needed (not just stylistic) when a field's initial value is itself ambiguous without
    /// it, e.g. `Stack<int>([])` — `vec![]` alone can't tell Rust what `T` is.
    pub(crate) fn emit_constructor_with_turbofish(&self, name: &str, type_args: &[Type], args: &[Arg]) -> String {
        let result = self.emit_constructor_inner(name, args);
        if let Some(rest) = result.strip_prefix(name) {
            let ty_args_s: Vec<String> = type_args.iter().map(|t| self.emit_type(t)).collect();
            return format!("{}::<{}>{}", name, ty_args_s.join(", "), rest);
        }
        result
    }

    pub(crate) fn emit_constructor_inner(&self, name: &str, args: &[Arg]) -> String {
        // Result constructors: `Ok(v)` / `Err(e)` are Rust built-ins, not struct types.
        if name == "Ok" || name == "Err" {
            let args_s = self.emit_args(args);
            return format!("{}({})", name, args_s);
        }
        // Non-fn type alias resolving to a qualified type: construct via the alias.
        // e.g. `AP(3, 4)` where `AP = APoint'owned` (Box<APoint>) → `Box::new(APoint::new(3, 4))`.
        // e.g. `ANode(99)` where `ANode = ATree'auto` (Rc<ATree>) → `Rc::new(ATree::new(99))`.
        if let Some(resolved) = self.non_fn_type_aliases.get(name) {
            let resolved = resolved.clone();
            match &resolved {
                Type::Qualified(inner, q) if q.is_owned_or_new() => {
                    if let Type::Named(inner_name) = inner.as_ref() {
                        let inner_s = self.emit_constructor_inner(inner_name, args);
                        // Managed mode: wrap in Arc<std::sync::Mutex<T>> or RefCell<T>
                        if self.is_managed_owned_user(&resolved) {
                            return match self.config.threading {
                                crate::transpiler::ThreadingMode::Multi =>
                                    format!("Arc::new(std::sync::Mutex::new({}))", inner_s),
                                crate::transpiler::ThreadingMode::Single =>
                                    format!("RefCell::new({})", inner_s),
                            };
                        }
                        return format!("Box::new({})", inner_s);
                    }
                }
                Type::Qualified(inner, OwnerQual::Shared) => {
                    if let Type::Named(inner_name) = inner.as_ref() {
                        let inner_s = self.emit_constructor_inner(inner_name, args);
                        return match self.config.threading {
                            crate::transpiler::ThreadingMode::Single => format!("Rc::new({})", inner_s),
                            crate::transpiler::ThreadingMode::Multi  => format!("Arc::new({})", inner_s),
                        };
                    }
                }
                Type::Qualified(inner, OwnerQual::Inline) => {
                    if let Type::Named(inner_name) = inner.as_ref() {
                        return self.emit_constructor_inner(inner_name, args);
                    }
                }
                Type::Named(inner_name) => {
                    // Simple named alias (e.g. `ADog2 = ADog`) → emit inner constructor.
                    let inner_name = inner_name.clone();
                    return self.emit_constructor_inner(&inner_name, args);
                }
                _ => {}
            }
        }
        // Newtype wrapper: `UserId(42)` → `UserId(42)` (tuple struct constructor).
        if self.newtype_types.contains(name) {
            let arg_s = if let Some(a) = args.first() {
                // String newtypes have inner type `String`; emit_expr_owned converts
                // string literals from `&str` → `"...".to_string()`.
                let inner = self.newtype_inner.get(name).cloned().unwrap_or_default();
                if inner == "String" {
                    // Newtype inner is String (owned); convert literals directly without Arc.
                    match &a.value.kind {
                        ExprKind::Str(s) => format!("\"{}\".to_string()", escape_str(s)),
                        ExprKind::StringInterp(_) => self.emit_expr(&a.value),
                        _ => {
                            // Variable or expression: may be Arc<str> — unwrap to String.
                            let raw = self.emit_expr(&a.value);
                            format!("(*{}).clone()", raw)
                        }
                    }
                } else {
                    self.emit_expr(&a.value)
                }
            } else {
                "Default::default()".to_string()
            };
            return format!("{}({})", name, arg_s);
        }
        if args.is_empty() {
            // Stdlib collection constructors need turbofish to avoid "type annotations needed".
            match name {
                "HashSet" => return "HashSet::<isize>::new()".into(),
                "HashMap" => return "HashMap::<Arc<str>, isize>::new()".into(),
                _ => {}
            }
            // No-field struct: emit `Struct {}` instead of `Struct::new()`.
            if self.struct_fields.get(name).map(|f| f.is_empty()).unwrap_or(false) {
                return format!("{} {{}}", name);
            }
            // If the struct has an init body, call ::new() — filling in defaults if available.
            if self.struct_has_init_body.contains(name) {
                if let Some(defaults) = self.struct_init_defaults.get(name).cloned() {
                    let def_args: Vec<String> = defaults.iter().filter_map(|d| d.clone()).collect();
                    return format!("{}::new({})", name, def_args.join(", "));
                }
                return format!("{}::new()", name);
            }
            // Struct has fields but no init body — use struct literal with defaults.
            if let Some(fields) = self.struct_fields.get(name).cloned() {
                if !fields.is_empty() {
                    let init_defaults = self.struct_init_defaults.get(name).cloned().unwrap_or_default();
                    let lit_fields: Vec<String> = fields.iter().enumerate()
                        .filter_map(|(i, (fname, fty))| {
                            let key = format!("{}::{}", name, fname);
                            if let Some((is_copy, _, default_val)) = self.transient_fields.get(&key) {
                                let init = if *is_copy {
                                    format!("std::cell::Cell::new({})", default_val)
                                } else {
                                    format!("std::cell::RefCell::new({})", default_val)
                                };
                                Some(format!("{}: {}", fname, init))
                            } else if let Some(Some(def)) = init_defaults.get(i) {
                                Some(format!("{}: {}", fname, def))
                            } else if matches!(fty, Type::Optional(_)) {
                                Some(format!("{}: None", fname))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if lit_fields.len() == fields.len() || !lit_fields.is_empty() {
                        return format!("{} {{ {} }}", name, lit_fields.join(", "));
                    }
                }
            }
            // Enum variant with no fields (unit variant).
            if let Some(enum_name) = self.enum_variants.get(name) {
                return format!("{}::{}", enum_name, name);
            }
            return format!("{}::new()", name);
        }
        // Struct spread: `Name(..base, field = override, ...)` → `Name { field: override, ..base }`
        // Rust struct update syntax requires the `..base` to be last.
        let has_spread = args.iter().any(|a| a.spread);
        if has_spread {
            let spread_exprs: Vec<String> = args.iter()
                .filter(|a| a.spread)
                .map(|a| self.emit_expr(&a.value))
                .collect();
            let labeled_fields: Vec<String> = args.iter()
                .filter(|a| a.label.is_some())
                .map(|a| {
                    let label = a.label.as_ref().unwrap();
                    let field_ty = self.struct_fields.get(name)
                        .and_then(|fs| fs.iter().find(|(n, _)| n == label))
                        .map(|(_, ty)| ty);
                    let val = self.emit_let_value(field_ty, &a.value);
                    format!("{}: {}", label, val)
                })
                .collect();
            // Combine: explicit fields first, then spread bases.
            // Use `.clone()` so that spreading the same base twice doesn't
            // move it on the first use and leave it inaccessible on the second.
            let mut parts = labeled_fields;
            parts.extend(spread_exprs.iter().map(|e| format!("..{}.clone()", e)));
            return format!("{} {{ {} }}", name, parts.join(", "));
        }

        // If all args are labeled (explicit label or closure-style `|field| expr`) → struct literal.
        // In Boring, `Struct(field: expr)` is parsed as a single-param closure |field| expr.
        // A bare `_` fill-rest marker is also allowed here — it carries no label/value
        // of its own and is filtered out below; it only flags `has_default_rest`.
        let has_default_rest = args.iter().any(|a| a.default_rest);
        let all_labeled = args.iter().all(|a| {
            a.default_rest || a.label.is_some() || matches!(&a.value.kind, ExprKind::Closure(params, _, _, _, _) if params.len() == 1)
        });
        if all_labeled {
            let mut fields: Vec<String> = args.iter()
                .filter(|a| !a.default_rest)
                .map(|a| {
                    // Determine the field label: explicit label or closure-style param name.
                    let label: String = if let Some(l) = &a.label {
                        l.clone()
                    } else if let ExprKind::Closure(params, _, _, _, _) = &a.value.kind {
                        params[0].name.clone()
                    } else { unreachable!() };
                    let label = &label;
                    // When arg is a closure-style labeled arg `|field| expr`, unwrap to get the value.
                    let eff_value: &Expr = if let ExprKind::Closure(params, _, body, _, _) = &a.value.kind {
                        if params.len() == 1 {
                            if let ClosureBody::Expr(e) = body { e.as_ref() } else { &a.value }
                        } else { &a.value }
                    } else { &a.value };
                    // Look up declared field type for proper Optional/string coercion
                    let field_ty = self.struct_fields.get(name)
                        .and_then(|fs| fs.iter().find(|(n, _)| n == label))
                        .map(|(_, ty)| ty);
                    let mutex_key = format!("{}::{}", name, label);
                    // Same wrapping regardless of whether the qualifier is explicit (`T'actor`)
                    // or inferred from usage — both populate these sets identically.
                    let val = if self.struct_mutex_fields.contains(&mutex_key) {
                        // If the value is already an actor/rc variable, just clone the Rc pointer.
                        let already_rc = matches!(&eff_value.kind, ExprKind::Var(v)
                            if self.var_mutex_types.contains(v.as_str()) || self.rc_vars.contains(v.as_str()));
                        if already_rc {
                            let raw = self.emit_expr(eff_value);
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Multi => format!("Arc::clone(&{})", raw),
                                crate::transpiler::ThreadingMode::Single => format!("Rc::clone(&{})", raw),
                            }
                        } else {
                        let inner_ty = field_ty.and_then(Self::mutex_inner);
                        let raw = self.emit_let_value(inner_ty, eff_value);
                        match self.config.threading {
                            crate::transpiler::ThreadingMode::Multi =>
                                self.emit_actor_new(&raw),
                            crate::transpiler::ThreadingMode::Single =>
                                format!("Rc::new(RefCell::new({}))", raw),
                        }
                        }
                    } else if self.struct_mutex_task_fields.contains(&mutex_key) {
                        let already_rc = matches!(&eff_value.kind, ExprKind::Var(v)
                            if self.var_mutex_task_types.contains(v.as_str()) || self.rc_vars.contains(v.as_str()));
                        if already_rc {
                            let raw = self.emit_expr(eff_value);
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Multi => format!("Arc::clone(&{})", raw),
                                crate::transpiler::ThreadingMode::Single => format!("Rc::clone(&{})", raw),
                            }
                        } else {
                            let inner_ty = field_ty.and_then(Self::mutex_inner);
                            let raw = self.emit_let_value(inner_ty, eff_value);
                            self.emit_actor_task_new(&raw)
                        }
                    } else if self.struct_rwlock_fields.contains(&mutex_key) {
                        let already_rc = matches!(&eff_value.kind, ExprKind::Var(v)
                            if self.var_rwlock_types.contains(v.as_str()) || self.rc_vars.contains(v.as_str()));
                        if already_rc {
                            let raw = self.emit_expr(eff_value);
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Multi => format!("Arc::clone(&{})", raw),
                                crate::transpiler::ThreadingMode::Single => format!("Rc::clone(&{})", raw),
                            }
                        } else {
                            let inner_ty = field_ty.and_then(Self::rwlock_inner);
                            let raw = self.emit_let_value(inner_ty, eff_value);
                            self.emit_guard_new(&raw)
                        }
                    } else if self.struct_rwlock_task_fields.contains(&mutex_key) {
                        let already_rc = matches!(&eff_value.kind, ExprKind::Var(v)
                            if self.var_rwlock_task_types.contains(v.as_str()) || self.rc_vars.contains(v.as_str()));
                        if already_rc {
                            let raw = self.emit_expr(eff_value);
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Multi => format!("Arc::clone(&{})", raw),
                                crate::transpiler::ThreadingMode::Single => format!("Rc::clone(&{})", raw),
                            }
                        } else {
                            let inner_ty = field_ty.and_then(Self::rwlock_inner);
                            let raw = self.emit_let_value(inner_ty, eff_value);
                            self.emit_guard_task_new(&raw)
                        }
                    } else if self.recursive_fields.contains(&mutex_key) {
                        // Recursive struct field — wrap in Box::new() at construction site.
                        let raw = self.emit_let_value(field_ty, eff_value);
                        if matches!(field_ty, Some(Type::Optional(_))) {
                            format!("{}.map(Box::new)", raw)
                        } else {
                            format!("Box::new({})", raw)
                        }
                    } else {
                        self.emit_let_value(field_ty, eff_value)
                    };
                    format!("{}: {}", label, val)
                })
                .collect();
            // Append transient fields that weren't provided by the user.
            let provided: std::collections::HashSet<String> = args.iter()
                .filter_map(|a| {
                    a.label.clone().or_else(|| {
                        if let ExprKind::Closure(params, _, _, _, _) = &a.value.kind {
                            if params.len() == 1 { Some(params[0].name.clone()) } else { None }
                        } else { None }
                    })
                })
                .collect();
            for (key, (is_copy, _, default_val)) in &self.transient_fields {
                if let Some(field_name) = key.strip_prefix(&format!("{}::", name)) {
                    if !provided.contains(field_name) {
                        let init = if *is_copy {
                            format!("std::cell::Cell::new({})", default_val)
                        } else {
                            format!("std::cell::RefCell::new({})", default_val)
                        };
                        fields.push(format!("{}: {}", field_name, init));
                    }
                }
            }
            // Append var T'task fields missing from the call with a Mutex-wrapped default.
            for key in &self.struct_mutex_fields.clone() {
                if let Some(field_name) = key.strip_prefix(&format!("{}::", name)) {
                    if !provided.contains(field_name) {
                        let init = match self.config.threading {
                            crate::transpiler::ThreadingMode::Multi =>
                                self.emit_actor_new("Default::default()"),
                            crate::transpiler::ThreadingMode::Single =>
                                "Rc::new(RefCell::new(Default::default()))".to_string(),
                        };
                        fields.push(format!("{}: {}", field_name, init));
                    }
                }
            }
            // Same, for the 'actor'task / 'guard / 'guard'task variants.
            for key in &self.struct_mutex_task_fields.clone() {
                if let Some(field_name) = key.strip_prefix(&format!("{}::", name)) {
                    if !provided.contains(field_name) {
                        fields.push(format!("{}: {}", field_name, self.emit_actor_task_new("Default::default()")));
                    }
                }
            }
            for key in &self.struct_rwlock_fields.clone() {
                if let Some(field_name) = key.strip_prefix(&format!("{}::", name)) {
                    if !provided.contains(field_name) {
                        fields.push(format!("{}: {}", field_name, self.emit_guard_new("Default::default()")));
                    }
                }
            }
            for key in &self.struct_rwlock_task_fields.clone() {
                if let Some(field_name) = key.strip_prefix(&format!("{}::", name)) {
                    if !provided.contains(field_name) {
                        fields.push(format!("{}: {}", field_name, self.emit_guard_task_new("Default::default()")));
                    }
                }
            }
            // Append regular optional/T'auto/T'weak fields not provided — default to None.
            if let Some(known_fields) = self.struct_fields.get(name).cloned() {
                for (fname, fty) in &known_fields {
                    if !provided.contains(fname.as_str()) {
                        // Skip transient and mutex/rwlock fields already handled above.
                        let tkey = format!("{}::{}", name, fname);
                        if self.transient_fields.contains_key(&tkey)
                            || self.struct_mutex_fields.contains(&tkey)
                            || self.struct_mutex_task_fields.contains(&tkey)
                            || self.struct_rwlock_fields.contains(&tkey)
                            || self.struct_rwlock_task_fields.contains(&tkey)
                        {
                            continue;
                        }
                        // Optional fields (including Optional<Qualified<...>>) default to None.
                        if matches!(fty, Type::Optional(_)) {
                            fields.push(format!("{}: None", fname));
                        } else if has_default_rest {
                            // `_` present: prefer the field's own declared `= expr` default
                            // (e.g. `float scale = 1.0`) over the blanket `Default::default()`
                            // tail below, which would silently reset it to the type's zero
                            // value instead. Only plain fields reach here — transient/mutex/
                            // rwlock fields were already filled (with their own defaults) above.
                            if let Some(def) = self.struct_field_defaults.get(&tkey).cloned() {
                                let val = self.emit_let_value(Some(fty), &def);
                                fields.push(format!("{}: {}", fname, val));
                            }
                        }
                    }
                }
            }
            // `_` fill-rest marker: any field not already emitted above (including
            // plain fields with no declared default, and fields of a type Boring
            // doesn't own at all, e.g. an external Bevy struct) is picked up from
            // `Default::default()`.
            if has_default_rest {
                fields.push("..Default::default()".to_string());
            }
            format!("{} {{ {} }}", name, fields.join(", "))
        } else {
            // Positional args: if struct fields are known and no explicit new() exists
            // (e.g. generic structs), emit a struct literal using fields in declaration order.
            // Otherwise fall back to ::new(args).

            // If the struct has an init with a body, route to ::new(args) — the body may
            // set computed fields (e.g. `self.area = 3.14 * r * r`) that can't be in a literal.
            if self.struct_has_init_body.contains(name) {
                // Fill in default args for any omitted trailing params.
                let mut all_args: Vec<String> = args.iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                if let Some(defaults) = self.struct_init_defaults.get(name).cloned() {
                    for def in defaults.iter().skip(all_args.len()).flatten() {
                        all_args.push(def.clone());
                    }
                }
                let args_s = all_args.join(", ");
                return format!("{}::new({})", name, args_s);
            }

            if let Some(fields) = self.struct_fields.get(name) {
                if !fields.is_empty() && args.len() <= fields.len() {
                    // Check if the struct has an init (new() function); if so, use ::new().
                    // Heuristic: if all args are positional and fields are known, use struct literal.
                    let lit_fields: Vec<String> = args.iter().enumerate()
                        .map(|(i, a)| {
                            let (fname, fty) = &fields[i];
                            // `name: expr` in Boring struct call is parsed as a single-param
                            // closure `|name| expr` when `:` is used. If the closure param
                            // matches the field name, unwrap and treat as a labeled value.
                            let effective_value = if let ExprKind::Closure(params, _, body, _, _) = &a.value.kind {
                                if params.len() == 1 && params[0].name == *fname {
                                    match body {
                                        ClosureBody::Expr(e) => e.as_ref(),
                                        _ => &a.value,
                                    }
                                } else { &a.value }
                            } else { &a.value };
                            let rec_key = format!("{}::{}", name, fname);
                            let val = if self.recursive_fields.contains(&rec_key) {
                                let raw = self.emit_let_value(Some(fty), effective_value);
                                if matches!(fty, Type::Optional(_)) {
                                    format!("{}.map(Box::new)", raw)
                                } else {
                                    format!("Box::new({})", raw)
                                }
                            } else {
                                self.emit_let_value(Some(fty), effective_value)
                            };
                            format!("{}: {}", fname, val)
                        })
                        .collect();
                    // Fill missing fields with defaults if any.
                    // Priority: init param defaults > transient Cell defaults > Optional → None.
                    let init_defaults = self.struct_init_defaults.get(name).cloned().unwrap_or_default();
                    let extra_fields: Vec<String> = fields.iter().skip(args.len()).enumerate()
                        .filter_map(|(offset_i, (fname, fty))| {
                            let param_idx = args.len() + offset_i;
                            let key = format!("{}::{}", name, fname);
                            if let Some((is_copy, _, default_val)) = self.transient_fields.get(&key) {
                                let init = if *is_copy {
                                    format!("std::cell::Cell::new({})", default_val)
                                } else {
                                    format!("std::cell::RefCell::new({})", default_val)
                                };
                                Some(format!("{}: {}", fname, init))
                            } else if let Some(Some(def)) = init_defaults.get(param_idx) {
                                // Init param had an explicit default value.
                                Some(format!("{}: {}", fname, def))
                            } else if matches!(fty, Type::Optional(_)) {
                                Some(format!("{}: None", fname))
                            } else {
                                None
                            }
                        })
                        .collect();
                    let mut all_fields = lit_fields;
                    all_fields.extend(extra_fields);
                    return format!("{} {{ {} }}", name, all_fields.join(", "));
                }
            }
            // Enum variant with positional args.
            if let Some(enum_name) = self.enum_variants.get(name) {
                let args_s = self.emit_args(args);
                return format!("{}::{}({})", enum_name, name, args_s);
            }
            // Fallback: literal tuple-struct call for hand-verified external types (see
            // `Transpiler::KNOWN_EXTERNAL_TUPLE_STRUCTS`'s doc in src/transpiler/mod.rs),
            // else ::new(args) (requires a new() function to exist).
            let args_s = self.emit_args(args);
            if self.is_known_external_tuple_struct(name) {
                return format!("{}({})", name, args_s);
            }
            // Semaphore::new and similar tokio primitives expect usize, but Boring's
            // `uint` maps to u64. Cast the first argument to usize automatically.
            if matches!(name, "Semaphore" | "RwLock") {
                return format!("{}::new({} as usize)", name, args_s);
            }
            format!("{}::new({})", name, args_s)
        }
    }

    /// Resolves the struct type name of an expression that denotes a struct-typed
    /// value (a plain var, `self`, or a chain of field accesses) — used by
    /// `infer_float_width` to look field types up in `struct_fields`. Mirrors the
    /// `var_struct_types` / `var_types` fallback already used by field-access
    /// emission elsewhere in this file (e.g. `field_is_arc` above).
    pub(crate) fn resolve_struct_name(&self, e: &Expr) -> Option<String> {
        let named = |t: &Type| -> Option<String> {
            match t {
                Type::Named(n) => Some(n.clone()),
                Type::Qualified(inner, _) => match inner.as_ref() {
                    Type::Named(n) => Some(n.clone()),
                    _ => None,
                },
                _ => None,
            }
        };
        match &e.kind {
            ExprKind::Var(v) if v == "self" => self.self_type.clone(),
            ExprKind::Var(v) => self.var_struct_types.get(v.as_str()).cloned()
                .or_else(|| self.var_types.get(v.as_str()).and_then(named)),
            ExprKind::Field(base, field) => {
                let struct_name = self.resolve_struct_name(base)?;
                let fields = self.struct_fields.get(struct_name.as_str())?;
                let (_, fty) = fields.iter().find(|(fname, _)| fname == field)?;
                named(fty)
            }
            _ => None,
        }
    }

    /// `Some("f32")`/`Some("f64")` when the expression's float width can be
    /// determined statically, `None` when it's ambiguous (e.g. an untyped literal)
    /// — used by `math_builtin_float_ty` below.
    fn infer_float_width(&self, e: &Expr) -> Option<&'static str> {
        let mut visiting = std::collections::HashSet::new();
        self.infer_float_width_inner(e, &mut visiting)
    }

    /// `visiting` guards the `var_init_exprs` fallback below against infinite recursion on
    /// self-referential shadowing (`let x = 1.0` then `let x = x * 2.0` — the second
    /// initializer's `Var("x")` refers to the *previous* binding, but `var_init_exprs` is a
    /// flat name→expr map that only remembers the latest one, so without a guard resolving
    /// it would recurse into itself forever instead of just giving up with `None`).
    fn infer_float_width_inner(&self, e: &Expr, visiting: &mut std::collections::HashSet<String>) -> Option<&'static str> {
        let width_of = |t: &Type| -> Option<&'static str> {
            match t {
                Type::Float32 => Some("f32"),
                // Lowercase source spelling (`let float32 a = ...`) parses as
                // `Type::Named` — the transpiler has no separate alias-resolution
                // pass, so this is stored as-is (same reason other var_types
                // lookups throughout this file match both forms).
                Type::Named(n) if n == "float32" || n == "f32" => Some("f32"),
                Type::Float64 => Some("f64"),
                Type::Named(n) if n == "float64" || n == "float" || n == "f64" => Some("f64"),
                _ => None,
            }
        };
        match &e.kind {
            ExprKind::Var(v) => {
                if let Some(w) = self.var_types.get(v.as_str()).and_then(width_of) {
                    return Some(w);
                }
                // Implicit self field access inside a struct method — a bare `x` inside
                // `req`/`def` on a struct with a `float32 x` field lowers to `self.x`
                // (see the `ExprKind::Var` case in `emit_expr` above), so its width must
                // be looked up the same way `emit_expr` resolves it, or a method body like
                // `sqrt(x * x + y * y)` falls through to the f64 default below even though
                // `x`/`y` are float32 fields.
                if !self.known_local_vars.contains(v.as_str()) {
                    if let Some(struct_name) = &self.self_type {
                        if let Some(fields) = self.struct_fields.get(struct_name.as_str()) {
                            if let Some((_, fty)) = fields.iter().find(|(fname, _)| fname == v) {
                                return width_of(fty);
                            }
                        }
                    }
                }
                // A `let`-bound local whose type wasn't recorded in `var_types` (e.g. an
                // unannotated `let rad = direction * 0.017453292` — arithmetic/unary-op
                // initializers aren't tracked into `var_types` at all, see emit_let.rs) —
                // fall back to the initializer expression's own inferred width, if we
                // still have it on hand.
                if !visiting.insert(v.clone()) {
                    return None;
                }
                let result = self.var_init_exprs.get(v.as_str()).and_then(|init| self.infer_float_width_inner(init, visiting));
                visiting.remove(v.as_str());
                result
            }
            // Struct field access — e.g. `v.x` where `Velocity.x` is declared `float32`.
            ExprKind::Field(base, field) => {
                let struct_name = self.resolve_struct_name(base)?;
                let fields = self.struct_fields.get(struct_name.as_str())?;
                let (_, fty) = fields.iter().find(|(fname, _)| fname == field)?;
                width_of(fty)
            }
            // Arithmetic/unary propagate the width of whichever operand has a
            // known one (Boring requires matching widths to mix float32/float64
            // in arithmetic — docs/float-width-types.md §3 — so if one side is
            // known the other, if also known, necessarily agrees).
            ExprKind::BinOp(_, l, r) => self.infer_float_width_inner(l, visiting).or_else(|| self.infer_float_width_inner(r, visiting)),
            ExprKind::UnaryOp(_, inner) => self.infer_float_width_inner(inner, visiting),
            ExprKind::Index(base, _) => match &base.kind {
                ExprKind::Var(v) => match self.var_types.get(v.as_str()) {
                    Some(Type::Array(elem)) => width_of(elem),
                    _ => None,
                },
                _ => None,
            },
            ExprKind::MethodCall(recv, method, _) => {
                let struct_name = self.resolve_struct_name(recv)?;
                let key = format!("{}::{}", struct_name, method);
                self.struct_method_return_types.get(&key).and_then(width_of)
            }
            ExprKind::Call(callee, args) => {
                if let ExprKind::Var(fname) = &callee.kind {
                    match fname.as_str() {
                        "float32" => return Some("f32"),
                        "float" | "float64" => return Some("f64"),
                        // Nested builtin math call: `sqrt(sqrt(x))` etc. lowers to
                        // `(<arg> as ty).sqrt()`, so its own result width is exactly
                        // its argument's inferred width.
                        "sqrt" | "abs" | "floor" | "ceil" | "round" | "sin" | "cos" | "tan"
                            | "asin" | "acos" | "atan" | "atan2" | "exp" | "tanh"
                            | "log" | "log2" | "log10" | "pow"
                            if !args.is_empty() => return self.infer_float_width_inner(&args[0].value, visiting),
                        _ => {}
                    }
                    self.fn_return_types.get(fname.as_str()).and_then(width_of)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// "f32" when `e` is known to be `float32`-typed, else the default "f64" —
    /// used by the free-function math builtins below (`sqrt(x)`, `abs(x)`, …) so
    /// a `float32` argument (including one buried in a struct field or an
    /// arithmetic expression over one, not just a bare variable) doesn't get
    /// silently widened to `f64` and back (docs/float-width-types.md —
    /// float32/float64 are distinct runtime types with the same method surface,
    /// so the cast target just needs to match the argument's own width instead of
    /// always assuming 64-bit).
    fn math_builtin_float_ty(&self, e: &Expr) -> &'static str {
        self.infer_float_width(e).unwrap_or("f64")
    }

    pub(crate) fn emit_builtin_call(&self, name: &str, args: &[Arg]) -> String {
        match name {
            // some(x) → Some(x): wrap a value in Option
            "some" if args.len() == 1 => {
                let v = self.emit_expr_owned(&args[0].value);
                format!("Some({})", v)
            }
            "print" | "println" => {
                self.emit_print_call(true, args)
            }
            "write" | "eprint" => {
                self.emit_print_call(false, args)
            }
            "format" => {
                self.emit_print_call_named("format", args)
            }
            // Log-level builtins: `[LEVEL] message` on stderr, matching the interpreter
            // exactly (`call_display_builtin`, methods.rs: `eprintln!("[INFO] {}", ...)`).
            // Previously mapped to `log::info!`/etc., which requires a registered logger
            // backend (`env_logger::init()` or similar) to print anything at all — the
            // `log` crate's macros are a silent no-op facade otherwise. Nothing in a
            // `boring build` full-project output ever registered one (no `env_logger`
            // dependency, no init call in `main()`), so every `info`/`warn`/`debug`/
            // `trace`/`error` call in a compiled binary silently produced zero output
            // (confirmed via examples/tokio.br running clean with no program output at
            // all). Plain `eprintln!` needs no setup and matches `boring run`'s output.
            "error" | "warn" | "info" | "debug" | "trace" => {
                let call = self.emit_print_call_named("eprintln", args);
                let level = name.to_uppercase();
                match call.find('"') {
                    Some(pos) => format!("{}\"[{}] {}", &call[..pos], level, &call[pos + 1..]),
                    // `emit_print_call_named` only omits the quote for a zero-arg call
                    // (`eprintln!()`) — still worth a level marker on its own.
                    None => format!("eprintln!(\"[{}]\")", level),
                }
            }
            "assert" => {
                if args.len() == 1 {
                    format!("assert!({})", self.emit_expr(&args[0].value))
                } else {
                    let cond = self.emit_expr(&args[0].value);
                    let msg = self.emit_expr(&args[1].value);
                    format!("assert!({}, \"{{:?}}\", {})", cond, msg)
                }
            }
            "assert_eq" => {
                let a = self.emit_expr(&args[0].value);
                let b = self.emit_expr(&args[1].value);
                if args.len() > 2 {
                    let msg = self.emit_expr(&args[2].value);
                    format!("assert_eq!({}, {}, \"{{:?}}\", {})", a, b, msg)
                } else {
                    format!("assert_eq!({}, {})", a, b)
                }
            }
            "assert_neq" => {
                let a = self.emit_expr(&args[0].value);
                let b = self.emit_expr(&args[1].value);
                format!("assert_ne!({}, {})", a, b)
            }
            "panic" => {
                if args.is_empty() {
                    "panic!(\"explicit panic\")".into()
                } else {
                    format!("panic!(\"{{:?}}\", {})", self.emit_expr(&args[0].value))
                }
            }
            "dbg" => {
                if args.is_empty() {
                    "dbg!()".into()
                } else if args.len() == 1 {
                    format!("dbg!({})", self.emit_expr(&args[0].value))
                } else {
                    let args_s: Vec<String> = args.iter().map(|a| self.emit_expr(&a.value)).collect();
                    format!("dbg!({})", args_s.join(", "))
                }
            }
            "todo" => {
                if args.is_empty() {
                    "todo!()".into()
                } else {
                    format!("todo!(\"{{:?}}\", {})", self.emit_expr(&args[0].value))
                }
            }
            "unreachable" => {
                if args.is_empty() {
                    "unreachable!()".into()
                } else {
                    format!("unreachable!(\"{{:?}}\", {})", self.emit_expr(&args[0].value))
                }
            }
            "len" => {
                // For actor (Arc<Mutex<T>>) variables, lock first.
                if let Some(first) = args.first() {
                    if let ExprKind::Var(v) = &first.value.kind {
                        if (self.var_mutex_types.contains(v.as_str()) || self.var_mutex_task_types.contains(v.as_str())) && self.in_async {
                            return format!("{}.len()", self.mutex_var_read(v, v));
                        }
                    }
                }
                let a = self.emit_expr(&args[0].value);
                format!("{}.len()", a)
            }
            // `int(x)`/`uint(x)`/`float(x)` on a string arg must parse, not `as`-cast --
            // Arc<str>/&str can't be cast to a numeric type at all in Rust ("non-primitive
            // cast"). Only numeric args (the common case) get the plain `as` cast.
            "int" | "uint" | "uint8"
                | "int8" | "int16" | "int32" | "int64" | "int128"
                | "uint16" | "uint32" | "uint64" | "uint128"
                if self.is_string_expr(&args[0].value) => {
                let rust_ty = normalize_type_name(name, self.use_rc_str());
                format!("{}.trim().parse::<{}>().unwrap_or(0)", self.emit_expr(&args[0].value), rust_ty)
            }
            "float" | "float64" if self.is_string_expr(&args[0].value) =>
                format!("{}.trim().parse::<f64>().unwrap_or(0.0)", self.emit_expr(&args[0].value)),
            "float32" if self.is_string_expr(&args[0].value) =>
                format!("{}.trim().parse::<f32>().unwrap_or(0.0)", self.emit_expr(&args[0].value)),
            "int" | "uint" | "uint8"
                | "int8" | "int16" | "int32" | "int64" | "int128"
                | "uint16" | "uint32" | "uint64" | "uint128" => {
                let rust_ty = normalize_type_name(name, self.use_rc_str());
                format!("({} as {})", self.emit_expr(&args[0].value), rust_ty)
            }
            "float" | "float64" => format!("({} as f64)", self.emit_expr(&args[0].value)),
            "float32" => format!("({} as f32)", self.emit_expr(&args[0].value)),
            "str"   => {
                // Single non-string arg → conversion.
                // String first arg (with optional extra args) → format like format().
                if let Some(first) = args.first() {
                    if matches!(&first.value.kind, ExprKind::StringInterp(_)) || args.len() >= 2 {
                        return self.emit_print_call_named("format", args);
                    }
                }
                format!("{}.to_string()", self.emit_expr(&args[0].value))
            }
            // Math functions: boring global → Rust method on f64.
            // Cast argument to f64 to avoid "ambiguous numeric type" errors on literals.
            "sqrt" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).sqrt()", self.emit_expr(&args[0].value))
            }
            "abs" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).abs()", self.emit_expr(&args[0].value))
            }
            "floor" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).floor()", self.emit_expr(&args[0].value))
            }
            "ceil" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).ceil()", self.emit_expr(&args[0].value))
            }
            "round" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).round()", self.emit_expr(&args[0].value))
            }
            "sin" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).sin()", self.emit_expr(&args[0].value))
            }
            "cos" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).cos()", self.emit_expr(&args[0].value))
            }
            "tan" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).tan()", self.emit_expr(&args[0].value))
            }
            "asin" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).asin()", self.emit_expr(&args[0].value))
            }
            "acos" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).acos()", self.emit_expr(&args[0].value))
            }
            "atan" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).atan()", self.emit_expr(&args[0].value))
            }
            "atan2"      => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                let y = self.emit_expr(&args[0].value);
                let x = self.emit_expr(&args[1].value);
                format!("({} as {ty}).atan2({} as {ty})", y, x)
            }
            "exp" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).exp()", self.emit_expr(&args[0].value))
            }
            "tanh" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).tanh()", self.emit_expr(&args[0].value))
            }
            "log"        => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).ln()", self.emit_expr(&args[0].value))
            }
            "log2" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).log2()", self.emit_expr(&args[0].value))
            }
            "log10" => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                format!("({} as {ty}).log10()", self.emit_expr(&args[0].value))
            }
            "pow"        => {
                let ty = self.math_builtin_float_ty(&args[0].value);
                let b = self.emit_expr(&args[0].value);
                let e = self.emit_expr(&args[1].value);
                format!("({} as {ty}).powf({} as {ty})", b, e)
            }
            "min"        => {
                if args.len() == 1 {
                    format!("{}.iter().cloned().reduce(f64::min).expect(\"cannot compute min of empty collection\")", self.emit_expr(&args[0].value))
                } else {
                    let a = self.emit_expr(&args[0].value);
                    let b = self.emit_expr(&args[1].value);
                    format!("({}).min({})", a, b)
                }
            }
            "max"        => {
                if args.len() == 1 {
                    format!("{}.iter().cloned().reduce(f64::max).expect(\"cannot compute max of empty collection\")", self.emit_expr(&args[0].value))
                } else {
                    let a = self.emit_expr(&args[0].value);
                    let b = self.emit_expr(&args[1].value);
                    format!("({}).max({})", a, b)
                }
            }
            "sum"        => {
                let a = self.emit_expr(&args[0].value);
                format!("{}.iter().copied().reduce(|acc, v| acc + v).unwrap_or_default()", a)
            }
            "clamp"      => {
                let x  = self.emit_expr(&args[0].value);
                let lo = self.emit_expr(&args[1].value);
                let hi = self.emit_expr(&args[2].value);
                format!("({}).clamp({}, {})", x, lo, hi)
            }
            "bitsToFloat" => format!("(f32::from_bits({} as u32) as f64)", self.emit_expr(&args[0].value)),
            "floatToBits" => format!("(({} as f32).to_bits() as isize)", self.emit_expr(&args[0].value)),
            "sign"       => format!("({}).signum()", self.emit_expr(&args[0].value)),
            "isNaN"      => format!("({}).is_nan()", self.emit_expr(&args[0].value)),
            "isInfinite" => format!("({}).is_infinite()", self.emit_expr(&args[0].value)),
            "readLine"   => {
                // Returns None on EOF, Some(line) otherwise (trim_end strips trailing \n/\r).
                format!("{{ let mut __line = String::new(); let __n = std::io::stdin().read_line(&mut __line).unwrap_or(0); if __n == 0 {{ None }} else {{ Some({}::<str>::from(__line.trim_end_matches('\\n').trim_end_matches('\\r'))) }} }}", self.str_ptr())
            }
            // drop(x) — explicitly releases ownership, maps directly to Rust's drop()
            "drop" => {
                let a = self.emit_expr(&args[0].value);
                format!("drop({})", a)
            }
            // args() — argv-style CLI arguments, C/Python convention: args()[0] is
            // the program name (the binary's own path, exactly as the OS was asked
            // to invoke it — reflects renames/symlinks/aliases), args()[1..] are its
            // real arguments. No `.skip(1)`: unlike `boring run` (interpreter/mod.rs),
            // there is no `boring`-prefix to strip here — `std::env::args()`'s own
            // argv[0] already is this program's name.
            "args" => {
                format!("std::env::args().map(|s| {}::<str>::from(s)).collect::<Vec<_>>()", self.str_ptr())
            }
            // raw_args() — same as args() in compiled binaries (there is no `boring run`
            // prefix to strip, and no implicit `--` filtering either way).
            "raw_args" => {
                format!("std::env::args().map(|s| {}::<str>::from(s)).collect::<Vec<_>>()", self.str_ptr())
            }
            "ord" => {
                let s = self.emit_expr(&args[0].value);
                format!("({}).chars().next().expect(\"ord: empty string\") as isize", s)
            }
            "chr" => {
                let n = self.emit_expr(&args[0].value);
                format!("{}::<str>::from(char::from_u32({} as u32).expect(\"chr: invalid codepoint\").to_string())", self.str_ptr(), n)
            }
            "exit" => {
                let code = self.emit_expr(&args[0].value);
                format!("{{ std::process::exit({} as i32) }}", code)
            }
            // json(v) → serde_json::to_string(&v).unwrap_or_default()
            "json" => {
                self.uses_serde.set(true);
                let a = self.emit_expr(&args[0].value);
                format!("serde_json::to_string(&{}).unwrap_or_default()", a)
            }
            _ => {
                // Look up registered signature for optional-arg coercion
                let args_s = self.emit_args_coerced(name, args);
                format!("{}({})", escape_rust_keyword(name), args_s)
            }
        }
    }

    pub(crate) fn emit_print_call(&self, newline: bool, args: &[Arg]) -> String {
        let macro_name = if newline { "println" } else { "print" };
        self.emit_print_call_named(macro_name, args)
    }

    pub(crate) fn emit_print_call_named(&self, macro_name: &str, args: &[Arg]) -> String {
        if args.is_empty() {
            return format!("{}!()", macro_name);
        }
        // Positional substitution: `print "..{}..", expr, expr2`
        // First arg is a string template where `{}` holes bind to extra args in order.
        // Inline `{name}` holes are interleaved naturally (left-to-right).
        if args.len() >= 2 {
            if let ExprKind::StringInterp(segs) = &args[0].value.kind {
                let positional: Vec<String> = args[1..].iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect();
                let (fmt, combined) = self.build_positional_format(segs, &positional);
                return if combined.is_empty() {
                    format!("{}!(\"{}\")", macro_name, fmt)
                } else {
                    format!("{}!(\"{}\", {})", macro_name, fmt, combined.join(", "))
                };
            }
        }
        // If the single arg is a string interp, unfold it
        if args.len() == 1 {
            if let ExprKind::StringInterp(segs) = &args[0].value.kind {
                let (fmt, extra_args) = self.build_format_string(segs);
                return if extra_args.is_empty() {
                    format!("{}!(\"{}\")", macro_name, fmt)
                } else {
                    format!("{}!(\"{}\", {})", macro_name, fmt, extra_args.join(", "))
                };
            }
            if let ExprKind::Str(s) = &args[0].value.kind {
                return format!("{}!(\"{}\")", macro_name, escape_str(s));
            }
        }
        // General case: println!("{}", arg) or println!("{} {}", a, b)
        // Vec collections use BoringFmt(&v) with "{}" so strings show without debug quotes.
        // HashMap/HashSet fall back to "{:?}" since they have no Display impl.
        // Optional values are unwrapped: `bm1` (Option<T>) → `bm1.as_ref().map_or(...)`.
        let args_with_specs: Vec<(String, &str)> = args.iter().map(|a| {
            let is_optional_var = matches!(&a.value.kind,
                ExprKind::Var(n) if self.optional_vars.contains(n.as_str()));
            // Also detect function calls that return Optional types.
            let is_optional_call = !is_optional_var && matches!(&a.value.kind,
                ExprKind::Call(callee, _) | ExprKind::GenericCall(callee, _, _)
                if matches!(&callee.kind, ExprKind::Var(n)
                    if matches!(self.fn_return_types.get(n.as_str()), Some(Type::Optional(_)))));
            // Same, for a *method* call on a known struct-typed variable, e.g. `print
            // s.peek()` where `peek` is declared `req T? peek()` — found wiring
            // `boring.collections`'s `Stack<T>`/`Queue<T>` (docs/cross-project-code-
            // sharing-gap.md's stdlib work): without this, `print s.pop()` compiled to
            // `println!("{}", s.pop())` and failed (`Option<T>` has no `Display` impl),
            // even though the exact same struct_method_return_types lookup already
            // recognizes this shape for `return`/tail-expression Some()-wrapping
            // elsewhere (emit_flow.rs, emit_stmt.rs).
            let is_optional_method_call = !is_optional_var && !is_optional_call
                && matches!(&a.value.kind, ExprKind::MethodCall(recv, method, _)
                    if matches!(&recv.kind, ExprKind::Var(v)
                        if self.var_struct_types.get(v.as_str()).map(|sty| {
                            self.struct_method_return_types
                                .get(&format!("{}::{}", sty, method))
                                .map(|t| matches!(t, Type::Optional(_)))
                                .unwrap_or(false)
                        }).unwrap_or(false)));
            let is_optional = is_optional_var || is_optional_call || is_optional_method_call;
            let expr_s = if is_optional {
                let v = self.emit_expr(&a.value);
                format!("{}.as_ref().map_or_else(|| \"nil\".to_string(), |v| format!(\"{{}}\", v))", v)
            } else {
                self.emit_expr(&a.value)
            };
            if is_optional {
                return (expr_s, "{}");
            }
            let is_vec_var = matches!(&a.value.kind, ExprKind::Var(n) if self.vec_vars.contains(n.as_str()))
                || self.expr_field_is_array(&a.value);
            let is_col = looks_like_collection(&expr_s)
                || matches!(&a.value.kind, ExprKind::Var(n) if self.collection_vars.contains(n.as_str()))
                || matches!(&a.value.kind, ExprKind::Array(_))
                || self.expr_returns_collection(&a.value);
            let (expr_s, spec) = boring_vec_fmt(expr_s, is_col, is_vec_var);
            (expr_s, spec)
        }).collect();
        let placeholders: String = args_with_specs.iter().map(|(_, s)| *s).collect::<Vec<_>>().join(" ");
        let args_s: Vec<String> = args_with_specs.into_iter().map(|(e, _)| e).collect();
        format!("{}!(\"{}\", {})", macro_name, placeholders, args_s.join(", "))
    }

}
