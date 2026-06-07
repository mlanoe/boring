use super::*;
use super::Transpiler;
use super::helpers::*;

impl Transpiler {
    pub(crate) fn emit_item(&mut self, item: &Item) {
        match item {
            Item::Use(u)    => self.emit_use(u),
            Item::Fn(f) => {
                // Top-level `def TypeName.method():` → emit inside its own `impl TypeName {}`.
                // Multiple impl blocks for the same type are valid in Rust.
                if let Some(type_name) = &f.qualifier.clone() {
                    let prev_self = self.self_type.replace(type_name.clone());
                    self.line(&format!("impl {} {{", type_name));
                    self.indent += 1;
                    self.emit_fn(f, Some(type_name));
                    self.blank();
                    self.indent -= 1;
                    self.line("}");
                    self.blank();
                    self.self_type = prev_self;
                } else {
                    self.emit_fn(f, None);
                }
            }
            Item::Struct(s) => self.emit_struct(s),
            Item::Enum(e)   => self.emit_enum(e),
            Item::Trait(t)  => self.emit_trait(t),
            Item::Ext(e)    => self.emit_ext(e),
            Item::Mod(m)    => self.emit_mod(m),
            Item::Alias(a)  => self.emit_alias(a),
            Item::Let(s)    => self.emit_let(s, false),
            Item::Stmt(s)   => self.emit_stmt(s, false),
        }
    }

    // ── use declarations ─────────────────────────────────────────────────────

    pub(crate) fn emit_use(&mut self, u: &UseDecl) {
        // `use boring_mod.*` / `use boring_mod.x` — `emit_mod` inlines items directly into the
        // current scope (no Rust `mod` block), so any `use` pointing at a Boring module would
        // produce an unresolvable `use boring_mod::*`.  Skip such imports entirely.
        if let Some(root) = u.path.first() {
            if self.boring_mod_names.contains(root.as_str()) {
                return;
            }
            // `use super.*` emitted from inside a Boring `mod` that is being inlined into the
            // top-level file also has no meaning (there is no enclosing Rust module).
            if root == "super" {
                return;
            }
            // Track external crate dependencies for Cargo.toml generation.
            if root == "reqwest" {
                self.uses_reqwest = true;
            }
        }
        let path = u.path.join("::");
        // Filter out items already imported by the standard prelude to avoid
        // "defined multiple times" errors (e.g. `use std.collections.HashMap`).
        let prelude_types = ["HashMap", "HashSet"];
        let filtered_items: Vec<&String> = if path == "std::collections" {
            u.items.iter().filter(|item| !prelude_types.contains(&item.as_str())).collect()
        } else {
            u.items.iter().collect()
        };
        // If all items were filtered out, or the single unqualified import is already covered, skip.
        let full_path_in_prelude = matches!(path.as_str(),
            "std::collections::HashMap" | "std::collections::HashSet"
        );
        if full_path_in_prelude { return; }
        if u.items.is_empty() && full_path_in_prelude { return; }
        let s = if u.glob {
            format!("use {}::*;", path)
        } else if filtered_items.is_empty() {
            // All items were already in prelude — skip entirely
            return;
        } else if filtered_items.len() == 1 {
            format!("use {}::{};", path, filtered_items[0])
        } else {
            format!("use {}::{{{}}};", path, filtered_items.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
        };
        self.line(&s);
    }

    // ── type aliases ──────────────────────────────────────────────────────────

    pub(crate) fn emit_alias(&mut self, a: &AliasDecl) {
        if a.newtype {
            return self.emit_newtype(a);
        }
        // Only emit type aliases for non-function types.
        // Function type aliases (`use F as int(str) throws`) require a trait object
        // or a concrete type, which is not straightforward in Rust.
        // For simple primitive / named aliases, emit `type Alias = RustType;`.
        match &a.ty {
            Type::Fn(_, _, _, _, _) => {
                // Fn types: stored as fn_type_aliases and expanded inline at usage sites.
                // No code emitted here — the alias is invisible in Rust output.
            }
            _ => {
                // Store non-fn type aliases so call-site argument coercion can resolve them.
                self.non_fn_type_aliases.insert(a.name.clone(), a.ty.clone());
                let rust_ty = self.emit_type(&a.ty);
                self.line(&format!("type {} = {};", a.name, rust_ty));
            }
        }
    }

    pub(crate) fn emit_newtype(&mut self, a: &AliasDecl) {
        let inner_raw = self.emit_type(&a.ty);
        // Newtype structs hold owned values — `&str` and `Arc<str>` both become `String`
        // so there are no lifetime parameters on the struct definition.
        let inner = match inner_raw.as_str() {
            "&str" | "Arc<str>" => "String".to_string(),
            other => other.to_string(),
        };
        // Register before emitting so constructors in the same file resolve correctly.
        self.newtype_types.insert(a.name.clone());
        self.newtype_inner.insert(a.name.clone(), inner.clone());

        // Derive Copy for primitive inner types (stack values, no heap allocation).
        // f64 cannot be Eq/Hash (NaN); String cannot be Copy.
        let derives = match inner.as_str() {
            "f64" | "f32" => "#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]",
            "u64" | "i64" | "u32" | "i32" | "u8" | "i8" | "u16" | "i16" |
            "usize" | "isize" | "bool" =>
                "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]",
            _ => "#[derive(Debug, Clone, PartialEq, Eq, Hash)]",
        };
        self.line(derives);
        self.line(&format!("struct {}({});", a.name, inner));
        self.line("");
        // From<inner> for Name — construction
        self.line(&format!("impl From<{}> for {} {{", inner, a.name));
        self.indent += 1;
        self.line(&format!("fn from(v: {}) -> Self {{ {}(v) }}", inner, a.name));
        self.indent -= 1;
        self.line("}");
        // From<Name> for inner — unwrap
        self.line(&format!("impl From<{}> for {} {{", a.name, inner));
        self.indent += 1;
        self.line(&format!("fn from(v: {}) -> {} {{ v.0 }}", a.name, inner));
        self.indent -= 1;
        self.line("}");
        // Display — delegate to inner
        self.line(&format!("impl std::fmt::Display for {} {{", a.name));
        self.indent += 1;
        self.line("fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {");
        self.indent += 1;
        self.line("write!(f, \"{}\", self.0)");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("");
    }

    // ── mod ───────────────────────────────────────────────────────────────────

    pub(crate) fn emit_mod(&mut self, m: &ModDecl) {
        // In Boring, mod items are accessible from the outer scope (duck-typed namespacing).
        // Emit items directly without a Rust mod block to keep them in scope.
        for item in &m.items {
            self.emit_item(item);
            self.blank();
        }
    }

    // ── Functions ─────────────────────────────────────────────────────────────

    pub(crate) fn emit_fn(&mut self, f: &FnDecl, self_ty: Option<&str>) {
        // Register param types for ALL top-level functions — including native stdlib ones
        // (wait, timeout, …) — so that DotIdent type hints can be resolved at call sites:
        //   wait(.fromSecs(1))  →  wait(Duration::from_secs(1))
        // Use `entry().or_insert` so user-defined overloads win over native declarations.
        if self_ty.is_none() {
            let param_types: Vec<Type> = f.params.iter()
                .filter_map(|p| p.ty.clone())
                .collect();
            self.fn_sigs.entry(f.name.clone()).or_insert(param_types);
            let param_names: Vec<String> = f.params.iter()
                .map(|p| p.name.clone())
                .collect();
            self.fn_param_names.entry(f.name.clone()).or_insert(param_names);
        }

        if f.is_native {
            // Native task functions (wait, timeout, …) must still be in task_fns so that
            // body_calls_task_fn() detects them and auto-promotes callers to async.
            if f.task && self_ty.is_none() {
                self.task_fns.insert(f.name.clone());
            }
            return;
        }

        // Attributes → #[...]
        for attr in &f.attrs {
            let args_s = if attr.args.is_empty() {
                String::new()
            } else {
                format!("({})", attr.args.join(", "))
            };
            self.line(&format!("#[{}{}]", attr.name, args_s));
        }
        // Auto-detect: a function that declares T'actor locals or consumes stream functions
        // will generate .await calls and therefore implicitly needs to be async.
        // Also implicit async when a param is a `task` closure (fn type with task=true).
        let has_task_fn_param = f.params.iter().any(|p| {
            let ty = p.ty.as_ref();
            let resolved = ty.and_then(|t| if let Type::Named(n) = t {
                self.fn_type_aliases.get(n.as_str())
            } else { None }).or(ty);
            matches!(resolved, Some(Type::Fn(_, _, _, true, _)))
        });
        let implicit_async = !f.task && self_ty.is_none()
            && (body_has_actor_binding(&f.body)
                || body_has_stream_for(&f.body, &self.stream_fns)
                || body_has_channel_or_task(&f.body)
                || has_task_fn_param);
        // For `main` specifically: also promote to async when the body calls any
        // task function (wait, timeout, user-defined task fns, etc.).
        // This lets `def main():` work without the `task` qualifier — the compiler
        // infers async automatically, so there is no caller that needs to know.
        let main_needs_async = !f.task
            && f.name == "main"
            && self_ty.is_none()
            && body_calls_task_fn(&f.body, &self.task_fns);
        let is_async = f.task || implicit_async || main_needs_async;
        // Register implicitly-async functions in task_fns so callers can add .await.
        if (implicit_async || main_needs_async) && self_ty.is_none() {
            self.task_fns.insert(f.name.clone());
        }

        // `main` with any async content needs the tokio runtime entry point.
        if is_async && f.name == "main" && self_ty.is_none() {
            self.line("#[tokio::main]");
        }

        // Visibility + async keyword + fn
        let vis = if f.is_pub { "pub " } else { "" };
        let async_kw = if is_async { "async " } else { "" };

        // Collect lifetimes used implicitly in the signature (params + return type).
        // These are emitted as bare `'a` params without any explicit <'a> declaration in Boring.
        let mut implicit_lifetimes: Vec<String> = Vec::new();
        for p in &f.params {
            if let Some(ty) = &p.ty { collect_lifetimes(ty, &mut implicit_lifetimes); }
        }
        if let Some(ret) = &f.return_ty { collect_lifetimes(ret, &mut implicit_lifetimes); }

        // Type parameters — add Clone bound so Vec<T>[i].clone() works.
        // Also incorporate where_clause constraints (e.g. `T as Display` → `T: std::fmt::Display`).
        // Lifetime parameters (starting with `'`) never receive trait bounds.
        // Explicit <'a> declarations in f.type_params are merged with implicit ones.
        //
        // Important: if a single-letter uppercase name that the parser auto-collected as a
        // type parameter (e.g. `A`, `X`) actually refers to a known concrete enum or struct,
        // it must NOT be emitted as a generic `<X: Clone>`.  Filter those names out here.
        let concrete_type_names: std::collections::HashSet<String> = {
            let mut s: std::collections::HashSet<String> = self.struct_fields.keys().cloned().collect();
            // All enum type names (values of enum_variants map) are also concrete.
            s.extend(self.enum_variants.values().cloned());
            s
        };
        // Rebuild type_params without names that are concrete types.
        // We work from a filtered copy so the rest of the function stays unchanged.
        let f_type_params_filtered: Vec<String> = f.type_params.iter()
            .filter(|p| !concrete_type_names.contains(p.as_str()))
            .cloned()
            .collect();
        let f = &FnDecl { type_params: f_type_params_filtered, ..f.clone() };
        let all_type_params: Vec<String> = {
            let mut merged = implicit_lifetimes.iter()
                .map(|lt| format!("'{}", lt))
                .filter(|lt| !f.type_params.contains(lt))
                .collect::<Vec<_>>();
            merged.extend(f.type_params.iter().cloned());
            // Filter out type params already declared at the impl<...> level
            // (set by emit_ext for generic impl blocks). Re-declaring them on the
            // method would cause "type parameter shadows outer type parameter" errors.
            merged.retain(|p| !self.impl_type_params.contains(p));
            merged
        };
        // Determine the return type of the task fn param (for the Future bound).
        let task_fn_output_ty: Option<String> = if has_task_fn_param {
            f.params.iter().find_map(|p| {
                let ty = p.ty.as_ref();
                let resolved = ty.and_then(|t| if let Type::Named(n) = t {
                    self.fn_type_aliases.get(n.as_str())
                } else { None }).or(ty);
                if let Some(Type::Fn(ret, _, throws, true, _)) = resolved {
                    let base = ret.as_ref().map(|r| self.emit_type(r)).unwrap_or("()".to_string());
                    let r = if *throws {
                        format!("Result<{}, Box<dyn std::error::Error + Send + Sync>>", base)
                    } else { base };
                    Some(r)
                } else { None }
            })
        } else { None };
        // Determine which type params are used in struct-pattern matches (need Any bound).
        // Identify which type params are used in generic-struct pattern matches.
        // A match `match s: MCircle(_): ...` where `s: S` (type param) needs Any bound.
        // Build: (1) map from param variable name → type-param name (e.g. "s" → "S"),
        //        (2) set of variable names typed with a type param,
        //        (3) which of those type params appear in struct match arms.
        let struct_match_type_params: std::collections::HashSet<String> = {
            let struct_names: std::collections::HashSet<String> =
                self.struct_fields.keys().cloned().collect();
            let type_param_set: std::collections::HashSet<&str> =
                f.type_params.iter().map(|s| s.as_str()).collect();
            // Collect variable names (params) whose type is a function type parameter.
            // Param types can be either Type::Named("S") or Type::TypeParam("S").
            let mut tp_var_names: std::collections::HashSet<String> = std::collections::HashSet::new();
            for param in &f.params {
                if let Some(ty) = &param.ty {
                    let ty_name = match ty {
                        Type::Named(n) => Some(n.as_str()),
                        Type::TypeParam(n) => Some(n.as_str()),
                        _ => None,
                    };
                    if let Some(tn) = ty_name {
                        if type_param_set.contains(tn) {
                            tp_var_names.insert(param.name.clone());
                        }
                    }
                }
            }
            // Also include type param names themselves (in case the var name == type param name).
            for p in &f.type_params { tp_var_names.insert(p.clone()); }

            if tp_var_names.is_empty() || !stmts_have_struct_match(&f.body, &tp_var_names, &struct_names) {
                std::collections::HashSet::new()
            } else {
                // Return the set of type param names that correspond to matched variables.
                // We check each type param: is there a var with that type that participates?
                f.type_params.iter()
                    .filter(|p| {
                        // Find vars typed as this param and check if those vars are used in struct matches.
                        let this_tp_vars: std::collections::HashSet<String> = f.params.iter()
                            .filter(|param| matches!(&param.ty, Some(Type::Named(tn)) if tn == *p)
                                || matches!(&param.ty, Some(Type::TypeParam(tn)) if tn == *p))
                            .map(|param| param.name.clone())
                            .collect();
                        // Also include if var name == type param name.
                        let mut check_vars = this_tp_vars;
                        check_vars.insert((**p).clone());
                        stmts_have_struct_match(&f.body, &check_vars, &struct_names)
                    })
                    .cloned()
                    .collect()
            }
        };

        let type_params = if all_type_params.is_empty() && task_fn_output_ty.is_none() {
            String::new()
        } else {
            let mut bounded: Vec<String> = all_type_params.iter()
                .map(|p| {
                    // Lifetime parameters: emit bare `'a`, no bounds.
                    if p.starts_with('\'') { return p.clone(); }
                    // Const generic parameters: emit `const N: usize`, no Clone bound.
                    if p.starts_with('$') { return emit_generic_param(p); }
                    // Collect extra trait bounds from the where clause.
                    let extra: Vec<String> = f.where_clause.iter()
                        .filter(|(tp, _)| tp == p)
                        .map(|(_, trait_name)| map_trait_bound(trait_name))
                        .collect();
                    // Type params used in generic struct-pattern matches need Any bound.
                    let needs_any = struct_match_type_params.contains(p.as_str());
                    let base = if needs_any { "Clone + std::any::Any" } else { "Clone" };
                    if extra.is_empty() {
                        format!("{}: {}", p, base)
                    } else {
                        format!("{}: {} + {}", p, base, extra.join(" + "))
                    }
                })
                .collect();
            // Add a generic type parameter for the Future returned by task fn params.
            if let Some(output_ty) = task_fn_output_ty {
                bounded.push(format!("__BoringFut__: std::future::Future<Output={}>", output_ty));
            }
            format!("<{}>", bounded.join(", "))
        };

        // Build the full parameter list: [&[mut] self,] params...
        // Detect cancellable task functions: if this is a task def and uses Task.cancelled(),
        // register it and inject an implicit __task_cancel parameter.
        let is_cancellable = f.task && self_ty.is_none() && stmts_use_task_cancelled(&f.body);
        if is_cancellable {
            self.cancellable_task_fns.insert(f.name.clone());
            self.uses_tokio_util.set(true);
        }
        let params_s: Vec<String> = f.params.iter().map(|p| self.emit_param(p)).collect();
        let all_params = match self_ty {
            Some(_) => {
                // For struct methods, use &mut self (def) or &self (req/task).
                // Task methods are called via Arc<Self> which provides &Self only (no DerefMut).
                // The T'actor Arc<Mutex<T>> wrapping is handled at call sites via .lock().await.
                // Exception: inside a trait impl block, mutating task methods must match the
                // trait signature (which uses &mut self), since calls go through .lock().await.
                // Enum methods: enums don't have mutable fields, so always use &self even for
                // `def` (mutating) methods — otherwise the method can't be called on non-mut vars.
                let is_enum_self = self_ty.map(|t| {
                    self.enum_variant_fields.keys().any(|k| k.starts_with(&format!("{}::", t)))
                }).unwrap_or(false);
                // Validate: an external `task fn` declaration (`def T.method()` outside the
                // struct body) relies on the receiver being accessed through `Arc<Self>`.
                // That is guaranteed only when `T` is arc-qualified (`'task`, `'actor`, or
                // `'guard`) somewhere in the program. Inline struct methods (declared inside
                // the struct body, `f.qualifier = None`) are exempt — they may be async for
                // other reasons (calling task fns) without requiring Arc semantics.
                if f.task && f.qualifier.is_some() && !self.inside_trait_impl && !is_enum_self {
                    let sname = self_ty.unwrap_or("");
                    if !sname.is_empty() && !self.arc_qualified_types.contains(sname) {
                        eprintln!(
                            "error: `task fn` method '{}::{}' requires '{}' to be used with a \
                             'task, 'actor, or 'guard qualifier at least once in the program \
                             (no arc-qualified binding found)",
                            sname, f.name, sname
                        );
                        std::process::exit(1);
                    }
                }
                let self_s = if f.mutating && (!f.task || self.inside_trait_impl) && !is_enum_self {
                    "&mut self"
                } else {
                    "&self"
                };
                if params_s.is_empty() {
                    self_s.to_string()
                } else {
                    format!("{}, {}", self_s, params_s.join(", "))
                }
            }
            None => {
                // For cancellable task fns, prepend the implicit cancel token parameter.
                if is_cancellable {
                    let cancel_param = "__task_cancel: tokio_util::sync::CancellationToken".to_string();
                    if params_s.is_empty() {
                        cancel_param
                    } else {
                        format!("{}, {}", cancel_param, params_s.join(", "))
                    }
                } else {
                    params_s.join(", ")
                }
            }
        };

        // ── Stream functions: wrap body in async_stream::stream! / try_stream! ──
        if f.stream {
            return self.emit_stream_fn(f, &type_params, &all_params);
        }

        // Return type
        let base_ret = f.return_ty.as_ref()
            .map(|t| {
                // If the return type is a known trait name, use `impl TraitName` (static dispatch).
                // Dynamic dispatch is expressed explicitly with `Type::Dyn` → Box<dyn Trait>.
                if let Type::Named(n) = t {
                    if self.trait_method_names.contains_key(n.as_str()) {
                        return format!("impl {}", normalize_type_name(n));
                    }
                }
                self.emit_type(t)
            })
            .unwrap_or_else(|| "()".to_string());
        // Helper: check if declared return type is the bare `Result` name → infer generics.
        let declared_result = matches!(&f.return_ty, Some(Type::Named(n)) if n == "Result");
        let ret_ty = if f.throws {
            // Always use Box<dyn std::error::Error + Send + Sync> to stay consistent with trait signatures
            // (emit_fn_sig always uses Box<dyn Error> for trait method declarations).
            // This ensures trait + impl return types always match.
            format!("Result<{}, Box<dyn std::error::Error + Send + Sync>>", base_ret)
        } else if declared_result {
            // Infer Result<T, E> from Ok/Err return statements.
            // Build a param-type map for variable type lookup inside Ok/Err arguments.
            // Note: boring uses lowercase type aliases (`int`, `uint`, etc.) which the
            // parser stores as `Type::Named("int")`, not as the primitive `Type::Int`.
            let param_tys: std::collections::HashMap<String, String> = f.params.iter()
                .filter_map(|p| {
                    let ty_s = match p.ty.as_ref()? {
                        Type::Int  | Type::Uint => Some("i64".to_string()),
                        Type::Float => Some("f64".to_string()),
                        Type::Bool  => Some("bool".to_string()),
                        Type::Str   => Some("Arc<str>".to_string()),
                        Type::Named(n) => match n.as_str() {
                            "int" | "i64" | "i32" | "i16" | "i8" | "isize" => Some("i64".to_string()),
                            "uint" | "u64" | "u32" | "u16" | "u8" | "usize" => Some("u64".to_string()),
                            "float" | "f64" | "f32" => Some("f64".to_string()),
                            "bool"   => Some("bool".to_string()),
                            "string" | "str" => Some("Arc<str>".to_string()),
                            _ => None,
                        },
                        _ => None,
                    }?;
                    Some((p.name.clone(), ty_s))
                })
                .collect();
            let (ok_ty, err_ty) = body_returns_result(&f.body, &param_tys);
            let t = ok_ty.as_deref().unwrap_or("()");
            let e = err_ty.as_deref().unwrap_or("()");
            format!("Result<{}, {}>", t, e)
        } else {
            base_ret
        };

        // Use mangled name for overloaded functions/methods so each variant compiles as a distinct Rust fn.
        let is_free_overload = self_ty.is_none() && self.overloaded_fn_names.contains(&f.name);
        let is_method_overload = self_ty.is_some() && {
            let key = format!("{}::{}", self_ty.expect("invariant: guarded by is_some() check above"), f.name);
            self.overloaded_method_keys.contains(&key)
        };
        let rust_fn_name = if is_free_overload || is_method_overload {
            mangle_overload_name(&f.name, &f.params)
        } else {
            f.name.clone()
        };
        let sig = format!(
            "{}{}fn {}{}({}) -> {}",
            vis, async_kw, rust_fn_name, type_params, all_params, ret_ty
        );

        // Body
        if f.body.is_empty() {
            self.line(&format!("{} {{}}", sig));
            return;
        }

        self.line(&format!("{} {{", sig));
        self.indent += 1;
        let prev_throws   = self.in_throws;
        let prev_async         = self.in_async;
        let prev_fn_returns_void = self.fn_returns_void;
        // Track the current function's type parameters for generic match detection.
        let prev_fn_type_params = std::mem::replace(
            &mut self.current_fn_type_params,
            f.type_params.iter().cloned().collect(),
        );
        // A function is "void" if it has no declared return type, or returns () / Nil / Void,
        // and is not a throws function (which wraps in Result<(),...>).
        let is_void_ret = !f.throws && match &f.return_ty {
            None => true,
            Some(Type::Void) | Some(Type::Nil) => true,
            Some(Type::Named(n)) if n == "void" || n == "nil" => true,
            _ => false,
        };
        self.fn_returns_void = is_void_ret;
        let prev_task_vars     = std::mem::take(&mut self.task_vars);
        let prev_arc_vars      = std::mem::take(&mut self.arc_vars);
        let prev_weak_vars     = std::mem::take(&mut self.weak_vars);
        // var_struct_types / var_mutex_types / var_rwlock_types accumulate across the whole program;
        // save/restore so variable names from one function body don't shadow another.
        let prev_var_struct_types  = std::mem::take(&mut self.var_struct_types);
        let prev_var_mutex_types   = std::mem::take(&mut self.var_mutex_types);
        let prev_var_rwlock_types  = std::mem::take(&mut self.var_rwlock_types);
        let prev_known_local_vars  = std::mem::take(&mut self.known_local_vars);
        let prev_optional_vars    = std::mem::take(&mut self.optional_vars);
        let prev_var_types        = std::mem::take(&mut self.var_types);
        let prev_string_vars      = std::mem::take(&mut self.string_vars);
        // Pre-seed known_local_vars, var_struct_types, var_types, and var_mutex_types from params.
        for p in &f.params {
            self.known_local_vars.insert(p.name.clone());
            if let Some(ty) = &p.ty {
                self.var_types.insert(p.name.clone(), ty.clone());
                // Track string params for string concatenation detection.
                if Self::is_string_type(ty) {
                    self.string_vars.insert(p.name.clone());
                }
                // T'actor params → var_mutex_types (Arc<Mutex<T>> method dispatch).
                if Self::is_mutex_binding(p.mutable, ty) {
                    self.var_mutex_types.insert(p.name.clone());
                    self.arc_vars.insert(p.name.clone());
                }
                // T'guard params → var_rwlock_types (Arc<RwLock<T>> method dispatch).
                if Self::is_rwlock_binding(p.mutable, ty) {
                    self.var_rwlock_types.insert(p.name.clone());
                    self.arc_vars.insert(p.name.clone());
                }
                // Arc-qualified and string params are also Arc<T> — must be cloned
                // before capture in `async move {}` blocks.
                if Self::is_arc_qualified(ty) || Self::is_string_type(ty) {
                    self.arc_vars.insert(p.name.clone());
                }
                if matches!(ty, Type::Optional(_)) {
                    self.optional_vars.insert(p.name.clone());
                }
                // Task fn params: calling them produces a Future → needs .await.
                // Throws fn params: calling them returns Result → needs `?` in throws context.
                {
                    let resolved = if let Type::Named(n) = ty {
                        self.fn_type_aliases.get(n.as_str()).unwrap_or(ty)
                    } else { ty };
                    if matches!(resolved, Type::Fn(_, _, _, true, _)) {
                        self.task_vars.insert(p.name.clone());
                    }
                    // Track non-task fn params that throw: `int f() throws` → Result-returning closure
                    if matches!(resolved, Type::Fn(_, _, true, false, _)) {
                        self.throws_fn_params.insert(p.name.clone());
                    }
                }
                // Plain Named struct params → var_struct_types (direct field/method dispatch).
                // Smart-pointer-qualified types (T'auto, T'weak, etc.) are excluded here;
                // they have their own dispatch path.
                if let crate::ast::Type::Named(tname) = ty {
                    if self.struct_fields.contains_key(tname.as_str()) {
                        self.var_struct_types.insert(p.name.clone(), tname.clone());
                    }
                    // Newtype params: `fn f(UserId id)` → track so `id as uint` → `id.0`.
                    if self.newtype_types.contains(tname.as_str()) {
                        self.var_newtype_type.insert(p.name.clone(), tname.clone());
                    }
                }
            }
        }
        self.in_throws = f.throws;
        self.in_async  = is_async;
        let prev_fn_return_ty = self.fn_return_ty.clone();
        self.fn_return_ty = f.return_ty.clone();
        let prev_in_cancellable_fn = self.in_cancellable_fn;
        if is_cancellable {
            self.in_cancellable_fn = true;
            // Register __task_cancel as a known local so emit_expr can resolve it.
            self.known_local_vars.insert("__task_cancel".to_string());
        }
        self.emit_body(&f.body);
        self.in_cancellable_fn = prev_in_cancellable_fn;
        self.in_throws         = prev_throws;
        self.fn_returns_void   = prev_fn_returns_void;
        self.fn_return_ty      = prev_fn_return_ty;
        self.task_vars         = prev_task_vars;
        self.arc_vars          = prev_arc_vars;
        self.weak_vars         = prev_weak_vars;
        self.var_struct_types  = prev_var_struct_types;
        self.var_mutex_types   = prev_var_mutex_types;
        self.var_rwlock_types  = prev_var_rwlock_types;
        self.known_local_vars  = prev_known_local_vars;
        self.optional_vars     = prev_optional_vars;
        self.var_types         = prev_var_types;
        self.string_vars       = prev_string_vars;
        self.in_async          = prev_async;
        self.current_fn_type_params = prev_fn_type_params;
        self.indent -= 1;
        self.line("}");
    }

    /// Emit a `stream` function: returns `impl Stream<Item = T>` and wraps the body
    /// in `async_stream::stream! { ... }` (or `try_stream!` when `throws` is set).
    pub(crate) fn emit_stream_fn(&mut self, f: &FnDecl, type_params: &str, all_params: &str) {
        let vis = if f.is_pub { "pub " } else { "" };
        // Stream items are owned values — map `string`/`str` → `String` (not Arc<str>/&str).
        let base_item_ty = f.return_ty.as_ref()
            .map(|t| self.emit_stream_item_type(t))
            .unwrap_or_else(|| "()".to_string());

        let (macro_name, item_ty) = if f.throws {
            let err_ty = f.throws_ty.as_ref()
                .map(|t| self.emit_type(t))
                .unwrap_or_else(|| "Box<dyn std::error::Error + Send + Sync>".into());
            ("try_stream", format!("Result<{}, {}>", base_item_ty, err_ty))
        } else {
            ("stream", base_item_ty.clone())
        };

        let sig = format!(
            "{}fn {}{}({}) -> impl futures_core::Stream<Item = {}>",
            vis, f.name, type_params, all_params, item_ty
        );
        self.line(&format!("{} {{", sig));
        self.indent += 1;
        self.line(&format!("async_stream::{}! {{", macro_name));
        self.indent += 1;

        // Emit the body with throws/async context (try_stream! supports `?` and `.await`)
        let prev_throws = self.in_throws;
        let prev_async  = self.in_async;
        let prev_fn_returns_void = self.fn_returns_void;
        let prev_fn_return_ty = self.fn_return_ty.clone();
        let prev_known  = std::mem::take(&mut self.known_local_vars);
        self.in_throws = f.throws;
        self.in_async  = true; // stream bodies are implicitly async (macros support .await)
        self.fn_returns_void = true; // yield is not a return
        self.fn_return_ty = None;
        for p in &f.params { self.known_local_vars.insert(p.name.clone()); }

        self.emit_body(&f.body);

        self.in_throws         = prev_throws;
        self.in_async          = prev_async;
        self.fn_returns_void   = prev_fn_returns_void;
        self.fn_return_ty      = prev_fn_return_ty;
        self.known_local_vars  = prev_known;

        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    /// Returns a borrow-compatible key expression for `HashMap<Arc<str>, V>::get()`.
    /// With `Arc<str>`, `.get()` takes `&str` directly.
    /// - String literals   → `"lit"`                (&str)
    /// - chars_vars (char) → `&ch.to_string()`      (&String coerces to &str)
    /// - Arc<str> vars     → `&*var`                (&str via Deref)
    pub(crate) fn emit_dict_key_borrow(&self, key: &Expr) -> String {
        match &key.kind {
            ExprKind::Str(s) => format!("\"{}\"", escape_str(s)),
            ExprKind::Var(v) if self.chars_vars.contains(v.as_str()) =>
                format!("&{}.to_string()", v),
            ExprKind::Var(v) => format!("&*{}", v), // Arc<str> → &str via Deref
            _ => {
                let k = self.emit_expr(key);
                format!("&{}", k)
            }
        }
    }

    /// Returns an owned `Arc<str>` key expression for `HashMap::insert()`.
    /// - String literals → `Arc::<str>::from("lit")`
    /// - chars_vars (Rust `char`) → `Arc::<str>::from(ch.to_string())`
    /// - Other vars (Arc<str>) → `var.clone()`
    pub(crate) fn emit_dict_key_owned(&self, key: &Expr) -> String {
        match &key.kind {
            ExprKind::Str(s) => format!("Arc::<str>::from(\"{}\")", escape_str(s)),
            ExprKind::Var(v) if self.chars_vars.contains(v.as_str()) =>
                format!("Arc::<str>::from({}.to_string())", v),
            ExprKind::Var(v) => format!("{}.clone()", v),
            _ => self.emit_expr_owned(key),
        }
    }

    /// Returns true when an expression is a function call whose return type is a collection.
    /// Used to select `{:?}` formatting in println! for HashMap/Vec/HashSet return values.
    pub(crate) fn expr_returns_collection(&self, expr: &Expr) -> bool {
        if let ExprKind::Call(callee, _) = &expr.kind {
            if let ExprKind::Var(fn_name) = &callee.kind {
                if let Some(ty) = self.fn_return_types.get(fn_name.as_str()) {
                    return is_collection_type(Some(ty));
                }
            }
        }
        false
    }

    pub(crate) fn emit_param(&self, p: &Param) -> String {
        // FnMut closure params need `mut` so the closure can be called.
        // `req` function params (Fn) do not need mut.
        let resolved_ty = p.ty.as_ref().and_then(|ty| {
            if let Type::Named(n) = ty { self.fn_type_aliases.get(n.as_str()) } else { None }
        }).or(p.ty.as_ref());
        let is_fnmut = matches!(resolved_ty, Some(Type::Fn(_, _, _, _, req)) if !req);
        let name = if p.mutable || is_fnmut { format!("mut {}", p.name) } else { p.name.clone() };
        match &p.ty {
            Some(ty) if p.variadic => format!("{}: Vec<{}>", name, self.emit_type(ty)),
            Some(ty) => {
                // If the param type is a known trait name, emit `impl TraitName`.
                let ty_s = if let Type::Named(n) = ty {
                    if self.trait_method_names.contains_key(n.as_str()) {
                        format!("impl {}", normalize_type_name(n))
                    } else {
                        self.emit_type(ty)
                    }
                } else {
                    self.emit_type(ty)
                };
                format!("{}: {}", name, ty_s)
            }
            None     => name,
        }
    }

    /// Emit an expression, coercing string literals to `Arc<str>` (not `&str`).
    /// Use this at binding/call/return sites where `Arc<str>` is expected.
    pub(crate) fn emit_expr_owned(&self, expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Str(s) => format!("Arc::<str>::from(\"{}\")", escape_str(s)),
            ExprKind::StringInterp(_) => {
                // format!(...) gives String; wrap in Arc
                format!("Arc::<str>::from({})", self.emit_expr(expr))
            }
            // self.field in owned context → .clone() so Arc<T> fields don't move out of &self
            ExprKind::Field(obj, field) => {
                // Type-level access: delegate to emit_expr which handles Counter::X
                if let ExprKind::Var(type_name) = &obj.kind {
                    if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        let key = format!("{}::{}", type_name, field);
                        if self.struct_type_var_names.contains(&key)
                            || self.struct_type_mut_var_names.contains(&key) {
                            return self.emit_expr(expr);
                        }
                    }
                    // oneshot rx.value → rx.await.unwrap() (receive the single value)
                    if field == "value" && self.oneshot_receivers.contains(type_name.as_str()) {
                        return if self.in_throws || self.in_try_body {
                            format!("{}.await?", type_name)
                        } else {
                            format!("{}.await.expect(\"oneshot channel sender dropped\")", type_name)
                        };
                    }
                    // watch rx.value → current value without waiting
                    if field == "value" && self.watch_receivers.contains(type_name.as_str()) {
                        return format!("{}.borrow().clone()", type_name);
                    }
                    // future.value / future.wait — delegate to emit_expr so throws JoinHandle
                    // vars get the correct .await.unwrap()? treatment (not just .await.unwrap()).
                    if (field == "value" || field == "wait")
                        && type_name != "self"
                        && !self.var_mutex_types.contains(type_name.as_str())
                        && !self.var_rwlock_types.contains(type_name.as_str())
                        && self.task_vars.contains(type_name.as_str())
                    {
                        return self.emit_expr(expr);
                    }
                }
                // Inline TaskWithTimeout: `.value` / `.wait` — delegate to emit_expr which
                // has the full throws-aware .await.unwrap()? logic for this case.
                if (field == "value" || field == "wait")
                    && matches!(&obj.kind, ExprKind::TaskWithTimeout(..))
                {
                    return self.emit_expr(expr);
                }
                let obj_s = self.emit_expr(obj);
                // Mutex var access (owned context): w.field → w.lock().await.field
                if let ExprKind::Var(v) = &obj.kind {
                    if self.var_mutex_types.contains(v.as_str()) {
                        return format!("{}.lock().await.{}", v, field);
                    }
                }
                // RwLock var access (owned context): c.field → c.read().await.field
                if let ExprKind::Var(v) = &obj.kind {
                    if self.var_rwlock_types.contains(v.as_str()) {
                        return format!("{}.read().await.{}", v, field);
                    }
                }
                // Mutex struct field (owned context): self.worker.field → self.worker.lock().await.field
                if let ExprKind::Field(inner_obj, mutex_field) = &obj.kind {
                    if let ExprKind::Var(v) = &inner_obj.kind {
                        if v == "self" {
                            let key = self.self_type.as_deref()
                                .map(|t| format!("{}::{}", t, mutex_field));
                            if let Some(k) = key {
                                if self.struct_mutex_fields.contains(&k) {
                                    return format!("self.{}.lock().await.{}", mutex_field, field);
                                }
                            }
                        }
                    }
                }
                // RwLock struct field (owned context): self.data.field → self.data.read().await.field
                if let ExprKind::Field(inner_obj, rwlock_field) = &obj.kind {
                    if let ExprKind::Var(v) = &inner_obj.kind {
                        if v == "self" {
                            let key = self.self_type.as_deref()
                                .map(|t| format!("{}::{}", t, rwlock_field));
                            if let Some(k) = key {
                                if self.struct_rwlock_fields.contains(&k) {
                                    return format!("self.{}.read().await.{}", rwlock_field, field);
                                }
                            }
                        }
                    }
                }
                // Transient field read (owned context): use Cell.get() or RefCell.borrow().clone()
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
                // Don't apply map_field to known user struct fields (same logic as emit_expr).
                let field_s = if let ExprKind::Var(v) = &obj.kind {
                    let is_user_field = (v == "self")
                        .then(|| self.self_type.as_deref())
                        .flatten()
                        .and_then(|t| self.struct_fields.get(t))
                        .map(|fields| fields.iter().any(|(fname, _)| fname == field))
                        .unwrap_or(false);
                    if is_user_field { field.as_str() } else { map_field(field) }
                } else {
                    map_field(field)
                };
                if obj_s == "self" && !field_s.ends_with(')') {
                    format!("{}.{}.clone()", obj_s, field_s)
                } else {
                    format!("{}.{}", obj_s, field_s)
                }
            }
            // Method calls that produce &str (e.g. `.trim()`) need Arc::<str>::from(x.to_string())
            // when used in an owned string (Arc<str>) context.
            ExprKind::MethodCall(_, method, _)
                if matches!(method.as_str(), "trim") =>
            {
                let s = self.emit_expr(expr);
                format!("Arc::<str>::from({}.to_string())", s)
            }
            // Struct variable in owned position: clone to avoid moving out.
            // In Boring, structs are copy-by-value semantics; Rust requires explicit .clone().
            ExprKind::Var(v) if self.var_struct_types.contains_key(v.as_str())
                && !self.arc_vars.contains(v.as_str())
                && !self.var_mutex_types.contains(v.as_str()) =>
            {
                format!("{}.clone()", self.emit_expr(expr))
            }
            _ => self.emit_expr(expr),
        }
    }

    /// Emit a function body, wrapping the last expression in Ok() for throws functions.
    /// Handles `defer` statements: collects them and emits in LIFO order before the
    /// final return value.
    pub(crate) fn emit_body(&mut self, stmts: &[Stmt]) {
        if stmts.is_empty() { return; }
        // Collect defer bodies (preserve order; will emit LIFO at end).
        let defers: Vec<&[Stmt]> = stmts.iter().filter_map(|s| {
            if let Stmt::Defer(d) = s { Some(d.as_slice()) } else { None }
        }).collect();
        let non_defers: Vec<&Stmt> = stmts.iter().filter(|s| !matches!(s, Stmt::Defer(_))).collect();

        if defers.is_empty() {
            let last_idx = non_defers.len().saturating_sub(1);
            for (i, stmt) in non_defers.iter().enumerate() {
                self.emit_stmt(stmt, i == last_idx);
            }
            // For throws functions whose last statement is NOT an expression (e.g. a for loop),
            // add implicit `Ok(())` so the function satisfies `Result<(), E>`.
            if self.in_throws {
                let last_stmt = non_defers.last();
                let needs_ok = match last_stmt {
                    Some(Stmt::Expr(_)) | Some(Stmt::Return(_)) | Some(Stmt::Throw(_)) => false,
                    _ => true,
                };
                if needs_ok { self.line("Ok(())"); }
            }
            return;
        }

        // With defers: wrap non-defer body in an immediately-invoked closure so that
        // early returns (guard, explicit return, throw) are all captured. Then run defers LIFO.
        //
        // Special case: function with only defers (no non-defer body) — emit defers and return.
        if non_defers.is_empty() {
            for defer_body in defers.iter().rev() {
                for ds in *defer_body {
                    self.emit_stmt(ds, false);
                }
            }
            return;
        }

        // Determine the return type annotation for the inner closure.
        // For throws functions: `Result<T, Box<dyn std::error::Error + Send + Sync>>`
        // For void functions: `()` — no `let __deferred_ret = ...` needed
        let fn_returns_void = self.fn_returns_void;
        let in_throws = self.in_throws;

        if fn_returns_void && !in_throws {
            // Void function: just emit body then defers (no capture needed).
            // Early returns in Boring void functions don't return a value.
            self.line("{");
            self.indent += 1;
            let last_idx = non_defers.len().saturating_sub(1);
            for (i, stmt) in non_defers.iter().enumerate() {
                self.emit_stmt(stmt, i == last_idx);
            }
            self.indent -= 1;
            self.line("}");
            for defer_body in defers.iter().rev() {
                for ds in *defer_body {
                    self.emit_stmt(ds, false);
                }
            }
            return;
        }

        // Non-void function with defers.
        //
        // If there are no early returns in the non-defer body (no `return`, `throw`,
        // or `guard`), we can emit the body inline — all body stmts first (non-tail),
        // then defers LIFO, then the last body statement as the tail (return value).
        // This avoids the closure-wrapper borrow issue where a var moved into the
        // closure is inaccessible to defer bodies that follow.
        let non_defer_stmts_owned: Vec<Stmt> =
            non_defers.iter().map(|s| (*s).clone()).collect();
        let has_early_return = body_has_early_return(&non_defer_stmts_owned);

        if !has_early_return {
            // Inline path: emit all but last stmt as non-tail, defers LIFO, then last as tail.
            let body_len = non_defers.len();
            for (i, stmt) in non_defers.iter().enumerate() {
                if i + 1 < body_len {
                    self.emit_stmt(stmt, false);
                }
                // The last stmt is emitted AFTER the defers (below).
            }
            // Defers in LIFO order
            for defer_body in defers.iter().rev() {
                for ds in *defer_body {
                    self.emit_stmt(ds, false);
                }
            }
            // Emit last stmt as tail (implicit return value).
            if let Some(last) = non_defers.last() {
                self.emit_stmt(last, true);
            }
            return;
        }

        // Non-void function with defers AND early returns: wrap body in closure
        // so that early returns are captured, then run defers after.
        let ret_ty = if in_throws {
            if let Some(ty) = &self.fn_return_ty {
                let inner = self.emit_type(ty);
                format!("Result<{}, Box<dyn std::error::Error + Send + Sync>>", inner)
            } else {
                "Result<(), Box<dyn std::error::Error + Send + Sync>>".into()
            }
        } else if let Some(ty) = &self.fn_return_ty {
            self.emit_type(ty)
        } else {
            "()".into()
        };

        self.line(&format!("let __deferred_ret = (|| -> {} {{", ret_ty));
        self.indent += 1;
        let last_idx = non_defers.len().saturating_sub(1);
        for (i, stmt) in non_defers.iter().enumerate() {
            self.emit_stmt(stmt, i == last_idx);
        }
        self.indent -= 1;
        self.line("})();");
        // Defers in LIFO order
        for defer_body in defers.iter().rev() {
            for ds in *defer_body {
                self.emit_stmt(ds, false);
            }
        }
        // Return the captured value
        self.line("__deferred_ret");
    }

    /// Emit loop body: all statements get semicolons (no implicit return).
    ///
    /// `known_local_vars` is saved and restored so that variables declared
    /// *inside* a block (if/for/while body) do not leak into the enclosing
    /// scope and incorrectly shadow implicit-self field references.
    pub(crate) fn emit_loop_body(&mut self, stmts: &[Stmt]) {
        // Save the outer set of known locals.  Any `let` emitted inside this
        // block will be inserted into known_local_vars; restoring the saved
        // snapshot on exit removes those inner-scope names from the outer view.
        let saved_locals = self.known_local_vars.clone();
        for stmt in stmts {
            self.emit_stmt(stmt, false);
        }
        self.known_local_vars = saved_locals;
    }

    // ── Types ─────────────────────────────────────────────────────────────────

    /// Returns true when a binding declared `T'actor` should become `Arc<Mutex<T>>`.
    /// Only applies to user-defined struct types (PascalCase names), not primitives or `string`.
    /// The `mutable` parameter is ignored — `T'actor` is always mutex regardless of let/var.
    pub(crate) fn is_mutex_binding(_mutable: bool, ty: &Type) -> bool {
        // Any T'actor binding gets Arc<Mutex<T>> semantics — both user structs (PascalCase)
        // and external types (e.g. mpsc::Receiver, BufReader, …).
        matches!(ty, Type::Qualified(_, OwnerQual::Actor))
    }

    /// Extract the inner `T` from a `T'actor` mutex type.
    pub(crate) fn mutex_inner(ty: &Type) -> Option<&Type> {
        if let Type::Qualified(inner, OwnerQual::Actor) = ty {
            Some(inner)
        } else {
            None
        }
    }

    /// Emit `Arc<tokio::sync::Mutex<T>>` for the inner type.
    pub(crate) fn emit_mutex_type(&self, inner: &Type) -> String {
        format!("Arc<tokio::sync::Mutex<{}>>", self.emit_type(inner))
    }

    /// Returns true when a binding declared `T'guard` should become `Arc<RwLock<T>>`.
    pub(crate) fn is_rwlock_binding(_mutable: bool, ty: &Type) -> bool {
        matches!(ty, Type::Qualified(_, OwnerQual::Guard))
    }

    /// Extract the inner `T` from a `T'guard` rwlock type.
    pub(crate) fn rwlock_inner(ty: &Type) -> Option<&Type> {
        if let Type::Qualified(inner, OwnerQual::Guard) = ty {
            Some(inner)
        } else {
            None
        }
    }

    /// Emit `Arc<tokio::sync::RwLock<T>>` for the inner type.
    pub(crate) fn emit_rwlock_type(&self, inner: &Type) -> String {
        format!("Arc<tokio::sync::RwLock<{}>>", self.emit_type(inner))
    }

    /// If `ty` is `T'task`, `T'actor`, or `T'guard`, return the name of the inner named type.
    /// Used by `pre_scan` to populate `arc_qualified_types`.
    pub(crate) fn arc_inner_type_name(ty: &Type) -> Option<&str> {
        match ty {
            Type::Qualified(inner, OwnerQual::Task | OwnerQual::Actor | OwnerQual::Guard) => {
                if let Type::Named(n) = inner.as_ref() { Some(n.as_str()) } else { None }
            }
            _ => None,
        }
    }

    /// Returns true if the Boring type maps to a `Copy` Rust type.
    /// Determines whether a `transient` field should use `Cell<T>` (Copy) or `RefCell<T>` (!Copy).
    pub(crate) fn is_copy_type(ty: &Type) -> bool {
        match ty {
            Type::Int | Type::Uint | Type::Float | Type::Bool | Type::Nil | Type::Void => true,
            // Lowercase aliases: `int`, `float`, `bool`, `uint` parse as Named in the user source.
            Type::Named(n) => matches!(n.as_str(), "int" | "uint" | "float" | "bool"),
            Type::Optional(inner) => Self::is_copy_type(inner),
            _ => false,
        }
    }

    /// Like `emit_type` but for stream item types: string/str map to `String` (owned),
    /// not Arc<str>/<&str>, since stream items are yielded as owned values.
    pub(crate) fn emit_stream_item_type(&self, ty: &Type) -> String {
        self.emit_type(ty)
    }

    pub(crate) fn emit_type(&self, ty: &Type) -> String {
        match ty {
            Type::Int   => "i64".into(),
            Type::Uint  => "u64".into(),
            Type::Float => "f64".into(),
            Type::Str   => "Arc<str>".into(),
            Type::Bool  => "bool".into(),
            Type::Nil | Type::Void => "()".into(),
            Type::Never => "!".into(),
            Type::Named(n) => {
                // Const-encoded type arg `"$N:usize"` → emit just the name `N` at use sites.
                if let Some(rest) = n.strip_prefix('$') {
                    if let Some((name, _)) = rest.split_once(':') {
                        return name.to_string();
                    }
                }
                // Expand function type aliases (e.g. `use Pure as req int(int)`) inline.
                if let Some(fn_ty) = self.fn_type_aliases.get(n.as_str()) {
                    return self.emit_type(&fn_ty.clone());
                }
                // Inside a trait impl block, associated type names must be qualified as `Self::Name`
                // in return types / parameter types (bare names are not in scope in Rust impl blocks).
                if self.current_trait_assoc_names.contains(n.as_str()) {
                    return format!("Self::{}", n);
                }
                normalize_type_name(n)
            }
            Type::TypeParam(n) => n.clone(),
            Type::Optional(inner) => format!("Option<{}>", self.emit_type(inner)),
            Type::Array(inner)    => format!("Vec<{}>", self.emit_type(inner)),
            Type::Dict(k, v)      => format!("HashMap<{}, {}>", self.emit_type(k), self.emit_type(v)),
            Type::Set(inner)      => format!("HashSet<{}>", self.emit_type(inner)),
            Type::Tuple(elems)    => format!("({})", elems.iter().map(|t| self.emit_type(t)).collect::<Vec<_>>().join(", ")),
            // Index<CollectionType> — opaque collection index; maps to Option<natural-index-type>:
            //   Index<[T]>     → Option<usize>  (array position)
            //   Index<{K:V}>   → Option<K>      (dict key)
            //   Index<{T}>     → Option<usize>  (set position)
            // The Option wrapper is correct because firstIndex()/nextIndex() return nil when exhausted.
            Type::Generic(name, args) if name == "Index" => {
                match args.first() {
                    Some(Type::Dict(k, _)) => format!("Option<{}>", self.emit_type(k)),
                    _ => "Option<usize>".into(),  // array and set both use positional usize
                }
            }
            Type::Generic(name, args) => format!("{}<{}>", name, args.iter().map(|t| self.emit_type(t)).collect::<Vec<_>>().join(", ")),
            Type::SelfAssoc(name) => format!("Self::{}", name),
            // `LinkedList.Index` → resolved concrete type if known, else `LinkedList::Index`.
            // For generic bases `Tree<T>.Node` we use only the base type name for the lookup.
            Type::AssocOf(base, assoc) => {
                let base_name = match base.as_ref() {
                    Type::Named(n)      => n.clone(),
                    Type::Generic(n, _) => n.clone(),
                    _ => return self.emit_type(base),
                };
                // Resolve to the concrete type if available (avoids inherent-assoc-type issues).
                if let Some(assoc_map) = self.struct_assoc_types.get(&base_name) {
                    if let Some(concrete) = assoc_map.get(assoc.as_str()) {
                        return self.emit_type(&concrete.clone());
                    }
                }
                // Fallback: emit `StructName::AssocName` (valid when defined in a trait impl).
                format!("{}::{}", normalize_type_name(&base_name), assoc)
            }
            Type::Impl(inner) => format!("impl {}", self.emit_type(inner)),
            Type::Dyn(inner) => format!("Box<dyn {}>", self.emit_type(inner)),
            Type::Fn(ret, params, throws, task, req) => {
                let ps = params.iter().map(|t| self.emit_type(t)).collect::<Vec<_>>().join(", ");
                let base = ret.as_ref().map(|r| self.emit_type(r)).unwrap_or_else(|| "()".into());
                let r = if *throws { format!("Result<{}, Box<dyn std::error::Error + Send + Sync>>", base) } else { base };
                // For task (async) closures, use `impl Future<Output=T>` in the closure return.
                // This requires the function to declare a generic Fut type parameter.
                // See emit_fn which adds <Fut: Future<Output=T>> when has_task_fn_param.
                let r = if *task {
                    // Use a generic type parameter for the future (avoids RPITIT restriction).
                    // The type parameter name is derived from the return type to avoid conflicts.
                    "__BoringFut__".to_string()
                } else { r };
                // `req` → pure closure (Fn), default → mutating (FnMut)
                let trait_name = if *req { "Fn" } else { "FnMut" };
                format!("impl {}({}) -> {}", trait_name, ps, r)
            }
            Type::Qualified(inner, qual) => match qual {
                OwnerQual::Owned | OwnerQual::Stack => {
                    // Stack is the default in Rust; Owned = Box
                    if matches!(qual, OwnerQual::Owned) {
                        format!("Box<{}>", self.emit_type(inner))
                    } else {
                        self.emit_type(inner)
                    }
                }
                OwnerQual::Copy    => self.emit_type(inner),
                OwnerQual::Const   => {
                    match inner.as_ref() {
                        Type::Str => "&str".into(),
                        // Arrays/collections with 'const → use Arc for thread-safe sharing.
                        Type::Array(_) | Type::Dict(_, _) | Type::Set(_) =>
                            format!("Arc<{}>", self.emit_type(inner)),
                        _ => format!("&'static {}", self.emit_type(inner)),
                    }
                }
                OwnerQual::Auto    => {
                    let inner_s = self.emit_type(inner);
                    let dyn_s = if matches!(inner.as_ref(), Type::Named(n) if self.trait_method_names.contains_key(n.as_str())) {
                        format!("dyn {}", inner_s)
                    } else { inner_s };
                    format!("Rc<{}>", dyn_s)
                }
                OwnerQual::Actor   => {
                    format!("Arc<tokio::sync::Mutex<{}>>", self.emit_type(inner))
                }
                OwnerQual::Guard   => {
                    format!("Arc<tokio::sync::RwLock<{}>>", self.emit_type(inner))
                }
                OwnerQual::Task    => {
                    // String'task = Arc<str> = string (no double-wrap)
                    if matches!(**inner, Type::Str) { "Arc<str>".into() }
                    else {
                        let inner_s = self.emit_type(inner);
                        // Trait objects require `dyn` in Rust.
                        let dyn_s = if matches!(inner.as_ref(), Type::Named(n) if self.trait_method_names.contains_key(n.as_str())) {
                            format!("dyn {}", inner_s)
                        } else { inner_s };
                        format!("Arc<{}>", dyn_s)
                    }
                }
                OwnerQual::Weak    => {
                    // Compound qualifier: `T'auto'weak` → `Weak<T>` (rc),
                    //                    `T'task'weak` → `std::sync::Weak<T>`,
                    //                    `T'actor'weak`→ `std::sync::Weak<Mutex<T>>`
                    // NOTE: use the BASE (innermost named) type, not Arc/Rc<base>.
                    // `Resource'task'weak` = std::sync::Weak<Resource>, not Weak<Arc<Resource>>.
                    match inner.as_ref() {
                        Type::Qualified(base, OwnerQual::Auto) =>
                            format!("Weak<{}>", self.emit_type(base)),
                        Type::Qualified(base, OwnerQual::Task) =>
                            format!("std::sync::Weak<{}>", self.emit_type(base)),
                        Type::Qualified(base, OwnerQual::Actor) =>
                            format!("std::sync::Weak<tokio::sync::Mutex<{}>>", self.emit_type(base)),
                        Type::Qualified(base, OwnerQual::Guard) =>
                            format!("std::sync::Weak<tokio::sync::RwLock<{}>>", self.emit_type(base)),
                        // Inferred `T'weak` (e.g. `let d'weak = c`) — assume rc::Weak.
                        // (For arc-weak inferred bindings, the type annotation is overridden
                        //  in emit_let when the RHS is Arc::downgrade.)
                        _ => format!("Weak<{}>", self.emit_type(inner)),
                    }
                }
                OwnerQual::Borrow  => {
                    if matches!(**inner, Type::Str)
                    || matches!(**inner, Type::Named(ref n) if n == "str" || n == "String")
                    { "&str".into() }
                    else { format!("&{}", self.emit_type(inner)) }
                }
                OwnerQual::BorrowMut    => format!("&mut {}",    self.emit_type(inner)),
                OwnerQual::BorrowOwned  => format!("&Box<{}>",  self.emit_type(inner)),
                OwnerQual::BorrowOption => format!("&Option<{}>", self.emit_type(inner)),
                OwnerQual::BorrowWeak   => format!("&Weak<{}>", self.emit_type(inner)),
                OwnerQual::BorrowAuto   => format!("&Rc<{}>",   self.emit_type(inner)),
                OwnerQual::BorrowTask   => format!("&Arc<{}>",  self.emit_type(inner)),
                OwnerQual::Lifetime(lt) => {
                    // `str` (Type::Str or Named("str")) already resolves to `&str` —
                    // emit `&'a str` directly to avoid the double-reference `&'a &str`.
                    let is_str_slice = matches!(**inner, Type::Str)
                        || matches!(**inner, Type::Named(ref n) if n == "str");
                    if is_str_slice { format!("&'{} str", lt) }
                    else { format!("&'{} {}", lt, self.emit_type(inner)) }
                }
            }
        }
    }

    // ── Structs ───────────────────────────────────────────────────────────────

}
