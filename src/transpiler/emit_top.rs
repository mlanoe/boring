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
            Item::Let(s)    => {
                // Top-level `static let` → emit as `const` at module scope. GPU targets
                // treat every top-level `let` this way too -- see `emit_program_items`'s
                // classification of `Item::Let` (gated on `is_gpu_target`, not on
                // `kernel_decls` being non-empty).
                if s.is_static || self.is_gpu_target {
                    // `_` is not a valid const item type in Rust (E0121) -- unlike a `let`,
                    // where the compiler can infer it. Without an explicit boring type
                    // annotation, infer one from a scalar literal initializer (the only
                    // shape `top_level_let_is_const_safe`, in `emit_program_items`, allows
                    // through without a declared type).
                    let ty_str = s.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| {
                        match s.value.as_ref().map(|v| &v.kind) {
                            Some(ExprKind::Int(_))   => "i64".to_string(),
                            Some(ExprKind::Float(_)) => "f64".to_string(),
                            Some(ExprKind::Bool(_))  => "bool".to_string(),
                            _ => "_".to_string(),
                        }
                    });
                    let val_str = s.value.as_ref().map(|v| self.emit_expr(v)).unwrap_or_else(|| "()".to_string());
                    // GPU targets: uppercase the Rust identifier. A fn parameter whose name
                    // matches an in-scope `const` is parsed by Rust as a refutable pattern
                    // matching that constant's *value*, not a fresh binding -- a hard type
                    // error the moment the types differ (E0308, "interpreted as a constant,
                    // not a new binding"). Kernel constructors commonly use `width`/`height`
                    // as parameter names (see `wgpu::host::emit_kernel_new`), which collides
                    // with exactly the top-level names the docs' own examples use
                    // (`let width = 800`, see examples/game_of_life.br). `map_builtin_var`
                    // does the matching read-side rewrite via `gpu_top_level_const_names`.
                    let const_name = if self.is_gpu_target {
                        s.name.to_uppercase()
                    } else {
                        s.name.clone()
                    };
                    self.line(&format!("const {}: {} = {};", const_name, ty_str, val_str));
                } else {
                    self.emit_let(s, false);
                }
            }
            Item::Stmt(s)   => self.emit_stmt(s, false),
            Item::Kernel(_) => { /* GPU kernel transpilation — not yet implemented */ }
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
        }

        // Check if this resolves to a Boring source file — if so, inline it.
        let rel: std::path::PathBuf = u.path.iter().collect::<std::path::PathBuf>().with_extension("br");
        let candidate = self.source_dir.join(&rel);
        if candidate.exists() {
            self.inline_boring_use(&candidate, u.line, u.col);
            return;
        }

        // Track external crate dependencies for Cargo.toml generation.
        if let Some(root) = u.path.first() {
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
            return;
        } else if filtered_items.len() == 1 {
            format!("use {}::{};", path, filtered_items[0])
        } else {
            format!("use {}::{{{}}};", path, filtered_items.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))
        };
        self.line(&s);
    }

    /// Recursively pre-scan all reachable `.br` use files without marking them as `loaded`.
    /// This populates fn_throws, fn_sigs, struct_fields, etc. for all files before any code is
    /// emitted, so forward references to throws functions work correctly across file boundaries.
    pub(crate) fn deep_pre_scan(&mut self, program: &crate::ast::Program, visited: &mut std::collections::HashSet<std::path::PathBuf>) {
        self.pre_scan(program);
        self.pre_infer_fn_qualifiers(program);
        for item in &program.items {
            if let crate::ast::Item::Use(u) = item {
                let rel: std::path::PathBuf = u.path.iter().collect::<std::path::PathBuf>().with_extension("br");
                let candidate = self.source_dir.join(&rel);
                if !candidate.exists() { continue; }
                let canonical = match candidate.canonicalize() {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                if !visited.insert(canonical.clone()) { continue; }
                let source = match std::fs::read_to_string(&canonical) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let tokens = match crate::lexer::lex(&source) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let sub_program = match crate::parser::parse(tokens) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let prev_dir = self.source_dir.clone();
                if let Some(dir) = canonical.parent() {
                    self.source_dir = dir.to_path_buf();
                }
                self.deep_pre_scan(&sub_program, visited);
                self.source_dir = prev_dir;
            }
        }
    }

    /// Pre-pass: compute every free function's parameter-qualifier inference up front (in
    /// declaration order, independent of emission order) and propagate the results into
    /// `fn_sigs` — mirrors the cross-function propagation `emit_fn` performs after its own
    /// `emit_body` (see the comment there), just run for every function in the file before any
    /// of them are actually emitted. Without this, a caller emitted *earlier* in the file than
    /// a callee (or, thanks to `deep_pre_scan`'s recursion, in another file) would see the
    /// callee's stale unqualified signature and never learn it takes e.g. `&Vec<T>` — this
    /// bit both bare-struct params (rare in practice) and bare array/dict/set params (the
    /// common case in a call-heavy file like a hand-written recursive-descent parser).
    pub(crate) fn pre_infer_fn_qualifiers(&mut self, program: &crate::ast::Program) {
        for item in &program.items {
            let crate::ast::Item::Fn(f) = item else { continue };
            if f.qualifier.is_some() || f.is_native { continue; }

            // Run the inference on a throwaway clone (make_sub), never on `self` directly.
            // infer_qualifiers is normally only ever run as part of a function's real emission,
            // where it's the sole thing observing/mutating this per-function scratch state; here
            // we're running it speculatively, once per function, before anything else in the
            // file has been emitted — doing that against `self` risks leaking half-computed
            // state into unrelated emission (a prior attempt at this corrupted enum-variant
            // handling in a completely unrelated file). The clone already carries everything
            // infer_qualifiers reads (fn_sigs, type_sizes, struct_fields, etc.) from pre_scan.
            let mut sub = self.make_sub();
            sub.fn_current_params = f.params.iter()
                .filter_map(|p| p.ty.as_ref().map(|ty| (p.name.clone(), ty.clone())))
                .collect();
            sub.fn_current_param_lines = f.params.iter().map(|p| (p.name.clone(), p.line)).collect();
            sub.fn_current_param_cols = f.params.iter().map(|p| (p.name.clone(), p.col)).collect();
            sub.fn_current_params_mut = f.params.iter()
                .filter(|p| p.mutable)
                .map(|p| p.name.clone())
                .collect();
            sub.fn_return_ty = f.return_ty.clone();
            sub.in_struct_method = false;

            sub.infer_qualifiers(&f.body);
            let pre_inferred: std::collections::HashMap<String, crate::ast::OwnerQual> = f.params.iter()
                .filter_map(|p| sub.inferred_qualifiers.get(&p.name).map(|q| (p.name.clone(), q.clone())))
                .collect();

            // Update both the plain-name entry and the mangled-name entry (pre_register_fn
            // always registers both — see its comment — regardless of whether this name turns
            // out overloaded). Overload dispatch (emit_call) looks calls up by the mangled key,
            // so it must carry the propagated qualifier too. Do NOT gate this on
            // `self.overloaded_fn_names.contains(&f.name)`: that flag is filled in cumulatively
            // as pre_scan walks files one at a time (deep_pre_scan runs pre_scan then this pass
            // per file before moving to the next), so a same-named overload declared in a
            // not-yet-visited file wouldn't be known about yet, and this update would silently
            // skip the mangled entry the real overloaded call site actually looks up.
            let keys = [f.name.clone(), mangle_overload_name(&f.name, &f.params)];
            for key in keys {
                let Some(sig) = self.fn_sigs.get_mut(&key) else { continue };
                for (i, param) in f.params.iter().enumerate() {
                    let Some(inferred_qual) = pre_inferred.get(&param.name).cloned() else { continue };
                    // Do not propagate 'stack: it's the default fallback, not a real signal
                    // (see the identical exclusion in emit_fn's post-emit_body propagation).
                    if matches!(inferred_qual, crate::ast::OwnerQual::Stack) { continue; }
                    let Some(param_ty) = sig.get_mut(i) else { continue };
                    let already_qualified = matches!(param_ty, crate::ast::Type::Qualified(..))
                        && !matches!(param_ty, crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Owned))
                        && !matches!(param_ty, crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Union(_)));
                    if !already_qualified {
                        *param_ty = crate::transpiler::infer_qualifiers::apply_inferred_qual(param_ty, inferred_qual);
                    }
                }
            }
        }
    }

    fn inline_boring_use(&mut self, path: &std::path::Path, use_line: usize, use_col: usize) {
        let canonical = match path.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                self.push_error(use_line, use_col, format!("cannot resolve '{}': {}", path.display(), e));
                return;
            }
        };
        // Circular / duplicate import guard.
        if self.loaded.contains(&canonical) {
            return;
        }
        self.loaded.insert(canonical.clone());

        let source = match std::fs::read_to_string(&canonical) {
            Ok(s) => s,
            Err(e) => {
                self.push_error(use_line, use_col, format!("cannot read '{}': {}", canonical.display(), e));
                return;
            }
        };
        let tokens = match crate::lexer::lex(&source) {
            Ok(t) => t,
            Err(e) => {
                self.push_error(use_line, use_col, format!("lex error in '{}': {}", canonical.display(), e));
                return;
            }
        };
        let program = match crate::parser::parse(tokens) {
            Ok(p) => p,
            Err(e) => {
                self.push_error(use_line, use_col, format!("parse error in '{}': {}", canonical.display(), e));
                return;
            }
        };
        // Derive module name from file stem (e.g. "eval.br" → "eval").
        let module_name = canonical.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "module".to_string());

        // Buffer this file's output separately so it can be written as its own .rs file.
        // Save the current output buffer, emit into a fresh one, then restore.
        let saved_out = std::mem::take(&mut self.out);
        let prev_source_dir = self.source_dir.clone();
        if let Some(dir) = canonical.parent() {
            self.source_dir = dir.to_path_buf();
        }
        self.emit_program(&program);
        self.source_dir = prev_source_dir;
        let module_code = std::mem::take(&mut self.out);
        self.out = saved_out;

        self.modules.push((module_name, module_code));
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
            "u128" | "i128" | "usize" | "isize" | "bool" =>
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
            let rebindable_flags: Vec<bool> = f.params.iter().map(|p| p.rebindable).collect();
            self.fn_rebindable.entry(f.name.clone()).or_insert(rebindable_flags);
            let mutable_flags: Vec<bool> = f.params.iter().map(|p| p.mutable).collect();
            self.fn_mutable.entry(f.name.clone()).or_insert(mutable_flags);
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
        // Managed mode: add #[track_caller] so panic messages report the call site rather
        // than the panic site inside the function body.
        if self.config.mode == crate::transpiler::TranspileMode::Managed
            && !f.is_native
            && !(f.name == "main" && self_ty.is_none())
        {
            self.line("#[track_caller]");
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
            && (body_has_stream_for(&f.body, &self.stream_fns)
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
            match self.config.threading {
                crate::transpiler::ThreadingMode::Single =>
                    self.line("#[tokio::main(flavor = \"current_thread\")]"),
                crate::transpiler::ThreadingMode::Multi =>
                    self.line("#[tokio::main]"),
            }
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
        // Pre-inference pass: run qualifier inference before emitting params so that
        // unqualified parameters can be emitted with their inferred qualifier.
        // emit_body will re-run it (it clears first), so there is no double-application.
        // We also snapshot the param-level inferred qualifiers here so that the fn_sigs
        // update (below) uses the correct result. emit_body may call infer_qualifiers
        // recursively for nested arm/block bodies (with fn_current_params still set),
        // which would overwrite inferred_qualifiers before the fn_sigs update runs.
        let pre_inferred_param_quals: std::collections::HashMap<String, crate::ast::OwnerQual>;
        {
            let prev = std::mem::take(&mut self.fn_current_params);
            let prev_param_lines = std::mem::take(&mut self.fn_current_param_lines);
            let prev_param_cols = std::mem::take(&mut self.fn_current_param_cols);
            let prev_mut = std::mem::take(&mut self.fn_current_params_mut);
            self.fn_current_params = f.params.iter()
                .filter_map(|p| p.ty.as_ref().map(|ty| (p.name.clone(), ty.clone())))
                .collect();
            self.fn_current_param_lines = f.params.iter()
                .map(|p| (p.name.clone(), p.line))
                .collect();
            self.fn_current_param_cols = f.params.iter()
                .map(|p| (p.name.clone(), p.col))
                .collect();
            self.fn_current_params_mut = f.params.iter()
                .filter(|p| p.mutable)
                .map(|p| p.name.clone())
                .collect();
            let prev_in_struct_method = self.in_struct_method;
            self.in_struct_method = self_ty.is_some();
            self.infer_qualifiers(&f.body);
            self.in_struct_method = prev_in_struct_method;
            // Snapshot only the qualifiers for THIS function's params.
            pre_inferred_param_quals = f.params.iter()
                .filter_map(|p| {
                    self.inferred_qualifiers.get(&p.name)
                        .map(|q| (p.name.clone(), q.clone()))
                })
                .collect();
            self.fn_current_params = prev;
            self.fn_current_param_lines = prev_param_lines;
            self.fn_current_param_cols = prev_param_cols;
            self.fn_current_params_mut = prev_mut;
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
                        self.push_error(f.line, f.col, format!(
                            "`task fn` method '{}::{}' requires '{}' to be used with a \
                             'task, 'actor, or 'guard qualifier at least once in the program \
                             (no arc-qualified binding found)",
                            sname, f.name, sname
                        ));
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

        // ── Stream functions: iterator (sequential) or async_stream (async) ──
        if f.stream {
            if self.stream_iter_fns.contains(&f.name) {
                return self.emit_iter_stream_fn(f, &type_params, &all_params);
            }
            return self.emit_stream_fn(f, &type_params, &all_params);
        }

        // Return type
        let base_ret = f.return_ty.as_ref()
            .map(|t| {
                // If the return type is a known trait name, use `impl TraitName` (static dispatch).
                // Dynamic dispatch is expressed explicitly with `Type::Dyn` → Box<dyn Trait>.
                if let Type::Named(n) = t {
                    if self.trait_method_names.contains_key(n.as_str()) {
                        return format!("impl {}", normalize_type_name(n, self.use_rc_str()));
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
                        Type::Int   => Some("isize".to_string()),
                        Type::Uint  => Some("usize".to_string()),
                        Type::Uint8 => Some("u8".to_string()),
                        Type::Int8   => Some("i8".to_string()),
                        Type::Int16  => Some("i16".to_string()),
                        Type::Int32  => Some("i32".to_string()),
                        Type::Int64  => Some("i64".to_string()),
                        Type::Int128 => Some("i128".to_string()),
                        Type::Uint16 => Some("u16".to_string()),
                        Type::Uint32 => Some("u32".to_string()),
                        Type::Uint64 => Some("u64".to_string()),
                        Type::Uint128 => Some("u128".to_string()),
                        Type::Float => Some("f64".to_string()),
                        Type::Bool  => Some("bool".to_string()),
                        Type::Str   => Some(if self.use_rc_str() { "Rc<str>" } else { "Arc<str>" }.to_string()),
                        Type::Named(n) => match n.as_str() {
                            "int" | "isize" => Some("isize".to_string()),
                            "uint" | "usize" => Some("usize".to_string()),
                            "uint8" | "u8" => Some("u8".to_string()),
                            "int8" | "i8" => Some("i8".to_string()),
                            "int16" | "i16" => Some("i16".to_string()),
                            "int32" | "i32" => Some("i32".to_string()),
                            "int64" | "i64" => Some("i64".to_string()),
                            "int128" | "i128" => Some("i128".to_string()),
                            "uint16" | "u16" => Some("u16".to_string()),
                            "uint32" | "u32" => Some("u32".to_string()),
                            "uint64" | "u64" => Some("u64".to_string()),
                            "uint128" | "u128" => Some("u128".to_string()),
                            "float" | "f64" | "f32" => Some("f64".to_string()),
                            "bool"   => Some("bool".to_string()),
                            "string" | "str" => Some(if self.use_rc_str() { "Rc<str>" } else { "Arc<str>" }.to_string()),
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
        let rust_fn_name = if f.name.is_empty() {
            // Anonymous call operator `def ()` / `req ()` → emit as `__call__`
            "__call__".to_string()
        } else if is_free_overload || is_method_overload {
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
        let prev_req_fn   = self.in_req_fn;
        let prev_async         = self.in_async;
        let prev_fn_returns_void = self.fn_returns_void;
        let prev_fn_declared_void = self.fn_declared_void;
        // Track the current function's type parameters for generic match detection.
        let prev_fn_type_params = std::mem::replace(
            &mut self.current_fn_type_params,
            f.type_params.iter().cloned().collect(),
        );
        // A function is "void" if it has no declared return type, or returns () / Nil / Void,
        // and is not a throws function (which wraps in Result<(),...>).
        let declared_void = match &f.return_ty {
            None => true,
            Some(Type::Void) | Some(Type::Nil) => true,
            Some(Type::Named(n)) if n == "void" || n == "nil" => true,
            _ => false,
        };
        let is_void_ret = !f.throws && declared_void;
        self.fn_returns_void = is_void_ret;
        self.fn_declared_void = declared_void;
        let prev_task_vars     = std::mem::take(&mut self.task_vars);
        let prev_arc_vars      = std::mem::take(&mut self.arc_vars);
        let prev_rc_vars       = std::mem::take(&mut self.rc_vars);
        let prev_shared_ref_params = std::mem::take(&mut self.shared_ref_params);
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
        let prev_string_arc_vars  = std::mem::take(&mut self.string_arc_vars);
        let prev_vec_vars         = std::mem::take(&mut self.vec_vars);
        let prev_collection_vars  = std::mem::take(&mut self.collection_vars);
        let prev_dict_vars        = std::mem::take(&mut self.dict_vars);
        let prev_managed_refcell_vars = std::mem::take(&mut self.managed_refcell_vars);
        let prev_managed_mutex_vars   = std::mem::take(&mut self.managed_mutex_vars);
        // Pre-seed known_local_vars, var_struct_types, var_types, and var_mutex_types from params.
        for p in &f.params {
            self.known_local_vars.insert(p.name.clone());
            if let Some(ty) = &p.ty {
                self.var_types.insert(p.name.clone(), ty.clone());
                // Track string params for string concatenation detection.
                if Self::is_string_type(ty) {
                    self.string_vars.insert(p.name.clone());
                }
                // T'actor / T'actor'task params → mutex tracking.
                if Self::is_mutex_binding(p.mutable, ty) {
                    if Self::is_mutex_task_binding(p.mutable, ty) {
                        self.var_mutex_task_types.insert(p.name.clone());
                    } else {
                        self.var_mutex_types.insert(p.name.clone());
                    }
                    self.arc_vars.insert(p.name.clone());
                    if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        self.rc_vars.insert(p.name.clone());
                    }
                }
                // T'guard / T'guard'task params → rwlock tracking.
                if Self::is_rwlock_binding(p.mutable, ty) {
                    if Self::is_rwlock_task_binding(p.mutable, ty) {
                        self.var_rwlock_task_types.insert(p.name.clone());
                    } else {
                        self.var_rwlock_types.insert(p.name.clone());
                    }
                    self.arc_vars.insert(p.name.clone());
                }
                // Arc/Rc-qualified and string params must be cloned before capture in `async move {}` blocks.
                if Self::is_arc_qualified(ty) || Self::is_rc_qualified(ty) || Self::is_string_type(ty) {
                    self.arc_vars.insert(p.name.clone());
                    // In single-thread mode, T'shared → Rc<T>; mark for Rc::clone.
                    if Self::is_rc_qualified(ty) && matches!(self.config.threading, crate::transpiler::ThreadingMode::Single) {
                        self.rc_vars.insert(p.name.clone());
                    }
                    // Params are now by-value (owned clone), so single deref (*var) suffices for match.
                }
                if matches!(ty, Type::Optional(_)) {
                    self.optional_vars.insert(p.name.clone());
                }
                // Managed mode T' params (OwnerQual::Owned over user type) → managed tracking.
                // Also resolve non-function type aliases (e.g. `use Pt as LPoint'` → LPoint').
                let resolved_ty_for_managed = if let Type::Named(n) = ty {
                    self.non_fn_type_aliases.get(n.as_str()).unwrap_or(ty)
                } else { ty };
                if crate::transpiler::Transpiler::is_managed_user_owned(
                    &self.config, &self.user_types, &self.unit_enums, resolved_ty_for_managed)
                {
                    match self.config.threading {
                        crate::transpiler::ThreadingMode::Multi => {
                            self.managed_mutex_vars.insert(p.name.clone());
                            self.arc_vars.insert(p.name.clone());
                        }
                        crate::transpiler::ThreadingMode::Single => {
                            self.managed_refcell_vars.insert(p.name.clone());
                        }
                    }
                }
                // Unqualified params whose type is a known actor-source type are inferred as 'actor.
                // Register them in var_mutex_types so field access and method calls emit .lock().
                if let Type::Named(n) = ty {
                    if self.actor_source_types.contains(n.as_str()) {
                        self.var_mutex_types.insert(p.name.clone());
                        self.arc_vars.insert(p.name.clone());
                    }
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
        self.in_req_fn = !f.mutating;
        self.in_struct_method = self_ty.is_some();
        self.in_async  = is_async;
        let prev_fn_return_ty = self.fn_return_ty.clone();
        self.fn_return_ty = f.return_ty.clone();
        let prev_fn_current_params = std::mem::take(&mut self.fn_current_params);
        self.fn_current_params = f.params.iter()
            .filter_map(|p| p.ty.as_ref().map(|ty| (p.name.clone(), ty.clone())))
            .collect();
        let prev_fn_current_param_lines = std::mem::take(&mut self.fn_current_param_lines);
        self.fn_current_param_lines = f.params.iter()
            .map(|p| (p.name.clone(), p.line))
            .collect();
        let prev_fn_current_param_cols = std::mem::take(&mut self.fn_current_param_cols);
        self.fn_current_param_cols = f.params.iter()
            .map(|p| (p.name.clone(), p.col))
            .collect();
        let prev_fn_current_params_mut = std::mem::take(&mut self.fn_current_params_mut);
        self.fn_current_params_mut = f.params.iter()
            .filter(|p| p.mutable)
            .map(|p| p.name.clone())
            .collect();
        let prev_immutable_local_vars = std::mem::take(&mut self.immutable_local_vars);
        // Plain params (neither `mut` nor `var`) are immutable.
        self.immutable_local_vars = f.params.iter()
            .filter(|p| !p.mutable && !p.rebindable)
            .map(|p| p.name.clone())
            .collect();
        let prev_mut_local_vars = std::mem::take(&mut self.mut_local_vars);
        // `mut` params are mutable but non-rebindable.
        self.mut_local_vars = f.params.iter()
            .filter(|p| p.mutable && !p.rebindable)
            .map(|p| p.name.clone())
            .collect();
        let prev_auto_ref_params = std::mem::take(&mut self.auto_ref_params);
        self.auto_ref_params = f.params.iter()
            .filter(|p| matches!(&p.ty,
                Some(Type::Qualified(inner, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Weak))
                if !matches!(inner.as_ref(), Type::Named(n) if self.trait_method_names.contains_key(n.as_str()))
            ))
            .map(|p| p.name.clone())
            .collect();
        let prev_var_primitive_params = std::mem::take(&mut self.var_primitive_params);
        self.var_primitive_params = f.params.iter()
            .filter(|p| p.rebindable && !matches!(&p.ty,
                Some(Type::Qualified(_, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Weak))
            ))
            .map(|p| p.name.clone())
            .collect();
        let prev_in_cancellable_fn = self.in_cancellable_fn;
        if is_cancellable {
            self.in_cancellable_fn = true;
            // Register __task_cancel as a known local so emit_expr can resolve it.
            self.known_local_vars.insert("__task_cancel".to_string());
        }
        // Managed multi-thread: emit a lock guard let-binding for each managed param
        // to avoid double-lock deadlock when the same param's fields are accessed
        // more than once in a single expression (std::sync::Mutex is not reentrant).
        let prev_managed_param_shadows = std::mem::take(&mut self.managed_param_shadows);
        if matches!(self.config.threading, crate::transpiler::ThreadingMode::Multi) {
            for p in &f.params {
                if self.managed_mutex_vars.contains(&p.name) {
                    let shadow = format!("__{}_mg", p.name);
                    self.line(&format!("let mut {shadow} = {name}.lock().unwrap();",
                        shadow = shadow, name = p.name));
                    self.managed_mutex_vars.remove(&p.name);
                    self.managed_param_shadows.insert(p.name.clone(), shadow);
                }
            }
        }
        if self.config.instrument {
            let span_label = if let Some(ty) = self_ty {
                format!("{}::{}", ty, f.name)
            } else {
                f.name.clone()
            };
            if f.name == "main" && self_ty.is_none() {
                self.line("let _boring_dump = __boring_instrument::DumpGuard;");
            }
            self.line(&format!("let _boring_span = __boring_instrument::Span::enter(\"{}\");", span_label));
        }
        self.emit_body(&f.body);
        // Cross-function propagation: update fn_sigs with inferred param qualifiers so that
        // callers defined after this function see the qualified signature and can propagate
        // the constraint to their own anonymous variables.
        // Use pre_inferred_param_quals (snapshotted before emit_body) rather than
        // self.inferred_qualifiers, which may have been overwritten by nested infer_qualifiers
        // calls made during arm/block body emission.
        if self_ty.is_none() {
            if let Some(sig) = self.fn_sigs.get_mut(&f.name) {
                for (i, param) in f.params.iter().enumerate() {
                    if let Some(inferred_qual) = pre_inferred_param_quals.get(&param.name).cloned() {
                        // Do not propagate 'stack back into fn_sigs: it is the default fallback and
                        // would prevent actor_source_types inference from propagating downstream.
                        // Only non-default qualifiers (actor/guard/shared/heap) are meaningful signals.
                        if matches!(inferred_qual, crate::ast::OwnerQual::Stack) {
                            continue;
                        }
                        if let Some(param_ty) = sig.get_mut(i) {
                            let already_qualified = matches!(param_ty, crate::ast::Type::Qualified(..))
                                && !matches!(param_ty, crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Owned))
                                && !matches!(param_ty, crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Union(_)));
                            if !already_qualified {
                                *param_ty = crate::transpiler::infer_qualifiers::apply_inferred_qual(
                                    param_ty, inferred_qual,
                                );
                            }
                        }
                    }
                }
            }
        }
        self.managed_param_shadows = prev_managed_param_shadows;
        self.in_cancellable_fn = prev_in_cancellable_fn;
        self.in_throws         = prev_throws;
        self.in_req_fn         = prev_req_fn;
        self.fn_returns_void   = prev_fn_returns_void;
        self.fn_declared_void  = prev_fn_declared_void;
        self.fn_return_ty      = prev_fn_return_ty;
        self.fn_current_params      = prev_fn_current_params;
        self.fn_current_param_lines = prev_fn_current_param_lines;
        self.fn_current_param_cols  = prev_fn_current_param_cols;
        self.fn_current_params_mut  = prev_fn_current_params_mut;
        self.immutable_local_vars   = prev_immutable_local_vars;
        self.mut_local_vars         = prev_mut_local_vars;
        self.auto_ref_params       = prev_auto_ref_params;
        self.var_primitive_params  = prev_var_primitive_params;
        self.task_vars             = prev_task_vars;
        self.arc_vars          = prev_arc_vars;
        self.rc_vars           = prev_rc_vars;
        self.shared_ref_params = prev_shared_ref_params;
        self.weak_vars         = prev_weak_vars;
        self.var_struct_types  = prev_var_struct_types;
        self.var_mutex_types   = prev_var_mutex_types;
        self.var_rwlock_types  = prev_var_rwlock_types;
        self.known_local_vars  = prev_known_local_vars;
        self.optional_vars     = prev_optional_vars;
        self.var_types         = prev_var_types;
        self.string_vars       = prev_string_vars;
        self.string_arc_vars   = prev_string_arc_vars;
        self.vec_vars          = prev_vec_vars;
        self.collection_vars   = prev_collection_vars;
        self.dict_vars         = prev_dict_vars;
        self.managed_refcell_vars = prev_managed_refcell_vars;
        self.managed_mutex_vars   = prev_managed_mutex_vars;
        self.in_async          = prev_async;
        self.current_fn_type_params = prev_fn_type_params;
        self.indent -= 1;
        self.line("}");
    }

    /// Emit a purely-sequential `stream` function as `impl Iterator<Item = T>`.
    /// `yield expr` → `__items.push(expr)`, body runs eagerly, returns `vec.into_iter()`.
    pub(crate) fn emit_iter_stream_fn(&mut self, f: &FnDecl, type_params: &str, all_params: &str) {
        let vis = if f.is_pub { "pub " } else { "" };
        let item_ty = f.return_ty.as_ref()
            .map(|t| self.emit_stream_item_type(t))
            .unwrap_or_else(|| "()".to_string());

        let sig = format!(
            "{}fn {}{}({}) -> impl Iterator<Item = {}>",
            vis, f.name, type_params, all_params, item_ty
        );
        self.line(&format!("{} {{", sig));
        self.indent += 1;
        self.line(&format!("let mut __items: Vec<{}> = Vec::new();", item_ty));

        let prev_fn_returns_void = self.fn_returns_void;
        let prev_fn_return_ty    = self.fn_return_ty.clone();
        let prev_known           = std::mem::take(&mut self.known_local_vars);
        let prev_iter_stream     = self.in_iter_stream;
        self.fn_returns_void = true;
        self.fn_return_ty    = None;
        self.in_iter_stream  = true;
        for p in &f.params { self.known_local_vars.insert(p.name.clone()); }

        self.emit_body(&f.body);

        self.fn_returns_void  = prev_fn_returns_void;
        self.fn_return_ty     = prev_fn_return_ty;
        self.known_local_vars = prev_known;
        self.in_iter_stream   = prev_iter_stream;

        self.line("__items.into_iter()");
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
        // !Send warning: stream item type is Rc/RefCell in single-thread mode.
        if matches!(self.config.threading, crate::transpiler::ThreadingMode::Single)
            && (base_item_ty.contains("Rc<") || base_item_ty.contains("RefCell<"))
        {
            self.push_warning(f.line, f.col, format!("stream `{}` item type `{}` is !Send in single-thread mode; stream<N> requires Send on the item type", f.name, base_item_ty));
        }

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
            // Known-primitive keys (e.g. a `{int=string}` dict keyed by a loop var of type
            // int/uint/float/bool) are Copy types, not Arc<str> — a plain reference, no Deref.
            // Lowercase source syntax (`int`, `uint`, ...) parses as `Type::Named`, not the
            // bare builtin variants — match both forms.
            ExprKind::Var(v) if matches!(
                self.var_types.get(v.as_str()),
                Some(Type::Int | Type::Uint | Type::Float | Type::Bool)
            ) || matches!(
                self.var_types.get(v.as_str()),
                Some(Type::Named(n)) if matches!(n.as_str(), "int" | "uint" | "float" | "bool")
            ) => format!("&{}", v),
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
            ExprKind::Str(s) => self.str_from(&escape_str(s)),
            ExprKind::Var(v) if self.chars_vars.contains(v.as_str()) =>
                self.str_from_expr(&format!("{}.to_string()", v)),
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
        // `mut` is added only when explicitly declared in Boring (`mut T param` or `var T param`).
        // FnMut closure params also need `mut` so the closure can be called.
        // Struct params do NOT get `mut` automatically — the developer must declare `mut`.
        let name = if p.mutable || is_fnmut { format!("mut {}", p.name) } else { p.name.clone() };
        match &p.ty {
            Some(ty) if p.variadic => format!("{}: Vec<{}>", name, self.emit_type(ty)),
            Some(ty) => {
                // If the param has no explicit qualifier but inference resolved one, apply it.
                let inferred_ty;
                let needs_inference = !matches!(ty, crate::ast::Type::Qualified(..))
                    || matches!(ty, crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Owned))
                    || matches!(ty, crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Union(_)))
                    || matches!(ty, crate::ast::Type::Optional(inner)
                        if matches!(inner.as_ref(), crate::ast::Type::Named(_)
                            | crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Owned)
                            | crate::ast::Type::Qualified(_, crate::ast::OwnerQual::Union(_))));
                let effective_ty = if needs_inference {
                    if let Some(qual) = self.inferred_qualifiers.get(&p.name) {
                        inferred_ty = crate::transpiler::infer_qualifiers::apply_inferred_qual(ty, qual.clone());
                        &inferred_ty
                    } else {
                        ty
                    }
                } else {
                    ty
                };
                // If the param type is a known trait name, emit `impl TraitName`.
                let ty_s = if let crate::ast::Type::Named(n) = effective_ty {
                    if self.trait_method_names.contains_key(n.as_str()) {
                        format!("impl {}", normalize_type_name(n, self.use_rc_str()))
                    } else {
                        self.emit_type(effective_ty)
                    }
                } else {
                    self.emit_type(effective_ty)
                };
                // `var` on any stack/primitive/heap param → out-param `&mut T`.
                // 'actor/'guard params (and task variants) are passed by reference to avoid atomic
                // refcount increments. Callers pass `&val`; the callee can still clone if needed.
                // 'shared and 'weak are passed by owned value (BorrowShared handles &Rc/&Arc separately).
                let ty_s = match effective_ty {
                    Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask) => format!("&{}", ty_s),
                    Type::Qualified(_, OwnerQual::Guard | OwnerQual::GuardTask) => format!("&{}", ty_s),
                    Type::Qualified(_, OwnerQual::Shared | OwnerQual::Weak) => ty_s,
                    // Borrow/BorrowMut already carry their own &/&mut — don't double-wrap.
                    Type::Qualified(_, OwnerQual::Borrow | OwnerQual::BorrowMut) => ty_s,
                    _ if p.rebindable => format!("&mut {}", ty_s),
                    _ => ty_s,
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
            ExprKind::Str(s) => self.str_from(&escape_str(s)),
            ExprKind::StringInterp(_) => {
                // format!(...) gives String; wrap in Rc/Arc
                self.str_from_expr(&self.emit_expr(expr))
            }
            // self.field in owned context → .clone() so Arc<T> fields don't move out of &self
            ExprKind::Field(obj, field) => {
                // GPU kernel field read (e.g. a kernel wrapper's `.unified`/`.global` array
                // field used as a throws function's tail expression, wrapped in `Ok(...)`
                // via emit_expr_owned rather than emit_expr) — same conversion as emit_expr's
                // Field arm, so the f32 GPU buffer element type doesn't leak into host code.
                if let Some(code) = self.try_emit_kernel_field_read(obj, field) {
                    return code;
                }
                // Type-level access: delegate to emit_expr which handles Counter::X
                if let ExprKind::Var(type_name) = &obj.kind {
                    if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                        let key = format!("{}::{}", type_name, field);
                        if self.struct_type_var_names.contains(&key)
                            || self.struct_type_mut_var_names.contains(&key) {
                            return self.emit_expr(expr);
                        }
                        // Enum variant or external PascalCase type: TokenKind.Let → TokenKind::Let
                        if self.enum_variant_fields.contains_key(&key)
                            || !self.known_local_vars.contains(type_name.as_str())
                        {
                            return format!("{}::{}", type_name, field);
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
                // Mutex var access (owned context): w.field → w.lock().await.field (multi) or w.borrow().field (single)
                if let ExprKind::Var(v) = &obj.kind {
                    if self.var_mutex_types.contains(v.as_str()) || self.var_mutex_task_types.contains(v.as_str()) {
                        let access = self.mutex_var_read(v, v);
                        // If the field itself is an actor (Rc/Arc<RefCell/Mutex<T>>), clone it
                        // to avoid moving out of a temporary borrow/lock guard.
                        let field_ty_opt = self.struct_fields.get(
                                self.var_types.get(v.as_str())
                                    .and_then(|t| match t { Type::Named(n) => Some(n.as_str()), Type::Qualified(inner, _) => if let Type::Named(n) = inner.as_ref() { Some(n.as_str()) } else { None }, _ => None })
                                    .or_else(|| self.var_struct_types.get(v.as_str()).map(|s| s.as_str()))
                                    .unwrap_or("")
                            ).and_then(|fields| fields.iter().find(|(fname,_)| fname == field))
                            .map(|(_, fty)| fty.clone());
                        let field_is_shared = field_ty_opt.as_ref()
                            .map(|fty| Self::is_arc_qualified(fty) || Self::is_rc_qualified(fty))
                            .unwrap_or(false);
                        let field_is_string = field_ty_opt.as_ref()
                            .map(Self::is_string_type)
                            .unwrap_or(false);
                        return if field_is_shared {
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Single =>
                                    format!("Rc::clone(&{}.{})", access, field),
                                crate::transpiler::ThreadingMode::Multi =>
                                    format!("Arc::clone(&{}.{})", access, field),
                            }
                        } else if field_is_string {
                            // Arc<str> fields can't be moved out of a MutexGuard — clone them.
                            format!("{}.{}.clone()", access, field)
                        } else {
                            format!("{}.{}", access, field)
                        };
                    }
                    // Managed param shadow: use pre-locked guard variable.
                    if let Some(shadow) = self.managed_param_shadows.get(v.as_str()) {
                        return format!("{}.{}", shadow, field);
                    }
                    // Managed-mode mutex var (std::sync::Mutex, synchronous):
                    if self.managed_mutex_vars.contains(v.as_str()) {
                        return format!("{}.lock().unwrap().{}", v, field);
                    }
                    // Managed-mode RefCell var (single-thread):
                    if self.managed_refcell_vars.contains(v.as_str()) {
                        return format!("{}.borrow().{}", v, field);
                    }
                }
                // RwLock var access (owned context): c.field → c.read().await.field (async) or c.read().unwrap().field (sync)
                if let ExprKind::Var(v) = &obj.kind {
                    if self.var_rwlock_types.contains(v.as_str()) || self.var_rwlock_task_types.contains(v.as_str()) {
                        let access = if self.var_rwlock_task_types.contains(v.as_str()) {
                            self.guard_task_read_access(v)
                        } else {
                            self.guard_read_access(v)
                        };
                        return format!("{}.{}", access, field);
                    }
                }
                // Mutex struct field (owned context): self.worker.field → self.worker.lock().[await|unwrap()].field
                if let ExprKind::Field(inner_obj, mutex_field) = &obj.kind {
                    if let ExprKind::Var(v) = &inner_obj.kind {
                        if v == "self" {
                            let key = self.self_type.as_deref()
                                .map(|t| format!("{}::{}", t, mutex_field));
                            if let Some(k) = key {
                                if self.struct_mutex_fields.contains(&k) || self.struct_mutex_task_fields.contains(&k) {
                                    let guard = self.mutex_field_write(&k, &format!("self.{}", mutex_field));
                                    return format!("{}.{}", guard, field);
                                }
                            }
                        }
                    }
                }
                // RwLock struct field (owned context): self.data.field → self.data.read().[await|unwrap()].field
                if let ExprKind::Field(inner_obj, rwlock_field) = &obj.kind {
                    if let ExprKind::Var(v) = &inner_obj.kind {
                        if v == "self" {
                            let key = self.self_type.as_deref()
                                .map(|t| format!("{}::{}", t, rwlock_field));
                            if let Some(k) = key {
                                if self.struct_rwlock_fields.contains(&k) || self.struct_rwlock_task_fields.contains(&k) {
                                    let guard = self.rwlock_field_write(&k, &format!("self.{}", rwlock_field));
                                    return format!("{}.{}", guard, field);
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
                        .then_some(self.self_type.as_deref())
                        .flatten()
                        .and_then(|t| self.struct_fields.get(t))
                        .map(|fields| fields.iter().any(|(fname, _)| fname == field))
                        .unwrap_or(false);
                    if is_user_field { field.as_str() } else { map_field(field) }
                } else {
                    map_field(field)
                };
                let result = if (obj_s == "self" || field.chars().all(|c| c.is_ascii_digit())) && !field_s.ends_with(')') {
                    // self.field or tuple index .0/.1/... — always clone in owned context (value semantics)
                    format!("{}.{}.clone()", obj_s, field_s)
                } else {
                    format!("{}.{}", obj_s, field_s)
                };
                if field_s.contains(" as ") { format!("({})", result) } else { result }
            }
            // Method calls that produce &str (e.g. `.trim()`) need Arc::<str>::from(x.to_string())
            // when used in an owned string (Arc<str>) context.
            ExprKind::MethodCall(_, method, _)
                if matches!(method.as_str(), "trim") =>
            {
                let s = self.emit_expr(expr);
                self.str_from_expr(&format!("{}.to_string()", s))
            }
            // Struct variable in owned position: clone to avoid moving out.
            // In Boring, structs are copy-by-value semantics; Rust requires explicit .clone().
            // Exception: `var` params are `&mut T` — emit as-is (the &mut is already the reference).
            ExprKind::Var(v) if self.var_struct_types.contains_key(v.as_str())
                && !self.arc_vars.contains(v.as_str())
                && !self.var_mutex_types.contains(v.as_str())
                && !self.var_primitive_params.contains(v.as_str()) =>
            {
                format!("{}.clone()", self.emit_expr(expr))
            }
            // String variable in owned position: clone to avoid moving Rc<str>.
            ExprKind::Var(v) if self.string_vars.contains(v.as_str())
                || self.string_arc_vars.contains(v.as_str()) =>
            {
                format!("{}.clone()", v)
            }
            // Arc/Rc var in owned position (actor/shared params passed by ref): clone the pointer.
            ExprKind::Var(v) if self.arc_vars.contains(v.as_str())
                || self.var_mutex_types.contains(v.as_str())
                || self.managed_mutex_vars.contains(v.as_str())
                || self.managed_refcell_vars.contains(v.as_str()) =>
            {
                format!("{}.clone()", v)
            }
            // Non-Copy named types (user enums/structs tracked in var_types): clone to preserve value semantics.
            ExprKind::Var(v) if matches!(
                self.var_types.get(v.as_str()),
                Some(crate::ast::Type::Named(_) | crate::ast::Type::Array(_) | crate::ast::Type::Dict(..) | crate::ast::Type::Set(_))
            ) => {
                format!("{}.clone()", v)
            }
            _ => self.emit_expr(expr),
        }
    }

    /// Emit a function body, wrapping the last expression in Ok() for throws functions.
    /// Handles `defer` statements: collects them and emits in LIFO order before the
    /// final return value.
    /// Like `emit_body` but wraps the last expression in `Some(...)` when it is not already
    /// nil/None/optional. Used for if-expression branches in an `Option<T>` context.
    pub(crate) fn emit_body_optional_last(&mut self, stmts: &[Stmt]) {
        if stmts.is_empty() {
            self.line("None");
            return;
        }
        // Temporarily set fn_return_ty to trigger Some() wrapping in emit_stmt for the last expr.
        let saved_return_ty = self.fn_return_ty.take();
        self.fn_return_ty = Some(crate::ast::Type::Optional(Box::new(crate::ast::Type::Named("__opt__".to_string()))));
        // Emit all but last normally, then last with Optional context.
        // We must run qualifier inference on the full slice first.
        self.infer_qualifiers(stmts);
        self.validate_union_constraints(stmts);
        self.suggest_param_annotations();
        let non_defers: Vec<&Stmt> = stmts.iter().filter(|s| !matches!(s, Stmt::Defer(_))).collect();
        let last_idx = non_defers.len().saturating_sub(1);
        for (i, stmt) in non_defers.iter().enumerate() {
            if i < last_idx {
                // Non-last stmts: emit normally without Optional context.
                let saved = self.fn_return_ty.take();
                self.emit_stmt(stmt, false);
                self.fn_return_ty = saved;
            } else {
                // Last stmt: emit with Optional context so nil→None, T→Some(T).
                match stmt {
                    Stmt::Expr(e) if matches!(&e.kind, ExprKind::Nil) => {
                        self.fn_return_ty = saved_return_ty.clone();
                        self.line("None");
                        self.fn_return_ty = Some(crate::ast::Type::Optional(Box::new(crate::ast::Type::Named("__opt__".to_string()))));
                    }
                    _ => {
                        self.emit_stmt(stmt, true);
                    }
                }
            }
        }
        self.fn_return_ty = saved_return_ty;
    }

    pub(crate) fn emit_body(&mut self, stmts: &[Stmt]) {
        if stmts.is_empty() { return; }
        // Priority 5: use-site qualifier inference — runs before size-based decisions, all modes.
        self.infer_qualifiers(stmts);
        // After inference: validate union constraints (both caller-side and body-side),
        // then emit hints for unqualified parameters that could benefit from annotation.
        self.validate_union_constraints(stmts);
        self.suggest_param_annotations();
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
            // Skip when suppress_ok_wrap is set — we're emitting a branch of an if/match expression,
            // not a function body, so the last value is already correct without a sentinel.
            if self.in_throws && !self.suppress_ok_wrap {
                let last_stmt = non_defers.last();
                let needs_ok = match last_stmt {
                    Some(Stmt::Expr(_)) | Some(Stmt::Return(_)) | Some(Stmt::Throw(_)) => false,
                    // A complete if/else or match covers all paths — no unreachable!() needed.
                    Some(Stmt::If(s)) if s.else_body.is_some() => false,
                    Some(Stmt::Match(_)) => false,
                    _ => true,
                };
                if needs_ok {
                    if self.fn_declared_void {
                        self.line("Ok(())");
                    } else {
                        self.line("unreachable!()");
                    }
                }
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
    /// Extract the top-level `OwnerQual` from a `Type::Qualified`, if present.
    /// Returns a placeholder variant that matches nothing if the type is not qualified.
    pub(crate) fn unwrap_qual(ty: &Type) -> &OwnerQual {
        if let Type::Qualified(_, q) = ty { q } else { &OwnerQual::Stack }
    }

    /// Only applies to user-defined struct types (PascalCase names), not primitives or `string`.
    /// The `mutable` parameter is ignored — `T'actor` is always mutex regardless of let/var.
    pub(crate) fn is_mutex_binding(_mutable: bool, ty: &Type) -> bool {
        matches!(ty, Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask))
    }

    pub(crate) fn is_mutex_task_binding(_mutable: bool, ty: &Type) -> bool {
        matches!(ty, Type::Qualified(_, OwnerQual::ActorTask))
    }

    /// Extract the inner `T` from a `T'actor` or `T'actor'task` mutex type.
    pub(crate) fn mutex_inner(ty: &Type) -> Option<&Type> {
        if let Type::Qualified(inner, OwnerQual::Actor | OwnerQual::ActorTask) = ty {
            Some(inner)
        } else {
            None
        }
    }

    /// Returns true if the program uses async actors (tokio::sync::Mutex/RwLock).
    /// When no task or stream functions exist, all actor access is synchronous and
    /// std::sync::Mutex is used instead, avoiding the need for async fn and .await.
    pub(crate) fn use_async_actors(&self) -> bool {
        matches!(self.config.threading, crate::transpiler::ThreadingMode::Multi)
            && (!self.task_fns.is_empty() || !self.stream_fns.is_empty())
    }

    /// Wrap `inner_expr` in the appropriate actor constructor for the current threading mode.
    /// multi + async fns → `Arc::new(tokio::sync::Mutex::new(v))`, multi + no async → `Arc::new(Mutex::new(v))`,
    /// single → `Rc::new(RefCell::new(v))`.
    // ── 'actor (std::sync::Mutex) ─────────────────────────────────────────────
    pub(crate) fn emit_actor_new(&self, inner_expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("Arc::new(std::sync::Mutex::new({}))", inner_expr),
            crate::transpiler::ThreadingMode::Single => format!("Rc::new(RefCell::new({}))", inner_expr),
        }
    }

    pub(crate) fn emit_actor_type(&self, inner: &Type) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("Arc<std::sync::Mutex<{}>>", self.emit_type(inner)),
            crate::transpiler::ThreadingMode::Single => format!("Rc<RefCell<{}>>", self.emit_type(inner)),
        }
    }

    pub(crate) fn emit_mutex_type(&self, inner: &Type) -> String { self.emit_actor_type(inner) }

    /// actor read: `.lock().unwrap()` (multi), `.borrow()` (single).
    pub(crate) fn actor_read_access(&self, expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("{}.lock().unwrap()", expr),
            crate::transpiler::ThreadingMode::Single => format!("{}.borrow()", expr),
        }
    }

    /// actor write: `.lock().unwrap()` (multi), `.borrow_mut()` (single).
    pub(crate) fn actor_write_guard(&self, expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("{}.lock().unwrap()", expr),
            crate::transpiler::ThreadingMode::Single => format!("{}.borrow_mut()", expr),
        }
    }

    // ── 'actor'task / 'task (tokio::sync::Mutex) ─────────────────────────────

    pub(crate) fn emit_actor_task_new(&self, inner_expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("Arc::new(tokio::sync::Mutex::new({}))", inner_expr),
            crate::transpiler::ThreadingMode::Single => format!("Rc::new(RefCell::new({}))", inner_expr),
        }
    }

    pub(crate) fn emit_actor_task_type(&self, inner: &Type) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("Arc<tokio::sync::Mutex<{}>>", self.emit_type(inner)),
            crate::transpiler::ThreadingMode::Single => format!("Rc<RefCell<{}>>", self.emit_type(inner)),
        }
    }

    /// actor'task read: `.lock().await` (multi+async), `.lock().unwrap()` (multi+sync), `.borrow()` (single).
    pub(crate) fn actor_task_read_access(&self, expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi if self.in_async => format!("{}.lock().await", expr),
            crate::transpiler::ThreadingMode::Multi                  => format!("{}.lock().unwrap()", expr),
            crate::transpiler::ThreadingMode::Single                 => format!("{}.borrow()", expr),
        }
    }

    /// actor'task write: `.lock().await` (multi+async), `.lock().unwrap()` (multi+sync), `.borrow_mut()` (single).
    pub(crate) fn actor_task_write_guard(&self, expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi if self.in_async => format!("{}.lock().await", expr),
            crate::transpiler::ThreadingMode::Multi                  => format!("{}.lock().unwrap()", expr),
            crate::transpiler::ThreadingMode::Single                 => format!("{}.borrow_mut()", expr),
        }
    }

    // ── 'guard (std::sync::RwLock) ────────────────────────────────────────────

    pub(crate) fn emit_guard_new(&self, inner_expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("Arc::new(std::sync::RwLock::new({}))", inner_expr),
            crate::transpiler::ThreadingMode::Single => format!("Rc::new(RefCell::new({}))", inner_expr),
        }
    }

    pub(crate) fn emit_guard_type(&self, inner: &Type) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("Arc<std::sync::RwLock<{}>>", self.emit_type(inner)),
            crate::transpiler::ThreadingMode::Single => format!("Rc<RefCell<{}>>", self.emit_type(inner)),
        }
    }

    /// guard write: `.write().unwrap()` (multi), `.borrow_mut()` (single).
    pub(crate) fn guard_write_guard(&self, expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("{}.write().unwrap()", expr),
            crate::transpiler::ThreadingMode::Single => format!("{}.borrow_mut()", expr),
        }
    }

    /// guard read: `.read().unwrap()` (multi), `.borrow()` (single).
    pub(crate) fn guard_read_access(&self, expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("{}.read().unwrap()", expr),
            crate::transpiler::ThreadingMode::Single => format!("{}.borrow()", expr),
        }
    }

    // ── 'guard'task (tokio::sync::RwLock) ────────────────────────────────────

    pub(crate) fn emit_guard_task_new(&self, inner_expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("Arc::new(tokio::sync::RwLock::new({}))", inner_expr),
            crate::transpiler::ThreadingMode::Single => format!("Rc::new(RefCell::new({}))", inner_expr),
        }
    }

    pub(crate) fn emit_guard_task_type(&self, inner: &Type) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("Arc<tokio::sync::RwLock<{}>>", self.emit_type(inner)),
            crate::transpiler::ThreadingMode::Single => format!("Rc<RefCell<{}>>", self.emit_type(inner)),
        }
    }

    /// guard'task write: `.write().await` (multi+async), `.write().unwrap()` (multi+sync), `.borrow_mut()` (single).
    pub(crate) fn guard_task_write_guard(&self, expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi if self.in_async => format!("{}.write().await", expr),
            crate::transpiler::ThreadingMode::Multi                  => format!("{}.write().unwrap()", expr),
            crate::transpiler::ThreadingMode::Single                 => format!("{}.borrow_mut()", expr),
        }
    }

    /// guard'task read: `.read().await` (multi+async), `.read().unwrap()` (multi+sync), `.borrow()` (single).
    pub(crate) fn guard_task_read_access(&self, expr: &str) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi if self.in_async => format!("{}.read().await", expr),
            crate::transpiler::ThreadingMode::Multi                  => format!("{}.read().unwrap()", expr),
            crate::transpiler::ThreadingMode::Single                 => format!("{}.borrow()", expr),
        }
    }

    // ── Dispatch helpers: var / field ─────────────────────────────────────────
    // These check the tracking sets and route to the right access variant.

    /// Read access for a local/param variable that may be `'actor` or `'actor'task`.
    pub(crate) fn mutex_var_read(&self, var: &str, expr: &str) -> String {
        if self.var_mutex_task_types.contains(var) {
            self.actor_task_read_access(expr)
        } else {
            self.actor_read_access(expr)
        }
    }

    /// Write access for a local/param variable that may be `'actor` or `'actor'task`.
    pub(crate) fn mutex_var_write(&self, var: &str, expr: &str) -> String {
        if self.var_mutex_task_types.contains(var) {
            self.actor_task_write_guard(expr)
        } else {
            self.actor_write_guard(expr)
        }
    }

    /// Write access for a struct field; key = "StructName::field_name".
    pub(crate) fn mutex_field_write(&self, key: &str, expr: &str) -> String {
        if self.struct_mutex_task_fields.contains(key) {
            self.actor_task_write_guard(expr)
        } else {
            self.actor_write_guard(expr)
        }
    }

    /// Read access for a struct field; key = "StructName::field_name".
    pub(crate) fn mutex_field_read(&self, key: &str, expr: &str) -> String {
        if self.struct_mutex_task_fields.contains(key) {
            self.actor_task_read_access(expr)
        } else {
            self.actor_read_access(expr)
        }
    }

    /// Write access for a guard (rwlock) struct field.
    pub(crate) fn rwlock_field_write(&self, key: &str, expr: &str) -> String {
        if self.struct_rwlock_task_fields.contains(key) {
            self.guard_task_write_guard(expr)
        } else {
            self.guard_write_guard(expr)
        }
    }

    /// Read access for a guard (rwlock) struct field.
    pub(crate) fn rwlock_field_read(&self, key: &str, expr: &str) -> String {
        if self.struct_rwlock_task_fields.contains(key) {
            self.guard_task_read_access(expr)
        } else {
            self.guard_read_access(expr)
        }
    }

    /// Emit the managed-mode actor type for anonymous T/T':
    /// multi → `Arc<std::sync::Mutex<T>>` (sync-compatible, no async needed in managed mode),
    /// single → `RefCell<T>` (local interior mutability — not shared, no Rc needed).
    pub(crate) fn emit_managed_actor(&self, inner: &Type) -> String {
        match self.config.threading {
            crate::transpiler::ThreadingMode::Multi  => format!("Arc<std::sync::Mutex<{}>>", self.emit_type(inner)),
            crate::transpiler::ThreadingMode::Single => format!("RefCell<{}>", self.emit_type(inner)),
        }
    }

    /// Returns true when a binding declared `T'guard` or `T'guard'task` should become `Arc<RwLock<T>>`.
    pub(crate) fn is_rwlock_binding(_mutable: bool, ty: &Type) -> bool {
        matches!(ty, Type::Qualified(_, OwnerQual::Guard | OwnerQual::GuardTask))
    }

    pub(crate) fn is_rwlock_task_binding(_mutable: bool, ty: &Type) -> bool {
        matches!(ty, Type::Qualified(_, OwnerQual::GuardTask))
    }

    /// Extract the inner `T` from a `T'guard` or `T'guard'task` rwlock type.
    pub(crate) fn rwlock_inner(ty: &Type) -> Option<&Type> {
        if let Type::Qualified(inner, OwnerQual::Guard | OwnerQual::GuardTask) = ty {
            Some(inner)
        } else {
            None
        }
    }

    /// If `ty` is `T'shared`, `T'actor`, or `T'guard`, return the name of the inner named type.
    /// Used by `pre_scan` to populate `arc_qualified_types`.
    pub(crate) fn arc_inner_type_name(ty: &Type) -> Option<&str> {
        match ty {
            Type::Qualified(inner, OwnerQual::Shared | OwnerQual::Actor | OwnerQual::ActorTask | OwnerQual::Guard | OwnerQual::GuardTask) => {
                if let Type::Named(n) = inner.as_ref() { Some(n.as_str()) } else { None }
            }
            _ => None,
        }
    }

    /// Returns true if the Boring type maps to a `Copy` Rust type.
    /// Determines whether a `transient` field should use `Cell<T>` (Copy) or `RefCell<T>` (!Copy).
    pub(crate) fn is_copy_type(ty: &Type) -> bool {
        match ty {
            Type::Int | Type::Uint | Type::Uint8 | Type::Float | Type::Bool | Type::Nil | Type::Void => true,
            Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128 => true,
            // Lowercase aliases: `int`, `float`, `bool`, `uint` parse as Named in the user source.
            Type::Named(n) => matches!(n.as_str(),
                "int" | "uint" | "uint8" | "float" | "bool"
                | "int8" | "int16" | "int32" | "int64" | "int128"
                | "uint16" | "uint32" | "uint64" | "uint128"
                | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
                | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"),
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
            Type::Int   => "isize".into(),
            Type::Uint  => "usize".into(),
            Type::Uint8 => "u8".into(),
            Type::Int8   => "i8".into(),
            Type::Int16  => "i16".into(),
            Type::Int32  => "i32".into(),
            Type::Int64  => "i64".into(),
            Type::Int128 => "i128".into(),
            Type::Uint16 => "u16".into(),
            Type::Uint32 => "u32".into(),
            Type::Uint64 => "u64".into(),
            Type::Uint128 => "u128".into(),
            Type::Float => "f64".into(),
            Type::Str   => if self.use_rc_str() { "Rc<str>".into() } else { "Arc<str>".into() },
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
                // Priority 4: `dyn Trait` positions — auto-box when T is a known trait name.
                // Params use `impl Trait` (handled in emit_param before calling emit_type).
                // Function return types use `impl Trait` (handled in emit_fn before calling emit_type).
                // All other positions (struct fields, collections, etc.) → Box<dyn Trait>.
                if self.trait_method_names.contains_key(n.as_str()) {
                    return format!("Box<dyn {}>", normalize_type_name(n, self.use_rc_str()));
                }
                // Priority 6: size-based auto-boxing (strict mode only).
                // If the type exceeds stack_auto_bytes, silently promote to Box<T>.
                if self.config.mode == crate::transpiler::TranspileMode::Strict {
                    if let Some(&size) = self.type_sizes.get(n.as_str()) {
                        if size > self.config.stack_auto_bytes {
                            return format!("Box<{}>", normalize_type_name(n, self.use_rc_str()));
                        }
                    }
                }
                normalize_type_name(n, self.use_rc_str())
            }
            Type::TypeParam(n) => n.clone(),
            Type::Optional(inner) => format!("Option<{}>", self.emit_type(inner)),
            Type::Array(inner)    => format!("Vec<{}>", self.emit_type(inner)),
            Type::ArrayN(inner, n) => format!("[{}; {}]", self.emit_type(inner), n),
            Type::ArrayNExpr(inner, _) => format!("[{}; _]", self.emit_type(inner)), // resolved during monomorphisation
            Type::ConstInt(n) => n.to_string(),
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
                format!("{}::{}", normalize_type_name(&base_name, self.use_rc_str()), assoc)
            }
            Type::Impl(inner) => {
                // `impl Trait` — emit the trait name directly (not Box<dyn Trait>).
                let inner_s = match inner.as_ref() {
                    Type::Named(n) => normalize_type_name(n, self.use_rc_str()),
                    other => self.emit_type(other),
                };
                format!("impl {}", inner_s)
            }
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
                    // Managed mode: T' → Arc<Mutex<T>> (multi) or RefCell<T> (single).
                    // Strict mode: Owned = Box<T>, Stack = T (default).
                    if matches!(qual, OwnerQual::Owned) && self.config.mode == crate::transpiler::TranspileMode::Managed
                        && !matches!(inner.as_ref(), Type::Named(n) if self.unit_enums.contains(n.as_str()))
                    {
                        self.emit_managed_actor(inner)
                    } else if matches!(qual, OwnerQual::Owned) {
                        format!("Box<{}>", self.emit_type(inner))
                    } else {
                        // T'stack: explicit stack — skip auto-boxing even if size > threshold.
                        match inner.as_ref() {
                            Type::Named(n) => normalize_type_name(n, self.use_rc_str()),
                            _ => self.emit_type(inner),
                        }
                    }
                }
                // T'shared — threading-aware: Arc<T> (multi) or Rc<T> (single).
                OwnerQual::Shared => {
                    // For T'shared where T is a trait: Arc<dyn Trait> / Rc<dyn Trait>.
                    // Use normalize_type_name (not emit_type) to avoid double-boxing.
                    let dyn_s = if let Type::Named(n) = inner.as_ref() {
                        if self.trait_method_names.contains_key(n.as_str()) {
                            format!("dyn {}", normalize_type_name(n, self.use_rc_str()))
                        } else {
                            self.emit_type(inner)
                        }
                    } else {
                        self.emit_type(inner)
                    };
                    match self.config.threading {
                        crate::transpiler::ThreadingMode::Single => format!("Rc<{}>", dyn_s),
                        crate::transpiler::ThreadingMode::Multi  => format!("Arc<{}>", dyn_s),
                    }
                }
                OwnerQual::Actor     => self.emit_actor_type(inner),
                OwnerQual::ActorTask => self.emit_actor_task_type(inner),
                OwnerQual::Guard     => self.emit_guard_type(inner),
                OwnerQual::GuardTask => self.emit_guard_task_type(inner),
                OwnerQual::Weak    => {
                    match inner.as_ref() {
                        Type::Qualified(base, OwnerQual::Shared) =>
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Single => format!("Weak<{}>", self.emit_type(base)),
                                crate::transpiler::ThreadingMode::Multi  => format!("std::sync::Weak<{}>", self.emit_type(base)),
                            },
                        Type::Qualified(base, OwnerQual::Actor) =>
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Single => format!("Weak<RefCell<{}>>", self.emit_type(base)),
                                crate::transpiler::ThreadingMode::Multi  => format!("std::sync::Weak<std::sync::Mutex<{}>>", self.emit_type(base)),
                            },
                        Type::Qualified(base, OwnerQual::ActorTask) =>
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Single => format!("Weak<RefCell<{}>>", self.emit_type(base)),
                                crate::transpiler::ThreadingMode::Multi  => format!("std::sync::Weak<tokio::sync::Mutex<{}>>", self.emit_type(base)),
                            },
                        Type::Qualified(base, OwnerQual::Guard) =>
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Single => format!("Weak<RefCell<{}>>", self.emit_type(base)),
                                crate::transpiler::ThreadingMode::Multi  => format!("std::sync::Weak<std::sync::RwLock<{}>>", self.emit_type(base)),
                            },
                        Type::Qualified(base, OwnerQual::GuardTask) =>
                            match self.config.threading {
                                crate::transpiler::ThreadingMode::Single => format!("Weak<RefCell<{}>>", self.emit_type(base)),
                                crate::transpiler::ThreadingMode::Multi  => format!("std::sync::Weak<tokio::sync::RwLock<{}>>", self.emit_type(base)),
                            },
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
                OwnerQual::BorrowOption    => format!("&Option<{}>",     self.emit_type(inner)),
                OwnerQual::BorrowOptionMut => format!("&mut Option<{}>", self.emit_type(inner)),
                OwnerQual::BorrowWeak   => format!("&Weak<{}>", self.emit_type(inner)),
                OwnerQual::BorrowShared => match self.config.threading {
                    crate::transpiler::ThreadingMode::Single => format!("&Rc<{}>",  self.emit_type(inner)),
                    crate::transpiler::ThreadingMode::Multi  => format!("&Arc<{}>", self.emit_type(inner)),
                },
                OwnerQual::Lifetime(lt) => {
                    // `mut T&a` is encoded as Lifetime(BorrowMut(T)) → emit `&'a mut T`.
                    if let Type::Qualified(inner2, OwnerQual::BorrowMut) = inner.as_ref() {
                        return format!("&'{} mut {}", lt, self.emit_type(inner2));
                    }
                    // `str` already resolves to `&str` — avoid `&'a &str`.
                    let is_str_slice = matches!(**inner, Type::Str)
                        || matches!(**inner, Type::Named(ref n) if n == "str");
                    if is_str_slice { format!("&'{} str", lt) }
                    else { format!("&'{} {}", lt, self.emit_type(inner)) }
                }
                // Qualifier union / named group (`'one`, `'many`, `'mut`, `'req`, `T'a|b|c`).
                // Emits as the plain inner type — the union is a Boring-level constraint only.
                OwnerQual::Union(_) => self.emit_type(inner),
                // 'new pseudo-qualifier: infer-excluding-stack, emits as Box<T> (same as Owned).
                OwnerQual::New => format!("Box<{}>", self.emit_type(inner)),
                // Host-context `'gpu'unified`/`'gpu'global`: emits as the plain inner
                // type. Confirmed against real usage (`examples/saxpy.br`'s
                // `var [float]'gpu'unified x = [0.0 for ..N]`, freely indexed/assigned
                // as an ordinary host array with no wrapper) — the qualifier only
                // matters at the point the value is passed into a kernel constructor
                // (upload happens there, see `emit_kernel::emit_kernel_construction`)
                // or read from a kernel field inside a `with` block (see
                // `emit_kernel::try_emit_gpu_resident_let`, `emit_stmt::emit_with`);
                // neither needs a special Rust-level wrapper type.
                OwnerQual::GpuUnified | OwnerQual::GpuGlobal => self.emit_type(inner),
                // Kernel-context-only qualifiers with no host-context form (see
                // docs/scoped-access-blocks.md) — still placeholders if ever annotated
                // on an ordinary host variable, which is not a legal/meaningful thing
                // to do in the first place.
                OwnerQual::GpuActorGlobal => format!("*mut {}", self.emit_type(inner)),
                OwnerQual::GpuSync    => format!("*mut {}", self.emit_type(inner)),
                OwnerQual::GpuLocal   => self.emit_type(inner), // local = stack in Rust
                OwnerQual::GpuConst   => format!("*const {}", self.emit_type(inner)),
                OwnerQual::GpuSurface => format!("*mut {}", self.emit_type(inner)),
            }
        }
    }

    /// Like `emit_type` but for struct/enum field positions.
    /// `impl Trait` is not a valid field type in Rust — use `Rc<dyn Fn(...)>` instead.
    /// `Rc` (rather than `Box`) is used so the field is `Clone`, matching auto-derived Clone on
    /// the containing struct/enum.
    pub(crate) fn emit_field_type(&self, ty: &Type, _rebindable: bool) -> String {
        match ty {
            Type::Fn(ret, params, throws, _task, _req) => {
                let ps = params.iter().map(|t| self.emit_type(t)).collect::<Vec<_>>().join(", ");
                let base = ret.as_ref().map(|r| self.emit_type(r)).unwrap_or_else(|| "()".into());
                let r = if *throws {
                    format!("Result<{}, Box<dyn std::error::Error + Send + Sync>>", base)
                } else {
                    base
                };
                // Always Fn (not FnMut): called through Rc so mutation would need RefCell anyway.
                // Rc makes the field Clone without any extra work.
                format!("Rc<dyn Fn({}) -> {}>", ps, r)
            }
            // All bare-T struct fields are always inline in the parent struct's allocation —
            // suppress size-based auto-boxing regardless of rebindability (var/let/mut).
            // A var field is still stored in-place inside the struct; boxing it would add
            // unnecessary indirection and fragment the allocation.
            Type::Named(n) => {
                if let Some(rest) = n.strip_prefix('$') {
                    if let Some((name, _)) = rest.split_once(':') {
                        return name.to_string();
                    }
                }
                if let Some(fn_ty) = self.fn_type_aliases.get(n.as_str()) {
                    return self.emit_type(&fn_ty.clone());
                }
                if self.current_trait_assoc_names.contains(n.as_str()) {
                    return format!("Self::{}", n);
                }
                // Priority 4 (dyn Trait) still applies — correctness constraint.
                if self.trait_method_names.contains_key(n.as_str()) {
                    return format!("Box<dyn {}>", normalize_type_name(n, self.use_rc_str()));
                }
                // Size-based auto-boxing is suppressed: field bytes are part of the parent layout.
                normalize_type_name(n, self.use_rc_str())
            }
            other => self.emit_type(other),
        }
    }

    // ── Structs ───────────────────────────────────────────────────────────────

}
