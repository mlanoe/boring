use super::*;
use super::Transpiler;
use super::helpers::*;

impl Transpiler {
    pub(crate) fn emit_struct(&mut self, s: &StructDecl) {
        if s.is_native { return; }
        // Auto-derive Clone (and Debug) for all structs so that Vec<Struct>.iter().cloned()
        // and {:?} format work without requiring explicit annotations.
        // Skip Clone when any field has a known non-Clone type (e.g. AtomicUsize/AtomicI64).
        const NON_CLONE_TYPES: &[&str] = &[
            "AtomicUsize", "AtomicIsize", "AtomicU8", "AtomicU16", "AtomicU32", "AtomicU64",
            "AtomicI8", "AtomicI16", "AtomicI32", "AtomicI64", "AtomicBool",
        ];
        let has_derive = s.attrs.iter().any(|a| a.name == "derive");
        if !has_derive {
            let has_non_clone_field = s.fields.iter().any(|f| {
                matches!(&f.ty, Type::Named(n) if NON_CLONE_TYPES.contains(&n.as_str()))
            });
            // Don't derive PartialEq when the struct has any comparison operator method —
            // emit_operator_trait_impls will generate PartialEq/PartialOrd impls that would conflict.
            let has_custom_cmp = ["eq", "ne", "lt", "le", "gt", "ge"].iter().any(|m| {
                self.struct_operator_methods.contains(&format!("{}::{}", s.name, m))
            });
            // In non-async multi-thread mode, actor fields map to Arc<std::sync::Mutex<T>> which
            // does NOT implement PartialEq — skip the derive to avoid a compile error.
            let has_sync_mutex_field = !self.use_async_actors()
                && matches!(self.config.threading, crate::transpiler::ThreadingMode::Multi)
                && s.fields.iter().any(|f| {
                    Self::is_arc_qualified(&f.ty)
                    || matches!(&f.ty, Type::Optional(inner) if Self::is_arc_qualified(inner))
                });
            if has_non_clone_field {
                self.line("#[derive(Debug)]");
            } else if has_custom_cmp || has_sync_mutex_field {
                self.line("#[derive(Debug, Clone)]");
            } else {
                self.line("#[derive(Debug, Clone, PartialEq)]");
            }
        }
        for attr in &s.attrs {
            let args_s = if attr.args.is_empty() { String::new() } else { format!("({})", attr.args.join(", ")) };
            self.line(&format!("#[{}{}]", attr.name, args_s));
        }
        let vis = if s.is_pub { "pub " } else { "" };
        let tp = type_params_str(&s.type_params);
        self.line(&format!("{}struct {}{} {{", vis, s.name, tp));
        self.indent += 1;

        // Fields from field declarations
        for f in &s.fields {
            let fvis = if f.is_pub { "pub " } else { "" };
            let ty_s = if f.transient {
                let inner = self.emit_type(&f.ty);
                if Self::is_copy_type(&f.ty) {
                    format!("std::cell::Cell<{}>", inner)
                } else {
                    format!("std::cell::RefCell<{}>", inner)
                }
            } else if let Some(inner) = Self::mutex_inner(&f.ty) {
                self.emit_mutex_type(inner)
            } else {
                let rec_key = format!("{}::{}", s.name, f.name);
                // Inferred actor/guard qualifier (from method/ext-block usage, no explicit
                // annotation on the field) — wrap the declared type to match the access
                // pattern (`.lock()`/`.read()`/`.write()`) that inference already assumes.
                if self.struct_mutex_fields.contains(&rec_key) {
                    self.emit_actor_type(&f.ty)
                } else if self.struct_mutex_task_fields.contains(&rec_key) {
                    self.emit_actor_task_type(&f.ty)
                } else if self.struct_rwlock_fields.contains(&rec_key) {
                    self.emit_guard_type(&f.ty)
                } else if self.struct_rwlock_task_fields.contains(&rec_key) {
                    self.emit_guard_task_type(&f.ty)
                } else {
                    let fmut = if f.mutable { "/* var */ " } else { "" };
                    if self.recursive_fields.contains(&rec_key) {
                        // Recursive struct field — wrap in Box<> to break the infinite-size cycle.
                        match &f.ty {
                            Type::Optional(inner) =>
                                format!("{}Option<Box<{}>>", fmut, self.emit_type(inner)),
                            other =>
                                format!("{}Box<{}>", fmut, self.emit_type(other)),
                        }
                    } else {
                        format!("{}{}", fmut, self.emit_field_type(&f.ty, f.mutable))
                    }
                }
            };
            self.line(&format!("{}{}: {},", fvis, f.name, ty_s));
        }

        // Fields from init (no-body form)
        for init in &s.inits {
            if init.body.is_empty() {
                for p in &init.params {
                    let fvis = if p.is_pub { "pub " } else { "" };
                    let fmut = if p.mutable { "/* var */ " } else { "" };
                    let ty = p.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "/* unknown */".into());
                    self.line(&format!("{}{}{}: {},", fvis, fmut, p.name, ty));
                }
            }
        }

        self.indent -= 1;
        self.line("}");
        self.blank();

        // Auto-emit Display for all structs (delegates to Debug) so that:
        //  - BoringFmt<T: Display> works for Vec<Struct>
        //  - String interpolation can use `{}` on struct values
        // Skip when the type already has (or will get) a Display impl from `as string:` (struct or ext).
        // The `display_types` set is populated during pre_scan for both struct and ext declarations.
        let has_derive_display = s.attrs.iter().any(|a| a.name == "derive" && a.args.iter().any(|arg| arg == "Display"));
        if !self.display_types.contains(&s.name) && !has_derive_display {
            // Build impl type params: add `+ std::fmt::Debug` so that `write!(f, "{:?}", self)` compiles
            // for generic structs (e.g. `impl<T: Clone + Debug> Display for Foo<T>`).
            let tp_impl_disp = if s.type_params.is_empty() {
                String::new()
            } else {
                let bounded: Vec<String> = s.type_params.iter()
                    .map(|p| if p.starts_with('\'') { p.clone() }
                             else if p.starts_with('$') { emit_generic_param(p) }
                             else { format!("{}: Clone + std::fmt::Debug", p) })
                    .collect();
                format!("<{}>", bounded.join(", "))
            };
            let tp_use_disp = type_params_use_str(&s.type_params);
            let impl_header = format!("impl{} std::fmt::Display for {}{} {{", tp_impl_disp, s.name, tp_use_disp);
            self.line(&impl_header);
            self.indent += 1;
            self.line("fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {");
            self.indent += 1;
            self.line("write!(f, \"{:?}\", self)");
            self.indent -= 1;
            self.line("}");
            self.indent -= 1;
            self.line("}");
            self.blank();
        }
        if self.typed_error_enums.contains(&s.name) {
            self.line(&format!("impl std::error::Error for {} {{}}", s.name));
            self.blank();
        }

        // impl block for methods + inits
        let prev = self.self_type.replace(s.name.clone());
        let tp_use = type_params_use_str(&s.type_params);
        // Use impl-bounded type params: `impl<T: Clone>` so method bodies can call `.clone()`.
        let tp_impl = type_params_impl_str(&s.type_params);
        // Register the struct's type params as impl-level so that methods
        // with the same type param name don't re-declare them (E0403).
        let prev_impl_type_params = std::mem::replace(&mut self.impl_type_params, s.type_params.clone());
        self.line(&format!("impl{} {}{} {{", tp_impl, s.name, tp_use));
        self.indent += 1;

        // Constructors
        for init in &s.inits {
            self.emit_init(init, &s.name, &s.fields);
        }

        // Auto-generate new() if no explicit init but all fields have defaults.
        if s.inits.is_empty() && !s.fields.is_empty()
            && s.fields.iter().all(|f| f.default.is_some() || matches!(&f.ty, Type::Optional(_)))
        {
            self.line("pub fn new() -> Self {");
            self.indent += 1;
            self.line(&format!("{} {{", s.name));
            self.indent += 1;
            for f in &s.fields {
                let struct_name_ref = s.name.clone();
                let key = format!("{}::{}", struct_name_ref, f.name);
                let is_transient = self.transient_fields.contains_key(&key);
                // These sets cover both explicit (`T'actor`) and inferred qualifiers.
                let is_actor = self.struct_mutex_fields.contains(&key);
                let is_actor_task = self.struct_mutex_task_fields.contains(&key);
                let is_guard = self.struct_rwlock_fields.contains(&key);
                let is_guard_task = self.struct_rwlock_task_fields.contains(&key);
                if let Some(def) = &f.default {
                    if is_actor || is_actor_task {
                        // Inferred fields carry the bare inner type already; explicit
                        // `T'actor`/`T'actor'task` fields need unwrapping via mutex_inner.
                        let inner = Self::mutex_inner(&f.ty).unwrap_or(&f.ty);
                        let raw = self.emit_let_value(Some(inner), def);
                        let init = if is_actor_task { self.emit_actor_task_new(&raw) } else { self.emit_actor_new(&raw) };
                        self.line(&format!("{}: {},", f.name, init));
                    } else if is_guard || is_guard_task {
                        let inner = Self::rwlock_inner(&f.ty).unwrap_or(&f.ty);
                        let raw = self.emit_let_value(Some(inner), def);
                        let init = if is_guard_task { self.emit_guard_task_new(&raw) } else { self.emit_guard_new(&raw) };
                        self.line(&format!("{}: {},", f.name, init));
                    } else {
                        let val = self.emit_let_value(Some(&f.ty), def);
                        if is_transient {
                            let is_copy = Self::is_copy_type(&f.ty);
                            let wrapped = if is_copy {
                                format!("std::cell::Cell::new({})", val)
                            } else {
                                format!("std::cell::RefCell::new({})", val)
                            };
                            self.line(&format!("{}: {},", f.name, wrapped));
                        } else {
                            self.line(&format!("{}: {},", f.name, val));
                        }
                    }
                } else if is_transient {
                    // transient fields without explicit default → Cell/RefCell::new(None)
                    let is_copy = Self::is_copy_type(&f.ty);
                    let init = if is_copy { "std::cell::Cell::new(None)" } else { "std::cell::RefCell::new(None)" };
                    self.line(&format!("{}: {},", f.name, init));
                } else {
                    self.line(&format!("{}: None,", f.name));
                }
            }
            self.indent -= 1;
            self.line("}");
            self.indent -= 1;
            self.line("}");
            self.blank();
        }

        // Collect every method name that belongs to at least one declared protocol.
        // Done eagerly (cloned into an owned set) to avoid holding a borrow on `self`
        // across the subsequent `emit_fn` calls.
        let trait_method_set: std::collections::HashSet<String> = s.protocols.iter()
            .flat_map(|proto| {
                self.trait_method_names
                    .get(proto.as_str())
                    .into_iter()
                    .flat_map(|names| names.iter().cloned())
            })
            .collect();
        // Emit plain (non-trait) methods in `impl Struct {}`.
        // If there are no declared protocols, emit everything here (no trait to split against).
        let has_protocols = !s.protocols.is_empty();
        for f in &s.methods {
            // Skip methods overridden by an ext block — the ext block's version takes precedence.
            let overridden = self.struct_ext_method_overrides
                .contains(&format!("{}::{}", s.name, f.name));
            if overridden { continue; }
            if !has_protocols || !trait_method_set.contains(&f.name) {
                self.emit_fn(f, Some(&s.name));
                self.blank();
            }
        }

        // Setters → fn set_X(&mut self, val: T)
        for setter in &s.setters {
            self.emit_setter(setter);
            self.blank();
        }

        // Type methods (no self) + type consts
        for tc in &s.type_vars {
            self.emit_type_var_const(tc);
            self.blank();
        }
        for tm in &s.type_methods {
            self.emit_type_method(tm, &s.name);
            self.blank();
        }

        // as-conversions — skip `as string` (emitted as Display impl below)
        for conv in &s.conversions {
            if !Self::is_string_conversion(conv) {
                self.emit_conversion(conv);
                self.blank();
            }
        }

        self.indent -= 1;
        self.line("}");

        // Protocol conformances declared in header: `struct X as Trait1, Trait2:`.
        // Emit `impl Trait for X { methods... }` so all trait methods are fulfilled.
        for proto in &s.protocols {
            {
                self.blank();
                self.line(&format!("impl{} {} for {}{} {{", tp, proto, s.name, tp_use));
                self.indent += 1;
                // Emit associated type bindings that this trait requires.
                let trait_assoc_names: Option<std::collections::HashSet<String>> =
                    self.trait_assoc_type_names.get(proto.as_str()).cloned();
                // Make assoc type names available to emit_type so it can emit `Self::Foo`.
                let prev_assoc_names = std::mem::replace(
                    &mut self.current_trait_assoc_names,
                    trait_assoc_names.clone().unwrap_or_default(),
                );
                for atd in &s.assoc_type_defs {
                    let belongs = trait_assoc_names.as_ref()
                        .is_some_and(|names| names.contains(&atd.name));
                    if belongs {
                        let ty_s = self.emit_type(&atd.ty);
                        self.line(&format!("type {} = {};", atd.name, ty_s));
                    }
                }
                // Emit only the struct methods that belong to this trait.
                // If the trait is unknown (no declaration in scope), fall back to all methods.
                // Clone the known names eagerly to avoid a borrow-checker conflict with emit_fn.
                let known: Option<std::collections::HashSet<String>> =
                    self.trait_method_names.get(proto.as_str()).cloned();
                let prev_in_trait = self.inside_trait_impl;
                self.inside_trait_impl = true;
                for f in &s.methods {
                    if known.as_ref().is_none_or(|names| names.contains(&f.name)) {
                        self.emit_fn(f, Some(&s.name));
                        self.blank();
                    }
                }
                self.inside_trait_impl = prev_in_trait;
                self.current_trait_assoc_names = prev_assoc_names;
                self.indent -= 1;
                self.line("}");
            }
        }

        // Implicit trait conformance: if the struct has all methods required by a trait
        // (and hasn't already declared `as Trait`), auto-emit `impl Trait for Struct`.
        {
            let struct_method_names: std::collections::HashSet<String> =
                s.methods.iter().map(|m| m.name.clone()).collect();
            let already_conforms: std::collections::HashSet<String> =
                s.protocols.iter().cloned().collect();
            // Clone to avoid borrow conflict with emit_fn (which borrows self mutably).
            let trait_names: Vec<String> = self.trait_method_names.keys().cloned().collect();
            for trait_name in &trait_names {
                // Skip traits explicitly declared in the header (already emitted above).
                if already_conforms.contains(trait_name.as_str()) { continue; }
                let required: std::collections::HashSet<String> =
                    self.trait_method_names.get(trait_name.as_str())
                        .cloned()
                        .unwrap_or_default();
                if required.is_empty() { continue; }
                // Don't auto-generate if all the required methods are already claimed by
                // some explicit protocol impl — that would create duplicate method impls.
                let already_covered = already_conforms.iter().any(|explicit_proto| {
                    let explicit_required = self.trait_method_names.get(explicit_proto.as_str())
                        .cloned()
                        .unwrap_or_default();
                    required.iter().all(|m| explicit_required.contains(m))
                });
                if already_covered { continue; }
                // Check that the struct has every method the trait requires.
                if required.iter().all(|m| struct_method_names.contains(m)) {
                    self.blank();
                    self.line(&format!("impl{} {} for {}{} {{", tp, trait_name, s.name, tp_use));
                    self.indent += 1;
                    let prev_in_trait = self.inside_trait_impl;
                    self.inside_trait_impl = true;
                    for f in &s.methods {
                        if required.contains(&f.name) {
                            self.emit_fn(f, Some(&s.name));
                            self.blank();
                        }
                    }
                    self.inside_trait_impl = prev_in_trait;
                    self.indent -= 1;
                    self.line("}");
                }
            }
        }

        // Display impl for `as string:` conversions
        for conv in &s.conversions {
            if Self::is_string_conversion(conv) {
                self.blank();
                self.emit_display_impl(conv, &s.name, &s.type_params);
            }
        }

        // Module-level statics for `type var` fields
        for tv in &s.type_vars {
            if tv.mutable {
                let vis = if tv.is_pub { "pub " } else { "" };
                let ty_s = tv.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "/* unknown */".to_string());
                let val_s = self.emit_let_value(tv.ty.as_ref(), &tv.default);
                self.blank();
                self.line(&format!(
                    "{}static {}: std::sync::Mutex<{}> = std::sync::Mutex::new({});",
                    vis, tv.name.to_uppercase(), ty_s, val_s
                ));
            }
        }

        self.self_type = prev;
        self.impl_type_params = prev_impl_type_params;
    }

    pub(crate) fn is_string_conversion(conv: &AsDecl) -> bool {
        matches!(&conv.ty, Type::Str)
            || matches!(&conv.ty, Type::Named(n) if n == "string")
    }

    pub(crate) fn emit_display_impl(&mut self, conv: &AsDecl, type_name: &str, type_params: &[String]) {
        let tp = type_params_str(type_params);
        let tp_use = type_params_use_str(type_params);
        self.line(&format!("impl{} std::fmt::Display for {}{} {{", tp, type_name, tp_use));
        self.indent += 1;
        self.line("fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {");
        self.indent += 1;
        let body = &conv.body;
        if body.is_empty() {
            self.line("write!(f, \"\")");
        } else {
            let last = body.len() - 1;
            for (i, stmt) in body.iter().enumerate() {
                if i == last {
                    // Normalize: `return expr` → treat as expression stmt for Display purposes.
                    let effective_stmt: &Stmt;
                    let owned_expr: Stmt;
                    if let Stmt::Return(r) = stmt {
                        if let Some(e) = &r.value {
                            owned_expr = Stmt::Expr(e.clone());
                            effective_stmt = &owned_expr;
                        } else {
                            effective_stmt = stmt;
                        }
                    } else {
                        effective_stmt = stmt;
                    }
                    match effective_stmt {
                        Stmt::Expr(e) => {
                            match &e.kind {
                                ExprKind::StringInterp(segs) => {
                                    let (fmt_s, args) = self.build_format_string(segs);
                                    let args_str = if args.is_empty() {
                                        String::new()
                                    } else {
                                        format!(", {}", args.join(", "))
                                    };
                                    self.line(&format!("write!(f, \"{}\"{})", fmt_s, args_str));
                                }
                                ExprKind::Str(s) => {
                                    self.line(&format!("write!(f, \"{}\")", escape_str(s)));
                                }
                                _ => {
                                    let s = self.emit_expr(e);
                                    self.line(&format!("write!(f, \"{{}}\", {})", s));
                                }
                            }
                        }
                        _ => {
                            // For other statement forms (e.g. Stmt::Match returning a string),
                            // emit as a value block: `write!(f, "{}", { body... })`
                            // This handles match/if expressions as the last statement.
                            // We temporarily emit as a value-expression.
                            // Emit previous (non-last) stmts as normal, then wrap last in write!.
                            // Since this is a single stmt (i == last && i == 0 for typical conversions),
                            // emit it as an expression via a block.
                            let prev_void = self.fn_returns_void;
                            let prev_throws = self.in_throws;
                            self.fn_returns_void = false;
                            self.in_throws = false;
                            self.line("let __disp_val = {");
                            self.indent += 1;
                            self.emit_stmt(stmt, true);
                            self.indent -= 1;
                            self.line("};");
                            self.fn_returns_void = prev_void;
                            self.in_throws = prev_throws;
                            self.line("write!(f, \"{}\", __disp_val)");
                        }
                    }
                } else {
                    self.emit_stmt(stmt, false);
                }
            }
        }
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    pub(crate) fn emit_init(&mut self, init: &InitDecl, struct_name: &str, fields: &[FieldDecl]) {
        if init.body.is_empty() {
            // Auto-field init: fn new(x: T, y: T) -> Self { Self { x, y } }
            let params_s: Vec<String> = init.params.iter().map(|p| {
                let ty = p.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "/* unknown */".into());
                let mut_kw = if p.mutable { "mut " } else { "" };
                format!("{}{}:{}", mut_kw, p.name, ty)
            }).collect();
            // Pad: field has default but param doesn't need to be listed... keep simple
            let params_joined = params_s.join(", ");
            self.line(&format!("pub fn new({}) -> Self {{", params_joined));
            self.indent += 1;
            self.line(&format!("{} {{", struct_name));
            self.indent += 1;
            for p in &init.params {
                self.line(&format!("{},", p.name));
            }
            // Fields with defaults not in init
            for f in fields {
                if !init.params.iter().any(|p| p.name == f.name) {
                    if let Some(def) = &f.default {
                        let key = format!("{}::{}", struct_name, f.name);
                        let is_actor = self.struct_mutex_fields.contains(&key);
                        let is_actor_task = self.struct_mutex_task_fields.contains(&key);
                        let is_guard = self.struct_rwlock_fields.contains(&key);
                        let is_guard_task = self.struct_rwlock_task_fields.contains(&key);
                        if is_actor || is_actor_task {
                            let inner = Self::mutex_inner(&f.ty).unwrap_or(&f.ty);
                            let raw = self.emit_let_value(Some(inner), def);
                            let init = if is_actor_task { self.emit_actor_task_new(&raw) } else { self.emit_actor_new(&raw) };
                            self.line(&format!("{}: {},", f.name, init));
                        } else if is_guard || is_guard_task {
                            let inner = Self::rwlock_inner(&f.ty).unwrap_or(&f.ty);
                            let raw = self.emit_let_value(Some(inner), def);
                            let init = if is_guard_task { self.emit_guard_task_new(&raw) } else { self.emit_guard_new(&raw) };
                            self.line(&format!("{}: {},", f.name, init));
                        } else {
                            let val = self.emit_let_value(Some(&f.ty), def);
                            if f.transient {
                                let is_copy = Self::is_copy_type(&f.ty);
                                let wrapped = if is_copy {
                                    format!("std::cell::Cell::new({})", val)
                                } else {
                                    format!("std::cell::RefCell::new({})", val)
                                };
                                self.line(&format!("{}: {},", f.name, wrapped));
                            } else {
                                self.line(&format!("{}: {},", f.name, val));
                            }
                        }
                    }
                }
            }
            self.indent -= 1;
            self.line("}");
            self.indent -= 1;
            self.line("}");
        } else {
            // Body init: fn new(params) -> Self { body }
            let params_s: Vec<String> = init.params.iter().map(|p| {
                let ty = p.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "/* unknown */".into());
                format!("{}: {}", p.name, ty)
            }).collect();
            self.line(&format!("pub fn new({}) -> Self {{", params_s.join(", ")));
            self.indent += 1;
            // Add init params to known_local_vars so bare param names don't resolve to self.field.
            for p in &init.params { self.known_local_vars.insert(p.name.clone()); }
            // Check if the body consists entirely of `self.field = expr` assignments.
            // If so, emit a struct literal instead of using `self` directly.
            let all_self_assigns = init.body.iter().all(|stmt| {
                matches!(stmt, Stmt::Expr(e) | Stmt::Return(ReturnStmt { value: Some(e), .. })
                    if matches!(&e.kind, ExprKind::Assign(target, _)
                        if matches!(&target.kind, ExprKind::Field(obj, _)
                            if matches!(&obj.kind, ExprKind::Var(v) if v == "self"))))
            });
            if all_self_assigns && !init.body.is_empty() {
                // Collect field → value assignments in order.
                let mut field_vals: Vec<(String, String)> = Vec::new();
                for stmt in &init.body {
                    let e = match stmt {
                        Stmt::Expr(e) => e,
                        Stmt::Return(ReturnStmt { value: Some(e), .. }) => e,
                        _ => continue,
                    };
                    if let ExprKind::Assign(target, value) = &e.kind {
                        if let ExprKind::Field(_, field) = &target.kind {
                            let val_s = self.emit_expr(value);
                            field_vals.push((field.clone(), val_s));
                        }
                    }
                }
                // Also collect fields with known defaults that weren't assigned.
                let assigned: std::collections::HashSet<&str> = field_vals.iter()
                    .map(|(f, _)| f.as_str()).collect();
                let mut extras: Vec<String> = Vec::new();
                for f in fields {
                    if !assigned.contains(f.name.as_str()) {
                        if let Some(def) = &f.default {
                            let val = self.emit_let_value(Some(&f.ty), def);
                            extras.push(format!("{}: {}", f.name, val));
                        } else if matches!(f.ty, Type::Optional(_)) {
                            extras.push(format!("{}: None", f.name));
                        } else {
                            // Zero-initialize numeric/bool fields
                            let zero = match &f.ty {
                                Type::Float32 | Type::Float64 => "0.0".to_string(),
                                Type::Bool  => "false".to_string(),
                                Type::Str   => "Arc::<str>::from(\"\")".to_string(),
                                _           => "Default::default()".to_string(),
                            };
                            extras.push(format!("{}: {}", f.name, zero));
                        }
                    }
                }
                let mut all_fields: Vec<String> = field_vals.iter()
                    .map(|(f, v)| format!("{}: {}", f, v)).collect();
                all_fields.extend(extras);
                self.line(&format!("{} {{ {} }}", struct_name, all_fields.join(", ")));
            } else {
                // General case: emit body as-is (may use self).
                // Wrap in a local `let mut __self = ...` pattern.
                let zero_fields: Vec<String> = fields.iter().map(|f| {
                    let zero = match &f.ty {
                        Type::Float32 | Type::Float64 => "0.0".to_string(),
                        Type::Bool  => "false".to_string(),
                        Type::Str   => "Arc::<str>::from(\"\")".to_string(),
                        _           => "Default::default()".to_string(),
                    };
                    format!("{}: {}", f.name, zero)
                }).collect();
                if !zero_fields.is_empty() {
                    self.line(&format!("let mut __self = {} {{ {} }};", struct_name, zero_fields.join(", ")));
                } else {
                    self.line(&format!("let mut __self = {} {{}};", struct_name));
                }
                // Emit body with `self` rewritten to `__self`.
                // All statements must be terminated with `;` — the actual return
                // value is the `__self` local emitted explicitly below.
                let prev_init = std::mem::replace(&mut self.in_init_body, true);
                for stmt in &init.body {
                    self.emit_stmt(stmt, false);
                }
                self.in_init_body = prev_init;
                self.line("__self");
            }
            self.indent -= 1;
            self.line("}");
            // Clean up init param names from known_local_vars.
            for p in &init.params { self.known_local_vars.remove(p.name.as_str()); }
        }
        self.blank();
    }

    pub(crate) fn emit_type_var_const(&mut self, tv: &TypeVar) {
        let vis = if tv.is_pub { "pub " } else { "" };
        let ty_s = tv.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "/* unknown */".to_string());
        let val_s = self.emit_let_value(tv.ty.as_ref(), &tv.default);
        if tv.mutable {
            // type var is emitted as a module-level static Mutex — see below impl block.
            // We emit a marker comment here so the impl block is clean.
            self.line(&format!("// type var {}: {} = {} — see static {} below", tv.name, ty_s, val_s, tv.name.to_uppercase()));
        } else {
            // type let → associated const
            self.line(&format!("{}const {}: {} = {};", vis, tv.name.to_uppercase(), ty_s, val_s));
        }
    }

    pub(crate) fn emit_type_method(&mut self, tm: &TypeMethod, struct_name: &str) {
        use crate::ast::TypeMethodKind;
        let vis = if tm.is_pub { "pub " } else { "" };
        let (fn_name, params_s) = match tm.kind {
            TypeMethodKind::Set => {
                let p = tm.params.first().map(|p| {
                    let ty = p.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_default();
                    format!("{}: {}", p.name, ty)
                }).unwrap_or_default();
                (format!("set_{}", tm.name), p)
            }
            _ => {
                let ps: Vec<String> = tm.params.iter().map(|p| {
                    let ty = p.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_default();
                    format!("{}: {}", p.name, ty)
                }).collect();
                (tm.name.clone(), ps.join(", "))
            }
        };
        let ret_ty = tm.return_ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "()".into());
        let ret_s = if ret_ty == "()" { String::new() } else { format!(" -> {}", ret_ty) };
        let _ = struct_name;
        self.line(&format!("{}fn {}({}){} {{", vis, fn_name, params_s, ret_s));
        self.indent += 1;
        let prev_setter = std::mem::replace(&mut self.in_type_setter, matches!(tm.kind, TypeMethodKind::Set));
        self.emit_body(&tm.body);
        self.in_type_setter = prev_setter;
        self.indent -= 1;
        self.line("}");
    }

    pub(crate) fn emit_setter(&mut self, setter: &SetDecl) {
        let vis = if setter.is_pub { "pub " } else { "" };
        let param = format!("{}: {}", setter.param_name, self.emit_type(&setter.param_ty));
        self.line(&format!("{}fn set_{}(&mut self, {}) {{", vis, setter.name, param));
        self.indent += 1;
        self.emit_body(&setter.body);
        self.indent -= 1;
        self.line("}");
    }

    /// If the conversion body is a bare `self.field` access, return the Rust field name.
    /// In that case we can emit a reference-returning pair (&T / &mut T) instead of a
    /// value-returning method (T by clone).
    pub(crate) fn self_field_access_body(body: &[Stmt]) -> Option<String> {
        if let [Stmt::Expr(e)] = body {
            if let ExprKind::Field(obj, field) = &e.kind {
                if matches!(&obj.kind, ExprKind::Var(v) if v == "self") {
                    return Some(field.clone());
                }
            }
        }
        None
    }

    pub(crate) fn emit_conversion(&mut self, conv: &AsDecl) {
        let ty = self.emit_type(&conv.ty);
        let method_name = format!("into_{}", ty.to_lowercase());
        let vis = if conv.is_pub { "pub " } else { "" };

        if let Some(field) = Self::self_field_access_body(&conv.body) {
            // Body is `self.field` — emit a value-returning immutable method (for `d as T` casts)
            // plus a mutable-reference method `into_T_mut` (for mutation of the inner field).
            self.line(&format!("{}fn {}(&self) -> {} {{", vis, method_name, ty));
            self.indent += 1;
            self.line(&format!("self.{}.clone()", field));
            self.indent -= 1;
            self.line("}");
            self.blank();
            let mut_method = format!("{}_mut", method_name);
            self.line(&format!("{}fn {}(&mut self) -> &mut {} {{", vis, mut_method, ty));
            self.indent += 1;
            self.line(&format!("&mut self.{}", field));
            self.indent -= 1;
            self.line("}");
        } else {
            // Computed body — emit a single value-returning method.
            self.line(&format!("{}fn {}(&self) -> {} {{", vis, method_name, ty));
            self.indent += 1;
            self.emit_body(&conv.body);
            self.indent -= 1;
            self.line("}");
        }
    }

    // ── Enums ─────────────────────────────────────────────────────────────────

    pub(crate) fn emit_enum(&mut self, e: &EnumDecl) {
        if e.is_native { return; }
        // Always derive Clone so Vec<EnumType>.iter().cloned() works.
        // When used as a typed error (`throws CalcError`), also derive Debug (required by Error).
        let is_error_type = self.typed_error_enums.contains(&e.name);
        let has_clone_derive = e.attrs.iter().any(|a| a.name == "derive" && a.args.iter().any(|arg| arg.contains("Clone")));
        let has_debug_derive = e.attrs.iter().any(|a| a.name == "derive" && a.args.iter().any(|arg| arg.contains("Debug")));
        // Pre-check for @error variant attrs so we know whether thiserror will add Debug.
        let has_variant_error_attr = e.variants.iter().any(|v| v.attrs.iter().any(|a| a.name == "error"));
        // Non-parametric enums (all unit variants) are inferred as Copy.
        let is_unit_enum = self.unit_enums.contains(&e.name);
        // Non-unit enums can derive PartialEq only when no variant field is actor/shared
        // (those map to Rc<RefCell<T>>/Arc<Mutex<T>> which don't implement PartialEq).
        let has_actor_field = !is_unit_enum && e.variants.iter().any(|v| {
            v.fields.iter().any(|f| matches!(&f.ty,
                Type::Qualified(_, OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Shared)))
        });
        if !has_clone_derive {
            // When thiserror will auto-inject Debug below, omit Debug here to avoid duplicates.
            let thiserror_will_add_debug = has_variant_error_attr && !has_debug_derive;
            if is_unit_enum {
                if thiserror_will_add_debug {
                    self.line("#[derive(Clone, Copy, PartialEq, Eq, Hash)]");
                } else {
                    self.line("#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]");
                }
            } else if has_actor_field {
                if thiserror_will_add_debug {
                    self.line("#[derive(Clone)]");
                } else {
                    self.line("#[derive(Debug, Clone)]");
                }
            } else {
                if thiserror_will_add_debug {
                    self.line("#[derive(Clone, PartialEq)]");
                } else {
                    self.line("#[derive(Debug, Clone, PartialEq)]");
                }
            }
        }
        // Auto-inject thiserror when any variant has @error("..."), unless already
        // declared explicitly via @derive(thiserror::Error, ...).
        let already_has_thiserror = e.attrs.iter().any(|a| {
            a.name == "derive" && a.args.iter().any(|arg| arg.contains("thiserror"))
        });
        if has_variant_error_attr && !already_has_thiserror {
            self.uses_thiserror.set(true);
            // thiserror requires Debug; add it if not already present.
            if !has_debug_derive {
                self.line("#[derive(Debug, thiserror::Error)]");
            } else {
                self.line("#[derive(thiserror::Error)]");
            }
        }
        for attr in &e.attrs {
            let args_s = if attr.args.is_empty() { String::new() } else { format!("({})", attr.args.join(", ")) };
            self.line(&format!("#[{}{}]", attr.name, args_s));
        }
        let vis = if e.is_pub { "pub " } else { "" };
        let tp = type_params_str(&e.type_params);
        self.line(&format!("{}enum {}{} {{", vis, e.name, tp));
        self.indent += 1;
        for v in &e.variants {
            // Per-variant attributes: @error("msg") → #[error("msg")]
            for attr in &v.attrs {
                let args_s = if attr.args.is_empty() {
                    String::new()
                } else {
                    format!("({})", attr.args.join(", "))
                };
                self.line(&format!("#[{}{}]", attr.name, args_s));
            }
            if v.fields.is_empty() {
                self.line(&format!("{},", v.name));
            } else {
                // Always emit tuple-style so match patterns Variant(x, y) work directly.
                // Unwrap `Owned`/Box qualifiers for enum variant fields (non-recursive) so that
                // nested pattern matching works: `Wrap(A(n))` would fail if the field is Box<T>.
                // Recursive enums that genuinely need Box wrap themselves (Box<Self>); other
                // `T'` annotations on variant fields are treated as plain T to avoid pattern issues.
                let fields_s: Vec<String> = v.fields.iter()
                    .enumerate()
                    .map(|(idx, f)| {
                        let rec_key = format!("{}::{}::{}", e.name, v.name, idx);
                        let is_recursive = self.recursive_fields.contains(&rec_key);
                        if is_recursive {
                            // Auto-inferred recursive field — wrap in Box to break the cycle.
                            format!("Box<{}>", self.emit_type(&f.ty))
                        } else {
                            // Non-recursive: unwrap explicit Owned/Box qualifiers so that
                            // nested pattern matching works without Box wrapping.
                            let unwrapped = match &f.ty {
                                Type::Qualified(inner, OwnerQual::Owned) => inner.as_ref().clone(),
                                other => other.clone(),
                            };
                            self.emit_field_type(&unwrapped, true)
                        }
                    })
                    .collect();
                self.line(&format!("{}({}),", v.name, fields_s.join(", ")));
            }
        }
        self.indent -= 1;
        self.line("}");
        self.blank();

        // Typed error enum: emit Display + Error so `throws CalcError` compiles.
        // Skip when thiserror is in use — it generates Display + Error automatically.
        if is_error_type && !has_variant_error_attr {
            let name = &e.name;
            self.line(&format!("impl std::fmt::Display for {} {{", name));
            self.indent += 1;
            self.line("fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {");
            self.indent += 1;
            self.line("write!(f, \"{:?}\", self)");
            self.indent -= 1;
            self.line("}");
            self.indent -= 1;
            self.line("}");
            self.line(&format!("impl std::error::Error for {} {{}}", name));
            self.blank();
        }

        // Collect named variant fields to generate getters for direct field access.
        // e.g. `enum EPair { Both(int first, int second) }` → `fn first(&self) -> i64`
        // Group by field name to avoid duplicate method definitions when multiple variants
        // share the same field name (e.g. `Add(int left, int right)` and `Mul(int left, int right)`).
        #[allow(clippy::type_complexity)]
        let mut named_field_getters: Vec<(String, Vec<(usize, &VariantField, &str)>)> = Vec::new();
        for v in &e.variants {
            for (idx, field) in v.fields.iter().enumerate() {
                if let Some(ref fname) = field.name {
                    if let Some(existing) = named_field_getters.iter_mut().find(|(n, _)| n == fname) {
                        existing.1.push((idx, field, v.name.as_str()));
                    } else {
                        named_field_getters.push((fname.clone(), vec![(idx, field, v.name.as_str())]));
                    }
                }
            }
        }
        // Skip getters where variants share the same field name but have DIFFERENT types —
        // a single getter cannot return multiple types without generics.
        named_field_getters.retain(|(_, cases)| {
            if cases.len() <= 1 { return true; }
            let first_ty = &cases[0].1.ty;
            cases.iter().all(|(_, f, _)| &f.ty == first_ty)
        });

        // impl block
        let prev = self.self_type.replace(e.name.clone());
        let tp_use = type_params_use_str(&e.type_params);
        let has_named_fields = !named_field_getters.is_empty();
        // Build set of method names belonging to declared protocols, so we can split them.
        let proto_method_set: std::collections::HashSet<String> = e.protocols.iter()
            .flat_map(|proto| {
                self.trait_method_names.get(proto.as_str())
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
            })
            .collect();
        let has_protocols = !e.protocols.is_empty();
        // Methods NOT in any protocol go in `impl EnumName {}`.
        let plain_methods: Vec<&FnDecl> = e.methods.iter()
            .filter(|f| !has_protocols || !proto_method_set.contains(&f.name))
            .collect();
        if !plain_methods.is_empty() || !e.conversions.is_empty() || !e.setters.is_empty() || has_named_fields {
            self.line(&format!("impl{} {}{} {{", tp, e.name, tp_use));
            self.indent += 1;
            for f in &plain_methods {
                self.emit_fn(f, Some(&e.name));
                self.blank();
            }
            for setter in &e.setters {
                self.emit_setter(setter);
                self.blank();
            }
            for conv in &e.conversions {
                // Skip `as string:` — emitted as Display impl below (not as into_* method).
                if !Self::is_string_conversion(conv) {
                    self.emit_conversion(conv);
                    self.blank();
                }
            }
            // Generate field accessor methods for named variant fields.
            for (fname, cases) in &named_field_getters {
                // Use the return type from the first case (assume consistent types across variants).
                // Unwrap Owned qualifier so the return type is T instead of Box<T> — enum variant
                // fields that had `own T` should store T directly (Box wrapping was removed from
                // the variant definition), so the accessor must also return T.
                let raw_field_ty = &cases[0].1.ty;
                let unwrapped_field_ty = match raw_field_ty {
                    Type::Qualified(inner, OwnerQual::Owned) => *inner.clone(),
                    other => other.clone(),
                };
                let ret_ty = self.emit_type(&unwrapped_field_ty);
                self.line(&format!("fn {}(&self) -> Option<{}> {{", fname, ret_ty));
                self.indent += 1;
                // Generate one if-let arm per variant that has this field.
                // Only use double-deref for fields actually stored as Box<T> (recursive fields).
                // Explicit Owned/Box qualifiers on non-recursive fields are unwrapped at storage
                // time, so the Rust field is T, not Box<T> — a single clone() suffices.
                let field_is_boxed = cases.iter().any(|(idx, _, variant_name)| {
                    let rec_key = format!("{}::{}::{}", e.name, variant_name, idx);
                    self.recursive_fields.contains(&rec_key)
                });
                let clone_expr = if field_is_boxed { "(**__fv).clone()" } else { "__fv.clone()" };
                for (idx, _field, variant_name) in cases {
                    let n_fields = e.variants.iter()
                        .find(|v| v.name == *variant_name)
                        .map(|v| v.fields.len())
                        .unwrap_or(1);
                    let pats: Vec<String> = (0..n_fields).map(|i| {
                        if i == *idx { "__fv".to_string() } else { "_".to_string() }
                    }).collect();
                    self.line(&format!("if let {}::{}({}) = self {{ return Some({}); }}",
                        e.name, variant_name, pats.join(", "), clone_expr));
                }
                self.line("None");
                self.indent -= 1;
                self.line("}");
                self.blank();
                // Register as an enum field getter so `var.field` → `var.field().expect(...)`.
                let getter_key = format!("{}::{}", e.name, fname);
                self.enum_field_getters.insert(getter_key);
            }
            self.indent -= 1;
            self.line("}");
        }
        // Display impl for `as string:` conversions
        for conv in &e.conversions {
            if Self::is_string_conversion(conv) {
                self.blank();
                self.emit_display_impl(conv, &e.name, &e.type_params);
            }
        }

        for proto in &e.protocols {
            self.blank();
            self.line(&format!("impl{} {} for {}{} {{", tp, proto, e.name, tp_use));
            self.indent += 1;
            let required = self.trait_method_names.get(proto.as_str()).cloned();
            for f in &e.methods {
                let in_trait = required.as_ref()
                    .map(|r| r.contains(f.name.as_str()))
                    .unwrap_or(true);
                if in_trait {
                    self.inside_trait_impl = true;
                    self.emit_fn(f, Some(&e.name));
                    self.inside_trait_impl = false;
                    self.blank();
                }
            }
            self.indent -= 1;
            self.line("}");
        }
        // Implicit trait conformance for enums: if the enum has all methods required by a trait
        // (and hasn't already declared `as Trait`), auto-emit `impl Trait for Enum`.
        // This mirrors the same logic in emit_struct for structural conformance.
        {
            let enum_method_names: std::collections::HashSet<String> =
                e.methods.iter().map(|m| m.name.clone()).collect();
            let already_conforms: std::collections::HashSet<String> =
                e.protocols.iter().cloned().collect();
            let trait_names: Vec<String> = self.trait_method_names.keys().cloned().collect();
            for trait_name in &trait_names {
                if already_conforms.contains(trait_name.as_str()) { continue; }
                let required: std::collections::HashSet<String> =
                    self.trait_method_names.get(trait_name.as_str())
                        .cloned()
                        .unwrap_or_default();
                if required.is_empty() { continue; }
                if required.iter().all(|m| enum_method_names.contains(m)) {
                    self.blank();
                    self.line(&format!("impl{} {} for {}{} {{", tp, trait_name, e.name, tp_use));
                    self.indent += 1;
                    let prev_in_trait = self.inside_trait_impl;
                    self.inside_trait_impl = true;
                    for f in &e.methods {
                        if required.contains(&f.name) {
                            self.emit_fn(f, Some(&e.name));
                            self.blank();
                        }
                    }
                    self.inside_trait_impl = prev_in_trait;
                    self.indent -= 1;
                    self.line("}");
                }
            }
        }

        self.self_type = prev;
    }

    // ── Traits ────────────────────────────────────────────────────────────────

    pub(crate) fn emit_trait(&mut self, t: &TraitDecl) {
        let parents = if t.parents.is_empty() {
            String::new()
        } else {
            format!(": {}", t.parents.join(" + "))
        };
        let tp = type_params_str(&t.type_params);
        self.line(&format!("pub trait {}{}{} {{", t.name, tp, parents));
        self.indent += 1;
        // Abstract associated type declarations: `type Item;` or `type Item: Trait;`
        // GAT lifetime params (`type Item<'a>`) are intentionally dropped: Boring's method
        // signatures never reference the lifetime explicitly, so emitting a plain associated
        // type keeps trait and impl declarations consistent without needing `Self::Item<'_>`.
        for assoc in &t.assoc_types {
            if let Some(constraint) = &assoc.constraint {
                let c = self.emit_type(constraint);
                // Only emit the bound if it's a trait name (a PascalCase identifier),
                // not a concrete type like `Arc<str>`, `i64`, etc.
                // Concrete type constraints from `type X as string` are dropped (emit plain `type X;`).
                let is_trait_bound = matches!(constraint,
                    Type::Named(n) if n.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
                    || matches!(constraint, Type::Generic(_, _));
                if is_trait_bound {
                    self.line(&format!("type {}: {};", assoc.name, c));
                } else {
                    self.line(&format!("type {};", assoc.name));
                }
            } else {
                self.line(&format!("type {};", assoc.name));
            }
        }
        for sig in &t.signatures {
            self.emit_fn_sig(sig, true);
        }
        for sig in &t.type_signatures {
            self.emit_fn_sig(sig, false);
        }
        // Default method implementations — use Some("") so &self is emitted
        for default in &t.defaults {
            self.emit_fn(default, Some(""));
            self.blank();
        }
        self.indent -= 1;
        self.line("}");
    }

    /// Emit a trait method signature.
    /// `with_self` = true for instance methods (adds `&self` / `&mut self`),
    /// false for associated functions (no self receiver).
    pub(crate) fn emit_fn_sig(&mut self, sig: &FnSignature, with_self: bool) {
        let params_s: Vec<String> = sig.params.iter().map(|p| self.emit_param(p)).collect();
        let all_params = if with_self {
            let self_ref = if sig.mutating { "&mut self" } else { "&self" };
            if params_s.is_empty() {
                self_ref.to_string()
            } else {
                format!("{}, {}", self_ref, params_s.join(", "))
            }
        } else {
            params_s.join(", ")
        };
        let base = sig.return_ty.as_ref().map(|t| {
            // Bare trait name in return position → `impl TraitName` (RPITIT, Rust 1.75+).
            // Dynamic dispatch is expressed explicitly with `Type::Dyn` → Box<dyn Trait>.
            if let Type::Named(n) = t {
                if self.trait_method_names.contains_key(n.as_str()) {
                    return format!("impl {}", normalize_type_name(n, self.use_rc_str()));
                }
            }
            self.emit_type(t)
        }).unwrap_or_else(|| "()".into());
        let ret = if sig.throws {
            format!("Result<{}, Box<dyn std::error::Error + Send + Sync>>", base)
        } else {
            base
        };
        let asynckw = if sig.task { "async " } else { "" };
        self.line(&format!("{}fn {}({}) -> {};", asynckw, sig.name, all_params, ret));
    }

    // ── Ext (impl Trait for Type) ─────────────────────────────────────────────

    pub(crate) fn emit_ext(&mut self, e: &ExtDecl) {
        // Build the fully-qualified type name: "Vec<T>", "HashMap<K, V>", or just "Foo"
        let full_type = if e.type_args.is_empty() {
            e.type_name.clone()
        } else {
            let args: Vec<String> = e.type_args.iter().map(|t| self.emit_type(t)).collect();
            format!("{}<{}>", e.type_name, args.join(", "))
        };

        // Build the generic params header: "<T: Clone>" or empty
        let generics = if e.type_params.is_empty() {
            String::new()
        } else {
            let params: Vec<String> = e.type_params.iter().map(|p| {
                let bounds: Vec<&str> = e.where_clause.iter()
                    .filter(|(name, _)| name == p)
                    .map(|(_, bound)| bound.as_str())
                    .collect();
                if bounds.is_empty() { p.clone() }
                else { format!("{}: {}", p, bounds.join(" + ")) }
            }).collect();
            format!("<{}>", params.join(", "))
        };

        let prev = self.self_type.replace(e.type_name.clone());
        // Set impl_type_params so methods don't re-declare type params already on the impl block.
        let prev_impl_type_params = std::mem::replace(&mut self.impl_type_params, e.type_params.clone());
        if e.traits.is_empty() {
            // Plain ext — just add methods to the type
            self.line(&format!("impl{} {} {{", generics, full_type));
        } else {
            // Multiple traits: emit one impl per trait, only including methods for that trait.
            for (i, trait_name) in e.traits.iter().enumerate() {
                if i > 0 { self.blank(); }
                self.line(&format!("impl{} {} for {} {{", generics, trait_name, full_type));
                self.indent += 1;
                // Emit associated type definitions
                for atd in &e.assoc_type_defs {
                    self.line(&format!("type {} = {};", atd.name, self.emit_type(&atd.ty)));
                }
                // Only emit methods that are members of this trait.
                // If we have trait_method_names for this trait, filter; otherwise emit all.
                let required = self.trait_method_names.get(trait_name.as_str()).cloned();
                for f in &e.methods {
                    let in_trait = required.as_ref()
                        .map(|r| r.contains(f.name.as_str()))
                        .unwrap_or(true); // No info: emit all (fallback)
                    if in_trait {
                        self.inside_trait_impl = true;
                        self.emit_fn(f, Some(&e.type_name));
                        self.inside_trait_impl = false;
                        self.blank();
                    }
                }
                self.indent -= 1;
                self.line("}");
            }
            self.impl_type_params = prev_impl_type_params;
            self.self_type = prev;
            return;
        }
        self.indent += 1;
        // Emit associated type definitions in plain impl block too
        for atd in &e.assoc_type_defs {
            self.line(&format!("type {} = {};", atd.name, self.emit_type(&atd.ty)));
        }
        for f in &e.methods {
            self.emit_fn(f, Some(&e.type_name));
            self.blank();
        }
        for setter in &e.setters {
            self.emit_setter(setter);
            self.blank();
        }
        for conv in &e.conversions {
            // Skip `as string:` — emitted as Display impl below (not as into_* method).
            if !Self::is_string_conversion(conv) {
                self.emit_conversion(conv);
                self.blank();
            }
        }
        self.indent -= 1;
        self.line("}");

        // Emit Rust operator trait impls for boring operator methods (add, sub, mul, etc.)
        self.emit_operator_trait_impls(e, &full_type, &generics);

        // Emit Display impl if the ext block adds an `as string:` conversion.
        for conv in &e.conversions {
            if Self::is_string_conversion(conv) {
                self.blank();
                self.emit_display_impl(conv, &e.type_name, &e.type_params);
            }
        }

        self.impl_type_params = prev_impl_type_params;
        self.self_type = prev;
    }

    /// Record which structs have operator methods for call-site dispatch.
    /// Also emit PartialEq/PartialOrd impls needed by the compiler.
    pub(crate) fn emit_operator_trait_impls(&mut self, e: &ExtDecl, full_type: &str, _generics: &str) {
        // Register operator methods so BinOp can emit `a.clone().add(b.clone())` instead of `(a + b)`.
        for f in &e.methods {
            let is_op = matches!(f.name.as_str(),
                "add" | "sub" | "mul" | "div" | "rem" | "neg" |
                "eq" | "ne" | "lt" | "le" | "gt" | "ge");
            if is_op {
                // Store: "StructName::method_name"
                let key = format!("{}::{}", e.type_name, f.name);
                self.struct_operator_methods.insert(key.clone());
                // Store param types for boxing at call site.
                let param_types: Vec<Type> = f.params.iter()
                    .filter_map(|p| p.ty.clone())
                    .collect();
                self.struct_operator_param_types.insert(key, param_types);
            }
        }

        // For structs that define `eq`, we need `PartialEq` for Rust.
        // The `#[derive(PartialEq)]` on the struct covers basic cases, but if the struct
        // has a custom `eq` method, emit a delegation impl.
        for f in &e.methods {
            if f.name == "eq" || f.name == "lt" {
                // We need PartialEq for PartialOrd to work.
                // Emit `impl PartialEq for TypeName` based on field-wise equality.
                // When the method param is Box<T> (strict T'), we must Box::new(rhs.clone()).
                // In managed mode, T' → Arc<Mutex<T>>, so wrap with Arc::new(Mutex::new(...)).
                let param_ty = f.params.first().and_then(|p| p.ty.as_ref());
                let param_is_box = param_ty
                    .map(|t| matches!(t, Type::Qualified(_, OwnerQual::Owned | OwnerQual::New)))
                    .unwrap_or(false);
                let param_is_managed = param_is_box && param_ty.map(|t|
                    crate::transpiler::Transpiler::is_managed_user_owned(
                        &self.config, &self.user_types, &self.unit_enums, t)
                ).unwrap_or(false);
                let rhs_arg = if param_is_managed {
                    match self.config.threading {
                        crate::transpiler::ThreadingMode::Multi =>
                            "Arc::new(std::sync::Mutex::new(rhs.clone()))".to_string(),
                        crate::transpiler::ThreadingMode::Single =>
                            "RefCell::new(rhs.clone())".to_string(),
                    }
                } else if param_is_box {
                    "Box::new(rhs.clone())".to_string()
                } else {
                    "rhs.clone()".to_string()
                };
                if f.name == "eq" {
                    self.blank();
                    self.line(&format!("impl PartialEq for {} {{", full_type));
                    self.indent += 1;
                    self.line("fn eq(&self, rhs: &Self) -> bool {");
                    self.indent += 1;
                    self.line(&format!("self.clone().eq({})", rhs_arg));
                    self.indent -= 1;
                    self.line("}");
                    self.indent -= 1;
                    self.line("}");
                }
                if f.name == "lt" {
                    let rhs_arg2 = rhs_arg.clone();
                    self.blank();
                    self.line(&format!("impl PartialEq for {} {{", full_type));
                    self.indent += 1;
                    self.line("fn eq(&self, rhs: &Self) -> bool {");
                    self.indent += 1;
                    let self_rhs = if param_is_managed {
                        match self.config.threading {
                            crate::transpiler::ThreadingMode::Multi =>
                                "Arc::new(std::sync::Mutex::new(self.clone()))".to_string(),
                            crate::transpiler::ThreadingMode::Single =>
                                "RefCell::new(self.clone())".to_string(),
                        }
                    } else if param_is_box {
                        "Box::new(self.clone())".to_string()
                    } else {
                        "self.clone()".to_string()
                    };
                    self.line(&format!("!self.clone().lt({rhs}) && !rhs.clone().lt({self_rhs})",
                        rhs = rhs_arg, self_rhs = self_rhs));
                    self.indent -= 1;
                    self.line("}");
                    self.indent -= 1;
                    self.line("}");
                    self.blank();
                    self.line(&format!("impl PartialOrd for {} {{", full_type));
                    self.indent += 1;
                    self.line("fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {");
                    self.indent += 1;
                    let rhs_box = if param_is_managed {
                        match self.config.threading {
                            crate::transpiler::ThreadingMode::Multi =>
                                "Arc::new(std::sync::Mutex::new(rhs.clone()))".to_string(),
                            crate::transpiler::ThreadingMode::Single =>
                                "RefCell::new(rhs.clone())".to_string(),
                        }
                    } else if param_is_box { "Box::new(rhs.clone())".to_string() } else { "rhs.clone()".to_string() };
                    let self_box = if param_is_managed {
                        match self.config.threading {
                            crate::transpiler::ThreadingMode::Multi =>
                                "Arc::new(std::sync::Mutex::new(self.clone()))".to_string(),
                            crate::transpiler::ThreadingMode::Single =>
                                "RefCell::new(self.clone())".to_string(),
                        }
                    } else if param_is_box { "Box::new(self.clone())".to_string() } else { "self.clone()".to_string() };
                    self.line(&format!("if self.clone().lt({rhs}) {{ Some(std::cmp::Ordering::Less) }}", rhs = rhs_box));
                    self.line(&format!("else if rhs.clone().lt({self_}) {{ Some(std::cmp::Ordering::Greater) }}", self_ = self_box));
                    self.line("else { Some(std::cmp::Ordering::Equal) }");
                    self.indent -= 1;
                    self.line("}");
                    self.indent -= 1;
                    self.line("}");
                    let _ = rhs_arg2;
                }
            }
        }
    }

    // ── Statements ────────────────────────────────────────────────────────────

}
