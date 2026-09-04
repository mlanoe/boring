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
        // Header protocols (`struct X as Trait1, Trait2:`) that are known derive macros
        // (built-in `KNOWN_DERIVABLE_TRAITS` or the project's `boring.toml` `[derives]`
        // supplement) are routed into the derive list below instead of the header-conformance
        // `impl Trait for X { ... }` loop further down — see `is_known_derivable_trait`'s doc
        // comment for why (no method body exists for a proc-macro-derived trait to attach to).
        let (proto_derive_names, proto_impl_names): (Vec<String>, Vec<String>) = s.protocols
            .iter()
            .cloned()
            .partition(|p| self.is_known_derivable_trait(p));
        // `Introspect` is neither a real `#[derive(...)]` macro nor a custom trait whose
        // methods the user wrote by hand — its `impl` body is synthesized directly from
        // `s.fields` (see `emit_introspect_struct_impl`), so it's pulled out of
        // `proto_impl_names` before that loop (further down) re-emits user-written methods
        // for every remaining header protocol.
        let (has_introspect, proto_impl_names): (Vec<String>, Vec<String>) = proto_impl_names
            .into_iter()
            .partition(|p| p == "Introspect");
        let has_introspect = !has_introspect.is_empty();
        let mut derive_names: Vec<String> = if has_derive {
            // Merge every explicit `@derive(...)` attr's args verbatim (preserves user order),
            // then qualify any bare Serialize/Deserialize — see `qualify_serde_derive_args`.
            let raw: Vec<String> = s.attrs.iter()
                .filter(|a| a.name == "derive")
                .flat_map(|a| a.args.iter().cloned())
                .collect();
            self.qualify_serde_derive_args(&raw, s.line, s.col)
        } else {
            let has_non_clone_field = s.fields.iter().any(|f| {
                // `mut AtomicUsize` (docs/book.md) wraps the field type in
                // `Type::Mut` to unlock `.fetch_add()`-style calls in Boring's own
                // content-mutation bookkeeping — peel it off first so a `mut`-qualified
                // atomic field is still recognized here, same as a bare one.
                let mut ty = &f.ty;
                while let Type::Mut(inner) = ty { ty = inner; }
                matches!(ty, Type::Named(n) if NON_CLONE_TYPES.contains(&n.as_str()))
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
            // Constructed somewhere with the `_` fill-rest marker (`Transform(x = 1.0, _)`,
            // see `helpers::collect_default_rest_targets`) → that call lowers to a trailing
            // `..Default::default()`, so the struct needs `Default` too.
            let needs_default = self.structs_needing_default.contains(&s.name);
            let mut names = vec!["Debug".to_string()];
            if has_non_clone_field {
                if needs_default { names.push("Default".to_string()); }
            } else if has_custom_cmp || has_sync_mutex_field {
                names.push("Clone".to_string());
                if needs_default { names.push("Default".to_string()); }
            } else {
                names.push("Clone".to_string());
                if needs_default { names.push("Default".to_string()); }
                names.push("PartialEq".to_string());
            }
            names
        };
        for name in &proto_derive_names {
            if !derive_names.contains(name) {
                derive_names.push(name.clone());
            }
        }
        if !derive_names.is_empty() {
            self.line(&format!("#[derive({})]", derive_names.join(", ")));
        }
        // Any other explicit attrs (not `derive`, already folded into `derive_names` above)
        // still emit verbatim, unchanged.
        for attr in &s.attrs {
            if attr.name == "derive" { continue; }
            let args_s = if attr.args.is_empty() { String::new() } else { format!("({})", attr.args.join(", ")) };
            self.line(&format!("#[{}{}]", attr.name, args_s));
        }
        let vis = if s.is_pub { "pub " } else { "" };
        let tp = type_params_str(&s.type_params);
        self.line(&format!("{}struct {}{} {{", vis, s.name, tp));
        self.indent += 1;

        // Fields from field declarations
        for f in &s.fields {
            // Per-field attributes (`@serde(rename = "...")`, etc.) — emitted verbatim
            // immediately above the field, same generic pass-through as struct-level attrs
            // above. See `FieldDecl::attrs`'s doc comment for why this exists: a struct-wide
            // `@serde(rename_all = "...")` can't cover a field whose JSON key doesn't
            // correspond to any single Boring spelling of its name.
            for attr in &f.attrs {
                let args_s = if attr.args.is_empty() { String::new() } else { format!("({})", attr.args.join(", ")) };
                self.line(&format!("#[{}{}]", attr.name, args_s));
            }
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
            } else if let Some(inner) = Self::rwlock_inner(&f.ty) {
                // Explicit `T'guard`/`T'guard'task` field — mirrors the `mutex_inner` branch
                // just above (which already correctly handles the analogous `T'actor`/
                // `T'actor'task` case). Without this, an EXPLICITLY `'guard`-qualified field
                // fell through to the `struct_rwlock_fields.contains(&rec_key)` branch below
                // — which ALSO matches an explicit qualifier, not just an inferred one (see
                // its own registration in `pre_scan_struct_item`) — and called
                // `emit_guard_type(&f.ty)` with `f.ty` STILL carrying its own `'guard`
                // qualifier, double-wrapping the field as `Arc<RwLock<Arc<RwLock<T>>>>`
                // (confirmed via a real `cargo build`, reproduces with zero Introspect
                // involvement — a plain `struct Outer: Inner'guard g` already hit this).
                self.emit_guard_type(inner)
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
        // This delegates to `{:?}` (Debug), so it must not be emitted for a struct that won't
        // actually have Debug — true whenever the struct has an explicit `@derive(...)` (the
        // `has_derive` branch above uses the user's list verbatim, unlike the auto-derive
        // `else` branch, which always includes Debug) that doesn't itself list Debug. Found via
        // docs/book.md's own `json`/`fromJson` example (`@derive(Serialize, Deserialize)`, no
        // Debug) failing to compile with "`Target` doesn't implement `Debug`" pointing straight
        // at this generated impl.
        let will_have_debug = derive_names.iter().any(|d| d == "Debug");
        if !self.display_types.contains(&s.name) && !has_derive_display && will_have_debug {
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
        // Whitelisted derive-macro names (`proto_derive_names`) were already folded into the
        // `#[derive(...)]` line above and are excluded here — only genuine
        // manually-implemented conformances reach this loop.
        for proto in &proto_impl_names {
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
                // Same split for type-level (`type def`/`type req`, no `self`) members: a
                // trait can require these via `type_signatures` (see "Type-level methods in
                // traits" in book.md), and the struct's matching `type_methods` must land in
                // this `impl Trait for Struct {}` block too -- they are ALSO always emitted
                // unconditionally into the plain `impl Struct {}` block above (that pass is
                // unfiltered by trait membership), which is fine: Rust allows an inherent
                // associated fn and a trait's fn of the same name to coexist on one type
                // (inherent resolution wins), so this is additive, not a conflict.
                let known_type: Option<std::collections::HashSet<String>> =
                    self.trait_type_method_names.get(proto.as_str()).cloned();
                for tm in &s.type_methods {
                    if known_type.as_ref().is_none_or(|names| names.contains(&tm.name)) {
                        self.emit_type_method(tm, &s.name);
                        self.blank();
                    }
                }
                self.inside_trait_impl = prev_in_trait;
                self.current_trait_assoc_names = prev_assoc_names;
                self.indent -= 1;
                self.line("}");
            }
        }

        if has_introspect {
            self.emit_introspect_struct_impl(s);
        }

        // Implicit trait conformance: if the struct has all methods required by a trait
        // (and hasn't already declared `as Trait`), auto-emit `impl Trait for Struct`.
        {
            let struct_method_names: std::collections::HashSet<String> =
                s.methods.iter().map(|m| m.name.clone()).collect();
            let struct_type_method_names: std::collections::HashSet<String> =
                s.type_methods.iter().map(|m| m.name.clone()).collect();
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
                // Type-level (`type def`/`type req`) requirements -- see `trait_type_method_names`.
                let required_type: std::collections::HashSet<String> =
                    self.trait_type_method_names.get(trait_name.as_str())
                        .cloned()
                        .unwrap_or_default();
                if required.is_empty() && required_type.is_empty() { continue; }
                // Don't auto-generate if all the required methods are already claimed by
                // some explicit protocol impl — that would create duplicate method impls.
                let already_covered = already_conforms.iter().any(|explicit_proto| {
                    let explicit_required = self.trait_method_names.get(explicit_proto.as_str())
                        .cloned()
                        .unwrap_or_default();
                    let explicit_required_type = self.trait_type_method_names.get(explicit_proto.as_str())
                        .cloned()
                        .unwrap_or_default();
                    required.iter().all(|m| explicit_required.contains(m))
                        && required_type.iter().all(|m| explicit_required_type.contains(m))
                });
                if already_covered { continue; }
                // Check that the struct has every (instance + type-level) member the trait requires.
                if required.iter().all(|m| struct_method_names.contains(m))
                    && required_type.iter().all(|m| struct_type_method_names.contains(m))
                {
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
                    for tm in &s.type_methods {
                        if required_type.contains(&tm.name) {
                            self.emit_type_method(tm, &s.name);
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

        // Module-level statics for `type var`/non-scalar `type let` fields. Named
        // `{STRUCT}_{FIELD}` (struct-name-prefixed), not just `{FIELD}` — two
        // structs with a same-named field (e.g. both called `default`) would
        // otherwise collide on the same Rust static name, a pre-existing bug fixed
        // here as part of generalizing this path for `type let`'s implicit
        // `'static` (docs/qualifiers.md's `'static` section).
        for tv in &s.type_vars {
            let is_scalar = tv.ty.as_ref().map(Self::type_is_scalar_primitive).unwrap_or(false);
            if !tv.mutable && is_scalar { continue; } // unchanged plain `const`, handled above
            let mangled = format!("{}_{}", s.name.to_uppercase(), tv.name.to_uppercase());
            let vis = if tv.is_pub { "pub " } else { "" };
            let ty_s = tv.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "/* unknown */".to_string());
            let val_s = self.emit_let_value(tv.ty.as_ref(), &tv.default);
            self.blank();
            if tv.mutable {
                self.line(&format!(
                    "{}static {}: std::sync::Mutex<{}> = std::sync::Mutex::new({});",
                    vis, mangled, ty_s, val_s
                ));
            } else {
                // `type let`, non-scalar → implicitly 'static (never annotated — see
                // `emit_type_var_const`'s doc comment). A Rust static/LazyLock cannot be
                // generic — reject rather than silently emit an invalid `impl<T>`-scoped
                // item (docs/qualifiers.md's `'static` section, "Generic structs").
                if !s.type_params.is_empty() && tv.ty.as_ref().map(|t| Self::type_mentions_type_params(t, &s.type_params)).unwrap_or(false) {
                    self.push_error(tv.line, tv.col, format!(
                        "type let '{}' cannot depend on {}'s own generic type parameter — a Rust \
                         static/LazyLock cannot be generic, so there is no single instance to share \
                         across every instantiation of {}<{}>",
                        tv.name, s.name, s.name, s.type_params.join(", ")
                    ));
                    continue;
                }
                self.line(&format!(
                    "{}static {}: std::sync::LazyLock<{}> = std::sync::LazyLock::new(|| {});",
                    vis, mangled, ty_s, val_s
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

    /// Whether `ty` is a primitive scalar — mirrors the `is_scalar` heuristic
    /// `top_level_let_is_const_safe` (`mod.rs`) uses for top-level `let`s. A
    /// `type let` field of one of these types keeps the existing plain-`const`
    /// emission unconditionally; anything else (struct/array/dict/etc.) is where
    /// `type let`'s `const` emission was unconditionally wrong for a non-const-safe
    /// initializer — see `emit_type_var_const`'s doc comment.
    pub(crate) fn type_is_scalar_primitive(ty: &Type) -> bool {
        match ty {
            Type::Int | Type::Uint | Type::Uint8 | Type::Float32 | Type::Float64 | Type::Bool => true,
            Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128 => true,
            Type::Named(n) => matches!(n.as_str(),
                "int" | "uint" | "uint8" | "float" | "float32" | "float64" | "bool"
                | "int8" | "int16" | "int32" | "int64" | "int128"
                | "uint16" | "uint32" | "uint64" | "uint128"
                | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
                | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"),
            Type::Optional(inner) | Type::Qualified(inner, _) => Self::type_is_scalar_primitive(inner),
            _ => false,
        }
    }

    /// Whether `ty` mentions any of `params` (a struct's own generic type
    /// parameters), directly or nested inside a generic argument/collection/
    /// qualifier — used to reject a `type let` field whose type depends on the
    /// struct's own `T`, since a Rust `static`/`LazyLock` cannot be generic (see
    /// docs/qualifiers.md's `'static` section, "Generic structs").
    fn type_mentions_type_params(ty: &Type, params: &[String]) -> bool {
        match ty {
            Type::Named(n) => params.iter().any(|p| p == n),
            Type::TypeParam(n) => params.iter().any(|p| p == n),
            Type::Optional(inner) | Type::Array(inner) | Type::ArrayN(inner, _)
                | Type::ArrayNExpr(inner, _) | Type::Set(inner) | Type::Qualified(inner, _)
                | Type::Dyn(inner) | Type::Impl(inner) =>
                Self::type_mentions_type_params(inner, params),
            Type::Dict(k, v) => Self::type_mentions_type_params(k, params) || Self::type_mentions_type_params(v, params),
            Type::Tuple(elems) => elems.iter().any(|t| Self::type_mentions_type_params(t, params)),
            Type::Generic(_, args) => args.iter().any(|t| Self::type_mentions_type_params(t, params)),
            _ => false,
        }
    }

    /// `type let`/`type var` — the class-scoped member forms
    /// (`docs/book.md`'s "Type-level members"). `type let` is always implicitly
    /// `'static` (docs/qualifiers.md's `'static` section) — there is no
    /// annotation to write or check, only a representation choice:
    ///
    /// - Primitive scalar type -> plain `const` inside `impl` (unchanged, already
    ///   correct — a scalar constant has no reference semantics to speak of).
    /// - Everything else -> emitted as a marker comment here; the real module-level
    ///   `static NAME: LazyLock<T>` is emitted by the loop just below
    ///   (`emit_struct`'s "Module-level statics" section), the same place `type
    ///   var`'s `Mutex`-wrapped statics already live, because Rust forbids a
    ///   `static` item inside `impl`. This replaces the previous unconditional
    ///   `const` emission for `type let`, which silently produced invalid Rust
    ///   the moment the constructor had a real `init` body (`Config::new(...)`
    ///   is never `const fn`) — a pre-existing bug documented and now fixed as
    ///   part of the `'static` qualifier work.
    pub(crate) fn emit_type_var_const(&mut self, tv: &TypeVar) {
        let vis = if tv.is_pub { "pub " } else { "" };
        let ty_s = tv.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "/* unknown */".to_string());
        let val_s = self.emit_let_value(tv.ty.as_ref(), &tv.default);
        let is_scalar = tv.ty.as_ref().map(Self::type_is_scalar_primitive).unwrap_or(false);
        if tv.mutable {
            // type var is emitted as a module-level static Mutex — see below impl block.
            // We emit a marker comment here so the impl block is clean.
            self.line(&format!("// type var {}: {} = {} — see static below", tv.name, ty_s, val_s));
        } else if is_scalar {
            // type let (scalar) → associated const, unchanged.
            self.line(&format!("{}const {}: {} = {};", vis, tv.name.to_uppercase(), ty_s, val_s));
        } else {
            // type let (non-scalar, implicitly 'static) → module-level LazyLock static,
            // see below impl block.
            self.line(&format!("// type let {}: {} = {} — see static below", tv.name, ty_s, val_s));
        }
    }

    pub(crate) fn emit_type_method(&mut self, tm: &TypeMethod, type_name: &str) {
        use crate::ast::TypeMethodKind;
        // Rust forbids explicit visibility qualifiers on trait-impl items (E0449: "trait items
        // always share the visibility of their trait") -- suppress `pub` when this type method
        // is being emitted into an `impl Trait for Struct {}` block (see the `proto_impl_names`
        // loop in `emit_struct`, which sets `inside_trait_impl` around this call). The same
        // method is still emitted `pub` in the plain `impl Struct {}` block it also always
        // lands in.
        let vis = if tm.is_pub && !self.inside_trait_impl { "pub " } else { "" };
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
        // A method is "void" if it has no declared return type, or returns () / Nil / Void --
        // mirrors emit_fn's `declared_void` (emit_top.rs) so a throwing type method's Ok(())
        // wrapping (in emit_stmt's tail-expression handling) matches a regular throwing
        // function's.
        let declared_void = match &tm.return_ty {
            None => true,
            Some(Type::Void) | Some(Type::Nil) => true,
            Some(Type::Named(n)) if n == "void" || n == "nil" => true,
            _ => false,
        };
        let base_ret = if declared_void {
            "()".to_string()
        } else {
            tm.return_ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| "()".into())
        };
        // `throws`/`throws Type:` on a type-level method previously had NO effect on codegen
        // at all: no `Result<_, _>` wrapping here, so `throw` inside the body fell through to
        // `emit_flow::emit_throw`'s `not_throws` branch and emitted a bare `panic!(...)`
        // instead of `return Err(...)`. As with a regular throwing `FnDecl` (see
        // `compute_fn_return_type` in emit_top.rs), the typed vs untyped distinction doesn't
        // change this wrapping -- both use `Box<dyn Error>` here; what a typed `throws Type:`
        // actually buys is `Type` being registered in `typed_error_enums` (done in
        // `pre_scan`/`transpiler/mod.rs`), which drives the enum's Display/Error impls and
        // `BoringError::Other` routing at the throw site.
        let ret_ty = if tm.throws {
            format!("Result<{}, Box<dyn std::error::Error + Send + Sync>>", base_ret)
        } else {
            base_ret
        };
        let ret_s = if ret_ty == "()" { String::new() } else { format!(" -> {}", ret_ty) };
        let _ = type_name;
        self.line(&format!("{}fn {}({}){} {{", vis, fn_name, params_s, ret_s));
        self.indent += 1;
        let prev_setter = std::mem::replace(&mut self.in_type_setter, matches!(tm.kind, TypeMethodKind::Set));
        // Same save/restore-around-emit_body dance as emit_fn (emit_top.rs) -- `in_throws`
        // gates `emit_throw`'s Err(...)-vs-panic! choice and the tail-expression Ok(...)
        // wrapping in emit_stmt.rs; `fn_return_ty` is consulted by `emit_return` for
        // explicit `return expr` statements.
        let prev_throws = self.in_throws;
        let prev_returns_void = self.fn_returns_void;
        let prev_declared_void = self.fn_declared_void;
        let prev_fn_return_ty = self.fn_return_ty.clone();
        self.in_throws = tm.throws;
        self.fn_returns_void = !tm.throws && declared_void;
        self.fn_declared_void = declared_void;
        self.fn_return_ty = tm.return_ty.clone();
        // A type-level method's params were never fed into the per-function-body
        // local-variable bookkeeping `seed_param_locals` populates (emit_top.rs) --
        // unlike `emit_fn`, which does this for every regular function/instance method.
        // Confirmed real gap, not just a throws-adjacent nicety: a plain (non-throwing)
        // `type def`/`type req` taking a `string` param already mis-transpiled `s.length`
        // as `s::length` (treating the unregistered param as a type/module path) --
        // exercised by tests/cases/type_def_typed_throws.br once it's wired into
        // tests/transpile.rs. Save/take/restore the same sets `seed_param_locals` writes
        // to, so entries seeded here don't leak into the next item's body (same rationale
        // as emit_fn's own per-function save/restore of these sets).
        let prev_known_local_vars   = std::mem::take(&mut self.known_local_vars);
        let prev_var_types          = std::mem::take(&mut self.var_types);
        let prev_dict_vars          = std::mem::take(&mut self.dict_vars);
        let prev_string_vars        = std::mem::take(&mut self.string_vars);
        let prev_var_mutex_types    = std::mem::take(&mut self.var_mutex_types);
        let prev_var_mutex_task_types = std::mem::take(&mut self.var_mutex_task_types);
        let prev_var_rwlock_types   = std::mem::take(&mut self.var_rwlock_types);
        let prev_var_rwlock_task_types = std::mem::take(&mut self.var_rwlock_task_types);
        let prev_arc_vars           = std::mem::take(&mut self.arc_vars);
        let prev_rc_vars            = std::mem::take(&mut self.rc_vars);
        let prev_optional_vars      = std::mem::take(&mut self.optional_vars);
        let prev_managed_mutex_vars = std::mem::take(&mut self.managed_mutex_vars);
        let prev_managed_refcell_vars = std::mem::take(&mut self.managed_refcell_vars);
        let prev_var_struct_types   = std::mem::take(&mut self.var_struct_types);
        let prev_task_vars          = std::mem::take(&mut self.task_vars);
        let prev_throws_fn_params   = std::mem::take(&mut self.throws_fn_params);
        let prev_var_newtype_type   = std::mem::take(&mut self.var_newtype_type);
        self.seed_param_locals(&tm.params);
        self.emit_body(&tm.body);
        self.var_newtype_type   = prev_var_newtype_type;
        self.throws_fn_params   = prev_throws_fn_params;
        self.task_vars          = prev_task_vars;
        self.var_struct_types   = prev_var_struct_types;
        self.managed_refcell_vars = prev_managed_refcell_vars;
        self.managed_mutex_vars = prev_managed_mutex_vars;
        self.optional_vars      = prev_optional_vars;
        self.rc_vars            = prev_rc_vars;
        self.arc_vars           = prev_arc_vars;
        self.var_rwlock_task_types = prev_var_rwlock_task_types;
        self.var_rwlock_types   = prev_var_rwlock_types;
        self.var_mutex_task_types = prev_var_mutex_task_types;
        self.var_mutex_types    = prev_var_mutex_types;
        self.string_vars        = prev_string_vars;
        self.dict_vars          = prev_dict_vars;
        self.var_types          = prev_var_types;
        self.known_local_vars   = prev_known_local_vars;
        self.fn_return_ty = prev_fn_return_ty;
        self.fn_declared_void = prev_declared_void;
        self.fn_returns_void = prev_returns_void;
        self.in_throws = prev_throws;
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

    /// Rewrites a `@derive(...)` argument list, qualifying bare `Serialize`/`Deserialize` to
    /// `serde::Serialize`/`serde::Deserialize`. Rust's derive-macro position accepts a path
    /// there (stable since edition 2018), so this sidesteps ever needing to emit a `use
    /// serde::{Serialize, Deserialize};` import — Boring emits no such import today, so the
    /// bare spelling docs/book.md's `json`/`fromJson` section actually documents
    /// (`@derive(Serialize, Deserialize)`) failed to compile: "cannot find derive macro
    /// `Deserialize` in this scope" / an unsatisfied `serde::Deserialize` trait bound.
    /// Mirrors the existing `thiserror::Error` auto-qualification in `emit_enum` below.
    ///
    /// Also flips `uses_serde` (Cargo.toml gets the `serde` dependency) on any qualifying
    /// name, not just when this file also happens to call `json()`/`fromJson()` — e.g. a
    /// struct declared in one file and (de)serialized from another.
    ///
    /// Also rejects `Introspect` here — the natural first mistake for anyone used to
    /// `@derive(Debug, Clone, ...)` is to reach for `@derive(Introspect)` too, but
    /// `Introspect`'s `impl` body is transpiler-synthesized (`emit_introspect_struct_impl`/
    /// `emit_introspect_enum_impl`), not a real proc-macro derive — left unqualified, it
    /// would silently emit `#[derive(Introspect)]`, which fails downstream with a confusing
    /// "cannot find derive macro `Introspect`" from rustc instead of a clear Boring-level
    /// error pointing at the fix (`as Introspect` in the header).
    fn qualify_serde_derive_args(&self, args: &[String], line: usize, col: usize) -> Vec<String> {
        args.iter().filter_map(|a| match a.as_str() {
            "Serialize" => { self.uses_serde.set(true); Some("serde::Serialize".to_string()) }
            "Deserialize" => { self.uses_serde.set(true); Some("serde::Deserialize".to_string()) }
            "serde::Serialize" | "serde::Deserialize" => { self.uses_serde.set(true); Some(a.clone()) }
            "Introspect" => {
                self.push_error(line, col,
                    "'Introspect' cannot be added via '@derive(...)' — it is not a real \
                     derive macro (its impl is synthesized by the compiler, not a proc macro). \
                     Declare it in the header instead: 'struct X as Introspect:' / \
                     'enum X as Introspect:'");
                None
            }
            _ => Some(a.clone()),
        }).collect()
    }

    /// Resolves a Boring field/param type to the Rust primitive type name it maps to, or
    /// `None` when it isn't one of the plain scalars/string `FieldValue` can
    /// represent (a nested struct/enum, a collection, `Option`, an actor/guard wrapper, a
    /// generic type-param, ...). Shared by every introspect codegen helper below — both the
    /// "produce a `FieldValue`" direction (`introspect_field_value_expr`, always total —
    /// falls back to `Other`/Debug for a `None` here) and the "consume a `FieldValue`"
    /// direction (`introspect_extract_expr`, genuinely partial — a `None` here means the
    /// field/param can't be round-tripped, so the caller must leave it out of the reflected
    /// `Field`/`Method` entirely rather than emit a setter/unpacker that would always fail).
    ///
    /// Primitive Boring type keywords (`int`, `float`, `bool`, `string`, ...) parse as
    /// `Type::Named("int")` etc, NOT the dedicated `Type::Int`/`Type::Float64`/... variants
    /// (those are reserved for the capitalized spellings, `Int`/`Float`/...) — see
    /// `parse_type_base`. Resolved through `normalize_type_name` (the same helper
    /// `emit_type`/`emit_field_type` use) to get the actual Rust primitive name regardless of
    /// which spelling/form was used, rather than duplicating that whole alias table here.
    fn introspect_resolve_ty(&self, ty: &Type) -> Option<String> {
        let mut ty = ty;
        while let Type::Mut(inner) = ty { ty = inner; }
        match ty {
            Type::Named(n) => {
                let r = normalize_type_name(n, self.use_rc_str());
                match r.as_str() {
                    "isize" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128"
                    | "u8" | "u16" | "u32" | "u64" | "u128"
                    | "f32" | "f64" | "bool" | "Rc<str>" | "Arc<str>" => Some(r),
                    _ => None,
                }
            }
            Type::Int => Some("isize".to_string()),
            Type::Uint => Some("usize".to_string()),
            Type::Int8 => Some("i8".to_string()),
            Type::Int16 => Some("i16".to_string()),
            Type::Int32 => Some("i32".to_string()),
            Type::Int64 => Some("i64".to_string()),
            Type::Int128 => Some("i128".to_string()),
            Type::Uint8 => Some("u8".to_string()),
            Type::Uint16 => Some("u16".to_string()),
            Type::Uint32 => Some("u32".to_string()),
            Type::Uint64 => Some("u64".to_string()),
            Type::Uint128 => Some("u128".to_string()),
            Type::Float32 => Some("f32".to_string()),
            Type::Float64 => Some("f64".to_string()),
            Type::Bool => Some("bool".to_string()),
            Type::Str => Some(if self.use_rc_str() { "Rc<str>" } else { "Arc<str>" }.to_string()),
            _ => None,
        }
    }

    /// Resolves a Boring field type to its `FieldValue` constructor expression, given the Rust
    /// expression string that reads the field's value. Shared by `emit_introspect_struct_impl`
    /// (`access` = `"s.x"`, a direct place through a downcast reference — no deref needed even
    /// for a scalar) and `emit_introspect_enum_impl` (`access` = a match-arm-bound variable
    /// name — e.g. `"f0"` — which, under Rust's default match ergonomics, binds as `&T` when
    /// matching through `&self`, so `is_ref_binding` must be `true` there to insert the `*` a
    /// scalar/bool cast needs; `Str`'s `.clone()` and `Other`'s `{:?}` work unchanged either
    /// way). Always total — a type `introspect_resolve_ty` doesn't recognize falls back to
    /// `FieldValue::Other`, Debug-formatted (needs no per-kind unwrapping logic: `{:?}` already
    /// works uniformly as long as the wrapped value itself is `Debug`, which every Boring
    /// struct/enum already derives by default).
    fn introspect_field_value_expr(&self, ty: &Type, access: &str, is_ref_binding: bool, allow_nested: bool) -> String {
        let resolved = self.introspect_resolve_ty(ty);
        let deref = if is_ref_binding { "*" } else { "" };
        match resolved.as_deref() {
            Some("isize") | Some("usize")
            | Some("i8") | Some("i16") | Some("i32") | Some("i64") | Some("i128")
            | Some("u8") | Some("u16") | Some("u32") | Some("u64") | Some("u128")
                => format!("FieldValue::Int({}{} as isize)", deref, access),
            Some("f32") | Some("f64")
                => format!("FieldValue::Float({}{} as f64)", deref, access),
            Some("bool")
                => format!("FieldValue::Bool({}{})", deref, access),
            Some("Rc<str>") | Some("Arc<str>")
                => format!("FieldValue::Str({}.clone())", access),
            _ => {
                // `allow_nested` restricts `Nested`/`Actor`/`Guard` detection to the two
                // call sites that reach a real struct FIELD (instance or type-level `type
                // let`) — see `introspect_nested_field_value_expr`'s doc comment for why
                // this is a separate flag from `is_ref_binding` (both struct fields and
                // method-return-value conversions call this fn with `is_ref_binding:
                // false`, but only fields are in scope for this iteration).
                if allow_nested {
                    if let Some(expr) = self.introspect_nested_field_value_expr(ty, access, is_ref_binding) {
                        return expr;
                    }
                }
                format!("FieldValue::Other(format!(\"{{:?}}\", {}))", access)
            }
        }
    }

    /// Detects whether a struct/type-level FIELD's own declared type qualifies for
    /// `FieldValue::Nested`/`Actor`/`Guard` instead of the `Other`/Debug fallback — see
    /// the `boring_introspect_trait_design` memo's "`Nested`/`Actor`/`Guard` FieldValue
    /// variants" section for the full design. `None` means "keep falling back to
    /// `Other`", exactly today's behavior — this check is purely additive.
    ///
    /// Only reachable via `introspect_field_value_expr`'s `allow_nested` flag, which is
    /// `true` only at the two call sites that resolve a real FIELD (`s.field` for an
    /// instance field, `Type::CONST` for a `type let`) — never for a method's return
    /// value, and never for an enum variant field (out of scope this round, see the
    /// design memo's scope note).
    ///
    /// A field qualifies only when its own type is itself a struct/enum that declares
    /// `as Introspect` in its header (checked via `struct_protocols`, which — see
    /// `pre_scan`/`pre_scan_struct_item` — is populated for both structs AND enums,
    /// despite the field's name predating enum support). For the `'inline`/`'owned`
    /// qualifiers (→ `FieldValue::Nested`, `T`/`Box<T>` in Rust) the target type must
    /// ALSO be `Clone` (`Box::new(access.clone())`/`access.clone()` needs `T: Clone` —
    /// checked via `struct_derives_clone`, or `all_enum_types` since every enum always
    /// derives `Clone`, see `emit_enum`). `'shared`/`'actor`/`'actor'task`/`'guard`/
    /// `'guard'task` need NO such check — `access.clone()` there only ever clones the
    /// OUTER handle (`Rc`/`Arc`, a cheap refcount bump — `impl<T: ?Sized> Clone for
    /// Rc<T>`/`Arc<T>` has no `T: Clone` bound at all), never `T` itself; the blanket
    /// `Rc<T>`/`Arc<T>` (strong) and `RefCell<T>`/`Mutex<T>`/`RwLock<T>`/tokio
    /// equivalents (weak) delegate impls in `emit_introspect_prelude` make the
    /// resulting `Box<Rc<RefCell<T>>>`-shaped value coerce to `Box<dyn Introspect>`
    /// with no `T: Clone` bound anywhere in the chain.
    fn introspect_nested_field_value_expr(&self, ty: &Type, access: &str, is_ref_binding: bool) -> Option<String> {
        let mut ty = ty;
        while let Type::Mut(inner) = ty { ty = inner; }
        let deref = if is_ref_binding { "*" } else { "" };

        // (FieldValue variant tag, the qualifier's inner type, whether the constructor
        // needs an explicit `Box::new(...)` wrapper — `'owned` doesn't, its `access` is
        // already a `Box<T>` that coerces directly — and whether `T: Clone` is required).
        let (variant, inner_ty, needs_box_new, needs_clone): (&str, &Type, bool, bool) = match ty {
            // Bare `Point` (no qualifier written) — struct fields are always inline in
            // the parent's own allocation by default (see `emit_field_type`'s doc), same
            // Rust shape as an explicit `'inline`.
            Type::Named(_) => ("Nested", ty, true, true),
            Type::Qualified(inner, OwnerQual::Inline) => ("Nested", inner.as_ref(), true, true),
            Type::Qualified(inner, OwnerQual::Owned) => ("Nested", inner.as_ref(), false, true),
            Type::Qualified(inner, OwnerQual::Shared) => ("Nested", inner.as_ref(), true, false),
            Type::Qualified(inner, OwnerQual::Actor | OwnerQual::ActorTask) => ("Actor", inner.as_ref(), true, false),
            Type::Qualified(inner, OwnerQual::Guard | OwnerQual::GuardTask) => ("Guard", inner.as_ref(), true, false),
            _ => return None,
        };
        let type_name = match inner_ty {
            Type::Named(n) => n.as_str(),
            _ => return None,
        };
        let implements_introspect = self.struct_protocols.get(type_name)
            .map(|protocols| protocols.iter().any(|p| p == "Introspect"))
            .unwrap_or(false);
        if !implements_introspect { return None; }
        if needs_clone {
            let is_clone = self.all_enum_types.contains(type_name)
                || self.struct_derives_clone.contains(type_name);
            if !is_clone { return None; }
        }
        let cloned = format!("{}{}.clone()", deref, access);
        let payload = if needs_box_new { format!("Box::new({})", cloned) } else { cloned };
        Some(format!("FieldValue::{}({})", variant, payload))
    }

    /// The reverse of `introspect_field_value_expr`: given a Rust expression evaluating to
    /// `&FieldValue` and the *target* Boring type, produces a Rust match-expression that
    /// extracts a concrete value of that type — or `None` when the type isn't one of the
    /// typed `FieldValue` variants (genuinely partial, unlike the two helpers above: a
    /// `FieldValue::Other(String)` only ever holds an already-flattened Debug string, so there
    /// is no way to reconstruct an arbitrary field/param type from it). Callers use `None` to
    /// mean "leave this field's setter / this method out of `introspect()`'s reflected list
    /// entirely" rather than emit code that would always fail at runtime.
    fn introspect_extract_expr(&self, ty: &Type, src: &str, label: &str) -> Option<String> {
        let resolved = self.introspect_resolve_ty(ty)?;
        Some(match resolved.as_str() {
            "isize" | "usize" | "i8" | "i16" | "i32" | "i64" | "i128"
            | "u8" | "u16" | "u32" | "u64" | "u128" => format!(
                "match {src} {{ FieldValue::Int(v) => *v as {resolved}, _ => return Err(\"wrong argument type for '{label}': expected an integer\".into()) }}"
            ),
            "f32" | "f64" => format!(
                "match {src} {{ FieldValue::Float(v) => *v as {resolved}, _ => return Err(\"wrong argument type for '{label}': expected a float\".into()) }}"
            ),
            "bool" => format!(
                "match {src} {{ FieldValue::Bool(v) => *v, _ => return Err(\"wrong argument type for '{label}': expected a bool\".into()) }}"
            ),
            "Rc<str>" | "Arc<str>" => format!(
                "match {src} {{ FieldValue::Str(v) => v.clone(), _ => return Err(\"wrong argument type for '{label}': expected a string\".into()) }}"
            ),
            _ => return None,
        })
    }

    /// True when `m` should appear in `introspect()`'s `methods` list AT ALL — per the
    /// "Method visibility expansion" (see the `boring_introspect_trait_design` memo),
    /// `native` (body is runtime-implemented, nothing Boring-level to call) is the ONLY
    /// true exclusion; `m.qualifier.is_some()` (a `def Type.method()` ext-style method,
    /// structurally not a plain method of the type whose `s.methods`/`e.methods` list is
    /// being walked) is excluded too, defensively, though it should never actually appear
    /// there. Every other shape — `task`/`stream`/`throws`/variadic/defaulted/non-typed-
    /// param methods — is now visible; whether a real call body exists for it is a
    /// SEPARATE question, answered by `introspect_method_is_callable` below.
    fn introspect_method_is_visible(&self, m: &FnDecl) -> bool {
        if m.is_native { return false; }
        if m.qualifier.is_some() { return false; }
        true
    }

    /// True when a real, working call body can be generated for `m`'s `request`/`modify`
    /// entry — this is the EXACT predicate that gates both `Method.isCallable` and whether
    /// `MethodAccess`'s dispatch slot is `Some`/`None` (built from this same check at the
    /// same call site in `build_introspect_methods`, so the two can never disagree, same
    /// discipline as the `isRebindable`/setter-availability fix). `task`/`stream` are
    /// excluded — invoking either is a genuinely different call shape (a `Future`/an
    /// iterator of `FieldValue`s, not one synchronous value) that `request`/`modify` don't
    /// attempt this iteration. A non-primitive PARAMETER type is excluded too — see
    /// `introspect_extract_expr`'s doc comment for why (no way to round-trip an arbitrary
    /// type back out of a `[FieldValue]` args array) — likewise variadic/defaulted params
    /// (no argument-count-with-defaults resolution attempted). `throws` is NOT excluded —
    /// genuinely callable now (the generated body propagates the underlying `Result` via
    /// `?`, since a `throws` method's real Rust return type, `Result<T, Box<dyn
    /// std::error::Error + Send + Sync>>`, is exactly `IntrospectError`'s alias target, no
    /// conversion needed). The return type is never a reason to exclude a method —
    /// `introspect_field_value_expr` is total (falls back to `Other`/Debug).
    fn introspect_method_is_callable(&self, m: &FnDecl) -> bool {
        if !self.introspect_method_is_visible(m) { return false; }
        if m.task || m.stream { return false; }
        for p in &m.params {
            if p.variadic || p.default.is_some() { return false; }
            match &p.ty {
                Some(ty) => if self.introspect_resolve_ty(ty).is_none() { return false; },
                None => return false,
            }
        }
        true
    }

    /// The `MethodKind` for `m` — `Task`/`Stream` when it's declared that way, `Plain`
    /// otherwise (including `throws` methods, which are orthogonal to this tag).
    pub(crate) fn introspect_method_kind_expr(m: &FnDecl) -> &'static str {
        if m.task { "MethodKind::Task" }
        else if m.stream { "MethodKind::Stream" }
        else { "MethodKind::Plain" }
    }

    /// Builds the free dispatch functions (one per reflectable method) for `methods`, and
    /// returns `(entry_literals, fn_item_texts)` — the `Method { ... }` entry-literal strings
    /// (in the same order) to place inside `introspect()`'s `methods: vec![...]`, and the raw
    /// Rust free-fn item text for each, to be emitted by the caller *after* the enclosing
    /// `impl Introspect for ... { fn introspect() { ... } }` block has been closed (this
    /// function does not emit anything itself — it only builds strings — precisely so a caller
    /// assembling `introspect()`'s body doesn't interleave top-level `fn` items into the middle
    /// of that body's own text stream). Shared verbatim by the struct and enum paths — an
    /// enum's methods are scoped to the WHOLE type regardless of variant (see
    /// `emit_introspect_enum_impl`'s doc comment), so the codegen is identical to a struct's:
    /// downcast `&dyn Introspect`/`&mut dyn Introspect` back to the concrete (possibly generic)
    /// type, unpack `args: &[FieldValue]` positionally, call the real method, re-wrap the
    /// return value.
    fn build_introspect_methods(
        &self, type_name: &str, methods: &[FnDecl], tp_impl: &str, tp_use: &str,
    ) -> (Vec<String>, Vec<String>) {
        let str_ty = if self.use_rc_str() { "Rc" } else { "Arc" };
        let mut entries = Vec::new();
        let mut fns: Vec<String> = Vec::new();
        for m in methods {
            if !self.introspect_method_is_visible(m) { continue; }
            let is_callable = self.introspect_method_is_callable(m);
            let tpg = if tp_impl.is_empty() { String::new() } else { format!("::{}", tp_use) };
            let fn_ref = if is_callable {
                let fn_name = format!("__introspectCall_{}_{}", type_name, m.name);
                let mut body: Vec<String> = Vec::new();
                let instance_ty = if m.mutating { "&mut dyn Introspect" } else { "&dyn Introspect" };
                let (as_any, downcast) = if m.mutating {
                    ("__introspectAsAnyMut", "downcast_mut")
                } else {
                    ("__introspectAsAny", "downcast_ref")
                };
                body.push(format!(
                    "let s = instance.{as_any}().{downcast}::<{type_name}{tp_use}>().ok_or(\"wrong instance type for method '{}'\")?;",
                    m.name
                ));
                body.push(format!(
                    "if args.len() != {} {{ return Err(\"wrong argument count for method '{}'\".into()); }}",
                    m.params.len(), m.name
                ));
                let mut arg_names = Vec::new();
                for (i, p) in m.params.iter().enumerate() {
                    let ty = p.ty.as_ref().expect("checked callable");
                    let extracted = self.introspect_extract_expr(ty, &format!("&args[{}]", i), &p.name)
                        .expect("checked callable");
                    let an = format!("a{}", i);
                    body.push(format!("let {} = {};", an, extracted));
                    arg_names.push(an);
                }
                // `throws` methods are genuinely callable now: their real Rust return
                // type is `Result<T, Box<dyn std::error::Error + Send + Sync>>` — exactly
                // `IntrospectError`'s alias target — so `?` propagates the underlying
                // error with no conversion needed.
                let call_expr = format!("s.{}({})", m.name, arg_names.join(", "));
                let call_expr = if m.throws { format!("{}?", call_expr) } else { call_expr };
                match &m.return_ty {
                    Some(rty) if !matches!(rty, Type::Void) => {
                        body.push(format!("let r = {};", call_expr));
                        body.push(format!("Ok({})", self.introspect_field_value_expr(rty, "r", false, false)));
                    }
                    _ => {
                        body.push(format!("{};", call_expr));
                        body.push("Ok(FieldValue::Other(\"()\".to_string()))".to_string());
                    }
                }
                fns.push(format!(
                    "#[allow(non_snake_case)]\nfn {fn_name}{tp_impl}({instance_arg}: {instance_ty}, args: &[FieldValue]) -> Result<FieldValue, IntrospectError> {{\n    {body}\n}}",
                    fn_name = fn_name, tp_impl = tp_impl, instance_arg = "instance", instance_ty = instance_ty,
                    body = body.join("\n    "),
                ));
                format!("Some({}{})", fn_name, tpg)
            } else {
                "None".to_string()
            };
            let access = if m.mutating {
                format!("MethodAccess::InstanceDef({})", fn_ref)
            } else {
                format!("MethodAccess::InstanceReq({})", fn_ref)
            };
            entries.push(format!(
                "Method {{ name: {str_ty}::from(\"{}\"), isPublic: {}, isInstance: true, isMutable: {}, isCallable: {}, isThrowing: {}, kind: {}, access: {} }}",
                m.name, m.is_pub, m.mutating, is_callable, m.throws, Self::introspect_method_kind_expr(m), access,
            ));
        }
        (entries, fns)
    }

    /// Builds type-level (`type let`/`type def`/`type req`) `Field`/`Method` reflection
    /// entries — the no-`instance` overloads (see `emit_introspect_prelude`'s
    /// `FieldAccess`/`MethodAccess::Type` doc). STRUCT-ONLY (there is no
    /// `EnumDecl::type_vars` at all, and the transpiler does not yet emit `type_methods`
    /// for enums regardless — see `EnumDecl::type_methods`'s own doc comment; an
    /// out-of-scope pre-existing gap, not something this feature works around) and
    /// NON-GENERIC-ONLY (a `type let`/`type def` inside `impl<T> Struct<T>` has no single
    /// T-independent Rust item to reference from a plain, non-generic `fn() -> ...`
    /// pointer — the call site gates this by only calling here when `s.type_params` is
    /// empty). Also `type let` only, never `type var` — a mutable type-level var lowers to
    /// a module-level `static ... Mutex<...>` (see `emit_type_var_const`), and wiring
    /// `getStatic`/`setStatic` through a lock is deferred, not attempted here (documented
    /// gap; there is no `getMutStatic` — `getMut()` was dropped entirely, see the design
    /// memo). `type set` type-methods are skipped too (a property-setter-
    /// shaped type method, distinct from a plain callable — same "not attempted yet"
    /// reason, see `introspect_type_method_is_visible`).
    fn build_introspect_type_members(
        &self, type_name: &str, type_vars: &[TypeVar], type_methods: &[TypeMethod],
    ) -> (Vec<String>, Vec<String>, Vec<String>) {
        let str_ty = if self.use_rc_str() { "Rc" } else { "Arc" };
        let mut field_entries = Vec::new();
        let mut method_entries = Vec::new();
        let mut fns: Vec<String> = Vec::new();

        for tv in type_vars {
            if tv.mutable { continue; } // `type var` — deferred, see doc comment above
            let Some(ty) = &tv.ty else { continue };
            let const_ref = format!("{}::{}", type_name, tv.name.to_uppercase());
            let fn_name = format!("__introspectGetStatic_{}_{}", type_name, tv.name);
            fns.push(format!(
                "#[allow(non_snake_case)]\nfn {fn_name}() -> Result<FieldValue, IntrospectError> {{\n    Ok({})\n}}",
                self.introspect_field_value_expr(ty, &const_ref, false, true),
            ));
            field_entries.push(format!(
                "Field {{ name: {str_ty}::from(\"{}\"), isPublic: {}, isInstance: false, isMutable: false, isRebindable: false, access: FieldAccess::Type {{ getter: {fn_name}, setter: None }} }}",
                tv.name, tv.is_pub,
            ));
        }

        for tm in type_methods {
            if !self.introspect_type_method_is_visible(tm) { continue; }
            let is_callable = self.introspect_type_method_is_callable(tm);
            let fn_ref = if is_callable {
                let fn_name = format!("__introspectCallStatic_{}_{}", type_name, tm.name);
                let mut body: Vec<String> = Vec::new();
                body.push(format!(
                    "if args.len() != {} {{ return Err(\"wrong argument count for method '{}'\".into()); }}",
                    tm.params.len(), tm.name,
                ));
                let mut arg_names = Vec::new();
                for (i, p) in tm.params.iter().enumerate() {
                    let ty = p.ty.as_ref().expect("checked callable");
                    let extracted = self.introspect_extract_expr(ty, &format!("&args[{}]", i), &p.name)
                        .expect("checked callable");
                    let an = format!("a{}", i);
                    body.push(format!("let {} = {};", an, extracted));
                    arg_names.push(an);
                }
                // Same `throws` handling as the instance-scoped path above — the real
                // Rust return type already matches `IntrospectError`, so `?` alone works.
                let call_expr = format!("{}::{}({})", type_name, tm.name, arg_names.join(", "));
                let call_expr = if tm.throws { format!("{}?", call_expr) } else { call_expr };
                match &tm.return_ty {
                    Some(rty) if !matches!(rty, Type::Void) => {
                        body.push(format!("let r = {};", call_expr));
                        body.push(format!("Ok({})", self.introspect_field_value_expr(rty, "r", false, false)));
                    }
                    _ => {
                        body.push(format!("{};", call_expr));
                        body.push("Ok(FieldValue::Other(\"()\".to_string()))".to_string());
                    }
                }
                fns.push(format!(
                    "#[allow(non_snake_case)]\nfn {fn_name}(args: &[FieldValue]) -> Result<FieldValue, IntrospectError> {{\n    {}\n}}",
                    body.join("\n    "),
                ));
                format!("Some({})", fn_name)
            } else {
                "None".to_string()
            };
            let kind = if tm.task { "MethodKind::Task" } else { "MethodKind::Plain" };
            method_entries.push(format!(
                "Method {{ name: {str_ty}::from(\"{}\"), isPublic: {}, isInstance: false, isMutable: {}, isCallable: {}, isThrowing: {}, kind: {}, access: MethodAccess::Type({}) }}",
                tm.name, tm.is_pub, matches!(tm.kind, crate::ast::TypeMethodKind::Def), is_callable, tm.throws, kind, fn_ref,
            ));
        }

        (field_entries, method_entries, fns)
    }

    /// True when `m` should appear in `introspect()`'s type-level `methods` at all —
    /// `TypeMethodKind::Set` is excluded outright (a property-setter-shaped type method,
    /// structurally closer to `Field.set()` than to a plain callable — not attempted here,
    /// see `build_introspect_type_members`'s doc comment); every other `type def`/`type
    /// req` (including `task`/`throws`) is visible. `TypeMethod` has no `native` concept
    /// at all (no runtime-implemented type-level members), so unlike the instance-method
    /// version there is no other exclusion.
    fn introspect_type_method_is_visible(&self, m: &TypeMethod) -> bool {
        !matches!(m.kind, crate::ast::TypeMethodKind::Set)
    }

    /// True when a real call body can be generated for `m`'s type-level `call()` entry —
    /// same discipline as `introspect_method_is_callable`: this exact check gates both
    /// `Method.isCallable` and whether `MethodAccess::Type`'s slot is `Some`/`None`.
    /// `task` is excluded (different call shape, same reasoning as the instance-method
    /// case); `throws` is NOT excluded — genuinely callable, same `?`-propagation as the
    /// instance-scoped path (see `build_introspect_type_members`).
    fn introspect_type_method_is_callable(&self, m: &TypeMethod) -> bool {
        if !self.introspect_type_method_is_visible(m) { return false; }
        if m.task { return false; }
        for p in &m.params {
            if p.variadic || p.default.is_some() { return false; }
            match &p.ty {
                Some(ty) => if self.introspect_resolve_ty(ty).is_none() { return false; },
                None => return false,
            }
        }
        true
    }

    /// Emits the free-fn item texts `build_introspect_methods` built, each preceded by a
    /// blank line — call only after the enclosing `impl Introspect for ...` block is closed.
    fn emit_introspect_method_fns(&mut self, fns: &[String]) {
        for f in fns {
            self.blank();
            self.out.push_str(f);
            self.out.push('\n');
        }
    }

    /// Synthesizes `impl Introspect for {StructName} { ... }` from the struct's field/method
    /// list — see `emit_introspect_prelude`'s doc comment for why this can't be a
    /// `#[derive(...)]`, and the module-level `boring_introspect_trait_design` memo for the
    /// overall v2 handle-based design this implements.
    ///
    /// Covers both ways a struct's fields reach the generated Rust struct: the `struct X:
    /// <fields>` block form (`s.fields`, used directly), and the no-body-`init` form (`struct
    /// X: init(int a, int b):` with an empty body — see the "Fields from init (no-body form)"
    /// pass in `emit_struct` above, which emits each such init's params as the struct's own
    /// Rust fields when `s.fields` itself is empty). `introspect_fields` below normalizes both
    /// into one list so the rest of this function doesn't care which form produced them.
    ///
    /// Each field gets one generated free getter function (always emitted — `get()`/
    /// `introspect_field_value_expr` is total), plus a setter only when the field is
    /// reassignable (`f.mutable`) AND its type round-trips through `FieldValue`
    /// (`introspect_extract_expr`). `isMutable` (from `ty.grants_mut()`) is reported on
    /// `Field` as informational metadata only — there is no `getMut()`/write-in-place form
    /// (dropped post-ship: a scalar field can never be `mut`-qualified in Boring at all, so
    /// the only fields it could ever apply to are non-scalar and would need an opaque,
    /// Boring-source-unconsumable handle regardless — see the design memo). Methods are
    /// handled by the shared `emit_introspect_methods` helper.
    pub(crate) fn emit_introspect_struct_impl(&mut self, s: &StructDecl) {
        // Bounded impl type params, same pattern as the auto-Display impl above: a generic
        // field with a bare type-param type (`T`) hits the `Other` fallback, which needs
        // `T: Debug` (`Clone` alone isn't enough) — plus `'static`, required here (unlike
        // Display) because `Introspect: std::any::Any` needs `Self: 'static`, which for a
        // generic impl only holds when every type param is itself bounded `'static`.
        let tp_impl = if s.type_params.is_empty() {
            String::new()
        } else {
            let bounded: Vec<String> = s.type_params.iter()
                .map(|p| if p.starts_with('\'') { p.clone() }
                         else if p.starts_with('$') { emit_generic_param(p) }
                         else { format!("{}: Clone + std::fmt::Debug + 'static", p) })
                .collect();
            format!("<{}>", bounded.join(", "))
        };
        let tp_use = type_params_use_str(&s.type_params);

        let introspect_fields: Vec<(String, Type)> = if !s.fields.is_empty() {
            s.fields.iter().map(|f| (f.name.clone(), f.ty.clone())).collect()
        } else {
            let mut fields = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for init in &s.inits {
                if init.body.is_empty() {
                    for p in &init.params {
                        if seen.insert(p.name.clone()) {
                            let ty = p.ty.clone().unwrap_or_else(|| Type::Named("_unknown".to_string()));
                            fields.push((p.name.clone(), ty));
                        }
                    }
                }
            }
            fields
        };
        // Field-level pub/mutable/rebindable metadata only exists on the `s.fields` block
        // form (`FieldDecl`) — a no-body-init-derived "field" has no such bookkeeping (see
        // `pre_scan_struct_item`'s identical limitation), so those default to
        // `isPublic: true, isMutable: false, isRebindable: false` (permissive read, no
        // write-through), matching how the plain Rust field emission above already treats
        // init-only fields as always `pub`.
        let field_meta = |name: &str| -> (bool, bool, bool) {
            s.fields.iter().find(|f| f.name == name)
                .map(|f| (f.is_pub, f.ty.grants_mut(), f.mutable))
                .unwrap_or((true, false, false))
        };

        let str_ty = if self.use_rc_str() { "Rc" } else { "Arc" };
        // Turbofish suffix for referencing a generic free fn as a value from inside
        // `impl<T: ...> Introspect for StructName<T>` — e.g. `__introspectGet_Foo_x::<T>`.
        let tpg = if tp_impl.is_empty() { String::new() } else { format!("::{}", tp_use) };

        self.blank();
        self.line("#[allow(non_snake_case)]");
        self.line(&format!("impl{} Introspect for {}{} {{", tp_impl, s.name, tp_use));
        self.indent += 1;
        self.line("fn __introspectAsAny(&self) -> &dyn std::any::Any { self }");
        self.line("fn __introspectAsAnyMut(&mut self) -> &mut dyn std::any::Any { self }");
        // Backed by a per-type cache, not rebuilt on every call — `IntrospectInfo`'s own
        // structural data (names, fn pointers, booleans) never depends on any particular
        // instance, so computing the `Vec<Field>`/`Vec<Method>` once and handing back a
        // `&'static` reference from then on avoids reallocating both vecs (plus cloning every
        // Field/Method name string) on every `.introspect()` call. Mirrors the `GPU(n)`
        // adapter-enumeration precedent (`src/transpiler/wgpu/host.rs`'s `emit_gpu_adapter_enumeration`)
        // — a static cache, not a Boring-language feature, since Boring already treats
        // every struct-typed value as reference-like (see the parameter-passing rules), this
        // is transparent to Boring source: `let info = w.introspect()` behaves identically
        // whether the Rust-level return is owned or `&'static`.
        //
        // Multi (`Arc<str>` fields, `Sync`): a plain `static OnceLock<IntrospectInfo>` works
        // directly — `static` items are checked for `Sync` unconditionally, and `Arc<str>` is.
        // Single (`Rc<str>` fields, `!Sync`): `Rc<...>`-bearing `IntrospectInfo` can never
        // live in a `static OnceLock` (E0277 — `static` needs `Sync` regardless of whether the
        // Boring program ever actually crosses a thread). Use a `thread_local!` `OnceCell`
        // instead — per-thread, so no `Sync` bound at all — holding a `Box::leak`ed (i.e.
        // deliberately, permanently leaked, matching the `static` case's "never freed for the
        // process's lifetime" behavior) `&'static IntrospectInfo` computed once per thread.
        self.line("fn introspect(&self) -> &'static IntrospectInfo {");
        self.indent += 1;
        if self.use_rc_str() {
            self.line("thread_local! { static INFO: std::cell::OnceCell<&'static IntrospectInfo> = std::cell::OnceCell::new(); }");
            self.line("INFO.with(|cell| *cell.get_or_init(|| Box::leak(Box::new(IntrospectInfo {");
        } else {
            self.line("static INFO: std::sync::OnceLock<IntrospectInfo> = std::sync::OnceLock::new();");
            self.line("INFO.get_or_init(|| IntrospectInfo {");
        }
        self.indent += 1;
        self.line(&format!("typeName: {str_ty}::from(\"{}\"),", s.name));
        self.line(&format!("variantName: {str_ty}::from(\"{}\"),", s.name));
        self.line("variantIndex: 0,");
        let mut field_entries: Vec<String> = Vec::new();
        for (fname, fty) in &introspect_fields {
            let (is_pub, grants_mut, rebindable) = field_meta(fname);
            // `isRebindable` must promise exactly what `set()` can deliver, not just echo the
            // language-level `var`/`mut` qualifier: a `var`-qualified field whose type doesn't
            // round-trip through `FieldValue` (anything falling back to `Other`/Debug — see
            // `introspect_extract_expr`'s doc comment) gets no setter fn generated at all, so
            // reporting `isRebindable: true` for it would be a lie `set()` immediately
            // contradicts at runtime ("has no setter"). Gate the *reported flag* on the same
            // condition that gates *emitting the setter fn*, so the two can never disagree.
            let has_setter = rebindable && self.introspect_extract_expr(fty, "&value", fname).is_some();
            let setter = if has_setter {
                format!("Some(__introspectSet_{}_{}{})", s.name, fname, tpg)
            } else { "None".to_string() };
            field_entries.push(format!(
                "Field {{ name: {str_ty}::from(\"{fname}\"), isPublic: {is_pub}, isInstance: true, isMutable: {grants_mut}, isRebindable: {has_setter}, access: FieldAccess::Instance {{ getter: __introspectGet_{sname}_{fname}{tpg}, setter: {setter} }} }}",
                sname = s.name,
            ));
        }
        let (method_entries, method_fns) = self.build_introspect_methods(&s.name, &s.methods, &tp_impl, &tp_use);
        let mut method_entries = method_entries;
        // Type-level (`type let`/`type def`/`type req`) members — see
        // `build_introspect_type_members`'s doc comment for the scope this covers
        // (struct only, non-generic only, `type let` only — no `type var`/`type set`).
        let (type_field_entries, type_method_entries, type_member_fns) = if s.type_params.is_empty() {
            self.build_introspect_type_members(&s.name, &s.type_vars, &s.type_methods)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        field_entries.extend(type_field_entries);
        method_entries.extend(type_method_entries);
        if field_entries.is_empty() {
            self.line("fields: vec![],");
        } else {
            self.line("fields: vec![");
            self.indent += 1;
            for e in &field_entries { self.line(&format!("{},", e)); }
            self.indent -= 1;
            self.line("],");
        }
        if method_entries.is_empty() {
            self.line("methods: vec![],");
        } else {
            self.line("methods: vec![");
            self.indent += 1;
            for e in &method_entries { self.line(&format!("{},", e)); }
            self.indent -= 1;
            self.line("],");
        }
        self.indent -= 1;
        if self.use_rc_str() {
            self.line("}))))");
        } else {
            self.line("})");
        }
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();

        // Free getter/setter functions — one per field. `get()` is always emitted (total);
        // `set` only when the field is rebindable.
        for (fname, fty) in &introspect_fields {
            let (_, _, rebindable) = field_meta(fname);
            self.blank();
            self.line("#[allow(non_snake_case)]");
            self.line(&format!(
                "fn __introspectGet_{}_{}{}(instance: &dyn Introspect) -> Result<FieldValue, IntrospectError> {{",
                s.name, fname, tp_impl,
            ));
            self.indent += 1;
            self.line(&format!(
                "let s = instance.__introspectAsAny().downcast_ref::<{}{}>().ok_or(\"wrong instance type for field '{}'\")?;",
                s.name, tp_use, fname,
            ));
            self.line(&format!("Ok({})", self.introspect_field_value_expr(fty, &format!("s.{}", fname), false, true)));
            self.indent -= 1;
            self.line("}");

            if rebindable {
                if let Some(extracted) = self.introspect_extract_expr(fty, "&value", fname) {
                    self.blank();
                    self.line("#[allow(non_snake_case)]");
                    self.line(&format!(
                        "fn __introspectSet_{}_{}{}(instance: &mut dyn Introspect, value: FieldValue) -> Result<(), IntrospectError> {{",
                        s.name, fname, tp_impl,
                    ));
                    self.indent += 1;
                    self.line(&format!(
                        "let s = instance.__introspectAsAnyMut().downcast_mut::<{}{}>().ok_or(\"wrong instance type for field '{}'\")?;",
                        s.name, tp_use, fname,
                    ));
                    self.line(&format!("s.{} = {};", fname, extracted));
                    self.line("Ok(())");
                    self.indent -= 1;
                    self.line("}");
                }
            }
        }
        self.emit_introspect_method_fns(&method_fns);
        self.emit_introspect_method_fns(&type_member_fns);
        self.blank();
    }

    /// Synthesizes `impl Introspect for {EnumName}` from `e.variants`/`e.methods`. Mirrors
    /// `emit_introspect_struct_impl`'s reasoning, with one structural difference per the v2
    /// design: `introspect()`'s `fields` are scoped to the CURRENT runtime variant (only its
    /// named fields — same rule v1 used), matched once inside `introspect()` itself, while
    /// `methods` is the same list regardless of variant (an enum's methods don't vary by
    /// variant, only its fields do) — handled by the shared `emit_introspect_methods` helper,
    /// identical to the struct path.
    ///
    /// Only NAMED variant fields (`VariantField.name.is_some()`) are exposed — a purely
    /// positional field (no name in the Boring source) has no string to key a `Field` by, so
    /// it's bound (to stay pattern-exhaustive) but otherwise ignored, same as v1. Rust always
    /// emits enum variants tuple-style regardless of whether the Boring source named its
    /// fields, so a named field's position — not its name — is what the generated Rust
    /// pattern actually binds; the name is Boring-side-only bookkeeping (`VariantField.name`)
    /// recovered here at transpile time.
    ///
    /// Per-field getter free functions are generated once per (variant, field) pair — keyed
    /// `__introspectGet_{Enum}_{Variant}_{field}` — since, unlike a struct, the downcast+match
    /// needed to reach the field is itself variant-specific. No setter (enum variant fields
    /// have no `var`/`mut` mechanism, see below) and no getMut (dropped entirely, see the
    /// design memo — never provided a real capability, see `emit_introspect_struct_impl`'s
    /// doc comment for why).
    pub(crate) fn emit_introspect_enum_impl(&mut self, e: &EnumDecl) {
        let tp_impl = if e.type_params.is_empty() {
            String::new()
        } else {
            let bounded: Vec<String> = e.type_params.iter()
                .map(|p| if p.starts_with('\'') { p.clone() }
                         else if p.starts_with('$') { emit_generic_param(p) }
                         else { format!("{}: Clone + std::fmt::Debug + 'static", p) })
                .collect();
            format!("<{}>", bounded.join(", "))
        };
        let tp_use = type_params_use_str(&e.type_params);
        let str_ty = if self.use_rc_str() { "Rc" } else { "Arc" };

        let bindings_for = |v: &EnumVariant| -> Vec<String> {
            v.fields.iter().enumerate()
                .map(|(idx, f)| if f.name.is_some() { format!("f{}", idx) } else { "_".to_string() })
                .collect()
        };
        let pat_for = |v: &EnumVariant| -> String {
            if v.fields.is_empty() { format!("{}::{}", e.name, v.name) }
            else { format!("{}::{}({})", e.name, v.name, bindings_for(v).join(", ")) }
        };

        self.blank();
        self.line("#[allow(non_snake_case)]");
        self.line(&format!("impl{} Introspect for {}{} {{", tp_impl, e.name, tp_use));
        self.indent += 1;
        self.line("fn __introspectAsAny(&self) -> &dyn std::any::Any { self }");
        self.line("fn __introspectAsAnyMut(&mut self) -> &mut dyn std::any::Any { self }");
        // Backed by a per-type `OnceLock<Vec<IntrospectInfo>>` (one entry per variant), not
        // rebuilt on every call — see `emit_introspect_struct_impl`'s doc comment for the full
        // rationale. Every variant's structural data (names, fn pointers, booleans) is
        // computed once regardless of which instance/variant is asking; only the final "which
        // entry is THIS instance" step actually needs to look at `self`.
        self.line("fn introspect(&self) -> &'static IntrospectInfo {");
        self.indent += 1;
        // See `emit_introspect_struct_impl`'s doc comment for the Single/Multi split: a
        // `Vec<IntrospectInfo>` holding `Rc<str>` fields (Single) is `!Sync` and can't live in
        // a `static OnceLock`, so it goes through a `thread_local!` `OnceCell` holding a
        // `Box::leak`ed `&'static Vec<IntrospectInfo>` instead — same "compute once, return
        // `&'static`" contract, no `Sync` bound required.
        if self.use_rc_str() {
            self.line("thread_local! { static VARIANTS: std::cell::OnceCell<&'static Vec<IntrospectInfo>> = std::cell::OnceCell::new(); }");
            self.line("let variants = VARIANTS.with(|cell| *cell.get_or_init(|| {");
        } else {
            self.line("static VARIANTS: std::sync::OnceLock<Vec<IntrospectInfo>> = std::sync::OnceLock::new();");
            self.line("let variants = VARIANTS.get_or_init(|| {");
        }
        self.indent += 1;
        let (method_entries, method_fns) = self.build_introspect_methods(&e.name, &e.methods, &tp_impl, &tp_use);
        if method_entries.is_empty() {
            self.line("let __methods: Vec<Method> = vec![];");
        } else {
            self.line("let __methods: Vec<Method> = vec![");
            self.indent += 1;
            for m in &method_entries { self.line(&format!("{},", m)); }
            self.indent -= 1;
            self.line("];");
        }
        if self.use_rc_str() {
            self.line("Box::leak(Box::new(vec![");
        } else {
            self.line("vec![");
        }
        self.indent += 1;
        for (i, v) in e.variants.iter().enumerate() {
            let named_fields: Vec<(usize, &str, &Type)> = v.fields.iter().enumerate()
                .filter_map(|(idx, f)| f.name.as_deref().map(|n| (idx, n, &f.ty)))
                .collect();
            let fields_expr = if named_fields.is_empty() {
                "vec![]".to_string()
            } else {
                // Enum variant fields have no `var`/`mut` mechanism at all in Boring —
                // `VariantField` carries only a name/type, no mutability flag (unlike a
                // struct's `FieldDecl`) — so `isRebindable` is always `false` and no setter
                // is ever generated: there is no Boring-source syntax (`shape.radius = x`)
                // that would grant this permission in the first place. `isMutable` still
                // follows the field's own type (`grants_mut()`) as informational metadata,
                // same as a struct field, but gates nothing (no `getMut()` exists).
                let entries: Vec<String> = named_fields.iter().map(|(idx, fname, fty)| {
                    let _ = idx;
                    format!(
                        "Field {{ name: {str_ty}::from(\"{fname}\"), isPublic: true, isInstance: true, isMutable: {gm}, isRebindable: false, access: FieldAccess::Instance {{ getter: __introspectGet_{ename}_{vname}_{fname}{tpg}, setter: None }} }}",
                        ename = e.name, vname = v.name, gm = fty.grants_mut(),
                        tpg = if tp_impl.is_empty() { String::new() } else { format!("::{}", tp_use) },
                    )
                }).collect();
                format!("vec![{}]", entries.join(", "))
            };
            self.line(&format!(
                "IntrospectInfo {{ typeName: {str_ty}::from(\"{ename}\"), variantName: {str_ty}::from(\"{vname}\"), variantIndex: {i}, fields: {fields_expr}, methods: __methods.clone() }},",
                ename = e.name, vname = v.name,
            ));
        }
        self.indent -= 1;
        if self.use_rc_str() {
            self.line("]))");
        } else {
            self.line("]");
        }
        self.indent -= 1;
        if self.use_rc_str() {
            self.line("}));");
        } else {
            self.line("});");
        }
        self.line("let __idx: usize = match self {");
        self.indent += 1;
        for (i, v) in e.variants.iter().enumerate() {
            self.line(&format!("{} => {},", pat_for(v), i));
        }
        self.indent -= 1;
        self.line("};");
        self.line("&variants[__idx]");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.emit_introspect_method_fns(&method_fns);
        self.blank();

        // Per-(variant, field) getter free functions (no setter — see the doc comment above).
        for v in &e.variants {
            let named_fields: Vec<(usize, &str, &Type)> = v.fields.iter().enumerate()
                .filter_map(|(idx, f)| f.name.as_deref().map(|n| (idx, n, &f.ty)))
                .collect();
            for (idx, fname, fty) in named_fields {
                let bindings = bindings_for(v);
                let bind_var = &bindings[idx];
                self.blank();
                self.line("#[allow(non_snake_case)]");
                self.line(&format!(
                    "fn __introspectGet_{}_{}_{}{}(instance: &dyn Introspect) -> Result<FieldValue, IntrospectError> {{",
                    e.name, v.name, fname, tp_impl,
                ));
                self.indent += 1;
                self.line(&format!(
                    "let s = instance.__introspectAsAny().downcast_ref::<{}{}>().ok_or(\"wrong instance type for field '{}'\")?;",
                    e.name, tp_use, fname,
                ));
                self.line("match s {");
                self.indent += 1;
                self.line(&format!("{} => Ok({}),", pat_for(v), self.introspect_field_value_expr(fty, bind_var, true, false)));
                self.line("_ => Err(\"field is not present on the current runtime variant\".into()),");
                self.indent -= 1;
                self.line("}");
                self.indent -= 1;
                self.line("}");

                // No setter free-fn for enum fields — see the doc comment above where
                // `isRebindable`/`setter` are hardcoded `false`/`None`: there is no
                // Boring-source mechanism that grants field reassignment on enum
                // variant data at all, so a setter here would never be referenced.
            }
        }
        self.blank();
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
        // A non-recursive `'owned`/`'new` field (OwnerQual::is_owned_or_new) is included too
        // when in managed mode: it resolves to the same Actor wrapper there (see emit_type's
        // Union(NEW_MEMBERS) arm / `emit_managed_actor`), not `Box<T>` — a recursive field is
        // excluded since `emit_enum` always boxes those unconditionally (`Box<T>` does
        // implement PartialEq when T does).
        let has_actor_field = !is_unit_enum && e.variants.iter().any(|v| {
            v.fields.iter().enumerate().any(|(fi, f)| match &f.ty {
                Type::Qualified(_, OwnerQual::Actor | OwnerQual::Guard | OwnerQual::Shared) => true,
                Type::Qualified(_, q) if q.is_owned_or_new() => {
                    self.config.mode == crate::transpiler::TranspileMode::Managed
                        && !self.recursive_fields.contains(&format!("{}::{}::{}", e.name, v.name, fi))
                }
                _ => false,
            })
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
        // Header protocols (`enum X as Trait1, Trait2:`) that are known derive macros
        // (built-in `KNOWN_DERIVABLE_TRAITS` or the project's `boring.toml` `[derives]`
        // supplement) are routed into an extra `#[derive(...)]` line here instead of the
        // header-conformance `impl Trait for X { ... }` loop further down — same rationale
        // as the struct path (see `is_known_derivable_trait`'s doc comment). Rust allows
        // multiple stacked `#[derive(...)]` attributes, so this is a pure addition on top of
        // the (intentionally untouched) derive logic above — it never rewrites those lines,
        // it only tracks which names they already cover so nothing gets derived twice.
        let (proto_derive_names, proto_impl_names): (Vec<String>, Vec<String>) = e.protocols
            .iter()
            .cloned()
            .partition(|p| self.is_known_derivable_trait(p));
        // Same "Introspect" carve-out as the struct path — see the comment at that
        // partition site (`emit_struct`) for why.
        let (has_introspect, proto_impl_names): (Vec<String>, Vec<String>) = proto_impl_names
            .into_iter()
            .partition(|p| p == "Introspect");
        let has_introspect = !has_introspect.is_empty();
        let mut emitted_derives: std::collections::HashSet<String> = std::collections::HashSet::new();
        if !has_clone_derive {
            let thiserror_will_add_debug = has_variant_error_attr && !has_debug_derive;
            if is_unit_enum {
                if !thiserror_will_add_debug { emitted_derives.insert("Debug".to_string()); }
                emitted_derives.extend(["Clone", "Copy", "PartialEq", "Eq", "Hash"].map(String::from));
            } else if has_actor_field {
                if !thiserror_will_add_debug { emitted_derives.insert("Debug".to_string()); }
                emitted_derives.insert("Clone".to_string());
            } else {
                if !thiserror_will_add_debug { emitted_derives.insert("Debug".to_string()); }
                emitted_derives.extend(["Clone", "PartialEq"].map(String::from));
            }
        }
        if has_variant_error_attr && !already_has_thiserror {
            emitted_derives.insert("thiserror::Error".to_string());
            if !has_debug_derive { emitted_derives.insert("Debug".to_string()); }
        }
        for a in e.attrs.iter().filter(|a| a.name == "derive") {
            emitted_derives.extend(a.args.iter().cloned());
        }
        let extra_derives: Vec<String> = proto_derive_names.into_iter()
            .filter(|p| !emitted_derives.contains(p))
            .collect();
        if !extra_derives.is_empty() {
            self.line(&format!("#[derive({})]", extra_derives.join(", ")));
        }
        for attr in &e.attrs {
            // `derive` args go through the same bare-Serialize/Deserialize qualification as
            // the struct path — enums have no equivalent `derive_names` merge step of their
            // own, so this attr is otherwise emitted completely verbatim.
            let args_s = if attr.name == "derive" {
                let qualified = self.qualify_serde_derive_args(&attr.args, attr.line, attr.col);
                if qualified.is_empty() { String::new() } else { format!("({})", qualified.join(", ")) }
            } else if attr.args.is_empty() {
                String::new()
            } else {
                format!("({})", attr.args.join(", "))
            };
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

        // Auto-emit Display for enums (delegates to Debug), mirroring the identical
        // auto-Display struct behavior above, so that:
        //  - BoringFmt<T: Display> works for Vec<Enum>
        //  - String interpolation / `print` can use `{}` on a plain enum value directly,
        //    not just on its pattern-matched fields or an explicit `as string:` conversion
        //    — this is what `print Foo.make()`/`print z` in
        //    tests/cases/enum_type_def{,_throws}.br exercise: those are plain,
        //    non-error enums with no Display impl otherwise.
        // This also covers typed error enums (`throws CalcError`), which previously had
        // their own copy of this exact same Display impl gated on `is_error_type` only —
        // folded into this general case; `impl Error` below is still emitted only for those.
        // Skipped when: thiserror will generate Display itself (has_variant_error_attr), the
        // enum (or a same-named `ext` block) already declares its own `as string:` conversion
        // (`display_types`, emitted as a real Display impl further down / in the ext path), or
        // the enum explicitly opts out via `@derive(Display)`.
        // Also skipped when the enum won't actually have a `Debug` impl for `{:?}` to use —
        // the same `will_have_debug` gate the struct auto-Display above already has, which
        // the enum path was missing: an explicit `@derive(Clone, Serialize, Deserialize)`
        // (used verbatim, so no Debug is auto-added) generated an `impl Display` that failed
        // to compile with "`X` doesn't implement `Debug`". `emitted_derives` is exactly the
        // set of derive names the lines above emitted (auto Debug/Clone/..., thiserror's
        // injected Debug, and the enum's own `@derive(...)` args); `extra_derives` is the
        // header-protocol addition on top.
        let has_derive_display = e.attrs.iter().any(|a| a.name == "derive" && a.args.iter().any(|arg| arg == "Display"));
        let will_have_debug = emitted_derives.iter().any(|d| d == "Debug")
            || extra_derives.iter().any(|d| d == "Debug");
        if !has_variant_error_attr && !self.display_types.contains(&e.name) && !has_derive_display && will_have_debug {
            let name = &e.name;
            // Bounded impl type params (`+ Debug`) plus bare use-site params, same as the
            // struct auto-Display above — needed for generic enums like `Result<T, E>`.
            let tp_impl_disp = if e.type_params.is_empty() {
                String::new()
            } else {
                let bounded: Vec<String> = e.type_params.iter()
                    .map(|p| if p.starts_with('\'') { p.clone() }
                             else if p.starts_with('$') { emit_generic_param(p) }
                             else { format!("{}: Clone + std::fmt::Debug", p) })
                    .collect();
                format!("<{}>", bounded.join(", "))
            };
            let tp_use_disp = type_params_use_str(&e.type_params);
            self.line(&format!("impl{} std::fmt::Display for {}{} {{", tp_impl_disp, name, tp_use_disp));
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
        if is_error_type && !has_variant_error_attr {
            self.line(&format!("impl std::error::Error for {} {{}}", e.name));
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
        if !plain_methods.is_empty() || !e.conversions.is_empty() || !e.setters.is_empty()
            || has_named_fields || !e.type_methods.is_empty() {
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
            // Type methods (`type def`/`type req`/`type set`) — same production and same
            // emission logic already used for a struct's type_methods (see emit_struct
            // above); `emit_type_method` takes a plain type-name string, so it was already
            // generic over struct vs. enum with no changes needed here beyond calling it.
            for tm in &e.type_methods {
                self.emit_type_method(tm, &e.name);
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

        // Whitelisted derive-macro names (`proto_derive_names`) were already folded into an
        // extra `#[derive(...)]` line above and are excluded here — only genuine
        // manually-implemented conformances reach this loop.
        for proto in &proto_impl_names {
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

        if has_introspect {
            self.emit_introspect_enum_impl(e);
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
            let mut emitted_count = 0;
            for trait_name in &e.traits {
                // A derive-macro name (`Component`, `Debug`, ...) has no valid translation
                // here: Rust can't retroactively attach `#[derive(...)]` to a struct/enum
                // defined by a separate item, which is exactly what an `ext` block is. Report
                // it instead of falling into the "unknown trait → emit every method" fallback
                // below, which would silently produce nonsense (or invalid) Rust for it.
                if self.is_known_derivable_trait(trait_name) {
                    self.push_error(e.line, e.col, format!(
                        "'{}' is a derive macro and cannot be added via 'ext {} as {}:' — \
                         declare 'as {}' (or '@derive({})') on {}'s own struct/enum definition instead",
                        trait_name, e.type_name, trait_name, trait_name, trait_name, e.type_name
                    ));
                    continue;
                }
                if emitted_count > 0 { self.blank(); }
                emitted_count += 1;
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
                // When the method param is Box<T> (strict T'owned), we must Box::new(rhs.clone()).
                // In managed mode, T'owned → Arc<Mutex<T>>, so wrap with Arc::new(Mutex::new(...)).
                let param_ty = f.params.first().and_then(|p| p.ty.as_ref());
                let param_is_box = param_ty
                    .map(|t| matches!(t, Type::Qualified(_, q) if q.is_owned_or_new()))
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
