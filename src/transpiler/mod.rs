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

// Boring → Rust transpiler.
//
// Entry point: `transpile(program)` returns a complete Rust source string.
// The generated code uses Rust 2021 edition idioms.

use crate::ast::*;

mod emit_top;
mod emit_struct;
mod emit_stmt;
mod emit_expr;
mod emit_methods;
pub(crate) mod helpers;
pub(crate) use helpers::*;

// ─── Public entry point ───────────────────────────────────────────────────────

pub struct TranspileOutput {
    pub code: String,
    pub has_streams: bool,
    /// True when the program uses `error`, `warn`, `info`, `debug`, or `trace`.
    /// The caller should add `log = "0.4"` to the project's Cargo.toml.
    pub uses_log: bool,
    /// True when an enum with @error("...") variants was emitted (thiserror auto-derive).
    /// The caller should add `thiserror = "1"` to the project's Cargo.toml.
    pub uses_thiserror: bool,
    /// True when the program imports from `reqwest`.
    /// The caller should add `reqwest` to the project's Cargo.toml.
    pub uses_reqwest: bool,
    /// True when the program uses `Task.cancelled()` or `f.cancel()`.
    /// The caller should add `tokio-util = { version = "0.7", features = ["sync"] }` to Cargo.toml.
    pub uses_tokio_util: bool,
    /// True when `json()` or `fromJson()` is used.
    /// The caller should add `serde` and `serde_json` to Cargo.toml.
    pub uses_serde: bool,
}

pub fn transpile(program: &Program) -> String {
    transpile_full(program).code
}

pub fn transpile_full(program: &Program) -> TranspileOutput {
    let mut t = Transpiler::new();
    t.emit_program(program);
    TranspileOutput { code: t.out, has_streams: t.has_streams, uses_log: t.uses_log.get(), uses_thiserror: t.uses_thiserror.get(), uses_reqwest: t.uses_reqwest, uses_tokio_util: t.uses_tokio_util.get(), uses_serde: t.uses_serde.get() }
}

// ─── Transpiler state ─────────────────────────────────────────────────────────

struct Transpiler {
    pub(crate) out: String,
    pub(crate) indent: usize,
    /// Are we inside a `throws` function body? (return values need Ok() wrapping)
    pub(crate) in_throws: bool,
    /// Are we inside a `task` (async) function body?
    pub(crate) in_async: bool,
    /// Name of the type currently being impl'd (for self-aware emit).
    pub(crate) self_type: Option<String>,
    /// Variables known to hold a collection (Vec/HashMap/HashSet) — use {:?} when formatting.
    pub(crate) collection_vars: std::collections::HashSet<String>,
    /// Variables known to hold a Vec specifically (not HashMap/HashSet, and not reduce scalars).
    /// Used to apply BoringFmt wrapping for Display-without-quotes printing.
    pub(crate) vec_vars: std::collections::HashSet<String>,
    /// Variables known to hold a HashSet (for `remove(&v)` and `add`→`insert` dispatch).
    pub(crate) set_vars: std::collections::HashSet<String>,
    /// Variables known to hold a HashMap/dict — subscript reads use `.get()`, writes use `.insert()`.
    pub(crate) dict_vars: std::collections::HashSet<String>,
    /// For-loop variables iterating over `.chars()` — need `.to_string()` when used as dict keys.
    pub(crate) chars_vars: std::collections::HashSet<String>,
    /// Top-level function return types: fn_name → return Type (for {:?} formatting in print).
    pub(crate) fn_return_types: std::collections::HashMap<String, Type>,
    /// Variables that hold an opaque collection index (Option<usize> from firstIndex/nextIndex).
    /// Subscripts on these use `.get_at(idx)` instead of `[(idx) as usize]`.
    pub(crate) index_vars: std::collections::HashSet<String>,
    /// Known function param types: fn_name → param types (for optional-arg coercion).
    pub(crate) fn_sigs: std::collections::HashMap<String, Vec<Type>>,
    /// Enum variant name → enum type name (for qualified patterns in match arms).
    pub(crate) enum_variants: std::collections::HashMap<String, String>,
    /// "EnumName::VariantName" → field names (always tuple-style in emit).
    pub(crate) enum_variant_fields: std::collections::HashMap<String, Vec<Option<String>>>,
    /// "EnumName::VariantName" → field types (for coercing string args to Arc<str>).
    pub(crate) enum_variant_field_types: std::collections::HashMap<String, Vec<Type>>,
    /// Variables declared with an Optional type (e.g. `let int? a = ...`) — never re-wrap in Some().
    pub(crate) optional_vars: std::collections::HashSet<String>,
    /// Pre-emitted default values per function param: fn_name → [Option<default_str>].
    pub(crate) fn_defaults: std::collections::HashMap<String, Vec<Option<String>>>,
    /// Struct name → [(field_name, field_type)] for constructor coercion.
    pub(crate) struct_fields: std::collections::HashMap<String, Vec<(String, Type)>>,
    /// Struct name → assoc type name → concrete type (for `T.AssocName` resolution).
    pub(crate) struct_assoc_types: std::collections::HashMap<String, std::collections::HashMap<String, Type>>,
    /// Top-level functions that declare `throws`.
    pub(crate) fn_throws: std::collections::HashSet<String>,
    /// Enum names used as typed error types (`throws CalcError`).
    /// The transpiler emits `impl std::error::Error` + `Display` for these automatically.
    pub(crate) typed_error_enums: std::collections::HashSet<String>,
    /// Type names (struct/enum) that already have a Display impl from `as string:` conversions.
    /// Auto-generated Display is suppressed for these to avoid E0119 conflicting impls.
    pub(crate) display_types: std::collections::HashSet<String>,
    /// Top-level functions that declare `task` (async).
    pub(crate) task_fns: std::collections::HashSet<String>,
    /// Instance method names that declare `task` (async) — used to add .await at call sites.
    pub(crate) instance_task_methods: std::collections::HashSet<String>,
    /// Local variables in the current function that hold a future (created with `task expr`).
    pub(crate) task_vars: std::collections::HashSet<String>,
    /// Function parameters whose type is `fn() throws` (returns Result).
    /// Calling them in a throws context requires `?` propagation.
    pub(crate) throws_fn_params: std::collections::HashSet<String>,
    /// Local variables that hold an `Arc<T>` value (string or T'task qualifier).
    /// These must be cloned before being moved into `async move {}` so the outer
    /// binding remains valid after the spawn.
    pub(crate) arc_vars: std::collections::HashSet<String>,
    /// Local variables declared with `T'weak` — already `Weak<T>`, skip Rc::downgrade.
    pub(crate) weak_vars: std::collections::HashSet<String>,
    /// Variadic param index per function: fn_name → index of the `...` param.
    pub(crate) fn_variadic: std::collections::HashMap<String, usize>,
    /// Inside a `try:` body closure — calls to throws functions get `?`.
    pub(crate) in_try_body: bool,
    /// True while emitting a `type set` body — prevents recursive setter dispatch.
    pub(crate) in_type_setter: bool,
    /// True while emitting an `init` body — `self` must be emitted as `__self`.
    pub(crate) in_init_body: bool,
    /// "StructName::var_name" → present if it's an immutable `type let`.
    pub(crate) struct_type_var_names: std::collections::HashSet<String>,
    /// "StructName::var_name" → present if it's a mutable `type var`.
    pub(crate) struct_type_mut_var_names: std::collections::HashSet<String>,
    /// StructName → { method_name → TypeMethodKind } for type method dispatch.
    pub(crate) struct_type_method_sigs: std::collections::HashMap<String, std::collections::HashMap<String, TypeMethodKind>>,
    /// "StructName::getter_name" → present for `req` (property getter) methods.
    /// These are accessed without parens in boring (`t.fahrenheit`) but emit as `t.fahrenheit()` in Rust.
    pub(crate) struct_getters: std::collections::HashSet<String>,
    /// "StructName::setter_name" → present for `set` methods.
    /// Assignment `t.prop = v` should emit `t.set_prop(v)`.
    pub(crate) struct_setters: std::collections::HashSet<String>,
    /// "StructName::field_name" → (is_copy, field_type, default_init): transient fields.
    /// `default_init` is the pre-computed inner default value string (e.g. "None", "0_i64").
    pub(crate) transient_fields: std::collections::HashMap<String, (bool, Type, String)>,
    /// Local variable name → struct type name, for variables bound to struct constructors.
    /// Used to resolve getter calls on non-`self` receivers (e.g. `t.fahrenheit`).
    pub(crate) var_struct_types: std::collections::HashMap<String, String>,
    /// Variable names declared as `T'actor` — hold `Arc<tokio::sync::Mutex<T>>`.
    /// All field reads/writes and method calls go through `.lock().await`.
    pub(crate) var_mutex_types: std::collections::HashSet<String>,
    /// "StructName::field_name" for fields typed `var T'task` (Arc<Mutex<T>> in Rust).
    pub(crate) struct_mutex_fields: std::collections::HashSet<String>,
    /// Variable names declared as `T'guard` — hold `Arc<tokio::sync::RwLock<T>>`.
    /// Reads go through `.read().await`, writes through `.write().await`.
    pub(crate) var_rwlock_types: std::collections::HashSet<String>,
    /// "StructName::field_name" for fields typed `T'guard` (Arc<RwLock<T>> in Rust).
    pub(crate) struct_rwlock_fields: std::collections::HashSet<String>,
    /// "StructName::method_name" for methods that are non-mutating (`req`).
    /// Used by 'guard dispatch to choose `.read()` vs `.write()`.
    pub(crate) struct_req_methods: std::collections::HashSet<String>,
    /// Struct types that implement the iterator protocol: a `def T? next():` method.
    /// When the iterable of a `for` loop is one of these types, the transpiler emits
    /// `while let Some(x) = __iter.next()` instead of `.into_iter()`.
    pub(crate) iterable_structs: std::collections::HashSet<String>,
    /// All local variable names in the current function scope.
    /// Used to distinguish module/type paths (use `::`) from instance variable access (use `.`).
    pub(crate) known_local_vars: std::collections::HashSet<String>,
    /// True when the current function's declared return type is `()` (void).
    /// Prevents expression-return without semicolon for void functions.
    pub(crate) fn_returns_void: bool,
    /// trait_name → set of method names declared in that trait (signatures + defaults).
    /// Used to split struct body methods between `impl Trait for Struct {}` and `impl Struct {}`.
    pub(crate) trait_method_names: std::collections::HashMap<String, std::collections::HashSet<String>>,
    /// Types reachable via a user-defined `as T:` conversion (lowercased).
    /// Used in emit_expr to route `x as T` to the generated `into_t()` method.
    pub(crate) user_conv_targets: std::collections::HashSet<String>,
    /// Local variables declared as mutable strings (`var x = ""`) — typed Arc<str>.
    /// Used to emit Arc::make_mut(&mut x) for read_line args and clear() calls.
    pub(crate) string_arc_vars: std::collections::HashSet<String>,
    /// All local variables known to hold `Arc<str>` (string type), including params.
    /// Used for string concatenation detection: `x + y` where both are strings.
    pub(crate) string_vars: std::collections::HashSet<String>,
    /// Type parameters already declared at the `impl<...>` level (set by emit_ext).
    /// Methods inside a generic impl must NOT re-declare these params on their own `fn<...>`.
    pub(crate) impl_type_params: Vec<String>,
    /// Declared return type of the current function, if known.
    /// Used to coerce last-expression returns with `Some()` when the return type is Optional.
    pub(crate) fn_return_ty: Option<Type>,
    /// Names declared as `type Name as InnerType` newtype wrappers.
    /// Used in emit_constructor to emit `Name(val)` (tuple struct) rather than `Name { field: val }`.
    pub(crate) newtype_types: std::collections::HashSet<String>,
    /// Maps newtype name → inner Rust type string (e.g. "UserId" → "u64").
    /// Used in emit_cast to emit `val.0` when unwrapping via `x as InnerType`.
    pub(crate) newtype_inner: std::collections::HashMap<String, String>,
    /// Maps local variable name → its newtype type name.
    /// Populated by emit_let and emit_fn (params) to enable `id as uint` → `id.0`.
    pub(crate) var_newtype_type: std::collections::HashMap<String, String>,
    /// Top-level functions that declare `stream` (async generator).
    pub(crate) stream_fns: std::collections::HashSet<String>,
    /// Stream functions that also declare `throws` — use `try_stream!` and unwrap at consumer.
    pub(crate) stream_throws_fns: std::collections::HashSet<String>,
    /// True when the file contains at least one stream function (adds async-stream deps).
    pub(crate) has_streams: bool,
    /// Variables that are `mpsc::Receiver<T>` — `for x in rx:` emits `rx.recv().await`.
    pub(crate) channel_receivers: std::collections::HashSet<String>,
    /// Subset of `channel_receivers` whose element type is `string` / `Arc<str>`.
    /// Values received from these channels are `Arc<str>` and need special handling in comparisons.
    pub(crate) string_channel_receivers: std::collections::HashSet<String>,
    /// Variables that are `mpsc::Sender<T>` — `tx.send(x)` emits `.send(x).await.unwrap()`.
    pub(crate) channel_senders: std::collections::HashSet<String>,
    /// Variables that are `oneshot::Receiver<T>` — consumed once via `.await.unwrap()`.
    pub(crate) oneshot_receivers: std::collections::HashSet<String>,
    /// Variables that are `oneshot::Sender<T>` — `tx.send(v)` emits `.send(v).ok()`.
    pub(crate) oneshot_senders: std::collections::HashSet<String>,
    /// Variables that are `broadcast::Receiver<T>`.
    pub(crate) broadcast_receivers: std::collections::HashSet<String>,
    /// Variables that are `broadcast::Sender<T>`.
    pub(crate) broadcast_senders: std::collections::HashSet<String>,
    /// Variables that are `watch::Receiver<T>`.
    pub(crate) watch_receivers: std::collections::HashSet<String>,
    /// Variables that are `watch::Sender<T>`.
    pub(crate) watch_senders: std::collections::HashSet<String>,
    /// Variables that hold a `JoinHandle<T>` (from `let f = task: expr`).
    /// `.value` on these emits `.await.unwrap()` or `.await?`.
    pub(crate) join_handle_vars: std::collections::HashSet<String>,
    /// Subset of `join_handle_vars` where the spawned function is `throws`.
    /// The JoinHandle wraps `Result<T, BoringError>`, so `.value` / `.wait` need a
    /// double unwrap: `f.await.unwrap()?` (throws ctx) or `f.await.unwrap().unwrap()`.
    pub(crate) throws_join_handle_vars: std::collections::HashSet<String>,
    /// trait_name → set of associated type names declared in that trait.
    /// Used to emit `type X = Y;` in `impl Trait for Struct` blocks.
    pub(crate) trait_assoc_type_names: std::collections::HashMap<String, std::collections::HashSet<String>>,
    /// True while emitting an `impl Trait for Struct` block.
    /// Task+mutating methods inside a trait impl use `&mut self` to match the trait signature.
    pub(crate) inside_trait_impl: bool,
    /// The set of associated type names defined by the current trait impl (e.g. "Output", "Inner").
    /// When non-empty, `emit_type` converts bare `Type::Named(n)` to `Self::n` for these names.
    pub(crate) current_trait_assoc_names: std::collections::HashSet<String>,
    /// When inside a match statement, the inferred enum type of the subject (if known).
    /// Used to resolve unqualified variant patterns to the correct enum.
    pub(crate) match_subject_enum: Option<String>,
    /// Local variable name → its declared boring type (for match subject enum inference).
    pub(crate) var_types: std::collections::HashMap<String, Type>,
    /// "StructName::method_name" for methods that are operator overloads (add, sub, mul, div,
    /// rem, neg, eq, ne, lt, le, gt, ge). BinOp dispatch emits `a.clone().method(b.clone())`
    /// instead of `(a op b)` when the left operand's struct has the method registered.
    pub(crate) struct_operator_methods: std::collections::HashSet<String>,
    /// "StructName::method_name" → Vec<Type> of param types (for operator method dispatch boxing).
    pub(crate) struct_operator_param_types: std::collections::HashMap<String, Vec<Type>>,
    /// Variable name → struct type name for locally-declared vars that hold struct values.
    /// Populated from `let x = StructName(...)` constructor calls.
    pub(crate) var_struct_type: std::collections::HashMap<String, String>,
    /// fn_name → ordered parameter names (for reordering named/labeled call arguments).
    pub(crate) fn_param_names: std::collections::HashMap<String, Vec<String>>,
    /// True when the user defines a type named `Box` (conflicts with std::boxed::Box).
    pub(crate) user_defines_box: bool,
    /// True when the user defines a type named `Result` (conflicts with std::result::Result).
    pub(crate) user_defines_result: bool,
    /// Structs that have an init with a non-empty body — constructor calls must use ::new() not a struct literal.
    pub(crate) struct_has_init_body: std::collections::HashSet<String>,
    /// struct_name → Vec<Option<String>> of default values for init params (parallel to init params).
    pub(crate) struct_init_defaults: std::collections::HashMap<String, Vec<Option<String>>>,
    /// Top-level mutable `var` declarations accessed inside function bodies.
    /// These can't be local to `main()` and must be emitted as module-level statics.
    /// Maps var_name → declared boring type (None = inferred as Arc<str>).
    pub(crate) global_var_types: std::collections::HashMap<String, Option<Type>>,
    /// Top-level var initial value expressions (pre-emitted strings) for global statics.
    pub(crate) global_var_inits: std::collections::HashMap<String, String>,
    /// Set of global var names that functions access — these are emitted as LazyLock<Mutex<T>>.
    pub(crate) global_vars_used_in_fns: std::collections::HashSet<String>,
    /// Variables that hold `Option<numeric>` (int/uint/float) — from string-to-numeric casts.
    /// Used in `else` coalescing: when the optional is numeric but default is String, use map_or_else.
    pub(crate) optional_numeric_vars: std::collections::HashSet<String>,
    /// Variables always assigned `None` (e.g. from `int as bool` casts) — `else` coalescing
    /// with a string default can emit the default directly (None always yields the fallback).
    pub(crate) always_none_vars: std::collections::HashSet<String>,
    /// Function type aliases: `use Pure as req int(int)` → "Pure" → Type::Fn(...)
    /// Used to expand named fn-type aliases inline in function parameter types.
    pub(crate) fn_type_aliases: std::collections::HashMap<String, Type>,
    /// Non-function type aliases: `use Pt as LPoint'` → "Pt" → Type::Qualified(LPoint, Owned)
    /// Used to coerce call arguments when the expected param type resolves to a boxed type.
    pub(crate) non_fn_type_aliases: std::collections::HashMap<String, Type>,
    /// Type parameter names of the currently-emitting function (e.g. ["S", "T"]).
    /// Used in emit_match to detect generic-typed match subjects.
    pub(crate) current_fn_type_params: std::collections::HashSet<String>,
    /// "StructName::method_name" for struct methods overridden by an ext block.
    /// When emit_struct emits the plain impl, it skips methods in this set — the ext block's
    /// version wins. This prevents E0592 duplicate definition errors.
    pub(crate) struct_ext_method_overrides: std::collections::HashSet<String>,
    /// Variable names that participate in `is` reference-identity comparisons between struct
    /// instances (e.g. `cdb is cda`). Such variables must be wrapped in `Rc<T>` so that
    /// assignment creates a reference alias (Rc::clone) and `is` uses Rc::ptr_eq.
    pub(crate) rc_identity_vars: std::collections::HashSet<String>,
    /// Names of Boring `mod` blocks in the current program.
    /// `use boring_mod.*` / `use boring_mod.x` must be suppressed because `emit_mod` inlines
    /// items directly — no Rust `mod` block is created, so `use boring_mod::*` is unresolvable.
    pub(crate) boring_mod_names: std::collections::HashSet<String>,
    /// True when the program calls any of the log-level builtins (error/warn/info/debug/trace).
    /// The CLI uses this to warn that `log = "0.4"` is needed in Cargo.toml.
    /// Uses Cell<bool> so it can be set from &self emit methods.
    pub(crate) uses_log: std::cell::Cell<bool>,
    /// True when an enum with @error("...") variants is emitted (thiserror auto-derive).
    /// The CLI uses this to add `thiserror = "1"` to Cargo.toml.
    pub(crate) uses_thiserror: std::cell::Cell<bool>,
    /// True when the program imports from `reqwest`.
    /// The CLI uses this to add `reqwest` to Cargo.toml.
    pub(crate) uses_reqwest: bool,
    /// Task def functions that contain `Task.cancelled()` — get an implicit `__task_cancel` param.
    pub(crate) cancellable_task_fns: std::collections::HashSet<String>,
    /// join_handle_var → cancel_token_local_var (e.g. "f" → "__cancel_f").
    pub(crate) cancel_token_vars: std::collections::HashMap<String, String>,
    /// True while emitting the body of a task def that uses Task.cancelled().
    pub(crate) in_cancellable_fn: bool,
    /// True when tokio-util is needed (for CancellationToken).
    pub(crate) uses_tokio_util: std::cell::Cell<bool>,
    /// Set when `json()` or `fromJson()` is used — triggers serde/serde_json deps.
    pub(crate) uses_serde: std::cell::Cell<bool>,
}

impl Transpiler {
    fn new() -> Self {
        Transpiler {
            out: String::new(),
            indent: 0,
            in_throws: false,
            in_async: false,
            self_type: None,
            collection_vars: std::collections::HashSet::new(),
            vec_vars: std::collections::HashSet::new(),
            set_vars: std::collections::HashSet::new(),
            dict_vars: std::collections::HashSet::new(),
            chars_vars: std::collections::HashSet::new(),
            fn_return_types: std::collections::HashMap::new(),
            index_vars: std::collections::HashSet::new(),
            fn_sigs: std::collections::HashMap::new(),
            enum_variants: std::collections::HashMap::new(),
            enum_variant_fields: std::collections::HashMap::new(),
            enum_variant_field_types: std::collections::HashMap::new(),
            optional_vars: std::collections::HashSet::new(),
            fn_defaults: std::collections::HashMap::new(),
            struct_fields: std::collections::HashMap::new(),
            struct_assoc_types: std::collections::HashMap::new(),
            fn_throws: std::collections::HashSet::new(),
            typed_error_enums: std::collections::HashSet::new(),
            display_types: std::collections::HashSet::new(),
            task_fns: std::collections::HashSet::new(),
            instance_task_methods: std::collections::HashSet::new(),
            task_vars: std::collections::HashSet::new(),
            throws_fn_params: std::collections::HashSet::new(),
            arc_vars: std::collections::HashSet::new(),
            weak_vars: std::collections::HashSet::new(),
            fn_variadic: std::collections::HashMap::new(),
            in_try_body: false,
            in_type_setter: false,
            in_init_body: false,
            struct_type_var_names: std::collections::HashSet::new(),
            struct_type_mut_var_names: std::collections::HashSet::new(),
            struct_type_method_sigs: std::collections::HashMap::new(),
            struct_getters: std::collections::HashSet::new(),
            struct_setters: std::collections::HashSet::new(),
            transient_fields: std::collections::HashMap::new(),
            var_struct_types: std::collections::HashMap::new(),
            var_mutex_types: std::collections::HashSet::new(),
            struct_mutex_fields: std::collections::HashSet::new(),
            var_rwlock_types: std::collections::HashSet::new(),
            struct_rwlock_fields: std::collections::HashSet::new(),
            struct_req_methods: std::collections::HashSet::new(),
            iterable_structs: std::collections::HashSet::new(),
            known_local_vars: std::collections::HashSet::new(),
            fn_returns_void: false,
            trait_method_names: std::collections::HashMap::new(),
            user_conv_targets: std::collections::HashSet::new(),
            string_arc_vars: std::collections::HashSet::new(),
            string_vars: std::collections::HashSet::new(),
            impl_type_params: Vec::new(),
            fn_return_ty: None,
            newtype_types: std::collections::HashSet::new(),
            newtype_inner: std::collections::HashMap::new(),
            var_newtype_type: std::collections::HashMap::new(),
            stream_fns: std::collections::HashSet::new(),
            stream_throws_fns: std::collections::HashSet::new(),
            has_streams: false,
            channel_receivers: std::collections::HashSet::new(),
            string_channel_receivers: std::collections::HashSet::new(),
            channel_senders: std::collections::HashSet::new(),
            oneshot_receivers: std::collections::HashSet::new(),
            oneshot_senders: std::collections::HashSet::new(),
            broadcast_receivers: std::collections::HashSet::new(),
            broadcast_senders: std::collections::HashSet::new(),
            watch_receivers: std::collections::HashSet::new(),
            watch_senders: std::collections::HashSet::new(),
            join_handle_vars: std::collections::HashSet::new(),
            throws_join_handle_vars: std::collections::HashSet::new(),
            trait_assoc_type_names: std::collections::HashMap::new(),
            inside_trait_impl: false,
            current_trait_assoc_names: std::collections::HashSet::new(),
            match_subject_enum: None,
            var_types: std::collections::HashMap::new(),
            struct_operator_methods: std::collections::HashSet::new(),
            struct_operator_param_types: std::collections::HashMap::new(),
            var_struct_type: std::collections::HashMap::new(),
            fn_param_names: std::collections::HashMap::new(),
            user_defines_box: false,
            user_defines_result: false,
            struct_has_init_body: std::collections::HashSet::new(),
            struct_init_defaults: std::collections::HashMap::new(),
            global_var_types: std::collections::HashMap::new(),
            global_var_inits: std::collections::HashMap::new(),
            global_vars_used_in_fns: std::collections::HashSet::new(),
            optional_numeric_vars: std::collections::HashSet::new(),
            always_none_vars: std::collections::HashSet::new(),
            fn_type_aliases: std::collections::HashMap::new(),
            non_fn_type_aliases: std::collections::HashMap::new(),
            current_fn_type_params: std::collections::HashSet::new(),
            struct_ext_method_overrides: std::collections::HashSet::new(),
            rc_identity_vars: std::collections::HashSet::new(),
            boring_mod_names: std::collections::HashSet::new(),
            uses_log: std::cell::Cell::new(false),
            uses_thiserror: std::cell::Cell::new(false),
            uses_reqwest: false,
            cancellable_task_fns: std::collections::HashSet::new(),
            cancel_token_vars: std::collections::HashMap::new(),
            in_cancellable_fn: false,
            uses_tokio_util: std::cell::Cell::new(false),
            uses_serde: std::cell::Cell::new(false),
        }
    }



    // ── Output helpers ────────────────────────────────────────────────────────

    fn ind(&self) -> String {
        "    ".repeat(self.indent)
    }

    fn line(&mut self, s: &str) {
        let ind = self.ind();
        self.out.push_str(&ind);
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn blank(&mut self) {
        self.out.push('\n');
    }

    // ── Program ───────────────────────────────────────────────────────────────

    fn emit_program(&mut self, program: &Program) {
        // Pre-scan: collect enum variants and fn defaults before emitting anything.
        self.pre_scan(program);

        // Standard prelude
        self.line("// Generated by boring --emit-rust");
        self.line("use std::collections::{HashMap, HashSet};");
        self.line("use std::hash::Hash;");
        self.line("use std::rc::{Rc, Weak};");
        self.line("use std::sync::Arc;");
        self.line("use std::f64::consts::{PI, E, TAU};");
        self.line("use std::time::Duration;");
        self.blank();
        // Collection index traits — implement the boring Index API on Rust collections.
        // Three separate traits so each collection has its own natural index type:
        //   Vec<T>         → Option<usize>  (BoringArrayIndex)
        //   HashMap<K,V>   → Option<K>      (BoringDictIndex)
        //   HashSet<T>     → Option<usize>  (BoringSetIndex, positional; set[i] uses .iter().nth())
        // `remove_at` accepts Option so callers can pass firstIndex()/nextIndex() directly.
        self.out.push_str(
            "trait BoringArrayIndex {\n\
             \x20   type Item: Clone;\n\
             \x20   fn first_index(&self) -> Option<usize>;\n\
             \x20   fn next_index(&self, i: Option<usize>) -> Option<usize>;\n\
             \x20   fn remove_at(&self, i: Option<usize>) -> Vec<Self::Item>;\n\
             \x20   fn get_at(&self, i: Option<usize>) -> Self::Item;\n\
             }\n\
             impl<T: Clone> BoringArrayIndex for Vec<T> {\n\
             \x20   type Item = T;\n\
             \x20   fn first_index(&self) -> Option<usize> {\n\
             \x20       if self.is_empty() { None } else { Some(0) }\n\
             \x20   }\n\
             \x20   fn next_index(&self, i: Option<usize>) -> Option<usize> {\n\
             \x20       i.and_then(|i| if i + 1 < self.len() { Some(i + 1) } else { None })\n\
             \x20   }\n\
             \x20   fn remove_at(&self, i: Option<usize>) -> Vec<T> {\n\
             \x20       let mut v = self.clone();\n\
             \x20       if let Some(i) = i { if i < v.len() { v.remove(i); } }\n\
             \x20       v\n\
             \x20   }\n\
             \x20   fn get_at(&self, i: Option<usize>) -> T { self[i.unwrap()].clone() }\n\
             }\n\
             trait BoringDictIndex<K: Clone + Eq + std::hash::Hash, V: Clone> {\n\
             \x20   fn first_index(&self) -> Option<K>;\n\
             \x20   fn next_index(&self, k: K) -> Option<K>;\n\
             \x20   fn remove_at(&self, k: Option<K>) -> Self where Self: Sized;\n\
             }\n\
             impl<K: Clone + Eq + std::hash::Hash, V: Clone> BoringDictIndex<K, V> for HashMap<K, V> {\n\
             \x20   fn first_index(&self) -> Option<K> { self.keys().next().cloned() }\n\
             \x20   fn next_index(&self, k: K) -> Option<K> {\n\
             \x20       let keys: Vec<_> = self.keys().collect();\n\
             \x20       let pos = keys.iter().position(|kk| *kk == &k);\n\
             \x20       pos.and_then(|p| keys.get(p + 1)).map(|kk| (*kk).clone())\n\
             \x20   }\n\
             \x20   fn remove_at(&self, k: Option<K>) -> Self { let mut m = self.clone(); if let Some(k) = k { m.remove(&k); } m }\n\
             }\n\
             trait BoringSetIndex {\n\
             \x20   type Item: Clone;\n\
             \x20   fn first_index(&self) -> Option<usize>;\n\
             \x20   fn next_index(&self, i: Option<usize>) -> Option<usize>;\n\
             \x20   fn remove_at(&self, i: Option<usize>) -> Self where Self: Sized;\n\
             \x20   fn get_at(&self, i: Option<usize>) -> Self::Item;\n\
             }\n\
             impl<T: Clone + Eq + std::hash::Hash> BoringSetIndex for HashSet<T> {\n\
             \x20   type Item = T;\n\
             \x20   fn first_index(&self) -> Option<usize> {\n\
             \x20       if self.is_empty() { None } else { Some(0) }\n\
             \x20   }\n\
             \x20   fn next_index(&self, i: Option<usize>) -> Option<usize> {\n\
             \x20       i.and_then(|i| if i + 1 < self.len() { Some(i + 1) } else { None })\n\
             \x20   }\n\
             \x20   fn remove_at(&self, i: Option<usize>) -> Self {\n\
             \x20       let elem = i.and_then(|i| self.iter().nth(i).cloned());\n\
             \x20       let mut s = self.clone();\n\
             \x20       if let Some(e) = elem { s.remove(&e); }\n\
             \x20       s\n\
             \x20   }\n\
             \x20   fn get_at(&self, i: Option<usize>) -> T { self.iter().nth(i.unwrap()).cloned().unwrap() }\n\
             }\n\n\
             // BoringFmt — displays Vec<T: Display> as [a, b, c] (no debug quotes on strings).\n\
             struct BoringFmt<'a, T: std::fmt::Display>(pub &'a [T]);\n\
             impl<'a, T: std::fmt::Display> std::fmt::Display for BoringFmt<'a, T> {\n\
             \x20   fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {\n\
             \x20       write!(f, \"[\")?;\n\
             \x20       for (i, v) in self.0.iter().enumerate() {\n\
             \x20           if i > 0 { write!(f, \", \")?; }\n\
             \x20           write!(f, \"{}\", v)?;\n\
             \x20       }\n\
             \x20       write!(f, \"]\")\n\
             \x20   }\n\
             }\n\n\
             // BoringVal — bridge trait that gives BoringError::Other both Display and Any (downcast).\n\
             // Every user-defined error enum automatically satisfies the blanket impl below.\n\
             trait BoringVal: std::fmt::Display + std::any::Any + Send + Sync {\n\
             \x20   fn as_any(&self) -> &dyn std::any::Any;\n\
             }\n\
             impl<T: std::fmt::Display + std::any::Any + Send + Sync + 'static> BoringVal for T {\n\
             \x20   fn as_any(&self) -> &dyn std::any::Any { self }\n\
             }\n\
             impl std::fmt::Debug for dyn BoringVal + Send + Sync {\n\
             \x20   fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {\n\
             \x20       write!(f, \"BoringVal({})\", self)\n\
             \x20   }\n\
             }\n\n\
             // BoringError — typed exception wrapper so `catch String:` / `catch Int:` / `catch MyError:` can dispatch.\n\
             // `Other(TypeId, error)` identifies the thrown type uniquely across modules via std::any::TypeId::of::<T>(),\n\
             // which requires no instance — fully collision-free even when two modules define identically-named types.\n\
             #[derive(Debug)]\n\
             enum BoringError {\n\
             \x20   Int(i64),\n\
             \x20   Float(f64),\n\
             \x20   Bool(bool),\n\
             \x20   Str(&'static str),\n\
             \x20   String(Arc<str>),\n\
             \x20   Other(std::any::TypeId, std::boxed::Box<dyn BoringVal + Send + Sync>),\n\
             }\n\
             impl std::fmt::Display for BoringError {\n\
             \x20   fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {\n\
             \x20       match self {\n\
             \x20           BoringError::Int(n)         => write!(f, \"{}\", n),\n\
             \x20           BoringError::Float(n)       => write!(f, \"{}\", n),\n\
             \x20           BoringError::Bool(b)        => write!(f, \"{}\", b),\n\
             \x20           BoringError::Str(s)         => write!(f, \"{}\", s),\n\
             \x20           BoringError::String(s)      => write!(f, \"{}\", s),\n\
             \x20           BoringError::Other(_, e)    => write!(f, \"{}\", e),\n\
             \x20       }\n\
             \x20   }\n\
             }\n\
             impl std::error::Error for BoringError {}\n\n\
             // Boring standard error enum — always available without import.\n\
             // Use `throw Error.Expired` / `catch Error:` / `match err: Error.Expired: ...`\n\
             #[derive(Debug, Clone)]\n\
             #[allow(dead_code)]\n\
             enum Error {\n\
             \x20   Expired,\n\
             \x20   Cancelled,\n\
             \x20   NotFound,\n\
             \x20   InvalidInput,\n\
             \x20   OutOfBounds,\n\
             }\n\
             impl std::fmt::Display for Error {\n\
             \x20   fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {\n\
             \x20       match self {\n\
             \x20           Error::Expired      => write!(f, \"timeout expired\"),\n\
             \x20           Error::Cancelled    => write!(f, \"task cancelled\"),\n\
             \x20           Error::NotFound     => write!(f, \"not found\"),\n\
             \x20           Error::InvalidInput => write!(f, \"invalid input\"),\n\
             \x20           Error::OutOfBounds  => write!(f, \"index out of bounds\"),\n\
             \x20       }\n\
             \x20   }\n\
             }\n\
             impl std::error::Error for Error {}\n\n"
        );

        // Emit module-level mutable vars that are accessed by functions as LazyLock<Mutex<T>> statics.
        // These cannot be locals inside main() because functions reference them as free variables.
        if !self.global_vars_used_in_fns.is_empty() {
            // Collect, then deduplicate by name keeping the LAST declaration (shadowing: the
            // last `var x = ...` wins).  Without dedup, two `var i = ...` shadow declarations
            // would produce two `static I: ...` causing E0428 "defined multiple times".
            let global_vars_raw: Vec<(String, Option<Type>, String)> = program.items.iter()
                .filter_map(|item| {
                    if let Item::Let(s) = item {
                        if s.mutable && self.global_vars_used_in_fns.contains(&s.name) {
                            let init = self.global_var_inits.get(&s.name).cloned()
                                .unwrap_or_else(|| "Default::default()".into());
                            Some((s.name.clone(), s.ty.clone(), init))
                        } else { None }
                    } else { None }
                })
                .collect();
            // Deduplicate: last declaration for each name wins.
            let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();
            let global_vars: Vec<(String, Option<Type>, String)> = global_vars_raw.into_iter().rev()
                .filter(|(name, _, _)| seen_names.insert(name.clone()))
                .collect::<Vec<_>>()
                .into_iter().rev().collect();
            for (name, ty, init) in &global_vars {
                let rust_ty = if let Some(t) = ty {
                    self.emit_type(t)
                } else if init.starts_with("Arc::new(") || init.starts_with("Arc::<str>::from(") || init.starts_with("\"") {
                    "Arc<str>".to_string()
                } else if init.starts_with("vec![") {
                    "Vec<i64>".to_string()
                } else {
                    "i64".to_string()
                };
                let static_name = name.to_uppercase();
                // Use std::sync::LazyLock<Mutex<T>> for module-level mutable state.
                self.line(&format!(
                    "static {}: std::sync::LazyLock<std::sync::Mutex<{}>> = std::sync::LazyLock::new(|| std::sync::Mutex::new({}));",
                    static_name, rust_ty, init
                ));
                self.blank();
            }
        }

        // Separate top-level declarations from statements
        let mut stmts: Vec<&Item> = Vec::new();
        let mut emitted_fn_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        for item in &program.items {
            match item {
                Item::Stmt(_) | Item::Let(_) => stmts.push(item),
                Item::Fn(f) if f.qualifier.is_none() => {
                    // Skip duplicate top-level function definitions (same name = same impl).
                    if !emitted_fn_names.insert(f.name.clone()) {
                        continue;
                    }
                    self.emit_item(item);
                    self.blank();
                }
                _ => {
                    self.emit_item(item);
                    self.blank();
                }
            }
        }

        // If a user-defined `main` function already exists (e.g. `def void main() task:`),
        // do NOT emit a second `fn main` from top-level statements to avoid duplicate symbol.
        let has_explicit_main = program.items.iter().any(|item| {
            matches!(item, Item::Fn(f) if f.name == "main" && f.qualifier.is_none())
        });

        // Top-level statements → fn main()
        // All stmts are side-effectful; in_throws stays false so no Ok() wrapping happens.
        // We always append Ok(()) explicitly.
        //
        // Auto-promotion: if any top-level stmt contains a `task expr` (detached or bound),
        // `main` must be async so `tokio::spawn` can be called.  We detect this here and
        // emit `#[tokio::main] async fn main()` instead of the plain `fn main()`.
        if stmts.is_empty() && !has_explicit_main {
            // Library-style file: no top-level statements, no user-defined main.
            // Emit a stub fn main() so the binary crate compiles.
            self.line("fn main() {}");
        } else if !stmts.is_empty() && !has_explicit_main {
            let top_stmts: Vec<Stmt> = stmts.iter().filter_map(|i| {
                if let Item::Stmt(s) = i { Some((*s).clone()) } else { None }
            }).collect();
            let needs_async = items_have_task(&stmts)
                || body_has_stream_for(&top_stmts, &self.stream_fns)
                || items_have_task_call(&stmts, &self.task_fns);
            // Use fully-qualified types when the user defines `Box` or `Result` to avoid
            // shadowing std::boxed::Box and std::result::Result in the main signature.
            let result_ty = if self.user_defines_result { "std::result::Result" } else { "Result" };
            let box_ty = if self.user_defines_box { "std::boxed::Box" } else { "Box" };
            let main_ret = format!("{}<(), {}<dyn std::error::Error + Send + Sync>>", result_ty, box_ty);
            let ok_ret = if self.user_defines_result { "std::result::Result::Ok(())" } else { "Ok(())" };
            if needs_async {
                self.line("#[tokio::main]");
                self.line(&format!("async fn main() -> {} {{", main_ret));
                self.in_async = true;
            } else {
                self.line(&format!("fn main() -> {} {{", main_ret));
            }
            self.indent += 1;
            // main() returns Result<(), Box<dyn Error>>, so throws function calls should get `?`.
            self.in_throws = true;
            for item in stmts.iter() {
                match item {
                    Item::Let(s) => {
                        // Skip top-level mutable vars that are emitted as module-level statics.
                        if s.mutable && self.global_vars_used_in_fns.contains(&s.name) {
                            // Already emitted as LazyLock<Mutex<T>> static — skip.
                        } else {
                            self.emit_let(s, false);
                        }
                    }
                    Item::Stmt(s) => self.emit_stmt(s, false),
                    _ => {}
                }
            }
            self.line(ok_ret);
            self.in_throws = false;
            self.indent -= 1;
            self.line("}");
            if needs_async {
                self.in_async = false;
            }
        }
    }

    fn pre_scan(&mut self, program: &Program) {
        // Pre-populate the stdlib `Error` enum so it's always available without a user declaration.
        self.typed_error_enums.insert("Error".to_string());
        for variant in &["Expired", "Cancelled", "NotFound", "InvalidInput", "OutOfBounds"] {
            self.enum_variants.insert(variant.to_string(), "Error".to_string());
            let key = format!("Error::{}", variant);
            self.enum_variant_fields.insert(key.clone(), vec![]);
            self.enum_variant_field_types.insert(key, vec![]);
        }

        for item in &program.items {
            match item {
                Item::Enum(e) => {
                    if e.name == "Result" { self.user_defines_result = true; }
                    for v in &e.variants {
                        self.enum_variants.insert(v.name.clone(), e.name.clone());
                        let key = format!("{}::{}", e.name, v.name);
                        let field_names: Vec<Option<String>> = v.fields.iter().map(|f| f.name.clone()).collect();
                        // Unwrap `Owned` (Box) qualifiers from enum variant field types so that
                        // constructor emission and pattern matching work without Box wrapping.
                        // (Recursive enums needing Box are rare and handled by emit_enum itself.)
                        let field_types: Vec<Type> = v.fields.iter().map(|f| match &f.ty {
                            Type::Qualified(inner, OwnerQual::Owned) => *inner.clone(),
                            other => other.clone(),
                        }).collect();
                        self.enum_variant_fields.insert(key.clone(), field_names);
                        self.enum_variant_field_types.insert(key, field_types);
                    }
                    // Register enum getter methods (req methods without params) in struct_getters.
                    for m in &e.methods {
                        if !m.mutating && !m.task && m.params.is_empty() && m.return_ty.is_some() {
                            let key = format!("{}::{}", e.name, m.name);
                            self.struct_getters.insert(key);
                        }
                    }
                    // Register enum `as T:` conversion targets.
                    for conv in &e.conversions {
                        let tname = self.emit_type(&conv.ty);
                        self.user_conv_targets.insert(tname.to_lowercase());
                    }
                }
                Item::Fn(f) => self.pre_register_fn(f),
                Item::Struct(s) if s.name == "Box" => {
                    self.user_defines_box = true;
                    // Also register struct fields (fall through below).
                    let mut fields: Vec<(String, Type)> = s.fields.iter()
                        .map(|f| (f.name.clone(), f.ty.clone()))
                        .collect();
                    if fields.is_empty() {
                        for init in &s.inits {
                            if init.body.is_empty() {
                                for p in &init.params {
                                    if let Some(ty) = &p.ty {
                                        fields.push((p.name.clone(), ty.clone()));
                                    }
                                }
                            }
                        }
                    }
                    self.struct_fields.insert(s.name.clone(), fields);
                }
                Item::Struct(s) => {
                    // Don't register method names in fn_sigs — they're only called as obj.method()
                    // and would shadow top-level functions with the same name.
                    let mut fields: Vec<(String, Type)> = s.fields.iter()
                        .map(|f| (f.name.clone(), f.ty.clone()))
                        .collect();
                    // Also include positional init params (those ARE the struct fields for init structs).
                    if fields.is_empty() {
                        for init in &s.inits {
                            if init.body.is_empty() {
                                // No-body init: params become fields.
                                for p in &init.params {
                                    if let Some(ty) = &p.ty {
                                        fields.push((p.name.clone(), ty.clone()));
                                    }
                                }
                            }
                        }
                    }
                    self.struct_fields.insert(s.name.clone(), fields);
                    // Track inits that have a body (constructor call must use ::new(), not struct literal).
                    // Also collect default values for init params (for filling in omitted args).
                    for init in &s.inits {
                        if !init.body.is_empty() {
                            self.struct_has_init_body.insert(s.name.clone());
                        }
                        // Collect defaults (for both body and body-less inits).
                        let defaults: Vec<Option<String>> = init.params.iter()
                            .map(|p| p.default.as_ref().map(|d| self.emit_expr(d)))
                            .collect();
                        if defaults.iter().any(|d| d.is_some()) {
                            self.struct_init_defaults.insert(s.name.clone(), defaults);
                        }
                    }
                    // Register concrete associated type definitions for `T.AssocName` resolution.
                    if !s.assoc_type_defs.is_empty() {
                        let map: std::collections::HashMap<String, Type> = s.assoc_type_defs
                            .iter()
                            .map(|a| (a.name.clone(), a.ty.clone()))
                            .collect();
                        self.struct_assoc_types.insert(s.name.clone(), map);
                    }
                    // Register type vars and type methods for emission dispatch.
                    for tv in &s.type_vars {
                        let key = format!("{}::{}", s.name, tv.name);
                        if tv.mutable {
                            self.struct_type_mut_var_names.insert(key);
                        } else {
                            self.struct_type_var_names.insert(key);
                        }
                    }
                    let mut method_map = std::collections::HashMap::new();
                    for tm in &s.type_methods {
                        method_map.insert(tm.name.clone(), tm.kind.clone());
                    }
                    if !method_map.is_empty() {
                        self.struct_type_method_sigs.insert(s.name.clone(), method_map);
                    }
                    // Track `req` getter methods (property read, emit as call) and
                    // `set` setter methods (property write, emit as set_X call).
                    for m in &s.methods {
                        // `req` methods without params that return a value are getters.
                        if !m.mutating && !m.task && m.params.is_empty() && m.return_ty.is_some() {
                            let key = format!("{}::{}", s.name, m.name);
                            self.struct_getters.insert(key);
                        }
                    }
                    for setter in &s.setters {
                        let key = format!("{}::{}", s.name, setter.name);
                        self.struct_setters.insert(key);
                    }
                    // Track instance methods that are `task` for .await at call sites.
                    for m in &s.methods {
                        if m.task { self.instance_task_methods.insert(m.name.clone()); }
                    }
                    // Track var T'task struct fields → Arc<Mutex<T>>.
                    for f in &s.fields {
                        if Self::is_mutex_binding(f.mutable, &f.ty) {
                            self.struct_mutex_fields.insert(format!("{}::{}", s.name, f.name));
                        }
                        if Self::is_rwlock_binding(f.mutable, &f.ty) {
                            self.struct_rwlock_fields.insert(format!("{}::{}", s.name, f.name));
                        }
                    }
                    // Track req (non-mutating) methods for 'guard read vs write dispatch.
                    for m in &s.methods {
                        if !m.mutating {
                            self.struct_req_methods.insert(format!("{}::{}", s.name, m.name));
                        }
                    }
                    // Iterator protocol: a struct with `def T? next():` is iterable.
                    // `for x in obj:` desugars to `while let Some(x) = __iter.next()`.
                    for m in &s.methods {
                        if m.name == "next" && m.params.is_empty()
                            && matches!(&m.return_ty, Some(Type::Optional(_)))
                        {
                            self.iterable_structs.insert(s.name.clone());
                        }
                    }
                    // Track transient fields (Cell vs RefCell based on Copy-ness).
                    for f in &s.fields {
                        if f.transient {
                            let key = format!("{}::{}", s.name, f.name);
                            let is_copy = Self::is_copy_type(&f.ty);
                            // Pre-compute the inner default value string for Cell/RefCell init.
                            let default_val = if let Some(def) = &f.default {
                                self.emit_let_value(Some(&f.ty), def)
                            } else {
                                "None".to_string()
                            };
                            self.transient_fields.insert(key, (is_copy, f.ty.clone(), default_val));
                        }
                    }
                    // Register user-defined `as T:` conversion targets.
                    for conv in &s.conversions {
                        let tname = self.emit_type(&conv.ty);
                        self.user_conv_targets.insert(tname.to_lowercase());
                        // `as string:` emits a Display impl — mark the type so auto-Display is skipped.
                        if Self::is_string_conversion(conv) {
                            self.display_types.insert(s.name.clone());
                        }
                    }
                }
                Item::Ext(e) => {
                    // Register user-defined `as T:` conversion targets from extensions too.
                    for conv in &e.conversions {
                        let tname = self.emit_type(&conv.ty);
                        self.user_conv_targets.insert(tname.to_lowercase());
                        // `as string:` in an ext block also emits Display — mark the type.
                        if Self::is_string_conversion(conv) {
                            self.display_types.insert(e.type_name.clone());
                        }
                    }
                    // Pre-scan operator methods so emit_struct can skip deriving PartialEq
                    // when a custom PartialEq impl will be generated by emit_operator_trait_impls.
                    const OPERATOR_METHOD_NAMES: &[&str] = &[
                        "add", "sub", "mul", "div", "rem", "neg",
                        "eq", "ne", "lt", "le", "gt", "ge",
                    ];
                    let tname = &e.type_name;
                    for m in &e.methods {
                        if OPERATOR_METHOD_NAMES.contains(&m.name.as_str()) {
                            self.struct_operator_methods.insert(format!("{}::{}", tname, m.name));
                        }
                        // Register `req` (getter) methods from ext blocks in struct_getters.
                        if !m.mutating && !m.task && m.params.is_empty() && m.return_ty.is_some() {
                            self.struct_getters.insert(format!("{}::{}", tname, m.name));
                        }
                    }
                    // Register setters from ext blocks.
                    for setter in &e.setters {
                        self.struct_setters.insert(format!("{}::{}", tname, setter.name));
                    }
                    // Track methods overriding the struct's own methods (no protocol).
                    // An ext block with no traits overrides plain struct methods of the same name.
                    if e.traits.is_empty() {
                        for m in &e.methods {
                            self.struct_ext_method_overrides.insert(format!("{}::{}", tname, m.name));
                        }
                    }
                }
                Item::Trait(t) => {
                    let mut names = std::collections::HashSet::new();
                    for sig in &t.signatures { names.insert(sig.name.clone()); }
                    for d   in &t.defaults   { names.insert(d.name.clone()); }
                    self.trait_method_names.insert(t.name.clone(), names);
                    // Track associated type names declared in this trait.
                    if !t.assoc_types.is_empty() {
                        let assoc_names: std::collections::HashSet<String> = t.assoc_types.iter()
                            .map(|a| a.name.clone())
                            .collect();
                        self.trait_assoc_type_names.insert(t.name.clone(), assoc_names);
                    }
                }
                Item::Alias(a) if matches!(&a.ty, Type::Fn(..)) => {
                    // Function type aliases: `use Pure as req int(int)` — store for inline expansion.
                    self.fn_type_aliases.insert(a.name.clone(), a.ty.clone());
                }
                Item::Let(s) if s.mutable => {
                    // Top-level mutable var declarations — collect type and initial value.
                    let init_val = self.emit_expr_owned(&s.value);
                    self.global_var_types.insert(s.name.clone(), s.ty.clone());
                    self.global_var_inits.insert(s.name.clone(), init_val);
                }
                Item::Mod(m) => {
                    // Track Boring module names so `use boring_mod.*` can be suppressed.
                    self.boring_mod_names.insert(m.name.clone());
                    // Module items are accessible from the outer scope; scan them too.
                    let pseudo_program = Program { items: m.items.clone() };
                    self.pre_scan(&pseudo_program);
                }
                _ => {}
            }
        }
        // Second pass: find which top-level vars are accessed inside function bodies.
        // Those cannot be local to main() and must be module-level statics.
        let top_var_names: std::collections::HashSet<String> = self.global_var_types.keys().cloned().collect();
        if !top_var_names.is_empty() {
            for item in &program.items {
                if let Item::Fn(f) = item {
                    let param_names: std::collections::HashSet<String> = f.params.iter()
                        .map(|p| p.name.clone()).collect();
                    // Collect locally declared var names inside this function body.
                    // If a function re-declares `var i = 0` locally, its uses of `i` are
                    // references to the local binding — NOT the top-level global.
                    let mut local_decls: std::collections::HashSet<String> = std::collections::HashSet::new();
                    collect_local_decl_names(&f.body, &mut local_decls);
                    let mut body_vars: Vec<String> = Vec::new();
                    for stmt in &f.body {
                        collect_vars_in_stmt(stmt, &mut body_vars);
                    }
                    for v in &body_vars {
                        if top_var_names.contains(v)
                            && !param_names.contains(v)
                            && !local_decls.contains(v)
                        {
                            self.global_vars_used_in_fns.insert(v.clone());
                        }
                    }
                }
            }
        }

        // Collect variables used in `x is y` reference-identity comparisons where both x and y
        // are plain variable names (not type names, nil, or enum variants). Such variables must
        // be wrapped in Rc<T> to support pointer-equality semantics.
        let type_names: std::collections::HashSet<String> =
            self.struct_fields.keys().cloned()
                .chain(self.enum_variants.keys().cloned())
                .chain(self.enum_variant_fields.keys()
                    .filter_map(|k| k.split("::").next().map(|s| s.to_string())))
                .collect();
        let mut identity_vars: std::collections::HashSet<String> = std::collections::HashSet::new();
        for item in &program.items {
            match item {
                Item::Let(l) => collect_is_identity_vars(&l.value, &type_names, &mut identity_vars),
                Item::Stmt(s) => collect_is_identity_stmts(s, &type_names, &mut identity_vars),
                Item::Fn(f) => {
                    for stmt in &f.body {
                        collect_is_identity_stmts(stmt, &type_names, &mut identity_vars);
                    }
                }
                _ => {}
            }
        }
        self.rc_identity_vars = identity_vars;
    }

    fn pre_register_fn(&mut self, f: &FnDecl) {
        let param_types: Vec<Type> = f.params.iter().filter_map(|p| p.ty.clone()).collect();
        let defaults: Vec<Option<String>> = f.params.iter().map(|p| {
            p.default.as_ref().map(|d| self.emit_expr_owned(d))
        }).collect();
        self.fn_sigs.insert(f.name.clone(), param_types);
        self.fn_defaults.insert(f.name.clone(), defaults);
        if let Some(ret_ty) = &f.return_ty {
            self.fn_return_types.insert(f.name.clone(), ret_ty.clone());
        }
        if f.throws {
            self.fn_throws.insert(f.name.clone());
        }
        // Track enum names used as typed error types so emit_enum can add Error impls.
        if let Some(Type::Named(err_name)) = &f.throws_ty {
            self.typed_error_enums.insert(err_name.clone());
        }
        if f.task {
            self.task_fns.insert(f.name.clone());
        }
        if f.stream {
            self.stream_fns.insert(f.name.clone());
            self.has_streams = true;
            if f.throws {
                self.stream_throws_fns.insert(f.name.clone());
            }
        }
        // Top-level `def TypeName.method() task:` → instance task method.
        if f.task && f.qualifier.is_some() {
            self.instance_task_methods.insert(f.name.clone());
        }
        if let Some(idx) = f.params.iter().position(|p| p.variadic) {
            self.fn_variadic.insert(f.name.clone(), idx);
        }
    }

}
