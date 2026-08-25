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
mod emit_let;
mod emit_match;
mod emit_loop;
mod emit_flow;
mod emit_expr;
mod emit_methods;
mod emit_kernel;
mod infer_qualifiers;
pub(crate) mod helpers;
pub(crate) use helpers::*;
pub mod kernel;
pub mod cuda;
pub mod metal;
pub mod wgpu;
pub mod rocm;

// ─── Transpilation config ─────────────────────────────────────────────────────

/// Memory management mode — controls how anonymous `T` and `T'` are resolved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TranspileMode {
    /// Production mode (default). `T` → stack, `T'` → `Box<T>`.
    Strict,
    /// Prototyping mode. `T` and `T'` → `Arc<Mutex<T>>` (multi) or `RefCell<T>` (single).
    Managed,
}

/// Threading model — controls how shared/actor/guard qualifiers are resolved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreadingMode {
    /// Multi-thread Tokio runtime (default). Uses `Arc`, `Mutex`, `RwLock`.
    Multi,
    /// Single-thread Tokio `current_thread` runtime. Uses `Rc`, `RefCell`.
    /// Not available for the Rust-for-Linux target.
    Single,
}

#[derive(Debug, Clone)]
pub struct TranspileConfig {
    pub mode: TranspileMode,
    pub threading: ThreadingMode,
    /// Stack size threshold for auto-boxing (default: 256 bytes). Types larger than this
    /// are silently promoted to Box<T>. Configurable via `--inline-auto-bytes N`.
    pub inline_auto_bytes: usize,
    /// When true, every function body is wrapped with a `__boring_instrument::Span` guard that
    /// records call counts and wall-clock durations.  On program exit the guard writes
    /// `boring_coverage.json` (aggregated stats) and `boring_trace.json` (Chrome Trace Format).
    pub instrument: bool,
    /// Sanitizer to enable in the generated Cargo project (`address`, `thread`, or `memory`).
    /// Requires a nightly toolchain.  Handled by the CLI — the transpiler itself is unaffected.
    pub sanitize: Option<&'static str>,
    /// Directory of the root source file — used to resolve `use` imports at transpile time.
    pub source_dir: std::path::PathBuf,
    /// `kernel Name: ...` declarations known to the program, passed in by GPU targets
    /// (wgpu/cuda/metal) so the general transpiler can special-case kernel construction,
    /// `kernel:` dispatch blocks, and kernel 'unified-field reads wherever they appear —
    /// including inside ordinary function bodies, not just top-level statements. Empty
    /// (the default) for every other target, which leaves current behavior untouched.
    pub gpu_kernels: Vec<crate::ast::KernelDecl>,
    /// True when this transpile_with_config call is producing the "general" (non-kernel)
    /// Rust code that a GPU target (wgpu/cuda/metal) splices into its own generated
    /// main.rs (see transpiler::wgpu::transpile_wgpu). Distinct from `gpu_kernels` being
    /// non-empty: a GPU-target program can have zero kernels *reachable from this
    /// particular file* (e.g. whisper-boring's main.br never imports its own
    /// audio_gpu.br) while still needing every top-level `let` treated as a real
    /// `const` and the auto-generated stub `fn main()` suppressed, because the GPU
    /// target provides its own entry point either way. Empty `gpu_kernels` must NOT
    /// imply "not a GPU target" — this flag is the direct signal for that instead.
    pub is_gpu_target: bool,
    /// True when the GPU target's own host backend already owns top-level kernel/Screen
    /// construction, dispatch, and read-back entirely by itself (wgpu's `emit_screen_main`,
    /// for a `Screen`-using program: kernel instances become `__App` struct fields and the
    /// render loop is inlined into the winit event handler -- see `wgpu::host`). Leftover
    /// top-level statements/non-const `let`s are silently dropped by this pass in that case
    /// instead of being pushed into a synthesized `boring_main` (see `emit_program_items`):
    /// nothing calls `boring_main` for a Screen program, and worse, `emit_kernel.rs`'s
    /// construction logic doesn't understand the `Dimension(w, h)`-shaped/no-`init`
    /// construction convention a render-loop kernel commonly uses, so trying anyway
    /// would panic rather than silently duplicate already-correct behavior.
    pub gpu_top_level_handled_by_host: bool,
    /// Project-declared supplement to `KNOWN_EXTERNAL_TUPLE_STRUCTS` (see that const's doc
    /// comment below), sourced from `boring.toml`'s `[external_types]` `tuple_structs`
    /// array. Lets a project register its own hand-verified external types (e.g. a bevy
    /// type not yet audited into this compiler's built-in list) without a PR against
    /// `boring` itself. Always additive, never overrides the built-in list. Empty unless
    /// `boring build` populated it from `BoringToml`.
    pub external_tuple_structs: Vec<String>,
    /// Project-declared supplement to `KNOWN_EXTERNAL_CONST_FNS`, `(type_name, method)`
    /// pairs sourced from `boring.toml`'s `[external_types]` `const_fns` array (each entry
    /// written `"Type::method"`). Same additive-only relationship to the built-in list as
    /// `external_tuple_structs`.
    pub external_const_fns: Vec<(String, String)>,
    /// Project-declared supplement to `KNOWN_DERIVABLE_TRAITS` (see that const's doc comment
    /// below), sourced from `boring.toml`'s `[derives]` `traits` array. A name in this list,
    /// when it appears in a struct/enum's header `as Trait1, Trait2:` list (or is pulled in
    /// via `[derives]`'s own `include`), is routed into `#[derive(...)]` instead of requiring
    /// a manual `impl Trait for X { ... }` — this is how project-specific derive macros with
    /// no method body (Bevy's `Component`/`Resource`/etc.) get registered without a PR
    /// against `boring` itself. Always additive, never overrides the built-in list. Empty
    /// unless `boring build` populated it from `BoringToml`.
    pub known_derives: Vec<String>,
    /// Named dependencies on other Boring projects, sourced from `boring.toml`'s `[deps]`
    /// (see `docs/cross-project-code-sharing-gap.md`). Maps a dependency name (the
    /// `use <name>.xxx` prefix) to its `src/` directory. Empty unless `boring build`
    /// populated it from `BoringToml::resolve_deps`.
    pub deps: std::collections::HashMap<String, std::path::PathBuf>,
    /// Project-declared supplement to `KNOWN_EXTERNAL_FN_BORROWS` (see that const's doc
    /// comment below), sourced from `boring.toml`'s `[external_fns]` section. Each entry is
    /// `(qualifier, method, arg_borrows)`: `qualifier` is the external type name for a
    /// method-call key (`"ZipFile"`) or the fully-qualified call-site path for a free
    /// function (`"std::mem::swap"`'s module part, `"std::mem"`) — see
    /// `Transpiler::external_fn_arg_borrows`'s doc comment for the exact split. `method` is
    /// the Rust (snake_case, post `camel_to_snake`/`map_method`) name actually emitted at the
    /// call site. `arg_borrows` is one entry per argument *after* the receiver (a method
    /// call's `self`/first free-function arg still counts as position 0 here for a free
    /// function, since a free function has no receiver to exclude) — each entry is `"&"`,
    /// `"&mut"`, or `""` (by value, the pre-existing default). Always additive, never
    /// overrides a built-in entry. Empty unless `boring build` populated it from
    /// `BoringToml`.
    pub external_fns: Vec<(String, String, Vec<String>)>,
}

impl Default for TranspileConfig {
    fn default() -> Self {
        Self {
            mode: TranspileMode::Strict,
            threading: ThreadingMode::Multi,
            inline_auto_bytes: 256,
            instrument: false,
            sanitize: None::<&'static str>,
            source_dir: std::path::PathBuf::new(),
            deps: std::collections::HashMap::new(),
            gpu_kernels: Vec::new(),
            is_gpu_target: false,
            gpu_top_level_handled_by_host: false,
            external_tuple_structs: Vec::new(),
            external_const_fns: Vec::new(),
            known_derives: Vec::new(),
            external_fns: Vec::new(),
        }
    }
}

// ─── Public entry point ───────────────────────────────────────────────────────

/// Shares its definition with the checker's/interpreter's error types (and the
/// checker's warning type); see `crate::errors::SourceError`'s doc comment. Also
/// doubles as `TranspileOutput::warnings`' element type -- same shape either way.
pub use crate::errors::SourceError as TranspileError;

pub struct TranspileOutput {
    pub code: String,
    pub errors: Vec<TranspileError>,
    pub warnings: Vec<TranspileError>,
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
    /// True when the program uses mpsc channels in single-thread mode (local_channel crate).
    /// The caller should add `local_channel` to Cargo.toml when threading is Single.
    pub uses_local_channel: bool,
    /// True when broadcast is used in single-thread mode (local prelude emitted inline).
    /// No extra Cargo.toml dependency — the prelude is self-contained.
    pub uses_local_broadcast: bool,
    /// True when `--instrument` was requested.  The generated code includes the inline
    /// `__boring_instrument` module; no external Cargo.toml dependency is needed.
    pub uses_instrument: bool,
    /// Per-file Rust output from `use` imports resolved to `.br` source files.
    /// Each entry is `(module_name, rust_code)` — written as `src/<module_name>.rs`
    /// and included into `src/main.rs` via `include!` macros.
    pub modules: Vec<(String, String)>,
    /// True when this call emitted a `fn boring_main()` wrapper for GPU-target
    /// top-level statements/non-const `let`s (see `emit_program_items`). GPU targets
    /// (wgpu/cuda/metal) need this to know whether to call `boring_main()` from their
    /// own generated entry point -- a synthesized wrapper exists only in `code`, not as
    /// an `Item::Fn` in the source `Program`, so it can't be detected by AST inspection.
    pub gpu_main_emitted: bool,
}

pub fn transpile(program: &Program) -> String {
    transpile_full(program).code
}

pub fn transpile_full(program: &Program) -> TranspileOutput {
    transpile_with_config(program, TranspileConfig::default())
}

pub fn transpile_with_config(program: &Program, config: TranspileConfig) -> TranspileOutput {
    let mut t = Transpiler::new(config);
    t.source_dir = t.config.source_dir.clone();
    t.deps = t.config.deps.clone();
    t.emit_program(program);
    let apply_single_thread_fixups = |s: String| -> String {
        s
            .replace("Arc::<str>::from", "Rc::<str>::from")
            .replace("Arc::from(", "Rc::from(")
            .replace("Arc::clone(", "Rc::clone(")
            .replace("Arc<str>", "Rc<str>")
            // Global statics require Send — restore Arc<str> inside Mutex for static vars.
            .replace("std::sync::Mutex<Rc<str>>", "std::sync::Mutex<Arc<str>>")
            .replace("std::sync::Mutex::new(Rc::<str>::from(", "std::sync::Mutex::new(Arc::<str>::from(")
            // Promoted top-level `pub let`/used-elsewhere `let` string-literal constants
            // (`top_level_let_is_string_literal`, emit_top.rs) are also a bare `static` —
            // same Send+Sync requirement, same undo, just without the `Mutex<...>` wrapper.
            .replace("std::sync::LazyLock<Rc<str>>", "std::sync::LazyLock<Arc<str>>")
            .replace("std::sync::LazyLock::new(|| Rc::<str>::from(", "std::sync::LazyLock::new(|| Arc::<str>::from(")
            // Assignments to static string vars (through Mutex) also need Arc<str>.
            .replace(".unwrap_or_else(|e| e.into_inner()) = Rc::<str>::from(", ".unwrap_or_else(|e| e.into_inner()) = Arc::<str>::from(")
            // In single-thread mode, error types and BoringVal don't need Send + Sync.
            .replace("Box<dyn std::error::Error + Send + Sync>", "Box<dyn std::error::Error>")
            .replace("trait BoringVal: std::fmt::Display + std::any::Any + Send + Sync", "trait BoringVal: std::fmt::Display + std::any::Any")
            .replace("impl<T: std::fmt::Display + std::any::Any + Send + Sync + 'static> BoringVal for T", "impl<T: std::fmt::Display + std::any::Any + 'static> BoringVal for T")
            .replace("impl std::fmt::Debug for dyn BoringVal + Send + Sync", "impl std::fmt::Debug for dyn BoringVal")
            .replace("Box<dyn BoringVal + Send + Sync>", "Box<dyn BoringVal>")
    };
    let code = if matches!(t.config.threading, ThreadingMode::Single) {
        // Apply fixups to modules too — previously only `t.out` was processed.
        t.modules = t.modules.into_iter()
            .map(|(name, src)| (name, apply_single_thread_fixups(src)))
            .collect();
        apply_single_thread_fixups(t.out)
    } else {
        t.out
    };
    // If the generated code (across main + all modules) defines both `enum Value` and
    // `fn value_equals`, inject a PartialEq impl into the main code so comparisons compile.
    let all_code_combined: String = std::iter::once(code.as_str())
        .chain(t.modules.iter().map(|(_, c)| c.as_str()))
        .collect::<Vec<_>>()
        .concat();
    let code = if all_code_combined.contains("enum Value {") && all_code_combined.contains("fn value_equals(") {
        code + "\nimpl PartialEq for Value {\n    fn eq(&self, other: &Self) -> bool { value_equals(self.clone(), other.clone()) }\n}\nimpl Eq for Value {}\n"
    } else {
        code
    };
    TranspileOutput { code, errors: t.errors.into_inner(), warnings: t.warnings.into_inner(), has_streams: t.has_streams, uses_log: t.uses_log.get(), uses_thiserror: t.uses_thiserror.get(), uses_reqwest: t.uses_reqwest, uses_tokio_util: t.uses_tokio_util.get(), uses_serde: t.uses_serde.get(), uses_local_channel: t.uses_local_channel.get(), uses_local_broadcast: t.uses_local_broadcast.get(), uses_instrument: t.config.instrument, modules: t.modules, gpu_main_emitted: t.gpu_main_emitted.get() }
}

// ─── Transpiler state ─────────────────────────────────────────────────────────

struct Transpiler {
    pub(crate) config: TranspileConfig,
    pub(crate) out: String,
    pub(crate) errors: std::cell::RefCell<Vec<TranspileError>>,
    pub(crate) warnings: std::cell::RefCell<Vec<TranspileError>>,
    pub(crate) indent: usize,
    /// Are we inside a `throws` function body? (return values need Ok() wrapping)
    pub(crate) in_throws: bool,
    /// True while emitting the LHS of an assignment — suppresses `.clone()` on Arc fields.
    /// Uses Cell<bool> because emit_expr takes &self.
    pub(crate) in_lhs_assign: std::cell::Cell<bool>,
    /// True while emitting an expression that will flow directly into an Optional-typed
    /// function return or `let T?` binding — set only around the exact outer expression
    /// (return tail / return stmt / let-value), not while recursing into unrelated
    /// subexpressions. Lets `map_method`'s "pop" mapping skip its `.unwrap_or_default()`
    /// suffix so a bare `arr.pop()` flows through as the raw `Option<T>` `Vec::pop()`
    /// already produces, matching Boring's own declared `req T? pop(): native` semantics
    /// (nil if empty) instead of discarding `None` — see helpers.rs's `map_method` doc.
    /// Uses Cell<bool> because emit_expr takes &self.
    pub(crate) want_raw_option_pop: std::cell::Cell<bool>,
    /// Are we inside a `req` (non-mutating, &self) function body?
    pub(crate) in_req_fn: bool,
    /// Are we emitting a struct/enum method (self_ty.is_some())?
    /// Auto-ref inference is disabled for method params: call sites can't add `&` automatically.
    pub(crate) in_struct_method: bool,
    /// Are we inside a `task` (async) function body?
    pub(crate) in_async: bool,
    /// Are we inside a sequential `stream` body? (`yield` → `__items.push(...)`)
    pub(crate) in_iter_stream: bool,
    /// Name of the type currently being impl'd (for self-aware emit).
    pub(crate) self_type: Option<String>,
    /// Variables known to hold a collection (Vec/HashMap/HashSet) — use {:?} when formatting.
    pub(crate) collection_vars: std::collections::HashSet<String>,
    /// Variables known to hold a Vec specifically (not HashMap/HashSet, and not reduce scalars).
    /// Used to apply BoringFmt wrapping for Display-without-quotes printing.
    pub(crate) vec_vars: std::collections::HashSet<String>,
    /// Variables known to hold a Vec<Arc<str>> — for loop vars iterating these are strings.
    pub(crate) str_vec_vars: std::collections::HashSet<String>,
    /// Variables known to hold a HashSet (for `remove(&v)` and `add`→`insert` dispatch).
    pub(crate) set_vars: std::collections::HashSet<String>,
    /// Variables known to hold a tuple — maps name → arity.
    /// Used to dispatch `.length()`, `.isEmpty()`, `.first()`, `.last()` on tuple vars.
    pub(crate) tuple_vars: std::collections::HashMap<String, usize>,
    /// Variables known to hold a HashMap/dict — subscript reads use `.get()`, writes use `.insert()`.
    pub(crate) dict_vars: std::collections::HashSet<String>,
    /// Variables known to hold a `std::time::Instant` (for `wait(deadline)` →
    /// `sleep_until` and `timeout(deadline)` → `timeout_at` dispatch).
    pub(crate) instant_vars: std::collections::HashSet<String>,
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
    /// "StructName::field" → the field's own declared `= expr` default (plain
    /// `struct` field declarations, not `init` param defaults — see
    /// `struct_init_defaults` for those). Consulted by the `_` fill-rest marker
    /// (`Arg::default_rest`) so a field like `float scale = 1.0` fills back to
    /// its own declared `1.0` instead of the type's blanket zero value.
    pub(crate) struct_field_defaults: std::collections::HashMap<String, Expr>,
    /// Boring-owned struct names ever constructed with the `_` fill-rest marker
    /// (`Arg::default_rest`) — computed by `helpers::collect_default_rest_targets`
    /// during `pre_scan`, before any struct is emitted. `emit_struct` consults this
    /// to add `Default` to the struct's derive list, since `_` lowers to a trailing
    /// `..Default::default()` that needs it. Not populated for/consulted on external
    /// types (e.g. Bevy's `Transform`) — those already implement `Default` themselves.
    pub(crate) structs_needing_default: std::collections::HashSet<String>,
    /// Estimated inline sizes (bytes) for user-defined types, computed during pre_scan.
    /// Used for strict-mode auto-boxing (> inline_auto_bytes → Box<T>).
    pub(crate) type_sizes: std::collections::HashMap<String, usize>,
    /// Structs that have at least one explicitly-qualified field ('actor, 'shared, 'guard, 'owned).
    /// These are eligible for T? parameter qualifier inference.
    pub(crate) qualified_struct_types: std::collections::HashSet<String>,
    /// Types T for which at least one function declares return type `T'actor` or `T'guard`.
    /// Bare `T` parameters of these types default to 'actor during qualifier inference instead
    /// of falling back to 'inline, enabling automatic propagation without explicit annotation.
    pub(crate) actor_source_types: std::collections::HashSet<String>,
    /// All user-defined struct names, including those whose size cannot be estimated (dynamic fields).
    /// Used to determine qualifier inference eligibility, separate from `type_sizes` which only
    /// stores structs with known sizes (for boxing/stack-size decisions).
    pub(crate) all_struct_types: std::collections::HashSet<String>,
    /// All user-defined enum names. Mirrors `all_struct_types` — used by `is_known_user_type`
    /// so method-call/field-access dispatch treats a receiver of enum type the same as one of
    /// struct type ("does this type have its own declared member?") instead of falling through
    /// to the builtin Iterator/collection name tables (`map_method`/`map_field`), which used to
    /// fire unconditionally for any enum receiver since only `struct_fields` was consulted.
    pub(crate) all_enum_types: std::collections::HashSet<String>,
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
    /// Parameters that are auto-ref ('shared/'actor/'guard/'weak) — emitted as &Arc/&Weak.
    /// When assigned to an owned context, the transpiler inserts Arc::clone automatically.
    pub(crate) auto_ref_params: std::collections::HashSet<String>,
    /// Per-function rebindable flags: fn_name → [is_rebindable per param index].
    /// A rebindable (var) param receives `&mut Wrapper<T>` at the call site.
    pub(crate) fn_rebindable: std::collections::HashMap<String, Vec<bool>>,
    /// Per-function mutable flags: fn_name → [is_mutable per param index].
    /// A mutable (mut) param is non-rebindable but mutable — caller must pass a mut/var binding.
    pub(crate) fn_mutable: std::collections::HashMap<String, Vec<bool>>,
    /// Variadic param index per function: fn_name → index of the `...` param.
    pub(crate) fn_variadic: std::collections::HashMap<String, usize>,
    /// Inside a `try:` body closure — calls to throws functions get `?`.
    pub(crate) in_try_body: bool,
    /// True while emitting the body of a `catch TypeName:` clause where `error` has
    /// already been bound as a concrete `&TypeName` enum reference (see emit_flow.rs's
    /// typed-catch dispatch). Tells emit_match's `match error:` special case to skip
    /// its BoringError re-downcast rewrite — `error` is not a `Box<dyn Error>` here,
    /// it's already the concrete enum, so a plain match on it is correct as-is.
    pub(crate) error_var_is_concrete_enum: bool,
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
    /// "EnumName::field_name" → present for enum variant field accessors that return `Option<T>`.
    pub(crate) enum_field_getters: std::collections::HashSet<String>,
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
    /// Variable names declared as `T'actor` — hold `Arc<std::sync::Mutex<T>>` (sync).
    pub(crate) var_mutex_types: std::collections::HashSet<String>,
    /// Variable names declared as `T'task` / `T'actor'task` — hold `Arc<tokio::sync::Mutex<T>>` (async).
    pub(crate) var_mutex_task_types: std::collections::HashSet<String>,
    /// "StructName::field_name" for fields typed `T'actor` (Arc<std::sync::Mutex<T>> in Rust).
    pub(crate) struct_mutex_fields: std::collections::HashSet<String>,
    /// "StructName::field_name" for fields typed `T'task` (Arc<tokio::sync::Mutex<T>> in Rust).
    pub(crate) struct_mutex_task_fields: std::collections::HashSet<String>,
    /// Variable names declared as `T'guard` — hold `Arc<std::sync::RwLock<T>>` (sync).
    pub(crate) var_rwlock_types: std::collections::HashSet<String>,
    /// Variable names declared as `T'guard'task` — hold `Arc<tokio::sync::RwLock<T>>` (async).
    pub(crate) var_rwlock_task_types: std::collections::HashSet<String>,
    /// "StructName::field_name" for fields typed `T'guard` (Arc<std::sync::RwLock<T>> in Rust).
    pub(crate) struct_rwlock_fields: std::collections::HashSet<String>,
    /// "StructName::field_name" for fields typed `T'guard'task` (Arc<tokio::sync::RwLock<T>> in Rust).
    pub(crate) struct_rwlock_task_fields: std::collections::HashSet<String>,
    /// "StructName::method_name" for methods that are non-mutating (`req`).
    /// Used by 'guard dispatch to choose `.read()` vs `.write()`.
    pub(crate) struct_req_methods: std::collections::HashSet<String>,
    /// "StructName::method_name" for methods declared `task`.
    /// Used by qualifier inference to disambiguate 'actor'task/'guard'task from
    /// 'actor/'guard when a task-captured variable has a task method called on it.
    pub(crate) struct_task_methods: std::collections::HashSet<String>,
    /// Free function name -> which positional params are declared `var` (out-parameter).
    /// Signature-only, collected once up front from the whole program — used by the
    /// `with` block mutation scan (see ast::with_block_mutates and docs/scoped-access-blocks.md)
    /// to decide whether passing the with-subject into a call grants write access.
    pub(crate) fn_var_params: std::collections::HashMap<String, Vec<bool>>,
    /// Names currently open in an enclosing `with` block. `with c:` shadows `c` with
    /// its already-acquired guard (`let mut c = c.lock().unwrap();` or the RwLock/GPU
    /// equivalent) once at block entry, so ordinary method/field codegen on `c` inside
    /// the block — which normally re-locks per access via `var_mutex_types` etc. — must
    /// be suppressed for the block's duration and fall through to plain-struct-receiver
    /// codegen instead; that's correct because the shadowed binding auto-derefs through
    /// the guard. See emit_stmt.rs's `Stmt::With` arm.
    pub(crate) with_open_names: std::collections::HashSet<String>,
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
    /// True when the function's declared return type is void (ignoring throws wrapper).
    /// Unlike fn_returns_void, this is true for `def void f() throws:` as well.
    pub(crate) fn_declared_void: bool,
    /// When true, suppress `Ok(...)` wrapping on the last expression even if `in_throws` is set.
    /// Used for if/match expression branches that need `?` propagation but not `Ok()` wrapping.
    pub(crate) suppress_ok_wrap: bool,
    /// trait_name → set of method names declared in that trait (signatures + defaults).
    /// Used to split struct body methods between `impl Trait for Struct {}` and `impl Struct {}`.
    pub(crate) trait_method_names: std::collections::HashMap<String, std::collections::HashSet<String>>,
    /// trait_name → (default method name → `mutating`). A struct that conforms to
    /// a trait but doesn't override a default method dispatches through the
    /// *trait's* Rust default impl directly (no per-struct copy — unlike the
    /// interpreter, which clones trait defaults into each struct's method list
    /// at registration; see `Interpreter`'s `Item::Struct` handling) — so
    /// `struct_req_methods`/`struct_task_methods` (populated only from each
    /// struct's own body) never gets an entry for an un-overridden default.
    /// This is the fallback the local-binding content-mutation diagnostic in
    /// `emit_methods.rs` consults via `struct_protocols` before concluding a
    /// call is unaccounted-for (and thus assumed mutating).
    pub(crate) trait_default_mutating: std::collections::HashMap<String, std::collections::HashMap<String, bool>>,
    /// struct_name → trait names it conforms to (`struct S as Trait1, Trait2:`).
    /// Paired with `trait_default_mutating` — see its doc.
    pub(crate) struct_protocols: std::collections::HashMap<String, Vec<String>>,
    /// Types reachable via a user-defined `as T:` conversion (lowercased).
    /// Used in emit_expr to route `x as T` to the generated `into_t()` method.
    pub(crate) user_conv_targets: std::collections::HashSet<String>,
    /// Local variables declared as mutable strings (`var x = ""`) — typed Arc<str>.
    /// Used to emit Arc::make_mut(&mut x) for read_line args and clear() calls.
    pub(crate) string_arc_vars: std::collections::HashSet<String>,
    /// All local variables known to hold `Arc<str>` (string type), including params.
    /// Used for string concatenation detection: `x + y` where both are strings.
    pub(crate) string_vars: std::collections::HashSet<String>,
    /// String variables/params (scoped to the current function/method body) that get
    /// indexed with a non-constant `name[idx]` somewhere in that body — see
    /// `collect_str_index_targets`. For names in this set that are also `let`-bound or
    /// non-`var` parameters (so the checker guarantees they're never reassigned), a
    /// `__strchars_<name>: Vec<char>` shadow is materialized once at the binding site,
    /// and single-index access reads from it (O(1)) instead of `.chars().nth(idx)`
    /// (O(idx), turning any sequential scan into O(n^2)).
    pub(crate) str_index_cache_vars: std::collections::HashSet<String>,
    /// Type parameters already declared at the `impl<...>` level (set by emit_ext).
    /// Methods inside a generic impl must NOT re-declare these params on their own `fn<...>`.
    pub(crate) impl_type_params: Vec<String>,
    /// Declared return type of the current function, if known.
    /// Used to coerce last-expression returns with `Some()` when the return type is Optional.
    pub(crate) fn_return_ty: Option<Type>,
    /// Parameters of the current function (name → declared type).
    /// Set before emit_body; used by the qualifier inference pass for annotation hints
    /// and body-compatibility checks on union-qualified parameters.
    pub(crate) fn_current_params: std::collections::HashMap<String, Type>,
    /// Source line of each parameter in the current function (name → line).
    pub(crate) fn_current_param_lines: std::collections::HashMap<String, usize>,
    pub(crate) fn_current_param_cols: std::collections::HashMap<String, usize>,
    /// Names of parameters declared as `mut` in the current function.
    /// Used by qualifier inference to determine auto-ref mutability and to detect
    /// def calls on immutable parameters.
    pub(crate) fn_current_params_mut: std::collections::HashSet<String>,
    /// Local variables and parameters that are immutable (`let` binding, or plain param without
    /// `mut`/`var`). Used to reject passing an immutable variable to a `mut` or `var` parameter.
    pub(crate) immutable_local_vars: std::collections::HashSet<String>,
    /// Local variables and parameters that are mutable but non-rebindable (`mut` binding, or
    /// `mut` param). Used to reject passing a `mut` binding to a `var` out-parameter.
    pub(crate) mut_local_vars: std::collections::HashSet<String>,
    /// Local variables whose content may be mutated — `def` calls, field writes
    /// (docs/mut-type-modifier.md). Independent of `mut_local_vars`/
    /// `immutable_local_vars` (the *rebind* axis): a plain `var Point p` is
    /// rebindable but NOT in this set; only `var mut`/bare-or-`let mut` is.
    /// Params are always included here (the parameter model's own `mut`/`var`
    /// split isn't enforced yet — see docs/mut-type-modifier.md's "Parameters"
    /// section — so both still grant full content access, unchanged).
    pub(crate) content_mutable_local_vars: std::collections::HashSet<String>,
    /// Names bound by an actual `let_stmt`/destructure/parameter — i.e. the
    /// binding forms docs/mut-type-modifier.md actually covers. `known_local_vars`
    /// also holds names from `if let`/`while let`/`match` patterns/`for` loop
    /// vars/etc., which this document does NOT touch (out of scope) and which
    /// keep their historical, permissive content-mutation behavior. The
    /// content-mutation diagnostics in `emit_methods.rs`/`emit_expr.rs` consult
    /// THIS set (not `known_local_vars`) before concluding a name is even
    /// eligible to be flagged — a name outside it is never rejected, no matter
    /// what `content_mutable_local_vars` says.
    pub(crate) mut_checked_local_vars: std::collections::HashSet<String>,
    /// Names declared as `type Name as InnerType` newtype wrappers.
    /// Used in emit_constructor to emit `Name(val)` (tuple struct) rather than `Name { field: val }`.
    pub(crate) newtype_types: std::collections::HashSet<String>,
    /// Maps newtype name → inner Rust type string (e.g. "UserId" → "u64").
    /// Used in emit_cast to emit `val.0` when unwrapping via `x as InnerType`.
    pub(crate) newtype_inner: std::collections::HashMap<String, String>,
    /// Maps local variable name → its newtype type name.
    /// Populated by emit_let and emit_fn (params) to enable `id as uint` → `id.0`.
    pub(crate) var_newtype_type: std::collections::HashMap<String, String>,
    /// Top-level functions that declare `stream` (async generator, needs tokio/async-stream).
    pub(crate) stream_fns: std::collections::HashSet<String>,
    /// Subset of stream functions that are purely sequential — emit `impl Iterator` instead.
    pub(crate) stream_iter_fns: std::collections::HashSet<String>,
    /// Stream functions that also declare `throws` — use `try_stream!` and unwrap at consumer.
    pub(crate) stream_throws_fns: std::collections::HashSet<String>,
    /// True when the file contains at least one async stream function (adds async-stream deps).
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
    /// Local variable name → its `let`/`var` initializer expression, recorded regardless of
    /// whether `var_types` itself got an entry (e.g. an unannotated `let rad = a * b` binop
    /// initializer isn't tracked into `var_types` at all). Used by `infer_float_width` as a
    /// fallback so a builtin math call on such a variable (`sin(rad)`) can still recover its
    /// float32/float64 width from the initializer instead of silently defaulting to f64.
    pub(crate) var_init_exprs: std::collections::HashMap<String, Expr>,
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
    /// Top-level *immutable* `let` bindings (scalar-const-safe per
    /// `top_level_let_is_const_safe`) that are referenced from inside a function body OR
    /// a struct/enum/ext method body. Such a `let` cannot stay a local inside the
    /// synthesized `fn main()` — the referencing function/method has no way to see it —
    /// so `emit_program_items` promotes it to a module-level `const` instead (mirroring
    /// what GPU targets already do unconditionally for every scalar-safe top-level `let`,
    /// see the `is_gpu_target` branch there). Gated on actual cross-scope usage (unlike the
    /// GPU case) so ordinary top-level scripting `let`s — including same-named shadowing
    /// re-declarations — keep behaving as sequential locals in `main()`.
    pub(crate) global_lets_used_elsewhere: std::collections::HashSet<String>,
    /// Names of top-level `let` string-literal constants that got promoted to a module-level
    /// `static ... LazyLock<Rc/Arc<str>>` (per `top_level_let_is_string_literal`, gated the
    /// same way as `global_lets_used_elsewhere`: `is_pub` unconditionally, or actual
    /// cross-scope usage otherwise — see `emit_program_items`'s `Item::Let` classification).
    /// Re-seeded into `string_vars`/`string_arc_vars` at the start of every function/method
    /// body (`emit_fn`) so a read of the constant there is recognized as a string value and
    /// gets `.clone()`d in owned position (e.g. as a by-value call argument) — without this,
    /// `emit_expr_owned`'s `Var` arm has no way to know a bare top-level identifier like
    /// `OP_ADD` is a string at all, since (unlike function params) it's never locally
    /// declared inside the function reading it.
    pub(crate) global_string_const_names: std::collections::HashSet<String>,
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
    /// Directory of the root source file — used to resolve relative `use` paths.
    pub(crate) source_dir: std::path::PathBuf,
    /// Named cross-project dependencies (copied from `TranspileConfig::deps` — see that
    /// field's doc comment). Consulted by `emit_top.rs`'s `emit_use`/`deep_pre_scan` before
    /// falling back to `source_dir`-relative resolution.
    pub(crate) deps: std::collections::HashMap<String, std::path::PathBuf>,
    /// Canonical paths already inlined — prevents duplicate / circular imports.
    pub(crate) loaded: std::collections::HashSet<std::path::PathBuf>,
    /// Per-file Rust output collected from inlined `.br` use imports.
    /// Each entry is (module_name, rust_code). Written as separate .rs files at build time.
    pub(crate) modules: Vec<(String, String)>,
    /// True once the standard prelude has been emitted — prevents re-emission for inlined files.
    pub(crate) prelude_emitted: bool,
    /// Function signatures already emitted — shared across all inlined files to deduplicate.
    pub(crate) emitted_fn_sigs: std::collections::HashSet<String>,
    /// True when the program calls any of the log-level builtins (error/warn/info/debug/trace).
    /// The CLI uses this to warn that `log = "0.4"` is needed in Cargo.toml.
    /// Uses Rc<Cell<bool>> so sub-transpilers share the same instance — any set(true) in a
    /// sub (e.g. inside a try: block) is immediately visible in the parent.
    pub(crate) uses_log: std::rc::Rc<std::cell::Cell<bool>>,
    /// True when an enum with @error("...") variants is emitted (thiserror auto-derive).
    /// The CLI uses this to add `thiserror = "1"` to Cargo.toml.
    pub(crate) uses_thiserror: std::rc::Rc<std::cell::Cell<bool>>,
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
    pub(crate) uses_tokio_util: std::rc::Rc<std::cell::Cell<bool>>,
    /// Set when `json()` or `fromJson()` is used — triggers serde/serde_json deps.
    pub(crate) uses_serde: std::rc::Rc<std::cell::Cell<bool>>,
    /// Functions that have multiple overloads — maps name to all FnDecl variants.
    pub(crate) fn_overload_decls: std::collections::HashMap<String, Vec<crate::ast::FnDecl>>,
    /// Names of overloaded functions (quick lookup).
    pub(crate) overloaded_fn_names: std::collections::HashSet<String>,
    /// Per-struct overloaded method declarations: "TypeName::method" → Vec<FnDecl>
    pub(crate) struct_method_overload_decls: std::collections::HashMap<String, Vec<crate::ast::FnDecl>>,
    /// Keys of overloaded struct methods (quick lookup): "TypeName::method"
    pub(crate) overloaded_method_keys: std::collections::HashSet<String>,
    /// Type names (structs/enums) that appear as the inner type of a `'shared`, `'actor`, or
    /// `'guard` qualifier in at least one struct field or function parameter across the program.
    /// `task fn` methods are only valid on types in this set — they rely on the receiver being
    /// accessed through `Arc<Self>`, which is only guaranteed when the type is arc-qualified.
    pub(crate) arc_qualified_types: std::collections::HashSet<String>,
    /// Enum names where every variant has no fields — inferred as T'copy (non-parametric enums).
    pub(crate) unit_enums: std::collections::HashSet<String>,
    /// "StructName::field_name" and "EnumName::VariantName::N" pairs that need Box wrapping
    /// because the field type directly or indirectly refers back to the containing type (recursion).
    pub(crate) recursive_fields: std::collections::HashSet<String>,
    /// Names of all user-defined structs and enums in the program.
    /// Used in managed mode to wrap anonymous `T`/`T'` types in Arc<Mutex<T>> or RefCell<T>.
    pub(crate) user_types: std::collections::HashSet<String>,
    /// True when the program uses channel primitives that require local_channel in single mode.
    pub(crate) uses_local_channel: std::rc::Rc<std::cell::Cell<bool>>,
    /// True when broadcast is used in single-thread mode — triggers inline prelude.
    pub(crate) uses_local_broadcast: std::rc::Rc<std::cell::Cell<bool>>,
    /// Variables declared as `T'shared` in single-thread mode — hold `Rc<T>` instead of `Arc<T>`.
    /// These must be pre-cloned with `Rc::clone` (not `Arc::clone`) before async-move captures.
    pub(crate) rc_vars: std::collections::HashSet<String>,
    /// Parameters of type `T'shared` or `T'actor` passed as `&Rc<T>` / `&Arc<T>`.
    /// Matching on these requires `(**var)` — two dereferences — to reach `T`.
    pub(crate) shared_ref_params: std::collections::HashSet<String>,
    /// `var` parameters of primitive/stack type — emitted as `&mut T`, so usages are auto-derefed.
    pub(crate) var_primitive_params: std::collections::HashSet<String>,
    /// Variables holding `Arc<std::sync::Mutex<T>>` in managed multi mode (anonymous T/T').
    /// Field reads, method calls, and optional chaining go through `.lock().unwrap()`.
    pub(crate) managed_mutex_vars: std::collections::HashSet<String>,
    /// Subset of managed_mutex_vars that were obtained from a function return value.
    /// These are fresh, unshared Arc values — no pre-lock guard is emitted for them
    /// (they may be moved, and deadlock is impossible for a freshly-created Arc).
    pub(crate) managed_mutex_fn_return_vars: std::collections::HashSet<String>,
    /// Variables holding `RefCell<T>` in managed single mode (anonymous T/T').
    /// Field reads go through `.borrow()`, method calls through `.borrow_mut()`.
    pub(crate) managed_refcell_vars: std::collections::HashSet<String>,
    /// Managed mutex PARAMETERS shadowed by a guard let-binding at function entry.
    /// Maps original param name → shadow guard name (e.g. `rhs` → `__rhs_mg`).
    /// Used to avoid double-lock deadlock when the same param's fields are accessed
    /// multiple times in a single expression (std::sync::Mutex is not reentrant).
    pub(crate) managed_param_shadows: std::collections::HashMap<String, String>,
    /// "TypeName::method_name" → return Type for all ext methods.
    /// Used in emit_let to infer whether an untyped binding should be tracked as managed.
    pub(crate) struct_method_return_types: std::collections::HashMap<String, crate::ast::Type>,
    /// Method names (bare, without struct prefix) that are declared as `throws`.
    /// Used to add `?` propagation when calling `self.method()` or `obj.method()` inside throws context.
    pub(crate) struct_method_throws: std::collections::HashSet<String>,
    /// Use-site qualifier inference (priority 5).
    /// Maps local variable name → inferred OwnerQual, populated by a pre-pass over each
    /// function body before emission. Cleared between function bodies.
    pub(crate) inferred_qualifiers: std::collections::HashMap<String, crate::ast::OwnerQual>,
    /// Temporary: local variables in the current function body that are assigned from a
    /// call whose declared return type is `'actor` or `'guard`. Populated by a pre-pass
    /// in `infer_qualifiers` and consumed by `walk_expr_for_qualifiers`. Cleared each call.
    pub(crate) infer_local_actor_vars: std::collections::HashSet<String>,
    /// Local variables (task/closure captures) on which a `task`-declared method was called
    /// during the current function's qualifier-inference pass. When both the plain and
    /// `'task` variant of 'actor/'guard remain candidates, presence in this set picks the
    /// `'task` variant. Cleared each call to `infer_qualifiers`.
    pub(crate) task_method_call_vars: std::collections::HashSet<String>,
    /// Struct field names (bare, unqualified within `infer_struct_field_qualifiers`'s scope)
    /// on which a `task`-declared method was called during struct field qualifier inference.
    /// Same purpose as `task_method_call_vars` but for `self.field` captures. Cleared per struct.
    pub(crate) task_method_call_fields: std::collections::HashSet<String>,
    /// Local variables declared with `lazy` — hold `OnceCell<T>`.
    /// `?=` on these emits `name.get_or_init(|| rhs)`.
    /// Reads of these variables emit `*name.get().expect("name used before lazy init")`.
    pub(crate) lazy_vars: std::collections::HashSet<String>,
    /// `lazy` variable name → declared boring type (for Copy detection on read).
    pub(crate) lazy_var_types: std::collections::HashMap<String, crate::ast::Type>,
    /// Struct type names that declare an anonymous `def ()` or `req ()` call operator.
    /// When `var(args)` is called and `var` resolves to one of these types, emit `var.__call__(args)`.
    pub(crate) callable_structs: std::collections::HashSet<String>,
    /// `kernel Name: ...` declarations, by name, passed in via `TranspileConfig::gpu_kernels`.
    /// Empty (the default) for every non-GPU-kernel-aware target — all of the special-case
    /// kernel codegen below is gated on this being non-empty, so behavior for those targets
    /// is completely unchanged.
    pub(crate) kernel_decls: std::collections::HashMap<String, crate::ast::KernelDecl>,
    /// Local variable name -> kernel type name, for `let`/`mut`/`var` bindings whose
    /// initializer is a call to one of `kernel_decls` (e.g. `mut k = StftPower(...)`).
    /// Populated by `emit_let`; consumed by kernel-construction, `kernel:` dispatch, and
    /// kernel 'unified-field-read codegen (GPU targets only -- see `kernel_decls`).
    pub(crate) kernel_vars: std::collections::HashMap<String, String>,
    /// Local variable name -> (kernel variable name, field name), for a `'gpu'unified`/
    /// `'gpu'global`-qualified `let`/`var` initialized directly from a bare kernel-field
    /// read (`let py'gpu'unified = k.y`) — see `emit_kernel::try_emit_gpu_resident_let`.
    /// Such a binding is a pure compile-time alias: no Rust variable is ever emitted for
    /// it, so its only legal use is as the subject of a `with` block, which resolves it
    /// back to `copy_{field}_to_host`/`copy_{field}_to_device` on the kernel variable
    /// (see `emit_stmt::emit_with`). The checker enforces this is the only legal use
    /// (`Binding::resident_from_field`). GPU targets only, same gate as `kernel_vars`.
    pub(crate) gpu_resident_vars: std::collections::HashMap<String, (String, String)>,
    /// Free function name -> its declared return type, for functions whose return
    /// type is `'gpu'unified`/`'gpu'global`-qualified — mirrors the checker's own
    /// `fn_returns_resident` (checker/mod.rs). Populated once up front (`pre_scan`).
    /// Drives: (1) `emit_top.rs`'s return-type emission (`BoringGpuArg<T>` instead of
    /// `Vec<T>`), (2) `emit_kernel::try_emit_gpu_resident_call_let` recognizing
    /// `let fc = some_fn(...)` as an interprocedural resident binding. See
    /// docs/scoped-access-blocks.md's interprocedural residency case.
    pub(crate) fn_returns_resident: std::collections::HashMap<String, Type>,
    /// Free function name -> per-position flags: `true` when that parameter is used
    /// *exclusively*, everywhere in the function's body, as a bare argument to a
    /// kernel constructor at a `'unified`/`'global` field position (mirrors the
    /// checker's own `fn_gpu_arg_params`, computed independently here from the same
    /// bounded scan, `ast::scan_var_call_arg_uses`). Such a parameter is emitted
    /// `BoringGpuArg<T>` instead of a plain host array (`emit_top.rs`), and the
    /// kernel-construction codegen that consumes it branches on the enum instead of
    /// always uploading (`emit_kernel::emit_kernel_construction`).
    pub(crate) fn_gpu_arg_params: std::collections::HashMap<String, Vec<bool>>,
    /// Free function name -> per-position flags, for a function whose return type is
    /// a `Type::Tuple` with at least one `'gpu'unified`/`'gpu'global`-qualified
    /// element (mirrors the checker's own `fn_returns_resident_tuple`). The tuple
    /// analogue of `fn_returns_resident`: `mha_step_gpu`-style `([float]'gpu'unified,
    /// [float], [float])` returns, chaining the tail tuple literal's resident
    /// elements instead of eagerly downloading them. See
    /// `emit_kernel::try_emit_gpu_resident_tuple_return`.
    pub(crate) fn_returns_resident_tuple: std::collections::HashMap<String, Vec<bool>>,
    /// Local variable name -> its declared/inferred resident `Type`, for a `let`/
    /// `var` bound directly to a call to a `fn_returns_resident` function (`let fc =
    /// linear_gpu(...)`) — the *interprocedural* counterpart to `gpu_resident_vars`.
    /// Unlike that map (a pure compile-time alias with no Rust binding, since the
    /// data is just a re-readable kernel field), this call already executed with
    /// real side effects, so `fc` gets a real `let` binding, just typed
    /// `BoringGpuArg<T>` — see `emit_kernel::try_emit_gpu_resident_call_let` and
    /// `emit_stmt::emit_with`'s matching materialization branch.
    pub(crate) resident_call_vars: std::collections::HashMap<String, Type>,
    /// Scoped to the function currently being emitted (saved/restored around the
    /// body in `emit_top.rs::emit_fn`, same convention as `fn_current_params` etc.):
    /// `Some(return type)` when this function's declared return is GPU-resident, so
    /// the tail-expression emitter can branch to `BoringGpuArg::Resident(...)`
    /// instead of the unconditional `copy_{field}_to_host()` download.
    pub(crate) current_fn_returns_resident: Option<Type>,
    /// Scoped to the function currently being emitted, same convention as
    /// `current_fn_returns_resident`: `Some(per-position flags)` copied from
    /// `fn_returns_resident_tuple` when this function's declared return is a
    /// resident tuple, so the tail-expression emitter can recognize a tuple-literal
    /// tail and chain its resident positions instead of materializing them.
    pub(crate) current_fn_returns_resident_tuple: Option<Vec<bool>>,
    /// Scoped to the function currently being emitted, same convention as
    /// `current_fn_returns_resident`: per-position flags copied from
    /// `fn_gpu_arg_params` for whichever function is being emitted right now, so
    /// `emit_kernel_construction`'s buffer-argument branch knows which of *this*
    /// function's own parameters to treat as `BoringGpuArg<T>`.
    pub(crate) current_fn_gpu_arg_params: Vec<bool>,
    /// Same information as `current_fn_gpu_arg_params`, but by parameter *name*
    /// rather than position — the convenient form for consumption sites that only
    /// have a bare `Var(name)` in hand (`emit_kernel_construction`'s buffer-argument
    /// branch, `emit_call`'s argument-wrapping), which don't have the enclosing
    /// `FnDecl`'s param list available to translate a position back to a name.
    pub(crate) current_fn_gpu_arg_param_names: std::collections::HashSet<String>,
    /// Local variable names bound to a `GPU(n)` device handle (`let g = GPU(0)`), or a
    /// `for` loop variable iterating `GPU.all()` — tracked so a later `g.name()`/
    /// `.totalMem()`/etc. method call can be rewritten to the introspection helpers
    /// emitted by `wgpu::host::emit_gpu_introspection_globals`. wgpu only ever has one
    /// real adapter (see that function's doc comment), so every `GPU(n)` and every
    /// element of `GPU.all()` resolves to it regardless of index — matching the
    /// interpreter's own single mock `GpuDevice` in simulation mode. GPU targets only.
    pub(crate) gpu_device_vars: std::collections::HashSet<String>,
    /// Names of every top-level `let` (across the whole reachable `use` graph, collected
    /// by `pre_scan`/`deep_pre_scan` before the prelude is emitted). Used to skip the
    /// prelude's `use std::f64::consts::{PI, E, TAU}` for any of those three names a
    /// program defines itself — otherwise a user's own top-level `let float PI = ...`
    /// collides with the auto-imported one (E0255).
    pub(crate) user_top_level_names: std::collections::HashSet<String>,
    /// Mirrors `TranspileConfig::is_gpu_target` for quick access. See that field's doc
    /// comment — this is NOT the same as `!kernel_decls.is_empty()`.
    pub(crate) is_gpu_target: bool,
    /// Set when `emit_program_items` synthesizes a `fn boring_main()` wrapper for
    /// GPU-target top-level statements/non-const `let`s (see `emit_program_items`).
    /// Reported back via `TranspileOutput::gpu_main_emitted` so
    /// `transpiler::wgpu::transpile_wgpu` knows whether to call `boring_main()` from
    /// its own generated entry point, without re-deriving it from AST inspection
    /// (which can't see a function that exists only in the emitted text).
    pub(crate) gpu_main_emitted: std::cell::Cell<bool>,
    /// Original (lowercase, boring-source) names of GPU-target top-level scalar `let`s
    /// promoted to a Rust `const` -- the actual emitted Rust identifier is uppercased
    /// (see `emit_item`'s `Item::Let` case) to avoid colliding with a same-named local
    /// variable or function parameter elsewhere in the file. Consumed by
    /// `map_builtin_var` to rewrite reads of these names to the uppercased const.
    pub(crate) gpu_top_level_const_names: std::collections::HashSet<String>,
}

impl Transpiler {
    fn new(config: TranspileConfig) -> Self {
        let kernel_decls = config.gpu_kernels.iter()
            .map(|k| (k.name.clone(), k.clone()))
            .collect();
        let is_gpu_target = config.is_gpu_target;
        Transpiler {
            config,
            out: String::new(),
            errors: std::cell::RefCell::new(Vec::new()),
            warnings: std::cell::RefCell::new(Vec::new()),
            indent: 0,
            in_throws: false,
            in_lhs_assign: std::cell::Cell::new(false),
            want_raw_option_pop: std::cell::Cell::new(false),
            in_req_fn: false,
            in_struct_method: false,
            in_async: false,
            in_iter_stream: false,
            self_type: None,
            collection_vars: std::collections::HashSet::new(),
            vec_vars: std::collections::HashSet::new(),
            str_vec_vars: std::collections::HashSet::new(),
            set_vars: std::collections::HashSet::new(),
            tuple_vars: std::collections::HashMap::new(),
            dict_vars: std::collections::HashSet::new(),
            instant_vars: std::collections::HashSet::new(),
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
            struct_field_defaults: std::collections::HashMap::new(),
            structs_needing_default: std::collections::HashSet::new(),
            type_sizes: std::collections::HashMap::new(),
            qualified_struct_types: std::collections::HashSet::new(),
            actor_source_types: std::collections::HashSet::new(),
            all_struct_types: std::collections::HashSet::new(),
            all_enum_types: std::collections::HashSet::new(),
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
            auto_ref_params: std::collections::HashSet::new(),
            fn_rebindable: std::collections::HashMap::new(),
            fn_mutable: std::collections::HashMap::new(),
            fn_variadic: std::collections::HashMap::new(),
            in_try_body: false,
            error_var_is_concrete_enum: false,
            in_type_setter: false,
            in_init_body: false,
            struct_type_var_names: std::collections::HashSet::new(),
            struct_type_mut_var_names: std::collections::HashSet::new(),
            struct_type_method_sigs: std::collections::HashMap::new(),
            struct_getters: std::collections::HashSet::new(),
            enum_field_getters: std::collections::HashSet::new(),
            struct_setters: std::collections::HashSet::new(),
            transient_fields: std::collections::HashMap::new(),
            var_struct_types: std::collections::HashMap::new(),
            var_mutex_types: std::collections::HashSet::new(),
            var_mutex_task_types: std::collections::HashSet::new(),
            struct_mutex_fields: std::collections::HashSet::new(),
            struct_mutex_task_fields: std::collections::HashSet::new(),
            var_rwlock_types: std::collections::HashSet::new(),
            var_rwlock_task_types: std::collections::HashSet::new(),
            struct_rwlock_fields: std::collections::HashSet::new(),
            struct_rwlock_task_fields: std::collections::HashSet::new(),
            struct_req_methods: std::collections::HashSet::new(),
            struct_task_methods: std::collections::HashSet::new(),
            fn_var_params: std::collections::HashMap::new(),
            with_open_names: std::collections::HashSet::new(),
            iterable_structs: std::collections::HashSet::new(),
            known_local_vars: std::collections::HashSet::new(),
            fn_returns_void: false,
            fn_declared_void: false,
            suppress_ok_wrap: false,
            trait_method_names: std::collections::HashMap::new(),
            trait_default_mutating: std::collections::HashMap::new(),
            struct_protocols: std::collections::HashMap::new(),
            user_conv_targets: std::collections::HashSet::new(),
            string_arc_vars: std::collections::HashSet::new(),
            string_vars: std::collections::HashSet::new(),
            str_index_cache_vars: std::collections::HashSet::new(),
            impl_type_params: Vec::new(),
            fn_return_ty: None,
            fn_current_params: std::collections::HashMap::new(),
            fn_current_param_lines: std::collections::HashMap::new(),
            fn_current_param_cols: std::collections::HashMap::new(),
            fn_current_params_mut: std::collections::HashSet::new(),
            immutable_local_vars: std::collections::HashSet::new(),
            mut_local_vars: std::collections::HashSet::new(),
            content_mutable_local_vars: std::collections::HashSet::new(),
            mut_checked_local_vars: std::collections::HashSet::new(),
            newtype_types: std::collections::HashSet::new(),
            newtype_inner: std::collections::HashMap::new(),
            var_newtype_type: std::collections::HashMap::new(),
            stream_fns: std::collections::HashSet::new(),
            stream_iter_fns: std::collections::HashSet::new(),
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
            var_init_exprs: std::collections::HashMap::new(),
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
            global_lets_used_elsewhere: std::collections::HashSet::new(),
            global_string_const_names: std::collections::HashSet::new(),
            optional_numeric_vars: std::collections::HashSet::new(),
            always_none_vars: std::collections::HashSet::new(),
            fn_type_aliases: std::collections::HashMap::new(),
            non_fn_type_aliases: std::collections::HashMap::new(),
            current_fn_type_params: std::collections::HashSet::new(),
            struct_ext_method_overrides: std::collections::HashSet::new(),
            rc_identity_vars: std::collections::HashSet::new(),
            boring_mod_names: std::collections::HashSet::new(),
            source_dir: std::path::PathBuf::new(),
            deps: std::collections::HashMap::new(),
            loaded: std::collections::HashSet::new(),
            modules: Vec::new(),
            prelude_emitted: false,
            emitted_fn_sigs: std::collections::HashSet::new(),
            uses_log: std::rc::Rc::new(std::cell::Cell::new(false)),
            uses_thiserror: std::rc::Rc::new(std::cell::Cell::new(false)),
            uses_reqwest: false,
            cancellable_task_fns: std::collections::HashSet::new(),
            cancel_token_vars: std::collections::HashMap::new(),
            in_cancellable_fn: false,
            uses_tokio_util: std::rc::Rc::new(std::cell::Cell::new(false)),
            uses_serde: std::rc::Rc::new(std::cell::Cell::new(false)),
            fn_overload_decls: std::collections::HashMap::new(),
            overloaded_fn_names: std::collections::HashSet::new(),
            struct_method_overload_decls: std::collections::HashMap::new(),
            overloaded_method_keys: std::collections::HashSet::new(),
            arc_qualified_types: std::collections::HashSet::new(),
            unit_enums: std::collections::HashSet::new(),
            recursive_fields: std::collections::HashSet::new(),
            user_types: std::collections::HashSet::new(),
            uses_local_channel: std::rc::Rc::new(std::cell::Cell::new(false)),
            uses_local_broadcast: std::rc::Rc::new(std::cell::Cell::new(false)),
            rc_vars: std::collections::HashSet::new(),
            shared_ref_params: std::collections::HashSet::new(),
            var_primitive_params: std::collections::HashSet::new(),
            managed_mutex_vars: std::collections::HashSet::new(),
            managed_mutex_fn_return_vars: std::collections::HashSet::new(),
            managed_refcell_vars: std::collections::HashSet::new(),
            managed_param_shadows: std::collections::HashMap::new(),
            struct_method_return_types: std::collections::HashMap::new(),
            struct_method_throws: std::collections::HashSet::new(),
            inferred_qualifiers: std::collections::HashMap::new(),
            infer_local_actor_vars: std::collections::HashSet::new(),
            task_method_call_vars: std::collections::HashSet::new(),
            task_method_call_fields: std::collections::HashSet::new(),
            lazy_vars: std::collections::HashSet::new(),
            lazy_var_types: std::collections::HashMap::new(),
            callable_structs: std::collections::HashSet::new(),
            kernel_decls,
            kernel_vars: std::collections::HashMap::new(),
            gpu_resident_vars: std::collections::HashMap::new(),
            fn_returns_resident: std::collections::HashMap::new(),
            fn_gpu_arg_params: std::collections::HashMap::new(),
            fn_returns_resident_tuple: std::collections::HashMap::new(),
            resident_call_vars: std::collections::HashMap::new(),
            current_fn_returns_resident: None,
            current_fn_returns_resident_tuple: None,
            current_fn_gpu_arg_params: Vec::new(),
            current_fn_gpu_arg_param_names: std::collections::HashSet::new(),
            gpu_device_vars: std::collections::HashSet::new(),
            user_top_level_names: std::collections::HashSet::new(),
            is_gpu_target,
            gpu_main_emitted: std::cell::Cell::new(false),
            gpu_top_level_const_names: std::collections::HashSet::new(),
        }
    }



    // ── Managed-mode helpers ──────────────────────────────────────────────────

    /// Returns true when `ty` is `T'owned` or `T'new` (`OwnerQual::is_owned_or_new`) in managed
    /// mode over a user-defined type that is NOT a unit enum. Suitable for field/method dispatch
    /// and param seeding.
    ///
    /// Must match `T'new` here, not just a committed `T'owned` — a raw, not-yet-inference-resolved
    /// `'new` (`Union([Owned, Shared, Actor, Guard])`) declared type reaches this check directly at
    /// some call sites (e.g. operator-overload RHS wrapping in `emit_expr.rs`, which looks at the
    /// method's raw declared param type, not its `inferred_qualifiers`-resolved one). Missing that
    /// shape here would silently emit `Box::new(...)` instead of the managed wrapper even though
    /// the function *signature* for that same parameter resolves through inference and correctly
    /// picks up managed-mode wrapping — see `emit_param`'s `needs_inference` handling of `Union`.
    pub(crate) fn is_managed_user_owned(config: &TranspileConfig,
        user_types: &std::collections::HashSet<String>,
        unit_enums: &std::collections::HashSet<String>,
        ty: &Type) -> bool
    {
        if config.mode != TranspileMode::Managed { return false; }
        match ty {
            Type::Qualified(inner, q) if q.is_owned_or_new() => match inner.as_ref() {
                Type::Named(n) => user_types.contains(n.as_str()) && !unit_enums.contains(n.as_str()),
                _ => false,
            },
            _ => false,
        }
    }

    // ── String type helpers ───────────────────────────────────────────────────

    /// Whether to use `Rc<str>` instead of `Arc<str>` for `string` — for passing to free functions.
    pub(crate) fn use_rc_str(&self) -> bool {
        matches!(self.config.threading, ThreadingMode::Single)
    }

    /// `"Rc"` in single-thread mode, `"Arc"` otherwise.
    pub(crate) fn str_ptr(&self) -> &'static str {
        if self.use_rc_str() { "Rc" } else { "Arc" }
    }

    /// `Rc::<str>::from("<s>")` or `Arc::<str>::from("<s>")` depending on threading mode.
    pub(crate) fn str_from(&self, s: &str) -> String {
        format!("{}::<str>::from(\"{}\")", self.str_ptr(), s)
    }

    /// `Rc::<str>::from(<expr>)` or `Arc::<str>::from(<expr>)` for a non-literal expression.
    pub(crate) fn str_from_expr(&self, expr: &str) -> String {
        format!("{}::<str>::from({})", self.str_ptr(), expr)
    }

    // ── Inference helpers ─────────────────────────────────────────────────────

    fn field_type_has_qualifier(ty: &Type) -> bool {
        match ty {
            Type::Qualified(_, q) => !(q.is_owned_or_new() || matches!(q, crate::ast::OwnerQual::Inline)),
            Type::Optional(inner) => Self::field_type_has_qualifier(inner),
            _ => false,
        }
    }

    /// Estimate the stack size in bytes of a type (best-effort, concrete types only).
    /// Returns None for generic parameters, arrays, or types with unknown inner sizes.
    fn estimate_size(ty: &Type, program: &Program) -> Option<usize> {
        Self::estimate_size_inner(ty, program, &mut std::collections::HashSet::new())
    }

    fn estimate_size_inner(ty: &Type, program: &Program, visiting: &mut std::collections::HashSet<String>) -> Option<usize> {
        match ty {
            Type::Int | Type::Uint | Type::Float64 => Some(8),
            Type::Float32 => Some(4),
            Type::Uint8 | Type::Int8 => Some(1),
            Type::Int16 | Type::Uint16 => Some(2),
            Type::Int32 | Type::Uint32 => Some(4),
            Type::Int64 | Type::Uint64 => Some(8),
            Type::Int128 | Type::Uint128 => Some(16),
            Type::Bool => Some(1),
            Type::Str => Some(16), // Arc<str> = 2 pointers
            Type::Nil | Type::Void => Some(0),
            // Primitive type names that may appear as Named() from init-param parsing
            Type::Named(n) if matches!(n.as_str(), "int" | "uint" | "isize" | "usize" | "i64" | "u64") => Some(8),
            // "f32" used to fall through to the catch-all below with no entry at all
            // (a pre-existing gap) — fixed alongside adding float32/float64 as real
            // types (docs/float-width-types.md).
            Type::Named(n) if matches!(n.as_str(), "float32" | "f32") => Some(4),
            Type::Named(n) if matches!(n.as_str(), "float" | "float64" | "f64") => Some(8),
            Type::Named(n) if matches!(n.as_str(), "uint8" | "int8" | "u8" | "i8") => Some(1),
            Type::Named(n) if matches!(n.as_str(), "int16" | "uint16" | "i16" | "u16") => Some(2),
            Type::Named(n) if matches!(n.as_str(), "int32" | "uint32" | "i32" | "u32") => Some(4),
            Type::Named(n) if matches!(n.as_str(), "int64" | "uint64") => Some(8),
            Type::Named(n) if matches!(n.as_str(), "int128" | "uint128" | "i128" | "u128") => Some(16),
            Type::Named(n) if n.as_str() == "bool" => Some(1),
            Type::Named(n) if matches!(n.as_str(), "str" | "string") => Some(16),
            Type::Qualified(inner, OwnerQual::Inline) => Self::estimate_size_inner(inner, program, visiting),
            // Pointer-sized types (Box, Arc, Rc, Option<Box>) — always 8 or 16 bytes
            Type::Qualified(_, OwnerQual::Owned | OwnerQual::Shared | OwnerQual::Actor | OwnerQual::Guard) => Some(16),
            Type::Optional(inner) => Self::estimate_size_inner(inner, program, visiting).map(|s| s + 8), // discriminant
            Type::Array(_) | Type::Dict(_, _) | Type::Set(_) => None, // dynamic
            Type::TypeParam(_) | Type::Generic(_, _) => None, // unknown
            Type::Named(name) => {
                // Cycle guard: a recursive type (Env → Env?) will be heap-allocated (Box/Rc),
                // so treat it as pointer-sized (16 bytes) when we detect a cycle.
                if visiting.contains(name.as_str()) {
                    return Some(16);
                }
                visiting.insert(name.clone());
                // Look up struct definition and sum field sizes.
                let result = (|| {
                    for item in &program.items {
                        if let Item::Struct(s) = item {
                            if &s.name == name {
                                // Collect field types: explicit fields first, then init params if no fields.
                                let mut field_types: Vec<Type> = s.fields.iter()
                                    .map(|f| f.ty.clone())
                                    .collect();
                                if field_types.is_empty() {
                                    for init in &s.inits {
                                        if init.body.is_empty() {
                                            for p in &init.params {
                                                if let Some(ty) = &p.ty { field_types.push(ty.clone()); }
                                            }
                                            break;
                                        }
                                    }
                                }
                                let mut total = 0usize;
                                for ty in &field_types {
                                    match Self::estimate_size_inner(ty, program, visiting) {
                                        Some(s) => total += s,
                                        None => return None,
                                    }
                                }
                                return Some(total);
                            }
                        }
                        if let Item::Enum(e) = item {
                            if &e.name == name {
                                // Enum size = max(variant sizes) + discriminant (8 bytes)
                                let mut max_variant = 0usize;
                                for v in &e.variants {
                                    let mut variant_size = 0usize;
                                    let mut ok = true;
                                    for f in &v.fields {
                                        match Self::estimate_size_inner(&f.ty, program, visiting) {
                                            Some(s) => variant_size += s,
                                            None => { ok = false; break; }
                                        }
                                    }
                                    if !ok { return None; }
                                    if variant_size > max_variant { max_variant = variant_size; }
                                }
                                return Some(max_variant + 8);
                            }
                        }
                    }
                    None
                })();
                visiting.remove(name.as_str());
                result
            }
            _ => None,
        }
    }

    /// Returns true if `ty` directly references a type named `name` (recursion check).
    /// Unwraps qualifiers and Optional/Array wrappers to find the inner named type.
    /// Also checks for indirect (transitive) cycles via `reachable` — a set of type names
    /// that eventually reach back to `name` through non-heap struct/enum fields.
    fn type_references_transitive(
        ty: &Type,
        name: &str,
        reachable: &std::collections::HashSet<String>,
    ) -> bool {
        match ty {
            Type::Named(n) => n == name || reachable.contains(n.as_str()),
            Type::Array(_) | Type::Dict(_, _) | Type::Set(_) | Type::Qualified(_, _) => false,
            Type::Optional(inner) | Type::Dyn(inner) | Type::Impl(inner) => {
                Self::type_references_transitive(inner, name, reachable)
            }
            Type::Generic(n, args) => {
                n == name || reachable.contains(n.as_str())
                    || args.iter().any(|a| Self::type_references_transitive(a, name, reachable))
            }
            _ => false,
        }
    }

    fn type_references(ty: &Type, name: &str) -> bool {
        match ty {
            Type::Named(n) => n == name,
            // Heap-allocated or pointer-sized containers: never cause infinite struct size.
            // Array/Dict/Set are Vec/HashMap/HashSet-backed (heap); Qualified types are pointer-sized
            // (Box<T>, Rc<T>, Arc<T>, Rc<RefCell<T>>, etc.) — do not follow.
            Type::Array(_) | Type::Dict(_, _) | Type::Set(_) | Type::Qualified(_, _) => false,
            Type::Optional(inner) | Type::Dyn(inner) | Type::Impl(inner) => {
                Self::type_references(inner, name)
            }
            Type::Generic(n, args) => {
                n == name || args.iter().any(|a| Self::type_references(a, name))
            }
            _ => false,
        }
    }

    // ── Output helpers ────────────────────────────────────────────────────────

    fn push_error(&self, line: usize, col: usize, msg: impl Into<String>) {
        self.errors.borrow_mut().push(TranspileError::at(msg, line, col));
    }

    fn push_warning(&self, line: usize, col: usize, msg: impl Into<String>) {
        self.warnings.borrow_mut().push(TranspileError::at(msg, line, col));
    }

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
        // Interprocedural GPU residency (docs/scoped-access-blocks.md), transitive
        // parameter case: needs the whole program's call graph in one shot for its
        // fixed point, so it runs once here rather than folding into `pre_scan`,
        // which recurses per-module-scope on a series of smaller pseudo-programs.
        self.compute_gpu_arg_params(program);
        // Pre-scan: collect enum variants and fn defaults before emitting anything.
        self.pre_scan(program);

        // Detect broadcast in single-thread mode early so the prelude can be emitted
        // before any function definitions.
        if matches!(self.config.threading, ThreadingMode::Single)
            && program_uses_broadcast(program)
        {
            self.uses_local_broadcast.set(true);
        }

        // Standard prelude — emitted once only (skipped for inlined `use` files).
        if self.prelude_emitted { return self.emit_program_items(program, false); }

        // Deep pre-scan: before emitting anything, recursively pre-scan all reachable `use` files.
        // This ensures fn_throws / fn_sigs / struct_fields are populated for all files before any
        // code is emitted, so forward references across file boundaries resolve correctly.
        {
            let mut visited = std::collections::HashSet::new();
            let prev_dir = self.source_dir.clone();
            self.deep_pre_scan(program, &mut visited);
            self.source_dir = prev_dir;
        }
        self.prelude_emitted = true;
        self.line("// Generated by boring build");
        self.line("use std::collections::{HashMap, HashSet};");
        self.line("use std::hash::Hash;");
        self.line("use std::rc::{Rc, Weak};");
        self.line("use std::sync::Arc;");
        // Single-thread mode uses RefCell for T'actor/T'guard — add import whenever threading=single.
        if matches!(self.config.threading, ThreadingMode::Single) {
            self.line("use std::cell::RefCell;");
        }
        // Multi-thread mode with no async fns uses std::sync::Mutex for T'actor (no .await needed).
        let has_async_fns = !self.task_fns.is_empty() || !self.stream_fns.is_empty();
        if matches!(self.config.threading, ThreadingMode::Multi) && !has_async_fns {
            self.line("use std::sync::{Mutex, RwLock};");
        }
        // Skip any of PI/E/TAU the program defines itself as a top-level `let` --
        // otherwise the two collide (E0255: name defined multiple times).
        let f64_consts: Vec<&str> = ["PI", "E", "TAU"].into_iter()
            .filter(|n| !self.user_top_level_names.contains(*n))
            .collect();
        if !f64_consts.is_empty() {
            self.line(&format!("use std::f64::consts::{{{}}};", f64_consts.join(", ")));
        }
        self.line("use std::time::Duration;");
        if matches!(self.config.threading, ThreadingMode::Single) && self.uses_local_channel.get() {
            self.line("use local_channel;");
        }
        self.blank();
        if matches!(self.config.threading, ThreadingMode::Single) && self.uses_local_broadcast.get() {
            self.emit_local_broadcast_prelude();
        }
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
             \x20   fn get_at(&self, i: Option<usize>) -> T {\n\
             \x20       self[i.expect(\"get_at called with no index (empty collection or exhausted cursor)\")].clone()\n\
             \x20   }\n\
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
             \x20   fn get_at(&self, i: Option<usize>) -> T {\n\
             \x20       let i = i.expect(\"get_at called with no index (empty collection or exhausted cursor)\");\n\
             \x20       self.iter().nth(i).cloned().expect(\"get_at index out of range\")\n\
             \x20   }\n\
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
             //\n\
             // `Scalar(ScalarKind, u128)` is the fixed-width numeric family's own path —\n\
             // int8..int128, uint8..uint128, and float32 — kept OUT of `Other` on purpose\n\
             // (docs/float-width-types.md §7): the compiler already knows, statically,\n\
             // that a thrown value here is exactly one of eleven small Copy kinds, so\n\
             // paying for a heap allocation + `dyn Any` downcast the way `Other` does for\n\
             // arbitrary user enums/structs would be pure overhead. The raw bits are\n\
             // reinterpreted per `kind` in `Display` below and at each `catch` arm —\n\
             // sign/zero-extended into the shared `u128` slot on the way in, truncated (or\n\
             // bit-reinterpreted, for float32) back to the exact original type on the way\n\
             // out; this round-trips exactly, it never touches `Other`'s TypeId machinery\n\
             // at all. `float`/`float64` deliberately stay on their own pre-existing\n\
             // `BoringError::Float(f64)` fast path instead of joining `Scalar` — unlike\n\
             // the other eleven kinds, they already had a dedicated, no-allocation\n\
             // representation before this family existed, and giving them a second,\n\
             // parallel one here would only create two disjoint representations for the\n\
             // same type with no single `catch` spelling reliably catching both (a literal\n\
             // float throw and a `float64`-typed variable throw would land in different\n\
             // variants). `Other` still exists, narrowed to what only it can do: genuine\n\
             // user-defined enums/structs the compiler can't enumerate in advance.\n\
             #[derive(Debug, Clone, Copy, PartialEq)]\n\
             enum ScalarKind { Int8, Int16, Int32, Int64, Int128, Uint8, Uint16, Uint32, Uint64, Uint128, Float32 }\n\
             impl BoringError {\n\
             \x20   fn scalar_i8(v: i8) -> Self { BoringError::Scalar(ScalarKind::Int8, v as i128 as u128) }\n\
             \x20   fn scalar_i16(v: i16) -> Self { BoringError::Scalar(ScalarKind::Int16, v as i128 as u128) }\n\
             \x20   fn scalar_i32(v: i32) -> Self { BoringError::Scalar(ScalarKind::Int32, v as i128 as u128) }\n\
             \x20   fn scalar_i64(v: i64) -> Self { BoringError::Scalar(ScalarKind::Int64, v as i128 as u128) }\n\
             \x20   fn scalar_i128(v: i128) -> Self { BoringError::Scalar(ScalarKind::Int128, v as u128) }\n\
             \x20   fn scalar_u8(v: u8) -> Self { BoringError::Scalar(ScalarKind::Uint8, v as u128) }\n\
             \x20   fn scalar_u16(v: u16) -> Self { BoringError::Scalar(ScalarKind::Uint16, v as u128) }\n\
             \x20   fn scalar_u32(v: u32) -> Self { BoringError::Scalar(ScalarKind::Uint32, v as u128) }\n\
             \x20   fn scalar_u64(v: u64) -> Self { BoringError::Scalar(ScalarKind::Uint64, v as u128) }\n\
             \x20   fn scalar_u128(v: u128) -> Self { BoringError::Scalar(ScalarKind::Uint128, v) }\n\
             \x20   fn scalar_f32(v: f32) -> Self { BoringError::Scalar(ScalarKind::Float32, v.to_bits() as u128) }\n\
             }\n\
             #[derive(Debug)]\n\
             enum BoringError {\n\
             \x20   Int(i64),\n\
             \x20   Float(f64),\n\
             \x20   Bool(bool),\n\
             \x20   Str(&'static str),\n\
             \x20   String(Arc<str>),\n\
             \x20   Scalar(ScalarKind, u128),\n\
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
             \x20           BoringError::Scalar(k, bits) => match k {\n\
             \x20               ScalarKind::Int8    => write!(f, \"{}\", *bits as i128 as i8),\n\
             \x20               ScalarKind::Int16   => write!(f, \"{}\", *bits as i128 as i16),\n\
             \x20               ScalarKind::Int32   => write!(f, \"{}\", *bits as i128 as i32),\n\
             \x20               ScalarKind::Int64   => write!(f, \"{}\", *bits as i128 as i64),\n\
             \x20               ScalarKind::Int128  => write!(f, \"{}\", *bits as i128),\n\
             \x20               ScalarKind::Uint8   => write!(f, \"{}\", *bits as u8),\n\
             \x20               ScalarKind::Uint16  => write!(f, \"{}\", *bits as u16),\n\
             \x20               ScalarKind::Uint32  => write!(f, \"{}\", *bits as u32),\n\
             \x20               ScalarKind::Uint64  => write!(f, \"{}\", *bits as u64),\n\
             \x20               ScalarKind::Uint128 => write!(f, \"{}\", *bits),\n\
             \x20               ScalarKind::Float32 => write!(f, \"{}\", f32::from_bits(*bits as u32)),\n\
             \x20           },\n\
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
             impl std::error::Error for Error {}\n\n\
             // Boring GPU error enum — thrown by kernel dispatch on a device-side\n\
             // failure. Always available without import. Currently wired up on the\n\
             // wgpu target only (its host codegen shares this same general pipeline);\n\
             // cuda/rocm/metal report GPU failures as plain string-message errors\n\
             // instead of this typed enum -- see `docs/gpu-module.md`'s error-handling\n\
             // notes for why (their host transpilers don't share this prelude).\n\
             // Use `catch GpuError.OutOfMemory:` / `match err: GpuError.Timeout: ...`\n\
             #[derive(Debug, Clone)]\n\
             #[allow(dead_code)]\n\
             enum GpuError {\n\
             \x20   LaunchError,\n\
             \x20   OutOfMemory,\n\
             \x20   IllegalAccess,\n\
             \x20   StackOverflow,\n\
             \x20   Timeout,\n\
             \x20   DeviceLost,\n\
             }\n\
             impl std::fmt::Display for GpuError {\n\
             \x20   fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {\n\
             \x20       match self {\n\
             \x20           GpuError::LaunchError   => write!(f, \"GPU kernel launch failed\"),\n\
             \x20           GpuError::OutOfMemory   => write!(f, \"GPU out of memory\"),\n\
             \x20           GpuError::IllegalAccess => write!(f, \"GPU illegal memory access\"),\n\
             \x20           GpuError::StackOverflow => write!(f, \"GPU stack overflow\"),\n\
             \x20           GpuError::Timeout       => write!(f, \"GPU operation timed out\"),\n\
             \x20           GpuError::DeviceLost    => write!(f, \"GPU device lost\"),\n\
             \x20       }\n\
             \x20   }\n\
             }\n\
             impl std::error::Error for GpuError {}\n\n"
        );

        if self.config.instrument {
            self.out.push_str(
                "mod __boring_instrument {\n\
                 \x20   use std::collections::HashMap;\n\
                 \x20   use std::sync::Mutex;\n\
                 \x20   use std::time::{Instant, SystemTime, UNIX_EPOCH};\n\
                 \x20   struct CallStats { count: u64, total_ns: u64 }\n\
                 \x20   static REGISTRY: Mutex<Option<HashMap<&'static str, CallStats>>> = Mutex::new(None);\n\
                 \x20   static JOURNAL: Mutex<Option<Vec<(u128, &'static str, u64)>>> = Mutex::new(None);\n\
                 \x20   pub struct Span { name: &'static str, start: Instant, ts_ns: u128 }\n\
                 \x20   impl Span {\n\
                 \x20       pub fn enter(name: &'static str) -> Self {\n\
                 \x20           { let mut r = REGISTRY.lock().unwrap_or_else(|e| e.into_inner()); if r.is_none() { *r = Some(HashMap::new()); } }\n\
                 \x20           { let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner()); if j.is_none() { *j = Some(Vec::new()); } }\n\
                 \x20           let ts_ns = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);\n\
                 \x20           Self { name, start: Instant::now(), ts_ns }\n\
                 \x20       }\n\
                 \x20   }\n\
                 \x20   impl Drop for Span {\n\
                 \x20       fn drop(&mut self) {\n\
                 \x20           let dur = self.start.elapsed().as_nanos() as u64;\n\
                 \x20           if let Some(r) = REGISTRY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {\n\
                 \x20               let e = r.entry(self.name).or_insert(CallStats { count: 0, total_ns: 0 });\n\
                 \x20               e.count += 1; e.total_ns += dur;\n\
                 \x20           }\n\
                 \x20           if let Some(j) = JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {\n\
                 \x20               j.push((self.ts_ns, self.name, dur));\n\
                 \x20           }\n\
                 \x20       }\n\
                 \x20   }\n\
                 \x20   pub struct DumpGuard;\n\
                 \x20   impl Drop for DumpGuard { fn drop(&mut self) { dump(); } }\n\
                 \x20   pub fn dump() {\n\
                 \x20       if let Some(r) = REGISTRY.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {\n\
                 \x20           let mut entries: Vec<_> = r.iter().collect();\n\
                 \x20           entries.sort_by_key(|(n, _)| *n);\n\
                 \x20           let mut out = String::from(\"{\\n\");\n\
                 \x20           let len = entries.len();\n\
                 \x20           for (i, (name, s)) in entries.iter().enumerate() {\n\
                 \x20               let avg = if s.count > 0 { s.total_ns / s.count / 1000 } else { 0 };\n\
                 \x20               let comma = if i + 1 < len { \",\" } else { \"\" };\n\
                 \x20               out.push_str(&format!(\"  \\\"{}\\\": {{\\\"calls\\\": {}, \\\"total_us\\\": {}, \\\"avg_us\\\": {}}}{}\n\", name, s.count, s.total_ns / 1000, avg, comma));\n\
                 \x20           }\n\
                 \x20           out.push('}');\n\
                 \x20           let _ = std::fs::write(\"boring_coverage.json\", out);\n\
                 \x20       }\n\
                 \x20       if let Some(j) = JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {\n\
                 \x20           let mut out = String::from(\"{\\n  \\\"traceEvents\\\": [\");\n\
                 \x20           let len = j.len();\n\
                 \x20           for (i, (ts, name, dur)) in j.iter().enumerate() {\n\
                 \x20               let ts_us = ts / 1000;\n\
                 \x20               let comma = if i + 1 < len { \",\" } else { \"\" };\n\
                 \x20               out.push_str(&format!(\"\\n    {{\\\"name\\\":\\\"{}\\\",\\\"ph\\\":\\\"X\\\",\\\"ts\\\":{},\\\"dur\\\":{},\\\"pid\\\":1,\\\"tid\\\":1,\\\"cat\\\":\\\"boring\\\"}}{}\", name, ts_us, dur / 1000, comma));\n\
                 \x20           }\n\
                 \x20           out.push_str(\"\\n  ],\\n  \\\"displayTimeUnit\\\": \\\"ms\\\"\\n}\");\n\
                 \x20           let _ = std::fs::write(\"boring_trace.json\", out);\n\
                 \x20       }\n\
                 \x20   }\n\
                 }\n\n"
            );
        }

        // Emit module-level mutable vars that are accessed by functions as LazyLock<Mutex<T>> statics.
        // These cannot be locals inside main() because functions reference them as free variables.
        if !self.global_vars_used_in_fns.is_empty() {
            // Collect, then deduplicate by name keeping the LAST declaration (shadowing: the
            // last `var x = ...` wins).  Without dedup, two `var i = ...` shadow declarations
            // would produce two `static I: ...` causing E0428 "defined multiple times".
            let global_vars_raw: Vec<(String, Option<Type>, String)> = program.items.iter()
                .filter_map(|item| {
                    if let Item::Let(s) = item {
                        if s.binding.is_mutable() && self.global_vars_used_in_fns.contains(&s.name) {
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
                } else if init.starts_with("Arc::new(") || init.starts_with("Arc::<str>::from(") || init.starts_with("Rc::<str>::from(") || init.starts_with("\"") {
                    "Arc<str>".to_string()
                } else if init.starts_with("vec![") {
                    "Vec<isize>".to_string()
                } else {
                    "isize".to_string()
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

        self.emit_program_items(program, true);
    }

    fn emit_program_items(&mut self, program: &Program, emit_main: bool) {
        // Separate top-level declarations from statements
        let mut stmts: Vec<&Item> = Vec::new();
        // emitted_fn_sigs is shared across all inlined files (self.emitted_fn_sigs) to deduplicate
        // functions that appear in multiple files (kept for historical compatibility).
        // GPU targets whose host backend already owns top-level orchestration by
        // itself (wgpu's `emit_screen_main`, for a `Screen`-using program -- see
        // `TranspileConfig::gpu_top_level_handled_by_host`'s doc) -- nothing here would
        // ever run it anyway, and some of it (Dimension-shaped/no-`init` kernel
        // construction) isn't a pattern `emit_kernel.rs` understands. Skip every
        // top-level statement/let entirely instead of emitting or falling through.
        let host_owns_top_level = self.is_gpu_target && self.config.gpu_top_level_handled_by_host;
        for item in &program.items {
            match item {
                Item::Stmt(_) if host_owns_top_level => {}
                Item::Stmt(_) => stmts.push(item),
                Item::Let(s) if s.is_static => {
                    // Top-level `static let` → emit as `const` at module scope.
                    self.emit_item(item);
                    self.blank();
                }
                Item::Let(_) if host_owns_top_level => {}
                // GPU targets: there is no auto-generated `fn main()` body for a plain
                // top-level `let` to live in as a local, so a scalar constant needs to
                // become a real global `const` other top-level functions can reference
                // (gated on `is_gpu_target`, NOT on `kernel_decls` being non-empty -- a
                // GPU-target file can have zero kernels reachable from it and still
                // needs this treatment). Anything that ISN'T a plain scalar constant
                // (a kernel constructor call, an array/dict/set/string value) has no
                // const constructor in Rust -- fall through to `stmts` below instead,
                // so it runs as an ordinary statement inside the synthesized/renamed
                // `boring_main` (see the `is_gpu_target` branch further down), the same
                // way it would for a non-GPU target's auto-generated `fn main()`.
                Item::Let(s) if self.is_gpu_target && self.top_level_let_is_const_safe(s) => {
                    self.emit_item(item);
                    self.blank();
                }
                // Non-GPU targets: a plain top-level `let` normally becomes a local
                // inside the synthesized `fn main()` (see the `stmts` fallback below),
                // which is invisible to every other function/method. When
                // `global_lets_used_elsewhere` (computed in `pre_scan`) says this
                // constant IS referenced from outside that synthesized `main()`, it must
                // be promoted to real module scope instead -- otherwise the emitted Rust
                // simply doesn't compile (E0425 "cannot find value"). `top_level_let_is_promotable`
                // (broader than the scalar-only `top_level_let_is_const_safe` used by the GPU
                // branch above) also admits a call into an external/opaque type
                // (`Color.srgb(...)`, `Vec2.new(...)`) -- `emit_item`'s `Item::Let` arm picks
                // `const` vs a lazily-initialized `static` for those, see
                // `top_level_let_external_call`'s doc. A `pub` let ALSO always takes this
                // branch regardless of `global_lets_used_elsewhere` -- `pub` at module scope
                // is a declaration of public API, exactly like `pub struct`/`pub def`/`pub
                // enum`, which promote unconditionally with no "is it used elsewhere in this
                // file" check at all. Gating a `pub let` on in-file usage would silently drop
                // (or, worse, mis-emit as an invalid local `pub let` inside `fn main()`, see
                // `emit_let`) a constant whose entire purpose is to be read from a sibling
                // module/crate that this file's own pre_scan can never see. A bare
                // (non-`pub`) `let` keeps the old behavior: gated on actual cross-scope usage,
                // so ordinary top-level scripting `let`s -- including same-named shadowing
                // re-declarations -- keep behaving as sequential locals in `main()`.
                Item::Let(s) if self.top_level_let_is_promotable(s)
                    && (s.is_pub || self.global_lets_used_elsewhere.contains(&s.name)) => {
                    self.emit_item(item);
                    self.blank();
                }
                Item::Let(_) => stmts.push(item),
                Item::Fn(f) if f.qualifier.is_none() => {
                    // Build a unique signature key: name + param types.
                    // For non-overloaded functions the key is just the name (same as before).
                    // For overloaded functions each variant has a distinct key.
                    let sig_key = if self.overloaded_fn_names.contains(&f.name) {
                        mangle_overload_name(&f.name, &f.params)
                    } else {
                        f.name.clone()
                    };
                    if !self.emitted_fn_sigs.insert(sig_key) {
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
        // GPU targets rename the user's `main` to `boring_main` (see
        // `wgpu::rename_top_level_main`) *before* this function ever sees the program, so by
        // the time we get here the name to look for is `boring_main`, not `main` -- checking
        // only "main" would miss it and wrongly re-synthesize a second `fn boring_main` from
        // any bare top-level statements, causing a duplicate-symbol compile error.
        let has_explicit_main = program.items.iter().any(|item| {
            matches!(item, Item::Fn(f) if f.qualifier.is_none()
                && (f.name == "main" || (self.is_gpu_target && f.name == "boring_main")))
        });

        // GPU targets (wgpu/cuda/metal) call transpile_with_config to get Rust code for
        // everything the kernel-specific backend doesn't handle itself (see
        // transpiler::wgpu::transpile_wgpu), and provide their own `fn main()` /
        // `async fn async_main()` separately. An explicit user `def main():` was already
        // renamed to `boring_main` before this call (see
        // `transpiler::wgpu::rename_top_level_main`) and is emitted normally via the
        // ordinary `Item::Fn` path above -- nothing more to do here for that case.
        // But bare top-level statements/non-const `let`s (e.g. `mut k = Scale(data)` /
        // `kernel: k(block=..)` / `print k.buf[0]`, the documented top-level kernel
        // usage pattern) have nowhere else to run: there's no auto-generated `fn main()`
        // for a GPU target, and silently dropping them would discard real program
        // behavior. Synthesize the same `boring_main` wrapper for them instead, so the
        // GPU target's own generated entry point can call it exactly like an explicit
        // user main -- `gpu_main_emitted` tells the caller (which can't see this
        // function in the source AST, only in this emitted text) that it now exists.
        if self.is_gpu_target {
            if !stmts.is_empty() && !has_explicit_main {
                self.emit_gpu_boring_main(&stmts);
                self.gpu_main_emitted.set(true);
            }
            return;
        }
        if !emit_main { return; }
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
            let use_local_set = needs_async && matches!(self.config.threading, ThreadingMode::Single);
            if needs_async {
                match self.config.threading {
                    ThreadingMode::Single => self.line("#[tokio::main(flavor = \"current_thread\")]"),
                    ThreadingMode::Multi  => self.line("#[tokio::main]"),
                }
                self.line(&format!("async fn main() -> {} {{", main_ret));
                self.in_async = true;
            } else {
                self.line(&format!("fn main() -> {} {{", main_ret));
            }
            self.indent += 1;
            if self.config.instrument {
                self.line("let _boring_dump = __boring_instrument::DumpGuard;");
                self.line("let _boring_span = __boring_instrument::Span::enter(\"main\");");
            }
            if use_local_set {
                self.line("tokio::task::LocalSet::new().run_until(async move {");
                self.indent += 1;
            }
            // main() returns Result<(), Box<dyn Error>>, so throws function calls should get `?`.
            self.in_throws = true;
            for item in stmts.iter() {
                match item {
                    Item::Let(s) => {
                        // Skip top-level mutable vars that are emitted as module-level statics.
                        if s.binding.is_mutable() && self.global_vars_used_in_fns.contains(&s.name) {
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
            if use_local_set {
                self.indent -= 1;
                self.line("}).await");
            }
            self.indent -= 1;
            self.line("}");
            if needs_async {
                self.in_async = false;
            }
        }
    }

    /// Synthesize `fn boring_main() -> Result<(), ...> { ...stmts...; Ok(()) }` for GPU
    /// targets with no explicit user `main`, from the same leftover top-level
    /// statements/non-const `let`s the non-GPU path above would otherwise wrap in an
    /// auto-generated `fn main()`. Deliberately synchronous (no `#[tokio::main]`/async
    /// promotion, unlike the non-GPU path): the GPU host backend's own generated entry
    /// point (`async_main`, see `transpiler::wgpu::host`) calls `boring_main()` without
    /// `.await`, and drives its own async setup (device/adapter requests) with
    /// `pollster`, not a tokio runtime -- `tokio::spawn`/tokio channels/timers would
    /// panic at runtime ("no reactor running") even if this function were made `async`
    /// and awaited. Fail at transpile time instead of generating code that compiles
    /// but panics the first time it actually runs.
    fn emit_gpu_boring_main(&mut self, stmts: &[&Item]) {
        let top_stmts: Vec<Stmt> = stmts.iter().filter_map(|i| {
            if let Item::Stmt(s) = i { Some((*s).clone()) } else { None }
        }).collect();
        let needs_async = items_have_task(stmts)
            || body_has_stream_for(&top_stmts, &self.stream_fns)
            || items_have_task_call(stmts, &self.task_fns);
        if needs_async {
            let (line, col) = match stmts.first() {
                Some(Item::Stmt(Stmt::Expr(e))) => (e.line, e.col),
                Some(Item::Let(s)) => (s.line, s.col),
                _ => (0, 0),
            };
            self.push_error(line, col,
                "top-level `task`/stream/channel usage isn't supported for GPU targets (wgpu/cuda/metal) yet -- \
                 the generated entry point runs under a minimal `pollster` executor, not a tokio runtime, so \
                 `tokio::spawn`/tokio channels/timers would panic at runtime (\"no reactor running\") even if this \
                 code were made async. Move this into a synchronous `def main():` instead, or avoid task/stream/\
                 channel usage in top-level GPU code for now."
            );
            self.line("fn boring_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) }");
            return;
        }
        let result_ty = if self.user_defines_result { "std::result::Result" } else { "Result" };
        let box_ty = if self.user_defines_box { "std::boxed::Box" } else { "Box" };
        let main_ret = format!("{}<(), {}<dyn std::error::Error + Send + Sync>>", result_ty, box_ty);
        let ok_ret = if self.user_defines_result { "std::result::Result::Ok(())" } else { "Ok(())" };
        self.line(&format!("fn boring_main() -> {} {{", main_ret));
        self.indent += 1;
        self.in_throws = true;
        for item in stmts {
            match item {
                Item::Let(s) => {
                    if s.binding.is_mutable() && self.global_vars_used_in_fns.contains(&s.name) {
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
    }

    /// Whether a top-level `let`'s value can compile as a Rust `const` -- true only for
    /// scalar types (int/uint/float/bool), false for anything backed by a heap
    /// allocation (`[T]`, `string`, `{K=V}`, `{T}`) or a kernel constructor call, none of
    /// which have a const constructor in stable Rust. Used to gate GPU targets'
    /// top-level-`let`-to-`const` promotion (see `emit_program_items`).
    fn top_level_let_is_const_safe(&self, s: &LetStmt) -> bool {
        fn is_scalar(t: &Type) -> bool {
            match t {
                Type::Int | Type::Uint | Type::Uint8 | Type::Float32 | Type::Float64 | Type::Bool => true,
                Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                    | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128 => true,
                Type::Named(n) => matches!(n.as_str(),
                    "int" | "uint" | "uint8" | "float" | "float32" | "float64" | "bool"
                    | "int8" | "int16" | "int32" | "int64" | "int128"
                    | "uint16" | "uint32" | "uint64" | "uint128"
                    | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
                    | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"),
                Type::Optional(inner) | Type::Qualified(inner, _) => is_scalar(inner),
                _ => false,
            }
        }
        if let Some(ty) = &s.ty {
            return is_scalar(ty);
        }
        // No declared type -- only trust a bare scalar literal initializer, unwrapping a
        // single leading unary op first (`-450.0` parses as `UnaryOp(Neg, Float(450.0))`,
        // not a bare `Float` literal -- without this, every negative numeric constant,
        // starting with this exact bug report's own `LEFT_WALL = -450.0` example, would
        // silently fail this check and never get promoted to a Rust `const` at all).
        // `-`/`!`/`~` all still compile fine as a `const` initializer in Rust.
        fn is_scalar_literal(v: &Expr) -> bool {
            match &v.kind {
                ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => true,
                ExprKind::UnaryOp(_, inner) => is_scalar_literal(inner),
                _ => false,
            }
        }
        s.value.as_ref().map(is_scalar_literal).unwrap_or(false)
    }

    /// Whether a top-level `let`'s initializer is a call into an external/opaque type --
    /// `Color.srgb(0.8, 0.8, 0.8)`, `Vec2.new(120.0, 20.0)`, or an external enum's
    /// tuple-variant dot-shorthand (`FontSize.Px(33.0)`) -- rather than one of this
    /// file's own structs/enums (`self.user_types`, populated before this ever runs --
    /// see `pre_scan`'s dedicated pass for why that has to happen first). Boring parses
    /// both `Type.method(args)` and `Type.Variant(args)` as the same
    /// `MethodCall(Var(Type), method, args)` shape (see `try_emit_type_method_call`'s
    /// identical match), so this only needs to look at the receiver, not the whole call
    /// or which of the two it is. Returns `(type_name, method)` so callers can both name
    /// the promoted item's Rust type (see `top_level_let_promoted_type`) and check
    /// `is_known_external_const_fn`/`is_external_enum_variant_construction` for the
    /// const-vs-static decision (`emit_item`'s `Item::Let` arm).
    fn top_level_let_external_call(&self, s: &LetStmt) -> Option<(String, String)> {
        let ExprKind::MethodCall(obj, method, _) = &s.value.as_ref()?.kind else { return None };
        let ExprKind::Var(type_name) = &obj.kind else { return None };
        if type_name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            && !self.user_types.contains(type_name.as_str())
        {
            Some((type_name.clone(), method.clone()))
        } else {
            None
        }
    }

    /// Whether a top-level `let`'s initializer is a plain string literal (no
    /// interpolation, no method call) with either no declared type or an explicit
    /// `string` type -- `pub let OP_ADD = "operator_add"` / `pub let string OP_ADD =
    /// "operator_add"`. Unlike the scalar case, `Rc/Arc::<str>::from(...)` is never a
    /// `const fn`, so a promoted string constant always takes the same lazily-
    /// initialized `static ... LazyLock<T>` form `emit_item`'s `Item::Let` arm already
    /// uses for a non-const-fn external call -- see that arm's dedicated branch for
    /// string literals. GPU/kernel (`no_std`) targets have neither `Rc`/`Arc` nor
    /// `std::sync::LazyLock` available, so this always returns `false` there, same
    /// exclusion `top_level_let_external_call` gets at its call sites.
    fn top_level_let_is_string_literal(&self, s: &LetStmt) -> bool {
        if self.is_gpu_target {
            return false;
        }
        if let Some(ty) = &s.ty {
            if !Self::is_string_type(ty) {
                return false;
            }
        }
        matches!(s.value.as_ref().map(|v| &v.kind), Some(ExprKind::Str(_)))
    }

    /// Whether a top-level `let` can be promoted out of the synthesized `fn main()` into
    /// real module scope at all -- the scalar case (`top_level_let_is_const_safe`, always
    /// a `const`) plus a call into an external/opaque type (`top_level_let_external_call`,
    /// `const` or a lazily-initialized `static` depending on `is_known_external_const_fn`
    /// -- see `emit_item`'s `Item::Let` arm for that split) plus a plain string literal
    /// (`top_level_let_is_string_literal`, always the lazily-initialized `static` form).
    /// Anything else (an array/dict/set value, a call into one of the file's own struct/
    /// enum constructors, a kernel constructor call) has no module-scope Rust
    /// representation this transpiler can emit yet and stays a `fn main()` local, same as
    /// before this function existed.
    fn top_level_let_is_promotable(&self, s: &LetStmt) -> bool {
        self.top_level_let_is_const_safe(s)
            || self.top_level_let_external_call(s).is_some()
            || self.top_level_let_is_string_literal(s)
    }

    /// Hand-verified external associated functions that ARE `const fn` in their defining
    /// crate -- used by `emit_item`'s `Item::Let` arm to decide `const` vs a lazily-
    /// initialized `static` for a promoted `let` whose initializer calls into an
    /// external/opaque type (`top_level_let_external_call`). Deliberately small: getting
    /// this wrong in the "yes, it's const" direction emits invalid Rust (E0015, "cannot
    /// call non-const fn in const context") that only surfaces at `cargo build` time,
    /// long after `boring build` reported success -- far worse than the always-valid
    /// `static` fallback just being slightly less idiomatic for a constructor that
    /// happens to be const and wasn't recognized. Extend this list as more such
    /// constructors (e.g. from `glam`/`bevy`) are hand-verified against their source.
    ///
    /// The `bevy_color::Color` entries below are hand-verified against
    /// `crates/bevy_color/src/color.rs` in bevyengine/bevy (main branch) -- every listed
    /// constructor is `pub const fn`. Deliberately NOT listed: `srgb_u32`/`srgba_u32`,
    /// which are plain `pub fn` in that same source (not const).
    ///
    /// A promoted `let` that stays a `static ... LazyLock<T>` (i.e. its call isn't on
    /// this list) auto-derefs to `T` at a method-call receiver or a `.field` read --
    /// `PADDLE_SIZE.x`, `SPAWN_DELAY.as_secs()` -- because Rust's method/field lookup
    /// walks the `Deref` chain on its own. It does NOT auto-deref at a struct-literal
    /// field value, a by-value function argument, or a return position -- those need an
    /// exact `T`, and Rust never inserts `Deref` coercion for owned values, only for
    /// references. Using the identifier bare there (`Sprite { color: PADDLE_COLOR, .. }`)
    /// is a real `cargo build` type error (E0308) that `boring build` cannot catch --
    /// write `PADDLE_COLOR.pointee` (see `.pointee` postfix deref) at that call site
    /// instead. See `docs/book.md`'s "External-typed top-level constants" section and
    /// `tests/cases/ext_const_promotion/src/main.br` for a worked example of both paths.
    const KNOWN_EXTERNAL_CONST_FNS: &[(&str, &str)] = &[
        ("Duration", "from_secs"), ("Duration", "from_millis"),
        ("Duration", "from_micros"), ("Duration", "from_nanos"),
        ("Vec2", "new"), ("Vec2", "splat"),
        ("Vec3", "new"), ("Vec3", "splat"),
        ("Vec3A", "new"), ("Vec3A", "splat"),
        ("Vec4", "new"), ("Vec4", "splat"),
        // bevy_color::Color -- see doc comment above for the source hand-verified against.
        ("Color", "srgb"), ("Color", "srgba"),
        ("Color", "srgb_from_array"), ("Color", "srgb_u8"), ("Color", "srgba_u8"),
        ("Color", "linear_rgb"), ("Color", "linear_rgba"),
        ("Color", "hsl"), ("Color", "hsla"),
        ("Color", "hsv"), ("Color", "hsva"),
        ("Color", "hwb"), ("Color", "hwba"),
        ("Color", "lab"), ("Color", "laba"),
        ("Color", "lch"), ("Color", "lcha"),
        ("Color", "oklab"), ("Color", "oklaba"),
        ("Color", "oklch"), ("Color", "oklcha"),
        ("Color", "xyz"), ("Color", "xyza"),
    ];

    /// Consults the built-in hand-verified list first, then `self.config.external_const_fns`
    /// (the project's own `boring.toml` `[external_types]` supplement — see
    /// `TranspileConfig::external_const_fns`'s doc comment). The project list is purely
    /// additive: it can never remove or override a built-in entry.
    fn is_known_external_const_fn(&self, type_name: &str, method: &str) -> bool {
        Self::KNOWN_EXTERNAL_CONST_FNS.iter().any(|&(t, m)| t == type_name && m == method)
            || self.config.external_const_fns.iter().any(|(t, m)| t == type_name && m == method)
    }

    /// Whether `top_level_let_external_call`'s `(type_name, method)` is actually an
    /// external *enum tuple-variant* construction (`FontSize.Px(33.0)` →
    /// `FontSize::Px(33.0)`) rather than an associated-function call (`Color.srgb(...)`).
    /// Unlike `KNOWN_EXTERNAL_CONST_FNS`, this needs no allowlist: constructing an enum
    /// variant is a plain data constructor in Rust, always usable in a `const`
    /// initializer no matter which enum or variant it is — there is no equivalent of a
    /// non-`const fn` to guess wrong about. `try_emit_type_method_call`
    /// (emit_methods.rs) uses the exact same signal (a capitalized `method` on a type it
    /// doesn't recognize as a registered enum) to decide this is a variant rather than a
    /// method, and emits it verbatim (case-preserved) for the same reason — a real Rust
    /// method name is never capitalized, so a capitalized `method` reaching either check
    /// can only be a variant (or associated constant, which this call shape — always
    /// with parens/args per `top_level_let_external_call`'s `MethodCall` match — never
    /// produces).
    fn is_external_enum_variant_construction(method: &str) -> bool {
        method.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
    }

    /// Hand-verified external types that ARE plain tuple structs with all-`pub` fields in
    /// their defining crate -- consulted by `emit_constructor_inner`'s positional-args
    /// fallback (emit_expr.rs) to decide between a literal tuple-struct call `Type(args)`
    /// and `Type::new(args)`. Same rationale as `KNOWN_EXTERNAL_CONST_FNS` just above:
    /// deliberately small, because getting this wrong in the "yes, it's a tuple struct"
    /// direction emits invalid Rust (calling `Type::new` on a plain tuple struct with no
    /// inherent `new()` -- E0599) that only surfaces at `cargo build` time, long after
    /// `boring build` reported success. `Mesh2d`/`MeshMaterial2d` (bevy 0.19,
    /// `bevy_mesh`/`bevy_sprite_render`) are both `pub struct Foo(pub Handle<T>);` with no
    /// inherent `new()` -- the motivating case (see `boring/breakout.br`'s ball-spawning
    /// code). Extend this list as more such bevy/glam types are hand-verified against
    /// their source. Do NOT add a type here on the assumption it "should" be a tuple
    /// struct -- unlike the const-fn list, a wrong entry here means Boring emits
    /// `Type(args)` for a type that is NOT a tuple struct, which is just as broken as the
    /// bug this list fixes, only inverted.
    ///
    /// A fully general fix (try the literal-call form whenever the callee resolves to a
    /// type rather than a function, since the two forms are never both valid for the same
    /// external identifier) isn't feasible without this list turning into real semantic
    /// resolution against the dependency's own Rust source -- Boring's transpiler has no
    /// symbol table for imported external items at all (see `emit_call`, emit_expr.rs:
    /// any capitalized callee is routed to `emit_constructor` purely on capitalization,
    /// whether or not a type by that name actually exists), so there is no "resolves to a
    /// type" signal to test in the first place; the entire outcome of this branch is
    /// always a guess between the two call forms, made by matching a bare identifier
    /// string. Flipping the *default* the other way (prefer `Type(args)`) would just trade
    /// one guess for its opposite -- it breaks `Box(v)` (tests/cases/pointee.br, a compiler
    /// lang item with no tuple-struct fields to call positionally) and every bevy type
    /// confirmed *not* to match this shape below (e.g. `BoxShadow`, `TextSpan` -- both real
    /// tuple structs but with a genuine inherent `new()`; `BorderColor`/`Outline`/
    /// `BorderRadius` -- named-field structs, not tuple structs at all in bevy 0.19). Hence
    /// this stays an explicit, hand-verified allowlist rather than a general check.
    ///
    /// Bevy 0.19.0 entries below were audited directly against
    /// `~/.cargo/registry/src/.../bevy_{text,ui}-0.19.0/src/{text.rs,ui_node.rs,
    /// gradients.rs}` (version pin confirmed against `breakout-boring/Cargo.lock`, which
    /// pins `bevy`/`bevy_text`/`bevy_ui` etc. all to exactly 0.19.0) -- each is a plain
    /// `pub struct Foo(pub T);` with no inherent `fn new(...)` anywhere in its defining
    /// crate (an `impl Foo` block with only associated *consts* like `BLACK`/`WHITE`/
    /// `DEFAULT` doesn't count as a constructor and doesn't block this):
    ///   - `TextColor`/`TextBackgroundColor`/`StrikethroughColor`/`UnderlineColor`
    ///     (`bevy_text::text`) -- `TextColor` is the motivating case (`Type(args)` used to
    ///     emit invalid `TextColor::new(args)`, blocking `breakout-boring/src/lib.rs`'s
    ///     `spawn_scoreboard_ui` from moving to Boring; `TextBackgroundColor`/
    ///     `StrikethroughColor`/`UnderlineColor` are the same wrapper shape in the same
    ///     file).
    ///   - `BackgroundColor`/`OuterColor`/`ZIndex`/`GlobalZIndex`/`UiTargetCamera`/
    ///     `ScrollPosition`/`IgnoreScroll` (`bevy_ui::ui_node`).
    ///   - `BackgroundGradient`/`BorderGradient` (`bevy_ui::gradients`).
    ///   - `ClearColor` (`bevy_camera::clear_color`, re-exported via `bevy::prelude`) --
    ///     audited against `~/.cargo/registry/src/.../bevy_camera-0.19.0/src/
    ///     clear_color.rs` (same version pin as above): `pub struct ClearColor(pub
    ///     Color);` with only a `Default` impl, no inherent `new()`. Motivating case:
    ///     `breakout-boring/src/lib.rs`'s `BACKGROUND_COLOR` constant moving to a genuine
    ///     Boring `Startup` system (`commands.insert_resource(ClearColor(BACKGROUND_COLOR))`)
    ///     instead of hand-written Rust -- same failure shape as `TextColor`
    ///     (`ClearColor::new(args)`, E0599).
    ///     NOT added despite looking similar -- confirmed wrong by reading source, do not
    ///     re-add without re-verifying against the actual bevy version in use:
    ///   - `BorderColor`/`Outline`/`BorderRadius` -- named-field structs in bevy 0.19, not
    ///     tuple structs (`BorderColor` was suggested as a candidate for this list; it
    ///     isn't one -- it has `top`/`right`/`bottom`/`left` fields).
    ///   - `BoxShadow`/`TextSpan` -- real tuple structs but each has a genuine inherent
    ///     `new()`, so the existing `Type::new(args)` default is already correct for them.
    const KNOWN_EXTERNAL_TUPLE_STRUCTS: &[&str] = &[
        "Mesh2d",
        "MeshMaterial2d",
        "TextColor",
        "TextBackgroundColor",
        "StrikethroughColor",
        "UnderlineColor",
        "BackgroundColor",
        "OuterColor",
        "ZIndex",
        "GlobalZIndex",
        "UiTargetCamera",
        "ScrollPosition",
        "IgnoreScroll",
        "BackgroundGradient",
        "BorderGradient",
        "ClearColor",
    ];

    /// Hand-verified argument-borrow requirements for external (Rust, non-Boring-authored)
    /// functions/methods whose signature takes `&T`/`&mut T` in a position *other than* the
    /// receiver (`self`) — the gap this whole table exists to close: `try_emit_type_method_call`
    /// and `emit_method_call_fallback`'s generic external-call fallback (emit_expr.rs/
    /// emit_methods.rs) have, by construction, no real external signature to consult when
    /// emitting a call's arguments, so every argument is emitted by value
    /// (`self.emit_expr(&a.value)`) unless something overrides it. A receiver `self` is NOT
    /// affected by this gap and needs no entry here — Rust auto-resolves `&mut self` from an
    /// owned place expression at a method-call receiver on its own (`entry.readToEnd(buf)`
    /// already transpiles fine); only arguments *after* the receiver need this.
    ///
    /// Each entry is `(qualifier, method, arg_borrows)`:
    /// - `qualifier` is the external type name for an instance-method call (matched against
    ///   `Transpiler::resolve_expr_struct_type(obj)`/`resolve_expr_external_type_name(obj)` —
    ///   the Boring-source type name the receiver was declared with, e.g. `"ZipFile"`), the
    ///   external type name itself for a static `Type.method(args)` call with no
    ///   Boring-registered `struct_type_method_sigs` entry (`"Tree"` for `Tree.from_str(...)`,
    ///   handled by `try_emit_type_method_call`'s own external fallback), or the fully-qualified module path as
    ///   normalized at the call site for a free function (e.g. `"std::mem"` for `mem.swap(...)`
    ///   /`std.mem.swap(...)`, both normalized to `std::mem` by `emit_method_call_fallback`'s
    ///   `obj_qualified` mapping before this table is consulted). Qualifying by type name (not
    ///   bare method name) is deliberate — the same lesson learned from a real bug this
    ///   session in an unrelated part of the transpiler (a bare-method-name lookup emitted a
    ///   stray `?` for a same-named-but-unrelated method on a different type, see
    ///   `emit_method_call_fallback`'s `receiver_struct`/`struct_throws` handling): matching by
    ///   method name alone here would apply another type's borrow requirements to an unrelated
    ///   method that happens to share its name.
    /// - `method` is the Rust name actually emitted at the call site — i.e. *after*
    ///   `camel_to_snake`/`map_method`, not necessarily the raw Boring source spelling
    ///   (`readToEnd` in `.br` source, `read_to_end` here).
    /// - `arg_borrows` is one entry per call argument, in order, each `"&"`, `"&mut"`, or `""`
    ///   (by value — the pre-existing, still-default behavior). For a method call this excludes
    ///   `self`/the receiver (index 0 of `arg_borrows` is the method's first *argument*, not the
    ///   receiver); for a free function every argument counts, since there is no receiver to
    ///   exclude. An `arg_borrows` shorter than the actual call's argument count leaves the
    ///   remaining trailing arguments at the by-value default rather than erroring — a
    ///   deliberately forgiving reading matching `Vec::get`-style out-of-bounds `None` handling
    ///   elsewhere in this file, not a validated arity check.
    ///
    /// `std::mem::swap`/`replace`/`take` are the motivating, zero-Cargo-dependency cases (a
    /// spike confirmed `mem.swap(a, b)` transpiles both arguments by value today, producing
    /// invalid Rust for `fn swap<T>(a: &mut T, b: &mut T)`) — hand-verified against
    /// `std::mem`'s actual signatures:
    ///   - `swap<T>(a: &mut T, b: &mut T)` — both arguments `&mut`.
    ///   - `replace<T>(dest: &mut T, src: T) -> T` — only the first argument `&mut`, the second
    ///     stays by value (moved in, not borrowed).
    ///   - `take<T: Default>(dest: &mut T) -> T` — its one argument `&mut`.
    ///     Extend this list as more such external functions/methods (e.g. `zip`'s
    ///     `ZipFile::read_to_end(&mut self, buf: &mut Vec<u8>)`, `resvg::render(tree: &Tree,
    ///     transform: Transform, pixmap: &mut PixmapMut)`) are hand-verified against their crate's
    ///     own source — or, for a project that doesn't want to patch the compiler for its own
    ///     third-party dependency, declared locally via `boring.toml`'s `[external_fns]` section
    ///     (see `TranspileConfig::external_fns`'s doc comment and docs/book.md's `[external_fns]`
    ///     section).
    const KNOWN_EXTERNAL_FN_BORROWS: &[(&str, &str, &[&str])] = &[
        ("std::mem", "swap", &["&mut", "&mut"]),
        ("std::mem", "replace", &["&mut", ""]),
        ("std::mem", "take", &["&mut"]),
    ];

    /// Consults the built-in hand-verified `KNOWN_EXTERNAL_FN_BORROWS` first, then
    /// `self.config.external_fns` (the project's own `boring.toml` `[external_fns]`
    /// supplement — see `TranspileConfig::external_fns`'s doc comment). The project list is
    /// purely additive: it can never remove or override a built-in entry. Returns `None` when
    /// neither list has an entry for `(qualifier, method)` — callers should fall back to the
    /// pre-existing by-value argument emission in that case, unchanged.
    ///
    /// `method` is always the already-`camel_to_snake`'d Rust name (both call sites pass
    /// `rust_method_name`/`rust_method`, computed *before* calling this). `KNOWN_EXTERNAL_FN_BORROWS`'s
    /// own entries are hand-written directly in that already-Rust form (`"swap"`, `"take"`, ...),
    /// so they compare equal as-is. A `boring.toml [external_fns]` key, by contrast, is stored
    /// verbatim from the TOML text as parsed by `BoringToml::parse_external_fn_key` — e.g.
    /// `"ZipFile::readToEnd"` stores the method half as literal `"readToEnd"`, the same casing a
    /// project author would naturally write matching their `.br` source's own call
    /// (`entry.readToEnd(buf)`). Without normalizing it here too, a project's own whitelist entry
    /// would silently never match at all (`"readToEnd" != "read_to_end"`) — found via this exact
    /// scenario (`ZipFile::readToEnd` against a real `zip` crate build) when `mem.swap`/`replace`/
    /// `take`'s own identical Boring-source/Rust spellings had never exposed the gap.
    pub(crate) fn external_fn_arg_borrows(&self, qualifier: &str, method: &str) -> Option<Vec<String>> {
        if let Some(&(_, _, borrows)) = Self::KNOWN_EXTERNAL_FN_BORROWS.iter()
            .find(|&&(q, m, _)| q == qualifier && m == method)
        {
            return Some(borrows.iter().map(|s| s.to_string()).collect());
        }
        self.config.external_fns.iter()
            .find(|(q, m, _)| q == qualifier && camel_to_snake(m) == method)
            .map(|(_, _, borrows)| borrows.clone())
    }

    /// Applies one `external_fn_arg_borrows` result to a call's already-emitted (by-value)
    /// argument strings — prefixing `&mut `/`&` at each position `borrows` names, leaving a
    /// `""` (or a position beyond `borrows`'s length) untouched. Shared by both call sites that
    /// consult the whitelist (the free-function path-receiver branch and the generic external
    /// instance-method fallback, both in emit_methods.rs) so the prefixing logic itself is
    /// written once.
    pub(crate) fn apply_arg_borrows(borrows: &[String], arg_strs: Vec<String>) -> Vec<String> {
        arg_strs.into_iter().enumerate().map(|(i, s)| {
            match borrows.get(i).map(String::as_str) {
                Some("&mut") => format!("&mut {}", s),
                Some("&") => format!("&{}", s),
                _ => s,
            }
        }).collect()
    }

    /// Consults the built-in hand-verified list first, then
    /// `self.config.external_tuple_structs` (the project's own `boring.toml`
    /// `[external_types]` supplement — see `TranspileConfig::external_tuple_structs`'s doc
    /// comment). The project list is purely additive: it can never remove or override a
    /// built-in entry.
    fn is_known_external_tuple_struct(&self, type_name: &str) -> bool {
        if self.config.external_tuple_structs.iter().any(|s| s == type_name) {
            return true;
        }
        Self::KNOWN_EXTERNAL_TUPLE_STRUCTS.contains(&type_name)
    }

    /// Rust traits that are always safe to route into `#[derive(...)]` when named in a
    /// struct/enum's header `as Trait1, Trait2:` list, instead of emitting a manual
    /// `impl Trait for X { ... }` — because these are proc-macro-derived with no method body
    /// for Boring code to ever provide (there is nothing a `def`/`req` in the struct body
    /// could be routed to). Limited to zero-Cargo-dependency std traits: `Serialize`/
    /// `Deserialize`/`thiserror::Error` are deliberately NOT pre-populated here even though
    /// they're also pure derive macros, because those three already have their own
    /// compiler-tracked auto-dependency injection (serde is added only when the `json()`/
    /// `fromJson()` builtins are used — see the `json`/`fromJson` doc section; thiserror is
    /// added only via `@error` variant attrs, `emit_enum`'s `already_has_thiserror` handling
    /// below) — routing them through this generic, unconditional whitelist would derive them
    /// without ever wiring up the matching Cargo dependency. A project can still add
    /// `Serialize`/`Deserialize`/etc. to its own `boring.toml` `[derives]` `traits = [...]`
    /// list, but is then responsible for also declaring the matching `[dependencies]` entry
    /// itself.
    ///
    /// Bevy-style project-specific derive macros (`Component`, `Resource`, `Event`,
    /// `States`, ...) are NOT pre-populated here either — see `TranspileConfig::known_derives`
    /// / `boring.toml`'s `[derives]` section (mirrors `[external_types]`) for how a project
    /// supplements this list.
    const KNOWN_DERIVABLE_TRAITS: &[&str] = &[
        "Debug", "Clone", "Copy", "PartialEq", "Eq", "PartialOrd", "Ord", "Hash", "Default",
    ];

    /// Consults the built-in list first, then `self.config.known_derives` (the project's own
    /// `boring.toml` `[derives]` supplement — see `TranspileConfig::known_derives`'s doc
    /// comment). Purely additive, same relationship as `is_known_external_tuple_struct`.
    fn is_known_derivable_trait(&self, name: &str) -> bool {
        if self.config.known_derives.iter().any(|s| s == name) {
            return true;
        }
        Self::KNOWN_DERIVABLE_TRAITS.contains(&name)
    }

    /// The Rust type to declare a promoted external-call `let` with (see
    /// `top_level_let_external_call`) -- an explicit Boring type annotation
    /// (`let Vec2 PADDLE_SIZE = ...`) wins when present, otherwise falls back to the
    /// call's own receiver name (`Vec2.new(...)` → `Vec2`), the standard Rust convention
    /// for a constructor-style associated function returning `Self`.
    fn top_level_let_promoted_type(&self, s: &LetStmt, external_type_name: &str) -> String {
        s.ty.as_ref().map(|t| self.emit_type(t)).unwrap_or_else(|| external_type_name.to_string())
    }

    /// Mirrors `Checker::collect_gpu_arg_params` (checker/mod.rs) — same fixed-point
    /// dataflow, recomputed independently here since the transpiler doesn't share
    /// checker state (same existing pattern as `kernel_decls`). See that function's
    /// doc comment for why this needs to be a whole-program fixed point rather than a
    /// single top-to-bottom walk: a parameter forwarded into another Boring function
    /// (not just a raw kernel constructor) qualifies too, transitively, when that
    /// callee's own corresponding parameter already qualifies.
    fn compute_gpu_arg_params(&mut self, program: &Program) {
        if self.kernel_decls.is_empty() { return; }
        let mut all_fns: Vec<&FnDecl> = Vec::new();
        for item in &program.items { Self::gather_gpu_arg_fns_item(item, &mut all_fns); }

        let mut flags_by_fn: std::collections::HashMap<&str, Vec<bool>> = all_fns.iter()
            .map(|f| (f.name.as_str(), vec![false; f.params.len()]))
            .collect();

        // See `Checker::collect_gpu_arg_params` — this bound is never actually hit,
        // just a guard against a future logic error turning this into an infinite loop.
        let max_passes = all_fns.iter().map(|f| f.params.len()).sum::<usize>() + 2;

        for _ in 0..max_passes {
            let mut changed = false;
            for f in &all_fns {
                let mut new_flags = vec![false; f.params.len()];
                for (i, p) in f.params.iter().enumerate() {
                    let kernel_decls = &self.kernel_decls;
                    let known = &flags_by_fn;
                    let mut classify = |fn_name: &str, arg_idx: usize| -> bool {
                        if let Some(decl) = kernel_decls.get(fn_name) {
                            let Some(init) = decl.inits.first() else { return false };
                            let Some(init_param) = init.params.get(arg_idx) else { return false };
                            let Some(field_name) = kernel_init_field_for_param(decl, &init_param.name) else { return false };
                            let Some(field) = decl.fields.iter().find(|fd| fd.name == field_name) else { return false };
                            return matches!(field.qual, crate::ast::GpuQual::Unified | crate::ast::GpuQual::Global)
                                && matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _));
                        }
                        known.get(fn_name).and_then(|flags| flags.get(arg_idx).copied()).unwrap_or(false)
                    };
                    let (_any, only_qualifying) = crate::ast::scan_var_call_arg_uses(&f.body, &p.name, &mut classify);
                    new_flags[i] = only_qualifying;
                }
                if flags_by_fn.get(f.name.as_str()) != Some(&new_flags) {
                    changed = true;
                    flags_by_fn.insert(f.name.as_str(), new_flags);
                }
            }
            if !changed { break; }
        }

        for (name, flags) in flags_by_fn {
            if flags.iter().any(|b| *b) {
                self.fn_gpu_arg_params.insert(name.to_string(), flags);
            }
        }
    }

    fn gather_gpu_arg_fns_item<'a>(item: &'a Item, out: &mut Vec<&'a FnDecl>) {
        match item {
            Item::Fn(f)   => out.push(f),
            Item::Mod(m)  => { for i in &m.items { Self::gather_gpu_arg_fns_item(i, out); } }
            Item::Stmt(s) => Self::gather_gpu_arg_fns_stmt(s, out),
            _ => {}
        }
    }

    fn gather_gpu_arg_fns_stmt<'a>(stmt: &'a Stmt, out: &mut Vec<&'a FnDecl>) {
        match stmt {
            Stmt::Fn(f)     => out.push(f),
            Stmt::Mod(m)    => { for i in &m.items { Self::gather_gpu_arg_fns_item(i, out); } }
            Stmt::If(s)     => { for (_, b) in &s.branches { for st in b { Self::gather_gpu_arg_fns_stmt(st, out); } } if let Some(b) = &s.else_body { for st in b { Self::gather_gpu_arg_fns_stmt(st, out); } } }
            Stmt::While(s)  => { for st in &s.body { Self::gather_gpu_arg_fns_stmt(st, out); } }
            Stmt::For(s)    => { for st in &s.body { Self::gather_gpu_arg_fns_stmt(st, out); } }
            Stmt::Loop(s)   => { for st in &s.body { Self::gather_gpu_arg_fns_stmt(st, out); } }
            Stmt::DoWhile(s) => { for st in &s.body { Self::gather_gpu_arg_fns_stmt(st, out); } }
            Stmt::Try(s)    => { for st in &s.body { Self::gather_gpu_arg_fns_stmt(st, out); } for c in &s.catch_clauses { for st in &c.body { Self::gather_gpu_arg_fns_stmt(st, out); } } }
            Stmt::Guard(s)  => { for st in &s.else_body { Self::gather_gpu_arg_fns_stmt(st, out); } }
            Stmt::Defer(b)  => { for st in b { Self::gather_gpu_arg_fns_stmt(st, out); } }
            Stmt::With(s)   => { for st in &s.body { Self::gather_gpu_arg_fns_stmt(st, out); } }
            _ => {}
        }
    }

    /// `pre_scan`'s per-struct registration: struct fields (incl. body-less-init params),
    /// callable-struct/init-body/init-default tracking, associated types, type vars/type
    /// methods, getters/setters, task/throws/return-type method tracking, overloaded-method
    /// detection, actor/rwlock/arc-qualified field tracking, iterator-protocol detection,
    /// transient fields, `as T:` conversion targets, and (last, since it depends on
    /// `struct_req_methods`) field-qualifier inference from method-body usage.
    fn pre_scan_struct_item(
        &mut self,
        s: &crate::ast::StructDecl,
        ext_methods_by_type: &std::collections::HashMap<&str, Vec<&crate::ast::FnDecl>>,
        ext_setters_by_type: &std::collections::HashMap<&str, Vec<&crate::ast::SetDecl>>,
    ) {
        self.struct_protocols.insert(s.name.clone(), s.protocols.clone());
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
        let has_qualified = fields.iter().any(|(_, ty)| Self::field_type_has_qualifier(ty));
        if has_qualified { self.qualified_struct_types.insert(s.name.clone()); }
        self.struct_fields.insert(s.name.clone(), fields);
        for f in &s.fields {
            if let Some(def) = &f.default {
                self.struct_field_defaults.insert(format!("{}::{}", s.name, f.name), def.clone());
            }
        }
        if s.methods.iter().any(|m| m.name.is_empty()) {
            self.callable_structs.insert(s.name.clone());
        }
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
            // Register throwing type-level methods the same way instance methods are
            // below -- gates the `?` propagation `try_emit_type_method_call` adds at
            // `TypeName.method(...)` call sites (emit_methods.rs). Qualified key
            // ("StructName::method") avoids a same-named-but-non-throwing type method
            // on a different struct picking up a stray `?` from the bare-name entry.
            if tm.throws {
                self.struct_method_throws.insert(format!("{}::{}", s.name, tm.name));
                self.struct_method_throws.insert(tm.name.clone());
            }
            // Typed `throws ErrType:` on a type-level method registers ErrType the same
            // way a regular throwing FnDecl's throws_ty does (line ~3283 below) -- so
            // emit_enum generates its Display/Error impls and emit_throw routes it
            // through BoringError::Other instead of stringifying it via Display.
            if let Some(Type::Named(err_name)) = &tm.throws_ty {
                self.typed_error_enums.insert(err_name.clone());
            }
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
            if m.throws {
                // Qualified key (checked first at call sites when the receiver's
                // struct type is known) avoids a same-named-but-non-throwing
                // method on a different struct picking up a stray `?` -- e.g.
                // `EncoderBlock.forward` (no throws) vs `AudioEncoder.forward`
                // (throws) both being named "forward".
                self.struct_method_throws.insert(format!("{}::{}", s.name, m.name));
                self.struct_method_throws.insert(m.name.clone());
            }
            // Track return types for already_opt detection in emit_stmt.
            if let Some(ret_ty) = &m.return_ty {
                let key = format!("{}::{}", s.name, m.name);
                self.struct_method_return_types.insert(key, ret_ty.clone());
            }
        }
        // Detect overloaded inline methods — same logic as ext blocks.
        for m in &s.methods {
            let method_key = format!("{}::{}", s.name, m.name);
            let (new_errors, overloaded) = {
                let method_variants = self.struct_method_overload_decls
                    .entry(method_key.clone())
                    .or_default();
                let this_mangled = mangle_overload_name(&m.name, &m.params);
                let already_registered = method_variants.iter()
                    .any(|v| mangle_overload_name(&v.name, &v.params) == this_mangled);
                let new_errors: Vec<_> = if !already_registered {
                    let errs = method_variants.iter()
                        .filter_map(|existing| overloads_conflict(existing, m).map(|n| (m.line, m.col, format!(
                            "ambiguous overload for method '{}::{}' — \
                             both match a call with {} argument(s)",
                            s.name, m.name, n
                        ))))
                        .collect();
                    method_variants.push(m.clone());
                    errs
                } else { Vec::new() };
                (new_errors, method_variants.len() > 1)
            };
            for (line, col, msg) in new_errors { self.push_error(line, col, msg); }
            if overloaded { self.overloaded_method_keys.insert(method_key); }
        }
        // Track T'actor / T'actor'task struct fields → Arc<Mutex<T>> or Arc<tokio::sync::Mutex<T>>.
        for f in &s.fields {
            if Self::is_mutex_binding(f.mutable, &f.ty) {
                let key = format!("{}::{}", s.name, f.name);
                if Self::is_mutex_task_binding(f.mutable, &f.ty) {
                    self.struct_mutex_task_fields.insert(key);
                } else {
                    self.struct_mutex_fields.insert(key);
                }
            }
            if Self::is_rwlock_binding(f.mutable, &f.ty) {
                let key = format!("{}::{}", s.name, f.name);
                if Self::is_rwlock_task_binding(f.mutable, &f.ty) {
                    self.struct_rwlock_task_fields.insert(key);
                } else {
                    self.struct_rwlock_fields.insert(key);
                }
            }
            // Note: infer_struct_field_qualifiers runs after this loop and may add
            // further entries to struct_mutex_fields / struct_rwlock_fields.
            // Collect arc-qualified inner type names for task fn method validation.
            if let Some(n) = Self::arc_inner_type_name(&f.ty) {
                self.arc_qualified_types.insert(n.to_string());
            }
        }
        // Collect arc-qualified types from method params too, and from local
        // `let`/`var` bindings inside each method body (see
        // `collect_arc_qualified_lets_stmts`'s doc comment).
        for m in &s.methods {
            for p in &m.params {
                if let Some(ty) = &p.ty {
                    if let Some(n) = Self::arc_inner_type_name(ty) {
                        self.arc_qualified_types.insert(n.to_string());
                    }
                }
            }
            self.collect_arc_qualified_lets_stmts(&m.body);
        }
        // Track req (non-mutating) methods for 'guard read vs write dispatch.
        for m in &s.methods {
            if !m.mutating {
                self.struct_req_methods.insert(format!("{}::{}", s.name, m.name));
            }
            if m.task {
                self.struct_task_methods.insert(format!("{}::{}", s.name, m.name));
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
        // Infer qualifiers for private unqualified fields from method body usage.
        // Must run after struct_req_methods is populated (used for def/req detection).
        // Also include same-file `ext` block methods/setters for this type.
        let empty_methods: Vec<&crate::ast::FnDecl> = Vec::new();
        let empty_setters: Vec<&crate::ast::SetDecl> = Vec::new();
        let ext_methods = ext_methods_by_type.get(s.name.as_str()).unwrap_or(&empty_methods);
        let ext_setters = ext_setters_by_type.get(s.name.as_str()).unwrap_or(&empty_setters);
        self.infer_struct_field_qualifiers(s, ext_methods, ext_setters);
    }

    /// `pre_scan`'s per-`ext`-block registration: arc-qualified param types, `as T:`
    /// conversion targets, operator-method detection (so `emit_struct` skips deriving
    /// `PartialEq` when a custom impl will be generated), getters/return-types/throws/
    /// overloaded-method tracking (mirroring inline struct methods), setters, and
    /// no-protocol method-override tracking.
    fn pre_scan_ext_item(&mut self, e: &crate::ast::ExtDecl) {
        // Collect arc-qualified types from ext method params, and from local
        // `let`/`var` bindings inside each method body (see
        // `collect_arc_qualified_lets_stmts`'s doc comment).
        for m in &e.methods {
            for p in &m.params {
                if let Some(ty) = &p.ty {
                    if let Some(n) = Self::arc_inner_type_name(ty) {
                        self.arc_qualified_types.insert(n.to_string());
                    }
                }
            }
            self.collect_arc_qualified_lets_stmts(&m.body);
        }
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
            // Track req (non-mutating) ext methods too — `s.methods`'s own loop
            // above only sees the struct's own body, missing `ext`-added methods
            // entirely (a pre-existing gap: the param-mutability diagnostic and
            // the equivalent local-binding one both consult this set to decide
            // "is this call actually mutating," not just getters).
            if !m.mutating {
                self.struct_req_methods.insert(format!("{}::{}", tname, m.name));
            }
            // Track return types for managed-mode inference.
            if let Some(ret_ty) = &m.return_ty {
                self.struct_method_return_types.insert(
                    format!("{}::{}", tname, m.name), ret_ty.clone());
            }
            if m.throws {
                // Qualified key (checked first at call sites when the receiver's
                // struct type is known) avoids a same-named-but-non-throwing
                // method on a different struct picking up a stray `?` -- see the
                // bare-name insert's doc comment.
                self.struct_method_throws.insert(format!("{}::{}", tname, m.name));
                self.struct_method_throws.insert(m.name.clone());
            }
            // Track overloaded struct methods (same name, different params).
            let method_key = format!("{}::{}", tname, m.name);
            let (new_errors, overloaded) = {
                let method_variants = self.struct_method_overload_decls.entry(method_key.clone()).or_default();
                let this_mangled = mangle_overload_name(&m.name, &m.params);
                let already_registered = method_variants.iter()
                    .any(|v| mangle_overload_name(&v.name, &v.params) == this_mangled);
                let new_errors: Vec<_> = if !already_registered {
                    let errs = method_variants.iter()
                        .filter_map(|existing| overloads_conflict(existing, m).map(|n| (m.line, m.col, format!(
                            "ambiguous overload for method '{}::{}' — both match a call with {} argument(s)",
                            tname, m.name, n
                        ))))
                        .collect();
                    method_variants.push(m.clone());
                    errs
                } else { Vec::new() };
                (new_errors, method_variants.len() > 1)
            };
            for (line, col, msg) in new_errors { self.push_error(line, col, msg); }
            if overloaded { self.overloaded_method_keys.insert(method_key); }
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

    /// Runs before all other `pre_scan` work so that `unit_enums`/`recursive_fields` are
    /// populated when `emit_struct`/`emit_enum` read them: builds a non-heap direct-children
    /// map per type, computes (via BFS) which types transitively reach each type without
    /// crossing a heap-backed container, then uses that to mark recursive enum variants/
    /// struct fields for `Box` wrapping and (strict mode only) emit stack-size warnings.
    fn pre_scan_infer_recursive_and_size_warnings(&mut self, program: &Program) {
        // Build non-heap direct-children map for transitive cycle detection.
        // direct_children[A] = {B, C, …} means type A has a non-heap field of type B or C.
        // Array/Dict/Set/Qualified are heap-backed so they break infinite-size cycles.
        let mut direct_children: std::collections::HashMap<String, std::collections::HashSet<String>> =
            std::collections::HashMap::new();
        fn collect_named(ty: &Type, out: &mut std::collections::HashSet<String>) {
            match ty {
                Type::Named(n) => { out.insert(n.clone()); }
                Type::Array(_) | Type::Dict(_, _) | Type::Set(_) | Type::Qualified(_, _) => {}
                Type::Optional(inner) | Type::Dyn(inner) | Type::Impl(inner) => collect_named(inner, out),
                Type::Generic(n, args) => {
                    out.insert(n.clone());
                    for a in args { collect_named(a, out); }
                }
                _ => {}
            }
        }
        for item in &program.items {
            match item {
                Item::Struct(s) => {
                    let mut children = std::collections::HashSet::new();
                    for f in &s.fields { collect_named(&f.ty, &mut children); }
                    direct_children.insert(s.name.clone(), children);
                }
                Item::Enum(e) => {
                    let mut children = std::collections::HashSet::new();
                    for v in &e.variants {
                        for f in &v.fields { collect_named(&f.ty, &mut children); }
                    }
                    direct_children.insert(e.name.clone(), children);
                }
                _ => {}
            }
        }
        // For each type T, compute which types can reach T without going through a heap container.
        // reachable_to[T] = set of types S such that S ──non-heap-path──> T
        let all_type_names: Vec<String> = direct_children.keys().cloned().collect();
        let mut reachable_to: std::collections::HashMap<String, std::collections::HashSet<String>> =
            std::collections::HashMap::new();
        for target in &all_type_names {
            // BFS: find all types that can reach `target`
            let mut reaches = std::collections::HashSet::new();
            let mut worklist = vec![target.clone()];
            while let Some(t) = worklist.pop() {
                for (src, children) in &direct_children {
                    if children.contains(&t) && !reaches.contains(src) {
                        reaches.insert(src.clone());
                        worklist.push(src.clone());
                    }
                }
            }
            reachable_to.insert(target.clone(), reaches);
        }

        for item in &program.items {
            match item {
                Item::Enum(e) => {
                    // Priority 2: non-parametric enum → infer T'copy.
                    if e.variants.iter().all(|v| v.fields.is_empty()) {
                        self.unit_enums.insert(e.name.clone());
                    }
                    // Priority 3: recursive enum variants → mark for Box wrapping.
                    // Use transitive cycle detection to catch mutual recursion (e.g. ExprKind↔Expr).
                    let reaches_back = reachable_to.get(&e.name)
                        .cloned().unwrap_or_default();
                    for v in &e.variants {
                        for (idx, f) in v.fields.iter().enumerate() {
                            if Self::type_references_transitive(&f.ty, &e.name, &reaches_back) {
                                let key = format!("{}::{}::{}", e.name, v.name, idx);
                                self.recursive_fields.insert(key);
                            }
                        }
                    }
                    // Size-based warning for enums (strict mode only).
                    if self.config.mode == TranspileMode::Strict && !self.unit_enums.contains(&e.name) {
                        let auto_bytes = self.config.inline_auto_bytes;
                        let variant_sizes: Vec<Option<usize>> = e.variants.iter().map(|v| {
                            v.fields.iter().try_fold(0usize, |acc, f| {
                                Self::estimate_size(&f.ty, program).map(|s| acc + s)
                            })
                        }).collect();
                        let known_sizes: Vec<usize> = variant_sizes.iter().filter_map(|s| *s).collect();
                        if !known_sizes.is_empty() {
                            let max_size = *known_sizes.iter().max().unwrap();
                            let total_size = max_size + 8; // discriminant
                            if total_size > auto_bytes {
                                let largest = e.variants.iter().zip(variant_sizes.iter())
                                    .filter_map(|(v, s)| s.map(|sz| (v.name.as_str(), sz)))
                                    .max_by_key(|(_, sz)| *sz)
                                    .map(|(n, _)| n).unwrap_or("?");
                                self.push_warning(e.line, e.col, format!("`{}` is {} bytes inline (largest variant: `{}`); consider `{}'owned` to heap-allocate", e.name, total_size, largest, e.name));
                            }
                            // Disproportionate variant warning: one variant dominates the median.
                            if known_sizes.len() > 1 {
                                let median = { let mut s = known_sizes.clone(); s.sort(); s[s.len()/2] };
                                if let Some((dom_name, dom_size, dom_variant)) = e.variants.iter().zip(variant_sizes.iter())
                                    .filter_map(|(v, s)| s.map(|sz| (v.name.as_str(), sz, v)))
                                    .find(|(_, sz, _)| *sz > median * 2 && *sz > auto_bytes / 4)
                                {
                                    let field_ty_s = if dom_variant.fields.len() == 1 {
                                        let ft = self.emit_type(&dom_variant.fields[0].ty);
                                        format!("{}({}'owned)", dom_name, ft)
                                    } else {
                                        let fts: Vec<String> = dom_variant.fields.iter()
                                            .map(|f| format!("{}'owned", self.emit_type(&f.ty)))
                                            .collect();
                                        format!("{}({})", dom_name, fts.join(", "))
                                    };
                                    self.push_warning(e.line, e.col, format!("variant `{}` ({} bytes) dominates `{}` ({} bytes median); consider boxing the payload: {}", dom_name, dom_size, e.name, median, field_ty_s));
                                }
                            }
                        }
                    }
                }
                Item::Struct(s) => {
                    // Priority 3: recursive struct fields → mark for Box wrapping (direct only).
                    // We do NOT use transitive detection here — boxing a struct field indirectly
                    // breaks all field-access patterns (expr.kind → (*expr.kind) everywhere).
                    // Mutual recursion between structs and enums is broken by boxing the enum side.
                    for f in &s.fields {
                        if Self::type_references(&f.ty, &s.name) {
                            let key = format!("{}::{}", s.name, f.name);
                            self.recursive_fields.insert(key);
                        }
                    }
                    // Size-based warning (strict mode only).
                    if self.config.mode == TranspileMode::Strict {
                        let auto_bytes = self.config.inline_auto_bytes;
                        if let Some(size) = Self::estimate_size(&Type::Named(s.name.clone()), program) {
                            if size > auto_bytes {
                                self.push_warning(s.line, s.col, format!("`{}` is {} bytes inline; consider `{}'owned` to heap-allocate", s.name, size, s.name));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn pre_scan(&mut self, program: &Program) {
        // Must run before any struct is emitted — `emit_struct` consults this set
        // to decide whether to add `Default` to a struct's derive list.
        helpers::collect_default_rest_targets(&program.items, &mut self.structs_needing_default);

        // Pre-populate the stdlib `Error` enum so it's always available without a user declaration.
        self.typed_error_enums.insert("Error".to_string());
        for variant in &["Expired", "Cancelled", "NotFound", "InvalidInput", "OutOfBounds"] {
            self.enum_variants.insert(variant.to_string(), "Error".to_string());
            let key = format!("Error::{}", variant);
            self.enum_variant_fields.insert(key.clone(), vec![]);
            self.enum_variant_field_types.insert(key, vec![]);
        }

        // ── Collect user-defined type names (for managed mode wrapping) ────────
        // Run this as its own pass, before the `Item::Let` classification below, so a
        // struct/enum declared LATER in source order than a `let` that constructs it is
        // still in `user_types` by the time `top_level_let_is_promotable` asks "is this
        // call's receiver one of the file's own types, or an external/opaque one?" --
        // a single interleaved pass would wrongly call such a call "external" for any
        // `let` appearing above its own type's declaration.
        for item in &program.items {
            match item {
                Item::Struct(s) => { self.user_types.insert(s.name.clone()); }
                Item::Enum(e)   => { self.user_types.insert(e.name.clone()); }
                _ => {}
            }
        }
        // Candidate top-level immutable `let` constants promotable to module scope (per
        // `top_level_let_is_promotable`) — whether each one actually gets promoted to a
        // Rust `const`/`static` is decided below, once cross-scope usage is known (see
        // `global_lets_used_elsewhere`).
        let mut const_let_candidates: std::collections::HashSet<String> = std::collections::HashSet::new();
        for item in &program.items {
            if let Item::Let(s) = item {
                self.user_top_level_names.insert(s.name.clone());
                // Populated here (before any emission) rather than inline where the
                // `const` is actually emitted, so a function appearing earlier in
                // source order than the `let` it references still sees the name in
                // this set -- Rust doesn't care about const declaration order, but
                // this scan does, being a single pass over `program.items`.
                // Despite the field's `gpu_`-prefixed name (kept to minimize the diff —
                // it started GPU-only), this now applies to every target: any promoted
                // top-level scalar `let` is emitted as an UPPERCASE Rust `const` (see
                // `emit_item`'s `Item::Let` arm) specifically to avoid colliding with a
                // same-named local variable or function parameter elsewhere in the file
                // — a fn param/pattern binding can't shadow a Rust `const` (E0530 /
                // E0308 "interpreted as a constant, not a new binding"). This used to
                // be gated on `is_gpu_target`, which was never the real boundary: a
                // plain CPU program with `let n = 255` at the top and an unrelated local
                // `n` anywhere else (e.g. `fn countdown(n: isize)`) hit the exact same
                // Rust restriction — confirmed via examples/hello.br.
                if self.top_level_let_is_const_safe(s) {
                    self.gpu_top_level_const_names.insert(s.name.clone());
                }
                if matches!(s.binding, crate::ast::BindingKind::Let) && self.top_level_let_is_promotable(s) {
                    const_let_candidates.insert(s.name.clone());
                }
            }
        }

        self.pre_scan_infer_recursive_and_size_warnings(program);

        // ── Build type_sizes cache for auto-boxing in emit_type ─────────────────
        for item in &program.items {
            match item {
                Item::Struct(s) => {
                    self.all_struct_types.insert(s.name.clone());
                    if let Some(size) = Self::estimate_size(&Type::Named(s.name.clone()), program) {
                        self.type_sizes.insert(s.name.clone(), size);
                    }
                }
                Item::Enum(e) if !self.unit_enums.contains(&e.name) => {
                    if let Some(size) = Self::estimate_size(&Type::Named(e.name.clone()), program) {
                        self.type_sizes.insert(e.name.clone(), size);
                    }
                }
                _ => {}
            }
        }
        for item in &program.items {
            if let Item::Enum(e) = item {
                self.all_enum_types.insert(e.name.clone());
            }
        }

        // Index `ext` block methods/setters by type name up front (order-independent), so
        // struct field qualifier inference below can see ext methods regardless of whether
        // the `ext` block appears before or after the `struct` in this file.
        let mut ext_methods_by_type: std::collections::HashMap<&str, Vec<&crate::ast::FnDecl>> =
            std::collections::HashMap::new();
        let mut ext_setters_by_type: std::collections::HashMap<&str, Vec<&crate::ast::SetDecl>> =
            std::collections::HashMap::new();
        for item in &program.items {
            if let Item::Ext(e) = item {
                ext_methods_by_type.entry(e.type_name.as_str()).or_default().extend(&e.methods);
                ext_setters_by_type.entry(e.type_name.as_str()).or_default().extend(&e.setters);
            }
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
                    // Track req (non-mutating) / task methods, same as structs
                    // (pre_scan_struct_item, mirrored above) — needed so
                    // `method_is_req_or_task` answers correctly for enum methods too,
                    // not just struct ones. Most enums never consult this (`def`==`req`
                    // there, docs/book.md's "Enum methods" section), but an enum with a
                    // `mut`-qualified variant field (docs/mut-type-modifier.md) now gets
                    // the same non-mut-binding `def`-call diagnostic as a struct — see
                    // `enum_has_mut_field`'s callers — and that diagnostic needs to tell
                    // a real `req` getter apart from a mutating `def` on such an enum.
                    for m in &e.methods {
                        if !m.mutating {
                            self.struct_req_methods.insert(format!("{}::{}", e.name, m.name));
                        }
                        if m.task {
                            self.struct_task_methods.insert(format!("{}::{}", e.name, m.name));
                        }
                    }
                    // Track return types for already_opt detection (the Some(...)-wrap
                    // guards in emit_stmt/emit_flow/emit_expr/emit_let) — mirrors the
                    // identical struct-method registration in `pre_scan_struct_item`.
                    // Without this, `struct_method_return_types` never had an entry for
                    // an enum method at all, so a `T?`-declared enum `req` method (e.g.
                    // a tagged-union enum's `as_str()`) was never recognized as
                    // already-Optional by any of those checks, even after they learned
                    // to consult the map for a non-`self`/non-bare-var receiver — the
                    // map itself was just missing every enum method's entry. See
                    // docs/option-return-double-some-wrap-bug.md.
                    for m in &e.methods {
                        if let Some(ret_ty) = &m.return_ty {
                            let key = format!("{}::{}", e.name, m.name);
                            self.struct_method_return_types.insert(key, ret_ty.clone());
                        }
                    }
                    // Register enum `as T:` conversion targets.
                    for conv in &e.conversions {
                        let tname = self.emit_type(&conv.ty);
                        self.user_conv_targets.insert(tname.to_lowercase());
                        // `as string:` emits a Display impl — mark the type so the
                        // general auto-Display emitted by `emit_enum` is skipped
                        // (mirrors the identical struct/ext registration above).
                        if Self::is_string_conversion(conv) {
                            self.display_types.insert(e.name.clone());
                        }
                    }
                    // Register type vars and type methods for emission dispatch — mirrors
                    // `pre_scan_struct_item`'s identical registration for a struct's
                    // `type_methods` (see that function's own comments for the rationale on
                    // each map). Enums have no `type var`/`type let` (out of scope for now),
                    // so only `type_methods` is registered here, unlike the struct side
                    // which also handles type vars.
                    let mut method_map = std::collections::HashMap::new();
                    for tm in &e.type_methods {
                        method_map.insert(tm.name.clone(), tm.kind.clone());
                        if tm.throws {
                            self.struct_method_throws.insert(format!("{}::{}", e.name, tm.name));
                            self.struct_method_throws.insert(tm.name.clone());
                        }
                        if let Some(Type::Named(err_name)) = &tm.throws_ty {
                            self.typed_error_enums.insert(err_name.clone());
                        }
                    }
                    if !method_map.is_empty() {
                        self.struct_type_method_sigs.insert(e.name.clone(), method_map);
                    }
                }
                Item::Fn(f) => {
                    // Collect arc-qualified types from top-level function params.
                    for p in &f.params {
                        if let Some(ty) = &p.ty {
                            if let Some(n) = Self::arc_inner_type_name(ty) {
                                self.arc_qualified_types.insert(n.to_string());
                            }
                        }
                    }
                    // ...and from local `let`/`var` bindings inside the body — see
                    // `collect_arc_qualified_lets_stmts`'s doc comment.
                    self.collect_arc_qualified_lets_stmts(&f.body);
                    // Signature-only var-param positions, for the `with` mutation scan.
                    self.fn_var_params.insert(f.name.clone(), f.params.iter().map(|p| p.rebindable).collect());
                    // Interprocedural GPU residency (docs/scoped-access-blocks.md) — mirrors
                    // the checker's own `fn_returns_resident`/`fn_gpu_arg_params`
                    // (checker/mod.rs), recomputed independently here since the transpiler
                    // doesn't share checker state (same existing pattern as `kernel_decls`,
                    // tracked separately on both sides).
                    if let Some(rt) = &f.return_ty {
                        if rt.gpu_resident_qual().is_some() {
                            self.fn_returns_resident.insert(f.name.clone(), rt.clone());
                        } else if let Type::Tuple(elems) = rt {
                            let flags: Vec<bool> = elems.iter().map(|t| t.gpu_resident_qual().is_some()).collect();
                            if flags.iter().any(|b| *b) {
                                self.fn_returns_resident_tuple.insert(f.name.clone(), flags);
                            }
                        }
                    }
                    // `fn_gpu_arg_params` itself is computed once, up front, over the
                    // whole program by `compute_gpu_arg_params` (called from
                    // `emit_program` before `pre_scan` even starts) — it needs the
                    // full call graph in one shot for its transitive fixed point, which
                    // this per-item, per-recursive-module-scope walk can't give it.
                    self.pre_register_fn(f);
                }
                Item::Struct(s) if s.name == "Box" => {
                    // A user-defined struct literally named `Box` (distinct from
                    // Rust's own `std::boxed::Box`, hence `user_defines_box`
                    // forcing `std::boxed::Box` everywhere `T'` would otherwise
                    // emit bare `Box`) — still needs the SAME full registration
                    // as any other struct (methods, `struct_req_methods`,
                    // `struct_protocols`, ext blocks, ...). This used to
                    // duplicate just the field-collection subset of
                    // `pre_scan_struct_item` inline and skip the rest — a real,
                    // narrow gap (e.g. `var Box b; b.some_req_method()` was
                    // never recognized as req, always assumed mutating) that
                    // predated this document but was silent until the new
                    // local-binding content-mutation diagnostic below started
                    // actually consulting `struct_req_methods`.
                    self.user_defines_box = true;
                    self.pre_scan_struct_item(s, &ext_methods_by_type, &ext_setters_by_type);
                    if s.methods.iter().any(|m| m.name.is_empty()) {
                        self.callable_structs.insert(s.name.clone());
                    }
                }
                Item::Struct(s) => {
                    self.pre_scan_struct_item(s, &ext_methods_by_type, &ext_setters_by_type);
                }
                Item::Ext(e) => {
                    self.pre_scan_ext_item(e);
                }
                Item::Trait(t) => {
                    let mut names = std::collections::HashSet::new();
                    for sig in &t.signatures { names.insert(sig.name.clone()); }
                    for d   in &t.defaults   { names.insert(d.name.clone()); }
                    self.trait_method_names.insert(t.name.clone(), names);
                    let default_mutating: std::collections::HashMap<String, bool> = t.defaults.iter()
                        .map(|d| (d.name.clone(), d.mutating && !d.task))
                        .collect();
                    self.trait_default_mutating.insert(t.name.clone(), default_mutating);
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
                Item::Let(s) => {
                    // Collect arc-qualified types from all top-level let/var bindings.
                    if let Some(ty) = &s.ty {
                        if let Some(n) = Self::arc_inner_type_name(ty) {
                            self.arc_qualified_types.insert(n.to_string());
                        }
                    }
                    if s.binding.is_mutable() {
                        // Top-level mutable var declarations — collect type and initial value.
                        let init_val = self.emit_expr_owned(s.value.as_ref().unwrap());
                        self.global_var_types.insert(s.name.clone(), s.ty.clone());
                        self.global_var_inits.insert(s.name.clone(), init_val);
                    }
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

        // Third pass: which top-level *immutable* `let` scalar constants
        // (`const_let_candidates`, collected above) are referenced from inside a
        // function body OR a struct/enum/ext method body. Those can't stay a local
        // inside the synthesized `fn main()` -- the referencing function/method has no
        // way to see it -- see `global_lets_used_elsewhere`'s doc comment for how this
        // result is used. Unlike `top_var_names` above (mutable top-level `var`s,
        // `Item::Fn` bodies only), this must also cover struct/enum/ext methods: an
        // enum `req`/`def` method body referencing a top-level constant is exactly the
        // shape that motivated this scan.
        for item in &program.items {
            match item {
                Item::Fn(f) => collect_const_let_usage(&const_let_candidates, &f.params, &f.body, &mut self.global_lets_used_elsewhere),
                Item::Struct(s) => for m in &s.methods {
                    collect_const_let_usage(&const_let_candidates, &m.params, &m.body, &mut self.global_lets_used_elsewhere);
                },
                Item::Enum(e) => for m in &e.methods {
                    collect_const_let_usage(&const_let_candidates, &m.params, &m.body, &mut self.global_lets_used_elsewhere);
                },
                Item::Ext(x) => for m in &x.methods {
                    collect_const_let_usage(&const_let_candidates, &m.params, &m.body, &mut self.global_lets_used_elsewhere);
                },
                _ => {}
            }
        }

        // Fourth pass: which top-level `let` string-literal constants
        // (`top_level_let_is_string_literal`) will actually be promoted to a module-level
        // `static` -- same gate `emit_program_items`'s `Item::Let` classification uses:
        // unconditionally for `pub`, or gated on `global_lets_used_elsewhere` (just
        // finished above) otherwise. Recorded so `emit_fn` can re-seed `string_vars`/
        // `string_arc_vars` with these names at the start of every function/method body --
        // see `global_string_const_names`'s doc comment for why that's needed at all.
        for item in &program.items {
            if let Item::Let(s) = item {
                if self.top_level_let_is_string_literal(s)
                    && (s.is_pub || self.global_lets_used_elsewhere.contains(&s.name))
                {
                    self.global_string_const_names.insert(s.name.clone());
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
                Item::Let(l) => { if let Some(v) = &l.value { collect_is_identity_vars(v, &type_names, &mut identity_vars); } }
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

        // Any enum ever directly `throw`n -- regardless of whether the throwing
        // function declares a typed `throws EnumName:` -- gets consistent typed-error
        // treatment (Display+Error impls, `BoringError::Other` routing at the throw
        // site instead of a Display-stringifying `BoringError::String`). Without
        // this, an enum thrown ONLY from bare `throws:` functions silently loses its
        // type end-to-end: `catch EnumName.Variant:` elsewhere can never downcast back
        // out of the stringified error, and instead of a compile-time signal it just
        // panics at runtime as an "unhandled error". See
        // `helpers::collect_thrown_enum_names`'s doc comment for the full mechanism
        // and `tests/cases/throws_untyped_enum_catch.br` for the regression case.
        let mut thrown_enum_names = std::collections::HashSet::new();
        helpers::collect_thrown_enum_names(program, &self.all_enum_types, &self.enum_variants, &mut thrown_enum_names);
        for name in thrown_enum_names {
            self.typed_error_enums.insert(name);
        }
    }

    fn pre_register_fn(&mut self, f: &FnDecl) {
        // Track overloads: if this name was already registered with a DIFFERENT mangled
        // signature, it's a genuine overload. Same mangled name = redefinition (ignore).
        let this_mangled = mangle_overload_name(&f.name, &f.params);
        let (new_errors, overloaded) = {
            let variants = self.fn_overload_decls.entry(f.name.clone()).or_default();
            let already_has_this_sig = variants.iter().any(|v| {
                mangle_overload_name(&v.name, &v.params) == this_mangled
            });
            let new_errors: Vec<_> = if !already_has_this_sig {
                // Before accepting the new variant, check for ambiguous conflicts with existing ones.
                let b_sig = f.params.iter()
                    .map(|p| {
                        let ty = p.ty.as_ref().map(mangle_type_name)
                            .unwrap_or_else(|| "_".into());
                        if p.default.is_some() { format!("{}=?", ty) } else { ty }
                    })
                    .collect::<Vec<_>>().join(", ");
                let errs = variants.iter()
                    .filter_map(|existing| overloads_conflict(existing, f).map(|conflict_arity| {
                        let a_sig = existing.params.iter()
                            .map(|p| {
                                let ty = p.ty.as_ref().map(mangle_type_name)
                                    .unwrap_or_else(|| "_".into());
                                if p.default.is_some() { format!("{}=?", ty) } else { ty }
                            })
                            .collect::<Vec<_>>().join(", ");
                        (f.line, f.col, format!(
                            "ambiguous overload for '{}' — \
                             '{}({})' and '{}({})' both match a call with {} argument(s)",
                            f.name, f.name, b_sig, existing.name, a_sig, conflict_arity
                        ))
                    }))
                    .collect();
                variants.push(f.clone());
                errs
            } else { Vec::new() };
            (new_errors, variants.len() > 1)
        };
        for (line, col, msg) in new_errors { self.push_error(line, col, msg); }
        if overloaded { self.overloaded_fn_names.insert(f.name.clone()); }

        let param_types: Vec<Type> = f.params.iter().filter_map(|p| p.ty.clone()).collect();
        let defaults: Vec<Option<String>> = f.params.iter().map(|p| {
            p.default.as_ref().map(|d| self.emit_expr_owned(d))
        }).collect();
        // Register under both the plain name (for non-overloaded path) and the mangled name
        // (for overloaded dispatch, so emit_args_coerced can find the correct param types).
        // `or_insert` (not a blind overwrite): pre_register_fn re-runs every time a `use`d
        // file is re-parsed and re-emitted via `inline_boring_use` (emit_program calls
        // pre_scan unconditionally on entry), which would otherwise stomp the qualifier
        // inference `pre_infer_fn_qualifiers` already propagated into this exact entry —
        // silently reverting e.g. `&Vec<T>` params back to the raw unqualified `Vec<T>`
        // for every caller emitted after that point. The raw param_types are identical
        // across re-registrations of the same declaration, so this is a no-op otherwise.
        self.fn_sigs.entry(f.name.clone()).or_insert_with(|| param_types.clone());
        self.fn_sigs.entry(this_mangled.clone()).or_insert(param_types);
        self.fn_defaults.insert(f.name.clone(), defaults.clone());
        self.fn_defaults.insert(this_mangled, defaults);
        let rebindable_flags: Vec<bool> = f.params.iter().map(|p| p.rebindable).collect();
        self.fn_rebindable.entry(f.name.clone()).or_insert(rebindable_flags);
        let mutable_flags: Vec<bool> = f.params.iter().map(|p| p.mutable).collect();
        self.fn_mutable.entry(f.name.clone()).or_insert(mutable_flags);
        if let Some(ret_ty) = &f.return_ty {
            // For overloaded functions, a non-void definition should not be overwritten by
            // a void one (e.g. exec_stmt: Signal? beats exec_stmt: void for already_opt detection).
            let existing = self.fn_return_types.get(f.name.as_str());
            let overwrite = !matches!((existing, ret_ty), (Some(existing_ty), Type::Void) if !matches!(existing_ty, Type::Void));
            if overwrite {
                self.fn_return_types.insert(f.name.clone(), ret_ty.clone());
            }
            // If this function returns T'actor or T'guard, register T as an "actor source type".
            // Bare T parameters of that type will then default to 'actor during qualifier inference.
            if let Type::Qualified(inner, OwnerQual::Actor | OwnerQual::Guard) = ret_ty {
                if let Type::Named(n) = inner.as_ref() {
                    self.actor_source_types.insert(n.clone());
                }
            }
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
            let sequential = crate::transpiler::helpers::body_is_sequential(&f.body, &self.task_fns);
            if sequential {
                self.stream_iter_fns.insert(f.name.clone());
            } else {
                self.stream_fns.insert(f.name.clone());
                self.has_streams = true;
                if f.throws {
                    self.stream_throws_fns.insert(f.name.clone());
                }
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

// ─── Pre-scan helpers ─────────────────────────────────────────────────────────

/// Which field a kernel's `init` assigns a given parameter name to (`field = param`,
/// the only pattern kernel-constructor codegen understands). Mirrors
/// `Checker::kernel_init_field_for_param` (checker/mod.rs) — duplicated rather than
/// shared since the transpiler doesn't depend on the checker (same existing pattern
/// as `kernel_decls`, tracked independently on both sides). Not reusing
/// `emit_kernel::kernel_param_to_field_map` (private to that module, builds a whole
/// map rather than a single lookup) to avoid a cross-module visibility change for
/// what's a small, self-contained scan.
fn kernel_init_field_for_param<'a>(decl: &'a crate::ast::KernelDecl, param_name: &str) -> Option<&'a str> {
    let init = decl.inits.first()?;
    for stmt in &init.body {
        if let crate::ast::Stmt::Expr(e) = stmt {
            if let ExprKind::Assign(lhs, rhs) = &e.kind {
                if let (ExprKind::Var(field), ExprKind::Var(param)) = (&lhs.kind, &rhs.kind) {
                    if param == param_name { return Some(field.as_str()); }
                }
            }
        }
    }
    None
}

fn program_uses_broadcast(program: &Program) -> bool {
    use crate::ast::{Item, Stmt, ExprKind};
    fn expr_is_broadcast(expr: &crate::ast::Expr) -> bool {
        matches!(&expr.kind,
            ExprKind::GenericCall(callee, _, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"))
        || matches!(&expr.kind,
            ExprKind::Call(callee, _)
            if matches!(&callee.kind, ExprKind::Var(n) if n == "broadcast"))
    }
    fn stmts_use(stmts: &[Stmt]) -> bool { stmts.iter().any(stmt_uses) }
    fn stmt_uses(s: &Stmt) -> bool {
        match s {
            Stmt::LetDestructure(s) => expr_is_broadcast(&s.value),
            Stmt::Let(s) => s.value.as_ref().map(expr_is_broadcast).unwrap_or(false),
            Stmt::Expr(e) => expr_is_broadcast(e),
            Stmt::If(s) => s.branches.iter().any(|(_, b)| stmts_use(b))
                || s.else_body.as_ref().map(|b| stmts_use(b)).unwrap_or(false),
            Stmt::While(s) => stmts_use(&s.body),
            Stmt::For(s)   => stmts_use(&s.body),
            Stmt::Loop(s)  => stmts_use(&s.body),
            _ => false,
        }
    }
    program.items.iter().any(|item| match item {
        Item::Fn(f) => stmts_use(&f.body),
        Item::Struct(s) => s.methods.iter().any(|m| stmts_use(&m.body)),
        Item::Enum(e)   => e.methods.iter().any(|m| stmts_use(&m.body)),
        _ => false,
    })
}

// ─── Local broadcast prelude ──────────────────────────────────────────────────

impl Transpiler {
    /// Emit a self-contained `!Send` broadcast primitive for single-thread mode.
    ///
    /// Mirrors the kernel Option-A design (per-receiver `VecDeque` slot) but uses
    /// `Rc<RefCell<...>>` and `tokio::sync::Notify` instead of `Arc<Mutex<...>>`
    /// and `CondVar`, since we are in a single-threaded async context.
    fn emit_local_broadcast_prelude(&mut self) {
        self.out.push_str(
            "// local_broadcast — !Send broadcast for single-thread mode.\n\
             struct LocalBcastSlot<T> {\n\
             \x20   buf:    std::collections::VecDeque<T>,\n\
             \x20   notify: std::rc::Rc<tokio::sync::Notify>,\n\
             }\n\
             \n\
             #[derive(Clone)]\n\
             struct LocalBroadcastSender<T> {\n\
             \x20   slots: std::rc::Rc<std::cell::RefCell<Vec<std::rc::Rc<std::cell::RefCell<LocalBcastSlot<T>>>>>>,\n\
             }\n\
             \n\
             struct LocalBroadcastReceiver<T> {\n\
             \x20   slot:   std::rc::Rc<std::cell::RefCell<LocalBcastSlot<T>>>,\n\
             }\n\
             \n\
             impl<T: Clone> LocalBroadcastSender<T> {\n\
             \x20   fn send(&self, value: T) {\n\
             \x20       for slot in self.slots.borrow().iter() {\n\
             \x20           let mut s = slot.borrow_mut();\n\
             \x20           s.buf.push_back(value.clone());\n\
             \x20           s.notify.notify_one();\n\
             \x20       }\n\
             \x20   }\n\
             \x20   fn subscribe(&self) -> LocalBroadcastReceiver<T> {\n\
             \x20       let notify = std::rc::Rc::new(tokio::sync::Notify::new());\n\
             \x20       let slot = std::rc::Rc::new(std::cell::RefCell::new(LocalBcastSlot {\n\
             \x20           buf: std::collections::VecDeque::new(),\n\
             \x20           notify: std::rc::Rc::clone(&notify),\n\
             \x20       }));\n\
             \x20       self.slots.borrow_mut().push(std::rc::Rc::clone(&slot));\n\
             \x20       LocalBroadcastReceiver { slot }\n\
             \x20   }\n\
             }\n\
             \n\
             impl<T> LocalBroadcastReceiver<T> {\n\
             \x20   async fn recv(&self) -> T {\n\
             \x20       loop {\n\
             \x20           let notify = std::rc::Rc::clone(&self.slot.borrow().notify);\n\
             \x20           if let Some(v) = self.slot.borrow_mut().buf.pop_front() {\n\
             \x20               return v;\n\
             \x20           }\n\
             \x20           notify.notified().await;\n\
             \x20       }\n\
             \x20   }\n\
             }\n\
             \n\
             fn local_broadcast<T: Clone>() -> LocalBroadcastSender<T> {\n\
             \x20   LocalBroadcastSender { slots: std::rc::Rc::new(std::cell::RefCell::new(Vec::new())) }\n\
             }\n\n"
        );
    }
}

// ─── GPU-target shared helpers (wgpu/cuda/metal) ──────────────────────────────
//
// These are shared by all three GPU backends' `transpile_*` entry points.
// wgpu uses them directly (its own kernel dispatch model matches the general
// pipeline's kernel-aware codegen in `emit_kernel.rs` exactly, so it can splice
// the general pass's output over the WHOLE program). cuda/metal additionally
// need `kernel_touching_fn_names`/`strip_top_level_fns` because their real
// kernel dispatch API (fallible construction, `__boring_launch`, CUDA streams /
// Metal command buffers) is richer than what `emit_kernel.rs` hardcodes (which
// is wgpu-shaped: infallible `new(device, queue)`, `.dispatch(gx,gy,gz)`) — see
// `cuda::transpile_cuda`'s doc comment for the full story of why a function
// that directly constructs/dispatches a kernel can't be transpiled by the
// general pipeline for those two backends, while every OTHER function can.

/// Clone `program`, renaming any top-level, non-method function literally named
/// `main` to `new_name`. Used so the general transpiler doesn't try to emit its
/// own `fn main()`/`#[tokio::main]` wrapper for a boring program's entry point
/// when it's being embedded into a GPU target's own generated `main.rs`.
pub(crate) fn rename_top_level_main(program: &Program, new_name: &str) -> Program {
    let items = program.items.iter().map(|item| {
        if let Item::Fn(f) = item {
            if f.name == "main" && f.qualifier.is_none() {
                let mut renamed = f.clone();
                renamed.name = new_name.to_string();
                return Item::Fn(renamed);
            }
        }
        item.clone()
    }).collect();
    Program { items }
}

/// Names of free functions whose body directly constructs a `kernel` instance
/// (`KernelName(...)`) or contains a `kernel: ...` dispatch block. These are the
/// only functions cuda/metal can't hand off to the general pipeline (see this
/// section's header comment) — everything else, including functions that merely
/// *call* one of these (e.g. a struct method that calls `linear_gpu(...)`), is
/// unaffected and goes through the general pipeline normally.
///
/// Scope: only free (`Item::Fn`) functions are checked. A `kernel`-touching
/// struct *method* is not detected here — see `strip_top_level_fns`'s doc for
/// why that combination isn't supported and is a documented gap, not silently
/// mishandled (callers should check for it separately if they care).
pub(crate) fn kernel_touching_fn_names(program: &Program, kernel_names: &std::collections::HashSet<String>) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            let param_is_kernel = f.params.iter().any(|p| matches!(&p.ty, Some(Type::Named(n)) if kernel_names.contains(n)));
            if param_is_kernel || stmts_touch_kernel(&f.body, kernel_names) {
                out.insert(f.name.clone());
            }
        }
    }
    out
}

/// True when any method of any `Item::Struct` in `program` is kernel-touching
/// (see `kernel_touching_fn_names`'s doc for the definition). `cuda`/`metal`'s
/// `transpile_*` entry points check this and, if true, do NOT attempt the
/// splice architecture for the affected struct at all (falling back to fully
/// custom emission for it) — routing such a struct through the general
/// pipeline while ALSO custom-emitting it for its kernel-touching method would
/// double-define the struct/impl block. Not exercised by this codebase's own
/// `.br` corpus (every kernel-touching construct here is a free function), so
/// this exists purely as a safety check, not a fully-engineered dual path.
pub(crate) fn kernel_touching_struct_names(program: &Program, kernel_names: &std::collections::HashSet<String>) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for item in &program.items {
        if let Item::Struct(s) = item {
            if s.methods.iter().any(|m| stmts_touch_kernel(&m.body, kernel_names)) {
                out.insert(s.name.clone());
            }
        }
    }
    out
}

/// True when the program's TOP-LEVEL statements/lets (as opposed to inside a
/// named function — see `kernel_touching_fn_names`) directly construct a
/// `kernel` instance or contain a `kernel: ...` dispatch block. Common in
/// practice for GPU-target example programs with no `def main():` at all
/// (e.g. `examples/vector_add_gpu.br`: `mut k = VectorAdd(...); kernel:
/// k(block=256)`, both bare top-level statements) — cuda/metal's
/// `transpile_*` entry points check this and, if true, keep top-level
/// statement/let handling entirely on their own custom emitter (matching
/// their pre-splice behavior, which already worked for this case) instead of
/// letting the general pipeline fold it into a synthesized `boring_main` (see
/// `TranspileConfig::gpu_top_level_handled_by_host`) — the general pipeline's
/// kernel-aware codegen for that fold-in is wgpu-shaped and can't run with
/// `gpu_kernels` empty (see `cuda::mod`'s doc comment for the full story).
pub(crate) fn top_level_touches_kernel(program: &Program, kernel_names: &std::collections::HashSet<String>) -> bool {
    for item in &program.items {
        match item {
            Item::Let(s)
                if s.value.as_ref().is_some_and(|v| expr_constructs_kernel(v, kernel_names) || expr_uses_gpu_device(v)) =>
            {
                return true;
            }
            Item::Stmt(s) if stmt_touches_kernel(s, kernel_names) || stmt_uses_gpu_device(s) => {
                return true;
            }
            _ => {}
        }
    }
    false
}

fn stmt_uses_gpu_device(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Let(s) => s.value.as_ref().is_some_and(expr_uses_gpu_device),
        Stmt::Expr(e) => expr_uses_gpu_device(e),
        _ => false,
    }
}

fn expr_constructs_kernel(e: &Expr, kernel_names: &std::collections::HashSet<String>) -> bool {
    match &e.kind {
        ExprKind::Call(callee, _) => matches!(&callee.kind, ExprKind::Var(n) if kernel_names.contains(n.as_str())),
        // `new(g) Scale(data)` — explicit-device kernel construction wraps the
        // same constructor call in `ExprKind::New`.
        ExprKind::New { ctor, .. } => expr_constructs_kernel(ctor, kernel_names),
        _ => false,
    }
}

/// True when `e` is a `GPU(n)`/`GPU.all()` use — cuda/metal's own custom
/// emitters have working GPU-device-introspection codegen (`gpu_vars`/
/// `emit_gpu_property`), but the general pipeline ALSO has its own version of
/// this (`emit_methods.rs`'s `.name()`/`.totalMem()`/etc mapping to
/// `__boring_gpu_name()` etc.), gated on `is_gpu_target` alone — unconditional,
/// same as the kernel-construction/resident-return issues this module's other
/// helpers work around — and only wgpu actually DEFINES those `__boring_gpu_*`
/// helpers. Used alongside `expr_constructs_kernel` so top-level `GPU(...)` use
/// also stays on cuda/metal's own custom emitter instead of the general splice.
fn expr_uses_gpu_device(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Call(callee, _) => matches!(&callee.kind, ExprKind::Var(n) if n == "GPU"),
        // `GPU.all()`.
        ExprKind::MethodCall(obj, _, _) => matches!(&obj.kind, ExprKind::Var(n) if n == "GPU"),
        _ => false,
    }
}

fn stmts_touch_kernel(body: &[Stmt], kernel_names: &std::collections::HashSet<String>) -> bool {
    body.iter().any(|s| stmt_touches_kernel(s, kernel_names))
}

fn stmt_touches_kernel(stmt: &Stmt, kernel_names: &std::collections::HashSet<String>) -> bool {
    match stmt {
        Stmt::KernelBlock(_) => true,
        Stmt::Let(s) => s.value.as_ref().is_some_and(|v| expr_constructs_kernel(v, kernel_names)),
        Stmt::LetDestructure(s) => expr_constructs_kernel(&s.value, kernel_names),
        Stmt::If(i) => i.branches.iter().any(|(_, b)| stmts_touch_kernel(b, kernel_names))
            || i.else_body.as_ref().is_some_and(|b| stmts_touch_kernel(b, kernel_names)),
        Stmt::While(w) => stmts_touch_kernel(&w.body, kernel_names),
        Stmt::DoWhile(w) => stmts_touch_kernel(&w.body, kernel_names),
        Stmt::For(f) => stmts_touch_kernel(&f.body, kernel_names),
        Stmt::Loop(l) => stmts_touch_kernel(&l.body, kernel_names),
        Stmt::Guard(g) => stmts_touch_kernel(&g.else_body, kernel_names),
        Stmt::With(w) => stmts_touch_kernel(&w.body, kernel_names),
        Stmt::Defer(b) => stmts_touch_kernel(b, kernel_names),
        Stmt::Try(t) => stmts_touch_kernel(&t.body, kernel_names)
            || t.catch_clauses.iter().any(|c| stmts_touch_kernel(&c.body, kernel_names)),
        Stmt::Match(m) => m.arms.iter().any(|a| match &a.body {
            MatchBody::Block(b) => stmts_touch_kernel(b, kernel_names),
            MatchBody::Expr(e) => expr_constructs_kernel(e, kernel_names),
        }),
        _ => false,
    }
}

/// Remove each top-level `pub fn {name}(...) { ... }` / `fn {name}(...) { ... }`
/// definition (matched by brace-depth, not just text search) from `code`, for
/// every `name` in `names`.
///
/// Used to discard the general pipeline's rendering of a kernel-touching
/// function's BODY (which is necessarily wrong for cuda/metal — see this
/// section's header comment: the pipeline emits wgpu-shaped kernel construction/
/// dispatch calls that don't exist on cuda/metal's real kernel structs), while
/// still routing the function through the general pipeline for the purpose of
/// registering its signature/`throws`-ness — every OTHER (non-kernel-touching)
/// function's calls TO `name` still went through the general pipeline's own
/// call-site emission (by-ref array/dict/set/struct coercion, `?`-propagation,
/// etc.), which needed `name`'s real declaration to get that right. The GPU
/// backend's own custom emitter supplies the real (native, fallible-dispatch)
/// replacement body for each stripped name separately, with a matching by-ref
/// signature (see `cuda::host`'s `fn_ref_params`).
///
/// Assumes (true of every function this transpiler emits, checked against every
/// sample in this codebase) that a function signature is never wrapped across
/// multiple lines — matching is done against a single trimmed line's prefix.
pub(crate) fn strip_top_level_fns(code: &str, names: &std::collections::HashSet<String>) -> String {
    if names.is_empty() { return code.to_string(); }
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    while i < code.len() {
        let line_end = code[i..].find('\n').map(|p| i + p + 1).unwrap_or(code.len());
        let line = &code[i..line_end];
        let trimmed = line.trim_start();
        let matched = names.iter().any(|n| {
            trimmed.starts_with(&format!("fn {}(", n)) || trimmed.starts_with(&format!("pub fn {}(", n))
        });
        if matched {
            let rest = &code[i..];
            let mut depth = 0i32;
            let mut seen_open = false;
            let mut end_offset = rest.len();
            for (off, ch) in rest.char_indices() {
                if ch == '{' { depth += 1; seen_open = true; }
                if ch == '}' {
                    depth -= 1;
                    if seen_open && depth == 0 {
                        let after = &rest[off + 1..];
                        end_offset = after.find('\n').map(|p| off + 1 + p + 1).unwrap_or(rest.len());
                        break;
                    }
                }
            }
            i += end_offset;
            continue;
        }
        out.push_str(line);
        i = line_end;
    }
    out
}

/// Given the renamed program (see `rename_top_level_main`) and the general
/// pipeline's output for it, determine whether a `boring_main` exists and
/// whether it returns a `Result` — shared by wgpu/cuda/metal so each backend's
/// own real `fn main()` knows whether/how to call it. See `wgpu::transpile_wgpu`
/// for the original derivation of this logic (this is a verbatim extraction).
pub(crate) fn detect_boring_main(renamed_program: &Program, general_out: &TranspileOutput) -> (bool, bool) {
    let explicit_boring_main_throws = renamed_program.items.iter().find_map(|item| {
        match item {
            Item::Fn(f) if f.name == "boring_main" && f.qualifier.is_none() => Some(f.throws),
            _ => None,
        }
    });
    let has_boring_main = general_out.gpu_main_emitted || explicit_boring_main_throws.is_some();
    let boring_main_throws = explicit_boring_main_throws.unwrap_or(true);
    (has_boring_main, boring_main_throws)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn transpile_src_with_config(src: &str, config: TranspileConfig) -> String {
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        transpile_with_config(&program, config).code
    }

    #[test]
    fn test_managed_multi_wraps_owned() {
        // T'owned → Arc<Mutex<T>> in managed+multi; plain Named is NOT wrapped.
        let src = "struct Counter:\n    init(pub int n)\ndef Counter'owned make(): Counter(n = 5)\n";
        let config = TranspileConfig { mode: TranspileMode::Managed, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("Arc<std::sync::Mutex<Counter>>"),
            "T'owned should become Arc<std::sync::Mutex> in managed+multi:\n{}", code);
    }

    #[test]
    fn test_managed_single_wraps_owned() {
        // T'owned → RefCell<T> in managed+single; plain Named is NOT wrapped.
        let src = "struct Counter:\n    init(pub int n)\ndef Counter'owned make(): Counter(n = 5)\n";
        let config = TranspileConfig { mode: TranspileMode::Managed, threading: ThreadingMode::Single, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("RefCell<Counter>"),
            "T'owned should become RefCell in managed+single:\n{}", code);
    }

    #[test]
    fn test_managed_named_not_wrapped() {
        // Plain Named (Type::Named) is NOT wrapped in managed mode — only T'owned is.
        let src = "struct Counter:\n    init(pub int n)\n\nlet Counter c = Counter(n = 0)\n";
        let config = TranspileConfig { mode: TranspileMode::Managed, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(!code.contains("Arc<tokio::sync::Mutex<Counter>>"),
            "managed mode: plain Named should NOT be wrapped, got:\n{}", code);
    }

    #[test]
    fn test_strict_mode_no_managed_wrapping() {
        let src = "struct Counter:\n    init(pub int n)\n\nlet Counter c = Counter(n = 0)\n";
        let config = TranspileConfig { mode: TranspileMode::Strict, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(!code.contains("Arc<tokio::sync::Mutex<Counter>>"),
            "strict mode: Counter should NOT be wrapped, got:\n{}", code);
    }

    #[test]
    fn test_managed_unit_enum_not_wrapped() {
        // Non-parametric enums must remain Copy — never wrapped in managed mode.
        let src = "enum Color:\n    Red\n    Green\n    Blue\n\nlet Color c = Color.Red\n";
        let config = TranspileConfig { mode: TranspileMode::Managed, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(!code.contains("Arc<tokio::sync::Mutex<Color>>"),
            "managed mode: unit enum should NOT be wrapped, got:\n{}", code);
    }

    #[test]
    fn test_for_loop_destructure_marks_mut_tuple_slot() {
        // `Query<(mut Position&, Velocity&)>`-shaped parameter (Bevy-ECS-style,
        // see docs/mut-type-modifier.md's motivating example): destructuring
        // its items in a `for` loop must bind the slot declared `mut T&` with
        // a Rust `mut` in the pattern — required for that item's type (e.g.
        // Bevy's `Mut<T>` change-detection wrapper) to be field-mutable, since
        // Boring has no source syntax to write `mut` on a for-loop variable
        // directly (only inferred from the iterable's declared type).
        let src = "\
struct Position:\n    pub var float x\n\nstruct Velocity:\n    pub float x\n\n\
def move_system(Query<(mut Position&, Velocity&)> query):\n    for pos, vel in query:\n        pos.x = pos.x + vel.x\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("for (mut pos, vel) in"),
            "expected `mut` on the slot declared `mut Position&`, got:\n{}", code);
    }

    #[test]
    fn test_for_loop_destructure_no_spurious_mut() {
        // Non-regression: a plain (non-`mut`-tagged) tuple/dict destructure
        // must NOT gain a `mut` binding it didn't have before this feature.
        let src = "let m = {\"a\" = 1}\nfor k, v in m:\n    print \"{k}={v}\"\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        // Check the actual `for` line, not a substring match anywhere in the
        // file — the standard-library boilerplate Boring always emits (e.g.
        // `BoringArrayIndex::remove_at`) legitimately contains `let mut v = ...`
        // elsewhere, which a bare `!code.contains("mut v")` would false-positive on.
        assert!(code.contains("for (k, v) in"),
            "plain dict destructure must not gain a spurious `mut`, got:\n{}", code);
    }

    #[test]
    fn test_for_loop_destructure_mixed_mut_slots() {
        // Both tuple positions independently `mut`-tagged, or neither, must
        // each be judged on their own slot — not "any mut anywhere -> all mut".
        let src = "\
struct A:\n    pub var float x\n\nstruct B:\n    pub var float x\n\n\
def sys(Query<(mut A&, mut B&)> query):\n    for a, b in query:\n        a.x = a.x + b.x\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("for (mut a, mut b) in"),
            "both slots declared `mut T&` should both bind `mut`, got:\n{}", code);
    }

    #[test]
    fn test_single_thread_uses_spawn_local() {
        // In --threading single mode, task expressions must emit tokio::task::spawn_local.
        let src = "task int work(int n):\n    return n\n\nlet t = task work(1)\n";
        let config = TranspileConfig { mode: TranspileMode::Strict, threading: ThreadingMode::Single, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("tokio::task::spawn_local"),
            "single mode: task should emit spawn_local, got:\n{}", code);
        assert!(!code.contains("tokio::spawn("),
            "single mode: should NOT emit bare tokio::spawn, got:\n{}", code);
    }

    #[test]
    fn test_single_thread_uses_local_channel() {
        // In --threading single mode, typed channel creation must use local_channel::mpsc.
        let src = "let tx, rx = channel<int>(10)\n";
        let config = TranspileConfig { mode: TranspileMode::Strict, threading: ThreadingMode::Single, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("local_channel::mpsc"),
            "single mode: channel should use local_channel::mpsc, got:\n{}", code);
    }

    #[test]
    fn test_multi_thread_uses_tokio_spawn() {
        // In --threading multi mode (default), task expressions must emit tokio::spawn.
        let src = "task int work(int n):\n    return n\n\nlet t = task work(1)\n";
        let config = TranspileConfig { mode: TranspileMode::Strict, threading: ThreadingMode::Multi, ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("tokio::spawn("),
            "multi mode: task should emit tokio::spawn, got:\n{}", code);
    }

    // ── Derive-trait whitelist (`as Trait:` routed into `#[derive(...)]`) ─────────────

    #[test]
    fn struct_header_mixes_builtin_derive_and_user_trait() {
        // `Hash`/`Eq` are in the built-in KNOWN_DERIVABLE_TRAITS whitelist (not part of the
        // struct's own auto-default Debug/Clone/PartialEq) and should merge into ONE derive
        // line; `Named` is a real Boring trait and must still get its own `impl` block with
        // only its own method.
        let src = "\
trait Named:\n    req string name()\n\n\
struct Point as Hash, Eq, Named:\n    int x\n    int y\n\n    req string name():\n        \"point\"\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("#[derive(Debug, Clone, PartialEq, Hash, Eq)]"),
            "whitelisted names should merge into the struct's single derive line, got:\n{}", code);
        assert!(code.contains("impl Named for Point {"),
            "user-declared trait must still get a manual impl block, got:\n{}", code);
        assert!(!code.contains("impl Hash for Point") && !code.contains("impl Eq for Point"),
            "whitelisted derive names must NOT also get an impl block, got:\n{}", code);
    }

    #[test]
    fn struct_header_derive_name_extended_via_config() {
        // A project-declared name (boring.toml's `[derives]`, threaded via
        // `TranspileConfig::known_derives`) is routed exactly like a built-in one.
        let src = "struct Velocity as Component:\n    float x\n    float y\n";
        let config = TranspileConfig { known_derives: vec!["Component".to_string()], ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("#[derive(Debug, Clone, PartialEq, Component)]"),
            "project-registered derive name should merge into the derive line, got:\n{}", code);
        assert!(!code.contains("impl Component for Velocity"),
            "project-registered derive name must NOT get an impl block, got:\n{}", code);
    }

    #[test]
    fn struct_header_derive_name_not_duplicated_against_explicit_attr() {
        // Explicit `@derive(...)` and a whitelisted header name both naming `Clone` must not
        // produce two `Clone` entries (which rustc would reject as a duplicate impl).
        let src = "@derive(Debug, Clone)\nstruct Point as Clone:\n    int x\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("#[derive(Debug, Clone)]"),
            "whitelisted name already present in the explicit @derive list must not repeat, got:\n{}", code);
    }

    #[test]
    fn bare_derive_serialize_deserialize_are_qualified_and_flag_uses_serde() {
        // `@derive(Serialize, Deserialize)` (the exact bare spelling docs/book.md's `json`/
        // `fromJson` section documents) must not emit a bare, unresolvable `Deserialize` in
        // `#[derive(...)]` -- Boring never emits `use serde::{Serialize, Deserialize};`, so an
        // unqualified name there fails to compile ("cannot find derive macro"/unsatisfied
        // trait bound). Fixed by qualifying to the full path, mirroring the existing
        // `thiserror::Error` auto-qualification. `uses_serde` must also flip so Cargo.toml
        // gets the `serde` dependency even without a `json()`/`fromJson()` call in this file.
        let src = "@derive(Serialize, Deserialize)\nstruct Costume:\n    string name\n";
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        let out = transpile_with_config(&program, TranspileConfig::default());
        assert!(out.code.contains("#[derive(serde::Serialize, serde::Deserialize)]"),
            "bare Serialize/Deserialize should be qualified to serde::, got:\n{}", out.code);
        assert!(out.uses_serde, "a struct deriving Serialize/Deserialize must flag uses_serde");
    }

    #[test]
    fn struct_display_impl_skipped_without_debug_in_explicit_derive() {
        // The auto Display impl delegates to `{:?}` (Debug). An explicit `@derive(...)` (used
        // verbatim, unlike the no-@derive-at-all auto path, which always includes Debug) that
        // doesn't itself list Debug must not get that impl -- book.md's own fromJson example
        // (`@derive(Serialize, Deserialize)`, no Debug) hit exactly this: "`Target` doesn't
        // implement `Debug`" pointing at the generated `impl Display`.
        let src = "@derive(Serialize, Deserialize)\nstruct Costume:\n    string name\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("impl std::fmt::Display for Costume"),
            "no Display impl should be generated when Debug isn't in the explicit derive list, got:\n{}", code);
    }

    #[test]
    fn struct_display_impl_still_emitted_when_debug_explicit() {
        // Same shape as above, but Debug IS in the explicit list -- the impl must still be
        // generated (this is a no-regression check alongside the skip case above).
        let src = "@derive(Debug, Serialize)\nstruct Costume:\n    string name\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("impl std::fmt::Display for Costume"),
            "Display impl should still be generated when Debug is explicitly derived, got:\n{}", code);
    }

    #[test]
    fn enum_display_impl_skipped_without_debug_in_explicit_derive() {
        // The *enum* auto-Display had no equivalent of the `will_have_debug` gate the two
        // struct tests above cover, so `@derive(Clone, Serialize, Deserialize)` on an enum
        // (explicit, hence used verbatim with no auto-injected Debug) generated
        // `write!(f, "{:?}", self)` for a non-Debug type: "`Num` doesn't implement `Debug`".
        // See tests/cases/enum_derive_no_debug.br.
        let src = "@derive(Clone, Serialize, Deserialize)\nenum Num:\n    NInt(int)\n    NStr(string)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(!code.contains("impl std::fmt::Display for Num"),
            "no Display impl should be generated for an enum when Debug isn't in the explicit derive list, got:\n{}", code);
    }

    #[test]
    fn enum_display_impl_still_emitted_when_debug_explicit() {
        // No-regression companion to the skip case above: with Debug explicitly derived the
        // enum's auto-Display must still be generated (this is what lets `print someEnum`
        // and `BoringFmt<Vec<Enum>>` work at all).
        let src = "@derive(Debug, Clone, Serialize, Deserialize)\nenum Num:\n    NInt(int)\n    NStr(string)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("impl std::fmt::Display for Num"),
            "Display impl should still be generated for an enum when Debug is explicitly derived, got:\n{}", code);
    }

    #[test]
    fn enum_display_impl_still_emitted_without_any_derive_attr() {
        // No `@derive(...)` at all -> the auto path emits `#[derive(Debug, Clone, ...)]`
        // itself, so Debug is present and the auto-Display must be emitted. Guards against
        // the `will_have_debug` gate being read off the *attrs* rather than off the derive
        // names actually emitted.
        let src = "enum Num:\n    NInt(int)\n    NStr(string)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("impl std::fmt::Display for Num"),
            "an enum with no explicit @derive still auto-derives Debug, so Display must be emitted, got:\n{}", code);
    }

    #[test]
    fn field_level_attr_emitted_verbatim_above_field() {
        // `@serde(rename = "...")` (or any other attribute) directly above a field must be
        // preserved and emitted as a real Rust attribute immediately above that field --
        // needed for a JSON key that isn't a valid Boring identifier at all (e.g. "1"), which
        // a struct-wide `@serde(rename_all = "...")` can't reach. See
        // docs/json-deserialize-rename-gap.md.
        let src = "@derive(Debug)\nstruct Costume:\n    string name\n    @serde(rename = \"1\")\n    int assetIndex\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("#[serde(rename=\"1\")]\n    pub assetIndex: isize,")
            || code.contains("#[serde(rename = \"1\")]\n    pub assetIndex: isize,"),
            "field-level attr should be emitted verbatim directly above its field, got:\n{}", code);
    }

    #[test]
    fn empty_derive_attr_suppresses_auto_default_alongside_header() {
        // An explicit `@derive()` (empty args) still counts as "has_derive" — it disables
        // the struct's auto-default Debug/Clone/PartialEq entirely, while a header `as
        // Trait, ...:` list still routes its whitelisted names into the (now otherwise
        // empty) derive line. This is the intended way to combine `as Trait:` syntax with
        // an exact, non-broadened derive set (no accidental Clone/PartialEq) — see
        // docs/book.md's "derive-trait whitelist" section.
        let src = "@derive()\nstruct Paddle as Component, Debug:\n    pass\n";
        let config = TranspileConfig { known_derives: vec!["Component".to_string()], ..TranspileConfig::default() };
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("#[derive(Component, Debug)]"),
            "empty @derive() should suppress the auto-default entirely, leaving only the \
             header's whitelisted names, got:\n{}", code);
        assert!(!code.contains("impl Component for Paddle"),
            "Component must still be routed to derive, not a manual impl, got:\n{}", code);
    }

    #[test]
    fn enum_header_derive_name_merges_as_extra_derive_line() {
        // A non-unit enum's auto-default is only Debug/Clone/PartialEq (unlike the
        // unit-only-variants case, which already includes Hash/Eq/Copy) — `Hash` here is
        // genuinely new and must appear in its own extra `#[derive(...)]` line.
        let src = "enum Shape as Hash:\n    Circle(int)\n    Square(int, int)\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("#[derive(Debug, Clone, PartialEq)]"),
            "existing auto-derive line should be untouched, got:\n{}", code);
        assert!(code.contains("#[derive(Hash)]"),
            "whitelisted enum header name not already covered by the auto-derive line \
             should get its own extra #[derive(...)] line, got:\n{}", code);
    }

    #[test]
    fn ext_block_rejects_whitelisted_derive_name() {
        // Rust can't retroactively attach #[derive(...)] to a type defined elsewhere — an
        // `ext ... as <whitelisted>:` must be a hard transpile error, not silently emit
        // `impl <Trait> for X { <every ext method> }`.
        let src = "\
struct Foo:\n    int x\n\n\
ext Foo as Debug:\n    req int double():\n        self.x * 2\n";
        let tokens = crate::lexer::lex(src).expect("lex error");
        let program = crate::parser::parse(tokens).expect("parse error");
        let out = transpile_with_config(&program, TranspileConfig::default());
        assert!(!out.errors.is_empty(), "expected a transpile error for 'ext Foo as Debug:'");
        assert!(out.errors.iter().any(|e| e.message.contains("derive macro")),
            "error should explain Debug is a derive macro, got: {:?}", out.errors);
        assert!(!out.code.contains("impl Debug for Foo"),
            "must not emit an impl block for a derive-macro name via ext, got:\n{}", out.code);
    }

    // ── `boring.toml [external_fns]` / `KNOWN_EXTERNAL_FN_BORROWS` ──────────────────────
    // `mem_borrow_builtins` (tests/cases/mem_borrow_builtins.br, tests/transpile.rs) already
    // covers the built-in table end-to-end through a real `cargo run`. The three unit tests
    // below cover project-declared `TranspileConfig::external_fns` entries at the transpiler
    // level (no real external crate needed, since these only check the generated Rust
    // *string* — a full compile+run of a project-declared entry against real crates like
    // `zip`/`resvg` was done separately, outside this repo's own test suite).

    #[test]
    fn external_fns_static_type_call_applies_project_declared_borrows() {
        // `Type.method(args)` — the "static call on an unregistered external type" shape
        // (`try_emit_type_method_call`'s own external fallback), the third and, until this
        // fix, unwired call site (found via the `Tree.from_str(text, opt)` real-crate repro
        // — `resvg.render(...)`, a free function, already worked; this one silently didn't).
        let src = "def main():\n    var int a = 1\n    var int b = 2\n    Widget.build(a, b)\n";
        let mut config = TranspileConfig::default();
        config.external_fns = vec![("Widget".to_string(), "build".to_string(), vec!["".to_string(), "&mut".to_string()])];
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("Widget::build(a, &mut b)"),
            "static Type.method(args) call should apply project-declared arg borrows, got:\n{}", code);
    }

    #[test]
    fn external_fns_method_name_is_normalized_to_snake_case_before_matching() {
        // Regression test: `boring.toml [external_fns]` stores a key's method half verbatim
        // from the TOML text (e.g. `"buildThing"`, matching the `.br` source's own camelCase
        // call spelling) — never run through `camel_to_snake` at parse time. Every call site
        // passes the *already*-`camel_to_snake`'d Rust method name to
        // `external_fn_arg_borrows`, so the lookup itself must normalize the stored entry
        // too, or a project's own whitelist entry never matches at all. Never caught by
        // `mem.swap`/`replace`/`take` (identical Boring-source/Rust spellings already) —
        // only surfaced via the `ZipFile::readToEnd` (→ `read_to_end`) real-crate repro.
        let src = "def main():\n    var int a = 1\n    var int b = 2\n    Widget.buildThing(a, b)\n";
        let mut config = TranspileConfig::default();
        config.external_fns = vec![("Widget".to_string(), "buildThing".to_string(), vec!["".to_string(), "&mut".to_string()])];
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("Widget::build_thing(a, &mut b)"),
            "camelCase TOML method name should still match the snake_case Rust call site, got:\n{}", code);
    }

    #[test]
    fn external_fns_preserves_positional_by_value_placeholder_mid_array() {
        // Regression test for the `resvg::render` real-crate repro's other bug:
        // `["&", "", "&mut"]`'s middle by-value placeholder must stay at index 1, not be
        // dropped and shift the trailing `"&mut"` left (`BoringToml::extract_borrow_array`,
        // src/main.rs — this test exercises the *consuming* end of that fix, the TOML
        // parsing side has its own dedicated unit tests in main.rs).
        let src = "def main():\n    var int a = 1\n    var int b = 2\n    var int c = 3\n    Lib.call3(a, b, c)\n";
        let mut config = TranspileConfig::default();
        config.external_fns = vec![("Lib".to_string(), "call3".to_string(),
            vec!["&".to_string(), "".to_string(), "&mut".to_string()])];
        let code = transpile_src_with_config(src, config);
        assert!(code.contains("Lib::call3(&a, b, &mut c)"),
            "middle by-value arg must stay unprefixed, first/last keep their own borrows, got:\n{}", code);
    }

    // ── docs/option-return-double-some-wrap-bug.md ────────────────────────────
    // A `T?`-returning function/method (or a plain assignment into an already-
    // `T?`-typed local) must NOT wrap an expression that is itself statically
    // Option-shaped (a `T?`-declared field read, or a call to another
    // `T?`-returning function/method) in an extra `Some(...)` — that produces
    // `Option<Option<T>>`, a real E0308. These four tests cover the doc's three
    // repros (tail-position field read, tail-position method call, plain
    // assignment) plus the still-needed-wrap case that must keep working.

    #[test]
    fn optional_return_of_optional_field_is_not_double_wrapped() {
        // Repro 1: `req string? next_id(): next` where `next` is a `string?` field.
        let src = "\
struct Block:\n    string? next\n\n    req string? next_id():\n        next\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn next_id(&self) -> Option<Arc<str>> {\n        self.next.clone()\n    }"),
            "tail-position read of a T?-declared field must be passed through raw, not Some(...)-wrapped, got:\n{}", code);
        assert!(!code.contains("Some(self.next"),
            "must not double-wrap an already-Option field read, got:\n{}", code);
    }

    #[test]
    fn optional_return_of_optional_method_call_is_not_double_wrapped() {
        // Repro 2: `def string? first_of([JVal] items): items[0].as_str()` where
        // `as_str()` is itself declared `string?` — receiver is an `Index` expr,
        // not a bare `Var`, so this also exercises the general (non-Var-only)
        // receiver resolution.
        let src = "\
enum JVal:\n    JStr(string)\n    JNum(float)\n\n    req string? as_str():\n        match self.clone():\n            JStr(s): s\n            _: nil\n\n\
def string? first_of([JVal] items):\n    items[0].as_str()\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("fn first_of(items: &Vec<JVal>) -> Option<Arc<str>> {\n    items[(0) as usize].as_str()\n}"),
            "tail call to a T?-returning method must be passed through raw, not Some(...)-wrapped, got:\n{}", code);
        assert!(!code.contains("Some(items"),
            "must not double-wrap an already-Option method-call result, got:\n{}", code);
    }

    #[test]
    fn optional_assignment_of_optional_method_call_is_not_double_wrapped() {
        // Repro 3: `id = items[0].as_str()` assigned into a `var string? id`.
        let src = "\
enum JVal:\n    JStr(string)\n    JNum(float)\n\n    req string? as_str():\n        match self.clone():\n            JStr(s): s\n            _: nil\n\n\
def string? parse([JVal] items):\n    var string? id = nil\n    id = items[0].as_str()\n    id\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("id = items[(0) as usize].as_str();"),
            "assignment RHS that's already T?-shaped must not be Some(...)-wrapped, got:\n{}", code);
        assert!(!code.contains("Some(items"),
            "must not double-wrap an already-Option assignment RHS, got:\n{}", code);
    }

    #[test]
    fn optional_return_of_bare_value_is_still_wrapped() {
        // Non-regression: a genuinely bare `T` (a plain, non-optional field) in
        // T?-returning tail position must still get Some(...) — this is the
        // legitimate case the fix must not break.
        let src = "\
struct Block:\n    string label\n\n    req string? label_opt():\n        label\n";
        let code = transpile_src_with_config(src, TranspileConfig::default());
        assert!(code.contains("Some(self.label"),
            "a bare (non-optional) field read must still be wrapped in Some(...), got:\n{}", code);
    }
}
