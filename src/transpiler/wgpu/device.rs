// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// WGSL device code emitter for the wgpu backend.

use crate::ast::*;
use crate::transpiler::TranspileError;
use crate::transpiler::helpers::{
    collect_vars_in_stmt,
    labeled_array_at_index, labeled_array_dim_literal,
    first_loop_index,
};

/// Emits the WGSL device module(s) for this program. Returns `(real, emulated)`:
/// `real` always uses `gpu.warp.*`'s real-subgroup mapping (`subgroupShuffle*`,
/// gated by `enable subgroups;`) — identical to today's single-module output for
/// any program that doesn't use `gpu.warp.*` at all. `emulated` is `Some(..)`
/// only when some kernel uses `gpu.warp.*`: a second module using the
/// shared-memory-emulated mapping, for adapters lacking `wgpu::Features::SUBGROUP`
/// (see `WarpMode`). The host (`host::emit_host_rs`) chooses between them at
/// runtime by querying that feature.
pub(super) fn emit_device_wgsl(
    program: &Program,
    effective_kernels: &[crate::ast::KernelDecl],
    kernel_consts: &super::KernelConstsMap,
) -> (String, Option<String>, Vec<TranspileError>) {
    let uses_warp = effective_kernels.iter().any(super::kernel_uses_gpu_warp);

    let mut real = DeviceEmitter::new(WarpMode::Real);
    real.program_uses_warp = uses_warp;
    real.emit_program(program, effective_kernels, kernel_consts);

    let mut errors = real.errors;

    let emulated = if uses_warp {
        let mut e = DeviceEmitter::new(WarpMode::Emulated);
        e.program_uses_warp = true;
        e.emit_program(program, effective_kernels, kernel_consts);
        // Both passes emit from the same kernel decls/bodies, so the same underlying
        // issue (a bad field qualifier, an out-of-range literal, ...) is normally
        // diagnosed identically by both — only append errors the emulated pass found
        // that the real pass didn't, so a program with one real bug doesn't get the
        // same diagnostic reported twice.
        for err in e.errors {
            let dup = errors.iter().any(|existing: &TranspileError| {
                existing.message == err.message && existing.line == err.line && existing.col == err.col
            });
            if !dup {
                errors.push(err);
            }
        }
        Some(e.out)
    } else {
        None
    };

    (real.out, emulated, errors)
}

/// Which `gpu.warp.*` codegen path this emitter pass produces. wgpu is the
/// only backend that needs this distinction — WGSL subgroup builtins require
/// an `enable subgroups;` directive and are only valid when the adapter has
/// `wgpu::Features::SUBGROUP`, so `emit_device_wgsl` always emits both a
/// `Real` module (used when the feature is present at runtime) and, only for
/// programs that use `gpu.warp.*`, an `Emulated` module (used when it isn't)
/// — see that function's doc comment.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WarpMode { Real, Emulated }

/// Fixed simulated warp size for the `Emulated` path — there's no real
/// subgroup to query a size from, so this is a documented constant rather
/// than a hardware value (matches the interpreter's `eval_gpu::WARP_SIZE` and
/// the existing host-side `__boring_gpu_warp_size` mock).
const EMULATED_WARP_SIZE: u32 = 32;

/// WGSL reserves every identifier starting with `__` (confirmed via a real
/// naga parse: "Identifier starts with a reserved prefix"). Buffer-qualified
/// fields are already routed through `current_buffer_renames`'s
/// `{kernel}_{field}` scheme, which never collides with this, but a bare
/// field name emitted verbatim — a params-uniform struct member/local unpack,
/// or a `'sync` fixed-array workgroup declaration — can still be one of a
/// few internally `__`-prefixed names: a labeled dynamic-shape array's
/// desugared shadow axes (`desugar_labeled_array.rs`'s `__name_axisN`) being
/// the one that actually reaches this backend. Reuses the existing `bp_`
/// (single underscore) convention this file already applies elsewhere (see
/// `try_emit_plain_index_method_stmt`'s discard-target rename) rather than
/// introducing a second one.
///
/// Separately, a *bare* Boring identifier (a kernel field, a `def()`-body
/// local, a function/method parameter, or a `for`-loop variable) can collide
/// with a real WGSL reserved keyword — same bug class as the Metal backend's
/// `msl_safe_ident` (`metal/device.rs`), except WGSL has no builtin *scalar*
/// type names in the same namespace as MSL's (`half`, `float`, ... — Boring
/// has no primitive spelled that way, so that specific risk doesn't reach this
/// backend at all), only real bare-word keywords. See `WGSL_RESERVED`'s own
/// doc comment for where that table comes from and `docs/wgpu-backend.md`'s
/// "Naming restrictions" section.
fn wgsl_safe_ident(name: &str) -> String {
    let base = match name.strip_prefix("__") {
        Some(rest) => format!("bp_{rest}"),
        None => name.to_string(),
    };
    if WGSL_RESERVED.contains(&base.as_str()) {
        format!("{base}_")
    } else {
        base
    }
}

/// A `'sync` fixed-size array workgroup field is always declared as
/// `var<workgroup> {name}: array<{elem}, {n}>;` (see `emit_kernel_decl`'s
/// "3. Workgroup" section) — module scope, and always typed with the literal
/// `array` type constructor. If the field's own (already `wgsl_safe_ident`-
/// sanitized) name is itself `array`, the declaration's *name* textually
/// equals the *type constructor keyword* used to spell its own type, and naga
/// rejects the whole declaration outright with `"declaration of `array` is
/// recursive"` — a distinct diagnostic from the reserved-keyword rejection
/// `wgsl_safe_ident` already guards against, and one that a same-named
/// *differently*-typed declaration (`var<workgroup> array: i32;`) does NOT
/// trigger, nor does a same-named *struct field* of any type (a struct
/// member is not itself a naga "declaration" in the sense this error checks).
/// Confirmed via a real naga parse (naga 30.0.1,
/// `naga::front::wgsl::parse_str`) — see CHANGELOG.md and
/// docs/wgpu-backend.md's "Naming restrictions" section.
///
/// This workgroup-array declaration is the *only* place this backend can
/// actually reach the bug from valid Boring source: a storage-buffer field's
/// WGSL name always goes through `current_buffer_renames`'s `{kernel}_{field}`
/// prefixing first (`emit_kernel_decl`'s "1. Array buffer fields" —
/// never bare `array`/`atomic`, so an atomic buffer field can't collide
/// either, even though its own type nests the `atomic<...>` keyword), and
/// every other bare-name declaration this backend emits (a params-uniform
/// scalar/array field, a `def()`-body local, a function/method parameter, a
/// `for`-loop variable) is either a struct field (verified harmless above) or
/// never itself spelled with the `array`/`atomic` type constructor as its own
/// declared type. Reuses `wgsl_safe_ident`'s existing trailing-`_` convention
/// rather than introducing a second one.
///
/// Kernel-prefixed (`{kernel}_{field}`, lowercased kernel name — matching
/// `current_buffer_renames`'s own scheme) for the same reason buffer fields
/// are: a `'sync`/`'actor` fixed-array field is *also* a module-scope WGSL
/// declaration (`var<workgroup> {name}: array<...>;`), so two unrelated
/// kernels that both happen to name their tile field the same thing (e.g.
/// `tile_x` in two independently-written tiled-matmul kernels) previously
/// collided at module scope — WGSL rejects the redefinition, but only at
/// real shader-module compile time (`Device::create_shader_module`), which
/// neither `boring build` nor `cargo build` reach. See CHANGELOG.md.
fn wgsl_workgroup_array_ident(kernel: &str, name: &str) -> String {
    let safe = format!("{}_{}", kernel.to_lowercase(), wgsl_safe_ident(name));
    if safe == "array" { format!("{safe}_") } else { safe }
}

/// WGSL reserved words, mirroring naga's own `RESERVED` table
/// (`naga::keywords::wgsl::RESERVED` — the actual parser/validator `wgpu`
/// uses at runtime to compile the WGSL text this backend emits, exactly the
/// tool the Metal backend's `MSL_KEYWORDS` table was verified against for the
/// analogous bug there) rather than the current WGSL spec's own "keyword
/// summary" table (https://www.w3.org/TR/WGSL/#keyword-summary): naga reserves
/// a considerably larger set than the spec's current grammar keywords, for
/// the language's own future growth (e.g. `class`, `import`, `std`, `crate`,
/// `Self`, `NULL` are all rejected as plain identifiers today despite naming
/// no current WGSL construct). Verified empirically against a real `naga`
/// parse (naga 30.0.1, `naga::front::wgsl::parse_str`) for a representative
/// sample of these, including every entry this fix's own regression tests
/// exercise — see CHANGELOG.md.
///
/// Deliberately does **not** include naga's separate `BUILTIN_IDENTIFIERS`
/// table (`array`, `atomic`, `ptr`, `bitcast`, `f32`, `vec2`, ...) — confirmed
/// via the same real `naga` parse that none of these are actually rejected as
/// a plain identifier (`word_as_ident` only consults `RESERVED`); they're
/// ordinary, shadowable identifiers outside of an actual type/builtin-call
/// position. (One real but narrower bug surfaced by that same probing — a
/// `'sync` fixed-array workgroup field named `array` hits a distinct naga
/// "declaration is recursive" parse error, since its own declared type also
/// spells `array` — is fixed separately by `wgsl_workgroup_array_ident`
/// rather than folded into this keyword fix; see that function's doc comment
/// for why an atomic-typed field can't reach the analogous case.)
///
/// Many entries below can never actually reach this backend as a Boring
/// identifier at all — Boring's own keyword list (`src/lexer/mod.rs`) already
/// reserves `let`/`var`/`struct`/`for`/`if`/`while`/`return`/etc. at the
/// Boring-source level — but are kept here anyway for exact parity with
/// naga's own source rather than hand-filtered, matching this file's
/// `MSL_KEYWORDS` precedent (`metal/device.rs`), which does the same.
const WGSL_RESERVED: &[&str] = &[
    "alias", "break", "case", "const", "const_assert", "continue", "continuing",
    "default", "diagnostic", "discard", "else", "enable", "false", "fn", "for",
    "if", "let", "loop", "override", "requires", "return", "struct", "switch",
    "true", "var", "while",
    "NULL", "Self",
    "abstract", "active", "alignas", "alignof", "as", "asm", "asm_fragment",
    "async", "attribute", "auto", "await", "become", "cast", "catch", "class",
    "co_await", "co_return", "co_yield", "coherent", "column_major", "common",
    "compile", "compile_fragment", "concept", "const_cast", "consteval",
    "constexpr", "constinit", "crate", "debugger", "decltype", "delete",
    "demote", "demote_to_helper", "do", "dynamic_cast", "enum", "explicit",
    "export", "extends", "extern", "external", "fallthrough", "filter",
    "final", "finally", "friend", "from", "fxgroup", "get", "goto",
    "groupshared", "highp", "impl", "implements", "import", "inline",
    "instanceof", "interface", "layout", "lowp", "macro", "macro_rules",
    "match", "mediump", "meta", "mod", "module", "move", "mut", "mutable",
    "namespace", "new", "nil", "noexcept", "noinline", "nointerpolation",
    "non_coherent", "noncoherent", "noperspective", "null", "nullptr", "of",
    "operator", "package", "packoffset", "partition", "pass", "patch",
    "pixelfragment", "precise", "precision", "premerge", "priv", "protected",
    "pub", "public", "readonly", "ref", "regardless", "register",
    "reinterpret_cast", "require", "resource", "restrict", "self", "set",
    "shared", "sizeof", "smooth", "snorm", "static", "static_assert",
    "static_cast", "std", "subroutine", "super", "target", "template", "this",
    "thread_local", "throw", "trait", "try", "type", "typedef", "typeid",
    "typename", "typeof", "union", "unless", "unorm", "unsafe", "unsized",
    "use", "using", "varying", "virtual", "volatile", "wgsl", "where", "with",
    "writeonly", "yield",
];

/// True when `ty` is a GPU-qualified array type (`[T]'global`, `'unified`, `'const`,
/// `'actor'global`, `'actor'unified`, `'surface`) — i.e. `Type::Qualified(Array-like,
/// OwnerQual::Gpu*)`, the shape a `[T]'global`/etc. *parameter* (not a kernel struct field
/// — those are `KernelFieldDecl`, handled entirely differently by `wgsl_buffer_type`)
/// parses to. `wgsl_type`/`wgsl_scalar` have no arm at all for this shape — `wgsl_scalar`'s
/// final catch-all silently produces `i32`, the original bug this module fixes.
///
/// The naive-looking follow-up fix — give such a parameter a WGSL `ptr<storage,
/// array<T>, access>` type and an explicit `&` at each call site, mirroring the Metal
/// backend's analogous `msl_free_fn_param_type` fix for a `device T*` parameter — does
/// **not** work on this backend: confirmed against real naga source (`naga-22.1.0` and
/// `naga-29.0.4`, both), `valid::function::validate_function` unconditionally rejects any
/// pointer function parameter whose address space isn't `Function`/`Private` — `Storage`
/// is hard-rejected with "is a pointer of space Storage {..}, which can't be passed into
/// functions". This isn't a naga gap that's since been closed either: WGSL's own spec
/// names this exact capability `unrestricted_pointer_parameters`, and naga still lists it
/// as `UnimplementedLanguageExtension` in its latest release (tracking issue
/// gfx-rs/naga#5158, open). So a storage-buffer pointer can never be a function parameter
/// on this backend, full stop — there is no `enable`/`requires` directive that lifts it.
///
/// The actual fix: a kernel's buffer-qualified field is *already* a module-scope WGSL
/// global (`@group(0) @binding(N) var<storage, ...> name: array<T>;`, see
/// `wgsl_buffer_type`) — visible to every function in the module without needing to be
/// passed as a parameter at all. So a GPU-array-qualified parameter is dropped from the
/// emitted signature and from every call site's argument list entirely, and every
/// reference to it inside the function's own body is substituted with the resolved
/// global's name instead (see `current_gpu_array_subst`). The one thing this needs that
/// an ordinary parameter doesn't: knowing, for every such parameter, which single global
/// it always resolves to — computed by `resolve_gpu_array_params` from a whole-program
/// scan of every call site (see `ParamSlot`/`GpuBinding`'s own doc comments for the
/// resolution rule and its deliberate scope limit).
///
/// `'local` (`OwnerQual::GpuLocal`) is deliberately excluded: it has no storage-buffer
/// backing at all (a per-thread/private array), a materially different shape this fix
/// doesn't attempt.
fn is_gpu_array_qualified_type(ty: &Type) -> bool {
    let Type::Qualified(inner, oq) = ty else { return false };
    let is_array = matches!(inner.as_ref(), Type::Array(_) | Type::ArrayN(_, _))
        || inner.as_labeled_array().is_some();
    is_array && matches!(oq,
        OwnerQual::GpuUnified | OwnerQual::GpuGlobal | OwnerQual::GpuConst
        | OwnerQual::GpuActorGlobal | OwnerQual::GpuActorUnified | OwnerQual::GpuSurface)
}

/// How a device function/method parameter is emitted — see `is_gpu_array_qualified_type`'s
/// doc comment for why a GPU-array-qualified parameter needs this at all.
#[derive(Clone, PartialEq, Eq)]
enum ParamSlot {
    /// An ordinary parameter — kept in the emitted signature and every call site's
    /// argument list unchanged, exactly as before this fix.
    Plain,
    /// A GPU-array-qualified parameter, along with what's known so far about which single
    /// buffer it's always called with (see `GpuBinding`).
    Gpu(GpuBinding),
}

/// Resolution state for one `ParamSlot::Gpu` parameter, accumulated by
/// `resolve_gpu_array_params` scanning every call site of the owning function across the
/// whole program. Deliberately scoped to the direct case only — a GPU-array-qualified
/// parameter resolves *only* when every call site passes a literal kernel buffer field
/// name (`Var(name)` where `name` is one of that call site's enclosing kernel's own
/// buffer-qualified fields); anything else (a free function forwarding one of its own such
/// parameters to another free function, a computed expression, or two call sites
/// disagreeing on which buffer) is `Poisoned`. Forwarding chains are a real, valid Boring
/// pattern this doesn't attempt to resolve — WGSL's own hard constraint (see
/// `is_gpu_array_qualified_type`) means there is no way to represent an *unresolved*
/// buffer-qualified parameter on this backend at all (unlike Metal/CUDA, where it's just
/// another pointer), so "not directly bound to one field everywhere" and "genuinely
/// unsupported here" are the same outcome for this backend, and get the same clear
/// `/* ERROR: ... */` placeholder rather than silently wrong codegen.
#[derive(Clone, PartialEq, Eq)]
enum GpuBinding {
    /// No call site has been examined yet.
    Unbound,
    /// Every call site examined so far passed this same kernel buffer field, whose
    /// resolved WGSL global name (`{kernel}_{field}`, matching `current_buffer_renames`'s
    /// own scheme) is this string.
    Bound(String),
    /// Two call sites disagreed, or some call site passed something this fix can't
    /// resolve to a single buffer (a computed expression, a forwarded parameter, ...).
    Poisoned,
}

impl GpuBinding {
    fn resolved(&self) -> Option<&str> {
        match self { GpuBinding::Bound(g) => Some(g.as_str()), _ => None }
    }

    /// Folds in one more call site's evidence for this parameter — `None` means
    /// "unresolvable" (see `GpuBinding`'s doc comment), `Some(global)` a literal kernel
    /// buffer field's resolved WGSL name.
    fn merge(&mut self, evidence: Option<String>) {
        *self = match (&*self, evidence) {
            (GpuBinding::Poisoned, _) => GpuBinding::Poisoned,
            (_, None) => GpuBinding::Poisoned,
            (GpuBinding::Unbound, Some(g)) => GpuBinding::Bound(g),
            (GpuBinding::Bound(existing), Some(g)) => {
                if *existing == g { GpuBinding::Bound(existing.clone()) } else { GpuBinding::Poisoned }
            }
        };
    }
}

/// One call site's evidence for a GPU-array-qualified argument: `Some(global)` only when
/// `arg` is a bare `Var(name)` naming a buffer-qualified field of the kernel whose method
/// body this call site appears in (`kernel_ctx`) — resolved to that field's WGSL global
/// name, matching `current_buffer_renames`'s `{kernel}_{field}` scheme. `None` (meaning
/// "unresolvable", per `GpuBinding::merge`) for everything else, including a free-function
/// caller (`kernel_ctx: None` — a free function has no fields of its own to pass, only its
/// own already-GPU-array-qualified parameters, and forwarding those is out of scope, see
/// `GpuBinding`'s doc comment) and any non-`Var` argument expression.
fn gpu_array_call_evidence(arg: &Expr, kernel_ctx: Option<&KernelDecl>) -> Option<String> {
    let ExprKind::Var(name) = &arg.kind else { return None };
    let kernel = kernel_ctx?;
    let field = kernel.fields.iter().find(|f| &f.name == name)?;
    if !is_buffer_field(field) { return None; }
    Some(format!("{}_{}", kernel.name.to_lowercase(), name))
}

/// Scans every call site in `stmts` — recursing into every expression/statement shape
/// `DeviceEmitter::expr`/`emit_stmt` actually support in device code (this backend already
/// falls back to a `/* unsupported */` placeholder for anything else, so a call hidden
/// inside an unsupported shape can never reach real codegen anyway) — folding each GPU-
/// array-qualified argument's evidence into `bindings`. `kernel_ctx` is `Some(k)` while
/// scanning one of kernel `k`'s own methods (including its entry point), `None` while
/// scanning a free function's body (see `gpu_array_call_evidence`'s doc comment for why
/// that distinction matters).
fn resolve_calls_in_stmts(
    stmts: &[Stmt],
    kernel_ctx: Option<&KernelDecl>,
    bindings: &mut std::collections::HashMap<String, Vec<ParamSlot>>,
) {
    for s in stmts { resolve_calls_in_stmt(s, kernel_ctx, bindings); }
}

fn resolve_calls_in_stmt(
    stmt: &Stmt,
    kernel_ctx: Option<&KernelDecl>,
    bindings: &mut std::collections::HashMap<String, Vec<ParamSlot>>,
) {
    match stmt {
        Stmt::Let(l) => { if let Some(v) = &l.value { resolve_calls_in_expr(v, kernel_ctx, bindings); } }
        Stmt::Expr(e) => resolve_calls_in_expr(e, kernel_ctx, bindings),
        Stmt::Return(r) => { if let Some(v) = &r.value { resolve_calls_in_expr(v, kernel_ctx, bindings); } }
        Stmt::If(i) => {
            for (cond, body) in &i.branches {
                resolve_calls_in_expr(cond, kernel_ctx, bindings);
                resolve_calls_in_stmts(body, kernel_ctx, bindings);
            }
            if let Some(b) = &i.else_body { resolve_calls_in_stmts(b, kernel_ctx, bindings); }
        }
        Stmt::For(f) => {
            resolve_calls_in_expr(&f.iterable, kernel_ctx, bindings);
            resolve_calls_in_stmts(&f.body, kernel_ctx, bindings);
        }
        Stmt::While(w) => {
            resolve_calls_in_expr(&w.condition, kernel_ctx, bindings);
            resolve_calls_in_stmts(&w.body, kernel_ctx, bindings);
        }
        _ => {}
    }
}

fn resolve_calls_in_expr(
    expr: &Expr,
    kernel_ctx: Option<&KernelDecl>,
    bindings: &mut std::collections::HashMap<String, Vec<ParamSlot>>,
) {
    /// Folds each arg's evidence into `key`'s tracked `ParamSlot::Gpu` positions, if `key`
    /// is a tracked function at all (untracked names — builtins, non-reachable free
    /// functions — are silently ignored, same as elsewhere in this file).
    fn fold_call(
        key: &str,
        args: &[Arg],
        kernel_ctx: Option<&KernelDecl>,
        bindings: &mut std::collections::HashMap<String, Vec<ParamSlot>>,
    ) {
        let Some(slots) = bindings.get_mut(key) else { return };
        for (i, slot) in slots.iter_mut().enumerate() {
            if let ParamSlot::Gpu(state) = slot {
                let evidence = args.get(i).and_then(|a| gpu_array_call_evidence(&a.value, kernel_ctx));
                state.merge(evidence);
            }
        }
    }

    match &expr.kind {
        ExprKind::BinOp(_, l, r) => {
            resolve_calls_in_expr(l, kernel_ctx, bindings);
            resolve_calls_in_expr(r, kernel_ctx, bindings);
        }
        ExprKind::UnaryOp(_, v) => resolve_calls_in_expr(v, kernel_ctx, bindings),
        ExprKind::Assign(l, r) => {
            resolve_calls_in_expr(l, kernel_ctx, bindings);
            resolve_calls_in_expr(r, kernel_ctx, bindings);
        }
        ExprKind::Index(a, i) => {
            resolve_calls_in_expr(a, kernel_ctx, bindings);
            resolve_calls_in_expr(i, kernel_ctx, bindings);
        }
        ExprKind::Field(o, _) => resolve_calls_in_expr(o, kernel_ctx, bindings),
        ExprKind::Cast(inner, _) => resolve_calls_in_expr(inner, kernel_ctx, bindings),
        ExprKind::Call(callee, args) => {
            if let ExprKind::Var(name) = &callee.kind {
                fold_call(name, args, kernel_ctx, bindings);
            }
            resolve_calls_in_expr(callee, kernel_ctx, bindings);
            for a in args { resolve_calls_in_expr(&a.value, kernel_ctx, bindings); }
        }
        ExprKind::MethodCall(obj, method, args) => {
            if matches!(&obj.kind, ExprKind::Var(n) if n == "self") {
                if let Some(kernel) = kernel_ctx {
                    let key = format!("{}_{}", kernel.name, method);
                    fold_call(&key, args, kernel_ctx, bindings);
                }
            }
            resolve_calls_in_expr(obj, kernel_ctx, bindings);
            for a in args { resolve_calls_in_expr(&a.value, kernel_ctx, bindings); }
        }
        ExprKind::If(i) => {
            for (cond, body) in &i.branches {
                resolve_calls_in_expr(cond, kernel_ctx, bindings);
                resolve_calls_in_stmts(body, kernel_ctx, bindings);
            }
            if let Some(b) = &i.else_body { resolve_calls_in_stmts(b, kernel_ctx, bindings); }
        }
        _ => {}
    }
}

struct DeviceEmitter {
    out: String,
    indent: usize,
    current_fields: Vec<KernelFieldDecl>,
    current_kernel: String,
    auto_sync: bool,
    top_level_scalars: std::collections::HashMap<String, String>,
    /// Buffer/uniform field name → kernel-prefixed WGSL variable name for the kernel
    /// currently being emitted. All kernels share one WGSL module, so their storage/uniform
    /// bindings live in one flat global namespace — two kernels both declaring a field
    /// named `a` would otherwise emit two conflicting `var<storage,...> a: ...` at module
    /// scope. Declarations get the prefixed name; body references are rewritten to match.
    current_buffer_renames: std::collections::HashMap<String, String>,
    /// Block sizes per kernel, extracted from call sites: kernel_name → (bx, by, bz).
    block_sizes: std::collections::HashMap<String, (u32, u32, u32)>,
    /// Errors accumulated during emission (e.g. unsupported qualifiers). Propagated up
    /// through `emit_device_wgsl`'s return value and from there into `WgpuOutput::errors`
    /// (see `transpile_wgpu`) — this used to be silently dropped at the `emit_device_wgsl`
    /// boundary, so a program that tripped one of these pushes still built "successfully"
    /// with silently broken generated WGSL instead of failing with the diagnostic already
    /// computed here.
    pub errors: Vec<TranspileError>,
    /// Real-subgroup vs. shared-memory-emulated `gpu.warp.*` codegen (see `WarpMode`).
    mode: WarpMode,
    /// Set once per program by `emit_device_wgsl`: does any kernel in this
    /// program use `gpu.warp.*`? Gates the `enable subgroups;` directive (`Real`
    /// mode) and whether any per-kernel warp scaffolding is emitted at all.
    program_uses_warp: bool,
    /// Uniqueness counter for the temporary variable names the `Emulated` shuffle
    /// expansion introduces (`Stmt::Let` handling) — WGSL has no shadowing, so two
    /// shuffle call sites in the same kernel can't reuse the same temp names.
    warp_tmp_counter: u32,
    /// `name -> concrete_value` const-generic substitution for the kernel currently
    /// being emitted (e.g. `{"W": 3, "H": 1}` for a monomorphised `Blur_3_1`), from
    /// `super::resolve_effective_kernels`'s `kernel_consts` map — see that map's doc
    /// comment for why the body needs this even though `fields` are already
    /// substituted by `monomorphise_type`. Empty for a non-generic kernel.
    current_kernel_consts: std::collections::HashMap<String, i64>,
    /// True while emitting the body of a `void`-returning device function/method --
    /// see `metal::device`'s identical field.
    current_fn_is_void: bool,
    /// Emitted device-function name (`decl.name` for a free function, `"{kernel}_{method}"`
    /// for a kernel struct's own device helper method) → per-position resolution of its
    /// GPU-array-qualified parameters (`[T]'global`/`'unified`/`'const`/`'actor'global`/
    /// `'actor'unified`/`'surface`). Built once, up front, in `emit_program` by two
    /// whole-program passes (`ParamSlot`'s doc comment explains why this can't be answered
    /// per-function the way an ordinary parameter type can).
    gpu_array_params: std::collections::HashMap<String, Vec<ParamSlot>>,
    /// Parameter name → WGSL substitution text, for the device function/method *currently
    /// being emitted*'s own GPU-array-qualified parameters (see `ParamSlot`). Every
    /// occurrence of such a parameter name inside this body (`expr()`'s `ExprKind::Var` arm)
    /// is replaced by this text instead of being treated as an ordinary local — either the
    /// resolved WGSL global variable name (used exactly like a kernel buffer field
    /// reference, since that's genuinely what it is at that point) or an inline
    /// `/* ERROR: ... */` placeholder. Reset (overwritten wholesale) at the top of
    /// `emit_free_device_fn`/`emit_device_fn`/`emit_entry_point` — exactly one of those
    /// three is ever "current" during body emission.
    current_gpu_array_subst: std::collections::HashMap<String, String>,
}

impl DeviceEmitter {
    fn new(mode: WarpMode) -> Self {
        Self {
            out: String::new(),
            indent: 0,
            current_fields: vec![],
            current_kernel: String::new(),
            auto_sync: false,
            top_level_scalars: std::collections::HashMap::new(),
            current_buffer_renames: std::collections::HashMap::new(),
            block_sizes: std::collections::HashMap::new(),
            errors: vec![],
            mode,
            program_uses_warp: false,
            warp_tmp_counter: 0,
            current_kernel_consts: std::collections::HashMap::new(),
            current_fn_is_void: true,
            gpu_array_params: std::collections::HashMap::new(),
            current_gpu_array_subst: std::collections::HashMap::new(),
        }
    }

    fn line(&mut self, s: &str) {
        let ind = "    ".repeat(self.indent);
        self.out.push_str(&ind);
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn blank(&mut self) { self.out.push('\n'); }

    fn emit_program(
        &mut self,
        program: &Program,
        effective_kernels: &[crate::ast::KernelDecl],
        kernel_consts: &super::KernelConstsMap,
    ) {
        // Pre-pass: collect top-level scalar lets for inlining.
        for item in &program.items {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    let is_scalar = crate::transpiler::helpers::is_scalar_let_value(val, s.ty.as_ref());
                    if is_scalar {
                        let rhs = self.expr(val);
                        self.top_level_scalars.insert(s.name.clone(), rhs);
                    }
                }
            }
        }

        self.line("// Generated by boring build --target wgpu.");
        // `enable` directives must precede every other module-scope declaration —
        // only emitted in the `Real` module, and only when some kernel actually
        // uses `gpu.warp.*` (see `WarpMode`'s doc comment).
        if self.mode == WarpMode::Real && self.program_uses_warp {
            self.line("enable subgroups;");
        }
        self.blank();

        // Emit user-defined structs used in kernel fields.
        self.emit_user_structs(program);

        // Built-in Dimension type (i32 fields for easy arithmetic with thread indices).
        self.line("struct Dimension {");
        self.indent += 1;
        self.line("width: i32,");
        self.line("height: i32,");
        self.indent -= 1;
        self.line("}");
        self.blank();

        // Pre-pass: extract block sizes from kernel call sites.
        self.block_sizes = collect_block_sizes(program);

        // Emit free functions as WGSL helpers, but only those actually reachable from a
        // kernel's device code (`def ()` entry point or device helper methods). A `use`d
        // file merged into the same program for its GPU kernels may pull in ordinary
        // host-only helpers (e.g. a CPU array-building `zeros()`/`vec_add()`) that a kernel
        // never calls and that don't even have a valid WGSL translation (dynamic `.push()`
        // growth, `HashMap`, `String`, ...) -- emitting those unconditionally produced
        // invalid WGSL (`/* unsupported: ... */` placeholders) that failed shader
        // compilation at runtime even though nothing in the actual kernel graph needed them.
        let free_fns: std::collections::HashMap<&str, &FnDecl> = program.items.iter()
            .filter_map(|item| match item {
                Item::Fn(decl) if decl.qualifier.is_none() && !decl.task => Some((decl.name.as_str(), decl)),
                _ => None,
            })
            .collect();
        let mut reachable: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut frontier: Vec<String> = Vec::new();
        for decl in effective_kernels {
            for method in &decl.methods {
                collect_called_fn_names(&method.body, &mut frontier);
            }
        }
        while let Some(name) = frontier.pop() {
            if !free_fns.contains_key(name.as_str()) { continue; }
            if !reachable.insert(name.clone()) { continue; } // already visited
            collect_called_fn_names(&free_fns[name.as_str()].body, &mut frontier);
        }

        // Pre-pass: resolve every GPU-array-qualified parameter's binding (see
        // `is_gpu_array_qualified_type`'s doc comment for why this exists at all) — a
        // single whole-program pass, up front, so it's ready before any function body
        // (whose own emission needs it, via `current_gpu_array_subst`) is emitted.
        let mut gpu_array_params: std::collections::HashMap<String, Vec<ParamSlot>> = std::collections::HashMap::new();
        for name in &reachable {
            let decl = free_fns[name.as_str()];
            let slots = decl.params.iter()
                .map(|p| if p.ty.as_ref().is_some_and(is_gpu_array_qualified_type) {
                    ParamSlot::Gpu(GpuBinding::Unbound)
                } else {
                    ParamSlot::Plain
                })
                .collect();
            gpu_array_params.insert(decl.name.clone(), slots);
        }
        for decl in effective_kernels {
            for method in &decl.methods {
                if method.name.is_empty() { continue; }
                let key = format!("{}_{}", decl.name, method.name);
                let slots = method.params.iter()
                    .map(|p| if p.ty.as_ref().is_some_and(is_gpu_array_qualified_type) {
                        ParamSlot::Gpu(GpuBinding::Unbound)
                    } else {
                        ParamSlot::Plain
                    })
                    .collect();
                gpu_array_params.insert(key, slots);
            }
        }
        for name in &reachable {
            resolve_calls_in_stmts(&free_fns[name.as_str()].body, None, &mut gpu_array_params);
        }
        for decl in effective_kernels {
            for method in &decl.methods {
                resolve_calls_in_stmts(&method.body, Some(decl), &mut gpu_array_params);
            }
        }
        self.gpu_array_params = gpu_array_params;

        for item in &program.items {
            if let Item::Fn(decl) = item {
                if decl.qualifier.is_none() && !decl.task && reachable.contains(decl.name.as_str()) {
                    self.emit_free_device_fn(decl);
                    self.blank();
                }
            }
        }

        for decl in effective_kernels {
            self.current_kernel_consts = kernel_consts.get(&decl.name).cloned().unwrap_or_default();
            self.emit_kernel_decl(decl);
        }
    }

    /// Builds `current_gpu_array_subst` for the device function/method named `key`
    /// (`decl.name` for a free function, `"{kernel}_{method}"` for a kernel device
    /// method) from its resolved `gpu_array_params` entry — see `current_gpu_array_subst`'s
    /// doc comment for what the substitution text means in each case.
    ///
    /// An unresolved (`Poisoned`/`Unbound`) parameter still gets an inline `/* ERROR: ... */`
    /// WGSL comment at its every reference (readable if someone inspects the generated
    /// shader directly), but that alone doesn't fail the build — naga happily compiles the
    /// comment plus whatever's emitted after it. So this also pushes a real error onto
    /// `self.errors`, which `emit_device_wgsl`/`transpile_wgpu` surface as an actual build
    /// failure with a clear diagnostic instead.
    fn build_gpu_array_subst(&mut self, key: &str, params: &[Param]) -> std::collections::HashMap<String, String> {
        let mut subst = std::collections::HashMap::new();
        let slots = match self.gpu_array_params.get(key) {
            Some(slots) => slots.clone(),
            None => return subst,
        };
        for (p, slot) in params.iter().zip(slots.iter()) {
            if let ParamSlot::Gpu(state) = slot {
                let text = match state.resolved() {
                    Some(global) => global.to_string(),
                    None => {
                        self.errors.push(TranspileError::at(format!(
                            "`{}`'s GPU-array-qualified parameter `{}` is not supported on \
                             --target wgpu here — WGSL forbids passing a storage buffer as a function \
                             parameter at all (the `unrestricted_pointer_parameters` WGSL extension isn't \
                             implemented by naga/wgpu — see gfx-rs/naga#5158); this parameter must be bound \
                             to the exact same kernel buffer field at every call site in the program",
                            key, p.name
                        ), p.line, p.col));
                        format!(
                            "/* ERROR: `{}`'s GPU-array-qualified parameter `{}` is not supported on \
                             --target wgpu here — see the reported diagnostic for details */ i32",
                            key, p.name
                        )
                    }
                };
                subst.insert(p.name.clone(), text);
            }
        }
        subst
    }

    /// Parameters of `key` that are GPU-array-qualified — such a parameter is dropped
    /// entirely from the emitted signature (see `is_gpu_array_qualified_type`'s doc
    /// comment): every reference to it inside the body is substituted via
    /// `current_gpu_array_subst` instead, so it needs no parameter slot of its own.
    fn plain_params_wgsl(&self, key: &str, params: &[Param]) -> Vec<String> {
        let empty = Vec::new();
        let slots = self.gpu_array_params.get(key).unwrap_or(&empty);
        params.iter().enumerate().filter_map(|(i, p)| {
            if matches!(slots.get(i), Some(ParamSlot::Gpu(_))) {
                None
            } else {
                let ty = p.ty.as_ref().map(wgsl_type).unwrap_or_else(|| "i32".into());
                Some(format!("{}: {}", wgsl_safe_ident(&p.name), ty))
            }
        }).collect()
    }

    fn emit_free_device_fn(&mut self, decl: &crate::ast::FnDecl) {
        let ret = decl.return_ty.as_ref().map(wgsl_type).unwrap_or_else(|| "void".into());
        self.current_gpu_array_subst = self.build_gpu_array_subst(&decl.name, &decl.params);
        let params: Vec<String> = self.plain_params_wgsl(&decl.name, &decl.params);
        self.line(&format!("fn {}({}) -> {} {{", decl.name, params.join(", "), ret));
        self.indent += 1;
        self.current_fn_is_void = ret == "void";
        let last_idx = decl.body.len().saturating_sub(1);
        for (i, stmt) in decl.body.iter().enumerate() { self.emit_stmt(stmt, i == last_idx); }
        self.indent -= 1;
        self.line("}");
    }

    /// Emit WGSL struct declarations for Boring structs referenced in kernel fields.
    fn emit_user_structs(&mut self, program: &Program) {
        // Collect struct names used in kernel fields.
        let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
        for item in &program.items {
            if let Item::Kernel(decl) = item {
                for f in &decl.fields {
                    if let Some(n) = inner_named_type(&f.ty) {
                        used.insert(n.to_string());
                    }
                }
            }
        }
        for item in &program.items {
            if let Item::Struct(s) = item {
                if used.contains(&s.name) {
                    self.line(&format!("struct {} {{", s.name));
                    self.indent += 1;
                    for f in &s.fields {
                        self.line(&format!("{}: {},", f.name, wgsl_type(&f.ty)));
                    }
                    self.indent -= 1;
                    self.line("}");
                    self.blank();
                }
            }
        }
    }

    fn emit_kernel_decl(&mut self, decl: &KernelDecl) {
        self.current_fields = decl.fields.clone();
        self.current_kernel = decl.name.clone();
        self.current_buffer_renames = decl.fields.iter()
            .filter(|f| is_buffer_field(f))
            .map(|f| (f.name.clone(), format!("{}_{}", decl.name.to_lowercase(), f.name)))
            .collect();

        // Validate: reject dynamic 'sync fields ([T]'sync without size).
        for f in &decl.fields {
            if matches!(f.qual, GpuQual::Actor)
                && matches!(f.ty, Type::Array(_)) {
                    self.errors.push(TranspileError::at(format!(
                        "kernel {}: dynamic '[T]'sync field '{}' is not supported on --target wgpu — \
                         WGSL requires a compile-time workgroup size; use '[T, N]'sync' instead",
                        decl.name, f.name
                    ), f.line, f.col));
                }
        }

        self.line(&format!("// ─── kernel {} ───", decl.name));
        self.blank();

        // Collect all fields that need bindings: unified, global, actor, surface, const.
        // Scalar local/const fields go into a params uniform struct.
        let has_params = decl.fields.iter().any(is_params_field);
        let mut binding: u32 = 0;

        // 1. Array buffer fields.
        for f in &decl.fields {
            if is_buffer_field(f) {
                let (access, ty) = wgsl_buffer_type(f);
                let var_name = &self.current_buffer_renames[&f.name];
                self.line(&format!("@group(0) @binding({}) var<storage, {}> {}: {};",
                    binding, access, var_name, ty));
                binding += 1;
            }
        }

        // 2. Uniform params struct (scalars + fixed arrays + Dimension fields).
        if has_params {
            self.emit_params_struct(decl, binding);
            binding += 1;
        }
        let _ = binding;

        // 3. Workgroup ('sync fixed arrays) — must be module-scope `var<workgroup>`
        // declarations in WGSL, not statements inside the function body.
        for f in &decl.fields {
            if matches!(f.qual, GpuQual::Actor) {
                if let Type::ArrayN(inner, n) = &f.ty {
                    self.line(&format!("var<workgroup> {}: array<{}, {}>;",
                        wgsl_workgroup_array_ident(&decl.name, &f.name), wgsl_scalar(inner), n));
                } else if let Some((elem, _)) = f.ty.as_labeled_array() {
                    // See cuda::device's identical `labeled_array_len()` note:
                    // `None` (a const-generic axis) isn't handled here yet.
                    if let Some(len) = f.ty.labeled_array_len() {
                        self.line(&format!("var<workgroup> {}: array<{}, {}>;",
                            wgsl_workgroup_array_ident(&decl.name, &f.name), wgsl_scalar(elem), len));
                    }
                }
            }
        }

        // 3.5. `Emulated`-mode `gpu.warp.shuffle_*` scratch buffers — one per
        // distinct WGSL scalar type actually shuffled in this kernel, sized to
        // its workgroup thread count (see `WarpMode`'s doc comment).
        if self.mode == WarpMode::Emulated && self.program_uses_warp {
            let mut elem_types = std::collections::BTreeSet::new();
            for m in &decl.methods {
                collect_shuffle_elem_types_stmts(&m.body, &decl.fields, &mut elem_types);
            }
            if !elem_types.is_empty() {
                let (bx, by, bz) = self.block_sizes.get(&decl.name)
                    .or_else(|| self.block_sizes.get(&decl.name.to_lowercase()))
                    .copied().unwrap_or((1, 1, 1));
                let wg_len = bx * by * bz;
                for ty in &elem_types {
                    self.line(&format!("var<workgroup> {}: array<{}, {}>;",
                        warp_scratch_var_name(&decl.name, ty), ty, wg_len));
                }
            }
        }

        self.blank();

        self.blank();

        // Device helper functions (named def methods).
        for method in &decl.methods {
            if !method.name.is_empty() {
                self.emit_device_fn(&decl.name, method);
                self.blank();
            }
        }

        // Entry point: `def ()`.
        if let Some(entry) = decl.methods.iter().find(|m| m.name.is_empty()) {
            self.emit_entry_point(decl, entry);
            self.blank();
        }
    }

    fn emit_params_struct(&mut self, decl: &KernelDecl, binding: u32) {
        let struct_name = format!("{}Params", decl.name);
        self.line(&format!("struct {} {{", struct_name));
        self.indent += 1;
        for f in &decl.fields {
            if is_params_field(f) {
                let name = wgsl_safe_ident(&f.name);
                match &f.ty {
                    Type::Named(n) if n == "Dimension" => {
                        self.line(&format!("{name}_w: i32,"));
                        self.line(&format!("{name}_h: i32,"));
                    }
                    Type::ArrayN(inner, n) => {
                        self.line(&format!("{}: array<{}, {}>,", name, wgsl_scalar(inner), n));
                    }
                    ty if ty.as_labeled_array().is_some() && ty.labeled_array_len().is_some() => {
                        let (elem, _) = ty.as_labeled_array().unwrap();
                        let len = ty.labeled_array_len().unwrap();
                        self.line(&format!("{}: array<{}, {}>,", name, wgsl_scalar(elem), len));
                    }
                    _ => {
                        self.line(&format!("{}: {},", name, wgsl_scalar(&f.ty)));
                    }
                }
            }
        }
        self.indent -= 1;
        self.line("}");
        let var_name = format!("{}_params", decl.name.to_lowercase());
        // See `super::kernel_params_use_storage`'s doc comment: a fixed-size
        // scalar array field can't live in `var<uniform>` (std140 requires a
        // 16-byte array stride there) — fall back to `var<storage, read>`
        // (std430, 4-byte stride) whenever one is present.
        let addr_space = if super::kernel_params_use_storage(decl) { "storage, read" } else { "uniform" };
        self.line(&format!(
            "@group(0) @binding({}) var<{}> {}: {};",
            binding, addr_space, var_name, struct_name
        ));
        self.blank();
    }

    fn emit_device_fn(&mut self, kernel: &str, method: &FnDecl) {
        let ret = method.return_ty.as_ref()
            .map(wgsl_type)
            .unwrap_or_else(|| "void".into());
        let fn_name = format!("{}_{}", kernel, method.name);
        self.current_gpu_array_subst = self.build_gpu_array_subst(&fn_name, &method.params);
        let params: Vec<String> = self.plain_params_wgsl(&fn_name, &method.params);
        if ret == "void" {
            self.line(&format!("fn {}({}) {{", fn_name, params.join(", ")));
        } else {
            self.line(&format!("fn {}({}) -> {} {{", fn_name, params.join(", "), ret));
        }
        self.indent += 1;
        self.current_fn_is_void = ret == "void";
        let last_idx = method.body.len().saturating_sub(1);
        for (i, stmt) in method.body.iter().enumerate() { self.emit_stmt(stmt, i == last_idx); }
        self.indent -= 1;
        self.line("}");
    }

    fn emit_entry_point(&mut self, decl: &KernelDecl, entry: &FnDecl) {
        // The entry point is always `void` (a GPU kernel entry has no return value).
        self.current_fn_is_void = true;
        // The entry point never has its own GPU-array-qualified extra parameters (only
        // kernel fields, already plain module-scope `var<storage, ...>` names) — clear
        // whatever the last-emitted device helper method left behind.
        self.current_gpu_array_subst.clear();
        let fn_name = format!("{}_main", decl.name);
        let uses_warp = self.program_uses_warp && super::kernel_uses_gpu_warp(decl);

        let (bx, by, bz) = self.block_sizes.get(&decl.name)
            .or_else(|| self.block_sizes.get(&decl.name.to_lowercase()))
            .copied().unwrap_or((1, 1, 1));
        self.line(&format!("@compute @workgroup_size({bx}, {by}, {bz})"));
        self.line(&format!("fn {}(", fn_name));
        self.indent += 1;
        self.line("@builtin(local_invocation_id) bp_tid:  vec3<u32>,");
        self.line("@builtin(workgroup_id)         bp_bid:  vec3<u32>,");
        self.line("@builtin(num_workgroups)       bp_gdim: vec3<u32>,");
        if uses_warp {
            match self.mode {
                // Real subgroup builtins directly give the lane/size values.
                WarpMode::Real => {
                    self.line("@builtin(subgroup_size)           bp_wsize: u32,");
                    self.line("@builtin(subgroup_invocation_id)  bp_lane:  u32,");
                }
                // No subgroup builtins on this path — derive lane/size from the
                // flattened in-workgroup index instead (synthesized below).
                WarpMode::Emulated => {
                    self.line("@builtin(local_invocation_index) bp_lidx: u32,");
                }
            }
        }
        self.indent -= 1;
        self.line(") {");
        self.indent += 1;
        self.line(&format!("let bp_bdim = vec3<u32>({bx}u, {by}u, {bz}u);"));
        if uses_warp && self.mode == WarpMode::Emulated {
            self.line(&format!("let bp_wsize: u32 = {}u;", EMULATED_WARP_SIZE));
            self.line(&format!("let bp_lane: u32 = bp_lidx % {}u;", EMULATED_WARP_SIZE));
        }

        // Unpack 'const scalars and Dimension fields from params struct.
        let has_params = decl.fields.iter().any(is_params_field);
        if has_params {
            let pvar = format!("{}_params", decl.name.to_lowercase());
            for f in &decl.fields {
                if is_params_field(f) {
                    let name = wgsl_safe_ident(&f.name);
                    match &f.ty {
                        Type::Named(n) if n == "Dimension" => {
                            self.line(&format!("let {name}: Dimension = Dimension({pvar}.{name}_w, {pvar}.{name}_h);"));
                        }
                        Type::ArrayN(_, _) => {
                            // Fixed arrays are accessed as {pvar}.field[i] directly.
                        }
                        _ => {
                            self.line(&format!("let {name}: {} = {pvar}.{name};", wgsl_scalar(&f.ty)));
                        }
                    }
                }
            }
        }

        // Declare 'local scalar fields as function vars.
        // Skip scalars already unpacked from the params uniform (is_params_field covers those).
        for f in &decl.fields {
            if matches!(f.qual, GpuQual::Local) && !is_params_field(f) {
                match &f.ty {
                    Type::Array(_) | Type::ArrayN(_, _) => {}
                    _ => {
                        self.line(&format!("var {}: {} = {};",
                            f.name, wgsl_scalar(&f.ty), wgsl_zero(&f.ty)));
                    }
                }
            }
        }

        let has_sync_fields = decl.fields.iter().any(|f| matches!(f.qual, GpuQual::Actor));
        self.auto_sync = has_sync_fields && !body_has_explicit_sync(&entry.body);

        if self.auto_sync {
            let split = first_loop_index(&entry.body);
            for stmt in &entry.body[..split] { self.emit_stmt(stmt, false); }
            if split < entry.body.len() {
                self.line("workgroupBarrier();");
            }
            for stmt in &entry.body[split..] { self.emit_stmt(stmt, false); }
        } else {
            for stmt in &entry.body { self.emit_stmt(stmt, false); }
        }

        self.indent -= 1;
        self.line("}");
    }

    // ── Statements ────────────────────────────────────────────────────────────

    /// Expands `let <name> = gpu.warp.shuffle_down/up/xor/shuffle(v, n)` into the
    /// shared-memory emulation: write this thread's value into a workgroup-scoped
    /// scratch slot, barrier, read the target lane's slot (falling back to the
    /// caller's own value if the target lane isn't a real participant — matches
    /// real hardware `_sync` shuffle intrinsics at a warp boundary), barrier again.
    /// Every temp name is suffixed with a per-emitter counter — WGSL has no
    /// shadowing, so two call sites in the same kernel can't reuse one name.
    fn emit_emulated_shuffle_let(&mut self, kw: &str, name: &str, ty: Option<&Type>, method: &str, args: &[Arg]) {
        let v = self.expr(&args[0].value);
        let operand = self.expr(&args[1].value);
        let elem_ty = infer_shuffle_elem_type(&args[0].value, &self.current_fields);
        let scratch = warp_scratch_var_name(&self.current_kernel, &elem_ty);
        let n = self.warp_tmp_counter;
        self.warp_tmp_counter += 1;

        self.line(&format!("{}[bp_lidx] = {};", scratch, v));
        self.line("workgroupBarrier();");
        let target_expr = match method {
            "shuffle_down" => format!("i32(bp_lane) + i32({})", operand),
            "shuffle_up"   => format!("i32(bp_lane) - i32({})", operand),
            "shuffle_xor"  => format!("i32(bp_lane) ^ i32({})", operand),
            "shuffle"      => format!("i32({})", operand),
            _ => unreachable!("is_gpu_warp_shuffle already restricts `method`"),
        };
        self.line(&format!("let bp_warp_target_{n}: i32 = {target_expr};"));
        self.line(&format!(
            "let bp_warp_valid_{n}: bool = (bp_warp_target_{n} >= 0) && (bp_warp_target_{n} < i32({}u));",
            EMULATED_WARP_SIZE,
        ));
        // `bp_lidx - bp_lane` is this thread's simulated warp's starting index
        // within the workgroup — `workgroupBarrier()` is workgroup-wide (WGSL has
        // no sub-workgroup barrier), but the scratch read must still stay scoped
        // to the calling thread's own simulated warp, exactly like a real
        // subgroup shuffle only exchanges data within one hardware subgroup.
        self.line(&format!(
            "let bp_warp_base_{n}: u32 = bp_lidx - bp_lane;"
        ));
        self.line(&format!(
            "let bp_warp_idx_{n}: u32 = bp_warp_base_{n} + u32(clamp(bp_warp_target_{n}, 0, i32({}u) - 1));",
            EMULATED_WARP_SIZE,
        ));
        let decl_ty = ty.map(wgsl_type).unwrap_or_else(|| elem_ty.clone());
        self.line(&format!(
            "{kw} {name}: {decl_ty} = select({scratch}[bp_lidx], {scratch}[bp_warp_idx_{n}], bp_warp_valid_{n});"
        ));
        self.line("workgroupBarrier();");
    }

    /// Rewrites `e`, replacing every `gpu.warp.shuffle_down/up/xor/shuffle(...)`
    /// call reachable through arithmetic/index/cast/call nesting with a fresh
    /// temp variable, emitting that shuffle's write/barrier/read/barrier
    /// expansion (`emit_emulated_shuffle_let`) as a side effect before the
    /// statement using the rewritten expression. This is what lets the
    /// realistic reduction idiom `v = v + gpu.warp.shuffle_xor(v, mask)` work
    /// under the emulated fallback, not just `let x = gpu.warp.shuffle_down(...)`
    /// alone — WGSL has no side-effecting expressions, so a shuffle used
    /// anywhere but the entire RHS of a `let`/assignment must be hoisted out
    /// into its own preceding statement first. Only `Emulated` mode calls this;
    /// mirrors exactly the shapes `collect_shuffle_types_expr` scans for.
    fn hoist_shuffles(&mut self, e: &Expr) -> Expr {
        if let ExprKind::MethodCall(obj, method, args) = &e.kind {
            if is_gpu_warp_receiver(obj) && is_gpu_warp_shuffle(method) && !args.is_empty() {
                let tmp = format!("bp_warp_hoist_{}", self.warp_tmp_counter);
                self.emit_emulated_shuffle_let("let", &tmp, None, method, args);
                return Expr { kind: ExprKind::Var(tmp), line: e.line, col: e.col, len: e.len };
            }
        }
        let kind = match &e.kind {
            ExprKind::BinOp(op, l, r) =>
                ExprKind::BinOp(op.clone(), Box::new(self.hoist_shuffles(l)), Box::new(self.hoist_shuffles(r))),
            ExprKind::UnaryOp(op, x) => ExprKind::UnaryOp(op.clone(), Box::new(self.hoist_shuffles(x))),
            ExprKind::Cast(x, ty) => ExprKind::Cast(Box::new(self.hoist_shuffles(x)), ty.clone()),
            ExprKind::Index(a, i) =>
                ExprKind::Index(Box::new(self.hoist_shuffles(a)), Box::new(self.hoist_shuffles(i))),
            ExprKind::Call(callee, args) => ExprKind::Call(
                callee.clone(),
                args.iter().map(|a| Arg { value: self.hoist_shuffles(&a.value), ..a.clone() }).collect(),
            ),
            ExprKind::MethodCall(obj, method, args) => ExprKind::MethodCall(
                Box::new(self.hoist_shuffles(obj)),
                method.clone(),
                args.iter().map(|a| Arg { value: self.hoist_shuffles(&a.value), ..a.clone() }).collect(),
            ),
            other => other.clone(),
        };
        Expr { kind, line: e.line, col: e.col, len: e.len }
    }

    fn emit_stmt(&mut self, stmt: &Stmt, is_last: bool) {
        match stmt {
            Stmt::Let(s) => {
                let mutable = matches!(s.binding, BindingKind::Mut | BindingKind::Var | BindingKind::Lazy);
                let kw = if mutable { "var" } else { "let" };
                if let Some(val) = &s.value {
                    let rewritten;
                    let val = if self.mode == WarpMode::Emulated {
                        rewritten = self.hoist_shuffles(val);
                        &rewritten
                    } else {
                        val
                    };
                    // Non-atomic `arr[idx].min/max/swap/cas(...)` needs two
                    // real WGSL statements (no statement-expression in WGSL)
                    // — see `try_emit_plain_index_method_stmt`'s own doc.
                    let ty_suffix = s.ty.as_ref().map(|t| format!(": {}", wgsl_type(t))).unwrap_or_default();
                    let name = wgsl_safe_ident(&s.name);
                    if self.try_emit_plain_index_method_stmt(Some((kw, &ty_suffix)), &name, val) {
                        return;
                    }
                    let rhs = self.expr(val);
                    if let Some(ty) = &s.ty {
                        self.line(&format!("{} {}: {} = {};", kw, name, wgsl_type(ty), rhs));
                    } else {
                        self.line(&format!("{} {} = {};", kw, name, rhs));
                    }
                }
            }
            Stmt::Expr(e) => {
                match &e.kind {
                    // `print` → silent no-op (no device-side print in WGSL).
                    ExprKind::Call(callee, _)
                        if matches!(&callee.kind, ExprKind::Var(n) if n == "print") => {}
                    ExprKind::Assign(lhs, rhs) => {
                        let plain_handled = if let ExprKind::Var(lhs_name) = &lhs.kind {
                            // `_` is WGSL's phony *write-only* discard target — it
                            // can never be read back (confirmed via a real naga
                            // parse: "no definition in scope for identifier: '_'"
                            // on `min(_, 9)`), but the plain (non-atomic) min/max/cas
                            // codegen needs to *read* the captured old value to
                            // compute the update. For `_ = buf[i].min(v)`, declare a
                            // fresh synthetic `let` instead of assigning to `_`
                            // directly — the real `_` target is never referenced
                            // again either way, so an unused real binding is
                            // harmless; for an ordinary already-declared variable
                            // being reassigned, emit a plain assignment as usual.
                            if lhs_name == "_" {
                                let n = self.warp_tmp_counter;
                                self.warp_tmp_counter += 1;
                                // WGSL reserves identifiers starting with `__`
                                // (confirmed via a real naga parse: "Identifier
                                // starts with a reserved prefix: '__boring_discard_0'"
                                // — same reserved-prefix constraint already noted
                                // elsewhere in this codebase for `__params`) — use
                                // the same `bp_` (single underscore) convention the
                                // existing shuffle-hoist temp names already do.
                                let tmp = format!("bp_discard_{}", n);
                                self.try_emit_plain_index_method_stmt(Some(("let", "")), &tmp, rhs)
                            } else {
                                let name = wgsl_safe_ident(lhs_name);
                                self.try_emit_plain_index_method_stmt(None, &name, rhs)
                            }
                        } else {
                            false
                        };
                        if let Some(line) = self.try_atomic_assign(lhs, rhs) {
                            self.line(&line);
                        } else if plain_handled {
                            // Already emitted by `try_emit_plain_index_method_stmt`
                            // above: `{lhs} = {target}; {target} = ...;` in place
                            // of the ordinary single-expression assignment below.
                        } else {
                            let l = self.expr(lhs);
                            let rewritten;
                            let rhs = if self.mode == WarpMode::Emulated {
                                rewritten = self.hoist_shuffles(rhs);
                                &rewritten
                            } else {
                                rhs
                            };
                            let r = self.expr(rhs);
                            self.line(&format!("{} = {};", l, r));
                        }
                    }
                    _ => {
                        let s = self.expr(e);
                        if is_last && !self.current_fn_is_void {
                            self.line(&format!("return {};", s));
                        } else {
                            self.line(&format!("{};", s));
                        }
                    }
                }
            }
            Stmt::Return(r) => {
                if let Some(val) = &r.value {
                    let s = self.expr(val);
                    self.line(&format!("return {};", s));
                } else {
                    self.line("return;");
                }
            }
            Stmt::If(i) => {
                for (idx, (cond, body)) in i.branches.iter().enumerate() {
                    let c = self.expr(cond);
                    if idx == 0 { self.line(&format!("if ({}) {{", c)); }
                    else        { self.line(&format!("}} else if ({}) {{", c)); }
                    self.indent += 1;
                    let last_idx = body.len().saturating_sub(1);
                    for (j, s) in body.iter().enumerate() { self.emit_stmt(s, is_last && j == last_idx); }
                    self.indent -= 1;
                }
                if let Some(else_body) = &i.else_body {
                    self.line("} else {");
                    self.indent += 1;
                    let last_idx = else_body.len().saturating_sub(1);
                    for (j, s) in else_body.iter().enumerate() { self.emit_stmt(s, is_last && j == last_idx); }
                    self.indent -= 1;
                }
                self.line("}");
            }
            Stmt::While(w) => {
                let cond = self.expr(&w.condition);
                self.line("loop {");
                self.indent += 1;
                self.line(&format!("if !({})", cond));
                self.indent += 1;
                self.line("{ break; }");
                self.indent -= 1;
                if self.auto_sync && body_accesses_sync_field(&w.body, &self.current_fields) {
                    self.line("workgroupBarrier();");
                }
                for s in &w.body { self.emit_stmt(s, false); }
                self.indent -= 1;
                self.line("}");
            }
            Stmt::For(f) => {
                let var = wgsl_safe_ident(&f.vars.first().cloned().unwrap_or_else(|| "_i".into()));
                // Check for negated range: UnaryOp(Neg, Range{...}) — e.g. `for dy in -1..<2`
                let neg_range = if let ExprKind::UnaryOp(UnaryOp::Neg, ref inner) = f.iterable.kind {
                    if let ExprKind::Range { start, end, inclusive } = &inner.kind {
                        Some((format!("-{}", self.expr(start)), self.expr(end), *inclusive))
                    } else { None }
                } else { None };
                if let Some((lo, hi, inclusive)) = neg_range {
                    let op = if inclusive { "<=" } else { "<" };
                    // Wrapped in its own block: WGSL has no shadowing, so sibling for-loops
                    // reusing the same loop-variable name (e.g. two `for j in ..<n` in the
                    // same enclosing scope) would otherwise emit two `var j` in one block,
                    // which naga rejects as a redefinition.
                    self.line("{");
                    self.indent += 1;
                    self.line(&format!("var {var}: i32 = {lo};"));
                    self.line("loop {");
                    self.indent += 1;
                    self.line(&format!("if !({var} {op} {hi}) {{ break; }}"));
                    if self.auto_sync && body_accesses_sync_field(&f.body, &self.current_fields) {
                        self.line("workgroupBarrier();");
                    }
                    for s in &f.body { self.emit_stmt(s, false); }
                    self.line(&format!("{var} = {var} + 1;"));
                    self.indent -= 1;
                    self.line("}");
                    self.indent -= 1;
                    self.line("}");
                } else { match &f.iterable.kind {
                    ExprKind::Range { start, end, inclusive } => {
                        let lo = self.expr(start);
                        let hi = self.expr(end);
                        let op = if *inclusive { "<=" } else { "<" };
                        // See comment above: own block to scope the loop variable.
                        self.line("{");
                        self.indent += 1;
                        self.line(&format!("var {var}: i32 = {lo};"));
                        self.line("loop {");
                        self.indent += 1;
                        self.line(&format!("if !({var} {op} {hi}) {{ break; }}"));
                        if self.auto_sync && body_accesses_sync_field(&f.body, &self.current_fields) {
                            self.line("workgroupBarrier();");
                        }
                        for s in &f.body { self.emit_stmt(s, false); }
                        self.line(&format!("{var} = {var} + 1;"));
                        self.indent -= 1;
                        self.line("}");
                        self.indent -= 1;
                        self.line("}");
                    }
                    _ => {
                        let iter = self.expr(&f.iterable);
                        self.line(&format!("/* for {var} in {iter} -- unsupported in WGSL */"));
                    }
                } } // end else { match
            }
            Stmt::Break(_label, _val) => self.line("break;"),
            Stmt::Continue(_label)    => self.line("continue;"),
            // `sync` statement → explicit workgroup barrier.
            Stmt::Comment(c) if c == "sync" => {
                self.line("workgroupBarrier();");
            }
            Stmt::Comment(_) => {}
            _ => { self.line("/* unsupported stmt in wgsl kernel */"); }
        }
    }

    /// `'actor'global` or `'actor'unified` — both are `atomic<T>` storage buffers in
    /// WGSL (see `wgsl_buffer_type`); only the buffer-usage flags differ (host.rs).
    fn is_atomic_field(&self, name: &str) -> bool {
        self.current_fields.iter().any(|f|
            f.name == name && matches!(f.qual, GpuQual::ActorGlobal | GpuQual::ActorUnified))
    }

    /// Detect `arr[i] OP= v` on an `'actor'global`/`'actor'unified` field → WGSL atomic
    /// intrinsic.
    fn try_atomic_assign(&mut self, lhs: &Expr, rhs: &Expr) -> Option<String> {
        let ExprKind::Index(arr, idx) = &lhs.kind else { return None; };
        let arr_name = match &arr.kind {
            ExprKind::Var(n) => n.clone(),
            _ => return None,
        };
        if !self.is_atomic_field(&arr_name) { return None; }
        let ExprKind::BinOp(op, _lhs_copy, value) = &rhs.kind else { return None; };

        let i = self.expr(idx);
        let v = self.expr(value);
        // Same cross-kernel namespacing as the general Var case (see its comment) —
        // this bypasses self.expr() entirely so it needs its own rename lookup.
        let arr_name = self.current_buffer_renames.get(&arr_name).cloned().unwrap_or(arr_name);
        // WGSL atomics require a pointer to the element. `u32(...)` (function-call
        // cast), not `... as u32` (invalid WGSL -- confirmed via a real naga parse:
        // "expected ']', found 'as'"; the general Index case just below already
        // gets this right). This was a real, pre-existing bug: every atomic op
        // emitted through this path -- `+= -= &= |= ^=` -- was unparseable WGSL,
        // undetected because `cargo check` only validates the Rust host side, not
        // the shader naga compiles at runtime.
        let ptr = format!("&{}[u32({})]", arr_name, i);

        let call = match op {
            BinOp::Add    => format!("atomicAdd({}, {})", ptr, v),
            BinOp::Sub    => format!("atomicSub({}, {})", ptr, v),
            BinOp::BitOr  => format!("atomicOr({}, {})", ptr, v),
            BinOp::BitAnd => format!("atomicAnd({}, {})", ptr, v),
            BinOp::BitXor => format!("atomicXor({}, {})", ptr, v),
            _ => return None,
        };
        Some(format!("{};", call))
    }

    /// Detect `arr[i].min/max/swap/cas(...)` where `arr` is an
    /// `'actor'global`/`'actor'unified` field and emit the corresponding
    /// WGSL atomic builtin. Handled in expression position — unlike
    /// `try_atomic_assign`'s statement-only compound-assign desugar, these
    /// return the previous value.
    ///
    /// `min`/`max`/`swap` map directly onto WGSL's
    /// `atomicMin`/`atomicMax`/`atomicExchange`, which already return the
    /// previous value. `cas` doesn't: WGSL's
    /// `atomicCompareExchangeWeak(ptr, cmp, val)` returns a struct
    /// (`{old_value, exchanged}`), not a bare value — `.old_value` field
    /// access on the call result gives exactly the previous value, matching
    /// this method's return-value contract on every other backend.
    fn try_atomic_method_call(&mut self, obj: &Expr, method: &str, args_s: &[String]) -> Option<String> {
        let ExprKind::Index(arr, idx) = &obj.kind else { return None; };
        let arr_name = match &arr.kind {
            ExprKind::Var(n) => n.clone(),
            _ => return None,
        };
        if !matches!(method, "min" | "max" | "swap" | "cas") { return None; }
        if !self.is_atomic_field(&arr_name) {
            // Non-atomic case is handled entirely at the statement level
            // (`try_emit_plain_index_method_stmt`, called from `Stmt::Let`/
            // `Stmt::Expr`'s `Assign` case) -- WGSL has no statement-expression
            // (unlike CUDA/HIP/Metal's `({ ... })`), so "read old, mutate,
            // yield old" can't be a single expression here; it needs two
            // WGSL statements. Reaching this function means the call appeared
            // somewhere that pre-check doesn't cover (nested in a larger
            // expression) -- flag it visibly rather than emit silently wrong
            // WGSL (this used to silently fall through to the *unrelated*
            // pre-existing scalar `.min`/`.max` builtin-method mapping, or to
            // genuinely invalid WGSL for `.swap`/`.cas` -- confirmed via a
            // real naga parse: "no definition in scope for identifier: 'swap'").
            return Some(format!(
                "/* unsupported here: {}.{}(...) needs 'actor'global/'actor'unified, or must be the entire RHS of a let/assignment on this backend */",
                self.expr(obj), method
            ));
        }
        let i = self.expr(idx);
        let arr_name = self.current_buffer_renames.get(&arr_name).cloned().unwrap_or(arr_name);
        // `u32(...)`, not `... as u32` — see `try_atomic_assign`'s identical fix.
        let ptr = format!("&{}[u32({})]", arr_name, i);
        match (method, args_s) {
            ("min", [v]) => Some(format!("atomicMin({}, {})", ptr, v)),
            ("max", [v]) => Some(format!("atomicMax({}, {})", ptr, v)),
            ("swap", [v]) => Some(format!("atomicExchange({}, {})", ptr, v)),
            ("cas", [expected, new]) => Some(format!("atomicCompareExchangeWeak({}, {}, {}).old_value", ptr, expected, new)),
            _ => None,
        }
    }

    /// `a.axis` — read-only shape-query property on a fixed-shape
    /// `LabeledArray` field (dynamic-shape ones are already desugared to a
    /// shadow field before codegen — see `desugar_labeled_array`). `a[width =
    /// w, height = h]`-style indexing is a distinct `ExprKind::LabeledIndex`
    /// node, handled directly in `expr()`'s main match instead of here.
    fn try_labeled_array_field_access(&mut self, obj: &Expr, field_name: &str) -> Option<String> {
        let ExprKind::Var(name) = &obj.kind else { return None; };
        let field = self.current_fields.iter().find(|f| &f.name == name)?;
        let (_, axes) = field.ty.as_labeled_array()?;
        labeled_array_dim_literal(axes, field_name)
    }

    /// `let name = arr[idx].min/max/swap/cas(...)` (or `name = ...` via
    /// `Assign`, when `decl` is `None`) where `arr` is *not* atomic-qualified:
    /// WGSL has no statement-expression, so unlike the atomic case (a single
    /// `atomicMin`/etc. call already returns the previous value) or CUDA/
    /// HIP/Metal's non-atomic fallback (bridged via `({ ... })`), this needs
    /// two real WGSL statements — bind/assign `name` to the current value
    /// first, then perform the plain update. Returns `true` only for exactly
    /// this shape; the caller falls through to ordinary expression emission
    /// otherwise (including the real atomic case, unaffected by this at all).
    fn try_emit_plain_index_method_stmt(&mut self, decl: Option<(&str, &str)>, name: &str, val: &Expr) -> bool {
        let ExprKind::MethodCall(obj, method, args) = &val.kind else { return false; };
        let ExprKind::Index(arr, _idx) = &obj.kind else { return false; };
        let ExprKind::Var(arr_name) = &arr.kind else { return false; };
        if !matches!(method.as_str(), "min" | "max" | "swap" | "cas") { return false; }
        if self.is_atomic_field(arr_name) { return false; }
        let args_s: Vec<String> = args.iter().map(|a| self.expr(&a.value)).collect();
        let target = self.expr(obj);
        match decl {
            Some((kw, ty_suffix)) => self.line(&format!("{} {}{} = {};", kw, name, ty_suffix, target)),
            None => self.line(&format!("{} = {};", name, target)),
        }
        match (method.as_str(), args_s.as_slice()) {
            ("min", [v]) => self.line(&format!("{} = min({}, {});", target, name, v)),
            ("max", [v]) => self.line(&format!("{} = max({}, {});", target, name, v)),
            ("swap", [v]) => self.line(&format!("{} = {};", target, v)),
            ("cas", [expected, new]) => {
                self.line(&format!("if ({} == {}) {{", name, expected));
                self.indent += 1;
                self.line(&format!("{} = {};", target, new));
                self.indent -= 1;
                self.line("}");
            }
            _ => {}
        }
        true
    }

    // ── Expressions ───────────────────────────────────────────────────────────

    fn expr(&mut self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Int(n)   => {
                // Values that fit in i32: plain literal (default concrete type).
                // Values in (i32::MAX, u32::MAX]: suffix 'u' to force u32.
                if *n > i32::MAX as i64 && *n <= u32::MAX as i64 {
                    format!("{}u", n)
                } else if *n < i32::MIN as i64 || *n > u32::MAX as i64 {
                    self.errors.push(TranspileError::at(format!(
                        "integer literal {} out of u32 range on --target wgpu",
                        n
                    ), e.line, e.col));
                    n.to_string()
                } else {
                    n.to_string()
                }
            }
            ExprKind::Float(f) => {
                let s = format!("{}", f);
                if s.contains('.') || s.contains('e') { s } else { format!("{}.0", s) }
            }
            ExprKind::Bool(b)  => if *b { "true".into() } else { "false".into() },
            ExprKind::Str(s)   => format!("\"{}\"", s),
            ExprKind::Nil      => "0".into(),
            ExprKind::Void     => "".into(),
            ExprKind::Var(name) => {
                // Const-generic kernel type params (e.g. `W`/`H` on `kernel Blur<int W, int H>`)
                // are substituted down to a concrete literal here — `monomorphise`/`monomorphise_type`
                // only rewrites `type_params` references inside a kernel's `fields`, never inside
                // `methods`/`inits` bodies (see `resolve_effective_kernels`'s doc comment), so a body
                // expression like `0..<W * H` still contains a bare `Var("W")`/`Var("H")` after
                // monomorphisation — checked first since a type-param name is never also a real
                // buffer/field/top-level name in practice, and this is the single choke point every
                // expression emission passes through, so no occurrence (however nested) is missed.
                if let Some(v) = self.current_kernel_consts.get(name) {
                    return v.to_string();
                }
                // A GPU-array-qualified parameter of the function/method currently being
                // emitted was dropped from the signature entirely (see
                // `is_gpu_array_qualified_type`'s doc comment) — every occurrence of its
                // name is substituted here instead, exactly like a kernel buffer field
                // reference (which is genuinely what a resolved one is), before any of the
                // ordinary lookups below get a chance to treat it as an unresolved local.
                if let Some(sub) = self.current_gpu_array_subst.get(name) {
                    return sub.clone();
                }
                // Buffer-qualified fields (`'unified`/`'global`/`'actor'global`) become WGSL
                // module-level globals shared across every kernel in the same shader file —
                // a bare field name like `a` collides the moment two kernels both happen to
                // name a buffer field `a` (e.g. two matmul-shaped kernels). Prefix with the
                // owning kernel's name to keep each kernel's globals in its own namespace,
                // matching the params-uniform variable's existing `{kernel}_params` naming.
                if let Some(prefixed) = self.current_buffer_renames.get(name) {
                    prefixed.clone()
                } else if let Some(field) = self.current_fields.iter().find(|f| f.name == *name) {
                    // A kernel field of the same name shadows the top-level scalar
                    // (e.g. `kernel Saxpy: let float alpha` vs. top-level `let alpha
                    // = 2.0` in examples/saxpy.br) -- the field is unpacked into a
                    // real local (`let alpha: f32 = params.alpha;`) above, so it must
                    // win, not the outer literal. Previously unguarded: silently
                    // miscompiled `alpha * x[i] + y[i]` to always use the top-level
                    // literal instead of the runtime parameter -- no compile error,
                    // just a wrong-value bug (confirmed via `boring build --target
                    // wgpu examples/saxpy.br`). The unpacked local's own name is
                    // WGSL-sanitized (see `wgsl_safe_ident`) for fields whose Boring
                    // name is `__`-prefixed (e.g. a desugared labeled-array shadow
                    // axis) -- reference it under the same sanitized spelling.
                    //
                    // A `'sync` fixed-array workgroup field needs the stricter
                    // `wgsl_workgroup_array_ident` instead — it's declared as a
                    // module-scope `var<workgroup>` (see `emit_kernel_decl`'s
                    // "3. Workgroup" section), so a reference here must agree with
                    // that declaration's own kernel-prefixed renaming or the two
                    // would name-mismatch.
                    let is_sync_array = matches!(field.qual, GpuQual::Actor)
                        && (matches!(field.ty, Type::ArrayN(_, _)) || field.ty.as_labeled_array().is_some());
                    if is_sync_array {
                        wgsl_workgroup_array_ident(&self.current_kernel, name)
                    } else {
                        wgsl_safe_ident(name)
                    }
                } else {
                    // Not a kernel field or a top-level scalar: an ordinary `def()`-body
                    // local, function/method parameter, or `for`-loop variable — same
                    // reserved-keyword sanitization as the field-shadow case above, so a
                    // reference always agrees with how `Stmt::Let`/`plain_params_wgsl`/
                    // `Stmt::For` spelled its declaration.
                    self.top_level_scalars.get(name).cloned().unwrap_or_else(|| wgsl_safe_ident(name))
                }
            }

            ExprKind::BinOp(op, lhs, rhs) => {
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                format!("({} {} {})", l, binop_wgsl(op), r)
            }
            ExprKind::UnaryOp(op, operand) => {
                let v = self.expr(operand);
                format!("({}{})", unaryop_wgsl(op), v)
            }
            ExprKind::Assign(lhs, rhs) => {
                format!("({} = {})", self.expr(lhs), self.expr(rhs))
            }
            ExprKind::Index(arr, idx) => {
                // A fixed-size (`'const`) array field lives inside the params uniform
                // struct, not as a bare module-scope var (`emit_params_struct`'s
                // `Type::ArrayN`/labeled-array branches) — unlike a scalar params field,
                // which DOES get unpacked into a same-named local by `emit_entry_point`
                // ("Unpack 'const scalars ... from params struct"), a fixed array is
                // deliberately left un-unpacked ("Fixed arrays are accessed as
                // {pvar}.field[i] directly" — see that comment) since copying a whole
                // array into a local WGSL `var` on every kernel invocation is wasteful.
                // But nothing else ever emitted that `{pvar}.field[i]` rewrite the
                // comment promises: a bare index like `weights[k]` (declared as
                // `let [float, W*H]'const weights`) fell through the generic `Var` case
                // as an un-prefixed `weights[...]`, an undefined WGSL identifier at
                // shader-validation time. Rewrite it here instead.
                if let ExprKind::Var(name) = &arr.kind {
                    if let Some(field) = self.current_fields.iter().find(|f| f.name == *name) {
                        let is_fixed_array = matches!(field.ty, Type::ArrayN(_, _)) || field.ty.as_labeled_array().is_some();
                        if is_params_field(field) && is_fixed_array {
                            let pvar = format!("{}_params", self.current_kernel.to_lowercase());
                            let idx_s = self.expr(idx);
                            return format!("{}.{}[u32({})]", pvar, wgsl_safe_ident(name), idx_s);
                        }
                    }
                }
                format!("{}[u32({})]", self.expr(arr), self.expr(idx))
            }
            ExprKind::LabeledIndex(obj, args) => {
                // Stringify every arg's value first (needs `&mut self`) —
                // only then borrow `self.current_fields` immutably, so the
                // two borrows never overlap. Cloning `axes` to owned releases
                // that borrow before the later `self.expr(obj)` call, which
                // also needs `&mut self` — `obj` must go through
                // `self.expr(...)`, not the raw AST `Var` name, since this
                // backend renames storage-buffer variables (e.g. field `img`
                // becomes WGSL global `img_img`).
                let pairs: Vec<(String, String)> = args.iter()
                    .filter_map(|a| a.label.clone().map(|l| (l, self.expr(&a.value))))
                    .collect();
                let axes: Option<Vec<crate::ast::LabeledAxis>> = if let ExprKind::Var(name) = &obj.kind {
                    self.current_fields.iter().find(|f| &f.name == name)
                        .and_then(|field| field.ty.as_labeled_array())
                        .map(|(_, axes)| axes.to_vec())
                } else {
                    None
                };
                match axes.and_then(|axes| labeled_array_at_index(&axes, &pairs)) {
                    Some(offset) => {
                        let target = self.expr(obj);
                        format!("{}[u32({})]", target, offset)
                    }
                    None => "/* unsupported labeled index */".to_string(),
                }
            }
            ExprKind::Field(obj, field) => {
                if let Some(wgsl) = self.try_labeled_array_field_access(obj, field) {
                    wgsl
                } else {
                    let obj_s = self.expr(obj);
                    map_gpu_field(&obj_s, field)
                }
            }
            ExprKind::Call(callee, args) => {
                // A GPU-array-qualified argument position is dropped entirely — the
                // callee's own signature no longer has a parameter there either (see
                // `is_gpu_array_qualified_type`'s doc comment), the callee body
                // substitutes the resolved global directly instead.
                let gpu_slots: Vec<ParamSlot> = if let ExprKind::Var(n) = &callee.kind {
                    self.gpu_array_params.get(n).cloned().unwrap_or_default()
                } else {
                    Vec::new()
                };
                let args_s: Vec<String> = args.iter().enumerate().filter_map(|(i, a)| {
                    if matches!(gpu_slots.get(i), Some(ParamSlot::Gpu(_))) {
                        None
                    } else {
                        Some(self.expr(&a.value))
                    }
                }).collect();
                let fn_s = match &callee.kind {
                    ExprKind::Var(n) => map_builtin_fn(n),
                    _ => self.expr(callee),
                };
                format!("{}({})", fn_s, args_s.join(", "))
            }
            ExprKind::MethodCall(obj, method, args) => {
                let args_s: Vec<String> = args.iter().map(|a| self.expr(&a.value)).collect();
                if is_gpu_warp_receiver(obj) {
                    if let Some(wgsl) = gpu_warp_method_call(method, &args_s, self.mode) {
                        return wgsl;
                    }
                    if is_gpu_warp_shuffle(method) {
                        // `Emulated` mode only supports `gpu.warp.shuffle_*` directly as a
                        // `let x = ...` statement's RHS (see `emit_stmt`'s `Stmt::Let` case),
                        // since the emulation needs statements (write, barrier, read, barrier)
                        // around the point of use, not a single expression. Reaching here means
                        // a shuffle call appeared somewhere else (nested in a larger expression,
                        // a bare statement, ...) — flag it visibly rather than emit silently
                        // wrong WGSL.
                        return format!(
                            "/* unsupported on the emulated-subgroup fallback: gpu.warp.{}(...) must be the entire RHS of a `let` */",
                            method
                        );
                    }
                }
                if let Some(wgsl) = self.try_atomic_method_call(obj, method, &args_s) {
                    return wgsl;
                }
                // `.len()` on a kernel array field — WGSL has no `len(x)` free function
                // (unlike CUDA/Metal's plain array-length arithmetic); a runtime-sized
                // storage-buffer field (`'global`/`'unified`/... ) needs the WGSL builtin
                // `arrayLength(&buf)`, while a compile-time-fixed-size field (`'const`
                // array, `[T, N]`) already has its length known at transpile time, so it's
                // emitted as a plain integer literal instead of a call. Checked before the
                // generic numeric-method fallback below, which would otherwise emit the
                // invalid `len(...)` call (`map_builtin_fn`'s catch-all passes unknown
                // method names through unchanged).
                if method == "len" && args.is_empty() {
                    if let ExprKind::Var(name) = &obj.kind {
                        if let Some(field) = self.current_fields.iter().find(|f| f.name == *name) {
                            if is_buffer_field(field) {
                                let obj_s = self.expr(obj);
                                // `arrayLength` returns `u32`, but every kernel-body index/
                                // loop-variable expression is emitted as `i32` (see the
                                // `gpu.thread.x`/`gpu.block.x` builtins and the `for`-loop
                                // variable further down in this file) -- WGSL's comparison
                                // and arithmetic operators require identical operand types,
                                // with no implicit signed/unsigned promotion like C/Rust.
                                // Comparing (or otherwise combining) the bare `u32` result
                                // against an `i32` index (`idx < input.len()`) is invalid
                                // WGSL and fails real shader validation at pipeline-creation
                                // time ("Operation Less can't work with ..."), even though it
                                // passes a plain `cargo build` (which never parses the
                                // embedded WGSL string) and this file's text-only codegen
                                // tests. Cast to `i32` here, once, at the single choke point
                                // that ever emits `arrayLength`, rather than at every call
                                // site that happens to compare/combine a `.len()` against an
                                // index.
                                return format!("i32(arrayLength(&{}))", obj_s);
                            }
                            if let Type::ArrayN(_, n) = &field.ty {
                                return n.to_string();
                            }
                            if let Some(len) = field.ty.labeled_array_len() {
                                return len.to_string();
                            }
                        }
                    }
                }
                if matches!(&obj.kind, ExprKind::Var(n) if n == "self") {
                    let fn_name = format!("{}_{}", self.current_kernel, method);
                    // See the free-function `ExprKind::Call` arm's identical comment — a
                    // kernel struct's own helper method can have GPU-array-qualified
                    // *extra* (non-field) parameters too, dropped the same way.
                    let gpu_slots: Vec<ParamSlot> = self.gpu_array_params.get(&fn_name).cloned().unwrap_or_default();
                    let call_args: Vec<String> = args.iter().enumerate().filter_map(|(i, a)| {
                        if matches!(gpu_slots.get(i), Some(ParamSlot::Gpu(_))) {
                            None
                        } else {
                            Some(self.expr(&a.value))
                        }
                    }).collect();
                    format!("{}({})", fn_name, call_args.join(", "))
                } else {
                    // Numeric builtin method call (`.sqrt()`, `.exp()`, `.tanh()`, `.pow(y)`,
                    // etc.) on a scalar expression — WGSL has no methods, only free
                    // functions, so `v.exp()` becomes `exp(v)`. Kernels are numeric-only
                    // (WGSL has no strings/dynamic collections), so any non-`self` method
                    // call reaching device code is expected to be one of these.
                    let obj_s = self.expr(obj);
                    let fn_s = map_builtin_fn(method);
                    let mut call_args = vec![obj_s];
                    call_args.extend(args_s);
                    format!("{}({})", fn_s, call_args.join(", "))
                }
            }
            ExprKind::Cast(inner, ty) => {
                format!("{}({})", wgsl_type(ty), self.expr(inner))
            }
            ExprKind::If(i) => {
                if let Some((cond, then_body)) = i.branches.first() {
                    let c = self.expr(cond);
                    let t = then_body.last().and_then(|s| {
                        if let Stmt::Expr(e) = s { Some(self.expr(e)) } else { None }
                    }).unwrap_or_else(|| "0".into());
                    let e = i.else_body.as_ref().and_then(|b| b.last()).and_then(|s| {
                        if let Stmt::Expr(e) = s { Some(self.expr(e)) } else { None }
                    }).unwrap_or_else(|| "0".into());
                    format!("select({}, {}, {})", e, t, c)
                } else {
                    "0".into()
                }
            }
            _ => "/* expr */".into(),
        }
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

/// Collect every identifier referenced anywhere in `body` (variable reads and call
/// callees alike — `collect_vars_in_stmt` doesn't distinguish them) into `out`. Used to
/// find which free functions a kernel's device code actually calls, so WGSL emission for
/// merged-in host-only helpers (unreachable from any kernel) can be skipped. Harmless
/// over-approximation: plain variable names that happen to collide with a free function's
/// name would also be swept in, but that only makes the reachable set too large, never
/// too small.
fn collect_called_fn_names(body: &[Stmt], out: &mut Vec<String>) {
    for stmt in body {
        collect_vars_in_stmt(stmt, out);
    }
}

/// Returns true if the field goes into a storage buffer binding.
fn is_buffer_field(f: &KernelFieldDecl) -> bool {
    match f.qual {
        GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface => {
            matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) || f.ty.as_labeled_array().is_some()
        }
        _ => false,
    }
}

/// Returns true if the field goes into the params uniform struct.
fn is_params_field(f: &KernelFieldDecl) -> bool {
    match f.qual {
        GpuQual::Const => true,
        GpuQual::Local => !matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)),
        _ => {
            // Named struct types (e.g. Dimension) in non-buffer fields go into params.
            matches!(&f.ty, Type::Named(_))
                && !matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface)
        }
    }
}

/// Returns the WGSL storage access modifier and array type for a buffer field.
fn wgsl_buffer_type(f: &KernelFieldDecl) -> (&'static str, String) {
    let access = match f.qual {
        GpuQual::ActorGlobal | GpuQual::ActorUnified => "read_write", // atomics need read_write
        _ => if matches!(f.binding, FieldBinding::Let) { "read" } else { "read_write" },
    };
    let inner_ty: Option<&Type> = match &f.ty {
        Type::Array(inner) | Type::ArrayN(inner, _) => Some(inner.as_ref()),
        ty if ty.as_labeled_array().is_some() => Some(ty.as_labeled_array().unwrap().0),
        _ => None,
    };
    let elem = match inner_ty {
        Some(inner) => {
            if matches!(f.qual, GpuQual::ActorGlobal | GpuQual::ActorUnified) {
                // atomic<i32> for int, atomic<u32> for uint
                match inner {
                    Type::Int => "atomic<i32>".into(),
                    Type::Named(n) if matches!(n.as_str(), "int" | "i32" | "i64") => "atomic<i32>".into(),
                    Type::Uint => "atomic<u32>".into(),
                    Type::Named(n) if matches!(n.as_str(), "uint" | "u32" | "u64") => "atomic<u32>".into(),
                    other => wgsl_scalar(other),
                }
            } else {
                wgsl_scalar(inner)
            }
        }
        None => wgsl_scalar(&f.ty),
    };
    let ty = format!("array<{}>", elem);
    (access, ty)
}

/// WGSL scalar type for kernel use (i32, u32, f32, bool→u32).
/// WGSL only has 32-bit `i32`/`u32`/`f32` scalars natively — no 8/16/64/128-bit integer
/// types. A kernel field declared with one of those widths can't be represented on this
/// target; rather than silently mis-narrowing it (the previous behavior for `Uint8`,
/// which fell through to `i32`), emit a WGSL comment naming the problem inline so the
/// generated shader fails to compile with a clear reason instead of running with wrong
/// data layout. Mirrors the same "emit a comment as a fallback" convention already used
/// for `float` on the `kernel` (no_std) target.
fn wgsl_unsupported_width(name: &str, fallback: &str) -> String {
    format!("/* ERROR: `{}` is not supported on --target wgpu (WGSL has no 8/16/64/128-bit integers) */ {}", name, fallback)
}

/// WGSL has no 64-bit float type at all — unlike the integer widths above, there is no
/// narrowing fallback to reach for (Metal at least has a 32-bit `float` to narrow into;
/// WGSL's widest float is `f32`, so a `float64`/`float` kernel field genuinely has no
/// representation here). docs/float-width-types.md §6.
fn wgsl_unsupported_f64(fallback: &str) -> String {
    format!("/* ERROR: `float64` is not supported on --target wgpu (WGSL has no 64-bit float type — use float32) */ {}", fallback)
}

/// `int`/`uint` are Boring's default-width integer types — 64-bit (`isize`/`usize`) on
/// every native GPU backend (metal/cuda/rocm, which have real 64-bit integer types).
/// WGSL has no 64-bit integer type at all, so unlike those backends this target can only
/// represent `int`/`uint` as 32-bit `i32`/`u32` — any value outside [-2^31, 2^31) (`int`)
/// or [0, 2^32) (`uint`) wraps silently on this target only. Emit a comment naming this
/// narrowing explicitly (mirrors `wgsl_unsupported_width`'s convention for the genuinely
/// unrepresentable widths) instead of narrowing in total silence.
fn wgsl_narrowed_width(name: &str, fallback: &str) -> String {
    format!("/* WARNING: `{}` is narrowed to 32-bit on --target wgpu (WGSL has no 64-bit integers) — values outside {} range wrap silently */ {}", name, fallback, fallback)
}

fn wgsl_scalar(ty: &Type) -> String {
    match ty {
        Type::Int              => wgsl_narrowed_width("int", "i32"),
        Type::Uint             => wgsl_narrowed_width("uint", "u32"),
        Type::Int32             => "i32".into(),
        Type::Uint32            => "u32".into(),
        Type::Uint8             => wgsl_unsupported_width("uint8", "u32"),
        Type::Int8               => wgsl_unsupported_width("int8", "i32"),
        Type::Int16              => wgsl_unsupported_width("int16", "i32"),
        Type::Uint16             => wgsl_unsupported_width("uint16", "u32"),
        Type::Int64               => wgsl_unsupported_width("int64", "i32"),
        Type::Uint64              => wgsl_unsupported_width("uint64", "u32"),
        Type::Int128              => wgsl_unsupported_width("int128", "i32"),
        Type::Uint128              => wgsl_unsupported_width("uint128", "u32"),
        Type::Float32            => "f32".into(),
        Type::Float64            => wgsl_unsupported_f64("f32"),
        // bool is not allowed in storage/uniform buffers in WGSL — use u32.
        Type::Bool             => "u32".into(),
        Type::Named(n) => match n.as_str() {
            "int"   => wgsl_narrowed_width("int", "i32"),
            "uint"  => wgsl_narrowed_width("uint", "u32"),
            "i32" => "i32".to_string(),
            "u32" => "u32".to_string(),
            "float32" | "f32" => "f32".to_string(),
            "float" | "float64" | "f64" => wgsl_unsupported_f64("f32"),
            "bool"                  => "u32".to_string(),
            "uint8"                 => wgsl_unsupported_width("uint8", "u32"),
            "int8"                  => wgsl_unsupported_width("int8", "i32"),
            "int16"                 => wgsl_unsupported_width("int16", "i32"),
            "uint16"                => wgsl_unsupported_width("uint16", "u32"),
            "int32"                 => "i32".to_string(),
            "uint32"                => "u32".to_string(),
            "int64" | "i64" | "int128" | "i128" => wgsl_unsupported_width("int64/int128", "i32"),
            "uint64" | "u64" | "uint128" | "u128" => wgsl_unsupported_width("uint64/uint128", "u32"),
            other                   => other.to_string(),
        },
        Type::Qualified(inner, _) => wgsl_scalar(inner),
        _ => "i32".into(),
    }
}

/// Full WGSL type including arrays and structs.
fn wgsl_type(ty: &Type) -> String {
    match ty {
        Type::Array(inner)        => format!("array<{}>", wgsl_scalar(inner)),
        Type::ArrayN(inner, n)    => format!("array<{}, {}>", wgsl_scalar(inner), n),
        Type::Named(n) if n == "Dimension" => "Dimension".into(),
        ty if ty.as_labeled_array().is_some() && ty.labeled_array_len().is_some() => {
            let (elem, _) = ty.as_labeled_array().unwrap();
            let len = ty.labeled_array_len().unwrap();
            format!("array<{}, {}>", wgsl_scalar(elem), len)
        }
        other                     => wgsl_scalar(other),
    }
}

/// Zero-value literal for a WGSL scalar type.
fn wgsl_zero(ty: &Type) -> &'static str {
    match ty {
        Type::Float32 | Type::Float64 | Type::Named(_) => "0.0",
        Type::Bool  => "0u",
        _           => "0",
    }
}

fn binop_wgsl(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",    BinOp::Sub => "-",  BinOp::Mul => "*",
        BinOp::Div => "/",    BinOp::Rem => "%",
        BinOp::Eq  => "==",   BinOp::NotEq => "!=",
        BinOp::Lt  => "<",    BinOp::Gt => ">",
        BinOp::LtEq => "<=",  BinOp::GtEq => ">=",
        BinOp::And => "&&",   BinOp::Or => "||",
        BinOp::BitAnd => "&",  BinOp::BitOr => "|", BinOp::BitXor => "^",
        BinOp::Shl => "<<",   BinOp::Shr => ">>",
        _ => "/*op*/",
    }
}

fn unaryop_wgsl(op: &UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg    => "-",
        UnaryOp::Not    => "!",
        UnaryOp::BitNot => "~",
    }
}

fn map_gpu_field(obj: &str, field: &str) -> String {
    match (obj, field) {
        ("gpu", "thread")    => "bp_tid".into(),
        ("gpu", "block")     => "bp_bid".into(),
        ("gpu", "block_dim") => "bp_bdim".into(),
        ("gpu", "grid_dim")  => "bp_gdim".into(),
        // Dimension accessors — cast to i32 for arithmetic (matches Boring int → i32).
        ("bp_tid",  "x") => "i32(bp_tid.x)".into(),
        ("bp_tid",  "y") => "i32(bp_tid.y)".into(),
        ("bp_tid",  "z") => "i32(bp_tid.z)".into(),
        ("bp_bid",  "x") => "i32(bp_bid.x)".into(),
        ("bp_bid",  "y") => "i32(bp_bid.y)".into(),
        ("bp_bid",  "z") => "i32(bp_bid.z)".into(),
        ("bp_bdim", "x") => "i32(bp_bdim.x)".into(),
        ("bp_bdim", "y") => "i32(bp_bdim.y)".into(),
        ("bp_bdim", "z") => "i32(bp_bdim.z)".into(),
        ("bp_gdim", "x") => "i32(bp_gdim.x)".into(),
        ("bp_gdim", "y") => "i32(bp_gdim.y)".into(),
        ("bp_gdim", "z") => "i32(bp_gdim.z)".into(),
        ("gpu", "warp")  => "__warp".into(),
        // Same identifiers on both `WarpMode::Real` (real subgroup builtin
        // params) and `WarpMode::Emulated` (synthesized `let`s) — see
        // `emit_entry_point`.
        ("__warp", "size") => "i32(bp_wsize)".into(),
        ("__warp", "lane") => "i32(bp_lane)".into(),
        _ => format!("{}.{}", obj, field),
    }
}

/// `gpu.warp.sync()` (both modes) and the real-subgroup `gpu.warp.shuffle_*`
/// mapping — valid as a plain expression anywhere, unlike the `Emulated`
/// shuffle mapping (needs statements around the point of use; handled
/// separately in `emit_stmt`'s `Stmt::Let` case).
fn gpu_warp_method_call(method: &str, args: &[String], mode: WarpMode) -> Option<String> {
    match (method, mode) {
        ("sync", WarpMode::Real)     => Some("subgroupBarrier()".into()),
        ("sync", WarpMode::Emulated) => Some("workgroupBarrier()".into()),
        ("shuffle_down", WarpMode::Real) => Some(format!("subgroupShuffleDown({}, {})", args[0], args[1])),
        ("shuffle_up", WarpMode::Real)   => Some(format!("subgroupShuffleUp({}, {})", args[0], args[1])),
        ("shuffle_xor", WarpMode::Real)  => Some(format!("subgroupShuffleXor({}, {})", args[0], args[1])),
        ("shuffle", WarpMode::Real)      => Some(format!("subgroupShuffle({}, {})", args[0], args[1])),
        _ => None,
    }
}

fn is_gpu_warp_receiver(obj: &Expr) -> bool {
    matches!(&obj.kind, ExprKind::Field(inner, name) if name == "warp"
        && matches!(&inner.kind, ExprKind::Var(v) if v == "gpu"))
}

fn is_gpu_warp_shuffle(method: &str) -> bool {
    matches!(method, "shuffle_down" | "shuffle_up" | "shuffle_xor" | "shuffle")
}

/// Kernel-prefixed (lowercased kernel name, matching `wgsl_workgroup_array_ident`'s
/// and `current_buffer_renames`'s own scheme) — this is also a module-scope
/// `var<workgroup>` declaration, so two unrelated kernels that both shuffle
/// the same element type (e.g. two kernels both `gpu.warp.shuffle_down`-ing an
/// `f32` accumulator) previously collided at module scope exactly like the
/// `'sync` fixed-array field bug above. See CHANGELOG.md.
fn warp_scratch_var_name(kernel: &str, elem_ty: &str) -> String {
    format!("bp_warp_scratch_{}_{}", kernel.to_lowercase(), elem_ty)
}

/// Best-effort WGSL scalar element type for a `gpu.warp.shuffle_*` value
/// argument — resolves field/local references via `fields`' declared types
/// (unwrapping one level of array), literals directly, and falls back to
/// `f32` for anything else (kernels are numeric-only, and `f32` matches the
/// most common shuffled-value shape — an accumulator in a tiled reduction).
fn infer_shuffle_elem_type(expr: &Expr, fields: &[KernelFieldDecl]) -> String {
    fn field_elem_ty(fields: &[KernelFieldDecl], name: &str) -> Option<String> {
        fields.iter().find(|f| f.name == name).map(|f| match &f.ty {
            Type::Array(inner) | Type::ArrayN(inner, _) => wgsl_scalar(inner),
            other => wgsl_scalar(other),
        })
    }
    match &expr.kind {
        ExprKind::Var(name) => field_elem_ty(fields, name).unwrap_or_else(|| "f32".into()),
        ExprKind::Field(_, name) => field_elem_ty(fields, name).unwrap_or_else(|| "f32".into()),
        ExprKind::Index(arr, _) => infer_shuffle_elem_type(arr, fields),
        ExprKind::Cast(_, ty) => wgsl_scalar(ty),
        ExprKind::Int(_) => "i32".into(),
        ExprKind::Float(_) => "f32".into(),
        _ => "f32".into(),
    }
}

/// Collects the distinct WGSL scalar types `gpu.warp.shuffle_*` shuffles in
/// `stmts`, restricted to the `let x = gpu.warp.shuffle_*(...)` statement
/// shape `Emulated` mode actually supports (see `emit_stmt`'s `Stmt::Let`
/// case) — recurses into the control-flow bodies kernel reduction loops
/// realistically use (`if`/`while`/`for`/`loop`).
/// Collects the distinct WGSL scalar types shuffled anywhere in `stmts` —
/// mirrors exactly the expression shapes `DeviceEmitter::hoist_shuffles` knows
/// how to hoist (`BinOp`/`UnaryOp`/`Index`/`Cast`/`Call` args/`MethodCall` args),
/// so the module-scope scratch buffers declared from this always match what
/// hoisting will actually reference. Recurses into the control-flow bodies
/// kernel reduction loops realistically use (`if`/`while`/`for`/`loop`).
fn collect_shuffle_elem_types_stmts(
    stmts: &[Stmt],
    fields: &[KernelFieldDecl],
    out: &mut std::collections::BTreeSet<String>,
) {
    for stmt in stmts {
        match stmt {
            Stmt::Let(s) => {
                if let Some(val) = &s.value { collect_shuffle_types_expr(val, fields, out); }
            }
            Stmt::Expr(e) => {
                if let ExprKind::Assign(_, rhs) = &e.kind {
                    collect_shuffle_types_expr(rhs, fields, out);
                }
            }
            Stmt::If(i) => {
                for (_, b) in &i.branches { collect_shuffle_elem_types_stmts(b, fields, out); }
                if let Some(eb) = &i.else_body { collect_shuffle_elem_types_stmts(eb, fields, out); }
            }
            Stmt::While(w) => collect_shuffle_elem_types_stmts(&w.body, fields, out),
            Stmt::For(f) => collect_shuffle_elem_types_stmts(&f.body, fields, out),
            Stmt::Loop(l) => collect_shuffle_elem_types_stmts(&l.body, fields, out),
            _ => {}
        }
    }
}

/// Expression-level counterpart of `collect_shuffle_elem_types_stmts` — walks
/// exactly the subset of `ExprKind` `hoist_shuffles` rewrites.
fn collect_shuffle_types_expr(e: &Expr, fields: &[KernelFieldDecl], out: &mut std::collections::BTreeSet<String>) {
    if let ExprKind::MethodCall(obj, method, args) = &e.kind {
        if is_gpu_warp_receiver(obj) && is_gpu_warp_shuffle(method) && !args.is_empty() {
            out.insert(infer_shuffle_elem_type(&args[0].value, fields));
            for a in args { collect_shuffle_types_expr(&a.value, fields, out); }
            return;
        }
    }
    match &e.kind {
        ExprKind::BinOp(_, l, r) => {
            collect_shuffle_types_expr(l, fields, out);
            collect_shuffle_types_expr(r, fields, out);
        }
        ExprKind::UnaryOp(_, x) | ExprKind::Cast(x, _) => collect_shuffle_types_expr(x, fields, out),
        ExprKind::Index(a, i) => {
            collect_shuffle_types_expr(a, fields, out);
            collect_shuffle_types_expr(i, fields, out);
        }
        ExprKind::Call(callee, args) => {
            collect_shuffle_types_expr(callee, fields, out);
            for a in args { collect_shuffle_types_expr(&a.value, fields, out); }
        }
        ExprKind::MethodCall(obj, _, args) => {
            collect_shuffle_types_expr(obj, fields, out);
            for a in args { collect_shuffle_types_expr(&a.value, fields, out); }
        }
        _ => {}
    }
}

fn map_builtin_fn(name: &str) -> String {
    match name {
        "int"     => "i32".into(),
        "uint"    => "u32".into(),
        "float"   => "f32".into(),
        // `float32(...)` casts: same f32 target as the bare `float` cast above —
        // WGSL has only one float width. `float64(...)` has no representation at
        // all here (see `wgsl_unsupported_f64`'s doc comment) — emit the same
        // clean compile-error comment the type-level check already produces,
        // instead of passing the call through verbatim as invalid WGSL.
        "float32" => "f32".into(),
        "float64" => wgsl_unsupported_f64("f32"),
        "abs"   => "abs".into(),
        "min"   => "min".into(),
        "max"   => "max".into(),
        "sqrt"  => "sqrt".into(),
        "sin"   => "sin".into(),
        "cos"   => "cos".into(),
        "tan"   => "tan".into(),
        "tanh"  => "tanh".into(),
        "exp"   => "exp".into(),
        "log"   => "log".into(),
        "log2"  => "log2".into(),
        "pow"   => "pow".into(),
        "floor" => "floor".into(),
        "ceil"  => "ceil".into(),
        "round" => "round".into(),
        other   => other.into(),
    }
}

/// Extract the inner Named type from an array/qualified type, if any.
fn inner_named_type(ty: &Type) -> Option<&str> {
    match ty {
        Type::Array(inner) | Type::ArrayN(inner, _) => inner_named_type(inner),
        Type::Qualified(inner, _) => inner_named_type(inner),
        Type::Named(n) => Some(n.as_str()),
        _ => None,
    }
}

// ── Auto-sync helpers (identical logic to Metal backend) ──────────────────────

fn body_has_explicit_sync(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|s| match s {
        Stmt::Comment(c) if c == "sync" => true,
        Stmt::While(w)   => body_has_explicit_sync(&w.body),
        Stmt::For(f)     => body_has_explicit_sync(&f.body),
        Stmt::If(i)      => i.branches.iter().any(|(_, b)| body_has_explicit_sync(b))
                         || i.else_body.as_ref().is_some_and(|b| body_has_explicit_sync(b)),
        _ => false,
    })
}

fn body_accesses_sync_field(stmts: &[Stmt], fields: &[KernelFieldDecl]) -> bool {
    let sync_names: Vec<&str> = fields.iter()
        .filter(|f| matches!(f.qual, GpuQual::Actor))
        .map(|f| f.name.as_str())
        .collect();
    if sync_names.is_empty() { return false; }
    stmts_reference_any(stmts, &sync_names)
}

fn stmts_reference_any(stmts: &[Stmt], names: &[&str]) -> bool {
    stmts.iter().any(|s| stmt_references_any(s, names))
}

fn stmt_references_any(stmt: &Stmt, names: &[&str]) -> bool {
    match stmt {
        Stmt::Expr(e)      => expr_references_any(e, names),
        Stmt::Return(r)    => r.value.as_ref().is_some_and(|v| expr_references_any(v, names)),
        Stmt::Let(s)       => s.value.as_ref().is_some_and(|v| expr_references_any(v, names)),
        Stmt::While(w)     => stmts_reference_any(&w.body, names),
        Stmt::For(f)       => stmts_reference_any(&f.body, names),
        Stmt::If(i)        => i.branches.iter().any(|(_, b)| stmts_reference_any(b, names))
                           || i.else_body.as_ref().is_some_and(|b| stmts_reference_any(b, names)),
        _ => false,
    }
}

fn expr_references_any(expr: &Expr, names: &[&str]) -> bool {
    match &expr.kind {
        ExprKind::Var(n)           => names.contains(&n.as_str()),
        ExprKind::Index(a, i)      => expr_references_any(a, names) || expr_references_any(i, names),
        ExprKind::Field(e, _)      => expr_references_any(e, names),
        ExprKind::BinOp(_, l, r)   => expr_references_any(l, names) || expr_references_any(r, names),
        ExprKind::UnaryOp(_, e)    => expr_references_any(e, names),
        ExprKind::Assign(l, r)     => expr_references_any(l, names) || expr_references_any(r, names),
        ExprKind::Call(f, args)    => expr_references_any(f, names)
                                   || args.iter().any(|a| expr_references_any(&a.value, names)),
        _ => false,
    }
}

/// Scan `kernel:` blocks for `kname(block = ...)` calls and return per-kernel block sizes.
/// `kernel:` blocks live anywhere a statement can — most commonly inside a free
/// function's body (e.g. `math_gpu.br`'s `linear_gpu`/`attention_gpu` helpers wrap
/// every kernel construction+dispatch in its own function), not just as bare
/// top-level statements — so every function body (and every nested statement
/// container within it) must be walked, not only `program.items` directly.
fn collect_block_sizes(program: &Program) -> std::collections::HashMap<String, (u32, u32, u32)> {
    let mut var_to_type: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut map = std::collections::HashMap::new();
    // A `k(block = N)` dispatch site commonly references a top-level scalar
    // `let` (e.g. `let N = 16`) rather than an int literal — WGSL's
    // `@workgroup_size` must still be a compile-time constant, so that
    // reference needs resolving here before the shader is emitted. Collected
    // up front (top-level only — a dispatch-site block size referencing a
    // local/function-scoped variable isn't statically resolvable this way and
    // falls back to the same conservative default as any other unresolvable
    // expression).
    let const_ints = collect_top_level_int_consts(program);
    for item in &program.items {
        match item {
            Item::Let(s) => resolve_let_kernel_type(s, &mut var_to_type),
            Item::Stmt(s) => scan_call_block_size(s, &mut map, &mut var_to_type, &const_ints),
            Item::Fn(f) => { for s in &f.body { scan_call_block_size(s, &mut map, &mut var_to_type, &const_ints); } }
            Item::Struct(s) => { for m in &s.methods { for st in &m.body { scan_call_block_size(st, &mut map, &mut var_to_type, &const_ints); } } }
            _ => {}
        }
    }
    map
}

/// Top-level `let NAME = <int literal>` bindings, by name — the only shape a
/// `block = NAME` dispatch-site argument can be resolved against at
/// transpile time (see `collect_block_sizes`'s doc comment above).
fn collect_top_level_int_consts(program: &Program) -> std::collections::HashMap<String, i64> {
    let mut consts = std::collections::HashMap::new();
    for item in &program.items {
        if let Item::Let(s) = item {
            if let Some(val) = &s.value {
                if let ExprKind::Int(n) = &val.kind {
                    consts.insert(s.name.clone(), *n);
                }
            }
        }
    }
    consts
}

fn resolve_let_kernel_type(s: &LetStmt, map: &mut std::collections::HashMap<String, String>) {
    if let Some(val) = &s.value {
        match &val.kind {
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(type_name) = &callee.kind {
                    map.insert(s.name.clone(), type_name.clone());
                }
            }
            // A turbofish const-generic construction (`Blur<3, 1>(...)`) parses as
            // `GenericCall`, a distinct AST variant from a plain `Call` — must be
            // handled separately or `var_to_type` never gets an entry for the
            // binding, and a later `blur(block = N)` dispatch-site lookup can never
            // resolve `blur` back to a kernel type name (see `collect_block_sizes`'s
            // doc comment). The recorded name must match the *monomorphised* decl
            // name (e.g. `"Blur_3_1"`) that `resolve_effective_kernels` in
            // `wgpu/mod.rs` produces and that `emit_kernel_decl`/`emit_entry_point`
            // key `block_sizes` by — so mangle it here the same way
            // `monomorphised_name` does: original name + concrete type args joined
            // with `_`, non-integer type args treated as `0` (matching
            // `collect_instantiations`/`build_subst` in `wgpu/mod.rs`).
            ExprKind::GenericCall(callee, type_args, _) => {
                if let ExprKind::Var(type_name) = &callee.kind {
                    let suffix: Vec<String> = type_args.iter().map(|t| match t {
                        Type::ConstInt(n) => n.to_string(),
                        _ => "0".to_string(),
                    }).collect();
                    let mangled = if suffix.is_empty() {
                        type_name.clone()
                    } else {
                        format!("{}_{}", type_name, suffix.join("_"))
                    };
                    map.insert(s.name.clone(), mangled);
                }
            }
            _ => {}
        }
    }
}

fn scan_call_block_size(
    s: &Stmt,
    map: &mut std::collections::HashMap<String, (u32, u32, u32)>,
    var_to_type: &mut std::collections::HashMap<String, String>,
    const_ints: &std::collections::HashMap<String, i64>,
) {
    match s {
        Stmt::Let(ls) => resolve_let_kernel_type(ls, var_to_type),
        Stmt::Loop(ls) => { for inner in &ls.body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        Stmt::While(ws) => { for inner in &ws.body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        Stmt::WhileLet(ws) => { for inner in &ws.body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        Stmt::DoWhile(ds) => { for inner in &ds.body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        Stmt::For(fs) => { for inner in &fs.body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        Stmt::Guard(gs) => { for inner in &gs.else_body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        Stmt::Try(ts) => {
            for inner in &ts.body { scan_call_block_size(inner, map, var_to_type, const_ints); }
            for c in &ts.catch_clauses { for inner in &c.body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        }
        Stmt::Defer(body) => { for inner in body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        Stmt::If(is) => {
            for (_, body) in &is.branches { for inner in body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
            if let Some(body) = &is.else_body { for inner in body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        }
        Stmt::IfLet(is) => {
            for inner in &is.then_body { scan_call_block_size(inner, map, var_to_type, const_ints); }
            for branch in &is.elif_branches { for inner in &branch.body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
            if let Some(body) = &is.else_body { for inner in body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        }
        Stmt::Match(ms) => {
            for arm in &ms.arms {
                match &arm.body {
                    MatchBody::Block(body) => { for inner in body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
                    MatchBody::Expr(_) => {}
                }
            }
        }
        Stmt::KernelBlock(block) => { for inner in &block.body { scan_call_block_size(inner, map, var_to_type, const_ints); } }
        Stmt::Expr(e) => {
            if let ExprKind::Call(callee, args) = &e.kind {
                if let ExprKind::Var(kname) = &callee.kind {
                    if let Some(ba) = args.iter().find(|a| a.label.as_deref() == Some("block")) {
                        let parse_u32 = |e: &Expr| -> u32 {
                            match &e.kind {
                                ExprKind::Int(n) => *n as u32,
                                // A top-level scalar `let` referenced by name (e.g.
                                // `let N = 16` then `k(block = N)`) — WGSL's
                                // `@workgroup_size` has to be a compile-time
                                // constant, so this is the one variable shape
                                // that's actually resolvable here. Anything else
                                // (a local/function-scoped variable, or a
                                // computed expression) still falls back to the
                                // same conservative default as before.
                                ExprKind::Var(name) => const_ints.get(name.as_str())
                                    .map(|n| *n as u32)
                                    .unwrap_or(1),
                                _ => 1,
                            }
                        };
                        let (bx, by, bz) = match &ba.value.kind {
                            ExprKind::Tuple(elems) => {
                                let g = |i: usize| elems.get(i).map(&parse_u32).unwrap_or(1);
                                (g(0), g(1), g(2))
                            }
                            _ => (parse_u32(&ba.value), 1, 1),
                        };
                        // Index by kernel type name (resolved from variable name if possible).
                        let key = var_to_type.get(kname).cloned().unwrap_or_else(|| kname.clone());
                        map.insert(key, (bx, by, bz));
                    }
                }
            }
        }
        _ => {}
    }
}
