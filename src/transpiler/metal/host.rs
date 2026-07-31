// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Rust host-side code emitter for the Metal backend.
// Uses the `metal` crate (objc2-metal migration planned).

use crate::ast::*;

pub(super) fn emit_host_rs(
    program: &Program,
    kernel_names: &[String],
    kernel_touching: &std::collections::HashSet<String>,
    general_code: &str,
    has_boring_main: bool,
    boring_main_throws: bool,
    top_level_kernel_touching: bool,
) -> String {
    let mut e = HostEmitter::new(kernel_names);
    e.emit_program(program, kernel_names, kernel_touching, general_code, has_boring_main, boring_main_throws, top_level_kernel_touching);
    e.out
}

struct HostEmitter {
    out: String,
    indent: usize,
    kernel_names: std::collections::HashSet<String>,
    kernel_decls: std::collections::HashMap<String, KernelDecl>,
    var_kernel_type: std::collections::HashMap<String, String>,
    gpu_vars: std::collections::HashSet<String>,
    // Top-level scalar lets (name → Rust expression), used to inline scalars
    // that are in scope in main() but not in kernel new() functions.
    top_level_scalars: std::collections::HashMap<String, String>,
    // Screen / display support
    screen_var: Option<String>,
    screen_width_expr: String,
    screen_height_expr: String,
    screen_title: String,
    // true while emitting render loop body (? → .expect() for kernel launches)
    in_render_loop: bool,
    /// Local variables (by name, unscoped) whose declared type or initializer is a
    /// `{K=V}` dict literal/type — used so `d[key]`/`d[key] = v` emit `HashMap`
    /// `.get(&key).cloned()`/`.insert(key, v)` instead of Vec-style `[key as usize]`
    /// indexing (see `is_dict_obj`). Mirrors `cuda::host`'s identical fields.
    dict_vars: std::collections::HashSet<String>,
    /// Struct field names (flat, not namespaced by struct) declared with a `{K=V}`
    /// dict type, so `self.field[key]`/`self.field[key] = v` get the same HashMap
    /// treatment as `dict_vars`. Populated once from every `Item::Struct`.
    dict_fields: std::collections::HashSet<String>,
    /// Names of every free (non-method) `throws` function. See `cuda::host`'s
    /// identical field for the full rationale and its documented scope limit
    /// (does not cover `throws` struct methods calling each other).
    fn_throws: std::collections::HashSet<String>,
    /// True while emitting the body of a `throws` function. See `cuda::host`'s
    /// identical field.
    in_throws: bool,
    /// Variant name → owning enum type name. See `cuda::host`'s identical field.
    variant_to_enum: std::collections::HashMap<String, String>,
    /// Free function name → per-parameter "does this position need `&`?" flags,
    /// per boring's by-ref contract. See `cuda::host`'s identical field for the
    /// full rationale.
    fn_ref_params: std::collections::HashMap<String, Vec<bool>>,
    /// Declared struct AND enum names. See `cuda::host`'s identical field.
    struct_names: std::collections::HashSet<String>,
    /// The currently-being-emitted function's own by-ref-typed parameter names.
    /// See `cuda::host`'s identical field.
    ref_params: std::collections::HashSet<String>,
    /// `Some(elem)` while emitting the body of a GPU-resident-returning
    /// function, where `elem` is the general pass's host element type (see
    /// `general_host_elem_type`) -- unlike `cuda::host`'s identical (bool)
    /// field, this backend needs the actual type string at the wrap site to
    /// convert from its own native `f32` buffer element to the `f64` every
    /// general-spliced caller expects.
    in_resident_return: Option<String>,
    /// Free function name → its `Type` when GPU-resident-returning. See
    /// `cuda::host`'s identical field.
    fn_returns_resident: std::collections::HashMap<String, Type>,
    /// Free function name → per-parameter "is this a `[float]` array?" flags.
    /// See `is_float_array_param`'s doc for why this backend, uniquely among
    /// the three GPU targets, needs this (Metal's native buffer width is f32,
    /// but the general pass's host convention is fixed at f64).
    fn_float_array_params: std::collections::HashMap<String, Vec<bool>>,
    /// Local variable names (current function only, reset per `emit_fn` call,
    /// mirrors `ref_params`' scoping) bound directly to a materializing call
    /// (`let k_t = transpose_gpu(...)`) -- these are `Vec<f64>` (the general
    /// pass's convention, see `general_host_elem_type`'s doc), NOT this
    /// backend's native `Vec<f32>`. A kernel constructor's own (untouched)
    /// field types are always native `f32`, so passing one of these needs an
    /// explicit `f64`→`f32` cast at that specific call site (see the kernel-
    /// constructor branch of `expr()`'s `Call` case).
    f64_array_locals: std::collections::HashSet<String>,
    /// Local variable names (current function only, reset per `emit_fn` call,
    /// same scoping as `f64_array_locals`) bound to a call into a
    /// `fn_returns_resident` function WHOSE OWN `let` type is itself
    /// `'gpu'unified` (`s.ty.gpu_resident_qual().is_some()`) -- these stay
    /// `BoringGpuArg<f64>`-typed (never unwrapped to a plain `Vec` at the
    /// binding), so a later kernel-constructor call passing one of these as
    /// an argument can hand the underlying `Buffer` straight through instead
    /// of reading it back to host and re-uploading. See the kernel-
    /// constructor branch of `expr()`'s `Call` case, and `Stmt::Let`'s own
    /// handling for where this set is populated.
    resident_locals: std::collections::HashSet<String>,
    /// One-shot flag: `Stmt::Let` sets this to `true` immediately before
    /// calling `self.expr(val)` for a call it has determined should stay
    /// `BoringGpuArg`-typed (see `resident_locals`'s doc). The `Call` arm of
    /// `expr()` takes (consumes) this flag for ITS OWN top-level call only,
    /// before recursing into argument sub-expressions, so a resident-
    /// preserving call passed as an argument to this outer call is not
    /// incorrectly also suppressed.
    suppress_resident_materialize: bool,
}

/// See `cuda::host`'s identical function.
fn is_ref_worthy_type(ty: &Type, struct_names: &std::collections::HashSet<String>) -> bool {
    match ty {
        Type::Array(_) | Type::ArrayN(_, _) | Type::Dict(_, _) | Type::Set(_) => true,
        Type::Named(n) => struct_names.contains(n),
        _ => false,
    }
}

/// True for a `[float]` array param -- this backend's own kernel-touching-
/// function signatures render these as `Vec<f32>` (`rust_type`'s Metal-native
/// convention), but every general-spliced caller's own local is `Vec<f64>`
/// (the general pass's fixed host convention, see `general_host_elem_type`'s
/// doc) -- a real E0308 confirmed via a real cross-compile `cargo check`
/// (`&Vec<f32>` vs `&Vec<f64>`). Such a param is instead DECLARED as
/// `&Vec<f64>` and immediately shadow-rebound to an owned `Vec<f32>` local
/// (see `emit_fn`).
fn is_float_array_param(ty: &Type) -> bool {
    fn is_float(ty: &Type) -> bool {
        matches!(ty, Type::Float) || matches!(ty, Type::Named(n) if n == "float" || n == "f32" || n == "f64")
    }
    match ty {
        Type::Array(inner) | Type::ArrayN(inner, _) => is_float(inner),
        _ => false,
    }
}

impl HostEmitter {
    fn new(kernel_names: &[String]) -> Self {
        Self {
            out: String::new(),
            indent: 0,
            kernel_names: kernel_names.iter().cloned().collect(),
            kernel_decls: std::collections::HashMap::new(),
            var_kernel_type: std::collections::HashMap::new(),
            gpu_vars: std::collections::HashSet::new(),
            top_level_scalars: std::collections::HashMap::new(),
            screen_var: None,
            screen_width_expr: String::new(),
            screen_height_expr: String::new(),
            screen_title: String::new(),
            in_render_loop: false,
            dict_vars: std::collections::HashSet::new(),
            dict_fields: std::collections::HashSet::new(),
            fn_throws: std::collections::HashSet::new(),
            in_throws: false,
            variant_to_enum: std::collections::HashMap::new(),
            fn_ref_params: std::collections::HashMap::new(),
            struct_names: std::collections::HashSet::new(),
            ref_params: std::collections::HashSet::new(),
            in_resident_return: None,
            fn_returns_resident: std::collections::HashMap::new(),
            fn_float_array_params: std::collections::HashMap::new(),
            f64_array_locals: std::collections::HashSet::new(),
            resident_locals: std::collections::HashSet::new(),
            suppress_resident_materialize: false,
        }
    }

    /// See `cuda::host`'s identical function.
    fn coerce_call_arg(&mut self, arg: &Expr, callee_expects_ref: bool) -> String {
        let is_ref_var = matches!(&arg.kind, ExprKind::Var(v) if self.ref_params.contains(v.as_str()));
        let s = self.expr(arg);
        match (callee_expects_ref, is_ref_var) {
            (true, true) => s,
            (true, false) => format!("&({})", s),
            (false, true) => format!("({}).clone()", s),
            (false, false) => s,
        }
    }

    /// True when `obj` (an `Index`/`Assign` target's receiver) is known to be a
    /// dict. See `cuda::host`'s identical helper for the full rationale.
    fn is_dict_obj(&self, obj: &Expr) -> bool {
        match &obj.kind {
            ExprKind::Var(v) => self.dict_vars.contains(v.as_str()),
            ExprKind::Field(_, f) => self.dict_fields.contains(f.as_str()),
            _ => false,
        }
    }

    /// Record `name` in `dict_vars` when its declared type or initializer marks it
    /// as a dict. See `cuda::host`'s identical helper.
    fn track_dict_var(&mut self, name: &str, ty: Option<&Type>, val: Option<&Expr>) {
        fn ty_is_dict(ty: &Type) -> bool {
            match ty {
                Type::Dict(..) => true,
                Type::Qualified(inner, _) => ty_is_dict(inner),
                _ => false,
            }
        }
        let is_dict = ty.is_some_and(ty_is_dict)
            || matches!(val.map(|v| &v.kind), Some(ExprKind::Dict(_)));
        if is_dict {
            self.dict_vars.insert(name.to_string());
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
        kernel_names: &[String],
        kernel_touching: &std::collections::HashSet<String>,
        general_code: &str,
        has_boring_main: bool,
        boring_main_throws: bool,
        top_level_kernel_touching: bool,
    ) {
        // Pass 1: struct/enum names, needed below to resolve whether a
        // `Type::Named(n)` function parameter is by-ref-worthy.
        for item in &program.items {
            match item {
                Item::Struct(s) => { self.struct_names.insert(s.name.clone()); }
                Item::Enum(e) => { self.struct_names.insert(e.name.clone()); }
                _ => {}
            }
        }
        for item in &program.items {
            if let Item::Kernel(decl) = item {
                self.kernel_decls.insert(decl.name.clone(), decl.clone());
            }
            if let Item::Struct(s) = item {
                for f in &s.fields {
                    if matches!(f.ty, Type::Dict(..)) {
                        self.dict_fields.insert(f.name.clone());
                    }
                }
            }
            if let Item::Fn(f) = item {
                if f.throws {
                    self.fn_throws.insert(f.name.clone());
                }
                let ref_flags: Vec<bool> = f.params.iter()
                    .map(|p| p.ty.as_ref().is_some_and(|ty| is_ref_worthy_type(ty, &self.struct_names)))
                    .collect();
                self.fn_ref_params.insert(f.name.clone(), ref_flags);
                let float_array_flags: Vec<bool> = f.params.iter()
                    .map(|p| p.ty.as_ref().is_some_and(is_float_array_param))
                    .collect();
                self.fn_float_array_params.insert(f.name.clone(), float_array_flags);
                if let Some(rt) = &f.return_ty {
                    if rt.gpu_resident_qual().is_some() {
                        self.fn_returns_resident.insert(f.name.clone(), rt.clone());
                    }
                }
            }
            if let Item::Enum(e) = item {
                for v in &e.variants {
                    self.variant_to_enum.insert(v.name.clone(), e.name.clone());
                }
            }
        }

        // Pre-pass: collect top-level scalar lets so they can be inlined inside
        // kernel new() functions (emitted before main(), where these vars live).
        for item in &program.items {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    let is_scalar = matches!(val.kind,
                        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_)
                    ) || s.ty.as_ref().map(|t| matches!(t,
                        Type::Int | Type::Uint | Type::Float | Type::Bool
                    )).unwrap_or(false);
                    if is_scalar {
                        let rhs = self.expr(val);
                        self.top_level_scalars.insert(s.name.clone(), rhs);
                    }
                }
            }
        }

        // Pre-pass: detect Screen before prelude so emit_prelude can add screen helpers.
        self.detect_screen(program);

        self.line("// Generated by boring build --target metal.");
        self.line("#![allow(dead_code, unused_variables, unused_parens, unexpected_cfgs)]");
        self.blank();
        self.emit_prelude();

        // Kernel structs (unchanged, own emission) + kernel-touching function
        // bodies (this backend's own real Metal API) -- see `metal::mod`'s doc
        // comment (mirrors `cuda::mod`'s identical architecture). A `Screen`-
        // using program keeps its ENTIRE top-level/main-building on this
        // backend's own existing (working) Screen-aware driver below, same as
        // wgpu's own `has_screen` carve-out -- the general splice only covers
        // ordinary fn/struct/enum either way. Every other item (plain
        // fn/struct/enum, non-Screen top-level stmt/let folded into
        // `boring_main`) is already rendered correctly in `general_code`.
        for item in &program.items {
            match item {
                Item::Kernel(decl) => {
                    self.blank();
                    self.emit_kernel_struct(decl);
                }
                Item::Fn(f) if kernel_touching.contains(&f.name) => {
                    self.blank();
                    if f.name == "main" {
                        let mut renamed = f.clone();
                        renamed.name = "boring_main".to_string();
                        self.emit_fn(&renamed, None);
                    } else {
                        self.emit_fn(f, None);
                    }
                }
                Item::Struct(s) if self.screen_var.is_some() => {
                    self.blank();
                    self.emit_struct(s);
                }
                Item::Enum(e) if self.screen_var.is_some() => {
                    self.blank();
                    self.emit_enum(e);
                }
                _ => {}
            }
        }

        if self.screen_var.is_none() {
            self.blank();
            self.out.push_str(general_code);
        }
        self.blank();

        if self.screen_var.is_some() || top_level_kernel_touching || !kernel_names.is_empty() || program.items.iter().any(|i| matches!(i, Item::Stmt(_) | Item::Let(_))) {
            self.line("fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
            self.indent += 1;
            if self.screen_var.is_some() || top_level_kernel_touching {
                // Bare top-level kernel construction/dispatch or a `Screen`
                // program -- this backend's own pre-splice top-level handling,
                // reinstated only for these cases (see `top_level_touches_kernel`'s
                // doc, and wgpu's own identical `has_screen` carve-out).
                let mut screen_setup_emitted = false;
                for item in &program.items {
                    match item {
                        Item::Let(s) => {
                            if self.screen_var.as_deref() == Some(s.name.as_str()) {
                                if !screen_setup_emitted {
                                    self.emit_screen_setup();
                                    screen_setup_emitted = true;
                                }
                                continue;
                            }
                            let binding = if s.binding.is_mutable() { "let mut" } else { "let" };
                            let ty_ann = s.ty.as_ref().map(|t| format!(": {}", rust_type(t))).unwrap_or_default();
                            if let Some(val) = &s.value {
                                self.track_kernel_var(&s.name, val);
                                self.track_dict_var(&s.name, s.ty.as_ref(), Some(val));
                                let rhs = self.expr(val);
                                let is_scalar = matches!(val.kind,
                                    ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_)
                                ) || s.ty.as_ref().map(|t| matches!(t,
                                    Type::Int | Type::Uint | Type::Float | Type::Bool
                                )).unwrap_or(false);
                                if is_scalar {
                                    self.top_level_scalars.insert(s.name.clone(), rhs.clone());
                                }
                                self.line(&format!("{} {}{} = {};", binding, s.name, ty_ann, rhs));
                            }
                        }
                        Item::Stmt(stmt) => self.emit_stmt(stmt),
                        _ => {}
                    }
                }
            } else if has_boring_main {
                if boring_main_throws {
                    self.line("boring_main()?;");
                } else {
                    self.line("boring_main();");
                }
            }
            self.line("Ok(())");
            self.indent -= 1;
            self.line("}");
        }
    }

    // ── Metal prelude ──────────────────────────────────────────────────────────

    fn emit_prelude(&mut self) {
        self.line("use metal::*;");
        self.line("use std::mem;");
        self.blank();
        self.line("const BORING_MSL: &str = include_str!(\"../kernels/main.metal\");");
        self.blank();
        // Cache the compiled MSL library and each kernel's compute pipeline
        // state thread-locally. Every kernel dispatch used to construct a
        // fresh kernel struct, which used to recompile BORING_MSL from source
        // AND rebuild the pipeline state on every single call -- for a model
        // with any nontrivial number of GPU calls (e.g. once per matmul per
        // layer per token) this made shader (re)compilation the dominant
        // cost, swamping the actual GPU compute it was meant to measure.
        // Metal's own types aren't Send/Sync (ObjC-backed), so this is a
        // thread_local rather than a `static`/`OnceLock` -- fine since kernel
        // dispatch here is single-threaded.
        self.line("thread_local! {");
        self.indent += 1;
        self.line("static __BORING_METAL_LIBRARY: std::cell::RefCell<Option<Library>> = std::cell::RefCell::new(None);");
        self.line("static __BORING_METAL_PIPELINES: std::cell::RefCell<std::collections::HashMap<&'static str, ComputePipelineState>> = std::cell::RefCell::new(std::collections::HashMap::new());");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn __boring_metal_pipeline(__device: &Device, __kernel_fn_name: &'static str) -> Result<ComputePipelineState, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("if let Some(__p) = __BORING_METAL_PIPELINES.with(|c| c.borrow().get(__kernel_fn_name).cloned()) {");
        self.indent += 1;
        self.line("return Ok(__p);");
        self.indent -= 1;
        self.line("}");
        self.line("let __library = __BORING_METAL_LIBRARY.with(|c| c.borrow().clone());");
        self.line("let __library = match __library {");
        self.indent += 1;
        self.line("Some(__l) => __l,");
        self.line("None => {");
        self.indent += 1;
        self.line("let __options = CompileOptions::new();");
        // Metal's compile options default `fastMathEnabled` to `true` (Apple's
        // own default), which permits the compiler to assume no NaN/Inf and
        // reorder/approximate transcendental functions (exp, tanh, ...) --
        // this silently produced actual NaN output from a provably finite,
        // in-range input (confirmed: GeluKernel's `tanh(58.68)` produced NaN
        // under the default with real model weights, where plain IEEE-754
        // tanh cannot). Boring's language semantics are standard predictable
        // float arithmetic, not opt-in fast-math, so turn it off.
        self.line("__options.set_fast_math_enabled(false);");
        self.line("let __l = __device.new_library_with_source(BORING_MSL, &__options)");
        self.indent += 1;
        self.line(".map_err(|e| format!(\"MSL compile error: {}\", e))?;");
        self.indent -= 1;
        self.line("__BORING_METAL_LIBRARY.with(|c| *c.borrow_mut() = Some(__l.clone()));");
        self.line("__l");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("};");
        self.line("let __func = __library.get_function(__kernel_fn_name, None)");
        self.indent += 1;
        self.line(".map_err(|e| format!(\"kernel function not found: {}\", e))?;");
        self.indent -= 1;
        self.line("let __pipeline = __device.new_compute_pipeline_state_with_function(&__func)?;");
        self.line("__BORING_METAL_PIPELINES.with(|c| c.borrow_mut().insert(__kernel_fn_name, __pipeline.clone()));");
        self.line("Ok(__pipeline)");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // Default device helper.
        self.line("fn boring_metal_device() -> Device {");
        self.indent += 1;
        self.line("Device::system_default().expect(\"no Metal device found — macOS 10.14+ required\")");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // GPU(n) → indexed device.
        self.line("fn boring_metal_device_n(idx: usize) -> Result<Device, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("let devices = Device::all();");
        self.line("devices.into_iter().nth(idx)");
        self.indent += 1;
        self.line(".ok_or_else(|| format!(\"GPU index {} out of range\", idx).into())");
        self.indent -= 1;
        self.indent -= 1;
        self.line("}");
        self.blank();
        // GPU family tier helper (computeCapability equivalent).
        self.line("fn boring_gpu_family(device: &Device) -> Vec<isize> {");
        self.indent += 1;
        self.line("let families: &[(MTLGPUFamily, [isize; 2])] = &[");
        self.indent += 1;
        self.line("(MTLGPUFamily::Apple9, [9, 0]),");
        self.line("(MTLGPUFamily::Apple8, [8, 0]),");
        self.line("(MTLGPUFamily::Apple7, [7, 0]),");
        self.line("(MTLGPUFamily::Apple6, [6, 0]),");
        self.line("(MTLGPUFamily::Apple5, [5, 0]),");
        self.line("(MTLGPUFamily::Apple4, [4, 0]),");
        self.line("(MTLGPUFamily::Apple3, [3, 0]),");
        self.line("(MTLGPUFamily::Apple2, [2, 0]),");
        self.line("(MTLGPUFamily::Apple1, [1, 0]),");
        self.indent -= 1;
        self.line("];");
        self.line("for (family, ver) in families {");
        self.indent += 1;
        self.line("if device.supports_family(*family) { return ver.to_vec(); }");
        self.indent -= 1;
        self.line("}");
        self.line("vec![0, 0]");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // Every kernel dispatch used to `commit()` + `wait_until_completed()`
        // synchronously inside its own `__boring_launch` -- for a model doing
        // many small GPU calls (once per matmul per layer per token), that
        // per-dispatch CPU-blocking wait was the dominant cost, dwarfing the
        // actual GPU compute it was meant to measure (confirmed: real wall
        // time was ~5x the combined user+sys CPU time on a real whisper-
        // boring run -- the CPU was mostly just blocked waiting). All kernel
        // structs now share ONE persistent command queue (`__boring_metal_queue`)
        // instead of each opening its own, and `__boring_launch` only commits
        // (no wait) -- Metal's default per-queue FIFO ordering plus automatic
        // buffer hazard tracking (on by default; not disabled by
        // `StorageModeShared`, an orthogonal storage-mode bit) means a LATER
        // dispatch on the same queue correctly sees an EARLIER one's writes
        // without any CPU-side wait between them. The wait is deferred to the
        // one place it's actually needed: reading a buffer's contents back to
        // the CPU (`read_<field>()`, `__boring_gpu_copy_d2h`) -- and even
        // then, waiting on only the LATEST committed buffer suffices, since
        // it cannot complete before everything queued ahead of it already has.
        self.line("thread_local! {");
        self.indent += 1;
        self.line("static __BORING_METAL_QUEUE: std::cell::RefCell<Option<CommandQueue>> = std::cell::RefCell::new(None);");
        self.line("static __BORING_METAL_PENDING: std::cell::RefCell<Option<CommandBuffer>> = std::cell::RefCell::new(None);");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn __boring_metal_queue(__device: &Device) -> CommandQueue {");
        self.indent += 1;
        self.line("__BORING_METAL_QUEUE.with(|c| {");
        self.indent += 1;
        self.line("if let Some(q) = &*c.borrow() { return q.clone(); }");
        self.line("let q = __device.new_command_queue();");
        self.line("*c.borrow_mut() = Some(q.clone());");
        self.line("q");
        self.indent -= 1;
        self.line("})");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // `wait_until_completed` alone never surfaced a GPU-side failure
        // (invalid threadgroup size rejected asynchronously, an out-of-bounds
        // buffer access, device removal, ...) -- the command buffer just
        // finished with `status() == Error` and nobody looked. Checking status
        // here (the one place every deferred dispatch's completion is
        // actually observed) is what turns that into a real, catchable error
        // instead of silent wrong behavior.
        self.line("fn __boring_metal_flush() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("__BORING_METAL_PENDING.with(|c| {");
        self.indent += 1;
        self.line("if let Some(buf) = c.borrow_mut().take() {");
        self.indent += 1;
        self.line("buf.wait_until_completed();");
        self.line("if buf.status() == MTLCommandBufferStatus::Error {");
        self.indent += 1;
        self.line("return Err(format!(\"Metal command buffer failed: {:?}\", buf.status()).into());");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("Ok(())");
        self.indent -= 1;
        self.line("})");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // KernelHandle — `.wait()`/`.done()` don't themselves force a GPU
        // sync; any real data read goes through `read_<field>()`/
        // `__boring_gpu_copy_d2h`, which already flush (see above).
        self.line("#[must_use = \"a KernelHandle must be waited on (.wait/.inner) or the launch may not be synchronized\"]");
        self.line("struct KernelHandle<T> { inner: T }");
        self.line("impl<T> KernelHandle<T> {");
        self.indent += 1;
        self.line("fn wait(self) -> T { self.inner }");
        self.line("fn done(&self) -> bool { true }");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // Boring built-in Dimension type used by 2-D kernels.
        self.line(crate::transpiler::helpers::DIMENSION_STRUCT_RUST);
        self.blank();
        // The general (std/wgpu-shared) transpiler pipeline's own pre-pass marks any
        // function whose declared return type is `'gpu'unified`/`'gpu'global`-qualified
        // as "GPU-resident-returning" and renders every CALLER of it (in the general-
        // pipeline-spliced "plain" code this backend embeds -- see `metal::mod`'s doc
        // comment) to expect a `BoringGpuArg<T>` value back, unconditionally.
        // Unlike CUDA (where a genuine device-to-device buffer handoff is the only way
        // to avoid a real `cudaMemcpy`), Metal's buffers use `StorageModeShared` --
        // real, unified CPU/GPU memory -- so `Resident` here holds a real `metal::Buffer`
        // handle (cheaply `Clone`-able: an ObjC retain, not a data copy -- see
        // `foreign_obj_type!` in the `metal` crate) instead of `wgpu::host`'s
        // `Arc<wgpu::Buffer>` convention. `emit_fn`'s tail-expression codegen
        // constructs this variant directly from a kernel struct's own output buffer
        // when the tail expression is a bare `k.field` read, skipping the
        // read-to-Vec-then-reupload round trip a chained GPU call used to always pay.
        self.line("#[allow(dead_code)]");
        self.line("enum BoringGpuArg<T> {");
        self.indent += 1;
        self.line("Resident(Buffer, usize),");
        self.line("Host(Vec<T>),");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("#[allow(dead_code)]");
        self.line("impl<T: Clone> Clone for BoringGpuArg<T> {");
        self.indent += 1;
        self.line("fn clone(&self) -> Self {");
        self.indent += 1;
        self.line("match self {");
        self.indent += 1;
        self.line("BoringGpuArg::Resident(b, n) => BoringGpuArg::Resident(b.clone(), *n),");
        self.line("BoringGpuArg::Host(v) => BoringGpuArg::Host(v.clone()),");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("#[allow(dead_code)]");
        self.line("impl<T> BoringGpuArg<T> {");
        self.indent += 1;
        self.line("fn len(&self) -> usize {");
        self.indent += 1;
        self.line("match self { BoringGpuArg::Resident(_, len) => *len, BoringGpuArg::Host(v) => v.len() }");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("#[allow(dead_code)] fn __boring_gpu_device() -> Device { boring_metal_device() }");
        self.line("#[allow(dead_code)] fn __boring_gpu_queue() -> Device { boring_metal_device() }");
        self.line("#[allow(dead_code)]");
        self.line("fn __boring_gpu_copy_d2h<T>(_device: &Device, _queue: &Device, buf: &Buffer) -> Vec<f32> {");
        self.indent += 1;
        // `__boring_launch` no longer waits synchronously (see the prelude's
        // `__boring_metal_flush` doc) -- flush here before reading, exactly
        // like `read_<field>()`. `.expect()`, not `?`, matching `cuda::host`'s
        // own identical (non-`Result`) signature for this same interprocedural-
        // materialization helper -- unlike `read_<field>()`, propagating a real
        // error here would require threading `Result` through every kernel-
        // touching function that calls another one returning a GPU-resident
        // value, not just this file's own always-`Result` `main()`.
        self.line("__boring_metal_flush().expect(\"metal: GPU dispatch failed before D2H copy\");");
        self.line("let n = buf.length() as usize / mem::size_of::<f32>();");
        self.line("let ptr = buf.contents() as *const f32;");
        self.line("unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }");
        self.indent -= 1;
        self.line("}");
        self.line("#[allow(dead_code)]");
        self.line("fn __boring_gpu_copy_h2d<T>(_device: &Device, _queue: &Device, _src: &[u8], _dst: &Buffer) {");
        self.indent += 1;
        self.line("unreachable!(\"metal backend never constructs a host-to-device upload through this path -- kernel-constructor call sites upload directly\")");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // A real device-to-device copy: allocate a fresh buffer and memcpy into
        // it, NOT `Buffer::clone()` -- confirmed against the real `metal` crate
        // source that `Clone` on an ObjC wrapper type (`foreign_type!`'s
        // generated impl) is just an ObjC `retain` (reference-count bump), not a
        // content copy. Using `.clone()` here meant two kernel structs silently
        // shared the exact same underlying `MTLBuffer` -- if the source kernel
        // was ever dispatched again afterward, the "copy"'s contents changed
        // too, with no compile error and no warning (unlike cuda::host's
        // equivalent bug, a real E0382 the compiler catches). The raw
        // `contents()` memcpy is valid because every buffer this backend
        // allocates uses `MTLResourceOptions::StorageModeShared` (CPU+GPU
        // unified memory) -- flushing first (see `__boring_metal_flush`'s doc)
        // is required since dispatch is deferred: without it, this could copy
        // from a buffer the GPU hasn't finished writing yet.
        self.line("fn __boring_metal_buffer_copy(dev: &Device, buf: &Buffer) -> Result<Buffer, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("__boring_metal_flush()?;");
        self.line("let len = buf.length();");
        self.line("let new_buf = dev.new_buffer(len, MTLResourceOptions::StorageModeShared);");
        self.line("unsafe {");
        self.indent += 1;
        self.line("std::ptr::copy_nonoverlapping(buf.contents() as *const u8, new_buf.contents() as *mut u8, len as usize);");
        self.indent -= 1;
        self.line("}");
        self.line("Ok(new_buf)");
        self.indent -= 1;
        self.line("}");
        self.blank();
        if self.screen_var.is_some() {
            self.emit_screen_prelude();
        }
    }

    // ── Screen / display support ───────────────────────────────────────────────

    fn detect_screen(&mut self, program: &Program) {
        for item in &program.items {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    if let ExprKind::Call(callee, args) = &val.kind {
                        if let ExprKind::Var(name) = &callee.kind {
                            if name == "Screen" {
                                let (w, h) = args.first()
                                    .map(|a| {
                                        if let ExprKind::Call(c2, a2) = &a.value.kind {
                                            if let ExprKind::Var(n) = &c2.kind {
                                                if n == "Dimension" {
                                                    let w = a2.first().map(|x| self.expr(&x.value)).unwrap_or_else(|| "800".into());
                                                    let h = a2.get(1).map(|x| self.expr(&x.value)).unwrap_or_else(|| "600".into());
                                                    return (w, h);
                                                }
                                            }
                                        }
                                        ("800".into(), "600".into())
                                    })
                                    .unwrap_or_else(|| ("800".into(), "600".into()));
                                let title_expr = args.iter()
                                    .find(|a| a.label.as_deref() == Some("title"))
                                    .or_else(|| args.get(1))
                                    .map(|a| self.expr(&a.value))
                                    .unwrap_or_else(|| "\"Boring\"".into());
                                let title_str = if title_expr.starts_with('"') && title_expr.ends_with('"') {
                                    title_expr[1..title_expr.len()-1].to_string()
                                } else {
                                    "Boring".to_string()
                                };
                                self.screen_var = Some(s.name.clone());
                                self.screen_width_expr = w;
                                self.screen_height_expr = h;
                                self.screen_title = title_str;
                                return;
                            }
                        }
                    }
                }
            }
        }
    }

    fn emit_screen_prelude(&mut self) {
        self.blank();
        self.line("#[macro_use] extern crate objc;");
        self.blank();
        self.line("fn boring_screen_present(");
        self.indent += 1;
        self.line("layer: &metal::MetalLayer,");
        self.line("queue: &metal::CommandQueue,");
        self.line("src: &metal::Buffer,");
        self.line("width: u64,");
        self.line("height: u64,");
        self.indent -= 1;
        self.line(") {");
        self.indent += 1;
        self.line("let drawable = match layer.next_drawable() { Some(d) => d, None => return };");
        self.line("let cmd_buf = queue.new_command_buffer();");
        self.line("let blit = cmd_buf.new_blit_command_encoder();");
        self.line("blit.copy_from_buffer_to_texture(");
        self.indent += 1;
        self.line("src, 0, width * 4, width * height * 4,");
        self.line("metal::MTLSize { width, height, depth: 1 },");
        self.line("drawable.texture(), 0, 0,");
        self.line("metal::MTLOrigin { x: 0, y: 0, z: 0 },");
        self.line("metal::MTLBlitOption::empty(),");
        self.indent -= 1;
        self.line(");");
        self.line("blit.end_encoding();");
        self.line("cmd_buf.present_drawable(&drawable);");
        self.line("cmd_buf.commit();");
        self.line("cmd_buf.wait_until_completed();");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn boring_key_str(key: winit::event::VirtualKeyCode) -> String {");
        self.indent += 1;
        self.line("use winit::event::VirtualKeyCode::*;");
        self.line("match key {");
        self.indent += 1;
        self.line("Escape => \"\\x1B\".into(),");
        self.line("Space => \" \".into(),");
        self.line("Return => \"\\r\".into(),");
        self.line("Up => \"\\x1B[A\".into(),");
        self.line("Down => \"\\x1B[B\".into(),");
        self.line("Left => \"\\x1B[D\".into(),");
        self.line("Right => \"\\x1B[C\".into(),");
        self.line("A => \"a\".into(), B => \"b\".into(), C => \"c\".into(),");
        self.line("D => \"d\".into(), E => \"e\".into(), F => \"f\".into(),");
        self.line("G => \"g\".into(), H => \"h\".into(), I => \"i\".into(),");
        self.line("J => \"j\".into(), K => \"k\".into(), L => \"l\".into(),");
        self.line("M => \"m\".into(), N => \"n\".into(), O => \"o\".into(),");
        self.line("P => \"p\".into(), Q => \"q\".into(), R => \"r\".into(),");
        self.line("S => \"s\".into(), T => \"t\".into(), U => \"u\".into(),");
        self.line("V => \"v\".into(), W => \"w\".into(), X => \"x\".into(),");
        self.line("Y => \"y\".into(), Z => \"z\".into(),");
        self.line("Key1 => \"1\".into(), Key2 => \"2\".into(), Key3 => \"3\".into(),");
        self.line("Key4 => \"4\".into(), Key5 => \"5\".into(), Key6 => \"6\".into(),");
        self.line("Key7 => \"7\".into(), Key8 => \"8\".into(), Key9 => \"9\".into(),");
        self.line("Key0 => \"0\".into(),");
        self.line("_ => format!(\"{:?}\", key).to_lowercase(),");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
    }

    fn emit_screen_setup(&mut self) {
        let w = self.screen_width_expr.clone();
        let h = self.screen_height_expr.clone();
        let title = self.screen_title.clone();
        self.line("use winit::platform::run_return::EventLoopExtRunReturn;");
        self.line("let mut boring_event_loop = winit::event_loop::EventLoop::new();");
        self.line("let boring_window = winit::window::WindowBuilder::new()");
        self.indent += 1;
        self.line(&format!(".with_title(\"{}\")", title));
        self.line(&format!(".with_inner_size(winit::dpi::LogicalSize::new({w} as u32, {h} as u32))"));
        self.line(".build(&boring_event_loop)?;");
        self.indent -= 1;
        self.line("let boring_device = boring_metal_device();");
        self.line("let boring_layer = metal::MetalLayer::new();");
        self.line("boring_layer.set_device(&boring_device);");
        self.line("boring_layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);");
        self.line("boring_layer.set_framebuffer_only(false);");
        self.line(&format!("boring_layer.set_drawable_size(core_graphics::geometry::CGSize::new({w} as f64, {h} as f64));"));
        self.line("unsafe {");
        self.indent += 1;
        self.line("use winit::platform::macos::WindowExtMacOS;");
        self.line("use objc::runtime::*;");
        self.line("let ns_view = boring_window.ns_view() as *mut Object;");
        self.line("let () = objc::msg_send![ns_view, setLayer: boring_layer.as_ref()];");
        self.line("let () = objc::msg_send![ns_view, setWantsLayer: YES];");
        self.indent -= 1;
        self.line("}");
        self.line("let boring_queue = boring_device.new_command_queue();");
        self.line("let mut boring_frame: isize = 0;");
        self.line("let boring_start = std::time::Instant::now();");
        self.line("let mut boring_keys: std::collections::HashSet<String> = Default::default();");
        self.line(&format!("let mut boring_screen_width: isize = {w};"));
        self.line(&format!("let mut boring_screen_height: isize = {h};"));
        self.line("let mut boring_screen_resized = false;");
        self.line("let mut boring_screen_closed = false;");
    }

    fn emit_render_loop(&mut self, loop_body: &[Stmt]) {
        self.line("boring_event_loop.run_return(|__boring_event, _, __boring_cf| {");
        self.indent += 1;
        self.line("*__boring_cf = winit::event_loop::ControlFlow::Poll;");
        self.line("boring_screen_resized = false;");
        self.line("match __boring_event {");
        self.indent += 1;
        self.line("winit::event::Event::WindowEvent { event: winit::event::WindowEvent::CloseRequested, .. } => {");
        self.indent += 1;
        self.line("boring_screen_closed = true;");
        self.line("*__boring_cf = winit::event_loop::ControlFlow::Exit;");
        self.indent -= 1;
        self.line("}");
        self.line("winit::event::Event::WindowEvent { event: winit::event::WindowEvent::Resized(__boring_size), .. } => {");
        self.indent += 1;
        self.line("boring_screen_width = __boring_size.width as isize;");
        self.line("boring_screen_height = __boring_size.height as isize;");
        self.line("boring_screen_resized = true;");
        // Don't resize the drawable — it stays fixed at the surface buffer dimensions.
        self.indent -= 1;
        self.line("}");
        self.line("winit::event::Event::WindowEvent { event: winit::event::WindowEvent::KeyboardInput { input: __boring_ki, .. }, .. } => {");
        self.indent += 1;
        self.line("if let Some(__boring_key) = __boring_ki.virtual_keycode {");
        self.indent += 1;
        self.line("let __boring_ks = boring_key_str(__boring_key);");
        self.line("match __boring_ki.state {");
        self.indent += 1;
        self.line("winit::event::ElementState::Pressed  => { boring_keys.insert(__boring_ks); }");
        self.line("winit::event::ElementState::Released => { boring_keys.remove(&__boring_ks); }");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("winit::event::Event::MainEventsCleared => {");
        self.indent += 1;
        self.in_render_loop = true;
        let stmts: Vec<Stmt> = loop_body.to_vec();
        for stmt in &stmts {
            self.emit_render_loop_stmt(stmt);
        }
        self.in_render_loop = false;
        self.line("boring_frame += 1;");
        self.indent -= 1;
        self.line("}");
        self.line("_ => {}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("});");
    }

    fn emit_render_loop_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Break(_, _) => {
                self.line("*__boring_cf = winit::event_loop::ControlFlow::Exit;");
                self.line("return;");
            }
            Stmt::Expr(e) => {
                // screen.present(pixels) special case
                if let ExprKind::MethodCall(obj, method, args) = &e.kind {
                    let screen_var = self.screen_var.clone();
                    if let ExprKind::Var(v) = &obj.kind {
                        if Some(v.as_str()) == screen_var.as_deref() && method == "present" {
                            if let Some(pixels_arg) = args.first() {
                                let pixels_ref = self.emit_pixels_ref(&pixels_arg.value);
                                // Use surface's own Dimension for blit, not window size.
                                let (w_expr, h_expr) = if let ExprKind::Field(pobj, pfield) = &pixels_arg.value.kind {
                                    if let ExprKind::Var(kvar) = &pobj.kind {
                                        let dim_field = self.kernel_decls.values()
                                            .find(|kd| kd.fields.iter().any(|f| &f.name == pfield && matches!(f.qual, GpuQual::Surface)))
                                            .and_then(|kd| kd.fields.iter().find(|f| matches!(&f.ty, Type::Named(n) if n == "Dimension")))
                                            .map(|f| f.name.clone());
                                        if let Some(df) = dim_field {
                                            (format!("{}.{}.width as u64", kvar, df),
                                             format!("{}.{}.height as u64", kvar, df))
                                        } else {
                                            ("boring_screen_width as u64".into(), "boring_screen_height as u64".into())
                                        }
                                    } else {
                                        ("boring_screen_width as u64".into(), "boring_screen_height as u64".into())
                                    }
                                } else {
                                    ("boring_screen_width as u64".into(), "boring_screen_height as u64".into())
                                };
                                self.line(&format!(
                                    "boring_screen_present(&boring_layer, &boring_queue, {pixels_ref}, {w_expr}, {h_expr});"
                                ));
                                return;
                            }
                        }
                    }
                }
                if let ExprKind::Assign(lhs, rhs) = &e.kind {
                    let l = self.expr(lhs);
                    let r = self.expr(rhs);
                    self.line(&format!("{l} = {r};"));
                } else if let Some(launch) = self.try_emit_kernel_launch_call(e) {
                    self.line(&format!("{launch};"));
                } else {
                    let s = self.expr(e);
                    self.line(&format!("{s};"));
                }
            }
            Stmt::If(i) => {
                for (idx, (cond, body)) in i.branches.iter().enumerate() {
                    let c = self.expr(cond);
                    if idx == 0 { self.line(&format!("if {c} {{")); }
                    else { self.line(&format!("}} else if {c} {{")); }
                    self.indent += 1;
                    let body_clone: Vec<Stmt> = body.to_vec();
                    for s in &body_clone { self.emit_render_loop_stmt(s); }
                    self.indent -= 1;
                }
                if let Some(else_body) = &i.else_body {
                    self.line("} else {");
                    self.indent += 1;
                    let else_clone: Vec<Stmt> = else_body.to_vec();
                    for s in &else_clone { self.emit_render_loop_stmt(s); }
                    self.indent -= 1;
                }
                self.line("}");
            }
            other => self.emit_stmt(other),
        }
    }

    fn emit_pixels_ref(&mut self, e: &Expr) -> String {
        if let ExprKind::Field(obj, field) = &e.kind {
            if let ExprKind::Var(obj_name) = &obj.kind {
                return format!("&{}.{}", obj_name, field);
            }
        }
        format!("&{}", self.expr(e))
    }

    // ── Kernel struct → Rust host wrapper ─────────────────────────────────────

    fn emit_kernel_struct(&mut self, decl: &KernelDecl) {
        let name = &decl.name;

        self.line(&format!("struct {} {{", name));
        self.indent += 1;
        self.line("__device: Device,");
        self.line("__queue: CommandQueue,");
        self.line("__pipeline: ComputePipelineState,");
        for field in &decl.fields {
            match field.qual {
                GpuQual::Sync | GpuQual::Local => {}
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Const | GpuQual::Surface => {
                    match &field.ty {
                        Type::Array(_) | Type::ArrayN(_, _) => {
                            self.line(&format!("{}: Buffer,", field.name));
                        }
                        _ => {
                            let ty = rust_type(&field.ty);
                            self.line(&format!("{}: {},", field.name, ty));
                        }
                    }
                }
            }
        }
        for field in &decl.fields {
            if matches!(field.qual, GpuQual::Local) {
                match &field.ty {
                    Type::Array(_) | Type::ArrayN(_, _) => {}
                    _ => {
                        let ty = rust_type(&field.ty);
                        self.line(&format!("{}: {},", field.name, ty));
                    }
                }
            }
        }
        self.indent -= 1;
        self.line("}");
        self.blank();

        self.line(&format!("impl {} {{", name));
        self.indent += 1;

        for init in &decl.inits {
            self.emit_kernel_new(name, &decl.fields, init);
            self.blank();
        }
        if decl.inits.is_empty() {
            self.emit_kernel_new_default(name, &decl.fields);
            self.blank();
        }

        // Read accessors for 'unified/'global array fields (D2H).
        for field in &decl.fields {
            match field.qual {
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Surface => {
                    if matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _)) {
                        let elem = elem_rust_type(&field.ty);
                        self.line(&format!(
                            "fn read_{}(&self) -> Result<Vec<{}>, Box<dyn std::error::Error + Send + Sync>> {{",
                            field.name, elem
                        ));
                        self.indent += 1;
                        // Ensure every dispatch committed so far (possibly
                        // including the one that wrote this very field) has
                        // actually finished before reading its contents -- and
                        // surface a real error if that dispatch's command
                        // buffer completed with an error status (see
                        // `__boring_metal_flush`'s doc), instead of silently
                        // reading back whatever garbage/zeroed memory a failed
                        // GPU write left behind.
                        self.line("__boring_metal_flush()?;");
                        self.line(&format!(
                            "let n = self.{}.length() as usize / mem::size_of::<{}>();",
                            field.name, elem
                        ));
                        self.line(&format!(
                            "let ptr = self.{}.contents() as *const {};",
                            field.name, elem
                        ));
                        self.line("Ok(unsafe { std::slice::from_raw_parts(ptr, n).to_vec() })");
                        self.indent -= 1;
                        self.line("}");
                        self.blank();
                    }
                }
                _ => {}
            }
        }

        self.emit_boring_launch(name, &decl.fields);

        self.indent -= 1;
        self.line("}");
    }

    fn emit_kernel_new(&mut self, name: &str, fields: &[KernelFieldDecl], init: &InitDecl) {
        let buffer_flags = self.kernel_ctor_buffer_flags(name).unwrap_or_default();
        let params: Vec<String> = init.params.iter().enumerate().map(|(i, p)| {
            // A buffer-passthrough param (see `kernel_ctor_buffer_flags`) takes
            // an already-built `Buffer` directly -- the call site is
            // responsible for either reusing a resident one or uploading a
            // fresh one from host data, instead of this constructor doing the
            // upload itself (see `emit_init_stmt`'s matching change).
            let ty = if buffer_flags.get(i).copied().unwrap_or(false) {
                "Buffer".to_string()
            } else {
                p.ty.as_ref().map(host_param_type).unwrap_or_else(|| "()".into())
            };
            format!("{}: {}", p.name, ty)
        }).collect();

        let all_params = if params.is_empty() {
            "__device: Device".into()
        } else {
            format!("__device: Device, {}", params.join(", "))
        };

        self.line(&format!(
            "fn new({}) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {{",
            all_params
        ));
        self.indent += 1;

        self.emit_pipeline_init(name);

        let assigned: std::collections::HashSet<String> = init.body.iter()
            .filter_map(|s| match s {
                Stmt::Expr(e) => {
                    if let ExprKind::Assign(lhs, _) = &e.kind {
                        if let ExprKind::Var(n) = &lhs.kind { Some(n.clone()) }
                        else { None }
                    } else { None }
                }
                _ => None,
            }).collect();

        for stmt in &init.body {
            self.emit_init_stmt(stmt, fields);
        }

        for field in fields {
            if !assigned.contains(&field.name) {
                self.emit_field_default(field);
            }
        }

        self.emit_struct_literal(name, fields);
        self.indent -= 1;
        self.line("}");
    }

    fn emit_kernel_new_default(&mut self, name: &str, fields: &[KernelFieldDecl]) {
        self.line("fn new(__device: Device) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.emit_pipeline_init(name);
        for field in fields {
            self.emit_field_default(field);
        }
        self.emit_struct_literal(name, fields);
        self.indent -= 1;
        self.line("}");
    }

    fn emit_pipeline_init(&mut self, kernel_name: &str) {
        self.line("let __queue = __boring_metal_queue(&__device);");
        self.line(&format!(
            "let __pipeline = __boring_metal_pipeline(&__device, \"{}_kernel\")?;",
            kernel_name
        ));
    }

    fn emit_field_default(&mut self, field: &KernelFieldDecl) {
        match field.qual {
            GpuQual::Surface => {
                // Surface pixel buffer defaults to single-pixel placeholder (32-bit)
                match &field.ty {
                    Type::Array(_) | Type::ArrayN(_, _) => {
                        self.line(&format!(
                            "let {}: Buffer = __device.new_buffer(mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);",
                            field.name
                        ));
                    }
                    _ => {
                        let ty = rust_type(&field.ty);
                        let val = field.default.as_ref()
                            .map(emit_scalar_default)
                            .unwrap_or_else(|| "Default::default()".into());
                        self.line(&format!("let {}: {} = {};", field.name, ty, val));
                    }
                }
            }
            GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Const => {
                match &field.ty {
                    Type::Array(inner) | Type::ArrayN(inner, _) => {
                        let elem = rust_type(inner);
                        self.line(&format!(
                            "let {}: Buffer = __device.new_buffer(mem::size_of::<{}>() as u64, MTLResourceOptions::StorageModeShared);",
                            field.name, elem
                        ));
                    }
                    _ => {
                        let ty = rust_type(&field.ty);
                        let val = field.default.as_ref()
                            .map(emit_scalar_default)
                            .unwrap_or_else(|| "Default::default()".into());
                        self.line(&format!("let {}: {} = {};", field.name, ty, val));
                    }
                }
            }
            GpuQual::Sync => {}
            GpuQual::Local => {
                match &field.ty {
                    Type::Array(_) | Type::ArrayN(_, _) => {}
                    _ => {
                        let ty = rust_type(&field.ty);
                        let val = field.default.as_ref()
                            .map(emit_scalar_default)
                            .unwrap_or_else(|| "Default::default()".into());
                        self.line(&format!("let {}: {} = {};", field.name, ty, val));
                    }
                }
            }
        }
    }

    fn emit_struct_literal(&mut self, name: &str, fields: &[KernelFieldDecl]) {
        self.line(&format!("Ok({} {{", name));
        self.indent += 1;
        self.line("__device,");
        self.line("__queue,");
        self.line("__pipeline,");
        for field in fields {
            match field.qual {
                GpuQual::Sync => {}
                GpuQual::Local => match &field.ty {
                    Type::Array(_) | Type::ArrayN(_, _) => {}
                    _ => self.line(&format!("{},", field.name)),
                },
                _ => match &field.ty {
                    Type::Array(_) | Type::ArrayN(_, _) => self.line(&format!("{},", field.name)),
                    _ => self.line(&format!("{},", field.name)),
                }
            }
        }
        self.indent -= 1;
        self.line("})");
    }

    fn emit_init_stmt(&mut self, stmt: &Stmt, fields: &[KernelFieldDecl]) {
        match stmt {
            Stmt::Expr(e) => {
                if let ExprKind::Assign(lhs, rhs) = &e.kind {
                    if let ExprKind::Var(fname) = &lhs.kind {
                        if let Some(field) = fields.iter().find(|f| &f.name == fname) {
                            match field.qual {
                                GpuQual::Surface => {
                                    // Surface pixel buffer: always 32-bit (BGRA8Unorm = 4 bytes/pixel)
                                    match &rhs.kind {
                                        ExprKind::ArrayFill { value: _, count } | ExprKind::ArrayAlloc { count } => {
                                            let n = self.expr(count);
                                            self.line(&format!(
                                                "let {fname}: Buffer = __device.new_buffer(({n} as usize * mem::size_of::<u32>()) as u64, MTLResourceOptions::StorageModeShared);"
                                            ));
                                            return;
                                        }
                                        ExprKind::Array(elems) => {
                                            let lit: Vec<String> = elems.iter().map(|e| self.expr(e)).collect();
                                            self.line(&format!(
                                                "let __data_{fname}: Vec<u32> = vec![{}];", lit.join(", ")
                                            ));
                                            self.line(&format!(
                                                "let {fname}: Buffer = __device.new_buffer_with_data(__data_{fname}.as_ptr() as *const _, (__data_{fname}.len() * mem::size_of::<u32>()) as u64, MTLResourceOptions::StorageModeShared);"
                                            ));
                                            return;
                                        }
                                        _ => {
                                            let rhs_s = self.expr(rhs);
                                            self.line(&format!(
                                                "let {fname}: Buffer = __device.new_buffer_with_data({rhs_s}.as_ptr() as *const _, ({rhs_s}.len() * mem::size_of::<u32>()) as u64, MTLResourceOptions::StorageModeShared);"
                                            ));
                                            return;
                                        }
                                    }
                                }
                                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Const => {
                                    match &rhs.kind {
                                        ExprKind::ArrayFill { value: _, count } | ExprKind::ArrayAlloc { count } => {
                                            let n = self.expr(count);
                                            let elem = elem_rust_type(&field.ty);
                                            self.line(&format!(
                                                "let {fname}: Buffer = __device.new_buffer(({n} as usize * mem::size_of::<{elem}>()) as u64, MTLResourceOptions::StorageModeShared);"
                                            ));
                                            return;
                                        }
                                        ExprKind::Array(elems) => {
                                            let elem = elem_rust_type(&field.ty);
                                            let lit: Vec<String> = elems.iter().map(|e| self.expr(e)).collect();
                                            self.line(&format!(
                                                "let __data_{fname}: Vec<{elem}> = vec![{}];", lit.join(", ")
                                            ));
                                            self.line(&format!(
                                                "let {fname}: Buffer = __device.new_buffer_with_data(__data_{fname}.as_ptr() as *const _, (__data_{fname}.len() * mem::size_of::<{elem}>()) as u64, MTLResourceOptions::StorageModeShared);"
                                            ));
                                            return;
                                        }
                                        _ => {
                                            let rhs_s = self.expr(rhs);
                                            match &field.ty {
                                                Type::Array(_) | Type::ArrayN(_, _) => {
                                                    // A bare `field = param` assignment (this codebase's
                                                    // only real pattern here) means the constructor's
                                                    // OWN param type is already `Buffer` -- see
                                                    // `kernel_ctor_buffer_flags`, which the call site
                                                    // consults to decide the SAME thing when building
                                                    // the argument. Anything else (a computed
                                                    // expression) falls back to the old upload-from-
                                                    // Vec behavior, matching what `kernel_ctor_buffer_flags`
                                                    // would ALSO decide (false) for a non-bare-Var RHS.
                                                    if matches!(&rhs.kind, ExprKind::Var(_)) {
                                                        self.line(&format!("let {fname}: Buffer = {rhs_s};"));
                                                    } else {
                                                        let elem = elem_rust_type(&field.ty);
                                                        self.line(&format!(
                                                            "let {fname}: Buffer = __device.new_buffer_with_data({rhs_s}.as_ptr() as *const _, ({rhs_s}.len() * mem::size_of::<{elem}>()) as u64, MTLResourceOptions::StorageModeShared);"
                                                        ));
                                                    }
                                                }
                                                _ => {
                                                    // Scalar 'const field: store the value directly.
                                                    let ty = rust_type(&field.ty);
                                                    self.line(&format!("let {fname}: {ty} = {rhs_s};"));
                                                }
                                            }
                                            return;
                                        }
                                    }
                                }
                                GpuQual::Local | GpuQual::Sync => {
                                    // Scalar/struct field — emit as local variable declaration
                                    let rhs_s = self.expr(rhs);
                                    let ty = rust_type(&field.ty);
                                    self.line(&format!("let {fname}: {ty} = {rhs_s};"));
                                    return;
                                }
                            }
                        }
                    }
                }
                let s = self.expr(e);
                self.line(&format!("{};", s));
            }
            Stmt::Let(s) => {
                let binding = if s.binding.is_mutable() { "let mut" } else { "let" };
                if let Some(val) = &s.value {
                    let rhs = self.expr(val);
                    self.line(&format!("{} {} = {};", binding, s.name, rhs));
                }
            }
            _ => self.emit_stmt(stmt),
        }
    }

    fn emit_boring_launch(&mut self, _name: &str, fields: &[KernelFieldDecl]) {
        // Auto-grid when there is at least one device array field.
        let auto_grid_field: Option<String> = fields.iter().find_map(|f| {
            match f.qual {
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Surface => {
                    match &f.ty {
                        Type::Array(_) | Type::ArrayN(_, _) => Some(f.name.clone()),
                        _ => None,
                    }
                }
                _ => None,
            }
        });

        if auto_grid_field.is_some() {
            self.line(
                "fn __boring_launch(&mut self, block_dim: (u32,u32,u32), grid_dim: Option<(u32,u32,u32)>, _after: &[()]) \
                 -> Result<(), Box<dyn std::error::Error + Send + Sync>> {"
            );
        } else {
            self.line(
                "fn __boring_launch(&mut self, block_dim: (u32,u32,u32), grid_dim: (u32,u32,u32), _after: &[()]) \
                 -> Result<(), Box<dyn std::error::Error + Send + Sync>> {"
            );
        }
        self.indent += 1;

        // Grid sizing.
        if let Some(field) = &auto_grid_field {
            let field_qual = fields.iter().find(|f| &f.name == field).map(|f| &f.qual).cloned();
            let is_surface = matches!(field_qual, Some(GpuQual::Surface));
            let dim_field = if is_surface {
                fields.iter().find(|f| matches!(&f.ty, Type::Named(n) if n == "Dimension"))
                    .map(|f| f.name.clone())
            } else {
                None
            };
            self.line("let grid_dim = grid_dim.unwrap_or_else(|| {");
            self.indent += 1;
            if let Some(df) = dim_field {
                // 2D grid from surface Dimension field
                self.line(&format!("let __w = self.{}.width; let __h = self.{}.height;", df, df));
                self.line("((__w + block_dim.0 - 1) / block_dim.0, (__h + block_dim.1 - 1) / block_dim.1, 1)");
            } else if is_surface {
                // 1D fallback for surface without Dimension — use u32 element size
                self.line(&format!(
                    "let n = (self.{}.length() as usize / mem::size_of::<u32>()) as u32;",
                    field
                ));
                self.line("((n + block_dim.0 - 1) / block_dim.0, 1, 1)");
            } else {
                let elem = elem_rust_type(&fields.iter().find(|f| &f.name == field).unwrap().ty);
                self.line(&format!(
                    "let n = (self.{}.length() as usize / mem::size_of::<{}>()) as u32;",
                    field, elem
                ));
                self.line("((n + block_dim.0 - 1) / block_dim.0, 1, 1)");
            }
            self.indent -= 1;
            self.line("});");
        }

        // Dynamic shared memory size (per-block).
        let _dyn_shared_terms: Vec<String> = fields.iter()
            .filter(|f| matches!(f.qual, GpuQual::Sync))
            .filter_map(|f| {
                if let Type::Array(inner) = &f.ty {
                    let sz = elem_size_bytes(inner);
                    Some(format!("block_dim.0 as usize * {}", sz))
                } else {
                    None
                }
            })
            .collect();

        // Dispatch.
        self.line("let __cmd_buf = self.__queue.new_command_buffer();");
        self.line("let __encoder = __cmd_buf.new_compute_command_encoder();");
        self.line("__encoder.set_compute_pipeline_state(&self.__pipeline);");

        // Set buffers in the same order as MSL parameters.
        let mut buf_idx: u64 = 0;
        let mut tg_idx: u64 = 0;

        for f in fields {
            match f.qual {
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Surface => {
                    if matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) {
                        self.line(&format!("__encoder.set_buffer({}, Some(&self.{}), 0);", buf_idx, f.name));
                        buf_idx += 1;
                    }
                }
                _ => {}
            }
        }
        for f in fields {
            if matches!(f.qual, GpuQual::Const) {
                if matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) {
                    self.line(&format!("__encoder.set_buffer({}, Some(&self.{}), 0);", buf_idx, f.name));
                } else {
                    let elem = elem_rust_type(&f.ty);
                    self.line(&format!(
                        "__encoder.set_bytes({}, mem::size_of::<{}>() as u64, &self.{} as *const _ as *const _);",
                        buf_idx, elem, f.name
                    ));
                }
                buf_idx += 1;
            }
        }
        for f in fields {
            if matches!(f.qual, GpuQual::Local) && !matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) {
                let elem = rust_type(&f.ty);
                self.line(&format!(
                    "__encoder.set_bytes({}, mem::size_of::<{}>() as u64, &self.{} as *const _ as *const _);",
                    buf_idx, elem, f.name
                ));
                buf_idx += 1;
            }
        }
        for f in fields {
            if matches!(f.qual, GpuQual::Sync) && matches!(f.ty, Type::Array(_)) {
                if let Type::Array(inner) = &f.ty {
                    let sz = elem_size_bytes(inner);
                    self.line(&format!(
                        "__encoder.set_threadgroup_memory_length({}, (block_dim.0 as usize * {}) as u64);",
                        tg_idx, sz
                    ));
                    tg_idx += 1;
                }
            }
        }

        // Static shared memory doesn't need set_threadgroup_memory_length — declared in MSL.
        let _ = tg_idx;

        self.line("__encoder.dispatch_thread_groups(");
        self.indent += 1;
        self.line("MTLSize { width: grid_dim.0 as u64, height: grid_dim.1 as u64, depth: grid_dim.2 as u64 },");
        self.line("MTLSize { width: block_dim.0 as u64, height: block_dim.1 as u64, depth: block_dim.2 as u64 },");
        self.indent -= 1;
        self.line(");");
        self.line("__encoder.end_encoding();");
        self.line("__cmd_buf.commit();");
        self.line("__BORING_METAL_PENDING.with(|c| *c.borrow_mut() = Some(__cmd_buf.to_owned()));");
        self.line("Ok(())");
        self.indent -= 1;
        self.line("}");
    }

    // ── Regular items ──────────────────────────────────────────────────────────

    fn emit_fn(&mut self, f: &FnDecl, self_ty: Option<&str>) {
        if f.is_native { return; }
        let vis = if f.is_pub { "pub " } else { "" };
        let params: Vec<String> = f.params.iter().map(|p| {
            let name = if p.mutable { format!("mut {}", p.name) } else { p.name.clone() };
            match &p.ty {
                Some(ty) => {
                    // boring's by-ref contract for array/dict/set/struct/enum
                    // params -- see `cuda::host`'s identical fix for the full
                    // rationale (confirmed necessary for cross-calls between
                    // this emitter and the general-pipeline splice to type-check).
                    let base = if is_float_array_param(ty) {
                        "Vec<f64>".to_string()
                    } else {
                        rust_type(ty)
                    };
                    let rendered = if is_ref_worthy_type(ty, &self.struct_names) {
                        format!("&{}", base)
                    } else {
                        base
                    };
                    format!("{}: {}", name, rendered)
                }
                None => name,
            }
        }).collect();
        let all_params = match self_ty {
            Some(_) => {
                let s = if f.mutating { "&mut self" } else { "&self" };
                if params.is_empty() { s.to_string() } else { format!("{}, {}", s, params.join(", ")) }
            }
            None => params.join(", "),
        };
        // GPU-resident return type -- see `cuda::host`'s identical `resident_elem`
        // for the full rationale. Uses `general_host_elem_type` (always `f64` for
        // float), NOT this backend's own `elem_rust_type` (`f32`) -- see that
        // function's doc for why the two diverge here specifically (Metal's
        // native device-buffer width vs the general pass's fixed host convention).
        let resident_elem = f.return_ty.as_ref().and_then(|t| t.gpu_resident_qual().map(|_| general_host_elem_type(t)));
        let plain_ret = match &resident_elem {
            Some(elem) => format!("BoringGpuArg<{}>", elem),
            None => f.return_ty.as_ref().map(rust_type).unwrap_or_else(|| "()".into()),
        };
        // `throws` → `Result<T, Box<dyn std::error::Error + Send + Sync>>`. See
        // `cuda::host`'s identical change for the full rationale — previously
        // ignored entirely, leaving `throw`/`guard ... else throw` with nothing
        // but the plain (non-Result) return type to `return Err(...)` into.
        let ret = if f.throws {
            format!("Result<{}, Box<dyn std::error::Error + Send + Sync>>", plain_ret)
        } else {
            plain_ret
        };
        let sig = format!("{}fn {}({}) -> {}", vis, f.name, all_params, ret);
        if f.body.is_empty() {
            self.line(&format!("{} {{}}", sig));
            return;
        }
        self.line(&format!("{} {{", sig));
        self.indent += 1;
        let outer_in_throws = self.in_throws;
        self.in_throws = f.throws;
        let outer_in_resident_return = self.in_resident_return.take();
        self.in_resident_return = resident_elem.clone();
        let outer_ref_params = std::mem::take(&mut self.ref_params);
        let outer_f64_array_locals = std::mem::take(&mut self.f64_array_locals);
        let outer_resident_locals = std::mem::take(&mut self.resident_locals);
        for p in &f.params {
            if let Some(ty) = &p.ty {
                // A `[float]` param is declared `&Vec<f64>` (see the param-
                // rendering fix above) but immediately shadow-rebound to an
                // OWNED `Vec<f32>` local below -- it must NOT be tracked as a
                // by-ref var (`ref_params`), or later code would wrongly try to
                // `.clone()`/re-borrow a value that's already a plain owned Vec.
                if is_ref_worthy_type(ty, &self.struct_names) && !is_float_array_param(ty) {
                    self.ref_params.insert(p.name.clone());
                }
            }
        }
        // Shadow-rebind `[float]` params from the general pass's `&Vec<f64>`
        // convention to this backend's own native `Vec<f32>` (owned) -- see
        // `is_float_array_param`'s doc for the full rationale.
        for p in &f.params {
            if let Some(ty) = &p.ty {
                if is_float_array_param(ty) {
                    self.line(&format!(
                        "let {name} = {name}.iter().map(|&x| x as f32).collect::<Vec<f32>>();",
                        name = p.name
                    ));
                }
            }
        }
        let len = f.body.len();
        for (i, stmt) in f.body.iter().enumerate() {
            if i + 1 == len {
                if f.throws {
                    if let Stmt::Expr(e) = stmt {
                        // Converts this backend's own native `f32` buffer element
                        // to the `f64` every general-spliced caller expects (see
                        // `general_host_elem_type`'s doc) -- a no-op cast when the
                        // element type already matches (e.g. int arrays). Tries
                        // `try_resident_field_expr` FIRST, before emitting the
                        // materializing read at all -- see its doc comment.
                        let wrapped = if self.in_resident_return.is_some() {
                            if let Some(resident) = self.try_resident_field_expr(e) {
                                resident
                            } else {
                                let s = self.expr(e);
                                let elem = self.in_resident_return.clone().unwrap();
                                format!("BoringGpuArg::Host(({}).iter().map(|&x| x as {elem}).collect::<Vec<{elem}>>())", s)
                            }
                        } else {
                            self.expr(e)
                        };
                        self.line(&format!("Ok({})", wrapped));
                        continue;
                    }
                } else if let Some(elem) = self.in_resident_return.clone() {
                    if let Stmt::Expr(e) = stmt {
                        let wrapped = if let Some(resident) = self.try_resident_field_expr(e) {
                            resident
                        } else {
                            let s = self.expr(e);
                            format!("BoringGpuArg::Host(({}).iter().map(|&x| x as {elem}).collect::<Vec<{elem}>>())", s)
                        };
                        self.line(&wrapped);
                        continue;
                    }
                }
                self.emit_stmt_last(stmt);
            } else {
                self.emit_stmt(stmt);
            }
        }
        self.in_throws = outer_in_throws;
        self.in_resident_return = outer_in_resident_return;
        self.ref_params = outer_ref_params;
        self.f64_array_locals = outer_f64_array_locals;
        self.resident_locals = outer_resident_locals;
        self.indent -= 1;
        self.line("}");
    }

    fn emit_struct(&mut self, s: &StructDecl) {
        if s.is_native { return; }
        let vis = if s.is_pub { "pub " } else { "" };
        self.line(&format!("{}struct {} {{", vis, s.name));
        self.indent += 1;
        for f in &s.fields {
            let fvis = if f.is_pub { "pub " } else { "" };
            self.line(&format!("{}{}: {},", fvis, f.name, rust_type(&f.ty)));
        }
        self.indent -= 1;
        self.line("}");
        if !s.methods.is_empty() {
            self.blank();
            self.line(&format!("impl {} {{", s.name));
            self.indent += 1;
            for m in &s.methods { self.emit_fn(m, Some(&s.name)); self.blank(); }
            self.indent -= 1;
            self.line("}");
        }
    }

    fn emit_enum(&mut self, e: &EnumDecl) {
        if e.is_native { return; }
        let vis = if e.is_pub { "pub " } else { "" };
        self.line(&format!("{}enum {} {{", vis, e.name));
        self.indent += 1;
        for v in &e.variants {
            if v.fields.is_empty() {
                self.line(&format!("{},", v.name));
            } else {
                let fs: Vec<String> = v.fields.iter().map(|f| rust_type(&f.ty)).collect();
                self.line(&format!("{}({}),", v.name, fs.join(", ")));
            }
        }
        self.indent -= 1;
        self.line("}");
    }

    // ── Statements ─────────────────────────────────────────────────────────────

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(s) => {
                let binding = if s.binding.is_mutable() { "let mut" } else { "let" };
                // A `let k_t = transpose_gpu(...)` (or any other call to a
                // `fn_returns_resident` function) is materialized to
                // `general_host_elem_type`'s convention (`f64`), NOT this
                // backend's own native `rust_type` (`f32`) -- otherwise the
                // annotation here disagrees with what the materializing `match`
                // expression this function's own `Call` handling builds actually
                // produces, a real E0308 confirmed via a real cross-compile
                // `cargo check`.
                let is_materializing_call = matches!(&s.value.as_ref().map(|v| &v.kind), Some(ExprKind::Call(callee, _))
                    if matches!(&callee.kind, ExprKind::Var(n) if self.fn_returns_resident.contains_key(n.as_str())));
                // A materializing call whose OWN `let` is itself explicitly
                // `'gpu'unified` (e.g. `let [float]'gpu'unified k_t =
                // transpose_gpu(...)`) means the programmer wants this value
                // to stay resident for a later chained kernel-touching call --
                // see `resident_locals`'s doc. Keep it `BoringGpuArg<f64>`-
                // typed and suppress the eager materializing wrap this
                // specific call would otherwise get.
                let is_resident_preserving = is_materializing_call
                    && s.ty.as_ref().and_then(|t| t.gpu_resident_qual()).is_some();
                let ty_ann = if is_resident_preserving {
                    self.resident_locals.insert(s.name.clone());
                    s.ty.as_ref().map(|t| format!(": BoringGpuArg<{}>", general_host_elem_type(t))).unwrap_or_default()
                } else if is_materializing_call {
                    self.f64_array_locals.insert(s.name.clone());
                    s.ty.as_ref().map(|t| format!(": Vec<{}>", general_host_elem_type(t))).unwrap_or_default()
                } else {
                    s.ty.as_ref().map(|t| format!(": {}", rust_type(t))).unwrap_or_default()
                };
                if let Some(val) = &s.value {
                    self.track_kernel_var(&s.name, val);
                    self.track_dict_var(&s.name, s.ty.as_ref(), Some(val));
                    if is_resident_preserving { self.suppress_resident_materialize = true; }
                    let rhs = self.expr(val);
                    self.line(&format!("{} {}{} = {};", binding, s.name, ty_ann, rhs));
                } else {
                    self.track_dict_var(&s.name, s.ty.as_ref(), None);
                    self.line(&format!("{} {}{};", binding, s.name, ty_ann));
                }
            }
            // `let (a, b, c) = expr` — see `cuda::host`'s identical case for the
            // full rationale and scope limit.
            Stmt::LetDestructure(s) => {
                let binding = if s.binding.is_mutable() { "let mut" } else { "let" };
                let names: Vec<String> = s.bindings.iter().map(|b| b.name.clone()).collect();
                let rhs = self.expr(&s.value);
                self.line(&format!("{} ({}) = {};", binding, names.join(", "), rhs));
            }
            // Non-tail-position `match` — see `cuda::host`'s identical case.
            Stmt::Match(m) => {
                let s = self.emit_match_expr(m);
                self.line(&format!("{};", s));
            }
            Stmt::Expr(e) => {
                match &e.kind {
                    ExprKind::Assign(lhs, rhs) => {
                        if let ExprKind::Index(obj, idx) = &lhs.kind {
                            // `dict[key] = v` / `self.field[key] = v` → HashMap::insert.
                            // See `cuda::host`'s identical case for the full rationale.
                            if self.is_dict_obj(obj) {
                                let obj_s = self.expr(obj);
                                let idx_s = self.expr(idx);
                                let rhs_s = self.expr(rhs);
                                self.line(&format!("{}.insert(({}).clone(), ({}).clone());", obj_s, idx_s, rhs_s));
                                return;
                            }
                            // Plain array index assignment -- built directly rather
                            // than via `self.expr(lhs)`, which routes through the
                            // `Index` READ case (now appending `.clone()` for by-ref-
                            // safe reads). See `cuda::host`'s identical case.
                            let obj_s = self.expr(obj);
                            let idx_s = self.expr(idx);
                            let rhs_s = self.expr(rhs);
                            self.line(&format!("{}[({}) as usize] = {};", obj_s, idx_s, rhs_s));
                            return;
                        }
                        let l = self.expr(lhs);
                        let r = self.expr(rhs);
                        self.line(&format!("{} = {};", l, r));
                    }
                    _ => {
                        let s = self.expr(e);
                        self.line(&format!("{};", s));
                    }
                }
            }
            Stmt::Return(r) => {
                if let Some(val) = &r.value {
                    let s = match self.in_resident_return.clone() {
                        Some(_) if self.try_resident_field_expr(val).is_some() => {
                            self.try_resident_field_expr(val).unwrap()
                        }
                        Some(elem) => {
                            let s = self.expr(val);
                            format!("BoringGpuArg::Host(({}).iter().map(|&x| x as {elem}).collect::<Vec<{elem}>>())", s)
                        }
                        None => self.expr(val),
                    };
                    if self.in_throws { self.line(&format!("return Ok({});", s)); }
                    else { self.line(&format!("return {};", s)); }
                } else if self.in_throws {
                    self.line("return Ok(());");
                } else {
                    self.line("return;");
                }
            }
            // `guard <cond> else throw "..."` — see `cuda::host`'s identical case for
            // the full rationale and scope limit (only `GuardCond::Expr` is handled).
            Stmt::Guard(g) => {
                if let GuardCond::Expr(cond) = &g.cond {
                    let c = self.expr(cond);
                    self.line(&format!("if !({}) {{", c));
                    self.indent += 1;
                    for s in &g.else_body { self.emit_stmt(s); }
                    self.indent -= 1;
                    self.line("}");
                } else {
                    self.line("/* unsupported stmt */");
                }
            }
            // `throw "msg"` — see `cuda::host`'s identical case.
            Stmt::Throw(t) => {
                if self.in_throws {
                    let msg = t.value.as_ref().map(|v| self.expr(v)).unwrap_or_else(|| "\"error\"".into());
                    self.line(&format!("return Err(({}).into());", msg));
                } else {
                    let msg = t.value.as_ref().map(|v| self.expr(v)).unwrap_or_else(|| "\"error\"".into());
                    self.line(&format!("panic!(\"{{}}\", {});", msg));
                }
            }
            Stmt::If(i) => {
                for (idx, (cond, body)) in i.branches.iter().enumerate() {
                    let c = self.expr(cond);
                    if idx == 0 { self.line(&format!("if {} {{", c)); }
                    else        { self.line(&format!("}} else if {} {{", c)); }
                    self.indent += 1;
                    for s in body { self.emit_stmt(s); }
                    self.indent -= 1;
                }
                if let Some(else_body) = &i.else_body {
                    self.line("} else {");
                    self.indent += 1;
                    for s in else_body { self.emit_stmt(s); }
                    self.indent -= 1;
                }
                self.line("}");
            }
            Stmt::While(w) => {
                let cond = self.expr(&w.condition);
                self.line(&format!("while {} {{", cond));
                self.indent += 1;
                for s in &w.body { self.emit_stmt(s); }
                self.indent -= 1;
                self.line("}");
            }
            Stmt::For(f) => {
                let is_enumerate = f.vars.len() >= 2;
                let var = if is_enumerate {
                    format!("({}, {})", f.vars[0], f.vars[1])
                } else {
                    f.vars.first().cloned().unwrap_or_else(|| "_i".into())
                };
                match &f.iterable.kind {
                    ExprKind::Range { start, end, inclusive } => {
                        let lo = self.expr(start);
                        let hi = self.expr(end);
                        let range = if *inclusive { format!("{}..={}", lo, hi) } else { format!("{}..{}", lo, hi) };
                        self.line(&format!("for {} in {} {{", var, range));
                    }
                    _ => {
                        let iter = self.expr(&f.iterable);
                        // `.iter().cloned()` regardless of owned/by-ref -- see
                        // `cuda::host`'s identical fix for the full rationale.
                        if is_enumerate {
                            self.line(&format!("for {} in {}.iter().cloned().enumerate() {{", var, iter));
                        } else {
                            self.line(&format!("for {} in {}.iter().cloned() {{", var, iter));
                        }
                    }
                }
                self.indent += 1;
                for s in &f.body { self.emit_stmt(s); }
                self.indent -= 1;
                self.line("}");
            }
            Stmt::Break(_label, _val) => self.line("break;"),
            Stmt::Continue(_label)    => self.line("continue;"),
            Stmt::Comment(_)          => {}
            Stmt::KernelBlock(kb) => self.emit_kernel_block(&kb.body),
            _ => { self.line("/* unsupported stmt */"); }
        }
    }

    fn emit_kernel_block(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            match stmt {
                Stmt::Expr(e) => {
                    if let Some(launch) = self.try_emit_kernel_launch_call(e) {
                        self.line(&format!("{launch};"));
                    } else {
                        let s = self.expr(e);
                        self.line(&format!("{s};"));
                    }
                }
                Stmt::Loop(l) => {
                    if self.screen_var.is_some() {
                        let body = l.body.clone();
                        self.emit_render_loop(&body);
                    } else {
                        self.line("loop {");
                        self.indent += 1;
                        for s in &l.body { self.emit_stmt(s); }
                        self.indent -= 1;
                        self.line("}");
                    }
                }
                other => self.emit_stmt(other),
            }
        }
    }

    fn try_emit_kernel_launch_call(&mut self, expr: &Expr) -> Option<String> {
        let ExprKind::Call(callee, args) = &expr.kind else { return None; };
        let ExprKind::Var(var_name) = &callee.kind else { return None; };
        let has_block = args.iter().any(|a| a.label.as_deref() == Some("block"));
        if !has_block { return None; }
        let is_kernel = self.var_kernel_type.contains_key(var_name.as_str())
            || self.kernel_names.contains(var_name.as_str());
        if !is_kernel { return None; }
        let kernel_type = self.var_kernel_type.get(var_name.as_str()).cloned()
            .or_else(|| Some(var_name.clone()));
        let auto_grid = kernel_type
            .as_ref()
            .and_then(|t| self.kernel_decls.get(t))
            .map(|decl| decl.fields.iter().any(|f|
                matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Surface)
                && matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _))))
            .unwrap_or(false);
        let block = args.iter().find(|a| a.label.as_deref() == Some("block"))
            .map(|a| self.dim3_expr(&a.value))
            .unwrap_or_else(|| "(1, 1, 1)".into());
        let grid: String = if let Some(g) = args.iter().find(|a| a.label.as_deref() == Some("grid")) {
            if auto_grid { format!("Some({})", self.dim3_expr(&g.value)) }
            else { self.dim3_expr(&g.value) }
        } else if auto_grid { "None".into() } else { "(1, 1, 1)".into() };
        // Metal is synchronous (&mut self, no move needed — wait_until_completed inside).
        // In render loop closures, ? is unavailable — use expect() instead.
        let launch = if self.in_render_loop {
            format!("{var_name}.__boring_launch({block}, {grid}, &[]).expect(\"kernel launch failed\")")
        } else {
            format!("{var_name}.__boring_launch({block}, {grid}, &[])?")
        };
        Some(launch)
    }

    fn emit_stmt_last(&mut self, stmt: &Stmt) {
        if let Stmt::Expr(e) = stmt {
            let s = self.expr(e);
            self.line(&s);
        } else if let Stmt::Match(m) = stmt {
            // No trailing `;` — see `cuda::host`'s identical case.
            let s = self.emit_match_expr(m);
            self.line(&s);
        } else {
            self.emit_stmt(stmt);
        }
    }

    /// `match <subject>: <arm> ...`. See `cuda::host`'s identical function for
    /// the full rationale and scope this covers.
    fn emit_match_expr(&mut self, m: &MatchStmt) -> String {
        let needs_as_str = m.arms.iter().any(|a| a.patterns.iter().any(|p| matches!(p, Pattern::Lit(LitPattern::Str(_)))));
        let subj = self.expr(&m.subject);
        let subj = if needs_as_str { format!("{}.as_str()", subj) } else { subj };
        let mut out = format!("match {} {{\n", subj);
        for arm in &m.arms {
            let pats: Vec<String> = arm.patterns.iter().map(|p| self.emit_pattern(p)).collect();
            let guard_s = arm.guard.as_ref().map(|g| format!(" if {}", self.expr(g))).unwrap_or_default();
            let body_s = match &arm.body {
                MatchBody::Expr(e) => self.expr(e),
                MatchBody::Block(stmts) => self.emit_sub_block(stmts),
            };
            out.push_str(&format!("    {}{} => {{ {} }}\n", pats.join(" | "), guard_s, body_s));
        }
        out.push('}');
        out
    }

    /// See `cuda::host`'s identical function.
    fn emit_pattern(&mut self, p: &Pattern) -> String {
        match p {
            Pattern::Wildcard => "_".into(),
            Pattern::Bind(name) => name.clone(),
            Pattern::Lit(lit) => match lit {
                LitPattern::Int(n) => n.to_string(),
                LitPattern::Float(f) => format!("{}", f),
                LitPattern::Str(s) => format!("\"{}\"", s),
                LitPattern::Bool(b) => b.to_string(),
                LitPattern::Nil => "None".into(),
            },
            Pattern::None => "None".into(),
            Pattern::Some(inner) => format!("Some({})", self.emit_pattern(inner)),
            Pattern::Tuple(elems) => {
                let s: Vec<String> = elems.iter().map(|e| self.emit_pattern(e)).collect();
                format!("({})", s.join(", "))
            }
            Pattern::Variant(name, subs) => {
                let qualified = self.variant_to_enum.get(name.as_str())
                    .map(|e| format!("{}::{}", e, name))
                    .unwrap_or_else(|| name.clone());
                if subs.is_empty() {
                    qualified
                } else {
                    let s: Vec<String> = subs.iter().map(|sp| self.emit_pattern(sp)).collect();
                    format!("{}({})", qualified, s.join(", "))
                }
            }
        }
    }

    /// Emit `stmts` through a fresh sub-emitter sharing this one's tracking
    /// state, returning the rendered body as a Rust block-expression string.
    /// See `cuda::host`'s identical function.
    fn emit_sub_block(&mut self, stmts: &[Stmt]) -> String {
        let mut sub = HostEmitter {
            out: String::new(),
            indent: 0,
            kernel_names: self.kernel_names.clone(),
            kernel_decls: self.kernel_decls.clone(),
            var_kernel_type: self.var_kernel_type.clone(),
            gpu_vars: self.gpu_vars.clone(),
            top_level_scalars: self.top_level_scalars.clone(),
            screen_var: self.screen_var.clone(),
            screen_width_expr: self.screen_width_expr.clone(),
            screen_height_expr: self.screen_height_expr.clone(),
            screen_title: self.screen_title.clone(),
            in_render_loop: self.in_render_loop,
            dict_vars: self.dict_vars.clone(),
            dict_fields: self.dict_fields.clone(),
            fn_throws: self.fn_throws.clone(),
            in_throws: self.in_throws,
            variant_to_enum: self.variant_to_enum.clone(),
            fn_ref_params: self.fn_ref_params.clone(),
            struct_names: self.struct_names.clone(),
            ref_params: self.ref_params.clone(),
            in_resident_return: self.in_resident_return.clone(),
            fn_returns_resident: self.fn_returns_resident.clone(),
            fn_float_array_params: self.fn_float_array_params.clone(),
            f64_array_locals: self.f64_array_locals.clone(),
            resident_locals: self.resident_locals.clone(),
            suppress_resident_materialize: self.suppress_resident_materialize,
        };
        let last = stmts.len().saturating_sub(1);
        for (i, st) in stmts.iter().enumerate() {
            if i == last { sub.emit_stmt_last(st); } else { sub.emit_stmt(st); }
        }
        format!("{{ {} }}", sub.out.trim())
    }

    // ── Expressions ────────────────────────────────────────────────────────────

    fn expr(&mut self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Int(n)    => n.to_string(),
            ExprKind::Float(f)  => {
                let s = format!("{}", f);
                if s.contains('.') || s.contains('e') { s } else { format!("{}.0", s) }
            }
            ExprKind::Bool(b)   => if *b { "true".into() } else { "false".into() },
            ExprKind::Str(s)    => format!("\"{}\"", s),
            ExprKind::Nil       => "None".into(),
            ExprKind::Void      => "()".into(),
            ExprKind::Var(name) => {
                // Inline top-level scalars when referenced inside kernel new()
                // where they are not in scope as Rust local variables.
                self.top_level_scalars.get(name).cloned().unwrap_or_else(|| name.clone())
            }

            ExprKind::BinOp(op, lhs, rhs) => {
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                format!("({} {} {})", l, binop_rust(op), r)
            }
            ExprKind::UnaryOp(op, operand) => {
                let v = self.expr(operand);
                format!("({}{})", unaryop_rust(op), v)
            }
            ExprKind::Assign(lhs, rhs) => {
                format!("({} = {})", self.expr(lhs), self.expr(rhs))
            }
            ExprKind::Index(arr, idx) => {
                if let ExprKind::Field(obj, field) = &arr.kind {
                    if let ExprKind::Var(obj_name) = &obj.kind {
                        if let Some(read_call) = self.try_gpu_field_read(obj_name, field) {
                            let i = self.expr(idx);
                            return format!("{}[{} as usize]", read_call, i);
                        }
                    }
                }
                // Dict-typed receiver (`vocab[key]`, `self.vocab[key]`) → HashMap::get,
                // not Vec-style index. See `cuda::host`'s identical case for the full
                // rationale (this is the "Metal only, worse" dict bug this fixes).
                if self.is_dict_obj(arr) {
                    let obj_s = self.expr(arr);
                    let key_s = self.expr(idx);
                    return format!("{}.get(&({})).cloned()", obj_s, key_s);
                }
                // Slice: a[M..N] / a[..N] / a[M..] / a[..] -- a proper Rust range index
                // returning an owned Vec. See `cuda::host`'s identical case for the full
                // rationale (this is the `layer_norm_seq` slice-indexing bug this fixes).
                if let ExprKind::SliceRange { start, end, inclusive } = &idx.kind {
                    let obj_s = self.expr(arr);
                    let start_s = start.as_deref().map(|e| format!("({}) as usize", self.expr(e)));
                    let end_s   = end.as_deref().map(|e| format!("({}) as usize", self.expr(e)));
                    let dots = if *inclusive { "..=" } else { ".." };
                    let range_s = match (start_s, end_s) {
                        (Some(s), Some(e)) => format!("{s}{dots}{e}"),
                        (Some(s), None)    => format!("{s}.."),
                        (None, Some(e))    => format!("{dots}{e}"),
                        (None, None)       => "..".to_string(),
                    };
                    return format!("{}[{}].to_vec()", obj_s, range_s);
                }
                // Plain read: `.clone()` unconditionally after indexing -- see
                // `cuda::host`'s identical case for the full rationale.
                // `Stmt::Expr(Assign(..))` bypasses this case entirely for an
                // assignment TARGET.
                format!("{}[{} as usize].clone()", self.expr(arr), self.expr(idx))
            }
            ExprKind::Field(obj, field) => {
                if let ExprKind::Var(obj_name) = &obj.kind {
                    // screen.X → boring_screen_* variables
                    if self.screen_var.as_deref() == Some(obj_name.as_str()) {
                        return match field.as_str() {
                            "frame"     => "boring_frame".into(),
                            "time"      => "boring_start.elapsed().as_secs_f64()".into(),
                            "closed"    => "boring_screen_closed".into(),
                            "resized"   => "boring_screen_resized".into(),
                            "width"     => "boring_screen_width".into(),
                            "height"    => "boring_screen_height".into(),
                            "dimension" => "Dimension(boring_screen_width as u32, boring_screen_height as u32)".into(),
                            other       => format!("boring_screen_{}", other),
                        };
                    }
                    if let Some(read_call) = self.try_gpu_field_read(obj_name, field) {
                        return read_call;
                    }
                    // `EnumName.Variant` → Rust's path-qualified `EnumName::Variant`.
                    // See `cuda::host`'s identical case for the full rationale.
                    if self.variant_to_enum.get(field.as_str()) == Some(obj_name) {
                        return format!("{}::{}", obj_name, field);
                    }
                }
                let o = self.expr(obj);
                // `.length` as a field (Boring style) → Rust `.len() as isize`
                if field == "length" || field == "count" {
                    return format!("{}.len() as isize", o);
                }
                format!("{}.{}", o, field)
            }
            ExprKind::Call(callee, args) => {
                // Consumed for THIS call only -- taken immediately, before any
                // nested `self.expr()` recursion into `args`, so a resident-
                // preserving call passed as an argument to this outer call
                // isn't incorrectly suppressed too. See its field doc.
                let __suppress_materialize = std::mem::take(&mut self.suppress_resident_materialize);
                // `GPU(n)` → `boring_metal_device_n(n)?`
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "GPU" {
                        let idx = args.first().map(|a| self.expr(&a.value)).unwrap_or_else(|| "0".into());
                        return format!("boring_metal_device_n({} as usize)?", idx);
                    }
                }
                // `float(expr)` → `expr as f32` (Metal uses f32, not f64)
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "float" {
                        if let Some(arg) = args.first() {
                            let inner = self.expr(&arg.value);
                            return format!("({} as f32)", inner);
                        }
                    }
                }
                // `int(expr)` → `expr as isize`
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "int" {
                        if let Some(arg) = args.first() {
                            let inner = self.expr(&arg.value);
                            return format!("({} as isize)", inner);
                        }
                    }
                }
                // `uint(expr)` → `expr as usize`
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "uint" {
                        if let Some(arg) = args.first() {
                            let inner = self.expr(&arg.value);
                            return format!("({} as usize)", inner);
                        }
                    }
                }
                // `ord(c)` — char/string → isize (Boring built-in)
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "ord" {
                        if let Some(arg) = args.first() {
                            let inner = self.expr(&arg.value);
                            return format!("({} as isize)", inner);
                        }
                    }
                }
                // `Scale(data)` → `Scale::new(boring_metal_device(), data)?`. Kernel
                // constructors always need OWNED args -- if an argument is one of
                // THIS function's own by-ref params, clone it back to owned first
                // (see `cuda::host`'s identical case).
                if let ExprKind::Var(name) = &callee.kind {
                    if self.kernel_names.contains(name.as_str()) {
                        let dev = if self.screen_var.is_some() {
                            "boring_device.clone()".to_string()
                        } else {
                            "boring_metal_device()".to_string()
                        };
                        let args_s = self.emit_kernel_ctor_args(name, &dev, args);
                        let all = std::iter::once(dev).chain(args_s).collect::<Vec<_>>();
                        return format!("{}::new({})?", name, all.join(", "));
                    }
                }
                // `print(expr)` → `println!("{}", expr)` or `println!("{}", format!(...))` → `println!(...)`
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "print" {
                        if let Some(arg) = args.first() {
                            let inner = self.expr(&arg.value);
                            // If the argument is already a format! call, lift it to println!
                            if inner.starts_with("format!(") {
                                let inner_args = &inner["format!(".len()..inner.len()-1];
                                return format!("println!({})", inner_args);
                            }
                            return format!("println!(\"{{}}\", {})", inner);
                        }
                        return "println!()".into();
                    }
                }
                // Ordinary function call -- coerce each argument to match the
                // callee's own by-ref/owned parameter convention. See
                // `cuda::host`'s identical case for the full rationale.
                let callee_name = if let ExprKind::Var(name) = &callee.kind { Some(name.as_str()) } else { None };
                let ref_flags = callee_name.and_then(|n| self.fn_ref_params.get(n)).cloned();
                let float_flags = callee_name.and_then(|n| self.fn_float_array_params.get(n)).cloned();
                let args_s: Vec<String> = args.iter().enumerate().map(|(i, a)| {
                    // A `[float]` position on the callee expects `&Vec<f64>` (see
                    // `is_float_array_param`'s doc), but every value flowing
                    // through this backend's own kernel-touching functions is
                    // this backend's native OWNED `Vec<f32>` (params were
                    // shadow-rebound to it, see `emit_fn`) -- cast up and wrap
                    // in `&(...)` directly, bypassing `coerce_call_arg`'s
                    // ref/owned logic (which doesn't know about this width
                    // conversion).
                    if float_flags.as_ref().and_then(|f| f.get(i).copied()).unwrap_or(false) {
                        let s = self.expr(&a.value);
                        return format!("&(({}).iter().map(|&x| x as f64).collect::<Vec<f64>>())", s);
                    }
                    let expects_ref = ref_flags.as_ref().and_then(|f| f.get(i).copied()).unwrap_or(false);
                    self.coerce_call_arg(&a.value, expects_ref)
                }).collect();
                let fn_s = self.expr(callee);
                let call = format!("{}({})", fn_s, args_s.join(", "));
                // `?`-propagate a call to another free `throws` function. See
                // `cuda::host`'s identical case for the scope limit (does not cover
                // `throws` struct methods calling each other).
                let call = if self.in_throws && callee_name.is_some_and(|n| self.fn_throws.contains(n)) {
                    format!("{}?", call)
                } else {
                    call
                };
                // Interprocedural GPU residency: a call from one kernel-touching
                // function to another that returns `BoringGpuArg<T>` must
                // materialize the result -- UNLESS the caller explicitly asked
                // to keep it resident (`__suppress_materialize`, set by
                // `Stmt::Let` for a `'gpu'unified`-typed binding; see
                // `resident_locals`'s doc), in which case the plain `call`
                // (already typed `BoringGpuArg<T>` by `emit_fn`) is returned
                // as-is. See `cuda::host`'s identical case for the unconditional
                // (non-suppressible) version of this same rule.
                if __suppress_materialize {
                    return call;
                }
                if let Some(ret_ty) = callee_name.and_then(|n| self.fn_returns_resident.get(n)).cloned() {
                    // `general_host_elem_type` (not this backend's own
                    // `elem_rust_type`) -- must match `BoringGpuArg<T>`'s actual
                    // `T` (see `emit_fn`'s `resident_elem`), which is always the
                    // general pass's `f64` convention, not this backend's native
                    // `f32`.
                    let elem = general_host_elem_type(&ret_ty);
                    return format!(
                        "match {call} {{ BoringGpuArg::Resident(buf, _) => __boring_gpu_copy_d2h::<f32>(&__boring_gpu_device(), &__boring_gpu_queue(), &buf).iter().map(|&x| x as {elem}).collect::<Vec<{elem}>>(), BoringGpuArg::Host(v) => v }}"
                    );
                }
                call
            }
            ExprKind::MethodCall(obj, method, args) => {
                // screen.key("q") → boring_keys.contains("q")
                if let ExprKind::Var(name) = &obj.kind {
                    if self.screen_var.as_deref() == Some(name.as_str()) {
                        match method.as_str() {
                            "key" => {
                                let k = args.first().map(|a| self.expr(&a.value)).unwrap_or_else(|| "\"\"".into());
                                return format!("boring_keys.contains({})", k);
                            }
                            "key_pressed" => {
                                let k = args.first().map(|a| self.expr(&a.value)).unwrap_or_else(|| "\"\"".into());
                                return format!("boring_keys.contains({})", k);
                            }
                            _ => {}
                        }
                    }
                }
                // `GPU.all()` → Device::all()
                if let ExprKind::Var(name) = &obj.kind {
                    if name == "GPU" && method == "all" {
                        return "Device::all().into_iter().enumerate().map(|(i, d)| (i, d)).collect::<Vec<_>>()".into();
                    }
                    if self.gpu_vars.contains(name.as_str()) {
                        return self.emit_gpu_property(name, method);
                    }
                    // `fs.writeBytes(path, bytes)` — write Vec<isize> as binary file
                    if name == "fs" && method == "writeBytes" {
                        let path  = args.first().map(|a| self.expr(&a.value)).unwrap_or_default();
                        let bytes = args.get(1).map(|a| self.expr(&a.value)).unwrap_or_default();
                        return format!("std::fs::write({}, {}.iter().map(|&b| b as u8).collect::<Vec<u8>>())?", path, bytes);
                    }
                    // `fs.write(path, text)` — write string as text file
                    if name == "fs" && (method == "write" || method == "writeText") {
                        let path = args.first().map(|a| self.expr(&a.value)).unwrap_or_default();
                        let text = args.get(1).map(|a| self.expr(&a.value)).unwrap_or_default();
                        return format!("std::fs::write({}, {}.as_bytes())?", path, text);
                    }
                }
                let o = self.expr(obj);
                let args_s: Vec<String> = args.iter().map(|a| self.expr(&a.value)).collect();
                match method.as_str() {
                    "wait" => format!("{}.wait()", o),
                    "done" => format!("{}.done()", o),
                    // `.chars()` — collect to Vec<char> so indexing and .len() work
                    "chars" if args.is_empty() => format!("{}.chars().collect::<Vec<char>>()", o),
                    // `.length` / `.count` — map to .len() as isize
                    "length" | "count" if args.is_empty() => format!("{}.len() as isize", o),
                    // `.add(x)` / `.insert(x)` — Vec push
                    "add" | "insert" if args.len() == 1 => format!("{}.push({})", o, args_s[0]),
                    // `.map(closure)` on an array isn't a real `Vec` method — go through
                    // iter/cloned/collect, matching the general (std/wgpu) transpiler's
                    // array-method fallback. `args_s[0]` is already a rendered `|v| ...`
                    // closure (see the `ExprKind::Closure` case below).
                    "map" if args_s.len() == 1 && matches!(&args[0].value.kind, ExprKind::Closure(..)) =>
                        format!("{}.iter().cloned().map({}).collect::<Vec<_>>()", o, args_s[0]),
                    // `.sum()` — Metal floats are always f32 (see `rust_type`'s `Float`
                    // case), unlike CUDA/the general pipeline's f64 default.
                    "sum" if args.is_empty() => format!("{}.iter().cloned().sum::<f32>()", o),
                    _ => format!("{}.{}({})", o, method, args_s.join(", ")),
                }
            }
            ExprKind::Pipe(lhs, method, args) => {
                let l = self.expr(lhs);
                // `args_s` was previously computed from `_args` and then unconditionally
                // dropped by every arm below (the "worse" Metal bug this fixes: a piped
                // `.map(closure)` emitted as bare `x.map()`, silently discarding the
                // lambda argument entirely).
                let args_s: Vec<String> = args.iter().map(|a| self.expr(&a.value)).collect();
                match method.as_str() {
                    // Metal __boring_launch now returns Result<(), _> — no .wait() needed
                    "wait" => l,
                    "done" => "true".into(),
                    // `x |> map((v): ...)` — see the identical `MethodCall` case above.
                    "map" if args_s.len() == 1 && matches!(&args[0].value.kind, ExprKind::Closure(..)) =>
                        format!("{}.iter().cloned().map({}).collect::<Vec<_>>()", l, args_s[0]),
                    // `x |> sum()` — see the identical `MethodCall` case above.
                    "sum" if args.is_empty() => format!("{}.iter().cloned().sum::<f32>()", l),
                    _ => format!("{}.{}({})", l, method, args_s.join(", ")),
                }
            }
            ExprKind::KernelLaunch { config, kernel } => {
                let auto_grid = self.resolve_kernel_type(kernel)
                    .and_then(|t| self.kernel_decls.get(&t))
                    .map(|decl| decl.fields.iter().any(|f|
                        matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Surface)
                        && matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _))))
                    .unwrap_or(false);
                let k = self.expr(kernel);
                let block = config.block.as_ref()
                    .map(|e| self.dim3_expr(e))
                    .unwrap_or_else(|| "(1, 1, 1)".into());
                let grid = if auto_grid {
                    match &config.grid {
                        None => "None".into(),
                        Some(e) => format!("Some({})", self.dim3_expr(e)),
                    }
                } else {
                    config.grid.as_ref()
                        .map(|e| self.dim3_expr(e))
                        .unwrap_or_else(|| "(1, 1, 1)".into())
                };
                // `after =` — Metal is synchronous (&mut self, wait_until_completed inside).
                if self.in_render_loop {
                    format!("{k}.__boring_launch({block}, {grid}, &[]).expect(\"kernel launch failed\")")
                } else {
                    format!("{k}.__boring_launch({block}, {grid}, &[])?")
                }
            }
            ExprKind::New { arena, ctor } => {
                // args go through the same buffer-upload handling as a plain
                // `Scale(data)` call -- see `emit_kernel_ctor_args`'s doc.
                if let ExprKind::Call(callee, args) = &ctor.kind {
                    if let ExprKind::Var(name) = &callee.kind {
                        if self.kernel_names.contains(name.as_str()) {
                            let default_dev = if self.screen_var.is_some() {
                                "boring_device.clone()".to_string()
                            } else {
                                "boring_metal_device()".to_string()
                            };
                            let dev = arena.as_ref()
                                .map(|a| self.expr(a))
                                .unwrap_or(default_dev);
                            let args_s = self.emit_kernel_ctor_args(name, &dev, args);
                            let all = std::iter::once(dev).chain(args_s).collect::<Vec<_>>();
                            return format!("{}::new({})?", name, all.join(", "));
                        }
                    }
                }
                self.expr(ctor)
            }
            ExprKind::Array(elems) => {
                let s: Vec<String> = elems.iter().map(|e| self.expr(e)).collect();
                format!("vec![{}]", s.join(", "))
            }
            ExprKind::ArrayFill { value, count } => {
                let v = self.expr(value);
                let n = self.expr(count);
                format!("vec![{}; {} as usize]", v, n)
            }
            ExprKind::ArrayAlloc { count } => {
                let n = self.expr(count);
                format!("vec![Default::default(); {} as usize]", n)
            }
            ExprKind::Tuple(elems) => {
                let s: Vec<String> = elems.iter().map(|e| self.expr(e)).collect();
                format!("({})", s.join(", "))
            }
            // `{=}` / `{"k" = v, ...}` — was entirely unhandled here (unlike
            // `cuda::host`, which already has this case), falling to this
            // function's catch-all `/* expr */` default. Part of the "Metal only,
            // worse" dict bug this fixes (tokenizer.br's `var {string=int} vocab = {=}`).
            ExprKind::Dict(pairs) => {
                let s: Vec<String> = pairs.iter().map(|(k, v)| {
                    format!("({}, {})", self.expr(k), self.expr(v))
                }).collect();
                format!("[{}].into_iter().collect::<std::collections::HashMap<_,_>>()", s.join(", "))
            }
            // A boring lambda, e.g. `(v): gelu_f(v)` — used as a `.map()`/pipe argument
            // (see the `map`/`sum` cases above). Was previously unhandled, falling to
            // this function's catch-all `/* expr */` default.
            ExprKind::Closure(params, _ret_ty, body, _throws, _task) => {
                let ps: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                let body_s = match body {
                    ClosureBody::Expr(be) => self.expr(be),
                    ClosureBody::Block(stmts) => self.emit_sub_block(stmts),
                };
                format!("|{}| {}", ps.join(", "), body_s)
            }
            ExprKind::Range { start, end, inclusive } => {
                let lo = self.expr(start);
                let hi = self.expr(end);
                if *inclusive { format!("({}..={})", lo, hi) } else { format!("({}..{})", lo, hi) }
            }
            ExprKind::Cast(inner, ty) => {
                format!("({} as {})", self.expr(inner), rust_type(ty))
            }
            ExprKind::Else(val, default) => {
                format!("{}.unwrap_or({})", self.expr(val), self.expr(default))
            }
            ExprKind::If(i) => {
                let mut out = String::new();
                for (idx, (cond, body)) in i.branches.iter().enumerate() {
                    let c = self.expr(cond);
                    let t = body.last().and_then(|s| {
                        if let Stmt::Expr(e) = s { Some(self.expr(e)) } else { None }
                    }).unwrap_or_else(|| "()".into());
                    if idx == 0 { out.push_str(&format!("if {} {{ {} }}", c, t)); }
                    else        { out.push_str(&format!(" else if {} {{ {} }}", c, t)); }
                }
                if let Some(else_body) = &i.else_body {
                    let e = else_body.last().and_then(|s| {
                        if let Stmt::Expr(e) = s { Some(self.expr(e)) } else { None }
                    }).unwrap_or_else(|| "()".into());
                    out.push_str(&format!(" else {{ {} }}", e));
                }
                out
            }
            ExprKind::StringInterp(segs) => {
                let mut parts = Vec::new();
                let mut fmt_str = String::from("\"");
                let mut fargs = Vec::new();
                for seg in segs {
                    match seg {
                        StringSegment::Lit(s) => fmt_str.push_str(s),
                        StringSegment::Expr(e) => {
                            fmt_str.push_str("{}");
                            fargs.push(self.expr(e));
                        }
                        StringSegment::FormattedExpr(e, fmt) => {
                            let rust_fmt = fmt.trim_end_matches(['f', 'd', 's', 'g', 'G']);
                            fmt_str.push_str(&format!("{{:{}}}", rust_fmt));
                            fargs.push(self.expr(e));
                        }
                    }
                }
                fmt_str.push('"');
                parts.push(fmt_str);
                parts.extend(fargs);
                format!("format!({})", parts.join(", "))
            }
            ExprKind::ArrayComp { expr, var, count } => {
                let n = self.expr(count);
                let body = self.expr(expr);
                format!(
                    "(0..({} as usize)).map(|__boring_i| {{ let {} = __boring_i as isize; {} }}).collect::<Vec<_>>()",
                    n, var, body
                )
            }
            ExprKind::ArrayCompIter { expr, var, iter } => {
                let it = self.expr(iter);
                let body = self.expr(expr);
                format!("{}.iter().map(|{}| {{ {} }}).collect::<Vec<_>>()", it, var, body)
            }
            _ => "/* expr */".into(),
        }
    }

    fn dim3_expr(&mut self, e: &Expr) -> String {
        if let ExprKind::Tuple(elems) = &e.kind {
            let x = elems.first().map(|e| format!("{} as u32", self.expr(e))).unwrap_or_else(|| "1".into());
            let y = elems.get(1).map(|e| format!("{} as u32", self.expr(e))).unwrap_or_else(|| "1".into());
            let z = elems.get(2).map(|e| format!("{} as u32", self.expr(e))).unwrap_or_else(|| "1".into());
            format!("({x}, {y}, {z})")
        } else {
            let v = self.expr(e);
            format!("({v} as u32, 1, 1)")
        }
    }

    fn track_kernel_var(&mut self, name: &str, val: &Expr) {
        if let Some(t) = self.resolve_kernel_type(val) {
            self.var_kernel_type.insert(name.to_string(), t);
        }
        if self.is_gpu_expr(val) {
            self.gpu_vars.insert(name.to_string());
        }
    }

    fn is_gpu_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(n) = &callee.kind { n == "GPU" } else { false }
            }
            ExprKind::Var(n) => self.gpu_vars.contains(n.as_str()),
            _ => false,
        }
    }

    /// Map a GPU variable method call to Metal API.
    fn emit_gpu_property(&self, var: &str, method: &str) -> String {
        match method {
            "name"              => format!("{var}.name().to_string()"),
            "totalMem"          => format!("{var}.recommended_max_working_set_size() as isize"),
            "freeMem"           => format!("({var}.recommended_max_working_set_size().saturating_sub({var}.current_allocated_size())) as isize"),
            "computeCapability" => format!("boring_gpu_family(&{var})"),
            "warpSize"          => "32isize".into(),
            "maxThreads"        => "1024isize".into(),
            "maxSharedMem"      => format!("{var}.max_threadgroup_memory_length() as isize"),
            "index"             => format!("Device::all().iter().position(|d| d.name() == {var}.name()).unwrap_or(0) as isize"),
            other               => format!("{var}.{other}()"),
        }
    }

    fn resolve_kernel_type(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(n) = &callee.kind {
                    if self.kernel_names.contains(n.as_str()) { return Some(n.clone()); }
                }
                None
            }
            ExprKind::KernelLaunch { kernel, .. } => self.resolve_kernel_type(kernel),
            ExprKind::New { ctor, .. }             => self.resolve_kernel_type(ctor),
            ExprKind::Pipe(lhs, method, _) if method == "wait" => self.resolve_kernel_type(lhs),
            ExprKind::Var(n) => self.var_kernel_type.get(n.as_str()).cloned(),
            _ => None,
        }
    }

    /// For `kernel_name`'s `init(...)` parameter list (in order), whether each
    /// positional param is a bare passthrough for an array-typed
    /// `'unified`/`'global`/`'const`/`'actor'global` field (`field = param`
    /// in the init body, this codebase's only real pattern for such fields --
    /// see `emit_init_stmt`'s identical check) -- these params render as
    /// `Buffer` (see `emit_kernel_new`) instead of a host `Vec`, and the
    /// kernel-constructor call site (this file's `expr()`'s `Call` case) must
    /// produce a `Buffer` for that argument position instead of a `Vec`.
    /// `None` if the kernel/init can't be found -- callers treat that as "no
    /// buffer params known" (preserves the pre-existing behavior).
    fn kernel_ctor_buffer_flags(&self, kernel_name: &str) -> Option<Vec<bool>> {
        let decl = self.kernel_decls.get(kernel_name)?;
        let init = decl.inits.first()?;
        Some(init.params.iter().map(|p| {
            init.body.iter().find_map(|stmt| {
                let Stmt::Expr(e) = stmt else { return None; };
                let ExprKind::Assign(lhs, rhs) = &e.kind else { return None; };
                let ExprKind::Var(fname) = &lhs.kind else { return None; };
                let ExprKind::Var(pname) = &rhs.kind else { return None; };
                if pname != &p.name { return None; }
                decl.fields.iter().find(|f| &f.name == fname)
            }).map(|f| {
                matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Const)
                    && matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _))
            }).unwrap_or(false)
        }).collect())
    }

    /// Renders `args` for a call to kernel `name`'s constructor targeting
    /// device `dev`, uploading each buffer-passthrough argument (see
    /// `kernel_ctor_buffer_flags`) via `new_buffer_with_data` -- or reusing
    /// it directly if already GPU-resident -- exactly the way a plain
    /// `Scale(data)` constructor call has always worked. Shared by both
    /// constructor call sites: plain `Scale(data)` (`ExprKind::Call`) and
    /// arena-qualified `new(g) Scale(data)` (`ExprKind::New`) -- the latter
    /// used to skip this entirely and pass `data` straight through as a bare
    /// `Vec`, the same E0308 `cuda::host`'s identical fix found and fixed
    /// there (see that module's `emit_kernel_ctor_args` doc).
    fn emit_kernel_ctor_args(&mut self, name: &str, dev: &str, args: &[Arg]) -> Vec<String> {
        let buffer_flags = self.kernel_ctor_buffer_flags(name);
        args.iter().enumerate().map(|(i, a)| {
            let is_buffer_pos = buffer_flags.as_ref().and_then(|f| f.get(i).copied()).unwrap_or(false);
            if is_buffer_pos {
                // `k_prev.field` passed directly as an argument (this
                // codebase's common inline-chaining shape, e.g.
                // `SoftmaxRowsKernel(k_scores.c, ...)`) -- `k_prev.field` is
                // ALREADY a `Buffer` (another kernel struct's own output
                // field), so give the new kernel its OWN independent copy via
                // `__boring_metal_buffer_copy` instead of routing through
                // `try_gpu_field_read`'s materializing `k_prev.read_field()`.
                // NOT `.clone()` -- see that helper's doc for why a bare
                // `.clone()` here used to silently alias the same `MTLBuffer`
                // between both kernel structs instead of copying it.
                if let ExprKind::Field(obj, field) = &a.value.kind {
                    if let ExprKind::Var(obj_name) = &obj.kind {
                        if self.var_kernel_type.contains_key(obj_name.as_str()) {
                            return format!("__boring_metal_buffer_copy(&{dev}, &{obj}.{field})?", dev = dev, obj = obj_name, field = field);
                        }
                    }
                }
                if let ExprKind::Var(vname) = &a.value.kind {
                    if self.resident_locals.contains(vname.as_str()) {
                        return format!(
                            "(match {v} {{ BoringGpuArg::Resident(buf, _) => __boring_metal_buffer_copy(&{dev}, &buf)?, BoringGpuArg::Host(v) => {dev}.new_buffer_with_data((v.iter().map(|&x| x as f32).collect::<Vec<f32>>()).as_ptr() as *const _, (v.len() * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared) }})",
                            v = vname, dev = dev
                        );
                    }
                    if self.f64_array_locals.contains(vname.as_str()) {
                        return format!(
                            "{dev}.new_buffer_with_data(({v}.iter().map(|&x| x as f32).collect::<Vec<f32>>()).as_ptr() as *const _, ({v}.len() * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared)",
                            v = vname, dev = dev
                        );
                    }
                }
                // Any other host-side arg (a fresh array literal, or a plain
                // `let`-bound Vec<f64> that isn't already tracked as
                // `resident_locals`/`f64_array_locals` above -- e.g. a
                // top-level `let data = [1.0, 2.0]`) is a `Vec<f64>` by the
                // general pipeline's own convention, NOT the `Vec<f32>`
                // `new_buffer_with_data` needs -- narrow it explicitly first.
                // Missing this cast silently copied half the intended bytes
                // (mem::size_of::<f32>() against actual f64 data) instead of
                // failing to compile, confirmed by inspecting the generated
                // Rust directly (`(data).as_ptr()` typed `*const f64` sized
                // as if `f32`).
                let s = self.coerce_call_arg(&a.value, false);
                return format!(
                    "{{ let __boring_buf: Vec<f32> = ({s}).iter().map(|&x| x as f32).collect(); {dev}.new_buffer_with_data(__boring_buf.as_ptr() as *const _, (__boring_buf.len() * mem::size_of::<f32>()) as u64, MTLResourceOptions::StorageModeShared) }}",
                    dev = dev, s = s
                );
            }
            // Scalar position -- unchanged. A `Vec<f64>` local bound
            // to a materializing call (see `f64_array_locals`'s doc)
            // needs an explicit cast back to this backend's own
            // native `Vec<f32>` convention.
            if let ExprKind::Var(vname) = &a.value.kind {
                if self.f64_array_locals.contains(vname.as_str()) {
                    return format!("{}.iter().map(|&x| x as f32).collect::<Vec<f32>>()", vname);
                }
            }
            self.coerce_call_arg(&a.value, false)
        }).collect()
    }

    fn try_gpu_field_read(&self, obj: &str, field: &str) -> Option<String> {
        let kernel_type = self.var_kernel_type.get(obj)?;
        let decl = self.kernel_decls.get(kernel_type)?;
        let kf = decl.fields.iter().find(|f| f.name == field)?;
        match kf.qual {
            GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Surface => {
                match &kf.ty {
                    Type::Array(_) | Type::ArrayN(_, _) => {
                        Some(format!("{}.read_{}()?", obj, field))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Recognizes the exact shape every kernel-touching free function's tail
    /// expression uses in practice: a bare `k.field` read of a local kernel-
    /// struct variable's own output array field. When matched (only ever
    /// called where `in_resident_return` is `Some`, i.e. this function's
    /// return type is itself `'gpu'unified`), skips `try_gpu_field_read`'s
    /// materializing `k.read_field()` entirely and instead hands back the
    /// kernel's own output `Buffer` wrapped as `BoringGpuArg::Resident` --
    /// letting a chained GPU caller consume it without a host round trip.
    /// Returns `None` for anything else (a more complex tail expression, or a
    /// var this pass doesn't know is a kernel-struct instance), which keeps
    /// the existing, always-correct materializing behavior as the fallback.
    fn try_resident_field_expr(&self, e: &Expr) -> Option<String> {
        let ExprKind::Field(obj, field) = &e.kind else { return None; };
        let ExprKind::Var(obj_name) = &obj.kind else { return None; };
        let kernel_type = self.var_kernel_type.get(obj_name)?;
        let decl = self.kernel_decls.get(kernel_type)?;
        let kf = decl.fields.iter().find(|f| &f.name == field)?;
        match kf.qual {
            GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Surface => {
                match &kf.ty {
                    Type::Array(_) | Type::ArrayN(_, _) => Some(format!(
                        "BoringGpuArg::Resident({obj}.{field}.clone(), ({obj}.{field}.length() as usize) / std::mem::size_of::<f32>())",
                        obj = obj_name, field = field
                    )),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

fn host_param_type(ty: &Type) -> String {
    match ty {
        Type::Qualified(inner, OwnerQual::GpuUnified)
        | Type::Qualified(inner, OwnerQual::GpuGlobal) => {
            format!("Vec<{}>", elem_rust_type(inner))
        }
        Type::Array(inner) => format!("Vec<{}>", rust_type(inner)),
        _ => rust_type(ty),
    }
}

fn elem_rust_type(ty: &Type) -> String {
    match ty {
        Type::Array(inner)        => rust_type(inner),
        Type::ArrayN(inner, _)    => rust_type(inner),
        Type::Qualified(inner, _) => elem_rust_type(inner),
        _                         => rust_type(ty),
    }
}

/// The general (backend-agnostic) transpiler pipeline's own host-side element
/// type convention for a `'gpu'unified` array (mirrors `emit_kernel.rs`'s
/// `kernel_host_element_type`) — always `f64` for a float array, UNLIKE this
/// backend's own `elem_rust_type`/`rust_type` (`f32`, Metal's device-native
/// float width, used for the kernel struct's own buffer/field types). Every
/// general-spliced caller of a `BoringGpuArg<T>`-returning function expects
/// `T` to follow the GENERAL pass's convention, not this backend's internal
/// one — confirmed via a real cross-compile `cargo check` (a genuine E0308,
/// `Vec<f32>` vs `Vec<f64>`, not hypothetical). See `emit_fn`'s `resident_elem`
/// and its wrap/unwrap call sites, which convert between the two explicitly.
fn general_host_elem_type(ty: &Type) -> String {
    match ty {
        Type::Array(inner) | Type::ArrayN(inner, _) => general_host_elem_type(inner),
        Type::Qualified(inner, _) => general_host_elem_type(inner),
        Type::Float => "f64".into(),
        Type::Named(n) if n == "float" || n == "f32" || n == "f64" => "f64".into(),
        other => rust_type(other),
    }
}

fn rust_type(ty: &Type) -> String {
    match ty {
        Type::Int            => "isize".into(),
        Type::Uint           => "usize".into(),
        Type::Uint8          => "u8".into(),
        Type::Int8            => "i8".into(),
        Type::Int16           => "i16".into(),
        Type::Int32           => "i32".into(),
        Type::Int64           => "i64".into(),
        Type::Int128          => "i128".into(),
        Type::Uint16          => "u16".into(),
        Type::Uint32          => "u32".into(),
        Type::Uint64          => "u64".into(),
        Type::Uint128         => "u128".into(),
        Type::Float          => "f32".into(),
        Type::Bool           => "bool".into(),
        Type::Str            => "String".into(),
        Type::Nil            => "()".into(),
        Type::Void           => "()".into(),
        Type::Never          => "!".into(),
        Type::Array(inner)   => format!("Vec<{}>", rust_type(inner)),
        Type::ArrayN(inner, n) => format!("[{}; {}]", rust_type(inner), n),
        // `{K=V}` dict type — was previously falling to this function's `_ => "()"`
        // default (the exact bug this fixes: tokenizer.br's `{string=int} vocab`
        // struct field / `var {string=int} vocab = {=}` local emitted as `()`).
        Type::Dict(k, v)     => format!(
            "std::collections::HashMap<{}, {}>", rust_type(k), rust_type(v)
        ),
        Type::Set(inner)     => format!("std::collections::HashSet<{}>", rust_type(inner)),
        Type::Optional(inner)  => format!("Option<{}>", rust_type(inner)),
        Type::Named(n) => match n.as_str() {
            "float" | "f64" => "f32",
            "f32"           => "f32",
            "int"           => "isize",
            "uint"          => "usize",
            "i64"           => "i64",
            "u64"           => "u64",
            "uint8"         => "u8",
            "int8"          => "i8",
            "int16"         => "i16",
            "int32"         => "i32",
            "int64"         => "i64",
            "int128"        => "i128",
            "uint16"        => "u16",
            "uint32"        => "u32",
            "uint64"        => "u64",
            "uint128"       => "u128",
            "bool"          => "bool",
            "string"        => "String",
            other           => return other.to_string(),
        }.into(),
        Type::TypeParam(p)     => p.clone(),
        Type::Qualified(inner, _) => rust_type(inner),
        Type::Generic(n, args) => {
            let s: Vec<String> = args.iter().map(rust_type).collect();
            format!("{}<{}>", n, s.join(", "))
        }
        _ => "()".into(),
    }
}

fn elem_size_bytes(ty: &Type) -> usize {
    match ty {
        Type::Float                               => 4,
        Type::Int | Type::Uint                    => 8,
        Type::Uint8 | Type::Int8                   => 1,
        Type::Int16 | Type::Uint16                 => 2,
        Type::Int32 | Type::Uint32                 => 4,
        Type::Int64 | Type::Uint64                 => 8,
        Type::Int128 | Type::Uint128               => 16,
        Type::Bool                                => 1,
        Type::Named(n) => match n.as_str() {
            "float" | "f64" => 4,
            "f32"           => 4,
            "int"   | "i64" => 8,
            "uint"  | "u64" => 8,
            "uint8" | "int8" => 1,
            "int16" | "uint16" => 2,
            "int32" | "uint32" => 4,
            "int64" | "uint64" => 8,
            "int128" | "uint128" => 16,
            "i32"           => 4,
            "u32"           => 4,
            _               => 8,
        },
        Type::Qualified(inner, _)                 => elem_size_bytes(inner),
        Type::Array(inner) | Type::ArrayN(inner, _) => elem_size_bytes(inner),
        _                                         => 8,
    }
}

fn emit_scalar_default(expr: &Expr) -> String {
    match &expr.kind {
        ExprKind::Int(n)   => n.to_string(),
        ExprKind::Float(f) => {
            let s = format!("{}", f);
            if s.contains('.') { s } else { format!("{}.0", s) }
        }
        ExprKind::Bool(b)  => b.to_string(),
        ExprKind::UnaryOp(UnaryOp::Neg, inner) => match &inner.kind {
            ExprKind::Int(n)   => format!("-{}", n),
            ExprKind::Float(f) => format!("-{}", f),
            _ => "Default::default()".into(),
        },
        _ => "Default::default()".into(),
    }
}

fn binop_rust(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",   BinOp::Sub => "-",  BinOp::Mul => "*",
        BinOp::Div => "/",   BinOp::Rem => "%",
        BinOp::Eq  => "==",  BinOp::NotEq => "!=",  BinOp::RefEq => "==",
        BinOp::Lt  => "<",   BinOp::Gt => ">",
        BinOp::LtEq => "<=", BinOp::GtEq => ">=",
        BinOp::And => "&&",  BinOp::Or => "||",
        BinOp::BitAnd => "&", BinOp::BitOr => "|", BinOp::BitXor => "^",
        BinOp::Shl => "<<",  BinOp::Shr => ">>",
        _ => "/*op*/",
    }
}

fn unaryop_rust(op: &UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg    => "-",
        UnaryOp::Not    => "!",
        UnaryOp::BitNot => "!",
    }
}
