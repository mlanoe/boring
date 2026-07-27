// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Rust host-side code emitter for the CUDA backend.

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
    /// Names of all `kernel` structs — used to translate `Scale(args)` → `Scale::new(...)`.
    kernel_names: std::collections::HashSet<String>,
    /// Kernel struct declarations indexed by name — for field type lookup at readback sites.
    kernel_decls: std::collections::HashMap<String, KernelDecl>,
    /// Tracks which local variables hold a kernel struct instance and which type.
    /// Used to emit `k.read_buf()?` instead of `k.buf` for device-side fields.
    var_kernel_type: std::collections::HashMap<String, String>,
    /// Variables that hold an `Arc<CudaContext>` (from `GPU(n)` or `new(g) ...`).
    /// Used to map `.name()`, `.total_mem()` etc. to cudarc 0.19 methods.
    gpu_vars: std::collections::HashSet<String>,
    /// Local variables (by name, unscoped) whose declared type or initializer is a
    /// `{K=V}` dict literal/type — used so `d[key]`/`d[key] = v` emit `HashMap`
    /// `.get(&key).cloned()`/`.insert(key, v)` instead of Vec-style `[key as usize]`
    /// indexing (see `is_dict_obj`).
    dict_vars: std::collections::HashSet<String>,
    /// Struct field names (flat, not namespaced by struct — this emitter has no
    /// per-struct-type field resolution) declared with a `{K=V}` dict type, so
    /// `self.field[key]`/`self.field[key] = v` get the same HashMap treatment as
    /// `dict_vars`. Populated once from every `Item::Struct` in the program.
    dict_fields: std::collections::HashSet<String>,
    /// Names of every free (non-method) `throws` function — used so a call to one
    /// of them, from inside another `throws` function, gets `?` appended (see
    /// `in_throws`). Does NOT cover `throws` struct methods (`self.foo()` chains,
    /// e.g. decoder.br/encoder.br) — resolving a method call's receiver type well
    /// enough to know whether it throws is not implemented here; see this
    /// module's doc comment for the reasoning and `emit_fn`'s call site.
    fn_throws: std::collections::HashSet<String>,
    /// True while emitting the body of a `throws` function — gates both `return
    /// Err(...)` for `Stmt::Throw`/`Stmt::Guard` and `?`-suffixing calls to other
    /// `fn_throws` functions.
    in_throws: bool,
    /// Variant name → owning enum type name (e.g. `"Tiny" -> "ModelSize"`), for
    /// rendering a bare `Pattern::Variant` match pattern as Rust's required
    /// path-qualified `EnumName::Variant` — flat/unscoped like `dict_fields`,
    /// so this assumes variant names are unique across the whole program (true
    /// for every enum this codebase's CUDA/Metal-targeted code actually matches
    /// on). Populated once from every `Item::Enum`.
    variant_to_enum: std::collections::HashMap<String, String>,
    /// Free function name → per-parameter "does this position need `&`?" flags,
    /// per boring's by-ref contract (`CLAUDE.md`: "Structs, enums, arrays, dicts,
    /// sets — always passed by reference"). Populated once from every `Item::Fn`
    /// in the whole (unfiltered) program — this backend only emits BODIES for
    /// kernel-touching functions now (see `cuda::transpile_cuda`'s doc comment),
    /// but every call site, in either emitter, must agree on every function's
    /// signature for cross-calls between the two to type-check.
    fn_ref_params: std::collections::HashMap<String, Vec<bool>>,
    /// Declared struct AND enum names (both, despite the field name) — a
    /// `Type::Named(n)` parameter is by-ref-worthy when `n` names either.
    struct_names: std::collections::HashSet<String>,
    /// The CURRENTLY-being-emitted function's own by-ref-typed parameter names
    /// (reset per `emit_fn` call). A bare read of one of these (`Stmt::For`'s
    /// iterable, `ExprKind::Index`, or forwarding it as an argument to something
    /// that expects an OWNED value, e.g. a kernel constructor) needs an explicit
    /// `.clone()`/`.iter().cloned()` — indexing or iterating a `&Vec<T>` yields
    /// `&T`, not `T`.
    ref_params: std::collections::HashSet<String>,
    /// True while emitting the body of a GPU-resident-returning function (see
    /// `emit_fn`'s `resident_elem`) — gates wrapping `Stmt::Return`'s value in
    /// `BoringGpuArg::Host(...)` to match the declared `BoringGpuArg<T>` return
    /// type every general-spliced caller unconditionally expects.
    in_resident_return: bool,
    /// Free function name → per-parameter "does this position render as
    /// `isize`/`usize`?" flags (see `narrowed_int_param_type`'s doc) — mirrors
    /// `fn_ref_params`, needed for the SAME reason: a call from one
    /// kernel-touching function to another (both on this emitter, e.g.
    /// `attention_heads_gpu` calling `transpose_gpu`) must cast its `i64`/`u64`
    /// locals to match the callee's narrowed param type, or it's a real E0308
    /// (confirmed via `cargo check`).
    fn_narrowed_int_params: std::collections::HashMap<String, Vec<Option<&'static str>>>,
    /// Free function name → its `Type` when GPU-resident-returning (see
    /// `emit_fn`'s `resident_elem`). A call from one kernel-touching function
    /// to another that returns `BoringGpuArg<T>` (e.g. `attention_heads_gpu`
    /// calling `transpose_gpu`) must materialize the result — this emitter has
    /// no cross-function residency-preservation optimization of its own, unlike
    /// the general pipeline's `resident_call_vars` (see `emit_methods.rs`'s
    /// identical-purpose fix there).
    fn_returns_resident: std::collections::HashMap<String, Type>,
}

/// True when `ty` is one of boring's always-by-reference parameter kinds (see
/// `HostEmitter::fn_ref_params`'s doc comment for the exact contract this
/// mirrors).
fn is_ref_worthy_type(ty: &Type, struct_names: &std::collections::HashSet<String>) -> bool {
    match ty {
        Type::Array(_) | Type::ArrayN(_, _) | Type::Dict(_, _) | Type::Set(_) => true,
        Type::Named(n) => struct_names.contains(n),
        _ => false,
    }
}

/// True when `ty` is boring's plain `int`/`uint` (as opposed to an explicit
/// fixed-width alias like `int32`). Returns `(param-position Rust type, this
/// file's existing i64/u64-based body-codegen convention)`.
///
/// The general (std/wgpu-shared) pipeline's own `emit_type` (`emit_top.rs`)
/// maps plain `int`/`uint` to Rust's `isize`/`usize` — NOT `i64`/`u64` as
/// `CLAUDE.md`'s doc table might suggest — confirmed empirically: a general-
/// spliced "plain" function (see `cuda::transpile_cuda`'s doc comment) calling
/// one of THIS backend's kernel-touching functions passes an `isize` local for
/// a plain-`int` argument position (a real E0308 otherwise). Rather than
/// propagate `isize`/`usize` through this whole file's existing `i64`/`u64`-
/// based expression/statement codegen (a much larger, riskier change), a
/// plain-int/uint PARAMETER is rendered as `isize`/`usize` to match the
/// caller, then immediately shadow-rebound to `i64`/`u64` on entry (see
/// `emit_fn`) — a plain `let x = x as i64;` is valid Rust shadowing, so no
/// other line of the function needs to change.
fn narrowed_int_param_type(ty: &Type) -> Option<(&'static str, &'static str)> {
    match ty {
        Type::Int => Some(("isize", "i64")),
        Type::Uint => Some(("usize", "u64")),
        Type::Named(n) if n == "int" => Some(("isize", "i64")),
        Type::Named(n) if n == "uint" => Some(("usize", "u64")),
        _ => None,
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
            dict_vars: std::collections::HashSet::new(),
            dict_fields: std::collections::HashSet::new(),
            fn_throws: std::collections::HashSet::new(),
            in_throws: false,
            variant_to_enum: std::collections::HashMap::new(),
            fn_ref_params: std::collections::HashMap::new(),
            struct_names: std::collections::HashSet::new(),
            ref_params: std::collections::HashSet::new(),
            in_resident_return: false,
            fn_narrowed_int_params: std::collections::HashMap::new(),
            fn_returns_resident: std::collections::HashMap::new(),
        }
    }

    /// See `is_ref_worthy_type` (module-level free fn) — bound method form for
    /// call sites already holding `&self`.
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
    /// dict — either a tracked local var (`dict_vars`) or a struct field declared
    /// with a dict type (`dict_fields`, e.g. `self.vocab`). See those fields' docs.
    fn is_dict_obj(&self, obj: &Expr) -> bool {
        match &obj.kind {
            ExprKind::Var(v) => self.dict_vars.contains(v.as_str()),
            ExprKind::Field(_, f) => self.dict_fields.contains(f.as_str()),
            _ => false,
        }
    }

    /// Record `name` in `dict_vars` when its declared type or initializer marks it
    /// as a dict — called from every `let`/`var` binding site (top-level, ordinary
    /// statements, and kernel `init` bodies).
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
        // Pass 1: struct/enum names, needed below (before pass 2) to resolve
        // whether a `Type::Named(n)` function parameter is by-ref-worthy.
        for item in &program.items {
            match item {
                Item::Struct(s) => { self.struct_names.insert(s.name.clone()); }
                Item::Enum(e) => { self.struct_names.insert(e.name.clone()); }
                _ => {}
            }
        }
        // Pass 2: everything else.
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
                let narrowed_flags: Vec<Option<&'static str>> = f.params.iter()
                    .map(|p| p.ty.as_ref().and_then(|ty| narrowed_int_param_type(ty)).map(|(narrowed, _)| narrowed))
                    .collect();
                self.fn_narrowed_int_params.insert(f.name.clone(), narrowed_flags);
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
        self.line("// Generated by boring build --target cuda.");
        self.blank();
        self.emit_prelude(kernel_names);

        // Kernel structs (unchanged, own emission) + kernel-touching function
        // bodies (this backend's own real CUDA API -- fallible construction,
        // `__boring_launch`, etc. -- see this module's doc comment for why the
        // general pipeline can't render these). Every OTHER item (plain
        // fn/struct/enum, top-level stmt/let folded into `boring_main`) is
        // already rendered correctly in `general_code`, spliced in below.
        for item in &program.items {
            match item {
                Item::Kernel(decl) => {
                    self.blank();
                    self.emit_kernel_struct(decl);
                }
                Item::Fn(f) if kernel_touching.contains(&f.name) => {
                    self.blank();
                    if f.name == "main" {
                        // The user's own kernel-touching `main` is emitted as
                        // `boring_main` instead, so this backend's own real
                        // `fn main()` (below) stays the sole Rust entry point
                        // -- matching the convention a non-kernel-touching
                        // `main` already gets from the general-pipeline splice
                        // (see `rename_top_level_main`).
                        let mut renamed = f.clone();
                        renamed.name = "boring_main".to_string();
                        self.emit_fn(&renamed, None);
                    } else {
                        self.emit_fn(f, None);
                    }
                }
                _ => {}
            }
        }

        self.blank();
        self.out.push_str(general_code);
        self.blank();

        self.line("fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        if !kernel_names.is_empty() {
            self.line("boring_gpu_init()?;");
        }
        if top_level_kernel_touching {
            // Bare top-level kernel construction/dispatch — see this backend's
            // module doc comment and `top_level_touches_kernel`'s doc for why
            // this can't go through the general-pipeline splice (`general_code`
            // above has none of the top-level content in this case; the general
            // pass left it alone entirely, per `gpu_top_level_handled_by_host`).
            // This is exactly this backend's OWN pre-splice top-level handling,
            // reinstated only for this case.
            for item in &program.items {
                match item {
                    Item::Let(s) => {
                        let binding = if s.binding.is_mutable() { "let mut" } else { "let" };
                        let ty_ann = s.ty.as_ref().map(|t| format!(": {}", rust_type(t))).unwrap_or_default();
                        if let Some(val) = &s.value {
                            self.track_kernel_var(&s.name, val);
                            self.track_dict_var(&s.name, s.ty.as_ref(), Some(val));
                            let rhs = self.expr(val);
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

    // ── GPU prelude ────────────────────────────────────────────────────────────

    fn emit_prelude(&mut self, kernel_names: &[String]) {
        // `PushKernelArg` brings `LaunchArgs::arg()` into scope -- it's a trait
        // method (real cudarc 0.19.8: `cudarc::driver::safe::launch::PushKernelArg`,
        // re-exported at `cudarc::driver::PushKernelArg`), not an inherent one;
        // omitting this import is a real E0599 ("no method named `arg`"),
        // confirmed via `cargo check`.
        self.line("use cudarc::driver::{CudaContext, CudaModule, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};");
        self.line("use cudarc::nvrtc::Ptx;");
        self.line("use std::sync::{Arc, OnceLock};");
        self.blank();
        self.line("static BORING_PTX: &str = include_str!(env!(\"BORING_PTX_PATH\"));");
        self.blank();
        self.line("static BORING_CTX: OnceLock<Arc<CudaContext>> = OnceLock::new();");
        self.line("static BORING_MODULE: OnceLock<Arc<CudaModule>> = OnceLock::new();");
        self.blank();
        self.line("fn boring_gpu_ctx() -> Arc<CudaContext> {");
        self.indent += 1;
        self.line("Arc::clone(BORING_CTX.get().expect(\"boring_gpu_init() not called\"))");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("fn boring_gpu_init() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        // `CudaContext::new`/`CudaContext::load_module` already return
        // `Arc<Self>` in real cudarc 0.19.8 -- wrapping again in `Arc::new(...)`
        // produced `Arc<Arc<_>>`, a real E0308 confirmed via `cargo check`.
        self.line("let ctx = CudaContext::new(0)?;");
        self.line("let ptx = Ptx::from_src(BORING_PTX);");
        self.line("let module = ctx.load_module(ptx)?;");
        self.line("let _ = BORING_CTX.set(ctx);");
        self.line("let _ = BORING_MODULE.set(module);");
        self.line("Ok(())");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // Multi-GPU context accessor: `GPU(n)` → `boring_gpu_ctx_n(n)?`.
        self.line("fn boring_gpu_ctx_n(idx: usize) -> Result<Arc<CudaContext>, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        // `CudaContext::new` already returns `Arc<Self>` -- see `boring_gpu_init`'s
        // identical fix.
        self.line("Ok(CudaContext::new(idx)?)");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // The general (std/wgpu-shared) transpiler pipeline's own pre-pass marks any
        // function whose declared return type is `'gpu'unified`/`'gpu'global`-qualified
        // (e.g. math_gpu.br's `transpose_gpu`/`linear_gpu`) as "GPU-resident-returning"
        // and renders every CALLER of it (in the general-pipeline-spliced "plain" code
        // this backend embeds -- see `cuda::transpile_cuda`'s doc comment) to expect a
        // `BoringGpuArg<T>` value back, unconditionally -- this is NOT gated on
        // `is_gpu_target`/`gpu_kernels` at all (confirmed by inspecting a PLAIN `boring
        // build`, no target, whose own output references these same symbols). Only
        // `wgpu::host` actually DEFINES `BoringGpuArg`/`__boring_gpu_device`/etc., since
        // wgpu is the only backend with a real cross-function GPU-buffer-residency
        // optimization built on top of the general pipeline's kernel-aware codegen.
        // This backend has no such optimization of its own -- every kernel-touching
        // function here (this file's own `emit_fn`, gated on `resident_elem`) always
        // returns the `Host(...)` variant -- so the `Resident` arm below is never
        // actually constructed, but the type still needs to exist and the match arms
        // still need to type-check wherever the general-spliced code references them.
        // `Resident`'s buffer is always `Vec<f32>`, NOT `Vec<T>` -- the general
        // pipeline's own call sites that pattern-match this (`materialize_resident_call`
        // in `emit_kernel.rs`) always call `__boring_gpu_copy_d2h::<f32>(..)` on it
        // regardless of `T`, since a real GPU buffer is always 32-bit device-side
        // (`kernel_host_scalar_type`'s convention, matching wgpu's actual buffers) even
        // when `T` (the HOST element type once materialized) is `f64`. Confirmed via a
        // real `cargo check`: this was a genuine E0308 (`&Arc<Vec<f32>>` vs
        // `&Arc<Vec<f64>>`) before this fix -- again unreachable in practice (see this
        // enum's doc above) but must type-check.
        self.line("#[allow(dead_code)]");
        self.line("enum BoringGpuArg<T> {");
        self.indent += 1;
        self.line("Resident(Arc<Vec<f32>>, usize),");
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
        self.line("BoringGpuArg::Resident(b, n) => BoringGpuArg::Resident(Arc::clone(b), *n),");
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
        self.line("#[allow(dead_code)] fn __boring_gpu_device() -> Arc<CudaContext> { boring_gpu_ctx() }");
        self.line("#[allow(dead_code)] fn __boring_gpu_queue() -> Arc<CudaContext> { boring_gpu_ctx() }");
        // `T` is unused (Rust allows an unused type param on a free fn, unlike a
        // struct) -- kept only so call sites' `::<f32>` turbofish (matching the
        // general pipeline's own hardcoded convention, see `BoringGpuArg`'s doc)
        // still parses; `buf`/the return type are concretely `Vec<f32>` to match
        // `BoringGpuArg::Resident`'s buffer type above.
        self.line("#[allow(dead_code)]");
        self.line("fn __boring_gpu_copy_d2h<T>(_device: &Arc<CudaContext>, _queue: &Arc<CudaContext>, buf: &Arc<Vec<f32>>) -> Vec<f32> {");
        self.indent += 1;
        self.line("(**buf).clone() // unreachable in practice -- see this fn's call site's doc comment");
        self.indent -= 1;
        self.line("}");
        self.line("#[allow(dead_code)]");
        self.line("fn __boring_gpu_copy_h2d<T>(_device: &Arc<CudaContext>, _queue: &Arc<CudaContext>, _src: &[u8], _dst: &Arc<Vec<f32>>) {");
        self.indent += 1;
        self.line("unreachable!(\"cuda backend never constructs BoringGpuArg::Resident\")");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // Each handle owns the CUDA stream its kernel was launched on, so `.wait`
        // synchronizes only that stream (not the whole device).  `after = [..]`
        // dependencies are wired GPU-side via stream ordering, no CPU sync.
        self.line("struct KernelHandle<T> { inner: T, stream: Arc<CudaStream> }");
        self.line("impl<T> KernelHandle<T> {");
        self.indent += 1;
        self.line("fn wait(self) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("self.stream.synchronize()?;");
        self.line("Ok(self.inner)");
        self.indent -= 1;
        self.line("}");
        self.line("fn done(&self) -> bool { true }");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // Boring built-in Dimension type used by 2-D kernels.
        self.line(crate::transpiler::helpers::DIMENSION_STRUCT_RUST);
        self.blank();
        // Stream priority helper. priority 0 = normal, -1 = high, 1 = low (CUDA
        // convention: lower int = higher priority). Previously hand-rolled via
        // raw `cuStreamCreateWithPriority`/`CudaStream::from_raw` FFI, neither
        // of which exists in real cudarc 0.19.8 (`CU_STREAM_NON_BLOCKING` isn't
        // in scope at that path and `CudaStream` has no `from_raw` — confirmed
        // via a real `cargo check`, both real E0425/E0599, not hypothetical).
        // cudarc 0.19.8 already has a safe equivalent directly on `CudaContext`.
        self.line("fn boring_new_stream_with_priority(ctx: &Arc<CudaContext>, priority: i32) -> Result<Arc<CudaStream>, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("if priority == 0 { return Ok(ctx.new_stream()?); }");
        self.line("Ok(ctx.new_stream_with_priority(priority)?)");
        self.indent -= 1;
        self.line("}");
        let _ = kernel_names; // used by caller for PTX loading
    }

    // ── Kernel struct → Rust host wrapper ─────────────────────────────────────

    fn emit_kernel_struct(&mut self, decl: &KernelDecl) {
        let name = &decl.name;

        // Struct fields.
        self.line(&format!("struct {} {{", name));
        self.indent += 1;
        self.line("__ctx: Arc<CudaContext>,");
        self.line("__stream: Arc<CudaStream>,");
        for field in &decl.fields {
            match field.qual {
                GpuQual::Sync | GpuQual::Local => {
                    // Block SRAM / registers — no host-side storage.
                }
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Const | GpuQual::Surface => {
                    let ty = self.host_field_type(field);
                    self.line(&format!("{}: {},", field.name, ty));
                }
            }
        }
        // Scalar non-GPU fields (Local without array, Const scalars).
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

        // impl block.
        self.line(&format!("impl {} {{", name));
        self.indent += 1;

        // Constructor(s).
        for init in &decl.inits {
            self.emit_kernel_new(name, &decl.fields, init);
            self.blank();
        }
        if decl.inits.is_empty() {
            self.emit_kernel_new_default(name, &decl.fields);
            self.blank();
        }

        // Accessors for 'unified fields (D2H).
        for field in &decl.fields {
            if matches!(field.qual, GpuQual::Unified) {
                let elem = elem_rust_type(&field.ty);
                self.line(&format!(
                    "fn read_{}(&self) -> Result<Vec<{}>, Box<dyn std::error::Error + Send + Sync>> {{",
                    field.name, elem
                ));
                self.indent += 1;
                self.line(&format!("Ok(self.__stream.clone_dtoh(&self.{})?)", field.name));
                self.indent -= 1;
                self.line("}");
                self.blank();
            }
        }

        // __boring_launch.
        self.emit_boring_launch(name, &decl.fields);

        self.indent -= 1;
        self.line("}");
    }

    fn emit_kernel_new(&mut self, name: &str, fields: &[KernelFieldDecl], init: &InitDecl) {
        let params: Vec<String> = init.params.iter().map(|p| {
            let ty = p.ty.as_ref()
                .map(|t| host_param_type(t, fields))
                .unwrap_or_else(|| "()".into());
            format!("{}: {}", p.name, ty)
        }).collect();
        let all_params = if params.is_empty() {
            "__ctx: Arc<CudaContext>".into()
        } else {
            format!("__ctx: Arc<CudaContext>, {}", params.join(", "))
        };

        self.line(&format!(
            "fn new({}) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {{",
            all_params
        ));
        self.indent += 1;

        // Emit init body statements, translating field assignments into device allocations.
        for stmt in &init.body {
            self.emit_init_stmt(stmt, fields);
        }

        // Fall back: fields not set in the body get defaults.
        // We track which fields were assigned in the init body (simple heuristic: name match).
        let assigned: std::collections::HashSet<String> = init.body.iter()
            .filter_map(|s| match s {
                Stmt::Expr(e) => {
                    if let ExprKind::Assign(lhs, _) = &e.kind {
                        if let ExprKind::Var(name) = &lhs.kind { Some(name.clone()) }
                        else { None }
                    } else { None }
                }
                _ => None,
            }).collect();

        for field in fields {
            if !assigned.contains(&field.name) {
                match field.qual {
                    GpuQual::Const if matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _)) => {
                        let elem = elem_rust_type(&field.ty);
                        self.line(&format!("let {}: Vec<{}> = Vec::new();", field.name, elem));
                    }
                    GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Const | GpuQual::Surface => {
                        let elem = elem_rust_type(&field.ty);
                        self.line(&format!(
                            "let {} = __ctx.default_stream().alloc_zeros::<{}>(1)?;",
                            field.name, elem
                        ));
                    }
                    GpuQual::Sync | GpuQual::Local => {
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
        }

        self.line(&format!("Ok({} {{", name));
        self.indent += 1;
        // `.default_stream()` takes `&Arc<Self>` (borrow) -- must come BEFORE
        // the `__ctx,` shorthand below, which MOVES `__ctx` into the struct
        // literal; Rust evaluates struct-literal field initializers in the
        // order written, so the original order (`__ctx,` first) was a real
        // E0382 ("value moved here" / "use of moved value"), confirmed via a
        // real `cargo check`.
        self.line("__stream: __ctx.default_stream(),");
        self.line("__ctx,");
        for field in fields {
            match field.qual {
                GpuQual::Sync => {} // no host field
                GpuQual::Local => {
                    match &field.ty {
                        Type::Array(_) | Type::ArrayN(_, _) => {}
                        _ => self.line(&format!("{},", field.name)),
                    }
                }
                _ => self.line(&format!("{},", field.name)),
            }
        }
        self.indent -= 1;
        self.line("})");
        self.indent -= 1;
        self.line("}");
    }

    fn emit_kernel_new_default(&mut self, name: &str, fields: &[KernelFieldDecl]) {
        self.line("fn new(__ctx: Arc<CudaContext>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        for field in fields {
            match field.qual {
                GpuQual::Const if matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _)) => {
                    let elem = elem_rust_type(&field.ty);
                    self.line(&format!("let {}: Vec<{}> = Vec::new();", field.name, elem));
                }
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Const | GpuQual::Surface => {
                    let elem = elem_rust_type(&field.ty);
                    self.line(&format!(
                        "let {} = __ctx.default_stream().alloc_zeros::<{}>(1)?;",
                        field.name, elem
                    ));
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
        self.line(&format!("Ok({} {{", name));
        self.indent += 1;
        // `.default_stream()` takes `&Arc<Self>` (borrow) -- must come BEFORE
        // the `__ctx,` shorthand below, which MOVES `__ctx` into the struct
        // literal; Rust evaluates struct-literal field initializers in the
        // order written, so the original order (`__ctx,` first) was a real
        // E0382 ("value moved here" / "use of moved value"), confirmed via a
        // real `cargo check`.
        self.line("__stream: __ctx.default_stream(),");
        self.line("__ctx,");
        for field in fields {
            match field.qual {
                GpuQual::Sync => {}
                GpuQual::Local => match &field.ty {
                    Type::Array(_) | Type::ArrayN(_, _) => {}
                    _ => self.line(&format!("{},", field.name)),
                },
                _ => self.line(&format!("{},", field.name)),
            }
        }
        self.indent -= 1;
        self.line("})");
        self.indent -= 1;
        self.line("}");
    }

    fn emit_init_stmt(&mut self, stmt: &Stmt, fields: &[KernelFieldDecl]) {
        match stmt {
            Stmt::Expr(e) => {
                if let ExprKind::Assign(lhs, rhs) = &e.kind {
                    if let ExprKind::Var(fname) = &lhs.kind {
                        // Is this a GPU-memory field?
                        if let Some(field) = fields.iter().find(|f| &f.name == fname) {
                            match field.qual {
                                GpuQual::Const if matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _)) => {
                                    // 'const array field: store as Vec<T> on host.
                                    // The data is uploaded to __constant__ memory in __boring_launch.
                                    let elem = elem_rust_type(&field.ty);
                                    match &rhs.kind {
                                        ExprKind::Array(elems) => {
                                            let lit: Vec<String> = elems.iter().map(|e| self.expr(e)).collect();
                                            self.line(&format!(
                                                "let {}: Vec<{}> = vec![{}];",
                                                fname, elem, lit.join(", ")
                                            ));
                                        }
                                        _ => {
                                            let rhs_s = self.expr(rhs);
                                            self.line(&format!(
                                                "let {}: Vec<{}> = {}.to_vec();",
                                                fname, elem, rhs_s
                                            ));
                                        }
                                    }
                                    return;
                                }
                                // Scalar `'const`/`'local` field (e.g. `let int rows`) — a
                                // plain kernel-launch parameter, no device upload at all
                                // (passed via `launcher.arg` later, see `emit_boring_launch`).
                                // Previously fell into the array-upload arm below
                                // unconditionally (the match only guarded Const-as-array in
                                // the FIRST arm, not here), producing e.g.
                                // `let rows = __ctx.default_stream().clone_htod::<i64>(&r)?;`
                                // for a bare `i64` init param -- a real E0277 (`i64` doesn't
                                // implement `HostSlice<i64>`), confirmed via a real `cargo
                                // check` against cudarc 0.19.8.
                                GpuQual::Const | GpuQual::Local
                                    if !matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _)) =>
                                {
                                    let ty = rust_type(&field.ty);
                                    let rhs_s = self.expr(rhs);
                                    self.line(&format!("let {}: {} = {};", fname, ty, rhs_s));
                                    return;
                                }
                                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Surface => {
                                    // Check if RHS is an `ArrayFill` / `[..n]` pattern.
                                    match &rhs.kind {
                                        ExprKind::ArrayFill { value: _, count } | ExprKind::ArrayAlloc { count } => {
                                            let n = self.expr(count);
                                            let elem = elem_rust_type(&field.ty);
                                            self.line(&format!(
                                                "let {} = __ctx.default_stream().alloc_zeros::<{}>({} as usize)?;",
                                                fname, elem, n
                                            ));
                                            return;
                                        }
                                        ExprKind::Array(elems) => {
                                            let elem = elem_rust_type(&field.ty);
                                            let lit: Vec<String> = elems.iter().map(|e| self.expr(e)).collect();
                                            // `clone_htod<T, Src: HostSlice<T> + ?Sized>` takes
                                            // TWO generic params -- `_` lets `Src` (here
                                            // `Vec<{elem}>`) infer from the argument (real
                                            // cudarc 0.19.8 signature; confirmed via `cargo
                                            // check`, was a real E0107 with only one supplied).
                                            self.line(&format!(
                                                "let {} = __ctx.default_stream().clone_htod::<{}, _>(&vec![{}])?;",
                                                fname, elem, lit.join(", ")
                                            ));
                                            return;
                                        }
                                        _ => {
                                            // Assume RHS is a Vec<T> param — upload to device.
                                            let rhs_s = self.expr(rhs);
                                            let elem = elem_rust_type(&field.ty);
                                            self.line(&format!(
                                                "let {} = __ctx.default_stream().clone_htod::<{}, _>(&{})?;",
                                                fname, elem, rhs_s
                                            ));
                                            return;
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                // Generic stmt.
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

    fn emit_boring_launch(&mut self, name: &str, fields: &[KernelFieldDecl]) {
        // Auto grid sizing: when the first field is a device array ('unified/'global/
        // 'actor'global), `grid_dim` becomes optional and is derived from its length.
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

        if let Some(field) = &auto_grid_field {
            self.line(
                "fn __boring_launch(mut self, block_dim: (u32,u32,u32), grid_dim: Option<(u32,u32,u32)>, after: &[&Arc<CudaStream>], priority: i32) \
                 -> Result<KernelHandle<Self>, Box<dyn std::error::Error + Send + Sync>> {"
            );
            self.indent += 1;
            self.line("let grid_dim = grid_dim.unwrap_or_else(|| {");
            self.indent += 1;
            self.line(&format!("let n = self.{}.len() as u32;", field));
            self.line("((n + block_dim.0 - 1) / block_dim.0, 1, 1)");
            self.indent -= 1;
            self.line("});");
        } else {
            self.line(
                "fn __boring_launch(mut self, block_dim: (u32,u32,u32), grid_dim: (u32,u32,u32), after: &[&Arc<CudaStream>], priority: i32) \
                 -> Result<KernelHandle<Self>, Box<dyn std::error::Error + Send + Sync>> {"
            );
            self.indent += 1;
        }
        // Compute smem_bytes from dynamic 'shared fields (extern __shared__ T arr[]).
        // Statically-sized 'shared ArrayN fields embed their size in the kernel declaration
        // and do not contribute to smem_bytes.
        let dyn_shared_terms: Vec<String> = fields.iter()
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
        let smem_expr = if dyn_shared_terms.is_empty() {
            "0u32".into()
        } else {
            format!("({}) as u32", dyn_shared_terms.join(" + "))
        };
        self.line(&format!("let smem_bytes: u32 = {};", smem_expr));
        self.line("let cfg = LaunchConfig {");
        self.indent += 1;
        self.line("block_dim,");
        self.line("grid_dim,");
        self.line("shared_mem_bytes: smem_bytes,");
        self.indent -= 1;
        self.line("};");
        // cudarc 0.19: get function from the global module, fork a fresh stream,
        // order it after dependency streams GPU-side, then launch with builder pattern.
        self.line(&format!(
            "let func = BORING_MODULE.get().unwrap().load_function(\"{}_kernel\")?;",
            name
        ));
        self.line("let stream = boring_new_stream_with_priority(&self.__ctx, priority)?;");
        self.line("for dep in after { dep.synchronize()?; }");

        // Upload 'const fixed-size arrays to __constant__ memory before launch.
        for f in fields {
            if matches!(f.qual, GpuQual::Const) && matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) {
                self.line(&format!(
                    "if !self.{name}.is_empty() {{",
                    name = f.name
                ));
                self.indent += 1;
                let elem = elem_rust_type(&f.ty);
                // Real cudarc 0.19.8 `CudaModule::get_global(name, stream)` takes the
                // stream as a second argument and always returns an untyped
                // `CudaViewMut<'_, u8>` (no `::<T>` turbofish on `get_global` itself) --
                // confirmed via `cargo check` (was a real E0107/wrong-type otherwise).
                // `.transmute_mut::<T>(len)` recovers the typed view, matching the
                // upstream doc example. `memcpy_htod`'s real parameter order is
                // `(src, dst)`, not `(dst, src)` as this line previously had it (a
                // real E0308: `CudaViewMut<u8>` doesn't implement `HostSlice`).
                self.line(&format!(
                    "let mut __sym_{name} = BORING_MODULE.get().unwrap().get_global(\"{name}\", &stream)?;",
                    name = f.name
                ));
                self.line(&format!(
                    "let mut __sym_{name} = unsafe {{ __sym_{name}.transmute_mut::<{elem}>(self.{name}.len()).unwrap() }};",
                    name = f.name, elem = elem
                ));
                self.line(&format!(
                    "stream.memcpy_htod(&self.{name}, &mut __sym_{name})?;",
                    name = f.name
                ));
                self.indent -= 1;
                self.line("}");
            }
        }

        // Build the launcher — one .arg() call per kernel parameter.
        self.line("let mut launcher = stream.launch_builder(&func);");
        for f in fields {
            match f.qual {
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::Surface => {
                    self.line(&format!("launcher.arg(&mut self.{});", f.name));
                }
                GpuQual::Const => {
                    if !matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) {
                        // Scalar 'const: passed as a kernel parameter.
                        self.line(&format!("launcher.arg(&self.{});", f.name));
                    }
                    // Array 'const: uploaded to __constant__ memory above, not a parameter.
                }
                GpuQual::Local => {
                    if !matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) {
                        self.line(&format!("launcher.arg(&self.{});", f.name));
                    }
                }
                GpuQual::Sync => {}
            }
        }
        self.line("unsafe { launcher.launch(cfg) }?;");

        // `stream` is already `Arc<CudaStream>` (from `boring_new_stream_with_priority`,
        // which itself already returns that) -- re-wrapping in `Arc::new(stream)` here
        // produced `Arc<Arc<CudaStream>>`, a real E0308 confirmed via `cargo check`.
        self.line("Ok(KernelHandle { inner: self, stream })");
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
                    // Plain `int`/`uint` params render as `isize`/`usize` (see
                    // `narrowed_int_param_type`'s doc) — matching what a general-
                    // spliced caller actually passes; shadowed back to this file's
                    // own `i64`/`u64` convention right after the signature (below).
                    let base = match narrowed_int_param_type(ty) {
                        Some((narrowed, _)) => narrowed.to_string(),
                        None => rust_type(ty),
                    };
                    // Boring's by-ref contract (CLAUDE.md: "Structs, enums, arrays,
                    // dicts, sets — always passed by reference") — this backend
                    // used to always emit these by value, which type-checks fine
                    // in isolation but breaks the moment the caller (now the
                    // general-pipeline splice for every plain function, which
                    // already follows this contract) passes `&x`, and breaks the
                    // callee's OWN body the moment it uses the param a second time
                    // after e.g. a `for` loop consumed it by value (confirmed via a
                    // minimal standalone rustc repro — a real E0382/E0308, not
                    // hypothetical).
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
        // A `'gpu'unified`/`'gpu'global`-qualified return type (e.g. math_gpu.br's
        // `pub req [float]'gpu'unified transpose_gpu(...)`) makes this a GPU-resident-
        // returning function per the general pipeline's OWN unconditional, target-
        // agnostic convention (`transpiler::mod.rs`'s `fn_returns_resident` pre-scan —
        // triggered purely by this return-type annotation, independent of
        // `is_gpu_target`/`gpu_kernels`; confirmed by inspecting a PLAIN `boring build`
        // (no target at all)'s own output for this same function, which shows the
        // identical `Result<BoringGpuArg<f64>, _>` return type). Every "plain" (non-
        // kernel-touching) caller of THIS function was rendered by that same general
        // pipeline (see `cuda::transpile_cuda`'s doc comment) and unconditionally
        // expects a `BoringGpuArg<T>` value back — this backend has no cross-function
        // GPU-buffer-residency optimization of its own, so it always returns the
        // `Host(...)` variant (see the module-level `BoringGpuArg` doc, `emit_prelude`).
        let resident_elem = f.return_ty.as_ref().and_then(|t| t.gpu_resident_qual().map(|_| elem_rust_type(t)));
        let plain_ret = match &resident_elem {
            Some(elem) => format!("BoringGpuArg<{}>", elem),
            None => f.return_ty.as_ref().map(rust_type).unwrap_or_else(|| "()".into()),
        };
        // `throws` → `Result<T, Box<dyn std::error::Error + Send + Sync>>`, matching this backend's
        // existing convention for fallible code (kernel constructors, `main()` — see
        // `emit_kernel_new`/`emit_program`). Previously ignored entirely: every
        // `throws` function was emitted with its plain (non-Result) return type, and
        // `throw`/`guard ... else throw` inside it fell to the catch-all
        // `/* unsupported stmt */` placeholder (see `Stmt::Throw`/`Stmt::Guard` above).
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
        let outer_in_resident_return = self.in_resident_return;
        self.in_resident_return = resident_elem.is_some();
        let outer_ref_params = std::mem::take(&mut self.ref_params);
        for p in &f.params {
            if let Some(ty) = &p.ty {
                if is_ref_worthy_type(ty, &self.struct_names) {
                    self.ref_params.insert(p.name.clone());
                }
            }
        }
        // Shadow-rebind narrowed int/uint params back to this file's own i64/u64
        // convention (see `narrowed_int_param_type`'s doc) — a plain `let x = x
        // as i64;` is valid Rust shadowing, so every other line of this
        // function's body needs no further changes.
        for p in &f.params {
            if let Some(ty) = &p.ty {
                if let Some((_, existing)) = narrowed_int_param_type(ty) {
                    self.line(&format!("let {name} = {name} as {existing};", name = p.name, existing = existing));
                }
            }
        }
        let len = f.body.len();
        for (i, stmt) in f.body.iter().enumerate() {
            if i + 1 == len {
                // A throws function's tail expression is its `Ok(...)` value (any
                // early exit already went through `Stmt::Return`/`Stmt::Throw` above).
                // Resident-returning: wrap in `BoringGpuArg::Host(...)` too (see
                // `resident_elem`'s doc comment above).
                if f.throws {
                    if let Stmt::Expr(e) = stmt {
                        let s = self.expr(e);
                        let wrapped = if self.in_resident_return { format!("BoringGpuArg::Host({})", s) } else { s };
                        self.line(&format!("Ok({})", wrapped));
                        continue;
                    }
                } else if self.in_resident_return {
                    if let Stmt::Expr(e) = stmt {
                        let s = self.expr(e);
                        self.line(&format!("BoringGpuArg::Host({})", s));
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
        self.indent -= 1;
        self.line("}");
    }

    // ── Statements ─────────────────────────────────────────────────────────────

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(s) => {
                let binding = if s.binding.is_mutable() { "let mut" } else { "let" };
                let ty_ann = s.ty.as_ref().map(|t| format!(": {}", rust_type(t))).unwrap_or_default();
                if let Some(val) = &s.value {
                    // Track kernel struct type for this variable so field accesses
                    // on it can be redirected to read_<field>() D2H calls.
                    self.track_kernel_var(&s.name, val);
                    self.track_dict_var(&s.name, s.ty.as_ref(), Some(val));
                    let rhs = self.expr(val);
                    self.line(&format!("{} {}{} = {};", binding, s.name, ty_ann, rhs));
                } else {
                    self.track_dict_var(&s.name, s.ty.as_ref(), None);
                    self.line(&format!("{} {}{};", binding, s.name, ty_ann));
                }
            }
            // `let (a, b, c) = expr` — Rust supports tuple-destructuring `let` natively,
            // so this is a direct translation; no GPU-resident special-casing (the
            // general transpiler's `emit_let_destructure` does more here) is needed
            // for any call site in this codebase's CUDA/Metal-targeted functions —
            // e.g. decoder.br's `let (sa, new_sa_k, new_sa_v) = mha_step_gpu(...)`.
            // Was previously unhandled entirely, falling to this function's catch-all
            // `/* unsupported stmt */` default.
            Stmt::LetDestructure(s) => {
                let binding = if s.binding.is_mutable() { "let mut" } else { "let" };
                let names: Vec<String> = s.bindings.iter().map(|b| b.name.clone()).collect();
                let rhs = self.expr(&s.value);
                self.line(&format!("{} ({}) = {};", binding, names.join(", "), rhs));
            }
            // Non-tail-position `match` — its value is discarded (see `emit_stmt_last`
            // for the tail case, which keeps the value as the function's return).
            Stmt::Match(m) => {
                let s = self.emit_match_expr(m);
                self.line(&format!("{};", s));
            }
            Stmt::Expr(e) => {
                match &e.kind {
                    ExprKind::Assign(lhs, rhs) => {
                        if let ExprKind::Index(obj, idx) = &lhs.kind {
                            // `dict[key] = v` / `self.field[key] = v` → HashMap::insert,
                            // not Vec-style `[key as usize] = v` (a real compile error for
                            // a non-integer key, e.g. tokenizer.br's `vocab[key] = id`).
                            if self.is_dict_obj(obj) {
                                let obj_s = self.expr(obj);
                                let idx_s = self.expr(idx);
                                let rhs_s = self.expr(rhs);
                                self.line(&format!("{}.insert(({}).clone(), ({}).clone());", obj_s, idx_s, rhs_s));
                                return;
                            }
                            // Plain array index assignment (`arr[i] = v`) — built
                            // directly rather than via `self.expr(lhs)`, which would
                            // route through the `Index` READ case (now appending
                            // `.clone()` for by-ref-safe reads, see that case's doc) —
                            // an assignment TARGET needs the bare lvalue index, not a
                            // cloned rvalue (`arr[i].clone() = v` doesn't compile).
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
                    let s = self.expr(val);
                    let s = if self.in_resident_return { format!("BoringGpuArg::Host({})", s) } else { s };
                    if self.in_throws { self.line(&format!("return Ok({});", s)); }
                    else { self.line(&format!("return {};", s)); }
                } else if self.in_throws {
                    self.line("return Ok(());");
                } else {
                    self.line("return;");
                }
            }
            // `guard <cond> else throw "..."` — only the `GuardCond::Expr` form is used
            // anywhere in the real corpus this backend was built against (see
            // `cuda::host`'s module doc); `GuardCond::Clauses` (`guard let x = ...`)
            // falls back to the catch-all placeholder rather than silently mishandling it.
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
            // `throw "msg"` — only valid inside a `throws` function (checker-enforced),
            // so `self.in_throws` is always true here in practice; the non-throws arm is
            // a defensive fallback (matches the general transpiler's own `panic!` choice
            // for a throw outside a `Result`-returning function).
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
                let c = self.expr(&w.condition);
                self.line(&format!("while {} {{", c));
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
                        let range = if *inclusive { format!("{}..={}", lo, hi) }
                                    else          { format!("{}..{}", lo, hi) };
                        self.line(&format!("for {} in {} {{", var, range));
                    }
                    _ => {
                        let iter = self.expr(&f.iterable);
                        // `.iter().cloned()` regardless of whether `iter` is an
                        // owned `Vec<T>` or one of this function's own by-ref
                        // params (`&Vec<T>`, see `ref_params`) -- works uniformly
                        // either way and produces owned `T` loop variables, matching
                        // what a boring `for v in arr:` body expects (e.g.
                        // `if v > mx: mx = v` needs `v: f64`, not `&f64`). Previously
                        // a bare `for v in x { ... }` — fine for an owned `Vec`
                        // consumed exactly once, but a real E0382/E0308 the moment
                        // `x` is used again afterward or is itself a `&Vec<T>` param.
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
                    // `k(block = N)` inside kernel: — desugar to k.__boring_launch(...)
                    if let Some(launch) = self.try_emit_kernel_launch_call(e) {
                        self.line(&format!("{launch};"));
                    } else {
                        let s = self.expr(e);
                        self.line(&format!("{s};"));
                    }
                }
                Stmt::Loop(l) => {
                    self.line("loop {");
                    self.indent += 1;
                    for s in &l.body { self.emit_stmt(s); }
                    self.indent -= 1;
                    self.line("}");
                }
                other => self.emit_stmt(other),
            }
        }
    }

    /// If `expr` is `k(block = N[, after = [...]])` where `k` is a tracked kernel
    /// variable, return the `__boring_launch(...)` call string. Otherwise `None`.
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
        let after_arg = match args.iter().find(|a| a.label.as_deref() == Some("after")) {
            None => "&[]".into(),
            Some(a) => match &a.value.kind {
                ExprKind::Array(elems) => {
                    let refs: Vec<String> = elems.iter()
                        .map(|e| format!("&{}.stream", self.expr(e)))
                        .collect();
                    format!("&[{}]", refs.join(", "))
                }
                _ => { let s = self.expr(&a.value); format!("&[&{s}.stream]") }
            },
        };
        let grid: String = if auto_grid { "None".into() } else { "(1, 1, 1)".into() };
        let priority_arg: String = match args.iter().find(|a| a.label.as_deref() == Some("priority")) {
            None => "0i32".into(),
            Some(a) => match &a.value.kind {
                ExprKind::Str(s) => match s.as_str() {
                    "high"   => "-1i32".into(),
                    "low"    =>  "1i32".into(),
                    _        =>  "0i32".into(),
                },
                _ => "0i32".into(),
            },
        };
        // Re-assign so the moved-into-launch value is returned to the variable.
        // `KernelHandle::wait(self) -> Result<T, _>` -- the trailing `?` was
        // missing, a real E0308 ("expected `T`, found `Result<T, _>`")
        // confirmed via `cargo check`.
        Some(format!("{var_name} = {var_name}.__boring_launch({block}, {grid}, {after_arg}, {priority_arg})?.wait()?"))
    }

    fn emit_stmt_last(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr(e) => { let s = self.expr(e); self.line(&s); }
            // No trailing `;` — same as the `Stmt::Expr` tail case above, this
            // match's value IS the function's return value (main.br's
            // `size_from_str`, model.br's `ModelConfig::for_size`).
            Stmt::Match(m) => { let s = self.emit_match_expr(m); self.line(&s); }
            _ => self.emit_stmt(stmt),
        }
    }

    /// `match <subject>: <arm> ...` — only the shapes this codebase's
    /// CUDA/Metal-targeted `.br` files actually use are handled: a bare literal or
    /// enum-variant subject with no bindings/guards (`main.br`'s `size_from_str`,
    /// `model.br`'s `ModelConfig::for_size`). Was previously unhandled entirely,
    /// falling to `emit_stmt`'s catch-all `/* unsupported stmt */` default.
    fn emit_match_expr(&mut self, m: &MatchStmt) -> String {
        // String-typed literal patterns need `.as_str()` on the subject —
        // matching `&str` literals against a `String` value directly is a type
        // error in Rust (deref coercion doesn't apply inside a match pattern).
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

    /// Render a Boring match pattern as Rust. See `emit_match_expr`'s doc for the
    /// scope this covers (no `Pattern::Tuple`/`Pattern::Some` nesting is exercised
    /// by any call site here, but they're implemented anyway since they're cheap
    /// and directly analogous).
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

    /// Emit `stmts` through a fresh sub-emitter sharing this one's tracking maps,
    /// returning the rendered body as a single Rust block-expression string
    /// (`{ ... }`). Shared by `ExprKind::Closure`'s `ClosureBody::Block` case and
    /// `emit_match_expr`'s `MatchBody::Block` case.
    fn emit_sub_block(&mut self, stmts: &[Stmt]) -> String {
        let mut sub = HostEmitter {
            out: String::new(),
            indent: 0,
            kernel_names: self.kernel_names.clone(),
            kernel_decls: self.kernel_decls.clone(),
            var_kernel_type: self.var_kernel_type.clone(),
            gpu_vars: self.gpu_vars.clone(),
            dict_vars: self.dict_vars.clone(),
            dict_fields: self.dict_fields.clone(),
            fn_throws: self.fn_throws.clone(),
            in_throws: self.in_throws,
            variant_to_enum: self.variant_to_enum.clone(),
            fn_ref_params: self.fn_ref_params.clone(),
            struct_names: self.struct_names.clone(),
            ref_params: self.ref_params.clone(),
            in_resident_return: self.in_resident_return,
            fn_narrowed_int_params: self.fn_narrowed_int_params.clone(),
            fn_returns_resident: self.fn_returns_resident.clone(),
        };
        let last = stmts.len().saturating_sub(1);
        for (i, st) in stmts.iter().enumerate() {
            if i == last { sub.emit_stmt_last(st); } else { sub.emit_stmt(st); }
        }
        format!("{{ {} }}", sub.out.trim())
    }

    // ── Expressions ────────────────────────────────────────────────────────────

    /// Convert a Boring dim expression (int literal or tuple) to a Rust `(u32,u32,u32)` literal.
    /// Scalar `N` → `(N as u32, 1, 1)`.  Tuple `(X, Y)` → `(X as u32, Y as u32, 1)`.  Etc.
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

            ExprKind::Var(name) => name.clone(),

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
                // `k.buf[i]` where buf is a GPU array field → `k.read_buf()?[i as usize]`
                if let ExprKind::Field(obj, field) = &arr.kind {
                    if let ExprKind::Var(obj_name) = &obj.kind {
                        if let Some(read_call) = self.try_gpu_field_read(obj_name, field) {
                            let i = self.expr(idx);
                            return format!("{}[{} as usize]", read_call, i);
                        }
                    }
                }
                // Dict-typed receiver (`vocab[key]`, `self.vocab[key]`) → HashMap::get,
                // not Vec-style index. Every real use here is paired with `else <default>`
                // (tokenizer.br), handled by the existing `Else` case below wrapping
                // whatever this returns in `.unwrap_or(...)` -- an `Option<V>` is exactly
                // what that expects.
                if self.is_dict_obj(arr) {
                    let obj_s = self.expr(arr);
                    let key_s = self.expr(idx);
                    return format!("{}.get(&({})).cloned()", obj_s, key_s);
                }
                // Slice: a[M..N] / a[..N] / a[M..] / a[..] -- a proper Rust range index
                // returning an owned Vec (matches how a sliced array is always consumed
                // here: bound to a `let`/`var` of array type -- e.g. math.br's
                // layer_norm_seq). Previously fell through to the plain-index case below,
                // which called `self.expr(idx)` on a `SliceRange` it has no case for,
                // producing `/* expr */` (this function's catch-all default).
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
                // Plain read: `.clone()` after indexing — needed unconditionally
                // (matches the general pipeline's own convention) because indexing
                // ALWAYS borrows in Rust (`Index::index` returns `&T`), whether
                // `arr` is an owned `Vec<T>` or one of this function's own by-ref
                // params (`&Vec<T>`, see `ref_params`). `Stmt::Expr(Assign(..))`'s
                // handler bypasses this case entirely for an assignment TARGET
                // (`arr[i] = v`), where a bare lvalue index — not a cloned rvalue —
                // is required; see that handler.
                format!("{}[{} as usize].clone()", self.expr(arr), self.expr(idx))
            }
            ExprKind::Field(obj, field) => {
                // `k.buf` where buf is a GPU array field → `k.read_buf()?`
                if let ExprKind::Var(obj_name) = &obj.kind {
                    if let Some(read_call) = self.try_gpu_field_read(obj_name, field) {
                        return read_call;
                    }
                    // `EnumName.Variant` (a fieldless-variant construction, e.g.
                    // main.br's `ModelSize.Tiny`) → Rust's path-qualified
                    // `EnumName::Variant`. Was previously unhandled, falling through
                    // to the plain `{obj}.{field}` case below and emitting the
                    // invalid `ModelSize.Tiny` verbatim.
                    if self.variant_to_enum.get(field.as_str()) == Some(obj_name) {
                        return format!("{}::{}", obj_name, field);
                    }
                }
                let o = self.expr(obj);
                // `.length` as a field (Boring style) → Rust `.len() as i64`
                if field == "length" || field == "count" {
                    return format!("{}.len() as i64", o);
                }
                format!("{}.{}", o, field)
            }
            ExprKind::Call(callee, args) => {
                // `GPU(n)` → `boring_gpu_ctx_n(n)?`
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "GPU" {
                        let idx = args.first().map(|a| self.expr(&a.value)).unwrap_or_else(|| "0".into());
                        return format!("boring_gpu_ctx_n({} as usize)?", idx);
                    }
                }
                // `ord(c)` — char → i64 (Boring built-in)
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "ord" {
                        if let Some(arg) = args.first() {
                            let inner = self.expr(&arg.value);
                            return format!("({} as i64)", inner);
                        }
                    }
                }
                // `Scale(data)` → `Scale::new(boring_gpu_ctx(), data)?`. Kernel
                // constructors always need OWNED args (this struct's `new()` is
                // the existing, untouched kernel-specific emission) -- if an
                // argument is one of THIS function's own by-ref params (see
                // `ref_params`), clone it back to owned first.
                if let ExprKind::Var(name) = &callee.kind {
                    if self.kernel_names.contains(name.as_str()) {
                        let args_s: Vec<String> = args.iter().map(|a| self.coerce_call_arg(&a.value, false)).collect();
                        let all = std::iter::once("boring_gpu_ctx()".to_string())
                            .chain(args_s)
                            .collect::<Vec<_>>();
                        return format!("{}::new({})?", name, all.join(", "));
                    }
                }
                // `print(expr)` → `println!(...)`
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "print" {
                        if let Some(arg) = args.first() {
                            let inner = self.expr(&arg.value);
                            if inner.starts_with("format!(") {
                                let inner_args = &inner["format!(".len()..inner.len()-1];
                                return format!("println!({})", inner_args);
                            }
                            return format!("println!(\"{{}}\", {})", inner);
                        }
                        return "println!()".into();
                    }
                }
                // `float(expr)` → `expr as f64`
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "float" {
                        if let Some(arg) = args.first() {
                            let inner = self.expr(&arg.value);
                            return format!("({} as f64)", inner);
                        }
                    }
                }
                // Ordinary function call -- coerce each argument to match the
                // callee's own by-ref/owned parameter convention (`fn_ref_params`,
                // populated from every `Item::Fn` in the program, whichever
                // emitter -- this one or the spliced general pipeline -- ends up
                // rendering that callee's body). Boring's own by-ref contract
                // (array/dict/set/struct/enum params always by reference) is
                // enforced by BOTH emitters identically, so a call from a
                // kernel-touching function (this emitter) to a plain function
                // (the general-pipeline splice) or vice versa always lines up.
                let callee_name = if let ExprKind::Var(name) = &callee.kind { Some(name.as_str()) } else { None };
                let ref_flags = callee_name.and_then(|n| self.fn_ref_params.get(n)).cloned();
                // Plain-int/uint positions on the callee also need a cast to
                // match its narrowed `isize`/`usize` param (see
                // `narrowed_int_param_type`'s doc) -- a call from one
                // kernel-touching function to another (both on this emitter,
                // e.g. `attention_heads_gpu` calling `transpose_gpu`) otherwise
                // passes this file's own `i64`/`u64` convention straight
                // through, a real E0308 confirmed via `cargo check`.
                let narrow_flags = callee_name.and_then(|n| self.fn_narrowed_int_params.get(n)).cloned();
                let args_s: Vec<String> = args.iter().enumerate().map(|(i, a)| {
                    let expects_ref = ref_flags.as_ref().and_then(|f| f.get(i).copied()).unwrap_or(false);
                    let s = self.coerce_call_arg(&a.value, expects_ref);
                    match narrow_flags.as_ref().and_then(|f| f.get(i).copied()).flatten() {
                        Some(narrowed) => format!("({}) as {}", s, narrowed),
                        None => s,
                    }
                }).collect();
                let fn_s = self.expr(callee);
                let call = format!("{}({})", fn_s, args_s.join(", "));
                // `?`-propagate a call to another free `throws` function (see
                // `fn_throws`'s doc comment for what this does NOT cover: `throws`
                // struct methods calling each other, e.g. decoder.br/encoder.br).
                let call = if self.in_throws && callee_name.is_some_and(|n| self.fn_throws.contains(n)) {
                    format!("{}?", call)
                } else {
                    call
                };
                // Interprocedural GPU residency: a call from one kernel-touching
                // function to another that returns `BoringGpuArg<T>` (e.g.
                // `attention_heads_gpu` calling `transpose_gpu`) must materialize
                // the result -- this emitter has no cross-function residency-
                // preservation optimization of its own (see `fn_returns_resident`'s
                // doc). A real `?`-operator/E0308 type mismatch otherwise,
                // confirmed via `cargo check`.
                if let Some(ret_ty) = callee_name.and_then(|n| self.fn_returns_resident.get(n)).cloned() {
                    let elem = elem_rust_type(&ret_ty);
                    return format!(
                        "match {call} {{ BoringGpuArg::Resident(buf, _) => __boring_gpu_copy_d2h::<f32>(&__boring_gpu_device(), &__boring_gpu_queue(), &buf).iter().map(|&x| x as {elem}).collect::<Vec<{elem}>>(), BoringGpuArg::Host(v) => v }}"
                    );
                }
                call
            }
            ExprKind::MethodCall(obj, method, args) => {
                // `GPU.all()` → iterator over all CUDA devices.
                if let ExprKind::Var(name) = &obj.kind {
                    if name == "GPU" && method == "all" {
                        return "(0..CudaContext::device_count()? as usize).map(|i| boring_gpu_ctx_n(i).unwrap())".into();
                    }
                    // GPU property methods on a GPU variable.
                    if self.gpu_vars.contains(name.as_str()) {
                        return self.emit_gpu_property(name, method);
                    }
                    // `fs.writeBytes(path, bytes)` — write Vec<i64> as binary file
                    if name == "fs" && method == "writeBytes" {
                        let path  = args.first().map(|a| self.expr(&a.value)).unwrap_or_default();
                        let bytes = args.get(1).map(|a| self.expr(&a.value)).unwrap_or_default();
                        return format!("std::fs::write({}, {}.iter().map(|&b| b as u8).collect::<Vec<u8>>())?", path, bytes);
                    }
                    if name == "fs" && (method == "write" || method == "writeText") {
                        let path = args.first().map(|a| self.expr(&a.value)).unwrap_or_default();
                        let text = args.get(1).map(|a| self.expr(&a.value)).unwrap_or_default();
                        return format!("std::fs::write({}, {}.as_bytes())?", path, text);
                    }
                }
                let o = self.expr(obj);
                let args_s: Vec<String> = args.iter().map(|a| self.expr(&a.value)).collect();
                match method.as_str() {
                    "wait" => format!("{}.wait()?", o),
                    "done" => format!("{}.done()", o),
                    // `.chars()` — collect to Vec<char> so indexing and .len() work
                    "chars" if args.is_empty() => format!("{}.chars().collect::<Vec<char>>()", o),
                    // `.length` / `.count` — map to .len() as i64
                    "length" | "count" if args.is_empty() => format!("{}.len() as i64", o),
                    // `.add(x)` / `.insert(x)` — Vec push
                    "add" | "insert" if args.len() == 1 => format!("{}.push({})", o, args_s[0]),
                    // `.map(closure)` on an array isn't a real `Vec` method — go through
                    // iter/cloned/collect, matching the general (std/wgpu) transpiler's
                    // array-method fallback. `args_s[0]` is already a rendered `|v| ...`
                    // closure (see the `ExprKind::Closure` case below).
                    "map" if args_s.len() == 1 && matches!(&args[0].value.kind, ExprKind::Closure(..)) =>
                        format!("{}.iter().cloned().map({}).collect::<Vec<_>>()", o, args_s[0]),
                    // `.sum()` likewise isn't a real `Vec` method. Every call site in this
                    // codebase sums a float array; a hardcoded `f64` turbofish would be
                    // wrong for an int array, but no such call exists here today (see
                    // `cuda::host`'s module doc for the same simplification on `Pipe`).
                    "sum" if args.is_empty() => format!("{}.iter().cloned().sum::<f64>()", o),
                    _ => format!("{}.{}({})", o, method, args_s.join(", ")),
                }
            }
            ExprKind::Pipe(lhs, method, args) => {
                let l = self.expr(lhs);
                let args_s: Vec<String> = args.iter().map(|a| self.expr(&a.value)).collect();
                match method.as_str() {
                    "wait" => format!("{}.wait()?", l),
                    "done" => format!("{}.done()", l),
                    // `x |> map((v): ...)` — see the identical `MethodCall` case above.
                    "map" if args_s.len() == 1 && matches!(&args[0].value.kind, ExprKind::Closure(..)) =>
                        format!("{}.iter().cloned().map({}).collect::<Vec<_>>()", l, args_s[0]),
                    // `x |> sum()` — see the identical `MethodCall` case above.
                    "sum" if args.is_empty() => format!("{}.iter().cloned().sum::<f64>()", l),
                    _ => format!("{}.{}({})", l, method, args_s.join(", ")),
                }
            }
            ExprKind::KernelLaunch { config, kernel } => {
                // Does the target kernel use automatic grid sizing?
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
                // `after = [h1, h2]` — collect stream references from dependency handles.
                // Each handle's `.stream` field is passed as a dependency to __boring_launch.
                let after_arg = match &config.after {
                    None => "&[]".into(),
                    Some(after_expr) => {
                        // Emit a slice literal of stream references from the dependency handles.
                        match &after_expr.kind {
                            ExprKind::Array(elems) => {
                                let refs: Vec<String> = elems.iter()
                                    .map(|e| format!("&{}.stream", self.expr(e)))
                                    .collect();
                                format!("&[{}]", refs.join(", "))
                            }
                            _ => {
                                let s = self.expr(after_expr);
                                format!("&[&{}.stream]", s)
                            }
                        }
                    }
                };
                format!("{k}.__boring_launch({block}, {grid}, {after_arg})?")
            }
            ExprKind::New { arena, ctor } => {
                // `new(g) Scale(data)` → `Scale::new(<g>, data)?` (explicit device placement).
                if let ExprKind::Call(callee, args) = &ctor.kind {
                    if let ExprKind::Var(name) = &callee.kind {
                        if self.kernel_names.contains(name.as_str()) {
                            let dev = arena.as_ref()
                                .map(|a| self.expr(a))
                                .unwrap_or_else(|| "boring_gpu_ctx()".into());
                            let args_s: Vec<String> = args.iter().map(|a| self.expr(&a.value)).collect();
                            let all = std::iter::once(dev).chain(args_s).collect::<Vec<_>>();
                            return format!("{}::new({})?", name, all.join(", "));
                        }
                    }
                }
                // Non-kernel new: ignore arena, emit the constructor directly.
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
            ExprKind::ArrayComp { expr, var, count } => {
                let n = self.expr(count);
                let body = self.expr(expr);
                format!(
                    "(0..({} as usize)).map(|__boring_i| {{ let {} = __boring_i as i64; {} }}).collect::<Vec<_>>()",
                    n, var, body
                )
            }
            ExprKind::ArrayCompIter { expr, var, iter } => {
                let it = self.expr(iter);
                let body = self.expr(expr);
                format!("{}.iter().map(|{}| {{ {} }}).collect::<Vec<_>>()", it, var, body)
            }
            ExprKind::Tuple(elems) => {
                let s: Vec<String> = elems.iter().map(|e| self.expr(e)).collect();
                format!("({})", s.join(", "))
            }
            // A boring lambda, e.g. `(v): gelu_f(v)` — used as a `.map()`/pipe argument
            // (see the `map`/`sum` cases above). Was previously unhandled, falling to
            // this function's catch-all `/* expr */` default — the exact bug this fixes
            // (`math.br`'s `gelu`/`softmax` emitted `a.map(/* expr */)`).
            ExprKind::Closure(params, _ret_ty, body, _throws, _task) => {
                let ps: Vec<String> = params.iter().map(|p| p.name.clone()).collect();
                let body_s = match body {
                    ClosureBody::Expr(be) => self.expr(be),
                    ClosureBody::Block(stmts) => self.emit_sub_block(stmts),
                };
                format!("|{}| {}", ps.join(", "), body_s)
            }
            ExprKind::Dict(pairs) => {
                let s: Vec<String> = pairs.iter().map(|(k, v)| {
                    format!("({}, {})", self.expr(k), self.expr(v))
                }).collect();
                format!("[{}].into_iter().collect::<std::collections::HashMap<_,_>>()", s.join(", "))
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
                let mut args = Vec::new();
                for seg in segs {
                    match seg {
                        StringSegment::Lit(s) => fmt_str.push_str(s),
                        StringSegment::Expr(e) => {
                            fmt_str.push_str("{}");
                            args.push(self.expr(e));
                        }
                        StringSegment::FormattedExpr(e, fmt) => {
                            let rust_fmt = fmt.trim_end_matches(['f', 'd', 's', 'g', 'G']);
                            fmt_str.push_str(&format!("{{:{}}}", rust_fmt));
                            args.push(self.expr(e));
                        }
                    }
                }
                fmt_str.push('"');
                parts.push(fmt_str);
                parts.extend(args);
                format!("format!({})", parts.join(", "))
            }
            _ => "/* expr */".into(),
        }
    }

    /// Record that variable `name` holds a kernel struct instance.
    ///
    /// Handles three patterns:
    /// - `Scale(args)` call  → var is type `Scale`
    /// - `kernel(...) k`     → var has same type as `k`
    /// - `... |> .wait`      → var has same type as the kernel being waited on
    fn track_kernel_var(&mut self, name: &str, val: &Expr) {
        let ty = self.resolve_kernel_type(val);
        if let Some(t) = ty {
            self.var_kernel_type.insert(name.to_string(), t);
        }
        // Track GPU device variables: `GPU(n)` or `boring_gpu_ctx_n(...)`.
        if self.is_gpu_expr(val) {
            self.gpu_vars.insert(name.to_string());
        }
    }

    fn is_gpu_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(n) = &callee.kind { n == "GPU" }
                else { false }
            }
            ExprKind::Var(n) => self.gpu_vars.contains(n.as_str()),
            _ => false,
        }
    }

    /// Map a GPU variable method call to cudarc 0.19 API.
    fn emit_gpu_property(&self, var: &str, method: &str) -> String {
        match method {
            "name"                => format!("{var}.name()?"),
            "totalMem"            => format!("{var}.total_mem()?"),
            "freeMem"             => format!("{var}.mem_get_info()?.0"),
            "computeCapability"   => format!("{var}.compute_capability()?"),
            "warpSize"            => format!("{var}.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)?"),
            "maxThreads"          => format!("{var}.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?"),
            "maxSharedMem"        => format!("{var}.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?"),
            "index"               => format!("{var}.ordinal()"),
            other                 => format!("{var}.{other}()"),
        }
    }

    fn resolve_kernel_type(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            // `Scale(args)` — direct constructor
            ExprKind::Call(callee, _) => {
                if let ExprKind::Var(n) = &callee.kind {
                    if self.kernel_names.contains(n.as_str()) { return Some(n.clone()); }
                }
                None
            }
            // `kernel(...) k` — launch; same type as the kernel variable
            ExprKind::KernelLaunch { kernel, .. } => self.resolve_kernel_type(kernel),
            // `new(g) Scale(args)` — placement; same type as the constructor
            ExprKind::New { ctor, .. } => self.resolve_kernel_type(ctor),
            // `expr |> .wait` — unwrap handle; same type as the launch
            ExprKind::Pipe(lhs, method, _) if method == "wait" => self.resolve_kernel_type(lhs),
            // bare variable — look up in tracked map
            ExprKind::Var(n) => self.var_kernel_type.get(n.as_str()).cloned(),
            _ => None,
        }
    }

    /// Check whether `obj.field` should be rewritten to a D2H read call.
    ///
    /// Returns `Some("obj.read_field()?")` when `obj` is a tracked kernel
    /// variable and `field` is a `'unified` or `'global` array field.
    /// Returns `None` for scalars or non-GPU fields (direct field access is fine).
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
                    _ => None, // scalar — direct access
                }
            }
            _ => None,
        }
    }

    fn host_field_type(&self, field: &KernelFieldDecl) -> String {
        let elem = elem_rust_type(&field.ty);
        // 'const fixed-size arrays are stored as Vec<T> on the host — they are
        // uploaded to __constant__ memory via memcpy_htod, not as CudaSlice args.
        if matches!(field.qual, GpuQual::Const) && matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _)) {
            return format!("Vec<{}>", elem);
        }
        // A scalar `'const` field (e.g. `let int rows`) is a plain kernel-launch
        // parameter, not a device buffer — was previously wrapped in
        // `CudaSlice<{elem}>` unconditionally here (only the array case above was
        // excluded), producing e.g. `rows: CudaSlice<i64>` for a struct whose
        // constructor then assigns it a bare `i64` (see `emit_init_stmt`'s
        // matching fix) — a real E0308 (`expected CudaSlice<i64>, found i64`),
        // confirmed via `cargo check`.
        if matches!(field.qual, GpuQual::Const) && !matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _)) {
            return elem;
        }
        format!("CudaSlice<{}>", elem)
    }
}

// ── Free helpers ──────────────────────────────────────────────────────────────

fn host_param_type(ty: &Type, _fields: &[KernelFieldDecl]) -> String {
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
        Type::Array(inner)     => rust_type(inner),
        Type::ArrayN(inner, _) => rust_type(inner),
        Type::Qualified(inner, _) => elem_rust_type(inner),
        _                      => rust_type(ty),
    }
}

fn rust_type(ty: &Type) -> String {
    match ty {
        Type::Int            => "i64".into(),
        Type::Uint           => "u64".into(),
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
        Type::Float          => "f64".into(),
        Type::Bool           => "bool".into(),
        Type::Str            => "String".into(),
        Type::Nil            => "()".into(),
        Type::Void           => "()".into(),
        Type::Never          => "!".into(),
        Type::Array(inner)   => format!("Vec<{}>", rust_type(inner)),
        Type::ArrayN(inner, n) => format!("[{}; {}]", rust_type(inner), n),
        Type::Tuple(ts) => {
            let s: Vec<String> = ts.iter().map(rust_type).collect();
            format!("({})", s.join(", "))
        }
        Type::Dict(k, v) => format!(
            "std::collections::HashMap<{}, {}>", rust_type(k), rust_type(v)
        ),
        Type::Optional(inner)  => format!("Option<{}>", rust_type(inner)),
        // Named primitives — the kernel field parser stores raw keyword strings.
        Type::Named(n) => match n.as_str() {
            "float" | "f64" => "f64",
            "f32"           => "f32",
            "int"   | "i64" => "i64",
            "uint"  | "u64" => "u64",
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

/// Emit a simple scalar expression as a Rust literal for use as a kernel field default.
fn emit_scalar_default(expr: &Expr) -> String {
    match &expr.kind {
        ExprKind::Int(n)   => n.to_string(),
        ExprKind::Float(f) => {
            let s = format!("{}", f);
            if s.contains('.') { s } else { format!("{}.0", s) }
        }
        ExprKind::Bool(b)  => b.to_string(),
        ExprKind::UnaryOp(crate::ast::UnaryOp::Neg, inner) => match &inner.kind {
            ExprKind::Int(n)   => format!("-{}", n),
            ExprKind::Float(f) => format!("-{}", f),
            _ => "Default::default()".into(),
        },
        _ => "Default::default()".into(),
    }
}

/// Returns the size in bytes of a scalar element type for shared-memory calculations.
fn elem_size_bytes(ty: &Type) -> usize {
    match ty {
        Type::Float                          => 8, // f64 / double
        Type::Int | Type::Uint               => 8, // i64 / u64
        Type::Uint8 | Type::Int8              => 1,
        Type::Int16 | Type::Uint16            => 2,
        Type::Int32 | Type::Uint32            => 4,
        Type::Int64 | Type::Uint64            => 8,
        Type::Int128 | Type::Uint128          => 16,
        Type::Bool                           => 1,
        Type::Named(n) => match n.as_str() {
            "float" | "f64"         => 8,
            "f32"                   => 4,
            "int"   | "i64"         => 8,
            "uint"  | "u64"         => 8,
            "uint8" | "int8"        => 1,
            "int16" | "uint16"      => 2,
            "int32" | "uint32"      => 4,
            "int64" | "uint64"      => 8,
            "int128" | "uint128"    => 16,
            "i32"                   => 4,
            "u32"                   => 4,
            _                       => 8, // conservative default
        },
        Type::Qualified(inner, _)            => elem_size_bytes(inner),
        Type::Array(inner) | Type::ArrayN(inner, _) => elem_size_bytes(inner),
        _                                    => 8,
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
