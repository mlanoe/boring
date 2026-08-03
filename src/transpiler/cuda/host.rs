// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Rust host-side code emitter for the CUDA backend.

use crate::ast::*;
use crate::transpiler::helpers::image_volume_grid_dim_expr;

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
    /// Free function name → its `Type` when GPU-resident-returning (see
    /// `emit_fn`'s `resident_elem`). A call from one kernel-touching function
    /// to another that returns `BoringGpuArg<T>` (e.g. `attention_heads_gpu`
    /// calling `transpose_gpu`) must materialize the result — this emitter has
    /// no cross-function residency-preservation optimization of its own, unlike
    /// the general pipeline's `resident_call_vars` (see `emit_methods.rs`'s
    /// identical-purpose fix there).
    fn_returns_resident: std::collections::HashMap<String, Type>,
    /// Local variable names (current function only, reset per `emit_fn` call)
    /// bound to a call into a `fn_returns_resident` function WHOSE OWN `let`
    /// type is itself `'gpu'unified` (`s.ty.gpu_resident_qual().is_some()`) --
    /// these stay `BoringGpuArg<f64>`-typed (never unwrapped to a plain
    /// `Vec`), so a later kernel-constructor call passing one of these as an
    /// argument can MOVE the underlying `CudaSlice` straight through instead
    /// of reading it back to host and re-uploading. See `Stmt::Let`'s own
    /// handling for where this is populated, and the kernel-constructor
    /// branch of `expr()`'s `Call` case for where it's consumed.
    resident_locals: std::collections::HashSet<String>,
    /// One-shot flag: `Stmt::Let` sets this to `true` immediately before
    /// calling `self.expr(val)` for a call it has determined should stay
    /// `BoringGpuArg`-typed (see `resident_locals`'s doc). The `Call` arm of
    /// `expr()` takes (consumes) this flag for ITS OWN top-level call only,
    /// before recursing into argument sub-expressions, so a resident-
    /// preserving call passed as an argument to this outer call isn't
    /// incorrectly also suppressed.
    suppress_resident_materialize: bool,
    /// Top-level scalar `let`s (boring name -> rendered Rust expression),
    /// collected up front so they can be inlined wherever they're referenced
    /// inside a kernel struct's `new()`/`init()` codegen (`emit_kernel_new`/
    /// `emit_init_stmt`) -- those bodies are emitted into their own `impl`
    /// block, textually and scope-wise separate from `fn main()`, which is
    /// the only place a bare top-level `let n = ...` actually becomes a Rust
    /// local (see this backend's own `top_level_kernel_touching` handling in
    /// `emit_program`, and the general (std/wgpu-shared) pipeline's `let`-to-
    /// local folding for the non-kernel-touching-top-level case) -- so a
    /// kernel `init` referencing it (`result = [0 for ..n]`) previously
    /// emitted a bare `n` identifier with nothing in scope to resolve it, a
    /// real E0425 confirmed via `cargo check`. Mirrors `metal::host`'s
    /// identical `top_level_scalars` field/fix.
    top_level_scalars: std::collections::HashMap<String, String>,
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
            fn_returns_resident: std::collections::HashMap::new(),
            resident_locals: std::collections::HashSet::new(),
            suppress_resident_materialize: false,
            top_level_scalars: std::collections::HashMap::new(),
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
        // See `top_level_scalars`'s doc.
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
                            let is_scalar = crate::transpiler::helpers::is_scalar_let_value(val, s.ty.as_ref());
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
        self.line("__boring_gpu_enable_peer_access(&ctx)?;");
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
        self.line("let ctx = CudaContext::new(idx)?;");
        self.line("__boring_gpu_enable_peer_access(&ctx)?;");
        self.line("Ok(ctx)");
        self.indent -= 1;
        self.line("}");
        self.blank();
        self.line("thread_local! {");
        self.indent += 1;
        self.line("static __BORING_GPU_PEER_CTXS: std::cell::RefCell<Vec<Arc<CudaContext>>> = std::cell::RefCell::new(Vec::new());");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // Automatically enables bidirectional peer access between `ctx` and every
        // other GPU context this program has already created, wherever the real
        // hardware topology actually supports it (checked via
        // `cuDeviceCanAccessPeer` first, per pair, per direction -- a grant is
        // one-directional, confirmed against the CUDA driver API docs). This is
        // the prerequisite `stream.join(dep)` (`cuStreamWaitEvent`) needs to
        // work AT ALL across two different devices -- without it, a cross-
        // device `after =` fails at runtime with a real (if generic) DriverError
        // instead of the `cudaStreamWaitEvent` actually succeeding. A pair the
        // topology doesn't support is silently skipped -- `after =` between
        // that specific pair still surfaces its own real error, unchanged from
        // before this existed; there is deliberately no user-facing flag to opt
        // in or out (see docs/cuda-module.md's "Multi-device" section) since
        // this is cheap and harmless to always attempt: single-GPU programs
        // never have a second context to loop over.
        self.line("fn __boring_gpu_enable_peer_access(ctx: &Arc<CudaContext>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("__BORING_GPU_PEER_CTXS.with(|ctxs| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("let mut ctxs = ctxs.borrow_mut();");
        self.line("for other in ctxs.iter() {");
        self.indent += 1;
        self.line("if Arc::ptr_eq(other, ctx) { continue; }");
        self.line("for (a, b) in [(ctx, other), (other, ctx)] {");
        self.indent += 1;
        self.line("let mut can_access: std::ffi::c_int = 0;");
        self.line("unsafe {");
        self.indent += 1;
        self.line("cudarc::driver::sys::cuDeviceCanAccessPeer(&mut can_access, a.cu_device(), b.cu_device()).result()?;");
        self.indent -= 1;
        self.line("}");
        self.line("if can_access != 0 {");
        self.indent += 1;
        self.line("a.bind_to_thread()?;");
        self.line("let code = unsafe { cudarc::driver::sys::cuCtxEnablePeerAccess(b.cu_ctx(), 0) };");
        self.line("// Already-enabled (e.g. `a`/`b` swapped on a later call) is not an error.");
        self.line("if code != cudarc::driver::sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {");
        self.indent += 1;
        self.line("code.result()?;");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.indent -= 1;
        self.line("}");
        self.line("ctxs.push(Arc::clone(ctx));");
        self.line("Ok(())");
        self.indent -= 1;
        self.line("})");
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
        // `Resident`'s buffer used to be `Arc<Vec<f32>>` -- host memory, not a
        // device buffer at all, because this backend never actually
        // constructed the variant (see the git history of this comment for
        // the old rationale). It now holds a real `CudaSlice<f64>` (device
        // memory -- CUDA kernel structs store `f64` natively, unlike Metal's
        // MSL-forced `f32`; confirmed via a real `cargo check` against a
        // stubbed-nvcc build, an actual E0308 before this fix):
        // `emit_fn`'s tail-expression codegen constructs this directly from
        // a kernel struct's own output buffer, by MOVING it out (not
        // cloning) -- safe because the only two shapes that ever reach this
        // (a bare `k.field` tail expression, or `k.field` passed directly as
        // another kernel constructor's argument) both mean `k` is never
        // referenced again afterward. This matters more here than on the
        // Metal backend: `CudaSlice::clone()` is a REAL device-to-device
        // `memcpy` (cudarc's own `try_clone`/`clone_dtod`), unlike Metal's
        // `Buffer::clone()` (a cheap ObjC retain) -- so the `Clone` impl
        // below is a correctness-preserving fallback for the rare case
        // something needs the SAME resident value twice, not the common path.
        self.line("#[allow(dead_code)]");
        self.line("enum BoringGpuArg<T> {");
        self.indent += 1;
        self.line("Resident(CudaSlice<f64>, usize),");
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
        self.line("#[allow(dead_code)] fn __boring_gpu_device() -> Arc<CudaContext> { boring_gpu_ctx() }");
        self.line("#[allow(dead_code)] fn __boring_gpu_queue() -> Arc<CudaContext> { boring_gpu_ctx() }");
        // `T`/`_queue` are unused (kept only so call sites' fixed
        // `::<f32>(&__boring_gpu_device(), &__boring_gpu_queue(), &buf)` shape,
        // shared verbatim with the wgpu/Metal backends by the general
        // pipeline's own call-site templates, still parses -- an unused
        // generic type param on a free fn is legal Rust regardless of what
        // concrete type the turbofish supplies) -- the actual D2H copy always
        // goes through the SAME persistent, priority-0 stream every kernel
        // dispatch shares (see `boring_new_stream_with_priority`'s doc), so a
        // later read is correctly ordered after an earlier write with no
        // separate stream-identity bookkeeping needed. Must stay infallible
        // (`Vec<f64>`, not `Result<..>`) -- the call site (`emit_kernel.rs`'s
        // shared materializing match) uses the result directly with no `?`.
        self.line("#[allow(dead_code)]");
        self.line("fn __boring_gpu_copy_d2h<T>(device: &Arc<CudaContext>, _queue: &Arc<CudaContext>, buf: &CudaSlice<f64>) -> Vec<f64> {");
        self.indent += 1;
        self.line("let stream = boring_new_stream_with_priority(device, 0).expect(\"cuda: failed to get shared stream for D2H copy\");");
        self.line("let v = stream.clone_dtoh(buf).expect(\"cuda: D2H copy failed\");");
        self.line("stream.synchronize().expect(\"cuda: stream sync failed after D2H copy\");");
        self.line("v");
        self.indent -= 1;
        self.line("}");
        self.line("#[allow(dead_code)]");
        self.line("fn __boring_gpu_copy_h2d<T>(_device: &Arc<CudaContext>, _queue: &Arc<CudaContext>, _src: &[u8], _dst: &CudaSlice<f64>) {");
        self.indent += 1;
        self.line("unreachable!(\"cuda backend never constructs a host-to-device upload through this path -- kernel-constructor call sites upload directly\")");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // Each handle owns the CUDA stream its kernel was launched on, so `.wait`
        // synchronizes only that stream (not the whole device).  `after = [..]`
        // dependencies are wired GPU-side via stream ordering, no CPU sync.
        self.line("#[must_use = \"a KernelHandle must be waited on (.wait/.inner) or the launch may not be synchronized\"]");
        self.line("struct KernelHandle<T> { inner: T, stream: Arc<CudaStream> }");
        self.line("impl<T> KernelHandle<T> {");
        self.indent += 1;
        self.line("fn wait(self) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("self.stream.synchronize().map_err(__boring_cuda_classify_error)?;");
        self.line("Ok(self.inner)");
        self.indent -= 1;
        self.line("}");
        self.line("fn done(&self) -> bool { true }");
        self.indent -= 1;
        self.line("}");
        self.blank();
        // cudarc's own `DriverError` Display already calls
        // `cuGetErrorName`/`cuGetErrorString` (confirmed against real cudarc
        // 0.19.8 source), so this doesn't replace that message -- it adds a
        // short category prefix classified from the real `CUresult` code,
        // for the handful of failure classes a caller most often cares about
        // at a glance (out of memory, illegal access, timeout, ...), applied
        // at kernel launch -- the one place a dispatch can actually fail.
        self.line("fn __boring_cuda_classify_error(e: cudarc::driver::DriverError) -> Box<dyn std::error::Error + Send + Sync> {");
        self.indent += 1;
        self.line("use cudarc::driver::sys::CUresult;");
        self.line("let category = match e.0 {");
        self.indent += 1;
        self.line("CUresult::CUDA_ERROR_OUT_OF_MEMORY => \"GPU out of memory\",");
        self.line("CUresult::CUDA_ERROR_ILLEGAL_ADDRESS => \"GPU illegal memory access\",");
        // The real, CUDA-driver-side rejection an oversized `block =` hits:
        // `cuLaunchKernel` returns CUDA_ERROR_INVALID_VALUE when the requested
        // block dimensions exceed the device's (or this specific kernel's)
        // max threads per block -- Boring deliberately does not duplicate
        // that check at compile time or in the interpreter (no validator/
        // eval_gpu check exists, and this doesn't add one); it defers
        // entirely to this real, already-classified runtime rejection.
        self.line("CUresult::CUDA_ERROR_INVALID_VALUE => \"GPU launch configuration invalid (e.g. block size exceeds device limits)\",");
        self.line("CUresult::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => \"GPU launch out of resources\",");
        self.line("CUresult::CUDA_ERROR_LAUNCH_TIMEOUT => \"GPU operation timed out\",");
        self.line("CUresult::CUDA_ERROR_HARDWARE_STACK_ERROR => \"GPU stack overflow\",");
        self.line("CUresult::CUDA_ERROR_ECC_UNCORRECTABLE | CUresult::CUDA_ERROR_CONTEXT_IS_DESTROYED => \"GPU device lost\",");
        self.line("_ => \"GPU kernel launch failed\",");
        self.indent -= 1;
        self.line("};");
        self.line("format!(\"{}: {}\", category, e).into()");
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
        //
        // A fresh stream used to be forked on EVERY single dispatch (this
        // function used to always allocate) -- besides losing the benefit of
        // FIFO single-stream ordering (every dispatch synchronized against
        // its OWN one-shot stream instead), cudarc's own `CudaContext::new_stream`
        // internally does a full-context `synchronize()` whenever its stream
        // count cycles 0→1 (`num_streams.fetch_add`'s guard, cudarc's
        // `core.rs`) -- and with the old eager-`.wait()` desugar dropping
        // each `KernelHandle` (and its sole `Arc<CudaStream>`) right after use,
        // that count cycled 1→0→1 on literally every dispatch, silently
        // paying a SECOND full-context blocking sync on top of the per-
        // dispatch `.wait()`. Caching one persistent stream per priority here
        // (an extra `Arc` keeping the count ≥1 forever after the first call)
        // fixes both at once: dispatches on the same priority now share one
        // real FIFO-ordered stream (matching the Metal backend's shared-
        // command-queue fix), and the guard-resync can never re-trigger.
        self.line("thread_local! {");
        self.indent += 1;
        self.line("static __BORING_CUDA_STREAMS: std::cell::RefCell<std::collections::HashMap<i32, Arc<CudaStream>>> = std::cell::RefCell::new(std::collections::HashMap::new());");
        self.indent -= 1;
        self.line("}");
        self.line("fn boring_new_stream_with_priority(ctx: &Arc<CudaContext>, priority: i32) -> Result<Arc<CudaStream>, Box<dyn std::error::Error + Send + Sync>> {");
        self.indent += 1;
        self.line("if let Some(s) = __BORING_CUDA_STREAMS.with(|c| c.borrow().get(&priority).cloned()) {");
        self.indent += 1;
        self.line("return Ok(s);");
        self.indent -= 1;
        self.line("}");
        self.line("let s = if priority == 0 { ctx.new_stream()? } else { ctx.new_stream_with_priority(priority)? };");
        self.line("__BORING_CUDA_STREAMS.with(|c| c.borrow_mut().insert(priority, Arc::clone(&s)));");
        self.line("Ok(s)");
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
                GpuQual::Actor | GpuQual::Local => {
                    // Block SRAM / registers — no host-side storage.
                }
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Const | GpuQual::Surface => {
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

        // Accessors for 'unified/'actor'unified fields (D2H) — both are host-visible.
        for field in &decl.fields {
            if matches!(field.qual, GpuQual::Unified | GpuQual::ActorUnified) {
                let elem = elem_rust_type(&field.ty);
                self.line(&format!(
                    "fn read_{}(&self) -> Result<Vec<{}>, Box<dyn std::error::Error + Send + Sync>> {{",
                    field.name, elem
                ));
                self.indent += 1;
                // `clone_dtoh` issues an async `memcpy_dtoh_async` on
                // `self.__stream` -- cudarc's own per-`CudaSlice` event
                // tracking already GPU-side-orders this copy after whatever
                // kernel last wrote `self.{field}` (confirmed via cudarc
                // source: `device_ptr()` auto-waits on the slice's tracked
                // write event), regardless of which stream did the writing.
                // What ISN'T guaranteed by cudarc's own Rust types is that
                // the returned host `Vec`'s memory is actually populated by
                // the time this function returns (a plain `Vec` destination
                // takes cudarc's synchronous-copy-into-pageable-memory
                // assumption on faith, unlike its own `PinnedHostSlice` API,
                // which explicitly syncs before trusting its data) -- so
                // `self.__stream.synchronize()` afterward is cheap defense in
                // depth: it can only wait on real, already-in-flight work on
                // the exact stream this copy itself was issued on.
                self.line(&format!("let v = self.__stream.clone_dtoh(&self.{}).map_err(__boring_cuda_classify_error)?;", field.name));
                self.line("self.__stream.synchronize().map_err(__boring_cuda_classify_error)?;");
                self.line("Ok(v)");
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
        let buffer_flags = self.kernel_ctor_buffer_flags(name).unwrap_or_default();
        let params: Vec<String> = init.params.iter().enumerate().map(|(i, p)| {
            // A buffer-passthrough param (see `kernel_ctor_buffer_flags`) takes
            // an already-built `CudaSlice` directly -- the call site is
            // responsible for either MOVING a resident one through or
            // uploading a fresh one from host data via `clone_htod`, instead
            // of this constructor doing the upload itself (see
            // `emit_init_stmt`'s matching change).
            let ty = if let Some(Some(elem)) = buffer_flags.get(i) {
                format!("CudaSlice<{}>", elem)
            } else {
                p.ty.as_ref().map(|t| host_param_type(t, fields)).unwrap_or_else(|| "()".into())
            };
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
                    GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Const | GpuQual::Surface => {
                        let elem = elem_rust_type(&field.ty);
                        self.line(&format!(
                            "let {} = __ctx.default_stream().alloc_zeros::<{}>(1)?;",
                            field.name, elem
                        ));
                    }
                    GpuQual::Actor | GpuQual::Local => {
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
                GpuQual::Actor => {} // no host field
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
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Const | GpuQual::Surface => {
                    let elem = elem_rust_type(&field.ty);
                    self.line(&format!(
                        "let {} = __ctx.default_stream().alloc_zeros::<{}>(1)?;",
                        field.name, elem
                    ));
                }
                GpuQual::Actor => {}
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
                GpuQual::Actor => {}
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
                                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface => {
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
                                            // A bare `field = param` assignment (this codebase's
                                            // only real pattern here) means the constructor's
                                            // OWN param type is already `CudaSlice<T>` -- see
                                            // `kernel_ctor_buffer_flags`, which the call site
                                            // consults to decide the SAME thing when building
                                            // the argument. Anything else (a computed
                                            // expression) falls back to the old upload-from-
                                            // Vec behavior, matching what `kernel_ctor_buffer_flags`
                                            // would ALSO decide (false) for a non-bare-Var RHS.
                                            if matches!(&rhs.kind, ExprKind::Var(_)) {
                                                let rhs_s = self.expr(rhs);
                                                self.line(&format!("let {} = {};", fname, rhs_s));
                                            } else {
                                                let rhs_s = self.expr(rhs);
                                                let elem = elem_rust_type(&field.ty);
                                                self.line(&format!(
                                                    "let {} = __ctx.default_stream().clone_htod::<{}, _>(&{})?;",
                                                    fname, elem, rhs_s
                                                ));
                                            }
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
        // 'actor'global), `grid_dim` becomes optional and is derived from its length
        // (1D) or, for a fixed-shape Image/Volume field, from its C/R/X/Y/Z dims (2D/3D).
        let auto_grid_field: Option<&KernelFieldDecl> = fields.iter().find(|f| {
            matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface)
                && (matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) || f.ty.as_image_volume().is_some())
        });

        if let Some(field) = auto_grid_field {
            self.line(
                "fn __boring_launch(mut self, block_dim: (u32,u32,u32), grid_dim: Option<(u32,u32,u32)>, after: &[&Arc<CudaStream>], priority: i32) \
                 -> Result<KernelHandle<Self>, Box<dyn std::error::Error + Send + Sync>> {"
            );
            self.indent += 1;
            self.line("let grid_dim = grid_dim.unwrap_or_else(|| {");
            self.indent += 1;
            if let Some((_, dims)) = field.ty.as_image_volume() {
                self.line(&image_volume_grid_dim_expr(dims));
            } else {
                self.line(&format!("let n = self.{}.len() as u32;", field.name));
                self.line("((n + block_dim.0 - 1) / block_dim.0, 1, 1)");
            }
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
            .filter(|f| matches!(f.qual, GpuQual::Actor))
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
        // GPU-side stream ordering (a `cuStreamWaitEvent`, non-blocking on the
        // CPU) instead of `dep.synchronize()?` -- the latter is a genuine
        // CPU-blocking `cuStreamSynchronize` per dependency, the exact same
        // per-dispatch stall bug this file's Metal-backend sibling fixed,
        // just via the `after=` mechanism instead of the default eager
        // `.wait()` (see `boring_new_stream_with_priority`'s doc and the
        // block-statement desugar, both fixed the same way).
        self.line("for dep in after { stream.join(dep)?; }");

        // Upload 'const fixed-size arrays (and fixed-shape Image/Volume) to
        // __constant__ memory before launch.
        for f in fields {
            if matches!(f.qual, GpuQual::Const) && (matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) || f.ty.as_image_volume().is_some()) {
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
                GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface => {
                    self.line(&format!("launcher.arg(&mut self.{});", f.name));
                }
                GpuQual::Const => {
                    if !matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) && f.ty.as_image_volume().is_none() {
                        // Scalar 'const: passed as a kernel parameter.
                        self.line(&format!("launcher.arg(&self.{});", f.name));
                    }
                    // Array/Image/Volume 'const: uploaded to __constant__ memory above, not a parameter.
                }
                GpuQual::Local => {
                    if !matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)) && f.ty.as_image_volume().is_none() {
                        self.line(&format!("launcher.arg(&self.{});", f.name));
                    }
                }
                GpuQual::Actor => {}
            }
        }
        self.line("unsafe { launcher.launch(cfg) }.map_err(__boring_cuda_classify_error)?;");

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
                    let base = rust_type(ty);
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
        let len = f.body.len();
        for (i, stmt) in f.body.iter().enumerate() {
            if i + 1 == len {
                // A throws function's tail expression is its `Ok(...)` value (any
                // early exit already went through `Stmt::Return`/`Stmt::Throw` above).
                // Resident-returning: wrap in `BoringGpuArg::Host(...)` too (see
                // `resident_elem`'s doc comment above).
                if f.throws {
                    if let Stmt::Expr(e) = stmt {
                        let wrapped = if self.in_resident_return {
                            self.try_resident_field_expr(e).unwrap_or_else(|| {
                                let s = self.expr(e);
                                format!("BoringGpuArg::Host({})", s)
                            })
                        } else {
                            self.expr(e)
                        };
                        self.line(&format!("Ok({})", wrapped));
                        continue;
                    }
                } else if self.in_resident_return {
                    if let Stmt::Expr(e) = stmt {
                        let wrapped = self.try_resident_field_expr(e).unwrap_or_else(|| {
                            let s = self.expr(e);
                            format!("BoringGpuArg::Host({})", s)
                        });
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
        self.indent -= 1;
        self.line("}");
    }

    // ── Statements ─────────────────────────────────────────────────────────────

    fn emit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(s) => {
                let binding = if s.binding.is_mutable() { "let mut" } else { "let" };
                // A materializing call (RHS calls a `fn_returns_resident`
                // function) whose OWN `let` is itself explicitly `'gpu'unified`
                // (e.g. `let [float]'gpu'unified k_t = transpose_gpu(...)`)
                // means the programmer wants this value to stay resident for
                // a later chained kernel-touching call -- see
                // `resident_locals`'s doc. `rust_type`'s own `Type::Qualified`
                // case ignores the qualifier (renders the same as a plain
                // `[float]`), so without this check the declared type here
                // would silently disagree with what a resident RHS actually
                // produces.
                let is_resident_preserving = s.ty.as_ref().and_then(|t| t.gpu_resident_qual()).is_some()
                    && matches!(&s.value.as_ref().map(|v| &v.kind), Some(ExprKind::Call(callee, _))
                        if matches!(&callee.kind, ExprKind::Var(n) if self.fn_returns_resident.contains_key(n.as_str())));
                let ty_ann = if is_resident_preserving {
                    self.resident_locals.insert(s.name.clone());
                    s.ty.as_ref().map(|t| format!(": BoringGpuArg<{}>", elem_rust_type(t))).unwrap_or_default()
                } else {
                    s.ty.as_ref().map(|t| format!(": {}", rust_type(t))).unwrap_or_default()
                };
                if let Some(val) = &s.value {
                    // Track kernel struct type for this variable so field accesses
                    // on it can be redirected to read_<field>() D2H calls.
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
                    let s = if self.in_resident_return {
                        self.try_resident_field_expr(val).unwrap_or_else(|| {
                            let s = self.expr(val);
                            format!("BoringGpuArg::Host({})", s)
                        })
                    } else {
                        self.expr(val)
                    };
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
                matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface)
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
                        .map(|e| format!("&{}.__stream", self.expr(e)))
                        .collect();
                    format!("&[{}]", refs.join(", "))
                }
                _ => { let s = self.expr(&a.value); format!("&[&{s}.__stream]") }
            },
        };
        let grid: String = if let Some(g) = args.iter().find(|a| a.label.as_deref() == Some("grid")) {
            if auto_grid { format!("Some({})", self.dim3_expr(&g.value)) }
            else { self.dim3_expr(&g.value) }
        } else if auto_grid { "None".into() } else { "(1, 1, 1)".into() };
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
        // Used to append `.wait()?` here unconditionally -- a real,
        // synchronous, CPU-blocking `cuStreamSynchronize` after EVERY single
        // dispatch (this is the sole reachable desugar path for `kernel:`
        // blocks; see this file's own doc comment on `KernelHandle` and
        // `boring_new_stream_with_priority`'s doc for the double-sync bug
        // this used to compound). Since every dispatch now shares one
        // persistent, FIFO-ordered stream per priority, a later dispatch on
        // the same stream is already correctly ordered after this one with
        // no CPU wait needed -- so this now just unwraps the `KernelHandle`
        // via its `.inner` field directly (no sync). The actual wait is
        // deferred to the one place it's genuinely needed: reading a field's
        // contents back to the CPU (`read_<field>()`, `__boring_gpu_copy_d2h`).
        Some(format!("{var_name} = {var_name}.__boring_launch({block}, {grid}, {after_arg}, {priority_arg})?.inner"))
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
            fn_returns_resident: self.fn_returns_resident.clone(),
            resident_locals: self.resident_locals.clone(),
            suppress_resident_materialize: self.suppress_resident_materialize,
            top_level_scalars: self.top_level_scalars.clone(),
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

            ExprKind::Var(name) => {
                // Inline top-level scalars when referenced inside kernel new()
                // where they are not in scope as Rust local variables (see
                // `top_level_scalars`'s doc).
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
                // `GPU(n)` → `boring_gpu_ctx_n(n)?`
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "GPU" {
                        let idx = args.first().map(|a| self.expr(&a.value)).unwrap_or_else(|| "0".into());
                        return format!("boring_gpu_ctx_n({} as usize)?", idx);
                    }
                }
                // `ord(c)` — char → isize (Boring built-in)
                if let ExprKind::Var(name) = &callee.kind {
                    if name == "ord" {
                        if let Some(arg) = args.first() {
                            let inner = self.expr(&arg.value);
                            return format!("({} as isize)", inner);
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
                        let args_s = self.emit_kernel_ctor_args(name, args);
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
                let args_s: Vec<String> = args.iter().enumerate().map(|(i, a)| {
                    let expects_ref = ref_flags.as_ref().and_then(|f| f.get(i).copied()).unwrap_or(false);
                    self.coerce_call_arg(&a.value, expects_ref)
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
                // the result -- UNLESS the caller explicitly asked to keep it
                // resident (`__suppress_materialize`, set by `Stmt::Let` for a
                // `'gpu'unified`-typed binding; see `resident_locals`'s doc), in
                // which case the plain `call` (already typed `BoringGpuArg<T>`
                // by `emit_fn`) is returned as-is. A real `?`-operator/E0308
                // type mismatch otherwise, confirmed via `cargo check`.
                if __suppress_materialize {
                    return call;
                }
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
                    // `fs.writeBytes(path, bytes)` — write Vec<isize> as binary file
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
                        matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface)
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
                // Each kernel struct's `.__stream` field is passed as a dependency to __boring_launch.
                let after_arg = match &config.after {
                    None => "&[]".into(),
                    Some(after_expr) => {
                        // Emit a slice literal of stream references from the dependency handles.
                        match &after_expr.kind {
                            ExprKind::Array(elems) => {
                                let refs: Vec<String> = elems.iter()
                                    .map(|e| format!("&{}.__stream", self.expr(e)))
                                    .collect();
                                format!("&[{}]", refs.join(", "))
                            }
                            _ => {
                                let s = self.expr(after_expr);
                                format!("&[&{}.__stream]", s)
                            }
                        }
                    }
                };
                format!("{k}.__boring_launch({block}, {grid}, {after_arg})?")
            }
            ExprKind::New { arena, ctor } => {
                // `new(g) Scale(data)` → `Scale::new(<g>, data)?` (explicit device
                // placement) -- args go through the same buffer-upload handling as
                // a plain `Scale(data)` call (see `emit_kernel_ctor_args`'s doc).
                if let ExprKind::Call(callee, args) = &ctor.kind {
                    if let ExprKind::Var(name) = &callee.kind {
                        if self.kernel_names.contains(name.as_str()) {
                            let dev = arena.as_ref()
                                .map(|a| self.expr(a))
                                .unwrap_or_else(|| "boring_gpu_ctx()".into());
                            let args_s = self.emit_kernel_ctor_args(name, args);
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
                    "(0..({} as usize)).map(|__boring_i| {{ let {} = __boring_i as isize; {} }}).collect::<Vec<_>>()",
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
    /// For `kernel_name`'s `init(...)` parameter list (in order), whether each
    /// positional param is a bare passthrough for an array-typed
    /// `'unified`/`'global`/`'actor'global`/`'surface` field (`field = param`
    /// in the init body, this codebase's only real pattern for such fields —
    /// see `emit_init_stmt`'s identical check) -- these params render as
    /// `CudaSlice<T>` (see `emit_kernel_new`) instead of a host `Vec`, and the
    /// kernel-constructor call site (this file's `expr()`'s `Call` case) must
    /// produce a `CudaSlice` for that argument position instead of a `Vec`.
    /// Returns the field's element type string (`Some("f64")` etc) for a
    /// buffer position, `None` for a scalar one. `None` for the whole `Vec`
    /// only if the kernel/init can't be found at all -- callers treat that as
    /// "no buffer params known" (preserves the pre-existing behavior).
    fn kernel_ctor_buffer_flags(&self, kernel_name: &str) -> Option<Vec<Option<String>>> {
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
            }).filter(|f| {
                matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface)
                    && matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _))
            }).map(|f| elem_rust_type(&f.ty))
        }).collect())
    }

    /// Renders `args` for a call to kernel `name`'s constructor, uploading
    /// each buffer-passthrough argument (see `kernel_ctor_buffer_flags`) via
    /// `clone_htod` -- or moving it directly if it's already GPU-resident --
    /// exactly the way a plain `Scale(data)` constructor call has always
    /// worked. Shared by both constructor call sites: plain `Scale(data)`
    /// (`ExprKind::Call`) and arena-qualified `new(g) Scale(data)`
    /// (`ExprKind::New`) -- the latter used to skip this entirely and just
    /// pass `data` straight through as a bare `Vec`, a real E0308 (`expected
    /// CudaSlice<f64>, found Vec<{float}>`) confirmed via `cargo check`
    /// against the stubbed-cudarc build, since `Scale::new` takes an
    /// already-uploaded `CudaSlice` (its body no longer uploads internally).
    fn emit_kernel_ctor_args(&mut self, name: &str, args: &[Arg]) -> Vec<String> {
        let buffer_flags = self.kernel_ctor_buffer_flags(name);
        args.iter().enumerate().map(|(i, a)| {
            let elem = buffer_flags.as_ref().and_then(|f| f.get(i).cloned()).flatten();
            if let Some(elem) = elem {
                if let ExprKind::Field(obj, field) = &a.value.kind {
                    if let ExprKind::Var(obj_name) = &obj.kind {
                        if self.var_kernel_type.contains_key(obj_name.as_str()) {
                            // `.clone()` -- a real device-to-device copy
                            // (`CudaSlice::clone()` calls `clone_dtod` under the
                            // hood, confirmed against real cudarc 0.19.8 source),
                            // NOT a host round-trip. A bare move here (no
                            // `.clone()`) used to be the "dtod inference"
                            // optimization, but it applied unconditionally
                            // whether or not `obj_name` (e.g. `k1` in
                            // `Scale(k1.buf)`) is used again later -- a real
                            // E0382 ("use of partially moved value") confirmed
                            // via `cargo check` the moment the source kernel is
                            // dispatched or read again afterward. `.clone()` is
                            // correct in every case and still far cheaper than
                            // the D2H+H2D round trip this same field read would
                            // otherwise take.
                            return format!("{}.{}.clone()", obj_name, field);
                        }
                    }
                }
                if let ExprKind::Var(vname) = &a.value.kind {
                    if self.resident_locals.contains(vname.as_str()) {
                        // Same reasoning as the `k_prev.field` case above: `.clone()`
                        // the resident buffer (real D2D copy) instead of moving it
                        // out of the enum, since `vname` isn't provably single-use.
                        return format!(
                            "(match {v} {{ BoringGpuArg::Resident(buf, _) => buf.clone(), BoringGpuArg::Host(v) => boring_new_stream_with_priority(&boring_gpu_ctx(), 0)?.clone_htod::<{elem}, _>(&v)? }})",
                            v = vname, elem = elem
                        );
                    }
                }
                let s = self.coerce_call_arg(&a.value, false);
                return format!(
                    "boring_new_stream_with_priority(&boring_gpu_ctx(), 0)?.clone_htod::<{elem}, _>(&{s})?",
                    elem = elem, s = s
                );
            }
            self.coerce_call_arg(&a.value, false)
        }).collect()
    }

    /// Returns `Some("obj.read_field()?")` when `obj` is a tracked kernel
    /// variable and `field` is a `'unified` or `'global` array field.
    /// Returns `None` for scalars or non-GPU fields (direct field access is fine).
    fn try_gpu_field_read(&self, obj: &str, field: &str) -> Option<String> {
        let kernel_type = self.var_kernel_type.get(obj)?;
        let decl = self.kernel_decls.get(kernel_type)?;
        let kf = decl.fields.iter().find(|f| f.name == field)?;
        match kf.qual {
            GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface => {
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

    /// Recognizes the exact shape every kernel-touching free function's tail
    /// expression uses in practice: a bare `k.field` read of a local kernel-
    /// struct variable's own output array field. When matched (only called
    /// where `in_resident_return` is true, i.e. this function's return type
    /// is itself `'gpu'unified`), skips `try_gpu_field_read`'s materializing
    /// `k.read_field()?` (a real D2H copy) entirely and instead MOVES the
    /// kernel's own output `CudaSlice` out, wrapped as `BoringGpuArg::
    /// Resident` -- `k` is never referenced again after this (its only use
    /// in this codebase's actual shape), so the move is sound; the `let __n`
    /// binding captures the length via a shared borrow before that move
    /// consumes `k.field`. See `try_gpu_field_read`'s doc for why only
    /// `GpuQual::Unified` is handled -- CUDA only emits a `read_<field>()`
    /// accessor (hence only tracks a field as GPU-resident-capable at all)
    /// for that qualifier, unlike Metal's broader set.
    fn try_resident_field_expr(&self, e: &Expr) -> Option<String> {
        let ExprKind::Field(obj, field) = &e.kind else { return None; };
        let ExprKind::Var(obj_name) = &obj.kind else { return None; };
        let kernel_type = self.var_kernel_type.get(obj_name)?;
        let decl = self.kernel_decls.get(kernel_type)?;
        let kf = decl.fields.iter().find(|f| &f.name == field)?;
        match kf.qual {
            GpuQual::Unified => match &kf.ty {
                Type::Array(_) | Type::ArrayN(_, _) => Some(format!(
                    "{{ let __n = {obj}.{field}.len(); BoringGpuArg::Resident({obj}.{field}, __n) }}",
                    obj = obj_name, field = field
                )),
                _ => None,
            },
            _ => None,
        }
    }

    fn host_field_type(&self, field: &KernelFieldDecl) -> String {
        let elem = elem_rust_type(&field.ty);
        // 'const fixed-size arrays (and fixed-shape Image/Volume) are stored as
        // Vec<T> on the host — they are uploaded to __constant__ memory via
        // memcpy_htod, not as CudaSlice args.
        let is_fixed_shape = matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _)) || field.ty.as_image_volume().is_some();
        if matches!(field.qual, GpuQual::Const) && is_fixed_shape {
            return format!("Vec<{}>", elem);
        }
        // A scalar `'const` field (e.g. `let int rows`) is a plain kernel-launch
        // parameter, not a device buffer — was previously wrapped in
        // `CudaSlice<{elem}>` unconditionally here (only the array case above was
        // excluded), producing e.g. `rows: CudaSlice<i64>` for a struct whose
        // constructor then assigns it a bare `i64` (see `emit_init_stmt`'s
        // matching fix) — a real E0308 (`expected CudaSlice<i64>, found i64`),
        // confirmed via `cargo check`.
        if matches!(field.qual, GpuQual::Const) && !is_fixed_shape {
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
        Type::Generic(..) if ty.as_image_volume().is_some() => rust_type(ty.as_image_volume().unwrap().0),
        _                      => rust_type(ty),
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
        Type::Int | Type::Uint               => 8, // isize / usize
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
