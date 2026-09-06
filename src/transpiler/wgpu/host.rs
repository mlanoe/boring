// Copyright (C) 2026 MickaÃ«l LANOÃ‹
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Rust host code emitter for the wgpu backend.

use crate::ast::*;
use crate::transpiler::TranspileError;

/// Returns the generated Rust source plus any errors found while checking a
/// `Screen` program's shape (see `HostEmitter::check_screen_program_shape`).
/// Callers (`transpile_wgpu`) must fold these into `WgpuOutput::errors` and
/// report/exit instead of writing out `host_rs` when non-empty -- exactly
/// like the general pass's own `TranspileConfig`-produced errors already are
/// (see `main.rs`'s `report_transpile_errors`).
pub(super) fn emit_host_rs(
    program: &Program,
    kernel_names: &[String],
    effective_kernels: &[KernelDecl],
    general_code: &str,
    has_boring_main: bool,
    boring_main_throws: bool,
    has_emulated_shader: bool,
) -> (String, Vec<TranspileError>) {
    let mut e = HostEmitter::new(program, kernel_names, effective_kernels, general_code, has_boring_main, boring_main_throws);
    e.has_emulated_shader = has_emulated_shader;
    e.emit();
    (e.out, e.errors)
}

struct HostEmitter<'a> {
    out: String,
    program: &'a Program,
    kernel_names: &'a [String],
    effective_kernels: &'a [KernelDecl],
    has_screen: bool,
    /// When true, render-stmt helpers emit `self.` prefix and `event_loop` for exit.
    in_method: bool,
    /// Rust source for everything the general (std-target) transpiler produced from
    /// this same program: regular fn/struct/enum/dict logic, plus the user's own
    /// `def main()` renamed to `boring_main` (see `transpiler::wgpu::transpile_wgpu`).
    /// Kernel construction/dispatch/field-reads inside those functions are already
    /// correctly special-cased by that pass (see `transpiler::emit_kernel`) — this
    /// backend only needs to splice the result in and, if present, call `boring_main()`.
    general_code: &'a str,
    has_boring_main: bool,
    /// Whether the emitted `boring_main` actually returns a `Result` — an explicit
    /// user `def main():` only does when it declares `throws`; a synthesized one
    /// (from bare top-level statements, `emit_gpu_boring_main`) always does. Gates
    /// whether `emit_main` calls it as `if let Err(e) = boring_main() { .. }` or as
    /// a plain statement — the two aren't interchangeable, calling a `()`-returning
    /// fn through the `Err` pattern is a real `cargo check` E0308, not just style.
    boring_main_throws: bool,
    /// Whether a second, shared-memory-emulated WGSL module was emitted
    /// alongside the real-subgroup one (`shaders/main_emulated.wgsl`) — only
    /// true when some kernel uses `gpu.warp.*` (see `device::WarpMode`). Gates
    /// the `wgpu::Features::SUBGROUP` runtime detection and dual-shader-module
    /// dispatch in `emit_pipeline_init`/wherever the shader module is created.
    has_emulated_shader: bool,
    /// Errors found while checking a `Screen` program's shape (see
    /// `check_screen_program_shape`/`emit_render_stmt`'s error fallbacks).
    /// Non-empty means `out` is incomplete/wrong and must not be written out
    /// — see `emit_host_rs`'s doc comment.
    errors: Vec<TranspileError>,
}

impl<'a> HostEmitter<'a> {
    fn new(
        program: &'a Program,
        kernel_names: &'a [String],
        effective_kernels: &'a [KernelDecl],
        general_code: &'a str,
        has_boring_main: bool,
        boring_main_throws: bool,
    ) -> Self {
        let has_screen = program.items.iter().any(|item| {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    if let ExprKind::Call(callee, _) = &val.kind {
                        if let ExprKind::Var(n) = &callee.kind {
                            return n == "Screen";
                        }
                    }
                }
            }
            false
        });
        Self { out: String::new(), program, kernel_names, effective_kernels, has_screen, in_method: false, general_code, has_boring_main, boring_main_throws, has_emulated_shader: false, errors: Vec::new() }
    }

    fn line(&mut self, s: &str) { self.out.push_str(s); self.out.push('\n'); }
    fn blank(&mut self) { self.out.push('\n'); }

    fn push_error(&mut self, line: usize, col: usize, msg: impl Into<String>) {
        self.errors.push(TranspileError::at(msg, line, col));
    }

    fn emit(&mut self) {
        self.emit_header();
        self.emit_kernel_structs();
        self.emit_gpu_copy_helpers();
        self.emit_device_queue_globals();
        self.out.push_str(self.general_code);
        self.blank();
        if self.has_screen {
            self.check_screen_program_shape();
        }
        self.emit_main();
    }

    /// Every top-level item of a `Screen` program is either transpiled by the
    /// general pass (fn/struct/enum/kernel/mod/use declarations — untouched by
    /// `TranspileConfig::gpu_top_level_handled_by_host`) or must be recognized
    /// by `emit_screen_main`'s own hardcoded shape-matchers (`Screen(...)`/
    /// kernel-constructor `let`s, scalar-literal `let`s, and the `kernel: loop:`
    /// render block) — the general pass unconditionally skips every other
    /// top-level `Stmt`/non-static `Let` for a `Screen` program (see
    /// `emit_program_items`'s `host_owns_top_level` branches), on the
    /// assumption that this emitter owns them instead. Historically it didn't:
    /// anything outside that narrow whitelist (a bare `print`, a `for` loop, a
    /// `let` with an array/string/GPU-introspection initializer, extra
    /// statements in a `kernel:` block beside its `loop:`) was silently
    /// dropped from the generated Rust with no diagnostic at all — confirmed
    /// against a real generated project. This walks the same top-level items
    /// `emit_screen_main` will (or won't) recognize and reports every one that
    /// falls outside that whitelist, so the build fails loudly instead of
    /// quietly losing code.
    fn check_screen_program_shape(&mut self) {
        let items = self.program.items.clone();
        for item in &items {
            match item {
                Item::Let(s) if s.is_static => {} // real `const`, emitted by the general pass.
                Item::Let(s) if !self.is_recognized_screen_let(s) => {
                    self.push_error(s.line, s.col, format!(
                        "wgpu target: top-level `let {}` is not supported inside a Screen program -- \
                         only a `Screen(...)`/kernel-constructor call or an int/float/bool literal \
                         initializer is recognized at the top level here",
                        s.name
                    ));
                }
                Item::Let(_) => {} // recognized shape — emit_screen_main handles it.
                Item::Stmt(stmt) => self.check_screen_top_level_stmt(stmt),
                // Fn/Struct/Enum/Kernel/Mod/Use/Alias/Trait/Ext: fully handled by the
                // general pass regardless of `host_owns_top_level` -- nothing to check.
                _ => {}
            }
        }
    }

    /// Mirrors exactly what `emit_screen_main`'s `scalar_fields`/`kernel_fields`
    /// filters (and `extract_screen_info`) accept for a top-level `let`.
    fn is_recognized_screen_let(&self, s: &LetStmt) -> bool {
        let Some(val) = &s.value else { return false; };
        match &val.kind {
            ExprKind::Call(callee, _) => matches!(&callee.kind, ExprKind::Var(n)
                if n == "Screen" || self.kernel_names.contains(n)),
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) => true,
            _ => false,
        }
    }

    /// A top-level statement is only recognized as the `kernel:` render block
    /// (`find_render_loop_body`/`emit_render_stmt` own its `loop:` body) —
    /// anything else at the top level, and any statement inside a `kernel:`
    /// block that isn't its single `loop:`, is unrecognized.
    fn check_screen_top_level_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            // A bare `#` comment line parses as a real top-level `Stmt::Comment`
            // (see parse_stmt.rs/parser/mod.rs's `parse_item`), not something the
            // lexer strips — inert, never actual code, so it must never trip this
            // shape check. Every other backend's kernel-body statement emitter
            // already special-cases `Stmt::Comment` the same way (device.rs,
            // cuda/device.rs, metal/device.rs); this top-level Screen-program
            // check was the one place that didn't, so a header comment (or any
            // top-level comment) before/around a Screen program's recognized
            // shape failed the whole build with one spurious error per line.
            Stmt::Comment(_) => {}
            Stmt::KernelBlock(block) => {
                let mut seen_loop = false;
                for inner in &block.body {
                    if matches!(inner, Stmt::Comment(_)) {
                        continue;
                    }
                    if matches!(inner, Stmt::Loop(_)) {
                        if seen_loop {
                            let (l, c) = stmt_line_col(inner);
                            self.push_error(l, c,
                                "wgpu target: only one `loop:` is supported inside a Screen `kernel:` block");
                        }
                        seen_loop = true;
                    } else {
                        let (l, c) = stmt_line_col(inner);
                        self.push_error(l, c,
                            "wgpu target: statement not supported inside a Screen `kernel:` block outside its `loop:` body");
                    }
                }
            }
            _ => {
                let (l, c) = stmt_line_col(stmt);
                self.push_error(l, c,
                    "wgpu target: top-level statement is not supported inside a Screen program -- \
                     only a `kernel: loop: ...` render block is recognized here");
            }
        }
    }

    /// Global GPU device/queue, set once by `async_main()` before any kernel-touching
    /// function (transpiled by the general pass above, potentially deep in the call
    /// graph — see `transpiler::emit_kernel`) can run. Boring function signatures are
    /// never changed to thread a device/queue parameter through, so kernel
    /// construction reaches them via these accessors instead.
    fn emit_device_queue_globals(&mut self) {
        self.line("static __BORING_GPU_DEVICE: std::sync::OnceLock<std::sync::Arc<wgpu::Device>> = std::sync::OnceLock::new();");
        self.line("static __BORING_GPU_QUEUE: std::sync::OnceLock<std::sync::Arc<wgpu::Queue>> = std::sync::OnceLock::new();");
        self.line("fn __boring_gpu_device() -> std::sync::Arc<wgpu::Device> { std::sync::Arc::clone(__BORING_GPU_DEVICE.get().expect(\"GPU device not initialized\")) }");
        self.line("fn __boring_gpu_queue() -> std::sync::Arc<wgpu::Queue> { std::sync::Arc::clone(__BORING_GPU_QUEUE.get().expect(\"GPU queue not initialized\")) }");
        self.blank();
        self.emit_gpu_introspection_globals();
    }

    /// Backing storage + accessors for the `GPU` type (`GPU(n)`, `GPU.all()`,
    /// `.name()`/`.totalMem()`/etc — see `emit_expr.rs`'s `gpu_device_vars`
    /// handling). Real per-adapter introspection (2026-09-01): the full list of
    /// adapters the system actually exposes is enumerated once at startup
    /// (`instance.enumerate_adapters`, see `emit_main`/`emit_screen_main`) and
    /// `GPU(n)` indexes into it directly, index 0 always being the adapter
    /// `device`/`queue` were actually created from. This is introspection
    /// only, same scope as CUDA/ROCm's own docs describe for the analogous
    /// case — `new(g) K` (placing a kernel's actual dispatch on a specific
    /// non-default adapter) is still not implemented on this backend; every
    /// kernel still dispatches on the single global `device`/`queue`
    /// regardless of which `GPU(n)` it was constructed with. See
    /// docs/wgpu-backend.md's "`GPU` type on wgpu" section.
    fn emit_gpu_introspection_globals(&mut self) {
        self.line("static __BORING_GPU_ADAPTERS: std::sync::OnceLock<Vec<std::sync::Arc<wgpu::Adapter>>> = std::sync::OnceLock::new();");
        self.line("fn __boring_gpu_adapter(idx: usize) -> std::sync::Arc<wgpu::Adapter> {");
        self.line("    let adapters = __BORING_GPU_ADAPTERS.get().expect(\"GPU adapters not initialized\");");
        self.line("    adapters.get(idx).map(std::sync::Arc::clone)");
        self.line("        .unwrap_or_else(|| panic!(\"GPU({}) out of range -- {} adapter(s) found\", idx, adapters.len()))");
        self.line("}");
        self.line("fn __boring_gpu_all() -> Vec<usize> {");
        self.line("    (0..__BORING_GPU_ADAPTERS.get().map(|a| a.len()).unwrap_or(0)).collect()");
        self.line("}");
        self.line("fn __boring_gpu_name(idx: usize) -> String { __boring_gpu_adapter(idx).get_info().name }");
        // wgpu's AdapterInfo has no memory-size fields on any backend -- there is
        // no real value to report, so these match the interpreter's/CUDA docs'
        // own "not available" convention (0) rather than fabricating a number.
        self.line("fn __boring_gpu_total_mem(_idx: usize) -> i64 { 0 }");
        self.line("fn __boring_gpu_free_mem(_idx: usize) -> i64 { 0 }");
        self.line("fn __boring_gpu_compute_capability(_idx: usize) -> Vec<i64> { vec![0, 0] }"); // CUDA-only concept
        self.line("fn __boring_gpu_warp_size(_idx: usize) -> i64 { 32 }"); // conservative default, not queryable via wgpu
        self.line("fn __boring_gpu_max_threads(idx: usize) -> i64 { __boring_gpu_adapter(idx).limits().max_compute_invocations_per_workgroup as i64 }");
        self.line("fn __boring_gpu_max_shared_mem(idx: usize) -> i64 { __boring_gpu_adapter(idx).limits().max_compute_workgroup_storage_size as i64 }");
        self.blank();
    }

    /// Builds the introspection adapter list for `__BORING_GPU_ADAPTERS`: the
    /// adapter actually selected for `device`/`queue` goes first (so `GPU(0)`
    /// always means "the adapter your kernels actually run on"), followed by
    /// any other physical adapter `enumerate_adapters` finds.
    ///
    /// `primary_var` must already be a `std::sync::Arc<wgpu::Adapter>` binding
    /// (matching the existing `device`/`queue` convention of wrapping right
    /// after creation) — **`wgpu::Adapter` itself implements neither `Clone`
    /// nor `PartialEq`** (confirmed against the real pinned `wgpu = "22"`
    /// source, `wgpu-22.1.0/src/lib.rs`'s `pub struct Adapter` only derives
    /// `Debug` — an earlier version of this function assumed otherwise from
    /// docs.rs for a newer wgpu release and failed a real `cargo check` with
    /// E0599/E0369 the moment it was tried against the actual generated
    /// project). Arc-wrapping the primary adapter once at the top lets this
    /// function `Arc::clone` it cheaply into the list while the caller's own
    /// `primary_var` binding stays valid afterward (needed in
    /// `emit_screen_main`, where `adapter` is later moved into the `__App`
    /// struct literal). Dedup compares `AdapterInfo` (`.get_info()`) instead,
    /// which *does* derive `PartialEq`/`Eq` (`wgpu-types-22.0.0/src/lib.rs`) —
    /// a real physical GPU can otherwise appear twice if the platform exposes
    /// it through more than one backend (e.g. Vulkan and GL on the same Linux
    /// box).
    fn emit_gpu_adapter_enumeration(&mut self, primary_var: &str) {
        self.line(&format!("    let __boring_primary_info = {primary_var}.get_info();"));
        self.line(&format!("    let mut __boring_gpu_adapters: Vec<std::sync::Arc<wgpu::Adapter>> = vec![std::sync::Arc::clone(&{primary_var})];"));
        self.line("    for __boring_other in instance.enumerate_adapters(wgpu::Backends::all()) {");
        self.line("        if __boring_other.get_info() != __boring_primary_info {");
        self.line("            __boring_gpu_adapters.push(std::sync::Arc::new(__boring_other));");
        self.line("        }");
        self.line("    }");
        self.line("    let _ = __BORING_GPU_ADAPTERS.set(__boring_gpu_adapters);");
    }

    fn emit_header(&mut self) {
        self.line("// Generated by boring build --target wgpu.");
        self.line("// Do not edit -- re-generate with: boring build --target wgpu");
        self.blank();
        // Note: no hardcoded `use std::collections::HashMap;` here -- the general
        // transpiler's own output (spliced in by `emit()`, see `general_code`) already
        // emits it when the merged program actually uses dicts, and a second identical
        // `use` at the same scope is a duplicate-name error (E0252).
        self.line("use wgpu::util::DeviceExt;");
        self.line("use bytemuck::{Pod, Zeroable};");
        if self.has_screen {
            self.line("use winit::application::ApplicationHandler;");
            self.line("use winit::event::{WindowEvent, ElementState};");
            self.line("use winit::event_loop::{ActiveEventLoop, EventLoop};");
            self.line("use winit::keyboard::{Key, NamedKey};");
            self.line("use winit::window::{Window, WindowAttributes, WindowId};");
        }
        self.blank();
    }

    // â"€â"€ Kernel structs â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    fn emit_kernel_structs(&mut self) {
        // Use effective_kernels (monomorphised) rather than raw program items.
        let decls: Vec<KernelDecl> = self.effective_kernels.to_vec();
        if !decls.is_empty() {
            // Every kernel struct's `new()` used to call `create_shader_module` +
            // `create_compute_pipeline` on EVERY instantiation -- and boring code
            // constructs a fresh kernel instance on every single `linear_gpu`/
            // `attention_gpu`/etc. call (see math_gpu.br's helpers), so a hot loop of
            // a few hundred calls recompiled the entire WGSL module (all kernels,
            // not just the one being constructed) that many times over. Shader
            // compilation dominates pipeline setup cost, so this was the single
            // largest cost in any wgpu-target program with more than a handful of
            // kernel dispatches. Compile the module and each kernel's pipeline
            // exactly once (lazily, on first use) and clone the cheap `Arc` handle
            // out for every later instantiation instead.
            self.line("static __BORING_SHADER_MODULE: std::sync::OnceLock<wgpu::ShaderModule> = std::sync::OnceLock::new();");
            self.blank();
        }
        for decl in &decls {
            self.emit_kernel_struct(decl);
        }
    }

    fn emit_kernel_struct(&mut self, decl: &KernelDecl) {
        let name = &decl.name;

        // Cache of this kernel type's compiled pipeline (see emit_kernel_structs'
        // doc comment on __BORING_SHADER_MODULE for why this exists).
        self.line(&format!(
            "static {}_PIPELINE: std::sync::OnceLock<std::sync::Arc<wgpu::ComputePipeline>> = std::sync::OnceLock::new();",
            name.to_uppercase()
        ));
        self.blank();

        // Params struct (scalars + Dimension fields for uniform buffer).
        let params_fields: Vec<&KernelFieldDecl> = decl.fields.iter()
            .filter(|f| is_params_field(f)).collect();
        if !params_fields.is_empty() {
            self.line("#[repr(C)]");
            self.line("#[derive(Debug, Clone, Copy, Pod, Zeroable)]");
            self.line(&format!("struct {}Params {{", name));
            for f in &params_fields {
                match &f.ty {
                    Type::Named(n) if n == "Dimension" => {
                        self.line(&format!("    {}_w: i32,", f.name));
                        self.line(&format!("    {}_h: i32,", f.name));
                    }
                    Type::ArrayN(inner, n) => {
                        self.line(&format!("    {}: [{}; {}],", f.name, host_scalar_type(inner), n));
                    }
                    ty => {
                        self.line(&format!("    {}: {},", f.name, host_scalar_type(ty)));
                    }
                }
            }
            self.line("}");
            self.blank();
        }

        // Kernel host struct.
        self.line(&format!("struct {} {{", name));
        self.line("    device: std::sync::Arc<wgpu::Device>,");
        self.line("    queue: std::sync::Arc<wgpu::Queue>,");
        self.line("    pipeline: std::sync::Arc<wgpu::ComputePipeline>,");
        self.line("    bind_group: wgpu::BindGroup,");
        for f in &decl.fields {
            match f.qual {
                GpuQual::Unified | GpuQual::Global | GpuQual::Surface | GpuQual::ActorGlobal | GpuQual::ActorUnified => {
                    if is_buffer_array_ty(&f.ty) {
                        // Arc-wrapped so a `'unified`/`'global` value returned across a
                        // function-call boundary can hand its buffer directly to the next
                        // kernel's constructor (an Arc::clone, no data copy) without the
                        // producing kernel instance itself needing to stay alive — see
                        // docs/scoped-access-blocks.md's interprocedural residency case.
                        self.line(&format!("    {}_buf: std::sync::Arc<wgpu::Buffer>,", f.name));
                        // Keep Vec mirror for host-visible fields ('unified/'actor'unified).
                        if matches!(f.qual, GpuQual::Unified | GpuQual::Surface | GpuQual::ActorUnified) {
                            let inner_ty = array_inner(&f.ty);
                            self.line(&format!("    {}: Vec<{}>,", f.name, host_scalar_type(&inner_ty)));
                        }
                    }
                }
                GpuQual::Const => {
                    match &f.ty {
                        Type::Named(n) if n == "Dimension" => {
                            self.line(&format!("    {}: (i32, i32),", f.name));
                        }
                        Type::ArrayN(inner, n) => {
                            self.line(&format!("    {}: [{}; {}],", f.name, host_scalar_type(inner), n));
                        }
                        ty => {
                            self.line(&format!("    {}: {},", f.name, host_scalar_type(ty)));
                        }
                    }
                }
                GpuQual::Local => {
                    // Scalar local fields (e.g. `var float t`) get a host mirror so the
                    // caller can update them between dispatches.
                    if !is_buffer_array_ty(&f.ty) {
                        self.line(&format!("    {}: {},", f.name, host_scalar_type(&f.ty)));
                    }
                }
                GpuQual::Actor => {
                    // workgroup memory, no host side.
                }
            }
        }
        if !params_fields.is_empty() {
            self.line("    params_buf: wgpu::Buffer,");
        }
        self.line("}");
        self.blank();

        // impl block.
        self.line(&format!("impl {} {{", name));

        // Constructor.
        self.emit_kernel_new(decl);
        self.blank();

        // Rebuilds `self.bind_group` from the current buffer fields — called after
        // `copy_{field}_to_device` replaces a buffer with a differently-sized one
        // (see that method and its doc comment on why the buffer can't just stay at
        // the size `new()` created it with).
        self.emit_kernel_rebuild_bind_group(decl);
        self.blank();

        // Launch method.
        self.emit_kernel_launch(decl);
        self.blank();

        // gpu.copy helpers: __copy_to_host / __copy_to_device.
        self.emit_kernel_copy_accessors(decl);

        self.line("}");
        self.blank();
    }

    fn emit_kernel_new(&mut self, decl: &KernelDecl) {
        let name = &decl.name;

        // Build constructor params from init method if present.
        let init_params: Vec<String> = if let Some(init) = decl.methods.iter().find(|m| m.name == "init") {
            init.params.iter().map(|p| {
                let ty = p.ty.as_ref().map(host_type).unwrap_or_else(|| "i64".into());
                format!("{}: {}", p.name, ty)
            }).collect()
        } else {
            vec![]
        };
        let params_sig = if init_params.is_empty() { String::new() }
                         else { format!("{}, ", init_params.join(", ")) };

        // Check whether this kernel has a Dimension const field (grid-sized buffers).
        let has_dim_field = kernel_has_dim_field(decl);
        let dim_param = if has_dim_field { "width: i32, height: i32, " } else { "" };

        self.line(&format!("    fn new({}{dim_param}device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>) -> Self {{", params_sig));
        self.blank();

        // Collect buffer fields.
        let buf_fields: Vec<&KernelFieldDecl> = decl.fields.iter()
            .filter(|f| matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::Surface | GpuQual::ActorGlobal | GpuQual::ActorUnified)
                     && is_buffer_array_ty(&f.ty))
            .collect();
        let params_fields: Vec<&KernelFieldDecl> = decl.fields.iter()
            .filter(|f| is_params_field(f)).collect();

        // Create GPU buffers.
        for f in &buf_fields {
            let usages = buffer_usages(f);
            let inner_ty = array_inner(&f.ty);
            let host_ty = host_scalar_type(&inner_ty);
            let (size_expr, init_data) = match &f.ty {
                Type::ArrayN(_, n) => {
                    (format!("({} * std::mem::size_of::<{}>()) as u64", n, host_ty),
                     format!("vec![{}::default(); {}]", host_ty, n))
                }
                // A fixed-shape labeled multi-dim array (every axis a literal
                // int, e.g. `[float32, width = 32, height = 32]`) has a
                // compile-time-known element count just like `ArrayN` above —
                // allocate it at its real size up front instead of falling to
                // the 4-byte placeholder below. A field with no explicit
                // `field = ...` in `init()` (a pure output, e.g. this kernel's
                // `c`) never reaches `copy_{field}_to_device` to grow it, so
                // without this branch it would stay stuck at 4 bytes forever
                // and fail wgpu's binding-size validation on first dispatch.
                ty if ty.labeled_array_len().is_some() => {
                    let n = ty.labeled_array_len().unwrap();
                    (format!("({} * std::mem::size_of::<{}>()) as u64", n, host_ty),
                     format!("vec![{}::default(); {}]", host_ty, n))
                }
                _ if has_dim_field => {
                    // Dynamic size derived from Dimension (width * height * sizeof<T>).
                    (format!("((width * height) as usize * std::mem::size_of::<{}>()) as u64", host_ty),
                     format!("vec![{}::default(); (width * height) as usize]", host_ty))
                }
                _ => {
                    // Placeholder size, not a real allocation: the actual data length
                    // isn't known until the first `copy_{field}_to_device` call (see
                    // that method in emit_kernel_copy_accessors), which replaces this
                    // buffer and rebuilds the bind group once the real size is known.
                    // It can't be 0 — `new()` builds the *initial* bind group below
                    // from whatever buffer exists right now, and wgpu rejects a storage/
                    // uniform binding under 4 bytes (one scalar element) at bind-group
                    // creation time, before the resize logic ever runs.
                    ("4u64".into(), format!("Vec::<{}>::new()", host_ty))
                }
            };
            let _ = init_data;
            self.line(&format!("        let {}_buf = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {{", f.name));
            self.line("            label: None,");
            self.line(&format!("            size: {},", size_expr));
            self.line(&format!("            usage: {},", usages));
            self.line("            mapped_at_creation: false,");
            self.line("        }));");
        }

        // Params buffer.
        if !params_fields.is_empty() {
            self.line("        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {");
            self.line("            label: None,");
            self.line(&format!("            size: std::mem::size_of::<{}Params>() as u64,", name));
            self.line("            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,");
            self.line("            mapped_at_creation: false,");
            self.line("        });");
        }
        self.blank();

        // Pipeline (workgroup size is hardcoded in WGSL, no override constants needed).
        // Compiled once and cached (see the `{}_PIPELINE`/`__BORING_SHADER_MODULE`
        // statics' doc comment) -- every later `new()` call just clones the Arc.
        self.line(&format!("        let pipeline = {}_PIPELINE.get_or_init(|| {{", name.to_uppercase()));
        self.line("            let shader = __BORING_SHADER_MODULE.get_or_init(|| {");
        if self.has_emulated_shader {
            // `device.features().contains(SUBGROUP)` is a necessary but not
            // sufficient check: it reflects hardware/backend capability, but a
            // wgpu version can advertise the feature while its bundled `naga`
            // WGSL frontend still doesn't parse the `enable subgroups;`
            // directive (observed in the wild — a real gap between HAL-level
            // feature plumbing and shader-frontend language support, not a
            // hypothetical). So: only ATTEMPT the real-subgroup module when the
            // feature is present, and wrap that attempt in an error scope —
            // catching a shader-compile failure and falling back to the
            // emulated module at runtime, instead of the uncaught-validation-error
            // panic device.create_shader_module would otherwise trigger.
            self.line("                let __boring_try_real = device.features().contains(wgpu::Features::SUBGROUP);");
            self.line("                let __boring_module = if __boring_try_real {");
            self.line("                    device.push_error_scope(wgpu::ErrorFilter::Validation);");
            self.line("                    let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {");
            self.line("                        label: None,");
            self.line("                        source: wgpu::ShaderSource::Wgsl(include_str!(\"../shaders/main.wgsl\").into()),");
            self.line("                    });");
            self.line("                    if pollster::block_on(device.pop_error_scope()).is_some() { None } else { Some(m) }");
            self.line("                } else {");
            self.line("                    None");
            self.line("                };");
            self.line("                __boring_module.unwrap_or_else(|| device.create_shader_module(wgpu::ShaderModuleDescriptor {");
            self.line("                    label: None,");
            self.line("                    source: wgpu::ShaderSource::Wgsl(include_str!(\"../shaders/main_emulated.wgsl\").into()),");
            self.line("                }))");
        } else {
            self.line("                device.create_shader_module(wgpu::ShaderModuleDescriptor {");
            self.line("                    label: None,");
            self.line("                    source: wgpu::ShaderSource::Wgsl(include_str!(\"../shaders/main.wgsl\").into()),");
            self.line("                })");
        }
        self.line("            });");
        self.line("            std::sync::Arc::new(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {");
        self.line("                label: None,");
        self.line("                layout: None,");
        self.line("                module: shader,");
        self.line(&format!("                entry_point: \"{}_main\",", name));
        self.line("                compilation_options: wgpu::PipelineCompilationOptions::default(),");
        self.line("                cache: None,");
        self.line("            }))");
        self.line("        }).clone();");
        self.blank();

        // Bind group.
        let mut binding_entries: Vec<String> = vec![];
        let mut binding_idx: u32 = 0;
        for f in &buf_fields {
            binding_entries.push(format!(
                "            wgpu::BindGroupEntry {{ binding: {binding_idx}, resource: {}_buf.as_entire_binding() }},",
                f.name, binding_idx = binding_idx
            ));
            binding_idx += 1;
        }
        if !params_fields.is_empty() {
            binding_entries.push(format!(
                "            wgpu::BindGroupEntry {{ binding: {binding_idx}, resource: params_buf.as_entire_binding() }},",
                binding_idx = binding_idx
            ));
        }
        let bg_layout = "pipeline.get_bind_group_layout(0)";
        self.line("        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {");
        self.line("            label: None,");
        self.line(&format!("            layout: &{},", bg_layout));
        self.line("            entries: &[");
        for entry in &binding_entries { self.line(entry); }
        self.line("            ],");
        self.line("        });");
        self.blank();

        // Return struct.
        self.line(&format!("        {} {{", name));
        self.line("            device: std::sync::Arc::clone(&device),");
        self.line("            queue: std::sync::Arc::clone(&queue),");
        self.line("            pipeline,");
        self.line("            bind_group,");
        for f in &buf_fields {
            self.line(&format!("            {}_buf,", f.name));
            if matches!(f.qual, GpuQual::Unified | GpuQual::Surface | GpuQual::ActorUnified) {
                let inner_ty = array_inner(&f.ty);
                let n = match &f.ty {
                    Type::ArrayN(_, n) => *n,
                    ty => ty.labeled_array_len().map(|n| n as usize).unwrap_or(0),
                };
                self.line(&format!("            {}: vec![{}::default(); {}],",
                    f.name, host_scalar_type(&inner_ty), n));
            }
        }
        if !params_fields.is_empty() {
            self.line("            params_buf,");
        }
        for f in &decl.fields {
            match f.qual {
                GpuQual::Const => {
                    match &f.ty {
                        Type::Named(n) if n == "Dimension" => {
                            // Use width/height params when available (has_dim_field kernel).
                            self.line(&format!("            {}: (width, height),", f.name));
                        }
                        Type::ArrayN(inner, n) => {
                            self.line(&format!("            {}: [{}::default(); {}],",
                                f.name, host_scalar_type(inner), n));
                        }
                        ty => {
                            let val = f.default.as_ref()
                                .map(emit_scalar_default)
                                .unwrap_or_else(|| format!("{}::default()", host_scalar_type(ty)));
                            self.line(&format!("            {}: {},", f.name, val));
                        }
                    }
                }
                GpuQual::Local
                    if !is_buffer_array_ty(&f.ty) => {
                        let val = f.default.as_ref()
                            .map(emit_scalar_default)
                            .unwrap_or_else(|| format!("{}::default()", host_scalar_type(&f.ty)));
                        self.line(&format!("            {}: {},", f.name, val));
                    }
                _ => {}
            }
        }
        self.line("        }");
        self.line("    }");
    }

    fn emit_kernel_launch(&mut self, decl: &KernelDecl) {
        let name = &decl.name;
        let params_fields: Vec<&KernelFieldDecl> = decl.fields.iter()
            .filter(|f| is_params_field(f)).collect();

        self.line("    fn dispatch(&self, gx: u32, gy: u32, gz: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
        // Validation errors (e.g. a workgroup count/size the device rejects)
        // are checked synchronously by `dispatch_workgroups` while the compute
        // pass is being encoded, below -- before this scope closes. No
        // `device.poll()` needed for this: unlike an execution-time fault
        // (out-of-bounds access, device lost), validation is decided on the
        // CPU side as the command buffer is built, not during GPU execution.
        //
        // Two nested scopes -- OutOfMemory pushed first (outer), Validation
        // second (inner) -- so a GpuError::OutOfMemory and a
        // GpuError::LaunchError can be told apart instead of collapsing into
        // one generic "kernel dispatch rejected" string, as this used to.
        self.line("        self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);");
        self.line("        self.device.push_error_scope(wgpu::ErrorFilter::Validation);");

        // Write params before dispatch.
        if !params_fields.is_empty() {
            self.line(&format!("        let params = {}Params {{", name));
            for f in &params_fields {
                match &f.ty {
                    Type::Named(n) if n == "Dimension" => {
                        self.line(&format!("            {}_w: self.{}.0,", f.name, f.name));
                        self.line(&format!("            {}_h: self.{}.1,", f.name, f.name));
                    }
                    Type::ArrayN(_, _) => {
                        self.line(&format!("            {}: self.{},", f.name, f.name));
                    }
                    _ => {
                        self.line(&format!("            {}: self.{},", f.name, f.name));
                    }
                }
            }
            self.line("        };");
            self.line("        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));");
        }
        self.blank();

        self.line("        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });");
        self.line("        {");
        self.line("            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });");
        self.line("            cpass.set_pipeline(&self.pipeline);");
        self.line("            cpass.set_bind_group(0, &self.bind_group, &[]);");
        self.line("            cpass.dispatch_workgroups(gx, gy, gz);");
        self.line("        }");
        self.line("        self.queue.submit(std::iter::once(encoder.finish()));");
        // No poll here — caller (present_buffer or explicit sync) handles synchronization.
        //
        // Pop order mirrors push order in reverse: Validation (pushed last)
        // comes off first, OutOfMemory (pushed first) comes off second. Each
        // popped error is classified into a typed `GpuError` and wrapped in
        // `BoringError::Other` so `catch GpuError.OutOfMemory:` / `catch
        // GpuError.LaunchError:` can dispatch on it in Boring source, instead
        // of the single opaque formatted-string error this used to return.
        self.line("        if pollster::block_on(self.device.pop_error_scope()).is_some() {");
        self.line("            let _ = pollster::block_on(self.device.pop_error_scope());");
        self.line("            return Err(Box::new(BoringError::Other(std::any::TypeId::of::<GpuError>(), Box::new(GpuError::LaunchError) as Box<dyn BoringVal + Send + Sync>)));");
        self.line("        }");
        self.line("        if pollster::block_on(self.device.pop_error_scope()).is_some() {");
        self.line("            return Err(Box::new(BoringError::Other(std::any::TypeId::of::<GpuError>(), Box::new(GpuError::OutOfMemory) as Box<dyn BoringVal + Send + Sync>)));");
        self.line("        }");
        self.line("        Ok(())");
        self.line("    }");
    }

    fn emit_kernel_rebuild_bind_group(&mut self, decl: &KernelDecl) {
        let buf_fields: Vec<&KernelFieldDecl> = decl.fields.iter()
            .filter(|f| matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::Surface | GpuQual::ActorGlobal | GpuQual::ActorUnified)
                     && is_buffer_array_ty(&f.ty))
            .collect();
        let has_params = decl.fields.iter().any(is_params_field);

        self.line("    fn rebuild_bind_group(&mut self) {");
        self.line("        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();");
        for (i, f) in buf_fields.iter().enumerate() {
            self.line(&format!(
                "        entries.push(wgpu::BindGroupEntry {{ binding: {}, resource: self.{}_buf.as_entire_binding() }});",
                i, f.name
            ));
        }
        if has_params {
            self.line(&format!(
                "        entries.push(wgpu::BindGroupEntry {{ binding: {}, resource: self.params_buf.as_entire_binding() }});",
                buf_fields.len()
            ));
        }
        self.line("        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {");
        self.line("            label: None,");
        self.line("            layout: &self.pipeline.get_bind_group_layout(0),");
        self.line("            entries: &entries,");
        self.line("        });");
        self.line("    }");
    }

    fn emit_kernel_copy_accessors(&mut self, decl: &KernelDecl) {
        for f in &decl.fields {
            // 'global fields are host-supplied inputs; 'unified fields are typically
            // kernel-written outputs (e.g. StftPower's `power`) but are just as much a
            // Vec<T> mirror on the host struct (see the field-emission loop above) and
            // need the same read-back/upload accessors to actually be usable from host code.
            // 'actor'unified needs the same accessors for the same reason (host-visible),
            // plus `kernel_output_fill_map`'s unconditional `copy_{field}_to_device` call
            // for any `field = [value for ..count]` init — see `emit_kernel.rs`.
            if matches!(f.qual, GpuQual::Global | GpuQual::Unified | GpuQual::ActorUnified) && is_buffer_array_ty(&f.ty) {
                let inner_ty = array_inner(&f.ty);
                let host_ty = host_scalar_type(&inner_ty);
                // D2H.
                self.line(&format!("    fn copy_{}_to_host(&self) -> Vec<{}> {{", f.name, host_ty));
                self.line(&format!("        __boring_gpu_copy_d2h::<{}>(&self.device, &self.queue, &self.{}_buf)", host_ty, f.name));
                self.line("    }");
                // H2D. `new()` creates every buffer field at size 0 (it has no host-side
                // notion of the real data size until a caller actually supplies some —
                // see wgpu::host::emit_kernel_new and emit_kernel::kernel_output_fill_map),
                // so the first write here almost always needs to grow the buffer. A
                // grown buffer needs a fresh wgpu::BindGroupEntry pointing at it — a
                // bind group is immutable once created and does not follow a buffer
                // being replaced — hence the rebuild_bind_group() call whenever the size
                // actually changes.
                self.line(&format!("    fn copy_{}_to_device(&mut self, data: &[{}]) {{", f.name, host_ty));
                self.line(&format!("        let needed = (data.len() * std::mem::size_of::<{}>()) as u64;", host_ty));
                self.line(&format!("        if self.{}_buf.size() != needed {{", f.name));
                self.line(&format!("            self.{}_buf = std::sync::Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {{", f.name));
                self.line("                label: None,");
                self.line("                size: needed,");
                self.line(&format!("                usage: {},", buffer_usages(f)));
                self.line("                mapped_at_creation: false,");
                self.line("            }));");
                self.line("            self.rebuild_bind_group();");
                self.line("        }");
                self.line(&format!("        __boring_gpu_copy_h2d(&self.device, &self.queue, bytemuck::cast_slice(data), &self.{}_buf);", f.name));
                self.line("    }");
                self.blank();
            }
        }
    }

    // â"€â"€ Staging buffer helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

    fn emit_gpu_copy_helpers(&mut self) {
        // Dual-mode argument for a free function whose parameter is consumed
        // directly by a kernel constructor at a `'unified`/`'global` field position
        // (`Checker::fn_gpu_arg_params`, transpiler-side `fn_gpu_arg_params` mirror) —
        // lets a caller pass an already GPU-resident value straight through (an
        // `Arc::clone`, no round-trip) or a plain host array (uploaded as today) with
        // the same call syntax either way. See docs/scoped-access-blocks.md's
        // "Kernel Constructor Interaction".
        self.line("#[allow(dead_code)]");
        self.line("#[derive(Clone)]");
        self.line("enum BoringGpuArg<T> {");
        self.line("    Resident(std::sync::Arc<wgpu::Buffer>, usize),");
        self.line("    Host(Vec<T>),");
        self.line("}");
        self.blank();
        // `<param>.length`/`.count` (mapped to `.len()` by the general `map_field`
        // convention, same as a plain array) needs an inherent method here since a
        // dual-typed param doesn't materialize just to answer a size query -- a
        // `Resident` buffer already carries its element count.
        self.line("#[allow(dead_code)]");
        self.line("impl<T> BoringGpuArg<T> {");
        self.line("    fn len(&self) -> usize {");
        self.line("        match self {");
        self.line("            BoringGpuArg::Resident(_, len) => *len,");
        self.line("            BoringGpuArg::Host(v) => v.len(),");
        self.line("        }");
        self.line("    }");
        self.line("}");
        self.blank();

        // Staging-buffer pool for D2H readbacks. GPU targets run their whole
        // dispatch/readback chain on a single thread (see the GPU-target
        // `task`/stream/channel ban in `transpiler::mod.rs`'s
        // `emit_gpu_boring_main` -- there's no tokio runtime here, only
        // `pollster`), so a plain `thread_local!` is sufficient: no
        // cross-thread contention to guard against. Exact-size matching
        // (rather than a size-bucketing scheme) is enough because a given
        // call site's buffer shape doesn't change across calls (kernel field
        // sizes are fixed once allocated) -- the same handful of distinct
        // sizes recur call after call.
        self.line("thread_local! {");
        self.line("    static __BORING_STAGING_POOL: std::cell::RefCell<Vec<wgpu::Buffer>> = std::cell::RefCell::new(Vec::new());");
        self.line("}");
        self.blank();

        // D2H helper.
        self.line("fn __boring_gpu_copy_d2h<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer) -> Vec<T> {");
        self.line("    let size = src.size();");
        self.line("    let staging = __BORING_STAGING_POOL.with(|pool| {");
        self.line("        let mut pool = pool.borrow_mut();");
        self.line("        match pool.iter().position(|b| b.size() == size) {");
        self.line("            Some(i) => pool.swap_remove(i),");
        self.line("            None => device.create_buffer(&wgpu::BufferDescriptor {");
        self.line("                label: None,");
        self.line("                size,");
        self.line("                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,");
        self.line("                mapped_at_creation: false,");
        self.line("            }),");
        self.line("        }");
        self.line("    });");
        self.line("    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });");
        self.line("    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);");
        self.line("    queue.submit(std::iter::once(encoder.finish()));");
        self.line("    device.poll(wgpu::MaintainBase::Wait);");
        self.line("    let slice = staging.slice(..);");
        self.line("    let (tx, rx) = std::sync::mpsc::channel();");
        self.line("    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });");
        self.line("    device.poll(wgpu::MaintainBase::Wait);");
        self.line("    rx.recv().unwrap().unwrap();");
        self.line("    let data = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();");
        self.line("    staging.unmap();");
        self.line("    __BORING_STAGING_POOL.with(|pool| pool.borrow_mut().push(staging));");
        self.line("    data");
        self.line("}");
        self.blank();

        // H2D helper.
        self.line("fn __boring_gpu_copy_h2d(device: &wgpu::Device, queue: &wgpu::Queue, src: &[u8], dst: &wgpu::Buffer) {");
        self.line("    let size = src.len() as u64;");
        self.line("    let staging = device.create_buffer(&wgpu::BufferDescriptor {");
        self.line("        label: None,");
        self.line("        size,");
        self.line("        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,");
        self.line("        mapped_at_creation: true,");
        self.line("    });");
        self.line("    staging.slice(..).get_mapped_range_mut().copy_from_slice(src);");
        self.line("    staging.unmap();");
        self.line("    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });");
        self.line("    encoder.copy_buffer_to_buffer(&staging, 0, dst, 0, size);");
        self.line("    queue.submit(std::iter::once(encoder.finish()));");
        self.line("}");
        self.blank();

        // D2D helper -- a real device-to-device copy: allocate a fresh buffer
        // and issue a `copy_buffer_to_buffer` GPU command, NOT
        // `Arc::clone(&src)`. `Arc::clone` only bumps a reference count --
        // both kernel structs would end up pointing at the exact same
        // underlying `wgpu::Buffer`, so if the source kernel is ever
        // dispatched again afterward, the "copy"'s contents silently change
        // too, with no error (unlike cuda::host's/rocm::host's equivalent
        // bug, a real E0382 the Rust compiler catches instead). No manual
        // wait/poll is needed here before OR after the copy: `copy_buffer_to_buffer`
        // is itself a GPU command, and wgpu guarantees commands submitted to
        // the same queue execute in submission order, so it's already
        // correctly ordered after whatever wrote the source buffer, and
        // anything reading the new buffer later goes through the existing
        // D2H helper above, which already polls.
        self.line("fn __boring_gpu_copy_d2d(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer) -> std::sync::Arc<wgpu::Buffer> {");
        self.line("    let size = src.size();");
        self.line("    let dst = device.create_buffer(&wgpu::BufferDescriptor {");
        self.line("        label: None,");
        self.line("        size,");
        self.line("        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,");
        self.line("        mapped_at_creation: false,");
        self.line("    });");
        self.line("    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });");
        self.line("    encoder.copy_buffer_to_buffer(src, 0, &dst, 0, size);");
        self.line("    queue.submit(std::iter::once(encoder.finish()));");
        self.line("    std::sync::Arc::new(dst)");
        self.line("}");
        self.blank();

        if self.has_screen {
            self.emit_present_buffer_helper();
        }
    }

    fn emit_present_buffer_helper(&mut self) {
        // GPU-only present: runs a render pipeline that reads the storage buffer in a fragment
        // shader and outputs directly to the surface texture. No CPU readback, no blocking poll.
        // A non-blocking Poll before get_current_texture() flushes completed frames back to the
        // swapchain so get_current_texture() doesn't block the Wayland event loop.
        self.line("fn __boring_present_buffer(device: &wgpu::Device, queue: &wgpu::Queue, surface: &wgpu::Surface, pipeline: &wgpu::RenderPipeline, bind_group: &wgpu::BindGroup) {");
        self.line("    device.poll(wgpu::MaintainBase::Poll);");
        self.line("    let output = match surface.get_current_texture() { Ok(t) => t, Err(_) => return };");
        self.line("    let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());");
        self.line("    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });");
        self.line("    {");
        self.line("        let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {");
        self.line("            label: None,");
        self.line("            color_attachments: &[Some(wgpu::RenderPassColorAttachment {");
        self.line("                view: &view, resolve_target: None,");
        self.line("                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },");
        self.line("            })],");
        self.line("            depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,");
        self.line("        });");
        self.line("        rpass.set_pipeline(pipeline);");
        self.line("        rpass.set_bind_group(0, bind_group, &[]);");
        self.line("        rpass.draw(0..6, 0..1);");
        self.line("    }");
        self.line("    queue.submit(std::iter::once(enc.finish()));");
        self.line("    output.present();");
        self.line("}");
        self.blank();
    }

    // Returns the `kernel.field_buf` reference for the first screen.present() call in the loop.
    fn find_present_buffer(&self) -> String {
        let loop_body = self.find_render_loop_body();
        for stmt in &loop_body {
            if let Stmt::Expr(e) = stmt {
                let args = match &e.kind {
                    ExprKind::MethodCall(obj, method, args) if method == "present" => {
                        if let ExprKind::Var(n) = &obj.kind { if n == "screen" { Some(args) } else { None } } else { None }
                    }
                    ExprKind::Call(callee, args) => {
                        if let ExprKind::Field(obj, method) = &callee.kind {
                            if method == "present" {
                                if let ExprKind::Var(n) = &obj.kind { if n == "screen" { Some(args) } else { None } } else { None }
                            } else { None }
                        } else { None }
                    }
                    _ => None,
                };
                if let Some(args) = args {
                    if let Some(arg) = args.first() {
                        if let ExprKind::Field(kobj, kfield) = &arg.value.kind {
                            if let ExprKind::Var(kname) = &kobj.kind {
                                return format!("{kname}.{kfield}_buf");
                            }
                        }
                    }
                }
            }
        }
        "pixels_buf".into()
    }

    fn emit_main(&mut self) {
        if self.has_screen {
            self.emit_screen_main();
        } else {
            self.line("fn main() {");
            self.line("    pollster::block_on(async_main());");
            self.line("}");
            self.blank();
            self.line("async fn async_main() {");
            self.line("    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());");
            self.line("    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.expect(\"No GPU adapter found\");");
            if self.has_emulated_shader {
                // `gpu.warp.*` is used somewhere in this program — opportunistically
                // request `Features::SUBGROUP` (only if the adapter actually supports
                // it; `request_device` hard-fails if you require an unsupported
                // feature) so `device.features()` at shader-creation time can pick
                // the real-subgroup module over the shared-memory-emulated one.
                self.line("    let __boring_use_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);");
                self.line("    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {");
                self.line("        required_features: if __boring_use_subgroups { wgpu::Features::SUBGROUP } else { wgpu::Features::empty() },");
                self.line("        ..Default::default()");
                self.line("    }, None).await.expect(\"Failed to create device\");");
            } else {
                self.line("    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.expect(\"Failed to create device\");");
            }
            // Defensive: make sure the device is fully settled before any real
            // dispatch work begins. There's nothing queued yet, so this returns
            // essentially immediately -- it's here only to rule out adapter/device
            // warm-up timing as a source of flaky early-kernel behavior.
            self.line("    device.poll(wgpu::MaintainBase::Wait);");
            // Without this, a validation error NOT already inside an explicit
            // push/pop_error_scope pair (e.g. pipeline creation at kernel `new()`
            // time — see `emit_kernel_new`, no error scope wraps it there) hits
            // wgpu's default uncaptured-error handler, which panics. Installed
            // once here so such an error is reported and the program can
            // continue instead of crashing outright — a later dispatch() call on
            // a pipeline that failed to create will raise its own validation
            // error, which dispatch's own push/pop_error_scope DOES already
            // catch and convert to a proper GpuError::LaunchError.
            self.line("    device.on_uncaptured_error(Box::new(|e| eprintln!(\"boring: uncaptured GPU error: {}\", e)));");
            self.line("    let device = std::sync::Arc::new(device);");
            self.line("    let queue = std::sync::Arc::new(queue);");
            self.line("    let _ = __BORING_GPU_DEVICE.set(std::sync::Arc::clone(&device));");
            self.line("    let _ = __BORING_GPU_QUEUE.set(std::sync::Arc::clone(&queue));");
            // `wgpu::Adapter` has no `Clone` of its own -- Arc-wrap it here (same
            // convention as device/queue above) so `emit_gpu_adapter_enumeration`
            // can cheaply `Arc::clone` it into the introspection list below.
            self.line("    let adapter = std::sync::Arc::new(adapter);");
            self.emit_gpu_adapter_enumeration("adapter");
            self.blank();
            if self.has_boring_main {
                if self.boring_main_throws {
                    self.line("    if let Err(e) = boring_main() {");
                    self.line("        eprintln!(\"error: {}\", e);");
                    self.line("        std::process::exit(1);");
                    self.line("    }");
                } else {
                    self.line("    boring_main();");
                }
            }
            self.line("}");
        }
    }

    fn emit_screen_main(&mut self) {
        let (w_expr, h_expr, title, _) = self.extract_screen_info();
        let scalars     = self.extract_top_level_scalars();
        let width       = resolve_scalar_expr(&w_expr, &scalars);
        let height      = resolve_scalar_expr(&h_expr, &scalars);
        let loop_body   = self.find_render_loop_body();
        let pixels_buf  = self.find_present_buffer();

        let items: Vec<Item> = self.program.items.clone();

        // Collect scalar names and kernel instantiations for the app struct.
        let scalar_fields: Vec<(String, String)> = items.iter().filter_map(|item| {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    let is_screen = matches!(&val.kind, ExprKind::Call(c, _) if matches!(&c.kind, ExprKind::Var(n) if n == "Screen"));
                    let is_kernel = matches!(&val.kind, ExprKind::Call(c, _) if matches!(&c.kind, ExprKind::Var(kn) if self.kernel_names.contains(kn)));
                    if !is_screen && !is_kernel && matches!(val.kind, ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_)) {
                        return Some((s.name.clone(), host_expr(val)));
                    }
                }
            }
            None
        }).collect();

        let kernel_fields: Vec<(String, String, String)> = items.iter().filter_map(|item| {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    if let ExprKind::Call(callee, args) = &val.kind {
                        if let ExprKind::Var(kname) = &callee.kind {
                            if self.kernel_names.contains(kname) {
                                // Only pass width/height through to `Kernel::new(...)` when
                                // the kernel actually declares a `Dimension` field --
                                // `emit_kernel_new` only adds those parameters in that case,
                                // and doing it unconditionally on the constructor call's
                                // shape (any `Kernel(Dimension(w, h))`) was an arity mismatch
                                // for kernels sized some other way (E0061).
                                let decl = self.effective_kernels.iter().find(|d| &d.name == kname);
                                let has_dim_field = decl.is_some_and(kernel_has_dim_field);
                                let new_args = if has_dim_field {
                                    extract_dimension_args(args.first())
                                        .map(|(w, h)| format!("{w}, {h}, "))
                                        .unwrap_or_default()
                                } else {
                                    String::new()
                                };
                                return Some((s.name.clone(), kname.clone(), new_args));
                            }
                        }
                    }
                }
            }
            None
        }).collect();

        // ── __App struct ──────────────────────────────────────────────────────
        self.line("struct __App {");
        self.line("    instance: wgpu::Instance,");
        // Arc-wrapped, not a bare `wgpu::Adapter` -- `wgpu::Adapter` has no `Clone`
        // of its own, and the introspection list (`__BORING_GPU_ADAPTERS`) needs a
        // second, cheap owner of the same adapter alongside this struct field's own.
        self.line("    adapter:  std::sync::Arc<wgpu::Adapter>,");
        self.line("    device:   std::sync::Arc<wgpu::Device>,");
        self.line("    queue:    std::sync::Arc<wgpu::Queue>,");
        for (name, _val) in &scalar_fields {
            self.line(&format!("    {name}: i64,"));
        }
        for (var, kname, _) in &kernel_fields {
            self.line(&format!("    {var}: {kname},"));
        }
        self.line("    __keys: std::collections::HashSet<String>,");
        self.line("    __start_time: std::time::Instant,");
        self.line("    window: Option<std::sync::Arc<Window>>,");
        self.line("    surface: Option<wgpu::Surface<'static>>,");
        self.line("    __blit_pipeline: Option<wgpu::RenderPipeline>,");
        self.line("    __blit_bg:       Option<wgpu::BindGroup>,");
        self.line("    __blit_dim_buf:  Option<wgpu::Buffer>,");
        self.line("}");
        self.blank();

        // ── ApplicationHandler impl ───────────────────────────────────────────
        self.line("impl ApplicationHandler for __App {");
        // resumed(): create window + surface + blit pipeline
        self.line("    fn resumed(&mut self, event_loop: &ActiveEventLoop) {");
        self.line(&format!(
            "        let window = std::sync::Arc::new(event_loop.create_window(WindowAttributes::default().with_title(\"{title}\").with_inner_size(winit::dpi::PhysicalSize::new({width}u32, {height}u32))).unwrap());"
        ));
        self.line("        let surface: wgpu::Surface<'static> = self.instance.create_surface(window.clone()).unwrap();");
        self.line("        let caps = surface.get_capabilities(&self.adapter);");
        self.line("        let fmt  = caps.formats.iter().find(|f| **f == wgpu::TextureFormat::Bgra8Unorm).copied().unwrap_or(caps.formats[0]);");
        self.line("        surface.configure(&self.device, &wgpu::SurfaceConfiguration {");
        self.line("            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt,");
        self.line(&format!("            width: {width}, height: {height},"));
        self.line("            present_mode: wgpu::PresentMode::Fifo,");
        self.line("            desired_maximum_frame_latency: 2,");
        self.line("            alpha_mode: caps.alpha_modes[0], view_formats: vec![],");
        self.line("        });");
        // Blit pipeline (inline shader)
        self.line("        let blit_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {");
        self.line("            label: Some(\"blit\"),");
        self.line("            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(r#\"");
        self.line("@group(0) @binding(0) var<storage, read> blit_pixels: array<u32>;");
        self.line("struct BlitDim { width: u32, height: u32 }");
        self.line("@group(0) @binding(1) var<uniform> blit_dim: BlitDim;");
        self.line("struct BlitVOut { @builtin(position) pos: vec4<f32> }");
        self.line("@vertex fn blit_vs(@builtin(vertex_index) idx: u32) -> BlitVOut {");
        self.line("    var quad = array<vec2<f32>, 6>(");
        self.line("        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0,  1.0),");
        self.line("        vec2<f32>(-1.0,  1.0), vec2<f32>(1.0, -1.0), vec2<f32>( 1.0,  1.0));");
        self.line("    var out: BlitVOut; out.pos = vec4<f32>(quad[idx], 0.0, 1.0); return out; }");
        self.line("@fragment fn blit_fs(in: BlitVOut) -> @location(0) vec4<f32> {");
        self.line("    let x = u32(in.pos.x); let y = u32(in.pos.y);");
        self.line("    let i = y * blit_dim.width + x;");
        self.line("    if i >= blit_dim.width * blit_dim.height { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }");
        self.line("    let raw = blit_pixels[i];");
        self.line("    let r = f32((raw >>  0u) & 0xFFu) / 255.0;");
        self.line("    let g = f32((raw >>  8u) & 0xFFu) / 255.0;");
        self.line("    let b = f32((raw >> 16u) & 0xFFu) / 255.0;");
        self.line("    let a = f32((raw >> 24u) & 0xFFu) / 255.0;");
        self.line("    return vec4<f32>(r, g, b, a); }");
        self.line("            \"#)),");
        self.line("        });");
        self.line("        let blit_bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {");
        self.line("            label: None, entries: &[");
        self.line("                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,");
        self.line("                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true },");
        self.line("                        has_dynamic_offset: false, min_binding_size: None }, count: None },");
        self.line("                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,");
        self.line("                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform,");
        self.line("                        has_dynamic_offset: false, min_binding_size: None }, count: None },");
        self.line("            ],");
        self.line("        });");
        self.line("        let blit_dim_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {");
        self.line("            label: Some(\"blit_dim\"),");
        self.line(&format!("            contents: bytemuck::cast_slice(&[{width}u32, {height}u32]),"));
        self.line("            usage: wgpu::BufferUsages::UNIFORM,");
        self.line("        });");
        self.line("        let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {");
        self.line("            label: None, layout: &blit_bgl, entries: &[");
        self.line(&format!("                wgpu::BindGroupEntry {{ binding: 0, resource: self.{pixels_buf}.as_entire_binding() }},"));
        self.line("                wgpu::BindGroupEntry { binding: 1, resource: blit_dim_buf.as_entire_binding() },");
        self.line("            ],");
        self.line("        });");
        self.line("        let blit_pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {");
        self.line("            label: None, bind_group_layouts: &[&blit_bgl], push_constant_ranges: &[],");
        self.line("        });");
        self.line("        let blit_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {");
        self.line("            label: None, layout: Some(&blit_pl),");
        self.line("            vertex: wgpu::VertexState { module: &blit_shader, entry_point: \"blit_vs\", buffers: &[],");
        self.line("                compilation_options: wgpu::PipelineCompilationOptions::default() },");
        self.line("            fragment: Some(wgpu::FragmentState { module: &blit_shader, entry_point: \"blit_fs\",");
        self.line("                targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: None,");
        self.line("                    write_mask: wgpu::ColorWrites::ALL })],");
        self.line("                compilation_options: wgpu::PipelineCompilationOptions::default() }),");
        self.line("            primitive: wgpu::PrimitiveState::default(), depth_stencil: None,");
        self.line("            multisample: wgpu::MultisampleState::default(), multiview: None, cache: None,");
        self.line("        });");
        self.line("        self.window = Some(window.clone());");
        self.line("        self.surface = Some(surface);");
        self.line("        self.__blit_pipeline = Some(blit_pipeline);");
        self.line("        self.__blit_bg = Some(blit_bg);");
        self.line("        self.__blit_dim_buf = Some(blit_dim_buf);");
        self.line("        window.request_redraw();");
        self.line("    }");
        self.blank();
        // window_event(): handle all window events
        self.line("    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {");
        self.line("        match event {");
        self.line("            WindowEvent::CloseRequested => { event_loop.exit(); }");
        self.line("            WindowEvent::KeyboardInput { event: ref key_event, .. } => {");
        self.line("                let key_str = match &key_event.logical_key {");
        self.line("                    Key::Named(NamedKey::Escape) => \"Escape\".to_string(),");
        self.line("                    Key::Named(NamedKey::Enter)  => \"Enter\".to_string(),");
        self.line("                    Key::Named(NamedKey::Space)  => \" \".to_string(),");
        self.line("                    Key::Character(c) => c.to_string(),");
        self.line("                    k => format!(\"{k:?}\"),");
        self.line("                };");
        self.line("                match key_event.state {");
        self.line("                    ElementState::Pressed  => { self.__keys.insert(key_str); }");
        self.line("                    ElementState::Released => { self.__keys.remove(&key_str); }");
        self.line("                    _ => {}");
        self.line("                }");
        self.line("            }");
        self.line("            WindowEvent::RedrawRequested => {");
        // Render loop body with self. prefix
        self.in_method = true;
        let w = width.clone();
        let h = height.clone();
        for stmt in &loop_body {
            self.emit_render_stmt(stmt, &w, &h, "                ");
        }
        self.in_method = false;
        self.line("                if let Some(w) = &self.window { w.request_redraw(); }");
        self.line("            }");
        self.line("            _ => {}");
        self.line("        }");
        self.line("    }");
        self.line("}");
        self.blank();

        // ── fn main() ─────────────────────────────────────────────────────────
        self.line("fn main() {");
        self.line("    let (instance, adapter, device, queue) = pollster::block_on(async {");
        self.line("        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());");
        self.line("        let adapter  = instance.request_adapter(&wgpu::RequestAdapterOptions::default())");
        self.line("            .await.expect(\"No GPU adapter found\");");
        self.line("        let (device, queue) = adapter");
        if self.has_emulated_shader {
            self.line("            .request_device(&wgpu::DeviceDescriptor {");
            self.line("                required_features: if adapter.features().contains(wgpu::Features::SUBGROUP) { wgpu::Features::SUBGROUP } else { wgpu::Features::empty() },");
            self.line("                ..Default::default()");
            self.line("            }, None)");
            self.line("            .await.expect(\"Failed to create device\");");
        } else {
            self.line("            .request_device(&wgpu::DeviceDescriptor::default(), None)");
            self.line("            .await.expect(\"Failed to create device\");");
        }
        self.line("        (instance, adapter, device, queue)");
        self.line("    });");
        // Defensive: see the non-Screen async_main path for why.
        self.line("    device.poll(wgpu::MaintainBase::Wait);");
        // See the non-Screen async_main path's identical call for why: reports
        // (instead of panicking on) a validation error not already inside an
        // explicit error scope, e.g. pipeline creation at kernel `new()` time.
        self.line("    device.on_uncaptured_error(Box::new(|e| eprintln!(\"boring: uncaptured GPU error: {}\", e)));");
        self.line("    let device = std::sync::Arc::new(device);");
        self.line("    let queue  = std::sync::Arc::new(queue);");
        self.line("    let _ = __BORING_GPU_DEVICE.set(std::sync::Arc::clone(&device));");
        self.line("    let _ = __BORING_GPU_QUEUE.set(std::sync::Arc::clone(&queue));");
        // `wgpu::Adapter` has no `Clone` of its own -- Arc-wrap it here (same
        // convention as device/queue above, and the `__App.adapter` field is typed
        // `Arc<wgpu::Adapter>` to match) so it can be `Arc::clone`d into the
        // introspection list below AND still moved into the `__App` struct literal
        // further down. This path used to skip GPU introspection entirely ("GPU
        // introspection is unsupported inside a Screen program"). Populating the
        // adapters global here removes the panic a `GPU(n)` call anywhere reachable
        // would have hit against an uninitialized `OnceLock`. `emit_screen_main`'s
        // own top-level (`scalar_fields`/`kernel_fields` above) and render-loop
        // (`emit_render_stmt`) emitters are still narrow, hardcoded shape-matchers,
        // not a splice into the general per-statement pipeline the way a plain
        // `def` function's body is -- but a bare `let g = GPU(0)` / `print`
        // statement at a Screen program's top level, or inside the render loop, no
        // longer vanishes silently: `check_screen_program_shape` (called from
        // `emit()`) and `emit_render_stmt`'s own error fallback both reject anything
        // outside the recognized whitelist with a `TranspileError` instead, so
        // `boring build` fails loudly (see `emit_host_rs`'s doc comment) rather than
        // quietly losing code. `emit_gpu_adapter_enumeration` is the same call the
        // non-Screen async_main path makes.
        self.line("    let adapter = std::sync::Arc::new(adapter);");
        self.emit_gpu_adapter_enumeration("adapter");
        for (name, val) in &scalar_fields {
            self.line(&format!("    let {name}: i64 = {val};"));
        }
        // Kernel instantiations in main() (need device/queue before struct init)
        for (var, kname, new_args) in &kernel_fields {
            // new_args may be "width, height, " — cast to i32 for the kernel constructor.
            let new_args_cast = if new_args.is_empty() { String::new() } else {
                new_args.split(", ").filter(|s| !s.is_empty())
                    .map(|a| format!("{a} as i32"))
                    .collect::<Vec<_>>().join(", ") + ", "
            };
            self.line(&format!(
                "    let {var} = {kname}::new({new_args_cast}std::sync::Arc::clone(&device), std::sync::Arc::clone(&queue));"
            ));
        }
        // Bug 3 fix: seed 'surface' array buffers with random initial data.
        for (var, kname, _) in &kernel_fields {
            let surface_fields: Vec<String> = self.program.items.iter().find_map(|item| {
                if let Item::Kernel(decl) = item {
                    if &decl.name == kname {
                        let fields: Vec<String> = decl.fields.iter()
                            .filter(|f| matches!(f.qual, GpuQual::Surface)
                                     && is_buffer_array_ty(&f.ty))
                            .map(|f| f.name.clone())
                            .collect();
                        return Some(fields);
                    }
                }
                None
            }).unwrap_or_default();
            for field in &surface_fields {
                self.line("    {");
                self.line("        let mut __rng: u64 = 0x12345678ABCDEF01u64;");
                self.line(&format!("        let __n = ({width} * {height}) as usize;"));
                self.line("        let __rand_data: Vec<u32> = (0..__n).map(|_| { __rng ^= __rng << 13; __rng ^= __rng >> 7; __rng ^= __rng << 17; if __rng % 10 < 3 { 1u32 } else { 0u32 } }).collect();");
                self.line(&format!("        queue.write_buffer(&{var}.{field}_buf, 0, bytemuck::cast_slice(&__rand_data));"));
                self.line("    }");
            }
        }
        self.line("    let mut app = __App {");
        self.line("        instance, adapter, device, queue,");
        for (name, val) in &scalar_fields {
            self.line(&format!("        {name}: {val},"));
        }
        for (var, _, _) in &kernel_fields {
            self.line(&format!("        {var},"));
        }
        self.line("        __keys: Default::default(),");
        self.line("        __start_time: std::time::Instant::now(),");
        self.line("        window: None, surface: None,");
        self.line("        __blit_pipeline: None, __blit_bg: None, __blit_dim_buf: None,");
        self.line("    };");
        self.line("    let event_loop = EventLoop::new().unwrap();");
        self.line("    event_loop.run_app(&mut app).unwrap_or_else(|_| {});");
        self.line("}");
    }

    // -- Screen helpers -------------------------------------------------------

    fn extract_screen_info(&self) -> (String, String, String, String) {
        for item in &self.program.items {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    if let ExprKind::Call(callee, args) = &val.kind {
                        if let ExprKind::Var(n) = &callee.kind {
                            if n == "Screen" {
                                let (w, h) = args.first()
                                    .and_then(|a| {
                                        if let ExprKind::Call(dc, da) = &a.value.kind {
                                            if let ExprKind::Var(dn) = &dc.kind {
                                                if dn == "Dimension" {
                                                    let w = da.first().map(|x| host_expr(&x.value)).unwrap_or_else(|| "800".into());
                                                    let h = da.get(1).map(|x| host_expr(&x.value)).unwrap_or_else(|| "600".into());
                                                    return Some((w, h));
                                                }
                                            }
                                        }
                                        None
                                    })
                                    .unwrap_or_else(|| ("800".into(), "600".into()));
                                let title = args.iter()
                                    .find(|a| a.label.as_deref() == Some("title"))
                                    .and_then(|a| if let ExprKind::Str(sv) = &a.value.kind { Some(sv.clone()) } else { None })
                                    .unwrap_or_else(|| "boring".into());
                                return (w, h, title, s.name.clone());
                            }
                        }
                    }
                }
            }
        }
        ("800".into(), "600".into(), "boring".into(), "screen".into())
    }

    fn extract_top_level_scalars(&self) -> std::collections::HashMap<String, i64> {
        let mut map = std::collections::HashMap::new();
        for item in &self.program.items {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    if let ExprKind::Int(n) = &val.kind { map.insert(s.name.clone(), *n); }
                }
            }
        }
        map
    }

    fn find_render_loop_body(&self) -> Vec<Stmt> {
        for item in &self.program.items {
            if let Item::Stmt(Stmt::KernelBlock(block)) = item {
                for stmt in &block.body {
                    if let Stmt::Loop(ls) = stmt { return ls.body.clone(); }
                }
            }
        }
        vec![]
    }

    fn emit_render_stmt(&mut self, stmt: &Stmt, width: &str, height: &str, indent: &str) {
        match stmt {
            Stmt::Expr(e) => {
                if let Some(s) = self.try_emit_screen_present(e, width, height) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_render_dispatch(e, width, height) {
                    self.line(&format!("{indent}{s};"));
                } else if let Some(s) = self.try_emit_field_swap(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_field_assign(e) {
                    self.line(&format!("{indent}{s}"));
                } else {
                    self.push_error(e.line, e.col,
                        "wgpu target: expression not supported inside a Screen render loop");
                }
            }
            Stmt::If(if_stmt) => {
                if let Some(s) = self.try_emit_screen_key_if(if_stmt) {
                    self.line(&format!("{indent}{s}"));
                } else {
                    self.push_error(if_stmt.line, if_stmt.col,
                        "wgpu target: `if` inside a Screen render loop only supports \
                         `if screen.key(\"...\"):  break` (a single, break-only branch)");
                }
            }
            // A bare, unconditional `break` (no enclosing `if`) means "stop after this
            // frame" -- there's no real Rust `loop` here to `break` out of (each frame
            // is one `WindowEvent::RedrawRequested` callback, driven by winit's own
            // event loop, not a Boring-generated loop construct), so this maps to the
            // same `event_loop.exit()` call `if screen.key(...): break` uses. Previously
            // this fell through the old catch-all `_ => {}` silently -- the loop simply
            // never stopped, which is exactly the kind of quiet miscompile this checker
            // now catches everywhere else; support it for real instead of erroring.
            Stmt::Break(_, None) => {
                let exit_call = if self.in_method { "event_loop.exit();" } else { "elwt.exit();" };
                self.line(&format!("{indent}{exit_call}"));
            }
            Stmt::Break(line, Some(_)) => {
                self.push_error(*line, 0,
                    "wgpu target: `break` with a value is not supported inside a Screen render loop");
            }
            other => {
                let (l, c) = stmt_line_col(other);
                self.push_error(l, c,
                    "wgpu target: statement not supported inside a Screen render loop");
            }
        }
    }

    fn try_emit_screen_present(&self, e: &Expr, _width: &str, _height: &str) -> Option<String> {
        let (obj_name, args) = match &e.kind {
            ExprKind::MethodCall(obj, method, args) if method == "present" => {
                if let ExprKind::Var(n) = &obj.kind { (n.as_str(), args) } else { return None; }
            }
            ExprKind::Call(callee, args) => {
                if let ExprKind::Field(obj, method) = &callee.kind {
                    if method != "present" { return None; }
                    if let ExprKind::Var(n) = &obj.kind { (n.as_str(), args) } else { return None; }
                } else { return None; }
            }
            _ => return None,
        };
        if obj_name != "screen" { return None; }
        // Verify the argument is a field access (e.g. render.pixels) — if not, bail.
        let arg = args.first()?;
        if !matches!(&arg.value.kind, ExprKind::Field(_, _)) { return None; }
        if self.in_method {
            Some("__boring_present_buffer(&self.device, &self.queue, self.surface.as_ref().unwrap(), self.__blit_pipeline.as_ref().unwrap(), self.__blit_bg.as_ref().unwrap());".into())
        } else {
            Some("__boring_present_buffer(&device, &queue, &surface, &__blit_pipeline, &__blit_bg);".into())
        }
    }

    fn try_emit_render_dispatch(&self, e: &Expr, width: &str, height: &str) -> Option<String> {
        let ExprKind::Call(callee, args) = &e.kind else { return None; };
        if !args.iter().any(|a| a.label.as_deref() == Some("block")) { return None; }
        let var_name = if let ExprKind::Var(n) = &callee.kind { n.clone() } else { return None; };
        let ba = args.iter().find(|a| a.label.as_deref() == Some("block"))?;
        let (bx, by) = match &ba.value.kind {
            ExprKind::Tuple(elems) => {
                let g = |i: usize| elems.get(i).map(host_expr).unwrap_or_else(|| "1".into());
                (g(0), g(1))
            }
            _ => (host_expr(&ba.value), "1".into()),
        };
        let gx = format!("({width} + {bx} - 1) / {bx}");
        let gy = format!("({height} + {by} - 1) / {by}");
        let pfx = if self.in_method { "self." } else { "" };
        // Inside the render loop's `event_loop.run(...)` closure, not a
        // `Result`-returning context -- `.expect()`, not `?`. Matches
        // `metal::host`'s identical carve-out for its own render-loop dispatch.
        Some(format!("{pfx}{var_name}.dispatch({gx}, {gy}, 1).expect(\"wgpu: kernel dispatch rejected\")"))
    }

    fn try_emit_screen_key_if(&self, if_stmt: &IfStmt) -> Option<String> {
        if if_stmt.branches.len() != 1 { return None; }
        let (cond, body) = &if_stmt.branches[0];
        let (obj_name, args) = match &cond.kind {
            ExprKind::MethodCall(obj, method, args) if method == "key" => {
                if let ExprKind::Var(n) = &obj.kind { (n.as_str(), args) } else { return None; }
            }
            ExprKind::Call(callee, args) => {
                if let ExprKind::Field(obj, method) = &callee.kind {
                    if method != "key" { return None; }
                    if let ExprKind::Var(n) = &obj.kind { (n.as_str(), args) } else { return None; }
                } else { return None; }
            }
            _ => return None,
        };
        if obj_name != "screen" { return None; }
        let key_arg = args.first()?;
        let key_str = if let ExprKind::Str(s) = &key_arg.value.kind { s.clone() } else { return None; };
        let key_rust = boring_key_to_rust(&key_str);
        let is_break = body.iter().any(|s| matches!(s, Stmt::Break(_, _)));
        if is_break {
            let (keys_var, exit_call) = if self.in_method {
                ("self.__keys", "event_loop.exit()")
            } else {
                ("__keys", "elwt.exit()")
            };
            Some(format!("if {keys_var}.contains({key_rust}) {{ {exit_call}; }}"))
        } else {
            Some(format!("// screen.key({key_rust:?}) -- no action"))
        }
    }


    /// Detect simultaneous field swap: `k.a, k.b = k.b, k.a`
    /// Emits `std::mem::swap(&mut k.a_buf, &mut k.b_buf);` for buffer fields.
    fn try_emit_field_swap(&self, e: &Expr) -> Option<String> {
        let ExprKind::Assign(lhs, rhs) = &e.kind else { return None; };
        let ExprKind::Tuple(lhs_list) = &lhs.kind else { return None; };
        let ExprKind::Tuple(rhs_list) = &rhs.kind else { return None; };
        if lhs_list.len() != 2 || rhs_list.len() != 2 { return None; }
        // Check it's a swap: lhs[0]==rhs[1] and lhs[1]==rhs[0] (by expression text)
        let field_name = |e: &Expr| -> Option<(String, String)> {
            if let ExprKind::Field(obj, field) = &e.kind {
                if let ExprKind::Var(obj_name) = &obj.kind {
                    return Some((obj_name.clone(), field.clone()));
                }
            }
            None
        };
        let (obj0, f0) = field_name(&lhs_list[0])?;
        let (obj1, f1) = field_name(&lhs_list[1])?;
        let (robj0, rf0) = field_name(&rhs_list[0])?;
        let (robj1, rf1) = field_name(&rhs_list[1])?;
        if obj0 == robj1 && f0 == rf1 && obj1 == robj0 && f1 == rf0 {
            // It's a swap of two fields on the same object
            if obj0 == obj1 {
                let pfx = if self.in_method { "self." } else { "" };
                return Some(format!(
                    "std::mem::swap(&mut {pfx}{}.{}_buf, &mut {pfx}{}.{}_buf);",
                    obj0, f0, obj1, f1
                ));
            }
        }
        None
    }

    /// Detect field assignment: `k.field = expr`
    fn try_emit_field_assign(&self, e: &Expr) -> Option<String> {
        let ExprKind::Assign(lhs, rhs) = &e.kind else { return None; };
        if let ExprKind::Field(obj, field) = &lhs.kind {
            if let ExprKind::Var(obj_name) = &obj.kind {
                // `step.cells_in = step.cells_out` â†' bind buffer alias
                if let ExprKind::Field(robj, rfield) = &rhs.kind {
                    if let ExprKind::Var(robj_name) = &robj.kind {
                        // Try to find the target kernel decl to rebuild its bind group.
                        if let Some(target_decl) = self.find_kernel_decl_for_var(obj_name) {
                            let pfx = if self.in_method { "self." } else { "" };
                            let buf_fields: Vec<&KernelFieldDecl> = target_decl.fields.iter()
                                .filter(|f| matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::Surface | GpuQual::ActorGlobal | GpuQual::ActorUnified)
                                         && is_buffer_array_ty(&f.ty))
                                .collect();
                            let params_fields: Vec<&KernelFieldDecl> = target_decl.fields.iter()
                                .filter(|f| is_params_field(f)).collect();
                            let mut entries: Vec<String> = vec![];
                            for (idx, f) in buf_fields.iter().enumerate() {
                                if f.name == field.as_str() {
                                    entries.push(format!(
                                        "                wgpu::BindGroupEntry {{ binding: {idx}, resource: {pfx}{robj_name}.{rfield}_buf.as_entire_binding() }},"));
                                } else {
                                    entries.push(format!(
                                        "                wgpu::BindGroupEntry {{ binding: {idx}, resource: {pfx}{obj_name}.{}_buf.as_entire_binding() }},", f.name));
                                }
                            }
                            if !params_fields.is_empty() {
                                let idx = buf_fields.len();
                                entries.push(format!(
                                    "                wgpu::BindGroupEntry {{ binding: {idx}, resource: {pfx}{obj_name}.params_buf.as_entire_binding() }},"));
                            }
                            let entries_str = entries.join("\n");
                            return Some(format!(
                                "{pfx}{obj_name}.bind_group = {pfx}{obj_name}.device.create_bind_group(&wgpu::BindGroupDescriptor {{\n    label: None,\n    layout: &{pfx}{obj_name}.pipeline.get_bind_group_layout(0),\n    entries: &[\n{entries_str}\n    ],\n}});"
                            ));
                        }
                        return Some(format!(
                            "// {}.{} = {}.{} (field alias not yet supported in wgpu)",
                            obj_name, field, robj_name, rfield
                        ));
                    }
                }
                // Scalar field assignment: `k.t = expr` → update the host-side field.
                let pfx = if self.in_method { "self." } else { "" };
                let rhs_str = self.host_expr(rhs);
                return Some(format!("{pfx}{obj_name}.{field} = {rhs_str};"));
            }
        }
        None
    }

    /// Translate a host-side Boring expression to a Rust string.
    /// Handles the subset of expressions that can appear in the top-level kernel loop
    /// (scalar assignments, screen.time, float() casts, arithmetic).
    fn host_expr(&self, e: &Expr) -> String {
        let pfx = if self.in_method { "self." } else { "" };
        match &e.kind {
            // screen.time → elapsed seconds as f32
            ExprKind::Field(obj, field) if field == "time" => {
                if let ExprKind::Var(n) = &obj.kind {
                    if n == "screen" {
                        return format!("{pfx}__start_time.elapsed().as_secs_f32()");
                    }
                }
                format!("{pfx}{}.{}", self.host_expr(obj), field)
            }
            ExprKind::Field(obj, field) => {
                format!("{}.{}", self.host_expr(obj), field)
            }
            // float32(expr) → expr as f32 (wgpu is f32-only on device, and this
            // host helper narrows the same way regardless of width). Bare
            // `float(expr)` must NOT join float32 here — `float` is a pure alias
            // of `float64` (see host_scalar_type/CLAUDE.md), not its own type, so
            // it belongs with the float64 branch below (a real f64). Grouping it
            // with float32 mis-cast a `var float t` field's assignment to f32,
            // mismatching the f64 field it actually is (plasma_metal.br's wgpu
            // regression).
            ExprKind::Call(callee, args) => {
                if let ExprKind::Var(n) = &callee.kind {
                    if n == "float32" && args.len() == 1 {
                        return format!("({} as f32)", self.host_expr(&args[0].value));
                    }
                    if (n == "float" || n == "float64") && args.len() == 1 {
                        return format!("({} as f64)", self.host_expr(&args[0].value));
                    }
                }
                // Generic call (best effort)
                let args_s: Vec<_> = args.iter().map(|a| self.host_expr(&a.value)).collect();
                format!("{}({})", self.host_expr(callee), args_s.join(", "))
            }
            ExprKind::Var(n) => n.clone(),
            ExprKind::Int(v)   => format!("{v}"),
            ExprKind::Float(v) => format!("{v}f32"),
            ExprKind::BinOp(op, l, r) => {
                let op_s = match op {
                    BinOp::Add => "+", BinOp::Sub => "-",
                    BinOp::Mul => "*", BinOp::Div => "/",
                    _ => "/* op */",
                };
                format!("({} {op_s} {})", self.host_expr(l), self.host_expr(r))
            }
            _ => "/* unsupported expr */".into(),
        }
    }

    /// Find the KernelDecl for a top-level kernel variable (e.g. `var render = Render(...)`).
    /// Returns the monomorphised decl from `effective_kernels`.
    fn find_kernel_decl_for_var(&self, var_name: &str) -> Option<&KernelDecl> {
        let ktype_name = self.program.items.iter().find_map(|item| {
            if let Item::Let(s) = item {
                if s.name == var_name {
                    if let Some(val) = &s.value {
                        match &val.kind {
                            ExprKind::Call(callee, _) => {
                                if let ExprKind::Var(kn) = &callee.kind {
                                    if self.kernel_names.contains(kn) {
                                        return Some(kn.clone());
                                    }
                                }
                            }
                            ExprKind::GenericCall(callee, type_args, _) => {
                                if let ExprKind::Var(kn) = &callee.kind {
                                    if self.kernel_names.contains(kn) {
                                        // Build the monomorphised name from the concrete args.
                                        let args: Vec<i64> = type_args.iter().map(|t| {
                                            if let Type::ConstInt(n) = t { *n } else { 0 }
                                        }).collect();
                                        if args.is_empty() {
                                            return Some(kn.clone());
                                        }
                                        let suffix: Vec<String> = args.iter().map(|n| n.to_string()).collect();
                                        return Some(format!("{}_{}", kn, suffix.join("_")));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            None
        })?;
        // Look up in effective_kernels (monomorphised) — the first match for this name.
        self.effective_kernels.iter().find(|decl| decl.name == ktype_name)
    }
}

//â"€â"€ Free helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

/// Best-effort `(line, col)` for an arbitrary `Stmt`, used to attribute an
/// "unsupported statement" error (see `check_screen_top_level_stmt`/
/// `emit_render_stmt`) to real source text. Covers every shape that can
/// plausibly appear at a Screen program's top level or inside its render
/// loop; anything else (e.g. `Stmt::Comment`, which carries no position)
/// falls back to `(0, 0)`, matching `SourceError::at_line`'s own convention
/// for "position unknown" (`report_transpile_errors` still prints the
/// message, just without a caret).
fn stmt_line_col(stmt: &Stmt) -> (usize, usize) {
    match stmt {
        Stmt::Let(s) => (s.line, s.col),
        Stmt::LetDestructure(s) => (s.line, s.col),
        Stmt::Return(s) => (s.line, s.col),
        Stmt::Break(line, _) => (*line, 0),
        Stmt::Continue(line) => (*line, 0),
        Stmt::Throw(s) => (s.line, s.col),
        Stmt::If(s) => (s.line, s.col),
        Stmt::IfLet(s) => (s.line, s.col),
        Stmt::Match(s) => (s.line, s.col),
        Stmt::While(s) => (s.line, s.col),
        Stmt::WhileLet(s) => (s.line, s.col),
        Stmt::DoWhile(s) => (s.line, s.col),
        Stmt::Loop(s) => (s.line, s.col),
        Stmt::Wait(e, line) => (*line, e.col),
        Stmt::For(s) => (s.line, s.col),
        Stmt::Guard(s) => (s.line, s.col),
        Stmt::Try(s) => (s.line, s.col),
        Stmt::Expr(e) => (e.line, e.col),
        Stmt::Fn(s) => (s.line, s.col),
        Stmt::Struct(s) => (s.line, s.col),
        Stmt::Enum(s) => (s.line, s.col),
        Stmt::Mod(s) => (s.line, s.col),
        Stmt::Alias(s) => (s.line, s.col),
        Stmt::Yield(e, line) => (*line, e.col),
        Stmt::KernelBlock(s) => (s.line, s.col),
        Stmt::With(s) => (s.line, s.col),
        Stmt::Defer(_) | Stmt::Comment(_) => (0, 0),
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
        ExprKind::UnaryOp(UnaryOp::Neg, inner) => match &inner.kind {
            ExprKind::Int(n)   => format!("-{}", n),
            ExprKind::Float(f) => format!("-{}", f),
            _ => "Default::default()".into(),
        },
        _ => "Default::default()".into(),
    }
}

fn is_params_field(f: &KernelFieldDecl) -> bool {
    match f.qual {
        GpuQual::Const => true,
        GpuQual::Local => !is_buffer_array_ty(&f.ty),
        _ => {
            matches!(&f.ty, Type::Named(n) if n != "Dimension")
                && !matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface)
        }
    }
}

/// True for a boring array type that gets a real storage-buffer field on the
/// generated host struct: a plain `[T]`/`[T; N]`, or a labeled multi-dim
/// array (`Type::LabeledArray`, fixed- or dynamic-shape alike). Mirrors
/// `wgpu::device::is_buffer_field`'s recognition of `LabeledArray` — that
/// side of the backend picked it up when multi-dimensional arrays landed,
/// but every one of *this* file's own "is this a buffer field" checks still
/// only matched `Array`/`ArrayN`, so a `[T, width = .., height = ..]'global`
/// field's buffer, struct field, and bind-group entry were all silently
/// dropped (matrix_mul_gpu.br's wgpu regression).
fn is_buffer_array_ty(ty: &Type) -> bool {
    matches!(ty, Type::Array(_) | Type::ArrayN(_, _)) || ty.as_labeled_array().is_some()
}

fn array_inner(ty: &Type) -> Type {
    match ty {
        Type::Array(inner) | Type::ArrayN(inner, _) => (**inner).clone(),
        ty if ty.as_labeled_array().is_some() => ty.as_labeled_array().unwrap().0.clone(),
        other => other.clone(),
    }
}

fn host_scalar_type(ty: &Type) -> &'static str {
    match ty {
        // GPU buffers always use 32-bit elements (WGSL narrows int→i32, uint→u32, float→f32).
        // This host-side mirror type must match the device layout, so it stays i32/u32 even
        // though `int`/`uint` are 64-bit (isize/usize) on the host and on every other GPU
        // backend — any host value outside i32/u32 range silently wraps once packed into
        // this buffer. The functional diagnostic for this narrowing is emitted on the device
        // side, see `wgsl_narrowed_width` in device.rs (the generated shaders/main.wgsl
        // itself names the risk); this comment documents the host-side half of the same limit.
        Type::Int   => "i32",
        Type::Uint  => "u32",
        Type::Float32 => "f32",
        Type::Bool  => "bool",
        // Explicit fixed-width fields keep their own exact width in the host-side buffer —
        // WGSL itself can't represent 8/16/64/128-bit ints (see `wgsl_scalar` in device.rs,
        // which emits a clear compile error for those widths); the host Rust type still
        // reflects the declared width so the mismatch is visible on the device side, not
        // silently mis-typed here too (the previous behavior for `Uint8`, which fell to `i64`).
        // `float64` gets the same treatment — it used to silently narrow to `f32` here,
        // masking the very WGSL-has-no-f64 error `wgsl_unsupported_f64` now raises on the
        // device side (docs/float-width-types.md §6).
        Type::Float64 => "f64",
        Type::Uint8 => "u8",
        Type::Int8   => "i8",
        Type::Int16  => "i16",
        Type::Int32  => "i32",
        Type::Int64  => "i64",
        Type::Int128 => "i128",
        Type::Uint16 => "u16",
        Type::Uint32 => "u32",
        Type::Uint64 => "u64",
        Type::Uint128 => "u128",
        Type::Named(n) => match n.as_str() {
            "int" | "i32"   => "i32",
            "uint" | "u32"  => "u32",
            "float32" | "f32" => "f32",
            "float" | "float64" | "f64" => "f64",
            "bool"                  => "bool",
            "uint8"                 => "u8",
            "int8"                  => "i8",
            "int16"                 => "i16",
            "uint16"                => "u16",
            "int32"                 => "i32",
            "uint32"                => "u32",
            "int64" | "i64"         => "i64",
            "uint64" | "u64"        => "u64",
            "int128" | "i128"       => "i128",
            "uint128" | "u128"      => "u128",
            _                       => "i64",
        },
        Type::Qualified(inner, _) => host_scalar_type(inner),
        _ => "i64",
    }
}

fn host_type(ty: &Type) -> String {
    match ty {
        Type::Array(inner)     => format!("Vec<{}>", host_scalar_type(inner)),
        Type::ArrayN(inner, n) => format!("[{}; {}]", host_scalar_type(inner), n),
        Type::Named(n) if n == "Dimension" => "(i32, i32)".into(),
        other => host_scalar_type(other).into(),
    }
}

fn buffer_usages(f: &KernelFieldDecl) -> String {
    match f.qual {
        GpuQual::Unified => "wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST".into(),
        GpuQual::Surface => "wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST".into(),
        GpuQual::Global  => "wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST".into(),
        GpuQual::ActorGlobal => "wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST".into(),
        // Same storage-only usage as 'unified — host access goes through the same
        // staging-buffer copy path (see `emit_read_field`/upload), never a direct
        // MAP_READ/MAP_WRITE on the atomic<T> storage buffer itself. This is what
        // sidesteps the open question of whether WGSL allows atomic<T> storage to
        // also carry MAP_READ/MAP_WRITE — it never needs to.
        GpuQual::ActorUnified => "wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST".into(),
        _ => "wgpu::BufferUsages::STORAGE".into(),
    }
}

/// Emit a host-side expression (simplified -- for init args and dispatch grid).
fn host_expr(e: &Expr) -> String {
    match &e.kind {
        ExprKind::Int(n)        => n.to_string(),
        ExprKind::Float(f)      => format!("{}", f),
        ExprKind::Bool(b)       => b.to_string(),
        ExprKind::Str(s)        => format!("\"{}\"", s),
        ExprKind::Var(n)        => n.clone(),
        ExprKind::BinOp(op, l, r) => {
            let op_s = match op {
                BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*",
                BinOp::Div => "/", _ => "/*op*/",
            };
            format!("({} {} {})", host_expr(l), op_s, host_expr(r))
        }
        ExprKind::Call(callee, args) => {
            let fn_s = if let ExprKind::Var(n) = &callee.kind { n.clone() } else { host_expr(callee) };
            let args_s: Vec<String> = args.iter().map(|a| host_expr(&a.value)).collect();
            format!("{}({})", fn_s, args_s.join(", "))
        }
        _ => "/* expr */".into(),
    }
}

fn resolve_scalar_expr(expr: &str, scalars: &std::collections::HashMap<String, i64>) -> String {
    if let Some(&n) = scalars.get(expr) { return n.to_string(); }
    expr.to_string()
}

fn boring_key_to_rust(s: &str) -> String {
    match s {
        "\x1B" | "\\x1B" | "Escape" | "escape" => "\"Escape\"".into(),
        " " | "Space" | "space"                 => "\" \"".into(),
        "\n" | "\\n" | "Enter" | "enter"        => "\"Enter\"".into(),
        other => format!("\"{}\"", other),
    }
}

/// True when `decl` declares a `Dimension`-typed `'const` field (inferred for a bare
/// `let Dimension name`, per the parser's binding-based qualifier rule). Such a kernel's
/// `Kernel::new(...)` (see `emit_kernel_new`) takes extra flat `width: i32, height: i32`
/// parameters before device/queue -- a `Kernel(Dimension(w, h))` constructor call only
/// gets to pass `w, h` through if the kernel actually declares this field; otherwise
/// `new()` takes no such parameters and passing them anyway is an arity mismatch.
fn kernel_has_dim_field(decl: &KernelDecl) -> bool {
    decl.fields.iter().any(|f| matches!(&f.ty, Type::Named(n) if n == "Dimension") && matches!(f.qual, GpuQual::Const))
}

/// Extract `(w_str, h_str)` from a `Dimension(w, h)` call arg if present.
fn extract_dimension_args(arg: Option<&Arg>) -> Option<(String, String)> {
    let arg = arg?;
    if let ExprKind::Call(dc, da) = &arg.value.kind {
        if let ExprKind::Var(n) = &dc.kind {
            if n == "Dimension" {
                let w = da.first().map(|x| host_expr(&x.value)).unwrap_or_else(|| "800".into());
                let h = da.get(1).map(|x| host_expr(&x.value)).unwrap_or_else(|| "600".into());
                return Some((w, h));
            }
        }
    }
    None
}
