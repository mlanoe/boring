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
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_host_rs(
    program: &Program,
    kernel_names: &[String],
    effective_kernels: &[KernelDecl],
    general_code: &str,
    has_boring_main: bool,
    boring_main_throws: bool,
    has_emulated_shader: bool,
    requires_f16: bool,
    screen_setup_fields: &[(String, String)],
) -> (String, Vec<TranspileError>) {
    let mut e = HostEmitter::new(program, kernel_names, effective_kernels, general_code, has_boring_main, boring_main_throws);
    e.has_emulated_shader = has_emulated_shader;
    e.requires_f16 = requires_f16;
    e.setup_fields = screen_setup_fields.to_vec();
    e.emit();
    let pending = std::mem::take(&mut *e.expr_errors.borrow_mut());
    e.errors.extend(pending);
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
    /// Receiver-type resolution for `.count`/`.length` (see `host_types`).
    tenv: crate::transpiler::host_types::HostTypeEnv,
    has_boring_main: bool,
    /// Whether the emitted `boring_main` actually returns a `Result` — an explicit
    /// user `def main():` only does when it declares `throws`; a synthesized one
    /// (from bare top-level statements, `emit_gpu_boring_main`) always does. Gates
    /// whether `emit_main` calls it as `if let Err(e) = boring_main() { .. }` or as
    /// a plain statement — the two aren't interchangeable, calling a `()`-returning
    /// fn through the `Err` pattern is a real `cargo check` E0308, not just style.
    boring_main_throws: bool,
    /// The `kernel.field_buf` the loop presents (`screen.present(k.pixels)`), as a path on `__App`.
    screen_pixels_buf: String,
    /// Loop variables of `for k in screen.key_events:` in scope (`Arc<str>`s: passed by clone).
    event_vars: std::cell::RefCell<Vec<String>>,
    /// Whether a second, shared-memory-emulated WGSL module was emitted
    /// alongside the real-subgroup one (`shaders/main_emulated.wgsl`) — only
    /// true when some kernel uses `.warp.*` (see `device::WarpMode`). Gates
    /// the `wgpu::Features::SUBGROUP` runtime detection and dual-shader-module
    /// dispatch in `emit_pipeline_init`/wherever the shader module is created.
    has_emulated_shader: bool,
    /// The emitted WGSL uses the optional `f16` extension, so device creation
    /// must require `Features::SHADER_F16` instead of failing later at pipeline creation.
    requires_f16: bool,
    /// `(name, type)` of the values `__boring_screen_setup` returns (`__App` fields).
    setup_fields: Vec<(String, String)>,
    /// Names of the `let`s of the render loop body seen so far (assignable when `var`).
    frame_locals: std::cell::RefCell<Vec<String>>,
    /// Declared/inferred types of those locals, for resolving overloaded free-function calls.
    frame_local_types: std::cell::RefCell<std::collections::HashMap<String, Type>>,
    /// Errors found while checking a `Screen` program's shape (see
    /// `check_screen_program_shape`/`emit_render_stmt`'s error fallbacks).
    /// Non-empty means `out` is incomplete/wrong and must not be written out
    /// — see `emit_host_rs`'s doc comment.
    errors: Vec<TranspileError>,
    /// Errors found by `host_expr` (which only has `&self`); folded into `errors` when emission ends.
    expr_errors: std::cell::RefCell<Vec<TranspileError>>,
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
        Self { out: String::new(), program, kernel_names, effective_kernels, has_screen, in_method: false, general_code, tenv: crate::transpiler::host_types::HostTypeEnv::from_program(program), has_boring_main, boring_main_throws, has_emulated_shader: false, requires_f16: false, setup_fields: Vec::new(), frame_locals: Default::default(), frame_local_types: Default::default(), screen_pixels_buf: String::new(), event_vars: Default::default(), errors: Vec::new(), expr_errors: Default::default() }
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
            // Every other top-level `let`/statement is ordinary code of `__boring_screen_setup`
            // (general pass); only the `kernel:` block has a shape of its own.
            if let Item::Stmt(stmt @ Stmt::KernelBlock(_)) = item {
                self.check_screen_top_level_stmt(stmt);
            }
        }
    }

    fn kernel_name_set(&self) -> std::collections::HashSet<String> {
        self.kernel_names.iter().cloned().collect()
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
        self.line("#[allow(dead_code)] fn __boring_last_choice(_kernel: &str) -> std::sync::Arc<str> { \"wgpu\".into() }");
        self.out.push_str(&crate::transpiler::unified::single_backends_fn("wgpu", "gpu", "wgpu::Instance::new(&wgpu::InstanceDescriptor::default()).enumerate_adapters(wgpu::Backends::all()).first().map_or(\"gpu\", |a| match a.get_info().device_type { wgpu::DeviceType::DiscreteGpu => \"discrete\", wgpu::DeviceType::IntegratedGpu => \"integrated\", _ => \"gpu\" })"));
        self.blank();
    }

    fn emit_header(&mut self) {
        self.line("// Generated by boring build --target wgpu.");
        self.line("// Do not edit -- re-generate with: boring build --target wgpu");
        self.line(crate::transpiler::helpers::GENERATED_HOST_LINT_ALLOW);
        self.blank();
        // Note: no hardcoded `use std::collections::HashMap;` here -- the general
        // transpiler's own output (spliced in by `emit()`, see `general_code`) already
        // emits it when the merged program actually uses dicts, and a second identical
        // `use` at the same scope is a duplicate-name error (E0252).
        self.line("use wgpu::util::DeviceExt;");
        self.line("use bytemuck::{Pod, Zeroable};");
        self.line("fn set_gpu_batch_limit(_limit: i64) {}");
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
                    ty if const_array_view(ty).is_some() => {
                        let (inner, n) = const_array_view(ty).unwrap();
                        let inner = &inner;
                        if crate::transpiler::emit_kernel::fixed_kernel_fmt(inner).is_some_and(|x| x.width() > 32) {
                            self.line(&format!("    {}: [u32; {}],", f.name, 2 * n));
                        } else {
                            self.line(&format!("    {}: [{}; {}],", f.name, params_slot_type(inner), n));
                        }
                    }
                    ty if crate::transpiler::emit_kernel::fixed_kernel_fmt(ty).is_some_and(|x| x.width() > 32) => {
                        self.line(&format!("    {}_lo: u32,", f.name));
                        self.line(&format!("    {}_hi: u32,", f.name));
                    }
                    ty => {
                        self.line(&format!("    {}: {},", f.name, params_slot_type(ty)));
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
                        // A packed-byte-kind field's buffer is padded up to a whole number
                        // of 4-byte words (`round_up_to_word_bytes`), so its wgpu buffer size
                        // no longer determines the real element count -- track it separately
                        // so `copy_{field}_to_host` can truncate the padding back off. See
                        // `emit_kernel_copy_accessors`.
                        if is_packed_byte_field(&array_inner(&f.ty)) {
                            self.line(&format!("    {}_len: usize,", f.name));
                        }
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
                        ty if const_array_view(ty).is_some() => {
                            let (inner, n) = const_array_view(ty).unwrap();
                            self.line(&format!("    {}: [{}; {}],", f.name, host_scalar_type(&inner), n));
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

        // Create GPU buffers. `packed_len_exprs` records each packed-byte-kind field's
        // real (unpadded) initial element count, in the same match arms below, so the
        // "Return struct" section further down can initialize that field's `_len`
        // without recomputing which arm applied.
        let mut packed_len_exprs: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        for f in &buf_fields {
            let usages = buffer_usages(f);
            let inner_ty = array_inner(&f.ty);
            let host_ty = host_scalar_type(&inner_ty);
            let packed = is_packed_byte_field(&inner_ty);
            let (size_expr, init_data) = match &f.ty {
                Type::ArrayN(_, n) => {
                    let bytes = format!("{} * std::mem::size_of::<{}>()", n, host_ty);
                    if packed { packed_len_exprs.insert(f.name.clone(), n.to_string()); }
                    (if packed { round_up_to_word_bytes(&bytes) } else { format!("({}) as u64", bytes) },
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
                    let bytes = format!("{} * std::mem::size_of::<{}>()", n, host_ty);
                    if packed { packed_len_exprs.insert(f.name.clone(), n.to_string()); }
                    (if packed { round_up_to_word_bytes(&bytes) } else { format!("({}) as u64", bytes) },
                     format!("vec![{}::default(); {}]", host_ty, n))
                }
                _ if has_dim_field => {
                    // Dynamic size derived from Dimension (width * height * sizeof<T>).
                    let bytes = format!("(width * height) as usize * std::mem::size_of::<{}>()", host_ty);
                    if packed { packed_len_exprs.insert(f.name.clone(), "(width * height) as usize".to_string()); }
                    (if packed { round_up_to_word_bytes(&bytes) } else { format!("({}) as u64", bytes) },
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
                    if packed { packed_len_exprs.insert(f.name.clone(), "0".to_string()); }
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

        // Params buffer. See `super::kernel_params_use_storage`'s doc comment: a
        // fixed-size scalar array field forces the matching WGSL binding
        // (`device::emit_params_struct`) into `var<storage, read>` instead of
        // `var<uniform>` — the buffer's own usage flags must agree, since a
        // buffer created with only `UNIFORM` usage can't satisfy a storage
        // binding at bind-group-creation time.
        if !params_fields.is_empty() {
            let usage = if super::kernel_params_use_storage(decl) {
                "wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST"
            } else {
                "wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST"
            };
            self.line("        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {");
            self.line("            label: None,");
            self.line(&format!("            size: std::mem::size_of::<{}Params>() as u64,", name));
            self.line(&format!("            usage: {},", usage));
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
        // Explicit layout holding every binding the host binds: the automatic one drops a
        // binding the entry point never touches, and `new()`/`rebuild_bind_group` bind all
        // of the kernel's buffer fields (an unused one used to fail bind group creation).
        self.line("            let layout_entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {");
        self.line("                binding,");
        self.line("                visibility: wgpu::ShaderStages::COMPUTE,");
        self.line("                ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },");
        self.line("                count: None,");
        self.line("            };");
        self.line("            let layout_entries = [");
        let mut layout_idx: u32 = 0;
        for f in &buf_fields {
            let read_only = super::device::wgsl_buffer_type(f).0 == "read";
            self.line(&format!("                layout_entry({layout_idx}, wgpu::BufferBindingType::Storage {{ read_only: {read_only} }}),"));
            layout_idx += 1;
        }
        if !params_fields.is_empty() {
            let ty = if super::kernel_params_use_storage(decl) { "wgpu::BufferBindingType::Storage { read_only: true }" } else { "wgpu::BufferBindingType::Uniform" };
            self.line(&format!("                layout_entry({layout_idx}, {ty}),"));
        }
        self.line("            ];");
        self.line("            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &layout_entries });");
        self.line("            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[] });");
        self.line("            std::sync::Arc::new(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {");
        self.line("                label: None,");
        self.line("                layout: Some(&pipeline_layout),");
        self.line("                module: shader,");
        self.line(&format!("                entry_point: Some(\"{}_main\"),", name));
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
            if let Some(len_expr) = packed_len_exprs.get(&f.name) {
                self.line(&format!("            {}_len: {},", f.name, len_expr));
            }
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
                        ty if const_array_view(ty).is_some() => {
                            let (inner, n) = const_array_view(ty).unwrap();
                            self.line(&format!("            {}: [{}::default(); {}],",
                                f.name, host_scalar_type(&inner), n));
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
        // second (inner) -- so a KernelError::OutOfMemory and a
        // KernelError::LaunchError can be told apart instead of collapsing into
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
                    ty if const_array_view(ty).is_some() => {
                        let (inner, n) = const_array_view(ty).unwrap();
                        let inner = &inner;
                        if crate::transpiler::emit_kernel::fixed_kernel_fmt(inner).is_some_and(|x| x.width() > 32) {
                            self.line(&format!(
                                "            {}: {{ let mut __a = [0u32; {}]; for (__i, __x) in self.{}.iter().enumerate() {{ __a[2 * __i] = (__x.0 as u64) as u32; __a[2 * __i + 1] = ((__x.0 as u64) >> 32) as u32; }} __a }},",
                                f.name, 2 * n, f.name));
                            continue;
                        }
                        let conv = params_slot_conv(inner, "__x");
                        if conv == "__x" {
                            self.line(&format!("            {}: self.{},", f.name, f.name));
                        } else {
                            self.line(&format!("            {}: self.{}.map(|__x| {}),", f.name, f.name, conv));
                        }
                    }
                    ty if crate::transpiler::emit_kernel::fixed_kernel_fmt(ty).is_some_and(|x| x.width() > 32) => {
                        self.line(&format!("            {}_lo: (self.{}.0 as u64) as u32,", f.name, f.name));
                        self.line(&format!("            {}_hi: ((self.{}.0 as u64) >> 32) as u32,", f.name, f.name));
                    }
                    ty => {
                        self.line(&format!("            {}: {},", f.name, params_slot_conv(ty, &format!("self.{}", f.name))));
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
        // popped error is classified into a typed `KernelError` and wrapped in
        // `BoringError::Other` so `catch KernelError.OutOfMemory:` / `catch
        // KernelError.LaunchError:` can dispatch on it in Boring source, instead
        // of the single opaque formatted-string error this used to return.
        self.line("        if pollster::block_on(self.device.pop_error_scope()).is_some() {");
        self.line("            let _ = pollster::block_on(self.device.pop_error_scope());");
        self.line("            return Err(Box::new(BoringError::Other(std::any::TypeId::of::<KernelError>(), Box::new(KernelError::LaunchError) as Box<dyn BoringVal + Send + Sync>)));");
        self.line("        }");
        self.line("        if pollster::block_on(self.device.pop_error_scope()).is_some() {");
        self.line("            return Err(Box::new(BoringError::Other(std::any::TypeId::of::<KernelError>(), Box::new(KernelError::OutOfMemory) as Box<dyn BoringVal + Send + Sync>)));");
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
            // A `'const` fixed-array field lives in the params uniform (a host `[T; N]`
            // mirror written at every dispatch), not in a storage buffer. Init-literal
            // fields (`w = [1.0, 2.0]`) reach it through the same `copy_{field}_to_device`
            // call `emit_kernel::kernel_output_fill_map` emits for every backend.
            if matches!(f.qual, GpuQual::Const) {
                if let Some((inner, n)) = const_array_view(&f.ty) {
                    let host_ty = host_scalar_type(&inner);
                    self.line(&format!("    fn copy_{}_to_device(&mut self, data: &[{}]) {{", f.name, host_ty));
                    self.line(&format!("        self.{}[..data.len().min({})].copy_from_slice(&data[..data.len().min({})]);", f.name, n, n));
                    self.line("    }");
                }
                continue;
            }
            // 'global fields are host-supplied inputs; 'unified fields are typically
            // kernel-written outputs (e.g. StftPower's `power`) but are just as much a
            // Vec<T> mirror on the host struct (see the field-emission loop above) and
            // need the same read-back/upload accessors to actually be usable from host code.
            // 'actor'unified needs the same accessors for the same reason (host-visible),
            // plus `kernel_output_fill_map`'s unconditional `copy_{field}_to_device` call
            // for any `field = [value for ..<count]` init — see `emit_kernel.rs`.
            if matches!(f.qual, GpuQual::Global | GpuQual::Unified | GpuQual::ActorUnified | GpuQual::Surface) && is_buffer_array_ty(&f.ty) {
                let inner_ty = array_inner(&f.ty);
                let host_ty = host_scalar_type(&inner_ty);
                let packed = is_packed_byte_field(&inner_ty);
                // D2H.
                self.line(&format!("    fn copy_{}_to_host(&self) -> Vec<{}> {{", f.name, host_ty));
                if packed {
                    // The buffer itself is padded up to a whole 4-byte word
                    // (`copy_{field}_to_device` below, `round_up_to_word_bytes`), so
                    // `__boring_gpu_copy_d2h` reads back up to 3 extra trailing zero
                    // elements — truncate to the real length tracked in `{field}_len`.
                    self.line(&format!("        let mut __v = __boring_gpu_copy_d2h::<{}>(&self.device, &self.queue, &self.{}_buf);", host_ty, f.name));
                    self.line(&format!("        __v.truncate(self.{}_len);", f.name));
                    self.line("        __v");
                } else {
                    self.line(&format!("        __boring_gpu_copy_d2h::<{}>(&self.device, &self.queue, &self.{}_buf)", host_ty, f.name));
                }
                self.line("    }");
                // Allocate or resize a device output without staging and uploading a
                // host-side zero vector. ArrayAlloc (`field = [..<count]`) promises
                // that the kernel overwrites the storage before it is observed.
                self.line(&format!("    fn resize_{}(&mut self, len: usize) {{", f.name));
                if packed {
                    self.line(&format!("        let needed = {};", round_up_to_word_bytes(&format!("len * std::mem::size_of::<{}>()", host_ty))));
                } else {
                    self.line(&format!("        let needed = (len * std::mem::size_of::<{}>()) as u64;", host_ty));
                }
                self.line("        let needed = needed.max(4);");
                self.line(&format!("        if self.{}_buf.size() != needed {{", f.name));
                self.line(&format!("            self.{}_buf = std::sync::Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {{", f.name));
                self.line("                label: None,");
                self.line("                size: needed,");
                self.line(&format!("                usage: {},", buffer_usages(f)));
                self.line("                mapped_at_creation: false,");
                self.line("            }));");
                self.line("            self.rebuild_bind_group();");
                self.line("        }");
                if packed {
                    self.line(&format!("        self.{}_len = len;", f.name));
                }
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
                if packed {
                    self.line(&format!("        let needed = {};", round_up_to_word_bytes(&format!("data.len() * std::mem::size_of::<{}>()", host_ty))));
                } else {
                    self.line(&format!("        let needed = (data.len() * std::mem::size_of::<{}>()) as u64;", host_ty));
                }
                self.line(&format!("        if self.{}_buf.size() != needed {{", f.name));
                self.line(&format!("            self.{}_buf = std::sync::Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {{", f.name));
                self.line("                label: None,");
                self.line("                size: needed,");
                self.line(&format!("                usage: {},", buffer_usages(f)));
                self.line("                mapped_at_creation: false,");
                self.line("            }));");
                self.line("            self.rebuild_bind_group();");
                self.line("        }");
                if packed {
                    // `__boring_gpu_copy_h2d`'s internal `copy_buffer_to_buffer` size is
                    // `src.len()`, which wgpu requires to be a multiple of
                    // `COPY_BUFFER_ALIGNMENT` (4 bytes) — pad the raw byte view up to
                    // `needed` (zero-filled tail) whenever `data`'s own byte length isn't
                    // already word-aligned, instead of passing the unpadded slice through.
                    self.line("        let __bytes: &[u8] = bytemuck::cast_slice(data);");
                    self.line("        let __padded: std::borrow::Cow<[u8]> = if (__bytes.len() as u64) == needed {");
                    self.line("            std::borrow::Cow::Borrowed(__bytes)");
                    self.line("        } else {");
                    self.line("            let mut __v = __bytes.to_vec();");
                    self.line("            __v.resize(needed as usize, 0);");
                    self.line("            std::borrow::Cow::Owned(__v)");
                    self.line("        };");
                    self.line(&format!("        __boring_gpu_copy_h2d(&self.device, &self.queue, &__padded, &self.{}_buf);", f.name));
                    self.line(&format!("        self.{}_len = data.len();", f.name));
                } else {
                    self.line(&format!("        __boring_gpu_copy_h2d(&self.device, &self.queue, bytemuck::cast_slice(data), &self.{}_buf);", f.name));
                }
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
        self.line("pub enum BoringKernelArg<T> {");
        self.line("    Resident(std::sync::Arc<wgpu::Buffer>, usize),");
        self.line("    Host(Vec<T>),");
        self.line("}");
        self.blank();
        self.line("impl<T: std::fmt::Debug> std::fmt::Debug for BoringKernelArg<T> {");
        self.line("    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { match self { BoringKernelArg::Resident(_, n) => f.debug_tuple(\"Resident\").field(n).finish(), BoringKernelArg::Host(v) => f.debug_tuple(\"Host\").field(v).finish() } }");
        self.line("}");
        self.line("impl<T: PartialEq> PartialEq for BoringKernelArg<T> {");
        self.line("    fn eq(&self, other: &Self) -> bool { match (self, other) { (BoringKernelArg::Resident(a, an), BoringKernelArg::Resident(b, bn)) => std::sync::Arc::ptr_eq(a, b) && an == bn, (BoringKernelArg::Host(a), BoringKernelArg::Host(b)) => a == b, _ => false } }");
        self.line("}");
        self.blank();
        // `<param>.length`/`.count` (mapped to `.len()` by the general `map_field`
        // convention, same as a plain array) needs an inherent method here since a
        // dual-typed param doesn't materialize just to answer a size query -- a
        // `Resident` buffer already carries its element count.
        self.line("#[allow(dead_code)]");
        self.line("impl<T> BoringKernelArg<T> {");
        self.line("    fn len(&self) -> usize {");
        self.line("        match self {");
        self.line("            BoringKernelArg::Resident(_, len) => *len,");
        self.line("            BoringKernelArg::Host(v) => v.len(),");
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
        self.line("    let _ = device.poll(wgpu::MaintainBase::Wait);");
        self.line("    let slice = staging.slice(..);");
        self.line("    let (tx, rx) = std::sync::mpsc::channel();");
        self.line("    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });");
        self.line("    let _ = device.poll(wgpu::MaintainBase::Wait);");
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
        self.line("    let staging_size = size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);");
        self.line("    let staging = device.create_buffer(&wgpu::BufferDescriptor {");
        self.line("        label: None,");
        self.line("        size: staging_size,");
        self.line("        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,");
        self.line("        mapped_at_creation: true,");
        self.line("    });");
        self.line("    {");
        self.line("        let mut mapped = staging.slice(..).get_mapped_range_mut();");
        self.line("        mapped[..src.len()].copy_from_slice(src);");
        self.line("    }");
        self.line("    staging.unmap();");
        self.line("    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });");
        self.line("    encoder.copy_buffer_to_buffer(&staging, 0, dst, 0, staging_size);");
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
        // No explicit `screen.present(k.field)` call in the loop (a `'surface`
        // field presents implicitly) -- fall back to scanning top-level kernel
        // instantiations for a `'surface`-qualified buffer-array field, and
        // qualify it with the kernel instance's own variable name. Without this,
        // the caller's bare `"pixels_buf"` default gets emitted as
        // `self.pixels_buf` in `__App` (which has no such field -- the kernel
        // instance is `self.{var}`, a field *on* `__App`, not `__App` itself) --
        // see `emit_kernel_rebuild_bind_group`'s `self.{var}.{field}_buf` for the
        // pattern this must match.
        for item in &self.program.items {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    if let ExprKind::Call(callee, _) = &val.kind {
                        if let ExprKind::Var(kname) = &callee.kind {
                            if let Some(decl) = self.effective_kernels.iter().find(|d| &d.name == kname) {
                                if let Some(field) = decl.fields.iter().find(|f| {
                                    matches!(f.qual, GpuQual::Surface) && is_buffer_array_ty(&f.ty)
                                }) {
                                    return format!("{}.{}_buf", s.name, field.name);
                                }
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
            self.line("    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());");
            self.line("    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.expect(\"No GPU adapter found\");");
            self.line("    let mut __boring_required_features = wgpu::Features::empty();");
            if self.has_emulated_shader {
                // `.warp.*` is used somewhere in this program — opportunistically
                // request `Features::SUBGROUP` (only if the adapter actually supports
                // it; `request_device` hard-fails if you require an unsupported
                // feature) so `device.features()` at shader-creation time can pick
                // the real-subgroup module over the shared-memory-emulated one.
                self.line("    if adapter.features().contains(wgpu::Features::SUBGROUP) { __boring_required_features |= wgpu::Features::SUBGROUP; }");
            }
            if self.requires_f16 {
                self.line("    assert!(adapter.features().contains(wgpu::Features::SHADER_F16), \"This GPU adapter does not support the shader-f16 feature required by this program\");");
                self.line("    __boring_required_features |= wgpu::Features::SHADER_F16;");
            }
            self.line("    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {");
            self.line("        required_features: __boring_required_features,");
            self.line("        required_limits: adapter.limits(),");
            self.line("        ..Default::default()");
            self.line("    }).await.expect(\"Failed to create device\");");
            // Defensive: make sure the device is fully settled before any real
            // dispatch work begins. There's nothing queued yet, so this returns
            // essentially immediately -- it's here only to rule out adapter/device
            // warm-up timing as a source of flaky early-kernel behavior.
            self.line("    let _ = device.poll(wgpu::MaintainBase::Wait);");
            // Without this, a validation error NOT already inside an explicit
            // push/pop_error_scope pair (e.g. pipeline creation at kernel `new()`
            // time — see `emit_kernel_new`, no error scope wraps it there) hits
            // wgpu's default uncaptured-error handler, which panics. Installed
            // once here so such an error is reported and the program can
            // continue instead of crashing outright — a later dispatch() call on
            // a pipeline that failed to create will raise its own validation
            // error, which dispatch's own push/pop_error_scope DOES already
            // catch and convert to a proper KernelError::LaunchError.
            self.line("    device.on_uncaptured_error(Box::new(|e| eprintln!(\"boring: uncaptured GPU error: {}\", e)));");
            self.line("    let device = std::sync::Arc::new(device);");
            self.line("    let queue = std::sync::Arc::new(queue);");
            self.line("    let _ = __BORING_GPU_DEVICE.set(std::sync::Arc::clone(&device));");
            self.line("    let _ = __BORING_GPU_QUEUE.set(std::sync::Arc::clone(&queue));");
            self.line("    let adapter = std::sync::Arc::new(adapter);");
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
        let on_demand = crate::transpiler::helpers::screen_redraw_on_demand(self.program);
        let (w_expr, h_expr, title, _) = self.extract_screen_info();
        let hidpi = self.screen_hidpi();
        let scalars     = self.extract_top_level_scalars();
        let width       = resolve_scalar_expr(&w_expr, &scalars);
        let height      = resolve_scalar_expr(&h_expr, &scalars);
        let loop_body   = self.find_render_loop_body();
        let pixels_buf  = self.find_present_buffer();
        self.screen_pixels_buf = pixels_buf.clone();
        // `screen.setAccessibility(..)` / `screen.accessibilityRequests` in the loop: wire AccessKit in
        // (a program without them gets none of it: no extra dependency, no user events, no adapter).
        let uses_ax = {
            let dbg = format!("{loop_body:?}");
            dbg.contains("setAccessibility") || dbg.contains("accessibilityRequests")
        };

        let items: Vec<Item> = self.program.items.clone();

        // Collect scalar names and kernel instantiations for the app struct.
        let loop_vars = crate::transpiler::helpers::render_loop_var_names(self.program);
        let scalar_fields: Vec<(String, String)> = items.iter().filter_map(|item| {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    let is_screen = matches!(&val.kind, ExprKind::Call(c, _) if matches!(&c.kind, ExprKind::Var(n) if n == "Screen"));
                    let is_kernel = matches!(&val.kind, ExprKind::Call(c, _) if matches!(&c.kind, ExprKind::Var(kn) if self.kernel_names.contains(kn)));
                    // Only the constants the window size or the render loop read: every other
                    // scalar `let` (including imported modules' constants) is a module-level `const`.
                    let needed = loop_vars.contains(&s.name)
                        || w_expr.split(|c: char| !c.is_alphanumeric() && c != '_').any(|t| t == s.name)
                        || h_expr.split(|c: char| !c.is_alphanumeric() && c != '_').any(|t| t == s.name);
                    if needed && !is_screen && !is_kernel && !s.binding.is_mutable() && matches!(val.kind, ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_)) {
                        return Some((s.name.clone(), host_expr(val)));
                    }
                }
            }
            None
        }).collect();

        let kernel_set = self.kernel_name_set();
        let kernel_fields: Vec<(String, String, String)> = items.iter().filter_map(|item| {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    if let ExprKind::Call(callee, args) = &val.kind {
                        if let ExprKind::Var(kname) = &callee.kind {
                            if self.kernel_names.contains(kname)
                                && crate::transpiler::helpers::classify_screen_let(s, &kernel_set)
                                    == crate::transpiler::helpers::ScreenLet::HostKernel
                            {
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

        // Shared input runtime (typed text, IME, modifiers, wheel, focus, key edges) + key naming.
        for chunk in [crate::transpiler::screen_input::INPUT_RUNTIME, crate::transpiler::screen_input::KEY_NAMES_WINIT_030] {
            for l in chunk.lines() { self.line(l); }
        }
        if uses_ax {
            for l in crate::transpiler::screen_input::AX_RUNTIME.lines() { self.line(l); }
            self.line(&format!("const __AX_TITLE: &str = \"{title}\";"));
        }
        self.blank();
        // ── __App struct ──────────────────────────────────────────────────────
        self.line("struct __App {");
        self.line("    instance: wgpu::Instance,");
        // Arc-wrapped, not a bare `wgpu::Adapter` (which has no `Clone` of its own).
        self.line("    adapter:  std::sync::Arc<wgpu::Adapter>,");
        self.line("    device:   std::sync::Arc<wgpu::Device>,");
        self.line("    queue:    std::sync::Arc<wgpu::Queue>,");
        for (name, _val) in &scalar_fields {
            self.line(&format!("    {name}: i64,"));
        }
        for (var, kname, _) in &kernel_fields {
            self.line(&format!("    {var}: {kname},"));
        }
        for (name, ty) in &self.setup_fields.clone() {
            self.line(&format!("    {name}: {ty},"));
        }
        self.line("    __keys: std::collections::HashSet<String>,");
        // Edge-triggered `screen.key_pressed()` set (cleared after each frame body) and
        // `screen.mouse` state in pixel-buffer coordinates.
        self.line("    __keys_pressed: std::collections::HashSet<String>,");
        self.line("    __mouse_x: isize,");
        self.line("    __mouse_y: isize,");
        self.line("    __mouse_left: bool,");
        self.line("    __mouse_right: bool,");
        self.line("    __mouse_middle: bool,");
        // Shared input state, the live window size (`screen.width`/`height`/`resized`) and the
        // size the pixel buffer was allocated for (mouse coordinates are buffer coordinates).
        self.line("    __input: __BoringInput,");
        self.line("    __win_w: isize,");
        self.line("    __win_h: isize,");
        self.line("    __resized: bool,");
        self.line("    __ime_on: bool,");
        self.line("    __safe: [isize; 4],");
        self.line("    __scale: f32,");
        self.line("    __buf_w: f64,");
        self.line("    __buf_h: f64,");
        self.line("    __surface_cfg: Option<wgpu::SurfaceConfiguration>,");
        self.line("    __start_time: std::time::Instant,");
        self.line("    __frame: isize,");
        self.line("    __redraw: bool,");
        self.line("    __wake_at: Option<std::time::Instant>,");
        if uses_ax {
            self.line("    __ax: Option<accesskit_winit::Adapter>,");
            self.line("    __ax_proxy: Option<winit::event_loop::EventLoopProxy<__AxEvent>>,");
            self.line("    __ax_active: bool,");
            self.line("    __ax_last: Option<(Vec<__AxNode>, f64)>,");
            self.line("    __ax_requests: Vec<__AccessRequest>,");
        }
        self.line("    window: Option<std::sync::Arc<Window>>,");
        self.line("    surface: Option<wgpu::Surface<'static>>,");
        self.line("    __blit_pipeline: Option<wgpu::RenderPipeline>,");
        self.line("    __blit_bg:       Option<wgpu::BindGroup>,");
        self.line("    __blit_dim_buf:  Option<wgpu::Buffer>,");
        self.line("}");
        self.blank();

        // After the kernel was reallocated for the window's current size (`k = K(...)` in the loop):
        // resize the swapchain to that size and point the blit at the new pixel buffer.
        self.line("impl __App {");
        if uses_ax {
            // Send the tree to AccessKit when it changed (its own diff does the rest); only called while
            // an assistive technology is connected.
            self.line("    fn __ax_push(&mut self, nodes: Vec<__AxNode>) {");
            self.line("        let scale = self.__scale as f64;");
            self.line("        if self.__ax_last.as_ref().is_some_and(|(n, s)| *n == nodes && *s == scale) { return; }");
            self.line("        if let Some(ax) = self.__ax.as_mut() { let tu = __boring_ax_tree(__AX_TITLE, &nodes, scale); ax.update_if_active(|| tu); }");
            self.line("        self.__ax_last = Some((nodes, scale));");
            self.line("    }");
        }
        // The safe area's insets in physical pixels, [top, left, bottom, right]: on iOS winit reports the safe area as the
        // window's inner rectangle (`inner_position` / `inner_size`) inside the whole screen (`outer_size`); other targets have none.
        self.line("    fn __update_safe(&mut self) {");
        self.line("        #[cfg(target_os = \"ios\")]");
        self.line("        if let Some(w) = &self.window {");
        self.line("            if let Ok(p) = w.inner_position() {");
        self.line("                let (is, os) = (w.inner_size(), w.outer_size());");
        self.line("                let (l, t) = (p.x as isize, p.y as isize);");
        self.line("                self.__safe = [t, l, (os.height as isize - is.height as isize - t).max(0), (os.width as isize - is.width as isize - l).max(0)];");
        self.line("            }");
        self.line("        }");
        self.line("    }");
        self.line("    fn __reblit(&mut self) {");
        self.line("        let (w, h) = (self.__win_w.max(1) as u32, self.__win_h.max(1) as u32);");
        self.line("        self.__buf_w = w as f64;");
        self.line("        self.__buf_h = h as f64;");
        self.line("        if let (Some(cfg), Some(surface)) = (self.__surface_cfg.as_mut(), self.surface.as_ref()) {");
        self.line("            cfg.width = w; cfg.height = h;");
        self.line("            surface.configure(&self.device, cfg);");
        self.line("        }");
        self.line("        let Some(pipeline) = self.__blit_pipeline.as_ref() else { return };");
        self.line("        let layout = pipeline.get_bind_group_layout(0);");
        self.line("        let dim_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {");
        self.line("            label: Some(\"blit_dim\"), contents: bytemuck::cast_slice(&[w, h]), usage: wgpu::BufferUsages::UNIFORM,");
        self.line("        });");
        self.line("        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {");
        self.line("            label: None, layout: &layout, entries: &[");
        self.line(&format!("                wgpu::BindGroupEntry {{ binding: 0, resource: self.{pixels_buf}.as_entire_binding() }},"));
        self.line("                wgpu::BindGroupEntry { binding: 1, resource: dim_buf.as_entire_binding() },");
        self.line("            ],");
        self.line("        });");
        self.line("        self.__blit_bg = Some(bg);");
        self.line("        self.__blit_dim_buf = Some(dim_buf);");
        self.line("    }");
        self.line("}");
        self.blank();
        // `screen.wake()`: the window, reachable from any thread, so a timer/watcher/network thread can wake a demand-mode loop.
        self.line("#[allow(dead_code)]");
        self.line("static __BORING_WAKE_WINDOW: std::sync::Mutex<Option<std::sync::Arc<Window>>> = std::sync::Mutex::new(None);");
        self.line("#[allow(dead_code)]");
        self.line("fn __boring_wake() { if let Ok(g) = __BORING_WAKE_WINDOW.lock() { if let Some(w) = g.as_ref() { w.request_redraw(); } } }");
        self.blank();
        // iOS: the UIScene life cycle (a no-op everywhere else).
        for l in IOS_SCENE_RS.lines() { self.line(l); }
        self.blank();
        // ── ApplicationHandler impl ───────────────────────────────────────────
        self.line(if uses_ax { "impl ApplicationHandler<__AxEvent> for __App {" } else { "impl ApplicationHandler for __App {" });
        // resumed(): create window + surface + blit pipeline
        self.line("    fn resumed(&mut self, event_loop: &ActiveEventLoop) {");
        // `hidpi = true`: the Dimension is logical points; otherwise it is physical pixels.
        let size_ty = if hidpi { "LogicalSize" } else { "PhysicalSize" };
        // AccessKit's adapter must exist before the window is first shown.
        let visible = if uses_ax { ".with_visible(false)" } else { "" };
        // A phone's window is always the whole screen: a requested size would only make winit build a
        // window of that size on top of it (wider than the screen, content cut off), so mobile targets skip it.
        self.line(&format!(
            "        let __attrs = WindowAttributes::default().with_title(\"{title}\"){visible};"
        ));
        self.line("        #[cfg(not(any(target_os = \"ios\", target_os = \"android\")))]");
        self.line(&format!(
            "        let __attrs = __attrs.with_inner_size(winit::dpi::{size_ty}::new({width}u32, {height}u32));"
        ));
        self.line("        let window = std::sync::Arc::new(event_loop.create_window(__attrs).unwrap());");
        self.line("        #[cfg(target_os = \"ios\")]");
        self.line("        __boring_scene::window_created(&window);");
        if uses_ax {
            self.line("        if let Some(p) = self.__ax_proxy.clone() { self.__ax = Some(accesskit_winit::Adapter::with_event_loop_proxy(event_loop, &window, p)); }");
            self.line("        window.set_visible(true);");
        }
        self.line("        self.__scale = window.scale_factor() as f32;");
        if hidpi {
            // Window and swapchain are in physical pixels; the app reallocates its buffer on `screen.resized`.
            self.line("        let __ps = window.inner_size();");
            self.line("        self.__win_w = __ps.width as isize;");
            self.line("        self.__win_h = __ps.height as isize;");
            self.line("        self.__resized = true;");
        }
        self.line("        let surface: wgpu::Surface<'static> = self.instance.create_surface(window.clone()).unwrap();");
        self.line("        let caps = surface.get_capabilities(&self.adapter);");
        self.line("        let fmt  = caps.formats.iter().find(|f| **f == wgpu::TextureFormat::Bgra8Unorm).copied().unwrap_or(caps.formats[0]);");
        // On a phone `set_ime_allowed(true)` raises the soft keyboard, so it starts off there and follows
        // `screen.setImeAllowed(...)`.
        self.line("        window.set_ime_allowed(cfg!(not(any(target_os = \"ios\", target_os = \"android\"))));");
        self.line("        let cfg = wgpu::SurfaceConfiguration {");
        self.line("            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt,");
        if hidpi {
            self.line("            width: __ps.width.max(1), height: __ps.height.max(1),");
        } else {
            self.line(&format!("            width: {width}, height: {height},"));
        }
        self.line("            present_mode: wgpu::PresentMode::Fifo,");
        self.line("            desired_maximum_frame_latency: 2,");
        self.line("            alpha_mode: caps.alpha_modes[0], view_formats: vec![],");
        self.line("        };");
        self.line("        surface.configure(&self.device, &cfg);");
        self.line("        self.__surface_cfg = Some(cfg);");
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
        // Pixels are packed 0xAARRGGBB (docs/gpu-display.md); the fragment output is logical RGBA,
        // the BGRA8Unorm surface does the byte swizzle itself.
        self.line("    let raw = blit_pixels[i];");
        self.line("    let r = f32((raw >> 16u) & 0xFFu) / 255.0;");
        self.line("    let g = f32((raw >>  8u) & 0xFFu) / 255.0;");
        self.line("    let b = f32((raw >>  0u) & 0xFFu) / 255.0;");
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
        self.line("            vertex: wgpu::VertexState { module: &blit_shader, entry_point: Some(\"blit_vs\"), buffers: &[],");
        self.line("                compilation_options: wgpu::PipelineCompilationOptions::default() },");
        self.line("            fragment: Some(wgpu::FragmentState { module: &blit_shader, entry_point: Some(\"blit_fs\"),");
        self.line("                targets: &[Some(wgpu::ColorTargetState { format: fmt, blend: None,");
        self.line("                    write_mask: wgpu::ColorWrites::ALL })],");
        self.line("                compilation_options: wgpu::PipelineCompilationOptions::default() }),");
        self.line("            primitive: wgpu::PrimitiveState::default(), depth_stencil: None,");
        self.line("            multisample: wgpu::MultisampleState::default(), multiview: None, cache: None,");
        self.line("        });");
        self.line("        self.window = Some(window.clone());");
        self.line("        if let Ok(mut g) = __BORING_WAKE_WINDOW.lock() { *g = Some(window.clone()); }");
        self.line("        self.surface = Some(surface);");
        self.line("        self.__blit_pipeline = Some(blit_pipeline);");
        self.line("        self.__blit_bg = Some(blit_bg);");
        self.line("        self.__blit_dim_buf = Some(blit_dim_buf);");
        if on_demand {
            self.line("        event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);");
        }
        self.line("        window.request_redraw();");
        self.line("    }");
        self.blank();
        // window_event(): handle all window events
        if uses_ax {
            // An assistive technology asked for the tree / for an action: wake the loop (also `redraw = "demand"`).
            self.line("    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: __AxEvent) {");
            self.line("        match event.0.window_event {");
            self.line("            accesskit_winit::WindowEvent::InitialTreeRequested => {");
            self.line("                self.__ax_active = true;");
            self.line("                self.__ax_last = None;");
            self.line("                let scale = self.__scale as f64;");
            self.line("                if let Some(ax) = self.__ax.as_mut() { let tu = __boring_ax_tree(__AX_TITLE, &[], scale); ax.update_if_active(|| tu); }");
            self.line("            }");
            self.line("            accesskit_winit::WindowEvent::ActionRequested(r) => { if let Some(q) = __boring_ax_request(&r) { self.__ax_requests.push(q); } }");
            self.line("            accesskit_winit::WindowEvent::AccessibilityDeactivated => { self.__ax_active = false; self.__ax_last = None; return; }");
            self.line("        }");
            self.line("        if let Some(w) = &self.window { w.request_redraw(); }");
            self.line("    }");
            self.blank();
        }
        if on_demand {
            // `screen.requestRedrawAfter(ms)`: the loop sleeps in `ControlFlow::WaitUntil(deadline)`; reaching it wakes a frame.
            self.line("    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: winit::event::StartCause) {");
            self.line("        if matches!(cause, winit::event::StartCause::ResumeTimeReached { .. }) { if let Some(w) = &self.window { w.request_redraw(); } }");
            self.line("    }");
            self.blank();
        }
        self.line("    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {");
        if uses_ax {
            self.line("        if let (Some(ax), Some(w)) = (self.__ax.as_mut(), self.window.as_ref()) { ax.process_event(w, &event); }");
        }
        if on_demand {
            // Input and resize invalidate the window; everything else (including the
            // frame's own RedrawRequested) does not.
            self.line("        if matches!(event, WindowEvent::KeyboardInput { .. } | WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. } | WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } | WindowEvent::Ime(_) | WindowEvent::ModifiersChanged(_) | WindowEvent::MouseWheel { .. } | WindowEvent::Focused(_) | WindowEvent::Touch(_)) {");
            self.line("            if let Some(w) = &self.window { w.request_redraw(); }");
            self.line("        }");
        }
        self.line("        match event {");
        self.line("            WindowEvent::CloseRequested => { event_loop.exit(); }");
        self.line("            WindowEvent::Resized(size) => {");
        self.line("                self.__win_w = size.width as isize;");
        self.line("                self.__win_h = size.height as isize;");
        self.line("                self.__resized = true;");
        self.line("            }");
        self.line("            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {");
        self.line("                self.__scale = scale_factor as f32;");
        self.line("                if let Some(w) = &self.window { let ps = w.inner_size(); self.__win_w = ps.width as isize; self.__win_h = ps.height as isize; }");
        self.line("                self.__resized = true;");
        self.line("            }");
        self.line("            WindowEvent::KeyboardInput { event: ref key_event, .. } => {");
        self.line("                let key_str = __boring_logical_key_name(&key_event.logical_key);");
        self.line("                match key_event.state {");
        self.line("                    ElementState::Pressed  => {");
        self.line("                        self.__input.key_down(&key_str);");
        self.line("                        // Text the key produced (an input method's own commit arrives as `Ime::Commit`).");
        self.line("                        if let Some(t) = &key_event.text { if self.__input.mods & 8 == 0 && (self.__input.mods & 2 == 0 || self.__input.mods & 4 != 0) { for c in t.chars() { self.__input.typed(c); } } }");
        self.line("                        if self.__keys.insert(key_str.clone()) { self.__keys_pressed.insert(key_str); }");
        self.line("                    }");
        self.line("                    ElementState::Released => { self.__input.key_up(&key_str); self.__keys.remove(&key_str); }");
        self.line("                    _ => {}");
        self.line("                }");
        self.line("            }");
        self.line("            WindowEvent::ModifiersChanged(m) => {");
        self.line("                let st = m.state();");
        self.line("                self.__input.set_mods(st.shift_key(), st.control_key(), st.alt_key(), st.super_key());");
        self.line("            }");
        self.line("            WindowEvent::Focused(f) => { self.__input.set_focus(f); }");
        self.line("            WindowEvent::Ime(ime) => {");
        self.line("                match ime {");
        self.line("                    winit::event::Ime::Preedit(s, cur) => self.__input.ime_preedit(&s, cur),");
        self.line("                    winit::event::Ime::Commit(s) => self.__input.ime_commit(&s),");
        self.line("                    _ => self.__input.ime_preedit(\"\", None),");
        self.line("                }");
        self.line("            }");
        self.line("            WindowEvent::MouseWheel { delta, .. } => {");
        self.line("                match delta {");
        self.line("                    winit::event::MouseScrollDelta::LineDelta(x, y) => self.__input.scroll(x * 40.0, y * 40.0),");
        self.line("                    winit::event::MouseScrollDelta::PixelDelta(p) => self.__input.scroll(p.x as f32, p.y as f32),");
        self.line("                }");
        self.line("            }");
        // Window pixels -> pixel-buffer coordinates (the window may be HiDPI-scaled or resized). The window size is the
        // one the surface was last sized to (`Resized`), not `inner_size()`: on iOS that is the safe area, smaller than the
        // view the touches are located in.
        self.line("            WindowEvent::CursorMoved { position, .. } => {");
        self.line("                if self.window.is_some() {");
        self.line("                    let ws = winit::dpi::PhysicalSize::new(self.__win_w.max(1) as u32, self.__win_h.max(1) as u32);");
        self.line("                    self.__mouse_x = (position.x * self.__buf_w / (ws.width.max(1) as f64)).floor() as isize;");
        self.line("                    self.__mouse_y = (position.y * self.__buf_h / (ws.height.max(1) as f64)).floor() as isize;");
        self.line("                }");
        self.line("            }");
        self.line("            WindowEvent::Touch(__boring_t) => {");
        self.line("                if self.window.is_some() {");
        self.line("                    let ws = winit::dpi::PhysicalSize::new(self.__win_w.max(1) as u32, self.__win_h.max(1) as u32);");
        for l in crate::transpiler::screen_input::touch_arm_body("self.__input",
            "__boring_t.location.x * self.__buf_w / (ws.width.max(1) as f64)",
            "__boring_t.location.y * self.__buf_h / (ws.height.max(1) as f64)") { self.line(&format!("                    {l}")); }
        self.line("                }");
        self.line("            }");
        self.line("            WindowEvent::MouseInput { state, button, .. } => {");
        self.line("                let down = state == ElementState::Pressed;");
        self.line("                match button {");
        self.line("                    winit::event::MouseButton::Left => { self.__mouse_left = down; self.__input.mouse_button(1, down); }");
        self.line("                    winit::event::MouseButton::Right => { self.__mouse_right = down; self.__input.mouse_button(2, down); }");
        self.line("                    winit::event::MouseButton::Middle => { self.__mouse_middle = down; self.__input.mouse_button(4, down); }");
        self.line("                    _ => {}");
        self.line("                }");
        self.line("            }");
        self.line("            WindowEvent::RedrawRequested => {");
        self.line("                self.__update_safe();");
        // A deadline this frame has reached is spent; the body may schedule the next one.
        self.line("                if self.__wake_at.is_some_and(|t| t <= std::time::Instant::now()) { self.__wake_at = None; }");
        // Render loop body with self. prefix
        self.in_method = true;
        let w = width.clone();
        let h = height.clone();
        for stmt in &loop_body {
            self.emit_render_stmt(stmt, &w, &h, "                ");
        }
        self.in_method = false;
        self.line("                self.__keys_pressed.clear();");
        self.line("                self.__input.end_frame();");
        if uses_ax { self.line("                self.__ax_requests.clear();"); }
        self.line("                self.__resized = false;");
        self.line("                self.__frame += 1;");
        if on_demand {
            // On-demand: only schedule another frame when the body asked for one
            // (`screen.requestRedraw()`); input/resize/expose events schedule their own.
            self.line("                if std::mem::take(&mut self.__redraw) { if let Some(w) = &self.window { w.request_redraw(); } }");
            self.line("                event_loop.set_control_flow(match self.__wake_at { Some(t) => winit::event_loop::ControlFlow::WaitUntil(t), None => winit::event_loop::ControlFlow::Wait });");
        } else {
            self.line("                self.__redraw = false;");
            self.line("                if let Some(w) = &self.window { w.request_redraw(); }");
        }
        self.line("            }");
        self.line("            _ => {}");
        self.line("        }");
        self.line("    }");
        self.line("}");
        self.blank();

        // ── fn main() ─────────────────────────────────────────────────────────
        self.line("fn main() {");
        self.line("    #[cfg(target_os = \"ios\")]");
        self.line("    __boring_scene::register();");
        self.line("    let (instance, adapter, device, queue) = pollster::block_on(async {");
        self.line("        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());");
        self.line("        let adapter  = instance.request_adapter(&wgpu::RequestAdapterOptions::default())");
        self.line("            .await.expect(\"No GPU adapter found\");");
        self.line("        let mut __boring_required_features = wgpu::Features::empty();");
        if self.has_emulated_shader {
            self.line("        if adapter.features().contains(wgpu::Features::SUBGROUP) { __boring_required_features |= wgpu::Features::SUBGROUP; }");
        }
        if self.requires_f16 {
            self.line("        assert!(adapter.features().contains(wgpu::Features::SHADER_F16), \"This GPU adapter does not support the shader-f16 feature required by this program\");");
            self.line("        __boring_required_features |= wgpu::Features::SHADER_F16;");
        }
        self.line("        let (device, queue) = adapter");
        self.line("            .request_device(&wgpu::DeviceDescriptor {");
        self.line("                required_features: __boring_required_features,");
        self.line("                required_limits: adapter.limits(),");
        self.line("                ..Default::default()");
        self.line("            })");
        self.line("            .await.expect(\"Failed to create device\");");
        self.line("        (instance, adapter, device, queue)");
        self.line("    });");
        // Defensive: see the non-Screen async_main path for why.
        self.line("    let _ = device.poll(wgpu::MaintainBase::Wait);");
        // See the non-Screen async_main path's identical call for why: reports
        // (instead of panicking on) a validation error not already inside an
        // explicit error scope, e.g. pipeline creation at kernel `new()` time.
        self.line("    device.on_uncaptured_error(Box::new(|e| eprintln!(\"boring: uncaptured GPU error: {}\", e)));");
        self.line("    let device = std::sync::Arc::new(device);");
        self.line("    let queue  = std::sync::Arc::new(queue);");
        self.line("    let _ = __BORING_GPU_DEVICE.set(std::sync::Arc::clone(&device));");
        self.line("    let _ = __BORING_GPU_QUEUE.set(std::sync::Arc::clone(&queue));");
        // Arc-wrapped (like device/queue above) because `wgpu::Adapter` has no `Clone`; the
        // `__App.adapter` field is typed `Arc<wgpu::Adapter>` to match.
        // This emitter and the render-loop one (`emit_render_stmt`) are narrow shape-matchers,
        // not a splice into the general per-statement pipeline: `check_screen_program_shape`
        // and `emit_render_stmt`'s error fallback reject anything outside the recognized
        // whitelist with a `TranspileError` instead of silently dropping it.
        self.line("    let adapter = std::sync::Arc::new(adapter);");
        for (name, val) in &scalar_fields {
            self.line(&format!("    let {name}: i64 = {val};"));
        }
        // The program's own top-level code (`__boring_screen_setup`): kernels with constructor
        // arguments (their `init` runs here), view runtimes, display lists... Needs the global
        // device/queue set above.
        if !self.setup_fields.is_empty() {
            let names: Vec<String> = self.setup_fields.iter().map(|(n, _)| n.clone()).collect();
            let tuple = format!("({}{})", names.join(", "), if names.len() == 1 { "," } else { "" });
            self.line(&format!("    let {tuple} = match __boring_screen_setup() {{"));
            self.line("        Ok(v) => v,");
            self.line("        Err(e) => { eprintln!(\"error: {}\", e); std::process::exit(1); }");
            self.line("    };");
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
        for (name, _) in &self.setup_fields.clone() {
            self.line(&format!("        {name},"));
        }
        self.line("        __keys: Default::default(),");
        self.line("        __keys_pressed: Default::default(),");
        self.line("        __mouse_x: 0, __mouse_y: 0,");
        self.line("        __mouse_left: false, __mouse_right: false, __mouse_middle: false,");
        self.line("        __input: __BoringInput::new(),");
        self.line(&format!("        __win_w: ({width}) as isize, __win_h: ({height}) as isize, __resized: false, __ime_on: cfg!(not(any(target_os = \"ios\", target_os = \"android\"))), __safe: [0; 4], __scale: 1.0,"));
        self.line(&format!("        __buf_w: ({width}) as f64, __buf_h: ({height}) as f64, __surface_cfg: None,"));
        self.line("        __start_time: std::time::Instant::now(),");
        self.line("        __frame: 0, __redraw: false, __wake_at: None,");
        if uses_ax { self.line("        __ax: None, __ax_proxy: None, __ax_active: false, __ax_last: None, __ax_requests: Vec::new(),"); }
        self.line("        window: None, surface: None,");
        self.line("        __blit_pipeline: None, __blit_bg: None, __blit_dim_buf: None,");
        self.line("    };");
        if uses_ax {
            self.line("    let event_loop = EventLoop::<__AxEvent>::with_user_event().build().unwrap();");
            self.line("    app.__ax_proxy = Some(event_loop.create_proxy());");
        } else {
            self.line("    let event_loop = EventLoop::new().unwrap();");
        }
        self.line("    event_loop.run_app(&mut app).unwrap_or_else(|_| {});");
        self.line("}");
    }

    // -- Screen helpers -------------------------------------------------------

    /// `Screen(..., hidpi = true)`: `Dimension` is logical points and the window follows the display's scale.
    fn screen_hidpi(&self) -> bool {
        for item in &self.program.items {
            if let Item::Let(s) = item {
                if let Some(val) = &s.value {
                    if let ExprKind::Call(callee, args) = &val.kind {
                        if matches!(&callee.kind, ExprKind::Var(n) if n == "Screen") {
                            return args.iter().any(|a| a.label.as_deref() == Some("hidpi") && matches!(&a.value.kind, ExprKind::Bool(true)));
                        }
                    }
                }
            }
        }
        false
    }

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
                } else if let Some(s) = self.try_emit_ime_allowed(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_ime_area(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_set_clipboard(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_request_redraw(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_request_redraw_after(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_wake(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_set_accessibility(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_render_dispatch(e, width, height) {
                    self.line(&format!("{indent}{s};"));
                } else if let Some(s) = self.try_emit_field_swap(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_field_assign(e) {
                    self.line(&format!("{indent}{s}"));
                } else if let Some(s) = self.try_emit_kernel_realloc(e) {
                    self.line(&format!("{indent}{s}"));
                } else if matches!(&e.kind, ExprKind::Call(c, _) if matches!(&c.kind, ExprKind::Var(f)
                    if self.is_general_fn(f)))
                    || matches!(&e.kind, ExprKind::MethodCall(o, m, a)
                        if self.setup_method_modes(o, m).is_some_and(|modes| modes.len() == a.len()))
                    || matches!(&e.kind, ExprKind::Call(c, a) if matches!(&c.kind, ExprKind::Field(o, m)
                        if self.setup_method_modes(o, m).is_some_and(|modes| modes.len() == a.len()))) {
                    // an ordinary Boring function (or a method of a setup value) called for its effect
                    let call = self.host_expr(e);
                    self.line(&format!("{indent}{call};"));
                } else if let Some(s) = self.try_emit_setup_assign(e) {
                    self.line(&format!("{indent}{s}"));
                } else {
                    let hint = match &e.kind {
                        ExprKind::Assign(l, _) if matches!(&l.kind, ExprKind::Var(_)) =>
                            "; the assigned variable must be a top-level `var` of the program's setup code (or a `var` declared in the loop body)",
                        ExprKind::MethodCall(..) =>
                            "; a method can be called on a top-level struct value (declared with `let`/`mut` at the top level) whose type is declared in this program",
                        _ => "; only calls of Boring functions, `screen.*` queries, kernel field updates and assignments to top-level variables are supported there",
                    };
                    self.push_error(e.line, e.col,
                        format!("wgpu target: expression not supported inside a Screen render loop{hint}"));
                }
            }
            // A per-frame local (`let c = palette(screen.frame)`), visible to the rest of the frame body.
            Stmt::Let(ls) if ls.value.is_some() => {
                self.tenv.track_local(&ls.name, ls.ty.as_ref(), ls.value.as_ref(), None);
                let v = self.host_expr(ls.value.as_ref().unwrap());
                let mutability = if ls.binding.is_mutable() { "mut " } else { "" };
                let local_ty = ls.ty.clone()
                    .or_else(|| self.host_arg_type(ls.value.as_ref().unwrap()))
                    .or_else(|| self.host_float_kind(ls.value.as_ref().unwrap())
                        .map(|k| match k {
                            "f16" => Type::Float16,
                            "f32" => Type::Float32,
                            _ => Type::Float64,
                        }));
                match local_ty {
                    Some(t) => { self.frame_local_types.borrow_mut().insert(ls.name.clone(), t); }
                    None => { self.frame_local_types.borrow_mut().remove(&ls.name); }
                }
                self.frame_locals.borrow_mut().push(ls.name.clone());
                self.line(&format!("{indent}let {mutability}{} = {v};", ls.name));
            }
            // `for k in screen.key_events:` / `key_up_events`: the frame's key events in order.
            Stmt::For(fs) if fs.vars.len() == 1 && matches!(&fs.iterable.kind, ExprKind::Field(o, m)
                if matches!(&o.kind, ExprKind::Var(n) if n == "screen")
                    && crate::transpiler::screen_input::event_list_field(m).is_some()) =>
            {
                let ExprKind::Field(_, m) = &fs.iterable.kind else { unreachable!() };
                let field = crate::transpiler::screen_input::event_list_field(m).unwrap();
                if crate::transpiler::screen_input::body_has_break(&fs.body) {
                    self.push_error(fs.line, fs.col, "wgpu target: `break` inside a `for ... in screen.key_events` loop is not supported");
                }
                let deeper = format!("{indent}    ");
                self.line(&format!("{indent}for __ev in self.__input.{field}.clone() {{"));
                self.line(&format!("{deeper}let {} = std::sync::Arc::<str>::from(__ev.as_str());", fs.vars[0]));
                self.event_vars.borrow_mut().push(fs.vars[0].clone());
                for st in &fs.body { self.emit_render_stmt(st, width, height, &deeper); }
                self.event_vars.borrow_mut().pop();
                self.line(&format!("{indent}}}"));
            }
            // `for r in screen.accessibilityRequests:` -- what the screen reader asked for this frame
            // (`r.id`, `r.action`, `r.arg`), in arrival order.
            Stmt::For(fs) if fs.vars.len() == 1 && matches!(&fs.iterable.kind, ExprKind::Field(o, m)
                if m == "accessibilityRequests" && matches!(&o.kind, ExprKind::Var(n) if n == "screen")) =>
            {
                let deeper = format!("{indent}    ");
                self.line(&format!("{indent}for {} in std::mem::take(&mut self.__ax_requests) {{", fs.vars[0]));
                self.event_vars.borrow_mut().push(fs.vars[0].clone());
                for st in &fs.body { self.emit_render_stmt(st, width, height, &deeper); }
                self.event_vars.borrow_mut().pop();
                self.line(&format!("{indent}}}"));
            }
            Stmt::If(if_stmt) => {
                if let Some(s) = self.try_emit_screen_key_if(if_stmt) {
                    self.line(&format!("{indent}{s}"));
                } else {
                    // General `if`/`elif`/`else`: host-expression conditions
                    // (`screen.key(...)`, `screen.key_pressed(...)`, `screen.mouse.*`, ...).
                    let deeper = format!("{indent}    ");
                    for (i, (cond, body)) in if_stmt.branches.iter().enumerate() {
                        let kw = if i == 0 { "if" } else { "} else if" };
                        let c = self.host_expr(cond);
                        self.line(&format!("{indent}{kw} {c} {{"));
                        for st in body { self.emit_render_stmt(st, width, height, &deeper); }
                    }
                    if let Some(eb) = &if_stmt.else_body {
                        self.line(&format!("{indent}}} else {{"));
                        for st in eb { self.emit_render_stmt(st, width, height, &deeper); }
                    }
                    self.line(&format!("{indent}}}"));
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

    /// `k = K(Dimension(screen.width, screen.height))` / `k = makeKernel(screen.width, screen.height)`
    /// in the render loop: build the new kernel, then resize the swapchain to the window's current
    /// size and re-point the blit at the new pixel buffer (`__App::__reblit`).
    fn try_emit_kernel_realloc(&self, e: &Expr) -> Option<String> {
        let ExprKind::Assign(lhs, rhs) = &e.kind else { return None };
        let ExprKind::Var(name) = &lhs.kind else { return None };
        let pfx = if self.in_method { "self." } else { "" };
        // `k` must be a kernel the host owns: a host-constructed one or a setup value of a kernel type.
        let host_kernel = self.program.items.iter().find_map(|item| match item {
            Item::Let(s) if &s.name == name => match s.value.as_ref().map(|v| &v.kind) {
                Some(ExprKind::Call(c, _)) => match &c.kind {
                    ExprKind::Var(k) if self.kernel_names.contains(k) => Some(k.clone()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        });
        // A kernel the setup code built from real constructor arguments: run its generated rebuild
        // function (`__boring_screen_rebuild_<k>`, see `helpers::loop_kernel_rebuilds`).
        if self.in_method && self.setup_fields.iter().any(|(f, _)| f == name) {
            let kernel_set = self.kernel_name_set();
            let rebuilds = crate::transpiler::helpers::loop_kernel_rebuilds(self.program, &kernel_set);
            if let Some((_, rebuild_rhs)) = rebuilds.iter().find(|(v, _)| v == name) {
                let returns: Vec<String> = self.setup_fields.iter().map(|(n, _)| n.clone()).collect();
                { let _ = rebuild_rhs;
                    let captured = crate::transpiler::helpers::rebuild_captures(name, rhs, &returns);
                    let mut args = vec![format!("&__ScreenSize {{ width: {pfx}__win_w, height: {pfx}__win_h, scale: {pfx}__scale }}")];
                    args.extend(captured.iter().map(|c| format!("&{pfx}{c}")));
                    return Some(format!(
                        "{pfx}{name} = __boring_screen_rebuild_{name}({}).expect(\"kernel rebuild failed\"); {pfx}__reblit();",
                        args.join(", ")));
                }
            }
        }
        let new_value = match (&rhs.kind, host_kernel) {
            (ExprKind::Call(c, args), Some(kname)) if matches!(&c.kind, ExprKind::Var(k) if *k == kname)
                && !self.setup_fields.iter().any(|(f, _)| f == name) =>
            {
                let decl = self.effective_kernels.iter().find(|d| d.name == kname);
                let dim_args = if decl.is_some_and(kernel_has_dim_field) {
                    match args.first().map(|a| &a.value.kind) {
                        Some(ExprKind::Call(dc, da)) if matches!(&dc.kind, ExprKind::Var(n) if n == "Dimension") && da.len() == 2 => {
                            format!("({}) as i32, ({}) as i32, ", self.host_expr(&da[0].value), self.host_expr(&da[1].value))
                        }
                        _ => return None,
                    }
                } else { String::new() };
                format!("{kname}::new({dim_args}std::sync::Arc::clone(&{pfx}device), std::sync::Arc::clone(&{pfx}queue))")
            }
            // `k = makeKernel(...)`: only when `k` is a kernel -- a per-frame local or an event variable
            // (`var changed = false` ... `changed = sync(...)`) is a plain assignment, not a rebuild.
            (ExprKind::Call(c, _), _) if matches!(&c.kind, ExprKind::Var(f) if self.is_general_fn(f))
                && !self.frame_locals.borrow().contains(name)
                && !self.event_vars.borrow().contains(name) => {
                self.host_expr(rhs)
            }
            _ => return None,
        };
        if !self.in_method { return None; }
        Some(format!("{pfx}{name} = {new_value}; {pfx}__reblit();"))
    }

    /// `dirty = true` / `n += 1` in the render loop where the target is a top-level `var` of the
    /// program's setup code (an `__App` field) or a per-frame local: a plain assignment.
    fn try_emit_setup_assign(&self, e: &Expr) -> Option<String> {
        let ExprKind::Assign(lhs, rhs) = &e.kind else { return None };
        let ExprKind::Var(name) = &lhs.kind else { return None };
        if self.event_vars.borrow().contains(name) { return None; }
        let is_field = self.setup_fields.iter().any(|(f, _)| f == name);
        if !is_field && !self.frame_locals.borrow().contains(name) { return None; }
        let pfx = if self.in_method && is_field { "self." } else { "" };
        Some(format!("{pfx}{name} = {};", self.host_expr(rhs)))
    }

    /// `screen.setImeArea(x, y, w, h)` -- where the input method may put its candidate window,
    /// in buffer coordinates (scaled like the mouse).
    fn try_emit_ime_area(&self, e: &Expr) -> Option<String> {
        let ExprKind::MethodCall(obj, method, args) = &e.kind else { return None };
        if method != "setImeArea" || args.len() != 4 { return None; }
        if !matches!(&obj.kind, ExprKind::Var(n) if n == "screen") || !self.in_method { return None; }
        let a: Vec<String> = args.iter().map(|a| self.host_expr(&a.value)).collect();
        Some(format!(
            "if let Some(w) = &self.window {{ let ws = winit::dpi::PhysicalSize::new(self.__win_w.max(1) as u32, self.__win_h.max(1) as u32); let (sx, sy) = (ws.width.max(1) as f64 / self.__buf_w, ws.height.max(1) as f64 / self.__buf_h); w.set_ime_cursor_area(winit::dpi::PhysicalPosition::new(({}) as f64 * sx, ({}) as f64 * sy), winit::dpi::PhysicalSize::new(({}) as f64 * sx, ({}) as f64 * sy)); }}",
            a[0], a[1], a[2], a[3]))
    }

    /// `screen.setImeAllowed(on)` -- whether the window takes input-method events; on a phone this is what
    /// shows and hides the soft keyboard (a program calls it with "a text field has focus").
    fn try_emit_ime_allowed(&self, e: &Expr) -> Option<String> {
        let ExprKind::MethodCall(obj, method, args) = &e.kind else { return None };
        if method != "setImeAllowed" || args.len() != 1 { return None; }
        if !matches!(&obj.kind, ExprKind::Var(n) if n == "screen") || !self.in_method { return None; }
        Some(format!(
            "if let Some(w) = &self.window {{ let on = {}; if on != self.__ime_on {{ self.__ime_on = on; w.set_ime_allowed(on); }} }}",
            self.host_expr(&args[0].value)))
    }

    /// `screen.setClipboard(s)` -- write text to the system clipboard (failures ignored).
    fn try_emit_set_clipboard(&self, e: &Expr) -> Option<String> {
        let ExprKind::MethodCall(obj, method, args) = &e.kind else { return None };
        if method != "setClipboard" || args.len() != 1 { return None; }
        if !matches!(&obj.kind, ExprKind::Var(n) if n == "screen") || !self.in_method { return None; }
        Some(format!("__boring_clipboard_set(&*({}));", self.host_expr(&args[0].value)))
    }

    /// `screen.setAccessibility(nodes)` -- hand the accessibility tree (a `[struct]` with the fields of boring-ui's
    /// `AxNode`) to AccessKit. The expression is only evaluated while an assistive technology is connected.
    fn try_emit_set_accessibility(&self, e: &Expr) -> Option<String> {
        let (obj, args) = match &e.kind {
            ExprKind::MethodCall(obj, m, args) if m == "setAccessibility" => (obj, args),
            ExprKind::Call(c, args) => match &c.kind {
                ExprKind::Field(obj, m) if m == "setAccessibility" => (obj, args),
                _ => return None,
            },
            _ => return None,
        };
        if args.len() != 1 || !matches!(&obj.kind, ExprKind::Var(n) if n == "screen") || !self.in_method { return None; }
        let nodes = self.host_expr(&args[0].value);
        // List-position fields exist only on newer node structs (boring-ui's `AxNode`): read them when declared.
        let has = |f: &str| self.program.items.iter().any(|i| matches!(i, Item::Struct(s) if s.fields.iter().any(|x| x.name == f)));
        let pos = if has("posInSet") { "n.posInSet as i64" } else { "0" };
        let size = if has("setSize") { "n.setSize as i64" } else { "0" };
        Some(format!(
            "if self.__ax_active {{ let __ax_in = {nodes}; self.__ax_push(__ax_in.iter().map(|n| __AxNode {{ id: n.id as i64, parent: n.parent as i64, role: n.role.to_string(), name: n.name.to_string(), value: n.value.to_string(), hint: n.hint.to_string(), level: n.level as i64, x: n.x as f64, y: n.y as f64, w: n.w as f64, h: n.h as f64, focused: n.focused, disabled: n.disabled, actions: n.actions.to_string(), children: n.children.iter().map(|c| *c as i64).collect(), pos_in_set: {pos}, set_size: {size} }}).collect()); }}"))
    }

    /// `screen.requestRedraw()` -- schedule another frame when the window redraws on demand.
    fn try_emit_request_redraw(&self, e: &Expr) -> Option<String> {
        let ExprKind::MethodCall(obj, method, args) = &e.kind else { return None };
        if method != "requestRedraw" || !args.is_empty() { return None; }
        if !matches!(&obj.kind, ExprKind::Var(n) if n == "screen") { return None; }
        let pfx = if self.in_method { "self." } else { "" };
        Some(format!("{pfx}__redraw = true;"))
    }

    /// `screen.requestRedrawAfter(ms)` -- schedule a frame `ms` milliseconds from now when the window redraws on
    /// demand; the earliest pending deadline wins, and the next frame's `screen.time` reflects the wait.
    fn try_emit_request_redraw_after(&self, e: &Expr) -> Option<String> {
        let ExprKind::MethodCall(obj, method, args) = &e.kind else { return None };
        if method != "requestRedrawAfter" || args.len() != 1 { return None; }
        if !matches!(&obj.kind, ExprKind::Var(n) if n == "screen") || !self.in_method { return None; }
        Some(format!(
            "{{ let __d = std::time::Instant::now() + std::time::Duration::from_secs_f64((({}) as f64 / 1000.0).clamp(0.0, 31_536_000.0)); self.__wake_at = Some(match self.__wake_at {{ Some(t) if t < __d => t, _ => __d }}); }}",
            self.host_expr(&args[0].value)))
    }

    /// `screen.wake()` -- wake the render loop from any thread (it schedules a frame like `requestRedraw`).
    fn try_emit_wake(&self, e: &Expr) -> Option<String> {
        let ExprKind::MethodCall(obj, method, args) = &e.kind else { return None };
        if method != "wake" || !args.is_empty() { return None; }
        if !matches!(&obj.kind, ExprKind::Var(n) if n == "screen") { return None; }
        Some("__boring_wake();".to_string())
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
        let pfx = if self.in_method { "self." } else { "" };
        // After a reallocation (`k = K(Dimension(screen.width, screen.height))`) the inferred grid
        // follows the buffer's current size, not the window's initial one.
        let (width, height) = if self.in_method {
            (format!("({pfx}__buf_w as u32)"), format!("({pfx}__buf_h as u32)"))
        } else {
            (width.to_string(), height.to_string())
        };
        let gx = format!("({width} + {bx} - 1) / {bx}");
        let gy = format!("({height} + {by} - 1) / {by}");
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
                    // Only a *kernel* on the right is a buffer alias; `k.f = enc.f` with a
                    // host local `enc` is an ordinary (array) assignment, handled below.
                    if let (ExprKind::Var(robj_name), true) = (&robj.kind,
                        matches!(&robj.kind, ExprKind::Var(rn) if self.find_kernel_decl_for_var(rn).is_some())) {
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
                let pfx = if self.in_method { "self." } else { "" };
                // Array field assignment: `k.colors = palette(screen.frame)` re-uploads the
                // whole array through the kernel's own `copy_<field>_to_device`, which
                // reallocates the buffer (and rebuilds the bind group) when the length changed.
                if let Some(decl) = self.find_kernel_decl_for_var(obj_name) {
                    if let Some(f) = decl.fields.iter().find(|f| f.name == field.as_str()
                        && matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified | GpuQual::Surface)
                        && is_buffer_array_ty(&f.ty))
                    {
                        let host_ty = host_scalar_type(&array_inner(&f.ty));
                        let rhs_str = self.host_expr(rhs);
                        let conv = if host_ty == "bool" { "x".to_string() } else { format!("x as {host_ty}") };
                        // A re-uploaded presented buffer (`'surface`, or any field the loop hands to
                        // `screen.present(k.field)`, e.g. a `'unified` one) may be a new buffer: the
                        // swapchain and the blit follow it (sized to the window, see `__App::__reblit`).
                        let presented = matches!(f.qual, GpuQual::Surface)
                            || self.screen_pixels_buf == format!("{obj_name}.{field}_buf");
                        let reblit = if presented && self.in_method { format!(" {pfx}__reblit();") } else { String::new() };
                        return Some(format!(
                            "{{ let __v = &({rhs_str}); {pfx}{obj_name}.copy_{field}_to_device(&__v.iter().map(|&x| {conv}).collect::<Vec<{host_ty}>>()); }}{reblit}"));
                    }
                }
                // Scalar field assignment: `k.t = expr` → update the host-side field.
                let mut rhs_str = self.host_expr(rhs);
                // `int`/`uint` host values are `isize`/`usize`; the kernel's host mirror is 32-bit.
                if let Some(f) = self.find_kernel_decl_for_var(obj_name)
                    .and_then(|d| d.fields.iter().find(|f| f.name == field.as_str()))
                {
                    let mut fty = &f.ty;
                    while let Type::Mut(inner) = fty { fty = inner; }
                    let host = match fty {
                        Type::Int => Some("i32"),
                        Type::Uint => Some("u32"),
                        Type::Named(n) if n == "int" => Some("i32"),
                        Type::Named(n) if n == "uint" => Some("u32"),
                        _ => None,
                    };
                    if let Some(host) = host {
                        rhs_str = format!("({rhs_str}) as {host}");
                    } else if (matches!(fty, Type::Float16)
                        || matches!(fty, Type::Named(n) if n == "float16" || n == "f16"))
                        && !crate::transpiler::host_types::gpu_host_expr_is_float16(rhs)
                    {
                        rhs_str = format!("half::f16::from_f32(({rhs_str}) as f32)");
                    } else if host_scalar_type(fty) == "f32" {
                        // `float`/`float64` kernel fields are f32 on the device (`host_scalar_type`), while
                        // `float(x)` itself is an f64 on the host: narrow at the assignment.
                        rhs_str = format!("({rhs_str}) as f32");
                    }
                }
                return Some(format!("{pfx}{obj_name}.{field} = {rhs_str};"));
            }
        }
        None
    }

    /// `screen.key("k")` -> held set, `screen.key_pressed("k")` -> edge-triggered set.
    fn screen_key_query(&self, method: &str, args: &[Arg]) -> String {
        let pfx = if self.in_method { "self." } else { "" };
        let set = if method == "key" { "__keys" } else { "__keys_pressed" };
        let k = match args.first().map(|a| &a.value.kind) {
            Some(ExprKind::Str(s)) => boring_key_to_rust(s),
            Some(_) => format!("&__boring_key_name(&*({}))", self.host_expr(&args[0].value)),
            None => "\"\"".into(),
        };
        format!("{pfx}{set}.contains({k})")
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
            // `screen.frame` -> the frame counter (`int`)
            ExprKind::Field(obj, field)
                if field == "frame" && matches!(&obj.kind, ExprKind::Var(n) if n == "screen") =>
            {
                format!("{pfx}__frame")
            }
            // `screen.width` / `.height` (live window size in pixels) / `.resized` (this frame follows a resize)
            ExprKind::Field(obj, field)
                if matches!(field.as_str(), "width" | "height" | "resized" | "scale" | "safeTop" | "safeLeft" | "safeBottom" | "safeRight")
                    && matches!(&obj.kind, ExprKind::Var(n) if n == "screen") =>
            {
                match field.as_str() {
                    "width" => format!("{pfx}__win_w"),
                    "height" => format!("{pfx}__win_h"),
                    "scale" => format!("{pfx}__scale"),
                    "safeTop" => format!("{pfx}__safe[0]"),
                    "safeLeft" => format!("{pfx}__safe[1]"),
                    "safeBottom" => format!("{pfx}__safe[2]"),
                    "safeRight" => format!("{pfx}__safe[3]"),
                    _ => format!("{pfx}__resized"),
                }
            }
            // `screen.clipboard`: read the system clipboard now (lazily, only when evaluated).
            ExprKind::Field(obj, field)
                if field == "clipboard" && matches!(&obj.kind, ExprKind::Var(n) if n == "screen") =>
            {
                crate::transpiler::screen_input::str_to_string("__boring_clipboard_get()")
            }
            // Input members of the shared runtime: `screen.text`, `.marked`, `.shift`, `.focused`, ...
            ExprKind::Field(obj, field)
                if matches!(&obj.kind, ExprKind::Var(n) if n == "screen")
                    && crate::transpiler::screen_input::property(field, "").is_some() =>
            {
                let (e, kind) = crate::transpiler::screen_input::property(field, &format!("{pfx}__input")).unwrap();
                if kind == crate::transpiler::screen_input::InKind::Str { crate::transpiler::screen_input::str_to_string(&e) } else { e }
            }
            // `screen.touch.x` / `.y`
            ExprKind::Field(obj, field)
                if matches!(&obj.kind, ExprKind::Field(sv, m) if m == "touch"
                    && matches!(&sv.kind, ExprKind::Var(n) if n == "screen"))
                    && crate::transpiler::screen_input::touch_property(field, "").is_some() =>
            {
                crate::transpiler::screen_input::touch_property(field, &format!("{pfx}__input")).unwrap()
            }
            // `screen.touchAt(i).x` / `.y`
            ExprKind::Field(obj, field)
                if matches!(&obj.kind, ExprKind::MethodCall(sv, m, a) if m == "touchAt" && a.len() == 1
                    && matches!(&sv.kind, ExprKind::Var(n) if n == "screen"))
                    && crate::transpiler::screen_input::touch_at_property(field, "", "").is_some() =>
            {
                let ExprKind::MethodCall(_, _, a) = &obj.kind else { unreachable!() };
                crate::transpiler::screen_input::touch_at_property(field, &self.host_expr(&a[0].value), &format!("{pfx}__input")).unwrap()
            }
            // `screen.mouse.scroll_x` / `.scroll_y`
            ExprKind::Field(obj, field)
                if matches!(&obj.kind, ExprKind::Field(sv, m) if m == "mouse"
                    && matches!(&sv.kind, ExprKind::Var(n) if n == "screen"))
                    && crate::transpiler::screen_input::mouse_property(field, "").is_some() =>
            {
                crate::transpiler::screen_input::mouse_property(field, &format!("{pfx}__input")).unwrap().0
            }
            // `screen.mouse.x` / `.y` / `.left` / `.right` / `.middle`
            ExprKind::Field(obj, field)
                if matches!(&obj.kind, ExprKind::Field(sv, m) if m == "mouse"
                    && matches!(&sv.kind, ExprKind::Var(n) if n == "screen"))
                    && matches!(field.as_str(), "x" | "y" | "left" | "right" | "middle") =>
            {
                format!("{pfx}__mouse_{field}")
            }
            // `screen.key_events.length` / `.count`
            ExprKind::Field(obj, field)
                if (field == "length" || field == "count")
                    && matches!(&obj.kind, ExprKind::Field(sv, m) if matches!(&sv.kind, ExprKind::Var(n) if n == "screen")
                        && crate::transpiler::screen_input::event_list_field(m).is_some()) =>
            {
                let ExprKind::Field(_, m) = &obj.kind else { unreachable!() };
                let list = crate::transpiler::screen_input::event_list_field(m).unwrap();
                format!("({pfx}__input.{list}.len() as isize)")
            }
            // `screen.key_events` / `screen.key_up_events` as a value: this frame's key names, in order.
            ExprKind::Field(obj, field)
                if matches!(&obj.kind, ExprKind::Var(n) if n == "screen")
                    && crate::transpiler::screen_input::event_list_field(field).is_some() =>
            {
                let list = crate::transpiler::screen_input::event_list_field(field).unwrap();
                format!("{pfx}__input.{list}.iter().map(|__k| std::sync::Arc::<str>::from(__k.as_str())).collect::<Vec<std::sync::Arc<str>>>()")
            }
            // `arr.count` / `arr.length` on a known array → length; a user struct's real
            // field of that name (or any receiver not known to be an array) stays a field read.
            ExprKind::Field(obj, field)
                if (field == "count" || field == "length")
                    && !self.tenv.is_real_member(obj, field, None)
                    && self.tenv.expr_is_array(obj, None) =>
            {
                format!("({}.len() as isize)", self.host_expr(obj))
            }
            ExprKind::Field(obj, field) => {
                // A read of a kernel's `int`/`uint` scalar field: the host mirror is 32-bit, host ints are `isize`/`usize`.
                if let ExprKind::Var(n) = &obj.kind {
                    if let Some(f) = self.find_kernel_decl_for_var(n).and_then(|d| d.fields.iter().find(|f| f.name == field.as_str())) {
                        let mut fty = &f.ty;
                        while let Type::Mut(inner) = fty { fty = inner; }
                        let wide = match fty {
                            Type::Int => Some("isize"),
                            Type::Uint => Some("usize"),
                            Type::Named(t) if t == "int" => Some("isize"),
                            Type::Named(t) if t == "uint" => Some("usize"),
                            _ => None,
                        };
                        if let Some(w) = wide {
                            return format!("({}.{} as {w})", self.host_expr(obj), field);
                        }
                    }
                }
                format!("{}.{}", self.host_expr(obj), field)
            }
            // `caret[0]` on a host array (a `let` bound to a call returning `[int]`, a setup value, ...).
            ExprKind::Index(base, idx) => {
                format!("{}[({}) as usize]", self.host_expr(base), self.host_expr(idx))
            }
            ExprKind::Array(elems) => {
                let es: Vec<_> = elems.iter().map(|x| self.host_expr(x)).collect();
                format!("vec![{}]", es.join(", "))
            }
            // `x.count()` / `x.length()`: a real struct method wins; a known array gets the builtin.
            ExprKind::MethodCall(obj, method, args)
                if args.is_empty() && (method == "count" || method == "length")
                    && !self.tenv.is_real_member(obj, method, None)
                    && self.tenv.expr_is_array(obj, None) =>
            {
                format!("({}.len() as isize)", self.host_expr(obj))
            }
            // A method of a struct value the setup code built (`pace.frame(rt, host, ..)`): each argument
            // is passed the way the method's generated signature takes it.
            ExprKind::MethodCall(obj, method, args)
                if self.setup_method_modes(obj, method).is_some_and(|m| m.len() == args.len()) =>
            {
                let modes = self.setup_method_modes(obj, method).unwrap();
                format!("{}.{method}({})", self.host_expr(obj), self.call_args(args, &modes).join(", "))
            }
            // Any other method call on a host value: `recv.method(args)`.
            ExprKind::MethodCall(obj, method, args)
                if !matches!(&obj.kind, ExprKind::Var(n) if n == "screen") =>
            {
                let args_s: Vec<_> = args.iter().map(|a| self.host_expr(&a.value)).collect();
                format!("{}.{}({})", self.host_expr(obj), method, args_s.join(", "))
            }
            // `screen.key("k")` (level) / `screen.key_pressed("k")` (edge) as expressions.
            ExprKind::MethodCall(obj, method, args)
                if (method == "key" || method == "key_pressed")
                    && matches!(&obj.kind, ExprKind::Var(n) if n == "screen") =>
            {
                self.screen_key_query(method, args)
            }
            // `screen.key_repeat(k)` / `key_released(k)` / `mouse_pressed(b)` / `mouse_released(b)`
            ExprKind::MethodCall(obj, method, args)
                if matches!(&obj.kind, ExprKind::Var(n) if n == "screen") && args.len() == 1
                    && crate::transpiler::screen_input::method(method, "", "").is_some() =>
            {
                crate::transpiler::screen_input::method(method, &self.host_expr(&args[0].value), &format!("{pfx}__input")).unwrap()
            }
            // `screen.mouse.scroll_x as float`: the member's own host type, converted.
            ExprKind::Cast(inner, ty) => match host_cast_type(ty) {
                Some("half::f16") => format!("half::f16::from_f32(({}) as f32)", self.host_expr(inner)),
                Some(t) => format!("({} as {t})", self.host_expr(inner)),
                None => self.host_expr(inner),
            },
            ExprKind::Str(v) => crate::transpiler::screen_input::str_to_string(&format!("{v:?}")),
            ExprKind::Bool(b) => format!("{b}"),
            ExprKind::UnaryOp(op, inner) => {
                let o = match op { UnaryOp::Neg => "-", UnaryOp::Not => "!", UnaryOp::BitNot => "!" };
                format!("({o}{})", self.host_expr(inner))
            }
            ExprKind::If(ifs) if ifs.else_body.is_some() => {
                // A branch that is a bare integer literal outside `i32` (`0xFF3060C0`) would be inferred `i32` when every
                // branch is a literal: pin it to `i64` (the surrounding `as` / assignment converts it).
                let all_lits = std::iter::once(ifs.else_body.as_ref().unwrap()).chain(ifs.branches.iter().map(|(_, b)| b))
                    .all(|b| matches!(b.last(), Some(Stmt::Expr(Expr { kind: ExprKind::Int(_), .. }))));
                let val = |b: &Vec<Stmt>| b.last().map(|st| match st {
                    Stmt::Expr(Expr { kind: ExprKind::Int(n), .. }) if all_lits && i32::try_from(*n).is_err() => format!("{n}i64"),
                    Stmt::Expr(e) => self.host_expr(e),
                    _ => self.unsupported_expr(e, "a statement other than a plain value as the last line of an `if` expression branch"),
                }).unwrap_or_default();
                let mut out = String::new();
                for (i, (c, b)) in ifs.branches.iter().enumerate() {
                    out.push_str(&format!("{}if {} {{ {} }}", if i == 0 { "" } else { " else " }, self.host_expr(c), val(b)));
                }
                let ev = val(ifs.else_body.as_ref().unwrap());
                format!("({out} else {{ {ev} }})")
            }
            // float32(expr) → expr as f32 (wgpu is f32-only on device, and this
            // host helper narrows the same way regardless of width). Bare
            // `float(expr)`/`float64(expr)` now join float32 here too —
            // `host_scalar_type` (this module) narrows `Type::Float64` to `"f32"`
            // for exactly the same reason (WGSL has no 64-bit float; the device
            // buffer/Params struct is always f32 in practice regardless of the
            // Boring-level width — see that function's own doc), so a `var float t`
            // Local field now gets an `f32` host mirror, not `f64`. This used to be
            // split (float/float64 cast to f64, matching a then-`f64` field type —
            // see plasma_metal.br's wgpu regression, the reason for the split in the
            // first place) — keeping that split after `host_scalar_type` changed
            // would reintroduce the exact same class of mismatch in the other
            // direction (`error[E0308]: expected f32, found f64` assigning this
            // cast's result into the now-`f32` field). Confirmed via a real
            // `cargo build` of a `var float t` Local field inside a `Screen`
            // render loop after narrowing `host_scalar_type` alone, tests/
            // wgpu_codegen.rs's own `host_bare_float_scalar_field_assign_casts_to_
            // f64_not_f32` only asserts the generated *text*, not that the project
            // actually compiles, so it didn't catch this on its own).
            ExprKind::Call(callee, args) => {
                if let ExprKind::Field(obj, method) = &callee.kind {
                    if let Some(modes) = self.setup_method_modes(obj, method).filter(|m| m.len() == args.len()) {
                        return format!("{}.{method}({})", self.host_expr(obj), self.call_args(args, &modes).join(", "));
                    }
                    if (method == "key" || method == "key_pressed")
                        && matches!(&obj.kind, ExprKind::Var(n) if n == "screen")
                    {
                        return self.screen_key_query(method, args);
                    }
                    if matches!(&obj.kind, ExprKind::Var(n) if n == "screen") && args.len() == 1 {
                        if let Some(q) = crate::transpiler::screen_input::method(method, &self.host_expr(&args[0].value), &format!("{pfx}__input")) {
                            return q;
                        }
                    }
                }
                if let ExprKind::Var(n) = &callee.kind {
                    if (n == "float16" || n == "f16") && args.len() == 1 {
                        let inner = self.host_expr(&args[0].value);
                        return if crate::transpiler::host_types::gpu_host_expr_is_float16(&args[0].value) {
                            inner
                        } else {
                            format!("half::f16::from_f32(({}) as f32)", inner)
                        };
                    }
                    // `int(x)`, `uint(x)`, `int32(x)`, ...: a numeric conversion (`as` truncates a float toward zero).
                    if args.len() == 1 && args[0].label.is_none() {
                        if let Some(t) = match n.as_str() {
                            "int" => Some("isize"), "uint" => Some("usize"),
                            "int8" => Some("i8"), "int16" => Some("i16"), "int32" => Some("i32"), "int64" => Some("i64"),
                            "uint8" => Some("u8"), "uint16" => Some("u16"), "uint32" => Some("u32"), "uint64" => Some("u64"),
                            _ => None,
                        } {
                            let inner = self.host_expr(&args[0].value);
                            return if crate::transpiler::host_types::gpu_host_expr_is_float16(&args[0].value) {
                                format!("(f32::from({inner}) as {t})")
                            } else {
                                format!("(({inner}) as {t})")
                            };
                        }
                    }
                    if (n == "float32" || n == "float" || n == "float64") && args.len() == 1 {
                        let inner = self.host_expr(&args[0].value);
                        return if crate::transpiler::host_types::gpu_host_expr_is_float16(&args[0].value) {
                            format!("f32::from({inner})")
                        } else {
                            // `float`/`float64` are f64 (`screen.time` is an f32), only `float32` is f32
                            let t = if n == "float32" { "f32" } else { "f64" };
                            format!("({inner} as {t})")
                        };
                    }
                }
                // A call to an ordinary Boring function: pass each argument the way the function's
                // generated signature takes it (`&x`, `&mut x`, or by value).
                if let ExprKind::Var(fname) = &callee.kind {
                    let fname = &self.resolve_free_overload(fname, args);
                    if let Some(modes) = self.general_fn_param_modes(fname) {
                        if modes.len() == args.len() {
                            let args_s = self.call_args(args, &modes);
                            return format!("{fname}({})", args_s.join(", "));
                        }
                    }
                }
                // Generic call (best effort)
                let args_s: Vec<_> = args.iter().map(|a| self.host_expr(&a.value)).collect();
                format!("{}({})", self.host_expr(callee), args_s.join(", "))
            }
            // A value built by `__boring_screen_setup` lives on `__App`.
            ExprKind::Var(n) if self.setup_fields.iter().any(|(f, _)| f == n) => format!("{pfx}{n}"),
            // A kernel instance (`var mut k = Fill(..)`) is a field of `__App`, read as `k.total`.
            ExprKind::Var(n) if !self.frame_locals.borrow().contains(n) && !self.event_vars.borrow().contains(n)
                && self.find_kernel_decl_for_var(n).is_some() => format!("{pfx}{n}"),
            // A top-level `let width = 8` is emitted as `const WIDTH` (see `emit_top`); a read of it
            // that is neither a setup value, a per-frame local nor an event variable names that const.
            ExprKind::Var(n) if !self.frame_locals.borrow().contains(n) && !self.event_vars.borrow().contains(n)
                && self.program.items.iter().any(|it| matches!(it, Item::Let(l) if &l.name == n)) => n.to_uppercase(),
            ExprKind::Var(n) => n.clone(),
            ExprKind::Int(v)   => format!("{v}"),
            // Untyped: takes the type of whatever it meets (`(x as f64) * 1000.0`, `k.t = 0.5` on an f32 field).
            ExprKind::Float(v) => {
                let t = format!("{v}");
                if t.contains(['.', 'e', 'E', 'N', 'i']) { t } else { format!("{t}.0") }
            }
            ExprKind::BinOp(op, l, r) => {
                let op_s = match op {
                    BinOp::Add => "+", BinOp::Sub => "-",
                    BinOp::Mul => "*", BinOp::Div => "/", BinOp::Rem => "%",
                    BinOp::Eq => "==", BinOp::NotEq => "!=",
                    BinOp::Lt => "<", BinOp::Gt => ">", BinOp::LtEq => "<=", BinOp::GtEq => ">=",
                    BinOp::And => "&&", BinOp::Or => "||",
                    BinOp::BitAnd => "&", BinOp::BitOr => "|", BinOp::BitXor => "^",
                    _ => return self.unsupported_expr(e, "this operator"),
                };
                let (mut ls, mut rs) = (self.host_expr(l), self.host_expr(r));
                // Rust's `half::f16` does not mix directly with primitive floats. Promote the
                // half side to the wider operand; f16/f16 arithmetic remains native half arithmetic.
                if !matches!(op, BinOp::And | BinOp::Or) {
                    match (self.host_float_kind(l), self.host_float_kind(r)) {
                        (Some("f16"), None) => rs = format!("half::f16::from_f32(({rs}) as f32)"),
                        (None, Some("f16")) => ls = format!("half::f16::from_f32(({ls}) as f32)"),
                        (Some("f16"), Some("f32")) => ls = format!("({ls}).to_f32()"),
                        (Some("f32"), Some("f16")) => rs = format!("({rs}).to_f32()"),
                        (Some("f16"), Some("f64")) => ls = format!("({ls}).to_f64()"),
                        (Some("f64"), Some("f16")) => rs = format!("({rs}).to_f64()"),
                        (Some("f32"), Some("f64")) => ls = format!("({ls} as f64)"),
                        (Some("f64"), Some("f32")) => rs = format!("({rs} as f64)"),
                        _ => {}
                    }
                }
                format!("({ls} {op_s} {rs})")
            }
            _ => self.unsupported_expr(e, "this expression"),
        }
    }

    /// A host expression the emitter cannot translate: record a compile error (never a placeholder
    /// comment in the generated Rust, which would silently drop code) and return a harmless token.
    fn unsupported_expr(&self, e: &Expr, what: &str) -> String {
        self.expr_errors.borrow_mut().push(TranspileError::at(
            format!("wgpu target: {what} is not supported inside a Screen render loop; \
                     move it into an ordinary Boring function and call that (`let v = helper(args)`)"),
            e.line, e.col));
        "()".into()
    }

    /// The float width a host expression evaluates to, when it is known.
    /// Untyped literals and non-float values are `None`.
    fn host_float_kind(&self, e: &Expr) -> Option<&'static str> {
        match &e.kind {
            ExprKind::Field(o, f) if f == "time" && matches!(&o.kind, ExprKind::Var(n) if n == "screen") => Some("f32"),
            ExprKind::Cast(_, ty) => match host_cast_type(ty) {
                Some("half::f16") => Some("f16"),
                Some(t @ ("f32" | "f64")) => Some(t),
                _ => None,
            },
            ExprKind::Call(c, a) if a.len() == 1 && matches!(&c.kind, ExprKind::Var(n) if matches!(n.as_str(), "float16" | "f16")) => Some("f16"),
            ExprKind::Call(c, a) if a.len() == 1 && matches!(&c.kind, ExprKind::Var(n) if matches!(n.as_str(), "float32" | "float" | "float64")) => Some("f32"),
            ExprKind::Var(n) => {
                let ty = self.frame_local_types.borrow().get(n).cloned()
                    .or_else(|| self.setup_fields.iter().find(|(f, _)| f == n).map(|(_, t)| Type::Named(t.clone())));
                match ty {
                    Some(Type::Float16) => Some("f16"),
                    Some(Type::Float32) => Some("f32"),
                    Some(Type::Float64) => Some("f64"),
                    Some(Type::Named(t)) => match t.as_str() {
                        "f16" | "float16" => Some("f16"),
                        "f32" | "float32" => Some("f32"),
                        "f64" | "float" | "float64" => Some("f64"),
                        _ => None,
                    },
                    _ => None,
                }
            }
            ExprKind::UnaryOp(UnaryOp::Neg, i) => self.host_float_kind(i),
            ExprKind::BinOp(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem, l, r) => {
                match (self.host_float_kind(l), self.host_float_kind(r)) {
                    (Some("f16"), Some("f16")) | (Some("f16"), None) | (None, Some("f16")) => Some("f16"),
                    (Some("f32"), Some("f32")) | (Some("f32"), None) | (None, Some("f32")) => Some("f32"),
                    (Some("f16"), Some("f32")) | (Some("f32"), Some("f16")) => Some("f32"),
                    (Some(_), Some(_)) => Some("f64"),
                    (Some("f64"), None) | (None, Some("f64")) => Some("f64"),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// The arguments of a call to a generated function/method, each passed the way its signature takes it.
    fn call_args(&self, args: &[Arg], modes: &[&'static str]) -> Vec<String> {
        args.iter().zip(modes).map(|(a, mode)| {
            let v = self.host_expr(&a.value);
            let is_field = matches!(&a.value.kind, ExprKind::Var(n)
                if self.setup_fields.iter().any(|(f, _)| f == n) || self.event_vars.borrow().contains(n));
            match *mode {
                "&mut" => format!("&mut {v}"),
                "&" => format!("&{v}"),
                // a bare trait parameter the body consumes (`Box<dyn Trait>`)
                "box" if is_field => format!("Box::new({v}.clone())"),
                "box" => format!("Box::new({v})"),
                // host floats are f32 (`screen.time`) or f64 (`as float`): follow the parameter
                "f64" => format!("(({v}) as f64)"),
                "f32" => format!("(({v}) as f32)"),
                _ if is_field => format!("{v}.clone()"),
                _ => v,
            }
        }).collect()
    }

    /// The parameter modes of method `method` of the struct value `recv` (a setup value of a
    /// struct type), read back from the generated `impl`; `None` when it is not such a call.
    fn setup_method_modes(&self, recv: &Expr, method: &str) -> Option<Vec<&'static str>> {
        let ExprKind::Var(n) = &recv.kind else { return None };
        let (_, ty) = self.setup_fields.iter().find(|(f, _)| f == n)?;
        let ty = ty.trim_start_matches('&');
        let mut in_impl = false;
        for l in self.general_code.lines() {
            if l.trim().is_empty() { continue; }
            if !l.starts_with(char::is_whitespace) {
                let t = l.trim_end();
                in_impl = t.starts_with("impl ") && t.trim_end_matches(" {").split_whitespace().last() == Some(ty)
                    || t.starts_with("impl<") && t.trim_end_matches(" {").ends_with(&format!(" {ty}"));
                continue;
            }
            if !in_impl { continue; }
            let t = l.trim_start();
            let t = t.strip_prefix("pub ").unwrap_or(t);
            let needle = format!("fn {method}(");
            if let Some(rest) = t.strip_prefix(&needle) {
                let mut modes = Self::param_modes(rest)?;
                if modes.is_empty() { return None; }
                modes.remove(0); // `self`
                return Some(modes);
            }
        }
        None
    }

    /// How the general pass's `fn name(...)` takes each parameter: `"&mut"`, `"&"` or `""` (by
    /// value), read back from the generated signature. `None` when `name` isn't one of its functions.
    /// Boring type of a host-side argument expression, where it can be told.
    fn host_arg_type(&self, e: &Expr) -> Option<Type> {
        if let ExprKind::Var(n) = &e.kind {
            if self.event_vars.borrow().contains(n) { return Some(Type::Str); }
            if let Some(t) = self.frame_local_types.borrow().get(n) { return Some(t.clone()); }
            if let Some((_, t)) = self.setup_fields.iter().find(|(f, _)| f == n) {
                return Some(match t.as_str() { "String" | "Arc<str>" | "std::sync::Arc<str>" => Type::Str, _ => Type::Named(t.clone()) });
            }
            return None;
        }
        let types = self.frame_local_types.borrow();
        crate::transpiler::helpers::infer_overload_expr_type(e, &types, &Default::default(), &Default::default())
    }

    /// Free-function overload candidates declared under `name` (empty unless overloaded).
    fn free_overloads(&self, name: &str) -> Vec<&FnDecl> {
        let v: Vec<&FnDecl> = self.program.items.iter().filter_map(|i| match i {
            Item::Fn(f) if f.name == name => Some(f), _ => None }).collect();
        if v.len() > 1 { v } else { Vec::new() }
    }

    fn is_general_fn(&self, name: &str) -> bool {
        self.general_fn_param_modes(name).is_some() || !self.free_overloads(name).is_empty()
    }

    /// The mangled Rust name of the overload of `name` that `args` select (`name` itself when not
    /// overloaded). An argument of unknown type matches any candidate; the first match wins.
    fn resolve_free_overload(&self, name: &str, args: &[Arg]) -> String {
        let decls = self.free_overloads(name);
        let chosen = decls.iter().find(|d| d.params.len() == args.len()
            && d.params.iter().zip(args).all(|(p, a)| match (&p.ty, self.host_arg_type(&a.value)) {
                (Some(exp), Some(act)) => crate::transpiler::helpers::types_compatible(exp, &act),
                _ => true,
            }));
        match chosen.or(decls.first()) {
            Some(d) => crate::transpiler::helpers::mangle_overload_name(name, &d.params),
            None => name.to_string(),
        }
    }

    fn general_fn_param_modes(&self, name: &str) -> Option<Vec<&'static str>> {
        let needle = format!("fn {name}(");
        let at = self.general_code.lines().find_map(|l| {
            // Column 0 only: an indented `fn name(` is a method of some impl (`fn release(&mut self, ..)`)
            // that merely shares the name, not the free function.
            if l.starts_with(char::is_whitespace) { return None; }
            let t = l.strip_prefix("pub ").unwrap_or(l);
            if t.starts_with(&needle) { Some(t.to_string()) } else { None }
        })?;
        Self::param_modes(&at[needle.len()..])
    }

    /// `rest` is a signature after `fn name(`: its parameters' modes.
    fn param_modes(rest: &str) -> Option<Vec<&'static str>> {
        let mut depth = 0i32;
        let mut end = None;
        for (i, c) in rest.char_indices() {
            match c {
                '(' | '<' | '[' => depth += 1,
                '>' | ']' => depth -= 1,
                ')' if depth == 0 => { end = Some(i); break; }
                ')' => depth -= 1,
                _ => {}
            }
        }
        let params = &rest[..end?];
        let mut out = Vec::new();
        let mut depth = 0i32;
        let mut cur = String::new();
        let flush = |cur: &mut String, out: &mut Vec<&'static str>| {
            let t = cur.trim();
            if !t.is_empty() {
                let ty = t.split_once(':').map(|(_, ty)| ty.trim()).unwrap_or("");
                out.push(if ty.starts_with("&mut ") { "&mut" } else if ty.starts_with('&') { "&" }
                    else if ty.starts_with("Box<dyn ") { "box" } else if ty == "f64" { "f64" } else if ty == "f32" { "f32" } else { "" });
            }
            cur.clear();
        };
        for c in params.chars() {
            match c {
                '(' | '<' | '[' => { depth += 1; cur.push(c); }
                ')' | '>' | ']' => { depth -= 1; cur.push(c); }
                ',' if depth == 0 => flush(&mut cur, &mut out),
                _ => cur.push(c),
            }
        }
        flush(&mut cur, &mut out);
        Some(out)
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
/// Element type and length of a fixed-length `'const` array: `[T, N]` or a labeled array whose axes
/// are all literal-sized (`[T, width = 4, height = 4]`, stored flat, row-major).
fn const_array_view(ty: &Type) -> Option<(Type, usize)> {
    match ty {
        Type::ArrayN(inner, n) => Some(((**inner).clone(), *n)),
        ty if ty.as_labeled_array().is_some() => {
            let (inner, _) = ty.as_labeled_array()?;
            Some((inner.clone(), ty.labeled_array_len()? as usize))
        }
        _ => None,
    }
}

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

/// True for a buffer field element type this backend represents on the device side as
/// a packed `array<u32>` (`uint8`/`int8`/`uint16`/`int16` — see
/// `super::device::PackedByteKind`'s doc comment). The host-facing `Vec<u8>`/`Vec<i8>`/
/// `Vec<u16>`/`Vec<i16>` type (`host_scalar_type`) is unaffected — only the wgpu
/// buffer's *allocated byte size* needs to be a whole number of 4-byte words to match
/// the device-side `array<u32>` declaration, which is what every call site of this
/// function rounds via `round_up_to_word_bytes`.
fn is_packed_byte_field(inner_ty: &Type) -> bool {
    super::device::packed_byte_kind_of_ty(inner_ty).is_some()
}

/// Host code for the ping-pong swap `k.a, k.b = k.b, k.a` of two buffer fields of one
/// kernel instance: exchanges the device buffer handles (and the element counts of packed
/// byte kinds) and rebuilds the bind group. The `Vec` mirrors are unused placeholders, and
/// `copy_<f>_to_host()` is an rvalue, so the generic tuple assignment cannot express this.
/// `None` unless both fields are buffer fields of the same element type.
pub(crate) fn kernel_field_swap(decl: &KernelDecl, kvar: &str, fa: &str, fb: &str) -> Option<String> {
    let find = |n: &str| decl.fields.iter().find(|f| f.name == n && is_buffer_array_ty(&f.ty)
        && matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal | GpuQual::ActorUnified));
    let (a, b) = (find(fa)?, find(fb)?);
    if array_inner(&a.ty) != array_inner(&b.ty) { return None; }
    let mut out = format!("std::mem::swap(&mut {kvar}.{fa}_buf, &mut {kvar}.{fb}_buf); ");
    if is_packed_byte_field(&array_inner(&a.ty)) {
        out.push_str(&format!("std::mem::swap(&mut {kvar}.{fa}_len, &mut {kvar}.{fb}_len); "));
    }
    out.push_str(&format!("{kvar}.rebuild_bind_group();"));
    Some(format!("{{ {out} }}"))
}

/// Rounds a byte-length Rust expression (a `usize`-typed source string) up to the next
/// multiple of 4 and casts to `u64` — the buffer-size unit `wgpu::BufferDescriptor`
/// wants. Needed only for packed-byte-kind fields (`is_packed_byte_field`): every other
/// element type is already 4 bytes wide (`i32`/`u32`/`f32`), so `n * size_of::<T>()` is
/// automatically a multiple of 4 and this rounding is a no-op there — this helper is
/// simply never called for those. Without it, a buffer sized to e.g. 34 raw bytes
/// (GGUF's Q8_0 block size) would violate `wgpu::COPY_BUFFER_ALIGNMENT` (4 bytes) the
/// first time it's used as a `copy_buffer_to_buffer` destination/source, and also
/// wouldn't hold a whole number of the `array<u32>` words the device side indexes by.
fn round_up_to_word_bytes(byte_len_expr: &str) -> String {
    format!("(((({}) + 3) / 4) * 4) as u64", byte_len_expr)
}

/// The Rust numeric type a host-side `expr as <ty>` converts to (`float` is `f64`, as in the general pass).
fn host_cast_type(ty: &Type) -> Option<&'static str> {
    Some(match ty {
        Type::Int => "isize", Type::Uint => "usize",
        Type::Float16 => "half::f16", Type::Float32 => "f32", Type::Float64 => "f64",
        Type::Int8 => "i8", Type::Int16 => "i16", Type::Int32 => "i32", Type::Int64 => "i64",
        Type::Uint8 => "u8", Type::Uint16 => "u16", Type::Uint32 => "u32", Type::Uint64 => "u64",
        Type::Named(n) => match n.as_str() {
            "float" | "float64" => "f64", "float16" | "f16" => "half::f16", "float32" => "f32", "int" => "isize", "uint" => "usize",
            _ => return None,
        },
        _ => return None,
    })
}

/// Host type of a scalar slot in the `{Kernel}Params` struct. Every slot is exactly
/// 32 bits, matching the WGSL params struct (`wgsl_scalar`: sub-32-bit and `bool`
/// fields fall back to `i32`/`u32`/`f32` there): `bytemuck::Pod` forbids both
/// implicit padding (a lone `u8` next to 4-byte fields) and `bool`.
fn params_slot_type(ty: &Type) -> &'static str {
    // A fixed-point scalar is uploaded as its raw 32-bit integer (`i32` signed, `u32` unsigned;
    // 8/16-bit raws are widened, the device re-narrows them).
    if let Some(f) = crate::transpiler::emit_kernel::fixed_kernel_fmt(ty) {
        return if f.signed { "i32" } else { "u32" };
    }
    let t = host_scalar_type(ty);
    match t {
        "i8" | "i16" | "i64" | "i128" => "i32",
        "u8" | "u16" | "u64" | "u128" | "bool" => "u32",
        "half::f16" => "f32",
        other => other,
    }
}

/// Expression converting host value `expr` (of `host_scalar_type(ty)`) to its params slot type.
fn params_slot_conv(ty: &Type, expr: &str) -> String {
    if crate::transpiler::emit_kernel::fixed_kernel_fmt(ty).is_some() {
        return format!("{}.0 as {}", expr, params_slot_type(ty));
    }
    let t = host_scalar_type(ty);
    let slot = params_slot_type(ty);
    if t == slot {
        expr.to_string()
    } else if t == "half::f16" {
        format!("{}.to_f32()", expr)
    } else {
        format!("{} as {}", expr, slot)
    }
}

fn host_scalar_type(ty: &Type) -> &'static str {
    // Fixed-point: the generated `repr(transparent)` newtype (`Fixed16_16`), bit-identical to
    // the device's raw-integer buffer elements.
    if let Some(f) = crate::transpiler::emit_kernel::fixed_kernel_fmt(ty) {
        return crate::transpiler::emit_kernel::interned_fixed_name(&f);
    }
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
        Type::Float16 => "half::f16",
        Type::Float32 => "f32",
        Type::Bool  => "bool",
        // Explicit fixed-width fields keep their own exact width in the host-side buffer —
        // WGSL itself can't represent 8/16/64/128-bit ints (see `wgsl_scalar` in device.rs,
        // which emits a clear compile error for those widths); the host Rust type still
        // reflects the declared width so the mismatch is visible on the device side, not
        // silently mis-typed here too (the previous behavior for `Uint8`, which fell to `i64`).
        //
        // `float`/`float64` is NOT one of those explicit fixed-width opt-ins, though — it's
        // Boring's *default* float type, the direct analogue of `int`/`uint` above, which
        // this same function DOES narrow to match the device layout. A prior revision kept
        // this at `"f64"`, reasoning that `wgsl_unsupported_f64` (device.rs) "now raises" a
        // real compile error for `float64` on this target, making the host/device mismatch
        // moot. It doesn't: `wgsl_unsupported_f64` only ever emits a `/* ERROR: ... */`
        // *comment* ahead of a still-emitted, still-narrowed `f32` in the generated WGSL —
        // naga accepts the comment and compiles the shader at `f32` regardless, and nothing
        // scans `device_wgsl` for that marker to turn it into a real `TranspileError` (see
        // `wgpu::transpile_wgpu`). So every `float`/`float64` kernel field's *device* buffer
        // is always `f32` in practice, while this function kept telling the *host* side
        // (`copy_{field}_to_device`'s parameter type, this struct's own field type, the
        // `Params` struct packed into the uniform buffer) it was `f64` — a real buffer
        // layout mismatch, not just a naming one: `bytemuck::cast_slice` on an `&[f64]`
        // produces twice the byte width the `f32` shader elements expect. Confirmed via
        // `linguist/samples/gpu.br --target wgpu` + `cargo build`: every kernel using a bare
        // `float` field (Saxpy's `alpha`, Blur's `weights`/`sigma`, not just a
        // const-generic kernel) failed with `error[E0308]: expected &[f64], found &Vec<f32>`
        // at its `copy_*_to_device` call site (`emit_kernel.rs`'s construction codegen
        // already casts to `f32` there, matching the real device width). Narrowed back to
        // `f32` to match what the device buffer actually is; the stale-but-harmless comment
        // in the generated WGSL is left as-is (a real hard error belongs in the checker, a
        // separate, larger gap this narrowing fix doesn't attempt — see
        // docs/float-width-types.md §6's own "one real, pre-existing gap remains" note
        // about the *different* `kernel_host_scalar_type`/`kernel_host_element_type` pair in
        // `emit_kernel.rs`, which already narrows here correctly and was never the bug).
        Type::Float64 => "f32",
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
            "float16" | "f16" => "half::f16",
            "float32" | "f32" => "f32",
            "float" | "float64" | "f64" => "f32",
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
    // The runtime normalizer (screen_input.rs) maps every accepted spelling to the canonical name.
    format!("&__boring_key_name({s:?})")
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

/// iOS: winit 0.30 creates its `UIWindow` without a scene, but an app linked against a recent SDK must use the UIScene life
/// cycle (iOS refuses it otherwise: "UIScene life cycle is required"). The Info.plist names a scene delegate class
/// (`UIApplicationSceneManifest` / `UISceneDelegateClassName` = `BoringSceneDelegate`); this registers it before the event loop
/// starts and attaches winit's window to the scene UIKit connects, whichever of the two comes first. UIKit also requires the
/// class to declare the scene protocols. Without a manifest no scene ever connects and nothing happens.
const IOS_SCENE_RS: &str = r#"#[cfg(target_os = "ios")]
mod __boring_scene {
    use objc2::runtime::{AnyObject, AnyProtocol, ClassBuilder, NSObject, Sel};
    use objc2::{msg_send, sel, ClassType};
    use std::sync::Mutex;
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    // (scene, window) as raw pointers, set from the main thread only: whichever arrives second attaches the window.
    static STATE: Mutex<(usize, usize)> = Mutex::new((0, 0));

    fn attach() {
        let (scene, window) = *STATE.lock().unwrap();
        if scene == 0 || window == 0 { return; }
        unsafe {
            let (s, w) = (scene as *mut AnyObject, window as *mut AnyObject);
            let _: () = msg_send![w, setWindowScene: s];
            let _: () = msg_send![w, makeKeyAndVisible];
        }
    }

    extern "C" fn will_connect(_this: &AnyObject, _cmd: Sel, scene: &AnyObject, _session: &AnyObject, _options: &AnyObject) {
        STATE.lock().unwrap().0 = scene as *const AnyObject as usize;
        attach();
    }

    pub fn register() {
        let mut class = ClassBuilder::new(c"BoringSceneDelegate", NSObject::class()).expect("BoringSceneDelegate is already registered");
        unsafe { class.add_method(sel!(scene:willConnectToSession:options:), will_connect as extern "C" fn(_, _, _, _, _)); }
        for name in [c"UISceneDelegate", c"UIWindowSceneDelegate"] {
            if let Some(protocol) = AnyProtocol::get(name) { class.add_protocol(protocol); }
        }
        class.register();
    }

    pub fn window_created(window: &winit::window::Window) {
        if let Ok(handle) = window.window_handle() {
            if let RawWindowHandle::UiKit(h) = handle.as_raw() {
                let view = h.ui_view.as_ptr() as *mut AnyObject;
                let ui_window: *mut AnyObject = unsafe { msg_send![view, window] };
                STATE.lock().unwrap().1 = ui_window as usize;
                attach();
            }
        }
    }
}"#;
