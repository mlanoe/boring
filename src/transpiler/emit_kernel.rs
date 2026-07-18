// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Kernel-aware codegen for GPU targets (wgpu/cuda/metal).
//
// The general transpiler treats `Item::Kernel` as a no-op — kernel struct
// definitions and dispatch scaffolding are generated separately by each GPU
// backend (`transpiler::wgpu::host`, etc.). But boring source can construct a
// kernel instance, launch it, and read back its results from *inside an
// ordinary function body* (see whisper-boring's `log_mel_spectrogram_gpu` for
// a real example) — code that also uses regular boring control flow, string
// interpolation, other function calls, and so on.
//
// Rather than write a second, separate mini-transpiler for those functions,
// this module adds three small, narrowly-gated special cases to the *same*
// statement/expression emitters everything else goes through, so a
// kernel-touching function gets full boring-language support "for free":
//   - a `let`/`mut`/`var` whose initializer is `KernelName(args...)` constructs
//     the GPU wrapper struct (`emit_stmt::emit_let` calls `try_emit_kernel_let`)
//   - a `kernel: k(block=.., grid=..)` block dispatches it
//     (`emit_stmt::emit_stmt` calls `try_emit_kernel_dispatch`)
//   - reading a `'unified` field on a tracked kernel variable reads back the
//     GPU buffer (`emit_expr`'s `Field` case calls `try_emit_kernel_field_read`)
//
// All three are gated on `self.kernel_decls` being non-empty, which only a
// GPU target's `TranspileConfig::gpu_kernels` populates — every other target
// (including `boring build` with no target, and `boring run`) is completely
// unaffected.
//
// Scope: this handles the straight-line pattern used by every kernel in
// practice — construct, dispatch, read back — appearing directly in a
// function's own statement list. It does not attempt to handle kernel
// construction/dispatch nested inside loops or conditionals, or an `init`
// body that does anything more than plain `field = param` assignments (the
// convention every kernel in this codebase follows). Those richer cases fall
// through to the general emitter and would need to be extended here if a
// future kernel needs them.

use super::*;
use super::Transpiler;
use crate::ast::{Arg, GpuQual, InitDecl, KernelBlockStmt, KernelDecl, KernelFieldDecl, Stmt, Type};

impl Transpiler {
    /// If `s`'s initializer is a call to a known kernel type, emit the GPU
    /// construction code (device/queue-backed wrapper + host->device buffer
    /// copies / scalar field sets) and return `true`. Otherwise, does nothing
    /// and returns `false` so the caller falls through to the normal `let`
    /// handling.
    pub(crate) fn try_emit_kernel_let(&mut self, s: &LetStmt) -> bool {
        if self.kernel_decls.is_empty() {
            return false;
        }
        let Some(val) = &s.value else { return false };
        let ExprKind::Call(callee, args) = &val.kind else { return false };
        let ExprKind::Var(kname) = &callee.kind else { return false };
        let Some(decl) = self.kernel_decls.get(kname).cloned() else { return false };

        self.emit_kernel_construction(&s.name, &decl, args);
        self.kernel_vars.insert(s.name.clone(), kname.clone());
        true
    }

    fn emit_kernel_construction(&mut self, var_name: &str, decl: &KernelDecl, args: &[Arg]) {
        // Dimension-sized kernels (a `Dimension`-typed 'const field -- e.g. `let Dimension
        // dim`, set via `init(Dimension d): ... dim = d`, see examples/game_of_life.br)
        // are constructed as `Kernel(Dimension(w, h))`. Their real `Kernel::new(...)` (see
        // wgpu::host::emit_kernel_new/has_dim_field) takes flat `width: i32, height: i32`
        // positional args instead of the field-by-field mapping below -- it already
        // zero-allocates every array field to `width * height` elements internally and
        // stores the dimension itself, none of which the generic `field = param` scan
        // below (`kernel_param_to_field_map`) can express. Handle it separately and
        // return before reaching that scan.
        if Self::kernel_has_dim_field(decl) {
            self.emit_dim_kernel_construction(var_name, decl, args);
            return;
        }

        self.line(&format!(
            "let mut {var_name} = {}::new(__boring_gpu_device(), __boring_gpu_queue());",
            decl.name
        ));

        let param_to_field = Self::kernel_param_to_field_map(decl);
        let init_param_names: Vec<&str> = decl.inits.first()
            .map(|i: &InitDecl| i.params.iter().map(|p| p.name.as_str()).collect())
            .unwrap_or_default();
        // param name -> the Rust expression this constructor call passed for it, so a
        // sibling `'unified` output field's size expression (below) can be translated
        // into the same caller-visible values instead of the init param names, which
        // don't exist as Rust bindings at this call site.
        let mut param_to_arg: std::collections::HashMap<&str, String> = std::collections::HashMap::new();

        for (i, arg) in args.iter().enumerate() {
            // Each of these three lookups failing means this constructor argument would
            // otherwise be silently dropped -- the field it should have initialized keeps
            // its zero-initialized default with no error or warning at all. Fail loudly
            // instead: the kernel's `init` needs to follow the plain `field = param`
            // convention this scan understands (see `kernel_param_to_field_map`'s doc).
            let Some(param_name) = init_param_names.get(i).copied() else {
                panic!(
                    "kernel '{}': constructor call passes {} argument(s), but its `init` only declares {} parameter(s) -- argument #{} would be silently dropped",
                    decl.name, args.len(), init_param_names.len(), i + 1
                );
            };
            let Some(field_name) = param_to_field.get(param_name) else {
                panic!(
                    "kernel '{}': `init` parameter '{}' is never assigned to a field via a plain `field = {}` statement -- only that pattern is supported for kernel constructor codegen, so this argument would be silently dropped",
                    decl.name, param_name, param_name
                );
            };
            let Some(field) = decl.fields.iter().find(|f| &f.name == field_name) else {
                panic!(
                    "kernel '{}': `init` assigns parameter '{}' to '{}', which is not a declared field of this kernel",
                    decl.name, param_name, field_name
                );
            };

            let arg_rust = self.emit_expr(&arg.value);
            param_to_arg.insert(param_name, arg_rust.clone());
            let is_buffer = matches!(field.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal)
                && matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _));
            if is_buffer {
                let inner = kernel_host_scalar_type(&array_inner_type(&field.ty));
                self.line(&format!(
                    "{var_name}.copy_{field_name}_to_device(&{arg_rust}.iter().map(|&x| x as {inner}).collect::<Vec<{inner}>>());"
                ));
            } else {
                let cast = kernel_host_scalar_type(&field.ty);
                self.line(&format!("{var_name}.{field_name} = ({arg_rust}) as {cast};"));
            }
        }

        // `'unified`/`'global` array fields the loop above never touched are outputs
        // allocated in `init()` via `field = [value for ..count]` (e.g. `mel = [0.0 for
        // ..n_mels * n_frames]`), not fed by a constructor argument. `Kernel::new()`
        // creates every buffer at size 0 (see wgpu::host::emit_kernel_new) because it
        // has no host-side notion of this fill expression — without this, the buffer
        // stays 0 bytes forever and the bind group (fixed at construction) permanently
        // references a too-small buffer, failing GPU validation the first time the
        // kernel actually runs with a real (non-empty) size. Zero-fill it here, through
        // the same copy_{field}_to_device the loop above already uses for inputs, which
        // resizes the buffer to match and rebuilds the bind group (see wgpu::host).
        for (field_name, (value, count)) in Self::kernel_output_fill_map(decl) {
            if param_to_field.values().any(|f| f == &field_name) { continue; }
            let Some(field) = decl.fields.iter().find(|f| f.name == field_name) else { continue };
            let inner = kernel_host_scalar_type(&array_inner_type(&field.ty));
            let value_rust = self.substitute_and_emit(&value, &param_to_arg);
            let count_rust = self.substitute_and_emit(&count, &param_to_arg);
            self.line(&format!(
                "{var_name}.copy_{field_name}_to_device(&vec![({value_rust}) as {inner}; ({count_rust}) as usize]);"
            ));
        }
    }

    /// Scan a kernel's (first) `init` body for `field = [value for ..count]`
    /// assignments (`ExprKind::ArrayFill`) — the convention this codebase's kernels use
    /// to zero-allocate a `'unified` output buffer to its runtime size. Returns
    /// `field name -> (value expr, count expr)`.
    fn kernel_output_fill_map(decl: &KernelDecl) -> std::collections::HashMap<String, (Expr, Expr)> {
        let mut map = std::collections::HashMap::new();
        if let Some(init) = decl.inits.first() {
            for stmt in &init.body {
                if let Stmt::Expr(e) = stmt {
                    if let ExprKind::Assign(lhs, rhs) = &e.kind {
                        if let ExprKind::Var(field) = &lhs.kind {
                            if let ExprKind::ArrayFill { value, count } = &rhs.kind {
                                map.insert(field.clone(), ((**value).clone(), (**count).clone()));
                            }
                        }
                    }
                }
            }
        }
        map
    }

    /// Translate a boring `Expr` (an init-body fill/count expression, referencing only
    /// `init()`'s own parameter names) into a Rust expression string, substituting each
    /// parameter name for the Rust expression the constructor call actually passed for
    /// it (`subst`). Handles the small arithmetic subset these expressions use in
    /// practice (bare param references, integer/float literals, `+ - * /`); anything
    /// richer falls back to the general emitter (which won't have the substitution
    /// applied, but is a safe default since these expressions are deliberately simple by
    /// convention — see `kernel_output_fill_map`'s doc).
    fn substitute_and_emit(&self, expr: &Expr, subst: &std::collections::HashMap<&str, String>) -> String {
        match &expr.kind {
            ExprKind::Var(name) => subst.get(name.as_str()).cloned().unwrap_or_else(|| name.clone()),
            ExprKind::Int(n) => n.to_string(),
            ExprKind::Float(f) => f.to_string(),
            ExprKind::BinOp(op, l, r) => {
                let l_s = self.substitute_and_emit(l, subst);
                let r_s = self.substitute_and_emit(r, subst);
                format!("({} {} {})", l_s, crate::transpiler::helpers::binop_str(op), r_s)
            }
            _ => self.emit_expr(expr),
        }
    }

    /// True when `decl` declares a `Dimension`-typed `'const` field (inferred for a
    /// bare `let Dimension name`, per the parser's binding-based qualifier rule) --
    /// matches `wgpu::host`'s own `has_dim_field` check, which this must stay
    /// consistent with since it's replicating that function's constructor signature.
    fn kernel_has_dim_field(decl: &KernelDecl) -> bool {
        decl.fields.iter().any(|f| {
            matches!(&f.ty, Type::Named(n) if n == "Dimension") && matches!(f.qual, GpuQual::Const)
        })
    }

    /// Construct a Dimension-sized kernel: `Kernel(Dimension(w, h))` → `Kernel::new((w)
    /// as i32, (h) as i32, device, queue)`, matching the flat `width, height` params
    /// `wgpu::host::emit_kernel_new` generates for a kernel with a Dimension field.
    fn emit_dim_kernel_construction(&mut self, var_name: &str, decl: &KernelDecl, args: &[Arg]) {
        let Some(dim_arg) = args.first() else {
            panic!(
                "kernel '{}' has a Dimension field but its constructor call passes no arguments -- expected `{}(Dimension(w, h))`",
                decl.name, decl.name
            );
        };
        let ExprKind::Call(dim_callee, dim_args) = &dim_arg.value.kind else {
            panic!(
                "kernel '{}': expected a `Dimension(w, h)` constructor argument (this kernel has a Dimension field), got a different expression",
                decl.name
            );
        };
        if !matches!(&dim_callee.kind, ExprKind::Var(n) if n == "Dimension") {
            panic!(
                "kernel '{}': expected a `Dimension(w, h)` constructor argument (this kernel has a Dimension field), got a different call",
                decl.name
            );
        }
        if args.len() > 1 {
            panic!(
                "kernel '{}': constructor call passes {} argument(s), but only the leading `Dimension(w, h)` is supported for a kernel with a Dimension field -- the rest would be silently dropped",
                decl.name, args.len()
            );
        }
        let w = dim_args.first().map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "0".into());
        let h = dim_args.get(1).map(|a| self.emit_expr(&a.value)).unwrap_or_else(|| "0".into());
        self.line(&format!(
            "let mut {var_name} = {}::new(({w}) as i32, ({h}) as i32, __boring_gpu_device(), __boring_gpu_queue());",
            decl.name
        ));
    }

    /// Scan a kernel's (first) `init` body for `field = param` assignments,
    /// returning a `param name -> field name` map. Every kernel in this
    /// codebase's convention assigns each init param straight to a field (with
    /// any remaining fields, e.g. `'unified` outputs, zero-initialized
    /// separately) — richer init bodies aren't recognized here.
    fn kernel_param_to_field_map(decl: &KernelDecl) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        if let Some(init) = decl.inits.first() {
            for stmt in &init.body {
                if let Stmt::Expr(e) = stmt {
                    if let ExprKind::Assign(lhs, rhs) = &e.kind {
                        if let (ExprKind::Var(field), ExprKind::Var(param)) = (&lhs.kind, &rhs.kind) {
                            map.insert(param.clone(), field.clone());
                        }
                    }
                }
            }
        }
        map
    }

    /// If `block` is a single `k(block=.., grid=..)` dispatch call on a
    /// tracked kernel variable, emit `k.dispatch(gx, gy, gz);` and return it.
    /// Otherwise returns `None` (caller falls through to naive passthrough).
    pub(crate) fn try_emit_kernel_dispatch(&mut self, block: &KernelBlockStmt) -> Option<String> {
        if self.kernel_decls.is_empty() {
            return None;
        }
        let [Stmt::Expr(e)] = block.body.as_slice() else { return None };
        let ExprKind::Call(callee, args) = &e.kind else { return None };
        let ExprKind::Var(var_name) = &callee.kind else { return None };
        if !self.kernel_vars.contains_key(var_name.as_str()) {
            return None;
        }

        let (gx, gy, gz) = if let Some(g) = args.iter().find(|a| a.label.as_deref() == Some("grid")) {
            match &g.value.kind {
                ExprKind::Tuple(elems) => {
                    let get = |i: usize| elems.get(i).map(|e| self.emit_expr(e)).unwrap_or_else(|| "1".into());
                    (get(0), get(1), get(2))
                }
                _ => (self.emit_expr(&g.value), "1".to_string(), "1".to_string()),
            }
        } else {
            ("1".to_string(), "1".to_string(), "1".to_string())
        };
        Some(format!("{var_name}.dispatch(({gx}) as u32, ({gy}) as u32, ({gz}) as u32);"))
    }

    /// If `obj.field` reads a `'unified`/`'global` array field on a tracked
    /// kernel variable, emit the GPU read-back call (converted back to the
    /// host's f64/i64 element type) instead of a plain field access.
    pub(crate) fn try_emit_kernel_field_read(&self, obj: &Expr, field: &str) -> Option<String> {
        if self.kernel_decls.is_empty() {
            return None;
        }
        // `copy_{field}_to_host()` returns an owned Vec -- an rvalue, not an lvalue.
        // Assigning through it (`k.output[i] = v`, `k.output[i] /= x`) would emit Rust
        // that tries to mutate a temporary. Fall through to the plain field access
        // instead, matching the analogous guard for array-index assignment above.
        if self.in_lhs_assign.get() {
            return None;
        }
        let ExprKind::Var(var_name) = &obj.kind else { return None };
        let kname = self.kernel_vars.get(var_name.as_str())?;
        let decl = self.kernel_decls.get(kname)?;
        let field_decl: &KernelFieldDecl = decl.fields.iter().find(|f| f.name == field)?;
        if !matches!(field_decl.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal)
            || !matches!(field_decl.ty, Type::Array(_) | Type::ArrayN(_, _))
        {
            return None;
        }
        let host_ty = kernel_host_element_type(&array_inner_type(&field_decl.ty));
        Some(format!(
            "{var_name}.copy_{field}_to_host().iter().map(|&x| x as {host_ty}).collect::<Vec<{host_ty}>>()"
        ))
    }
}

fn array_inner_type(ty: &Type) -> Type {
    match ty {
        Type::Array(inner) | Type::ArrayN(inner, _) => (**inner).clone(),
        other => other.clone(),
    }
}

/// GPU-buffer element type for a boring scalar type (matches
/// `wgpu::host::host_scalar_type` — GPU buffers always use 32-bit elements).
fn kernel_host_scalar_type(ty: &Type) -> &'static str {
    match ty {
        Type::Int   => "i32",
        Type::Uint  => "u32",
        Type::Float => "f32",
        Type::Bool  => "bool",
        Type::Named(n) => match n.as_str() {
            "int"                    => "i32",
            "uint"                   => "u32",
            "float"                  => "f32",
            "bool"                   => "bool",
            _                        => "i64",
        },
        Type::Qualified(inner, _) => kernel_host_scalar_type(inner),
        _ => "i64",
    }
}

/// Host-side (non-GPU) element type a value read back from a GPU buffer
/// should be converted to — boring `int`/`float` are `i64`/`f64` everywhere
/// outside a kernel buffer.
fn kernel_host_element_type(ty: &Type) -> &'static str {
    match ty {
        Type::Int | Type::Uint => "i64",
        Type::Float => "f64",
        Type::Bool => "bool",
        Type::Named(n) => match n.as_str() {
            "int" | "uint" => "i64",
            "float"        => "f64",
            "bool"         => "bool",
            _              => "i64",
        },
        Type::Qualified(inner, _) => kernel_host_element_type(inner),
        _ => "i64",
    }
}
