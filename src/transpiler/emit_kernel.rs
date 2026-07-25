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
    /// If `s` declares a name initialized directly from a bare kernel-field read
    /// (`let py'gpu'unified = k.y`, or, with the qualifier inferred, plain
    /// `let py = k.y`), registers it as a resident alias (`gpu_resident_vars`) and
    /// emits nothing at all — no Rust binding exists for `py`; its only legal use is
    /// as the subject of a `with` block (`emit_stmt::emit_with` resolves it back to
    /// `k.copy_y_to_host`/`copy_y_to_device`), which is also all the checker allows
    /// (`Binding::resident_from_field`, `Checker::infer_gpu_resident` — same
    /// inference, mirrored here). Returns `true` when handled so the caller skips
    /// ordinary `let` codegen.
    ///
    /// Otherwise returns `false` — this also covers a `'gpu'unified`/`'gpu'global`
    /// array *literal* (`examples/saxpy.br`'s `var [float]'gpu'unified x = [0.0 for
    /// ..N]`), which is just a plain host array today (freely indexed/assigned, no
    /// `with` required) and falls through to ordinary `let` codegen unchanged.
    pub(crate) fn try_emit_gpu_resident_let(&mut self, s: &LetStmt) -> bool {
        if self.kernel_decls.is_empty() {
            return false;
        }
        // An explicit annotation of anything OTHER than 'gpu'unified/'gpu'global
        // means this isn't our concern (e.g. `let [float] n = k.alpha`, reading a
        // kernel field into a differently-qualified/plain binding on purpose).
        let has_explicit_qual = match &s.ty {
            Some(ty) => ty.gpu_resident_qual().is_some(),
            None => false,
        };
        if s.ty.is_some() && !has_explicit_qual {
            return false;
        }
        let Some(val) = &s.value else { return false };
        let ExprKind::Field(obj, field) = &val.kind else { return false };
        let ExprKind::Var(kvar) = &obj.kind else { return false };
        let Some(kname) = self.kernel_vars.get(kvar.as_str()) else { return false };
        if !has_explicit_qual {
            // No annotation at all — only infer residency when the kernel's own
            // field declaration actually says `'unified`/`'global` on an array (the
            // same gate `try_emit_kernel_field_read` uses below); an untyped read of
            // a scalar or other-qualified field is just an ordinary field access.
            let is_gpu_array_field = self.kernel_decls.get(kname)
                .and_then(|decl| decl.fields.iter().find(|f| &f.name == field))
                .map(|f| matches!(f.qual, GpuQual::Unified | GpuQual::Global)
                    && matches!(f.ty, Type::Array(_) | Type::ArrayN(_, _)))
                .unwrap_or(false);
            if !is_gpu_array_field { return false; }
        }
        self.gpu_resident_vars.insert(s.name.clone(), (kvar.clone(), field.clone()));
        true
    }

    /// If `s` declares a name initialized directly from a call to a function whose
    /// own declared return type is GPU-resident (`self.fn_returns_resident`) — `let
    /// fc = linear_gpu(...)`, explicit `'gpu'unified`/`'gpu'global` annotation or
    /// inferred — this is the *interprocedural* counterpart to
    /// `try_emit_gpu_resident_let` above. Unlike that same-scope alias (no Rust
    /// binding at all, since the data is just a re-readable kernel field), this call
    /// already executed with real side effects, so `fc` needs a genuine `let`
    /// binding — just typed `BoringGpuArg<T>` (via the callee's own return-type
    /// codegen, see `emit_top.rs`) instead of a plain host array. Registers `fc` in
    /// `resident_call_vars` so `with fc:` (see `emit_stmt::emit_with`) and further
    /// chained calls (see `emit_call`) know to treat it residently. Returns `true`
    /// when handled.
    ///
    /// A caller that does *not* opt in — an explicit, ordinary (non-resident) type
    /// annotation, e.g. `let [float] fc = linear_gpu(...)`, exactly how every call
    /// site written before `linear_gpu` grew a resident return type still reads —
    /// is intercepted here too, not left to fall through: the callee's Rust
    /// signature returns `BoringGpuArg<T>` unconditionally now, regardless of what
    /// this particular call site wants, so falling through to ordinary `let`
    /// codegen would bind that enum to a `Vec<T>`-declared binding, a hard type
    /// mismatch (confirmed against a real `cargo check` — this is not
    /// hypothetical). Opting a function's return type into residency must stay
    /// purely additive for every existing, unannotated caller: it materializes,
    /// once, right here — exactly the eager download this call site already paid
    /// before the function had a resident return type at all, just now expressed
    /// as a match instead of a plain field read.
    pub(crate) fn try_emit_gpu_resident_call_let(&mut self, s: &LetStmt) -> bool {
        if self.fn_returns_resident.is_empty() {
            return false;
        }
        let Some(val) = &s.value else { return false };
        let ExprKind::Call(callee, _) = &val.kind else { return false };
        let ExprKind::Var(fn_name) = &callee.kind else { return false };
        let Some(ret_ty) = self.fn_returns_resident.get(fn_name.as_str()).cloned() else { return false };

        let has_explicit_qual = match &s.ty {
            Some(ty) => ty.gpu_resident_qual().is_some(),
            None => false,
        };
        if s.ty.is_some() && !has_explicit_qual {
            let materialized = self.materialize_resident_call(val, &ret_ty);
            let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
            self.line(&format!("{kw} {name} = {materialized};", kw = kw, name = s.name));
            self.var_types.insert(s.name.clone(), s.ty.clone().expect("checked by the outer `if`"));
            self.known_local_vars.insert(s.name.clone());
            return true;
        }

        let call_rust = self.emit_expr(val);
        let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
        self.line(&format!("{} {} = {};", kw, s.name, call_rust));
        self.resident_call_vars.insert(s.name.clone(), ret_ty);
        self.known_local_vars.insert(s.name.clone());
        true
    }

    /// Emits a materializing match over `call_expr` (a call to a function whose
    /// return type is `ret_ty`, itself GPU-resident) that downloads a `Resident`
    /// buffer or passes a `Host` vec straight through — the shared shape behind
    /// every "caller doesn't opt in" fallback (`try_emit_gpu_resident_call_let`
    /// above, and the tail-expression/plain-assignment call sites in
    /// `emit_stmt.rs`/`emit_expr.rs`). Same pattern `emit_with`'s
    /// `resident_call_vars` branch already uses to unwrap a `BoringGpuArg<T>` back
    /// to a plain host `Vec<T>`, just built from a fresh call instead of a
    /// pre-bound variable — no `.clone()` needed on the `Host` arm since we own
    /// the enum value outright here.
    pub(crate) fn materialize_resident_call(&self, call_expr: &Expr, ret_ty: &Type) -> String {
        let call_rust = self.emit_expr(call_expr);
        let inner_ty = match ret_ty {
            Type::Qualified(inner, _) => array_inner_type(inner),
            other => array_inner_type(other),
        };
        let host_ty = kernel_host_element_type(&inner_ty);
        let device_ty = kernel_host_scalar_type(&inner_ty);
        format!(
            "match {call_rust} {{ \
             BoringGpuArg::Resident(buf, _) => __boring_gpu_copy_d2h::<{device_ty}>(&__boring_gpu_device(), &__boring_gpu_queue(), &buf).iter().map(|&x| x as {host_ty}).collect::<Vec<{host_ty}>>(), \
             BoringGpuArg::Host(v) => v \
             }}"
        )
    }

    /// If `expr` is a bare call to a `fn_returns_resident` function, emits the
    /// same materializing match `materialize_resident_call` does and returns
    /// `Some`. For any consumption context that doesn't opt into keeping the
    /// value resident — a tail expression of a function whose own return type
    /// isn't itself resident, or a plain (re-)assignment — the callee's Rust
    /// signature returns `BoringGpuArg<T>` unconditionally now, so skipping this
    /// check is a real type mismatch, not just a missed optimization (confirmed
    /// against real `cargo check` failures on whisper-boring's own call sites:
    /// `Ok(linear_gpu(normed, ...))` as a plain-`[float]`-returning function's
    /// tail, and `self.cache_ca_k = linear_gpu(...)` as a struct-field
    /// assignment, both broke this way before this check existed).
    pub(crate) fn try_materialize_resident_call(&self, expr: &Expr) -> Option<String> {
        let ExprKind::Call(callee, _) = &expr.kind else { return None };
        let ExprKind::Var(fn_name) = &callee.kind else { return None };
        let ret_ty = self.fn_returns_resident.get(fn_name.as_str()).cloned()?;
        Some(self.materialize_resident_call(expr, &ret_ty))
    }

    /// If `s`'s initializer is `GPU(n)`, registers `s.name` as a GPU-device handle
    /// (`gpu_device_vars`) and emits it as a plain `usize` index. See
    /// `emit_call`'s own `"GPU"` special case (this function only adds the
    /// tracking `try_emit_gpu_device_let` needs on top of that shared emission —
    /// see `gpu_device_vars`'s doc comment for why every index resolves to the
    /// same single real adapter on wgpu) and `emit_methods.rs`'s method-call
    /// rewrite for `.name()`/`.totalMem()`/etc on a tracked variable.
    pub(crate) fn try_emit_gpu_device_let(&mut self, s: &LetStmt) -> bool {
        if !self.is_gpu_target {
            return false;
        }
        let Some(val) = &s.value else { return false };
        let ExprKind::Call(callee, _) = &val.kind else { return false };
        let ExprKind::Var(name) = &callee.kind else { return false };
        if name != "GPU" { return false; }
        // Routes through emit_call's own "GPU" special case (emits a plain usize).
        let expr_s = self.emit_expr(val);
        self.gpu_device_vars.insert(s.name.clone());
        let kw = if s.binding.is_mutable() { "let mut" } else { "let" };
        self.line(&format!("{} {} = {};", kw, s.name, expr_s));
        true
    }

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

    /// Resolves `expr` to a `(source kernel var, source field)` pair when it names a
    /// resident `'unified`/`'global` kernel buffer that can be handed to another
    /// kernel constructor directly (`Arc::clone`), bypassing any host round-trip.
    /// Two shapes recognized, both needing the exact same qualifying check:
    ///   - `ExprKind::Var(name)` — a same-scope alias registered in
    ///     `gpu_resident_vars` (`let fc = k1.y`, no Rust binding of its own).
    ///   - `ExprKind::Field(Var(kvar), field)` — a bare kernel-field read used
    ///     directly as the argument, with no `let` alias at all
    ///     (`Kernel2(k1.y, ...)`) — the shape `attention_heads_gpu` uses internally
    ///     to chain its three kernel stages (`SoftmaxRowsKernel(k_scores.c, ...)`,
    ///     `MatMulHeadsKernel(k_soft.probs, ...)`): the alias case above only ever
    ///     fires when boring source happens to bind the field read to a name first,
    ///     which nothing forces a kernel-internal chain to do.
    ///
    ///     Either way `kvar` must be a tracked kernel instance and `field` must be
    ///     declared `'unified`/`'global` on an array on that kernel — otherwise this is
    ///     just an ordinary value (or a scalar/differently-qualified field) that the
    ///     normal argument-emission path should keep handling as before.
    fn resident_field_alias(&self, expr: &Expr) -> Option<(String, String)> {
        match &expr.kind {
            ExprKind::Var(name) => self.gpu_resident_vars.get(name.as_str()).cloned(),
            ExprKind::Field(obj, field) => {
                let ExprKind::Var(kvar) = &obj.kind else { return None };
                let kname = self.kernel_vars.get(kvar.as_str())?;
                let decl = self.kernel_decls.get(kname)?;
                let field_decl = decl.fields.iter().find(|f| &f.name == field)?;
                let is_gpu_array_field = matches!(field_decl.qual, GpuQual::Unified | GpuQual::Global)
                    && matches!(field_decl.ty, Type::Array(_) | Type::ArrayN(_, _));
                is_gpu_array_field.then(|| (kvar.clone(), field.clone()))
            }
            _ => None,
        }
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
        // param name -> a ready-made `.length`/`.count` expression (already a full
        // `usize` count, not a value `.len()` can be appended to) for a param consumed
        // via the resident-alias fast path below, which has no Rust array/Vec value at
        // all to call `.len()` on -- only a raw `Arc<wgpu::Buffer>` reachable through
        // the *source* kernel var. Checked by `substitute_and_emit` before it falls
        // back to appending `.len()` to whatever `param_to_arg` has for the same name.
        let mut param_to_len: std::collections::HashMap<&str, String> = std::collections::HashMap::new();

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

            let is_buffer = matches!(field.qual, GpuQual::Unified | GpuQual::Global | GpuQual::ActorGlobal)
                && matches!(field.ty, Type::Array(_) | Type::ArrayN(_, _));

            // A resident source for this argument — either a same-scope alias
            // (`let fc = k1.y` — `gpu_resident_vars`, no Rust binding of its own) or a
            // bare kernel-field read used *inline*, with no `let` at all
            // (`Kernel2(k1.y, ...)`, e.g. `attention_heads_gpu`'s
            // `SoftmaxRowsKernel(k_scores.c, ...)`) — see `resident_field_alias`'s doc
            // for why both need the same check. Either way the source kernel's buffer
            // is cloned straight across, no upload, no `with` required
            // (docs/scoped-access-blocks.md's "no new syntax at all" rule). Must be
            // checked before `self.emit_expr(&arg.value)` below — an alias has no Rust
            // identifier to emit at all, and an inline field read would otherwise go
            // through `try_emit_kernel_field_read`'s unconditional download.
            let resident_alias = if is_buffer {
                self.resident_field_alias(&arg.value)
            } else {
                None
            };
            if let Some((src_kvar, src_field)) = resident_alias {
                self.line(&format!("{var_name}.{field_name}_buf = std::sync::Arc::clone(&{src_kvar}.{src_field}_buf);"));
                self.line(&format!("{var_name}.rebuild_bind_group();"));
                // No Rust value exists for `param_name` to record in `param_to_arg` --
                // record only the one thing a sibling output field's fill expression
                // might still need from it (its element count), computed straight from
                // the buffer's byte size divided by this field's own device element
                // width (the aliased buffer is reused as-is, so the width matches).
                let inner = kernel_host_scalar_type(&array_inner_type(&field.ty));
                param_to_len.insert(param_name, format!(
                    "({src_kvar}.{src_field}_buf.size() as usize / std::mem::size_of::<{inner}>())"
                ));
                continue;
            }

            let arg_rust = self.emit_expr(&arg.value);
            param_to_arg.insert(param_name, arg_rust.clone());
            if is_buffer {
                let inner = kernel_host_scalar_type(&array_inner_type(&field.ty));
                // If this constructor argument is literally one of the *current*
                // function's own parameters, and that parameter is typed
                // `BoringGpuArg<T>` (`current_fn_gpu_arg_param_names` —
                // `Checker::scan_fn_gpu_arg_params`/its transpiler-side mirror found it
                // used *only* this way), or a same-function local bound directly to a
                // `fn_returns_resident` call (`resident_call_vars` —
                // `try_emit_gpu_resident_call_let`, also a genuine `BoringGpuArg<T>`-typed
                // Rust binding), branch on the enum instead of always uploading: a
                // resident argument hands its buffer over directly (an `Arc::clone`, no
                // data copy), a host argument uploads exactly as before. See
                // docs/scoped-access-blocks.md's "Kernel Constructor Interaction".
                let is_gpu_arg_param = matches!(&arg.value.kind, ExprKind::Var(pname)
                    if self.current_fn_gpu_arg_param_names.contains(pname.as_str())
                        || self.resident_call_vars.contains_key(pname.as_str()));
                if is_gpu_arg_param {
                    self.line(&format!("match &{arg_rust} {{"));
                    self.line("    BoringGpuArg::Resident(buf, _len) => {");
                    self.line(&format!("        {var_name}.{field_name}_buf = std::sync::Arc::clone(buf);"));
                    self.line(&format!("        {var_name}.rebuild_bind_group();"));
                    self.line("    }");
                    self.line("    BoringGpuArg::Host(v) => {");
                    self.line(&format!(
                        "        {var_name}.copy_{field_name}_to_device(&v.iter().map(|&x| x as {inner}).collect::<Vec<{inner}>>());"
                    ));
                    self.line("    }");
                    self.line("}");
                } else {
                    self.line(&format!(
                        "{var_name}.copy_{field_name}_to_device(&{arg_rust}.iter().map(|&x| x as {inner}).collect::<Vec<{inner}>>());"
                    ));
                }
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
            let value_rust = self.substitute_and_emit(&value, &param_to_arg, &param_to_len);
            let count_rust = self.substitute_and_emit(&count, &param_to_arg, &param_to_len);
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
    fn substitute_and_emit(
        &self,
        expr: &Expr,
        subst: &std::collections::HashMap<&str, String>,
        len_subst: &std::collections::HashMap<&str, String>,
    ) -> String {
        match &expr.kind {
            ExprKind::Var(name) => subst.get(name.as_str()).cloned().unwrap_or_else(|| name.clone()),
            ExprKind::Int(n) => n.to_string(),
            ExprKind::Float(f) => f.to_string(),
            ExprKind::BinOp(op, l, r) => {
                let l_s = self.substitute_and_emit(l, subst, len_subst);
                let r_s = self.substitute_and_emit(r, subst, len_subst);
                format!("({} {} {})", l_s, crate::transpiler::helpers::binop_str(op), r_s)
            }
            // `init_param.length`/`.count` (e.g. `y = [0.0 for ..xs.length]`) — substitute
            // the object first, then `.len()`, same as the general `map_field` convention.
            // Falling through to `self.emit_expr(expr)` for this shape (the old
            // behavior) is wrong two ways over: it never applies the substitution (the
            // init param name isn't a real Rust binding at this call site), and
            // `emit_expr` itself treats an unknown lowercase name as a module path,
            // producing `xs::length` — not merely unsubstituted but not even valid
            // field-access syntax. `len_subst` takes priority when present: a param
            // consumed via the resident-alias fast path (see
            // `emit_kernel_construction`) has no Rust value to append `.len()` to at
            // all, only a pre-computed count expression.
            ExprKind::Field(obj, field) if field == "length" || field == "count" => {
                if let ExprKind::Var(name) = &obj.kind {
                    if let Some(ready) = len_subst.get(name.as_str()) {
                        return ready.clone();
                    }
                }
                format!("{}.len()", self.substitute_and_emit(obj, subst, len_subst))
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

    /// If the function currently being emitted has a GPU-resident return type
    /// (`current_fn_returns_resident`, set in `emit_top.rs::emit_fn`) and `expr` is a
    /// bare `k.field` read on a tracked kernel instance whose field is
    /// `'unified`/`'global`, emits `BoringGpuArg::Resident(Arc::clone(&k.field_buf),
    /// len)` instead of the unconditional `copy_field_to_host()` download
    /// `try_emit_kernel_field_read` would otherwise produce. The buffer survives the
    /// kernel instance's own drop at function exit because the field is already
    /// `Arc<wgpu::Buffer>` (`wgpu::host::emit_kernel_struct`) — cloning the `Arc` out
    /// is all this needs; no `Rc<RefCell<_>>`/`Arc<Mutex<_>>` wrapping of the whole
    /// kernel instance is required (see docs/scoped-access-blocks.md's
    /// "Implementation Notes" for why that was originally thought necessary).
    /// Called from `emit_stmt.rs`'s tail-expression handling, in place of the normal
    /// expression emitter, when this returns `Some`.
    pub(crate) fn try_emit_gpu_resident_return(&self, expr: &Expr) -> Option<String> {
        self.current_fn_returns_resident.as_ref()?;
        let ExprKind::Field(obj, field) = &expr.kind else { return None };
        let ExprKind::Var(var_name) = &obj.kind else { return None };
        let kname = self.kernel_vars.get(var_name.as_str())?;
        let decl = self.kernel_decls.get(kname)?;
        let field_decl: &KernelFieldDecl = decl.fields.iter().find(|f| &f.name == field)?;
        if !matches!(field_decl.qual, GpuQual::Unified | GpuQual::Global)
            || !matches!(field_decl.ty, Type::Array(_) | Type::ArrayN(_, _))
        {
            return None;
        }
        // The GPU buffer holds device-native (32-bit) elements regardless of the
        // host-facing element type -- divide by *that* size to get the element count.
        let device_ty = kernel_host_scalar_type(&array_inner_type(&field_decl.ty));
        Some(format!(
            "BoringGpuArg::Resident(std::sync::Arc::clone(&{var_name}.{field}_buf), ({var_name}.{field}_buf.size() as usize) / std::mem::size_of::<{device_ty}>())"
        ))
    }

    /// One element of a resident-tuple return's tail tuple-literal (see
    /// `try_emit_gpu_resident_tuple_return`): is `expr` itself already resident —
    /// either a bare `k.field` read (same construction `try_emit_gpu_resident_return`
    /// builds for the single-value case), or a bare `Var` already bound to a
    /// `BoringGpuArg<T>` Rust value (a same-scope `'gpu'unified` alias
    /// (`gpu_resident_vars`), a local bound to a `fn_returns_resident` call
    /// (`resident_call_vars`), or the *enclosing* function's own
    /// transitively-qualifying parameter (`current_fn_gpu_arg_param_names`) forwarded
    /// straight through)? `None` means "not detectably resident" — the caller falls
    /// back to wrapping it in `BoringGpuArg::Host(...)`, since the Rust return type at
    /// this tuple position is `BoringGpuArg<T>` regardless of what this particular
    /// tail expression happens to produce.
    fn try_resident_tuple_element(&self, expr: &Expr) -> Option<String> {
        if let ExprKind::Field(obj, field) = &expr.kind {
            let ExprKind::Var(var_name) = &obj.kind else { return None };
            let kname = self.kernel_vars.get(var_name.as_str())?;
            let decl = self.kernel_decls.get(kname)?;
            let field_decl: &KernelFieldDecl = decl.fields.iter().find(|f| &f.name == field)?;
            if !matches!(field_decl.qual, GpuQual::Unified | GpuQual::Global)
                || !matches!(field_decl.ty, Type::Array(_) | Type::ArrayN(_, _))
            {
                return None;
            }
            let device_ty = kernel_host_scalar_type(&array_inner_type(&field_decl.ty));
            return Some(format!(
                "BoringGpuArg::Resident(std::sync::Arc::clone(&{var_name}.{field}_buf), ({var_name}.{field}_buf.size() as usize) / std::mem::size_of::<{device_ty}>())"
            ));
        }
        if let ExprKind::Var(name) = &expr.kind {
            if self.gpu_resident_vars.contains_key(name.as_str())
                || self.resident_call_vars.contains_key(name.as_str())
                || self.current_fn_gpu_arg_param_names.contains(name.as_str())
            {
                return Some(format!("{name}.clone()"));
            }
        }
        None
    }

    /// Tuple counterpart of `try_emit_gpu_resident_return`: if the function currently
    /// being emitted has a resident-tuple return type (`current_fn_returns_resident_tuple`,
    /// set in `emit_top.rs::emit_fn`) and `expr` is a tuple literal matching that
    /// arity, emits each element according to its own position's residency —
    /// `try_resident_tuple_element` (chained, no download) when recognized,
    /// `BoringGpuArg::Host((expr).clone())` otherwise (a plain host value surfacing at
    /// a position whose Rust type is `BoringGpuArg<T>` regardless), and the ordinary
    /// expression emitter for a non-resident position. Called from `emit_stmt.rs`'s
    /// tail-expression handling, same convention as `try_emit_gpu_resident_return`.
    pub(crate) fn try_emit_gpu_resident_tuple_return(&self, expr: &Expr) -> Option<String> {
        let flags = self.current_fn_returns_resident_tuple.as_ref()?;
        let ExprKind::Tuple(elems) = &expr.kind else { return None };
        if elems.len() != flags.len() { return None; }
        let parts: Vec<String> = elems.iter().enumerate().map(|(i, el)| {
            if flags.get(i).copied().unwrap_or(false) {
                self.try_resident_tuple_element(el)
                    .unwrap_or_else(|| format!("BoringGpuArg::Host(({}).clone())", self.emit_expr(el)))
            } else {
                self.emit_expr_owned(el)
            }
        }).collect();
        Some(format!("({})", parts.join(", ")))
    }
}

pub(crate) fn array_inner_type(ty: &Type) -> Type {
    match ty {
        Type::Array(inner) | Type::ArrayN(inner, _) => (**inner).clone(),
        other => other.clone(),
    }
}

/// GPU-buffer element type for a boring scalar type (matches
/// `wgpu::host::host_scalar_type` — GPU buffers always use 32-bit elements).
pub(crate) fn kernel_host_scalar_type(ty: &Type) -> &'static str {
    match ty {
        Type::Int   => "i32",
        Type::Uint  => "u32",
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
        Type::Float => "f32",
        Type::Bool  => "bool",
        Type::Named(n) => match n.as_str() {
            "int"                    => "i32",
            "uint"                   => "u32",
            "uint8"                  => "u8",
            "int8"                   => "i8",
            "int16"                  => "i16",
            "int32"                  => "i32",
            "int64"                  => "i64",
            "int128"                 => "i128",
            "uint16"                 => "u16",
            "uint32"                 => "u32",
            "uint64"                 => "u64",
            "uint128"                => "u128",
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
/// outside a kernel buffer. Explicit fixed-width types keep their own exact
/// width (there's no "narrow for GPU-buffer efficiency" concept once the
/// width is already explicit).
pub(crate) fn kernel_host_element_type(ty: &Type) -> &'static str {
    match ty {
        Type::Int | Type::Uint => "i64",
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
        Type::Float => "f64",
        Type::Bool => "bool",
        Type::Named(n) => match n.as_str() {
            "int" | "uint" => "i64",
            "uint8"        => "u8",
            "int8"         => "i8",
            "int16"        => "i16",
            "int32"        => "i32",
            "int64"        => "i64",
            "int128"       => "i128",
            "uint16"       => "u16",
            "uint32"       => "u32",
            "uint64"       => "u64",
            "uint128"      => "u128",
            "float"        => "f64",
            "bool"         => "bool",
            _              => "i64",
        },
        Type::Qualified(inner, _) => kernel_host_element_type(inner),
        _ => "i64",
    }
}
