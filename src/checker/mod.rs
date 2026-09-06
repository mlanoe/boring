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

// Semantic checker — runs after parsing, before interpretation or transpilation.
//
// Current checks:
//   1. Immutability: assignment to a `let` or `lazy` binding.
//   2. Qualifier constraint: `mut 'shared` is always an error.
//   3. `lazy` misuse: `lazy` binding assigned via `=` after declaration
//      (the correct operator is `?=`).

use std::collections::HashMap;
use crate::ast::*;

// ─── Public interface ─────────────────────────────────────────────────────────

/// Shares its definition with the interpreter's/transpiler's error type (and the
/// transpiler's warning type); see `crate::errors::SourceError`'s doc comment.
pub use crate::errors::SourceError as CheckError;
pub use crate::errors::SourceError as CheckWarning;

pub struct CheckResult {
    pub errors:   Vec<CheckError>,
    pub warnings: Vec<CheckWarning>,
}

pub fn check(program: &Program) -> CheckResult {
    let mut checker = Checker::new();
    checker.collect_signatures(program);
    // Second pass: depends on `kernel_decls` being fully populated (a function may
    // be declared before the kernel it constructs), so it can't be folded into the
    // single-pass walk above. See `Checker::scan_fn_gpu_arg_params`.
    checker.collect_gpu_arg_params(program);
    checker.check_program(program);
    CheckResult { errors: checker.errors, warnings: checker.warnings }
}

/// Runs ONLY the kernel-dispatch-qualifier check (`check_kernel_dispatch_qualifier`)
/// -- every other check (`mut 'shared`, immutability, `lazy` misuse, GPU-resident
/// opacity) is walked but silenced. Used by the four GPU `emit_*` targets
/// (cuda/metal/rocm/wgpu), which never ran the full checker at all before (see
/// `check_kernel_dispatch_qualifier`'s doc). The full `check()` isn't safe to turn
/// on wholesale for those targets yet: `check_gpu_opacity` has a real, pre-existing
/// false positive on the GPU-resident-tuple-return pattern
/// (`tests/wgpu_codegen.rs`'s `test_gpu_resident_tuple_return_*` -- the transpiler's
/// `try_emit_gpu_resident_tuple_return` supports it, the checker's opacity walk
/// doesn't know about tail-position tuple returns) -- discovered by wiring the full
/// checker in and watching three previously-passing wgpu tests newly fail. That gap
/// predates and is unrelated to kernel-dispatch-qualifier rejection; fixing it is
/// its own separate task, not silently bundled in here.
pub fn check_kernel_dispatch_only(program: &Program) -> CheckResult {
    let mut checker = Checker::new();
    checker.kernel_dispatch_only = true;
    checker.collect_signatures(program);
    checker.collect_gpu_arg_params(program);
    checker.check_program(program);
    CheckResult { errors: checker.errors, warnings: checker.warnings }
}

// ─── Internal ─────────────────────────────────────────────────────────────────

/// One variable entry in the scope stack.
#[derive(Clone)]
struct Binding {
    kind: BindingKind,
    /// Declared type, when known (`let`/`var` with an explicit annotation, or a typed
    /// parameter). Needed for `with` blocks: a name's GPU-residency/actor/guard
    /// qualifier, and the struct it names for method-mutability lookup, both come
    /// from here. `None` when the type is inferred — such a binding can never be the
    /// subject of a `with` block that needs qualifier-specific codegen, only the
    /// no-op-everywhere-else path.
    ty: Option<Type>,
    /// `true` only for a `'gpu'unified`/`'gpu'global` binding initialized directly
    /// from a bare kernel-field read (`let py'gpu'unified = k.y`). This is the only
    /// case that is actually GPU-resident and opaque outside a `with` block.
    ///
    /// A `'gpu'unified`/`'gpu'global`-qualified variable initialized from anything
    /// else (an array literal/comprehension, a plain function call, ...) is just an
    /// ordinary host array up until it's passed into a kernel constructor — see
    /// `examples/saxpy.br`'s `var [float]'gpu'unified x = [0.0 for ..N]`, freely
    /// indexed and assigned on the host with no `with` wrapper anywhere. Gating
    /// opacity on the initializer's shape, rather than on the qualifier alone,
    /// is what keeps that existing, working pattern legal.
    resident_from_field: bool,
    /// For `let k = SomeKernel(...)` — the kernel declaration's name, when the
    /// initializer is a call to a known `kernel Name:` type. Lets a *later*,
    /// unannotated `let result = k.y` still infer GPU residency (see
    /// `Checker::infer_gpu_resident`) without requiring `'gpu'unified`/`'gpu'global`
    /// to be written out by hand.
    kernel_type: Option<String>,
}

struct Checker {
    errors:   Vec<CheckError>,
    warnings: Vec<CheckWarning>,
    /// Stack of scopes; each scope maps a name to its binding info.
    scopes:   Vec<HashMap<String, Binding>>,
    /// Names currently open in an enclosing `with` block — used to enforce that a
    /// `'gpu'unified`/`'gpu'global` value is opaque (no indexing/`.length`/iteration/
    /// interpolation) outside its own `with` wrapper. See docs/scoped-access-blocks.md.
    open_with_names: std::collections::HashSet<String>,
    /// Free function name -> which positional params are declared `var` (out-parameter).
    /// Signature-only, collected once up front — same lookup `def`/`req` legality
    /// already relies on elsewhere, reused here for the `with` mutation scan.
    fn_var_params: HashMap<String, Vec<bool>>,
    /// (struct name, method name) -> `true` if `def` (mutating), `false` if `req`.
    method_mutating: HashMap<(String, String), bool>,
    /// `kernel Name: ...` declarations, by name — collected once up front so a
    /// `let result = k.y` with no explicit qualifier can still be recognized as
    /// GPU-resident by checking `k`'s kernel type's own field declarations.
    kernel_decls: HashMap<String, KernelDecl>,
    /// Free function name -> its declared return type, for functions whose return
    /// type is `'gpu'unified`/`'gpu'global`-qualified. Lets `let fc = some_fn(...)`
    /// be recognized as GPU-resident (opaque outside `with`) the same way a bare
    /// `k.field` read already is — see `define_let`/`infer_gpu_resident` and
    /// docs/scoped-access-blocks.md's interprocedural case.
    fn_returns_resident: HashMap<String, Type>,
    /// Free function name -> per-position flags: `true` when the checker's bounded
    /// body scan (`scan_fn_gpu_arg_params`) found that parameter used *exclusively*
    /// as a bare-argument to a kernel constructor at a `'unified`/`'global` field
    /// position — the transpiler uses this to type that parameter `BoringGpuArg<T>`
    /// instead of a plain host array, and this carve-out is what lets a caller pass
    /// a GPU-resident value into such a call without a `with` block first.
    fn_gpu_arg_params: HashMap<String, Vec<bool>>,
    /// Free function name -> per-position flags, for a function whose return type is
    /// a `Type::Tuple` with at least one `'gpu'unified`/`'gpu'global`-qualified
    /// element — the tuple analogue of `fn_returns_resident`. Lets `let (sa, k, v) =
    /// mha_step_gpu(...)` infer per-binding residency from the corresponding tuple
    /// position, the same way a plain `let fc = linear_gpu(...)` infers it from the
    /// (single) return type. See `check_let_destructure`.
    fn_returns_resident_tuple: HashMap<String, Vec<bool>>,
    /// Free function name -> each positional parameter's declared type, when
    /// annotated (`None` for an unannotated/inferred parameter). Best-effort,
    /// static-annotation-only, same limitation as `Binding.ty` — used solely for
    /// the labeled multi-dim array cross-label check (`check_label_compat`),
    /// which only fires when both sides of a call are statically known to be
    /// `Type::LabeledArray`. See docs/array-multidim-proposal.md,
    /// "Cross-label compatibility between same-shape types".
    fn_param_types: HashMap<String, Vec<Option<Type>>>,
    /// When `true`, every check EXCEPT `check_kernel_dispatch_qualifier` is silenced
    /// (the tree is still walked, to keep scope/binding tracking correct, but no
    /// other error is pushed). See `check_kernel_dispatch_only`'s doc for why this
    /// exists.
    kernel_dispatch_only: bool,
    /// `'static` provenance gate (docs/qualifiers.md's `'static` section): `true` while checking a
    /// top-level `let` (`check_item`'s `Item::Let` arm never touches this field, so it
    /// keeps whatever value `check_fn` last restored it to — see that field's own
    /// default below) or the body of `fn main()`. `false` everywhere else. A
    /// `T'static NAME = Ctor(...)` constructor-call initializer is only legal while
    /// this is `true` — see `check_let_stmt`'s use of it. Saved/restored around each
    /// `check_fn` call so a nested `fn` (via `Stmt::Fn`) doesn't inherit `main`'s
    /// authorization.
    in_authorized_static_site: bool,
}

impl Checker {
    fn new() -> Self {
        Checker {
            errors: Vec::new(), warnings: Vec::new(), scopes: vec![HashMap::new()],
            open_with_names: std::collections::HashSet::new(),
            fn_var_params: HashMap::new(),
            method_mutating: HashMap::new(),
            kernel_decls: HashMap::new(),
            fn_returns_resident: HashMap::new(),
            fn_gpu_arg_params: HashMap::new(),
            fn_returns_resident_tuple: HashMap::new(),
            fn_param_types: HashMap::new(),
            kernel_dispatch_only: false,
            // Top-level `let`s are checked directly from `check_item`, never through
            // `check_fn` — this default is what makes them authorized without either
            // arm needing to set it explicitly.
            in_authorized_static_site: true,
        }
    }

    // ── Scope helpers ─────────────────────────────────────────────────────────

    fn push_scope(&mut self) { self.scopes.push(HashMap::new()); }

    fn pop_scope(&mut self) { self.scopes.pop(); }

    fn define(&mut self, name: &str, kind: BindingKind) {
        self.define_typed(name, kind, None);
    }

    fn define_typed(&mut self, name: &str, kind: BindingKind, ty: Option<Type>) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), Binding { kind, ty, resident_from_field: false, kernel_type: None });
        }
    }

    fn define_let(&mut self, name: &str, kind: BindingKind, ty: Option<Type>, value: Option<&Expr>) {
        // `let k = SomeKernel(...)` — track which kernel type `k` is an instance of,
        // so a later unannotated read of one of its fields can be recognized too.
        let kernel_type = value.and_then(|v| match &v.kind {
            ExprKind::Call(callee, _) => match &callee.kind {
                ExprKind::Var(n) if self.kernel_decls.contains_key(n.as_str()) => Some(n.clone()),
                _ => None,
            },
            _ => None,
        });

        let (resident_from_field, ty) = if ty.as_ref().map(|t| t.gpu_resident_qual().is_some()).unwrap_or(false) {
            // Explicit `'gpu'unified`/`'gpu'global` annotation — resident if sourced
            // from a bare kernel-field read (same-scope case) or a call to a function
            // whose own return type is resident (interprocedural case, `let fc =
            // linear_gpu(...)` — docs/scoped-access-blocks.md).
            let resident = matches!(value, Some(e) if is_kernel_field_read(e) || self.is_resident_call(e));
            (resident, ty)
        } else if ty.is_none() {
            // No annotation at all — infer from `value`'s shape: a bare `k.field`
            // read where `k` is a tracked kernel instance and its declared field is
            // `'unified`/`'global` (device-side `GpuQual`, on the kernel struct decl).
            match self.infer_gpu_resident(value) {
                Some(inferred_ty) => (true, Some(inferred_ty)),
                None => (false, ty),
            }
        } else {
            (false, ty)
        };

        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), Binding { kind, ty, resident_from_field, kernel_type });
        }
    }

    /// If `value` is `k.field` where `k` is bound to a known kernel type and that
    /// kernel's `field` is declared `'unified`/`'global` (`GpuQual`), returns the
    /// equivalent host-context `Type::Qualified(_, OwnerQual::GpuUnified|GpuGlobal)` —
    /// the same type an explicit `'gpu'unified`/`'gpu'global` annotation would give.
    /// Also recognizes a call to a function whose own declared return type is
    /// already resident (`let fc = linear_gpu(...)`, no annotation) — the
    /// interprocedural counterpart to the same-scope field-read case, same rationale.
    fn infer_gpu_resident(&self, value: Option<&Expr>) -> Option<Type> {
        let value = value?;
        if let ExprKind::Call(callee, _) = &value.kind {
            if let ExprKind::Var(fn_name) = &callee.kind {
                if let Some(ret_ty) = self.fn_returns_resident.get(fn_name) {
                    return Some(ret_ty.clone());
                }
            }
        }
        let ExprKind::Field(obj, field) = &value.kind else { return None };
        let ExprKind::Var(kvar) = &obj.kind else { return None };
        let kernel_type = self.lookup(kvar)?.kernel_type.as_ref()?;
        let decl = self.kernel_decls.get(kernel_type)?;
        let field_decl = decl.fields.iter().find(|f| &f.name == field)?;
        let qual = match field_decl.qual {
            GpuQual::Unified => OwnerQual::GpuUnified,
            GpuQual::Global => OwnerQual::GpuGlobal,
            _ => return None,
        };
        Some(Type::Qualified(Box::new(field_decl.ty.clone()), qual))
    }

    /// Is `expr` a call to a function whose declared return type is already
    /// GPU-resident (`self.fn_returns_resident`)? See `infer_gpu_resident`'s doc for
    /// why this and `is_kernel_field_read` are the two initializer shapes that make a
    /// `'gpu'unified`/`'gpu'global`-annotated `let` actually opaque outside `with`.
    fn is_resident_call(&self, expr: &Expr) -> bool {
        matches!(&expr.kind, ExprKind::Call(callee, _)
            if matches!(&callee.kind, ExprKind::Var(n) if self.fn_returns_resident.contains_key(n)))
    }

    fn lookup(&self, name: &str) -> Option<&Binding> {
        for scope in self.scopes.iter().rev() {
            if let Some(b) = scope.get(name) { return Some(b); }
        }
        None
    }

    // ── Signature collection (pre-pass) ───────────────────────────────────────
    // Signature-only lookup for the `with` mutation scan — never opens a callee's
    // body, matching how `def`/`req` legality is already resolved without reading
    // the called method's body.

    fn collect_signatures(&mut self, program: &Program) {
        for item in &program.items { self.collect_item_signatures(item); }
    }

    fn collect_item_signatures(&mut self, item: &Item) {
        match item {
            Item::Fn(f)     => self.collect_fn_signature(f),
            Item::Struct(s) => self.collect_struct_signature(s),
            Item::Kernel(k) => { self.kernel_decls.insert(k.name.clone(), k.clone()); }
            Item::Mod(m)    => { for i in &m.items { self.collect_item_signatures(i); } }
            Item::Stmt(s)   => self.collect_stmt_signatures(s),
            _ => {}
        }
    }

    fn collect_stmt_signatures(&mut self, stmt: &Stmt) {
        // Local (nested) fn/struct declarations are visible to `with` blocks in the
        // same or later scopes, so their signatures are worth collecting too.
        match stmt {
            Stmt::Fn(f)     => self.collect_fn_signature(f),
            Stmt::Struct(s) => self.collect_struct_signature(s),
            Stmt::Mod(m)    => { for i in &m.items { self.collect_item_signatures(i); } }
            Stmt::If(s)     => { for (_, b) in &s.branches { for st in b { self.collect_stmt_signatures(st); } } if let Some(b) = &s.else_body { for st in b { self.collect_stmt_signatures(st); } } }
            Stmt::While(s)  => { for st in &s.body { self.collect_stmt_signatures(st); } }
            Stmt::For(s)    => { for st in &s.body { self.collect_stmt_signatures(st); } }
            Stmt::Loop(s)   => { for st in &s.body { self.collect_stmt_signatures(st); } }
            Stmt::DoWhile(s) => { for st in &s.body { self.collect_stmt_signatures(st); } }
            Stmt::Try(s)    => { for st in &s.body { self.collect_stmt_signatures(st); } for c in &s.catch_clauses { for st in &c.body { self.collect_stmt_signatures(st); } } }
            Stmt::Guard(s)  => { for st in &s.else_body { self.collect_stmt_signatures(st); } }
            Stmt::Defer(b)  => { for st in b { self.collect_stmt_signatures(st); } }
            Stmt::With(s)   => { for st in &s.body { self.collect_stmt_signatures(st); } }
            _ => {}
        }
    }

    fn collect_fn_signature(&mut self, f: &FnDecl) {
        let var_flags: Vec<bool> = f.params.iter().map(|p| p.rebindable).collect();
        self.fn_var_params.insert(f.name.clone(), var_flags);
        self.fn_param_types.insert(f.name.clone(), f.params.iter().map(|p| p.ty.clone()).collect());
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
        for stmt in &f.body { self.collect_stmt_signatures(stmt); }
    }

    fn collect_struct_signature(&mut self, s: &StructDecl) {
        for m in &s.methods {
            self.method_mutating.insert((s.name.clone(), m.name.clone()), m.mutating);
        }
    }

    // ── GPU-resident parameter scan (second pass) ─────────────────────────────
    // Depends on `kernel_decls` being fully populated by the pass above, so it
    // can't be folded into `collect_signatures` itself — a function may be
    // declared before the kernel type it constructs.

    /// A parameter becomes `BoringGpuArg`-dual-typed (see docs/scoped-access-blocks.md,
    /// "Kernel Constructor Interaction") when it's used *exclusively*, everywhere in
    /// its function's body, as a bare argument at a position that either (a) is a raw
    /// kernel constructor's `'unified`/`'global` array-field init position, or (b) is
    /// another Boring function whose own corresponding parameter *already* qualifies —
    /// transitively, any number of call-graph hops deep (`wrap_scale(x, n)` forwarding
    /// into `scale(x, n)` forwarding into a raw `ScaleKernel(x, n)` construction, say).
    /// Any other use anywhere (indexing, `.length`, arithmetic, a disqualifying call
    /// position, ...) disqualifies it for that parameter — no regression, just no
    /// speedup for that one parameter.
    ///
    /// Because qualification of a parameter can depend on a *callee's* qualification
    /// (computed from the same analysis), this can't be a single top-to-bottom walk —
    /// it's a fixed point over the whole program's functions: repeat full passes,
    /// each pass re-deriving every function's flags from the previous pass's known
    /// flags, until a full pass changes nothing. Each pass only ever flips a flag
    /// `false -> true` (a use only *gains* a qualifying classification as a callee's
    /// own flags fill in — see `scan_var_call_arg_uses`'s "any use that isn't
    /// qualifying disqualifies" rule), so this is a monotone dataflow problem over a
    /// finite lattice and is guaranteed to converge — a pure recursive/mutual-recursion
    /// cycle with no base case ever reaching a real kernel constructor simply never
    /// flips anything and converges immediately, qualifying nothing (correct: there's
    /// no actual kernel underneath to be resident from). File order doesn't matter —
    /// all functions in the program are gathered up front, so a callee defined *after*
    /// its caller in source is visible to every pass just like one defined before.
    fn collect_gpu_arg_params(&mut self, program: &Program) {
        if self.kernel_decls.is_empty() { return; }
        let mut all_fns: Vec<&FnDecl> = Vec::new();
        for item in &program.items { Self::gather_fns_item(item, &mut all_fns); }

        let mut flags_by_fn: HashMap<&str, Vec<bool>> = all_fns.iter()
            .map(|f| (f.name.as_str(), vec![false; f.params.len()]))
            .collect();

        // Defensive bound on passes: monotonicity over a lattice of this total size
        // guarantees convergence in at most (total param slots + 1) passes, so this
        // cap is never actually hit — it only guards against a future logic error
        // turning this into an infinite loop.
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
                            return matches!(field.qual, GpuQual::Unified | GpuQual::Global)
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

    fn gather_fns_item<'a>(item: &'a Item, out: &mut Vec<&'a FnDecl>) {
        match item {
            Item::Fn(f)   => out.push(f),
            Item::Mod(m)  => { for i in &m.items { Self::gather_fns_item(i, out); } }
            Item::Stmt(s) => Self::gather_fns_stmt(s, out),
            _ => {}
        }
    }

    fn gather_fns_stmt<'a>(stmt: &'a Stmt, out: &mut Vec<&'a FnDecl>) {
        match stmt {
            Stmt::Fn(f)     => out.push(f),
            Stmt::Mod(m)    => { for i in &m.items { Self::gather_fns_item(i, out); } }
            Stmt::If(s)     => { for (_, b) in &s.branches { for st in b { Self::gather_fns_stmt(st, out); } } if let Some(b) = &s.else_body { for st in b { Self::gather_fns_stmt(st, out); } } }
            Stmt::While(s)  => { for st in &s.body { Self::gather_fns_stmt(st, out); } }
            Stmt::For(s)    => { for st in &s.body { Self::gather_fns_stmt(st, out); } }
            Stmt::Loop(s)   => { for st in &s.body { Self::gather_fns_stmt(st, out); } }
            Stmt::DoWhile(s) => { for st in &s.body { Self::gather_fns_stmt(st, out); } }
            Stmt::Try(s)    => { for st in &s.body { Self::gather_fns_stmt(st, out); } for c in &s.catch_clauses { for st in &c.body { Self::gather_fns_stmt(st, out); } } }
            Stmt::Guard(s)  => { for st in &s.else_body { Self::gather_fns_stmt(st, out); } }
            Stmt::Defer(b)  => { for st in b { Self::gather_fns_stmt(st, out); } }
            Stmt::With(s)   => { for st in &s.body { Self::gather_fns_stmt(st, out); } }
            _ => {}
        }
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    fn error(&mut self, msg: impl Into<String>, line: usize, col: usize) {
        self.errors.push(CheckError::at(msg, line, col));
    }

    // No current check calls this yet -- `CheckResult::warnings` is already wired up
    // end-to-end (consumed and printed by `main::report_check_result`), so this is a
    // ready extension point for the next non-fatal check, not dead infrastructure.
    #[allow(dead_code)]
    fn warning(&mut self, msg: impl Into<String>, line: usize, col: usize) {
        self.warnings.push(CheckWarning::at(msg, line, col));
    }

    // ── Qualifier constraint: `mut 'shared` / `mut 'weak` ─────────────────────

    /// `var_mut` — see `LetStmt.var_mut`'s doc: `binding == Mut` (bare/`let
    /// mut`) always requests content-mutation; `binding == Var` only does with
    /// an explicit second `mut` (`var mut`). Pass `false` for a parameter,
    /// which has no `var_mut` concept of its own.
    fn requests_mut(binding: &BindingKind, var_mut: bool) -> bool {
        matches!(binding, BindingKind::Mut) || (matches!(binding, BindingKind::Var) && var_mut)
    }

    fn check_qualifier_constraint(&mut self, binding: &BindingKind, var_mut: bool, ty: &Option<Type>, line: usize, col: usize) {
        if self.kernel_dispatch_only { return; }
        if !Self::requests_mut(binding, var_mut) { return; }
        let Some(ty) = ty else { return };
        // `mut` always wraps the parsed type in `Type::Mut` now (§1) — strip it
        // before inspecting the shape.
        let ty = ty.without_mut();
        if self.type_has_shared(ty) {
            self.error(
                "cannot combine `mut` with `'shared`: shared references are immutable by design; use `'actor` for interior mutability",
                line, col,
            );
        }
        // `'static` (`&'static T`) has exactly as little interior mutability as
        // `'shared` (`Rc`/`Arc<T>`) — a bare reference, nothing for `mut` to unlock.
        // See docs/qualifiers.md's `'static` section, "No interior mutability".
        if self.type_has_static(ty) {
            self.error(
                "cannot combine `mut` with `'static`: a &'static reference has no interior mutability to unlock",
                line, col,
            );
        }
        // A `'weak` reference has no operations besides `.upgrade()`/`.clone()`
        // (both non-mutating) until it's upgraded — nothing for `mut` to unlock
        // on the weak reference itself, regardless of what the *upgraded* value
        // would allow (`T'shared'weak`, `T'actor'weak`, `T'guard'weak` alike —
        // docs/book.md's rejection table). Checked on the
        // *outermost* qualifier only — `'weak` is always the last link in the
        // chain (`T'actor'weak`, never `T'weak'actor`).
        if matches!(ty, Type::Qualified(_, OwnerQual::Weak)) {
            self.error(
                "cannot combine `mut` with `'weak`: a weak reference has no operations besides `.upgrade()`/`.clone()` until upgraded — there is nothing for `mut` to unlock",
                line, col,
            );
        }
    }

    fn type_has_shared(&self, ty: &Type) -> bool {
        match ty {
            Type::Qualified(_, OwnerQual::Shared) => true,
            Type::Qualified(inner, _) => self.type_has_shared(inner),
            // NOT Type::Array/Dict/Set: `mut [T] arr` grants *structural* mutation
            // (push/pop) on the collection itself, entirely independent of whatever
            // `mut` would or wouldn't unlock on its element type — recursing into the
            // element here rejected `mut [Point'shared] arr = []`, which is valid and
            // compiles fine, as if it were `mut Point'shared p` (content mutation on a
            // single 'shared value, which really has nothing for `mut` to unlock).
            Type::Optional(inner) | Type::Dyn(inner) | Type::Impl(inner) => {
                self.type_has_shared(inner)
            }
            _ => false,
        }
    }

    // ── `'static` provenance gate ────────────────────────────────────────────
    //
    // docs/qualifiers.md's `'static` section: a `T'static NAME = Ctor(...)` constructor-call
    // initializer is legal only at top level or inside `main` (tracked via
    // `in_authorized_static_site`, set in `check_fn`) — the two authorized
    // construction sites this check covers (the third, `type let`, is implicit and
    // has no `'static` annotation to check here at all). Anywhere else, the
    // initializer must already be a reference to an existing 'static-typed value
    // (a bare name, not a fresh construction) — never verified beyond "not a
    // constructor call" today (confirming the referenced name is itself genuinely
    // 'static-typed would need a real expression-type-inference pass this checker
    // doesn't have; a real gap, not silently assumed correct).
    fn check_static_provenance(&mut self, ty: &Option<Type>, value: Option<&Expr>, line: usize, col: usize) {
        let Some(Type::Qualified(_, OwnerQual::Static)) = ty else { return };
        let Some(value) = value else { return };
        if self.in_authorized_static_site { return; }
        if Self::is_constructor_call_expr(value) {
            self.error(
                "cannot construct a 'static instance here — 'static values may only be constructed at top level or inside `main`",
                line, col,
            );
        }
    }

    /// The provenance gate's other half: `check_static_provenance` only covers a
    /// `let`'s own initializer — it says nothing about passing an *existing*,
    /// non-`'static` value into a call argument whose parameter demands `'static`.
    /// A local `Config`, or `self.field`, or a fresh `Config(...)` written inline
    /// as the argument, all produce real `cargo build` failures (or worse, would
    /// be unsound if they somehow compiled) once the callee treats the parameter
    /// as genuinely program-lifetime. Only a bare `Var` whose own declared type is
    /// already `'static` is accepted — a `self.field`, a method-call result, or
    /// any other expression this checker can't statically type as `'static` is
    /// rejected rather than risked (conservative by design: nothing depends on
    /// `'static` yet, so erring towards rejecting an as-yet-unrecognized-but-valid
    /// pattern is the safer default than silently letting an unsound one through).
    fn check_static_arg_provenance(&mut self, target_ty: Option<&Type>, arg: &Expr, line: usize, col: usize) {
        if !matches!(target_ty, Some(Type::Qualified(_, OwnerQual::Static))) { return; }
        let is_provably_static = match &arg.kind {
            ExprKind::Var(name) => matches!(
                self.lookup(name).and_then(|b| b.ty.as_ref()),
                Some(Type::Qualified(_, OwnerQual::Static))
            ),
            _ => false,
        };
        if !is_provably_static {
            self.error(
                "cannot pass a non-'static value where 'static is expected — the argument must \
                 already be a 'static-typed binding (a name whose own type is T'static), not a \
                 local value, a fresh construction, or a field read",
                line, col,
            );
        }
    }

    /// Best-effort recognition of "this expression constructs a fresh instance" —
    /// a call whose callee is a capitalized name (`Config(...)`, `Point.new(...)`).
    /// Mirrors the same heuristic `top_level_let_external_call` (transpiler side)
    /// and this file's own `top_level_let_is_string_literal`-style checks use for
    /// "is this initializer a constructor call" — mirrors, not literally reuses,
    /// since checker and transpiler are separate passes.
    fn is_constructor_call_expr(value: &Expr) -> bool {
        match &value.kind {
            ExprKind::Call(callee, _) => matches!(&callee.kind, ExprKind::Var(n) if n.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)),
            ExprKind::MethodCall(obj, _, _) => matches!(&obj.kind, ExprKind::Var(n) if n.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)),
            _ => false,
        }
    }

    fn type_has_static(&self, ty: &Type) -> bool {
        match ty {
            Type::Qualified(_, OwnerQual::Static) => true,
            Type::Qualified(inner, _) => self.type_has_static(inner),
            // See type_has_shared's comment just above — same reasoning applies here:
            // `mut [T] arr` is structural, independent of the element type's own
            // qualifier, so `mut [Point'static] arr = []` must not be rejected either.
            Type::Optional(inner) | Type::Dyn(inner) | Type::Impl(inner) => {
                self.type_has_static(inner)
            }
            _ => false,
        }
    }

    // ── Qualifier constraint: `mut` on a tuple-typed binding ───────────────────
    //
    // A tuple has no in-place mutation surface — no field-index assignment
    // (`t.0 = v`), no user-definable methods. `mut` on an owned tuple binding
    // (`mut (T1, T2) t = ...`) would therefore never unlock anything: unlike
    // `mut Counter c` (permits `c.inc()`), there is no operation that becomes
    // legal on `t` under `mut` that wasn't already legal under `let`. `var`
    // remains meaningful — it allows reassigning `t` to a whole new tuple.
    //
    // Checked both when the tuple type is explicit (`mut (T1, T2) t = ...`)
    // and when it's only inferred from a tuple-literal initializer
    // (`mut t = (1, 2)`) — the same constraint applies either way, so the
    // absence of a type annotation must not let it slip through.
    fn check_tuple_mut_constraint(&mut self, binding: &BindingKind, var_mut: bool, ty: &Option<Type>, value: &Option<Expr>, line: usize, col: usize) {
        if self.kernel_dispatch_only { return; }
        if !Self::requests_mut(binding, var_mut) { return; }
        // `mut` always wraps the parsed type in `Type::Mut` now (§1) — strip it
        // before inspecting the shape, same as `check_qualifier_constraint`.
        let is_tuple = match ty {
            Some(ty) => matches!(ty.without_mut(), Type::Tuple(_)),
            None => matches!(value, Some(v) if matches!(v.kind, ExprKind::Tuple(_))),
        };
        if is_tuple {
            self.error(
                "cannot mark a tuple binding as `mut`: tuples have no in-place mutation — use `let` (fixed) or `var` (reassignable)",
                line, col,
            );
        }
    }

    // ── Qualifier constraint: `mut` on a scalar ────────────────────────────────
    //
    // No `def` methods exist on a primitive — nothing for `mut` to unlock.
    // Retires the historical "`mut` ≡ `var` for scalars" shortcut
    // (docs/book.md): `mut int x = 0` is a checker error now,
    // not a silent downgrade to `var int x = 0`. Checked both when the type is
    // explicit and when it's only inferred from a literal initializer, mirroring
    // `check_tuple_mut_constraint` exactly.
    fn is_scalar_type(ty: &Type) -> bool {
        matches!(ty,
            Type::Int | Type::Uint | Type::Uint8
                | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128
                | Type::Float32 | Type::Float64 | Type::Bool)
        // Lowercase Boring keywords (`int`, `uint`, `float`, `bool`, sized
        // variants) parse as `Type::Named` — resolved to the variants above only
        // later (interpreter alias table / transpiler size lookup), which the
        // checker doesn't have access to — match the spelling directly instead,
        // same list `emit_top.rs`'s `is_copy_type` already keys off of.
        // `f32`/`f64` are included here too — a pre-existing gap (this list
        // covered every fixed-width int alias but not the float ones) closed
        // alongside adding float32/float64 (docs/float-width-types.md).
        || matches!(ty, Type::Named(n) if matches!(n.as_str(),
            "int" | "uint" | "uint8" | "float" | "float32" | "float64" | "bool"
            | "int8" | "int16" | "int32" | "int64" | "int128"
            | "uint16" | "uint32" | "uint64" | "uint128"
            | "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
            | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "f32" | "f64"))
    }

    fn check_scalar_mut_constraint(&mut self, binding: &BindingKind, var_mut: bool, ty: &Option<Type>, value: &Option<Expr>, line: usize, col: usize) {
        if self.kernel_dispatch_only { return; }
        if !Self::requests_mut(binding, var_mut) { return; }
        let is_scalar = match ty {
            Some(ty) => Self::is_scalar_type(ty.without_mut()),
            None => matches!(value, Some(v) if matches!(v.kind, ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_))),
        };
        if is_scalar {
            self.error(
                "cannot mark a scalar as `mut`: primitives have no `def` methods to unlock — use `var` for a rebindable scalar",
                line, col,
            );
        }
    }

    // ── Qualifier constraint: `{mut T}` (set element mutability) ───────────────
    //
    // `HashSet<T>` exposes no mutable element access in Rust at all — no
    // `iter_mut()`, no `get_mut()` — because mutating an element in place could
    // change its `Hash`/`Eq` behavior and silently corrupt the set's buckets
    // (docs/book.md's "Sets — `{T}`" section documents the resulting rule —
    // this was, for a long time, the one item in this whole area that was
    // documented as rejected but never actually wired up). Unlike `check_tuple_mut_constraint`/
    // `check_scalar_mut_constraint`, this does NOT gate on whether the *outer*
    // binding itself requests `mut` (`Self::requests_mut`) — `let {mut Point}
    // pts = {}` is illegal even though `pts` is a plain `let`, because the
    // illegality lives on the Set's element type, wherever it's nested, not on
    // the binding. `mut {T}` (mutable on the set itself — structural
    // add/remove) is a different axis entirely and is unaffected.
    fn check_set_mut_constraint(&mut self, ty: &Option<Type>, line: usize, col: usize) {
        if self.kernel_dispatch_only { return; }
        if let Some(ty) = ty {
            if ty.contains_illegal_mut_set() {
                self.error(
                    "cannot use `mut` on a set's element type (`{mut T}`): `HashSet<T>` has no mutable element access in Rust (no `iter_mut`/`get_mut`) — `mut {T}` (mutable on the set itself, for structural add/remove) is unaffected",
                    line, col,
                );
            }
        }
    }

    // ── Kernel dispatch: reject a `'shared`/`'actor`/`'guard`-qualified instance ──

    /// A kernel struct instance dispatched via `kernel:` is launched through
    /// `__boring_launch(mut self, ...)` — it needs direct, exclusive ownership on
    /// the host side. `'shared`/`'actor`(`'task`)/`'guard`(`'task`) wrap the value in
    /// `Rc`/`Arc`/`RefCell`/`Mutex`/`RwLock`, none of which the generated dispatch
    /// code knows how to unwrap; nothing previously rejected this combination at
    /// compile time (see `docs/cuda-module.md`'s "Known limitations").
    fn qualifier_name_for_kernel_dispatch(&self, ty: &Type) -> Option<&'static str> {
        match ty {
            Type::Qualified(_, OwnerQual::Shared)    => Some("'shared"),
            Type::Qualified(_, OwnerQual::Actor)     => Some("'actor"),
            Type::Qualified(_, OwnerQual::ActorTask) => Some("'actor'task"),
            Type::Qualified(_, OwnerQual::Guard)     => Some("'guard"),
            Type::Qualified(_, OwnerQual::GuardTask) => Some("'guard'task"),
            Type::Qualified(inner, _) => self.qualifier_name_for_kernel_dispatch(inner),
            _ => None,
        }
    }

    fn check_kernel_dispatch_qualifier(&mut self, kernel: &Expr, line: usize, col: usize) {
        let ExprKind::Var(name) = &kernel.kind else { return };
        let Some(binding) = self.lookup(name) else { return };
        if binding.kernel_type.is_none() { return; }
        let Some(ty) = &binding.ty else { return };
        if let Some(qual) = self.qualifier_name_for_kernel_dispatch(ty) {
            self.error(
                format!(
                    "cannot dispatch `{name}` via `kernel:` — it is `{qual}`-qualified; \
                     kernel dispatch needs direct, exclusive ownership, not a shared/actor/guard \
                     wrapper, so declare `{name}` without a wrapping qualifier"
                ),
                line, col,
            );
        }
    }

    // ── Top-level ─────────────────────────────────────────────────────────────

    fn check_program(&mut self, program: &Program) {
        for item in &program.items { self.check_item(item); }
    }

    fn check_item(&mut self, item: &Item) {
        match item {
            Item::Let(s)    => self.check_let_stmt(s),
            Item::Fn(f)     => self.check_fn(f),
            Item::Struct(s) => self.check_struct(s),
            Item::Enum(e)   => self.check_enum(e),
            Item::Ext(e)    => self.check_ext(e),
            Item::Mod(m)    => { for i in &m.items { self.check_item(i); } }
            Item::Stmt(s)   => self.check_stmt(s),
            Item::Kernel(k) => self.check_kernel_decl(k),
            Item::Trait(t)  => self.check_trait(t),
            Item::Use(_) | Item::Alias(_) => {}
        }
    }

    // ── Kernel field types: `LabeledArray` shape ────────────────────────────────
    //
    // Deliberately narrow: only `Type::labeled_array_shape_error` on each field's
    // declared type. Not gated behind `kernel_dispatch_only` — this must fire for
    // every real target (`boring run`, `boring build`, and `--target
    // cuda`/`rocm`/`metal`/`wgpu` via `check_kernel_dispatch_only`), unlike this
    // checker's other rules, which are `boring run`/`boring build`-only. Kernel
    // bodies (methods/inits) are intentionally not walked here — that's
    // unrelated, pre-existing scope this pass has never covered, and adding it
    // isn't this check's job.

    fn check_kernel_decl(&mut self, k: &KernelDecl) {
        for field in &k.fields {
            if let Some(msg) = field.ty.labeled_array_shape_error() {
                self.error(msg, field.line, field.col);
            }
            // Axis-count cap is kernel-field-specific (GPU thread.x/y/z), not a
            // property of the type itself — CPU-side labeled arrays are unbounded
            // (docs/array-multidim-proposal.md, "Generalizing beyond 3 axes"), so
            // this lives here rather than inside labeled_array_shape_error.
            if let Some((_, axes)) = field.ty.as_labeled_array() {
                if axes.len() > 3 {
                    self.error(
                        format!(
                            "kernel fields support at most 3 axes (GPU thread.x/y/z) — \
                             got {} ({})",
                            axes.len(),
                            axes.iter().map(|a| a.label.as_str()).collect::<Vec<_>>().join(", "),
                        ),
                        field.line, field.col,
                    );
                }
            }
        }
    }

    // ── Labeled multi-dimensional array checks ────────────────────────────────
    // See docs/array-multidim-proposal.md. Two rules, both best-effort and
    // static-annotation-only — same limitation as `Binding.ty` elsewhere in this
    // file (silently skipped whenever a type isn't statically known; never a
    // false positive from a type this checker can't see):
    //   1. Passing a `Type::LabeledArray` across a call/assignment boundary
    //      where the target's axis labels differ from the source's requires an
    //      explicit `as [...]` mapping (`check_label_compat`).
    //   2. `as [...]` itself must be a complete bijection over the source's own
    //      axis labels — every source axis mapped exactly once
    //      (`check_relabel_cast`).

    /// Peels ownership-qualifier wrappers (`'unified`, `'global`, ...) down to
    /// the base type — a labeled array's qualifier is irrelevant to whether its
    /// axis labels match another one's.
    fn strip_qualifiers(ty: &Type) -> &Type {
        match ty {
            Type::Qualified(inner, _) => Self::strip_qualifiers(inner),
            _ => ty,
        }
    }

    /// Best-effort static type of an expression, for the labeled-array checks
    /// only. Handles exactly the two cases needed: a bound variable's declared
    /// type, and a `RelabelCast`'s own resulting type (synthesized from its
    /// mapping, so passing a relabeled value onward — or a chain of two casts —
    /// can still be checked). `None` for anything else; this is deliberately
    /// not a general type-inference pass.
    fn static_labeled_array_type(&self, expr: &Expr) -> Option<Type> {
        match &expr.kind {
            ExprKind::Var(name) => self.lookup(name)?.ty.clone(),
            ExprKind::RelabelCast(inner, pairs) => {
                let inner_ty = self.static_labeled_array_type(inner)?;
                let (elem, axes) = Self::strip_qualifiers(&inner_ty).as_labeled_array()?;
                let mut mapped = Vec::with_capacity(pairs.len());
                for (target_label, source_label) in pairs {
                    let axis = axes.iter().find(|a| &a.label == source_label)?;
                    mapped.push(LabeledAxis { label: target_label.clone(), size: axis.size.clone() });
                }
                Some(Type::LabeledArray(Box::new(elem.clone()), mapped))
            }
            _ => None,
        }
    }

    /// The core cross-label rule: if both `source_ty` and `target_ty` are
    /// (possibly qualified) `Type::LabeledArray`s with the same arity but
    /// different axis labels at some position, `source_expr` must itself be a
    /// `RelabelCast` — otherwise this is exactly the silent-transpose risk
    /// docs/array-multidim-proposal.md's "Cross-label compatibility" section
    /// exists to close off (e.g. `width, height` passed where `line, column` is
    /// expected). Does nothing when arities differ (a more basic type error,
    /// not this rule's job) or when either side isn't a labeled array at all.
    fn check_label_compat(&mut self, source_expr: &Expr, source_ty: &Type, target_ty: &Type, line: usize, col: usize) {
        let Some((_, source_axes)) = Self::strip_qualifiers(source_ty).as_labeled_array() else { return };
        let Some((_, target_axes)) = Self::strip_qualifiers(target_ty).as_labeled_array() else { return };
        if source_axes.len() != target_axes.len() { return; }
        let labels_match = source_axes.iter().zip(target_axes.iter()).all(|(s, t)| s.label == t.label);
        if labels_match { return; }
        if matches!(source_expr.kind, ExprKind::RelabelCast(..)) { return; }
        let source_labels: Vec<&str> = source_axes.iter().map(|a| a.label.as_str()).collect();
        let target_labels: Vec<&str> = target_axes.iter().map(|a| a.label.as_str()).collect();
        self.error(
            format!(
                "label sets ({}) \u{2260} ({}) — use `as [{}]` to map explicitly",
                source_labels.join(", "), target_labels.join(", "),
                target_labels.iter().zip(source_labels.iter())
                    .map(|(t, s)| format!("{t} = {s}"))
                    .collect::<Vec<_>>().join(", "),
            ),
            line, col,
        );
    }

    /// `as [...]`'s own validity: the mapping must be a complete bijection over
    /// the source's axis labels — every source axis named exactly once as some
    /// pair's `source_label`, no unknown or duplicated source axes. Silently
    /// skipped when the source's type isn't statically known (same best-effort
    /// limitation as `check_label_compat`).
    fn check_relabel_cast(&mut self, inner: &Expr, pairs: &[(String, String)], line: usize, col: usize) {
        let Some(inner_ty) = self.static_labeled_array_type(inner) else { return };
        let Some((_, axes)) = Self::strip_qualifiers(&inner_ty).as_labeled_array() else { return };
        let axes = axes.to_vec(); // release the borrow on inner_ty before pushing errors

        let mut seen_targets = std::collections::HashSet::new();
        for (target, _) in pairs {
            if !seen_targets.insert(target.as_str()) {
                self.error(format!("duplicate target axis '{target}' in `as [...]` mapping"), line, col);
            }
        }

        let mut source_use_count: HashMap<&str, usize> = HashMap::new();
        for (_, source) in pairs {
            *source_use_count.entry(source.as_str()).or_insert(0) += 1;
            if !axes.iter().any(|a| &a.label == source) {
                self.error(format!("`as [...]` mapping references unknown source axis '{source}'"), line, col);
            }
        }
        for axis in &axes {
            match source_use_count.get(axis.label.as_str()) {
                None | Some(0) => {
                    self.error(
                        format!("`as [...]` mapping is missing axis '{}' — every source axis must appear exactly once", axis.label),
                        line, col,
                    );
                }
                Some(1) => {}
                Some(n) => {
                    self.error(
                        format!("`as [...]` mapping references source axis '{}' {} times — every source axis must appear exactly once", axis.label, n),
                        line, col,
                    );
                }
            }
        }
    }

    // ── Struct / enum / ext ───────────────────────────────────────────────────

    fn check_struct(&mut self, s: &StructDecl) {
        for f in &s.fields {
            self.check_set_mut_constraint(&Some(f.ty.clone()), f.line, f.col);
        }
        for init in &s.inits { self.check_init(init); }
        for m in &s.methods { self.check_fn(m); }
        for m in &s.type_methods {
            self.push_scope();
            for p in &m.params { self.define_typed(&p.name, param_binding(p), p.ty.clone()); }
            for stmt in &m.body { self.check_stmt(stmt); }
            self.pop_scope();
        }
        for sd in &s.setters { self.check_set_decl(sd); }
        for conv in &s.conversions { self.check_as_decl(conv); }
    }

    fn check_enum(&mut self, e: &EnumDecl) {
        for v in &e.variants {
            for f in &v.fields {
                self.check_set_mut_constraint(&Some(f.ty.clone()), v.line, v.col);
            }
        }
        for m in &e.methods { self.check_fn(m); }
        // `type def`/`type req`/`type set` factory/static methods — mirrors
        // `check_struct`'s identical loop.
        for m in &e.type_methods {
            self.push_scope();
            for p in &m.params { self.define_typed(&p.name, param_binding(p), p.ty.clone()); }
            for stmt in &m.body { self.check_stmt(stmt); }
            self.pop_scope();
        }
        for sd in &e.setters { self.check_set_decl(sd); }
        for conv in &e.conversions { self.check_as_decl(conv); }
    }

    fn check_ext(&mut self, e: &ExtDecl) {
        for m in &e.methods { self.check_fn(m); }
        for sd in &e.setters { self.check_set_decl(sd); }
        for conv in &e.conversions { self.check_as_decl(conv); }
    }

    /// `init(...)` bodies were never walked by this checker at all — none of
    /// its rules (immutability, `mut`/`'shared`/`'static`/`'weak`, ...) applied
    /// inside a constructor, so `boring run` silently accepted an illegal
    /// mutation there and `boring build --emit-rust` transpiled it into valid,
    /// running Rust. Mirrors `check_fn`'s param-binding + body-walk shape;
    /// `InitParam` has no `rebindable` axis (see its own doc comment), so
    /// there is no `Var` binding kind to consider here, unlike `param_binding`.
    fn check_init(&mut self, init: &InitDecl) {
        self.push_scope();
        for p in &init.params {
            let kind = if p.mutable { BindingKind::Mut } else { BindingKind::Let };
            if p.mutable {
                self.check_qualifier_constraint(&kind, false, &p.ty, p.line, p.col);
            }
            self.check_set_mut_constraint(&p.ty, p.line, p.col);
            self.define_typed(&p.name, kind, p.ty.clone());
            if let Some(def) = &p.default { self.check_expr(def); }
        }
        for stmt in &init.body { self.check_stmt(stmt); }
        self.pop_scope();
    }

    /// `set name(param): body` — same gap as `check_init`: the body was never
    /// walked. `SetDecl` always has exactly one parameter with an explicit
    /// type, always by-value (no `mut`/`var` on a setter's own parameter).
    fn check_set_decl(&mut self, sd: &SetDecl) {
        self.push_scope();
        self.check_set_mut_constraint(&Some(sd.param_ty.clone()), sd.line, sd.col);
        self.define_typed(&sd.param_name, BindingKind::Let, Some(sd.param_ty.clone()));
        for stmt in &sd.body { self.check_stmt(stmt); }
        self.pop_scope();
    }

    /// `as Type: body` conversion — same gap as `check_init`/`check_set_decl`:
    /// the body was never walked. No parameters (the conversion body only
    /// sees `self`, already in scope via the enclosing struct/enum/ext).
    fn check_as_decl(&mut self, conv: &AsDecl) {
        for stmt in &conv.body { self.check_stmt(stmt); }
    }

    /// Trait default method bodies (`t.defaults`) — previously entirely
    /// unchecked (`Item::Trait(_) => {}` in `check_item`). Abstract
    /// signatures (`t.signatures`/`t.type_signatures`) have no body to walk.
    fn check_trait(&mut self, t: &TraitDecl) {
        for def in &t.defaults { self.check_fn(def); }
    }

    // ── Functions ─────────────────────────────────────────────────────────────

    fn check_fn(&mut self, f: &FnDecl) {
        self.push_scope();
        // `'static` provenance gate — authorized inside `main`'s body only, never a
        // nested fn (including one declared inside `main` via `Stmt::Fn`, which would
        // otherwise wrongly inherit the saved `true` below through this same call).
        let prev_static_site = self.in_authorized_static_site;
        self.in_authorized_static_site = f.name == "main";
        for p in &f.params {
            if p.mutable {
                self.check_qualifier_constraint(&BindingKind::Mut, false, &p.ty, p.line, p.col);
            }
            // `{mut T}` is illegal regardless of whether the parameter itself
            // is `mut` — the illegality lives on the Set's element type.
            self.check_set_mut_constraint(&p.ty, p.line, p.col);
            self.define_typed(&p.name, param_binding(p), p.ty.clone());
        }
        for stmt in &f.body { self.check_stmt(stmt); }
        self.in_authorized_static_site = prev_static_site;
        self.pop_scope();
    }

    // ── Statements ────────────────────────────────────────────────────────────

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let(s) => self.check_let_stmt(s),
            Stmt::LetDestructure(s) => self.check_let_destructure(s),
            Stmt::Expr(e)      => self.check_expr(e),
            Stmt::Return(r)    => { if let Some(v) = &r.value { self.check_expr(v); } }
            Stmt::Throw(t)     => { if let Some(v) = &t.value { self.check_expr(v); } }
            Stmt::If(s)        => self.check_if(s),
            Stmt::IfLet(s)     => self.check_if_let(s),
            Stmt::While(s)     => {
                self.check_expr(&s.condition);
                self.check_block(&s.body);
            }
            Stmt::WhileLet(s)  => {
                self.check_expr(&s.value);
                self.push_scope();
                self.define(&s.name, BindingKind::Let);
                self.check_block_in_current_scope(&s.body);
                self.pop_scope();
            }
            Stmt::DoWhile(s)   => {
                self.check_block(&s.body);
                self.check_expr(&s.condition);
            }
            Stmt::Loop(s)      => self.check_block(&s.body),
            Stmt::For(s)       => {
                self.check_expr(&s.iterable);
                self.push_scope();
                for v in &s.vars { self.define(v, BindingKind::Let); }
                self.check_block_in_current_scope(&s.body);
                self.pop_scope();
            }
            Stmt::Match(s)     => self.check_match_stmt(s),
            Stmt::Guard(s)     => {
                match &s.cond {
                    GuardCond::Expr(e)      => self.check_expr(e),
                    GuardCond::Clauses(cs)  => self.check_cond_clauses(cs),
                }
                self.check_block(&s.else_body);
            }
            Stmt::Try(s)       => {
                self.check_block(&s.body);
                for clause in &s.catch_clauses { self.check_block(&clause.body); }
            }
            Stmt::Defer(body)  => self.check_block(body),
            Stmt::Yield(e, _)  => self.check_expr(e),
            Stmt::Wait(e, _)   => self.check_expr(e),
            Stmt::Break(_, v)  => { if let Some(e) = v { self.check_expr(e); } }
            Stmt::Fn(f)        => self.check_fn(f),
            Stmt::Struct(s)    => self.check_struct(s),
            Stmt::Enum(e)      => self.check_enum(e),
            Stmt::Mod(m)       => { for i in &m.items { self.check_item(i); } }
            Stmt::Continue(_) | Stmt::Alias(_) | Stmt::Comment(_) => {}
            Stmt::KernelBlock(s) => { for stmt in &s.body { self.check_stmt(stmt); } }
            Stmt::With(s) => self.check_with_stmt(s),
        }
    }

    fn check_let_stmt(&mut self, s: &LetStmt) {
        self.check_static_provenance(&s.ty, s.value.as_ref(), s.line, s.col);
        self.check_qualifier_constraint(&s.binding, s.var_mut, &s.ty, s.line, s.col);
        self.check_tuple_mut_constraint(&s.binding, s.var_mut, &s.ty, &s.value, s.line, s.col);
        self.check_scalar_mut_constraint(&s.binding, s.var_mut, &s.ty, &s.value, s.line, s.col);
        self.check_set_mut_constraint(&s.ty, s.line, s.col);
        if let Some(v) = &s.value { self.check_expr(v); }
        // Labeled multi-dim array cross-label check — only when this `let` has
        // an explicit type annotation to check the initializer against.
        if let (Some(target_ty), Some(v)) = (&s.ty, &s.value) {
            if let Some(source_ty) = self.static_labeled_array_type(v) {
                self.check_label_compat(v, &source_ty, target_ty, s.line, s.col);
            }
        }
        self.define_let(&s.name, s.binding.clone(), s.ty.clone(), s.value.as_ref());
    }

    /// Tuple analogue of `define_let`'s interprocedural case: if `s.value` is a call
    /// to a `fn_returns_resident_tuple` function, a destructured binding at a
    /// resident tuple position stays resident only with an *explicit*
    /// `'gpu'unified`/`'gpu'global` opt-in annotation (e.g. `let
    /// ([float]'gpu'unified sa, ...) = mha_step_gpu(...)`) — the opposite default
    /// from the single-value interprocedural case (`let fc = linear_gpu(...)`
    /// stays resident with *no* annotation at all). Tuple destructuring predates
    /// this residency feature everywhere in real code, so every existing
    /// unannotated `let (a, b, c) = some_tuple_fn(...)` was written expecting a
    /// plain, already-materialized value it can index/iterate immediately — an
    /// opt-in default would silently change that. Confirmed against a real
    /// `cargo check` failure otherwise: `test_math_gpu.br`'s `let (step_out, ...) =
    /// mha_step_gpu(...)` indexes `step_out` right away with no annotation, which
    /// only compiles if the default is "materialize unless told otherwise".
    fn check_let_destructure(&mut self, s: &LetDestructureStmt) {
        self.check_expr(&s.value);
        let tuple_flags: Option<Vec<bool>> = match &s.value.kind {
            ExprKind::Call(callee, _) => match &callee.kind {
                ExprKind::Var(fn_name) => self.fn_returns_resident_tuple.get(fn_name).cloned(),
                _ => None,
            },
            _ => None,
        };
        // Corresponding tuple-literal element, when the RHS is a literal tuple —
        // lets each slot's `mut` constraints see an inferred-type value the same
        // way a plain `let_stmt`'s `check_scalar_mut_constraint`/
        // `check_tuple_mut_constraint` already do for `mut t = (1, 2)`.
        let literal_elems: Option<&Vec<Expr>> = match &s.value.kind {
            ExprKind::Tuple(elems) => Some(elems),
            _ => None,
        };
        for (i, b) in s.bindings.iter().enumerate() {
            if b.bare_unmarked_after_keyworded_sibling {
                self.warning(
                    format!(
                        "`{}` is unmarked in a bare destructure right after a differently-keyworded slot — it defaults to plain `let`, not the previous slot's keyword; write it explicitly if that's not what you meant",
                        b.name,
                    ),
                    s.line, s.col,
                );
            }
            if b.name != "_" {
                let elem_value = literal_elems.and_then(|elems| elems.get(i)).cloned();
                self.check_qualifier_constraint(&b.binding, b.var_mut, &b.ty, s.line, s.col);
                self.check_tuple_mut_constraint(&b.binding, b.var_mut, &b.ty, &elem_value, s.line, s.col);
                self.check_scalar_mut_constraint(&b.binding, b.var_mut, &b.ty, &elem_value, s.line, s.col);
                self.check_set_mut_constraint(&b.ty, s.line, s.col);
            }
            if b.name == "_" { continue; }
            let position_resident = tuple_flags.as_ref().and_then(|f| f.get(i).copied()).unwrap_or(false);
            let has_explicit_resident_ty = b.ty.as_ref().map(|t| t.gpu_resident_qual().is_some()).unwrap_or(false);
            let resident_from_field = position_resident && has_explicit_resident_ty;
            if let Some(scope) = self.scopes.last_mut() {
                // Each slot's own resolved binding, not the statement's overall
                // one (they can now differ per-element — docs/book.md
                // §4).
                scope.insert(b.name.clone(), Binding {
                    kind: b.binding.clone(),
                    ty: b.ty.clone(),
                    resident_from_field,
                    kernel_type: None,
                });
            }
        }
    }

    fn check_if(&mut self, s: &IfStmt) {
        for (cond, body) in &s.branches {
            self.check_expr(cond);
            self.check_block(body);
        }
        if let Some(body) = &s.else_body { self.check_block(body); }
    }

    fn check_if_let(&mut self, s: &IfLetStmt) {
        self.push_scope();
        self.check_cond_clauses(&s.clauses);
        self.check_block_in_current_scope(&s.then_body);
        self.pop_scope();
        for branch in &s.elif_branches {
            self.push_scope();
            self.check_cond_clauses(&branch.clauses);
            self.check_block_in_current_scope(&branch.body);
            self.pop_scope();
        }
        if let Some(body) = &s.else_body { self.check_block(body); }
    }

    fn check_cond_clauses(&mut self, clauses: &[CondClause]) {
        for clause in clauses {
            match clause {
                CondClause::Expr(e)       => self.check_expr(e),
                CondClause::Let(name, e)  => {
                    self.check_expr(e);
                    self.define(name, BindingKind::Let);
                }
                CondClause::LetPat(_, e)  => self.check_expr(e),
            }
        }
    }

    fn check_match_stmt(&mut self, s: &MatchStmt) {
        self.check_expr(&s.subject);
        for arm in &s.arms {
            if let Some(g) = &arm.guard { self.check_expr(g); }
            self.push_scope();
            for pat in &arm.patterns {
                bind_in_pattern(pat, arm.line, arm.col, &mut |name, _line, _col| {
                    self.define(name, BindingKind::Let);
                });
            }
            match &arm.body {
                MatchBody::Expr(e)      => self.check_expr(e),
                MatchBody::Block(stmts) => self.check_block_in_current_scope(stmts),
            }
            self.pop_scope();
        }
    }

    // ── Block helpers ─────────────────────────────────────────────────────────

    fn check_block(&mut self, stmts: &[Stmt]) {
        self.push_scope();
        for stmt in stmts { self.check_stmt(stmt); }
        self.pop_scope();
    }

    fn check_block_in_current_scope(&mut self, stmts: &[Stmt]) {
        for stmt in stmts { self.check_stmt(stmt); }
    }

    // ── Expressions ───────────────────────────────────────────────────────────

    fn check_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            // ── Assignment — the core immutability check ──────────────────
            ExprKind::Assign(lhs, rhs) => {
                self.check_expr(rhs);
                self.check_assign_target(lhs, expr.line, expr.col);
                // Still recurse into lhs for nested expressions (e.g. index sub-expr).
                match &lhs.kind {
                    ExprKind::Var(_) => {}
                    _ => self.check_expr(lhs),
                }
                // Labeled multi-dim array cross-label check — best-effort, only
                // fires when the target `Var`'s declared type is statically known.
                if let ExprKind::Var(name) = &lhs.kind {
                    let target_ty = self.lookup(name).and_then(|b| b.ty.clone());
                    if let Some(target_ty) = target_ty {
                        if let Some(source_ty) = self.static_labeled_array_type(rhs) {
                            self.check_label_compat(rhs, &source_ty, &target_ty, expr.line, expr.col);
                        }
                    }
                }
            }
            ExprKind::QuestionAssign(lhs, rhs) => {
                // `?=` is always legal (lazy initialisation / nil-coalescing).
                self.check_expr(rhs);
                self.check_expr(lhs);
            }

            // ── Recurse into sub-expressions ──────────────────────────────
            ExprKind::BinOp(_, l, r) => { self.check_expr(l); self.check_expr(r); }
            ExprKind::UnaryOp(_, e)  => self.check_expr(e),
            ExprKind::Field(e, _) => self.check_expr(e),
            ExprKind::OptionalField(e, _) => self.check_expr(e),
            ExprKind::Index(obj, idx) => {
                self.check_expr(obj);
                self.check_expr(idx);
            }
            ExprKind::LabeledIndex(obj, args) => {
                self.check_expr(obj);
                for a in args { self.check_expr(&a.value); }
            }
            ExprKind::Call(callee, args) => {
                self.check_expr(callee);
                // `k(block = ...)` inside a `kernel:` block is a dispatch call, not a
                // constructor call (a constructor's callee is the kernel TYPE name,
                // e.g. `Scale(data)`; only an actual instance VARIABLE bound to a
                // known kernel type can reach here as `callee`) -- see
                // `check_kernel_dispatch_qualifier`'s doc for why this needs checking.
                // Note: `ExprKind::KernelLaunch` looks like the obvious place for this
                // (and is also checked there, defensively), but the parser never
                // actually constructs that node for the `kernel:` block's `k(...)`
                // call sites -- they parse as an ordinary `Call`, confirmed by
                // grepping every parser file for `KernelLaunch` construction (none).
                self.check_kernel_dispatch_qualifier(callee, expr.line, expr.col);
                // A resident value passed as a bare argument at a position the callee
                // is known to consume residently (`fn_gpu_arg_params`, populated by
                // `scan_fn_gpu_arg_params`) is legal without `with` first — that's the
                // whole point of the interprocedural carve-out (docs/scoped-access-
                // blocks.md's "Kernel Constructor Interaction"). Every other
                // host-materializing use of a resident `Var` still bottoms out at the
                // `ExprKind::Var` leaf below and gets flagged as usual.
                let gpu_arg_positions: Option<Vec<bool>> = match &callee.kind {
                    ExprKind::Var(n) => self.fn_gpu_arg_params.get(n.as_str()).cloned(),
                    _ => None,
                };
                for (i, a) in args.iter().enumerate() {
                    let carved_out = matches!(&a.value.kind, ExprKind::Var(_))
                        && gpu_arg_positions.as_ref().map(|flags| flags.get(i).copied().unwrap_or(false)).unwrap_or(false);
                    if !carved_out {
                        self.check_expr(&a.value);
                    }
                    // Labeled multi-dim array cross-label check (best-effort,
                    // positional args only — a labeled/named call argument isn't
                    // matched back to its declared parameter position here).
                    // See docs/array-multidim-proposal.md.
                    if a.label.is_none() {
                        if let ExprKind::Var(fn_name) = &callee.kind {
                            let target_ty: Option<Type> = self.fn_param_types.get(fn_name.as_str())
                                .and_then(|types| types.get(i))
                                .and_then(|t| t.clone());
                            if let Some(target_ty) = &target_ty {
                                if let Some(source_ty) = self.static_labeled_array_type(&a.value) {
                                    self.check_label_compat(&a.value, &source_ty, target_ty, expr.line, expr.col);
                                }
                            }
                            self.check_static_arg_provenance(target_ty.as_ref(), &a.value, expr.line, expr.col);
                        }
                    }
                }
            }
            ExprKind::MethodCall(recv, _, args) | ExprKind::OptionalMethodCall(recv, _, args) => {
                self.check_expr(recv);
                for a in args { self.check_expr(&a.value); }
            }
            ExprKind::GenericCall(callee, _, args) => {
                self.check_expr(callee);
                for a in args { self.check_expr(&a.value); }
            }
            ExprKind::Pipe(lhs, _, args) => {
                self.check_expr(lhs);
                for a in args { self.check_expr(&a.value); }
            }
            ExprKind::New { ctor, arena } => {
                self.check_expr(ctor);
                if let Some(a) = arena { self.check_expr(a); }
            }
            ExprKind::Cast(e, _)  => self.check_expr(e),
            ExprKind::Else(e, d)  => { self.check_expr(e); self.check_expr(d); }
            ExprKind::TryElse(e, d) => { self.check_expr(e); self.check_expr(d); }
            ExprKind::TryElseBlock(body, els) => {
                self.check_block(body);
                self.check_block(els);
            }
            ExprKind::Array(elems) => { for e in elems { self.check_expr(e); } }
            ExprKind::ArrayFill { value, count } => {
                self.check_expr(value); self.check_expr(count);
            }
            ExprKind::ArrayAlloc { count } => { self.check_expr(count); }
            ExprKind::ArrayComp { expr, var, count } => {
                self.check_expr(count);
                self.push_scope();
                self.define(var, BindingKind::Let);
                self.check_expr(expr);
                self.pop_scope();
            }
            ExprKind::ArrayCompIter { expr, var, iter } => {
                self.check_expr(iter);
                self.push_scope();
                self.define(var, BindingKind::Let);
                self.check_expr(expr);
                self.pop_scope();
            }
            // Full shape/cross-label validation lands in the checker's labeled-array
            // pass (docs/array-multidim-proposal.md); this recurses so undefined-var/
            // GPU-opacity checks still see every sub-expression in the meantime.
            ExprKind::LabeledArrayComp { expr, clauses } => {
                self.push_scope();
                for (var, count) in clauses {
                    self.check_expr(count);
                    self.define(var, BindingKind::Let);
                }
                self.check_expr(expr);
                self.pop_scope();
            }
            ExprKind::RelabelCast(inner, pairs) => {
                self.check_expr(inner);
                self.check_relabel_cast(inner, pairs, expr.line, expr.col);
            }
            ExprKind::Tuple(elems) => { for e in elems { self.check_expr(e); } }
            ExprKind::Dict(pairs)  => {
                for (k, v) in pairs { self.check_expr(k); self.check_expr(v); }
            }
            ExprKind::Set(elems)   => { for e in elems { self.check_expr(e); } }
            ExprKind::Range { start, end, .. } => { self.check_expr(start); self.check_expr(end); }
            ExprKind::SliceRange { start, end, .. } => {
                if let Some(s) = start { self.check_expr(s); }
                if let Some(e) = end   { self.check_expr(e); }
            }
            ExprKind::StringInterp(segs) => {
                for seg in segs {
                    if let StringSegment::Expr(e) | StringSegment::FormattedExpr(e, _) = seg {
                        self.check_expr(e);
                    }
                }
            }
            ExprKind::If(s)    => self.check_if(s),
            ExprKind::Match(s) => self.check_match_stmt(s),
            ExprKind::Block(stmts) | ExprKind::Do(stmts) => self.check_block(stmts),
            ExprKind::Loop(s)  => self.check_block(&s.body),
            ExprKind::Task(e)  => self.check_expr(e),
            ExprKind::TaskWithTimeout(dur, e) => { self.check_expr(dur); self.check_expr(e); }
            ExprKind::JoinAll(exprs) => { for e in exprs { self.check_expr(e); } }
            ExprKind::KernelLaunch { kernel, config } => {
                self.check_expr(kernel);
                self.check_kernel_dispatch_qualifier(kernel, expr.line, expr.col);
                if let Some(b) = &config.block { self.check_expr(b); }
                if let Some(g) = &config.grid  { self.check_expr(g); }
            }
            ExprKind::Closure(params, _, body, _, _) => {
                self.push_scope();
                for p in params { self.define_typed(&p.name, param_binding(p), p.ty.clone()); }
                match body {
                    ClosureBody::Expr(e)      => self.check_expr(e),
                    ClosureBody::Block(stmts) => self.check_block_in_current_scope(stmts),
                }
                self.pop_scope();
            }
            ExprKind::MacroCall { args, .. } => {
                for a in args { self.check_expr(a); }
            }

            // A bare variable reference is where every host-materializing use of a
            // GPU-resident name bottoms out — indexing, `.length`, iteration, string
            // interpolation, an argument in a call — since check_expr always recurses
            // down to this leaf for each of those. One check here covers all of them.
            ExprKind::Var(name) => self.check_gpu_opacity(name, expr.line, expr.col),

            // Leaves — nothing to recurse into.
            ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_)
            | ExprKind::Str(_) | ExprKind::Bool(_) | ExprKind::Nil
            | ExprKind::Void | ExprKind::DotIdent(_) => {}
        }
    }

    // ── Immutability check on assignment targets ───────────────────────────────

    fn check_assign_target(&mut self, lhs: &Expr, assign_line: usize, assign_col: usize) {
        if self.kernel_dispatch_only { return; }
        if let ExprKind::Var(name) = &lhs.kind {
            // `_` is the discard wildcard — never an error as assignment target.
            if name == "_" { return; }
            if let Some(binding) = self.lookup(name) {
                match binding.kind {
                    BindingKind::Let => {
                        self.error(
                            format!("cannot assign to `{name}`: declared as `let` (immutable)"),
                            assign_line, assign_col,
                        );
                    }
                    BindingKind::Lazy => {
                        self.error(
                            format!("cannot assign to `{name}` with `=`: `lazy` bindings are written with `?=`"),
                            assign_line, assign_col,
                        );
                    }
                    // `mut` is never rebindable — docs/book.md
                    // decided this with no exception (retiring the historical
                    // "mut ≡ var for scalars" shortcut). Previously this arm was
                    // `Mut | Var => {}`, silently allowing reassignment through a
                    // `mut` binding for any type, not just scalars — a bug this
                    // document's model corrects.
                    BindingKind::Mut => {
                        self.error(
                            format!("cannot assign to `{name}`: declared as `mut` (fixed) — use `var mut` for a reassignable, content-mutable binding"),
                            assign_line, assign_col,
                        );
                    }
                    BindingKind::Var => {}
                }
            }
            // Unknown variable — undefined-var check belongs to the interpreter/transpiler.
        }
        // Field and index targets are not checked here: mutability of those
        // requires type information not yet available at this pass.
    }

    // ── `with` scoped-access blocks ─────────────────────────────────────────────
    // See docs/scoped-access-blocks.md. Two things are checked here (both target-
    // independent, so they fire under `boring run` too, not just `boring build`):
    //   - nesting a `with` block on the same name inside itself (double-acquire);
    //   - using a `'gpu'unified`/`'gpu'global` value's host-materializing operations
    //     (indexing, `.length`, iteration, string interpolation) outside a `with`
    //     wrapper that opens it.
    // The two-step read/write access scan itself (`with_block_mutates` in ast::mod)
    // doesn't produce an error here — nothing about a block's chosen access level is
    // ever illegal — it's consumed by the transpiler at `with` codegen time to pick
    // map-for-read vs map-for-read-write / a shared vs exclusive lock.

    fn check_with_stmt(&mut self, s: &WithStmt) {
        let mut newly_opened = Vec::new();
        for name in &s.names {
            if self.open_with_names.contains(name.as_str()) {
                if !self.kernel_dispatch_only {
                    self.error(
                        format!("nested `with {name}:` block on the same name is not allowed (double-acquire)"),
                        s.line, s.col,
                    );
                }
            } else {
                self.open_with_names.insert(name.clone());
                newly_opened.push(name.clone());
            }
        }
        self.check_block(&s.body);
        for name in &newly_opened { self.open_with_names.remove(name); }
    }

    /// If `name` is a `'gpu'unified`/`'gpu'global` binding sourced from a bare
    /// kernel-field read (`resident_from_field` — see `Binding`) and isn't currently
    /// open in an enclosing `with` block, records a compile error: any use at all
    /// (indexing, `.length`, iteration, string interpolation, passed as an argument,
    /// ...) requires a `with` wrapper first. A `'gpu'unified`/`'gpu'global` binding
    /// that is just a plain array (not sourced from a kernel field) is unrestricted —
    /// see `examples/saxpy.br`.
    fn check_gpu_opacity(&mut self, name: &str, line: usize, col: usize) {
        if self.kernel_dispatch_only { return; }
        if self.open_with_names.contains(name) { return; }
        let Some(binding) = self.lookup(name) else { return };
        if !binding.resident_from_field { return; }
        let Some(ty) = &binding.ty else { return };
        if ty.gpu_resident_qual().is_some() {
            self.error(
                format!("`{name}` is GPU-resident (sourced from a kernel field) and cannot be used outside a `with {name}:` block"),
                line, col,
            );
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Does `expr` look like a bare kernel-field read (`k.field`)? Purely syntactic —
/// the checker doesn't track which names are kernel instances (that's transpiler
/// state); the shape alone is enough to distinguish "sourced from a kernel field"
/// from "a plain array literal/expression", which is what `resident_from_field`
/// needs. See `Binding::resident_from_field`.
fn is_kernel_field_read(expr: &Expr) -> bool {
    matches!(&expr.kind, ExprKind::Field(obj, _) if matches!(&obj.kind, ExprKind::Var(_)))
}

/// Which field a kernel's `init` assigns a given parameter name to (`field = param`,
/// the only pattern kernel-constructor codegen understands — mirrors the
/// transpiler's own `emit_kernel::kernel_param_to_field_map`, duplicated here rather
/// than shared since the checker doesn't depend on the transpiler).
fn kernel_init_field_for_param<'a>(decl: &'a KernelDecl, param_name: &str) -> Option<&'a str> {
    let init = decl.inits.first()?;
    for stmt in &init.body {
        if let Stmt::Expr(e) = stmt {
            if let ExprKind::Assign(lhs, rhs) = &e.kind {
                if let (ExprKind::Var(field), ExprKind::Var(param)) = (&lhs.kind, &rhs.kind) {
                    if param == param_name { return Some(field.as_str()); }
                }
            }
        }
    }
    None
}

fn param_binding(p: &Param) -> BindingKind {
    if p.rebindable { BindingKind::Var }
    else if p.mutable { BindingKind::Mut }
    else { BindingKind::Let }
}

fn bind_in_pattern(pat: &Pattern, line: usize, col: usize, f: &mut impl FnMut(&str, usize, usize)) {
    match pat {
        Pattern::Bind(name)       => f(name, line, col),
        Pattern::Some(inner)      => bind_in_pattern(inner, line, col, f),
        Pattern::Variant(_, sub)  => { for p in sub { bind_in_pattern(p, line, col, f); } }
        Pattern::Tuple(sub)       => { for p in sub { bind_in_pattern(p, line, col, f); } }
        Pattern::Wildcard | Pattern::None | Pattern::Lit(_) => {}
    }
}

#[cfg(test)]
mod tuple_mut_tests {
    use crate::lexer::lex;
    use crate::parser::parse;

    fn errors_for(src: &str) -> Vec<String> {
        let tokens = lex(src).expect("lex error");
        let program = parse(tokens).expect("parse error");
        super::check(&program).errors.into_iter().map(|e| e.message).collect()
    }

    #[test]
    fn mut_on_typed_tuple_variable_is_rejected() {
        let src = "def main():\n    mut (int, int) t = (1, 2)\n    print t.0\n";
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("tuple") && e.contains("mut")), "expected a tuple/mut rejection, got {errs:?}");
    }

    #[test]
    fn let_on_typed_tuple_variable_is_fine() {
        let src = "def main():\n    let (int, int) t = (1, 2)\n    print t.0\n";
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn var_on_typed_tuple_variable_is_fine() {
        let src = "def main():\n    var (int, int) t = (1, 2)\n    t = (9, 9)\n    print t.0\n";
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn mut_on_tuple_destructure_is_unaffected() {
        // `mut (a, b) = t` applies `mut` to the two extracted variables
        // individually, not to the tuple as a whole — that part is still legal.
        // Retired by docs/book.md ("no exceptions"): `a`/`b` are
        // `int` here, and `mut` is never rebindable regardless of type (the
        // historical "mut ≡ var for scalars" shortcut this test used to pin
        // down) — `a = 5` is now a compiler-flagged error, exactly like a plain
        // `mut a = 1; a = 5` local binding already was before this document.
        let src = "def main():\n    mut (a, b) = (1, 2)\n    a = 5\n    print a\n";
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("mut") && e.contains("a")), "expected a reassign-to-mut rejection, got {errs:?}");
    }

    #[test]
    fn mut_on_inferred_tuple_type_is_rejected() {
        // No explicit `(int, int)` annotation — the tuple type is only
        // inferred from the literal initializer. The constraint must still
        // fire; only checking the explicit annotation would let this slip.
        let src = "def main():\n    mut t = (1, 2)\n    print t.0\n";
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("tuple") && e.contains("mut")), "expected a tuple/mut rejection, got {errs:?}");
    }

    #[test]
    fn let_on_inferred_tuple_type_is_fine() {
        let src = "def main():\n    let t = (1, 2)\n    print t.0\n";
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }
}

#[cfg(test)]
mod set_mut_tests {
    use crate::lexer::lex;
    use crate::parser::parse;

    fn errors_for(src: &str) -> Vec<String> {
        let tokens = lex(src).expect("lex error");
        let program = parse(tokens).expect("parse error");
        super::check(&program).errors.into_iter().map(|e| e.message).collect()
    }

    #[test]
    fn mut_set_element_type_is_rejected_on_a_let_binding() {
        // Illegal regardless of the outer binding's own mutability — `pts` is
        // a plain `let` here, but the element type itself is `mut`.
        let src = "struct Point:\n    var float x\ndef main():\n    let {mut Point} pts = {}\n    print \"ok\"\n";
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("set") && e.contains("mut")), "expected a set/mut rejection, got {errs:?}");
    }

    #[test]
    fn mut_on_the_set_itself_is_unaffected() {
        // `mut {T}`/`var mut {T}` (structural mutation of the set) is a
        // different axis and must stay legal.
        let src = "struct Point:\n    var float x\ndef main():\n    var mut {Point} pts = {}\n    pts.add(Point(1.0))\n    print \"ok\"\n";
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn mut_set_element_type_is_rejected_when_nested_in_an_array() {
        let src = "struct Point:\n    var float x\ndef main():\n    let [{mut Point}] arr = []\n    print \"ok\"\n";
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("set") && e.contains("mut")), "expected a set/mut rejection, got {errs:?}");
    }

    #[test]
    fn mut_set_element_type_is_rejected_on_a_struct_field() {
        let src = "struct Point:\n    var float x\nstruct Holder:\n    {mut Point} pts\ndef main():\n    print \"ok\"\n";
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("set") && e.contains("mut")), "expected a set/mut rejection, got {errs:?}");
    }

    #[test]
    fn mut_set_element_type_is_rejected_on_a_parameter() {
        let src = "struct Point:\n    var float x\ndef takes({mut Point} pts):\n    print \"ok\"\ndef main():\n    takes({})\n";
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("set") && e.contains("mut")), "expected a set/mut rejection, got {errs:?}");
    }
}

#[cfg(test)]
mod with_stmt_tests {
    use crate::lexer::lex;
    use crate::parser::parse;

    fn errors_for(src: &str) -> Vec<String> {
        let tokens = lex(src).expect("lex error");
        let program = parse(tokens).expect("parse error");
        super::check(&program).errors.into_iter().map(|e| e.message).collect()
    }

    #[test]
    fn with_block_grants_indexing_of_gpu_resident_value() {
        let src = r#"
let k = Kernel()
let [float]'gpu'unified fc = k.y
with fc:
    print "{fc[0]}"
"#;
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn indexing_gpu_resident_value_outside_with_is_an_error() {
        let src = r#"
let k = Kernel()
let [float]'gpu'unified fc = k.y
print "{fc[0]}"
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("with fc:")), "expected an opacity error, got {errs:?}");
    }

    #[test]
    fn length_of_gpu_resident_value_outside_with_is_an_error() {
        let src = r#"
let k = Kernel()
let [float]'gpu'global tok_emb = k.embeddings
let int n = tok_emb.length
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("tok_emb")), "expected an opacity error, got {errs:?}");
    }

    #[test]
    fn iterating_gpu_resident_value_outside_with_is_an_error() {
        let src = r#"
let k = Kernel()
let [float]'gpu'unified fc = k.y
for x in fc:
    print "{x}"
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("fc")), "expected an opacity error, got {errs:?}");
    }

    #[test]
    fn string_interpolation_of_gpu_resident_value_outside_with_is_an_error() {
        let src = r#"
let k = Kernel()
let [float]'gpu'unified fc = k.y
print "value: {fc}"
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("fc")), "expected an opacity error, got {errs:?}");
    }

    #[test]
    fn nested_with_on_same_name_is_double_acquire_error() {
        let src = r#"
let k = Kernel()
let [float]'gpu'unified fc = k.y
with fc:
    with fc:
        print "{fc[0]}"
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("double-acquire")), "expected a double-acquire error, got {errs:?}");
    }

    #[test]
    fn nested_with_on_different_names_is_fine() {
        let src = r#"
let k = Kernel()
let [float]'gpu'unified fc = k.y
let [float]'gpu'unified act = k.z
with fc:
    with act:
        print "{fc[0]} {act[0]}"
"#;
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn plain_gpu_qualified_array_literal_is_unrestricted() {
        // examples/saxpy.br's real pattern: a 'gpu'unified array literal, not sourced
        // from a kernel field, is just a plain host array — freely indexed/assigned
        // with no `with` wrapper required anywhere.
        let src = r#"
var [float]'gpu'unified x = [0.0, 0.0, 0.0]
x[0] = 1.0
print "{x[0]}"
for v in x:
    print "{v}"
"#;
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    // ── Inferred GPU residency (no explicit 'gpu'unified/'gpu'global annotation) ──

    const KERNEL_DECL: &str = r#"
kernel Saxpy:
    mut [float]'unified y
    def ():
        y[0] = 1.0
"#;

    #[test]
    fn unannotated_kernel_field_read_infers_residency_and_needs_with() {
        let src = format!("{KERNEL_DECL}\nlet k = Saxpy()\nlet result = k.y\nprint \"{{result[0]}}\"\n");
        let errs = errors_for(&src);
        assert!(errs.iter().any(|e| e.contains("result")), "expected an inferred opacity error, got {errs:?}");
    }

    #[test]
    fn unannotated_kernel_field_read_with_block_is_fine() {
        let src = format!("{KERNEL_DECL}\nlet k = Saxpy()\nlet result = k.y\nwith result:\n    print \"{{result[0]}}\"\n");
        let errs = errors_for(&src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn unannotated_read_of_non_kernel_field_is_not_gpu_resident() {
        // `k.y` only infers residency when `k` is actually bound to a known kernel
        // type. An ordinary struct field read is unaffected.
        let src = r#"
struct Point:
    float y

let k = Point(1.0)
let result = k.y
print "{result}"
"#;
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    // ── Kernel dispatch qualifier rejection ────────────────────────────────────

    #[test]
    fn shared_qualified_kernel_instance_cannot_be_dispatched() {
        let src = format!("{KERNEL_DECL}\nlet k'shared = Saxpy()\nkernel:\n    k(block = 256)\n");
        let errs = errors_for(&src);
        assert!(errs.iter().any(|e| e.contains("'shared")), "expected a qualifier-rejection error, got {errs:?}");
    }

    #[test]
    fn actor_qualified_kernel_instance_cannot_be_dispatched() {
        let src = format!("{KERNEL_DECL}\nlet k'actor = Saxpy()\nkernel:\n    k(block = 256)\n");
        let errs = errors_for(&src);
        assert!(errs.iter().any(|e| e.contains("'actor")), "expected a qualifier-rejection error, got {errs:?}");
    }

    #[test]
    fn guard_qualified_kernel_instance_cannot_be_dispatched() {
        let src = format!("{KERNEL_DECL}\nlet k'guard = Saxpy()\nkernel:\n    k(block = 256)\n");
        let errs = errors_for(&src);
        assert!(errs.iter().any(|e| e.contains("'guard")), "expected a qualifier-rejection error, got {errs:?}");
    }

    #[test]
    fn unqualified_kernel_instance_dispatch_is_fine() {
        let src = format!("{KERNEL_DECL}\nmut k = Saxpy()\nkernel:\n    k(block = 256)\n");
        let errs = errors_for(&src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

}

// ── Labeled multi-dimensional arrays (docs/array-multidim-proposal.md) ─────────
// Separate from `with_stmt_tests` — new syntax, new rules, no shared fixtures.

#[cfg(test)]
mod labeled_array_tests {
    use crate::lexer::lex;
    use crate::parser::parse;

    fn errors_for(src: &str) -> Vec<String> {
        let tokens = lex(src).expect("lex error");
        let program = parse(tokens).expect("parse error");
        super::check(&program).errors.into_iter().map(|e| e.message).collect()
    }

    // ── Shape validation ──────────────────────────────────────────────────

    #[test]
    fn valid_dynamic_and_fixed_kernel_fields_are_accepted() {
        let src = r#"
kernel K:
    mut [float, width, height]'unified a
    mut [float, width = 16, height = 16]'actor b
    def ():
        pass
"#;
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn duplicate_axis_label_is_rejected() {
        let src = r#"
kernel K:
    mut [float, width, width]'unified a
    def ():
        pass
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("duplicate axis label")), "expected a duplicate-label error, got {errs:?}");
    }

    #[test]
    fn more_than_three_axes_is_rejected_for_kernel_fields() {
        let src = r#"
kernel K:
    mut [float, a, b, c, d]'unified x
    def ():
        pass
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("at most 3 axes")), "expected an axis-count-cap error, got {errs:?}");
    }

    // ── Cross-label compatibility ─────────────────────────────────────────

    #[test]
    fn matching_labels_pass_through_with_no_error() {
        let src = r#"
def use_grid([float, width, height] grid):
    pass

let [float, width, height] a
use_grid(a)
"#;
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn mismatched_labels_as_a_call_argument_is_rejected() {
        let src = r#"
def use_grid([float, line, column] grid):
    pass

let [float, width, height] a
use_grid(a)
"#;
        let errs = errors_for(src);
        assert!(
            errs.iter().any(|e| e.contains("width") && e.contains("line") && e.contains("as [")),
            "expected a cross-label error suggesting `as [...]`, got {errs:?}"
        );
    }

    #[test]
    fn relabel_cast_at_a_call_site_silences_the_cross_label_error() {
        let src = r#"
def use_grid([float, line, column] grid):
    pass

let [float, width, height] a
use_grid(a as [line = width, column = height])
"#;
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn mismatched_labels_in_assignment_is_rejected() {
        let src = r#"
let [float, width, height] a
var [float, line, column] b
b = a
"#;
        let errs = errors_for(src);
        assert!(
            errs.iter().any(|e| e.contains("width") && e.contains("line")),
            "expected a cross-label error, got {errs:?}"
        );
    }

    #[test]
    fn matching_labels_in_assignment_is_fine() {
        let src = r#"
let [float, width, height] a
var [float, width, height] b
b = a
"#;
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn mismatched_labels_in_let_initializer_is_rejected() {
        let src = r#"
let [float, width, height] a
let [float, line, column] b = a
"#;
        let errs = errors_for(src);
        assert!(
            errs.iter().any(|e| e.contains("width") && e.contains("line")),
            "expected a cross-label error, got {errs:?}"
        );
    }

    // ── `as [...]` bijection completeness ──────────────────────────────────

    #[test]
    fn complete_relabel_mapping_is_accepted() {
        let src = r#"
let [float, width, height, depth] a
let b = a as [x = width, y = height, z = depth]
"#;
        let errs = errors_for(src);
        assert!(errs.is_empty(), "expected no errors, got {errs:?}");
    }

    #[test]
    fn relabel_mapping_missing_an_axis_is_rejected() {
        let src = r#"
let [float, width, height, depth] a
let b = a as [x = width, y = height]
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("missing axis 'depth'")), "expected a missing-axis error, got {errs:?}");
    }

    #[test]
    fn relabel_mapping_with_unknown_source_axis_is_rejected() {
        let src = r#"
let [float, width, height] a
let b = a as [x = width, y = height, z = bogus]
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("unknown source axis 'bogus'")), "expected an unknown-axis error, got {errs:?}");
    }

    #[test]
    fn relabel_mapping_with_duplicated_source_axis_is_rejected() {
        let src = r#"
let [float, width, height] a
let b = a as [x = width, y = width]
"#;
        let errs = errors_for(src);
        assert!(
            errs.iter().any(|e| e.contains("missing axis 'height'"))
                && errs.iter().any(|e| e.contains("2 times")),
            "expected both a missing-axis and a duplicated-source-axis error, got {errs:?}"
        );
    }

    #[test]
    fn relabel_mapping_with_duplicated_target_axis_is_rejected() {
        let src = r#"
let [float, width, height] a
let b = a as [x = width, x = height]
"#;
        let errs = errors_for(src);
        assert!(errs.iter().any(|e| e.contains("duplicate target axis 'x'")), "expected a duplicate-target error, got {errs:?}");
    }
}
