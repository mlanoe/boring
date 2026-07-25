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

    // ── Qualifier constraint: `mut 'shared` ───────────────────────────────────

    fn check_qualifier_constraint(&mut self, binding: &BindingKind, ty: &Option<Type>, line: usize, col: usize) {
        if !matches!(binding, BindingKind::Mut) { return; }
        let Some(ty) = ty else { return };
        if self.type_has_shared(ty) {
            self.error(
                "cannot combine `mut` with `'shared`: shared references are immutable by design; use `'actor` for interior mutability",
                line, col,
            );
        }
    }

    fn type_has_shared(&self, ty: &Type) -> bool {
        match ty {
            Type::Qualified(_, OwnerQual::Shared) => true,
            Type::Qualified(inner, _) => self.type_has_shared(inner),
            Type::Optional(inner) | Type::Array(inner) | Type::Dyn(inner) | Type::Impl(inner) => {
                self.type_has_shared(inner)
            }
            _ => false,
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
            Item::Use(_) | Item::Alias(_) | Item::Trait(_) | Item::Kernel(_) => {}
        }
    }

    // ── Struct / enum / ext ───────────────────────────────────────────────────

    fn check_struct(&mut self, s: &StructDecl) {
        for m in &s.methods { self.check_fn(m); }
        for m in &s.type_methods {
            self.push_scope();
            for p in &m.params { self.define_typed(&p.name, param_binding(p), p.ty.clone()); }
            for stmt in &m.body { self.check_stmt(stmt); }
            self.pop_scope();
        }
    }

    fn check_enum(&mut self, e: &EnumDecl) {
        for m in &e.methods { self.check_fn(m); }
    }

    fn check_ext(&mut self, e: &ExtDecl) {
        for m in &e.methods { self.check_fn(m); }
    }

    // ── Functions ─────────────────────────────────────────────────────────────

    fn check_fn(&mut self, f: &FnDecl) {
        self.push_scope();
        for p in &f.params {
            if p.mutable {
                self.check_qualifier_constraint(&BindingKind::Mut, &p.ty, p.line, p.col);
            }
            self.define_typed(&p.name, param_binding(p), p.ty.clone());
        }
        for stmt in &f.body { self.check_stmt(stmt); }
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
        self.check_qualifier_constraint(&s.binding, &s.ty, s.line, s.col);
        if let Some(v) = &s.value { self.check_expr(v); }
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
        for (i, b) in s.bindings.iter().enumerate() {
            if b.name == "_" { continue; }
            let position_resident = tuple_flags.as_ref().and_then(|f| f.get(i).copied()).unwrap_or(false);
            let has_explicit_resident_ty = b.ty.as_ref().map(|t| t.gpu_resident_qual().is_some()).unwrap_or(false);
            let resident_from_field = position_resident && has_explicit_resident_ty;
            if let Some(scope) = self.scopes.last_mut() {
                scope.insert(b.name.clone(), Binding {
                    kind: s.binding.clone(),
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
            ExprKind::Call(callee, args) => {
                self.check_expr(callee);
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
            ExprKind::Int(_) | ExprKind::Float(_)
            | ExprKind::Str(_) | ExprKind::Bool(_) | ExprKind::Nil
            | ExprKind::Void | ExprKind::DotIdent(_) => {}
        }
    }

    // ── Immutability check on assignment targets ───────────────────────────────

    fn check_assign_target(&mut self, lhs: &Expr, assign_line: usize, assign_col: usize) {
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
                    BindingKind::Mut | BindingKind::Var => {}
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
                self.error(
                    format!("nested `with {name}:` block on the same name is not allowed (double-acquire)"),
                    s.line, s.col,
                );
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
}
