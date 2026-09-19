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

//! Rust-backend-specific semantic checks.
//!
//! Split out of `checker/mod.rs` so the line between "universal Boring semantics"
//! (stays in `mod.rs`) and "artifacts of Rust's ownership/borrow model, or of the
//! current Rust-only GPU codegen pipeline" (this file) is visible at the file
//! level, not just by reading each check's own reasoning. See
//! `docs/design-notes/checker-portability-draft.md` for the full inventory and
//! rationale behind this split — done in anticipation of future non-Rust
//! backends (Swift/Kotlin), where none of the checks below apply as-is:
//! `Box`/`Rc`/`Arc` move-and-borrow semantics, `std::sync::atomic`'s exact type
//! support, GPU-kernel dispatch unwrapping `Rc`/`Arc`/`RefCell`/`Mutex`/`RwLock`,
//! and the GPU-residency/`with`-block memory-mapping model built around them.
//!
//! Mechanical split only, no behavior change: every function below is still a
//! method on the same `Checker` struct defined in `mod.rs`, sharing its scope/
//! binding-tracking state (`scopes`, `moved`, `open_with_names`, …) — Rust's
//! privacy rules let a submodule (`checker::rust_checks`) freely access its
//! parent module's private fields, so no visibility changes were needed in
//! `mod.rs` to support this split.

use crate::ast::*;
use super::Checker;

impl Checker {
    pub(super) fn check_qualifier_constraint(&mut self, binding: &BindingKind, var_mut: bool, ty: &Option<Type>, line: usize, col: usize) {
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
    // (a bare `Var` whose own declared type is itself `'static`) — never a fresh
    // construction.
    //
    // Fixed gap (previously): this used to recognize "fresh construction" only via
    // `is_constructor_call_expr`'s syntactic heuristic (an uppercase-first-letter
    // callee, e.g. `A(...)`) — so `let x'static = create()`, where `create()` is an
    // ordinary lowercase function/method that itself returns a freshly constructed
    // value, sailed through unrejected at a non-authorized site, silently violating
    // the same provenance guarantee this gate exists to enforce. Knowing for certain
    // whether an arbitrary call's return value is "fresh" would need a real
    // expression-type/provenance-inference pass this checker doesn't have. Rather
    // than special-case more callee shapes (always one indirection away from the
    // next false negative), this now follows `check_static_arg_provenance`'s
    // existing, already-conservative model: at a non-authorized site, only a bare
    // `Var` provably typed `'static` is accepted as the initializer; every `Call`,
    // `MethodCall`, or anything else this checker cannot positively prove is a
    // reference to an existing `'static` binding is rejected — erring towards
    // rejecting an as-yet-unrecognized-but-valid pattern rather than letting an
    // unsound one through.
    pub(super) fn check_static_provenance(&mut self, ty: &Option<Type>, value: Option<&Expr>, line: usize, col: usize) {
        let Some(Type::Qualified(_, OwnerQual::Static)) = ty else { return };
        let Some(value) = value else { return };
        if self.in_authorized_static_site { return; }
        let is_provably_static_ref = match &value.kind {
            ExprKind::Var(name) => matches!(
                self.lookup(name).and_then(|b| b.ty.as_ref()),
                Some(Type::Qualified(_, OwnerQual::Static))
            ),
            _ => false,
        };
        if !is_provably_static_ref {
            self.error(
                "cannot construct a 'static instance here — 'static values may only be constructed \
                 at top level or inside `main`; elsewhere the initializer must already be a \
                 reference to an existing 'static-typed binding (a name whose own type is \
                 T'static), not a fresh construction, a call, or a field read",
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
    pub(super) fn check_static_arg_provenance(&mut self, target_ty: Option<&Type>, arg: &Expr, line: usize, col: usize) {
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
    // ── `'atomic` type-compatibility gate ─────────────────────────────────────
    //
    // `'atomic` may only wrap the scalar int/bool family (`Type::is_atomic_eligible_scalar`)
    // — never a float (no stable `std::sync::atomic` equivalent) and never a
    // struct/enum/collection (no atomic representation at all). Checked at every
    // declared-type site this checker already visits for `check_set_mut_constraint`
    // (let/var/mut bindings, destructured bindings, fn/init/setter params, struct/enum
    // fields) — see that function's call sites, which this mirrors exactly, since both
    // are unconditional type-shape legality checks independent of `mut`-ness.
    pub(super) fn check_atomic_compatibility(&mut self, ty: &Option<Type>, line: usize, col: usize) {
        if self.kernel_dispatch_only { return; }
        let Some(ty) = ty else { return };
        if let Some(bad_inner) = ty.find_atomic_incompatibility() {
            self.error(
                format!(
                    "cannot combine `'atomic` with `{}`: `'atomic` only wraps a scalar integer or `bool` type \
                     (`int`, `uint`, `bool`, `int8`/`int16`/`int32`/`int64`, `uint8`/`uint16`/`uint32`/`uint64`) — \
                     floats have no stable `std::sync::atomic` equivalent, `int128`/`uint128` have no `AtomicI128`/`AtomicU128` \
                     in stable `std`, and structs/enums have no atomic representation at all; use `'actor` or `'guard` \
                     for interior mutability on this type",
                    Self::describe_type_for_atomic_error(bad_inner),
                ),
                line, col,
            );
        }
    }

    // ── `'observed` composability gate ────────────────────────────────────────
    //
    // `T'shared'observed` is rejected unconditionally (not just under `mut`, unlike
    // `check_qualifier_constraint`'s `mut 'shared` check above) — `'shared` has no
    // interior mutability at all (`Rc`/`Arc<T>`, no `Mutex`/`RwLock`/`RefCell`), so an
    // observed cell wrapping one could never have anything to notify subscribers
    // about, regardless of whether the binding itself is `mut`. Mirrors the
    // `mut 'shared` rejection's style/reasoning (`check_qualifier_constraint` above)
    // but fires independently of `mut` — see docs/book.md's "'observed" section,
    // composition table, `'shared'observed` row.
    pub(super) fn check_observed_compatibility(&mut self, ty: &Option<Type>, line: usize, col: usize) {
        if self.kernel_dispatch_only { return; }
        let Some(ty) = ty else { return };
        let ty = ty.without_mut();
        if let Type::Qualified(inner, OwnerQual::Observed) = ty {
            if matches!(inner.as_ref(), Type::Qualified(_, OwnerQual::Shared)) {
                self.error(
                    "cannot combine `'observed` with `'shared`: `'shared` has no interior \
                     mutability at all (no `Mutex`/`RwLock`/`RefCell`) — an observed cell \
                     wrapping it could never have anything to notify subscribers about; use \
                     `'actor'observed` or `'guard'observed` for a shared, mutable, observed value",
                    line, col,
                );
            }
        }
    }

    fn describe_type_for_atomic_error(ty: &Type) -> String {
        match ty {
            Type::Float32 => "float32".to_string(),
            Type::Float64 => "float64 (`float`)".to_string(),
            Type::Int128 => "int128".to_string(),
            Type::Uint128 => "uint128".to_string(),
            // `_` is the unresolved-placeholder base type left behind when a
            // name-position `x'atomic = <initializer>` couldn't infer a concrete
            // base type at parse time (an initializer more complex than a bare
            // literal) — not a real named type to surface verbatim.
            Type::Named(n) if n == "_" => "this type".to_string(),
            Type::Named(n) => n.clone(),
            _ => "this type".to_string(),
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

    pub(super) fn check_kernel_dispatch_qualifier(&mut self, kernel: &Expr, line: usize, col: usize) {
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

    pub(super) fn check_kernel_decl(&mut self, k: &KernelDecl) {
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

    pub(super) fn check_with_stmt(&mut self, s: &WithStmt) {
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
                // `'atomic` bindings have no lock/guard object to hold across a scoped
                // critical section — every access is already a single, independent
                // atomic operation (load/store/fetch_add/...), so `with x: ...` on an
                // `'atomic`-qualified `x` is a hard compile error, not a silent
                // fallback to per-access codegen. See docs/qualifiers.md's `'atomic`
                // section, "with-block incompatibility".
                if self.lookup(name).and_then(|b| b.ty.as_ref()).is_some_and(Self::type_is_atomic_qualified) {
                    self.error(
                        format!(
                            "cannot use `with {name}:` — `{name}` is `'atomic`-qualified; atomics have no \
                             lock/guard object to hold across a scoped block (each access is already a single, \
                             independent atomic operation) — remove the `with` wrapper and access `{name}` directly"
                        ),
                        s.line, s.col,
                    );
                }
                self.open_with_names.insert(name.clone());
                newly_opened.push(name.clone());
            }
        }
        self.check_block(&s.body);
        for name in &newly_opened { self.open_with_names.remove(name); }
    }

    fn type_is_atomic_qualified(ty: &Type) -> bool {
        matches!(ty.without_mut(), Type::Qualified(_, OwnerQual::Atomic))
    }

    /// If `name` is a `'gpu'unified`/`'gpu'global` binding sourced from a bare
    /// kernel-field read (`resident_from_field` — see `Binding`) and isn't currently
    /// open in an enclosing `with` block, records a compile error: any use at all
    /// (indexing, `.length`, iteration, string interpolation, passed as an argument,
    /// ...) requires a `with` wrapper first. A `'gpu'unified`/`'gpu'global` binding
    /// that is just a plain array (not sourced from a kernel field) is unrestricted —
    /// see `examples/saxpy.br`.
    pub(super) fn check_gpu_opacity(&mut self, name: &str, line: usize, col: usize) {
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

    // ── Use-after-move: committed-`'owned` struct-constructor arguments ────────
    //
    // Scope note: `'owned` (`Box<T>`) is the one qualifier where Boring's normal
    // "everything is passed by reference, the caller keeps ownership" model
    // (docs/book.md's parameter-passing rules) doesn't hold outright — but even
    // there, only ONE call shape is a genuine, exclusive Rust move: a struct
    // *constructor* call storing the argument straight into an `'owned` field
    // (`Holder(ac)`, `init(...)`'s body doing `oc = c`, or the implicit
    // all-fields constructor when there's no explicit `init` at all) — the field
    // is the value's new, sole, longer-lived owner, so the source variable is
    // gone for good (confirmed empirically against a clean `main` checkout:
    // `boring build --emit-rust` + `cargo build` on `let h1 = Holder(ac); let h2
    // = Holder(ac)` fails with a raw `E0382`, pointing at *generated* code the
    // user never wrote, not the actual Boring source line — this check moves
    // that diagnostic to the real source line, before the Rust step ever runs).
    //
    // A PLAIN FUNCTION CALL to an `'owned` parameter is deliberately NOT a move
    // source, even though the parameter itself is `Param.owned` too — see
    // `tests/cases/owned_call_arg_no_double_box.br`'s own header comment: unless
    // that parameter is also `mut`/`var`, the transpiler clones the box at the
    // call site instead of moving it (`bump(ac.clone())`), specifically so the
    // caller's variable stays usable afterward — reusing it is correct,
    // documented, tested behavior, not a bug. Treating every `'owned` function
    // parameter as a move (this check's first draft did) is a confirmed false
    // positive against that exact, already-passing regression test — a real
    // lesson from building this feature, not a hypothetical: the mere fact that
    // a Rust `Box<T>` gets passed somewhere doesn't by itself mean Boring's own
    // ownership model treats it as consumed; only construction genuinely does.
    //
    // What's covered: an argument that is a bare local variable (`ExprKind::Var`),
    // passed positionally or by label, to a struct-constructor call, at an
    // init-param/field position this checker can statically resolve to committed
    // `'owned` (`struct_ctor_owned` — reusing `Param.owned`'s exact predicate,
    // `Type::Qualified(_, OwnerQual::Owned)`). Once moved, *any* subsequent read
    // of that name — a call argument (to a function OR another constructor), a
    // method-call receiver, a field access, a bare mention in an expression — is
    // flagged, because every one of those bottoms out at the same
    // `ExprKind::Var` leaf `check_expr` already visits (see
    // `check_gpu_opacity`'s identical shape).
    //
    // What's NOT covered (documented gaps, not silent unsoundness — this never
    // produces a false positive on legal code, only misses some illegal code):
    //   - `'new` (`T'new`, or `new Ctor()`) — a candidate-set qualifier that only
    //     *becomes* `Owned` after the transpiler's own per-usage inference
    //     (`infer_qualifiers.rs`) runs, which happens well after this checker.
    //     Duplicating that inference here would be a much larger project;
    //     `'new` values are silently skipped rather than guessed at.
    //   - A method-call argument (`obj.method(ac)`) — this checker doesn't track
    //     per-struct method parameter ownership.
    //   - Anything that isn't straight-line code in the *same* block: a move in
    //     one `if`/`match` branch followed by a reuse after the branch, a reuse
    //     across a loop's iterations, a move inside a closure — see `moved`'s own
    //     doc comment for why (each nested block gets its own fresh move-frame).
    //   - A spread (`..expr`) or `_` (default-rest) constructor argument is never
    //     treated as a move source — matching those back to a specific field
    //     needs more than positional/label matching.
    //   - A struct with more than one `init` overload is skipped entirely
    //     (ambiguous which one a given call resolves to without real overload
    //     resolution) — see `struct_ctor_owned`'s own doc.

    /// Records `name` as moved-away in the *current* (innermost) move-frame only
    /// — see `moved`'s doc comment for why this is deliberately not visible to
    /// an enclosing or sibling block.
    pub(super) fn record_move(&mut self, name: &str, line: usize, col: usize) {
        if self.kernel_dispatch_only { return; }
        if let Some(frame) = self.moved.last_mut() {
            frame.insert(name.to_string(), (line, col));
        }
    }

    /// If `name` was already moved in the current move-frame, a compile error:
    /// a value can only be moved once. See this section's header comment for
    /// exactly what is and isn't caught.
    pub(super) fn check_move_read(&mut self, name: &str, line: usize, col: usize) {
        if self.kernel_dispatch_only { return; }
        if self.moved.last().and_then(|f| f.get(name)).is_some() {
            self.error(
                format!(
                    "`{name}` was already moved (passed to an owned parameter) here; \
                     a value can only be moved once — clone it explicitly first if you \
                     need to use it again"
                ),
                line, col,
            );
        }
    }

    /// Is the init-param/field at `callee`'s constructor-argument position
    /// `index` (or, for a labeled argument, named `label`) committed `'owned`?
    /// `callee` must be a bare `Var` naming a known, unambiguous struct
    /// constructor (`struct_ctor_owned`) — anything else (a plain function call
    /// — deliberately not a move source, see this section's header comment — a
    /// method call, an unknown/overloaded struct name, a computed callee)
    /// returns `false`, matching this check's best-effort, never-false-positive
    /// design.
    pub(super) fn owned_target_for_arg(&self, callee: &Expr, index: usize, label: Option<&str>) -> bool {
        let ExprKind::Var(name) = &callee.kind else { return false };
        let Some(params) = self.struct_ctor_owned.get(name.as_str()) else { return false };
        if let Some(label) = label {
            return params.iter().find(|(n, _)| n == label).map(|(_, o)| *o).unwrap_or(false);
        }
        params.get(index).map(|(_, o)| *o).unwrap_or(false)
    }
}
