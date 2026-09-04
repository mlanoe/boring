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

// Abstract Syntax Tree for the Boring language.
//
// Every node carries a `line` field for error reporting.
// Types are represented as `Option<Type>`: None means "to be inferred".

// ─── Attributes ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct Attr {
    pub name: String,
    pub args: Vec<String>,  // raw string args, may be "key=value" pairs
    pub line: usize,
    pub col: usize,
}

// ─── Top-level ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Use(UseDecl),
    Fn(FnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Trait(TraitDecl),
    Ext(ExtDecl),
    Mod(ModDecl),
    Let(LetStmt),
    Alias(AliasDecl),
    Kernel(KernelDecl),
    Stmt(Stmt),
}

/// `mod name:` — groups items into a named module for Rust transpilation.
/// The interpreter executes items in the current scope (flat).
#[derive(Debug, Clone)]
pub struct ModDecl {
    pub name: String,
    pub items: Vec<Item>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct AliasDecl {
    pub name: String,       // alias name, e.g. "Callable"
    pub type_params: Vec<String>, // generic type params, e.g. ["T"] for `use Callable<T> as …`
    pub ty: Type,           // the expanded type
    pub newtype: bool,      // true for `type Name as InnerType`, false for `use Name as Type`
    pub line: usize,
    pub col: usize,
}

/// `ext TypeName [as Trait1, Trait2]:` — adds methods/conversions to an existing struct.
#[derive(Debug, Clone)]
pub struct ExtDecl {
    pub type_name: String,           // base name: "Vec", "HashMap", etc.
    pub type_args: Vec<Type>,        // generic args: [TypeParam("T")] for Vec<T as Clone>
    pub type_params: Vec<String>,    // extracted param names: ["T"]
    pub where_clause: Vec<(String, String)>, // bounds: [("T","Clone")]
    pub traits: Vec<String>,              // traits implemented via `ext A as Trait:`
    pub methods: Vec<FnDecl>,
    pub setters: Vec<SetDecl>,
    pub conversions: Vec<AsDecl>,
    pub assoc_type_defs: Vec<AssocTypeDef>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct UseDecl {
    pub path: Vec<String>,
    pub glob: bool,
    /// Selective import list — `use mod.path(A, B)`.
    /// Empty ⇒ import the whole module (all pub items).
    /// Non-empty ⇒ import only the named items.
    pub items: Vec<String>,
    pub line: usize,
    pub col: usize,
}

// ─── Declarations ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: String,
    pub qualifier: Option<String>,  // "Animal" in `def Animal.speak()`
    pub params: Vec<Param>,
    pub return_ty: Option<Type>,
    pub body: Vec<Stmt>,
    pub is_pub: bool,
    pub throws: bool,
    pub task: bool,
    pub stream: bool,
    pub stream_capacity: Option<usize>,  // Some(N) if stream<N>, None for default capacity
    pub mutating: bool,         // true for `def`, false for `req`
    pub return_mutable: bool,   // true for `def mut` / `req mut` — mutable return value
    pub is_native: bool,        // body is `native` — implemented by the runtime
    /// Optional error type: `def foo() throws MyError:` → `Result<_, MyError>` in Rust.
    /// `None` = untyped throw (transpiler emits `Box<dyn Error>` or equivalent).
    pub throws_ty: Option<Type>,
    pub type_params: Vec<String>,
    pub where_clause: Vec<(String, String)>,
    pub attrs: Vec<Attr>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Option<Type>,
    pub mutable: bool,
    pub rebindable: bool, // true when declared with `var` — out-parameter semantics
    pub owned: bool,
    pub variadic: bool,        // `int... args` — collects remaining args as Array
    pub default: Option<Expr>, // `string name = "world"` — used when arg is absent
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct SetDecl {
    pub name: String,
    pub param_name: String,
    pub param_ty: Type,
    pub is_pub: bool,
    pub throws: bool,
    pub task: bool,
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: String,
    pub is_pub: bool,
    pub is_native: bool, // body is `native` — implemented by the runtime
    pub protocols: Vec<String>,
    pub fields: Vec<FieldDecl>,
    pub inits: Vec<InitDecl>,
    pub methods: Vec<FnDecl>,
    pub conversions: Vec<AsDecl>,
    pub type_params: Vec<String>,
    pub where_clause: Vec<(String, String)>,
    pub setters: Vec<SetDecl>,
    pub type_methods: Vec<TypeMethod>,
    pub type_vars: Vec<TypeVar>,
    pub assoc_type_defs: Vec<AssocTypeDef>,
    pub attrs: Vec<Attr>,
    pub line: usize,
    pub col: usize,
}

/// Kind of a type-level method (no `self` receiver).
#[derive(Debug, Clone, PartialEq)]
pub enum TypeMethodKind {
    Def,  // `type def` — may mutate type vars
    Req,  // `type req` — read-only, no side effects on type vars
    Set,  // `type set prop(T v):` — setter with logic for a type var
}

/// A type-level method: `[pub] type def/req/set name(params) -> ret: body`
/// Called as `TypeName.name(args)`.
#[derive(Debug, Clone)]
pub struct TypeMethod {
    pub kind: TypeMethodKind,
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: Option<Type>,
    pub body: Vec<Stmt>,
    pub is_pub: bool,
    pub throws: bool,
    /// Optional error type: `type def T make() throws MyError:` → mirrors
    /// `FnDecl::throws_ty` (see there). `None` = untyped `throws:`.
    pub throws_ty: Option<Type>,
    pub task: bool,
    pub line: usize,
    pub col: usize,
}

/// A type-level variable or constant: `[pub] type var/let T name = expr`
/// Accessed as `TypeName.name`. `type let` is immutable, `type var` is mutable.
/// A default value is required.
#[derive(Debug, Clone)]
pub struct TypeVar {
    pub name: String,
    pub ty: Option<Type>,
    pub default: Expr,
    pub is_pub: bool,
    pub mutable: bool, // true = `type var`, false = `type let`
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct AsDecl {
    pub is_pub: bool,
    pub ty: Type,
    pub throws: bool,
    pub task: bool,
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub name: String,
    pub is_pub: bool,
    /// `true` for `var`/`var mut` (the field's own name — `self.field = x` — may be
    /// reassigned to a different instance); `false` for `let`/`mut` (bare or
    /// explicit `let mut`). This is the *reassignment* axis only, independent of
    /// content mutation — see docs/book.md. Before that document,
    /// this single flag conflated both (`var Point p` granted `self.p = x` AND
    /// `self.p.inc()`); content-mutation permission ("can `self.field.method()` be
    /// called, can a nested field be written") now comes from `ty.grants_mut()`
    /// instead, matching local bindings/tuple slots/collection elements exactly.
    pub mutable: bool,
    pub transient: bool,
    pub ty: Type,
    pub default: Option<Expr>,
    /// `@name(args)` lines directly above the field, e.g. `@serde(rename = "current_costume")`.
    /// Parsed the same way as struct-/enum-level attrs (`parse_attrs`) and emitted verbatim as
    /// `#[name(args)]` immediately above the field in the generated Rust struct — see
    /// `emit_struct.rs`'s field-emission loop. Exists specifically for the case a struct field's
    /// JSON key (via `@derive(Deserialize)`/`fromJson<T>()`) doesn't match any single
    /// Boring-spelling of the field name, so the struct-level `@serde(rename_all = "...")`
    /// blanket rule can't cover it — see docs/json-deserialize-rename-gap.md.
    pub attrs: Vec<Attr>,
    pub line: usize,
    pub col: usize,
}

// ─── GPU / Kernel AST ────────────────────────────────────────────────────────

/// Qualifier for a field inside a `kernel` struct.
///
/// In kernel context (device code) the `'gpu` prefix is dropped:
///   `'unified` — cudaMallocManaged, accessible from host and device
///   `'global`  — device DRAM, write from host via init, read/write on device
///   `'shared`  — block SRAM (declared as `__shared__`)
///   `'local`   — per-thread registers / local mem
///   `'const`   — read-only constant memory
///
/// Host-side GPU fields (rare) use the `'gpu'*` prefixed forms.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuQual {
    Unified,
    Global,
    /// Bare `'actor` (kernel-context) — block SRAM (threadgroup memory), formerly spelled
    /// `'sync`. Barriers are inserted automatically unless the kernel `def` contains at
    /// least one explicit `sync` statement (manual mode) — that statement keyword is
    /// unrelated to this qualifier and is unaffected by the rename.
    Actor,
    Local,
    Const,
    /// `'actor'global` — device DRAM accessed via atomics on device.
    /// Behaves like `Global` for memory placement; compound assigns become atomic ops.
    ActorGlobal,
    /// `'actor'unified` — host+device DRAM accessed via atomics on device.
    /// Behaves like `Unified` for memory placement (host-visible, gets a `read_<field>()`
    /// accessor where the backend supports one); compound assigns become atomic ops
    /// exactly like `ActorGlobal`. CUDA/ROCm implement this; Metal/wgpu reject it for
    /// now with a target-specific error (see each backend's `device.rs`).
    ActorUnified,
    /// `'surface` — pixel buffer with backend-appropriate placement.
    /// Metal: `MTLStorageModePrivate` (GPU-only); CUDA: `cudaMallocManaged`; simulation: `Vec<u32>`.
    Surface,
}

/// Binding kind for a kernel field: `let`, `mut`, or `var`.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldBinding {
    Let,
    Mut,
    Var,
}

/// A field declaration inside a `kernel` struct.
#[derive(Debug, Clone)]
pub struct KernelFieldDecl {
    pub name: String,
    pub binding: FieldBinding,
    pub qual: GpuQual,
    pub ty: Type,
    pub default: Option<Expr>,
    pub line: usize,
    pub col: usize,
}

/// A `kernel` struct declaration.
///
/// Kernel structs differ from regular structs in that:
/// - Every field has an explicit binding (`let`/`mut`/`var`) and a GPU memory qualifier.
/// - Only one anonymous `def ()` is allowed (the kernel entry point).
/// - `init` constructors allocate GPU buffers.
#[derive(Debug, Clone)]
pub struct KernelDecl {
    pub name: String,
    pub is_pub: bool,
    pub fields: Vec<KernelFieldDecl>,
    pub inits: Vec<InitDecl>,
    pub methods: Vec<FnDecl>,
    /// Generic parameters — same encoding as `StructDecl.type_params`.
    /// `"$W:i64"` = const generic `int W`, `"$N:usize"` = const generic `uint N`, `"T"` = type param.
    pub type_params: Vec<String>,
    pub where_clause: Vec<(String, String)>,
    pub line: usize,
    pub col: usize,
}

/// Launch configuration passed to the `kernel(...)` expression.
#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Threads per block: int or tuple.
    pub block: Option<Expr>,
    /// Blocks per grid: int or tuple (inferred if None).
    pub grid: Option<Expr>,
    /// Ordering dependency: handle or tuple of handles.
    pub after: Option<Expr>,
    /// Scheduling priority: `high`, `normal`, or `low` (as string).
    pub priority: Option<String>,
    pub line: usize,
    pub col: usize,
}

/// A constructor declaration.
///
/// Two strictly separate forms — no mixing allowed:
/// - No body (`body` is empty): every param auto-declares a struct field.
///   `pub` controls visibility, `var` controls mutability.
/// - With body: every param is a plain local variable (same as method params).
///   Fields are declared in the struct body and assigned via `self.field = value`.
#[derive(Debug, Clone)]
pub struct InitDecl {
    pub params: Vec<InitParam>,
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

/// A parameter in an `init` declaration.
/// Semantics depend on whether the `InitDecl` has a body:
/// - No body: `is_pub`/`mutable` control the declared field's visibility/mutability.
/// - With body: `is_pub` is ignored; `mutable` means the local param is re-assignable.
#[derive(Debug, Clone)]
pub struct InitParam {
    pub is_pub: bool,
    pub mutable: bool,
    pub name: String,
    pub ty: Option<Type>,
    pub default: Option<Expr>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub is_pub: bool,
    pub is_native: bool, // body is `native` — implemented by the runtime
    pub type_params: Vec<String>,
    /// Trait conformances declared in the header: `enum Color as Drawable, Printable:`
    pub protocols: Vec<String>,
    pub variants: Vec<EnumVariant>,
    pub methods: Vec<FnDecl>,
    pub setters: Vec<SetDecl>,
    pub conversions: Vec<AsDecl>,
    /// Type-level (`type def`/`type req`/`type set`) factory/static methods —
    /// same production as `StructDecl::type_methods`. `boring run` only; the
    /// transpiler does not yet emit these for enums.
    pub type_methods: Vec<TypeMethod>,
    pub attrs: Vec<Attr>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Vec<VariantField>,
    pub attrs: Vec<Attr>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct VariantField {
    pub name: Option<String>,   // optional name: `Ok(T value)` → name = Some("value")
    pub ty: Type,
}

/// An associated type declaration inside a trait: `type Output` or `type Display as string`.
#[derive(Debug, Clone, PartialEq)]
pub struct AssocTypeDecl {
    pub name: String,
    pub constraint: Option<Type>,  // None = unconstrained; Some(T) = `type Name as T`
    /// Generic / lifetime parameters on the associated type itself (GAT syntax).
    /// E.g. `type MonType<&a>` → `type_params = ["'a"]`.
    /// Empty for ordinary associated types.
    pub type_params: Vec<String>,
    pub line: usize,
    pub col: usize,
}

/// An associated type definition inside a struct: `type Output = int`.
#[derive(Debug, Clone, PartialEq)]
pub struct AssocTypeDef {
    pub name: String,
    pub ty: Type,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct TraitDecl {
    pub name: String,
    pub parents: Vec<String>,
    pub signatures: Vec<FnSignature>,
    /// Methods with a default body — used when the implementing struct doesn't override them.
    pub defaults: Vec<FnDecl>,
    /// Associated function signatures (no `self`): `type def/req name(params) -> ret`
    pub type_signatures: Vec<FnSignature>,
    pub type_params: Vec<String>,
    pub assoc_types: Vec<AssocTypeDecl>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct FnSignature {
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: Option<Type>,
    pub throws: bool,
    pub task: bool,
    pub stream: bool,
    pub mutating: bool,
    pub return_mutable: bool,
    pub type_params: Vec<String>,
    pub line: usize,
    pub col: usize,
}

// ─── Binding kind ────────────────────────────────────────────────────────────

/// Describes the binding semantics of a `let` / `mut` / `var` / `lazy` declaration.
#[derive(Debug, Clone, PartialEq)]
pub enum BindingKind {
    /// `let` — fixed binding, immutable instance.
    Let,
    /// `mut` — fixed binding, mutable instance.
    Mut,
    /// `var` — rebindable, mutable instance.
    Var,
    /// `lazy` — deferred, write-once binding.  Transpiles to `OnceCell<T>`.
    /// Initialized on first `?=` call; subsequent calls are no-ops.
    Lazy,
}

/// Whether a `let_stmt`/destructure-slot's binding keyword + type together grant
/// content-mutation permission — `def` calls, field writes, structural
/// collection mutation (`Type::grants_mut` is the real source of truth —
/// see its own doc comment). Used by the interpreter/transpiler wherever a `LetStmt` or
/// `DestructureBinding` is bound, to decide whether to also mark the binding
/// content-mutable (separately from whether it's *rebindable*, `BindingKind::Var`).
///
/// When `ty` is explicit, the parser has already wrapped it in `Type::Mut`
/// wherever `mut` was requested (bare, `let mut`, or `var mut`), so
/// `Type::grants_mut` alone is authoritative. When `ty` is `None` (inferred
/// from the initializer), there's no `Type` node to carry that — fall back to
/// the keyword phrase itself, mirroring `check_tuple_mut_constraint`'s existing
/// "inferred, not just explicit, type" handling. Callers needing the inferred
/// case to also honor the scalar/`'shared`/`'weak`/tuple-whole rejection table
/// must run that check (as the checker's `check_scalar_mut_constraint` etc.
/// already do) — this function only reports what was *requested*, not whether
/// the request was legal.
pub fn binding_grants_mut(binding: &BindingKind, var_mut: bool, ty: Option<&Type>) -> bool {
    match ty {
        Some(t) => t.grants_mut(),
        None => matches!(binding, BindingKind::Mut) || (matches!(binding, BindingKind::Var) && var_mut),
    }
}

impl BindingKind {
    /// Returns `true` for `Mut` and `Var` — both need a `let mut` Rust binding
    /// (`Var` to allow reassignment, `Mut` because the type it always carries is
    /// `mut`-wrapped and needs `&mut self` to call through). Still correct for
    /// that one purpose — transpiler codegen deciding `let` vs `let mut` for the
    /// *outer* Rust local — after docs/book.md.
    ///
    /// **No longer the source of truth for "is `def`/field-write/collection-mutation
    /// allowed."** That permission now comes from whether the *resolved type*
    /// carries `mut` (`Type::grants_mut`), not from `BindingKind` alone — a plain
    /// `var Point p` (this returns `true`) does NOT permit `p.inc()` under the new
    /// rules; only `var mut Point p` does. See docs/book.md's
    /// Implementation checklist item 0.
    pub fn is_mutable(&self) -> bool {
        matches!(self, BindingKind::Mut | BindingKind::Var)
    }
}

// ─── Statements ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(LetStmt),
    /// Tuple / multi-binding destructure: `let (a, b) = expr`
    LetDestructure(LetDestructureStmt),
    Return(ReturnStmt),
    Break(usize, Option<Expr>),  // break  /  break value
    Continue(usize),
    Throw(ThrowStmt),
    If(IfStmt),
    IfLet(IfLetStmt),
    Match(MatchStmt),
    While(WhileStmt),
    WhileLet(WhileLetStmt),
    DoWhile(DoWhileStmt),
    Loop(LoopStmt),
    /// `wait expr` — sleep for a duration in a task context.
    /// Emits `tokio::time::sleep(expr).await;`.
    /// Position is explicit: put it before or after the body as needed.
    Wait(Expr, usize),
    For(ForStmt),
    Guard(GuardStmt),
    Try(TryStmt),
    /// `defer: block` — registers cleanup code to run when the enclosing function exits.
    /// Multiple defers in a function execute in reverse order (LIFO).
    Defer(Vec<Stmt>),
    Expr(Expr),
    Fn(FnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Mod(ModDecl),
    /// `use Name as Type` — local type alias, visible from this point in the enclosing scope.
    Alias(AliasDecl),
    /// `yield expr` — produces a value from a stream function.
    Yield(Expr, usize),
    /// Full-line comment `# text` — preserved for the transpiler, ignored by interpreter.
    Comment(String),
    /// `kernel:` execution block — unnamed GPU execution context.
    /// Body is a sequence of statements (may contain an inner `loop:` for GPU-driven rendering).
    /// Distinct from `kernel Foo:` (named struct declaration) by the absence of a name.
    KernelBlock(KernelBlockStmt),
    /// `with <name> [, <name> ...]:` — scoped access block.
    /// Grants extended, multi-statement host access to a value normally touched
    /// one operation at a time (`'gpu'unified`/`'gpu'global` residency, `'actor`/`'guard` locks).
    /// The qualifier and read/write access level are NOT resolved here — the checker
    /// looks each name's binding and qualifier up in scope, exactly like `def`/`req`
    /// method-call legality already is.
    With(WithStmt),
}

/// An unnamed `kernel:` execution block.
///
/// ```boring
/// kernel:
///     k(block = (16, 16))
///     screen.present(k.pixels)
/// ```
///
/// Or with a render loop (GPU-driven):
/// ```boring
/// kernel:
///     loop:
///         k(block = (16, 16))
///         screen.present(k.pixels)
/// ```
///
/// `let f = kernel: ...` stores the future in `f` (detached execution).
#[derive(Debug, Clone)]
pub struct KernelBlockStmt {
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

/// `let (a, b, c) = expr`  — destructures a tuple into named bindings.
/// Each binding may optionally carry a type annotation: `(int x, string y)`.
#[derive(Debug, Clone)]
pub struct LetDestructureStmt {
    pub binding: BindingKind,
    /// The group's own leading `var mut` — see `LetStmt.var_mut`'s doc. Also
    /// serves as (half of) the parenthesised form's per-element default
    /// (docs/book.md).
    pub var_mut: bool,
    pub bindings: Vec<DestructureBinding>,
    pub value: Expr,
    pub line: usize,
    pub col: usize,
}

/// One slot in a destructure: a name with an optional type.
/// Use `_` as name for a wildcard slot.
///
/// Each slot carries its own fully-resolved `binding`/`var_mut`, per
/// docs/book.md — the parser resolves an unmarked slot's default
/// (bare form → always `Let`; parenthesised form → inherits the group's own
/// leading keyword, `LetDestructureStmt.binding`/`.var_mut`) before ever
/// constructing this struct, so nothing downstream needs to re-derive it.
#[derive(Debug, Clone)]
pub struct DestructureBinding {
    pub name: String,          // variable name, or "_" for wildcard
    pub ty: Option<Type>,
    /// This slot's own resolved binding kind (never `Lazy` — not valid in a
    /// destructure slot). When `ty` is explicit and this grants mutation
    /// (`Mut`, or `Var` with `var_mut`), the parser has already wrapped `ty` in
    /// `Type::Mut` — see `Type::grants_mut`. When `ty` is `None` (inferred from
    /// the corresponding tuple position), this field is what the checker
    /// consults once the concrete type is known.
    pub binding: BindingKind,
    /// `true` only for this slot's own explicit `var mut` — see `LetStmt.var_mut`'s
    /// doc for why this can't be folded into `binding` alone.
    pub var_mut: bool,
    /// `true` when this slot was left unmarked in a *bare* (no parens) destructure
    /// immediately after a *different* slot in the same statement carried its own
    /// explicit keyword — docs/book.md's readability trap
    /// (`mut a, b = t` reads like `b` inherits `mut`; it quietly defaults to
    /// `let` instead). Correct either way — this only drives a lint warning, not
    /// an error.
    pub bare_unmarked_after_keyworded_sibling: bool,
}

/// One clause in a multi-condition `if let` or `guard let`.
///
/// ```boring
/// if let x = a, let y = b, x > 0:
/// guard let x = a, let y = b, x > 0 else:
/// ```
#[derive(Debug, Clone)]
pub enum CondClause {
    /// `let name = expr` — binds `name` if `expr` is non-nil; fails the whole condition otherwise.
    Let(String, Expr),
    /// `let Some(x) = expr` / `let Ok(v) = expr` — pattern destructuring; fails if pattern doesn't match.
    LetPat(Pattern, Expr),
    /// A boolean expression — must evaluate to `true`.
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct IfLetStmt {
    /// One or more clauses: `let` bindings and/or boolean expressions.
    /// All must succeed for the `then` body to execute.
    pub clauses: Vec<CondClause>,
    pub then_body: Vec<Stmt>,
    /// `elif let ...:` / `elif ...:` branches, tried in order if `clauses` fails.
    pub elif_branches: Vec<IfLetBranch>,
    pub else_body: Option<Vec<Stmt>>,
    pub line: usize,
    pub col: usize,
}

/// One `elif` branch of an `if let` chain: its own clause list plus body.
#[derive(Debug, Clone)]
pub struct IfLetBranch {
    pub clauses: Vec<CondClause>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub struct LetStmt {
    pub binding: BindingKind,
    pub is_pub: bool,
    pub is_static: bool,
    pub name: String,
    pub ty: Option<Type>,
    /// `true` only for an explicit `var mut Type a` (a second `mut` keyword
    /// following `var`). Meaningless unless `binding == BindingKind::Var`.
    ///
    /// Why this can't just be folded into `binding`: `BindingKind` has no
    /// `VarMut` variant, and `mut Type` composing into `ty` (via `Type::Mut`,
    /// wrapped by the parser whenever `ty` is explicit) already carries the
    /// permission for `BindingKind::Mut`/bare-mut — but when `ty` is `None`
    /// (inferred from the initializer), there is no `Type` node yet for the
    /// parser to wrap, so this flag is what the checker consults once the
    /// concrete type is known, exactly the same way `binding ==
    /// BindingKind::Mut` already needs to be consulted for the inferred-type
    /// bare-`mut` case (see `check_tuple_mut_constraint`'s existing precedent
    /// for "inferred, not just explicit, type" handling).
    pub var_mut: bool,
    /// `None` for deferred initialisation: `let v` / `var v` without `= expr`.
    pub value: Option<Expr>,
    /// `true` for `lazy` bindings — deferred, write-once via `?=`.
    pub is_lazy: bool,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub value: Option<Expr>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub branches: Vec<(Expr, Vec<Stmt>)>,
    pub else_body: Option<Vec<Stmt>>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct MatchStmt {
    pub subject: Expr,
    pub arms: Vec<MatchArm>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub patterns: Vec<Pattern>,
    /// Optional guard: `pattern if cond:` — arm only fires when cond is true.
    pub guard: Option<Expr>,
    pub body: MatchBody,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub enum MatchBody {
    Expr(Expr),
    Block(Vec<Stmt>),
}

#[derive(Debug, Clone)]
pub struct WhileStmt {
    pub condition: Expr,
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

/// `while let name = expr:` — loops while `expr` returns `Some(value)`.
/// Equivalent to Rust's `while let Some(name) = expr { ... }`.
#[derive(Debug, Clone)]
pub struct WhileLetStmt {
    pub name: String,              // bound variable name (used when pattern is None)
    pub pattern: Option<Pattern>,  // Some(pat) for `while let Some(x) = expr`
    pub value: Expr,               // the expression to unwrap
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct DoWhileStmt {
    pub body: Vec<Stmt>,
    pub condition: Expr,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct LoopStmt {
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct TryStmt {
    pub body: Vec<Stmt>,
    pub catch_clauses: Vec<CatchClause>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct CatchClause {
    pub types: Vec<String>,
    /// `catch Error.Expired:` — specific variant to match inside the enum.
    /// When set, only that variant fires; unhandled variants are re-thrown.
    pub variant: Option<String>,
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct ThrowStmt {
    pub value: Option<Expr>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct ForStmt {
    pub vars: Vec<String>,
    pub iterable: Expr,
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

/// `with <name> [, <name> ...]:` — scoped access block.
///
/// ```boring
/// with pr:
///     print "pr[0] = {pr[0]}"
///
/// with c:
///     c.value += 1
///     c.value += 1
/// ```
///
/// Deliberately does *not* carry each name's qualifier or read/write access level —
/// both are resolved later by the checker/transpiler from each name's already-known
/// binding and qualifier, the same way `def`/`req` method-call legality is resolved
/// without being baked into the AST.
#[derive(Debug, Clone)]
pub struct WithStmt {
    pub names: Vec<String>,
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct GuardStmt {
    pub cond: GuardCond,
    pub else_body: Vec<Stmt>,  // must contain return/throw/break/continue
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub enum GuardCond {
    /// Simple boolean expression: `guard x > 0 else:`
    Expr(Expr),
    /// One or more `let`/bool clauses: `guard let x = a, let y = b, x > 0 else:`
    /// Variables bound here are visible in the enclosing scope after the guard.
    Clauses(Vec<CondClause>),
}

// ─── Patterns ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Bind(String),
    Variant(String, Vec<Pattern>),
    Lit(LitPattern),
    None,
    Some(Box<Pattern>),
    /// Tuple pattern: `(a, 1, _)` — each element is itself a pattern
    Tuple(Vec<Pattern>),
}

#[derive(Debug, Clone)]
pub enum LitPattern {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Nil,
}

// ─── String interpolation ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum StringSegment {
    Lit(String),
    Expr(Box<Expr>),
    FormattedExpr(Box<Expr>, String), // (expr, fmt_spec)
}

// ─── Expressions ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub line: usize,
    pub col: usize,
    pub len: usize,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    // Literals
    Int(i64),
    /// A decimal integer literal whose magnitude overflows `i64` but fits `u64`
    /// (e.g. `18446744073709551615`, `u64::MAX`). Lexed/parsed separately from
    /// `Int` specifically so the literal's true, non-negative value survives
    /// into evaluation/codegen intact — only there can it be checked against
    /// (and only make sense for) an unsigned target, typically via an explicit
    /// `as uintNN` cast.
    UInt64(u64),
    Float(f64),
    Str(String),
    StringInterp(Vec<StringSegment>),
    Bool(bool),
    Nil,
    Void,

    // Variables
    Var(String),

    // Operators
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    UnaryOp(UnaryOp, Box<Expr>),

    // Assignment
    Assign(Box<Expr>, Box<Expr>),
    /// `lhs ?= rhs` — write-once / nil-coalescing assign.
    /// For `lazy` variables: emits `lhs.get_or_init(|| rhs)`.
    /// For optional variables: emits `if lhs.is_none() { lhs = Some(rhs); }`.
    QuestionAssign(Box<Expr>, Box<Expr>),

    // Access
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    /// `a[width = w, height = h, ...]` — mandatory-labeled indexing into a
    /// `Type::LabeledArray` (2+ dims only; 1D arrays keep using plain `Index`).
    /// Order-free at the use site — reuses `Arg` (label + value), the same
    /// shape already used for labeled call arguments. See
    /// docs/array-multidim-proposal.md, "Indexing".
    LabeledIndex(Box<Expr>, Vec<Arg>),

    // Calls
    Call(Box<Expr>, Vec<Arg>),
    MethodCall(Box<Expr>, String, Vec<Arg>),
    // Generic call: `f<T1, T2>(args)` — type arguments resolved at emit time
    GenericCall(Box<Expr>, Vec<Type>, Vec<Arg>),

    // Pipe operator: `lhs |> f(args)`
    // Desugars at emit time: if f is in fn_sigs → f(lhs, args), else → lhs.f(args)
    Pipe(Box<Expr>, String, Vec<Arg>),

    /// `new Constructor()` — placement expression, qualifier inferred excluding 'inline.
    /// `new(arena) Constructor()` — GPU arena placement; arena expression stored but not yet emitted.
    /// `ctor` is the constructor call expression (e.g. `Counter()`).
    /// `arena` is the expression inside `new(...)` if present.
    New { arena: Option<Box<Expr>>, ctor: Box<Expr> },

    /// `kernel(block = N, ...) expr` — GPU kernel launch expression.
    /// Returns a `KernelHandle<T>` value.
    KernelLaunch { config: Box<KernelConfig>, kernel: Box<Expr> },

    // try expr else default  — calls a throws fn, returns default on exception
    TryElse(Box<Expr>, Box<Expr>),

    // try: block else: block  — multi-line try/else; `error` is bound in the else block
    TryElseBlock(Vec<Stmt>, Vec<Stmt>),


    // Collections
    Array(Vec<Expr>),
    /// `[v for ..n]` — fill array of length `count` with `value`
    ArrayFill { value: Box<Expr>, count: Box<Expr> },
    /// `[..n]` — allocate array of length `count` without initialisation
    ArrayAlloc { count: Box<Expr> },
    /// `[f(i) for i in ..n]` — computed array of length `count` with `var` bound to index
    ArrayComp { expr: Box<Expr>, var: String, count: Box<Expr> },
    /// `[f(x) for x in collection]` — map over an existing collection
    ArrayCompIter { expr: Box<Expr>, var: String, iter: Box<Expr> },
    /// `[f(w, h) for w in ..W for h in ..H]` — chained comprehension for a
    /// labeled multi-dim array (2+ `for` clauses; a single clause keeps using
    /// `ArrayComp` unchanged). `clauses[0]` is axis 1 — the **declaration**
    /// order of axes, i.e. the fastest-varying index in row-major storage —
    /// NOT the syntactic nesting order. The desugared fill loop always
    /// iterates axis 1 innermost regardless of which `for` was written first,
    /// so `a[width=w, height=h]` addresses the same element the comprehension
    /// produced at `w + h*W`. Each clause is `(var, count)`, count always the
    /// `..N` range-count expression (only the range form is chainable — a
    /// collection-iteration clause stays a single-axis `ArrayCompIter`). See
    /// docs/array-multidim-proposal.md's "Rejected shorthand" section for why
    /// this is a chained `for...for...`, not a comma-separated clause list.
    LabeledArrayComp { expr: Box<Expr>, clauses: Vec<(String, Box<Expr>)> },
    Tuple(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Set(Vec<Expr>),

    // Dot-prefix enum shorthand: `.Red`
    DotIdent(String),

    // Range literals
    Range { start: Box<Expr>, end: Box<Expr>, inclusive: bool },

    // Slice index — only valid as the idx argument to Index: a[M..N], a[..N], a[M..], a[..]
    SliceRange { start: Option<Box<Expr>>, end: Option<Box<Expr>>, inclusive: bool },

    // Cast
    Cast(Box<Expr>, Type),
    /// `img as [line = width, column = height]` — explicit cross-label mapping
    /// between two `Type::LabeledArray`s at a function/assignment boundary.
    /// A distinct variant from `Cast`, not a `Type`-directed cast: the RHS is a
    /// `target_label = source_label` mapping table (`parse_type()` cannot
    /// parse this — `line` alone parses as `Type::Named("line")`, then hits
    /// `=` and fails). The mapping must be a bijection over the source's axis
    /// labels (every source axis named exactly once as some pair's RHS) —
    /// enforced by the checker, not here. See docs/array-multidim-proposal.md,
    /// "Cross-label compatibility between same-shape types".
    RelabelCast(Box<Expr>, Vec<(String, String)>),

    // nil-coalescing: `expr else default`
    Else(Box<Expr>, Box<Expr>),

    // Optional chaining: `expr?.field` or `expr?.method()`
    OptionalField(Box<Expr>, String),
    OptionalMethodCall(Box<Expr>, String, Vec<Arg>),

    // Closures — throws/task inferred from body content
    Closure(Vec<Param>, Option<Type>, ClosureBody, bool, bool),

    // Control flow as expressions
    If(Box<IfStmt>),
    Match(Box<MatchStmt>),

    // Block expression — evaluates stmts, last expression is the value
    Block(Vec<Stmt>),

    // `do:` scoped block — own scope + own defer frame; last expression is the value
    Do(Vec<Stmt>),

    // `loop:` as an expression — evaluates to the value passed to `break`
    Loop(LoopStmt),

    // task [:]? expr  OR  task: block — creates a Future / JoinHandle
    Task(Box<Expr>),

    // task(duration): body  OR  task(timeout = duration): body
    // Spawns a task with a built-in timeout; throws Error.Expired if it elapses.
    TaskWithTimeout(Box<Expr>, Box<Expr>),

    /// `join [f1, f2, ...]` — wait for multiple JoinHandles in parallel.
    JoinAll(Vec<Expr>),

    /// Rust macro invocation: `name!(args)`, `name![args]`, or `name!{args}`.
    /// The delimiter is irrelevant at the AST level.
    MacroCall { name: String, args: Vec<Expr> },
}

#[derive(Debug, Clone)]
pub struct Arg {
    pub label: Option<String>,
    pub value: Expr,
    /// `..expr` — spread all fields from a struct value into this call.
    /// The `label` is always `None` for spread args; `value` is the source object.
    pub spread: bool,
    /// Bare `_` — fill every remaining field of a struct-construction call with
    /// `Default::default()`, e.g. `Transform(translation = ..., scale = ..., _)`.
    /// Mirrors the `_` wildcard used in `match` arms and discard bindings.
    /// `label` is always `None` and `value` is an unused placeholder when this is set.
    pub default_rest: bool,
}

#[derive(Debug, Clone)]
pub enum ClosureBody {
    Expr(Box<Expr>),
    Block(Vec<Stmt>),
}

// ─── Operators ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    Eq, NotEq, RefEq, Lt, Gt, LtEq, GtEq,
    And, Or,
    BitAnd, BitOr, BitXor, Shl, Shr,
    Is,    // identity / type check / nil check
    IsNot, // negated is
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

// ─── Ownership qualifiers ────────────────────────────────────────────────────

/// Ownership / allocation qualifier for boring types.
///
/// Boring defaults to inline storage, no indirection (like Rust). Indirection is explicit.
///
/// `Dog`         → (no qual)       (Dog        — inline, default; Rust default)
/// `Dog&`        → Borrow          (&Dog       — borrow / reference, alias-compatible)
/// `Dog'owned`   → Owned           (Box<Dog>   — heap-owned, exclusive move)
/// `Dog'inline`  → Inline          (Dog        — explicit inline, equivalent to bare Dog)
/// `Dog'new`     → Union([Owned, Shared, Actor, Guard])
///                                 (Box/Rc/Arc/Mutex/RwLock<Dog> — any indirection, inferred;
///                                  bare tick `Dog'` is removed, this is its replacement)
/// `Dog'const`   → Const           (&'static D — compile-time constant / string literal)
/// `Dog'shared`  → Shared          (Arc<Dog> multi / Rc<Dog> single — threading-aware)
/// `Dog'weak`    → Weak            (Weak<Dog>  — non-owning ref, must upgrade to Dog'shared)
/// `Dog&a`         → Lifetime("a")   (&'a Dog        — borrow with explicit lifetime)
/// `Dog'&a`        → Lifetime("a")   (&'a Box<Dog>   — borrow a heap value with lifetime)
/// `Dog'shared&a`  → Lifetime("a")   (&'a Arc<Dog>   — borrow qualified type with lifetime)
/// `Dog?&`       → BorrowOption    (&Option<Dog> — borrow an optional value)
/// `Dog&weak`    → BorrowWeak      (&Weak<Dog> — borrow a weak pointer)
/// `Dog'weak&`   → BorrowWeak      (&Weak<Dog> — postfix form)
/// `var Dog&`    → BorrowMut       (&mut Dog   — mutable borrow; `var` prefix on any borrow type)
#[derive(Debug, Clone, PartialEq)]
pub enum OwnerQual {
    Owned,
    Actor,     // Arc<std::sync::Mutex<T>> (multi) or Rc<RefCell<T>> (single)
    ActorTask, // Arc<tokio::sync::Mutex<T>> (multi) or Rc<RefCell<T>> (single) — alias: 'task
    Guard,     // Arc<std::sync::RwLock<T>> (multi) or Rc<RefCell<T>> (single)
    GuardTask, // Arc<tokio::sync::RwLock<T>> (multi) or Rc<RefCell<T>> (single)
    /// Arc<T> (multi-thread) or Rc<T> (single-thread).
    /// The `--threading` flag determines which is emitted.
    /// Replaces the deprecated `T'auto` and `T'task` qualifiers.
    Shared,
    Weak,
    /// Force inline allocation for the Rust transpiler: `Point'inline` → plain `Point` in Rust.
    /// By default structs are heap-allocated (`Box<T>`); `'inline` opts out of that.
    /// The interpreter treats this identically to `Owned` at runtime (no difference).
    Inline,
    /// `T'static` → `&'static T` in Rust. No `Rc`/`Arc`, no refcount — a bare reference to a
    /// constant, program-lifetime instance that is never freed and never has more than one
    /// logical owner. Distinct from `OwnerQual::Lifetime` (an arbitrary named lifetime with
    /// zero Boring-level checking) even though both emit `&'<name> T`: `'static` additionally
    /// carries a provenance requirement (the value must trace back to one of three authorized
    /// construction sites — top-level `let`, inside `main`, or an implicit `type let` field),
    /// a `Sync` requirement independent of `--threading`, and a `mut`/`'weak` prohibition. See
    /// `docs/qualifiers.md`'s `'static` section for the full design. The interpreter treats this like a plain
    /// borrow at runtime (no runtime enforcement, matching every other qualifier).
    Static,
    /// Explicit lifetime annotation for Rust transpilation: `string'a` → `&'a str`.
    /// The interpreter treats this identically to a plain borrow (no runtime enforcement).
    Lifetime(String),
    /// Internal: threading-aware borrow of the smart pointer → `&Arc<T>` / `&Rc<T>`.
    /// No longer produced by the parser. Kept for backwards compatibility with serialized ASTs.
    BorrowShared,
    /// Universal borrow: `T&` → `&T`. The transpiler coerces any qualifier at the call site.
    Borrow,
    /// Internal: borrow of an owned (Box) value → `&Box<T>`.
    /// No longer produced by the parser (`T'owned&` / `T&owned` are removed).
    BorrowOwned,
    /// Borrow of an optional value: `Dog?&` → `&Option<Dog>` in Rust.
    BorrowOption,
    /// Mutable borrow of an optional: `mut T?&` → `&mut Option<T>` in Rust.
    BorrowOptionMut,
    /// Borrow of a weak pointer: `Dog&weak` or `Dog'weak&` → `&Weak<Dog>` in Rust.
    BorrowWeak,
    /// Mutable borrow: `var T&` → `&mut T` in Rust.
    /// Produced when the `var` keyword precedes a borrow type (`T&`) in a parameter
    /// or binding declaration.  The interpreter treats this like `Borrow` at runtime;
    /// the transpiler emits `&mut T`.
    BorrowMut,
    /// GPU memory qualifiers.
    /// Host-side: `T'gpu'unified`, `T'gpu'global`. `'const` has no host-side form — it has
    /// no host access (like `'local`), so it can only appear inside a `kernel` struct.
    /// Kernel-side (no 'gpu prefix): `T'unified`, `T'global`, `T'local`, `T'const`.
    /// Block-shared memory (formerly `T'sync`) is now bare `T'actor` in kernel-struct field
    /// position — see `OwnerQual::Actor`, reinterpreted by `parse_kernel_field`.
    GpuUnified,
    GpuGlobal,
    GpuLocal,
    GpuConst,
    /// `T'actor'global` — device DRAM with atomic access (kernel-side).
    GpuActorGlobal,
    /// `T'actor'unified` — host+device DRAM with atomic access (kernel-side).
    GpuActorUnified,
    /// `T'surface` — pixel buffer with backend-differentiated placement.
    GpuSurface,
    /// Qualifier union: `T'inline|owned|actor` — restricts which qualifiers callers may provide.
    /// At the Rust emission level this is a plain generic (no wrapping); the Boring compiler
    /// validates that every call site provides one of the listed qualifiers.
    /// Also used for the named groups: `'one` (`Inline|Owned`), `'many`
    /// (`Shared|Actor|Guard`), `'mut` (`Inline|Owned|Actor|Guard`), `'req`
    /// (`Shared|Static`).
    ///
    /// `'new` (the candidate-set pseudo-qualifier written as `T'new`, or implied by
    /// `new Ctor()` on a `let` RHS) is ALSO represented as a `Union` — specifically
    /// `Union([Owned, Shared, Actor, Guard])`, i.e. "any indirection, inferred,
    /// `'inline` excluded". Unlike the four groups above, `'new` is not a caller-facing
    /// acceptance contract (meaningful only on parameters) — it's a candidate-set seed,
    /// the same category as a bare `T`, so it must also narrow by usage on local
    /// variables (see `OwnerQual::is_owned_or_new` and its call sites in
    /// `infer_qualifiers.rs`/`collect_anonymous_vars`, which special-case exactly this
    /// shape to preserve that local-narrowing behavior).
    Union(Vec<OwnerQual>),
}

impl OwnerQual {
    /// The four members of the `'new` candidate-set qualifier, in canonical order —
    /// see the `Union` variant's doc comment above.
    pub const NEW_MEMBERS: [OwnerQual; 4] =
        [OwnerQual::Owned, OwnerQual::Shared, OwnerQual::Actor, OwnerQual::Guard];

    /// True for `'owned` (a single, committed indirect qualifier) and for `'new`
    /// (`Union([Owned, Shared, Actor, Guard])`) — i.e. any *declared* qualifier that
    /// guarantees indirection (never `'inline`) before per-usage inference has
    /// resolved it to one concrete member. Replaces the old `Owned | New` match arms
    /// from when `'new` had its own dedicated enum variant instead of being a `Union`.
    pub fn is_owned_or_new(&self) -> bool {
        match self {
            OwnerQual::Owned => true,
            OwnerQual::Union(members) => members.as_slice() == Self::NEW_MEMBERS,
            _ => false,
        }
    }

    /// True specifically for `'new` (`Union([Owned, Shared, Actor, Guard])`) — narrower
    /// than `is_owned_or_new`, which also matches a committed `'owned`. A committed
    /// `'owned` is a fixed contract (like `'shared`/`'actor`/`'guard`) and must NOT be
    /// seeded for per-usage inference the way `'new` is — see the inference-seeding
    /// call sites in `infer_qualifiers.rs` (parameter seeding and `collect_anonymous_vars`)
    /// that use this predicate specifically instead of `is_owned_or_new`.
    pub fn is_new(&self) -> bool {
        matches!(self, OwnerQual::Union(members) if members.as_slice() == Self::NEW_MEMBERS)
    }
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// Wrapper around an `Expr` used as a const-generic array-size expression.
/// PartialEq always returns `false` — two size expressions are never considered equal
/// structurally (they are resolved to concrete integers before comparison is needed).
#[derive(Debug, Clone)]
pub struct ConstExpr(pub Box<Expr>);
impl PartialEq for ConstExpr {
    fn eq(&self, _other: &Self) -> bool { false }
}

/// One axis of a labeled multi-dimensional array type (`[T, width, height]` /
/// `[T, width = W, height = H]`). `size: None` means a dynamic axis (shape known
/// only at construction); `size: Some(_)` means a compile-time-fixed axis, reusing
/// the same `ConstExpr` machinery as `ArrayNExpr` (arithmetic over const generic
/// params, e.g. `width = W * 2` is exactly as legal as `[T, W * 2]` today).
///
/// Per axis order is significant and permanent for the type: `axes[0]` is axis 1
/// (the fastest-varying index in row-major storage, and — for kernel fields —
/// the axis mapped to `gpu.thread.x`), `axes[1]` is axis 2 (`gpu.thread.y`), etc.
/// `[T, width, height]` and `[T, height, width]` are different (transposed) types.
/// See docs/array-multidim-proposal.md, "Axis order is fixed at declaration".
///
/// Within one declaration, axes are all-dynamic or all-fixed, never mixed —
/// enforced at parse time, checked again defensively in
/// `Type::labeled_array_shape_error`.
#[derive(Debug, Clone, PartialEq)]
pub struct LabeledAxis {
    pub label: String,
    pub size: Option<ConstExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Bare `int` — maps to Rust `isize`.
    Int,
    /// Bare `uint` — maps to Rust `usize`.
    Uint,
    /// Unsigned 8-bit integer (Rust `u8`) — distinct from `Uint` (usize).
    Uint8,
    /// Signed 8/16/32/64/128-bit integers (Rust `i8`/`i16`/`i32`/`i64`/`i128`).
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    /// Unsigned 16/32/64/128-bit integers (Rust `u16`/`u32`/`u64`/`u128`).
    Uint16,
    Uint32,
    Uint64,
    Uint128,
    /// 32-bit floating-point (Rust `f32`) — distinct from `Float64`.
    Float32,
    /// 64-bit floating-point (Rust `f64`). `float` is a pure alias of this
    /// variant, resolved at the alias table — see docs/float-width-types.md.
    Float64,
    Str,
    Bool,
    Nil,
    /// Unit return type — functions that produce no meaningful value.
    /// Distinct from Nil (absent optional) and Never (unreachable).
    Void,
    Never,
    Named(String),
    Optional(Box<Type>),
    Array(Box<Type>),
    /// Fixed-size array: `[T, N]` → Rust `[T; N]`
    ArrayN(Box<Type>, usize),
    /// Fixed-size array whose length is a compile-time expression over const generic params.
    /// E.g. `[bool, W * H]` where W and H are const params declared on the kernel.
    /// Monomorphised to `ArrayN` before code generation (concrete values substituted).
    ArrayNExpr(Box<Type>, ConstExpr),
    /// Labeled multi-dimensional array: `[T, width, height]` (all-dynamic axes)
    /// or `[T, width = W, height = H]` (all-fixed axes) — replaces `Image`/`Volume`.
    /// Never produced for a single bare identifier after the comma — that keeps
    /// meaning "reference to an existing const generic param" (`[T, N]`,
    /// `ArrayNExpr`), unchanged; this variant only ever has 2+ axes.
    /// See docs/array-multidim-proposal.md and `LabeledAxis`.
    LabeledArray(Box<Type>, Vec<LabeledAxis>),
    /// Integer literal in type-argument position: `GameOfLife<64, 64>`.
    ConstInt(i64),
    Tuple(Vec<Type>),
    Dict(Box<Type>, Box<Type>),
    Set(Box<Type>),
    // Function/closure type: `req Int (Int, Int) throws task` — optional return type + param types + flags
    // req=true → pure/readonly (Fn), req=false → mutating (FnMut, the default when no prefix)
    Fn(Option<Box<Type>>, Vec<Type>, bool, bool, bool),  // (return_ty?, param_types, throws, task, req)
    // Qualified type: `Dog'`, `Dog'copy`, `Dog'const`, `Dog'local`, `Dog'shared`
    Qualified(Box<Type>, OwnerQual),
    /// Bare trait name used as type → `Box<dyn Trait>` (dynamic dispatch, heap).
    Dyn(Box<Type>),
    /// `<Trait>` angle-bracket shorthand → `impl Trait` (static dispatch).
    Impl(Box<Type>),
    // Single type parameter reference: T, K, V (single uppercase letter)
    TypeParam(String),
    // Generic type application: Future<int>, Dict<string, int>
    Generic(String, Vec<Type>),
    /// Associated type reference: `Self.Output` or bare `Output` inside trait/struct context.
    SelfAssoc(String),
    /// Associated type access on a named type: `LinkedList.Index`, `Tree<T>.Node`.
    /// First element is the base type, second is the associated type name.
    AssocOf(Box<Type>, String),
    /// `mut Type` (owned form, no `&`) — see docs/book.md. A Boring-only
    /// permission with no distinct Rust type behind it: unlocks `def` calls and field
    /// writes on whatever this type nests into (a local binding, a tuple slot, a
    /// struct field, an array element, a dict value). The *borrowed* form (`mut
    /// Type&` → `&mut Type`) does NOT use this wrapper — it reuses the existing
    /// `Qualified(_, OwnerQual::BorrowMut)` representation, since that already has a
    /// real distinct Rust type. Parses everywhere a `type` does (grammar-unrestricted,
    /// per the doc); the checker rejects it outside the positions listed above.
    Mut(Box<Type>),
}

/// Copy-ness of a single owner qualifier, matching the rules in `Type::is_copy`.
/// Used to evaluate the members of an `OwnerQual::Union` (`'inline|actor`, `'mut`, ...).
fn owner_qual_is_copy(q: &OwnerQual) -> bool {
    match q {
        OwnerQual::Owned | OwnerQual::Inline => false,
        OwnerQual::Union(quals) => quals.iter().all(owner_qual_is_copy),
        _ => true,
    }
}

impl Type {
    /// True if this is a `mut Type` (owned-form) wrapper.
    pub fn is_mut(&self) -> bool {
        matches!(self, Type::Mut(_))
    }

    /// Strips one level of `mut Type` wrapper, if present. Every consumer that only
    /// cares about the underlying Rust representation (size/Copy-ness, codegen
    /// target type, etc.) should go through this first — `mut` has no Rust-level
    /// representation of its own (see `Type::Mut`'s doc), so those consumers are
    /// meant to be blind to it. Only the checker's permission logic and the type's
    /// own display/coercion logic should look at `Type::Mut` directly.
    pub fn without_mut(&self) -> &Type {
        match self {
            Type::Mut(inner) => inner.without_mut(),
            other => other,
        }
    }

    /// Owning version of `without_mut`.
    pub fn into_without_mut(self) -> Type {
        match self {
            Type::Mut(inner) => inner.into_without_mut(),
            other => other,
        }
    }

    /// True if this type grants content-mutation permission — `def` calls, field
    /// writes, structural collection mutation — wherever it's the declared type of a
    /// binding/slot. Covers both spellings: the owned `mut Type` wrapper (no
    /// distinct Rust type — see `Type::Mut`'s doc) and the borrowed `mut Type&` →
    /// `&mut Type` (`OwnerQual::BorrowMut`/`BorrowOptionMut`, which already existed
    /// before this proposal). This is the single source of truth checklist item 0
    /// asks for — permission comes from the *type*, never from `BindingKind` alone.
    pub fn grants_mut(&self) -> bool {
        matches!(
            self,
            Type::Mut(_)
                | Type::Qualified(_, OwnerQual::BorrowMut | OwnerQual::BorrowOptionMut)
        )
    }

    /// The element/value type an index read (`arr[i]`, `dict[k]`) yields,
    /// for a collection type — `[mut Point] arr` vs plain `[Point] arr`
    /// controls whether `arr[i].method()` may call a `def` method, exactly
    /// like a struct field's own type does (docs/book.md).
    /// Strips one level of `Type::Mut` first (the *collection's own* mut,
    /// `mut [Point]`, is a different axis — structural mutation of the
    /// collection itself — and doesn't affect this). Keys/sets have no
    /// element-mut position — `None` for `Set`, and only the value side of
    /// `Dict` is meaningful here.
    pub fn index_element_type(&self) -> Option<&Type> {
        match self.without_mut() {
            Type::Array(elem) | Type::ArrayN(elem, _) | Type::ArrayNExpr(elem, _)
                | Type::LabeledArray(elem, _) => Some(elem),
            Type::Dict(_, v) => Some(v),
            _ => None,
        }
    }

    /// If this type is (or wraps, through one level of `Generic<...>`,
    /// `Array`/`ArrayN`, `Optional`, `Mut`, or `Qualified`) a tuple type,
    /// returns one `grants_mut()` flag per tuple element. Used to decide,
    /// for `for a, b in iterable:`, which destructured loop variables need a
    /// Rust `mut` binding — the tuple slot's own `mut T&`/`mut T` doesn't
    /// give the *loop variable* a Rust-level mutable binding by itself (e.g.
    /// Bevy's `Query<(mut Position&, Velocity&)>` yields `Mut<Position>` per
    /// item, whose `DerefMut` needs `mut pos` in the pattern to call).
    /// Returns `None` if this type isn't recognizably tuple-shaped — callers
    /// then fall back to no `mut` on any slot (today's behavior).
    pub fn tuple_slot_mut_flags(&self) -> Option<Vec<bool>> {
        match self {
            Type::Tuple(elems) => Some(elems.iter().map(Type::grants_mut).collect()),
            Type::Generic(_, args) if args.len() == 1 => args[0].tuple_slot_mut_flags(),
            Type::Array(inner) | Type::ArrayN(inner, _) => inner.tuple_slot_mut_flags(),
            Type::Optional(inner) | Type::Mut(inner) | Type::Qualified(inner, _) => inner.tuple_slot_mut_flags(),
            _ => None,
        }
    }

    /// True if this type is a tuple with a `mut`-qualified slot, or an array/`ArrayN`
    /// with a `mut`-qualified element type — the two owned-`mut Type` positions
    /// (docs/book.md) that have no per-slot Rust representation at
    /// all. Rust has no per-tuple-element `mut` and no per-index `Vec` mutability, so
    /// the checker's per-slot permission tracking (`grants_mut`, already correct —
    /// see `tuple_slot_mut_flags`/`index_element_type`) has nothing to attach to on
    /// the Rust side except the *whole* binding: `t.0.move_to(...)` /
    /// `arr[0].move_to(...)` only compiles if the underlying Rust `let`/`var`
    /// binding itself is `mut`, regardless of what Boring keyword (`let`/`var`) was
    /// written. Used by `emit_let.rs` to force `let mut` on a tuple/array binding
    /// whose Boring-level binding kind alone wouldn't otherwise request it — this is
    /// the "Transpiler honesty" invariant from "Interactions and invariants" in that
    /// doc, not a change to what Boring source is allowed to do.
    ///
    /// Also covers dict values (`{K = mut V}`): `d.get_mut(key)` needs the
    /// underlying `HashMap` binding itself to be `mut` in Rust, same as
    /// `arr.get_mut`/`arr[i]` needs it for arrays — independent of whatever
    /// Boring binding keyword (`let`/`var`) was written on `d` itself. Paired
    /// with the `emit_expr.rs` fix routing a `def` call through a dict value
    /// (`d[k].method()`) to `get_mut` instead of `.get(k).cloned()` — see
    /// "Known implementation bugs" in the doc for the full writeup of that
    /// (worse, silent) bug this alone would not have fixed.
    pub fn nested_slot_grants_mut(&self) -> bool {
        match self {
            Type::Tuple(elems) => elems.iter().any(Type::grants_mut),
            Type::Array(elem) | Type::ArrayN(elem, _) | Type::ArrayNExpr(elem, _)
                | Type::LabeledArray(elem, _) => elem.grants_mut(),
            Type::Dict(_, val) => val.grants_mut(),
            Type::Mut(inner) | Type::Optional(inner) | Type::Qualified(inner, _) => {
                inner.nested_slot_grants_mut()
            }
            _ => false,
        }
    }

    /// True if a `Set` (`{T}`) appears anywhere in this type's structure with a
    /// `mut`-qualified element type (`{mut T}`) — illegal unconditionally, per
    /// docs/book.md: `std::collections::HashSet<T>` exposes no
    /// mutable element access in Rust at all (no `iter_mut()`, no `get_mut()`),
    /// because mutating an element in place could change its `Hash`/`Eq`
    /// behavior and silently corrupt the set's buckets. Unlike
    /// `nested_slot_grants_mut` (which tracks which slot needs a Rust-level
    /// `let mut` for an otherwise-legal `mut` placement), this is a pure
    /// well-formedness scan — `{mut T}` has no legal transpiler target at all,
    /// regardless of any *outer* binding's own mutability. Recurses into every
    /// position a `Set` could be nested (tuple slots, array/dict elements,
    /// generic arguments, qualifiers) so `[{mut T}]`, `({mut T}, int)`, etc.
    /// are all caught too, not just a bare `{mut T}` at the top level.
    pub fn contains_illegal_mut_set(&self) -> bool {
        match self {
            Type::Set(elem) => elem.grants_mut() || elem.contains_illegal_mut_set(),
            Type::Tuple(elems) => elems.iter().any(Type::contains_illegal_mut_set),
            Type::Dict(k, v) => k.contains_illegal_mut_set() || v.contains_illegal_mut_set(),
            Type::Array(inner) | Type::ArrayN(inner, _) | Type::ArrayNExpr(inner, _)
                | Type::LabeledArray(inner, _) | Type::Optional(inner) | Type::Mut(inner)
                | Type::Qualified(inner, _) | Type::Dyn(inner) | Type::Impl(inner) => {
                inner.contains_illegal_mut_set()
            }
            Type::Generic(_, args) => args.iter().any(Type::contains_illegal_mut_set),
            _ => false,
        }
    }

    pub fn is_copy(&self) -> bool {
        match self {
            Type::Mut(inner) => inner.is_copy(),
            Type::Int | Type::Uint | Type::Uint8
                | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128
                | Type::Float32 | Type::Float64 | Type::Str | Type::Bool | Type::Nil | Type::Void | Type::Never => true,
            Type::Optional(inner) => inner.is_copy(),
            Type::Tuple(elems) => elems.iter().all(|t| t.is_copy()),
            Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::Dict(_, _) | Type::Set(_) | Type::Named(_) => false,
            Type::ConstInt(_) => true,
            Type::Fn(..) => true,  // functions are copy (shared under the hood)
            // Owned = exclusive move → never copy
            Type::Qualified(_, OwnerQual::Owned | OwnerQual::Inline) => false,
            // Lifetime refs and borrows of smart pointers are copy at the borrow level
            Type::Qualified(_, OwnerQual::Lifetime(_) | OwnerQual::BorrowShared | OwnerQual::Borrow | OwnerQual::BorrowMut) => true,
            // A qualifier union is only Copy if every member qualifier it allows is Copy —
            // e.g. `'inline|actor` includes 'inline (move-only), so the union as a whole is not Copy.
            Type::Qualified(_, OwnerQual::Union(quals)) => quals.iter().all(owner_qual_is_copy),
            // All other qualifiers give copy/shared semantics
            Type::Qualified(_, _) => true,
            Type::TypeParam(_) => true,   // assumed copy at runtime, erased
            Type::Generic(_, _) => false, // heap type
            // Same shape (flat buffer) as Array/ArrayN — never Copy, regardless
            // of fixed vs dynamic axes.
            Type::LabeledArray(_, _) => false,
            Type::Dyn(inner) | Type::Impl(inner) => inner.is_copy(),
            Type::SelfAssoc(_)  => false, // conservative, like Named
            Type::AssocOf(_, _) => false, // conservative, like Named
        }
    }

    /// True if the type is task-safe (can be captured by a task).
    pub fn is_task_safe(&self) -> bool {
        match self {
            Type::Mut(inner) => inner.is_task_safe(),
            // Primitive copy types are always safe
            Type::Int | Type::Uint | Type::Uint8
                | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128
                | Type::Float32 | Type::Float64 | Type::Str | Type::Bool | Type::Nil | Type::Void | Type::Never => true,
            Type::Fn(..) => true,
            Type::Optional(inner) => inner.is_task_safe(),
            Type::Tuple(elems) => elems.iter().all(|t| t.is_task_safe()),
            // Unqualified collections / named types are not safe (sharing semantics undefined)
            Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::Dict(_, _) | Type::Set(_) | Type::Named(_) => false,
            Type::ConstInt(_) => true,
            // Qualifiers
            Type::Qualified(_, OwnerQual::Owned | OwnerQual::Inline) => true,  // exclusive move → source invalidated
            Type::Qualified(_, OwnerQual::Shared)     => true,  // Arc<T> (multi) / Rc<T> (single) — qualifier intent is task-safe
            Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask) => true,
            Type::Qualified(_, OwnerQual::Guard | OwnerQual::GuardTask) => true,
            Type::Qualified(_, OwnerQual::Weak)       => false, // Weak<T> — non-owning, conservative
            Type::Qualified(_, OwnerQual::Lifetime(_)) => true, // borrow — task-safe for transpilation
            Type::Qualified(_, OwnerQual::Static) => true, // &'static T outlives any task, unconditionally safe
            Type::Qualified(_, OwnerQual::BorrowShared) => true, // &Arc<T> / &Rc<T> — threading-aware borrow
            Type::Qualified(_, OwnerQual::BorrowOwned)  => false,
            Type::Qualified(_, OwnerQual::BorrowOption | OwnerQual::BorrowOptionMut) => false,
            Type::Qualified(_, OwnerQual::BorrowWeak)   => false,
            Type::Qualified(_, OwnerQual::Borrow)       => false, // unknown until alias resolved — conservative
            Type::Qualified(_, OwnerQual::BorrowMut)    => false, // &mut T — conservative (target unknown)
            Type::Qualified(inner, OwnerQual::Union(_)) => inner.is_task_safe(), // union: delegate to inner
            Type::TypeParam(_) => true,
            Type::Generic(_, _) => false, // unless qualified, keep simple for now
            // Unqualified, like Array/ArrayN — sharing semantics undefined without a qualifier.
            Type::LabeledArray(_, _) => false,
            Type::Dyn(inner) | Type::Impl(inner) => inner.is_task_safe(),
            Type::SelfAssoc(_)  => false, // conservative, like Named
            Type::AssocOf(_, _) => false, // conservative, like Named
            Type::Qualified(_, OwnerQual::GpuUnified | OwnerQual::GpuGlobal | OwnerQual::GpuLocal | OwnerQual::GpuConst | OwnerQual::GpuActorGlobal | OwnerQual::GpuActorUnified | OwnerQual::GpuSurface) => false,
        }
    }

    /// The host-context GPU-residency qualifier (`'gpu'unified` / `'gpu'global`) at the
    /// outermost qualifier layer, if any. Qualifiers are always the outermost wrapper
    /// around a declared type (`[T]'gpu'unified` parses as `Qualified(Array(T), GpuUnified)`),
    /// so no recursion is needed. See docs/scoped-access-blocks.md, "The qualifier".
    pub fn gpu_resident_qual(&self) -> Option<&OwnerQual> {
        match self {
            Type::Qualified(_, q @ (OwnerQual::GpuUnified | OwnerQual::GpuGlobal)) => Some(q),
            _ => None,
        }
    }

    /// If this is a `Type::LabeledArray`, returns the element type and its axis
    /// list. The single recognition point for the labeled multi-dim array
    /// type. See docs/array-multidim-types.md.
    pub fn as_labeled_array(&self) -> Option<(&Type, &[LabeledAxis])> {
        match self {
            Type::LabeledArray(elem, axes) => Some((elem, axes)),
            _ => None,
        }
    }

    /// Total element count for a fully-fixed-shape `LabeledArray`, when every
    /// axis's size is directly an integer literal (`width = 16`, not
    /// `width = W` or `width = W * 2`). `None` for a dynamic-shape array (any
    /// axis has `size: None`) *and* for a fixed-shape array whose sizes involve
    /// a const generic reference or arithmetic expression — those need a
    /// subst-map evaluation context to resolve (see `eval_const_expr` in
    /// `transpiler/wgpu/mod.rs`, the same machinery `ArrayNExpr` already relies
    /// on), which this `&self`-only method has no access to. Returns `None`
    /// rather than a silently-wrong `Some` whenever the length can't be
    /// resolved from the type alone.
    pub fn labeled_array_len(&self) -> Option<i64> {
        let (_, axes) = self.as_labeled_array()?;
        if axes.is_empty() || axes.iter().any(|a| a.size.is_none()) {
            return None;
        }
        axes.iter().try_fold(1i64, |acc, a| match &a.size.as_ref().unwrap().0.kind {
            ExprKind::Int(n) => Some(acc * n),
            _ => None,
        })
    }

    /// Validates a `Type::LabeledArray`'s axis list and returns a human-readable
    /// error message if malformed; `None` if `self` isn't a `LabeledArray` at
    /// all, or is well-formed. Checks (defense in depth — the parser already
    /// enforces most of this at parse time, see `docs/array-multidim-proposal.md`):
    ///   1. arity — at least 2 axes (a single label never reaches this variant,
    ///      see `LabeledArray`'s doc comment, but guard against a malformed AST
    ///      built by anything other than the parser, e.g. a future desugar pass);
    ///   2. duplicate axis labels;
    ///   3. mixed fixed/dynamic axes within one declaration (D1: all-or-nothing).
    ///
    /// Does **not** enforce a 3-axis cap — that's a GPU-kernel-field-specific
    /// restriction (thread.x/y/z), not a property of the type itself; CPU-side
    /// labeled arrays are unbounded. Callers checking a kernel field must apply
    /// that cap themselves alongside this function (see `check_kernel_decl`).
    pub fn labeled_array_shape_error(&self) -> Option<String> {
        let (_, axes) = self.as_labeled_array()?;
        if axes.len() < 2 {
            return Some(
                "labeled array types need at least 2 axes — a single axis has no \
                 ambiguity to resolve and should be written as a plain [T] / [T, N]"
                    .to_string(),
            );
        }
        let mut seen = std::collections::HashSet::new();
        for axis in axes {
            if !seen.insert(axis.label.as_str()) {
                return Some(format!("duplicate axis label '{}'", axis.label));
            }
        }
        let fixed = axes.iter().filter(|a| a.size.is_some()).count();
        if fixed != 0 && fixed != axes.len() {
            return Some(
                "labeled array axes must be all dynamic or all fixed — mixing \
                 e.g. [T, width, height = H] is not supported"
                    .to_string(),
            );
        }
        None
    }
}

// ─── `with` scoped-access-block analysis ────────────────────────────────────
//
// Shared, pure AST walk used by both the checker (opacity enforcement) and the
// transpiler (deciding map-for-read vs map-for-read-write / RwLock::read vs
// write / Mutex::lock at `with` codegen time) — see docs/scoped-access-blocks.md,
// "Read vs. write access level".
//
// A `let`-bound `with` name is always read-only (checked by the caller from the
// binding's `BindingKind` before ever calling this). For a `mut`/`var`-bound name,
// this scans the block's own body — recursing into `if`/`while`/`do-while`/`loop`/
// `for`/`match`/`try`/`guard`/closures/nested-`with` lexically inside it, but never
// into the body of a called function or method (only its signature, via the two
// callbacks) — for:
//   - a direct or index/field assignment targeting `name`;
//   - `name` passed as an argument at a position the callee declares `var`;
//   - a `def` (mutating) method called on `name`.
// Any of these grant the block write access; none found means read-only, even
// though the binding itself could support a mutation elsewhere.

/// Returns `true` if `name` is mutated anywhere in `body`, per the rules above.
///
/// `is_var_param(callee_name, arg_index)` — does the free function named
/// `callee_name` declare a `var` parameter at position `arg_index`? Signature-only
/// lookup, matching how `def`/`req` legality is already resolved elsewhere.
///
/// `is_mutating_method(receiver_name, method_name)` — when `receiver_name == name`
/// (the with-block subject), is `method_name` a `def` (mutating) method rather than
/// a `req`? Callers only need to answer this for the with-block's own name(s); it is
/// still invoked for other receivers so a caller may choose to ignore those safely.
pub fn with_block_mutates(
    body: &[Stmt],
    name: &str,
    is_var_param: &mut dyn FnMut(&str, usize) -> bool,
    is_mutating_method: &mut dyn FnMut(&str, &str) -> bool,
) -> bool {
    body.iter().any(|s| with_stmt_mutates(s, name, is_var_param, is_mutating_method))
}

fn with_stmt_mutates(
    stmt: &Stmt,
    name: &str,
    ivp: &mut dyn FnMut(&str, usize) -> bool,
    imm: &mut dyn FnMut(&str, &str) -> bool,
) -> bool {
    let e = |ex: &Expr, ivp: &mut dyn FnMut(&str, usize) -> bool, imm: &mut dyn FnMut(&str, &str) -> bool| with_expr_mutates(ex, name, ivp, imm);
    let b = |body: &[Stmt], ivp: &mut dyn FnMut(&str, usize) -> bool, imm: &mut dyn FnMut(&str, &str) -> bool| with_block_mutates(body, name, ivp, imm);
    match stmt {
        Stmt::Let(s) => s.value.as_ref().map(|v| e(v, ivp, imm)).unwrap_or(false),
        Stmt::LetDestructure(s) => e(&s.value, ivp, imm),
        Stmt::Return(r) => r.value.as_ref().map(|v| e(v, ivp, imm)).unwrap_or(false),
        Stmt::Throw(t) => t.value.as_ref().map(|v| e(v, ivp, imm)).unwrap_or(false),
        Stmt::If(s) => {
            s.branches.iter().any(|(c, body)| e(c, ivp, imm) || b(body, ivp, imm))
                || s.else_body.as_ref().map(|body| b(body, ivp, imm)).unwrap_or(false)
        }
        Stmt::IfLet(s) => {
            s.clauses.iter().any(|c| with_cond_clause_mutates(c, name, ivp, imm))
                || b(&s.then_body, ivp, imm)
                || s.elif_branches.iter().any(|br| {
                    br.clauses.iter().any(|c| with_cond_clause_mutates(c, name, ivp, imm)) || b(&br.body, ivp, imm)
                })
                || s.else_body.as_ref().map(|body| b(body, ivp, imm)).unwrap_or(false)
        }
        Stmt::Match(s) => with_match_mutates(s, name, ivp, imm),
        Stmt::While(s) => e(&s.condition, ivp, imm) || b(&s.body, ivp, imm),
        Stmt::WhileLet(s) => e(&s.value, ivp, imm) || b(&s.body, ivp, imm),
        Stmt::DoWhile(s) => b(&s.body, ivp, imm) || e(&s.condition, ivp, imm),
        Stmt::Loop(s) => b(&s.body, ivp, imm),
        Stmt::For(s) => e(&s.iterable, ivp, imm) || b(&s.body, ivp, imm),
        Stmt::Guard(s) => {
            let cond_hit = match &s.cond {
                GuardCond::Expr(ex) => e(ex, ivp, imm),
                GuardCond::Clauses(cs) => cs.iter().any(|c| with_cond_clause_mutates(c, name, ivp, imm)),
            };
            cond_hit || b(&s.else_body, ivp, imm)
        }
        Stmt::Try(s) => {
            b(&s.body, ivp, imm) || s.catch_clauses.iter().any(|c| b(&c.body, ivp, imm))
        }
        Stmt::Defer(body) => b(body, ivp, imm),
        // Lexically nested `with` (same or different name) is still part of this
        // block's body — recurse into it, same as any other nested construct.
        Stmt::With(s) => b(&s.body, ivp, imm),
        Stmt::Yield(ex, _) | Stmt::Wait(ex, _) => e(ex, ivp, imm),
        Stmt::Break(_, v) => v.as_ref().map(|ex| e(ex, ivp, imm)).unwrap_or(false),
        Stmt::KernelBlock(s) => b(&s.body, ivp, imm),
        Stmt::Expr(ex) => e(ex, ivp, imm),
        // New scope / new callable — signature only, never the body.
        Stmt::Fn(_) | Stmt::Struct(_) | Stmt::Enum(_) | Stmt::Mod(_) | Stmt::Alias(_)
        | Stmt::Continue(_) | Stmt::Comment(_) => false,
    }
}

/// Scans `body` for every occurrence of `Var(name)`, classifying each one via
/// `classify(fn_name, arg_index)` — called when the occurrence is a bare argument at
/// position `arg_index` of a call to `fn_name(...)` (this also covers a kernel
/// constructor call, since `let mut k = Kernel(x, ...)` is just an `ExprKind::Call`
/// initializer like any other); return `true` from it when that particular position
/// counts as a "qualifying" use. Recurses into `if`/`while`/`for`/`match`/closures/
/// etc. nested in `body` (same bounded-scan convention `with_block_mutates` uses),
/// never into a called function's own body. Returns `(has_any_use,
/// has_only_qualifying_uses)` — the second is only meaningful when the first is
/// `true`. See `Checker::scan_fn_gpu_arg_params` (checker/mod.rs) for the caller.
pub fn scan_var_call_arg_uses(
    body: &[Stmt],
    name: &str,
    classify: &mut dyn FnMut(&str, usize) -> bool,
) -> (bool, bool) {
    let mut any = false;
    let mut other = false;
    for s in body { scan_stmt_var_arg(s, name, classify, &mut any, &mut other); }
    (any, any && !other)
}

fn scan_stmt_var_arg(
    stmt: &Stmt,
    name: &str,
    classify: &mut dyn FnMut(&str, usize) -> bool,
    any: &mut bool,
    other: &mut bool,
) {
    // Local macros rather than closures: two sibling closures both capturing
    // `classify` (a `&mut dyn FnMut`) by move would conflict with each other, since
    // that reference can't be split. A macro just reborrows it fresh at each
    // expansion site, which is all a `&mut` parameter needs across sequential calls.
    macro_rules! e { ($ex:expr) => { scan_expr_var_arg($ex, name, classify, any, other) }; }
    macro_rules! b { ($body:expr) => { for s in $body { scan_stmt_var_arg(s, name, classify, any, other); } }; }
    match stmt {
        Stmt::Let(s) => { if let Some(v) = &s.value { e!(v); } }
        Stmt::LetDestructure(s) => e!(&s.value),
        Stmt::Return(r) => { if let Some(v) = &r.value { e!(v); } }
        Stmt::Throw(t) => { if let Some(v) = &t.value { e!(v); } }
        Stmt::If(s) => {
            for (c, body) in &s.branches { e!(c); b!(body); }
            if let Some(body) = &s.else_body { b!(body); }
        }
        Stmt::IfLet(s) => {
            for c in &s.clauses { scan_cond_clause_var_arg(c, name, classify, any, other); }
            b!(&s.then_body);
            for br in &s.elif_branches {
                for c in &br.clauses { scan_cond_clause_var_arg(c, name, classify, any, other); }
                b!(&br.body);
            }
            if let Some(body) = &s.else_body { b!(body); }
        }
        Stmt::Match(s) => scan_match_var_arg(s, name, classify, any, other),
        Stmt::While(s) => { e!(&s.condition); b!(&s.body); }
        Stmt::WhileLet(s) => { e!(&s.value); b!(&s.body); }
        Stmt::DoWhile(s) => { b!(&s.body); e!(&s.condition); }
        Stmt::Loop(s) => b!(&s.body),
        Stmt::For(s) => { e!(&s.iterable); b!(&s.body); }
        Stmt::Guard(s) => {
            match &s.cond {
                GuardCond::Expr(ex) => e!(ex),
                GuardCond::Clauses(cs) => { for c in cs { scan_cond_clause_var_arg(c, name, classify, any, other); } }
            }
            b!(&s.else_body);
        }
        Stmt::Try(s) => {
            b!(&s.body);
            for c in &s.catch_clauses { b!(&c.body); }
        }
        Stmt::Defer(body) => b!(body),
        Stmt::With(s) => b!(&s.body),
        Stmt::Yield(ex, _) | Stmt::Wait(ex, _) => e!(ex),
        Stmt::Break(_, v) => { if let Some(ex) = v { e!(ex); } }
        Stmt::KernelBlock(s) => b!(&s.body),
        Stmt::Expr(ex) => e!(ex),
        // New scope / new callable — signature only, never the body.
        Stmt::Fn(_) | Stmt::Struct(_) | Stmt::Enum(_) | Stmt::Mod(_) | Stmt::Alias(_)
        | Stmt::Continue(_) | Stmt::Comment(_) => {}
    }
}

fn scan_match_var_arg(
    s: &MatchStmt,
    name: &str,
    classify: &mut dyn FnMut(&str, usize) -> bool,
    any: &mut bool,
    other: &mut bool,
) {
    scan_expr_var_arg(&s.subject, name, classify, any, other);
    for a in &s.arms {
        if let Some(g) = &a.guard { scan_expr_var_arg(g, name, classify, any, other); }
        match &a.body {
            MatchBody::Expr(ex) => scan_expr_var_arg(ex, name, classify, any, other),
            MatchBody::Block(body) => { for s in body { scan_stmt_var_arg(s, name, classify, any, other); } }
        }
    }
}

fn scan_cond_clause_var_arg(
    c: &CondClause,
    name: &str,
    classify: &mut dyn FnMut(&str, usize) -> bool,
    any: &mut bool,
    other: &mut bool,
) {
    match c {
        CondClause::Expr(ex) | CondClause::Let(_, ex) | CondClause::LetPat(_, ex) => scan_expr_var_arg(ex, name, classify, any, other),
    }
}

fn scan_expr_var_arg(
    expr: &Expr,
    name: &str,
    classify: &mut dyn FnMut(&str, usize) -> bool,
    any: &mut bool,
    other: &mut bool,
) {
    macro_rules! e { ($ex:expr) => { scan_expr_var_arg($ex, name, classify, any, other) }; }
    macro_rules! b { ($body:expr) => { for s in $body { scan_stmt_var_arg(s, name, classify, any, other); } }; }
    match &expr.kind {
        ExprKind::Var(v) => { if v == name { *any = true; *other = true; } }
        ExprKind::Assign(lhs, rhs) | ExprKind::QuestionAssign(lhs, rhs) => { e!(lhs); e!(rhs); }
        ExprKind::Call(callee, args) => {
            e!(callee);
            if let ExprKind::Var(fn_name) = &callee.kind {
                for (i, a) in args.iter().enumerate() {
                    if matches!(&a.value.kind, ExprKind::Var(v) if v == name) {
                        *any = true;
                        if !classify(fn_name, i) { *other = true; }
                    } else {
                        e!(&a.value);
                    }
                }
            } else {
                for a in args { e!(&a.value); }
            }
        }
        ExprKind::MethodCall(recv, _, args) | ExprKind::OptionalMethodCall(recv, _, args) => {
            e!(recv);
            for a in args { e!(&a.value); }
        }
        ExprKind::GenericCall(callee, _, args) => { e!(callee); for a in args { e!(&a.value); } }
        ExprKind::Pipe(lhs, _, args) => { e!(lhs); for a in args { e!(&a.value); } }
        ExprKind::New { ctor, arena } => { e!(ctor); if let Some(a) = arena { e!(a); } }
        ExprKind::BinOp(_, l, r) => { e!(l); e!(r); }
        ExprKind::UnaryOp(_, ex) | ExprKind::Cast(ex, _) => e!(ex),
        // `<name>.length`/`.count` is a size query, not a host materialization -- a
        // `BoringGpuArg<T>` value can answer it without ever touching the buffer
        // (`Resident` already carries its length, `Host` can `.len()` its Vec — see
        // the `BoringGpuArg::len()` helper `wgpu::host::emit_gpu_copy_helpers` emits).
        // Every kernel-launcher wrapper in practice sizes its dispatch block off the
        // very array it also passes to the kernel constructor (`k(block = x.length)`),
        // so treating this as disqualifying would make the exclusive-ctor-arg scan
        // never actually fire for a realistic function -- count it as a qualifying use
        // instead of falling through to the generic `Field` recursion below.
        ExprKind::Field(ex, field) | ExprKind::OptionalField(ex, field)
            if (field == "length" || field == "count")
                && matches!(&ex.kind, ExprKind::Var(v) if v == name) =>
        {
            *any = true;
        }
        ExprKind::Field(ex, _) | ExprKind::OptionalField(ex, _) => e!(ex),
        ExprKind::Index(obj, idx) => { e!(obj); e!(idx); }
        ExprKind::LabeledIndex(obj, args) => { e!(obj); for a in args { e!(&a.value); } }
        ExprKind::Else(ex, d) | ExprKind::TryElse(ex, d) => { e!(ex); e!(d); }
        ExprKind::TryElseBlock(body, els) => { b!(body); b!(els); }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) => { for ex in elems { e!(ex); } }
        ExprKind::ArrayFill { value, count } => { e!(value); e!(count); }
        ExprKind::ArrayAlloc { count } => e!(count),
        ExprKind::ArrayComp { expr: ex, count, .. } => { e!(count); e!(ex); }
        ExprKind::ArrayCompIter { expr: ex, iter, .. } => { e!(iter); e!(ex); }
        ExprKind::LabeledArrayComp { expr: ex, clauses } => { for (_, count) in clauses { e!(count); } e!(ex); }
        ExprKind::RelabelCast(ex, _) => e!(ex),
        ExprKind::Dict(pairs) => { for (k, v) in pairs { e!(k); e!(v); } }
        ExprKind::Range { start, end, .. } => { e!(start); e!(end); }
        ExprKind::SliceRange { start, end, .. } => {
            if let Some(s) = start { e!(s); }
            if let Some(ex) = end { e!(ex); }
        }
        ExprKind::StringInterp(segs) => {
            for seg in segs {
                match seg {
                    StringSegment::Expr(ex) | StringSegment::FormattedExpr(ex, _) => e!(ex),
                    _ => {}
                }
            }
        }
        ExprKind::If(s) => {
            for (c, body) in &s.branches { e!(c); b!(body); }
            if let Some(body) = &s.else_body { b!(body); }
        }
        ExprKind::Match(s) => scan_match_var_arg(s, name, classify, any, other),
        ExprKind::Block(stmts) | ExprKind::Do(stmts) => b!(stmts),
        ExprKind::Loop(s) => b!(&s.body),
        ExprKind::Task(ex) => e!(ex),
        ExprKind::TaskWithTimeout(dur, ex) => { e!(dur); e!(ex); }
        ExprKind::JoinAll(exprs) => { for ex in exprs { e!(ex); } }
        ExprKind::KernelLaunch { kernel, config } => {
            e!(kernel);
            if let Some(ex) = &config.block { e!(ex); }
            if let Some(ex) = &config.grid { e!(ex); }
        }
        ExprKind::Closure(_, _, body, _, _) => match body {
            ClosureBody::Expr(ex) => e!(ex),
            ClosureBody::Block(stmts) => b!(stmts),
        },
        ExprKind::MacroCall { args, .. } => { for ex in args { e!(ex); } }
        ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Bool(_)
        | ExprKind::Nil | ExprKind::Void | ExprKind::DotIdent(_) => {}
    }
}

fn with_match_mutates(
    s: &MatchStmt,
    name: &str,
    ivp: &mut dyn FnMut(&str, usize) -> bool,
    imm: &mut dyn FnMut(&str, &str) -> bool,
) -> bool {
    with_expr_mutates(&s.subject, name, ivp, imm)
        || s.arms.iter().any(|a| {
            a.guard.as_ref().map(|g| with_expr_mutates(g, name, ivp, imm)).unwrap_or(false)
                || match &a.body {
                    MatchBody::Expr(ex) => with_expr_mutates(ex, name, ivp, imm),
                    MatchBody::Block(body) => with_block_mutates(body, name, ivp, imm),
                }
        })
}

fn with_cond_clause_mutates(
    c: &CondClause,
    name: &str,
    ivp: &mut dyn FnMut(&str, usize) -> bool,
    imm: &mut dyn FnMut(&str, &str) -> bool,
) -> bool {
    match c {
        CondClause::Expr(ex) | CondClause::Let(_, ex) | CondClause::LetPat(_, ex) => with_expr_mutates(ex, name, ivp, imm),
    }
}

fn with_expr_mutates(
    expr: &Expr,
    name: &str,
    ivp: &mut dyn FnMut(&str, usize) -> bool,
    imm: &mut dyn FnMut(&str, &str) -> bool,
) -> bool {
    let e = |ex: &Expr, ivp: &mut dyn FnMut(&str, usize) -> bool, imm: &mut dyn FnMut(&str, &str) -> bool| with_expr_mutates(ex, name, ivp, imm);
    let b = |body: &[Stmt], ivp: &mut dyn FnMut(&str, usize) -> bool, imm: &mut dyn FnMut(&str, &str) -> bool| with_block_mutates(body, name, ivp, imm);
    match &expr.kind {
        ExprKind::Assign(lhs, rhs) | ExprKind::QuestionAssign(lhs, rhs) => {
            let target_hit = match &lhs.kind {
                ExprKind::Var(v) => v == name,
                ExprKind::Index(obj, _) | ExprKind::Field(obj, _) | ExprKind::OptionalField(obj, _)
                | ExprKind::LabeledIndex(obj, _) => {
                    matches!(&obj.kind, ExprKind::Var(v) if v == name)
                }
                _ => false,
            };
            target_hit || e(lhs, ivp, imm) || e(rhs, ivp, imm)
        }
        ExprKind::Call(callee, args) => {
            let mut hit = e(callee, ivp, imm) || args.iter().any(|a| e(&a.value, ivp, imm));
            if let ExprKind::Var(fn_name) = &callee.kind {
                for (i, a) in args.iter().enumerate() {
                    if matches!(&a.value.kind, ExprKind::Var(v) if v == name) && ivp(fn_name, i) {
                        hit = true;
                    }
                }
            }
            hit
        }
        ExprKind::MethodCall(recv, method, args) | ExprKind::OptionalMethodCall(recv, method, args) => {
            let mut hit = e(recv, ivp, imm) || args.iter().any(|a| e(&a.value, ivp, imm));
            if matches!(&recv.kind, ExprKind::Var(v) if v == name) && imm(name, method) {
                hit = true;
            }
            hit
        }
        ExprKind::GenericCall(callee, _, args) => e(callee, ivp, imm) || args.iter().any(|a| e(&a.value, ivp, imm)),
        ExprKind::Pipe(lhs, _, args) => e(lhs, ivp, imm) || args.iter().any(|a| e(&a.value, ivp, imm)),
        ExprKind::New { ctor, arena } => e(ctor, ivp, imm) || arena.as_ref().map(|a| e(a, ivp, imm)).unwrap_or(false),
        ExprKind::BinOp(_, l, r) => e(l, ivp, imm) || e(r, ivp, imm),
        ExprKind::UnaryOp(_, ex) | ExprKind::Cast(ex, _) => e(ex, ivp, imm),
        ExprKind::Field(ex, _) | ExprKind::OptionalField(ex, _) => e(ex, ivp, imm),
        ExprKind::Index(obj, idx) => e(obj, ivp, imm) || e(idx, ivp, imm),
        ExprKind::LabeledIndex(obj, args) => e(obj, ivp, imm) || args.iter().any(|a| e(&a.value, ivp, imm)),
        ExprKind::Else(ex, d) | ExprKind::TryElse(ex, d) => e(ex, ivp, imm) || e(d, ivp, imm),
        ExprKind::TryElseBlock(body, els) => b(body, ivp, imm) || b(els, ivp, imm),
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) => elems.iter().any(|ex| e(ex, ivp, imm)),
        ExprKind::ArrayFill { value, count } => e(value, ivp, imm) || e(count, ivp, imm),
        ExprKind::ArrayAlloc { count } => e(count, ivp, imm),
        ExprKind::ArrayComp { expr: ex, count, .. } => e(count, ivp, imm) || e(ex, ivp, imm),
        ExprKind::ArrayCompIter { expr: ex, iter, .. } => e(iter, ivp, imm) || e(ex, ivp, imm),
        ExprKind::LabeledArrayComp { expr: ex, clauses } => {
            clauses.iter().any(|(_, count)| e(count, ivp, imm)) || e(ex, ivp, imm)
        }
        ExprKind::RelabelCast(ex, _) => e(ex, ivp, imm),
        ExprKind::Dict(pairs) => pairs.iter().any(|(k, v)| e(k, ivp, imm) || e(v, ivp, imm)),
        ExprKind::Range { start, end, .. } => e(start, ivp, imm) || e(end, ivp, imm),
        ExprKind::SliceRange { start, end, .. } => {
            start.as_ref().map(|s| e(s, ivp, imm)).unwrap_or(false) || end.as_ref().map(|ex| e(ex, ivp, imm)).unwrap_or(false)
        }
        ExprKind::StringInterp(segs) => segs.iter().any(|seg| match seg {
            StringSegment::Expr(ex) | StringSegment::FormattedExpr(ex, _) => e(ex, ivp, imm),
            _ => false,
        }),
        ExprKind::If(s) => {
            s.branches.iter().any(|(c, body)| e(c, ivp, imm) || b(body, ivp, imm))
                || s.else_body.as_ref().map(|body| b(body, ivp, imm)).unwrap_or(false)
        }
        ExprKind::Match(s) => with_match_mutates(s, name, ivp, imm),
        ExprKind::Block(stmts) | ExprKind::Do(stmts) => b(stmts, ivp, imm),
        ExprKind::Loop(s) => b(&s.body, ivp, imm),
        ExprKind::Task(ex) => e(ex, ivp, imm),
        ExprKind::TaskWithTimeout(dur, ex) => e(dur, ivp, imm) || e(ex, ivp, imm),
        ExprKind::JoinAll(exprs) => exprs.iter().any(|ex| e(ex, ivp, imm)),
        ExprKind::KernelLaunch { kernel, config } => {
            e(kernel, ivp, imm)
                || config.block.as_ref().map(|ex| e(ex, ivp, imm)).unwrap_or(false)
                || config.grid.as_ref().map(|ex| e(ex, ivp, imm)).unwrap_or(false)
        }
        ExprKind::Closure(_, _, body, _, _) => match body {
            ClosureBody::Expr(ex) => e(ex, ivp, imm),
            ClosureBody::Block(stmts) => b(stmts, ivp, imm),
        },
        ExprKind::MacroCall { args, .. } => args.iter().any(|ex| e(ex, ivp, imm)),
        ExprKind::Var(_) | ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Bool(_)
        | ExprKind::Nil | ExprKind::Void | ExprKind::DotIdent(_) => false,
    }
}
