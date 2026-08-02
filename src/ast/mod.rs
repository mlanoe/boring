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
    pub mutable: bool,
    pub transient: bool,
    pub ty: Type,
    pub default: Option<Expr>,
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

impl BindingKind {
    /// Returns `true` for `Mut` and `Var` — both produce a mutable Rust binding.
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
    pub bindings: Vec<DestructureBinding>,
    pub value: Expr,
    pub line: usize,
    pub col: usize,
}

/// One slot in a destructure: a name with an optional type.
/// Use `_` as name for a wildcard slot.
#[derive(Debug, Clone)]
pub struct DestructureBinding {
    pub name: String,          // variable name, or "_" for wildcard
    pub ty: Option<Type>,
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

    // Calls
    Call(Box<Expr>, Vec<Arg>),
    MethodCall(Box<Expr>, String, Vec<Arg>),
    // Generic call: `f<T1, T2>(args)` — type arguments resolved at emit time
    GenericCall(Box<Expr>, Vec<Type>, Vec<Arg>),

    // Pipe operator: `lhs |> f(args)`
    // Desugars at emit time: if f is in fn_sigs → f(lhs, args), else → lhs.f(args)
    Pipe(Box<Expr>, String, Vec<Arg>),

    /// `new Constructor()` — placement expression, qualifier inferred excluding 'stack.
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
/// Boring defaults to stack allocation (like Rust). Heap allocation is explicit.
///
/// `Dog`         → (no qual)       (Dog        — stack-owned, default; Rust default)
/// `Dog&`        → Borrow          (&Dog       — borrow / reference, alias-compatible)
/// `Dog'`        → Owned           (Box<Dog>   — heap-owned; same as Dog'heap)
/// `Dog'heap`    → Owned           (Box<Dog>   — explicit heap alias for bare tick)
/// `Dog'stack`   → Stack           (Dog        — explicit stack, equivalent to bare Dog)
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
    /// Force stack allocation for the Rust transpiler: `Point'stack` → plain `Point` in Rust.
    /// By default structs are heap-allocated (`Box<T>`); `'stack` opts out of that.
    /// The interpreter treats this identically to `Owned` at runtime (no difference).
    Stack,
    /// Pseudo-qualifier written as `'new` (or implied by `new Constructor()` on the RHS).
    /// Means "infer excluding 'stack" — identical inference starting set to the bare `T'` tick.
    /// Used in delayed-init position: `Counter'new v`.
    New,
    /// Explicit lifetime annotation for Rust transpilation: `string'a` → `&'a str`.
    /// The interpreter treats this identically to a plain borrow (no runtime enforcement).
    Lifetime(String),
    /// Internal: threading-aware borrow of the smart pointer → `&Arc<T>` / `&Rc<T>`.
    /// No longer produced by the parser. Kept for backwards compatibility with serialized ASTs.
    BorrowShared,
    /// Universal borrow: `T&` → `&T`. The transpiler coerces any qualifier at the call site.
    Borrow,
    /// Internal: borrow of a heap (Box) value → `&Box<T>`.
    /// No longer produced by the parser (`T'heap&` / `T&heap` are removed).
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
    /// Qualifier union: `T'stack|heap|actor` — restricts which qualifiers callers may provide.
    /// At the Rust emission level this is a plain generic (no wrapping); the Boring compiler
    /// validates that every call site provides one of the listed qualifiers.
    /// Also used for the named groups: `'one` (`Stack|Owned`), `'many`
    /// (`Shared|Actor|Guard`), `'mut` (`Stack|Owned|Actor|Guard`), `'req` (`Shared`).
    Union(Vec<OwnerQual>),
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
    Float,
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
}

/// Copy-ness of a single owner qualifier, matching the rules in `Type::is_copy`.
/// Used to evaluate the members of an `OwnerQual::Union` (`'stack|actor`, `'mut`, ...).
fn owner_qual_is_copy(q: &OwnerQual) -> bool {
    match q {
        OwnerQual::Owned | OwnerQual::Stack => false,
        OwnerQual::Union(quals) => quals.iter().all(owner_qual_is_copy),
        _ => true,
    }
}

impl Type {
    pub fn is_copy(&self) -> bool {
        match self {
            Type::Int | Type::Uint | Type::Uint8
                | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128
                | Type::Float | Type::Str | Type::Bool | Type::Nil | Type::Void | Type::Never => true,
            Type::Optional(inner) => inner.is_copy(),
            Type::Tuple(elems) => elems.iter().all(|t| t.is_copy()),
            Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::Dict(_, _) | Type::Set(_) | Type::Named(_) => false,
            Type::ConstInt(_) => true,
            Type::Fn(..) => true,  // functions are copy (shared under the hood)
            // Owned = exclusive move → never copy
            Type::Qualified(_, OwnerQual::Owned | OwnerQual::Stack) => false,
            // Lifetime refs and borrows of smart pointers are copy at the borrow level
            Type::Qualified(_, OwnerQual::Lifetime(_) | OwnerQual::BorrowShared | OwnerQual::Borrow | OwnerQual::BorrowMut) => true,
            // A qualifier union is only Copy if every member qualifier it allows is Copy —
            // e.g. `'stack|actor` includes 'stack (move-only), so the union as a whole is not Copy.
            Type::Qualified(_, OwnerQual::Union(quals)) => quals.iter().all(owner_qual_is_copy),
            // All other qualifiers give copy/shared semantics
            Type::Qualified(_, _) => true,
            Type::TypeParam(_) => true,   // assumed copy at runtime, erased
            Type::Generic(_, _) => false, // heap type
            Type::Dyn(inner) | Type::Impl(inner) => inner.is_copy(),
            Type::SelfAssoc(_)  => false, // conservative, like Named
            Type::AssocOf(_, _) => false, // conservative, like Named
        }
    }

    /// True if the type is task-safe (can be captured by a task).
    pub fn is_task_safe(&self) -> bool {
        match self {
            // Primitive copy types are always safe
            Type::Int | Type::Uint | Type::Uint8
                | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 | Type::Int128
                | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Uint128
                | Type::Float | Type::Str | Type::Bool | Type::Nil | Type::Void | Type::Never => true,
            Type::Fn(..) => true,
            Type::Optional(inner) => inner.is_task_safe(),
            Type::Tuple(elems) => elems.iter().all(|t| t.is_task_safe()),
            // Unqualified collections / named types are not safe (sharing semantics undefined)
            Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::Dict(_, _) | Type::Set(_) | Type::Named(_) => false,
            Type::ConstInt(_) => true,
            // Qualifiers
            Type::Qualified(_, OwnerQual::Owned | OwnerQual::Stack) => true,  // exclusive move → source invalidated
            Type::Qualified(_, OwnerQual::Shared)     => true,  // Arc<T> (multi) / Rc<T> (single) — qualifier intent is task-safe
            Type::Qualified(_, OwnerQual::Actor | OwnerQual::ActorTask) => true,
            Type::Qualified(_, OwnerQual::Guard | OwnerQual::GuardTask) => true,
            Type::Qualified(_, OwnerQual::Weak)       => false, // Weak<T> — non-owning, conservative
            Type::Qualified(_, OwnerQual::Lifetime(_)) => true, // borrow — task-safe for transpilation
            Type::Qualified(_, OwnerQual::BorrowShared) => true, // &Arc<T> / &Rc<T> — threading-aware borrow
            Type::Qualified(_, OwnerQual::BorrowOwned)  => false,
            Type::Qualified(_, OwnerQual::BorrowOption | OwnerQual::BorrowOptionMut) => false,
            Type::Qualified(_, OwnerQual::BorrowWeak)   => false,
            Type::Qualified(_, OwnerQual::Borrow)       => false, // unknown until alias resolved — conservative
            Type::Qualified(_, OwnerQual::BorrowMut)    => false, // &mut T — conservative (target unknown)
            Type::Qualified(inner, OwnerQual::Union(_)) => inner.is_task_safe(), // union: delegate to inner
            Type::TypeParam(_) => true,
            Type::Generic(_, _) => false, // unless qualified, keep simple for now
            Type::Dyn(inner) | Type::Impl(inner) => inner.is_task_safe(),
            Type::SelfAssoc(_)  => false, // conservative, like Named
            Type::AssocOf(_, _) => false, // conservative, like Named
            Type::Qualified(_, OwnerQual::New) => false, // pseudo-qualifier: conservative, like Named
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
        ExprKind::Else(ex, d) | ExprKind::TryElse(ex, d) => { e!(ex); e!(d); }
        ExprKind::TryElseBlock(body, els) => { b!(body); b!(els); }
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) => { for ex in elems { e!(ex); } }
        ExprKind::ArrayFill { value, count } => { e!(value); e!(count); }
        ExprKind::ArrayAlloc { count } => e!(count),
        ExprKind::ArrayComp { expr: ex, count, .. } => { e!(count); e!(ex); }
        ExprKind::ArrayCompIter { expr: ex, iter, .. } => { e!(iter); e!(ex); }
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
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Bool(_)
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
                ExprKind::Index(obj, _) | ExprKind::Field(obj, _) | ExprKind::OptionalField(obj, _) => {
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
        ExprKind::Else(ex, d) | ExprKind::TryElse(ex, d) => e(ex, ivp, imm) || e(d, ivp, imm),
        ExprKind::TryElseBlock(body, els) => b(body, ivp, imm) || b(els, ivp, imm),
        ExprKind::Array(elems) | ExprKind::Tuple(elems) | ExprKind::Set(elems) => elems.iter().any(|ex| e(ex, ivp, imm)),
        ExprKind::ArrayFill { value, count } => e(value, ivp, imm) || e(count, ivp, imm),
        ExprKind::ArrayAlloc { count } => e(count, ivp, imm),
        ExprKind::ArrayComp { expr: ex, count, .. } => e(count, ivp, imm) || e(ex, ivp, imm),
        ExprKind::ArrayCompIter { expr: ex, iter, .. } => e(iter, ivp, imm) || e(ex, ivp, imm),
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
        ExprKind::Var(_) | ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Str(_) | ExprKind::Bool(_)
        | ExprKind::Nil | ExprKind::Void | ExprKind::DotIdent(_) => false,
    }
}
