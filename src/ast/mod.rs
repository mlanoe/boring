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
    Stmt(Stmt),
}

/// `mod name:` — groups items into a named module for Rust transpilation.
/// The interpreter executes items in the current scope (flat).
#[derive(Debug, Clone)]
pub struct ModDecl {
    pub name: String,
    pub items: Vec<Item>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct AliasDecl {
    pub name: String,       // alias name, e.g. "Callable"
    pub type_params: Vec<String>, // generic type params, e.g. ["T"] for `use Callable<T> as …`
    pub ty: Type,           // the expanded type
    pub newtype: bool,      // true for `type Name as InnerType`, false for `use Name as Type`
    pub line: usize,
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
}

#[derive(Debug, Clone)]
pub struct AsDecl {
    pub is_pub: bool,
    pub ty: Type,
    pub throws: bool,
    pub task: bool,
    pub body: Vec<Stmt>,
    pub line: usize,
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
}

#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Vec<VariantField>,
    pub attrs: Vec<Attr>,
    pub line: usize,
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
}

/// An associated type definition inside a struct: `type Output = int`.
#[derive(Debug, Clone, PartialEq)]
pub struct AssocTypeDef {
    pub name: String,
    pub ty: Type,
    pub line: usize,
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
}

/// `let (a, b, c) = expr`  — destructures a tuple into named bindings.
/// Each binding may optionally carry a type annotation: `(int x, string y)`.
#[derive(Debug, Clone)]
pub struct LetDestructureStmt {
    pub binding: BindingKind,
    pub bindings: Vec<DestructureBinding>,
    pub value: Expr,
    pub line: usize,
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
    pub else_body: Option<Vec<Stmt>>,
    pub line: usize,
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
    /// `let b' = a`  — move ownership from `a` into `b`; `a` becomes invalid after this.
    /// Without `'`, the default is a borrow: `let b = a` gives `b: T` (reference).
    pub is_move: bool,
    /// `true` for `lazy` bindings — deferred, write-once via `?=`.
    pub is_lazy: bool,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub value: Option<Expr>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct IfStmt {
    pub branches: Vec<(Expr, Vec<Stmt>)>,
    pub else_body: Option<Vec<Stmt>>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct MatchStmt {
    pub subject: Expr,
    pub arms: Vec<MatchArm>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub patterns: Vec<Pattern>,
    /// Optional guard: `pattern if cond:` — arm only fires when cond is true.
    pub guard: Option<Expr>,
    pub body: MatchBody,
    pub line: usize,
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
}

#[derive(Debug, Clone)]
pub struct DoWhileStmt {
    pub body: Vec<Stmt>,
    pub condition: Expr,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct LoopStmt {
    pub body: Vec<Stmt>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct TryStmt {
    pub body: Vec<Stmt>,
    pub catch_clauses: Vec<CatchClause>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct CatchClause {
    pub types: Vec<String>,
    /// `catch Error.Expired:` — specific variant to match inside the enum.
    /// When set, only that variant fires; unhandled variants are re-thrown.
    pub variant: Option<String>,
    pub body: Vec<Stmt>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct ThrowStmt {
    pub value: Option<Expr>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct ForStmt {
    pub vars: Vec<String>,
    pub iterable: Expr,
    pub body: Vec<Stmt>,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct GuardStmt {
    pub cond: GuardCond,
    pub else_body: Vec<Stmt>,  // must contain return/throw/break/continue
    pub line: usize,
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

    // try expr else default  — calls a throws fn, returns default on exception
    TryElse(Box<Expr>, Box<Expr>),

    // try: block else: block  — multi-line try/else; `error` is bound in the else block
    TryElseBlock(Vec<Stmt>, Vec<Stmt>),


    // Collections
    Array(Vec<Expr>),
    /// `[v for ..n]` — fill array of length `count` with `value`
    ArrayFill { value: Box<Expr>, count: Box<Expr> },
    /// `[f(i) for i in ..n]` — computed array of length `count` with `var` bound to index
    ArrayComp { expr: Box<Expr>, var: String, count: Box<Expr> },
    Tuple(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Set(Vec<Expr>),

    // Dot-prefix enum shorthand: `.Red`
    DotIdent(String),

    // Range literals
    Range { start: Box<Expr>, end: Box<Expr>, inclusive: bool },

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
    /// Qualifier union: `T'stack|heap|actor` — restricts which qualifiers callers may provide.
    /// At the Rust emission level this is a plain generic (no wrapping); the Boring compiler
    /// validates that every call site provides one of the listed qualifiers.
    /// Also used for the named groups: `'one` (`Stack|Owned`), `'many`
    /// (`Shared|Actor|Guard`), `'mut` (`Stack|Owned|Actor|Guard`), `'req` (`Shared`).
    Union(Vec<OwnerQual>),
}

// ─── Types ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Uint,
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

impl Type {
    pub fn is_copy(&self) -> bool {
        match self {
            Type::Int | Type::Uint | Type::Float | Type::Str | Type::Bool | Type::Nil | Type::Void | Type::Never => true,
            Type::Optional(inner) => inner.is_copy(),
            Type::Tuple(elems) => elems.iter().all(|t| t.is_copy()),
            Type::Array(_) | Type::ArrayN(_, _) | Type::Dict(_, _) | Type::Set(_) | Type::Named(_) => false,
            Type::Fn(..) => true,  // functions are copy (shared under the hood)
            // Owned = exclusive move → never copy
            Type::Qualified(_, OwnerQual::Owned | OwnerQual::Stack) => false,
            // Lifetime refs and borrows of smart pointers are copy at the borrow level
            Type::Qualified(_, OwnerQual::Lifetime(_) | OwnerQual::BorrowShared | OwnerQual::Borrow | OwnerQual::BorrowMut) => true,
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
            Type::Int | Type::Uint | Type::Float | Type::Str | Type::Bool | Type::Nil | Type::Void | Type::Never => true,
            Type::Fn(..) => true,
            Type::Optional(inner) => inner.is_task_safe(),
            Type::Tuple(elems) => elems.iter().all(|t| t.is_task_safe()),
            // Unqualified collections / named types are not safe (sharing semantics undefined)
            Type::Array(_) | Type::ArrayN(_, _) | Type::Dict(_, _) | Type::Set(_) | Type::Named(_) => false,
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
        }
    }
}
