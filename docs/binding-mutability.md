# Binding and mutability in Boring

> **Superseded for local bindings by the language reference's [Variables and
> Mutability](book.md#2-variables-and-mutability) chapter.** This document
> predates that model and still describes `mut`/`var` as a single
> binding-keyword axis (`is_mutable()` true for both). Under the current
> model, mutability comes from the *type* (`mut Type`), not from the keyword
> alone — `var Type a` is rebindable only; `var mut Type a` is required for
> both. The tables below are updated to match; see `book.md` for the full
> model (destructuring, tuples, struct fields, collections).

## Concepts

Two orthogonal axes:

- **Rebindable**: the variable can point to a different instance
- **Mutable**: the pointed instance can be modified

## Keywords

| Syntax | Rebindable | Mutable |
|---|---|---|
| `let` | no | no |
| `mut` | no | yes |
| `var` | yes | no |
| `var mut` | yes | yes |

The progression is intentionally graduated from most rigid to most permissive.
`var` alone no longer implies `mut` (see book.md's [Rebindable bindings — `var`](book.md#rebindable-bindings--var)) — a plain
`var Type a` is rebindable but not content-mutable; only the explicit `var
mut Type a` combination is both. Retired: the old table's "Mutable: depends
on qualifier" reading of `var`, which came from `var` auto-implying `mut`
regardless of qualifier.

## Qualifiers

Qualifiers are placed after the type (`let T'inline a`) or after the variable name if the type is inferred (`let a'inline`).

They carry three kinds of information: the Rust mapping, the passing semantics between variables, and constraints on mutability. The decision is always made at the **usage site**, not at the struct declaration — the transpiler inserts `.clone()` as needed.

| Qualifier | Rust impl | Passing semantics | Mutability |
|---|---|---|---|
| `'shared` | `Arc<T>` (multi) / `Rc<T>` (single) | pointer shared | forbidden |
| `'actor` | `Arc<std::sync::Mutex<T>>` (multi) / `Rc<RefCell<T>>` (single) | pointer shared | allowed |
| `'guard` | `Arc<std::sync::RwLock<T>>` (multi) / `Rc<RefCell<T>>` (single) | pointer shared | under lock only |
| `'inline` | `T`, no indirection | move | determined by `let`/`mut`/`var` |
| `'owned` | `Box<T>` | move | determined by `let`/`mut`/`var` |
| `'weak` | `sync::Weak<T>` (multi) / `Rc::Weak<T>` (single) | non-owning pointer | none (no interior mutability) |

`'weak` is a compound qualifier used with an owning qualifier: `T'shared'weak`, `T'actor'weak`, `T'guard'weak`. It holds a non-owning reference to the inner value of the corresponding owning smart pointer and must be upgraded before use.

> **`'actor'task` / `'guard'task`:** these are multi-thread only qualifiers (`Arc<tokio::sync::Mutex<T>>` and `Arc<tokio::sync::RwLock<T>>` respectively) and do not fall back to a single-thread form. They require explicit annotation inside `task` functions — qualifier inference does not automatically promote `'actor` or `'guard` to their `'task` variants.

`'inline` and `'owned` are the only qualifiers where `mut` has its full meaning — and where compiler optimizations from `mut` are most impactful.

## Valid combinations

`yes` below means "parses, and grants what the row's keyword nominally
promises" — for `mut`/`var mut`, that's content mutation (`def` calls); for
`var`, it's rebinding only. Per book.md's [Fixed mutable bindings — `mut`](book.md#fixed-mutable-bindings--mut) model, this is a **no
special case**: `'actor`/`'guard` are checked exactly like any other type,
not given their own exception the way an earlier draft of this table did.

| Binding | `'shared` | `'actor` | `'guard` | `'inline` | `'owned` |
|---|---|---|---|---|---|
| `let` (no mutation) | yes | yes | yes | yes | yes |
| `mut` (content-mutable, fixed) | **error** | yes | yes | yes | yes |
| `var` (rebindable only) | yes | yes | yes | yes | yes |
| `var mut` (rebindable + content-mutable) | **error** | yes | yes | yes | yes |

`mut`/`var mut` with `'shared` is an error because `'shared` is an immutable
reference-counted pointer — there is no instance to mutate through it
directly.

**Changed from the previous revision of this table:** `var T'actor x` /
`var T'guard x` alone used to be listed as sufficient for `def` calls (since
`is_mutable()` returned `true` for `Var` unconditionally). That's retired —
`var` alone is rebind-only now, full stop, matching every other type; only
`var mut T'actor x` / `var mut T'guard x` unlock `def` calls on a rebindable
actor/guard binding. `mut T'actor x` (bare, non-rebindable) is unaffected —
it already granted `def` calls and still does.

## Type-level immutability

A type that exposes no `def` method is immutable by design — the developer intentionally chose not to allow mutation. In that case:

- `mut` on such a type is redundant: no mutation is possible regardless. The compiler may emit a warning.
- `var` becomes purely rebindable: the binding can point to a different instance, but neither instance can be mutated.

This mirrors how Java works: `final` controls rebindability only, and immutability is a contract enforced by the type itself (exposing no mutating methods).

The two levels are therefore:

1. **Declaration site** (`let` / `mut` / `var`) — controls rebindability and signals mutation intent
2. **Type definition** — controls whether mutation is actually possible

## Why `mut` matters

`let` and `var` are sufficient to write correct Boring code. `mut` is an optional precision that unlocks compiler optimizations:

- **Alias analysis** — the compiler knows the pointer never changes. It can safely assume that all accesses through this variable always refer to the same memory location, enabling more aggressive optimizations across the binding's scope.

- **Loop-invariant code motion** — a `mut` binding used inside a loop has a stable base address. The compiler can hoist pointer dereferences and bounds checks out of the loop body.

- **Register allocation** — a fixed binding can be kept in a register for its entire scope. With `var`, the compiler must conservatively reload the pointer from memory on each access in case it was rebound.

- **Devirtualization / inlining** — if `mut T a` points to a concrete type, the compiler can inline or devirtualize method calls knowing the target never changes.

`var` forces the compiler to be pessimistic on all of the above, even when the binding is never actually rebound in practice.

## Function parameters

`mut` is allowed in function parameter declarations. `let` is implicit in parameter declarations — a parameter with no keyword is `let`.

Conceptually, the hierarchy `var` ≥ `mut` ≥ `let` is intended to apply: a caller should be able to pass down the hierarchy but never up (e.g. a `let`-bound caller value passed into a `var` parameter would let the callee rebind something the caller declared non-rebindable). For `'shared`, `mut` on a parameter is likewise meant to be rejected, mirroring the binding-level rule above.

**This hierarchy is not currently enforced by the compiler.** There is no cross-check today between a caller's binding kind (`let`/`mut`/`var`) and the callee's parameter binding kind — neither the checker (`src/checker` is presently a minimal stub) nor the transpiler/interpreter validate this. Treat the tables below as the intended convention/design target, not a guarantee backed by a compile error at present:

| Caller | Param `let` | Param `mut` | Param `var` |
|---|---|---|---|
| `let` | yes | not enforced | not enforced |
| `var` | yes | not enforced | yes |

**`'actor`, `'guard`, `'inline`, `'owned`** (full three levels):

| Caller | Param `let` | Param `mut` | Param `var` |
|---|---|---|---|
| `let` | yes | not enforced | not enforced |
| `mut` | yes | yes | not enforced |
| `var` | yes | yes | yes |

`let` → `var` is intended to be forbidden across all qualifiers: allowing it would let the callee rebind the caller's variable when combined with reference passing (`&`), violating the `let` non-rebindable contract. As above, this is not currently checked by the compiler.

## Pass by reference

Adding `&` passes by reference. The binding keyword defines what the callee can do with the reference:

| Syntax | Rust equivalent | Semantics |
|---|---|---|
| `let T& m` / `T& m` | `&T` | read-only — callee cannot modify content or binding |
| `mut T& m` | `&mut T` | callee can modify the content of the caller's instance |
| `var T& m` | `&mut T` | callee can rebind the caller's variable |

This is consistent with the general semantics of `let`/`mut`/`var`:
- `let` — nothing changes
- `mut` — content changes
- `var` — binding changes

`var T&` is rare in practice but unambiguous: it is the only way for a callee to replace what the caller's variable points to. In most cases, returning a value is preferred over `var T&`.

Note: at the Rust level, `mut T&` and `var T&` currently both transpile to the same `&mut T` — there is no distinct `Box`-wrapped representation for `var T&`. The distinction above is about intent (content mutation vs. rebinding) rather than a difference in the generated code.

## Return types

A function or method can return a mutable instance using `mut` after the `def`/`req` keyword:

```boring
def mut T foo()       # returns a mutable instance
req mut T get()       # req: does not modify self, returns a mutable instance
```

The `mut` here applies to the **return value**, not to `self`.

### Receiving a return value

Conceptually, the same hierarchy is intended to apply: the caller can always downgrade, never upgrade a non-mutable return into a mutable binding.

**This is not currently enforced by the compiler.** The `mut` flag on a `def mut`/`req mut` return type is parsed into the AST but is not read back anywhere in the checker, transpiler, or interpreter — so binding a non-`mut`-returning function's result with `mut`/`var` is not currently rejected. The table below describes the intended semantics, not enforced behavior:

| Return type | Caller binding | Intended validity |
|---|---|---|
| `T` | `let` | yes |
| `T` | `mut` | intended to be invalid — cannot upgrade to mutable (not enforced) |
| `T` | `var` | intended to be invalid — cannot upgrade to mutable (not enforced) |
| `mut T` | `let` | yes — caller downgrades to immutable |
| `mut T` | `mut` | yes |
| `mut T` | `var` | yes |

### `req mut` — the factory pattern

`req mut` makes sense when the method constructs and returns a **new instance** that does not belong to `self`. The typical use case is a factory:

```boring
struct Registry:
    req mut Entry get_entry():
        Entry(...)      # fresh instance, fully owned by the caller
```

The caller receives full ownership and can choose any binding:

```boring
let e  = registry.get_entry()   # ok — immutable view
mut e  = registry.get_entry()   # ok — mutable instance
var e  = registry.get_entry()   # ok — rebindable
```

`mut` is still required even for qualifier-typed returns — the qualifier defines *how* mutation works, `mut` defines *whether* the caller receives mutation rights:

```boring
def Entry'actor foo()      # caller gets let — cannot mutate despite 'actor
def mut Entry'actor foo()  # caller gets mutation rights via 'actor
def mut Entry'shared foo() # intended to be an error — 'shared forbids mut (not currently enforced by the compiler)
```

### Method combinations

| Declaration | Modifies self | Returns mutable |
|---|---|---|
| `req T foo()` | no | no |
| `req mut T foo()` | no | yes — typically a factory |
| `def T foo()` | yes | no |
| `def mut T foo()` | yes | yes |
