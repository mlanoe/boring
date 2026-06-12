# Binding and mutability in Boring

## Concepts

Two orthogonal axes:

- **Rebindable**: the variable can point to a different instance
- **Mutable**: the pointed instance can be modified

## Keywords

| Syntax | Rebindable | Mutable |
|---|---|---|
| `let` | no | no |
| `mut` | no | yes |
| `var` | yes | depends on qualifier |

The progression is intentionally graduated from most rigid to most permissive.

## Qualifiers

Qualifiers are placed after the type (`let T'stack a`) or after the variable name if the type is inferred (`let a'stack`).

They carry three kinds of information: the Rust mapping, the passing semantics between variables, and constraints on mutability. The decision is always made at the **usage site**, not at the struct declaration — the transpiler inserts `.clone()` as needed.

| Qualifier | Rust impl | Passing semantics | Mutability |
|---|---|---|---|
| `'shared` | `Rc<T>` / `Arc<T>` | pointer shared | forbidden |
| `'actor` | `Rc<RefCell<T>>` / `Arc<Mutex<T>>` | pointer shared | allowed |
| `'guard` | `Mutex<T>` / `RwLock<T>` | pointer shared | under lock only |
| `'stack` | `T` on stack | move | determined by `let`/`mut`/`var` |
| `'heap` | `Box<T>` | move | determined by `let`/`mut`/`var` |

`'stack` and `'heap` are the only qualifiers where `mut` has its full meaning — and where compiler optimizations from `mut` are most impactful.

## Valid combinations

| Binding | `'shared` | `'actor` | `'guard` | `'stack` | `'heap` |
|---|---|---|---|---|---|
| `let` | yes | yes | yes | yes | yes |
| `mut` | **error** | yes | yes | yes | yes |
| `var` | yes | yes | yes | yes | yes |

`mut` with `'shared` is an error because `'shared` is an immutable reference-counted pointer — there is no instance to mutate through it directly.

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

`mut` is allowed in function parameter declarations. The hierarchy `var` ≥ `mut` ≥ `let` applies uniformly — a caller can always pass down the hierarchy, never up. `let` is implicit in parameter declarations — a parameter with no keyword is `let`.

For `'shared`, `mut` is an error:

| Caller | Param `let` | Param `mut` | Param `var` |
|---|---|---|---|
| `let` | yes | **error** | **no** |
| `var` | yes | **error** | yes |

**`'actor`, `'guard`, `'stack`, `'heap`** (full three levels):

| Caller | Param `let` | Param `mut` | Param `var` |
|---|---|---|---|
| `let` | yes | **no** | **no** |
| `mut` | yes | yes | **no** |
| `var` | yes | yes | yes |

`let` → `var` is forbidden across all qualifiers: allowing it would let the callee rebind the caller's variable when combined with reference passing (`&`), violating the `let` non-rebindable contract.

## Pass by reference

Adding `&` passes by reference. The binding keyword defines what the callee can do with the reference:

| Syntax | Rust equivalent | Semantics |
|---|---|---|
| `let T& m` / `T& m` | `&T` | read-only — callee cannot modify content or binding |
| `mut T& m` | `&mut T` | callee can modify the content of the caller's instance |
| `var T& m` | `&mut Box<T>` / Swift `inout` | callee can rebind the caller's variable |

This is consistent with the general semantics of `let`/`mut`/`var`:
- `let` — nothing changes
- `mut` — content changes
- `var` — binding changes

`var T&` is rare in practice but unambiguous: it is the only way for a callee to replace what the caller's variable points to. In most cases, returning a value is preferred over `var T&`.

## Return types

A function or method can return a mutable instance using `mut` after the `def`/`req` keyword:

```boring
def mut T foo()       # returns a mutable instance
req mut T get()       # req: does not modify self, returns a mutable instance
```

The `mut` here applies to the **return value**, not to `self`.

### Receiving a return value

The same hierarchy applies: the caller can always downgrade, never upgrade.

| Return type | Caller binding | Valid |
|---|---|---|
| `T` | `let` | yes |
| `T` | `mut` | **no** — cannot upgrade to mutable |
| `T` | `var` | **no** — cannot upgrade to mutable |
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
def mut Entry'shared foo() # error — 'shared forbids mut
```

### Method combinations

| Declaration | Modifies self | Returns mutable |
|---|---|---|
| `req T foo()` | no | no |
| `req mut T foo()` | no | yes — typically a factory |
| `def T foo()` | yes | no |
| `def mut T foo()` | yes | yes |
