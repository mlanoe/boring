# Qualifier Coercion — Temporary References Across Wrappers

> **Status: implemented.** All items below are transpiled. Open questions are resolved.

---

## Motivation

Boring's qualifier system encodes ownership and sharing in the type of every value. A `Counter'actor` is an `Arc<Mutex<Counter>>`; a `Counter'stack` is a plain `Counter`. These are distinct Rust types with no implicit relationship.

This creates friction at function boundaries. A utility function that only reads a `Counter` for display purposes should not need to know whether the caller holds an `Arc<Mutex<Counter>>` or a plain `Counter`. In idiomatic Rust, such functions take `&Counter` — a temporary borrow that any wrapper can produce:

| Rust wrapper | How to get `&T` | Cost |
|---|---|---|
| `T` (stack) | `&val` | none |
| `Box<T>` | `&**box_val` | none |
| `Arc<T>` / `Rc<T>` | `&**arc` | none |
| `Arc<Mutex<T>>` | `let g = m.lock()?; &*g` | lock acquisition |
| `Arc<RwLock<T>>` | `let g = rw.read()?; &*g` | read-lock acquisition |

The lock guard (`MutexGuard`, `RwLockReadGuard`) implements `Deref<Target=T>`, so `&*guard` yields a `&T` valid for as long as the guard is alive. The lock is released when the guard drops — automatically at the end of the enclosing block.

The idea: Boring could express this pattern natively. A function that takes a **coerced reference** would accept any qualifier, and the transpiler would insert the deref — and the lock, if needed — at the call site.

---

## Parameter passing summary

| Parameter syntax | Rust emitted | Semantics |
|---|---|---|
| `Counter c` | inferred — see below | qualifier inferred from body; or `Counter&` / `mut Counter&` if no storage signal |
| `mut Counter c` | inferred — see below | mutable; infers `mut Counter&` if no storage, mutable qualifier otherwise |
| `Counter'stack c` | `Counter` | move (or copy for primitives) |
| `Counter'heap c` | `Box<Counter>` | move |
| `Counter'shared c` | `&Arc<Counter>` | auto-ref, transparent to the developer |
| `Counter'actor c` | `&Arc<Mutex<Counter>>` | auto-ref, callee controls lock granularity |
| `Counter'guard c` | `&Arc<RwLock<Counter>>` | auto-ref, callee controls lock granularity |
| `Counter'shared'weak c` | `&Weak<Counter>` | auto-ref, callee calls `.upgrade()` explicitly |
| `Counter'actor'weak c` | `&Weak<Mutex<Counter>>` | auto-ref, callee calls `.upgrade()` explicitly |
| `Counter'guard'weak c` | `&Weak<RwLock<Counter>>` | auto-ref, callee calls `.upgrade()` explicitly |
| `Counter& c` | `&Counter` | universal borrow, any qualifier, no move, no storage |
| `mut Counter& c` | `&mut Counter` | universal mutable borrow, any mutable qualifier |

`'stack` and `'heap` follow standard Rust move semantics. `'shared`, `'actor`, and `'guard` are always passed by reference — the reference is fully transparent to the developer, who writes and reads these parameters as owned values. `Counter&` is the explicit form for borrowing the inner value universally, regardless of qualifier.

In struct fields and local bindings all qualifiers remain owned. A reference in a struct field would require a lifetime annotation on the struct — exactly what Boring's qualifier system is designed to avoid.

**Struct and enum method parameters are excluded from automatic `Counter&` inference.** The call-site coercion that injects `&` applies only to free-function call sites; method calls (`p.distance(q)`) are not rewritten. Use the explicit `Counter& n` form to get universal borrowing in a method parameter.

---

## `Counter&` — universal borrow

`Counter&` always produces `&Counter`, regardless of which qualifier the caller holds. The transpiler unwraps the qualifier at the call site:

```boring
req display(Counter& c):
    print c.value
```

```boring
let a = Counter'stack(0)
let b = Counter'actor(0)
let c = Counter'heap(0)
let d = Counter'shared(0)

display(a)   # &a
display(b)   # { let g = b.lock()?; display(&*g) }
display(c)   # &**c
display(d)   # &**d  — Arc<T> derefs to T, no lock needed
```

The caller never writes the lock — it is implicit and scoped to the call.

For generics, `T&` is also systematic: whatever qualifier `T` is instantiated with, the transpiler unwraps it to `&T_base`. A `T&` parameter always receives `&Counter`, never `&Arc<Counter>` or `&Arc<Mutex<Counter>>`.

### `Counter&` vs `Counter'actor` as parameter

These two forms are complementary:

```boring
# Counter& c — transpiler acquires the lock, passes &Counter
# lock held for the entire duration of the call
def process_batch(Counter& c):
    for item in batch:
        c.value += item

# Counter'actor c — callee receives &Arc<Mutex<Counter>>
# callee controls lock granularity
def process_batch(Counter'actor c):
    for item in batch:
        let g = c.lock()
        g.value += item
        # lock released here, not at end of call
```

### Why not `Counter c` (bare, no marker)?

`Counter c` without a qualifier is subject to qualifier inference. The inference algorithm has been extended to include universal borrowing as a possible output: when the parameter is never stored and all body usages are compatible with a borrow, the transpiler resolves `Counter c` to `Counter&` (or `mut Counter&`) rather than falling back to a concrete qualifier.

```boring
req display(Counter c):       # c never stored, read-only
    print c.value
# infers Counter& → fn display(c: &Counter)
# any qualifier accepted at call site

def reset(mut Counter c):     # c never stored, mut declared
    c.value = 0
# infers mut Counter& → fn reset(c: &mut Counter)
# any mutable qualifier accepted at call site
```

The `mut` keyword is not inferred — it must be written explicitly. A `def` call on a `Counter c` parameter (without `mut`) is a compile error: `"parameter 'c' is immutable but 'inc' is a def method — declare mut Counter c"`. This keeps mutability visible in the signature regardless of what the body does.

`Counter& c` remains the explicit form — use it to document intent, or to lock in universal borrowing regardless of future body changes that might add a qualifier-demanding usage.

---

## Auto-ref for `'shared`, `'actor`, `'guard`

The rationale: moving an `Arc` silently increments the reference counter, a cost invisible in the source. A reference suffices in the vast majority of call sites. The auto-ref convention makes this the default — the developer writes `Counter'actor c` and treats `c` as an owned value throughout the body. Refcount increments and `Arc::clone` calls are inserted by the transpiler only when needed.

The transpiler inserts `Arc::clone` (or `Rc::clone` in single-thread mode) automatically whenever a reference parameter is used in an owned position. This covers field assignment, `let` bindings, `match` and `if let` bindings, tuple construction, and call-site arguments:

```boring
struct Processor:
    Counter'actor counter

def init(Counter'actor c):
    counter = c              # field assign → Arc::clone(c)
    let x = c                # let binding → Arc::clone(c)
    let (a, b) = (c, other)  # tuple → Arc::clone(c)

match c:
    _: store(c)              # match binding → Arc::clone(c)
```

Call sites where a `'actor`/`'guard`/`'shared` value is passed to a function expecting an owned parameter also trigger auto-clone:

```boring
def store(Counter'actor c):
    …

let x = Counter'actor(0)
store(x)     # Arc::clone(&x) at call site
store(x)     # x is still valid — clone was inserted, not a move
```

---

## Lock scope and guard lifetime

When the argument is `'actor` or `'guard` and the parameter is `Counter&`, the transpiler generates a temporary binding before the call. The guard must outlive the call but be dropped immediately after:

```rust
// display(b) where b: Arc<Mutex<Counter>>
{
    let __g = b.lock()?;
    display(&*__g);
}   // lock released here
```

Inlining (`display(&*b.lock()?.deref())`) would drop the guard before the call in some Rust editions — the explicit binding avoids the issue.

For `'guard` (`RwLock`), a `Counter&` parameter uses `read()` (shared read lock). A `mut Counter&` parameter uses `write()` (exclusive write lock).

---

## Mutable coercion

`mut Counter&` marks a mutable coerced reference, consistent with `mut x = …` and `def mut T foo()`. The transpiler generates a mutable borrow or a write-lock as needed:

```boring
def reset(mut Counter& c):
    c.value = 0

mut a = Counter'stack(0)
let b = Counter'actor(0)

reset(a)   # &mut a
reset(b)   # { let mut g = b.lock()?; reset(&mut *g) }
```

`'shared` (`Arc<T>` without interior mutability) cannot produce `&mut T` — passing a `Counter'shared` to a `mut Counter&` parameter is a compile error.

---

## Interaction with qualifier inference

`Counter&` does not participate in qualifier inference. The `&` replaces inference: the parameter has no qualifier to resolve, only a reference to produce. The qualifier of the argument at the call site is unchanged after the call.

A call to a `Counter&` parameter contributes a signal to the argument's inference: it signals read-only access (no mutation, no sharing demand). If the argument has no other signals, the size-based fallback applies.

---

## Resolved questions

**1. `var` parameters — implemented**

`var` on a parameter signals an out-parameter: the callee can rebind the caller's variable. The transpiler emits a mutable reference to the wrapper itself. For primitives (`var int x`), `var` means a mutable local copy, not an out-parameter.

| Parameter | Rust emitted |
|---|---|
| `var Counter'stack c` | `&mut Counter` |
| `var Counter'heap c` | `&mut Box<Counter>` |
| `var Counter'shared c` | `&mut Arc<Counter>` |
| `var Counter'actor c` | `&mut Arc<Mutex<Counter>>` |
| `var Counter'guard c` | `&mut Arc<RwLock<Counter>>` |
| `var Counter'shared'weak c` | `&mut Weak<Counter>` |
| `var Counter'actor'weak c` | `&mut Weak<Mutex<Counter>>` |
| `var Counter'guard'weak c` | `&mut Weak<RwLock<Counter>>` |

```boring
def swap(var Counter'actor c):
    c = Counter(1)

var v'actor = Counter(0)
swap(v)   # call site emits: swap(&mut v)
```

```rust
fn swap(c: &mut Arc<Mutex<Counter>>) {
    *c = Arc::new(Mutex::new(Counter(1)));
}
```

**2. Nested field access — resolved by design**

If the function receives `Counter&` and accesses `c.stats` where `stats` is `Counter'actor`, the lock on `stats` must be taken inside the function body — the outer coercion does not propagate into fields. No transpiler change needed.

**3. Weak references at call sites — implemented**

Passing a `'weak` value to any non-weak parameter (`Counter&`, `Counter'shared`, `Counter'actor`, `Counter'guard`) is a compile error. The transpiler emits:

```
error: cannot pass `x` (weak reference) to a non-weak parameter — weak references may be
       invalid. Call .upgrade() first and handle the Option.
```

**4. Error messages for `'shared` → mutable coercion — implemented**

Passing a `Counter'shared` to a `mut Counter&` parameter is a compile error. The transpiler emits:

```
error: cannot pass `x` ('shared) to `mut Counter&` — 'shared does not support mutable
       references. Use 'actor (Arc<Mutex<T>>) or 'guard (Arc<RwLock<T>>) instead.
```

**5. Coercion across async boundaries — implemented**

Passing a `'actor` or `'guard` argument to a `mut Counter&` parameter in an async function is rejected: holding a `MutexGuard`/`RwLockGuard` across `.await` makes the future `!Send`. The transpiler emits:

```
error: cannot pass 'actor argument to `mut Counter&` in async function `f` — holding a
       MutexGuard across .await makes the future !Send. Acquire the lock inside the
       callee body instead.
```

---

## Preliminary assessment

`Counter&` and `mut Counter&` handle universal borrowing to the inner type, unwrapping any qualifier at the call site. For `'shared`, `'actor`, and `'guard`, the auto-ref convention is the systematic default in parameter position — the reference is transparent, and `Arc::clone` is inserted implicitly when the callee stores the value. `Counter'actor` and `Counter'guard` give the callee explicit control over lock granularity. The main cost of `Counter&` is hidden lock acquisitions at call sites — visible in generated Rust but not in Boring source. A warning when a `Counter&` call coerces from `'actor` or `'guard` would make the cost explicit without requiring a syntax change.
