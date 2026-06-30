# Qualifiers — Complete Reference

> **Status: implemented** for `'actor`, `'guard`, `'actor'task`, `'guard'task`, `'shared`, `'weak`, `'stack`, `'heap`.
> Sections marked ⚠️ are open questions or future work.

---

## Overview

Every Boring type carries an optional **qualifier** that describes how the value is stored and shared at runtime. The qualifier is written after the type name, separated by a tick:

```boring
let Counter'actor  c = Counter()   # Arc<Mutex<Counter>>
let Counter'shared r = Counter()   # Arc<Counter>
let Counter'stack  s = Counter()   # Counter  (on the stack)
```

Qualifiers are resolved at transpile time. The interpreter ignores them (all values are reference-counted by the runtime). In generated Rust the qualifier determines the exact wrapper type.

---

## Qualifier table

| Boring | Rust (multi-thread) | Rust (single-thread) | Mutable | Notes |
|---|---|---|---|---|
| `'stack` | `T` | `T` | via `mut` binding | Rust default, no wrapper |
| `'heap` / `T'` | `Box<T>` | `Box<T>` | via `mut` binding | heap-allocated, exclusive |
| `'shared` | `Arc<T>` | `Rc<T>` | no | read-only shared ownership |
| `'actor` | `Arc<std::sync::Mutex<T>>` | `Rc<RefCell<T>>` | interior mutability | sync, no tokio required |
| `'actor'task` / `'task` | `Arc<tokio::sync::Mutex<T>>` | `Rc<RefCell<T>>` | interior mutability | async context |
| `'guard` | `Arc<std::sync::RwLock<T>>` | `Rc<RefCell<T>>` | interior mutability | reader-writer, sync |
| `'guard'task` | `Arc<tokio::sync::RwLock<T>>` | `Rc<RefCell<T>>` | interior mutability | async context |
| `'weak` | `Weak<T>` / `sync::Weak<T>` | `Rc::Weak<T>` | no | non-owning, inferred from RHS |

`'actor'task` is an alias for `'task`. Both produce the tokio async lock.

---

## Binding × qualifier interaction

| Syntax | Qualifier constraint | Notes |
|---|---|---|
| `let x'actor = …` | `'actor` | immutable binding, interior mutability via lock |
| `mut x'actor = …` | `'actor` | mutable binding (same Arc, rebindable) |
| `var x'actor = …` | `'actor` | rebindable Arc pointer |
| `let x'shared = …` | `'shared` | read-only, no `def` methods |
| `mut x'shared` | compile error | `'shared` + mutability is incoherent |

### `'actor` and `'guard` on `let` bindings

`let` bindings are normally immutable. `'actor` and `'guard` are exceptions: they provide **interior mutability**, so `def` methods may be called even on a `let`-bound variable. The lock/borrow is acquired automatically.

```boring
let Counter'actor c = Counter()
c.inc()    # OK — def method, interior mutability via Mutex
c.inc()
print c.get()    # → 2
```

This is distinct from `var`/`mut` binding mutability — the `let` binding is not rebindable, but the inner value is mutable through the lock.

---

## Sync vs async variants

Use `'actor` / `'guard` when the code does not use `.await` while the lock is held:

```boring
let Counter'actor c = Counter()
c.inc()                   # std::sync::Mutex — no await needed
```

Use `'actor'task` / `'task` or `'guard'task` inside `task` functions when you need to hold the lock across `.await`:

```boring
task def void worker(Counter'task c):
    c.inc()               # tokio::sync::Mutex — .lock().await
    wait(Duration.fromMillis(100))
    print c.get()
```

**Why the distinction matters in Rust:**
`std::sync::MutexGuard` is `!Send` — you cannot hold it across an `.await` point without making the future `!Send`. `tokio::sync::MutexGuard` is `Send`, designed for async use.

| Qualifier | Lock type | Hold across `.await`? |
|---|---|---|
| `'actor` | `std::sync::Mutex` | ❌ |
| `'actor'task` / `'task` | `tokio::sync::Mutex` | ✅ |
| `'guard` | `std::sync::RwLock` | ❌ |
| `'guard'task` | `tokio::sync::RwLock` | ✅ |

### Inferring `'actor'task` / `'guard'task` from task-method calls

When a variable (or `self.field`) is captured by a `task` expression or closure and used as a method receiver, the inference pass keeps **both** the plain (`'actor`/`'guard`) and `'task` (`'actor'task`/`'guard'task`) variant as candidates instead of jumping straight to the sync lock. It then looks at which methods are actually called on the captured variable inside that body:

```boring
struct Counter:
    var int value = 0

    task def inc():          # declared `task` → needs the tokio lock
        value += 1

def void run(Counter c):
    task c.inc()              # c infers 'actor'task — Arc<tokio::sync::Mutex<Counter>>
```

```boring
struct Counter:
    var int value = 0

    def inc():                # plain `def`, not `task`
        value += 1

def void run(Counter c):
    task c.inc()              # c infers 'actor' — Arc<std::sync::Mutex<Counter>>
```

If any method called on the captured variable is itself declared `task`, the inferrer picks `'actor'task` (or `'guard'task` if the receiver is otherwise constrained to a reader-writer lock); if none are, it falls back to the plain sync variant. This only resolves the ambiguity when the disambiguating signal is a method call — a task body that reads/writes the captured value without calling a `task` method, then separately awaits something unrelated while still holding the lock, still needs an explicit `'actor'task`/`'guard'task` annotation.

---

## Method dispatch

### On local variables

```boring
let Store'guard s = Store()
s.write(42)      # def → RwLock::write().unwrap()
s.read()         # req → RwLock::read().unwrap()

let Store'guard'task st = Store()
st.write(42)     # def → RwLock::write().await
st.read()        # req → RwLock::read().await
```

The transpiler distinguishes `req` (read-only, `&self`) from `def` (mutating, `&mut self`) to select the appropriate lock mode:

| Method kind | `'actor` | `'actor'task` | `'guard` | `'guard'task` |
|---|---|---|---|---|
| `req` | `lock().unwrap()` | `lock().await` | `read().unwrap()` | `read().await` |
| `def` | `lock().unwrap()` | `lock().await` | `write().unwrap()` | `write().await` |

### On struct fields

```boring
struct Node:
    Counter'actor stats

    def record():
        stats.inc()     # self.stats.lock().unwrap().inc()

    req int total():
        stats.get()     # self.stats.lock().unwrap().get()
```

Field dispatch follows the same `req`/`def` split as local variables.

### Field reads (non-method)

```boring
struct Tag:
    string label

let Tag'guard t = Tag(label = "x")
print t.label            # t.read().unwrap().label
```

Direct field access on an `'actor` or `'guard` variable always acquires the appropriate lock.

---

## Move semantics

By default, assigning a value moves it — the source binding becomes invalid after the assignment. This applies to all qualifiers.

```boring
let a = Counter(0)
let b = a          # a is moved into b — a is no longer accessible
```

To share a value without moving it, call `.clone()` explicitly:

```boring
let a = Counter(0)
let b = a.clone()  # deep copy — a and b are independent
```

### Clone semantics by qualifier

| Qualifier | `.clone()` cost | Result |
|---|---|---|
| `'stack` | deep copy — allocates new value | independent copy |
| `'heap` | deep copy — allocates new `Box<T>` + clones content | independent heap allocation |
| `'shared` | O(1) — increments `Arc` refcount | shared reference to the same value |
| `'actor` | O(1) — increments `Arc` refcount | shared reference to the same mutex |
| `'guard` | O(1) — increments `Arc` refcount | shared reference to the same rwlock |

For `'shared`, `'actor`, and `'guard`, `.clone()` is cheap — it clones the pointer, not the data. All clones refer to the same underlying value.

---

## Qualifier upgrade coercions

A value can be promoted to a richer qualifier at construction time. These are **explicit** coercions — the developer calls them when moving a value from one ownership context to another.

### Upgrade table

| From | To | Boring | Rust emitted | Notes |
|---|---|---|---|---|
| `'stack` | `'heap` | `let b'heap = a` | `Box::new(a)` | move into heap |
| `'stack` | `'shared` | `let b'shared = a` | `Arc::new(a.clone())` | source is cloned, not moved, in the emitted Rust |
| `'stack` | `'actor` | `let b'actor = a` | `Arc::new(std::sync::Mutex::new(a.clone()))` | source is cloned, not moved, in the emitted Rust |
| `'stack` | `'guard` | `let b'guard = a` | `Arc::new(std::sync::RwLock::new(a.clone()))` | source is cloned, not moved, in the emitted Rust |
| `'heap` | `'shared` | `let b'shared = a` | `Arc::from(a)` | no double allocation — Rust optimisation |
| `'heap` | `'actor` | `let b'actor = a` | `Arc::new(Mutex::new(*a))` | unboxes then wraps |
| `'heap` | `'guard` | `let b'guard = a` | `Arc::new(RwLock::new(*a))` | unboxes then wraps |
| `'heap` | `'stack` | `let b'stack = a` | `a.clone()` | clones the boxed value; `a` remains a valid, unmoved `Box<Counter>` in the emitted Rust |

At the Boring-semantics level, all upgrades consume the source value — the interpreter marks the source binding as moved, and reading it afterward raises "use of moved value". However, this is not always a move in the *emitted Rust*: for `'stack` → `'shared`/`'actor`/`'guard` and `'heap` → `'stack`, the transpiler emits `.clone()` on the source, so the original Rust variable remains alive and valid under the hood even though Boring forbids reading it. `Arc::from(box_val)` (used for `'heap` → `'shared`) is the idiomatic Rust way to convert `Box<T>` into `Arc<T>` without a double allocation — the `Arc` reuses the existing heap allocation, and this is the one upgrade that is a true move in the emitted Rust.

### Downgrade

Downgrades (e.g. `'shared` → `'stack`) are not available implicitly. Shared references (`Arc`) cannot be converted back to owned values without an explicit `.clone()` or `.try_unwrap()` (which fails if other references exist).

```boring
let a'shared = Counter(0)
let b = a.clone()    # Arc::clone — b is still 'shared (Arc<Counter>), sharing the same value as a
```

`.clone()` on an `'shared` value is a pointer clone (see the clone-cost table above), not a deep copy — there is no implicit way to obtain an independent `'stack` copy from an `'shared` value. To get one, deref and clone the inner value explicitly (e.g. via a method that returns an owned copy).

---

## Parameter passing

### Full parameter table

| Parameter syntax | Rust emitted | Semantics |
|---|---|---|
| `Counter c` | inferred — see [Inference](#inference) | qualifier inferred from body; or `Counter&` / `mut Counter&` if no storage signal |
| `mut Counter c` | inferred — see [Inference](#inference) | mutable; infers `mut Counter&` if no storage, mutable qualifier otherwise |
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

`'stack` and `'heap` follow standard Rust move semantics. `'shared`, `'actor`, and `'guard` are always passed by reference — the reference is fully transparent to the developer, who writes and reads these parameters as owned values.

### Auto-ref for `'shared`, `'actor`, `'guard`

The rationale: moving an `Arc` silently increments the reference counter, a cost invisible in the source. A reference suffices in the vast majority of call sites. The auto-ref convention makes this the default.

The transpiler inserts `Arc::clone` (or `Rc::clone` in single-thread mode) automatically whenever a reference parameter is used in an owned position: field assignment, `let` bindings, `match` and `if let` bindings, tuple construction, and call-site arguments.

```boring
struct Processor:
    Counter'actor counter

def init(Counter'actor c):
    counter = c              # field assign → Arc::clone(c)
    let x = c                # let binding → Arc::clone(c)
```

```boring
def store(Counter'actor c):
    …

let x'actor = Counter(0)
store(x)     # Arc::clone(&x) at call site
store(x)     # x is still valid — clone was inserted, not a move
```

### `var` out-parameters

`var` on a parameter signals an out-parameter — the callee can rebind the caller's variable:

| Parameter | Rust emitted |
|---|---|
| `var Counter'stack c` | `&mut Counter` |
| `var Counter'heap c` | `&mut Box<Counter>` |
| `var Counter'shared c` | `&mut Arc<Counter>` |
| `var Counter'actor c` | `&mut Arc<Mutex<Counter>>` |
| `var Counter'guard c` | `&mut Arc<RwLock<Counter>>` |

```boring
def swap(var Counter'actor c):
    c = Counter(1)

var v'actor = Counter(0)
swap(v)   # call site emits: swap(&mut v)
```

### `Counter&` — universal borrow

`Counter&` always produces `&Counter`, regardless of which qualifier the caller holds. The transpiler unwraps the qualifier at the call site:

```boring
req display(Counter& c):
    print c.value

let a'stack = Counter(0)
let b'actor = Counter(0)
let c'heap = Counter(0)
let d'shared = Counter(0)

display(a)   # &a
display(b)   # { let g = b.lock()?; display(&*g) }
display(c)   # &**c
display(d)   # &**d  — Arc<T> derefs to T, no lock needed
```

The caller never writes the lock — it is implicit and scoped to the call.

#### `Counter&` vs `Counter'actor` as parameter

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

#### Mutable coercion

```boring
def reset(mut Counter& c):
    c.value = 0

mut a'stack = Counter(0)
let b'actor = Counter(0)

reset(a)   # &mut a
reset(b)   # { let mut g = b.lock()?; reset(&mut *g) }
```

`'shared` (`Arc<T>` without interior mutability) cannot produce `&mut T` — passing a `Counter'shared` to a `mut Counter&` parameter is a compile error.

#### Lock scope and guard lifetime

When the argument is `'actor` or `'guard` and the parameter is `Counter&`, the transpiler generates a temporary binding:

```rust
// display(b) where b: Arc<Mutex<Counter>>
{
    let __g = b.lock()?;
    display(&*__g);
}   // lock released here
```

For `'guard`, a `Counter&` parameter uses `read()` (shared read lock); `mut Counter&` uses `write()` (exclusive write lock).

#### Struct and enum method parameters

Universal borrow inference is **disabled** for parameters of `req` and `def` methods defined on a struct or enum. Use the explicit `Counter& n` form to get universal borrowing in a method parameter.

#### Error conditions

```
error: cannot pass `x` (weak reference) to a non-weak parameter — weak references may be
       invalid. Call .upgrade() first and handle the Option.

error: cannot pass `x` ('shared) to `mut Counter&` — 'shared does not support mutable
       references. Use 'actor (Arc<Mutex<T>>) or 'guard (Arc<RwLock<T>>) instead.

error: cannot pass 'actor argument to `mut Counter&` in async function `f` — holding a
       MutexGuard across .await makes the future !Send. Acquire the lock inside the
       callee body instead.
```

---

## Inference

Boring's qualifier inference works from usage signals — in most programs you never write qualifiers. The zero-annotation goal: qualifier-free Boring code emits the same Rust as hand-annotated code.

### Constraint elimination

Each unqualified local variable starts with a candidate set of all possible qualifiers. Every usage signal narrows the set by eliminating incompatible qualifiers. When exactly one candidate remains it is chosen. When none remain the constraints are contradictory and a compile error is reported. When several remain a size-based fallback resolves the tie.

#### Candidate sets

| Declaration form | Initial candidate set | Fallback (multiple remaining) |
|---|---|---|
| `T` (bare, no qualifier) | `{Stack, Owned, Shared, Actor, Guard}` | priority-ordered fallback (see below) |
| `T'` (tick, indirection hint) | `{Owned, Shared, Actor, Guard}` | `'heap` (`Box<T>`) |
| `T?` (optional, bare) | `{Stack, Owned, Shared, Actor, Guard}` | same priority-ordered fallback |
| `T'?` (optional tick) | `{Owned, Shared, Actor, Guard}` | `Option<Box<T>>` |

`T'` and `T'?` restrict the initial set to indirection qualifiers. For optional forms, the inferred qualifier is applied to the **inner type** of the `Option` — `T?` with inferred `'actor` emits `Option<Arc<Mutex<T>>>`, not `Arc<Mutex<Option<T>>>`.

#### Signal table

| Signal | Compatible qualifiers |
|---|---|
| Call site demanding `T'shared` | `{Shared}` |
| Call site demanding `T'actor` | `{Actor}` |
| Call site demanding `T'guard` | `{Guard}` |
| Call site demanding `T'stack` | `{Stack}` |
| Call site demanding `T'heap` | `{Owned}` |
| `def` method call on the variable | `{Stack, Owned, Actor, Guard}` |
| `mut` binding (`mut x = …`) | `{Stack, Owned, Actor, Guard}` |
| `var` binding reassigned (`x = …`, `x.field = …`, `x.a.b.c = …`, `x[i] = …`) | `{Stack, Owned, Actor, Guard}` |
| Closure capturing `x` as method receiver | `{Actor, Guard}` |
| Closure capturing `x` read-only | `{Shared, Actor, Guard}` |
| `set` property setter body | `{Stack, Owned, Actor, Guard}` |
| Task capture as method receiver | `{Actor, Guard}` |
| Task capture, read-only | `{Shared, Actor, Guard}` |
| `req` method call | *(no constraint — all qualifiers remain)* |

Each signal intersects the current candidate set. The order of signals does not matter.

#### Priority-ordered fallback

When the candidate set still contains multiple qualifiers after all signals are applied:

**Step 1 — `'stack` candidate**

| Context | Decision |
|---|---|
| Struct field (any binding) | `'stack` — field bytes are part of the parent allocation |
| Local variable, sizeof(T) ≤ `--stack-auto-bytes` | `'stack` |
| Local variable, sizeof(T) > `--stack-auto-bytes` | skip `'stack`; continue to step 2 |

The threshold is configurable: `boring build --stack-auto-bytes 512` (default: 256 bytes).

**Step 2 — ordered chain**

If `'stack` was not selected, pick the first qualifier from the remaining set:

`'heap` > `'shared` > `'actor` > `'guard`

### Examples

```boring
let c = Counter(0)
share_read(c)           # expects Counter'shared → c infers 'shared
```

```boring
let c = Counter(0)
c.inc()                 # def call → {Stack, Owned, Actor, Guard}
                        # no further signal → size fallback → 'stack (if small)
```

```boring
let c = Counter(0)
c.inc()                 # def call → {Stack, Owned, Actor, Guard}
share_read(c)           # 'shared → intersect → {}  → ERROR
```

```boring
let a = Counter(0)
let b = a               # b is an alias of a — same qualifier group
spawn_actor(b)          # demands 'actor → both a and b infer 'actor
```

### Universal borrow as inference output

When a parameter has no explicit qualifier and no storage signals, the inference can resolve to a universal borrow (`Counter&` or `mut Counter&`) — evaluated before the priority-ordered fallback.

**Storage signals** prevent universal borrow inference: field assignment, return with ownership qualifier, closure/task capture, field destructuring (`let x = n.field`).

**Qualifier demand signals** also prevent it: passing to a function parameter with an explicit qualifier.

| Declaration | Signals | Inferred form | Rust emitted |
|---|---|---|---|
| `Counter n` | none | `Counter&` | `&Counter` |
| `mut Counter n` | none | `mut Counter&` | `&mut Counter` |
| `Counter n` | qualifier demand | qualifier via constraints | — |
| `Counter n` | storage | qualifier via constraints | — |

The `mut` keyword is **not inferred** — it must be written explicitly.

```boring
req display(Counter n):
    print n.value
# no storage, read-only → infers Counter& → fn display(n: &Counter)

def reset(mut Counter n):
    n.value = 0
# no storage, mutable → infers mut Counter& → fn reset(n: &mut Counter)
```

Lock acquisition at call sites works the same way as explicit `Counter&`:

```boring
let b'actor = Counter(0)
display(b)
# emits: { let __g = b.lock()?; display(&*__g); }
```

### Parameter auto-apply

A pre-inference pass runs `infer_qualifiers` on the function body before emitting parameters. `emit_param` then consults `inferred_qualifiers` and applies the resolved qualifier automatically:

```boring
def process(Counter c):   # no qualifier written
    spawn_actor(c)        # demands 'actor → inferred for c
# emits: fn process(c: Arc<Mutex<Counter>>)
```

### Cross-function propagation

After each function body is emitted, `fn_sigs` is updated with the inferred parameter qualifiers. Functions defined later in the file that call this function see the qualified signature and propagate the constraint.

**`'stack` is not propagated** — it would poison callers with a spurious constraint from file-ordering artifacts.

**Return-type–driven parameter inference:** when a constructor returns `T'actor`, the transpiler records `T` as an actor source type. Subsequent bare `T` parameters automatically infer `'actor`:

```boring
def Interpreter'actor new_interpreter():
    …

def Value eval_expr(Interpreter interp, Expr e):
    # interp infers 'actor because Interpreter is an actor source type
    …
```

### Struct field inference

The same constraint-elimination algorithm applies to struct fields. A pre-pass scans all method bodies of the struct for `self.field` access patterns:

```boring
struct Service:
    Counter stats        # no qualifier

    def record():
        spawn_actor(stats)   # demands 'actor → stats infers 'actor → Arc<Mutex<Counter>>
```

**Generics are not inferred.** `[Counter]` is `Vec<Counter>` and `[Counter'actor]` is `Vec<Arc<Mutex<Counter>>>` — these are distinct Rust types. The qualifier must be written explicitly in the element position.

**Cross-file inference is not planned** — see the open questions section.

### Qualifier unions and groups

A parameter can accept a restricted but not singleton set of qualifiers using a pipe-separated union:

```boring
def process(Counter'stack|heap c):   # accepts 'stack or 'heap, not 'shared or 'actor
    c.inc()
```

Named qualifier groups expand to the corresponding member sets:

| Group | Members |
|---|---|
| `'one` | `'stack`, `'heap` |
| `'many` | `'shared`, `'actor`, `'guard` |
| `'mut` | `'stack`, `'heap`, `'actor`, `'guard` |
| `'req` | `'shared` |

**Scope: parameters only.** Qualifier groups are not useful on local variables — the inference starting set already covers the same information.

### Explicit annotation — escape hatch

When inference cannot resolve a qualifier, an explicit annotation overrides everything:

```boring
let Counter'actor c = Counter(0)   # explicit — inference is skipped for c
```

---

## Single-thread vs multi-thread mode

The `--threading single` / `--threading multi` flag (default: multi) selects the wrapper implementation:

| Qualifier | `--threading multi` | `--threading single` |
|---|---|---|
| `'shared` | `Arc<T>` | `Rc<T>` |
| `'actor` | `Arc<std::sync::Mutex<T>>` | `Rc<RefCell<T>>` |
| `'guard` | `Arc<std::sync::RwLock<T>>` | `Rc<RefCell<T>>` |
| `'actor'task` | `Arc<tokio::sync::Mutex<T>>` | `Rc<RefCell<T>>` |
| `'guard'task` | `Arc<tokio::sync::RwLock<T>>` | `Rc<RefCell<T>>` |

In single-thread mode `'actor` and `'guard` both map to `Rc<RefCell<T>>` — there is no semantic difference between reader and writer locks in a single-threaded context. The same collapse applies to the `'task` variants: single-thread mode still runs under a tokio `current_thread` runtime (`#[tokio::main(flavor = "current_thread")]`, `tokio::task::spawn_local`), but since everything runs on one thread there is no need for `Send + Sync` locks, so `'actor'task` / `'guard'task` reuse plain `Rc<RefCell<T>>` instead of the tokio async locks.

---

## `'weak` references

A `'weak` qualifier produces a non-owning reference. The base qualifier is inferred from the right-hand side:

```boring
let a'shared  = Resource(label = "hello")
let b'weak    = a        # Weak<Resource> — inferred from a's 'shared qualifier

let r = b.upgrade()
print r.label            # "hello"
```

Explicit compound forms for type annotations and function signatures:

| Boring | Rust |
|---|---|
| `T'shared'weak` | `Weak<T>` (rc) or `sync::Weak<T>` (arc) |
| `T'actor'weak` | `sync::Weak<Mutex<T>>` |
| `T'guard'weak` | `sync::Weak<RwLock<T>>` |

Passing a `'weak` value to any non-weak parameter is a compile error — the transpiler requires an explicit `.upgrade()`.

---

## `string` and primitive types

Primitives (`int`, `uint`, `float`, `bool`) have no meaningful qualifier — they are always `Copy` in Rust. The bare names are the canonical form:

| Boring | Rust |
|---|---|
| `int` | `i64` |
| `uint` | `u64` |
| `float` | `f64` |
| `bool` | `bool` |
| `str` | `&str` |
| `string` | `Arc<str>` (multi-thread) / `Rc<str>` (single-thread) / `&'static str` (literals, strict mode) |

`string` uses `Arc<str>` to enable arbitrary value lifetimes and sharing. In single-thread mode it uses `Rc<str>`. Strict mode restricts `string` to compile-time literals only (`&'static str`); computed or interpolated values require explicit annotation.

---

## Open questions

### ⚠️ Cross-file struct field inference

The current struct field inference scans only the methods defined in the same file as the struct. Full cross-file inference would require a two-phase compilation model (parse all → infer globally → emit). Given that:

- Mutable fields already require an explicit `mut` or `var` keyword; adding a qualifier annotation is a small additional step,
- `'stack` / `'heap` field qualifiers are part of the module API and should not be silently changed by remote usage signals,
- The two-phase pipeline would complicate incremental builds,

the decision is to **keep cross-file inference out of scope** and document explicit annotation as the expected pattern for field qualifiers that depend on external callers.

---

## Implementation notes

### Tracking sets

The transpiler maintains four sets per scope for dispatch:

| Set | Contents |
|---|---|
| `var_mutex_types` | local vars with `'actor` |
| `var_mutex_task_types` | local vars with `'actor'task` / `'task` |
| `var_rwlock_types` | local vars with `'guard` |
| `var_rwlock_task_types` | local vars with `'guard'task` |

Parallel sets exist for struct fields (`struct_mutex_fields`, `struct_mutex_task_fields`, `struct_rwlock_fields`, `struct_rwlock_task_fields`).

These sets are populated during statement emission and propagated into sub-transpilers (method bodies) via `make_sub()`.

### Dispatch helpers

| Helper | Emits |
|---|---|
| `mutex_var_read(var, expr)` | `var.lock().unwrap().expr` or `.await` |
| `mutex_var_write(var, expr)` | same, write guard |
| `mutex_field_read(key, expr)` | field via read lock |
| `mutex_field_write(key, expr)` | field via write lock |
| `rwlock_field_read(key, expr)` | field via `read()` |
| `rwlock_field_write(key, expr)` | field via `write()` |
| `guard_read_access(v)` | `v.read().unwrap()` |
| `guard_write_guard(v)` | `v.write().unwrap()` |
| `guard_task_read_access(v)` | `v.read().await` |
| `guard_task_write_guard(v)` | `v.write().await` |

### Parser lookahead for compound qualifiers

`'actor'task` and `'guard'task` are two-token qualifiers. The `is_type_start_before_ident()` lookahead was extended to consume both tokens so that `let Counter'actor'task c = …` is parsed as a type annotation rather than an expression.

### Interior mutability in the interpreter

The interpreter's `Env` tracks `actor_bindings: HashSet<String>` — variables declared with an interior-mutable qualifier. Calls to `def` methods on these variables skip the "cannot call mutating method on immutable binding" check, matching the transpiler's semantics.

### Inference implementation status

| Case | Status |
|---|---|
| Local variable, call-site demand | ✅ implemented |
| Local variable, return-type demand | ✅ implemented |
| Local variable, task capture | ✅ implemented |
| Local variable, alias propagation | ✅ implemented |
| `T'` (tick) variables with inference | ✅ implemented |
| `mut` keyword as mutation signal | ✅ implemented |
| `var` reassignment as mutation signal (incl. nested fields) | ✅ implemented |
| `set` setter body as mutation signal (struct fields) | ✅ implemented |
| Closure captures (receiver / read-only) | ✅ implemented |
| Mutation + sharing conflict → conflict error | ✅ implemented |
| Qualifier union validation | ✅ implemented |
| Parameter qualifier auto-apply | ✅ implemented |
| Universal borrow inference (free functions only) | ✅ implemented |
| Universal borrow — field-destructuring suppression | ✅ implemented |
| Universal borrow — disabled for struct/enum method params | ✅ implemented |
| Universal borrow — task capture suppresses auto-ref | ✅ implemented |
| Cross-function propagation | ✅ implemented (single forward pass) |
| Struct field inference (all fields, single-file) | ✅ implemented |
| Optional (`T?`, `T'?`) inner-type inference | ✅ implemented |
| `'actor'task`/`'guard'task` vs `'actor`/`'guard` disambiguation (task-method-call signal) | ✅ implemented |
| Cross-file inference | not implemented |
| Fixed-point propagation (mutual recursion) | not implemented |

> **Note on `'actor'task` / `'guard'task`:** inference picks the `'task` variant when a `task`-declared method is called on the captured variable/field (see "Inferring `'actor'task` / `'guard'task` from task-method calls" above); otherwise it falls back to the plain sync variant. A task body that needs the async lock without calling a `task` method on the captured value itself still requires an explicit annotation.

### Rust research directions

#### Polonius — next-generation borrow checker

Polonius replaces NLL with a path-based analysis that eliminates false positives. Targeting stabilisation in 2026 H2. Some patterns today requiring `'actor` may become expressible with `'stack` or `'heap` under Polonius.

#### View types / field projections

Feature gate `view_types` on nightly (tracking [#155938](https://github.com/rust-lang/rust/issues/155938)) — functions declare which struct fields they borrow, giving the borrow checker field-level visibility. Aligns with Boring's per-field qualifier model.

#### "Beyond the &" umbrella goal (2026)

Groups `&pin` references, field projections, and reborrow traits (`Reborrow`, `CoerceShared`). `CoerceShared` could give `Arc<Mutex<T>>` first-class reborrow semantics, making `'actor` emission more transparent at the Rust level.
