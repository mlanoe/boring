# Qualifiers — Reference & Design Draft

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
| `'actor'task` / `'task` | `Arc<tokio::sync::Mutex<T>>` | *(multi only)* | interior mutability | async context |
| `'guard` | `Arc<std::sync::RwLock<T>>` | `Rc<RefCell<T>>` | interior mutability | reader-writer, sync |
| `'guard'task` | `Arc<tokio::sync::RwLock<T>>` | *(multi only)* | interior mutability | async context |
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

## Parameter passing

`'actor`, `'guard`, and `'shared` variables are passed by reference at call sites (auto-ref convention). The caller's lock is not acquired — the callee receives the Arc and controls its own lock granularity:

```boring
def int read_counter(Counter'actor c):
    c.get()       # c.lock().unwrap().get()

let Counter'actor c = Counter()
c.inc()
print read_counter(c)    # &c at call site — no refcount increment
```

The transpiler distinguishes three call-site cases automatically:

| Argument | Emitted Rust |
|---|---|
| Owned `Arc` local | `&v` |
| Forwarding an `&Arc` parameter | `v` (no double-borrow) |
| Plain non-actor value | `&Arc::new(Mutex::new(v))` |

`Arc::clone` is inserted only when the callee stores the value into an owned position (field assignment, `let` binding, etc.).

For `Counter&` (universal borrow) parameters, the transpiler acquires the lock at the call site and passes `&Counter`:

```boring
req display(Counter& c):
    print c.value

let Counter'actor a = Counter()
display(a)    # { let __g = a.lock().unwrap(); display(&*__g); }
```

See [qualifier-coercion.md](qualifier-coercion.md) for the full coercion rules.

---

## Single-thread vs multi-thread mode

The `--threading single` / `--threading multi` flag (default: multi) selects the wrapper implementation:

| Qualifier | `--threading multi` | `--threading single` |
|---|---|---|
| `'shared` | `Arc<T>` | `Rc<T>` |
| `'actor` | `Arc<std::sync::Mutex<T>>` | `Rc<RefCell<T>>` |
| `'guard` | `Arc<std::sync::RwLock<T>>` | `Rc<RefCell<T>>` |
| `'actor'task` | `Arc<tokio::sync::Mutex<T>>` | not supported |
| `'guard'task` | `Arc<tokio::sync::RwLock<T>>` | not supported |

In single-thread mode `'actor` and `'guard` both map to `Rc<RefCell<T>>` — there is no semantic difference between reader and writer locks in a single-threaded context.

Passing a single-thread variable to a function expecting the same qualifier type correctly uses `Rc::clone` rather than re-wrapping the value.

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

## Ownership qualifiers in managed mode ⚠️

> **Status: kept as-is.** Managed mode (`--mode managed`) is the prototyping / instrumentation environment. It does not fully support all qualifier variants. `'actor'task` / `'guard'task` emit in `--mode strict` only. No alignment planned.

---

## Open questions

### ⚠️ `'actor'task` in single-thread mode

`tokio::sync::Mutex` is `Send + Sync` but requires a tokio runtime. Single-thread mode avoids tokio entirely. The current behaviour is to reject `'actor'task` and `'guard'task` in single-thread builds. A warning at the syntax level (before codegen) would be more ergonomic than a rustc error.

### ⚠️ Qualifier on collection element types

`[Counter'actor]` is `Vec<Arc<Mutex<Counter>>>` — the qualifier is on the element, not the collection. The collection itself has no qualifier (it is always owned, stack or heap). This is clear but can be verbose:

```boring
var [User'actor] pool = []
```

A shorthand like `[User]'actor` to mean "Vec of Mutex-wrapped Users" has been discussed but not designed. The current explicit form is unambiguous.

### ⚠️ Qualifier inference for `'actor'task` vs `'actor`

The inference algorithm currently infers `'actor` for task-captured mutable variables. It does not distinguish whether the mutation happens inside a `task` function (which needs tokio locks) or a plain async closure. A signal based on the presence of `.await` in the task body could refine this. For now, explicit annotation is required for `'actor'task`.

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
