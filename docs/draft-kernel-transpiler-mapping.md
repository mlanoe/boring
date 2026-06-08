# Draft — Mapping Boring → Rust-for-Linux

> Status: preliminary analysis — not yet implemented

---

## Primitives

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `int` | `i64` | `i64` | ✅ | identical |
| `uint` | `u64` | `u64` | ✅ | identical |
| `float` | `f64` | — | ❌ | FPU forbidden in kernel (except explicit cases) |
| `bool` | `bool` | `bool` | ✅ | identical |
| `string` | `Arc<str>` | `kernel::str::CStr` / `CString` | ⚠️ | C-compatible kernel strings |
| `void` | `()` | `()` | ✅ | identical |

---

## Compound types

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `T?` | `Option<T>` | `Option<T>` | ✅ | available in `core::` |
| `[T]` | `Vec<T>` | `kernel::prelude::Vec<T>` | ✅ | kernel has its own Vec with kernel allocator |
| `{K: V}` | `HashMap<K,V>` | `kernel::rbtree::RBTree<K,V>` | ⚠️ | partial mapping — ordered, O(log n) vs O(1); keys must be `Ord` |
| `{T}` | `HashSet<T>` | `kernel::rbtree::RBTree<T,()>` | ⚠️ | set emulated via RBTree with `()` as value; keys must be `Ord` |
| `(T, U)` | tuples | `core::` tuples | ✅ | |
| `Box<T>` | `Box<T>` | `Box<T, KernelAllocator>` | ⚠️ | different allocator |

---

## Ownership qualifiers

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `T'` | `Box<T>` | `Box<T>` | ✅ | with kernel allocator |
| `T'auto` | `Rc<T>` | `kernel::sync::Arc` | ⚠️ | replaced by Arc<T> (Rc unavailable in kernel) |
| `T'task` | `Arc<T>` | `kernel::sync::Arc` | ✅ | |
| `T'actor` | `Arc<Mutex<T>>` | `kernel::sync::Mutex` | ✅ | |
| `T'guard` | `Arc<RwLock<T>>` | `kernel::sync::RwLock` | ✅ | |
| `T'weak` | `Weak<T>` | via `kernel::sync::Arc` | ✅ | |
| `T'stack` | `T` | `T` | ✅ | |
| `T&` / `var T&` | `&T` / `&mut T` | `&T` / `&mut T` | ✅ | |

---

## Functions and error handling

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `throws` | `Result<T, Box<dyn Error>>` | `Result<T, kernel::error::Error>` | ⚠️ | fixed error type (errno-based) |
| `throws MyError` | `Result<T, MyError>` | `Result<T, kernel::error::Error>` | ⚠️ | custom errors must be declared with `type … as kernel.error.Error(ERRNO)` |
| `try/catch` | pattern matching | pattern matching | ✅ | |
| `catch MyError` | typed catch branch | `Err(e) if e.code() == ERRNO =>` | ⚠️ | requires errno binding — see section below |
| `guard … else throw` | early return | early return | ✅ | |

### Kernel error type binding

In the kernel backend, all errors collapse to `kernel::error::Error` (an errno-based `i32`).
To preserve typed `catch` branches, each custom error type must be declared as a **distinct
nominal type** bound to a specific errno:

```boring
type NetworkError as kernel.error.Error(ENETDOWN)
type TimeoutError as kernel.error.Error(ETIMEDOUT)
type NoMemError   as kernel.error.Error(ENOMEM)
```

`type` (not `use`) is required here: `use` would create an alias — `NetworkError` and
`TimeoutError` would be the same type, indistinguishable at `catch`. `type` declares a
nominally distinct type, allowing the compiler to route each `catch` branch to the correct
errno guard:

```boring
// Boring source
try:
    page = fetch_page(url)
catch NetworkError e:
    log("network down")
catch TimeoutError e:
    log("timed out")
```

```rust
// Generated Rust-kernel
match fetch_page(url) {
    Ok(page) => { ... }
    Err(e) if e.code() == ENETDOWN  => { pr_err!("network down"); }
    Err(e) if e.code() == ETIMEDOUT => { pr_err!("timed out"); }
    Err(e) => { ... }  // catch-all
}
```

The validation pass must reject any `catch MyError` where `MyError` has no
`type … as kernel.error.Error(…)` declaration. A stdlib of pre-declared common
kernel errors (`NoMemError`, `InvalidArgError`, `NoDevError`…) will cover most cases
so developers only need to declare domain-specific errors.

---

## Async / concurrency

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `task def` | `async fn` + tokio | struct `XxxWork: Work` + `KernelFuture<T>` | ⚠️ | see section below |
| `task expr` | `tokio::spawn` | `system_wq.enqueue(work)` | ✅ | |
| `stream def` | `futures::Stream` | bounded chan + work item | ⚠️ | implemented on top of `chan` — see section below |
| `chan<T>(n)` / `tx.send` / `rx.recv` | MPSC tokio | ring buffer + `Mutex` + `CondVar` | ⚠️ | bounded capacity set at construction — see section below |
| `join` / `let a,b = (task f1(), task f2()).map(:value)` | `tokio::join!` | sequential `.wait()` on concurrent work items | ✅ | both items enqueued first, waited sequentially — wall time is max(t1,t2) |
| `wait Duration` | `tokio::sleep` | `kernel::delay::coarse_sleep` | ⚠️ | available but without await |
| `Future<T>` | `tokio::JoinHandle` | `KernelFuture<T>` | ⚠️ | blocking — `.wait()` in process context only |
| `future.done()` | `JoinHandle` + `Arc<Mutex<Option<T>>>` | `try_lock` + `is_some` | ✅ | non-blocking poll |

### Mapping `task def` → `Work` + `KernelFuture<T>`

A `task def` in Boring designates a method that may take time or block on a resource.
In the kernel, this pattern maps to a work item dispatched on `system_wq`.

**Generation principle:**

```rust
// Boring source
task def fetch_page(url: string) -> Page { ... }

// Generated Rust-kernel
struct FetchPageWork {
    url: CString,
    result: Arc<Mutex<Option<Result<Page, Error>>>>,
    done_cond: Arc<CondVar>,
    // Work field required by the trait
    work: Work<FetchPageWork>,
}

impl kernel::workqueue::Work for FetchPageWork {
    fn run(this: Arc<Self>) {
        let r = fetch_page_body(&this.url);
        *this.result.lock() = Some(r);
        this.done_cond.notify_all();
    }
}

// KernelFuture<T>: blocking handle returned to the caller
pub struct KernelFuture<T> {
    result: Arc<Mutex<Option<Result<T, Error>>>>,
    done_cond: Arc<CondVar>,
}

impl<T> KernelFuture<T> {
    pub fn wait(self) -> Result<T, Error> {
        let mut guard = self.result.lock();
        self.done_cond.wait_while(&mut guard, |r| r.is_none());
        guard.take().unwrap()
    }
}

// Call site: `task fetch_page("...")` emits:
let work = Arc::new(FetchPageWork::new(url));
let future = work.kernel_future();
system_wq.enqueue(work);
future  // : KernelFuture<Page>
```

**Usage constraint:** `.wait()` blocks the current thread — valid only in process context
(not from an IRQ handler or atomic section). The validation pass must reject any `.wait()`
call in a non-sleeping context.

---

### `KernelFuture<T>` — interface

`Future<T>` exposes two methods, symmetrically implemented across both backends:

```rust
impl<T> KernelFuture<T> {
    // Non-blocking — returns true if the result is available.
    // Boring: future.done()
    pub fn done(&self) -> bool {
        self.result.try_lock().map(|g| g.is_some()).unwrap_or(false)
    }

    // Blocking — waits for the result (process context only).
    // Boring: future.wait()
    pub fn wait(self) -> Result<T, Error> {
        let mut guard = self.result.lock();
        self.done_cond.wait_while(&mut guard, |r| r.is_none());
        guard.take().unwrap()
    }
}
```

On the tokio side, `done()` is implemented identically via `Arc<Mutex<Option<T>>>` + `try_lock` —
both backends are symmetric. The developer can freely compose from these two primitives
(polling, blocking wait, custom first-wins patterns…).

---

### Mapping `chan` → bounded ring buffer

No `kfifo` Rust binding exists in the kernel, and `VecDeque` is unavailable in `no_std`.
The channel is instead implemented as a **bounded ring buffer** backed by a `Vec` pre-allocated
at construction time, with two rotating indices under a `Mutex`:

```boring
// Boring source
let tx, rx = chan<Page>(32)   // ring buffer of capacity 32
tx.send(page)
let page = rx.recv()
```

```rust
// Generated Rust-kernel
struct KernelChan<T> {
    buf:       Vec<Option<T>>,  // pre-allocated to `capacity`
    capacity:  usize,
    read_idx:  usize,           // protected by mutex
    write_idx: usize,           // protected by mutex
    mutex:     Mutex<()>,
    not_full:  CondVar,         // wakes blocked send()
    not_empty: CondVar,         // wakes blocked recv()
}

// send(): blocks if full
fn send(&self, value: T) {
    let mut guard = self.mutex.lock();
    self.not_full.wait_while(&mut guard, |_| self.is_full());
    self.buf[self.write_idx] = Some(value);
    self.write_idx = (self.write_idx + 1) % self.capacity;
    self.not_empty.notify_one();
}

// recv(): blocks if empty
fn recv(&self) -> T {
    let mut guard = self.mutex.lock();
    self.not_empty.wait_while(&mut guard, |_| self.is_empty());
    let value = self.buf[self.read_idx].take().unwrap();
    self.read_idx = (self.read_idx + 1) % self.capacity;
    self.not_full.notify_one();
    value
}
```

**Constraints:**
- Capacity is **mandatory** in kernel context: `chan<T>` without a size is rejected at validation.
- Both `send()` and `recv()` block — valid in process context only.
- Bounded capacity is a feature in kernel context: no surprise dynamic allocation at runtime.

---

### Mapping `stream def` → chan + Work item

A `stream def` is implemented on top of `chan`: the stream body runs as a work item on
`system_wq` and sends each `yield`ed value into an internal channel. The caller receives
a `Receiver<T>` and consumes values with `.recv()`. End-of-stream is signalled by closing
the sender side.

```boring
// Boring source
stream def lines(file: File) -> string:
    for line in file.readLines():
        yield line
```

```rust
// Generated Rust-kernel
fn lines(file: File) -> KernelReceiver<CString> {
    let (tx, rx) = KernelChan::new(/* default capacity, or annotation-driven */);
    let work = Arc::new(LinesWork { file, tx });
    system_wq.enqueue(work);
    rx
}

impl Work for LinesWork {
    fn run(this: Arc<Self>) {
        for line in this.file.read_lines() {
            this.tx.send(line);  // blocks if consumer is slow
        }
        // tx dropped here → signals end-of-stream to rx
    }
}
```

**Constraint:** the internal channel capacity for a `stream def` must be specifiable via
an annotation (e.g. `stream(32) def lines(…)`) — otherwise a default capacity is used.
The validation pass must ensure the capacity is set explicitly or a sensible default is documented.

---

### Open question — capturing `self` in a `Work` item

> **Note:** the current tokio implementation implicitly assumes `Arc<Self>` for any
> `task def` method (it always emits `&self`, see `emit_top.rs:406`) but **does not validate**
> the actual qualifier of `self`. The constraint below would therefore be stricter than the
> current behaviour — and reveals a validation gap to fix on the tokio side as well.

When `task def` is an **instance method**, the work item must capture `self`.
Several options depending on the ownership qualifier of `self`:

| Boring qualifier | `self` type | Capture in Work | Potential issue |
|-----------------|-------------|-----------------|-----------------|
| `T'task` | `Arc<T>` | `Arc::clone(&self)` in the work item | none — natural |
| `T'actor` | `Arc<Mutex<T>>` | clone the `Arc` | who holds the lock? if `run()` acquires it and the caller already holds it → deadlock |
| `T'guard` | `Arc<RwLock<T>>` | clone the `Arc` | same issue with the write lock |
| `T'` (owned) | `Box<T>` | move into the work item | `self` inaccessible after the `task` — must be validated statically |
| `T&` / `T&mut` | reference | **impossible** — the work item will outlive the reference | must be rejected at validation |

**Decision: `task def` on `self` is restricted to `'task`, `'actor`, and `'guard`** — the only
qualifiers with a shareable lifetime compatible with a work item. `T&`, `T&mut`, and `T'` are
rejected at validation. This constraint is already enforced in the tokio backend.

---

## Stdlib

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `print!` | `println!` | `pr_info!` / `pr_err!` | ⚠️ | different kernel macros |
| `assert_eq!` | `assert_eq!` | `kernel::build_assert!` | ⚠️ | panics forbidden → `WARN_ON` |
| `panic(msg)` | `panic!` | — | ❌ | a panic = kernel oops/crash |
| Math (`sqrt`, `sin`…) | `std::f64` | — | ❌ | FPU forbidden |
| `Vec` methods | `std::vec` | `kernel::prelude::Vec` | ⚠️ | slightly different API |
| `HashMap` | `std::collections` | `kernel::rbtree::RBTree<K,V>` | ⚠️ | ordered, O(log n), keys must be `Ord` |

---

## Summary

### What maps well (~60% of the language)

- All primitives except `float`
- Ownership qualifiers (`Box`, `Arc`, `Mutex`, `RwLock`)
- Structs, enums, traits, generics
- Pattern matching, control flow
- Error handling (with error type adaptation)
- `Vec`, tuples, `Option`

### What requires an alternative mapping

- `string` → kernel `CStr` / `CString`
- `HashMap` / `HashSet` → `RBTree<K,V>` / `RBTree<T,()>` — ordered, O(log n), keys must be `Ord`
- `throws MyError` → `kernel::error::Error` (errno-based)
- `print!` / assertions → kernel macros
- `Box<T>` → kernel allocator

### What is incompatible / must be disabled

- `task def` on `self` with `T&` / `T&mut` — references incompatible with work item lifetime (must be rejected)
- `task def` on `self` with `Box<T>` — move semantics must be validated statically
- `chan<T>` without explicit capacity — forbidden in kernel context (bounded size mandatory)
- Poll-based `Future<T>` — replaced by blocking `KernelFuture<T>`
- `float` and all floating-point math
- `panic`
- `T'auto` (replaced by `Arc<T>` — validation emits a warning, not an error)

---

## Planned transpiler architecture

A second emission backend within the same binary — no duplication of parser, AST, or typing passes.

- **Shared (untouched)**: parser, AST, typing passes, existing `emit_*.rs` files
- **New: `src/validator/kernel.rs`** — validation pass that runs before emission when `--target kernel`;
  rejects incompatible constructs with explicit error messages
- **New: `src/transpiler/kernel/`** — kernel emission backend:
  - `emit_kernel_top.rs` — struct/impl/fn declarations, `Work` item generation
  - `emit_kernel_stmt.rs` — statements, `task` expr, `chan`, `yield`
  - `emit_kernel_expr.rs` — expressions, string literal → `c_str!`, method calls
  - `emit_kernel_helpers.rs` — `KernelFuture<T>`, `KernelChan<T>` runtime types
- The `--target kernel` flag selects the kernel validator + emitter at compile time
- Zero regression risk on the standard backend — existing files are not modified

---

## Next steps (to be decided)

1. Validation pass — reject float, async, panic, Rc, HashMap with explicit error messages
2. New emitter — kernel type and macro substitution
3. Kernel stdlib — `no_std`-compatible subset of the Boring stdlib
