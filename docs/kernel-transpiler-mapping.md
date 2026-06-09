# Boring → Rust-for-Linux mapping

This document describes how Boring language constructs map to Rust-for-Linux kernel abstractions,
and the architecture of the kernel emission backend (`--target kernel`).

---

## Primitives

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `int` | `i64` | `i64` | ✅ | identical |
| `uint` | `u64` | `u64` | ✅ | identical |
| `float` | `f64` | — | ❌ | FPU disabled in kernel — forbidden, use integer arithmetic |
| `bool` | `bool` | `bool` | ✅ | identical |
| `string` | `Arc<str>` | `kernel::str::CStr` / `CString` | ⚠️ | C-compatible kernel strings; literals emitted as `c_str!("…")` |
| `void` | `()` | `()` | ✅ | identical |

---

## Compound types

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `T?` | `Option<T>` | `Option<T>` | ✅ | available in `core::` |
| `[T]` | `Vec<T>` | `kernel::prelude::Vec<T>` | ✅ | kernel allocator used |
| `{K: V}` | `HashMap<K,V>` | `kernel::rbtree::RBTree<K,V>` | ⚠️ | ordered, O(log n) vs O(1); keys must implement `Ord` |
| `{T}` | `HashSet<T>` | `kernel::rbtree::RBTree<T,()>` | ⚠️ | set emulated via RBTree with `()` value; keys must implement `Ord` |
| `(T, U)` | tuples | `core::` tuples | ✅ | |
| `Box<T>` | `Box<T>` | `Box<T, KernelAllocator>` | ⚠️ | different allocator, same semantics |

---

## Ownership qualifiers

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `T'` | `Box<T>` | `Box<T>` | ✅ | kernel allocator |
| `T'auto` | `Rc<T>` | `kernel::sync::Arc` | ⚠️ | `Rc` unavailable in kernel — replaced by `Arc`, validation emits a warning |
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
| `throws` | `Result<T, Box<dyn Error>>` | `Result<T, kernel::error::Error>` | ⚠️ | fixed errno-based error type |
| `throws MyError` | `Result<T, MyError>` | `Result<T, kernel::error::Error>` | ⚠️ | requires `type MyError as kernel.error.Error(ERRNO)` |
| `try / catch` | pattern matching | pattern matching | ✅ | |
| `catch MyError` | typed catch branch | `Err(e) if e.code() == ERRNO =>` | ⚠️ | requires errno binding — see section below |
| `guard … else throw` | early return | early return | ✅ | |

### Kernel error type binding

In the kernel backend all errors collapse to `kernel::error::Error` (an errno `i32`).
To preserve typed `catch` branches, each custom error type must be declared as a distinct
nominal type bound to a specific errno:

```boring
type NetworkError as kernel.error.Error(ENETDOWN)
type TimeoutError as kernel.error.Error(ETIMEDOUT)
type NoMemError   as kernel.error.Error(ENOMEM)
```

`type` is required — not `use`. `use` creates an alias: `NetworkError` and `TimeoutError`
would be the same type, indistinguishable at `catch`. `type` declares a nominally distinct
type, letting the compiler route each `catch` branch to the correct errno guard:

```boring
try:
    page = fetch_page(url)
catch NetworkError e:
    log("network down")
catch TimeoutError e:
    log("timed out")
```

```rust
// Generated
match fetch_page(url) {
    Ok(page)  => { … }
    Err(e) if e.code() == ENETDOWN  => { pr_info!("network down"); }
    Err(e) if e.code() == ETIMEDOUT => { pr_info!("timed out"); }
    Err(e)    => { … }
}
```

The validation pass rejects any `catch MyError` where `MyError` has no
`type … as kernel.error.Error(…)` declaration. A stdlib of pre-declared common
kernel errors (`NoMemError`, `InvalidArgError`, `NoDevError`…) covers most cases.

---

## Async / concurrency

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `task def` | `async fn` + tokio | struct `XxxWork: Work` + `KernelFuture<T>` | ⚠️ | see section below |
| `task expr` | `tokio::spawn` | `system_wq.enqueue(work)` | ✅ | |
| `join` / `let a, b = (task f1(), task f2()).map(:value)` | `tokio::join!` | sequential `.wait()` on concurrent work items | ✅ | both items enqueued first — wall time is max(t1, t2) |
| `stream<N> def` / `stream def` | `futures::Stream` | bounded channel + work item | ⚠️ | N defaults to 2 — see section below |
| `channel<T, N>` / `channel<T>` / `tx.send` / `rx.recv` | tokio MPSC | ring buffer + `Mutex` + `CondVar` | ⚠️ | N defaults to 2 — see section below |
| `wait Duration` | `tokio::sleep` | `kernel::delay::coarse_sleep` | ⚠️ | blocking, no await |
| `Future<T>` | `tokio::JoinHandle` | `KernelFuture<T>` | ⚠️ | blocking — `.wait()` in process context only |
| `future.done()` | `Arc<Mutex<Option<T>>>` + `try_lock` | `try_lock` + `is_some` | ✅ | non-blocking poll |

### `task def` → `Work` + `KernelFuture<T>`

A `task def` designates a method that may take time or block on a resource.
In the kernel this maps to a work item dispatched on `system_wq`.

**Constraint on `self`:** `task def` as an instance method is restricted to `'task`,
`'actor`, and `'guard` — the only qualifiers with a lifetime compatible with a work item.
`T&`, `T&mut`, and `T'` are rejected at validation. This mirrors the constraint already
enforced in the tokio backend.

```boring
task def fetch_page(url: string) -> Page: …
```

```rust
// Generated
struct FetchPageWork {
    url:      CString,
    result:   Arc<Mutex<Option<Result<Page, Error>>>>,
    done_cond: Arc<CondVar>,
    work:     Work<FetchPageWork>,
}

impl kernel::workqueue::Work for FetchPageWork {
    fn run(this: Arc<Self>) {
        let r = fetch_page_body(&this.url);
        *this.result.lock() = Some(r);
        this.done_cond.notify_all();
    }
}

// Call site: `task fetch_page("…")` emits:
let work   = Arc::new(FetchPageWork::new(url));
let future = work.kernel_future();
system_wq.enqueue(work);
future   // : KernelFuture<Page>
```

### `KernelFuture<T>` — interface

`Future<T>` exposes two methods, symmetrically implemented across both backends:

```rust
impl<T> KernelFuture<T> {
    // Non-blocking poll — Boring: future.done()
    pub fn done(&self) -> bool {
        self.result.try_lock().map(|g| g.is_some()).unwrap_or(false)
    }

    // Blocking wait — Boring: future.wait()  (process context only)
    pub fn wait(self) -> Result<T, Error> {
        let mut guard = self.result.lock();
        self.done_cond.wait_while(&mut guard, |r| r.is_none());
        guard.take().unwrap()
    }
}
```

On the tokio side, `done()` is implemented identically via `Arc<Mutex<Option<T>>>` + `try_lock`.
Both backends are symmetric, allowing developers to compose freely from these two primitives.

### `channel<T, N>` → bounded ring buffer

The capacity `N` is a compile-time constant. It is optional: when omitted, a default of **2**
is used — enough to absorb small scheduling variations between producer and consumer without
implying meaningful buffering. Explicit capacity is recommended for production code.
This applies to both backends: the tokio backend passes `N` to `tokio::sync::mpsc::channel(N)`.

The kernel implementation uses a **ring buffer** backed by a `Vec` pre-allocated to `N`,
with two rotating indices (`read_idx`, `write_idx`) protected by a `Mutex`:

- `send()` blocks if the buffer is full — wakes on `not_full` CondVar
- `recv()` blocks if the buffer is empty — wakes on `not_empty` CondVar

```boring
let tx, rx = channel<string>      // default capacity: 2
let tx, rx = channel<string, 32>  // explicit capacity
tx.send("hello")
let msg = rx.recv()
```

```rust
// Generated (kernel backend)
let chan    = Arc::new(KernelChan::<CString>::new(32));
let (tx, rx) = chan.endpoints();

tx.send(c_str!("hello"));
let msg = rx.recv();
```

### `stream<N> def` → channel + Work item

A `stream<N> def` runs its body as a work item on `system_wq`, sending each `yield`ed
value into an internal `channel<T, N>`. The caller receives a `Receiver<T>` and consumes
values with `.recv()`. Closing the sender side signals end-of-stream.

```boring
stream def lines(file: File) -> string:  // default capacity: 2
    for line in file.readLines():
        yield line

stream<32> def lines(file: File) -> string:  // explicit capacity
    for line in file.readLines():
        yield line
```

```rust
// Generated
fn lines(file: File) -> KernelReceiver<CString> {
    let chan = Arc::new(KernelChan::<CString>::new(32));
    let (tx, rx) = chan.endpoints();
    let work = Arc::new(LinesWork { file, tx });
    system_wq.enqueue(work);
    rx
}

impl Work for LinesWork {
    fn run(this: Arc<Self>) {
        for line in this.file.read_lines() {
            this.tx.send(line);  // blocks if consumer is slow
        }
        // tx dropped → signals end-of-stream
    }
}
```

---

## Stdlib

| Boring | Rust std | Rust-kernel | Status | Notes |
|--------|----------|-------------|--------|-------|
| `print!` | `println!` | `pr_info!` / `pr_err!` | ⚠️ | different kernel macros |
| `assert_eq!` | `assert_eq!` | `kernel::build_assert!` | ⚠️ | panics forbidden → `WARN_ON` |
| `panic(msg)` | `panic!` | — | ❌ | kernel oops/crash — forbidden, use `throws` / `Result` |
| Math (`sqrt`, `sin`…) | `std::f64` | — | ❌ | FPU disabled — forbidden |
| `Vec` methods | `std::vec` | `kernel::prelude::Vec` | ⚠️ | slightly different API |
| `HashMap` | `std::collections` | `kernel::rbtree::RBTree<K,V>` | ⚠️ | ordered, O(log n), keys must implement `Ord` |

---

## Summary

### Maps directly (~60% of the language)

- All primitives except `float`
- Ownership qualifiers (`Box`, `Arc`, `Mutex`, `RwLock`)
- Structs, enums, traits, generics
- Pattern matching, control flow
- Error handling (with error type adaptation)
- `Vec`, tuples, `Option`
- `task def`, `Future<T>`, `join`, `channel<T,N>`, `stream<N> def`

### Requires adaptation

- `string` → kernel `CStr` / `CString`; literals → `c_str!("…")`
- `{K: V}` / `{T}` → `RBTree<K,V>` / `RBTree<T,()>` — ordered, O(log n), keys must be `Ord`
- `throws MyError` → requires `type MyError as kernel.error.Error(ERRNO)` declaration
- `print!` / assertions → kernel macros
- `Box<T>` → kernel allocator
- `HashMap` / `HashSet` → `RBTree` — ordered, O(log n), keys must implement `Ord`
- `T'auto` → replaced by `Arc<T>`, validation emits a warning

### Forbidden

- `float` and floating-point math — FPU disabled
- `panic` — kernel oops/crash; use `throws` / `Result` instead
- `task def` on `self` with `T&`, `T&mut`, or `T'` — lifetime incompatible with a work item
- `channel<T>` without explicit capacity — defaults to 2; explicit capacity recommended for production code

---

## Transpiler architecture

A second emission backend within the same binary — parser, AST, and typing passes are shared
and untouched.

```
src/
  validator/
    kernel.rs          # validation pass: rejects forbidden constructs with explicit errors
  transpiler/
    mod.rs             # existing — unchanged
    emit_top.rs        # existing — unchanged
    emit_stmt.rs       # existing — unchanged
    emit_expr.rs       # existing — unchanged
    helpers.rs         # existing — unchanged
    kernel/
      mod.rs           # entry point, selected by --target kernel
      emit_top.rs      # struct/impl/fn declarations, Work item generation
      emit_stmt.rs     # statements, task expr, channel, yield
      emit_expr.rs     # expressions, string literal → c_str!, method calls
      helpers.rs       # KernelFuture<T>, KernelChan<T> runtime types
```

The `--target kernel` flag activates the kernel validator then the kernel emitter.
The standard backend (`--target rust`, default) is not modified.

## Implementation order

1. **Validation pass** — reject `float`, `panic`, `T&`/`T&mut` on `task def`; warn on `channel`/`stream` without explicit capacity, `T'auto`
2. **Primitives and types** — `emit_kernel_top.rs`: structs, enums, ownership qualifiers, error type binding
3. **`KernelFuture<T>`** — `task def` → `Work` item generation, `.done()` / `.wait()`
4. **`KernelChan<T, N>`** — ring buffer, `send()` / `recv()`
5. **`stream<N> def`** — built on top of `KernelChan`
6. **Kernel stdlib** — `no_std`-compatible subset: `pr_info!`, `WARN_ON`, `RBTree` wrappers, pre-declared error types
