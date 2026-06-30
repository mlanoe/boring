# Rust-for-Linux target

`boring build --target kernel` compiles Boring source to **Rust-for-Linux** — the `no_std` Rust dialect used to write Linux kernel modules, drivers, and subsystems.

The same language applies: structs, enums, traits, ownership qualifiers, error handling, tasks, channels, streams. What changes is the mapping underneath. This document describes those differences relative to the standard (tokio) backend.

---

## Why a kernel target?

The Linux kernel imposes constraints that make the standard Rust backend unusable:

- **No `std`** — only `core` and the `kernel` crate are available.
- **No FPU** — floating-point is disabled on real hardware; the kernel target does not yet fully enforce this (see note below).
- **No tokio** — no async executor, no `JoinHandle`, no `select!`.
- **No `panic`** — a panic is a kernel oops/crash.
- **Fixed error type** — errors are errno-based (`kernel::error::Error`), not `Box<dyn Error>`.
- **Bounded allocations** — all channels and streams must have a fixed compile-time capacity.

## Activating the kernel target

```sh
boring build --target kernel main.br
# → generates main_kernel/ with src/lib.rs + Cargo.toml
# Build with: make -C /path/to/linux M=$PWD
```

The kernel backend runs a **validation pass** before emission. It rejects incompatible constructs with explicit error messages and warns on constructs that need attention.

---

## What changes vs the standard backend

### Primitives

| Boring | Rust std | Rust-kernel |
|--------|----------|-------------|
| `int` | `i64` | `i64` — identical |
| `uint` | `u64` | `u64` — identical |
| `float` | `f64` | `f64` — **not actually rejected** (see note) |
| `bool` | `bool` | `bool` — identical |
| `string` | `Arc<str>` | `kernel::str::CString` / `&kernel::str::CStr` |
| `void` | `()` | `()` — identical |

String literals are emitted as `c_str!("…")`.

> **Known gap:** the validator only rejects the capitalized `Float` type alias and float
> *literals* (e.g. `1.5`) plus calls to a fixed list of float math builtins. A `float`-typed
> parameter, return value, or `let` binding with no float literal in it currently passes
> validation uncaught and is emitted as plain `f64` (with a `/* float forbidden */` comment
> marker in the generated code) instead of being rejected. Treat `float` as unsupported in
> kernel context regardless of whether the validator currently catches every case — this is
> tracked as a validator bug, not a language feature.

### Ownership qualifiers

| Boring | Rust std | Rust-kernel |
|--------|----------|-------------|
| `T'` | `Box<T>` | `Box<T, KVmalloc>` — kernel allocator |
| `T'shared` | `Arc<T>` / `Rc<T>` | `Arc<T>` (via `kernel::prelude::Arc`) — `Rc` unavailable in kernel |
| `T'actor` | `Arc<tokio::sync::Mutex<T>>` | `Arc<kernel::sync::Mutex<T>>` |
| `T'guard` | `Arc<tokio::sync::RwLock<T>>` | `Arc<kernel::sync::RwLock<T>>` |

### Compound types

| Boring | Rust std | Rust-kernel |
|--------|----------|-------------|
| `[T]` | `Vec<T>` | `kernel::prelude::Vec<T>` |
| `{K=V}` | `HashMap<K,V>` | `kernel::rbtree::RBTree<K,V>` — ordered, O(log n), keys must be `Ord` |
| `{T}` | `HashSet<T>` | `kernel::rbtree::RBTree<T,()>` — same constraints |

### Error handling

All errors collapse to `kernel::error::Error` (errno-based `i32`).

> **Planned, not yet implemented.** The design intent is to preserve typed `catch` branches
> by letting the user declare each custom error type as a distinct nominal type bound to an
> errno:
>
> ```boring
> type NetworkError as kernel.error.Error(ENETDOWN)
> type TimeoutError as kernel.error.Error(ETIMEDOUT)
> ```
>
> `type` would be required — not `use`. `use` creates an alias; `type` would declare a
> distinct type so the compiler could route `catch` branches correctly:
>
> ```boring
> try:
>     page = fetch_page(url)
> catch NetworkError e:
>     log("network down")
> catch TimeoutError e:
>     log("timed out")
> ```
>
> ```rust
> // Intended generated code (not yet produced)
> match fetch_page(url) {
>     Ok(page)  => { … }
>     Err(e) if e.code() == ENETDOWN  => { pr_info!("network down"); }
>     Err(e) if e.code() == ETIMEDOUT => { pr_info!("timed out"); }
>     Err(e)    => { … }
> }
> ```
>
> **Current behavior:** the `kernel.error.Error(ENETDOWN)` errno-binding call syntax is not
> recognized by the parser (`type … as …` parses the right-hand side as a plain type, with no
> notion of an errno-bound constructor). The kernel transpiler's `try`/`catch` handling is
> currently a stub that emits only the `try` body and **silently drops all `catch` clauses** —
> it does not yet generate the `match … Err(e) if e.code() == …` dispatch shown above. Do not
> rely on typed `catch` branches in kernel-target code today.

### Print

| Boring | Rust std | Rust-kernel |
|--------|----------|-------------|
| `print "…"` | `println!("…")` | `kernel::pr_info!("…\n")` |

---

## `task def` — work items

In the standard backend, `task def` compiles to `async fn` + `tokio::spawn`.
In the kernel backend, it generates a **work item** dispatched on `system_wq`:

```boring
task def Page fetch_page(string url):
    # … fetch logic
```

```rust
// Generated (kernel)
struct FetchPageWork {
    url: kernel::str::CString,
    result: Arc<kernel::sync::Mutex<Option<Result<Page, kernel::error::Error>>>>,
    done_cond: Arc<kernel::sync::CondVar>,
    work: kernel::workqueue::Work<FetchPageWork>,
}

impl kernel::workqueue::Work<FetchPageWork> for FetchPageWork {
    fn run(this: Arc<Self>) {
        let r = fetch_page_body(this.url);
        *this.result.lock() = Some(r);
        this.done_cond.notify_all();
    }
}

fn fetch_page(url: kernel::str::CString) -> KernelFuture<Page> { … }
fn fetch_page_body(url: kernel::str::CString)
    -> Result<Page, kernel::error::Error> { … }
```

**Constraint on `self`:** `task def` as an instance method is only allowed on `'shared`, `'actor`, and `'guard` receivers — the only qualifiers with a lifetime compatible with a work item. `T&`, `T&mut`, `T'`, and (currently) `'task` itself are rejected — the validator does not special-case the `'task` qualifier, so it falls through to the same error as an unqualified receiver.

### `KernelFuture<T>`

Every `task def` returns a `KernelFuture<T>` instead of a `JoinHandle`:

| Boring | Rust std | Rust-kernel |
|--------|----------|-------------|
| `f.wait()` | `f.await` | `f.wait()` — blocks the current thread (process context only) |
| `f.done()` | — | `f.done()` — non-blocking poll, returns `bool` |

`wait()` blocks — it must not be called from an IRQ handler or atomic context.

### `join`

```boring
let a, b = join (task f1(), task f2())
```

Both work items are enqueued concurrently; `.wait()` is called on each in turn. Wall time is `max(t1, t2)`.

---

## `channel` — two ring-buffer variants

The kernel backend provides two channel implementations depending on how the capacity is expressed. Both use blocking `send()` / `recv()` — valid in process context only.

### `channel<T, N>` — const-generic, stack buffer

The capacity is a compile-time type parameter. The buffer is a fixed-size array `[Option<T>; N]` allocated inline (no heap). Emits `KernelSender<T, N>` / `KernelReceiver<T, N>`.

```boring
let tx, rx = channel<string, 32>   # const-generic, stack buffer
tx.send("hello")                   # blocks if full
let msg = rx.recv()                # blocks if empty
```

```rust
// Generated (kernel)
let (tx, rx) = kernel_channel::<CString, 32>();
tx.send(c_str!("hello"));
let msg = rx.recv();
```

### `channel<T>(cap)` — runtime capacity, heap buffer

The capacity is a call argument. The buffer is a `Vec<Option<T>>` pre-allocated on the kernel heap. Emits `DynKernelSender<T>` / `DynKernelReceiver<T>`.

```boring
let tx, rx = channel<string>(32)   # runtime cap, heap buffer
tx.send("hello")
let msg = rx.recv()
```

```rust
// Generated (kernel)
let (tx, rx) = dyn_kernel_channel::<CString>(32);
tx.send(c_str!("hello"));
let msg = rx.recv();
```

Omitting the capacity entirely uses the **runtime/heap variant** (`dyn_kernel_channel`, `DynKernelSender`/`DynKernelReceiver`) with a fallback capacity of `2` passed as a runtime argument — not the const-generic stack variant. A warning is emitted recommending an explicit value via `channel<T, N>` or `channel<T>(cap)`.

---

## `stream` — sequential iterator or Work item

The kernel backend applies the same two-strategy rule as the tokio backend.

### Sequential stream

If the body has no `wait` and no `task` calls, the stream is emitted as a plain iterator — no workqueue involved. `yield` → `__items.push(...)`, returns `__items.into_iter()`.

```boring
stream int range(int n):
    for i in 0..n:
        yield i
```

```rust
// Generated (kernel)
fn range(n: i64) -> impl Iterator<Item = i64> {
    let mut __items: kernel::prelude::Vec<i64> = kernel::prelude::Vec::new();
    for i in 0i64..n { __items.push(i); }
    __items.into_iter()
}
```

### Async stream — Work item + `KernelReceiver<T, N>`

If the body contains `wait` or `task` calls, the stream becomes a workqueue work item. The function returns a `KernelReceiver<T, N>`; the caller consumes values with `.recv()`. Use `stream<N>` to set the capacity explicitly (defaults to 2).

```boring
stream<16> string lines(File file):
    for line in file.readLines():
        yield line
```

```rust
// Generated (kernel) — three pieces:
struct LinesWork {
    file: File,
    tx:   KernelSender<CString, 16>,
    work: kernel::workqueue::Work<LinesWork>,
}
impl kernel::workqueue::Work<LinesWork> for LinesWork {
    fn run(this: Arc<Self>) {
        for line in this.file.read_lines() {
            this.tx.send(line);   // blocks if consumer is slow
        }
        // tx dropped → signals end-of-stream
    }
}
fn lines(file: File) -> KernelReceiver<CString, 16> { … }
```

---

## Forbidden constructs and warnings

**Errors** — rejected before emission:

| Construct | Reason |
|-----------|--------|
| `float`, floating-point math | FPU disabled in kernel context |
| `panic(…)` | kernel oops/crash — use `throws` / `Result` |
| `task def` on `self` with `T&`, `T&mut`, `T'`, or `'task` | lifetime incompatible with a work item (only `'shared`, `'actor`, `'guard` are accepted) |

**Warnings** — emitted with a default, but explicit specification is recommended:

| Construct | Behaviour |
|-----------|-----------|
| `channel<T>` without capacity | defaults to the heap-buffered `dyn_kernel_channel` variant with cap=2 — specify `channel<T, N>` or `channel<T>(cap)` explicitly |
| `stream` without `<N>` (async body) | defaults to N=2 — specify `stream<N>` explicitly |
| `T'shared` | `Rc<T>` replaced by `Arc<T>` (`kernel::prelude::Arc`) — `Rc` is unavailable in `no_std` |
