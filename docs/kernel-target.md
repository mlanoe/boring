# Rust-for-Linux target

`boring build --target kernel` compiles Boring source to **Rust-for-Linux** — the `no_std` Rust dialect used to write Linux kernel modules, drivers, and subsystems.

The same language applies: structs, enums, traits, ownership qualifiers, error handling, tasks, channels, streams. What changes is the mapping underneath.

```sh
boring build --target kernel main.br
# → generates main_kernel/ with src/lib.rs + Cargo.toml
# Build with: make -C /path/to/linux M=$PWD
```

The kernel backend runs a **validation pass** before emission. It rejects incompatible constructs with explicit error messages and warns on constructs that need attention.

---

## Constraints

The Linux kernel imposes constraints that make the standard Rust backend unusable:

- **No `std`** — only `core` and the `kernel` crate are available.
- **No FPU** — floating-point is disabled on real hardware.
- **No tokio** — no async executor, no `JoinHandle`, no `select!`.
- **No `panic`** — a panic is a kernel oops/crash.
- **Fixed error type** — errors are errno-based (`kernel::error::Error`), not `Box<dyn Error>`.
- **Bounded allocations** — all channels and streams must have a fixed compile-time capacity.

---

## Type mapping

### Primitives

| Boring | Rust std | Rust-kernel | Notes |
|--------|----------|-------------|-------|
| `int` | `i64` | `i64` | identical |
| `uint` | `u64` | `u64` | identical |
| `float` | `f64` | — | FPU disabled — forbidden |
| `bool` | `bool` | `bool` | identical |
| `string` | `Arc<str>` / `Rc<str>` | `kernel::str::CStr` / `CString` | literals emitted as `c_str!("…")` |
| `void` | `()` | `()` | identical |

### Compound types

| Boring | Rust std | Rust-kernel | Notes |
|--------|----------|-------------|-------|
| `T?` | `Option<T>` | `Option<T>` | available in `core::` |
| `[T]` | `Vec<T>` | `kernel::prelude::Vec<T>` | kernel allocator |
| `{K=V}` | `HashMap<K,V>` | `kernel::rbtree::RBTree<K,V>` | ordered, O(log n); keys must implement `Ord` |
| `{T}` | `HashSet<T>` | `kernel::rbtree::RBTree<T,()>` | set via RBTree with `()` value; keys must implement `Ord` |
| `(T, U)` | tuples | `core::` tuples | |

### Ownership qualifiers

| Boring | Rust std | Rust-kernel | Notes |
|--------|----------|-------------|-------|
| `T'new` | `Box<T>` | `Box<T, KVmalloc>` | kernel allocator |
| `T'shared` | `Arc<T>` / `Rc<T>` | `Arc<T>` (`kernel::prelude::Arc`) | `Rc` unavailable in `no_std` |
| `T'actor` | `Arc<Mutex<T>>` / `Rc<RefCell<T>>` | `Arc<kernel::sync::Mutex<T>>` | |
| `T'guard` | `Arc<RwLock<T>>` | `Arc<kernel::sync::RwLock<T>>` | |
| `T'weak` | `Weak<T>` | `Weak<T>` | |
| `T'inline` | `T` | `T` | |
| `T&` / `var T&` | `&T` / `&mut T` | `&T` / `&mut T` | |

### Print

| Boring | Rust std | Rust-kernel |
|--------|----------|-------------|
| `print "…"` | `println!("…")` | `kernel::pr_info!("…\n")` |

---

## Error handling

All errors collapse to `kernel::error::Error` (errno-based `i32`).

`throws` without a type produces `Result<T, kernel::error::Error>`. Typed `catch` branches are a known limitation — see [Known limitations](#known-limitations).

---

## `task def` — work items

In the standard backend, `task def` compiles to `async fn` + `tokio::spawn`.
In the kernel backend it generates a **work item** dispatched on `system_wq`:

```boring
task def Page fetch_page(string url):
    # fetch logic
```

```rust
// Generated (kernel)
struct FetchPageWork {
    url:       kernel::str::CString,
    result:    Arc<kernel::sync::Mutex<Option<Result<Page, kernel::error::Error>>>>,
    done_cond: Arc<kernel::sync::CondVar>,
    work:      kernel::workqueue::Work<FetchPageWork>,
}

impl kernel::workqueue::Work<FetchPageWork> for FetchPageWork {
    fn run(this: Arc<Self>) {
        let r = fetch_page_body(this.url.clone());
        *this.result.lock() = Some(r);
        this.done_cond.notify_all();
    }
}

fn fetch_page(url: CString) -> KernelFuture<Page> { … }
fn fetch_page_body(url: CString) -> Result<Page, kernel::error::Error> { … }
```

**Constraint on `self`:** `task def` as an instance method is only allowed on `'shared`, `'actor`, and `'guard` receivers — the only qualifiers with a lifetime compatible with a work item. `T&`, `T&mut`, and `T'new` are rejected by the validator.

### `KernelFuture<T>`

Every `task def` returns a `KernelFuture<T>` instead of a tokio `JoinHandle`:

| Boring | Rust std | Rust-kernel |
|--------|----------|-------------|
| `f.wait()` | `f.await` | blocks the current thread (process context only) |
| `f.done()` | — | non-blocking poll, returns `bool` |

`wait()` must not be called from an IRQ handler or atomic context.

```rust
impl<T> KernelFuture<T> {
    pub fn done(&self) -> bool {
        self.result.try_lock().map(|g| g.is_some()).unwrap_or(false)
    }

    pub fn wait(self) -> Result<T, kernel::error::Error> {
        let mut guard = self.result.lock();
        self.done_cond.wait_while(&mut guard, |r| r.is_none());
        guard.take().unwrap()
    }
}
```

### `join`

```boring
let a, b = join (task f1(), task f2())
```

Both work items are enqueued concurrently; `.wait()` is called on each in turn. Wall time is `max(t1, t2)`.

---

## `channel` — two ring-buffer variants

Both variants use blocking `send()` / `recv()` — valid in process context only.

### `channel<T, N>` — const-generic, stack buffer

The capacity is a compile-time type parameter. The buffer is a fixed-size array `[Option<T>; N]` allocated inline — no heap allocation.

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

`KernelChan<T, const N: usize>` is emitted as a prelude:

```rust
pub struct KernelChan<T, const N: usize> {
    buf:       [Option<T>; N],
    read_idx:  usize,
    write_idx: usize,
    count:     usize,
}
// KernelSender / KernelReceiver wrap Arc<Mutex<KernelChan<T, N>>>
// + two CondVars (not_empty, not_full).
// send() blocks on not_full; recv() blocks on not_empty.
```

### `channel<T>(cap)` — runtime capacity, heap buffer

The capacity is a call argument. The buffer is a `Vec<Option<T>>` pre-allocated on the kernel heap.

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

| Variant | Boring syntax | Buffer | Rust types |
|---------|--------------|--------|------------|
| Const-generic | `channel<T, N>` | `[Option<T>; N]` — stack | `KernelSender<T, N>` / `KernelReceiver<T, N>` |
| Dynamic | `channel<T>(cap)` | `Vec<Option<T>>` — heap | `DynKernelSender<T>` / `DynKernelReceiver<T>` |

Omitting the capacity uses the heap variant with a fallback of `cap = 2` and emits a warning — specify the capacity explicitly in production code.

---

## `stream` — sequential iterator or work item

### Sequential stream

If the body has no `wait` and no `task` calls, the stream compiles to a plain iterator — no workqueue involved.

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

### Async stream — work item + `KernelReceiver<T, N>`

If the body contains `wait` or `task` calls, the stream becomes a work item. The function returns a `KernelReceiver<T, N>`; the caller pulls values with `.recv()`. Use `stream<N>` to set the capacity explicitly (defaults to 2).

```boring
stream<32> string read_lines(string path):
    let file = File.open(path)
    for line in file.lines():
        yield line
```

```rust
// Generated (kernel) — three pieces:

struct ReadLinesWork {
    path: CString,
    tx:   KernelSender<CString, 32>,
    work: kernel::workqueue::Work<ReadLinesWork>,
}

impl kernel::workqueue::Work<ReadLinesWork> for ReadLinesWork {
    fn run(this: Arc<Self>) {
        let file = File::open(&this.path);
        for line in file.lines() {
            this.tx.send(line);
        }
        // tx dropped → signals end-of-stream
    }
}

fn read_lines(path: CString) -> KernelReceiver<CString, 32> {
    let (tx, rx) = kernel_channel::<CString, 32>();
    let work = Arc::new(ReadLinesWork { path, tx, work: kernel::workqueue::Work::new() });
    kernel::workqueue::system().enqueue(Arc::clone(&work));
    rx
}
```

| Strategy | Condition | Rust emitted |
|----------|-----------|--------------|
| Sequential | No `wait`, no `task` | `impl Iterator<Item = T>` |
| Async | Has `wait` or `task` | `KernelReceiver<T, N>` via `Work` on `system_wq` |

---

## Other concurrency primitives

| Boring | Rust-kernel |
|--------|-------------|
| `oneshot<T>()` | kernel oneshot prelude |
| `watch<T>(initial)` | kernel watch prelude — latest-value broadcast |
| `broadcast<T, N>` | kernel static broadcast prelude |
| `broadcast<T>(cap)` | kernel dynamic broadcast prelude |
| `wait Duration` | `kernel::delay::coarse_sleep` — blocking, no await |

---

## Stdlib

| Boring | Rust std | Rust-kernel | Notes |
|--------|----------|-------------|-------|
| `print "…"` | `println!` | `kernel::pr_info!` | always `pr_info!` — no severity routing |
| `assert_eq!` | `assert_eq!` | plain `assert_eq!` | no `WARN_ON` mapping |
| `panic(…)` | `panic!` | — | forbidden — use `throws` / `Result` |
| math (`sqrt`, `sin`…) | `std::f64` | — | forbidden — FPU disabled |
| `Vec` methods | `std::vec` | `kernel::prelude::Vec` | slightly different API |
| `{K=V}` | `HashMap` | `kernel::rbtree::RBTree<K,V>` | ordered, O(log n) |

---

## Forbidden constructs

**Errors** — rejected before emission:

| Construct | Reason |
|-----------|--------|
| `float`, floating-point math | FPU disabled |
| `panic(…)` | kernel oops/crash — use `throws` / `Result` |
| `task def` on `self` with `T&`, `T&mut`, `T'new` | lifetime incompatible with a work item |
| `kernel Foo: ...` (GPU kernel struct) | no host/device split under `no_std` — GPU kernels require `--target cuda`, `--target metal`, `--target wgpu`, or `--target rocm` |

**Warnings** — emitted, but explicit specification is recommended:

| Construct | Behaviour |
|-----------|-----------|
| `channel<T>` without capacity | defaults to heap variant with `cap = 2` |
| `stream` without `<N>` (async body) | defaults to `N = 2` |
| `T'shared` | `Rc<T>` replaced by `Arc<T>` — `Rc` unavailable in `no_std` |

---

## Known limitations

- **`float` detection is incomplete** — the validator rejects float literals and a fixed list of float math builtins, but a `float`-typed parameter or binding with no literal passes validation and is emitted as `f64` with a comment marker. Treat `float` as unsupported regardless.
- **Typed `catch` branches** — `throws MyError` and `catch MyError` are not implemented. The `type X as kernel.error.Error(ERRNO)` errno-binding syntax is not recognized by the parser. All errors collapse to `kernel::error::Error`; `catch` clauses are silently dropped in generated code.
- **`'actor'task` / `'guard'task`** — these qualifiers have no match arm in the kernel emitter and fall through to `&T`. Use plain `'actor` / `'guard` in kernel context.
- **`assert_eq!`** — no kernel-specific override; plain `assert_eq!` is emitted without mapping to `WARN_ON` or `kernel::build_assert!`.

---

## Transpiler architecture

A second emission backend within the same binary — parser, AST, and typing passes are shared and untouched.

```
src/
  validator/
    kernel.rs          # validation pass: rejects forbidden constructs
  transpiler/
    kernel/
      mod.rs           # entry point, selected by --target kernel
      emit_top.rs      # struct/impl/fn declarations, Work item generation
      emit_stmt.rs     # statements, task expr, channel, yield
      emit_expr.rs     # expressions, string literal → c_str!, method calls
      helpers.rs       # KernelFuture<T>, KernelChan<T> runtime types
```

The `--target kernel` flag activates the kernel validator then the kernel emitter. The standard backend (`--target rust`, default) is not modified.
