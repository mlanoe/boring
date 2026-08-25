# Transpilation Modes

Two orthogonal flags control how the transpiler handles memory management and concurrency:

```
boring build --mode strict|managed     # memory management (default: strict)
boring build --threading single|multi  # concurrency model (default: multi)
```

The flags are independent — any combination is valid. `--threading` is not available with `--target kernel`, which has its own concurrency model.

---

## Qualifier vocabulary

Explicit qualifiers are **contracts** — neither `--mode` nor `--threading` affects them.

| Qualifier          | `--threading multi` (default)        | `--threading single`       |
|--------------------|--------------------------------------|----------------------------|
| `T'inline`         | `T`                                  | `T`                        |
| `T'owned`          | `Box<T>`                             | `Box<T>`                   |
| `T'shared`         | `Arc<T>`                             | `Rc<T>`                    |
| `T'actor`          | `Arc<std::sync::Mutex<T>>`           | `Rc<RefCell<T>>`           |
| `T'guard`          | `Arc<std::sync::RwLock<T>>`          | `Rc<RefCell<T>>`           |
| `T'actor'task`     | `Arc<tokio::sync::Mutex<T>>`         | `Rc<RefCell<T>>`           |
| `T'guard'task`     | `Arc<tokio::sync::RwLock<T>>`        | `Rc<RefCell<T>>`           |
| `T'shared'weak`    | `std::sync::Weak<T>`                 | `Weak<T>`                  |
| `T'actor'weak`     | `std::sync::Weak<Mutex<T>>`          | `Weak<RefCell<T>>`         |
| `T'guard'weak`     | `std::sync::Weak<RwLock<T>>`         | `Weak<RefCell<T>>`         |
| `T'option`         | `Option<T>`                          | `Option<T>`                |
| `string`           | `Arc<str>`                           | `Rc<str>`                  |

`T'shared'weak`, `T'actor'weak`, `T'guard'weak` are weak non-owning pointers that do not prevent the pointee from being dropped.

`T'guard` in single-thread maps to `Rc<RefCell<T>>` (same as `T'actor`) — `RefCell` has no read/write distinction. The qualifier is preserved as documentation of intent; no warning is emitted.

`T'actor'task` and `T'guard'task` use Tokio's async-aware mutexes under `--threading multi`. Under `--threading single` they fall back to `Rc<RefCell<T>>` (same as `T'actor`/`T'guard`) — access goes through `.borrow()`/`.borrow_mut()` instead of `.lock().await`. This is not an error; the `'task` hint is simply moot when there is no async mutex to use.

---

## Anonymous forms — `T` and `T'new`

Both forms delegate the memory decision to the inference pass, then to the active flags as a last resort.

| Form  | Meaning                                                         |
|-------|-----------------------------------------------------------------|
| `T`   | Anonymous — inference decides; fallback to flags               |
| `T'new` | Anonymous with indirection hint — inference decides; fallback `Box<T>` |
| `T?`  | Always `Option<T>` — flag-independent                          |

The difference between `T` and `T'new`: `T'new` restricts the inference candidate set to `{Owned, Shared, Actor, Guard}` (Inline and Const are excluded). If inference cannot resolve a unique qualifier, the fallback is `'owned` instead of `'inline`.

See [qualifiers.md](qualifiers.html) — **Inference** section — for the full inference algorithm.

### Resolution after inference

|                       | `--threading multi`          | `--threading single`         |
|-----------------------|------------------------------|------------------------------|
| **`--mode strict`**   | `T` → `T`, `T'new` → `Box<T>`  | `T` → `T`, `T'new` → `Box<T>`  |
| **`--mode managed`**  | `T`/`T'new` → `Arc<Mutex<T>>`  | `T`/`T'new` → `RefCell<T>`     |

Threading does not affect the `T`/`T'new` fallback in strict mode — inline storage and heap allocation are thread-agnostic. It only matters in managed mode and for explicit `T'shared`, `T'actor`, `T'guard` qualifiers.

---

## Inference priority table

Before applying flag defaults, the transpiler runs a series of inference passes. Each pass has a fixed priority; the first pass that resolves a qualifier wins.

| Priority | Situation | Result |
|----------|-----------|--------|
| 1 | Explicit qualifier written by the developer | as written |
| 2 | Recursive type position | `Box<T>` inserted on the recursive field/variant |
| 3 | `dyn Trait` position | `Box<dyn Trait>` |
| 4 | Use-site qualifier inference | qualifier demanded by call sites — all modes |
| 5 | `sizeof(T) > --inline-auto-bytes` in strict mode | `T` promoted to `Box<T>` |

Priorities 2–3 are correctness constraints — they are never overridden. Priority 4 (use-site inference) runs before size-based decisions and applies in all flag combinations. Priority 5 applies only in `--mode strict` when priority 4 yields no result. Flag defaults are applied last.

---

## `--mode strict` (default)

Targets production code. After inference, unresolved anonymous forms:

- `T` → plain `T` (inline, single owner)
- `T'new` → `Box<T>`

---

## `--mode managed`

Targets prototyping and scripting. After inference, both `T` and `T'new` map to the thread-appropriate wrapped type — `Arc<Mutex<T>>` with `--threading multi`, `RefCell<T>` with `--threading single`.

**Managed mode and field access.** `std::sync::Mutex` is non-reentrant. Accessing the same managed variable's fields multiple times in one expression would normally require multiple `lock()` calls, causing deadlock. The transpiler avoids this by emitting a shadow guard immediately after each managed variable is declared:

```rust
// Generated Rust — managed + multi
let ap = Arc::new(std::sync::Mutex::new(APoint { x: 3, y: 4 }));
let mut __ap_mg = ap.lock().unwrap();   // shadow guard — emitted automatically
println!("{}", (__ap_mg.x + __ap_mg.y)); // uses guard, not ap.lock() twice
```

The shadow guard is emitted for both function parameters and local `T`/`T'new` declarations, but only under `--threading multi` (where the managed wrapper is `Mutex`). Under `--threading single` the managed wrapper is `RefCell`, which supports independent `.borrow()`/`.borrow_mut()` calls without a deadlock risk, so no shadow guard is emitted — field accesses go through `managed_refcell_vars`-based `.borrow()`/`.borrow_mut()` calls directly instead.

**Typical workflow:**

```
# Phase 1 — prototype without explicit ownership
boring build --mode managed

# Phase 2 — annotate hot paths, switch to strict
boring build --mode strict
```

---

## `--threading single` — runtime and limitations

`--threading single` emits to `<project>_rust_single/` instead of `<project>_rust/`, so both builds can coexist side by side in the same directory.

### Tokio runtime

`--threading single` generates a `current_thread` Tokio runtime:

```rust
#[tokio::main(flavor = "current_thread")]
async fn main() { ... }
```

Everything runs on the main thread cooperatively. `Rc<T>` and `RefCell<T>` are safe because there is no other thread for values to escape to. `spawn` is replaced by `spawn_local`, which accepts `!Send` futures.

### Channels — `!Send` constraint

Even in `current_thread` mode, Tokio channels (`mpsc`, `oneshot`, `broadcast`) require `T: Send` on the values they carry — this is a generic bound in Tokio's API, not a runtime requirement. `Rc<T>` and `RefCell<T>` cannot be sent through a channel even if there is only one thread.

Consequences:

- `Rc<T>` and `RefCell<T>` are usable **within a task** — fields, local variables, intra-task computation.
- As soon as a value must **cross a task boundary** (channel, stream, `spawn_local` with a moved value containing `Rc`), it must be `Send`. The `T'shared`/`T'actor` qualifiers on communicated types effectively resolve to `Arc`/`Mutex` even in single-thread mode.

The transpiler uses the **`local-channel`** crate (v0.1.5) for `!Send` async channels in `LocalSet`/`current_thread` contexts:

```toml
[dependencies]
local-channel = "0.1"
```

```rust
use local_channel::mpsc;

let (tx, mut rx) = mpsc::channel();
spawn_local(async move { tx.send(42).unwrap(); });
spawn_local(async move {
    while let Some(val) = rx.next().await { println!("{val}"); }
});
```

### Streams

Two sub-cases depending on the stream body:

| Case | Detection | Implementation | All targets |
|------|-----------|----------------|-------------|
| In-memory stream | no `await`, no I/O in body | `impl Iterator<Item = T>` | ✓ |
| Async source (I/O, timer) | `await` or I/O call present | target-specific (see below) | |

| Target | Async stream implementation |
|--------|-----------------------------|
| multi-thread | `tokio::sync::mpsc` channel + `spawn` |
| single-thread | `local-channel` + `spawn_local` / `LocalSet` |
| kernel | `KernelChan<T, N>` ring buffer + `system_wq` work item |

### Why `Rc` is not made `Send`

`Rc` is defined in the Rust stdlib — its `!Send` impl cannot be overridden from outside. A newtype with `unsafe impl Send` is technically possible but introduces unsoundness if generated code is ever used as a library dependency in a multi-thread context.

`!Send` is a deliberate invariant. The correct answer is `LocalSet` + `spawn_local` + `local-channel`, which keeps the compiler as the safety guarantor.

---

## Size-based auto-boxing (strict mode only)

When use-site inference (priority 5) does not resolve a qualifier for an anonymous `T`, the transpiler falls back to a size-based decision:

| Estimated size | Action |
|---|---|
| ≤ `--inline-auto-bytes` (default: 256 B) | leave as `T` (inline) |
| > `--inline-auto-bytes` | silently promote to `Box<T>` |

Configurable: `boring build --inline-auto-bytes N`.

**Suppressed for bare `T` struct fields, regardless of rebindability.** When a struct field uses the bare anonymous form `T` (no qualifier), size-based promotion does not apply and no warning is emitted — this holds for `let`, `mut`, and `var` fields alike. A struct field is always inline in the parent's allocation — boxing it would add an indirection without reducing the parent's layout.

This suppression applies only to bare `T`. A `T'new` field (indirection hint) is **not** suppressed: its fallback remains `'owned` (`Box<T>`) regardless of size, because the developer explicitly requested indirection.

```boring
struct BigData:
    let float[256] samples    # 2048 bytes — bare T, no Box, no warning; inline in BigData
    let string name

struct Wrapper:
    var BigData inner         # bare T (var too) — sizeof(BigData) folds into sizeof(Wrapper), no promotion
    let BigData'new backup    # T'new — always Box<BigData>, even on a let field
```

## Enum size warnings (strict mode only)

```
# Level 1 — overall enum too large
warning: `Message` is 260 bytes inline (largest variant: `Data`);
         consider `Message'owned` to heap-allocate the whole enum

# Level 2 — one variant disproportionate
warning: variant `Data` (256 bytes) dominates `Message` (4 bytes median);
         consider boxing the payload: Data(u8[256]'owned)
```

Level 2 keeps the enum inline while boxing only the heavy payload.
