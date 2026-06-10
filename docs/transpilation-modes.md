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

| Qualifier     | `--threading multi` (default)        | `--threading single`       |
|---------------|--------------------------------------|----------------------------|
| `T'stack`     | `T`                                  | `T`                        |
| `T'heap`      | `Box<T>`                             | `Box<T>`                   |
| `T'copy`      | `T` (Copy)                           | `T` (Copy)                 |
| `T'shared`    | `Arc<T>`                             | `Rc<T>`                    |
| `T'actor`     | `Arc<tokio::sync::Mutex<T>>`         | `RefCell<T>`               |
| `T'guard`     | `Arc<tokio::sync::RwLock<T>>`        | `RefCell<T>`               |
| `T'wshared`   | `std::sync::Weak<T>`                 | `Weak<T>`                  |
| `T'wactor`    | `std::sync::Weak<Mutex<T>>`          | `Weak<RefCell<T>>`         |
| `T'wguard`    | `std::sync::Weak<RwLock<T>>`         | `Weak<RefCell<T>>`         |
| `T'const`     | `&'static T`                         | `&'static T`               |
| `T'option`    | `Option<T>`                          | `Option<T>`                |

`T'wshared`, `T'wactor`, `T'wguard` are shorthands for `T'shared'weak`, `T'actor'weak`, `T'guard'weak` — they produce a non-owning pointer that does not prevent the pointee from being dropped.

`T'guard` in single-thread maps to `RefCell<T>` (same as `T'actor`) — `RefCell` has no read/write distinction. The qualifier is preserved as documentation of intent; no warning is emitted.

---

## Anonymous forms — `T` and `T'`

Both forms delegate the memory decision to the active flags.

| Form  | Meaning                                        |
|-------|------------------------------------------------|
| `T`   | Anonymous, no indirection hint — flags decide  |
| `T'`  | Anonymous, indirection needed — flags decide   |
| `T?`  | Always `Option<T>` — flag-independent          |

### Resolution by flag combination

|                   | `--threading multi`          | `--threading single`         |
|-------------------|------------------------------|------------------------------|
| **`--mode strict`**   | `T` → `T`, `T'` → `Box<T>`  | `T` → `T`, `T'` → `Box<T>`  |
| **`--mode managed`**  | `T`/`T'` → `Arc<Mutex<T>>`  | `T`/`T'` → `RefCell<T>`     |

Threading does not affect `T`/`T'` in strict mode — stack and heap are thread-agnostic. It only matters in managed mode and for explicit `T'shared`, `T'actor`, `T'guard` qualifiers.

---

## Inference — structural constraints

Before applying flag defaults, the transpiler applies structural inference. These take priority over the flags; the developer can always override with an explicit qualifier.

| Priority | Situation | Result |
|----------|-----------|--------|
| 1 | Explicit qualifier written by the developer | as written |
| 2 | Non-parametric enum (all unit variants) | `T'copy` — always, regardless of flags |
| 3 | Recursive type position | `Box<T>` inserted on the recursive field/variant |
| 4 | `dyn Trait` position | `Box<dyn Trait>` |
| 5 | `sizeof(T) > 1024` in strict mode | `T` auto-promoted to `Box<T>` |

Priorities 2–4 are correctness constraints — they apply in all flag combinations. Priority 5 applies only in `--mode strict` and can be suppressed with an explicit `T'stack` qualifier.

**Non-parametric enums** — an enum whose every variant carries no payload is a C-style discriminant, always `Copy`:

```
enum Color { Red, Green, Blue }         # → T'copy, even in managed mode
enum Direction { North, South, East }   # → T'copy, even in managed mode
```

---

## `--mode strict` (default)

Targets production code. After inference, unresolved anonymous forms:

- `T'` → `Box<T>`
- `T` → plain `T` (stack, single owner)

---

## `--mode managed`

Targets prototyping and scripting. After inference, both `T` and `T'` map to the thread-appropriate wrapped type — `Arc<Mutex<T>>` with `--threading multi`, `RefCell<T>` with `--threading single`.

**Managed mode and field access.** `std::sync::Mutex` is non-reentrant. Accessing the same managed variable's fields multiple times in one expression would normally require multiple `lock()` calls, causing deadlock. The transpiler avoids this by emitting a shadow guard immediately after each managed variable is declared:

```rust
// Generated Rust — managed + multi
let ap = Arc::new(std::sync::Mutex::new(APoint { x: 3, y: 4 }));
let mut __ap_mg = ap.lock().unwrap();   // shadow guard — emitted automatically
println!("{}", (__ap_mg.x + __ap_mg.y)); // uses guard, not ap.lock() twice
```

The shadow guard is emitted for both function parameters and local `T`/`T'` declarations.

**Typical workflow:**

```
# Phase 1 — prototype without explicit ownership
boring build --mode managed

# Phase 2 — annotate hot paths, switch to strict
boring build --mode strict
```

---

## `--threading single` — runtime and limitations

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

The transpiler uses the **`local-channel`** crate (v0.1.5, maintained by the actix-web team) for `!Send` async channels in `LocalSet`/`current_thread` contexts:

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

Internally it uses `Rc<RefCell<VecDeque<T>>>` + standard Tokio `Waker`. No unsafe code is required.

### Streams

Two sub-cases depending on the stream body:

| Case | Detection | Implementation | All targets |
|------|-----------|----------------|-------------|
| In-memory stream | no `await`, no I/O in body | `impl Iterator<Item = T>` | ✓ |
| Async source (I/O, timer) | `await` or I/O call present | target-specific (see below) | |

**In-memory streams compile to `impl Iterator<Item = T>` on all targets.** `Iterator` requires neither `Send` nor a runtime.

| Target | Async stream implementation |
|--------|-----------------------------|
| multi-thread | `tokio::sync::mpsc` channel + `spawn` |
| single-thread | `local-channel` + `spawn_local` / `LocalSet` |
| kernel | `KernelChan<T, N>` ring buffer + `system_wq` work item |

### Why `Rc` is not made `Send`

`Rc` is defined in the Rust stdlib — its `!Send` impl cannot be overridden from outside. A newtype with `unsafe impl Send` is technically possible but introduces unsoundness if generated code is ever used as a library dependency in a multi-thread context.

`!Send` is a deliberate invariant. The correct answer is `LocalSet` + `spawn_local` + `local-channel`, which keeps the compiler as the safety guarantor.

---

## Size-based inference (strict mode only)

| Range | Action |
|-------|--------|
| `sizeof(T) > 1024` | Auto-promote `T` → `Box<T>` (silent in output, warning to stderr) |
| `32 < sizeof(T) <= 1024` | Warning: suggest explicit `T'` |
| `sizeof(T) <= 32` | No action |

Size is computed statically by the transpiler by summing field sizes. Generic type parameters and extern types have unknown size and are skipped (no action taken).

Configurable thresholds: `--stack-auto-bytes N` (default 1024) and `--stack-warn-bytes N` (default 32).

## Enum size warnings (strict mode only)

```
# Level 1 — overall enum too large
warning: `Message` is 1025 bytes on the stack (largest variant: `Data`);
         consider `Message'` to heap-allocate the whole enum

# Level 2 — one variant disproportionate
warning: variant `Data` (1024 bytes) dominates `Message` (4 bytes median);
         consider boxing the payload: Data([u8; 1024]')
```

Level 2 keeps the enum on the stack while boxing only the heavy payload.
