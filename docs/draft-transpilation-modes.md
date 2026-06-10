# Draft — Transpilation Modes

> Status: draft — not yet implemented. Design agreed, implementation deferred.

## Motivation

The current qualifier system is expressive but requires the developer to annotate every type with an explicit ownership model. For prototyping or scripts, this overhead is unnecessary. Two orthogonal flags let the developer choose the right trade-off independently.

---

## Two orthogonal flags

```
boring build --mode strict|managed     # memory management (default: strict)
boring build --threading single|multi  # concurrency model (default: multi)
```

These flags are independent. Any combination is valid.

---

## Qualifier vocabulary

`T'auto` and `T'task` are replaced by a single `T'shared` qualifier. The `--threading` flag determines the Rust type — the qualifier expresses intent, the flag expresses the execution context.

### Full qualifier table

| Qualifier   | `--threading multi` (default)    | `--threading single`  |
|-------------|----------------------------------|-----------------------|
| `T'stack`   | `T`                              | `T`                   |
| `T'heap`    | `Box<T>`                         | `Box<T>`              |
| `T'copy`    | `T` (Copy)                       | `T` (Copy)            |
| `T'shared`  | `Arc<T>`                         | `Rc<T>`               |
| `T'actor`   | `Arc<tokio::sync::Mutex<T>>`     | `RefCell<T>`          |
| `T'guard`   | `Arc<tokio::sync::RwLock<T>>`    | `RefCell<T>`          |
| `T'wshared` | `std::sync::Weak<T>`             | `Weak<T>`             |
| `T'wactor`  | `std::sync::Weak<Mutex<T>>`      | `Weak<RefCell<T>>`    |
| `T'wguard`  | `std::sync::Weak<RwLock<T>>`     | `Weak<RefCell<T>>`    |
| `T'const`   | `&'static T`                     | `&'static T`          |
| `T'option`  | `Option<T>`                      | `Option<T>`           |

Explicit qualifiers are **contracts** — neither `--mode` nor `--threading` can override them.

> **Note — `T'task` in single-thread mode**: `T'shared` makes `T'task` and `T'auto` redundant. If the developer writes `T'task`, the transpiler should emit a warning suggesting `T'shared` and treat it identically.

> **Note — `T'guard` and `T'wguard` in single-thread mode**: `T'guard` is a deliberate developer choice expressing read-write lock semantics. In single-thread mode it maps to `RefCell<T>` (same as `T'actor`) — `RefCell` has no read/write distinction, but the qualifier is preserved as documentation of intent. No warning emitted. Same applies to `T'wguard` → `Weak<RefCell<T>>`.

> **Note — `--mode managed` + `--threading single`**: valid combination, accepted for prototyping. `T`/`T'` resolve to `RefCell<T>` — lightweight interior mutability without any threading overhead.

---

## Short forms: `T` and `T'`

Both short forms are **anonymous** — the developer delegates the memory decision to the active flags.

| Form  | Meaning                                       |
|-------|-----------------------------------------------|
| `T`   | anonymous, no indirection hint — flags decide |
| `T'`  | anonymous, indirection needed — flags decide  |
| `T?`  | always `Option<T>` (flag-independent)         |

### Short form resolution by flag combination

| | `--threading multi` | `--threading single` |
|---|---|---|
| **`--mode strict`** | `T` → `T`, `T'` → `Box<T>` | `T` → `T`, `T'` → `Box<T>` |
| **`--mode managed`** | `T`/`T'` → `Arc<Mutex<T>>` | `T`/`T'` → `RefCell<T>` |

Threading does not affect `T`/`T'` in strict mode — stack and heap are thread-agnostic. It only matters in managed mode and for explicit shared/actor/guard qualifiers.

---

## Inference hierarchy

Before applying flag defaults, the transpiler runs a structural inference pass. Inference conclusions take priority over the flags. The developer can always override with an explicit qualifier.

| Priority | Situation | Result | Both flags |
|----------|-----------|--------|------------|
| 1 | Explicit qualifier written by the developer | as written | ✓ |
| 2 | Non-parametric enum (all unit variants) | `T'copy` | ✓ |
| 3 | Recursive type position | `Box<T>` on the recursive field/variant | ✓ |
| 4 | `dyn Trait` position | `Box<dyn Trait>` | ✓ |
| 5 | `size_of::<T>() > 1024` | silent promotion to `Box<T>` | strict only |
| 6 | `32 < size_of::<T>() <= 1024` | warning — suggest `T'` | strict only |
| 7 | Disproportionate enum variant (payload >> other variants) | warning — suggest boxing the variant payload | strict only |
| 8 | None of the above | flag defaults | |

Priorities 2–4 are correctness constraints — they apply regardless of flags. Priority 5 is the only case of silent promotion: above 1 KB the stack cost is high enough that the transpiler decides without asking. Priority 6–7 are warnings only.

### Size thresholds

Size is computed via `std::mem::size_of::<T>()` — a compile-time constant in Rust, exact for all concrete types. Generic type parameters and extern types have unknown size and are skipped (no inference, no warning).

| Range | Action |
|-------|--------|
| `size_of::<T>() > 1024` | Auto-promote `T` → `Box<T>`, no warning |
| `32 < size_of::<T>() <= 1024` | Warning: suggest `T'` |
| `size_of::<T>() <= 32` | No action |

### Non-parametric enums

An enum whose every variant carries no payload is a C-style discriminant — always `Copy`, always tiny:

```
enum Color { Red, Green, Blue }       # → T'copy always, even in managed mode
enum Direction { North, South, East } # → T'copy always
```

### Enum-specific size warnings

For enums with parametric variants, on-stack size is `max(variant sizes) + discriminant`:

```
enum Message {
    Ping,              # 0 bytes
    Pong,              # 0 bytes
    Data([u8; 1024]), # 1024 bytes — dictates the whole enum size
}
```

Two distinct warnings emitted in strict mode:

```
# Level 1 — overall enum too large
warning: `Message` is 1025 bytes on the stack (largest variant: `Data`);
         consider `Message'` to heap-allocate the whole enum

# Level 2 — one variant disproportionate (variant >> median)
warning: variant `Data` (1024 bytes) dominates `Message` (4 bytes median);
         consider boxing the payload: Data([u8; 1024]')
```

Level 2 is more useful in practice: it keeps the enum on the stack while boxing only the heavy payload.

---

## Mode details

### `--mode strict` (default)

Targets production code. After inference, unresolved anonymous forms:

1. `T'` → `Box<T>`
2. `T` → plain `T` (stack, single owner)

Size-based warnings active.

### `--mode managed`

Targets prototyping and scripting. After inference, both `T` and `T'` map to the thread-appropriate actor type (`Arc<Mutex<T>>` in multi, `RefCell<T>` in single). Size-based warnings disabled.

#### Typical workflow

```
# Phase 1 — prototype
boring build --mode managed

# Phase 2 — annotate hot paths, switch to strict
boring build --mode strict
```

---

## Summary

|               | `T` / `T'` (multi)  | `T` / `T'` (single) | non-parametric enum | explicit qualifier |
|---------------|---------------------|---------------------|---------------------|--------------------|
| **strict**    | `T` / `Box<T>`      | `T` / `Box<T>`      | `T'copy`            | respected as-is    |
| **managed**   | `Arc<Mutex<T>>`     | `RefCell<T>`        | `T'copy`            | respected as-is    |

---

## `--threading single` — runtime and limitations

### Tokio runtime

`--threading single` generates a `current_thread` Tokio runtime:

```rust
#[tokio::main(flavor = "current_thread")]
async fn main() { ... }
```

Everything runs on the main thread cooperatively. `Rc<T>` and `RefCell<T>` are safe because there is no other thread for values to escape to. The Rust compiler still enforces the boundary — it just never triggers in this context. `spawn` is replaced by `spawn_local`, which accepts `!Send` futures.

### Limitation — channels and streams require `Send`

Even in `current_thread` mode, Tokio channels (`mpsc`, `oneshot`, `broadcast`) require `T: Send` on the values they carry — this is a generic bound in Tokio's API, not a runtime requirement. A `Rc<T>` or `RefCell<T>` cannot be sent through a channel even if there is only one thread.

Consequences for boring:

- `Rc<T>` and `RefCell<T>` are usable **locally within a task** — fields, local variables, intra-task computation.
- As soon as a value must **cross a task boundary** (channel, stream, `spawn_local` with a moved value that contains `Rc`), it must be `Send`. The `T'shared`/`T'actor` qualifiers on communicated types effectively resolve to `Arc`/`Mutex` even in single-thread mode.
- `stream<N>` in boring (worker-based, channel-backed) requires `Send` on the stream item type. Single-thread mode does not relax this.

**Practical scope of `--threading single`**: useful for sequential async I/O without inter-task communication — lightweight scripts, CLI tools, simple request handlers. As soon as streams or channels carry non-trivial shared state, the `Send` constraint reappears and `Arc`/`Mutex` become necessary anyway.

The transpiler should emit a warning when a `!Send` type (`Rc`, `RefCell`) is detected in a position that requires `Send` (channel payload, stream item, `spawn_local` capture that crosses an await point).

### Design note — why `Rc` is not made `Send` in single-thread mode

One might consider making `Rc<T>` implement `Send` when the transpiler knows the target is single-thread. This is not viable:

- `Rc` is defined in the Rust stdlib — its `!Send` impl cannot be overridden from outside.
- A newtype `struct LocalRc<T>(Rc<T>)` with `unsafe impl Send` is technically possible but introduces unsoundness: if generated code is ever used as a dependency in a multi-thread context (library, FFI, test harness), the compiler no longer protects against data races — silent undefined behaviour.
- There is no Rust compiler mode that relaxes `Send` checks globally based on a single-thread promise.

`!Send` is a deliberate invariant, not a limitation. The correct answer is `LocalSet` + `spawn_local` + `local-channel`, which keeps the compiler as the safety guarantor. The constraint is a feature, not a workaround.

### Streams — in-memory vs async source

Two sub-cases depending on the stream body:

| Case | Detection | Implementation | All targets |
|------|-----------|----------------|-------------|
| In-memory stream | no `await`, no I/O, no external `yield` in body | `impl Iterator<Item = T>` | ✓ |
| Async source (I/O, timer) | `await` or I/O call present in body | target-specific (see below) | |

**In-memory streams compile to `impl Iterator<Item = T>` on all targets** — multi-thread, single-thread, and kernel. `Iterator` requires neither `Send` nor a runtime. The `for` loop consumes it lazily on the current thread/context with zero scheduling overhead. The transpiler detects this case statically: if the `stream def` body contains no `await`, no I/O call, and no yield from an external source, it emits an iterator regardless of target or mode.

| Target | Async stream implementation |
|--------|-----------------------------|
| multi-thread | channel-backed (`tokio::sync::mpsc`) + `spawn` |
| single-thread | `local-channel` + `spawn_local` / `LocalSet` |
| kernel | `KernelChan<T, N>` (ring buffer + `Mutex`/`CondVar`) + `system_wq` work item |

**Reference — Rust-for-Linux stream implementation**: `src/transpiler/kernel/helpers.rs` lines 137–215 (ring buffer) and `src/transpiler/kernel/emit_top.rs` lines 244–375 (stream generation). The ring buffer pattern is reusable for single-thread by replacing `Arc<Mutex<...>> + CondVar` with `Rc<RefCell<VecDeque<T>>> + Waker`.

### `!Send` channels in single-thread mode — resolved

A `!Send` async channel for `LocalSet` / `current_thread` is fully feasible. The crate **`local-channel`** (maintained by the actix-web team, v0.1.5) provides exactly this:

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

Internally it uses `Rc<RefCell<VecDeque<T>>>` + standard Tokio `Waker` (cloned from `cx.waker()`). No unsafe code required in boring.

**Implementation notes if a custom channel is needed** (bounded, priority, etc.):
- Always replace the waker on every `poll` — not just when `None`.
- `take()` the waker before calling `wake()` to avoid double-borrow on `RefCell`.
- Decrement `tx_count` in `Drop for Sender`; wake the receiver when it reaches zero (EOF signal).
- For bounded channels, store a `VecDeque<Waker>` for blocked senders; wake one on each `recv`.

---

## Implementation notes

- Two CLI flags: `--mode strict|managed` (default: strict), `--threading single|multi` (default: multi).
- `--threading` is only available for Tokio-based targets. It is not available for the Rust-for-Linux target, which has its own concurrency model (workqueues, spinlocks, kernel interrupts) independent of Tokio. Passing `--threading` with a Rust-for-Linux target is an error.
- `T'auto` and `T'task` removed from the qualifier set. Emit a deprecation warning if encountered, treat as `T'shared`.
- Inference pass runs before flag resolution in all combinations. Passes in order:
  1. Detect non-parametric enums → mark as `Copy`.
  2. Detect recursive type positions → insert `Box` on the offending field/variant.
  3. Detect `dyn Trait` positions → wrap in `Box`.
  4. (strict only) Compute enum variant size distribution → emit level-2 warning if one variant dominates.
  5. (strict only) Compute overall type size → emit level-1 warning if above threshold.
- Size computation via `std::mem::size_of::<T>()` — compile-time constant, exact for concrete types. Skip generics and extern types.
- Thresholds: `> 1024` bytes → auto `Box<T>`; `> 32` bytes → warning. Configurable via `--stack-auto-bytes N` (default 1024) and `--stack-warn-bytes N` (default 32).
- The qualifier table in `book.md` §21 needs updating: replace `T'auto`/`T'task` with `T'shared`, add threading column, document short form matrix.
