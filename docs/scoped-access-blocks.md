# Scoped Access Blocks — `with`

> **Status: Shipped**, in a narrower form than originally scoped for the GPU side — see "Implementation Notes" near the end of this document for exactly what's implemented, what's verified, and what's still open. In short: `'actor`/`'actor'task`/`'guard`/`'guard'task` per-block locking is fully implemented, tested, and documented in the language book ([chapter 21](book.html#scoped-access-blocks--with)). `'gpu'unified`/`'gpu'global` GPU-residency — the original motivating problem in the Problem Statement below — is now also implemented, for the *intra-procedural* case (a kernel constructed and its field read back within the same function/scope, e.g. `examples/vector_add_gpu.br`/`matrix_mul_gpu.br`): the round-trip-per-access problem is genuinely fixed there, verified by codegen snapshot tests and cross-checked against the interpreter. Still open: an *inter-procedural* resident value returned across a function boundary (the `linear_gpu`-chain example below), and feeding a resident value directly into another kernel's constructor without a host round-trip at all (the `BoringGpuArg` sketch) — both need the kernel-struct-storage changes described in "Implementation Notes".

## Problem Statement

Two unrelated-looking performance/ergonomics gaps turned out to share one root cause: **Boring has no way to hold extended, multi-statement access to a value that is normally accessed one operation at a time.**

### 1. GPU kernel chaining forces a host round-trip

A `kernel Name:` struct's `'unified` field is read back to the host **eagerly, on every single property access** — `k.y` always triggers a full `submit → poll(Wait) → map_async → poll(Wait) → recv` round-trip, even when the very next line just re-uploads that same data into another kernel's input buffer. Chaining three GPU calls (`linear_gpu` → `gelu_gpu` → `linear_gpu`, the shape of every MLP block) costs 3 dispatches, 3 blocking readbacks, and 2 completely unnecessary re-uploads of data that was never touched by the host in between. This was found and measured in `whisper-boring`: the decoder/encoder pipeline is latency-bound on these round-trips (CPU time stays near zero for the whole run), not compute-bound.

### 2. `'actor` / `'guard` lock per call, not per critical section

A `T'actor` value acquires and releases its `Mutex` **once per method call or field access** (`c.increment()` transpiles to `c.lock().unwrap().increment()`). There is currently no way to say "hold the lock across these three statements as one atomic unit" — each call is its own critical section, so a caller who needs several operations to happen without another thread interleaving has no tool for it short of restructuring the mutation into a single method on the struct itself.

Both gaps are instances of the same missing primitive: **an explicit, lexically-scoped block that grants extended access to a value that is normally accessed atomically per-operation**, and closes by doing whatever that value's qualifier requires on exit (write back to the GPU, release a lock).

## Rejected Alternative: Implicit Lazy Materialization

The first design considered for the GPU side was fully automatic: kernel output stays GPU-resident until something forces it to materialize, transparently, with no new syntax. Concretely this means a runtime-tagged value:

```rust
enum GpuArray<T> {
    Resident { buffer: Arc<wgpu::Buffer>, len: usize },
    Materialized(Vec<T>),
}
```

Any host-side use (indexing, `.length`, `print`, passing to a non-kernel function) checks the tag, forces materialization if `Resident`, and **memoizes** the result so a second host access doesn't re-fetch. Any use as another kernel's input checks the tag the other way and skips the upload if still `Resident`.

This was rejected for two reasons:

1. **Memoization requires interior mutability behind an ostensibly-immutable `let`.** Boring's current invariant — a `let` binding is inert after creation — would silently stop being true for this one type, only for an implementation reason (avoid re-reading the same value twice), not a language reason.
2. **The clone-at-call-site convention becomes dangerous.** Boring auto-clones array *method* arguments at call sites (see `is_user_struct_receiver` in `emit_methods.rs`) so the caller keeps its own copy. Cloning a `Resident` value naively would mean a real GPU-to-GPU buffer copy — an extra dispatch, defeating the entire optimization — unless this one type quietly switches to move semantics, which is a real semantic special-case, not just an implementation detail.

Both problems come from trying to make the state **invisible**. Making it **explicit** (this document's proposal) sidesteps both: there is no tag to check at runtime, because the compiler already knows statically, from lexical scope, whether host access is legal at a given point.

## Proposed Design

### The qualifier

`'gpu'unified` and `'gpu'global` already exist in the grammar as host-context qualifiers (`parse_type.rs`), and are already documented (`gpu-module.md`, "Host-context" table) — but both are currently placeholders: the transpiler emits them as a bare `*mut T` (`emit_top.rs`, "GPU memory qualifiers: emitted as pointer types (placeholders)") with no real buffer-residency behavior behind either. This document proposes to make both the real, working representation of "GPU-resident, host-materializable on demand," backed by the `with` block below — with different cost characteristics per qualifier (see below).

**A note on the keyword**: `with` is already a reserved keyword (`spec/grammar.bnf`) with an existing use — inline `match`: `match expr with Pat1: val1, Pat2: val2, _: val3` (`parser/parse_stmt.rs:734`). Worth checking whether that's a real conflict rather than assuming either way: it isn't. That `with` is only ever consumed *inside* `parse_match_stmt`, after `match` and the subject expression have already been parsed — it never appears at the top-level statement dispatch (`parse_stmt`'s main match has no `TokenKind::With` arm at all). This document's `with <name>:` block, by contrast, would *be* a top-level statement, starting with `with` as its first token — a grammatical position the existing keyword never occupies. No lookahead, no disambiguation, no conflict: reusing `with` here is safe.

`use` — considered as an alternative — would be the worse choice despite reading naturally, precisely because it's *already* a top-level statement-dispatch keyword: `use module_path` (import) and, inside a function body, `use Name as Type` (local type alias, `parse_stmt.rs:184-194`). Adding a third meaning under the same leading token would need real lookahead to tell `use math_gpu` (import) apart from `use Name as Type` (alias) apart from `use fc:` (this proposal) — solvable, but genuine added parser complexity that `with` doesn't have to pay.

`'sync` and `'local` are **not** part of this proposal — both are documented with host access **"no"**, flatly, in `gpu-module.md`'s kernel-context qualifier table (not "direct" like `'unified`, not "via `gpu.copy()`" like `'global`), and neither has a host-context (`'gpu'...`) form to begin with. There is no readback path for a `with` block to lower to, so neither has a materialization story missing (the way `'global` did before this revision) — they structurally cannot have one.

`pub` is, for now, intentionally disallowed on kernel struct fields entirely (`KernelFieldDecl` has no `is_pub`, and the kernel-body field-parsing loop recognizes only `let`/`mut`/`var` to start a field, unlike regular struct-body parsing) — not a gap to fix. A kernel field's visibility is already expressed through its qualifier, not a separate marker: `'unified` is host-visible, `'global` needs an explicit copy, `'sync`/`'local`/`'const` have no host access at all. That's the same reason `'sync`/`'local` are out of scope here — no host access point exists to gate, by design, not because it's missing.

#### `'unified` vs `'global`: same block, different cost

Both qualifiers already have documented kernel-context meanings (`gpu-module.md`):

- **`'unified`** — "share physical memory between host and device. No explicit H2D/D2H copy is needed." On a backend with genuine unified memory (Metal on Apple Silicon, CUDA managed memory), this is close to free — no data actually moves, at most a synchronization barrier.
- **`'global`** — "device-only DRAM. Host reads/writes require explicit `gpu.copy()`" — and that copy is documented as "backend-specific (staging buffers on wgpu, `cudarc` transfers on CUDA, blit on Metal)": a real transfer, unconditionally, every time.

`gpu.copy(k.result, host_buf)` / `gpu.copy(host_buf, k.input)` is documented (`gpu-module.md`, `wgpu-backend.md`, `cuda-module.md`) as the mechanism for `'global` host access — but it turns out to be documentation for an API that was never actually built: there is no `gpu.copy` handling anywhere in the parser, checker, interpreter, or transpiler (the interpreter's runtime `gpu` object exposes `thread`/`block`/`block_dim`/`grid_dim` for device-side indexing — nothing named `copy`; the real, working mechanism is the auto-generated `copy_{field}_to_device`/`copy_{field}_to_host` kernel methods, called through ordinary method syntax, never through a free `gpu.copy(...)` form).

So this isn't a case of picking between two working mechanisms — `with` on `'gpu'global` isn't replacing something that runs today, it's simply the thing that gets built instead of the documented-but-unimplemented `gpu.copy()`. `gpu.copy()` should be dropped from the docs (`gpu-module.md`, `wgpu-backend.md`, `cuda-module.md` all reference it) once `with` lands, rather than kept alongside it — there is no reason to carry two names for the same D2H/H2D transfer, and `gpu.copy()` has zero real call sites to migrate away from.

**Caveat, verified against the current wgpu backend rather than assumed**: today, on wgpu specifically, `'unified`'s own readback (`__boring_gpu_copy_d2h`) *also* goes through a real staging-buffer copy + map — there is no zero-copy path implemented yet on this backend for either qualifier. So the cost difference described above is currently more a statement of *intent* (and of what other backends can already do) than something a wgpu-target program will feel today. It still matters for the design: `'unified` leaves room for a genuine zero-copy implementation later without changing any Boring source that uses it, while `'global` is documented never to be free, on any backend.

### The block

```boring
with <name> [, <name> ...] :
    <body>
```

The syntax itself, the AST/parser/checker below, and everything in "Typing Rules" apply uniformly to every qualifier this document covers. What's actually **implemented** differs by qualifier — `'actor`/`'guard` (and their `'task` variants) get real, working codegen today; `'gpu'unified`/`'gpu'global` do not yet (see "Implementation Notes"). The design-level reasoning in the rest of this section is unchanged either way.

#### AST shape

A new dedicated `Stmt::With(WithStmt)` variant, not a reuse of an existing block-statement shape:

```rust
#[derive(Debug, Clone)]
pub struct WithStmt {
    pub names: Vec<String>,
    pub body: Vec<Stmt>,
    pub line: usize,
    pub col: usize,
}
```

No existing `Stmt` variant is a fit. The closest by body shape, `Defer(Vec<Stmt>)`, has no name-list field at all — it just runs a body at scope exit, with nothing to look up. The closest by *name-list* shape is `ForStmt` (`vars: Vec<String>`, `body: Vec<Stmt>`, `line`, `col` — used for `for k, v in dict:`), which is exactly the field layout `WithStmt` needs too; it just isn't a `for` loop semantically, so it can't be reused as-is. This also matches the codebase's existing convention: `Guard`, `Defer`, `Try`, `KernelBlock` are each their own dedicated variant with their own minimal payload struct, rather than a shared generic "block with modifiers" shape — nothing currently plays that generic role, and inventing one just for this would be a bigger, less legible change than adding one more variant to a list that already has fifteen.

Deliberately *not* in `WithStmt`: each name's qualifier, and whether the block mutates it. Both are resolved later, by the checker/transpiler looking up each name's already-known binding and qualifier in scope — exactly how `def`/`req` method-call legality and index-assignment legality are already resolved without being baked into any `Stmt` payload. `WithStmt` only needs to know *which names*, and *what body*; the two-step hybrid rule from above is entirely a semantic-analysis concern, not a parse-time one.

#### Cross-target behavior

`'gpu'unified`/`'gpu'global` residency, and everything `with` does for them (map-for-read, map-for-write, write-back), is only meaningful where a real GPU buffer exists to be resident — wgpu/cuda/metal. Two other places the same source can be built or run needed an answer:

- **`boring run` (interpreter)**: checked directly rather than assumed — `GpuQual` (the enum distinguishing `'unified`/`'global`/`'const`/`'sync`/`'local`/`'actor'global`) is never referenced anywhere in `src/interpreter/`. The interpreter already treats every *kernel-context* qualifier uniformly, as a plain interpreted value, because a single-threaded sequential simulation has no host/device split to model in the first place. The same precedent extends cleanly to the new *host-context* qualifiers: under `boring run`, `'gpu'unified`/`'gpu'global` degrade to a plain array type, and `with fc:` is a pure no-op wrapper — it just runs the body directly, nothing to acquire or write back.
- **Plain `boring build` (no `--target`, the std/Rust backend)**: checked rather than assumed too — `TranspileConfig::gpu_kernels` defaults to empty (`transpiler/mod.rs`), and only the wgpu/cuda/metal entry points populate it before invoking the shared transpiler. A plain build never sees kernel declarations as anything other than ordinary structs, with their GPU qualifiers falling through `emit_top.rs`'s placeholder path (bare `*mut T`/`*const T` — see "The qualifier" above) — meaning `kernel Name:` code doesn't have a working code path under this target *today*, independent of this proposal. Since a `'gpu'unified`/`'gpu'global` value only ever comes from calling a kernel-dispatching function (`linear_gpu`, etc.), a program that uses either only makes sense where kernels themselves already work. For consistency, the same degrade-to-plain-array/no-op rule applies here as under the interpreter, rather than inventing a third, different behavior for a target combination that isn't really a going concern.

Net effect: `with` and the GPU host-context qualifiers have exactly one real backend (wgpu/cuda/metal) and one uniform fallback (no-op) everywhere else — no per-target special-casing beyond that split.

#### Async (`'task`) behavior

`'actor'task`/`'guard'task` need `with` to acquire `.lock().await`/`.read().await`/`.write().await` (tokio's async primitives) instead of the sync `.lock().unwrap()`/`.read().unwrap()`/`.write().unwrap()` that plain `'actor`/`'guard` use — but checking the transpiler rather than assuming turned up that this entire dispatch **already exists**, built for today's per-call locking, and `with` only needs to call into it:

```rust
// emit_top.rs — already there, used today for e.g. `c.increment()`
pub(crate) fn actor_task_write_guard(&self, expr: &str) -> String {
    match self.config.threading {
        ThreadingMode::Multi if self.in_async => format!("{}.lock().await", expr),
        ThreadingMode::Multi                  => format!("{}.lock().unwrap()", expr),
        ThreadingMode::Single                 => format!("{}.borrow_mut()", expr),
    }
}
```

Four functions already cover all of `'actor'task`'s read, `'actor'task`'s write, `'guard'task`'s read, and `'guard'task`'s write (`actor_task_read_access`, `actor_task_write_guard`, `guard_task_read_access`, `guard_task_write_guard`), each branching on the same two axes: single-vs-multi-threaded mode, and — in multi mode — whether the current function is async (`self.in_async`). Plain `'actor`/`'guard` (no `'task`) use the non-task siblings (`actor_write_guard`, etc.), which never emit `.await` at all, unconditionally. `with c:`'s lock-acquisition codegen calls whichever of these eight already-correct functions matches `c`'s qualifier and the detected read/write access — no new sync-vs-async logic needs writing.

Two things worth noting rather than assuming:

- **A `'task` value is not required to be inside an async function to be used.** The existing fallback (`Multi` non-async arm → `.lock().unwrap()`) means `'actor'task`/`'guard'task` degrade to a sync lock outside async code, rather than refusing to compile — looser than `gpu-module.md`'s "use `'actor'task` inside `task` functions" framing suggests. `with` inherits this exact behavior for free; it does not need its own restriction requiring an async context.
- **Holding the guard across further `.await` points inside the `with` block body is the intended use, not a risk to guard against.** That capability is the entire reason `'actor'task`/`'guard'task` use tokio's async-aware guards instead of `std::sync`'s (which are `!Send` across `.await` and couldn't do this at all) — `with` is, if anything, the first construct that lets a caller actually hold the lock across *multiple* statements including further awaits, more naturally than one-`.await`-per-call ever could.

The mutation-detection scan (the two-step hybrid rule above) is unaffected by any of this — it decides read-vs-write exactly the same way regardless of sync or async; only which of the eight guard-acquisition functions gets called changes.

One keyword, not two. Earlier drafts of this proposal used separate `read`/`write` keywords, requiring the block author to state up front which access level they needed. That was dropped in favor of a single `with` — but the access level it grants is **not** determined purely by how `<name>` was bound. A `mut`/`var`-bound value is *capable* of mutation, but a specific `with` block might only ever read it — forcing a write-back (or an exclusive lock) every time just because the binding happens to allow mutation would waste a real transfer on `'gpu'global` (always costly) and an unneeded exclusive lock on `'guard` (shrinking concurrency for no reason) on every block that merely inspects the value.

So the rule is a two-step hybrid:

1. **If `<name>` is `let`-bound, the block is read-only — no analysis needed.** Mutation is already a compile error today for a `let` array (`cannot assign to immutable variable`) or a `let T'actor`/`let T'guard` value (existing rule: `def` calls rejected, only `req`), so there is nothing to check; the answer is guaranteed by rules the language already enforces.
2. **If `<name>` is `mut`/`var`-bound, the compiler scans the block's own body** (recursing into `if`/`while`/`for` *within* the block, but not into the body of any function or method it calls — only their *signatures*) for any of:
   - a direct or index assignment targeting `<name>` (`<name> = ...`, `<name>[i] = ...`);
   - `<name>` passed as an argument at a parameter position the callee declares `var` (the callee's signature is enough — no need to look inside its body, exactly like `def`/`req` already tells you mutability without reading the method body);
   - a `def` (mutating) method called on `<name>`.

   If any of these occur, the block gets write access (map-for-read-write + write-back for GPU arrays, exclusive lock for `'guard`, `def` calls legal for `'actor`). If none occur, it gets read-only access (map-for-read only, no write-back, shared `RwLock::read()` for `'guard`) — even though the binding *could* have supported a mutation, this particular block didn't need one.

This bounded, local scan is the same kind of signature-only lookup the language already needs for other purposes (resolving whether a call passes into a `var` parameter) — it never has to open the body of whatever gets called inside the block, only check how that callee declared the relevant parameter or method.

#### Scan boundaries: aliasing and closures

Two cases needed a real answer: does the scan have to follow a second variable that aliases `<name>` (`let other = <name>`, then mutating `other`), and does passing `<name>` into a closure invoked later need special handling? Both turn out to be non-issues, for the same underlying reason — but getting there surfaces one design decision that wasn't explicit before.

**The reason both are safe**: the rule "host access to a `'gpu'unified`/`'gpu'global` value requires a `with` block wrapping *that name*, checked at the point of access" is a property of the *value's type*, enforced independently at every use site — not something tracked through data flow from the original declaration. So:

- **Aliasing.** `let other = <name>` produces a second binding of the *same* GPU-resident type. `other` is exactly as opaque outside a `with` block as `<name>` was — mutating `other[i] = v` directly, without its own `with other:`, is already a compile error under the existing rule (no new rule needed). If the author *does* write a nested `with other:` to mutate it, that inner block correctly detects the mutation and writes back — to the same underlying buffer, since it's an alias — at its own close, independent of whatever the outer `with fc:` block concluded about `fc`. No mutation is lost; it just gets written back by whichever block actually performed it, which is always a `with` block, by construction.
- **Closures.** The exact same argument applies regardless of *when* a closure capturing `<name>` runs. If it mutates the captured value, that mutation still has to happen inside some `with` block wrapping it — the closure's own body, wherever and whenever it executes — because the type-level rule doesn't stop applying just because the value got captured. A closure invoked synchronously, still inside the outer block, is caught by the scan recursing into it exactly like `if`/`while`/`for`; a closure stored and invoked later, after the outer block has already closed, needs (and, being a compile error otherwise, is forced to have) its own `with` block at that point.

Given this, the scan for `with <name>:` never needs interprocedural or alias tracking — it only ever needs to look at direct operations on the *literal names* listed in that block's own header, recursing into any construct — `if`/`while`/`for`/`match`/closures — lexically nested in the block's body, since a closure defined there is as much "in the block" as an `if` branch is.

**Design decision this surfaces**: for the "aliasing is safe" argument to hold, `let other = fc` must be a *cheap alias* (an `Arc::clone` of the buffer handle) — not a deep copy of the buffer's contents. That's not automatic; it's a choice this proposal is making explicitly, following the precedent already set by `'shared`/`'actor`/`'guard` ("assignment is an implicit alias — refcount increment, both bindings remain valid," `book.md`), rather than the deep-clone-on-rebind behavior plain arrays get today (confirmed in `emit_stmt.rs`'s `emit_let_value`: the general fallback path calls `emit_expr_owned`, a real `Vec` clone for ordinary arrays). Since the GPU-resident type's host-side representation is already `Arc<wgpu::Buffer>`-shaped (see "Kernel Constructor Interaction"), aliasing it *is* the cheap operation — cloning the `Arc`, not the buffer — so this isn't extra work, just a rule that needs to be stated rather than left to fall out of the generic-array codegen path by accident.

**One caveat worth documenting, not solving**: aliasing `'actor`/`'guard` specifically, then calling a mutating method on the alias *while the outer `with` block still holds the lock*, is a real self-deadlock risk — `Mutex`/`RwLock` aren't reentrant. This is a pre-existing category of footgun in any lock-holding code (Rust's own compiler doesn't catch it either), not something `with` introduces or that the scan should try to prevent — it widens the *window* during which it's possible (a lock held across a whole block rather than one call), but the hazard itself already exists today. Worth a line in the eventual user-facing docs, not a blocking design change.

Concretely, per qualifier:

- **`'gpu'unified`/`'gpu'global`, no mutation detected** (whether `let`-bound, or `mut`/`var`-bound but this block only reads): map-for-read only; no write-back on close.
- **`'gpu'unified`/`'gpu'global`, mutation detected** (only possible if `mut`/`var`-bound): map-for-read-write; automatic write-back (re-upload) on close.
- **`'guard`, no mutation detected**: `RwLock::read()` (shared — other readers may enter concurrently).
- **`'guard`, mutation detected**: `RwLock::write()` (exclusive).
- **`'actor`, `let`-bound**: `Mutex::lock()` still (no shared-read mode exists for `Mutex`), but only `req` calls are legal inside the block — matching the existing rule for a `let T'actor` binding today. Since a `let` binding can never contain a `def` call in the first place, there is nothing left to scan for.
- **`'actor`, `mut`/`var`-bound**: `Mutex::lock()`; `def` calls legal if the scan finds one, `req`-only otherwise (the lock itself is the same either way — `Mutex` has one mode — so this only affects which method calls type-check inside the block, not what gets acquired).

Other rules:

- Multiple names may be listed together (`with fc, act:`) when several values need to be inspected/mutated as one unit — each gets its own acquire/release, but the block reads as one critical section.
- Inside the block, `<name>` behaves exactly like the plain unqualified type would (`[float]` for a `'gpu'unified`/`'gpu'global` array, `T` behind `&T`/`&mut T` for `'guard`/`'actor`, gated by the table above). Outside the block, `<name>` is opaque to host operations: indexing, `.length`, iteration, string interpolation, or passing to a plain (non-kernel-construction) function are all **compile errors** with a message pointing at the missing `with` wrapper.
- Nesting a block on the **same** name inside itself is a compile error (double-acquire). Nesting on **different** names is unrestricted.
- A `'gpu'unified`/`'gpu'global` value that is never opened and only ever passed directly into another kernel's matching `'global`/`'unified` init parameter needs **no new syntax at all** — this is the default, zero-copy-at-the-Boring-source-level path this whole feature exists to make possible. The block is only needed at the point host code actually wants to look at or change the data.

### Example — implemented: materializing a kernel field once, in the same scope

> **Implemented and tested** — see `tests/wgpu_codegen.rs::test_with_gpu_resident_read_only_single_readback`/`test_with_gpu_resident_write_back_on_mutation`, and `examples/vector_add_gpu.br`/`matrix_mul_gpu.br`, which use exactly this pattern to fix a real, pre-existing round-trip-per-access bug in those examples' own print loops.

This is the shape that's actually built: a kernel constructed and its field read back *within the same function/scope* — the common case in every real kernel-using example in this repo.

```boring
var k = VectorAdd(host_a, host_b)
kernel:
    k(block = 256)

let [int]'gpu'unified result = k.result   # compile-time alias — no Rust binding, no transfer yet
with result:                              # `result` is `let`-bound -> read-only, no write-back
    for i in 0..n:
        print "c[{i}] = {result[i]}"      # readback happens once, here, however many times the loop indexes it
```

Before this existed, `for i in 0..n: print "c[{i}] = {k.result[i]}"` read the *entire* buffer back from the GPU on every one of the `n` iterations.

### Example — the motivating whisper-boring case (inter-procedural — still open)

> **Not implemented.** This example returns the resident value *across a function call boundary* (`linear_gpu`'s own return value) rather than reading a kernel field directly in the same scope as the example above — that needs the kernel-struct `Rc<RefCell<>>`/`Arc<Mutex<>>`-wrapping changes described in "Implementation Notes", not yet done. Today, `linear_gpu`'s real signature returns a plain `[float]` and its body's `k.y` still round-trips unconditionally on every call, exactly as in the "before" snippet below.

Before (current behavior, three round-trips):

```boring
let [float] fc  = linear_gpu(h3, mlp_fc_w, mlp_fc_b, 1, d, d * 4)   # dispatch + blocking readback
let [float] act = gelu_gpu(fc)                                      # re-upload, dispatch + blocking readback
let [float] pr  = linear_gpu(act, mlp_pr_w, mlp_pr_b, 1, d * 4, d)  # re-upload, dispatch + blocking readback
```

After (three dispatches, one readback — for whichever value the caller actually needs on the host):

```boring
let [float]'gpu fc  = linear_gpu(h3, mlp_fc_w, mlp_fc_b, 1, d, d * 4)   # dispatch, stays resident
let [float]'gpu act = gelu_gpu(fc)                                      # dispatch, stays resident (fc consumed directly)
let [float]'gpu pr  = linear_gpu(act, mlp_pr_w, mlp_pr_b, 1, d * 4, d)  # dispatch, stays resident

with pr:                       # `pr` is `let`-bound -> read-only, no write-back
    print "pr[0] = {pr[0]}"    # readback happens exactly here, once
```

If `pr` itself is only ever going to feed the *next* kernel in the caller (the common case — an MLP block's output goes straight into the residual add, which is a GPU op too), no block is written at all.

### Example — `'gpu'global`, always-transfer inspection

> **Not implemented** — inter-procedural (`load_embedding_matrix(...)` is a function call, not a same-scope kernel-field read), same status as the example above.

```boring
let [float]'gpu'global tok_emb = load_embedding_matrix(...)   # uploaded once, lives in device DRAM only

# ... many kernel dispatches use tok_emb directly as a 'global input, no host mirror ever kept ...

with tok_emb:
    print "tok_emb[0] = {tok_emb[0]}"   # a real D2H transfer happens here, unconditionally
```

Same syntax as the `'unified` case; the difference is entirely in what it costs, not in how it reads. `tok_emb` is `let`-bound here too, so this is read-only — no H2D write-back on close.

### Example — mutating a GPU array in place

> **Not implemented for this exact shape** — `gelu_gpu(fc)` is a function call (inter-procedural), same status as the two examples above. The write-back mechanics shown here (mutation scan → mandatory `mut` alias → `copy_..._to_device` on close) **are** implemented and tested for the same-scope case — see `test_with_gpu_resident_write_back_on_mutation`.

```boring
var [float]'gpu act = gelu_gpu(fc)   # `var` -> mutation is possible for this value

with act:
    act[0] = 0.0          # index-assignment detected in this block's body
    print act[0]
# write-back (H2D) happens automatically here, because a mutation was found
```

### Example — `var`-bound, but read-only *in this particular block*

> **Not implemented for this exact shape** (inter-procedural, same reason as above) — but the read/write scan itself is implemented and this exact behavior (no write-back when nothing in the block mutates) is verified both for the same-scope GPU case (`test_with_gpu_resident_read_only_single_readback`) and for `'actor`/`'guard` — see the next example and "Implementation Notes".

```boring
var [float]'gpu act = gelu_gpu(fc)   # `var` overall -- mutated elsewhere in the function, say

with act:
    print "act[0] = {act[0]}"   # only ever reads -- no assignment, no var-param call, no def call
# no write-back: the scan found nothing that mutates `act` in THIS block,
# even though the binding itself allows it
```

This is exactly the case a binding-only rule would get wrong — `act` being `var` doesn't mean *every* `with act:` block pays for a write-back, only the ones that actually mutate it.

### Example — `'actor`, wider critical section

> **Implemented and tested** — see `tests/cases/with_stmt.br` and `src/interpreter/tests/tests_part3.rs::test_with_actor_write`.

```boring
var c'actor = Counter()   # `var`, not `let` -> with grants def (mutating) calls

with c:
    c.value += 1
    c.value += 1
    print c.value    # all three operations under one lock acquisition
```

Existing per-call behavior (`c.increment()` locking and releasing on its own) is unchanged for code that does not use a block — this is purely additive.

## Typing Rules

Whether a given `with <name>:` block ends up read-only or write depends on both the binding and (for `mut`/`var`) a scan of the block's own body, per the two-step rule above:

| Qualifier | Binding | Detected access | Inside `with` | On close | Transfer/lock cost |
|---|---|---|---|---|---|
| `[T]'gpu'unified` | `let` | read-only (forced) | `[T]`, read-only | nothing to write back | free/cheap where the backend has real unified memory; a real staging copy on wgpu today |
| `[T]'gpu'unified` | `mut`/`var` | read-only (no mutation found) | `[T]`, read-only | nothing to write back | same as `let` |
| `[T]'gpu'unified` | `mut`/`var` | write (mutation found) | `[T]`, mutable | re-upload to the GPU buffer | same, plus the H2D leg |
| `[T]'gpu'global` | `let` | read-only (forced) | `[T]`, read-only | nothing to write back | D2H on entry — always a real transfer |
| `[T]'gpu'global` | `mut`/`var` | read-only (no mutation found) | `[T]`, read-only | nothing to write back | same as `let` |
| `[T]'gpu'global` | `mut`/`var` | write (mutation found) | `[T]`, mutable | H2D on exit | D2H on entry, H2D on exit |
| `T'guard` | `let` | read-only (forced) | `&T` | release `RwLock` read guard | in-process lock, no data transfer |
| `T'guard` | `mut`/`var` | read-only (no mutation found) | `&T` | release `RwLock` read guard | same as `let` |
| `T'guard` | `mut`/`var` | write (mutation found) | `&mut T` | release `RwLock` write guard | in-process lock, no data transfer |
| `T'actor` | `let` | read-only (forced) | `&T` (only `req` calls legal) | release `Mutex` guard | in-process lock, no data transfer |
| `T'actor` | `mut`/`var` | either | `&T` or `&mut T` depending on scan | release `Mutex` guard | same lock either way — `Mutex` has one mode, so this only gates which method calls type-check |

A value's qualifier and binding are unaffected by entering or leaving a block — `fc` is still whatever it was declared before, during, and after a `with fc:` block; the block only changes what operations are legal on it at that lexical point, and — for the GPU/`'guard` cases — what transfer or lock mode happens at the boundary.

## Kernel Constructor Interaction

> **Not implemented** — see "Implementation Notes". This needs the same inter-procedural resident-value machinery the whisper-boring chaining example does; today a resident alias's only legal use is materializing via `with`, not feeding into another kernel's constructor.

For this to actually eliminate round-trips, kernel constructors that take a `'global`/`'unified` init parameter must accept **either** a plain host array **or** an already-resident `'gpu'unified` value, and skip the upload in the latter case. Concretely, the generated Rust constructor branches on which the argument is:

```rust
// today: always uploads
fn new(x: &Vec<f64>, ...) -> Self { ... let x_buf = /* H2D copy of x */ ...; ... }

// proposed: reuse the buffer directly when the argument is already GPU-resident
fn new(x: BoringGpuArg<f64>, ...) -> Self {
    let x_buf = match x {
        BoringGpuArg::Resident(buf) => buf,          // no copy
        BoringGpuArg::Host(vec)     => /* H2D copy of vec, as today */,
    };
    ...
}
```

This is an implementation-level change to kernel constructor codegen (`wgpu/host.rs`'s `emit_kernel_new`, and the analogous cuda/metal paths), not something visible in Boring source — a `linear_gpu(fc, ...)` call looks identical whether `fc` is a plain `[float]`, a `[float]'gpu'unified`, or a `[float]'gpu'global`. The buffer-reuse path is identical for both GPU qualifiers; only what happens inside a `with` block on that same value differs.

## Open Questions

1. **Compiler lint for `'gpu'global` inside a hot loop** (accepted): since every `with` on `'gpu'global` is an unconditional real transfer (unlike `'unified`, which may eventually be free on some backends), a block on a `'global` value executed per-iteration in a tight loop is a likely performance footgun. A lint, not a hard error.

## Implementation Notes

`with` landed for the `'actor`/`'actor'task`/`'guard`/`'guard'task` side, end to end: AST (`Stmt::With`/`WithStmt`), parser, checker (opacity + double-acquire), interpreter (no-op), and transpiler codegen for every host target that shares the general `Transpiler`/`emit_stmt.rs` pipeline.

The `'gpu'unified`/`'gpu'global` residency side is implemented for the **intra-procedural** case — a kernel constructed and its field read back within the same function/scope, which turns out to be the shape *every* real kernel-using example in this repo actually uses (`examples/vector_add_gpu.br`'s `for i in 0..n: print k.result[i]`, `matrix_mul_gpu.br`'s equivalent — both re-read the whole buffer on every loop iteration before this landed). The **inter-procedural** case (a resident value returned across a function call boundary, e.g. whisper-boring's `linear_gpu`) is still open — see below for exactly why, and what it needs.

**What's real now:**

- `Type::gpu_resident_qual()` (`src/ast/mod.rs`) identifies `'gpu'unified`/`'gpu'global` at the outermost qualifier layer.
- The host-context placeholder is gone: `emit_top.rs`'s `OwnerQual::GpuUnified`/`GpuGlobal` used to emit a bare `*mut T` regardless of the initializing value's actual shape — a real bug, confirmed by transpiling `examples/saxpy.br` (`var [float]'gpu'unified x = [0.0 for ..N]`, freely indexed/assigned) for `--target wgpu` and watching `rustc` reject `let mut x: *mut Vec<f64> = vec![0.0; N as usize];` outright (E0308/E0599/E0608). It now emits the plain inner type — matching what every existing example already assumed a `'gpu'unified`/`'gpu'global` array *is*: an ordinary host `Vec`, right up until it's consumed by a kernel constructor (upload happens there, unrelated to `with`) or read from a kernel field.
- **`Binding::resident_from_field`** (checker): a `'gpu'unified`/`'gpu'global` binding is only actually opaque-outside-`with` when its initializer is syntactically a bare `k.field` read (`ExprKind::Field(Var(_), _)`) — a plain array literal/expression is unrestricted. This is what keeps `saxpy.br`'s pattern legal while still gating the genuinely-resident case; the check is purely syntactic (the checker never needs to know which names are real kernel instances).
- **`gpu_resident_vars: HashMap<name, (kernel_var, field)>`** (transpiler): `emit_kernel::try_emit_gpu_resident_let` recognizes `let py'gpu'unified = k.y` (`kvar` must be a tracked kernel var — `self.kernel_vars`) and registers it as a **pure compile-time alias** — no Rust binding is ever emitted for `py`. Its only legal use is as the subject of `with`.
- **The `'gpu'unified`/`'gpu'global` annotation is inferred, not required.** `let py = k.y` (no qualifier at all) behaves identically to the explicit form: both the checker (`Checker::infer_gpu_resident`, using a new `kernel_decls`/`Binding::kernel_type` pre-pass it didn't have before) and the transpiler (`try_emit_gpu_resident_let`'s untyped branch) recognize the same shape — a bare `k.field` read where `k` is a known kernel instance and `field` is actually declared `'unified`/`'global` on an array — and apply the exact same rules. An untyped read of a scalar or differently-qualified field is untouched (falls through to an ordinary field read, same as before this existed).
- **`emit_stmt::emit_with`** resolves a `gpu_resident_vars` name back to `k.copy_{field}_to_host()`/`copy_{field}_to_device()` — the exact same conversion `emit_kernel::try_emit_kernel_field_read` already used for a bare `k.y` — exactly once per block, regardless of how many times the body indexes it. Write-back only happens if the shared mutation scan (`ast::with_block_mutates`) finds an index-assignment into the alias.
- Verified three ways: (1) `tests/wgpu_codegen.rs::test_with_gpu_resident_read_only_single_readback`/`test_with_gpu_resident_write_back_on_mutation` — exact codegen-shape snapshot tests (exactly one `copy_field_to_host`/`copy_field_to_device` call, correct `mut`, no leftover `*mut`); (2) cross-checked against the interpreter on the same source (identical output, since the interpreter treats it as a plain passthrough); (3) `examples/vector_add_gpu.br`/`matrix_mul_gpu.br` updated to use this pattern for their own print loops, confirmed identical output to before via the interpreter.
- **Not independently verified against real GPU hardware for correct numeric output.** This machine has only integrated graphics (Intel UHD, no discrete GPU) and the wgpu backend has several pre-existing, unrelated correctness gaps that block a clean end-to-end run today, found while trying: `GPU(0)` device-info API has no wgpu implementation; an array-comprehension emits `Vec<i64>` where an `isize`-typed binding expects `Vec<isize>` (E0308); the checked-in `examples/saxpy_wgpu` snapshot fails to build against the current `naga`/`wgpu` versions (`Identifier starts with a reserved prefix: '__params'`); a from-scratch Saxpy kernel (`x`/`y` both `'unified`, uploaded post-construction via separate `copy_x_to_device`/`copy_y_to_device` calls) dispatches and compiles cleanly but returns each element's *initial* value rather than the computed one, on real hardware — confirmed to reproduce identically with a **plain `k.y[i]` control (no `with`, no alias, nothing this session touched)**, so it predates and is unrelated to this feature. None of these four are `with`-related; fixing them is separate work.

**Still open — the inter-procedural case** (a resident value crossing a function-return boundary, e.g. `linear_gpu`):

1. **No existing `'gpu'unified`/`'gpu'global`-qualified *return value* anywhere in real code.** `whisper-boring/src/math_gpu.br`'s actual `linear_gpu` returns a plain `[float]`, tail-expression `k.y` — a kernel-field read that always round-trips via `try_emit_kernel_field_read`. Making `let fc'gpu'unified = linear_gpu(...)` real means changing `linear_gpu`'s own signature and body-emission (skip the eager readback when the *caller* wants residency) plus a new interprocedural `fn name -> (kernel type, field name)` table — a fundamentally different mechanism from the same-scope alias above, which needs no interprocedural bookkeeping at all.
2. **Kernel struct buffer fields are not `Arc`-shaped.** `wgpu::host::emit_kernel_new`/`emit_kernel_struct` store every `'unified`/`'global` field as a bare `wgpu::Buffer` owned directly by the kernel struct, with a `bind_group` built once (and rebuilt on resize) referencing those buffers by value. A resident value that must outlive the function that constructed its kernel needs the kernel instance itself to stay reachable from both the return site and wherever it's later used, which means `Rc<RefCell<KernelStruct>>` (single-thread) / `Arc<Mutex<KernelStruct>>` (multi-thread) wrapping — a real change to kernel-struct storage, needed only for kernel instances actually returned as a `'gpu`-resident value, not the direct `kernel:`-block-in-`main()` usage every existing example/test (and the same-scope alias above) already relies on unchanged.
3. **`cuda`/`metal` targets don't share the general pipeline `with` was implemented against**, for either half. Checked `src/transpiler/cuda/host.rs`/`metal/host.rs`: unlike wgpu (whose `mod.rs` runs the *same* `Transpiler`/`emit_stmt.rs` as `boring build` with no target, then splices its output in — see `transpile_wgpu`'s `general_out`), cuda and metal have their own separate, much smaller, kernel-only host transpilers with no `OwnerQual::Actor`/`Guard` support at all and a catch-all `_ => "/* unsupported stmt */"` that silently swallows `with`. Getting either half of `with` working there is blocked on first giving those backends the general-purpose statement/expression support wgpu already gets by reusing the shared pipeline — a gap that predates this feature.

Net effect: the inter-procedural GPU-residency case remains a strictly bigger unit of work — (a) a return-type/body-emission change to every `'gpu`-returning function, (b) a new interprocedural fn→(kernel, field) table, (c) a conditional `Rc`/`Arc`-wrapping change to kernel-struct storage — and, for cuda/metal, (d) building general-purpose host codegen those backends don't have yet at all. Recommend fixing the four unrelated wgpu correctness gaps above first (on real discrete-GPU hardware, which this session didn't have), so the *already-implemented* same-scope case can be confirmed correct at runtime, before investing in (a)-(c).

## Summary

| Concern | Solution |
|---|---|
| GPU kernel chaining forces a host round-trip per call | **Implemented for the same-scope case**: `let py'gpu'unified = k.y` is a compile-time alias (no Rust binding); a `with` block materializes it exactly once via `k.copy_y_to_host()`, however many times the body indexes it. The inter-procedural case (a resident value returned across a function boundary) is still open — see "Implementation Notes" |
| `'actor`/`'guard` lock per call, no multi-statement critical section | Same `with` block, reused: acquires once, releases on block exit |
| Implicit lazy materialization (rejected alternative) | Requires hidden mutable state behind `let`, and dangerous clone-at-call-site semantics for GPU buffers — explicit blocks avoid both |
| AST shape for `with` | New dedicated `Stmt::With(WithStmt { names: Vec<String>, body: Vec<Stmt>, .. })` — no existing variant fits (`Defer` has no name-list; `ForStmt`'s `vars`/`body` shape is the right field layout but the wrong construct); qualifier and mutation-detection are resolved later, by the checker, not baked into the AST |
| Cross-target behavior (`boring run`, plain `boring build`) | Verified the interpreter never references `GpuQual` at all — every kernel-context qualifier is already just a plain value there. Same rule extends to the new host-context qualifiers: `'gpu'unified`/`'gpu'global` degrade to a plain array, `with` becomes a no-op wrapper, under both the interpreter and a plain (non-GPU-target) `boring build` — the latter never had a working kernel code path to begin with (`gpu_kernels` defaults empty outside wgpu/cuda/metal) |
| `'actor'task`/`'guard'task` async locking | Already fully implemented for today's per-call locking (`actor_task_write_guard` and 3 siblings in `emit_top.rs`) — branches on threading mode and `self.in_async`, falling back to a sync lock outside async code rather than requiring one. `with` calls the same 8 existing functions; no new sync-vs-async logic needed, and holding the guard across further `.await`s in the block body is the intended use, not a new risk |
| Read vs. write access level (rejected: two keywords) | `let` binding → read-only, free, no analysis. `mut`/`var` binding → the compiler scans the block's own body (assignment, `var`-parameter calls, `def`-method calls, signatures only, never callee bodies) to grant read-only or write per block — so a `var` value read in one block doesn't pay for a write-back it never needed |
| Scan boundaries: aliasing (`let other = <name>`), closures | Both a non-issue: host access is gated by the value's *type*, checked at every use site, not tracked through data flow — an alias or a captured closure needs its own `with` block to do anything, and that block correctly handles its own write-back whenever it runs. Surfaces one explicit decision: `'gpu'unified`/`'gpu'global` rebinding must be a cheap `Arc`-alias (like `'actor`/`'guard`/`'shared`), not the deep clone plain arrays get on rebind today |
| Kernel constructors uploading data that's already GPU-resident | **Not implemented** — the `BoringGpuArg` dual-mode acceptance sketched in "Kernel Constructor Interaction" needs the inter-procedural resident-value machinery (see "Implementation Notes"); today a resident alias's only legal use is materializing via `with`, not feeding directly into another kernel's constructor |
| `'global`'s documented `gpu.copy()` mechanism | Never actually implemented (no parser/checker/interpreter/transpiler support found) — dropped in favor of `with`, not kept alongside it; docs referencing it (`gpu-module.md`, `wgpu-backend.md`, `cuda-module.md`) need updating when this lands |
| `'unified` vs `'global` cost | Same block syntax, different transfer guarantee: `'unified` free/cheap where the backend supports real unified memory (a real copy on wgpu today); `'global` is always a real transfer, by definition, on every backend |
| `'sync` / `'local` / `pub` on kernel fields | Out of scope, by design, not by gap: both qualifiers have **no** host access at all per `gpu-module.md`; a kernel field's visibility is already expressed through its qualifier, which is why `pub` is disallowed on kernel fields entirely rather than needing its own rule |
| `'gpu'global` in a hot loop | Every `with` on it is an unconditional real transfer — accepted as worth a compiler lint flagging repeated use inside a loop, not a hard error |
