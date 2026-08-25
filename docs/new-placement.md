# Boring — `new` placement operator

> Status: syntax and binding/placement semantics implemented. Transpilation is partial — see "Transpilation" section below for current vs. planned behavior. Integrated into `book.md`.

---

## Motivation

Boring's qualifier inference works from usage signals. `new` is an additive operator that makes the placement intent explicit at the construction site, without requiring a qualifier to be named. It also enables GPU device placement, which has no equivalent in the existing syntax.

Existing qualifier syntax is unchanged and complemented, not replaced.

---

## Binding × placement — complete table

### Initialized bindings

```boring
let v = Counter()             # inferred — 'inline included in candidates
let v = new Counter()         # inferred — 'inline excluded from candidates
let v'actor = Counter()       # explicit qualifier
```

`let v'new = new Counter()` is redundant — `new` on the right already excludes `'inline`. `'new` does not exist as a qualifier on an initialized binding.

### Delayed init (no `=`)

```boring
let Counter v                 # inferred — 'inline included in candidates
let Counter'new v             # inferred — 'inline excluded from candidates
let Counter'actor v           # explicit qualifier
```

`'new` is a pseudo-qualifier meaning "inferred, excluding `'inline`" — the mirror of `new` on the right-hand side. It carries no Rust representation of its own; it only constrains the inference starting set (`Union([Owned, Shared, Actor, Guard])`), and the transpiler resolves it to a concrete qualifier the same way it resolves a bare `T`.

### Symmetry

| Intent | Initialized | Delayed init |
|---|---|---|
| Inferred, `'inline` included | `let v = Counter()` | `let Counter v` |
| Inferred, `'inline` excluded | `let v = new Counter()` | `let Counter'new v` |
| Explicit qualifier | `let v'actor = Counter()` | `let Counter'actor v` |

---

## API

`new` has two overloads. `Constructor<T>` is a compile-time token — the constructor call passed as the last argument.

```boring
def T new(Constructor<T>)               # qualifier inferred, excluding 'inline
def T new(Arena& arena, Constructor<T>) # GPU placement, CPU qualifier from binding
```

There is no `Arena` trait in the implementation. `GPU(n)` is a built-in runtime device handle (recognized ad hoc by the interpreter and by the CUDA/Metal backends); there is currently no generic/trait-based mechanism a user type could implement to become a valid `new(...)` placement target. The `Arena&` signature above describes the intended API shape, not an implemented constraint.

### Examples

```boring
new Counter()       # qualifier inferred, excluding 'inline
new(g0) Scale(n)    # GPU device g0 — only works when Scale is a `kernel`-declared type
```

> Note: `new(arena) X(...)` only produces device placement when `X` is declared with `kernel`. For an ordinary `struct` such as `Counter`, the arena argument is currently ignored and `new(g0) Counter()` transpiles identically to `Counter()`, with no placement effect.

---

## Decisions

### `new` is additive

`new` does not replace existing qualifier syntax. It adds two capabilities: inference excluding `'inline`, and GPU arena placement.

### `Counter'` (bare tick) removed

`Counter'` without a qualifier name carried no inference signal — the right-hand side and usage context already determine the qualifier. It is replaced by `Counter'new` for the delayed-init case where `'inline` should be excluded.

### `v'qualifier` kept for initialized bindings only

`v'qualifier` on an initialized binding is the concise form for explicit qualifiers when the type is obvious from the constructor. `Type'qualifier` is reserved for delayed init, where the type cannot be inferred.


### Named arenas rejected

`let a = arena(g0, 'owned)` then `new(a) Scale(n)` was considered and **rejected** — storing an `Arena` in a variable would make placement runtime-dependent, preventing Rust type emission.

### Future qualifiers

`new(arena)` is the extension point for CUDA-specific qualifiers (`'gpu.*` CPU-side, device-side qualifiers). These are specified in the CUDA module reference.

---

## Transpilation

Current implementation status (not yet the design above):

- Generic backend: `new(...)` is not yet emitted specially — the arena is discarded and the constructor call is emitted as-is (`src/transpiler/emit_expr.rs`, `ExprKind::New`).
- CUDA/Metal host backends: `new(arena) X(...)` is rewritten to `X::new(<device-expr>, args...)?` only when `X` is a registered `kernel` type (`src/transpiler/cuda/host.rs`, `src/transpiler/metal/host.rs`). This is a hand-written codegen rule, not a generic allocator mechanism. For any non-kernel type the arena is ignored and the plain constructor call is emitted.
- Interpreter: arena placement has no runtime effect; `new(...)` evaluates the constructor normally (`src/interpreter/eval_expr.rs`).

There is no use of Rust's `Allocator` trait or parameterised smart pointers (`Box<T, A>`, `Vec<T, A>`) anywhere in the codebase. A compile-time, allocator-trait-based mechanism is the long-term direction but is **not yet implemented** — treat it as planned, not current behavior.
