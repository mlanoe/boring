# Boring — `new` placement operator — design draft

> Status: implemented. Integrated into `book.md`.

---

## Motivation

Boring's qualifier inference works from usage signals. `new` is an additive operator that makes the placement intent explicit at the construction site, without requiring a qualifier to be named. It also enables GPU device placement, which has no equivalent in the existing syntax.

Existing qualifier syntax is unchanged and complemented, not replaced.

---

## Binding × placement — complete table

### Initialized bindings

```boring
let v = Counter()             # inferred — 'stack included in candidates
let v = new Counter()         # inferred — 'stack excluded from candidates
let v'actor = Counter()       # explicit qualifier
```

`let v'new = new Counter()` is redundant — `new` on the right already excludes `'stack`. `'new` does not exist as a qualifier on an initialized binding.

### Delayed init (no `=`)

```boring
let Counter v                 # inferred — 'stack included in candidates
let Counter'new v             # inferred — 'stack excluded from candidates
let Counter'actor v           # explicit qualifier
```

`'new` is a pseudo-qualifier meaning "inferred, excluding `'stack`" — the mirror of `new` on the right-hand side. It carries no Rust representation; it only constrains the inference starting set.

### Symmetry

| Intent | Initialized | Delayed init |
|---|---|---|
| Inferred, `'stack` included | `let v = Counter()` | `let Counter v` |
| Inferred, `'stack` excluded | `let v = new Counter()` | `let Counter'new v` |
| Explicit qualifier | `let v'actor = Counter()` | `let Counter'actor v` |

---

## API

`new` has two overloads. `Constructor<T>` is a compile-time token — the constructor call passed as the last argument.

```boring
def T new(Constructor<T>)               # qualifier inferred, excluding 'stack
def T new(Arena& arena, Constructor<T>) # GPU placement, CPU qualifier encoded in arena
```

`GPU` implements `Arena` — a device value is a valid placement context. `Arena` is a compile-time constraint, not a runtime value.

### Examples

```boring
new Counter()             # qualifier inferred, excluding 'stack
new(g0) Counter()         # GPU device g0, CPU side inferred
new(g0('heap)) Counter()  # GPU device g0, CPU side explicit 'heap
```

The CPU-side qualifier for GPU placement is passed to the arena expression. `GPU` only accepts `'stack` or `'heap` — shared qualifiers (`'actor`, `'guard`, `'shared`) are a compile error at the `g0(...)` call site.

```boring
g0           # CPU side inferred ('stack)
g0('heap)    # CPU side explicit 'heap
g0('actor)   # ERROR — GPU arena only accepts 'stack or 'heap
```

---

## Decisions

### `new` is additive

`new` does not replace existing qualifier syntax. It adds two capabilities: inference excluding `'stack`, and GPU arena placement.

### `Counter'` (bare tick) removed

`Counter'` without a qualifier name carried no inference signal — the right-hand side and usage context already determine the qualifier. It is replaced by `Counter'new` for the delayed-init case where `'stack` should be excluded.

### `v'qualifier` kept for initialized bindings only

`v'qualifier` on an initialized binding is the concise form for explicit qualifiers when the type is obvious from the constructor. `Type'qualifier` is reserved for delayed init, where the type cannot be inferred.

### Named arenas rejected

`let a = arena(g0, 'heap)` then `new(a) Scale(n)` was considered and **rejected** — storing an `Arena` in a variable would make placement runtime-dependent, preventing Rust type emission.

### Future qualifiers

`new(arena)` is the extension point for CUDA-specific qualifiers (`'gpu.*` CPU-side, device-side qualifiers). These are specified in the CUDA draft.

---

## Transpilation

`new(...)` is resolved entirely at compile time. It maps to Rust's `Allocator` trait and parameterised smart pointers (`Box<T, A>`, `Vec<T, A>`). No runtime dispatch — the placement strategy is a type-level constraint, erased after monomorphisation.
