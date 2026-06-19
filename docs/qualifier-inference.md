# Qualifier Inference

Boring's ownership qualifiers (`'stack`, `'heap`, `'shared`, `'actor`, `'guard`, `'const`) describe how a value is stored and shared at runtime. In most programs you never write them — the transpiler infers the right one from how each variable is used.

The zero-annotation goal: qualifier-free Boring code emits the same Rust as hand-annotated code. This document explains the full inference algorithm.

---

## Constraint elimination

Each unqualified local variable starts with a candidate set of all possible qualifiers. Every usage signal narrows the set by eliminating incompatible qualifiers. When exactly one candidate remains it is chosen. When none remain the constraints are contradictory and a compile error is reported. When several remain a size-based fallback resolves the tie.

### Candidate sets

| Declaration form | Initial candidate set | Fallback (multiple remaining) |
|---|---|---|
| `T` (bare, no qualifier) | `{Stack, Owned, Shared, Actor, Guard, Const}` | priority-ordered fallback (see below) |
| `T'` (tick, indirection hint) | `{Owned, Shared, Actor, Guard}` | `'heap` (`Box<T>`) |
| `T?` (optional, bare) | `{Stack, Owned, Shared, Actor, Guard, Const}` | same priority-ordered fallback |
| `T'?` (optional tick) | `{Owned, Shared, Actor, Guard}` | `Option<Box<T>>` |

`T'` and `T'?` restrict the initial set to indirection qualifiers. For optional forms, the inferred qualifier is applied to the **inner type** of the `Option` — `T?` with inferred `'actor` emits `Option<Arc<Mutex<T>>>`, not `Arc<Mutex<Option<T>>>`.

### Signal table

| Signal | Compatible qualifiers |
|---|---|
| Call site demanding `T'shared` | `{Shared}` |
| Call site demanding `T'actor` | `{Actor}` |
| Call site demanding `T'guard` | `{Guard}` |
| Call site demanding `T'stack` | `{Stack}` |
| Call site demanding `T'heap` | `{Owned}` |
| `def` method call on the variable | `{Stack, Owned, Actor, Guard}` |
| `mut` binding (`mut x = …`) | `{Stack, Owned, Actor, Guard}` |
| `var` binding reassigned (`x = …`, `x.field = …`, `x.a.b.c = …`, `x[i] = …`) | `{Stack, Owned, Actor, Guard}` |
| Closure capturing `x` as method receiver | `{Actor, Guard}` |
| Closure capturing `x` read-only | `{Shared, Actor, Guard}` |
| `set` property setter body | `{Stack, Owned, Actor, Guard}` |
| Task capture as method receiver | `{Actor, Guard}` |
| Task capture, read-only | `{Shared, Actor, Guard}` |
| `req` method call | *(no constraint — all qualifiers remain)* |

Each signal intersects the current candidate set. The order of signals does not matter.

### Priority-ordered fallback

When the candidate set still contains multiple qualifiers after all signals are applied, the transpiler resolves the tie using the following ordered algorithm.

#### Step 1 — `'const` candidate

`'const` is the read-only counterpart of `'stack`, in the same way `'shared` is the read-only counterpart of `'actor`. It emits as `&'static T` and is therefore only valid for non-struct types (primitives, expressions). When `'const` is in the candidate set:

| Context | Decision |
|---|---|
| Struct field, or known struct type | skip to Step 2 |
| Non-struct local, sizeof(T) ≤ `--stack-auto-bytes` | `'const` |
| Non-struct local, sizeof(T) > `--stack-auto-bytes` | skip BOTH `'const` AND `'stack` (they share the same size criteria); continue with the ordered chain |

Because `mut` and `var` signals eliminate `Const` from the candidate set, when `'const` survives to fallback the binding is implicitly read-only.

#### Step 2 — `'stack` candidate

If `'stack` is in the candidate set:

| Context | Decision |
|---|---|
| Struct field (any binding) | `'stack` — field bytes are part of the parent allocation; no indirection regardless of size |
| Local variable, sizeof(T) ≤ `--stack-auto-bytes` | `'stack` |
| Local variable, sizeof(T) > `--stack-auto-bytes` | skip `'stack`; continue to step 3 with the remaining candidates |

All bare-T struct fields suppress size-based auto-boxing. This applies to `var` fields as well as `let`/`mut` fields — a `var` field is still stored in-place in the struct, and boxing it would add unnecessary indirection. `T'` fields are not affected: an indirection hint always produces `Box<T>`.

#### Step 3 — ordered chain

If neither `'const` nor `'stack` was selected, the transpiler picks the first qualifier from the remaining candidate set according to this chain:

`'heap` > `'shared` > `'actor` > `'guard`

#### Size threshold

The threshold is configurable: `boring build --stack-auto-bytes 512` (default: 256 bytes).

> The size estimate is best-effort: it sums struct fields recursively but treats `Vec`, `HashMap`, and pointer-sized types as 8–16 bytes. When in doubt the transpiler prefers `'heap` over a potentially large stack frame.

---

## Examples

### Sharing demand → `'shared`

```boring
let c = Counter(0)
share_read(c)           # expects Counter'shared → c infers 'shared
```

```rust
let c = Arc::new(Counter::new(0));
share_read(c);
```

### Mutation demand → size fallback

```boring
let c = Counter(0)
c.inc()                 # def call → {Stack, Owned, Actor, Guard}
                        # no further signal → size fallback → 'stack (if small)
```

### Sharing + mutation → conflict error

```boring
let c = Counter(0)
c.inc()                 # def call → {Stack, Owned, Actor, Guard}
share_read(c)           # 'shared → intersect → {}  → ERROR
```

```
error: `c` has no valid qualifier — usage constraints are incompatible
  fix: annotate `c` explicitly
```

`'shared` requires an immutable `Arc<T>`; `def` requires direct mutation. No qualifier satisfies both in Rust.

### Return-type demand

```boring
Counter'actor make():
    Counter(0)          # tail expression inherits return qualifier → 'actor
```

### Alias propagation

```boring
let a = Counter(0)
let b = a               # b is an alias of a — same qualifier group
spawn_actor(b)          # demands 'actor → both a and b infer 'actor
```

### Task capture

```boring
let c = Counter(0)
task:
    c.inc()             # capture as receiver → {Actor, Guard}
                        # if no other signal: fallback between Actor/Guard → explicit annotation needed
```

```boring
let c = Counter(0)
task:
    c.inc()             # {Actor, Guard}
spawn_actor(c)          # demands 'actor → {Actor} → inferred
```

### `mut` binding as early signal

```boring
mut c = Counter(0)      # mut → {Stack, Owned, Actor, Guard} immediately
spawn_actor(c)          # demands 'actor → {Actor} → inferred
```

### `T'` with inference

```boring
let c' = Counter(0)     # tick → initial set: {Owned, Shared, Actor, Guard}
spawn_actor(c)          # demands 'actor → {Actor} → inferred, emits Arc<Mutex<Counter>>
```

```boring
let c' = Counter(0)     # tick → {Owned, Shared, Actor, Guard}
                        # no further signal → fallback → 'heap → Box<Counter>
```

### Optional with inference

The qualifier is applied to the inner type of the `Option`, not to the `Option` itself.

```boring
let c? = some(Counter(0))   # T? → full candidate set
spawn_actor(c?)             # demands Counter'actor → infers 'actor
# emits: let c: Option<Arc<Mutex<Counter>>> = Some(Arc::new(Mutex::new(Counter::new())));
```

```boring
let c'? = some(Counter(0))  # T'? → restricted set {Owned, Shared, Actor, Guard}
                             # no signal → fallback → Option<Box<Counter>>
```

A conflict on an optional variable is reported the same way as for a bare variable — the Optional wrapper is transparent to the constraint-elimination algorithm.

---

## Universal borrow as inference output

When a parameter has no explicit qualifier and no storage signals, the inference can resolve to a universal borrow (`Counter&` or `mut Counter&`) rather than a concrete qualifier. This is evaluated as a pre-fallback step, before the priority-ordered chain.

### Mutability is declared, not inferred

The `mut` keyword on a parameter is **not inferred** — it must be written explicitly by the developer. This is consistent with the rest of the language (`let`/`mut`/`var`, `req`/`def`, `def mut T`): mutability is a contract visible in the signature, not an implementation detail the transpiler resolves from the body.

- `Counter n` — read-only parameter. The transpiler infers `Counter&` when `n` is not stored.
- `mut Counter n` — mutable parameter. The transpiler infers `mut Counter&` when `n` is not stored.

A `def` method call on `Counter n` (without `mut`) is a compile error:

```
error: parameter `n` is immutable but `inc` is a `def` method — declare `mut Counter n`
```

### Conditions for universal borrow inference

A parameter resolves to a universal borrow if it triggers neither a **storage signal** nor a **qualifier demand signal**.

**Storage signals** — `n` is never:
- assigned to a struct field (`self.x = n`, `field = n`)
- returned with an ownership qualifier
- captured by a closure or task
- destructured through a field access (`guard let Some(x) = n.field`, `if let Some(x) = n.field`, `let x = n.field`)

**Qualifier demand signals** — `n` is never:
- passed to a function parameter with an explicit qualifier (e.g. `Counter'actor`)

Passes to `Counter&` or `mut Counter&` parameters do not count as qualifier demand signals — they are borrow-compatible and leave the inference open.

| Declaration | Signals | Inferred form | Rust emitted |
|---|---|---|---|
| `Counter n` | none | `Counter&` | `&Counter` |
| `mut Counter n` | none | `mut Counter&` | `&mut Counter` |
| `Counter n` | qualifier demand | qualifier via constraints | — |
| `mut Counter n` | qualifier demand | mutable qualifier via constraints | — |
| `Counter n` | storage | qualifier via constraints | — |
| `mut Counter n` | storage | mutable qualifier via constraints | — |

`'shared` is excluded from `mut Counter n` by construction — it cannot produce `&mut Counter`.

**Not applicable to:** `Counter' n` (tick — stays on the indirection path), `Counter? n` (optional — excluded from auto-ref inference), `var Counter n` (out-parameter), explicit qualifier groups (`Counter'mut n`), and **struct or enum method parameters** (see below).

### Examples

```boring
req display(Counter n):
    print n.value
# no storage, read-only → infers Counter& → fn display(n: &Counter)
# caller can pass any qualifier: 'stack, 'heap, 'shared, 'actor, 'guard
```

```boring
def reset(mut Counter n):
    n.value = 0
# no storage, mutable → infers mut Counter& → fn reset(n: &mut Counter)
# caller can pass any mutable qualifier: 'stack, 'heap, 'actor, 'guard
# 'shared is rejected at the call site
```

```boring
def reset(Counter n):
    n.value = 0         # def call on immutable parameter → compile error
# error: parameter `n` is immutable but `value` is mutated — declare `mut Counter n`
```

```boring
def process(mut Counter n):
    n.inc()              # def call — ok, mut declared
    spawn_actor(n)       # storage signal: demands 'actor
# storage signal wins → infers 'actor, not mut Counter&
# 'shared excluded because mut
```

### Lock acquisition at call sites

When a parameter infers `Counter&` or `mut Counter&` and the caller passes a `'actor` or `'guard` argument, the transpiler must acquire the lock at the call site — identical to the behavior of an explicit `Counter&` or `mut Counter&` parameter.

```boring
req display(Counter n):   # infers Counter&
    print n.value

let b = Counter'actor(0)
display(b)
# emits: { let __g = b.lock()?; display(&*__g); }
```

```boring
def reset(mut Counter n):   # infers mut Counter&
    n.value = 0

let b = Counter'actor(0)
reset(b)
# emits: { let mut __g = b.lock()?; reset(&mut *__g); }
```

For `'guard`, `Counter&` uses `read()` (shared read lock) and `mut Counter&` uses `write()` (exclusive write lock).

This is the same code generation as explicit `Counter&` / `mut Counter&` — the inferred and explicit forms are identical at the Rust level. What `fn_sigs` records as `Counter&` (inferred) is treated exactly like `Counter&` (explicit) at every call site.

### Interaction with qualifier groups

If the parameter has an explicit qualifier group (`Counter'mut n`, `Counter'req n`), the universal borrow inference does not apply — the group is the declared constraint and the body narrows within it. Universal borrow inference applies only to bare `Counter n`.

### Generics

Universal borrow inference applies to generic parameters the same way as concrete types. `T n` with no storage signal and no qualifier demand resolves to `T&`; `mut T n` resolves to `mut T&`.

```boring
req display(T n):       # no storage, no qualifier demand
    print n.value
# infers T& → fn display<T>(n: &T)
# caller can pass any qualifier
```

```boring
def reset(mut T n):     # no storage, mut declared
    n.value = 0
# infers mut T& → fn reset<T>(n: &mut T)
```

The lock acquisition rule at call sites applies identically: if the concrete type instantiated for `T` is `'actor` or `'guard`, the transpiler inserts the lock at the call site.

### Struct and enum method parameters

Universal borrow inference is **disabled** for parameters of `req` and `def` methods defined on a struct or enum. The call-site coercion that injects `&` or `&mut` is only available for free functions — method calls (`p.distance(q)`) are not rewritten by the transpiler, so the argument `q` would never receive the implicit `&`. Applying auto-ref to method params would cause a type mismatch between the emitted parameter type (`&Counter`) and the unannotated argument at every call site.

```boring
struct Point:
    float x
    float y

    req distance(Point other):   # auto-ref NOT applied — other stays Point, not &Point
        ...
```

If a method parameter needs universal borrowing, use the explicit `Counter& n` form.

### Explicit form

`Counter& n` and `mut Counter& n` remain valid as explicit forms. Use them to:
- Document intent — make universal borrowing visible in the signature.
- Lock in the behavior regardless of future body changes that might introduce a storage signal.
- Enable universal borrowing in struct/enum method parameters (where inference is disabled).

When a bare `Counter n` inference resolves to `Counter&`, the emitted signature is identical to an explicit `Counter& n`.

---

## Parameter auto-apply

A pre-inference pass runs `infer_qualifiers` on the function body before emitting parameters. `emit_param` then consults `inferred_qualifiers`: if the parameter has no explicit qualifier but inference resolved one, it is applied at emission automatically.

```boring
def process(Counter c):   # no qualifier written
    spawn_actor(c)        # demands 'actor → inferred for c
# emits: fn process(c: Arc<Mutex<Counter>>)
```

The developer never writes the qualifier; the emitted Rust carries the correct wrapped type.

`T'` parameters also benefit from auto-apply: if inference resolves the tick parameter to a specific qualifier, that qualifier replaces the default `Box<T>`.

---

## Cross-function propagation

After each function body is emitted, `fn_sigs` is updated with the inferred parameter qualifiers. Functions defined later in the file that call this function see the qualified signature and propagate the constraint to their own anonymous variables.

```boring
def process(Counter c):   # inferred 'actor from body → fn_sigs updated
    spawn_actor(c)

let c = Counter(0)
process(c)                # fn_sigs now shows Counter'actor → c infers 'actor
```

**Limitation:** functions must be declared before their callers. Mutual recursion and reverse declaration order are not covered — a fixed-point iteration would be needed for full coverage.

---

## Struct field inference

The same constraint-elimination algorithm applies to struct fields. A pre-pass scans all method bodies of the struct for `self.field` access patterns. Only fields with no explicit qualifier and a `Named` type are considered. Results are written into the `struct_mutex_fields` / `struct_rwlock_fields` registries used by the emission layer.

```boring
struct Service:
    Counter stats        # no qualifier
    string name

    def record():
        spawn_actor(stats)   # demands 'actor → stats infers 'actor → Arc<Mutex<Counter>>

    req get_name():
        name               # read-only, no sharing demand → no inference (fallback)
```

**Scope:** all fields (public and private) are resolved from internal usage only. The struct exports a single fixed type per field; external callers with different needs must annotate explicitly. Generating per-qualifier variants (monomorphisation) would break module boundaries.

**Generics are not inferred.** Collection types (`[Counter]`, `{K: Counter}`, `{Counter}`) and other generic forms carry the element qualifier as part of their declared type. `[Counter]` is `Vec<Counter>` and `[Counter'actor]` is `Vec<Arc<Mutex<Counter>>>` — these are distinct Rust types and the transpiler cannot promote one to the other based on usage. The qualifier must be written explicitly in the element position.

**Limitation:** cross-file inference is not supported.

---

## Cross-file struct field inference

> **Status: not planned.** This section documents the problem space and open questions. No implementation is scheduled.

### Problem

The current struct field inference scans only the methods defined in the same file as the struct. If a struct is defined in `models.br` but its field is mutated in `service.br`, the inference is blind to that usage:

```boring
# models.br
struct Counter:
    int value        # inferred from this file only

# service.br
let Counter'actor c = Counter(0)
c.value += 1         # this signal is invisible to the inference of `value`
```

The field `value` would not be inferred as `var` from the cross-file usage — the developer would need to annotate it explicitly.

### Algorithmic complexity

Full cross-file inference would require:

1. **Constraint collection pass** — walk the AST of every source file and emit one constraint per `self.field` or `x.field` access pattern. O(N) where N is the total number of AST nodes.
2. **Constraint solving pass** — propagate constraints through the qualifier lattice (`'stack` < `'heap` < `'shared` < `'actor`) until a fixed point is reached. Bounded by lattice height (≈ 5 levels), so O(N × H) ≈ O(N) in theory.
3. **Cycle handling** — recursive structs (`struct Node { children: [Node] }`) require strongly-connected-component detection (Tarjan, O(V+E) on the type graph) before propagation.

Algorithmically linear, but the architecture changes significantly: the transpiler must load and partially analyse all files before emitting any of them. The current single-pass, file-at-a-time pipeline does not support this.

### Open questions

**Does it provide a real benefit?**

The gain is ergonomic: the developer does not need to annotate struct fields that are only mutated from external callers. The cost is:
- A two-phase compilation model (parse all → infer globally → emit).
- Harder incremental builds: changing how one file uses a field invalidates the inferred type of that field for all files.
- Richer error messages: a field qualifier conflict must cite the two sites that produced incompatible constraints, which requires constraint provenance tracking.

**Is the problem real in practice?**

Mutable struct fields already require an explicit binding keyword — `mut` for mutable-in-place, `var` for mutable and rebindable — but the qualifier is still inferred: `mut value = 0` and `var value = 0` are both valid and resolve type and qualifier independently. The additional burden of writing an explicit qualifier (`mut int'actor value`) in the struct definition when the field is used across files may be acceptable — the annotation sits at the definition site, which is where a reader expects to find ownership information.

**Alternative: explicit annotation as the norm for struct fields**

Requiring explicit qualifier annotations on all struct fields (with inference limited to local variables and parameters) would eliminate the cross-file problem entirely. Field qualifiers would always be visible at the definition site, and the single-pass pipeline would remain unchanged. The cost is a slightly more verbose struct syntax.

### `'stack` and `'heap` are API, not hints

In C, the stack-vs-heap distinction is transparent to callers: you always pass either a copied value or a pointer, and the allocation site does not change the calling convention. In Rust, `T` and `Box<T>` are **distinct types** — a function expecting `Counter` and one expecting `Box<Counter>` are not interchangeable at the call site.

This means `'stack` and `'heap` on a struct field are not storage hints that can be silently upgraded: they are part of the **effective signature** of every method and every caller that touches the field. Changing a field from `'stack` to `'heap` based on a cross-file usage signal would cascade type changes across all callers — it is an API break, not an optimization.

This further strengthens the case for explicit field annotations: unlike local variable qualifiers (which affect only internal codegen), field qualifiers are observable at module boundaries and should be declared, not inferred.

### Preliminary verdict

The complexity is architectural rather than algorithmic. Given that:
- Mutable fields already require an explicit `mut` or `var` keyword; adding a qualifier annotation is a small additional step,
- `'stack` / `'heap` field qualifiers are part of the module API and should not be silently changed by remote usage signals,
- The inference benefit is limited to saving a few annotations per struct, and
- The two-phase pipeline would complicate incremental builds,

the likely decision is to **keep cross-file inference out of scope** and document explicit annotation as the expected pattern for field qualifiers that depend on external callers.

---

## Explicit annotation — escape hatch

When inference cannot resolve a qualifier (conflict, or insufficient signals), an explicit annotation overrides everything:

```boring
let Counter'actor c = Counter(0)   # explicit — inference is skipped for c
```

Explicit qualifiers have priority 1 in the inference chain and are never affected by flags or usage signals.

---

## Qualifier unions and groups

A parameter can accept a restricted but not singleton set of qualifiers using a pipe-separated union:

```boring
def process(Counter'stack|heap c):   # accepts 'stack or 'heap, not 'shared or 'actor
    c.inc()
```

The Union members become the **initial candidate set** for that parameter. Body usage signals then narrow the set further. If inference resolves to a single qualifier, the emitted Rust carries the concrete type; if multiple remain, the fallback is the first member of the union.

```boring
def process(Counter'mut c):   # initial: {Stack, Owned, Actor, Guard}
    spawn_actor(c)            # demands 'actor → {Actor} → emits Arc<Mutex<Counter>>
```

Named qualifier groups expand to the corresponding member sets:

| Group | Members |
|---|---|
| `'one` | `'stack`, `'heap`, `'const` |
| `'many` | `'shared`, `'actor`, `'guard` |
| `'mut` | `'stack`, `'heap`, `'actor`, `'guard` |
| `'req` | `'shared`, `'const` |

**Scope: parameters only.** Qualifier groups are useful as parameter-level constraints — they express "this parameter accepts any mutable qualifier" and let the inference narrow to the one actually used in the body. They have no Rust representation (no trait bound is emitted for the union itself) — the constraint is enforced at the Boring level only. On local variables they have no value: the inference starting set already covers the same information, and a developer who knows the qualifier can write it directly.

The transpiler also verifies that callers do not pass a qualifier outside the declared union.

---

## Implementation status

| Case | Status |
|---|---|
| Local variable, call-site demand | ✅ implemented |
| Local variable, return-type demand | ✅ implemented |
| Local variable, task capture | ✅ implemented |
| Local variable, alias propagation | ✅ implemented |
| `T'` (tick) variables with inference | ✅ implemented |
| `mut` keyword as mutation signal | ✅ implemented |
| `var` reassignment as mutation signal (incl. nested fields) | ✅ implemented |
| `set` setter body as mutation signal (struct fields) | ✅ implemented |
| Closure captures (receiver / read-only) | ✅ implemented |
| Mutation + sharing conflict → conflict error | ✅ implemented |
| Qualifier union validation | ✅ implemented |
| Parameter qualifier auto-apply | ✅ implemented |
| Universal borrow inference (free functions only) | ✅ implemented |
| Universal borrow — field-destructuring suppression (`guard let`, `if let`, `let`) | ✅ implemented |
| Universal borrow — disabled for struct/enum method params | ✅ implemented |
| Universal borrow — task capture suppresses auto-ref | ✅ implemented |
| Cross-function propagation | ✅ implemented (single forward pass) |
| Struct field inference (all fields, single-file) | ✅ implemented |
| Optional (`T?`, `T'?`) inner-type inference | ✅ implemented |
| Generic element types (`[Counter]`, `{K: Counter}`) | not applicable — qualifier is part of the declared type |
| Cross-file inference | not implemented |
| Fixed-point propagation (mutual recursion) | not implemented |

---

## Rust research directions (2025–2026)

Active work in the Rust project that may influence Boring's qualifier model.

### Polonius — next-generation borrow checker

Polonius replaces the current NLL borrow checker with a path-based analysis that reasons on access paths rather than regions. It eliminates a class of false positives where the current checker refuses valid code. Targeting stabilisation in 2026 H2.

**Relevance for Boring:** some patterns today requiring `'actor` (interior mutability) to satisfy the borrow checker may become expressible with `'stack` or `'heap` under Polonius, which would shift inference results. Worth revisiting the signal table once Polonius stabilises.

- [Project goal: stabilizable Polonius support on nightly](https://github.com/rust-lang/rust-project-goals/issues/118)

### View types / field projections

A pre-RFC feature (tracking issue [#155938](https://github.com/rust-lang/rust/issues/155938), feature gate `view_types` on nightly) that lets functions declare which struct fields they borrow. Today the borrow checker sees a method call as borrowing the entire struct, blocking simultaneous access to other fields. View types give the borrow checker field-level visibility.

**Relevance for Boring:** this is the closest Rust analogue to per-field qualifier annotations. If view types stabilise, the "one qualifier per field" model Boring already uses could align naturally with what Rust expresses natively, potentially simplifying the emission layer. Worth tracking closely.

- [RFC discussion: view types for partial borrowing](https://github.com/rust-lang/rfcs/issues/3269)

### "Beyond the &" umbrella goal (2026)

A 2026 project goal grouping several related initiatives: `&pin` references for `Pin<&mut T>` ergonomics, field projections (see above), and reborrow traits (`Reborrow`, `CoerceShared`) for user-defined reference types with custom borrow behaviour.

**Relevance for Boring:** `CoerceShared` and `Reborrow` traits could eventually give library-defined wrappers (like `Arc<Mutex<T>>`) first-class reborrow semantics, which would make the `'actor` emission more transparent at the Rust level.

- [2026 Rust project goals](https://blog.rust-lang.org/2026/05/18/project-goals-2026-04/)

### `&move` references (stalled)

A proposed third reference kind that transfers ownership without moving the pointee — useful for `Pin<&mut T>` and similar patterns. RFC [#1617](https://github.com/rust-lang/rfcs/pull/1617) has been open since 2016 with no recent momentum.

**Relevance for Boring:** limited for now. If it resurfaces, it could map to a new qualifier between `'heap` and `'shared`.

### Match ergonomics — RFC 3627 (adopted 2024)

Normalises default binding modes in pattern matching, reducing reference noise in `match` expressions. Already adopted; no qualifier impact.

- [RFC 3627](https://github.com/rust-lang/rfcs/pull/3627)
