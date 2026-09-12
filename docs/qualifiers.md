# Qualifiers — Complete Reference

---

## Overview

Every Boring type carries an optional **qualifier** that describes how the value is stored and shared at runtime. The qualifier is written after the type name, separated by a tick:

```boring
let Counter'actor  c = Counter()   # Arc<Mutex<Counter>>
let Counter'shared r = Counter()   # Arc<Counter>
let Counter'inline s = Counter()   # Counter  (no indirection)
```

Qualifiers are resolved at transpile time. The interpreter ignores them (all values are reference-counted by the runtime). In generated Rust the qualifier determines the exact wrapper type.

---

## Qualifier table

| Boring | Rust (multi-thread) | Rust (single-thread) | Mutable | Notes |
|---|---|---|---|---|
| `'inline` | `T` | `T` | via `mut` binding | Rust default, no wrapper |
| `'owned` | `Box<T>` | `Box<T>` | via `mut` binding | heap-allocated, exclusive |
| `'shared` | `Arc<T>` | `Rc<T>` | no | read-only shared ownership |
| `'actor` | `Arc<std::sync::Mutex<T>>` | `Rc<RefCell<T>>` | interior mutability | sync, no tokio required |
| `'actor'task` / `'task` | `Arc<tokio::sync::Mutex<T>>` | `Rc<RefCell<T>>` | interior mutability | async context |
| `'guard` | `Arc<std::sync::RwLock<T>>` | `Rc<RefCell<T>>` | interior mutability | reader-writer, sync |
| `'guard'task` | `Arc<tokio::sync::RwLock<T>>` | `Rc<RefCell<T>>` | interior mutability | async context |
| `'atomic` | `Arc<AtomicX>` | `Rc<Cell<X>>` | lock-free | scalar-only (`int`/`uint`/`bool`/sized ints) — see below |
| `'weak` | `Weak<T>` / `sync::Weak<T>` | `Rc::Weak<T>` | no | non-owning, inferred from RHS |
| `'static` | `&'static T` | `&'static T` | no | constant global instance, no refcount — see below |

`'actor'task` is an alias for `'task`. Both produce the tokio async lock.

### `'static` — constant global instances

Unlike every other qualifier above, `'static` doesn't wrap a freshly
constructed value in some form of indirection — it names an instance that
is constructed exactly **once**, lives for the entire program, and is
referenced everywhere as a bare `&'static T`. No `Rc`/`Arc`, no refcount,
no heap allocation for the reference itself.

```boring
let Config'static APP_CONFIG = Config(debug = false)   # top level

req show(Config'static cfg):
    print "debug: {cfg.debug}"

show(APP_CONFIG)
```

**Rust equivalent**
```rust
static APP_CONFIG: std::sync::LazyLock<Config> =
    std::sync::LazyLock::new(|| Config { debug: false });

fn show(cfg: &'static Config) { println!("debug: {}", cfg.debug); }

show(&APP_CONFIG);
```

#### Authorized construction sites

A `T'static` value can only be **constructed** at one of three sites —
anywhere else, its initializer must already be a reference to an
existing `'static` value: a bare name whose own declared type is
`'static`, never a fresh construction:

| Site | Form | Notes |
|---|---|---|
| Top level | `let T'static NAME = Ctor(...)` | promoted to a module-level `static`/`LazyLock<T>` |
| Inside `main` | same form, in `main`'s body | hoisted to the same module-level `static` before `main` runs |
| A `type let` field | `type let T NAME = Ctor(...)` on a struct | `'static` is implicit here — **never annotated**, see below |

At an authorized site, the initializer can be a direct constructor call
(`Config(...)`) or a call to an ordinary function/method that itself
returns a freshly constructed value (`create_config()`) — both are
promoted the same way. Anywhere else, **any** call is rejected, not just
a direct constructor call — the checker cannot verify whether an
arbitrary function's return value is a fresh construction or a reference
to something already `'static`, so it conservatively rejects every
`Call`/`MethodCall` initializer outside an authorized site, regardless
of whether the callee looks like a constructor:

```boring
def Config create_config():
    Config(debug = false)

req make():
    let Config'static cfg = Config(debug = false)  # error: cannot construct
                                                     # a 'static instance here
    let Config'static cfg2 = create_config()        # error: same rejection —
                                                     # wrapping the construction
                                                     # in an ordinary function
                                                     # doesn't launder it
```

A `type let` field has no other possible interpretation — it names
exactly one instance per struct declaration, so there is no scenario
where it should be `'shared`/`'actor`/`'guard` instead. The existing
syntax is unchanged; the qualifier is simply never written:

```boring
struct Config:
    pub type let Point origin = Point(1.0, 2.0)   # implicitly 'static
```

This only matters for non-primitive `type let` fields — a scalar (`type
let int MAX = 100`) keeps its existing plain `const` form; `'static`
only materializes for struct/array/dict-typed fields.

#### Provenance also applies to call arguments

Passing a value that isn't itself already `'static`-typed into a
parameter that demands `'static` is rejected the same way as an illegal
construction:

```boring
req show(Config'static cfg): ...

def main():
    let c = Config(debug = false)
    show(c)   # error: cannot pass a non-'static value where 'static is
              # expected — the argument must already be a 'static-typed
              # binding, not a local value, a fresh construction, or a
              # field read
```

#### No interior mutability

`mut 'static` (and `var mut 'static`) is a compile error, exactly like
`mut 'shared` — a bare `&'static T` has nothing for `mut` to unlock. Only
`let T'static` is meaningful. `T'static'weak` is rejected too — `'weak`
exists to detect deallocation, and nothing `'static` is ever deallocated.

#### `Sync` requirement, independent of `--threading`

A Rust `static`/`LazyLock<T>` requires `T: Sync` regardless of
`--threading single|multi`. A `'static` value that nests a
`'shared`/`'actor`/`'guard`/`'weak` field is rejected under `--threading
single` specifically — those qualifiers collapse to non-`Sync`
`Rc`/`RefCell` in single-thread mode (fine under `--threading multi`,
where they're already `Arc`-based and `Sync`):

```boring
struct Outer:
    Inner'shared inner

let Outer'static G = Outer(inner = Inner())
# --threading single: error, 'shared collapses to Rc<T>, not Sync
# --threading multi:  fine, 'shared is Arc<T>, Sync
```

#### Generic structs

A `type let` field whose type does **not** depend on the struct's own
type parameter becomes a genuine cross-instantiation singleton — one
static, shared regardless of how many concrete instantiations exist. A
field whose type **does** depend on the type parameter is rejected: a
Rust `static` cannot be generic, so there is no single instance to share
across instantiations — **unless** a concrete turbofish specialization of
the struct exists somewhere in the program (see `book.md`'s "Turbofish
monomorphization"): the specialized, non-generic copy has no type
parameter left to depend on, so the field compiles fine *for that copy*,
while the struct's still-generic form keeps rejecting it.

```boring
struct Display<T>:
    T value
    type let Logger shared_logger = Logger()   # OK — one instance, any T
    type let T default = value                 # error on the generic form —
                                                # OK if Display<Concrete> is
                                                # specialized elsewhere
```

#### Converting `'static` into `'shared`/`'actor`/`'guard`

There's no special mechanism for this — `'shared c = expr` / `'actor c =
expr` / `'guard c = expr` already accept any expression producing a `T`,
so going from `'static` to one of these is just ordinary construction
whose initializer happens to read the global:

```boring
let Counter'shared c = GLOBAL.clone()   # one allocation + clone, like any other 'shared construction
```

This has a real cost (`T: Clone`, one heap allocation) and is never
inserted implicitly. For `'actor`/`'guard`, this seeds a **new,
independent** mutable instance from the global's current value — it does
not make the global itself mutable.

#### `'static` vs. GPU memory qualifiers

`'static` is a host-side qualifier with no relationship to the GPU
`kernel struct` qualifiers (`'const`, `'local`, `'global`, `'unified`,
etc. — see the GPU module docs) — those describe device memory hierarchy,
a completely different axis, and none of them are ever backed by a Rust
`static`. `--target kernel` (the unrelated `no_std` Rust-for-Linux driver
target) does support `'static`, since plain Rust `static` items work
fine there.

---

## Binding × qualifier interaction

| Syntax | Qualifier constraint | Notes |
|---|---|---|
| `let x'actor = …` | `'actor` | immutable binding, interior mutability via lock |
| `mut x'actor = …` | `'actor` | mutable binding (same Arc, rebindable) |
| `var x'actor = …` | `'actor` | rebindable Arc pointer |
| `let x'shared = …` | `'shared` | read-only, no `def` methods |
| `mut x'shared` | compile error | `'shared` + mutability is incoherent |

### `'actor` and `'guard` on `let` bindings

`let` bindings are normally immutable. `'actor` and `'guard` are exceptions: they provide **interior mutability**, so `def` methods may be called even on a `let`-bound variable. The lock/borrow is acquired automatically.

```boring
let Counter'actor c = Counter()
c.inc()    # OK — def method, interior mutability via Mutex
c.inc()
print c.get()    # → 2
```

This is distinct from `var`/`mut` binding mutability — the `let` binding is not rebindable, but the inner value is mutable through the lock.

---

## Sync vs async variants

Use `'actor` / `'guard` when the code does not use `.await` while the lock is held:

```boring
let Counter'actor c = Counter()
c.inc()                   # std::sync::Mutex — no await needed
```

Use `'actor'task` / `'task` or `'guard'task` inside `task` functions when you need to hold the lock across `.await`:

```boring
task def void worker(Counter'task c):
    c.inc()               # tokio::sync::Mutex — .lock().await
    wait(Duration.fromMillis(100))
    print c.get()
```

**Why the distinction matters in Rust:**
`std::sync::MutexGuard` is `!Send` — you cannot hold it across an `.await` point without making the future `!Send`. `tokio::sync::MutexGuard` is `Send`, designed for async use.

| Qualifier | Lock type | Hold across `.await`? |
|---|---|---|
| `'actor` | `std::sync::Mutex` | ❌ |
| `'actor'task` / `'task` | `tokio::sync::Mutex` | ✅ |
| `'guard` | `std::sync::RwLock` | ❌ |
| `'guard'task` | `tokio::sync::RwLock` | ✅ |

### Inferring `'actor'task` / `'guard'task` from task-method calls

When a variable (or `self.field`) is captured by a `task` expression or closure and used as a method receiver, the inference pass keeps **both** the plain (`'actor`/`'guard`) and `'task` (`'actor'task`/`'guard'task`) variant as candidates instead of jumping straight to the sync lock. It then looks at which methods are actually called on the captured variable inside that body:

```boring
struct Counter:
    var int value = 0

    task def inc():          # declared `task` → needs the tokio lock
        value += 1

def void run(mut Counter c):
    task c.inc()              # c infers 'actor'task — Arc<tokio::sync::Mutex<Counter>>
```

```boring
struct Counter:
    var int value = 0

    def inc():                # plain `def`, not `task`
        value += 1

def void run(mut Counter c):
    task c.inc()              # c infers 'actor' — Arc<std::sync::Mutex<Counter>>
```

If any method called on the captured variable is itself declared `task`, the inferrer picks `'actor'task` (or `'guard'task` if the receiver is otherwise constrained to a reader-writer lock); if none are, it falls back to the plain sync variant. This resolves the ambiguity whenever the disambiguating signal is a method call itself declared `task` — but that was, until the two extensions below, the *only* signal the inferrer recognized.

#### Local live-range analysis: a `with` block holding an unrelated await

A `with name:` block (see "Method dispatch" above and `docs/scoped-access-blocks.md`) conceptually holds its named value's lock for its **entire span** — not just for one method call. That means a genuine await *anywhere* inside that span is proof the async lock variant is required, even when nothing called on the guarded value in the block is itself declared `task`:

```boring
struct Counter:
    var int value = 0

    def inc():                 # plain `def`, not `task` — the method-call signal
        value += 1               # above alone sees nothing here

task def void worker(mut Counter c):
    with c:
        c.inc()                            # not a task-method call
        wait(Duration.fromMillis(100))     # …but this unrelated wait, inside the SAME
                                            # with-held span, is still proof the lock
                                            # is held across an await
    print c.get()
```

The inferrer scans every `with` block in the function/task body it already walks (the same boundary the task-method-call heuristic above uses — recursing into `if`/`while`/`for`/`match`/nested `with`/task/closure bodies, never into a called function's own body) for an unambiguous await point inside that block's own span: an explicit `wait(...)`, a `join […]`, or a direct call into an already-known `task`-declared function/method. Finding one upgrades every named value in that block to the `'task` variant, exactly as if a task method had been called on it directly — `worker`'s `c` above now infers `'actor'task` on its own, with no explicit annotation, closing the gap the previous paragraph used to describe. A `with` block referencing a value is *also*, on its own (regardless of whether it holds across an await), enough to rule out a plain auto-ref borrow for that value — the same "storage signal" a task/closure capture already provides.

The scan is deliberately narrow, matching this section's "conservative toward sync" design (see below): a blocking `.value`/`.wait` on a spawned task handle is a real await too, but recognizing it needs type information this early pre-pass doesn't have, so it isn't recognized — which only means such a case falls into the same residual gap described below, never a false upgrade.

#### Cross-function propagation

Parameter-qualifier inference already propagates forward through `fn_sigs` between functions in the same file (see "Cross-function propagation" below): once a function's own parameter is inferred, callers processed later see the qualified signature and their own matching argument is constrained to it. The `'actor'task`/`'guard'task` decision rides the same mechanism, with no separate fact to propagate — if a callee's own parameter is inferred (by either signal above) to `'actor'task`/`'guard'task`, that qualifier is exactly what ends up in `fn_sigs`, and a caller passing its own `'actor`/`'guard` value into that parameter picks up the same requirement automatically, without needing any local `.await` of its own:

```boring
task def void holdAndWait(mut Counter c):
    with c:
        c.inc()
        wait(Duration.fromMillis(10))     # holdAndWait's own `c` infers 'actor'task

task def void caller(mut Counter c):
    holdAndWait(c)                        # caller has no with-block or wait of its own —
                                           # but its `c` must be the SAME concrete lock type
                                           # as holdAndWait's, so it infers 'actor'task too
```

Because the two lock types are different concrete Rust types (`tokio::sync::Mutex<T>`'s guard is `Send`; `std::sync::Mutex<T>`'s is not, and `tokio::sync::Mutex::lock()` only exists as an `async fn` — there is no synchronous way to call it at all), a caller cannot simply decline a callee's `'actor'task`/`'guard'task` demand the way it can decline, say, a `'shared` demand from a plain value: it must adopt the same variant, or the two would disagree about the very type being shared. This is the one place the plain `'actor`/`'guard` candidate for an otherwise-untouched value is ever widened to also consider the `'task` variant outside an actual task/closure capture — and it only ever fires because some callee has already proven (or been told, via an explicit annotation) that it needs the async lock, never merely because a caller couldn't rule an await out.

This is file-order-dependent, the same known limitation `fn_sigs` propagation already has for every other qualifier: a function only sees a callee's fact if that callee was already processed (its own inference run, whether speculatively via `pre_infer_fn_qualifiers` or for real) by the time the caller's own inference runs. A caller of a callee not yet visible this way — declared later and not yet reached by the speculative forward pass, defined in another module/file this pass doesn't see, or a genuinely external/opaque function boring has no body for — cannot pick up the fact at all.

#### The residual gap, and what to do about it

After both extensions above, there remain cases the analysis simply cannot decide — most commonly a value passed into a function the propagation above hasn't (yet, or ever will) reach. In every one of those cases the default stays the plain `'actor`/`'guard` variant — **never** the `'task` variant, and never a compile error. This is a deliberate design choice, not an oversight: this module's inference always treats an unresolved case as a reason to stay on the cheaper, narrower contract, not as license to guess in the "safe but expensive" direction. The `with`-access-scan two sections above is a documented example of the same policy — "Found → the block gets write access; not found → read-only, even though the binding could support a mutation elsewhere in the program" (see `docs/book.md`) — and the reasoning here is stronger, not just parallel: `'actor`/`'guard` values are used specifically *because* they are shared, so they routinely escape into other functions; treating "cannot prove no await" as a signal to upgrade would make that escape — the ordinary case for this qualifier, not a rare edge one — silently promote almost every `'actor`/`'guard` value in real code to the heavier lock. Worse, since `tokio::sync::Mutex::lock()` has no synchronous form at all, such a silent promotion would not just cost performance — it would stop every other, genuinely-synchronous call site sharing that same value from compiling, a correctness regression cascading from one code path the analysis merely failed to rule out.

The practical consequence: a value that genuinely does need the async lock, but only through a path this analysis cannot see (an opaque external function, a call the forward propagation hasn't reached yet, a blocking `.value`/`.wait` await this pre-pass has no type information for), stays on the plain sync variant and is **not** automatically corrected. The failure mode this produces at runtime is a real one — a task holding a `std::sync::Mutex` guard across what turns out to be an await point either won't compile (the guard is `!Send`, so the enclosing future itself becomes `!Send`) or, if the await is well hidden enough to avoid that, risks a genuine deadlock under contention. Both are bounded and, once hit, straightforward to fix: the developer writes the explicit `'actor'task`/`'guard'task` annotation themselves. The compiler deliberately does not attempt to guess the safe-but-expensive direction on the developer's behalf — see the design rationale above for why that would be worse, not better.

---

## Method dispatch

### On local variables

```boring
let Store'guard s = Store()
s.write(42)      # def → RwLock::write().unwrap()
s.read()         # req → RwLock::read().unwrap()

let Store'guard'task st = Store()
st.write(42)     # def → RwLock::write().await
st.read()        # req → RwLock::read().await
```

The transpiler distinguishes `req` (read-only, `&self`) from `def` (mutating, `&mut self`) to select the appropriate lock mode:

| Method kind | `'actor` | `'actor'task` | `'guard` | `'guard'task` |
|---|---|---|---|---|
| `req` | `lock().unwrap()` | `lock().await` | `read().unwrap()` | `read().await` |
| `def` | `lock().unwrap()` | `lock().await` | `write().unwrap()` | `write().await` |

### On struct fields

```boring
struct Node:
    Counter'actor stats

    def record():
        stats.inc()     # self.stats.lock().unwrap().inc()

    req int total():
        stats.get()     # self.stats.lock().unwrap().get()
```

Field dispatch follows the same `req`/`def` split as local variables.

### Field reads (non-method)

```boring
struct Tag:
    string label

let Tag'guard t = Tag(label = "x")
print t.label            # t.read().unwrap().label
```

Direct field access on an `'actor` or `'guard` variable always acquires the appropriate lock.

---

## Move semantics

By default, assigning a value moves it — the source binding becomes invalid after the assignment. This applies to all qualifiers.

```boring
let a = Counter(0)
let b = a          # a is moved into b — a is no longer accessible
```

To share a value without moving it, call `.clone()` explicitly:

```boring
let a = Counter(0)
let b = a.clone()  # deep copy — a and b are independent
```

### Clone semantics by qualifier

| Qualifier | `.clone()` cost | Result |
|---|---|---|
| `'inline` | deep copy — allocates new value | independent copy |
| `'owned` | deep copy — allocates new `Box<T>` + clones content | independent heap allocation |
| `'shared` | O(1) — increments `Arc` refcount | shared reference to the same value |
| `'actor` | O(1) — increments `Arc` refcount | shared reference to the same mutex |
| `'guard` | O(1) — increments `Arc` refcount | shared reference to the same rwlock |

For `'shared`, `'actor`, and `'guard`, `.clone()` is cheap — it clones the pointer, not the data. All clones refer to the same underlying value.

---

## Qualifier upgrade coercions

A value can be promoted to a richer qualifier at construction time. These are **explicit** coercions — the developer calls them when moving a value from one ownership context to another.

### Upgrade table

| From | To | Boring | Rust emitted | Notes |
|---|---|---|---|---|
| `'inline` | `'owned` | `let b'owned = a` | `Box::new(a)` | move into heap |
| `'inline` | `'shared` | `let b'shared = a` | `Arc::new(a.clone())` | source is cloned, not moved, in the emitted Rust |
| `'inline` | `'actor` | `let b'actor = a` | `Arc::new(std::sync::Mutex::new(a.clone()))` | source is cloned, not moved, in the emitted Rust |
| `'inline` | `'guard` | `let b'guard = a` | `Arc::new(std::sync::RwLock::new(a.clone()))` | source is cloned, not moved, in the emitted Rust |
| `'owned` | `'shared` | `let b'shared = a` | `Arc::from(a)` | no double allocation — Rust optimisation |
| `'owned` | `'actor` | `let b'actor = a` | `Arc::new(Mutex::new(*a))` | unboxes then wraps |
| `'owned` | `'guard` | `let b'guard = a` | `Arc::new(RwLock::new(*a))` | unboxes then wraps |
| `'owned` | `'inline` | `let b'inline = a` | `a.clone()` | clones the boxed value; `a` remains a valid, unmoved `Box<Counter>` in the emitted Rust |

At the Boring-semantics level, all upgrades consume the source value — the interpreter marks the source binding as moved, and reading it afterward raises "use of moved value". However, this is not always a move in the *emitted Rust*: for `'inline` → `'shared`/`'actor`/`'guard` and `'owned` → `'inline`, the transpiler emits `.clone()` on the source, so the original Rust variable remains alive and valid under the hood even though Boring forbids reading it. `Arc::from(box_val)` (used for `'owned` → `'shared`) is the idiomatic Rust way to convert `Box<T>` into `Arc<T>` without a double allocation — the `Arc` reuses the existing heap allocation, and this is the one upgrade that is a true move in the emitted Rust.

### Downgrade

Downgrades (e.g. `'shared` → `'inline`) are not available implicitly. Shared references (`Arc`) cannot be converted back to owned values without an explicit `.clone()` or `.try_unwrap()` (which fails if other references exist).

```boring
let a'shared = Counter(0)
let b = a.clone()    # Arc::clone — b is still 'shared (Arc<Counter>), sharing the same value as a
```

`.clone()` on an `'shared` value is a pointer clone (see the clone-cost table above), not a deep copy — there is no implicit way to obtain an independent `'inline` copy from an `'shared` value. To get one, deref and clone the inner value explicitly (e.g. via a method that returns an owned copy).

---

## Parameter passing

### Full parameter table

| Parameter syntax | Rust emitted | Semantics |
|---|---|---|
| `Counter c` | inferred — see [Inference](#inference) | qualifier inferred from body; or `Counter&` / `mut Counter&` if no storage signal |
| `mut Counter c` | inferred — see [Inference](#inference) | mutable; infers `mut Counter&` if no storage, mutable qualifier otherwise |
| `Counter'inline c` | `Counter` | move (or copy for primitives) |
| `Counter'owned c` | `Box<Counter>` | move |
| `Counter'shared c` | `&Arc<Counter>` | auto-ref, transparent to the developer |
| `Counter'actor c` | `&Arc<Mutex<Counter>>` | auto-ref, callee controls lock granularity |
| `Counter'guard c` | `&Arc<RwLock<Counter>>` | auto-ref, callee controls lock granularity |
| `Counter'shared'weak c` | `&Weak<Counter>` | auto-ref, callee calls `.upgrade()` explicitly |
| `Counter'actor'weak c` | `&Weak<Mutex<Counter>>` | auto-ref, callee calls `.upgrade()` explicitly |
| `Counter'guard'weak c` | `&Weak<RwLock<Counter>>` | auto-ref, callee calls `.upgrade()` explicitly |
| `Counter& c` | `&Counter` | universal borrow, any qualifier, no move, no storage |
| `mut Counter& c` | `&mut Counter` | universal mutable borrow, any mutable qualifier |

`'inline` and `'owned` follow standard Rust move semantics. `'shared`, `'actor`, and `'guard` are always passed by reference — the reference is fully transparent to the developer, who writes and reads these parameters as owned values.

### Auto-ref for `'shared`, `'actor`, `'guard`

The rationale: moving an `Arc` silently increments the reference counter, a cost invisible in the source. A reference suffices in the vast majority of call sites. The auto-ref convention makes this the default.

The transpiler inserts `Arc::clone` (or `Rc::clone` in single-thread mode) automatically whenever a reference parameter is used in an owned position: field assignment, `let` bindings, `match` and `if let` bindings, tuple construction, and call-site arguments.

```boring
struct Processor:
    Counter'actor counter

def init(Counter'actor c):
    counter = c              # field assign → Arc::clone(c)
    let x = c                # let binding → Arc::clone(c)
```

```boring
def store(Counter'actor c):
    …

let x'actor = Counter(0)
store(x)     # Arc::clone(&x) at call site
store(x)     # x is still valid — clone was inserted, not a move
```

### `var` out-parameters

`var` on a parameter signals an out-parameter — the callee can rebind the caller's variable:

| Parameter | Rust emitted |
|---|---|
| `var Counter'inline c` | `&mut Counter` |
| `var Counter'owned c` | `&mut Box<Counter>` |
| `var Counter'shared c` | `&mut Arc<Counter>` |
| `var Counter'actor c` | `&mut Arc<Mutex<Counter>>` |
| `var Counter'guard c` | `&mut Arc<RwLock<Counter>>` |

```boring
def swap(var Counter'actor c):
    c = Counter(1)

var v'actor = Counter(0)
swap(v)   # call site emits: swap(&mut v)
```

### `Counter&` — universal borrow

`Counter&` always produces `&Counter`, regardless of which qualifier the caller holds. The transpiler unwraps the qualifier at the call site:

```boring
req display(Counter& c):
    print c.value

let a'inline = Counter(0)
let b'actor = Counter(0)
let c'owned = Counter(0)
let d'shared = Counter(0)

display(a)   # &a
display(b)   # { let g = b.lock()?; display(&*g) }
display(c)   # &**c
display(d)   # &**d  — Arc<T> derefs to T, no lock needed
```

The caller never writes the lock — it is implicit and scoped to the call.

#### `Counter&` vs `Counter'actor` as parameter

```boring
# Counter& c — transpiler acquires the lock, passes &Counter
# lock held for the entire duration of the call
def process_batch(Counter& c):
    for item in batch:
        c.value += item

# Counter'actor c — callee receives &Arc<Mutex<Counter>>
# callee controls lock granularity
def process_batch(Counter'actor c):
    for item in batch:
        let g = c.lock()
        g.value += item
        # lock released here, not at end of call
```

#### Mutable coercion

```boring
def reset(mut Counter& c):
    c.value = 0

mut a'inline = Counter(0)
let b'actor = Counter(0)

reset(a)   # &mut a
reset(b)   # { let mut g = b.lock()?; reset(&mut *g) }
```

`'shared` (`Arc<T>` without interior mutability) cannot produce `&mut T` — passing a `Counter'shared` to a `mut Counter&` parameter is a compile error.

#### Lock scope and guard lifetime

When the argument is `'actor` or `'guard` and the parameter is `Counter&`, the transpiler generates a temporary binding:

```rust
// display(b) where b: Arc<Mutex<Counter>>
{
    let __g = b.lock()?;
    display(&*__g);
}   // lock released here
```

For `'guard`, a `Counter&` parameter uses `read()` (shared read lock); `mut Counter&` uses `write()` (exclusive write lock).

#### Struct and enum method parameters

Universal borrow inference is **disabled** for parameters of `req` and `def` methods defined on a struct or enum. Use the explicit `Counter& n` form to get universal borrowing in a method parameter.

#### Error conditions

```
error: cannot pass `x` (weak reference) to a non-weak parameter — weak references may be
       invalid. Call .upgrade() first and handle the Option.

error: cannot pass `x` ('shared) to `mut Counter&` — 'shared does not support mutable
       references. Use 'actor (Arc<Mutex<T>>) or 'guard (Arc<RwLock<T>>) instead.

error: cannot pass 'actor argument to `mut Counter&` in async function `f` — holding a
       MutexGuard across .await makes the future !Send. Acquire the lock inside the
       callee body instead.
```

---

## Inference

Boring's qualifier inference works from usage signals — in most programs you never write qualifiers. The zero-annotation goal: qualifier-free Boring code emits the same Rust as hand-annotated code.

`'static` is deliberately **not** part of this system — it's never a candidate for a bare `T`, never joins the fallback chain, and is never inferred from usage. Every other qualifier here answers "how should this freshly-constructed value be stored?", a question inference can answer from usage signals alone; `'static` answers "does this value have a proven lineage back to an authorized construction site?" — a provenance question, checked directly (see the qualifier table above) rather than folded into constraint elimination.

### Constraint elimination

Each unqualified local variable starts with a candidate set of all possible qualifiers. Every usage signal narrows the set by eliminating incompatible qualifiers. When exactly one candidate remains it is chosen. When none remain the constraints are contradictory and a compile error is reported. When several remain a size-based fallback resolves the tie.

#### Candidate sets

| Declaration form | Initial candidate set | Fallback (multiple remaining) |
|---|---|---|
| `T` (bare, no qualifier) | `{Inline, Owned, Shared, Actor, Guard}` | priority-ordered fallback (see below) |
| `T'new` (indirection hint) | `{Owned, Shared, Actor, Guard}` | `'owned` (`Box<T>`) |
| `T?` (optional, bare) | `{Inline, Owned, Shared, Actor, Guard}` | same priority-ordered fallback |
| `T'new?` (optional, indirection hint) | `{Owned, Shared, Actor, Guard}` | `Option<Box<T>>` |

`T'new` and `T'new?` restrict the initial set to indirection qualifiers. For optional forms, the inferred qualifier is applied to the **inner type** of the `Option` — `T?` with inferred `'actor` emits `Option<Arc<Mutex<T>>>`, not `Arc<Mutex<Option<T>>>`.

#### Signal table

| Signal | Compatible qualifiers |
|---|---|
| Call site demanding `T'shared` | `{Shared}` |
| Call site demanding `T'actor` | `{Actor}` |
| Call site demanding `T'guard` | `{Guard}` |
| Call site demanding `T'atomic` | `{Atomic}` |
| Call site demanding `T'inline` | `{Inline}` |
| Call site demanding `T'owned` | `{Owned}` |
| `def` method call on the variable | `{Inline, Owned, Actor, Guard}` |
| `mut` binding (`mut x = …`) | `{Inline, Owned, Actor, Guard}` |
| `var` binding reassigned (`x = …`, `x.field = …`, `x.a.b.c = …`, `x[i] = …`) | `{Inline, Owned, Actor, Guard}` |
| Closure capturing `x` as method receiver | `{Actor, Guard}` |
| Closure capturing `x` read-only | `{Shared, Actor, Guard}` |
| `set` property setter body | `{Inline, Owned, Actor, Guard}` |
| Task capture as method receiver | `{Actor, Guard}` |
| Task capture, read-only | `{Shared, Actor, Guard}` |
| `req` method call | *(no constraint — all qualifiers remain)* |

Each signal intersects the current candidate set. The order of signals does not matter.

#### Priority-ordered fallback

When the candidate set still contains multiple qualifiers after all signals are applied:

**Step 1 — `'inline` candidate**

| Context | Decision |
|---|---|
| Struct field (any binding) | `'inline` — field bytes are part of the parent allocation |
| Local variable, sizeof(T) ≤ `--inline-auto-bytes` | `'inline` |
| Local variable, sizeof(T) > `--inline-auto-bytes` | skip `'inline`; continue to step 2 |

The threshold is configurable: `boring build --inline-auto-bytes 512` (default: 256 bytes).

**Step 2 — ordered chain**

If `'inline` was not selected, pick the first qualifier from the remaining set:

`'owned` > `'shared` > `'actor` > `'atomic` > `'guard`

`'atomic`'s position here is inert for default-selection purposes — see "`'atomic` — lock-free scalar qualifier" below for why (`'actor` always precedes it, so it's reachable only via explicit annotation or an explicit call-site demand, never the plain fallback).

### Examples

```boring
let c = Counter(0)
share_read(c)           # expects Counter'shared → c infers 'shared
```

```boring
let c = Counter(0)
c.inc()                 # def call → {Inline, Owned, Actor, Guard}
                        # no further signal → size fallback → 'inline (if small)
```

```boring
let c = Counter(0)
c.inc()                 # def call → {Inline, Owned, Actor, Guard}
share_read(c)           # 'shared → intersect → {}  → ERROR
```

```boring
let a = Counter(0)
let b = a               # b is an alias of a — same qualifier group
spawn_actor(b)          # demands 'actor → both a and b infer 'actor
```

### Universal borrow as inference output

When a parameter has no explicit qualifier and no storage signals, the inference can resolve to a universal borrow (`Counter&` or `mut Counter&`) — evaluated before the priority-ordered fallback.

**Storage signals** prevent universal borrow inference: field assignment, return with ownership qualifier, closure/task capture, field destructuring (`let x = n.field`).

**Qualifier demand signals** also prevent it: passing to a function parameter with an explicit qualifier.

| Declaration | Signals | Inferred form | Rust emitted |
|---|---|---|---|
| `Counter n` | none | `Counter&` | `&Counter` |
| `mut Counter n` | none | `mut Counter&` | `&mut Counter` |
| `Counter n` | qualifier demand | qualifier via constraints | — |
| `Counter n` | storage | qualifier via constraints | — |

The `mut` keyword is **not inferred** — it must be written explicitly.

```boring
req display(Counter n):
    print n.value
# no storage, read-only → infers Counter& → fn display(n: &Counter)

def reset(mut Counter n):
    n.value = 0
# no storage, mutable → infers mut Counter& → fn reset(n: &mut Counter)
```

Lock acquisition at call sites works the same way as explicit `Counter&`:

```boring
let b'actor = Counter(0)
display(b)
# emits: { let __g = b.lock()?; display(&*__g); }
```

### Parameter auto-apply

A pre-inference pass runs `infer_qualifiers` on the function body before emitting parameters. `emit_param` then consults `inferred_qualifiers` and applies the resolved qualifier automatically:

```boring
def process(Counter c):   # no qualifier written
    spawn_actor(c)        # demands 'actor → inferred for c
# emits: fn process(c: Arc<Mutex<Counter>>)
```

### Cross-function propagation

After each function body is emitted, `fn_sigs` is updated with the inferred parameter qualifiers. Functions defined later in the file that call this function see the qualified signature and propagate the constraint.

**`'inline` is not propagated** — it would poison callers with a spurious constraint from file-ordering artifacts.

**Return-type–driven parameter inference:** when a constructor returns `T'actor`, the transpiler records `T` as an actor source type. Subsequent bare `T` parameters automatically infer `'actor`:

```boring
def Interpreter'actor new_interpreter():
    …

def Value eval_expr(Interpreter interp, Expr e):
    # interp infers 'actor because Interpreter is an actor source type
    …
```

### Struct field inference

The same constraint-elimination algorithm applies to struct fields. A pre-pass scans all method bodies of the struct for `self.field` access patterns:

```boring
struct Service:
    Counter stats        # no qualifier

    def record():
        spawn_actor(stats)   # demands 'actor → stats infers 'actor → Arc<Mutex<Counter>>
```

**Generics are not inferred.** `[Counter]` is `Vec<Counter>` and `[Counter'actor]` is `Vec<Arc<Mutex<Counter>>>` — these are distinct Rust types. The qualifier must be written explicitly in the element position.

**Cross-file inference is not supported** — see the known limitations section.

### Qualifier unions and groups

A parameter can accept a restricted but not singleton set of qualifiers using a pipe-separated union:

```boring
def process(Counter'inline|owned c):   # accepts 'inline or 'owned, not 'shared or 'actor
    c.inc()
```

Named qualifier groups expand to the corresponding member sets:

| Group | Members |
|---|---|
| `'one` | `'inline`, `'owned` |
| `'many` | `'shared`, `'actor`, `'guard` |
| `'mut` | `'inline`, `'owned`, `'actor`, `'guard` |
| `'req` | `'shared`, `'static` |

**Scope: parameters only.** Qualifier groups are not useful on local variables — the inference starting set already covers the same information.

### Explicit annotation — escape hatch

When inference cannot resolve a qualifier, an explicit annotation overrides everything:

```boring
let Counter'actor c = Counter(0)   # explicit — inference is skipped for c
```

---

## Single-thread vs multi-thread mode

The `--threading single` / `--threading multi` flag (default: multi) selects the wrapper implementation:

| Qualifier | `--threading multi` | `--threading single` |
|---|---|---|
| `'shared` | `Arc<T>` | `Rc<T>` |
| `'actor` | `Arc<std::sync::Mutex<T>>` | `Rc<RefCell<T>>` |
| `'guard` | `Arc<std::sync::RwLock<T>>` | `Rc<RefCell<T>>` |
| `'actor'task` | `Arc<tokio::sync::Mutex<T>>` | `Rc<RefCell<T>>` |
| `'guard'task` | `Arc<tokio::sync::RwLock<T>>` | `Rc<RefCell<T>>` |
| `'atomic` | `Arc<AtomicX>` | `Rc<Cell<X>>` |

In single-thread mode `'actor` and `'guard` both map to `Rc<RefCell<T>>` — there is no semantic difference between reader and writer locks in a single-threaded context. The same collapse applies to the `'task` variants: single-thread mode still runs under a tokio `current_thread` runtime (`#[tokio::main(flavor = "current_thread")]`, `tokio::task::spawn_local`), but since everything runs on one thread there is no need for `Send + Sync` locks, so `'actor'task` / `'guard'task` reuse plain `Rc<RefCell<T>>` instead of the tokio async locks.

---

## `'atomic` — lock-free scalar qualifier

`'atomic` is a storage/synchronization qualifier structurally in the same family as `'actor`/`'guard` (a smart-pointer wrapper choice) — it is **not** a provenance qualifier like `'static`, and it participates in the ordinary candidate-elimination inference system rather than being walled off from it. Unlike `'actor`/`'guard`, it is never a real lock: it always maps to a genuinely lock-free hardware/std-library primitive, so it exists only for a narrower family of types.

### Syntax

```boring
let int'atomic counter = 0        # Type'atomic name = value
let counter'atomic = 0            # name'atomic = value — base type inferred from the literal
var flag'atomic = false
```

Like every other qualifier, it can be written on the type or on the name (never both — see "Qualifier on the type OR the name, never both" below), and it participates in explicit qualifier unions: `T'shared|atomic`.

### Type compatibility

`'atomic` only wraps the scalar family with a real `std::sync::atomic` (or, single-thread, `Cell`) equivalent:

| Boring scalar | `--threading multi` | `--threading single` |
|---|---|---|
| `int` | `Arc<AtomicIsize>` | `Rc<Cell<isize>>` |
| `uint` | `Arc<AtomicUsize>` | `Rc<Cell<usize>>` |
| `bool` | `Arc<AtomicBool>` | `Rc<Cell<bool>>` |
| `int8`/`int16`/`int32`/`int64` | `Arc<AtomicI8/16/32/64>` | `Rc<Cell<i8/16/32/64>>` |
| `uint8`/`uint16`/`uint32`/`uint64` | `Arc<AtomicU8/16/32/64>` | `Rc<Cell<u8/16/32/64>>` |

Rejected as a **compile error**, with a message naming the offending type:

- `float`/`float32`/`float64` — no stable `std::sync::atomic` float type exists.
- `int128`/`uint128` — no `AtomicI128`/`AtomicU128` in stable `std`.
- Any struct, enum, collection, or other non-scalar type — no atomic representation at all; use `'actor`/`'guard` for interior mutability there instead.

The single-thread collapse to `Rc<Cell<X>>` reuses the same precedent as `transient` fields (`Cell<T>` for `Copy` types, `RefCell<T>` otherwise, in "Advanced — `transient` fields" in `docs/book.md`) — there is no real concurrency to protect against single-threaded, so the heavier lock-free-atomic representation is unnecessary; `Cell`'s `get`/`set`/`replace` give the same load/store/swap vocabulary at a fraction of the cost.

### Memory ordering

Every generated atomic operation uses `std::sync::atomic::Ordering::SeqCst` — the strongest, safest ordering, consistent with the project's existing "conservative by default" philosophy (compare `--inline-auto-bytes`'s own conservative size estimate). This first version does not expose ordering tuning; a future extension could add it (e.g. `'atomic'relaxed`) once a real workload demonstrates the need.

### Operation mapping

| Boring | Multi-thread | Single-thread |
|---|---|---|
| bare read (`x` as a value) | `x.load(Ordering::SeqCst)` | `x.get()` |
| `x = n` | `x.store(n, Ordering::SeqCst)` | `x.set(n)` |
| `x += n` | `x.fetch_add(n, Ordering::SeqCst)` | `{ let v = x.get(); x.set(v + n); v }` |
| `x -= n` | `x.fetch_sub(n, Ordering::SeqCst)` | `{ let v = x.get(); x.set(v - n); v }` |
| `x.swap(n)` | `x.swap(n, Ordering::SeqCst)` | `x.replace(n)` |

**Deferred**: a recognizable compare-and-swap pattern (`if x == a: x = b`) is not pattern-matched into `compare_exchange` in this first version — left as a documented gap rather than a fragile heuristic. A binding that needs CAS semantics should stay on `'actor`/`'guard` (or use an explicit, hand-written pattern) for now.

### Position in the priority-ordered fallback chain — inert by construction

`'atomic` is inserted into the ordered chain (see "Priority-ordered fallback" above) immediately after `'actor`(/`'actor'task`):

`'owned` > `'shared` > `'actor`(/`'actor'task`) > `'atomic` > `'guard`(/`'guard'task`)

Its exact position relative to `'guard` doesn't matter for default-selection purposes: `'actor` is checked *before* both `'atomic` and `'guard` in the chain, so whenever `{Actor, Guard, Atomic}` (or any subset containing `Actor`) remain candidates simultaneously, `'actor` wins the tie-break regardless of where `'atomic`/`'guard` sit relative to each other — exactly the reason `'guard` is already never chosen by the plain fallback today. This means **`'atomic` is never chosen by inference alone** — it is reachable only via:

1. An explicit annotation (`x'atomic`).
2. An explicit call-site demand elsewhere in the program (a parameter typed `T'atomic`, added to the signal table as "Call site demanding `T'atomic`" → `{Atomic}`, exactly like the existing `'shared`/`'actor`/`'guard` demand signals).

A bare scalar whose only signals are ambiguous between `{Actor, Guard, Atomic}` still resolves to `'actor` by default — confirmed as a regression test against `resolve_fallback` directly (`src/transpiler/infer_qualifiers.rs`'s test module).

### `with`-block incompatibility

A `with`-block (see [chapter 21, Scoped access blocks — `with`](book.md#scoped-access-blocks--with) in `docs/book.md`, and `docs/scoped-access-blocks.md`) lets a `'actor`/`'guard` binding hold its lock across multiple operations instead of acquiring/releasing per access. `'atomic` has no lock/guard object to hold — every access already is a single, independent atomic operation — so `with x: ...` on an `'atomic`-qualified `x` is a **hard compile error**, not a silent fallback to per-access codegen:

```boring
var counter'atomic = 0
with counter:              # ERROR: 'atomic has no lock/guard to hold across a `with` block
    counter += 1
```

### Automatic `'actor`/`'guard` → `'atomic` promotion

This is a **separate, purely additive, behavior-preserving, and conservative optimization pass** — architecturally distinct from `'atomic`'s participation in candidate-elimination inference above. It runs *after* ordinary qualifier resolution has already committed a local binding to `'actor` or `'guard` (never as another candidate competing in the priority-ordered fallback — inserting it there alone would never fire it, since `'actor` always wins that tie-break, which is exactly why this is a separate mechanism).

**False negatives (missing a safe promotion) are acceptable; false positives (an unsound promotion) are not.**

A local (never a struct field, parameter, or return value) `'actor`/`'guard`-qualified scalar binding is promoted to the `'atomic` representation only when **all four** of the following hold, checked within the same function-local analysis scope `with`'s own mutation scan already uses (recursing into `if`/`while`/`for`/`match`/`loop`/`do-while`/`guard`/`try`/`defer`/closures nested in the same function — never into a called function's own body):

1. **Atomic-eligible scalar type** — never a struct, never a float, never `int128`/`uint128`.
2. **Never escapes the local scope** — never returned, never assigned into a struct field, never captured by a `task`/closure, never passed as an argument to a function/method call (its own `swap` excepted).
3. **Every access decomposes into a single atomic primitive** from the operation-mapping table above: a bare read, `x = <value not referencing x>`, `x = x + <value not referencing x>` / `x = x - <value not referencing x>` (a genuinely single `fetch_add`/`fetch_sub` instruction), or `x.swap(<value not referencing x>)`. Any other shape referencing `x` on an assignment's RHS (`x = x * 2`, `x = f(x)`, `x = x + x`, ...) would require decomposing into a separate load then store under the atomic representation — **not** equivalent to the original lock-protected read-modify-write, so it blocks promotion rather than firing an unsound rewrite.
4. **Never used inside a `with` block** anywhere in the same local scope — same non-whole-program boundary as `with`'s own existing scan, a known, already-accepted limitation, not a new one.

Both `'actor` and `'guard` sources are treated identically: promoting a `'guard`-resolved scalar is at least as safe as promoting `'actor` — a `RwLock`'s whole benefit (cheap concurrent reads) is preserved and improved by a lock-free atomic `load()`.

**Deferred** (future work, not attempted in this first version):

- **Compare-and-swap pattern detection** — a `'actor`/`'guard` scalar used only via an `if x == a: x = b` pattern is not recognized as CAS-safe and stays on the lock-based representation.
- **Cross-function whole-program promotion** — a promoted variable passed to another Boring function/method is always treated as escaping (criterion 2), exactly like `with`'s own local-only scan never opens a called function's body. A future version could extend the analysis to also examine the callee's own body when it's defined in the same file/module, under the same conservative false-negative-ok principle.

Implementation: `src/transpiler/promote_atomic.rs` (`scan_atomic_promotions`, run once per function body immediately after `infer_qualifiers`/`infer_struct_field_qualifiers`, before any statement is emitted) populates `self.promoted_atomic_vars`; `emit_let.rs`'s `try_emit_qualified_let` consults it before falling through to the ordinary `'actor`/`'guard` mutex/rwlock emission.

---

## `'weak` references

A `'weak` qualifier produces a non-owning reference. The base qualifier is inferred from the right-hand side:

```boring
let a'shared  = Resource(label = "hello")
let b'weak    = a        # Weak<Resource> — inferred from a's 'shared qualifier

let r = b.upgrade()
print r.label            # "hello"
```

Explicit compound forms for type annotations and function signatures:

| Boring | Rust |
|---|---|
| `T'shared'weak` | `Weak<T>` (rc) or `sync::Weak<T>` (arc) |
| `T'actor'weak` | `sync::Weak<Mutex<T>>` |
| `T'guard'weak` | `sync::Weak<RwLock<T>>` |

Passing a `'weak` value to any non-weak parameter is a compile error — the transpiler requires an explicit `.upgrade()`.

---

## `string` and primitive types

Primitives (`int`, `uint`, `float`, `bool`) have no meaningful qualifier — they are always `Copy` in Rust. The bare names are the canonical form:

| Boring | Rust |
|---|---|
| `int` | `i64` |
| `uint` | `u64` |
| `float` | `f64` |
| `bool` | `bool` |
| `str` | `&str` |
| `string` | `Arc<str>` (multi-thread) / `Rc<str>` (single-thread) / `&'static str` (literals, strict mode) |

`string` uses `Arc<str>` to enable arbitrary value lifetimes and sharing. In single-thread mode it uses `Rc<str>`. Strict mode restricts `string` to compile-time literals only (`&'static str`); computed or interpolated values require explicit annotation.

---

## Known limitations

### Cross-file struct field inference

Struct field inference scans only the methods defined in the **same file** as the struct. Qualifiers used by callers in other files are not visible to the inferrer.

This is a deliberate constraint, not a gap to fill later:

- Mutable fields already require an explicit `mut` or `var` keyword; adding a qualifier annotation is a small additional step.
- `'inline` / `'owned` field qualifiers are part of the module API and must not be silently changed by remote usage signals.
- Full cross-file inference would require a two-phase compilation model (parse all → infer globally → emit), which complicates incremental builds.

**Expected pattern:** write explicit qualifier annotations on fields whose qualifier depends on external callers. The inference handles everything within a file automatically.

### Managed mode and a bare oversized local variable (fixed)

A bare (unqualified) local variable whose type is a struct larger than
`--inline-auto-bytes` is resolved by the size-based fallback (see
`docs/transpilation-modes.md` "Size-based auto-boxing (strict mode only)") —
`resolve_fallback` (`src/transpiler/infer_qualifiers.rs`). This fallback used to
not check `--mode` at all, so it resolved a bare oversized local to `'owned`
regardless of mode; in `--mode managed`, `'owned` means `Arc<Mutex<T>>`/
`RefCell<T>` rather than `Box<T>`. A function with a bare (also unqualified)
return type of the same oversized struct does **not** get its signature promoted
in managed mode (see "Function return types" in `docs/transpilation-modes.md` —
managed mode's own promotion is keyed on an explicit qualifier), so a bare
oversized local returned as a tail expression from such a function could end up
managed-wrapped while the function's own signature stayed a plain, unwrapped `T`
— a real `cargo build` `E0308` mismatch. The same root cause also broke a bare
oversized local used only as a struct-field initializer inside `main()` (never
returned at all): the local's own `let` got wrapped while its constructor-call
initializer stayed bare and unwrapped.

Fixed by making `resolve_fallback` itself mode-aware: size-based auto-boxing now
only ever escalates a bare local past `'inline` in strict mode (matching the
"strict mode only" scope size-based auto-boxing has always been documented
with) — in managed mode a bare oversized local now stays plain, unboxed `T`
regardless of size, agreeing by construction with the (also unpromoted)
managed-mode function-return signature and with any bare, unwrapped constructor
call that initializes it. See `tests/cases/oversized_return_boxed_via_let.br`
and `tests/cases/oversized_struct_field_stays_inline.br` (both now run all four
mode/threading variants in `tests/transpile.rs`, no `ignore_managed`).

---

## Implementation notes

### Tracking sets

The transpiler maintains four sets per scope for dispatch:

| Set | Contents |
|---|---|
| `var_mutex_types` | local vars with `'actor` |
| `var_mutex_task_types` | local vars with `'actor'task` / `'task` |
| `var_rwlock_types` | local vars with `'guard` |
| `var_rwlock_task_types` | local vars with `'guard'task` |
| `var_atomic_types` | local vars with `'atomic` (explicit, or promoted from `'actor`/`'guard` — see `promoted_atomic_vars` below) |

Parallel sets exist for struct fields (`struct_mutex_fields`, `struct_mutex_task_fields`, `struct_rwlock_fields`, `struct_rwlock_task_fields`).

These sets are populated during statement emission and propagated into sub-transpilers (method bodies) via `make_sub()`.

### Dispatch helpers

| Helper | Emits |
|---|---|
| `mutex_var_read(var, expr)` | `var.lock().unwrap().expr` or `.await` |
| `mutex_var_write(var, expr)` | same, write guard |
| `mutex_field_read(key, expr)` | field via read lock |
| `mutex_field_write(key, expr)` | field via write lock |
| `rwlock_field_read(key, expr)` | field via `read()` |
| `rwlock_field_write(key, expr)` | field via `write()` |
| `guard_read_access(v)` | `v.read().unwrap()` |
| `guard_write_guard(v)` | `v.write().unwrap()` |
| `guard_task_read_access(v)` | `v.read().await` |
| `guard_task_write_guard(v)` | `v.write().await` |

### Parser lookahead for compound qualifiers

`'actor'task` and `'guard'task` are two-token qualifiers. The `is_type_start_before_ident()` lookahead was extended to consume both tokens so that `let Counter'actor'task c = …` is parsed as a type annotation rather than an expression.

### Qualifier on the type OR the name, never both

A `let` binding's qualifier can be written on the type (`let Counter'actor c = …`) or, when the type can be inferred, on the name (`let c'actor = …`) — see the book's "Qualifier on the variable name" section. Writing it in **both** positions on the same binding (`let Counter'actor c'guard = …`) is a parse error (`parse_let_stmt_pub` in `src/parser/parse_stmt.rs`), not silently resolved by any precedence rule — there is no defined semantics for which qualifier would win, and the transpiler has no sane way to emit a doubly-qualified type. This is unrelated to a legitimate compound chain (`'actor'task`, `'shared'weak`): those are written as a single tick-sequence in one position and are unaffected.

### Interior mutability in the interpreter

The interpreter's `Env` tracks `actor_bindings: HashSet<String>` — variables declared with an interior-mutable qualifier. Calls to `def` methods on these variables skip the "cannot call mutating method on immutable binding" check, matching the transpiler's semantics.

### Inference implementation status

| Case | Status |
|---|---|
| Local variable, call-site demand | ✅ implemented |
| Local variable, return-type demand | ✅ implemented |
| Local variable, task capture | ✅ implemented |
| Local variable, alias propagation | ✅ implemented |
| `T'new` (indirection hint) variables with inference | ✅ implemented |
| `mut` keyword as mutation signal | ✅ implemented |
| `var` reassignment as mutation signal (incl. nested fields) | ✅ implemented |
| `set` setter body as mutation signal (struct fields) | ✅ implemented |
| Closure captures (receiver / read-only) | ✅ implemented |
| Mutation + sharing conflict → conflict error | ✅ implemented |
| Qualifier union validation | ✅ implemented |
| Parameter qualifier auto-apply | ✅ implemented |
| Universal borrow inference (free functions only) | ✅ implemented |
| Universal borrow — field-destructuring suppression | ✅ implemented |
| Universal borrow — disabled for struct/enum method params | ✅ implemented |
| Universal borrow — task capture suppresses auto-ref | ✅ implemented |
| Cross-function propagation | ✅ implemented (single forward pass) |
| Struct field inference (all fields, single-file) | ✅ implemented |
| Optional (`T?`, `T'?`) inner-type inference | ✅ implemented |
| `'actor'task`/`'guard'task` vs `'actor`/`'guard` disambiguation (task-method-call signal) | ✅ implemented |
| `'atomic` — explicit qualifier, scalar type-compatibility gate, `with`-incompatibility | ✅ implemented |
| `'atomic` — priority-chain position + call-site demand signal | ✅ implemented (inert by construction — see above) |
| `'actor`/`'guard` → `'atomic` automatic promotion (local scalars, single-op compound assign) | ✅ implemented |
| `'atomic` — compare-and-swap pattern detection | not implemented |
| `'atomic` — cross-function whole-program promotion | not implemented |
| Cross-file inference | not implemented |
| Fixed-point propagation (mutual recursion) | not implemented |

> **Note on `'actor'task` / `'guard'task`:** inference picks the `'task` variant when a `task`-declared method is called on the captured variable/field (see "Inferring `'actor'task` / `'guard'task` from task-method calls" above); otherwise it falls back to the plain sync variant. A task body that needs the async lock without calling a `task` method on the captured value itself still requires an explicit annotation.

### Rust research directions

#### Polonius — next-generation borrow checker

Polonius replaces NLL with a path-based analysis that eliminates false positives. Targeting stabilisation in 2026 H2. Some patterns today requiring `'actor` may become expressible with `'inline` or `'owned` under Polonius.

#### View types / field projections

Feature gate `view_types` on nightly (tracking [#155938](https://github.com/rust-lang/rust/issues/155938)) — functions declare which struct fields they borrow, giving the borrow checker field-level visibility. Aligns with Boring's per-field qualifier model.

#### "Beyond the &" umbrella goal (2026)

Groups `&pin` references, field projections, and reborrow traits (`Reborrow`, `CoerceShared`). `CoerceShared` could give `Arc<Mutex<T>>` first-class reborrow semantics, making `'actor` emission more transparent at the Rust level.
