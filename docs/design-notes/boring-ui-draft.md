# Draft — boring-ui: a SwiftUI-flavored, custom-rendered GUI toolkit for Boring

Status: **working draft**, not a spec. Nothing in this document is implemented yet except the
language mechanisms explicitly marked as shipped below (§1's array-block sugar, and `'observed`
in §3 — including, as of this update, struct fields/parameters/return types, not just local
bindings). This is a starting point to be enriched before any implementation work begins — several
rows in the comparative table are deliberately left as open questions rather than guessed at.

## Goal

Give Boring a way to build real desktop and mobile applications — not the in-engine, Bevy-rendered
UI that `bevy-boring`/`breakout-boring`/`scratch-boring` already use for games, but standalone
apps with their own window, driven by application logic, in the spirit of SwiftUI: a declarative,
composable view syntax, adapted to Boring's own idioms and to Rust's ownership model rather than
copied wholesale from Swift.

## Non-goals / explicit scope exclusions

- **No `--target kernel` support.** A `no_std` Rust-for-Linux kernel module has no windowing
  system and no concept of a mounted view tree — `view` (below) is meaningless there, the same way
  GPU kernel functions are meaningless in a plain CPU build. `boring-ui` is a std-target-only
  concern; the kernel backend's own qualifiers (`'unified`/`'sync`/`'global`) and this feature's
  mechanisms (`'observed`, `state`) never interact.
- **Not native-widget wrapping.** Deliberately Flutter/Compose-shaped (one custom Rust/wgpu
  renderer, reusing the rendering groundwork already proven out in `bevy-boring`/`breakout-boring`),
  not SwiftUI's actual strategy of bridging to real AppKit/UIKit widgets everywhere. A **native
  escape hatch** (below) exists for the specific, narrow cases where custom rendering is the wrong
  tool (native text editing/IME, OS-mediated pickers and dialogs), not as the default.
- **Not a drag-and-drop editor.** Code-first, like SwiftUI's own text-based `body`, not a visual
  designer.

## Rendering model, in one line

Custom rendering (own arbitrary widget tree, own paint pipeline, presumably wgpu-based reusing the
`bevy-boring`/`breakout-boring` groundwork) chosen over native-widget wrapping, because the latter
would mean binding four-plus separate native toolkits (AppKit, UIKit, a Linux toolkit, Android
Views/JNI) — an order of magnitude more engineering than this project can sustain, and the closest
real precedent for "native widgets across desktop *and* mobile from one non-Apple/non-Google
toolkit" (Xamarin.Forms/.NET MAUI) has a rocky, leaky-abstraction track record over a decade.

## Language mechanisms

### 1. Composition — trailing array-block sugar (**shipped**)

`Column:` / `Column(spacing: 8):` followed by an indented block of expression lines (plus
`if`/`elif`/`else`/`for`) desugars to a trailing `[dyn Trait]` array argument — recovers
SwiftUI-style nested indentation for widget trees with zero new grammar production: it's the
*same* trailing-closure call shape Boring already had, resolved by looking up the callee's
signature:

| `Ident` resolves to... | Interpretation |
|---|---|
| known callable, last param `[dyn Trait]` | array-block ("collect") semantics |
| known callable, last param `Fn(...)` | ordinary trailing-closure sugar (tail semantics) — unchanged |
| known callable, other last-param type | compile error |
| not a known callable | ordinary closure literal, fresh implicit parameter — unchanged |

Deliberately scoped to trait-object arrays (`[dyn Trait]`), not generic `[T]`, to avoid surprises.
Generic higher-order closures (`.map n: ...`, `.filter`, `.reduce`) are completely unaffected —
resolved by their own callee's signature before this path is ever reached.

Implemented on `main` (commit `b4f8a88`, `src/desugar_array_block.rs` +
`src/parser/parse_array_block.rs`); documented in `docs/book.md`'s "Trailing array-block sugar"
section and `spec/grammar.bnf`. Full test suite green (unit 789/0, run 182/0, transpile 628/0,
interpreter_build 4/0, interpreter_functional 83/0). The shipped resolver currently consults only
same-file declarations.

#### Imported array-block callables (**designed, not shipped**)

**Design decision (2026-09-30):** extend the same resolution rule to imported callable signatures.
This is required for `boring-ui`; components such as `Column` must remain ordinary library symbols,
not language keywords or compiler intrinsics.

```boring
# boring-ui declaration/facade
Column(float spacing = 8, [dyn View] children): native

# application
use boring_ui.Column

Column:
    Text("Bonjour")
```

The parser continues producing the existing ambiguous call/block node. Before array-block
desugaring, name resolution must make exported signatures from directly imported Boring modules
and project dependencies available alongside same-file signatures. A native/Rust implementation
exposes a Boring `native` facade with its real public signature, so the same mechanism covers both
Boring-written and native components.

No UI-specific names participate in this rule. An imported callable whose last parameter is
`[dyn Trait]` selects collect semantics; one ending in `Fn(...)` selects ordinary trailing-closure
semantics; any other known signature produces the existing error. Visibility, aliases, overload
selection, ambiguous glob imports, and dependency metadata/incremental-compilation details must
follow ordinary import resolution rather than introduce an array-block-only namespace.

This language extension is a prerequisite for implementing the toolkit as a reusable library.

`for`-generated dynamic children have an optional keyed form, designed below under Identity &
lifecycle: `for item in items with item.id:`. It is not implemented yet.

### 2. `view` — a declaration kind distinct from `struct`

A `view` is not sugar for `struct ... as View:`. Its fields have different storage semantics: a
`struct`'s fields are ordinary owned data, moved/copied/borrowed with the value; a `view`'s
instance is **rebuilt on every refresh** (its `body()` runs again, producing a fresh tree to diff
against the previous one — see Identity & lifecycle), so a `view`'s reactive fields cannot simply
live inline in that throwaway value. They need to be handles into storage that survives across
rebuilds.

```boring
view FormView:
    state var string name = ""

    body():
        Column:
            TextField(value: name)
            Button("Submit").onClick (): print "submitted {name}"
```

`state` is a `view`-specific field-declaration keyword, not an attribute or a qualifier (§3).
`var`, not `mut`, still follows the ordinary scalar rule: `state` never changes `name`'s Rust
representation (`var` for a rebindable scalar — see `docs/book.md`/`CLAUDE.md`'s
"Binding × mutability (scalars)").

### 3. `'observed` and `state` — sharing and reactivity, kept as two separate axes

Two questions turned out to be independent and are answered by two different mechanisms:
*"can more than one owner reference this value?"* (an ownership-qualifier concern, `'observed`) and
*"should changing this specific field refresh a specific view?"* (a per-field, per-view concern,
`state`). Earlier revisions of this section conflated them into one qualifier (`'state`); kept
apart, each ends up simpler and each generalizes correctly beyond `boring-ui`.

#### `'observed` — a general, composable sharing suffix (not GUI-specific) (**shipped**, including struct fields/parameters/return types)

Language-mechanism status update: `'observed` shipped for local bindings first, then — specifically
to unblock this design, which puts `'observed` on `view`/model *fields* below, not just locals — was
extended to struct field declarations, function/method parameters (the `@ObservedObject`-shaped
case, §4's comparative table), and return types. Every example in this section that reads
`mut FormModel'observed model = FormModel()` as a struct field, or a parameter receiving an
already-observed value from outside, is real, working syntax today, not a forward-looking sketch —
see `docs/book.md`'s "'observed" section for the full spec (composition table, construction,
`subscribe()`, the auto-derive interaction, and a short list of known remaining gaps, e.g. field-level
`mut`/`var mut` permission not yet enforced the way a local binding's already is). `state` (the rest
of this section) remains an unimplemented design only.

`'observed` adds an embedded, independently-locked notification mechanism to a value, without
fixing what the value itself is stored as. Representationally:

```
'observed T  ≈  struct ObservedCell<T> { value: <base qualifier's storage>, subscribers: Arc<Mutex<Vec<Box<dyn Fn()>>>> }
```

It composes as a **suffix** on an existing ownership qualifier — the same shape as `'actor'task`/
`'actor'weak` already in the language — rather than standing alone with one fixed mapping, because
the two fields are genuinely independent (a separate lock each) and the *value* half is free to
vary:

| Base | `'observed` composition | Verdict |
|---|---|---|
| `'inline` | `'inline'observed` | valid — no sharing across owners, but a real observation mechanism can still exist for the sole owner |
| `'owned` | `'owned'observed` | valid — same as above, plus heap indirection for large values |
| `'actor` | `'actor'observed` | valid — the common case: shared, exclusive-lock (`Mutex`) access |
| `'guard` | `'guard'observed` | valid — shared, reader-writer (`RwLock`) access; better than `'actor'observed` for read-heavy shared models (concurrent reads, occasional writes) |
| `'shared` | `'shared'observed` | **rejected** — `'shared` has no interior mutability at all (`mut 'shared` is already a compile error); an observed cell that can never be written can never have anything to notify about |
| `'actor'weak` / `'guard'weak` | `'actor'weak'observed` / `'guard'weak'observed` | valid in principle, low priority — a non-owning reference to an observed cell (e.g. a cache that shouldn't keep the model alive) |

Bare `'observed` (no explicit base) resolves through the *existing* candidate-elimination /
priority-fallback qualifier-inference chain (`docs/book.md` §30) — defaulting to `'actor'observed`
when usage shows the value is referenced by more than one owner, matching how an otherwise-bare
struct already infers toward `'actor` today. The `--inline-auto-bytes` size-based auto-boxing
threshold (`docs/transpilation-modes.md`) decides `'inline'observed` vs `'owned'observed` the same
way it already decides inline-vs-boxed for any other otherwise-unqualified struct — no new
inference rule needed, just reusing what's already there.

##### `subscribe()` / `Subscription` — a general, explicit, standalone primitive

```boring
struct Subscription:
    # opaque handle — unsubscribes when dropped (RAII)

def T'observed.subscribe(fn () callback) -> Subscription:
    ...
```

General and standalone, not tied to `view` — the same layering every mature reactive ecosystem
uses (Rx's `Disposable`, Combine's `AnyCancellable`, Vue's `@vue/reactivity` + `onCleanup`,
SolidJS's `onCleanup`, .NET's `IObservable`/`IDisposable`): a general primitive first, with the UI
framework as just its first, automated caller. Rust's deterministic `Drop` makes this cheaper and
more principled here than in any of those languages — Combine needs ARC to make `AnyCancellable`
clean up on deallocation; Boring gets the same result for free from ordinary destructor semantics,
no weak-reference-plus-lazy-pruning workaround needed (an earlier revision used `Weak` subscriber
entries pruned lazily on the next write — a garbage-collected-language idiom, discarded once it was
clear Boring already has the better-fitting primitive).

##### Getting plain, un-observed access — field access, not a conversion

`'observed T` is honestly a two-field struct, so reaching for its `value` field directly (whatever
base qualifier it holds) is ordinary, visible field access, not a hidden downgrade — nothing
capability is silently dropped, because nothing was silently granted in the first place: no
`subscribe()` call means no notification, whether or not you happen to also hold a plain-field
reference alongside it.

```boring
let plain = myField.value   # same shared cell, ordinary access — no subscribe() ever called on it
```

#### `state` — a view-local field declaration, exclusive and persistent

`state` is deliberately a new field-declaration keyword available only inside `view`. A view is
already a distinct declaration kind with identity-scoped storage, so its grammar can express that
storage directly instead of pretending it is an ordinary field carrying a general-purpose
attribute. A `state` field is **exclusive** (no qualifier, or `'inline`/`'owned`), persistent for the
mounted view identity, and private after construction. `pub state`, `state pub`, and
`state ...'observed` are compile errors.

The keyword does not change the stored value's Rust representation. Its jobs are to tell the
transpiler that the value lives in the view identity's persistent slot and where to insert a
refresh call. It also synthesizes an optional named constructor argument, so callers can choose the
initial value without declaring an `init` and without gaining later field access:

```boring
view Counter:
    string title
    state var int count = 0

let counter = Counter(title: "Inbox", count: 10)  # seeds count on first mount
counter.count = 11                                 # compile error: state is private
```

The constructor argument is an **initial-state seed**, not a parent-controlled input. On the first
mount of an identity it initializes the persistent slot; rebuilding the same identity does not
overwrite that slot with a newly supplied seed. Ordinary non-`state` view fields are inputs and are
updated from the rebuilt value. This distinction must be visible in diagnostics, because silently
treating a changing parent argument as state would otherwise be surprising.

Within the owning view:

- **On an exclusive field** — this is the *only* mechanism available at all, since there is no
  subscriber list anywhere for such a field (nothing else can ever reference it to subscribe). The
  one and only possible observer is always exactly the same `self` performing the write, so the
  transpiler inserts a **direct, statically-resolved call** (`self.__schedule_refresh()`) right at
  the recognized assignment site — no lock, no list, no `Subscription`, nothing stored anywhere.
  This holds even through a `bind`-mediated write from a child (below): the slot handle writes
  through the parent's persistent state and schedules the same direct refresh. The
  field's runtime representation is therefore *identical* to an unmarked field of the same type —
  `state`'s mutation cost is that one inserted call (persistent slot storage is specified
  separately by the view lifecycle).
- **On an `'observed` field, no attribute needed at all — subscription is automatic and
  unconditional.** An earlier revision put `@state` here, deciding at mount whether to
  `.subscribe()` — reasoned away on review: the write path of an `'observed` field must lock and
  walk its subscriber list *regardless*, to notify whichever other views are watching — that's
  `'observed`'s entire reason to exist, paid on every write no matter how many subscribers exist,
  including zero. So gating *this* view's own membership in that list saved nothing on the write
  side; the only thing it changed was whether this view gets refreshed by *other* writers' changes,
  and for that, defaulting to "always subscribe" turns out to be the actual mainstream answer, not a
  compromise — it's exactly how SwiftUI's classic `ObservableObject`/`@Published` already behaves
  (any `@ObservedObject`/`@StateObject` reference re-evaluates `body` on *any* published change to
  that whole object, no per-property opt-out). The "wasted" recompute this can cause when a view
  doesn't actually render the part that changed is bounded and cheap — a `body()` recompute and a
  diff that finds nothing to apply — the same trade-off every coarse-diffing UI framework
  (React/Flutter/Compose/SwiftUI) already accepts rather than asking for manual opt-outs.

```boring
# view-local, exclusive — state is the entire refresh mechanism
view FormView:
    state var string name = ""

# — versus, once `name` needs to be shared across sibling views —

struct FormModel:
    var string name = ""
    def setName(string s): name = s

view FormView:
    mut FormModel'observed model = FormModel()   # no state — subscribes at mount unconditionally;
                                                  # any def call on `model` then refreshes every
                                                  # view that holds an 'observed reference to it
```

"Promote local state to a shared model" stays a plain refactor either way (extract the field, change
its qualifier) — it just also means dropping `state`, since the destination no longer needs it.

##### Binding keyword

Since `state` never wraps the type, the *ordinary*, already-existing binding rules apply
unmodified, keyed only on whatever qualifier (if any) is actually present — `state` itself changes
nothing about them:

- No qualifier, scalar (`state var string name = ""`) — plain scalar rule: `var`, never `mut`
  (`docs/book.md`/`CLAUDE.md`, "Binding × mutability (scalars)" — a bare scalar has no `def` methods
  for `mut` to unlock). This is exactly right for the mechanism too: a scalar's only kind of "change"
  *is* rebinding (`name = "x"`) — there's no separate "mutate in place" operation to distinguish —
  and rebinding is already legal on a plain `var` today, so `state` needs nothing special here at
  all, just recognizing that assignment at the site.
- No qualifier, struct (`state mut FormModel model = FormModel()`) — plain struct rule: `mut`
  (content-mutable, so `.setName(...)`-style `def` calls work) unless the field should also be
  rebindable.
- `'observed`-qualified struct (`mut FormModel'observed model = FormModel()`, no `state`) — follows
  the *existing* `'actor`/`'guard` row of the qualifier/mutability table (`docs/book.md` §21): `var`
  alone is rebind-only and does **not** unlock `def` calls ("an earlier revision let `var T'actor x`
  unlock `def` calls on the qualifier's strength alone; that exception is retired") — `mut` (or
  `var mut`) is what's actually needed, exactly as for any other `'actor`/`'guard` field today.
  Nothing new: `'observed` slots into rows the checker already enforces for its base qualifier.

##### `bind` — a view input that writes through to reactive storage

**Design decision (2026-09-30):** `bind` is a second `view`-specific field-declaration keyword. It
expresses a non-owning, persistent read/write connection to reactive storage owned by an ancestor,
without exposing a general-purpose `Binding<T>` type or SwiftUI-style `$` projection in source.

```boring
view NameEditor:
    bind var string name

    body():
        Column:
            Text(name)
            TextField(value: name)

view FormView:
    state var string name = ""

    body():
        NameEditor(name: name)
```

The expected `bind` field at the call site makes the projection contextual. `NameEditor(name:
name)` passes a handle to the parent's persistent slot rather than copying the current string.
There is no special call-site marker. Passing an existing `bind` onward to another `bind` field is
the same operation.

The ordinary binding keywords describe permitted operations on the **value in the source slot**,
not rebinding the child's internal handle:

```boring
bind var string name          # `name = "x"` replaces the parent's scalar value
bind mut Profile profile      # mutating `def` calls modify the parent's struct in place
bind var mut Profile profile  # both replacement and in-place mutation are permitted
```

The handle itself remains connected to the same source slot for the mounted child's lifetime.
When the parent rebuild supplies a different compatible source slot to the same child identity,
the update phase replaces the handle before the child's next `body()` evaluation.

A `bind` field has no initializer and must be supplied by its parent. A compatible argument is a
modifiable `state` field, another `bind`, or a writable projection into an `'observed` value. A
temporary, calculated expression, immutable input, or ordinary local variable is rejected because
it has neither suitable persistent storage nor the required reactive write path. Field projections
such as `profile.name` are permitted only when the entire path preserves the requested `var`/`mut`
permission and notification semantics.

Within the child, a `bind` name reads transparently as its value type and assignments/mutating calls
write through to the source. Such a write invokes the source's refresh mechanism: a direct refresh
for ancestor `state`, or observed notification for an `'observed` projection. Internally this needs
a slot handle with a lifetime tied to the mounted subtree; it is deliberately not an ordinary Rust
borrow, because callbacks use it after the constructing `body()` call has returned.

Standard input components use the same conceptual contract, for example:

```boring
TextField(bind var string value)
Toggle(bind var bool value)
Slider(bind var float value, float min, float max)
```

The exact declaration mechanism for native/Rust-implemented components remains an implementation
detail, but their Boring-facing signature and checker behavior must match a `bind` field.

##### UI events and batched refresh

**Design decision (2026-09-30):** reactive writes invalidate views immediately but defer rebuilding
until the outermost synchronous UI event callback returns. The runtime deduplicates invalidated view
identities, so any number of writes affecting the same view during one event produce one `body()`
evaluation for that view in the following update pass.

```boring
Button("Reset").onClick ():
    name = ""
    selection = nil
    error = nil
# One update pass starts after onClick returns.
```

This transaction boundary applies uniformly to direct `state` assignments, writes through `bind`,
and notifications from `'observed` mutations. A callback invoked synchronously by another callback
joins the outer event transaction; it does not trigger an intermediate rebuild. Consequently,
`body()` never observes the deliberately sequential writes of one synchronous event halfway
through that event.

An asynchronous continuation runs in a later transaction. Writes before a suspension point and
writes after resumption can therefore cause separate update passes. Multiple writes performed by
the same uninterrupted continuation are still batched. This rule does not imply rollback: if a
callback throws after making writes, those writes remain and the affected views update before the
error is presented or propagated according to the event error policy, which remains to be designed.

Invalidation during an active update pass is queued for a subsequent pass rather than recursively
re-entering `body()`. The runtime should diagnose an update loop that repeatedly mutates state while
building or updating views; the exact limit and diagnostic remain open.

**Why refresh cost is provably bounded for an exclusive field, not just usually small.** An
`'observed` field's subscriber count is genuinely dynamic — any number of views may have referenced
and subscribed to it, so a write there must lock and walk a real list. An exclusive field's
subscriber count is **provably at most one** — the qualifier itself guarantees no second owner can
exist to subscribe — so the compiler doesn't just skip building a list, it can skip lock and list
*machinery* entirely and still be correct: same reasoning as monomorphizing a turbofish call site
once the concrete type is statically known (`docs/book.md`, "Turbofish monomorphization"), applied
here to "statically known there is exactly one observer" instead of "statically known the concrete
type."

**Rejected alternatives, for the record:**
- `@State` implying an invisible `'actor` underneath, with no separate representation choice at all
  — rejected: hides a representation decision from the visible type.
- `'actor'state` as a compound suffix with one fixed representation — rejected once cross-view
  sharing required a genuine, separate subscriber-list mechanism, which briefly argued for `'state`
  as a *standalone* qualifier with its own fixed mapping instead of a suffix.
- `'state` as that *standalone* qualifier, aliasing to `'inline'observed`/`'owned'observed` —
  superseded once it became clear the mechanism has zero effect on Rust representation and
  therefore doesn't belong in the qualifier system at all — it is a `view` field keyword.
- `@state` gating whether an `'observed` field subscribes at mount — the version immediately before
  this one. Reasoned away on review: the write path of an `'observed` field must lock and walk its
  subscriber list regardless (to notify *other* observers), so gating this view's own membership in
  that list never saved the cost it looked like it saved. Superseded by unconditional subscription
  for every `'observed` field a view declares — matching SwiftUI's classic `ObservableObject`
  behavior (see above), not a compromise.
- A global `dict`/table keyed by the observed instance, valued by observing views — rejected in
  favor of the subscriber list traveling *with* the value itself (like a refcount already does),
  which also directly handles an `'observed` handle passed through arbitrary unrelated functions
  before landing as a field on some other, unrelated view.
- "Any field write on a `view` refreshes it," no marker at all, for *every* field including
  exclusive ones — rejected: over-triggers on incidental, non-rendered local fields (a debounce
  counter, a memoized helper value) — a risk judged more likely for arbitrary local fields than for
  a deliberately-referenced shared model, which is why the two cases ended up with different
  defaults (opt-in `state` for exclusive fields, unconditional for `'observed` ones).

### 4. Identity & lifecycle

Identity associates ephemeral view values with persistent state, focus, subscriptions, and render
resources across rebuilds.

- **Static structural identity**: type + static position in the block-array tree. A `TextField` at a
  fixed textual position in `Column:` is the same logical instance across rebuilds while its
  structural path remains present. Conditional alternatives occupy distinct structural branches;
  changing branch unmounts the old subtree and mounts the new one, even if their leaf view types
  happen to match.
- **Dynamic identity**: a `for` inside a view array-block identifies each iteration either through
  the element's `Identifiable` conformance or through an explicit `with` expression:

  ```boring
  # Uses item.id supplied by Identifiable.
  for item in items:
      ItemRow(item)

  # Works for foreign/scalar types, or when this screen needs a different identity.
  for item in items with item.id:
      ItemRow(item)
  ```

  `with item.id` is evaluated once per iteration after `item` is bound. The result must support
  equality and hashing and must remain stable while the element represents the same logical item.
  The explicit expression takes precedence over an available `Identifiable` conformance.
- The key identifies the **iteration subtree**, not the first child expression or `ItemRow` itself.
  If one iteration emits several sibling views, their static positions below that keyed root
  distinguish them. Reordering keyed iterations therefore preserves their local state, focus, and
  resources. Removing a key unmounts it; inserting the same key later starts a new lifetime.
- The `with` extension is valid only for `for` statements collected by a trailing array-block whose
  element type is a view trait object. An imperative `for ... with ...` is a compile error because
  iteration identity has no meaning there. Existing tuple destructuring remains unchanged:
  `for item, index in pairs:` still binds two fields from each iterated element.
- Keys must be unique among the sibling iterations produced by the same `for`. Duplicate dynamic
  keys cannot be proven invalid statically in general, so the runtime must detect them and reject
  that rebuilt subtree with a clear diagnostic rather than reconcile arbitrarily. The exact
  production error-reporting policy remains open.

`Identifiable` will be a standard trait exposing an `id`-shaped value subject to the same equality,
hashing, and stability requirements. Its exact associated-type/member spelling remains to be
designed; conformance is a convenience, never a requirement on collection element types because
the explicit `with` form covers types the application cannot or should not modify.

- **Mount / update / unmount**: derived from diffing a freshly-rebuilt `body()` tree against the
  previous one. A new identity mounts resources and observed subscriptions; an identity present in
  both trees updates in place; a missing identity unmounts resources, drops subscriptions, and
  invalidates its state/bind handles. One diffing pass drives all three concerns.
- An update retains subscriptions only while the referenced observed instances remain identical.
  If an input changes from observed model A to B, update drops A's `Subscription` and subscribes to
  B before evaluating the view from the new inputs.

#### Lifecycle and identity-scoped tasks

**Design decision (2026-09-30):** lifecycle behavior is expressed by ordered view modifiers, not
reserved methods on a `view` declaration. Each modifier wraps the preceding view and owns its own
structural identity and lifetime.

```boring
Content().onMount ():
    analytics.opened()

Content().onUnmount ():
    analytics.closed()

Content().onTask () task:
    let result = fetchData().value
    data = result

Content().onTask(id: query) () task:
    results = search(query).value
```

The `task` modifier on the trailing closure is Boring's existing async-function modifier; it makes
the callback a task context in which awaiting a `Future` through `.value`/`.wait` is legal.

- `onMount` runs once after the wrapper identity and its subtree have mounted.
- `onUnmount` runs once while captured external/bound inputs are still valid, immediately before
  the wrapper releases its resources. Its own disappearing `state` can be read for cleanup but a
  write to that state cannot schedule another render.
- `onTask` starts after mount. An ordinary rebuild of the same identity does not restart a finished
  or running task. Unmount requests cancellation and releases the task handle.
- `onTask(id: value)` additionally stores the supplied value. When it changes by equality, update
  requests cancellation of the old task and starts the new task with the rebuilt inputs. The id
  must be stable/hashable under the same broad requirements as dynamic view identity, but it is a
  restart token, not part of the view's structural identity: changing it does not reset unrelated
  state in the subtree.
- Cancellation is cooperative while the task is running, but slot handles also carry a mount
  generation. A late continuation that attempts to write after unmount cannot mutate recycled or
  unrelated state. In development it produces a diagnostic; in production the stale write is a
  no-op. Writes completed before cancellation keep their ordinary effect.
- Multiple lifecycle modifiers are allowed. Their nesting/order determines mount and unmount order;
  the exact ordering (outer-to-inner on mount and the reverse on unmount is the proposed rule) must
  be covered by conformance tests. Boring's existing restriction on chaining *after* a multiline
  trailing closure still applies, so complex chains may require inline/named callbacks or an
  extracted child view; no extra parser exception is implied by this API.

Errors from `onTask` must not disappear silently. The default reporting path, an optional local
error callback, and whether cancellation is catchable by user code remain to be specified.

### 5. Native escape hatch — two flavors, not one

- **Native view** — a persistent widget bridging a real OS control (flagship case: `TextField`).
  Implements the ordinary `View`/widget trait like any custom-rendered widget — nothing special in
  Boring syntax, the bridging is entirely inside its Rust implementation. Critically: only the
  accepted value round-trips through the app's `bind` storage — transient
  editing state (cursor, selection, IME composition, native undo) stays inside the native widget
  between renders and is never modeled in Boring at all. Over-round-tripping every keystroke would
  defeat the entire point of bridging to native in the first place.
- **Native action** — a one-shot OS interaction (photo picker, share sheet, a permission prompt).
  Not a view at all — needs no new mechanism, maps directly onto Boring's existing `task ... throws`:
  ```boring
  task string pickPhoto() throws:
      # bridge to UIImagePickerController / NSOpenPanel / etc.
  ```

### 6. MVU — demoted to a userland pattern, not a language mechanism

Model/Message/`update` (à la Elm/`iced`) was seriously considered — `req`/`def`'s existing
read-only/mutating split maps onto it almost for free. Superseded once the async-command problem
surfaced (`update` needing a `Command`/`Task`-returning escape hatch to trigger e.g. a native photo
picker and feed its result back) — `state` resolves that case with no extra machinery (the event
handler closure is itself the `task`, and assigns directly to the field when done). MVU remains
fully expressible by hand (plain `struct` + `enum` + `match`) for anyone who wants a centralized
reducer, but it isn't a first-class language feature here.

## Layout and modifier composition

**Design decision (2026-09-29):** follow SwiftUI's composition model. Both modifier orders are
legal, but they need not produce the same result. Layout and decoration modifiers wrap the
previous view rather than set unordered properties on one shared style object.

```boring
Text("Bonjour").width(200).padding(10)
# Inner fixed-width frame: 200; outer padding: 10 on each side; total width: 220.

Text("Bonjour").padding(10).width(200)
# Outer fixed-width frame: 200; padding occupies 20 within that width.

Text("Bonjour").border(red).padding(10)
# Border around the text, with padding outside the border.

Text("Bonjour").padding(10).border(red)
# Border around the padded region.
```

These are API sketches, not implemented signatures. Widths are layout units; the mapping to
physical pixels remains to be specified. The examples describe layout bounds, not clipping:
overflow and clipping behavior still need their own rules.

- Component-specific methods configure that component, for example `Text(...).font(...)`.
- General layout and decoration modifiers compose wrappers and are available on every view.
- Conceptually, the first width example yields `Padding<Width<Text>>`, the second
  `Width<Padding<Text>>`. Concrete generic wrappers versus an erased representation remain an
  implementation choice; a uniform return type must not erase the ordered semantics.
- The availability of component-specific methods after a general modifier, repeated-modifier
  behavior, and exact Boring return types remain to be designed.

This follows Apple's documented rule that modifiers wrap preceding results and that their order
matters: [Configuring views](https://developer.apple.com/documentation/swiftui/configuring-views).
SwiftUI is the semantic reference for subsequent design; departures should identify a concrete
benefit for Boring rather than merely accommodate current language limitations.

### Size negotiation and default sizing

**Design decision (2026-09-29):** content components size themselves to their content by default;
filling available space is explicit. Content size is evaluated against the proposal (a text can
wrap), not necessarily its unconstrained size. Flexible layout primitives such as `Spacer` have
their own explicit expansion role.

Layout follows three steps: the parent proposes a size, the child reports its chosen size, and
the parent places the child using that size and the requested alignment. A proposal is not a hard
constraint: a child may report a larger size. Overflow does not implicitly clip content; clipping
requires an explicit modifier.

Each axis can be unspecified, independently of the other axis. An unspecified dimension requests
an ideal/content size and is distinct from both zero and infinity. The internal representation
and whether infinity is accepted as an explicit proposal remain open.

| Element | Layout contract |
|---|---|
| `Text` | Measures its content under the proposed width; constrained width can cause wrapping. Line limits and truncation APIs remain open. |
| `width(200)` | Proposes width 200 to its child, passes through the height proposal, and reports width 200 with the child's measured height. Centers the child horizontally by default; alignment will be configurable. Does not clip an overflowing child. |
| `padding(10)` | Subtracts 20 from each specified proposal dimension, clamped to zero, measures the child, then adds 20 to each reported dimension. An unspecified proposal dimension stays unspecified. Places the child at the padding inset. |
| `Row` | Allocates horizontal space by priority and flexibility (below) and aligns children vertically. |
| `Column` | Allocates vertical space by priority and flexibility (below) and aligns children horizontally. |
| Overlay stack | Proposes a common available space to its children and places them using alignment. Its reported-size rule remains open. |
| `Spacer` | Expands on the containing stack's main axis under the allocation rules below. Its default minimum size remains open. |

### Frame constraints and explicit filling

**Design decision (2026-09-30):** keep finite frame constraints separate from the request to fill
available space. Unlike SwiftUI's `frame(maxWidth: .infinity)`, boring-ui does not use an infinite
numeric-looking value to mean "accept the parent's finite proposal." Filling is expressed by its
own modifier.

```boring
Text("Bonjour").frame(width: 200)
Text("Bonjour").frame(minWidth: 100, maxWidth: 300)
Text("Bonjour").frame(
    minWidth: 100,
    idealWidth: 200,
    maxWidth: 300,
    alignment: leading,
)

Text("Bonjour").fillWidth(alignment: leading)
Text("Bonjour").fillHeight()
Text("Bonjour").fill()
```

`frame` accepts optional finite `width`/`height`, `minWidth`/`minHeight`,
`idealWidth`/`idealHeight`, and `maxWidth`/`maxHeight` constraints. Supplying a fixed dimension is
shorthand for setting that axis's minimum, ideal, and maximum to the same value. Invalid ranges
(for example `minWidth > maxWidth`) are errors; whether statically knowable invalid ranges are
checker errors and dynamic ones are runtime errors remains to be specified.

Convenience modifiers preserve the same ordered wrapper semantics and desugar to `frame`:

```boring
Text("Bonjour").width(200)
Text("Bonjour").height(40)
Text("Bonjour").size(width: 200, height: 40)
```

`fillWidth`, `fillHeight`, and `fill` accept the finite size proposed by their parent on the
corresponding axes. They do not manufacture space when the proposal is unspecified, and do not
clip a child that reports a larger minimum size. Their alignment argument controls placement of
the child inside the extra space; it defaults to center. A fill modifier participates as a flexible
child in stack allocation and remains subject to frame minimum/maximum constraints applied outside
it. Because modifiers are ordered, applying constraints inside versus outside a fill wrapper can
produce different results.

The exact `Length` type, handling of negative values, percentage/relative units, and interaction
between contradictory nested frames remain open. Percentages are not required for the first
implementation target.

### Stack allocation: priority and flexibility

**Design decision (2026-09-29):** follow SwiftUI's negotiation by layout priority and flexibility,
rather than first reserving every child's natural size and then dividing only the surplus.

- Subtract inter-child spacing from the available main-axis space before allocating to children.
- Support layout priority, defaulting to zero. Higher-priority children receive preferential
  allocation, so they resist compression and obtain expansion space before lower-priority peers.
  Priority does not make an intrinsically fixed-size child stretch.
- Within the same priority, negotiate with less flexible children first. Propose a share of the
  remaining space, deduct the size actually reported, and recalculate the share for the remaining
  children. Measurement order does not change their visual order.
- Equal sharing is a special case for equally flexible children with equal priorities and
  equivalent constraints, such as two identical `Spacer` instances. It is not a universal
  guarantee for every combination of a filling view and a spacer.
- The same negotiation must account for insufficient space; do not assume that every child's
  unconstrained content size can be reserved. A child still determines how it responds (for
  example, text wrapping). Allocation alone does not introduce clipping.

For example, in a row of width 400 with zero spacing, two fixed-width children of 80 leave 240
for one spacer, or 120 each for two identical spacers, provided their minimum sizes allow this.

Sources: Apple's [stack layout explanation](https://developer.apple.com/videos/play/wwdc2019/237/)
and [layout priority contract](https://developer.apple.com/documentation/swiftui/view/layoutpriority(_:)).
These establish the design direction, not a promise to reproduce every internal SwiftUI detail.

Still specify the flexibility measurement protocol, allocation across priority groups (including
minimum-size reservations), ties and rounding, proposals with unspecified main-axis size, and
overflow when children's minimum sizes exceed available space. The signatures for priority,
explicit filling, and clipping also remain open.

### Stack alignment and spacing

**Design decision (2026-09-30):** follow SwiftUI's axis-specific alignment and adaptive default
spacing. Omitting spacing is semantically different from specifying zero.

```boring
Row(alignment: center, spacing: 8):
    Text("Name")
    TextField(value: name)

Column(alignment: leading):
    Text("Title")
    Text("Details")
```

- A `Row` aligns children on the cross (vertical) axis and defaults to `center`.
- A `Column` aligns children on the cross (horizontal) axis and defaults to `center`.
- An omitted `spacing` asks the active theme for adaptive spacing based on the adjacent component
  pair and platform conventions. `spacing: 0` explicitly removes the gap; any other finite,
  non-negative value is an exact gap in layout units.
- Horizontal `leading` and `trailing` are direction-aware and reverse in right-to-left layout.
  Absolute `left` and `right` alignments may exist for the uncommon cases that require physical
  direction, but application layout should normally use `leading` and `trailing`.
- Rows support text baselines, at minimum `firstBaseline` and `lastBaseline`. A component can
  publish the corresponding alignment guides; a component without a requested guide falls back to
  a specified default derived from its bounds. The exact fallback remains open.
- Adaptive spacing is resolved between each adjacent pair after conditional/dynamic children have
  produced the current child list. Explicit stack spacing overrides every pair uniformly in the
  initial API; per-gap overrides are deferred.

Theme spacing and alignment guides are layout metadata, independent of visual styling. The first
implementation may ship one platform-neutral default table, but the API must leave room for
platform themes without changing application code.

### Overflow, clipping, and scrolling

**Design decision (2026-09-30):** layout bounds do not implicitly clip drawing and overflow does
not implicitly create scrolling. Both behaviors require explicit composition.

```boring
Image(photo).frame(width: 100, height: 100).clip()
Image(photo).frame(width: 100, height: 100).clip(RoundedRect(radius: 8))

Scroll(axis: vertical):
    Column:
        for item in items with item.id:
            ItemRow(item)
```

A child that reports or draws beyond the size proposed by a parent keeps its reported layout size
for placement. Drawing outside those layout bounds can overlap nearby content, but does not push
siblings away and does not enlarge an ancestor after layout. Hit testing follows the rendered
content by default; whether an ancestor can restrict hit testing independently of visual clipping
remains open.

`clip()` wraps a view and restricts its rendering to the wrapper's rectangular bounds. Supplying a
shape clips to that shape within the same bounds. Like every wrapper modifier, its position in the
modifier chain matters. The initial implementation need not support arbitrary user-defined clip
paths, but the public contract should not rule them out.

`Scroll(axis: vertical)` creates a finite viewport from the proposal it receives. It proposes an
unspecified height to its content and passes through the available width; the horizontal form does
the symmetric operation. Its content can therefore choose its full extent on the scrolling axis,
while the viewport reports a finite size to its parent and clips rendering at its own bounds.
Two-axis scrolling can be added as `axis: both`; whether it belongs in the first implementation is
open because completely unspecified proposals on both axes interact poorly with wrapping content.

Scroll input, momentum, overscroll behavior, scrollbar visibility, keyboard accessibility,
programmatic position, restoration, and nested-scroll arbitration remain to be designed. Native
platform conventions should supply defaults, with explicit APIs only where applications need
control. Lazy child creation is a separate container concern (`LazyColumn`/`LazyRow` or a keyed
list), not an automatic property of `Scroll`.

## First implementation target and remaining design work

The first end-to-end target is a small editable list application: add an item, edit it, select it,
delete it, and reorder it while preserving the state of surviving rows. Define its syntax and
observable behavior before implementation; a complete future widget catalogue is not required.

Proposed minimum component set (names and signatures still open):

| Area | Components |
|---|---|
| Application and layout | Window, `Column`, `Row`, overlay stack, `Spacer`, separator |
| Display | `Text`, `Image` |
| Input | `Button`, `TextField`, checkbox or toggle |
| Dynamic content | Scroll container, keyed list |
| Presentation | One dialog or modal mechanism |

For each component, specify parameters and defaults, bindings, events, size negotiation, focus
and keyboard behavior, and accessibility semantics, with Boring usage examples.

Before developing this vertical slice, resolve these shared contracts:

1. **State and inputs:** ephemeral view values versus identity-scoped persistent storage; local
   initialization versus parent-provided inputs; initialization from changing parent parameters;
   callback access to persistent state. The working direction is implicit persistence for an
   observed model declared and created locally, with parent-supplied references updated as inputs.
   The exact syntactic distinction remains open. Earlier claims that `state` costs only an inserted
   refresh call are incomplete: persistent storage and access to it must also be specified.
2. **Identity:** structural identity, conditional branches, keyed dynamic rows, and state reset on
   removal are defined above. Still specify the exact `Identifiable` trait and duplicate-key
   production diagnostics.
3. **Layout:** size negotiation, content-sized defaults, finite frame constraints, explicit filling,
   ordered modifiers, priority/flexibility-based stack negotiation, axis-specific alignment, and
   adaptive spacing, explicit clipping, and single-axis scrolling are defined above. Still resolve
   the detailed allocation/compression algorithm, baseline fallback, scroll behavior and control,
   hit testing outside bounds, and arbitrary clip shapes.
4. **Events and bindings:** `bind` ownership, contextual projection, mutation permissions, and
   synchronous-event batching are defined above. Still specify callback captures, UI execution
   context, event propagation, and error handling.
5. **Lifecycle and async:** mount/unmount modifiers, identity-scoped task start/restart/cancellation,
   stale-write protection, and observed subscription replacement are defined above. Still specify
   task error reporting, cancellation observability, and exact nested-hook ordering.

## Comparative table — SwiftUI → boring-ui

| SwiftUI | Role | boring-ui equivalent |
|---|---|---|
| `struct MyView: View` | declares a view | `view MyView:` |
| `var body: some View` | declarative body | `body():` method, returning a widget tree via the array-block sugar |
| `@State private var x` | view-local observed state | `state var x = ...` — exclusive by default (no `'observed`), persistent and private by construction (§3) |
| `@Binding var x: T` | read/write reference to a parent's state, not owned here | `bind var T x` field; the parent passes its `state`/`bind` lvalue directly and the expected field contract creates the persistent slot handle (§3) |
| `@ObservedObject var model: Model` | external reference to a shared, not-owned-here model | struct + `mut Model'observed model` field, populated via an `init(Model'actor'observed model): self.model = model` constructor parameter rather than constructed by this view — both the field and the parameter are real, shipped `'observed` syntax today (`docs/book.md`'s "'observed" section) — subscribes at mount unconditionally, no `state` needed |
| `@StateObject var model = Model()` | model *owned* by this view, created once, survives rebuilds | `state mut Model model = Model()` (no `'observed` — exclusive, and `state`'s direct-refresh path is the entire mechanism) constructed inline in the view declaration — "created once" follows from its identity-keyed persistent slot, not the throwaway rebuilt value |
| `ObservableObject` / `@Published` | observable model, Combine-driven, property-level in the newer `@Observable` macro | plain `struct` + a `'observed`-qualified reference on the referencing view's field — see §3 |
| `Text`, `Button`, `VStack`, `HStack` | base widgets | `Text`, `Button`, `Column`, `Row` — to be written as the actual `boring-ui` stdlib |
| `@ViewBuilder` (implicit result builder on `body`) | lets `body` read as nested indentation | the array-block sugar (§1, shipped) |
| `.onTapGesture { }` / `Button(action:)` | interaction callback | `.onClick (): ...` — existing trailing-closure sugar |
| `TextField("...", text: $name)` | native-backed input bound to state | `TextField(value: name)` where `value` is declared `bind var string`; transient native editing state stays in the control (§3, §5) |
| `List(items) { }` / `ForEach(items, id: \.id)` | dynamic content with stable per-element identity | `for item in items with item.id:` inside the array-block sugar; omit `with` when the element implements `Identifiable` (§4) |
| `Identifiable` protocol | supplies the stable identity `ForEach` needs | standard `Identifiable` trait with an `id`-shaped member; its exact declaration syntax remains open (§4) |
| `.onAppear { }` / `.onDisappear { }` | lifecycle hooks | ordered `.onMount ():` / `.onUnmount ():` view modifiers scoped to the wrapper identity (§4) |
| `.task { }` / `.task(id:)` | lifecycle-scoped async action | `.onTask () task:` / `.onTask(id: value) () task:`; starts on mount, restarts on id change, and cancels on unmount (§4) |
| `@EnvironmentObject` / `.environmentObject()` | implicit injection down the tree, bypassing per-parameter threading | **not a boring-ui-specific mechanism, implemented and shipped** — see [book.md §33](../book.md#33-dependency-injection), a standalone dependency-injection/IoC design (`@inject` field + a `@provide`-attributed composition-root function, compile-time-resolved, no runtime reflection). Covers the "resolve without naming the concrete type, across a module boundary" half of this row for free (an `@inject`-annotated `'actor'observed` field is the direct answer to this row's shared-model case); true per-subtree ambient overrides (a different value for one branch of the view tree, à la `.environment(\.x, y)`/Compose `CompositionLocalProvider`) remain an open question left to `boring-ui` to design on top of it, not solved by it. |
| `.sheet(isPresented:) { }` / navigation push | modal presentation / pushing a new view | **open question** — unclear whether this is "a view mounted dynamically" or closer to a native action (§5); not decided |

Rows marked "open question" are genuine gaps, not settled design compressed for brevity — treat
them as the next things to work through, not as implied answers.

## Where this came from

This draft consolidates a long design conversation (2026-09-13 onward) that started from "what
should Boring's GUI story look like, given SwiftUI/Kotlin/Flutter as reference points" and converged
step by step: custom rendering over native-widget wrapping → the array-block sugar (shipped) → MVU
considered and superseded by an explicit reactive mechanism → that mechanism's design revised three
times as its real requirements became concrete — first a suffix on `'actor` (assumed one fixed
representation), then a standalone qualifier with an embedded subscriber list (once cross-view
sharing required a genuine notification mechanism, not just a compile-time hook), then finally split
into two independent, orthogonal answers once it became clear *sharing* (`'observed`, a composable
qualifier suffix, general-purpose) and *"does this specific field refresh this specific view"*
(`state`, a view-local persistent field keyword) were never the same question. `bind` was then
added as the corresponding non-owning, write-through view field for parent-owned reactive storage.
