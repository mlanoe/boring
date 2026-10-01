# Draft — boring-ui: a SwiftUI-flavored, custom-rendered GUI toolkit for Boring

Status: **working draft**, not a spec. Nothing in this document is implemented yet except the
language mechanisms explicitly marked as shipped below (§1's array-block sugar, and `'observed`
in §3 — including, as of this update, struct fields/parameters/return types, not just local
bindings). It now defines the design basis and acceptance criteria for a first implementation;
later-scope rows in the comparative table may remain open rather than being guessed at.

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
  renderer; the existing Bevy applications prove the graphics stack on the target platforms but
  are not themselves the UI runtime),
  not SwiftUI's actual strategy of bridging to real AppKit/UIKit widgets everywhere. A **native
  service/action escape hatch** (below) exists for the specific, narrow cases where the operating
  system must participate (text input/IME, pickers, dialogs), not as the default.
- **Not a drag-and-drop editor.** Code-first, like SwiftUI's own text-based `body`, not a visual
  designer.

## Rendering model, in one line

Custom rendering (own arbitrary widget tree, own paint pipeline, wgpu-based and independent of
Bevy) chosen over native-widget wrapping, because the latter
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
    state string name = ""

    body():
        Column:
            TextField(value: name, label: "Name")
            Button("Submit").onActivate (): print "submitted {name}"
```

`state` is a `view`-specific field-declaration keyword, not an attribute or a qualifier (§3).
The omitted binding permission defaults to `var` for this scalar; explicit `let`/`var`/`mut` forms
and the defaults for other value categories are specified below.

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
    state int count = 0

let counter = Counter(title: "Inbox", count: 10)  # seeds count on first mount
counter.count = 11                                 # compile error: state is private
```

The constructor argument is an **initial-state seed**, not a parent-controlled input. On the first
mount of an identity it initializes the persistent slot; rebuilding the same identity does not
overwrite that slot with a newly supplied seed. Ordinary non-`state` view fields are inputs and are
updated from the rebuilt value. This distinction must be visible in diagnostics, because silently
treating a changing parent argument as state would otherwise be surprising.

When the seed type implements equality, rebuilding an existing identity with a seed different from
the value recorded at first mount emits a development warning at the call site and keeps the
persistent slot unchanged. The comparison is against the original seed rather than the current
state value. Repeating the same mismatch for the same mounted identity is deduplicated. A type that
cannot be compared receives no runtime seed-change check; this never changes semantics and callers
that intend parent-controlled updates must use an ordinary input or `bind`. Release builds may omit
the warning and stored comparison value.

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
    state string name = ""

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

##### Implicit and explicit binding permissions

**Design decision (2026-10-01):** omitting `let`/`var`/`mut` after `state` or `bind` selects the
ordinary useful permission for the value category. Value-like types (primitives, strings, enums,
and options) default to `var`, because their change operation is replacement. Structs and mutable
collections default to `mut`, because their usual change operation is in-place mutation.

```boring
state string name = ""                 # state var string
state Field? focus = nil               # state var Field?
state Profile profile = Profile()      # state mut Profile
state [Item] items = []                 # state mut [Item]

bind string name                       # bind var string
bind Profile profile                   # bind mut Profile
bind Field? focus                       # bind var Field?
```

The explicit forms override the default without changing the underlying value representation:

```boring
state var Profile profile = Profile()      # replacement only
state var mut Profile profile = Profile()  # replacement and in-place mutation
state let Token token = Token.random()     # seeded once, then immutable

bind var Profile profile
bind var mut Profile profile
```

`state let` is meaningful identity-scoped storage: it evaluates its seed only on first mount and
retains the resulting immutable value across rebuilds (for example a random token or expensive
immutable resource). It never schedules refresh because it cannot be written. `bind let` is
rejected: a read-only parent value is an ordinary view input and needs no persistent write-through
handle.

The checker uses a closed, predictable default classification rather than behavioral inference:

- primitives, strings, enums, newtypes, tuples, function values, and every optional type default to
  `var`;
- structs, arrays, dictionaries, and sets default to `mut`;
- trait objects and opaque/external values require an explicit permission when the checker cannot
  classify their mutation surface.

Generic parameters use the declaration's constraint when it proves one category; otherwise they
also require an explicit permission. The checker never changes a default because it notices a
particular method call later in the body. Explicit `let`/`var`/`mut`/`var mut` remains available for
every otherwise valid category and is the required escape hatch for unusual APIs.

An unqualified `'observed` struct field remains outside this sugar and follows the existing
`'actor`/`'guard` permission table: `mut` unlocks mutating `def` calls, `var` permits replacement,
and `var mut` permits both.

##### `bind` — a view input that writes through to reactive storage

**Design decision (2026-09-30):** `bind` is a second `view`-specific field-declaration keyword. It
expresses a non-owning, persistent read/write connection to reactive storage owned by an ancestor,
without exposing a general-purpose `Binding<T>` type or SwiftUI-style `$` projection in source.

```boring
view NameEditor:
    bind string name

    body():
        Column:
            Text(name)
            TextField(value: name, label: "Name")

view FormView:
    state string name = ""

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
bind string name              # defaults to var; assignment replaces the parent's scalar value
bind Profile profile          # defaults to mut; `def` calls modify the parent's struct in place
bind var Profile profile      # explicit replacement-only struct binding
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
TextField(bind string value, string label, string placeholder = "")
Toggle(bind bool value)
Slider(bind float value, float min, float max)
```

Native/Rust-implemented components use the Boring facade plus `body(): native` convention specified
under Runtime architecture; their visible signature and checker behavior exactly match a `bind`
field.

##### UI events and batched refresh

**Design decision (2026-09-30):** reactive writes invalidate views immediately but defer rebuilding
until the outermost synchronous UI event callback returns. The runtime deduplicates invalidated view
identities, so any number of writes affecting the same view during one event produce one `body()`
evaluation for that view in the following update pass.

```boring
Button("Reset").onActivate ():
    name = ""
    selection = nil
    error = nil
# One update pass starts after onActivate returns.
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
error is reported according to the event error policy below.

Invalidation during an active update pass is queued for a subsequent pass rather than recursively
re-entering `body()`. A window may perform at most 64 consecutive update passes without processing
a new external event, timer, or asynchronous wake-up. Reaching that limit reports an update-loop
error with the repeatedly invalidated view identities, clears the pending invalidation queue, and
keeps the last successfully committed tree mounted. The same safety limit applies in release builds;
only the diagnostic detail may be reduced. This catches mutation from `body()` or update hooks
without permitting recursive tree mutation or freezing the platform event loop.

##### Event dispatch, capture, and errors

**Design decision (2026-10-01):** the platform host translates input into normalized physical
events (pointer, scroll, key, text/IME, focus, and window events). Hit testing chooses a mounted
target, then dispatches a physical event along the retained path from the window root to that target
and back. Components turn physical input into semantic actions such as activate, submit, toggle,
move, or delete.

Application-facing control callbacks receive semantic actions. They are independent of the input
device: pointer release, Space/Enter, an accessibility invoke action, and an appropriate platform
command can all produce the same `Button.onActivate` callback. A semantic action belongs to its
control and does not bubble implicitly into ancestor controls. Nesting one activatable control in
another is rejected because its interaction and accessibility semantics would be ambiguous.

The physical dispatch path has three steps:

1. **capture** from the root toward the target, for framework mechanisms such as disabled state,
   modal routing, gesture arbitration, and future advanced input modifiers;
2. **target** delivery to the deepest eligible component;
3. **bubble** from the target toward the root, for unconsumed physical input and container gestures.

The first public component set does not need general `onPointerCapture`/`onKeyBubble` modifiers.
Those can be added with an explicit event-response type after custom-control use cases establish the
API. Internal dispatch must nevertheless distinguish two independent outcomes: stop further
propagation and suppress the component's default behavior. Conflating them would prevent, for
example, an ancestor observer from seeing a key that a text field has already consumed for editing.

A node participates in pointer hit testing inside its effective hit region: its layout bounds
intersected with ancestor clips and modal routing. Merely visible overflow does not enlarge the hit
region. A future `contentShape`/`hitShape` modifier may explicitly replace that region. `Layer`
tests children from front to back, the reverse of paint order; other containers use their
component-defined visual order. Disabled nodes remain in the accessibility and layout trees but
are skipped as action targets.

Pointer-down may establish pointer capture for a mounted control or gesture recognizer. Subsequent
move/up/cancel events for that pointer go to the captured identity even outside its hit region.
Unmounting it, losing the window, or beginning a modal presentation sends cancellation and releases
capture. Pointer capture is unrelated to keyboard focus.

##### Callback lifetime and captures

Callbacks declared while evaluating `body()` are ephemeral inputs to the modifier/control node.
Reconciliation replaces them on every successful update, so a later event observes the newest
ordinary inputs. An event already being dispatched keeps the callback snapshot selected at the
start of that dispatch.

References to `state` or `bind` names inside such a callback capture typed logical-location handles,
not the values read during `body()`. Access therefore sees the value at callback execution time and
retains the source permissions. Capturing a keyed loop element similarly captures its keyed locator.
Ordinary immutable values follow Boring's existing closure capture rules and are snapshots.

`self` inside a callback declared by a `view` is lowered to a mounted-view context: ordinary input
fields come from that callback's current ephemeral snapshot, while `state` and `bind` fields resolve
through their handles. It is never a Rust borrow of the temporary value whose `body()` created the
closure. This makes calls such as `self.add()` valid when the method obeys the same field permissions.
The mounted context cannot escape except inside a framework-managed callback or lifecycle task.

A synchronous callback may call another ordinary callback directly; both share the current event
transaction. Platform events and accessibility actions arriving during dispatch are queued until
the transaction and its resulting update pass finish. This prevents re-entrant tree mutation and
gives events a deterministic order. Blocking nested platform event loops are unsupported; native
actions that would create one must expose an asynchronous `task` API.

An async callback or `onTask` captures the same handles plus snapshots of ordinary values at task
start. At each suspension boundary it ends the current transaction. Its continuation is queued on
the UI executor, validates every mounted handle generation, and begins a new transaction. A parent
rebuild does not silently replace the ordinary snapshots of a running task; use `onTask(id:)` when
changed inputs must restart it.

##### Error boundary for event dispatch

Control callbacks are non-throwing unless their declared callback type explicitly includes
`throws`; the checker therefore requires ordinary throwing calls to be handled locally in the
common `onActivate` case. When a framework callback is allowed to throw, the dispatch boundary
catches the error. Writes already completed remain committed, pending invalidations are processed,
the current semantic action stops, and the error is reported to the nearest enclosing error
handler, falling back to the window/application reporter.

The initial error handler is a runtime facility rather than a replacement-view mechanism: it
receives the error, callback kind, and mounted source identity for logging or presentation. A later
`.onError` API may expose it in the view tree once its interaction with errors from `body()`, layout,
rendering, and lifecycle tasks is designed together. An unhandled callback error is logged with its
view path; development builds additionally surface it prominently. It does not terminate the
process unless the configured application reporter chooses to do so.

Panics and violated native-component invariants are programming failures, not typed callback
errors. They may abort the affected window or process according to the build/runtime policy and are
not converted into ordinary Boring `Error` values.

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
- When the iterated source is a writable `state`/`bind` collection, the loop variable retains
  writable projection provenance. It reads like an ordinary element, but a child field expecting
  `bind Element` or `bind` on one of its writable fields can receive it contextually:

  ```boring
  for item in items with item.id:
      TodoRow(item: item)  # TodoRow declares `bind Todo item`
  ```

  The runtime binding resolves through the iteration's stable key, not a permanently captured array
  index, so reordering cannot redirect a later edit to another element. A structural collection
  mutation invalidates index-based projections until reconciliation; keyed projections remain valid
  only while their key still exists. This rule is required for editable rows without reintroducing
  SwiftUI's `$items`/`$item` projection syntax.
- The `with` extension is valid only for `for` statements collected by a trailing array-block whose
  element type is a view trait object. An imperative `for ... with ...` is a compile error because
  iteration identity has no meaning there. Existing tuple destructuring remains unchanged:
  `for item, index in pairs:` still binds two fields from each iterated element.
- Keys must be unique among the sibling iterations produced by the same `for`. Duplicate dynamic
  keys cannot be proven invalid statically in general, so the runtime validates the complete key
  sequence before reconciling any iteration subtree.

#### `IdentityKey` and `Identifiable`

**Design decision (2026-10-01):** both are standard-library traits rather than language keywords.
Their Boring-facing shape is:

```boring
trait IdentityKey as Clone, Eq, Hash:
    pass

trait Identifiable:
    type ID as IdentityKey
    req ID id()
```

`IdentityKey` is a marker backed by a native blanket implementation for owned, static values that
satisfy `Clone + Eq + Hash`. The checker understands that blanket implementation when validating
the associated type and an explicit `with` expression; applications do not write an empty
`as IdentityKey` implementation for every key type. Integers, booleans, characters, strings,
suitable tuples/options, enums, newtypes, and structs deriving the required traits can be keys.
Floating-point values are not keys because NaN prevents total equality. Borrowed values and mutable
shared qualifiers are rejected; the iteration stores an owned key snapshot.

When `with` is omitted, lowering calls `item.id()` exactly once per iteration and stores the result.
There is no general property syntax added to Boring: the trait member is an ordinary `req` method.
An explicit expression such as `with item.id` may still read a real field and takes precedence over
the trait. Its result must satisfy the same `IdentityKey` bound.

Conformance remains optional. It is appropriate when one identity is intrinsic to the element;
screens needing another identity, scalar/foreign elements, or types the application cannot modify
use `with`. For example:

```boring
struct Todo as Identifiable, Eq, Hash:
    pub int id
    pub var string title

    type ID = int
    req int id(): self.id

# Intrinsic identity
for item in todos:
    TodoRow(item)

# Screen-specific identity
for result in results with (result.source, result.localId):
    ResultRow(result)
```

The key type is fixed statically for one `for` site. Hashing selects candidates and equality decides
identity, so hash collisions are harmless. Keys are compared only within one dynamic `for` site;
the static site id distinguishes equal keys used by separate or nested loops.

A key must remain stable for as long as the element represents the same logical item. Changing it
is defined as removing one identity and inserting another: local state, focus, tasks, subscriptions,
and control resources do not migrate. Mutating an element's key through a row binding can make that
row's existing keyed handles stale immediately after the mutation, so identity fields should not be
editable through the row that they identify.

#### Duplicate-key failure

The builder evaluates all iteration keys and checks uniqueness before it mutates the mounted tree.
On duplicate, the diagnostic contains the `for` source location, a printable key when available,
and the first and repeated iteration indices. It never selects the first/last element or falls back
to positional identity.

If a previously valid subtree exists, the failed update leaves that entire `for` subtree mounted
unchanged and reports the error through the window/application runtime reporter. Other already
completed state writes remain committed, but the invalid description is not partially reconciled.
The next valid rebuild can reconcile normally from the retained subtree. On initial mount, where no
previous subtree exists, the containing scene fails to mount and `runUI` reports the error; showing
an arbitrary empty list would conceal a data-integrity bug.

Development and production builds have the same reconciliation behavior. Development output may
include richer view paths and key formatting, while production may redact unavailable/debug-only
representations. Duplicate identity is a recoverable update error, not a panic and not undefined
behavior.

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

- `onMount` runs once after the wrapper identity and its subtree have committed layout, focus, and
  accessibility state. It does not mean that the GPU frame has already been presented.
- `onUnmount` runs once while captured external/bound inputs are still valid, immediately before
  the wrapper releases its resources. Its own disappearing `state` can be read for cleanup but a
  write to that state cannot schedule another render.
- `onTask` starts after mount hooks and their resulting synchronous update have completed, provided
  the identity is still mounted. An ordinary rebuild of the same identity does not restart a
  finished or running task. Unmount requests cancellation and releases the task handle.
- `onTask(id: value)` additionally stores the supplied value. When it changes by equality, update
  requests cancellation of the old task and starts the new task with the rebuilt inputs. The id
  must be stable/hashable under the same broad requirements as dynamic view identity, but it is a
  restart token, not part of the view's structural identity: changing it does not reset unrelated
  state in the subtree.
- Cancellation is cooperative while the task is running. Writes completed before cancellation keep
  their ordinary effect.
- Multiple lifecycle modifiers are allowed. Their nesting/order determines mount, task-start, and
  unmount order under the rules below. Boring's existing restriction on chaining *after* a multiline
  trailing closure still applies, so complex chains may require inline/named callbacks or an
  extracted child view; no extra parser exception is implied by this API.

#### Commit and hook ordering

Mounting first creates identities, initializes state, installs current inputs/bindings/environments,
and establishes observed subscriptions. It then builds and lays out descendants. No user lifecycle
hook runs against a partially reconciled tree.

After a successful tree commit, mount hooks run in structural preorder: parent before descendant;
for modifiers wrapping the same content, outermost before innermost. All hooks created by that
commit run in one UI transaction. Their writes are batched, then the runtime performs the resulting
update. Only after that update reaches a stable commit are `onTask` bodies for surviving identities
started, in the same preorder. Consequently, an `onMount` callback can immediately remove its own
subtree without starting work that would be cancelled at once.

Unmounting uses the reverse order: descendants before parents and innermost modifier before
outermost. Before the first unmount hook, the runtime marks the retiring subtree unavailable to new
platform events, releases pointer capture/focus as appropriate, and requests cancellation of every
scoped task in that subtree. Each synchronous `onUnmount` hook then runs with teardown-only access:
it may read its own state and read/write still-mounted ancestor bindings for cleanup, but writes to
its disappearing state are rejected and cannot invalidate the tree. After hooks finish, the runtime
drops subscriptions, control/platform resources, state slots, and finally the identities.

When an animated removal retains a subtree for its exit transition, the unavailable/capture/focus/
task-cancellation part happens when the retiring phase begins; hooks and resource destruction wait
until the transition completes. The frozen retiring subtree cannot receive new events or restart
tasks during that interval. Forced teardown skips the remaining animation and proceeds directly to
the hooks.

Sibling hooks follow structural order on mount and reverse structural order on unmount. These rules
apply to conditional removal, keyed deletion, scene close, and application shutdown. A failed mount
that never commits runs no `onMount` or `onUnmount`; resources acquired internally before commit are
released by ordinary runtime cleanup.

#### Scoped task runs and cancellation

Each start of an `onTask`/`onTask(id:)` body receives a private task-run generation in addition to
the mounted-node generation. State/bind access from its continuation is valid only while both match.
When an id changes, the runtime increments the task-run generation before requesting cancellation
of the previous run, then starts the replacement after the current update commits. This prevents a
late result from the old query from overwriting the new query's state even though the view itself
never unmounted.

Lifecycle cancellation uses Boring's existing `Future.cancel()`/`Error.Cancelled` mechanism. A
cancellation-aware suspension observes the request and the task may catch `Error.Cancelled` for
non-UI cleanup. Suppressing cancellation does not restore its task-run capability: reads or writes
through mounted handles from that obsolete run are stale. The runtime never waits synchronously for
a task during reconciliation or shutdown, so a task that ignores cancellation cannot block the UI;
its future and captured non-UI values live until it exits or the executor is torn down.

Unmount invalidates the mount generation as teardown finishes. An old continuation then cannot
mutate recycled or unrelated state. In development, stale access reports the task and view source
locations; in production a stale write is a no-op and a stale read terminates that continuation as
specified by the slot-handle contract.

Cancellation initiated by lifecycle removal or id replacement is an expected completion and is not
reported as an application error when it reaches the task boundary. If user code catches it and
throws another error, the replacement error is reported normally. Explicit cancellation of an
unrelated application `Future` retains ordinary Boring semantics.

#### Task errors

An `onTask` body may handle errors locally with Boring's existing `try`/`catch`. Any uncaught error
other than expected lifecycle cancellation is delivered to the same window/application runtime
reporter as callback errors, with the mounted view path, modifier source location, and task id/run
generation when present. The task then remains finished; an ordinary rebuild with the same identity
and id does not retry it. Changing the id or remounting creates a new run.

The first runtime has no implicit retry, replacement view, alert, or process termination. The
application reporter decides presentation and fatality. A future view-tree `.onError` modifier can
intercept both callback and task errors once the unified boundary API is designed; adding it will
not change cancellation or restart semantics above.

### 5. Native boundary — services and actions, not embedded widgets

**Design decision (2026-10-01):** `TextField` is laid out, painted, hit-tested, focused, and exposed
to accessibility by boring-ui like every other control. It is not an AppKit/UIKit/Win32/GTK widget
superposed on the GPU surface. Embedded native controls would introduce a second layout, clipping,
z-order, animation, theming, and accessibility tree precisely at the most frequently composed
control.

Text entry still uses a **native text-input service**. The focused field supplies its editable text,
selection, surrounding-text context, input purpose, and caret rectangle; the platform adapter
returns committed text, marked/preedit text, selection changes, and keyboard commands. Desktop
adapters may use the window system's IME events. Mobile adapters additionally own the platform
text-input client needed to show the software keyboard. This is a retained service attached to the
focused mounted identity, not a visible view and not part of Boring syntax.

The field keeps transient editing state — cursor, selection, marked text, scroll offset, and undo
grouping — in its mounted control state. A committed edit updates its `bind string value` in the
current event transaction. Parent updates replace the editing buffer only when the bound value
actually differs; the text contract must define selection preservation for that case. Text shaping,
bidirectional layout, grapheme navigation, and glyph rasterization belong to the shared text engine.

A **native action** is a one-shot OS interaction such as a photo picker, share sheet, file dialog,
or permission prompt. It is not a view, needs no new mechanism, and maps directly onto Boring's
existing `task ... throws`:
  ```boring
  task string pickPhoto() throws:
      # bridge to UIImagePickerController / NSOpenPanel / etc.
  ```

If a future integration truly needs to embed a native view (web view, map, camera preview), it must
be designed as a separate platform-surface feature with explicit clipping/compositing limitations.
It is not part of the first UI runtime.

### 6. MVU — demoted to a userland pattern, not a language mechanism

Model/Message/`update` (à la Elm/`iced`) was seriously considered — `req`/`def`'s existing
read-only/mutating split maps onto it almost for free. Superseded once the async-command problem
surfaced (`update` needing a `Command`/`Task`-returning escape hatch to trigger e.g. a native photo
picker and feed its result back) — `state` resolves that case with no extra machinery (the event
handler closure is itself the `task`, and assigns directly to the field when done). MVU remains
fully expressible by hand (plain `struct` + `enum` + `match`) for anyone who wants a centralized
reducer, but it isn't a first-class language feature here.

## Application entry point and scenes

**Design decision (2026-09-30):** application startup is an ordinary blocking library function
named `runUI`, not a new `app` declaration or language keyword.

```boring
use boring_ui.runUI
use boring_ui.Window

def main():
    runUI:
        Window(title: "Todos"):
            TodoList()
```

Conceptually, `runUI` accepts `[dyn Scene]`, initializes the UI runtime and its main executor, and
returns when the last application window closes. It must be called once from the process entry
point. Accepting a scene collection permits multi-window applications without imposing a separate
application type on the common case.

`Window` is a library-provided `Scene` and accepts an array-block of views. In the initial API it
requires exactly one root view so that the scene never invents a layout for siblings:

```boring
Window(title: "Todos"):
    Column:
        Toolbar()
        TodoList()
```

A static block with zero or several roots is a checker error. If conditional collection prevents
the checker from proving cardinality, mounting validates it and reports the same error; applications
should put conditional roots inside an explicit `Column`, `Row`, `Layer`, or `Group` with
defined semantics. On mobile, `Window` represents a platform scene and unsupported presentation
options such as a desktop title may be ignored. Further `Scene` implementations can be added as
library types without changing `runUI` or the language.

## Runtime architecture

**Design decision (2026-10-01):** boring-ui is a standalone runtime, not a Bevy plugin. Existing
Bevy projects demonstrate that Boring-generated Rust can open windows and render on the intended
graphics stack, but Bevy's world/ECS schedule is not the ownership or lifecycle model specified by
`view`, `state`, and keyed reconciliation. Depending on it would also make the smallest application
carry a game engine runtime.

The implementation has five layers with one-way dependencies:

1. **Language lowering** turns `view` declarations and modifier calls into ephemeral view
   descriptions, typed state-slot access, contextual `bind` projections, and stable static site
   identifiers. It contains no windowing or GPU code.
2. **Core runtime** owns the mounted tree, identity reconciliation, persistent slots, environments,
   invalidation, event transactions, focus, lifecycle hooks, and task handles. It must run headless
   in tests.
3. **Layout and scene building** measures and places nodes, performs hit testing, and emits a
   backend-neutral display list plus an accessibility tree. Controls contribute semantics and draw
   commands; they do not call wgpu directly.
4. **Renderer** consumes the display list, manages GPU resources, glyph/image caches, clipping, and
   compositing. A first implementation uses wgpu, a small batched shape/image renderer, and a shared
   text engine rather than introducing an ECS.
5. **Platform host** owns the event loop, windows/surfaces, pointer and keyboard translation,
   timers, clipboard, accessibility adapter, text-input service, and native actions. It schedules
   update/layout/paint passes but does not know application view types.

The dependency direction is `language -> core -> layout/scene`; the renderer and platform host
consume the resulting scene and send normalized events back into core. State slots never hold
window handles or GPU objects, and component implementations never reach into the application's
generated fields.

### `View`, opaque return types, and native components

**Design decision (2026-10-01):** use Boring's existing distinction between `<Trait>` (`impl
Trait`, one statically known concrete type) and bare/dynamic trait storage. No `some View` syntax and
no uniform boxed return type are added.

`View` is a standard, compiler-recognized trait. A `view` declaration automatically conforms and
its source-level `body():` is checked as if it returned `<View>`:

```boring
view Badge:
    string title

    body():                         # implicit `<View> body()`
        Text(title).padding(6)
```

The body may declare local values before its final view expression but produces exactly one root.
Different conditional branches lower to concrete conditional/optional wrapper types so the hidden
Rust return type remains one type. Multiple roots require an explicit `Row`, `Column`, `Layer`, or
`Group`. Application code may write `<View>` explicitly on an ordinary helper function when useful;
the omission is special only for a `view`'s `body` declaration.

A view value is an owned, ephemeral description consumed by its parent during that build. `View`
does not require `Clone`, `Eq`, `Send`, or `'observed`, and a description is not a mounted control.
Ordinary move checking prevents inserting the same value twice. Reusing UI means calling a helper or
constructing another value, not sharing one mounted instance.

General modifiers are ordinary generic functions/extensions whose concrete result is a wrapper such
as `ModifiedView<Content, PaddingModifier>`, but their public return is spelled `<View>`:

```boring
<View> padding<V as View>(V content, Insets insets): native
<View> disabled<V as View>(V content, bool disabled): native
```

The real declaration may use method/extension syntax; the example shows the type relationship. Each
call picks one concrete Rust wrapper, so
`Text("x").width(200).padding(10)` retains ordered nested types with no box per modifier. Because an
opaque result exposes only its trait contract, component-specific configuration must occur before a
general modifier erases the concrete component API. Component configuration methods such as
`Image.resizable()` may return the same concrete component type to remain chainable; general
environment modifiers such as `font`, `style`, and `accessibilityLabel` are available on every
`View` where their semantics apply.

Heterogeneous children deliberately cross a dynamic boundary. The shipped array-block rule builds
`[dyn View]`, represented initially as one boxed trait value per emitted child. A complete modifier
chain sits inside that one box; its wrappers are still statically composed. Container implementations
consume the array during description building and must not retain references into the temporary
array. A later arena/small-value optimization may change allocations without changing Boring types,
identity, or lifecycle semantics. `AnyView` is therefore unnecessary in the initial public API.

#### Boring facade for Rust primitives

Public component signatures live in importable Boring source so ordinary name/type resolution,
default arguments, `bind` checking, documentation, and array-block disambiguation all see them. A
Rust implementation is reached through a `native` body rather than by teaching the compiler the
names `Text`, `Column`, or `Button`:

```boring
pub view Text:
    string content
    body(): native

pub view Column:
    AxisAlignment alignment = .Center
    float? spacing = nil
    [dyn View] children
    body(): native

pub view TextField:
    bind string value
    string label
    string placeholder = ""
    body(): native
```

`body(): native` is the only additional native-view convention required. It means that the linked
boring-ui runtime supplies the component's description/mount/update/layout/paint/semantics adapter,
keyed by the facade view's fully qualified type. It does not turn the component name into a keyword
or bypass its visible fields. A missing adapter or a facade/adapter signature mismatch is a link or
library-build error, never a runtime fallback.

Application-defined `view` bodies compose other views and receive the generated state/bind/env
lowering. Native-body views may declare ordinary inputs and `bind` fields plus runtime-owned mounted
control state, but not application-visible `state`: persistent implementation details belong to the
native mounted component, while reusable Boring-level state belongs in a composed non-native view.
`env` dependencies may be declared in the facade or requested by the native adapter through the
same typed environment registry.

The internal Rust-side view trait consumes/visits descriptions and is free to use object-safe build
methods, arenas, and vtables. Those ABI details are not mirrored as callable Boring methods on
`View`. Manual `struct Foo as View` conformance is rejected because an ordinary struct has neither a
`body` nor a native adapter; authors use `view Foo` instead. This keeps `View` values valid by
construction while leaving all component names and APIs library-defined.

### Mounted tree and update pass

Each mounted node contains at least:

- a runtime node id and its structural or keyed identity;
- the component/view type and current ephemeral inputs;
- identity-scoped state slots and control state;
- environment dependencies and observed subscriptions;
- lifecycle/task handles;
- layout, paint, hit-test, focus, and accessibility data;
- child nodes in structural order.

These are logical fields, not a commitment to one large Rust struct. Specialized storage may keep
view nodes, layout nodes, and render resources in separate arenas while sharing stable ids.

After the outermost event transaction ends, one update cycle performs:

1. drain and deduplicate invalidated view identities;
2. evaluate their `body()` methods using the latest inputs and persistent slots;
3. reconcile children by static site and, within dynamic iterations, by key;
4. mount new identities, update retained identities, and mark removed identities for unmount;
5. recompute dirty layout and build the display/accessibility updates;
6. submit drawing, then run post-mount work and dispose removed resources in the specified order.

Invalidation should propagate only as far as needed: a changed view rebuilds its description,
layout dirtiness travels upward only when measured size can change, and paint dirtiness remains at
the affected compositing scope. The first implementation may conservatively rebuild the whole
window after any invalidation, provided identity and lifecycle behavior are already correct; the
layer boundaries must allow incremental work without changing application semantics.

### State slots and binding handles

**Design decision (2026-10-01):** a `state` field lowers to a typed, identity-scoped runtime slot;
a `bind` field lowers to a typed handle that locates such storage. Neither is an ordinary Rust
reference into an ephemeral view value. The UI runtime and all `body()`/event callbacks execute on
one UI executor, so exclusive state needs no `Arc`, `Mutex`, or subscriber list.

For every `view`, the compiler emits a private mounted-state layout containing its `state` fields
in declaration order. Each field receives a compile-time slot ordinal and retains its Boring type
and permission metadata. The mounted node owns one instance of that generated layout, hidden behind
the runtime's mounted-view interface; the ephemeral value produced by calling the view constructor
contains ordinary inputs plus construction descriptors, never the persistent values themselves.

Conceptually:

```text
TodoApp(...) ephemeral value
    inputs: ...
    state seeds: lazy descriptors

MountedNode<TodoApp>
    identity: ...
    generation: 7
    state: __TodoAppState {
        items: [...],
        nextId: 3,
        selection: nil,
        ...
    }
```

The concrete Rust representation may use generated structs, typed cells, or an arena plus typed
accessors. It must not expose `Any` downcasts in generated application code, and a field's type must
remain statically checked across construction, access, projection, and native component calls.

#### Initialization and seeds

A state initializer lowers to a lazy, non-escaping seed closure. A parent-supplied state seed
replaces the declaration's default seed for that construction. On mount, the runtime evaluates
exactly one selected seed and installs its result. On update of the same identity, it discards the
new descriptor without evaluating its seed expression and retains the existing slot value.

This makes the already-specified behavior precise:

```boring
Counter(title: title, count: expensiveInitialCount())
```

`expensiveInitialCount()` runs only when that `Counter` identity mounts, not on every parent
rebuild. The seed may capture values available at the construction site, but it executes
synchronously during that mount and cannot be stored or invoked later. A throwing seed makes the
mount fail under the view-update error policy; asynchronous seeds are rejected and belong in
`onTask`.

`state let` uses the same storage and initialization path, then exposes only read access. State
slots are destroyed on unmount after the specified unmount callbacks and task cancellation. A new
mount of the same structural/key value receives a new generation and evaluates a new seed.

Because ignored update seeds are not evaluated, a changing parent expression is harmless rather
than a hidden side effect. When an explicit seed descriptor carries comparable captured inputs, a
development build should diagnose those inputs changing for an already-mounted identity, since it
commonly indicates that the author intended an ordinary input or `bind`; the value is still ignored.
For other seed shapes, a compiler lint may flag a syntactically varying explicit seed, but the
runtime must not evaluate it merely to produce a diagnostic.

#### Typed logical locations

A binding handle identifies a logical source location with:

- the source mounted-node id and mount generation;
- a typed root slot;
- zero or more typed field/index/key projection steps;
- its read/replace/mutate permission capability;
- the invalidation operation for the source owner.

The list describes semantics, not a requirement to allocate a vector of projection objects. The
compiler may monomorphize direct field paths and native component adapters.

A direct `state`/`bind` projection resolves in constant time. A keyed collection-element binding
stores the collection slot and stable key, not the element's current array index. Each separate
event transaction resolves that key against the current collection before borrowing the element;
the implementation may maintain a key-to-index cache invalidated by structural mutation. A move
therefore keeps the binding aimed at the same logical element. If the key has been removed, the
handle is stale: development builds report the source location and key, production ignores a write,
and no other element may be read or modified accidentally.

Field projection composes with the element locator, so `item.title` in a keyed loop can satisfy a
child `bind string`. Index-based projections are limited to the current synchronous transaction.
They cannot be captured by a callback, stored in a mounted field, or passed as `bind`, because a
structural collection mutation could redirect them. The checker should suggest the keyed `for`
form when persistent projection is required.

The mount generation prevents an old callback, native event, or late task continuation from
accessing a node id reused after unmount. Validation happens before every callback transaction and
again when a suspended task resumes; lifecycle tasks additionally validate their task-run generation
defined under Lifecycle below. A stale read cannot manufacture a default `T`; it aborts that callback
with the stale-handle diagnostic in development and cancels/ignores the originating event in
production. A stale write remains the previously specified no-op in production.

#### Access and mutation lowering

Inside a `view`, source syntax stays transparent while lowering makes the access explicit:

```boring
Text(name)              # read slot/handle
name = "Ada"            # replace, then invalidate source owner
profile.rename("Ada")   # exclusive mutable access, then invalidate source owner
```

A replacement writes only after the right-hand side has evaluated successfully. A mutating method
borrows the slot exclusively for the duration of the call and invalidates after the call returns,
including when it reports an error after partial mutation. Nested access to the same slot obeys
Boring's ordinary exclusivity rules; runtime re-entrancy checks remain as a defensive diagnostic
for native implementations.

Permissions are part of the compiler's typed location capability: read-only, replace (`var`),
in-place mutation (`mut`), or both (`var mut`). Projection can preserve or reduce a capability but
never add one. Native/Rust components receive corresponding typed adapter capabilities even though
the public Boring signature continues to say `bind T`.

The write schedules invalidation of the view that owns the root `state`, not necessarily the child
holding the handle. Further writes in the same event are coalesced by the existing transaction
rule. A projection into an `'observed` root instead uses that value's normal notification mechanism.
Changing which source location is supplied to a retained child's `bind` replaces its handle before
the child body is evaluated; it does not copy a value between old and new sources.

UI state is executor-confined. A worker thread cannot dereference or mutate a slot handle directly;
it sends a result to the UI executor. `onTask` continuations that access view state resume on that
executor. This preserves exclusive storage without locks and gives event batching one deterministic
ordering.

### Initial Rust substrate

The current candidates are `winit` for window/event-loop integration, `wgpu` for presentation,
`cosmic-text` for shaping, layout, rasterization and editing primitives, and AccessKit for the
platform accessibility tree. Image decoding and SVG rasterization can reuse the `image` and
`resvg`/`tiny-skia` approach already exercised in the repository. These are implementation
dependencies rather than public API: pinning exact versions waits for a vertical platform spike.

`winit` does not by itself provide one uniform mobile IME protocol, so the platform-host boundary
must permit dedicated iOS and Android text-input adapters. This is a reason to keep platform input
behind a boring-ui service rather than expose native widgets in the component API.

The repository may initially implement these layers as Rust modules in one crate. Their interfaces
should remain explicit; separate crates are justified only when platform feature selection or
independent headless testing requires it. Premature crate splitting would slow down the first
vertical slice without improving the design.

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

These are API sketches, not implemented signatures. Widths use platform-independent logical units;
the window's platform scale factor converts them to physical pixels at rendering/input boundaries.
Scale changes invalidate layout and text rasterization while preserving logical geometry. The
The examples describe layout bounds rather than clipping. Overflow remains visible unless an
explicit `clip()` modifier or a viewport-owning component such as `Scroll`/`List` clips it, as
specified below.

- Component-specific methods configure that component, for example `Text(...).font(...)`.
- General layout and decoration modifiers compose wrappers and are available on every view.
- Conceptually, the first width example yields `Padding<Width<Text>>`, the second
  `Width<Padding<Text>>`. These are concrete generic wrappers hidden behind Boring's existing
  `<View>`/Rust `impl View` return spelling; dynamic erasure happens only when collected into
  `[dyn View]`.
- Repeating a modifier creates another ordered wrapper. Component-specific configuration methods
  must be called before a general modifier returns opaque `<View>`; general modifiers remain
  chainable on every opaque result.

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
an ideal/content size and is distinct from both zero and infinity. Public proposals and dimensions
never accept infinity; the layout protocol may use a private unbounded-maximum sentinel when
describing flexibility.

| Element | Layout contract |
|---|---|
| `Text` | Measures its content under the proposed width; constrained width can cause wrapping. `lineLimit` and truncation follow the contract below. |
| `width(200)` | Proposes width 200 to its child, passes through the height proposal, and reports width 200 with the child's measured height. Centers the child horizontally by default; alignment will be configurable. Does not clip an overflowing child. |
| `padding(10)` | Subtracts 20 from each specified proposal dimension, clamped to zero, measures the child, then adds 20 to each reported dimension. An unspecified proposal dimension stays unspecified. Places the child at the padding inset. |
| `Row` | Allocates horizontal space by priority and flexibility (below) and aligns children vertically. |
| `Column` | Allocates vertical space by priority and flexibility (below) and aligns children horizontally. |
| `Layer` | Proposes a common available space to its children, reports their aligned union, and paints them in source order. |
| `Spacer` | Expands on the containing stack's main axis under the allocation rules below. Its omitted minimum comes from the active theme. |

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
(for example `minWidth > maxWidth`) are errors. Statically known invalid values are checker errors;
dynamic invalid values fail that view update with a layout diagnostic and leave the last valid
mounted inputs in place.

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

The initial `Length` is a finite floating-point number in platform-independent logical units;
integer literals convert normally. Negative, NaN, and infinite dimensions are invalid under the
same checker/runtime rule. Nested frames are not merged: each validates its own constraints and
then participates as an ordered wrapper, so individually valid inner and outer constraints may
produce deliberate overflow. Percentage/relative units are deferred and can later use a distinct
length value rather than overloading the scalar contract.

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

#### Deterministic stack allocation

**Design decision (2026-10-01):** measurement exposes a child-derived interval on each axis under
the current cross-axis proposal: minimum, ideal, and maximum. Maximum may be internally unbounded;
this sentinel is part of the layout protocol and is not the public `.infinity` value rejected above.
All three values are non-negative and ordered. Wrappers transform the interval: a fixed frame makes
all three equal, a finite frame clamps them, padding adds its insets, and fill makes the maximum
unbounded while retaining the child's minimum.

`layoutPriority(float value)` is an ordered wrapper with finite value, default `0`. Priority is
compared only with siblings in the containing stack; negative priorities are valid and receive
space after zero-priority children. It does not alter the child's interval.

For a finite main-axis proposal, `Row`/`Column` use this algorithm:

1. Resolve every adjacent adaptive gap and subtract their total from the proposal.
2. Assign every child its minimum. This reservation applies across all priority groups.
3. Visit priority groups from highest to lowest and distribute space toward each child's ideal.
4. Visit the groups in the same order again and distribute remaining space from ideal toward
   maximum.
5. Measure each child with its assigned main-axis size and the stack's cross-axis proposal, then
   place children in structural order.

Within one priority group, distribution is water-filling by flexibility: children with the smallest
remaining range reach their next bound first; the unsatisfied children share the rest equally.
Equal ranges tie by structural order only for deterministic residual rounding, not for a meaningful
allocation advantage. A child may still report more than its assignment when that is its minimum;
the stack uses the reported size.

If the available size is below the sum of minimums, every child keeps its minimum and the stack
reports the minimum aggregate, which exceeds the proposal and therefore overflows. Compression
never invents a size below a child's declared minimum. Text and other intrinsically compressible
content express compression by deriving a smaller minimum and remeasuring under the assigned
proposal, for example through wrapping.

With an unspecified main-axis proposal, the stack assigns ideal sizes, uses minimum/ideal spacing,
and reports their aggregate. Unbounded children such as `Spacer` do not expand without a finite
proposal; their ideal equals their themed minimum. This prevents a content-sized stack from
manufacturing an arbitrary size.

Layout calculations remain in floating-point logical units. Conversion to physical pixels happens
at paint/compositing boundaries. When integral device coordinates are required, accumulated edges
are rounded rather than rounding every child independently, and any residual pixel is assigned in
visual leading-to-trailing order (reversed under right-to-left layout). This keeps the total equal
to the rounded container extent.

### Stack alignment and spacing

**Design decision (2026-09-30):** follow SwiftUI's axis-specific alignment and adaptive default
spacing. Omitting spacing is semantically different from specifying zero.

```boring
Row(alignment: center, spacing: 8):
    Text("Name")
    TextField(value: name, label: "Name")

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
  its bottom layout edge. Multiline text publishes the typographic baseline of its first or last
  visible line respectively. A row places children so the requested guides coincide, then reports
  the union of their resulting bounds; baseline alignment can therefore increase row height above
  the tallest unaligned child.
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
siblings away and does not enlarge an ancestor after layout. As specified by event dispatch, visible
overflow does not enlarge the hit region: hit testing uses layout bounds intersected with ancestor
clips unless an explicit future hit-shape modifier replaces it.

`clip()` wraps a view and restricts rendering and hit testing to the wrapper's rectangular bounds.
Supplying a shape clips to that shape within the same bounds. Like every wrapper modifier, its
position in the modifier chain matters. The first implementation supports rectangles and rounded
rectangles. Arbitrary paths may be added without changing the modifier contract; a path affects
painting and hit testing but never measurement.

`Scroll(axis: vertical)` creates a finite viewport from the proposal it receives. It proposes an
unspecified height to its content and passes through the available width; the horizontal form does
the symmetric operation. Its content can therefore choose its full extent on the scrolling axis,
while the viewport reports a finite size to its parent and clips rendering at its own bounds.
Two-axis scrolling is deferred: completely unspecified proposals on both axes interact poorly with
wrapping content and are unnecessary for the first vertical slice.

The scroll offset is identity-scoped mounted control state, initialized to zero and preserved across
ordinary rebuilds. After layout it is clamped to `[0, max(contentExtent - viewportExtent, 0)]`;
content shrinkage can therefore move the visible region back into range. The logical offset never
includes overscroll. A platform adapter may paint a temporary elastic effect around it, but that
effect disappears when input ends and does not affect layout, hit testing, or programmatic state.

Wheel/trackpad input and touch dragging update the offset in the current event transaction. Momentum
is driven by platform-timed follow-up transactions. In nested scroll containers, the deepest
eligible scroll consumes only the delta it can apply; any remainder bubbles to an ancestor scrolling
on the same axis. Pointer capture during a drag stays with the scroll gesture until release or
cancellation.

Scrollbars use platform/theme defaults and overlay the viewport, so their appearance does not change
content measurement. The viewport exposes accessibility scroll actions and keyboard scrolling when
it or a descendant has focus. Moving focus to a descendant outside the visible viewport performs
the minimum scroll needed to reveal its layout bounds. Explicit page/line increments derive from
the viewport and theme.

Mounted identity already provides basic position restoration while a `Scroll` remains mounted.
External/programmatic scroll position, named anchors, initial anchors, persistent restoration across
unmount, and scrollbar policy modifiers are deferred until a concrete application needs them. Lazy
child creation remains a separate container concern (`LazyColumn`/`LazyRow` or a keyed list), not an
automatic property of `Scroll`.

## Painting, compositing, and render invalidation

**Design decision (2026-10-01):** layout produces a backend-neutral display list. Components never
issue wgpu commands and the renderer never calls `body()` or performs layout. The display list is
deterministic data that can be inspected in headless tests, translated to GPU batches, or replayed
after surface/device recreation.

The initial drawing vocabulary is deliberately small:

- filled rectangles and rounded rectangles;
- inside borders for those shapes;
- shaped glyph runs plus text decorations, selections, marked-text decoration, and carets;
- raster images and current video frames with source/destination rectangles and sampling mode;
- push/pop rectangular or rounded-rectangle clips;
- push/pop affine transforms;
- push/pop opacity/compositing groups;
- outer shadows for rectangular/rounded shapes.

Arbitrary vector paths, gradients, blend-mode APIs, filters, backdrop blur, mesh drawing, and custom
shaders are deferred. The internal command format is versioned so adding them does not alter `View`
or layout contracts.

### Paint order and modifier effects

Each mounted node emits commands into its assigned paint scope in structural order. `Layer` paints
children from first to last, so later children cover earlier ones. A wrapper controls the relative
order of its own effects and child:

- `background(style)` paints its shape/style, then the child;
- `border(style, width)` paints an inside stroke over the child at the receiver's bounds;
- `overlay(view)` paints the child, then the aligned overlay subtree;
- `clip(shape)` pushes the clip around all commands emitted by its child;
- `opacity(value)` composites the complete child result with that opacity;
- `shadow(...)` paints the shadow for the receiver's composited result without changing layout.

This ordering follows the same nested-wrapper rule as layout. A clip outside a shadow clips the
shadow; a shadow outside a clip uses the clipped child as its shadow source but may draw beyond that
inner clip. Effect overflow does not alter measured bounds or sibling placement. Hit testing and
accessibility continue to use the separately specified geometry rather than alpha-testing pixels.

Opacity is finite and clamped to `0...1`; constants outside the range are checker errors and dynamic
values fail the view update. Opacity applies to the group result, not independently to every
primitive, so overlapping descendants do not become darker. The renderer may skip an offscreen
surface when it can prove direct blending is equivalent. Opacity zero suppresses painting but does
not implicitly remove layout, hit testing, focus, or accessibility; conditional mounting or
`accessibilityHidden` expresses those separate intents.

Affine paint transforms do not participate in size negotiation. They transform painting, hit-test
geometry, focus rings, and reported accessibility bounds consistently around an explicit/default
anchor. Their inverses are used for input; a non-invertible transform is invalid. Translation can be
exposed first as `.offset(x:y:)`; scale/rotation and animated transforms may follow once their public
APIs are needed.

### Clips, layers, and physical pixels

Clip commands intersect with the current clip stack. Axis-aligned rectangular clips should lower to
GPU scissors; rounded/transformed clips may use stencil or mask textures. Empty intersections skip
their subtree. Clip antialiasing must be consistent with shape edges and must not expand the logical
hit region.

The renderer draws directly into the current target unless group opacity, a future filter/blend
mode, or an overlapping effect requires isolation. An isolated layer has explicit logical bounds
expanded by its effect radius and clipped to device limits. Texture allocation uses a reusable pool;
layer creation is a rendering optimization detail and never creates view identity or lifecycle.

Logical coordinates stay floating point through layout and scene construction. The platform scale
factor maps them to physical coordinates. Hairlines and control borders may snap at paint time to
device-pixel centers; text origins retain subpixel positioning. Snapping never feeds a rounded value
back into layout, preventing cumulative drift. A scale-factor change rebuilds scale-dependent glyph
and raster caches and repaints the window.

Colors enter the display list as explicit color values plus color-space metadata. The initial
surface is standard dynamic-range sRGB; blending uses premultiplied alpha in a linear working space
before conversion to the surface format. Wide-gamut/HDR output is a later surface capability, not a
different application color API. Image metadata is honored during decode when available; unknown
images default to sRGB.

### Text and media resources

The scene stores shaped glyph ids/positions and font/resource keys, not prebuilt wgpu buffers.
`cosmic-text` owns shaping/layout and rasterization inputs; the renderer maintains glyph atlases per
device/scale configuration. Missing glyphs use font fallback from the same shaped run, never a
renderer-time character substitution. Atlas eviction invalidates only the affected GPU entries,
not view layout.

Decoded images and video frames are referenced by generation-checked resource handles. Upload
completion marks paint dirty without rebuilding `body()` when intrinsic dimensions are unchanged;
a newly discovered intrinsic size additionally marks layout dirty. Late media generations are
discarded under the media contract. Video frame arrival requests repaint/composite work at its
presentation cadence without invalidating application state.

### Dirty flags and frame scheduling

Mounted nodes track independent dirty causes:

- **view**: re-evaluate/reconcile descriptions after reactive invalidation;
- **layout**: remeasure/place the affected layout boundary and necessary ancestors;
- **paint**: regenerate display commands for a node/subtree;
- **composite**: reuse commands/resources but submit a new frame, for example video or opacity
  animation;
- **semantics**: regenerate/publish semantic properties or bounds.

Dirty causes propagate only through the dependencies they affect. Layout implies paint and semantic
bounds updates; paint does not imply layout; a caret blink is paint-only; hover/pressed style is
usually paint plus semantics state; a cached texture upload is paint/composite. The first vertical
slice may conservatively rebuild the complete window display list and redraw the complete surface,
but it must preserve these cause distinctions in component/runtime APIs so later incremental work
does not alter behavior.

At most one frame request is pending per window. Multiple writes/resource completions before the
platform redraw callback coalesce. A frame uses one committed tree/layout/display-list generation;
events arriving during submission queue for the next transaction. Damage rectangles and partial
surface redraw are optional optimizations and do not change display-list order.

Surface resize to zero suspends acquisition/rendering while retaining the mounted tree. Surface
loss or GPU device loss drops backend resources, recreates the surface/device and caches, then
replays the latest committed display list; it does not remount views or rerun lifecycle hooks. An
unrecoverable renderer error goes to the application runtime reporter with window/device context.

Headless conformance tests compare normalized display commands, logical bounds, clip/transform
stacks, and resource keys. GPU tests cover batch translation and representative pixel output, but
pixel snapshots are not the primary semantic oracle because rasterization varies by device and
font backend.

## Animation and transitions

**Design decision (2026-10-01):** model state changes immediately; animation interpolates retained
presentation properties between the previous and new committed trees. `body()` and application
callbacks always observe the target state, never synthetic intermediate state values.

No state change animates by default. An ordinary library function establishes animation metadata on
the current UI transaction:

```boring
animate(.EaseInOut(duration: Duration.milliseconds(200))):
    expanded = !expanded
    opacity = 1
```

`animate` accepts a synchronous non-throwing callback. All writes in that callback join its
transaction and eligible resulting presentation changes share the animation. A nested `animate`
applies its innermost animation to writes it contains. Writes after an async suspension occur in a
later transaction and animate only when that continuation establishes another animation. A
`withoutAnimation:` transaction explicitly snaps eligible changes and overrides an inherited
animation.

The value-triggered modifier follows SwiftUI's scoped form:

```boring
Details()
    .animation(.EaseOut(duration: Duration.milliseconds(150)), value: expanded)
```

When `value` changes by equality after the initial mount, animatable presentation differences inside
that wrapper use its animation. The modifier never animates merely because an unrelated ancestor
rebuilt. Its trigger must be `Eq + Clone`; it is a change detector, not view identity. Precedence is
an explicit `withoutAnimation` transaction, then the nearest triggered `.animation` wrapper, then
the originating `animate` transaction, then no animation.

Initial curves are `.Linear`, `.EaseIn`, `.EaseOut`, `.EaseInOut`, and `.Spring(response: damping:)`.
Durations/parameters must be finite and non-negative. A zero duration is a valid immediate change.
The animation clock is monotonic and driven by the platform frame scheduler.

### Animatable presentation values

The first interpolation set includes finite scalar lengths, positions/sizes/rectangles/insets,
corner radii, opacity, premultiplied colors in the renderer's linear working space, and invertible
affine transforms. Compound values interpolate corresponding fields. Discrete values switch at the
end unless a component defines an explicit semantic interpolation.

Layout properties may animate. The runtime interpolates the affected proposal/placement value and
reruns the necessary layout boundary for each frame; siblings therefore move consistently rather
than merely painting a scaled snapshot. Paint-only properties skip layout. Component/style adapters
declare which retained presentation fields are animatable and which dirty cause each requires.

If a property receives a new animated target while running, the new animation starts from its
current presentation value with the new curve. A non-animated update cancels interpolation and snaps
to the target. Removing the modifier or changing its trigger without a property change has no
visible effect.

Painting, hit testing, focus rings, pointer geometry, and accessibility bounds use presentation
geometry during animation. Semantic values, enabled/selected state, and bindings expose the target
commit immediately. Input delivered during a moving transition therefore targets what is visibly
under the pointer, while assistive technology hears the new value without waiting for decoration.

### Insertion and removal transitions

`.transition(...)` describes how a conditional/keyed subtree enters and leaves when the transaction
that inserts/removes it is animated:

```boring
if showingDetails:
    Details().transition(.Opacity)

if showingEditor:
    Editor().transition(
        .Asymmetric(insertion: .Scale(0.96), removal: .Opacity),
    )
```

Initial transitions are `.Identity`, `.Opacity`, `.Scale(float)`, `.Offset(x:y:)`, and
`.Asymmetric(insertion:removal:)`; compatible transitions can be combined. The transition modifier
does nothing when insertion/removal occurs without an animation transaction, except for a
component-provided platform transition such as `Sheet`'s default presentation.

An entering subtree mounts and commits normally, runs `onMount`, then presents from the insertion
state toward its committed state. An exiting subtree enters a **retiring** mounted phase: it is
removed from active hit testing, focus traversal, accessibility, and ordinary reconciliation;
pointer capture is cancelled and scoped tasks receive cancellation immediately. Its last committed
layout/display data and state/resources remain available for the removal transition. `onUnmount`
runs when the transition finishes, after which slots/resources and identity are dropped.

Because the retiring subtree is presentation-frozen, it does not re-evaluate `body()` in response to
its own state changes. Teardown hooks may still read its last state and update valid ancestor binds
under the lifecycle rules. If the same structural/keyed identity is reinserted before retirement
finishes, the runtime cancels/reverses the removal presentation, returns that mounted identity to the
active tree, updates its inputs, and does not run a second `onMount` or an intervening `onUnmount`.

Removing an ancestor without a compatible transition ends descendant transitions and retires the
subtree under the ancestor's transition as one presentation group. Window close, application
shutdown, renderer loss, and fatal mount failure skip visual completion and perform teardown
immediately; lifecycle correctness never depends on displaying the final frame.

Keyed moves are not insertion/removal. Under an animated transaction, retained children interpolate
from old to new layout placement and keep state/focus. Without one they move immediately. A change
of key remains removal plus insertion and uses the corresponding transitions.

### Modal transitions and reduced motion

`Sheet` supplies a platform/theme default transition even when the presenting state write used no
explicit animation. Writing `false` starts its retiring presentation; its content stays mounted but
inactive until completion under the rules above. An explicit transaction may select an allowed
duration/curve while the platform component retains modality and placement behavior. Forced owner
unmount skips the remaining animation and tears down immediately.

The inherited accessibility `reducedMotion` preference transforms animations before they start.
Springs, scale, offset, and animated layout normally become immediate changes or a short opacity
cross-fade selected by the platform theme; no application state or completion ordering changes.
Opacity-only feedback may remain with a capped short duration. The first API has no casual override
for this preference; components with genuinely essential continuous motion require a separately
reviewed semantic option.

Animation completion is not a general lifecycle hook in the first API. Internal transitions use it
to finish retirement, while application logic reacts to state changes rather than frame timing.
Explicit completion callbacks, phase/keyframe animation, matched geometry, arbitrary animatable
user types, and timeline views are deferred.

## Initial component contracts

### Layout primitives

`Row` and `Column` provide one-dimensional layout under the priority/flexibility and
alignment/spacing rules above. `Spacer`, `Separator`, and `Layer` complete the initial primitive
set:

```boring
Column(alignment: leading, spacing: 8):
    Text("Title")
    Separator()
    Row:
        Text("Status")
        Spacer()
        Toggle(value: enabled, label: "Enabled")
```

`Spacer()` is flexible on its containing `Row`/`Column` main axis and uses the active theme's
adaptive minimum length. `Spacer(min: 0)` permits complete compression. A spacer outside a
one-dimensional stack is flexible on every axis for which it receives a finite proposal; whether
that uncommon use remains supported after implementation experience is open.

`Separator()` infers a vertical orientation inside `Row` and a horizontal orientation inside
`Column` or `List`. Outside a container that supplies this layout context, `axis:` is required.
Thickness, color, and surrounding adaptive spacing come from the theme and contribute normally to
measurement.

`Layer` is the overlay stack name:

```boring
Layer(alignment: bottomTrailing):
    Image(photo)
    Text("New").padding(6).background(badgeColor)
```

It proposes the same available size to every child, reports the aligned union of their chosen
bounds, and places them using its alignment. Children paint in source order, so later children
appear above earlier ones. Pointer hit testing examines that order in reverse (topmost first), while
accessibility traversal follows semantic/source order unless explicitly overridden. `Layer` leaves
the name `overlay` available for a later modifier that decorates one existing view.

### Text

`Text(string content)` displays a string and is content-sized under the proposal it receives. A
finite proposed width may cause wrapping; text measurement uses the resolved font and locale.
The first implementation needs theme-derived font/color defaults plus ordered modifiers for font,
weight, color, line limit, truncation, and text alignment. Rich attributed runs, selectable text,
and advanced typography can follow without changing the base constructor.

```boring
Text("3 items")
Text(title).font(titleFont).lineLimit(2)
```

`lineLimit(int? lines)` defaults to `nil`, meaning that the proposal and content determine the
number of lines. A non-null limit must be at least one. Text wraps before applying the limit; when
content does not fit, the last visible line is truncated with an ellipsis. The truncation position
is selected by `truncation(.Head | .Middle | .Tail)` and defaults to `.Tail`. `lineLimit(1)` is the
ordinary single-line form. A finite height may expose fewer lines than the requested limit and clips
the remainder at the text view's bounds.

`textAlignment(.Leading | .Center | .Trailing)` aligns line fragments within the measured text
width and defaults to direction-aware `.Leading`; it does not position the `Text` view in its
parent. Font, weight, foreground color, line limit, truncation, and text alignment configure the
`Text` value and must therefore appear before a general modifier returns opaque `<View>`.

### Image, Video, and asynchronous media

**Design decision (2026-09-30):** every media source is potentially asynchronous. `Image` does not
promise synchronous file/decode/GPU work, and there is no separate `AsyncImage`. Images and videos
share source/loading/cache infrastructure but remain distinct view types because video exposes
time-based playback behavior.

```boring
Image.asset("logo")
Image.file(path)
Image.bytes(data)
Image.url(avatarURL)
Image.system("trash")

Video.asset("intro")
Video.file(path)
Video.url(streamURL)
```

Conceptually these constructors create a common `MediaSource` (`asset`, `file`, `bytes`, `url`, or
platform/system resource). The loader chooses I/O, permissions, decoding, cache policy, and GPU
upload for the source. A cached resource may become ready immediately, but application behavior
must not rely on synchronous completion.

Each mounted media view owns a `loading → ready | failed` request generation. Changing its source
cancels or abandons the previous generation; unmount releases its request; a late result is ignored
unless its generation still matches. Byte/decoded/texture caches may be shared across identities,
and cancellation of one consumer must not cancel work still needed by another.

```boring
Image.url(avatarURL)
    .placeholder(ProgressIndicator())
    .fallback(Image.asset("default-avatar"))
    .resizable()
    .aspectRatio(mode: fill)
    .frame(width: 48, height: 48)
    .clip(Circle())
```

An image is intrinsic-sized by default; `resizable()` permits proposal-driven scaling and
`aspectRatio(mode: fit|fill)` preserves its ratio. `fill` can overflow and therefore commonly pairs
with explicit clipping. While loading, a bundled asset can use dimensions from its manifest; other
sources use the placeholder's measurement or an explicit frame. With neither, their provisional
ideal size is zero and successful metadata/decoding invalidates layout.

`Video` uses the same source and placeholder/fallback concepts but owns a playback session with
buffering, time, audio, tracks, and hardware decoding. Video-only APIs remain type-safe:

```boring
state bool playing = false

Video.url(trailerURL)
    .playing(playing)
    .controls()
    .loop()
    .aspectRatio(mode: fit)
```

`playing` conceptually accepts `bind bool` (implicitly `var`), so application changes control playback and native
controls update application state. Playback is paused by default unless explicitly requested.
Unmount stops/releases the session; default background, interruption, audio-focus, streaming, and
position-restoration policies remain to be designed. Animated image formats remain an `Image`
decoder concern when they have no user-controlled timeline; interactive/time-addressable media uses
`Video`.

Media has no accessible label derived from a file/resource name. Meaningful content requires
`accessibilityLabel(...)`; decorative content uses `accessibilityHidden()`. A video's captions and
controls participate separately in accessibility.

### Button

`Button(string label, ButtonRole role = .Normal)` creates a semantic action control. Its action uses
`onActivate`, not device-specific `onClick`: activation includes pointer/touch input, keyboard,
shortcuts, and assistive technology. The control owns its hover/pressed/focused visual state and
exposes the appropriate accessibility role automatically.

```boring
enum ButtonRole:
    Normal
    Destructive
    Cancel
```

```boring
Button("Add").onActivate ():
    addItem()

Button("Delete", role: .Destructive).disabled(selection is nil).onActivate ():
    deleteItem()
```

`.Normal`, `.Destructive`, and `.Cancel` are the initial semantic roles. The closed enum lets the
runtime, accessibility layer, and themes handle every recognized meaning exhaustively. A role informs theme and
platform behavior but does not hard-code a color or require confirmation. `disabled(bool)` is a
general ordered view modifier: a disabled subtree does not accept semantic activation or editing,
reports disabled accessibility state, and uses theme-defined appearance. Placing `disabled` before
the multiline action in the example is required because Boring does not allow further chaining
after a multiline trailing closure.

`.defaultAction(bool enabled = true)` registers a button as the Enter/Return default in its nearest
focus scope. At most one mounted, enabled default action may exist in a scope; a conflicting update
is rejected transactionally and keeps the previous committed registration. Enter first belongs to
an active IME and then to a focused control's own semantic command, such as `TextField.onSubmit`.
Only an otherwise unconsumed Enter activates the scope's default button. Activation is emitted once
per platform key action rather than on key repeat.

An enabled `.Cancel` button is the Escape/cancel action of its nearest presentation focus scope.
At most one may be mounted in that scope. Escape first cancels IME composition or a transient
control operation; an otherwise unconsumed Escape activates that button. If none exists, the active
`Sheet` performs its ordinary dismissal by writing `false`; if no presentation handles it, the
command reaches the platform host. Pointer, keyboard, platform-command, and accessibility
activation all converge on the same `onActivate` callback and event transaction.

### TextField

Conceptually, `TextField` declares `bind string value` (implicitly `var`) and ordinary
`string label` plus `string placeholder = ""`. The semantic label is required and remains stable
while editing; a placeholder is only a visual input hint and never substitutes for the accessible
name.
It is custom-rendered and connects to the platform text-input service for software keyboards and
IME correctness. Accepted text writes through `value`; transient composition, selection, cursor,
scroll offset, and undo state remain inside the persistent mounted control. `onSubmit` is a semantic
action generated by Enter or the platform's equivalent input action.

```boring
TextField(value: name, label: "Name", placeholder: "Ada").onSubmit ():
    save()
```

#### Accepted value, editing buffer, and IME

**Design decision (2026-10-01):** `TextField` is single-line. Its mounted control keeps an editing
buffer, selection/caret, marked IME range, horizontal scroll, and undo grouping. The bound string is
the last **accepted** value. Usually the buffer and binding match; marked/preedit text is the
deliberate exception.

IME preedit updates only the mounted buffer and marked range. It may be painted, positioned, and
reported through the platform text-input protocol, but it does not write the binding or trigger
application validation on every composition step. An IME commit replaces the marked range and
writes the resulting accepted string once in the current event transaction. Cancelling composition
restores the accepted value for that range. Enter/Return first lets an active IME consume or commit
the composition and must not also submit the field from the same key event.

Ordinary keyboard insertion, deletion, paste, dictation commit, and accessibility set-value actions
are accepted edits and update the binding synchronously. Navigation or selection alone does not.
Editing uses Unicode grapheme boundaries for caret/deletion, the text engine's bidi visual movement,
and platform word/line boundary conventions. CR/LF sequences from every accepted source are
normalized to spaces before writing the binding. If a parent supplies a value containing newlines,
the field displays the normalized value, emits a development diagnostic, and schedules one
post-update binding correction so buffer and source converge without mutating state during
reconciliation.

Each accepted write carries a control-origin revision. When reconciliation returns the same value,
the field preserves buffer, selection, composition, and undo state. A different parent value is
authoritative: the field cancels marked text, replaces its buffer, creates an undo boundary, clears
redo, and maps the selection through the longest common prefix/suffix of the old and new strings;
a selection intersecting the changed range collapses to the end of the inserted range. Indices are
then clamped to grapheme boundaries. This prevents routine formatting outside the edited range from
jumping the caret to the end.

Undo/redo is local mounted control state but every undo/redo result is another accepted binding
write. An authoritative external replacement prevents undo from crossing that boundary, so local
history cannot later overwrite newer application data. Unmount discards the history.

#### Submission and editing commands

With no active composition, Enter/Return invokes `onSubmit` when present. Otherwise it invokes the
eligible `.defaultAction()` button in the active focus scope; if neither exists, it has no semantic
effect. Submission does not resign focus automatically. The callback may clear focus explicitly or
change presentation state.

Escape first cancels active marked text, then any transient selection/drag operation. Only an
otherwise unconsumed Escape reaches modal cancellation or a `.Cancel` button. Clipboard commands use
the platform clipboard service and participate in the same event transaction as other accepted
edits.

#### Validation and input constraints

Arbitrary validation does not reject keystrokes in the initial API. Rejecting intermediate strings
breaks IME, paste, and natural entry of temporarily incomplete values such as `-` or `1.`. The
application derives validation from the accepted bound value and supplies presentation state:

```boring
enum Validation:
    None
    Valid
    Invalid(string message)

TextField(value: email, label: "Email")
    .validation(validateEmail(email))
```

`.validation(...)` changes semantic invalid state, announces/exposes the message, and lets the theme
style the control; it never rewrites the value. Form submission decides whether invalid data blocks
the action. Built-in structural constraints may be added when their behavior is unambiguous. The
initial `.maxLength(int graphemes)` permits composition to proceed, then truncates a committed
insertion/paste at a grapheme boundary and reports the limit through accessibility. Negative limits
are invalid layout-style inputs. Custom formatter/parser bindings for numbers, dates, and other
typed values are a later API rather than hidden behavior in the string field.

#### Input purpose and platform hints

Text-input hints are typed semantic options, not raw platform strings:

```boring
TextField(value: email, label: "Email")
    .contentType(.EmailAddress)
    .keyboard(.EmailAddress)
    .capitalization(.Never)
    .autocorrection(.Disabled)
    .submitLabel(.Next)
```

`contentType` describes meaning for autofill; `keyboard` requests an appropriate software-keyboard
layout; capitalization, autocorrection, spell checking, and submit-label preferences refine input.
They are hints where a platform cannot honor them and never change validation or the bound type.
The initial enums cover ordinary text, name, username, email, telephone, URL, search, password, new
password, and one-time code without exposing platform-specific constants.

#### Secure and multiline input

`SecureField(bind string value, string label, string placeholder = "")` shares the single-line
editing/IME contract but is a distinct semantic control. It obscures glyphs except for any brief
platform-standard reveal, never exposes its value through accessibility, screenshots produced by
the toolkit, logs, or semantic diagnostics, and disables copy/cut and ordinary undo-history export;
paste and password-manager/autofill insertion remain available. The binding still contains the real
string in application memory, so this is input privacy rather than a secure-memory type. A reveal
control must be an explicit component option with an accessible state/action.

`TextEditor(bind string value, string label)` is the future multiline counterpart. It accepts
newlines, scrolls internally, uses vertical selection/navigation, and treats Return as text input;
submission requires an explicit command or callback. It uses the same accepted-value, IME, external
replacement, validation, focus, and accessibility rules. It is not required by the editable-list
vertical slice and does not complicate `TextField` with a multiline mode flag.

`TextField` reports the theme's single-line control height and an ideal width based on a bounded
sample of label/placeholder/current text; it remains horizontally flexible in a `Row`. Long content
scrolls inside the editing viewport rather than increasing the field's reported width without
bound. Selection binding, attributed text, input masks, and rich editing remain deferred.

### Toggle

`Toggle` declares `bind bool value` (implicitly `var`) plus a textual label. It is a semantic binary control; a
theme may render it as a checkbox or switch without changing application state or event handling.

```boring
Toggle(value: completed, label: "Completed")
```

If the application requires a specific interaction idiom rather than a themed choice, explicit
styles or later `Checkbox`/`Switch` controls can constrain presentation while retaining the same
binding contract.

### Common control rules

Control constructors receive current inputs on every rebuilt view value while their mounted nodes
retain transient interaction state. All controls inherit enabled state, theme, locale, layout
direction, and accessibility environment from their ancestors. Semantic callbacks participate in
the synchronous event transaction defined above. Visual styling must not change the control's role,
keyboard behavior, or accessible name.

### Accessibility semantics

**Design decision (2026-10-01):** accessibility is a backend-neutral semantic tree derived from the
same committed mounted tree as layout, focus, and painting. It is not inferred from pixels and
component implementations do not call AccessKit or platform APIs directly. The platform host maps
the semantic tree and its incremental updates to AccessKit adapters.

Every semantic node has a stable runtime id derived from its mounted identity, a role, optional
name/value/description, state flags, bounds, ordered semantic children, and supported actions.
Reordering a keyed row moves the existing semantic node rather than destroying it. Failed view
updates do not publish partial semantic updates. Bounds are produced after layout in window logical
coordinates and converted by the platform adapter.

Layout-only wrappers and `Row`/`Column`/`Layer` are transparent by default: their semantic children
are attached to the nearest semantic ancestor in structural order. The initial components emit:

| Component | Default semantics |
|---|---|
| `Text` | static text with its content as name; `.accessibilityHeading(level:)` additionally marks document/section hierarchy |
| `Button` | button role, constructor label as name, enabled/focused state, activate action; `ButtonRole` adds semantic intent without changing the role |
| `TextField` | single-line text-input role, required `label` as name, accepted text and selection as editable value, focus/set-value actions; placeholder is only a hint |
| `SecureField` | protected single-line text-input role and required label; never exposes plaintext value or selection content |
| `Toggle` | toggle/check role, constructor label as name, checked value, enabled/focused state, toggle action |
| `Image` | no node until labeled; `.accessibilityHidden()` explicitly confirms decorative intent |
| `Video` | media/group role when labeled; captions, playback state, and visible controls are separate semantic children/actions |
| `Separator` | separator role with inferred orientation |
| `Scroll` | scrollable group with axis, range, current offset, and forward/backward scroll actions |
| `List` | list role; keyed rows retain order/position metadata, selected state, and available selection/reorder/delete actions |
| `Sheet` | modal dialog/group scope whose semantic subtree temporarily hides underlying content from traversal |

An unlabeled interactive node is a development validation error containing its view path and source
location. Production retains the node so it remains operable, reports the error once per mounted
identity, and lets the platform use a generic localized role name as a last resort. Resource names,
file paths, icon identifiers, and placeholders are never guessed as accessible names.

An unlabeled `Image`/`Video` produces a development warning until it is explicitly labeled or
hidden. This forces the author to decide whether the media conveys information while keeping a
temporary loading image from failing an otherwise valid mount.

#### Semantic modifiers

The initial general modifiers are:

```boring
Image.asset("warning")
    .accessibilityLabel("Warning")

Text("Settings")
    .accessibilityHeading(level: 1)

UserSummary(name: name, status: status)
    .accessibilityGroup(label: name)
```

- `.accessibilityLabel(string)` replaces the computed name of the receiving semantic element.
- `.accessibilityHint(string)` supplies optional usage guidance, not a second label.
- `.accessibilityValue(string)` overrides a display value for a custom read-only component; standard
  editable controls continue to expose their typed live value and cannot be made read-only by this
  visual override.
- `.accessibilityHidden(bool hidden = true)` removes the receiver and its semantic descendants from
  the accessibility tree without changing layout, painting, or ordinary input.
- `.accessibilityHeading(int level)` marks heading level `1...6`; invalid constants are checker
  errors and invalid dynamic values fail the view update.
- `.accessibilityGroup(string? label = nil)` inserts a semantic grouping node while preserving its
  descendants as children. Supplying a label names the group but does not erase child names.
- `.accessibilityLive(LiveRegion politeness = .Polite)` requests announcement when the element's
  semantic text/value changes; `.Assertive` is available for urgent, exceptional updates.

Modifiers are ordered wrappers. A label outside a grouping wrapper names the group; a label inside
names the child. Hiding an outer wrapper hides its complete semantic subtree. Visual modifiers such
as color, font weight, clipping, opacity, or style do not alter semantics except that a node removed
from rendering/mounting is also absent from the semantic tree.

#### Actions and custom components

Semantic actions use the same UI event dispatcher and transaction rules as pointer/keyboard input.
An accessibility activate request invokes `onActivate`; setting a toggle, editing text, scrolling,
focusing, selection, deletion, and reordering call the corresponding component behavior rather than
a parallel accessibility-only callback. Disabled controls expose disabled state and omit mutating
actions while remaining readable.

The backend-neutral action set initially includes focus, activate, set text/value, set selection,
increment/decrement, scroll in each supported direction, expand/collapse, dismiss, and named custom
actions. A future public `.accessibilityAction(name:)` may expose named actions to application views;
the first vertical slice does not need it because its controls already have semantic operations.

Boring `view` declarations normally obtain complete semantics by composing standard controls. A
native/Rust primitive implements an internal `SemanticComponent` contract that returns semantic
properties and action handlers from its mounted state. This contract receives stable ids and typed
event callbacks from the runtime; it cannot mutate the platform accessibility tree directly. A
custom visual element intended to behave as a button should compose/use `Button`, preserving its
keyboard and action behavior, rather than attach only a button role to an inert drawing.

#### Accessibility focus and environment

Assistive-technology reading focus is distinct from keyboard focus and is owned by the platform
adapter. A semantic focus action on a keyboard-focusable control requests ordinary UI focus and
therefore updates `.focused` bindings; moving a screen-reader cursor among static elements does not.
When its semantic node disappears, the adapter chooses an appropriate nearby node using the updated
tree and platform convention.

Modal presentation publishes only the active modal subtree plus the required dialog ancestry for
traversal, then restores the underlying semantic tree on dismissal. Announcements generated during
a failed/rolled-back tree update are discarded.

The accessibility environment exposes read-only preferences such as reduced motion, increased
contrast, text scale, and screen-reader/assistive-technology activity. Standard components consume
them internally; application views may declare the corresponding `env` values when their content or
motion genuinely needs to adapt. These preferences affect presentation and animation without
changing semantic identity.

### Environment, theme, and control styles

**Design decision (2026-09-30):** inherited runtime context is represented by an explicit `env`
field on views that read it. Environment values are neither process globals nor silently declared
fields.

```boring
Theme appTheme():
    Theme.system()
        .color(.Accent, Color.hex("#7357FF"))
        .metric(.ControlRadius, 10)

def main():
    runUI(theme: appTheme()):
        Window(title: "Todos"):
            TodoList()

view Card:
    env Theme theme
    string title

    body():
        Text(title)
            .font(theme.font(.Title))
            .background(theme.color(.Surface))
```

`env` is a third `view` field category alongside `state` and `bind`: it is implicitly and always
`let`, a read-only value
resolved from the nearest provider in the mounted ancestor path. It has no constructor argument and
cannot be assigned; `env let` is redundant and `env var`/`env mut` are checker errors. A changed
inherited value updates the field and invalidates views that depend
on it. Standard controls declare internal environment dependencies for theme, locale, layout
direction, enabled state, and accessibility preferences; application views declare them only when
they read those values directly.

`runUI(theme: ...)` installs the root `Theme`; `.theme(...)` wraps a subtree with a nearer provider.
A future general `.environment(value)` can use the declared environment type as its key, with
distinct wrapper types when an application needs several values of the same underlying type. This
runtime, per-subtree mechanism is separate from compile-time `@inject` dependency resolution.

`ButtonRole` is closed semantic information; visual styles are extensible values implementing
component-specific traits:

```boring
trait ButtonStyle:
    # Conceptual: produce appearance from a read-only ButtonStyleContext.
    <View> body(ButtonStyleContext context)

struct RoundedButtonStyle as ButtonStyle:
    float radius

    <View> body(ButtonStyleContext context):
        context.label().padding(8).clip(RoundedRect(radius: radius))

Button("Save").style(RoundedButtonStyle(radius: 12))
```

`<View>` is the exact opaque return spelling. A button style context exposes the label, role,
enabled/pressed/hovered/focused state, and relevant environment values read-only. The
style controls appearance but cannot replace activation, focus, keyboard, or accessibility
semantics. Separate `ButtonStyle`, `ToggleStyle`, `TextFieldStyle`, and `ListStyle` traits keep their
different configuration surfaces typed. Zero-field built-in styles are zero-cost values rather
than public singleton identities; parameterized application styles are ordinary structs.

#### Theme value and inheritance

**Design decision (2026-10-01):** `Theme` is a concrete immutable library value, internally backed
by shared persistent tables. It is not a trait and not a process global. `Theme.system()` is complete
on every supported platform; builder-style overrides return a new theme sharing unchanged data.
Applications can therefore replace one token without implementing an entire theme contract.

`runUI(theme:)` installs the root value; omitting it uses `Theme.system()`. `.theme(theme)` installs a
complete derived value for a subtree. Specialized providers such as `.tint(color)`,
`.buttonStyle(style)`, or `.textScale(...)` may override one environment facet without manufacturing
a new public theme type. Nearest provider wins, and removal reveals the next ancestor value.

Theme equality is identity/version based rather than a deep comparison of style objects. Standard
components register the particular token/style categories they consume, so a color-only override
does not force unrelated layout work. An application view declaring `env Theme theme` is
conservatively invalidated by any change to the nearest theme because arbitrary method calls cannot
be dependency-tracked token by token.

`Theme.system()` follows the platform light/dark appearance, accent, default fonts, density, and
control conventions. `Theme.fixed(ColorScheme scheme)` freezes only the appearance selection while
retaining platform metrics. System appearance changes update the inherited theme and repaint/reflow
only where resolved tokens differ. A theme may supply light, dark, increased-contrast, and disabled
variants; a single override value applies to all variants when the application intentionally wants
that behavior.

#### Semantic tokens

Colors are addressed by `ColorRole`, initially:

```text
WindowBackground, Surface, ElevatedSurface,
Text, SecondaryText, PlaceholderText, DisabledText,
Accent, OnAccent, Destructive, OnDestructive,
Border, Separator, FocusRing, Selection, Shadow
```

`theme.color(role)` returns an explicit `Color` resolved for appearance/contrast. `Color` supports
sRGB construction (`rgb`, `hex`) and alpha in the initial API; display-list conversion handles the
linear working space. Component styles may derive additional colors but semantic application code
should prefer roles so dark/high-contrast variants remain functional.

Typography uses `FontRole`, initially `LargeTitle`, `Title`, `Headline`, `Body`, `Label`, `Caption`,
and `Monospace`. A resolved `Font` contains a prioritized family list, logical size, weight, style,
line height, and letter spacing. `.font(.Title)` is shorthand for the inherited role; `.font(Font)`
uses an explicit value. Accessibility text scale applies according to the role's scaling policy
before measurement. A platform/system font fallback remains at the end of every family list.

Layout/control metrics use `MetricRole`, initially:

```text
ControlHeight, MinimumHitSize, ControlRadius, ControlBorderWidth,
FocusRingWidth, SeparatorThickness, SpacerMinimum,
ScrollbarThickness, SheetCornerRadius, ShadowRadius
```

Values are finite non-negative logical units and follow the ordinary invalid-value policy. Metrics
are defaults, not hard caps: `MinimumHitSize` contributes to a control's intrinsic minimum layout
size, and explicit outer layout wrappers retain their ordinary proposal/overflow behavior. Focus
rings and shadows do not affect measurement.

Adaptive stack spacing is a theme query over axis plus the adjacent children's `SpacingRole`.
Initial roles include text, control, container, separator, and neutral/custom. Native components
publish their role; transparent application views forward the nearest meaningful descendant roles.
The platform-neutral fallback table provides small text-to-text gaps, standard label-to-control
gaps, and larger section/container gaps. Explicit `spacing:` bypasses the table entirely.

#### Component styles

The active theme stores type-erased defaults separately for `ButtonStyle`, `ToggleStyle`,
`TextFieldStyle`, and `ListStyle`. The built-in styles consume semantic color/font/metric roles, so
token overrides automatically restyle controls. The theme maps each `ButtonRole` to a style choice;
`.Normal`, `.Destructive`, and `.Cancel` remain semantic roles, while `Primary`, `Secondary`,
`Plain`, and `Compact` are built-in visual styles.

A style's `<View> body(Context)` produces only the presentation subtree hosted inside the standard
control's semantic/event/focus wrapper. The context is read-only and includes its label/content,
role/value, enabled, hovered, pressed, focused, validation, and relevant environment state. For a
button, `context.label()` returns the standard label view for composition. The style cannot replace
the control's binding, actions, focus node, keyboard behavior, or accessibility role.

Transient context changes such as hover, press, focus, validation, or toggle value invalidate the
style presentation with the narrowest paint/layout cause it declares. A custom style is an ordinary
Boring struct implementing the component-specific trait; it need not be thread-safe because UI
evaluation is executor-confined. The environment stores one type-erased style value per category,
not one allocation per control rebuild.

`Button(...).style(MyButtonStyle())` applies to that concrete control and must precede a general
modifier that returns opaque `<View>`. `.buttonStyle(...)` is the general inherited provider for all
buttons in a subtree; corresponding providers exist for other component categories. Local `.style`
wins over an inherited provider, which wins over the theme default. Neither form changes
`ButtonRole` or any semantic behavior.

#### Theme validation and fallback

A constructed theme is validated once when installed. Missing entries inherit from its base;
`Theme.system()` itself has no missing entries. Invalid numeric tokens fail the view update/root
mount. Development builds warn when foreground/background pairs chosen by built-in styles fall below
the runtime's contrast target, while custom style authors remain responsible for combinations they
draw themselves.

Themes contain immutable values and style factories only, never mounted state, window handles, GPU
resources, locale-specific rendered strings, or application model references. GPU colors, fonts,
and shadows are resolved/cached downstream by the display-list renderer, so swapping a theme does
not require remounting controls or rerunning lifecycle hooks.

### Focus

**Design decision (2026-09-30):** focus is synchronized through ordinary `state`/`bind`; it does
not require a third storage category analogous to SwiftUI's `@FocusState`.

For one control, `focused` conceptually receives `bind bool` (implicitly `var`):

```boring
view Form:
    state bool nameFocused = false

    body():
        TextField(value: name, label: "Name").focused(nameFocused)
```

Native focus writes `true`/`false` through the binding. Assigning `true` requests focus when the
control is mounted; assigning `false` resigns it.

For several controls, an overload receives a bound optional token plus the value representing that
control:

```boring
enum Field:
    newItem
    item(int id)

view TodoList:
    state Field? focus = nil

    body():
        Column:
            TextField(value: newTitle, label: "New item title")
                .focused(focus, equals: Field.newItem)

            for item in items with item.id:
                TextField(value: item.title, label: "Item title")
                    .focused(focus, equals: Field.item(item.id))
```

Setting `focus` to a token requests the matching mounted control; setting it to `nil` clears focus.
When native focus moves, the binding is updated in the same event transaction. If the focused view
unmounts, it clears the binding only if the binding still contains its own token. Reordering a keyed
iteration preserves focus because the mounted identity survives the move.

Focus tokens satisfy `IdentityKey` and must be unique among controls connected to the same bound
slot. Connecting one boolean focus binding to multiple mounted controls is likewise invalid. The
runtime validates registrations before committing the rebuilt tree. A duplicate fails that update
and retains the previous valid tree, with the same initial-mount/reporting policy as duplicate list
keys; silently choosing one control would make two-way synchronization incoherent.

#### Focus scopes and traversal

Each `Window` owns a root focus scope and at most one logical keyboard-focused mounted identity.
`Sheet` creates an active child scope that traps traversal while presented. A general
`.focusScope()` wrapper creates a nested scope without trapping by default; Tab may leave it for the
next control in its parent scope. `.focusScope(trap: true)` is reserved for components with genuine
modal semantics and should rarely appear in application code.

The default sequential order is structural order after conditionals and keyed loops have produced
the mounted tree. Disabled, unmounted, hidden, or explicitly non-focusable nodes are skipped.
Source order remains the logical order under right-to-left layout; visual mirroring does not reverse
Tab. Portals participate in their active presentation scope rather than at the `Sheet` placeholder's
layout position.

`.focusOrder(float value)` overrides ordering among siblings in the nearest scope. Values must be
finite, default to `0`, sort ascending, and tie by structural order. It changes keyboard traversal
and accessibility order together; there is no separate visual-only tab index. Negative values move
a control earlier without removing it. `.focusable(false)` removes a control from keyboard focus
while leaving layout, hit testing, and accessibility reading order intact; disabled controls are
already ineligible for keyboard focus without this modifier.

Tab/Shift-Tab move forward/backward in this order. At an ordinary nested-scope edge traversal
continues in the parent; a trapped scope wraps internally. If a platform convention does not use
Tab for all controls, the platform/theme may restrict the eligible control classes while explicit
full-keyboard-access preferences override it.

Arrow/directional navigation first searches the current `.focusSection()`; if none exists or no
candidate is found, it searches the active scope. Candidates must lie in the requested geometric
half-plane. The runtime minimizes main-axis distance, then cross-axis distance, then traversal
order. A focus section contributes the union of its descendants' bounds when moving between
sections, then applies the same rule inside the selected section. Platforms without directional
focus navigation need not bind arrow keys globally, but gamepad/TV/accessibility adapters use the
same algorithm.

#### Requests, default focus, and synchronization

Writes through `.focused(...)` create a focus request processed after the current tree commit and
layout. If its target is not mounted or eligible, the bound `true`/token is retained as a pending
request; it is reconsidered on later commits. The current focus is cleared when the request names a
different unavailable target. Writing `false` resigns focus only when that boolean's control is
focused; writing `nil` clears the target associated with that token binding.

User, traversal, accessibility, and platform focus changes update both sides in one event
transaction: the previous control receives `false`/`nil`, then the new control receives
`true`/its token. A focusable control without `.focused` still participates but has no application
binding to update. If several independent bindings request focus in one transaction, the last
explicit write wins; simultaneous initial requests tie by structural order, and losing bindings
are synchronized back to `false`/`nil`.

`.defaultFocus()` marks the fallback control for its nearest scope. At most one eligible default may
be mounted per scope; duplicates fail validation like duplicate focus tokens. When a scope first
activates with no explicit/pending request and no restorable identity, the runtime chooses its
default. If none exists, desktop keyboard platforms may choose the first eligible control while
touch platforms may leave focus empty to avoid opening a software keyboard. Initializing a bound
focus state to `true` remains the way to make an explicit cross-platform initial request.

The resolution priority on scope activation is therefore: newest explicit bound request, retained
focus identity, `.defaultFocus()`, then the platform initial-focus policy.

#### Loss and restoration

Keyed reconciliation preserves focus because the mounted identity survives movement. If the focused
identity becomes disabled, non-focusable, hidden, or unmounts, the runtime synchronizes its binding
to `false`/`nil` and chooses no replacement automatically during an ordinary update; a pending
explicit request or subsequent traversal can choose one. This avoids surprising focus jumps after
deletion.

Window deactivation is different from focus removal. The logical identity and binding remain, its
visual focus state becomes inactive, and text input/IME is suspended. Reactivation restores native
focus to that identity if it is still eligible, otherwise it applies the activation priority above.

Presenting a modal scope records the previously focused mounted identity, deactivates the underlying
scope, synchronizes its focus binding to `false`/`nil`, and resolves initial focus inside the modal.
While the modal is active, requests aimed at the underlying scope remain pending and cannot escape
the trap. On dismissal, the newest such explicit request wins; otherwise the runtime restores the
recorded identity when it still exists and is eligible, then falls back to the underlying scope's
default/platform policy. Restoration writes the corresponding binding again in the dismissal
transaction.

Nested modals maintain a restoration stack per window. Closing a scene discards that stack. Focus
restoration across actual unmount/remount or application relaunch is not implicit; applications can
persist and restore their ordinary focus token state when that behavior is desired.

### List, selection, deletion, and reordering

**Design decision (2026-09-30):** `List` consumes keyed child subtrees; it does not also own or
repeat the source collection. Dynamic identity remains the responsibility of the array-block's
`for` statement.

```boring
List(
    selection: selection,
    onMove: (from, to): items.move(from, to),
    onDelete: (indices): items.removeIndices(indices),
):
    for item in items with item.id:
        ItemRow(item)
```

`selection` is optional. For single selection its contract is conceptually `bind ID?` (implicitly
`var`); for multiple selection an overload accepts `bind {ID}` (implicitly `mut`). `ID` must match the key type produced by
the child `for`. User selection writes through the binding in the current event transaction.
During update, a single selected key that no longer exists is cleared, and a multi-selection is
intersected with the remaining keys.

The optional semantic callbacks enable their corresponding platform interactions:

- `onMove(int from, int to)` enables reordering. Both positions refer to the current displayed
  child order; `to` is the element's final zero-based index after the move. The callback must mutate
  the source model. If it does not, the next rebuild restores the previous model order. A successful
  model move preserves mounted rows through their keys, including row state and focus.
- `onDelete([int] indices)` enables deletion gestures, keyboard commands, or platform affordances.
  Indices refer to one pre-event snapshot, are unique, and arrive in ascending order; the standard
  collection helper must remove them safely as one operation rather than shifting later indices
  accidentally.

The callbacks operate on positions because the source is commonly an array, while selection uses
stable keys because positions are not identity. Key-based move/delete callback variants can be
added if real applications demonstrate a need; they are not required for the initial editable-list
target.

The standard mutable-array helpers used above have exact snapshot semantics:

- `array.move(int from, int to)` moves one element so it occupies final index `to`. Both indices
  must be within the array before mutation; equal indices are a no-op. Removing the source and
  inserting at the requested final index defines the result, so moving index 1 to index 3 in
  `[a, b, c, d]` produces `[a, c, d, b]`.
- `array.removeIndices([int] indices)` requires unique, strictly ascending indices into the
  pre-mutation array and removes all of them atomically. Implementations may delete from the end,
  but the public indices always describe the original snapshot.

Invalid indices or ordering are programmer errors reported before either helper mutates the array.
`List` guarantees valid callback arguments; direct callers receive the normal runtime collection
error. Each helper is one logical mutation and therefore emits one observed/state invalidation.

`List` is a semantic, scrolling collection and clips to its viewport. The first implementation may
mount every row; lazy/virtualized mounting is a later performance feature and must document whether
offscreen row-local `state` is retained. Virtualization must never change collection identity or
selection semantics.

### Modal presentation

**Design decision (2026-10-01):** Boring-rendered modal content is mounted through semantic portal
components in the view tree. It is not modeled as a one-shot native action and does not require an
array-block modifier on an arbitrary receiver.

```boring
view TodoList:
    state bool showingEditor = false

    body():
        Column:
            Button("Add").onActivate ():
                showingEditor = true

            List:
                # rows

            Sheet(isPresented: showingEditor):
                ItemEditor()
```

`Sheet` has `bind bool isPresented` (implicitly `var`) and exactly one root view. It occupies a
structural position but contributes no bounds to its layout parent; when presented, its child is
mounted in the containing window's presentation layer and inherits the environment from the
`Sheet` site.

Writing `true` presents and mounts the content. Native dismissal, Escape, or a platform dismissal
gesture writes `false` through the binding. Writing `false` begins dismissal; the child remains
mounted until its exit transition completes, then unmounts and loses its identity-scoped state. A
later presentation starts a new lifetime. Unmounting the `Sheet` owner cancels the presentation and
its scoped tasks even if an exit transition is in progress.

The platform chooses an appropriate default presentation (sheet on mobile, dialog/panel on
desktop), while later options can constrain size and presentation style. Modal presentation moves
focus into the content, traps traversal as required by the platform, restores focus to the previous
control on dismissal when it still exists, and exposes the correct accessibility modality.

`Alert` and `Popover` should be separate portal components because their action, focus, placement,
and accessibility contracts differ. A file/photo picker or permission prompt remains a native
`task ... throws` action: it is owned by the OS and does not mount a Boring view subtree.

## First implementation target and remaining design work

The first end-to-end target is a small editable list application: add an item, edit it, select it,
delete it, and reorder it while preserving the state of surviving rows. Define its syntax and
observable behavior before implementation; a complete future widget catalogue is not required.

### End-to-end syntax sketch

The following design-level program exercises the agreed contracts together. It is intentionally
not claimed to compile yet: `view`/`state`/`bind`/`env`, imported array-block resolution, keyed
`for ... with ...`, and the UI library are the work this design precedes.

```boring
use boring_ui.*

struct Todo:
    init(
        pub int id,
        pub var string title,
        pub var bool completed,
    )

enum TodoFocus:
    Item(int id)

view TodoRow:
    bind Todo item
    bind TodoFocus? focus

    body():
        Row(spacing: 8):
            Toggle(value: item.completed, label: "Completed")
            TextField(value: item.title, label: "Item title")
                .focused(focus, equals: TodoFocus.Item(item.id))

view NewTodoSheet:
    bind [mut Todo] items
    bind int nextId
    bind bool isPresented

    state string title = ""
    state bool titleFocused = true

    def add():
        let cleanTitle = title.trim()
        if cleanTitle.isEmpty(): return

        items.push(Todo(id: nextId, title: cleanTitle, completed: false))
        nextId += 1
        isPresented = false

    body():
        Column(alignment: leading, spacing: 12):
            Text("New item").font(.Title)
            TextField(value: title, label: "Title", placeholder: "Title")
                .focused(titleFocused)

            Row:
                Spacer()
                Button("Cancel", role: .Cancel).onActivate ():
                    isPresented = false
                Button("Add").disabled(title.trim().isEmpty()).defaultAction().onActivate ():
                    self.add()

view TodoApp:
    state [mut Todo] items = [
        Todo(id: 1, title: "Design boring-ui", completed: false),
        Todo(id: 2, title: "Implement the vertical slice", completed: false),
    ]
    state int nextId = 3
    state int? selection = nil
    state TodoFocus? focus = nil
    state bool showingNewItem = false

    body():
        Column(spacing: 0):
            Row(spacing: 8):
                Text("Todos").font(.Title)
                Spacer()
                Button("Add").onActivate ():
                    showingNewItem = true

            Separator()

            List(
                selection: selection,
                onMove: (from, to): items.move(from, to),
                onDelete: (indices): items.removeIndices(indices),
            ):
                for item in items with item.id:
                    TodoRow(item: item, focus: focus)

            Sheet(isPresented: showingNewItem):
                NewTodoSheet(
                    items: items,
                    nextId: nextId,
                    isPresented: showingNewItem,
                )

def main():
    runUI:
        Window(title: "Todos"):
            TodoApp()
```

Important implications made concrete by the sketch:

- `[mut Todo]` grants mutable element projections; the enclosing `state`/`bind` defaults to `mut`
  for collection structure. A keyed loop variable preserves the projection needed by `TodoRow`'s
  `bind Todo item`.
- `items.move(from, to)` and `items.removeIndices(indices)` are proposed standard collection
  helpers required by `List`'s semantic callbacks; they do not exist in the shipped collection API
  yet. `removeIndices` consumes unique ascending snapshot indices and performs safe bulk removal.
- Adding uses a modal with its own draft `state`; dismissing destroys that draft. The parent-owned
  collection, id counter, and presentation flag cross the boundary through `bind`.
- Editing a row writes directly into its parent collection element. Reordering changes positions
  while stable keys preserve row identity and focus. Deletion removes identities and causes the
  list to clear/intersect selection as specified above.
- The example uses the `FontRole.Title` semantic token through its contextual `.Title` shorthand.

Minimum component set for the vertical slice:

| Area | Components |
|---|---|
| Application and layout | Window, `Column`, `Row`, `Layer`, `Spacer`, `Separator` |
| Display | `Text` (`Image` and `Video` use the future media pipeline but are not required by the first vertical slice) |
| Input | `Button`, `TextField`, `Toggle` |
| Dynamic content | `Scroll`, keyed `List` |
| Presentation | `Sheet` |

The contracts above specify parameters and defaults, bindings, events, size negotiation, focus and
keyboard behavior, and accessibility semantics for this set, with Boring usage examples.

The shared contracts are sufficiently defined to begin the vertical slice. This inventory separates
implementation work from features deliberately deferred beyond it:

1. **State and inputs:** `state`, `bind`, `env`, initial-state seeds, slot layout, binding handles,
   keyed element projections, executor confinement, implicit binding permissions, and ephemeral
   ordinary inputs and seed-change development warnings are defined above. The conceptual
   typed-location capabilities still need concrete generated Rust interfaces.
2. **Identity:** structural identity, conditional branches, `IdentityKey`, `Identifiable`, keyed
   dynamic rows, key stability, state reset on removal, and transactional duplicate-key failure are
   defined above. The compiler still needs native blanket-bound support for `IdentityKey`, but no
   identity semantics remain open for the vertical slice.
3. **Layout:** size negotiation, logical units, finite frame constraints, explicit filling, ordered
   modifiers, deterministic priority/flexibility allocation, minimum-size overflow, axis-specific
   alignment, baseline fallback, adaptive spacing, clipping, hit testing, and single-axis scrolling
   are defined above. Programmatic scroll position, arbitrary clip paths, percentage lengths, and
   lazy layout remain deferred features rather than blockers for the vertical slice.
4. **Events and bindings:** `bind` ownership, contextual projection (including keyed collection
   elements), mutation permissions, callback capture, executor confinement, physical-event
   propagation, semantic actions, pointer capture, synchronous batching, and callback error handling
   are defined above. The public low-level event-response API needed by third-party custom controls
   and the unified view-tree error-handler API are deferred.
5. **Lifecycle and async:** commit-relative mount/task timing, deterministic nested-hook ordering,
   teardown access, identity-scoped task start/restart/cancellation, task-run generations, stale
   access protection, observed subscription replacement, cancellation observability, and task error
   reporting are defined above. A unified optional view-tree error boundary remains deferred.
6. **Accessibility:** stable semantic identities, default component roles/names/values/actions,
   transparent layout containers, semantic modifiers, modal exposure, focus separation, validation,
   and the backend-neutral custom-component contract are defined above. Named custom actions and
   richer document semantics are deferred beyond the vertical slice.
7. **View typing and native boundary:** `body()` has implicit `<View>` return, modifiers use concrete
   generic wrappers behind existing opaque-return syntax, `[dyn View]` is the heterogeneous boundary,
   and native primitives expose importable Boring facades with `body(): native`. The compiler/runtime
   ABI remains an implementation detail but the source and allocation semantics are fixed.
8. **Rendering:** the display-list vocabulary, wrapper paint order, group opacity, clips, transforms,
   logical/device coordinates, color blending, text/media resources, dirty causes, frame coalescing,
   device-loss recovery, and headless test oracle are defined above. Advanced paths, gradients,
   filters, custom shaders, and damage tracking are deferred.
9. **Theme and styles:** immutable inherited `Theme`, system/default behavior, semantic color/font/
   metric/spacing tokens, component-specific style traits, local versus subtree precedence,
   validation, and render-resource separation are defined above. Platform-specific token values are
   implementation data to establish during the visual spike, not an open API contract.
10. **Animation:** transaction and value-triggered animation, interpolation/retargeting, presentation
    geometry, insertion/removal retirement, keyed moves, modal transitions, lifecycle interaction,
    and reduced-motion behavior are defined above. Completion callbacks, keyframes, matched geometry,
    and user-defined animatable types are deferred.

### Implementation sequence and acceptance criteria

Implementation should proceed in independently testable layers rather than beginning with the full
widget catalogue:

1. **Language prerequisites:** add parser/AST/checker/lowering support for `view`, `state`, `bind`,
   `env`, `body(): native`, and keyed `for ... with ...`; make imported callable signatures
   participate in array-block resolution; generate the typed state/binding interfaces; add
   `IdentityKey`/`Identifiable` support and the two mutable-array helpers.
2. **Headless reactive core:** implement descriptors, mounted identities, typed state slots,
   bindings and keyed element projections, environment lookup, reconciliation, event transactions,
   lifecycle hooks, and identity-scoped tasks. Test conditional identity, reorder preservation,
   stale handles, duplicate keys, batching, and the 64-pass loop guard without a renderer.
3. **Headless layout and scene production:** implement the agreed primitives, modifier wrappers,
   layout negotiation, hit-test data, accessibility nodes, and display-list output. Golden tests use
   deterministic font/media fixtures and compare semantic scene data rather than GPU pixels.
4. **Desktop platform bring-up:** connect one desktop host to window/input/IME/clipboard,
   accessibility, text shaping, and GPU rendering through the platform interfaces. macOS is a
   practical first host for the current development environment; this choice must not leak into the
   core APIs, and Windows/Linux adapters follow the same contracts.
5. **Vertical-slice controls:** implement `Text`, `Button`, `Toggle`, `TextField`, `Scroll`, keyed
   `List`, and `Sheet`, then add theme/style and focus behavior required by those controls. `Image`,
   `Video`, advanced animation, navigation, virtualization, and low-level custom event APIs may land
   later because the target application does not depend on them.
6. **Todo acceptance application:** compile the sketch above as an executable example and keep it as
   an integration test of the language and UI runtime boundary.

The slice is accepted when the example launches; adds and edits text through IME-correct input;
handles default and cancel commands; selects, deletes, and reorders rows without losing surviving
row identity or focus; destroys an uncommitted sheet draft on dismissal; produces a usable
accessibility tree; rejects duplicate keys without corrupting the mounted tree; and passes the
headless reconciliation, layout, and display-list tests. These criteria are the development start
line; the deferred APIs above do not need speculative designs before implementation begins.

## Comparative table — SwiftUI → boring-ui

| SwiftUI | Role | boring-ui equivalent |
|---|---|---|
| `struct MyView: View` | declares a view | `view MyView:` |
| `var body: some View` | declarative body with an opaque static return type | `body():` with implicit existing `<View>`/Rust `impl View` return; array blocks erase only heterogeneous children to `[dyn View]` |
| `@State private var x` | view-local observed state | `state T x = ...` — persistent and private; permission defaults to `var` for value-like types and `mut` for structs/collections (§3) |
| `@Binding var x: T` | read/write reference to a parent's state, not owned here | `bind T x` field with the same permission defaults; the parent passes its `state`/`bind` lvalue directly and the expected field contract creates the persistent slot handle (§3) |
| `@ObservedObject var model: Model` | external reference to a shared, not-owned-here model | struct + `mut Model'observed model` field, populated via an `init(Model'actor'observed model): self.model = model` constructor parameter rather than constructed by this view — both the field and the parameter are real, shipped `'observed` syntax today (`docs/book.md`'s "'observed" section) — subscribes at mount unconditionally, no `state` needed |
| `@StateObject var model = Model()` | model *owned* by this view, created once, survives rebuilds | `state Model model = Model()` (implicitly `mut`, no `'observed`) constructed inline in the view declaration — "created once" follows from its identity-keyed persistent slot, not the throwaway rebuilt value |
| `ObservableObject` / `@Published` | observable model, Combine-driven, property-level in the newer `@Observable` macro | plain `struct` + a `'observed`-qualified reference on the referencing view's field — see §3 |
| `Text`, `Button`, `VStack`, `HStack` | base widgets | `Text`, `Button`, `Column`, `Row` — to be written as the actual `boring-ui` stdlib |
| `@ViewBuilder` (implicit result builder on `body`) | lets `body` read as nested indentation | the array-block sugar (§1, shipped) |
| `.onTapGesture { }` / `Button(action:)` | interaction callback | `.onActivate (): ...` for semantic controls — existing trailing-closure sugar |
| `TextField("...", text: $name)` | labeled editable input bound to state | `TextField(value: name, label: "Name")` where `value` is declared `bind string` (implicitly `var`); the control is custom-rendered, retains transient editing state, and uses a native text-input service for IME/software keyboards (§3, §5) |
| `List(items) { }` / `ForEach(items, id: \.id)` | dynamic content with stable per-element identity | `for item in items with item.id:` inside the array-block sugar; omit `with` when the element implements `Identifiable` (§4) |
| `Identifiable` protocol | supplies the stable identity `ForEach` needs | standard `Identifiable` trait with associated `ID as IdentityKey` and ordinary `req ID id()` method; explicit `with` remains available (§4) |
| `.onAppear { }` / `.onDisappear { }` | lifecycle hooks | ordered `.onMount ():` / `.onUnmount ():` view modifiers scoped to the wrapper identity (§4) |
| `.task { }` / `.task(id:)` | lifecycle-scoped async action | `.onTask () task:` / `.onTask(id: value) () task:`; starts on mount, restarts on id change, and cancels on unmount (§4) |
| `@Environment` / `.environment()` | inherited per-subtree runtime context | explicit read-only `env T name` view field plus root/specialized providers such as `runUI(theme:)` and `.theme(...)`; a changed nearest provider invalidates dependent views (§ Initial component contracts) |
| `@EnvironmentObject` / `.environmentObject()` | inherited shared observable model | `env Model'observed model` combines subtree lookup with the ordinary automatic observed subscription; compile-time application-wide construction without subtree overrides remains available separately through shipped `@inject`/`@provide` ([book.md §33](../book.md#33-dependency-injection)) |
| `.sheet(isPresented:) { }` | modal presentation | `Sheet(isPresented: flag):` portal component with one root, a bound presentation flag, identity-scoped content, and platform presentation defaults (§ Initial component contracts) |
| `NavigationStack` / navigation push | hierarchical navigation | **open question** — not required by the first editable-list vertical slice |

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
