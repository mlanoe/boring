# Draft — boring-ui: a SwiftUI-flavored, custom-rendered GUI toolkit for Boring

Status: **working draft**, not a spec. Nothing in this document is implemented yet except the one
language mechanism explicitly marked as shipped below. This is a starting point to be enriched
before any implementation work begins — several rows in the comparative table are deliberately left
as open questions rather than guessed at.

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
  mechanisms (`'observed`, `@state`) never interact.
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
*same* trailing-closure call shape Boring already had, resolved by looking up the callee in the
same-file signature table:

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
interpreter_build 4/0, interpreter_functional 83/0).

**Not yet designed**: a keyed form for `for`-generated dynamic children (see Identity & lifecycle,
below) — something like `for item in items keyBy: item.id:` — needed once list reordering has to
preserve per-element identity/state, not resolved yet.

### 2. `view` — a declaration kind distinct from `struct`

A `view` is not sugar for `struct ... as View:`. Its fields have different storage semantics: a
`struct`'s fields are ordinary owned data, moved/copied/borrowed with the value; a `view`'s
instance is **rebuilt on every refresh** (its `body()` runs again, producing a fresh tree to diff
against the previous one — see Identity & lifecycle), so a `view`'s reactive fields cannot simply
live inline in that throwaway value. They need to be handles into storage that survives across
rebuilds.

```boring
view FormView:
    @state var string name = ""

    body():
        Column:
            TextField(value: name).onInput (s): name = s
            Button("Submit").onClick (): print "submitted {name}"
```

`var`, not `mut` — see "Binding keyword" under §3. `@state` is an attribute, not a qualifier (§3):
it never changes `name`'s Rust representation, so the ordinary, unmodified scalar rule applies
(`var` for a rebindable scalar — see `docs/book.md`/`CLAUDE.md`'s "Binding × mutability (scalars)").

### 3. `'observed` and `@state` — sharing and reactivity, kept as two separate axes

Two questions turned out to be independent and are answered by two different mechanisms:
*"can more than one owner reference this value?"* (an ownership-qualifier concern, `'observed`) and
*"should changing this specific field refresh a specific view?"* (a per-field, per-view concern,
`@state`). Earlier revisions of this section conflated them into one qualifier (`'state`); kept
apart, each ends up simpler and each generalizes correctly beyond `boring-ui`.

#### `'observed` — a general, composable sharing suffix (not GUI-specific)

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

#### `@state` — a field attribute meaning "refresh this view", exclusive fields only

`@state` never changes a field's Rust representation — it's a pure compile-time marker (the same
kind of thing `@derive`/`@serde`/`@error` already are), because its whole job is telling the
transpiler *where to insert a refresh call*, not *how to store the value*. Unlike the version of
this section before review, it applies to **exclusive fields only** (no qualifier, or
`'inline`/`'owned`) — it does not apply to `'observed` fields at all (see below for why):

- **On an exclusive field** — this is the *only* mechanism available at all, since there is no
  subscriber list anywhere for such a field (nothing else can ever reference it to subscribe). The
  one and only possible observer is always exactly the same `self` performing the write, so the
  transpiler inserts a **direct, statically-resolved call** (`self.__schedule_refresh()`) right at
  the recognized assignment site — no lock, no list, no `Subscription`, nothing stored anywhere.
  This holds even through a `Binding`-mediated write from a child (below): the binding's `write`
  closure still executes as the parent's own captured `self`, so the same direct call applies. The
  field's runtime representation is therefore *identical* to an unmarked field of the same type —
  `@state`'s entire cost is that one inserted call.
- **On an `'observed` field, no attribute needed at all — subscription is automatic and
  unconditional.** An earlier revision put `@state` here too, deciding at mount whether to
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
# view-local, exclusive — @state is the entire refresh mechanism
view FormView:
    @state var string name = ""

# — versus, once `name` needs to be shared across sibling views —

struct FormModel:
    var string name = ""
    def setName(string s): name = s

view FormView:
    mut FormModel'observed model = FormModel()   # no @state — subscribes at mount unconditionally;
                                                  # any def call on `model` then refreshes every
                                                  # view that holds an 'observed reference to it
```

"Promote local state to a shared model" stays a plain refactor either way (extract the field, change
its qualifier) — it just also means dropping `@state`, since the destination no longer needs it.

##### Binding keyword

Since `@state` never wraps the type, the *ordinary*, already-existing binding rules apply
unmodified, keyed only on whatever qualifier (if any) is actually present — `@state` itself changes
nothing about them:

- No qualifier, scalar (`@state var string name = ""`) — plain scalar rule: `var`, never `mut`
  (`docs/book.md`/`CLAUDE.md`, "Binding × mutability (scalars)" — a bare scalar has no `def` methods
  for `mut` to unlock). This is exactly right for the mechanism too: a scalar's only kind of "change"
  *is* rebinding (`name = "x"`) — there's no separate "mutate in place" operation to distinguish —
  and rebinding is already legal on a plain `var` today, so `@state` needs nothing special here at
  all, just recognizing that assignment at the site.
- No qualifier, struct (`@state mut FormModel model = FormModel()`) — plain struct rule: `mut`
  (content-mutable, so `.setName(...)`-style `def` calls work) unless the field should also be
  rebindable.
- `'observed`-qualified struct (`mut FormModel'observed model = FormModel()`, no `@state`) — follows
  the *existing* `'actor`/`'guard` row of the qualifier/mutability table (`docs/book.md` §21): `var`
  alone is rebind-only and does **not** unlock `def` calls ("an earlier revision let `var T'actor x`
  unlock `def` calls on the qualifier's strength alone; that exception is retired") — `mut` (or
  `var mut`) is what's actually needed, exactly as for any other `'actor`/`'guard` field today.
  Nothing new: `'observed` slots into rows the checker already enforces for its base qualifier.

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
  therefore doesn't belong in the qualifier system at all — it's an attribute.
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
  defaults (opt-in `@state` for exclusive fields, unconditional for `'observed` ones).

### 4. Identity & lifecycle

**Not fully designed yet** — the piece this draft most needs enriching on.

- **Identity**: default is positional (type + static position in the block-array tree) — a
  `TextField` at a fixed textual position in `Column:` is "the same" logical instance across
  rebuilds as long as the block's structure doesn't change, same default SwiftUI/Compose use.
  Breaks down for `for`-generated dynamic children (a reordered list needs a stable per-element key,
  same problem `ForEach(id:)`/React's `key` solve) — needs its own syntax, not designed yet.
- **Mount / update / unmount**: derived from diffing a freshly-rebuilt `body()` tree against the
  previous one — new identity appearing → mount (create native/GPU resources; call `.subscribe()`
  and store the `Subscription` for every `'observed` field, unconditionally — `@state` fields need
  nothing further, their refresh call was already inserted at compile time); identity present in
  both → update in place (nothing to redo for subscriptions, they're still valid); identity missing
  from the new tree → unmount (destroy resources; `'observed` unsubscription is free via `Drop`, see
  above). One diffing pass drives all three concerns — reused directly from the performance need to
  avoid recreating widgets/native resources on every rebuild in the first place.

### 5. Native escape hatch — two flavors, not one

- **Native view** — a persistent widget bridging a real OS control (flagship case: `TextField`).
  Implements the ordinary `View`/widget trait like any custom-rendered widget — nothing special in
  Boring syntax, the bridging is entirely inside its Rust implementation. Critically: only the
  *committed* value round-trips through the app's own state (`.onInput`/`.onSubmit`) — transient
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
picker and feed its result back) — `@state` resolves that case with no extra machinery (the event
handler closure is itself the `task`, and assigns directly to the field when done). MVU remains
fully expressible by hand (plain `struct` + `enum` + `match`) for anyone who wants a centralized
reducer, but it isn't a first-class language feature here.

## Comparative table — SwiftUI → boring-ui

| SwiftUI | Role | boring-ui equivalent |
|---|---|---|
| `struct MyView: View` | declares a view | `view MyView:` |
| `var body: some View` | declarative body | `body():` method, returning a widget tree via the array-block sugar |
| `@State private var x` | view-local observed state | `@state var x = ...` — exclusive by default (no `'observed`), private by construction (§3) |
| `@Binding var x: T` | read/write reference to a parent's state, not owned here | a small stdlib `Binding<T>` struct (get/set closure pair) — needed for real, not free: an exclusive `@state` field genuinely can't be shared, so hand a child a *derived* accessor closing over the parent's own `self`, matching what SwiftUI's `$x` actually produces |
| `@ObservedObject var model: Model` | external reference to a shared, not-owned-here model | struct + `mut Model'observed model` passed in as a parameter, not constructed by this view — subscribes at mount unconditionally, no `@state` needed |
| `@StateObject var model = Model()` | model *owned* by this view, created once, survives rebuilds | struct + `@state mut Model model = Model()` (no `'observed` — exclusive, and `@state`'s direct-call path is the entire mechanism) constructed inline in the view's own declaration — "created once" falls out for free from the identity-keyed persistent slot, not the throwaway rebuilt value |
| `ObservableObject` / `@Published` | observable model, Combine-driven, property-level in the newer `@Observable` macro | plain `struct` + a `'observed`-qualified reference on the referencing view's field — see §3 |
| `Text`, `Button`, `VStack`, `HStack` | base widgets | `Text`, `Button`, `Column`, `Row` — to be written as the actual `boring-ui` stdlib |
| `@ViewBuilder` (implicit result builder on `body`) | lets `body` read as nested indentation | the array-block sugar (§1, shipped) |
| `.onTapGesture { }` / `Button(action:)` | interaction callback | `.onClick (): ...` — existing trailing-closure sugar |
| `TextField("...", text: $name)` | native-backed input bound to state | `TextField(value: name).onInput (s): name = s` — native escape hatch, committed value only (§5) |
| `List(items) { }` / `ForEach(items, id: \.id)` | dynamic content with stable per-element identity | `for item in items:` inside the array-block sugar — **keying not designed yet** (§4) |
| `Identifiable` protocol | supplies the stable identity `ForEach` needs | likely a `trait` (`as Identifiable`) requiring an `id`-shaped member — **open question**, tied directly to the `for`-keying gap above |
| `.onAppear { }` / `.onDisappear { }` | lifecycle hooks | **open question** — presumably optional recognized methods on `view` (`onMount()`/`onUnmount()`), not designed |
| `.task { }` (async work scoped to a view's mount, auto-cancelled on disappearance) | lifecycle-scoped async action | **open question** — likely `task ... throws` wired to the mount hook above, with cancellation on unmount; the wiring itself isn't designed |
| `@EnvironmentObject` / `.environmentObject()` | implicit injection down the tree, bypassing per-parameter threading | **not a boring-ui-specific mechanism** — see [`boring-di-draft.md`](boring-di-draft.md), a standalone dependency-injection/IoC design (`Trait'inject` field + a `provide Trait ...` composition-root declaration, compile-time-resolved, no runtime reflection). Covers the "resolve without naming the concrete type, across a module boundary" half of this row for free; true per-subtree ambient overrides (a different value for one branch of the view tree, à la `.environment(\.x, y)`/Compose `CompositionLocalProvider`) remain an open question left to `boring-ui` to design on top of it, not solved by it. |
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
(`@state`, a field attribute with zero representational cost) were never the same question.
