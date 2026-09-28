# Draft — a general-purpose dependency-injection / inversion-of-control mechanism for Boring

Status: **partially implemented, both `boring build` and `boring run`**. `@singleton` (§4),
`@provide`'s `pub` requirement (§3), `id`/`env` (§5-§6), `'static` (§2, checker/registry-level —
see caveat below), cycle detection (§7, best-effort — see caveat below), and a first slice of
`@inject` (§1-§2 — same-`Program` providers only,
bare-field inference against a `@singleton` provider (always) or a transient one (only when the base
type is a trait and the provider returns it bare too — a real `boring build`-specific gap for every
other transient shape, not a design limitation, see "Before implementation begins"), a struct can't
combine `@inject` with its own
`init` yet) are real and tested on both backends (`src/desugar_inject.rs`, `src/checker/mod.rs`'s
`check_di_provider_attrs`, `src/interpreter/call.rs`'s `singleton_cache`,
`tests/dependency_injection.rs`, `tests/cases/{singleton,inject}_di.br`). The self-hosted-in-Boring
interpreter (`boring/interpreter/*.br`) remains v3, deliberately deferred — see "`boring run`
parity" under Open Questions for why the interpreter/transpiler split turned out cheaper than
originally planned. **`'static` caveat**: `@inject`/`@provide` themselves accept and resolve it
correctly, but a full end-to-end `'static` example is currently blocked by two separate,
pre-existing, unrelated transpiler gaps found and filed while testing this (task_ce5a4ff9 — a bare
constructor-call return value isn't wrapped in the `&` reference a `'static` return type needs, the
same bug class already fixed for `'actor`/`'guard`/`'shared` but not yet for `'static`'s own
representation; a further, not-yet-fully-diagnosed issue for a `'static` reference to a *trait* type
specifically). **Cycle-detection caveat**: sound-by-omission, not sound-by-construction — an edge in
the dependency graph only exists where a provider's body is a bare constructor-call tail expression
(`provider_target_struct`); a cycle hidden behind a more elaborate provider body isn't caught. Still
design-only: cross-project (`[deps]`) resolution. This
is a standalone design topic, not scoped to `boring-ui` — see "Where this came from" at the bottom.

## Goal

Give Boring a way for one piece of code to depend on *an abstraction*, and have some other,
possibly distant piece of code (in a different module, a different `boring.toml` project, or a
test) supply the concrete implementation — without the dependent code naming the concrete type,
and without every intermediate caller manually threading it through as a parameter.

This is dependency injection / inversion of control in the conventional sense: the same underlying
concept behind Spring, Google Guice/Dagger, and Swift's `Dependencies` library (point-free),
regardless of which corner of a program first needs it.

## Where this surfaced, and why it's general-purpose

This came up while designing `boring-ui` ([`boring-ui-draft.md`](boring-ui-draft.md)), specifically
its `@EnvironmentObject` comparison row: SwiftUI lets a deeply nested view reference a shared model
injected once by some ancestor, without threading it through every intermediate view's parameters.
A first proposal — a `'static'observed` top-level named singleton — was rejected there: it requires
the consuming view to have direct visibility of the concrete declaration site (an exact top-level
identifier). That's a service locator reached by direct reference, not dependency injection, and it
breaks down exactly when the consumer lives in a different module or project than the provider —
which is the normal case for a real app (a view in a shared UI library, a concrete analytics client
in the app that embeds it).

The fix isn't GUI-specific, so it doesn't belong in `boring-ui`'s own draft. Anything that
constructs objects — a CLI tool, a server handler, a background job — has the identical problem
whenever it wants to depend on "a database client" or "a logger" as an abstraction rather than a
concrete type it has to know how to build. This document designs that mechanism once, and
`boring-ui`'s environment-injection need becomes one consumer of it, not its origin.

## Constraints specific to Boring

- **No runtime reflection.** Boring transpiles to Rust; there is no `ApplicationContext`-style
  container that can enumerate annotated classes and wire them up at process startup by inspecting
  types at runtime. Any mechanism modeled on a reflective container is the wrong structural fit
  from the start — see "Spring" below.
- **Auditability ethos.** Recent Boring design work (the trailing-array-block sugar, the `'observed`
  qualifier) consistently favors mechanisms with **one fixed desugaring rule**, checkable by
  reading the type signature, over anything that infers intent from usage patterns. Whatever DI
  design is chosen needs to be name-that-rule simple, not a hidden dependency graph a reader has to
  reconstruct.
- **Cross-project visibility already exists.** `boring.toml`'s `[deps]` + `use <name>.xxx`
  (`docs/book.md` §15) already gives the compiler AST-level visibility into a whole graph of
  sibling Boring projects, not just the current one. Whatever "the provider search space" means for
  DI resolution, it's natural to make it exactly the same visibility Boring already computes for
  trait-impl resolution and cross-project imports — no new project-topology concept needed.
- **An existing qualifier system.** `'static`, `'shared`, `'actor`, etc. (`docs/book.md` §21)
  already express *where a value lives* and *how many things can point at it*. A DI design that
  reinvents its own notion of "singleton vs. transient" scope, orthogonal to this table, would be
  needless duplication — see the recommended direction below.

## The three precedents, and why each does or doesn't fit

### Spring (reflective runtime container) — rejected outright

Beans registered via annotations, resolved by reflection at `ApplicationContext` startup, fails at
*startup* rather than at compile time if something is unsatisfied. This requires a runtime type
registry and reflective construction — neither exists in a Rust binary produced by `boring build`,
and retrofitting one (a hand-rolled vtable-of-constructors keyed by a runtime type-id) would be a
large, un-auditable subsystem bolted onto a language whose entire selling point is compiling down
to plain, inspectable Rust. **Not a candidate.**

### Guice / Dagger (Java) — split verdict

Guice itself is Spring's model minus the XML config: `Module`s bind interfaces to implementations
in code, resolved by an `Injector` via reflection at runtime. Same reflection dependency, same
verdict as Spring.

Dagger, its annotation-processor-based sibling, is the interesting one: `@Inject`-annotated
constructors and `@Provides` methods are read by a compile-time annotation processor, which
**generates ordinary Java code** that constructs the object graph directly — no reflection at
runtime, and the generated code is inspectable. This is structurally close to what Boring's own
compiler already does at transpile time for everything else (desugaring `'atomic`'s `+=` into
`fetch_add`, `'observed`'s mutating-call-triggered lock+mutate+notify, the trailing array-block sugar
into a `Vec` literal). Boring doesn't need a separate annotation-processing pass — the compiler
*is* already the thing that would do this rewriting. **Dagger's actual mechanism (compile-time
constructor injection, not the ceremony of `@Component`/`@Module` classes) is the strongest fit.**

### Swift `Dependencies` (point-free) — good precedent for the *key* design, wrong precedent for the *storage* mechanism

Dependencies against statically-typed keys (protocol conformances), consumed via a
`@Dependency(\.someClient)` property wrapper, checked at compile time (the key must exist, types
must match) — this half of the design translates directly to Boring's trait system (see below).

The other half — an ambient "current `DependencyValues`" bag threaded via Swift's `TaskLocal`, with
scoped push/pop override for tests (`withDependencies { $0.someClient = mock }`) — is a genuine
runtime, dynamically-scoped global. It's the right choice *for SwiftUI specifically*, because
SwiftUI views are value types silently recreated by the framework; there's no stable place to pass
an explicit container reference down by hand. Boring's `view` doesn't have that constraint forced
on it the same way (see the reframing below), and Boring's ethos leans toward "static and
explicit" wherever a static alternative genuinely exists. **The key design is worth keeping; the
ambient dynamic-scope storage is not adopted as the default** — see Open Questions for when it
might still be worth revisiting.

## A necessary reframing: DI is not the same problem as ambient tree propagation

SwiftUI's `@EnvironmentObject` actually conflates two different problems under one name:

1. **"How do I construct this object, given it needs collaborators I don't want to hand-wire
   myself?"** — the actual DI problem, solved by Spring/Guice/Dagger/Swift `Dependencies` alike.
2. **"How does an already-existing value reach a deeply nested consumer in a tree, without every
   intermediate node forwarding it?"** — a tree-shaped ambient-propagation problem.

Android's own ecosystem, which uses Dagger/Hilt for (1), did **not** solve (2) with Hilt — Jetpack
Compose has a separate, purpose-built mechanism for it, `CompositionLocal`, entirely independent of
the DI container. That split is good evidence that the two problems don't actually have one shared
best answer: a DI container resolves "what do I construct this with" once, at construction time; a
tree-scoped ambient value can vary **per subtree**, decided by whichever ancestor happens to be
rendering above a given node right now (a different `Theme` for the "dark preview" branch of a view
tree, say) — something a single program-wide provider-per-trait binding structurally cannot express.

**This document scopes itself to problem (1) only.** `boring-ui`'s per-subtree environment
overrides, if it ever needs them, are `boring-ui`'s own follow-up design, informed by the primitive
this document produces (see Open Questions) — not solved by it.

With that reframing, it turns out problem (1) alone — the general DI mechanism — already covers the
*specific* motivating case (a `view` field resolved without threading), because injection happens
by the trait's identity at the point a struct/view is *constructed*, regardless of how many
ancestors sit above it in a tree. What it does not cover is a value that must differ between two
live subtrees at once — see Open Questions.

## Recommended direction: compile-time constructor injection

Two new attributes — `@inject` and `@provide` — and, deliberately, **zero new grammar
productions**: both decorate a declaration shape Boring already has in full (a struct field, an
ordinary return-type-first function), reusing several further existing mechanisms on top of that
(defaulted parameters, the qualifier system, supertraits, labeled-argument calls, `[deps]`
visibility) rather than inventing parallel ones.

### 1. `@inject` — a field attribute, not a qualifier

```boring
trait NetworkClient:
    req [byte] fetch(string url) throws

struct UserRepository:
    @inject
    NetworkClient'shared client

    req [User] fetchUsers() throws:
        client.fetch("/users")
```

An earlier version of this draft made this a qualifier, `'inject`, composed onto the field's own
ownership qualifier (`NetworkClient'shared'inject`). **Reconsidered**: every real qualifier in
`docs/book.md` §21 earns its tick-syntax by picking a **Rust representation** (`'shared` →
`Arc<T>`, `'actor` → `Arc<Mutex<T>>`, …); `'inject` never did — the field's actual Rust type was
always going to be inherited unchanged from whichever ownership qualifier sat next to it, never a
representation `'inject` itself introduced (no `Inject<Arc<T>>` wrapper, nothing). A mechanism that
adds a compiler-synthesized default and a provider-compatibility check to an **already fully typed**
declaration, without touching that declaration's own type, is exactly what Boring's `@attribute`s
already do — and there's a precedent for exactly this granularity: a per-field `@name(args)` line
directly above a field, decorating that one field without altering its type (`docs/book.md`'s
"Field names and JSON keys": `@serde(rename = "1")` above `int assetIndex`, `docs/book.md` §24
gives the general "applies to the next declaration" rule this reuses). `@inject` follows that same
shape — one line, directly above the field it targets, the field's own type left exactly as it
would be without `@inject` at all.

This is also a real simplification, not just a change of spelling — but which of two very different
rules applies depends on a second axis this earlier phrasing hadn't yet separated out: **whether the
matched provider is also `@singleton`-attributed or not (§4)**. A singleton exists as exactly one physical
value with one fixed Rust representation, so every consumer must match that representation exactly —
there's nothing else it *could* mean. A transient (default) provider's body is really just a
value-producing expression, re-evaluated fresh at every `@inject` site that reaches it (§4) — no
different, mechanically, from writing that same constructor call directly at each call site — so
there is no fixed representation for a consumer to "match" at all; the consumer's own declared
qualifier (or lack of one) simply decides how that fresh value gets wrapped, exactly as it would for
any other constructor call assigned to a `'shared`/`'actor`/`'owned`-qualified binding today. Crossed
with whether the field itself writes a qualifier, that gives four cases:

- **Explicit field, `@singleton` provider** — a real compatibility *check*: the provider's
  declared return-type qualifier must match the field's written one exactly, or it's a compile error
  naming both sides (§"`@inject`'s relationship to the qualifier system" below) — there's only one
  physical instance in play, so this is the only shape it can take.
- **Explicit field, transient (default) provider** — no check at all, because there's nothing to
  check against: the field's own qualifier is simply how the freshly-constructed value gets wrapped
  at that call site (`Box::new`, `Arc::new`, `Arc::new(Mutex::new(...))`, …), the same as it would be
  for an ordinary bare constructor call assigned to a qualified binding anywhere else in Boring.
- **Bare field, `@singleton` provider** — **not** run through chapter 30's ordinary usage-based
  inference chain at all (§2 revises this further): the field simply **takes the provider's
  qualifier verbatim**, since that's the one and only representation the shared value actually has.
- **Bare field, transient (default) provider** — **does** run through chapter 30's ordinary
  usage-based inference chain, exactly like any other unqualified struct field, because there's no
  fixed provider-side representation to copy in the first place — the provider is just a bare
  expression, and its wrapping is decided the normal way, from how the field itself is used.

There is no possible mismatch to check for either transient case, by construction — that removes a
failure mode a naive usage-based-inference-then-check design would otherwise have (local inference
landing on `'actor` while a `@singleton` provider returns `'shared`, for a field that never wrote
either explicitly — still a real error, but only in the singleton row above, never the transient
one).

`@inject` marks the field's constructor argument as **resolved, not supplied**: it becomes
optional, exactly like an ordinary defaulted parameter (`init(float radius = 1.0)`, `docs/book.md`
§9 "Constructors") — except the default expression is compiler-synthesized (a call to the one
resolved provider) rather than user-written. This is the same omit-and-fall-back call-site
mechanic Boring already implements for default parameters; `@inject` does not need any new
call-resolution machinery, only a new source for the default expression.

**Open gap this reopens: bare function parameters.** Every attribute precedent in the language
(`@derive`, `@error`, `@serde(rename = ...)`) decorates a declaration written on **its own line** —
a struct, an enum variant, a field. A free function's parameter list is written inline, comma-
separated, on one line (`def eval(Block b, var Vars vars, ..., NetworkClient'shared client):`,
`interpreter.br:44` in `scratch-boring` — a real 15-parameter example) — there is no existing slot
in Boring's grammar for an attribute to target *one parameter among several on the same line*. The
qualifier-suffix design didn't have this problem (`'inject` could sit inline on exactly the one
parameter's type, no new line needed); the attribute design, chosen because it's the right category
for what `@inject` actually does, currently has no answer for anything but struct fields. Whether
DI ever needs to reach a bare function parameter at all is itself worth questioning — none of the
three real-world precedents (Dagger, Guice, Swift `Dependencies`) inject into an arbitrary free
function's parameters either, only into constructors (and, in Dagger's case, into fields/methods of
a class the container manages) — so restricting `@inject` to fields and `init` parameters may
simply be the right scope, not a gap to close. **Resolved in Open Questions**: checked against a
wider precedent search (pytest/JUnit 5 included, not just the original three) — fields/`init` stays
the whole scope.

### 2. Accepted qualifiers: `'shared`, `'actor`, `'guard`, `'static` always — `'observed` layers on top of `'actor`/`'guard` — `'owned` only without `@singleton` — `'inline`/scalars never

Not every qualifier in `docs/book.md` §21 makes sense under `@inject`/`@provide` — but, per §1's
transient/singleton split, this restriction lands differently depending on which side, and on
whether `@singleton` (§4) is present. The **`@inject` site's own resolved type** is *always*
restricted, regardless of mode, to `'shared`, `'actor`, `'guard`, `'static`, or (transient
only, i.e. no `@singleton` involved) `'owned` — that's the field's actual Rust representation, and
it has to be one of these no matter how the value was produced. `'observed` (`boring-ui-draft.md`
§3) is not a further alternative alongside these — it's a composable **suffix**, a second,
independently-locked field riding alongside whichever base qualifier's own storage, that layers onto
`'actor` or `'guard` (`'actor'observed`/`'guard'observed`; `boring-ui-draft.md`'s own table rejects
`'shared'observed`, for the identical reason `'shared` is already excluded here — no interior
mutability, nothing to ever notify about). A field or return type spelled `'actor'observed` is
exactly as accepted as plain `'actor`, for the same underlying reason (below). A **`@singleton`-attributed**
function's own return type is restricted to that same set, for the same reason: the return type *is*
the one shared representation every consumer will reference. A **plain `@provide` function with no
`@singleton`** isn't restricted by this section at all: it's an ordinary function returning an
ordinary value (bare, `'inline`, whatever's natural), and §1's per-call-site wrapping is what
actually produces a `'shared`/`'actor`/etc. value at each `@inject` site that consumes it — no
different from writing that same constructor call directly wherever a qualified binding needs it
today. (`'weak` forms, where one exists, are excluded
across the board on the `@inject`-site side: a weak reference can vanish, and a DI-injected field
needs to guarantee its dependency stays alive.) `'inline`, `'atomic`, and bare scalar types are hard
compile errors on the `@inject`-site side regardless of mode, for two separate reasons, not one
vague "borrow-checker friction"
(`'owned`'s conditional acceptance is covered separately, just below):

- **`'inline` is not just awkward, it's often literally impossible.** `@inject` almost always keys
  on a **trait** — that's the entire point of the abstraction. A bare trait type is dynamically
  dispatched and always heap-allocated (`Box<dyn Trait>`, `docs/book.md` "Traits as types") because
  `dyn Trait` is unsized — there is no `Sized` representation for `'inline` (a bare, no-indirection
  `T`) to store at all. This isn't friction to design around, it's a `Sized` requirement Rust simply
  won't compile.
- **Scalars have nothing for DI to abstract.** DI earns its keep either by letting behavior vary
  across implementations (needs a trait — meaningless for `int`/`bool`/`float`, which have exactly
  one "implementation") or by giving many consumers the same shared identity (`'shared`/`'actor`/
  `'guard`/`'observed`, all indirection-based). A scalar dependency is precisely what an ordinary defaulted
  constructor parameter or a plain config field already covers, with none of this machinery. This
  also follows an existing precedent: Boring's own scalar-`mut` restriction ("primitives have no
  `def` methods to unlock", `docs/book.md`) and `'atomic` being the *one* qualifier carved out
  specifically and only for scalars already treat scalars as a special case throughout the qualifier
  system — `@inject` simply doesn't grow a scalar exception of its own the way `'atomic` did.

**`'owned`, unlike the two above, is not a blanket rejection — it's incompatible with `@singleton`
specifically (§4).** `Box<dyn Trait>` is mechanically fine even for an unsized trait (a `Box` is just
a fat pointer), and a freshly-constructed `Box` on every resolution is exactly what `'owned` already
means everywhere else in the language — no mismatch there. What `'owned` genuinely cannot do is be
**shared**: a `Box<T>` is exclusive, so a function that's both `@singleton` and returns `'owned` is a
compile error ("`'owned` cannot be `@singleton` — exactly one consumer could ever hold it"), the same
way it would be for any other function `@singleton` decorates, `@provide` or not. Without
`@singleton`, `'owned` is fully legal, and is in fact the natural qualifier for a genuinely
*transient* dependency (a fresh, per-resolution instance, never shared) — see §4.

`'actor'observed`/`'guard'observed` (`boring-ui-draft.md` §3) belong in the accepted set for the
same reason plain `'shared`/`'actor`/`'guard` do: representationally, `'observed` is just a second,
independently-locked field riding alongside the base qualifier's own storage
(`boring-ui-draft.md`'s `ObservedCell<T>` shape — `{ value: <base storage>, subscribers:
Arc<Mutex<Vec<Box<dyn Fn()>>>> }`), cheap to clone at the base-qualifier layer, and *meant* for many
simultaneous holders. It's also, concretely, the single most likely real case: the
`@EnvironmentObject` row this whole document traces back to (`boring-ui-draft.md`) is exactly "a
`view` field observing a shared model provided by some ancestor" — `@inject`-ing a
`UserSession'actor'observed session` into a deeply nested `view`, resolved against a
`@provide`-attributed function elsewhere, is the direct, load-bearing answer to the original
motivating example, not an edge case bolted on afterward. Nothing about `@inject`/`@provide` needs
to know or touch `@state` at all — `boring-ui-draft.md` §3 keeps *sharing* (`'observed`, an
ownership-qualifier concern `@inject` already understands) entirely separate from `@state`, which
now doesn't apply to `'observed` fields in the first place: subscription there is automatic and
unconditional (the write path already has to lock and walk the subscriber list for whoever's on it,
so gating that behind an extra per-field opt-in bought nothing but a footgun — a view that forgets
the attribute and silently never refreshes). A `view` holding an `@inject`-resolved `'observed` field
is, by construction, already subscribed the moment it's mounted — resolution just hands over the
shared cell, exactly as if it had arrived as an ordinary constructor argument; the "no representation
of its own" property from §1 is what makes this fall out for free rather than needing its own case.

`'static` also belongs — it's arguably the *cheapest* member of the accepted set: a bare
`&'static T`, no refcount at all, and already forbidden from carrying interior mutability by
`docs/book.md` §21 itself ("`mut 'static` is a compile error, exactly like `mut 'shared`") — the
read-only guarantee the checker enforces for the rest of this set falls out for `'static` with zero
extra work. It has no `Sized` problem either: `&'static dyn Trait` is an ordinary Rust fat reference,
same as `Box<dyn Trait>`/`Arc<dyn Trait>`. But accepting it surfaces two concrete consequences worth
being explicit about rather than assuming away:

- **It needs a fourth legal construction site.** `docs/book.md` §21 currently permits constructing a
  fresh `T'static` value at exactly three sites: top level, inside `main`, or a `type let` field. A
  `@provide`-attributed function's body is none of those — it's a plain top-level function, not a
  `let` binding. Accepting `'static` under `@provide` means this document is proposing a fourth
  legal site (`@provide`-attributed function bodies) be added to that rule upstream, not merely that
  `'static` "happens to already work" here — a real, if small, extension to §21 this design would
  need to request. Note this is about `@provide` specifically, not `@singleton` (§4): a
  `'static`-returning `@provide` function is exactly-once regardless of whether `@singleton` is also
  written, because that's what `'static` already, unconditionally, means — unlike the other four
  accepted qualifiers, it never has a genuinely transient reading, `@singleton` or not.
- **The §6 test-override escape hatch breaks for a `'static`-qualified field, structurally.**
  `docs/book.md` §21 also states: "passing a value that isn't itself already `'static`-typed into a
  parameter that demands `'static` is rejected." So `UserRepository(client = MockNetworkClient())`
  (§6's whole appeal — an ordinary, freshly-constructed mock passed inline) **does not typecheck**
  when `client` is `NetworkClient'static` — the mock itself would first have to be declared at one
  of the legal `'static` sites (most realistically, a top-level `let MockNetworkClient'static M =
  MockNetworkClient()` in the test file) before it could be passed as the override. Still possible,
  but no longer the one-line inline substitution §6 advertises as the general story — a real
  ergonomic cost specific to `'static`, not shared by `'shared`/`'actor`/`'guard`/`'observed`. Anyone
  reaching for `'static` under `@inject` should be doing so because the dependency is one that
  genuinely never needs substituting in a test (a parsed embedded resource, a truly immutable
  constant table) — for anything that might ever need a mock, `'shared` remains the better default
  even though `'static` is technically legal.

Per §1's transient/singleton split, this is also where a bare `@inject`-annotated field's relationship
to chapter 30's inference chain finally divides cleanly: against a **`@singleton`** provider,
inference is skipped entirely (not just narrowed) — the field's qualifier is copied verbatim from the
one fixed, shared representation the provider already committed to (`'shared`/`'actor`/`'guard`/
`'observed`, or `'static`). Against a **transient (default, no `@singleton`)** provider, there is no
fixed representation to copy — a bare field runs chapter 30's *ordinary* usage-based inference chain,
exactly as it would for any other unqualified struct field, and can land on `'owned` there precisely
because `'owned` is legal in this (and only this) mode. `'inline` never enters the picture in either
mode, because no legal `@inject`-site representation could ever be `'inline` in the first place (the
`Sized` problem above), regardless of what the provider does.

**This has a real compiler-ordering cost, not just a call-site one — though only for `@singleton`
providers.** A bare `@inject` field matched against a `@singleton` provider takes that provider's
qualifier verbatim (above), which means the struct containing it cannot have *its own layout*
finalized until the compiler has already found that one matching provider and read off its return
qualifier. Concretely, this forces a **collection pass across the whole reachable program** — every
file in the current project (scoped below) — that builds a complete registry of `(base type, id) →
provider` *before* transpilation can finalize the Rust type of any struct with a bare `@inject` field
resolved against a `@singleton` provider. This isn't a brand-new capability (the "Constraints
specific to Boring" section already assumes the compiler has whole-program AST visibility for
trait-impl resolution and `use`/`[deps]` imports), but it is a **stronger** dependency than those:
resolving a trait impl doesn't change a struct's own field layout, only what `impl` blocks exist for
it after the layout is already fixed, whereas a bare field's copied `@singleton`-provider qualifier
*is* part of the struct's layout. Neither an **explicit** `@inject` field nor a **bare field matched
against a transient provider** has this dependency: the explicit field's Rust type is fixed
immediately from what's written (and, per §1, a transient match needs no compatibility check to
defer at all); a bare field against a transient provider just needs to know *that* a matching
provider exists (to know `@inject` applies here at all) before falling through to chapter 30's
ordinary, purely local inference — no whole-program qualifier to wait on. That's a second,
independent reason (beyond `'static`'s auditability argument above) to prefer writing the qualifier
explicitly, or to prefer a transient provider when either would do, rather than leaning on a
bare-plus-`@singleton` match as a matter of course.

**The fix: scope bare resolution to the current project only.** Rather than requiring the full,
transitively-included `[deps]` graph (potentially several git-cloned projects, per `docs/book.md`
§15) to be resolved before a bare `@inject` field can be typed, restrict bare resolution to
providers declared in the **same project** being compiled — the same scope bare `use <name>`
already uses to split one project across sibling `.br` files. A bare `@inject` field with no
matching `@provide` in the current project is a compile error, even if a provider for that type
exists in a `[deps]` project — with a diagnostic that says so explicitly ("no local provider for
`NetworkClient`; one exists in `[deps]` project `analytics-impl`, but a cross-project provider must
be written explicitly here: `NetworkClient'shared`"). This turns the expensive, network/git-lock-
sensitive part of resolution (`[deps]`) into something only the *explicit* form ever needs to wait
on, and keeps the *bare* form's cost bounded to "the current project's own files," which the
compiler needs fully parsed anyway for ordinary same-project name resolution. It also lines up
exactly with how every worked example in this document already happens to be written: the one
cross-project case (`SettingsView`'s `AnalyticsService'shared analytics`, in the worked example
below) already spells the qualifier out, and the same-project case (`ProfileBadge`'s
`UserSession'actor'observed session`) would have been free to go bare under this rule — the flagship
motivating scenario (injection across a module boundary) is exactly the case that already wants the
more auditable, explicit spelling on its own merits (see `'static`'s argument above), so this
restriction costs the ergonomic case nothing it was actually using.

`'static` is the one deliberate exception to "just copy the provider," but **not** for the reason
`'static` is usually never inferred elsewhere in the language. That existing rule (`docs/book.md`
§21) exists to stop chapter 30's inference from silently *constructing* a fresh `'static` value
outside its three legal sites — and that risk simply doesn't exist here: nothing is constructed at
an `@inject` site at all, ever. The value already exists — legally constructed inside the
`@provide`-attributed function, which (like every Boring function) already states its return type,
`'static` included, explicitly by nature of being a function declaration — and copying a reference
to an already-existing `'static` value is exactly as safe as passing one into an ordinary
`Config'static cfg` parameter anywhere else in the language; nothing new is ever built at the copy
site.

The real reason is narrower and specific to this design: `'static`, uniquely among the accepted set,
carries the two sharp, easy-to-miss consequences from §2 (the test-override escape hatch breaks; a
fourth legal construction site is needed upstream) — and the whole point of this document is that a
`@provide`-attributed provider can live in a *different project* than the `@inject` site consuming
it. A bare, unqualified `NetworkClient client` field gives a reader in the consuming project no way
to know, without going and reading the provider's source elsewhere, that they've inherited
`'static`'s test-mocking cost. Requiring `'static` to be written explicitly at the `@inject` site
makes that cost visible exactly where a reader is standing, rather than one more thing to discover
by tracing a resolution the way the "action at a distance" open question already worries about for
plain ambiguity. If a bare `@inject` field's only visible provider is `'static`-qualified and the
field itself doesn't say `'static`, that's a compile error naming the mismatch ("no compatible
provider for `NetworkClient` — the only one visible returns `NetworkClient'static`, which must be
written explicitly here").

**Why this doesn't generalize to the other four qualifiers.** Every qualifier has behavioral
consequences — the question is whether *hiding which one applies* costs anything beyond "one more
fact to look up," and for `'shared`/`'actor`/`'guard`/`'observed` it doesn't:

- `'shared`'s read-only restriction is enforced by the checker at the exact point of misuse (a `def`
  call), regardless of whether the field's qualifier was written or copied — there's nothing to
  discover in advance that the compiler won't also catch locally.
- `'actor` vs. `'guard` (mutex vs. read-write lock) is invisible at the call site either way —
  calling code looks identical regardless of which one backs it — and §6's override keeps working
  for both with no caveat.
- `'observed`'s subscription is automatic and unconditional (`boring-ui-draft.md` §3 — no `@state`
  gate to worry about at all), so it doesn't depend on whether the field's qualifier was written or
  copied either — a view mounts and subscribes the same way regardless — and §6's override keeps
  working for it too.

`'static` is the only one of the five where hiding it makes a capability this document advertises as
the general story (§6's one-line override) *silently stop working*, with nothing in the consuming
code hinting why. That's a difference in kind, not just one more qualifier flavor to memorize — it's
the only case where "bare" and "explicit" produce a real difference in what the field's own code can
still do, not just in what a reader happens to already know.

### 3. `@provide` — a function attribute, not a new declaration kind

```boring
struct RealNetworkClient as NetworkClient:
    req [byte] fetch(string url) throws:
        ...

@provide
@singleton
NetworkClient'shared networkClient():
    RealNetworkClient()
```

(Stacked with `@singleton` from the start here since this exact example is reused, under that
assumption, in "`@inject`'s relationship to the qualifier system" below — see §4 for `@singleton`
itself, the transient un-annotated default, and for why a transient provider's return type doesn't
need a qualifier at all the way this one does.)

Same reconsideration as §1, and for the same reason: an earlier version of this draft introduced
`provide <Type> <name>(): <body>` as a brand-new top-level declaration keyword — but
`NetworkClient'shared networkClient(): RealNetworkClient()` is already a completely ordinary
Boring function declaration (the ordinary return-type-first form, `docs/book.md`'s Functions
section: `int add(int a, int b): a + b`). `provide` never needed its own grammar at all — the only
genuinely new thing it added was "register this function's return type as the one DI source for
it," a marker on an already fully-formed declaration, i.e. exactly what `@attribute`s are for.
Bare `@provide` needs no arguments for the basic case — the decorated function's own name, params,
and return type already say everything a provider needs to say. It does take one *optional* named
argument, `id` (§5, for multiple bindings of one type), following the same `@name(args)` shape
`@serde(rename = "1")` already established. Memoized/shared scope is a **separate, stackable**
attribute, `@singleton` (§4) — not an argument of `@provide` at all, since it has meaning
independent of DI (§4). Either way, `@provide` registers the **one** concrete source for its return
type (and `id`, if given), visible program-wide — including across `[deps]` project boundaries,
using exactly the same visibility the compiler already resolves `use <name>.xxx` and trait impls
against (no new project-topology concept).

Top-level only, the same restriction `'static` construction already has for its "top level /
inside `main`" sites (`docs/book.md` §21) — a `@provide`-attributed function nested inside another
function's body would reintroduce exactly the "conditionally registered at runtime" flavor this
design is trying to avoid.

**`@provide` requires `pub`.** A `@provide`-attributed function that isn't also `pub` is a compile
error. A private, module-only provider has no legitimate use case — `@provide`'s entire reason to
exist is to be found by `@inject` sites that don't know its concrete declaration, which is exactly
what keeping it private would prevent. A helper that's genuinely meant to stay internal simply
shouldn't carry `@provide` at all; it can be called *by* a `pub @provide` function without needing
the attribute itself. This isn't a new visibility concept — `@provide` just participates in
whatever `pub` already means for an ordinary function (`docs/book.md` §15), with the added
constraint that it must always opt in.

### 4. Scope, corrected: `@provide` calls fresh by default, `@singleton` opts into memoizing — and stands on its own

Two earlier versions of this section both got this wrong, in opposite directions, and the
contradiction between them is worth stating plainly rather than quietly patching over: the first
claimed the provider's return-type qualifier alone already distinguished "one shared instance"
(`'shared`/`'static`) from "a fresh instance per use" (plain `'owned`) — which conflicts with §1's
own description of `@inject` as "exactly like a defaulted parameter," since nothing about a
qualifier says how often the *initializer expression* behind it runs. The second version noticed the
`'owned` exclusion but overcorrected into claiming *every* accepted qualifier is automatically
memoized behind a hidden `LazyLock`, unconditionally — which is `'static`'s own, pre-existing
behavior smuggled onto qualifiers that have never meant that anywhere else in the language (`let
c'shared = Counter()` doesn't mean "the only `Counter'shared` in the program"; it's an ordinary,
freshly-evaluated expression, same as any other). Neither is right. The correct model keeps §1's
original claim literally true:

**By default, a `@provide`-attributed function's body runs exactly like any other function call —
fresh, every single time an `@inject` site resolves against it.** No hidden caching, no new
call-resolution machinery, exactly what §1 already promised. This is genuinely **transient** scope,
and it's why `'owned` (§2) is legal here: a fresh `Box<T>` on every call is exactly what `'owned`
already means, with nothing left to reconcile.

```boring
@provide
Logger consoleLogger():
    ConsoleLogger()   # a brand-new ConsoleLogger, constructed on every resolution
```

**Who decides the final Rust representation is exactly the mirror of who decides the call
frequency.** Transient: the provider is just a value-producing expression with no representation
of its own to speak of (`consoleLogger()` above returns a bare `Logger`, no qualifier at all,
exactly like `ConsoleLogger()` written inline anywhere else would) — so the **`@inject` site**
decides, the same way any qualified binding already decides how a bare constructor call gets wrapped
(`'shared`/`'actor`/`'owned`/chapter-30-inferred, whichever the field asks for). Singleton: there is
physically one instance with one fixed representation, so the **provider** decides, and every
`@inject` site consuming it must match — there's no other consumer to disagree with, since they're
all looking at the same value.

**`@singleton` opts into the other case — as its own, separate, stackable attribute, not an
argument of `@provide`.** An earlier version of this draft made this an argument
(`@provide(singleton = true)`) before realizing that memoizing a function's result on first call and
sharing it thereafter has nothing to do with DI specifically — it's a general capability (the same
thing Swift's `lazy var`, C++'s function-local `static`, or a hand-rolled Rust `LazyLock` already
give you), and `@provide` shouldn't be the only door to it. `@singleton` is legal on **any**
function, `@provide`-attributed or not:

```boring
@singleton
def string expensiveGreeting():
    print "computing the greeting (only once, ever)"
    "hello"
```

It memoizes the function's result behind a compiler-synthesized `LazyLock`: the body runs *at most
once*, the first time the function is called from **anywhere** — an `@inject` site resolving it, or
completely ordinary code calling it directly by name — and every subsequent call, from anywhere,
gets a clone/reference to that same instance. This isn't a special DI-only codegen path: the
"Illustrative Rust equivalent" shown below (`@inject`'s relationship to the qualifier system) makes
this concrete — `fn network_client() -> Arc<dyn NetworkClient> { NETWORK_CLIENT.clone() }` reads
from the memoized `static` unconditionally, so a plain, direct call to `networkClient()` anywhere in
ordinary code returns the exact same instance an `@inject` site would have gotten. `@singleton`
genuinely doesn't need `@inject`/`@provide` to have meaning; DI is one consumer of it, not its
reason for existing.

**Lazy (first-use), never eager (at startup) — deliberately.** Beyond matching `'static`'s own
`LazyLock` behavior, this avoids constructing `@singleton` providers a given run never actually
touches, and it sidesteps provider-to-provider ordering entirely rather than merely easing it: an
eager scheme would need a real topological sort over the whole `@singleton` graph before startup
(harder still once providers span `[deps]` projects), while lazy needs no schedule at all — a
provider whose body needs another `@singleton` value just triggers that one's own on-demand
construction the moment it's touched, the same mechanism that already resolves ordinary
function-call dependencies. Genuine cycles are still caught at compile time (§7) either way; what's
given up is only startup-time discovery of a provider that panics for its own reasons (a bad env
var, a malformed config) — and nothing stops an app that wants that anyway from calling its own
providers once, on purpose, during startup, since they're just ordinary functions.

```boring
@provide
@singleton
NetworkClient'shared networkClient():
    RealNetworkClient()   # constructed once, the same instance handed out everywhere —
                           # to @inject sites and to any ordinary direct call alike
```

`@singleton` requires its function's return type to be a qualifier that supports being referenced by
more than one consumer — `'shared`/`'actor`/`'guard`/`'observed` (or, redundantly but harmlessly,
`'static`, which is already exactly-once regardless of this attribute) — never `'owned` (§2), which
is exclusive by definition and cannot back more than one holder, `@provide` or not. In practice,
`'observed` (§2) will almost always want `@singleton`: an `'observed` cell that resolved fresh per
`@inject` site would give every observing `view` its *own*, unrelated cell, defeating the entire
point of the shared-observed-model pattern this document traces back to — technically legal without
`@singleton`, essentially never what's wanted.

This resolves the previous "no transient scope at all" gap cleanly: transient dependencies (a
per-request DTO, a fresh per-screen view model) are simply `@provide`-attributed functions without
`@singleton`, using `'owned` (or a non-singleton `'shared`/`'actor`/`'guard`/`'observed`, if Arc-style
internal cheap-cloning is wanted without cross-consumer sharing) exactly as ordinary function-call
semantics already provide, no special casing needed.

This also means the earlier "Scope annotations (`@Singleton`, Dagger-style) as new syntax" rejection
(see Rejected Alternatives) was wrong on the merits, not just costly: the qualifier alone genuinely
cannot express call-frequency, so an explicit scope marker turned out to be necessary after all —
and, once it's clear that marker is really "memoize this function," general-purpose and independent
of DI, giving it its own attribute (`@singleton`) rather than burying it as one more `@provide`
argument is the more honest reflection of what it actually is, closer to Dagger's own separate
`@Singleton` annotation than the first `singleton = true` sketch was.

### 5. Multiple implementations of one interface — supertraits *and* a literal `id`

Dagger's `@Named`/qualifier annotations and Swift `Dependencies`'s "one `DependencyKey` type per
distinct binding" both solve "I need two different `Logger`s" by giving each binding its own
identity. Boring has two ways to express that identity, for two different shapes of the problem.

**When the two bindings are genuinely different concepts** — an audit trail versus debug output are
different *kinds* of logger, not two configurations of the same one — a marker sub-trait needs no
new syntax at all:

```boring
trait Logger:
    req void log(string msg)

trait AuditLogger as Logger:      # supertrait — no new members, just a distinct identity
trait DebugLogger as Logger:

@provide
@singleton
AuditLogger'shared auditLogger(): FileAuditLogger()

@provide
@singleton
DebugLogger'shared loggerForDebug(): ConsoleLogger()

struct PaymentService:
    @inject
    AuditLogger'shared audit     # unambiguous — resolves against AuditLogger's own provider
```

`@inject` resolves against the exact trait named at the field, and two distinct (super)traits are
two distinct resolution keys even though both conform to `Logger`.

**When the bindings are the same concept, just multiple instances or configurations of it** — three
database shards, one client per environment — declaring a fresh marker trait per instance is
ceremony for its own sake. For that case, `@inject`/`@provide` also take an `id` argument, matching
Dagger's actual `@Named`/Guice's `@Named` more directly:

```boring
trait Database:
    req [Row] query(string sql) throws

@provide(id = "primary")
@singleton
Database'shared primaryDb(): PostgresDatabase(host = "primary.internal")

@provide(id = "replica")
@singleton
Database'shared replicaDb(): PostgresDatabase(host = "replica.internal")

struct ReportGenerator:
    @inject(id = "replica")
    Database'shared db     # resolves against replicaDb() specifically
```

The resolution key becomes **(base type, `id`)** rather than base type alone — an `@inject` with no
`id` only ever matches a `@provide` with no `id`, and vice versa, so adding `id`s to a few variants
of a type never silently collides with an existing unnamed binding of the same type elsewhere. This
is also a partial answer to the "action at a distance" ambiguity risk raised later in this
document: two providers for the same base type are no longer automatically an error, as long as at
least one carries a distinct `id`.

**Guardrail, non-negotiable: `id` must be a compile-time string literal on both sides, never a
computed expression.** The entire reason `@inject`/`@provide` avoided a Spring-style runtime
container was that "resolve a dependency by looking up a name" is exactly the reflective,
un-auditable pattern this document rejects outright. Restricting `id` to literals keeps resolution
exactly as static as the base-type-only case — mismatches (a typo'd `id`, a missing provider for
that `id`) are still whole-program compile errors, not a runtime lookup that can fail in production.
`@inject(id = someVariable)` would reopen that door and is rejected the same way a hypothetical
`provide(env())` would be.

Picking between the two mechanisms is a judgment call left to the author, not something the checker
enforces — both compile down to the same resolution machinery (§7), just keyed differently.

### 6. Test/mock substitution — a labeled-argument override for the shallow case, `@provide(env = "...")` for the deep one

Because an `@inject`-annotated field is exactly an omittable, defaulted constructor argument,
overriding it is already expressible with Boring's existing named-argument call syntax:

```boring
test "fetchUsers hits the network client":
    let repo = UserRepository(client = MockNetworkClient(canned = [...]))
    let users = repo.fetchUsers()
    assert users.length == 1
```

No mock-container, no test-only `Module`/`Component` class, no scoped-override runtime mechanism —
the override is a plain, visible constructor argument at the exact call site the test controls.
This is the most auditable possible answer for the case it reaches: a dependency of the exact object
the test constructs. §2's `'static` paragraph adds a second, narrower limit on top: this one-line
override only works as shown for `'shared`/`'actor`/`'guard`/`'observed`-qualified fields — a
`'static`-qualified field requires the substitute value to itself be declared at a legal `'static`
site first.

**For a dependency several constructions deep — resolved by code the test never calls directly — the
labeled-argument override cannot reach it at all**, and the previous version of this document left
that gap entirely open. The fix keeps the same "fully static, no ambient state" posture as everything
else here: `@provide` takes one more optional argument, `env`, a free-form compile-time string naming
the build it applies to (`"test"`, `"debug"`, `"prod"`, `"preprod"`, or something project-specific
like `"preprod-redhat"` — Boring never parses or structures the string, it's opaque, exactly like
`id`). An `env`-tagged provider **replaces** the plain one for its (type, `id`) pair whenever the
compiler is invoked for that exact environment:

```boring
# --- production code, elsewhere ---
trait AuditLogger:
    def void log(string event)

struct FileAuditLogger as AuditLogger:
    def void log(string event): ...

@provide
@singleton
AuditLogger'shared auditLogger(): FileAuditLogger()   # the plain default — every build
                                                        # not overridden below uses this

# --- test file ---
struct MockAuditLogger as AuditLogger:
    var [string] events = []
    def void log(string event): events.push(event)

@provide(env = "test")
AuditLogger'shared testAuditLogger(): MockAuditLogger()

test "report generator logs an audit event":
    let report = ReportGenerator.create()   # PaymentService, resolved several
    ...                                       # constructions down inside create(),
                                              # gets testAuditLogger() instead —
                                              # nothing in ReportGenerator or
                                              # PaymentService changed at all
```

`"test"` here is just a convention, not a keyword — this single mechanism covers the original
test-substitution motivation and generalizes past it for free. A build-specific concern that has
nothing to do with testing composes the exact same way:

```boring
trait DiskInfo:
    req uint availableBytes(string path) throws

struct GenericDiskInfo as DiskInfo:
    req uint availableBytes(string path) throws: ...   # statfs(2) or equivalent

@provide
DiskInfo genericDiskInfo(): GenericDiskInfo()

# --- a RedHat-specific build wants to also check SELinux-labeled volumes ---
struct RedHatDiskInfo as DiskInfo:
    req uint availableBytes(string path) throws: ...   # same, plus SELinux context checks

@provide(env = "release-preprod-redhat")
DiskInfo redHatDiskInfo(): RedHatDiskInfo()
```

This is a **compile-time swap of the registry itself for that build**, not a runtime ambient
override — the same distinction §"Ambient dynamically-scoped override storage" (Rejected
Alternatives) already draws and rejects for the general case still applies here: there is no
push/pop scope stack, no `TaskLocal`-equivalent, nothing active only "during one test's execution" or
"while running on RedHat." An `env`-tagged provider either matches the one environment this exact
compilation was invoked for, or it doesn't — decided once, the same way the ordinary registry is
built (§2), never re-evaluated at runtime. Concretely:

- **Resolution gains one more filtering step, ahead of the existing (type, `id`) uniqueness check**:
  for a given (type, `id`), a provider with no `env` is always a *candidate*; one with `env` set is a
  candidate only if it matches the current build's environment exactly (a plain string comparison,
  decided once for the whole compilation, before any `@inject` site is resolved). Among the
  candidates, an `env`-matching one **outranks** a plain, env-less one for the same (type, `id`) — an
  explicit, deliberate exception to the base "no favorites" rule, the entire point of this mechanism.
  Two candidates at the *same* rank (two plain providers, or two providers matching the *same* `env`
  string) still hit the ordinary "ambiguous provider" error (§ "Ambiguity UX") — `env` narrows
  candidates and breaks ties between ranks, it doesn't relax uniqueness *within* a rank.
- **`env` is provider-side only — `@inject` never mentions it, unlike `id`.** `id` exists because
  several *simultaneously valid* bindings of one type can coexist in the same build, and the
  consumer has to say which one it wants (`@inject(id = "replica")`). `env` describes *mutually
  exclusive builds* — exactly one environment is ever active for a given compilation, so there is
  nothing for any single `@inject` site to choose between; the same field resolves to whichever
  provider is active for that build, completely transparently. This asymmetry is the whole feature:
  consuming code never changes between environments, only which provider answers it does.
- **`env`, like `id`, must be a compile-time string literal — never computed** (§5's guardrail
  applies identically, for the identical reason).
- **Composes freely with `id` and `@singleton`** — `@provide(id = "primary", env = "preprod-redhat")`,
  optionally stacked with `@singleton`, works with no extra rules: `id` and `env` are independent
  axes of the same key.
- **Scope is the whole build, not a narrower unit** — every `@inject` site (every test in a `"test"`-
  environment binary, every code path in a `"preprod-redhat"` one) sees the same substitution. A
  single test wanting a *different* double than its neighbors in the same environment still reaches
  for the labeled-argument override (§ above) on whatever it constructs directly.

**How the compiler actually learns the current `env` — resolved and shipped.** A `--env <value>`
flag, exactly as anticipated above, on both `boring build` and `boring run` (`main.rs`'s
`current_env_flag`) — read once, directly from `std::env::args()` rather than threaded through each
subcommand's own bespoke argument parser (`parse_build_command`/`parse_run_flags`/the GPU targets'
own parsing all just need to *recognize* `--env <value>` so it isn't rejected as an unknown flag;
none of them need to store or forward it themselves). Confirms the reasoning below: no dependency on
Rust/Cargo test-target machinery at all — `@provide`/`@inject` resolve entirely inside
`src/desugar_inject.rs`, before any Rust is emitted, so `env` never needs Cargo's own `#[cfg(test)]`
to exist or apply. `env = "test"` needs nothing from Cargo, only from `boring`'s own CLI.

### 7. Transitive resolution falls out for free

A `@provide`-attributed function's body is ordinary code — if the concrete type it constructs
itself has `@inject` fields, those resolve recursively at *that* construction, using the same rule:

```boring
struct RealNetworkClient as NetworkClient:
    @inject
    Logger'shared logger        # resolved when RealNetworkClient() runs, wherever that happens
    string baseUrl

    req [byte] fetch(string url) throws:
        logger.log("fetching {url}")
        ...

@provide
@singleton
NetworkClient'shared networkClient():
    RealNetworkClient(baseUrl = "https://api.example.com")
    # `logger` is not passed here — it resolves on its own, recursively
```

`networkClient()`'s own `@singleton` has no bearing on `logger`'s scope — each function's
`@singleton` is independent, so `Logger` could be resolved fresh on every construction of
`RealNetworkClient` even though `RealNetworkClient` itself, once built the first time, is then
reused forever (§4).

A cycle (the function `@provide`d for `A` needs a `B`-typed `@inject` field, the one `@provide`d
for `B` needs an `A`-typed one back) is a compile error, resolved by the same class of static,
whole-program traversal the checker's existing recursion-depth guard infrastructure already models
(`docs/book.md` §21's checker inventory) — reported as the full chain, not just "cycle detected."

## `@inject`'s relationship to the qualifier system

`@inject` decorates a field whose type is written exactly as it would be without DI at all — an
explicit `NetworkClient'shared`, or a bare unqualified `NetworkClient`. `@inject` never introduces a
new representation and never appears inside the `'`-tick chain itself; it sits entirely outside the
qualifier system, the same way `@serde(rename = ...)` sits outside it while decorating a field that
has one. What `@inject` checks against the qualifier system depends on §1's four-way split
(explicit/bare × transient/singleton): against a **singleton** provider, there's exactly one
physical representation in play, so a bare field just takes it verbatim and an explicit field is a
real **compatibility constraint** — the provider's return type must match the field's declared one
exactly, or it's a compile error naming both sides ("`client` is declared `NetworkClient'actor`, but
the only visible provider (`networkClient` at network.br:12) returns `NetworkClient'shared`").
Against a **transient** provider, there's no fixed representation to check *or* copy — the field's
own qualifier (explicit or chapter-30-inferred) simply decides how the freshly-produced value gets
wrapped at that call site, the same as it would for any other constructor call.

**Illustrative Rust equivalent**, following the `RealNetworkClient`/`UserRepository` example above,
assuming `networkClient()` is `@provide` + `@singleton`-attributed (§4):

```rust
static NETWORK_CLIENT: std::sync::LazyLock<Arc<dyn NetworkClient>> =
    std::sync::LazyLock::new(|| Arc::new(RealNetworkClient::new(/* logger resolved here */)) as Arc<dyn NetworkClient>);

fn network_client() -> Arc<dyn NetworkClient> { NETWORK_CLIENT.clone() }

struct UserRepository { client: Arc<dyn NetworkClient> }
impl UserRepository {
    pub fn new() -> Self { UserRepository { client: network_client() } }
    pub fn new_with(client: Arc<dyn NetworkClient>) -> Self { UserRepository { client } }
    pub fn fetch_users(&self) -> Result<Vec<User>, Error> { self.client.fetch("/users") }
}
```

Two constructors fall out exactly the way an ordinary defaulted parameter already generates one
call-site path for "omitted" and one for "supplied" — `@inject` is not a special case of the
codegen, it's the same mechanism with a synthesized default. `Arc<dyn NetworkClient>` here comes
entirely from `client`'s own declared type (`NetworkClient'shared`) exactly as it would without
`@inject` on the field at all — `@inject` only adds the compatibility check against
`networkClient()`'s return type and the synthesized default. Without `@singleton` (§4's default),
`fn network_client()` would have no accompanying `static`/`LazyLock` at all — just
`Arc::new(RealNetworkClient::new(...))` evaluated directly in its body, called fresh every time
`UserRepository::new()` runs, same as any other function. Note also that `fn network_client()`
above has no idea it's ever consulted by `@inject` at all — it just reads `NETWORK_CLIENT.clone()`
unconditionally, so ordinary code calling `network_client()` directly gets the exact same instance,
confirming §4's point that `@singleton` stands on its own.

A plain `NetworkClient client` field with **no** `@inject` above it never becomes DI-resolved just
because a `@provide`-attributed function returning `NetworkClient` happens to exist somewhere in
the program — that would be exactly the kind of implicit-magic-from-usage-pattern Boring's `@state`
design explicitly rejected
("any field write on a `view` refreshes it" — rejected for over-triggering on a rule a reader can't
see from the declaration). `@inject` must always be written for resolution to happen at all; only
the field's own *storage* qualifier, not the fact of injection itself, is ever left implicit (and
even then, "implicit" means copied from the provider, §1 — never independently inferred).

## Worked end-to-end example — the original motivating case

```boring
# --- project: app-lib (a shared UI/service library; ships a trait, no concrete impl) ---
trait AnalyticsService:
    req void track(string event)

view SettingsView:
    @inject
    AnalyticsService'shared analytics

    body():
        Column:
            Button("Log out").onClick ():
                analytics.track("logout_clicked")
```

```boring
# --- project: analytics-impl (a different sibling project entirely) ---
use app_lib.AnalyticsService

struct SegmentAnalyticsService as AnalyticsService:
    string apiKey

    req void track(string event):
        ...
```

```boring
# --- project: app (the composition root; boring.toml [deps] on BOTH app-lib and analytics-impl) ---
use app_lib.SettingsView
use analytics_impl.SegmentAnalyticsService, AnalyticsService

@provide
@singleton
AnalyticsService'shared analyticsService():
    SegmentAnalyticsService(apiKey = env("SEGMENT_KEY"))

def main():
    mount(SettingsView())
    # SettingsView's `analytics` field resolves against the provider above even though
    # app-lib's own boring.toml has never heard of analytics-impl.
```

`app-lib` depends on nothing but the trait. `analytics-impl` depends on nothing but the trait.
Only `app`, the composition root, depends on both and wires them together — real inversion of
control, and the exact cross-module case the original `'static'observed` proposal couldn't reach.

`AnalyticsService` above is deliberately a plain shared *behavioral* dependency (no observation, no
refresh) — the literal `@EnvironmentObject`/`@StateObject` case from `boring-ui-draft.md` is closer
to an **observed, shared model** instead, which is exactly what §2's inclusion of `'observed` is for.

**Why `'actor` alone can't stand in for it here — concretely, who needs notifying.** A bare `'actor`
field has no subscriber list at all — full stop, no attribute involved — so nothing could ever tell
a second, unrelated holder of the same value that a write happened. `@state` doesn't enter into this
either way: per `boring-ui-draft.md` §3, it no longer applies to `'observed` fields at all — a `view`
holding one subscribes automatically and unconditionally the moment it's mounted, no attribute
needed, no opt-in to remember. The flagship reason `@inject` exists at all for a shared model is that
**more than one view** ends up holding the *same* instance — if only one view ever touched it, it
would just be that view's own local `@state var mut ...` field (`boring-ui-draft.md`'s own example),
with no DI involved. So picture a second, unrelated consumer alongside `ProfileBadge`:

```boring
struct UserSession:
    var mut string username = "guest"
    def setUsername(string name): username = name

view AccountMenu:
    @inject
    UserSession'actor'observed session   # this view writes, but never displays `username` itself —
                                          # its own subscription (automatic, per §3) just goes unused

    body():
        Button("Log in as Alice").onClick ():
            session.setUsername("Alice")   # the write happens here — inside AccountMenu's own
                                            # code, not ProfileBadge's

view ProfileBadge:
    @inject
    UserSession'actor'observed session   # @inject resolves the shared cell; subscribing to it is
                                          # automatic and unconditional (boring-ui-draft.md §3) —
                                          # nothing else to write here for the refresh to happen

    body():
        Text("{session.username}")

@provide
@singleton
UserSession'actor'observed currentUserSession():
    UserSession()
```

The write happens inside `AccountMenu`'s own method, never inside `ProfileBadge`'s — `ProfileBadge`
contains no assignment to `session.username` at all, only a read. Nothing about a plain `'actor`
field could ever notice that write and relay it, from any struct, with any attribute: the
representation itself has no subscriber list to walk. Only `'observed`'s subscriber list —
walked on every write, `boring-ui-draft.md` §3 — lets `ProfileBadge` learn about a write some *other*
view made, which is exactly who needs notifying: not the writer (`AccountMenu`, which has no reason
to refresh itself over its own write), but every *other*, unrelated consumer of the same shared
instance, subscribed automatically simply by holding the field. A bare `'actor` field would still let
both views read and write the same `UserSession` correctly (the `Mutex` handles that part) — it just
wouldn't tell `ProfileBadge` when to redraw.

`@singleton` is not optional here in any practical sense (§4) — a plain, un-`@singleton`
`UserSession'actor'observed` provider would hand `AccountMenu` and `ProfileBadge` two *different*,
disconnected session cells, each starting at `"guest"`, so `AccountMenu`'s write would have nothing
to do with what `ProfileBadge` displays at all — the entire premise of the example. Neither view
constructs or names `currentUserSession`; any view anywhere in the tree that declares an
`@inject`-annotated `UserSession'actor'observed` field is, by construction, already subscribed the
moment it mounts (`boring-ui-draft.md` §3) and refreshes when the shared session changes, with no
parameter threaded through a single intermediate ancestor and nothing extra to write for it. This is
the `@EnvironmentObject` row's actual shape, not an approximation of it.

## Rejected alternatives, for the record

- **`'static'observed` top-level named singleton** (the original proposal from `boring-ui-draft.md`)
  — rejected: requires the consumer to name the concrete declaration directly, which is a
  service-locator-by-reference, not DI, and cannot cross a module boundary without the consumer
  importing the concrete type it was supposed to be decoupled from.
- **Spring-style reflective runtime container** — rejected: no runtime reflection exists in
  transpiled Rust, and building one would be a large, opaque subsystem contrary to Boring's whole
  premise.
- **`'inject` as a qualifier, composed onto the field's own ownership qualifier
  (`NetworkClient'shared'inject`)** — this document's own first draft. Reconsidered once it became
  clear `'inject` never picks a Rust representation the way every real qualifier in `docs/book.md`
  §21 does — it only ever inherited the representation of whichever qualifier it was written next
  to. A mechanism with no representation of its own doesn't belong on the qualifier axis; `@inject`
  (§1) — a field attribute decorating an already fully-typed declaration, following the existing
  `@serde(rename = ...)` per-field-attribute precedent — is the better fit and is what the rest of
  this document now describes.
- **`provide <Type> <name>(): <body>` as a new top-level declaration keyword** — this document's
  own first draft for the provider side too. Reconsidered for the identical reason as `'inject`
  above: `NetworkClient'shared networkClient(): RealNetworkClient()` is already an ordinary Boring
  function declaration (`docs/book.md`'s return-type-first function form) with nothing left for a
  new keyword to contribute except the "register this as a DI source" marker — which is exactly an
  attribute's job. `@provide` (§3) replaces it, at zero new grammar cost.
- **Scope annotations (`@Singleton`, Dagger-style) as new syntax — rejected twice, then adopted,
  landing on almost exactly Dagger's own shape after all.** First rejected as redundant with the
  qualifier system ("the provider's own `'shared`/`'actor`/`'guard`/`'observed`/`'static` return type
  already expresses this") — wrong: a qualifier picks a *representation*, never a *call frequency*,
  and nothing about `Arc<T>` says whether the expression that produced it ran once or a thousand
  times. A second pass tried to route around that by declaring every accepted qualifier
  automatically memoized — also wrong, and directly contradicted §1's own "exactly like a defaulted
  parameter" description (§4 now documents both false starts explicitly). A third pass adopted a
  minimal fix, `singleton = true` as one argument on `@provide(...)` — reconsidered once more (§4)
  once it became clear memoizing a function's result has nothing to do with DI specifically (calling
  the function directly, with no `@inject` involved, should still return the memoized value) and
  deserves to work on any function, not just `@provide`-attributed ones. Landed on `@singleton`, its
  own separate, stackable attribute — which is, in the end, essentially Dagger's own `@Singleton`
  after all, just spelled lowercase to match Boring's other attributes and without `@Component`'s
  surrounding ceremony.
- **A "named binding" qualifier/attribute for multiple implementations, rejected in favor of
  supertraits alone** — this document's own earlier position. Reconsidered: supertraits cover the
  case where two bindings are genuinely different concepts, but forcing a fresh marker trait onto
  "three configurations of the same database client" is ceremony with no payoff. `id` (§5) is now
  adopted alongside supertraits, restricted to compile-time string literals on both sides
  specifically so it doesn't reintroduce the runtime-name-lookup pattern Spring was rejected for.
- **Ambient dynamically-scoped override storage (Swift `Dependencies`'s `TaskLocal` model), as the
  default mechanism** — not adopted as the default: it's a genuine runtime global (thread/task-local
  push-pop stack), which the plain labeled-argument override avoids entirely for every case that
  override can reach. Not ruled out forever — see Open Questions for the gap it would actually
  close.

## Open questions

- ~~`boring run` parity for `@inject`/`@provide`?~~ **Resolved, and shipped ahead of the original
  phased-rollout plan.** The original plan below (`boring build` in v1, `boring run` in v2) assumed
  `@inject` would need its own, separate interpreter-side resolution logic. It didn't: the actual v1
  implementation (`src/desugar_inject.rs`) resolves `@inject` entirely as an AST-level desugaring
  pass, synthesizing an ordinary `init` before *either* backend ever runs — so `boring run`'s
  tree-walking interpreter already gets `@inject`/`@provide` for free, with zero interpreter-specific
  code, simply by being wired into the same pipeline stage (`main.rs`'s `run_file`, right after
  `desugar_array_block`). The one piece that genuinely needed separate interpreter work was
  `@singleton`'s own memoization — the transpiler's `LazyLock`-based codegen has no interpreter
  equivalent, so a first `boring run` of a `@singleton` function silently re-ran its body on every
  call. Fixed with a small name-keyed cache on `Interpreter` (`singleton_cache`, checked/populated in
  `call_fn`) — sound because the checker already guarantees an `@singleton` function is zero-parameter
  with exactly one declaration per name. Confirmed via the same `inject_di.br` scenario running
  correctly under both `boring build` and `boring run`.

  <details><summary>Original phased-rollout reasoning (superseded above, kept for the record)</summary>

  `boring build` needs the collection pass described in §2 to complete —
  now bounded to the current project's own files, not the whole `[deps]` graph — before it can fix a
  bare `@inject` field's Rust layout. `boring run`'s tree-walking interpreter has no equivalent
  pre-pass requirement at all: it can resolve a bare (or explicit) `@inject` site lazily, at the
  moment a struct is actually constructed during execution, exactly like it already evaluates any
  other default expression. Whenever v2's `boring run` support lands, the same-project scope
  restriction (§2) must bind identically to `boring build`'s — a language-level rule about what
  `@inject` *means*, not a backend-specific detail — matching Boring's own precedent from the
  checker-portability work (`checker-portability-draft.md`): a rule's *semantic* nature, not which
  pass or backend implements it, is what should govern whether it's uniform. *How* each backend
  arrives at the answer (upfront collection for `build`, lazy on-construction lookup for `run`) is
  exactly the kind of backend-specific implementation detail that's fine to differ.

  </details>
- ~~Self-hosted-in-Boring interpreter support?~~ **Resolved: v3, well after both of the above.** The
  interpreter written in Boring itself (`boring/interpreter/*.br` — `CLAUDE.md`) is not a variant of
  `boring run` — it's a Boring *program* (`lexer.br`/`parser_core.br`/`ast.br`/`exec.br`/`eval.br`/
  `methods.br`/`stdlib.br`/`main.br`) that must itself be transpiled and `cargo build`'t into a
  standalone binary before it can interpret *other* `.br` source at all; that binary then has **no
  access whatsoever** to the native `boring` toolchain's checker or transpiler once built. Supporting
  `@inject`/`@provide` there is a genuinely separate, third implementation of the mechanism — written
  in Boring itself, against whatever AST/value model `ast.br`/`eval.br` already define — not
  something that falls out of whatever `boring run`'s native interpreter (v2) ends up doing.
  Concretely, this self-hosted interpreter would need to reimplement, in Boring: (a) recognizing
  `@inject`/`@provide` at parse time (itself contingent on whether its parser already handles the
  general `@attribute` grammar at all — unverified); (b) its own resolution/registry logic, mirroring
  whichever strategy `boring run` adopts; and (c) some story for the validation rules from §2/§5
  (exactly-one-provider, the accepted-qualifier set, `id`-must-be-a-literal) — though, being an
  interpreter rather than a static compiler, it has the option of deferring these to a runtime error
  at the point of construction instead of replicating full static analysis, a simpler (if less
  strict) strategy already somewhat in the spirit of how interpreters typically trade upfront
  guarantees for implementation simplicity. Cross-project (`[deps]`) resolution is very likely out of
  reach here regardless of any DI-specific design choice, since it would additionally require
  re-implementing `[deps]` file-system/git/lockfile resolution in Boring itself, a much larger,
  unrelated lift — so the same-project-only scope (§2) isn't just sufficient for this case, it's
  plausibly the only case reachable at all even by v3. Worth
  tracking, not worth blocking on, and only relevant once `tests/cases/*.br` (the self-hosted
  interpreter's own validation suite, per `CLAUDE.md`) has an actual need to exercise a DI-using
  program.
- ~~No transient scope at all.~~ **Resolved (§4)**: transient is now the *default* — a
  `@provide`-attributed function runs fresh on every resolution unless also marked `@singleton` —
  matching real DI frameworks' own default posture (Guice/Dagger also default to unscoped/transient)
  rather than inverting it as an earlier version of this draft did. Kept here, struck through, as a
  record that this was genuinely unresolved for a while, not silently always fine.
- ~~Deeply-nested test substitution?~~ **Resolved and shipped (§6): `@provide(env = "...")`**, a
  compile-time registry swap keyed by a free-form environment string, never a runtime ambient scope —
  landing on the "something more static" option this bullet used to leave open rather than the
  ambient-dynamic-scope one, and generalizing past testing specifically (`env = "test"` is just one
  conventional value) to cover build/target-specific providers too (`env = "preprod-redhat"`). The
  "does a distinct test-build target exist upstream" question this used to raise is now moot: `env`
  is read once by Boring's own CLI (`--env <value>` on both `boring build` and `boring run`,
  `main.rs`'s `current_env_flag`), entirely independent of Cargo/Rust build profiles, so it needs no
  `#[cfg(test)]`-equivalent hook from anything downstream. Implemented in `src/desugar_inject.rs`
  (`resolve_provider`: an env-matching candidate outranks a plain one for the same `(base type, id)`
  key; falls back to the plain one when nothing matches), tested in `tests/dependency_injection.rs`.
- ~~Ambiguity UX?~~ **Mostly resolved.** First, a distinction worth making explicit since "lazy" now
  means two unrelated things in this document: `@singleton`'s laziness (§4) is a **runtime** property
  of the *generated code* (when a provider's body actually executes, once the compiled program is
  running); ambiguity detection is entirely different and always a **compile-time** (transpilation)
  check, over the `@provide` *declarations themselves* — it has nothing to do with when, or whether,
  any provider is ever constructed at runtime, and nothing about it is deferred to program execution.

  The check itself is the natural byproduct of building the registry (§2) at all: a single,
  deterministic-order scan over `@provide` declarations, inserting each into a `(base type, id) →
  declaration` map (env-tagged ones layered per §6's priority rule), erroring the moment a second
  declaration would collide with an already-inserted key at the same priority tier — not a search
  triggered per `@inject` site, just a property of the declarations that a single linear pass already
  surfaces. Concretely, per the scope split §2 already establishes:
  - **Same-project `@provide` declarations are always scanned exhaustively.** The compiler already
    parses the whole project for ordinary compilation, so checking its own providers for collisions
    costs nothing extra — caught regardless of whether any `@inject` site in the project actually
    consumes that type.
  - **Cross-project (`[deps]`) `@provide` declarations are only pulled into the check for a
    (type, id, env) key that some *explicit* `@inject` site in the current project actually asks for**
    — not exhaustively re-validated as a whole. A dependency project's own internal ambiguity, for
    types the consuming project never touches, is that dependency's own problem, already caught when
    *it* was compiled standalone under the same same-project-exhaustive rule. This avoids needing to
    fully parse/validate every transitively-included `[deps]` project's entire provider set just to
    compile a project that only actually uses a handful of their types.
  - **Diagnostic anchoring** (your proposal, adopted): the primary error points at the **second**
    declaration the scan encounters for a colliding key — e.g. "ambiguous provider for `NetworkClient`
    (no `id`)" at `analytics-impl/client.br:9` — with a `note:` pointing back at the first one
    (`app-lib/client.br:12`), and, when the check was triggered by cross-project demand rather than
    same-project exhaustive scanning, a further `note:` at the `@inject` site that pulled the
    second project's provider into consideration in the first place, since that's usually what a
    developer is actually looking at when the build fails. One error per colliding key, not one per
    consuming `@inject` site, once the key is known ambiguous.
  Still open: exact message wording, and whether to list every consuming `@inject` site or just the
  one that triggered the check — polish, not a design gap. (This same-project-exhaustive /
  cross-project-on-demand split is also the mechanical basis for the distance-based priority
  described under "Resolution is keyed by..." below — that entry adds *ranking* between tiers on top
  of the scanning strategy described here.)
- **Per-subtree ambient overrides.** Still explicitly out of scope for this document, and still
  `boring-ui`'s own follow-up design if it needs "a different value for one branch of the view tree"
  (SwiftUI `.environment(\.x, y)` / Compose `CompositionLocalProvider`) — but with one concrete
  constraint worth recording now rather than rediscovering later. The natural shape of a solution is
  a `@provide` function that reads an *ambient, implicitly-supplied context* (not a normal, explicit
  argument written at the call site) to decide what to return — different context, potentially
  different answer. That pins down one hard rule immediately: **such a provider can never be
  `@singleton`** — the same underlying reason `'owned` + `@singleton` is already rejected (§2):
  `@singleton` promises exactly one shared, invariant answer, and a result that legitimately varies
  by caller breaks that promise whether the variance comes from exclusive ownership or from ambient
  context. This isn't a new kind of provider, either — a context-consulting provider is simply an
  ordinary **transient** one (§4, called fresh on every resolution) that happens to also read one
  more piece of implicit information from its environment while it runs, no different in kind from a
  transient provider reading an env var or a config file today. What this document cannot resolve on
  its own is *what "the current context" actually is and how it gets there invisibly* — that requires
  `boring-ui`'s own rendering/diffing engine to thread an ambient value down through `body()` calls as
  it walks the view tree (conceptually what SwiftUI's engine does internally with `EnvironmentValues`)
  before there's anything for a provider to even read. Without that engine, "an invisible parameter"
  has nothing to be invisibly supplied *from*. Left for `boring-ui` to design: the context's type, how
  it's threaded, and the concrete syntax for a provider to declare it wants one.
- ~~Generic dependencies?~~ **Mostly resolved by checking real precedent.** Dagger, Guice, and Swift
  `Dependencies` all key bindings by *concrete* instantiation only — `Repository<User>` and
  `Repository<Order>` are simply two unrelated binding keys, never one generic binding (Guice even
  has a dedicated `TypeLiteral<T>` mechanism purely to work around Java's type erasure and let people
  write one key per concrete instantiation). That case already works in this design with **zero new
  machinery**: `Repository<User>'shared`/`Repository<Order>'shared` are just two ordinary, unrelated
  concrete types as far as `@inject`/`@provide` are concerned (each can carry its own `id` if
  disambiguating several instantiations of the same generic base is ever needed) — no different from
  any other two distinct types. The one real precedent for a **single, genuinely generic** provider
  (one implementation, valid for any `T`) is .NET's DI container's open-generic registration
  (`services.AddScoped(typeof(IRepository<>), typeof(Repository<>))`), which works by closing the
  generic *at resolution time, via reflection* — not available here. Boring has a cleaner path to the
  same feature if it's ever wanted, though: since Rust generics are monomorphized at *compile* time,
  a genuinely generic `@provide` (`@provide Repository<T>'shared repoFor<T>(): InMemoryRepository<T>()`)
  could in principle be specialized per concrete `T` by the compiler itself, statically, for whichever
  `@inject` site demands it — no reflection needed, unlike .NET's version. This only helps the
  "one uniform implementation for any `T`" case, not "different implementation per `T`" (which the
  already-working concrete-instantiation path already covers) — worth designing properly if a real
  use case turns up, not needed by anything in this document today.
- ~~Eager vs. lazy construction for `@singleton`?~~ **Resolved: lazy**, for two independent reasons,
  not just consistency with `'static`'s own `LazyLock`:
  - **Avoids constructing providers nothing actually uses.** A real app's composition root can
    easily declare more `@singleton` providers than any single run path touches (different features,
    different environments) — eager construction would pay for all of them unconditionally, on every
    startup, defeating one of the practical reasons to reach for a provider in the first place.
  - **Sidesteps provider-to-provider ordering entirely, not just "mostly."** Eager construction would
    require computing a topological order over the whole `@singleton` dependency graph before
    startup — genuinely nontrivial once providers span `[deps]` projects — and get it wrong exactly
    when a provider indirectly depends on one declared to construct after it. Lazy needs no such
    schedule: each `@singleton` behaves like its own independent `LazyLock` (exactly as `'static`
    already does), so a provider whose body needs another `@singleton` value simply triggers that
    other one's on-demand construction the moment it's touched — precisely the same mechanism that
    already resolves ordinary function-call dependencies, with no separate ordering pass to design
    or verify.

  The real cost being accepted: a provider that panics on construction is only discovered on first
  use, not at process startup — genuine **cycles** are still caught at compile time regardless (§7),
  so this only affects failures from a provider's own logic (a malformed config value, a missing env
  var), not graph-shape errors. Nothing stops an app that wants fail-fast startup validation from
  getting it anyway, with zero new machinery: since a `@singleton` function is an ordinary,
  directly-callable function (§4), a startup routine can simply call every provider it cares about
  once, on purpose, before serving traffic — `@singleton` doesn't need its own eager mode for this to
  be available.
- ~~`--target kernel` interaction?~~ **Resolved: explicit non-goal, confirmed.** `--target kernel`
  compiles to a `no_std` Rust-for-Linux module — too low-level for `@inject`/`@provide` to make sense
  there at all, the same way `boring-ui` itself excludes `--target kernel` for the identical reason:
  dynamic dispatch (`Box<dyn Trait>`/`Arc<dyn Trait>`) and a composition-root-style `@provide` graph
  both assume a full std runtime this target never has. The validator should reject `@inject`/
  `@provide` outright under this target (§ "Before implementation begins" already lists this as a
  cheap, easy-to-forget check).
- ~~Does resolution visibility need its own rule?~~ **Resolved: `@provide` requires `pub`, no
  conditional visibility at all.** A private, module-scoped `@provide` has no legitimate use case —
  the entire point of `@provide` is to be found by `@inject` sites that don't know its concrete
  declaration, which is exactly what keeping it private would prevent; the composition-root use case
  this whole document is built around requires the provider to be visible outside its own module by
  construction. A helper function that's genuinely meant to stay internal simply shouldn't carry
  `@provide` at all — it can be called *by* a `pub @provide` function without needing the attribute
  itself. So: the checker rejects a `@provide`-attributed function that isn't also `pub`, full stop.
  This isn't a new visibility *concept* — `@provide` just participates in whatever `pub` already
  means for an ordinary function (`docs/book.md` §15), with the added constraint that it must always
  opt in, since resolution across module/`[deps]` boundaries is `@provide`'s whole reason to exist.
  (One narrow residual case, not chased further here: a project that's also consumed as a `[deps]`
  library elsewhere might one day want "visible to this project's own `@inject` sites, but not
  re-exported to consumers of this project as a library" — a `pub(crate)`-shaped distinction Boring's
  `pub` may or may not already draw. Left alone until it's a real, not hypothetical, need.)
- ~~Does `@inject` need to reach bare function parameters?~~ **Resolved: no, scope stays
  fields/`init` only** — checked against real precedent, not just the three cited in §1. Dagger,
  Guice, Spring, and Angular are overwhelmingly constructor injection; Guice/Spring's "setter
  injection" *does* inject into a method's parameters, but only a method the container itself calls
  once during construction, never an arbitrary free function ordinary code calls; ASP.NET Core's
  `[FromServices]` controller-action parameters are the same shape — framework-invoked entry point,
  not a free function. The one real precedent for injecting into an arbitrary free function's
  parameters is pytest fixtures / JUnit 5's `ParameterResolver` — genuinely popular, but built on
  runtime reflection (inspecting the function's parameter names at the moment it's collected/called),
  exactly the mechanism this whole design has rejected for Boring from the start. `interpreter.br`'s
  `eval` (`scratch-boring`, the case that surfaced this gap) doesn't resemble either shape — it's not
  a framework-invoked lifecycle method, and it's not a test function — so there's no existing pattern
  to borrow from even if this were pursued. Fields/`init` remains the whole scope.
- ~~Action at a distance from global (type, `id`) keying?~~ **Substantially mitigated: resolution
  becomes distance-prioritized, not flat.** Instead of "every visible provider is an equally-valid
  candidate, ambiguous the instant two exist," candidates are ranked by how far they are from the
  `@inject` site — **same project** outranks **any `[deps]` project**, and (if worth the extra
  granularity once real `[deps]` graphs get deep) a **direct** `[deps]` dependency outranks a
  **transitive** one. The search takes the first non-empty tier and stops there — a provider at a
  farther tier is never even considered once a closer one exists, so it can't collide with it,
  silently or otherwise. **Ambiguity is still a real, reported error, but only *within* a tier**: two
  providers in the *same* project, or two in `[deps]` projects at the *same* distance, still hit the
  ordinary "ambiguous provider" error (§ "Ambiguity UX") — narrowed, not eliminated, exactly as
  intended. This directly shrinks the original `eval` scenario: an unrelated `[deps]` project adding
  its own `Clock`-returning function no longer breaks anything as long as `eval`'s own project already
  provides one — the far-away addition is simply never in contention. Two developers independently
  adding a second `Clock` provider *within the same project* remains a real, reportable conflict, and
  should — that blast radius is small and locally discoverable (same project, presumably same review
  process), unlike the original whole-program-flat design.

  Layering with the two other keys already in place: **`env` filtering (§6) happens first** — a
  provider whose `env` doesn't match the current build is removed from consideration entirely, before
  distance is even considered; **among what's left, `env`-specificity outranks distance** (an
  `env`-matching provider in `[deps]` still beats a plain, env-less one in the same project) — the
  build-identity question ("is this even a candidate for this compilation") is answered before the
  locality question ("which candidate is closest"). `id` remains an exact-match filter throughout,
  unaffected by any of this — two providers with *different* `id`s were never in contention in the
  first place.

  **This does not help item 3 (per-subtree ambient overrides), and it's worth being explicit about
  why not**: distance-based priority is still a purely *compile-time* decision, resolved once for the
  whole build. Item 3's actual problem is a *runtime* one — two different values needed
  *simultaneously*, in different branches of one running program (a light-mode preview and a
  dark-mode preview on screen at once) — which no search-path refinement, however precise, can
  express: there is still only ever one winning `@provide` per key, for the entire compiled binary.
  The two problems don't share a fix, which is exactly the point the "necessary reframing" section
  made at the very start of this document.

  Left open: exactly how many distance tiers are worth distinguishing (same-project vs. all of
  `[deps]` flat is the minimum useful version; direct vs. transitive `[deps]` is a plausible later
  refinement, not required to ship the core idea) — and the ambiguity error's diagnostic quality (§
  "Ambiguity UX") should name which tier the conflict occurred at, not just the two colliding sites.

## Before implementation begins

This draft has more open questions than most `docs/design-notes/*.md` entries at this stage, because
most of them surfaced by testing the design against real code (`scratch-boring`'s `eval`) and real
follow-up questions rather than being invented up front. Not all of them block starting — this is a
rough priority order.

**Actually blocking — settle before writing checker/transpiler code:**

1. ~~Is "singleton only" an acceptable permanent scope?~~ **Resolved and shipped (§4)**: transient
   is the default, `@singleton` (a separate, stackable attribute — not a `@provide` argument, §4)
   opts into memoizing — `'owned` + `@singleton` is enforced as a compile error
   (`check_di_provider_attrs`, `src/checker/mod.rs`; `check_singleton_owned_return`,
   `src/checker/rust_checks.rs`), tested in `tests/dependency_injection.rs`.
2. ~~`'static` under `@provide` needs a fourth legal construction site added to `docs/book.md` §21
   itself~~ **Turned out to be unnecessary — resolved and shipped without it.** The anticipated
   amendment assumed an existing site-authorization list that would need extending; investigating the
   actual checker while implementing found no such list actually gates a function's own return-type
   provenance or a defaulted-parameter's default-value provenance today
   (`check_static_provenance`/`check_static_arg_provenance` only cover a `let` statement's initializer
   and a call argument, respectively) — so there was nothing to extend for either a `@provide`
   function's tail expression or `desugar_inject`'s own synthesized default. `'static` is now in
   `@inject`'s/`@provide`'s accepted set (`desugar_inject.rs`/`check_field_qualifier_accepted`),
   including the "must be written explicitly, never bare-copied" carve-out (§2). **Caveat**: a full
   end-to-end example is still blocked by two separate, pre-existing, unrelated transpiler gaps found
   and filed while testing this (task_ce5a4ff9) — see the Status line at the top of this document.
3. ~~Cross-project visibility default for `@provide`?~~ **Resolved and shipped (§3)**: `@provide`
   requires `pub`, unconditionally — the checker rejects a non-`pub` `@provide`
   (`check_di_provider_attrs`), tested in `tests/dependency_injection.rs`.
4. ~~`boring run` vs. `boring build` parity?~~ **Resolved and shipped, both backends, ahead of the
   original phased plan** — see the fuller writeup under Open Questions. `@inject`/`@provide` need no
   interpreter-specific code at all (`desugar_inject.rs` resolves them at the AST level before either
   backend runs); `@singleton` needed one small addition, a name-keyed memoization cache on
   `Interpreter` (`singleton_cache`, `src/interpreter/call.rs`'s `call_fn`), now shipped too. The
   self-hosted-in-Boring interpreter (`boring/interpreter/*.br`) remains v3, unaffected by this.
5. ~~Eager vs. lazy construction for `@singleton`?~~ **Resolved and shipped (§4): lazy** —
   `emit_singleton_fn` (`src/transpiler/emit_top.rs`) compiles to a `std::sync::LazyLock`, each
   `@singleton` its own independent static. Non-`@singleton` providers don't need this decision at
   all — they're just ordinary function calls.
6. ~~Explicitly reject `@inject`/`@provide` under `--target kernel`~~ **Resolved and shipped** —
   `src/validator/kernel.rs` rejects all three attributes, tested in `tests/dependency_injection.rs`.

**Real new implementation work to scope, not just "reuse existing infrastructure":**

- ~~Cycle detection over the `@provide`/`@inject` graph~~ **Resolved and shipped (§7)**:
  `desugar_inject.rs`'s `detect_cycles`, a DFS with an explicit path stack over a graph of provider
  *functions* (an edge `P -> Q` means "the struct `P` constructs has an `@inject` field that resolves
  to `Q`"), reporting the first back-edge found as the full chain (`provideA -> provideB ->
  provideA`), not just "cycle detected". **One real, accepted limitation**: an edge only exists where
  a provider's body is simple enough for `provider_target_struct` to see through — a bare tail
  expression or `return` that's itself a direct constructor call (every worked example in this
  document is exactly this shape). A genuine cycle hidden behind a provider whose body does anything
  more elaborate (an `if`/`match`, an intermediate variable, a call to another function that itself
  constructs the struct) silently isn't caught — sound-by-omission, never a false positive, same
  posture as this pass's other best-effort checks. Tested in `tests/dependency_injection.rs`
  (a two-provider cycle, a one-provider self-cycle, and a non-cyclic transitive chain that must still
  compile — §7's "falls out for free" claim, re-verified alongside the cycle checks).
- **The whole-program (now same-project-only, §2) collection pass** that must complete before a
  struct with a bare `@inject` field can have its Rust layout finalized — **resolved and shipped for
  the `@singleton` case**: `desugar_inject.rs`'s two-pass structure (`collect_providers` builds the
  whole registry, *then* `desugar_items` processes every struct) already satisfies this ordering by
  construction, so a bare field matched against a `@singleton` provider copies its return type
  verbatim with no special handling needed at all. The bare-field-against-a-*transient*-provider case
  turned out to have a different, unrelated blocker instead (not the ordering problem this bullet
  worried about) — see §2's own updated text and `synthesize_init`'s doc comment: a real `boring
  build`-specific gap in how a defaulted `init` parameter's call-site value gets wrapped when its
  representation is decided later, by chapter 30 inference, than when the default is rendered.
  **Partially resolved**: fixed for the one sub-case where there's actually nothing for chapter 30 to
  decide — a bare field whose base type is a *trait*, matched against a provider that returns that
  trait bare too, since a trait object's representation is a fixed `Box<dyn Trait>` rule (unsized,
  no other option), not an inference outcome (`struct_init_defaults`, `src/transpiler/mod.rs`, now
  renders that default through the qualifier-aware `emit_let_value` the same way an explicit
  `'owned`/`'new` param's default always did). The general case — a plain struct/generic base type, or
  a trait base type whose only visible provider returns it already qualified (`'shared`/`'owned`/etc.)
  — is unchanged and still rejected explicitly: `boring run`-vs-`boring build` parity kept intact
  rather than shipping a combination that only works on one backend.
- **Ambiguity and unresolved-provider diagnostics** — **basic version shipped**:
  `desugar_inject.rs`'s `collect_providers` scans the whole `Program` once, keyed by `(base type,
  id)` (§5) with `env`-filtering (§6) applied at resolution time (`resolve_provider`) rather than at
  collection time — erroring at the second declaration sharing both key *and* `env` value (an
  unconditional collision, independent of which `env` a given build ends up using), and naming the
  type/id/env clearly when nothing matches at resolution. What's genuinely still missing, per §
  "Ambiguity UX"'s fuller design: a `note:` pointing back at the *first* declaration as a separate
  structured diagnostic (folded into one message for now, since this pass reuses `ParseError`, which
  has no note/multi-span shape), and the distance-based priority ranking (§ "Resolution is keyed
  by..." — no cross-project `[deps]` resolution exists yet for it to rank against).
- ~~`@provide(env = "...")`'s CLI shape (§6)~~ **Resolved and shipped**: a `--env <value>` flag on
  both `boring build` and `boring run` (`main.rs`'s `current_env_flag`, read directly from
  `std::env::args()` rather than threaded through each subcommand's own argument parser). No
  dependency on Cargo/Rust build profiles either way — `env` is read once by Boring's own CLI, before
  any `@provide`/`@inject` resolution begins.

**Deliberately deferrable — document as "not in v1," don't design now:**

- `@inject` on bare function parameters (§1's "Open gap") — scope to fields/`init` only, matching all
  three real-world precedents; reconsider only if a real need turns up.
- A genuinely generic `@provide<T>` (one uniform implementation for any `T`) — per-concrete-type
  generics already work today with no new work (§ "Generic dependencies?"); this is only the
  .NET-style open-generic extension, and no worked example needs it yet.
- The self-hosted-in-Boring interpreter (`boring/interpreter/*.br`) — a third, independent
  implementation surface with its own prerequisites; not relevant until its own test suite needs it.
- Per-subtree ambient overrides — explicitly `boring-ui`'s own follow-up design, not this document's;
  the one constraint pinned down here is that any such provider is necessarily transient, never
  `@singleton` (§ "Per-subtree ambient overrides").
- **`@inject` combined with a struct's own hand-written `init`** — the v1 implementation
  (`desugar_inject.rs`) rejects this combination outright ("already declares its own `init`") rather
  than merging a synthesized parameter into a user-written one; every worked example in this document
  has no custom `init` at all, so this cost nothing yet. Revisit once a real case needs both.
- The `id`-based "action at a distance" risk (§ "Resolution is keyed by...") — accepted for now;
  revisit the diagnostic and default scope if it causes real pain.

**Pre-existing transpiler bugs found while validating this implementation, unrelated to this
design and filed separately rather than fixed here** (none of them are `@inject`/`@provide`/
`@singleton`-specific — each reproduces with plain, hand-written Boring source):
- A function whose return type carries `'actor`/`'guard`/`'shared` never wrapped a bare
  constructor-call return value in the qualifier's Rust representation — **fixed** on `main` during
  this work (found via `@provide`'s own worked examples, which use exactly this shape).
- Bare (implicit-self) field access skips lock dispatch for an `'actor`/`'guard`/`'observed`-qualified
  struct field, and even bypasses the mut-permission checker — explicit `self.field` already works
  correctly for the identical code. Blocks the natural, bare-field spelling of every worked example
  in this document that mutates or reads a shared model from inside its own struct's methods (all of
  them currently need to spell out `self.` explicitly as a workaround). Filed, not fixed here.
- An explicit `init` parameter typed `Trait'owned` emits a doubly-boxed Rust type
  (`Box<Box<dyn Trait>>`) and doesn't wrap a bare-typed default expression in `Box::new(...)` —
  blocks `@inject`'s `'owned` + transient-provider case specifically (§2's "natural qualifier for a
  genuinely transient dependency"). Filed, not fixed here.

## Where this came from

Split out of a design conversation started while drafting `boring-ui`
([`boring-ui-draft.md`](boring-ui-draft.md)'s `@EnvironmentObject` row, 2026-09-16) once it became
clear the actual gap — injecting a dependency across a module boundary without a direct reference to
its concrete declaration — has nothing to do with views or rendering. Tracked there as
`task_8a93d818`; this document is that task's output.
