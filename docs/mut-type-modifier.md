# `mut` as a type modifier: `mut Type` and `mut Type&`

> **Status: Draft — not implemented.** No grammar, checker, interpreter, or
> transpiler changes exist yet for anything described here. This document
> records the design as worked out in discussion, as a basis for
> implementation. See [binding-mutability.md](binding-mutability.md) for the
> current (shipped) `let`/`mut`/`var` model this proposal extends.

## Problem Statement

Today, `let`/`mut`/`var` are three mutually exclusive keywords occupying a
single grammar slot on `let_stmt` and `param` — plus two narrower,
lower-fidelity siblings, `field_decl` (struct fields: `var` or nothing, a
single `bool`, "let"/"mut" not accepted at all) and `kernel_field_decl`
(`let`/`mut`/`var`, but `mut`/`var` are pure synonyms there — see
["Kernel structs"](#kernel-structs-a-distinct-related-model-one-axis-not-two)):

```bnf
param             ::= ("var" | "mut")? type IDENT ...
let_stmt          ::= ("let" | "mut" | "var") type? IDENT "'"? ("=" expr)? NEWLINE
                    | ("let" | "mut" | "var") destructure "=" expr NEWLINE
field_decl        ::= "var"? type IDENT ("=" expr)?             # today: 2 states, no "mut"/"let"
kernel_field_decl ::= ("let" | "mut" | "var") type IDENT ("=" expr)?  # today: 3 keywords, 2 states
```

`mut` never appears inside the `type` production itself. This conflates two
independent axes — matching Rust's own split, where `let mut x` (binding
rebindability) and `&mut T` (a distinct *type*) are unrelated mechanisms:

- **Rebindable** — can the binding be pointed at a different instance?
  (`let`/`var`)
- **Mutable** — can the referent be mutated through this binding?
  (`mut`, today entangled with the rebindability keyword)

Two concrete problems follow from `mut` living only at the binding-keyword
position:

1. **References can't nest.** `&T`/`&mut T` are genuinely distinct Rust
   types, so `mut` on a reference is meaningful anywhere a type can appear —
   but today it can't be written inside a tuple or a generic argument list,
   only at the top of a `let_stmt`/`param`. This blocks patterns like a
   Bevy-ECS-style query tuple: `Query<(Position&, mut Velocity&)>` has no
   Boring spelling today.
2. **Destructuring can't mix.** `let (a, b) = t` / `mut (a, b) = t` apply one
   keyword uniformly to every extracted variable — there's no way to say "`a`
   stays `let`, `b` becomes `mut`" in one statement.

Separately, this session's work on `'actor`/`'guard` established that Boring
already polices mutation **beyond what Rust's own type system requires**:
`let T'actor x` now rejects `def` calls even though the underlying
`Arc<Mutex<T>>` would technically allow it through the lock — matching the
existing precedent of `let`/`var` struct fields (Rust has no field-level
`const`; Boring enforces it anyway). That precedent is what makes `mut Type`
(no `&`) viable as a **Boring-only** permission, with no Rust type behind it,
rather than a fiction with no implementation target.

**Pre-existing doc conflict this proposal resolves:** [binding-mutability.md](binding-mutability.md)
already documents `mut` as `Rebindable: no` unconditionally (line 15) — but
`CLAUDE.md`'s quick reference says `mut` on a primitive is "equivalent to
`var`" (rebindable). The two contradict each other today. The model below
matches `binding-mutability.md` and retires the scalar special case in
`CLAUDE.md`.

## Proposed Design

### 1. The bare `mut` keyword is sugar for `let mut Type` — always, no exceptions

```boring
mut Type a = expr     # ≡  let mut Type a = expr — no other reading
var Type a = expr     # rebindable, not mutable — never implies mut
var mut Type a = expr # the only way to get both: rebindable AND mutable
```

`mut` alone means `let mut Type`, full stop — the binding is fixed
(non-rebindable), and `mut` attaches to the **type**. This is deliberately
the *only* thing the bare keyword can mean, with no context-dependent
collapsing and no degradation mechanism: `mut` never implies `var`, in a
`let_stmt` or anywhere else. This mirrors, exactly, what `mut` already means
on a function parameter (`let` is implicit there too — see
["Parameters"](#parameters-a-distinct-related-model-let-stays-implicit-under-mut)
below) — the same keyword now means the same thing everywhere it appears,
with no local-binding-only special case to keep track of.

**An earlier revision of this document proposed `mut Type a ≡ var mut Type
a` instead — a "give me the most permissive combination" reading, retired
here.** The trade being made was: recover the historical "`mut` ≡ `var` for
scalars" shortcut as a corollary of a general degradation rule, in exchange
for `mut` meaning two *different* things depending on context (`var mut` on
a local binding, `let mut` on a parameter — the exact same token sequence,
opposite rebindability). That trade isn't worth it:

- It reintroduces, at the language level, precisely the confusion this
  whole proposal exists to remove — a keyword whose meaning depends on
  *where* it's written, not what it says.
- It's a **silent** relaxation, not a compiler-flagged one. `var` no longer
  auto-implying `mut` (below) is a safe breaking change — every affected
  call site becomes a compile error, an exhaustive, self-auditing worklist.
  `mut` auto-implying `var` is the opposite kind of change: code that relied
  on `mut c = Counter(); ...; c = Counter2()` correctly *failing to compile*
  (a real, load-bearing guarantee — that's the whole reason to write `mut`
  instead of `var` in the first place) would silently start compiling
  instead, with no error anywhere to point at it.

Retiring the historical scalar shortcut is the accepted cost instead: `mut
int x = 0` is a checker error now (see the table below), not a silent
downgrade to `var int x = 0`. This is a narrow, **compiler-flagged** breaking
change — `mut x = 42; x = 43` simply goes back to being the error it already
is in the current, shipped semantics — nowhere near the scope of the `var`
migration below, and it removes an entire mechanism (bare-vs-explicit
degradation, its own lint warning) that the alternate reading required.

So the four combinations stay exactly as independent as parameters already
are, with no collapsing in either direction:

| Form | Rebindable | Mutable |
|---|---|---|
| `let Type a` | no | no |
| `let mut Type a` (≡ `mut Type a`) | no | yes |
| `var Type a` | yes | no |
| `var mut Type a` | yes | yes |

#### This is a breaking change, on purpose — and the migration is compiler-driven

Retiring the `var` → `mut` auto-upgrade changes what compiles: **today**,
`var x = Counter(); x.inc()` already works (`BindingKind::is_mutable()`
currently returns `true` for both `Mut` and `Var`, and every `def`-call
check in the checker/interpreter/self-hosted interpreter keys off it — see
this session's `'actor`/`'guard` fix). Under this proposal, that exact line
would need to become `var mut x = Counter(); x.inc()` to keep compiling.

The scope of this is real — every `var` binding in the existing codebase
(stdlib, examples, `whisper-boring`, the self-hosted interpreter's own
sources, the test suite) that calls a `def` method or writes a field is
affected — but **it should not be pre-audited by hand.** Ship the stricter
rule first; the checker/transpiler will reject every affected call site with
a concrete "`cannot call mutating method '{method}' on non-mut binding
'{name}'`"-style error, at the exact line that needs `mut` added. That error
list *is* the audit — exhaustive by construction, unlike a grep-based sweep,
and self-correcting: a `var` that never actually needed mutation surfaces
as "nothing to fix here," which is itself useful information (it was
over-permissioned before, silently).

**Decided: `'actor`/`'guard` get no special case.** This revisits the table
this session already shipped in `binding-mutability.md`: today, `var
T'actor x` allows `def` (rebind ✓, `def` ✓) without needing `mut` written
anywhere, because `is_mutable()` already covers `Var`. That table is now
superseded — full consistency with the four-combination table above wins:
`var T'actor x` (no `mut`) stops being enough; only `var mut T'actor x`
allows `def`, mirroring plain owned types exactly, with no qualifier-specific
exception. This narrows what compiles beyond the general `var` migration
above — every existing `var T'actor x` / `var T'guard x` that calls a `def`
method needs `mut` added too, on top of every plain `var Counter x` case —
and is why it's called out separately here rather than folded silently into
the general count: it's an *additional* pass over `'actor`/`'guard` usage
specifically, not automatically covered by fixing `is_mutable()` for plain
owned types alone. `binding-mutability.md`'s existing `'actor`/`'guard`
table needs updating to match once this ships.

### Where `mut Type` is rejected outright

`mut Type` — bare or explicit, they're the same thing now (§1) — is a
checker error, not a no-op and not a silent downgrade, in every one of
these cases:

| Form | Why |
|---|---|
| `mut int x`, `mut float f`, … (any scalar) | no `def` methods exist on primitives — nothing for `mut` to unlock. Retires the historical "`mut` ≡ `var` for scalars" shortcut (§1) — write `var` explicitly for a rebindable scalar, same as for anything else |
| `mut Type'shared x` | `'shared` has no interior mutability at all — already enforced today (`check_qualifier_constraint`), independent of this proposal, and this decision keeps it that way with no future revisiting needed |
| `mut t = (1, 2)` / `mut (T1, T2) t = ...` | tuples have no in-place mutation surface, whether the type is explicit or only inferred from a literal initializer — shipped this session (`check_tuple_mut_constraint`, extended to also cover the inferred case alongside this document), and — like the `'shared` case — this decision means that check is *permanently* correct as shipped, not an interim strict default awaiting a later loosening |
| `mut Type'shared'weak x`, `mut Type'actor'weak x`, `mut Type'guard'weak x` | a `'weak` reference has no operations besides `.upgrade()`/`.clone()` (both non-mutating) until it's upgraded — there's nothing for `mut` to unlock on the weak reference itself, regardless of what the *upgraded* value would allow. **Not currently enforced** — `check_qualifier_constraint` only walks into `'shared`, never checks for `'weak` at all; this is a real gap to close alongside the others, not merely a corollary already covered |
| `{mut T}` (set element type) | `HashSet<T>` has no mutable element access at all (`iter_mut`/`get_mut` don't exist on it, unlike `Vec`/`HashMap`) — a Rust API limitation, not a Boring design choice (§3) |

**Side finding while checking this:** `T'actor'weak` and `T'guard'weak` didn't even *parse* until a session fix alongside this document — the parser's `'actor`/`'guard` branches eagerly consumed the tick right after matching, only accepting `'task`/`'global`/`'unified` (`'actor`) or `'task` (`'guard`) as a continuation, so `'weak` could never be reached even though the generic chained-`'weak` logic a few lines down already correctly listed `Actor`/`Guard` alongside `Shared`. `T'shared'weak` was unaffected and already worked. Both parser sites fixed (`src/parser/parse_type.rs`, `src/parser/mod.rs`'s separate lookahead scanner), with regression tests (`test_actor_weak_parses_and_upgrades`, `test_guard_weak_parses_and_upgrades`).

### 2. The borrow form composes freely: `mut Type&`

Since `mut` now lives in the `type` production, `T&`/`mut T&` (`&T`/`&mut T`)
nest anywhere a `type` can appear — tuples, generic argument lists (foreign
types included), array/dict element positions:

```boring
Query<(Position&, mut Velocity&)>
let (Position&, mut Velocity&) pair = (p, v)
```

This is the motivating case from the Bevy-ECS discussion earlier this
session and has no remaining semantic question — `&T`/`&mut T` are ordinary,
distinct Rust types, and nesting them is exactly the kind of grammar
generalization the `type` production already supports for every other
compound form (`type "&" borrow_qual` is already a `type` alternative — see
`spec/grammar.bnf`; it simply isn't reachable through a `"mut"?` prefix yet).

**Spelling: `mut Type&`, not a sigil.** A terser single-character alternative
(`Type!` for the mutable form, mirroring Ruby's bang-mutates convention) was
considered and dropped: `!` already does double duty in Boring (prefix
boolean negation, macro-call suffix), and a third meaning for the same
glyph was judged too much to carry even though the grammar positions don't
collide. `Type*`/`Type&` (borrowing C/C++'s pointer/reference sigils) was
also considered and dropped for a sharper reason: `*` reads unambiguously as
"raw pointer" to any C/C++ developer, which is not what it would mean here
(mutable *reference*, not a pointer) — recycling it would be actively
misleading rather than merely unfamiliar. The verbose, keyword-based
`mut Type&` stays — consistent with the rest of Boring's surface syntax,
which already favors spelled-out keywords (`let`/`mut`/`var`/`req`/`def`)
over sigils wherever Swift/Python precedent allows it.

### 3. The owned form composes selectively: `mut Type`

`mut Type` (bare, no `&`) is meaningful only where a **stable, finite,
statically-known position** exists to attach a permission flag to — the same
requirement per-field mutability on structs needs too, whether or not it's
fully built yet (it isn't — see below):

- **Tuple slots** — `(mut Point, string)`. Each position is fixed and
  addressable (`t.0`, `t.1`), exactly like a struct field.
- **Enum variant fields** — **shipped**, unlike the struct-field case right
  below (`enum Holder: Value(mut Point p)`). Each variant field is exactly as
  fixed/addressable as a tuple slot or struct field, so the same mechanism
  applies with no new grammar — `mut` reaching a variant field's type is an
  incidental consequence of `parse_type` accepting the `mut`-prefix anywhere
  a type can appear (§2), not special-cased parsing. What *is* new, specific
  to enums: whether the enum's own `def` methods get `&self` or `&mut self`.
  By default every enum method (`req` or `def`) transpiles to `&self` —
  variants have no mutable fields to justify anything else, so `def` there is
  documentation intent only, exactly like a top-level free function (see
  `binding-mutability.md`/`CLAUDE.md`'s enum section). An enum with at least
  one `mut`-qualified variant field is the one exception: it's checked **per
  enum type**, not per method body — if the type has such a field anywhere,
  every `def` method on it gets a genuine `&mut self`, and calling one now
  requires the enum instance itself to carry `mut`/`var mut`, same as a
  struct's `def` method. Matching bare `self` inside such a method hits
  Rust's own match ergonomics (implicit `&mut` binding mode), which binds a
  matched variant field directly as `&mut T` with no `mut`/`ref mut`
  annotation on the pattern — and rejects one if written, since an explicit
  `mut` binding modifier conflicts with an implicit by-reference binding
  mode. Matching a plain *owned* enum local (not `self`) still needs the
  usual `mut`-promotion on the bound pattern name to call a `def` method
  through it, same as any owned match subject. Full walkthrough, with the
  Rust output for both the mut-field and no-mut-field cases, in
  [book.md §9, "Enum variant fields — `mut Type`"](book.md#enum-variant-fields--mut-type).
- **Struct fields** — **not actually shipped yet, an earlier revision of
  this document overclaimed it was.** `mut` isn't even a valid field
  keyword today (`struct S: mut Point p` fails to parse — fields only
  accept `var` or no keyword, i.e. `FieldDecl { mutable: bool }`, a single
  boolean, not the two independent axes this document is about). And
  `var Point p`, verified empirically, currently grants **both**
  reassignment (`self.p = newInstance`) **and** content mutation
  (`self.p.inc()`) under the same keyword — exactly the conflation §1
  retires for local bindings, just never addressed for fields at all.
  `self.p = x` (does the field point at a different instance?) and
  `self.p.inc()` (can whatever it currently holds be mutated?) are as
  independent for a field as they are for a local variable — a struct
  field can itself hold a struct with its own methods, so this isn't
  degenerate the way it is for scalars. Fields should get the same
  four-combination table as §1, not a two-state `mutable: bool`:

  | Field | Reassignable | Content mutable |
  |---|---|---|
  | `let Point p` (today's default) | no | no |
  | `mut Point p` / `let mut Point p` (doesn't parse today) | no | yes |
  | `var Point p` (today grants both, incorrectly) | yes | no |
  | `var mut Point p` (doesn't parse today) | yes | yes |

  This carries the same breaking-change shape as §1's `var` migration,
  applied to fields: every existing `var Point p` field whose owning
  struct calls a mutating method on `p` would need to become
  `var mut Point p` to keep compiling. Not part of this session's shipped
  work — flagged here as a real, in-scope gap this document should own,
  not a separate proposal, since it's the exact same model applied one
  level down. **Kernel struct fields are the one place this genuinely
  doesn't apply**: a kernel struct has exactly one method (the anonymous
  `def ()` body), so there is no second, independent "call a method on
  what this field currently holds" operation to distinguish from
  "write the field" — confirmed by reading every `FieldBinding` use site
  across the checker, interpreter, and all four GPU backends, where `Mut`
  and `Var` are never treated differently. A kernel field's write
  permission is genuinely one axis, correctly (if redundantly — three
  keywords for two states) expressed today.

Collections add a wrinkle this document originally got wrong: there are
**two independent places** `mut` can attach to a collection, not one, and
they control genuinely different things:

- **`mut [Point] arr`** — `mut` on the *collection's own type*. Controls
  *structural* mutation: `arr[i] = v`, `.push()`, `.insert()`,
  `.remove()` — adding, removing, and replacing elements. This is not a
  special case at all: arrays/dicts/sets already have genuine in-place
  mutation, so this is the exact same general rule as any struct, applied
  to the collection as a whole.
- **`[mut Point] arr`** — `mut` *inside the brackets*, on the **element
  type**. Controls whether `def` methods can be called on whatever comes
  back from indexing or iterating — `arr[0].inc()` — uniformly, for every
  element, regardless of which index you fetch.

These compose independently, exactly like the field-vs-binding split
established earlier in this document: `let [mut Point] arr` — the array
itself is fixed (no push/pop, no reassigning `arr`), but every element it
already holds can still be mutated in place. `mut [Point] arr` — the array
can grow/shrink/have entries replaced, but the elements it holds can't have
`def` called on them (their type is plain `Point`). `mut [mut Point] arr` —
both. This isn't a new mechanism — it's `mut Type` nested inside `[T]`,
exactly the same nesting `mut Type&` already gets in §2, just for the owned
form instead of the borrowed one.

**An earlier version of this document rejected `[mut Point]` outright,
conflating two different questions.** The rejection was about *per-index*
differentiation — "index 3 is mutable, index 5 isn't" — which is correctly
unsupportable (no stable identity across resizes). But `[mut Point]` doesn't
ask for that: it declares one *uniform* element type for the whole
collection, the same way `[Point]` always has one uniform element type.
There's no per-index bookkeeping needed at all — only "what element type
was this array declared with," tracked once per collection binding, same as
today.

**Dicts follow the same pattern, on the value position:** `{K = mut V}`
controls whether `def` calls work on values fetched via `d[k]` or
iteration; `mut {K=V} d` controls `d[k] = v`/insertion/removal, structurally.
Keys aren't a candidate for `mut` in either language — mutating a key in
place would invalidate the hash table, for either Boring or the underlying
Rust `HashMap`.

**Sets cannot support the element-mutable form at all — not a design
choice, a Rust limitation.** `{T}` transpiles to `HashSet<T>` (`CLAUDE.md`),
and `std::collections::HashSet<T>` deliberately exposes **no** mutable
element access whatsoever — no `iter_mut()`, no `get_mut()` — because
mutating an element in place could change its `Hash`/`Eq` behavior and
silently corrupt the set's internal buckets. `Vec<T>` (`iter_mut()`,
`get_mut()`) and `HashMap<K, V>` (`values_mut()`, `get_mut()`) both expose
exactly the mutable access `[mut Point]`/`{K = mut V}` need; `HashSet<T>`
structurally cannot. `{mut T}` should be a checker error, not a silent
no-op or a best-effort attempt — there is no Rust API for the transpiler to
target even if the checker allowed it.

One consequence worth stating plainly: since *structural* collection
mutation (`arr[i] = v`, `.push()`, `.add()`/`.remove()`) is already gated on
`var`/`mut` today the same way struct `def` calls are (`CLAUDE.md`: *"Index
assignment ... requires a `var`/`mut` binding"*), the `var` → `mut`
auto-upgrade retirement
[above](#this-is-a-breaking-change-on-purpose--and-the-migration-is-compiler-driven)
affects every `var [T] arr = ...; arr.push(...)` /
`var {K=V} d = ...; d[k] = v` pattern in the existing codebase too, not just
struct method calls — the same compiler-driven migration story applies, just
with a broader set of affected call sites than struct `def` calls alone.

Generic Boring struct type parameters (`Container<mut Point>` for
`struct Container<T>: T item`) are also **out of scope** for this proposal —
propagating a `mut`-qualified type argument through generic instantiation
into a field's permission check is real, separate design and
implementation work, not a corollary of anything here.

### 4. Destructuring: per-element keyword, two different defaulting rules

Each destructured element may carry its own explicit `let`/`mut`/`var`
keyword. The only question is what an element **without** one resolves to —
and that depends on whether the destructure is parenthesised:

- **Parenthesised** — an unmarked element inherits the keyword written
  before the opening parenthesis (the group's own leading keyword).
- **Bare** (no parentheses) — an unmarked element defaults to `let`,
  unconditionally, regardless of what keyword a different element in the
  same statement carries.

Worked through every combination. `let mut` is written out below, but per §1
the bare `mut` shorthand means exactly the same thing — there's no
degradation or context-dependent reading to account for, so this table is
the complete story, not a simplified stand-in for a more complex bare-`mut`
case:

```boring
let a, b = t                  #  let a, let b       — bare, b unmarked → let (default)
let (a, b) = t                  #  let a, let b       — parens, b unmarked → inherits let
let mut a, b = t                #  let mut a, let b   — bare, b unmarked → let (default,
                              #                        ignores a's keyword)
let mut a, let b = t            #  let mut a, let b   — both explicit, no defaulting
let a, let mut b = t            #  let a, let mut b   — both explicit, no defaulting
let mut (a, b) = t               #  let mut a, let mut b  — parens, leading `let mut` →
                              #                        group default; both unmarked
let (let mut a, b) = t           #  let mut a, let b   — parens, a's own `let mut`
                              #                        overrides; b unmarked → inherits
                              #                        the group's leading `let`
let mut (a, let b) = t           #  let mut a, let b   — parens, a unmarked → inherits
                              #                        group default `let mut`;
                              #                        b explicit, overrides to `let`
```

This is a real readability trap on the bare form specifically: `let mut a,
b = t` reads, at a glance, like the keyword phrase governs the whole line —
it doesn't; `b` quietly defaults to plain `let`. The rule is unambiguous
once learned, but a linter warning is worth adding for exactly this shape
(an unmarked bare element following a differently-keyworded one) — see the
implementation checklist.

Today, `let a, b = t` and `let (a, b) = t` are declared fully interchangeable
(`book.md`, "Destructuring": *"All four forms are equivalent"*) — and they
still are, for the all-same-keyword case. Parentheses now matter, but only
in the one case where an element is left unmarked next to a differently-
keyworded one: bare defaults to `let` independently per element; parens
share the group's own keyword as the default instead. Every statement that
uses a single keyword throughout, marked or not, means exactly what it does
today.

Grammar-wise, both forms use the **same** `binding` production — the
difference is purely in the semantic default resolution, not the grammar.
Each element's own keyword is the same two-part shape as a top-level
`let_stmt`'s (§1: `let`, `let mut`, `var`, `var mut`, or the bare `mut`
shorthand), not a single token:

```bnf
destructure ::= "(" binding ("," binding)* ")"
              | binding ("," binding)*
binding      ::= (("let" | "var") "mut"? | "mut")? type? IDENT
```

with the statement's own leading keyword phrase still mandatory on the
*first* element (to disambiguate a `let_stmt` from a plain expression
statement), and used as the parenthesised form's per-element default — but,
per the rule above, not propagated across a bare list. This is purely
additive — every existing destructure keeps its current meaning.

### 5. Coercion: `mut Type` widens to `Type`, never the reverse

A `mut Type` value carries strictly more permission than a plain `Type`
value (it may additionally be passed wherever a `def` call is required) —
so it is always safe to use a `mut Type` value where only `Type` is
expected, and never safe the other way around:

```boring
def readOnly(Point p): ...        # accepts Type
def mutate(mut Point p): p.x = 1  # requires mut Type

mut Point a = ...
let Point b = ...

readOnly(a)   # OK — mut Point widens to Point
mutate(a)     # OK — a already is mut Point
mutate(b)     # ERROR — Point cannot be used where mut Type is required;
              #         b was never granted that permission
```

This is a one-way coercion, exactly mirroring `mut T&`/`T&` today (a `&mut T`
reference could always be reborrowed as `&T`, never the reverse) and the
existing `let`/`mut`/`var` parameter-passing hierarchy
(`binding-mutability.md`'s "Function parameters": *"a caller should be able
to pass down the hierarchy but never up"*). The **transpiler** must verify
this at every call site, field assignment, and return: a `Type`-typed value
flowing into a `mut Type`-typed position is a compile error, not something
Rust's own type system would catch on its own (since, as established
throughout this document, there is frequently no distinct Rust type behind
`mut Type` at all — the checker is the only thing enforcing the direction).

### 6. Type inference never upgrades to `mut`

When a variable's type is inferred from another variable's value rather
than written explicitly, inference resolves to the **plain** type, never to
`mut Type`, even if the source value happened to be `mut`-typed:

```boring
let mut Point b = ...
let a = b       # a's inferred type is Point, NOT mut Point
```

`mut`-ness is a property requested at a specific binding site, not something
that flows automatically through aliasing or copying. If it propagated
automatically, a `let a = b` far away from `b`'s declaration could silently
grant `a` mutation rights nobody asked for at that specific line — exactly
the kind of implicit widening [§5](#5-coercion-mut-type-widens-to-type-never-the-reverse)
forbids in the other direction, applied here to *inference* rather than
*coercion*. Getting `a` typed `mut Point` requires writing it explicitly —
`let mut Point a = b` or `mut a = b` — at which point [§5](#5-coercion-mut-type-widens-to-type-never-the-reverse)'s
rule applies normally (legal here, since `b` already is `mut Point`).

## Grammar changes required

```bnf
# type production gains a modifier form, usable anywhere `type` is:
type ::= ...
       | "mut"? type "&" borrow_qual   # mut Type& → &mut Type (always valid)
       | "mut"? type                   # mut Type  → valid only where checker
                                        #   allows it (tuple slot, struct field,
                                        #   array/dict-value element type)

# array and dict types accept `mut` on the element/value position:
# "[" type "]"        already covers "[" "mut"? type "]" via the type rule above
# "{" type "=" type "}" ditto for the value half — keys never accept `mut`

# destructure binding gains its own optional keyword:
binding ::= ("let" | "mut" | "var")? type? IDENT

# struct field declarations move from a single `mutable: bool` to the same
# three-way (plus the "mut"-alone shorthand) as everything else — see §3:
field_decl ::= (("let" | "var") "mut"? | "mut")? type IDENT ("=" expr)?
```

The bare `"mut"? type` form is intentionally **not** restricted at the
grammar level to any particular position — parser-level context isn't the
right place to enforce this (a tuple type used as a function return type or
a generic argument, for instance, is still just a `type`). The restriction
to "stable, checkable positions" is enforced by the **checker**: `mut Type`
parses everywhere a `type` does, and is rejected wherever the checker can't
attach a permission to it — a bare local variable (§1's rules there apply
instead), a set element (§3 — no Rust API to target), a generic argument to
a foreign type, etc. Tuple slots, struct fields, array elements, and dict
values are all valid; arbitrary nesting elsewhere is not assumed to be
until specifically checked.

## Interactions and invariants

- **Transpiler honesty.** Rust has no per-tuple-element `mut` — a tuple with
  any `mut` slot still requires `let mut t = (...)` on the *whole* Rust
  binding, exactly as `let mut` is already required in Rust to call any
  `&mut self` method regardless of Boring's per-field tracking. The checker
  must independently guarantee that Boring source never calls a `def` method
  or writes through a non-`mut` slot — the Rust binding being uniformly `mut`
  is not itself a green light.
- **Qualifier capping still applies.** `mut Type'shared` remains impossible
  to make meaningful — `'shared` forbids `def` unconditionally, matching the
  existing `mut 'shared` compile error (`checker/mod.rs`'s
  `check_qualifier_constraint`). `mut Type'actor`/`mut Type'guard` match this
  session's shipped fix: `let T'actor x` forbids `def`, `mut T'actor x`
  allows it (mutation goes through the lock, not `&mut self`, but Boring
  still gates it on the binding, same as every other type) — `var T'actor x`
  alone, decided in §1, no longer suffices either; `var mut T'actor x` is
  required, no qualifier-specific exception.
- **No conflict with the shipped tuple check.** `check_tuple_mut_constraint`
  (this session) rejects `mut` **before** a tuple's parentheses — `mut (T1,
  T2) t = ...`, mut on the whole tuple, which stays correctly rejected (a
  tuple as a block has no mutation surface, only per-slot does). This
  proposal's per-slot form is **inside** the parentheses — a different
  grammar position, no collision.

## Implementation checklist

0. **`BindingKind::is_mutable()` (`src/ast/mod.rs`) stops being the source of
   truth for "`def` calls allowed."** It currently returns `true` for both
   `Mut` and `Var` and is read directly by every `def`-call check across the
   checker, interpreter, and self-hosted interpreter (this session's
   `'actor`/`'guard` fix included). Under this proposal, permission comes
   from whether the *resolved type* carries `mut`, not from the
   `BindingKind` alone — `Var` without a `mut`-typed value must no longer
   satisfy these checks, **`'actor`/`'guard` included, decided in §1 with no
   exception** (`var T'actor x` alone stops being enough; only `var mut
   T'actor x` does) — update `binding-mutability.md`'s `'actor`/`'guard`
   table to match. This is the change with the widest blast radius in this
   list; do it first, and expect it alone to surface the bulk of the
   migration errors described above.
1. `spec/grammar.bnf` — the two `type` alternatives above; `binding`'s own
   optional keyword, usable in both the bare and parenthesised destructure
   forms; `field_decl` gains the same `let`/`mut`/`var` three-way (today it
   only accepts `var`/no-keyword — a single `mutable: bool`, `src/ast/mod.rs`'s
   `FieldDecl`), so `mut`/`let mut`/`var mut` become writable on a field at
   all.
2. `src/checker/mod.rs` — per-slot permission tracking for tuple types and
   destructure bindings (new; today only whole-bindings are tracked via
   `BindingKind` — struct fields track only reassignment via `mutable: bool`,
   not content mutation at all, see §3); reject `mut Type` (bare) everywhere
   outside tuple-slot / struct-field / array-element / dict-value position
   with a clear diagnostic; reject `mut` on scalars outright (§1), not
   silently ignore it; resolve each destructured element's default per the
   bare-vs-parenthesised rule above; **lint warning** (not an error) on a
   bare unmarked element following a differently-keyworded one in the same
   statement (e.g. `mut a, b = t`) — correct but a readability trap, worth
   flagging even though it's valid; enforce the one-way `mut Type` → `Type`
   coercion at every call site, assignment, and return (§5), and resolve
   inferred `let`/`var` types to the plain type, never `mut Type`, when the
   source is `mut`-typed (§6); split `self.field = x` (reassignment) from
   `self.field.method()` (content mutation) into the same two independently
   gated operations a local binding already gets (§3) — today `var Point p`
   grants both under one flag.
3. `src/interpreter/*.rs` — matching per-slot bookkeeping for `boring run`
   (extends the per-binding-name `mutable_vars`/`is_mutable` machinery to
   per-tuple-position and to the field-reassign/field-mutate split above).
4. `src/transpiler/*.rs` — emit `let mut` on the whole Rust tuple binding
   whenever any slot is `mut`; rely on (2)/(3) having already rejected any
   source that would misuse the other slots.
5. `boring/interpreter/*.br` (self-hosted interpreter) — same two changes as
   (3)/(2), scaled down to its simpler qualifier-blind model, once the Rust
   side is settled.
6. Retire the "`mut` ≡ `var` for scalars" line from `CLAUDE.md`'s cheat
   sheet once (1) ships, to resolve the doc conflict noted above.

## Explicitly out of scope (future work, not corollaries)

- `{mut T}` on **sets specifically** — not a scoping choice but a hard Rust
  limitation: `HashSet<T>` exposes no mutable element access at all (see
  §3). `[mut Point]` (arrays) and `{K = mut V}` (dicts) are both in scope —
  an earlier revision of this document incorrectly excluded them too.
- Full checker enforcement of the `FieldBinding`/`BindingKind` unification
  proposed in
  ["Kernel structs"](#kernel-structs-a-distinct-related-model-one-axis-not-two)
  below.
- Generic Boring struct type parameters (`Container<mut Point>`) —
  requires propagating qualified type arguments through generic
  instantiation into field checks; separate design effort.
- Full checker enforcement of the parameter model in
  ["Parameters"](#parameters-a-distinct-related-model-let-stays-implicit-under-mut)
  below — the model itself is specified there, but implementing it (closing
  the "`mut T&`/`var T&` both transpile to `&mut T` today" enforcement gap)
  is separate work from this document's `let_stmt`/destructuring focus.

## Parameters: a distinct, related model — `let` stays implicit under `mut`

Function/method parameters use `mut`/`var` too, but the bare `mut` keyword
does **not** collapse to `var mut` there the way §1 says it does for
`let_stmt`s — parameters need their own, asymmetric rule, for a reason that
doesn't apply to local bindings at all.

`let` is already implicit on every parameter with no keyword (`compute(Point
a)` ≡ `compute(let Point a)` — an established rule, not new). Adding `mut`
doesn't remove that implicit `let` — `compute(mut Point a)` ≡ `compute(let
mut Point a)`: the callee may mutate `a`'s content, but cannot replace what
the *caller's* variable holds. Only `var`, written explicitly, unlocks that:

| Param form | Rebind caller | Mutate content |
|---|---|---|
| `Point a` (no keyword) | no | no |
| `mut Point a` (≡ `let mut Point a`) | no | yes |
| `var Point a` | yes | no |
| `var mut Point a` | yes | yes |

**This is the opposite collapse direction from §1's local-binding rule, on
purpose.** For a local binding, `var` only ever affects the *same scope* —
rebinding a name to point elsewhere later costs nothing outside that scope,
so letting the bare `mut` shorthand quietly include it is a reasonable
convenience. For a parameter, `var` grants the callee the ability to
overwrite the *caller's own variable*, across the call boundary — a
materially bigger capability that must stay opt-in, spelled out explicitly,
never implied by `mut` alone. `compute(mut Point a)` silently becoming
`compute(var mut Point a)` would hand every `mut`-parameter callee
caller-rebind rights it doesn't have today, as a side effect of a rule
designed for a completely different context — exactly the trap to avoid.

This table describes intent that mostly isn't enforced yet — `mut T&` and
`var T&` currently both transpile to the identical Rust `&mut T`
(`binding-mutability.md`), so nothing today actually stops a `var`-parameter
callee from mutating content it's only supposed to reassign, or a
`mut`-parameter callee from attempting to reseat the caller's variable
undetected by the checker. Closing that gap — and introducing `var mut T&
m` as a real, checker-distinguished combination — is real, related work,
but it belongs in its own follow-up proposal with its own migration
accounting, not folded into this one.

## Kernel structs: a distinct, related model — one axis, not two

`kernel struct` fields are in the same position as parameters: `let`/`mut`/
`var` appear there, using the same words, but mean something narrower than
anywhere else in this document. They deserve the same treatment as
Parameters — spelled out here, not left as a line in "out of scope" —
precisely so the difference is explicit rather than something to
rediscover by testing.

A kernel field's declaration parses into `KernelFieldDecl`
(`src/ast/mod.rs`), not `LetStmt` — a structurally separate AST node, with
its own `binding: FieldBinding` (a distinct enum from `BindingKind` — same
three names, `Let`/`Mut`/`Var`, different Rust type) and its own
`qual: GpuQual` (`Unified`/`Global`/`Actor`/`Local`/`Const`/`ActorGlobal`/
`ActorUnified`, …), with no overlap at all with `OwnerQual`
(`Shared`/`Actor`/`Guard`/`Stack`/`Heap`/`Weak`), which this entire document
is built on. `check_let_stmt` — where this session's checks live — only
ever sees `Stmt::Let(LetStmt)`; a `KernelFieldDecl` never reaches it.
Verified directly: a `mut (int, int)` kernel field, the exact shape
`check_tuple_mut_constraint` rejects for an ordinary `let_stmt`, compiles
without error today via `boring run` (which uses the full
`checker::check()`, not the separate, more restricted
`check_kernel_dispatch_only` used only for `--target cuda/rocm/metal/wgpu`
builds — an earlier revision of this document incorrectly attributed the
exclusion to that guard instead of to the AST split).

**Why `FieldBinding` only needs one axis, unlike a struct field.** A
`kernel struct` has exactly one method — the anonymous `def ()` kernel
body (`KernelDecl`'s own doc comment: *"Only one anonymous `def ()` is
allowed"*). A regular struct field can hold a value with its *own* methods,
which is exactly why fields need the same reassign/mutate split as local
bindings (previous section). A kernel field has no second method to call
through — the kernel body either writes the field directly (`y[0] = 1.0`)
or it doesn't; there is no `self.field.someOtherMethod()` to distinguish
from `self.field = x`, because nothing else ever touches the field. One
axis is not a gap here — it's the correct shape for what a kernel field
actually is.

**But the *encoding* of that one axis is redundant, and worth cleaning up
regardless.** `FieldBinding` demands one of three keywords for what is
verified, exhaustively, to be a two-state outcome — every use site across
the checker, the interpreter's GPU simulation, and all four transpiler
backends (CUDA, ROCm, Metal, wgpu) treats `Mut` and `Var` identically:

```rust
// src/interpreter/eval_gpu.rs
match field_decl.binding {
    FieldBinding::Mut | FieldBinding::Var => thread_env.borrow_mut().define_mut(name, val),
    FieldBinding::Let                     => thread_env.borrow_mut().define(name, val),
}
// every transpiler backend: matches!(f.binding, FieldBinding::Let) → const/read-only, else → writable
```

`mut` and `var` are pure synonyms on a kernel field, in every code path
that reads `FieldBinding`, with no exception found. This is real,
independent of everything else in this document: `FieldBinding` could be
dropped entirely and replaced with `BindingKind` (or reduced to a plain
two-state flag, matching how a *regular* struct field is represented today
— `FieldDecl { mutable: bool }`), with zero behavior change, since the
third state was never distinguishable to begin with. Recommended as a
small, low-risk, in-scope cleanup for this document to carry — not a
corollary of the local-binding or struct-field models above (kernel fields
still don't need reassign/mutate split), just a redundant enum worth
retiring on its own merits.

`GpuQual` stays exactly as separate as it already is — real, hardware-tied
memory-space and atomics information (`gpu-module.md`) with no CPU-side
analog, unrelated to `mut Type`'s subject matter regardless of what happens
to `FieldBinding`.
