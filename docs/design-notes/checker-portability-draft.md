# Draft — anticipating errors on Boring's side, with a future multi-backend target (Rust/Swift/Kotlin)

Status: **working draft**, not a spec.

**Update**: step 1 (mechanical separation) has been carried out — the 11 Rust/GPU-specific
functions listed in §1 were moved out of `checker/mod.rs` into a new file
`checker/rust_checks.rs`, keeping a single `struct Checker` and a single shared scope-tracking
state (no duplicated tracking). No behavior change: the same 41 checker unit tests and the
`owned_ctor_use_after_move_build_fails`, `owned_param_mut_def_call_build_fails`,
`qualifier_conflict_build_fails`, `static_qualifier` integration tests pass identically. Technical
details:
- The moved functions stayed methods of `Checker` (same struct, same private fields
  `scopes`/`moved`/`open_with_names`/... defined in `mod.rs`) — a Rust submodule
  (`checker::rust_checks`) can freely read its parent module's private fields, so no field
  visibility had to change.
- The **methods** themselves, however, when called from `mod.rs`, had to go from private to
  `pub(super)` (Rust privacy is scoped to the module where the item is *textually defined*, not to
  the struct — a private item defined in a submodule isn't visible from the parent). 11 methods are
  therefore `pub(super)` in `rust_checks.rs`; the 5 helpers purely internal to that file
  (`type_has_shared`, `type_has_static`, `describe_type_for_atomic_error`,
  `qualifier_name_for_kernel_dispatch`, `type_is_atomic_qualified`) stayed private.

**Update 2**: two new universal checks were added to `checker/mod.rs` (item 2 in the
recommendations below) — dead/unreachable code (`check_dead_code`) and missing-return-on-some-path
(`check_missing_return`), sharing one control-flow "does this settle" engine (`stmt_settles`/
`block_settles`, plus `loop_body_has_own_break` for the `loop:`/`while true:`/`do...while true:`
case). Both ship as non-fatal warnings (the previously-unused `Checker::warning()` channel now has
its first two callers), deliberately conservative in opposite directions for the two checks: never
claim dead code that might not be, and never claim a missing return that might actually be covered
(match/if exhaustiveness is optimistically assumed, not proven — see §3 item 3/open questions on
real exhaustiveness). Validated empirically against the full non-fixture `.br` corpus of this repo
(313 files) plus every sibling Boring project under `perso/` with real content (`scratch-boring`,
`whisper-boring`, `bevy-boring`, `breakout-boring` — 46 more files): **zero false positives**; the
only hits were genuine dead code, including two cases where the affected project's own code comment
had already manually documented the exact same issue as a known workaround (see
`check_dead_code`'s doc comment for the `while true:`/`do...while true:` caveat this uncovered — a
real, current transpiler asymmetry: a bare `loop:` with no break lowers to a Rust `loop {}` that
`rustc` recognizes as diverging, but `while true:`/`do...while true:` don't get the same treatment,
so removing the resulting dead code would currently regress `boring build` even though it's
genuinely unreachable in Boring's own semantics — the warning message says so explicitly rather
than giving misleading advice). 16 new unit tests added (`dead_code_tests`, `missing_return_tests`).
Full `cargo test` green.

**Update 3**: a third universal check was added — function-call arity / labeled-argument
validation (`check_call_arity`), also a warning. Covers four purely structural, unambiguous cases:
unknown labeled argument, an argument bound both positionally and by label, too many positional
arguments, and a missing required argument. Scoped to plain free-function calls only
(`ExprKind::Call(Var(name), _)` — method calls need receiver-type-based overload resolution this
checker doesn't have) and guarded against the two real ways this could otherwise misfire: a name
with more than one `FnDecl` (Boring resolves overloads by argument *type*, which this checker can't
infer — skipped entirely, mirroring `struct_ctor_owned`'s "more than one `init` → skip" precedent),
and a name shadowed by a local variable (a closure/function-typed binding called through the same
AST shape — skipped via `self.lookup(name)`). Validated the same way as checks #3/#4: zero false
positives across the same 313+46-file corpus, and the **one** real hit was a direct, textbook
confirmation of the check's value — `tests/cases/error_variadic_too_few_args.br`, whose own header
comment already documented "never validated by the checker" as the root cause of a transpiler
*panic* (`emit_args_coerced` slicing out of bounds) that a prior audit had to patch defensively
rather than prevent at the source. 10 new unit tests (`call_arity_tests`).

**Update 4**: the fourth candidate — `match` exhaustiveness (`check_match_exhaustiveness`) — was
added too, scoped exactly to the narrower version this document originally recommended rather than
full enum-variant tracking: a subject statically known to be `bool` (a boolean literal, a
comparison/logical/identity expression, or a `Var` declared `bool`) or an optional (`T?`, a `Var`
declared `Type::Optional`). `Pattern::Wildcard`/`Pattern::Bind(_)` covers every remaining case; a
guarded arm (`pattern if cond:`) never counts toward coverage on its own, matching real Rust's own
match-guard semantics exactly (confirmed with a dedicated test: a guarded `true` arm plus a plain
`false` arm still correctly warns "missing case `true`"). Full user-declared enum exhaustiveness
remains out of scope — still needs the variant-list registry this checker doesn't have. Validated
the same way as the other three: zero hits across the full 313+46-file corpus (this codebase's
existing bool/optional matches are already exhaustive or wildcard-covered), confirmed to actually
fire via synthetic positive cases (both bool and optional), and confirmed silent on the matching
exhaustive/wildcard/unresolvable-subject negative cases. 7 new unit tests
(`match_exhaustiveness_tests`). All four candidates from the original inventory are now implemented.

**Update 5**: the "still open" item — full exhaustiveness over user-declared enum variants — is
now implemented too. `Checker` gained an `enums: HashMap<String, Vec<String>>` registry (enum name
-> its variant names in declaration order), collected up front in `collect_item_signatures`/
`collect_stmt_signatures` alongside the existing `kernel_decls`/`fn_arity` collection, covering both
top-level and nested (`mod`/local) enum declarations. `check_match_exhaustiveness` now tries a third
resolution after bool/optional: `static_enum_subject` (same `Var`-only, best-effort limitation as
`static_bool_or_optional_subject` — a statically-typed local/param/field binding, not a general
type-checker) resolves the subject's base type (after peeling `mut`/ownership-qualifier wrappers,
reusing `strip_qualifiers`) against the `enums` registry; `check_enum_match_exhaustiveness` then
diffs the arms' covered variant set against the full registered variant list, same guard-arm
semantics as bool/optional (a guarded arm never counts toward coverage on its own), and names every
missing variant in the warning rather than just flagging non-exhaustiveness generically. A qualified
pattern (`Error.Expired`, parsed as `Pattern::Variant("Error::Expired", _)`) is matched by its
trailing segment. A `native` enum (`enum Name: native` — body provided by the runtime, not parsed
variants) registers with an empty variant list, which naturally disables the check for it (nothing
can ever be "missing" against an empty set) with no separate skip condition needed. Validated
empirically against the full corpus (346 local `.br` files under `examples/`,
`boring/interpreter/`, `stdlib/`, `linguist/`, `tests/`, plus the 46-file sibling-project corpus
under `perso/`): **zero hits, bool/optional/enum alike** — every real match in this codebase is
already exhaustive or wildcard-covered. Confirmed to actually fire via synthetic positive cases
(single missing variant, multiple missing variants — correctly pluralizes "variant"/"variants" and
lists all missing names), confirmed silent on exhaustive/wildcard/native/non-`Var`-subject negative
cases. 8 new unit tests (`enum_match_*`, `native_enum_is_never_flagged`, plus the pre-existing
`unresolvable_subject_type_is_never_flagged` rewritten to use a call-expression subject now that a
typed-`Var`-over-an-enum subject is no longer unresolvable). This closes the last item from
recommendation #2 below — every candidate originally inventoried, plus this follow-on, is now
shipped.

Goal of the rest of this document (still valid): map out what already exists in `src/checker/`
and `src/validator/`, and distinguish what is a rule **universal to Boring** (portable as-is to a
future Swift/Kotlin backend) from what is an **artifact of the current Rust model** (ownership/
borrow, or of the Rust-only GPU pipeline), in order to prioritize where to invest first to
maximize cross-backend reuse.

## Context / why this document

Medium-term product goal: `boring` today only transpiles to Rust, but the long-term ambition is
to add Swift and Kotlin backends. Any validation rule that lives in the **checker** (AST phase,
before transpilation, backend-agnostic by construction) and that encodes semantics **native to
Boring** (not to Rust) is an investment that counts for all three backends at once. Conversely, a
rule that *lives* in the checker but actually encodes a constraint of Rust's ownership model
(`Rc`/`Arc`/`Box`, borrowing) will be of no use to Swift (ARC) or Kotlin (GC) without being
rethought — and worse, could be a semantic non-sequitur in those languages (e.g. use-after-move on
`'owned` isn't a bug at all under ARC/GC).

Reminder of the principle already established earlier in the conversation: **a check's physical
location (in `checker/` vs. in the transpiler) does not by itself guarantee portability** — only
its semantic nature does. This document applies that lens to both existing modules.

## 1. `src/checker/mod.rs` — inventory

### Universal (pure Boring rules, no Rust dependency)

| Check | Location | Why it's universal |
|---|---|---|
| Immutability of `let`/assignment to a fixed `mut`/`lazy` (`?=` vs `=`) | `checker/mod.rs:1632-1670` | Boring's own binding model (`let`/`var`/`mut`/`var mut`) — true regardless of the target language |
| `mut` on a tuple (no mutation surface) | `checker/mod.rs:723-738` | Boring itself exposes no method/field assignment on its tuples — a language design fact, not a Rust one |
| `mut` on a scalar (no `def` methods) | `checker/mod.rs:748-783` | Boring's own `def`/`req` model — primitives never have user-defined methods |
| Cross-label consistency of multi-dim arrays + `as [...]` bijection | `checker/mod.rs:960-1063` | A type-shape rule native to the Boring array-multidim proposal, no link to the target at all |
| Checker recursion-depth guard | `checker/mod.rs:1185-1197` | Defensive infrastructure, not a semantic rule — backend-independent |

**Borderline case**: `{mut T}` on a set (`checker/mod.rs:799-809`) — the *rule* is universal
(mutating an element in place would invalidate the hash bucket, true for any hash-set in any
language), but the **error message literally cited `HashSet<T>`, `iter_mut`, `get_mut`** — Rust
vocabulary. Easy to generalize (reword the message), unlike the cases below, which are
structurally Rust. **Now fixed** — see item 4 in the recommendations below.

### Rust-specific (artifacts of ownership/borrow, or of the current Rust-only GPU pipeline)

| Check | Location | What anchors it to Rust |
|---|---|---|
| `mut 'shared`/`'static`/`'weak` — qualifier incompatibility | `checker/mod.rs:562-597` | Direct reasoning on `Rc`/`Arc` (no interior mutability), `&'static` (no internal mutation), `Weak` (nothing to unlock) |
| `'static` provenance gate (construction only authorized top-level/`main`) | `checker/mod.rs:643-694` | A workaround for Rust's `&'static` lifetime — Swift/Kotlin (GC) have no such notion of program-lifetime reference provenance |
| `'atomic` type compatibility (scalar int/bool only, no float/i128) | `checker/mod.rs:820-852` | The message cites `std::sync::atomic`, `AtomicI128` — follows exactly what stable Rust std exposes |
| Use-after-move on `'owned` constructor arguments | `checker/mod.rs:1744-1846` | Rust's move semantics (`Box<T>` exclusive, moved exactly once) — meaningless in Swift (ARC) or Kotlin (GC) |
| Rejecting kernel dispatch on a `'shared`/`'actor`/`'guard` instance | `checker/mod.rs:862-889` | The generated dispatch code doesn't know how to "unwrap" `Rc`/`Arc`/`RefCell`/`Mutex`/`RwLock` |
| `with`-block: double-acquire, GPU-resident opacity, `'atomic` incompatibility | `checker/mod.rs:1684-1742` | A GPU-residency model designed around Rust-side memory mapping |
| Kernel field `LabeledArray` shape + 3-axis cap | `checker/mod.rs:923-946` | A GPU hardware constraint, but entirely tied to the current GPU-only pipeline |

**Overall verdict for checker/mod.rs**: ~5 universal families, 1 universal-but-mislabeled, 7
Rust/GPU-specific. The central knot: **the qualifier system** (`'owned`/`'shared`/`'actor`/`'guard`/
`'static`/`'atomic`) is the source of almost every non-portable check — nearly each one is a
variation on "what operation does this particular *Rust* memory qualification unlock".

## 2. `src/validator/` — inventory

### General architecture (`validator/mod.rs`)

`validator/mod.rs:19-40` defines a generic `DiagLevel`/`KernelDiagnostic` type and a
`validate_kernel()` function that delegates to `kernel::KernelValidator`. **This wrapper is
itself a neutral, already-reusable pattern**: "a validation pass dedicated to one particular
backend/target, producing `{level, line, message}` diagnostics, called separately from the main
checker." This is an important structural difference from `checker/mod.rs`: where the checker
tries to be a single universal pass *contaminated* by Rust-specific rules, `validator/` is
*already* organized as a **per-target** extension, never conflated with the generic checker.

### `validator/kernel.rs` — nature of the module

Its header comment is explicit: *"rejects Boring constructs that are incompatible with
Rust-for-Linux (no FPU, no Rc, no panic, …)"* (`validator/kernel.rs:12-13`). This is **not** a
general Boring-semantics pass like `checker/mod.rs` — it's a validator of the **capabilities of
one specific compilation target**: `--target kernel` (a Rust-for-Linux, `no_std` module). In other
words, this entire module is *even more* specific than "Rust" broadly — it validates against the
limits of one particular Rust runtime environment (no FPU, no `Rc` heap, no `panic!`, no GPU
pipeline). None of these constraints concern a hypothetical "normal" (app/server) Swift/Kotlin
backend at all — they would only become relevant again if Swift/Kotlin ever targeted an
equivalent "kernel module" context, which isn't on the roadmap.

| Rule | Location | Classification |
|---|---|---|
| `float32`/`float64` types/literals rejected ("FPU is disabled") | `validator/kernel.rs:100-119`, `:185-187`, `:515-517` | **Specific to the `kernel` target** — an environment hardware constraint, not a fact about Rust as a language |
| Floating-point math functions rejected (`sqrt`, `sin`, …) | `validator/kernel.rs:37-42`, `:189-197`, `:239-246`, `:264-273` | Same reason — FPU disabled in this context |
| `panic(...)` rejected ("use throws/Result instead") | `validator/kernel.rs:198-204` | Specific to a kernel-module panic in Rust being catastrophic — a Rust mechanism (`panic!`), in a kernel context |
| `task` method on `self` requires `'shared`/`'actor`/`'guard` | `validator/kernel.rs:757-776` | Rust qualifiers (concurrency primitives available in `no_std`) |
| `'shared` → always `Arc<T>` in kernel context (no `Rc`) | `validator/kernel.rs:120-127` | **100% Rust** — the `Rc` vs `Arc` distinction doesn't exist in any other target language |
| `channel`/`stream` without an explicit capacity → warning (defaults to 2) | `validator/kernel.rs:206-215`, `:247-253`, `:749-755` | Arguably **universal in principle** (a stdlib API design warning, not a Rust fact), but currently scoped to the kernel target only |
| GPU kernel launch rejected in kernel context (no GPU in Rust-for-Linux) | `validator/kernel.rs:216-232`, `:454-485` | Specific to this target combination (this particular Rust backend has no GPU pipeline) |
| `with` forbidden inside kernel device code | `validator/kernel.rs:728-730` | Same family as the checker's GPU with-block — a Rust residency model |
| `kernel Name:` (GPU) entirely rejected under `--target kernel` | `validator/kernel.rs:966-983` | Universal in spirit ("no GPU on a target that has none") but phrased in terms of the current Rust target matrix |
| Assigning to an `'actor`/`'local` field from a host-side `init` | `validator/kernel.rs:987-1044` | Same family as the checker's GPU residency — Rust/GPU-specific |
| `LabeledArray` shape of a field | `validator/kernel.rs:150-167` | **Already universal by construction** — the code itself documents that it delegates to `labeled_array_shape_error`, described as "shared, target-agnostic" in its own comment |
| Recursion-depth guard | `validator/kernel.rs:49-58`, `:528-537` | Infrastructure, universal |

### Key structural observation

`validator/kernel.rs` is **not** a candidate to "universalize" the way part of `checker/mod.rs` is
— by nature it validates the limits of *one* exotic Rust compilation target. But its
**architectural pattern is exactly the right template** going forward: a separate, per-target
validator with its own diagnostic type, called alongside the generic checker rather than merged
into it. The day Swift or Kotlin get their own target-specific constraints (e.g. Kotlin/JVM having
no unsigned 128-bit integers, or Swift 6's strict-concurrency rules), the right answer is
presumably not to complicate `checker/mod.rs`, but to create `validator/swift.rs` /
`validator/kotlin.rs` on that same `DiagLevel`/`Diagnostic` model.

## 3. Summary and recommendations

1. ~~Two quite different families of debt exist today~~ **Done**: `checker/mod.rs` no longer
   mixes universal and Rust-ownership-specific rules in a single pass — the 11 Rust/GPU-specific
   checks now live in `checker/rust_checks.rs` (same `Checker` struct, same shared scope state, see
   the update note at the top of this document). `validator/kernel.rs` was already correctly
   isolated per target and needed no change — its shape is the template `rust_checks.rs` followed.

2. **Investment priority to maximize cross-backend value**: keep enriching the checker's universal
   family (types, arity, exhaustiveness, control flow, literal/shape consistency) — every check
   added there counts for Rust + a future Swift + a future Kotlin at no extra cost.
   ~~Dead-code/unreachable-statement detection~~, ~~missing-return-on-some-paths~~,
   ~~function-call arity/labeled-argument validation~~, and ~~`match` exhaustiveness (bool/optional)~~
   **all done** — see Updates 2/3/4 above (`check_dead_code`/`check_missing_return`/
   `check_call_arity`/`check_match_exhaustiveness`, all warnings). All four candidates from the
   original inventory are now shipped. `check_missing_return`'s optimistic "assume the arms as
   written are exhaustive" stance is now backed by a real (if narrower-than-full-enum) check for two
   of the most common non-exhaustive-match shapes.
   ~~Still open: real exhaustiveness over user-declared enum variants~~ **Done** — see Update 5
   above (`check_enum_match_exhaustiveness`, backed by the new `enums` registry). Every candidate
   from the original inventory, including this follow-on, is now shipped.

3. **The qualifier system remains the real underlying project.** Almost every Rust-specific check
   in both files is really a variation on "what operation does this particular Rust memory
   qualification unlock" (`'owned`, `'shared`, `'actor`, `'guard`, `'static`, `'atomic`). When
   Swift/Kotlin land, this won't be a set of checks to rewrite one by one, but a redefinition of
   what each qualifier *means* per backend:
   - `'owned` (exclusive Box) has no direct equivalent in Swift (ARC) or Kotlin (GC) — the
     "use-after-move" bug class simply doesn't exist in those memory models;
   - `'shared`/`'actor`/`'guard` will need reinterpreting (`class` + `@MainActor` in Swift, an
     object + `Mutex`/coroutine in Kotlin) rather than being mapped 1:1 onto
     `Rc`/`Arc`/`RefCell`/`Mutex`;
   - `'atomic` will need to follow each platform's real atomic primitives (different from
     `std::sync::atomic`).
   The checker will likely have to become **parameterized by backend** for this family of rules,
   while it can stay identical for the ~5-6 universal rules identified above.

4. ~~Low-cost action~~ **Done**: the `{mut T}` error message (`check_set_mut_constraint`, now
   `checker/mod.rs:656-673`) and `Type::contains_illegal_mut_set`'s doc comment (`src/ast/mod.rs`)
   no longer cite `HashSet`/`iter_mut`/`get_mut` — the universal rule (mutating a set element =
   possible hash-bucket-placement corruption) is now stated independently of Rust, with a "how
   today's Rust backend realizes this" note demoted to a secondary remark rather than the main
   justification.

5. **Don't conflate this with the "parse rustc errors" project** (discussed earlier, see the
   linked note): that project remains orthogonal and strictly Rust-only in value — it does nothing
   for Swift/Kotlin and becomes a residual safety net rather than a strategic investment, once the
   multi-backend goal is on the table.

## Open questions / to be settled later

- ~~Should there be an explicit tag (`CheckDomain::Universal` / `CheckDomain::RustBackend`, or a
  section annotation) on each `check_*` function in `checker/mod.rs`?~~ **Resolved, superseded**:
  the file split itself now carries that distinction — which file a check lives in *is* its domain
  tag, no separate enum/annotation needed. Revisit only if a single check ever needs to be
  universal-with-a-Rust-specific-carve-out (doesn't exist today; none of the 5 universal checks are
  like this).

- ~~Should the `validator/<target>.rs` pattern eventually absorb the main checker's Rust-ownership
  checks (i.e. `checker/rust_checks.rs` → `validator/rust.rs`)?~~ **Decided for now, not closed
  forever**: kept as `checker/rust_checks.rs` (shared `Checker` state, no independent pass) rather
  than a true `validator/rust.rs`, specifically to avoid duplicating scope/binding-tracking
  machinery for a second backend that doesn't exist yet (see the trade-off discussion in the
  conversation this draft comes from). Revisit this once a second backend (Swift or Kotlin) is
  concretely underway and its own capability validator is being designed — at that point, whether
  `rust_checks.rs`'s checks should become a real independent `validator/rust.rs` pass (so every
  backend, Rust included, activates only its own capability validator symmetrically) is worth
  reopening on its own merits, informed by how that second validator ends up being structured.

- **Still fully open, and the largest remaining piece of work**: what data structure/design
  replaces today's qualifier system (`'owned`/`'shared`/`'actor`/`'guard`/`'static`/`'atomic`) so
  it can be parameterized per backend, without duplicating all of the type-resolution logic three
  times over? This is where essentially all of the remaining Rust-specific reasoning lives (see §3
  item 3) and is a design project in its own right, not a mechanical refactor like the split done
  here — needs its own dedicated design pass once there's a concrete second backend (even a
  minimal/toy one) to design against, rather than being speculated on in the abstract now.
