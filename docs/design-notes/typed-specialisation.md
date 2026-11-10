# Typed specialisation, `.bri` files and the typing walk (design record)

Status (2026-11-11): **implemented**, apart from the two-phase compiler (open, below). This file started as the
draft of 2026-11-09 and is now the record of what was decided, built and measured; sections 1 to 8 are the original
proposal, kept as written, the later ones say what actually happened. Where each part is documented for users and
where it is tested:

| Part | User documentation | Tests |
|---|---|---|
| Typed specialisation of untyped parameters (free functions, methods, `ext`, `mod`, across files) | book, "Advanced — Typed specialisation of untyped parameters"; CHANGELOG | `tests/cases/implicit_generics_specialise*.br`, `tests/specialize_cross_file.rs`, `tests/implicit_generics_errors.rs` |
| `<T>` carries the qualifier of its argument (`<T'q>` removed) | book, "A type argument carries its qualifier" | `tests/cases/monomorphize_qualified_*.br` |
| `.bri` files (`mono`, `full`, instances, origin comments) | book, "Advanced — Boring inferred files"; CHANGELOG | `tests/bri_oracle.rs` (report), `tests/bri_cli.rs` |
| Typing walk / `TypeTable` (`BORING_TYPE_TABLE=replay\|walk\|check\|stats`), decisions pass | this file, sections 16 to 18 | `tests/type_table.rs` |

Open: the two-phase compiler (sections 11, 13, 16.7 to 16.9, 18): making the walk and the decisions pass the only
source of what the emitter reads, and removing its per-body tables. Measured and blocked by decisions the emitter
takes while emitting; no part of it is needed by the features above. Related: the book's "Untyped parameters",
"Untyped fields", "Qualifier parameters", "Turbofish monomorphization".

## Decisions (2026-11-09)

- The existing monomorphizer stays the single specialisation engine and is fed typed call
  sites; no second mechanism (open question 2).
- The qualifier table of 3.3 is confirmed, including the row where an argument whose
  qualifier is not written is accepted by a slot that writes one and resolved to it.
- `.bri` files are complete: a full Boring printer, not only the touched declarations
  (open question 8), so a `.bri` is always a valid, whole file.
- `mono` is built before `full` (phasing, step 6).
- **`<T'q>` is removed entirely** (decided 2026-11-09; Boring is not 1.0 and only the sibling
  projects of `perso/` use it). A plain `T` carries its own qualifier: `T = Counter'actor`
  binds both, and a qualifier written on a slot (`T'owned a`, `a'owned`) is a constraint, not
  a parameter. The linked-parameter sugar goes; what stays is the unlinked long form
  `<T, 'q>` where a qualifier parameter is genuinely wanted (a slot can then write `T'q`).
  Known loss to accept or fix: a field that follows a qualifier while another stays fixed
  (`let T'q res` next to `let T'inline default`) is written with `<T, 'q>`, whose type
  argument is not inferred (only from an expected type naming every argument).
- Qualifier parameters keep their tick syntax (`<'q>`, `elem'actor`, `<T, 'q>`). An uppercase
  form (`<T, Q>`, `let T'Q v`, `<T = Counter, Q = actor>`) was considered and rejected: it
  needs a declaration-side marker (`Q as qualifier`) to tell it from a type parameter, and a
  rename of about twenty test cases, the parser, `qual_infer.rs` and the docs.
- `mono` is taken right after the specialisation pass, not after later passes.
- `.bri` files go in `<stem>_rust[...]/bri/`, next to `src/` in the generated project
  (`boring build` writes `<stem>_rust`, `_rust_single`, `_rust_managed`... beside the
  source, checked in `src/main.rs`), not in Cargo's `target/` (a `cargo clean` would wipe
  them, and it is not Boring's directory). `boring check` has no project: stdout, or an
  explicit `--out <dir>`.

## 1. The problem

```boring
struct Counter:
    var int n = 0
    def inc(): n += 1

def bump(a):          # `a` is an implicit type parameter T_a
    a.inc()

let Counter'owned o = Counter()
let Counter'actor c = Counter()
bump(o)               # a.inc()               direct call
bump(c)               # a.lock().unwrap().inc()   lock taken
```

The body of `bump` is the same text, the emitted Rust is not. Rust generics cannot pick
"lock or not" from the type argument, and today the case never arises because a method
call on an untyped parameter is a Boring error (D5, "cannot use the untyped parameter `a`
with the method call `.inc()`"): there is no bound that could say `T_a` has `inc`.

The same wall stands behind three other open items: GPU code needs monomorphised
helpers, `--show-types` cannot show per-use instantiations, and an implicit field given
`Counter'actor` compiles only by accident (`T_item` absorbs `Arc<Mutex<Counter>>`).

## 2. What exists today (verified in the tree on 2026-11-09)

- `src/transpiler/monomorphize.rs` (2069 lines) is already a Boring-side pass: it clones a
  generic struct/fn/method into a concrete, non-generic declaration (`Pair_int_string`).
  It specialises **explicit turbofish sites only** (`identity<int>(42)`); every other use
  stays on the generic-Rust path where rustc infers the arguments.
- Method resolution there is **name-only**: a method name owned by more than one
  struct/ext/enum is ambiguous and falls back to generic Rust. The receiver type is never
  tracked.
- Qualifier parameters (`struct Board<'elem>`) have no generic-Rust form: every use is
  specialised (`Board_actor`), type-position uses are rewritten by `substitute_type` in a
  thread-local mode.
- `<T'q>` is implemented: a qualifier is inferred from the **written** qualifier of the
  arguments, an expected type, or an explicit argument; there is no inference by usage and
  no type-argument inference ("the type of an argument expression is not known at that
  stage").
- The book rejects hidden qualifier parameters on untyped fields (contagion, instance
  explosion).

So the Boring → Boring stage this note asks for exists in embryo. What it lacks is the
one thing the book names as its blocker: **expression typing**.

## 3. Proposal

### 3.1 Architecture

```
Boring with generics ──► [typed specialisation pass] ──► Boring, no generics except
                                                         Rust-compatible ones ──► Rust
```

- The pass runs after parsing and the existing inference passes, before the checker's
  emission-side work, and rewrites the AST. Output is ordinary Boring that the current
  emitter already handles (a concrete `Counter'actor` parameter emits the lock).
- "Rust-compatible generics" stay generic: a declaration whose body only needs trait
  bounds (Display, PartialEq, PartialOrd, same-param arithmetic + Copy, Clone) and does
  not depend on a qualifier. These are what implicit generics produce today, and the
  existing transpile cases keep passing unchanged.
- A declaration is **specialised** when, and only when, one of these holds:
  1. its body calls a method on a generic-typed value (`a.inc()`);
  2. its body reads a field of one (`a.n`);
  3. an instantiation carries a qualifier that changes emission of a use in the body
     (`'actor`/`'guard` lock, `'owned`/`'shared` indirection) — see 3.3;
  4. device code (GPU targets) instantiates it.

### 3.2 Expression typing

The pass needs the concrete type (and qualifier) of each argument at each call site:
literals, locals with an annotation or a typed initialiser, calls to functions with a
declared return type, field reads, constructor calls. Same list as the book gives for
why `Cache(5)` cannot infer `T`.

Scope control: a call site whose argument type the pass cannot determine does **not**
specialise; it keeps the generic path, and if the declaration needs specialisation
(3.1.1–3.1.3) it is a Boring error naming the argument and suggesting a written type
(`bump<T_a = Counter'actor>(x)` or an annotation on `x`). Failing loudly is the rule; no
silent fallback to wrong code.

This is deliberately a typing of the *call-site arguments*, not a whole-program type
inference. How far it must go is the main open question (section 5).

### 3.3 Qualifiers on a plain `T` (user's design, 2026-11-09)

A plain implicit `T` accepts a qualified type. A slot that **writes** a qualifier
constrains it:

| Slot                | `Counter'actor` | `Counter'owned` | `Counter` (inferred qualifier) |
|---------------------|-----------------|-----------------|--------------------------------|
| `let T a` / untyped | accepted        | accepted        | accepted                       |
| `let T'owned a`     | **error**       | accepted        | accepted, qualifier resolved to `'owned` |
| `let T'actor a`     | accepted        | **error**       | accepted, resolved to `'actor` |

The error is a Boring (transpilation) error naming the slot, the written qualifier and
the argument's qualifier. This needs no `<T'q>` on the declaration: it is the same
specialisation, with the qualifier as part of the instance key. `<T'q>` stays the way to
*link* a field qualifier to a parameter explicitly when a declaration has several slots.

An unwritten qualifier is no evidence (existing rule): `let p = Point(1)` carries no
qualifier to propagate.

### 3.4 Instance key and naming

Key: `(declaration, [concrete type + qualifier per implicit parameter])`. Mangling extends
the existing scheme: `bump_Counter_actor`, `bump_Counter_owned`. Two arguments of one
type but different qualifiers are two instances (3.3 requires it).

Instance count is bounded by the distinct keys actually used; a cap with a clear error
(not a silent explosion) is worth adding.

### 3.5 Errors keep their origin

A diagnostic raised inside a specialised copy must point at the **source** line of the
generic declaration and say which instantiation triggered it (`in bump, instantiated with
a = Counter'actor at main.br:12`). Without this the pass makes errors worse, so it is a
requirement of the pass, not a nicety.

### 3.6 `.bri` — "Boring inferred" files

Today the specialised program exists only as AST nodes in memory. Proposal: an opt-in
output of the Boring intermediate as real files, extension `.bri`, so a developer can
read what the compiler decided and debug rustc errors against it.

- **Content.** One `.bri` per source `.br`; each specialised copy follows the original,
  with an origin comment (`# bump_Counter_actor: bump at main.br:3, instantiated at
  main.br:12`). Valid Boring, no generics except the Rust-compatible ones (3.1).
- **Two levels**, chosen by a flag (`--emit-bri=mono` / `--emit-bri=full`):
  - `mono`: AST right after the specialisation pass, nothing else added.
  - `full`: additionally every inferred qualifier written out (`let Point p` becomes
    `let Point'inline p`, a bare field becomes `'actor` or `'owned`), i.e. the output of
    `src/qual_infer.rs` made visible (today only a list via `--show-qualifiers`).
- **Opt-in**, like `--emit-rust`; never written by a plain `boring run`.
- **Placement.** Decided: `<stem>_rust[...]/bri/` inside the generated project (see
  Decisions); never beside the sources; stdout or `--out <dir>` for `boring check`. Not
  checked: whether the generated project directory is already git-ignored by the
  generated `.gitignore`.
- **Test oracle.** The `.bri` is parsable: `boring run x.bri` must print what `boring run
  x.br` prints, and `boring build` of both must give the same Rust. Run over the
  existing transpile cases this checks the pass with almost no new test code, and makes
  a faithful printer mandatory (an unfaithful one breaks the round-trip at once).
- **Caveat on `full`: the round-trip is not guaranteed identical.** The book says an
  *inferred* qualifier is not evidence for `<T'q>` (`let p = Point(1)` commits to
  nothing), while a *written* one is (`let Point'inline p = ...`). Reading a `full` file
  back turns inferred qualifiers into written evidence, so an instantiation that was
  free can become constrained or conflicting. A round-trip mismatch on `full` is
  therefore a possible legitimate difference, not necessarily a bug; the size of the
  effect is unmeasured and must be measured on the existing suite before `full` is
  promised as an oracle. `mono` carries no such caveat and is the oracle.
- **Pass order for `full`.** Qualifier inference depends on usage across the whole
  program (several owners give `'actor`). The specialisation pass must run first,
  because the `'actor` copy and the `'owned` copy of one function have different usage;
  `full` is printed after qualifier inference. How much emission-time qualifier
  decision the Rust emitter makes outside `qual_infer.rs` was not checked, so how much
  simpler the emitter gets is unknown.
- **Replacing the `--show-*` options.** The `.bri` makes some of them redundant, but only
  if the printer covers what they show:
  - `boring check --show-types` (inferred headers of untyped parameters/fields): replaced
    if the printer writes the inferred generic headers with their bounds
    (`<T_a as Display>`). Not automatic, since Rust-compatible generics stay generic in
    the `.bri`; the printer must be told to write them out. Recommended.
  - `--show-qualifiers`: also lists `'kernel'global`/`'kernel'unified` for
    `kernel.tensors` operands, not only CPU ownership. Replaced only if the printer
    handles kernels (open question 8).
  - Trajectory: build the `.bri` (`mono` first); check every piece of information of
    each option appears in it; then deprecate the option (warning + CHANGELOG entry);
    remove at the next version. No one-step removal (breaking CLI change). Whether
    `--show-types` already shipped in a release was not checked; if not, it can be
    removed as soon as the `.bri` replaces it.
- **Which AST for `mono`.** Right after the pass (readable) rather than after every later
  pass (closer to what goes to Rust). Chosen: right after the pass.

## 4. Consequences for what is already built

- **D5** (`unbounded_use`): a method call or field read on an untyped parameter stops
  being an error when every call site types the argument; it becomes "specialise". It
  stays an error where typing fails (3.2). This is a behaviour change to a documented
  error: the book sections "What an untyped parameter or field cannot do" must be
  rewritten, and `implicit_generics_errors` cases move from error to success.
- **`--show-types`** (D7): the per-use instantiation view needs the specialised Boring
  printed, which needs an AST printer. None exists in `src/` today (checked 2026-11-09:
  `--emit-rust`, `--show-types`, `--show-qualifiers` are the only views, none shows an
  intermediate Boring). That printer is a real cost; section 3.6 makes it the `.bri`
  output.
- **GPU (D6):** `implicit_struct_in_device_code` stops being a blanket error for structs
  whose instantiation is typed at the kernel's call site. Device generics already
  monomorphise by explicit instantiation; this removes the "explicit" requirement where
  typing succeeds.
- **`boring run`:** ignores qualifiers today. It should run the original (un-specialised)
  program: the interpreter has no static types to specialise on, and method dispatch is
  dynamic there. Behaviour is identical by construction as long as 3.3's errors are
  raised in a pass both commands run (the checker side), not only in `build`.

## 5. Open questions

1. **How much expression typing?** Minimum viable: annotated locals, parameters, literals,
   constructor calls, calls with a declared return type. Anything else → loud error. Does
   a reusable typing helper already exist in the checker (it types expressions for the
   `mut`/collection rules, "positively known to be a built-in collection")? If so, reuse
   it; if it is purely syntactic, this is the largest piece of work.
2. **Where in the pipeline?** The existing monomorphizer runs in the transpiler
   (`emit_top.rs` pre-scan). Running the typed pass earlier (in `parser::parse` order, or
   as a separate module) would let the checker and `--show-types` see its output; running
   it in the transpiler keeps one mechanism. **Decided:** keep the monomorphizer as the one
   specialisation engine and feed it typed call sites, rather than writing a second one.
3. **Methods, not only free functions.** Name-only method resolution is the existing
   limit. A typed receiver would lift it, but ambiguity fallback must stay for untyped
   receivers.
4. **Struct fields.** `struct Response: payload` then `r.payload.inc()` inside a method
   of `Response`: the instance key is the struct's instantiation; specialising a struct
   instantiates all its methods. Interaction with D8 self-sharing (`share_self_params`).
5. **Recursion and mutual recursion** (polymorphic recursion termination), overloads
   (per-overload return inference already exists), `mod` scopes, imported declarations.
6. **Bounds that fail on lock handles.** `Display` on `Arc<Mutex<Counter>>` does not
   exist. With specialisation the body is retyped, so `"{a}"` on an `'actor` value either
   works (auto lock) or is a clear error on the instance; generic-path bounds must not
   promise more than the lock handle can do.
7. **Interaction with `<T'q>` defaults and `'new`** (candidate-set qualifier): is `'new`
   resolved before or after instance keys are formed?
8. **`.bri` printer scope** (3.6): **decided: a full Boring printer.** It must cover every
   construct the pass can produce, including sugar already desugared by earlier passes
   (trailing array blocks, labelled arguments resolved to positional, `dispatch`). The
   cheaper alternative (print only touched declarations, copy the rest from source) was
   rejected because it breaks the "valid, complete file" property the round-trip oracle
   relies on.

## 6. Phasing

1. Spike on one function: `def bump(a): a.inc()` called with `'owned` then `'actor`
   arguments with annotated locals. Goal: prove the key + retyped body gives the right
   Rust, and measure how much typing the call sites need. Include a throwaway `mono`
   printer for this function: debugging the spike without it means reading mangled Rust.
2. Free functions with methods/fields on untyped parameters, annotated arguments only;
   lift D5 for exactly that case; errors keep origin (3.5).
3. 3.3's qualifier table on plain `T` slots (written qualifier on a slot; mismatch error).
4. Untyped struct fields with methods (open question 4).
5. GPU (D6) and `--show-types` per-use output.
6. `.bri` output (3.6): `mono` printer and round-trip test first; `full` only after the
   round-trip mismatch on the existing suite has been measured.

Each phase keeps generic Rust for everything it does not specialise, so the existing
transpile suite is the regression net.

## 7. Tests (sketch)

- Transpile cases (all four configs): `bump` with `'owned`/`'actor`/`'shared`; two
  qualifiers in one program; recursion; mod-scoped.
- Error cases: untyped argument at a specialisation-required site; `T'owned` slot given
  `Counter'actor`; instance cap; error origin text.
- `boring run` parity on every success case.
- `.bri` round-trip over the transpile cases: `mono` output must parse and give the same
  `boring run` output and the same generated Rust; `full` mismatches are listed and
  triaged (3.6 caveat).
- Self-hosted interpreter suites (the pass could touch `boring/interpreter/*.br`).

## 8. Files likely touched

`src/transpiler/monomorphize.rs` (engine, instance keys), `src/transpiler/emit_top.rs`
(pre-scan / candidates), `src/infer_param_generics.rs` (D5 becomes conditional),
`src/show_types.rs`, `src/qual_infer.rs` (`full` output), a new AST printer for `.bri`,
`src/main.rs` (`--emit-bri`), `src/desugar_gpu_struct.rs`, the checker (typing helper, 3.3
errors),
`docs/book.md`, `CHANGELOG.md`, `tests/cases/`, `tests/transpile.rs`.

## 9. Spike result (phase 1, 2026-11-09)

Implemented in `src/specialize_untyped.rs` (a parser pass, after `desugar_struct_params`),
with the D5 error relaxed in `parse_fn.rs` for a method call on an untyped parameter of a free
function. Tests: `tests/cases/implicit_generics_specialise.br` (4 configs) and three cases in
`tests/implicit_generics_errors.rs`.

What it proves:
- `def bump(var mut a): a.inc()` called with `Counter'owned` and `Counter'actor` produces
  `bump_Counter_owned` / `bump_Counter_actor`. The emitted Rust is identical to the two
  hand-written typed versions: `(*a).inc();` and `a.lock().unwrap().inc();`. rustc accepts
  it and the program prints the right result. `boring run` and `boring build` agree.
- The substitution engine is the monomorphizer's own (`substitute_fn_decl`); nothing new was
  needed there.
- The typing needed by this case is small: a call argument that is a plain variable whose
  `let` or parameter type is written. Anything else is a loud error naming the argument
  (`bump(Counter())` is rejected, an uncalled `bump` keeps the original D5 error).

What it taught:
- **Propagation is required.** `def twice(var mut b): bump(b); bump(b)` has no method call
  of its own, yet cannot stay generic: it passes `b` to a function that needs its type. The
  set of functions to specialise is a fixpoint (same shape as the bound propagation), done
  in `passes_param_to`. Without it the chain fails with "type not known".
- **The permission of the argument's binding is not part of the parameter's type.** Copying
  the `mut` wrapper from `var mut Counter'actor c` onto the parameter emitted
  `&mut Arc<Mutex<..>>` and failed in rustc; the parameter keeps the permission it was
  declared with, so the instance types are `into_without_mut()`.
- **`describe_type` is not a `.bri` printer.** It prints a qualified type as `Counter'...`
  (qualifier elided). The `.bri` printer must write qualifiers in full.
- **`boring run` runs the specialised program**, not the original, because D5 is a parse-time
  error today. Section 4 of this note said `run` should keep the original; that needs D5 to
  move out of the parser first.
- A pre-existing, unrelated bug was found: in managed mode, passing a `'owned` `var mut`
  value to a `var mut` parameter clones it (`bump(&mut RefCell::new(o.borrow().clone()))`), so
  the mutation is lost. The hand-typed version prints `0` too. Proposed as its own task.

Phase 2 additions (same day): field reads/writes and indexes on an untyped parameter are
specialised like method calls (`takes_over` in `src/specialize_untyped.rs`; `a + b` of two
untyped parameters stays an error, only uses whose meaning depends on the argument's type are
taken over); labelled arguments give the type of the parameter they name; a name bound more
than once in a function with different or unwritten types (a shadowing `let`, an unannotated
rebinding, a `for` variable) is *ambiguous* and an error when used as an argument, instead of
the flat environment picking one type silently. Tests: `tests/cases/implicit_generics_specialise_fields.br`
(4 configs) and three more CLI cases.

Phase 3 (same day): the qualifier slot of 3.3, in the two syntaxes decided ("both"):
- `def bump(var mut a'owned)`: a qualifier written after an untyped parameter's name (parser:
  `slot_qual_ok`, kept on `FnDecl.slot_quals`, the parameter stays untyped). Such a function is
  always taken over by the pass; a call whose argument is a variable written with another
  qualifier is an error naming both (`the parameter a'owned of bump expects that qualifier,
  but this argument is written 'actor`). An uncalled one is kept as is.
- `def show<T>(T'owned a)`: a qualifier on an explicit generic. Nothing is specialised, the
  call stays on the generic path; only the mismatch of a written qualifier is reported.
- The qualifier compared is the one written directly on the base type (`Counter'actor'weak` is
  `'actor`). An argument whose qualifier is not written is not an error, and no qualifier is
  resolved onto it: passing a bare `Counter d` to an `'actor` slot wraps a temporary exactly as
  it does for a hand-typed `Counter'actor` parameter (`d` is moved), so the table row of 3.3
  that said "accepted, resolved to the slot's qualifier" is, in practice, "accepted, with the
  existing coercion".
Tests: `tests/cases/implicit_generics_specialise_slot.br` (4 configs) and three CLI cases.

Remaining limits (not design decisions): same file only until 14.2 (a `pub def` that other files call is
then specialised by the callers' files), top-level free functions and struct/enum methods of a unique name (14: no `ext`, trait or
kernel method, no `task`/`stream`; `mod` items included, since 14.1), all untyped parameters of a function are specialised together,
`unbounded_use` reports only the first unbounded use (a later `a + b` on two untyped
parameters is left to rustc), and the environment is per function, not per scope (so an
inner-scope shadow makes the name ambiguous rather than resolved).

Not yet decided from the spike: the specialisation currently lives in the parser rather than
in the transpiler's monomorphizer (open question 2 decided the monomorphizer as the engine;
moving the pass there needs the typed call sites to survive until the transpiler).

## 10. What replacing `<T'q>` by `<T>` needs (found 2026-11-09, before the removal)

`struct Entry<T>: var mut T value` already accepts a qualified argument at the Rust type level:
`Entry<Counter'actor>(c)` and the bare `Entry(c)` (with `c` written `Counter'actor`) both emit
`Entry<Arc<Mutex<Counter>>>`. But `e.value.inc()` then fails in rustc (E0599: no method `inc` on
`Arc<Mutex<Counter>>`): a generic body cannot take the lock. Same problem as `bump(a)`, same
answer: the struct must be specialised with `T = Counter'actor`.

Why it is not specialised today: the turbofish monomorphizer bails when a type argument does not
mangle, and `mangle_type` (`src/transpiler/monomorphize.rs`) has no case for `Type::Qualified`,
so `Entry<Counter'actor>` stays generic Rust. The qualifier-parameter machinery (`Board_actor`)
is what `<T'q>` used to reach specialisation.

So the removal of `<T'q>` is not a deletion, it is a migration of its capability onto `<T>`:
1. `mangle_type` / instantiation accept a qualified type argument (`Entry_Counter_actor`), so the
   explicit `Entry<Counter'actor>(...)` is specialised like any other concrete argument.
2. A bare construction or call (`Entry(c)`, `show(c)` with an explicit `<T>`) infers `T` from a
   variable whose type is written, the typed call-site mechanism of `specialize_untyped.rs`
   extended from free functions with untyped parameters to generic structs and generic functions.
3. Then `<T'q>` (parser, `generic_args.rs`, `qual_infer.rs`, tests, book) is deleted, and
   `tests/cases/qualified_generics.br` is rewritten on `<T>`.

## 11. The `.bri` as a real stage: informative, then transpilable, then a two-phase compiler

Raised 2026-11-09 (user): the `.bri` can be an intermediate step that splits the transpilation in
two phases and makes both clearer. Three levels of ambition, built in this order:

1. **Informative and oracle** (3.6): written on request (`--emit-bri`), and `boring run x.br`
   and `boring run x.bri` give the same output over `tests/cases/`.
2. **`boring build x.bri` transpiles it.** A `.bri` is valid Boring, so the compiler can read it;
   the check is that the Rust generated from `x.br` and from `x.bri` is identical. Cheap once the
   printer exists.
3. **Two real phases.** Phase 1 (read, infer, specialise) produces an explicit program; phase 2
   (Rust generation) consumes it and infers nothing. The boundary is the **in-memory AST after
   phase 1**, and the `.bri` is its text form: a normal `boring build` neither writes nor rereads
   a file, and `boring build x.bri` stays available as the level-2 entry point.

Why the boundary is the AST and not a file read back every time: rereading runs the inference
passes again on an already explicit program, and the book says a *written* qualifier is evidence
while an *inferred* one is not, so a `full` file read back can change decisions (see 3.6, the
round-trip caveat). Passing the AST avoids it.

Open, unchecked, and the main risk of level 3: the transpiler still makes inference decisions of
its own (`src/transpiler/infer_qualifiers.rs`, `infer_shared_elems.rs`, `share_dyn_traits.rs`, the
lock and clone passes). For a real split they must move to phase 1, otherwise the `.bri` does not
show what the Rust generator will decide. How much of that is separable is unknown; each move is
validated by the same test, identical generated Rust before and after. The `full` level of 3.6
(inferred qualifiers written out) is the first step in that direction.

## 12. The `.bri` output, as built (2026-11-09)

Implemented: `src/bri.rs` (printer, items and types), `src/bri/stmt.rs`, `src/bri/expr.rs`,
`src/bri/full.rs` (`full` level), `src/bri/tests.rs`; `--emit-bri[=mono|full]` on `boring check`
(stdout, or `--out <dir>`) and on `boring build` (`<stem>_rust[...]/bri/<file>.bri`, for the entry file
and the sibling files it reaches through a bare `use <name>`; it does not combine with `--emit-rust`,
`--linux` or `--target`). Tests: `tests/bri_oracle.rs` (the oracle), `tests/bri_cli.rs` (the flags).

### What `mono` is (a deviation from the decision above)

`mono` is the AST as `parser::parse` leaves it, printed before the later desugarings
(`desugar_inline_kernel`, `desugar_labeled_array`, `desugar_array_block`, `desugar_inject`), which a
reader of the `.bri` runs again. That includes `specialize_untyped` (it runs inside `parse`), so the
specialised copies of an untyped-parameter function (`bump_Counter_actor`) are in it. It does **not**
include the transpiler's monomorphizer (`transpiler/monomorphize.rs`: explicit `<..>` instances,
qualifier-generic instances). It was tried (`Transpiler::global_pre_scan` + `monomorphize_program` on a
clone, as `specialise_qualifier_params` does) and measured on the corpus, then removed: of 609 cases it
broke 14, for four reasons that are the real cost of including it:
1. It runs on the *desugared* AST (it panics on `labeled_array`, whose `[v for w in .. for h in ..]` is
   still sugar after `parse`), and the desugarings produce `Block` expressions that have no source
   spelling, so printing after them is not readable.
2. An instance of an imported generic lives in the file that *declares* the generic (`Stack_int` goes
   into the stdlib module, `Wrapper_string` into the other file), so the `.bri` of the calling file
   calls an instance it does not declare: `boring run x.bri` fails with `undefined variable 'Stack_int'`.
   A per-file `.bri` is not self-contained once monomorphizer output is in it.
3. The substituted copies carry `mut` onto a scalar (`mut int value`), which the checker rejects when the
   `.bri` is read back, and a `type let` default that depends on the struct's own parameter.
4. It needs the project's `[deps]`/sibling files at print time (global, cross-file state).
Putting it into `mono` therefore needs the two-phase compiler of section 11 level 3 (or an explicit
"instances of imported generics go into the importing `.bri`" rule), not a printer change.

### The printer

- **Complete**: every construct the parser produces has an arm (items fn/struct/enum/trait/ext/mod/use/
  alias/kernel/view; every statement and expression; patterns; types with qualifier chains, borrows,
  `mut`, labelled arrays, function types). The three it cannot write are explicit errors: a
  `SliceRange` outside an index, a `Block` expression outside `else`/`task`/`kernel`, and a qualifier
  argument whose declaration is in another file (`Board<elem'actor>` needs the label `elem`, read from the
  declaration of the same file). Those never occur in the corpus.
- **Implicit generics are written back untyped.** After `parse` an untyped `def show(x)` is
  `def show<T_x as Display>(T_x x)`, and bounds such as `PartialOrd<isize>`,
  `std::ops::Add<f64, Output = T_x>` are Rust spellings the `as` bound of Boring cannot read. The printer
  writes the parameter untyped again (`def show(x):  # inferred: T_x as Display`), the same for untyped
  fields and for a bare struct parameter (`describe(Response r)`, expanded by `desugar_struct_params`),
  and the reader re-derives the same generics. The bounds are a *trailing* comment on the header: a
  full-line comment is kept by the parser as a statement and reaches the generated Rust. This is the one
  place where the `.bri` shows less than the AST; `--show-types` is therefore not replaced yet.
- Quirks of the parser the printer has to undo (each found by the oracle): `pass` and `sync` are parsed to
  `Stmt::Comment` markers; a body that is empty is `def f(): pass` on the header line, a block with `pass`
  is a statement; `def Worker.run()` must say `void` (`def void Worker.run()`), or `Worker` is read as the
  return type; a statement that starts with `if`/`match` is the statement form, so an `ExprKind::If`
  statement (the `f?(x)` sugar) is parenthesised; `.thread`/`.tensors` are `gpu.<name>` in the AST;
  `join` takes `(`..`)`; a cast is parenthesised as a receiver and as the left operand of `<`, `>`, `<<`,
  `>>` (`x as T.f()` and `x as T < y` read the type differently); `x = x + 1` is folded back to `x += 1`;
  `let x'q = Ctor()` keeps the qualifier on the name when the parser left the `_` placeholder base.
- Expressions that extend to the right (`if`, `match`, closures, `do`, `task`...) are parenthesised as the
  left operand of an operator, an `else` or a pipe, and not as an argument; a multi-line argument puts
  the next `,` / the closing `)` on its own line at the statement's indentation.

### Oracle results (`tests/cases`, 609 `.br`, 447 with an `.expected`)

| level | check | result |
|---|---|---|
| `mono` | `boring run x.bri` == `boring run x.br` | 447 / 447 comparable (3 more fail alone and are skipped) |
| `mono` | `boring build --emit-rust x.bri` == `x.br` | 519 / 519 that emit (90 do not emit at all) |
| `full` | run | 432 identical, 15 different (8 print/reparse failures on GPU-kernel programs and a few more, 7 reparse), 3 skipped |
| `full` | build | 401 identical, 118 different, 90 skipped (about 22%) |

The build identity needs one normalisation: `__stream_<n>` takes the source line of the `for`, which a
reprint changes (`tests/bri_oracle.rs` normalises it). One case (`supertraits`) emits in a hash-map order
that differs between runs of the *source* alone; the test compares the sets of outputs of several runs of each side.

**`full` is not an oracle** (the caveat of 3.6, measured). The mismatches are one family: a written
qualifier is an explicit constraint, so the reader emits differently, not wrongly:
a written `'inline` on a scalar adds a Rust annotation (`let mut x: isize = 0`, the largest group, 13+
cases) or turns `let x = 0` into `let x: Arc<isize>`; a parameter written `Counter&` loses the inferred
`mut x: &mut X` binding of the same parameter; a by-value read becomes `.clone()`; `Box<dyn T>` becomes a
plain struct. Behaviour (`boring run`) differs in 15 of 447, the failures being a binding whose written
qualifier makes a later reassignment or a `def` call a type error (`mut _'inline`, `declare it mut
Game_inline g3`). `full` is useful to read (the inferred choices, in source order) and is kept opt-in; it
should not be promised as a round trip. Not written by `full`: the GPU residency decisions
(`desugar_tensor_quals`: `'kernel'global`/`'kernel'unified` on `kernel.tensors` operands, which are
operands in expressions, not declarations), a `let` whose initialiser is not a constructor call or a
scalar literal (the printer cannot give it a base type, so `let p = [1, 2]` is left unqualified), and a
`kernel` program (`full` runs the std transpile to collect decisions, which rejects a kernel without a GPU
target).

### Remaining (not started, as asked)

- The two-phase compiler of section 11 level 3 (inference moved out of the transpiler: `infer_qualifiers`,
  `infer_shared_elems`, `share_dyn_traits`, the lock and clone passes). `full` shows only the
  `infer_qualifiers` part, collected by running the transpile and read back by position.
- The transpiler's monomorphizer in `mono` (see above) and its rule for instances of imported generics.
- `--show-types` / `--show-qualifiers` are not deprecated: `--show-types` prints the bounds the `.bri`
  keeps as a comment; `--show-qualifiers` lists GPU residency, which `full` does not write.
- `.bri` for `--target`/`--linux` builds, and a `.bri` per imported file that is not a bare sibling `use`.
- Origin comments of 3.6 (`# bump_Counter_actor: bump at main.br:3, instantiated at main.br:12`) on the
  specialised copies: the copies exist in `mono` but carry no comment yet.
- Idempotence is not guaranteed everywhere (`trait X: pass` followed by a top-level `pass` gains a line
  per reprint, `trait_scenarios_4`); the corpus has one such case.

### Decision on the transpiler's monomorphizer in `mono` (2026-11-09)

`mono` stays without `src/transpiler/monomorphize.rs`: it is the AST right after `parser::parse`, so it
includes the specialisation of `src/specialize_untyped.rs` but not the instances the transpiler's
monomorphizer makes (`Entry<Counter'actor>` appears as written, generic). It is added later, not now.
How, in the order of preference:

1. **With the two-phase compiler** (section 11 level 3). The whole program's AST exists after phase 1,
   the monomorphizer runs on the desugared AST, and the instances of an imported generic are written
   where they are used. The `.bri` is faithful by construction, with no placement rule to invent.
2. **Before that**, a rule that puts the instances of imported generics into the `.bri` of the file that
   uses them. Possible, but it makes a `.bri` that is not the image of one source file.

Three obstacles found by the first attempt (it broke 14 of 609 cases): the monomorphizer needs the
desugared AST (it panics on `labeled_array`), the instances of an imported generic live in the declaring
file (a per-file `.bri` would call `Stack_int` without declaring it), and it substitutes `mut` onto
scalars. The last one is a defect of the monomorphizer itself, independent of the `.bri`, to fix on its
own.

## 13. Sounding of level 3 (the two-phase compiler), 2026-11-09

Measured by reading the tree, nothing built. Level 3 of section 11 asks that phase 1 produce an explicit
program and phase 2 infer nothing. What the transpiler holds today:

- `Transpiler` has **285 fields**, about **48** of them decision state filled while emitting
  (`inferred_qualifiers`, `promoted_atomic_vars`, `var_lock_scalar`, `trait_borrow_resolved`,
  `trait_borrowed_methods`, `borrowed_def_methods`, `last_use_sites`, `shared_dyn_traits`,
  `fn_sigs`, `cpu_decision_notes`, ...). The decisions are read again by the emitters.
- The emitters that consume them are the bulk of the transpiler: `emit_methods.rs` (8157 lines),
  `emit_expr.rs` (6810), `emit_top.rs` (4354), `emit_let.rs` (3810), `emit_struct.rs` (3737). Several
  decisions are not passes at all but are taken at the point of emission (a clone before a by-value
  use, a lock at a call whose receiver is `'actor`, `box_if_trait_typed`).
- The passes that are separable as such are small: `infer_qualifiers.rs` (2707, the largest),
  `infer_trait_borrow.rs` (805), `last_use.rs` (460), `promote_atomic.rs` (411), `share_dyn_traits.rs`
  (295), `copied_field_literals.rs` (282), `infer_field_scan.rs` (246), plus `infer_shared_elems.rs` and
  `infer_field_cross_file.rs`. About 6000 lines.

Conclusion: level 3 as written (all inference out of the transpiler, the emitter reading an explicit
AST) is not a matter of moving a few passes: most decisions are interleaved with emission. It is a
rewrite-sized refactor, and what it buys (a `.bri` that is faithful by construction) is partly
obtained already: `full` reads back the decisions the transpiler recorded
(`take_cpu_decisions`), and `mono` shows what the front end specialised.

A cheaper route, in steps that each keep the generated Rust identical (the same test as the oracle):

1. Move the small passes listed above to the front end one at a time, each writing its decision into
   the AST (an explicit qualifier, an explicit borrow) instead of a `Transpiler` field, and checking
   that the emitter then reads the AST. Validation per step: the full transpile suite, byte-identical
   Rust.
2. Extend `full` to print each decision that was moved, so the `.bri` shows it without a side channel.
3. Only then reconsider the monomorphizer in `mono` (section 12): once the AST is explicit, the
   instance of an imported generic can be written where it is used.

Not started. Cross-file specialisation of an untyped-parameter function (`pub def bump(a)` in a library,
called from another file) has the same placement question as the monomorphizer's imported instances,
and waits with them: today such a function is refused with the "write its type" error in its
declaring file.

### 13.1 First step of the stepwise route, tried on paper (2026-11-09): no cheap first step

The route above starts with "the smallest pass". Reading the three smallest candidates shows that none
is independent, so the line counts understate the cost:

- `copied_field_literals.rs` (282) is not a decision but a **diagnostic** ("a struct field copied into a
  literal whose elements need a shared handle"). It reads `fn_sigs`, `struct_fields`,
  `resolve_expr_struct_type`, `body_local_types`, `method_receiver_type`, the overload tables and
  `emit_type`. Moving it means porting those tables to the front end or the checker, not moving a pass.
- `share_dyn_traits.rs` (295) is already an AST rewrite (`Trait` becomes `Trait'shared`), the easiest
  kind to make explicit, but it is interleaved with borrow inference (`BorrowView`, a by-value
  parameter becomes `'shared` only once `infer_trait_borrow` has decided it is not borrowed) and its
  facts come from a scan of the whole reachable file graph (`dyn_fact_pass`).
- `promote_atomic.rs` (411) runs **after** `infer_qualifiers` has committed a binding to `'actor` or
  `'guard`: it consumes the output of the 2707-line root pass.

So the dependency order is `infer_qualifiers` first, then the passes that read its decisions, and the
borrow and shared-trait decisions are entangled with each other. The honest size of level 3 is the size
of the qualifier inference and what hangs off it, not the sum of the small files. A step that stays
cheap and useful is to extend `full` with the decisions the transpiler already records or computes
(for example export `shared_dyn_traits` and the borrow view the way the CPU decisions are exported)
and print them, which shows more of the program without moving any inference.

### 13.2 `full` extended with the shared traits (2026-11-10)

Done instead of moving a pass (13.1): the transpiler's decision about shared traits is exported
(`take_shared_decision`: the shared traits and the outcome of borrow inference) and `full` applies the
same rewrite (`apply_shared_decision`), so a field, `let` or return naming a shared trait is written
`Trait'shared` and a parameter borrow inference left borrowed stays bare.

What it taught:
- **Applying only the traits is wrong.** Writing `'shared` on fields and returns but leaving the
  parameters bare changed the generated Rust for 24 of 43 cases: a by-value parameter of a shared trait is
  `Arc<dyn Trait>` in the source, but read back next to explicit `'shared` positions it was decided again
  and came out `Box<dyn Trait>`. The borrow outcome has to travel with the traits.
- **A written qualifier removes the evidence that made the decision.** A trait is shared because some
  struct field names it *bare*; once the `.bri` writes `'shared` there, a bare use that a desugaring
  creates later (the trailing array-block builder `Vec<Widget>`) is no longer rewritten. The sugar now keeps
  the callee's element type (`Widget'shared`) for its builder, and accepts `[Trait'shared]` as a
  trait-object array.
- **Result over `tests/cases`:** `full` run 432 of 447 identical (unchanged), `full` build 403 of 519
  identical (401 before): the 40 cases the shared rewrite changes produce byte-identical Rust.
This is the pattern for every decision that moves into the `.bri`: export what the transpiler decided,
write it, and check that the reader, which now takes it as written, decides the rest the same way.

### 13.3 `full` writes decisions, not defaults (2026-11-10)

The oracle's failures showed that a qualifier written by hand is a constraint: `inline` on a local or enum parameter, `&` on a parameter and `shared` on a captured scalar all changed the Rust or broke the file. `full` now leaves out what is the default (`inline` locals and parameters, parameter borrows, scalar `shared`). Result: build 519/520 identical (was 403), run 447/447 (was 432). Also left out: `'actor` on a `mut` binding and `'owned` on a local (usage-lifted locks and boxed oversized values change how the value is lent when written). The one case left, `shared_dyn_tree_copy`, only loses a redundant `as Arc<dyn Widget>` cast the typed `.bri` makes unnecessary. The oracle blanks compiler-made names that carry source positions (`__comp_4_9`, `__a_axis0_4_1`).

### 13.4 `full` writes GPU residency (2026-11-10)

`desugar_tensor_quals` decisions (scope, name, position, `'global`/`'unified`) are taken right after the inline-kernel desugaring, before the check drains them, and applied to the parameter or `let` at that position, top-level statements included. Checked over `tests/cases` with `--target metal`: 15/16 identical Metal source (the other differs by `__inline_kernel_<line>_<col>`). `--show-qualifiers` still adds the reason of each decision, so it stays. The printer tracks whether it is inside device code (a `kernel` declaration or a `kernel(...)` block): host code is printed `kernel.tensors.matmul`, device code `.tensors`.

### 14. Methods and top-level statements (2026-11-10)

`specialize_untyped` now also takes over a struct or enum method with an untyped parameter used by a method call, field access or index, when the method name is declared once among all methods (struct, enum, `ext`, trait defaults and required signatures, kernels): the receiver's type is not needed, the call `o.m(x)` is found by the name and rewritten to `o.m_Counter(x)`, and the instance is added to the declaring type. A method that forwards an untyped parameter to a taken-over function or method is taken over too (fixpoint). Calls are rewritten from free functions, methods, `mod` bodies (unless the `mod` declares the same name) and the file's own statements (one synthetic body whose locals are the top-level `let`s). Refused as before, with the D5 error: a method in an `ext`, a trait default or a kernel, a function inside a `mod`, a method name declared twice. A bare call of a sibling method (`bump(a)` inside the struct) is not valid Boring for the transpiler (it asks for `self.`), so it is not handled.

### 14.1 `mod` (2026-11-10)

A `mod`'s items are reached by their plain name from anywhere (the interpreter runs modules flat, the transpiler flattens them), so there is no qualified call to resolve: a function or method of a `mod`, nested ones included, is a candidate when its name is unique across the file and every `mod` (a name declared in two `mod`s is refused with the D5 error). Calls are found by name wherever they are, and each instance is put back where its original was declared (`replace_originals`).

### 14.2 Across files (2026-11-10)

Placement: the caller's file, not the declaring one. Each Rust file is its own module, so a typed copy `bump_Counter` in `main.rs` cannot collide with another file's, and no instantiation registry is needed. In the declaring file a top-level `pub` candidate is kept as the generic original with an internal `specialise_only` attribute (not printed in the `.bri`, derived again when the file is parsed); the transpiler skips it (`emit_fn`), so the library never has a generic Rust form of it, and an uncalled one is no longer an error. In a file that `use`s it, `specialize_imported` (called where the other import-aware passes run, `desugar_array_block_with_imports`) takes the marked imports that no local function shadows (a name imported from two files is ambiguous and not taken), rewrites the calls with the same machinery as the same-file case and appends the instances as private functions. A forwarder (`twice` calling `bump`) is marked too, its copy asks for the `bump` copy.

Limits: top-level free functions only (a `pub` method or `mod` function is not marked, so it keeps the D5 error); the copy is checked and emitted in the caller's scope, so a private helper of the library that the body names is not visible there (rustc says so); the monomorphizer's own cross-file instances (explicit `<T>`) are a different path and still go through its registry.

### 15. The transpiler's monomorphizer in the `.bri`, tried again (2026-11-10): not shippable

Tried a second time, now in `full` only (`mono` stays the output of the front end): a
`Transpiler::global_pre_scan` + `monomorphize_program` on a clone of the parsed program (as
`specialise_qualifier_params` does), a placement step (each instance right after the generic it comes
from, the generic's `type let`s that depend on its parameters removed, because the instances are
appended at the end of the file where `boring run x.bri` cannot see them from the statements above),
and an order-insensitive comparison of the generated items in the oracle. It fell back to the
unmonomorphized program when the scan panicked (a labeled multi-dimensional array). Result over
`tests/cases`: `full` build identical went from 521/522 to 508/522 and `run` from 449/449 to 444/449.
The remaining blockers, each the size of a pass of its own:

1. **Patterns follow the subject's instance, which the monomorphizer does not rewrite.** After
   specialisation `match slot: Slot.Full(p)` still names the dropped generic `Slot`; the transpiler
   resolves it from the subject's type, the interpreter cannot (`qualifier_gaps_enums`: wrong result,
   `qualifier_gaps_headers`: other output). Rewriting a pattern needs the type of its subject, i.e. the
   typing pass of level 3.
2. **Instances of an imported generic are made in the declaring file.** `Stack_int` of `boring.collections`,
   `Wrapper_string`/`copy_first_int` of a sibling file are not in the calling file's program
   (`boring_stdlib_collections`, `monomorphize_cross_file_*`: `undefined variable`). The calling `.bri`
   would have to declare them (a `.bri` that is no longer the image of one source file) and the reader's
   own transpile must not make them again.
3. **The same instance is read back differently.** An instance method or function written with the
   concrete type (`Board_owned other`) gets a different parameter-passing decision than the same
   instance made from the generic (`Board<elem'owned>`): `merge_owned(&mut self, other: &Board_owned)`
   against `other: Board_owned`, in eight `qualifier_*` cases. Both are valid Rust; the second is what
   the original pipeline produces, so the file is not faithful.

So the earlier conclusion stands: this belongs with the two-phase compiler (the typed AST gives the
patterns' instances, the placement of imported instances and the parameter decisions at one place), not
before. What `mono` and `full` already show is the same-file specialisation of untyped parameters
(`bump_Counter_actor`), the qualifier decisions, the shared traits and the GPU residency.

### 15.1 `--show-types` and `--show-qualifiers` stay (2026-11-10)

Decision after the `.bri` work: neither is deprecated.

- `--show-qualifiers` gives the reason of every decision (`default: ...`, `shares its qualifier with ...`,
  notes) and the full list, including what `full` leaves out on purpose because writing it is a
  constraint, not a report (`inline` locals and parameters, parameter borrows, `'actor` on a `mut`
  binding, `'owned` on a local). The `.bri` shows the result; this shows why.
- `--show-types` is mostly redundant with `mono` (the inferred headers and the bounds as a trailing
  comment). It keeps the marking of types that came from a default value and the per-use instantiation
  view. No upkeep cost, so it stays; both help texts point at `--emit-bri` for the whole-program view.

### 15.2 Origin comments (2026-11-10)

The origin comments of 3.6 exist, in a shorter form: the specialisation records, per copy, the base function (or `Owner.method`) with its line, the written types of the untyped parameters and the line of the first call that asked for it (`ORIGINS` in `specialize_untyped.rs`, cleared before each `.bri` and read by the printer). It is a trailing comment on the `def` line, the place of the `# inferred:` note, so no comment item reaches the generated Rust. No file name: a `.bri` is the image of one file, and a copy of an imported function says so instead. The transpiler's own instances (`Entry_Counter_actor`) have none, since they are not in the `.bri` (section 15).

## 16. Measuring what a shared typing layer would replace (2026-11-10)

Question: is "two phases" really "a typer first"? What exists, and how far do the existing resolvers agree?

### What exists

Expression typing is done at least seven times, none sharing code:

| Where | What it answers | State it reads |
|---|---|---|
| `emit_methods.rs` `resolve_expr_type` (209 lines, 119 call sites) | declared `Type` of an expression | per-local tables filled while emitting |
| `emit_methods.rs` `resolve_expr_struct_type` (44 call sites) | the struct *name* of a receiver | same tables |
| `emit_methods.rs` `resolve_expr_declared_type`, `infer_std_type`, `overload_arg_type`, `resolve_iterable_type` ... | variations (declared, std result, overload argument, loop element) | same tables |
| `infer_field_scan.rs` `body_local_types` | locals of a body, outside emission | its own `HashMap` |
| `specialize_untyped.rs` `local_types` | written types of locals, ambiguity set | its own |
| `infer_return_types.rs` `ty_of`, `field_ty` | types for return-type inference | declarations |
| `qual_infer.rs` `of_expr`, `subject_type`, `receiver_base`; `desugar_tensor_quals.rs` `receiver_type`; `desugar_array_block.rs` `infer` | qualifier / residency / array-block decisions | each its own |

The emission-time tables are the core: `var_types`, `var_struct_type`, `var_struct_types`,
`var_scalar_types`, `var_std_types`, `var_init_exprs`, `var_newtype_type`, `known_local_vars`,
`var_mutex_types`, `var_rwlock_types`, `var_atomic_types`, `var_lock_scalar`, ... (about 15), written
at roughly 140 sites and read at roughly 190 while the emitter walks the program. A type is therefore only
known *during* emission, in the order the walk reaches it: this is what keeps decisions out of the AST.

### Agreement, measured

Temporary instrumentation (not committed): every call of `resolve_expr_struct_type` on a variable or a
field also computed `resolve_expr_type(..)` reduced to a struct name, over the `.br` files of
`tests/cases` (`build --emit-rust`). 34,787 queries (weighted by how often an expression is asked, not
distinct expressions):

| Outcome | Queries | Share |
|---|---|---|
| same struct | 21,275 | 61% |
| both unknown | 11,823 | 34% |
| only the type resolver knows | 1,466 | 4.2% |
| only the struct resolver knows | 98 | 0.3% |
| different | 105 | 0.3% |

Reading the 203 disagreements: the 105 "different" are variables declared with a trait and built from a
struct (`Visitor v = A()`: one resolver answers the declared type, the other the concrete one). Both are
right, to different questions; a typed AST would need both (declared and dynamic). The 98 "struct only" are
implicit-generics locals with no written type (`let r = Request(...)`), where only the construction says
what they are. The 1,466 "type only" are cases the struct resolver cannot follow (calls, indexes). One
case showed a `Type::Named` that holds emitted Rust text (`Rc<CDog { name: ... }>`), i.e. a type made from a
string.

### What it says

- The resolvers mostly agree where both answer (99.7%), so a shared table is feasible without redefining
  semantics: the disagreements are two questions, not two truths.
- The cost is not the resolvers but their inputs: they read tables that exist only during emission.
  Moving them before emission means computing those tables in a separate walk (the `deep_pre_scan` /
  `infer_field_scan::body_local_types` pattern, which already exists as a second, partial copy), then
  making the emitter read the result instead of its own tables.
- Size of the first useful step: one `TypeTable` (expression position to declared type and concrete
  struct), built by a walk that reuses `resolve_expr_type` on a transpiler with the tables filled as
  the emitter fills them; then replace the 190 reads one family at a time (`var_struct_type(s)` first,
  11 + 32 reads). Each step keeps the generated Rust identical, so the 1535-case suite is the test.
- What it does not give: the decisions (qualifiers, borrows, clones, locks) still live in
  `infer_qualifiers` and the emitters; a typing layer is their prerequisite, not their replacement.

### 16.1 The `TypeTable`, first version (2026-11-11)

Built: `src/transpiler/type_table.rs`, wired through `Transpiler::type_table` (shared with the
sub-transpilers), `type_scope` (set by `emit_fn`) and a wrapper around `resolve_expr_struct_type`. Off unless
`BORING_TYPE_TABLE` is set. Key: `(scope, line, col, "v:<name>" | "f:<name>")`: the scope because the
copies made by the specialisations keep the positions of their declaration, the name because `self.items`
(sugar for an implicit field) is two expressions on one token.

**Record then replay.** Over the 522 `.br` of `tests/cases`: 31,158 questions answered from the table, 27
asked to the emitter after all, Rust identical (the two files that differ between two runs, `supertraits`
and `generics`, differ the same way without the table: hash-map order). So the question "what struct is this
variable or field?" is answerable from a table.

**Position alone does not always decide.** 12 positions in 2 files (and the 27 questions above) get two
different answers: a variable asked before and after its binding is registered (`clone_deep_copy`), and
synthetic `self` nodes at `(1, 1)` in a scope the key does not tell apart (`protocols`, emitted by a path
other than `emit_fn`). Such a position is marked ambiguous and falls back to the emitter. These are the
order-dependent questions a pre-pass will have to settle by a rule, not by position.

**A walk of its own** (`build_struct_names`, no emission, flow-insensitive: annotated parameters and
`let`s, constructor calls, `self`, fields of a resolved struct, implicit `self.field`): against the
recorded answers, 5,243 agree (81%), 1,127 not covered (17.6%), 64 different (1%), 41 claimed where the
emitter says unknown (0.6%). Reading them: the different ones are specialised instances (`Board_inline`
for `Board`: the walk runs before the monomorphizer, it must run on its output); the over-claims are
enum-typed locals the emitter's struct resolver does not know (`V attr` matched in `match attr:`) and
names rebound by a pattern. The 17.6% are what needs the emitter's richer rules: call results, indexing,
loop elements, `if let`/`match` bindings, std results.

**Next steps, in this order** (each keeps the Rust identical, checked by replay):
1. run the walk after the monomorphizer, so instance scopes match;
2. cover calls (`call_return_type`, overloads), indexing and loop elements, then pattern bindings;
3. when the walk agrees with the record on every position it claims, replace the emitter's tables for
   this question (`var_struct_type`, `var_struct_types`: 43 reads) by the table; repeat with `var_types`
   (80 reads), `var_scalar_types`, `known_local_vars`;
4. the record then becomes a regression check rather than the source of the table.

### 16.2 The walk after the monomorphizer, and with types (2026-11-11)

Done: the walk runs on the program each emission actually emits (`emit_program_items`, so after
`monomorphize_program` and every rewrite before it), accumulating over the files a build inlines; it types
expressions (not only names): a table of declared types for parameters and `let`s, `type_of` for variables,
fields, indexes, free-function calls (not overloaded), constructors, method calls (owner known, name
declared once for it), `for x in typed_array`; the top-level statements are one body (scope `""`), `ext`
methods have their owner.

Over `tests/cases`, against the recorded answers (6,478 decided positions): 5,775 agree (89.2%), 649 not
covered (10.0%), 0 different (64 before the walk ran after the monomorphizer), 51 claimed where the
emitter says unknown (0.8%), 3 undecided. Steps: 81.0% (first walk) → 83.2% (after the monomorphizer) →
87.5% (top-level statements) → 89.2% (types, `ext`).

What is left of the 649: 273 are `self` (42%) in methods where the emitter asks about a `self` it builds
itself while emitting (implicit `self.field`), an expression that is not in the AST at all: answered by
`self` = owner when asked, not by a walk; 123 are fields of receivers the walk cannot type; 253 are other
variables (93 in two files about the standard library and type introspection): bindings made by patterns
(`match`/`if let`), closure parameters, std call results, generic `T`. The 51 over-claims: enum-typed
locals the emitter's struct question does not answer, and `observed_qualifier` (24, qualified fields
`'observed`) where the emitter deliberately says unknown.

Next: pattern bindings and closure parameters (they need the type of the matched subject, which the walk
now has), then the synthetic `self`; then the 43 reads of `var_struct_type(s)` can start to come from the
table, with the replay as the check.

### 16.3 Patterns, closures, variants, implicit fields (2026-11-11)

The walk now binds pattern variables with their types (`match` arms and `elif`/`if let`/`while let`/`guard let`
patterns: `Some(p)` against `T?` gives `T`, `Variant(a, b)` gives the variant's declared payload types, a
tuple pattern its elements), types closure parameters (written type, else unknown), types `Enum.Variant(..)`
as the enum, and records for a bare field name `x` in a method both the variable and the `self.x` access the
emitter builds on the same token. A method's `self` is a rule (its owner), counted apart.

Over `tests/cases`, 6,475 decided positions: 5,525 found by the walk, 629 by the `self` rule (together
95.0%), 270 not covered (4.2%), 0 different, 51 claimed where the emitter says unknown (0.8%), 3 undecided.
Steps: 89.2% (16.2) → 93.5% (patterns, closures, `self` rule) → 95.0% (variants, implicit fields).

Left of the 270: locals typed by what the walk cannot see from declarations (calls of overloaded or
imported functions, std results, `T` generics), fields of those receivers, and the two files about the
standard library and type introspection (43 of them). The 51 over-claims are unchanged: enum-typed locals
the emitter's struct question does not answer, and `'observed` fields it deliberately leaves unknown.

What it means for the reads: the walk finds what the emitter knows for 95% of the positions and never
contradicts it, so the first reads can move to the table where the walk claims the position (the emitter
stays the fallback for the rest, and the replay is the check): `var_struct_type` (11 reads) and
`var_struct_types` (32) are the candidates.

### 16.4 The walk answers instead of the emitter (`BORING_TYPE_TABLE=walk`, 2026-11-11)

First time the emitter's behaviour can come from the walk: `resolve_expr_struct_type` for a variable or field
asks the table the walk built for the program being emitted (after the monomorphizer) and falls back to its
own tables where the walk does not claim the position. Rust identical over the 522 `.br` of `tests/cases`
(`tests/type_table.rs` checks ten of them, in `replay` and `walk`).

Live-weighted, the picture differs from the per-position one (95%): of the 33,872 questions asked while
emitting, the walk answered 15,035 (44%); of the 18,837 left to the emitter, 13,052 are "no struct" answers
and 5,785 (17% of all questions) are names the walk missed. So the walk covers 72% of the *positive*
answers by weight (15,035 of 20,820): the questions asked many times are the hard ones (loop elements,
call chains, results of std calls).

Why this is still not the end of `var_struct_type(s)`: they are read directly by name in 60 more places,
each with its own fallbacks and meaning (`contains_key` as "is a known struct variable"), not only through
`resolve_expr_struct_type`. Moving them is a site-by-site job where the expression is at hand, each batch
checked by the corpus in both modes; the unknown-or-no-struct distinction (13,052 answers) needs the walk
to say "known, and not a struct" rather than stay silent. What would change the picture most: the 5,785
missed names (std results, overloaded and imported functions), which the type of a call needs the callee's
declaration for: imported ones need the file graph the walk does not read yet.

### 16.5 `self` by rule (2026-11-11)

Of the 5,785 names the emitter knew and the walk did not, 4,347 (75%) were the `self` of a method, asked about
expressions the emitter builds itself while emitting (an implicit `self.field`): they are not in the AST. In
walk mode `v:self` is now answered from the scope (`Owner::method`), which is what the emitter answers too
(same Rust over the corpus).

Result, live-weighted over `tests/cases`: of 33,872 questions, 19,315 answered by the walk (57%), 13,052
"no struct" left to the emitter, 1,505 names the walk misses (4.4% of all questions). The walk now covers
92.8% of the positive answers by weight (19,315 of 20,820).

What the 1,505 are: 815 are one file (`boring_stdlib_collections`, instances of an imported generic
`Stack_int`), then locals bound from std results and from overloaded or imported functions, and
`T`-typed generics. The imported ones need the walk to read the files `use` brings in (their structs, enums
and function signatures), as `infer_return_types::collect_inference_imports` already does for the front end.

### 16.6 Imports (2026-11-11)

The walk reads the declarations of the files the program imports (`collect_inference_imports`: sibling
files, `[deps]` projects, the embedded `boring.*` stdlib; their structs, enums with variant payloads,
fields, method and function return types), and knows that a generic struct `Stack` makes `Stack_int`, an
instance the monomorphizer names in the file that declares the generic.

Live-weighted over `tests/cases`: 19,488 of 33,872 questions answered by the walk (57.5%), 1,332 names it
still misses (3.9%; 1,505 before), so 93.6% of the positive answers by weight (19,488 of 20,820). Same Rust in
both modes (`generics` and `supertraits` vary between two runs of the same binary, in both modes, over the same
two outputs: hash-map order). The 1,332 are now spread flat: the introspection files (`type_var_introspect_*`,
`introspect_thread_safety`: 282), `json_scenarios`, trait-object scenarios; none dominates, each needs its own
rule (std results, `Type`/`Info` values, trait objects built from structs).

The two-phase line of work stops being "find the missing rule" and becomes "decide what to move": the walk
answers the struct question well enough to start replacing the emitter's by-name reads (60 sites), the
1,332 stay with the emitter until their rules are written.

### 14.3 `ext` methods (2026-11-11)

A method of an `ext` of a plain type (no type parameter, no type argument: not `ext Board<elem'actor>`) is a
candidate like a struct method, owner = the extended type, name unique among all methods; the instance is
added to the same `ext`. An `ext` of a generic's instance keeps the D5 error, as do trait defaults and
kernels. Test: `tests/cases/implicit_generics_specialise_ext.br`.

### 14.4 `pub` methods across files (2026-11-11)

The marking of 14.2 now covers the `pub` methods of top-level structs and enums: the declaring file keeps the
original marked (`swap_methods`, never emitted), and `specialize_imported` also takes the marked methods of
the imported structs and enums (a name two imports declare, or that a local method has, is not taken). The
copies of a method go in an `ext <Type>:` the calling file gains: a cross-file `ext` of an imported type
works, private fields included (checked by building and running the generated project). A caller that
declares its own method of the same name hides the imported one: a call on a receiver written with the
imported type is then an error naming both (`check_shadowed_calls`); a receiver whose type is not written stays
silent, since the call may mean the local method. Tests:
`tests/specialize_cross_file.rs` (run, the generated project, the library without the original).

### 14.5 `pub` functions of a `mod` across files (2026-11-11)

Same mechanism as 14.2: a `pub` free function is kept, marked, at any depth (`replace_originals` no longer
requires the top level for functions), and `specialize_imported` looks for marked functions inside the
imported `mod`s too. The copy lands at the top level of the calling file, where the plain name resolves, as
in the same-file case. `pub` methods of a struct declared inside a `mod` are marked and found the same way (14.6): the caller's `ext <Type>:`
needs no `mod` path, since the plain name resolves. Tests: `tests/specialize_cross_file.rs`.

### 14.6 `pub` methods of a struct inside a `mod`, across files (2026-11-11)

`swap_methods` marks a `pub` method at any depth, `specialize_imported` and `check_shadowed_calls` look for
marked methods inside imported `mod`s. Nothing else: the copies go in an `ext <Type>:` of the caller, and the
type's plain name resolves because a `mod`'s items are flat. Test: `tests/specialize_cross_file.rs`.

### 16.7 Toward a complete walk, and the emitter's quirks (2026-11-11)

Rules added to the walk, each measured on `tests/cases` against the recorded answers: traits (a trait-typed `let`
takes the struct its initialiser builds, as the emitter does; a trait-typed *parameter* answers the trait name, a
loop element or local of a trait type does not), enum variants looked up per enum (variant names collide across
enums) for unit variants, payload constructions and patterns, `as` casts, array literals, the reflection
types the transpiler registers itself (`IntrospectInfo`, `Field`, `Method`, `value.introspect()`), inits, setters,
conversions and trait defaults (their scope: the owner), and the emitter's own refusals (`'observed` values,
enum-typed declared locals and parameters, `Enum.Variant` paths).

Steps over the 6,478 decided positions: 95.0% → 96.5% (traits, casts, arrays, enum variants) → 97.4%
(reflection types) → 97.6% (inits/setters/conversions/trait defaults, trait parameters). Not covered: 130 (2.0%);
claimed where the emitter says unknown: 25 (0.4%); different: 0.

What the over-claims show is not a missing rule: the emitter's answer for the *same shape* depends on state
that is not in the program. A parameter `Counter c` is a struct for it in one function and unknown in another
(`error_trait_mut_immutable_arg` against `mut_param_type_method`), depending on how the parameter is passed and
what was registered before. A typing pass written from the AST cannot reproduce that without copying the quirk,
and the quirks are exactly what the emitter's per-local tables encode as a side effect. So two goals pull apart:
*agreeing with the emitter* (what the oracle measures) and *a clean typed table* (what a two-phase compiler
wants). The honest route is to make the table the specification and let the emitter's quirks that change Rust
show up as differences to review one by one, instead of copying them.

`BORING_TYPE_TABLE=check` is the gate for that review: it runs over every program a suite transpiles (not only
`tests/cases`), changes nothing, and logs each position where the walk claims something different. Over the
whole transpile suite and the self-hosted interpreter build (1,539 + 4 + 186 tests, before the rules above) it
logged 180 lines, all of one kind ("the walk claims a struct, the emitter says unknown"), from six variable
names in a handful of programs, and not one contradiction.

### 16.8 The by-name reads, and why the type table is not enough (2026-11-11)

27 reads of `var_struct_types` by variable name (`get(v).cloned()`, `contains_key(v)`) now go through one accessor
(`Transpiler::local_struct`, `is_struct_local`); the ten composite ones (`.get(v).map(..)`, `.or_else(..)`) stay
direct: replacing them changed the Rust of four programs in walk mode, each for a reason specific to its site.
The accessor's walk-mode answer comes from a per-body table of locals the walk builds (`(scope, name)` to
struct), flow-insensitive: a name bound to two different things in one body has no entry.

What a walk of the AST cannot see, found by running `check` over the corpus and fixing the largest classes: the
emitter does not keep in `var_struct_types` a local that has a written qualifier (it is tracked through its
qualified `var_types` entry), nor one decided to live behind a lock (`inferred_qualifiers`), nor one read out of
a field or an element (`let d = denominator`: and the Rust differs, `d.clone()` against `d`), nor one bound by an
`if let t = x as T` downcast, nor one declared with an enum type. Each is a *decision about representation*,
not a type. With those taken into account the walk's claims agree with the table 97.9% of the time (8,814 against
190, live-weighted) and walk mode gives the same Rust as the emitter over all of `tests/cases`.

So `var_struct_types` is not a typing table: it is type x representation. A table of types alone cannot replace
it, and the representation is exactly what `infer_qualifiers` and its neighbours decide (step 4). The two-phase
design has to carry decisions as data next to the types; that is the next work.

Concurrent work note: another session was editing the tensor files of the same checkout while this was done
(`tensor_host.rs`, `ast/mod.rs`, `checker/tensor.rs`, `desugar_tensor_*.rs`), which stopped the tree from
building for a while; this work was validated in a clean `git worktree` of the previous commit plus these files.

### 16.9 The decisions pass (step 4, first part), 2026-11-11

`decide_program` (in `emit_top.rs`, run where the walk is built, on the program the emitter is about to emit) is a
phase-1 pass for the qualifier decisions: for every function and method body, `infer_qualifiers` on a clone of the
transpiler with the facts `emit_fn` gives it just before emitting (parameters, lent parameters, mutability, the
method's forced borrows from its signature), results kept by `(scope, name)` in `decision_table`. The 30 reads of
`inferred_qualifiers` outside the pass itself go through `Transpiler::decided_qual`.

Two attempts, measured with `check` over `tests/cases`:
1. reuse the *speculative* runs `pre_infer_fn_qualifiers` already makes (a probe for signature qualifiers, with
   another configuration: lent parameters unlent, methods treated as free functions): 93.1% agreement, the
   table saying `Inline` where the emitter borrows. A probe is not the decisions pass.
2. a run in the configuration of the real emission: 98.7%, every difference "the table says `Inline`, the
   emitter has no entry". `Inline` is the default and an absent entry is not the same to the emitter (an
   explicit one annotates the `let`: `let p: LPoint = ..`), so the table is read only for a non-default decision:
   1,303 reads, 0 differences, and walk mode gives the same Rust for all of `tests/cases`.

What this does not give, and the reason a table computed in advance is not simply the emitter's state: the emitter
re-runs the inference on nested blocks while it emits a body ("emit_body may call infer_qualifiers recursively for
nested arm/block bodies ... which would overwrite inferred_qualifiers", `emit_fn`'s own comment), so the value of
a read can depend on how far the emission is. The corpus does not hit it (0 differences); a program that does
would show as a difference in `check`, which is why `check` has to run over every suite before the emitter stops
computing these itself. Not covered: the decisions of the top-level statements (the file's own body), closures'
captured bodies, and the other decision families (borrows of trait parameters, shared traits, atomic promotion,
lock and clone choices), each with its own interleaving.

### 17. The monomorphizer in `full` (step 6), 2026-11-11

Done, in `full` (not `mono`, which stays the output of the front end): `transpiler::monomorphized_for_bri` runs the
transpiler's `global_pre_scan` + `monomorphize_program` on a clone of the parsed program (falling back to the
generics as written if the scan panics on sugar the later desugarings remove) and four steps make the result a file a
reader can run and build:

1. **Placement** (`place_instances`): each instance goes right after the generic it was made from, or where that generic
   was; instances of a generic this file does not declare (`Stack_int`) go after the `use`s (`imported_instances`, from
   the transpiler's registry of generic declarations and instantiations); a generic left next to its instances loses
   the `type let`s that depend on its parameters.
2. **Patterns** (`type_table::rewrite_variant_patterns`): the walk's typing gives the subject's type, a variant pattern
   of a dropped generic enum is renamed to the subject's instance (`Slot::Full` -> `Slot_inline::Full`).
3. **By-value parameters** (`mark_by_value_instance_params`): the third blocker of section 15, solved by experiment: an
   instance method whose parameter came from a generic's instance type (`Board<elem'q> other`) is passed by value by
   the emitter, and spelled with the plain name (`Board_owned other`) it is borrowed. Writing `Board_owned'inline other`
   gives back the original Rust. (The one residual, `qualifier_gaps_calls`, is a method of a generic struct whose
   by-value parameter the emitter also clones: no spelling gives "by value, cloned".)
4. **The oracle** compares generated items as a set, ignoring blank lines (`sorted_items`): the monomorphizer's instances
   come last in the transpiler's file, next to their generic in the `.bri`.

Result over `tests/cases` (450 with an `.expected`, 523 built): run 450/450; build identical 519/523 (522/523 without the
monomorphizer). The four: a redundant cast (`shared_dyn_tree_copy`), a redundant clone (`qualifier_gaps_calls`), the
comment of a dropped static (`monomorphize_struct`) and `monomorphize_cross_file_main`, where the library's own
generic `Wrapper<T>` has a `type let` that depends on `T` and nothing in the reader's world specialises it.
The by-value finding also says something about the two-phase design: a parameter's passing mode is a decision
(here derived from the generic form of its type) that must live in the AST as data, not in how the type is spelled.

### 18. Sharing the typer with the checker and the interpreter (step 7), 2026-11-11: what it would be, measured

**The interpreter** has no static typing to share: values carry their type at run time (`Value::Struct`, ...), so
nothing there duplicates a typing pass. Nothing to merge.

**The checker** types on purpose as little as it needs. Its `Binding` carries three independent flags, each
feeding one check and each "positively known, never guessed": `ctor_type` (the struct of an unannotated
`let h = S(...)`, for `check_same_object_lend`), `builtin_collection` (for the content-mutation gate),
`kernel_type`. Its own comment says overload resolution and call-argument typing are out of reach "without type
inference it doesn't have". So the checker/typer question is not a duplication to remove, it is a capability the
checker lacks.

Measured with the walk over `tests/cases` (unannotated `let`s, counted per emission): 542 are a direct constructor
call, the only shape `ctor_type` knows; 133 more (+25%) have a struct type the walk knows from something else (a
call of a typed function, a method with a declared return type, an enum variant, a cast, an element of a typed
array, a pattern binding); 1,377 are unknown to the walk as a *struct* (scalars, strings, collections, std results:
the walk answers the struct question only). So sharing would let the checker's struct-based checks reach a quarter
more bindings; it would add diagnostics, not remove code, and a new diagnostic is a behaviour change to review.

Decision: not done as a refactor. If wanted, the way in is the one used for the emitter: an accessor in the checker
for "the struct of this binding", `check` mode logging where the walk knows more than the checker, and each new
check enabled on its own. The walk is already a standalone function of a `Program` (`build_struct_names`), so it
needs no emitter state to be called from the checker.

### 19. All by-name reads through the accessor (point 1 of the remaining work), 2026-11-11

The 19 composite reads of `var_struct_types` that section 16.8 left direct (`.get(v).map(..)`, `.or_else(..)`) go
through `local_struct` too, so there is no read of the emitter's table by variable name left outside the accessor.
In walk mode they had changed the Rust of four programs; each was a shape the emitter does not track as a struct
local while the walk typed it, and became a rule of the walk: `x.introspect()` on a type-level path
(`Shape.Unit.introspect()`), `let sa = x as T` (a conversion; `x as T else ..` is tracked), and `T c = b` where `b`
is another local (an alias).

Result over `tests/cases`: walk mode gives the same Rust for all 523 programs (the two that varied between runs
before no longer differ), and `check` finds, live-weighted, 8,894 by-name reads that agree with the emitter against
138 where the walk claims a struct and the emitter has none (98.5%), 20,653 positions against 92, and 1,303
decisions against 0.

### 19.1 Two questions, two projections; what is left of point 1 (2026-11-11)

The emitter answers two neighbouring questions with different rules: *by position* (`resolve_expr_struct_type`, which
falls back on the declared types of the locals) and *by name* (`var_struct_types`, the table of struct locals). A plain
cast, an alias of a local (`T c = b`) and a local read out of a field or an element (`let d = denominator`) are
answered by position and not tracked by name. The walk now keeps the two apart: those shapes are left out of its table
of locals only, not of its answers by position, which brought the position coverage back up. Closure parameters given
to an array method (`items.map((x): ..)`) take the array's element type.

Over `tests/cases`: walk mode gives the same Rust for all 523 programs; by position, 97.8% of the recorded answers are
found (118 not covered, 23 claimed where the emitter says unknown, none different); live-weighted, `check` finds
20,871 positions agreeing against 97, 8,898 by-name reads against 138, and 1,303 decisions against 0.

What is left of point 1 is a long tail, not a missing mechanism. The 118 positions are bindings the walk makes
ambiguous because a name is reused in a body (`s` as a closure parameter and as a loop variable), std results, and
fields of such receivers. The 23 + 138 are the emitter's own inconsistencies (a parameter `Counter c` is a struct
for it in one function and unknown in another); settling them means choosing which answer is right, which can
change the Rust, so they are a list to review, not rules to add.
