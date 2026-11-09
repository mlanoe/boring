# Typed monomorphization — Boring → Boring (no generics) → Rust

Status: draft; the `.bri` output (section 12) and the spike of section 9 are implemented, the rest is not. Written 2026-11-09 after a design discussion on
qualifiers for implicit generics. Related: the book's "Untyped parameters", "Untyped
fields", "Qualifier parameters", "Qualified generics — `<T'q>`", "Turbofish
monomorphization".

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

Remaining limits (not design decisions): same file only (a `pub def` that other files call is
still specialised by its own file's calls alone), top-level free functions only (no `mod`, no methods,
no `task`/`stream`), all untyped parameters of a function are specialised together,
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
