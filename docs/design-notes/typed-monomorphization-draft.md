# Typed monomorphization — Boring → Boring (no generics) → Rust

Status: draft, nothing implemented. Written 2026-11-09 after a design discussion on
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
