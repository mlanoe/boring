# Draft — implicit generics: type parameters you never write

Status: **working draft**, nothing implemented. The "Impact inventory" below was read from the
compiler source on 2026-11-07 (not from memory); the "Decisions" section is a proposal to be
argued with, not a spec.

## Goal

Let a program omit type annotations that the program itself fully determines, and have the
compiler turn each omission into an ordinary generic parameter:

```boring
struct Response:
    let status                          # field with no type
    let payload                         # field with no type

def fetchGreeting():                    # no return type
    return Response(status = 200, payload = "hello world")

def fetchScores():
    return Response(status = 200, payload = [12, 7, 31])

let greeting = fetchGreeting()
print greeting.payload                  # hello world
let scores = fetchScores()
print scores.payload.len()              # 3
```

`status` and `payload` become implicit type parameters of `Response` (emitted as
`Response<status, payload>`, shown below as `Response<T0, T1>`). `fetchGreeting` is inferred to return
`Response<int, string>` and `fetchScores` `Response<int, [int]>`: one struct, two instantiations, each
fully typed. The language keeps its Rust output (real monomorphised generics, zero cost); only the
syntax disappears.

**Invariant that keeps this sound:** every member must be fully typed at each construction site, so an
instantiation is never partial. Unconstrained values (`Response(status = 200, payload = [])` with
nothing ever pushed to `payload`, or `payload = nil`) are an error, not a guess.

## Non-goals

- No dynamic typing: after inference every binding has one concrete type, exactly as if written.
- No inference in the signature of an `@api` item (see D1); everywhere else inference is free.
- No inferred trait *bounds* in the first version (see D5).
- No change to qualifier parameters (`Board<elem'actor>`): book §13 says "no implicit propagation, no
  invisible generic" for **qualifiers**, deliberately. This draft is about **type** parameters; the
  book wording must be reconciled so the two do not read as contradictory.

## What exists today (measured with `target/debug/boring`)

| Input | Result |
|---|---|
| `struct Response:` / `    let payload` | parse error `expected identifier, got Newline` (`parser/mod.rs`, `parse_struct_body`) |
| `def wrap(x):` | parse error `parameter 'x' ... has no type annotation` (`require_typed_params`, `parser/parse_fn.rs:253`, called at :97, :580, :973) |
| `def mk(): return Box(v = 3)` (Box explicit `<T>`) | emitted `fn mk() -> ()`; `boring run` fails `cannot access field 'v' on Void` |
| `Box<int> mk(): return Box(v = 3)` | OK, `-> Box<isize>` |
| `let e = Box(v = [])` | emitted verbatim `Box { v: vec![] }`; rustc resolves the type |

So generic structs and functions **already exist** (`struct Box<T>`, book §13, `tests/cases/generics.br`,
`labelled_generics*`, `monomorphize_*`). Inference *inside a body* works only because rustc does it.
Boring has **no unifier**: expression types on the Boring side are strings
(`infer_expr_type -> Option<String>`, `transpiler/helpers.rs:2104`), and there are no type variables.
The interpreter is dynamic and re-derives bindings from runtime values
(`infer_struct_type_params`, `interpreter/methods.rs:3962`).

## Impact inventory

Counts are occurrences of `type_params`; they size the blast radius, they are not a to-do list.

| Area | Where | Impact |
|---|---|---|
| AST | `ast/mod.rs`: `FnDecl.return_ty: Option<Type>`, `Param.ty: Option<Type>` (already optional); `FieldDecl.ty: Type` (**not** optional) | Fields need an "inferred" representation (`Option<Type>` or a placeholder `Type`). Every `field.ty` reader changes or must be proven to run after the pass. |
| Parser | `parse_struct_body` (`parser/mod.rs:2286`); `require_typed_params` (3 call sites) and its two tests (`parser/tests.rs:558,569`) | Accept an untyped field and an untyped `def` parameter. Those two tests assert the current error and must be rewritten. Trait methods (abstract signatures) must **keep** requiring types (D4). |
| New pass | `main.rs`: three duplicated entry points (`run` ~2415, `build` ~2589, a third ~2684) each repeat `parse → desugar_inline_kernels → desugar_labeled_array → desugar_array_block_or_exit → desugar_inject_or_exit → check_with_imports` | Insert once, **after `desugar_inject`** (needs the final item set) and **before `check_with_imports`** (the checker's `collect_fn_signature`/`collect_struct_signature`, `checker/mod.rs:779,821`, read declared types). Three call sites, plus project mode and `--emit-rust`. Must see imported declarations (same-project sibling files, `[deps]`, `boring.*` stdlib): mirror `desugar_array_block_with_imports` / `desugar_inject::collect_imported_items`. |
| Checker | `checker/mod.rs` (4856 lines) | If the pass rewrites the AST to explicit types first, no change. Error messages must still point at the user's *original* line/col, not a synthetic one. |
| Interpreter | `interpreter/methods.rs` (26 refs), `eval_expr.rs`, `call.rs` | If the AST is rewritten, the interpreter sees ordinary generics and `Type::Named("_")` fallbacks stay unused. Parity (`docs/interpreter-parity.md`) needs a test per phase: `boring run` and `boring build` must print the same. |
| Transpiler | `emit_struct.rs` (51), `emit_top.rs` (46), `mod.rs` (16) | Emits `impl<T: Clone> ...` and `Display for Box<T>` with `T: Clone + Debug`: implicit parameters inherit exactly these blanket bounds. A field used with any other trait (`.len()`, `+`) fails in **rustc**, not in Boring (D5). |
| Monomorphiser | `transpiler/monomorphize.rs` (105 refs, 2043 lines) | Handles explicit turbofish only (`Struct<T>(...)`). Not needed by phase 1 (plain Rust generics suffice); it is the natural tool if we later want specialised copies for GPU or for `pub` signatures. Name-only method resolution means two generic methods sharing a name stay ambiguous: unchanged. |
| GPU targets | `transpiler/kernel_generic.rs:68-70` ("type-generic kernels are not supported on this target"), `desugar_gpu_struct.rs`, wgpu/metal/cuda/rocm | A struct with implicit parameters that a kernel touches must be fully concrete at that point. Either reject with a clear message or monomorphise first (D6). |
| Qualifiers | `infer_actor_params.rs`, `infer_actor_elems.rs`, `qual_infer.rs`, `infer_qualifiers.rs` | These run on explicit types today and skip declarations with qualifier params. Order versus the new pass must be fixed (the new pass first), and a field `T0` that ends up `string` vs a struct matters for `'actor` promotion. To be verified, not assumed. |
| Docs/tests | `docs/book.md` §13 (generics), ~line 4659 "Inference at construction"; `tests/cases/` (1120 files) | New book subsection; new `tests/cases/implicit_generics_*`; `generics.*` and `labelled_generics*` must stay byte-identical. |
| Tooling | `--show-qualifiers` exists as precedent | Add `--show-types` (D7). |

## Decisions (proposal)

**D1 — Stable signatures are opt-in with `@api`.** `pub` keeps its current meaning (visibility; a
bare `use` of a sibling file sees everything, and only a multi-segment or `[deps]` `use` needs `pub`).
It says nothing about stability, so it is **not** what triggers a "write your types" rule. Inference
is free everywhere, `pub` or not, except on an item marked `@api`:

| Item | Signature | Inference |
|---|---|---|
| anything without `@api` (private or `pub`) | may omit types | yes, no warning |
| `@api` | **complete and explicit**: parameter types, return type, field types, generic parameters written out | bodies only (locals) |

Rules for `@api` (same attribute family as `@provide`/`@singleton`/`@inject`):

- Requires `pub` (error otherwise), as `@provide` does.
- Any omitted type in the signature is an error pointing at the missing annotation. A `struct`/`enum`
  under `@api` must declare its generic parameters (`struct Response<S, P>`) and type its fields: an
  inferred field type would silently become part of the API.
- **Scope on a type (decided):** `@api` on a `struct`/`enum`/`trait` covers every `pub` member inside it
  (fields, methods, variants' payloads): each one needs a complete signature and none repeats the
  attribute. Non-`pub` members are not part of the API and stay free to infer.
- Arguments are compile-time literals: `@api(since = "1.0")`, optionally `deprecated = "use X"`. **`since`
  is free text (decided)**: documentation for readers and tooling, never parsed or compared with
  `boring.toml`'s `version`.
- **Type leak lint:** an `@api` signature that mentions a type that is not itself `@api` is a warning
  (the equivalent of Rust's private-in-public), since the API would depend on a type free to change.
- `@api` guarantees **form**, not stability: detecting a breaking change needs a snapshot comparison
  (`boring api --check`), deferred to phase 4. `src/semver.rs` exists; not read yet.

Why not infer "library" from the absence of `main` or from a `boring.toml` key: `boring-ui` has no
`main` today, but a library may reasonably grow one (a demo), and a heuristic cannot say which of a
project's `pub` items other projects actually use. A per-item attribute says it where the code is.

**D2 — Mechanism: unify, then rewrite.** One new pass `infer_generics.rs`, same shape as
`infer_actor_params.rs`: a union-find unifier over the AST that writes the missing annotations
back (`return_ty`, `Param.ty`, field parameters). Everything after it sees explicit generics. This keeps
interpreter/checker/transpiler untouched and is the main reason to do it as a pass rather than inside
the transpiler.

**D3 — Defaults and ambiguity.** Literals default (`42` → `int`, `1.0` → `float`). A variable left
unconstrained at the end of a function is an error naming it (`cannot infer type of `payload`, write
a type`). Two pieces of evidence that disagree are an error, never a pick (same rule as qualifier
inference).

**D4 — Where omission is allowed.** Struct fields, `def` parameters, `def` return types: yes.
Trait/protocol method signatures: no (they are the contract). Recursive and mutually recursive
functions: return type required (inference would need a fixpoint with polymorphic recursion).
`T? ` with `nil` binds nothing (matches the interpreter's existing rule).

**D5 — Bounds.** Version 1 adds none: the existing blanket `T: Clone` is all an implicit parameter
gets. A body that needs more (`.len()`, `+`, a user method) is a **Boring** error ("`payload` is used
with `.len()`; give it a type or a bound") raised by the pass where it can detect it, not an rustc error
from generated code. Real bound inference is a later phase.

**D6 — GPU.** Reject an implicit-parameter struct reaching a kernel, with a message asking for an
explicit `Name<type>`. Revisit with the monomorphiser once phase 1 ships.

**D7 — Visibility.** `boring check --show-types` prints each inferred header
(`struct Response<status = int, payload = string>` per use, `def fetchGreeting() -> Response<int, string>`), and every inference error shows
the line that fixed the type and the line that contradicts it.

**D8 — Structs.** Self-reference shares the parameters (`next: Node?` is `Node<T0>`, never a fresh
parameter); `[Response]` is homogeneous (mixing `fetchGreeting()` and `fetchScores()` results is an error,
`dyn`/enum is the escape hatch); an explicit `Response<int, string>` is always accepted and wins over inference.

**D9 — Qualifiers: the written qualifier of the argument at construction is the evidence.** Same rule
as qualifier parameters (book §13, "Inference at construction"): a qualifier counts only when it is
*written* on the argument; an inferred one is no evidence (no inference by usage).

```boring
let counter = Counter'actor(0)
let r = Response(status = 200, payload = counter)   # payload binds to Counter'actor
```

- Mechanism: each untyped field yields a type parameter (`T0`) **and** a hidden qualifier parameter
  (`'q0`), the field being typed `T0'q0`. Construction infers `'q0` from the written qualifier of the
  argument, else the default; this reuses the existing qualifier-parameter machinery instead of adding
  a new one. An argument with no written qualifier binds the bare type, and the field then follows
  today's inference for bare fields.
- Two constructions with different written qualifiers are two distinct instances
  (`Response_actor`, `Response_owned`), as with `Board<elem'actor>`.
- **Constraint:** Rust cannot abstract over "plain or locked" (book, "Element mutability": a generic
  over the element handle would force a `with(|p| ..)` closure at every access). So a binding to a
  lock-like qualifier (`'actor`, `'guard`, `'observed`) needs a Boring-side **specialised copy**
  (`monomorphize.rs`), not a Rust generic. `'inline`, `'owned`, `'shared` keep the same access shape
  and can stay ordinary generics.
- Known obstacle: the book requires a function that declares both type and qualifier parameters to
  receive its generic arguments written out (`f<T = int, q'actor>(..)`, "type arguments are not
  inferred"). That is exactly what this feature removes, so the restriction must be lifted, or
  bypassed for implicit parameters. The code that enforces it has not been read yet.
- `infer_actor_params`/`infer_actor_elems` already skip declarations with qualifier parameters, so a
  struct with qualified implicit parameters is skipped by them: no ordering conflict.
- **Phasing:** phase 3 supports only qualifiers that keep the access shape (`inline`, `owned`,
  `shared`), and reports "write explicit generics" for `'actor`/`'guard`/`'observed`; full
  specialisation lands in phase 4.

**D10 — Diagnostics name the field, not `T0`.** The user never wrote `T0`, so no message, `--show-types`
output or hover shows it. An implicit parameter is displayed by the field it comes from: `payload`
(`Response<status = int, payload = string>`), and `payload`'s qualifier parameter as `payload'q`.
Errors read `cannot infer the type of `payload` in `Response(...)``, not "cannot infer `T0`". Internally
(and in the emitted Rust) the parameter still needs a unique identifier; it is derived from the field name
(`Payload`-style, suffixed on a clash with a declared type name), so generated code stays readable and
the `T0`/`T1` shorthand in this draft's examples is purely illustrative.

**D11 — A default value types the parameter only when the construction does not override it.**

```boring
struct Request:
    let url
    let retries = 0                     # default value, no type written
```

- `Request(url = "/scores")` omits `retries`: the default applies, `retries` is `int`, the instance is
  `Request<url = string, retries = int>`.
- `Request(url = "/scores", retries = 1.5)` supplies it: the argument wins, `retries` is `float`, the
  instance is `Request<url = string, retries = float>`. The default expression is never evaluated there,
  so it need not be compatible with the overriding type.
- This is the existing "then the declared default" rung of qualifier-parameter inference, applied to a
  type parameter: order is explicit arguments, then the construction's own evidence, then the default.
- No implicit numeric coercion: `Request(url = "/x", retries = 1)` with `let retries = 0.5` binds `int`
  (the argument is evidence, the default is not), so a `float` is obtained by writing `1.0`. This can
  surprise; `--show-types` marks which parameters came from a default (`retries = int (default)`).
- A `var` field's type is fixed at construction: `r.retries = 1.5` on an `int` instance is an ordinary
  type error, not a re-instantiation.

## Phases (each shippable alone, each its own task)

1. **Return types, existing generics. Shipped** (`src/infer_return_types.rs`, called from `parser::parse`
   before the actor passes, so every entry point and every file gets it). Infers `return_ty` for
   non-recursive top-level `def`s from explicit `return <expr>` statements, using today's
   `struct Box<T>` constructors. Findings that shaped it:
   - The parser gives every `def` without a written type `Some(Void)`, so "omitted" and a written
     `void` are indistinguishable; the pass acts only when the body returns a value.
   - A closure cannot contain `return` in Boring, so only nested `def`s need excluding.
   - It is best effort and **never reports an error** (an expression it cannot type may be a valid
     `void` call); the "cannot infer" error of D3 waits for a stage that knows every call.
   - A trailing expression is not used for inference. Boring does return the last expression when a
     return type is written, but with none the function is `void`; inferring from the last expression
     would change the signature of every function ending on a call (decision: only `return <expr>`).
   - Methods, imported callees, qualified/borrowed/`mut` types are out of scope; `@api` functions are
     skipped and now error when they return a value with no written type.
2. **Untyped `def` parameters** as fresh generics. **Shipped** (`src/infer_param_generics.rs`, wired in
   `parse_fn_decl`, which replaces the missing type by `Type::TypeParam("T_<name>")` and adds the bounds
   once the body is parsed, so `mod` members and nested `def`s are covered the same way). Findings:
   - A generic function today only gets `T: Clone`: `x + x` or `"{x}"` on a `T` is rejected by rustc.
     Phase 2 was therefore useless beyond pass-through without a first slice of bound inference, which
     it carries: `Display` (print/interpolation), `PartialEq`/`PartialOrd` (comparisons),
     `Add`/`Sub`/`Mul`/`Div`/`Rem` with `Output = T` plus `Copy` (same parameter on both sides). The
     rest of phase 4's bound inference (through calls into other generic functions, `x + 1`, methods)
     is still open; those uses fail in rustc with the bound named.
   - Each untyped parameter is its own type parameter, so `def larger(a, b)` returning either cannot be
     typed. This is the real cost of "no `T` written": relating two parameters needs the `T`.
   - `infer_return_types` accepts the function's own `T_*` parameters (a parameter's type, `n + n`), and
     a callee's own parameter never leaks into a caller (`fn_ret(..).filter(is_plain)`).
   - Methods, `task`/`stream`/`native` and defaulted parameters keep the written-type requirement.
   - The old interpreter test asserting `def add(a, b): a + b` is a parse error was rewritten. Note that
     `add(a, b)` (two parameters) parses, but its return type stays unwritten (`T_a + T_b` has no
     known output), so `boring run` yielded nil and `boring build` failed. Fixed by `check_untyped_returns`
     (D3, narrowly): for a function with untyped parameters whose return stayed unwritten, an error when a
     returned expression *certainly produces a value* (literal, variable, field/index read, array, tuple,
     cast, arithmetic) — conditions around the return do not matter, only the expression does, so a `void`
     function or a returned call (possibly `void`) is never reported. Functions without untyped
     parameters keep the best-effort, never-an-error behaviour of phase 1.
3. **Untyped struct fields** as implicit parameters: `FieldDecl` change, parser, emitter, `--show-types`.
4. *(later)* bound inference; GPU monomorphisation; `boring api --check` (signature snapshot for `@api`).

`@api` itself (attribute parsing, the "complete signature" check, the `pub` requirement) is
independent of inference and can ship before phase 1; it is what makes the later phases safe for
library authors.

## Test plan

Per phase: a `tests/cases/implicit_generics_*.br` pair run in both `boring run` and `boring build`
(parity), a negative case per error in D3/D4/D5/D6, the existing `generics*` and `monomorphize_*` cases
unchanged, and the two rewritten parser tests. Edge cases to force, per the "stateful loops" lesson: a
fixpoint (a function whose return type depends on a call to itself), a field typed only by a later
`push`, a struct reused with two different instantiations in one file, and a cross-file use where the
struct and its construction live in different `.br` files.

## Related, separate topic

A declarative syntax for generics that carry a qualifier, `struct Point<T'qual>` at the declaration and
`let p = Point<Object'actor>` at the use site. Not part of this draft (it concerns how generics are
*written*, not whether they can be omitted), but D9 and it meet on the qualifier parameter, so the two
should be designed with the same surface syntax in mind. To get its own note.

## Open questions

(D1's two earlier questions are settled: `@api` on a type covers all its `pub` members, `since` is free
text.)

(The `infer_actor_elems` ordering question is settled by D9: those passes skip declarations with
qualifier parameters. What remains to read is the code enforcing "type arguments are not inferred" for
functions with both kinds of parameters.)

(Settled by D10: diagnostics name the field, not `T0`.)

(Settled by D11: a default value types the parameter only when the construction does not override it.)

None left as a design question. Still to do before coding: read the code enforcing "type arguments are
not inferred" for functions with both type and qualifier parameters (D9), and confirm how the parser
handles an untyped field that has a default (`let retries = 0`).
