# Qualifier parameters on structs — draft

Status: **phases 0, 1 and 2 implemented** (labelled generic arguments; explicit instantiation of structs, enums and `ext` blocks; inference at construction, section 8), **plus default values and chained qualifiers** (section 10). The pass-through of a
parameter (the old phase 3, now phase 4) remains a draft. Decisions taken while implementing are in section 7. Purpose of
the document: list the impacts before deciding whether and how to build it. Origin: the `let q = arr[i]` aliasing work (`src/infer_actor_elems.rs`, book.md
"Element mutability") picks one qualifier per *declared* type, so a struct field that wants
different qualifiers in different places has no answer except two hand-written structs.

## 1. Idea

A struct may declare a **qualifier parameter** and use it in field types; each use site resolves
it, and the transpiler emits one specialised, non-generic copy per distinct resolution
(`Board_actor`, `Board_inline`), reusing the Boring-side monomorphizer
(`src/transpiler/monomorphize.rs`). The generated Rust stays plain and auditable: no Rust
generic over "plain or locked" (rejected earlier, see book.md "One qualifier per declared type").

```boring
struct Board<'elem>:
    var [mut Point'elem] cells = []

let Board<elem'actor> a = Board()      # explicit, on the binding
mut b = Board<elem'inline>()           # explicit, constructor form
```

A qualifier parameter is a **labelled** generic parameter, with the tick on the side that says what
it is doing: **declared** with a leading tick (`'elem`, like a lifetime parameter, so it can never
be mistaken for a type parameter), **passed** as `label'value` (`elem'actor`, the qualifier
flavour of a labelled argument), **used** as `T'label` (`Point'elem`). Labelled generic arguments
are a separate, prerequisite task (see section 5): `label = value` for types and consts,
`label'value` for qualifiers.

### Declaration syntax

| Form | Meaning |
|---|---|
| `struct Board<'elem>` | `elem` ranges over every qualifier a bare `T` may take: `{inline, owned, shared, actor, guard}` |
| `struct Board<'elem as 'new>` | the indirection set of `T'new`: `{owned, shared, actor, guard}` |
| `struct Board<'elem as 'inline\|owned>` | explicit list, same `\|` union as `T'inline\|owned` (grammar.bnf `owner_qual`) |
| `struct Pair<'a as 'actor\|guard, 'b>` | several parameters, each with its own optional bound |
| `struct Cache<T, 'q as 'new>` | mixes with type (`T`), const and lifetime (`&a`) parameters |

The bound is a *candidate set*, the notion the inference already uses (`T'new` = `{owned, shared,
actor, guard}`, docs/qualifiers.md "Candidate sets"); without `as` it is the full set, like a bare
`T`. A qualifier argument outside it is a checker error. `mut` stays on the element
(`[mut Point'elem]`), it is not part of the parameter. A label must not be a qualifier keyword
(`'actor` is rejected), and a struct cannot declare a type parameter and a qualifier parameter of the same name (the argument `T'actor` is read against the callee's declaration, see labelled-generic-args.md).

### Use of the parameter

`'elem` (the label as a qualifier) is accepted exactly where a written qualifier is: `T'elem`, `[T'elem]`, `{K = mut V'elem}`,
`T'elem?`, and as the argument of another qualifier-generic struct (`Board<elem'elem>` inside
`struct Game<'elem as 'new>` reads "pass my `elem` on as `Board`'s `elem`"). Not accepted: as a chained
qualifier (`T'elem'observed`) in a first version, and in a `req`/`def` signature of a
*non-generic* struct (it is not in scope).

## 2. Resolution

Inference is **not** part of the first iteration: every use of a qualifier-generic struct names its
arguments.

1. **Explicit constructor** (`Board<elem'actor>()`): substitute, check against the bound.
2. **Written on a binding** (`let Board<elem'actor> a`): same.
3. **In a signature** (`def f(Board<elem'actor> b)`): just a concrete type, `Board_actor`. The
   argument says how `Board` stores its elements *internally*; it imposes nothing on the function and
   is unrelated to the qualifier of the parameter itself (`Board<elem'actor>'shared b`, `mut
   Board<elem'actor> b`, ...), which the existing mechanisms infer as for any struct. The function
   is not generic and not specialised. (A function that wants to stay generic over the element
   qualifier would declare its own parameter, `def f<'q>(Board<elem'q> b)`; that is the optional
   phase 2.)
4. A bare `Board` with no arguments is an error (unless every parameter has a default, section 10.1) ("qualifier parameter `elem` of `Board` must be
   given") rather than a guess; a later iteration may infer it per instance from the usage signals
   (an alias read of an element wants `actor`/`guard`/`shared`, a `def` call through an element
   excludes `shared`, thread crossing wants a lock).

The current file-level promotion (`infer_actor_elems`) stays the default for an *unparameterised*
`[mut T]` field; a qualifier parameter is the opt-in for programs that need two behaviours.

## 3. Mechanism

Reuse `monomorphize.rs`: specialisation clones the struct (and its methods) with `elem` replaced by
the concrete qualifier, name-mangled like `Pair_int_string` (`Board_actor`). The existing
precedent for a substituted *permission* is `Container<mut Point>` (`tests/cases/
monomorphize_mut_arg.br`): the substituted type becomes the field's own type and ordinary
field/mutability support applies. Here the substituted thing is an `OwnerQual` rather than a `Type`,
so the substitution table gets a second kind of entry.

## 4. Impact by component

### 4.1 Lexer / parser / grammar

**The conflict with single-line raw strings.** `'...'` is a raw string (commit 2e6efcfc) and the
lexer (`src/lexer/mod.rs`, `tick_is_qualifier`) decides tick-vs-string only from the character
directly before the `'`: letter/digit/`_`/`]`/`}`/`)`/`?`/`&`/`>` means qualifier, anything else
means string. The glued forms need nothing: `Board<elem'actor>`, `Point'elem`, `elem'inline|owned`
all have the tick right after a letter, which the existing rule already reads as a qualifier.
Only the **declaration** forms `<'elem>`, `, 'b`, `as 'new` have the tick after `<`, a space or a
comma, i.e. on the string side.

**Rule for declarations (the only lexer change).** In a declaration header a raw string can never
occur, so inside the `<...>` of a `struct`/`enum`/`trait`/`ext` header line every tick is a
qualifier. The lexer already walks the characters once; it needs one bit of state, "inside the
angle list of a header": set when a line starts (after `pub`; attributes sit on their own lines)
with one of those keywords and a `<` follows the declared name directly, cleared at the matching
`>` (angle depth counted). No parser feedback, no look-ahead beyond the line. Use sites need no
rule, so the fragile look-ahead pattern considered earlier (`Board<'actor>` vs a raw string on the
same line) is not needed. Lexer tests: each declaration form, `struct Pair<T, 'q> as Show:`, an
unterminated tick in a header, and the existing `x < 'a'` / `f(a < 'b', 'c')` unchanged.

Parser/grammar:
- `type_param` grows a `qual_param` alternative: `"'" IDENT ("as" qual_bound)?`, with
  `qual_bound ::= "'new" | owner_qual ("|" owner_qual)*`.
- `owner_qual` accepts an in-scope label; the parser resolves labels before keywords and rejects a
  label equal to a qualifier keyword.
- Generic arguments accept `label'value` (`Board<elem'actor>`, `Pair<K = int, q'actor>` once
  labelled arguments exist).
- `spec/grammar.bnf`, the Linguist grammar and the docs build need the new forms.

### 4.2 AST
`StructDecl.type_params: Vec<String>` is a flat list of names. Needs a parameter kind
(`Type | Const | Lifetime | Qual(bound set)`), or a parallel `qual_params`. `Type` gets
`Qualified(inner, OwnerQual::Param(name))` (or a new `OwnerQual::Param`). Every pass that matches
`OwnerQual` exhaustively must handle it (checker, interpreter, `infer_*`, emitters): the safe
rule is that no pass outside the specialiser ever sees `Param` (it is substituted first, or the
struct is rejected).

### 4.3 Checker
- Arity and bound check of qualifier arguments; error if a bound names a qualifier invalid for the
  use (`'shared` on a `mut` element, existing `mut 'shared` rule applies after substitution).
- Checks that depend on the element qualifier (`[mut T]` content-mutation rules,
  `check_field_def_call_mut_gate`) must run on the *specialised* declaration, or treat `elem` as the
  intersection of its candidates (conservative). Decision needed.
- `boring run` and `boring build` must agree: the interpreter ignores qualifiers, so it parses and
  erases the label, but still runs the checker.

### 4.4 Inference (`infer_qualifiers.rs`, `infer_actor_elems.rs`, `infer_actor_params.rs`)
- New per-instance candidate sets for a binding of a qualifier-generic struct, fed by the signals
  of 2.3. This is the largest piece and the riskiest (a wrong choice shows up as a rustc error
  far from the source).
- No propagation: a function or method receiving or returning `Board<elem'actor>` sees the
  concrete `Board_actor`; parameter/local qualifier inference (including the lent-`mut` rules) runs
  as today against the specialised declaration. The element qualifier never flows from the callee
  into `Board`'s argument, nor from the argument into the parameter's own qualifier.
- Pass order matters, not the semantics: `infer_actor_params` (a `mut` param stored by name into an
  `'actor` member/element becomes `'actor`) and `infer_actor_elems` run right after parsing, i.e.
  *before* specialisation, where the field type is still `[mut Point'elem]`. They must either be
  re-run on the specialised copies or learn to skip an unresolved `'elem`; otherwise a `mut` param
  stored into `Board<elem'actor>.cells` would keep the lent-param store error.
- `--show-qualifiers` must print the resolved argument per instance and the signals that led to it.

### 4.5 Transpiler (general path)
- Monomorphizer: substitution of qualifier arguments; instances registry keyed by
  `(struct, qualifier args)`; name mangling; cross-file registry already exists
  (`global_generic_structs`, `global_instantiations`).
- Method bodies of an instance are emitted with the field qualifier fixed, so the existing
  lock-aware element access (`resolve_lock_field_ref`, `try_emit_qualified_let`) applies
  unchanged.
- Today only *explicit turbofish* sites are specialised; unspecialised uses keep the generic Rust
  path. A qualifier parameter has no generic-Rust fallback, so **every** use must be resolved:
  unresolved use = compile error, never silent fallback.

### 4.6 GPU host emitters (metal / cuda / rocm) and wgpu/simd
- The kernel-file host emitters read the program before/independently of the general pass; they
  would need the specialised program (run the monomorphizer first) or they see `Board<elem'...>`.
  They already resolve fields per `(struct, field)` (`HostTypeEnv`), so a specialised
  `Board_actor` works with no further change once they receive it.
- Structs a kernel mentions stay plain: a qualifier parameter on such a struct must be rejected or
  forced to `inline` (device data).

### 4.7 `--linux` (no_std)
`'actor`/`'guard` have no lock-aware element access there today (see the `disable()` in
`infer_actor_elems`); instances resolved to them must be rejected on that target with a clear
error, bounds like `as 'inline|owned` being the way to declare a kernel-safe struct.

### 4.8 Traits, `ext`, enums, dyn
- `struct Board<'elem as 'new> as Drawable`: one impl per instance; trait objects over several instances
  (`[Drawable]` holding `Board_actor` and `Board_inline`) are fine, they are distinct concrete types.
- A trait method whose signature names `Board<elem'...>` concretely (`Board<elem'actor>`) is fine; one
  that would need to be generic over `elem` (`Board<elem'q>`) has nothing to quantify `q`: rejected in v1.
- **Enums and `ext` are in scope** (decided), not only structs. The syntax already exists in the
  grammar (`enum_decl` and `ext_decl` both take `type_params?`), so the declaration form is the same:
  `enum Slot<'elem as 'new>: Full([mut Point'elem])`, `ext Board<'elem>: def …`.
  - *Enum*: the variant payload types use `Point'elem`; one specialised enum per qualifier
    argument (`Slot_actor`), with `match` arms and constructors rewritten to it. The monomorphizer
    today specialises generic **methods** of struct/ext/enum owners (`MethodOwnerKind`) and has a
    body-substitution helper for enums (`substitute_types_in_enum_body`), but its declaration
    registry for whole-declaration specialisation covers structs and free functions only: it must
    grow an enum entry.
  - *`ext`*: `ext Board<'elem>` adds methods to an existing generic type, so the ext block is
    specialised **in lock-step with the struct instance** (`ext Board_actor`), never on its own;
    its parameters must match the extended type's. An ext on a non-generic type keeps ignoring
    qualifier parameters (it has none to resolve).
  - An `ext` header can also name a concrete instance: `ext Board<elem'actor>:` adds methods to
    `Board_actor` only, while `ext Board<'elem>:` declares the parameter and applies to every instance.
  - `trait` headers also take `type_params?`; a qualifier-generic **trait** is left out of v1
    (see the trait-method bullet above).

### 4.9 Cross-file and `[deps]`
Qualifier arguments appear in `use`d declarations and in callers in other files; the registry is
global over the reachable file graph, so instances created in two files must be deduplicated by
the mangled name (existing behaviour for type arguments).

### 4.10 Docs, tooling, tests
- book.md (struct generics, qualifiers), docs/qualifiers.md (candidate sets, `as` meaning),
  grammar.bnf, Linguist grammar, CHANGELOG.
- Tests: parse/check errors (bad bound, arity, unknown label), per-config transpile cases (four
  configs), `run` parity, GPU host text tests (metal/cuda/rocm), `--linux` rejection, cross-file.

## 5. Phases

Inference is out of scope for the first iteration (decided); section 8 proposes it as phase 2.

0. **Labelled generic arguments** (separate task, prerequisite): `Name<label = value, ...>` for type
   and const parameters in parser, checker and monomorphizer, with tests. Useful on its own
   (`Pair<K = int, V = string>`). The qualifier form `label'value` is defined here but implemented
   with phase 1.
1. **Explicit instantiation** (`Board<elem'actor>`, bound check, specialisation; structs, enums and `ext` blocks).
   Self-contained, verifiable; solves "two variants of one struct" without inference.
2. **Inference at construction** (section 8: expected type, then declared qualifier of arguments,
   else an error; `<...>` only when the construction does not determine it).
3. **Default values and chained qualifiers** (section 10).
4. **Passing the parameter on** (optional): other structs holding `Board<elem'elem>` (they declare
   their own `'elem`), and functions/methods that declare a qualifier parameter of their own
   (`def f<'q>(Board<elem'q> b)`). Plain functions taking a concrete `Board<elem'actor>` need
   nothing and work from phase 1.

## 6. Open questions

- Whether `'new` as a bound means exactly `T'new`'s set (draft: yes).
- ~~Should a qualifier parameter accept chained qualifiers (`'actor'observed`)?~~ Decided and implemented: yes, see section 10.
- Checker on the unspecialised declaration: conservative intersection vs deferring all
  qualifier-dependent checks to the specialised copy.
- ~~Whether a qualifier parameter may be defaulted~~ Decided and implemented: yes, see section 10.
- Interaction with a future implicit parameter for unqualified `[mut T]` fields (unifying with
  `infer_actor_elems`): attractive but explodes every struct into instances; out of scope here.

## 7. Phase 1 as implemented

- **Encoding** (phase 1; chains and defaults extend it, see 10.4). A qualifier parameter is the `type_params` entry `^elem` / `^elem:inline,owned`
  (`ast::qual_param_decl`; the bound is always the expanded member list); an argument is the
  positional carrier `Type::QualArg(OwnerQual)`, a use `Qualified(T, OwnerQual::Param("elem"))`.
  `label'qualifier` is recognised by the callee's declaration (`generic_args.rs`), so `T'actor`
  keeps its meaning for a callee without a qualifier parameter `T`.
- **Lexer.** `tick_in_decl_header` (text of the current line only): inside the `<...>` of a
  `struct`/`enum`/`trait`/`ext` header line every tick is a qualifier tick. A header whose list
  spans several lines is not covered.
- **Specialisation** reuses the monomorphizer: `QualArg` mangles to its name, `build_type_subst`
  keys a qualifier parameter `^elem`, and the type walker (`substitute_type`) replaces `T'elem`.
  Since no generic-Rust fallback exists, type-position uses are rewritten by the same walker
  (`QUAL_REWRITE`), run on every reachable file before the pre-scan (`rewrite_qual_program`) so
  signatures are registered under the emitted names; the walker also collects the instances
  (`QUAL_COLLECT`) and finds bare uses (`QUAL_CHECK`, at parse time, in `check_bare_qual_uses`).
  Enums grew a registry entry (`global_qual_enums`), `ext` blocks are specialised in the file that
  declares them. The generic declarations are dropped from the emitted program.
- **Checker.** Argument errors are parse-time (shared by `run` and `build`). The unspecialised
  body is checked once; the qualifier-dependent lent-`mut`-parameter store check is skipped inside
  a qualifier-generic declaration (rustc catches what remains for `inline`/`owned` copies).
- **Inference passes.** `infer_actor_params` already skips generic structs and `infer_actor_elems`
  leaves an explicitly qualified element (`Point'elem`) alone; both run before specialisation and
  are **not** replayed on the copies. A method parameter written `Point'elem p` follows the instance
  instead of needing the `mut` promotion.
- **Interpreter.** `resolve_type` erases `QualArg` (qualifiers are ignored), so `run` and `build`
  agree.
- **Targets.** GPU host emitters reuse the general emission (text tests for metal/cuda/rocm);
  `--linux` rejects declarations and uses (`validator/kernel.rs`), a stricter rule than the
  "bounds like `as 'inline|owned` are kernel-safe" idea of 4.7, which would need a specialisation
  pass in that emitter.
- **Enums.** Constructors and patterns carry no argument: use the binding form
  `let Slot<elem'inline> s = Slot.Full(p)`; a qualified pattern or a bare `Slot.Full(p)` is not
  supported.

## 8. Phase 2 as implemented: inference at construction

Idea (the usual mechanism of Rust, Swift, Kotlin, C++ CTAD): when the constructor call itself carries
the information, `<...>` is not written. `Pair(1, "a")` is `Pair<T = int, U = string>`; by the same
rule `Board(cells = [a, b])` is `Board<elem'actor>` when `a`/`b` are `Point'actor`. `<...>` stays
necessary exactly when the construction does not determine the parameter (`Board()`, `Board([])`).

**Why it fits.** The parameter is fixed at construction and never changes afterwards, so one local
decision per construction site is enough; no whole-program flow, and no change to section 3's
mechanism (the result is the same explicit `Board<elem'actor>` that phase 1 already specialises).

**Evidence, in order** (first one that applies wins; two that disagree are an error, never a guess):
1. Explicit arguments, if any (`Board<elem'actor>(...)`): unchanged.
2. The expected type: `let Board<elem'actor> b = Board()`, a labelled field/parameter/return of a
   known `Board<...>` type. Costs nothing and covers the empty-literal case.
3. The arguments that feed a `T'elem` slot (a field or constructor parameter written `T'elem`, `[mut
   T'elem]`, `{K = mut V'elem}`): the *declared* qualifier of the argument's type.
4. Nothing determines it: error "cannot infer qualifier parameter `elem` of `Board`, write
   `Board<elem'...>`". No usage-based guess (that stays the deferred item of section 2.4).

**Where qualifiers differ from types** (why this needs more care than CTAD):
- Qualifiers are themselves inferred in Boring. Only a qualifier **written** on the argument (a
  declared `Point'actor` binding or parameter, an explicit-qualifier constructor) counts as evidence;
  an argument whose qualifier is itself inferred (`let p = Point(1, 2)`) or a fresh literal
  (`Board([Point(1, 2)])`) gives none, which is the case where `<...>` stays required. Using
  inferred qualifiers would need a fixpoint between this step and `infer_qualifiers` (the order
  decided earlier is generics first, qualifier inference after), so it is excluded.
- A qualifier is a representation choice, not part of a value's meaning: changing the qualifier of a
  variable far from the construction silently changes which `Board` instance is built. Mitigation:
  the checker reports the inferred instance on request (`boring check --explain-inference`, to be
  designed) and the conflict rule above turns a second disagreeing argument into an error.
- `'inline|owned`-style bounds: evidence outside the bound is a bound error, not a fallback.

**Impacts.** A pass between label resolution and monomorphization, over `ExprKind::Call` of a type
with qualifier parameters; reuses the declared-qualifier lookup of `infer_actor_params`. No parser,
lexer or AST change. The unspecialised-declaration checker is unaffected (it only sees the
specialised copy). `boring run` must apply the same rule so parity holds (the interpreter erases the
parameter, so it needs the checker diagnostics only). Tests: each evidence rule, conflict, "cannot
infer" error, empty literal with and without expected type, cross-file, parity.

**Type arguments** follow the same order (expected type, then argument types); check what the
monomorphizer already infers for types before adding anything: the proposal should only extend it,
not introduce a second mechanism.

Phase numbering: this becomes phase 2; passing the parameter on (former phase 2) becomes phase 3.

### As implemented

- **One pass, `src/qual_infer.rs`**, over one file plus the declarations (fields, `init`s, variants, free-function parameters) of the files it `use`s. It rewrites a resolved bare construction to the mangled instance name (`rename_ctor`) and reports the instance; no lexer, parser or AST change. It runs twice on purpose: in `parser::parse` (via `find_bare_qual_uses`, on a clone, for the diagnostics, so `boring run` agrees) and in the transpiler (`rewrite_qual_program` registers the instances, `monomorphize_program` renames on the real program, before the generic declarations are dropped). Gated by `qual_decls_seen()` so projects without qualifier parameters pay nothing.
- **Evidence** is as proposed, with these precisions. Expected types: `let` with a written type, field defaults, return/tail expression, assignment to a name with a written type, and arguments of calls to a free function, a struct constructor (field or unique `init`) or an enum variant, matched by label or position; array literals pass the element type down. Arguments: a name declared with a written qualifier (local, parameter, field of the enclosing type or `self.x`), directly or as elements of an array/set/dict/tuple literal. A name declared twice with different types, or without a written type, is unknown. Expected type and arguments that disagree are a conflict, as are two arguments.
- **No type-argument inference was added.** The monomorphizer specialises explicit turbofish sites only; every other generic use stays on the generic-Rust path. So a declaration mixing type and qualifier parameters (`Cache<T, 'q>`) is resolved only through an expected type that names all its arguments; argument evidence applies to declarations whose parameters are all qualifiers.
- **Diagnostics**: `cannot infer qualifier parameter(s) ... of `X`, write `X<...>`` (also for a bare type position), `conflicting qualifiers for ...`, `qualifier ... is outside the bound ...`; the first in source order is reported.
- **Known gaps**: method-call arguments (`obj.m(Board())`) give no expected type; a `return` inside a closure is read against the enclosing function's return type; `--explain-inference` was not built.

## 9. Idea considered: invisible qualifier parameters (not built)

Proposal: a struct with a field whose qualifier the qualifier inference leaves undetermined is
implicitly generic over it, as if written with a hidden parameter:

```boring
struct MonPoint:
    let Point p          # same as: struct MonPoint<'qual>: let Point'qual p
```

This is elision, like Rust lifetimes: the parameter exists, is not written, and only inference fills
it. It would also fold `infer_actor_elems` (today's fixed per-declaration rule) into the same
mechanism as the default value. The draft's section 6 already listed it as attractive but too costly;
restricting it to qualifiers that inference did not settle makes it more defensible, but the costs
remain:

1. **Contagion.** `def f(MonPoint m)` would need the hidden parameter too, otherwise `f` cannot know
   its instance. This contradicts the phase 1 decision that `Board<elem'actor>` imposes nothing on a
   function that receives it. Every type holding a `MonPoint` inherits a hidden parameter in turn.
2. **Instance explosion.** One parameter per field gives N^k specialisations (N = candidate set size,
   k = fields); even one per struct multiplies generated code, and recursive structs need a fixed
   point to avoid infinite instantiation.
3. **Whole-program inference.** Choosing the instance at each construction needs the usage analysis
   deliberately left out (alias read wants `actor`, `def` call excludes `shared`, thread crossing
   wants a lock), unlike section 8, which is a local decision.
4. **Hidden ABI.** Editing a function body far away changes the instance built, hence the
   representation, with nothing visible in the code; same across files and `[deps]`.
5. **Targets.** Kernel and `--linux` accept few qualifiers, so the candidate set must be constrained
   there.

Possible shape if ever built: keep explicit parameters (phases 1 and 2) as the only selection
mechanism; apply the invisible form only to fields still ambiguous after the struct-local analysis,
with **one parameter per struct, never per field**, under a **no-contagion** rule (a function that
receives the struct fixes or infers its own qualifier and does not become generic), and provide a way
to display the chosen instance.

**Decision: not built.** First measure how many real structs remain ambiguous after local inference;
if rare, an explicit `Board<elem'actor>` is enough.

## 10. Default values and chained qualifiers (phase 3, as implemented)

Two open questions of section 6 were settled; the design below is implemented, the divergences and
decisions taken on the way are in 10.4.

### 10.1 Default value

```boring
struct Board<'elem = 'inline>:                          # no bound: the full set, default inline
struct Cache<'elem as 'inline|owned = 'inline>:         # bound, then default
```

- Syntax: `= <qualifier>` after the optional `as <bound>`. Unambiguous with the `=` of labelled
  type/const arguments (`Name<T = Point>`) because a qualifier parameter declaration always starts
  with `'`; the lexer rule `tick_in_decl_header` already treats every tick of a header as a qualifier tick.
- The default must belong to the bound (else a declaration error). A parameter may be defaulted
  independently of the others; a defaulted parameter may be followed by a non-defaulted one (all
  arguments are labelled, so there is no positional-order constraint).
- **Evidence order** (section 8) gains a last step before the "cannot infer" error:
  explicit arguments, expected type, written qualifier of the arguments, **then the default**,
  else the error. The default is a declared, explicit choice, not a guess, so it does not contradict
  "no usage-based inference". A bare `Board` (construction or type position) of a struct whose
  parameters all have defaults is therefore valid and means the default instance; one without a
  default keeps today's error.
- Disagreeing evidence is still a conflict: the default never overrides evidence and never
  participates in a conflict.
- A struct with all parameters defaulted is the explicit way to say "this struct normally behaves
  as X, but a caller may pick another instance"; it covers the common case of the invisible
  parameters of section 9 without hidden generics.
- Interaction with `infer_actor_elems`: unchanged for an unparameterised `[mut T]`; the default is
  not read by it. (Folding the two is the section 9 idea, still not built.)
- Mangled names are unchanged: the default instance and the explicit `Board<elem'inline>` are the
  same specialisation.

### 10.2 Chained qualifiers

`Board<elem'actor'observed>`, bound `as 'actor'observed|'inline`, default `= 'inline'observed`.

- An argument, a bound member and a default are a **chain**: a base qualifier optionally followed by
  composable suffixes (`'observed`, `'task`, `'weak`), exactly the chains already valid on a type
  (`T'actor'observed`). The same validity rules apply (`'shared'observed` stays an error), checked
  where the parameter is substituted and at the bound.
- Bound membership is by whole chain: `as 'actor` does **not** admit `'actor'observed`; write
  `as 'actor|'actor'observed` (no wildcard form in v1).
- Use sites substitute the whole chain: `Point'elem` becomes `Point'actor'observed`.
- Mangled name: the chain joined with `_` (`Board_actor_observed`), so it cannot collide with a
  plain qualifier instance.
- Encoding: the `QualArg(OwnerQual)` carrier and the bound member list (section 7) must hold a
  chain (`Vec<OwnerQual>` or the existing chained form of `Qualified`); the parse of an argument
  `label'a'b` reads all glued ticks.
- Kernel and `--linux` stay rejected (a chain adds nothing to that rule).

### 10.3 Impacts and tests

Parser (`generic_args.rs`, header parse of defaults and chains), the `^elem:` encoding of
`ast::qual_param_decl` (default and chain members), `qual_infer.rs` (default step, chain evidence:
the written qualifier of an argument may itself be a chain), monomorphizer mangling, checker
(default within bound, chain validity, bound membership), interpreter erasure (unchanged),
docs (book.md, qualifiers.md, grammar.bnf, Linguist/VS Code grammars, CHANGELOG).
Tests: bare use with default in construction/type/field position, default overridden by evidence
and by explicit argument, default outside bound, chained argument and bound, `'shared'observed`
error, chain mangling, enum and `ext`, cross-file, four configs, GPU host text tests, `--linux`
rejection, `run`/`build` parity.

### 10.4 As implemented

- **Encoding.** `Type::QualArg(Vec<OwnerQual>)` (alias `QualChain`) holds the chain, outermost layer last
  (`[Actor, Observed]`; `[Param(name)]` in an `ext` header). A `type_params` entry is
  `^elem[:member,member][=default]`; a member and the default are chains spelled `actor'observed`
  (`OwnerQual::chain_name` / `chain_from_name`, `ast::qual_param_decl` returns `Vec<QualChain>`,
  `qual_param_default` the default). `actor'task` and `guard'task` are single bases
  (`ActorTask`/`GuardTask`), as in the type parser.
- **Validity** is `OwnerQual::check_chain`: one base from `{inline, owned, shared, actor, guard,
  actor'task, guard'task}` and at most one suffix, `'observed` (not on `'shared`) or `'weak` (only on
  `'shared`/`'actor`/`'guard`). A bound member or default that is invalid is a declaration error; an
  invalid argument is reported at the use. `'inline'weak` is not even parsed on a type, so it is a
  parse error there too.
- **Header parse.** `parse_qual_param_decl` reads `as <members> = <default>`; groups (`new`, `one`,
  `many`) stay single-qualifier sets and take no suffix; `'weak`, `'observed` or `'task` alone cannot
  be a member. The default is checked against the (possibly implicit, full) bound at the declaration.
- **Inference.** `qual_infer.rs` carries chains in evidence (`written_chain`: the layers written on an
  argument's declared type, ignored when they are not a valid chain, e.g. a borrow) and takes the
  default as the last step, per parameter. A construction whose expected type is a *bare* name is now
  resolved on its own evidence and default (that is the route a field default takes). The monomorphizer
  rewrites a bare `Type::Named` of an all-defaulted declaration to the instance in the same type walker
  (`QUAL_DEFAULTS`, computed from the declarations the file reaches), registering the instance.
- **Mangling.** The chain joined by `_` (`Board_actor_observed`); the default instance and the explicit
  one are the same specialisation.
- **Runtime support.** `program_uses_observed` scans the entry file's syntax only; an `'observed` chain
  that only appears as an argument or default in a `use`d file now also sets it, from the instances
  collected by `rewrite_qual_program` (`emit_top::deep_pre_scan`).
- **Known limits.** A bare type annotation is the default instance, so
  `let Board b = Board(cells = [actorPoint])` is a type mismatch (write `Board<elem'actor>` or drop the
  annotation); a bare `Board` inside `Board`'s own declaration is the default instance, not "myself";
  `T'elem'observed` as a use stays unsupported; an `'observed` element inside an enum variant payload is
  not supported by `'observed` itself (pre-existing); the default is not used when a declaration mixes
  type and qualifier parameters and no expected type names all of them (type arguments are not
  inferred); `--linux` still rejects the whole feature; the interpreter has no `'observed`, so `run`
  parity covers chains only for code that does not use `.value`/`subscribe`.
- **Tests.** `tests/cases/qualifier_defaults_chains.br`, `tests/cases/qualifier_defaults_xfile`
  (`run` + `build`, four configs), `tests/qualifier_params_errors.rs`, `tests/qualifier_params_targets.rs`,
  `src/parser/tests_qualifier_params.rs`.

Phase numbering: passing the parameter on (functions and enclosing structs declaring their own
parameter) is now phase 4.
