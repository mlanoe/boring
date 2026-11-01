# Qualifier parameters on structs — draft

Status: **phases 0, 1 and 2 implemented** (labelled generic arguments; explicit instantiation of structs, enums and `ext` blocks; inference at construction, section 8), **plus default values and chained qualifiers** (section 10), **passing the parameter to an enclosing struct** (section 11) and **functions and methods that declare a parameter of their own** (`def f<'q>(Board<elem'q> b)`, section 12). Decisions taken while implementing are in section 7. Purpose of
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
4. **Passing the parameter to an enclosing struct** (section 11, implemented): other structs, enums and
   `ext` blocks holding `Board<elem'cells>` inside their own `'cells`. Plain functions taking a concrete
   `Board<elem'actor>` need nothing and work from phase 1.
5. **Functions and methods that declare a parameter of their own** (`def f<'q>(Board<elem'q> b)`,
   section 12, implemented): one specialisation per argument chain, a declaration that is optional
   in a signature, calls inferred from written qualifiers, methods specialised per method instance.

## 6. Open questions

- Whether `'new` as a bound means exactly `T'new`'s set (draft: yes).
- ~~Should a qualifier parameter accept chained qualifiers (`'actor'observed`)?~~ Decided and implemented: yes, see section 10.
- Checker on the unspecialised declaration: conservative intersection vs deferring all
  qualifier-dependent checks to the specialised copy.
- ~~Whether a qualifier parameter may be defaulted~~ Decided and implemented: yes, see section 10.
- ~~Whether a struct may pass its parameter to a contained struct~~ Decided and implemented: yes, written explicitly, see section 11.
- ~~Whether a function may declare its own qualifier parameter~~ Decided and implemented: yes, with an optional declaration in signatures, see section 12.
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

Phase numbering: passing the parameter to an enclosing struct is phase 4 (section 11); functions
declaring their own parameter are phase 5.

## 11. Passing the parameter to an enclosing struct (phase 4)

### Design

An enclosing struct, enum or `ext` that holds a qualifier-generic struct hands its own parameter on:

```boring
struct Board<'elem = 'inline>:
    var [mut Point'elem] cells = []

struct Game<'cells = 'inline>:
    var mut Board<elem'cells> board = Board<elem'cells>()   # 'cells of Game is 'elem of Board
    var int turn = 0

let Game<cells'actor> g = Game()        # Game_actor holds Board_actor holds Point'actor
let Game g2 = Game()                    # Game_inline / Board_inline (defaults)
let Board<elem'actor> b = Board()
let g3 = Game(board = b)                # cells = actor, from the written qualifier of b
```

Rules:

1. In a field, parameter, return type or body of a declaration that declares `'cells`, the argument
   `elem'cells` reads "the `elem` of `Board` receives the value of my `cells`". The names are independent.
2. The specialisation cascades: struct, enum and `ext`, chains included (`Game<cells'actor'observed>` →
   `Board<elem'actor'observed>`, `Game_actor_observed`).
3. The passed name must be declared by the enclosing declaration (`unknown qualifier parameter`, with position).
4. The enclosing bound must fit inside the receiving one, checked at the declaration, not only at the
   instantiation. The receiving default applies only when the parameter is not passed.
5. **Choice (A), always explicit.** A bare `Board` or `Board()` in a field of `Game` takes `Board`'s own default,
   never `Game`'s parameter. There is no implicit propagation (no invisible generics, section 9) and no
   usage-based inference; only written qualifiers are evidence. A bare construction under an expected type that
   passes a parameter (`var Board<elem'cells> b = Board()`) is an error rather than a silent default.
6. A concrete argument in the field (`var Board<elem'actor> b`) freezes the instance; the enclosing declaration
   needs no parameter.
7. Section 8's inference extends: `Game(board = b)` takes `cells` from the written qualifier of `b`, same conflict
   rule; an expected `Game<cells'actor>` makes the `board` argument a `Board<elem'actor>`.
8. Recursive and mutually recursive holders (`Node<q'q>?`, `Ping`/`Pong`) resolve with a fixed point over the
   reachable `(declaration, chain)` instances.
9. Cross-file and `[deps]`: the registry is global, as for every instance.
10. `boring run` erases the parameters (same diagnostics, same output); GPU host emitters share the general
    emission; `--linux` keeps rejecting the feature.

Out of scope: functions and methods that declare their own parameter, invisible generics, a qualifier-generic
trait, `--explain-inference`.

### As implemented

- **Parser.** `elem'cells` parses as a labelled qualifier argument whose chain is `[Param("cells")]` (the type
  parser already produced `Param` for a name in `qual_scope`; an unknown name now reports ``unknown qualifier parameter `x` `` with the
  position). `resolve_generic_args` accepts a pass-through chain and checks the enclosing bound (kept in
  `Parser::qual_bounds` next to `qual_scope`) against the receiving bound member by member: ``the bound of `c` (...) is not within the bound of `q` in `Cell`: `'inline` is outside it``.
  `parse_without_inference` (used to read the declarations of other files) now resolves labelled lists too, with a
  re-entrancy guard so a cycle of imports cannot recurse; before, a field `Board<elem'cells>` of an imported
  struct was read in the lenient (unresolved) shape.
- **Specialisation.** `substitute_type` replaces a `QualArg([Param(p)])` by the instance's argument, so
  specialising `Game` for `actor` yields `Board<elem'actor>` in its fields, field defaults, `init`s, method
  signatures and bodies, which the existing rewrite turns into `Board_actor`. The instance set is closed in
  `build_instantiation_map` (worklist over struct and enum instances, plus the `ext` blocks that apply to them,
  `Transpiler::global_qual_exts`): the uses recorded while specialising an instance are new candidates.
- **Termination.** An argument is either a written chain or a pass-through of one, never built from other
  arguments, so the `(declaration, chain)` pairs reachable are finite and the worklist, which drops pairs it has
  already seen, always ends. A non-terminating cycle therefore cannot occur and has no error; the only
  remaining failure of a self-holding struct is the ordinary one of Rust (a recursive type without indirection,
  e.g. two structs holding each other directly through `Option`, `E0072`), unchanged by this feature (use an
  array or an `'owned` link).
- **Inference.** `qual_infer.rs`: `from_ty` reads a pass-through slot against a written `Board<elem'actor>`
  (declared type of a name, or an explicit `Board<elem'actor>()`), and `resolve_site` pushes the resolved
  instance's parameter types (`subst_pass`) into the arguments, also for `Game<cells'actor>(board = Board())`.
  An expected type that still contains a pass-through is not evidence: the bare construction is reported.
- **Tests.** `tests/cases/qualifier_enclosing.br`, `qualifier_enclosing_chains.br`, `qualifier_enclosing_xfile`
  (`run` + `build`, four configs), `tests/qualifier_params_errors.rs` (unknown name, bound, bare construction,
  conflict, out-of-bound evidence), `tests/qualifier_params_targets.rs` (metal/cuda/rocm text, `--linux`),
  `src/parser/tests_qualifier_params.rs`.
- **Known gaps.** An enum payload of an `'actor`/managed-`'owned` instance hits the existing `PartialEq` derive
  limits of enums (unrelated to the parameter); `elem'cells'observed` (a suffix added on top of a passed parameter)
  is not accepted.

## 12. Qualifier-generic functions and methods (phase 5)

### Design

A function or a method declares a qualifier parameter of its own, so that it works for every instance
of a qualifier-generic struct instead of one `Board<elem'actor>`:

```boring
int count<'q>(Board<elem'q> b):                  # declared
    b.cells.len()

int count2(Board<elem'q> b):                     # declared by its use in the signature
    b.cells.len()

def add<'q as 'inline|owned = 'inline>(Board<elem'q> b, Point'q p):
    b.cells.push(p)

struct Game<'cells = 'inline>:
    var Board<elem'cells> board = Board<elem'cells>()
    def merge<'q>(Board<elem'q> other):          # the method's own parameter, distinct from 'cells
        ...

let Board<elem'actor> a = Board()
count(a)                                         # q = actor, from the written qualifier of a
count<q'owned>(Board<elem'owned>())              # explicit
```

Rules (decided with the user):

1. **Declaration.** `<'q>` with bound and default exactly as on a struct (`<'q as 'inline|owned = 'inline>`),
   the bound checked at the declaration (the default must belong to it; a parameter passed on must fit
   inside the receiving bound, section 11). A function is **specialised per argument chain**, named like a
   type instance (`count_actor`, `count_actor_observed`).
2. **The declaration is optional for functions and methods only.** A qualifier identifier that appears in
   the signature (parameters and return type) without being declared and that is not a known qualifier
   word is introduced by its first occurrence. The list of words is the compiler's real one, not a copy
   of this document (`generic_args::QUALIFIER_WORDS`, the same list that forbids naming a parameter
   `'actor`). The explicit declaration stays **mandatory** when a bound or a default is needed. The
   omission does **not** extend to a struct/enum/`ext`/trait (there `<'cells>` is part of the type) nor
   to the body (a `'q` the signature did not introduce is an error). It is still *visible*: no invisible
   generics (section 9), the parameter is written in the signature. Typo guard: an implicit parameter
   written once in the whole function (signature and body) produces the warning ``qualifier parameter `q`
   is never related to another use``, never an error.
3. **Methods, choice (A).** A method may declare its own parameter independent of the struct's. It is
   specialised by instance of the struct **and** by instance of the method, only for the calls that are
   in the program.
4. **Calls infer with the existing rules** (sections 8 and 10): explicit arguments, the expected type, the
   written qualifier of the arguments, the default, otherwise ``cannot infer qualifier parameter `q` of
   `count`, write `count<q'...>` ``; two disagreeing proofs are a conflict, a proof outside the bound a
   bound error.
5. **The parameter is passed on**: in the body `Board<elem'q>` (locals, constructions), calls to other
   generic functions `g<q'q>(...)` or inferred; the specialisation cascades by a fixed point over the
   reachable instances, with a clear error if it does not terminate.
6. **Cross-file**, 7. **parity** `run`/`build` (erasure, same diagnostics at parse time), GPU hosts share
   the general emission, `--linux` rejection unchanged. 8. **No inference by usage**: only written
   qualifiers are proofs.

Out of scope: invisible generics (section 9), `--explain-inference`, qualifier-generic traits,
`elem'cells'observed` (a suffix on a passed parameter), type-argument inference.

### As implemented

- **Lexer.** `tick_in_fn_header`: the `<...>` list that follows a function's name and closes with a `>`
  directly followed by `(` is a declaration header, every tick in it is a qualifier tick. The prefix
  before the name must not contain `=` nor start with a statement keyword, which keeps `x < 'a'`,
  `f(a < 'b', c > (d))` and the like raw strings.
- **Parser.** The return type is written before the `<'q>` list, so `parse_fn_decl` first reads the
  signature ahead (`prescan_fn_qual_params`, up to the `:` at depth 0): a tick preceded by `<` or `,` is a
  declaration (read with `parse_qual_param_decl`, which gives the real bound for the pass-through check of
  the return type), any other `'name` that is not a qualifier word and not already in scope is introduced.
  They are pushed on `qual_scope` for the whole function and truncated after it, so nothing leaks into
  the next declaration, and a name equal to one of the enclosing struct's is an error for an explicit
  declaration (``has the same name as one of the enclosing declaration``). An implicit parameter is
  appended to `type_params` as `^q` (after the explicit ones, in order of first occurrence) and recorded
  in `FnDecl::implicit_quals` with its occurrence count over the whole function, which the checker turns
  into the warning. A body occurrence of an unintroduced name keeps the old "unknown qualifier parameter"
  error, whose text now says only a signature introduces one. A trait method declaring a parameter is an
  error. Also fixed: a parameter type whose generic/qualifier list is wrong was retried as a bare name
  and reported as ``expected RParen, got Lt``; `parse_params` now reports the real error.
- **Specialisation** reuses the free-function and generic-method machinery: a function with `^q` in
  `type_params` is in `global_generic_fns`, a call `f<q'actor>(..)` is a candidate, the instance is
  `f_actor`, `substitute_fn_decl` substitutes `T'q` and `Board<elem'q>`, and the generic declaration is
  dropped (no generic-Rust fallback exists). `build_instantiation_map` is now **one worklist** over
  struct, enum, `ext`, function and method instances; every specialised body is harvested for the
  qualifier uses it names (`Board<elem'actor>`) and for the calls it makes (free and method), which
  become new candidates once concrete. That is the fixed point of rule 5. Termination: an argument is a
  written chain or a pass-through of one, never built from other arguments (`q'q'observed` is rejected at
  the call), so the set of `(declaration, chain)` pairs reachable is finite and a pair already seen is
  never specialised again: polymorphic recursion, mutual recursion and self-holding structs resolve. A
  non-terminating cycle therefore does not exist and has no error (same conclusion as section 11).
- **Methods.** A method's own parameter (`^q` in `m.type_params`, the struct's are not copied there) goes
  through the existing generic-method registry (`global_generic_methods`, name-only resolution, a name
  owned by several declarations is ambiguous). `monomorphize_program` appends one instance per method
  instance to the struct, `ext` or enum (and to the stored copy of a generic struct, so every instance
  of the struct receives it) and drops the generic method (`substitute_*_decl` and a final retain).
  Consequence of the name-only resolution: a method instance is attached to **every instance of its
  type that exists**, not only to the ones the calls use; correct, with some unused code. Tightening it
  would need the receiver's instance at each call site.
- **Inference** (`src/qual_infer.rs`). `Decls` gained the qualifier-generic functions (`qfns`) and methods
  (`qmethods`, with the count of methods per name for receiver-less resolution). `qcall_args` collects the
  evidence (the expected type against the return type, then each argument's written type against its
  parameter slot, with parameters of the enclosing function accepted as pass-through evidence, which
  struct constructions deliberately do not accept), chooses per parameter (conflict, bound, default) and
  the call becomes `GenericCall(count, [q'actor], args)`, rewritten to `count_actor` by the ordinary
  call-site rewrite. A method call needs the receiver's declaration name: written type of the receiver,
  `self` inside its own type, or a method name that is unique; otherwise the error asks for
  `merge<q'...>`. The explicit forms (`count<q'owned>(Board())`) push the instance's parameter types into
  their arguments like a constructor does. A function that also has type parameters is not inferred.
  The initialiser of an untyped top-level `let` is walked too.
- **Cross-file.** Declarations come from the same `Decls`/`GenericDecls` readers as for structs.
  Specialised bodies are registered before any file is emitted: `Transpiler::pre_scan_qual_instances`
  runs the specialisation over the registry of generic declarations and scans the result (struct shapes,
  signatures, borrow qualifiers), because the per-file specialisation otherwise happens too late for a
  caller in another file or earlier in the same one (it passed `count_actor(x.clone())` where the callee
  takes a borrow).
- **Checker and targets.** The warning is raised in `check_fn`. `--linux` rejects any function or method
  with a qualifier parameter, called or not (`validator/kernel.rs`); the GPU host emitters share the
  general emission.
- **Known limits.** An explicit `count<q'owned>(..)` or `Board<elem'owned>()` written inside a string
  interpolation is not resolved (the interpolation is parsed on its own, without the declarations);
  positions of an error inside an interpolation point at the start of the string. A method instance is
  attached to every instance of its type (above). The interpreter's behaviour is unchanged: it erases
  the parameters. Two pre-existing `managed` limits showed up in tests and are not related to the
  feature: a `Point'owned` local passed to a `Point'owned` parameter is wrapped twice in single-thread
  managed mode, and a `'owned` parameter pushed into an array takes a lock it then moves in multi-thread
  managed mode.
- **Tests.** `tests/cases/qualifier_fns.br`, `qualifier_fns_cascade.br`, `qualifier_fns_methods.br`,
  `qualifier_fns_xfile` (`run` + `build`, four configs), `tests/qualifier_fns_errors.rs` (conflict, bound,
  cannot infer, unknown label, body and struct introduction, shadowing, passed bound, suffix on a passed
  parameter, the warning), `tests/qualifier_params_targets.rs` (metal/cuda/rocm text, `--linux`),
  `src/parser/tests_qualifier_params.rs`, lexer tests.
