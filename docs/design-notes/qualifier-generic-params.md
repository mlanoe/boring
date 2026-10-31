# Qualifier parameters on structs — draft

Status: **draft, nothing implemented.** Purpose: list the impacts before deciding whether and
how to build it. Origin: the `let q = arr[i]` aliasing work (`src/infer_actor_elems.rs`, book.md
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
4. A bare `Board` with no arguments is an error ("qualifier parameter `elem` of `Board` must be
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

Inference is out of scope for the first iteration (decided).

0. **Labelled generic arguments** (separate task, prerequisite): `Name<label = value, ...>` for type
   and const parameters in parser, checker and monomorphizer, with tests. Useful on its own
   (`Pair<K = int, V = string>`). The qualifier form `label'value` is defined here but implemented
   with phase 1.
1. **Explicit instantiation** (`Board<elem'actor>`, bound check, specialisation; structs, enums and `ext` blocks).
   Self-contained, verifiable; solves "two variants of one struct" without inference.
2. **Passing the parameter on** (optional): other structs holding `Board<elem'elem>` (they declare
   their own `'elem`), and functions/methods that declare a qualifier parameter of their own
   (`def f<'q>(Board<elem'q> b)`). Plain functions taking a concrete `Board<elem'actor>` need
   nothing and work from phase 1.

## 6. Open questions

- Whether `'new` as a bound means exactly `T'new`'s set (draft: yes).
- Should a qualifier parameter accept chained qualifiers (`'actor'observed`)? (Draft: no, v1.)
- Checker on the unspecialised declaration: conservative intersection vs deferring all
  qualifier-dependent checks to the specialised copy.
- Whether a qualifier parameter may be defaulted (`<elem'inline\|owned = inline>`),
  which would also make the existing unparameterised `[mut T]` inference a special case.
- Interaction with a future implicit parameter for unqualified `[mut T]` fields (unifying with
  `infer_actor_elems`): attractive but explodes every struct into instances; out of scope here.
