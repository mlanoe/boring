# Labelled generic arguments — draft

Status: **type and const arguments implemented** (parser-time resolution, `src/parser/generic_args.rs`; the resolution site is a lenient first parse + resolving second parse, see `docs/book.md` "Labelled generic arguments"); the qualifier form `label'value` and the `'Q` declaration remain a draft. Prerequisite (phase 0) of
[qualifier-generic-params.md](qualifier-generic-params.md), useful on its own: today a generic
argument list is positional only (`Pair<int, string>`), which says nothing about what each
argument is for, and cannot name a qualifier at all.

## 1. Syntax

Declaration (unchanged, plus the qualifier parameter of the other draft):

```boring
struct Cache<T, U as Display, 'Q as 'new>:
```

Arguments, at every use:

| Parameter kind | Positional (today) | Labelled (new) |
|---|---|---|
| type `T` | `Cache<Point>` | `Cache<T = Point>` |
| const `int N` | `Grid<4>` | `Grid<N = 4>` |
| qualifier `'Q` | — (no positional form) | `Cache<Q'actor>` |

- A type or const argument is labelled with `label = value`; a **qualifier** argument is written
  `label'value` (tick glued to the label, the same glued shape as `T'actor`). A qualifier
  parameter has no positional form: its label is always written.
- Positional arguments come first, then labelled ones (`Cache<Point, U = string, Q'actor>`), as
  for call arguments. A parameter cannot be given twice; labels may come in any order.
- Every parameter must be given (no defaults in v1). Trailing comma allowed, like today.
- Same list in every position a generic argument list appears: constructor/call
  (`Cache<T = Point, Q'actor>()`), method turbofish (`obj.m<T = int>()`), type position
  (`let Cache<T = Point, Q'actor> c`, fields, parameters, returns), `use Name<T> as Type` aliases,
  `enum`/`ext`/`trait` headers.

`enum` and `ext` headers use the same lists as `struct`. In an `ext` header the arguments after the
extended name are either parameters being declared (`ext Pair<T, U>:`, `ext Board<'elem>:`) or
concrete arguments (`ext Pair<T = int, U = string>:`, `ext Board<elem'actor>:`, a block for that
instance only); a declaration is told from an argument the same way as elsewhere, by the `'`
prefix for qualifiers and by whether the name is already a type in scope for types.

### The `T'actor` ambiguity

`Container<T'actor>` already means "type argument `T`, qualified `'actor`" (T an enclosing type
parameter). With labelled qualifier arguments the same text can mean "qualifier parameter `T` of
`Container` is `actor`". It is resolved **against the callee's declaration**: if `Container`
declares a qualifier parameter `'T`, `T'actor` is that label; otherwise it is an ordinary
qualified type argument (today's meaning, unchanged). A struct that declares both a type
parameter and a qualifier parameter of the same name is a declaration error, so the reading is
never ambiguous for a given callee. This is why the parser cannot decide it alone (see 3).

## 2. Meaning

Labels resolve to positions; everything downstream keeps seeing positional arguments.
Normalising immediately keeps the interpreter, the checker, the monomorphizer
(`src/transpiler/monomorphize.rs`, `GenericCall`/`Type::Generic` carry `Vec<Type>`) and the
emitters unchanged: `GenericCall` has 105 occurrences in 38 files and `Type::Generic` 63 in 23, so
changing their shape is the expensive path. A qualifier argument needs a positional carrier;
the draft proposes a new `Type` variant for "an `OwnerQual` in argument position" (matches on
`Type` mostly have wildcard arms, to be confirmed by a build).

## 3. Impacts

- **Parser.** `is_generic_call_ahead` (`src/parser/mod.rs`) decides `f<...>(` vs a comparison by
  scanning to the matching `>` followed by `(`, and bails out on `=`: `Cache<T = Point>()` would
  be read as a comparison. It must tolerate `IDENT =` at depth 1 (`Eq` directly after an
  identifier directly after `<` or `,`) and `IDENT '` (qualifier argument); `f < T = x > (y)` is not a valid expression anyway (an assignment cannot be an operand of `<`), so accepting `=` there only changes inputs that did not parse before. Type-position lists (`Type::Generic`) have no such
  heuristic and just gain the labelled form. Needs a `GenericArg` parse step shared by the 4+
  places that build `type_args` (`parse_expr.rs` lines around 823, 920, 1782, and the type parser).
- **Resolution pass.** Labels cannot be resolved at parse time: the callee's declaration may be
  in another file. The module `infer_actor_params` shows one way (a parse-time pass with an import
  context reading `use`d files); `monomorphize.rs`'s `deep_pre_scan` already visits every reachable
  file once and holds a global registry of generic declarations. Decision needed: resolve in
  the parse-time pass with the import context, or in a post-merge pass (the GPU path already merges
  files in `parse_and_merge_program` before emitting; `boring run` loads files its own way, so both
  paths must run the same pass).
- **Checker.** Errors with positions: unknown label, label given twice, positional after
  labelled, missing parameter, arity, kind mismatch (`T = 'actor`, `Q = Point`, `Q'new` on a
  type parameter), qualifier outside the parameter's bound.
- **Interpreter.** Erases type arguments today; it must parse and accept the new form and report
  the same checker errors, so `boring run` and `boring build` agree.
- **Monomorphizer.** No change beyond receiving positional arguments, once resolution has run.
  Mangled names (`Pair_int_string`) use the positional order, so a labelled and a positional
  spelling of the same instance share one specialisation.
- **Lexer.** None for the arguments (`T = Point` and `Q'actor` have no tick-first or raw-string
  shapes); the declaration form `'Q` is covered in the other draft.
- **Docs and tooling.** `spec/grammar.bnf` (`generic_args`), book.md generics sections, Linguist
  grammar, CHANGELOG.

## 4. Tests

Parser: positional, labelled, mixed, reordered labels, trailing comma, each position (call,
method, type, alias, enum/ext), `a < b` / `a < b > (c)` comparisons unchanged. Resolution:
cross-file declaration, same instance spelled both ways (one specialisation), `T'actor` as label
vs as qualified type argument. Checker: each error above with its span. Parity: `run` vs `build`.

## 5. Open questions

- Defaults (`<T = Point>` in a declaration) are out of scope here, and `=` in a declaration would
  clash with the argument syntax; deferred.
- Whether a labelled argument may repeat a label in a nested list (`Pair<T = Cache<T = int>>`): yes,
  labels are scoped per list; a test pins it.
- Resolution site (parse-time with import context vs post-merge pass), see 3.
- The positional carrier for a qualifier argument: new `Type` variant vs a side table.
