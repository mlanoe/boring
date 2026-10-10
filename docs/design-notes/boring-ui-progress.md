# boring-ui — implementation progress

Companion to [`boring-ui-draft.md`](boring-ui-draft.md) (the design). This file tracks **what is
implemented, what is decided, and what to do next**, so work can resume on another machine.
Last updated: 2026-10-10 (all work below is committed on `main`).

The draft's "Implementation sequence" has 6 layers. Only **layer 1 (language prerequisites)** has
been started; layers 2–6 (headless reactive core, layout/scene, desktop host, controls, Todo
acceptance app) have not.

## Layer 1 — language prerequisites

| Item | Status | Notes |
|---|---|---|
| Imported array-block callables | **done** (commits `51b6f9d6`, `a11b94c6`) | `desugar_array_block_with_imports` in `src/desugar_array_block.rs`; sibling files, `[deps]`, `boring.*` stdlib, `use … as` aliases. Tests: `tests/imported_array_block.rs`. |
| `array.move(from, to)` / `removeIndices(idxs)` | **done** (`a11b94c6`) | Interpreter (`src/interpreter/methods.rs`), transpiler prelude (`BoringArrayIndex::move_item`/`remove_indices`, `src/transpiler/mod.rs`), checker mutability gate. Case: `tests/cases/array_move_remove_indices.br`. |
| `for x in xs with <key>:` | **done, syntax + lowering** (`a11b94c6`) | `ForStmt.key`; in a collect block each iteration lowers to `KeyedGroup(key, [items])` (a library symbol the caller must have in scope — **my design choice, not in the draft; revisit if the runtime wants another contract**). Checker rejects it outside a collect block. Cases: `array_block_keyed_for*`. |
| `IdentityKey` / `Identifiable` | **done** (`a11b94c6`) | `use boring.identity.*` (`stdlib/identity.br`). Transpiler emits a blanket impl and emits trait-bound assoc types by name; interpreter checks `type ID as IdentityKey` (rejects floats/dicts/sets/fn). Cases: `boring_stdlib_identity*`. |
| `view` / `state` / `bind` / `env` / `body()` | **parse + checker done** (`e0f4173d`) | Declarations only; no lowering. |

### `view` parse + checker (commit `e0f4173d`)

Files: `src/ast/mod.rs` (`StructDecl.is_view`, `FieldDecl.view_kind` / `explicit_perm`,
`ViewFieldKind`), `src/parser/mod.rs` (`is_view_decl_start`, `parse_struct_or_view`, view arms in
`parse_struct_body`), `src/parser/parse_fn.rs` (`parse_view_field`, `parse_view_body`),
`src/checker/mod.rs` (`check_view_decl`, `view_tests`), `src/transpiler/emit_struct.rs` (explicit
"cannot be transpiled yet" error), `src/interpreter/eval_gpu.rs`, `tests/cases/view_decl_parse.*`.

Verified on `main` at that commit: `cargo test --bin boring` (1011), `--test run` (234),
`--test imported_array_block`, `--test dependency_injection`, and the **full** `--test transpile`
(846 passed, 14 ignored) — all green.

Contextual keywords: `view`, `state`, `bind`, `env` stay ordinary identifiers everywhere except as
`view <Ident> (':' | 'as' | '<')` at item position and as field prefixes inside a view body.

Checker rules in place: no `pub state/bind/env`; no `'observed` state; no `bind let`; no
initializer on `bind`/`env`; `env var`/`env mut` rejected (`env let` accepted as redundant); a
trait-object/opaque `state`/`bind` needs an explicit permission; `body()` takes no parameters and
appears at most once; `struct X as View` rejected.

### Not done in layer 1

1. **Default-permission resolution** for `state`/`bind` without `let`/`var`/`mut` (draft, "Implicit and
   explicit binding permissions"): `var` for primitives/strings/enums/options/tuples/newtypes/fn
   values, `mut` for structs/arrays/dicts/sets. The parser only records `explicit_perm`; the
   resolution needs type classification (struct vs enum) → do it in the checker/lowering with the
   program's declaration table.
2. **View lowering**: typed state-slot layout, seed closures (lazy, evaluated once at mount),
   `bind` slot handles with generation checks, `env` lookup, callback `self` as mounted-view
   context, invalidation call at write sites. `boring build` currently errors for any `view`.
3. **`IdentityKey` bound check** on the `with` key expression (needs expression type info), and
   **omitting `with` for an `Identifiable` element** (lower to `item.id()`).
4. Runtime duplicate-key validation (belongs to layer 2).
5. Native-body (`body(): native`) facade → runtime adapter linkage (layer 2/4).

## Next steps (suggested order)

1. Layer 1 remainder: items 1–3 above.
2. Layer 2: headless reactive core as a Rust crate/module (mounted tree, typed slots, reconciliation,
   event transactions, lifecycle, 64-pass loop guard) — independent of the language work, so it can
   proceed in parallel.

## Decisions made while implementing

- **Labeled arguments use `=`, not `:`** (`Column(spacing = 8)`, `Todo(id = 1, …)`). `Name(x: 1)` is
  parsed as a closure shorthand; the draft was rewritten accordingly.
- `view` is a contextual keyword, not a reserved word.
- `KeyedGroup` is the lowering target of a keyed `for` (see table above).

## Known boring bugs found along the way (filed as tasks)

- Uppercase-led bare function shorthand with a typed parameter did not parse — **fixed** (`b845606d`).
- `use … as` aliases not followed by array-block resolution — **fixed** (`51b6f9d6`).
- `boring run` overflowed the stack when a struct had a field and a method of the same name (e.g.
  `pub int id` + `req int id()`) — **fixed** (`445a1c1d`: a stored field wins over a same-named
  zero-param method in `get_field`; case `tests/cases/field_method_same_name`).
  `tests/cases/boring_stdlib_identity.br` now uses the draft's `id` field + `id()` method shape.
- Gotcha: a single-letter uppercase struct name (`T`) is treated as a generic parameter —
  `T(age = 4)` does not construct a struct. Use real names in tests.

## Useful commands

```bash
cargo build
cargo test --bin boring                                  # unit tests (incl. checker::view_tests)
cargo test --test run                                    # interpreter cases (tests/cases/*.br)
cargo test --test imported_array_block
cargo test --test transpile -- array_move array_block_keyed boring_stdlib_identity   # quick subset
cargo test --test transpile                              # full suite, ~10 min of test time + compile — run in background
```
