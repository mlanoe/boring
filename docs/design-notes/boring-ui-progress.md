# boring-ui — implementation progress

Companion to [`boring-ui-draft.md`](boring-ui-draft.md) (the design). This file tracks **what is
implemented, what is decided, and what to do next**, so work can resume on another machine.
Last updated: 2026-10-11.

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
| `view` / `state` / `bind` / `env` / `body()` | **parse + checker + default permissions + state lowering (slice 1) done** | `src/view_perm.rs`, `src/lower_view.rs` (see below). `bind`/`env`/native views not lowered. |

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

### Default-permission resolution (`src/view_perm.rs`, 2026-10-11)

`resolve_view_permissions` runs on the merged `Program` right after `desugar_inject` (inside
`desugar_inject_or_exit` and the `boring build` loader). For each `state`/`bind` field written
without `let`/`var`/`mut` it rewrites the field into the encoding the explicit spelling would
produce — `mutable = true` for the `var` category, `Type::Mut` for the `mut` category — and leaves
`explicit_perm = false`. Classification is the draft's closed table: primitives/strings/enums/
newtypes/tuples/fn values/optionals → `var`; structs/arrays/dicts/sets → `mut`; trait objects,
generic parameters and unknown/external names → hard error asking for an explicit permission.
`'observed` fields and `env` fields are left untouched. The parser records primitives as
`Type::Named("int")`, so the pass matches primitive names explicitly. Unit tests: `view_perm::tests`.
Not done: a generic parameter whose `where` constraint proves one category (currently always an
error — conservative).

### `IdentityKey` bound check + omitted `with` (`src/desugar_array_block.rs`, 2026-10-11)

Done in the array-block desugar (it consumes the key before the checker runs, and the checker has
no type inference). `Signatures` now carries a struct table (`defs`: fields, conformances,
zero-arg method return types) and a flat per-body `locals` environment (params, enclosing struct
fields, annotated `let`s, keyed-loop variables). `resolve_identity_keys` runs on each collect
block: an explicit key whose type is *statically known* to be a float, dict, set, fn value, tuple/
option of those, or `'actor`/`'guard`/`'observed` is a hard error ("not an `IdentityKey`"); an omitted
`with` over `[T]` where `T` declares `as Identifiable` lowers to `with item.id()`. Cases:
`array_block_keyed_for_float_key`, `array_block_keyed_for_identifiable`.
**Best-effort by design:** an unknown type never errors (no false positives), so a key like
`f(x)` or a field of an un-annotated local is not checked. `Identifiable` conformance via `ext`
is not seen. Closing those needs real type inference in the checker.

### View lowering, slice 1: `state` (2026-10-11, `src/lower_view.rs` + `src/ast_walk.rs`)

**Decision (user, 2026-10-11):** lower to ordinary Boring structs with an explicit context
parameter, *not* wait for the layer-2 Rust runtime. Runs after `view_perm`, before the checker
(first validating the declarative view rules via `checker::check_view_declarations`, since the
lowered program no longer carries `state`/`bind`/`env` markers).

`view Badge` with inputs + `state` fields becomes `struct Badge__State` (the mounted-state layout,
`state` fields in order, `state let` → immutable field) and `struct Badge` (inputs only, the
ephemeral description). Every method and `body()` gains a trailing `mut Badge__State __st`
parameter and becomes `req`; bare reads/writes of a state field and `self.<state>` become
`__st.<state>`; `self.method(..)` forwards `__st`. The default seed is `Badge__State()`. Validation:
a `state` needs an initializer; a local/param/loop var/closure param may not shadow a state name.
`src/ast_walk.rs` is a new generic mutable AST walker (pre-order, expr+stmt callbacks) — the repo had
none; reusable for later passes. Cases: `view_state_lowering` (interp + all 4 transpile configs),
`view_body_lowering`, `view_state_shadow`, `view_state_no_initializer`.

Known limits of slice 1:
- `body()` lowers and type-checks but **cannot be called or transpiled** yet: its return type is
  `impl View` and no `View` trait exists (needs the UI stdlib). `view_state_lowering` therefore has no
  `body`; `view_body_lowering` (interpreter only) never calls it.
- No write-site invalidation: the runtime must treat any callback that received `__st` as a possible
  writer (conservative).
- Content-mutable state slots (`state mut [T] f`) lower to a plain `mut` field of `<View>__State`: the
  interpreter now honours in-place mutation (`f.push(..)`, `f[i] = ..`) through a `mut`-only field while
  still rejecting whole-field replacement, so the draft's "`mut` default forbids replacement" holds.
- `bind`/`env`/native views stay `view` declarations; `boring build` still errors for them.

### Not done in layer 1

1. **View lowering, remaining parts**: `bind` slot handles with generation checks, `env` lookup,
   parent-supplied seeds (`Counter(count = expensive())`), precise write-site invalidation, native
   views (`body(): native`), and the `View` trait / `<View>` body type (see slice 1 below).
2. Runtime duplicate-key validation (belongs to layer 2).
3. Native-body (`body(): native`) facade → runtime adapter linkage (layer 2/4).

## Next steps (suggested order)

1. Layer 1 remainder: view lowering (item 1 above).
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
