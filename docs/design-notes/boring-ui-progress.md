# boring-ui — implementation progress

Companion to [`boring-ui-draft.md`](boring-ui-draft.md) (the design). This file tracks **what is
implemented, what is decided, and what to do next**, so work can resume on another machine.
Last updated: 2026-10-11.

The draft's "Implementation sequence" has 6 layers. **Layer 1 (language prerequisites)** is mostly
done (view lowering below); **layer 2 (headless reactive core)** has a first slice in the sibling
project `../boring-ui` (see "Layer 2" below); layers 3–6 (layout/scene, desktop host, controls, Todo
acceptance app) have not been started.

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
- `body()` needs the `View` trait: with `use boring.ui.*` it works (next section); without it the
  `impl View` return type is unresolved. `view_state_lowering`/`view_body_lowering` predate the stdlib and
  deliberately avoid `body()`.
- No write-site invalidation: the runtime must treat any callback that received `__st` as a possible
  writer (conservative).
- Content-mutable state slots (`state mut [T] f`) lower to a plain `mut` field of `<View>__State`: the
  interpreter now honours in-place mutation (`f.push(..)`, `f[i] = ..`) through a `mut`-only field while
  still rejecting whole-field replacement, so the draft's "`mut` default forbids replacement" holds.
- `bind`/`env` views stay `view` declarations (they now also build under `boring build`, see "Known gaps of `bind`").

### `boring.ui` stdlib: the `View` trait and native facades (2026-10-12, `stdlib/ui.br`)

`use boring.ui.*` provides `trait View` (an empty marker), `enum AxisAlignment`, and the facade views
`Text(content)`, `Column`/`Row(alignment, spacing, [View] children)` declared with `body(): native`
exactly as in the draft's "Boring facade for Rust primitives". Until the runtime exists they lower
to plain structs `as View` (the native placeholder body is dropped; native views may not declare
`state`). Any non-native view in a program that has the `View` trait in scope gets `as View` added, so
`body()` composes them and returns `impl View`; trailing array-block sugar collects children into
`[View]`. Case: `boring_stdlib_ui` (interpreter + 4 transpile configs).

Supporting changes made along the way:
- **Module loaders lower views too**: `lower_view::lower_module` (= `view_perm` + `lower_views`) is
  applied right after parsing in the interpreter's three `use` loaders and the transpiler's sibling-file
  and stdlib inliners (the main pipeline never sees imported source). Before this, a `view` in an
  imported file or `boring.*` module was never lowered.
- **Manual `struct X as View` rejection moved** from `check_struct` to `check_view_declarations`: it runs
  before lowering, which now makes every view a `View`-conforming struct itself.
- **Trailing array-block label**: when a call already has labeled args and the callee is a struct, the
  synthesized trailing array is passed labeled with the struct's last field name (a positional arg after
  labeled ones was matched by position — `Column(spacing = 8.0):` put the spacing into `alignment`).

Omitted defaulted fields in labeled/positional struct construction are now filled from their declared
`= default` by the transpiler, so `Column(spacing = 8.0):` and bare `Row:` work from `boring build`. Not done: `TextField` and other `bind`-carrying components, `Button`, modifiers, the
`AnyView`-free `[View]` boxing story beyond what the transpiler already does, book.md §15 entry.

### `bind` and `env` lowering (2026-10-13, `src/lower_view.rs`)

Same explicit-context style as `state`:
- **`env T name`** → read-only field of a generated `V__Env` struct, passed as a second trailing
  parameter `__env` (after `__st`); `self.m()` forwards both. The runtime will fill it from the nearest
  providers. Interpreter + all 4 transpile configs (`view_env_only`).
- **`bind T name`** → input field `name: Slot<T>'actor` (`Slot` is the new shared cell in `boring.ui`;
  `use boring.ui.*` is required). Bare reads/writes of `name` become `name.value`. At a call
  `Child(name = x)` whose callee field is a `bind`, the argument must be a `state`/`bind` field of the
  enclosing view (literals, expressions, locals, projections → error "needs a `state` or `bind` field")
  and the *slot* is passed. A `state` handed to some child's `bind` is itself stored as
  `Slot<T>'actor` in `V__State`, parent accesses go through `.value`, so a child's write is visible to
  the parent. Callee info comes from the program plus embedded `boring.*` modules named by a `use`
  (sibling-file views are not seen). Native views may carry `bind` (new `TextField` facade). Cases:
  `view_bind_env` (parent state → child bind → native `TextField`, write-through verified),
  `view_bind_literal_arg`, `view_bind_without_ui`.
- `lower_view.rs` was restructured around a per-view `Ctx`; field accesses are rewritten in three passes
  (shadow check, rewrite, strip `@` keep-markers that prevent re-wrapping a rewritten name).

Known gaps of `bind`:
- ~~Interpreter only~~ — fixed: `view_bind_env` now also builds under `boring build` (all 4 transpile configs).
  Three transpiler gaps were closed for it (case `actor_field_through`): a field read/written/method-called
  *through* an `'actor`/`'guard` struct field (`bind.value`, `st.name.value`, `bind.value.rename(s)`) now locks
  the handle (`resolve_lock_field_ref`, `emit_methods.rs`); passing an already-`'actor` field as a labeled
  constructor argument for an `'actor` field clones the handle instead of re-wrapping it; and a string
  literal for a generic `T` field (`Slot(value = "a")`) becomes `Rc<str>`/`Arc<str>`, not `&'static str`.
- Permission levels are not enforced (the slot is `var mut`), no projections (`profile.name`, keyed loop
  elements), a bound `state let` is rejected, no generation checks / stale-handle diagnostics (that belongs to
  the runtime's real handles).

### Not done in layer 1

1. **View lowering, remaining parts**: `bind` slot handles with generation checks, `env` lookup,
   parent-supplied seeds (`Counter(count = expensive())`), `bind` projections/keyed elements and
   precise write-site invalidation.
2. Runtime duplicate-key validation (belongs to layer 2).
3. Native-body (`body(): native`) facade → runtime adapter linkage (layer 2/4).

## Layer 2 — headless reactive core (`../boring-ui`, 2026-10-14)

**Decision (user):** write the core in pure Boring as a sibling project `boring-ui/` (own git repo,
plain `src/` consumed via `[deps] ui = "../boring-ui"`, like `boring-numlib`), not as a Rust crate.
First slice, `src/core.br`: `Desc` (erased tree shape: kind, site, key, children), `Mounted`, `Handle`,
`Runtime` (reconciliation, hook log, handles with recycled ids + generations, duplicate-key validation
before mutation, event transactions, update loop with the 64-pass guard), `Builder` trait. 2 test files,
all green under `boring run`. Details and semantics: `boring-ui/README.md`.

**Typed state is now connected (2026-10-13).** The downcast landed (`trait_object as Struct` → `Struct?`), so:
`boring.ui`'s `View` trait gained runtime-facing methods, all with defaults (`viewKind`, `viewChildren`,
`viewKey`/`viewSite`, `isComposite`, `newState`, `expand`), plus `StateBox`/`NoState`; `lower_view` generates
them for every lowered view (`V__State as StateBox`; `expand` = `mut s = st as V__State else …; self.body(s)`)
and `KeyedGroup(key, children, site)` is a native facade. The array-block lowering now passes the key as a
string (`KeyedGroup("{key}", items)`). In `boring-ui`, `Desc` can carry a composite `View`, `Mounted` holds the
`StateBox`, and `describe`/`ViewRoot` adapt `ViewBuilder`s. `tests/core_views_test.br` proves typed state
persists across rebuilds/reorders and dies with its identity.

**Transpiler status (corrected 2026-10-16).** The earlier "interpreter only" note was wrong/stale: the generated methods
are emitted under `boring build` too; the build switch only turns `body()`'s `impl View` into `Arc<dyn View>`. All of
`boring-ui/tests/*_test.br` (incl. the Todo app) transpile, compile and pass under `boring build` with output identical
to `boring run` (`boring-ui/tests/run_tests_transpiled.sh`). Remaining: a real Rust host/renderer on top. Gotchas met: a view named `Screen` collides with the interpreter's
GPU `Screen` builtin; a top-level script variable named like a struct field (`root`) shadows the implicit
`self.root` inside methods.

**Stable sites (2026-10-13).** The array-block lowering now tags every element of a `[View]` block that has
control flow as `Sited(site = n, children = [e])` (static preorder position in the source; only when
`boring.ui`'s `Sited` is in scope, so ordinary trait-object blocks are untouched; unkeyed `for` items stay
positional). `Sited` is a native facade in `stdlib/ui.br`; `describe` in `boring-ui` flattens it. Cases:
`view_sited_blocks` (boring), `core_sites_test.br` (boring-ui, with a negative control: without sites the
test fails with duplicate keys / remounted counters).

**Layer 3 started (2026-10-13).** `boring-ui/src/layout.br`: size negotiation + placement (Text with fixture
font, Padding, Frame, Priority, Spacer, Row, Column; deterministic stack allocation, modifier order) over the
mounted tree, returning a `Placed` tree; tests `layout_test.br` (draft examples) and `layout_views_test.br`
(composites, keyed groups, unspecified proposals). Needed in `boring`: native views expose their scalar fields
through a generated `viewProps()` (`View` trait method), `Padding`/`Frame`/`Priority`/`Spacer` facades in
`boring.ui` (case `view_props`). Still missing for layer 3: fill/Layer/clip/Scroll, baselines, display list,
hit-test data, accessibility nodes (see `boring-ui/README.md`).

**Scene slice (2026-10-13).** `boring-ui/src/scene.br`: display list (`FillRect`/`DrawText`/`PushClip`/`PopClip`) and hit
testing (path root→target; layout bounds ∩ ancestor clips; front-to-back) over `Placed`; `Background`/`Clip`
wrapper facades in `boring.ui`; `Placed` now carries the node's props. Test `scene_test.br`.

**Accessibility slice (2026-10-13).** `boring-ui/src/semantics.br`: semantic tree over `Placed` (roles staticText/
heading/textField/group; name/value/hint/level/bounds/actions; transparent layout wrappers; ordered modifiers;
hidden subtrees; validation errors). `boring.ui` gained the `Accessibility{Label,Hint,Heading,Hidden,Group}`
wrappers, and `viewProps()` now exposes a `bind` field's *current value* (a lowered `Slot`) so a `TextField`'s
value is readable. Test `semantics_test.br`. With layout, scene and semantics, the headless part of layer 3 has a
first slice; remaining: the primitives/modifiers listed in `boring-ui/README.md`.

**Controls + events slice (2026-10-14, layer 4 start).** `boring.ui`: `UIAction` alias, `View.viewAct(action, arg)`
(generated for native controls: `onActivate` → activate, bool/string `value` bind → toggle/setText),
`Button`/`Toggle` facades. `boring-ui`: `src/events.br` (pointer press, keys, typing, semantic `act`, focus order,
one transaction + one update pass per event), `Runtime.focus`/`find`/`invalidateRoot`, controls in layout/scene/
semantics. Test `events_test.br`; boring case `view_actions`. Callbacks capture the body's `__st` (reference
semantics, interpreter only). Gap found (since fixed in the lexer — a closing `)` now ends the closure body's block, see
`Lexer::scan_line_delims`): a multi-line closure argument inside an array-block item did not parse. Pitfalls hit: `native` is a keyword, trait defaults for `void` need a
block body, a field closure is called via `let f = field; f()` inside methods.

**Todo application (2026-10-14).** `boring-ui/examples/todo` + `tests/todo_test.br`: the first end-to-end app, headless,
through core → layout → scene → semantics → events (details and deviations in `boring-ui/README.md`). Needed in
`boring`: (1) module loaders (`use` of a file / dep / stdlib) now run the array-block pass on the imported module
(before: `Column:` blocks in an imported file reached the evaluator undesugared; case in
`tests/imported_array_block.rs`); (2) a `bind` argument may be a field projection (`item.completed`), passed
through untouched — it must already evaluate to a `Slot`. Pitfalls found (one task filed): a global function named
like a struct field shadows the implicit-self field in the interpreter (`key`, `focus`, `root` all bit), and a
function-typed field can't be called as `onX(...)` inside a method (use `let f = onX; f()`).

**Lifecycle hooks and tasks (2026-10-17).** `boring.ui` gained `OnMount`/`OnUnmount`/`OnTask` facades, `TaskRun`, the
`TaskAction` alias and `View.viewTask`; `lower_view` generates `viewAct("mount"/"unmount")` and `viewTask` from the
facade fields (case `view_lifecycle_actions`). `boring-ui/src/core.br`: mount hooks after commit in preorder, unmount
hooks innermost-first with task cancellation before the first one, task start only after a stable update, `id` restarts
with per-node task-run generations, `isCurrent`/`deliver` (details in `boring-ui/README.md`); test `core_hooks_test.br`
passes under `boring run` and `boring build`. Transpiler gaps found (filed as tasks, worked around): string literals/locals
passed to a trait-object method are emitted as `&str`, not `Arc<str>`; `.join` on a `state` array field; a `let`-bound
closure capturing view state is not `'static` (inline closures are fine).

**Method-form modifiers (2026-10-17).** `src/desugar_view_modifiers.rs` (run first thing in `lower_views`): a method call whose
name is a modifier view's lowerCamel name (`padding`, `frame`, `background`, `onMount`, …) on a receiver that is
syntactically a view construction (or an already-rewritten modifier) becomes the ordered wrapper
`View(<positional args as the view's input fields>, children = [receiver])`. No type inference, hence the syntactic
receiver rule. Cases `view_method_modifiers` (interpreter + 4 transpile configs); `boring-ui/tests/layout_test.br` has the
draft's two order examples in method form. Not done: convenience modifiers (`width(200)` → `frame`), modifiers on
helper-function results / `<View>` locals, generic `ext View` modifiers.

Design choices not in the draft: ids are recycled only *after* the pass that released them (a node
created in the same pass gets a fresh id); an invalid description refuses the *whole* update, not just
the offending `for` subtree; hooks are an event log (`events`), not user callbacks; update-pass counter is
`passesRun`.

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
