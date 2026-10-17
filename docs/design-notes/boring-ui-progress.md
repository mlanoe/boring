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
passes under `boring run` and `boring build`. Three transpiler gaps found on the way (string literals passed to a trait-object method, `.join` on a `state` array
field, a `let`-bound closure capturing view state) were fixed on `boring` main (45f3aac5, 752498e4, 0d727020) and the
workarounds removed.

**Method-form modifiers (2026-10-17).** `src/desugar_view_modifiers.rs` (run first thing in `lower_views`): a method call whose
name is a modifier view's lowerCamel name (`padding`, `frame`, `background`, `onMount`, …) on a receiver that is
syntactically a view construction (or an already-rewritten modifier) becomes the ordered wrapper
`View(<positional args as the view's input fields>, children = [receiver])`. No type inference, hence the syntactic
receiver rule. Cases `view_method_modifiers` (interpreter + 4 transpile configs); `boring-ui/tests/layout_test.br` has the
draft's two order examples in method form. Not done: convenience modifiers (`width(200)` → `frame`), modifiers on
helper-function results / `<View>` locals, generic `ext View` modifiers.

**Layer 1 remainder (2026-10-17, projections below).** Parent-supplied state seeds (`Counter(count = expensive())`: each `state` field gets an
optional-in-practice `<state>__seed` closure input whose default reads `V__State()`'s own default, `newState()` builds the
state from the seeds; cases `view_state_seed`, `view_state_seed_values`) and `env` expansion (`EnvValue` marker trait,
`Provide`/`EnvMissing` facades, generated `usesEnv`/`expandIn`/`viewProvides`; case `view_env_provide`, `boring-ui` test
`core_env_test.br`) are done, interpreter and `boring build`. Still open: `bind` projections into keyed collection
elements (design: pass the collection slot + key, rewrite the child's bare `item` to `coll.value[indexOf(key)]`; needs a
stale-key policy) and write-site invalidation (conservative whole-window rebuild is allowed by the draft). A seed that is a
bare string literal is not supported under `boring build` yet (string literals returned from closures are emitted as `&str`;
transpiler task proposed).

**`bind` projections of keyed elements (2026-10-17, voie A).** `for item in items with item.id: TodoRow(item = item)` with
`bind Todo item` in the child: `lower_view` detects the keyed loop (the `KeyedGroup` key and the bind-target argument),
adds a hidden typed registry per site to the parent's state (`<View>__Proj<k>` records: collection slot, projection slot,
base copy, key) and rewrites the argument to a projection slot; `Runtime.commitProjections` runs before every update pass and
writes changed projections back by key (stale key: dropped and reported). Cases `view_bind_projection_no_key` (error);
`boring-ui/tests/core_projection_test.br` passes interpreted and under `boring build`. Findings: the interpreter aliases
structs and compares them by identity where transpiled output copies and compares contents (the commit compares printed
forms to be mode-independent); `boring build` cannot carry trait objects in a shared state or closures with loops, so the
design uses concrete per-site structs; a field named `done` is mis-emitted; two reads of one `'actor` slot in a single
expression deadlock (task proposed). Field projections (`item.title` into a native control's `bind string`) remain the
per-field `Slot` workaround.

**Layer 3, fourth slice (2026-10-17).** `boring.ui` gained `Layer`, `Fill`, `Scroll`, `Border`, `Opacity`, `Shadow`, `Offset`, the
`Alignment`/`ScrollAxis` enums and `Text.lineLimit`. `boring-ui`: layout for all of them (`Mounted.scroll` is mounted scroll
state, clamped by layout; `Runtime.scrollTo`), `events.scrollBy` (deepest-first consumption, remainder bubbles), scene commands
`StrokeRect`/`DrawShadow`/`PushOpacity`/`PopOpacity` and a clip for scroll viewports; tests `layout_extra_test`,
`scene_effects_test`, `scroll_test` (interpreter and `boring build`). Notes: the two-axis `Alignment` middle row is
`MiddleLeading`/`Middle`/`MiddleTrailing` (a `.Center` would clash with `AxisAlignment`); the fixture ellipsis is ASCII `...`
because `String.length` counts bytes in transpiled output and characters in the interpreter; a recursive `mut Mounted`
write does not persist under `boring build`, so scroll offsets are written by rebuilding the tree. Still open for layer 3:
baselines, device-pixel rounding, head/middle truncation, overlays, rounded/transformed clips, images.

**Layer 4 interaction slice (2026-10-17).** `boring.ui` gained `Disabled`, `DefaultAction`, `ButtonRole` (`Button.role`) and
`TextField.onSubmit` (an optional callback: `viewProps` skips it, the lowering generates `viewAct("submit")` and a new
`View.viewHandles(action)`). `boring-ui/src/events.br`: disabled subtrees (no focus/capture/action, skipped by Tab,
`SemNode.disabled`), pointer capture (`pointerDown` captures, `pointerUp` activates only over the captured control,
`pointerCancel`, `click`; `Runtime.capture` is cleared on unmount), Enter (TextField `onSubmit`, else the `DefaultAction`
button) and Escape (the `Cancel` button), TextField editing with caret and selection as mounted control state
(`typeText`, Backspace/Delete/arrows/Home/End/Shift variants/SelectAll, caret from a click; ASCII only). Tests
`events_extra_test.br`; `events_test`/`scroll_test`/`todo_test` now use `click` (a press alone no longer activates). The
projection test no longer needs the double-read workaround (deadlock fix 876f5518). Still open: IME/undo, `.focused(...)`
bindings, focus scopes, `focusOrder`/`focusable`, modal routing/`Sheet`, `List`, focus rings, callback error reporting.

**Theme slice (2026-10-17).** `boring.ui` gained `FontRole`, `ButtonStyle`, `Text.font` and `Button.style`. `boring-ui/src/theme.br`:
an immutable `Theme` of overrides on a light/dark base (`ColorRole`, `MetricRole`, font sizes, role→button-style map, builders,
`issues()` validation), installed with `Provide(value = theme)`; the core resolves the nearest theme onto every `Mounted`
(`Mounted.theme`, `Placed.theme`) and refuses to install an invalid one (reported in `errors`). Layout reads spacing, control and
text-field metrics, `MinimumHitSize` and per-role fonts off the node; painting resolves tokens to `Command.hex` and paints
buttons with their built-in style. Tests `theme_test.br`, `theme_scene_test.br`; `view_props.expected` gained `font=Body`.
Deviations: `PrimaryText`/`ControlBorder`/`ShadowColor`/`Danger` role names (variants named like views shadow them),
hex-string colors, entry-list token tables (dictionary iteration differs under `boring build`), `Theme` is a library name,
`Theme.validate` is `issues` (the interpreter's checker infers mutability by method name; `Runtime.validate` is mutating).
Still open: adaptive spacing pairs, text scale, per-token variants, custom style traits, inherited style providers.

**Animation slice (2026-10-17).** `boring.ui` gained `AnimationScope`, `Transition` and `TransitionKind` (`Identity`/`Fade`/`Scale`/`Slide`).
`boring-ui`: `src/anim.br` (curves, springs, reduced motion, linear-light premultiplied color blending), `src/animator.br`
(deterministic presentation between two committed layouts: transactions, scopes, retargeting from the current presentation,
transitions with presentation-only ghosts, reduced motion), core `animate`/`withoutAnimation`/`takeAnimation`/`takeTriggers`
and `AnimationScope` triggers, `Placed.alpha`/`ghost`/`live`, `PushOpacity` for group alpha, ghosts skipped by hit testing and
semantics. Tests `anim_test.br`, `animator_test.br` (interpreter and `boring build`, identical float results). Deviations: the
runtime still unmounts immediately (only the display of a removal is retained), `Fade`/`Slide` variant names, string scope
triggers. Transpiler gaps worked around (one task proposed): `.parseFloat()` on a non-variable string, forwarding a closure
parameter, returning an optional bound by `let`, assigning a plain value to an optional `var`; and `channel` is a builtin name
in the interpreter. Still open: completion callbacks, keyframes, matched geometry, `Sheet` transitions.

**Sheet, focus bindings, List and the acceptance app (2026-10-17).** `boring.ui` gained `Sheet`, `DefaultFocus`, `Focused`, `FocusedAs`,
`List` and `MoveAction`/`DeleteAction`; the lowering generates `viewAct` for `setBool`, `dismiss`, `select`, `move` and `delete`.
`boring-ui`: a closed sheet mounts nothing, a presented one is laid out in a presentation layer (`withSheets`), painted last, modal for hit
testing and semantics, traps focus and restores it (`Runtime.openSheets`/`focusReturn`/`syncModal`); `syncFocusBindings` reconciles native
focus and `Focused`/`FocusedAs` values per pass (per-node `syncedOn` state: arrival, request, clear, loss, first-request-wins);
`List` is a scrolling keyed-row viewport (`Placed.key`) with selection, `pruneSelection`, Up/Down/Delete/AltUp/AltDown and `onMove`/`onDelete`;
`focusControl` moves focus as one event. `examples/todo` is now the draft's acceptance app on these pieces; tests `sheet_test`,
`focus_binding_test`, `list_test`, `todo_test` (interpreter and `boring build`). Behaviour change: a press now focuses in its own event
(two update passes per click). Transpiler/interpreter gaps worked around or proposed: a `Handle?` passed to a method is moved, `var T? x = nil`
syntax, a derive-PartialEq decision that ignores field type derives and an interpreter bind-hop that loses slot writes (task proposed), and
name collisions in the interpreter (a global function named like a struct field, enum variants named like views). Still open: IME/undo, non-modal
focus scopes, multiple selection, focus rings, the renderer/host.

**Workarounds removed (2026-10-17).** With the four transpiler fixes on main (f9889c3a: `parseFloat`/`parseInt` on expression receivers, a closure
parameter forwarded to another function, an optional returned from a `let`, assignment to an optional `var`), `core.br` and `animator.br` went back
to the natural forms and the generated `List` actions parse their arguments inline (the `intOf` helper is gone). One workaround stays: an `if let`
on an optional moves it under `boring build`, so `collectScopes` pushes after recursing.

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
