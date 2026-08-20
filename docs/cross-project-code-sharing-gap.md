# Gap: no way to share Boring source across sibling projects

**Status (2026-08-20): SOLVED — both option 2 (named cross-project `[deps]`)
and option 3 (first-party `boring/stdlib/*.br`) below are now implemented.**

- **Option 3** (first-party stdlib): `use boring.<module>` is a real,
  working import in both `boring run` and `boring build`, backed by
  `stdlib/*.br` embedded into the compiler binary (`src/stdlib_embed.rs`);
  see `stdlib/README.md` and `docs/book.md` §15. A *fixed* set of modules
  shipped with the compiler itself — does not let one project import
  another arbitrary project's own source.
- **Option 2** (general cross-project sharing — the actual subject of this
  document, and what solves the `BigUint`-across-`scratch-boring`/
  `whisper-boring` motivating case below): a project declares named
  dependencies on other Boring projects in its own `boring.toml` `[deps]`
  section, each pointing at a plain directory (just a `src/` convention, no
  `boring.toml` required in the dependency itself) by local path —
  `numlib = "../boring-numlib"` or `numlib = { path = "../boring-numlib" }`.
  `use numlib.big_uint.*` then resolves real `.br` source from that
  directory, transpiled/interpreted together with the consumer — or by
  `git = "..."` (optionally `branch`/`tag`/`rev`), cloned into a persistent
  local cache (`src/git_deps.rs`, shells out to the `git` CLI — no new
  Cargo dependency) the first time it's needed and reused/refreshed on
  later builds, same as a real package manager's git dependency but without
  a lockfile. See `docs/book.md` §15's "Named cross-project dependencies"
  subsection for full syntax and `BoringToml::resolve_deps` (`src/main.rs`)
  for the resolution rules (reserved names `std`/`crate`/`boring`; no
  transitive resolution, same single-level rule `[external_types]`/
  `[derives]`'s own `include` already follows). Wired into all three
  consumers — the interpreter (`boring run`,
  `Interpreter::exec_named_dep_use`), the transpiler's std/tokio target
  (`boring build`, `emit_top.rs`'s `emit_use`/`deep_pre_scan`), and the
  GPU-target CLI merge path (`main.rs`'s `merge_into`) — the same three
  places option 3's `boring.*` mechanism needed fixing.

The rest of this document (below) is preserved as originally written,
recording the motivating case and the investigation that led here.

## The motivating case

`scratch-boring/boring/spikes/biguint_spike.br` implements a pure-Boring
`BigUint`/`BigInt`/`BigFraction`/`ScratchNumber` exact-arithmetic tower
(113 passing assertions under `boring run`), written to replace that
project's hand-written-Rust `value_rt.rs` — the effort drove 12 compiler
bugs to a fix along the way. It's now wired into
`scratch-boring/boring/scratch.br` for real.

The natural next question — "should `BigUint`/`BigInt`/`BigFraction` move
into a shared stdlib so other Boring projects don't have to copy-paste
~1000 lines of tested numeric code?" — exposed that **Boring currently has
no mechanism for one Boring project to import `.br` source from another.**
Today the code exists in exactly two places, both copies: the original spike
file (kept as a historical/reference copy) and a second, adapted copy
embedded directly in `scratch.br` (renamed methods to route around
now-fixed compiler bugs, plus a few extra conversions `scratch.br` needed).
A future project (a Boring-authored interpreter, a game needing precise
physics, `whisper-boring` if it ever needs exact reductions) would have to
copy a third time.

## What exists today, and why it doesn't solve this

### `boring/stdlib/*.br` — was dead, now wired in for real (2026-08-20)

**Historical note (accurate as of the original 2026-08-20 investigation,
before the same-day fix below):** the `boring` repo root's `stdlib/`
directory (`array.br`, `builtins.br`, `channel.br`, `collections.br`,
`dict.br`, `future.br`, `string.br`, `task.br` — 662 lines total) was pure
doc/stub content, never loaded by anything — grepping `src/` and `tests/`
for those filenames returned zero hits. `use std.collections.HashMap`
resolved to Rust's real stdlib; any other `use module.item` fell through to
a generic filesystem loader resolving relative to the current project only,
with no fixed `stdlib/` search path and no `"boring"`-prefix handling
anywhere.

**Current state:** fixed the same day. `stdlib/` now holds only genuinely
*loadable* `boring.*` modules — the always-native content that used to live
in `array.br`/`builtins.br`/`channel.br`/`dict.br`/`future.br`/`string.br`
(all of it already unconditionally global, so it was never truly
"importable" in the first place) was corrected for drift and folded into
`docs/book.md` §15's "Built-in Types & Functions Reference" instead.
`use boring.<module>` resolves against `stdlib/*.br` **embedded into the
compiler binary** (`src/stdlib_embed.rs`, via `include_str!` — the same
pattern `interpreter::gpu_profile` already used for built-in GPU profiles)
in three independent places that each needed their own fix: the interpreter
(`exec_use_decl`/`exec_boring_stdlib_use`, `src/interpreter/*.rs`), the
transpiler's std/tokio target (`emit_use`/`deep_pre_scan`/
`inline_boring_stdlib_use`, `src/transpiler/emit_top.rs`), and the GPU-target
CLI merge path (`merge_stdlib_into`, `src/main.rs`). An unrecognized
`boring.*` module name is a hard compiler error in all three, unlike the
generic filesystem loader's silent no-op (there's no legitimate "native
Rust module" fallback for a `boring.*` path — `boring` is never a real
crate). See `stdlib/README.md` for the current module list and how to add
another.

There is a *separate*, unrelated `boring/interpreter/stdlib.br` (inside the
self-hosted-interpreter-written-in-Boring subproject, `boring/interpreter/`)
that *is* real and wired in (`main.br`'s `use stdlib` → its own
`register_stdlib(interp)`) — but that's the runtime prelude for programs run
*by* the self-hosted interpreter, not a stdlib for programs written *in*
Boring generally. Easy to conflate the two by name; they're unrelated.

### `docs/library-distribution.md` — addresses a different, narrower problem

That document (status: **Draft**, "no decision is final") is about
distributing a Boring library to *third parties* — the current position is
"follow Rust": ship the generated Rust crate, not `.br` source, and lean on
Rust's own crate ecosystem/`Cargo.toml` `[dependencies]` for reuse.

That's a heavier, differently-shaped problem (confidentiality, ABI
stability, ownership-qualifier inference across a compiled boundary) than
what this gap is actually about: **the same author's own sibling projects
wanting to share plain `.br` source with no compiled-artifact/registry
concerns at all.** `docs/library-distribution.md`'s "follow Rust" answer,
taken literally, would mean: compile `BigUint` to a Rust crate, publish it
(even just locally via a path dependency), and add it to each consuming
project's `Cargo.toml` — which defeats the actual goal here (see
`boring-goal-pure-boring-rust-parity` project memory): the point was to stop
depending on a Rust crate (`num-bigint`) for this, not to depend on a
different, self-published one instead.

### The one existing precedent: `boring-bevylib` — config only, not code

`boring-bevylib` (a sibling repo to `scratch-boring`/`breakout-boring`) is
the only existing example of sharing anything across Boring projects. It
shares `boring.toml` **config** (`[external_types] include =
"../boring-bevylib/external_types.toml"`, a TOML-level `include` key added
to `boring`'s own `BoringToml` parser specifically for this) — not Boring
*source code*. Its own `src/lib.rs` is still just a doc comment: "no shared
Rust code has been extracted yet... scaffolded and ready to receive real
shared code... once a second real game actually needs something
`breakout-boring` already has." That day has now effectively arrived, just
for exact-arithmetic code rather than Bevy glue, and the answer isn't
obviously "put it in `boring-bevylib`" since that repo's whole reason to
exist is Bevy-specific.

## Options (not decided, not implemented)

1. **Vendoring convention** — a new sibling repo (e.g. `boring-numlib`)
   holding the canonical `.br` source, with consuming projects literally
   copying it in (as `scratch.br` already did from the spike) plus a
   regen/sync script, same spirit as `boring-bevylib`'s `external_types.toml`
   `include` but for actual code instead of config. Cheap, no compiler
   changes, but still copy-paste under the hood — drift between copies is
   possible if a consuming project patches its local copy (as `scratch.br`
   already had to, for the now-fixed transpiler bugs) and the sync script
   doesn't reconcile that.
2. ~~**A real cross-project `.br` import mechanism**~~ — **done** (2026-08-20,
   see above): `boring.toml` `[deps]` + `use <name>.xxx`, path *and* git
   (`branch`/`tag`/`rev`, cached locally). Chose the "named prefix,
   configured in `boring.toml`" shape over a flat `BORING_PATH`-style
   search-list addition specifically for provenance (`use numlib.BigUint`
   says where it's from) and to avoid filename collisions between two
   unrelated dependencies. Whichever source a dependency resolves from
   (local path or git clone), its `.br` source is spliced straight into the
   consumer's *generated Rust project* as its own `.rs` file — same
   mechanism `boring build` already used for local sibling-file `use`
   imports (`inline_boring_use`, `src/transpiler/emit_top.rs`) — so the
   generated project is self-contained: `cargo build` on it never touches
   `boring.toml`, the dependency's path, or the git cache again, only
   `boring build` (the transpile step) does.
3. ~~**Wire up `boring/stdlib/*.br` for real**~~ — **done** (2026-08-20, see
   above).

Option 1 (vendoring/copy-paste with a sync script) is superseded by option
2 now being real — no reason to prefer manual copying over a declared
`[deps]` entry going forward. This file is kept as the historical record of
the investigation and the options considered, not because the gap is still
open.

## Future work — gaps vs. a real package manager (2026-08-20)

`[deps]` solves the actual motivating case (sharing `.br` source between
sibling projects) but is deliberately minimal next to Cargo/npm/Go
modules. Recorded here so a future session doesn't have to re-derive the
comparison. Roughly in priority order for Boring's actual use case
(personal, single-author, a handful of sibling projects) — not necessarily
implementation order:

1. **Lockfile** (highest priority). Nothing today records which exact
   commit a `branch`/`tag`/default-branch git dependency last resolved to
   — it can silently drift between builds or machines. A minimal
   `boring.lock` (just `name → resolved commit sha` per git dependency)
   would fix this without needing a real dependency-resolution algorithm.
   Would also want `boring update` (refresh the lock deliberately) and
   `--locked`/`--offline` flags (fail instead of silently re-resolving or
   silently falling back to a stale cache).
2. **Transitive dependencies.** `[deps]` is explicitly single-level — a
   dependency's own `[deps]` (if it has any) are never followed, same rule
   `[external_types]`/`[derives]`'s `include` already uses. Fine while no
   dependency itself depends on another Boring project; becomes a real gap
   the day one does. Doing this properly also opens the "diamond
   dependency" question (two deps needing the same third dep at possibly
   different refs) — no need to solve that until transitive resolution
   itself is needed.
3. **Version ranges / real resolution.** No `^1.2`/`~1.0`-style semver
   constraints or a solver — a dependency is always pinned exactly (a
   path is what it is; `rev` is exact; `branch`/`tag` float but aren't
   "resolved" against any other constraint). Only becomes meaningful once
   (2) exists and two dependencies could disagree on what they need.
4. **A public registry** (crates.io/npm-style: resolve "name + version"
   against a central index instead of always naming a path or git URL).
   Ecosystem-scale investment that doesn't fit a single-author, sibling-
   project use case — not worth pursuing unless that changes.
5. **Workspaces** (Cargo-style: several projects sharing one lockfile/
   build cache). Partially moot already — the git cache
   (`src/git_deps.rs`) is keyed by `(url, ref)` and shared globally on one
   machine regardless of which project asks for it, so two sibling
   projects depending on the same git repo at the same ref already don't
   duplicate the clone.
6. **dev-dependencies / feature-gated deps.** No distinction between "needed
   to build" and "needed only for tests/examples" — every `[deps]` entry is
   unconditional. Low priority; no concrete case has needed it yet.

Not real gaps despite looking like ones: **auth for private git repos**
works for free today (shelling out to the user's own `git` inherits their
SSH/credential-helper config — nothing to build); **integrity/checksums**
are largely redundant for a `rev`-pinned dependency (a commit sha is
already content-addressed by git itself).
