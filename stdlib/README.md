# `boring/stdlib/` — the first-party `boring.*` standard library

Every `.br` file in this directory is a real, loadable module, embedded
directly into the compiler binary (`src/stdlib_embed.rs`, via `include_str!`
— the same pattern already used for the built-in GPU profiles in
`src/interpreter/gpu_profile.rs`). A Boring program imports one with:

```boring
use boring.<module>          # all pub items
use boring.<module>.*        # glob (same effect today — no submodules)
use boring.<module>(A, B)    # selective import
```

This works identically under `boring run` and `boring build` (and the GPU
targets) — no file needs to exist on disk next to the `boring` binary,
because the source text is compiled into the binary itself.

## What belongs here

Only **new, genuinely-gated functionality** — composite types or functions
built on top of the language's native primitives that don't already exist
for free. `[T]`/`{T}`/`{K=V}` literals and every native method already
callable on them, plus the always-loaded global functions (`print`, `assert`,
`pow`, …) and the built-in `Error` enum, stay unconditionally global exactly
as today — see docs/book.md's "Built-in Types & Functions Reference"
appendix for those. Do **not** add a stdlib module that just re-documents
something already free; it would make `use boring.x` a confusing no-op.

## Adding a new module

1. Add `stdlib/<name>.br` with `pub` on every item you want importable.
2. In `src/stdlib_embed.rs`, add one `include_str!` constant and one arm in
   `lookup()`. That's the entire wiring — no other file needs to change.
3. Add a test case exercising `use boring.<name>` (see `tests/cases/
   boring_stdlib_collections.br` for the pattern), registered in both
   `tests/run.rs` (`interp_test!`) and `tests/transpile.rs`
   (`transpile_test!`).
4. Document the module in docs/book.md §15 (Modules).

See `docs/cross-project-code-sharing-gap.md` for what this mechanism
deliberately does *not* solve (importing `.br` source across sibling
projects — still an open, separate problem).
