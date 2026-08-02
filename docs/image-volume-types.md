# `Image<C, R>` / `Volume<X, Y, Z>` — named-shape buffer types

> **Status: Proposed.** No implementation yet — this document is the task
> spec for that work. Companion to [`actor-qualifier-unification.md`](actor-qualifier-unification.html)
> (in progress in a parallel session) — see "Interaction with the `'actor`
> qualifier rename" below; the two should land compatible with each other,
> not necessarily in the same change.

## Problem Statement

Two related gaps, discussed leading up to this document:

1. **No 2D/3D grid inference for `'unified`/`'global` fields.** Grid
   inference (`cuda-module.md`, "Dispatch parameters") only handles the 1D
   case: `grid = ceil(len(buf) / block)` from a flat array field's length.
   A 2D/3D kernel (image processing, tiled matmul, convolution) gets no
   inference at all — omitting `grid` silently defaults to `(1, 1, 1)`,
   dispatching a single block over what the author likely meant as a whole
   image or volume.

2. **Real code already hand-rolls 2D indexing on flat buffers, error-prone.**
   Checked directly rather than assumed: `whisper-boring/src/math_gpu.br`'s
   `TransposeKernel` (lines 61-82) takes `rows`/`cols` as separate `int`
   fields alongside a flat `[float]'global src`/`[float]'unified dst`, and
   manually derives 2D position from a linear thread index:
   ```boring
   let i = cell / cols
   let j = cell % cols
   dst[j * rows + i] = src[cell]
   ```
   This is exactly the shape every 2D kernel in this codebase needs today —
   width/height carried as separate plain `int` fields, with every
   loop/kernel re-deriving row/column from a linear index by hand. Nothing
   type-level ties the flat buffer's declared length to `rows * cols`, or
   validates that `dst`'s length actually matches `rows * cols` at
   construction time.

An earlier direction considered fixing (1) alone by allowing **nested fixed
arrays** (`[[T, N1], N2]`, which already parse and run correctly outside
kernel context — verified directly) on `'unified`/`'global` fields, and
teaching grid inference to read the nested shape. Rejected: nesting order
is purely positional (does `[[T, N1], N2]` mean `N1` columns × `N2` rows, or
the reverse?) — there is no way to make that unambiguous from array syntax
alone without imposing an arbitrary row/column convention on every kernel
author. Named dimensions avoid the question entirely.

## Proposed Design

Two new built-in generic types:

```boring
Image<int C, int R>          # C columns, R rows
Volume<int X, int Y, int Z>
```

- Backed internally by nested multidimensional arrays (`[[T, C], R]` for
  `Image`, three levels for `Volume`) — no new runtime representation
  invented, this reuses the nesting that already parses and runs correctly
  today, just wrapped in a named type instead of exposed as raw positional
  nesting.
- Element type `T` and the dimensions are both generic:
  `Image<T, int C, int R>` (parameter order to be settled — see "Open
  Questions").
- **Precabled methods**, replacing the hand-rolled index math seen in
  `TransposeKernel` above: `.width()`/`.height()` (`Image`), `.depth()`
  (`Volume` also gets these two plus depth), `.at(c, r)`/`.at(x, y, z)` for
  named-axis indexing instead of manually reconstructing `j * rows + i`.
  Exact method surface is part of the implementation work, not fixed here.
- **Grid inference reads `C`/`R` (or `X`/`Y`/`Z`) directly off the field's
  declared type**: `grid = (ceil(C/bx), ceil(R/by), 1)` for `Image`,
  `(ceil(X/bx), ceil(Y/by), ceil(Z/bz))` for `Volume` — no separate
  `Dimension` field to keep in sync, no manual `rows`/`cols` `int` fields,
  no risk of the buffer's actual length silently disagreeing with the
  dimensions used to compute grid size.

### Relationship to `'surface`

`'surface` stays **entirely separate, unchanged** — it encodes a
presentation-path constraint (how a pixel buffer reaches the screen, which
differs by backend: direct on CUDA, blit-only on Metal/wgpu — see
`gpu-display.md`), not just a 2D shape. `Image`/`Volume` are for compute
buffers; they do not replace or subsume `'surface` in any way. A kernel
that both computes into an `Image` and presents pixels would still use
`'surface` + `screen.present()` for the presentation half, separately.

### Interaction with the `'actor` qualifier rename

`actor-qualifier-unification.md`'s rename has already landed: `'sync` is now
bare `'actor`, and `'actor'unified` exists alongside `'actor'global`.
`Image<C, R>`/`Volume<X, Y, Z>` (still proposed, not implemented) need to
compose with **every** qualifier a flat `[T]` field can carry today,
including bare `'actor` (block-shared — only meaningful for `Image`/`Volume`
fields sized to fit block-shared memory, likely a narrower case),
`'unified`, `'global`, `'const`, `'actor'unified`, and `'actor'global`.
Concretely: `mut Image<256, 256>'actor'global histogram` should be as valid
as today's `mut [int]'actor'global counts`. Since this document doesn't touch
qualifier *semantics* (only field *shape*), the qualifier × field-type
validation matrix (`gpu-module.md`) needs a final pass once `Image`/`Volume`
land, to add them as a third field-shape category alongside today's `[T]`
dynamic and `[T, N]` fixed rows.

## Scope of impact

- **Grammar/parser**: `Image`/`Volume` as new recognized generic type
  names, parsed like any other built-in generic (`Type::Generic`), not a
  new dedicated grammar production — closer to how `Dimension` is already
  a plain named struct type than to a new syntax form.
- **AST/checker**: field-type validation matrix gains `Image`/`Volume` rows
  (per qualifier: which are legal, same shape as today's `[T]`/`[T, N]`
  rows in `gpu-module.md`).
- **Device-side codegen** (per backend `device.rs`): index-generation for
  `.at(c, r)`-style access needs to lower to the same flat-buffer address
  arithmetic (`c + r * C`, row-major, or the equivalent for `Volume`) every
  backend's device code already does by hand today — this is mostly a
  codegen convenience, not new addressing capability.
- **Host-side codegen** (per backend `host.rs`): grid-inference logic
  (`auto_grid_field`-equivalent, currently 1D-array-only — see
  `cuda/host.rs:emit_boring_launch`) gains an `Image`/`Volume`-typed-field
  branch computing a 2D/3D grid instead of the existing 1D
  `ceil(n/block)`.
- **Docs**: `gpu-module.md` (new type section, qualifier matrix update),
  `cuda-module.md`/`rocm-backend.md`/`metal-backend.md`/`wgpu-backend.md`
  (grid-inference rules section each maintains), `book.md`. `.md` + `.html`
  mirrors throughout, per this repo's convention.
- **Tests**: new codegen tests per backend for `Image`/`Volume`-typed
  fields (grid inference shape, method codegen), plus the qualifier-matrix
  validation cases (legal/illegal combinations).

## Compatibility

Same situation as `actor-qualifier-unification.md`: no real Boring programs
exist yet except **whisper-boring**. `math_gpu.br`'s `TransposeKernel`
(`src=rows×cols` flat, `dst=cols×rows` flat, manual `i`/`j` derivation) is
the concrete migration candidate and motivating example quoted above — not
a blocker, since the flat-array form keeps working unchanged; migrating it
to `Image<C, R>` is a natural follow-up once this lands, not a requirement
to ship it.

## Open Questions

1. **Generic parameter order**: `Image<T, C, R>` vs `Image<C, R, T>` vs
   inferring `T` from the `init(...)` argument the way kernel struct fields
   already do for plain arrays (`gpu-module.md`'s qualifier-inference
   table) — needs a decision before implementation, not fixed here.
2. **Row-major vs column-major storage**: `Image`/`Volume` remove the
   *naming* ambiguity nested arrays had, but the actual flat-buffer layout
   (row-major assumed above) still needs to be picked and documented
   explicitly — every backend's device-code address arithmetic must agree.
3. **`'actor` (bare, block-shared) on `Image`/`Volume`**: block-shared
   memory is small (`maxSharedMem()`, a few KB to tens of KB per block) —
   does an `Image`/`Volume`-shaped `'actor` field need a compile-time check
   that `C*R*sizeof(T)` (or `X*Y*Z*sizeof(T)`) actually fits, the same kind
   of validation already missing for plain bare-`'actor` fields (see the
   "Block size validation at compile time" known limitation in
   `cuda-module.md`)? Worth deciding alongside that existing gap rather
   than inventing a separate check.
4. **Method surface**: exact list of precabled methods beyond
   `.width()`/`.height()`/`.depth()`/`.at(...)` — deferred to
   implementation, not blocking the type/grid-inference design above.
