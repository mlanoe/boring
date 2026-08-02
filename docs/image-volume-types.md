# `Image<T, C, R>` / `Volume<T, X, Y, Z>` — named-shape buffer types

> **Status: Implemented.** Parser/AST, checker/validator, all four GPU backends
> (device + host codegen), and docs are done — see the resolved "Open Questions"
> section below for the decisions made. The `TransposeKernel` migration in
> whisper-boring (mentioned under "Compatibility" below) remains a follow-up,
> not required to ship this.

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
Image<T, C, R>          # T = element type, C columns, R rows
Volume<T, X, Y, Z>
```

- Backed by a flat buffer of `T` (`Vec<T>`/`CudaSlice<T>`/`DeviceBuffer<T>`/
  `Buffer` depending on qualifier and backend — the same host representation
  dynamic `[T]` already uses) rather than the nested-multidimensional-array
  representation this design originally sketched — see "Open Questions" §1-2
  for why. `C`/`R`/`X`/`Y`/`Z` are compile-time integer constants (`Type::ConstInt`
  in the AST, the same mechanism `GameOfLife<64, 64>`-style const generics
  already use), carried as type-level shape metadata for `.at(...)` addressing
  and grid inference, not a competing runtime length.
- Element type `T` comes first: `Image<T, C, R>` — consistent with every other
  generic in the language (`Dict<K, V>`, `Future<T>`).
- **Precabled methods**, replacing the hand-rolled index math seen in
  `TransposeKernel` above: `.width()`/`.height()` (`Image`), `.depth()`
  (`Volume` also gets these two plus depth), `.at(c, r)`/`.at(x, y, z)` for
  named-axis indexing instead of manually reconstructing `j * rows + i`.
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
`Image<T,C,R>`/`Volume<T,X,Y,Z>` compose with **every** qualifier a flat `[T]`
field can carry today, including bare `'actor` (block-shared — only
meaningful for `Image`/`Volume` fields sized to fit block-shared memory, a
narrower case), `'unified`, `'global`, `'const`, `'actor'unified`, and
`'actor'global` — implemented in `parser/mod.rs`'s `parse_kernel_field`
(the single qualifier-legality choke point, not `gpu-module.md`'s validation
matrix — that doc is descriptive, not enforcing). Concretely:
`mut Image<int,256,256>'actor'global histogram` is as valid as today's
`mut [int]'actor'global counts`. The qualifier × field-type matrix in
`gpu-module.md` has been updated with `Image`/`Volume` as a third field-shape
row alongside `[T]` dynamic and `[T, N]` fixed — see that doc for the one
place `Image`/`Volume` legality actually *differs* from `[T, N]` (bare
`'unified`/`'global` are valid for `Image`/`Volume`, unlike `[T, N]`).

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

**Where the actual implementation deviated from this section's plan:**

- Qualifier legality lives entirely in `parser/mod.rs`'s `parse_kernel_field`
  (a single choke point), not split across checker/validator — `gpu-module.md`'s
  matrix is descriptive documentation of that function's behavior, not a
  separate enforcement layer.
- `Image`/`Volume` are backed by the same flat/dynamic buffer representation
  as `[T]` (not nested arrays) — see "Proposed Design" above; this is *why*
  bare `'unified`/`'global` are valid for `Image`/`Volume` but not `[T, N]`.
- wgpu's grid inference doesn't live in a per-backend `host.rs` — it's in the
  shared `src/transpiler/emit_kernel.rs` (`try_emit_kernel_dispatch`), since
  wgpu dispatch call sites desugar through that shared code instead of a
  per-backend `__boring_launch` method. See `wgpu-backend.md`'s "Grid inference"
  section.
- `book.md` was **not** touched — kernel-specific built-ins (this includes the
  precedent `Dimension`, which also isn't in `book.md`) are documented in
  `gpu-module.md` only.
- One unrelated bug found and fixed alongside this work: wgpu's pipeline
  creation (`create_compute_pipeline`) wasn't wrapped in an error scope,
  unlike shader-module creation and `dispatch()` — an oversized fixed-`'actor`
  field would panic via wgpu's default uncaptured-error handler instead of
  being reported. See `wgpu-backend.md`'s "Error handling" section.

## Compatibility

Same situation as `actor-qualifier-unification.md`: no real Boring programs
exist yet except **whisper-boring**. `math_gpu.br`'s `TransposeKernel`
(`src=rows×cols` flat, `dst=cols×rows` flat, manual `i`/`j` derivation) is
the concrete migration candidate and motivating example quoted above — not
a blocker, since the flat-array form keeps working unchanged; migrating it
to `Image<C, R>` is a natural follow-up once this lands, not a requirement
to ship it.

## Open Questions — resolved

1. **Generic parameter order**: settled as `Image<T, C, R>` / `Volume<T, X, Y, Z>`
   — `T` first, consistent with every other generic in the language
   (`Dict<K,V>`, `Future<T>`). Inferring `T` from `init(...)` (like plain array
   fields do) was considered and rejected as unnecessarily implicit for a
   type that already needs `C`/`R` spelled out explicitly.
2. **Row-major vs column-major storage**: row-major — `.at(c, r)` lowers to
   flat index `c + r*C`; `.at(x, y, z)` lowers to `x + y*X + z*(X*Y)` (the
   natural generalization of the 2D formula to a third axis). Matches every
   example in this document and the `TransposeKernel` migration target's own
   `j*rows+i` convention. Implemented once, shared across all four backends,
   in `transpiler::helpers::image_volume_at_index`.
3. **`'actor` (bare, block-shared) on `Image`/`Volume`**: no new compile-time
   size check added — matches the existing gap for plain bare-`'actor` fields
   (see cuda-module.md's "Grid inference rules"/error-handling section: block
   size, and by the same reasoning shared-memory size, is validated at
   runtime by the real launch call, not statically). Revisit both together
   if that gap is ever closed.
4. **Method surface**: `.width()`/`.height()`/`.depth()`/`.at(...)` only, for
   v1 — no `.len()`, no `.set(...)`. Smallest surface that unblocks the
   `TransposeKernel` migration; can grow later without a breaking change.
