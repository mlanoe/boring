# `Image` / `Volume` — named-shape GPU buffer types

`Image` and `Volume` are built-in generic types for GPU kernel fields that
need a 2D (`Image`) or 3D (`Volume`) shape — an image, a tile, a voxel grid —
instead of a flat 1D buffer. They replace the pattern of carrying `width`/
`height` as separate `int` fields and manually computing `row * width + col`
everywhere: with `Image`/`Volume`, the shape lives on the field itself, index
math is a method call (`.at(...)`), and the kernel's dispatch grid is sized
automatically.

Two forms are available, for two different situations:

| | `Image<T, C, R>` / `Volume<T, X, Y, Z>` | `Image<T>` / `Volume<T>` |
|---|---|---|
| Shape known | at compile time | only at construction time (runtime) |
| Typical use | fixed-size tiles, block-shared memory | buffers whose size depends on caller-provided data |
| Example | a 16×16 GEMM tile | a runtime-sized image loaded from a file |

Both forms share the same method surface (`.at()`, `.width()`, `.height()`,
`.depth()`) and the same automatic grid dispatch — pick whichever matches
whether the shape is known when you write the kernel or only when you
construct it.

## Fixed shape: `Image<T, C, R>` / `Volume<T, X, Y, Z>`

`T` is the element type; `C`/`R` (columns/rows) or `X`/`Y`/`Z` are
compile-time integer constants:

```boring
kernel Tile:
    let  Image<float, 16, 16>'global   src
    mut  Image<float, 16, 16>'unified  dst

    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        dst.at(c, r) = src.at(c, r) * 2.0
```

A fixed-shape field needs no `init()` assignment for the pattern above — it's
zero-initialized automatically, the same as a fixed-size `[T, N]` field.
Passing one into `init()` works exactly like a regular typed parameter:

```boring
init(Image<float, 16, 16>'global input):
    src = input
```

This form is the right choice for block-shared tiles (`'actor`), where the
size has to be known to allocate the memory — see `LinearKernel`'s
`tile_x`/`tile_w` in `whisper-boring/src/math_gpu.br` for a real 16×16-tiled
GEMM using it.

## Dynamic shape: `Image<T>` / `Volume<T>`

One type argument — the element type only. The shape is determined at
construction time, from whatever the caller passes to `init()`:

```boring
kernel Fill:
    mut Image<float>'unified img

    init(int w, int h):
        img = Image(w, h)

    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        if c < img.width() and r < img.height():
            img.at(c, r) = img.at(c, r) + 1.0
```

Construct and dispatch it like any other kernel — no `Dimension` field, no
manual `rows`/`cols` bookkeeping:

```boring
mut k = Fill(64, 64)
kernel:
    k(block = (16, 16))
```

This form is the right choice whenever the shape comes from the data itself
— a loaded image, a caller-provided grid size — rather than from the kernel
author.

### Construction forms

Three forms of `Image(...)`/`Volume(...)`, used inside `init()` to assign a
dynamic-shape field — pick whichever matches what you already have:

| Form | When to use |
|---|---|
| `Image(data, w, h)` | you already have a buffer (loaded from a file, computed elsewhere) |
| `Image(w, h)` | you don't — allocate a new, zero-filled buffer of size `w * h` |
| `Image(w, h, fill = v)` | same allocation, filled with `v` instead of `0` |

```boring
img = Image(data, w, h)          # wrap an existing buffer
img = Image(w, h)                # allocate, zero-filled
img = Image(w, h, fill = 1.0)    # allocate, filled with 1.0
```

`Volume` follows the same three forms with one more axis:
`Volume(data, x, y, z)` / `Volume(x, y, z)` / `Volume(x, y, z, fill = v)`.

Passing your own `data` (first form) makes you responsible for its length
actually matching `w * h` (`* depth` for `Volume`) — nothing checks this
automatically today. The other two forms don't have this risk, since the
buffer is allocated to the right size here.

## Accessing elements

Both forms share the same four methods:

| Method | Returns |
|---|---|
| `.at(c, r)` (`Image`) / `.at(x, y, z)` (`Volume`) | the element at that position — usable for both reads and writes: `img.at(c, r) = v` |
| `.width()` | the size along the first axis |
| `.height()` | the size along the second axis |
| `.depth()` | the size along the third axis (`Volume` only) |

Storage is row-major: `.at(c, r)` addresses `c + r * width`, and
`.at(x, y, z)` addresses `x + y * width + z * (width * height)` — the
natural generalization to a third axis.

There's no `.len()` or `.set(...)` — `.at(...)` covers both reading and
writing, and `.width()`/`.height()`/`.depth()` cover shape queries.

## Qualifiers

`Image`/`Volume` fields accept the same GPU memory qualifiers a flat `[T]`
field does: `'unified`, `'global`, `'const`, bare `'actor` (block-shared),
`'actor'unified`, `'actor'global`. A kernel can freely mix fixed-shape and
dynamic-shape `Image`/`Volume` fields, and plain `[T]`/`[T, N]` fields,
side by side.

## Dispatch and grid sizing

Omitting `grid = (...)` at dispatch infers a 2D (`Image`) or 3D (`Volume`)
grid automatically, from the field's shape and the `block = (...)` size —
whether the shape is a compile-time constant or was only known at
construction time:

```boring
kernel:
    k(block = (16, 16))   # grid inferred from img's width/height — no grid= needed
```

Pass `grid = (...)` explicitly if you need a different dispatch size than
the field's own shape implies.

> **`boring run` caveat:** the auto-inference above is fully correct in the
> real code `boring build` generates for every GPU target. The interpreter's
> own simulation (`boring run`), used for fast local iteration without a GPU,
> doesn't yet read an `Image`/`Volume` field's shape when `grid =` is
> omitted — it falls back to a cruder heuristic that can under-cover the
> grid once the shape needs more than one block along an axis, silently
> leaving part of the buffer unwritten. Until that's fixed, pass `grid =`
> explicitly when running such a kernel through `boring run`:
> `grid = ((width + 15) / 16, (height + 15) / 16)` for a `block = (16, 16)`
> dispatch, for example.

## A migration example

`whisper-boring`'s `TransposeKernel` (`src/math_gpu.br`) is the pattern this
feature replaces — `rows`/`cols` as separate `int` fields, with every index
computed by hand:

```boring
# Before
kernel TransposeKernel:
    let [float]'global   src
    mut [float]'unified  dst
    let int              rows
    let int              cols

    init([float]'global s, int r, int c):
        src  = s
        rows = r
        cols = c
        dst  = [0.0 for ..r * c]

    def ():
        let cell = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        if cell < rows * cols:
            let i = cell / cols
            let j = cell % cols
            dst[j * rows + i] = src[cell]
```

```boring
# After
kernel TransposeKernel:
    let Image<float>'global   src
    mut Image<float>'unified  dst

    init([float]'global s, int rows, int cols):
        src = Image(s, cols, rows)
        dst = Image(rows, cols)

    def ():
        let cell = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        if cell < src.width() * src.height():
            let i = cell / src.width()
            let j = cell % src.width()
            dst.at(i, j) = src.at(j, i)
```

(`whisper-boring` itself hasn't been migrated yet — that's tracked
separately.)

## Notes

- Field names starting with `__` are reserved for the compiler's own use —
  avoid them for your own kernel fields.
- `Volume` follows `Image` exactly, one axis further: `Image<T,C,R>` ↔
  `Volume<T,X,Y,Z>`, `Image<T>` ↔ `Volume<T>`, `.at(c,r)` ↔ `.at(x,y,z)`.
- `'surface` (pixel buffers presented to the screen) is unrelated to
  `Image`/`Volume` and doesn't compose with them — see `gpu-display.md`.

## See also

- `examples/matrix_mul_gpu.br` — the fixed-shape form (`Image<float,32,32>`).
- `examples/mandelbrot_gpu.br` — the dynamic-shape form (`Image<float>`).
