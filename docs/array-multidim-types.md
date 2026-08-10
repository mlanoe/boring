# Labeled multi-dimensional arrays

**Status:** implemented — parser, checker, interpreter, and all 4 GPU
transpiler backends (cuda/rocm/metal/wgpu) support the syntax below. This
feature replaced the older `Image<T,C,R>`/`Volume<T,X,Y,Z>` types, deleted
once `whisper-boring` (the one real consumer of either) migrated off them.

Labeled multi-dimensional arrays fix the readability problems `Image<T,C,R>`
/ `Volume<T,X,Y,Z>` had: unclear qualifier placement, and index arguments
(`.at(c, r)`, `.at(x, y, z)`) whose meaning depends on remembering a fixed
argument position rather than being spelled out at the call site.

The guiding principle behind every choice below: **readability wins over
flexibility.** Where a choice would make some code shorter but ambiguous,
the more verbose, unambiguous option was taken instead.

Boring's 1D arrays are unaffected by any of this:

```boring
let [type] a = [ f(x) for x in ..N ]
let [type, N] a
let v = a[x]
```

## Type syntax

### Declaring a labeled array type

The existing fixed-size array grammar is `[T, N]` — type, comma, a
compile-time size. Labeled arrays generalize the *slot after the comma* from
a single unlabeled size into a comma-separated list of **labels**, each
either bare (dynamic axis) or `label = value` (compile-time-fixed axis):

```boring
# Dynamic shape — size known only at construction (replaces Image<T> / Volume<T>)
let [type, width, height] a
let [type, width, height, depth] a

# Fixed shape — compile-time constants (replaces Image<T,C,R> / Volume<T,X,Y,Z>)
let [type, width = W, height = H] a
let [type, width = W, height = H, depth = D] a
```

This form (over the alternative `[type width, height]`, label attached
directly to the type with no comma) is a direct, minimal generalization of
the grammar rule `[T, N]` already used
(`"[" type "," const_expr "]"`, `spec/grammar.bnf`) — one production
extended, not a second parallel form with an inconsistent comma rule between
the first and later labels.

> **Grammar note — disambiguating from `[T, N]`.** A single bare identifier
> after the comma already means something: a reference to an existing
> const-generic parameter (`spec/grammar.bnf`'s `const_expr`, e.g. `[T, N]`
> where `N` was declared elsewhere as `<uint N>`). That meaning is
> unchanged. Two or more comma-separated bare identifiers is the labeled
> form: a declaration of that many labeled, dynamically-sized axes. The two
> forms are disambiguated purely by arity (1 vs 2+), which is backward
> compatible since the 1-identifier case never changes meaning.

### Axis order is fixed at declaration

Order is fixed in the *type declaration* itself:

- The first label declared is axis 1, the second is axis 2, the third is
  axis 3 — permanently, for that type. `[T, width, height]` and
  `[T, height, width]` are different (transposed) types.
- This order determines:
  - **Storage layout** — the natural generalization of `Image`'s row-major
    layout (`.at(c, r)` addresses `c + r * width`): axis 1 is the
    fastest-varying index.
  - **GPU dispatch axis mapping** — axis 1 ↔ `gpu.thread.x`, axis 2 ↔
    `gpu.thread.y`, axis 3 ↔ `gpu.thread.z`, independent of the label names
    chosen. This is what lets automatic grid-size inference at kernel
    dispatch keep working the same way it did for `Image`/`Volume`.

Without pinning this down explicitly, the "which index means which axis"
confusion would just relocate from numeric argument positions to label
declaration order, solving nothing.

## Qualifiers

Qualifier placement follows the existing convention for flat arrays —
**after the closing bracket**, never inside it. This is already how
`[float]'unified`, `[float, W * H]'const` etc. are written
(`examples/saxpy.br`, `linguist/samples/gpu.br`):

```boring
let  [float, width, height]'global   src
mut  [float, width, height]'unified  dst
```

Never `[float'global, width, height]` or `[float, width'global, height]`.

## Construction

### Comprehension

```boring
let a = [ f(width, height) for width in ..W for height in ..H ]
let a = [ f(width, height, depth) for width in ..W for height in ..H for depth in ..D ]
```

Type inference works exactly as it does for 1D comprehensions
(`let squares = [i * i for i in ..5]` already infers `[int]`) — the labeled
multi-dim type, including its label names, is inferred from the chained
`for` clauses.

**Why not `for width, height in (..W, ..H)`?** That form reuses the existing
destructuring grammar `for IDENT ("," IDENT)* "in" expr`, whose natural
meaning — destructuring pairs out of one iterable — is a **zip** (pairwise,
truncated to `min(W, H)`), not a cartesian product. It would look
interchangeable with the chained form for `W == H` and silently produce a
`min(W, H)`-sized result instead of `W * H` the moment they differ. One
canonical form (the chained `for`) avoids that trap.

### Fill shorthand (no bound variable)

For filling every element with the same constant, two terser forms sit
alongside the chained-`for` comprehension:

```boring
let a = [ 0.0 for n ]                          # 1D — no label needed
let a = [ 0.0 for width = w, height = h ]      # 2+ axes — labels required
```

Both are pure surface sugar over the same AST nodes the comprehension form
above already uses:

- `[value for n]` (no `..` required, unlike the range form — there's no
  loop variable here to justify demanding explicit range syntax) is
  identical to `[value for ..n]`.
- `[value for width = w, height = h]` uses each label directly as that
  axis's loop variable name in the exact same construct the chained-`for`
  comprehension produces — just with variable names nobody intends to
  reference.

**The labels are deliberately not bound as usable variables in `value`.**
`width`/`height` here are purely descriptive of shape, not loop variables —
letting `value` reference them would make this a second way to write the
same general per-position comprehension the chained `for...for...` form
already covers (rejected above for the same reason `for a, b in (..W, ..H)`
was). Keeping this form fill-only means it and the chained form serve
genuinely different, non-overlapping purposes: constant fill vs.
position-dependent computation. Referencing `width` inside `value` here
simply fails as an undefined variable — an explicit error, not a silent
miscompile.

**Why not `[0.0, n]` / `[0.0, width = w, height = h]` (comma instead of
`for`)?** The 1D form collides with the already-existing 2-element array
literal — `[a, b]` already means "an array containing exactly these two
elements", so `[0.0, n]` is indistinguishable from a literal 2-element array
at the token level. `for` sidesteps this entirely: it's a reserved keyword,
never a value, so `[0.0 for n]` can never be confused with a literal, and it
reuses the exact dispatch point the parser already has for `for` right
after the first bracket element.

## Indexing

```boring
let v = a[width = w, height = h]
let v = a[width = w, height = h, depth = d]
```

- **Labels are mandatory** for 2+ dimensions. There is no positional
  fallback (`a[w, h]`) — the point of this syntax is that an index's axis
  is always readable at the call site, and a positional shorthand would
  immediately start eroding that.
- **Order is free at the use site.** `a[height = h, width = w]` is exactly
  as valid as `a[width = w, height = h]` — since the names disambiguate,
  forcing a specific order in addition to requiring the names would combine
  the verbosity of labels with the rigidity of positions, without gaining
  the benefit of either. This mirrors how named function arguments and
  `fill = v`-style keyword args already behave elsewhere in the language.

## Shape queries: `a.axis`

Each declared axis label is exposed directly as a read-only property —
`a.width`, `a.height` — the same no-parens convention every other
argument-free, computed value already uses in Boring (a struct/enum `req`
getter, `arr.length` on a plain collection):

```boring
a.width
a.height
```

This needs no new language feature. It's the same `Field`-access path
`arr.length` already goes through — the compiler recognizes, for a
`[T, ...]`-typed receiver, that a field name matching one of that type's own
declared axis labels resolves to that axis's size rather than an ordinary
struct field (there is no ordinary field to shadow: arrays have no
user-visible fields other than these axis properties and the built-in
`.length`/`.count`/`.len`).

`a.axis` is read-only — `a.width = ...` is not assignable, matching
`arr.length`'s own read-only status; a labeled array's shape is fixed at
construction (see "Construction" above) and changes only through
`.reshape()`, never through a property write.

`a.axis` is kept separate from indexing — indexing always uses bare labels
as *keyword arguments* (`a[width = w, ...]`), never as a receiver-less
property (`a[.width = w, ...]`) — so a given label has exactly one lexical
form per role (bare identifier for values/positions, dotted-property for the
shape query), rather than two competing spellings for "the same" label.

> **History.** An earlier revision of this design exposed shape queries as
> `a.size(.axis)`, a method call taking a compiler-synthesized "axis
> selector" enum, modeled on Boring's leading-dot enum-variant shorthand.
> That mechanism was never actually needed: every real call site (this repo
> and `whisper-boring`, its one consumer) passed a literal `.axis` argument,
> never a value carried through a variable or function boundary — and the
> implementation could not have supported that anyway, since `.size(.axis)`
> was resolved by pattern-matching the literal AST node at the call site, not
> through a real runtime enum value. With no parametric use case to justify
> the extra method-call indirection, `a.axis` — direct, read-only, no
> parentheses — replaced it outright rather than keeping both spellings for
> the same query.

## Converting to and from a 1D buffer

Conversion between a 1D array and a labeled multi-dim array is **not**
allowed implicitly (see the next section for why). But wrapping an existing
flat buffer under a shape is a real use case — mainly loading external data
(files, network) without a copy — and has an explicit equivalent, plus its
symmetric inverse:

```boring
let flat = [f(i) for i in ..(W * H)]
let a    = flat.reshape(width = W, height = H)   # [T] → [T, width, height]
let back = a.flatten()                           # [T, width, height] → [T]
```

`.reshape(...)`:

- checks `flat.len() == W * H` (compile-time error if both sizes are known
  constants and mismatched, otherwise a runtime check);
- **moves** the flat buffer rather than copying it, matching `Image`'s
  zero-copy semantics — important for large GPU-resident buffers;
- is an explicit, named operation, not an implicit conversion — consistent
  with the rest of the language never doing implicit conversions (e.g. no
  `string(x)`, only interpolation).

`.flatten()` is the exact symmetric inverse:

- takes no arguments — the target length (`W * H`, `W * H * D`, …) and the
  element order are already fully determined by the array's own shape and
  its fixed axis order ("Axis order is fixed at declaration" above), so
  there is nothing left to specify;
- **moves** the underlying buffer out (zero-copy), same direction as
  `.reshape()`;
- preserves whatever ownership qualifier the array had (`'unified`,
  `'global`, …) on the returned `[T]`.

Because storage order is canonical (axis 1 fastest-varying, matching
`Image`'s `c + r * width` layout), the round trip is exact:
`flat.reshape(width = W, height = H).flatten()` always yields back the same
buffer, element for element, as `flat` — reshape/flatten never reorders
data, only attaches or drops the shape.

## No 1D ↔ multi-D conversion

Reinterpreting a flat array as a multi-dim one (or vice versa) implicitly is
disallowed. `Image(data, w, h)` (the old, now-deleted constructor) placed the
entire responsibility for `data.len() == w * h` on the caller with no check
— `.reshape()` above closes that hole by making the check explicit instead
of a bare type-punning constructor.

## Cross-label compatibility between same-shape types

Two array types with the same dimensionality and the same per-axis
compile-time sizes (if any), but **different label names**, are structurally
the same type — labels are local aliases for readability, not part of the
type's identity. A function parameter can use its own label vocabulary
independent of what the caller's variable is labeled:

```boring
def sum2d([float w, h] grid): ...   # fine to call with an array labeled (width, height)
```

**This is not, however, a blanket "any relabeling is safe" rule.** Some
label vocabularies encode *conflicting axis-order conventions*, and the type
system has no way to tell a harmless rename from a dangerous one — both look
identical (same arity, same per-axis sizes) to the compiler.

Concrete counter-example: `width, height` conventionally means
horizontal-then-vertical (axis 1 = columns, matching `Image.at(c, r)`).
`line, column` conventionally means vertical-then-horizontal (axis 1 =
rows) — the **opposite** order. Passing a `[float, width = 800, height = 600]`
value into a `[float, line, column]` parameter under a purely positional,
implicit relabeling rule would silently bind `line = 800` — reading the
width as a row count. For any non-square shape this is a silent transpose
bug, indistinguishable at compile time from a safe rename like
`width, height` → `w, h`.

Cross-label passing across a function/assignment boundary therefore
requires an **explicit per-axis mapping** whenever label names differ,
rather than an implicit pass-through based on arity alone:

```boring
def f([float line, column] grid): ...

let img = [float, width = 800, height = 600](...)

f(img)                                     # error: label sets (width,height) ≠ (line,column)
f(img as [line = width, column = height])  # explicit — the mismatch, if any, is visible here
```

This costs nothing in the safe case (`width, height` → `w, h` is a trivial,
obviously-correct line to write) and forces the dangerous case into the
open, where a reviewer has a chance to notice the axes don't actually
correspond — rather than letting the compiler wave it through because the
shapes happen to match numerically.

## Notes

- **Vocabulary consistency** (`width/height` vs `line/column`, etc.) is a
  convention, not something the type system enforces project-wide. The
  cross-label rule above is the actual enforcement mechanism in practice —
  any place two vocabularies meet is forced into an explicit `as [...]`,
  visible to a reviewer. The recommended default vocabulary for Boring code
  is `width, height, depth` for axis 1/2/3, matching the axis order
  `Image`'s `.at(c, r)` already used.
- **Parity with `Image`/`Volume`'s `'actor` (block-shared) fixed-shape
  tiles**: `[T, width = W, height = H]` is the generalization of `[T, N]`,
  and `[T, N]` field declarations are already zero-initialized automatically
  with no `init()` needed. The same rule extends unchanged to the
  multi-axis form: total element count `W * H` (or `W * H * D`),
  zero-initialized, no special-casing required. Example (16×16 GEMM tiles
  in shared memory, `whisper-boring/src/math_gpu.br`):

  ```boring
  # Image
  mut Image<float, 16, 16>'actor tile_x  # tile of x
  mut Image<float, 16, 16>'actor tile_w  # tile of w

  # Labeled array
  mut [float, width = 16, height = 16]'actor tile_x  # tile of x
  mut [float, width = 16, height = 16]'actor tile_w  # tile of w
  ```

  Qualifier placement is unchanged from the general rule above (after the
  closing bracket), so stacked forms like `'actor'unified` carry over
  exactly as they work on flat `[T]` fields.
- **Beyond 3 axes**: CPU-side arrays have no limit — the label list, the
  fixed declaration order, the `a.axis` shape-query properties, and
  `.reshape()` / `.flatten()` all generalize to any number of axes without
  change. A 4D `[T, batch, channel, width, height]` is exactly as
  well-formed as a 2D one. GPU-side kernel fields are capped at 3, same as
  `Volume<T,X,Y,Z>` always was — CUDA/Metal/wgpu thread and block indices
  only go up to `x`/`y`/`z`, and automatic grid-size inference at dispatch
  has nowhere to put a 4th axis. For GPU-resident data with more than 3
  logical axes, the pattern is to manually collapse the extra axes into one
  of the three GPU axes (e.g. fold `channel` into `height` at construction
  time) and recover the individual indices inside the kernel body.

## See also

- `Image<T,C,R>`/`Volume<T,X,Y,Z>` — the types this feature replaced.
  Deleted once `whisper-boring`, the one real consumer, migrated off them.
