# `fixedN_M` / `ufixedN_M`: binary fixed-point scalar types

> **Status: Phase 1 (core) implemented** in the Rust interpreter, the
> transpiler (generated newtype, `W ∈ {8, 16, 32, 64}`) and the self-hosted
> interpreter, with a shared test corpus (`tests/cases/fixed_point_*.br`).
> Not implemented yet: [Members](#members), `wrapping*`/`saturating*`/
> `checked*`/`mulRound`/`divRound` (Phase 2), GPU backends (Phase 3),
> 128-bit formats (Phase 4), math functions (Phase 5). The former open
> questions are resolved in [Decisions](#decisions).
>
> **Implementation notes** (where the code differs from, or narrows, the text):
>
> - Names are reserved by shape: `fixedN_M`/`ufixedN_M` always (an invalid one
>   is a parse error), `iN_M`/`uN_M` only when valid, so `i3_4` as a variable
>   name still works. A *user type* named like a valid format is not rejected yet.
> - `0.1 as fixed16_16` — a cast of an untyped literal — rounds to nearest like
>   a contextual literal; only casts of *computed* values truncate.
> - Strictness against int/float *variables* is enforced at `let` bindings and
>   in operators; passing a `float64` variable to a `fixed` parameter/field is
>   still accepted by the interpreters (same leniency `float32` has today).
> - `fixed → fixed` / `int → fixed` / `float → fixed` casts that do not fit give
>   `nil` in the interpreters and panic in generated Rust.
> - The Rust newtype definitions are appended to the generated program for every
>   format that appears in it; module-split output (`--emit-rust` with modules)
>   is untested.

## Problem Statement

Boring has integers (`int8`..`int128`, `uint8`..`uint128`) and IEEE floats
(`float16`/`float32`/`float64`), and nothing in between. The newtype
`type Q as int` gives a distinct type but no fixed-point semantics: no
shifted `*` and `/`, no `1.5` literals, no formatting.

Fixed-point reals are wanted for:

- **The kernel target** (Rust-for-Linux, `no_std`): floating point is not
  usable in kernel code, so fixed-point is the only way to have fractional values.
- **Determinism**: identical bits on CPU, Metal, wgpu and CUDA. Floats are not
  (FMA contraction, `float16`, no `f64` on Metal). Matters for the Whisper /
  LLM benchmarks and for simulations (game logic, `scratch-boring`).
- **GPU kernels**: MSL / WGSL / CUDA all have native integers, so a fixed-point
  value lowers to plain integer operations.

Out of scope here: **decimal** fixed-point (scale 10⁻ᵈ, for money). That is a
separate type family with its own naming; this document is binary only
(scale 2⁻ᴹ).

## The types

A fixed-point type has two sizes: the number of **integer bits** `N` and the
number of **fractional bits** `M`. `N + M` is the storage width and must be
one of `8`, `16`, `32`, `64`, `128`.

| Form | Meaning |
|---|---|
| `fixedN_M` | signed; `N` integer bits **including the sign bit**, `M` fractional bits |
| `ufixedN_M` | unsigned; `N` integer bits, `M` fractional bits |

The value of a variable whose raw storage is the integer `r` is `r / 2^M`.
Range of `fixedN_M`: `-2^(N-1) .. 2^(N-1) - 2^-M`; resolution `2^-M`.

### Aliases

Following the existing `iX` / `uX` / `fX` aliases of `intX` / `uintX` /
`floatX`:

| Canonical | Alias |
|---|---|
| `fixedN_M` | `iN_M` |
| `ufixedN_M` | `uN_M` |

The family is told apart by shape alone: one number → integer (`i32`) or
float (`f32`); **two numbers** joined by `_` → fixed-point (`i16_16`).

Common formats (any `N + M` in the valid widths is accepted):

| Width | Signed | Unsigned |
|---|---|---|
| 16 | `fixed8_8` / `i8_8` | `ufixed8_8` / `u8_8` |
| 32 | `fixed16_16` / `i16_16` | `ufixed16_16` / `u16_16` |
| 32 | `fixed8_24` / `i8_24` | `ufixed8_24` / `u8_24` |
| 64 | `fixed32_32` / `i32_32` | `ufixed32_32` / `u32_32` |
| 64 | `fixed24_40` / `i24_40` | `ufixed24_40` / `u24_40` |
| 128 | `fixed64_64` / `i64_64` | `ufixed64_64` / `u64_64` |

Constraints:

- `N + M` not in `{8, 16, 32, 64, 128}` → checker error.
- `fixedN_M` needs `N >= 1` (sign bit). `ufixedN_M` allows `N = 0`
  (`ufixed0_16`: values in `[0, 1)`). `M = 0` is legal but degenerate
  (identical to the integer type); the checker accepts it with no warning.
- The names `fixedN_M`, `ufixedN_M`, `iN_M`, `uN_M` (with valid `N`, `M`) are
  reserved type names, exactly as `i32` is today.
- There is **no** `fixed` / `real` bare alias to a default format. A bare name
  would hide the precision, which is the whole point of the type.

Why not `real…`: `real32` (float) and `real16_16` (fixed) would both be
32 bits yet behave very differently (NaN/inf and relative precision vs. exact
absolute resolution and overflow). `float` already carries that distinction.

## Literals

Float literals are already contextual (`let float32 a = 1.0; a + 3.14`
resolves `3.14` to `float32`). The same rule applies:

```boring
let fixed16_16 x = 1.5
let y = x + 0.25            # 0.25 resolves to fixed16_16
let ufixed8_8 z = 200.5     # ok
let fixed8_8 w = 200.5      # ERROR — out of range [-128, 127.99609375]
```

- A float literal not exactly representable is **rounded to nearest** (ties
  away from zero) at compile time. No warning is emitted for an inexact
  literal (`0.1` in `fixed16_16`): it would fire on nearly every literal.
- Out-of-range literals are a compile error, same as `300 as uint8`.
- An integer literal (`3`) also resolves to the fixed type from context
  (`let fixed16_16 n = 3`).
- Digit separators work as everywhere (`0.000_015`).

## Mixing and casts

Same strict rule as fixed-width integers and floats: two *different*
fixed-point formats never mix implicitly, nor do fixed-point and int/float.

```boring
let fixed16_16 a = 1.5
let fixed8_24  b = 0.5
let c = a + b                     # ERROR — cannot mix fixed16_16 and fixed8_24
let c = a + (b as fixed16_16)     # OK
let d = a * 2                     # OK — `2` is an untyped literal
let n = 3
let e = a * n                     # ERROR — int and fixed16_16 do not mix; cast
let e = a * (n as fixed16_16)     # OK
```

`as` conversions (all **runtime** conversions truncate toward zero; only
*literals* round to nearest, since they are a decimal value the author wrote
rather than a computation):

| From → to | Behaviour |
|---|---|
| fixed → fixed (other format) | shift by the difference of `M`; narrowing the integer part is range-checked, `nil` / error if it doesn't fit (same as integer narrowing); dropping fractional bits **truncates toward zero** |
| int → fixed | exact, range-checked |
| fixed → int | truncates toward zero |
| float → fixed | truncates toward zero (like Rust's `as`), range-checked; `NaN`/`inf` → out of range |
| fixed → float | nearest representable float (exact whenever the float mantissa is wide enough) |
| fixed → `string` | via interpolation only, as everywhere (no `string(x)`) |

## Arithmetic semantics

For `fixedN_M` stored in `intW` (`W = N + M`), with raw values `ra`, `rb`:

| Op | Definition | Result type |
|---|---|---|
| `a + b`, `a - b` | `ra ± rb` | same format |
| `-a` | `-ra` (signed only) | same format |
| `a * b` | `(ra * rb) >> M`, computed in a 2W-bit intermediate | same format |
| `a / b` | `(ra << M) / rb`, computed in a 2W-bit intermediate | same format |
| `a % b` | `ra % rb` | same format |
| `a < b`, `==`, … | `ra < rb` | `bool` |
| `<<`, `>>`, `&`, `\|`, `^` | not defined on fixed-point (use `.raw`) | |

- **Result is never widened**: `fixed16_16 * fixed16_16` is `fixed16_16`.
  Callers who want headroom cast first.
- **Rounding** of `*` and `/`: truncate toward zero, consistent with integer
  `/` and with `fixed → int`, and symmetric (`(-a) * b == -(a * b)`). An
  arithmetic shift (floor) is cheaper but asymmetric; round-to-nearest is
  more accurate (see `mulRound`/`divRound` in Phasing).
- **Overflow** follows the integer policy: panic in debug, wrap in release.
  Explicit variants, mirroring whatever the integer types expose:
  `a.wrappingMul(b)`, `a.saturatingMul(b)`, `a.checkedMul(b)` (→ `T?`).
- Division by zero: panic, as for integers.
- The 2W-bit intermediate for `W = 128` (`fixed64_64`, …) does not exist
  natively. See [Phasing](#phasing): 128-bit formats are not in the first cut.

### Members

```boring
let x = 1.75 as fixed16_16
x.raw                       # int32 — the underlying integer (0x0001_C000)
fixed16_16.fromRaw(0x1_C000)
x.floor()   x.ceil()   x.round()   x.trunc()   x.fract()    # same format
x.abs()     x.signum()  x.min(y)    x.max(y)   x.clamp(lo, hi)
fixed16_16.epsilon          # 2^-16, the resolution
fixed16_16.min   fixed16_16.max
```

`sqrt`, trigonometry, `ln`/`exp` are **not** in the first cut (see Phasing).

### Formatting

- `"{x}"` prints the exact value as a decimal with trailing zeros trimmed.
  This is always finite (denominator is a power of two): `fixed16_16` of
  `1.5` prints `1.5`, of raw `1` prints `0.0000152587890625`.
- `"{x:.3f}"` rounds to nearest to 3 places, same spec as floats.

## Rust mapping (transpiler)

Generated newtype, **no external dependency** (works in `no_std` and in the
Linux kernel target). Emitted once per format actually used, into the
generated project's prelude module:

```rust
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Fixed16_16(pub i32);

impl Fixed16_16 {
    pub const FRAC_BITS: u32 = 16;
    pub const EPSILON: Self = Self(1);
    pub const fn from_raw(r: i32) -> Self { Self(r) }
    pub const fn raw(self) -> i32 { self.0 }
}

impl core::ops::Add for Fixed16_16 { type Output = Self; fn add(self, o: Self) -> Self { Self(self.0 + o.0) } }
impl core::ops::Mul for Fixed16_16 {
    type Output = Self;
    fn mul(self, o: Self) -> Self { Self(((self.0 as i64 * o.0 as i64) / (1i64 << 16)) as i32) }
}
// Sub, Neg, Div, Rem, the *Assign variants, Display, Debug, …
```

`i64 / (1 << 16)` truncates toward zero, matching the stated semantics (an
arithmetic `>>` would floor). Overflow checks go through the same
`debug_assert`/`wrapping_*` policy the transpiler already uses for integers.

Formats are emitted by a small generator keyed on `(signed, N, M)`, not
hand-written per format. Literals are folded to their raw integer at
transpile time (`1.5` in `fixed16_16` → `Fixed16_16(98304)`).

## Interpreter (`boring run`) and self-hosted interpreter

New runtime variant `Value::Fixed { raw: i128, signed, n, m }` with a matching
`Type::Fixed { signed, n, m }`. The interpreter implements exactly the
semantics table above using `i128` math (hence no 2W-bit problem for any
format ≤ 64 bits). This is the reference semantics; the transpiled Rust, the
self-hosted interpreter (`boring/interpreter/`) and GPU lowerings must agree
with it, and the shared test-case corpus (`tests/cases/*.br` + `.expected`)
gets fixed-point cases so all four builds are checked against one answer.

## GPU backends

Fixed-point values in a `kernel` struct lower to their raw integer type; `+`,
`-`, compares are plain integer ops; `*` and `/` lower to a widening
multiply/shift.

| Backend | Support |
|---|---|
| CUDA | any `W ≤ 64` (native `long long` for the intermediate) |
| Metal (MSL) | `W ≤ 32` with a `long` intermediate; `W = 64` needs `mulhi`-based emulation |
| wgpu (WGSL) | no 64-bit integers: `W ≤ 16` direct in `i32`; `W = 32` needs 16-bit-limb emulation of the 64-bit product |
| ROCm | as CUDA |

First cut: formats not supported by a backend are a **compile error on that
target** (same stance as `float64` on Metal in
[float-width-types.md](float-width-types.md)), never a silent narrowing.
Buffer fields use the existing integer buffer packing (see wgpu
`uint8`/`int16` packing) on the raw type.

## Checker / grammar changes

- `spec/grammar.bnf`: extend `numeric_type` with
  `("fixed" | "ufixed") INT "_" INT` and the aliases
  `("i" | "u") INT "_" INT`. They are plain identifiers to the lexer; the
  checker recognises the shape (a digits-underscore-digits suffix after a
  reserved prefix), so no lexer change is needed — and no `.` in a type name
  (`real16.16` would collide with `t.0` tuple access and `a as T.16`).
- Scalar rules apply: `mut` on a fixed-point is an error (use `var`), no
  generic arguments, `<` after `as fixedN_M` is always an operator.
- `type Alias as fixed16_16` newtypes work like any numeric newtype
  (`Copy` derived).

## Phasing

1. **Core**: parser/checker names + validity rules, literals, strict mixing,
   `as` casts, `+ - * / %` and comparisons, formatting, interpreter, Rust
   newtype for `W ∈ {8, 16, 32, 64}`, tests in the shared corpus.
2. **Members**: `floor/ceil/round/fract/abs/min/max/clamp`, `wrapping*`/
   `saturating*`/`checked*`, and `mulRound`/`divRound` (round to nearest).
3. **GPU**: CUDA first, then Metal, then wgpu (with the limb emulation).
4. **128-bit formats** (`fixed64_64`, …): needs a 256-bit intermediate for
   `*`/`/` (limb decomposition). Deferred.
5. **Math functions** (`sqrt`, `sin`/`cos`, `ln`/`exp`) written in Boring
   itself (CORDIC / Newton), kept out of the compiler.

## Decisions

Questions raised while drafting, and how they were resolved:

1. **Rounding of `*` / `/`**: truncate toward zero. Round-to-nearest is
   available as `mulRound`/`divRound` (Phase 2); the default is unchanged.
2. **Inexact literals** (`0.1`): rounded to nearest, **no warning**. Only an
   out-of-range literal is an error.
3. **`fixed` mixed with `int`**: an error in the first cut (cast
   explicitly); only untyped literals are exempt. Possible later extension:
   allow `fixed * int`, `int * fixed`, `fixed / int` as exact scaling, if
   the cast proves too noisy in practice (then `+`, `-` and comparisons with
   `int` must be decided too).
4. **Narrowing casts**: bare `as` is allowed and every runtime conversion
   truncates toward zero (consistent with `float → int`); only literals
   round to nearest.
5. **`fixed` crate backend**: dropped. The generated newtype is the only
   backend; revisit only if Phase 5's math functions show a real need.
