# `float32` / `float64`: real fixed-width float types, `float` as a pure alias

> **Status: Draft — not implemented.** No lexer, parser, checker, interpreter,
> transpiler, or GPU-backend changes exist yet for anything described here.
> This document records the design as worked out in discussion, as a basis
> for implementation.

## Problem Statement

Boring already has `int8`/`int16`/.../`int128` and `uint8`/.../`uint128` as
**real, distinct runtime types** — each its own `Type`/`Value` variant, each
with its own Rust mapping, each refusing to mix implicitly with any other
fixed-width type (`CLAUDE.md`, `spec/grammar.bnf`'s `numeric_type` section).
`float` has no equivalent. Today `f32`/`f64` are accepted as spellings, but
both are pure aliases collapsing onto the *same* `Type::Float` /
`Value::Float(f64)` — there is exactly one float type at runtime, always
64-bit, regardless of which spelling was written:

```rust
// src/interpreter/mod.rs
aliases.insert("f32".into(), Type::Qualified(Box::new(Type::Float), OwnerQual::Stack));
aliases.insert("f64".into(), Type::Qualified(Box::new(Type::Float), OwnerQual::Stack));
```

The transpiler relays the literal Rust text `"f32"` for the type annotation
(`src/transpiler/helpers.rs`'s `normalize_type_name`), so `f32 x = 1.0`
*transpiles* to `x: f32` and compiles — but `boring run` still evaluates `x`
as a full `f64` underneath, with no truncation, no range narrowing, and no
cast machinery distinguishing it from plain `float`. Interpreted and
transpiled behavior silently diverge the moment a program does anything
beyond passing the value straight through (arithmetic, comparison, printing
a value near the edge of `f32`'s precision).

This is exactly the gap `int8`..`int128` already closed for integers, just
never done for floats. Two goals follow:

1. **`float32` and `float64` become real, independent runtime types** —
   mirroring `int8`/`int32`/etc. exactly: their own `Type`/`Value` variants,
   genuine 32-bit truncation in the interpreter (not a label over an `f64`),
   and the same no-implicit-mixing rule between distinct fixed-width types
   that integers already enforce.
2. **`float` becomes a pure alias of `float64`** — not a third, independent
   type the way `int`/`uint` are independent of `int64`/`uint64` (those stay
   pointer-width and orthogonal to the fixed-width family). `float` is
   Boring's *only* fixed-width numeric family member with no independent
   identity: write `float`, get exactly `float64`, indistinguishable at
   every stage after alias resolution. This is a deliberate asymmetry with
   `int`/`uint` (which map to `isize`/`usize`, never to `int64`/`uint64`) —
   called out explicitly here so it isn't mistaken for an oversight when the
   `int`/`uint` precedent is the first thing implementation will reach for.

**Three known correctness/design gaps get closed as a side effect**, not
scope creep — each is a consequence of `float32`/`float64` becoming real
types with a real identity, not a new problem being bundled in:

- **Metal (MSL) device code lies about precision today.** MSL has no native
  `double`; `Type::Float` on Metal's device side (`src/transpiler/metal/device.rs`)
  already silently maps to 32-bit `float`, meaning a Boring `float` (nominally
  64-bit) used inside a `kernel` targeting Metal is *actually* computed at
  32-bit precision on-device today, with nothing telling the author. Once
  `float64` is a real, checked type, this stops being representable at all —
  see [Metal below](#metal-mslfloat64-device-side-is-a-compile-error-not-a-silent-narrowing).
- **The generic `'gpu'unified` residency path already narrows `float` to
  32-bit on the device side, on purpose — `float32`/`float64` make that
  explicit instead of implicit.** `kernel_host_scalar_type` (device-facing)
  maps `Type::Float` to `"f32"` while `kernel_host_element_type`
  (host-facing) maps the same `Type::Float` to `"f64"`
  (`src/transpiler/emit_kernel.rs`) — a deliberate choice (this path targets
  wgpu, which has no 64-bit float at all), not a bug; see
  [§6's note](#hostdevice-buffer-narrowing-in-the-generic-gpuunified-residency-path--not-a-bug-after-all)
  for why an earlier revision of this document mischaracterized it as one.
  `float32`/`float64` as separate types at least make the two functions'
  differing treatment legible by name instead of both silently reading the
  same `Type::Float`.
- **`BoringError` boxes every fixed-width scalar through the same generic
  path as arbitrary user types, today.** `catch Int8:`/`catch Uint32:`/etc.
  already exist and already work, but only by falling through to
  `BoringError::Other(TypeId, Box<dyn BoringVal>)` — the identical mechanism
  used for `catch MyEnum:`/`catch MyStruct:`. A plain `i8` or `u32` gets
  heap-allocated and type-erased behind `dyn Any` to be thrown, then
  downcast back on catch — real cost and indirection for what is, at
  runtime, just a few bytes. `float32`/`float64` were about to join that
  same fallback (an earlier revision of this document accepted that
  explicitly, see the retired paragraph in
  [§7](#7-boringerrorscalar--one-dedicated-variant-for-the-whole-fixed-width-family-replacing-other)
  below) — instead, this document now also fixes it for the *entire*
  fixed-width family, `int8`..`int128`/`uint8`..`uint128` included, not just
  the two types this document is nominally about. This is a genuine scope
  expansion beyond floats, called out explicitly rather than smuggled in:
  it was surfaced by asking "what should `catch Float32:` do" and the answer
  turned out to generalize to a pre-existing gap, not a floats-only fix.

## Proposed Design

### 1. `Type::Float32` / `Type::Float64` — real variants, not qualified aliases

```rust
pub enum Type {
    ...
    Uint128,
    Float32,   // new — Rust f32
    Float64,   // new — Rust f64; identical to today's Type::Float in every mapping
    ...
}
```

`Type::Float` is retired as a distinct variant used past alias resolution —
every site that pattern-matches `Type::Float` today (~450 occurrences across
`src/`, see the exploration notes at the end of this document) is a
mechanical rename to `Type::Float64`, since `Type::Float`'s existing mapping
*is* what `Float64` should do. This is the lowest-risk path: no site
currently distinguishing "float" from "float64" behavior needs new logic,
only a rename. `Value::Float(f64)` similarly gains siblings:

```rust
pub enum Value {
    ...
    Uint128(u128),
    Float32(f32),   // new — genuine 32-bit storage, not a label
    Float64(f64),   // new — identical layout/behavior to today's Value::Float(f64)
    ...
}
```

`Value::Float32` is the one place actual 32-bit truncation must be real:
`Value::Float32(3.14_f64 as f32)`, not `Value::Float32(3.14)` reinterpreted —
mirrors `Value::Int8(i8)` doing a genuine range-checked narrowing, not a tag
over an `isize`.

### 2. `float` resolves to `Type::Float64` at the alias table — nowhere else sees `Type::Float`

Both the Rust interpreter's alias table (`src/interpreter/mod.rs`) and the
self-hosted interpreter's parser (`boring/interpreter/parser_core.br`) are
where `float`/`Float`/`f64` collapse into `Type::Float64` and `f32` collapses
into `Type::Float32`:

```rust
aliases.insert("float".into(),   Type::Float64);   // was Type::Float
aliases.insert("Float".into(),   Type::Float64);   // was Type::Float
aliases.insert("f64".into(),     Type::Float64);   // was Type::Qualified(Float, Stack)
aliases.insert("f32".into(),     Type::Float32);   // was Type::Qualified(Float, Stack)
aliases.insert("float64".into(), Type::Float64);   // new spelling
aliases.insert("Float64".into(), Type::Float64);   // new spelling
aliases.insert("float32".into(), Type::Float32);   // new spelling
aliases.insert("Float32".into(), Type::Float32);   // new spelling
```

Everything downstream of alias resolution — checker, interpreter arithmetic,
transpiler, GPU backends — only ever sees `Type::Float32` or `Type::Float64`,
never a third `Type::Float` value. This is the same shape `int`/`i64`/`Int64`
already use to converge on one canonical variant; `float` simply converges on
`Float64` instead of getting its own variant. `Type::Float` is removed from
the enum entirely once every site is migrated — keeping it around as a dead
variant would reopen exactly the "which one does this code path mean"
ambiguity this document exists to close (mirrors [mut-type-modifier.md](mut-type-modifier.md)'s
own retirement of a shortcut once its replacement covers every case).

The parser's `parse_type_base` (`src/parser/parse_type.rs`) gains
`"Float32" => Type::Float32, "Float64" => Type::Float64` alongside the
existing `"Float" => Type::Float` line (updated to `Type::Float64`); the
lowercase spellings continue to resolve via the alias table, as `float`/`f32`
already do today. `is_type_name` (`src/parser/mod.rs`) gains
`"float32" | "float64"` to its lowercase type-name list — `f32`/`f64` are
already present there.

### 3. Mixing rule: strict, matching `int8`/`int16`

`float32` and `float64` do not mix implicitly, exactly like distinct
fixed-width integers today:

```boring
float32 a = 1.0
float64 b = 2.0
let c = a + b          # ERROR — cannot mix float32 and float64 without an explicit cast
let c = (a as float64) + b   # OK
```

An untyped float literal (`3.14`) mixes freely with either, resolved by
context — exactly as an untyped int literal (`42`) mixes freely with
`int8`/`int32`/etc. today:

```boring
float32 a = 1.0
let c = a + 3.14        # OK — 3.14 resolves to float32 in this context
```

`float` participates in this rule as `float64` — since it *is* `float64`
after alias resolution, `float_var + float32_var` is exactly as much an
error as `float64_var + float32_var`, with no special case needed: the
checker never sees `float` as a separate case to reason about.

This closes the current inconsistency where fixed-width integers already
enforce strict mixing (`docs/book.md`'s "Fixed-width integers" section) but
floats have no fixed-width family to be strict about at all yet.

### 4. Casts: `as float32` / `as float64` truncate; `else` fallback parses with the matching Rust type

`x as float32` / `x as float64` follow the same shape as `as int8`/`as
uint32`: a genuine narrowing conversion (`as f32` / `as f64` in the emitted
Rust and in the interpreter), with no range check needed in either direction
(unlike integer narrowing, every finite `f64` converts to *some* `f32` value —
overflow saturates to `±inf` in Rust's own `as` semantics, which Boring
inherits unchanged rather than inventing a stricter float-narrowing rule).

The `x as float32 else default` string-parse form (used today for `x as
float else default`, e.g. parsing user input) gets its own dedicated arm
parsing with `str::parse::<f32>()` / `str::parse::<f64>()` respectively,
matching `as float`'s existing arm — **not** the generic fallback path
`f32`/`f64` currently take (`src/transpiler/emit_expr.rs`'s
`is_specific_numeric_type(&dst_ty) && dst_ty != "f32" && dst_ty != "f64"`
guard, which today excludes `f32`/`f64` from ever getting *any* real
parse-and-validate behavior here — see gap notes below). This is a
correctness fix riding along with real-type status, not new scope: once
`float32`/`float64` are checked types the size of `int8`/`uint32`, they
should get the same cast infrastructure those already have, and today's
`f32`/`f64` spellings were never wired into it despite already existing as
type-annotation spellings.

### 5. `--target kernel` (Rust-for-Linux, no_std): both forbidden, same reason as today's `float`

The kernel target already forbids `float` outright (`src/validator/kernel.rs`:
*"float is not allowed in kernel context — FPU is disabled"*) because kernel
code runs with the FPU unavailable. `float32` and `float64` are both banned
for the identical reason — narrower storage doesn't change whether the FPU
exists. The validator's `Type::Float` match arm becomes
`Type::Float32 | Type::Float64`, and the error message generalizes to name
whichever was actually written (`"{float32,float64} is not allowed in kernel
context — FPU is disabled"`, not a message that only ever says "float"
regardless of which spelling triggered it).

### 6. GPU kernel backends (`kernel Name:` — CUDA/ROCm/Metal/wgpu)

This is the one area where `float32`/`float64` genuinely need **different**
per-backend rules, not just a mechanical rename — floats have hardware
support constraints that fixed-width integers mostly don't:

| Backend | `float32` (device) | `float64` (device) |
|---|---|---|
| CUDA | `float` | `double` — full native support |
| ROCm | `float` | `double` — full native support |
| Metal (MSL) | `float` | **compile error** — see below |
| wgpu (WGSL) | `f32` | **compile error** — WGSL has no `f64` at all |

#### Metal (MSL): `float64` device-side is a compile error, not a silent narrowing

MSL has no native `double`. Today, `Type::Float` (nominally 64-bit)
transpiles to MSL `float` on the device side unconditionally
(`src/transpiler/metal/device.rs`) — meaning a `kernel` targeting Metal that
declares a `float` field is silently computed at 32-bit precision on-device
today, with the host side still believing it's 64-bit. Once `float32` and
`float64` are distinguishable, this stops being representable as a silent
narrowing: **`float64` (and therefore also plain `float`, its alias) used in
a Metal device-side kernel position is a validator error** — "MSL has no
native 64-bit float; use `float32` in this kernel, or target a backend that
supports 64-bit floats (`cuda`, `rocm`)". `float32` continues to map to MSL
`float` exactly as it (silently, incorrectly labeled) does today — the
mapping doesn't change, only the fact that using `float`/`float64` here now
surfaces instead of being swallowed.

#### wgpu (WGSL): `float64` is a compile error for the same underlying reason

WGSL has no 64-bit float type at all (unlike Metal, not even a narrowing
target exists) — `float64`/`float` in a `kernel` targeting `--target wgpu`
is a validator error: "WGSL has no 64-bit float type; use `float32`."
`float32` maps to WGSL's native `f32` (`src/transpiler/wgpu/device.rs`
already does this for `Type::Float` today — no behavior change, same
correctness gap being closed as Metal's).

#### Host/device buffer narrowing in the generic `'gpu'unified` residency path — not a bug after all

**An earlier revision of this section claimed `kernel_host_scalar_type`
(device-facing, narrows to `"f32"`) vs. `kernel_host_element_type`
(host-facing, stays `"f64"`) — both in `src/transpiler/emit_kernel.rs` —
disagreeing for a `float`/`float64` buffer was a bug this document would fix
"by construction." Investigated further and reverted: it isn't a bug.**
These two functions are specifically the generic, backend-agnostic
`'gpu'unified` residency path (interprocedural resident values, the feature
`with`/dual-typed-parameter tests exercise) — `kernel_host_scalar_type`'s own
doc comment says so directly: *"GPU buffers always use 32-bit elements"*,
modeled on `wgpu::host::host_scalar_type` because wgpu is this path's primary
real target and WGSL has no 64-bit float at all. Making `Type::Float64` stop
narrowing here (as an earlier draft of this feature did) breaks exactly the
case it exists for — a `float`/`float64` value flowing through this generic
path on `--target wgpu` needs its GPU-side buffer to genuinely be `f32`,
independent of whatever the host-side "general convention" (`f64`) reads it
back as. `Type::Float32` needs no narrowing here since it's already 32-bit;
`Type::Float64` keeps the pre-existing narrow-to-`f32` behavior, unchanged.

This is a **different code path** from an actual `kernel struct`'s own field
type, which each backend maps in its own `device.rs`/`host.rs` (§6 above) —
those correctly do NOT narrow `float64` for CUDA/ROCm (native `double`) and
correctly DO reject it outright for Metal/wgpu (`msl_unsupported_f64`/
`wgsl_unsupported_f64`). The narrowing discussed here is specific to the
generic residency functions, which have no per-backend knowledge to
condition on. One real, pre-existing gap remains, called out honestly rather
than silently left implied: this generic path silently narrows a
`float`/`float64` resident value to `f32` instead of erroring the way an
actual `kernel struct` field declared `float64` now does on Metal/wgpu — a
narrower, pre-existing inconsistency (identical to `float`'s behavior before
this document), not introduced by float32/float64 and not fixed here.

### 7. `BoringError::Scalar` — one dedicated variant for the whole fixed-width family, replacing `Other`

**An earlier revision of this section proposed no dedicated variant at all**
— routing `catch Float32:`/`catch Float64:` through `BoringError::Other`,
matching what `catch Int8:`/`catch Uint32:`/etc. already do today. Retired:
on reflection, sharing `Other` between fixed-width scalars and arbitrary
user enums/structs was never the right precedent to extend — it's an
accident of `Int8`..`Uint128` never having been given their own path when
they shipped, not a deliberate design worth perpetuating onto
`float32`/`float64` too. `Other` (`TypeId` + `Box<dyn BoringVal + Send +
Sync>`, downcast via `Any`) exists for types the runtime can say nothing
structural about in advance — a user's own enum or struct, defined anywhere,
with arbitrary shape. A fixed-width numeric is the opposite: the compiler
already knows, statically, that it is exactly one of twelve possible kinds,
each a few bytes, Copy, with no fields to speak of — paying for a heap
allocation and a dynamic downcast to throw an `i8` is real, avoidable cost
that the closed, fully-enumerable nature of this family should never have
incurred in the first place.

**New shape: one variant, not twelve.** Rather than mirroring `Int8(i8)` ..
`Float64(f64)` as twelve separate `BoringError` variants (rejected — it
would duplicate the twelve-way match in `Display`, in every `catch`
dispatch site, and in mangling, for a benefit no different from the single
shared variant below), `BoringError` gains **one** new variant carrying a
kind tag plus a bit-packed raw value, wide enough for the largest member
(`int128`/`uint128`, 128 bits):

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
enum ScalarKind {
    Int8, Int16, Int32, Int64, Int128,
    Uint8, Uint16, Uint32, Uint64, Uint128,
    Float32, Float64,
}

enum BoringError {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(&'static str),
    String(Arc<str>),
    Scalar(ScalarKind, u128),   // new — kind tag + raw bits, reinterpreted per kind
    Other(std::any::TypeId, std::boxed::Box<dyn BoringVal + Send + Sync>),
}
```

`Int8`..`Int64` sign-extend into the `u128` slot (bit-reinterpreted as
`i128`, `as`-truncated back to the exact original width on read — lossless,
since narrowing an `as`-widened-then-narrowed integer round-trips exactly);
`Uint8`..`Uint64` zero-extend the same way; `Int128`/`Uint128` fill the slot
directly; `Float32`/`Float64` store `f32::to_bits()`/`f64::to_bits()`
zero-extended into the same `u128`, reinterpreted via `f32::from_bits`/
`f64::from_bits` on the way out (bit pattern only — never a numeric `as`
cast between the two, which would silently change the value). No heap
allocation, no `TypeId`, no `dyn Any` — `Display` and `catch` dispatch both
become a plain match on `ScalarKind` followed by one bit reinterpretation,
exactly the same shape `Int(i64)`/`Float(f64)` already get, just
parameterized over which of the twelve kinds it is instead of having a
fixed width baked into the variant itself.

`catch Int8:` / `catch Float32:` / etc. each compile to a guard on
`ScalarKind`:

```rust
BoringError::Scalar(ScalarKind::Int8, __bits) => { let error: i8 = __bits as i8; ... }
BoringError::Scalar(ScalarKind::Float32, __bits) => { let error: f32 = f32::from_bits(__bits as u32); ... }
```

`catch Float:` (i.e. `float`/`float64`, since `float` is `float64`'s alias,
[§2](#2-float-resolves-to-typefloat64-at-the-alias-table--nowhere-else-sees-typefloat))
keeps its existing fast-path `BoringError::Float(f64)` arm — `Scalar` is
strictly for the fixed-width family (`float32`/`float64`, `int8`..`int128`,
`uint8`..`uint128`); the bare, flexible `int`/`uint`/`float` kinds keep
their own dedicated `Int(i64)`/`Float(f64)` variants exactly as today,
unaffected by any of this.

**`Other` still exists, unchanged, for exactly what it always covered:**
arbitrary user-defined enums/structs (`catch MyEnum:`, `catch CalcError:`),
where the runtime genuinely cannot know the shape in advance and the
`TypeId` + `Any`-downcast machinery is the only option. This document
narrows what `Other` is *for*, it does not remove it.

### 8. Interpreter arithmetic, methods, comparisons: duplicate `Value::Float`'s existing logic

`src/interpreter/eval_expr.rs`'s binary operators (`+`/`-`/`*`/`/`/`%`),
`src/interpreter/methods.rs`'s ~25 native float methods (`sqrt`, `abs`,
`floor`, `ceil`, `round`, `exp`, `log`/`log2`/`log10`, trig functions,
`pow`, `atan2`, `clamp`, …), and comparison/`partial_cmp` logic all gain a
`Value::Float32(f32)` arm alongside the existing `Value::Float64(f64)` arm
(renamed from `Value::Float`), each doing the identical operation at the
matching precision — Rust's `f32` has the same method surface as `f64`, so
this is direct duplication, not new algorithm design. Mixed-width arithmetic
(`Value::Float32` operated on with `Value::Float64`) is a runtime error at
the interpreter level mirroring the checker's static rejection (§3) — the
interpreter is the last line of defense for code paths the checker doesn't
statically cover (e.g. `any`-typed dynamic values).

### 9. Stdlib (`stdlib/builtins.br`): duplicate each `float` signature

Every `native` float function declared in `stdlib/builtins.br` (`sqrt`,
`abs`, `floor`, `pow`, the trig family, `clamp`, `sign`, `isNaN`,
`isInfinite`, `min`, `max`, ~25 signatures) gets a `float32` and a `float64`
overload alongside the existing `float` signature — since `float` is now a
pure alias of `float64`, the `float64` overload and the existing `float`
signature are the same function under two names; only the `float32`
overload is genuinely new declaration text. Boring's overload resolution
(already used for the existing `int8`/`int16`/… numeric builtins, per
`normalize_type_name`/mangling in the transpiler) picks the right one from
the argument's static type, same mechanism as today.

## Grammar changes required

```bnf
# primitive_type keeps "float" (now documented as an alias of float64,
# not an independent type) — no grammar change needed for the keyword itself,
# only for the two new spellings, which join numeric_type alongside f32/f64:
numeric_type ::= "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
               | "u8" | "u16" | "u32" | "u64" | "u128" | "usize"
               | "f32" | "f64" | "float32" | "float64"
```

No new grammar production is needed — `float32`/`float64` parse as ordinary
identifiers resolved through the alias table exactly as `f32`/`int8`/etc.
already do (§2); the lexer requires no changes at all (confirmed during
exploration: `float`/`int8`/`f32` are not lexer keywords today, just
`Ident` tokens resolved later).

`spec/grammar.bnf`'s comment block documenting the fixed-width family
(currently: *"Floats: f32 f64 (→ interpreter: float'copy)"*) is rewritten to
match the integer rows immediately above it — naming `float32`/`float64` as
real, distinct runtime types with the same strict-mixing rule, and calling
out `float` as the odd one out (alias of `float64`, not an independent
pointer-width-style type like `int`/`uint`).

## Interactions and invariants

- **`is_scalar_type` (`src/checker/mod.rs`) and `is_copy_type`
  (`src/transpiler/emit_top.rs`) gain `Type::Float32 | Type::Float64`** in
  their match arms, alongside the `Type::Float` → `Type::Float64` rename —
  both already list every fixed-width int variant explicitly, so this is
  the same shape extended by two entries. **Pre-existing gap fixed in the
  same pass**: neither of these lists currently include `"f32"`/`"f64"` in
  their `Type::Named` string-matching arms at all — an existing
  inconsistency (int aliases like `"i8"`/`"u32"` are covered, `"f32"`/`"f64"`
  are not) that becomes impossible to leave unfixed once floats get the same
  named-alias treatment integers already have.
- **`estimate_size_inner` (`src/transpiler/mod.rs`)**, used for strict-mode
  auto-boxing size estimates, gains `Type::Float32 => Some(4), Type::Float64
  => Some(8)` — today's `Type::Named` arm only recognizes `"float"`/`"f64"`
  as 8 bytes and has no `"f32"` entry at all (silently falls through to
  whatever the default case does) — fixed as part of this work, not a
  separate cleanup.
- **Mangled names** (`src/transpiler/helpers.rs`'s `mangle_type_name`, used
  for overload resolution) get `Type::Float32 => "float32"`, `Type::Float64
  => "float64"` — distinct manglings, so `describe(float32)` and
  `describe(float64)` overloads never collide, matching how `describe(int32)`
  and `describe(int64)` already don't.
- **`normalize_type_name`** (`src/transpiler/helpers.rs`) becomes the single
  place where every spelling converges: `"float32"|"Float32"|"f32" =>
  "f32"`, and `"float64"|"Float64"|"float"|"Float"|"f64" => "f64"` — `float`
  is textually folded into the `f64` bucket here, exactly mirroring how the
  alias table folds it into `Type::Float64` at the `Type` level (§2). Two
  independent tables encoding the same alias is not duplication to resolve —
  one operates on `Type` values (interpreter/checker), the other on raw
  `Type::Named` strings the transpiler sees before/instead of going through
  full type resolution; both must agree, but they're inherently separate
  data structures already, matching how every other fixed-width alias
  (`"i8"` vs `Type::Int8`) is already handled twice today.
- **`docs/book.md`'s Scalar Types table** documents `float32`/`float64` as
  the two real members of the family, with `float` called out as a
  documented alias of `float64` rather than a third table row — matching how
  this document's Problem Statement frames it, so the book and the design
  doc never disagree about which one is the "real" type and which is sugar.
- **`boring_type_to_boring_val_arms` (`src/transpiler/helpers.rs`)** — today
  emits a real match arm only for `Int`/`Float`/`Bool`/`String`; every other
  type name (including, today, `Int8`/`Uint32`/etc.) falls into a literal
  `/* unreachable catch {} */` placeholder arm, with the actual dispatch
  handled entirely by a separate `TypeId`-guard code path in
  `emit_flow.rs`. Once `BoringError::Scalar` exists (§7), this function
  gains one real arm per `ScalarKind` (`"Int8" | "int8" | "i8"` →
  `BoringError::Scalar(ScalarKind::Int8, __bits)` with the `i8`
  reinterpretation, and so on through `Float64`), and the corresponding
  `TypeId`-guard branch in `emit_flow.rs` (`"Named error types (enums) →
  BoringError::Other guard arms"`) stops being reached for any fixed-width
  scalar — it still exists, unchanged, for actual named enum/struct types.

## Implementation checklist

1. `src/ast/mod.rs` — add `Type::Float32`, `Type::Float64`; remove
   `Type::Float` once every reference is migrated (step 2 makes the
   compiler enumerate every site that needs updating — do not pre-audit by
   hand, same rationale as [mut-type-modifier.md](mut-type-modifier.md)'s
   `BindingKind::is_mutable()` migration).
2. Mechanical rename pass: every `Type::Float` → `Type::Float64` across
   `src/checker/`, `src/transpiler/` (including all four GPU backend
   dirs — `cuda/`, `rocm/`, `metal/`, `wgpu/` — and `emit_kernel.rs`,
   `kernel/helpers.rs`), `src/validator/kernel.rs`, `src/interpreter/`
   (including `eval_gpu.rs`'s `ThreadValue`). The Rust compiler's
   exhaustiveness check on `match` surfaces every site once `Type::Float`
   is removed from the enum — treat that error list as the authoritative
   worklist, not this document's line numbers (which are a starting map,
   not a guarantee against drift).
3. `src/interpreter/mod.rs` — alias table: `float`/`Float`/`f64`/`float64`/
   `Float64` → `Type::Float64`; `f32`/`float32`/`Float32` → `Type::Float32`
   (§2). Add `Value::Float32(f32)`/`Value::Float64(f64)` to the `Value` enum
   (rename existing `Value::Float(f64)` → `Value::Float64(f64)` first, then
   add `Float32` alongside).
4. `src/interpreter/eval_expr.rs`, `methods.rs`, `exec.rs`, `call.rs`,
   `eval_gpu.rs` — duplicate every `Value::Float`-matching arm for
   `Value::Float32`, with genuine `as f32` truncation where values cross
   from `Value::Int`/`Value::Float64`/parsed strings into a `Float32` slot
   (§8). Enforce the strict no-implicit-mixing rule between `Float32` and
   `Float64` at arithmetic/comparison sites (§3), mirroring existing
   fixed-width-int mixing errors.
5. `src/checker/mod.rs` — `is_scalar_type` and any other `Type::Float`
   match arms gain `Float32`/`Float64` (rename + extend); `is_scalar_type`'s
   `Type::Named` string arm gains `"float32"|"float64"|"f32"|"f64"` (closing
   the pre-existing `f32`/`f64` gap noted above); `src/validator/kernel.rs`
   bans `Float32`/`Float64` alongside the renamed `Float64`, with a
   generalized error message (§5).
6. `src/parser/parse_type.rs` — `"Float32"`/`"Float64"` capitalized
   branches; `src/parser/mod.rs`'s `is_type_name` — add
   `"float32"|"float64"` to the lowercase type-name list.
7. `src/transpiler/emit_top.rs` — `emit_type`/`is_copy_type`: add
   `Float32`/`Float64` arms (§Interactions); `src/transpiler/helpers.rs` —
   `normalize_type_name`, `mangle_type_name`, `estimate_size_inner`'s
   `Type::Named` arm, `is_specific_numeric_type`, `wider_numeric_type`/`rank`
   (verify `"float32"`/`"float64"` route to the same rank entries as
   `"f32"`/`"f64"` already do). `emit_expr.rs`/`emit_let.rs`/`emit_stmt.rs`
   — extend every `Type::Float`-matching list (~15 sites) and give
   `float32`/`float64` (and, while there, the already-existing but
   unwired `f32`/`f64` spellings) real `as float32 else default` /
   `as float64 else default` parse-and-validate cast arms (§4), instead of
   the generic fallback they silently get today.
8. GPU backends — `src/transpiler/{cuda,rocm}/{device,host}.rs`:
   `Type::Float32 => "float"`, `Type::Float64 => "double"` (both sides, no
   error case — full native support). `src/transpiler/metal/device.rs`:
   `Type::Float32 => "float"`, `Type::Float64 =>` **validator error**
   (§6); `metal/host.rs` keeps `Type::Float64 => "f64"` (host-side Rust has
   no such restriction) and `Type::Float32 => "f32"`.
   `src/transpiler/wgpu/{device,host}.rs`: `Type::Float32 => "f32"`,
   `Type::Float64 =>` **validator error** (§6, no host-side exception —
   wgpu buffers are shared layout, not just device code). `emit_kernel.rs`'s
   `kernel_host_scalar_type`/`kernel_host_element_type` — both map
   `Float32`/`Float64` identically on both sides (§6, "fixed by
   construction").
9. **`BoringError::Scalar` (§7)** — add the `ScalarKind` enum and the
   `Scalar(ScalarKind, u128)` variant to the generated Rust prelude
   (`src/transpiler/mod.rs`'s embedded `BoringError` definition, string-built
   today, see the exploration notes at the end of this document for exact
   line ranges); update its `Display` impl with the bit-reinterpretation
   match (§7); `emit_flow.rs`'s throw-value emission gains a `Scalar`
   construction arm for all twelve kinds (`Box::new(BoringError::Scalar(...))`
   instead of the boxed `Other(TypeId::of::<i8>(), Box::new(v))` path it uses
   today for every fixed-width numeric); `helpers.rs`'s
   `boring_type_to_boring_val_arms` gains the twelve real match arms
   described in Interactions above, replacing today's `/* unreachable catch
   {} */` placeholder for these specific type names; verify the existing
   `TypeId`-guard catch-dispatch path in `emit_flow.rs` is now reached only
   for genuine named enum/struct types, never for `Int8`..`Uint128`/
   `Float32`/`Float64`. **This step's scope is the whole fixed-width family,
   not just `float32`/`float64`** — `int8`..`int128`/`uint8`..`uint128` move
   off `Other` in the same pass, since it's the same mechanism (see the
   Problem Statement's third bullet).
10. `stdlib/builtins.br` — duplicate each `native` float signature into
    `float32`/`float64` overloads (§9); `stdlib/string.br`'s `parseFloat`
    gains `parseFloat32`/`parseFloat64` siblings (keep bare `parseFloat` as
    the `float64`/default form, matching `float`'s alias status).
11. `boring/interpreter/*.br` (self-hosted interpreter) — same rename +
    extend pass as steps 1–4, scaled to its simpler model: `ast.br`'s
    `Type`/`value.br`'s `Value` enums, `parser_core.br`'s `parse_type_base`
    table, `eval.br`'s arithmetic/cast/method/display logic, `exec.br`,
    `methods.br`. The self-hosted interpreter's own error-handling model is
    simpler than the Rust transpiler's `BoringError` (no typed-catch
    downcast machinery to mirror) — confirm during implementation whether
    step 9's `Scalar` split has any self-hosted equivalent to update at all,
    or whether it's Rust-transpiler-only.
12. `spec/grammar.bnf` — `numeric_type` gains `"float32" | "float64"`;
    rewrite the fixed-width-family comment block (§Grammar changes).
13. `docs/book.md` — Scalar Types table split into `float32`/`float64` rows
    with `float` documented as `float64`'s alias (not a third row); GPU
    width-support table gains float rows per backend (§6); `CLAUDE.md`'s
    "Common types" table gets the same two-row treatment; the "Advanced —
    Error handling internals" section's `BoringError`/typed-`catch`
    documentation is updated to describe `Scalar` and the narrowed meaning
    of `Other` (§7).
14. Tests — new `tests/cases/float_width_cross_eq.br` (mirroring the
    existing `uint_int_cross_eq.br`, the closest existing precedent for
    "strict mixing between same-family fixed-width types") covering:
    `float32`/`float64` rejection when mixed directly, literal mixing with
    either, `as float32`/`as float64` casts (including the `else` fallback
    form), and `float` behaving identically to `float64` at every one of
    these sites. A new `tests/cases/scalar_catch.br` covering `catch Int8:`/
    `catch Uint32:`/`catch Float32:`/`catch Float64:` round-tripping the
    thrown value correctly through `BoringError::Scalar` — no such test
    exists today (the current `Other`-based path has no dedicated coverage
    either, per the exploration notes). GPU codegen tests
    (`tests/{cuda,rocm,metal,wgpu}_codegen.rs`) get cases for the Metal/wgpu
    `float64` compile-error paths specifically, since those are new failure
    modes with no existing test coverage. The existing `numeric.br` test
    (already declares an `f32` function) should be re-verified once `f32`
    stops being a `Type::Float` alias — its expected output may need
    updating if it previously relied on `f32` behaving exactly like `f64` at
    runtime.

## Explicitly out of scope (future work, not corollaries)

- **A dedicated `BoringError` variant per fixed-width type** (`Int8(i8)`,
  `Float32(f32)`, … — twelve variants) — decided against in §7 in favor of
  one shared `Scalar(ScalarKind, u128)` variant; revisit only if a future
  need arises for something a shared representation can't express (none
  identified while writing this document).
- **A stricter float-narrowing cast** (e.g. erroring on `as float32` when the
  value doesn't round-trip losslessly) — Rust's own `as f32` semantics
  (silent precision loss, overflow saturates to infinity) are inherited
  unchanged; a checked/fallible narrowing cast would be a separate, additive
  proposal (`x as float32 checked` or similar), not assumed here.
- **Per-field/per-element `mut float32`** — this document is orthogonal to
  [mut-type-modifier.md](mut-type-modifier.md); `mut` on any scalar
  (`float32` included) remains a checker error under that document's rules,
  unchanged by anything here.
- **New float constants for `float32`** (`PI` etc. are `Value::Float64`
  today) — no product requirement surfaced for `float32`-native constants;
  `PI as float32` covers the need if it ever arises.
- **Half-precision (`float16`/`bfloat16`)** — genuinely different hardware
  support story (only some GPU backends, no native Rust scalar type without
  a crate dependency); not a corollary of this document's all-native-Rust-type
  approach and would need its own design work.
