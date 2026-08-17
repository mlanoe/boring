# Known issues found while spiking a pure-Boring `BigUint`

**Status (2026-08-20): all 12 items are now fixed**, including item 5's
`boring build` (transpiler) gap, closed last — see its own "Fixed" note.
Items 1-4 (the original `BigUint`/`BigInt`/`BigFraction` pass) and 5-8 (a
same-day follow-up adding `ScratchNumber`) are fixed — see each item's own
"**Fixed:**" note for the exact file/line. Three side-effect bugs found while
fixing 1-8 are also now fixed (commits `bb4f18d`, `77cd91b`, `81c8216` on
`feat/breakout-boring-toml-migration`): (a) the interpreter's multi-clause
`catch Enum.A: / catch Enum.B:` dispatch always ran the first clause
regardless of the actual thrown variant; (b) the transpiler's "`catch
TypeName:` + `match error:`" style (book.md's error-handling "Style 2") failed
to compile the generated Rust; (c) a `type def`/`type req` method's `throws`
(typed or untyped) was never wrapped in `Result<T, E>` by the transpiler at
all. Items 9-12 (below) were a *different* batch: found 2026-08-20 while
actually wiring `ScratchNumber` into `scratch-boring`'s real `scratch.br` and
building it via `boring build --emit-rust` — the interpreter-only validation
of items 1-8 (via `boring run`) never exercised the transpiler backend, and
these four were transpiler-only bugs invisible to that validation. Now fixed
too — see each item's own "**Fixed:**" note. `boring/spikes/biguint_spike.br`
and `scratch-boring/boring/scratch.br` were both intentionally left untouched
throughout all of this (they still carry their original workarounds for every
item, including 9-12's) — `scratch-boring` was re-verified end-to-end against
the fully fixed compiler (`boring/boring/regen.sh` + `cargo build` + `cargo
test`, all suites green including `fibonacci.rs`/`scratch_number.rs`), and
its regenerated `src/boring_gen.rs` came out **byte-identical** to the
already-committed one — confirming the 9-12 workarounds still produce correct
code post-fix, just no longer required.

Found 2026-08-19 while writing `scratch-boring/boring/spikes/biguint_spike.br` — a
pure-Boring arbitrary-precision unsigned integer (little-endian `[uint32]` limbs),
meant as a candidate replacement for `scratch-boring`'s hand-written Rust
`num_bigint` dependency (see that project's README.md / CLAUDE.md memory
`boring-goal-pure-boring-rust-parity`). All four items below were worked around in
the spike itself (it fully passes, 17/17 assertions) — this doc exists to track the
underlying language/interpreter issues for a real fix, not to report a blocked task.

Only `boring run` (the tree-walk interpreter) was exercised — none of these were
re-checked against `boring build`'s transpiled-Rust path, so it's possible some are
interpreter-only gaps that already work when transpiled. Worth confirming before
fixing, since the fix (and its scope) may differ between the two backends.

## 1. Integer literal range-checked before its `as` cast, not after

```boring
let x = 18446744073709551615 as uint64   # u64::MAX
```

fails with `error: integer literal out of range`, even though the literal is
immediately cast to `uint64` (which can hold it). The range check appears to run
against the literal's default type (`isize`/`int`) before the `as uint64` conversion
is applied, rather than validating against the cast's target type.

**Workaround used:** build the value arithmetically instead of as a literal —
`let u32_max64 = 4294967295 as uint64; let u64_max = (u32_max64 << 32) | u32_max64`.

**Suggested fix:** when parsing/checking an integer literal immediately followed by
`as <IntType>`, validate the literal's range against `<IntType>`, not the default
inferred type.

**Fixed:** the actual root cause was one level lower than "range-checked before the
cast" — the literal couldn't even be *lexed* past `i64::MAX`, regardless of any
following cast (`TokenKind`/`ExprKind::Int` only ever stored `i64`). Added a new
`TokenKind::UInt64(u64)` / `ExprKind::UInt64(u64)` literal form, produced by the
lexer only when `i64` parsing overflows but `u64` parsing still succeeds
(`src/lexer/mod.rs`, `lex_number`, ~line 1048). It evaluates directly to a real
`Value::Uint64` in the interpreter (`src/interpreter/eval_expr.rs`), so the
*existing* checked `cast_value` logic (`src/interpreter/methods.rs`) already does
the right thing from there with no further changes — `18446744073709551615 as
uint64` now round-trips exactly, and `as uint32` correctly yields `nil` (out of
range), matching the documented checked-cast semantics (book.md line 394).
Threaded the new AST variant through the transpiler (`src/transpiler/emit_expr.rs`,
`src/transpiler/kernel/emit_expr.rs`, `src/transpiler/helpers.rs`), checker
(`src/checker/mod.rs`), and validator (`src/validator/kernel.rs`) — 9 sites total,
found via the compiler's own non-exhaustive-match errors after adding the enum
variant. Regression tests: `tests/cases/int_literal_overflow_cast.{br,expected}`
(registered in both `tests/run.rs` and `tests/transpile.rs`, all 4
mode/threading combos pass) plus two lexer unit tests in `src/lexer/mod.rs`
(`test_integer_overflowing_i64_but_fits_u64`,
`test_integer_overflowing_u64_is_still_an_error`).

## 2. `book.md` documents `&&`/`||`, but the lexer only accepts `and`/`or`

`docs/book.md` (e.g. the `.filter(w: w.length > 3 && w.length < 8)` example) shows
`&&`/`||` as the logical operators. In reality the lexer has no `&&`/`||` token at
all — only `and`/`or` (keywords) for logical combination, plus single `&`/`|` for
bitwise. `words.filter(w: w.length > 3 && w.length < 8)` fails with
`error: unexpected token in expression: Ampersand`. Every real test under
`tests/cases/*.br` already uses `and`/`or`, consistent with the lexer, not the doc —
this looks like a **documentation bug**, not a compiler bug.

**Suggested fix:** correct `book.md`'s examples to use `and`/`or` (or add real
`&&`/`||` lexer support if that's actually the intended long-term syntax — worth a
deliberate decision either way rather than leaving the doc and the lexer disagreeing).

**Fixed:** confirmed as a pure documentation bug — no lexer change made, per the
"worth a deliberate decision" note above resolved in favor of keeping `and`/`or`
as the only logical operators (consistent with every real test under
`tests/cases/*.br` and `boring/interpreter/exec.br`). `docs/book.md`'s filter
example fixed to use `and` (commit `ae3bcee`, pre-existing before this pass —
verified by grepping the rest of `book.md` for remaining `&&`/`||`: every
other hit is inside a Rust code block or a shell command, not Boring syntax).
`docs/book.html` had gone stale relative to that fix (still showed the old `&&`
example) — regenerated via `python3 docs/build.py`.

## 3. Built-in `Error` enum not resolved by the interpreter

`book.md` (Error Handling, "Standard error enum") documents a built-in `Error` type
"always available without any import". Under `boring run`:

```boring
throw Error.InvalidInput
```

fails with `error: undefined variable 'Error' — did you mean 'error'?`. Not a single
file under `tests/cases/*.br` actually relies on the built-in `Error` type either —
they all declare their own error enums — which is consistent with this being a real
gap rather than a documented-but-untested feature that happens to work.

**Workaround used:** declared a small local error enum (`enum BigUintError:
Underflow, InvalidInput`) instead of using the built-in one.

**Suggested fix:** confirm whether `boring build`'s transpiled Rust path resolves
`Error` correctly (if so, this is interpreter-only — the tree-walk evaluator's
builtin-name resolution is presumably missing an entry the transpiler's codegen
already has). If it's broken on both backends, `Error` needs to be registered as a
real builtin in whichever central place builtin names/types are seeded.

**Fixed — interpreter-only, as suspected.** `boring build` already resolved
`Error` correctly (`src/transpiler/mod.rs`'s `pre_scan`, ~line 2836, pre-populates
`typed_error_enums`/`enum_variants` for it) — confirmed by transpiling+compiling a
`throw Error.InvalidInput` repro, which emits a real `enum Error { .. }` and
correct `TypeId`-based catch dispatch. The interpreter had no equivalent at all.
Added `Interpreter::builtin_error_enum_decl()` (`src/interpreter/mod.rs`),
registered via `exec_item` at the very start of `exec_program` (before any user
item runs), mirroring the transpiler's variant list exactly (`Expired`,
`Cancelled`, `NotFound`, `InvalidInput`, `OutOfBounds`). Both `catch
Error.Variant:` (Style 1) and typed `catch Error:` + `match error:` (Style 2)
now work under `boring run`. Regression test:
`tests/cases/builtin_error_enum.{br,expected}` (registered in both
`tests/run.rs` and `tests/transpile.rs`).

## 4. `type def` (static/factory method) rejects a typed `throws Type` clause

Instance methods (`req`/`def`) accept a typed throws clause fine:

```boring
req BigUint sub(BigUint other) throws BigUintError:
    ...
```

But the equivalent on a type-level factory method fails to parse:

```boring
type def BigUint from_decimal_string(string s) throws BigUintError:
    ...
```

→ `error: expected Colon, got Ident("BigUintError")`. Only the untyped form
(`throws:`) is accepted on `type def`.

**Workaround used:** declared `from_decimal_string` with untyped `throws:` and still
threw concrete `BigUintError` variants from inside it — this compiles and the thrown
values remain catchable/matchable normally, so it's a narrowing-precision loss on
that one declaration, not a functional blocker.

**Suggested fix:** the parser rule for `type def ... throws <Type>:` presumably needs
the same grammar production already used for `req`/`def ... throws <Type>:` — likely
a parser-only fix (whichever function parses a `type def` header probably just never
calls the typed-throws-clause parser that `req`/`def` do).

**Fixed — parser only, as suspected**, with one caveat below. `TypeMethod`
(`src/ast/mod.rs`) had no `throws_ty: Option<Type>` field at all (unlike
`FnDecl`, which already has one), and `parse_type_member`'s `type def`/`type
req`/`type set` arms (`src/parser/parse_fn.rs`) only ever ate a bare `throws`
token. Added the field, and a shared `parse_type_method_throws_task` helper
(same file) that calls the same `parse_throws_type()` routine `FnDecl` already
uses, preserving the existing order-independent `throws Type task:` /
`task throws Type:` handling. `type def BigUint from_decimal_string(string s)
throws BigUintError:` now parses. Regression tests: two parser unit tests
(`test_type_def_typed_throws`, `test_type_req_typed_throws_task_either_order`
in `src/parser/tests.rs`) plus `tests/cases/type_def_typed_throws.{br,expected}`
— registered in `tests/run.rs` (`boring run`) only at the time of this fix,
**not** `tests/transpile.rs`, because of the separate pre-existing transpiler
gap noted at the top of this file (item (c)): `emit_type_method` didn't wrap a
type-level method's `throws` in `Result<T, E>` at all yet (typed or untyped),
so `boring build` failed independent of this parsing fix. **Update: that gap
is now also fixed** (commit `81c8216`, same day) — `type_def_typed_throws` is
registered in both `tests/run.rs` and `tests/transpile.rs` as of that commit,
confirmed still true by direct inspection while writing this update.

## New issues found 2026-08-20 while adding `ScratchNumber` to the same spike

Four more issues surfaced while extending `biguint_spike.br` with a pure-Boring
`ScratchNumber` (the exact-arithmetic numeric tower for `scratch-boring`, a
drop-in replacement candidate for its hand-written `value_rt.rs`). All four are
routed around in the spike, which passes 113/113 assertions (the original 71
plus 42 new ones covering `ScratchNumber`). Only `boring run` was exercised,
same caveat as above.

## 5. `type def` is not parseable inside an `enum` body at all

Every `enum ... req/def` example in book.md section 9 shows only instance
methods. Turns out that's not incidental — a **type-level** factory method
(`type def`, as opposed to instance `req`/`def`) fails to parse inside an enum
body, even for the simplest possible case:

```boring
enum Foo:
    A(int)

    type def Foo make():
        Foo.A(1)
```

→ `error: expected newline, got Def`, pointing at the `type def` line. Fails
identically whether or not a blank line separates it from the variant list, and
regardless of what precedes it in the file. The same `type def` works fine
inside a `struct` body (used throughout this file for `BigUint.from_u64`,
`BigInt.from_i64`, etc.) — this is enum-specific.

**Workaround used:** gave up on `ScratchNumber` being the enum directly
(option (a) from the task spec, "the most natural approach in Boring") and
instead wrapped the enum (`ScratchNumberKind`, holding the actual `Int`/
`BigIntV`/`Frac`/`BigFracV`/`FloatV` variants) inside a `struct ScratchNumber`
with a single field (`kind`). Instance methods (`is_zero`, `add_owned`, etc.)
still dispatch with `match self.kind:` exactly like the enum-direct design
would have; the struct wrapper is only there so `from_i64`/`from_f64`/
`from_str` can be real `type def` factories reachable as
`ScratchNumber.from_i64(...)`.

**Suggested fix:** whatever parses an enum body's member list (presumably
`parse_enum` or similar in `src/parser/parse_fn.rs`/`parse_type.rs`) needs to
route `type def`/`type req`/`type set` through the same production `struct`
bodies already use, instead of only recognizing instance `req`/`def`.

**Fixed — parser + interpreter, `boring run` only.** `EnumDecl` had no
`type_methods` field at all (`src/ast/mod.rs`, unlike `StructDecl`), and
`parse_enum_decl_with_attrs`'s body loop (`src/parser/mod.rs`, the enum
equivalent of `parse_struct_body`) never had a `TokenKind::Type` arm, so
`type` was falling into the catch-all `parse_enum_variant()` case — matching
the observed error exactly (`type` got parsed as a bare variant name, then
`def` failed as an unexpected token where a newline was expected). Added the
field and a `TokenKind::Type` arm (plus the `pub type ...` combo) that calls
the same `parse_type_member` production `struct` bodies already use,
exactly as suggested above. `type var`/`type let` (an *associated type-level
variable*, as opposed to a factory/static method) is intentionally still
rejected inside an enum body with a clear parse error — that's a separate,
larger feature (no `EnumNamespace`/interpreter storage for it exists) out of
scope for this fix; only `type def`/`type req`/`type set` are now accepted,
matching the bug report's repro and suggested fix.

Wired through the interpreter for `boring run`: `Value::EnumNamespace`
(`src/interpreter/mod.rs`) gained a `type_methods: Vec<TypeMethod>` field
(populated from `EnumDecl::type_methods` in the `Item::Enum` exec handler,
and threaded through the `ext`-on-enum merge path), and
`eval_expr_method_call`'s existing `Value::Struct` type-method dispatch
(`src/interpreter/eval_expr.rs`) got an analogous `Value::EnumNamespace` arm
— both call the same already-generic `call_type_method` helper
(`src/interpreter/call.rs`), unchanged. `checker::check_enum`
(`src/checker/mod.rs`) now type-checks `type_methods` bodies too, mirroring
`check_struct`'s identical loop. **Not** wired into the transpiler
(`boring build`) — none of the locked-for-another-session transpiler files
were touched, and enum type-level methods were out of this fix's stated
scope (`boring run` only, per this doc's own caveat above). Regression test:
`tests/cases/enum_type_def.{br,expected}`, registered in `tests/run.rs`
only.

**Was open (transpiler) — now Fixed 2026-08-20.** `boring build --emit-rust`
on the exact repro above parsed fine (item 5's parser fix applies to both
backends) but silently dropped the enum's `type_methods` entirely from
codegen — the emitted Rust had `enum Foo { A(isize) }` with no `impl Foo {
fn make() ... }` block at all, and the call site (`Foo::make()`) was left
in place regardless. `boring build` itself exited 0 with no error or
warning; the failure only surfaced one step later, at `cargo build` on the
generated project (`error[E0599]: no variant or associated item named
'make' found for enum 'Foo'`) — worse than a clean compile-time rejection,
since `boring build` reported success for a project that could not actually
build.

**Fixed — transpiler.** `emit_enum` (`src/transpiler/emit_struct.rs`) built
its `impl EnumName { ... }` block only when the enum had plain methods,
conversions, setters, or named-field getters — `e.type_methods` was never
consulted, so an enum with *only* `type def`/`type req`/`type set` members
(no other content) skipped the whole `impl` block. Added `e.type_methods` to
that block's opt-in condition (`emit_struct.rs`, the `if !plain_methods...`
check just before `impl{} {}{} {{` is emitted) and a loop that calls
`self.emit_type_method(tm, &e.name)` for each one — the exact same method
already used for a struct's `type_methods`, now generic over a plain
`type_name: &str` instead of a struct-only `struct_name` (renamed
accordingly). `pre_scan`'s per-item registration (`src/transpiler/mod.rs`,
the `Item::Enum` arm) now also populates `struct_type_method_sigs` /
`struct_method_throws` / `typed_error_enums` for an enum's type methods,
mirroring `pre_scan_struct_item` — this is what makes a typed `throws Type:`
clause on an enum's type method get wrapped in `Result<T, E>` by
`emit_type_method`, same as the already-fixed struct case (item 4). Two
small helper loops (`collect_default_rest_targets` and
`collect_thrown_enum_names`, both in `src/transpiler/helpers.rs`) were also
extended to walk `EnumDecl::type_methods` bodies, mirroring their existing
`EnumDecl::methods`/`setters` handling.

A second, previously-masked gap surfaced once the `impl` block above was
emitted: plain (non-error) enums had no `Display` impl at all, so `print
Foo.make()`/`print z` (`Foo` being an ordinary non-error enum) failed to
compile the generated Rust with `error[E0277]: 'Foo' doesn't implement
'std::fmt::Display'` — no existing test before this fix ever printed a
plain enum value directly (only via pattern-matched fields, a getter, or an
explicit `as string:` conversion), so this had never been exercised. Fixed
by generalizing `emit_enum`'s existing typed-error-enum-only Display block
(previously gated on `is_error_type`) into an unconditional auto-Display for
every enum that doesn't already have one — same rationale and same
`write!(f, "{:?}", self)` delegate-to-Debug body as the struct auto-Display
just above it in the same file, including matching generic-type-param
bounds (`<T: Clone + std::fmt::Debug>`) for enums like a user-defined
`Result<T, E>`. Skipped when: a `derive`-macro trait will generate Display
itself (`@error` variants → `thiserror::Error`), the enum (or a same-named
`ext` block) already declares its own `as string:` conversion (tracked via
the existing `display_types` set, now also populated for an enum's *own*
inline `as string:` conversion in `pre_scan`, alongside the pre-existing
struct/ext registrations), or the enum explicitly opts out via
`@derive(Display)`. `impl std::error::Error for EnumName {}` is still
emitted only for typed error enums, unchanged.

Regression tests: `tests/cases/enum_type_def.{br,expected}` (untyped
`type def`/`type req` on a plain enum, the exact repro above plus a second
type method and two `print` calls on the results) and
`tests/cases/enum_type_def_throws.{br,expected}` (a typed `throws FooError:`
clause on an enum's `type def`/`type req`, mirroring
`tests/cases/type_def_typed_throws.br`'s struct case) — both registered in
`tests/run.rs` (`interp_test!`, `boring run`) and `tests/transpile.rs`
(`transpile_test!`, `boring build` across all four mode/threading
combinations). Manually verified end-to-end beyond the automated suite: the
exact repro above transpiled with `boring build` (project mode, not just
`--emit-rust`), the generated Cargo project compiled with `cargo build` and
ran with `cargo run`, printing `A(1)` — identical to `boring run`'s output
on the same source. Full `cargo test` is green (all suites, zero
regressions) and `scratch-boring` (`boring/regen.sh` + `cargo build` +
`cargo test`) remains green against the fixed compiler.

## 6. `if let x = expr: A else: B` used as a nested (non-tail) expression silently always evaluates to `B`

This is the most dangerous issue found — it does not fail to compile, it
**runs and silently produces the wrong value**. Two distinct manifestations,
depending on where the two-branch `if let ... else ...` sits:

**(a) As the direct top-level tail expression of a function** — this is
caught, but as an unrelated-looking hard error:

```boring
Foo make(bool ok):
    if let i = maybe(ok):
        Foo(v = i)
    else:
        Foo(v = 2)
```

→ `error: return value discarded — bind it with 'let', discard with '_ = f()'`,
pointing at the `else` branch's expression — as if the checker had decided the
whole `if let` wasn't the function's return value and its final expression was
therefore a discarded statement.

**(b) Nested one level deeper (inside another `if`, inside a `match` arm,
etc.)** — no error at all, but the `if let`'s "then" branch is never taken,
even when the option is genuinely `Some`:

```boring
int? maybe(bool ok):
    if ok: 1 else: nil

int? test(bool flag):
    if flag:
        if let i = maybe(true):   # condition is true, i.e. Some(1)
            i
        else:
            nil
    else:
        nil

print test(true)   # prints "nil" -- should print "1"
```

Confirmed with a match-arm body in the same shape (`match k: A(v): if let i =
maybe(...): i else: nil`) — same silent wrong-answer behavior.

**Workaround used:** never write a two-branch `if let X = expr: A else: B` as
a value-producing expression outside the literal, unnested top level of a
function. Instead use `guard let x = expr else return <else-value>` followed
by the plain `then`-expression as the next statement — confirmed safe even
several levels of nesting deep (inside `match` arms, inside other `if`
blocks). Every `if`-let in `ScratchNumber`'s implementation follows one of two
verified-safe shapes: either `guard let ... else return ...` (this fix), or a
plain `if let x = expr: return A` with **no** attached `else` clause, followed
by a trailing fallback expression/statement (this shape was independently
verified safe at arbitrary nesting depth too, and is used throughout the
`_tier` dispatch helpers below).

**Suggested fix:** whatever lowers `if let`-as-expression (presumably in
`src/interpreter/eval_expr.rs` or the checker's expression-type inference) is
only correctly wired for the case where the `if let` is literally the last
statement of a function body evaluated in tail position at the *outermost*
level — nested occurrences fall back to some default (`else`-branch,
seemingly) instead of actually evaluating the condition. Needs a real fix, not
a doc note — this is a silent-miscompilation-class bug, worse than a compile
error.

**Fixed — interpreter, `src/interpreter/eval_expr.rs` + `call.rs`.** The root
cause was exactly as suspected: `if let` had no expression-producing
evaluation path at all, only a statement-executing one (`exec_if_let`,
`src/interpreter/exec.rs`, returns `()`). Two separate call sites needed it
and were both missing a `Stmt::IfLet` arm:

- `eval_block_as_expr`'s "is this the block's last statement?" special-case
  (~line 1725) handled `Stmt::Expr`/`Stmt::If`/`Stmt::Match` but fell through
  to plain `exec_stmt` for `Stmt::IfLet` — so a block whose last statement was
  an `if let` always kept its already-initialized `last = Value::Nil`
  regardless of which branch actually ran. This is manifestation (b): nested
  one level deep (inside another `if`'s body, a `match` arm, etc.), the
  surrounding block silently discarded the real result.
- `call_fn`'s `last_produces_value` classifier (`src/interpreter/call.rs`,
  ~line 185) listed `Stmt::If`/`Stmt::Match` as value-producing tail
  statements but not `Stmt::IfLet`, so as the *literal* tail statement of a
  function body it took the "non-value-producing" path and ran via plain
  `exec_stmt` → `exec_if_let` → `exec_block` on the chosen branch, whose own
  last statement (a bare struct-constructor call) then tripped the unrelated
  must-use "return value discarded" check. This is manifestation (a).

Added `Interpreter::eval_if_let_expr` (`eval_expr.rs`) — mirrors
`exec_if_let`'s clause/elif/else logic exactly, but evaluates the chosen
branch with `eval_block_as_expr` instead of `exec_block` so the branch's own
tail expression becomes the `if let`'s value (same pattern `eval_if_expr`
already used for plain `if`/`else`). Wired it into both `eval_tail_stmt` and
`eval_block_as_expr`'s last-statement match, and added `Stmt::IfLet(_)` to
`call_fn`'s `last_produces_value` match. Both manifestations now work:
`if flag: if let i = maybe(true): i else: nil else: nil` correctly prints
`1`, and `if let i = maybe(ok): Foo(v = i) else: Foo(v = 2)` as a function's
direct tail statement now compiles and returns the right `Foo` instead of
erroring. Regression test: `tests/cases/if_let_expr_nested.{br,expected}`
(covers both manifestations, plus the match-arm-nested variant also
mentioned above), registered in `tests/run.rs`.

## 7. Unary negation (`-x`) on an `int64`/`int128`-tagged value fails ("cannot negate Int64/Int128")

```boring
def show(int64 v):
    let n = -v      # error: cannot negate Int64
    print n

show(-42)
```

Reproduces for both `int64` and `int128` in ordinary `def`/`req` functions and
methods. Oddly, it does *not* reproduce when the value reaching the negation
is a bare integer literal passed straight into a typed parameter at the call
site (e.g. `Foo.from_i64(-42)` computing `-v` inside a `type def` factory
works fine) — apparently such values stay tagged as the interpreter's generic,
untyped `Value::Int` under the hood despite the `int64` parameter annotation,
and only a value that has gone through an explicit `as int64`/`as int128` cast
(or arithmetic that produces one) ends up genuinely tagged and hits the bug.
This inconsistency makes the bug easy to miss in a small handwritten example
and easy to hit once real cast-based arithmetic is involved.

**Workaround used:** never apply unary `-` to an `int64`/`int128` value.
Compute negation via subtraction from zero instead: `(0 as int64) - x` / `(0
as int128) - x` — verified to work reliably regardless of context or nesting.
Used throughout `ScratchNumber`'s i64/i128-tier arithmetic (sign
normalization in `make_fraction_i64`, etc.)

**Suggested fix:** the interpreter's unary-negate evaluation (presumably in
`src/interpreter/eval_expr.rs`) only has a case for the generic `Value::Int`,
missing `Value::Int64`/`Value::Int128` (and, untested here, likely every other
explicitly-tagged sized integer type — `int8`/`int16`/`int32`/the unsigned
family).

**Fixed — exactly as suspected, `src/interpreter/eval_expr.rs`.** The
`UnaryOp::Neg` match (~line 1230) only had arms for `Value::Int`,
`Value::Float64`, and `Value::Float32` — every tagged *signed* sized-integer
variant (`Value::Int8`/`Int16`/`Int32`/`Int64`/`Int128`) fell through to the
catch-all `other => Err(...)` arm. Added a case for each, mirroring the
generic `Value::Int` arm exactly (`Value::IntN(n) => Ok(Value::IntN(-n))`).
The unsigned family (`Value::Uint8`..`Value::Uint128`) deliberately got no
new arm — negating an unsigned value stays a legitimate error, unchanged.
Regression test: `tests/cases/tagged_int_negate.{br,expected}`, registered in
`tests/run.rs`.

## 8. Interpreter's move-checker treats `int64`/`int128`-tagged scalars as non-Copy

book.md documents all primitive numeric types as Copy (assigning them always
copies, both bindings stay usable). That holds for the untagged generic `int`
(`isize`), but not for a value explicitly tagged `int64`/`int128`:

```boring
var m = 48 as int64
var n = 18 as int64
while n != (0 as int64):
    let t = n           # "moves" n
    n = m % n           # error: use of moved value 'n'
    m = t
```

→ `error: use of moved value 'n': the value was moved and is no longer
accessible — use .clone() to make a copy`. The identical loop with plain `int`
(no `as int64` cast anywhere) has no such error. Since `int64`/`int128` are
Copy types in the emitted Rust (`i64`/`i128`), this looks like the
interpreter's static move-checker not special-casing every sized-integer
`Value` variant as Copy the way it already does for the generic `Value::Int`.

**Workaround used:** call `.clone()` on the tagged scalar before a
move-sensitive reuse, exactly as if it were a non-Copy struct — e.g. `let t =
n.clone()` in `gcd_i64`'s swap loop (`ScratchNumber`'s i64-tier gcd helper).

**Suggested fix:** the move-checker's Copy-type allowlist (wherever it decides
a `let`/reassignment "moves" vs. "copies" a value) needs every sized integer/
float `Value` variant (`Int8`..`Int128`, `Uint8`..`Uint128`, `Float32`,
`Float64`) added alongside the generic `Value::Int`/`Value::Uint`/`Value::Float`
it apparently already exempts.

**Fixed — exactly as suspected, `src/interpreter/exec.rs`.** Found the
single allowlist function, `Interpreter::is_copy_value` (~line 1753,
used by both the `let`-statement move-check and the general assignment
move-check in `exec.rs`/`mod.rs`) — it only exempted `Value::Int`,
`Value::Uint`, `Value::Float64`, `Value::Bool`, `Value::Str`, `Value::Nil`,
and `Value::Void`. Added every remaining sized/tagged numeric variant
(`Int8`..`Int128`, `Uint8`..`Uint128`, `Float32`) as suggested — a single,
shared function, so both call sites picked up the fix with no further
changes needed. Regression test: `tests/cases/tagged_int_copy.{br,expected}`
(the exact gcd-swap-loop repro from this section, for both `int64` and
`int128`), registered in `tests/run.rs`.

## New issues found 2026-08-20 while wiring `ScratchNumber` into real `scratch.br` (transpiler-only, `boring build --emit-rust`)

Items 1-8 above were all found and verified via `boring run` (the tree-walk
interpreter) only. Wiring the same `ScratchNumber`/`BigUint`/`BigInt`/
`BigFraction` code from `boring/spikes/biguint_spike.br` directly into
`scratch-boring/boring/scratch.br` and building it for real via `boring build
--emit-rust` (the only backend `scratch-boring` actually uses — see its
`boring/regen.sh`) surfaced four new bugs, none hit by the interpreter-only
spike — confirming the suspicion already flagged above ("some [bugs] are
interpreter-only gaps that already work when transpiled... worth confirming
before fixing" — here it's the reverse: transpiler-only gaps invisible to the
interpreter). All four are worked around directly in `scratch-boring`'s
`scratch.br` (not in `boring/spikes/biguint_spike.br`). See `scratch.br`'s own
inline comments at each workaround site for the precise before/after — the
workarounds are left in place (same non-regression-fixture rationale as
`biguint_spike.br`'s own workarounds for items 1-8) even though all four
underlying bugs are now fixed in the compiler too; see each item's own
"**Fixed:**" note below.

## 9. Implicit-`self` array field's `.length` mis-transpiles to invalid `::` path syntax outside tail position

```boring
struct Foo:
    var mut [uint32] limbs

    def normalize():
        while limbs.length > 1 and limbs[limbs.length - 1] == 0:
            limbs.pop()
        if limbs.length == 0:
            limbs.push(0)
```

`boring build --emit-rust` on the above produces:

```rust
fn normalize(&mut self) -> () {
    while ((self.limbs::length > 1) && (self.limbs[((self.limbs::length - 1)) as usize].clone() == 0)) {
        self.limbs.pop().unwrap_or_default();
    }
    if (self.limbs::length == 0) {
        self.limbs.push(0);
    }
}
```

`self.limbs::length` is not valid Rust (`::` is a path separator, not a
field/property accessor) — it should be `self.limbs.len() as isize`, exactly
like every *other* `.length` access in this same file transpiles correctly
(e.g. `other.limbs.length` → `other.limbs.len() as isize`, or a `.length` on
any plain local-variable receiver, e.g. `let items = table[id] else []` then
`items.length`). Confirmed by direct experimentation that the bug is specific
to `.length` used on a **bare, implicit-`self`** field reference — i.e. inside
a method body, plain `limbs.length` (no explicit `self.` prefix; Boring
resolves `limbs` to the struct's own field automatically) — and **only when
that expression appears anywhere other than the method's own final tail
expression**. A `.length` access that *is* the whole method body's return
value transpiles fine even with implicit `self`:

```boring
req int just_length():
    limbs.length          # tail position -> transpiles to `self.limbs.len() as isize`, fine

req bool second_method():
    limbs.length > 1      # non-tail (used in a comparison) -> `self.limbs::length > 1`, broken
```

Explicitly writing `self.limbs.length` instead of bare `limbs.length` does
**not** help — it produces the identical broken `self.limbs::length` output
whenever used outside tail position. Neither does assigning it to a `var`
(`var idx = limbs.length` breaks too, since the RHS is the same expression
shape). What *does* work: aliasing the field into a fresh local variable
*first*, then reading `.length` off that local:

```boring
let ls = limbs
ls.length > 1              # -> `let ls = self.limbs.clone(); (ls.len() as isize) > 1` — fine
```

...but that requires re-aliasing after every mutation inside a loop (e.g.
`normalize`'s `while ... limbs.pop() ...`), which is awkward.

**Workaround used (in `scratch-boring/boring/scratch.br`):** route every
non-tail `limbs.length` through a tiny free function taking the array as an
**explicit parameter** (not `self`):

```boring
int len32([uint32] v):
    v.length
```

Every problem call site becomes `len32(limbs)` instead of `limbs.length`.
This transpiles to `len32(&self.limbs)` — a fresh borrow evaluated at each
call site, always up to date even inside a mutating loop — and sidesteps the
bug entirely, since the bug is specifically about `.length` chained directly
off an implicit-`self` receiver, not about function arguments.

**Impacted code:** every `BigUint` method that reads `limbs.length` outside
tail position (`normalize`, `is_zero`, `compare`, `add`, `to_f64`,
`mul_small`, `mul`, `divmod_small`, `to_u64_checked`) — 18 call sites total in
`scratch-boring/boring/scratch.br`.

**Suggested fix:** whatever emits property access for a bare (implicit-self)
identifier that resolves to a struct field (presumably in
`src/transpiler/emit_expr.rs` or `helpers.rs`) has a special-cased path for
`.length` that only correctly qualifies it as `self.<field>.len() as isize`
when the access sits in tail position; the general/non-tail codegen path for
the same identifier resolution seems to fall through to a different, buggy
"property access" emission (`::`) instead of reusing the tail-position logic.

**Fixed — exactly as suspected, `src/transpiler/emit_expr.rs`.** The root
cause was in `emit_expr_field`'s `is_path_receiver` heuristic (~line 1887,
just above the generic `obj.field` emission that both tail and non-tail
positions eventually reach): a bare lowercase identifier not present in
`known_local_vars` was assumed to be an imported-module-style path receiver
(`mpsc.foo` → `mpsc::foo`), so `format!("{}::{}", obj_s, field)` fired. An
implicit-self field reference (`limbs`, resolved to `self.limbs` a few lines
earlier by the *separate* `ExprKind::Var` implicit-self logic at the top of
`emit_expr`) is *also* absent from `known_local_vars` — it's a field, not a
local var — so it tripped the same "unknown lowercase name → path" branch,
turning the already-correct `obj_s = "self.limbs"` into `self.limbs::length`.
The tail-position case never hit this at all: `emit_stmt.rs`'s tail-expression
handling routes non-Optional-return tails through `emit_expr_owned` (`src/
transpiler/emit_top.rs`), which has its own independent `ExprKind::Field` arm
with no `is_path_receiver`-style check — that's *why* tail position was
already correct, not because of any shared fix. Added an `is_implicit_self_field`
guard (mirroring the implicit-self detection in `emit_expr`'s own `Var` arm:
`self.self_type` is set, the name isn't in `known_local_vars`, and it matches
one of the current struct's real field names) that short-circuits
`is_path_receiver` to `false` before the lowercase/`known_local_vars` check
can misfire. Regression test: `tests/cases/implicit_self_length_nontail.{br,expected}`
(registered in `tests/transpile.rs` only — confirmed via `boring run` that the
interpreter never had this bug, consistent with the framing above).

## 10. Nested call inside a `throws` function gets a spurious `?` based on the callee's *name*, not its actual (type-resolved) throwing-ness

```boring
struct Inner:
    int v
    req Inner mul(Inner other):              # non-throwing
        Inner(v = v * other.v)

struct Outer2:
    int v
    req Outer2 mul(Outer2 other) throws MyErr:  # unrelated type, also named `mul`, throws
        guard true else throw MyErr.Bad
        Outer2(v = v * other.v)

struct Outer:
    Inner numerator
    Inner denominator
    req Outer combo(Outer other) throws MyErr:
        guard true else throw MyErr.Bad
        let n = self.numerator.mul(other.denominator)   # Inner.mul -- does NOT throw
        Outer(numerator = n, denominator = self.denominator)
```

Generated Rust:

```rust
fn combo(&self, other: Outer) -> Result<Outer, Box<dyn std::error::Error + Send + Sync>> {
    if !(true) { return Err(...); }
    let n = self.numerator.mul(other.denominator)?;   // <- invalid: Inner::mul returns Outer, not Result
    Ok(Outer { numerator: n, denominator: self.denominator })
}
```

`self.numerator.mul(...)` calls `Inner::mul`, which returns a plain `Inner`
(no `Result`) — yet the transpiler appends `?`, which doesn't compile (`the
"?" operator can only be applied to values that implement "Try"`). Confirmed
by experimentation that the trigger is **purely name-based**: as long as
*some* method named `mul` *anywhere in the whole program* is
`throws`-declared, *every* call to `.mul(...)` inside any `throws`-declared
function gets `?` appended, regardless of the actual receiver's type and
regardless of whether the enclosing function's own name matches. Renaming the
enclosing method (`combo` above, vs. a same-named `mul`) makes no difference;
only removing the colliding throws-declared name anywhere in the program
fixes it.

This exact shape occurs for real in `scratch-boring/boring/scratch.br`:
`BigFraction.add`/`sub`/`mul`/`div` are `throws BigFractionError`, while
`BigInt`/`BigUint`'s own `add`/`mul` (and `BigInt`'s `sub`) are plain
non-throwing methods of the same name — every `self.numerator.mul(...)`/
`.add(...)`/`.sub(...)` call inside `BigFraction`'s own bodies (operating on
its `BigInt` fields) was getting a spurious, invalid `?`. `BigUint.sub`
(throws `BigUintError.Underflow`) additionally collided with `BigInt.sub`
(non-throwing) the same way.

**Workaround used (in `scratch-boring/boring/scratch.br`):** renamed every
throwing method to a name that doesn't collide with any non-throwing method
of the "natural" name elsewhere in the file: `BigUint.sub` → `sub_checked`;
`BigFraction.add`/`sub`/`mul`/`div` → `add_frac`/`sub_frac`/`mul_frac`/
`div_frac`.

**Suggested fix:** whatever decides to append `?` after a method call inside
a `throws` function (presumably in `src/transpiler/emit_expr.rs`) needs to
resolve the callee's *actual receiver type* first and check whether *that
type's* method of the given name throws, instead of checking a single global
name→throws map keyed only by method name.

**Fixed — `src/transpiler/emit_methods.rs`, ~line 2063 (the general
method-call fallback, `emit_method_call_fallback`'s throws-propagation tail).**
The receiver-type resolution guarding the qualified `"StructName::method"`
lookup already existed (per the comment right above it) but was too narrow: it
only matched `ExprKind::Var(v)` (`self` or a plain local), falling straight
through to `None` — and from there to the unqualified, bare-name
`struct_method_throws.contains(method)` fallback — for any other receiver
shape, including exactly the field-of-field chain in the bug report
(`self.numerator.mul(...)`, receiver `self.numerator` is `ExprKind::Field`,
not `ExprKind::Var`). Replaced the inline `match &obj.kind { ExprKind::Var
... }` with a call to the already-existing `resolve_expr_struct_type` helper
(`src/transpiler/emit_methods.rs` ~line 50), which recursively resolves
`self`, a plain local, *and* a field-of-field/index chain uniformly — it was
already used elsewhere in this file and in `emit_expr.rs` for the identical
"don't misidentify a builtin vs. a user field/type" class of problem, just
never plugged into this particular throws check. The old `var_struct_type`
(singular) lookup is kept as a secondary fallback via `.or_else(...)` so no
existing resolution path is lost. Regression test:
`tests/cases/throws_method_name_collision.{br,expected}` (registered in
`tests/transpile.rs` only — the interpreter has no notion of `?`-insertion at
all, so this class of bug is transpiler-only by construction).

## 11. Narrowing numeric cast (`int128 -> int64` observed) inside `if let`/`guard let` produces a non-`Option` raw cast, failing to compile

book.md documents narrowing casts as checked: *"Casting to a narrower type
checks the range and produces `nil` ... if the value doesn't fit."* And
indeed, `if let n = (v as int64):` on an `int128` correctly transpiles its
**pattern** to `if let Some(n) = ...` — but the right-hand side stays a bare,
infallible `(v as i64)`, which is not `Option<i64>`:

```boring
def iflet_cast(int128 v):
    if let n = (v as int64):
        print "{n}"
    else:
        print "nope"
```

```rust
fn iflet_cast(v: i128) {
    if let Some(n) = (v as i64) {   // (v as i64) has type i64, not Option<i64> -- E0308
        println!("{}", n);
    } else {
        println!("nope");
    }
}
```

This reproduces for `int64 -> int32` too, so it isn't `int128`-specific
narrowly, but `int128 -> int64` is the width pair `scratch.br`'s own
arithmetic needs throughout (the overflow-checked-via-128-bit-intermediate
technique used by its `add_i64_tier`/`sub_i64_tier`/`mul_i64_tier`/
`div_i64_tier`/`rem_i64_tier`/`compare_i64_tier` helpers). A **bare** narrowing
cast used *outside* `if let`/`guard let` (e.g. a plain tail expression, not
pattern-matched) transpiles fine as an ordinary infallible truncating `as`
cast — the bug is specific to the combination of a narrowing numeric `as`
cast used as the scrutinee of `if let`/`guard let`.

**Workaround used (in `scratch-boring/boring/scratch.br`):** a hand-written
checked-narrowing helper that does the range check itself with plain
comparisons, then a *bare* cast (never inside `if let`, so it hits the
working codegen path):

```boring
int64? checked_i128_to_i64(int128 v):
    let int64 i64_min_64 = -9223372036854775807 - 1
    let i64_min = i64_min_64 as int128
    let i64_max = 9223372036854775807 as int128
    if v >= i64_min and v <= i64_max:
        v as int64
    else:
        nil
```

(The `let int64 i64_min_64 = ...` explicit-type annotation on the literal is
load-bearing too — without it, the untyped integer literal defaults to `i32`
in the generated Rust and `-9223372036854775807` doesn't fit, producing
`error: literal out of range for 'i32'`. This is the same family as known
issue #1 above, just triggered a different way: there the literal was
range-checked before its cast; here an *unconstrained* literal — used only as
the input to a later `as int128` cast, which doesn't itself pin down a source
width — needs an explicit type annotation to avoid rustc's `i32` default for
bare integer literals.)

Every `guard let n = (n128 as int64) else return nil` / `if let n = (n128 as
int64):` call site across `scratch.br`'s i64-tier arithmetic (22 occurrences)
is routed through `checked_i128_to_i64(...)` instead.

**Suggested fix:** whatever emits the checked-narrowing-cast-as-`Option`
codegen (presumably in `src/transpiler/emit_expr.rs`, the same general area
as issue #1's cast handling) needs to also apply when that cast expression is
specifically the scrutinee of `if let`/`guard let`, not only when it's a bare
tail/statement expression.

**Fixed — `src/transpiler/emit_expr.rs` + call sites in `emit_match.rs`/
`emit_flow.rs`.** There was no pre-existing "checked-narrowing-cast-as-Option"
codegen path at all to plug into — a bare numeric-to-integer cast, in or out
of `if let`, was always emitted as an unconditional infallible `(src as dst)`
by `emit_expr_cast` (this is unchanged, pre-existing behavior, not part of
this fix's scope); the *pattern* side of `if let`/`guard let`
(`emit_cond_clauses`/`emit_guard` in `emit_match.rs`/`emit_flow.rs`) always
hard-codes a `Some(name)` binding regardless of the scrutinee's shape, which
is what actually required an `Option` here and didn't get one. Added
`Transpiler::try_emit_checked_int_cast_as_option` (`emit_expr.rs` ~line 1105):
for a `Cast(inner, ty)` expression whose target is a fixed-width/pointer-width
integer type and whose source isn't itself already string/bool-typed (those
already get correct `Option`-producing codegen from the ordinary `emit_expr`
path — `.parse().ok()` / literal `None` — which this must not shadow), emits
Rust's standard `{dst}::try_from({src}).ok()` instead of a raw cast. `TryFrom`
is implemented in `std` for every pair of the twelve integer types (narrowing,
widening, or same-width), so this isn't limited to the `int128 -> int64` pair
the bug was found with. Wired into both call sites that build the `Some(...)`
pattern's right-hand side: `emit_cond_clauses`'s `CondClause::Let` arm
(`emit_match.rs` ~line 149, used by `if let`) and `emit_guard`'s
`CondClause::Let` arm (`emit_flow.rs` ~line 307, used by `guard let`) — both
try the new helper first and fall back to the ordinary `emit_expr` for every
other scrutinee shape. Regression test:
`tests/cases/narrowing_cast_if_let.{br,expected}` (registered in
`tests/transpile.rs` only — confirmed via `boring run` that the interpreter's
existing checked-cast semantics already handled this correctly).

## 12. Spelled-out `try EXPR else nil` used as an `if let` condition produces a type-mismatched `match`; `try? EXPR` (the documented-equivalent shorthand) works correctly

book.md: *"`try? expr` is shorthand for `try expr else nil`. It converts a
`throws` function call ... into an optional value: success becomes the
value, any error becomes `nil`."* The two spellings are documented as
interchangeable, but only `try?` actually transpiles as described:

```boring
def use_try_else_nil(int v):
    if let r = (try risky(v) else nil):     # risky throws MyErr
        print "{r}"

def use_try_question(int v):
    if let r = (try? risky(v)):
        print "{r}"
```

`use_try_question` transpiles cleanly to `if let Some(r) = risky(v.clone()).ok()`.
`use_try_else_nil` instead produces a `match` whose arms have incompatible
types — the `Ok` arm evaluates to the raw success value, while the `Err` arm
evaluates to `None`, and the two are never reconciled into a single
`Option<T>` the way `try?`'s dedicated `.ok()` codegen does automatically.

**Workaround used (in `scratch-boring/boring/scratch.br`):** use `try?
BigInt.from_decimal_string(s)` instead of the spike's original `try
BigInt.from_decimal_string(s) else nil`, inside `ScratchNumber.from_str`'s
`if let big = (...)` condition. Pure spelling swap per book.md's own stated
equivalence — no semantic change.

**Suggested fix:** whatever lowers the spelled-out `try EXPR else nil` form
(presumably in `src/transpiler/emit_expr.rs`) should desugar to the exact
same codegen `try?` already uses, rather than a separate, incompletely
type-unified `match` lowering.

**Fixed — one level higher than expected, in the parser rather than the
transpiler: `src/parser/parse_expr.rs`'s `parse_else_expr`, the spelled-out
`try EXPR else ...` production (~line 220).** This form *unconditionally*
folded into `ExprKind::TryElseBlock` — the general "try body + else body"
statement-block AST node, which the transpiler lowers to the `match` with the
unreconciled `Ok(__boring_v) => __boring_v` / `Err(...) => { ...; None }` arms
described above — no matter what the else expression actually was.
`try? EXPR`, a few lines above in the same function, instead builds a
dedicated `ExprKind::TryElse(expr, Nil)` node, which `emit_expr.rs` already
lowers correctly to `{expr}.ok()` (that codegen was already correct and
untouched by this fix). Added a check for exactly the shape `try EXPR else
nil` (the else-clause parses as a bare, non-block `ExprKind::Nil` — i.e. the
literal repro shape, not an `else:` block that merely computes to `nil` after
side effects, which still needs `TryElseBlock` for its `error` binding/
statement sequencing and is unaffected by this fix): when detected, build the
same `TryElse(expr, Nil)` node `try?` already builds, instead of
`TryElseBlock`. This turned out to fix a strictly larger class of cases than
the bug report's literal `if let` repro — a bare, non-`if-let` `let x = try
risky(v) else nil` was *also* broken the same way before this fix (confirmed
by direct experimentation) and is fixed by the same change, since both are
just `parse_else_expr` call sites hitting the same production. Regression
test: `tests/cases/try_else_nil_if_let.{br,expected}` (registered in
`tests/transpile.rs` only, covers both the `if let` shape from the bug report
and the plain-`let` shape found while verifying it — confirmed via `boring
run` that the interpreter's dynamic evaluation was never affected by this
class of bug, consistent with the framing above).

## Also encountered (not a new bug, a recurrence of an already-known pattern)

`ScratchNumber.from_str_owned`'s body in the original spike was a bare tail
call, `ScratchNumber.from_str(s)` — itself already `ScratchNumber?`-returning.
This hits the same "a `T?`-returning function auto-wraps a *plain* tail value
in `Some(...)` but doesn't know to skip that wrap when the value is already an
`Option`" double-wrap pattern that `scratch.br`'s own pre-existing sb3-loader
code already called out and worked around. Fixed the same way: an explicit
`if let v = ScratchNumber.from_str(s): v else: nil` instead of the bare tail
call.

## Reference

Full spike: `scratch-boring/boring/spikes/biguint_spike.br` (all workarounds are
commented in place at their use site, left in place even now that all 8
interpreter-side items (1-8) are fixed — the spike is a non-regression
fixture, not a place to un-work-around things). All 8 items above were routed
around, not silently ignored — the spike's 113/113 passing assertions (71
original + 42 for `ScratchNumber`) never depended on any of them being fixed,
and still pass unchanged now that they are (re-verified `cargo build
--release` + `boring run`, 2026-08-20). Items 1-4 were found 2026-08-19 while
writing the original `BigUint`/`BigInt`/`BigFraction` pass; items 5-8 were
found 2026-08-20 while adding `ScratchNumber` on top of that same foundation.
All 8 are fixed — see each item's own "**Fixed:**" note for the exact
file/line and what changed.

Items 9-12 (the transpiler-only batch, found while wiring `ScratchNumber` into
`scratch-boring/boring/scratch.br` for real via `boring build --emit-rust`)
are fixed too, same day. `scratch.br`'s own workarounds for all four (the
`len32` helper, the `sub_checked`/`add_frac`/`sub_frac`/`mul_frac`/`div_frac`
renames, the `checked_i128_to_i64` helper, and the `try?`-instead-of-`try...
else nil` spelling) were deliberately left in place, same non-regression-
fixture rationale as the spike's own workarounds — confirmed still correct
(not merely "still compiles") by regenerating `scratch-boring/src/boring_gen.rs`
from the fixed compiler (`boring/regen.sh`) and diffing it against the
already-committed version: **byte-identical**, i.e. the workarounds produce
exactly the same Rust the fixed compiler would now produce for the
non-worked-around form too. `cargo build` + `cargo test` on the regenerated
project are fully green, including `tests/fibonacci.rs` (3 tests) and
`tests/scratch_number.rs` (8 tests).
