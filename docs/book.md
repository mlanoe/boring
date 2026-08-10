# The new programming language is Boring

Boring is a high-level language that transpiles to Rust. It is designed to feel lighter than Rust while retaining full access to Rust's type system, ownership model, and performance. Every Boring program can be run directly (interpreter) or compiled with `boring build`.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Variables and Mutability](#2-variables-and-mutability)
3. [Data Types](#3-data-types)
4. [Functions](#4-functions)
5. [Comments](#5-comments)
6. [Control Flow](#6-control-flow)
7. [Collections](#7-collections)
8. [Structs](#8-structs)
9. [Enums and Pattern Matching](#9-enums-and-pattern-matching)
10. [Traits and Extensions](#10-traits-and-extensions)
11. [Error Handling](#11-error-handling)
12. [Optionals](#12-optionals)
13. [Generics](#13-generics)
14. [Closures and Higher-Order Functions](#14-closures-and-higher-order-functions)
15. [Modules](#15-modules)
16. [Ownership Qualifiers](#21-ownership-qualifiers)
17. [Defer](#22-defer)
18. [Streams](#17-streams-stream--yield)
19. [Channels](#18-channels-channel)
19. [Tasks (Async)](#23-tasks-async)
20. [Attributes](#24-attributes)
21. [Format Specifiers](#25-format-specifiers)
22. [Built-in Functions](#26-built-in-functions)
23. [Appendix: Boring → Rust Mapping](#27-appendix-boring--rust-mapping)
24. [Diagnostics](#28-diagnostics)
25. [Advanced](#29-advanced)
26. [Qualifier Inference](#30-qualifier-inference)
27. [Rust-for-Linux target](#31-rust-for-linux-target)

---

## 1. Getting Started

### Creating a project

```sh
boring new my_project   # scaffold a project directory
cd my_project
boring run              # interpret main.br
boring build                      # emit a Cargo project (→ my_project_rust/)
boring build --threading single   # emit to my_project_rust_single/ (both can coexist)
boring build --compile            # emit a Cargo project and immediately compile it
```

`boring new` creates:

```
my_project/
├── boring.toml   # project metadata
└── main.br       # entry point
```

**`boring.toml`**

```toml
[project]
name    = "my_project"
version = "0.1.0"
main    = "main.br"     # optional, defaults to main.br
```

### Running a single file

```sh
boring hello.br              # interpret directly
boring build hello.br        # emit a Cargo project next to the file
```

### Hello, World

```boring
print "Hello, World!"
```

**Rust equivalent**
```rust
fn main() {
    println!("Hello, World!");
}
```

String interpolation is built-in — no explicit `format!` needed:

```boring
let name = "World"
print "Hello, {name}!"
```

---

## 2. Variables and Mutability

Boring separates two orthogonal concepts:

- **Rebindable**: can the variable point to a different instance?
- **Mutable** (content-mutable): can the pointed instance be modified — `def` calls, field writes, structural collection mutation?

Unlike many languages, these two axes are controlled **independently**, and `mut` lives in the **type**, not just the binding keyword — `mut Type`/`mut Type&` compose the same way anywhere a type appears (a tuple slot, a struct field, an array element, a generic argument), not only at the top of a variable declaration. Four combinations exist for a local binding:

| Form | Rebindable | Mutable |
|---|---|---|
| `let Type a` | no | no |
| `mut Type a` (≡ `let mut Type a`) | no | yes |
| `var Type a` | yes | no |
| `var mut Type a` | yes | yes |

`lazy` is a fifth, orthogonal keyword: a deferred, write-once binding — no rebind, no mutation, assigned exactly once via `?=`.

**`mut` never implies `var`, and `var` never implies `mut`** — this is deliberate and has no exceptions (not even for `'actor`/`'guard`-qualified types, where the underlying `Arc<Mutex<T>>`/`Arc<RwLock<T>>` would technically permit mutation through the lock regardless — Boring's own bookkeeping still gates it on the binding, matching every other type). `mut x = 42; x = 43` is a compile error — that's the whole point of writing `mut` instead of `var` in the first place.

### Immutable bindings — `let`

```boring
struct Counter:
    var int value = 0
    def inc(): value += 1
    req int get(): value

let Counter c = Counter()
c.get()          # ok — req methods work on let
# c.inc()        # ERROR — def requires the type to carry mut
# c = Counter()  # ERROR — let cannot be rebound
```

Fixed binding, immutable instance. Neither the binding nor the instance can change.

### Fixed mutable bindings — `mut`

```boring
mut Counter c = Counter()
c.inc()          # ok — def methods work: c's type is mut Counter
c.inc()
c.get()          # 2
# c = Counter()  # ERROR — mut cannot be rebound
```

Fixed binding, content-mutable instance. `mut Counter c` is exactly `let mut Counter c` — the bare keyword is sugar for that, always, with no other reading. The pointer never changes — the compiler can apply alias analysis, loop-invariant hoisting, and register allocation optimizations that `var` prevents.

`let` and `var` are sufficient to write correct code. `mut` is an optional precision when you know the binding will never be rebound.

**`mut` on a scalar is a compile error**, not a silent downgrade to `var`. Primitives (`int`, `uint`, `float`, `bool`) expose no `def` methods, so there's nothing for `mut` to unlock:

```boring
# mut a = 0        # ERROR — no def methods exist on int; use `var` for a rebindable scalar
var a = 0          # ok — rebindable scalar
a = 1              # ok — rebind
mut Counter c = Counter()  # ok — fixed binding, content-mutable instance
# c = Counter()    # still an error — mut on struct is non-rebindable
```

(An earlier version of Boring treated `mut` on a scalar as equivalent to `var` — that shortcut is retired.)

### Rebindable bindings — `var`

```boring
var Counter c = Counter()
# c.inc()          # ERROR — var alone does not grant content mutation
c = Counter()      # ok — rebind to a fresh instance
c.get()            # 0
```

Rebindable binding — the variable can point to a different instance. On its own, `var` grants **only** rebinding, not content mutation — write `var mut` for both:

```boring
var mut Counter c = Counter()
c.inc()          # ok — both rebindable and content-mutable
c = Counter()    # ok — rebind
c.get()          # 0 — started over
```

The difference between `mut` and `var mut` is rebindability:

```boring
mut Counter a = Counter()
var mut Counter b = Counter()

a.inc()          # ok
b.inc()          # ok
# a = Counter()  # ERROR — mut cannot be rebound
b = Counter()    # ok — var mut can
```

### `mut` composes into any type position

Since `mut` is a modifier on the **type**, not a fifth binding keyword, it nests anywhere a type can appear — a tuple slot, a struct field (see [§8](#8-structs)), an array element or dict value (see [§7](#7-collections)), or a generic argument:

```boring
let (mut Point, string) t = (Point(0, 0), "origin")   # slot 0 is content-mutable, slot 1 isn't
t.0.move_to(1, 1)   # ok
# t.1 = "elsewhere" # ERROR — the tuple binding itself isn't `var`/reassignable at that slot

Query<(Position&, mut Velocity&)>   # a borrowed tuple slot can carry mut too — see below
```

`mut Type` (no `&`) is only accepted where the checker can attach a stable, addressable permission to it — a tuple slot, a struct field, an array element, or a dict value. It's rejected on a scalar (above), on `'shared` (no interior mutability to unlock), on a whole tuple as a block (`mut (T1, T2) t` — no in-place mutation surface for a tuple *as a whole*, only per-slot), on a `'weak` reference (nothing but `.upgrade()`/`.clone()` to call on it), and on a set element (`HashSet<T>` has no mutable element access in Rust).

### The borrow form — `mut Type&`

`Type&` is a borrow (`&Type` in Rust); `mut Type&` is a mutable borrow (`&mut Type`). Because it's a genuine, distinct Rust type — unlike the owned `mut Type` form above — it composes freely into tuples and generic arguments, which is exactly what a Bevy-ECS-style query needs:

```boring
struct Position: var float x; var float y
struct Velocity: var float dx; var float dy

var p = Position(x = 0.0, y = 0.0)
var v = Velocity(dx = 1.0, dy = 0.0)
let (Position&, mut Velocity&) pair = (p, v)
pair.1.dx = pair.1.dx + 1.0   # ok — slot 1 is a mutable borrow
# pair.0.x = 5.0              # ERROR — slot 0 is a plain (immutable) borrow
```

There's no sigil for this (`Type!`/`Type*` were both considered and dropped) — `mut Type&` stays consistent with the rest of Boring's spelled-out-keyword surface syntax.

### Coercion is one-way

A `mut Type` value carries strictly more permission than a plain `Type` value, so it's always safe to use where only `Type` is expected, and never safe the other way:

```boring
def readOnly(Point p): ...
def mutate(mut Point p): p.x = 1

mut Point a = Point(0, 0)
let Point b = Point(0, 0)

readOnly(a)   # ok — mut Point widens to Point
mutate(a)     # ok — a already is mut Point
# mutate(b)   # ERROR — Point cannot be used where mut Type is required
```

Inference follows the same direction: a variable's type inferred from another variable's value always resolves to the **plain** type, never to `mut Type`, even when the source is `mut`-typed — `mut`-ness is requested explicitly at each binding site, never inherited implicitly through aliasing.

```boring
let mut Point b = Point(0, 0)
let a = b       # a's inferred type is Point, NOT mut Point
```

### Compound assignment operators

The operators `+=`, `-=`, `*=`, `/=` and `%=` update a variable in place.
They are statements only (not expressions).

```boring
var x = 10
x += 5    # 15
x -= 3    # 12
x *= 2    # 24
x /= 4    # 6
x %= 4    # 2
```

**Rust equivalent**
```rust
let mut x: isize = 10;
x += 5;   // 15
x -= 3;   // 12
x *= 2;   // 24
x /= 4;   // 6
x %= 4;   // 2
```

These operators work on all numeric types (`int`, `uint`, `float`).
String concatenation is not done with `+=` — use reassignment instead: `var string s = "hi"` then `s = "{s} more"`.

### Explicit type annotations

The type is written **before** the name:

```boring
let int x = 42
let float pi = 3.14159
let string label = "ok"
var bool flag = true
```

**Rust equivalent**
```rust
let x: isize = 42;
let pi: f64 = 3.14159;
let label: &str = "ok";   // literal stays &str; promoted to Rc<str> when stored
let mut flag: bool = true;
```

### Deferred initialisation

Declare a variable without an initial value and assign it in each branch of an `if`/`else` or `match`. Useful when the computation is too imperative for the expression form `let v = if ...:`.

```boring
# if/else
let v
if condition:
    v = "big"
else:
    v = "small"
print v

# match
var n
match status:
    200: n = "ok"
    404: n = "not found"
    _:   n = "error"
```

**Rules:**
- Works with `let`, `mut`, and `var`.
- Reading the variable before any assignment is a **runtime error**: `variable 'v' used before being assigned`.
- The transpiler emits `let v;` / `let mut v;` and Rust's own control-flow analysis ensures every path assigns the variable before use.

**Rust equivalent**
```rust
let v;
if condition {
    v = "big";
} else {
    v = "small";
}
```

> **Prefer the expression form** when branches are simple:
> `let v = if condition: "big" else: "small"`

### Move semantics

Assigning a non-Copy value to a new binding **moves** it — the source binding becomes inaccessible:

```boring
let a = [1, 2, 3]
let b = a          # a is moved into b
print b            # [1, 2, 3]
print a            # error: use of moved value 'a': the value was moved
                    #        and is no longer accessible — use .clone() to make a copy
```

- Copy types (`int`, `uint`, `float`, `bool`, `string`, `nil`) are unaffected — assigning them always copies, both bindings stay usable.
- Non-Copy types (structs, arrays, dicts, sets, enums, tuples) are moved on assignment and rebinding. **Function parameters are an exception** — the transpiler passes them by reference automatically, so calling a function never moves a value.
- For `'shared`/`'actor`/`'guard` bindings, assignment is an implicit alias (no move, no deep copy — just a refcount increment). Both bindings remain valid.
- For other non-Copy types, call `.clone()` to make an explicit deep copy instead of moving: `let b = a.clone()` leaves `a` usable.
- There is no explicit move-marker syntax — the transpiler infers moves vs. copies from the value's type.

---

## 3. Data Types

### Scalar types

| Boring         | Rust            | Notes                              |
|----------------|-----------------|-------------------------------------|
| `int`          | `isize`         | pointer-width signed integer        |
| `uint`         | `usize`         | pointer-width unsigned integer      |
| `int8`         | `i8`            | 8-bit signed integer                |
| `int16`        | `i16`           | 16-bit signed integer               |
| `int32`        | `i32`           | 32-bit signed integer               |
| `int64`        | `i64`           | 64-bit signed integer               |
| `int128`       | `i128`          | 128-bit signed integer              |
| `uint8`        | `u8`            | 8-bit unsigned integer              |
| `uint16`       | `u16`           | 16-bit unsigned integer             |
| `uint32`       | `u32`           | 32-bit unsigned integer             |
| `uint64`       | `u64`           | 64-bit unsigned integer             |
| `uint128`      | `u128`          | 128-bit unsigned integer            |
| `float32`      | `f32`           | 32-bit floating-point               |
| `float64`      | `f64`           | 64-bit floating-point (`float` is a pure alias of this — see below) |
| `bool`         | `bool`          | `true` / `false`                    |
| `string`       | `&str` (literal) / `Rc<str>` · `Arc<str>` (stored/computed) | Literals stay `&str`; the transpiler promotes to `Rc<str>` (single-thread) or `Arc<str>` (multi-thread) when the context requires it |

The Rust-style spellings `i8`/`i16`/`i32`/`i64`/`i128`/`isize`, `u8`/`u16`/`u32`/`u64`/`u128`/`usize`, and `f32`/`f64` are accepted as aliases for the corresponding Boring name (`i32` ≡ `int32`, `isize` ≡ `int`, `f32` ≡ `float32`, etc.) and transpile identically.

**`float` is `float64`, not a third type.** Unlike `int`/`uint` (which map to pointer-width `isize`/`usize`, independent of `int64`/`uint64`), `float` has no independent identity — write `float`, get exactly `float64`, indistinguishable at every stage after parsing. This is why the table above lists `float32`/`float64` as the two real members of the family rather than giving `float` its own row.

### Fixed-width integers

Each fixed-width type is a distinct type at runtime — mixing two *different* explicit fixed-width types in one expression (`a_uint16 + a_int32`) is a type error, same as Rust's own refusal to implicitly coerce between distinct integer types. Cast explicitly instead:

```boring
let uint16 a = 100
let int32 b = 5
let c = (a as int32) + b   # explicit cast required
```

A fixed-width value mixes freely with a bare integer literal or an `int`/`uint`-typed value (the flexible kind literals use by default):

```boring
let uint32 count = 10
let bigger = count + 1     # fine — `1` is the flexible `int` literal kind
```

Casting to a narrower type checks the range and produces `nil` (or errors, depending on context) if the value doesn't fit — same as the existing `uint8` behavior (`300 as uint8` is out of range).

**GPU targets.** Each GPU backend's own numeric type system limits which widths a `kernel` struct field can use:

| Width | `--target wgpu` (WGSL) | `--target cuda` (CUDA C) | `--target rocm` (HIP C++) | `--target metal` (MSL) |
|-------|-------------------------|---------------------------|-----------------------------|--------------------------|
| 8-bit | not supported (compile error) | full support (`uint8_t`/`int8_t`) | full support (`uint8_t`/`int8_t`) | full support (`uchar`/`char`) |
| 16-bit | not supported (compile error) | full support | full support | full support (`ushort`/`short`) |
| 32-bit | full support (`i32`/`u32`) | full support | full support | full support (`int`/`uint`) |
| 64-bit | not supported (compile error) | full support | full support | not supported (compile error) |
| 128-bit | not supported (compile error) | supported via the non-standard `__int128` GCC/NVCC extension | supported via the non-standard `__int128` GCC/HIP-clang extension | not supported (compile error) |
| 32-bit float (`float32`) | full support (native `f32`) | full support (`float`) | full support (`float`) | full support (native `float`) |
| 64-bit float (`float64`/`float`) | not supported (compile error — no `f64` in WGSL at all) | full support (`double`) | full support (`double`) | not supported (compile error — MSL has no native `double`) |

WGSL has no native integer type below or above 32 bits, and no 64-bit float type at all; MSL has no native 64/128-bit integer and no native `double` (Apple GPUs historically lack native 64-bit ALU ops, for both integers and floats). ROCm's HIP C++ mirrors CUDA C's numeric type system (same underlying LLVM/Clang toolchain), so it has identical width support, floats included. Using an unsupported width — or `float64`/`float` on Metal or wgpu — on a kernel field produces a clear error at the point the type would be emitted, rather than silently mis-narrowing the data; use `float32` there instead.

### Fixed-width floats

`float32`/`float64` follow the exact same strict-mixing rule as the fixed-width integers above — two different widths never mix implicitly:

```boring
let float32 a = 1.0
let float64 b = 2.0
let c = a + b               # ERROR — cannot mix float32 and float64
let c = (a as float64) + b  # OK — explicit cast
```

An untyped float literal mixes freely with either width, resolved from context — the same precedent as an untyped int literal mixing with any fixed-width integer:

```boring
let float32 a = 1.0
let c = a + 3.14   # OK — 3.14 resolves to float32 here
```

`float` is `float64`'s alias (see "Scalar types" above), so it participates in this rule exactly as `float64` would — `a_float32_var + a_float_var` is exactly as much a type error as mixing two explicit `float32`/`float64` values.

### Integer literals

```boring
let a = 42
let b = -7
```

Any base is supported:

```boring
let hex = 0xFF_AA_00      # hexadecimal
let bin = 0b1111_0000     # binary
let oct = 0o755           # octal
```

#### Digit separators `_`

Use `_` anywhere inside a numeric literal to improve readability. The underscores are stripped — they have no effect on the value:

```boring
let million  = 1_000_000
let billion  = 1_000_000_000
let color    = 0xFF_AA_00     # hex
let byte     = 0b1111_0000    # binary
```

Works in all bases (decimal, hex, binary, octal) and in floats.

### Float literals

```boring
let pi = 3.14159
let e  = 2.71828
```

```boring
let pi_precise = 3.141_592_653   # digit separator for readability
let avogadro   = 6.022_140_76
```

### Boolean literals

```boring
let yes = true
let no  = false
```

### String types and inference

**Use bare `string` by default** — it works everywhere: function parameters, return types, variables, and fields. The compiler figures out the memory representation automatically.

```boring
let a = "hello"                # string — inferred from literal
let string b = "world"         # same
let string c = a + " " + b     # string — concatenation
let string d = "Hi, {a}!"      # string — interpolation
```

> Boring infers the string representation automatically. String literals stay as `&str`; the transpiler promotes them to `Rc<str>` (single-thread) or `Arc<str>` (multi-thread) when the context requires it (storage in a variable, field, collection, or concatenation). See [Advanced — Strings](#advanced--strings-string-and-stringshared) for details.

### String interpolation

Any expression can be embedded with `{expr}`:

```boring
let name = "Alice"
let age  = 30
print "Name: {name}, Age: {age}"
print "Next year: {age + 1}"
```

**Rust equivalent**
```rust
println!("Name: {}, Age: {}", name, age);
println!("Next year: {}", age + 1);
```

### Multi-line strings — `"""..."""`

Triple-quoted strings span multiple lines. The indentation of the closing `"""` is stripped from every content line automatically, so the string body can be aligned with the surrounding code.

```boring
let sql = """
    SELECT *
    FROM users
    WHERE age > 18
    """

let html = """
    <div class="box">
        <p>Hello!</p>
    </div>
    """
```

The first newline (right after the opening `"""`) and the last newline (right before the closing `"""`) are stripped. The result is a clean string without leading or trailing newlines.

**Interpolation** works the same as in regular strings — `{expr}` is a hole:

```boring
let table  = "orders"
let limit  = 100

let query = """
    SELECT *
    FROM {table}
    LIMIT {limit}
    """
```

**Literal `{` and `}`** follow the same escaping rules as regular strings: use `{{` for a literal `{` and `}}` for a literal `}`:

```boring
let json_tpl = """
    {{"id": {user_id}, "active": true}}
    """
# → {"id": 42, "active": true}
```

**Single-line form** — `"""text"""` is valid and equivalent to `"text"`:

```boring
let msg = """Hello, World!"""
```

**Rust equivalent** — triple strings are expanded at lex time; the emitted Rust uses a regular string with `\n` for newlines and `format!` for interpolation:

```rust
let sql = format!("SELECT *\nFROM users\nWHERE age > 18\n");
let query = format!("SELECT *\nFROM {}\nLIMIT {}\n", table, limit);
```

### `print` and `write`

`print` outputs with a trailing newline; `write` outputs without one.

```boring
print "hello, {name}!"    # with newline → println!
write "loading..."        # without newline → print!
print ""                  # blank line
```

Both accept two styles — choose whichever reads more naturally:

```boring
# Inline interpolation — embed expressions directly
print "a={a}, b={b}"

# Positional substitution — `{}` placeholders, arguments after the string
print "a={}, b={}", a, b
write "{} / {}", numerator, denominator
```

### Tuples

Boring tuples are anonymous, fixed-length groups of values with per-element types.

#### Tuple literals and index access

```boring
let t = (42, "hello")      # inferred element types
let a = t.0                # 42  (int)
let b = t.1                # "hello"  (string)
```

**Rust equivalent**
```rust
let t = (42isize, Arc::from("hello"));
let a = t.0;
let b = t.1;
```

#### Typed tuple variable

The type annotation `(T1, T2)` is written **before** the name, consistent with all other type annotations. This is especially useful to guide generic inference:

```boring
let (int, string) t = (0, "hello")
```

**Rust equivalent**
```rust
let t: (isize, Arc<str>) = (0isize, Arc::from("hello"));
```

A tuple *as a whole* has no in-place mutation surface — no user-definable
methods on the tuple itself — so `mut` on a typed tuple **variable** is
rejected at check time (whether the tuple type is explicit or only inferred
from a literal initializer): there is no operation `mut` on the whole tuple
would unlock that `let` doesn't already allow. `var` remains valid and
meaningful — it allows reassigning `t` to a whole new tuple:

```boring
let (int, string) t = (0, "hello")   # fixed, read-only — fine
var (int, string) t = (0, "hello")   # reassignable — fine
t = (1, "world")

# mut (int, string) t = (0, "hello")   # ERROR — tuples have no in-place mutation
# mut t = (0, "hello")                 # ERROR too — same rule, inferred type
```

**Individual slots are a different matter.** `mut` composes into a tuple's
*element* positions — each slot is fixed and addressable (`t.0`, `t.1`),
exactly like a struct field, so a slot's `mut`-ness controls whether `def`
can be called on whatever it holds:

```boring
struct Point: var int x; var int y

let (mut Point, string) t = (Point(0, 0), "origin")
t.0.move_to(5, 5)     # ok — slot 0's type is `mut Point`
# t.1 = "elsewhere"   # ERROR — slot 1 isn't reassignable (the binding is `let`)
```

This does not conflict with the whole-tuple rejection above — `mut` *before*
the tuple's parentheses (`mut (T1, T2) t`) governs the whole tuple and stays
rejected; `mut` *inside* the parentheses (`(mut T1, T2) t`) is a different
grammar position, governing one slot.

#### Destructuring

Each destructured element may carry its own explicit `let`/`mut`/`var`/`var mut`
keyword. An unmarked element's default depends on whether the destructure is
parenthesised:

```boring
let a, b = t                    # bare — a: let, b: let (bare always defaults to let)
let (a, b) = t                   # parenthesised — a: let, b: let (inherits the group's `let`)
let mut a, b = t                 # bare — a: mut, b: LET — b does NOT inherit `mut`
let mut (a, b) = t                # parenthesised — a: mut, b: mut (both inherit the group's `mut`)
let (mut a, b) = t                 # parenthesised — a: mut (explicit), b: let (inherits the group's `let`)
```

**This is a real readability trap on the bare form specifically**: `let mut
a, b = t` reads, at a glance, like the keyword phrase governs the whole
line — it doesn't; `b` quietly defaults to plain `let`. The rule is
unambiguous once learned (bare defaults to `let` unconditionally, per
element, regardless of a sibling's own keyword; parenthesised inherits the
group's own leading keyword instead) but is exactly the kind of thing worth
double-checking — the compiler emits a lint warning for precisely this shape
(an unmarked bare element right after a differently-keyworded one).

Every statement that uses a single keyword throughout, marked or not, means
exactly what it always did — only the *mixed* case above is new:

```boring
let a, b = t               # bare
let (a, b) = t             # parenthesised
let int a, string b = t    # bare with per-variable types
let (int a, string b) = t  # parenthesised with per-variable types
```

In these, each variable gets its own type annotation; types on one variable do **not** apply to the others.

**Rust equivalent**
```rust
let (a, b) = t;   // types inferred from tuple
```

#### Tuple return type

```boring
(int, int) minmax([int] nums):
    (nums[0], nums[1])
```

**Rust equivalent**
```rust
fn minmax(nums: Vec<isize>) -> (isize, isize) {
    (nums[0].clone(), nums[1].clone())
}
```

#### Tuple methods

Tuples expose a small set of methods. Because tuple elements can have different types, methods like `filter` or `sort` that assume a homogeneous element type do not exist on tuples.

| Boring                  | Description                                      |
|-------------------------|--------------------------------------------------|
| `t.length()`            | Number of elements (compile-time constant)       |
| `t.isEmpty()`           | `true` for the unit tuple `()`                   |
| `t.first()`             | First element (`t.0`)                            |
| `t.last()`              | Last element (`t.N-1`)                           |
| `t.map(x: expr)`        | Apply a closure to each slot; returns a new tuple of the same arity |
| `t.all(x: cond)`        | `true` if the closure returns `true` for every element |
| `t.any(x: cond)`        | `true` if the closure returns `true` for at least one element |

`map` works correctly for heterogeneous tuples because Rust infers the result type of each slot independently. The primary use case is applying the same operation to every element — for example extracting a field from a pair of structs, or testing a condition across a group of futures:

```boring
# Extract a field from each element
struct Point:
    int x
    int y

let points = (Point(1, 2), Point(3, 4))
let (x1, x2) = points.map(:x)      # (1, 3)

# Poll a group of futures
let f1 = task compute1()
let f2 = task compute2()
while !(f1, f2).all(:done): wait fromSecs(1)
let (r1, r2) = (f1, f2).map(:value)
```

**Rust equivalent of `map`**
```rust
let (x1, x2) = {
    let __t = points;
    ({ let __x = __t.0.clone(); __x.x }, { let __x = __t.1.clone(); __x.x },)
};
```

### Type casting — `as`

```boring
let n    = "42" as int          # Option<int>
let f    = 42 as float
let s    = 3 as string
let safe = ("42" as int) else -1
```

**Rust equivalent**
```rust
let n: Option<isize> = "42".parse().ok();
let f = 42isize as f64;
let s = 3isize.to_string();
let safe = "42".parse::<isize>().unwrap_or(-1);
```

Supported cast targets: `int`, `uint`, `float`, `string`, `bool`, and any type that implements `Display` or `FromStr`.

### Reference identity — `===`

`===` tests whether two variables point to the **exact same object in memory**, bypassing any user-defined `==`. Two objects may compare equal with `==` while being distinct instances; `===` distinguishes them.

```boring
struct Point:
    init(pub int x, pub int y)

let Point'shared a = Point(x = 1, y = 2)
let b = a                      # b is an alias of a — same reference (implicit Rc/Arc clone)
let c = Point(x = 1, y = 2)   # c is a new, distinct object

print "{a === b}"   # true  — same reference
print "{a === c}"   # false — same value, different object
```

For `'shared`, `'actor`, and `'guard` qualifiers, assignment is always an **implicit alias** — the reference count is incremented automatically, and both bindings stay valid. No `.clone()` needed.

For unqualified structs and `'stack`/`'heap` types, plain assignment **moves** the value (see "Move semantics" above).

For primitive types (`int`, `float`, `bool`) which have value semantics, `===` behaves like `==`.

**Rust equivalent**
```rust
Arc::ptr_eq(&a, &b)   // T'shared objects (multi-thread)
Rc::ptr_eq(&a, &b)    // T'shared objects (single-thread)
```

---

## 4. Functions

### Basic definition

When a return type is explicitly written, `def` can be omitted — the declaration reads like Java or C++:

```boring
int add(int a, int b):      # def omitted — return type present
    a + b

def int add(int a, int b):  # equivalent — explicit def
    a + b
```

The last expression in the body is the implicit return value. No `return` keyword needed.

**Rust equivalent**
```rust
fn add(a: isize, b: isize) -> isize {
    a + b
}
```

### The `def` keyword is optional

When a return type is present, `def` can be omitted entirely:

```boring
int f(int n): n * 2       # no keyword — return type implies def
def int f(int n): n * 2   # identical — explicit def
```

`def` is **required** when there is no return type (void functions):

```boring
def log(string msg):      # def required — no return type
    print "[LOG] {msg}"
```

> `req` (introduced in section 8 — Structs) is also available for top-level functions to signal that a function is pure. Both produce the same Rust `fn` at the top level.

Functions can return any collection type — arrays, sets, and dicts are all supported return types:

```boring
[int] first_n(int n):
    var result = []
    for i in 1..=n: result.push(i)
    result

{int} unique_squares():
    {1, 4, 9, 4, 1}                # deduplicates automatically

{string=int} char_count(string s):
    var counts = {=}
    for ch in s.chars():
        counts[ch] = (counts[ch] else 0) + 1
    counts
```

**Rust equivalent**
```rust
fn first_n(n: isize) -> Vec<isize> { (1..=n).collect() }
fn unique_squares() -> std::collections::HashSet<isize> { [1,4,9].into() }
fn char_count(s: &str) -> std::collections::HashMap<Arc<str>, isize> { ... }
```

### Multi-line parameter lists

Both definitions and call sites accept newlines inside `(...)`. A trailing comma is optional.

```boring
int add(
    int a,
    int b,
):
    a + b

let result = add(
    10,
    20,
)
```

Works for `def`, `req`, and any call expression.

### Inferred types — closures only

Closures can omit parameter types — the type is inferred at call time:

```boring
let double = (n): n * 2      # n inferred as int or float at call site
let add    = (a, b): a + b

double(3)      # 6
double(1.5)    # 3.0
```

`def` / `req` declarations **require** explicit type annotations on every parameter. This keeps function signatures self-documenting and ensures valid Rust output:

```boring
def int add(int a, int b):   # required
    a + b
```

Omitting a type on a `def` parameter is a parse error.

### `void` functions

`void` can be omitted — a function with no return type is implicitly void. Three forms are all equivalent:

```boring
def greet(string name):        # no return type — implicitly void
    print "Hello, {name}!"

def void greet(string name):   # explicit void with def
    print "Hello, {name}!"

void greet(string name):       # explicit void, def omitted
    print "Hello, {name}!"
```

**Rust equivalent**
```rust
fn greet(name: &str) {
    println!("Hello, {}!", name);
}
```

### Explicit `return`

```boring
int abs_val(int n):
    if n < 0: return -n
    n
```

### Default parameters

```boring
string say(string msg = "hi"):
    msg
```

```boring
say()        # "hi"
say("hello") # "hello"
```

**Rust equivalent** (default values are inlined at call sites)
```rust
fn say(msg: &str) -> Arc<str> { Arc::<str>::from(msg.to_string()) }
```

### Labeled arguments

Any argument can be passed by name using `name= value` syntax. Labels allow calling out of declaration order and make call sites self-documenting.

```boring
string greet(string name, string greeting):
    "{greeting}, {name}!"

# Positional
greet("Alice", "Hello")                        # Hello, Alice!

# All labels — any order
greet(greeting = "Hi", name = "Bob")             # Hi, Bob!

# Mix: positional first, then labeled
greet("Carol", greeting = "Hey")                # Hey, Carol!
```

Labels also combine naturally with default parameters:

```boring
string format_num(int n, int base = 10, bool pad = false):
    ...

format_num(255)                    # base=10, pad=false
format_num(255, base = 16)          # hex, pad=false
format_num(255, base = 16, pad = true)
```

**Rust equivalent**
```rust
// Rust has no labeled args — keyword crates simulate them;
// boring compiles labeled calls to positional in declaration order.
greet("Alice".into(), "Hello".into())
greet("Bob".into(), "Hi".into())   // reordered to declaration order
```

### Pass-by-reference — automatic

Structs, enums, arrays, dicts, and sets are **always passed by reference** — you never write `&`. The transpiler injects `&` automatically; the caller keeps ownership and the value is never copied.

```boring
struct Point:
    float x
    float y

float length(Point p):          # p is &Point in Rust — no annotation needed
    (p.x * p.x + p.y * p.y)

def print_points(Point a, Point b):
    print "({a.x}, {a.y}) and ({b.x}, {b.y})"

let p = Point(3.0, 4.0)
let q = Point(0.0, 1.0)
let l = length(p)               # passes &p
print_points(p, q)              # both available after the call — no move
```

**Rust equivalent** (generated automatically)
```rust
fn length(p: &Point) -> f64 { p.x * p.x + p.y * p.y }
fn print_points(a: &Point, b: &Point) { ... }

length(&p);
print_points(&p, &q);
```

Primitive types (`int`, `float`, `bool`, `uint`) are `Copy` in Rust — they are always passed by value, with no overhead.

> When you need to **modify** the caller's variable, use `var` (see below). When you need to **share ownership** across threads or store the value, use a qualifier (`'shared`, `'actor`, etc.).

### Mutable parameters — `var`

Prefix a parameter with `var` to allow the callee to modify the caller's variable. The transpiler passes the argument by `&mut` — changes to the parameter are visible at the call site.

```boring
def add_one(var int x):
    x += 1

var n = 5
add_one(n)
print n    # 6
```

**Rust equivalent**
```rust
fn add_one(x: &mut isize) { *x += 1; }
let mut n = 5isize;
add_one(&mut n);
println!("{}", n); // 6
```

**For reference-counted types** (`T'shared`), `var` makes the local pointer reassignable — the object itself remains immutable through a shared reference. Field mutation and mutating methods are still forbidden.

- For `T'shared` (single-thread, `Rc<T>`): if you need to mutate, hold the value directly with `var` instead of sharing it via `T'shared`.
- For `T'shared` (multi-thread, `Arc<T>`): use `T'actor` when shared state needs to be written.

```boring
struct Counter:
    var int value = 0

def show(var Counter'shared c):
    print c.value          # OK — reading is always allowed
    c = Counter()          # OK — var lets you rebind the Arc/Rc pointer
    # c.value = c.value + 1  # ERROR — cannot mutate through T'shared; use T'actor
```

> For variadic parameters (`values...`), see [Advanced — Variadic parameters](#advanced--variadic-parameters).

### Function overloading

Multiple functions can share the same name as long as their parameter types differ. The compiler selects the right variant at the call site based on the argument types:

```boring
string describe(int n):    "number: {n}"
string describe(float f):  "float: {f}"
string describe(string s): "text: {s}"
string describe(bool b):   "flag: {b}"

describe(42)       # → "number: 42"
describe(3.14)     # → "float: 3.14"
describe("hello")  # → "text: hello"
describe(true)     # → "flag: true"
```

Multi-parameter overloads work the same way — the full signature (all parameter types) must be unique:

```boring
string fn(int a, string b): "{a}/{b}"
string fn(int a, bool c):   "{a}/{c}"
string fn(int a):           "{a}"
```

**Rust equivalent** — the transpiler mangles names with the parameter types:
```rust
fn describe__int(n: isize) -> Arc<str> { ... }
fn describe__float(f: f64) -> Arc<str> { ... }
fn describe__string(s: Arc<str>) -> Arc<str> { ... }
fn describe__bool(b: bool) -> Arc<str> { ... }
```
Call sites are resolved statically: `describe(42)` → `describe__int(42)`.

**Struct methods** can also be overloaded within `ext` blocks:

```boring
struct Animal:
    string name

ext Animal:
    req string speak(int times):    "{self.name} x{times}"
    req string speak(string sound): "{self.name}: {sound}!"

let a = Animal(name = "Dog")
print a.speak(3)       # → Dog x3
print a.speak("woof")  # → Dog: woof!
```

**Rust equivalent** — method names are also mangled:
```rust
fn speak__int(&self, times: isize) -> Arc<str> { ... }
fn speak__string(&self, sound: Arc<str>) -> Arc<str> { ... }
```

---

**Conflict detection** — the compiler rejects overloads that create ambiguity. A function with default parameters can be called with fewer arguments; if that reduced call matches another overload, it is an error:

```boring
string fn(int n, string s = "x"):  # callable as fn(int) OR fn(int, string)
string fn(int n):                   # ERROR — conflicts at arity 1
```

```
error: ambiguous overload for 'fn' — 'fn(int, string=default)' and 'fn(int)'
       both match a call with 1 argument(s)
```

**Limitations**

| Context | Status |
|---|---|
| Free functions in the same file | ✅ fully supported |
| Struct methods declared inline in `struct` | ✅ fully supported |
| Struct methods in `ext` blocks (same file) | ✅ fully supported |
| Functions / methods across separate files or modules | ❌ not supported |

For cross-file scenarios, declare all overloads in the same file. If the overloads must live in separate files, use different function names.

---

### Throwing functions — `throws`

A function that may fail is annotated with `throws`. It returns `Result<T, Box<dyn Error>>` in Rust.

```boring
int divide(int a, int b) throws:
    guard b != 0 else throw "division by zero"
    a / b
```

**Rust equivalent**
```rust
fn divide(a: isize, b: isize) -> Result<isize, Box<dyn std::error::Error>> {
    if b == 0 { return Err("division by zero".into()); }
    Ok(a / b)
}
```

### Function-typed parameters (higher-order)

```boring
int apply(int f(int), int x):
    f(x)
```

**Rust equivalent**
```rust
fn apply(f: impl Fn(isize) -> isize, x: isize) -> isize {
    f(x)
}
```

---

## 5. Comments

```boring
# This is a comment
let x = 42   # inline comment (not preserved by the transpiler)
```

**Rust equivalent** (full-line comments are preserved by `boring build`)
```rust
// This is a comment
let x: isize = 42;
```

Only full-line comments are preserved in the transpiled output. Inline comments are stripped.

---

## 6. Control Flow

### `if` / `elif` / `else`

`if` is an **expression** — it returns a value.

```boring
string sign(int n):
    if n > 0: "+" elif n < 0: "-" else "0"
```

Block form:

```boring
if score >= 90:
    print "A"
elif score >= 80:
    print "B"
else:
    print "C"
```

**Rust equivalent**
```rust
if score >= 90 {
    println!("A");
} else if score >= 80 {
    println!("B");
} else {
    println!("C");
}
```

The `then` and `else` branches can be independently inline or multiline — all four combinations are valid:

```boring
# both inline
if ok: print "yes" else: print "no"

# inline then, multiline else
if ok: print "yes"
else:
    log "no"
    return

# multiline then, inline else
if ok:
    log "yes"
    proceed()
else: return

# both multiline (standard block form — shown above)
```

### `if let`

Unwrap an optional and bind it in a single step:

```boring
let string? maybe = "found"
if let v = maybe:
    print "got: {v}"
```

**Rust equivalent**
```rust
if let Some(v) = maybe {
    println!("got: {}", v);
}
```

#### `if let` shorthand

When the binding name is the same as the variable being unwrapped, the `= expr` part can be omitted — like Swift:

```boring
let string? name = get_name()

if let name:            # ≡  if let name = name:
    print "hello {name}"
else:
    print "anonymous"
```

Multiple clauses work too:

```boring
let int? age = get_age()

if let name, let age:
    print "{name} is {age}"
```

#### `elif let`

`elif` chains onto an `if let`, with its own comma-separated clause list — tried in order, first match wins:

```boring
if let v = a:
    print "a = {v}"
elif let v = b, v > 0:
    print "b = {v} (positive)"
elif c > 10:
    print "c is large"
else:
    print "nothing matched"
```

A plain boolean `elif` (no `let`) is also allowed, as in the `c > 10` branch above.

### `guard`

Early exit if a condition fails:

```boring
string check(int n):
    guard n >= 0 else return "negative"
    "ok ({n})"
```

**Rust equivalent**
```rust
fn check(n: isize) -> Arc<str> {
    if n < 0 { return Arc::from("negative"); }
    Arc::<str>::from(format!("ok ({})", n))
}
```

### `guard let`

Unwrap an optional, or exit early:

```boring
string connect(string? host):
    guard let h = host else return "no host"
    "Connecting to {h}"
```

The shorthand form works here too — omit `= var` when names match:

```boring
string connect(string? host):
    guard let host else return "no host"   # ≡  guard let host = host else …
    "Connecting to {host}"
```

**Rust equivalent**
```rust
fn connect(host: Option<Arc<str>>) -> Arc<str> {
    let Some(host) = host else { return Arc::from("no host"); };
    Arc::<str>::from(format!("Connecting to {}", host))
}
```

### `match`

```boring
string describe(int n):
    match n:
        0:           "zero"
        1 | 2 | 3:   "small"
        x if x < 0: "negative"
        _:           "large"
```

**Rust equivalent**
```rust
fn describe(n: isize) -> Arc<str> {
    match n {
        0          => "zero",
        1 | 2 | 3  => "small",
        x if x < 0 => "negative",
        _          => "large",
    }
}
```

### `while`

```boring
var i = 0
while i < 5:
    i += 1
```

Inline form:

```boring
var i = 0
while i < 5: i += 1
```

**Rust equivalent**
```rust
let mut i = 0isize;
while i < 5 { i += 1; }
```

### `while let`

Loop while an expression returns a non-`nil` value, binding it each iteration.
The loop stops as soon as the expression returns `nil`.

Inline form:

```boring
while let v = next_item(): process(v)
```

Block form:

```boring
int? next_item(int i):
    if i < 3: i else nil   # returns nil when exhausted

var idx = 0
while let v = next_item(idx):
    print "got {v}"
    idx += 1
```

**Rust equivalent**
```rust
while let Some(v) = next_item(idx) {
    println!("got {}", v);
    idx += 1;
}
```

The idiom is especially useful with channels and iterators — stop looping
when the producer closes (returns `nil`):

```boring
# consume all ids sent over a channel
while let task_id = receiver.recv():
    process(task_id)
```

**Rust equivalent**
```rust
while let Some(task_id) = receiver.recv().await {
    process(task_id).await;
}
```

The shorthand also applies:

```boring
var line = read_line()

while let line:           # ≡  while let line = line:
    process(line)
    line = read_line()
```

### `for` over a collection

```boring
for word in ["boring", "is", "fun"]:
    print "  {word}"
```

Inline form — body on the same line as `for`:

```boring
for word in ["boring", "is", "fun"]: print "  {word}"
```

**Rust equivalent**
```rust
for word in ["boring", "is", "fun"].iter() {
    println!("  {}", word);
}
```

### `for` with index — auto-enumerate

When two variables are given and the collection is an array of **non-tuple** values, the index is automatically injected as the first variable — no `.enumerate()` call needed:

```boring
let fruits = ["apple", "banana", "cherry"]

# Shorthand — i = index, v = element
for i, v in fruits:
    print "{i}: {v}"
# 0: apple
# 1: banana
# 2: cherry

# Equivalent long form — still works
for i, v in fruits.enumerate():
    print "{i}: {v}"
```

**Rust equivalent**
```rust
for (i, v) in fruits.iter().enumerate() {
    println!("{}: {}", i, v);
}
```

> If the array already contains tuples (e.g. from `.zip()` or `.enumerate()`), the two-variable form destructures the tuple instead of re-enumerating:
> ```boring
> let pairs = [(1, "one"), (2, "two")]
> for a, b in pairs:    # a=1, b="one" — tuple destructuring, not auto-enumerate
>     print "{a}-{b}"
> ```

### `for` over a dict — key-value destructuring

Two variables over a dict automatically bind to key and value, in order. No `firstIndex`/`nextIndex` needed:

```boring
let scores = {"Alice" = 90, "Bob" = 85, "Carol" = 92}

for name, score in scores:
    print "{name}: {score}"
# Alice: 90
# Bob: 85
# Carol: 92
```

**Rust equivalent**
```rust
for (name, score) in scores.iter() {
    println!("{}: {}", name, score);
}
```

A single variable over a dict binds the key only:

```boring
for name in scores:
    print name   # Alice, Bob, Carol
```

### `for` over a range

```boring
for k in 1..=5:       # inclusive: 1, 2, 3, 4, 5
    print "{k}"

for k in 1..5:        # exclusive: 1, 2, 3, 4
    print "{k}"
```

**Rust equivalent**
```rust
for k in 1isize..=5 { println!("{}", k); }
for k in 1isize..5  { println!("{}", k); }
```

### `for` without a variable — repeat N times

When you only need to repeat a body without using the index, omit the binding entirely:

```boring
for 1..=5:
    print "tick"          # prints "tick" five times

# The explicit form with _ is equivalent
for _ in 1..=5:
    print "tick"
```

**Rust equivalent**
```rust
for _ in 1isize..=5 { println!("tick"); }
```

### `loop` with `break` and `continue`

```boring
var acc = 0
var idx = 0
loop:
    idx += 1
    if idx % 2 == 0: continue
    acc += idx
    if idx >= 7: break
```

Inline form:

```boring
loop: tick()
```

**Rust equivalent**
```rust
let mut acc = 0isize;
let mut idx = 0isize;
loop {
    idx += 1;
    if idx % 2 == 0 { continue; }
    acc += idx;
    if idx >= 7 { break; }
}
```

### `loop` as expression

A `loop:` block can produce a value. Use `break expr` to return the value; the result is bound with `let`:

```boring
let found = loop:
    let x = next_item()
    if x > 10: break x

print "found: {found}"   # first x > 10
```

Counting with an accumulator:

```boring
var n = 0
let total = loop:
    n += 1
    if n >= 5: break n * 2

print "total: {total}"   # 10
```

**Rust equivalent**
```rust
let total = loop {
    n += 1;
    if n >= 5 { break n * 2; }
};
```

---

### `do` / `while`

```boring
var j = 0
do:
    j += 1
while j < 3
```

Inline form:

```boring
var j = 0
do: j += 1 while j < 3
```

**Rust equivalent**
```rust
let mut j = 0isize;
loop {
    j += 1;
    if !(j < 3) { break; }
}
```

### `do:` — scoped block

`do:` without `while` creates an isolated scope. It has its own `defer` frame and returns the value of its last expression. Useful to limit variable lifetimes or compute a value in a block:

```boring
let result = do:
    let a = expensive_compute()
    let b = a * 2
    b + 1          # value of the do block

# a and b are not accessible here
```

**Rust equivalent**
```rust
let result = {
    let a = expensive_compute();
    let b = a * 2;
    b + 1
};
```

`defer` inside a `do:` block runs when the block exits, not when the function exits:

```boring
let msg = do:
    defer: print "block done"
    "hello"
print "after: {msg}"   # prints "block done" first, then "after: hello"
```

### `;` — statement separator

A semicolon separates statements on the same line. It is equivalent to a newline:

```boring
x = 1; print x
let a = 10; let b = 20; print a + b
```

This works inside any block, including loop and condition bodies:

```boring
for i in 0..<5: print i; total += i
if ok: log "start"; proceed()
```

Semicolons are optional everywhere — they exist for cases where grouping related statements on a single line improves readability.

---

## 7. Collections

### Arrays — `[T]`

```boring
let numbers = [1, 2, 3, 4, 5]
let [int] empty = []
```

**Rust equivalent**
```rust
let numbers: Vec<isize> = vec![1, 2, 3, 4, 5];
let empty: Vec<isize> = Vec::new();
```

#### Element mutability — `[mut T]` vs `mut [T]`

`mut` on the collection's own type (`mut [Point] arr`) and `mut` on the **element** type (`[mut Point] arr`) control two independent things — see [§2](#2-variables-and-mutability):

- **`[mut Point] arr`** — the array itself can't grow/shrink/reassign entries (unless also `mut`/`var mut`, see below), but every element already in it can have `def` called on it: `arr[0].move_to(...)`.
- **`mut [Point] arr`** — the reverse: `arr` supports structural mutation (`push`, `insert`, index-assign), but elements are plain `Point` — no `def` calls through an index read.
- **`mut [mut Point] arr`** — both.

```boring
struct Point: var int x; var int y

let [mut Point] a = [Point(0, 0), Point(1, 1)]
a[0].x = 5           # ok — element type is mut Point
a[0].move_to(1, 1)   # ok

let [Point] b = [Point(0, 0)]
# b[0].x = 5         # ERROR — element type is plain Point, not mut Point
# b[0].move_to(1, 1) # ERROR — same reason
```

This is checked for a bare `Var` collection with an explicit type annotation (an inferred-type collection's element permission isn't tracked).

#### Array fill and comprehension

Two shorthand forms create arrays without listing every element:

```boring
let zeros  = [0 for ..10]             # fill: 10 zeros
let squares = [i * i for i in ..5]   # comprehension: [0, 1, 4, 9, 16]
```

The range must be `..n` (exclusive, starting at 0) or `0..n`. In the comprehension form, `i` is bound to the index (0-based, type `int`). Both forms produce a `[T]` (`Vec<T>`).

**Rust equivalent**
```rust
let zeros: Vec<isize>   = vec![0isize; 10];
let squares: Vec<isize> = (0..5isize).map(|i| i * i).collect();
```

#### Fixed-size arrays — `[T, N]`

A compile-time size can be given as a second element of the type, separated by a comma. The result is a stack-allocated array of exactly `N` elements, analogous to Rust's `[T; N]`.

```boring
let [float, 4] v = [1.0, 2.0, 3.0, 4.0]
let [int, 3]   z = [0 for ..3]

struct Mat2:
    [float, 4] data                   # inline in a struct field

let m = Mat2(data= [0.0, 1.0, 0.0, 1.0])
print m.data[0]                       # 0
```

**Rust equivalent**
```rust
let v: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
let z: [isize; 3] = [0isize; 3];
```

`N` must be a non-negative integer literal — it cannot be a runtime variable. Use `[T]` (`Vec<T>`) when the size is dynamic.

#### Labeled multi-dimensional arrays — `[T, width, height]`

For 2 or more dimensions, `[T, N]`'s comma-separated slot generalizes into a
list of **labels** — every index and every axis size is then read by name,
not by position, so it's never ambiguous which one means rows and which
means columns:

```boring
let [float, width, height] a = [ 0.0 for width = 3, height = 4 ]  # dynamic shape
let [float, width = 3, height = 4] b                              # fixed shape

let v = a[width = 1, height = 2]   # order-free — a[height = 2, width = 1] is identical
let w = a.width                    # 3 — read-only shape-query property
```

- `[T, width, height]` (no `=`) — dynamic shape, sized at construction.
- `[T, width = W, height = H]` — fixed shape, compile-time sizes (`W`/`H` may
  be integer literals or const-generic expressions).
- The **first label declared is the fastest-varying axis** in memory
  (row-major storage) — `[T, width, height]` and `[T, height, width]` are
  different (transposed) types.
- `a[label = value, ...]` indexing requires every axis to be labeled — there
  is no positional form — and reads correctly regardless of the order the
  labels are written in.
- `a.label` — each declared axis is a read-only property, same no-parens
  convention as `arr.length` or a struct's `req` getter.
- `flat.reshape(width = W, height = H)` / `a.flatten()` convert to and from
  a plain `[T]` explicitly — never implicitly.

Chained `for` clauses build one directly, one clause per axis:

```boring
let grid = [ f(width, height) for width in ..W for height in ..H ]
```

Full reference — the fill shorthand (`[0.0 for width = w, height = h]`),
`.reshape()`/`.flatten()`, and the cross-label safety rule for passing
arrays between differently-labeled parameters:
[`array-multidim-types.md`](array-multidim-types.html). GPU-kernel-specific
behavior (qualifiers, grid inference): [`gpu-module.md`](gpu-module.html).

#### Array slicing

A sub-array can be extracted with slice syntax. The result is a new `[T]` containing a copy of the selected elements.

```boring
let a = [10, 20, 30, 40, 50]

let b = a[1..3]    # [20, 30]        — exclusive: indices 1 and 2
let c = a[..3]     # [10, 20, 30]    — from start up to (not including) 3
let d = a[2..]     # [30, 40, 50]    — from index 2 to end
let e = a[..]      # [10, 20, 30, 40, 50]  — full copy
let f = a[1..=3]   # [20, 30, 40]   — inclusive: indices 1, 2 and 3
```

Out-of-range bounds are clamped silently. An empty range (e.g. `a[3..1]`) produces `[]`.

**Rust equivalent**
```rust
let b = a[1..3].to_vec();
let c = a[..3].to_vec();
let d = a[2..].to_vec();
let e = a[..].to_vec();
let f = a[1..=3].to_vec();
```

#### Array methods

| Boring                      | Rust                              |
|-----------------------------|-----------------------------------|
| `arr.length`                | `arr.len()`                       |
| `arr.first()`                | `arr.first().cloned()`            |
| `arr.last()`                 | `arr.last().cloned()`             |
| `arr.push(v)`               | `arr.push(v)`                     |
| `arr.pop()`                 | `arr.pop()`                       |
| `arr.append(other)`         | `arr.extend(other)`               |
| `arr.insert(i, v)`          | `arr.insert(i, v)`                |
| `arr.remove(i)`             | `arr.remove(i)`                   |
| `arr.contains(v)`           | `arr.contains(&v)`                |
| `arr.sort()`                | `arr.sort()`                      |
| `arr.reverse()`             | `arr.reverse()`                   |
| `arr.map((x): expr)`        | `arr.iter().map(\|x\| expr).collect()` |
| `arr.filter((x): cond)`     | `arr.iter().filter(\|x\| cond).cloned().collect()` |
| `arr.reduce(init, (a,b): expr)` | `arr.iter().fold(init, \|a,b\| expr)` |
| `arr.any((x): cond)`        | `arr.iter().any(\|x\| cond)`      |
| `arr.all((x): cond)`        | `arr.iter().all(\|x\| cond)`      |
| `arr.join(sep)`             | `arr.join(sep)`                   |
| `arr.flat()`                | `arr.concat()`                    |
| `arr.isEmpty()`             | `arr.is_empty()`                  |

```boring
let doubled  = numbers.map((n): n * 2)
let evens    = numbers.filter((n): n % 2 == 0)
let total    = numbers.reduce(0, (acc, n): acc + n)
```

### Dictionaries — `{K=V}`

```boring
let {string=int} scores = {"Alice" = 90, "Bob" = 85}
let {int=int} empty_map = {=}
```

**Rust equivalent**
```rust
let scores: HashMap<Arc<str>, isize> = HashMap::from([...]); // literals coerced via .to_arc()
let empty_map: HashMap<isize, isize> = HashMap::new();
```

`mut` on the **value** position — `{K = mut V}` — is the dict analogue of `[mut T]`: it controls whether `def` calls work on values fetched via `d[k]` or iteration, independent of `mut {K=V} d` (structural `d[k]=v`/insertion/removal on the dict itself). Keys never accept `mut` — mutating one in place would invalidate the hash table, for either Boring or the underlying Rust `HashMap`.

#### Dictionary methods

| Boring                      | Rust                              |
|-----------------------------|-----------------------------------|
| `d.keys()`                  | `d.keys().cloned().collect()`     |
| `d.values()`                | `d.values().cloned().collect()`   |
| `d.contains(k)`             | `d.contains_key(&k)`              |
| `d.remove(k)`               | `d.remove(&k)`                    |
| `d.length`                  | `d.len()`                         |
| `d.isEmpty()`               | `d.is_empty()`                    |
| `d.map((k,v): expr)`        | `d.iter().map(\|(k,v)\| expr).collect()` |
| `d.filter((k,v): cond)`     | `d.into_iter().filter(\|(k,v)\| cond).collect()` |

### Sets — `{T}`

```boring
let {int} unique = {1, 2, 3, 2, 1}   # deduplicates → {1, 2, 3}
let {int} empty_set = {}
```

**Rust equivalent**
```rust
let unique: HashSet<isize> = HashSet::from([1, 2, 3]);
let empty_set: HashSet<isize> = HashSet::new();
```

`{mut T}` (element-`mut` on a set) is rejected outright — not a Boring design choice, a hard Rust limitation: `HashSet<T>` deliberately exposes no mutable element access at all (no `iter_mut()`, no `get_mut()`), because mutating an element in place could change its `Hash`/`Eq` behavior and silently corrupt the set's buckets. `mut {T}` (mutable on the *set itself* — structural `add`/`remove`) is unaffected and works normally.

#### Set methods

| Boring                | Rust                          |
|-----------------------|-------------------------------|
| `s.contains(v)`       | `s.contains(&v)`              |
| `s.add(v)`            | `s.insert(v)`                 |
| `s.remove(v)`         | `s.remove(&v)`                |
| `s.length`            | `s.len()`                     |
| `s.isEmpty()`         | `s.is_empty()`                |

### Index — opaque collection cursor

Boring provides safe iteration over collections through an opaque `Index<T>` type. `firstIndex()` and `nextIndex(idx)` return `nil` when the collection is exhausted — no integer arithmetic needed.

**Array example**
```boring
var nums = [10, 20, 30, 40]
var i = nums.firstIndex()
while let idx = i:
    print "elem: {nums[idx]}"
    i = nums.nextIndex(idx)
nums = nums.removeAt(nums.firstIndex())   # new array without first element
print nums                                 # [20, 30, 40]
```

**Modifying elements with `[]`** — `nums[idx] = value` works for arrays and dicts:
```boring
var nums = [10, 20, 30]
var i = nums.firstIndex()
while let idx = i:
    nums[idx] = nums[idx] * 2             # double each element in place
    i = nums.nextIndex(idx)
print nums                                 # [20, 40, 60]

var d = {"a" = 1, "b" = 2}
var k = d.firstIndex()
while let idx = k:
    d[idx] = d[idx] + 10                  # update each value in place
    k = d.nextIndex(idx)
print d                                    # {"a" = 11, "b" = 12}
```

> Sets are read-only via index — `s[idx]` is not allowed. Use `getAt` to read and `remove` + `add` to modify.

**Set example** — sets cannot be subscripted with `s[idx]`; use `getAt` instead:
```boring
var s = {10, 20, 30}
var j = s.firstIndex()
while let idx = j:
    print "set: {s.getAt(idx)}"
    j = s.nextIndex(idx)
s = s.removeAt(s.firstIndex())
```

**Dict example** — the index is the key:
```boring
var d = {"a" = 1, "b" = 2}
var k = d.firstIndex()
while let idx = k:
    print "{idx} → {d[idx]}"
    k = d.nextIndex(idx)
```

**Optional type annotation**
```boring
var Index<[int]> cursor = nums.firstIndex()
```

**Index method reference**

| Method                    | Array / Set         | Dict                  | Description                          |
|---------------------------|---------------------|-----------------------|--------------------------------------|
| `c.firstIndex()`          | `Option<usize>`     | `Option<K>`           | First index, or `nil` if empty       |
| `c.nextIndex(idx)`        | `Option<usize>`     | `Option<K>`           | Next index after `idx`, or `nil`     |
| `c.removeAt(idx)`         | new collection      | new collection        | Copy of collection without `idx`     |
| `c.getAt(idx)`            | element             | value                 | Element at index (required for sets) |

**Rust equivalent** — the transpiler emits `BoringArrayIndex`, `BoringDictIndex`, and `BoringSetIndex` extension traits in the preamble:
```rust
// arrays / sets → Option<usize>; dicts → Option<K>
let mut i = nums.boring_first_index();
while let Some(idx) = i {
    println!("{}", nums[idx]);
    i = nums.boring_next_index(idx);
}
```

---

## 8. Structs

```boring
struct Vec2:
    float x
    float y

    req float len():
        sqrt(x * x + y * y)      # implicit self: x → self.x

    req Vec2 add(Vec2 other):
        Vec2(x = x + other.x, y = y + other.y)

    as string:
        "({x}, {y})"
```

Explicit `self.` is always valid too — both forms are equivalent:

```boring
    req float len():
        sqrt(self.x * self.x + self.y * self.y)   # same as above
```

**Rust equivalent**
```rust
struct Vec2 { x: f64, y: f64 }

impl Vec2 {
    fn len(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    fn add(&self, other: Vec2) -> Vec2 {
        Vec2 { x: self.x + other.x, y: self.y + other.y }
    }
}
impl std::fmt::Display for Vec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
```

### Empty structs — `pass`

A struct with no fields and no methods uses `pass` as a placeholder body, either inline or indented — both are equivalent:

```boring
struct Empty: pass

struct AlsoEmpty:
    pass

let e = Empty()
```

**Rust equivalent**
```rust
struct Empty;
```

### Constructors

Without an explicit `init`, structs are constructed with named fields:

```boring
let v = Vec2(x = 3.0, y = 4.0)
```

**Rust equivalent**
```rust
let v = Vec2 { x: 3.0, y: 4.0 };
```

### `init` — custom constructors

`init` defines a constructor. Without a body, each parameter automatically declares and initialises the matching field. `pub` makes the field public, `var` makes it mutable.

```boring
struct Point:
    init(pub float x, pub float y)

let p = Point(1.0, 2.0)
```

**Rust equivalent**
```rust
struct Point { pub x: f64, pub y: f64 }

impl Point {
    pub fn new(x: f64, y: f64) -> Self { Self { x, y } }
}
let p = Point::new(1.0, 2.0);
```

With a body, `init` parameters are local variables — assign fields explicitly via `self`. `pub` has no meaning here:

```boring
struct Circle:
    pub float x
    pub float y
    pub float radius

    init(float x, float y, float radius = 1.0):
        self.x = x
        self.y = y
        self.radius = radius

let c = Circle(0.0, 0.0)        # radius defaults to 1.0
let d = Circle(1.0, 2.0, 5.0)
```

**Rust equivalent**
```rust
impl Circle {
    pub fn new(x: f64, y: f64, radius: f64) -> Self {
        Self { x, y, radius }
    }
}
```

### Mutable fields

Fields get the same four-way as local bindings ([§2](#2-variables-and-mutability)) — `var` (reassignment, `self.field = x`) and `mut` (content mutation, `self.field.method()`) are independent axes, not one flag:

| Field | Reassignable (`self.field = x`) | Content-mutable (`self.field.method()`) |
|---|---|---|
| `let Point p` (default, no keyword) | no | no |
| `mut Point p` (≡ `let mut Point p`) | no | yes |
| `var Point p` | yes | no |
| `var mut Point p` | yes | yes |

```boring
struct Inner:
    var int n = 0
    def bump(): n += 1

struct Outer:
    mut Inner a       # content-mutable only
    var mut Inner b    # both
    let Inner c         # neither

mut o = Outer(a = Inner(), b = Inner(), c = Inner())
o.a.bump()     # ok — a's type is mut Inner
o.b.bump()     # ok — b's type is mut Inner too (var mut includes mut)
o.b = Inner()  # ok — b is reassignable (var)
# o.a = Inner() # ERROR — a isn't reassignable
# o.c.bump()    # ERROR — c grants neither
```

A field's *own* `mut`/`var` only matters once the containing binding (`o`
above) already permits reaching it at all — `o` itself still needs `mut`/`var
mut` to read or write **any** field this way, exactly like calling a method
on `o` directly would; a plain `let o` or `var o` (without the second `mut`)
blocks every field access above regardless of each field's own keyword.

Scalar (primitive) fields work the same as scalar local bindings: `mut` on
one is a checker error (nothing to unlock), so a plain non-struct field only
ever needs `var` if it should be reassignable.

### `req` vs `def`

`req` ("request") declares a **read-only** method — it does not mutate the object and can be called on any receiver. `def` declares a **mutating** method — it may modify the object and can only be called on a receiver whose *type* carries `mut` (`mut Type`/`var mut Type` — plain `var Type` is **not** enough on its own).

| Boring keyword | Mutates `self` | Callable on   | Rust receiver  |
|----------------|----------------|---------------|----------------|
| `req`          | no             | any binding | `&self`        |
| `def`          | yes            | `mut`/`var mut` only | `&mut self`    |

```boring
struct Counter:
    var int value = 0

    req int get():        # read-only — &self
        self.value

    def inc():            # mutating — &mut self
        self.value += 1

let c = Counter()
c.get()    # ok — req on any binding
# c.inc()  # error — def requires the type to carry mut

var mc = Counter()
# mc.inc()   # error — var alone does not grant content mutation

var mut mc2 = Counter()
mc2.inc()   # ok — var mut grants both
mc2.get()   # ok — req works on any binding
```

**`req` and `def` for top-level functions**

Both keywords are also valid for top-level free functions (outside any struct). At the top level there is no `self`, so both produce the same `fn` in Rust — the distinction is **documentation intent**:

```boring
req bool positive(int n): n > 0        # pure, no side effects — use req
req string upper(string s): s.upper()  # pure — use req

def log(string msg):                   # has side effects — use def
    print "[LOG] {msg}"
```

> **Summary** — all three forms are equivalent when a return type is present:
> ```boring
> int f(int n): n * 2       # shortest — def implied
> def int f(int n): n * 2   # explicit def
> req int f(int n): n * 2   # pure — signals no side effects
> ```

### Implicit `self`

Inside a method body, bare field names automatically resolve to `self.field` — the explicit `self.` prefix is optional:

```boring
struct Point:
    float x
    float y

    req float distance(Point other):
        let dx = x - other.x    # x  → self.x
        let dy = y - other.y    # y  → self.y
        sqrt(dx * dx + dy * dy)

    req string label():
        "({x}, {y})"            # interpolation works too
```

Writes work the same way — assigning to a bare name that matches a field updates `self.field`:

```boring
struct Counter:
    var int value = 0

    def inc():
        value = value + 1       # reads and writes self.value

    def add(int n):
        value = value + n
```

**Local variables shadow fields.** If a local is declared with the same name as a field, the local takes precedence — `self.field` is still available with the explicit prefix:

```boring
struct Box:
    int width
    int height

    req int scaled(int factor):
        let width = factor * 2  # local — shadows self.width
        width + height          # local width, self.height
```

### Properties — `req` getter + `set` setter

A computed property pairs a `req` getter (no parentheses when called) with a `set` setter:

```boring
struct Temperature:
    var float celsius

    req float fahrenheit():
        self.celsius * 9.0 / 5.0 + 32.0

    set fahrenheit(float f):
        self.celsius = (f - 32.0) * 5.0 / 9.0

var mut t = Temperature(celsius = 0.0)   # a setter call content-mutates — needs mut/var mut
print "{t.fahrenheit}"         # 32.0  — getter via req
t.fahrenheit = 212.0           # setter via set
print "{t.celsius}"            # 100.0
```

**Rust equivalent**
```rust
impl Temperature {
    fn fahrenheit(&self) -> f64 {
        self.celsius * 9.0 / 5.0 + 32.0
    }
    fn set_fahrenheit(&mut self, f: f64) {
        self.celsius = (f - 32.0) * 5.0 / 9.0;
    }
}
// t.fahrenheit()        — getter
// t.set_fahrenheit(212.0) — setter
```

`set` can also be declared `pub` and supports `throws` / `task`:

```boring
pub set name(string value) throws:
    guard value.length > 0 else throw "name cannot be empty"
    self._name = value
```

### Type-level members — `type`

`type` methods and variables belong to the struct itself, not to any instance. They are accessed via `TypeName.member`.

```boring
struct Counter:
    int value

    type let int MAX = 100          # immutable constant (private by default)
    pub type var int count = 0      # mutable, shared across all instances

    pub type def Counter zero():    # factory method
        Counter(value = 0)

    pub type req int max_val():     # read-only associated function
        Counter.MAX

    pub type set count(int v):      # setter with validation logic
        if v >= 0: Counter.count = v
```

```boring
Counter.MAX              # 100
Counter.count            # 0
Counter.count = 42       # direct assignment (calls type set if one exists)
let c = Counter.zero()   # factory method
Counter.max_val()        # 100
```

| Boring | Rust |
|--------|------|
| `type let T NAME = v` | `const NAME: T = v` inside `impl` |
| `type var T name = v` | `static NAME: Mutex<T> = Mutex::new(v)` |
| `type def/req ret f()` | `fn f() -> ret` (no `self`) inside `impl` |
| `type set name(T v):` | `fn set_name(v: T)` (no `self`) inside `impl` |
| `Type.method()` | `Type::method()` |
| `Type.CONST` | `Type::CONST` |

### `as string:`

Defines how a struct is converted to a string (implements `Display`):

```boring
as string:
    "({self.x}, {self.y})"
```

**Rust equivalent**
```rust
impl std::fmt::Display for Vec2 { ... }
```

### `as TraitName:` — inline conformance

See [Traits and Extensions](#10-traits-and-extensions).

### Composition and implicit conversion — `as Type:`

Boring does not support struct inheritance. Use composition (embed the other struct as a named field) and define `as Type:` for an implicit, user-controlled conversion.

```boring
struct Animal:
    init(pub string name)
    var string sound = "..."
    req string speak(): self.sound
    set sound(string s): self.sound = s

struct Dog:
    init(pub Animal base, pub string breed)
    as Animal:          # conversion body — produces an Animal from a Dog
        self.base
    req string describe(): self.base.name + " (" + self.breed + ")"

var d = Dog(base = Animal(name = "Rex"), breed = "Labrador")
let a = d as Animal                 # explicit cast — calls the as Animal: body
let Animal b = d                    # implicit — type annotation triggers conversion
string greet(Animal a): "Hello " + a.name
print greet(d)                      # implicit — argument coerced at call site
```

The conversion is applied automatically whenever the expected type is known:
- explicit cast: `d as Animal`
- typed let: `let Animal b = d`
- function argument: `greet(d)` where `greet` expects `Animal`

#### Mutable access to the inner struct

When the conversion body is a bare field access (`self.base`), the transpiler also generates a `_mut` variant that returns a mutable reference to the inner struct. This allows calling mutating methods directly:

```boring
d.into_animal_mut().set_sound("woof")   # mutates d.base.sound in place
```

**Rust equivalent**
```rust
impl Dog {
    fn into_animal(&self) -> Animal { self.base.clone() }        // for casts
    fn into_animal_mut(&mut self) -> &mut Animal { &mut self.base } // for mutation
    fn describe(&self) -> Arc<str> { ... }
}
// Rust doesn't coerce automatically — call .into_animal() explicitly
greet(d.into_animal());
```

The `_mut` variant is only generated when the body is exactly `self.field`. For computed bodies (constructed values, string conversions, etc.) only the immutable method is emitted — returning `&mut T` on a temporary would be invalid Rust.

The `as Type:` body is a single expression that produces a value of the target type. Method dispatch is always on the concrete type — there is no dynamic dispatch between structs.

### Anonymous call operator — `def ()` / `req ()`

A struct can be made callable by defining a method with an empty name. Use `req ()` for a read-only call (callable on `let` and `var` bindings), or `def ()` for a mutating call (callable on `var` bindings only).

```boring
struct Adder:
    int base
    req int ():          # callable on let, returns int
        base + 10

let a = Adder(base= 5)
print a()                # 15
```

```boring
struct Counter:
    var int value = 0
    def ():              # callable on var, mutates state
        value += 1

var c = Counter()
c()
c()
c()
print c.value            # 3
```

**Rust equivalent**
```rust
impl Adder {
    fn __call__(&self) -> isize { self.base + 10 }
}
impl Counter {
    fn __call__(&mut self) { self.value += 1; }
}
```

`obj()` at a call site dispatches to `__call__` when `obj` is a known struct instance. Calling a struct that has no `def ()`/`req ()` defined is a runtime error.

---

## 9. Enums and Pattern Matching

```boring
enum Expr:
    Num(int value)
    Add(int left, int right)
    Mul(int left, int right)
```

**Rust equivalent**
```rust
#[derive(Clone)]
enum Expr {
    Num(isize),
    Add(isize, isize),
    Mul(isize, isize),
}
```

### Constructing variants

```boring
let e = Expr.Num(7)
let a = Expr.Add(3, 4)
```

**Rust equivalent**
```rust
let e = Expr::Num(7);
let a = Expr::Add(3, 4);
```

### Matching on enums

```boring
int eval(Expr e):
    match e:
        Num(v):      v
        Add(l, r):   l + r
        Mul(l, r):   l * r
```

**Rust equivalent**
```rust
fn eval(e: Expr) -> isize {
    match e {
        Expr::Num(v)      => v,
        Expr::Add(l, r)   => l + r,
        Expr::Mul(l, r)   => l * r,
    }
}
```

### Match guards

```boring
match e:
    Num(v) if v == 0: "zero"
    Num(_):           "literal"
    Add(_, _):        "addition"
    _:                "other"
```

> **Inline form** — for simple cases that fit on one line, use `match expr with Pat: val, ...:
> ```boring
> let kind = match e with Num(_): "number", _: "operator"
> ```
> See [Inline match — `match … with`](#inline-match--match--with) for details.

### Struct destructuring in `match`

`match` works on struct values just like on enum variants. List the field bindings positionally inside parentheses:

```boring
struct Point:
    init(pub float x, pub float y)

string describe_point(Point p):
    match p:
        Point(0.0, 0.0): "origin"
        Point(x, 0.0):   "on x-axis at {x}"
        Point(0.0, y):   "on y-axis at {y}"
        Point(x, y):     "({x}, {y})"

print describe_point(Point(0.0, 0.0))   # origin
print describe_point(Point(3.0, 0.0))   # on x-axis at 3
print describe_point(Point(1.0, 2.0))   # (1, 2)
```

Guards work on destructured struct fields too:

```boring
match p:
    Point(x, y) if x == y: "on diagonal"
    Point(x, y):            "off diagonal ({x}, {y})"
```

**Rust equivalent**
```rust
match p {
    Point { x, y } if x == y => "on diagonal",
    Point { x, y }           => format!("off diagonal ({}, {})", x, y),
}
```

### Enum variant shorthand

When the expected type is known from context — a typed function parameter or an explicit `let T` annotation — you can write `.Variant` instead of `Enum.Variant`:

```boring
enum Direction:
    North
    South
    East
    West

string label(Direction d):
    match d:
        North: "north"
        South: "south"
        East:  "east"
        West:  "west"

# Full form
print label(Direction.North)

# Shorthand — type inferred from the parameter annotation
print label(.North)
print label(.South)
```

The dot signals that `North` is a variant of whichever enum is expected at that position. It works in assignments too:

```boring
let Direction dir = .East     # same as Direction.East
```

**Rust equivalent**
```rust
// No direct equivalent — Rust requires the full path Direction::North.
// The transpiler resolves .North to Direction::North using the type context.
```

### Enum methods — `req` and `def`

Enums can declare methods with the same `req`/`def` keywords as structs ([§8](#8-structs), [`req` vs `def`](#req-vs-def)) — inside the body, after the variants:

```boring
enum EColor:
    Red
    Green
    Blue

    def string ename():        # mutating keyword, but nothing to mutate
        match self:
            Red: "red"
            Green: "green"
            Blue: "blue"

enum EDirection:
    North
    South
    East
    West

    req string label():        # read-only getter
        match self:
            North: "nord"
            South: "sud"
            East: "est"
            _: "ouest"

let ec = EColor.Red
print ec.ename()          # "red" — def works on a plain let binding

let ed = EDirection.South
print ed.label             # "sud" — req getter, no parens needed
```

By default `def` and `req` are **interchangeable** on an enum: `EColor` and
`EDirection` above have no variant field declared `mut` (see the next
section), so there is no `self.field = x` for `def` to unlock, and both
keywords transpile to `&self`. Neither requires the receiver's type to carry
`mut`/`var mut` — `ec` and `ed` above are plain `let` bindings and both calls
compile and run. The choice between them is documentation intent only,
exactly like `req`/`def` on top-level free functions ([§8](#8-structs)): use
`req` for a pure accessor (`label`), `def` if the method reads as an action
even though it can't actually mutate (`ename`). A future compiler version may
tighten this and reject `def` on such enums outright, since it currently
promises a mutation guarantee it cannot enforce there — prefer `req` in new
code unless mirroring existing `def` usage. (An enum with a `mut`-qualified
variant field, next section, is the one case where this interchangeability
*doesn't* hold — `def` there is real.)

**Rust equivalent**
```rust
impl EColor {
    fn ename(&self) -> Rc<str> {
        match self {
            EColor::Red => "red".into(),
            EColor::Green => "green".into(),
            EColor::Blue => "blue".into(),
        }
    }
}
```

### Enum variant fields — `mut Type`

A variant field can itself carry `mut`, the same modifier a tuple slot,
struct field, or collection element can ([mut-type-modifier.md](mut-type-modifier.md)).
This is the one case where an enum's own `def` method is a *real* mutation,
not just documentation intent — it needs (and gets) a genuine `&mut self`,
and the enum instance itself needs `mut`/`var mut` to call it, exactly like a
struct:

```boring
struct Point:
    var int x = 0

    def bump(): x += 1
    req int getx(): x

enum Holder:
    Value(mut Point p)     # this field grants content mutation

    def bumpit():
        match self:
            Value(p):
                p.bump()    # `p` is bound `&mut Point` here — no `mut`/`ref mut`
                            # needed on the pattern itself (see below)

mut h = Holder.Value(Point(x = 5))
h.bumpit()
match h:
    Value(p): print p.getx()   # 6 — the mutation is real, not a disconnected copy

# let h2 = Holder.Value(Point(x = 0))
# h2.bumpit()   # ERROR — h2 is not declared `mut`, same rule as a struct

# enum Holder2:
#     Value(Point p)        # no `mut` on the field
#     def bumpit(): ...      # back to the previous section: def == req, `&self`
```

Matching bare `self` inside a `&mut self` method always matches a reference
(`&mut Self` in the generated Rust) — Rust's own match ergonomics then bind
`p` as `&mut Point` automatically, with no `mut p`/`ref mut p` annotation
needed or accepted (writing `mut p` there is a hard compile error: "cannot
mutably bind by value within an implicitly-borrowing pattern"). This only
applies to matching `self` itself; matching a plain owned local of enum type
elsewhere still needs the usual `mut` promotion to call a `def` method
through a bound field, same as any owned match subject.

Whether `def` gets `&mut self` is decided **per enum type**, not per method
body: if the enum has a `mut`-qualified field anywhere, every `def` method on
it gets `&mut self`, even one that never touches that field. Use `req` for a
method that only reads, to avoid requiring a `mut` receiver unnecessarily.

**Rust equivalent**
```rust
enum Holder {
    Value(Point),
}
impl Holder {
    fn bumpit(&mut self) {
        match self {
            Holder::Value(p) => { p.bump(); }
        }
    }
}
```

### Leading-dot static method calls

The same shorthand works for **static method calls** on external types when the expected
type is known. Instead of writing the full `TypeName.method(args)`, use `.method(args)`:

```boring
# Full form
wait(Duration.fromSecs(5))
wait(Duration.fromMillis(500))
timeout(Duration.fromSecs(10)): fetch(url)

# Shorthand — type inferred from the parameter annotation
wait(.fromSecs(5))
wait(.fromMillis(500))
timeout(.fromSecs(10)): fetch(url)
```

Works wherever the expected type is unambiguous:

```boring
void schedule(Duration delay): ...

schedule(.fromSecs(1))           # Duration inferred from parameter type
schedule(.fromMillis(100))

task(.fromSecs(5)): heavy()      # Duration inferred from task(Duration)
```

The transpiler applies `camelCase → snake_case` conversion automatically:

| Boring | Rust |
|---|---|
| `.fromSecs(5)` | `Duration::from_secs(5)` |
| `.fromMillis(100)` | `Duration::from_millis(100)` |
| `.fromMicros(500)` | `Duration::from_micros(500)` |
| `.North` | `Direction::North` |
| `.Expired` | `Error::Expired` |

> **Limitation** — the shorthand requires a single unambiguous type at the call site.
> When the type cannot be inferred (no annotation, overloaded parameter), use the full form.

### Inline match — `match … with`

When a match fits on one line, use the `with` form instead of an indented block:

```boring
let s = match x with 0: "zero", 1: "one", _: "other"
```

Arms are separated by `,`. Each arm is `Pattern: expression` — the same syntax as block match arms, just without the newline and indent.

```boring
# Block form (multi-line)
match status:
    200: "ok"
    404: "not found"
    _:   "error"

# Inline form (single line)
let msg = match status with 200: "ok", 404: "not found", _: "error"
```

Useful inside expressions, closures, and function arguments:

```boring
scores.map((s): match s with 0: "fail", 100: "perfect", _: "pass")
```

**Rules:**
- `with` replaces the `:` + newline + indent of the block form.
- Arm bodies are expressions (no block statements).
- Guards (`if cond`) and pattern alternatives (`|`) are not supported in the inline form — use the block form for those.

**Rust equivalent** — the transpiler emits a standard `match` block:
```rust
let msg = match status {
    200 => "ok",
    404 => "not found",
    _   => "error",
};
```

---

## 10. Traits and Extensions

### Defining a trait

```boring
trait Named:
    req string name()

trait Describable:
    req string describe()
```

**Rust equivalent**
```rust
trait Named       { fn name(&self)     -> Arc<str>; }
trait Describable { fn describe(&self) -> Arc<str>; }
```

### Type-level methods in traits

A trait can require associated functions (no `self`) using the same `type def` / `type req` syntax:

```boring
trait Factory:
    type def Self create()        # must return an instance of Self
    type req string type_name()   # read-only, no instance needed
```

Implementing structs provide matching `type def` / `type req` members:

```boring
struct Dog as Factory:
    string name

    pub type def Dog create():
        Dog(name = "Rex")

    pub type req string type_name():
        "Dog"
```

```boring
let d = Dog.create()    # Dog.type_name() = "Dog"
```

**Rust equivalent**
```rust
trait Factory {
    fn create() -> Self;
    fn type_name() -> Arc<str>;
}
impl Dog {
    pub fn create() -> Dog { Dog { name: Arc::from("Rex") } }
    pub fn type_name() -> Arc<str> { Arc::from("Dog") }
}
```

### Declaring trait conformance — header or `ext`

Traits can be claimed in the struct header (`struct Foo as Trait1, Trait2:`) and their methods implemented directly in the struct body:

```boring
struct Animal as Named, Describable:
    string species
    string sound

    req string name():
        self.species

    req string describe():
        "{self.species} says {self.sound}"
```

Alternatively, use an `ext` block to attach conformance to a type that is already defined:

```boring
struct Animal:
    string species
    string sound

ext Animal as Named:
    req string name():
        self.species

ext Animal as Describable:
    req string describe():
        "{self.species} says {self.sound}"
```

Both forms produce the same Rust:

**Rust equivalent**
```rust
impl Named for Animal {
    fn name(&self) -> Arc<str> { self.species.clone() }
}
impl Describable for Animal {
    fn describe(&self) -> Arc<str> {
        Arc::<str>::from(format!("{} says {}", self.species, self.sound))
    }
}
```

### Adding methods to an existing type — `ext`

```boring
ext Animal:
    req bool louder_than(Animal other):
        self.sound.length > other.sound.length
```

**Rust equivalent**
```rust
impl Animal {
    fn louder_than(&self, other: &Animal) -> bool {
        self.sound.len() > other.sound.len()
    }
}
```

A single method can also be declared at the top level using the qualified form `TypeName.method()`, which is exactly equivalent to wrapping it in an `ext` block:

```boring
req bool Animal.louder_than(Animal other):
    self.sound.length > other.sound.length

# equivalent to:
ext Animal:
    req bool louder_than(Animal other):
        self.sound.length > other.sound.length
```

All qualifiers (`def`, `req`, `set`, `task`) are accepted in the qualified form.

### Implementing a trait for an existing type — `ext … as Trait:`

```boring
trait Greetable:
    req string greet()

ext Animal as Greetable:
    req string greet():
        "Hi, I'm a {self.species}!"
```

**Rust equivalent**
```rust
impl Greetable for Animal {
    fn greet(&self) -> Arc<str> {
        Arc::<str>::from(format!("Hi, I'm a {}!", self.species))
    }
}
```

### Supertraits — `trait B as A:`

A trait can require another trait using `as`. Any type implementing `B` must also implement `A`:

```boring
trait Named:
    req string name()

# Describable requires Named — implementing types must satisfy both
trait Describable as Named:
    req string describe():           # default implementation can call name()
        "{self.name()} (no description)"

struct Animal as Describable:        # satisfies both Named and Describable
    string species
    req string name(): self.species
    req string describe(): "{self.species} says meow"
```

**Rust equivalent**
```rust
trait Named { fn name(&self) -> Arc<str>; }

trait Describable: Named {           // supertrait
    fn describe(&self) -> Arc<str> {
        Arc::<str>::from(format!("{} (no description)", self.name()))
    }
}
```

Multiple supertraits: `trait C as A, B:`.

---

### Default method implementations

A trait can provide a default body for any method. Implementing types inherit it automatically when they don't override it.

```boring
trait Greetable:
    req string name()

    req string greet():   # default — uses name()
        "Hello, I'm {self.name()}!"

struct Dog as Greetable:
    string name

    req string name():
        self.name

    # greet() not overridden → uses the default
```

Overriding the default:

```boring
struct Robot as Greetable:
    string id

    req string name():
        self.id

    req string greet():         # custom override
        "Beep boop, I am {self.id}."
```

Using `ext` when the type is defined separately:

```boring
struct Cat:
    string name

ext Cat as Greetable:
    req string name():
        self.name
    # greet() inherited from trait default
```

**Rust equivalent**
```rust
trait Greetable {
    fn name(&self) -> Arc<str>;
    fn greet(&self) -> Arc<str> {
        Arc::<str>::from(format!("Hello, I'm {}!", self.name()))
    }
}
```

### Associated types

A trait can declare an *associated type* — a type placeholder that each implementing struct fills in concretely.

```boring
trait Container:
    type Item               # abstract — each struct provides a concrete type

    req Item first()
    req int  count()
```

The implementing struct provides the concrete type with `type Item = …`:

```boring
struct IntBox:
    [int] values

    type Item = int         # concrete definition

    as Container:
        req Item first():
            self.values[0]

        req int count():
            self.values.length
```

Inside the trait body `Item` is shorthand for `Self.Item` — both spellings are accepted:

```boring
trait Transformer:
    type Input
    type Output

    req Self.Output transform(Self.Input value)
```

**Rust equivalent**
```rust
trait Container {
    type Item;
    fn first(&self) -> Self::Item;
    fn count(&self) -> isize;
}
impl Container for IntBox {
    type Item = isize;
    fn first(&self) -> isize { self.values[0].clone() }
    fn count(&self) -> isize { self.values.len() as isize }
}
```

### Generic Associated Types (GAT)

An associated type can itself be parameterised by a lifetime, forming a *Generic Associated Type* (GAT). Write the lifetime parameter as `&a` inside angle brackets after the type name.

```boring
trait Producer:
    type Item<&a>               # GAT — parameterised by lifetime 'a

    req Item<&a> next()

struct Counter:
    init(pub int value)

    type Item<&a> = int         # concrete definition at struct body level

ext Counter as Producer:
    req Item<&a> next():
        self.value + 1

let c = Counter(value = 10)
print c.next()                  # 11
```

**Rust equivalent**
```rust
trait Producer {
    type Item<'a>;
    fn next<'a>(&'a self) -> Self::Item<'a>;
}
impl Producer for Counter {
    type Item<'a> = isize;
    fn next<'a>(&'a self) -> isize { self.value + 1 }
}
```

### Traits as types

A trait name used as a return or parameter type means **dynamic dispatch** — the concrete type is selected at runtime via a vtable. The value is heap-allocated (`Box<dyn Trait>`):

```boring
trait Drawable:
    req void draw()

req Drawable clone_shape()         # → Box<dyn Drawable>  (heap, dynamic dispatch)
[Drawable] shapes                  # → Vec<Box<dyn Drawable>>
```

**Rust equivalent**
```rust
fn clone_shape(&self) -> Box<dyn Drawable> { … }
shapes: Vec<Box<dyn Drawable>>,
```

For **static dispatch** (no heap allocation, function determines the concrete type), use the `<Trait>` shorthand — see [Generics — trait shorthand](#trait-shorthand----trait).

---

## 11. Error Handling

### Standard error enum — `Error`

Boring provides a built-in `Error` enum that is always available without any import. Use it instead of string errors for common failure conditions.

```boring
throw Error.Expired       # timeout expired
throw Error.Cancelled     # task was cancelled
throw Error.NotFound      # element not found
throw Error.InvalidInput  # bad argument
throw Error.OutOfBounds   # index out of range
```

Both styles below are equivalent — choose whichever you prefer:

**Style 1 — `catch` per variant (concise):**

```boring
try:
    let data = timeout(.fromSecs(5), fetch())
catch Error.Expired:
    print "timed out"
catch Error.Cancelled:
    print "task cancelled"
```

Each `catch Error.Variant:` clause handles exactly one variant.
Unlisted variants are **re-thrown** automatically to the caller.

**Style 2 — global `catch` + `match error:` (flexible):**

```boring
try:
    let data = timeout(.fromSecs(5), fetch())
catch Error:
    match error:
        Error.Expired:    print "timed out"
        Error.Cancelled:  print "task cancelled"
        _:                print "other error: {error}"
```

`catch Error:` binds `error` as a typed `Error` value.  
`{error}` in interpolation calls `Display` (e.g. `"timeout expired"`).  
`match error: Error.Variant:` dispatches on variants — `_:` catches the rest.

**With a final `catch:` for other error types:**

```boring
try:
    process()
catch Error.InvalidInput:
    print "invalid input"
catch Error.NotFound:
    print "resource not found"
catch:
    print "unexpected error: {error}"   # catches everything else
```

**Rust equivalent**
```rust
#[derive(Debug, Clone)]
enum Error { Expired, Cancelled, NotFound, InvalidInput, OutOfBounds }
impl std::fmt::Display for Error { … }
impl std::error::Error for Error {}
```

| Variant | Thrown by | Meaning |
|---------|-----------|---------|
| `Error.Expired` | `timeout()` | Timer fired before the future completed |
| `Error.Cancelled` | `timeout()` in cancellable fn | Enclosing task was cancelled |
| `Error.NotFound` | user code | Key, file, or element not found |
| `Error.InvalidInput` | user code | Argument failed validation |
| `Error.OutOfBounds` | user code | Index outside valid range |

---

### Throwing functions

Annotate a function with `throws` to indicate it can fail:

```boring
int divide(int a, int b) throws:
    guard b != 0 else throw "division by zero"
    a / b
```

### `try … else` — expression with fallback

`try … else` is an **expression** that calls a throwing function (or runs a block of statements) and returns a fallback value if an exception is raised.
Unlike `try … catch:` (which is a statement), it always produces a value.

The variable **`error`** is automatically bound in the `else` branch to the **original thrown value** — the same value that was passed to `throw`. Its type is preserved:
- `throw "msg"` → `error` is a `string`
- `throw MyEnum.Variant` → `error` is `MyEnum.Variant` (matchable with `match`)
- `throw 42` → `error` is an `int`

String interpolation **`{error}`** always works regardless of the thrown type, because every Boring value has a Display representation.

All four combinations of inline / block bodies are valid:

```boring
# 1. inline try, inline else
let r1 = try divide(10, 0) else -1

# 2. inline try, block else  (error available)
let r2 = try divide(10, 0) else:
    print "failed: " + error
    -1

# 3. block try, inline else
let r3 = try:
    let x = compute()
    transform(x)
else -1

# 4. block try, block else — match on typed error
enum AppError:
    NotFound
    Timeout
    InvalidInput(string)

risky(int n) throws AppError:
    if n == 1: throw AppError.NotFound
    if n == 2: throw AppError.InvalidInput("bad value")

let r4 = try risky(1) else:
    match error:
        AppError.NotFound:          "not found"
        AppError.InvalidInput(msg): "invalid: {msg}"
        _:                          "other: {error}"
print r4   # not found
```

| Form | Try body | Else body | `error` bound |
|------|----------|-----------|---------------|
| `try expr else expr` | expression | expression | ✓ original thrown value |
| `try expr else: block` | expression | block | ✓ original thrown value |
| `try: block else expr` | block | expression | ✓ original thrown value |
| `try: block else: block` | block | block | ✓ original thrown value |

The last expression in each branch is the value of the whole expression.
Intermediate statements in a block try body propagate exceptions automatically — any throwing call that fails triggers the else branch.

> **`error` binding summary** — consistent across all error-handling forms:
> - `try … else` → `error` is the **original thrown value** (any type, matchable)
> - `catch:` → `error` is the **original thrown value** (same — `match error:` works)
> - `catch String:` → `error: string` — the original string value
> - `catch Int:` → `error: int` — the original integer value
> - `catch Float:` → `error: float` — the original float value
> - `catch Bool:` → `error: bool` — the original boolean value
> - `catch String, Int:` → desugared to one arm per type — `error: string` in the String arm, `error: int` in the Int arm
> - `catch MyEnum:` → `error: &MyEnum` — typed value, for dispatch

### `try: … catch:` — block form

`catch:` is a **statement** (does not produce a value) and binds `error` to the **original thrown value**, exactly like `try … else`. `match error:` with typed patterns works the same way:

```boring
try:
    risky()
catch:
    match error:
        AppError.NotFound:          print "not found"
        AppError.InvalidInput(msg): print "invalid: {msg}"
        _:                          print "other: {error}"
```

Bare `throw` (re-throw) inside a `catch:` body forwards the original error unchanged:

```boring
try:
    fetch(url)
catch:
    log_error(error)
    throw   # re-throw the original error to the caller
```

### Typed `catch`

Catch only errors of a specific type:

```boring
try:
    throw "something"
catch String:
    print "string error: {error}"
catch:
    print "other error"
```

The `error` variable is automatically bound in each catch block to the **original thrown value** with its native type:
- `catch:` → `error` is the original thrown value — matchable with `match error:`
- `catch String:` → `error: string`  •  `catch Int:` → `error: int`  •  `catch Float:` → `error: float`
- `catch MyEnum:` → `error: &MyEnum` — typed value for variant dispatch
- `catch String, Int:` → desugared to one arm per type — body duplicated with native error type per arm

String interpolation `{error}` always works regardless of the thrown type.

### Variant-level `catch`

For enum errors, catch individual variants with dot notation:

```boring
task string fetch(string url) throws:
    if url == "":
        throw Error.InvalidInput
    if url == "notfound":
        throw Error.NotFound
    return "ok"

try:
    let data = fetch(url)
    print "got: {data}"
catch Error.InvalidInput:
    print "invalid URL"
catch Error.NotFound:
    print "resource missing"
# Error.Expired, Error.Cancelled, etc. are re-thrown if not listed
```

Rules:
- Each `catch Enum.Variant:` clause handles exactly one variant.
- You can list any number of variants for the same enum.
- Variants of the same enum are grouped into a single runtime dispatch.
- Any variant **not** listed is **re-thrown** to the caller (or panics in `main` if uncaught).
- You can still add a plain `catch Enum:` or `catch:` after variant clauses to handle everything else.

### Multi-catch — multiple types, one handler

List types separated by commas to handle them in a single block:

```boring
try:
    throw "oops"
catch String, Int:
    print "string or int: {error}"

try:
    throw 42
catch String, Int:
    print "string or int: {error}"    # same handler, different type
```

`catch String, Int:` desugars to one arm per type — the body is duplicated, and `error` keeps its native type in each arm (`string` in the String arm, `int` in the Int arm). Rust type-checking validates the body against each arm independently:

```boring
# OK — {error} works for all types (Display)
catch String, Int: print "error: {error}"

# OK — each arm handles its type correctly
catch String, Int:
    match error:
        string s: print "string: {s}"
        int n:    print "int: {n}"
```

Any number of types can be combined:

```boring
try:
    do_something()
catch String, Int, Float:
    print "any scalar error: {error}"
catch:
    print "other: {error}"
```

Multiple typed clauses are still matched top-down — the first matching clause runs, the rest are skipped:

```boring
try:
    throw "hello"
catch Int:
    print "int"         # not reached
catch String:
    print "string"      # ← runs
```

### `guard … else throw`

```boring
string read_file(string path) throws:
    guard path.length > 0 else throw "empty path"
    "contents of {path}"
```

### Propagation across nested `throws` functions

A `throw` automatically propagates through every `throws` function in the call stack until it is caught by a `try:` block. No explicit forwarding is required.

```boring
int parse_int(string s) throws:
    guard s.length > 0 else throw "empty string"
    guard let n = (s as int) else throw "not a number: {s}"
    n

int double_parse(string s) throws:
    parse_int(s) * 2          # error from parse_int propagates up automatically

string process(string s) throws:
    let n = double_parse(s)   # error from double_parse propagates up automatically
    "result: {n}"

try:
    print process("42")      # result: 84
    print process("abc")     # throws → propagates through double_parse → process → caught here
catch:
    print "caught: {error}"  # caught: not a number: abc
```

A `throw` inside `parse_int` travels up through `double_parse` and `process` without any extra
syntax in those intermediate functions — they simply need to be marked `throws`.

> For the generated Rust representation (`BoringError`, `?` insertion), see [Advanced — Error handling internals](#advanced--error-handling-internals).

### Typed error — `throws ErrorType`

Append a concrete type name after `throws` to narrow the error from `Box<dyn Error>` to that type.
The type is most naturally an enum, but any type works.

```boring
enum CalcError:
    DivByZero
    Overflow

int checked_divide(int a, int b) throws CalcError:
    guard b != 0 else throw CalcError.DivByZero
    guard (a / b) < 1000000 else throw CalcError.Overflow
    a / b

let r1 = try checked_divide(10, 2) else -1     # 5
let r2 = try checked_divide(10, 0) else -1     # -1  (DivByZero)
```

The error type can also be a **module-qualified path**: `throws io.Error`, `throws db.QueryError` — the dot separator maps to `::` in Rust.

> For the internal Rust representation (`BoringError::Other`, `TypeId`, `BoringError::Str`/`String`) and the full untyped/typed comparison table, see [Advanced — Error handling internals](#advanced--error-handling-internals).

### `try?` — Result to Option

`try? expr` is shorthand for `try expr else nil`. It converts a `throws` function call (or a raw `Result<T, E>`) into an optional value: success becomes the value, any error becomes `nil`.

**With `throw` (idiomatic Boring)**
```boring
int divide(int a, int b) throws:
    if b == 0: throw "division by zero"
    return a / b

let int? r1 = try? divide(10, 2)   # 5
let int? r2 = try? divide(10, 0)   # nil

if let v = r1:
    print "got {v}"                 # got 5
```

**Rust equivalent**
```rust
fn divide(a: isize, b: isize) -> Result<isize, Box<dyn std::error::Error>> {
    if b == 0 { return Err("division by zero".into()); }
    Ok(a / b)
}

let r1 = divide(10, 2).ok();   // Some(5)
let r2 = divide(10, 0).ok();   // None

if let Some(v) = r1 { println!("got {}", v); }
```

**With `Result<T, E>` (Rust interop)**

When calling a function that returns a raw `Result<T, E>` (e.g. from a Rust library), `try?` works the same way:

```boring
Result<int, string> parse_count(string s):
    if s == "": return Err("empty input")
    return Ok(s.len())

let int? n = try? parse_count("hello")   # 5
let int? m = try? parse_count("")        # nil
```

> For a detailed compatibility matrix between Boring's `throw`/`catch` system and Rust's native `Result<T, E>` types, see [Advanced — Compatibility with Rust `Result` types](#advanced--compatibility-with-rust-result-types).

---

## 12. Optionals

An optional type is written `T?`. The absence of a value is `nil`.

```boring
let int? some_val = 42
let int? no_val   = nil
```

**Rust equivalent**
```rust
let some_val: Option<isize> = Some(42);
let no_val:   Option<isize> = None;
```

### `else` — nil coalescing

```boring
print "{some_val else 0}"   # 42
print "{no_val   else 0}"   # 0
```

**Rust equivalent**
```rust
println!("{}", some_val.unwrap_or(0));
println!("{}", no_val.unwrap_or(0));
```

### Optional fields in structs

```boring
struct Config:
    string? host
    int?    port
```

### `if let` — conditional unwrap

```boring
if let host = cfg.host:
    print "host = {host}"
```

### `guard let` — unwrap or early return

```boring
string connect(Config c):
    guard let host = c.host else return "no host"
    guard let port = c.port else return "no port"
    "Connecting to {host}:{port}"
```

### Optional chaining — `?.`

```boring
let name = user?.profile?.name
```

**Rust equivalent**
```rust
let name = user.and_then(|u| u.profile).map(|p| p.name);
```

---

## 13. Generics

### Generic functions

```boring
T identity(T x):
    x

T first([T] items):
    items[0]
```

**Rust equivalent**
```rust
fn identity<T>(x: T) -> T { x }
fn first<T>(items: Vec<T>) -> T { items[0].clone() }
```

### Type parameter inferred from arguments

```boring
identity(42)      # T = int
identity("hello") # T = string
first([10, 20])   # T = int
```

### Generic structs

```boring
struct Pair<T, U>:
    T first
    U second

let p = Pair(10, "hello")
print "{p.first} / {p.second}"   # 10 / hello
```

**Rust equivalent**
```rust
struct Pair<T, U> { first: T, second: U }
```

### Trait constraints — `as`

Use `as` inside `<…>` to require a trait on a type parameter.

```boring
struct Wrapper<T as Display>:
    T item

string describe<T as Display>(T x):
    "value: {x}"

let w = Wrapper(3.14)
print describe(w.item)            # value: 3.14
```

Multiple constraints use `+`, mirroring Rust:

```boring
struct Tagged<T as Display + Eq>:
    T value
    string tag

let t = Tagged(42, "answer")
print "{t.tag} = {t.value}"       # answer = 42
```

**Rust equivalent**
```rust
struct Wrapper<T: Display> { item: T }
fn describe<T: Display>(x: T) -> String { format!("value: {}", x) }
struct Tagged<T: Display + Eq> { value: T, tag: String }
```

### Trait shorthand — `<Trait>`

When a type parameter is constrained by a trait and used **only once**, you can omit the explicit type parameter name and write `<Trait>` directly. This is a specialized generic — the compiler generates a fresh anonymous type parameter.

```boring
# <Drawable> in return position → impl Drawable (static dispatch, no heap allocation)
<Drawable> scale(float factor):
    Circle(radius = self.radius * factor)

# <Drawable> in parameter position → impl Drawable parameter
string describe(<Drawable> shape):
    shape.draw()

# Multiple <Drawable> — each is INDEPENDENT (different concrete types allowed)
<Drawable> transform(<Drawable> other):
    Circle(radius = 1.0)
```

**Rust equivalent**
```rust
fn scale(&self, factor: f64) -> impl Drawable { … }
fn describe(&self, shape: impl Drawable) -> Arc<str> { … }
fn transform(&self, other: impl Drawable) -> impl Drawable { … }
```

When you need the **same concrete type** in multiple positions — or you need to name the type parameter — use the explicit form:

```boring
# T must be the SAME type for parameter and return value
T echo<T as Drawable>(T shape):
    shape

# Two uses of the same type T
bool same_size<T as Drawable>(T a, T b):
    a.width() == b.width()
```

**Full comparison table**

| Boring | Rust | When |
|---|---|---|
| `Drawable f()` | `Box<dyn Drawable>` | Different concrete types at runtime |
| `<Drawable> f()` | `impl Drawable` | Single use, function picks type |
| `<Drawable> f(<Drawable> x)` | `(impl Drawable) -> impl Drawable` | Independent impl params |
| `T f<T as Drawable>(T x)` | `fn<T: Drawable>(T) -> T` | Same type in multiple positions |
| `struct S<T as Drawable>` | `struct S<T: Drawable>` | Generic struct with bound |

> **Rule of thumb:** use `<Trait>` for the common single-use case. Switch to `Trait` (bare) only when you need to collect or store values of *different* concrete types at runtime. Use explicit `T<T as Trait>` when the same type must appear in multiple positions.

### Lifetime and bound arguments at use sites

When *calling* a generic function or annotating a variable, you can supply type arguments that include bare lifetimes and trait bounds for documentation purposes:

```boring
# Definition — constraint is declared here
copy_items<T as Clone>(Container<T> src, Container<T> dst):
    for item in src.items:
        print "copying: {item}"

# Use site — just pass the concrete type, no constraint needed
copy_items<int>(src, dst)

# Variable declaration with lifetime in type argument
var Container<&a, string> buf = Container(items = ["x", "y"])
```

**Rules:**
- `&a` in a type argument position becomes `'a` in the emitted Rust.
- Constraints (`as Bound`) belong only on the **definition** — never repeat them at call sites.

**Rust equivalent**
```rust
copy_items::<isize>(src, dst);           // bound already on fn, not repeated at call site
let buf: Container<'a, Arc<str>>;   // lifetime preserved
```

### Const generics — `<uint N>`

A **const generic** is a compile-time integer or bool value that is part of a type's signature. Write it as `<uint N>` (or `<int N>`, `<bool N>`) inside the generic parameter list — the type comes first, then the name, just like a regular variable declaration.

```boring
struct Stack<T, uint N>:
    [T]  data          # runtime storage
    int  len

    req uint capacity():
        N              # use N as a compile-time constant value

    req bool is_full():
        len == N as int
```

**Rust equivalent**
```rust
struct Stack<T: Clone + std::fmt::Debug, const N: usize> {
    data: Vec<T>,
    len:  isize,
}
impl<T: Clone + std::fmt::Debug, const N: usize> Stack<T, N> {
    fn capacity(&self) -> usize { N }
    fn is_full(&self)  -> bool  { self.len == N as isize }
}
```

Const generics work on functions too. The implicit type-param collection picks up `<uint N>` from parameter types automatically:

```boring
# N is inferred from the param type `Stack<T, uint N>` — no explicit <T, uint N> needed
T get(Stack<T, uint N> s, int i):
    guard i < N as int else throw Error.OutOfBounds
    s.data[i]

uint capacity_of(Stack<T, uint N> s):
    N
```

**Rust equivalent**
```rust
fn get<T: Clone + std::fmt::Debug, const N: usize>(s: &Stack<T, N>, i: isize) -> T {
    if !(i < N as isize) { return Err(…) }
    s.data[i as usize].clone()
}
fn capacity_of<T: Clone + std::fmt::Debug, const N: usize>(s: &Stack<T, N>) -> usize { N }
```

Or declare them explicitly and use `N` directly as a value in the body:

```boring
string describe<T, uint N>(Stack<T, uint N> s):
    "capacity={N}, used={s.len}"
```

| Boring syntax | Rust equivalent | Notes |
|---------------|-----------------|-------|
| `<uint N>` | `const N: usize` | Most common — array lengths, capacities |
| `<int N>` | `const N: isize` | Signed integer constant |
| `<bool B>` | `const B: bool` | Feature flag at compile time |

> `uint` maps to `usize` (not `u64`) for const generics — `usize` is the standard Rust type for sizes and indices.

---

## 14. Closures and Higher-Order Functions

### Inline closure

```boring
let double  = (n): n * 2
let add_one = (n): n + 1
```

**Rust equivalent**
```rust
let double  = |n: isize| n * 2;
let add_one = |n: isize| n + 1;
```

### Block closure

```boring
let classify = (int n):
    if n > 0: "pos"
    elif n < 0: "neg"
    else: "zero"
```

**Rust equivalent**
```rust
let classify = |n: isize| {
    if n > 0 { "pos" } else if n < 0 { "neg" } else { "zero" }
};
```

### Closures as arguments

```boring
numbers.map((n): n * 2)
numbers.filter((n): n % 2 == 0)
```

### Trailing closures

When the last argument of a call is a closure, it can be written **outside the parentheses** — or the parentheses can be omitted entirely when there are no other arguments. The closure follows the call with a space, then the usual `(params): body` syntax.

```boring
# Standard — closure inside parens
numbers.map((n): n * 2)

# Trailing — closure outside parens (no other args)
numbers.map (n): n * 2

# Trailing with prior args
numbers.reduce(0) (acc, n): acc + n
```

**Multi-line trailing closure** — the body is indented as a block:

```boring
let big = [1, 10, 2, 9, 3].filter (n):
    n > 5
# [10, 9]
```

**No-paren single-parameter shorthand** — when the closure takes exactly one parameter and its body is a single expression, the parentheses around the parameter can also be dropped:

```boring
# (n): ... → n: ...
numbers.map n: n * 2
numbers.filter n: n % 2 == 0
```

All three forms produce identical results:

```boring
let words = ["hello", "world", "boring"]

# equivalent — all yield ["HELLO", "WORLD", "BORING"]
let a = words.map((w): w.upper())
let b = words.map (w): w.upper()
let c = words.map(:upper())          # closure shorthand (field/method on arg) — parens required
```

**Rust equivalent**
```rust
numbers.iter().map(|n| n * 2).collect::<Vec<_>>()
numbers.iter().filter(|n| n % 2 == 0).cloned().collect::<Vec<_>>()
```

> **Note** — a multiline trailing closure cannot be immediately chained with `.method()`. Wrap the whole call in parentheses instead:
> ```boring
> # Error — ambiguous where the closure ends
> # list.filter (n):
> #     n > 0
> # .map((n): n * 2)
>
> # OK — explicit parens
> let result = (list.filter (n):
>     n > 0).map((n): n * 2)
> ```

**Zero-arg trailing body** — when the trailing closure takes no parameters, three forms are available:

```boring
# Inside the argument list — explicit zero-arg closure
timeout(.fromSecs(5), (): fetch(url))

# After the argument list — bare colon separator
timeout(.fromSecs(5)): fetch(url)

# Without any separator — command style (same-line only)
timeout(.fromSecs(5)) fetch(url)
```

The three forms are equivalent. Choose the one that reads most naturally:
- `():` inside parens — always unambiguous, works anywhere
- `f(args): body` — clean for single-expression bodies
- `f(args) body` — closest to `task(dur) expr` style; same-line only

Multi-line zero-arg body uses the colon form:

```boring
timeout(.fromSecs(10)):
    let data = download(url)
    parse(data)
```

**Limitation — single-identifier argument ambiguity**

When the content between `()` consists **only of bare identifiers**, the parser cannot tell whether they are call arguments or closure parameters, and resolves them as **closure parameters**:

```boring
timeout(deadline): fetch(url)    # (deadline): is a 1-param closure, NOT a call arg
timeout(dur1, dur2): fetch(url)  # (dur1, dur2): is a 2-param closure
```

The truly unambiguous forms are those where at least one argument contains something that cannot be a closure parameter — an operator, a dot, a literal, or a function call:

```boring
timeout(Instant.now() + dur): fetch(url)   # unambiguous (contains .)
timeout(.fromSecs(5)): fetch(url)          # unambiguous (contains .)
timeout(5): fetch(url)                     # unambiguous (literal)
```

**`do` — unambiguous trailing closure marker**

Use `do` after a call to introduce a trailing closure when the arguments are plain identifiers. `do` is optional — it is only needed when other forms would be ambiguous. It supports all closure shapes:

```boring
# Zero-arg
timeout(deadline) do: fetch(url)       # ← deadline is a call arg, not a closure param
timeout(deadline) do fetch(url)        # same, no separator

# Single param
nums.map do (n): n * 2                 # explicit parens
nums.map do n: n * 2                   # no-paren shorthand

# Multi-param
nums.reduce(0) do (acc, n): acc + n    # explicit parens
nums.reduce(0) do acc, n: acc + n      # no-paren shorthand

# Multiline
timeout(deadline) do:
    let data = download(url)
    parse(data)
```

All existing trailing closure forms remain valid and are preferred when unambiguous. `do` is a deliberate escape hatch for the identifier-only argument case.

### Closures in collections

```boring
let pipeline = [double, add_one, double]
var val = 3
for fn in pipeline:
    val = fn(val)
```

### Closure shorthand

When a closure works on a single argument, three compact forms are available:

| Form | Meaning | Example |
|---|---|---|
| `(p): expr` | standard closure | `(p): p.name` |
| `(p, q): expr` | two-param closure | `(acc, n): acc + n` |
| `(): expr` | zero-arg closure | `(): fetch()` |
| `p: expr` | no-paren single param | `p: p.length > 3` |
| `:member` | implicit param, field/method | `:name`, `:upper()` |
| `:member op value` | implicit param + operator | `:length > 3`, `:age == 18` |

```boring
struct Person:
    string name
    int    age

    req bool is_adult():
        self.age >= 18

let people = [Person(name = "Alice", age = 30), Person(name = "Bob", age = 15)]

# All equivalent forms
let names   = people.map((p): p.name)    # standard
let names2  = people.map(p: p.name)      # no-paren param
let names3  = people.map(:name)          # implicit param

let adults  = people.filter((p): p.age >= 18)   # standard
let adults2 = people.filter(p: p.age >= 18)      # no-paren param
let adults3 = people.filter(:age >= 18)          # implicit param + operator
let adults4 = people.filter(:is_adult())         # implicit param, method
```

The `:` prefix signals an implicit-param closure; what follows is a member access on the argument, optionally continued with a binary operator:

```boring
let words = ["the", "quick", "brown", "fox"]

words.filter(:length > 3)          # keeps words longer than 3 chars
words.map(:upper())                # upper-cases each word
words.filter(:length == 5)         # exact length match
```

Chaining is supported — `:a.b` becomes `(x): x.a.b`:

```boring
struct Address:
    string city

struct Employee:
    Address address

let employees = [Employee(address = Address(city = "Paris"))]
let cities = employees.map(:address.city)    # (e): e.address.city
```

The `:field op expr` form covers only a single operator. For compound conditions, use the named-param form:

```boring
# compound condition — use named param
words.filter(w: w.length > 3 && w.length < 8)
```

**Rust equivalent**
```rust
people.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
people.iter().filter(|p| p.age >= 18).cloned().collect::<Vec<_>>();
words.iter().filter(|__x| __x.len() as isize > 3).cloned().collect::<Vec<_>>();
```

---

## 15. Modules

### Defining a module

```boring
mod math_utils:
    pub float hypot(float a, float b):
        sqrt(a * a + b * b)

    pub float clamp_f(float v, float lo, float hi):
        max(lo, min(v, hi))
```

**Rust equivalent**
```rust
mod math_utils {
    pub fn hypot(a: f64, b: f64) -> f64 { (a*a + b*b).sqrt() }
    pub fn clamp_f(v: f64, lo: f64, hi: f64) -> f64 { lo.max(v.min(hi)) }
}
```

### Importing — `use`

Import all public items from a module:

```boring
use math_utils.*
```

**Rust equivalent**
```rust
use math_utils::*;
```

Import one or several items by name:

```boring
use math_utils.hypot
use math_utils.hypot, clamp_f
```

**Rust equivalent**
```rust
use math_utils::hypot;
use math_utils::{hypot, clamp_f};
```

Rust standard library:

```boring
use std.collections.HashMap, HashSet
use std.io.Write, BufRead
```

**Rust equivalent**
```rust
use std::collections::{HashMap, HashSet};
use std::io::{Write, BufRead};
```

### Type aliases — `use … as`

Rename or qualify a type:

```boring
use NodeRef as Node'shared      # Arc<Node> or Rc<Node> depending on --threading
use Score   as int              # isize
```

**Rust equivalent**
```rust
type NodeRef = Arc<Node>;
type Score   = isize;
```

### Newtype wrappers — `type … as`

A newtype creates a **distinct type** that wraps an existing one. Unlike a type alias, `UserId` and `OrderId` are incompatible even though both wrap `uint` — the compiler rejects accidental mix-ups.

```boring
type UserId  as uint
type OrderId as uint
type Email   as string
```

**Construction** — same syntax as an enum variant or struct call:

```boring
let id    = UserId(42)
let order = OrderId(99)
let email = Email("alice@example.com")
```

**Unwrapping** — use the existing `as` cast syntax:

```boring
let n = id as uint      # → id.0
```

**Display** — the inner value is used automatically in string interpolation:

```boring
print "user {id}"       # prints "user 42"
```

**Rust equivalent**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct UserId(usize);

impl From<usize> for UserId { fn from(v: usize) -> Self { UserId(v) } }
impl From<UserId> for usize { fn from(v: UserId) -> usize { v.0 } }
impl std::fmt::Display for UserId { … }
```

The `Copy` derive is added automatically for numeric inner types (`uint`, `int`, `float`, `bool`). String newtypes use `String` as the inner type and do not derive `Copy`.

Alias a function type — the return type comes first, then the parameter types in parentheses. Optional prefixes: `req` (pure, `Fn`) or `def` (mutating, `FnMut`, the default); `task` (async). Optional suffix: `throws`.

```boring
use Predicate   as req bool(int)              # pure:    impl Fn(isize) -> bool
use Transformer as def int(int)               # mutable: impl FnMut(isize) -> isize
use Parser      as string(string) throws         # impl FnMut(&str) -> Result<…>
use AsyncFetch  as task string(string) throws    # async FnMut → Result<…>
use Worker      as req task int() throws      # pure async: impl Fn() -> Future<…>
```

**Rust equivalent**
```rust
type Predicate  = fn(isize) -> bool;
type Parser     = fn(&str) -> Result<Arc<str>, Box<dyn std::error::Error>>;
// async and throws/task equivalents are function signatures, not type aliases in plain Rust
```

Built-in primitives (`int`, `uint`, `float`, `bool`) are always `Copy` Rust types — they carry no qualifier. `string` literals emit as `&str`; the transpiler promotes them to `Rc<str>` (single-thread) or `Arc<str>` (multi-thread) when they are stored or concatenated. See [Advanced — Strings](#advanced--strings-string-and-arc-str).

---

## 16. Pipe Operator (`|>`)

The pipe operator `|>` chains operations left-to-right, passing the left-hand value as the first argument (or receiver) of the right-hand call. The syntax is inspired by **F#** and **Julia**, where `|>` is the idiomatic way to compose pipelines without nesting parentheses.

```boring
let result = value |> f(extra_args)
```

**Dispatch rules** (resolved at compile time):

| Right-hand side | Boring desugaring | Rust output |
|-----------------|-------------------|-------------|
| Known function  | `f(value, args)`  | `f(value, args)` |
| Method name     | `value.f(args)`   | `value.f(args)` |

### Examples

```boring
int double(int n): n * 2
int inc(int n):    n + 1

# Single-line
let x = 5 |> double() |> inc() |> inc()   # 12

# Multi-line (continuation lines indented)
let y = 5
    |> double()
    |> inc()
# y == 11
```

String methods via pipe:

```boring
let shout = "  hello  "
    |> trim()
    |> upper()
# shout == "HELLO"
```

Mixed function and method dispatch:

```boring
[int] keep_positive([int] arr):
    arr.filter(n: n > 0)

let nums = [-1, 2, -3, 4]
    |> keep_positive()   # function dispatch: keep_positive(nums)
    |> map(n: n * 2)     # method dispatch:  nums.map(...)
    |> reversed()        # method dispatch:  nums.reversed()
```

### Rules

- The RHS of `|>` must be an identifier followed by an optional argument list `(…)`.
- Argument-free calls still require `()`: `value |> double()`.
- Multi-line pipes must indent the continuation lines relative to the start of the expression.

---

## 17. Streams (`stream` / `yield`)

A **stream function** is declared with the `stream` keyword instead of `def` or `task`. It lazily produces a sequence of values with `yield`, and callers consume it with a plain `for` loop — no extra syntax needed at the call site.

```boring
stream int count_up(int n):
    var i = 0
    while i < n:
        yield i
        i += 1

for x in count_up(5):
    print x   # 0  1  2  3  4
```

### Semantics

| Aspect | Behaviour |
|--------|-----------|
| Execution | Lazy — the body runs only as the consumer pulls values |
| Concurrency | Stream bodies are implicitly async; `await` is allowed inside them |
| Consumption | `for item in stream_fn(args):` — no `await` needed at the call site |
| Error handling | Add `throws ErrorType` to propagate errors; consumed with `try`/`catch` |

### Composing streams

Stream functions can consume other streams — just `for` over them inside the body:

```boring
stream string greet_msgs(string name):
    yield "Hello, " + name + "!"
    yield "Goodbye, " + name + "!"

stream string greet_all([string] names):
    for name in names:
        for msg in greet_msgs(name):
            yield msg

for line in greet_all(["Alice", "Bob"]):
    print line
# Hello, Alice!  Goodbye, Alice!  Hello, Bob!  Goodbye, Bob!
```

### Error-yielding streams

```boring
stream string read_lines(string path) throws IOError:
    let file = File.open(path)
    for line in file.lines():
        yield line
```

The `throws` annotation wraps each item in `Result<T, E>`. Callers unwrap with `try`:

```boring
for line in read_lines("/etc/hosts"):
    let text = try line else continue
    print text
```

### Transpilation

The transpiler chooses between two emission strategies depending on whether the stream body is **sequential** or **async**:

| Body contains | Strategy | Rust emitted |
|---------------|----------|--------------|
| No `wait`, no `task` calls | Sequential — `impl Iterator` | Collects all values into a `Vec`, returns `vec.into_iter()` |
| Has `wait` or `task` calls | Async — `impl Stream` | `async_stream::stream! { … }` |

#### Sequential streams — `impl Iterator`

When the body has no `wait` statements and makes no `task` function calls, the transpiler treats it as a plain iterator. The body runs eagerly: every `yield` pushes a value into a `Vec`, and the function returns `vec.into_iter()`. No async runtime is needed.

```boring
stream int count_up(int n):
    var i = 0
    while i < n:
        yield i
        i += 1
```

**Rust equivalent**
```rust
fn count_up(n: isize) -> impl Iterator<Item = isize> {
    let mut __items: Vec<isize> = Vec::new();
    let mut i = 0isize;
    while i < n {
        __items.push(i);
        i += 1;
    }
    __items.into_iter()
}
```

Consumers compile to a plain Rust `for` loop — no `await`, no pinning:

```rust
for x in count_up(5) { … }
```

#### Async streams — `impl Stream`

When the body contains `wait` or calls a `task` function, the transpiler emits an `async_stream::stream!` block:

```boring
stream string fetch_pages([string] urls):
    for url in urls:
        let html = task fetch(url)
        yield html.value
```

**Rust equivalent**
```rust
fn fetch_pages(urls: Vec<Arc<str>>) -> impl futures_core::Stream<Item = Arc<str>> {
    async_stream::stream! {
        for url in urls {
            let html = tokio::spawn(async move { fetch(url).await });
            yield html.await.unwrap();
        }
    }
}
```

Consumers compile to a `while let Some(item) = stream.next().await` loop pinned with `std::pin::pin!`.

#### `throws` streams

Adding `throws` wraps each item in `Result<T, E>` and uses `async_stream::try_stream!`:

```boring
stream string read_lines(string path) throws IOError:
    let file = File.open(path)
    for line in file.lines():
        yield line
```

**Rust equivalent**
```rust
fn read_lines(path: Arc<str>) -> impl futures_core::Stream<Item = Result<String, IOError>> {
    async_stream::try_stream! {
        let file = File::open(&*path)?;
        for line in file.lines() {
            yield line?;
        }
    }
}
```

---

## 18. Channels (`channel`)

Boring provides **typed mpsc channels** via the built-in `channel` function. A channel carries values from one or more senders to a single receiver; both endpoints are obtained from one call.

### Creating a channel

Three equivalent syntaxes — choose whichever reads more naturally:

```boring
# A — explicit type argument, capacity as call argument
let tx, rx = channel<int>(32)

# B — type inferred from the binding annotation
let int tx, rx = channel(32)

# C — capacity as second type parameter
let tx, rx = channel<int, 32>
```

All three forms are fully interchangeable. The transpiler extracts the capacity from whichever position it appears — second type argument or first call argument — and forwards it to the backend. On the **tokio backend** it becomes the argument to `tokio::sync::mpsc::channel(N)`. On the **kernel backend** it sets the size of the pre-allocated ring buffer; see [Rust-for-Linux target](#30-rust-for-linux-target).

#### Capacity in single-thread mode

When the transpiler targets single-thread mode (`--threading single`), `channel` maps to `local_channel::mpsc` instead of tokio mpsc. `local_channel` is **unbounded** — the capacity argument is accepted syntactically but has no effect at runtime. Prefer omitting it to make the intent clear, or leave it as documentation:

```boring
# single-thread mode — capacity is ignored; channel is unbounded
let tx, rx = channel<int>(4)
```

**Transpilation comparison**

| Mode | Boring | Rust emitted |
|------|--------|--------------|
| Multi-thread (default) | `channel<int>(32)` or `channel<int, 32>` | `tokio::sync::mpsc::channel::<isize>(32)` |
| Single-thread | `channel<int>(32)` or `channel<int, 32>` | `local_channel::mpsc::channel::<isize>()` — unbounded |
| Kernel (`channel<T, N>`) | `channel<int, 32>` | `kernel_channel::<isize, 32>()` → `KernelSender/Receiver<isize, 32>` |
| Kernel (`channel<T>(cap)`) | `channel<int>(32)` | `dyn_kernel_channel::<isize>(32)` → `DynKernelSender/Receiver<isize>` |


### Sending and receiving

```boring
# Sender — .send() is async; it blocks until there is room in the buffer
tx.send(42)

# Clone the sender to share it across multiple tasks
let tx2 = tx.clone()

# Receiver — iterate with a plain for loop; exits when all senders are dropped
for n in rx:
    print n
```

### Full example

```boring
let tx, rx = channel<int>(4)

let producer = task:
    tx.send(10)
    tx.send(20)
    tx.send(30)
    drop(tx)          # closing the sender ends the for loop on the receiver side

let consumer = task:
    for n in rx:
        print n       # prints 10, 20, 30

producer.wait
consumer.wait
```

String channels use the binding-annotation form:

```boring
let string tx, rx = channel(8)

let h = task:
    tx.send("hello")
    tx.send("world")
    drop(tx)

for msg in rx:
    print msg
h.wait
```

### Transpilation

```rust
// let tx, rx = channel<int>(4)
let (tx, mut rx) = tokio::sync::mpsc::channel::<isize>(4);

// tx.send(v)
tx.send(v).await.unwrap();

// for n in rx:
while let Some(n) = rx.recv().await { … }
```

Functions that contain a `channel(…)` call or a `task:` expression are automatically marked `#[tokio::main] async`.

### Rules

- The receiver variable is always `mut` (required by `tokio::sync::mpsc::Receiver`).
- Dropping all senders closes the channel — the receiver loop exits cleanly.
- Multiple senders are supported via `tx.clone()`.

### Other channel kinds

Boring provides three additional channel types for specialised patterns:

| Boring | Tokio (multi-thread) | Single-thread | Use case |
|--------|----------------------|---------------|----------|
| `oneshot<T>()` | `tokio::sync::oneshot` | `tokio::sync::oneshot` | Single response — request/reply |
| `broadcast<T>(cap)` | `tokio::sync::broadcast` | `local_broadcast` (`!Send`) | Fan-out to N independent consumers |
| `watch<T>(initial)` | `tokio::sync::watch` | `tokio::sync::watch` | Observable current value |

#### `oneshot<T>()` — single-shot response

A oneshot channel carries exactly one value from sender to receiver. The sender is consumed when `send` is called; the receiver is consumed when the value is read.

```boring
task run():
    let tx, rx = oneshot<int>()
    task: tx.send(42)        # sender consumed on send
    let result = rx.value    # waits for the single value
    print result
```

**Rust equivalent**
```rust
let (tx, rx) = tokio::sync::oneshot::channel::<isize>();
tokio::spawn(async move { tx.send(42).ok(); });
let result = rx.await.unwrap();
```

| Operation | Boring | Rust emitted |
|-----------|--------|--------------|
| Create | `let tx, rx = oneshot<T>()` | `tokio::sync::oneshot::channel::<T>()` |
| Send | `tx.send(v)` | `tx.send(v).ok()` — non-async, consumes `tx` |
| Receive | `rx.value` or `rx.recv()` | `rx.await.unwrap()` — consumes `rx` |

**Rules:**
- `tx.send(v)` is synchronous (non-blocking) — it moves the value into the channel immediately.
- Both endpoints can be sent across threads (`Send`). The same behavior applies in single-thread mode — `tokio::sync::oneshot` is used in both.
- Reading `rx.value` (or calling `rx.recv()`) consumes `rx`; it cannot be used again.

#### `broadcast<T>(cap)` — fan-out

A broadcast channel delivers every sent message to **all current subscribers** independently. Each receiver maintains its own read cursor.

```boring
task run():
    let tx, rx = broadcast<string>(16)
    let rx2 = tx.subscribe()        # second independent consumer

    task:
        tx.send("hello")
        tx.send("world")

    for msg in rx:
        print "rx1: {msg}"
    for msg in rx2:
        print "rx2: {msg}"
```

**Rust equivalent (multi-thread)**
```rust
let (tx, mut rx) = tokio::sync::broadcast::channel::<Arc<str>>(16);
let mut rx2 = tx.subscribe();
// ...
while let Ok(msg) = rx.recv().await { println!("rx1: {}", msg); }
while let Ok(msg) = rx2.recv().await { println!("rx2: {}", msg); }
```

| Operation | Boring | Rust emitted |
|-----------|--------|--------------|
| Create | `let tx, rx = broadcast<T>(cap)` | `tokio::sync::broadcast::channel::<T>(cap)` |
| Add consumer | `let rx2 = tx.subscribe()` | `tx.subscribe()` |
| Send | `tx.send(v)` | `tx.send(v).ok()` — non-async |
| Receive | `rx.recv()` | `rx.recv().await.unwrap()` |
| Iterate | `for msg in rx:` | `while let Ok(msg) = rx.recv().await {` |
| Poll ready | `f.done` | `tokio::time::timeout(Duration::ZERO, rx.recv()).await.is_ok()` |

**Single-thread mode** — In single-thread mode, broadcast maps to an inlined `local_broadcast` primitive (emitted as a prelude in the generated file). It uses `Rc<RefCell<VecDeque<T>>>` per subscriber slot and `tokio::sync::Notify` for wakeups. The only bound on `T` is `Clone`. The API is identical at the Boring level; only the emitted Rust differs:

```rust
// single-thread broadcast — emitted by the transpiler
let tx = local_broadcast::<Arc<str>>();
let mut rx = tx.subscribe();
let mut rx2 = tx.subscribe();
// send
tx.send(Arc::from("hello"));
// receive (returns T directly, not Result<T, _>)
while let Some(msg) = rx.recv().await { … }
```

The capacity argument is accepted but has no effect for `local_broadcast` (the buffer grows dynamically).

#### `watch<T>(initial)` — observable value

A watch channel holds a single *current* value. Senders overwrite it; receivers either read the current value without waiting or wait for the next change.

```boring
task run():
    let tx, rx = watch<int>(0)      # initial value = 0

    task:
        tx.send(1)
        tx.send(2)

    for val in rx:                  # fires on each change
        print "changed: {val}"
        if val >= 2: break

    print "current: {rx.value}"     # current value, no wait
```

**Rust equivalent**
```rust
let (tx, mut rx) = tokio::sync::watch::channel::<isize>(0);
// ...
while rx.changed().await.is_ok() {
    let val = rx.borrow().clone();
    println!("changed: {}", val);
    if val >= 2 { break; }
}
println!("current: {}", rx.borrow().clone());
```

| Operation | Boring | Rust emitted |
|-----------|--------|--------------|
| Create | `let tx, rx = watch<T>(init)` | `tokio::sync::watch::channel::<T>(init)` |
| Send | `tx.send(v)` | `tx.send(v).ok()` — non-async, overwrites current value |
| Wait for change + read | `rx.recv()` | `rx.changed().await.ok(); rx.borrow().clone()` |
| Read current (no wait) | `rx.value` | `rx.borrow().clone()` |
| Iterate changes | `for val in rx:` | `while rx.changed().await.is_ok() { let val = rx.borrow().clone();` |

**Rules:**
- `rx.value` never blocks — it returns the most recent value immediately.
- Intermediate values can be skipped: if the sender writes twice before the receiver polls, only the latest value is seen.
- `watch` is available in both multi-thread and single-thread modes — `tokio::sync::watch` is used in both (its internals use `Arc` which is always `Send`).

---

## 19. Task handles and parallel awaiting

### Capturing a JoinHandle — `let f = task: expr`

A bare `task:` statement spawns a task and discards its handle. Prefix it with a `let` binding to capture the `JoinHandle`:

```boring
let f1 = task: fetch_users()
let f2 = task: fetch_products()
```

Both tasks run concurrently from the moment they are spawned.

### Awaiting a handle — `f.value`

Read `.value` on a handle to await its result:

```boring
let users    = f1.value
let products = f2.value
```

Inside a `throws` function `.value` becomes `.await?`; elsewhere it becomes `.await.unwrap()`.

### Parallel await — `let a, b = join(task f1(), task f2())`

`join` drives multiple tasks to completion simultaneously using `tokio::join!`.
Tasks can be inlined directly — no intermediate variable needed:

```boring
let users, products = join(task fetch_users(), task fetch_products())
print users
print products
```

Use `var` for mutable bindings, `_` to discard a result:

```boring
var users, _ = join(task fetch_users(), task log_access())
```

The parenthesised form `let (a, b) = join(...)` is also accepted and equivalent.

### Full example

```boring
int fetch_users() throws:
    wait .fromMillis(10)
    return 42

int fetch_products() throws:
    wait .fromMillis(10)
    return 99

void run() throws:
    # Sequential await
    let f1 = task fetch_users()
    let f2 = task fetch_products()
    let users    = f1.value
    let products = f2.value
    print users    # 42
    print products # 99

    # Parallel await — inline tasks
    let users2, products2 = join(task fetch_users(), task fetch_products())
    print users2    # 42
    print products2 # 99
```

### Transpilation

```rust
// let f1 = task fetch_users()
let f1 = tokio::spawn(async move { fetch_users().await });

// let users = f1.value
let users = f1.await.expect("task panicked");

// let users2, products2 = join(task fetch_users(), task fetch_products())
let (__jh0, __jh1) = tokio::join!(
    tokio::spawn(async move { fetch_users().await }),
    tokio::spawn(async move { fetch_products().await }),
);
let users2    = __jh0?;
let products2 = __jh1?;
```

---

## 20. `Future<T>` — polling and cancellation

`task f()` returns a `Future<T>` handle immediately. The handle exposes three properties and one method:

| Member | Form | Description |
|--------|------|-------------|
| `f.value` | property / `f.value(dur)` | Await the result. Throws `Error.Expired` on timeout, `Error.Cancelled` on cancel. |
| `f.done` | property / `f.done()` | Non-blocking poll — `true` if the result is ready. |
| `f.cancel()` | method | Signal the task to stop. `.value` then throws `Error.Cancelled`. |
| `f.wait` | property | Await without returning the value (fire-and-forget style). |

### Polling with `f.done`

```boring
task int compute(int x):
    wait .fromMillis(50)
    x * 2

task void run():
    let f = task compute(21)

    while !f.done:
        wait .fromMillis(10)

    print f.value   # 42
```

### Waiting with a timeout

```boring
task void run() throws:
    let f = task slow_op()

    try:
        let result = f.value(.fromSecs(5))
        print "got: {result}"
    catch Error.Expired:
        f.cancel()
        print "timed out — task cancelled"
```

### Polling two futures (replaces `select:`)

```boring
task void run() throws:
    let f1 = task fetch_users()
    let f2 = task fetch_posts()

    while !f1.done and !f2.done:
        wait .fromMillis(10)

    if f1.done: print f1.value
    if f2.done: print f2.value
```

### Transpilation

```rust
// let f = task compute(21)
let f = tokio::spawn(async move { compute(21).await });

// f.done
tokio::time::timeout(std::time::Duration::ZERO, f).await.is_ok()

// f.value(.fromSecs(5))
tokio::time::timeout(Duration::from_secs(5), f).await
    .map_err(|_| Error::Expired)??

// f.cancel()
__cancel_f.cancel()
```

## `wait` — pause async

`wait dur` suspends the current task for the specified duration. Equivalent to `tokio::time::sleep(dur).await`.

```boring
wait .fromMillis(500)
wait .fromSecs(1)
```

Used freely inside `loop:` to create periodic loops with explicit placement:

```boring
task monitor():
    var int ticks = 0
    loop:
        wait .fromSecs(1)   # pause first, then body
        ticks += 1
        info "heartbeat #{ticks}"
        if ticks >= 5: break
```

```boring
task process():
    loop:
        let item = queue.recv()      # body first
        handle(item)
        wait .fromMillis(10)  # then pause
```

**Rust equivalent**
```rust
async fn monitor() {
    let mut ticks: isize = 0;
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        ticks += 1;
        log::info!("heartbeat #{}", ticks);
        if ticks >= 5 { break; }
    }
}
```

| Form | Semantics |
|---|---|
| `wait dur` | pause for `dur` at the current point |
| `loop: wait dur; body` | pause **before** each iteration |
| `loop: body; wait dur` | run **immediately**, then pause |

### `wait` with absolute deadline — `Instant`

Instead of a relative duration, `wait` also accepts an `Instant` (an absolute point in time). The transpiler automatically picks the right primitive based on the argument type:

```boring
# Relative duration → tokio::time::sleep(dur).await
wait Duration.fromSecs(5)
wait(Duration.fromMillis(500))

# Absolute deadline → tokio::time::sleep_until(instant).await
wait(Instant.now() + Duration.fromSecs(5))

let deadline = Instant.now() + Duration.fromSecs(10)
wait(deadline)   # explicit form recommended for Instant variables
```

| Boring | Generated Rust |
|---|---|
| `wait(Duration.fromSecs(n))` | `tokio::time::sleep(Duration::from_secs(n)).await` |
| `wait(Instant.now() + dur)` | `tokio::time::sleep_until(instant).await` |
| `wait(deadline)` *(Instant variable)* | `tokio::time::sleep_until(deadline).await` |

---

### Task with built-in timeout — `task(Duration): body`

Pass a `Duration` directly to `task` to spawn a task that automatically expires:

```boring
# Positional — duration is the first argument
let f = task(Duration.fromSecs(5)): fetch(url)

# Labeled — same thing, more explicit
let f = task(timeout = Duration.fromMillis(500)):
    let data = download(url)
    parse(data)
```

When the deadline elapses before the body completes, awaiting the handle throws `Error.Expired`. Handle it with `try … else` or `catch`:

```boring
# Inline else — concise fallback
let result = try f.value else "timed out"

# Block else — full error dispatch
let result = try:
    let f = task(Duration.fromSecs(3)): fetch(url)
    f.value
else:
    match error:
        Error.Expired: "request timed out"
        _:             "other error: {error}"
```

**vs `timeout(dur, fut)` and `f.value(dur)`**

| Syntax | When to prefer |
|--------|---------------|
| `task(dur): body` | Spawning a new task that should expire; result is a JoinHandle |
| `timeout(dur, fut)` | Applying a deadline to an existing future expression |
| `f.value(dur)` / `f.wait(dur)` | Deadline on the **await** only — task keeps running on expiry |

`task(dur): body` is equivalent to spawning `tokio::spawn(async move { tokio::time::timeout(dur, body).await? })`.

**Rust equivalent**
```rust
let f = tokio::spawn(async move {
    tokio::time::timeout(Duration::from_secs(5), async move {
        fetch(url).await
    }).await?
});
// f.await.unwrap()?  →  T or Error::Expired
```

---

## `timeout` — race a future against a timer

`timeout` applies a time limit to an asynchronous computation. If the computation completes in time, its result is returned. Otherwise, `Error.Expired` is thrown.

### Supported syntaxes

```boring
# ① Explicit form — two arguments
timeout(Duration.fromSecs(5), fetch)           # Callable<T> (function reference)
timeout(Duration.fromSecs(5), (): fetch(url))  # explicit zero-arg closure

# ② Trailing body with `:`
timeout(Duration.fromSecs(5)): fetch(url)

# ③ Command style — no separator (same line)
timeout(Duration.fromSecs(5)) fetch(url)

# ④ Multi-line
timeout(Duration.fromSecs(10)):
    let data = download(url)
    parse(data)
```

Forms ②③④ are syntactic sugar for form ①. They are interchangeable.

### Automatic Duration / Instant dispatch

The same `timeout` name works with a `Duration` (relative delay) **or** an `Instant` (absolute deadline). The transpiler automatically picks `timeout` or `timeout_at` based on the type of the first argument:

```boring
# Duration → tokio::time::timeout(dur, …).await?
timeout(Duration.fromSecs(5)): fetch(url)

# Inline Instant → tokio::time::timeout_at(instant, …).await?
timeout(Instant.now() + Duration.fromSecs(5)): fetch(url)

# Instant variable — use the two-arg form (avoids syntactic ambiguity)
let deadline = Instant.now() + Duration.fromSecs(10)
timeout(deadline, (): fetch(url))
```

The main use of absolute deadlines is to **share a time limit** across multiple operations:

```boring
let deadline = Instant.now() + Duration.fromSecs(10)

let data   = timeout(deadline, (): fetchData(url))
let parsed = timeout(deadline, (): parse(data))
let saved  = timeout(deadline, (): save(parsed))
```

### Error handling

`timeout` throws `Error.Expired` if the deadline is exceeded. Use `try … else` or `catch`:

```boring
# Inline fallback
let result = try timeout(Duration.fromSecs(5)) fetch(url)
             else "timed out"

# Block try/catch
try:
    let body = timeout(Duration.fromSecs(10)): download(url)
    process(body)
catch Error.Expired:
    print "request timed out"
```

### Inside a cancellable function

Inside a **cancellable** `task` function (one that can receive `.cancel()`), `timeout` races three branches simultaneously using `select!`: the future itself, the timer, and the cancellation token. Both expiry and cancellation throw distinct typed errors (`Error.Expired` / `Error.Cancelled`).

```boring
task string fetch_user(string url) throws:
    let body = timeout(Duration.fromSecs(10)) download(url)
    parse(body)
```

```boring
try:
    fetch_user(url)
catch Error.Expired:
    print "request timed out"
catch Error.Cancelled:
    print "task was cancelled"
```

> **Prefer `task(Duration): body`** for the common case of spawning a task with a deadline — it is simpler and does not require a cancellable function context. Use `timeout(dur): body` when you need a deadline on an expression without spawning.

### Transpilation

| Boring | Generated Rust |
|---|---|
| `timeout(dur): f()` | `tokio::time::timeout(dur, f()).await?` |
| `timeout(Instant.now() + d): f()` | `tokio::time::timeout_at(instant, f()).await?` |
| `timeout(deadline, (): f())` *(Instant variable)* | `tokio::time::timeout_at(deadline, f()).await?` |
| *(in cancellable)* `timeout(dur): f()` | `tokio::select! { r = f() => Ok(r), _ = sleep(dur) => Err(Expired), _ = cancel => Err(Cancelled) }?` |

---

## 21. Ownership Qualifiers

Boring exposes Rust's ownership model through **type qualifiers** written after a `'` tick.

### Qualifier on the variable name — type inferred

When the type can be inferred from the right-hand side, the qualifier can be attached to the **variable name** instead of the type. The two forms are equivalent:

```boring
let Worker'shared w  = Worker(name = "alice", jobs = 5)   # explicit type
let w'shared         = Worker(name = "alice", jobs = 5)   # inferred — qualifier on name

var Counter'actor c = Counter()   # explicit type
var c'actor         = Counter()   # inferred — qualifier on name
```

**Rust equivalent**
```rust
let w: Arc<Worker>               = Arc::new(Worker { name: Arc::from("alice"), jobs: 5 });
let mut c: Arc<Mutex<Counter>>   = Arc::new(Mutex::new(Counter { value: 0 }));
```

The qualifier-on-name form is especially concise when the type is obvious from context. The full `Type'qualifier name` form remains valid and is preferred when the type needs to be explicit.

### Placement operator — `new`

`new` is a placement operator that signals non-stack allocation without naming a qualifier. The transpiler infers the qualifier from usage (excluding `'stack`):

```boring
let v = Counter()       # inferred — 'stack included in candidates
let v = new Counter()   # inferred — 'stack excluded from candidates
let v'actor = Counter() # explicit qualifier
```

For delayed initialisation, `'new` is the equivalent pseudo-qualifier:

```boring
let Counter v           # delayed init — 'stack included
let Counter'new v       # delayed init — 'stack excluded
let Counter'actor v     # delayed init — explicit qualifier
```

`new` also accepts a GPU arena as first argument (see the CUDA section):

```boring
new(g0) Counter()   # GPU device g0
```

Ownership transfer between bindings is implicit — see "Move semantics" below; there is no explicit move-marker syntax.

Shorthands cover the most common cases without writing a qualifier explicitly:

| Boring shorthand | Strict mode (`--mode strict`) | Managed mode (`--mode managed`) | Meaning |
|------------------|-------------------------------|----------------------------------|---------|
| `T`  | `T` (stack) | `Arc<Mutex<T>>` / `RefCell<T>` | Anonymous — transpiler decides |
| `T'new` | `Box<T>` | `Arc<Mutex<T>>` / `RefCell<T>` | Non-stack placement, qualifier inferred by transpiler |
| `T?` | `Option<T>` | `Option<T>` | Optional value |
| `[T]` | `Vec<T>` | `Vec<T>` | Dynamic array |
| `{T}` | `HashSet<T>` | `HashSet<T>` | Unordered set |
| `{K=V}` | `HashMap<K, V>` | `HashMap<K, V>` | Key-value map |

`T` and `T'new` are **anonymous forms** — the transpiler resolves them based on the active flags. Explicit qualifiers (`T'stack`, `T'heap`, etc.) are **contracts** and are never affected by the mode.

In managed mode, `Arc<Mutex<T>>` is used with `--threading multi` (default) and `RefCell<T>` with `--threading single`.

All ownership qualifiers:

| Boring type        | `--threading multi`                   | `--threading single`  | Semantics                             |
|--------------------|---------------------------------------|-----------------------|---------------------------------------|
| `T'stack`          | `T`                                   | `T`                   | Stack allocation — copied on pass for primitives |
| `T'heap`           | `Box<T>`                              | `Box<T>`              | Exclusive heap ownership              |
| `T'shared`         | `Arc<T>`                              | `Rc<T>`               | Shared ref-counted, read-only         |
| `T'actor`          | `Arc<std::sync::Mutex<T>>`            | `Rc<RefCell<T>>`      | Shared mutable — sync, no tokio required |
| `T'actor'task` / `T'task` | `Arc<tokio::sync::Mutex<T>>`   | not supported         | Shared mutable — async, hold lock across `.await` |
| `T'guard`          | `Arc<std::sync::RwLock<T>>`           | `Rc<RefCell<T>>`      | Shared mutable — reader-writer, sync  |
| `T'guard'task`     | `Arc<tokio::sync::RwLock<T>>`         | not supported         | Reader-writer — async context         |
| `T'shared'weak`    | `std::sync::Weak<T>`                  | `Weak<T>`             | Weak ref to `T'shared`                |
| `T'actor'weak`     | `std::sync::Weak<Mutex<T>>`           | `Weak<RefCell<T>>`    | Weak ref to `T'actor`                 |
| `T'guard'weak`     | `std::sync::Weak<RwLock<T>>`          | `Weak<RefCell<T>>`    | Weak ref to `T'guard`                 |
| `T?`               | `Option<T>`                           | `Option<T>`           | Optional value                        |

### Qualifier groups — parameter constraints

On function parameters, a qualifier group expresses "this parameter accepts any qualifier from this set". The transpiler narrows the set further using the same inference signals as for anonymous variables.

```boring
def process(Counter'mut c):   # 'mut → accepts 'stack, 'heap, 'actor, 'guard
    spawn_actor(c)            # demands 'actor → infers 'actor for c
```

| Group | Accepted qualifiers |
|---|---|
| `T'one` | `'stack`, `'heap` — single-owner forms |
| `T'many` | `'shared`, `'actor`, `'guard` — shared-owner forms |
| `T'mut` | `'stack`, `'heap`, `'actor`, `'guard` — any mutable form |
| `T'req` | `'shared` — always immutable |

Pipe-separated unions are also valid: `T'stack|heap` accepts only `'stack` or `'heap`.

Groups have no Rust representation — no trait bound is emitted. The constraint is enforced at the Boring level: the transpiler rejects callers that pass a qualifier outside the declared group, and uses the body's inference signals to resolve to a single concrete qualifier for emission. If inference cannot resolve to one qualifier, the first member of the group is used as fallback.

> Groups are meaningful only on parameters, not on local variables. On a local variable the inference starting set already covers this information, and writing an explicit qualifier is clearer.

### Transpilation flags

```
boring build --mode strict|managed     # memory management (default: strict)
boring build --threading single|multi  # concurrency model (default: multi)
```

`--threading` is not available for the `--target kernel` target.

### Binding × qualifier combinations

Each qualifier imposes constraints on `mut`. `mut`/`var mut` are forbidden with `'shared` — it is a compile error (no interior mutability to unlock).

`'actor`/`'guard` get no special case in the table below — they're checked exactly like every other type ([§2](#2-variables-and-mutability)): `var` alone is rebind-only, never content-mutable, full stop. (An earlier revision of Boring let `var T'actor x` unlock `def` calls on the strength of the qualifier alone; that exception is retired — see `var mut`.)

| Binding | `'shared` | `'actor` | `'guard` | `'stack` | `'heap` |
|---|---|---|---|---|---|
| `let` | yes | yes | yes | yes | yes |
| `mut` | **error** | yes | yes | yes | yes |
| `var` | yes | yes | yes | yes | yes |
| `var mut` | **error** | yes | yes | yes | yes |
| `lazy` | yes | yes | yes | yes | yes |

Qualifiers carry three kinds of information: Rust mapping, passing semantics, and mutability constraints:

| Qualifier | Passing semantics | Mutability |
|---|---|---|
| `'shared` | by reference (`&Arc<T>`) — no refcount increment | forbidden — immutable shared ref |
| `'actor` | by reference (`&Arc<Mutex<T>>`) — no refcount increment | interior mutability — exclusive lock (Mutex) |
| `'guard` | by reference (`&Arc<RwLock<T>>`) — no refcount increment | interior mutability — shared reads / exclusive writes (RwLock) |
| `'stack` | move | determined by `let`/`mut`/`var` |
| `'heap` | move | determined by `let`/`mut`/`var` |

### `let`, `mut`, `var` with `T'shared` and `T'actor`

`var` on a reference-counted type allows **reassigning the pointer** but never unlocks `def` method calls on its own — the shared value stays read-only unless the type also carries `mut`. For mutation: hold the value with `mut`/`var mut` + plain ownership, or use `T'actor`/`T'guard` (with `mut`/`var mut`) for shared mutable state.

| Declaration | Reassign | `req` methods | `def` methods |
|---|---|---|---|
| `let T'shared x` / `let x'shared` | ✗ | ✓ | ✗ |
| `var T'shared x` / `var x'shared` | ✓ | ✓ | ✗ — `'shared` has no interior mutability, `mut`/`var mut` are compile errors on it |
| `let T'actor x` / `let x'actor` | ✗ | ✓ | ✗ |
| `mut T'actor x` / `mut x'actor` | ✗ | ✓ | ✓ |
| `var T'actor x` / `var x'actor` | ✓ | ✓ | ✗ — `var` alone no longer suffices; use `var mut` |
| `var mut T'actor x` | ✓ | ✓ | ✓ |
| `let T'guard x` / `let x'guard` | ✗ | ✓ | ✗ |
| `mut T'guard x` / `mut x'guard` | ✗ | ✓ | ✓ |
| `var T'guard x` / `var x'guard` | ✓ | ✓ | ✗ — same as `'actor` above |
| `var mut T'guard x` | ✓ | ✓ | ✓ |

```boring
struct Counter:
    var int value = 0
    def inc(): self.value += 1
    req int get():  self.value

var c'shared  = Counter()
var c2'shared = Counter()
c = c2        # OK — reassign the Arc/Rc pointer
c.get()       # OK — req (non-mutating) methods work fine
# c.inc()     # ERROR — def methods are forbidden on T'shared regardless of mut/var
              #         use T'actor (with mut/var mut) for shared mutable state

var mut d'actor = Counter()
d.inc()       # OK — var mut grants both rebind and content mutation
d = Counter() # OK — rebind
```

### Explicit borrow syntax — `T&`

> **You rarely need this.** Structs and enums are already passed by reference automatically (see [Pass-by-reference — automatic](#pass-by-reference--automatic)). Reach for `T&` only when you need an explicit lifetime annotation or must lock in the borrow convention regardless of how the function body evolves.

Borrows are written with `&` directly after the type name. The binding keyword defines what the callee can do with the reference:

| Syntax | Rust type | Semantics |
|---|---|---|
| `T& m` / `let T& m` | `&T` | read-only — callee cannot modify content or binding |
| `mut T& m` | `&mut T` | callee can modify the content of the caller's instance |
| `var T m` | `&mut T` | callee can modify the caller's variable — equivalent to `mut T& m` without the explicit borrow syntax |

| Form          | Rust type     | Notes                                                  |
|---------------|---------------|--------------------------------------------------------|
| `T&`          | `&T`          | Universal borrow — coerced from any qualifier          |
| `mut T&`      | `&mut T`      | Mutable universal borrow                               |
| `T?&`         | `&Option<T>`  | Borrow an optional                                     |
| `T&a`         | `&'a T`       | Borrow with explicit lifetime                          |

Lifetimes are only valid in borrow position (`&`), never on owned qualifiers (`'`). The lifetime letter is declared as a type parameter with `<'a>` and can appear anywhere a borrow is written:

```boring
# both params and return tied to the same lifetime 'a
# no need to declare <'a> — lifetimes are inferred from usage
string&a longest(string&a x, string&a y):
    if x.len() > y.len(): x else y
```

**Rust equivalent**
```rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

Borrow forms with lifetime:

| Boring | Rust | Meaning |
|--------|------|---------|
| `T&a` | `&'a T` | borrow a value |
| `T'&a` | `&'a Box<T>` | borrow a heap-allocated value |

### Built-in type aliases

The primitive types are defined as aliases with an explicit qualifier:

Primitives (`int`, `uint`, `float`, `bool`) are always `Copy` Rust types — they carry no qualifier. `string` always compiles to `Rc<str>` (single-thread) or `Arc<str>` (multi-thread).

```boring
let int   x = 42    # isize
let uint  n = 100   # usize
```

> For `string` details see [Advanced — Strings](#advanced--strings-string-and-arc-str).

### Weak references — `T'shared'weak`, `T'actor'weak`, `T'guard'weak`

A `'weak` qualifier produces a non-owning pointer that does not prevent the pointee from being dropped. The base qualifier is written explicitly or inferred from the right-hand side:

| Qualifier          | Rust type (multi)               | Meaning                         |
|--------------------|---------------------------------|---------------------------------|
| `T'shared'weak`    | `std::sync::Weak<T>`            | Weak ref to `T'shared`          |
| `T'actor'weak`     | `std::sync::Weak<Mutex<T>>`     | Weak ref to `T'actor`           |
| `T'guard'weak`     | `std::sync::Weak<RwLock<T>>`    | Weak ref to `T'guard`           |

At a **binding site** the base qualifier can be inferred from the right-hand side — writing `'weak` alone is enough:

| RHS qualifier  | Inferred weak type      | Rust                                      |
|----------------|-------------------------|-------------------------------------------|
| `'shared`      | `T'shared'weak`         | `Weak<T>` / `std::sync::Weak<T>`          |
| `'actor`       | `T'actor'weak`          | `std::sync::Weak<Mutex<T>>`               |
| `'guard`       | `T'guard'weak`          | `std::sync::Weak<RwLock<T>>`              |

**Assigning** a strong reference to a weak binding automatically calls `Rc::downgrade` or `Arc::downgrade`. Calling **`.upgrade()`** returns the strong reference or `nil` if the object was already dropped.

```boring
struct Resource:
    init(pub string label)

# Strong ref (Rc in single-thread, Arc in multi-thread)
let Resource'shared strong = Resource(label = "config.toml")
let Resource'shared'weak w1 = strong   # downgrade → Weak<Resource>

# Qualifier inferred from RHS
let a'shared = Resource(label = "shared")
let b'weak   = a                       # Weak<Resource> (inferred from a)

# .upgrade() recovers the strong reference
let r = w1.upgrade()
print r.label                          # config.toml
```

**Rust equivalent (multi-thread)**
```rust
let strong: Arc<Resource> = Arc::new(Resource { label: Arc::from("config.toml") });
let w1: std::sync::Weak<Resource> = Arc::downgrade(&strong);

let r = w1.upgrade().unwrap();
println!("{}", r.label);
```

Weak refs break **reference cycles** — a parent holds a strong ref to its children, each child holds a weak back-ref to its parent.

```boring
string describe(Resource'shared'weak w):
    let r = w.upgrade()
    "resource: {r.label}"

print describe(w1)   # resource: config.toml
```

> The transpiler selects `Weak<T>` (single-thread) or `std::sync::Weak<T>` (multi-thread) based on `--threading`.

> In most programs you never need to write a qualifier explicitly. The transpiler infers the right one from how each variable is used. See [chapter 30 — Qualifier Inference](#30-qualifier-inference) for the full inference algorithm, including signal table, size-based fallback, and cross-function propagation.

### Scoped access blocks — `with`

A `T'actor`/`T'guard` value normally acquires and releases its lock **once per method call or field access** — `c.increment()` transpiles to `c.lock().unwrap().increment()`. `with` holds the lock across an entire block instead, so several operations happen under one critical section:

```boring
struct Counter:
    var int value = 0

    def increment():
        value += 1

var mut c'actor = Counter(0)

with c:
    c.increment()
    c.increment()
    c.increment()
print c.value   # 3
```

**Rust equivalent**
```rust
let mut c: Arc<std::sync::Mutex<Counter>> = Arc::new(std::sync::Mutex::new(Counter { value: 0 }));
{
    let mut c = c.lock().unwrap();
    c.increment();
    c.increment();
    c.increment();
}
println!("{}", c.lock().unwrap().value);
```

Existing per-call locking is unchanged for code that doesn't use `with` — this is purely additive.

`with` is one keyword, not a read/write pair — the access level is decided per block:

- **`let`/`var`-bound name (no `mut`)** — always read-only (`var` alone doesn't grant content mutation either — see [§2](#2-variables-and-mutability) — so there is nothing to scan for).
- **`mut`/`var mut`-bound name** — the compiler scans the block's own body (recursing into `if`/`while`/`for`/`match`/closures nested inside it, but never into a called function's own body) for a direct assignment, an index/field assignment, or a `def` (mutating) method call on the name. Found → the block gets write access; not found → read-only, even though the binding could support a mutation elsewhere in the program.

```boring
struct Cell:
    var int value = 0

    req int peek():
        value

    def bump():
        value += 1

var mut b'guard = Cell(10)

with b:
    print b.peek()   # read-only block — RwLock::read()

with b:
    b.bump()          # mutation found — RwLock::write()
```

`'actor`/`'actor'task` always acquire the same `Mutex` either way — a `Mutex` has no shared-read mode, so the scan only decides which method calls are legal inside the block (`def` vs `req`), not what gets locked.

Multiple names may be listed together — each gets its own acquire/release, but the block reads as one critical section:

```boring
with a, b:
    a.increment()
    b.increment()
```

Nesting a block on the **same** name inside itself is a compile error (double-acquire — `Mutex`/`RwLock` are not reentrant). Nesting on **different** names is unrestricted.

### `with` for GPU kernel fields — `'gpu'unified`, `'gpu'global`

A `kernel Name:` struct's `'unified`/`'global` field is read back from the GPU **on every single access** — `k.result[i]` inside a loop re-reads the *entire* buffer back from the GPU on every iteration. Binding the field through a `'gpu'unified`/`'gpu'global`-qualified alias and reading it inside a `with` block materializes it exactly once instead, however many times the block's body indexes it:

```boring
var k = VectorAdd(host_a, host_b)
kernel:
    k(block = 256)

let [int]'gpu'unified result = k.result   # compile-time alias — no transfer yet
with result:                              # `result` is `let`-bound -> read-only
    for i in 0..n:
        print "c[{i}] = {result[i]}"      # readback happens once, here
```

This is purely additive — `k.result[i]` without an alias still works exactly as before, and is the right choice for a single access. The alias only pays off when the same field is read (or written) more than once in the same scope. A `mut`/`var`-bound alias follows the same read/write scan as `'actor`/`'guard`: mutating it inside the block (`result[i] = v`) writes the whole array back to the GPU once, at block close; a read-only block never does.

The `'gpu'unified`/`'gpu'global` annotation itself is optional here — it's inferred whenever the initializer is a bare `k.field` read on a tracked kernel instance whose field is actually declared `'unified`/`'global`, so `let result = k.result` behaves identically to the explicit form above:

```boring
let result = k.result   # qualifier inferred from k.result's own 'unified declaration
with result:
    for i in 0..n:
        print "c[{i}] = {result[i]}"
```

A `'gpu'unified`/`'gpu'global`-qualified variable that is **not** initialized from a bare kernel-field read — an ordinary array literal or expression — is just a plain host array, freely indexed and assigned with no `with` wrapper required anywhere; the qualifier only matters once it's passed into a kernel constructor, which uploads it.

### Residency across a function boundary

Everything above works when the kernel is constructed and read back in the same scope. Residency also survives a **function-call boundary**: a function declared with a `'gpu'unified`/`'gpu'global` return type hands its result straight to the next call as a still-resident value, with no host round-trip in between — only the final consumer's `with` pays a real transfer:

```boring
req [float]'gpu'unified linear_gpu([float] x, [float] w, [float] b, int seq, int d_in, int d_out):
    mut k = Linear(x, w, b, seq, d_in, d_out)
    kernel:
        k(block = seq * d_out)
    k.y

req [float]'gpu'unified gelu_gpu([float] x, int n):
    mut k = Gelu(x)
    kernel:
        k(block = n)
    k.out

let fc = linear_gpu(h, w1, b1, 1, d, d * 4)   # dispatch, stays resident
let act = gelu_gpu(fc, d * 4)                  # fc consumed directly — no upload, no download
with act:
    print "act[0] = {act[0]}"                 # the only real device->host transfer in the chain
```

A parameter is only ever eligible for this zero-copy handoff when it's used *exclusively* as a kernel-constructor argument in the function's body — `x` above, in both functions; the block size instead comes from an ordinary `int` parameter (`n`, `d * 4`), which is why `gelu_gpu` doesn't read `x.length`. A parameter used more richly than that (indexed, measured, passed elsewhere) keeps the ordinary host-array behavior for that one parameter — no speedup, but no error either. The `'gpu'unified`/`'gpu'global` annotation on `let fc`/`let act` is optional here too, inferred the same way as the same-scope case.

> Both halves of this design — same-scope kernel-field materialization and cross-function residency — are implemented and shipped. See [Scoped Access Blocks](scoped-access-blocks.html) for the full design, the codegen this actually produces, and current known limitations (cuda/metal targets don't share this yet).

---

## 22. Defer

`defer` registers a block of code to run when the enclosing function exits, regardless of how it exits. Multiple defers execute in **LIFO** order (last registered, first executed).

```boring
string with_cleanup():
    var log = ""
    defer: log = "{log}+closed"
    log = "open+work"
    log
# result: "open+work+closed"
```

```boring
string lifo():
    var log = ""
    defer: log = "{log}A"
    defer: log = "{log}B"
    defer: log = "{log}C"
    log
# result: "CBA"
```

**Rust equivalent**
```rust
fn with_cleanup() -> Arc<str> {
    let mut log = Arc::<str>::from("");
    // body
    log = Arc::from("open+work");
    let __ret = log.clone();
    // defers (LIFO)
    log = Arc::<str>::from(format!("{}+closed", log));
    __ret.clone()
}
```

---

## 23. Tasks (Async)

### Declaring a `task` function

Mark a function as asynchronous with `task`. It behaves as a normal function in the interpreter
(runs synchronously); `task` only affects the transpiled Rust output.

```boring
task int fetch(int a, int b):
    a + b
```

**Rust equivalent**
```rust
async fn fetch(a: isize, b: isize) -> isize { a + b }
```

> The `task` prefix works with all function qualifiers: `task req` (non-mutating async), `task set` (async setter), and `task def` (explicit mutating async — `def` is always optional when a return type is present). Write `task int fetch(...)` or `task f()` for the common cases.

### Spawning a task — `task expr` / `task: expr`

Use `task` as a keyword-expression to spawn an async computation and obtain a future.
Both inline and block forms are available:

```boring
let f1 = task fetch(10, 20)     # inline — no colon needed

let f2 = task: fetch(99, 1)     # inline with colon

let f3 = task:                  # block form
    fetch(1, 2)
```

**Rust equivalent**
```rust
let f1 = tokio::spawn(async move { fetch(10, 20).await });
let f2 = tokio::spawn(async move { fetch(99, 1).await });
let f3 = tokio::spawn(async move { fetch(1, 2).await });
```

### Detached task — fire and forget

Omit the `let` binding to spawn a task without keeping a future.
The task runs in the background; its result is discarded.

```boring
task log(string msg):
    print "bg: {msg}"

def main():
    task log("hello")   # detached — no future returned
    task log("world")   # detached
    print "main done"
```

**Rust equivalent**
```rust
tokio::spawn(async move { log("hello").await });  // JoinHandle dropped → detached
tokio::spawn(async move { log("world").await });
println!("main done");
```

> **`main` never needs `task`**
> The compiler always treats `main` as an async entry-point when its body uses any async
> construct (`task`, `wait`, `timeout`, calls to task functions, etc.).
> Writing `def main():` is equivalent to `task main():` — the `task` qualifier is accepted
> but redundant.

```boring
task log(string msg):
    print "bg: {msg}"

# main is NOT marked task — the compiler handles it
task log("hello")
task log("world")
print "main done"
```
>
> Transpiles to the same `#[tokio::main] async fn main()` shown above.

### CPU-bound tasks — automatic `spawn_blocking`

When `task` is applied to a **synchronous** function (declared with `def`, not `task`),
the compiler automatically uses `tokio::task::spawn_blocking` instead of `tokio::spawn`.
No annotation needed — the function's declaration tells the compiler everything.

```boring
# Async function — runs on the tokio runtime
task string fetch(string url):
    download(url)

# Sync function — CPU-intensive, will block if not offloaded
Data compress(Data d):
    heavy_computation(d)

def main():
    # task fn → tokio::spawn (async, non-blocking)
    let f1 = task fetch(url)

    # def fn → tokio::task::spawn_blocking (blocking thread pool)
    let f2 = task compress(data)

    print f1.value   # awaits the async task
    print f2.value   # awaits the blocking task
```

**Rust equivalent**
```rust
let f1 = tokio::spawn(async move { fetch(url).await });
let f2 = tokio::task::spawn_blocking(move || compress(data));

println!("{}", f1.await.unwrap());
println!("{}", f2.await.unwrap());
```

The rule is simple:

| Function declared as | `task f(args)` emits |
|---|---|
| `task f(…):` (async) | `tokio::spawn(async move { f(args).await })` |
| `def f(…):` (sync) | `tokio::task::spawn_blocking(move \|\| f(args))` |

> **Note** — block-form `task: { ... }` always uses `tokio::spawn`. Blocks may contain
> channel sends, actor method calls, or other async operations that look synchronous
> but require the async runtime.
>
> If the block is purely synchronous (no `task` calls, no channel operations) and you
> want `spawn_blocking`, extract it into a named `def` function — `def` functions cannot
> call `task` functions, so extracting truly synchronous work is always valid:
>
> ```boring
> def int heavy(): crunch_numbers(1_000_000)   # purely sync — valid def
>
> let f = task heavy()   # → spawn_blocking, auto-detected
> ```
>
> If the block contains async calls (`task`, `wait`, `timeout`, channel sends…),
> `spawn_blocking` is not appropriate — those operations require the async runtime.
> In that case, `tokio::spawn` is the correct primitive and the block form is correct as-is.

### Awaiting a future — `.value` and `.wait`

Assign the task to a variable and call `.value` to wait for its result, or `.wait` to wait
without capturing the return value (void). If the task threw, the error is re-thrown in the caller.
**The calling function must also be marked `task`** — you cannot await outside an async context.

```boring
task int compute(int n):
    n * n

task notify(string msg):
    print "done: {msg}"

def main():
    let f = task compute(7)
    let result = f.value        # waits and captures the result: 49
    print result

    let fn = task notify("ping")
    fn.wait                     # waits but discards the result (void)
    print "notified"
```

**Rust equivalent**
```rust
async fn compute(n: isize) -> isize { n * n }

#[tokio::main]
async fn main() {
    let f = tokio::spawn(async move { compute(7).await });
    let result = f.await.unwrap();   // .value
    println!("{}", result);

    let fn = tokio::spawn(async move { notify("ping").await });
    fn.await.unwrap();               // .wait — result discarded
    println!("notified");
}
```

| Form | Returns | Throws | Use when |
|------|---------|--------|----------|
| `f.value` | the task's result | any error from the task | you need the return value |
| `f.wait`  | nothing (void)    | any error from the task | you only need synchronisation |

### Awaiting with a timeout — `.value(dur)` and `.wait(dur)`

Pass a `Duration` or an `Instant` to set a deadline on the await call.
If the deadline elapses before the task finishes, `Error.Expired` is thrown in
**the caller** — the task itself keeps running unaffected.

```boring
task int fetch(string url):
    # ... slow network call ...
    42

def main():
    let f = task fetch("https://example.com")

    # Relative timeout — throw Error.Expired after 5 s
    let result = try f.value(Duration.fromSecs(5)) else -1
    print result                       # -1 if timed out, 42 otherwise

    # Absolute deadline shared across several awaits
    let deadline = Instant.now() + Duration.fromSecs(10)
    let a = task fetch("https://a.com")
    let b = task fetch("https://b.com")
    print a.value(deadline)            # throws Error.Expired if past deadline
    print b.value(deadline)            # same shared deadline

    # Discard the result but still enforce a cap
    let g = task fetch("https://bg.com")
    g.wait(Duration.fromMillis(500))   # throws Error.Expired if too slow
```

**Rust equivalent**
```rust
let handle = tokio::spawn(async move { fetch(url).await });

// f.value(dur)
let result = tokio::time::timeout(dur, handle).await??;

// f.value(inst)
let result = tokio::time::timeout_at(inst, handle).await??;
```

| Form | Returns | Timeout arg | Throws on expiry |
|------|---------|-------------|-----------------|
| `f.value` / `f.value()` | `T` | — | — |
| `f.wait`  / `f.wait()`  | void | — | — |
| `f.value(Duration)` | `T` | relative | `throws` (Error.Expired or task error) |
| `f.value(Instant)`  | `T` | absolute | `throws` (Error.Expired or task error) |
| `f.wait(Duration)`  | void | relative | `throws` (Error.Expired or task error) |
| `f.wait(Instant)`   | void | absolute | `throws` (Error.Expired or task error) |

> **Note:** `Error.Expired` is thrown in the **caller** only. The spawned task
> continues running in the background. To also stop it, call `f.cancel()` after
> catching the error.

### Parallel tasks

Spawn multiple tasks before awaiting any of them to run them in parallel:

```boring
task int fetch(int a, int b):
    a + b

task string greet(string name):
    "hello, {name}"

def main():
    let f1 = task fetch(10, 20)
    let f2 = task greet("world")
    print f1.value             # 30
    print f2.value             # hello, world
```

**Rust equivalent**
```rust
async fn fetch(a: isize, b: isize) -> isize { a + b }
async fn greet(name: &str) -> Arc<str> { Arc::<str>::from(format!("hello, {}", name)) }

#[tokio::main]
async fn main() {
    let f1 = tokio::spawn(async move { fetch(10, 20).await });
    let f2 = tokio::spawn(async move { greet("world").await });
    println!("{}", f1.await.unwrap());   // 30
    println!("{}", f2.await.unwrap());   // hello, world
}
```

### `main` — entry point

`main` is always the program entry point. It never needs the `task` qualifier — the compiler
detects async usage automatically and wraps it in `#[tokio::main]` when needed:

```boring
def main():                  # plain — sync if body has no async constructs
    print "hello"

def main():                  # auto-promoted to async — uses task, wait, timeout, etc.
    let f = task fetch(1, 2)
    print f.value

def main() throws:           # can also throw
    let f = task fetch(1, 2)
    print f.value
```

**Rust equivalent**
```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = tokio::spawn(async move { fetch(1, 2).await });
    println!("{}", f.await.unwrap());
    Ok(())
}
```

> The interpreter runs `task` functions synchronously. Tasks and `.value` behave like a regular
> function call at runtime; the async machinery only appears in the transpiled Rust output.

### Data capture — no explicit `move` needed

Boring analyses the variables used inside each `task` body and applies the right ownership
strategy automatically — no `move` keyword or explicit `.clone()` required.

| Variable type          | Capture strategy                                             |
|------------------------|--------------------------------------------------------------|
| `int`, `float`, `bool` | Copied into the task (Copy types)                           |
| `string`               | `Rc::clone` / `Arc::clone` — cheap reference count bump, no data copy |
| `T'actor`              | `Arc::clone` — the mutex is shared across the tasks          |
| `T'` (owned)           | Moved into the task; the outer binding is invalidated        |
| Array / Dict / Set without qualifier | **Blocked** — use `T'shared` or `T'actor` instead |

```boring
task string transform(string s):
    "done: {s}"

def main():
    let string label = "hello"      # string — Rc<str>, cheap clone into the task
    let f = task transform(label)   # Rc::clone — no data copy
    print label                    # hello  (outer binding intact)
    print f.value                  # done: hello
```

**Rust equivalent**
```rust
async fn transform(s: &str) -> Arc<str> { Arc::<str>::from(format!("done: {}", s)) }

#[tokio::main]
async fn main() {
    let label: Rc<str> = Rc::from("hello");
    // Rc::clone is cheap — no data copy:
    let f = tokio::spawn(async move { transform(label).await });
    println!("{}", label);              // hello
    println!("{}", f.await.unwrap());   // done: hello
}
```

### Task methods — `self` must be `T'shared`

A method marked `task` takes ownership of the receiver as `Arc<Self>`.
This allows the future to safely cross thread boundaries without borrowing.
The struct variable must therefore be declared with the `'shared` qualifier.

```boring
struct Worker:
    string name
    int jobs

task Worker.run():
    print "worker {self.name} processing {self.jobs} jobs"

def main():
    let w'shared = Worker(name = "alice", jobs = 5)
    let f = task w.run()      # Arc::clone(&w) is inserted automatically
    print w.name             # alice — w is still accessible
    f.value                   # wait for run() to finish
```

**Rust equivalent**
```rust
async fn run(self: Arc<Self>) -> () {
    println!("worker {} processing {} jobs", self.name, self.jobs);
}

#[tokio::main]
async fn main() {
    let w: Arc<Worker> = Arc::new(Worker { name: "alice", jobs: 5 });
    let f = tokio::spawn({ let w = Arc::clone(&w); async move { w.run().await } });
    println!("{}", w.name);    // alice — Arc still accessible
    f.await.unwrap();
}
```

A plain struct (without `'shared`) cannot be used with `task w.method()` — the interpreter
will reject the capture at runtime, and the generated Rust would not compile
(`tokio::spawn` requires `'static` bounds that a borrowed `&self` cannot satisfy).

### Shared mutable state — `T'actor` / `T'actor'task`

`T'shared` gives read-only shared access (`Arc<T>`). When multiple tasks need to
**read and write** the same struct, use **`T'actor`** — this wraps the value in
`Arc<std::sync::Mutex<T>>` and inserts `.lock().unwrap()` automatically at every
field access and method call. No explicit locking is ever written in Boring source.

Mutation goes through the lock, not through `&mut self` on the binding — but Boring
still gates it on the binding itself, with **no exception** for `'actor`/`'guard`
([§2](#2-variables-and-mutability)): `mut`/`var mut` unlock `def` calls, plain `var`
(no `mut`) only allows reassigning the pointer, and `let` gives a **read-only**
handle — `req` methods and field reads work, but `def` calls are rejected, exactly as
for any other type. Use `mut`/`var mut` when the binding itself needs to mutate the
shared value; use `let` (or plain `var`, if it also needs reassigning) to hand out a
read-only view of the same underlying lock.

In an **async context** (inside a `task` function), use **`T'actor'task`** instead —
this uses `Arc<tokio::sync::Mutex<T>>` and inserts `.lock().await` so the lock can
be held across `.await` points.

```boring
struct SharedCount:
    var int value = 0
    def inc():
        self.value += 1
    req int get():
        self.value

def main():
    mut c'actor = SharedCount()
    c.inc()                    # → c.lock().unwrap().inc()
    c.inc()
    print "count = {c.value}"  # → c.lock().unwrap().value

task async_main():
    mut c'actor'task = SharedCount()
    c.inc()                    # → c.lock().await.inc()
    c.inc()
    print "count = {c.value}"  # → c.lock().await.value
```

**Rust equivalent (sync)**
```rust
let mut c: Arc<std::sync::Mutex<SharedCount>> =
    Arc::new(std::sync::Mutex::new(SharedCount::new()));
c.lock().unwrap().inc();
c.lock().unwrap().inc();
println!("count = {}", c.lock().unwrap().value);
```

**Rust equivalent (async)**
```rust
let mut c: Arc<tokio::sync::Mutex<SharedCount>> =
    Arc::new(tokio::sync::Mutex::new(SharedCount::new()));
c.lock().await.inc();
c.lock().await.inc();
println!("count = {}", c.lock().await.value);
```

The same pattern works for struct fields:

```boring
struct App:
    SharedCount'actor'task counter    # field is Arc<tokio::sync::Mutex<SharedCount>>

    task bump():
        self.counter.inc()               # → self.counter.lock().await.inc()
        print "now = {self.counter.value}"
```

| Qualifier | Rust type | Semantics |
|---|---|---|
| `T'shared` | `Arc<T>` (multi) / `Rc<T>` (single) | read-only shared; task methods use `self: Arc<Self>` |
| `T'actor` | `Arc<std::sync::Mutex<T>>` | shared mutable; sync lock (`.lock().unwrap()`) |
| `T'actor'task` | `Arc<tokio::sync::Mutex<T>>` | shared mutable; async lock (`.lock().await`), safe across await points |
| `T'guard` | `Arc<std::sync::RwLock<T>>` | reader-writer; sync (`read()`/`write()`) |
| `T'guard'task` | `Arc<tokio::sync::RwLock<T>>` | reader-writer; async (`read().await`/`write().await`) |

> Use `'actor'task` / `'guard'task` inside `task` functions. Use bare `'actor` / `'guard` in sync code.

### Read-heavy shared state — `T'guard` / `T'guard'task`

When reads dominate, **`T'guard`** uses a `RwLock` instead of a `Mutex`. Multiple
tasks can read concurrently; a write acquires an exclusive lock. The transpiler
automatically chooses the right lock mode based on the method declaration:

- `req` methods and field reads → `.read()` / `.read().await` (shared lock, concurrent)
- `def` methods and field writes → `.write()` / `.write().await` (exclusive lock)

```boring
struct Counter:
    var int value = 0
    req int get():
        self.value
    def inc():
        self.value += 1

task run():
    mut c'guard'task = Counter()
    c.inc()                   # → c.write().await.inc()
    c.inc()
    print "count = {c.get()}" # → c.read().await.get()
```

**Rust equivalent**
```rust
let c: Arc<tokio::sync::RwLock<Counter>> =
    Arc::new(tokio::sync::RwLock::new(Counter::new()));
c.write().await.inc();
c.write().await.inc();
println!("count = {}", c.read().await.get());
```

Weak references work the same as with `'actor`:

```boring
let c'guard'task  = Counter()
let w'guard'weak  = c          # Arc::downgrade(&c)

if let strong = w.upgrade():
    print strong.get()         # → strong.read().await.get()
```

| `'actor` / `'actor'task` | `'guard` / `'guard'task` |
|---|---|
| `Arc<std::sync::Mutex<T>>` / `Arc<tokio::sync::Mutex<T>>` | `Arc<std::sync::RwLock<T>>` / `Arc<tokio::sync::RwLock<T>>` |
| `.lock()` for everything | `.read()` / `.write()` |
| Fair, no starvation risk | Can starve writers under heavy read load |
| Lower overhead for write-heavy workloads | Better throughput when reads dominate |

> Use `'actor` / `'actor'task` as the default. Switch to `'guard` / `'guard'task` when profiling shows lock contention and reads significantly outnumber writes.

### Task cancellation — `f.cancel()`

Boring supports **graceful cancellation** of spawned tasks via `f.cancel()`. Once signalled, any subsequent `.value` call throws `Error.Cancelled`.

#### Simple cancellation

```boring
task void run() throws:
    let f = task worker()

    wait .fromMillis(100)
    f.cancel()

    try:
        f.wait
    catch Error.Cancelled:
        print "worker cancelled"
```

#### Cancel + throw pattern

```boring
task string process() throws:
    wait .fromMillis(500)
    return "result"

task void run() throws:
    let f = task process()
    f.cancel()

    try:
        let v = f.value
        print v
    catch Error.Cancelled:
        print "cancelled before result"
```

#### How the transpiler handles cancellation

| Boring concept | Transpiled Rust |
|----------------|-----------------|
| `f.cancel()` | `__cancel_f.cancel()` (via the per-spawn `CancellationToken`) |
| `f.done` | `tokio::time::timeout(Duration::ZERO, f).await.is_ok()` |
| `f.wait` on a `throws` task (in `throws` context) | `{ let _ = f.await.unwrap()?; }` — propagates the task error |
| `f.value` on a `throws` task (in `throws` context) | `f.await.unwrap()?` — propagates the task error |

The `tokio-util` crate (`tokio-util = { version = "0.7", features = ["sync"] }`) is added to `Cargo.toml` automatically when `f.cancel()` is used.

---

## 24. Attributes

Attributes are written with `@` and apply to the next declaration.
**Parentheses are optional** — both forms are accepted and produce identical output.

```boring
# Without parentheses (preferred for brevity)
@derive Debug, Clone, PartialEq
struct Color:
    int r
    int g
    int b

# With parentheses (also accepted, backward compatible)
@derive(Debug, Clone, PartialEq)
struct Color:
    int r
    int g
    int b
```

**Rust equivalent**
```rust
#[derive(Debug, Clone, PartialEq)]
struct Color { r: isize, g: isize, b: isize }
```

The no-parentheses form applies to all `@` attributes, including `@error` on enum variants:

```boring
enum Shape:
    @error "unknown shape: {0}"    # no parentheses
    Unknown(string)

    @error("unsupported operation") # parentheses — also valid
    Unsupported
```

---

## 25. Format Specifiers

Format specifiers in Boring are **identical to Rust's** `std::fmt` specifiers. They follow the colon inside an interpolation `{expr:spec}`.

| Specifier   | Example           | Output (for `n=255`, `f=0.753`)  |
|-------------|-------------------|----------------------------------|
| `x`         | `{n:x}`           | `ff`                             |
| `X`         | `{n:X}`           | `FF`                             |
| `b`         | `{n:b}`           | `11111111`                       |
| `o`         | `{n:o}`           | `377`                            |
| `e`         | `{f:.2e}`         | `7.53e-1`                        |
| `E`         | `{f:.2E}`         | `7.53E-1`                        |
| `.N`        | `{f:.3}`          | `0.753`                          |
| `+.N`       | `{f:+.2}`         | `+0.75`                          |
| `W`         | `{n:8}`           | `     255`                       |
| `0W`        | `{n:08}`          | `00000255`                       |
| `<W`        | `{n:<8}`          | `255     `                       |
| `^W`        | `{n:^8}`          | `  255   `                       |
| `?`         | `{val:?}`         | debug representation             |
| `#x`        | `{n:#x}`          | `0xff`                           |

```boring
let n     = 255
let ratio = 0.753
print "hex: {n:x}, bin: {n:b}"
print "float: {ratio:.3}, sci: {ratio:.2e}"
print "padded: '{n:8}', zero: '{n:08}'"
```

---

## 26. Built-in Functions

These functions are available without any import.

### I/O

| Function       | Description                        |
|----------------|------------------------------------|
| `print s`               | Print with newline — inline interpolation: `print "hi, {name}"` |
| `print "{}", expr`      | Print with newline — positional substitution: `print "a={}, b={}", a, b` |
| `write(s)`              | Print without newline              |
| `readLine()`            | Read a line from stdin             |
| `str(s)`                | Convert to String; also accepts format styles: `str("{x}")`, `str("{}", x)`, `str("a={}, b={}", a, b)` |

### Log-level builtins

`error`, `warn`, `info`, `debug`, and `trace` work exactly like `print` — same string
interpolation syntax, same `{}` positional holes — but write to **stderr** with a log level
prefix.

In the interpreter each call uses `eprintln!` with a `[LEVEL]` prefix.
In transpiled code they map to the matching `log` crate macros (`log::error!`, `log::warn!`,
etc.). When any log-level builtin is used, `log = "0.4"` is automatically added to the
generated `Cargo.toml`.

```boring
error "connection failed: {reason}"
warn "retrying in {delay}s"
info "server started on port {port}"
debug "value = {x}"
trace "entering parse_expr"
```

**Rust equivalent**
```rust
log::error!("connection failed: {}", reason);
log::warn!("retrying in {}s", delay);
log::info!("server started on port {}", port);
log::debug!("value = {}", x);
log::trace!("entering parse_expr");
```

| Builtin   | Interpreter (stderr)         | Transpiled            |
|-----------|------------------------------|-----------------------|
| `error …` | `[ERROR] …`                  | `log::error!(…)`      |
| `warn …`  | `[WARN] …`                   | `log::warn!(…)`       |
| `info …`  | `[INFO] …`                   | `log::info!(…)`       |
| `debug …` | `[DEBUG] …`                  | `log::debug!(…)`      |
| `trace …` | `[TRACE] …`                  | `log::trace!(…)`      |

### File system — `fs`

File system operations are available as methods on the built-in `fs` namespace.
No import is required.  All operations are `throws` — call them inside a `throws` function or wrap with `try … else`.

In **async** functions (`task`) the calls use `tokio::fs` with `.await`; in synchronous functions they use `std::fs` (blocking).

```boring
def main() throws:
    fs.write("out.txt", "hello\nworld")
    let content = fs.read("out.txt")        # string
    let lines   = fs.readLines("out.txt")   # [string]

    fs.append("log.txt", "event\n")

    if fs.exists("config.json"):
        let cfg = fs.read("config.json")

    fs.mkdir("data/archive")                # create_dir_all
    let entries = fs.list("data")           # [string] of entry names

    fs.rename("old.txt", "new.txt")
    fs.copy("src.txt", "dst.txt")
    fs.remove("tmp.txt")                    # file or directory tree
```

| Expression | Return type | Throws | Rust (async) |
|---|---|:---:|---|
| `fs.read(path)` | `string` | ✓ | `tokio::fs::read_to_string(path).await?` |
| `fs.readLines(path)` | `[string]` | ✓ | read + `.lines()` collect |
| `fs.readBytes(path)` | `[int]` | ✓ | `tokio::fs::read(path).await?` |
| `fs.write(path, content)` | `void` | ✓ | `tokio::fs::write(path, …).await?` |
| `fs.writeBytes(path, bytes)` | `void` | ✓ | `tokio::fs::write(path, &bytes).await?` |
| `fs.append(path, content)` | `void` | ✓ | `OpenOptions::append(true).open(path).await?` |
| `fs.exists(path)` | `bool` | — | `tokio::fs::metadata(path).await.is_ok()` |
| `fs.isDir(path)` | `bool` | — | `Path::new(path).is_dir()` |
| `fs.isFile(path)` | `bool` | — | `Path::new(path).is_file()` |
| `fs.mkdir(path)` | `void` | ✓ | `tokio::fs::create_dir_all(path).await?` |
| `fs.remove(path)` | `void` | ✓ | `remove_file` or `remove_dir_all` |
| `fs.rename(from, to)` | `void` | ✓ | `tokio::fs::rename(from, to).await?` |
| `fs.copy(src, dst)` | `void` | ✓ | `tokio::fs::copy(src, dst).await?` |
| `fs.list(path)` | `[string]` | ✓ | `tokio::fs::read_dir` + collect names |

### Math

| Function        | Rust equivalent       |
|-----------------|-----------------------|
| `abs(x)`        | `x.abs()`             |
| `min(a, b)`     | `a.min(b)`            |
| `max(a, b)`     | `a.max(b)`            |
| `clamp(v,lo,hi)`| `v.clamp(lo, hi)`     |
| `floor(x)`      | `x.floor()`           |
| `ceil(x)`       | `x.ceil()`            |
| `round(x)`      | `x.round()`           |
| `sqrt(x)`       | `x.sqrt()`            |
| `pow(x, y)`     | `x.powf(y)`           |
| `log(x)`        | `x.ln()`              |
| `log2(x)`       | `x.log2()`            |
| `log10(x)`      | `x.log10()`           |
| `sin(x)`        | `x.sin()`             |
| `cos(x)`        | `x.cos()`             |
| `tan(x)`        | `x.tan()`             |
| `asin(x)`       | `x.asin()`            |
| `acos(x)`       | `x.acos()`            |
| `atan(x)`       | `x.atan()`            |
| `atan2(y, x)`   | `f64::atan2(y, x)`    |

### Math constants (also available as identifiers)

| Boring | Rust                    | Value        |
|--------|-------------------------|--------------|
| `PI`   | `std::f64::consts::PI`  | 3.14159…     |
| `E`    | `std::f64::consts::E`   | 2.71828…     |
| `TAU`  | `std::f64::consts::TAU` | 6.28318…     |

### String methods

| Method                       | Rust equivalent                      |
|------------------------------|--------------------------------------|
| `s.length`                   | `s.len()`                            |
| `s.contains(sub)`            | `s.contains(sub)`                    |
| `s.startsWith(prefix)`       | `s.starts_with(prefix)`              |
| `s.endsWith(suffix)`         | `s.ends_with(suffix)`                |
| `s.trim()`                   | `s.trim().to_string()`               |
| `s.upper()` / `s.toUpper()`  | `s.to_uppercase()`                   |
| `s.lower()` / `s.toLower()`  | `s.to_lowercase()`                   |
| `s.replace(from, to)`        | `s.replace(from, to)`                |
| `s.split(sep)`               | `s.split(sep).collect::<Vec<_>>()`   |
| `s.chars()`                  | `s.chars().collect::<Vec<_>>()`      |
| `s.repeat(n)`                | `s.repeat(n)`                        |
| `s.isEmpty()`                | `s.is_empty()`                       |

### Ownership

| Function | Description |
|----------|-------------|
| `drop(x)` | Explicitly release ownership of `x`. Maps to Rust's `drop()`. In the interpreter this is a no-op (reference-counting handles cleanup automatically). |

```boring
let buf = [1, 2, 3, 4, 5]
drop(buf)                   # free early, before the scope ends
```

**Rust equivalent**
```rust
let buf = vec![1isize, 2, 3, 4, 5];
drop(buf);
```

### JSON serialization — `json` / `fromJson`

`json(v)` serializes any `@derive(Serialize)` value to a JSON string.
`fromJson<T>(s)` parses a JSON string into a `T` that implements `@derive(Deserialize)`.

When either builtin is used, the compiler automatically adds `serde` (with the `derive` and `rc`
features) and `serde_json` to the generated `Cargo.toml`.

```boring
@derive(Serialize, Deserialize)
struct User:
    string name
    int age

let u = User(name = "Alice", age = 30)

let s = json(u)                     # → '{"name":"Alice","age":30}'

let u2 = fromJson<User>(s)         # → User? (nil on parse error)
```

**Rust equivalent**
```rust
#[derive(Clone, Serialize, Deserialize)]
struct User {
    name: Arc<str>,
    age:  isize,
}

let s: String = serde_json::to_string(&u).unwrap_or_default();

let u2: Option<User> = serde_json::from_str::<User>(&s).ok();
```

#### `fromJson` in a `throws` context

In a function declared with `throws`, `fromJson` propagates the parse error automatically:

```boring
task User fetch_user(string url) throws:
    let resp = reqwest.get(url).send().value
    let body = resp.text().value
    let user = fromJson<User>(body)   # throws on parse error
    return user
```

**Rust equivalent**
```rust
async fn fetch_user(url: &str) -> Result<User, Box<dyn Error>> {
    let resp = reqwest::get(url).await?.text().await?;
    let user = serde_json::from_str::<User>(&resp)?;
    Ok(user)
}
```

#### `try?` form

Outside a `throws` context use `try? fromJson<T>(s)` to get `Option<T>`:

```boring
let user = try? fromJson<User>(raw)
if let user:
    print "got {user.name}"
```

| Boring | Context | Rust |
|--------|---------|------|
| `json(v)` | any | `serde_json::to_string(&v).unwrap_or_default()` |
| `fromJson<T>(s)` | `throws` / `try` body | `serde_json::from_str::<T>(&s)?` |
| `fromJson<T>(s)` | plain | `serde_json::from_str::<T>(&s).ok()` |
| `try? fromJson<T>(s)` | any | `serde_json::from_str::<T>(&s).ok()` |

### Debugging

These builtins are available without any import and are designed for interactive debugging.

#### `dbg`

Prints the expression's value to **stderr** with the source file and line number, then returns
the value unchanged.  Usable inline inside any expression.

```boring
let x = dbg(add(3, 4))     # [src/main.rs:12] add(3, 4) = 7
let y = dbg(x * x)         # [src/main.rs:13] x * x = 49
```

```rust
let x = dbg!(add(3, 4));
let y = dbg!(x * x);
```

#### `todo` / `unreachable`

Placeholder panics for unfinished code paths.

```boring
int not_yet(int x):
    todo()                  # panics with "not yet implemented"

int impossible(int x):
    unreachable()           # panics with "entered unreachable code"
```

```rust
fn not_yet(x: isize) -> isize { todo!() }
fn impossible(x: isize) -> isize { unreachable!() }
```

Both accept an optional message argument:

```boring
todo("implement sorting")
unreachable("variant {v} should have been handled above")
```

| Builtin | Rust | Purpose |
|---------|------|---------|
| `dbg(expr)` | `dbg!(expr)` | Print value + location, return value |
| `todo()` | `todo!()` | Mark unfinished code — panics if reached |
| `unreachable()` | `unreachable!()` | Assert a code path is never taken |
| `assert(cond)` | `assert!(cond)` | Runtime assertion — panics on failure |
| `assert(cond, msg)` | `assert!(cond, "{:?}", msg)` | Assertion with message |
| `assert_eq(a, b)` | `assert_eq!(a, b)` | Assert equality |
| `assert_neq(a, b)` | `assert_ne!(a, b)` | Assert inequality |
| `panic()` | `panic!("explicit panic")` | Unconditional panic |
| `panic(msg)` | `panic!("{:?}", msg)` | Panic with message |

---

## 27. Appendix: Boring → Rust Mapping

### Declarations

| Boring                              | Rust                                          |
|-------------------------------------|-----------------------------------------------|
| `let x = v`                         | `let x = v;`                                  |
| `var x = v`                         | `let mut x = v;`                              |
| `let T x = v`                       | `let x: T = v;`                               |
| `let x'q = v`                       | `let x: Q<T> = Q::new(v);` — qualifier on name, type inferred |
| `req R f(T a):  body`               | `fn f(&self, a: T) -> R { body }`             |
| `def R f(T a):  body`               | `fn f(&mut self, a: T) -> R { body }`         |
| `R f(T a):  body`                   | same as `def R f(T a): body` — `def` implicit when return type is present |
| `req R ():  body`                   | `fn __call__(&self) -> R { body }` — anonymous call, callable on `let`/`var` |
| `def ():  body`                     | `fn __call__(&mut self) { body }` — anonymous call, callable on `var` only |
| `set prop(T v): body`               | `fn set_prop(&mut self, v: T) { body }`       |
| `def R f(T a) throws:  body`        | `fn f(&mut self, a: T) -> Result<R, Box<dyn Error>>` |
| `def R f(T a) throws E.T: body`     | `fn f(&mut self, a: T) -> Result<R, E::T>`           |
| `task R f(T a):  body`              | `async fn f(a: T) -> R`                       |
| `task R f(T a):  body`          | same as `task R f(T a): body` — `def` always optional when return type is present |
| `task req R f(T a):  body`          | `async fn f(&self, a: T) -> R`                |
| `task set prop(T v):  body`         | `async fn set_prop(&mut self, v: T)`          |
| `def R f(T vals...):  body`         | `fn f(&mut self, vals: Vec<T>) -> R`          |
| `type N as T`                       | `struct N(T)` + `From<T>`, `From<N>`, `Display` impls |
| `struct S:  fields`                 | `struct S { fields }` + `impl S { ... }`      |
| `S(..base, field = v)`              | `S { field: v, ..base }` — struct update syntax |
| `field` (bare name in method body)  | `self.field` — implicit self; local variables shadow fields |
| `as T:  expr` (in struct body)      | `fn into_t(&self) -> T { expr }` inside `impl S`; if body is `self.field`, also `fn into_t_mut(&mut self) -> &mut T` |
| `enum E:  variants`                 | `enum E { variants }` (always `#[derive(Clone)]`) |
| `trait T:  req methods`             | `trait T { fn methods; }`                     |
| `ext S:  methods`                   | `impl S { methods }`                          |
| `ext S as T:  methods`              | `impl T for S { methods }`                    |
| `def S.f(T a):  body`               | `impl S { fn f(&mut self, a: T) { body } }`   |
| `mod m:  items`                     | `mod m { items }`                             |
| `pub def …`                         | `pub fn …`                                    |
| `@attr(…)`                          | `#[attr(…)]`                                  |

### Types

| Boring            | Alias for       | Rust                                  |
|-------------------|-----------------|---------------------------------------|
| `int`             | `Int'stack`     | `isize`                               |
| `uint`            | `Uint'stack`    | `usize`                               |
| `float`           | `Float64'stack` | `f64` (pure alias of `float64`)       |
| `float32`         | `Float32'stack` | `f32`                                 |
| `float64`         | `Float64'stack` | `f64`                                 |
| `bool`            | `Bool'stack`    | `bool`                                |
| `string` (literal)         | —               | `&str` — zero allocation; promoted to `Rc<str>`/`Arc<str>` when stored |
| `string` (stored/computed) | —               | `Rc<str>` (single-thread) / `Arc<str>` (multi-thread)                 |
| `T`     | `T'stack`   | `T` (stack, Rust default)             |
| `T'`    | `T'heap`    | `Box<T>`                              |
| `T?`    | —           | `Option<T>`                           |
| `T?&`   | —           | `&Option<T>`                          |
| `[T]`      | `Vec<T>`    | `Vec<T>` — dynamic array              |
| `[T, N]`   | —           | `[T; N]` — fixed-size stack array     |
| `{K=V}` | `Dict<K,V>` | `HashMap<K, V>`                       |
| `{T}`   | `Set<T>`    | `HashSet<T>`                          |
| `(T1, T2)`         | —           | `(T1, T2)`                            |
| `T'stack`          | —           | `T` — stack (read only), copied on pass |
| `T'shared`         | —           | `Arc<T>` (multi) / `Rc<T>` (single)  |
| `T'actor`          | —           | `Arc<std::sync::Mutex<T>>` — shared mutable, sync lock |
| `T'actor'task`     | —           | `Arc<tokio::sync::Mutex<T>>` — shared mutable, async lock |
| `T'guard`          | —           | `Arc<std::sync::RwLock<T>>` — concurrent reads, exclusive writes (sync) |
| `T'guard'task`     | —           | `Arc<tokio::sync::RwLock<T>>` — concurrent reads, exclusive writes (async) |
| `T'shared'weak`    | —           | `Weak<T>` — weak ref to `Arc<T>`      |
| `T'actor'weak`     | —           | `Weak<Mutex<T>>` — weak ref to actor  |
| `T'guard'weak`     | —           | `Weak<RwLock<T>>` — weak ref to guard |
| `Index<T>`         | —           | `Option<usize>` (array/set) or `Option<K>` (dict) |
| `Trait` (bare)     | —           | `Box<dyn Trait>` — dynamic dispatch, heap |
| `<Trait>`          | —           | `impl Trait` — static dispatch, no allocation |
| `<uint N>` in generic list | — | `const N: usize` — const generic parameter |
| `<int N>` in generic list  | — | `const N: isize` |
| `<bool B>` in generic list | — | `const B: bool` |

### Expressions

| Boring                     | Rust                                        |
|----------------------------|---------------------------------------------|
| `"Hello, {name}!"`         | `format!("Hello, {}!", name)`               |
| `"""…"""`                  | multi-line string — dedented at lex time, same interpolation rules |
| `{{` / `}}`  in a string   | literal `{` / `}` (both in `"…"` and `"""…"""`) |
| `1_000_000` / `0xFF_AA_00` | digit separator `_` — stripped at lex time, any base |
| `nil`                      | `None`                                      |
| `val else default`         | `val.unwrap_or(default)`                    |
| `expr as T`                | `expr as T` / `expr.parse::<T>().ok()` / `expr.into_t()` (user-defined) / `expr.0` (newtype unwrap) |
| `(expr as T) else default` | `expr.parse::<T>().unwrap_or(default)`      |
| `a === b`                   | `Arc::ptr_eq(&a, &b)` — reference identity (bypasses user-defined `==`) |
| `x += n` / `-=` / `*=` / `/=` / `%=` | `x += n` etc. (numeric types only) |
| `x ?= expr`                           | `x = x.unwrap_or_else(\|\| expr)` — assign if nil |
| `let x = …` (name already in scope)   | `let x = …` — Rust shadowing; type may change freely |
| `1..=5`                     | `1isize..=5`                                |
| `1..4`                    | `1isize..5`                                 |
| `x?.field`                 | `x.map(\|v\| v.field)`                      |
| `task fn(args)`            | `tokio::spawn(async move { fn(args).await })`  |
| `future.value`             | `future.await.unwrap()`                     |
| `future.wait`              | `future.await.unwrap()` (result discarded)  |
| `drop(x)`                  | `drop(x)`                                   |
| `name!(args)`              | `name!(args)` (pass-through)                |
| `f(args) (p): body`        | `f(args, \|p\| body)` — trailing closure, explicit params |
| `f(args) p: body`          | `f(args, \|p\| body)` — trailing closure, no-paren single param |
| `f(args): body`            | `f(args, \|\| body)` — zero-arg trailing body |
| `f(args) expr`             | `f(args, \|\| expr)` — zero-arg trailing body (no separator) |
| `f(args) do (p, q): body`  | `f(args, \|p, q\| body)` — unambiguous `do` trailing, multi-param |
| `f(args) do p: body`       | `f(args, \|p\| body)` — unambiguous `do` trailing, no-paren |
| `f(args) do: body`         | `f(args, \|\| body)` — unambiguous `do` trailing, zero-arg |
| `f(args) do expr`          | `f(args, \|\| expr)` — unambiguous `do` trailing, zero-arg no separator |
| `wait(dur)`                | `tokio::time::sleep(dur).await` |
| `wait(instant)`            | `tokio::time::sleep_until(instant).await` |
| `timeout(dur): f()`        | `tokio::time::timeout(dur, f()).await?` |
| `timeout(instant): f()`    | `tokio::time::timeout_at(instant, f()).await?` |
| `:field` / `:method()`     | `\|x\| x.field` / `\|x\| x.method()`  (closure shorthand) |
| `.Variant`                 | `Enum::Variant` — enum variant shorthand, type inferred from context |
| `.method(args)`            | `Type::method(args)` — static method shorthand, type inferred from context |
| `val \|> f(args)`          | `f(val, args)` (known function) or `val.f(args)` (method) |
| `for x in stream_fn():` | `while let Some(x) = stream.next().await {` (pinned) |
| `channel<T>(n)` / `let T tx, rx = channel(n)` | `tokio::sync::mpsc::channel::<T>(n)` |
| `tx.send(v)` | `tx.send(v).await.unwrap()` |
| `for msg in rx:` | `while let Some(msg) = rx.recv().await {` |
| `oneshot<T>()` | `tokio::sync::oneshot::channel::<T>()` |
| `rx.value` / `rx.recv()` (oneshot) | `rx.await.unwrap()` |
| `broadcast<T>(cap)` | `tokio::sync::broadcast::channel::<T>(cap)` |
| `for msg in rx:` (broadcast) | `while let Ok(msg) = rx.recv().await {` |
| `tx.subscribe()` | `tx.subscribe()` — new broadcast receiver |
| `watch<T>(init)` | `tokio::sync::watch::channel::<T>(init)` |
| `rx.value` (watch) | `rx.borrow().clone()` — current value, no wait |
| `rx.recv()` (watch) | `rx.changed().await.ok(); rx.borrow().clone()` |
| `for val in rx:` (watch) | `while rx.changed().await.is_ok() { let val = rx.borrow().clone();` |
| `tx.send(v)` (oneshot / broadcast / watch) | `tx.send(v).ok()` — non-async |
| `json(v)` | `serde_json::to_string(&v).unwrap_or_default()` — auto-adds `serde`+`serde_json` deps |
| `fromJson<T>(s)` | `serde_json::from_str::<T>(&s).ok()` (plain) / `?` in `throws` context |
| `let f = task: expr` | `let f = tokio::spawn(async move { expr.await })` |
| `f.value` | `f.await.unwrap()` (or `f.await?` in `throws` context) |
| `f.done` | `tokio::time::timeout(Duration::ZERO, f).await.is_ok()` |
| `f.cancel()` | `__cancel_f.cancel()` — signals the spawned task via its token |
| `let a, b = join(f1, f2)` | `let (__jh0, __jh1) = tokio::join!(f1, f2); let a = __jh0.unwrap(); …` |
| `wait dur` | `tokio::time::sleep(dur).await` |
| `task(dur): body` | `tokio::spawn(async move { tokio::time::timeout(dur, async move { body }).await? })` |

### Control flow

| Boring                         | Rust                                   |
|--------------------------------|----------------------------------------|
| `if c: a elif c2: b else d`    | `if c { a } else if c2 { b } else { d }` |
| `if let x = opt:`              | `if let Some(x) = opt {`               |
| `if let x:`                    | `if let Some(x) = x {` (shorthand)     |
| `if let x = a: … elif let y = b: …` | `if let Some(x) = a { … } else if let Some(y) = b { … }` |
| `while let x = expr:`         | `while let Some(x) = expr {`           |
| `guard c else return v`       | `if !c { return v; }`                  |
| `guard let x = opt else return v` | `let Some(x) = opt else { return v; };` |
| `guard let x else return v`    | `let Some(x) = x else { return v; };` (shorthand) |
| `match v:  p: e`               | `match v { p => e, }`                  |
| `for x in coll:`               | `for x in coll.iter().cloned() {`      |
| `for i, v in arr:`             | `for (i, v) in arr.iter().enumerate() {` — auto-enumerate when elements are not tuples |
| `for i, v in arr.enumerate():` | same as above — explicit form still works |
| `for a, b in tuple_arr:`       | `for (a, b) in tuple_arr.iter() {` — tuple destructuring when elements are tuples |
| `for k, v in dict:`            | `for (k, v) in dict.iter() {`          |
| `for k in a..=b:`              | `for k in a..=b {`                     |
| `for k in a..b:`               | `for k in a..b {`                      |
| `for a..=b:`                   | `for _ in a..=b {`                     |
| `loop:`                        | `loop {`                               |
| `let x = loop: break v`        | `let x = loop { break v; }`            |
| `do: … while c`                | `loop { …; if !c { break; } }`         |
| `do: …` (no while)             | `{ … }` (scoped block expression)      |
| `throw "msg"`                  | `return Err("msg".into())`             |
| `try e else d`                 | closure/async block pattern with `error` bound |
| `catch String:`                | typed catch — dispatches on `BoringError` variant |
| `catch String, Int:`           | multi-catch — same handler for multiple types |
| `defer: block`                 | (LIFO cleanup before return)           |

---

## 28. Diagnostics

Boring prints errors in the same format as `rustc` — file path, line number, column, the source line, and a caret pointing at the exact token.

### Error format

```
error: <message>
 --> path/to/file.br:<line>:<col>
  |
N | source line text
  |   ^^^
```

The caret (`^`) spans the width of the offending token. Single-character tokens produce one `^`; identifiers and literals produce as many `^` as the token is wide.

Example — accessing an undefined variable:

```
error: undefined variable 'conter' — did you mean 'counter'?
 --> hello.br:3:7
  |
3 | print conter
  |       ^^^^^^
```

### "Did you mean?"

When an undefined variable name is close to a name in scope, Boring suggests the likely intended name. The suggestion uses edit distance: a name is proposed only if it differs by at most `max(2, len ÷ 3)` characters from the unknown identifier.

### Warnings

Warnings use the same format but print in yellow with a `warning:` prefix instead of `error:`:

```
warning: `BigStruct` is 320 bytes on the stack; consider `BigStruct'heap` to heap-allocate
 --> main.br:12:1
  |
12 | struct BigStruct:
   | ^
```

### Multiple errors

The lexer collects **all** errors in a single pass before stopping, so a file with several bad characters reports each one individually rather than stopping at the first.

Parse errors (unexpected token, unterminated string, bad indentation) and runtime errors (type mismatches, array index out of bounds, division by zero, uncaught throws) all use the same format.

---

## 29. Advanced

This chapter covers features you will rarely need in everyday code. They exist for performance tuning, low-level Rust interop, or unusual architectural patterns. Feel free to skip it until a specific need arises.

---

### Advanced — Strings: `string`

Boring has one user-facing string type — `string` — whose concrete Rust representation is inferred automatically. You should use bare `string` everywhere; the compiler picks the right form.

#### Inference rules

| Context                                                  | Rust type emitted                          |
|----------------------------------------------------------|--------------------------------------------|
| Bare string literal (not stored or compared)             | `&str`                                     |
| Literal stored in a variable, field, or collection       | `Rc::<str>::from("…")` / `Arc::<str>::from("…")` |
| Concatenation or interpolation (`a + b`, `"Hi, {x}!"`)  | `Rc::<str>::from(format!(…))` / `Arc::<str>::from(format!(…))` |
| Already a `Rc<str>` / `Arc<str>` expression             | used as-is                                 |

String literals start as `&str` and are promoted to `Rc<str>` (single-thread) or `Arc<str>` (multi-thread) whenever the context requires heap ownership. This happens automatically — no annotation needed.

#### String representation in tasks

When a string is captured by a task:

| String kind          | Capture strategy                                            |
|----------------------|-------------------------------------------------------------|
| Literal `string`     | `&str` copied directly into the task — zero allocation             |
| Stored `string`      | `Arc::clone` — task and outer scope both keep access, no data copy |

```boring
task string transform(string s):
    "done: {s}"

def main():
    let string label = "hello"      # literal — stored as Rc<str>
    let f = task transform(label)   # Arc::clone — no data copy
    print label                     # hello
    print f.value                   # done: hello
```

### Advanced — Variable shadowing

A `let` or `var` can redeclare a name that is already in scope. The new binding **hides** the previous one — the old value is gone, the new one takes over. The type may change freely:

```boring
let val = 10
let val = "toto"   # shadows the int; val is now a string
print val          # toto

let n = "42"
let n = (n as int) else 0   # parse in place — no temporary name needed
print n            # 42
```

The mutability of the shadow is independent of the original:

```boring
var score = 100
let score = score + 1   # shadow as immutable — score is now let
# score = 0             # error: cannot assign to immutable variable 'score'

let count = 0
var count = count + 1   # shadow as mutable — count is now var
count += 4
print count             # 5
```

Shadowing inside an inner block does not affect the outer binding — this follows the same scoping rules as Rust:

```boring
let x = 7
if true:
    let x = x * 2   # inner shadow
    print x          # 14
print x              # 7 — outer binding unchanged
```

**Rust equivalent** — translates directly to Rust shadowing:
```rust
let val: isize = 10;
let val: &str = "toto";

let n: &str = "42";
let n: isize = n.trim().parse().unwrap_or(0);
```

### Advanced — `transient` fields and `?=` nil-coalescing assignment

These two features are designed to work together for the **lazy-initialisation / cache pattern**: a `transient` field can be written from a `req` (non-mutating) method, and `?=` initialises it only on the first access.

#### `transient` — mutable field from a `req` method

A `transient` field may be written **from a `req` (non-mutating) method**, enabling a lazy cache without making the whole method `def`. Boring compiles `transient` fields to Rust's `std::cell::Cell<T>` (for `Copy` types) or `std::cell::RefCell<T>` (for non-`Copy` types).

```boring
struct TextStats:
    string text
    transient int? _length = nil     # cache for expensive computation

    req int length():
        _length ?= text.length       # compute once, cache; no-op if already set
        _length else 0

let s = TextStats(text = "hello, world")
print s.length    # computes: 12
print s.length    # served from cache: 12
```

Rules:
- A `transient` field is always `var` (mutable) — the `var` keyword is redundant and omitted.
- A `transient` field with a non-`Copy` type (e.g. `string?`) uses `RefCell` in the emitted Rust.
- Reads from outside the struct are plain field accesses — no special syntax.
- `transient` fields are **not** serialised by default (their name conventionally starts with `_`).

**Rust equivalent**
```rust
use std::cell::Cell;

struct TextStats {
    text: Arc<str>,
    _length: Cell<Option<isize>>,
}
impl TextStats {
    fn length(&self) -> isize {
        if let Some(cached) = self._length.get() { return cached; }
        let n = self.text.len() as isize;
        self._length.set(Some(n));   // &self is enough — Cell provides interior mutability
        n
    }
}
```

#### `?=` — nil-coalescing assignment

`x ?= expr` assigns `expr` to `x` **only if `x` is currently `nil`**. If `x` already has a value, the expression is not evaluated and `x` is unchanged. It is equivalent to `x = x else expr`.

```boring
var host = nil
host ?= "localhost"    # nil    → assigns "localhost"
host ?= "example.com"  # non-nil → no-op
print host             # localhost
```

**Rust equivalent** — `x ?= expr` maps to `x = x.unwrap_or_else(|| expr)`:
```rust
host = host.unwrap_or_else(|| Arc::from("localhost"));
```

### Advanced — `lazy` binding

`lazy` is a binding keyword for **deferred, write-once initialisation**. Unlike `let`, which requires a value immediately, a `lazy` binding is declared without a value and must be initialised later with `?=`. After the first `?=`, the binding becomes immutable.

```boring
lazy int x
x ?= compute_value()   # first call — initialises x
x ?= compute_value()   # subsequent calls — no-op
```

#### Binding table (extended)

| Binding | Rebindable | Mutable | Initialisation |
|---|---|---|---|
| `let` | no | no | immediate — `=` required |
| `mut` | no | yes | immediate — `=` required |
| `var` | yes | no | immediate — `=` required |
| `var mut` | yes | yes | immediate — `=` required |
| `lazy` | no | no | deferred — `?=` required, immutable after first assignment |

#### Rules

- **Declaration**: `lazy T name` — no `=` and no value at the declaration site.
- **Initialisation**: `name ?= expr` — evaluates `expr` and assigns it; if already initialised, `expr` is not evaluated and the call is a no-op.
- **Plain `=` is forbidden**: assigning a `lazy` binding with `=` is a compile-time error (transpiler) or a runtime error (interpreter).
- **Qualifiers**: `lazy` is incompatible with `mut` and `var` — the binding is always immutable once set.

#### Typical use — one-time computed constant

```boring
lazy [float] weights
weights ?= load_weights("model.bin")   # loaded once; subsequent calls no-op
```

#### Difference from `transient`

| | `transient` | `lazy` |
|---|---|---|
| Scope | struct field only | any binding (local, global) |
| Write | multiple times | once — immutable after first `?=` |
| From `req` | yes | depends on binding location |
| Rust | `Cell<T>` / `RefCell<T>` | `std::cell::OnceCell<T>` |

`transient` is for mutable caches in struct methods. `lazy` is for deferred constants: it enforces that the value is set once and never changed.

#### Rust equivalent

```rust
let weights: std::cell::OnceCell<Vec<f64>> = std::cell::OnceCell::new();
weights.get_or_init(|| load_weights("model.bin"));
```

### Advanced — Struct spread — `..other`

Pass `..base` as an argument to copy all fields from `base`, then override only the ones you want:

```boring
struct Config:
    init(pub string host, pub int port, pub bool tls)

let defaults = Config(host = "localhost", port = 8080, tls = false)

let prod    = Config(..defaults, host = "prod.example.com", tls = true)
let staging = Config(..defaults, port = 9090)
```

Arguments are resolved left-to-right — later entries override earlier ones. This means:
- `Config(..defaults, host = "prod")` — spread first, then override specific fields ✓
- `Config(host = "prod", ..defaults)` — spread overwrites the earlier `host`

**Rust equivalent** — translates to Rust's struct update syntax:
```rust
let prod = Config { host: "prod.example.com".to_string(), tls: true, ..defaults.clone() };
```

### Advanced — `thiserror` integration

When any enum variant carries an `@error` attribute, the transpiler automatically adds `#[derive(Debug, thiserror::Error)]` to the enum and `thiserror = "1"` to the generated `Cargo.toml`. No `@derive(thiserror::Error)` is needed on the enum itself.

The `@error` message supports thiserror's `{0}`, `{1}` field references and named field syntax.

```boring
enum AppError:
    @error "file not found: {0}"
    NotFound(string)
    @error "permission denied"
    Denied
    @error "io error: {0}"
    Io(string)

riskyOp() throws AppError:
    throw AppError.Denied

def main():
    try:
        riskyOp()
    catch AppError:
        print "caught: {error}"   # error = thiserror's Display output
```

**Rust equivalent**
```rust
#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("file not found: {0}")]
    NotFound(String),
    #[error("permission denied")]
    Denied,
    #[error("io error: {0}")]
    Io(String),
}
```

> **Catching errors from native Rust functions** — `catch TheirErrorType:` also works for
> errors propagated via `?` from native Rust functions called in a `throws` context. This lets
> you integrate with any third-party Rust library that returns `Result<T, E>` where
> `E: std::error::Error`.

### Advanced — Compatibility with Rust `Result` types

Boring's `throw`/`catch` system wraps errors in `BoringError` at runtime. Rust functions that return `Result<T, E>` (where `E` is not `BoringError`) integrate differently depending on how you consume them:

| Syntax | Works with `throws` | Works with `def Result<T,E>` | Notes |
|---|:---:|:---:|---|
| `try f() else d` | ✓ | ✓ | any mix of inline/block bodies; `error` bound in else |
| `try? f()` | ✓ | ✓ | emits `.ok()` — `Result<T,E>` → `Option<T>`; no `error` binding |
| `try: block else …` | ✓ | ✓ | block try body; same `error` binding rules |
| `try:` … `catch:` (untyped) | ✓ | ✓ | statement; `error` bound as string via `to_string()` |
| `try:` … `catch String e:` (typed) | ✓ | ✗ | typed `catch` dispatches via `BoringError` downcast — does not match errors from `def Result<T,E>` functions |
| `if let Ok(v) = f():` | — | ✓ | idiomatic pattern for `def Result<T,E>` |
| `if let Err(e) = f():` | — | ✓ | idiomatic pattern for `def Result<T,E>` |

**Rule of thumb:** use `throw`/`catch` for Boring-defined error logic; use `if let Ok`/`Err` or `try?` for functions that return `Result<T, E>` directly.

### Advanced — Variadic parameters

The last parameter can collect remaining arguments with `...`:

```boring
int sum(int values...):
    var total = 0
    for v in values:
        total = total + v
    total

sum(1, 2, 3, 4, 5)    # 15
```

**Rust equivalent**
```rust
fn sum(values: Vec<isize>) -> isize {
    values.iter().sum()
}
sum(vec![1, 2, 3, 4, 5]);
```

### Advanced — Error handling internals

This section describes the generated Rust representation for Boring's `throw`/`catch` system.

#### Propagation — `BoringError` and `?`

Every `throws` function returns `Result<T, Box<dyn std::error::Error>>`. String-literal throws are wrapped in `BoringError::Str` (zero allocation); interpolated-string throws use `BoringError::String`. The Boring compiler inserts `?` automatically on every call to a `throws` function from inside another `throws` function, so errors propagate up the call stack without explicit forwarding:

```rust
fn parse_int(s: Arc<str>) -> Result<isize, Box<dyn std::error::Error>> {
    if !(s.len() as isize > 0) {
        return Err(Box::new(BoringError::Str("empty string")));
    }
    let Some(n) = s.trim().parse::<isize>().ok() else {
        return Err(Box::new(BoringError::String(Arc::<str>::from(format!("not a number: {}", s)))));
    };
    Ok(n)
}

fn double_parse(s: Arc<str>) -> Result<isize, Box<dyn std::error::Error>> {
    Ok(parse_int(s.clone())? * 2)   // ? inserted by the compiler
}

fn process(s: Arc<str>) -> Result<Arc<str>, Box<dyn std::error::Error>> {
    let n = double_parse(s.clone())?;
    Ok(Arc::<str>::from(format!("result: {}", n)))
}
```

#### Typed errors — `BoringError::Other` and `TypeId`

When a function declares `throws CalcError`, typed errors are wrapped in `BoringError::Other` with a `TypeId` so `catch CalcError:` can dispatch correctly at the catch site. The return type remains `Result<T, Box<dyn Error>>` in both typed and untyped cases.

```rust
fn checked_divide(a: isize, b: isize) -> Result<isize, Box<dyn std::error::Error>> {
    if !(b != 0) {
        return Err(Box::new(BoringError::Other(
            std::any::TypeId::of::<CalcError>(),
            Box::new(CalcError::DivByZero),
        )));
    }
    if !((a / b) < 1000000) {
        return Err(Box::new(BoringError::Other(
            std::any::TypeId::of::<CalcError>(),
            Box::new(CalcError::Overflow),
        )));
    }
    Ok(a / b)
}

let r1 = checked_divide(10, 2).unwrap_or_else(|_| -1);   // 5
let r2 = checked_divide(10, 0).unwrap_or_else(|_| -1);   // -1
```

The `TypeId` uniquely identifies `CalcError` at the catch site regardless of module — two types with the same name in different modules are never confused.

| | Untyped `throws` | Typed `throws CalcError` |
|---|---|---|
| Return type | `Result<T, Box<dyn Error>>` | `Result<T, Box<dyn Error>>` |
| Throw wrapping | `BoringError::Str` / `BoringError::String` | `BoringError::Other(TypeId::of::<CalcError>(), …)` |
| `catch:` (untyped) | ✓ catches everything | ✓ catches unmatched errors |
| `catch CalcError:` | ✗ | ✓ dispatches via `TypeId` |
| `try … else` | ✓ | ✓ |
| Propagate with `?` | ✓ | ✓ (unwrapped at the catching `try:` block) |

#### Fixed-width scalar throws — `BoringError::Scalar`

`int8`..`int128`, `uint8`..`uint128`, `float32`, and `float64` route through their own `BoringError::Scalar(ScalarKind, u128)` variant, not `BoringError::Other`. The compiler already knows, statically, that a thrown value here is exactly one of twelve small `Copy` kinds — `Other`'s `TypeId` + heap-allocated `Box<dyn Any>` machinery exists for arbitrary user-defined enums/structs the compiler *can't* enumerate in advance, and paying that cost for a plain `i8` or `f32` would be pure overhead.

```boring
def risky(bool asFloat) throws:
    if asFloat:
        let float32 f = 3.5
        throw f
    let int8 x = 5
    throw x
```

```rust
fn risky(as_float: bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if as_float {
        let f: f32 = 3.5;
        return Err(Box::new(BoringError::scalar_f32(f)));
    }
    let x: i8 = 5;
    return Err(Box::new(BoringError::scalar_i8(x)));
}
```

`catch Int8:` / `catch Float32:` each compile to a guard on the `ScalarKind` tag, binding `error` as the real native type (`i8`, `f32`, …) reconstructed from the shared `u128` payload — not a stringified `Arc<str>` the way the untyped `Int`/`Float`/`Bool` fast paths do:

```rust
BoringError::Scalar(ScalarKind::Int8, __bits) => { let error: i8 = __bits as i128 as i8; /* body */ }
BoringError::Scalar(ScalarKind::Float32, __bits) => { let error: f32 = f32::from_bits(__bits as u32); /* body */ }
```

`Other` still exists, unchanged, for exactly what only it can do — genuine user-defined `throws ErrorType` enums/structs.

#### Qualified error type paths

The error type in a `throws` clause can be a **module-qualified path** using dot notation — the dot separator is translated to `::` in Rust:

```boring
string read_file(string path) throws io.Error:
    guard path != "" else throw io.Error.NotFound
    "content of {path}"
```

```rust
fn read_file(path: Arc<str>) -> Result<Arc<str>, io::Error> { ... }
```

Any depth of qualification is supported: `throws a.b.c.Error` → `a::b::c::Error`.

---

### Advanced — Macros

Boring supports calling Rust macros directly with `!`:

```boring
eprintln!("debug: value = {}", 42)   # stderr — no Boring built-in for this
assert!(2 + 2 == 4)
assert_eq!(add(2, 3), 5)
let msg = format!("{} + {} = {}", 1, 2, add(1, 2))
```

All three call forms are supported:

```boring
name!(args)    # parentheses
name![args]    # brackets
name!{args}    # braces
```

**Rust equivalent** — passed through verbatim:
```rust
println!("value = {}", 42);
assert!(2 + 2 == 4);
assert_eq!(add(2, 3), 5);
let msg = format!("{} + {} = {}", 1, 2, add(1, 2));
```

---

## 30. Qualifier Inference

Boring's ownership qualifiers (`'stack`, `'heap`, `'shared`, `'actor`, `'guard`) describe how a value is stored and shared at runtime. In most code you never write them — the compiler infers the right one from how each variable is used. This chapter explains the full inference system.

### The zero-annotation goal

Boring aims to let you write code that looks like a high-level scripting language while emitting Rust that is as precise and zero-cost as hand-written systems code. The qualifier system is the bridge: it maps naturally to `T`, `Box<T>`, `Arc<T>`, `Arc<Mutex<T>>`, and `Arc<RwLock<T>>` without forcing you to spell those types out.

```boring
let c = Counter(0)      # qualifier not written — inferred from use
spawn_actor(c)          # demands 'actor → c inferred as Arc<Mutex<Counter>>
```

The emitted Rust is identical to what you would write by hand, but you never had to decide.

### Constraint elimination

Each unqualified local variable starts as a candidate for every qualifier: `{Stack, Owned, Shared, Actor, Guard}`. Every usage signal narrows the set by eliminating qualifiers that are incompatible. When exactly one candidate remains, it is chosen. When none remain, the constraints are contradictory and the compiler reports an error. When several remain, the compiler applies a priority-ordered fallback (see below).

| Signal | Compatible qualifiers |
|---|---|
| Call site demanding `T'shared` | `{Shared}` |
| Call site demanding `T'actor` | `{Actor}` |
| Call site demanding `T'guard` | `{Guard}` |
| Call site demanding `T'stack` | `{Stack}` |
| Call site demanding `T'heap` | `{Owned}` |
| `def` method call on the variable | `{Stack, Owned, Actor, Guard}` |
| `mut` binding (`mut x = ...`) | `{Stack, Owned, Actor, Guard}` |
| Task capture as method receiver | `{Actor, Guard}` |
| Task capture, read-only | `{Shared, Actor, Guard}` |
| `req` method call | *(no constraint — all qualifiers remain)* |

Each signal intersects the current candidate set. The order of signals does not matter.

### Priority-ordered fallback

When the candidate set still contains multiple qualifiers after all signals are applied, the transpiler resolves the tie using the following algorithm:

**Step 1 — `'stack` candidate.** If `'stack` is in the candidate set:

| Context | Decision |
|---|---|
| Struct field (any binding — `let`, `mut`, `var`) | `'stack` — field bytes are part of the parent allocation; no indirection regardless of size |
| Local variable, sizeof(T) ≤ `--stack-auto-bytes` | `'stack` |
| Local variable, sizeof(T) > `--stack-auto-bytes` | skip `'stack`; continue with step 3 |

All bare-T struct fields suppress size-based auto-boxing. This applies to `var` fields as well as `let`/`mut` fields — a `var` field is still stored in-place inside the struct. Boxing it would add unnecessary indirection and fragment the allocation. `T'` fields are not affected: an indirection hint always produces `Box<T>`.

```boring
struct Wrapper:
    let BigData inner    # bare T — always inline regardless of size
    var BigData heavy    # var field — also inline, no size-based Box
    let BigData' backup  # T' — always Box<BigData>
```

**Step 2 — ordered chain.** If `'stack` was not selected, the transpiler picks the first qualifier present in the remaining candidates, in this order:

`'heap` > `'shared` > `'actor` > `'guard`

### Threshold

The stack size threshold is configurable:

```
boring build --stack-auto-bytes 512  # promote locals > 512 bytes to Box<T>
```

Default: 256 bytes.

> The size estimate is best-effort: it sums struct fields recursively but treats `Vec`, `HashMap`, and pointer-sized types as 8–16 bytes. The estimate is conservative — when in doubt the transpiler prefers `'heap` over a potentially large stack frame.

### Example: sharing + mutation → conflict

```boring
let c = Counter(0)
c.inc()             # def call → eliminates Shared → {Stack, Owned, Actor, Guard}
share_with(c)       # demands 'shared → intersect to {Shared}
                    # result: {} → error: no qualifier satisfies all constraints
```

This is correct: `Arc<T>` does not support `.inc()` (a mutating method). The developer must choose: either pass `c` to `share_with` only, or use `T'guard` explicitly to get `Arc<RwLock<T>>`.

### Return-type demand

If the function has a declared return type with a qualifier, a bare variable in tail position or in a `return` statement inherits that qualifier.

```boring
Counter'actor make_counter():
    let c = Counter(0)   # anonymous
    c                    # tail expression → infers 'actor from return type
```

### Alias propagation

`let y = x` makes `y` an alias of `x`. Any constraint applied to either one is propagated to the whole group.

```boring
let c = Counter(0)
let d = c               # d is an alias of c
spawn_actor(d)          # demands 'actor on d → also applied to c
```

### Task captures

Variables captured by a `task:` body must be `Arc`-based so they can be cloned and moved into the async closure.

- Captured as a method receiver (e.g. `c.inc()` inside the task) → `{Actor, Guard}`
- Captured read-only → `{Shared, Actor, Guard}`

```boring
let c = Counter(0)
task:
    c.inc()    # c is a receiver inside the task → infers 'actor
```

### Parameter auto-apply

Parameters without an explicit qualifier, with a tick (`T'`), or with a qualifier group (`T'mut`, `T'many`, …) are all subject to body inference. The inferred qualifier is applied automatically at emission — the Rust function signature carries the correct type even if the Boring source does not.

```boring
def process(Counter c):     # bare — full candidate set
    spawn_actor(c)          # demands 'actor → fn process(c: Arc<Mutex<Counter>>)

def process(Counter' c):    # tick — {Owned, Shared, Actor, Guard}
    spawn_actor(c)          # demands 'actor → fn process(c: Arc<Mutex<Counter>>)

def process(Counter'mut c): # group — {Stack, Owned, Actor, Guard}
    spawn_actor(c)          # demands 'actor → fn process(c: Arc<Mutex<Counter>>)
```

If the body provides no narrowing signal, the fallback for bare `T` is size-based, for `T'` is `'heap`, and for a group is the first member of the group.

### Universal borrow as inference output

When a bare parameter has no storage signal and no qualifier demand signal, the inference resolves to a **universal borrow** — `Counter&` or `mut Counter&` — as a pre-fallback step, before the size-based chain.

```boring
req display(Counter c):   # no storage, read-only → infers Counter& → fn display(c: &Counter)
    print c.value

def reset(mut Counter c): # no storage, mut declared → infers mut Counter& → fn reset(c: &mut Counter)
    c.value = 0
```

Callers can pass any qualifier. The transpiler acquires the lock at the call site for `'actor` and `'guard` arguments, exactly as for an explicit `Counter&` parameter.

**Mutability is declared, not inferred.** `mut` on a parameter is always written explicitly; it is not inferred from the body. A `def` method call on an immutable parameter is a compile error (`declare mut Counter n`).

A **storage signal** (field assignment, task capture, return with ownership qualifier, or field-destructuring pattern such as `guard let Some(x) = n.field` / `if let Some(x) = n.field` / `let x = n.field`) or a **qualifier demand signal** (passed to a function expecting a specific qualifier) causes the inference to fall back to the normal constraint-elimination path instead.

| Parameter | Signals | Emitted form |
|---|---|---|
| `Counter c` | none | `&Counter` |
| `mut Counter c` | none | `&mut Counter` |
| `Counter c` | qualifier demand | concrete qualifier |
| `Counter c` | storage | concrete qualifier |

The same rule applies to generic parameters: `T c` without signals infers `&T`; `mut T c` infers `&mut T`. Optionals (`Counter? c`), tick parameters (`Counter' c`), `var` parameters, explicit qualifier groups, and **struct/enum method parameters** are excluded from universal borrow inference. The explicit forms `Counter& c` and `mut Counter& c` lock in the behavior regardless of future body changes — and are the only way to get universal borrowing in a method parameter.

### Cross-function propagation

After a function body is processed, `fn_sigs` is updated with the inferred parameter qualifiers. Functions defined later in the same file that call this function will see the qualified signature and propagate the constraint to their own variables.

```boring
def process(Counter c):   # inferred 'actor from body
    spawn_actor(c)

let c = Counter(0)
process(c)                # fn_sigs now shows Counter'actor → c infers 'actor
```

**Limitation:** propagation is a single forward pass. If a caller is defined before the callee in the file, no propagation occurs. Mutual recursion is not covered.

### Struct field inference

Private and public fields with no explicit qualifier are inferred from the struct's own method bodies using the same constraint-elimination algorithm. Only fields of struct types (`Named` types) are candidates.

```boring
struct Service:
    Counter stats        # no qualifier

    def record():
        spawn_actor(stats)   # demands 'actor
        # inferred: stats: Arc<Mutex<Counter>>
```

Results are applied at emit time: the field is emitted with the inferred Rust wrapper, and all `self.stats` accesses in method bodies are wrapped or unwrapped accordingly. Public fields are resolved from internal usage only — external callers must work with the inferred type. Generating per-qualifier struct variants would leak implementation details through the module boundary.

**Limitation:** cross-file inference is not supported. A field accessed only from another module retains its fallback type.

### The `mut` binding keyword

`mut x = expr` marks a fixed binding with a mutable instance. It contributes a mutation signal at the declaration site — equivalent to a `def` method call on the same line, but earlier. This allows the compiler to narrow the candidate set before any method calls are seen.

```boring
mut c = Counter(0)      # eliminates Shared → {Stack, Owned, Actor, Guard}
spawn_actor(c)          # demands 'actor → infers 'actor
```

### Explicit annotation as escape hatch

When inference cannot resolve a unique qualifier — for example when a variable is used in two mutually exclusive ways — you annotate explicitly:

```boring
let Counter'guard c = Counter(0)   # developer decides: RwLock
```

An explicit qualifier has the highest priority and overrides all inference signals. The compiler will still validate that the declared qualifier is compatible with the usage in the body.

### `T'` — the indirection hint

`T'` (a type followed by a lone tick) signals that the value must not live on the stack, but leaves the exact kind of indirection to the inference pass. It restricts the initial candidate set to `{Owned, Shared, Actor, Guard}`, eliminating `Stack` from the start.

```boring
let c' = Counter(0)        # tick → candidates: {Owned, Shared, Actor, Guard}
spawn_actor(c)             # demands 'actor → {Actor} → emits Arc<Mutex<Counter>>
```

If no signal further constrains the set, the fallback is `'heap` (`Box<T>`):

```boring
let c' = Counter(0)        # tick, no further signal → Box<Counter>
```

This is distinct from the plain `T` fallback, which is `'stack` for small types. `T'` is the right form when you know a value should live on the heap or be shared, but the specific qualifier depends on how it is used.

### Optional forms — `T?` and `T'?`

Optional variables (`T?` and `T'?`) participate in inference the same way as their non-optional counterparts. The inferred qualifier is applied to the **inner type** of the `Option`, not to the `Option` itself.

```boring
let c? = some(Counter(0))   # T? → full candidate set
spawn_actor(c?)             # demands Counter'actor → infers 'actor
# emits: Option<Arc<Mutex<Counter>>>
```

```boring
let c'? = some(Counter(0))  # T'? → restricted set {Owned, Shared, Actor, Guard}
                             # no signal → fallback → Option<Box<Counter>>
```

`T?` with inferred `'actor` becomes `Option<Arc<Mutex<Counter>>>` — the `Option` wraps the qualified value, not the other way around. Conflict detection works the same as for bare variables.

## 32. Debugging & Profiling

Boring provides several layers of debugging support, from language-level builtins to build flags that activate Rust's own tooling.  All flags are orthogonal and composable.

| Tool | Activation | What it does |
|------|------------|--------------|
| `dbg`, `todo`, `unreachable`, `assert` | language builtins | Inspect values and guard invariants at the source level |
| `--mode managed` | build flag | Automatic backtraces + `#[track_caller]` on every function |
| `--sanitize` | build flag | Detect memory errors and data races via Rust sanitizers |
| `--instrument` | build flag | Per-function call counts and wall-clock timings |
| `--compile` | build flag | Transpile then immediately invoke `cargo build` |
| `--rust-options` | build flag | Pass extra flags to `cargo build` (implies `--compile`) |

---

### 32.1 Language builtins

These builtins are available without any import.  They are described in full in [§26 Built-in Functions](#26-built-in-functions); this section summarises the debugging-specific ones.

| Builtin | Rust | Purpose |
|---------|------|---------|
| `dbg(expr)` | `dbg!(expr)` | Print `[file:line] expr = value` to stderr; returns value |
| `todo()` | `todo!()` | Panic placeholder for unfinished code paths |
| `unreachable()` | `unreachable!()` | Assert a code path is never taken |
| `assert(cond)` | `assert!(cond)` | Runtime assertion — panics on failure |
| `assert(cond, msg)` | `assert!(cond, "{:?}", msg)` | Assertion with message |
| `assert_eq(a, b)` | `assert_eq!(a, b)` | Assert equality |
| `assert_neq(a, b)` | `assert_ne!(a, b)` | Assert inequality |
| `panic()` / `panic(msg)` | `panic!(…)` | Unconditional panic |

`dbg` is usable inline inside any expression and returns its argument unchanged:

```boring
let x = dbg(add(3, 4))     # stderr: [src/main.rs:12] add(3, 4) = 7
let y = dbg(x * x)         # stderr: [src/main.rs:13] x * x = 49
```

---

### 32.2 `--mode managed` — enhanced error reporting

When you build with `--mode managed`, two behaviours are activated automatically — no source changes required.

#### Automatic backtraces

A `.cargo/config.toml` is written into the generated project:

```toml
[env]
RUST_BACKTRACE = "1"
```

Every panic prints a full stack trace without having to set any environment variable by hand.

#### `#[track_caller]` on all functions

Every function and method receives `#[track_caller]`.  When a function panics (via `assert`, `todo`, `unreachable`, or `panic`), Rust reports the **call site** rather than the panic site deep inside the standard library.

```boring
def assert_bigger(int a, int b):
    assert a > b, "expected {a} > {b}"

assert_bigger(3, 10)   # ← this line appears in the panic message
```

Combined with `RUST_BACKTRACE=1`, panics in managed mode give you the full picture immediately.

```sh
boring build --mode managed main.br
cd main_rust && cargo run
```

---

### 32.3 `--sanitize` — memory and concurrency error detection

Enables a Rust sanitizer in the generated Cargo project.  A `.cargo/config.toml` is written with the appropriate `rustflags` and the host target triple (detected automatically via `rustc --version --verbose`):

```toml
[build]
rustflags = ["-Zsanitizer=address"]
target    = "aarch64-apple-darwin"
```

> **Requires a nightly toolchain.**  Run the generated project with `cargo +nightly run`.

```sh
boring build --sanitize address main.br   # heap/stack overflows, use-after-free
boring build --sanitize thread  main.br   # data races
boring build --sanitize memory  main.br   # reads of uninitialised memory
```

| Sanitizer | Detects |
|-----------|---------|
| `address` | Heap buffer overflow, stack buffer overflow, use-after-free, use-after-return |
| `thread` | Data races — concurrent reads/writes without synchronisation |
| `memory` | Use of uninitialised memory |

When `--mode managed` and `--sanitize` are both active, the generated `.cargo/config.toml` includes both the `[env]` and `[build]` sections.

---

### 32.4 `--instrument` — call counts and wall-clock profiling

`boring build --instrument` adds zero-dependency, RAII-based profiling to every function.  No changes to Boring source files are needed.

```sh
boring build --instrument main.br
```

An inline `__boring_instrument` module is prepended to the Rust output — no extra `Cargo.toml` dependency.  Every function body receives a `Span` guard:

```rust
fn add(a: isize, b: isize) -> isize {
    let _boring_span = __boring_instrument::Span::enter("add");
    // … body …
}
impl Greeter {
    fn hello(&self) -> Arc<str> {
        let _boring_span = __boring_instrument::Span::enter("Greeter::hello");
        // … body …
    }
}
```

`Span` fires on normal return, early return, and panic unwind alike.  `main` additionally receives a `DumpGuard` that flushes results to disk even when a panic propagates to the top level:

```rust
fn main() {
    let _boring_dump = __boring_instrument::DumpGuard;   // dropped last → writes files
    let _boring_span = __boring_instrument::Span::enter("main");
    // …
}
```

> **Note** — `panic = "abort"` bypasses all destructors; the output files are not written on abort.  The default (`panic = "unwind"`) is unaffected.

#### Output files

Two files are written in the current working directory on program exit:

**`boring_coverage.json`** — per-function aggregated statistics, sorted alphabetically:

```json
{
  "Greeter::hello": {"calls": 1, "total_us": 66, "avg_us": 66},
  "add":            {"calls": 1, "total_us": 0,  "avg_us": 0},
  "fact":           {"calls": 6, "total_us": 24, "avg_us": 4},
  "main":           {"calls": 1, "total_us": 829,"avg_us": 829}
}
```

| Field | Description |
|-------|-------------|
| `calls` | Number of times the function was called |
| `total_us` | Cumulative wall-clock time across all calls (µs) |
| `avg_us` | Average wall-clock time per call (µs) |

**`boring_trace.json`** — all calls in [Chrome Trace Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/), directly openable in **Perfetto** (`ui.perfetto.dev`) and **Speedscope** (`speedscope.app`):

```json
{
  "traceEvents": [
    {"name":"add", "ph":"X","ts":1781467928688709,"dur":0,  "pid":1,"tid":1,"cat":"boring"},
    {"name":"fact","ph":"X","ts":1781467928689060,"dur":14, "pid":1,"tid":1,"cat":"boring"},
    {"name":"main","ph":"X","ts":1781467928688708,"dur":383,"pid":1,"tid":1,"cat":"boring"}
  ],
  "displayTimeUnit": "ms"
}
```

| Field | Value |
|-------|-------|
| `name` | Function name (`Type::method` for methods) |
| `ph` | `"X"` — complete event (timestamp + duration in one record) |
| `ts` | Call entry timestamp — microseconds since Unix epoch |
| `dur` | Wall-clock duration of this specific call (µs) |
| `cat` | `"boring"` — filterable category in Perfetto |

To open in Perfetto: `ui.perfetto.dev` → *Open trace file* → `boring_trace.json`.  
To open in Speedscope: `speedscope.app` → drag and drop `boring_trace.json`.

---

### 32.5 `--compile` and `--rust-options` — single-pass build

By default `boring build` only generates the Cargo project — you then run `cargo build` yourself.  `--compile` fuses the two steps:

```sh
boring build --compile main.br          # transpile + cargo build (debug)
boring build --compile                  # same, using boring.toml
```

`--rust-options "<flags>"` passes extra arguments verbatim to `cargo build` and implies `--compile`:

```sh
boring build --rust-options "--release" main.br
boring build --rust-options "--release --features tracing"
boring build --rust-options "--release"   # project-mode via boring.toml
```

The flags are split on whitespace so a single quoted string covers multiple cargo arguments.

`--rust-options` is independent of `--sanitize`.  To enable a sanitizer and build in one step:

```sh
boring build --sanitize address --rust-options "+nightly build" main.br
```

Or keep the two steps separate when nightly is required — `boring build --sanitize` writes `.cargo/config.toml` and then you run `cargo +nightly run` manually.

### 32.6 Combining all debug tools

All flags are independent and composable:

```sh
boring build \
    --mode managed \
    --sanitize address \
    --instrument \
    main.br
cd main_rust && cargo +nightly run
```

With `--compile` the last two lines collapse into one when a sanitizer is not needed:

```sh
boring build \
    --mode managed \
    --instrument \
    --compile \
    main.br
```

This single command produces a binary that:
- Prints full stack traces on every panic (`RUST_BACKTRACE=1`)
- Reports the exact call site on panics (`#[track_caller]`)
- Writes `boring_coverage.json` + `boring_trace.json` on exit

| Flag | Cargo.toml dep | `.cargo/config.toml` | Toolchain |
|------|---------------|----------------------|-----------|
| `--mode managed` | — | `[env] RUST_BACKTRACE=1` | stable |
| `--sanitize` | — | `[build] rustflags + target` | **nightly** |
| `--instrument` | — | — | stable |
| `--compile` | — | — | stable |
| `--rust-options` | — | — | stable |

---

## Further Reading

The following documents cover topics in greater depth or address areas still under active design.

### Ownership and qualifiers

**[Qualifiers — Complete Reference](qualifiers.html)**
All ownership qualifiers (`'stack`, `'heap`, `'shared`, `'actor`, `'guard`, `'weak`): semantics, Rust mapping, thread-safety, move semantics, qualifier upgrade coercions (`'stack`→`'heap`→`'shared`→`'actor`), parameter passing, zero-annotation inference algorithm, and known limitations.

**[Binding and mutability](binding-mutability.html)**
Deep dive into the three binding forms (`let` / `mut` / `var`), their interaction with qualifiers, and how they map to Rust's ownership and mutability model.

**[`new` placement operator](new-placement.html)**
Explicit placement syntax for arena, heap, and GPU device allocators — `new(arena) T(...)`. Covers qualifier interaction, GPU device placement, and the full inference override rules.

**[Scoped Access Blocks — `with`](scoped-access-blocks.html)**
Full design and implementation notes for `with` (see [chapter 21](#scoped-access-blocks--with) above for the language reference, including [residency across a function boundary](#residency-across-a-function-boundary)). The `'actor`/`'guard` per-block locking half, the same-scope `'gpu'unified`/`'gpu'global` kernel-field materialization half, and the inter-procedural case (a resident value returned across a function boundary and chained into a further call) are all implemented and shipped. This document records the full design, exactly what the generated Rust looks like, and current known limitations (cuda/metal targets don't share this yet).

### GPU computing

**[GPU computing — language reference](gpu-module.html)**
Complete GPU language reference: `kernel` struct syntax, const generic kernel declarations (`kernel Blur<int N>:`), memory qualifiers and inference rules, dispatch, the `GPU` type, atomics, multi-device dispatch, `after =` ordering, and simulation mode and profiles.

**[CUDA backend](cuda-module.html)**
CUDA codegen internals: generated file layout, `cudarc` host API, PTX compilation via `build.rs`, Boring construct to CUDA C mapping, and known limitations.

**[ROCm backend](rocm-backend.html)**
AMD GPU target (`boring build --target rocm`): HIP C++ device codegen (near-identical to CUDA C), hand-rolled HIP FFI host API (no `cudarc`-equivalent crate exists for ROCm), `hipcc --genco` code-object compilation, and limitations vs CUDA.

**[Metal backend](metal-backend.html)**
macOS GPU target (`boring build --target metal`): qualifier → MSL address space mapping, built-in substitution, runtime MSL compilation via `newLibraryWithSource`, limitations vs CUDA.

**[wgpu backend](wgpu-backend.html)**
Cross-platform GPU target (`boring build --target wgpu`): runs on DirectX 12 (Windows), Vulkan (Windows / Linux), and Metal (macOS) — no NVIDIA GPU or external toolchain required. Covers qualifier → WGSL mapping, pipeline overrides for dynamic workgroup size, generic kernel monomorphisation, and limitations vs CUDA.

**[GPU display](gpu-display.html)**
Live GPU rendering to a native OS window: `'surface` pixel buffer, `Screen` object, and `kernel: loop:` render loop. Metal implemented; CUDA and simulation pending.

**[Warp-level primitives](warp-level-primitives.html)**
`gpu.warp.*` — warp/wavefront/SIMD-group/subgroup built-ins (`size`, `lane`, `sync()`, `shuffle_down/up/xor/shuffle`) for intra-warp reductions that skip the shared-memory round-trip and block barrier a full `sync` costs. Per-backend mapping, the wgpu real-subgroup/shared-memory-emulated fallback split, and the divergent-branch mask caveat.

### Compilation targets

**[Rust-for-Linux target](kernel-target.html)**
`boring build --target kernel`: type mapping in `no_std`, errno-based error handling, `task def` as workqueue work items, const-generic and heap-allocated channels, sequential and async streams, and the kernel emission backend architecture.

**[Transpilation Modes](transpilation-modes.html)**
The two orthogonal flags (`--threading` and `--mode`) that control memory management and concurrency in the standard Rust backend: `single` vs `multi`, `managed` vs `strict`.

### Language internals

**[Interpreter self-hosting plan](interpreter-port-plan.html)**
How and why the Boring interpreter was rewritten in Boring itself (`boring/interpreter/`), the porting strategy, and the current implementation status.

### Design drafts and exploration

**[Library Distribution Model](library-distribution.html)**
Boring transpiles to Rust — dependency management and distribution are handled by Cargo on the generated project. A Boring-native package manager is out of scope: source files remain private, only the generated Rust is shipped.
