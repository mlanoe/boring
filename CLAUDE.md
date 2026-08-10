# Boring — language quick reference for Claude

Boring is a high-level language that transpiles to Rust. Source files use `.br` extension.

## Core syntax rules

- Type is written **before** the name: `let int x = 42`
- Qualifier is written **after** the type: `let int'stack x = 42`
- Qualifier after variable name when type is inferred: `let x'stack = 42`
- Indentation-based blocks (Python-style), no braces
- `#` for comments

## Variables

```boring
let x = 42              # immutable binding, immutable instance
var x = 42              # rebindable, NOT content-mutable (mut ≠ implied)

struct Counter:
    var int value = 0
mut c = Counter(0)      # fixed binding, content-mutable instance
var mut c2 = Counter(0) # rebindable AND content-mutable
```

`mut`/`var mut` attach to the **type**, not just the binding keyword (see
[mut-type-modifier.md](docs/mut-type-modifier.md)) — this is what lets `mut`
compose into tuple slots, struct fields, array/dict elements, and borrows
(`mut Type&`), not just a bare local. `mut` on a scalar (`mut int x`) is a
checker error: primitives have no `def` methods for it to unlock — use `var`
for a rebindable scalar.

## Functions

```boring
def greet(string name):         # void, def required
    print "Hello, {name}!"

int add(int a, int b):          # return type present, def optional
    a + b

def int add(int a, int b):      # explicit def, equivalent
    a + b

def mut T foo():                # returns a mutable instance
req mut T get():                # read-only method, returns a mutable instance
```

- `req` — read-only method, callable on `let` and `var` bindings → `&self`
- `def` — mutating method, callable on `var` bindings only → `&mut self`
- `def mut` / `req mut` — the `mut` after the keyword applies to the **return value**, not `self`

### Parameter passing

- **Structs, enums, arrays, dicts, sets** — always passed by reference (`&T`) automatically. Never write `&`. The caller keeps ownership.
- **Primitives** (`int`, `float`, `bool`, `uint`) — passed by value (Copy types), no overhead.
- **`var` param** — passes `&mut T`; changes are visible at the call site.
- **`T&` explicit borrow** — advanced only; needed only for explicit lifetime annotations.

## Structs

```boring
struct Point:
    float x
    float y

struct Counter:
    var int value = 0       # var field = mutable

    def inc():
        value += 1
```

Fields get the same `let`/`mut`/`var`/`var mut` four-way as local bindings
(mut-type-modifier.md §3) — `mutable` (`var`) controls **reassignment**
(`self.field = x`), independently of **content mutation** (`self.field.method()`),
which comes from the field's own type carrying `mut`:

```boring
struct Outer:
    let Point a         # neither reassignable nor content-mutable
    mut Point b         # content-mutable only: o.b.move_to(...) OK, o.b = ... error
    var Point c         # reassignable only: o.c = ... OK, o.c.move_to(...) error
    var mut Point d     # both

mut o = Outer(...)      # reading/writing ANY field also needs `o` itself
                         # declared mut/var mut — a plain `let`/`var o` blocks
                         # every field access above regardless of the field's own keyword
```

## Binding × mutability (scalars)

`mut` on a primitive (`int`, `uint`, `float`, `bool`) is a **checker error** — there are no `def` methods on a scalar for `mut` to unlock, and (per [mut-type-modifier.md](docs/mut-type-modifier.md) §1) `mut` never silently degrades to `var`. Use `var` for a rebindable scalar. (Retired: the historical "`mut` ≡ `var` for scalars" shortcut this section used to document.)

## Ownership qualifiers

| Qualifier | Rust impl | Mutable | Notes |
|---|---|---|---|
| `'shared` | `Rc<T>` / `Arc<T>` | no | immutable shared ref |
| `'actor` | `Rc<RefCell<T>>` / `Arc<Mutex<T>>` | yes | interior mutability |
| `'guard` | `Mutex<T>` / `RwLock<T>` | under lock | |
| `'stack` | `T` | neutral | stack allocation hint |
| `'heap` | `Box<T>` | neutral | heap allocation hint |

```boring
let Counter'actor c = Counter(0)
let int'stack n = 10
```

## Binding × mutability

| Syntax | Rebindable | Content-mutable | Notes |
|---|---|---|---|
| `let` | no | no | |
| `mut` (bare, ≡ `let mut`) | no | yes | error on a scalar — nothing to unlock |
| `var` | yes | no | no longer implies `mut` |
| `var mut` | yes | yes | only form with both |

Permission comes from the **type** (`mut Type`), not the binding keyword alone — see [mut-type-modifier.md](docs/mut-type-modifier.md). This is why `'actor`/`'guard` get no exception below: `var T'actor x` alone no longer suffices for `def` calls, only `var mut T'actor x` does — the lock provides the *mechanism*, Boring's own `mut` bookkeeping still gates it, matching every other type.

Qualifier constraints: `mut 'shared` (and `var mut 'shared`) → compile error in both `boring run` and `boring build` (caught by the semantic checker) — `'shared` has no interior mutability for `mut` to unlock. `var 'guard` compiles cleanly with no warning today.

Parameter passing hierarchy (`var` ≥ `mut` ≥ `let`, caller can pass down, never up) is the intended design but is **not currently enforced** — passing a `let` binding into a `var` parameter compiles and runs without error. (Unlike local bindings above, the parameter model itself is unchanged by mut-type-modifier.md — see that document's "Parameters" section.)

## Enums

```boring
enum Shape:
    Circle(float)
    Rect(float, float)
```

## Pattern matching

```boring
match shape:
    Circle(r): print "circle {r}"
    Rect(w, h): print "rect {w}x{h}"
```

## String interpolation

```boring
let name = "World"
print "Hello, {name}!"
```

- **No `string(x)` conversion function** — to convert a value to string, use interpolation: `let s = "{x}"` or inline `"prefix {x} suffix"`
- `{{` → literal `{` ; `}}` → literal `}`
- `{}` (empty hole) → literal `{}`
- `{expr:fmt}` → formatted interpolation with a **static** format specifier (e.g. `{n:.2f}`, `{n:x}` for hex); the format part is a raw string, not an expression — `{n:{fmt}}` does **not** interpolate `fmt`, it produces the literal `{:{fmt}}`
- `{:x}` (empty expr, non-empty fmt) → literal `{:x}`, nothing is formatted
- A lone `{` inside a string is **always** treated as the start of an interpolation hole — it is **not** a literal brace. `"{"` is a lexer error (unterminated string). Use `"{{"` instead.

## Common types

| Boring | Rust |
|---|---|
| `int` | `isize` |
| `uint` | `usize` |
| `float32` | `f32` |
| `float64` | `f64` |
| `float` | `f64` (pure alias of `float64` — not an independent type like `int`/`uint`) |
| `bool` | `bool` |
| `string` | `Rc<str>` (single-thread) / `Arc<str>` (multi-thread) — or `&'static str` in strict mode (literals only) |
| `[T]` | `Vec<T>` |
| `{K=V}` | `HashMap<K, V>` — **not** `{K: V}`, that's a different (ordered) map used only in kernel/GPU code |
| `{T}` | `HashSet<T>` |

### `string` implementation

- **Default mode**: `Rc<str>` (`--threading single`) or `Arc<str>` (`--threading multi`) — enables sharing and arbitrary value lifetimes.
- **Strict mode**: `&'static str` — restricted to compile-time string literals; forbidden for computed or interpolated values.

The mode is inferred by the transpiler from usage context; there is no explicit qualifier to force it.

## Collections

Dict and set literals/types use `=`, **not** `:` — a common mistake:

```boring
let [int] arr = [1, 2, 3]
let {string=int} scores = {"Alice" = 90, "Bob" = 85}   # NOT {"Alice": 90} — that's not valid syntax
let {int} unique = {1, 2, 3}                            # set — deduplicates
```

Empty literals: `[]` = empty array, `{}` = empty **set**, `{=}` = empty **dict**.

```boring
var [int] arr   = []     # empty array
var {int} s     = {}     # empty set
var {string=int} d = {=} # empty dict  ← NOT {} which would be an empty set
```

Index assignment (`arr[i] = v`, `dict[k] = v`) mutates in place and requires a `var`/`mut` binding — `let` raises `cannot assign to immutable variable`. Dict assignment inserts the key if absent, updates it otherwise. Sets are **not** index-assignable (`s[i] = v` is a compile/runtime error) — use `s.add(v)` / `s.remove(v)`.

`mut` on the **element/value type** (inside the brackets) is a separate axis from `mut` on the collection itself (mut-type-modifier.md §3): `[mut Point] arr` — `arr` itself can't grow/shrink/reassign entries, but every element already in it can have `def` called on it (`arr[0].move_to(...)`); `mut [Point] arr` — the reverse, structural mutation only. `{K = mut V}` is the dict analogue (value position only — keys never accept `mut`, mutating one in place would invalidate the hash table). `{mut T}` (sets) is rejected outright — `HashSet<T>` has no mutable element access in Rust (`iter_mut`/`get_mut` don't exist on it), not a Boring design choice.

```boring
var {string=int} md = {"a" = 1}
md["x"] = 99     # insert
md["a"] = 100    # update
```

## Project structure

```sh
boring run          # interpret
boring build        # emit Cargo project
```

Source: `docs/book.md` for the full language reference.

## Self-hosted interpreter (`boring/interpreter/`)

The interpreter/stdlib written **in Boring itself** lives under `boring/interpreter/` (`.br` files: `lexer.br`, `parser_core.br`, `parser_exprstmt.br`, `ast.br`, `exec.br`, `eval.br`, `methods.br`, `stdlib.br`, `main.br`). It is never run directly with `boring run` for validation — it must always be **transpiled then compiled**:

1. `boring build [--threading multi|single] [--mode strict|managed]` from `boring/interpreter/` → emits a Cargo project (`main_rust`, `main_rust_single`, `main_rust_managed`, `main_rust_managed_single`).
2. `cargo build` on that generated project.

`tests/interpreter_build.rs` does step 1+2 for all four mode/threading combinations; `tests/interpreter_functional.rs` then runs `tests/cases/*.br` against all four compiled binaries and checks output against `.expected` files. That functional suite is what actually validates the self-hosted interpreter — always run both (`cargo test --test interpreter_build` before `cargo test --test interpreter_functional`) after changing anything under `boring/interpreter/`, not just `boring run` on the `.br` sources.
