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
var x = 42              # rebindable, mutable (qualifier permitting)

struct Counter:
    var int value = 0
mut c = Counter(0)      # fixed binding, mutable instance
mut x = 42              # scalar: rebindable (equivalent to var)
```

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

## Binding × mutability (scalars)

For primitives (`int`, `uint`, `float`, `bool`), `mut` is accepted and means **rebindable** — equivalent to `var`. There is no "mutable instance" concept for value types, so `mut x = 42` and `var x = 42` are identical in behavior.

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

| Syntax | Rebindable | Mutable | Notes |
|---|---|---|---|
| `let` | no | no | |
| `mut` | no (structs) / yes (scalars) | yes | for primitives, equivalent to `var` |
| `var` | yes | depends on qualifier | |

Qualifier constraints: `mut 'shared` → compile error in both `boring run` and `boring build` (caught by the semantic checker). `var 'guard` compiles cleanly with no warning today.

Parameter passing hierarchy (`var` ≥ `mut` ≥ `let`, caller can pass down, never up) is the intended design but is **not currently enforced** — passing a `let` binding into a `var` parameter compiles and runs without error.

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
| `float` | `f64` |
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
