# The new programming language is Boring

Rust is arguably the best programming language for the future.
It is fast, safe, expressive, and gives you fine-grained control over memory without a garbage collector.
The community is thriving, the ecosystem is mature, and more and more systems — from operating systems to web backends to embedded firmware — are being rewritten in Rust.

There is only one problem: **Rust is not boring enough to write**.

```rust
// A simple function that reads a file and parses its first line as an integer
fn read_first_line(path: &str) -> Result<i64, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let line = content.lines().next().ok_or("empty file")?;
    let n: i64 = line.trim().parse()?;
    Ok(n)
}
```

Ownership annotations, lifetime specifiers, trait bounds, `Arc<Mutex<T>>`, `Box<dyn Error>`, `.unwrap()` everywhere — Rust asks a lot of the programmer before the first line of business logic is written.
That cognitive overhead is the price you pay for the guarantees Rust provides.
It is a fair trade.
But it makes Rust genuinely *hard* to write quickly, especially for newcomers.

**Boring** is an attempt to make that trade cheaper.

---

## What is Boring?

Boring is a language that compiles to Rust.
It borrows its look and feel from **Python** and **Swift** — two languages celebrated for their readability — and maps every concept directly onto idiomatic Rust.
The result is code that is fast to write, easy to read, and that produces real, auditable Rust output.

```boring
string greet(string? name, int visits) throws:
    guard visits > 0 else throw "no visits recorded"
    let who = name else "stranger"
    "Welcome back, {who}! Visit number {visits}."
```

```rust
fn greet(name: Option<Arc<str>>, visits: i64)
    -> Result<Arc<str>, Box<dyn std::error::Error>>
{
    if visits <= 0 {
        return Err(Box::new(BoringError::Str("no visits recorded")));
    }
    let who = name.unwrap_or_else(|| Arc::from("stranger"));
    Ok(Arc::from(format!("Welcome back, {}! Visit number {}.", who, visits).as_str()))
}
```

Same semantics. A fraction of the noise.

Boring is not a toy. It has a full interpreter for rapid prototyping and two transpiler backends that output clean, auditable Rust source you can inspect, modify, and ship.

---

## Targets

Boring compiles the same source to two distinct Rust targets:

| Command | Target | Runtime | Use case |
|---------|--------|---------|----------|
| `boring build` | Rust std + tokio | userspace | servers, CLIs, desktop apps |
| `boring build --mode managed` | Rust std + tokio | userspace | managed memory (`T'` → `Arc<Mutex<T>>`) |
| `boring build --threading single` | Rust std + tokio | userspace | single-thread (`Arc` → `Rc`, `spawn` → `spawn_local`) |
| `boring build --target kernel` | Rust-for-Linux (`no_std`) | Linux kernel | drivers, subsystems, kernel modules |

The kernel backend applies the same language — structs, enums, traits, ownership qualifiers, error handling, async tasks — but maps every construct to its kernel-native equivalent: `Arc<kernel::sync::Mutex<T>>` instead of `tokio::sync::Mutex`, work items on `system_wq` instead of `tokio::spawn`, ring-buffer channels instead of MPSC, and errno-based errors instead of `Box<dyn Error>`.

```boring
# Same source, two targets
task def Page fetch_page(string url) throws:
    # ... fetch logic
```

```sh
boring build main.br                   # → Cargo project with tokio
boring build --target kernel main.br   # → Rust-for-Linux module
```

The kernel backend validates your source before emitting — `float` and `panic` are rejected with explicit error messages; channels and streams without an explicit capacity emit a warning and default to 2.

See [`docs/kernel-transpiler-mapping.md`](docs/kernel-transpiler-mapping.md) for the full mapping reference.

---

## A taste of the simplifications

### Functions and error handling

Rust requires explicit `Result` types, `?` operators, and trait objects for errors.
Boring replaces all of that with a single `throws` keyword — just like Swift.

| | Boring | Rust |
|---|---|---|
| Declaration | `int divide(int a, int b) throws:` | `fn divide(a: i64, b: i64) -> Result<i64, Box<dyn Error>>` |
| Early exit | `guard b != 0 else throw "division by zero"` | `if b == 0 { return Err("division by zero".into()); }` |
| Call + fallback | `let r = try divide(10, 0) else -1` | `let r = divide(10, 0).unwrap_or(-1)` |

### Strings and interpolation

Rust has no built-in string interpolation; every formatted value goes through `format!`.
Boring supports Swift-style `{expression}` inside any string literal.

```boring
let name = "World"
let n = 42
print "Hello, {name}! The answer is {n}."
print "pi ≈ {3.14159:.3}, hex = {255:x}"
```

```rust
// Rust equivalent
println!("Hello, {}! The answer is {}.", name, n);
println!("pi ≈ {:.3}, hex = {:x}", 3.14159_f64, 255_i64);
```

### Types and ownership

Rust's ownership system is powerful but verbose.
Boring provides a concise qualifier syntax inspired by Swift's value/reference types — the right ownership is inferred from context, and common cases have short aliases.

| Boring | Rust | Meaning |
|---|---|---|
| `int` | `i64` | copy integer |
| `float` | `f64` | copy float |
| `string` | `Arc<str>` | thread-safe shared string — the default |
| `T?` | `Option<T>` | optional value |
| `T'` | `Box<T>` | heap-allocated exclusive |
| `T'shared` | `Arc<T>` (multi) / `Rc<T>` (single) | shared reference — threading-aware |

```boring
string greet(string? name):
    guard let n = name else return "Hello, stranger!"
    "Hello, {n}!"
```

```rust
fn greet(name: Option<Arc<str>>) -> Arc<str> {
    let Some(n) = name else {
        return Arc::from("Hello, stranger!");
    };
    Arc::from(format!("Hello, {}!", n).as_str())
}
```

### Structs and methods

Rust separates struct definitions from their `impl` blocks.
Boring puts methods directly inside the struct, like Swift and Python classes.
`req` (read-only, `&self`) and `def` (mutating, `&mut self`) replace the Rust receiver distinction.

```boring
struct Counter:
    var int value = 0

    req int get():
        self.value

    def inc():
        self.value = self.value + 1
```

```rust
struct Counter { value: i64 }

impl Counter {
    fn get(&self) -> i64 { self.value }
    fn inc(&mut self) { self.value += 1; }
}
```

### Pipe operator and data pipelines

Rust has no pipe operator — chaining transformations requires nesting calls or intermediate variables.
Boring's `|>` threads a value left-to-right through any sequence of operations.

```boring
let words = "the quick brown fox jumps over the lazy dog".split(" ")

let result = words
    |> filter(:length > 3)
    |> map(:upper())
    |> sorted()

for w in result:
    print w
```

```rust
let words: Vec<&str> = "the quick brown fox jumps over the lazy dog".split(' ').collect();

let mut result: Vec<Arc<str>> = words.iter()
    .filter(|__x| __x.len() as i64 > 3)
    .map(|__x| Arc::from(__x.to_uppercase().as_str()))
    .collect();
result.sort();

for w in &result { println!("{}", w); }
```

### Async / concurrency

Rust async requires `async fn`, `tokio::spawn`, `.await` at every call site, and explicit `Arc::clone` before moving values across threads.
Boring uses a single `task` keyword; capture analysis is automatic.

```boring
task string transform(string s):
    "done: {s}"

task main():
    let label = "shared"
    let f = task transform(label)   # Arc::clone inserted automatically
    print label                    # still accessible
    print f.value                  # awaits the result
```

```rust
async fn transform(s: Arc<str>) -> Arc<str> {
    Arc::from(format!("done: {}", s).as_str())
}

#[tokio::main]
async fn main() {
    let label: Arc<str> = Arc::from("shared");
    let f = tokio::spawn({
        let label = Arc::clone(&label);
        async move { transform(label).await }
    });
    println!("{}", label);
    println!("{}", f.await.unwrap());
}
```

---

## Getting started

### Prerequisites

- [Rust toolchain](https://rustup.rs/) (edition 2021)
- For async features: `tokio` (added automatically when using `cargo`)

### Build and install

```sh
git clone https://github.com/mlanoe/boring
cd boring
cargo install --path .
```

This compiles the `boring` binary and places it in `~/.cargo/bin`, which is already on your `PATH` if you installed Rust via `rustup`.

### Run a program

```sh
# Interpret directly
boring examples/hello.br

# Transpile to Rust (generates a Cargo project)
boring build examples/hello.br
cd examples/hello_rust && cargo run
```

### The showcase

`examples/hello.br` is a runnable tour of every language feature: bindings, functions, control flow, structs, enums, traits, error handling, closures, generics, modules, async tasks, and more. Run it or read it — it is the fastest way to see what Boring looks like.

---

## Repository layout

```
boring/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── parser/              # Lexer + recursive-descent parser → AST
│   ├── interpreter/         # Tree-walk interpreter (for rapid iteration)
│   ├── validator/           # kernel.rs — pre-emission validation pass
│   └── transpiler/
│       ├── *.rs             # Standard backend → Rust std + tokio
│       └── kernel/          # Kernel backend → Rust-for-Linux (no_std)
├── docs/
│   ├── book.md              # Full language reference
│   └── kernel-transpiler-mapping.md  # Boring → Rust-for-Linux mapping
├── spec/
│   └── grammar.bnf          # Formal BNF grammar
├── examples/
│   ├── hello.br             # Feature showcase (Boring source)
│   └── hello.rs             # Generated Rust (from boring build)
├── LICENSE                  # GNU General Public License v3
├── CLA.md                   # Contributor License Agreement
└── CLA-signatories.md       # List of CLA signatories
```

---

## Language reference

The complete language reference is in [`docs/book.md`](docs/book.md).
It covers every construct with Boring source, Rust equivalent, and explanatory notes.

---

## Contributing

Contributions are welcome — bug reports, feature suggestions, and pull requests alike.

Before your first Pull Request can be merged, you must sign the [Contributor License Agreement](CLA.md).
The process is simple: add your name to [`CLA-signatories.md`](CLA-signatories.md) as part of your PR.
The CLA lets you keep your copyright while giving the project owner the flexibility to relicense the project in the future.

---

## License

Copyright (C) 2026 Mickaël LANOË

This program is free software: you can redistribute it and/or modify it under the terms of the
[GNU General Public License v3.0](LICENSE) as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but **without any warranty**.
See the LICENSE file for the full terms.

---

## Philosophy

The name is intentional.
Boring code is predictable code.
It does what it says, says what it does, and does not surprise you at 2 a.m.
The goal of this language is not to be clever — it is to make writing correct, fast, Rust-backed programs as uneventful as possible.

---

*Mickaël LANOË*
