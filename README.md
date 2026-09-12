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
fn greet(name: Option<Arc<str>>, visits: isize)
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

Boring compiles the same source to several distinct Rust targets — from userspace CLIs down to GPU kernels and Linux kernel modules:

| Command | Target | Use case |
|---------|--------|----------|
| `boring build` | Rust std + tokio | servers, CLIs, desktop apps |
| `boring build --target cuda` | Rust + CUDA C (NVIDIA) | GPU compute |
| `boring build --target rocm` | Rust + HIP C++ (AMD) | GPU compute |
| `boring build --target metal` | Rust + MSL (Apple Silicon/macOS) | GPU compute |
| `boring build --target wgpu` | Rust + WGSL (any DX12/Vulkan/Metal GPU) | GPU compute, cross-platform |
| `boring build --mode managed` | Rust std + tokio | managed memory (unqualified `T` → `Arc<Mutex<T>>`) |
| `boring build --threading single` | Rust std + tokio | single-thread (`Arc` → `Rc`, `spawn` → `spawn_local`) |
| `boring build --target kernel` | Rust-for-Linux (`no_std`) | Linux drivers, subsystems, kernel modules |

The GPU targets are the most fully developed: the same `kernel` struct, ownership qualifiers, and `gpu.*` built-ins transpile unchanged to CUDA C, HIP C++, Metal Shading Language, or WGSL depending on the flag — see [GPU computing](#gpu-computing) below. The kernel target is a smaller, more experimental backend: it maps the same language (structs, enums, traits, ownership qualifiers, `throws`, `task`) onto Rust-for-Linux equivalents (`Arc<kernel::sync::Mutex<T>>`, `system_wq` work items, ring-buffer channels, errno-based errors) and validates kernel-incompatible constructs (`float`, `panic`) at build time — see [`docs/kernel-target.md`](docs/kernel-target.md).

---

## GPU computing

GPU code is ordinary Boring: a `kernel` struct groups device memory fields, an `init` allocator, and an entry point, dispatched from regular host code with `kernel:` — no separate device-language file, no manual memory-transfer boilerplate.

```boring
kernel Scale:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] *= 2.0

mut k = Scale(data)
kernel:
    k(block = 256)

print k.buf[0]
```

The same source targets four backends, chosen purely by build flag:

| Target | OS | GPU |
|---|---|---|
| `--target cuda` | Windows / Linux | NVIDIA only |
| `--target rocm` | Windows / Linux | AMD only |
| `--target metal` | macOS only | Apple / Intel Mac |
| `--target wgpu` | Windows / Linux / macOS | Any DirectX 12, Vulkan, or Metal GPU |

Ownership qualifiers carry the host/device split: `'unified` (zero-copy managed memory), `'global` (device-only), bare `'actor` (shared/threadgroup/workgroup memory), `'const` (read-only constant memory), and `'actor'global`/`'actor'unified` for atomics — each mapped to the right construct per backend (`cudaMallocManaged` + `__shared__` on CUDA, `MTLStorageModeShared` + `threadgroup` on Metal, `storage`/`workgroup` buffers on wgpu, and so on).

```sh
boring build --target metal main.br    # → Rust + MSL project
cd main_rust && cargo run
```

See [`docs/gpu-module.md`](docs/gpu-module.md) for the full language reference (kernel structs, generics, qualifier inference), and [`docs/cuda-module.md`](docs/cuda-module.md), [`docs/rocm-backend.md`](docs/rocm-backend.md), [`docs/metal-backend.md`](docs/metal-backend.md), [`docs/wgpu-backend.md`](docs/wgpu-backend.md) for backend-specific codegen.

---

## A taste of the simplifications

### Functions and error handling

Rust requires explicit `Result` types, `?` operators, and trait objects for errors.
Boring replaces all of that with a single `throws` keyword — just like Swift.

| | Boring | Rust |
|---|---|---|
| Declaration | `int divide(int a, int b) throws:` | `fn divide(a: isize, b: isize) -> Result<isize, Box<dyn Error>>` |
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
println!("pi ≈ {:.3}, hex = {:x}", 3.14159_f64, 255_isize);
```

### Types and ownership

Rust's ownership system is powerful but verbose.
Boring provides a concise qualifier syntax inspired by Swift's value/reference types — the right ownership is inferred from context, and common cases have short aliases.

| Boring | Rust | Meaning |
|---|---|---|
| `int` | `isize` | copy integer, pointer-width |
| `float` | `f64` | copy float |
| `string` | `Rc<str>` (single) / `Arc<str>` (multi) | shared string — threading-aware |
| `T?` | `Option<T>` | optional value |
| `T'owned` | `Box<T>` | heap-allocated exclusive |
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
struct Counter { value: isize }

impl Counter {
    fn get(&self) -> isize { self.value }
    fn inc(&mut self) { self.value += 1; }
}
```

### Closures

Rust closures need `|params|` bars and, in a chain, plenty of `.iter()`/`.collect()` scaffolding around them. Boring closures read like a lambda calculus cheat sheet: a trailing one drops its parens entirely, and a single-parameter one can drop its own parens too.

```boring
let numbers = [1, 2, 3, 4, 5]
let words = ["hello", "world", "boring"]

numbers.map (n): n * 2              # trailing closure, no parens around the call
numbers.filter n: n % 2 == 0        # trailing + no-paren single param
words.map(:upper())                 # shorthand: field/method on the implicit arg
```

```rust
numbers.iter().map(|n| n * 2).collect::<Vec<_>>();
numbers.iter().filter(|n| n % 2 == 0).cloned().collect::<Vec<_>>();
words.iter().map(|w| w.to_uppercase()).collect::<Vec<_>>();
```

A multi-line body is just an indented block, no braces:

```boring
let big = [1, 10, 2, 9, 3].filter (n):
    n > 5
# [10, 9]
```

See [`docs/book.md`](docs/book.md) §14 (Closures and Higher-Order Functions) for the full set of forms, including zero-arg trailing closures and the `do` disambiguation keyword.

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
    .filter(|__x| __x.len() as isize > 3)
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

## Standard library and dependencies

Boring ships a small first-party standard library and lets a project depend on another Boring project's source directly — no registry involved, just `boring.toml`.

```boring
use boring.collections.*             # first-party stdlib (Stack<T>, Queue<T>, ...)
use numlib.big_uint.*                # a named project dependency (see below)
```

```toml
# boring.toml
[deps]
numlib  = { path = "../boring-numlib", version = "^1.2" }  # a sibling project, by path
somelib = { git = "https://github.com/user/somelib" }      # ...or by git (branch/tag/rev)
```

A `git` dependency is cloned into a persistent local cache and pinned in an auto-generated `boring.lock` (commit it, same as `Cargo.lock`) so it never silently drifts between builds — run `boring update [name]` to deliberately move it forward. `boring run --locked`/`--offline` and `boring build --locked`/`--offline` turn "silently resolve/refetch" into a hard error, for CI and reproducible builds. An optional `version` requirement (`^`/`~`/`=`, Cargo semantics) is checked against the dependency's own declared version — a compatibility assertion, not a solver (there's still exactly one target per `[deps]` line either way).

See [`docs/book.md`](docs/book.md) §15 (Modules) for the full syntax and [`docs/cross-project-code-sharing-gap.md`](docs/cross-project-code-sharing-gap.md) for the design rationale and known limitations compared to a full package manager.

---

## Repository layout

```
boring/
├── src/
│   ├── main.rs              # CLI entry point + boring.toml/boring.lock handling
│   ├── git_deps.rs          # Named git [deps] resolution (cache, boring.lock, --locked/--offline)
│   ├── semver.rs            # Hand-rolled version parsing/matching for [deps]'s `version` key
│   ├── stdlib_embed.rs      # Embeds stdlib/*.br into the compiler binary
│   ├── parser/              # Lexer + recursive-descent parser → AST
│   ├── interpreter/         # Tree-walk interpreter (for rapid iteration)
│   ├── validator/           # kernel.rs — pre-emission validation pass
│   └── transpiler/
│       ├── *.rs             # Standard backend → Rust std + tokio
│       ├── cuda/            # GPU backend → CUDA C
│       ├── rocm/            # GPU backend → HIP C++ (AMD)
│       ├── metal/           # GPU backend → Metal Shading Language
│       ├── wgpu/            # GPU backend → WGSL (cross-platform)
│       └── kernel/          # Kernel backend → Rust-for-Linux (no_std)
├── stdlib/                  # First-party `use boring.<module>` standard library
├── docs/
│   ├── book.md              # Full language reference
│   ├── gpu-module.md        # GPU computing language reference
│   ├── cuda-module.md / rocm-backend.md / metal-backend.md / wgpu-backend.md  # Per-backend codegen
│   ├── cross-project-code-sharing-gap.md  # Dependency system design + known limitations
│   └── kernel-target.md     # Boring → Rust-for-Linux mapping
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
