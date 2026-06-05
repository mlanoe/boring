# Changelog

All notable changes to Boring are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.2.0] — 2026-06-05

### Added

- **`Future<T>`** — stdlib type with `.value()` (blocking) and `.wait()` (async) method syntax
- **`task(duration): body`** — built-in timeout syntax for async tasks
- **tmLanguage grammar** — syntax highlighting for VS Code and GitHub Linguist submission

### Fixed

- `task_context` restoration after task completion
- Mixed `Int`/`Uint` arithmetic operations
- Stack overflow on Windows: main thread now spawned with an 8 MB stack

### Tests

- 4 new integration tests covering audit-identified edge cases (39 total)

### Docs

- `sleep` → `wait` in all async examples; `timeout(dur, fut)` form demoted
- Book: corrected trailing closure ambiguity description
- README: Rust transpiler examples updated from `Arc<String>` to `Arc<str>`

---

## [0.1.0] — 2026-05-22

Initial public release.

### Language features

- **Types** — `int`, `float`, `bool`, `string`, `str`, optionals (`T?`), lists, dicts, tuples, ranges
- **Functions** — named parameters, default values, variadic args, closures, pipes (`|>`)
- **Structs & enums** — constructors, methods, setters, conversions, generics
- **Traits / protocols** — `trait`, `impl`, default methods, protocol conformance checks
- **Error handling** — `throws`, `throw`, `try/catch`, `try expr else default`, typed catch (`catch MyError:`), `guard … else throw`
- **Async** — `task` functions, `stream` functions, channels (`chan`, `tx.send`, `rx.receive`)
- **Pattern matching** — `match` with guards, destructuring, `if let`, `while let`
- **Control flow** — `for`, `while`, `loop`, `break`/`continue`, `defer`, `do` blocks
- **Macros** — `assert_eq`, `assert_neq`, `print`, string interpolation `"{expr}"`
- **Modules** — `mod`, `use`, `pub`, separate file compilation
- **Ownership helpers** — `move`, immutable-by-default parameters, `#[must_use]`
- **Newtypes** — single-field wrapper types with automatic coercion

### Compiler / toolchain

- **Interpreter** — direct execution of `.br` files (`boring run file.br`)
- **Rust transpiler** — `--emit-rust` generates a ready-to-compile Cargo project
- **`BoringVal` typed exceptions** — prim + named error types dispatched via `std::any::TypeId` (collision-free across modules)
- **35 integration tests** covering all language constructs

### Documentation

- Full language reference: `docs/book.md`

---

[0.2.0]: https://github.com/mlanoe/boring/releases/tag/v0.2.0
[0.1.0]: https://github.com/mlanoe/boring/releases/tag/v0.1.0
