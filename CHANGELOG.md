# Changelog

All notable changes to Boring are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.2.3] — 2026-06-08

### Added

- **Inline (monoline) loop forms** — `while`, `for`, `loop`, `do…while` now accept a single statement on the same line as the colon: `while i < 3: i = i + 1`, `for x in list: print x`, `loop: x = x + 1`, `do: x = x + 1 while x < 10`
- **`;` statement separator** — semicolons are treated as newlines by the lexer, allowing multiple statements on one line: `let a = 1; let b = 2; print a + b`
- **Tuple methods** — `length()`, `isEmpty()`, `first()`, `last()`, `map(closure)`, `all(pred)`, `any(pred)` on tuple values; `map` preserves per-slot type inference; `all`/`any` short-circuit across slots; field shorthand works: `boxes.map(:value)`
- **Arc-qualified receiver validation** — methods declared `task` on a struct must use an Arc-qualified receiver (`T'task`, `T'actor`, `T'guard`); using a plain receiver is now a compile-time error

### Spec

- `grammar.bnf` updated: `block` rule now documents the inline (monoline) form; `while_stmt`, `for_stmt`, `loop_stmt`, `do_while_stmt` annotated; `;` documented as `SEMICOLON`; tuple methods section added; Arc-qualified receiver constraint documented in task/concurrency semantics

---

## [0.2.2] — 2026-06-07

### Added

- **GitHub Pages** — language book published at `https://mlanoe.github.io/boring/` via GitHub Actions
- **Landing page** — `index.html` with tagline, code snippet and links to the book and repository
- Multiline syntax for arrays, sets, dicts, tuples, type parameters, trait lists, macro args, destructuring and match patterns

### Fixed

- `try?` section in the book now shows idiomatic `throws` syntax as primary example (`Result<T,E>` moved to interop note)
- Removed `T'shared` qualifier from kernel mapping draft (superseded by `T'task`)
- `T'auto` mapped to `kernel::sync::Arc` in kernel transpiler draft (Rc unavailable in kernel context)

### Removed

- Beta warning banner removed from the language book
- `trait B: A` supertrait form removed (only `trait B as A:` is accepted)
- `let [a, b] = join [...]` array destructure removed — use `let (a, b) = join(...)` tuple form

### Docs

- `try?` example uses `int f() throws:` syntax (was incorrectly `throws int f():`)
- Kernel mapping draft updated: `T'shared` removed, `T'auto` remapped

---

## [0.2.1] — 2026-06-06

### Fixed

- Enum field accessors now return `Option<T>` instead of panicking when the field is absent from the current variant
- Unhandled `catch` variants and unmatched errors print to stderr before panicking instead of crashing silently
- Replace bare `unwrap()` in transpiler internals with `expect()` and invariant messages
- Replace bare `unwrap()` in generated code: mutex locks recover from poisoning, channel send/recv propagate errors in `throws` context, JoinHandle await uses descriptive `expect()`
- CI: use `macos-13` runner for `x86_64-apple-darwin` build (fixes `E0463` on arm64 `macos-latest`)

### Removed

- Deprecated `every dur: body` syntax removed from documentation (was never implemented)
- 62 compiler warnings eliminated (unused imports, dead code, unused fields)

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

[0.2.1]: https://github.com/mlanoe/boring/releases/tag/v0.2.1
[0.2.0]: https://github.com/mlanoe/boring/releases/tag/v0.2.0
[0.1.0]: https://github.com/mlanoe/boring/releases/tag/v0.1.0
