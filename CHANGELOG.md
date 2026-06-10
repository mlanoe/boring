# Changelog

All notable changes to Boring are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.4.0] — 2026-06-10

### Added

- **`--threading single` mode** — single-thread async backend: `task` spawns via `tokio::task::spawn_local` instead of `tokio::spawn`; channels use `local_channel::mpsc` instead of `tokio::sync::mpsc`; `T'shared` resolves to `Rc<T>` instead of `Arc<T>`; `T'actor` resolves to `RefCell<T>`; the `local-channel = "0.1"` dependency is injected automatically into the generated `Cargo.toml`.
- **`--mode managed` mode** — managed ownership: all user-defined struct and enum types (except unit enums, which are `Copy`) are automatically wrapped in `Arc<Mutex<T>>` (multi-thread) or `RefCell<T>` (single-thread), eliminating explicit ownership qualifiers for the common shared-mutable pattern.
- **`T'shared` ownership qualifier** — threading-aware ref-counted pointer: `Arc<T>` in multi-thread mode, `Rc<T>` in single-thread mode. Replaces the deprecated `T'auto` (always `Rc`) and `T'task` (always `Arc`), which now produce hard errors.
- **`T'wshared` and `T'wactor` ownership qualifiers** — weak-pointer shorthands: `T'wshared` → `Weak<T>` (threading-aware); `T'wactor` → `Weak<Mutex<T>>` (multi) / `Weak<RefCell<T>>` (single). Complement the existing `T'wguard`.
- **`--output-dir` CLI flag** — specifies the destination directory for the generated Cargo project, allowing multiple configurations to coexist side-by-side.
- **`--stack-auto-bytes` / `--stack-warn-bytes` CLI flags** — configure the size thresholds used by the inference pass to decide between stack and heap allocation, and to emit warnings for oversized stack values.
- **`dyn Trait` auto-boxing** — bare trait types used in value positions are automatically wrapped in `Box<dyn Trait>` by the transpiler; no explicit annotation needed at call sites.
- **Size-based auto-boxing in strict mode** — struct fields whose estimated stack size exceeds `--stack-auto-bytes` (default 1 024 B) are automatically promoted to `Box<T>` at emission time. Primitive type names in `Named` form (`"int"`, `"float"`, …) are now correctly mapped to their known sizes by the inference pass. `T'stack` bypasses auto-boxing explicitly when stack placement is intentional.
- **Enum warning level 2** — the inference pass now detects disproportionate variant sizes (one variant significantly larger than the median) and suggests boxing the outlier field.
- **`!Send` warnings in single-thread mode** — the transpiler warns when a type or value that is not `Send` is used in a context that would require it (e.g. captured into a `tokio::spawn` task), pointing to `--threading single` as the fix.
- **`LocalSet` support** — single-thread async entry point uses `tokio::task::LocalSet` and `local_set.run_until(main())` so that `spawn_local` futures are driven on the same thread.
- **Broadcast channels in single-thread mode** — `broadcast<T, N>` now works in `--threading single` via a prelude that re-exports a `!Send`-compatible local broadcast implementation.
- **Kernel transpiler: `oneshot`, `watch`, and `broadcast`** — the `--target kernel` backend now maps `oneshot<T>`, `watch<T>`, and `broadcast<T, N>` to their Linux-kernel equivalents (completion + `Mutex`-guarded state, ring buffer).

### Removed

- **`--emit-rust` CLI flag** — removed; the `--output-dir` flag covers all use cases more cleanly.
- **`T'auto` and `T'task`** — completely removed from the parser. These qualifiers are no longer recognized; use `T'shared` instead.

### Fixed

- `T'actor` in multi-thread mode now consistently emits `tokio::sync::Mutex` (was incorrectly emitting `std::sync::Mutex` in some code paths, causing async-context deadlocks).
- `T'weak` in single-thread mode now correctly emits `Rc::downgrade` (was using `Arc::downgrade`, causing a type mismatch when `T'shared` resolves to `Rc<T>`).
- Managed-mode mutex parameters no longer cause a deadlock when their fields are accessed multiple times in a single expression: a `let mut __param_mg = param.lock().unwrap()` guard binding is now emitted at function entry, and all field reads go through the guard (std::sync::Mutex is not reentrant).

### Tests

- 4-combination transpile suite (strict/managed × multi/single) covering all language constructs.
- `optionals`, `operators`, `modules`, and `ownership` test cases promoted from `ignore_managed` / `ignore_single_managed` to fully green across all four configurations.

---

## [0.3.0] — 2026-06-09

### Added

- **`--target kernel` — Rust-for-Linux transpiler backend** — a second emission backend that targets the Linux kernel (`no_std` + kernel crates). Parser, AST, and typing passes are shared; only the emission layer changes. Activation: `boring build --target kernel file.br` (single file) or `boring build --target kernel` (project from `boring.toml`).

  Key mappings:
  - `string` → `kernel::str::CStr` / `CString`; string literals → `c_str!("…")`
  - `{K: V}` / `{T}` → `kernel::rbtree::RBTree<K,V>` / `RBTree<T,()>` (O(log n), keys must implement `Ord`)
  - `throws MyError` → `Result<T, kernel::error::Error>` with `type MyError as kernel.error.Error(ERRNO)` binding
  - `task def` → `struct XxxWork: Work` dispatched on `system_wq`; `task expr` → `system_wq.enqueue(work)` returning `KernelFuture<T>`
  - `channel<T, N>` → ring buffer + `Mutex` + `CondVar`; `stream<N> def` → channel + work item
  - `Future<T>` → `KernelFuture<T>` with `.done()` (non-blocking poll via `try_lock`) and `.wait()` (blocking, process context only)
  - `print!` / assertions → `pr_info!` / `WARN_ON`; `panic` and `float` forbidden at validation time
  - `T'task`, `T'actor`, `T'guard` → `kernel::sync::Arc`, `kernel::sync::Mutex`, `kernel::sync::RwLock`

  A validation pass runs before emission and rejects: `float`, `panic`, `T&`/`T&mut` receivers on `task def`, and warns on implicit channel capacity.

  Architecture: `src/transpiler/kernel/` — `mod.rs`, `emit_top.rs`, `emit_stmt.rs`, `emit_expr.rs`, `helpers.rs` (KernelFuture/KernelChan runtime types). See `docs/kernel-transpiler-mapping.md` for the full mapping table.

- **`Future.done`** — non-blocking poll: `req bool done()` returns `true` if the result is already available, without blocking and without throwing. Both property (`f.done`) and call (`f.done()`) syntax are valid. Transpiles to `handle.is_finished()`.
- **`Future.cancel()`** — signal cancellation: the running task receives `Error.Cancelled` on its next await; any subsequent `f.value` also throws `Error.Cancelled`. Transpiles to `handle.abort()`. In the interpreter, this is a no-op (no cancellation tokens available).
- **`Task.cancelled()`** — check whether the current task has been cancelled. Returns `false` in interpreted mode (no cancellation token). Allows graceful cancellation loops: `while not Task.cancelled(): …`
- **`args()`** builtin — returns `[string]`, the CLI arguments passed to the program (argv[0] excluded). Transpiles to `std::env::args().skip(1).collect()`.
- **`ord(string)`** builtin — returns the Unicode codepoint (`int`) of the first character of the string.
- **`chr(int)`** builtin — returns a single-character string for a Unicode codepoint.
- **`{}` as empty Set** — an empty brace literal `{}` now parses as an empty `HashSet` (`HashSet::new()`). The empty dict literal is `{=}` (unchanged).

### Removed

- **`select:`** — fully removed from the language and AST. The keyword now produces a clear compile-time error pointing to `Future.done()` polling as the replacement. `Stmt::Select`, `SelectStmt`, and `SelectArm` removed from the AST; all dead code in lexer, parser, interpreter, and transpiler cleaned up.

### Fixed

- `f.wait` in a `throws` context now propagates `JoinError` as `BoringError` instead of silently discarding it
- `Future.cancel()` no longer crashes in the interpreter (returns `Nil`)
- `Task.cancelled()` no longer crashes in the interpreter (returns `false`)
- Nested struct declarations inside function bodies are no longer leaked into global scope
- Bare non-void call results now produce a compile-time `must-use` error ("return value discarded")
- `tokio-util` dependency removed from generated Cargo.toml (the `sync` feature does not exist)

### Spec

- `spec/grammar.bnf`: `Future<T> methods` section added documenting `done`, `cancel()`, `value`/`wait` overloads, `Task.cancelled()`, and transpilation targets; `select` removed from reserved keywords list with a migration note

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
- **Rust transpiler** — `boring build` generates a ready-to-compile Cargo project
- **`BoringVal` typed exceptions** — prim + named error types dispatched via `std::any::TypeId` (collision-free across modules)
- **35 integration tests** covering all language constructs

### Documentation

- Full language reference: `docs/book.md`

---

[0.4.0]: https://github.com/mlanoe/boring/releases/tag/v0.4.0
[0.3.0]: https://github.com/mlanoe/boring/releases/tag/v0.3.0
[0.2.1]: https://github.com/mlanoe/boring/releases/tag/v0.2.1
[0.2.0]: https://github.com/mlanoe/boring/releases/tag/v0.2.0
[0.1.0]: https://github.com/mlanoe/boring/releases/tag/v0.1.0
