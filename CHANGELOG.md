# Changelog

All notable changes to Boring are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.9.0] — 2026-07-07 *(interpreter: 78/78 · transpiler: 216/216 tests passing)*

### Fixed

- **Windows CRLF in triple-string preprocessor** — source files with `\r\n` line endings caused `strip_prefix('\n')` / `strip_suffix('\n')` to fail inside `preprocess_triple_strings`, leaving a stray `\r` in the dedented string content. CRLF is now normalised to LF before preprocessing; zero overhead on macOS/Linux.

### Improved — Diagnostics

- **Multi-character caret spans** — runtime errors, warnings, and lexer diagnostics now emit a `^^^` caret that spans the full token width (`len` field on `Expr` / `RuntimeError`). Previously all carets were a single `^`.
- **Multiple lexer errors** — the lexer now accumulates all per-line errors (unexpected character, unterminated string, integer overflow) before returning, instead of stopping at the first one. Structural errors (mixed indentation, invalid dedent) still abort immediately.
- **Precise column on runtime errors** — undefined-variable, type-mismatch, division-by-zero, underflow, and index-out-of-bounds errors now report the exact column and token length of the offending operand.
- **Transpiler column in parameter errors** — `cannot assign to field` and `cannot call def method` errors now point to the parameter's source column instead of column 0.
- **Warning span** — `report_warning` accepts a `len` argument; multi-character tokens in warnings are now underlined with the correct number of carets.

### Changed

- **`spec/grammar.bnf`** — added missing reserved keywords: `lazy`, `new`, `with`, `sync`.
- **`linguist/Boring.tmLanguage.json`** — added `lazy`, `new`, `with`, `sync` to the `declaration-keywords` pattern.
- **`docs/book.md` §28 Diagnostics** — fully rewritten to document the new caret-span format, multi-error output, and warning layout.

---

## [0.8.0] — 2026-07-04 *(interpreter: 65/65 tests passing)*

### Added (post-release)

- **Self-hosted interpreter — streams, channels, tasks, generics complete** — the interpreter now passes all 65 test cases:
  - **`stream` functions** — `exec_stream_fn` collects all `yield` values into an array; `Yield` statements inside a stream body append to `interp.stream_yields` instead of returning a `YieldSignal`; `for` loops over stream results work transparently.
  - **`channel<T>(N)`** — `channel` expressions create a sender/receiver pair backed by `interp.channel_queues` (a `{string=[Value]}` map keyed by a unique channel ID). `tx.send(v)` appends to the queue; `for n in rx:` drains it via `eval_channel_rx_for` / `collect_iterable_with_interp`.
  - **`task` expressions** — evaluated synchronously in the interpreter; the result is returned immediately as a plain value (no actual concurrency).
  - **Generic calls `f<T>(args)`** — `ExprKind.GenericCall` is handled: type arguments are ignored and the call is evaluated as a regular function call.
  - **`parser_peek_is_generic_call`** — detects `Name<Type>(` at the current position (checks offsets 1–4 for `Lt`, a type-like token, and `Gt`/`LParen`).

- **Transpiler fix — dict field index-assignment** — `self.field[k] = v` where `field` is a `{K=V}` dict was falling through to the array-index path, emitting `self.field[(k) as usize]` instead of `self.field.insert(k, v)`. The transpiler now matches the same codegen as local dict variables.

### Added

- **GPU kernel structs — CUDA and Metal backends** — `kernel` structs declare device-resident data (fields with GPU memory qualifiers), a host-side `init` allocator, optional device-side helpers, and an anonymous entry-point `def ()` executed once per thread. The same source compiles unchanged to both backends:
  - `boring build --target cuda` — emits a Rust + cudarc project with a PTX kernel compiled via `nvcc`.
  - `boring build --target metal` — emits a Rust + Metal project with MSL compiled at runtime via `newLibraryWithSource` (no toolchain beyond macOS required).
  - Launch syntax: `kernel(block = 256) k` returns a `KernelHandle<T>`; `|> .wait` synchronises and returns the updated struct.
  - GPU memory qualifiers (`'unified`, `'global`, `'shared`, `'local`, `'const`) replace standard ownership qualifiers inside `kernel` struct fields. Scalar `let`/`mut`/`var` fields infer their qualifier automatically.
  - GPU built-ins available device-side without `use`: `gpu.thread.x/y/z`, `gpu.block.x/y/z`, `gpu.block_dim.x/y/z`, `gpu.grid_dim.x/y/z`, `sync`.
  - `gpu-profiles/` directory — pre-tuned block/grid defaults for common GPUs (A100, H100, RTX 3090/4090, V100).

- **Qualifier inference for `kernel` struct fields** — the transpiler infers `'const` for scalar/fixed-array `let` fields and `'local` for `mut`/`var` fields; explicit qualifiers remain valid and always take precedence. Dynamic `[T]` fields still require an explicit qualifier.

- **Self-hosted interpreter — major expansion** — the Boring-in-Boring interpreter (`boring/interpreter/`) received a large batch of new capabilities:
  - **Macro call evaluation** — `vec!`, `format!`, `println!`, `print!`, `concat!`, `assert!`, `assert_eq!` are fully evaluated by the interpreter.
  - **Trailing closure detection** — `parser_peek_is_trailing_closure` and `parser_peek_is_trailing_closure_no_paren` are now implemented; the parser correctly distinguishes trailing `(params): body` from regular argument lists.
  - **Typed closure detection** — `parser_peek_is_typed_closure` delegates to `parser_is_type_start_before_ident`; borrow-annotated types (`T&`) are handled in the type-start lookahead.
  - **Pipe operator `|>`** — `ExprKind.Pipe` is evaluated: tries a free function first, falls back to a method call on the left-hand value.
  - **`task` and `join` expressions** — `ExprKind.Task`, `ExprKind.TaskWithTimeout`, `ExprKind.JoinAll` are handled (interpreter runs them synchronously).
  - **`as`-cast conversions** — `try_call_conversion_method` looks for a `__as__<typename>` method; struct-to-float and struct-to-string extension methods are resolved.
  - **`clone()` method** — returns the receiver value unchanged (interpreter has no move semantics).
  - **`upgrade()` on weak refs** — returns `self` (all interpreter refs are strong).
  - **Math functions in stdlib** — `sin`, `cos`, `tan`, `round`, `floor`, `ceil`, `pow`, `log`, `ln`, `log2`, `log10` registered as native functions.
  - **`Ok`/`Err`/`Some`/`None` constructors** — registered as enum-variant values in the global environment.
  - **Trait default methods** — when a struct declares conformance to a trait, default implementations from the trait declaration are merged into the struct's method table (own methods take priority).
  - **Lazy binding** — `stmt.is_lazy` is checked; lazy variables are registered with `define_lazy` instead of a concrete initial value.
  - **`ExprKind.Void`** — evaluates to `Value.Nil`.
  - **Parser: macro-call detection** — `parser_peek_is_macro_call` now correctly detects `name!` by checking that the next token is `Bang`.
  - **Parser: `parser_skip_to_offset` fix** — reads `p.pos` into a local before adding the offset (avoids a double-read of the actor-guarded field).
  - **Parser: keyword identifiers expanded** — `Wait`, `Task`, and `Use` are now accepted as valid identifiers where an identifier-or-keyword is expected.

### Changed

- **`spec/grammar.bnf`** — comprehensive update:
  - Ownership qualifier table: replaced deprecated `'auto` and `'task` with `'shared`; `T'weak.upgrade()` return type corrected to `T'shared?`; builtin alias `string` updated from `String'task` to `String'shared`.
  - Borrow qualifier table: removed `T&auto` and `T&task`; added `T&shared`.
  - Native type comment: `string → Arc<String>` corrected to `Arc<str>`.
  - New `kernel_decl` / `kernel_member` / `kernel_field_decl` rules added; `kernel_decl` added to `item`.
  - New GPU kernel struct section: full documentation of GPU memory qualifiers, qualifier inference, launch syntax, and GPU built-ins.
  - Emission targets: `--target cuda` and `--target metal` documented.
  - `kernel` added to the reserved keywords list.
- **`linguist/Boring.tmLanguage.json`** — `kernel` added to `declaration-keywords` pattern.
- **`linguist/samples/gpu.br`** — new sample file demonstrating SAXPY, shared-memory tile reduction, and host-side qualifier usage.
- **`tests/cases/collections.br`** — struct copy test updated to use explicit `.clone()` (was relying on implicit copy semantics, which now requires a `mut` binding).
- **`tests/cases/triple_string.expected`** — leading blank lines removed; triple-quoted strings no longer emit extra newlines before the content.

### Fixed

- **Metal codegen** — GPU qualifier inference now correctly handles `kernel` struct fields in `ext` blocks; `'actor` and `'guard` fields are wrapped at both declaration and construction sites.
- **Qualifier inference** — `'actor'task` and `'guard'task` are disambiguated from plain `'actor`/`'guard` via a task-method-call signal; prevents spurious `Arc<Mutex<Arc<Mutex<T>>>>` double-wrapping.

---

## [0.7.0] — 2026-06-18

### Added

- **Boring interpreter written in Boring** — `boring/interpreter/main.br` is a working self-hosted interpreter skeleton that compiles via `boring build`. It supports function declaration lookup (`fn_decls` map, `set_fn_decl` / `get_fn_decl` / `lookup_fn_decl`) and executes `Item.Fn` nodes.
- **Non-async multi-thread mode for `'actor` / `'guard` types** — programs that use `'actor` or `'guard` qualifiers but contain no `task` / `stream` functions now emit `std::sync::Mutex` / `std::sync::RwLock` (blocking) instead of `tokio::sync::Mutex` / `tokio::sync::RwLock` (async). This allows the boring interpreter and other CPU-bound programs to build in `--threading multi` mode without depending on the async runtime for locking.

  Technical details:
  - New `use_async_actors()` predicate: returns `true` only when the program contains at least one `task` or `stream` function.
  - All actor/guard construction sites (`emit_actor_new`, `emit_guard_new`) and access sites branch on this predicate.
  - `std::sync::{Mutex, RwLock}` are injected into the generated `use` block only when needed.
  - Structs with `std::sync::Mutex` fields skip `#[derive(PartialEq)]` (`Mutex` does not implement `PartialEq`).
  - Local actor `let` bindings no longer generate a spurious `let mut __x_mg = x.lock().unwrap()` shadow guard (only function parameters need one).

### Changed

- **`spec/grammar.bnf`** — comprehensive update to bring the grammar in sync with the parser:
  - `owner_qual`: removed deprecated `"auto"` and `"task"`; added `"shared"`; added qualifier union syntax `T'stack|heap` (resolved at inference time).
  - `borrow_qual`: aligned with `owner_qual` (`"shared"` replaces `"auto"` / `"task"`).
  - `primitive_type`: corrected to lowercase (`int`, `uint`, `float`, `bool`, `string`, `nil`, `never`).
  - `use_decl` selective import: corrected to parenthesised form `use a.b(X, Y)` (was incorrectly shown as `use a.b.X, Y`).
  - Added `join_expr` to `primary_expr`: `join [f1, f2, f3]` — await all tasks concurrently.
  - Added `alias_decl` to top-level `item` rule.
  - Added variadic parameter form: `type "..." IDENT`.
  - Added `assoc_type_decl` and `type_method_sig` to `struct_member` and `ext_member`.
  - Added `break_stmt`, `continue_stmt`, `yield_stmt` to `stmt`.
  - Made `let_stmt` initializer optional (`("=" expr)?`).
  - Fixed `catch_type_list` to support dotted catch variants (`catch Mod.Error:`).
  - Added spread arg `".." expr` to `arg`.
  - Added generic call postfix form `expr<Type, …>(args)`.
- **`linguist/Boring.tmLanguage.json`** — `ownership-qualifier` pattern updated to match the current qualifier vocabulary: removed `'auto` and `'task`; retained `heap`, `stack`, `shared`, `actor`, `guard`, `weak`, `copy`.

### Fixed

- `Arc::clone` emitted correctly in both `emit_expr` and `emit_expr_owned` for actor struct fields in multi-thread mode (was missing from the owned path, causing a move-out-of-`MutexGuard` error).
- `child.clone()` as `ExprKind::MethodCall` is now recognized by `is_existing_arc` in `emit_let_value`, preventing a double `Arc::new(Arc::new(...))` wrap.

---

## [0.6.0] — 2026-06-14

### Added

- **`dbg(expr)`** — new builtin that maps to Rust's `dbg!()`: prints `[file:line] expr = value` to stderr and returns the value unchanged.  Usable inline inside any expression.
- **`todo()` / `todo(msg)`** — panic placeholder for unfinished code paths; maps to `todo!()`.
- **`unreachable()` / `unreachable(msg)`** — assertion that a code path is never reached; maps to `unreachable!()`.
- **`--mode managed` debug enhancements** — building with `--mode managed` now also:
  - Writes `.cargo/config.toml` with `RUST_BACKTRACE = "1"` so panics always print a full stack trace without any manual environment variable.
  - Adds `#[track_caller]` to every emitted function and method so panic messages report the call site rather than the panic site deep in the standard library.
- **`--sanitize address|thread|memory`** — new build flag.  Writes `.cargo/config.toml` with `-Zsanitizer=<san>` and the host target triple (detected via `rustc --version --verbose`).  Combinable with all other flags.  Requires a nightly toolchain (`cargo +nightly run`).
- **`--instrument`** — new build flag.  Prepends an inline `__boring_instrument` module (no external dependency) that wraps every function body with a RAII `Span` guard tracking call counts and wall-clock durations.  On program exit (including unwind panics via a `DumpGuard` in `main`) two files are written:
  - `boring_coverage.json` — per-function aggregated stats (`calls`, `total_us`, `avg_us`), sorted alphabetically.
  - `boring_trace.json` — all calls in Chrome Trace Format, directly openable in Perfetto (`ui.perfetto.dev`) and Speedscope (`speedscope.app`) without conversion.
  - Methods are labelled `Type::method` in both outputs.

### Changed

- **`grammar.bnf`** — emission targets section expanded with documentation of `--instrument`, `--sanitize`, and `--mode managed` debug enhancements; builtin debugging functions table added.
- **`docs/book.md`** — chapters 32–33 merged into a single **Chapter 32 — Debugging & Profiling** with five numbered subsections (32.1 builtins · 32.2 managed mode · 32.3 sanitizers · 32.4 instrumentation · 32.5 combining all tools).
- **`Cargo.toml`** — version bumped from 0.4.0 to 0.6.0 (0.5.0 was released without bumping the crate manifest).

---

## [0.5.0] — 2026-06-14

### Added

- **Qualifier inference — constraint elimination** — unqualified variables start with the full candidate set `{Stack, Owned, Shared, Actor, Guard, Const}`. Each usage signal eliminates incompatible qualifiers (`retain`). When exactly one remains it is chosen automatically; when none remain a compile error is reported; when several remain a size-based fallback resolves the tie (≤ 256 B → `'stack`, > 256 B → `'heap`). The zero-annotation goal: qualifier-free Boring code emits the same Rust as hand-annotated code.
- **Signal table** — the signals that constrain the candidate set: explicit call-site qualifier demand, `def` method call (eliminates `'shared`/`'const`), `mut` binding (eliminates `'shared`/`'const`), task capture as method receiver (`{Actor, Guard}`), task capture read-only (`{Shared, Actor, Guard}`).
- **`mut` keyword** — new binding form `mut x = expr`: fixed binding, mutable instance. Adds a mutation constraint to the inference candidate set (eliminates `'shared` and `'const`). Recognised in the grammar, AST, transpiler, and syntax-highlighting files.
- **`T'` inference** — tick variables (`T'`) now participate in constraint elimination with a restricted initial candidate set `{Owned, Shared, Actor, Guard}` (Stack and Const excluded). Inference can promote a tick variable to `'shared`, `'actor`, or `'guard` based on usage signals; fallback when unresolved is `'heap` (`Box<T>`). The suppression of size-based auto-boxing for non-rebindable bare `T` struct fields does not apply to `T'` fields — their fallback is always `Box<T>`.
- **Parameter auto-apply** — inferred qualifiers are applied to function parameters at emission time; a pre-inference pass runs before `emit_param` so that the emitted Rust signature already carries the correct type wrapper. Applies to both `T` and `T'` parameters.
- **Cross-function propagation** — after a function body is emitted, inferred parameter qualifiers are written back into `fn_sigs`; callers defined later in the file benefit without re-analysis.
- **Struct field inference** — `infer_struct_field_qualifiers` scans all method and setter bodies for `self.field` access patterns and resolves each unqualified field to `'actor` or `'guard` using the same signal table. Results are written into the existing `struct_mutex_fields` / `struct_rwlock_fields` registries; no change to the emission layer. All fields are resolved from internal usage only, consistent with module-boundary constraints.
- **`var` reassignment as mutation signal** — assigning to a `var` variable (`x = …`, `x.field = …`, `x.a.b.c = …`, `x[i] = …`) now constrains its qualifier set to `{Stack, Owned, Actor, Guard}`. The assignment target is walked recursively to find the root variable, so deeply nested field and index assignments are covered. Previously only `def` method calls triggered this constraint.
- **`set` setter as mutation signal (struct fields)** — setter bodies (`set prop(T v):`) are now walked by `infer_struct_field_qualifiers` in addition to `def` method bodies, so field mutation performed through a setter is correctly accounted for in field qualifier inference.
- **Closure capture signals** — closures are now treated like `task` bodies for qualifier inference: a variable captured as a method receiver constrains to `{Actor, Guard}`; a variable captured read-only constrains to `{Shared, Actor, Guard}`. Previously only explicit `task` blocks triggered capture-based constraints.
- **`T?` / `T'?` optional inference** — optional variables participate in constraint elimination; the inferred qualifier is applied to the inner type of the `Option` (`Option<Arc<Mutex<T>>>`, not `Arc<Mutex<Option<T>>>`).
- **Qualifier unions / groups** — parameter forms `T'one`, `T'many`, `T'mut`, `T'req` seed the inference with the corresponding member set as the initial candidates. Useful for expressing "any mutable qualifier" without writing an explicit one; the body signals then narrow to a single candidate.
- **Parameter seeding** — parameters were not previously seeded into the inference system; only local `let`/`var` bindings were tracked. All `T`, `T'`, and `T'<group>` parameters now participate in constraint elimination from the start of `infer_qualifiers`.

### Changed

- **`--stack-auto-bytes` default lowered from 1 024 to 256 bytes** — aligned with Clippy's `large_types_passed_by_value` lint; more conservative default that avoids silently placing large structs on the stack.
- **`--stack-warn-bytes` removed** — the intermediate warning zone ("suggest `'heap`") conflicted with the zero-annotation goal by nudging developers to write explicit qualifiers. The size-based fallback is now a single binary threshold: ≤ `--stack-auto-bytes` → `'stack`, above → `'heap` silently.
- **Syntax highlighting** — `mut` added to the `declaration-keywords` pattern in the tmLanguage files for VSCode, Eclipse, and Linguist.
- **`grammar.bnf`** — `let_stmt` now accepts `"let" | "mut" | "var"`.
- **`transpilation-modes.md` split** — qualifier inference content extracted to a dedicated `docs/qualifier-inference.md`; `transpilation-modes.md` now focuses on flags, qualifier vocabulary, and mode/threading behaviour.

### Fixed

- Enum disproportionate-variant warning threshold previously used the removed `stack_warn_bytes`; now derived from `stack_auto_bytes / 4`.
- Warning messages for oversized structs and enum variants now suggest `'heap` explicitly instead of the ambiguous `T'` sigil.

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

[0.7.0]: https://github.com/mlanoe/boring/releases/tag/v0.7.0
[0.6.0]: https://github.com/mlanoe/boring/releases/tag/v0.6.0
[0.5.0]: https://github.com/mlanoe/boring/releases/tag/v0.5.0
[0.4.0]: https://github.com/mlanoe/boring/releases/tag/v0.4.0
[0.3.0]: https://github.com/mlanoe/boring/releases/tag/v0.3.0
[0.2.1]: https://github.com/mlanoe/boring/releases/tag/v0.2.1
[0.2.0]: https://github.com/mlanoe/boring/releases/tag/v0.2.0
[0.1.0]: https://github.com/mlanoe/boring/releases/tag/v0.1.0
