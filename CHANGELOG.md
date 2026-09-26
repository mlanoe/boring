# Changelog

All notable changes to Boring are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- **`@singleton` attribute — general-purpose, DI-independent function memoization**, the first implemented piece of the dependency-injection design (`docs/design-notes/boring-di-draft.md`). Decorating any zero-parameter, non-generic, non-`throws`/`task`/`stream`, top-level free function memoizes its result behind a compiler-synthesized `LazyLock`: the body runs at most once, the first time the function is called from anywhere (a direct call, or an `@inject` site), and every later call — from anywhere — gets a `.clone()` of the same instance. Implemented in the transpiler (`Transpiler::emit_singleton_fn`, `src/transpiler/emit_top.rs`) by emitting the original function body unchanged under a mangled internal name, then adding a `static ...: LazyLock<T> = ...` plus a public wrapper that clones out of it — reuses `emit_fn`'s full existing machinery rather than a parallel codegen path. Checker-enforced (`check_di_provider_attrs`, `src/checker/mod.rs`): the return type can't be `'owned` (exclusive `Box<T>` can't back more than one caller), rejected under `--threading single` when the return type resolves to a non-`Sync` `Rc`/`RefCell`-based form (mirrors `'static`'s own pre-existing `Sync` gate), and rejected outright under `--target kernel` (`src/validator/kernel.rs`) — dependency injection assumes a full std runtime this target never has. `@provide`'s `pub` requirement is enforced the same way. Along the way, fixed a real parser gap this surfaced: an attribute (`@derive`, `@singleton`, or any other) written directly above a **return-type-first function with no explicit `def`/`req` keyword** (`NetworkClient'shared networkClient(): ...` — the form the DI design doc uses almost exclusively) used to be silently discarded, with zero error and zero effect (`src/parser/mod.rs`'s `TokenKind::At` dispatch, missing the `is_fn_decl_shorthand()` arm the un-attributed path already had). See `tests/dependency_injection.rs` (checker/codegen) and `tests/cases/singleton_di.br` (full compile+run, `transpile_test!`).

- **`@inject` — first working slice of dependency-injection field resolution** (`docs/design-notes/boring-di-draft.md` §1-§2), building on `@singleton`/`@provide` above. A struct field decorated `@inject` becomes an omittable, defaulted constructor argument, resolved against a `pub @provide` function returning the same type elsewhere in the program — `UserRepository()` (zero args) is rewritten to `UserRepository::new(networkClient())` at the omitting call site, exactly the same call-site-substitution mechanic Boring already uses for an ordinary defaulted `init` parameter. Implemented entirely as a new AST-level desugaring pass, `src/desugar_inject.rs` (wired into every `boring build`/`boring run`/GPU-target pipeline entry point in `src/main.rs`, right after `desugar_array_block`), which builds a `(base type) -> provider` registry from every `@provide` function in the program and, for each struct with an `@inject` field, synthesizes a real `init(...)` (one parameter per field, in declaration order, each assigned via `self.field = field` — the same shape `docs/book.md`'s own `Circle`-with-a-defaulted-parameter example already uses) — this reuses the transpiler's existing labeled-argument-call defaulting machinery (`emit_expr.rs`'s `try_emit_labeled_init_call`) completely unmodified, no new codegen. Validated: the accepted-qualifier set (`'shared`/`'actor`/`'guard`/`'observed`, `'owned` only without a `@singleton` provider; a bare/unqualified field is rejected for now, not yet inferred), exact qualifier match required against a `@singleton` provider's return type, ambiguous-provider and no-provider-found errors, and `@inject` combined with a struct's own hand-written `init` (rejected outright — v1 scope). `v1 scope, by design` (see the doc's own "Before implementation begins" for what's deliberately deferred): same-`Program` providers only (no `[deps]` cross-project resolution yet), no `id`/`env`, and no cycle detection yet. See `tests/dependency_injection.rs` (18 checker/codegen cases) and `tests/cases/inject_di.br` (full compile+run: two independent structs sharing one `@singleton`-provided instance with zero parameters threaded by hand — the `@EnvironmentObject`-shaped motivating case this whole design traces back to).

### Fixed

- **GPU-kernel helper functions/methods silently dropped their return value on all four backends (Metal, CUDA, ROCm, wgpu)** — a silent-correctness bug, not a compile error: a `kernel` struct's own `def <T> helper(...)` method (`docs/gpu-module.md`'s documented syntax) or a plain free function called from a kernel's `def ()` body compiled cleanly and ran to completion, but its implicit tail expression (the function's last statement, with no explicit `return`) was emitted as a bare, discarded statement (`(x + 1.0);`) instead of `return x + 1.0;` — so the caller always got back whatever value it passed in, unchanged. Any GPU-kernel code that factors logic into a helper function/method (e.g. dequantization math for a future quantized-model inference project) was affected on every backend. Root cause: each backend's `DeviceEmitter::emit_stmt` (`src/transpiler/{metal,cuda,rocm,wgpu}/device.rs` — four independent implementations, not a shared code path) had no tail-position awareness, unlike the Rust-target emitter (`src/transpiler/emit_stmt.rs`), which already threads an explicit `is_last: bool` from `emit_body` down through `If`/`Match` branches to give a function's tail expression Rust's own implicit-return treatment. Fixed by adopting the same `is_last` convention in all four backends, but emitting an explicit `return <expr>;` (C/CUDA/HIP/WGSL have no implicit tail-return, unlike Rust) whenever `is_last` is true and the enclosing function isn't `void`. See the new tail-expression regression tests in each `tests/*_codegen.rs` (source-snapshot assertions that `return` is emitted, per backend) plus `wgpu_codegen.rs`'s `test_kernel_helper_method_real_shader_value`, a real end-to-end GPU dispatch asserting the actual numeric output.

- **Interpreter (`boring run`): a kernel's own helper method called via explicit `self.helper(...)` from inside `def ()` failed with `no method 'helper' on type ''`** — `self`'s `ObjectInner.type_name` was hardcoded to an empty string in `run_one_kernel_thread` (`src/interpreter/eval_gpu.rs`), and the kernel type was never registered in the per-thread interpreter's own (fresh, empty) global environment (`Interpreter::new_for_kernel`), so `call_method`'s struct-dispatch lookup could never succeed regardless of the type name. A bare (non-`self.`) call to the same method already worked, via a separate free-function-injection path — only the idiomatic, explicit `self.method()` form (see `CLAUDE.md`'s "implicit self" note) was broken. Fixed by threading the kernel's declared name through to `run_one_kernel_thread` and registering a synthetic `StructDecl` (real `name`/`methods`, defaulted everything else) under that name in the per-thread global, so `self.helper(...)` dispatches exactly like an ordinary struct's own method call. This unblocks using `boring run` as a fast, GPU-free correctness check for kernel helper methods before a real GPU build. See `test_kernel_helper_method_via_self` in `src/interpreter/tests/tests_gpu.rs`.

- **`self.field.method()` on an `'actor`/`'guard`(/`'observed`) field now enforces the field's own `mut`/`var mut` permission** — a `def` (mutating) method dispatched through a struct field of one of these qualifiers used to skip the "not declared `mut`" checker diagnostic entirely, unlike the identical check that already fired for a local binding of the same qualifier (`try_emit_mutex_method`/`try_emit_rwlock_method`'s local-var branches), and unlike the plain (non-qualified) struct-field case, which was already correctly enforced by a separate, unaffected code path. Fixed with a new shared `Transpiler::check_field_def_call_mut_gate` helper (`src/transpiler/emit_methods.rs`), wired into both mutex/rwlock field-dispatch branches and the `'observed` field-dispatch functions (`try_emit_observed_field_method_direct`/`try_emit_observed_field_method`, including the `.value` escape hatch) alike — the two families had been left in deliberate, documented parity on the bug rather than the fix. Also fixes `qualified_named_type_name` to peel an arbitrary number of nested ownership-qualifier layers instead of just one, needed for the doubly-qualified `'actor'observed`/`'guard'observed` shape to resolve to its struct name at all. See `docs/book.md`'s `'observed` section ("Known gaps") and `CLAUDE.md`'s matching note.

- **Implicit self ("bare field access inside a method resolves to `self.field` automatically" — `CLAUDE.md`) didn't route an `'actor`/`'guard`/`'observed` field through its lock at all, unlike the identical `self.field` spelling** — found while validating `@inject` above (confirmed pre-existing and unrelated to it). Every shape-based receiver recognizer in `src/transpiler/emit_methods.rs` (`try_emit_mutex_method`/`try_emit_rwlock_method`/`try_emit_actor_field_method`/`resolve_observed_field_receiver`, and the `.value`-hop resolution in `resolve_observed_value_base`) pattern-matches the literal parsed shape `Field(Var("self"), field)`; implicit self is a source-level convention only, so a bare `field.method()`/`field.value.property` parses to a plain `Var("field")`/`Field(Var("field"), "value")` instead, never producing that shape. Every one of those recognizers declined, so the call fell through to the generic fallback and emitted an unlocked, unnotified call straight on the field's real wrapper type (`Rc<RefCell<T>>`/`Arc<Mutex<T>>`/`BoringObserved<T>`) — usually a hard `cargo build` E0599/E0609 — and, more seriously, silently skipped `check_field_def_call_mut_gate` (above) too: calling a `def` method on a bare non-`mut` field compiled with **zero** Boring-level diagnostic, unlike `self.field.method()`'s correct rejection. Fixed with a new `Transpiler::normalize_implicit_self_field` helper (`src/transpiler/emit_methods.rs`) that rewrites a bare implicit-self-field receiver into the equivalent explicit `Field(Var("self"), field)` `Expr` shape up front — in `emit_method_call`, and at the two `.value`-hop resolution points in `try_emit_observed_field_method`/`resolve_observed_value_base` — so the rest of the dispatch chain (and its mut-gating) can no longer tell the bare and explicit forms apart. See `tests/implicit_self_field_dispatch.rs` (new file, 9 cases covering plain `'actor`/`'guard`, `'observed` method-call + `.value` field-read dispatch, and mut-gate rejection parity for the bare form).

- **`'atomic` now gets the same `mut`/`var mut` binding-permission discipline as `'actor`/`'guard`** — a bare `let x'atomic = 0; x += 5` used to compile (no `mut`/`var mut` required at all for the mutating operation set: compound assignment, plain store, `.swap()`), inconsistent with `'atomic`'s documented membership in the `'actor`/`'guard` family, where a bare `let` is read-only. Enforced in `src/checker/mod.rs`'s `check_assign_target` (assignment-shaped ops) and `src/transpiler/emit_methods.rs`'s `try_emit_atomic_method` (`.swap()`); `var` alone now also requires promotion to `var mut` since a scalar `'atomic` binding has no separate rebind-the-pointer operation the way a struct `'actor`/`'guard` does. The automatic `'actor`/`'guard` → `'atomic` promotion pass (`promote_atomic.rs`) is unaffected — promoted names keep whatever permission their pre-promotion `'actor`/`'guard` declaration already had. See `docs/qualifiers.md`'s new "Binding permission" subsection and `docs/book.md` §21's "Binding × qualifier combinations" table (`'atomic` column added).

---

## [0.9.8] — 2026-09-13 *(cargo test: 1974/1974 passing across 38 suites, 8 ignored · clippy: clean · self-hosted interpreter functional: 83/83 × 4 modes)*

### Added

- **`'atomic` ownership qualifier + automatic `'actor`/`'guard` → `'atomic` promotion pass** — a new qualifier for scalar `int`/`bool` locals (`Arc<AtomicX>` multi-thread / `Rc<Cell<X>>` single-thread), with lock-free `load`/`store`/`fetch_add`/`fetch_sub`/`swap` mapping (all `Ordering::SeqCst`) and a hard rejection of `with` on an `'atomic`-qualified binding. Alongside it, a conservative, fully automatic optimization pass (`src/transpiler/promote_atomic.rs`) promotes an already `'actor`/`'guard`-resolved scalar local to `'atomic` whenever it provably never escapes its function, every access decomposes into a single recognized atomic primitive, and it's never used inside a `with` block — applies identically to `'actor` and `'guard` sources, absence of proof never upgrades. Also fixes a related pre-existing gap where a scalar `'actor`/`'guard` local initialized from a plain literal (not a constructor call) got an unresolved `Mutex<_>` placeholder type. Deferred as documented future work: compare-and-swap pattern detection, cross-function whole-program promotion, `.swap()` in the interpreter, ordering-tuning surface. See `docs/qualifiers.md`'s `'atomic` section and `docs/book.md` ch. 21/30.
- **`'actor'task`/`'guard'task` inference extended beyond task-method calls** — the qualifier-inference pass previously only picked the `'task` lock variant when a `task`-declared method was actually called on the captured value; it now also detects a `with` block that reads/writes the guarded value with no task-method call inside it, but which itself spans a genuine `await` (a `wait(...)`, a `join […]`, or a call into an already-known `task` function/method) — via a new bounded live-range scan (`collect_with_await_signals`). Cross-function propagation was also extended: a callee's signature demanding `ActorTask`/`GuardTask` now correctly upgrades a caller's plain `Actor`/`Guard` candidate instead of intersecting to a spurious "no valid qualifier" error. Conservative by construction — absence of proof never upgrades to the task variant; a residual forward-reference gap (parallel to `with`'s own accepted read/write false-negative) is documented in `docs/qualifiers.md`, not silently guessed at.
- **`..=` (inclusive) now accepted for array-alloc, fill, and comprehension**, closing an asymmetry where it already worked for bare ranges and slices but not these forms: `[..=n]` allocates n+1 elements, `[v for ..=n]` fills n+1 elements, `[f(i) for i in ..=n]` (and its chained multi-dim form) comprehends over 0..=n. Desugars to a synthesized `n + 1` at parse time, so no transpiler/interpreter/GPU-backend codegen changes were needed.

### Changed

- **Exclusive range operator renamed `..` → `..<`** (Swift-style) — bare `x..y` read ambiguously (does "from x to y" include y?); Boring already borrows Swift syntax elsewhere, so this adopts Swift's own fix verbatim rather than inventing a new spelling. Applies everywhere a bare `..` meant "exclusive range": `range_expr`, `slice_range` (`a[M..N]` → `a[M..<N]`, `a[..N]` → `a[..<N]`), `for`-loops (`for k in 1..5:` → `for k in 1..<5:`), array comprehension (`for i in ..n` → `for i in ..<n`, incl. the chained multi-dim form), array alloc (`[..n]` → `[..<n]`), and fill's dotted form (`[v for ..n]` → `[v for ..<n]`). The old bare `..` in these positions is now a **parse error** pointing at `..<` — no alias kept, matching how the earlier `'stack`/`'heap` → `'inline`/`'owned` rename was handled. `x..=y` (inclusive) is unaffected, and `a[M..]`/`a[..]` (no upper bound, nothing to disambiguate) keep their bare `..`. `..expr` as a spread argument (unpacks into a variadic slot) is a distinct prefix operator, unrelated to ranges, and is also unaffected. See `docs/book.md` and `spec/grammar.bnf`.

### Fixed

Qualifier/lock correctness (parser and transpiler):
- **`T'actor|<anything>` (e.g. `'actor|shared`, `'actor|guard`, `'actor|atomic`) failed to parse** with a misleading "expected Eq, got Ident" error, while the same union with `'actor` in any other position already worked — the `let`/`var` type-vs-name lookahead had no awareness of a pipe-separated qualifier union continuation after `'actor` (also affected `'guard`/`'static` as a union's first member, fixed the same way).
- **A qualifier written on both the type and the name** (`let Type'qual1 name'qual2 = value`) parsed without error into a doubly-nested, semantically undefined qualified type — e.g. `let Counter'actor c'guard = Counter(0)` transpiled to a stray, invalid `&Arc::new(...)`. Now rejected with a `ParseError` naming both positions; legitimate single-position and compound-chain (`'actor'task`) forms are unaffected.
- **A scalar (`int`/`uint`/`bool`/`float`/…) local qualified `'actor`/`'guard` never routed its own bare-variable read/write through the lock** — only struct-typed `'actor`/`'guard` bindings were ever exercised in practice (every mutation there is a field write or `def`-method call, both already lock-routed); a scalar binding has no fields, so every mutation *is* a bare-variable read/write, producing invalid Rust (`E0368` — no arithmetic ops on the raw `Arc<Mutex<T>>` handle itself). Fixed for both the top-level case and inside a `with` block.
- **A parameter whose `'actor'task`/`'guard'task` (etc.) qualifier came from inference rather than an explicit annotation got the correct Rust signature but broken body codegen** — `seed_param_locals` populated the mutex/rwlock/arc tracking sets from a bare parameter's declared type only, never consulting the inference pass's own results, so a method call or `with` block on it never routed through `.lock()`/`.lock().await`/`.read()`/`.write()`. Also fixes a related stale-state bug this surfaced: a `type def`/`type req` method's own inferred-qualifiers map could leak stale entries from whichever function was emitted immediately before it, wrongly treating an unrelated same-named parameter as mutex/rwlock/arc-qualified.
- **`check_static_provenance` only recognized a "fresh construction" via an uppercase-first-letter callee heuristic** — a call to an ordinary lowercase wrapper function/method that itself constructs and returns a fresh value (`def A create(): A()`) sailed through unrejected as a `'static` initializer outside `main`/top level, silently violating the provenance guarantee the gate exists to enforce. Now mirrors the stricter, already-existing argument-provenance model: outside an authorized site, only a bare `'static`-typed variable is accepted, any `Call`/`MethodCall` is rejected regardless of callee spelling.

Owned-value / size-based auto-boxing correctness:
- **A bare (unqualified) function return type or oversized local sized past `--inline-auto-bytes` got a size-boxed (`Box<T>`) *type* with no matching change to the *value*** — a direct tail constructor-call return stayed an unwrapped literal (`E0308`), and a `let`-then-tail-variable return produced a doubly-wrong `Box<Box<T>>` annotation. Fixed by rewriting the bare oversized return type up front to the same `'owned`-qualified representation the rest of the pipeline already knows how to emit correctly, and by re-deriving a bare local's value (not just its type annotation) against its resolved qualifier.
- **The above size-based auto-boxing also applied in `--mode managed`**, where it isn't supposed to (documented strict-mode-only) — a bare oversized local got wrapped while the (correctly unpromoted) managed-mode function-return signature stayed unboxed, a guaranteed mismatch. Managed mode's local qualifier resolution is now mode-gated to match.
- **A bare oversized `&T`/`&mut T` parameter (resolved via the "universal borrow" pre-fallback) got re-boxed a second time by `emit_type`'s `Borrow`/`BorrowMut` rendering**, producing `&Box<T>`/`&mut Box<T>` against call sites still passing a plain struct reference (`E0308`) — same-shape fix as the return-type case, suppressing the size-boxing pass for an already-decided borrow.
- **A value already `'owned`-qualified (`Box<T>`) got re-wrapped in an extra `Box::new(...)`** when passed into a function/constructor call or used as a struct-operator method's RHS operand (`E0308`) — the wrap decision only inspected the *target* type's qualifier, never whether the *source* was already boxed. Also fixed a related bug where a `mut`/`var mut` `'owned`/`'new` binding emitted a bare, unwrapped struct literal against its own correctly-boxed type annotation.
- **A `def`-method call or field assignment on a non-`mut`/`var` `T'owned`/`T'inline` parameter (or a `'new` param resolved to one of those) raised no Boring-level diagnostic** — `boring build` "succeeded" emitting e.g. `fn bump(c: Box<Counter>)` (missing `mut`), failing only later at `cargo build` with a confusing raw `E0596`. The existing "not declared mut" checks only resolved a receiver's struct name for a plain `Type::Named` parameter; extended to also unwrap `Owned`/`Inline`/inference-resolved `'new` parameter types.
- **Passing an already-moved `'owned` local into a second struct-constructor call raised no Boring-level diagnostic** — `boring build --emit-rust` silently transpiled a double-move into Rust that only failed at `cargo build` with a raw `E0382` pointing at generated code the user never wrote. New checker-level use-after-move diagnostic scoped specifically to committed-`'owned` struct-constructor arguments (straight-line, single-block tracking — branches/loops/closures are a documented gap, not flow-sensitive analysis). Deliberately does **not** cover a plain function call to an `'owned` parameter (the transpiler clones instead of moving there unless the parameter is also `mut`/`var` — confirmed intentional, tested behavior) or `'new` (resolved too late, by the transpiler's own inference pass, to duplicate here).
- **The interpreter (`boring run`) invalidated the caller's variable after *any* call to a function with a matching `'owned` parameter, regardless of `mut`/`var`** — disagreeing with the transpiler, which only clones (never truly moves) at a plain call site unless the parameter is `mut`/`var`. Removed the interpreter's incorrect move/invalidation for this case so both backends agree.

Struct-literal and `init()` codegen:
- **Positional struct-literal construction (no custom `init()` body) emitted a bare borrow for `'actor`/`'guard`-qualified fields, a guaranteed type mismatch** — fixed by routing these fields through the same by-value Arc-qualified emission the `::new(args)` path already used.
- **`init()` bodies written with bare (non-`self.`-prefixed) field assignments hard-failed real `cargo build`** — bare assignment implicitly means `self.field` per book.md, but `emit_init`'s fast-path struct-literal check only recognized the explicit `self.field = expr` form, so a bare-assignment body always fell into the slow zero-prefill "general case" — a guaranteed `E0277` for any field whose type doesn't implement Rust's `Default`. Fixed by extending the fast-path check to recognize bare field-name assignment targets too.
- **A struct constructor call with a custom `init()` body didn't clone a `'shared`/`'actor`/`'guard` argument**, and a positional struct literal (no `init()`) didn't clone an `'actor`/`'guard` field either — both produced an owned move where a shared-reference clone was required.
- **A labeled-argument struct constructor call (`Name(label = value, ...)`) on a struct whose `init()` has a body always built a Rust struct literal keyed by argument label, regardless of whether the struct actually had a custom `init()`** — unlike the positional-argument path, which already routed to `::new(...)` correctly. Produced non-compiling Rust whenever an init parameter's name didn't match the real field name, and silently skipped the `init()` body's own logic entirely. Now reorders labeled args against the init's own parameter names and emits `Name::new(...)`, matching the positional path.

GPU kernels (interpreter and Metal/CUDA/ROCm/wgpu backends), all found while verifying `linguist/samples/gpu.br` end-to-end after 0.9.7:
- **Interpreter never bound a kernel's const-generic type params** (`kernel Foo<int W, int H>:`) — a body reference to `W`/`H` failed with "undefined variable" the moment it was used; now resolved from the turbofish construction and threaded through every launch.
- **Metal/CUDA/ROCm: a `'const`-qualified array field whose size is a const-generic expression** fell through every "is this an array field" check in host codegen, emitting an invalid unit-type (`()`) struct field.
- **Metal/CUDA/ROCm: a turbofish kernel construction (`Blur<3, 1>(...)`) wasn't transpiled at all**, nor recognized at its later dispatch site, degrading to a bogus ordinary function call.
- **Metal/CUDA/ROCm: `.reduce(seed, closure)`/`.fold(seed, closure)` on an array** inside kernel-touching code wasn't recognized by any of the three backends' own method-call table; now emits `.iter().cloned().fold(...)`.
- **Parser: a range's end bound ignored operator precedence** — `0..W * H` mis-parsed as `(0..W) * H` instead of `0..(W * H)`; fixed to parse the range end at multiplicative precedence, matching Rust's own.
- **Interpreter never auto-inserted GPU kernel barriers for a bare `'actor` field with no explicit `sync`** — per `docs/gpu-module.md`'s auto-barrier rules, already implemented in all four transpiler backends but missing from the interpreter's block-scoped thread simulation, so a kernel relying on the documented auto-barrier behavior (e.g. a tile-reduction sum) raced every thread to completion with no cross-thread ordering, silently producing zero instead of the correct sum.
- **wgpu: a turbofish const-generic kernel construction never populated the block-size lookup table**, silently dispatching every such kernel with a single GPU thread regardless of the requested block size, with no error or warning.
- **wgpu: a kernel body comparing an `i32` loop-derived index against a `.len()` call on a storage-buffer array field failed real WGSL shader validation** (`arrayLength(...)` returns `u32`; WGSL requires identical comparison operand types, unlike Rust/C) — fixed by casting to `i32` at the single emission choke point.
- **wgpu: a kernel params struct with a fixed-size scalar array field failed real WGSL shader validation** (`var<uniform>`'s std140-like layout requires 16-byte-aligned array strides) — fixed by switching such a params struct to `var<storage, read>` (std430, 4-byte alignment) whenever it has a fixed-array field.
- **wgpu: a const-generic kernel's monomorphized body never had its type-param references (`W`/`H`) substituted with their concrete values**, emitting undefined WGSL identifiers — unlike Metal/CUDA, which get this for free from Rust's own const generics. Also fixed, found alongside: `.len()` on a storage-buffer field emitting an invalid `len(buf)` call instead of `arrayLength(&buf)`, and indexing a `'const` fixed-array field emitting a bare unprefixed identifier instead of the documented params-struct-prefixed form.
- **wgpu: an implicitly-presented `'surface` buffer field (no explicit `.present()` call) resolved to an unprefixed default field name** that doesn't exist on the generated `__App` struct (`E0609`) — fixed by qualifying it with its owning kernel instance's own variable name.
- **wgpu: a const-generic kernel's turbofish construction (`KernelName<W, H>(...)`) wasn't recognized by wgpu's own kernel-construction special-casing**, falling through to ordinary generic-struct codegen against the pre-monomorphization name (`E0425`); also fixed alongside: a `'const` fixed-size array constructor argument was cast to a scalar instead of converted to its real array type, and the host-side `float`/`float64` scalar type stayed `f64` where device width required `f32`.

### Docs

- Added a rule-of-thumb callout for choosing between `let`/`mut`/`var`/`var mut` day-to-day (`docs/book.md` §2).
- Documented that a kernel struct's fields never alias the host variable passed into its constructor, for any qualifier (`docs/gpu-module.md`).
- `spec/grammar.bnf`'s `owner_qual` production was missing `'atomic` (added alongside the qualifier itself but never reflected in the grammar file) — added, with a matching semantics line.

---

## [0.9.7] — 2026-09-08 *(cargo test: 1845/1845 passing across 33 suites · clippy: clean · self-hosted interpreter functional: 83/83 × 4 modes)*

### Added

- **Turbofish monomorphization for generic structs, free functions, methods, and — via `?.` — optional-chained method calls** (`src/transpiler/monomorphize.rs`) — an explicit, fully-concrete turbofish call/construction site (`Struct<T>(...)`, `fn<T>(...)`, `obj.method<T>(...)`, `obj?.method<T>(...)`) now gets a specialized, non-generic Rust clone, alongside — not replacing — the existing `impl<T: Clone> ...` generic-emission path every non-turbofish use of the same generic still goes through unchanged. For a specialized instantiation this lifts three real Rust-generics limitations: a `type let` field depending on the struct's own type param no longer hard-errors when a concrete specialization exists (previously always rejected — a Rust `static`/`LazyLock` can't be generic); the specialized copy carries no forced `Clone` bound; and a `mut`-qualified type argument (`Container<mut Point>`) now works for the specialized copy. Resolves same-file, cross-file (`use`), and cross-project (`boring.toml [deps]`) call sites with no extra re-parse — a one-time global-declaration collection pass folded into the existing `deep_pre_scan` walk. Generic methods (`obj.method<T>(...)`) are specialized only when the method name resolves unambiguously to exactly one struct/ext-block/enum method with its own type parameter across the whole reachable file graph; an ambiguous or unresolvable name falls back safely to ordinary generic Rust. Known V1 limits: only turbofish call sites are specialized (inferred/non-turbofish call sites are untouched); same-file generic declarations only when the call site can't otherwise resolve cross-file; wgpu/cuda/metal/rocm kernel targets are untouched (separate, pre-existing const-generic-only monomorphizer). See `docs/book.md`'s "Turbofish monomorphization" section.

### Fixed

Recursion-depth guards, closing the remaining stack-overflow gaps left after 0.9.6's initial `MAX_EXPR_DEPTH` work:
- **`parse_or`/`parse_type`/`parse_block`/`parse_pattern` had no depth guard, or one only reachable through `parse_expr`'s own wrapper** — a chain of nested no-paren closures (`a: a: a: ... 0`), deeply nested array types, thousands of nested `if` blocks, or deeply nested match patterns each recursed unbounded and could `SIGABRT`. The guard moved to `parse_or` itself (the real shared entry point every path funnels through) plus independent guards on `parse_type`/`parse_block`/`parse_pattern`.
- **`resolve_interp` (string-interpolation parsing) started every nested interpolation hole with a brand-new `Parser`, discarding the enclosing depth counter** — a string interpolated inside another interpolated string, nested thousands of levels deep, recursed unbounded. Fixed by propagating `self.depth` into the sub-parser.
- **The semantic checker and kernel validator had no depth guard of their own**, relying entirely on the parser never handing them a too-deep AST — added an independent `MAX_CHECK_DEPTH` (200) to both as a second line of defense.
- **The interpreter's `call_fn`/`call_closure`/`call_type_method` cycle had no call-depth counter at all** — an ordinary recursive Boring function with no base case overflowed the real Rust call stack and aborted the whole process instead of raising a clean runtime error. Added a `call_depth` counter (`MAX_CALL_DEPTH` 500).
- **The self-hosted (Boring-in-Boring) parser's own recursion-depth guard (`parser_depth`/`_inc`/`_dec`) was a permanently-inert stub** — always returned 0, so `parse_unary`/`parse_primary`/`parse_pattern`/`parse_block` were completely unbounded. Implemented as a real `var int depth` field, extended the guard to all of these, and lowered this parser's own `MAX_EXPR_DEPTH` from 200 to 60 (the primary compiler's 200 assumes the 256 MB stack its own `main()` spawns for exactly this reason — `boring build`'s generated `main()` never does that). Also fixed a self-deadlock this surfaced: `p.depth = p.depth + 1` on an `'actor'task` (`Arc<Mutex<Parser>>`) locks the same non-reentrant mutex twice in one statement; worked around by reading into a local first.

Lexer/parser diagnostic accuracy:
- **`Token.col` was systematically wrong on every indented line** — `lex_line` runs on already-`trim_start()`'d content, and the resulting column was never re-offset by the stripped indentation, throwing off caret placement in nearly every real-world error message.
- **`Expr.len` captured the wrong token's length at 112 call sites** — computed *after* the (sub-)expression had already been parsed, measuring whatever token comes next rather than what was actually consumed (`12345 / 0` underlined `/` instead of `12345`). Added `Parser::span_len` and fixed all 112 sites; updated two stale diagnostic snapshots this surfaced.
- **Several parser-constructed spans/nodes hardcoded `col: 0`** (the implicit-`Return` wrapper for a single-expression function body, its "no type annotation" error span, a synthetic `type set` `Param`, `InitParam`, and `CatchClause`) instead of propagating a real column.
- **Mixed tab/space indentation was only checked within a single line** — two lines each internally consistent (one all-tabs, one all-spaces) went undetected and were silently compared after an arbitrary 1-tab=4-spaces conversion. Now tracks the file's first established indent style and rejects a later line indenting with the other character (mirrors CPython's `tabnanny`).
- **Hex/octal/binary integer literals above `i64::MAX` silently truncated** (`0xFFFFFFFFFFFFFFFF` → `-1`) instead of promoting to `UInt64` the way the decimal branch already did.

Parser correctness gaps:
- **`throws`/`task` inference from a closure body missed `Match`, `TryElse`, and `UnaryOp`** — a `task compute(v)` hidden inside a match arm was never detected, so the closure transpiled without the `async move` wrapper a real task call needs.
- **`'new` wasn't recognized in a `let`-statement's post-name qualifier lookahead** — `let c'new = Counter(0)` fell through to closure-shorthand parsing and choked on the bare `Tick` token, silently losing the initializer.
- **A type parameter appearing only in a function-typed parameter's own return type could be lost**, and **struct/trait `type def`/`type req` methods and trait signatures/defaults never enforced "every param needs an explicit type"** — both produced invalid Rust only caught by `cargo build`, not `boring build`.

Checker gaps:
- **`check_struct`/`check_enum`/`check_ext` never visited `inits`/`setters`/`conversions`, and a trait's default methods were never checked at all** — every checker rule (immutability, `mut`/`'shared`/`'static`/`'weak`, ...) was silently skipped inside an `init`/`setter`/conversion body or a trait default: `boring run` accepted illegal mutations there, and `boring build` transpiled them into Rust that violates documented semantics.
- **`mut [T]` (structural array mutation) with a `'shared`/`'static` element was wrongly rejected as if it were `mut T'shared`** — `type_has_shared`/`type_has_static` recursed into `Type::Array`, conflating the array's own structural-mutation axis with its element's qualifier. `mut [Point'shared] arr = []` now compiles as it always should have.

Interpreter runtime safety:
- **`INT_MIN / -1`, `INT_MIN % -1`, and `-INT_MIN` crashed the whole process** — `eval_div`'s plain `Int` arm and `UnaryOp::Neg` did raw arithmetic instead of `checked_div`/`checked_neg`; only `eval_rem` checked for a zero divisor, never MIN-by-`-1`. Reachable from ordinary arithmetic via `wrapping_sub` elsewhere in the same file, not just a crafted input.
- **`arr[i].min/max/swap/cas` with too few arguments, and negative/oversized allocation counts (`.repeat(n)`, array fill/alloc/comprehension), panicked or could OOM the process** — added arity checks and clamped negative counts to 0, matching the existing convention used elsewhere in the same files. A GPU kernel launch's block/grid dimensions are now also checked for `usize` overflow and capped before allocating.
- **A struct's instance setter recursing into itself (`set balance(balance): self.balance = balance`) stack-overflowed `boring run`** — the interpreter-side analog of the identical transpiler bug already fixed in 0.9.6; same `in_instance_setter` guard added on the interpreter side.
- **`.slice()` on `string`/`[T]` in the tree-walk interpreter didn't clamp negative or out-of-range indices** the way the native `a[m..n]` slice syntax already did — `arr.slice(-1, 3)` crashed instead of slicing from the end.
- **`floor`/`ceil`/`round`/`log`/`exp`/`tanh`/`clamp`/`sign`/`sum` rejected `float32`** with "expected number" (unlike `abs`/`sqrt`/`sin`/`cos`/`tan`, which already handled it), and **`isNaN`/`isInfinite` silently returned `false` for a real `float32` NaN/Infinity** instead of erroring loudly.
- **`assert_eq`/`assert_neq` compared Rust's derived `PartialEq`, sensitive to `Value`'s internal numeric-kind tag** — a default-kind `int` literal and an explicitly `int64`-typed value holding the same number are different enum variants, so `assert_eq(a, b)` silently failed even when `a == b` (Boring's own `==` operator, and the equivalent transpiled Rust, both correctly treat them as equal). Fixed by routing through the existing `values_equal` widening helper.
- **A `guard let ... else: panic(...)` (or single-line `else panic(...)`) inside a `throws` function failed to compile** (`E0308`: "`else` clause of `let...else` does not diverge") — `emit_guard` reused the function-body Ok-wrap padding logic for its else-block, appending a spurious `Ok(())` after the diverging `panic!` call.
- **An untyped, immutable `let x = "literal"` local was never promoted to `Rc<str>`/`Arc<str>`**, only a mutable one was — Rust then defaulted to `&'static str`, which compiled at the `let` site but mismatched downstream at a call site expecting the normal string representation (`error[E0308]: expected Arc<str>, found &str`).
- Self-hosted (Boring-in-Boring) interpreter: **the lexer's own hex/octal/binary literal parsing overflowed silently**, and **`UnaryOp.Neg` couldn't negate any fixed-width signed int** (`int8`..`int128`) — both now match the primary interpreter's equivalent handling.

Transpiler codegen crashes and correctness:
- **Dozens of builtins (`len`, `assert_eq`, `sum`, `clamp`, `atan2`, every math function and numeric cast, ...) indexed call arguments unconditionally in `emit_builtin_call`**, panicking `boring build` itself on too few arguments instead of failing cleanly. Added a minimum-arity table checked before dispatch. The same class of unguarded indexing panicked the transpiler for a top-level `var` with no initializer, a `T?`-let binding assuming any `throws` call is already `Option<T>`-shaped, `dict.contains()`/`.has()` with no arguments, and a variadic call site passing fewer arguments than its variadic parameter's position — all now degrade to a clean compile error (or, where valid, correct codegen) instead of a raw Rust index-out-of-bounds panic in the compiler itself.
- **A compound self-assignment through an `'actor`/`'guard`-qualified binding or field (`p.field += 1`, and every `op=` form) self-deadlocked at runtime** — it took the write lock, then, still holding it, took a second lock on the same non-reentrant mutex/rwlock to read the old value. Fixed by evaluating the RHS into a temporary before taking the write guard.
- **Qualifier-inference conflicts were reported via `eprintln!` instead of a real transpiler error** — a detected qualifier conflict printed to stderr but `boring build` still exited 0; the worst case (zero candidates remaining) also silently emitted a bare, unqualified type with no wrapper at all. Now a real build error via `push_error` (the advisory "annotate this explicitly" hint is a warning, correctly, not promoted to an error).
- **A setter whose parameter shares its field's name (`set balance(balance): self.balance = balance`) transpiled to a silent no-op, then — once that was fixed — an infinite-recursion stack overflow** — `emit_setter` never registered its own parameter as a local, so the RHS resolved as an implicit `self.balance` read; separately, the instance-setter dispatch had no guard against re-entering the very setter whose body it was emitting.
- **A bare `struct.field as T` / `list[idx] as T` numeric cast mis-transpiled to a string-parse fallback** (`.trim().parse::<T>()`) even when the field/element's type was statically known to be numeric — only the exact bare form was affected; `(expr * struct.field) as T` already worked.
- **`task(Duration): body` (`TaskWithTimeout`) had three real bugs**: a bare `?` tail inside a spawned future that doesn't itself return `Result`; the block-form body and its `try:`/`else:` case both inheriting the *enclosing* function's void-ness instead of treating themselves as their own value-producing scope; and `body_has_channel_or_task`/`expr_calls_task_fn` never recursing through `ExprKind::Field`, leaving `(task(dur): body).wait` in tail position un-promoted to `async fn` (a real `E0728` "await outside async fn").
- **A `kernel:` dispatch's `block = N` argument referencing a top-level scalar constant (`let N = 16` then `k(block = N)`) resolved to a hardcoded workgroup size of `1` on `--target wgpu`**, silently computing only the first thread's result — WGSL's `@workgroup_size` must be a compile-time constant, and the transpiler's block-size scan only recognized an integer literal, never a named constant. Now resolves a top-level `let NAME = <int literal>` the same way an inline literal already worked; a `block=` referencing anything else (a local/function-scoped variable, a computed expression) still falls back to the same conservative default as before. Found via this cycle's examples-verification pass — `examples/saxpy.br`'s wgpu build silently produced `y[i] = 1` for every `i` instead of `2*i + 1` (confirmed against `boring run`'s own interpreter simulation, which got it right).
- **A `kernel:` dispatch (`k(block = ...)`) with no `grid=` and no `LabeledArray` field to auto-infer one from silently dispatched exactly one workgroup/block regardless of the actual data size**, dropping every element beyond the first `block=` threads with no error or warning at all — the same examples-verification pass found this independently affected `examples/vector_add_gpu.br`'s wgpu build (`c[i]` correct for `i < 256`, silently `0` beyond it; `boring run`'s interpreter got every index right). This dispatch-emission code (`src/transpiler/emit_kernel.rs`) is shared verbatim by the default target and `--target wgpu`. Left as the existing default (a kernel whose `block=` already covers every element legitimately relies on it) but now flagged with a `boring build` warning naming the kernel and recommending an explicit `grid=` — CUDA/ROCm/Metal have their own separate, narrower auto-grid checks with the same silent-default gap, not changed here. Both `examples/saxpy.br` and `examples/vector_add_gpu.br`, plus `linguist/samples/gpu.br`, were updated to compute and pass the correct `grid=` regardless, since their actual data exceeds one workgroup.

GPU backends:
- **Metal: every atomic op cast its element to `atomic_long*` (8 bytes) regardless of the field's real element type** — a 4-byte `'actor'global`/`'actor'unified` `int32`/`uint32` field read/wrote 8 bytes at an address with only 4 valid, corrupting GPU memory or folding a neighboring element's bits into the result. Added `atomic_msl_cast`, deriving `atomic_int`/`atomic_uint`/`atomic_long` from the field's actual element type; widths with no portable MSL atomic now emit a flagged `/* ERROR: ... */` comment instead of miscompiling.
- **wgpu: `int`/`uint` (64-bit on every native GPU backend) silently narrowed to WGSL's 32-bit `i32`/`u32` with no diagnostic at all**, unlike the genuinely-unsupported 8/16/64/128-bit widths right next to it in the same code — added the same style of narrowing comment (`wgsl_narrowed_width`) directly in the generated shader.
- **All four GPU backends' auto-sync-barrier insertion (`first_loop_index`) missed a loop nested inside a top-level `if`** — e.g. a conditionally-guarded accumulation loop right after `if tid == 0: shared[0] = 0` — silently emitting the whole body, including its cross-thread reads, with no barrier: a real, silent race condition duplicated identically across metal/cuda/rocm/wgpu. Fixed once in the shared `helpers.rs` implementation.
- **A struct method that constructs/dispatches a kernel degraded to an `eprintln!` warning, then fell through to codegen already documented as producing `E0382`/`E0308` in the generated Rust** — promoted to a real, hard `boring build` error in all three non-wgpu backends (metal/cuda/rocm), naming the offending struct and line.
- **Kernel-validator (`--target kernel`, Rust-for-Linux `no_std`): `abs`/`floor`/`ceil`/`round` were wrongly rejected for `int`-typed arguments** — these four (unlike `sqrt`/`sin`/...) have a real, pure int no-op path in the interpreter and don't touch the FPU; a float-typed argument to any of them is still rejected independently, by its own literal or declared type.
- Kernel-target validator: a GPU kernel-launch expression reaching the kernel-target validator (defense in depth for an AST shape the parser doesn't currently construct) now rejects with a clear error instead of lowering to a syntactically-invalid Rust comment; the actually-reachable `block=`/`grid=`-labeled-arg `Call` shape of the same gap is now also rejected, not just the unreachable `KernelLaunch` node.
- wgpu: fixed a stale kernel-thread-stack-size doc comment (claimed 8 MB, code has used 64 MB for a while); fixed a use-after-move test fixture bug in `labeled_array.br`; added `MAX_ALLOC_COUNT` capping CPU-side array fill/alloc/comprehension and `.repeat()`; disambiguated synthesized shadow-axis names against a user field/binding spelled the same way.

Examples:
- **`examples/saxpy.br` (wgpu) and `examples/vector_add_gpu.br` (wgpu) silently produced wrong results** — both omitted `grid=` on a plain (non-`LabeledArray`) array field, hitting the always-dispatch-one-workgroup bug described above; `saxpy.br` additionally hit the symbolic-`block=`-resolution bug. Regenerated from scratch after both transpiler fixes and re-verified against `boring run`'s interpreter output. All six `--target wgpu` examples and both Metal examples (`game_of_life`, `plasma_metal`'s wgpu port; `game_of_life_metal`) rebuilt cleanly; the four headless compute ones (`saxpy`, `vector_add_gpu`, `matrix_mul_gpu`, `mandelbrot_gpu`) re-run and numerically re-verified.

Cleanup (no behavior change):
- Removed a dead no-op sort pass in `Array.sortedBy` and a dead, always-shadowed duplicate registration of `ord`/`chr`.
- Renamed the validator's `is_task_actor_or_guard` to `is_shared_actor_or_guard` and corrected its error message, which named the deprecated `'task` alias instead of `'shared`, the qualifier actually being checked.
- Factored `.await`/`.wait` dispatch (previously reimplemented at 3+ near-identical call sites, the documented source of past double-wrap regressions) into two shared helpers.
- Refactored the checker's three mut-binding constraint checks (`check_qualifier_constraint`/`check_tuple_mut_constraint`/`check_scalar_mut_constraint`), always called together with identical arguments, behind one `check_mut_constraints` entry point.
- Clippy: fixed `mem_replace_option_with_some` (use `Option::replace()`) and `type_complexity` (new `GenericMethodEntry`/`GenericMethodMap` type aliases) lints.

### Spec

- **`spec/grammar.bnf`**: added the `obj.method<T>(...)` and `obj?.method<T>(...)` generic-method-call productions introduced by this cycle's monomorphization work — `obj.method<T>(...)` previously had no distinct grammar production at all and silently mis-parsed as a chained comparison (`(obj.method < T) > (args)`).
- **`docs/wgpu-backend.md`, `docs/metal-backend.md`, `docs/kernel-target.md`, `docs/book.md`, `docs/qualifiers.md`, `docs/gpu-module.md`**: brought back in sync with this cycle's fixes, none of which had been reflected yet — the wgpu int/uint narrowing comment (was documented as silent), Metal's width-derived atomic casts (was documented as a blanket `atomic_long`), the kernel-target `abs`/`floor`/`ceil`/`round` int/float split (was documented as blanket-forbidden), the turbofish-monomorphization carve-out for a generic struct's type-dependent `type let` field (two passages in `book.md`/`qualifiers.md` still said this was rejected outright, one of them directly contradicting `book.md`'s own newer "Turbofish monomorphization" section), and the `kernel:` dispatch `grid=` contract (`gpu-module.md` had documented a `ceil(len/block)` 1D auto-inference for flat arrays that was never actually implemented — corrected to describe the real behavior instead: a single-workgroup default, now flagged with a build warning when it applies, `grid=` recommended explicitly whenever the field holds more than one `block=` worth of elements).

---

## [0.9.6] — 2026-09-05 *(cargo test: 1685/1685 passing across 27 suites · clippy: clean · self-hosted interpreter functional: 78/78 × 4 modes)*

### Added

- **`'static` qualifier** (`T'static → &'static T`) — a constant global instance with no refcount at all, more restrictive than `'shared`. Provenance-gated: only a genuinely `'static`-lived value (a top-level `let`/`const`, a `type let`/`type var` singleton, or another already-`'static` value) can be assigned into one — enforced by the checker, not just the parser. `'req` (the "always immutable, read-only-callable" qualifier group) now includes `'static` alongside `'shared`. See `docs/qualifiers.md`.
- **`Introspect` built-in trait** — no `trait Introspect:` declaration needed; gives any struct or enum Java-`getClass()`-style read-only reflection at runtime (`typeName()`, `fieldNames()`, and friends) via `as Introspect`.
- **Real per-adapter `GPU(n)` introspection on `--target wgpu`** — `GPU(n)`/`GPU.all()`/`.name()`/`.maxThreads()`/`.maxSharedMem()`/`.index()` now resolve against a real adapter list from `instance.enumerate_adapters(...)` instead of simulating a single device.
- **`Screen`/`screen.present()` on `--target cuda` and `--target rocm`** — neither has a native presentation API, so both take a software-blit path (device-to-host readback + `softbuffer` window presentation, winit 0.28 event loop) — the same rendering surface `--target wgpu`/`--target metal` already had.
- **`boring.toml [dependencies]` is now transitively resolved** — a dependency's own `[dependencies]` are followed too, not just the top level, so a `use` reaching a dependency-of-a-dependency now resolves. **`boring.lock`** pins every git dependency's resolved commit (same idea as `Cargo.lock`) so a branch/tag/default-branch dependency no longer silently drifts between builds; `boring update [name]` deliberately moves it forward, and `--locked`/`--offline` turn "silently resolve/refetch" into a hard error.
- **Real `use` module-import resolution in the self-hosted (Boring-in-Boring) interpreter** — previously every `use` form except the unrelated `use X as Y` type-alias shorthand was parsed but silently discarded at exec time.
- **Real `json(v)` serializer and `fromJson<T>()`** in the tree-walk interpreter — both were stubs (`json(v)` printed the interpreter's own `Value` debug repr, not JSON; `fromJson<T>()` returned its string argument unparsed for every `T`), so `boring run` and `boring build` disagreed for any program that serializes. Fixed enum `Display` to match `Debug` as part of the same fix.
- **`@derive(Serialize/Deserialize)` now actually compiles**, plus field renaming support for camelCase JSON keys against snake_case Boring fields.

### Changed

- **BREAKING: `args()` now follows the C/Python argv[0] convention** — `args()[0]` is the program name (the `.br` script path under `boring run`, the binary's own invoked path under a `boring build` binary — so it reflects renames/symlinks/aliases), and `args()[1..]` are the program's real arguments. Previously `args()[0]` was already the first real argument (the program name was silently excluded in every mode). `raw_args()` is aligned the same way, for consistency. Every existing `args()`/`raw_args()` call site needs its indices shifted by one (or its loop changed to skip index 0) — see `docs/book.md`'s "Global functions" entry for the exact new contract.
- **`'stack`/`'heap` renamed to `'inline`/`'owned`**, bare tick (`T'`) retired in favor of explicit `'new` — the old names implied a literal memory region ownership qualifiers never actually guaranteed.
- **`'mut`/`'req` qualifier groups are now recognized on parameters**, not just local bindings — a parser gap fixed this cycle (`'mut` → any qualifier with interior mutability; `'req` → always-immutable, now including `'static`).
- **Built-in enum variants are now seeded once per build, not once per file** — a performance fix with no behavior change for correct programs.

### Fixed

Beyond the many individual transpiler/interpreter correctness fixes landed this cycle (dict-of-dict chained assignment, `T?`-returning methods double-wrapping or mis-handling `else default`, struct/enum field mutation checks in `guard`/`if let`/`elif let`, `try`/`try?` prefix parsing inside those same clauses, tuple/array mutable slot bindings, numeric `as` casts on `expr else default`, and more — see the commit log for the full list), this release's examples-verification pass (regenerating and `naga`-validating every `--target wgpu` example from scratch) turned up and fixed:

- **`float32` values ignored `:.Nf` precision and crashed on `:e`/`:E`** in the tree-walk interpreter's `apply_format` — `to_float` didn't match `Value::Float32`, so a `float32`-typed value silently fell through to string-precision truncation (`"{x:.4f}"` on `2.4424f32` printed `"2.44"` instead of `"2.4424"`) and threw a hard error on `:e`/`:E`. Fixed by adding the missing match arm.
- **`{v:?}` dropped the trailing `.0` on whole-number floats** (both `float32` and `float64`) in the same interpreter, and independently in the self-hosted (Boring-in-Boring) interpreter's own `format_value` — neither had a real Debug-style branch for `?`, both fell back to plain Display. Fixed in the native interpreter by routing through the existing `debug_repr` helper, and in the self-hosted interpreter with a new `float_debug_str` helper.
- **wgpu: a leading `#` comment (or any top-level comment) before/around a `Screen` program failed the whole build**, one spurious error per comment line — a bare comment parses as a real top-level `Stmt::Comment` (never stripped by the lexer), which the Screen-program shape validator didn't special-case, unlike every other backend's kernel-body statement emitter. Broke regenerating `examples/game_of_life_wgpu` and `examples/plasma_metal_wgpu` outright.
- **wgpu: `float32(...)` casts inside a kernel body emitted invalid WGSL** — the builtin-function name mapping had a `float` → `f32` arm but no `float32` arm, so the call passed through unmapped. `float64(...)` now also produces the same clean compile-error comment the type-level check already gives instead of passing through as invalid WGSL.
- **wgpu: a labeled dynamic-shape array's desugared shadow-axis fields (`__name_axis0`/`__name_axis1`/...) used a `__`-prefixed name**, which WGSL specifically reserves — confirmed via a real `naga` parse (`Identifier starts with a reserved prefix`). Broke `examples/mandelbrot_gpu_wgpu` both as previously checked in and as freshly regenerated (two distinct instances of the same root cause). Fixed by sanitizing any `__`-prefixed field name reaching WGSL codegen to the existing `bp_`-prefixed convention this backend already uses elsewhere, applied consistently across the params-uniform struct, its unpacking locals, and kernel-body variable references.

All six `--target wgpu` examples (`saxpy`, `vector_add_gpu`, `matrix_mul_gpu`, `mandelbrot_gpu`, `game_of_life`, `plasma_metal`) now regenerate cleanly, `cargo build` cleanly, and their emitted WGSL parses cleanly under a real `naga` parser — not just `cargo build`, which never actually parses the embedded shader text.

### Spec

- **`spec/grammar.bnf`** brought back in sync with this cycle's parser changes, none of which had been reflected yet: added `'static` to the ownership-qualifier comment block and production; added the previously-promised-but-never-written "caller-facing qualifier groups" section (`'one`/`'many`/`'mut`/`'req`); added `const` to `owner_qual`; added the missing `type_member_decl` production for `type let`/`type var` singleton fields; added `type` to the reserved-keywords list; and fixed two stale `T'task → Arc<T>` mappings (long predating this cycle) that should have read `Arc<Mutex<T>>` (`T'task` is an alias for `T'actor'task`, not a read-only-sharing qualifier).
- **`linguist/Boring.tmLanguage.json`**: the `ownership-qualifier` regex now matches `'static` and `'const` (previously fell through to a two-scope fallback instead of highlighting as one qualifier token); `builtin-types` now matches `never`. `linguist/samples/*.br` re-verified against the current parser/interpreter — no drift found.
- Removed `docs/mut-type-modifier.html`, a stale generated file with no corresponding `.md` source (content was folded into `book.md` in an earlier cycle).

---

## [0.9.5] — 2026-08-21 *(interpreter: 642/642 · functional: 76/76 × 4 modes)*

### Added

- **`with <name>[, <name>...]:` scoped-access blocks** — a lexically-scoped block that grants extended, multi-statement access to a value normally touched one operation at a time: eliminates the host round-trip on every `'gpu'unified`/`'gpu'global` kernel-chain step, and lets `'actor'`/`'guard'` hold a lock across several statements instead of acquiring/releasing on every call. Read-only vs. mutating access is inferred from the value's own `let`/`mut`/`var` binding plus a bounded scan of the block body — no separate read/write keyword. Implemented across the checker, interpreter (no-op wrapper), and every host transpile target. See [docs/scoped-access-blocks.md](docs/scoped-access-blocks.md).
- **`mut` as a type modifier** (`mut Type`, `mut Type&`) — composes into tuple slots, struct fields, array/dict/set elements, and borrows, not just a bare local. See [docs/book.md](docs/book.md#2-variables-and-mutability) (design proposal originally tracked in `docs/mut-type-modifier.md`, since folded into the language reference).
- **Fixed-width integer scalars** — `int8`/`16`/`32`/`64`/`128` and `uint8`/`16`/`32`/`64`/`128` as real distinct types across every target (interpreter, `boring build`, kernel `no_std`, wgpu/CUDA/Metal, and the self-hosted Boring-in-Boring interpreter). Mixing two distinct fixed-width types requires an explicit `as` cast; each GPU backend enforces its own real width support (WGSL 32-bit only, MSL no 64/128-bit, CUDA full range via `__int128`).
- **`float32`/`float64` fixed-width float types**, and `mut`-qualified enum variant fields.
- **`--target rocm`** — new AMD GPU/HIP backend, joining wgpu/CUDA/Metal.
- **GPU residency across function-call boundaries** — a `'gpu'unified`/`'gpu'global` value now survives being passed into or returned from another function (including transitively, through multi-hop parameter forwarding and resident-tuple returns), collapsing a kernel-chaining pipeline down to a single host round-trip instead of one per call.
- **`gpu.warp.*` warp-level primitives**, classified GPU error reporting, and real cross-thread `'sync` barrier semantics in the interpreter's kernel simulation (genuine OS-thread barriers within a block, not a no-op) — makes `boring run` a faithful simulator for manual-mode `'sync` kernels like tiled GEMM.
- **`Image<T>`/`Volume<T>`** dynamic-shape GPU buffer types; **multi-dimensional arrays**; labeled-array axes exposed as read-only properties (`a.axis`).
- **`GPU` type on `--target wgpu`** — `GPU(n)`, `GPU.all()`, and instance methods (`.name()`, `.totalMem()`, `.warpSize()`, etc.) now emit real codegen against the single adapter wgpu opens.
- **`.pointee`** postfix dereference for opaque Rust types; **`_` fill-rest marker** for struct construction; empty struct body via indented `pass`.
- **`boring.toml` `[dependencies]`** section (plain `boring build`, not `--emit-rust`) and **`[external_types]`** section (with an `include` key to pull in a shared whitelist file) to supplement the transpiler's built-in external-type knowledge.
- **First-party `boring.*` stdlib** wired up as real modules (previously placeholders).
- **String slicing** (`s[lo..hi]`) and **25 float math methods** (`.sqrt`, `.cos`, `.pow`, `.tanh`, etc.) callable via method syntax, not just as free functions.

### Changed

- **`int`/`uint` now transpile to `isize`/`usize`** (previously `i64`/`u64`), consistent with the new fixed-width types sitting alongside them.
- **Array push / index-assignment are now O(1) amortized** — `Value::Array` moved from a plain `Vec` to `Rc<Vec<Value>>` with copy-on-write, replacing an implicit full-array clone on every mutating call (10k pushes: ~107s → ~0.15s).
- **wgpu shader/pipeline compiled once per kernel**, lazily behind `OnceLock`, instead of once per kernel *instance* — up to ~2x wall-clock on programs with many dispatches.
- **Metal backend**: real GPU buffer residency and deferred command-buffer sync instead of synchronous per-call transfer.
- **The self-hosted (Boring-in-Boring) interpreter is now genuinely runnable end-to-end** and accepts the same launch parameters as `boring run` (`[--gpu <profile>] [file.br] [-- args...]`). Fixed two bugs that had silently corrupted it since its introduction: `.append(x)` used where `.push(x)` was meant (~215 call sites), and labeled struct construction written as `Name(field: value)` instead of `Name(field= value)` (misparsed as a closure).
- **Clippy is now blocking in CI** (`-D warnings`); all outstanding warnings across the compiler were fixed.

### Fixed

- **`for i, v in <plain array>:` auto-enumerate shorthand was silently broken under `boring build`** — documented (book.md's "`for` with index") and correctly implemented in the interpreter since day one, but the transpiler's general `for`-loop codegen only ever handled the two-variable case as dict/tuple-array destructuring, never injecting the implicit `.enumerate()` a non-tuple array needs. Any `for i, v in arr:` over a plain (non-tuple-element) array failed to compile under **every** `boring build` target — long-standing, only masked because the regression test for this exact pattern (`tests/cases/for_destructure.br`) was wired into the interpreter suite but never into the transpile suite. Fixed by detecting, at transpile time, whether the iterable is already tuple-shaped (a `HashMap`/dict, or an array of tuple literals) via its declared type, its tracked dict-ness, or its recorded initializer's own shape — and injecting `.enumerate()` (with an `as isize` cast on the index) otherwise. `for_destructure` is now also registered in `tests/transpile.rs` across all 4 modes so this can't regress silently again.
- Cross-type numeric equality (`int`/`uint`/`uint8`/`float`) in both the interpreter and transpiler.
- `@derive(...)` before `pub struct`/`pub def` no longer silently drops the attribute.
- Auto-ref borrowing extended to array/dict/set parameters on free functions (previously struct/enum params only) — avoids a full clone on every call.
- `use` import resolution now also searches a project's `src/` directory from `boring run`/`boring build`'s GPU targets, matching non-GPU builds.
- Numerous GPU-backend correctness fixes across CUDA/Metal/ROCm/wgpu: kernel block/grid dispatch scanning recursing into function bodies, unreachable free-function WGSL emission, cross-kernel buffer-name collisions, zero-sized output buffers, array-index method-call receivers being cloned before mutation, unary/range operator precedence, top-level scalar inlining, and unsupported kernel/async patterns now reported as clean compile errors instead of panicking.
- **wgpu backend: a `Type::LabeledArray` kernel field (`[T, width = .., height = ..]'global`/`'unified`) was silently dropped from every host-side codegen path** — the multi-dimensional-array feature added `LabeledArray` alongside `Array`/`ArrayN`, but the wgpu host struct/constructor/bind-group/copy-accessor emission (`transpiler::wgpu::host`) and the shared kernel-construction-call codegen (`transpiler::emit_kernel`) still matched only the latter two. The field's struct field, GPU buffer, and bind-group entry vanished entirely, and its constructor argument silently cast straight to `i64` instead of being uploaded (`examples/matrix_mul_gpu_wgpu`, previously a Known Issue below).
- **wgpu backend: an array comprehension's implicit loop var (`[expr for i in 0..n]`) still cast to `i64`** — a leftover from this release's `int`/`uint` → `isize`/`usize` move that only this one codegen path never picked up, producing a `Vec<i64>` where an explicitly `[int]`-typed (`Vec<isize>`) binding was expected (`examples/vector_add_gpu_wgpu`, previously a Known Issue below).
- **wgpu backend: a kernel's `init()`-body output-fill count referencing a promoted top-level `const` used the stale pre-promotion name** — `field = [0 for ..n]` inside `init()`, where `n` names a top-level `let n = ...` rather than an init parameter, reproduced the boring-source identifier verbatim instead of resolving it through the same uppercasing rewrite (`gpu_top_level_const_names`) every other read site already gets (`examples/vector_add_gpu_wgpu`, previously a Known Issue below).
- **wgpu backend: a bare `float(expr)` scalar kernel-field assignment (`k.t = float(...)`) narrowed to `f32` instead of `f64`** — `float` is a pure alias of `float64`, not its own type, but the host-side scalar-assignment codegen grouped it with `float32`'s device-only 32-bit narrowing, mismatching a `var float t` field's real `f64` host-struct type (`examples/plasma_metal_wgpu`, previously a Known Issue below).

### Spec

- **`spec/grammar.bnf`** brought back in sync with this cycle's parser changes: the `_` fill-rest marker in call args, an indented `struct Foo:\n    pass` body, the `'sync` → `'actor` kernel-qualifier rename (plus the new `'actor'global`/`'actor'unified` atomic forms), and `mut`-qualified enum variant fields in the `mut`-generalization notes. `.pointee` and labeled-array axis properties (`a.width`) were confirmed to already parse via the existing generic field-access production and got a one-line clarifying comment each.
- **`linguist/Boring.tmLanguage.json`** (TextMate/GitHub-Linguist grammar): `builtin-types` now also matches `float32`/`float64` and every fixed-width alias (`int8`..`int128`, `uint8`..`uint128`, `i8`/`u8`/… , `f32`/`f64`) — none of these were highlighted as types before, including ones that predate this release.
- **`linguist/samples/gpu.br`** no longer fails to parse: it was still written against the pre-rename `'sync` qualifier and a since-removed kernel-dispatch pipe syntax (`kernel(block=..) k |> .wait`), plus a couple of unrelated stale constructs (`.map()` on a `Range`, `[v] * n` array-repeat). Updated to current syntax throughout; the Saxpy kernel now runs end-to-end via `boring run`. The Blur/TileSum kernels in the same sample still hit two separate, pre-existing gaps unrelated to this fix (see Known Issues) — not addressed here.

### Known Issues

Found while regenerating the `examples/*_wgpu` projects against the current compiler for this release (the committed projects were last generated against 0.9.3-era codegen). `matrix_mul_gpu_wgpu`, `vector_add_gpu_wgpu`, and `plasma_metal_wgpu` were regenerated and fixed after this entry was first written — see the four wgpu-backend bullets above under Fixed — leaving one unrelated item still open:

- The `linguist/samples/gpu.br` `Blur` kernel (`[float, W * H]'const weights`, a const-generic-sized array field) isn't resolvable by `boring run`'s kernel simulation (`undefined variable 'W'`), and the same file's `TileSum` kernel hits an f32/f64 buffer-element width mismatch (its `[float]` — i.e. `float64` — fields aren't representable in WGSL storage) when built for `--target wgpu`. Device-side, not the host-side scalar-field-cast class of bug the Fixed entries above address — left for its own fix.

`examples/saxpy_wgpu` was the one `examples/*_wgpu` project already broken in its *committed* form before this release (missing `.enumerate()` on `for i, v in k.y:`, the same root cause as the auto-enumerate Fixed entry above) — that one was regenerated and fixed earlier in this same release cycle.

### Testing

- Full suite: 642 interpreter unit tests (up from 445) and 76 functional cases × 4 transpile-mode combinations (up from 69), all green.
- The self-hosted interpreter's own build+functional suite (`interpreter_build.rs` / `interpreter_functional.rs`) is verified in all four mode/threading combinations, per [CLAUDE.md](CLAUDE.md).
- `tests/wgpu_codegen.rs` gained 4 regression tests for the `LabeledArray`/isize/const-promotion/`float`-alias bugs above (48 wgpu snapshot tests total), and `matrix_mul_gpu_wgpu`/`vector_add_gpu_wgpu`/`plasma_metal_wgpu` were regenerated against the fixed compiler and confirmed with a real `cargo check`.
- Added `for_destructure` to `tests/transpile.rs` (all 4 mode combinations) — the auto-enumerate/dict/tuple-array `for`-loop cases were previously interpreter-only.

---

## [0.9.4] — 2026-07-16 *(interpreter: 445/445 · functional: 69/69 × 4 modes)*

### Added

- **Array slice syntax** — `a[M..N]`, `a[..N]`, `a[M..]`, `a[..]`, and `a[M..=N]` (inclusive) extract a sub-array as a new `[T]`. All five forms work in the interpreter and transpile to `arr[M..N].to_vec()` in Rust. Out-of-bounds indices are clamped; negative indices count from the end.

### Spec

- **`spec/grammar.bnf`** — added `slice_range` non-terminal and a new `postfix` alternative `"[" slice_range "]"` covering all six slice forms.

---

## [0.9.3] — 2026-07-15 *(interpreter: 433/433 · functional: 69/69 × 4 modes)*

### Added

- **wgpu GPU backend** — `boring build --target wgpu` transpiles `.br` source to a self-contained Rust/wgpu Cargo project targeting DirectX 12, Vulkan, and Metal. Supports compute kernels, ping-pong buffers, screen display via a blit render pipeline, and the winit `ApplicationHandler` API.

- **Const generic params on kernel structs** — `kernel Foo<int W, int H>:` declares compile-time constants. Array sizes may reference them directly or as arithmetic expressions (`[float, W * H]`). wgpu monomorphises each unique instantiation into a distinct WGSL shader with concrete sizes.

- **Default values on scalar kernel fields** — `let float sigma = 1.0` initialises the field to a literal and omits it from the generated `new()` constructor signature. Array/buffer fields cannot carry a default (parser rejects them). Applies to all three GPU backends: wgpu, CUDA, Metal.

- **`mut` on scalar types** — `mut x = 42` is now valid and equivalent to `var x = 42` for primitives (`int`, `uint`, `float`, `bool`). The mutability/rebinding distinction is meaningless for value types; `mut` naturally means rebindable. Rejected for struct and array bindings.

### Changed

- **`--mode managed` output directory** — managed-mode builds now write to a distinct directory (`main_rust_managed` / `main_rust_managed_single`) instead of overwriting the strict-mode project. All four combinations `strict/managed × multi/single` now have isolated output directories.

- **`#[track_caller]` excluded from `fn main`** — in managed mode the attribute is no longer emitted on the entry point, which Rust forbids.

### Testing

- **Test suite runs in all 4 transpiler modes** — every interpreter unit test (`run` / `run_src`) now also transpiles its source through all four mode combinations (Strict×Multi, Strict×Single, Managed×Multi, Managed×Single) and asserts zero transpiler errors.

- **`interpreter_build` covers all 4 modes** — previously only `strict+multi` and `strict+single` were built; all four variants are now compiled and verified.

- **`interpreter_functional` tests against all 4 binaries** — each of the 69 `.br` case files is run against the four transpiled interpreter binaries and output compared against `.expected`.

- **`interpreter_functional` binary path** — the test now scans `target/` dynamically instead of a hardcoded path, making it work on Windows (target-triple subdirectory, `.exe` extension) and Linux/macOS without changes.

### Spec

- **`spec/grammar.bnf`** — updated to reflect all additions: `kernel_decl` gains `type_params?`; new `const_param` rule; `kernel_field_decl` gains `("=" expr)?`; `[T, const_expr]` fixed-size array form and `const_expr` rule added; `--target wgpu` documented in build flags; `mut` scalar semantics clarified in `let_stmt`.

- **`linguist/samples/gpu.br`** — updated with a const-generic example (`kernel Blur<int W, int H>:`) showing field defaults and `W * H` array sizes.

---

## [0.9.2] — 2026-07-10 *(interpreter: 425/425 · transpiler: 216/216 · cuda: 34/34 · metal: 69/69 tests passing)*

### Added

- **`[..n]` ArrayAlloc syntax** — allocates an array of `n` elements without initialization. Distinct from `[v for ..n]` (fill) and `[v, v, ...]` (literal). Used in `kernel init` for `'sync` dynamic-size fields and for `'unified`/`'global` device buffers. Transpiles to `vec![Default::default(); n]` in Rust and `Vec<T>::with_capacity(n)` in simulation.

- **`priority =` dispatch parameter** — sets the stream scheduling priority for a CUDA kernel launch. Accepted values: `"high"`, `"normal"` (default), `"low"`. Maps to `cuStreamCreateWithPriority` with priorities `-1`, `0`, `1` respectively. Silently ignored on Metal and `boring run`.

### Changed

- **`smem =` dispatch parameter removed** — shared-memory byte counts are now computed automatically by the transpiler from the `'sync` field types and the `block` dimension. No user-visible `smem` argument is needed or accepted.

- **Dynamic `[T]'sync` fields** — declare the field without a size and assign `[..block_size]` in `init()`. The transpiler forwards `block_dim.x * sizeof(T)` as `shared_mem_bytes` automatically.

- **`.cargo/config.toml` — `RUST_MIN_STACK=4194304`** — the stack growth from the `ArrayAlloc` variant required increasing the minimum test-thread stack from 2 MB to 4 MB. This is set via an environment variable in `.cargo/config.toml` and does not affect final binaries.

- **`sync` keyword in module paths** — `use std.sync.atomic.AtomicUsize` now parses correctly. The parser accepts any reserved keyword as a path segment in `use` declarations.

- **`examples/plasma_metal.br`** — `var Dimension dim` corrected to `let Dimension dim` (the field is never reassigned after `init`).

- **Docs updated** — `gpu-module.md` and `cuda-module.md` document `[..n]`, `priority`, dynamic `'sync` fields, and the removed `smem` parameter. HTML files regenerated.

---

## [0.9.1] — 2026-07-07 *(interpreter: 78/78 · transpiler: 245/245 tests passing)*

### Changed

- **`'shared` → `'sync` GPU qualifier** — the block-SRAM qualifier is renamed from `'shared` to `'sync` inside `kernel` structs. The old name collided with the host `'shared` qualifier (Rc/Arc); `'sync` makes the synchronisation contract explicit.

- **Auto-barrier insertion for `'sync` fields** — the transpiler now inserts `__syncthreads()` / `threadgroup_barrier(...)` automatically. No explicit `sync` statement needed in the common case:
  - A barrier is emitted before the first loop in the `def ()` body (write-phase → loop-phase boundary).
  - A barrier is emitted at the top of each loop iteration that accesses a `'sync` field (covers stride-reduction and similar patterns).
  - **Manual mode** — if at least one explicit `sync` statement appears in the `def`, auto-insertion is disabled for the entire `def`; the developer owns all barrier placement.

- **`struct 'sync`** — a user-defined struct can be qualified with `'sync` to group compound state that must be observed coherently across threads. The barrier covers all fields of the struct. Use this instead of per-field `'actor'global` atomics when two or more values must be read together.

- **`sync` and `kernel` added to syntax highlighter** — `docs/build.py` now colours both as keywords in generated HTML documentation.

- **Docs updated** — `gpu-module.md`, `metal-backend.md`, and all generated HTML files reflect the new qualifier name, auto-barrier rules, and the struct `'sync` pattern.

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
