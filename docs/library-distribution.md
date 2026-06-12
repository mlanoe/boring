# Library Distribution Model

> **Status: Draft** — This document explores the design space. No decision is final.

## Problem Statement

When a Boring library is distributed to third parties, two structural constraints must be satisfied simultaneously:

1. **Stack allocation** — to place a struct on the stack (qualifier `'stack`), the compiler must know its exact size at compile time. This requires the full struct layout, including private fields.
2. **Generics** — monomorphization requires access to the source of generic functions and types. There is no way to pre-compile all instantiations.

These two constraints both push toward distributing source code, or at minimum a rich interface representation that includes struct layouts and generic bodies.

## How Other Languages Handle This

### C / C++

Headers (`.h` / `.hpp`) separate interface from implementation. Template definitions live in headers too, since they must be visible at instantiation sites.

**Advantages**: explicit contract, implementation can be distributed as binary (`.a`, `.so`).

**Drawbacks**:
- Headers leak internal struct members even when they are private — the caller must see the layout to compute sizes.
- Template implementations are fully exposed.
- Header/source duplication is a maintenance burden.

### Java

Source and declarations are unified. The compiler emits `.class` bytecode, which serves as the distribution artifact.

**Advantages**: clean separation between contract and implementation; bytecode is the unit of distribution.

**Drawbacks**:
- Generics use type erasure — no monomorphization, performance is limited.
- Stack allocation of value types is limited (Project Valhalla is attempting to fix this).
- Everything is on the heap by default; the caller never needs to know struct sizes.

### Rust

Source is the distribution unit (via `crates.io` or direct path). The compiler performs monomorphization at the call site from source. Visibility is controlled with `pub` / `pub(crate)` — not by hiding source.

**Advantages**: solves both constraints naturally; no format duplication.

**Drawbacks**:
- Full source is visible to consumers (confidentiality / copyright concern).
- Compile times grow because dependency source is recompiled per target.

### Swift

Swift introduced `.swiftinterface` files: a stable, textual interface derived from source. Struct layouts are included (so stack allocation works). Private members are represented as opaque stored properties with their size and alignment preserved but names and types hidden.

**Advantages**: solves the confidentiality problem while satisfying stack-layout requirements; generics are handled via source inclusion in the module.

**Drawbacks**:
- Added toolchain complexity; interface files must be regenerated on every ABI-relevant change.
- Generics still require source access for full monomorphization.

## Constraints Specific to Boring

Boring transpiles to Rust. The natural distribution unit is therefore a **Rust crate**, produced by `boring build`. This means:

- The Rust output is distributable as-is, following Rust's own ecosystem conventions.
- A consumer who only has the compiled Rust crate gets monomorphization and stack allocation for free — those are Rust's concerns at that point.

The question of whether to also distribute the **Boring source** (`.br` files) is separate from whether the generated Rust crate works correctly.

## Current Position

**Follow Rust.** For now, Boring inherits Rust's distribution model:

- `boring build` produces a Rust crate (source + `Cargo.toml`).
- Distribution of the Rust crate follows standard Rust conventions.
- Visibility in Boring (`pub` / private) maps directly to Rust visibility and is enforced there.
- No Boring-specific interface format (`.bri`) is planned.

This is not a loss: since Boring transpiles to Rust, a consumer who receives the Rust crate already has all the information the Rust compiler needs. The Boring source layer is only relevant if the consumer wants to use Boring tooling directly (e.g. IDE support, Boring-level documentation).

## Header + Binary Distribution (Confidentiality Without Full Source)

For libraries that meet all three conditions — no generics, no `'stack` structs with inferred layout, and all qualifiers written explicitly (no inference) — it would be theoretically possible to ship a **header file** (public declarations only) paired with a **compiled binary**, without exposing any Boring source.

The header would use existing Boring syntax. Two candidates exist:

- `native` — already supported for FFI declarations; reusing it for "pre-compiled Boring" would be consistent syntactically but conflates two distinct concepts.
- `extern` — a clean alternative that signals "implementation provided externally, not transpiled here."

Either way, a consumer would link against the binary rather than recompile the library.

### The Rust ABI blocker

This approach is **currently blocked** by Rust itself. Rust does not guarantee a stable ABI between compiler versions. There is no way to link a pre-compiled Rust artifact against a different `rustc` version without risking breakage — symbol mangling, struct layout, calling conventions can all change between releases.

Concretely: a library author compiles with `rustc 1.X`, ships the binary; a consumer compiles their code with `rustc 1.Y` and links against it. There is no guarantee this works.

Until Rust stabilises an ABI (e.g. via the [Stable ABI initiative](https://github.com/rust-lang/rust/issues/111423) or a project like `abi_stable`), this distribution model is not viable for general use. It could work in a controlled environment where library author and consumer pin the exact same `rustc` version, but that is too fragile to be a supported Boring feature.

**Current conclusion**: header + binary distribution is a valid design direction but depends on an upstream Rust prerequisite. Revisit when Rust ABI stabilisation progresses.

## Possible Future Direction: `.bri` Interface Files

If confidentiality of the Boring source becomes a real requirement, a `.bri` (Boring Interface) format could be introduced, inspired by Swift's `.swiftinterface`:

- Public struct declarations with full field layout (required for `'stack`).
- Private fields replaced by opaque placeholders preserving size and alignment.
- Generic function and type signatures with their bodies (required for monomorphization).
- Private implementation details stripped.

This is a significant toolchain investment and is **not planned** until there is a concrete use case.

## Qualifier Inference and Cross-File Analysis

Including full Boring source (`.br` files) alongside the generated Rust crate unlocks a class of analyses that are impossible from the Rust layer alone.

### Qualifier propagation across library boundaries

The Boring transpiler infers qualifiers (`'stack`, `'heap`, `'shared`, `'actor`, …) from usage context. When a library function returns a value whose qualifier depends on how *its* dependencies are qualified, the transpiler needs to see the full call graph to propagate constraints correctly.

Without source, a consumer importing the library sees only Rust types (`Box<T>`, `Arc<T>`, …) — the qualifier intent is erased. With source, the transpiler can:

- Re-infer qualifiers end-to-end across the library boundary.
- Detect qualifier conflicts early (e.g. a `'stack` expectation colliding with a `'heap` return).
- Produce more precise diagnostics that name the originating `.br` file and line, not a generated Rust artifact.

### Cross-file analyses enabled by full source inclusion

| Analysis | Why it needs cross-file source |
|---|---|
| Qualifier flow | A qualifier constraint in file A may originate from a type defined in file B |
| Mutability audit | `mut` / `var` discipline across a module boundary is invisible from compiled Rust |
| Ownership cycle detection | `'shared` / `'actor` graphs span files; cycle detection requires the full graph |
| Dead-code elimination | Accurate reachability requires knowing which `.br` symbols are referenced across files |
| Documentation generation | Boring-level doc comments and inferred qualifiers are not preserved in generated Rust |

These analyses are additive — they do not block the current "follow Rust" position, but they provide a concrete reason to preserve `.br` sources when distributing libraries, even when confidentiality is not a concern.

## Summary

| Concern | Solution |
|---|---|
| Stack allocation (`'stack`) | Struct layout must be visible → follow Rust, expose layout in generated crate |
| Generics / monomorphization | Source required → follow Rust, ship generated Rust source |
| Qualifier inference across boundaries | Full `.br` source enables cross-file propagation and precise diagnostics |
| Cross-file analyses | Mutability audit, ownership cycles, dead-code — all require full source graph |
| Confidentiality of `.br` source | Not a current constraint → revisit if `.bri` becomes necessary |
| Copyright of Boring source | Orthogonal to the compiler; a licensing concern, not a language design concern |
