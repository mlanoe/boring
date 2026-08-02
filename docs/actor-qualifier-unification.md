# Unifying `'actor` across GPU memory qualifiers

> **Status: Implemented.** The rename (design points 1–2) and `'actor'unified`
> (design point 3) have both landed, across the parser/AST, all four backends
> (CUDA, ROCm, Metal, wgpu), the interpreter, `whisper-boring/src/math_gpu.br`,
> and this file's own cross-referenced docs (`gpu-module.md`, `cuda-module.md`,
> `rocm-backend.md`, `metal-backend.md`, `wgpu-backend.md`,
> `scoped-access-blocks.md`). See "Open Questions" below — both were resolved
> during implementation, not left open. Original problem-statement analysis
> retained below for context.

## Problem Statement

Boring's GPU memory qualifiers currently name the *protection mechanism*
inconsistently:

| Qualifier | Memory | GPU-side protection | Name says |
|---|---|---|---|
| `'sync` | block-shared (`__shared__`/`threadgroup`/`var<workgroup>`) | auto-inserted barrier before the first loop + at the top of every loop iteration touching a `'sync` field | *when* it's safe (in sync with other threads) — not *how* |
| `'actor'global` | device-only DRAM | atomics (`atomicAdd` etc.) | explicit |
| bare `'actor` | — | alias for `'actor'global` | reuses the name for one specific case only |
| `'unified` | host+device DRAM | **none** — concurrent writers race exactly like plain `'global` | n/a |

Outside kernel structs, `'actor` already means one specific thing,
consistently: *"the compiler automatically wraps every access with the
protection this data needs"* — `Rc<RefCell<T>>`/`Arc<Mutex<T>>`, chosen once
at the qualifier, vs. `'guard`'s explicit manual lock. `'sync`'s auto-barrier
is the exact same idea (automatic, compiler-inserted protection) applied to
GPU block-shared memory — it just has a different name today for
historical reasons (organically grew from the CUDA `__shared__` mapping
before the CPU-side `'actor`/`'guard` distinction existed).

Separately: `'actor'global` and `'unified` today differ in **two**
independent ways — protection (atomics vs. none) and host-visibility
(device-only vs. host+device). There's no way to get atomic protection on
host-visible memory (`'actor'unified` doesn't exist), which is a real gap,
not a design choice — nothing in any of the four backends (CUDA, ROCm,
Metal, wgpu) requires atomics and host-visibility to be mutually exclusive.

## Proposed Design

1. **`'sync` is renamed to bare `'actor`.** Same memory space, same
   auto-barrier semantics, same restriction to fixed-size arrays and
   kernel-struct context — pure rename, no behavior change. This is the
   *kernel-context* meaning of `'actor`, exactly as `'unified`/`'global`/
   `'const`/`'local`/`'sync` today only exist inside a `kernel struct` and
   already replace the ordinary (non-kernel) meaning of whatever token they
   use (`gpu-module.md`: "GPU memory qualifiers ... replace the usual
   ownership qualifiers inside a `kernel` struct") — reusing `'actor` this
   way is the same kind of context-dependent meaning the language already
   has, not a new category of ambiguity.

2. **The bare-`'actor`-as-alias-for-`'actor'global` shortcut is dropped.**
   Forced by (1): bare `'actor` inside a kernel struct now means
   block-shared memory. Atomics on device-only DRAM must be spelled out in
   full: `'actor'global`. (This is a real ergonomics regression for the
   single most common atomics pattern — histogram/counter kernels — that
   the rename accepts as its cost.)

3. **`'actor'unified` is added** — atomics on host+device DRAM. New
   capability, not currently implemented in any backend.

4. **Consequence, not a separate change**: once atomics work uniformly
   across `'global` and `'unified`, and non-atomic `'global`/`'unified`
   already share an identical protection story (`with` block for host
   access, `after =` for cross-dispatch ordering — see `scoped-access-blocks.md`'s
   Typing Rules table), the *only* remaining difference between `'global`
   and `'unified` is whether the backend can elide the host↔device copy.
   That's a real, verified-against-current-docs claim (`scoped-access-blocks.md`:
   "`'unified` — ... close to free ... `'global` — ... a real transfer,
   unconditionally") — not a new simplification being introduced here, just
   one made airtight once (3) removes the last independent axis of
   difference between the two.

### Final qualifier table (kernel-context)

| Qualifier | Memory | GPU-side protection |
|---|---|---|
| `'const` | constant cache | none needed — read-only during kernel execution |
| `'local` | registers | none needed — thread-private |
| `'actor` (bare) | block-shared | auto-barrier (was `'sync`) |
| `'actor'global` | device-only DRAM | atomics |
| `'actor'unified` | host+device DRAM | atomics *(new)* |
| `'global` | device-only DRAM | none (plain, no protection) |
| `'unified` | host+device DRAM | none (plain, no protection) |
| `'surface` | host+device DRAM, 32-bit pixels | none (special-cased for `Screen`) |

## Scope of impact

This is a rename + new capability across the whole pipeline, not a
docs-only change:

- **Lexer/parser** (`src/parser/parse_type.rs`): `TokenKind::Sync` and its
  dedicated `'sync` grammar production go away; `"actor"`'s branch
  (currently `line 349-358`, only accepting `'task`/`'global` after
  `'actor'`) gains a bare-in-kernel-context form and a `'unified` arm, and
  loses the "bare `'actor'` = `GpuActorGlobal`" shortcut *outside* that
  context change. Needs care: `'actor'task` (CPU-side, `Arc<Mutex>` for
  async) must keep working unchanged — only the *kernel-context* meaning of
  bare `'actor` is new.
- **AST** (`src/ast/mod.rs`): `GpuQual`/`OwnerQual` enums — rename
  `Sync`/`GpuSync` variants (or keep the variant name and just change what
  parses into it — either is fine, but pick one and update every match arm
  across the codebase consistently), add `GpuActorUnified` (or equivalent)
  for the new combination.
- **Checker** (`src/checker/*`, `src/validator/kernel.rs`): field-type
  validation matrix (`gpu-module.md`'s "Valid qualifier × field-type
  matrix") needs a row for the renamed qualifier and the new combination.
- **Device-side codegen** (per backend, `device.rs`): the four
  `is_actor_global_field`-style atomic-detection functions (`cuda/device.rs`,
  `rocm/device.rs`, `metal/device.rs`, `wgpu/device.rs`'s equivalent) need
  a parallel path for `'actor'unified`, and the auto-barrier insertion logic
  needs its qualifier check updated from `GpuQual::Sync` to whatever the
  renamed variant ends up being.
- **Host-side codegen** (per backend, `host.rs`): buffer-allocation flags
  differ by backend for `'actor'unified` specifically — this is where the
  real new work is (see "Per-backend feasibility" below).
- **Docs**: `gpu-module.md` (qualifier tables, Atomics section),
  `cuda-module.md`, `rocm-backend.md`, `metal-backend.md`, `wgpu-backend.md`
  (qualifier-mapping tables each maintains), `book.md` (language reference
  chapter), `scoped-access-blocks.md` (its own "`'sync`/`'local` ... out of
  scope" note needs updating now that `'sync` no longer exists under that
  name) — `.md` **and** `.html` mirrors for all of them, per this repo's
  established convention.
- **Tests**: every codegen test asserting literal `'sync`/`GpuQual::Sync`/
  bare-`'actor`-means-`'actor'global` text (`tests/cuda_codegen.rs`,
  `tests/rocm_codegen.rs`, `tests/metal_codegen.rs`, `tests/wgpu_codegen.rs`)
  needs updating, plus new regression tests for `'actor'unified` per
  backend.
- **Examples**: grep the repo's own `examples/*.br` for `'sync` before
  starting — any that use it need the rename applied too.

## Compatibility

No real Boring programs exist yet except **whisper-boring**, which the user
develops separately. Checked directly rather than assumed:
`whisper-boring/src/math_gpu.br` uses `'sync` **4 times** (two 16×16
tile-shared-memory pairs, one per kernel, in what looks like a tiled
matmul (`LinearKernel`) and a tiled attention kernel — `tile_x`/`tile_w` and
`tile_q`/`tile_k`) and **no** `'actor` usage at all today. This is the one
real migration site: the rename must be applied there too as part of this
task (not a follow-up), since it's the only real consumer of the qualifier
being renamed.

## Per-backend feasibility for `'actor'unified` (new work, not a rename)

- **CUDA**: low risk. `atomicAdd(&ptr, v)` etc. take a plain pointer of the
  base type — no special "atomic" type exists in CUDA C, so an atomic op on
  `cudaMallocManaged` memory is exactly as valid as on `cudaMalloc` memory.
  Should be close to a copy-paste of the existing `'actor'global` codegen
  path with the allocation call swapped.
- **ROCm**: same reasoning as CUDA (HIP's atomic intrinsics mirror CUDA's
  exactly, confirmed already for `'actor'global` — see `rocm-backend.md`).
- **Metal**: needs verification, not assumed. MSL requires atomic ops to
  target `atomic_int`/`atomic_uint`-typed pointers, but that's a *type*
  constraint on the kernel-parameter declaration, independent of the
  buffer's `MTLResourceOptions` (`StorageModeShared` for unified vs.
  whatever `'global` uses today) — should compose, but this needs an actual
  `cargo check`/build-level confirmation before relying on it, matching this
  project's established verify-before-documenting standard.
- **wgpu — real open risk, not just unverified.** WGSL requires
  `atomic<i32>`/`atomic<u32>` as the *storage type itself* — this is
  already how `'actor'global` works today (`wgpu-backend.md`: `'actor'global`
  fields are `array<atomic<i32>>` in `storage`). The question for
  `'actor'unified` is whether a WGSL `atomic<T>` buffer can also carry
  `MAP_READ`/`MAP_WRITE` usage flags for direct host mapping. **This needs
  checking against the real WebGPU/wgpu validation rules before any codegen
  is written** — worth noting that `wgpu-backend.md`'s current documented
  `'unified` flags (`STORAGE | MAP_READ | MAP_WRITE | COPY_SRC | COPY_DST`)
  are themselves worth re-verifying against real wgpu independently of this
  task (WebGPU's buffer-usage validation rules restrict what `MAP_READ`/
  `MAP_WRITE` may combine with) — if that combination turns out to already
  be invalid, `'actor'unified` inherits the same problem and the fix likely
  has to happen in `'unified` first.

## Actual implementation order

The suggested order below split the rename (step 2) from dropping the old
alias (step 4), as if they were independently landable. They aren't: design
points 1 and 2 are the same change (point 2 is explicitly "forced by" point
1) — the moment bare `'actor` means block-shared in kernel-struct context,
the old "bare `'actor'` = `'actor'global`" alias *must* go in the same
change, or bare `'actor` would ambiguously mean two things at once at the
exact same parse site. Steps 2–4 landed as one atomic change instead.
Steps 5 (CUDA/ROCm/Metal/wgpu) and 6–7 (docs, tests) proceeded roughly as
suggested, with Metal and wgpu both turning out lower-risk than the
feasibility analysis worried:

- **Metal**: the existing `'actor'global` codegen already casts the buffer
  pointer to `atomic_long*` *at the atomic-op call site*, rather than
  declaring the buffer parameter itself as an atomic-typed pointer — so the
  MSL type-system question the feasibility analysis raised never actually
  applies to this codegen's shape. `'actor'unified` reuses the identical
  cast, with `MTLStorageMode.shared` (which every buffer on this backend
  already uses uniformly, including plain `'global`). Verified with a real
  `cargo build` of a generated `--target metal` project.
- **wgpu**: see "Open Questions" above — the real implementation never
  combines `atomic<T>` storage with `MAP_READ`/`MAP_WRITE` for `'unified`
  today, so `'actor'unified` inherits nothing to be gated on.

One real implementation-time bug, not anticipated by this spec: the
`'unified`-only accessor-generation gates in each backend's `host.rs` (the
`read_<field>()`/`copy_<field>_to_host`/`copy_<field>_to_device` methods that
make a field host-readable) needed `GpuQual::ActorUnified` added explicitly —
missing it compiled `boring` itself cleanly (these are `matches!()` boolean
checks, not exhaustive `match`es, so the compiler can't catch an omission)
but produced *generated* Rust that failed to build, caught only by actually
building a generated wgpu project end-to-end.

## Open Questions — resolved during implementation

1. **Does wgpu allow `atomic<T>` storage combined with `MAP_READ`/`MAP_WRITE`
   usage?** Moot — the real wgpu host codegen (`src/transpiler/wgpu/host.rs`)
   never attempts that combination for `'unified` in the first place. Its
   `buffer_usages()` returns `STORAGE | COPY_SRC | COPY_DST` for `'unified`
   (no `MAP_READ`/`MAP_WRITE`); host reads/writes go through a *separate*
   staging buffer (`MAP_READ | COPY_DST` for D2H, `MAP_WRITE | COPY_SRC` for
   H2D — see `gpu.copy()`'s staging-buffer pattern), copied via
   `copy_buffer_to_buffer`. `'actor'unified` reuses this exact same path — its
   `atomic<i32>`/`atomic<u32>` storage buffer gets the identical
   `STORAGE | COPY_SRC | COPY_DST` usage, never combined with a map flag.
   Confirmed by a real `cargo build` of a generated `--target wgpu` project
   using an `'actor'unified` field (clean compile).
2. **Is the documented `'unified` wgpu buffer-usage-flag combination
   (`STORAGE | MAP_READ | MAP_WRITE | COPY_SRC | COPY_DST`) actually valid
   against real wgpu today?** No — that combination was never what the actual
   implementation does (see Q1); the documented flags in `wgpu-backend.md`
   were simply wrong, independent of this task, and have been corrected as
   part of this doc sync.
3. **Naming for the renamed `GpuQual`/`OwnerQual` enum variants** — renamed
   (`GpuQual::Sync` → `GpuQual::Actor`, plus the new `GpuQual::ActorUnified`
   and `OwnerQual::GpuActorUnified`), applied uniformly across every match
   site. `OwnerQual::GpuSync` was removed outright rather than kept as a dead
   variant, since the parser never produces it.

### One thing the original spec missed

`TokenKind::Sync` is not exclusively the `'sync` qualifier — it is also the
lexer token for the unrelated `sync` *statement* keyword (explicit
thread-group barrier inside a kernel `def` body, `parser/parse_stmt.rs`). The
rename only removes `'sync`'s *qualifier* grammar production
(`parser/parse_type.rs`); the `TokenKind::Sync` lexer token itself, and the
`sync:` statement it still spells, are untouched and unaffected by this task.
