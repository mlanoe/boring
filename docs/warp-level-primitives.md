# `gpu.warp.*` — warp/wavefront/SIMD-group primitives

> **Status: Proposed.** No implementation yet — this document is the task
> spec for that work. Scope was settled in conversation before writing this
> up: **shuffle intrinsics are in scope**, not just the `warp.size`/
> `warp.lane`/`warp.sync` trio `cuda-module.md`'s "Known limitations"
> currently names — without shuffles, warp-level programming has no actual
> data-exchange primitive, which is the main reason it's useful.

## Problem Statement

A **warp** (CUDA), **wavefront** (AMD/HIP — 32 or 64 threads depending on
architecture, RDNA vs CDNA), **SIMD-group** (Metal), or **subgroup** (WGSL)
is a hardware-level grouping of threads below the thread-block level that
execute in lockstep. Boring today has no way to express warp-level
algorithms at all — `gpu.thread`/`gpu.block` (block-level indexing) and
bare `'actor` (block-shared memory + full-block barrier) are the only
synchronization primitives available. This means every intra-block
reduction pays for a full block-wide barrier (`__syncthreads()`/
`threadgroup_barrier`/`workgroupBarrier()`) even for the portion of the
reduction that's actually confined to a single warp, where threads are
*already* synchronous by construction and a much cheaper warp-local
barrier (or no barrier at all, via register-to-register shuffle) would do.

**Real motivating pattern, checked directly**: `whisper-boring/src/math_gpu.br`'s
tiled matmul kernels (`LinearKernel`, `AttentionKernel`) already reason
about this in a comment — "consecutive GPU threads (same warp/wavefront,
same reduction step `p`)" (line 54) — and use plain `sync` (full block
barrier) at every tile step (lines 158/161/244/247) because that's the only
tool available today. A warp-level shuffle-based reduction would let the
portion of the reduction confined to one warp skip the shared-memory
round-trip and block barrier entirely.

## Proposed Design

New built-ins, nested under the existing `gpu` namespace (consistent with
`gpu.thread`/`gpu.block`, not a sibling top-level `warp` namespace as the
current "Known limitations" wording might suggest):

| Built-in | Meaning |
|---|---|
| `gpu.warp.size` | threads per warp (32 on NVIDIA; 32 or 64 on AMD; SIMD-group/subgroup size elsewhere) |
| `gpu.warp.lane` | this thread's index within its warp, `0..gpu.warp.size` |
| `gpu.warp.sync()` | warp-local barrier — cheaper than `sync` (block-wide) |
| `gpu.warp.shuffle_down(v, delta)` | read `v` from the lane `delta` above this one |
| `gpu.warp.shuffle_up(v, delta)` | read `v` from the lane `delta` below this one |
| `gpu.warp.shuffle_xor(v, mask)` | read `v` from lane `this_lane XOR mask` (butterfly pattern — reductions) |
| `gpu.warp.shuffle(v, src_lane)` | broadcast/read `v` from an arbitrary lane |

### Per-backend mapping

| Boring | CUDA | HIP (ROCm) | MSL (Metal) | WGSL (wgpu) |
|---|---|---|---|---|
| `gpu.warp.size` | `warpSize` (built-in var) | `warpSize` (built-in var, 32 or 64) | `[[threads_per_simdgroup]]` param attribute | `subgroup_size` builtin |
| `gpu.warp.lane` | `threadIdx.x % warpSize` (linearized for 2D/3D — see below) | same as CUDA | `[[thread_index_in_simdgroup]]` param attribute | `subgroup_invocation_id` builtin |
| `gpu.warp.sync()` | `__syncwarp(mask)` | `__syncwarp` equivalent (HIP mirrors CUDA's name) | `simdgroup_barrier(mem_flags::mem_none)` | `subgroupBarrier()` |
| `gpu.warp.shuffle_down/up/xor` | `__shfl_down_sync`/`__shfl_up_sync`/`__shfl_xor_sync` | same names (HIP mirrors CUDA) | `simd_shuffle_down`/`simd_shuffle_up`/`simd_shuffle_xor` | `subgroupShuffleDown`/`subgroupShuffleUp`/`subgroupShuffleXor` |
| `gpu.warp.shuffle(v, lane)` | `__shfl_sync` | same | `simd_shuffle` | `subgroupShuffle` |

**Mask handling (CUDA/ROCm), decided here rather than left open**: CUDA's
`_sync` shuffle/barrier intrinsics (post-Volta) require an explicit
32-bit active-lane mask, needed for correctness under thread divergence.
Boring hides this: every `gpu.warp.*` call emits the full mask
(`0xffffffff`) rather than exposing a mask parameter in Boring source —
consistent with the language's level of abstraction elsewhere (Boring
doesn't expose raw CUDA streams either, for example). **Caveat to document
alongside the feature, not solve here**: this means `gpu.warp.*` inside a
divergent branch (an `if` that not all lanes in the warp take) is only as
safe as passing `0xffffffff` to CUDA's `_sync` intrinsics actually is —
correct for reconverged/uniform control flow, undefined behavior otherwise.
This is an inherent constraint of warp-level programming, not something
Boring can fully paper over.

**Lane linearization for CUDA/ROCm 2D/3D blocks**: `warpSize` divides a
*linear* thread index, but Boring kernel blocks can be 2D/3D
(`gpu.thread.x/y/z`). `gpu.warp.lane` linearizes first:
`tid = thread.x + thread.y * block_dim.x + thread.z * block_dim.x * block_dim.y`,
then `lane = tid % warpSize` — the same linearization CUDA programmers do
by hand today, just generated once instead of at every call site.

### wgpu fallback (decided in conversation, not left as an open question)

Subgroup support in wgpu/naga is real but **not universally available** —
verified via a live check (not assumed): subgroup operations landed in
wgpu (tracked in [gfx-rs/wgpu#5555](https://github.com/gfx-rs/wgpu/issues/5555)),
naga implements essentially the full builtin set except `subgroupElect()`
([gfx-rs/wgpu#7396](https://github.com/gfx-rs/wgpu/issues/7396)), and the
WebGPU spec itself only reached Candidate Recommendation for subgroups in
January 2025 — meaning it's recent, and gated behind a `wgpu::Features::SUBGROUP`
adapter feature that must be queried and is not guaranteed present on
every backend/GPU combination (DX12 vs Vulkan vs Metal-via-wgpu).

Decision: **when `Features::SUBGROUP` is unavailable at runtime, `gpu.warp.*`
falls back to an emulation over bare-`'actor`-equivalent (workgroup) memory** —
`shuffle_down(v, delta)` becomes "write `v` to a workgroup-shared array
indexed by `local_invocation_id`, `workgroupBarrier()`, read back index
`lane + delta`, `workgroupBarrier()`" — rather than a hard runtime error.
This is real additional implementation work (a second codegen path per
`gpu.warp.*` built-in, active only on wgpu, selected by a runtime feature
check at pipeline-creation time) and a real semantic caveat to document:
the fallback emulates the *API*, not the *hardware lockstep guarantee* —
it costs a real shared-memory round trip and full barrier per shuffle,
not the register-to-register, near-free operation a true subgroup shuffle
is. Callers get correct results either way; only the performance
justification for using `gpu.warp.*` over bare `'actor` disappears on hardware
that lacks subgroup support.

**Open sub-question**: what "warp size" does the wgpu fallback use for
`gpu.warp.size`/lane arithmetic, given there's no real subgroup to query?
Likely a fixed, documented constant (e.g. 32, matching the common case) —
needs to be picked and clearly called out as *not* a real hardware value on
this fallback path specifically.

## Scope of impact

- **Lexer/parser**: `gpu.warp` as a new nested namespace under the
  existing `gpu` built-in — parsed the same way `gpu.thread`/`gpu.block`
  already are, not a new grammar production.
- **Device-side codegen** (per backend `device.rs`): new kernel-parameter
  attributes for Metal (`threads_per_simdgroup`, `thread_index_in_simdgroup`
  — same mechanism as existing `thread_position_in_threadgroup` etc.), new
  builtin params for WGSL (`subgroup_size`, `subgroup_invocation_id`, plus
  an `enable subgroups;` directive when any kernel uses `gpu.warp.*`), and
  straightforward intrinsic-call emission for CUDA/ROCm (already-existing
  builtin variable, no new parameter plumbing needed there).
- **wgpu-specific**: runtime feature detection (`adapter.features().contains(wgpu::Features::SUBGROUP)`)
  at pipeline-creation time, branching codegen or runtime dispatch between
  the real-subgroup path and the shared-memory emulation path described
  above.
- **Docs**: `gpu-module.md` (new built-ins section), `cuda-module.md`/
  `rocm-backend.md`/`metal-backend.md`/`wgpu-backend.md` (each maintains a
  built-in mapping table — add `gpu.warp.*` rows), `book.md`. Remove the
  "Known limitations" bullet in `cuda-module.md` once shipped. `.md` + `.html`
  mirrors throughout.
- **Tests**: per-backend codegen tests for each `gpu.warp.*` built-in,
  including a wgpu-specific test asserting the emulation path's exact
  codegen shape when the feature is unavailable (can't easily test the
  real-subgroup path without real hardware exposing that feature — same
  caveat this project already documents for CUDA/Metal/ROCm hardware
  verification elsewhere).

## Compatibility

No real Boring programs exist yet except **whisper-boring**. Its tiled
matmul/attention kernels (`math_gpu.br`, quoted above) are the motivating
example, not a migration requirement — they keep working unchanged with
plain `sync`; adopting `gpu.warp.*` for their reduction step is a natural,
separate follow-up once this lands, expected to reduce full-block barriers
in the tiled-reduction inner loop.

## Open Questions

1. **Fallback warp size constant on wgpu** (see above) — needs a specific
   documented value, not left implicit.
2. **Divergent-branch safety** — document, not solve: `gpu.warp.*` inside
   non-uniform control flow is only as safe as the underlying `_sync`
   intrinsics with a full mask; Boring doesn't detect or warn about this
   today.
3. **Should the checker warn (not error) when `gpu.warp.*` is used inside
   an `if`/`while` that isn't provably warp-uniform?** A static analysis
   nicety, not required for a first version — flagged here so it isn't
   forgotten, not because it blocks shipping without it.
