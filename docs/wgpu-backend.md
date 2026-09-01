# wgpu backend

`boring build --target wgpu` generates a Rust + WGSL project that runs GPU kernels via the [wgpu](https://wgpu.rs) crate — a cross-platform GPU abstraction that runs on DirectX 12 (Windows), Vulkan (Windows / Linux), and Metal (macOS).

The Boring source is identical: the same `kernel` structs, qualifiers, and `gpu.*` built-ins work unchanged across CUDA, Metal, and wgpu targets.

---

## Motivation

CUDA requires an NVIDIA GPU and the CUDA toolkit. Metal requires macOS. wgpu runs on any modern GPU on Windows, Linux, and macOS — no vendor lock-in, no external toolchain.

| Backend | OS | GPU |
|---|---|---|
| CUDA | Windows / Linux | NVIDIA only |
| Metal | macOS only | Apple / Intel Mac |
| **wgpu** | Windows / Linux / macOS | Any DirectX 12, Vulkan, or Metal GPU |

On Windows, wgpu uses DirectX 12 by default and falls back to Vulkan if DX12 is unavailable.

---

## Qualifier mapping

| Boring qualifier | CUDA C | MSL address space | **WGSL / wgpu** |
|---|---|---|---|
| `'unified` | `cudaMallocManaged` | `device` + `MTLStorageMode.shared` | `storage` buffer, `STORAGE \| COPY_SRC \| COPY_DST` — host-visible via a staging-buffer copy, **not** `MAP_READ`/`MAP_WRITE` on this buffer directly (see "Host access" below) |
| `'global` | `cudaMalloc` | `device` | `storage` buffer, GPU-only |
| `'surface` | `cudaMallocManaged` (u32) | `device uint*` | `storage` buffer of `u32`, host-visible — same layout as `'unified` |
| bare `'actor` | `__shared__` | `threadgroup` | `var<workgroup>` |
| `'local` | registers | thread-private | `var<function>` |
| `'const` scalar | `__constant__ T name;` | `constant T* [[buffer(N)]]` | `var<uniform>` in a dedicated uniform buffer |
| `'const` fixed array (`[T, N]`) | `__constant__ T name[N];` | `constant T* [[buffer(N)]]` | `var<uniform>` array in a dedicated uniform buffer |
| `'actor'global` | `atomicAdd` etc. | `atomic_fetch_add_explicit` | `atomic<i32>` / `atomic<u32>` fields in `storage` buffer, `STORAGE \| COPY_SRC \| COPY_DST` |
| `'actor'unified` | `atomicAdd` etc. | `atomic_fetch_add_explicit` | same `atomic<i32>`/`atomic<u32>` storage buffer as `'actor'global` — host-visible via the same staging-buffer copy path as `'unified` |

---

## Built-in mapping

| Boring | CUDA C | MSL | **WGSL** |
|---|---|---|---|
| `gpu.thread.x/y/z` | `threadIdx.x/y/z` | `thread_position_in_threadgroup.x/y/z` | `@builtin(local_invocation_id).x/y/z` |
| `gpu.block.x/y/z` | `blockIdx.x/y/z` | `threadgroup_position_in_grid.x/y/z` | `@builtin(workgroup_id).x/y/z` |
| `gpu.block_dim.x/y/z` | `blockDim.x/y/z` | `threads_per_threadgroup.x/y/z` | `@builtin(local_invocation_size).x/y/z` |
| `gpu.grid_dim.x/y/z` | `gridDim.x/y/z` | `threadgroups_per_grid.x/y/z` | `@builtin(num_workgroups).x/y/z` |
| `sync` (manual) | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | `workgroupBarrier()` |
| `gpu.warp.size` | `warpSize` | `threads_per_simdgroup` | `@builtin(subgroup_size)` (real) / fixed `32u` (emulated) |
| `gpu.warp.lane` | linearized `threadIdx`, `% warpSize` | `thread_index_in_simdgroup` | `@builtin(subgroup_invocation_id)` (real) / `local_invocation_index % 32u` (emulated) |
| `gpu.warp.sync()` | `__syncwarp(0xffffffff)` | `simdgroup_barrier(mem_flags::mem_none)` | `subgroupBarrier()` (real) / `workgroupBarrier()` (emulated) |
| `gpu.warp.shuffle_down(v, delta)` | `__shfl_down_sync(0xffffffff, v, delta)` | `simd_shuffle_down(v, delta)` | `subgroupShuffleDown(v, delta)` (real) / workgroup-scratch emulation (see below) |
| `gpu.warp.shuffle_up(v, delta)` | `__shfl_up_sync(0xffffffff, v, delta)` | `simd_shuffle_up(v, delta)` | `subgroupShuffleUp(v, delta)` (real) / emulated |
| `gpu.warp.shuffle_xor(v, mask)` | `__shfl_xor_sync(0xffffffff, v, mask)` | `simd_shuffle_xor(v, mask)` | `subgroupShuffleXor(v, mask)` (real) / emulated |
| `gpu.warp.shuffle(v, lane)` | `__shfl_sync(0xffffffff, v, lane)` | `simd_shuffle(v, lane)` | `subgroupShuffle(v, lane)` (real) / emulated |
| bare-`'actor` auto-barrier | inserted before first loop + at top of each loop iteration | idem | idem |
| atomics (`'actor'global`/`'actor'unified`) | `atomicAdd` etc. | `atomic_fetch_add_explicit` | `atomicAdd` / `atomicSub` / `atomicOr` / `atomicAnd` / `atomicXor` |
| `[i].min/max/swap(v)` | `atomicMin`/`atomicMax`/`atomicExch` | `atomic_fetch_min/max_explicit` / `atomic_exchange_explicit` | `atomicMin` / `atomicMax` / `atomicExchange` |
| `[i].cas(expected, new)` | `atomicCAS` | `atomic_compare_exchange_weak_explicit` (bridged — see below) | `atomicCompareExchangeWeak(...).old_value` |

`.min`/`.max`/`.swap`/`.cas` are methods on an indexed `'actor'global`/`'actor'unified` element (not compound-assign operators — there's no natural infix form for min/max/exchange/compare-and-swap), and unlike `+= -= &= |= ^=` they're handled in expression position: all four return the previous value, matching `atomicMin`/`atomicMax`/`atomicExch`/`atomicCAS`'s real CUDA/HIP semantics on every backend. WGSL's `atomicCompareExchangeWeak` returns a struct (`{old_value, exchanged}`, not a bare value) — `.old_value` field access on the call result gives exactly the previous value. Metal's `atomic_compare_exchange_weak_explicit` is a bigger shape mismatch (takes a pointer to the expected value, returns a `bool`) — see `metal-backend.md` for how that's bridged.

**Real bug found and fixed while verifying this against real WGSL** (not just `cargo check`, which only validates the Rust host side): the atomic-pointer helper shared by the compound-assign path and the new methods used to emit `&buf[i as u32]` — `as` is Rust cast syntax, invalid inside a WGSL index expression at all. A real `naga::front::wgsl::parse_str` on a plain `counts[bucket] += 1` failed with `expected ']', found 'as'`, meaning **every** atomic op emitted through this path — `+= -= &= |= ^=`, and now `.min`/`.max`/`.swap`/`.cas` — was unparseable WGSL until this fix, undetected because nothing in the test suite had run generated WGSL through a real WGSL parser before. Fixed to `u32(i)` (WGSL's real cast syntax — a function-call form, matching the plain array-index case elsewhere in this backend, which already got this right).

### `.min`/`.max`/`.swap`/`.cas` without `'actor` — two WGSL statements, not one

These four methods also work on a plain, non-`'actor'` field (matching `+= -= &= |= ^=`'s existing degrade-to-plain-arithmetic behavior), but WGSL has no statement-expression the way CUDA/HIP/Metal do (`({ ... })`) — "read old, mutate, yield old" can't be a single WGSL expression. The plain fallback only works when the call is the **entire right-hand side** of a `let`/assignment statement:

```wgsl
let old = scale_buf[u32(i)];   // capture the current value
scale_buf[u32(i)] = min(old, v);   // plain, non-atomic update
```

Anywhere else — nested inside a larger expression — this genuinely isn't representable in WGSL as one unit; the generated shader carries a visible `/* unsupported here: ... */` marker at that spot instead of silently falling back to the *unrelated* pre-existing scalar `.min`/`.max` builtin (a pure comparison that never touches the buffer at all) or emitting genuinely invalid WGSL for `.swap`/`.cas` (confirmed via a real naga parse: `no definition in scope for identifier: 'swap'`, before this fix existed).

Two more real bugs found and fixed while verifying the discard case (`_ = buf[i].min(v)`) against real naga, neither just "unverified" but genuinely wrong:

1. WGSL's `_` is a **write-only** phony discard target — it can never be read back. The plain min/max/cas codegen needs to read the captured old value, so assigning to `_` and then reading `_` failed with a real naga parse error: `no definition in scope for identifier: '_'`.
2. The first fix (a synthetic name) tried `__boring_discard_0` — WGSL reserves identifiers starting with `__`, confirmed via naga: `Identifier starts with a reserved prefix`, the same constraint already noted elsewhere in this document for `__params`.

Fixed by declaring a fresh `bp_discard_N` `let` (matching the existing shuffle-hoist temp-naming convention, single underscore) instead of touching `_` at all.

### `gpu.warp.*` — real subgroup support vs. shared-memory emulation

WGSL subgroup builtins (`subgroup_size`, `subgroup_invocation_id`,
`subgroupShuffle*`, `subgroupBarrier`) need an explicit `enable subgroups;`
module directive and are only valid when the adapter has
`wgpu::Features::SUBGROUP` — support is real in wgpu/naga but **not
universally available** (gated behind an adapter feature, not guaranteed
present on every backend/GPU combination).

Whenever a program uses `gpu.warp.*`, `boring build --target wgpu` emits
**two** WGSL modules:

- `shaders/main.wgsl` — the real-subgroup mapping (table above), gated by
  `enable subgroups;`.
- `shaders/main_emulated.wgsl` — a shared-memory emulation: `shuffle_down(v,
  delta)` becomes "write `v` to a workgroup-shared scratch array indexed by
  `local_invocation_index`, `workgroupBarrier()`, read back the target lane's
  slot (clamped to the caller's own value at the simulated warp's boundary —
  same convention as the real `_sync` shuffle intrinsics), `workgroupBarrier()`
  again". The simulated warp size is a fixed, documented `32` — there's no
  real subgroup to query a size from on this path.

The generated host code queries `adapter.features().contains(wgpu::Features::SUBGROUP)`
before `request_device` and requests the feature only if actually supported.
That check alone isn't sufficient, though — confirmed against a real `wgpu
22` install: the HAL/backend layer can report `Features::SUBGROUP` as present
(true for Metal-backed adapters, whose native SIMD-group support predates
`naga`'s WGSL frontend catching up to the syntax) while `naga` still can't
parse the `enable subgroups;` directive at all, which would otherwise be an
uncaught-validation-error panic at shader-module creation. So shader-module
creation additionally wraps the real-module attempt in a
`push_error_scope(wgpu::ErrorFilter::Validation)` / `pop_error_scope()` pair
and falls back to the emulated module if that scope reports an error —
catching exactly this HAL-vs-shader-frontend gap at runtime instead of
crashing. Both paths give correct results — the emulated path just costs a
real shared-memory round trip and full barrier per shuffle instead of a
near-free register-to-register exchange, so the performance case for
`gpu.warp.*` over bare `'actor` disappears (but correctness doesn't) when the
real path isn't usable.

The emulated path only supports `gpu.warp.shuffle_*` inside the value of a
`let` statement or the right-hand side of an assignment (`let x =
gpu.warp.shuffle_down(v, n)`, or the more common reduction idiom `v = v +
gpu.warp.shuffle_xor(v, mask)`) — reachable through arithmetic/index/cast/call
nesting. Each shuffle call found is hoisted into its own preceding
write/barrier/read/barrier sequence assigning a fresh temp variable, which is
then substituted back into the original expression — WGSL has no
side-effecting expressions, so a shuffle can't stay inline the way a real
`subgroupShuffle*` call can. A shuffle call outside a `let`/assignment
(nested in a condition, a function argument to something other than a
recognized numeric builtin, etc.) emits a visible `/* unsupported ... */`
marker rather than silently wrong WGSL.

See [warp-level primitives](warp-level-primitives.html) for the full design.

### WGSL shader signature

WGSL uses `@group` / `@binding` indices for all buffers, and built-in attributes for thread/block indices. The transpiler assigns binding indices automatically in declaration order.

```wgsl
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id)   tid:     vec3<u32>,
    @builtin(workgroup_id)          bid:     vec3<u32>,
    @builtin(local_invocation_size) bdim:    vec3<u32>,
) {
    let i = tid.x + bid.x * bdim.x;
    buf[i] = buf[i] * 2.0;
}
```

---

## Workgroup size and pipeline overrides

In CUDA and Metal, the block size is passed at dispatch time. WGSL encodes the workgroup size as a static shader annotation (`@workgroup_size`). The transpiler bridges this gap using **pipeline overrides** — a WGSL 1.0 feature that fixes a constant at pipeline-creation time without recompiling the shader source:

```wgsl
override block_x: u32 = 256;
override block_y: u32 = 1;
override block_z: u32 = 1;

@compute @workgroup_size(block_x, block_y, block_z)
fn main(...) { ... }
```

The Rust host code sets the override values when creating the compute pipeline, matching the `block =` argument from the Boring `kernel:` block:

```rust
// boring: k(block = 256)
device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    // ...
    compilation_options: wgpu::PipelineCompilationOptions {
        constants: &[("block_x", 256.0), ("block_y", 1.0), ("block_z", 1.0)]
            .into_iter().collect(),
        ..Default::default()
    },
    ..Default::default()
});
```

This means a single WGSL shader covers all block sizes — no shader duplication.

---

## Grid inference

Unlike CUDA/ROCm/Metal (where the kernel struct's own `__boring_launch`/`dispatch`
method computes `grid_dim`), wgpu's dispatch call-site codegen lives in the
shared `src/transpiler/emit_kernel.rs` (`try_emit_kernel_dispatch`), since
`.dispatch(gx, gy, gz)` is a plain three-`u32` method with no computation of
its own. When a `kernel:` block omits `grid =`:

- A fixed-shape `[T, width=W, height=H]` labeled-array field (`'unified`/`'global`/
  `'actor'global`/`'actor'unified`) defaults `(gx, gy, gz)` from its compile-time
  axis sizes and the dispatch site's own `block =` argument — `gx = ceil(W/bx)`,
  `gy = ceil(H/by)`, `gz = 1`, generalized to 3 axes.
- Otherwise, `(gx, gy, gz)` defaults to `(1, 1, 1)` — this includes plain `[T]'unified`/
  `'global` array fields, which get no 1D auto-grid inference here (a pre-existing gap,
  not something labeled arrays introduce) — always pass `grid =` explicitly for those.

---

## WGSL compilation

WGSL is compiled by wgpu at runtime using the `naga` compiler (bundled in the wgpu crate). No external toolchain is required. The generated project has no `build.rs`.

An optional AOT path (`boring build --target wgpu --aot`) may be added in a future iteration to emit pre-compiled SPIR-V or DirectX bytecode for workloads where startup latency matters.

---

## Scalar uniform fields

All scalar kernel fields (`let float alpha`, `var float t`, `var Dimension dim`, inferred-`'const` scalars) are packed into a single generated WGSL struct bound as a `uniform` buffer:

```wgsl
struct KernelParams {
    t:     f32,
    dim_w: u32,
    dim_h: u32,
    alpha: f32,
}
@group(0) @binding(N) var<uniform> params: KernelParams;
```

The binding index `N` follows all `'unified`, `'global`, `'actor'global`, and `'actor'unified` array bindings. The Rust host writes the struct before every dispatch via `queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params))`.

Push constants are not used — they require the `PUSH_CONSTANTS` feature flag (off by default in wgpu) and are limited to 128 bytes on DX12/Vulkan, which conflicts with the goal of zero-configuration portability.

`Dimension` fields are flattened into two `u32` fields (`dim_w`, `dim_h`) in the params struct — WGSL structs in uniform buffers require explicit padding rules, and a two-field flat layout avoids any alignment ambiguity.

---

## `gpu.copy()` — staging buffer pattern

`'global` buffer fields are GPU-only (`STORAGE | COPY_DST`). Host reads and writes go through staging buffers. The transpiler emits two helpers in `src/main.rs`:

```rust
fn __boring_gpu_copy_d2h(device, queue, src: &wgpu::Buffer, dst: &mut [u8]);
fn __boring_gpu_copy_h2d(device, queue, src: &[u8], dst: &wgpu::Buffer);
```

**D2H** (`gpu.copy(k.result, host_buf)`):
1. Allocate staging buffer `MAP_READ | COPY_DST`, same size as `src`.
2. `encoder.copy_buffer_to_buffer(src → staging)`, submit.
3. `device.poll(Wait)`.
4. `slice.map_async(Read)` → memcpy to `dst` → `staging.unmap()`.

**H2D** (`gpu.copy(host_buf, k.input)`):
1. Allocate staging buffer `MAP_WRITE | COPY_SRC`, same size as `dst`.
2. `slice.map_async(Write)` → memcpy from `src` → `staging.unmap()`.
3. `encoder.copy_buffer_to_buffer(staging → dst)`, submit.

Staging buffers are created on demand per call and not pooled (initial implementation). A reusable pool over a pre-allocated staging arena can be added later without changing the Boring source.

---

## `'unified` vs `'global` on wgpu

wgpu buffers are not implicitly host-visible. The transpiler uses different wgpu `BufferUsages` flags depending on the qualifier:

| Qualifier | `BufferUsages` | Host read/write |
|---|---|---|
| `'unified` | `STORAGE \| COPY_SRC \| COPY_DST` | `copy_{field}_to_host`/`copy_{field}_to_device`, via a staging buffer (see "`gpu.copy()`" above) |
| `'global` | `STORAGE \| COPY_DST` | via `gpu.copy()` (staging buffer) |
| `'actor'global` | `STORAGE \| COPY_SRC \| COPY_DST` | via `gpu.copy()` (staging buffer) — device-only |
| `'actor'unified` | `STORAGE \| COPY_SRC \| COPY_DST` | `copy_{field}_to_host`/`copy_{field}_to_device`, same staging-buffer path as `'unified` |

`'unified`'s own storage buffer never carries `MAP_READ`/`MAP_WRITE` — host access
always goes through a separate staging buffer (`MAP_READ | COPY_DST` for reads,
`MAP_WRITE | COPY_SRC` for writes — see "`gpu.copy()`" above), copied to/from the
real storage buffer with `copy_buffer_to_buffer`. This is deliberate, not an
implementation gap: WebGPU's buffer-usage validation restricts `MAP_READ` to pairing
only with `COPY_DST` and `MAP_WRITE` only with `COPY_SRC`, so a single buffer
carrying `STORAGE | MAP_READ | MAP_WRITE` together (an earlier draft of this table
claimed exactly that combination) would not validate — the two-buffer staging design
avoids the question entirely rather than relying on an unverified combination.
`'actor'unified` reuses the identical staging-buffer path, which is also why it
doesn't inherit any WGSL `atomic<T>` + direct-mapping risk: the `atomic<i32>`/
`atomic<u32>` storage buffer is never itself asked to carry `MAP_READ`/`MAP_WRITE`.

---

## `GPU` type on wgpu

**Real per-adapter introspection (2026-09-01), dispatch still single-device.**
At startup, `instance.enumerate_adapters(wgpu::Backends::all())` builds the
real list of adapters the system exposes (`__BORING_GPU_ADAPTERS`,
`src/transpiler/wgpu/host.rs`'s `emit_gpu_adapter_enumeration`) — index 0 is
always the adapter `device`/`queue` were actually created from (the one
`request_adapter`'s own heuristic picked), so `GPU(0)` reliably means "the
adapter your kernels actually run on"; any other index is a real, distinct
physical adapter if the system has one, purely for introspection. `GPU(n)`
and `GPU.all()`'s elements resolve to genuinely different adapters when more
than one is present — this replaces the previous single-simulated-device
stand-in (every index resolving to the same adapter, `docs/gpu-compute-display-split.md`'s
"category A" work).

Deduplication (a physical GPU can be enumerated once per backend it's
reachable through, e.g. Vulkan and GL on the same Linux box) compares
`AdapterInfo` (`.get_info()`, which derives `PartialEq`/`Eq`) rather than the
`wgpu::Adapter` handle itself — **`wgpu::Adapter` implements neither `Clone`
nor `PartialEq`** on the pinned `wgpu = "22"` (confirmed against the real
`wgpu-22.1.0`/`wgpu-types-22.0.0` source, not docs.rs, which can describe a
newer release's API by default) — this is why the adapter is Arc-wrapped
once right after creation (same convention as `device`/`queue`) rather than
cloned.

**Still not implemented**: real per-device *dispatch* — `new(g) K` (placing
a kernel's actual GPU work on a specific non-default adapter, like CUDA's
`CudaContext::new(idx)`/Metal's per-index `MTLDevice`). Every kernel still
dispatches on the single global `device`/`queue` regardless of which
`GPU(n)` it was constructed with — see "Known limitations vs CUDA" below and
`docs/gpu-compute-display-split.md`'s "category A, part (b)".

| Boring method | wgpu API | Notes |
|---|---|---|
| `GPU(n)` | — | a plain `usize` index into the real enumerated adapter list |
| `GPU.all()` | — | one element per real adapter found, `0..count` |
| `.name()` | `adapter.get_info().name` | real, per-adapter name |
| `.totalMem()` | — | always `0` — `wgpu::AdapterInfo` has no memory-size field on any backend (checked against `wgpu-types` 22.0.0's struct definition) |
| `.freeMem()` | — | always `0`, same reason |
| `.computeCapability()` | — | always `[0, 0]` — a CUDA-only concept |
| `.warpSize()` | — | always `32` — a conservative default, not queryable via wgpu |
| `.maxThreads()` | `limits.max_compute_invocations_per_workgroup` | real, per-adapter limit |
| `.maxSharedMem()` | `limits.max_compute_workgroup_storage_size` | real, per-adapter limit, bytes per workgroup |
| `.index()` | — | echoes back whatever index was passed to `GPU(n)` — now a real index into the adapter list |

---

## `after =` ordering

wgpu uses a single command queue per device. Command buffers submitted to the same queue execute in order. The transpiler maps `after =` to submission ordering — kernels with `after =` dependencies are submitted in a separate `submit` call after the dependency's `submit` has been flushed.

This is not GPU-side pipelining (unlike CUDA streams), but it preserves the ordering semantics with no CPU round-trip within a `kernel:` block.

---

## Error handling

Each kernel struct's `dispatch()` opens **two** WebGPU error scopes before encoding — `push_error_scope(wgpu::ErrorFilter::OutOfMemory)`, then `push_error_scope(wgpu::ErrorFilter::Validation)` — and checks both (`pop_error_scope()`, bridged to sync via the same `pollster` already used for device/adapter setup) right after submit, popping in reverse order (Validation first, then OutOfMemory) — returning `Result<(), Box<dyn std::error::Error + Send + Sync>>` instead of `()`. A regular `kernel:` block's dispatch call propagates that error via `?` into `boring_main()`'s own `Result` (synthesized as `Result`-returning whenever any kernel is involved, even with no explicit `throws`); inside a `Screen` program's render loop (a plain closure, not a `Result`-returning context) it's `.expect(...)` instead.

Previously, no error-scope/callback machinery existed at all: `dispatch()` returned `()` unconditionally and a rejected launch configuration (e.g. an invalid workgroup count) went completely unobserved.

Validation is checked at command-buffer *encoding* time (synchronously, on the CPU, as `dispatch_workgroups` is called) — no `device.poll()` is needed to catch it. This does **not** cover execution-time faults (out-of-bounds access, device loss), which WebGPU instead reports via device-lost callbacks/uncaptured-error events, not scoped validation; those aren't wired up.

Pipeline creation (`create_compute_pipeline`, inside the per-kernel `PIPELINE.get_or_init` cache in `emit_kernel_new` — see `Workgroup size and pipeline overrides` above) is **not** wrapped in an error scope either, unlike shader-module creation and `dispatch()`: `OnceLock::get_or_init`'s closure can't itself return a `Result`, and `new()` returns bare `Self`. A validation error here (e.g. an oversized fixed-shape bare-`'actor` `Image`/`Volume` field exceeding `.maxSharedMem()`) would otherwise hit wgpu's default uncaptured-error handler, which panics. Fixed by installing a non-panicking `device.on_uncaptured_error(...)` handler once at device-creation time (`async_main`/`emit_screen_main`'s `main()`) — the error is reported instead of crashing the process, and a subsequent `dispatch()` call against the resulting (unusable) pipeline raises its own validation error, which the existing per-dispatch error scope above *does* catch and convert to `GpuError::LaunchError`.

### Typed `GpuError`

Each popped scope's error is classified into the built-in [`GpuError`](gpu-module.html#gpu-error-handling) enum and wrapped in `BoringError::Other` — the Validation scope maps to `GpuError::LaunchError`, the OutOfMemory scope to `GpuError::OutOfMemory` — instead of the single generic formatted-string error this used to return. Because wgpu's host codegen shares the same general transpiler pipeline `BoringError`/`catch`-by-variant already lives in (unlike CUDA/ROCm/Metal — see below), this is the one backend where `catch GpuError.OutOfMemory:` genuinely dispatches on the real cause:

```boring
mut k = Scale(data)
try:
    kernel:
        k(block = 256)
catch GpuError.OutOfMemory:
    print "GPU ran out of memory"
```

Verified end to end via a real `cargo check` against real wgpu 22.1.0: the generated downcast/match codegen for `catch GpuError.OutOfMemory:` is identical in shape to what a user-declared `throws CalcError` enum already produces (`book.md`).

---

## Device-to-device chaining

Feeding one kernel's output directly into another kernel's constructor (`Scale(k1.buf)`) copies the buffer via a generated `__boring_gpu_copy_d2d` helper — allocate a fresh `wgpu::Buffer` and issue a `copy_buffer_to_buffer` command on the shared queue:

```rust
fn __boring_gpu_copy_d2d(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer) -> std::sync::Arc<wgpu::Buffer> {
    let size = src.size();
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src, 0, &dst, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    std::sync::Arc::new(dst)
}
```

No manual `device.poll()`/wait is needed around the copy itself — it's a GPU command, and wgpu guarantees submission order within the same queue, so a later kernel dispatch that reads `dst` is correctly ordered after this copy completes.

This is deliberately **not** `Arc::clone(&wgpu::Buffer)`: cloning the `Arc` only bumps a reference count on the same underlying `wgpu::Buffer` — it does not allocate new GPU memory or copy any bytes. Using `Arc::clone` here used to mean two kernel structs silently shared the exact same buffer; if the source kernel was ever dispatched again afterward, the "copy"'s contents changed too, with no compile error and no warning (the same class of bug as the Metal backend's `Buffer::clone()`, and worse than the CUDA/ROCm backends' bug, which at least surfaces as a real `E0382` compile error).

---

## Type mapping

WGSL does not support 64-bit integers. Inside a `kernel` struct and its `def` body, Boring types are narrowed as follows:

| Boring type | Host Rust type | WGSL kernel type |
|---|---|---|
| `int` | `i64` | `i32` |
| `uint` | `u64` | `u32` |
| `float` | `f64` | `f32` |
| `bool` | `bool` | `u32` (see note) |

The narrowing is **silent for variables and fields** — the transpiler emits `i32` without warning. Integer literals inside a kernel body are checked at compile time: a literal outside the `i32` range (−2 147 483 648 to 2 147 483 647) is a compile error.

```boring
kernel Bad:
    let int n = 3_000_000_000    # compile error: literal 3000000000 exceeds i32 range on wgpu target
```

Arithmetic overflow in WGSL wraps silently (two's complement). This matches CUDA C `int` behavior on GPU.

### `bool` in storage buffers

WGSL forbids `bool` inside `storage` and `uniform` buffers (`array<bool>` is invalid). The transpiler maps `bool` fields and `bool` array elements to `u32` in all GPU memory contexts (`'unified`, `'global`, bare `'actor`, struct fields). `true` → `1u`, `false` → `0u`. Comparisons and conditionals that receive a `u32` from a GPU field coerce back to WGSL `bool` via `!= 0u` automatically.

---

## Generic kernel monomorphisation

Generic kernels (`kernel Blur<int N>:`) are specialised at `boring build --target wgpu` time — the transpiler scans the program for all `Name<arg, ...>()` instantiations and emits one WGSL struct + entry point per unique argument list.

### Naming scheme

| Boring instantiation | WGSL entry point | WGSL params struct |
|---|---|---|
| `Blur<3>()` | `Blur_3_main` | `Blur_3Params` |
| `Blur<7>()` | `Blur_7_main` | `Blur_7Params` |
| `GameOfLife<64, 64>()` | `GameOfLife_64_64_main` | `GameOfLife_64_64Params` |

Non-generic kernels keep their original name (`Scale_main`).

### Expression evaluation

Array sizes that are const-evaluable expressions are reduced at monomorphisation time:

```boring
kernel Tile<int W, int H>:
    mut [float, W * H]'unified weights   # → array<f32, 64> when W=8, H=8
```

Supported operators: `+`, `-`, `*`, `/`, `%`, unary `-`.

### Restrictions

- `[T]'actor` (dynamic workgroup memory) is not supported in WGSL — use `[T, N]'actor` with a const generic param instead.
- A generic kernel with no instantiations in the program emits no WGSL (no code is generated for unused generics).

---

## Known limitations vs CUDA

| Feature | CUDA | wgpu |
|---|---|---|
| `print` in kernel | `printf` | silent no-op — WGSL has no device-side print |
| Double precision | full support | **hard compile error** on a `kernel` field (`float64`/`float`) — WGSL has no 64-bit float type at all, see `float-width-types.md` §6 and "Will these gaps close?" below |
| `priority =` | `cuStreamCreateWithPriority` | no-op — wgpu has no queue priority API |
| `freeMem()` | `cuMemGetInfo` | always 0 — wgpu does not expose free VRAM |
| `computeCapability()` | CUDA SM version | always `[0, 0]` — not applicable |
| Multi-device (`GPU(1)`, `new(g1) K`) | full support — distinct `CudaContext` per index | `GPU(n)`/`.name()`/`.maxThreads()`/etc. resolve to real, distinct physical adapters when more than one is present (see "`GPU` type on wgpu" above) — introspection only. `new(g1) K` (actually placing a kernel's *dispatch* on a specific device) is still not implemented; every kernel dispatches on the single global `device`/`queue` regardless of which `GPU(n)` it was constructed with |
| Windows / Linux / macOS | Windows + Linux | yes — Windows (DX12), Linux (Vulkan), macOS (Metal via wgpu) |

### Will these gaps close?

wgpu (`gfx-rs/wgpu`) is actively developed — releases continuing through
2026 (v29.x in March 2026), not a frozen compatibility shim. But the rows
above split into two very different buckets for how likely (and how soon)
they are to close:

**Boring-side integration gaps, not wgpu gaps** — closable by Boring
engineering alone, no upstream dependency: multi-device introspection
(`GPU(n)`/`.name()`/etc. above) was exactly this, and is now real (real
per-adapter enumeration, 2026-09-01 — see "`GPU` type on wgpu" above).
`new(g1) K` real per-device *dispatch* remains in this bucket — a bigger
task (needs a `wgpu::Device`/`wgpu::Queue` per selected adapter instead of
today's single global pair, plus routing every kernel instance's
buffers/dispatch through the device it was actually constructed on;
comparable in scope to CUDA's real per-context multi-device model), but
still just Boring's own work whenever a project needs cross-GPU placement.

**Genuine WebGPU-spec-level gaps, outside wgpu's own control** — these are
structural to the spec, not a wgpu backlog item, and unlikely to close on
any timeline relevant to this project:
- **Double precision**: WGSL's real f64 path is `naga`'s own **non-standard
  extension** (`enable naga_ext_f64;`), not a stable, portable wgpu feature —
  SPIR-V/Vulkan f64 support depends on the underlying driver actually
  exposing `shaderFloat64` with **no portable way to probe for it** ahead of
  shader compilation, and DX12 has had outright compilation failures with
  64-bit-type features. Standardization is tracked upstream
  ([gpuweb/gpuweb#2805](https://github.com/gpuweb/gpuweb/issues/2805)) but
  unresolved. This is why `float64` stays a hard compile error on this
  target (`float-width-types.md` §6) rather than a silent narrowing or a
  "just enable the feature" fix — that's the right call to keep, not a gap
  to close, until `naga_ext_f64` (or a ratified spec extension) becomes a
  reliably portable, probeable feature upstream.
- **Device-side `print`/debug**: no `printf` equivalent in WGSL. Under
  active discussion upstream ([gpuweb#4704 "shaderLog"](https://github.com/gpuweb/gpuweb/issues/4704),
  [gpuweb#4348 debugPrintfEXT request](https://github.com/gpuweb/gpuweb/issues/4348)),
  but unratified — goes through the W3C spec process across every browser
  vendor, not through wgpu alone. Multi-year horizon at best.
- **Queue/stream priority**: no portable equivalent of CUDA streams'
  priority scheduling across Vulkan/Metal/DX12. Open wgpu issues in this
  area ([gfx-rs/wgpu#5576](https://github.com/gfx-rs/wgpu/issues/5576),
  [#5525](https://github.com/gfx-rs/wgpu/discussions/5525)) are about
  transfer queues and multithreading throughput, not priority — no sign of
  movement toward a portable priority primitive, likely because it isn't one
  across backends.
- **`computeCapability()`-style vendor introspection**: deliberately
  excluded — the entire point of WebGPU is a portable common denominator,
  so a CUDA-specific versioning concept is unlikely to appear generically.

Practical takeaway: a project needing debug printf, stream priority, or
CUDA-specific device introspection has no reason to expect wgpu to close
that gap soon and should stay on `--target cuda`/`--target rocm`. A project
whose only real need from a GPU backend is portable compute *and*
GPU-native display (games, visualizations) essentially never needs any of
these — wgpu's current feature set already covers that case in full, with
genuine GPU-native presentation today on Vulkan/DX12/Metal (see
`gpu-display.md`) and zero CPU round-trip, unlike `--target cuda`/`rocm`'s
software-blit display path (`clone_dtoh` + `softbuffer` every frame).

---

## GPU display

`boring build --target wgpu` supports live GPU rendering via `Screen`, `'surface`, and `kernel: loop:`. The surface buffer (a `'surface` field of type `[uint]`) is treated as a `STORAGE | MAP_READ | COPY_SRC` buffer. Each frame, `screen.present()` blits it to a `wgpu::Surface` texture via a copy command.

See [`gpu-display.md`](gpu-display.html) for the full display reference. wgpu-specific notes:

- **Window**: winit 0.30 + `wgpu::Surface` tied to the window handle.
- **Pixel format**: `Bgra8Unorm` — same packing as the Metal backend (`0xAARRGGBB`). The backend requests `Bgra8Unorm` explicitly (supported on DX12, Vulkan, and Metal) so pixel code is portable across `--target metal` and `--target wgpu` with no source change.
- **Blit**: `encoder.copy_buffer_to_texture(surface_buf → swapchain_texture)` each frame, followed by `queue.present()`.
- **Drawable size**: fixed at the kernel's `Dimension` — not updated on window resize.
- **Extra dependencies** added to `Cargo.toml` when `Screen` is present: `winit = "0.30"`.

---

## Generated project layout

```
<stem>_wgpu/
  src/main.rs        # Rust host code (wgpu crate)
  shaders/main.wgsl  # WGSL compute shader
  Cargo.toml         # wgpu = "22", no build.rs
```

Requires wgpu 22+ and a DirectX 12, Vulkan, or Metal-capable GPU. No external GPU toolkit needed.

```sh
boring build --target wgpu main.br
cd main_wgpu && cargo build
```

**Multi-file projects**: `use <file>.br` in the entry file is resolved and
inlined before transpilation — first relative to the importing file's own
directory, then against each path in the `BORING_PATH` environment variable
(same search order as `boring run`). Circular and duplicate imports are
merged once. A `use` that doesn't resolve to a `.br` file on disk (e.g.
`use std.collections`) is left as an ordinary import for the general
transpiler to handle.

---

## Implementation notes

- **Host crate**: `wgpu` v22.
- **Binding index ordering**: `'unified`, `'global`, `'actor'global`, `'actor'unified`, and `'surface` arrays first (in declaration order), then the `'const`/scalar-`'local` params struct as a single uniform buffer with the next binding index. The same order is used in the WGSL `@group(0) @binding(N)` annotations and the Rust `BindGroupLayoutEntry` list.
- **Workgroup size**: encoded via WGSL pipeline overrides (`override block_x: u32`); set at pipeline-creation time from the `block =` dispatch argument.
- **Dispatch**: the Rust host calls `encoder.dispatch_workgroups(gx, gy, gz)` and submits immediately. The `kernel:` block calls `queue.submit(...)` followed by `device.poll(wgpu::MaintainBase::Wait)` to ensure completion before the next host line.
- **Bare `'actor` (workgroup memory)**: fixed-size `[T, N]'actor` fields are emitted as `var<workgroup> tile: array<T, N>`. Dynamic `[T]'actor` fields are not supported in WGSL — the transpiler rejects them with a compile-time error and suggests using `[T, N]'actor` instead.
- **Atomics**: `'actor'global`/`'actor'unified` fields are emitted as `array<atomic<i32>>` (or `atomic<u32>` for `uint`). All five compound-assignment operations (`+= -= &= |= ^=`) map to the corresponding WGSL `atomicAdd` / `atomicSub` / `atomicAnd` / `atomicOr` / `atomicXor`.
- **`atomicSub`**: WGSL has a native `atomicSub` — no negation workaround needed (unlike the Metal backend).
