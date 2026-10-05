# Metal backend

`boring build --target metal` generates a Rust + MSL project that runs GPU kernels on macOS using Apple's Metal framework — a native alternative to `--target cuda` that requires no NVIDIA GPU or CUDA toolkit.

The Boring source is identical: the same `kernel` structs, qualifiers, and `gpu.*` built-ins work unchanged across CUDA and Metal targets.

---

## Motivation

CUDA requires an NVIDIA GPU and the CUDA toolkit — unavailable on macOS. Metal is available on every Mac (M-series and Intel with discrete GPU). On Apple Silicon, unified memory makes `'unified` fields zero-copy between CPU and GPU — a better physical match than `cudaMallocManaged` on x86.

---

## Qualifier mapping

| Boring qualifier | CUDA C | MSL address space |
|---|---|---|
| `'unified` | `cudaMallocManaged` | `device` + `MTLStorageMode.shared` (zero-copy on Apple Silicon) |
| `'global` | `__global__` | `device` |
| `'surface` | `cudaMallocManaged` (u32) | `device uint*` — `MTLStorageModeShared`, 32-bit per pixel (BGRA8Unorm) |
| bare `'actor` | `__shared__` | `threadgroup` |
| `'local` | registers | thread-private (default) |
| `'const` scalar | `__constant__ T name;` | `constant T* name [[buffer(N)]]` — dereferenced (`*name`) in body |
| `'const` fixed array (`[T, N]`) | `__constant__ T name[N];` | `constant T* name [[buffer(N)]]` — accessed as `name[i]` in body |
| `'actor'global` | device DRAM, atomic access | `device T*` — cast to the atomic type matching `T`'s own width at the atomic call site (`atomic_int`/`atomic_uint` for `int32`/`uint32`, `atomic_long` only for the genuinely-64-bit `int`/`uint`) |
| `'actor'unified` | unified DRAM, atomic access | `device T*` + `MTLStorageModeShared` — same width-matched atomic cast as `'actor'global` |

---

## Built-in mapping

| Boring | CUDA C | MSL |
|---|---|---|
| `gpu.thread.x/y/z` | `threadIdx.x/y/z` | `thread_position_in_threadgroup.x/y/z` |
| `gpu.block.x/y/z` | `blockIdx.x/y/z` | `threadgroup_position_in_grid.x/y/z` |
| `gpu.block_dim.x/y/z` | `blockDim.x/y/z` | `threads_per_threadgroup.x/y/z` |
| `gpu.grid_dim.x/y/z` | `gridDim.x/y/z` | `threadgroups_per_grid.x/y/z` |
| `sync` (manual) | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `gpu.warp.size` | `warpSize` | `threads_per_simdgroup` |
| `gpu.warp.lane` | linearized `threadIdx`, `% warpSize` | `thread_index_in_simdgroup` |
| `gpu.warp.sync()` | `__syncwarp(0xffffffff)` | `simdgroup_barrier(mem_flags::mem_none)` |
| `gpu.warp.shuffle_down(v, delta)` | `__shfl_down_sync(0xffffffff, v, delta)` | `simd_shuffle_down(v, delta)` |
| `gpu.warp.shuffle_up(v, delta)` | `__shfl_up_sync(0xffffffff, v, delta)` | `simd_shuffle_up(v, delta)` |
| `gpu.warp.shuffle_xor(v, mask)` | `__shfl_xor_sync(0xffffffff, v, mask)` | `simd_shuffle_xor(v, mask)` |
| `gpu.warp.shuffle(v, lane)` | `__shfl_sync(0xffffffff, v, lane)` | `simd_shuffle(v, lane)` |
| bare-`'actor` auto-barrier | inserted before first loop + at top of each loop iteration accessing bare-`'actor` fields | idem |
| atomics (`'actor'global`/`'actor'unified`) | `atomicAdd` etc. | `atomic_fetch_add_explicit` etc. |
| `[i].min/max/swap(v)` | `atomicMin`/`atomicMax`/`atomicExch` | `atomic_fetch_min_explicit`/`atomic_fetch_max_explicit`/`atomic_exchange_explicit` |
| `[i].cas(expected, new)` | `atomicCAS` | `atomic_compare_exchange_weak_explicit` (bridged — see below) |

`.min`/`.max`/`.swap`/`.cas` are methods on an indexed `'actor'global`/`'actor'unified` element, handled in expression position (they return the previous value, matching CUDA/HIP's real semantics) rather than as a statement-only compound-assign desugar like `+= -= &= |= ^=`. `min`/`max`/`swap` map straight onto MSL's `atomic_fetch_min/max_explicit`/`atomic_exchange_explicit`, which already return the previous value — cast to the atomic type matching the field's own element width (`atomic_int`/`atomic_uint` for a 4-byte `int32`/`uint32` element, `atomic_long` only for the genuinely-64-bit bare `int`/`uint`), not a blanket 8-byte `atomic_long` regardless of element width — an earlier version of this backend cast every atomic op to `atomic_long` unconditionally, which read/wrote 4 bytes past a 4-byte element's valid allocation. An element type with no portable MSL atomic (8/16-bit ints, floats, the already-unsupported 64/128-bit named widths) emits a flagged `/* ERROR: ... */` comment instead of an atomic op. (`atomic_fetch_min/max_explicit` on 64-bit `atomic_long` specifically is not independently verified against a real Metal compiler in this environment, same caveat this backend's docs already carry elsewhere for untestable-locally MSL codegen.)

`.cas` is a real shape mismatch: MSL's `atomic_compare_exchange_weak_explicit(object, &expected, desired, ...)` takes a *pointer* to the expected value (overwritten with the real current value on failure) and returns a `bool` — unlike CUDA/HIP's `atomicCAS`, which just returns the previous value directly. Bridged via a GNU/Clang statement-expression (`({ ... })`, supported by Metal's Clang-based compiler — `metal`'s own generated `Debug` impl already relies on the same compiler being Clang-based for other things) so the whole thing is still usable as one expression, e.g. for a 4-byte `int32` element:

```msl
({ int __exp = (int)(expected); atomic_compare_exchange_weak_explicit((device atomic_int*)&x, &__exp, (int)(new), memory_order_relaxed, memory_order_relaxed); __exp; })
```

(The genuinely-64-bit bare `int`/`uint` case still casts to `atomic_long`/`long`, as before.)

**Without `'actor'global`/`'actor'unified`**, `.min`/`.max`/`.swap`/`.cas` still work — matching `+= -= &= |= ^=`'s existing degrade-to-plain-arithmetic behavior off a non-actor field — bridged via the same GNU/Clang statement-expression, just without the atomic cast or memory order: `({ auto __old = x; x = min(x, (v)); __old; })`.

`gpu.warp.size`/`.lane` are emitted as new `[[thread_index_in_simdgroup]]`/
`[[threads_per_simdgroup]]` kernel parameters, unconditionally alongside the
existing position parameters below — MSL SIMD-group builtins need no
capability/enable step, unlike wgpu's subgroup builtins (see
[warp-level primitives](warp-level-primitives.html)).

### MSL kernel signature

CUDA C uses flat parameter lists; MSL uses annotated parameters. The transpiler generates `[[buffer(N)]]` / `[[threadgroup(N)]]` indices automatically:

```msl
kernel void scale(
    device float* buf       [[buffer(0)]],
    threadgroup float* tile [[threadgroup(0)]],
    uint3 thread_pos        [[thread_position_in_threadgroup]],
    uint3 block_pos         [[threadgroup_position_in_grid]],
    uint3 block_dim         [[threads_per_threadgroup]]
) {
    uint i = thread_pos.x + block_pos.x * block_dim.x;
    buf[i] *= 2.0;
}
```

---

## Naming restrictions

Unlike Rust, MSL's builtin scalar type names (`half`, `float`, `int`, `long`, `size_t`, ...), its builtin vector/matrix types (`float4`, `half2x2`, ...), and its real C++/MSL keywords (`class`, `template`, `kernel`, `device`, `threadgroup`, ...) live in the *same identifier namespace* as ordinary variables — declaring a variable literally named `half` is a genuine MSL parse error, not merely a style warning.

A kernel field, a `def()`-body local, a for-loop variable, or a function/method parameter that collides with one of these reserved words is automatically renamed (a trailing `_` appended) by `msl_safe_ident` (`src/transpiler/metal/device.rs`), consistently at both its declaration and every later reference — this is transparent and requires no action from Boring source. `half` is the motivating real-world case (an unremarkable name for a RoPE positional-encoding kernel's `d_head / 2`), but the same mangling covers the entire reserved-word list, not just that one name.

Without this, the failure would be especially confusing: `boring build --target metal` transpiles successfully with no warning, since this backend never parses the MSL it emits — the collision only surfaces later, at runtime, when the generated binary compiles its own MSL source via `newLibraryWithSource`, as a wall of unrelated-looking parse errors cascading from the single bad declaration.

---

## MSL compilation

MSL is compiled at runtime via `newLibraryWithSource` — the Metal compiler is built into macOS. No external toolchain (`xcrun`, LLVM) is needed.

The generated runtime caches Metal's default device, indexed-device enumeration,
command queue, compiled MSL library, and compute pipeline state objects for the
life of the calling thread. Subsequent dispatches clone the cached Objective-C
handles (a cheap retain) instead of rediscovering the device through WindowServer,
recompiling the shader source, or rebuilding the pipeline. `GPU(n)` keeps its
existing indexed-device behavior; its `MTLCopyAllDevices` result is cached too.

The generated project has no `build.rs`. Compilation happens once at app startup. An AOT path (`boring build --target metal --aot`) may be added in a later iteration for workloads where startup latency matters.

---

## `GPU` type on Metal

| Boring method | Metal API |
|---|---|
| `name()` | `device.name()` |
| `totalMem()` | `device.recommendedMaxWorkingSetSize()` |
| `freeMem()` | total − `device.currentAllocatedSize()` |
| `computeCapability()` | Metal GPU family tier as `[major, minor]` |
| `warpSize()` | 32 (conservative default — matches Apple Silicon SIMD group size) |
| `maxThreads()` | 1024 (conservative default — per pipeline, not device-global) |
| `maxSharedMem()` | `device.maxThreadgroupMemoryLength()` |
| `index()` | index passed to `GPU(i)` |

`GPU(0)` → `MTLCreateSystemDefaultDevice()`.
`GPU.all()` → `MTLCopyAllDevices()`, sorted by index.

---

## Error handling

Kernel dispatch is deferred: `__boring_launch` only commits the command buffer, and the actual `wait_until_completed()` happens lazily, at the next point host code actually reads GPU-written data back (`read_<field>()`, or the shared `__boring_gpu_copy_d2h` helper) — see the generated prelude's `__boring_metal_flush` for the full rationale (this is a real, measured performance win, not just plumbing).

That flush point is also where a GPU-side failure (invalid threadgroup size, out-of-bounds buffer access, device removal, ...) actually surfaces: the command buffer's own `status()` is checked after `wait_until_completed()`, and `read_<field>()` returns `Result<Vec<T>, Box<dyn std::error::Error + Send + Sync>>` instead of the plain `Vec<T>` it used to — propagating via `?` up to `boring_main()`'s own `Result` (always present once any kernel is involved). Previously, a failed dispatch completed with `status() == Error` and nothing ever looked — `read_<field>()` read back whatever garbage or zeroed memory was left, and the Boring program reported success regardless.

There is no synchronous rejection at dispatch time the way CUDA's `cuLaunchKernel` can reject an invalid config immediately — Metal's `dispatch_thread_groups` has no `Result`-returning signature at all, so an invalid config is either caught by the (async) command-buffer status above, or — if Metal's own API validation layer is active — an assertion/abort outside Boring's control.

### Classified error messages, not typed `GpuError`

`status() == MTLCommandBufferStatus::Error` used to be the whole story — `{:?}` on the status enum just prints the literal word `Error`, no indication of the real cause. The `metal` crate exposes no safe `.error()` getter on `CommandBufferRef` (checked against real `metal` 0.29 source — no such method exists), but `CommandBufferRef` does implement `objc::Message`, so the real `NSError` is one `objc::msg_send![buf_ref, error]` away, classified against Apple's own `MTLCommandBufferError` codes (out of memory, page fault, timeout, device removed, ...). `objc` is now an **unconditional** dependency of every Metal-generated project (previously only added when `Screen` was present) since this flush path needs it regardless of whether the program does any display work.

This is a message improvement, not a catchable [`GpuError`](gpu-module.html#gpu-error-handling) — `catch GpuError.OutOfMemory:` is not reachable here. Metal's host transpiler is its own small, kernel-only transpiler (like CUDA's and ROCm's), not the general pipeline `BoringError`/`catch`-by-variant lives in — the same prerequisite gap already documented for `with` in `scoped-access-blocks.md`.

---

## Device-to-device chaining

Feeding one kernel's output directly into a read-only field of another kernel's constructor (`Consume(k1.out)`) retains the same `MTLBuffer` when that expression is the final use of `k1`. Queue ordering makes the consumer wait for the producer without a CPU flush. If the source kernel is used again later, or the destination field is mutable, `__boring_metal_buffer_copy` preserves independent-value semantics by allocating a fresh buffer and enqueuing a Metal blit on the shared command queue. The blit is asynchronous: it performs neither `wait_until_completed()` nor a host `memcpy`.

The independent-value path is deliberately **not** `Buffer::clone()`: in the real `metal` crate, `Clone` on an Objective-C wrapper type is just an ObjC `retain` (a reference-count bump), not a content copy. Aliasing is only selected for a read-only destination at a proven final use of the source kernel; otherwise a later source dispatch must not change the consumer's value.

---

## Known limitations vs CUDA

| Feature | CUDA | Metal |
|---|---|---|
| `print` in kernel | `printf` | silent no-op — no device-side printf in MSL |
| Float atomics | all GPUs | A13 / M1 and later only |
| Warp intrinsics | full support | SIMD-group operations (different API) |
| `after =` ordering | CUDA streams | synchronous dispatch — `after` is a no-op |
| Windows / Linux | yes | macOS only |
| Tensor-core (`gpu.tensor.*`) | not yet implemented on either backend | not yet implemented — see [tensor-core primitives](tensor-core-primitives.html) for a proposed design (`simdgroup_matrix`, 8×8 fp16/fp32 only, Apple7+/A14+) |

---

## GPU display

`boring build --target metal` supports live GPU rendering via `Screen`,
`'surface`, and `kernel: loop:`. See [`gpu-display.md`](gpu-display.html) for
the full reference. Metal-specific notes:

- **Window**: winit 0.28 + `CAMetalLayer` attached to the `NSView` via objc.
- **Pixel format**: `BGRA8Unorm` — pack pixels as `0xFF000000 | (r << 16) | (g << 8) | b`.
- **Blit**: `MTLBlitCommandEncoder` from the surface buffer to the `CAMetalDrawable` texture each frame.
- **Drawable size**: fixed at the kernel's surface `Dimension` — not updated on window resize.
- **2D dispatch**: when a kernel has a `'surface` field and a `Dimension` field, the grid is inferred as `(ceil(w/bx), ceil(h/by), 1)` automatically. A fixed-shape `[T, width=W, height=H]` labeled-array field (`'unified`/`'global`/`'actor'global`/`'actor'unified`) gets the same treatment from its compile-time axis sizes instead — `(ceil(W/bx), ceil(H/by), 1)`, generalized to 3 axes — independently of the `'surface`/`Dimension` case (see [`gpu-module.md`](gpu-module.html)).
- **Extra dependencies** added to `Cargo.toml` when `Screen` is present: `winit = "0.28"`, `objc = "0.2"`, `core-graphics = "0.23"`.

---

## Generated project layout

```
<stem>_metal/
  src/main.rs          # Rust host code (metal crate v0.29)
  kernels/main.metal   # MSL device code
  Cargo.toml           # metal = "0.29", no build.rs
```

Requires macOS 11+ with a Metal-capable GPU.

```sh
boring build --target metal main.br
cd main_metal && cargo build
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

- **Host crate**: `metal` v0.29 — simpler API, still maintained.
- **Buffer index ordering**: unified/global/actor arrays → const fields → local scalars → dynamic shared (`threadgroup`). Device and host use the same fixed order.
- **Dispatch**: synchronous — `wait_until_completed()` is called inside `__boring_launch`. `after =` is accepted syntactically but has no effect.
- **Atomics**: `atomicSub` has no MSL equivalent — emitted as `atomic_fetch_add_explicit` with a negated value, `memory_order_relaxed`.
