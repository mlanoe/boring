# ROCm backend

`boring build --target rocm` generates a Rust + HIP C++ project that runs GPU kernels on AMD GPUs using ROCm's HIP runtime — a native alternative to `--target cuda` for AMD hardware, requiring no NVIDIA GPU or CUDA toolkit.

The Boring source is identical: the same `kernel` structs, qualifiers, and `gpu.*` built-ins work unchanged across CUDA, Metal, wgpu, and ROCm targets. For the language reference, see [`gpu-module.md`](gpu-module.html). For the canonical qualifier model and dispatch-parameter reference, see [`cuda-module.md`](cuda-module.html) — this document only covers what's different about ROCm.

---

## Motivation

CUDA requires an NVIDIA GPU and the CUDA toolkit; ROCm is AMD's equivalent stack, and HIP (Heterogeneous-compute Interface for Portability) is deliberately designed to be a near-1:1 source-level match for CUDA C and the CUDA driver API — `--target rocm` gives AMD GPU users a native target without going through wgpu's cross-platform (but less direct) WGSL path.

---

## Device-side mapping

HIP C++'s kernel-side syntax is source-compatible with CUDA C by design: `__global__`/`__device__`/`__constant__`/`__shared__` qualifiers, `threadIdx`/`blockIdx`/`blockDim`/`gridDim`, `atomicAdd` and friends, `__syncthreads()`, and device-side `printf` all carry the exact same names and semantics. The device-code emitter (`src/transpiler/rocm/device.rs`) is therefore a near-verbatim clone of the CUDA one — same tables as [`cuda-module.md`](cuda-module.html)'s "CUDA C mapping" section apply unchanged, with `#include <hip/hip_runtime.h>` in place of `#include <cuda_runtime.h>`.

| Boring | CUDA C | HIP C++ |
|---|---|---|
| `gpu.thread.x` | `threadIdx.x` | `threadIdx.x` |
| `gpu.block.x` | `blockIdx.x` | `blockIdx.x` |
| `gpu.block_dim.x` | `blockDim.x` | `blockDim.x` |
| `gpu.grid_dim.x` | `gridDim.x` | `gridDim.x` |
| `sync` | `__syncthreads()` | `__syncthreads()` |
| `gpu.warp.size` | `warpSize` | `warpSize` (device built-in variable, but a *runtime* value on HIP — 32 or 64, RDNA vs CDNA — unlike CUDA's compile-time constant) |
| `gpu.warp.lane` | `threadIdx.x/y/z` linearized, `% warpSize` | same |
| `gpu.warp.sync()` | `__syncwarp(0xffffffff)` | `__syncwarp(0xffffffff)` |
| `gpu.warp.shuffle_down(v, delta)` | `__shfl_down_sync(0xffffffff, v, delta)` | same |
| `gpu.warp.shuffle_up(v, delta)` | `__shfl_up_sync(0xffffffff, v, delta)` | same |
| `gpu.warp.shuffle_xor(v, mask)` | `__shfl_xor_sync(0xffffffff, v, mask)` | same |
| `gpu.warp.shuffle(v, lane)` | `__shfl_sync(0xffffffff, v, lane)` | same |
| `'unified` field | `cudaMallocManaged` | `hipMalloc` + host-visible copy via `DeviceBuffer<T>` (see below — HIP has no single-call managed-memory equivalent used here) |
| `'global` field | `cudaMalloc` | `hipMalloc` |
| bare `'actor` field | `__shared__` | `__shared__` |
| `'const` scalar field | `__constant__ T name;` | `__constant__ T name;` |
| `'const` fixed array field (`[T, N]`) | `__constant__ T name[N];` | `__constant__ T name[N];` |
| atomic `[i] +=` on `'actor'global`/`'actor'unified` | `atomicAdd` | `atomicAdd` |
| `[i].min(v)` on `'actor'global`/`'actor'unified` | `atomicMin(&x, v)` | `atomicMin(&x, v)` |
| `[i].max(v)` on `'actor'global`/`'actor'unified` | `atomicMax(&x, v)` | `atomicMax(&x, v)` |
| `[i].swap(v)` on `'actor'global`/`'actor'unified` | `atomicExch(&x, v)` | `atomicExch(&x, v)` |
| `[i].cas(expected, new)` on `'actor'global`/`'actor'unified` | `atomicCAS(&x, expected, new)` | `atomicCAS(&x, expected, new)` |
| `print` in kernel | `printf` | `printf` |

---

## Host-side compilation model

There is no mature, widely-used safe Rust crate for ROCm/HIP analogous to `cudarc` (the crate the CUDA backend's host code is built on). Rather than depend on an unverified or unmaintained third-party binding, this backend hand-rolls a small `extern "C"` FFI layer directly into the generated `src/main.rs`, linked against `libamdhip64` (ROCm's stable, documented HIP runtime C API — `hipModuleLoadData`, `hipModuleLaunchKernel`, `hipMemcpy*`, `hipStreamCreateWithPriority`, etc.), plus a safe wrapper around it that deliberately mirrors cudarc's own shape and method names:

| cudarc (CUDA backend) | ROCm backend equivalent |
|---|---|
| `CudaContext` | `HipContext` |
| `CudaStream` | `HipStream` |
| `CudaModule` / `CudaFunction` | `HipModule` / `HipFunction` |
| `CudaSlice<T>` | `DeviceBuffer<T>` |
| `.alloc_zeros()` / `.clone_htod()` / `.clone_dtoh()` | same names |
| `.launch_builder()` / `.arg()` / `.launch()` | same names — `.arg()` takes a small `BoringKernelArg` trait implemented for `&mut DeviceBuffer<T>` (pushes the device pointer's own address) and for scalar/`Dimension` types (pushes the value's address directly) |
| PTX loaded via `nvrtc`/`Ptx::from_src` | a precompiled HIP code object, embedded via `include_bytes!` and loaded with `hipModuleLoadData` |

Keeping the same method names/shapes as cudarc means the ~1300 lines of Boring-AST-to-Rust statement/expression codegen shared with the CUDA backend (`emit_fn`/`emit_stmt`/`expr`/pattern matching/dict handling/GPU-residency tracking) carry over with only mechanical type-name substitutions, instead of re-deriving that whole pipeline from scratch.

### Compilation pipeline

```
kernels/main.hip  --hipcc --genco-->  code object (embedded via include_bytes!)
                                            |
                                    hipModuleLoadData (runtime)
```

`build.rs` invokes `hipcc --genco` (HIP's code-object generator, the HIP analogue of `nvcc --ptx`) to compile `kernels/main.hip` into a loadable code object, and links `libamdhip64` via `ROCM_PATH` (defaults to `/opt/rocm`). Set `BORING_ROCM_ARCH` (e.g. `gfx1100`) to target a specific GPU architecture; left unset, `hipcc` uses its own default detection.

---

## `GPU` type on ROCm

| Boring method | ROCm/HIP API |
|---|---|
| `name()` | `hipDeviceGetName` |
| `totalMem()` | `hipDeviceTotalMem` |
| `freeMem()` | `hipMemGetInfo().0` |
| `computeCapability()` | `hipDeviceComputeCapability` |
| `warpSize()` | `hipDeviceGetAttribute` + build-time header probe — see below |
| `maxThreads()` | `hipDeviceGetAttribute` + build-time header probe — see below |
| `maxSharedMem()` | `hipDeviceGetAttribute` + build-time header probe — see below |
| `index()` | index passed to `GPU(i)` |

`hipDeviceAttribute_t`'s numeric enum values are not guaranteed ABI-stable across ROCm releases, unlike CUDA's driver enum (which cudarc depends on directly). Rather than hardcode an attribute ID that could silently mean something different on a different ROCm version, `build.rs` compiles and runs a tiny C probe against whatever `hip/hip_runtime_api.h` is actually installed on the build machine, reads back the real `hipDeviceAttributeWarpSize`/`hipDeviceAttributeMaxThreadsPerBlock`/`hipDeviceAttributeSharedMemPerBlock` values, and bakes them into a generated `OUT_DIR/boring_hip_attrs.rs` that the host prelude `include!()`s. `warpSize()`/`maxThreads()`/`maxSharedMem()` then call `hipDeviceGetAttribute` with those probed constants. If the probe can't be compiled or run (e.g. no ROCm headers available at build time), the constants fall back to `-1` and the three methods report a clean runtime error instead of querying with a guessed ID.

`GPU(0)` → `hipSetDevice(0)` (HIP has no separate context-creation call the way the CUDA driver API does — a `HipContext` records which device ordinal every subsequent op should target).
`GPU.all()` → `hipGetDeviceCount()`, indexed 0..count.

---

## Known limitations vs CUDA

| Feature | CUDA | ROCm |
|---|---|---|
| `warpSize`/`maxThreads`/`maxSharedMem` | full support, hardcoded CUDA driver enum | full support — via a build-time probe of the local ROCm headers instead of a hardcoded ordinal, since `hipDeviceAttribute_t` isn't ABI-stable across ROCm versions (see above) |
| Constant-memory upload | `get_global` + `transmute_mut` + `memcpy_htod` (3 cudarc calls) | single `HipModule::upload_constant` helper — same effect, one call, since this backend controls both sides of the FFI |
| `after =` ordering | CUDA streams + events | HIP streams + events (`hipStreamWaitEvent`) — same GPU-side, non-blocking semantics |
| Toolchain | `nvcc` | `hipcc` |
| Verified against real hardware | — | **yes**, as of 2026-08-30 — HIP SDK 7.2 on an AMD Radeon RX 6600 (`gfx1032`, RDNA2/Navi 23; runs natively, no `HSA_OVERRIDE_GFX_VERSION` needed despite the official HIP SDK Windows support matrix listing `gfx1032` as unsupported). `examples/vector_add_gpu.br`, `matrix_mul_gpu.br`, and `mandelbrot_gpu.br` all compile, run, and produce output numerically identical (byte-identical for `mandelbrot_gpu.br`'s output PPM) to `boring run`'s interpreter simulation. Found and fixed 6 real bugs in the process — see below and the `map_builtin_fn`/`Stmt::With`/`try_gpu_field_read`/`kernel_ctor_buffer_flags`/`auto_grid`/`__stream` doc comments in `rocm::device`/`rocm::host` (all six also existed identically in `cuda::device`/`cuda::host`, this backend's near-verbatim clone, and were fixed there too — still unverified on real CUDA hardware, none available in this project's dev environment) |

### Bugs found via this hardware validation (all fixed)

1. **`float32(x)`/`uint8(x)`/etc. call-style casts in device code** emitted an invalid, undeclared function call (`hipcc`: `use of undeclared identifier 'float32'`) instead of a C cast — `map_builtin_fn` was missing every scalar type name except `int`/`float`. Metal's own `map_builtin_fn` already had the equivalent fix; it had just never been ported to CUDA/ROCm.
2. **`with buf: body` silently dropped its entire body** on the host side (fell through to the `_ => "/* unsupported stmt */"` catch-all) — a program would compile and exit 0 while printing nothing, since every one of these examples' result-printing loop lives inside a `with` block.
3. **GPU→host readback never triggered for a fixed-shape (`LabeledArray`) `'unified`/`'global` field** (e.g. `[float32, width=32, height=32]`) — `try_gpu_field_read` only recognized bare `Type::Array`/`Type::ArrayN`, so `k.c` stayed an un-read-back raw `DeviceBuffer<f32>` (`error[E0608]: cannot index into a value of type DeviceBuffer<f32>`).
4. **Host→GPU upload never triggered for a fixed-shape ctor parameter** — `kernel_ctor_buffer_flags` had the identical `LabeledArray`-blind-spot, so a host `Vec<f32>` was passed directly where a `DeviceBuffer<f32>` was expected (`error[E0308]`).
5. **`__boring_launch`'s generated signature and its call site disagreed on `grid_dim`'s type** for a fixed-shape kernel — the signature (correctly) used `.as_labeled_array()` to decide `Option<(u32,u32,u32)>` vs. a bare tuple; three separate call-site `auto_grid` checks did not, and passed a bare tuple against the `Option<...>`-typed parameter (`error[E0308]`).
6. **Silent stream-synchronization race producing wrong (near-zero) output**, not a compile error: a kernel instance's own `__stream` field (used by its `read_{field}()` readback accessor) was a brand-new, independently-created HIP stream (`__ctx.default_stream()`), with no ordering relationship to the *different* cached per-priority stream every other operation (uploads, the actual kernel dispatch) used — so a D2H readback could complete before the kernel it was reading from had. Every buffer-allocation/upload site in the constructor had the same issue. Fixed by routing all of it through the same cached-per-priority stream (`boring_new_stream_with_priority`), matching this file's own documented stream design. Also found and fixed alongside it: an unassigned fixed-shape field's default zero-fill allocation was hardcoded to a 1-element buffer regardless of its actual shape (`error: index out of bounds` at readback).

---

## Generated project layout

```
<stem>_rocm/
  src/main.rs        # Rust host code (hand-rolled HIP FFI + safe wrapper, no external GPU crate)
  kernels/main.hip   # HIP C++ device code
  build.rs           # invokes hipcc --genco, links libamdhip64
  Cargo.toml         # no GPU crate dependency
```

Requires the ROCm toolkit (`hipcc`) and an AMD GPU.

```sh
boring build --target rocm main.br
cd main_rocm && cargo build
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

- **Host FFI**: hand-rolled `extern "C"` bindings to the subset of the HIP runtime API needed (module loading, buffer alloc/copy, stream/event, kernel launch) — no external crate dependency. Linked against `libamdhip64` via `build.rs`'s `cargo:rustc-link-lib`.
- **`DeviceBuffer<T>`**: wraps a raw `hipMalloc`'d pointer + length + the `Arc<HipStream>` it was allocated on. `Drop` calls `hipFree`. `Clone` does a real device-to-device `hipMemcpyDtoD` — every kernel-chaining call site (`Scale(k1.buf)`) uses it, matching the CUDA backend's identical `CudaSlice::clone()` behavior. A bare move used to be the actual codegen here, but it applied unconditionally regardless of whether the source kernel variable is used again afterward — a real `E0382` ("use of partially moved value") the moment it was. `.clone()` is correct in every case and still far cheaper than a host round trip.
- **Kernel launch**: `hipModuleLaunchKernel` takes a `void**` array where each entry points to the actual parameter value — `LaunchBuilder::arg` pushes a pointer to the device-pointer field itself for buffer args, and a pointer to the scalar's own storage for scalar/`Dimension` args. Both are valid only for the duration of the synchronous launch call, which happens while the borrows are still alive.
- **Streams**: one persistent, FIFO-ordered `HipStream` cached per dispatch priority (mirrors the CUDA backend's identical fix) — GPU-side ordering (`hipStreamWaitEvent`) for `after =` dependencies instead of a CPU-blocking sync on every dispatch. Every kernel-instance operation (constructor buffer alloc/upload, the struct's own readback `__stream` field, `__boring_launch`'s dispatch) now goes through this same cached stream — see the "Known limitations" table's bug #6 above for what broke before that was consistent (a silent cross-stream data race, not a compile error).
- **Error handling**: same story as CUDA's — see [`cuda-module.md`](cuda-module.html#error-handling). The built-in [`GpuError`](gpu-module.html#gpu-error-handling) enum is not catchable by variant here either (this backend's own small host transpiler doesn't share the general pipeline `BoringError`/`catch`-by-variant lives in). `HipError`'s `Display` now prefixes its existing `hipGetErrorString` message with a classified category (out of memory, illegal access, timeout, ...), using numeric codes assumed to mirror CUDA's `CUresult` values one-for-one (HIP is designed as a near-1:1 match for the CUDA driver API) — not independently verified (no error path was hit during the real-hardware validation run — see the "Known limitations" table above).
- **Verified against real ROCm hardware** as of 2026-08-30 — see the "Known limitations" table above for hardware, scope, and the bugs this surfaced and fixed.
