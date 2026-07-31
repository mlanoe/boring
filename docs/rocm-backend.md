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
| `'unified` field | `cudaMallocManaged` | `hipMalloc` + host-visible copy via `DeviceBuffer<T>` (see below — HIP has no single-call managed-memory equivalent used here) |
| `'global` field | `cudaMalloc` | `hipMalloc` |
| `'shared` field | `__shared__` | `__shared__` |
| `'const` scalar field | `__constant__ T name;` | `__constant__ T name;` |
| `'const` fixed array field (`[T, N]`) | `__constant__ T name[N];` | `__constant__ T name[N];` |
| atomic `[i] +=` on `'actor'global` | `atomicAdd` | `atomicAdd` |
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
| Verified against real hardware | — | not independently verified against a real AMD GPU/ROCm toolchain (none available in this project's dev environment) — validated via `cargo check` against the generated Rust with a stubbed build step instead, same caveat `metal-backend.md` documents for lacking a macOS toolchain |

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
- **Streams**: one persistent, FIFO-ordered `HipStream` cached per dispatch priority (mirrors the CUDA backend's identical fix) — GPU-side ordering (`hipStreamWaitEvent`) for `after =` dependencies instead of a CPU-blocking sync on every dispatch.
- **Not independently verified against real ROCm hardware** — see the "Known limitations" table above.
