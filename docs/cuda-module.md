# CUDA backend

For the language reference, see [`gpu-module.md`](gpu-module.html).

This document covers what the transpiler generates, how Boring constructs map to CUDA C, and the current limitations of the backend.

---

## Generated project layout

`boring build --target cuda` emits a Cargo project:

- `src/main.rs` — Rust host code using the `cudarc` crate
- `kernels/main.cu` — CUDA C device code
- `build.rs` — invokes `nvcc` to compile `.cu` → PTX
- `Cargo.toml`

Requires the CUDA toolkit (`nvcc`) and a CUDA-capable GPU.

---

## Qualifier model

GPU qualifiers appear in two contexts with different syntax:

- **Host context** — allocations and bindings outside a `kernel` struct. Full `'gpu'..` prefix is mandatory.
- **Kernel context** — fields inside a `kernel` struct. The `'gpu` prefix is dropped; short forms `'unified`, `'global`, `'shared`, `'local`, `'const` are used.

### Host-context qualifiers

| Qualifier | Location | CUDA |
|---|---|---|
| `'gpu'unified` | unified host + device DRAM | `cudaMallocManaged` |
| `'gpu'global` | device-only DRAM | device pointer |
| `'gpu'const` | GPU constant cache | `__constant__` |

### Kernel-context qualifiers

| Qualifier | CUDA memory space | Host access | Default |
|---|---|---|---|
| `'local` | registers / thread-local memory | no | yes — may be omitted |
| `'shared` | block SRAM (`__shared__`) | no | no |
| `'global` | device-only DRAM | via `gpu.copy()` | no |
| `'unified` | unified DRAM (`cudaMallocManaged`) | direct | no |
| `'const` | constant cache | no | implicit via `let` |

---

## Kernel struct field rules

| Field declaration | Memory space | Host access | CUDA mapping |
|---|---|---|---|
| `let [float]'unified input` | unified DRAM, device read-only | direct | `const float* const input` |
| `mut [float]'unified output` | unified DRAM, device writable | direct | `float* const output` |
| `let [float]'global input` | device-only DRAM, device read-only | via `gpu.copy()` | `const float* const input` |
| `mut [float]'global output` | device-only DRAM, device writable | via `gpu.copy()` | `float* const output` |
| `var [float]'global buf` | device-only DRAM, ptr + data mutable | via `gpu.copy()` | `float* buf` |
| `mut [float, N]'shared tile` | block SRAM, static | no | `__shared__ float tile[N]` |
| `mut [float]'shared tile` | block SRAM, dynamic | no | `extern __shared__ float tile[]` |
| `let float sigma` | `'const` implicit — scalar | no | `const float sigma` |
| `var int i` | `'local` implicit — register | no | `int i` |
| `mut [float, N] tile` | `'local` implicit — thread-local | no | `float tile[N]` |
| CPU qualifiers (`'heap`, `'actor`…) | — | — | compile-time error |

---

## CUDA C mapping

| Boring | CUDA C |
|---|---|
| `gpu.thread.x` | `threadIdx.x` |
| `gpu.block.x` | `blockIdx.x` |
| `gpu.block_dim.x` | `blockDim.x` |
| `gpu.grid_dim.x` | `gridDim.x` |
| `sync` | `__syncthreads()` |
| `'unified` field | `cudaMallocManaged` |
| `'global` field | `cudaMalloc` |
| `'shared` field | `__shared__` |
| `'const` scalar field | `__constant__ T name;` (file scope) |
| `'const` fixed array field (`[T, N]`) | `__constant__ T name[N];` (file scope) |
| atomic `[i] +=` on `'actor'global` | `atomicAdd` |
| `print` in kernel | `printf` |

### `[T, N]'const` codegen detail

A `let [T, N]` field (inferred `'const`) is emitted as a file-scope `__constant__` array — not as a kernel parameter. Device code accesses it directly by name:

```boring
kernel Lookup:
    let [float, 4] lut = [0.0, 0.25, 0.5, 1.0]
    mut [float]'unified output

    def ():
        let i = gpu.thread.x
        output[i] = lut[i % 4]
```

Generated CUDA C (excerpt):

```cuda
__constant__ float lut[4];

__global__ void lookup(float* output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output[i] = lut[i % 4];
}
```

The constant array is uploaded once via `cudaMemcpyToSymbol` before launch.

---

## Dispatch parameters

All dispatch parameters are passed inside a `kernel:` block as labeled args to the kernel variable.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `block` | `int` or `(int, int)` or `(int, int, int)` | yes | threads per block — 1D, 2D, or 3D |
| `grid` | `int` or tuple | no | blocks per grid — inferred from kernel shape if omitted |
| `after` | kernel var or `[k1, k2]` | no | GPU-side ordering — kernel starts after listed ones complete |
| `priority` | `"high"` / `"normal"` / `"low"` | no | stream scheduling priority — CUDA only, default `"normal"` |

**Grid inference rules (current implementation):**

- `[T]'global` / `'unified` 1D array field → `grid = ceil(len(buf) / block)`
- Otherwise, if `grid` is omitted, the transpiler defaults to `grid = (1, 1, 1)` — always pass `grid` explicitly unless relying on 1D array inference.

---

## `KernelHandle`

`kernel(...)` returns a `KernelHandle<T>` where `T` is the kernel struct type. The handle owns the kernel object until `.wait` is called.

```boring
struct KernelHandle<T>:
    req bool done()
    req T    wait()
```

`.wait` is the only way to recover the kernel object.

---

## `'sync` — block SRAM

`'sync` fields are allocated in per-block shared memory (`__shared__` in CUDA C). The transpiler computes `shared_mem_bytes` automatically from the block dimension and element size — no `smem =` dispatch parameter needed.

### Fixed size — `[T, N]'sync`

```boring
kernel Reduce:
    let [float]'unified      input
    mut [float]'unified      output
    mut [float, 256]'sync    tile

    init(int n):
        input  = [..n]
        output = [..n / 256]

    def ():
        let tid = gpu.thread.x
        tile[tid] = input[gpu.block.x * gpu.block_dim.x + tid]
        sync
        ...

mut k = Reduce(n)
kernel:
    k(block = 256)
```

### Dynamic size — `[T]'sync`

The size is `block_dim.x * sizeof(T)` — one element per thread in the block. Declare the field without a size; the transpiler passes `block_dim.0 * sizeof(T)` as `shared_mem_bytes`.

```boring
kernel Reduce:
    let [float]'unified  input
    mut [float]'unified  output
    mut [float]'sync     tile

    init(int n):
        input  = [..n]
        output = [..n / 256]

    def ():
        let tid = gpu.thread.x
        tile[tid] = input[gpu.block.x * gpu.block_dim.x + tid]
        sync
        ...

mut k = Reduce(n)
kernel:
    k(block = 256)
```

---

## Atomics — `'actor` on kernel fields

Only `'actor'global` is implemented. `'actor'unified` and `'actor'shared` are not yet supported (parse error on those suffixes).

| Qualifier | Location | Atomic scope |
|---|---|---|
| `mut [int]'actor'global bins` | device DRAM | all threads, all blocks |

**Operations transpiled automatically on `'actor'global` fields:**

| Boring | CUDA |
|---|---|
| `x += v` | `atomicAdd(&x, v)` |
| `x -= v` | `atomicSub(&x, v)` |
| `x \|= v` | `atomicOr(&x, v)` |
| `x &= v` | `atomicAnd(&x, v)` |
| `x ^= v` | `atomicXor(&x, v)` |

```boring
kernel Histogram:
    let [float]'global     input
    mut [int]'actor'global bins

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        if i < len(input):
            let bucket = int(input[i] * 10.0)
            bins[bucket] += 1    # → atomicAdd
```

---

## CPU simulation — substitution table

`boring run` executes kernels sequentially on the CPU. No source changes required.

| GPU primitive | CPU simulation |
|---|---|
| `'unified` fields (kernel) | heap allocation (`Vec<T>`) |
| `'global` fields (kernel) | heap allocation (`Vec<T>`) |
| `init` allocation | `Vec<T>::with_capacity(n)` |
| `'shared` (block SRAM) | local array (stack or heap) |
| `'local` (register) | local variable |
| `'gpu'const` | `let` binding |
| kernel launch | sequential loop over all threads |
| `gpu.thread.x/y/z`, `gpu.block.x/y/z`… | loop variables |
| `sync` | no-op |
| `'actor'global` fields | plain arithmetic (no contention) |
| `kernel(...)` streams | no-op — single-threaded sequential |
| `after =` | sequential execution in declaration order |

Sequential simulation hides data races. A kernel correct in simulation may be incorrect on real hardware if `sync` barriers are missing.

---

## Error handling

CUDA errors fall into two categories mapped to the two observation points of a `KernelHandle`.

### Synchronous errors — raised at `kernel(...)`

Detected immediately before execution. The handle is never created.

```boring
let h = kernel(block = 99999) k    # raise GpuLaunchError
```

### Asynchronous errors — raised at `.wait`

Detected only at synchronisation.

```boring
let h = kernel(block = 256) k
h.wait                              # raise GpuIllegalAccess if kernel faulted
```

### Error types

| Error | Phase | Cause |
|---|---|---|
| `GpuLaunchError` | `kernel(...)` | invalid config — block size, grid size |
| `GpuOutOfMemory` | `kernel(...)` | allocation failed |
| `GpuIllegalAccess` | `.wait` | out-of-bounds, invalid pointer |
| `GpuStackOverflow` | `.wait` | kernel recursion too deep |
| `GpuTimeout` | `.wait` | OS watchdog killed the kernel |
| `GpuDeviceLost` | `.wait` | device reset or crash |

### `after =` and error propagation

If a dependency handle failed, the dependent kernel is not launched — the error propagates:

```boring
let h1 = kernel(block = 256) ka
let h2 = kernel(block = 256, after = h1) kb    # not launched if h1 failed
h2.wait                                         # raises h1's error
```

---

## Multi-device

```boring
let g0 = GPU(0)
let g1 = GPU(1)

mut ka = new(g0) Scale(n)
mut kb = new(g1) Scale(n)

mut ka = kernel(block = 256) ka |> .wait
mut kb = kernel(block = 256) kb |> .wait
```

`after =` for cross-device ordering uses the same syntax as single-device. Same-device ordering is implemented via CUDA events.

---

## Known limitations

The following features are not yet implemented:

- `atomic.cas`, `atomicMin`, `atomicMax`, `atomicExch` — no `atomic` namespace or `.cas` method in lexer, parser, interpreter, or transpiler
- `'actor'unified`, `'actor'shared` — parse error; only `'actor'global` is accepted
- `warp.size`, `warp.lane`, `warp.sync` — no `warp` namespace inside kernel bodies
- dtod inference (device-to-device auto copy) — `ka.output` passed to another kernel always triggers D2H + H2D
- Cross-device `after =` (peer access) — `after =` codegen handles same-device dependencies only; no `cudaStreamWaitEvent` cross-device path, no device-mismatch detection
- `--peer-access` CLI flag — not registered in `src/main.rs`
- `gpu.const(...)` callable builtin — `GpuConst` is a field-qualifier enum variant only, not a callable
- Kernel qualifier rejection (`'shared`/`'actor` binding) — `kernel(...)` does not inspect the qualifier of the kernel struct value; `'shared`/`'actor`/`'guard`-qualified instances are not rejected at compile time
- Must-use `KernelHandle` — no `#[must_use]` attribute on the generated `KernelHandle<T>`; dropping a handle without calling `.wait` compiles silently
- Block size validation at compile time — oversized `block` values are not checked in `src/validator/kernel.rs` or `src/interpreter/eval_gpu.rs`
- Error types (`GpuLaunchError`, `GpuOutOfMemory`, `GpuIllegalAccess`, `GpuStackOverflow`, `GpuTimeout`, `GpuDeviceLost`) — none of these exist in the codebase
- `.shape`-based grid inference — no built-in `Image`/`Volume` types; omitting `grid` when no 1D array field is present silently defaults to `(1, 1, 1)`
