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

**Multi-file projects**: `use <file>.br` in the entry file is resolved and
inlined before transpilation — first relative to the importing file's own
directory, then against each path in the `BORING_PATH` environment variable
(same search order as `boring run`). Circular and duplicate imports are
merged once. A `use` that doesn't resolve to a `.br` file on disk (e.g.
`use std.collections`) is left as an ordinary import for the general
transpiler to handle.

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

`'const` has no host-context form — it has no host access at all (see below), so a
host-side binding could never be read from or written to. It's only meaningful as a
kernel-struct field qualifier.

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

Passing `grid` explicitly always overrides inference, on every backend — `k(block = (16, 16, 1), grid = (4, 4, 1))` dispatches exactly the requested `(4, 4, 1)` grid, whether or not the kernel has a 1D auto-grid-capable field. (An earlier version of the CUDA/ROCm codegen silently dropped an explicit `grid` argument and always substituted the inferred/default value instead — confirmed via a real generated project, fixed by threading the caller's `grid` through to `__boring_launch` the same way `metal::host` already did.)

**Ownership qualifier restriction:** a kernel struct instance dispatched via `kernel:` must be declared with no wrapping ownership qualifier — `'shared`/`'actor`(`'task`)/`'guard`(`'task`) are rejected at compile time (semantic checker), since `kernel:` dispatch needs direct, exclusive ownership on the host side, not an `Rc`/`Arc`/`RefCell`/`Mutex`/`RwLock` handle.

```boring
let k'shared = Scale(data)   # error: cannot dispatch `k` via `kernel:` — it is `'shared`-qualified
kernel:
    k(block = 256)
```

---

## `KernelHandle`

`KernelHandle<T>` is a real type in the generated Rust (`__boring_launch` returns one), but it's an internal implementation detail — Boring source never names it or calls a method on it directly. A `kernel:` block's dispatch (`k(block = ...)`) desugars to launching through the handle and immediately unwrapping it back into `k` behind the scenes:

```boring
mut k = Scale(data)
kernel:
    k(block = 256)   # transpiles to roughly `k = k.__boring_launch(...)?.inner;`
```

There is no Boring-level syntax for holding onto a `KernelHandle` and deciding later whether/when to wait on it — every dispatch inside a `kernel:` block is synchronized (or, on Metal, deferred to the next host-side read — see `docs/metal-backend.md`'s "Error handling") before the next statement runs.

The generated `KernelHandle<T>` struct itself carries `#[must_use]`, so if hand-written or future-generated Rust ever drops one without calling `.wait` (or otherwise consuming `.inner`), `rustc` emits a warning instead of silently discarding it — the current transpiler never does this, but the attribute guards against a future codegen change introducing that path unnoticed.

---

## Device-to-device chaining

Feeding one kernel's output directly into another kernel's constructor (`Scale(k1.buf)`) never round-trips through the host — the buffer is copied device-to-device via `CudaSlice::clone()` (`clone_dtod` under the hood), a real GPU-to-GPU memcpy:

```boring
mut k1 = Scale(data)
kernel:
    k1(block = 256)
mut k2 = Scale(k1.buf)   # k1.buf.clone() -- a real D2D copy, not D2H+H2D
kernel:
    k1(block = 256)      # k1 is still independently usable
    k2(block = 256)
```

This is a `.clone()`, not a move: `k1` stays fully usable afterward, including dispatching it again — the earlier version of this optimization moved the buffer unconditionally, which compiled fine right up until the source kernel was used again, at which point it was a real `E0382` ("use of partially moved value"). `.clone()` is correct in every case and still far cheaper than a full host round trip.

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
| `'const` (kernel) | `let` binding |
| kernel launch | sequential loop over all threads |
| `gpu.thread.x/y/z`, `gpu.block.x/y/z`… | loop variables |
| `sync` | no-op |
| `'actor'global` fields | plain arithmetic (no contention) |
| `kernel:` block dispatch | no-op — single-threaded sequential |
| `after =` | sequential execution in declaration order |

Sequential simulation hides data races. A kernel correct in simulation may be incorrect on real hardware if `sync` barriers are missing.

---

## Error handling

There is no dedicated Boring-level named error type (`GpuLaunchError`/`GpuOutOfMemory`/`GpuIllegalAccess`/`GpuStackOverflow`/`GpuTimeout`/`GpuDeviceLost`, as earlier drafts of this section described) anywhere in the codebase — see "Known limitations" below. Every kernel-dispatch error surfaces as whatever cudarc itself reports (`cudarc::driver::DriverError`), wrapped generically as `Box<dyn std::error::Error + Send + Sync>` and propagated with `?` up to `boring_main()`'s own `Result` — a plain, un-typed error, not a Boring-specific enum a `match` could branch on.

That said, cudarc's own two observation points still apply, mirroring the two phases a real CUDA launch can fail at:

### Synchronous errors — raised at launch

An invalid launch config (block/grid size, etc.) is rejected by `cudaLaunchKernel` itself, before the kernel ever runs:

```boring
kernel:
    k(block = 99999)   # a bad config surfaces via `?` right here, before k runs at all
```

### Asynchronous errors — raised at synchronization

A fault during execution (out-of-bounds access, device reset, ...) is only detected when the stream is synchronized — which every dispatch inside a `kernel:` block already does before the next statement runs, so it surfaces at the same call site as a launch-config error, just one GPU round-trip later:

```boring
kernel:
    k(block = 256)   # if k's own kernel body faults, that error also surfaces here
```

### `after =` and error propagation

If an earlier `kernel:` block statement's dispatch returns an error, `?` exits the enclosing function immediately — a later statement in the same block (including one depending on the failed kernel via `after =`) is simply never reached:

```boring
kernel:
    ka(block = 256)                  # if this fails, execution never reaches the next line
    kb(block = 256, after = ka)
```

---

## Multi-device

```boring
let g0 = GPU(0)
let g1 = GPU(1)

mut ka = new(g0) Scale(n)
mut kb = new(g1) Scale(n)

kernel:
    ka(block = 256)
    kb(block = 256)
```

`after =` for cross-device ordering uses the same syntax as single-device — same-device ordering is implemented via CUDA events (`stream.join`, i.e. `cuStreamWaitEvent`), which also works cross-device, but only once peer access is enabled between the two GPUs involved.

There is no `--peer-access` flag to opt into this — every GPU context created via `GPU(n)`/`new(g) ...` automatically attempts bidirectional peer access with every other context the program has already created, checking real hardware capability first (`cuDeviceCanAccessPeer`) and silently skipping any pair the topology doesn't support (no NVLink/shared PCIe root, etc.) — that pair's own `after =` then surfaces its own real `DriverError` at the `cuStreamWaitEvent` call site instead of silently doing the wrong thing. This is cheap to always attempt: a single-GPU program only ever registers one context, so the check loop never runs.

---

## Known limitations

The following features are not yet implemented:

- `atomic.cas`, `atomicMin`, `atomicMax`, `atomicExch` — no `atomic` namespace or `.cas` method in lexer, parser, interpreter, or transpiler
- `'actor'unified`, `'actor'shared` — parse error; only `'actor'global` is accepted
- `warp.size`, `warp.lane`, `warp.sync` — no `warp` namespace inside kernel bodies
- Block size validation at compile time — oversized `block` values are not checked in `src/validator/kernel.rs` or `src/interpreter/eval_gpu.rs`
- Error types (`GpuLaunchError`, `GpuOutOfMemory`, `GpuIllegalAccess`, `GpuStackOverflow`, `GpuTimeout`, `GpuDeviceLost`) — none of these exist in the codebase
- `.shape`-based grid inference — no built-in `Image`/`Volume` types; omitting `grid` when no 1D array field is present silently defaults to `(1, 1, 1)`
