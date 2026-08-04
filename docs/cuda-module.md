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
- **Kernel context** — fields inside a `kernel` struct. The `'gpu` prefix is dropped; short forms `'unified`, `'global`, bare `'actor`, `'local`, `'const` are used.

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
| bare `'actor` | block SRAM (`__shared__`) | no | no |
| `'global` | device-only DRAM | via `gpu.copy()` | no |
| `'unified` | unified DRAM (`cudaMallocManaged`) | direct | no |
| `'const` | constant cache | no | implicit via `let` |
| `'actor'global` | device-only DRAM, atomic access | via `gpu.copy()` | no |
| `'actor'unified` | unified DRAM (`cudaMallocManaged`), atomic access | direct | no |

---

## Kernel struct field rules

| Field declaration | Memory space | Host access | CUDA mapping |
|---|---|---|---|
| `let [float]'unified input` | unified DRAM, device read-only | direct | `const float* const input` |
| `mut [float]'unified output` | unified DRAM, device writable | direct | `float* const output` |
| `let [float]'global input` | device-only DRAM, device read-only | via `gpu.copy()` | `const float* const input` |
| `mut [float]'global output` | device-only DRAM, device writable | via `gpu.copy()` | `float* const output` |
| `var [float]'global buf` | device-only DRAM, ptr + data mutable | via `gpu.copy()` | `float* buf` |
| `mut [float, N]'actor tile` | block SRAM, static | no | `__shared__ float tile[N]` |
| `mut [float]'actor tile` | block SRAM, dynamic | no | `extern __shared__ float tile[]` |
| `let float sigma` | `'const` implicit — scalar | no | `const float sigma` |
| `var int i` | `'local` implicit — register | no | `int i` |
| `mut [float, N] tile` | `'local` implicit — thread-local | no | `float tile[N]` |
| `mut [int]'actor'global bins` | device DRAM, atomic access | via `gpu.copy()` | `int64_t* bins` |
| `mut [int]'actor'unified bins` | unified DRAM, atomic access | direct | `int64_t* bins` |
| CPU qualifiers (`'heap`, `'shared`…) | — | — | compile-time error |

---

## CUDA C mapping

| Boring | CUDA C |
|---|---|
| `gpu.thread.x` | `threadIdx.x` |
| `gpu.block.x` | `blockIdx.x` |
| `gpu.block_dim.x` | `blockDim.x` |
| `gpu.grid_dim.x` | `gridDim.x` |
| `sync` | `__syncthreads()` |
| `gpu.warp.size` | `warpSize` |
| `gpu.warp.lane` | `threadIdx.x/y/z` linearized, `% warpSize` |
| `gpu.warp.sync()` | `__syncwarp(0xffffffff)` |
| `gpu.warp.shuffle_down(v, delta)` | `__shfl_down_sync(0xffffffff, v, delta)` |
| `gpu.warp.shuffle_up(v, delta)` | `__shfl_up_sync(0xffffffff, v, delta)` |
| `gpu.warp.shuffle_xor(v, mask)` | `__shfl_xor_sync(0xffffffff, v, mask)` |
| `gpu.warp.shuffle(v, lane)` | `__shfl_sync(0xffffffff, v, lane)` |
| `'unified` field | `cudaMallocManaged` |
| `'global` field | `cudaMalloc` |
| bare `'actor` field | `__shared__` |
| `'const` scalar field | `__constant__ T name;` (file scope) |
| `'const` fixed array field (`[T, N]`) | `__constant__ T name[N];` (file scope) |
| atomic `[i] +=` on `'actor'global`/`'actor'unified` | `atomicAdd` |
| `[i].min(v)` on `'actor'global`/`'actor'unified` | `atomicMin(&x, v)` |
| `[i].max(v)` on `'actor'global`/`'actor'unified` | `atomicMax(&x, v)` |
| `[i].swap(v)` on `'actor'global`/`'actor'unified` | `atomicExch(&x, v)` |
| `[i].cas(expected, new)` on `'actor'global`/`'actor'unified` | `atomicCAS(&x, expected, new)` |
| `print` in kernel | `printf` |

`gpu.warp.*`'s `_sync` intrinsics always pass the full `0xffffffff` active-lane
mask — Boring doesn't expose a mask parameter in source. This means
`gpu.warp.*` inside a divergent branch (an `if` not every lane in the warp
takes) is only as safe as passing a full mask to CUDA's `_sync` intrinsics
actually is: correct for reconverged/uniform control flow, undefined behavior
otherwise. See [warp-level primitives](warp-level-primitives.html).

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
- Fixed-shape `[T, width=W, height=H]` labeled-array field (`'unified`/`'global`/`'actor'global`/`'actor'unified`) → `grid = (ceil(W/block.x), ceil(H/block.y), 1)`, generalized to 3 axes
- Otherwise, if `grid` is omitted, the transpiler defaults to `grid = (1, 1, 1)` — always pass `grid` explicitly unless relying on array/labeled-array inference.

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

## Bare `'actor` — block SRAM

Bare `'actor` fields (formerly spelled `'sync`) are allocated in per-block shared memory (`__shared__` in CUDA C). The transpiler computes `shared_mem_bytes` automatically from the block dimension and element size — no `smem =` dispatch parameter needed.

### Fixed size — `[T, N]'actor`

```boring
kernel Reduce:
    let [float]'unified      input
    mut [float]'unified      output
    mut [float, 256]'actor   tile

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

### Dynamic size — `[T]'actor`

The size is `block_dim.x * sizeof(T)` — one element per thread in the block. Declare the field without a size; the transpiler passes `block_dim.0 * sizeof(T)` as `shared_mem_bytes`.

```boring
kernel Reduce:
    let [float]'unified  input
    mut [float]'unified  output
    mut [float]'actor    tile

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

## Atomics — `'actor'global`/`'actor'unified` on kernel fields

Both are implemented. Bare `'actor` (block SRAM, above) is a separate qualifier and is
not an alias for either.

| Qualifier | Location | Atomic scope |
|---|---|---|
| `mut [int]'actor'global bins` | device DRAM | all threads, all blocks |
| `mut [int]'actor'unified bins` | unified DRAM (host + device) | all threads, all blocks |

**Operations transpiled automatically on `'actor'global`/`'actor'unified` fields:**

| Boring | CUDA |
|---|---|
| `x += v` | `atomicAdd(&x, v)` |
| `x -= v` | `atomicSub(&x, v)` |
| `x \|= v` | `atomicOr(&x, v)` |
| `x &= v` | `atomicAnd(&x, v)` |
| `x ^= v` | `atomicXor(&x, v)` |
| `x.min(v)` | `atomicMin(&x, v)` |
| `x.max(v)` | `atomicMax(&x, v)` |
| `x.swap(v)` | `atomicExch(&x, v)` |
| `x.cas(expected, new)` | `atomicCAS(&x, expected, new)` |

`atomicAdd` etc. take a plain pointer of the base type — no special "atomic" type
exists in CUDA C, so the same intrinsics apply unconditionally whether the pointer
came from `cudaMalloc` (`'actor'global`) or `cudaMallocManaged` (`'actor'unified`).

`min`/`max`/`swap`/`cas` are **methods**, not compound-assign operators — min/max/
exchange/compare-and-swap have no natural infix operator the way add/sub/or/and/xor
do — and unlike the operators above, they're handled in **expression** position
(they return the previous value, matching `atomicMin`/`atomicMax`/`atomicExch`/
`atomicCAS`'s own real CUDA semantics) rather than as a statement-only
compound-assign desugar. The checker requires that return value be explicitly bound
or discarded (`_ = bins[bucket].swap(0)`), same as any other non-void call used as a
bare statement.

```boring
kernel Histogram:
    let [float]'global     input
    mut [int]'actor'global bins

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        if i < len(input):
            let bucket = int(input[i] * 10.0)
            bins[bucket] += 1              # → atomicAdd
            _ = bins[bucket].min(100)      # → atomicMin
            let old = bins[bucket].cas(0, 1)  # → atomicCAS, old holds the previous value
```

`'actor'unified` is identical except `bins` is also directly readable from the host
(no `gpu.copy()` needed) via the generated `read_bins()` accessor.

**`.min`/`.max`/`.swap`/`.cas` also work on a plain, non-`'actor'` field** — matching
`+= -= &= |= ^=`, which already degrade to ordinary (non-atomic) arithmetic off a
non-actor field rather than erroring. There's no intrinsic to lean on for the
"return the previous value" contract in that case, so it's bridged via a GNU/Clang
statement-expression (`({ ... })` — `nvcc`'s device-code compiler accepts this GNU C
extension):

```boring
kernel Scale:
    mut [int]'unified buf   # plain, not 'actor'global/'actor'unified

    def ():
        let old = buf[i].min(v)   # ({ auto __old = buf[i]; buf[i] = min(buf[i], v); __old; })
                                   # plain read-modify-write, no cross-thread protection
```

---

## CPU simulation — substitution table

`boring run` executes kernels sequentially on the CPU. No source changes required.

| GPU primitive | CPU simulation |
|---|---|
| `'unified` fields (kernel) | heap allocation (`Vec<T>`) |
| `'global` fields (kernel) | heap allocation (`Vec<T>`) |
| `init` allocation | `Vec<T>::with_capacity(n)` |
| bare `'actor` (block SRAM) | local array (stack or heap) |
| `'local` (register) | local variable |
| `'const` (kernel) | `let` binding |
| kernel launch | sequential loop over all threads |
| `gpu.thread.x/y/z`, `gpu.block.x/y/z`… | loop variables |
| `sync` | no-op |
| `'actor'global`/`'actor'unified` fields | plain arithmetic (no contention) |
| `kernel:` block dispatch | no-op — single-threaded sequential |
| `after =` | sequential execution in declaration order |

Sequential simulation hides data races. A kernel correct in simulation may be incorrect on real hardware if `sync` barriers are missing.

---

## Error handling

The built-in [`GpuError`](gpu-module.html#gpu-error-handling) enum exists (`LaunchError`/`OutOfMemory`/`IllegalAccess`/`StackOverflow`/`Timeout`/`DeviceLost`), but **isn't catchable by variant on this backend** — `catch GpuError.OutOfMemory:` needs the `BoringError`-downcast machinery, which lives in the general transpiler pipeline this backend's own small, kernel-only host transpiler doesn't share (same prerequisite gap `scoped-access-blocks.md` documents for `with`). What CUDA does do: `__boring_cuda_classify_error` inspects the real `cudarc::driver::DriverError`'s underlying `CUresult` code at kernel launch and both stream-sync points, and prefixes cudarc's own message (already real — it calls `cuGetErrorName`/`cuGetErrorString` internally) with a short classified category (`"GPU out of memory: ..."`, `"GPU illegal memory access: ..."`, etc.) — still a plain `Box<dyn std::error::Error + Send + Sync>`, not a `GpuError` a `catch`/`match` could branch on, just a more informative message than the bare cudarc error used to be on its own.

**Block size is validated here, at runtime, by design — not at compile time.** An oversized `block =` (exceeding the device's, or this specific kernel's, max threads per block) makes `cuLaunchKernel` itself reject the launch with `CUDA_ERROR_INVALID_VALUE`, classified as `"GPU launch configuration invalid (e.g. block size exceeds device limits)"`. Neither `src/validator/kernel.rs` nor the interpreter (`src/interpreter/eval_gpu.rs`) duplicates that check — there's no hardcoded "max threads per block" constant to keep in sync with real hardware (which varies by compute capability) or to silently drift out of date; the real launch call is always the ground truth, and now surfaces a real, classified error instead of the caller needing to guess why a dispatch failed.

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

- `'actor'shared` — parse error; only `'actor'global`/`'actor'unified` (or bare `'actor` for block-shared memory) are accepted
- Omitting `grid` when no 1D array field, `Image`, or `Volume` field is present silently defaults to `(1, 1, 1)` — see the grid-inference rules above for what fields DO get 2D/3D inference
