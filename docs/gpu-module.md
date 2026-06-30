# GPU / CUDA — full reference

Boring supports GPU computing through `kernel` structs — a dedicated declaration form that groups device memory fields, an `init` allocator, device-side helpers, and an anonymous entry point (`def ()`).

For the Metal backend (macOS), see [`metal-backend.md`](metal-backend.html).

---

## Quick start

```boring
kernel Scale:
    mut [float]'unified buf     # unified host+device DRAM

    init([float]'unified data):
        buf = data

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] *= 2.0

mut k = Scale(data)                          # instantiate — init called
mut k = kernel(block = 256) k |> .wait      # launch → wait → get result back
print k.buf[0]
```

---

## `kernel` struct

```boring
kernel Name:
    <binding> [<type>]['<qualifier>] <field>  # qualifier optional for scalars and fixed arrays
    ...

    init(<params>):
        <body>                               # allocate / initialise fields

    def <helper>(<params>):                  # device-side helper method
        <body>

    def ():                                  # entry point — invoked once per thread
        <body>
```

`let`/`mut`/`var` are mandatory on every field. GPU memory qualifiers (`'unified`, `'global`, `'shared`, `'local`, `'const`) replace the usual ownership qualifiers inside a `kernel` struct. For scalars and fixed-size arrays the qualifier may be omitted — the transpiler infers it from the binding (see [Qualifier inference](#qualifier-inference) below).

---

## GPU memory qualifiers

**Kernel-context** (inside `kernel` struct fields):

| Qualifier | Memory space | Host access |
|---|---|---|
| `'unified` | unified DRAM (host + device) | direct |
| `'global` | device-only DRAM | via `gpu.copy()` |
| `'shared` | block SRAM (`__shared__`) | no |
| `'local` | registers / thread-local | no — default |
| `'const` | constant cache | no |

### Qualifier inference

Inside a `kernel` struct, the qualifier may be omitted for scalars and fixed-size arrays. The transpiler infers it from the binding keyword:

| Declaration | Inferred qualifier | Rule |
|---|---|---|
| `let float alpha` | `'const` | scalar `let` → constant cache |
| `mut float acc` | `'local` | scalar `mut`/`var` → thread-private register |
| `var int step` | `'local` | idem |
| `let [float, 4] lut` | `'const` | fixed array `let` → constant cache |
| `mut [float, 8] tile` | `'local` | fixed array `mut`/`var` → thread-private |
| `var [int, 4] buf` | `'local` | idem |

Explicit qualifiers remain valid (`let float'const alpha` is equivalent to `let float alpha`; `let [float, 4]'const lut` is equivalent to `let [float, 4] lut`). Dynamic arrays (`[T]`) must still carry an explicit qualifier.

**Invalid combinations** — parse errors:

| Combination | Error |
|---|---|
| `[T]'local` | Dynamic arrays cannot be thread-local on GPU |
| `[T]'const` | Constant cache requires a compile-time size — use `[T, N]` |
| `[T, N]'unified` | Fixed arrays cannot use `'unified` (size implicit from init) |
| `[T, N]'global` | Fixed arrays cannot use `'global` |

**Valid qualifier × field-type matrix:**

| | `'unified` | `'global` | `'shared` | `'local` | `'const` |
|---|---|---|---|---|---|
| `[T]` dynamic | ✅ explicit | ✅ explicit | ✅ explicit | ❌ error | ❌ error |
| `[T, N]` fixed | ❌ error | ❌ error | ✅ explicit | ✅ inferred (`mut`/`var`) | ✅ inferred (`let`) |
| scalar | — | — | — | ✅ inferred (`mut`/`var`) | ✅ inferred (`let`) |

> **Key point:** `'const` and `'local` are never written in practice — they are always inferred. `'unified`, `'global`, and `'shared` are always explicit (semantic choice the transpiler cannot infer).

---

**Host-context** (bindings outside `kernel` struct):

| Qualifier | Location |
|---|---|
| `'gpu'unified` | unified host + device DRAM |
| `'gpu'global` | device-only DRAM |
| `'gpu'const` | GPU constant cache |

---

## Launch expression

```boring
kernel(block = N) k                         # 1D, N threads per block, 1 block
kernel(block = N, grid = M) k               # 1D, N threads × M blocks
kernel(block = (16, 16)) k                  # 2D — grid inferred from kernel shape
kernel(block = 256, after = h1) k           # ordered after h1
kernel(block = 256, smem = {tile = 4096}) k # named dynamic shared-memory partitions
kernel(block = 256, priority = high) k      # scheduling priority: high, normal, low
```

Returns a `KernelHandle`:

```boring
struct KernelHandle<T>:
    req bool done()    # true if completed (always true in simulation)
    req T    wait()    # block until complete, return kernel object
```

Common pattern with the pipe operator:

```boring
mut k = kernel(block = 256) k |> .wait     # launch and wait in one expression
```

---

## Execution context built-ins

Inside `def ()` and device helpers, `gpu` is available:

| Built-in | CUDA equivalent |
|---|---|
| `gpu.thread.x/y/z` | `threadIdx.x/y/z` |
| `gpu.block.x/y/z` | `blockIdx.x/y/z` |
| `gpu.block_dim.x/y/z` | `blockDim.x/y/z` |
| `gpu.grid_dim.x/y/z` | `gridDim.x/y/z` |
| `sync` | `__syncthreads()` |

---

## `GPU` type

`GPU` is a built-in type for device selection and property queries:

```boring
let g = GPU(0)
print "Device: {g.name()} — {g.totalMem() / 1_073_741_824} GB"
print "SM {g.computeCapability()[0]}.{g.computeCapability()[1]}, warp {g.warpSize()}"

for g in GPU.all():
    print "[{g.index()}] {g.name()} — {g.freeMem()} bytes free"
```

| Method | Returns |
|---|---|
| `name()` | device model name |
| `totalMem()` | total VRAM in bytes |
| `freeMem()` | available VRAM in bytes |
| `computeCapability()` | `[major, minor]` |
| `warpSize()` | threads per warp |
| `maxThreads()` | max threads per block |
| `maxSharedMem()` | max shared memory per block (bytes) |
| `index()` | device index |

---

## Memory safety model

### `'unified` — zero-copy host/device

`'unified` fields use `cudaMallocManaged`. Host and device access the same physical memory; no explicit `H2D`/`D2H` copy is needed. Concurrent access from both sides during kernel execution is undefined behaviour — guard with `wait()` before reading on the host.

### `'global` — device-only

`'global` fields live in device DRAM. Host reads/writes require explicit `gpu.copy()`:

```boring
gpu.copy(k.result, host_buf)    # D2H — copies device field to host array
gpu.copy(host_buf, k.input)     # H2D — copies host array to device field
```

### `'shared` — block SRAM

`'shared` fields are allocated in per-block shared memory (`__shared__`). Use `sync` between producer and consumer threads in the same block.

```boring
kernel Reduce:
    mut [float]'unified input
    mut float'unified   result
    mut [float]'shared  tile    # block SRAM — size set via launch `smem = {tile = bytes}`

    def ():
        let tid = gpu.thread.x
        tile[tid] = input[gpu.block.x * gpu.block_dim.x + tid]
        sync
        if tid == 0:
            var float sum = 0.0
            for v in tile:
                sum += v
            result = sum
```

---

## Multi-device

> **Not yet implemented.** The `GPU` type can enumerate and query devices (see below), but the launch expression has no `device =` parameter — there is currently no way to dispatch a `kernel(...)` launch to a specific GPU. All launches run on the default device.

### `after =` ordering

```boring
let h0 = kernel(block = 256) k0
let h1 = kernel(block = 256, after = h0) k1   # k1 starts after k0 completes
```

`after =` accepts a single handle or a list: `after = [h0, h1]`.

---

## Atomics

Tag a field with `'actor'global` to enable atomic operations. Atomic codegen only applies to **indexed** access into an `'actor'global` array field — a bare scalar field compound-assign is emitted as a regular, non-atomic read-modify-write:

```boring
kernel Histogram:
    mut [int]'actor'global counts = [0, 0, 0, 0]

    def ():
        counts[bucket] += 1     # compiled to atomicAdd
```

Supported atomic operations: `+= -= &= |= ^=` (compiled to `atomicAdd` / `atomicSub` / `atomicAnd` / `atomicOr` / `atomicXor`). `*=`, `/=`, and `%=` are not supported as atomics. There is no `swap()` or `compareSwap()` method.

---

## Simulation mode

`boring run` executes kernels sequentially on the CPU — each thread's entry point runs in order, `sync` is a no-op, `gpu.thread.x` is the loop index. The same source file works without a GPU, enabling unit tests and CI.

```sh
boring run main.br                  # default simulation profile
boring run --gpu a100 main.br       # simulate A100 device properties
boring run --gpu h100 main.br       # simulate H100 device properties
boring run --gpu path/to/my.toml main.br   # custom profile
```

### Built-in profiles

| Name | GPU | VRAM | SM |
|---|---|---|---|
| `default` | generic | 8 GB | 8.6 |
| `v100` | Tesla V100 SXM2 | 16 GB | 7.0 |
| `a100` | A100 SXM4 | 80 GB | 8.0 |
| `rtx3090` | RTX 3090 | 24 GB | 8.6 |
| `rtx4090` | RTX 4090 | 24 GB | 8.9 |
| `h100` | H100 SXM5 | 80 GB | 9.0 |

### Custom TOML profile

```toml
name = "My GPU"
totalMem = 8589934592   # bytes
warpSize = 32
maxThreads = 1024
maxSharedMem = 49152
computeCapability = [8, 6]
```

> **Note:** sequential simulation hides data races between threads. A kernel correct in simulation may be incorrect on real hardware if `sync` barriers are missing.

---

## CUDA codegen

`boring build --target cuda` generates a Cargo project with:

- `src/main.rs` — Rust host code using the `cudarc` crate
- `kernels/main.cu` — CUDA C device code
- `build.rs` — invokes `nvcc` to compile `.cu` → PTX
- `Cargo.toml`

Requires the CUDA toolkit (`nvcc`) and a CUDA-capable GPU.

### CUDA C mapping

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
| atomic `[i] += ` on `'actor'global` | `atomicAdd` |
| `print` in kernel | `printf` |

#### `[T, N]'const` codegen detail

A `let [T, N]` field (inferred `'const`) is emitted as a file-scope `__constant__` array — **not** as a kernel parameter. Device code accesses it directly by name:

```boring
kernel Lookup:
    let [float, 4] lut = [0.0, 0.25, 0.5, 1.0]   # inferred 'const
    mut [float]'unified output

    def ():
        let i = gpu.thread.x
        output[i] = lut[i % 4]
```

Generated CUDA C (excerpt):

```cuda
__constant__ float lut[4];   // file scope — not a kernel parameter

__global__ void lookup(float* output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    output[i] = lut[i % 4];
}
```

The constant array is uploaded once via `cudaMemcpyToSymbol` before launch and remains cached across all threads in the block.
