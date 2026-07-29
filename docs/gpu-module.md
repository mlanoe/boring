# GPU computing — language reference

Boring supports GPU computing through `kernel` structs — a dedicated declaration form that groups device memory fields, an `init` allocator, device-side helpers, and an anonymous entry point (`def ()`).

This document covers the **language syntax and semantics** shared by all backends. For backend-specific codegen details, see:

- [`cuda-module.md`](cuda-module.html) — CUDA C mapping, generated project layout, `cudarc` host API, PTX compilation
- [`rocm-backend.md`](rocm-backend.html) — HIP C++ mapping (near-identical to CUDA C), hand-rolled HIP FFI host API, AMD GPU support
- [`metal-backend.md`](metal-backend.html) — MSL address space mapping, Metal runtime compilation
- [`wgpu-backend.md`](wgpu-backend.html) — WGSL mapping, pipeline overrides, cross-platform GPU support

---

## Quick start

```boring
kernel Scale:
    mut [float]'unified buf

    init([float]'unified data):
        buf = data

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] *= 2.0

mut k = Scale(data)
kernel:
    k(block = 256)

print k.buf[0]
```

This pattern — construct, dispatch, read back — is not limited to top-level
code: it works the same way inside an ordinary function body, mixed freely
with regular Boring control flow, string interpolation, and other function
calls.

```boring
def [float] scaleAll([float] data):
    mut k = Scale(data)
    kernel:
        k(block = 256)
    k.buf
```

---

## `kernel` struct

```boring
kernel Name:
    <binding> [<type>]['<qualifier>] <field>

    init(<params>):
        <body>

    def <helper>(<params>):
        <body>

    def ():
        <body>
```

`let`/`mut`/`var` are mandatory on every field. GPU memory qualifiers (`'unified`, `'global`, `'sync`, `'local`, `'const`) replace the usual ownership qualifiers inside a `kernel` struct. For scalars and fixed-size arrays the qualifier may be omitted — the transpiler infers it from the binding (see [Qualifier inference](#qualifier-inference) below).

---

## Generic kernel declarations

Kernel structs support generic parameters with the same syntax as regular structs.

### Const generics — compile-time array sizes

```boring
kernel Blur<int N>:
    mut [float, N] kernel_weights
    mut float result

    def ():
        result = kernel_weights[0]

kernel GameOfLife<int W, int H>:
    mut [bool, W * H]'unified cells

    def ():
        let i = gpu.thread.x
        cells[i] = not cells[i]
```

- `int`, `uint`, `float`, and `bool` are valid const generic types — the type comes before the name, consistent with all variable declarations.
- Array sizes may be **const-evaluable expressions** involving multiple params (`W * H`, `N + 1`, etc.).
- All scalar types are accepted: `<int N>`, `<uint N>`, `<float Alpha>`, `<bool Flag>`.

### Type generics and trait bounds

```boring
kernel MonKernel<A, B>:
    ...

kernel ConstrainedKernel<A as Displayable>:
    ...
```

Type generics follow the same rules as struct generics. See [Section 13 — Generics](book.html#13-generics) for the full generic syntax.

### Instantiation

```boring
let gol   = GameOfLife<64, 64>()
let small = Blur<3>()
let large = Blur<7>()
```

Multiple instantiations of the same generic kernel generate **distinct** code objects per backend — for example, `Blur<3>` and `Blur<7>` produce separate WGSL entry points `Blur_3_main` and `Blur_7_main` when building with `--target wgpu`.

---

## GPU memory qualifiers

**Kernel-context** (inside `kernel` struct fields):

| Qualifier | Memory space | Host access |
|---|---|---|
| `'unified` | unified DRAM (host + device) | direct |
| `'global` | device-only DRAM | via `gpu.copy()` |
| `'surface` | unified DRAM, 32-bit pixels | direct (CUDA); blit-only (Metal / wgpu) — use `screen.present()` |
| `'sync` | block SRAM (`__shared__` / `threadgroup` / `var<workgroup>`) | no |
| `'local` | registers / thread-local | no — default |
| `'const` | constant cache | no |
| `'actor'global` (or bare `'actor`) | device-only DRAM, atomic access | via `gpu.copy()` — see [Atomics](#atomics) |

`'surface` is restricted to `[uint]` fields and is intended for pixel buffers
presented to a `Screen`. See [`gpu-display.md`](gpu-display.html).

`'actor` alone is an alias for `'actor'global` — atomics are only implemented for
device-global memory, so there's no other qualifier it could mean.

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

Explicit qualifiers remain valid (`let float'const alpha` is equivalent to `let float alpha`). Dynamic arrays (`[T]`) must still carry an explicit qualifier.

**Invalid combinations** — parse errors:

| Combination | Error |
|---|---|
| `[T]'local` | Dynamic arrays cannot be thread-local on GPU |
| `[T]'const` | Constant cache requires a compile-time size — use `[T, N]` |
| `[T, N]'unified` | Fixed arrays cannot use `'unified` |
| `[T, N]'global` | Fixed arrays cannot use `'global` |

**Valid qualifier × field-type matrix:**

| | `'unified` | `'global` | `'sync` | `'local` | `'const` |
|---|---|---|---|---|---|
| `[T]` dynamic | explicit | explicit | explicit | error | error |
| `[T, N]` fixed | error | error | explicit | inferred (`mut`/`var`) | inferred (`let`) |
| scalar | — | — | — | inferred (`mut`/`var`) | inferred (`let`) |

> `'const` and `'local` are always inferred and rarely written explicitly. `'unified`, `'global`, and `'sync` are always explicit.

---

**Host-context** (bindings outside `kernel` struct):

| Qualifier | Location |
|---|---|
| `'gpu'unified` | unified host + device DRAM |
| `'gpu'global` | device-only DRAM |

`'const` (like `'sync` and `'local`) has no host-context form — it has no host access at
all (see the table above), so a host-side binding could never be read from or written to.
It's only meaningful as a field qualifier inside a `kernel` struct.

---

## `kernel:` block

GPU kernels are dispatched inside a `kernel:` block. The block is **synchronous**
— execution resumes after the closing line, and all kernel fields are accessible
directly.

```boring
mut k = Scale(data)
kernel:
    k(block = 256)

print k.buf[0]    # safe: kernel: block has completed
```

### Dispatch call

```boring
k(block = N)               # 1D block of N threads
k(block = (16, 16))        # 2D block of 16×16 threads
k(block = (8, 8, 4))       # 3D block
k(block = N, grid = M)     # explicit grid of M blocks
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `block` | `int` or `(int, int)` or `(int, int, int)` | yes | threads per block |
| `grid` | `int` or tuple | no | blocks per grid — inferred from field length if omitted |
| `after` | kernel var or `[k1, k2, ...]` | no | ordering: this dispatch starts after the listed ones complete (GPU-side on CUDA, submission-ordered on wgpu, sequential on Metal and `boring run`) |
| `priority` | `"high"` / `"normal"` / `"low"` | no | scheduling priority — CUDA only; ignored on all other backends |

When `grid` is omitted it is inferred:
- **1D**: `ceil(n / block)` where `n` is the length of the first array field.
- **2D** (kernel has a `Dimension` field alongside a `'surface` field): `(ceil(w/bx), ceil(h/by), 1)`.

### Multi-pass with `after =`

`after =` declares that a dispatch must start after the listed kernels have completed. The ordering guarantee is always observed; the implementation varies by backend (see each backend's reference for details).

```boring
kernel:
    loop:
        k_sim(block = 256)
        k_shade(block = (16, 16), after = k_sim)
        screen.present(k_shade.pixels)
```

`after =` accepts a single kernel variable or a list:

```boring
k_render(block = (16, 16), after = [k_a, k_b])
```

### Render loop

`loop:` inside `kernel:` drives a render loop. See [`gpu-display.md`](gpu-display.html).

```boring
kernel:
    loop:
        k(block = (16, 16))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
```

---

## Execution context built-ins

Inside `def ()` and device helpers, `gpu` is available:

| Built-in | Description |
|---|---|
| `gpu.thread.x/y/z` | thread index within block |
| `gpu.block.x/y/z` | block index within grid |
| `gpu.block_dim.x/y/z` | threads per block |
| `gpu.grid_dim.x/y/z` | blocks per grid |
| `sync` | explicit block-level barrier (manual mode — see `'sync`) |

---

## `GPU` type

`GPU` is a built-in type for device selection and property queries:

```boring
let g = GPU(0)
print "Device: {g.name()} — {g.totalMem() / 1_073_741_824} GB"

for g in GPU.all():
    print "[{g.index()}] {g.name()} — {g.freeMem()} bytes free"
```

| Method | Returns | Notes |
|---|---|---|
| `name()` | device model name | |
| `totalMem()` | total VRAM in bytes | may be 0 on unified-memory GPUs on wgpu |
| `freeMem()` | available VRAM in bytes | always 0 on wgpu — not exposed by the API |
| `computeCapability()` | `[major, minor]` | CUDA SM version; `[0, 0]` on wgpu and Metal |
| `warpSize()` | threads per warp | conservative default (32) on wgpu |
| `maxThreads()` | max threads per block | |
| `maxSharedMem()` | max shared memory per block (bytes) | |
| `index()` | device index | |

---

## Memory safety model

### `'unified` — zero-copy host/device

`'unified` fields share physical memory between host and device. No explicit H2D/D2H copy is needed. The `kernel:` block guarantees completion before the next host line.

### `'global` — device-only

`'global` fields live in device DRAM. Host reads/writes require explicit `gpu.copy()`:

```boring
gpu.copy(k.result, host_buf)    # D2H
gpu.copy(host_buf, k.input)     # H2D
```

The copy mechanism is backend-specific (staging buffers on wgpu, `cudarc` transfers on CUDA, blit on Metal) but the Boring source is identical across targets.

### `'sync` — block SRAM

`'sync` fields are allocated in per-block shared memory. The transpiler inserts thread-group barriers automatically — no explicit `sync` statement needed.

#### Fixed size — `[T, N]'sync`

The size is baked into the kernel declaration. No `init()` assignment needed — the field exists for all threads in the block.

```boring
kernel Reduce:
    mut [float]'unified      input
    mut float'unified        result
    mut [float, 256]'sync    tile   # 256 floats, fixed at compile time

    init([float]'unified data):
        input = data

    def ():
        let tid = gpu.thread.x
        tile[tid] = input[gpu.block.x * gpu.block_dim.x + tid]
        # barrier inserted automatically before the loop
        if tid == 0:
            var float sum = 0.0
            for v in tile:
                sum += v
            result = sum
```

#### Dynamic size — `[T]'sync`

When the tile size must match the block dimension at runtime, declare the field without a size and allocate it in `init()` using `[..n]`. The transpiler passes `block_dim.x * sizeof(T)` as the dynamic shared memory size automatically.

> **wgpu limitation**: dynamic `[T]'sync` is not supported in WGSL — use `[T, N]'sync` with a const generic param instead.

```boring
kernel Reduce:
    mut [float]'unified  input
    mut float'unified    result
    mut [float]'sync     tile   # one float per thread in the block

    init([float]'unified data, int block_size):
        input = data
        tile  = [..block_size]   # allocate without initialization

    def ():
        let tid = gpu.thread.x
        tile[tid] = input[gpu.block.x * gpu.block_dim.x + tid]
        if tid == 0:
            var float sum = 0.0
            for v in tile:
                sum += v
            result = sum
```

#### Auto-barrier rules

The transpiler operates in **auto mode** when a kernel `def` has no explicit `sync` statement:

1. A barrier is inserted before the first loop in the body (write-phase → loop-phase boundary).
2. A barrier is inserted at the top of each loop iteration that accesses a `'sync` field.

#### Manual mode

If the developer writes at least one explicit `sync` in the `def` body, the transpiler disables auto-insertion for the entire `def` and emits barriers only where `sync` appears.

```boring
    def ():
        tile[tid] = data[i]
        sync                    # developer controls all barriers
        while stride > 0:
            if tid < stride:
                tile[tid] = tile[tid] + tile[tid + stride]
            sync
            stride = stride / 2
```

### `'sync` field rules

`'sync` fields cannot escape the kernel body. The compiler rejects:

- returning a `'sync` value from a kernel
- storing a `'sync` value in a field accessible from the host
- passing a `'sync` reference outside the kernel invocation scope

### Struct `'sync` — compound state

```boring
struct Stats:
    float sum
    int   count

kernel BlockStats:
    let [float]'global    data
    mut [Stats]'sync      acc
    mut [float]'unified   result

    def ():
        let tid = gpu.thread.x
        acc[tid] = Stats(sum = data[tid], count = 1)
        var stride = gpu.block_dim.x / 2
        while stride > 0:
            if tid < stride:
                acc[tid] = Stats(
                    sum   = acc[tid].sum   + acc[tid + stride].sum,
                    count = acc[tid].count + acc[tid + stride].count,
                )
            stride = stride / 2
        if tid == 0:
            result[gpu.block.x] = acc[0].sum / acc[0].count as float
```

---

## Atomics

Tag a field with `'actor'global` to enable atomic operations on indexed access:

```boring
kernel Histogram:
    mut [int]'actor'global counts = [0, 0, 0, 0]

    def ():
        counts[bucket] += 1     # compiled to atomicAdd
```

`'actor` alone is an alias for `'actor'global` — atomics are only meaningful (and only
implemented) for device-global memory here, so there's no other qualifier `'actor` could
mean:

```boring
kernel Histogram:
    mut [int]'actor counts = [0, 0, 0, 0]   # same as 'actor'global
```

Supported atomic operations: `+= -= &= |= ^=`.

---

## Simulation mode

`boring run` executes kernels sequentially on the CPU — same source file, no GPU required.

```sh
boring run main.br
boring run --gpu a100 main.br
boring run --gpu h100 main.br
boring run --gpu path/to/my.toml main.br
```

### Built-in profiles

These profiles simulate the reported properties of named GPU models. No real GPU is required.

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
totalMem = 8589934592
warpSize = 32
maxThreads = 1024
maxSharedMem = 49152
computeCapability = [8, 6]
```

> Sequential simulation hides data races between threads. A kernel correct in simulation may be incorrect on real hardware if barriers are missing. In auto mode the transpiler inserts them; in manual mode (`sync` present in the `def`) the developer is responsible.
