# GPU computing — language reference

Boring supports GPU computing through `kernel` structs — a dedicated declaration form that groups device memory fields, an `init` allocator, device-side helpers, and an anonymous entry point (`def ()`).

For the Metal backend (macOS), see [`metal-backend.md`](metal-backend.html).

For CUDA codegen details (generated Rust/CUDA C, limitations, substitution table), see [`cuda-module.md`](cuda-module.html).

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

mut k = Scale(data)
mut k = kernel(block = 256) k |> .wait
print k.buf[0]
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

## GPU memory qualifiers

**Kernel-context** (inside `kernel` struct fields):

| Qualifier | Memory space | Host access |
|---|---|---|
| `'unified` | unified DRAM (host + device) | direct |
| `'global` | device-only DRAM | via `gpu.copy()` |
| `'sync` | block SRAM (`__shared__`) | no |
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
| `'gpu'const` | GPU constant cache |

---

## Launch expression

```boring
kernel(block = N) k
kernel(block = N, grid = M) k
kernel(block = (16, 16)) k
kernel(block = 256, after = h1) k
kernel(block = 256, smem = {tile = 4096}) k
kernel(block = 256, priority = high) k
```

Returns a `KernelHandle`:

```boring
struct KernelHandle<T>:
    req bool done()
    req T    wait()
```

Common pattern with the pipe operator:

```boring
mut k = kernel(block = 256) k |> .wait
```

### Dispatch parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `block` | `int` or `(int, int)` or `(int, int, int)` | yes | threads per block |
| `grid` | `int` or tuple | no | blocks per grid — inferred from field length if omitted (1D) |
| `smem` | `{string = int}` | no | named dynamic `'sync` partitions and their byte sizes |
| `after` | handle or `[handle]` | no | kernel starts after all listed handles complete |
| `priority` | `high` / `normal` / `low` | no | stream scheduling priority — default `normal` |

Device is bound at instantiation (`new(g) Scale(n)`), not at dispatch.

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

`'unified` fields share physical memory between host and device. No explicit H2D/D2H copy is needed. Guard with `wait()` before reading on the host after a launch.

### `'global` — device-only

`'global` fields live in device DRAM. Host reads/writes require explicit `gpu.copy()`:

```boring
gpu.copy(k.result, host_buf)    # D2H
gpu.copy(host_buf, k.input)     # H2D
```

### `'sync` — block SRAM

`'sync` fields are allocated in per-block shared memory. The transpiler inserts thread-group barriers automatically — no explicit `sync` statement needed.

```boring
kernel Reduce:
    mut [float]'unified input
    mut float'unified   result
    mut [float]'sync    tile

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

#### Auto-barrier rules

The transpiler operates in **auto mode** when a kernel `def` has no explicit `sync` statement:

1. A barrier is inserted before the first loop in the body (write-phase → loop-phase boundary).
2. A barrier is inserted at the top of each loop iteration that accesses a `'sync` field (covers cross-thread read patterns such as stride reduction).

#### Manual mode

If the developer writes at least one explicit `sync` in the `def` body, the transpiler disables auto-insertion for the entire `def` and emits barriers only where `sync` appears. Use this when the automatic placement is incorrect or suboptimal.

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

### Ownership and launch

`kernel(...)` moves the kernel object into the handle. `.wait` returns it. This prevents host access to the kernel's fields while the device is running.

```boring
mut k = Scale(1024)
k.buf[0] = 1.0
let h = kernel(block = 256) k    # k moved into h
mut k = h.wait                   # k returned after completion
print k.buf[0]
```

The kernel is reusable — `.wait` returns the same object:

```boring
var k = Scale(1024)
for batch in batches:
    for i in ..n: k.buf[i] = batch[i]
    k = kernel(block = 256) k |> .wait
    results.push(k.buf[0])
```

### `'sync` field rules

`'sync` fields cannot escape the kernel body. The compiler rejects:

- returning a `'sync` value from a kernel
- storing a `'sync` value in a field accessible from the host
- passing a `'sync` reference outside the kernel invocation scope

### Struct `'sync` — compound state

When multiple scalars must be observed as a consistent unit across threads, group them in a struct and apply `'sync` to the field. The barrier covers all fields of the struct atomically — no per-field synchronisation is needed.

Use this instead of `'actor'global` (which protects a single scalar at a time): atomics cannot guarantee that two independently updated values are seen together by a third thread.

```boring
struct Stats:
    float sum
    int   count

kernel BlockStats:
    let [float]'global    data
    mut [Stats]'sync      acc      # compound state — barrier covers both fields
    mut [float]'unified   result

    def ():
        let tid = gpu.thread.x
        acc[tid] = Stats(sum = data[tid], count = 1)
        # barrier inserted automatically before the loop

        var stride = gpu.block_dim.x / 2
        while stride > 0:
            if tid < stride:
                acc[tid] = Stats(
                    sum   = acc[tid].sum   + acc[tid + stride].sum,
                    count = acc[tid].count + acc[tid + stride].count,
                )
            stride = stride / 2
        # barrier inserted automatically at the top of each iteration

        if tid == 0:
            result[gpu.block.x] = acc[0].sum / acc[0].count as float
```

`'actor'global` is appropriate for independent counters (one atomic per slot). `struct'sync` is appropriate when two or more values must be read together coherently.

### Dynamic `'sync` — size from launch

```boring
kernel Reduce:
    mut [float]'sync tile    # size from smem at launch

    def (): ...

mut k = Reduce(n)
mut k = kernel(block = 256, smem = {tile = 256 * 4}) k |> .wait
```

Multiple named partitions:

```boring
mut k = kernel(block = 256, smem = {tile = 256 * 4, flags = 64 * 4}) k |> .wait
```

The transpiler generates the byte-offset arithmetic automatically.

---

## Atomics

Tag a field with `'actor'global` to enable atomic operations on indexed access:

```boring
kernel Histogram:
    mut [int]'actor'global counts = [0, 0, 0, 0]

    def ():
        counts[bucket] += 1     # compiled to atomicAdd
```

Supported atomic operations: `+= -= &= |= ^=`.

---

## Multi-device

Device placement is controlled at construction time with `new(gpu)`:

```boring
let g0 = GPU(0)
let g1 = GPU(1)

let k0 = new(g0) Scale(input)
let k1 = new(g1) Scale(input)

kernel(block = 256) k0
kernel(block = 256) k1
```

`GPU.count()` returns the number of available devices.

### `after =` ordering

```boring
let h0 = kernel(block = 256) k0
let h1 = kernel(block = 256, after = h0) k1
```

`after =` accepts a single handle or a list: `after = [h0, h1]`.

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

---

## CUDA codegen

`boring build --target cuda` generates a Cargo project with Rust host code, CUDA C device code, and a `build.rs` that invokes `nvcc`. Requires the CUDA toolkit and a CUDA-capable GPU.

For the full mapping of Boring constructs to CUDA C, generated file layout, and backend limitations, see [`cuda-module.md`](cuda-module.html).
