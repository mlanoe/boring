# GPU computing — language reference

Boring supports GPU computing through `kernel` structs — a dedicated declaration form that groups device memory fields, an `init` allocator, device-side helpers, and an anonymous entry point (`def ()`).

This document covers the **language syntax and semantics** shared by all backends. For backend-specific codegen details, see:

- [`cuda-module.md`](cuda-module.html) — CUDA C mapping, generated project layout, `cudarc` host API, PTX compilation
- [`rocm-backend.md`](rocm-backend.html) — HIP C++ mapping (near-identical to CUDA C), hand-rolled HIP FFI host API, AMD GPU support
- [`metal-backend.md`](metal-backend.html) — MSL address space mapping, Metal runtime compilation
- [`wgpu-backend.md`](wgpu-backend.html) — WGSL mapping, pipeline overrides, cross-platform GPU support

A `kernel` struct requires one of the four GPU targets above. The default `boring build` (`std` target) has no GPU backend at all: it emits a warning and drops the struct — along with its device code and any `k(...)`/`kernel:` dispatch that referenced it — instead of transpiling it. `boring build --target kernel` (Rust-for-Linux) rejects it outright as an error, since that target has no host/device split either.

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

`let`/`mut`/`var` are mandatory on every field. GPU memory qualifiers (`'unified`, `'global`, bare `'actor`, `'local`, `'const`) replace the usual ownership qualifiers inside a `kernel` struct. For scalars and fixed-size arrays the qualifier may be omitted — the transpiler infers it from the binding (see [Qualifier inference](#qualifier-inference) below).

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
| bare `'actor` | block SRAM (`__shared__` / `threadgroup` / `var<workgroup>`) | no |
| `'local` | registers / thread-local | no — default |
| `'const` | constant cache | no |
| `'actor'global` | device-only DRAM, atomic access | via `gpu.copy()` — see [Atomics](#atomics) |
| `'actor'unified` | unified DRAM (host + device), atomic access | direct — see [Atomics](#atomics) |

`'surface` is restricted to `[uint]` fields and is intended for pixel buffers
presented to a `Screen`. See [`gpu-display.md`](gpu-display.html).

Bare `'actor` inside a `kernel` struct means block-shared memory with an
auto-inserted barrier — this used to be spelled `'sync`; the rename reuses the
same "compiler automatically wraps every access with the protection this data
needs" meaning `'actor` already has outside kernel context (`Rc<RefCell<T>>`/
`Arc<Mutex<T>>`). Atomics on device-only or unified DRAM must be spelled out in
full (`'actor'global` / `'actor'unified`) — bare `'actor` is no longer an alias
for either.

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

| | `'unified` | `'global` | bare `'actor` | `'local` | `'const` |
|---|---|---|---|---|---|
| `[T]` dynamic | explicit | explicit | explicit | error | error |
| `[T, N]` fixed | error | error | explicit | inferred (`mut`/`var`) | inferred (`let`) |
| `[T, width, height]` labeled (dynamic) | explicit | explicit | explicit | error | error |
| `[T, width=W, height=H]` labeled (fixed) | explicit | explicit | explicit | inferred (`mut`/`var`) | inferred (`let`) |
| scalar | — | — | — | inferred (`mut`/`var`) | inferred (`let`) |

> `'const` and `'local` are always inferred and rarely written explicitly. `'unified`, `'global`, and bare `'actor` are always explicit.

> **Labeled arrays vs `[T, N]` fixed — the one place they differ**: `[T, N]'unified`/`[T, N]'global` are errors ("size is implicit from the init parameter"), but `[T, width=W, height=H]'unified`/`'global` are valid. The reason: `[T, N]`'s host-side representation is a true fixed-size Rust array, which conflicts with `'unified`/`'global`'s always-dynamic host buffer (`Vec`/`CudaSlice`/`DeviceBuffer`/`Buffer`, sized at runtime from the constructor's argument). A fixed-shape labeled array doesn't have that conflict — its host representation *is* the same dynamic buffer type as `[T]`; the axis sizes are compile-time shape metadata used for indexing and grid inference, not a competing fixed length. See [`array-multidim-types.md`](array-multidim-types.html). `'actor'global`/`'actor'unified` compose with labeled arrays the same as with `[T]`/`[T,N]` — e.g. `mut [int, width=256, height=256]'actor'global histogram` is valid.

---

**Host-context** (bindings outside `kernel` struct):

| Qualifier | Location |
|---|---|
| `'gpu'unified` | unified host + device DRAM |
| `'gpu'global` | device-only DRAM |

`'const` (like bare `'actor` and `'local`) has no host-context form — it has no host access at
all (see the table above), so a host-side binding could never be read from or written to.
It's only meaningful as a field qualifier inside a `kernel` struct.

---

## Labeled multi-dimensional arrays — `[T, width, height]`

`[T, width, height]` / `[T, width = W, height = H]` are built-in named-axis
types for 2D/3D compute buffers — replacing the pattern of a flat `[T]` field
plus separate plain-`int` `rows`/`cols` fields and hand-rolled linear-index
math. Every index and every axis size is spelled out by label instead of by
argument position:

```boring
kernel Transpose:
    let [float, width = C, height = R]'global src
    mut [float, width = R, height = C]'unified dst

    def ():
        let c = gpu.thread.x
        let r = gpu.thread.y
        dst[width = r, height = c] = src[width = c, height = r]
```

- **Indexing**: `a[width = w, height = h]` — no `.at(...)` method call; labels
  are mandatory for 2+ axes and order-free at the use site
  (`a[height=h, width=w]` is identical to `a[width=w, height=h]`).
- **Shape queries**: `a.width` / `a.height` / `a.depth` — each declared axis
  is a read-only property, no separate accessor method per axis and no call
  syntax at all.
- **Layout**: row-major — the *first declared label* is the fastest-varying
  axis (`a[width=w, height=h]` lowers to flat index `w + h*width_size`),
  regardless of what the labels are named.
- **Qualifiers**: same set as `[T]`/`[T, N]` combined (see the matrix above)
  — `'unified`, `'global`, bare `'actor`, `'actor'global`, `'actor'unified`,
  `'const`, `'local` are all valid; `'surface` is not (see below). Always
  placed after the closing bracket, e.g. `[float, width=16, height=16]'actor`,
  never inside it.
- **Grid inference**: a `kernel:` block with no explicit `grid=` defaults
  from the field's fixed axis sizes (or, for a dynamic-shape field, from the
  shape it was constructed with) instead of falling back to the 1D
  `ceil(len/block)` used for flat arrays — see each backend doc's own
  grid-inference section (`cuda-module.md`, `rocm-backend.md`,
  `metal-backend.md`, `wgpu-backend.md`).
- **No `.at(...)` positional form** — this is deliberate, not a gap: the
  whole point is that swapping which argument means which axis is a parse
  error, not a silent bug.

Full design rationale, the fill-shorthand construction forms
(`[value for width=w, height=h]`), `.reshape()`/`.flatten()`, and the
cross-label safety rule: [`array-multidim-types.md`](array-multidim-types.html).

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
| `sync` | explicit block-level barrier (manual mode — see bare `'actor`) |
| `gpu.warp.size` | threads per warp/wavefront/SIMD-group/subgroup |
| `gpu.warp.lane` | this thread's index within its warp, `0..gpu.warp.size` |
| `gpu.warp.sync()` | warp-local barrier — cheaper than `sync` (block-wide) |
| `gpu.warp.shuffle_down(v, delta)` | read `v` from the lane `delta` above this one |
| `gpu.warp.shuffle_up(v, delta)` | read `v` from the lane `delta` below this one |
| `gpu.warp.shuffle_xor(v, mask)` | read `v` from lane `this_lane XOR mask` (butterfly pattern — reductions) |
| `gpu.warp.shuffle(v, src_lane)` | broadcast/read `v` from an arbitrary lane |

`gpu.warp.*` is device-side, current-thread state — not to be confused with
the host-side `GPU(0).warpSize()` below, which queries a device by index from
ordinary (non-kernel) code. Same hardware concept, unrelated call sites.

See [warp-level primitives](warp-level-primitives.html) for the full design
(per-backend mapping, the wgpu real-subgroup/emulated-fallback split, and the
divergent-branch caveat).

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

### Bare `'actor` — block SRAM

Bare `'actor` fields (formerly spelled `'sync`) are allocated in per-block shared memory. The transpiler inserts thread-group barriers automatically — no explicit `sync` statement needed.

#### Fixed size — `[T, N]'actor`

The size is baked into the kernel declaration. No `init()` assignment needed — the field exists for all threads in the block.

```boring
kernel Reduce:
    mut [float]'unified      input
    mut float'unified        result
    mut [float, 256]'actor   tile   # 256 floats, fixed at compile time

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

#### Dynamic size — `[T]'actor`

When the tile size must match the block dimension at runtime, declare the field without a size and allocate it in `init()` using `[..n]`. The transpiler passes `block_dim.x * sizeof(T)` as the dynamic shared memory size automatically.

> **wgpu limitation**: dynamic `[T]'actor` is not supported in WGSL — use `[T, N]'actor` with a const generic param instead.

```boring
kernel Reduce:
    mut [float]'unified  input
    mut float'unified    result
    mut [float]'actor    tile   # one float per thread in the block

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
2. A barrier is inserted at the top of each loop iteration that accesses a bare-`'actor` field.

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

### Bare `'actor` field rules

Bare-`'actor` fields cannot escape the kernel body. The compiler rejects:

- returning a bare-`'actor` value from a kernel
- storing a bare-`'actor` value in a field accessible from the host
- passing a bare-`'actor` reference outside the kernel invocation scope

### Struct bare `'actor` — compound state

```boring
struct Stats:
    float sum
    int   count

kernel BlockStats:
    let [float]'global    data
    mut [Stats]'actor     acc
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

Tag a field with `'actor'global` (device-only DRAM) or `'actor'unified` (host + device
DRAM) to enable atomic operations on indexed access:

```boring
kernel Histogram:
    mut [int]'actor'global counts = [0, 0, 0, 0]

    def ():
        counts[bucket] += 1     # compiled to atomicAdd
```

`'actor'unified` behaves identically for the atomic op itself — the only difference is
memory placement, exactly like plain `'global` vs. `'unified`:

```boring
kernel Histogram:
    mut [int]'actor'unified counts = [0, 0, 0, 0]

    def ():
        counts[bucket] += 1     # compiled to atomicAdd; counts is host-readable directly
```

Bare `'actor` is a different qualifier entirely (block-shared memory — see above) and is
**not** an alias for either atomic form; both must be spelled out in full.

Supported compound-assign atomic operations: `+= -= &= |= ^=`.

### `.min` / `.max` / `.swap` / `.cas`

Four more atomic operations are available as **methods** on an indexed element of an `'actor'global`/`'actor'unified` field, rather than an operator — `min`/`max`/`exchange`/`compare-and-swap` have no natural infix operator the way add/sub/or/and/xor do. No `atomic` prefix on the method names either: the qualifier already establishes that every access to this field is atomic, so repeating it in each method name would be redundant.

```boring
kernel Histogram:
    mut [int]'actor'global counts = [0, 0, 0, 0]

    def ():
        _ = counts[bucket].min(v)             # atomicMin — new value is min(old, v)
        _ = counts[bucket].max(v)             # atomicMax — new value is max(old, v)
        _ = counts[bucket].swap(v)            # atomicExch — unconditional exchange
        _ = counts[bucket].cas(expected, new) # compare-and-swap
```

All four **return the previous value** — matching CUDA/HIP's real `atomicMin`/`atomicMax`/`atomicExch`/`atomicCAS` semantics exactly, on every backend. For `.cas`, the caller compares the returned value against `expected` to tell whether the swap actually happened:

```boring
let old = counts[bucket].cas(0, 1)
if old == 0:
    print "claimed it"
else:
    print "someone else got there first, current value was {old}"
```

Like any other call whose return value is used as a bare statement, the checker requires the result to be explicitly bound (`let old = ...`) or discarded (`_ = ...`) — silently dropping it is a compile error, not a warning.

### `.min`/`.max`/`.swap`/`.cas` without `'actor`

These four methods work on **any** indexed element, not only an `'actor'global`/`'actor'unified` one — matching `+= -= &= |= ^=`'s existing behavior, which already degrades to plain (non-atomic) arithmetic off a non-actor field instead of erroring:

```boring
kernel Scale:
    mut [int]'unified buf   # plain, not 'actor'global/'actor'unified

    def ():
        let old = buf[i].min(v)   # plain read-modify-write, not atomicMin — no
                                   # cross-thread protection, same as buf[i] += v
                                   # would give on this same field
```

The mechanism differs by backend, since only CUDA/HIP/Metal can express "read the old value, mutate, yield the old value" as a single expression (a GNU/Clang statement-expression, `({ ... })`) — WGSL has nothing equivalent. On wgpu, the plain fallback only works when the call is the **entire right-hand side** of a `let`/assignment statement (`let old = buf[i].min(v)`, or `_ = buf[i].min(v)` to discard); anywhere else — nested inside a larger expression — it isn't representable in WGSL as a single unit, and the generated shader carries a visible, unmistakable marker at that spot rather than silently computing something else or emitting invalid WGSL.

---

## GPU error handling

`GpuError` is a built-in enum, always available without import — the same mechanism as the built-in `Error` enum ([`book.md`](book.html), "Error Handling"):

```boring
enum GpuError:
    LaunchError
    OutOfMemory
    IllegalAccess
    StackOverflow
    Timeout
    DeviceLost
```

```boring
mut k = Scale(data)
try:
    kernel:
        k(block = 256)
catch GpuError.OutOfMemory:
    print "ran out of GPU memory"
catch GpuError.LaunchError:
    print "kernel launch was rejected"
```

**Only genuinely catchable on `--target wgpu`.** `catch GpuError.Variant:` desugars to a `BoringError` downcast — the exact mechanism `throws CalcError`/`catch CalcError.Variant:` already uses for a user-declared enum (`book.md`). wgpu's host codegen shares the same general transpiler pipeline that mechanism lives in; CUDA, ROCm, and Metal each have their own smaller, kernel-only host transpiler that doesn't (the same prerequisite gap already documented for `with` in `scoped-access-blocks.md`). On those three backends, a GPU failure is still reported as a `Box<dyn Error>` with a classified message (e.g. `"GPU out of memory: ..."`) — informative, but not something a Boring `catch GpuError.OutOfMemory:` block will match. See each backend's own doc for exactly what's classified:

- [`wgpu-backend.md`](wgpu-backend.html) — full typed `GpuError`, catchable
- [`cuda-module.md`](cuda-module.html), [`rocm-backend.md`](rocm-backend.html), [`metal-backend.md`](metal-backend.html) — classified message only

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
