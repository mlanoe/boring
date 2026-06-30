# Boring GPU

> Target: **CUDA only**. No OpenCL support is planned.
>
> **Implementation status:**
> - `kernel` struct declaration, `init`, device-side methods, and the anonymous entry point `def ()` — **implemented** (interpreter + CUDA codegen)
> - GPU memory qualifiers (`'unified`, `'global`, `'shared`, `'local`, `'const`, `'gpu'*`) — **implemented** (parser + interpreter + CUDA codegen)
> - `kernel(block = N, grid = M) expr` launch expression, `KernelHandle` with `.wait` / `.done()` — **implemented** (interpreter + CUDA codegen)
> - `|> .wait` pipe syntax — **implemented**
> - `gpu.thread.x/y/z`, `gpu.block.x/y/z`, `gpu.block_dim`, `gpu.grid_dim` built-ins — **implemented** (interpreter simulation + CUDA codegen)
> - `sync` — **implemented** (no-op in simulation, `__syncthreads()` in CUDA codegen)
> - CPU simulation mode (`boring run`) — **implemented** (sequential thread loop)
> - CUDA codegen (`boring build --target cuda`) — **implemented** (single-GPU, 1D grids)
> - D2H readback — **implemented** (`k.buf` → `read_buf()?` inferred automatically)
> - Dynamic `smem_bytes` — **implemented** (inferred from `'shared` array fields)
> - Automatic 1D grid sizing from a `[T]'global`/`'unified` array field's length, 2D/3D grids via tuple `block=`, `after =` + streams, atomics via `'actor'global` (`+=`/`-=`/`|=`/`&=`/`^=` only) — **implemented** (CUDA codegen + interpreter)
> - `'actor'unified`, `'actor'shared`, `atomicMin`/`atomicMax`/`atomicExch` auto-transpile, `atomic.cas(...)`, `warp.size`/`warp.lane`/`warp.sync`, `gpu.const(...)`, `.shape`-based grid inference, must-use `KernelHandle`, kernel-qualifier rejection, `--peer-access`, typed `Gpu*Error`s — **not yet implemented** (see inline notes below)
> - Multi-GPU (`new(g) K(...)`, `GPU.all()`), `GPU` built-in type with device properties — **implemented**
> - `'shared` / `'local` in `init` → validation error, `print` in kernel → `printf` — **implemented**
> - dtod inference (no explicit copy) — **not yet implemented**

---

## What GPU computing fundamentally requires

1. **Memory spaces** — global, local (on-chip), constant, unified. Storage qualifiers, not ownership qualifiers.
2. **Thread hierarchy** — thread index, block index, block dimensions — implicit built-ins inside every kernel.
3. **Kernel functions** — launched from the host, executed on the GPU.
4. **Launch configuration** — number of blocks and threads per block, passed at launch time.
5. **Barriers** — explicit synchronisation between threads, atomics.
6. **Memory transfers** — explicit host↔device copies or unified memory.

The target is **CUDA**.

---

## Qualifier model: two contexts

GPU qualifiers appear in two contexts with different syntax:

- **Host context** — allocations and bindings outside a `kernel` struct. Full `'gpu'..` prefix is mandatory.
- **Kernel context** — fields inside a `kernel` struct. The `'gpu` prefix is dropped; short forms `'unified`, `'global`, `'shared`, `'local`, `'const` are used. `'local` is the default and may be omitted.

### Host-context qualifiers

| Qualifier | Location | CUDA |
|---|---|---|
| `'gpu'unified` | unified host + device DRAM | `cudaMallocManaged` |
| `'gpu'global` | device-only DRAM | device pointer |
| `'gpu'const` | GPU constant cache | `__constant__` |

### Kernel-context qualifiers (inside `kernel` struct fields)

| Qualifier | CUDA memory space | Host access | Default |
|---|---|---|---|
| `'local` | registers / thread-local memory | no | yes — may be omitted |
| `'shared` | block SRAM (`__shared__`) | no | no |
| `'global` | device-only DRAM | via `gpu.copy()` | no |
| `'unified` | unified DRAM (`cudaMallocManaged`) | direct | no |
| `'const` | constant cache | no | no — implicit via `let` |

`'unified` and `'global` fields cross the host/device boundary and are allocated in `init`. `'shared` and `'local` fields are internal to the kernel — never accessible from the host.

### Binding × mutability in kernel struct fields

`let`, `mut`, and `var` are **mandatory** for all kernel struct fields. They are orthogonal to the memory qualifier.

| | `let` — immutable | `mut` — fixed ptr, mutable data | `var` — ptr + data mutable |
|---|---|---|---|
| scalar | `const float sigma` | — | `int i` (register) |
| `'global` array | `const float* const` | `float* const` | `float*` |
| `'shared` static | — | `__shared__ float tile[N]` | — |
| `'shared` dynamic | — | `extern __shared__ float tile[]` | — |

**Implicit qualifier rules:**
- `let` scalar or array without explicit qualifier → `'const` implicit
- `var` scalar → `'local` implicit (register)
- `mut` array without explicit qualifier → `'local` implicit (thread-local array)

### Three-axis model

Boring qualifiers encode three independent dimensions:

```
qualifier = storage location × access mode × lifetime
```

**Axis 1 — Storage location (host context)**

| Qualifier | Location | Rust impl |
|---|---|---|
| (implicit) | stack | `T` |
| `'heap` | heap, exclusive | `Box<T>` |
| `'shared` / `'actor` / `'guard` | heap, ref-counted | `Rc`/`Arc` + wrapper |
| `'gpu'unified` | GPU+CPU DRAM, unified | `cudaMallocManaged` |
| `'gpu'global` | GPU DRAM only | device pointer |
| `'gpu'const` | GPU constant cache | `__constant__` |

**Axis 2 — Access mode**

| Qualifier | Readers | Writers | Coordination |
|---|---|---|---|
| (implicit) / `'heap` | 1 owner | 1 owner | none |
| `'shared` | N | 0 | none (immutable) |
| `'actor` | N | N | built-in (RefCell / Mutex) |
| `'guard` | N (under lock) | 1 (under lock) | explicit |
| `'gpu'unified` | host + all threads | host + all threads | `h.wait` — host access resumes after |
| `'gpu'global` | all threads | all threads | none built-in → atomics or sync |
| `'gpu'const` | all threads | 0 | none (hardware-enforced ro) |

**Key insight:** for GPU memory, scope is a consequence of physical location — `'global` is always grid-wide, `'shared` is always block-scoped, registers are always thread-private. No separate scope qualifier is needed.

**Axis 3 — Lifetime**

| Qualifier | Lifetime | Mechanism |
|---|---|---|
| (implicit) | owner scope | automatic — stack frame |
| `'heap` | owner scope | RAII — Rust drop |
| `'shared` / `'actor` / `'guard` | last Arc drop | reference counting |
| `'gpu'global` | host owner scope | RAII — `cudaFree` on drop |
| `'gpu'const` | program lifetime | static — loaded at startup |
| `'gpu'unified` | host owner scope | RAII — `cudaFree` on drop |

---

## Memory safety model

### Ownership model — the kernel owns its buffers

A `kernel` struct owns its `'unified` and `'global` buffers for its entire lifetime. Buffers are allocated in the `init` method. The host reads and writes `'unified` fields directly on the kernel object between launches.

**`kernel(...)` moves the kernel object into the handle. `.wait` returns it.** This prevents host access to the kernel's fields while the device is running — the same compile-time guarantee as Rust's borrow checker.

```boring
kernel Scale:
    mut [float]'unified buf

    init(int n):
        buf = [..n]

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] *= 2.0

mut k = Scale(1024)              # init called — buf allocated
k.buf[0] = 1.0                   # host write before launch
let h = kernel(block = 256) k    # k moved into h — inaccessible
# k.buf[0]                       # compile-time error — k moved
mut k = h.wait                   # k returned after kernel completion
print k.buf[0]                   # host read — ok
```

The kernel is reusable — `.wait` returns the same object; re-launch without reallocating:

```boring
var k = Scale(1024)
for i in ..n: k.buf[i] = data[i]

for batch in batches:
    for i in ..n: k.buf[i] = batch[i]
    k = kernel(block = 256) k |> .wait   # reassign — k moved in RHS, result bound back
    results.push(k.buf[0])
```

`var` is required at the outer scope — `mut` would create a new loop-body-scoped binding on each iteration, leaving the outer `k` moved and inaccessible. Plain assignment `k = ...` reassigns the existing `var` binding.

### Device-side safety

Inside the kernel, device threads access `'global` buffers concurrently. Boring cannot verify the absence of data races between threads. The `kernel` body is an **implicit unsafe zone** for device-side memory access.

`sync` and `'actor'global` compound-assign atomics are the programmer's tools in this zone today; `atomic.cas` is planned but not yet implemented (see [Atomics](#atomics--actor-on-kernel-fields)).

### `'shared` — lifetime enforced by the type system

`'shared` fields (block SRAM) cannot escape the kernel body. The compiler rejects:

- returning a `'shared` value from a kernel
- storing a `'shared` value in a field accessible from the host
- passing a `'shared` reference outside the kernel invocation scope

### Pipelines — sharing buffers between kernels

The kernel-owns-buffers model means two kernel structs cannot share a buffer directly. For pipelines, the options are:

**Device-to-device transfer** — inferred automatically when a field is passed only to another kernel:

```boring
mut ka = StageA(n)
for i in ..n: ka.input[i] = data[i]
mut ka = kernel(block = 256) ka |> .wait

mut kb = StageB(ka.output)             # ka.output not read on host → dtod inferred
mut kb = kernel(block = 256) kb |> .wait
print kb.output[0]                     # D2H only here
```

**Kernel fusion** — when two stages always chain, write a single kernel with both passes. Avoids the intermediate DRAM round-trip entirely.

### Open: device-side safety innovation

The device-side unsafe zone is an open design space. Possible directions:

- **Phase types** — the compiler tracks whether `sync` has occurred since the last write, making unsynchronised reads a type error.
- **Thread ownership types** — a value tagged with its owning thread index; cross-thread access requires an explicit cast or barrier.
- **Warp-level ownership** — values owned by a warp rather than a thread or block; warp-synchronous access is safe by construction.

None of these exist in current GPU tooling. This is where Boring could bring something genuinely new.

---

## Kernel functions

`kernel` struct groups fields, an `init` method, device-side helpers, and an entry point. Top-level declaration — the host/device boundary is always explicit before the body is read.

Inline kernels (anonymous block at the call site) are not supported: the body would look like sequential host code while executing as thousands of GPU threads, which is a source of confusion.

A kernel launch returns a `KernelHandle`. Calling `.wait` blocks until execution completes. The kernel object retains ownership of its `'global` fields — results are read directly from the kernel object after `.wait`.

**Qualifier constraints on kernel struct instances (planned, not yet implemented).** `kernel(...)` moves the instance into the handle — ownership must be unambiguous, so the design intends to reject `'shared` (`Rc<T>`), `'actor` (`Rc<RefCell<T>>`), and `'guard` (`Mutex<T>`) bindings, since none of them allow moving out, leaving only `'stack` (owned value) and `'heap` (`Box<T>`) as valid. **This check does not exist today** — `check_kernel`/`KernelLaunch` handling in `src/validator/kernel.rs` only recurses into the launch config's sub-expressions and does not inspect the qualifier of the kernel-struct value being launched, and the qualifier inference pass (`src/transpiler/infer_qualifiers.rs`) has no kernel-specific logic either. Passing a `'shared`/`'actor`/`'guard`-qualified kernel struct to `kernel(...)` is not currently rejected at compile time.

**`'stack` vs `'heap` inference.** A kernel struct whose fields are `'unified` or `'global` holds only GPU buffer *handles* (pointers + metadata) on the CPU side — the bulk of the data lives in device memory. The struct's CPU footprint is therefore small regardless of the buffer sizes, which biases the inference toward `'stack`. The transpiler must not count GPU buffer capacity when estimating struct size for stack/heap placement.

### Struct kernel

A `kernel` struct groups fields (buffers and shared memory), an `init` method for allocation, device-side helpers (`req`/`def` methods), and the entry point (`def ()` — the call operator).

**Field qualifier rules:**

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
| `let [float] lut` | `'const` implicit — array | no | `const float* const lut` |
| `var int i` | `'local` implicit — register | no | `int i` |
| `mut [float, N] tile` | `'local` implicit — thread-local | no | `float tile[N]` |
| CPU qualifiers (`'heap`, `'actor`…) | — | — | compile-time error |

Rules:
- `let`/`mut`/`var` are mandatory — no implicit binding in `kernel` struct fields.
- `'unified` and `'global` fields are allocated in `init` and owned by the kernel struct.
- `'unified` fields are accessible directly from the host between launches.
- `'global` fields require `gpu.copy()` for host/device transfers.
- `'shared` and `'local` fields are internal — never accessible from the host.
- `let` on `'unified`/`'global` = device read-only (the host can still write via `'unified`).
- `let` without explicit qualifier → `'const` implicit.
- `var` scalar without qualifier → `'local` implicit (register).
- `mut` array without qualifier → `'local` implicit (thread-local).

```boring
kernel Reduce:
    let [float]'unified  input       # host writes before launch, device read-only
    mut [float]'unified  output      # device writes, host reads after .wait
    mut [float, 256]'shared tile     # block SRAM — 256 floats, static

    init(int n):
        input  = [..n]
        output = [..n / 256]

    def fill():
        let i = gpu.thread.x
        tile[i] = input[gpu.block.x * gpu.block_dim.x + i]

    def combine():
        var s = gpu.block_dim.x / 2
        while s > 0:
            if gpu.thread.x < s:
                tile[gpu.thread.x] += tile[gpu.thread.x + s]
            sync
            s /= 2

    def ():
        fill()
        sync
        combine()
        if gpu.thread.x == 0:
            output[gpu.block.x] = tile[0]
```

**Allocation and launch are separated.**

`init` allocates buffers. Launch configuration (`block`, `grid`, `smem`, `after`, device) is passed to `kernel(...)` at the dispatch site. The host reads and writes fields directly on the kernel object.

```boring
mut k = Reduce(1024)                                       # init called

# fill input — direct ('unified)
for i in ..1024: k.input[i] = data[i]

# dispatch — k moved into handle
let h = kernel(block = 256) k                              # 1D
let h = kernel(block = (16, 16)) k                         # 2D, grid inferred
let h = kernel(block = (16, 16), grid = (w/16, h/16)) k   # explicit grid

mut k = h.wait                                             # k returned
print k.output[0]                                          # read result
```

**`def ()` — the kernel entry point**

`def ()` is the device entry point. It receives launch configuration implicitly (`block`, `grid`, `smem` are available as built-ins inside the body). It can have additional parameters and multiple overloads:

```boring
kernel Saxpy:
    let float          a
    let [float]'global x
    mut [float]'global y

    init(float a, int n):
        self.a = a
        x = [..n]
        y = [..n]

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        if i < len(x): y[i] += a * x[i]
```

```boring
mut k = Saxpy(2.0, n)
for i in ..n: k.x[i] = x[i]
for i in ..n: k.y[i] = y[i]
mut k = kernel(block = 256) k |> .wait
# k.y holds the result
```

> `def ()` and `req ()` are general Boring concepts — callable structs. `req ()` is callable on `let` and `var` bindings (read-only self), `def ()` only on `var` bindings (mutating self). `set ()` is not permitted. See `book.md §8 — Anonymous call operator` for the full reference.
>
> **GPU-specific:** `kernel obj` auto-invokes `def ()` — only valid when `def ()` takes no parameters.

### `init` — allocation method

`init` follows the same semantics as in regular Boring structs — see `book.md §5 — Structs`. In a `kernel` struct, `init` is also where `'unified` and `'global` fields are allocated. Multiple `init` overloads are allowed.

### `KernelHandle<T>`

`kernel(...)` returns a `KernelHandle<T>` where `T` is the kernel struct type. The handle owns the kernel object until `.wait` is called.

Illustrative interface (pseudocode — `eq` is not a real Boring method-modifier keyword; only `req`, `def`, `mut`, `var`, `let` are recognized modifiers):

```boring
struct KernelHandle<T>:
    req bool done()      # non-blocking — true if the kernel has completed
    req T wait()         # blocking — waits for completion and returns the kernel object
```

`.wait` is the only way to recover the kernel object. **Planned, not yet implemented:** dropping the handle without calling `.wait` is intended to be a compile-time error (must-use) — no such diagnostic exists today, and the generated Rust `KernelHandle<T>` struct (`src/transpiler/cuda/host.rs`) has no `#[must_use]` attribute, so a dropped handle currently compiles silently.

### `GPU`

`GPU` is a built-in type. Each instance represents one physical GPU device.

```boring
struct GPU:
    type req GPU    (int i)    # handle for device i  — e.g. GPU(0)
    type req [GPU]  all()      # list of all available devices

    # Device property methods (read-only)
    req string  name()                  # device model name
    req int     totalMem()              # total VRAM in bytes
    req int     freeMem()               # available VRAM in bytes
    req [int]   computeCapability()     # [major, minor] — e.g. [8, 0] for A100
    req int     warpSize()              # threads per warp (typically 32)
    req int     maxThreads()            # max threads per block
    req int     maxSharedMem()          # max shared memory per block (bytes)
    req int     index()                 # device index (same as the int passed to GPU(i))
```

**Example:**
```boring
let g = GPU(0)
print "Device:  {g.name()}"
print "VRAM:    {g.totalMem() / 1_073_741_824} GB"
print "SM:      {g.computeCapability()[0]}.{g.computeCapability()[1]}"
print "Threads: {g.maxThreads()} per block"

for g in GPU.all():
    print "[{g.index()}] {g.name()} — {g.freeMem() / 1_073_741_824} GB free"
```

In simulation (`boring run`), property values come from the active GPU profile (see [GPU simulation profiles](#gpu-simulation-profiles)).

### Dispatch parameters

All dispatch parameters are passed to `kernel(...)`.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `block` | `int` or `(int, int)` or `(int, int, int)` | yes | threads per block — 1D, 2D, or 3D |
| `grid` | `int` or tuple | no | blocks per grid — inferred from kernel shape if omitted |
| `smem` | `{string = int}` | no | named dynamic `'shared` partitions and their byte sizes |
| `after` | handle or `[handle]` | no | ordering — kernel starts only after all listed handles complete |
| `priority` | `high` / `normal` / `low` | no | stream scheduling priority — default `normal` |

Device is bound at instantiation (`new(g) Scale(n)`), not at dispatch — see [Multi-device](#multi-device).

**`block` and `grid` as tuples — 2D and 3D kernels:**

```boring
# 1D
kernel(block = 256) k
kernel(block = 256, grid = 512) k                          # explicit grid

# 2D
kernel(block = (16, 16)) k                                 # grid inferred from k's shape
kernel(block = (16, 16), grid = (w/16, h/16)) k            # explicit grid

# 3D
kernel(block = (8, 8, 8)) k
```

**Grid inference rules (current implementation):**

- `[T]'global` / `'unified` 1D array field → `grid = ceil(len(buf) / block)`
- Otherwise, if `grid` is omitted, the transpiler currently defaults to `grid = (1, 1, 1)` rather than raising an error — this is likely a footgun rather than intended behavior, so always pass `grid` explicitly unless relying on 1D array inference.

**Planned, not yet implemented:**

- Type with `.shape` (e.g. `Image`, `Volume`) → `grid = ceil(shape / block)` per dimension. No `.shape`-based inference exists in the transpiler today, and there are no built-in `Image`/`Volume` types.
- Omitting `grid` when no inference rule applies raising a compile-time error (see current fallback behavior above).

---

## Execution context

Inside a kernel, `gpu` is available as a global implicit built-in:

```boring
kernel Scale:
    mut [float]'unified buf

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] *= 2.0
```

The execution context fields:

| Field | CUDA | Description |
|---|---|---|
| `gpu.thread.x/y/z` | `threadIdx` | thread index within block |
| `gpu.block.x/y/z` | `blockIdx` | block index within grid |
| `gpu.block_dim.x/y/z` | `blockDim` | threads per block |
| `gpu.grid_dim.x/y/z` | `gridDim` | blocks per grid |
| `sync` | `__syncthreads()` | block-level barrier keyword |

**Planned, not yet implemented** — no `warp` namespace or `atomic` namespace exists inside a kernel body today:

| Field | CUDA | Description |
|---|---|---|
| `warp.size` | `warpSize` | threads per warp (typically 32) |
| `warp.lane` | `threadIdx.x % warpSize` | thread index within warp |
| `warp.sync` | `__syncwarp()` | warp-level barrier keyword |
| `atomic.cas(ref, exp, new)` | `atomicCAS` | compare-and-swap |

The only currently implemented warp-related builtin is the host-side `GPU.warpSize()` device-property method (see [`GPU`](#gpu)) — it is not usable inside a kernel body.

`.x`, `.y`, `.z` address the three spatial dimensions. Most kernels use only `.x` (1D data); image kernels use `.x` and `.y`; volume kernels use all three.

**Namespace hierarchy inside a kernel:**

```
gpu.thread   — thread index within block
gpu.block    — block index within grid
gpu.block_dim / gpu.grid_dim — dimensions
sync         — block-level barrier keyword

# planned, not yet implemented:
warp         — subgroup of 32 threads within a block (CUDA-only)
warp.sync    — warp-level barrier keyword (CUDA-only)
atomic.cas   — explicit compare-and-swap
```

---

## Memory allocation and transfers

In `init`, `'unified` and `'global` fields are allocated with the standard array syntax:

```boring
init(int n):
    input  = [..n]           # 'unified — uninitialized, size n
    output = [0.0 for ..n]   # 'unified — filled with 0.0
    scratch = [..n]          # 'global — device-only scratch buffer
```

### Device-to-device transfers — dtod inference

When a `'unified` or `'global` field is passed directly to another kernel without
being read on the host, the transpiler emits a device-to-device copy instead of a
D2H + H2D round-trip. No explicit call needed.

```boring
mut ka = kernel(block = 256) ka |> .wait
mut kb = Gather(ka.output)        # ka.output not read on host → dtod automatically
mut kb = kernel(block = 256) kb |> .wait
let result = kb.output            # D2H only happens here
```

### Constant memory — `'gpu'const`

Read-only broadcast memory, hardware-enforced. Written from the host only. Useful for weights or LUTs shared by all threads.

```boring
# value known at compile time
let [float]'gpu'const weights = [1.0, 2.0, 3.0]
```

`'gpu'const` is incompatible with `mut` and `var` — only `let` is valid today.

**Planned, not yet implemented.** A lazy upload pattern combining the generic `lazy`/`?=` syntax with a GPU-specific constant-memory upload call:

```boring
# computed once, then frozen
lazy [float]'gpu'const weights
weights ?= gpu.const(compute_weights())   # first call — upload
weights ?= gpu.const(compute_weights())   # subsequent calls — no-op
```

`lazy` and `?=` exist generically in Boring, but `gpu.const(...)` as a callable builtin does not exist in the interpreter or transpiler yet — `GpuConst` is currently only a field-qualifier enum variant, not a callable.

---

## Multi-device

Single-GPU by default. Multi-device is opt-in via `in device` at instantiation — the device is a `GPU` value, static or dynamic.

> Device placement uses `new(arena)`. See [`new-placement.md`](new-placement.html) for the full reference.
>
> ```boring
> Scale(n)         # single GPU, qualifier inferred
> new(g0) Scale(n) # explicit device placement
> ```

```boring
# static — two named devices
let g0 = GPU(0)
let g1 = GPU(1)

mut ka = new(g0) Scale(n)    # init allocates buffers on device 0
mut kb = new(g1) Scale(n)    # init allocates buffers on device 1

mut ka = kernel(block = 256) ka |> .wait
mut kb = kernel(block = 256) kb |> .wait
```

```boring
# dynamic — distribute across N GPUs
let gpus = GPU.all()
mut [KernelHandle<Scale>] handles = []

for g in gpus:
    mut k = new(g) Scale(n)
    for i in ..n: k.input[i] = data[i]
    handles.push(kernel(block = 256) k)

for h in handles:
    mut k = h.wait
    results.push(k.output[0])
```

The device is propagated automatically to all fields allocated in `init`. A device mismatch between a kernel and its fields is a launch-time error.

Cross-device dependencies use `after =` as for single-device syntactically, but **the peer-access-aware synchronisation described below is planned, not yet implemented** — `after =` codegen in `src/transpiler/cuda/host.rs` currently only handles same-stream/handle dependencies with no device-mismatch branching, `cudaStreamWaitEvent` cross-device codegen, or peer-access detection.

```boring
let h0 = kernel(block = 256) ka
let h1 = kernel(block = 256, after = h0) kb   # waits for h0 before launching on g1
mut kb = h1.wait
```

---

## Full example — sum reduction

```boring
kernel ReduceSum:
    let [float]'unified  input       # host writes data, device read-only
    mut [float]'unified  output      # device writes results, host reads after .wait
    mut [float]'shared   tile        # block SRAM dynamic — size from smem at launch

    init(int n):
        input  = [..n]
        output = [..n / 256]

    def load(int tid, int i):
        tile[tid] = if i < len(input): input[i] else: 0.0

    def sweep():
        var s = gpu.block_dim.x / 2
        while s > 0:
            if gpu.thread.x < s:
                tile[gpu.thread.x] += tile[gpu.thread.x + s]
            sync
            s /= 2

    def ():
        let tid = gpu.thread.x
        let i   = gpu.block.x * gpu.block_dim.x + tid
        load(tid, i)
        sync
        sweep()
        if tid == 0:
            output[gpu.block.x] = tile[0]


mut k = ReduceSum(n)

# fill input — direct ('unified)
for i in ..n: k.input[i] = data[i]

mut k = kernel(block = 256, smem = {tile = 256 * 4}) k |> .wait
print k.output[0]
```

---

## `'shared` — block SRAM

Block-scoped on-chip SRAM. Fast, limited (typically 48–96 KB per SM). Lifetime: kernel invocation.

In a `kernel` struct, `'shared` fields are always internal — they cannot be passed via the constructor and cannot escape the kernel.

### Static — size known at compile time

Size is part of the type — `[float, 256]'shared` declares 256 floats of SRAM.

```boring
kernel Reduce:
    let [float]'unified      input
    mut [float]'unified      output
    mut [float, 256]'shared  tile    # 256 floats of SRAM

    init(int n):
        input  = [..n]
        output = [..n / 256]

    def ():
        let tid = gpu.thread.x
        tile[tid] = input[gpu.block.x * gpu.block_dim.x + tid]
        sync
        ...

mut k = Reduce(n)
for i in ..n: k.input[i] = data[i]
mut k = kernel(block = 256) k |> .wait   # no smem parameter needed
```

### Dynamic — size passed at launch

No size in the type. The size comes from the named `smem` launch parameter.

```boring
kernel Reduce:
    let [float]'unified  input
    mut [float]'unified  output
    mut [float]'shared   tile    # size from smem at launch

    init(int n):
        input  = [..n]
        output = [..n / 256]

    def ():
        let tid = gpu.thread.x
        tile[tid] = input[gpu.block.x * gpu.block_dim.x + tid]
        sync
        ...

mut k = Reduce(n)
for i in ..n: k.input[i] = data[i]
mut k = kernel(block = block_size, smem = {tile: block_size * 4}) k |> .wait
```

Useful when the tile size must match `block_size`, which is a runtime value.

### Multiple dynamic arrays

In CUDA, only one `extern __shared__` array is allowed per kernel — multiple arrays require manual byte-offset arithmetic. Boring removes this by naming smem partitions at the launch site:

```boring
kernel Reduce:
    let [float]'unified  input
    mut [float]'unified  output
    mut [float]'shared   tile    # partition "tile"
    mut [int]'shared     flags   # partition "flags"

    init(int n):
        input  = [..n]
        output = [..n / 256]

    def (): ...

mut k = Reduce(n)
for i in ..n: k.input[i] = data[i]
mut k = kernel(block = 256, smem = {tile: 256 * 4, flags: 64 * 4}) k |> .wait
```

The transpiler generates the offset arithmetic automatically.

`[T, N]` is a first-class Boring type (see `book.md §7 — Fixed-size arrays`).

---

## Kernel ordering and dependencies

`kernel(...)` dispatches a kernel asynchronously and returns a `KernelHandle`. The host continues immediately. Ordering between kernels is declared with `after =` — no explicit stream management.

### `after =`

Sequential ordering is declared with `after =`. A single handle or a tuple of handles — the kernel starts as soon as all listed handles have completed.

```boring
# sequential pipeline
let h1 = kernel(block = 256) ka
let h2 = kernel(block = 256, after = h1) kb   # runs after h1
h2.wait

# fork then join
let h1 = kernel(block = 256) ka
let h2 = kernel(block = 256) kb
let h3 = kernel(block = 256, after = [h1, h2]) km
h3.wait
```

`after =` replaces the need to explicitly name streams for ordering — the runtime deduces which operations can share a stream from the dependency graph.

**Cross-device dependencies (design target — planned, not yet implemented except for the same-device case):**

| Dependency | Implementation | Notes |
|---|---|---|
| same device | CUDA event | native, zero CPU involvement — implemented |
| device 0 → device 1 | `cudaStreamWaitEvent` cross-device | requires peer access — planned, not yet implemented |
| device 0 → device 1 (no peer access) | host-mediated | correct but adds CPU sync — planned, not yet implemented |
| GPU kernel → CPU | host-mediated | always possible — planned, not yet implemented |

```boring
let g0 = GPU(0)
let g1 = GPU(1)
mut ka = new(g0) StageA(n)
mut kb = new(g1) StageB(n)

let h1 = kernel(block = 256) ka
let h2 = kernel(block = 256, after = h1) kb
# peer access enabled  → CUDA cross-device event
# peer access disabled → host-mediated sync (correct, slower)
mut kb = h2.wait
```

**Compiler warnings (planned, not yet implemented):**

- Cross-device `after =` without `--peer-access` flag → warning: dependency may be host-mediated. **No `--peer-access` CLI flag is registered** in `src/main.rs` today, and no cross-device dependency detection or warning logic exists in the transpiler.
- Cyclic `after =` graph → compile-time error

### Cooperative cancellation

CUDA kernels are not preemptible — once launched, a kernel runs to completion. Cooperative cancellation requires the kernel to check a flag in device memory, and the host to write that flag while the kernel owns its buffers (which the move/wait model prevents directly).

> **To explore.** Two directions:
>
> - `@cancellable` annotation on the kernel struct — the runtime injects a `'unified` cancel flag and a `gpu.cancelled` built-in readable inside `def ()`. `h.cancel()` writes the flag from the host.
> - `Cancellable` trait implemented on a kernel struct — opt-in, explicit. Also raises the open question of trait support in `kernel` structs more generally, which has not yet been designed.
>
> Both approaches keep the opt-in character: a kernel that does not need cancellation pays no overhead.

### CUDA library interop

`gpu.stream()` is not part of the Boring GPU language — all ordering and concurrency is expressed via `kernel(...)` and `after =`. The runtime manages streams internally.

The only case where a raw `cudaStream_t` is needed is interop with external CUDA libraries (cuBLAS, cuDNN, cuFFT) that require an explicit stream handle. This is an FFI concern, not a language concept. The underlying stream of a kernel handle is accessible in unsafe/FFI context:

```boring
let h = kernel(block = 256) k
ffi::cublas_sgemm(h.native_stream(), ...)   # FFI — raw cudaStream_t
```

> **Open:** exact backend mapping — whether each `kernel(...)` dispatch owns a dedicated CUDA stream or whether the runtime manages a stream pool; how `after =` maps to CUDA events vs stream dependencies.

---

## CPU simulation mode

`boring run` always executes in simulation mode — GPU primitives are replaced with CPU equivalents. The same source file runs without a GPU, which is useful for CI, unit testing, and logic debugging.

```
boring run main.br                      # simulation — no GPU required
boring run --gpu a100 main.br           # simulation with A100 profile
boring build --target cuda main.br      # CUDA codegen — emits a Cargo project
```

No changes to the source code are required.

### GPU simulation profiles

Device property methods (`name()`, `totalMem()`, etc.) return values from a **GPU profile** — a TOML file that describes a virtual device. Select a profile with `--gpu <name>`:

```
boring run --gpu default  main.br   # generic GPU (8 GB, SM 8.6) — default
boring run --gpu v100     main.br   # Tesla V100 SXM2, 16 GB, SM 7.0
boring run --gpu a100     main.br   # A100 SXM4, 80 GB, SM 8.0
boring run --gpu rtx3090  main.br   # GeForce RTX 3090, 24 GB, SM 8.6
boring run --gpu rtx4090  main.br   # GeForce RTX 4090, 24 GB, SM 8.9
boring run --gpu h100     main.br   # H100 SXM5, 80 GB, SM 9.0
```

A custom profile is a TOML file with the following fields:

```toml
# my-gpu.toml
name = "My Custom GPU"
totalMem = 8589934592       # bytes
warpSize = 32
maxThreads = 1024
maxSharedMem = 49152        # bytes
computeCapability = [8, 6]  # [major, minor]
```

```
boring run --gpu path/to/my-gpu.toml main.br
```

Built-in profiles are embedded in the `boring` binary (no install-time path required). The profile only affects property queries — kernel logic, memory allocation, and thread simulation are identical regardless of the active profile.

### Substitution table

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
| `atomic.cas` (planned, not yet implemented) | plain compare-and-swap (no contention) |
| `kernel(...)` dispatch | sequential loop over all threads |
| `kernel(...)` streams | no-op — single-threaded sequential |
| `after =` | sequential execution in declaration order |

### Guarantees and limitations

**What simulation validates:**
- Kernel logic — index calculations, conditionals, memory access patterns
- Data flow — buffer ownership, `.wait` completion, `after =` ordering
- Host/device interaction — unified memory access, phase transitions

**What simulation does not validate:**
- **Race conditions** — sequential execution hides all data races between threads. A kernel that is correct in simulation may be incorrect on real hardware.
- **`sync` necessity** — no reordering occurs, so missing barriers are invisible.
- **`'actor'global` / `atomic.cas` correctness** — no concurrent writes, so atomics degrade to plain reads/writes without exposing contention bugs (`atomic.cas` itself is planned, not yet implemented).
- **Performance** — thousands of GPU threads execute sequentially; timings are not representative.

Simulation mode validates **logic**, not **concurrent correctness**. Race condition testing requires real GPU hardware or a dedicated thread-level simulator.

---

## Atomics — `'actor` on kernel fields

`'actor` means "safe concurrent access" — the mechanism depends on context. **Currently only `'actor'global` is implemented.** `'actor'unified`, `'actor'shared`, and a host-side `'actor'gpu'unified` are design-stage only: the parser's `'actor` qualifier accepts only `'task` or `'global` as the following suffix (see `src/parser/parse_type.rs`); any other suffix, including `'unified` and `'shared`, is a parse error today.

| Context | Qualifier | Implementation | Status |
|---|---|---|---|
| CPU host | `'actor'gpu'unified` | host/device barrier — waits for kernel completion | planned, not yet implemented |
| GPU kernel | `'actor'unified` | atomic instructions on unified memory | planned, not yet implemented |
| GPU kernel | `'actor'global` | atomic instructions on device DRAM | **implemented** |
| GPU kernel | `'actor'shared` | atomic instructions on block SRAM | planned, not yet implemented |

Inside a kernel, `'actor'global` fields generate atomic instructions automatically for compound assignment. `atomic.cas` as an explicit builtin is **planned, not yet implemented** — see below.

| Field declaration | Location | Atomic scope | Status |
|---|---|---|---|
| `mut [int]'actor'unified bins` | unified DRAM | all threads, all blocks | planned, not yet implemented |
| `mut [int]'actor'global bins` | DRAM device | all threads, all blocks | **implemented** |
| `mut [int, 256]'actor'shared local_bins` | block SRAM | threads of the same block | planned, not yet implemented |

The following example illustrates the target design once `'actor'shared` lands. Today, only the `bins` (`'actor'global`) line transpiles to an atomic; `local_bins` would need to be a plain `'shared` field with explicit `sync` instead:

```boring
kernel Histogram:
    let [float]'global              input
    mut [int]'actor'global          bins            # global — atomic across all blocks
    mut [int, 256]'actor'shared     local_bins      # atomic within block (planned, not yet implemented)

    def ():
        local_bins[gpu.thread.x] = 0
        sync

        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        if i < len(input):
            let bucket = int(input[i] * 10.0)
            local_bins[bucket] += 1   # planned: → atomicAdd on SRAM — fast, no inter-block contention

        sync
        bins[gpu.thread.x] += local_bins[gpu.thread.x]   # → atomicAdd on DRAM — once per block
```

**Operations transpiled automatically on `'actor'global` fields today:**

| Boring | CUDA | Status |
|---|---|---|
| `x += v` | `atomicAdd(&x, v)` | implemented |
| `x -= v` | `atomicSub(&x, v)` | implemented |
| `x \|= v` | `atomicOr(&x, v)` | implemented |
| `x &= v` | `atomicAnd(&x, v)` | implemented |
| `x ^= v` | `atomicXor(&x, v)` | implemented |
| `x = min(x, v)` | `atomicMin(&x, v)` | planned, not yet implemented |
| `x = max(x, v)` | `atomicMax(&x, v)` | planned, not yet implemented |
| `x = v` | `atomicExch(&x, v)` | planned, not yet implemented |

**(Planned, not yet implemented.)** Compare-and-swap as an explicit builtin — its structure (expected value + new value) has no natural operator mapping. There is currently no `atomic` namespace or `.cas` method in the lexer, parser, interpreter, or transpiler:

```boring
atomic.cas(ref, expected, new)   # planned — not yet implemented
```

**`'actor'shared` vs `sync` (planned, not yet implemented)**

`sync` coordinates threads across a full barrier — all writes visible to all threads after the barrier. The design intent is for `'actor` on a `'shared` field to allow threads to write to the **same slot** without a barrier being possible between writes, but `'actor'shared` does not exist yet — only plain `'shared` is available today:

```boring
mut [int, 256]'shared       tile    # plain — sync required between write and read
mut [int, 256]'actor'shared tile    # planned — atomic, concurrent writes to same slot safe without barrier
```

In simulate mode, `'actor` fields use plain arithmetic (no contention).

---

## Placement — `new(...)`

`new` is implemented — see [`new-placement.md`](new-placement.html) for the full reference. GPU-relevant forms:

```boring
new(g0) Scale(n)   # explicit device placement
```

---

## Error handling

**Planned, not yet implemented.** None of the typed error names below (`GpuLaunchError`, `GpuOutOfMemory`, `GpuIllegalAccess`, `GpuStackOverflow`, `GpuTimeout`, `GpuDeviceLost`) exist anywhere in the codebase today, and there is no block-size/launch-config validation in `src/validator/kernel.rs` or `src/interpreter/eval_gpu.rs` — oversized `block` values are not currently checked at all. The design below describes the intended error model.

CUDA errors fall into two categories that map to the two observation points of a `KernelHandle`.

### Synchronous errors — raised at `kernel(...)`

Detected immediately before execution begins. The handle is never created.

```boring
let h = kernel(block = 99999) k   # raise GpuLaunchError — block size exceeds limit
let h = kernel(block = 256) k     # ok — h created, kernel queued
```

### Asynchronous errors — raised at `.wait`

Detected only at synchronisation. The kernel ran but something went wrong on the device.

```boring
let h = kernel(block = 256) k
h.wait                                  # raise GpuIllegalAccess if kernel faulted
```

With natural propagation:

```boring
def [float] process(ReduceSum k) throws:
    kernel(block = 256) k |> .wait      # error propagates to caller
    k.output
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

`GpuDeviceLost` is non-recoverable — all in-flight handles are invalidated and all GPU buffers are lost. The only recovery is reinitialising the device context.

### `after =` and error propagation

If a dependency handle failed, the dependent kernel is not launched — the error propagates through the chain:

```boring
let h1 = kernel(block = 256) ka
let h2 = kernel(block = 256, after = h1) kb   # not launched if h1 failed
h2.wait                                        # raises h1's error
```

---

## Implementation roadmap

| # | Feature | Status |
|---|---|---|
| 1 | Automatic grid sizing | ✅ implemented |
| 2 | `'local` array fields (`[T, N]'local`) | ✅ implemented |
| 3 | Atomics via `'actor'global` | ✅ implemented |
| 4 | `after =` + streams | ✅ implemented |
| 5 | dtod inference (no explicit copy) | pending |
| 6 | Multi-GPU via `new(g) K(...)` + `GPU.all()` | ✅ implemented |
| 7 | 2D/3D grids via tuple `block = (Bx, By)` | ✅ implemented |
| 8 | `'shared` in `init` → validation error | ✅ implemented |
| 9 | `print` in kernel body → `printf` | ✅ implemented |
| 10 | `GPU` as built-in type with device properties | ✅ implemented |

### Pending: dtod inference

Single-pass dataflow over the statement list. If `k.field` appears only as constructor input to another kernel, emit a device-to-device copy instead of the D2H + H2D round-trip. No syntax change — entirely transparent optimisation.
