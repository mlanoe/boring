# Boring GPU — design draft

> Status: draft / exploration — nothing is set in stone yet.
>
> **Portability legend:** ✓ portable (CUDA + OpenCL) — ⚠ CUDA-only (no direct OpenCL equivalent)

---

## What GPU computing fundamentally requires

1. **Memory spaces** — global, local (on-chip), constant, unified. Storage qualifiers, not ownership qualifiers.
2. **Thread hierarchy** — thread index, block index, block dimensions — implicit built-ins inside every kernel.
3. **Kernel functions** — launched from the host, executed on the GPU.
4. **Launch configuration** — number of blocks and threads per block, passed at launch time.
5. **Barriers** — explicit synchronisation between threads, atomics.
6. **Memory transfers** — explicit host↔device copies or unified memory.

The target is **CUDA**. OpenCL is out of scope for this draft — features with no OpenCL equivalent are marked ⚠ CUDA-only for reference.

---

## Qualifier model: three axes

Boring qualifiers encode three independent dimensions simultaneously:

```
qualifier = storage location × access mode × lifetime
```

### Axis 1 — Storage location

| Qualifier | Location | Rust/CUDA | OpenCL |
|---|---|---|---|
| (implicit) | stack — inferred default for small value types | `T` | — |
| `'heap` | heap, exclusive ownership | `Box<T>` | — |
| `'shared` / `'actor` / `'guard` | heap, reference-counted | `Rc` / `Arc` + wrapper | — |
| `'gpu'shared` | managed DRAM, host + device | `cudaMallocManaged` | — ⚠ CUDA-only |
| `'gpu'global` | GPU device DRAM only | device pointer | `__global` ✓ |
| `'gpu'local` | GPU on-chip SRAM, block-scoped | `__shared__` | `__local` ✓ |
| `'gpu'const` | GPU constant cache | `__constant__` | `__constant` ✓ |

`'gpu` alone is a shorthand for `'gpu'shared` — the default qualifier gives host+device access. `'gpu'global` must be written explicitly when device-only memory is needed.

The `'gpu` prefix is mandatory for all GPU memory qualifiers. It disambiguates from Boring's CPU qualifiers and makes the device boundary explicit. A bare `'global` or `'local` would be ambiguous; `'gpu'global` and `'gpu'local` are not.

### Axis 2 — Access mode

| Qualifier | Readers | Writers | Coordination |
|---|---|---|---|
| (implicit) / `'heap` | 1 owner | 1 owner | none |
| `'shared` | N | 0 | none (immutable) |
| `'actor` | N | N | built-in (RefCell / Mutex) |
| `'guard` | N (under lock) | 1 (under lock) | explicit |
| `'gpu'shared` | host + all threads | host + all threads | `v.value` at phase boundary |
| `'gpu'global` | all threads | all threads | none built-in → atomics or sync |
| `'gpu'local` | block threads | block threads | `sync` keyword |
| `'gpu'const` | all threads | 0 | none (hardware-enforced ro) |

**Key insight:** for GPU memory, scope (who can access) is not an independent axis — it is a consequence of the physical location. `'gpu'global` is always grid-wide, `'gpu'local` is always block-scoped, registers are always thread-private. No separate scope qualifier is needed.

### Axis 3 — Lifetime

| Qualifier | Lifetime | Mechanism |
|---|---|---|
| (implicit) | owner scope | automatic — stack frame |
| `'heap` | owner scope | RAII — Rust drop |
| `'shared` / `'actor` / `'guard` | last Arc drop | reference counting |
| `'gpu'global` | host owner scope | RAII — `cudaFree` on drop |
| `'gpu'local` | kernel invocation | automatic — kernel end |
| `'gpu'const` | program lifetime | static — loaded at startup |
| `'gpu'shared` | host owner scope | RAII — `cudaFree` on drop |

### Complete three-axis table

```
qualifier       location            access mode              lifetime
──────────────  ──────────────────  ───────────────────────  ─────────────────────
(implicit)      stack               1 owner, rw              owner scope (auto)
'heap           heap                1 owner, rw              owner scope (RAII)
'shared         heap (Arc)          N readers, ro            last Arc drop
'actor          heap (Arc+Mutex)    N rw, sync built-in      last Arc drop
'guard          heap (Arc+RwLock)   N rw, sync explicit      last Arc drop

'gpu / 'gpu'shared  GPU+CPU DRAM    host + device, rw        host owner (RAII)
'gpu'global     GPU DRAM only       N threads, rw, no sync   host owner (RAII)
'gpu'local      GPU SRAM            block threads, rw         kernel invocation
'gpu'const      GPU const cache     N threads, ro             program lifetime
```

The table shows what GPU computing removes: reference counting disappears entirely.
`'gpu'shared` and `'gpu'global` are single-owner RAII handles on the host side, never shared via a counter.

**Aliases:**

```
'gpu              →  'gpu'shared           # default — host + device access
'actor'gpu        →  'actor'gpu'shared     # implicit host/device coordination
```

---

## Memory safety model

### The fundamental tension

Boring inherits Rust's memory safety guarantees, which rest on two invariants:

- **single owner** — one owner → the compiler tracks the drop → RAII works
- **shared** — multiple owners → a reference counter (`Arc`) determines when to free

GPU memory breaks both: `'gpu'global` data has a single host-side owner, but N device threads access it concurrently — with no counter, no borrow checker, no compiler-enforced coordination.

### Decision: `'gpu'global` is single-owner on the host side

The GPU buffer is a single-owner RAII handle on the host. The kernel launch borrows it for its duration — the host cannot free the buffer while the kernel runs.

Inside the kernel, device threads access the buffer concurrently. Boring cannot verify the absence of data races between threads. The `kernel` body is an **implicit unsafe zone** for device-side memory access.

`sync` and `atomic.cas` are the programmer's tools in this zone. The compiler can enforce structural rules (e.g. `'gpu'local` cannot escape the kernel body) but cannot guarantee correctness of concurrent access in the general case.

### `'gpu'local` — lifetime enforced by the type system

`'gpu'local` variables cannot escape the kernel body. The compiler rejects:

- returning a `'gpu'local` value from a kernel
- storing a `'gpu'local` value in a struct field
- passing a `'gpu'local` reference outside the kernel invocation scope

This is the one case where Boring *can* enforce a lifetime guarantee on the device side.

### `'gpu'shared` — ownership and phase transitions

`'gpu'shared` data has two active phases:

```
phase host   →  readable and writable from CPU, normal Boring rules
phase device →  readable and writable from GPU threads, implicit unsafe zone
```

**Decision: constructing a kernel moves its buffers into the kernel handle.**

The handle owns the buffers for the duration of the kernel. `.value` consumes the handle and returns the buffers. This is the only model that is safe across function boundaries — a borrow-based approach would require explicit lifetime annotations (which Boring avoids) and breaks as soon as the launch and the wait happen in different functions.

```boring
kernel Scale:
    [float]'gpu buf

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] *= 2.0

let buf = gpu.shared<float>(n)
buf[0] = 1.0                              # phase host, buf owned here

let v = Scale(buf = buf)(block = 256)     # buf moved into v
buf[0]                                    # compile-time error — buf moved into v
let (buf,) = v.value                      # v consumed, buf returned
buf[0]                                    # ok — buf owned here again
```

Multiple buffers return as a tuple:

```boring
kernel Process:
    [float]'gpu x
    [float]'gpu y
    def (): ...

let (x, y) = Process(x = x, y = y)(block = 256).value
```

**Why not borrow semantics (option C)?**

A borrow-based model (`v` holds `&mut buf`) breaks when the kernel launch and the `.wait()` call are in different functions:

```boring
def launch_work() -> KernelHandle:
    let buf = gpu.shared<float>(n)
    let v = Scale(buf = buf)(block = 256)
    v              # buf freed here — v holds a dangling &mut buf

def main():
    let v = launch_work()
    v.value        # undefined behavior — buf already freed
```

Move semantics avoid this: `buf` lives as long as `v`, regardless of where `.value` is called.

**Cross-function usage:**

```boring
def launch_work() -> KernelHandle:
    let buf = gpu.shared<float>(n)
    buf[0] = 1.0
    Scale(buf = buf)(block = 256)      # buf moved into handle, handle returned

def main():
    let v = launch_work()
    let (buf,) = v.value               # buf recovered here
    print buf[0]                        # ok
```

### `'actor'gpu'shared` — implicit coordination

`'guard'gpu` has no GPU equivalent: the host/device boundary is binary — there is no "multiple concurrent host readers while the device writes". Only `'actor` maps to GPU coordination.

`'actor'gpu'shared` makes host/device transitions implicit, the same way `'actor` makes `Mutex` locking implicit on CPU. The runtime handles synchronisation at every access; the programmer never calls `.value`.

```boring
kernel Scale:
    [float]'actor'gpu'shared buf
    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] *= 2.0

let buf = [float]'actor'gpu'shared = gpu.shared(n)

buf[0] = 1.0                         # host access — sync implicit if needed
Scale(buf = buf)(block = 256)
print buf[0]                         # waits for kernel implicitly, then host access
```

Trade-off: the synchronisation cost is hidden. `print buf[0]` may block silently waiting for the kernel. Use `'gpu'shared` with explicit `.value` when synchronisation points need to be visible in the code.

| Qualifier | Coordination | Transition |
|---|---|---|
| `'gpu'shared` | none — explicit move | `.value` |
| `'actor'gpu'shared` | implicit — at every access | automatic |

### Open: device-side safety innovation

The device-side unsafe zone is an open design space. Possible directions:

- **Phase types** — the compiler tracks whether `sync` has occurred since the last write, making unsynchronised reads a type error.
- **Thread ownership types** — a value tagged with its owning thread index; cross-thread access requires an explicit cast or barrier.
- **Warp-level ownership** — values owned by a warp rather than a thread or block; warp-synchronous access is safe by construction.

None of these exist in current GPU tooling. This is where Boring could bring something genuinely new.

---

## Kernel functions

`kernel` has two forms: **function-style** for simple kernels without shared memory or device helpers, and **struct** for complex kernels. Both are top-level declarations — the host/device boundary is always explicit before the body is read.

Inline kernels (anonymous block at the call site) are not supported: the body would look like sequential host code while executing as thousands of GPU threads, which is a source of confusion.

A kernel launch returns a `KernelHandle`. Calling `.value` (property, no parentheses — like a future) waits for completion and returns moved buffers as a tuple.

### Struct kernel

A `kernel` struct groups fields (buffers and shared memory), device-side helpers (`req`/`def` methods), and the entry point (`def ()` — the call operator).

**Field qualifier rules:**

| Field declaration | Meaning |
|---|---|
| `[float]'gpu` | buffer — host-allocated, passed via constructor |
| `[float]'gpu'global` | buffer — device-only, host transfers explicitly |
| `[float]'gpu'const` | constant — read-only on device |
| `[float, 256]` (no qualifier) | implicitly `'gpu'local` — SRAM internal to the kernel |
| `[float]` (no qualifier, no size) | implicitly `'gpu'local` dynamic — size from `smem` at launch |
| `'heap`, `'stack`, `'actor`… | compile-time error — no CPU qualifiers inside a kernel |

Fields without a `'gpu` prefix are always internal — they cannot be passed via the constructor and cannot escape the kernel. The `'gpu` prefix is the signal that a field crosses the host/device boundary.

```boring
kernel Reduce:
    [float]'gpu    input       # buffer from host
    [float]'gpu    output      # buffer from host
    [float, 256]   tile        # 'gpu'local implicit — 256 floats of SRAM

    req fill():
        let i = gpu.thread.x
        tile[i] = input[gpu.block.x * gpu.block_dim.x + i]

    req combine():
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

**Data and launch configuration are separated.**

The constructor takes data fields only. Launch configuration (`block`, `grid`, `smem`, `after`, device) is passed to `kernel(...)` at the dispatch site — it configures execution, not the kernel's data.

```boring
# construct — data only
let k = Reduce(input = data, output = result)

# dispatch
let h = kernel(block = 256) k                              # 1D
let h = kernel(block = (16, 16)) k                         # 2D, grid inferred
let h = kernel(block = (16, 16), grid = (w/16, h/16)) k   # explicit grid

# recover
let (data, result) = h.value
```

Or chained:

```boring
let (data, result) = kernel(block = 256) Reduce(input = data, output = result) |> .value
```

**`def ()` — the kernel entry point**

`def ()` is the device entry point. It receives launch configuration implicitly (`block`, `grid`, `smem` are available as built-ins inside the body). It can have additional parameters and multiple overloads:

```boring
kernel Saxpy:
    float          a
    [float]'gpu    x
    [float]'gpu    y

    def ():                              # default — full range
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        if i < len(x): y[i] += a * x[i]

    def (int offset, int length):        # overload — sub-range
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x + offset
        if i < length: y[i] += a * x[i]
```

```boring
kernel(block = 256) Saxpy(a = 2.0, x = x, y = y)
kernel(block = 256) Saxpy(a = 2.0, x = x, y = y, offset = 512, length = 1024)
```

> **Note:** `def ()` and `req ()` are general Boring concepts — callable structs / functor objects. `req ()` is callable on `let` and `var` bindings (read-only self), `def ()` only on `var` bindings (mutating self). Calling requires explicit parentheses: `obj()`. Two exceptions: `kernel obj` and `task obj` auto-invoke `def ()` without parentheses — only valid when `def ()` has no parameters. `set ()` is not permitted. Both `def ()` and `req ()` should be added to `book.md`.

### Function-style kernel

For kernels without shared memory or device helpers:

```boring
kernel saxpy(float a, [float]'gpu x, [float]'gpu y):
    let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
    if i < len(x):
        y[i] += a * x[i]

kernel(block = 256) saxpy(2.0, x, y)
```

### Dispatch parameters

All dispatch parameters are passed to `kernel(...)`.

| Parameter | Type | Description |
|---|---|---|
| `block` | int or tuple | threads per block — scalar (1D), pair (2D), triple (3D) |
| `grid` | int or tuple | blocks per grid — inferred if omitted |
| `smem` | dict | named dynamic `'gpu'local` partitions |
| `after` | handle or tuple | dependency — starts after handle(s) complete |
| `priority` | `high` / `normal` / `low` | stream scheduling priority — default `normal` |

**Device selection:**

```boring
kernel(block = 256) K(data)               # level 1 — implicit single device
kernel<0>(block = 256) K(data)            # level 2 — static device, compile-time
kernel(gpu = 0, block = 256) K(data)      # level 3 — dynamic device, runtime
```

**`block` and `grid` as tuples — 2D and 3D kernels:**

```boring
# 1D
kernel(block = 256) K(buf = buf)
kernel(block = 256, grid = 512) K(buf = buf)               # explicit grid

# 2D
kernel(block = (16, 16)) K(img = img)                      # grid inferred from img.shape
kernel(block = (16, 16), grid = (w/16, h/16)) K(img = img) # explicit grid

# 3D
kernel(block = (8, 8, 8)) K(vol = vol)
```

**Grid inference rules:**

- Type with `.shape` (e.g. `Image`, `Volume`) → `grid = ceil(shape / block)` per dimension
- `[T]'gpu` 1D → `grid = ceil(len(buf) / block)`
- Otherwise → `grid` is required; omitting it is a compile-time error

---

## GpuCtx — the execution context type

A `GpuCtx[T]` is not just a buffer — it carries the data, the device identity, and the execution context for that device. Allocation methods return a `GpuCtx[T]`, not a raw qualified array.

```boring
struct GpuCtx[T]:
    int    device          # physical device id
    [T]'gpu'shared data   # the buffer (qualifier depends on allocation method)
    # gpu field: execution context — thread, block, block_dim, grid_dim, warp_size
```

Inside a kernel, the execution context is accessed via `.gpu` on the context object:

```boring
kernel Scale:
    [float]'gpu buf

    def ():
        let gpu = buf.gpu                                      # bind context
        let i   = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf.data[i] *= 2.0
```

For the common single-device case, `gpu` is also available as a global implicit built-in inside any kernel — shorthand for the single context in scope:

```boring
kernel Scale:
    [float]'gpu buf

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x  # implicit single-device
        buf.data[i] *= 2.0
```

The execution context fields:

| Field | CUDA | OpenCL | Description |
|---|---|---|---|
| `gpu.thread.x/y/z` | `threadIdx` | `get_local_id()` | thread index within block |
| `gpu.block.x/y/z` | `blockIdx` | `get_group_id()` | block index within grid |
| `gpu.block_dim.x/y/z` | `blockDim` | `get_local_size()` | threads per block |
| `gpu.grid_dim.x/y/z` | `gridDim` | `get_num_groups()` | blocks per grid |
| `warp.size` | `warpSize` | `get_sub_group_size()` | threads per warp (typically 32) ⚠ CUDA-only |
| `warp.lane` | `threadIdx.x % warpSize` | `get_sub_group_local_id()` | thread index within warp ⚠ CUDA-only |
| `sync` | `__syncthreads()` | `barrier(...)` | block-level barrier keyword ✓ |
| `warp.sync` | `__syncwarp()` | — | warp-level barrier keyword ⚠ CUDA-only |
| `atomic.cas(ref, exp, new)` | `atomicCAS` | `atomic_cmpxchg` | compare-and-swap ✓ |

`.x`, `.y`, `.z` address the three spatial dimensions. Most kernels use only `.x` (1D data); image kernels use `.x` and `.y`; volume kernels use all three.

**Namespace hierarchy inside a kernel:**

```
gpu.thread   — thread index within block
gpu.block    — block index within grid
gpu.block_dim / gpu.grid_dim — dimensions
warp         — subgroup of 32 threads within a block (CUDA-only)
sync         — block-level barrier keyword
warp.sync    — warp-level barrier keyword (CUDA-only)
atomic.cas   — explicit compare-and-swap
```

---

## Memory allocation and transfers

### Array creation syntax

Boring uses a unified array creation syntax for both CPU and GPU. The qualifier on the binding determines the storage location.

```boring
# CPU
let [float] v = [..n]                   # uninitialized — size n
let [float] v = [0.0 for ..n]           # fill with 0.0
let [float] v = [1.0 for ..n]           # fill with 1.0
let [float] v = [f(i) for i in ..n]     # computed — i goes from 0 to n-1

# GPU — qualifier drives the allocation space
let buf'gpu         = [0.0 for ..n]     # 'gpu'shared, implicit device
let buf'gpu'global  = [0.0 for ..n]     # device-only DRAM
let buf'gpu'global'0 = [0.0 for ..n]    # device 0, static
let buf'gpu         = [f(i) for i in ..n]  # computed, transferred to device
```

`..n` is shorthand for `0..n`. The range must always start at 0 — a non-zero start would leave leading elements uninitialized:

```boring
let [float] v = [0.0 for 1..n]   # compile-time error — element 0 undefined
let [float] v = [0.0 for 0..n]   # compile-time error — write ..n instead
```

`[..n]` (no value, no `for`) allocates uninitialized memory — equivalent to `gpu.alloc<T>(n)` for GPU, useful when performance matters and the caller guarantees initialization before use.

`gpu.alloc<T>(n)`, `gpu.shared<T>(n)` — these explicit forms remain available as escape hatches but the qualifier syntax is preferred.

### Constant memory

`'gpu'const` uses `gpu.const(data)` since it requires an existing host buffer to upload:

```boring
let cbuf'gpu'const = gpu.const(host_data)
```

### `gpu.copy()` — low-level escape hatch

With `'gpu'shared` as the default qualifier, host access is direct (`buf[i] = ...`) and explicit copies are rarely needed. `gpu.copy()` remains for two specific cases:

```boring
# device-to-device — without going through host
gpu.copy(dst'gpu'global, src'gpu'global)

# update an existing 'gpu'global buffer with new host data — without reallocating
gpu.copy(dst'gpu'global, src_cpu)
```

For frequent host↔device data exchange, `'gpu'shared` is the right qualifier — not `'gpu'global` + `gpu.copy()`.

---

## Multi-device

Multi-GPU usage is almost always statically structured — device assignments are fixed at program start, even when the device id is not known at compile time. Three levels of device tracking are available, chosen per project:

### Level 1 — Single-device (default)

`'gpu` carries no device information. No tracking, no checks. The common case.

```boring
let x = gpu.alloc<float>(n)   # [float]'gpu — device implicit, no tracking
```

Compilation flag `--single-gpu` locks this behaviour globally: all device qualifiers are treated as single-device, all runtime checks are removed.

### Level 2 — Static multi-device ⚠ CUDA-only

Device id is a compile-time qualifier. `[float]'gpu'0` and `[float]'gpu'1` are distinct types. Mismatch is a compile-time error.

`gpu<N>` is the generic device namespace for static multi-device allocation. The device index is a compile-time constant in the generic parameter, not a qualifier on the `gpu` identifier.

```boring
let x = gpu<0>.alloc<float>(n)   # [float]'gpu'0
let y = gpu<1>.alloc<float>(n)   # [float]'gpu'1

kernel(x, y, block = 256)        # compile-time error — 'gpu'0 ≠ 'gpu'1
```

Use when the device topology is fixed in the source code (e.g. always GPU 0 for input, GPU 1 for output).

### Level 3 — Dynamic multi-device ⚠ CUDA-only

Device id is a runtime value. `GPU(id)` creates a device handle; buffers allocated on it carry the id at runtime. Mismatch is caught before the kernel launch.

```boring
kernel Scale:
    [float]'gpu buf
    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] *= 2.0

let g1 = GPU(0)
let g2 = GPU(1)

let buf1 = g1.alloc<float>(n)   # [float]'gpu, device=0 at runtime
let buf2 = g2.alloc<float>(n)   # [float]'gpu, device=1 at runtime

let v1 = Scale(buf = buf1)(block = 256)
let v2 = Scale(buf = buf2)(block = 256)

let (buf1,) = v1.value
let (buf2,) = v2.value

kernel Merge:
    [float]'gpu x
    [float]'gpu y
    def (): ...

Merge(x = buf1, y = buf2)(block = 256)   # runtime error before launch — device mismatch
```

Use when device id comes from config, CLI args, or MPI rank.

### Compilation flags

| Flag | Effect |
|---|---|
| `--single-gpu` | All `'gpu'N` treated as `'gpu`; all device checks removed |
| `--multi-gpu=static` | Enables `'gpu'N` qualifiers; mismatch is compile-time error |
| `--multi-gpu=dynamic` | Enables `GPU(id)` runtime handles; mismatch caught pre-launch |

---

## Full example — sum reduction

```boring
kernel ReduceSum:
    [float]'gpu   input
    [float]'gpu   output
    [float]       tile              # 'gpu'local implicit — sized by smem at launch

    req load(int tid, int i):
        tile[tid] = if i < len(input): input[i] else: 0.0

    req sweep():
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


let data   = gpu.alloc<float>(n)
let result = gpu.alloc<float>(n / 256)

# ... fill data ...

let h = kernel(block = 256, smem = {tile: 256 * 4}) ReduceSum(input = data, output = result)
let (data, result) = h.value
print result[0]
```

---

## `'gpu'const` — constant memory

Read-only on device (hardware-enforced), broadcast cache. Written from host only. Two cases, two binding keywords.

### Static — value known at compile time

```boring
let [float]'gpu'const w = [1.0, 2.0, 3.0]   # embedded in binary
```

`let` binding — eager initialization, program lifetime.

### Runtime — computed once, then frozen

```boring
lazy [float]'gpu'const w
w ?= gpu.const(compute_weights())   # first call — cudaMemcpyToSymbol
w ?= gpu.const(compute_weights())   # subsequent calls — no-op
```

`lazy` binding — deferred initialization via `?=`, immutable after first assignment.
Plain `=` on a `lazy` binding is a compile-time error.

### Why `lazy` and not `transient` + `?=`

`transient` makes a field writable from `req` methods — it is always `var`-like and allows plain `=` reassignment. It does not enforce write-once. `lazy` is the right binding keyword: it forces `?=` as the only valid initializer and becomes immutable after the first assignment. The write-once guarantee is then in the type system, not a convention.

### Binding constraints

`'gpu'const` is incompatible with `mut` and `var` — mutable bindings would allow re-upload while a kernel is reading the data:

```boring
let  [float]'gpu'const w = gpu.const(data)   # ok
lazy [float]'gpu'const w                      # ok
mut  [float]'gpu'const w = gpu.const(data)   # compile-time error
var  [float]'gpu'const w = gpu.const(data)   # compile-time error
```

### `lazy` — language-level addition

`lazy` is a general Boring binding keyword, not specific to GPU. It fits the existing binding table:

| Binding | Rebindable | Mutable | Initialization |
|---|---|---|---|
| `let` | no | no | immediate — `=` required |
| `mut` | no | yes | immediate — `=` required |
| `var` | yes | yes | immediate — `=` required |
| `lazy` | no | no | deferred — `?=` required, immutable after first assignment |

> `lazy` is an existing Boring binding keyword — see `book.md` for the reference.

---

## `'gpu'local` — on-chip shared memory

Block-scoped SRAM. Fast, limited (typically 48–96 KB per SM). Lifetime: kernel invocation.

In a `kernel` struct, fields without a `'gpu` prefix are implicitly `'gpu'local` — they are internal to the kernel and cannot be passed via the constructor. The explicit `'gpu'local` qualifier is only needed in function-style kernels.

### Static — size known at compile time

Size is part of the type — `[float, 256]` declares 256 floats of SRAM.

```boring
kernel Reduce:
    [float]'gpu    buf
    [float, 256]   tile        # 'gpu'local implicit — 256 floats of SRAM

    def ():
        tile[gpu.thread.x] = buf[gpu.thread.x + gpu.block.x * gpu.block_dim.x]
        sync
        ...

Reduce(buf = buf)(block = 256)           # no smem parameter needed
```

### Dynamic — size passed at launch

No size in the type. The size comes from the named `smem` launch parameter.

```boring
kernel Reduce:
    [float]'gpu   buf
    [float]       tile         # 'gpu'local implicit — size from smem at launch

    def ():
        tile[gpu.thread.x] = buf[gpu.thread.x + gpu.block.x * gpu.block_dim.x]
        sync
        ...

Reduce(buf = buf)(block = block_size, smem = {tile: block_size * 4})
```

Useful when the tile size must match `block_size`, which is a runtime value.

### Multiple dynamic arrays

In CUDA, only one `extern __shared__` array is allowed per kernel — multiple arrays require manual byte-offset arithmetic. Boring removes this by naming smem partitions at the launch site:

```boring
kernel Reduce:
    [float]'gpu   buf
    [float]       tile         # partition "tile"
    [int]         flags        # partition "flags"

    def (): ...

Reduce(buf = buf)(block = 256, smem = {tile: 256 * 4, flags: 64 * 4})
```

The transpiler generates the offset arithmetic automatically.

### Syntax rule

| Context | Static form | Dynamic form |
|---|---|---|
| kernel struct field | `[float, 256]` | `[float]` |
| function-style kernel | `gpu.local<float, 256>()` | `gpu.local<float>()` |

In a struct field, the size is part of the type — `[float, 256]` is a fixed-size array, distinct from `[float]` which is a `Vec<float>` in normal Boring. In a function-style kernel, the expression form is required since there is no field declaration site.

`gpu.local<float>(256)` — size as a call argument — is **rejected** in both contexts. A runtime value in argument position would make the static/dynamic distinction ambiguous.

> **Note:** `[T, N]` is now a first-class type in Boring, generalised beyond the GPU module. It maps to `[T; N]` in Rust (stack-allocated fixed-size array). It can be used in any context — struct fields, function parameters, local bindings — not only in kernel code.

---

## Concurrency — `task`, pools, and dependencies

### `task` as the universal concurrency primitive

Boring's `task` is the single concurrency primitive for both CPU and GPU work. On CPU, a task runs on the async runtime. On GPU, a task maps to a CUDA stream. The surface syntax is identical.

```boring
# CPU — task unchanged
let t1 = task compute_a(data1)
let t2 = task compute_b(data2)

# GPU — kernel keyword for dispatch, two implicit streams
let h1 = kernel(block = 256) KernelA(data1)
let h2 = kernel(block = 256) KernelB(data2)
```

`task` remains the CPU concurrency primitive, unchanged. `kernel(...)` is the GPU dispatch keyword — it carries GPU-specific parameters (`block`, `grid`, `smem`, `after`, device) that have no meaning on CPU.

### Dependencies — `after =`

Sequential ordering is declared with `after =`. A single handle or a tuple of handles — the kernel starts as soon as all listed handles have completed.

```boring
# sequential pipeline
let h1 = kernel(block = 256) StageA(data)
let h2 = kernel(block = 256, after = h1) StageB(data)   # runs after h1
let (data,) = h2.value

# fork then join
let h1 = kernel(block = 256) KernelA(buf1)
let h2 = kernel(block = 256) KernelB(buf2)
let h3 = kernel(block = 256, after = (h1, h2)) Merge(buf1, buf2)
let (buf1, buf2) = h3.value
```

`after =` replaces the need to explicitly name streams for ordering — the runtime deduces which operations can share a stream from the dependency graph.

**Cross-device dependencies ⚠ CUDA-only:**

| Dependency | Implementation | Notes |
|---|---|---|
| same device | CUDA event | native, zero CPU involvement |
| `kernel<0>` → `kernel<1>` | `cudaStreamWaitEvent` cross-device | requires peer access |
| `kernel<0>` → `kernel<1>` (no peer access) | host-mediated | correct but adds CPU sync |
| CPU task → GPU kernel | host-mediated | always possible |
| GPU kernel → CPU task | host-mediated | always possible |

```boring
let h1 = kernel<0>(block = 256) KernelA(buf0)
let h2 = kernel<1>(block = 256, after = h1) KernelB(buf1)
# peer access enabled  → CUDA cross-device event
# peer access disabled → host-mediated sync (correct, slower)
```

**Compiler warnings:**

- Cross-device `after =` without `--peer-access` flag → warning: dependency may be host-mediated
- Cyclic `after =` graph → compile-time error

### `duration` and `deadline` — not supported on GPU kernels

CUDA kernels are not preemptible. Once launched, a kernel runs to completion — there is no CUDA API to cancel it cleanly. The only exit is destroying the entire GPU context (`cudaDeviceReset`), which invalidates all buffers and streams in flight.

`duration` and `deadline` are a **compile-time error** on `kernel(...)`:

```boring
kernel(block = 256, deadline = 100ms) KernelA(buf)
# compile-time error — deadline is not enforceable on GPU kernels
```

These parameters remain valid on CPU `task`, where cancellation is supported.

**Workaround — cooperative cancellation via `'gpu'shared`:**

```boring
let cancel = gpu.shared<bool>(1)
cancel[0] = false

kernel LongKernel:
    [float]'gpu       buf
    [bool]'gpu'shared cancel

    def ():
        var i = gpu.thread.x
        while i < len(buf):
            if cancel[0]: return
            buf[i] = compute(buf[i])
            i += gpu.block_dim.x * gpu.grid_dim.x

let h = kernel(block = 256) LongKernel(buf = buf, cancel = cancel)
cancel[0] = true        # request cancellation from host
let (buf,) = h.value
```

### CUDA library interop ⚠ CUDA-only

`gpu.stream()` is not part of the Boring GPU language — all ordering and concurrency is expressed via `kernel(...)` and `after =`. The runtime manages streams internally.

The only case where a raw `cudaStream_t` is needed is interop with external CUDA libraries (cuBLAS, cuDNN, cuFFT) that require an explicit stream handle. This is an FFI concern, not a language concept. The underlying stream of a kernel handle is accessible in unsafe/FFI context:

```boring
let h = kernel(block = 256) MyKernel(data)
ffi::cublas_sgemm(h.native_stream(), ...)   # FFI — raw cudaStream_t
```

### `task` with callable structs — auto-invocation of `def ()`

When a struct has a `def ()` with no parameters, `task obj` is a shorthand for `task obj()` — the call operator is invoked automatically. If `def ()` takes parameters, the explicit form is required.

```boring
kernel Fill:
    [float]'gpu buf
    float        val

    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] = val

let k = Fill(buf = data, val = 1.0)

let h = kernel(block = 256) k   # explicit
let h = kernel k                 # shorthand — only valid when def () has no params
```

This rule applies to any callable struct in Boring, not only GPU kernels.

> **Open:** exact backend mapping — whether each `kernel(...)` dispatch owns a dedicated CUDA stream or whether the runtime manages a stream pool; how `after =` maps to CUDA events vs stream dependencies.

---

## CPU simulation mode

`--gpu=simulate` replaces all GPU primitives with CPU equivalents. The same source file runs without a GPU — useful for CI, unit testing, and logic debugging.

```
boring run --gpu=simulate main.br
boring build --gpu=simulate
```

No changes to the source code are required.

### Substitution table

| GPU primitive | CPU simulation |
|---|---|
| `'gpu'global`, `'gpu'shared` | heap allocation (`Vec<T>`) |
| `'gpu'local` | local array (stack or heap) |
| `'gpu'const` | `let` binding |
| `gpu.alloc<T>(n)` | `Vec<T>::with_capacity(n)` |
| `gpu.shared<T>(n)` | `Vec<T>::with_capacity(n)`, direct access |
| kernel launch | sequential loop over all threads |
| `gpu.thread.x/y/z`, `gpu.block.x/y/z`… | loop variables |
| `sync` | no-op |
| `'actor` fields | plain arithmetic (no contention) |
| `atomic.cas` | plain compare-and-swap (no contention) |
| `kernel(...)` dispatch | sequential loop over all threads |
| `kernel(...)` streams | no-op — single-threaded sequential |
| `after =` | sequential execution in declaration order |

### Exceptions lifted in simulate mode

`duration` and `deadline` are a compile-time error on `kernel(...)` in normal mode. In simulate mode, kernel dispatch becomes sequential CPU execution and these parameters are valid again.

### Guarantees and limitations

**What simulation validates:**
- Kernel logic — index calculations, conditionals, memory access patterns
- Data flow — buffer ownership, `.value` recovery, `after =` ordering
- Host/device interaction — unified memory access, phase transitions

**What simulation does not validate:**
- **Race conditions** — sequential execution hides all data races between threads. A kernel that is correct in simulation may be incorrect on real hardware.
- **`sync` necessity** — no reordering occurs, so missing barriers are invisible.
- **`'actor` / `atomic.cas` correctness** — no concurrent writes, so atomics degrade to plain reads/writes without exposing contention bugs.
- **Performance** — thousands of GPU threads execute sequentially; timings are not representative.

Simulation mode validates **logic**, not **concurrent correctness**. Race condition testing requires real GPU hardware or a dedicated thread-level simulator.

---

## Atomics — `'actor` on GPU fields

`'actor` on a kernel struct field generates atomic instructions automatically — the same concept as `'actor` on CPU (Mutex), applied to GPU thread coordination. No explicit atomic calls in user code — only `atomic.cas` remains explicit.

| Field declaration | Location | Atomic scope |
|---|---|---|
| `[int]'gpu'actor` | DRAM global | all threads, all blocks |
| `[int, 256]'actor` | SRAM (implicit `'gpu'local`) | threads of the same block |

```boring
kernel Histogram:
    [float]'gpu      input
    [int]'gpu'actor  bins            # global — atomic across all blocks
    [int, 256]'actor local_bins      # 'gpu'local'actor — atomic within block

    def ():
        local_bins[gpu.thread.x] = 0
        sync

        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        if i < len(input):
            let bucket = int(input[i] * 10.0)
            local_bins[bucket] += 1   # → atomicAdd on SRAM — fast, no inter-block contention

        sync
        bins[gpu.thread.x] += local_bins[gpu.thread.x]   # → atomicAdd on DRAM — once per block
```

**Operations transpiled automatically:**

| Boring | CUDA |
|---|---|
| `x += v` | `atomicAdd(&x, v)` |
| `x -= v` | `atomicSub(&x, v)` |
| `x = min(x, v)` | `atomicMin(&x, v)` |
| `x = max(x, v)` | `atomicMax(&x, v)` |
| `x = v` | `atomicExch(&x, v)` |

Compare-and-swap remains explicit — its structure (expected value + new value) has no natural operator mapping:

```boring
atomic.cas(ref, expected, new)   # only explicit atomic remaining
```

**`'actor` on `'gpu'local` vs `sync`**

`sync` coordinates threads across a full barrier — all writes visible to all threads after the barrier. `'actor` on a `'gpu'local` field is useful when threads write to the **same slot** without a barrier being possible between writes:

```boring
[int, 256]       tile    # plain 'gpu'local — sync required between write and read
[int, 256]'actor tile    # atomic — concurrent writes to same slot safe without barrier
```

In simulate mode, `'actor` fields use plain arithmetic (no contention).

---

## Error handling

CUDA errors fall into two categories that map to the two observation points of a `KernelHandle`.

### Synchronous errors — raised at `kernel(...)`

Detected immediately before execution begins. The handle is never created.

```boring
let h = kernel(block = 99999) K(data)   # raise GpuLaunchError — block size exceeds limit
let h = kernel(block = 256) K(data)     # ok — h created, kernel queued
```

### Asynchronous errors — raised at `.value`

Detected only at synchronisation. The kernel ran but something went wrong on the device.

```boring
let h = kernel(block = 256) K(data)
let (data,) = h.value                   # raise GpuIllegalAccess if kernel faulted
```

With natural propagation:

```boring
def process() -> [float]:
    let h = kernel(block = 256) K(data)
    let (data,) = h.value               # error propagates to caller
    data
```

### Error types

| Error | Phase | Cause |
|---|---|---|
| `GpuLaunchError` | `kernel(...)` | invalid config — block size, grid size |
| `GpuOutOfMemory` | `kernel(...)` | allocation failed |
| `GpuIllegalAccess` | `.value` | out-of-bounds, invalid pointer |
| `GpuStackOverflow` | `.value` | kernel recursion too deep |
| `GpuTimeout` | `.value` | OS watchdog killed the kernel |
| `GpuDeviceLost` | `.value` | device reset or crash |

`GpuDeviceLost` is non-recoverable — all in-flight handles are invalidated and all GPU buffers are lost. The only recovery is reinitialising the device context.

### `after =` and error propagation

If a dependency handle failed, the dependent kernel is not launched — the error propagates through the chain:

```boring
let h1 = kernel(block = 256) StageA(data)
let h2 = kernel(block = 256, after = h1) StageB(data)   # not launched if h1 failed
let (data,) = h2.value                                   # raises h1's error
```

---

## Open questions

- **`kernel(...)` × CUDA stream mapping** — streams are hidden from the language. Whether the runtime uses a dedicated stream per dispatch or a pool is an implementation detail, not a language design question.
