# wgpu backend

`boring build --target wgpu` generates a Rust + WGSL project that runs GPU kernels via the [wgpu](https://wgpu.rs) crate — a cross-platform GPU abstraction that runs on DirectX 12 (Windows), Vulkan (Windows / Linux), and Metal (macOS).

The Boring source is identical: the same `kernel` structs, qualifiers, and `gpu.*` built-ins work unchanged across CUDA, Metal, and wgpu targets.

---

## Motivation

CUDA requires an NVIDIA GPU and the CUDA toolkit. Metal requires macOS. wgpu runs on any modern GPU on Windows, Linux, and macOS — no vendor lock-in, no external toolchain.

| Backend | OS | GPU |
|---|---|---|
| CUDA | Windows / Linux | NVIDIA only |
| Metal | macOS only | Apple / Intel Mac |
| **wgpu** | Windows / Linux / macOS | Any DirectX 12, Vulkan, or Metal GPU |

On Windows, wgpu uses DirectX 12 by default and falls back to Vulkan if DX12 is unavailable.

---

## Qualifier mapping

| Boring qualifier | CUDA C | MSL address space | **WGSL / wgpu** |
|---|---|---|---|
| `'unified` | `cudaMallocManaged` | `device` + `MTLStorageMode.shared` | `storage` buffer, `STORAGE \| COPY_SRC \| COPY_DST` — host-visible via a staging-buffer copy, **not** `MAP_READ`/`MAP_WRITE` on this buffer directly (see "Host access" below) |
| `'global` | `cudaMalloc` | `device` | `storage` buffer, GPU-only |
| `'surface` | `cudaMallocManaged` (u32) | `device uint*` | `storage` buffer of `u32`, host-visible — same layout as `'unified` |
| bare `'actor` | `__shared__` | `threadgroup` | `var<workgroup>` |
| `'local` | registers | thread-private | `var<function>` |
| `'const` scalar | `__constant__ T name;` | `constant T* [[buffer(N)]]` | `var<uniform>` in a dedicated uniform buffer |
| `'const` fixed array (`[T, N]`) | `__constant__ T name[N];` | `constant T* [[buffer(N)]]` | `var<uniform>` array in a dedicated uniform buffer |
| `'actor'global` | `atomicAdd` etc. | `atomic_fetch_add_explicit` | `atomic<i32>` / `atomic<u32>` fields in `storage` buffer, `STORAGE \| COPY_SRC \| COPY_DST` |
| `'actor'unified` | `atomicAdd` etc. | `atomic_fetch_add_explicit` | same `atomic<i32>`/`atomic<u32>` storage buffer as `'actor'global` — host-visible via the same staging-buffer copy path as `'unified` |

---

## Built-in mapping

| Boring | CUDA C | MSL | **WGSL** |
|---|---|---|---|
| `gpu.thread.x/y/z` | `threadIdx.x/y/z` | `thread_position_in_threadgroup.x/y/z` | `@builtin(local_invocation_id).x/y/z` |
| `gpu.block.x/y/z` | `blockIdx.x/y/z` | `threadgroup_position_in_grid.x/y/z` | `@builtin(workgroup_id).x/y/z` |
| `gpu.block_dim.x/y/z` | `blockDim.x/y/z` | `threads_per_threadgroup.x/y/z` | `@builtin(local_invocation_size).x/y/z` |
| `gpu.grid_dim.x/y/z` | `gridDim.x/y/z` | `threadgroups_per_grid.x/y/z` | `@builtin(num_workgroups).x/y/z` |
| `sync` (manual) | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | `workgroupBarrier()` |
| bare-`'actor` auto-barrier | inserted before first loop + at top of each loop iteration | idem | idem |
| atomics (`'actor'global`/`'actor'unified`) | `atomicAdd` etc. | `atomic_fetch_add_explicit` | `atomicAdd` / `atomicSub` / `atomicOr` / `atomicAnd` / `atomicXor` |

### WGSL shader signature

WGSL uses `@group` / `@binding` indices for all buffers, and built-in attributes for thread/block indices. The transpiler assigns binding indices automatically in declaration order.

```wgsl
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id)   tid:     vec3<u32>,
    @builtin(workgroup_id)          bid:     vec3<u32>,
    @builtin(local_invocation_size) bdim:    vec3<u32>,
) {
    let i = tid.x + bid.x * bdim.x;
    buf[i] = buf[i] * 2.0;
}
```

---

## Workgroup size and pipeline overrides

In CUDA and Metal, the block size is passed at dispatch time. WGSL encodes the workgroup size as a static shader annotation (`@workgroup_size`). The transpiler bridges this gap using **pipeline overrides** — a WGSL 1.0 feature that fixes a constant at pipeline-creation time without recompiling the shader source:

```wgsl
override block_x: u32 = 256;
override block_y: u32 = 1;
override block_z: u32 = 1;

@compute @workgroup_size(block_x, block_y, block_z)
fn main(...) { ... }
```

The Rust host code sets the override values when creating the compute pipeline, matching the `block =` argument from the Boring `kernel:` block:

```rust
// boring: k(block = 256)
device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    // ...
    compilation_options: wgpu::PipelineCompilationOptions {
        constants: &[("block_x", 256.0), ("block_y", 1.0), ("block_z", 1.0)]
            .into_iter().collect(),
        ..Default::default()
    },
    ..Default::default()
});
```

This means a single WGSL shader covers all block sizes — no shader duplication.

---

## WGSL compilation

WGSL is compiled by wgpu at runtime using the `naga` compiler (bundled in the wgpu crate). No external toolchain is required. The generated project has no `build.rs`.

An optional AOT path (`boring build --target wgpu --aot`) may be added in a future iteration to emit pre-compiled SPIR-V or DirectX bytecode for workloads where startup latency matters.

---

## Scalar uniform fields

All scalar kernel fields (`let float alpha`, `var float t`, `var Dimension dim`, inferred-`'const` scalars) are packed into a single generated WGSL struct bound as a `uniform` buffer:

```wgsl
struct KernelParams {
    t:     f32,
    dim_w: u32,
    dim_h: u32,
    alpha: f32,
}
@group(0) @binding(N) var<uniform> params: KernelParams;
```

The binding index `N` follows all `'unified`, `'global`, `'actor'global`, and `'actor'unified` array bindings. The Rust host writes the struct before every dispatch via `queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params))`.

Push constants are not used — they require the `PUSH_CONSTANTS` feature flag (off by default in wgpu) and are limited to 128 bytes on DX12/Vulkan, which conflicts with the goal of zero-configuration portability.

`Dimension` fields are flattened into two `u32` fields (`dim_w`, `dim_h`) in the params struct — WGSL structs in uniform buffers require explicit padding rules, and a two-field flat layout avoids any alignment ambiguity.

---

## `gpu.copy()` — staging buffer pattern

`'global` buffer fields are GPU-only (`STORAGE | COPY_DST`). Host reads and writes go through staging buffers. The transpiler emits two helpers in `src/main.rs`:

```rust
fn __boring_gpu_copy_d2h(device, queue, src: &wgpu::Buffer, dst: &mut [u8]);
fn __boring_gpu_copy_h2d(device, queue, src: &[u8], dst: &wgpu::Buffer);
```

**D2H** (`gpu.copy(k.result, host_buf)`):
1. Allocate staging buffer `MAP_READ | COPY_DST`, same size as `src`.
2. `encoder.copy_buffer_to_buffer(src → staging)`, submit.
3. `device.poll(Wait)`.
4. `slice.map_async(Read)` → memcpy to `dst` → `staging.unmap()`.

**H2D** (`gpu.copy(host_buf, k.input)`):
1. Allocate staging buffer `MAP_WRITE | COPY_SRC`, same size as `dst`.
2. `slice.map_async(Write)` → memcpy from `src` → `staging.unmap()`.
3. `encoder.copy_buffer_to_buffer(staging → dst)`, submit.

Staging buffers are created on demand per call and not pooled (initial implementation). A reusable pool over a pre-allocated staging arena can be added later without changing the Boring source.

---

## `'unified` vs `'global` on wgpu

wgpu buffers are not implicitly host-visible. The transpiler uses different wgpu `BufferUsages` flags depending on the qualifier:

| Qualifier | `BufferUsages` | Host read/write |
|---|---|---|
| `'unified` | `STORAGE \| COPY_SRC \| COPY_DST` | `copy_{field}_to_host`/`copy_{field}_to_device`, via a staging buffer (see "`gpu.copy()`" above) |
| `'global` | `STORAGE \| COPY_DST` | via `gpu.copy()` (staging buffer) |
| `'actor'global` | `STORAGE \| COPY_SRC \| COPY_DST` | via `gpu.copy()` (staging buffer) — device-only |
| `'actor'unified` | `STORAGE \| COPY_SRC \| COPY_DST` | `copy_{field}_to_host`/`copy_{field}_to_device`, same staging-buffer path as `'unified` |

`'unified`'s own storage buffer never carries `MAP_READ`/`MAP_WRITE` — host access
always goes through a separate staging buffer (`MAP_READ | COPY_DST` for reads,
`MAP_WRITE | COPY_SRC` for writes — see "`gpu.copy()`" above), copied to/from the
real storage buffer with `copy_buffer_to_buffer`. This is deliberate, not an
implementation gap: WebGPU's buffer-usage validation restricts `MAP_READ` to pairing
only with `COPY_DST` and `MAP_WRITE` only with `COPY_SRC`, so a single buffer
carrying `STORAGE | MAP_READ | MAP_WRITE` together (an earlier draft of this table
claimed exactly that combination) would not validate — the two-buffer staging design
avoids the question entirely rather than relying on an unverified combination.
`'actor'unified` reuses the identical staging-buffer path, which is also why it
doesn't inherit any WGSL `atomic<T>` + direct-mapping risk: the `atomic<i32>`/
`atomic<u32>` storage buffer is never itself asked to carry `MAP_READ`/`MAP_WRITE`.

---

## `GPU` type on wgpu

Implemented as a **single simulated device**, not real multi-adapter support: wgpu already only ever opens one adapter at program startup (for `device`/`queue`), so `GPU(n)` and every element of `GPU.all()` resolve to that same adapter regardless of `n` — this exists so `GPU`-introspecting source (e.g. `examples/saxpy.br`'s `let g = GPU(0); print g.name()`) is portable between the interpreter's simulation mode (also a single mock device, `Value::GpuDevice`) and `--target wgpu`, without requiring genuine multi-device selection.

| Boring method | wgpu API | Notes |
|---|---|---|
| `GPU(n)` | — | a plain `usize` index; not a real device selector |
| `GPU.all()` | — | always a single-element array, `[GPU(0)]` |
| `.name()` | `adapter.get_info().name` | real adapter name |
| `.totalMem()` | — | always `0` — `wgpu::AdapterInfo` has no memory-size field on any backend (checked against `wgpu-types` 22.0.0's struct definition) |
| `.freeMem()` | — | always `0`, same reason |
| `.computeCapability()` | — | always `[0, 0]` — a CUDA-only concept |
| `.warpSize()` | — | always `32` — a conservative default, not queryable via wgpu |
| `.maxThreads()` | `limits.max_compute_invocations_per_workgroup` | real adapter limit |
| `.maxSharedMem()` | `limits.max_compute_workgroup_storage_size` | real adapter limit, bytes per workgroup |
| `.index()` | — | echoes back whatever index was passed to `GPU(n)`, even though it isn't a real per-device index |

Real multi-device support (a distinct adapter per index, like CUDA's `CudaContext::new(idx)`/Metal's per-index `MTLDevice`) is **not implemented** — see "Known limitations vs CUDA" below.

---

## `after =` ordering

wgpu uses a single command queue per device. Command buffers submitted to the same queue execute in order. The transpiler maps `after =` to submission ordering — kernels with `after =` dependencies are submitted in a separate `submit` call after the dependency's `submit` has been flushed.

This is not GPU-side pipelining (unlike CUDA streams), but it preserves the ordering semantics with no CPU round-trip within a `kernel:` block.

---

## Error handling

Each kernel struct's `dispatch()` opens a WebGPU validation error scope (`push_error_scope(wgpu::ErrorFilter::Validation)`) before encoding, and checks it (`pop_error_scope()`, bridged to sync via the same `pollster` already used for device/adapter setup) right after submit — returning `Result<(), Box<dyn std::error::Error + Send + Sync>>` instead of `()`. A regular `kernel:` block's dispatch call propagates that error via `?` into `boring_main()`'s own `Result` (synthesized as `Result`-returning whenever any kernel is involved, even with no explicit `throws`); inside a `Screen` program's render loop (a plain closure, not a `Result`-returning context) it's `.expect(...)` instead.

Previously, no error-scope/callback machinery existed at all: `dispatch()` returned `()` unconditionally and a rejected launch configuration (e.g. an invalid workgroup count) went completely unobserved.

Validation is checked at command-buffer *encoding* time (synchronously, on the CPU, as `dispatch_workgroups` is called) — no `device.poll()` is needed to catch it. This does **not** cover execution-time faults (out-of-bounds access, device loss), which WebGPU instead reports via device-lost callbacks/uncaptured-error events, not scoped validation; those aren't wired up.

---

## Device-to-device chaining

Feeding one kernel's output directly into another kernel's constructor (`Scale(k1.buf)`) copies the buffer via a generated `__boring_gpu_copy_d2d` helper — allocate a fresh `wgpu::Buffer` and issue a `copy_buffer_to_buffer` command on the shared queue:

```rust
fn __boring_gpu_copy_d2d(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer) -> std::sync::Arc<wgpu::Buffer> {
    let size = src.size();
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src, 0, &dst, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    std::sync::Arc::new(dst)
}
```

No manual `device.poll()`/wait is needed around the copy itself — it's a GPU command, and wgpu guarantees submission order within the same queue, so a later kernel dispatch that reads `dst` is correctly ordered after this copy completes.

This is deliberately **not** `Arc::clone(&wgpu::Buffer)`: cloning the `Arc` only bumps a reference count on the same underlying `wgpu::Buffer` — it does not allocate new GPU memory or copy any bytes. Using `Arc::clone` here used to mean two kernel structs silently shared the exact same buffer; if the source kernel was ever dispatched again afterward, the "copy"'s contents changed too, with no compile error and no warning (the same class of bug as the Metal backend's `Buffer::clone()`, and worse than the CUDA/ROCm backends' bug, which at least surfaces as a real `E0382` compile error).

---

## Type mapping

WGSL does not support 64-bit integers. Inside a `kernel` struct and its `def` body, Boring types are narrowed as follows:

| Boring type | Host Rust type | WGSL kernel type |
|---|---|---|
| `int` | `i64` | `i32` |
| `uint` | `u64` | `u32` |
| `float` | `f64` | `f32` |
| `bool` | `bool` | `u32` (see note) |

The narrowing is **silent for variables and fields** — the transpiler emits `i32` without warning. Integer literals inside a kernel body are checked at compile time: a literal outside the `i32` range (−2 147 483 648 to 2 147 483 647) is a compile error.

```boring
kernel Bad:
    let int n = 3_000_000_000    # compile error: literal 3000000000 exceeds i32 range on wgpu target
```

Arithmetic overflow in WGSL wraps silently (two's complement). This matches CUDA C `int` behavior on GPU.

### `bool` in storage buffers

WGSL forbids `bool` inside `storage` and `uniform` buffers (`array<bool>` is invalid). The transpiler maps `bool` fields and `bool` array elements to `u32` in all GPU memory contexts (`'unified`, `'global`, bare `'actor`, struct fields). `true` → `1u`, `false` → `0u`. Comparisons and conditionals that receive a `u32` from a GPU field coerce back to WGSL `bool` via `!= 0u` automatically.

---

## Generic kernel monomorphisation

Generic kernels (`kernel Blur<int N>:`) are specialised at `boring build --target wgpu` time — the transpiler scans the program for all `Name<arg, ...>()` instantiations and emits one WGSL struct + entry point per unique argument list.

### Naming scheme

| Boring instantiation | WGSL entry point | WGSL params struct |
|---|---|---|
| `Blur<3>()` | `Blur_3_main` | `Blur_3Params` |
| `Blur<7>()` | `Blur_7_main` | `Blur_7Params` |
| `GameOfLife<64, 64>()` | `GameOfLife_64_64_main` | `GameOfLife_64_64Params` |

Non-generic kernels keep their original name (`Scale_main`).

### Expression evaluation

Array sizes that are const-evaluable expressions are reduced at monomorphisation time:

```boring
kernel Tile<int W, int H>:
    mut [float, W * H]'unified weights   # → array<f32, 64> when W=8, H=8
```

Supported operators: `+`, `-`, `*`, `/`, `%`, unary `-`.

### Restrictions

- `[T]'actor` (dynamic workgroup memory) is not supported in WGSL — use `[T, N]'actor` with a const generic param instead.
- A generic kernel with no instantiations in the program emits no WGSL (no code is generated for unused generics).

---

## Known limitations vs CUDA

| Feature | CUDA | wgpu |
|---|---|---|
| `print` in kernel | `printf` | silent no-op — WGSL has no device-side print |
| Double precision | full support | optional feature (`SHADER_F64`); not enabled by default |
| `priority =` | `cuStreamCreateWithPriority` | no-op — wgpu has no queue priority API |
| `freeMem()` | `cuMemGetInfo` | always 0 — wgpu does not expose free VRAM |
| `computeCapability()` | CUDA SM version | always `[0, 0]` — not applicable |
| Multi-device (`GPU(1)`, `new(g1) K`) | full support — distinct `CudaContext` per index | `GPU(n)` compiles and runs (see "`GPU` type on wgpu" above), but every index resolves to the same single real adapter — not real multi-device selection. `new(g1) K` (placing a kernel on a specific device) is not implemented |
| Windows / Linux / macOS | Windows + Linux | yes — Windows (DX12), Linux (Vulkan), macOS (Metal via wgpu) |

---

## GPU display

`boring build --target wgpu` supports live GPU rendering via `Screen`, `'surface`, and `kernel: loop:`. The surface buffer (a `'surface` field of type `[uint]`) is treated as a `STORAGE | MAP_READ | COPY_SRC` buffer. Each frame, `screen.present()` blits it to a `wgpu::Surface` texture via a copy command.

See [`gpu-display.md`](gpu-display.html) for the full display reference. wgpu-specific notes:

- **Window**: winit 0.30 + `wgpu::Surface` tied to the window handle.
- **Pixel format**: `Bgra8Unorm` — same packing as the Metal backend (`0xAARRGGBB`). The backend requests `Bgra8Unorm` explicitly (supported on DX12, Vulkan, and Metal) so pixel code is portable across `--target metal` and `--target wgpu` with no source change.
- **Blit**: `encoder.copy_buffer_to_texture(surface_buf → swapchain_texture)` each frame, followed by `queue.present()`.
- **Drawable size**: fixed at the kernel's `Dimension` — not updated on window resize.
- **Extra dependencies** added to `Cargo.toml` when `Screen` is present: `winit = "0.30"`.

---

## Generated project layout

```
<stem>_wgpu/
  src/main.rs        # Rust host code (wgpu crate)
  shaders/main.wgsl  # WGSL compute shader
  Cargo.toml         # wgpu = "22", no build.rs
```

Requires wgpu 22+ and a DirectX 12, Vulkan, or Metal-capable GPU. No external GPU toolkit needed.

```sh
boring build --target wgpu main.br
cd main_wgpu && cargo build
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

- **Host crate**: `wgpu` v22.
- **Binding index ordering**: `'unified`, `'global`, `'actor'global`, `'actor'unified`, and `'surface` arrays first (in declaration order), then the `'const`/scalar-`'local` params struct as a single uniform buffer with the next binding index. The same order is used in the WGSL `@group(0) @binding(N)` annotations and the Rust `BindGroupLayoutEntry` list.
- **Workgroup size**: encoded via WGSL pipeline overrides (`override block_x: u32`); set at pipeline-creation time from the `block =` dispatch argument.
- **Dispatch**: the Rust host calls `encoder.dispatch_workgroups(gx, gy, gz)` and submits immediately. The `kernel:` block calls `queue.submit(...)` followed by `device.poll(wgpu::MaintainBase::Wait)` to ensure completion before the next host line.
- **Bare `'actor` (workgroup memory)**: fixed-size `[T, N]'actor` fields are emitted as `var<workgroup> tile: array<T, N>`. Dynamic `[T]'actor` fields are not supported in WGSL — the transpiler rejects them with a compile-time error and suggests using `[T, N]'actor` instead.
- **Atomics**: `'actor'global`/`'actor'unified` fields are emitted as `array<atomic<i32>>` (or `atomic<u32>` for `uint`). All five compound-assignment operations (`+= -= &= |= ^=`) map to the corresponding WGSL `atomicAdd` / `atomicSub` / `atomicAnd` / `atomicOr` / `atomicXor`.
- **`atomicSub`**: WGSL has a native `atomicSub` — no negation workaround needed (unlike the Metal backend).
