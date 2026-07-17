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
| `'unified` | `cudaMallocManaged` | `device` + `MTLStorageMode.shared` | `storage` buffer, `MAP_READ \| MAP_WRITE` (host-visible) |
| `'global` | `cudaMalloc` | `device` | `storage` buffer, GPU-only |
| `'surface` | `cudaMallocManaged` (u32) | `device uint*` | `storage` buffer of `u32`, host-visible — same layout as `'unified` |
| `'sync` | `__shared__` | `threadgroup` | `var<workgroup>` |
| `'local` | registers | thread-private | `var<function>` |
| `'const` scalar | `__constant__ T name;` | `constant T* [[buffer(N)]]` | `var<uniform>` in a dedicated uniform buffer |
| `'const` fixed array (`[T, N]`) | `__constant__ T name[N];` | `constant T* [[buffer(N)]]` | `var<uniform>` array in a dedicated uniform buffer |
| `'actor'global` | `atomicAdd` etc. | `atomic_fetch_add_explicit` | `atomic<i32>` / `atomic<u32>` fields in `storage` buffer |

---

## Built-in mapping

| Boring | CUDA C | MSL | **WGSL** |
|---|---|---|---|
| `gpu.thread.x/y/z` | `threadIdx.x/y/z` | `thread_position_in_threadgroup.x/y/z` | `@builtin(local_invocation_id).x/y/z` |
| `gpu.block.x/y/z` | `blockIdx.x/y/z` | `threadgroup_position_in_grid.x/y/z` | `@builtin(workgroup_id).x/y/z` |
| `gpu.block_dim.x/y/z` | `blockDim.x/y/z` | `threads_per_threadgroup.x/y/z` | `@builtin(local_invocation_size).x/y/z` |
| `gpu.grid_dim.x/y/z` | `gridDim.x/y/z` | `threadgroups_per_grid.x/y/z` | `@builtin(num_workgroups).x/y/z` |
| `sync` (manual) | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` | `workgroupBarrier()` |
| `'sync` auto-barrier | inserted before first loop + at top of each loop iteration | idem | idem |
| atomics (`'actor'global`) | `atomicAdd` etc. | `atomic_fetch_add_explicit` | `atomicAdd` / `atomicSub` / `atomicOr` / `atomicAnd` / `atomicXor` |

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

The binding index `N` follows all `'unified`, `'global`, and `'actor'global` array bindings. The Rust host writes the struct before every dispatch via `queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params))`.

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
| `'unified` | `STORAGE \| MAP_READ \| MAP_WRITE \| COPY_SRC \| COPY_DST` | direct via `map_async` |
| `'global` | `STORAGE \| COPY_DST` | via `gpu.copy()` (staging buffer) |

`MAP_READ` and `MAP_WRITE` on the same buffer are valid in wgpu but require explicit `map_async` / `unmap` calls around host access. The Boring runtime wraps these transparently — host reads after a `kernel:` block are safe without explicit copy.

---

## `GPU` type on wgpu

| Boring method | wgpu API |
|---|---|
| `name()` | `adapter.get_info().name` |
| `totalMem()` | `adapter.get_info().dedicated_video_memory` (may be 0 on unified-memory GPUs — returns `shared_system_memory` as fallback) |
| `freeMem()` | not available via wgpu — returns 0 |
| `computeCapability()` | `[0, 0]` — not meaningful outside CUDA |
| `warpSize()` | 32 (conservative default) |
| `maxThreads()` | `limits.max_compute_invocations_per_workgroup` |
| `maxSharedMem()` | `limits.max_compute_workgroup_storage_size` (bytes per workgroup) |
| `index()` | index passed to `GPU(i)` |

`GPU(0)` → `wgpu::Instance::enumerate_adapters`, first adapter.
`GPU.all()` → all adapters, sorted by index.

---

## `after =` ordering

wgpu uses a single command queue per device. Command buffers submitted to the same queue execute in order. The transpiler maps `after =` to submission ordering — kernels with `after =` dependencies are submitted in a separate `submit` call after the dependency's `submit` has been flushed.

This is not GPU-side pipelining (unlike CUDA streams), but it preserves the ordering semantics with no CPU round-trip within a `kernel:` block.

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

WGSL forbids `bool` inside `storage` and `uniform` buffers (`array<bool>` is invalid). The transpiler maps `bool` fields and `bool` array elements to `u32` in all GPU memory contexts (`'unified`, `'global`, `'sync`, struct fields). `true` → `1u`, `false` → `0u`. Comparisons and conditionals that receive a `u32` from a GPU field coerce back to WGSL `bool` via `!= 0u` automatically.

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

- `[T]'sync` (dynamic workgroup memory) is not supported in WGSL — use `[T, N]'sync` with a const generic param instead.
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
| Multi-device (`GPU(1)`, `new(g1) K`) | full support | supported — one wgpu adapter per `GPU(i)` |
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
- **Binding index ordering**: `'unified` and `'global` arrays first (in declaration order), then `'const` fields as uniform buffers, then `'actor'global` arrays. The same order is used in the WGSL `@group(0) @binding(N)` annotations and the Rust `BindGroupLayoutEntry` list.
- **Workgroup size**: encoded via WGSL pipeline overrides (`override block_x: u32`); set at pipeline-creation time from the `block =` dispatch argument.
- **Dispatch**: the Rust host calls `encoder.dispatch_workgroups(gx, gy, gz)` and submits immediately. The `kernel:` block calls `queue.submit(...)` followed by `device.poll(wgpu::MaintainBase::Wait)` to ensure completion before the next host line.
- **`'sync` (workgroup memory)**: fixed-size `[T, N]'sync` fields are emitted as `var<workgroup> tile: array<T, N>`. Dynamic `[T]'sync` fields are not supported in WGSL — the transpiler rejects them with a compile-time error and suggests using `[T, N]'sync` instead.
- **Atomics**: `'actor'global` fields are emitted as `array<atomic<i32>>` (or `atomic<u32>` for `uint`). All five compound-assignment operations (`+= -= &= |= ^=`) map to the corresponding WGSL `atomicAdd` / `atomicSub` / `atomicAnd` / `atomicOr` / `atomicXor`.
- **`atomicSub`**: WGSL has a native `atomicSub` — no negation workaround needed (unlike the Metal backend).
