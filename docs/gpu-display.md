# GPU display

> **Status: Metal, wgpu, CUDA, and ROCm — implemented.** Simulation
> (`boring run`) is pending. CUDA and ROCm take a software-blit path
> (`softbuffer`, CPU-side presentation) since neither has a native
> presentation API — see their own backend docs
> ([`cuda-module.md`](cuda-module.html#gpu-display-screen),
> [`rocm-backend.md`](rocm-backend.html#gpu-display-screen)) for that
> approach's specifics. CUDA is unverified on real hardware (none available
> in this project's dev environment); ROCm was verified on real AMD
> hardware.

Live GPU rendering to a native OS window: a `'surface` pixel buffer, a `Screen`
object, and a `kernel: loop:` render loop. A single Boring source file is
intended to compile unchanged across `--target metal`/`wgpu`/`cuda`/`rocm` —
the backend differences (GPU-native blit vs. software blit via `softbuffer`)
are transparent.

---

## Quick start

```boring
let screen = Screen(Dimension(800, 600), title = "Plasma")

kernel Plasma:
    mut [uint]'surface pixels
    var Dimension dim
    var float t

    init(Dimension d):
        pixels = [0 for ..d.width * d.height]
        dim    = d
        t      = 0.0

    def ():
        let col = gpu.block.x * gpu.block_dim.x + gpu.thread.x
        let row = gpu.block.y * gpu.block_dim.y + gpu.thread.y
        if col < dim.width and row < dim.height:
            let x = float(col) / float(dim.width)
            let y = float(row) / float(dim.height)
            let v = sin(x * 10.0 + t) + sin(y * 10.0 + t)
            let r = uint((sin(v + t)         * 0.5 + 0.5) * 255.0)
            let g = uint((sin(v + t + 2.094) * 0.5 + 0.5) * 255.0)
            let b = uint((sin(v + t + 4.189) * 0.5 + 0.5) * 255.0)
            pixels[row * dim.width + col] = 0xFF000000 | (r << 16) | (g << 8) | b

var k = Plasma(Dimension(800, 600))

kernel:
    loop:
        k.t = float(screen.time)
        k(block = (16, 16))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
```

---

## `'surface` qualifier

`'surface` is a GPU memory qualifier for `[uint]` pixel buffers. It describes a
buffer whose content will be presented to a window via `screen.present()`.

| Backend | Maps to | Element size | Presentation path |
|---|---|---|---|
| Metal | `MTLStorageModeShared` buffer | `uint` (32-bit) | GPU blit to `CAMetalDrawable` (`BGRA8Unorm`) |
| wgpu | `storage` buffer, `MAP_READ \| COPY_SRC` | `uint` (32-bit) | Fragment shader reads the storage buffer directly, renders to the swapchain (`BGRA8Unorm`) — no CPU readback |
| CUDA | device buffer (same as `'unified`/`'global`) | host side: same as the field's declared element type (e.g. `usize` for `uint`) | D2H readback (`clone_dtoh`) each frame, narrowed to `u32` and blitted via [`softbuffer`](https://docs.rs/softbuffer) — pure-CPU presentation, no GPU-graphics interop |
| ROCm | device buffer (same as `'unified`/`'global`) | same as CUDA | same software-blit path as CUDA (`clone_dtoh` equivalent + `softbuffer`) |
| Simulation | `Vec<u32>` | `u32` | PPM write or `minifb` *(pending)* |

### Why not `'unified` directly

On Metal, `screen.present()` performs a GPU-side blit — no CPU access to the
pixel buffer is needed. `'surface` signals this intent and ensures the correct
32-bit element type for BGRA8Unorm. Using `'unified` with `screen.present()`
is also valid but carries minor coherency overhead and is less explicit.

### Constraints

- Only `[uint]` is valid for `'surface` — other element types are a compile error.
- A kernel may have at most one `'surface` field.
- `'surface` cannot be combined with other GPU qualifiers.

### Pixel format

All backends use `BGRA8Unorm`. Pixels are packed as 32-bit `uint` values:
`0xAARRGGBB` in source — `(alpha << 24) | (r << 16) | (g << 8) | b`.
The alpha channel should be `0xFF` for opaque pixels.

The wgpu backend requests `BGRA8Unorm` explicitly from the surface (supported on
DX12, Vulkan, and Metal) so that pixel packing is identical to the Metal backend.
No source change is needed when switching between `--target metal` and `--target wgpu`.

---

## `Dimension` type

Built-in struct for named width/height pairs.

```boring
let dim = Dimension(1024, 768)
dim.width   # 1024
dim.height  # 768
```

On the GPU side, `Dimension` fields are passed as uniform constants and have
`uint` (32-bit) width and height.

---

## `Screen` type

`Screen` creates a native OS window — backed by a Metal layer on `--target
metal`, a `wgpu::Surface` on `--target wgpu`, and a `softbuffer` surface on
`--target cuda`/`--target rocm` (see each backend's own doc for specifics).

```boring
let screen = Screen(Dimension(800, 600))
let screen = Screen(Dimension(800, 600), title = "My app")
```

### Properties

| Property | Type | Description |
|---|---|---|
| `screen.dimension` | `Dimension` | window size at creation |
| `screen.width` | `int` | shorthand for `screen.dimension.width` |
| `screen.height` | `int` | shorthand for `screen.dimension.height` |
| `screen.resized` | `bool` | true on the frame the window was resized |
| `screen.closed` | `bool` | true if the user closed the window |
| `screen.frame` | `int` | frame counter, starts at 0 |
| `screen.time` | `float` | seconds elapsed since loop start |

### Methods

| Method | Description |
|---|---|
| `screen.present(pixels)` | blit the pixel buffer to the window; called once per frame |
| `screen.key(k)` | `bool` — true while key `k` is held (string literal) |

### Key strings

`screen.key()` accepts a string literal for the key name:

| Key | String |
|---|---|
| Escape | `"\x1B"` |
| Space | `" "` |
| Enter | `"\r"` |
| Letters / digits | `"a"`, `"1"`, … |

---

## `kernel: loop:` — render loop

`loop:` inside a `kernel:` block drives the render loop. On Metal/CUDA/ROCm it
maps to a winit `run_return` event loop; on wgpu it maps to a winit
`ApplicationHandler` (a `struct __App` + `resumed`/`window_event` methods).
`screen.present()` issues a GPU blit (Metal/wgpu) or a D2H readback + software
blit (CUDA/ROCm) and presents the frame. `break` exits the loop cleanly.

```boring
kernel:
    loop:
        k.t = float(screen.time)
        k(block = (16, 16))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
```

### Dispatch grid

When a kernel has a `Dimension` field alongside a `'surface` field, the
transpiler infers a 2D dispatch grid automatically from that field:

```
grid = (ceil(dim.width / block_x), ceil(dim.height / block_y), 1)
```

An explicit `grid =` parameter overrides the inferred value.

### Kernel scalar fields and `screen.time`

`var float t` in a kernel is a `f32` scalar passed as a uniform constant on
each dispatch. Assign it before the kernel call:

```boring
k.t = float(screen.time)   # screen.time is float (seconds), assigned to k.t (f32)
k(block = (16, 16))
```

---

## Backend notes (Metal)

- Window: winit 0.28 + `CAMetalLayer` attached to the `NSView`.
- Pixel format: `BGRA8Unorm` — pixels packed as `0xAARRGGBB`.
- Blit: `MTLBlitCommandEncoder.copyFromBuffer(toTexture:)` each frame.
- Drawable size: fixed at the kernel's surface dimensions — not updated on window resize.
- Dependencies: `winit = "0.28"`, `objc = "0.2"`, `core-graphics = "0.23"`.
- Requires macOS 11+ with a Metal-capable GPU.

---

## Backend notes (wgpu)

- Window: winit 0.30 + `wgpu::Surface` tied to the window handle. No platform-specific bindings.
- Pixel format: `BGRA8Unorm` — same packing as Metal (`0xAARRGGBB`). Requested explicitly so pixel code is portable across `--target metal` and `--target wgpu`.
- Present: a full-screen-quad render pipeline whose fragment shader reads the `'surface` storage buffer directly and writes to the swapchain texture — no CPU readback, no `copy_buffer_to_texture`.
- Drawable size: fixed at the kernel's surface dimensions — not updated on window resize.
- Dependencies: `wgpu = "22"`, `winit = "0.30"` (added to the generated `Cargo.toml` only when the program uses `Screen`).
- Requires a DX12 (Windows), Vulkan (Windows / Linux), or Metal (macOS) capable GPU.

---

## Backend notes (CUDA / ROCm)

Neither CUDA nor HIP has a native presentation API, so both take a
**software-blit** path instead of a GPU-native one — see
[`cuda-module.md`](cuda-module.html#gpu-display-screen) /
[`rocm-backend.md`](rocm-backend.html#gpu-display-screen) for the full
writeup. Summary:

- Window: winit **0.28** (not 0.30, unlike wgpu/Metal-adjacent code) — the
  last release with the `EventLoopExtRunReturn::run_return` API, matching
  these backends' hand-rolled statement/expression emitter (see either
  module's own top-of-file doc comment for why the general pipeline can't be
  reused here the way wgpu's is).
- Present: the `'surface` buffer is read back to the host every frame
  (`clone_dtoh`) and blitted into the window via
  [`softbuffer`](https://docs.rs/softbuffer) `0.3` (the release built against
  `raw-window-handle 0.5`, matching winit 0.28).
- Pixel format: same `0xAARRGGBB` packing; softbuffer's own `0RGB` format
  ignores the top byte, so no conversion is needed beyond narrowing the D2H
  read to `u32`.
- Drawable size: fixed at the kernel's surface dimensions, requested via
  `winit::dpi::PhysicalSize` (not `LogicalSize`) so the window's physical
  pixel count matches the pixel buffer regardless of the display's DPI scale
  factor — a real bug on a >100%-scaled display otherwise (see
  `rocm-backend.md`'s bug #8).
- A kernel's `'surface` field paired with a sibling `Dimension` field drives
  a genuinely 2D dispatch grid (`ceil(width/block_x), ceil(height/block_y)`)
  — the same inference rule the "Dispatch grid" section above documents for
  Metal/wgpu.
- Dependencies: `winit = "0.28"`, `softbuffer = "0.3"` (added to the
  generated `Cargo.toml` only when the program uses `Screen`).
- Requires the CUDA toolkit + NVIDIA GPU, or the ROCm toolkit + AMD GPU,
  respectively — same as either backend's compute-only requirements.
- **Verified on real hardware**: ROCm only (AMD Radeon RX 6600). CUDA has no
  NVIDIA hardware available in this project's dev environment — ships as a
  by-symmetry port plus codegen snapshot tests, unverified on real hardware.

---

## Pending

- **`boring run`**: `screen.present()` writes a PPM on loop exit; `--preview` opens a `minifb` window.
- **Resize handling**: `screen.resized` detected but the kernel surface is not reallocated automatically — user code must reinitialise `k`. Same limitation on every implemented backend.
- **`screen.key_pressed()`**, **`screen.mouse`**: not yet implemented on Metal/wgpu. Implemented on CUDA/ROCm (mirrors `screen.key()`).
- **`screen.pixels_rgba()`**: portable RGBA read from `'surface` buffer.
- **CUDA real-hardware verification**: no NVIDIA GPU/CUDA toolkit available in this project's dev environment — see `cuda-module.md`'s Screen section.
