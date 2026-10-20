# GPU display

> **Status: Metal, wgpu, CUDA, ROCm, and CPU SIMD — implemented.** Simulation
> (`boring run`) is pending. CUDA, ROCm and SIMD take a software-blit path
> (`softbuffer`, CPU-side presentation) since none has a native
> presentation API — see their own backend docs
> ([`cuda-module.md`](cuda-module.html#gpu-display-screen),
> [`rocm-backend.md`](rocm-backend.html#gpu-display-screen),
> [`simd-backend.md`](simd-backend.html#gpu-display)) for that
> approach's specifics. CUDA is unverified on real hardware (none available
> in this project's dev environment); ROCm was verified on real AMD
> hardware; SIMD is covered by generated-code tests (no window in CI) plus a
> manual launch of both examples on macOS/Apple Silicon.

Live GPU rendering to a native OS window: a `'surface` pixel buffer, a `Screen`
object, and a `kernel: loop:` render loop. A single Boring source file is
intended to compile unchanged across `--target metal`/`wgpu`/`cuda`/`rocm`/`simd` —
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
        pixels = [0 for ..<d.width * d.height]
        dim    = d
        t      = 0.0

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
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
| SIMD | `Vec<u32>` (plain CPU vector) | `u32` | frame handed to the main-thread event loop, blitted via `softbuffer` |
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

### Initial contents

The contents of a `'surface` buffer before the kernel has written it are **unspecified**: on most hosts a
`Dimension`-only kernel (`K(Dimension(w, h))`) starts from a sparse random pattern, never from zeros, and nothing
guarantees either. Zero-filling costs a pass over the whole buffer, so no host does it for you: a kernel that
reads `'surface` pixels it has not written itself (e.g. a Game-of-Life seed) must not rely on their value, and
one that needs a defined starting picture writes it (a first dispatch, or an `init` that assigns the field).

### Setup code before the window opens

A `Screen` program's top-level `let`s and statements (other than the `Screen(...)` itself and the render loop)
are ordinary code that runs once before the window opens. A kernel built there with constructor arguments
(`var mut k = TileKernel(frame.f, ...)`) runs its `init(...)` on every host (array uploads, scalar fields,
`Dimension` sizing); the render loop then reads the kernel and any host values it uses. On wgpu this code is
the synthesized `fn __boring_screen_setup()`. Module-level constants of imported modules are emitted as
module-level `const`s, visible to the functions that use them. Covered by `tests/screen_setup.rs`.

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

### Updating arrays between frames

Inside the render loop a kernel's array field (`'unified`/`'global`) can be assigned a host array, and the
length may change from frame to frame:

```boring
kernel:
    loop:
        let enc = encodeFrame(screen.frame)   # per-frame host local (a user function)
        k.f = enc.f                           # whole-array upload
        k.count = enc.count                   # int scalar, narrowed to the field's 32-bit mirror
        k(block = (16, 16))
        screen.present(k.out)
```

The assignment lowers to the kernel's own upload: wgpu `copy_<f>_to_device` (reallocates the buffer and rebuilds
the bind group when the length changed), SIMD `copy_<f>_to_device`, Metal a fresh `MTLBuffer` of exactly the
array's length, CUDA/ROCm a fresh device buffer (`Arc`-wrapped for a `let` field). Elements are converted to the
field's width. `a.x = b.y` with a *kernel* on the right stays the zero-copy buffer handoff. `screen.frame` is
readable in loop expressions on every target. Re-uploading a `'surface` field this way is not supported (the
presenting bind group keeps the old buffer). `.count`/`.length` on a value whose struct type declares a real field or method of that name (typed
locals/params, `fn`/method return types, struct field chains, and `self`) reads that member on every backend
(Metal/CUDA/ROCm/wgpu share a receiver-type resolver, `src/transpiler/host_types.rs`); only a receiver whose type is not
resolvable as a user struct (or a real array) still gets the array-length builtin. Pinned by
`tests/struct_count_field_receiver.rs`.

### On-demand redraw

```boring
let screen = Screen(Dimension(640, 360), title = "UI", redraw = "demand")
```

With `redraw = "demand"` the loop body runs once at start and afterwards only when the window is invalidated: a
keyboard, mouse or resize event, or `screen.requestRedraw()` called in the body (to schedule one more frame, e.g.
for an animation). The loop otherwise sleeps (wgpu `ControlFlow::Wait`; Metal/CUDA/ROCm `Wait` with `Poll` only
while a redraw is pending; SIMD blocks inside `screen.present` on a condition variable). `screen.requestRedraw()`
is a no-op in the default continuous mode. Regression tests: `tests/screen_render_loop_arrays.rs`.


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

## Input (simd, Metal, CUDA, ROCm, wgpu)

Example: `examples/screen_input.br`.

- `screen.key("k")` — level-triggered: true while held.
- `screen.key_pressed("k")` — edge-triggered: true only for the frame in which the key went from released to pressed (OS auto-repeat does not retrigger it). Previously an alias of `key()` on Metal/CUDA/ROCm.
- `screen.mouse.x` / `.y` (`int`) — cursor position in **pixel-buffer** coordinates (window pixels are rescaled to the buffer, so HiDPI/resized windows are handled; the last position is kept when the cursor leaves the window) — and `screen.mouse.left` / `.right` / `.middle` (`bool`). Only the field forms are supported (not a bare `screen.mouse` value on the GPU hosts).
- State is consistent within one loop iteration: simd snapshots it at the end of `screen.present(...)`; the winit-driven hosts update it from events delivered before the frame body, and clear the pressed set after it.
- Caveat: kernel `int` fields are `i32` on simd but `isize` on Metal — pass a `float32(...)` (as the example does) to stay portable.
- Only simd/Metal/wgpu codegen is compile-checked here (`cargo check`); CUDA/ROCm have codegen snapshot tests only, no hardware verification of the new input path.
- wgpu: `screen.key(...)` / `key_pressed(...)` / `screen.mouse.*` are general host expressions (usable in any `if`/`elif`/`else` condition or scalar field assignment inside the render loop, via `self.__keys` / `self.__keys_pressed` / `self.__mouse_*` on the `ApplicationHandler`); the pressed set is cleared at the end of each `RedrawRequested` body.

---

## Backend notes (CPU SIMD)

See [`simd-backend.md`](simd-backend.html#gpu-display) for the full writeup.
Summary: software blit through winit **0.28** + `softbuffer` **0.3** like
CUDA/ROCm, but because the CPU backend's `kernel: loop:` is an ordinary Rust
`loop` (not a `run_return` closure), the program runs on a worker thread and
`main` keeps the window's event loop; `screen.present()` passes the frame across
and waits for the blit. A `let Dimension` kernel gets flat-extent construction,
`w * h` array allocation, randomly seeded `'surface` buffers, and the inferred
2D dispatch grid, exactly as on the other backends. Dependencies are added to the
generated `Cargo.toml` only when the program uses `Screen`.



---

## Pending

- **`boring run`**: `screen.present()` writes a PPM on loop exit; `--preview` opens a `minifb` window.
- **Resize handling**: `screen.resized` detected but the kernel surface is not reallocated automatically — user code must reinitialise `k`. Same limitation on every implemented backend.
- **`screen.key_pressed()`**, **`screen.mouse`**: implemented on every target (simd, Metal, CUDA, ROCm, wgpu) with the same semantics — see "Input" above.
- **`screen.pixels_rgba()`**: portable RGBA read from `'surface` buffer.
- **CUDA real-hardware verification**: no NVIDIA GPU/CUDA toolkit available in this project's dev environment — see `cuda-module.md`'s Screen section.
