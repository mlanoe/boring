# GPU display

> **Status: Metal — implemented.** CUDA and simulation (`boring run`) are pending.

Live GPU rendering to a native OS window: a `'surface` pixel buffer, a `Screen`
object, and a `kernel: loop:` render loop. A single Boring source file is
intended to compile unchanged on both Metal and CUDA — the backend differences
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
| Metal | `MTLStorageModeShared` buffer | `uint` (32-bit) | GPU blit to `CAMetalDrawable` (BGRA8Unorm) |
| CUDA | `cudaMallocManaged` | `uint32_t` | host upload via SDL2/OpenGL *(pending)* |
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

`screen.present()` on Metal uses `BGRA8Unorm`. Pixels are packed as 32-bit
`uint` values: `0xAARRGGBB` in the source — `(alpha << 24) | (r << 16) | (g << 8) | b`.
The alpha channel should be `0xFF` for opaque pixels.

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

`Screen` creates a native OS window backed by a Metal layer (macOS).

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

`loop:` inside a `kernel:` block drives the render loop. On Metal it maps to a
winit `run_return` event loop; `screen.present()` issues a GPU blit and presents
the drawable. `break` exits the loop cleanly.

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

## Pending

- **CUDA**: SDL2 + OpenGL PBO upload path, `screen.present()` via `glTexSubImage2D`.
- **`boring run`**: `screen.present()` writes a PPM on loop exit; `--preview` opens a `minifb` window.
- **Resize handling**: `screen.resized` detected but the kernel surface is not reallocated automatically — user code must reinitialise `k`.
- **`screen.key_pressed()`**, **`screen.mouse`**: not yet implemented.
- **`screen.pixels_rgba()`**: portable RGBA read from `'surface` buffer.
