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
| `screen.width` | `int` | **current** window width in pixels (follows resizes) |
| `screen.height` | `int` | **current** window height in pixels |
| `screen.resized` | `bool` | true for the frame that follows a resize — see [Resizing](#resizing) |
| `screen.closed` | `bool` | true if the user closed the window |
| `screen.frame` | `int` | frame counter, starts at 0 |
| `screen.time` | `float` | seconds elapsed since loop start |

The input members (`screen.key…`, `screen.mouse…`, `screen.text`, …) are listed under [Input](#input-simd-metal-cuda-rocm-wgpu).

### Methods

| Method | Description |
|---|---|
| `screen.present(pixels)` | blit the pixel buffer to the window; called once per frame |
| `screen.key(k)` | `bool` — true while key `k` is held |
| `screen.key_pressed(k)` / `key_repeat(k)` / `key_released(k)` | key edges — see [Input](#input-simd-metal-cuda-rocm-wgpu) |
| `screen.mouse_pressed(b)` / `mouse_released(b)` | mouse-button edges |

### Key strings

Every key query takes a string — a literal or any string expression — and every host maps it to the same
**canonical key name**, so a source file means the same thing on every target:

| Key | Canonical name | Also accepted |
|---|---|---|
| letters, digits, punctuation | the lowercase character (`"a"`, `"1"`, `"-"`) | any letter case (`"A"`) |
| Space | `" "` | `"Space"` |
| Escape | `"Escape"` | `"\x1B"`, `"Esc"` |
| Enter / Return | `"Enter"` | `"\r"`, `"\n"`, `"Return"` |
| Tab, Backspace, Delete, Insert | `"Tab"`, `"Backspace"`, `"Delete"`, `"Insert"` | `"\t"`, `"\x08"`, `"\x7F"` |
| Arrows | `"Left"`, `"Right"`, `"Up"`, `"Down"` | `"ArrowLeft"`…, the ANSI `"\x1B[D"`… |
| Home, End, PageUp, PageDown | same | any letter case |
| `F1`…`F12` | `"F1"`…`"F12"` | `"f5"` |
| modifiers | `"Shift"`, `"Control"`, `"Alt"`, `"Meta"` | `"Ctrl"`, `"Cmd"`, `"Super"`, `"Win"` |
| CapsLock | `"CapsLock"` | |

Left and right modifier keys share one name; named keys are matched case-insensitively.

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
readable in loop expressions on every target. Re-uploading a `'surface` field this way works on wgpu only (it reallocates the blit too, see [Resizing](#resizing)); on the other hosts reassign the kernel instead. `.count`/`.length` on a value whose struct type declares a real field or method of that name (typed
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
- Drawable size: the presented buffer's dimensions (set when the kernel is reallocated, see [Resizing](#resizing)).
- Dependencies: `winit = "0.28"`, `objc = "0.2"`, `core-graphics = "0.23"`.
- Requires macOS 11+ with a Metal-capable GPU.

---

## Backend notes (wgpu)

- Window: winit 0.30 + `wgpu::Surface` tied to the window handle. No platform-specific bindings.
- Pixel format: `BGRA8Unorm` — same packing as Metal (`0xAARRGGBB`). Requested explicitly so pixel code is portable across `--target metal` and `--target wgpu`.
- Present: a full-screen-quad render pipeline whose fragment shader reads the `'surface` storage buffer directly and writes to the swapchain texture — no CPU readback, no `copy_buffer_to_texture`.
- Swapchain size: the presented buffer's dimensions (reconfigured when the kernel is reallocated, see [Resizing](#resizing)).
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
- Drawable size: the presented buffer's dimensions (initially the kernel's surface size), requested via
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

Examples: `examples/screen_input.br` (mouse + key edges); the members below are exercised by
`tests/screen_input_events.rs` on every target.

**Model.** Events accumulate between frames and the frame body sees them as a *per-frame snapshot*: held state
(`key`, `mouse.*` buttons, modifiers, `focused`, `marked`) is the state when the frame starts, and edge/event
members (`key_pressed`, `text`, `scroll_*`, …) cover everything that happened since the previous frame and are cleared
after it (simd snapshots at the end of `screen.present(...)`, the winit-driven hosts clear after the body). There is
no unbounded queue: a frame that is slow to start sees all its events at once — use `screen.text` (the whole string
typed) rather than assuming one character per frame. All members work in any `if`/`elif`/`else` condition or scalar
field assignment in the render loop, and in `redraw = "demand"` mode every input event invalidates the window.

| Member | Type | Meaning |
|---|---|---|
| `screen.key(k)` | `bool` | held — level-triggered |
| `screen.key_pressed(k)` | `bool` | went down this frame (OS auto-repeat does **not** retrigger it) |
| `screen.key_repeat(k)` | `bool` | went down this frame **or** an OS auto-repeat of it arrived (use for held-key editing: Backspace, arrows) |
| `screen.key_released(k)` | `bool` | released this frame |
| `screen.last_key` | `string` | canonical name of the last key that went down or repeated this frame, `""` if none |
| `for k in screen.key_events:` / `screen.key_up_events` | statement | every key-down (**including auto-repeats**) / key-up event of the frame as a canonical name, **in arrival order and with duplicates** — the event queue; nothing is lost when a frame is slow. `k` is a `string`; `break` inside is not supported |
| `screen.setImeArea(x, y, w, h)` | statement | tell the input method where the text caret is (buffer coordinates, scaled like the mouse) so its candidate window opens next to it. winit 0.30 (wgpu) passes the rectangle; the winit 0.28 hosts pass the point just below it |
| `screen.mouse.x`, `.y` | `int` | cursor position in **pixel-buffer** coordinates (window pixels are rescaled to the buffer, so HiDPI and resized windows work; the last position is kept when the cursor leaves the window) |
| `screen.mouse.left`, `.right`, `.middle` | `bool` | button held |
| `screen.mouse_pressed(b)`, `screen.mouse_released(b)` | `bool` | button `"left"`/`"right"`/`"middle"` went down / up this frame (a click shorter than a frame is both) |
| `screen.mouse.scroll_x`, `.scroll_y` | `float32` | wheel / trackpad movement accumulated this frame, in **pixels** (a wheel notch is 40). `scroll_y > 0` = rolled away from the user (content moves down) |
| `screen.shift`, `.ctrl`, `.alt`, `.meta` | `bool` | modifier state (`meta` = Command on macOS, the Windows key elsewhere) |
| `screen.text` | `string` | text typed this frame: printable characters and IME-committed text, in order. No control characters (Enter/Backspace/Tab are keys, not text) and none while Ctrl (without Alt) or Meta is held, so shortcuts do not type |
| `screen.marked` | `string` | the IME **composition** (marked / pre-edit) text being edited, `""` when none |
| `screen.marked_start`, `.marked_end` | `int` | the caret/selection inside `marked` as byte offsets, `-1` when there is no composition or the IME gave none |
| `screen.focused` | `bool` | the window has keyboard focus (true until the first focus change) |
| `screen.focus_gained`, `.focus_lost` | `bool` | focus changed this frame |

- **Key names** are the canonical names under [Key strings](#key-strings): `screen.key(name)` works with any string
  expression, e.g. a key table read from data.
- **IME.** The window opts in to input-method events at creation (`set_ime_allowed(true)`). While a composition is
  active, what the input method consumes arrives as `marked`/`text` (winit does not report those key presses to the app). An
  input method reports the same characters through two channels; the runtime delivers them once. Candidate-window
  placement (a cursor-area hint) is not exposed yet.
- **Strings.** `screen.text`, `.marked`, `.last_key` and the loop variable of `for k in screen.key_events:` can be
  passed to an ordinary Boring function taking a `string` on every target (on Metal/CUDA/ROCm, whose own string
  literals are `&str`, they and string literals are converted to the function's `string` type at the call; in a
  comparison they are plain `String`s: `screen.text == "x"`).
- Only the field forms of `screen.mouse` are supported (not a bare `screen.mouse` value on the GPU hosts).
- State is consistent within one loop iteration on every host.
- Caveat: kernel `int` fields are `i32` on simd/wgpu (the host narrows what you assign) but `isize` on Metal.
- Verification: the simd, wgpu, Metal (on macOS), CUDA (against `tools/fake-nvcc`) and ROCm (build script stubbed)
  generated crates are `cargo check`ed. No test opens a window; real-IME behaviour (macOS Pinyin/Kotoeri, Windows IME,
  ibus) was not exercised here.
- wgpu: `self.__keys`/`self.__keys_pressed`/`self.__input` on the `ApplicationHandler`; Metal/CUDA/ROCm/simd keep the
  same state in `boring_input`/`display.input`. A text-input event is `KeyEvent.text` on winit 0.30 (wgpu) and
  `ReceivedCharacter` on winit 0.28 (the others); both are filtered through the same rules.

### Resizing

`screen.width` / `screen.height` are the window's **current** size in pixels and `screen.resized` is true for the
frame that follows a resize. The pixel buffer is allocated once, by the kernel's constructor, and is not resized behind
your back: to follow the window, **reallocate the kernel for the new size** in the loop:

```boring
var mut k = Plasma(Dimension(800, 600))

kernel:
    loop:
        if screen.resized:
            k = Plasma(Dimension(screen.width, screen.height))   # new surface buffer, window-sized
        k(block = (16, 16))
        screen.present(k.pixels)
```

After `k = K(Dimension(screen.width, screen.height))` the presentation follows the new buffer: the wgpu swapchain is
reconfigured to the window's size and the blit is re-pointed at the new buffer (`__App::__reblit`), the Metal drawable and
the CUDA/ROCm softbuffer surface are resized to the presented buffer's `Dimension`, and the simd blit uses the frame's
size. `screen.mouse.x/y` then map window pixels to the **new** buffer. The inferred dispatch grid also follows the buffer
(wgpu). The new kernel starts from its constructor's contents (the old picture is dropped), so a program that keeps state
in the surface must redraw it. Sizes in `Dimension(...)` are `int`s: the hosts narrow them to the kernel's 32-bit
fields. Reassignment covers a kernel the program constructs as `K(Dimension(w, h))` on every target. On **wgpu** it also
covers a kernel built in the program's top-level setup code from real constructor arguments (uploaded data, sizes):
`k = Tiles(pal, screen.width, screen.height)` runs a generated function (`__boring_screen_rebuild_k`) that repeats the
construction for the current window size, reading the setup values the arguments mention (`pal`) as they were left by
the setup code.

**wgpu, alternatively.** On wgpu you can instead re-upload the `'surface` field (and update the size fields the kernel
reads) from a host array of the new length — the swapchain/blit follow as above:

```boring
        if screen.resized:
            k.width  = screen.width
            k.height = screen.height
            k.out    = blank(screen.width * screen.height)     # `def [uint] blank(int n)`: a host array
```

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



### State between frames (wgpu)

On wgpu the loop body can use the program's top-level setup values directly (tested by
`tests/screen_render_loop_state.rs`):

- **Top-level `var`s are assignable.** `var dirty = false` at the top level, `dirty = true` (or `n += 1`) in
  `loop:`: the variable becomes a field of the generated `__App`. An unannotated `var` scalar literal gets its
  default type (`bool`, `int`, `float`); a `var` declared inside the loop body is a per-frame local. Assigning a
  name that is neither is rejected with an error that says so.
- **`screen.key_events` / `screen.key_up_events` are values**, not only `for` sources: `screen.key_events.length > 0`,
  `anyKeys(screen.key_events)` (a `[string]` parameter). Metal, CUDA and ROCm accept the `.length` form too (a
  `Vec<String>` there, not usable as a function argument); simd still only supports the `for` form.
- **Floats follow their context.** Float literals in loop expressions are untyped, so `(screen.time as float) * 1000.0`
  is a plain `f64`. `screen.time` itself is an `f32`; passed to a function's `float` parameter it is converted, so
  a `float` (or `float32`) parameter works without a `float32` workaround.
- **Methods of top-level struct values can be called**: `pace.frame(rt, host, ...)` passes each argument the way the
  method's signature takes it (`&mut rt`, `&host` for a trait object), exactly as for a free function.

Top-level `var`s are also assignable on simd (a local of the generated `boring_main`, not a `const`; covered by
`simd_render_loop_assigns_top_level_vars`).

Methods of top-level struct values also work on simd (`pace.frame(rt, host, 2)` is emitted as
`pace.frame(&mut rt, &host, 2)` on the locals of `boring_main`; covered by `simd_render_loop_state_method_call`).

Metal builds the top-level struct values and `var` scalars the loop reads (`let Root host = Root(5)`,
`mut Pace pace = Pace()`, `var dirty = false`) in the general pass's `__boring_screen_setup`, bound as locals of
`main` before the window opens, like wgpu; `dirty = true` / `n += 1` assign them and `pace.frame(rt, host, 2)` is
emitted as `pace.frame(&mut rt, &host, 2)` (covered by `metal_render_loop_state_values_and_methods`). On Metal
`screen.key_up_events` still cannot be passed to a function and `screen.time` is not converted to a `float`
parameter.

Not yet on ROCm: assigning a top-level `var` in the loop and method calls on a top-level struct value.
Use free functions and per-frame locals there.

---

## Pending

- **`boring run`**: `screen.present()` writes a PPM on loop exit; `--preview` opens a `minifb` window.
- **Resize handling**: resizing is explicit — read `screen.resized`/`screen.width`/`screen.height` and reallocate the
  kernel (see [Resizing](#resizing)); nothing reallocates automatically.
- **`screen.key_pressed()`**, **`screen.mouse`** and the rest of the input members: implemented on every target
  (simd, Metal, CUDA, ROCm, wgpu) with the same semantics — see [Input](#input-simd-metal-cuda-rocm-wgpu).
- **`screen.pixels_rgba()`**: portable RGBA read from `'surface` buffer.
- **CUDA real-hardware verification**: no NVIDIA GPU/CUDA toolkit available in this project's dev environment — see `cuda-module.md`'s Screen section.
