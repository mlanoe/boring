# Draft — boring-ui renderer: native Boring kernels, built incrementally

Status: **working draft**, not a spec. Nothing here is implemented. It refines layer 4 ("Renderer") of
[`boring-ui-draft.md`](boring-ui-draft.md#runtime-architecture) and its "Painting, compositing, and render
invalidation" section; progress is tracked in [`boring-ui-progress.md`](boring-ui-progress.md).

## Question

How should boring-ui paint its display list? Candidates considered:

| Option | Verdict |
|---|---|
| Hand-written wgpu pipeline (instanced SDF quads, WGSL by hand) | Works for the small vocabulary, but all the hard parts (AA clips, opacity groups, text) are ours, in non-Boring code |
| **Skia** (`skia-safe`, what Flutter used) | Shortest path to a correct renderer; `Canvas` maps 1:1 onto the display list. Heavy C++ dependency, runtime shader generation (the jank Flutter left Skia for) |
| **Vello** (Rust, wgpu compute) | Closest in spirit; young text/layer story; large to port in full |
| `tiny-skia` (Rust, CPU) | Fine as a reference renderer, no GPU |
| **Native Boring kernels** (this document) | Chosen direction: a fixed, small shader set written in Boring, validating the kernel backends on a real graphics workload |

**Decision (2026-10-19, direction only):** try the native route, incrementally, and treat every gap it
exposes as a language/backend task (see the repo's bug-task workflow) rather than working around it.
Skia/Vello stay available later as alternative consumers of the same display list; that is the point of the
backend-neutral boundary.

Rationale, in short:

- The drawing vocabulary is deliberately small (rects, rounded rects, inside borders, shadows, glyph runs,
  clips, opacity groups). A fixed set of precompiled kernels is exactly the Impeller lesson: a *bounded*
  shader set removes runtime shader-compilation jank, which is why Flutter moved off Skia.
- Boring **already ships the display plumbing**: `Screen`, the `'surface` `[uint]` pixel buffer and
  `kernel: loop:` present on Metal, wgpu, CUDA, ROCm and SIMD ([`gpu-display.md`](../gpu-display.md)), with
  `key`/`key_pressed`/`mouse.*` and HiDPI-correct mouse coordinates. No hand-written winit/wgpu host is
  needed for the first renderer.
- It is the most demanding real test the kernels will have had outside LLM inference (atomics, workgroup
  memory, barriers, packed bytes).

## Status (2026-10-20)

- **Step 0 done**: `boring-ui/render/src/raster.br` (+ `raster_font.br`) draws `FillRect`, `StrokeRect`, `DrawShadow`,
  `DrawText` (5x7 fixture font), `PushClip`/`PopClip`, `PushOpacity`/`PopOpacity`. `DrawImage` and `MarkedText` are
  not drawn yet. The colour pipeline is still sRGB-encoded source-over; only `blend` changes when it is decided.
- **Step 1 done**: `raster_gpu.br` is the one-thread-per-pixel kernel; `encode` bakes clips, expands text to glyph
  commands and bounds opacity nesting to 4. It matches the CPU reference within one LSB on the test corpus under
  `boring run`, `--target simd` and `--target wgpu` (Metal). Not yet measured: speed.
- The renderer lives in its own project (`render/`) so it can be built for kernel targets: they cannot yet
  compile the view runtime. Gaps found on the way, each filed as a compiler task: kernel targets do not add the
  unicode crates nor read `[dependencies]`; reading a `[uint]` kernel field back is mistyped on the host; kernel
  struct helper methods work under `boring run` only (top-level functions work everywhere); a kernel local named
  like a host function pulls that function into the WGSL.

## What stays outside Boring kernels

Even on this route the platform host depends on existing crates; none of them is a rendering technology:

| Concern | Provided by |
|---|---|
| Window, event loop, surface | `Screen` (winit 0.30 + wgpu under the hood) |
| Text shaping, bidi, editing primitives | `cosmic-text` (per the main draft), via `[external_types]` |
| Glyph rasterization | CPU, into an atlas the kernel samples (see step 3) |
| Accessibility tree | AccessKit |
| Image / font decoding, clipboard, IME, software keyboard | existing crates / platform services |

## Contract

The renderer consumes the display list from [`boring-ui/src/scene.br`](../../../boring-ui/src/scene.br)
(`[Command]`) and produces a `[uint]` pixel buffer. It never calls `body()` or performs layout, and
components never touch it.

Two representations are needed, because `Command` carries strings (`kind`, `text`, `color`, `hex`):

1. `Command` — the semantic, deterministic list (unchanged, used by tests and hit testing).
2. **Resolved commands** — a flat encoding built by one CPU pass: `kind` as an integer opcode, color as a
   premultiplied `uint`, geometry as `float`s, text as glyph-run offsets into an atlas. Stored as parallel
   arrays (struct-of-arrays) so kernels read plain numeric buffers.

**Open decision (must be settled before step 1):** color pipeline. The main draft specifies premultiplied
colors in a *linear* working space; `Screen` presents BGRA8. The sRGB conversion belongs in one place — the
final write of the kernel (and the same function in the CPU reference) — so both produce identical bytes.

## Incremental plan

Each step is independently testable and ends with the CPU reference as the oracle.

### Step 0 — CPU reference renderer in pure Boring (no language work expected)

- `boring-ui/src/raster.br`: `[Command] → [uint]` for `FillRect`, `StrokeRect`, `DrawShadow`,
  `PushClip`/`PopClip`, `PushOpacity`/`PopOpacity`. Text is a placeholder rectangle until step 3.
- First increment: `FillRect` + rectangular `PushClip` + PPM output, exercised on `examples/todo`.
- Golden tests under `boring run`, wired into `tests/run_tests.sh` and `run_tests_transpiled.sh`.
- Fixes the exact semantics: rounding, edge anti-aliasing, paint order, group opacity (no darkening of
  overlapping descendants), and the color pipeline above.

### Step 1 — same renderer as a one-thread-per-pixel kernel

- The encoding pass produces the resolved commands.
- A `'surface` kernel walks the command list in paint order per pixel: O(pixels × commands), intentionally
  naive.
- Acceptance: equal to the CPU reference within 1 LSB, first on `--target simd`, then wgpu and Metal.
- Expected friction: struct-of-arrays buffers, a clip stack inside a kernel, nested control flow. Each one is
  a task, not a workaround.

### Step 2 — tiled rendering

- One workgroup per 16×16 tile. Bin commands per tile with atomics (`'actor'global`), then render from
  workgroup memory (`'actor`) with barriers.
- This is the real architecture (Vello/Impeller-like at a fraction of the size: two or three stages, not ten)
  and the point where performance becomes measurable against the naive kernel.
- Acceptance: pixel-identical to step 1.

### Step 3 — text

- Glyph atlas stored as `[uint8]` (packed byte buffers are supported on wgpu since 2026-09-28). `DrawText`
  becomes an alpha-mask blit.
- First with the layout fixture font, then `cosmic-text` shaping/rasterization feeding the atlas via
  `[external_types]`.
- Atlas eviction invalidates only the affected entries, per the main draft.

### Step 4 — runtime integration (where real language/runtime additions are likely)

Known gaps in `Screen` today, each to be filed as its own task when reached:

| Gap | Detail |
|---|---|
| Continuous loop | `kernel: loop:` renders every frame; a UI wants redraw-on-demand, coalesced with invalidation |
| Event vocabulary | only `key("k")` (literal), `key_pressed`, `mouse.*`; missing text input, modifiers, wheel, IME, focus |
| Resize | `screen.resized` is detected but the surface buffer is not reallocated (documented limitation) |
| Scale factor | logical/physical conversion at the host boundary, per the main draft |

The platform host then drives `Runtime.flush → layout → displayList → encode → kernel → present`.

## Risks and mitigations

- **Generated WGSL/MSL quality** decides whether the tiled kernels are competitive. Mitigation: compare
  step 1/2 timings against a hand-written WGSL baseline early.
- **Backend divergence** (Metal vs wgpu vs SIMD). Mitigation: the CPU reference is the oracle for every
  backend; run the same golden set on each.
- **Bugs found on the way** (the uint8 packed-buffer bug found in boring-llm is the precedent). Mitigation:
  one task per bug, no workarounds in shared libraries.
- **Scope creep toward Vello.** Mitigation: only grow the vocabulary the draft already lists; paths,
  gradients, blur and filters stay deferred and are added through the versioned command format.

## Fallback

If the kernel route stalls, the display list lets us swap in an alternative renderer without touching `View`,
layout or the runtime: Skia (fastest to a correct result) or Vello (Rust-only), with `tiny-skia` or the
step-0 CPU renderer kept as the headless oracle either way.
