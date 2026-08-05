# docker-wgpu-check

Checks a `boring build --target wgpu` generated project's `shaders/*.wgsl`
with `naga` — the same WGSL front-end `wgpu` itself uses at runtime — no GPU
required at all, and optionally executes the whole generated project for
real against a **software** Vulkan device (Mesa's lavapipe), so this tool
can go further than `docker-nvcc-check`/`docker-hipcc-check`: it can
genuinely run the compute dispatch, not just statically check the shader
source.

## What this actually checks

**`MODE=validate`** (default):

1. `naga shaders/main.wgsl` (and `shaders/main_emulated.wgsl` when the
   `gpu.warp.*` emulated-fallback module is present) — parses and
   type-checks the WGSL, exactly what `wgpu` does internally before it ever
   touches a device. No Vulkan, no GPU, no device involved.

**`MODE=run`** (opt-in):

2. All of the above, then `cargo run --release` inside the mounted project
   with `WGPU_BACKEND=vulkan`, against `mesa-vulkan-drivers`' lavapipe — a
   real (CPU-emulated) Vulkan implementation. `wgpu` opens a genuine adapter,
   allocates real buffers, dispatches the real compute shader, and reads
   back real results. This actually exercises correctness (not just syntax)
   without needing any GPU hardware — slow (CPU-bound), but a real run.

## What this does NOT check

- **`MODE=validate`** does not run anything — no numerical correctness,
  workgroup-size/dispatch-count mismatches, or runtime validation-layer
  errors are exercised, only WGSL syntax/type validity.
- **`MODE=run`** still isn't a real GPU: lavapipe is a software
  rasterizer/compute path, so timing/performance numbers from it are
  meaningless and some vendor-specific behavior (subgroup/`gpu.warp.*`
  support, real memory bandwidth) can't be exercised — check
  `wgpu::Features::SUBGROUP` support separately on real hardware for that.
- First `cargo run --release` in `MODE=run` needs network access
  (crates.io) and is a full build — much slower than `MODE=validate` or the
  CUDA/ROCm docker checks.

**Necessary, not sufficient** — a clean `MODE=validate` run is a fast static
filter; a clean `MODE=run` is a real but CPU-emulated execution, still not a
substitute for testing on an actual GPU-backed adapter (Vulkan/Metal/DX12).

## Usage

Generate the project first (on whichever machine has `boring` built —
building `boring` itself needs no GPU, it's a plain Rust CLI):

```sh
boring build --target wgpu path\to\file.br
```

Then, with the generated `<stem>_wgpu` project copied over:

```powershell
.\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_wgpu
.\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_wgpu -Mode run
```

Any `naga FAILED` is a genuine bug in the transpiler's WGSL codegen. Any
`cargo run FAILED` under `-Mode run` is either a host-code bug or a real
runtime/validation error from wgpu itself.
