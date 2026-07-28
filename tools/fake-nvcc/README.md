# fake-nvcc

A stub `nvcc` for smoke-testing `boring build --target cuda`'s generated Rust
**host** code on a machine with no CUDA toolkit — this repo's own dev
environment (macOS) has no path to a real NVIDIA GPU at all, so without this,
every CUDA-backend change is completely unverifiable locally, and the first
real feedback would come from CI or a contributor's machine.

## What this actually checks

`boring build --target cuda` emits a Cargo project whose `build.rs` shells
out to `nvcc --ptx -O2 --output-file <path> kernels/main.cu` to compile the
kernel source, then makes that PTX path available to `main.rs` via
`BORING_PTX_PATH` — but that env var is only read at **runtime**
(`nvrtc`/`cuModuleLoad`), not needed for `main.rs` to type-check or compile.
This stub does nothing with `kernels/main.cu` — it just finds `--output-file`
and writes an empty placeholder there so `build.rs` succeeds, letting
`cargo check`/`cargo build` proceed all the way through compiling the
generated Rust against the **real** `cudarc` crate.

That's a real, useful check: it catches exactly the class of bug a change to
`src/transpiler/cuda/host.rs` is most likely to introduce — a type mismatch
between what the transpiler emits and cudarc's actual API (wrong buffer
element type, wrong method signature, a botched `Result`/`?` chain, and so
on). It is not hypothetical: this tool caught a real `CudaSlice<f32>` vs.
`CudaSlice<f64>` mismatch (Metal's MSL forces `f32`; CUDA stores `f64`
natively — a mismatch introduced by porting Metal-shaped code without
checking this) during development, on this exact machine, with no other way
to have found it before a contributor's GPU did.

## What this does NOT check

- **The actual CUDA kernel source** (`kernels/main.cu`) — never touched,
  never compiled, might not even be valid CUDA C.
- **Whether the program runs, or runs correctly** — no GPU execution happens
  at all. Numerical correctness, kernel launch parameters, race conditions,
  memory hazards — none of this is exercised.
- **Linking against real CUDA driver libraries** — `cudarc`'s
  `dynamic-loading` feature means the actual `libcuda`/`libnvrtc` are
  `dlopen`'d at runtime, not linked at build time, so a clean `cargo build`
  here proves nothing about whether those libraries are even present, let
  alone compatible, on a real target machine.

**A clean run of this tool is necessary, not sufficient.** Treat it as a fast
first-pass filter before a real GPU test, never as a replacement for one —
the exact lesson of the Metal backend's `fastMathEnabled` bug, which no
amount of type-checking could have caught (it produced a real `NaN` from a
provably finite input, discovered only by actually running the model).

## Usage

```sh
export PATH="$(pwd)/tools/fake-nvcc:$PATH"
cd path/to/generated/project   # from `boring build --target cuda`
```

`cudarc`'s own `build.rs` additionally requires a concrete `cuda-XXXXX`
feature (it won't guess a CUDA version) — add one to the generated
`Cargo.toml`'s `cudarc` dependency before checking, e.g.:

```toml
cudarc = { version = "0.19", features = ["driver", "nvrtc", "cuda-12060"] }
```

Then:

```sh
cargo check     # or `cargo build --release`
```

Any real `error[...]` in the output is a genuine bug in the transpiler's
CUDA backend. Warnings (unused imports, snake_case naming, redundant
parens) are pre-existing codegen noise, not something this tool is meant to
police.
