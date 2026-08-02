# docker-nvcc-check

Compiles a `boring build --target cuda` generated project's `kernels/main.cu`
with the **real** `nvcc`/`ptxas` — no NVIDIA GPU required, because device-code
compilation to PTX/cubin never touches a driver or a device, only the CUDA
toolkit. Meant for a machine with no NVIDIA hardware at all (e.g. a Windows
box with no GPU): just Docker Desktop (WSL2 backend), which runs the
`linux/amd64` image natively on an x86_64 Windows host — no emulation layer,
unlike running the same image on an Apple Silicon Mac.

## What this actually checks

For each architecture in `$ARCHS` (default: every real GPU in
[`gpu-profiles/`](../../gpu-profiles/) plus Colab's T4):

1. `nvcc -arch=sm_XX --ptx kernels/main.cu` — is the generated CUDA C valid,
   does it compile to PTX for that compute capability.
2. `ptxas -arch=sm_XX` on that PTX — does it assemble to a real cubin for
   that architecture.

| Profile | Arch |
|---|---|
| v100 | `sm_70` |
| — (Colab free tier) | `sm_75` |
| a100 | `sm_80` |
| default, rtx3090 | `sm_86` |
| rtx4090 | `sm_89` |
| h100 | `sm_90` |

A clean run here is a real, useful check that `tools/fake-nvcc` structurally
cannot do (it stubs `nvcc` out entirely and never looks at the `.cu` file) —
and it covers more GPU generations statically than a single physical/cloud
GPU ever could in one run.

## What this does NOT check

- **Host Rust code** (`main.rs`, `build.rs`) — not built at all here. See
  `tools/fake-nvcc` for that.
- **Execution** — no kernel ever runs, so numerical correctness, launch
  parameters, race conditions, and memory hazards are not exercised. See
  `tools/colab-cuda-smoke-test.ipynb` for that.

**Necessary, not sufficient** — same caveat as `fake-nvcc`: a clean run here
is a fast static filter, not a replacement for an actual GPU test.

## Usage

Generate the project first (on whichever machine has `boring` built —
building `boring` itself needs no GPU, it's a plain Rust CLI):

```sh
boring build --target cuda path\to\file.br
```

Then, on the Windows machine, with the generated `<stem>_cuda` project
copied over:

```powershell
.\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_cuda
```

Any `nvcc FAILED` or `ptxas FAILED` in the output is a genuine bug in the
transpiler's CUDA backend for that architecture. Restrict to specific
architectures with `-Archs "80 90"`.
