# docker-hipcc-check

Compiles a `boring build --target rocm` generated project's `kernels/main.hip`
with the **real** `hipcc` device-code compiler — no AMD GPU required, because
`--genco` ("generate code object") compiles device code to a loadable
fat-binary offline, the HIP analogue of `nvcc --ptx`, and never touches a
driver or `/dev/kfd`. Meant for a machine with no AMD hardware at all: just
Docker Desktop (WSL2 backend), which runs the `linux/amd64` image natively on
an x86_64 Windows host — no emulation layer, unlike running the same image on
an Apple Silicon Mac.

This is the ROCm counterpart to
[`tools/docker-nvcc-check`](../docker-nvcc-check/README.md) — same rationale,
same caveats, different toolchain.

## What this actually checks

For each architecture in `$GFX_ARCHS` (default: a spread of consumer and
datacenter AMD GPUs):

1. `hipcc --genco --offload-arch=gfxXXX kernels/main.hip` — is the generated
   HIP C++ valid, does it compile to a loadable code object for that
   architecture.

| gfx target | GPU family |
|---|---|
| `gfx1030` | RDNA2 (RX 6000 series) |
| `gfx1100` / `gfx1101` | RDNA3 (RX 7000 series) |
| `gfx90a` | CDNA2 (MI200) |
| `gfx942` | CDNA3 (MI300) |

A clean run here covers more GPU generations statically than a single
physical GPU ever could in one run — including datacenter archs (`gfx90a`,
`gfx942`) nobody is likely to own locally.

## What this does NOT check

- **Host Rust code** (`main.rs`, `build.rs`) — not built at all here.
- **Execution** — no kernel ever runs, so numerical correctness, launch
  parameters, race conditions, and memory hazards are not exercised. Real
  execution still needs an actual AMD GPU (ROCm has no official simulator,
  same situation as CUDA).

**Necessary, not sufficient** — same caveat as `docker-nvcc-check`: a clean
run here is a fast static filter, not a replacement for an actual GPU test.

## Usage

Generate the project first (on whichever machine has `boring` built —
building `boring` itself needs no GPU, it's a plain Rust CLI):

```sh
boring build --target rocm path\to\file.br
```

Then, on the Windows machine, with the generated `<stem>_rocm` project
copied over:

```powershell
.\run.ps1 -ProjectDir C:\path\to\examples\vector_add_gpu_rocm
```

Any `hipcc FAILED` in the output is a genuine bug in the transpiler's ROCm
backend for that architecture. Restrict to specific architectures with
`-GfxArchs "gfx1100 gfx942"`.

## Note on image size

`rocm/dev-ubuntu-22.04` bundles the full ROCm LLVM/HIP toolchain and is
several GB — expect a slow first `docker build` (image pull), same as
`nvidia/cuda:*-devel`.
