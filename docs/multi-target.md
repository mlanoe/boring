# Multi-target builds

One binary can contain several GPU/CPU backends and pick one when it starts.

```sh
boring build --target all main.br          # every backend this host can build
boring build --target cuda,metal main.br   # an explicit list
cd main_multi && cargo build --release
```

`--target` takes a backend (`cuda`, `rocm`, `metal`, `wgpu`, `simd`), `all`, or a comma-separated
list. `all` keeps what the host can build — Metal only on macOS, CUDA when `nvcc` is on `PATH`,
ROCm when `hipcc` is found, wgpu and simd always — and prints the others as skipped with the
reason. An explicit list is taken as written. `--linux` and `--emit-rust` cannot be combined with
several targets.

## There is no placement syntax

A kernel does not name the backend or device it runs on. `Device(...)`, `Device.all()`,
`kernel(device) Name(...)` and `kernel(target = ..., device = ...)` were removed (the compiler
rejects them with a diagnostic): the choice belongs to the run-time policy below, so the same
source works unchanged on any machine and with any subset of backends.

```boring
mut scale = Scale(n)
scale(block = 256, grid = 64)
```

## The generated crate

`<stem>_multi/` is one Cargo project:

| File | Content |
|---|---|
| `src/backend_<name>.rs` | that backend's host code, behind the Cargo feature `<name>` |
| `src/policy.rs` | the start-up policy (standard library only) |
| `src/main.rs` | probes the compiled-in backends, ranks them, runs the winner |
| `kernels/`, `shaders/` | the device sources, one per backend (`main.cu`, `main.hip`, `main.metal`, `main.wgsl`) |
| `build.rs` | one section per toolchain step (`nvcc`, `hipcc`), run only for enabled features |
| `Cargo.toml` | one feature per backend; crates only one backend needs are optional |

If backends need different versions of a crate (for example `winit` 0.28 and 0.30 in programs with
a `Screen`), each gets it under its own key and its module imports it back under the real name.
The crates require Rust 1.89 or later (`File::try_lock`).

## Start-up policy

1. **Probe.** Each compiled-in backend reports the device it would use, or that it is unusable
   (no driver, no adapter). Probing never prints a panic.
2. **Measure.** Each usable backend runs the same compute-bound benchmark once (1M elements,
   256 dependent multiply-adds each; the score is millions of multiply-adds per second). A
   memory-bound kernel would fit in a CPU cache and wrongly favour the CPU. Scores are cached per
   device for 30 days in `$BORING_CACHE_DIR`, else `~/.cache/boring` (`%LOCALAPPDATA%\boring` on
   Windows). A different device, or an expired entry, measures again.
3. **Load.** A running program holds a *lease* on its backend: an operating-system file lock that
   disappears when the process exits or crashes. Other programs count the leases held on each
   backend as its load.
4. **Rank.** `score / (1 + load)`, best first. A backend earlier in the static order (cuda, rocm,
   metal, wgpu, simd) wins over a later one that is less than 25% faster, since measurements are
   noisy and a native backend is the safer choice.

A backend that cannot be measured keeps a fixed prior (cuda 10M, rocm 8M, metal 2M, wgpu 1M,
simd 0.1M). ROCm is always in that case: its kernels are code objects built ahead of time, so
there is no run-time compiler to build a benchmark with. The CUDA benchmark goes through nvrtc.

## Environment variables

| Variable | Effect |
|---|---|
| `BORING_BACKEND=a,b` | use the first usable backend of this list, in this order; no benchmark |
| `BORING_BACKEND=list` | print each backend with its device and cached score, then exit |
| `BORING_BENCH=off` | skip ranking: the static order applies |
| `BORING_BENCH=force` | measure again, ignoring the cache |
| `BORING_BACKEND_VERBOSE=1` | print the ranking and the chosen backend |
| `BORING_CACHE_DIR=dir` | where scores and leases are kept |

## Not implemented yet

The policy ranks whole programs. Choosing a device per kernel (from the kernel's cost and the
current load) needs a backend-independent kernel interface that the five host emitters do not
share yet. Within one program, every kernel runs on the chosen backend's default device.
