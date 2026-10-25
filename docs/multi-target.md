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

## How a build runs

The program runs **once, on the CPU host** (the simd backend, always part of a multi-target
build). Every kernel launch asks the run-time router (`unified.rs`) where to run:

1. If the launch is small (under 16384 threads, `BORING_GPU_MIN_THREADS`), it stays on the CPU:
   uploading, launching and reading back costs more than computing.
2. Otherwise the backends ranked above the CPU (see "Start-up policy") are tried in order. A
   backend takes the launch if it has an **adapter** for that kernel, can initialise its device,
   and accepts this launch.
3. If none does, the CPU runs the kernel itself.

An adapter takes the kernel's fields as host copies, uploads them, launches the backend's own
build of the kernel, and copies every `mut`/`var` buffer back, so the CPU host and later launches
see the result. The cost is one transfer each way per launch; keeping data resident between
launches is not implemented.

A kernel is routable when its fields are numeric/bool scalars and plain `[T]` buffers in
`'unified` or `'global` memory. Anything else (`'surface`, shared memory, atomics, struct or
fixed-array fields, generic kernels, programs with a `Screen`) runs on the CPU. An adapter can
also decline one launch: wgpu fixes the workgroup size in the shader from the kernel's launch
site, so a launch with another block size goes elsewhere.

## The generated crate

`<stem>_multi/` is one Cargo project:

| File | Content |
|---|---|
| `src/backend_<name>.rs` | that backend's host code and its kernel adapters, behind the Cargo feature `<name>` |
| `src/unified.rs` | the router: `dispatch`, the `Arg` type, adapter registry (standard library only) |
| `src/policy.rs` | the start-up policy (standard library only) |
| `src/main.rs` | probes the compiled-in backends, ranks them, installs the routes, runs the program |
| `kernels/`, `shaders/` | the device sources, one per backend (`main.cu`, `main.hip`, `main.metal`, `main.wgsl`) |
| `build.rs` | one section per toolchain step (`nvcc`, `hipcc`), run only for enabled features |
| `Cargo.toml` | one feature per backend; crates only one backend needs are optional |

If backends need different versions of a crate (for example `winit` 0.28 and 0.30 in programs with
a `Screen`), each gets it under its own key and its module imports it back under the real name.
A backend that cannot compile the program is left out with a warning. The crates require Rust
1.89 or later (`File::try_lock`).

## Start-up policy

1. **Probe.** Each compiled-in backend reports the device it would use, or that it is unusable
   (no driver, no adapter). Probing never prints a panic.
2. **Measure.** Each usable backend runs the same compute-bound benchmark once (1M elements,
   256 dependent multiply-adds each; the score is millions of multiply-adds per second). A
   memory-bound kernel would fit in a CPU cache and wrongly favour the CPU. Scores are cached per
   device for 30 days in `$BORING_CACHE_DIR`, else `~/.cache/boring` (`%LOCALAPPDATA%\boring` on
   Windows). A different device, or an expired entry, measures again.
3. **Load.** A running program holds a *lease* on each backend it uses: an operating-system file
   lock that disappears when the process exits or crashes. Other programs count the leases held
   on each backend as its load.
4. **Rank.** `score / (1 + load)`, best first. A backend earlier in the static order (cuda, rocm,
   metal, wgpu, simd) wins over a later one that is less than 25% faster, since measurements are
   noisy and a native backend is the safer choice. The CPU is ranked too: only the backends above
   it receive kernels.

A backend that cannot be measured keeps a fixed prior (cuda 10M, rocm 8M, metal 2M, wgpu 1M,
simd 0.1M). ROCm is always in that case: its kernels are code objects built ahead of time, so
there is no run-time compiler to build a benchmark with. The CUDA benchmark goes through nvrtc.

## Environment variables

| Variable | Effect |
|---|---|
| `BORING_BACKEND=a,b` | offer kernels to these backends, in this order; no benchmark and no small-launch rule. The CPU runs what none takes |
| `BORING_BACKEND=list` | print each backend with its device and cached score, then exit |
| `BORING_BENCH=off` | skip ranking: the static order applies |
| `BORING_BENCH=force` | measure again, ignoring the cache |
| `BORING_BACKEND_VERBOSE=1` | print the ranking and where each kernel runs |
| `BORING_GPU_MIN_THREADS=n` | launches under `n` threads stay on the CPU (default 16384) |
| `BORING_CACHE_DIR=dir` | where scores and leases are kept |

## Not implemented yet

- Placement hints in the source (a preferred or required class of backend for a kernel or a
  launch) and an inventory of the backends from Boring code.
- Device residency between launches: each routed launch copies its buffers in and out.
- Routing for programs with a `Screen`, and for kernels with shared memory, atomics or struct
  fields.
