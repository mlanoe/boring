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

## Placement hints

A kernel has no device in its source, but it can say what kind of device it wants. The classes
are `.cpu`, `.gpu`, and the two kinds of GPU, `.discrete` (a card with its own memory: CUDA, ROCm, a discrete wgpu adapter) and `.integrated` (a GPU sharing the CPU's memory: Apple silicon, an integrated adapter). `.discrete` and `.integrated` are narrower than `.gpu`:

```boring
@prefer(.gpu)              # wish: skip the small-launch rule, fall back to the CPU if no GPU can
kernel MatMul:
    ...

@only(.cpu)                # requirement: always the CPU
kernel Tally:
    ...

kernel:
    big(block = 256, grid = 4096, prefer = .cpu)    # this launch only; wins over @prefer
```

| Hint | Effect |
|---|---|
| `@prefer(.gpu)` / `prefer = .gpu` | the small-launch rule does not apply; the CPU still runs it if no backend can |
| `@prefer(.cpu)` / `prefer = .cpu` | the CPU runs it, even a large launch |
| `@only(.cpu)` | always the CPU, even with `BORING_BACKEND` |
| `@only(.gpu)` | a GPU backend must run it: the launch fails with an error otherwise (no GPU adapter, a launch the adapter cannot run, `BORING_BACKEND=simd`) |

A launch's `prefer =` wins over the kernel's `@prefer`. `BORING_BACKEND=a,b` overrides a `prefer`
(the user knows this machine) but never an `@only`. `@only` is a requirement a single-target
build must also meet: `boring build --target simd` rejects an `@only(.gpu)` kernel, and a GPU
target rejects an `@only(.cpu)` one. `prefer =` and `@prefer` are accepted and ignored by
single-target builds and by `boring run`. `@prefer(.discrete)` tries the discrete GPUs first and falls back to any backend, while
`@only(.discrete)` fails the launch when none runs it. A wgpu adapter of an unknown kind counts
as `.gpu` only. `kernel.backends()` reports each backend's `class` (`discrete`, `integrated`, `gpu` when
unknown, `cpu`). Anything other than these four is a compile error.

## Where did it run?

```boring
kernel:
    k(block = 256, grid = 4096)
print "scale ran on {kernel.lastChoice(k)}"     # cpu, wgpu, metal, cuda or rocm
```

`kernel.lastChoice(k)` is the name of the backend `k`'s most recent launch ran on. In a
multi-target build it is the router's answer; in a single-target build it is that target (`cpu`
for simd); `boring run` has no backends and always answers `cpu`. It takes a kernel variable.

## Which backends can I use?

```boring
for b in kernel.backends():
    print "{b.name} ({b.kind}): score {b.score}, load {b.load}"
```

`kernel.backends()` lists the backends the program may run kernels on, best first, the CPU
among them (`b.name` is `cpu`, `wgpu`, `metal`, `cuda` or `rocm`; `b.kind` is `gpu` or `cpu`;
`b.score` is the start-up benchmark score, or the device-table score of a backend that cannot be measured;
`b.load` is how many other programs were using it at start-up). It lists only what this run
installed: with `BORING_BACKEND=wgpu,simd` that is those two. A single-target build lists its own
target and `boring run` lists the CPU, both with score 0 and load 0.

## How a build runs

The program runs **once, on the CPU host** (the simd backend, always part of a multi-target
build). Every kernel launch asks the run-time router (`unified.rs`) where to run:

1. If the launch is small (under 16384 threads, `BORING_GPU_MIN_THREADS`), it stays on the CPU:
   uploading, launching and reading back costs more than computing.
2. Otherwise the backends ranked above the CPU (see "Start-up policy") are tried in order. A
   backend takes the launch if it has an **adapter** for that kernel, can initialise its device,
   and accepts this launch.
3. If none does, the CPU runs the kernel itself.

Buffers stay on the backend between launches. The router tracks, for each kernel instance and
each buffer, whether the backend's copy is valid and whether it is newer than the CPU's:

- an unchanged input is uploaded once, however many times the kernel is launched;
- a result is copied back only when the CPU needs it: the program reads the field
  (`k.buf[i]`), edits it (`k.buf[i] = v`), launches the kernel on the CPU, or the instance moves
  to another backend (its newer data comes back first);
- when the program drops the kernel, the backend's copy is freed.

Scalars are re-sent at every launch. `BORING_BACKEND_VERBOSE=1` ends with
`boring: buffer transfers: N up, M down`.

Reading one element (`k.buf[i]`) reads just that element from the backend; reading the field
whole (`let result = k.buf`) downloads it all. To read many elements after a launch, bind the
field once and read the binding: each element read of a buffer that lives on a backend is a
small transfer of its own. An element read of a buffer the CPU already holds costs nothing.

The ping-pong swap `a.x, a.y = a.y, a.x` and the copy `b.cells = a.cells` also stay on the
backend when both buffers are resident on it: the swap exchanges the two device buffers, the copy
is a device-side copy. Otherwise (another backend, a buffer the CPU changed, a backend with no
device-side copy: ROCm has none, wgpu does it for 8-, 16- and 32-bit element types) they go through the
CPU as before.

A kernel is routable when its fields are numeric/bool scalars, `Dimension`s and plain `[T]`
buffers in `'unified`, `'global`, `'surface`, `'actor'unified` or `'actor'global` memory.
Block-shared (`'actor`) and per-thread (`'local`) fields have no host data and are simply not
arguments, so a kernel with a shared-memory tile is routed; an atomic buffer is a buffer. A
fixed array in `'const` memory is a read-only buffer, and a buffer of structs is already split into
one buffer per field before the backends see it, so both are routed. Generic kernels and fields of
any other type run on the CPU. An adapter can also decline one
launch: wgpu fixes the workgroup size in the shader from the kernel's launch site, so a launch
with another block size goes elsewhere. A backend that cannot compile a kernel (Metal has no
64-bit signed atomics, wgpu rejects a shader it cannot validate) declines it the same way: the
next backend, or the CPU, runs it (`BORING_BACKEND_VERBOSE=1` says why). A backend compiles its
kernels together, so one kernel it rejects takes the others of the program with it.

A device error during a launch (a lost context, out of memory, a failed launch) is not fatal
either: the backend is given up for the rest of the run (`boring: <backend> failed on kernel <K>:
...` on stderr), the instance's data comes back from it, and the launch goes to the next backend
or the CPU. If the device cannot give the data back, the read reports it and the data is lost.

### Programs with a `Screen`

The `Screen` lives on the CPU host, which presents its own copy of the pixels. A `'surface`
buffer is a buffer like any other for the router: kernels that write it run on a backend, and
`screen.present(k.pixels)` (or any host read) first brings the newest pixels back from it. A
render loop such as `examples/game_of_life.br` therefore runs its kernels on the best GPU while
the window stays on the CPU host. The pixels come back to the CPU every frame (the swap and the copy of a ping-pong
loop stay on the backend), so a light kernel can still be faster left on the CPU:
`@prefer(.cpu)` says so.

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

A backend that cannot be measured is scored from an embedded table of known devices
(`DEVICES` in `src/multi_policy.rs`): the device's approximate peak FP32 throughput from the
vendors' public figures (GeForce RTX 50/40/30/20 and GTX 16/10, data-centre NVIDIA, Radeon RX
7000/6000 and Instinct, Intel Arc, Apple M1 to M4), converted to benchmark points (60 thousand
per TFLOPS, calibrated on an Apple M3, and 60% of that through wgpu since it reaches less of the
device than the native API). A device the table does not know keeps a fixed prior per backend
(cuda 1.2M, rocm 0.9M, metal 0.4M, wgpu 0.25M, simd 0.1M). ROCm is always in that case: its
kernels are code objects built ahead of time, so there is no run-time compiler to build a
benchmark with. The CUDA benchmark goes through nvrtc.

The table is approximate: it orders very different devices (a discrete GPU against an
integrated one, an RX 6600 against a CPU), not close ones, and a measured score always replaces
it. Run with `BORING_BACKEND_VERBOSE=1` to see which scores are priors.

## Environment variables

| Variable | Effect |
|---|---|
| `BORING_BACKEND=a,b` | offer kernels to these backends, in this order; no benchmark and no small-launch rule. The CPU runs what none takes |
| `BORING_BACKEND=list` | print each backend with its device and cached score, then exit |
| `BORING_BENCH=off` | measure nothing: rank by the embedded device table and the load |
| `BORING_BENCH=force` | measure again, ignoring the cache |
| `BORING_BACKEND_VERBOSE=1` | print the ranking and where each kernel runs |
| `BORING_FAULT=backend:Kernel:reject\|fail` | testing: make a backend refuse, or fail with a device error on, a kernel |
| `BORING_GPU_MIN_THREADS=n` | launches under `n` threads stay on the CPU (default 16384) |
| `BORING_CACHE_DIR=dir` | where scores and leases are kept |

## Not implemented yet

- A remote service as a backend.
- Device-side copy and ranged reads on ROCm (needs a HIP device-to-device copy in its wrapper).
