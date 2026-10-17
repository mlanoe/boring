# CPU SIMD target

`boring build --target simd program.br` generates `program_simd/`, a standalone
Rust crate that runs Boring kernels on the CPU and needs no GPU runtime.

## Execution model

The launch grid is unchanged. `block=(x,y,z)` describes work-items in one CPU
block and `grid=(x,y,z)` describes the number of blocks. Every work-item sees the
same implicit `.thread`, `.block`, and `.block_dim` coordinates as the GPU
backends. The x dimension is a tight contiguous loop, allowing release-mode LLVM
to choose its own vector width and tail strategy for independent work-items.

Fixed-shape labeled arrays retain their row-major layout and support labeled
kernel indexing. Fixed arrays and fixed-shape kernel outputs are allocated to
their declared size even when `init` does not assign them explicitly.
Kernel references to immutable top-level scalar constants preserve their device
numeric type, including mixed integer/`float32` expressions and explicit scalar
constructors such as `float32(value)`.

At runtime the generated crate selects AVX-512, AVX2, SSE2, or AArch64 NEON.
Unknown CPUs use the baseline ISA path, which is the correctness-preserving
fallback.
The x86 paths are separate `#[target_feature]` functions, allowing LLVM to emit
instructions unavailable in the baseline binary without making that binary
unsafe on older processors. Set `BORING_SIMD_SCALAR=1` to force the baseline ISA
path for debugging, portability checks, or result comparison. LLVM may still
auto-vectorize that baseline path when the architecture guarantees vector
instructions (notably AArch64 NEON), so the variable does not promise scalar
machine code.
The output is one target-specific crate today; comma-separated targets such as
`--target wgpu,simd` are intentionally not accepted yet.

`'kernel'unified` and `'kernel'global` are ordinary CPU vectors. There is no
host/device copy, although the generated API keeps explicit copy methods so it
shares the host-lowering boundary with GPU targets. Resident (`'kernel'unified`)
function results and kernel-to-kernel buffer aliases are plain `Vec` copies.
Thread-local kernel fields are per-work-item locals, and `'actor'global` /
`'actor'unified` atomics (`+=`, `.min`, `.max`, `.swap`, `.cas`) are plain
read-modify-writes that return the previous value, since general kernels run on one
thread. `'surface` buffers are ordinary `Vec<u32>` fields (see "GPU display" below).
Identity and runtime type-test operators are rejected by the SIMD code generator
rather than emitted with altered semantics. Unsupported
kernel statements and expressions likewise produce source diagnostics.
Integer bitwise operators (`&`, `|`, `^`, shifts, and `~`) preserve their scalar
Rust semantics inside kernels. Free math functions (`sin cos tan asin acos atan sinh
cosh tanh exp exp2 sqrt cbrt floor ceil round trunc log log2 log10 abs pow atan2 min
max`) lower to the matching Rust method at the operand's own float width.

Kernel `int` is a 32-bit device integer; host `int` is `isize`. Calls from a
kernel to an ordinary Boring function convert numeric arguments and results at the
call. A user `struct` used as a kernel field type (numeric and `bool` fields only)
is mirrored as `__Device_<Name>` at device widths and converted when the host
assigns it.

## Barriers and shared memory

Bare `'actor` storage is block-shared memory and `sync` is a block barrier, with
the same implicit discipline as the GPU backends: with no explicit `sync`, one
barrier is inserted before the first loop and another at the top of every loop
iteration that touches shared storage.

Two lowerings implement barriers:

* **Phase splitting** (fast path). When every barrier is block-uniform and the
  state carried across it is replayable, the work-item loop is split into
  collective phases that LLVM can still vectorize.
* **Cooperative fallback**. Any kernel the phase splitter rejects (per-thread
  accumulators carried across a `sync`, divergent control around a barrier,
  shuffles of computed values, ...) is emitted as one closure per work-item and
  run by a small block scheduler: each work-item is a stackful fiber
  (hand-written AArch64 context switch; on other targets one parked OS thread per
  work-item with strict baton passing). `sync` parks a work-item until every live
  work-item of the block arrives; a work-item that `return`s early stops taking
  part, as on a GPU. A barrier reached by only part of its group aborts with a
  `deadlock` message instead of hanging. Fibers have 256 KiB stacks
  (`BORING_SIMD_STACK_KB`); an overflow is detected and aborts.

The thread backend can be forced for testing with
`RUSTFLAGS="--cfg boring_coop_threads"`. It is correct but much slower than fibers
(blocks of 256 work-items cost one thread each); an x86-64 fiber backend is future
work.

## Warp and tensors

The logical warp is **32 lanes wide on every ISA path**, matching Metal's SIMD-group
and CUDA's warp so kernels written against `.warp.size` or assuming 32 behave
identically. `.warp.size`, `.warp.lane`, `.warp.sync()` and the four shuffles
(`shuffle`, `shuffle_down`, `shuffle_up`, `shuffle_xor`) are supported with any
operand: a warp is 32 consecutive work-items by linear thread index, a shuffle whose
source lane is outside the block or has already returned yields the caller's own
value, and shuffled values are limited to 8 bytes.

Fixed-shape floating-point `kernel.tensors.matmul`, `mma`, and `linear` calls use
the existing portable CPU loop lowering. Their contiguous inner loops remain
eligible for release-mode auto-vectorization. Very large operations split output
rows into disjoint mutable chunks across the available CPU cores; smaller ones
run inline to avoid thread-spawn and cache-contention overhead.
Backend-specific BLAS integration and quantized SIMD microkernels remain future
work.

Very large operations use the host's available parallelism. Set
`BORING_SIMD_THREADS=N` to cap them to `N` workers; `BORING_SIMD_THREADS=1`
provides a single-worker reference path. The default parallel threshold is
268,435,456 multiply-adds and can be changed with
`BORING_SIMD_PARALLEL_THRESHOLD=N` (`0` forces the multicore path). General
kernels remain single-threaded because their arbitrary indexed writes cannot be
parallelized safely without dependence analysis.

## Build

```sh
boring build --target simd examples/saxpy.br
cd examples/saxpy_simd
cargo run --release
```

Use release mode for vectorization. Debug builds are intended for diagnostics,
not performance measurements.

For a like-for-like native/baseline comparison, build
`examples/simd_saxpy_bench.br` once and run the generated release binary twice:

```sh
boring build --target simd examples/simd_saxpy_bench.br
cd examples/simd_saxpy_bench_simd
cargo run --release
BORING_SIMD_SCALAR=1 cargo run --release
```

Both commands execute the same binary and workload. The environment variable
changes only the runtime-selected native-feature versus baseline dispatch path.

`examples/simd_matmul_bench.br` compares the default adaptive fixed-tensor path
with `BORING_SIMD_PARALLEL_THRESHOLD=0`, which forces multicore execution using
the same generated binary.

## GPU display

`Screen`, `Dimension`, the `'surface` qualifier and `kernel: loop:` work as described
in [`gpu-display.md`](gpu-display.html), so `examples/plasma_metal.br` and
`examples/game_of_life.br` build and run unchanged with `--target simd`:

```sh
boring build --target simd examples/plasma_metal.br
cd examples/plasma_metal_simd
cargo build --release && ./target/release/plasma-metal-simd
```

There is no GPU presentation path, so the window is a **software blit**
([`softbuffer`](https://docs.rs/softbuffer) 0.3 on winit 0.28, the same pair the
CUDA/ROCm backends use). `winit`/`softbuffer` are added to the generated `Cargo.toml`
only when the program constructs a `Screen`; compute-only programs stay
dependency-free.

* `'surface` is an ordinary `Vec<u32>` (pixels `0xAARRGGBB`; softbuffer ignores the
  alpha byte).
* A kernel with a `let Dimension` field is constructed as `K(Dimension(w, h))`. The
  generated `new` takes the flat extent, sizes every array field to `w * h`, and
  seeds `'surface` buffers with the same sparse random 0/1 pattern the GPU Screen
  hosts use (`game_of_life` relies on it). Without an explicit `grid=`, a dispatch
  infers `ceil(w / block_x) x ceil(h / block_y)` from that field. `dim.width/height`
  are `i32` inside kernels, like `.thread`/`.block`.
* The user's `loop:` is an ordinary Rust `loop` (bare `k(block = ..)` statements
  inside it are dispatches; `a.x, a.y = a.y, a.x` swaps the buffers in place;
  `b.x = a.y` copies). So the whole program runs on a worker thread while `main`
  keeps the OS event loop (required on macOS). `screen.present(k.pixels)` hands the
  frame over by reference and blocks until it has been blitted.
* `screen.time/frame/width/height/dimension/closed/resized` are plain fields refreshed
  by each `present`; `screen.key("...")` uses the same key table as the other
  software-blit backends. Closing the window ends the process with exit code 0.
* The window is created with `PhysicalSize`, so its pixel count matches the buffer
  on any DPI scale. If the user resizes it, rows are clipped/padded rather than
  stretched; the kernel surface is not reallocated (same limitation as every
  backend).
* `Screen(Dimension(w, h))` without `title =` uses the title `boring`.

Only a top-level `Screen` is supported (a `Screen` constructed inside a function is
not recognised). Display programs cannot be exercised headlessly, so the test suite
checks generated code and `cargo check`s the crates but never opens a window.

## Not yet covered (relative to `--target metal`)

(`screen.key_pressed()` / `screen.mouse` are implemented here — see the Input section of [gpu-display.md](gpu-display.html).)

* `Device(...)` reports the CPU name and total system memory; usage and telemetry queries return 0.
