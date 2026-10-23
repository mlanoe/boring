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
`'actor'unified` atomics (`+= -= &= |= ^=`, `.min`, `.max`, `.swap`, `.cas`) on 32/64-bit integers are
real hardware atomics (`Atomic*::from_ptr`, relaxed ordering) that return the previous value, so such
kernels run their blocks in parallel like any other; other element types (floats, small integers) stay
plain read-modify-writes and keep the kernel sequential. `'surface` buffers are ordinary `Vec<u32>` fields (see "GPU display" below).
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
  (hand-written AArch64 and x86-64 SysV context switches on Unix, Win32 fibers on Windows; on any
  other target one parked OS thread per work-item with strict baton passing). `sync` parks a work-item until every live
  work-item of the block arrives; a work-item that `return`s early stops taking
  part, as on a GPU. A barrier reached by only part of its group aborts with a
  `deadlock` message instead of hanging. Fibers have 256 KiB stacks
  (`BORING_SIMD_STACK_KB`); an overflow is detected and aborts.

The thread backend can be forced for testing with
`RUSTFLAGS="--cfg boring_coop_threads"`. It is correct but much slower than fibers
(blocks of 256 work-items cost one thread each; only unsupported targets use it by default). The AArch64 fibers are exercised by
the test suite on Apple silicon; the x86-64 switch assembles and links
(cross-compiled for `x86_64-apple-darwin`) and the Windows fibers type-check
(`cargo check --target x86_64-pc-windows-gnu`), but neither has been run yet — run
`cargo test --test simd_codegen cooperative` on an x86-64 machine to validate it.

## Warp and tensors

The logical warp is **32 lanes wide on every ISA path**, matching Metal's SIMD-group
and CUDA's warp so kernels written against `.warp.size` or assuming 32 behave
identically. `.warp.size`, `.warp.lane`, `.warp.sync()` and the four shuffles
(`shuffle`, `shuffle_down`, `shuffle_up`, `shuffle_xor`) are supported with any
operand: a warp is 32 consecutive work-items by linear thread index, a shuffle whose
source lane is outside the block or has already returned yields the caller's own
value, and shuffled values are limited to 8 bytes.

Fixed-shape floating-point `kernel.tensors.matmul`, `mma`, and `linear` calls use
the existing portable CPU loop lowering. Dot-product forms (`linear`, whose weights are `[n, k]`) accumulate in 16 independent lanes and the
k-major forms update whole output rows with a contiguous multiply-add, so both vectorize
(partial sums are reassociated, so results can differ from a strict left-to-right sum in the last bit).
On x86-64 these loops have an AVX2+FMA twin selected at run time (baseline x86-64 has no FMA
instruction). When there are fewer rows than workers (down to the one-row decode shape), each row's columns are split
across workers instead. Very large operations split output
rows into disjoint mutable chunks across the available CPU cores; smaller ones
run inline to avoid thread-spawn and cache-contention overhead.
Quantized `linear` (Q4_0/Q5_0/Q8_0/IQ4_NL and the K-quants) pays a dequantization per weight, so it
switches to several workers at a much lower threshold than the float operations (1M
multiply-accumulates instead of 268M; `BORING_SIMD_QUANT_THRESHOLD` overrides it). It splits the
output into many more tasks than workers and the workers pull them from a shared queue, which keeps
heterogeneous performance/efficiency cores busy. On an Apple M3 the one-row decode of a
2048x11008 Q8_0 weight went from 13.5 ms to 4.2 ms per call and Q4_0 from 20.4 ms to 5.7 ms. Every cell is
computed in the same order whatever the worker count, so results are bit-identical to
`BORING_SIMD_THREADS=1` (checked by `tests/simd_quant_linear.rs`).
### Quantized `linear` schedules

Three schedules exist for the quantized `linear` (Q8_0, Q4_0, Q5_0, IQ4_NL, Q4_K, Q6_K, Q3_K, Q2_K);
the default is the middle one and the others are selected with environment variables.

| Schedule | Selected by | Numerics |
|---|---|---|
| reference | `BORING_SIMD_QUANT_REFERENCE=1` | original per-element decoder, horizontal sum after every block |
| float (default) | nothing | flat-loop block dequantizers (`__boring_dequant_<format>`) and sixteen float lanes kept across blocks, one horizontal sum per column |
| int8 | `BORING_SIMD_QUANT_INT8=1` | single-row decode only: activations rounded to int8 |

**Float (default).** The block dequantizers produce exactly the weights of the reference decoder
(same operations in the same order) as vectorizable loops, and keeping the sixteen partial sums
across blocks removes the sixteen dependent additions the reference pays per block. Sums are
reassociated, so results differ from the reference by about 1e-6 of the output scale (declared and
tested bound: 1e-5), exactly like the float `linear`; they never depend on the number of workers.

**Int8 (opt-in, `m == 1` only).** The activation row is rounded to int8 per 32-element block
(`scale = max|x| / 127`), every weight block is expressed as small integers times a float scale per
sub block (plus a float minimum for Q4_K and Q2_K), and each sub block is one integer dot product,
`vmull_s8` + `vpadalq_s16` on AArch64, and `maddubs` on x86-64: a 256-bit AVX2 form
(`__boring_idot_avx2`) when the CPU has AVX2 and the 128-bit SSSE3 form otherwise (`vdotq_s32` is
still unstable in Rust), with a scalar fallback. The weights are exact; the
only error is the activation rounding. **Declared tolerance: 6e-2 of the largest output magnitude**
(observed up to 4.9e-2 for Q4_K, where the minimum term amplifies the rounding, typically 2e-3 to
1e-2), checked in `tests/simd_quant_linear.rs`. A separate diagnostic value,
`BORING_SIMD_QUANT_INT8=activations`, rounds the activations the same way but computes in float:
the integer kernels must match it within 2e-5, which isolates their correctness from the rounding
(measured about 2e-7 on AArch64 and, through Rosetta with the real SSSE3 code, on x86-64). Any
`m > 1` call keeps the float schedule untouched.

**AVX2 caveat.** The 256-bit AVX2 kernel was written on a machine whose Rosetta has no AVX2, so it has
been type-checked for `x86_64` (the generated crate builds) but never executed. Two safeguards follow
from that. Before its first use it verifies itself against scalar arithmetic on 140 random and extreme
vectors (including -128 x -127), and on any disagreement it disables itself for the process, prints
`the AVX2 int8 kernel failed its self-test; using the SSSE3 kernel instead` once, and the SSSE3 kernel
(which was executed, through Rosetta) takes over. `BORING_SIMD_QUANT_NO_AVX2=1` forces the SSSE3 kernel.
On an AVX2 machine `cargo test --test simd_quant_linear` also runs
`avx2_int8_kernel_is_bit_identical_to_the_ssse3_kernel_when_available`, and every test in that file
fails if the self-test message appears. The self-test vectors themselves were run through the SSSE3
kernel under Rosetta with no mismatch, so a failure on real hardware would mean a real AVX2 bug, not a
bad vector.

**Measured** on an Apple M3, one-row decode of a 2048x11008 weight, best of three runs per mode
while another process kept the machine at a load average of 10-20 (use the ratios; `ms/call`):

| Format | threads | reference | float | int8 | float vs ref | int8 vs float |
|---|---|---|---|---|---|---|
| Q8_0 | 1 / 8 | 29.9 / 12.1 | 17.2 / 4.7 | 2.9 / 2.2 | 1.7x / 2.6x | 5.9x / 2.1x |
| Q4_0 | 1 / 8 | 51.6 / 11.2 | 12.9 / 4.1 | 6.5 / 1.5 | 4.0x / 2.8x | 2.0x / 2.7x |
| Q5_0 | 1 / 8 | 47.8 / 19.4 | 8.8 / 6.1 | 10.5 / 4.3 | 5.5x / 3.2x | **0.8x** / 1.4x |
| IQ4_NL | 1 / 8 | 72.7 / 22.5 | 8.8 / 3.5 | 7.0 / 2.9 | 8.3x / 6.5x | 1.2x / 1.2x |
| Q4_K | 1 / 8 | 62.5 / 18.6 | 7.3 / 3.2 | 2.9 / 1.6 | 8.6x / 5.9x | 2.5x / 2.0x |
| Q6_K | 1 / 8 | 47.3 / 15.5 | 9.5 / 3.9 | 5.8 / 2.6 | 5.0x / 4.0x | 1.6x / 1.5x |
| Q3_K | 1 / 8 | 55.6 / 17.1 | 9.6 / 3.5 | 3.6 / 1.6 | 5.8x / 4.8x | 2.7x / 2.2x |
| Q2_K | 1 / 8 | 52.6 / 16.8 | 7.0 / 2.5 | 3.4 / 1.6 | 7.5x / 6.8x | 2.1x / 1.5x |

The int8 gain is smaller than a bare integer dot product would give (about 6x in
`tools/simd_quant_microbench.rs`) because every block still pays its weight-extraction step, and Q5_0
single-worker is slower in int8 because rebuilding its fifth bits costs more than the integer dot
saves, so int8 is worth enabling per workload, not blindly.

Backend-specific BLAS integration also remains future work.

Very large operations use the host's available parallelism. Set
`BORING_SIMD_THREADS=N` to cap them to `N` workers; `BORING_SIMD_THREADS=1`
provides a single-worker reference path. The default parallel threshold is
268,435,456 multiply-adds and can be changed with
`BORING_SIMD_PARALLEL_THRESHOLD=N` (`0` forces the multicore path). Kernel grids are distributed over the same workers block by block, as on a GPU:
blocks may run in any order and concurrently, so blocks that write the same element
race exactly as they would on a GPU. A grid goes parallel once it has at least
`BORING_SIMD_PARALLEL_ITEMS` work-items (default 32768; `0` forces it, handy in
tests); smaller launches stay on the calling thread to avoid spawn cost. Kernels using atomics on element types without a hardware atomic (floats, 8/16-bit integers)
run sequentially. `BORING_SIMD_THREADS=1` restores fully
sequential execution.

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

## Notes relative to `--target metal`

(`screen.key_pressed()` / `screen.mouse` are implemented here — see the Input section of [gpu-display.md](gpu-display.html).)

* `Device` introspection: `name()` is the CPU ISA, `totalMem()`/`freeMem()` the system
  memory, `warpSize()` 32, `maxThreads()` 1024, `maxSharedMem()` 32768,
  `computeCapability()` `[lane count, 0]`, `Device.all()` one device; GPU usage,
  temperature and power read 0.
