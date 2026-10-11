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
shares the host-lowering boundary with GPU targets. Bare `'actor` storage and
`sync` require block-cooperative execution and are not yet supported. They, as
well as thread-local kernel fields and atomic/surface storage, are rejected by
the SIMD code generator rather than emitted with altered semantics. Unsupported
kernel statements and expressions likewise produce source diagnostics.
Integer bitwise operators (`&`, `|`, `^`, shifts, and `~`) preserve their scalar
Rust semantics inside kernels; identity and runtime type-test operators are
rejected because their host object semantics do not have a SIMD device model.

## Warp and tensors

The logical SIMD subgroup width is the detected lane count. `.warp.size` and
`.warp.lane` are available; the lane is the linear thread index within its block
modulo that width. Cross-lane `.warp` collectives are still rejected at
generation time rather than silently given different semantics. They require a
later two-phase subgroup lowering because CPU work-items currently execute as a
single contiguous loop.

Fixed-shape floating-point `kernel.tensors.matmul`, `mma`, and `linear` calls use
the existing portable CPU loop lowering. On the SIMD target, output rows are
split into disjoint mutable chunks across the available CPU cores, while each
worker's contiguous inner loops remain eligible for release-mode
auto-vectorization. Dynamic quantized pipelines still
depend on inter-function GPU-resident returns, which this backend rejects until
they have a CPU representation. Backend-specific BLAS integration and quantized
SIMD microkernels remain future work.

Set `BORING_SIMD_THREADS=N` to cap fixed tensor operations to `N` workers. The
default is the host's available parallelism; `BORING_SIMD_THREADS=1` provides a
single-worker reference path. General kernels remain single-threaded because
their arbitrary indexed writes cannot be parallelized safely without dependence
analysis.

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

`examples/simd_matmul_bench.br` similarly compares the default multicore fixed
tensor path with `BORING_SIMD_THREADS=1` using the same generated binary.
