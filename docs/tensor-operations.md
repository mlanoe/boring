# Tensor operations

> **Status: implemented.** Host operations, device tile collectives, interpreter
> and CPU fallbacks, all GPU backends, dynamic `linear`, packed GGUF weights,
> and selected native schedules are available. Native matrix hardware is an
> optimization; every accepted operation has a portable implementation.

Boring exposes matrix operations on ordinary labeled arrays. It does not expose
hardware fragments, introduce a dynamic `Tensor` object, or require tensor-core
hardware. Shapes, storage qualifiers, allocation, and host access remain visible
in the source.

## Operation families

There are two related APIs:

- `kernel.tensors.*` performs a complete host-side operation and owns GPU
  dispatch when compiling for a GPU target;
- `.tensors.*Tile` is used inside a `kernel` entry point. Every thread in a
  block participates in one collective operation on an explicit output tile.

The camel-case spellings below are canonical. Compatibility aliases for the
older GPU naming style remain accepted.

| Whole operation | Device tile operation | Result |
|---|---|---|
| `kernel.tensors.matmul(a, b, c)` | `.tensors.matmulTile(...)` | overwrite `c` with `a × b` |
| `kernel.tensors.mma(a, b, c)` | `.tensors.mmaTile(...)` | accumulate `a × b` into `c` |
| `kernel.tensors.linear(x, w, y, ...)` | `.tensors.linearTile(...)` | overwrite `y` with `x × wᵀ` |

All forms use destination passing: the caller supplies a mutable output. Tensor
operations do not allocate a public result implicitly.

## Shapes and layout

The API uses Boring's existing labeled arrays. The first declared axis is the
fastest-varying axis; labels document a shape but do not change storage order.
For rank-two matrix multiplication:

```boring
let [float32, k = 32, m = 16]'kernel'global a
let [float32, n = 24, k = 32]'kernel'global b
mut [float32, n = 24, m = 16]'kernel'unified c
```

`a` is M×K, `b` is K×N, and `c` is M×N. Operand position and axis order define
the contraction; the names `m`, `n`, and `k` are not reserved. The checker
validates fixed extents and rejects incompatible shapes.

`linear` uses the row-major weight layout common to PyTorch and GGUF: `x` is
M×K, `weight` is N×K, and `y` is M×N. It computes `x × weightᵀ` without
materializing a transposed weight buffer.

Arbitrary strided views, implicit transpose, broadcasting, runtime rank, and
batched rank-three products are not part of the current contract.

## Host operations

### Fixed shapes

```boring
let [float32, k = 3, m = 2] a = [
    1.0 as float32, 2.0 as float32, 3.0 as float32,
    4.0 as float32, 5.0 as float32, 6.0 as float32,
]
let [float32, n = 2, k = 3] b = [
    1.0 as float32, 0.0 as float32,
    0.0 as float32, 1.0 as float32,
    1.0 as float32, 1.0 as float32,
]
mut [float32, n = 2, m = 2] c = [0.0 as float32 for ..<4]

kernel.tensors.matmul(a, b, c)
with c:
    print "{c[0]}"
```

For GPU builds the compiler synthesizes and dispatches a private kernel. For
`boring run` and the ordinary Rust/CPU path it executes a correctness-first CPU
implementation. Qualifiers may be written explicitly or inferred for operands
connected to a tensor call: read-only inputs default to `'kernel'global` and a
mutable result to `'kernel'unified`.

Host calls are accepted at top level, in functions and methods, and in supported
control-flow bodies. A GPU-target function containing a call declares `throws`
because construction or dispatch can fail. A result returned immediately stays
resident; `with` materializes it when the host needs element access.

Struct fields can be operands. A destination field must be `mut`, its method
must be `def`, and the receiver must permit mutation. Nested fields such as
`self.layer.weight` are not operands, and one call cannot combine fields from
two different receiver objects.

### Runtime dimensions

Flat buffers use explicit dimensions because their types do not carry extents:

```boring
req [float32]'kernel'unified project(
    [float32]'kernel'global x,
    [float32]'kernel'global weight,
    int rows,
    int outputs,
    int inputs,
) throws:
    mut [float32]'kernel'unified y = [0.0 as float32 for ..<rows * outputs]
    kernel.tensors.linear(
        x, weight, y,
        m = rows, n = outputs, k = inputs,
    )
    y
```

`m`, `n`, and `k` are all required, distinct named arguments. They must be
positive and the runtime validates input, weight, bias, and destination lengths
before executing. An optional one-dimensional bias is fused and broadcast over
the M rows:

```boring
kernel.tensors.linear(x, weight, bias, y, m = m, n = n, k = k)
```

## Packed and 16-bit weights

Dynamic `linear` accepts a `uint8` weight buffer plus a literal `format`. The
decoder is fused into the operation; no full float32 weight copy is allocated.

```boring
kernel.tensors.linear(
    x, packedWeight, bias, y,
    m = m, n = n, k = k,
    format = "q8_0",
)
```

Accepted formats are:

| Family | Formats |
|---|---|
| GGML blocks of 32 values | `q8_0`, `q5_0`, `q5_1`, `q4_0`, `q4_1`, `iq4_nl` |
| GGML K-quants of 256 values | `q6_k`, `q5_k`, `q4_k`, `q3_k`, `q2_k` |
| Unquantized 16-bit weights | `f16`, `bf16` |

The K dimension must be divisible by the format's block size. Inputs,
accumulation, optional bias, and outputs are float32. Each format's scale,
minimum, sign-bit, and codebook rules follow its GGML storage contract.

## Device tile collectives

Inside a kernel, tile operations reduce over the complete K dimension and work
on a selected output region:

```boring
kernel MatrixMul:
    let [float32, k = 35, m = 33]'global a
    let [float32, n = 67, k = 35]'global b
    mut [float32, n = 67, m = 33]'unified c

    def ():
        let row = .block.y * 16
        let col = .block.x * 16
        .tensors.matmulTile(
            a, b, c,
            row = row, col = col,
            rows = 16, cols = 16,
        )
```

`rows` and `cols` are positive compile-time integer literals. `row` and `col`
must be non-negative, block-uniform expressions. Edge tiles are clipped and do
not read or write out of bounds.

The collective contract is:

1. every thread in the block reaches the same call in uniform control flow;
2. all threads use the same operands, origins, and extents;
3. the block exclusively owns its destination tile during the call;
4. the destination does not overlap either input;
5. the operation provides the block synchronization and memory ordering needed
   around its reads and writes;
6. completion is not a grid-wide or host barrier.

Do not guard a call with `if .thread.x == 0`. Calling it on every lane means
participating in one collective, not launching one multiplication per lane.
Several blocks may operate on the same matrices only when their destination
tiles are disjoint. The explicit device API leaves this disjointness to the
developer; whole host operations construct the mapping themselves.

`'global` and `'unified` retain their ordinary memory and residency meanings.
Tensor operations introduce no new qualifier. Block-shared bare-`'actor` and
per-thread `'local` arrays are not accepted as general tensor operands.

## Numerical contract

The default profile uses float32 inputs, accumulation, and output. Native paths
must preserve that profile unless the project explicitly opts into reduced
precision. Backends may change reduction order or use fused multiply-add, so
cross-device comparisons use an absolute/relative tolerance rather than
requiring bitwise identity.

`matmul` and `linear` overwrite the destination. `mma` starts from the existing
destination value. Empty outputs perform no accesses; an empty reduction writes
zero for `matmul` and preserves the destination for `mma`.

## Backend behavior

| Target | Portable behavior | Native/optimized behavior |
|---|---|---|
| Interpreter | correctness reference over ordinary array values | none |
| Rust CPU | nested-loop fallback | none |
| CPU SIMD | portable CPU lowering, multithreaded where configured | SIMD dot products and specialized quantized schedules |
| CUDA | scalar/tiled generated kernels | cuBLAS SGEMM for eligible host float operations; opt-in TF32 WMMA and half-precision quantized prefill |
| Metal | scalar/tiled generated kernels | `simdgroup_matrix` for eligible float operations and selected native quantized schedules |
| ROCm | scalar/tiled HIP kernels | optional rocBLAS SGEMM; no MFMA/WMMA dependency |
| wgpu | scalar/tiled WGSL kernels | no cooperative-matrix dependency |

Every accepted call remains correct when a native path is unavailable. CUDA
toolchain coverage exists for generated tensor code, but broad numerical CUDA
hardware coverage is more limited than Metal, ROCm, wgpu, and CPU coverage.

## Schedule configuration

Schedules are selected when `boring build` generates the target project:

```toml
[tensor.matmul]
algorithm = "auto"

[tensor.linear]
algorithm = "auto"

[tensor.linear.decode]
q8_0 = "warp-broadcast"

[tensor.linear.prefill]
algorithm = "auto"
precision = "float"
```

The common linear schedule values are `auto`, `native`, `scalar`, `portable`,
`warp`, and `warp-broadcast`; Metal decode diagnostics additionally accept
`rows` and `split-k`. Per-format keys override the general decode or prefill
choice. `scalar` is the correctness-oriented accumulation reference and
`portable` keeps generated cross-backend schedules where Metal would otherwise
choose a hand-written native path.

For fixed matrices, `auto` is the default. CUDA additionally accepts `tf32`, an
explicit reduced-precision choice requiring sm_80+. ROCm accepts `rocblas`,
which links rocBLAS and uses SGEMM where eligible.

`[tensor.linear.prefill] precision = "half"` is an explicit speed/accuracy
tradeoff for native Metal prefill and CUDA WMMA prefill. It rounds staged
activations and dequantized weights to float16 while keeping float32
accumulation. It is never selected implicitly; `precision = "float"` is the
default.

`precision = "mixed"` (Metal only; other targets treat it as `float`) keeps the
activations in float32 and rounds only the dequantized weights to float16, which
lets the half-weight matrix multiply run on a 128x64 tile. On Qwen2.5-3B Q4_K_M it
moved no logit statistic measurably (same top-1 and mean logit difference as
`float` against llama.cpp) and gave the same generated text, at about +8% end-to-end
prefill (`half`: +13%).

## Current limitations

- No public hardware-fragment types or direct `mma.sync`-style API.
- No implicit transpose, arbitrary contraction, broadcasting, or strided view.
- No runtime-rank or rank-three batched attention operation yet.
- Explicit device tile calls require conservative, statically verifiable
  collective control flow and block-uniform origins.
- Nested struct fields and fields from multiple receiver objects cannot be
  mixed directly in one host tensor call.
- Mutable resident destinations retain value semantics and can require a
  device-to-device copy between operations.
- Backend-native schedules cover selected shapes, types, and formats; the
  portable fallback remains the contract for everything else.

## Related documentation

- [Labeled multi-dimensional arrays](array-multidim-types.html)
- [GPU computing language reference](gpu-module.html)
- [CUDA backend](cuda-module.html)
- [ROCm backend](rocm-backend.html)
- [Metal backend](metal-backend.html)
- [wgpu backend](wgpu-backend.html)
- [CPU SIMD backend](simd-backend.html)
