# `gpu.tensor.*` — portable matrix operations on labeled arrays

> **Status: device tile fallback, host CPU/interpreter fallback, automatic
> fixed-shape host GPU dispatch, and native fixed-shape acceleration on Metal
> and CUDA are implemented.**
> CamelCase GPU names and compatibility
> aliases are implemented. Validated `gpu.tensor.matmulTile` and `mmaTile`
> calls run in the interpreter and lower to scalar code on CUDA, Metal, ROCm,
> and wgpu. Top-level host `gpu.tensor.matmul`/`mma` calls synthesize private
> multi-block kernels on all four GPU targets.

## Accepted API direction (2026-09-30)

The decisions in this section supersede the earlier single-block milestone
and the open multi-block question in the feasibility review below.

- Keep `gpu.` explicit in host and kernel code. No leading-dot shorthand or
  implicit `tensor`/`warp`/`block` aliases are proposed.
- Use camelCase: `blockDim`, `gridDim`, `shuffleDown`, `shuffleUp`,
  `shuffleXor`, `matmulTile`, and `mmaTile`. Existing snake_case GPU names
  remain compatibility aliases during migration.
- Host `gpu.tensor.matmul(a, b, c)` and `gpu.tensor.mma(a, b, c)` own
  dispatch. Inputs and the preallocated mutable output use existing
  `'gpu'global` or `'gpu'unified` qualifiers. No new qualifier is introduced.
- Device `gpu.tensor.matmulTile` and `gpu.tensor.mmaTile` are block
  collectives. Explicit `row`, `col`, `rows`, and `cols` select the output
  region; initially tile extents are compile-time constants. Each block
  reduces over all K and owns a disjoint output tile. Handle output edges
  without out-of-bounds accesses. No split-K or grid-wide barrier is implied.
- Host dispatch partitions M/N into output tiles. Dependent whole-matrix
  operations use ordered kernel launches, not block barriers. The developer
  owns disjointness for explicit device calls; the host API guarantees it.
Transpose flags and optimized host scheduling still need detailed contracts.
Runtime shape validation and the optional row-broadcast bias are implemented
for dynamic `linear`. The existing compiler review remains relevant to each
tile's checking and memory visibility.

## Objective

Expose matrix multiplication as an operation on ordinary labeled arrays.
The developer describes the data, its memory placement, and the calculation;
Boring implements it with native matrix instructions where appropriate and
GPU loops otherwise. Tensor-core availability is an optimization choice,
not a prerequisite for a correct program.

This follows the useful part of PyTorch's approach: mathematical operations
on tensors rather than public hardware fragments. It does not propose a
PyTorch implementation, autograd, a graph compiler, or a dynamically ranked
universal Tensor class. Boring already has typed, labeled multidimensional
arrays, including axes whose extents are known only at runtime.

The device API is block-collective and executes inside a user-written kernel.
The host API arranges the same tiled multi-block multiplication automatically;
it has a separate execution contract because it owns dispatch.

## Design decisions

- Reuse `[T, axis1, axis2, ...]` with fixed or dynamic extents.
- Introduce no keywords, qualifiers, or public fragment types.
- Preserve the existing meaning of `'global`, `'unified`, and `'actor`.
- Keep the real logical dimensions of each array: A is M×K, B is K×N,
  and C is M×N. Do not add artificial axes to encode instruction metadata.
- Use destination-passing functions: allocation and residency remain visible.
- Provide a correct fallback for every accepted type/shape combination on
  every supported target. Unsupported element types remain explicit errors.
- Keep precision policy explicit. Hardware acceleration must not silently
  quantize inputs or reduce their declared precision.
- Keep native tile shapes, fragment layouts, and architecture dispatch private
  to the backend.

## Existing array syntax and layout

See [labeled multidimensional arrays](array-multidim-types.html) for the
implemented syntax. The first declared axis is the fastest-varying index;
labels do not change that storage rule.

```boring
# Fixed-size, contiguous matrices: first axis is the column axis.
let [float32, k = 32, m = 32]'global a
let [float32, n = 32, k = 32]'global b
mut [float32, n = 32, m = 32]'unified c

# Existing syntax for runtime extents; rank and labels remain static.
let [float32, k, m]'global dynamic_a
let [float32, n, k]'global dynamic_b
mut [float32, n, m]'unified dynamic_c
```

For the initial rank-two API, operand position determines the role and
axis order determines the matrix orientation. A's first axis is K, B's
second axis is K; the output's first and second axes are N and M. The names
`m`, `n`, and `k` improve readability but are not reserved or magic labels.
For example, existing `width`/`height` arrays can also be arguments.
Matching label names alone neither proves equal extents nor requests an
implicit transpose. Arbitrary axis contraction, strided/transposed views,
and broadcasting are later extensions requiring their own contracts.

The checker verifies fixed extents. The runtime `linear` overload takes explicit
`m`, `n`, and `k` arguments because flat GPU buffers do not carry axis extents in
their type. The interpreter, ordinary CPU fallback, and synthesized GPU host
code reject non-positive dimensions and validate all three buffer lengths before
dispatch.

## API

```boring
gpu.tensor.matmulTile(a, b, c, row = row, col = col, rows = 16, cols = 16)
gpu.tensor.mmaTile(a, b, c, row = row, col = col, rows = 16, cols = 16)
gpu.tensor.linearTile(x, weight, y, row = row, col = col, rows = 16, cols = 16)
```

Both functions reduce over the complete K dimension and operate on the
selected `rows` by `cols` output tile, clipped at the M/N boundaries.
`matmulTile` does not read previous destination values. `mmaTile` requires an
initialized destination and adds the product to it. Neither operation
allocates the public output. `rows` and `cols` are positive compile-time
integer literals in the initial implementation; `row` and `col` are uniform,
non-negative integer expressions shared by every thread in the block.

`linearTile` has the overwrite behavior of `matmulTile`, but accepts the right
operand in the row-major weight layout used by PyTorch linear layers and GGUF:
A is `[K,M]`, weight is `[K,N]`, and C is `[N,M]`. It computes
`A * weight transpose` without materializing a transposed weight buffer. The
whole-operation spelling is `gpu.tensor.linear(x, weight, y)`. The
runtime-dimension overload also accepts an optional one-dimensional bias as
`gpu.tensor.linear(x, weight, bias, y, m = m, n = n, k = k)`. The bias has
exactly `n` float32 elements, is broadcast across the `m` rows, and is fused
into the accumulation rather than dispatched as a separate operation.

For runtime dimensions, the implemented host spelling is:

```boring
req [float32]'gpu'unified linear(
    [float32]'gpu'global x,
    [float32]'gpu'global weight,
    int seq,
    int dOut,
    int dIn,
) throws:
    mut [float32]'gpu'unified y = [0.0 as float32 for ..<seq * dOut]
    gpu.tensor.linear(x, weight, y, m = seq, n = dOut, k = dIn)
    y
```

With a bias, the call becomes:

```boring
gpu.tensor.linear(x, weight, bias, y, m = seq, n = dOut, k = dIn)
```

Q8_0-, Q5_0-, Q4_0-, IQ4_NL-, Q6_K-, Q4_K-, Q3_K-, and Q2_K-packed GGUF weights use the same operation with an explicit format:

```boring
gpu.tensor.linear(x, packedWeight, bias, y,
                  m = seq, n = dOut, k = dIn, format = "q8_0")
```

GPU projects may override the generated dynamic-linear schedule in
`boring.toml`. Configuration is applied by `boring build`; changing it requires
regenerating and recompiling the application, but never rebuilding the Boring
compiler itself:

```toml
[tensor.linear]
algorithm = "auto"

[tensor.linear.decode]
q8_0 = "warp-broadcast"

[tensor.linear.prefill]
algorithm = "auto"
```

Statically shaped `matmul`, `mma`, and `linear` calls have a separate matrix
schedule switch:

```toml
[tensor.matmul]
algorithm = "auto" # auto, native, or scalar
```

On Metal, `auto` (the default) and `native` lower float32 matrices whose `m`,
`n`, and `k` dimensions are multiples of eight to Metal's native 8x8
`simdgroup_matrix` operations. One 32-lane SIMD group computes each output
tile with `simdgroup_multiply_accumulate`. `scalar` retains the portable tile
loop. Shapes with an incomplete 8x8 tile also retain that fallback, so the
same source remains valid on Metal, CUDA, ROCm, and WGPU. Other backends
CUDA uses cuBLAS SGEMM in pedantic math mode for the same aligned fixed-shape
operations, preserving float32 inputs and accumulation without an implicit TF32
conversion. Unaligned shapes retain the portable generated CUDA kernel. ROCm
and WGPU currently treat this fixed-matrix setting as the portable schedule. The same
selection happens at run time for dynamically shaped, non-quantized
`gpu.tensor.linear`: compatible calls launch the 32-thread native kernel and
other shapes launch the 256-thread scalar kernel.

Every packed dynamic linear format has a tiled prefill schedule on WGPU, CUDA,
and ROCm. Single-row decode keeps its warp reduction. For `m > 1`, these
portable targets use a 16x16 output tile: the 256-thread block cooperatively
loads one activation tile and dequantizes one packed weight tile into shared
memory, then reuses both tiles for the inner products. Q4_K and Q6_K decode
their super-block scales/minima while staging the current K slice; Q5_0, Q4_0,
IQ4_NL, Q2_K, and Q3_K use the same fused scheme for their packed layouts.
Partial row, column, and K tiles are zero padded and output writes are bounds
checked. No float32 weight copy is allocated.

Metal uses its native schedule for Q8_0, Q4_K, and Q6_K prefill. A 256-thread threadgroup
computes a 32x16 output tile as eight 8x8 matrix fragments. Four activation
fragments and two packed-weight fragments are staged cooperatively; the two
weight fragments are dequantized once and reused across all four row fragments.
Q8_0 scales and the Q4_K/Q6_K super-block metadata are decoded while staging
the current K slice. Partial row, column, and K tiles are zero padded and bounds
checked.

On an Apple M3 with 10 GPU cores, the motivating 512x896x4864 benchmark took
about 15.3 ms/call (~290 GFLOP/s) with the earlier one-SIMD-group-per-8x8-tile
schedule. With the 32x16 schedule, three warmed 20-call runs took 8.11-10.60
ms/call (median 8.84 ms, ~505 GFLOP/s): a 1.73x median speedup. The other model
shapes measured 10.75-11.06 ms (~404-415 GFLOP/s) for 512x4864x896 and
1.47-1.68 ms (~488-559 GFLOP/s) for 512x896x896. These figures include the
generated call path and synchronize only after the final call. The ad-hoc
benchmark is `examples/tensor_q8_prefill_bench.br`; its `partial` shape also
checks a 35x896x19 result against a scalar CPU oracle.

`auto` is the default. The implementation accepts `auto`, `native`, `scalar`,
`warp`, and `warp-broadcast`; `auto` and `native` select the native Metal Q8_0
prefill schedule (and native Metal Q4_K/Q6_K schedules) or the portable tiled
packed-weight schedule on WGPU, CUDA, and ROCm.
`scalar` explicitly retains the one-thread-per-output-cell reference schedule.
On Metal, Q4_K single-row decode (`auto`) uses a dedicated multi-row schedule instead: each SIMD
group computes two output rows, four 256-value super-blocks at a time with eight lanes per
super-block, accumulating quantized-domain dot products and applying the per-group
scale/minimum once; `scalar` selects the one-thread-per-row reference.
The two warp schedules are available
for Q8_0, Q5_0, Q4_0, IQ4_NL, Q6_K, Q4_K, Q3_K, and Q2_K single-row decode.
`auto` selects warp reduction for all of these formats when `m == 1`.
`warp-broadcast` reads shared block metadata once and distributes it with
`gpu.warp.shuffle`; the native Metal Q4_K/Q6_K kernels broadcast their float16
super-block factors and packed scale/minimum metadata this way while lanes decode
their strided values. Q2_K/Q3_K use the portable warp-reduction lowering. Partial
float32 dot products are combined with `gpu.warp.shuffleXor`. Launch geometry
uses eight 32-lane warps per block on CUDA, Metal, and WGPU, and four 64-lane
warps on ROCm. Unsupported format/algorithm combinations fail during
`boring build` instead of silently falling back. More algorithms and quantized
formats can be added without changing source-level tensor calls.

The portable prefill benchmark is
`examples/tensor_portable_quant_prefill_bench.br`. It covers all eight packed
formats at `m = 128`, `k = 2048`, and `n = 11008`, warms the pipeline, queues
20 calls, and synchronizes on only the final readback.

The decode benchmark is `examples/tensor_k_quant_decode_bench.br`. It covers
Q4_K, Q6_K, Q3_K, and Q2_K with `m = 1`, `k = 2048`, and both `n = 11008`
(the motivating Qwen2.5-3B FFN shape) and `n = 151936` (the large output
projection shape). It warms the pipeline, queues 20 calls, and synchronizes on
only the final readback.

`packedWeight` is a `[uint8]'gpu'global` or `[uint8]'gpu'unified` array in
native GGUF block layout. Q8_0 uses a little-endian float16 scale followed by
32 signed int8 values in each 34-byte block. Q5_0 uses the scale, a 32-bit
high-bit field, and 16 low-nibble pairs in each 22-byte block. Q4_0 uses the
scale followed by 16 packed nibble pairs in each 18-byte block; each nibble is
offset by minus eight. `k` must be divisible by 32 and the buffer must contain exactly
`(n * k / 32) * blockBytes` bytes. Dequantization is fused into the generated
linear kernel; no float32 weight copy is allocated.

IQ4_NL also uses 18-byte blocks but treats each nibble as an index into GGML's
fixed 16-value non-linear codebook rather than as a linear four-bit integer.
Q6_K uses 210-byte superblocks for 256 values: low four-bit data, high two-bit
data, sixteen signed sub-block scales, and one float16 superblock scale. Its
`k` extent must therefore be divisible by 256.
Q4_K uses 144-byte superblocks for 256 values. It combines packed four-bit
values with eight packed six-bit scales, eight packed six-bit minima, and two
float16 superblock factors.
Q3_K uses 110-byte superblocks for 256 values. It combines packed two-bit
values, a separate high-bit mask, sixteen packed six-bit signed scales, and a
float16 superblock scale.
Q2_K uses 84-byte superblocks for 256 values, with sixteen packed scale/minimum
pairs, 64 bytes of two-bit values, and two float16 superblock factors.

This overload accepts flat float32 arrays in either `'gpu'global` or
`'gpu'unified` storage. Scalar schedules assign one output element to each GPU
thread; tiled Q8_0 prefill uses a two-dimensional output grid. Calls must be direct
function-body statements for GPU builds, and the containing function must
declare `throws`. Returning the destination immediately returns the generated
kernel's resident output and avoids a host readback.

The initial numeric profile is float32 inputs, float32 accumulation, and
float32 output. It must have a loop implementation on CUDA, Metal, ROCm, and
wgpu. Native instructions may be used only when their arithmetic satisfies
the selected profile. A native matrix unit's mere existence is insufficient.
In particular, converting float32 inputs to TF32 or float16 is not an
implicit optimization allowed by this profile.

A subsequent explicit float16-input/float32-accumulation profile is useful
for tensor cores, but requires supported half storage, casts, and arithmetic
throughout the relevant Boring backend. A loop fallback does not by itself
make an unsupported source element type legal.

Changing reduction order and using fused multiply-add can change rounding.
Numerical equivalence is tolerance-based, not bitwise equality across devices.
Tests must specify absolute/relative tolerances and cover cancellation,
large magnitudes, and non-finite values. A more permissive precision policy
may be added later with an explicit API; no policy spelling is settled here.

## Collective execution contract

An invocation is **one operation performed cooperatively by one block**
(workgroup on wgpu, threadgroup on Metal):

1. Every thread in the block reaches the same call in uniform control flow,
   with the same arrays, extents, and operation.
2. The block exclusively owns the destination region for the duration of the
   operation. Other blocks must not write it or race with reads of it.
3. The destination must not overlap either input. Inputs may overlap each
   other because both are read-only.
4. The operation includes block synchronization and the memory ordering needed
   for prior input writes and subsequent output reads by the same block.
   This includes appropriate storage/global-memory ordering when required,
   not just a shared-memory barrier.
5. Completion provides no cross-block barrier and no host synchronization
   beyond the existing kernel-dispatch contract.

Do not put the call inside `if gpu.thread.x == 0`. Calling it once per lane
is participation in one collective operation, not independent matrix
multiplications. Divergent participation is invalid; static diagnostics
should reject provable violations, with interpreter checks where practical.

The backend may distribute native tiles among several subgroups, stage data
in shared memory, or use scalar loops. Thread counts and resource limits
can prevent a native path; select the fallback in that case. The initial
contract must not require a fixed hardware subgroup size.

Multiple blocks may call the operation on the same arrays when their output
tiles are disjoint. The normal mapping is `row = gpu.block.y * rows` and
`col = gpu.block.x * cols`, with grid dimensions `ceil(N / cols)` by
`ceil(M / rows)`. The compiler checks that origins are block-uniform, but the
explicit device API leaves disjointness to the developer. Overlapping output
tiles are a data race. The host API computes the normal mapping and guarantees
disjoint ownership.

## Memory qualifiers remain the developer's choice

`'global` and `'unified` describe existing memory/access behavior, not the
matrix instruction family. Either can be used for readable inputs or a
mutable destination, subject to Boring's normal initialization, lifetime,
residency, and access rules:

```boring
# Alternative output declarations, not simultaneous declarations.
mut [float32, n = 32, m = 32]'global c
mut [float32, n = 32, m = 32]'unified c
```

The first suits a result consumed by subsequent GPU work; the second exposes
Boring's existing unified-access behavior when host reads are desired.
`'unified` does not promise physically shared memory or zero transfers on all
backends. Tensor operations introduce no additional host access permissions.

Block-shared `'actor` arrays are also a natural input/output extension for
fusion, but must have dedicated address-space lowering and synchronization
tests before being accepted. Per-thread `'local` arrays do not automatically
become shared collective operands.

Native instructions may require alignment or layouts the public arrays do
not provide. The backend must stage into suitable private scratch storage or
fall back. It must not impose undocumented native alignment requirements on
ordinary arrays. Any scratch allocation must respect device resource limits.

## Worked example using existing Boring syntax

This follows [the existing matrix multiplication example](../examples/matrix_mul_gpu.br).
Only the tensor call and its execution contract are proposed. The example
uses float32 to avoid making half-precision support a prerequisite.

```boring
kernel MatrixMul:
    let [float32, k = 35, m = 33]'global a
    let [float32, n = 67, k = 35]'global b
    mut [float32, n = 67, m = 33]'unified c

    init([float32, k = 35, m = 33]'global input_a,
         [float32, n = 67, k = 35]'global input_b):
        a = input_a
        b = input_b
        # Fixed-size c is automatically zero-initialized.

    def ():
        let row = gpu.block.y * 16
        let col = gpu.block.x * 16
        gpu.tensor.matmulTile(a, b, c, row = row, col = col,
                             rows = 16, cols = 16)

var [float32] host_a = [float32(i % 7) for i in 0..<35 * 33]
var [float32] host_b = [float32(i % 5) for i in 0..<67 * 35]

var multiplication = MatrixMul(
    host_a.reshape(k = 35, m = 33),
    host_b.reshape(n = 67, k = 35)
)

kernel:
    multiplication(block = (32, 1), grid = ((67 + 15) / 16, (33 + 15) / 16))

let result = multiplication.c.flatten()
with result:
    print "C[0, 0] = {result[0]}"
    print "C[0, 1] = {result[1]}"
```

The final blocks in both dimensions are partial tiles. The implementation
clips them without out-of-bounds accesses. The one-dimensional block is an
example launch configuration, not an API restriction. A portable lowering
linearizes x/y/z thread indices when the block is multidimensional.

## Reference fallback

A correctness-first implementation assigns each output element to exactly
one thread. The following is illustrative lowering pseudocode for the
one-dimensional example, not a new user-visible API:

```boring
# Collective entry synchronization with appropriate memory ordering.
sync
let lane = gpu.thread.x
let stride = gpu.block_dim.x
let count = c.n * c.m

for round in 0..<(count + stride - 1) / stride:
    let index = round * stride + lane
    if index < count:
        let row = index / c.n
        let col = index % c.n
        var float32 sum = 0.0
        for inner in 0..<a.k:
            sum += a[k = inner, m = row] * b[n = col, k = inner]
        c[n = col, m = row] = sum

# Collective completion synchronization with appropriate memory ordering.
sync
```

For `mma`, initialize `sum` from the corresponding C element instead of zero.
The actual backend must implement the memory semantics specified above;
the `sync` spelling in this sketch is not proof that current codegen already
provides every required address-space fence.

This fallback uses no fragments, handles non-tile-multiple extents, and
avoids duplicate writes. A faster fallback can stage tiles in block-shared
memory. Hardware paths must zero-pad input tails and mask output tails, or
use a scalar remainder. No cooperative instruction may access out of bounds.
For an empty reduction, `matmul` writes zero and `mma` preserves C; empty
outputs do no memory accesses while maintaining collective participation.

## Backend strategy

| Target | Baseline | Optional accelerated implementation |
|---|---|---|
| CUDA | Distributed float32 loops, then shared-memory tiling | WMMA or later PTX instructions for compatible architecture/type/precision profiles |
| Metal | Distributed float32 loops, then threadgroup-memory tiling | SIMD-group matrix operations where device, language version, and arithmetic profile permit |
| ROCm | Distributed float32 loops, then shared-memory tiling | rocWMMA on supported architectures and profiles; retain loops on RDNA2, including RX 6600 |
| wgpu | WGSL float32 loops and workgroup-memory tiling | Revisit cooperative matrices only after dependency, feature-query, and shader support are verified |
| Interpreter | Collective operation over ordinary array values | Correctness reference; no performance claim |

The 32×32 logical example may be decomposed into smaller native operations.
There is no requirement for all backends to share a native `(M,N,K)` tuple.

Availability detection stays internal. Depending on the backend, selection
may happen at transpilation, device compilation, or host pipeline creation.
Unsupported declarations and intrinsics must be absent from fallback shader
modules; an ordinary runtime `if` is not sufficient to hide illegal types.
CUDA target configuration, ROCm's later architecture selection, and Metal/
wgpu device-feature checks each need explicit integration and validation.

No public `gpu.tensor.available` guard is required for correctness. A later
diagnostic facility could report the selected implementation for profiling;
it should not become a prerequisite for writing portable code.

The interpreter must coordinate block participation, execute the collective
once per block, and make the result visible after completion. Independently
running a full matrix multiply for each simulated lane would hide races and
misrepresent the API.

## Higher ranks and dynamic extents

Existing arrays can express real batch dimensions without a new Tensor type:

```boring
let [float32, k, m, batch]'global a
let [float32, n, k, batch]'global b
mut [float32, n, m, batch]'unified c
```

A future batched operation could multiply corresponding batch slices.
The initial extension should require equal batch shapes, with no implicit
broadcasting. Further batch axes are possible, but their iteration, ownership,
and dispatch rules must be specified. Keep three concepts separate:

- rank: number of axes, known in the type;
- extents: sizes, fixed or runtime values;
- native tile geometry: private backend implementation detail.

Supporting arbitrary runtime rank is not necessary to support useful dynamic
shapes and is outside this proposal.

## Quantized weights and `boring-llm`

A portable matmul is useful for linear layers, but does not automatically
solve quantized inference or guarantee a speedup. Start by measuring an
independent kernel against existing `linear_gpu`/`q8_linear_gpu` workloads.
Prefill and token-by-token decode need separate measurements.

For Q8 weights, two distinct implementations are possible:

1. Dequantize weights into floating-point staging tiles and use floating-point
   multiplication, measuring conversion cost and numerical error.
2. Quantize activations too, then use int8×int8→int32 operations with explicit
   rounding, clipping, scaling, overflow bounds, and an accuracy budget.

For symmetric block quantization:

```text
y[m,n] ≈ Σ_g sx[m,g] * sw[n,g] * Σ_{k in group g} qx[m,k] * qw[n,k]
```

Apply each scale product to the corresponding integer partial sum before
combining groups in floating point. One scale after the whole K reduction
is incorrect when scales vary along K. Differing quantization boundaries
must be reconciled; affine quantization also needs zero-point corrections.
Q8 weights alone are not sufficient for an int8 multiplication path.

Quantized multiplication has a separate, explicit contract rather than hidden
behavior inside ordinary `matmul`: dynamic `linear` accepts packed `uint8`
weights and a literal `format` argument. Inputs and accumulation remain
float32, and each format defines its own block geometry and decoding rule.

## `boring-llm` migration gap review (updated 2026-10-01)

The tensor API now covers `boring-llm`'s linear weight formats and all of its
quantized production paths use `gpu.tensor.linear`. The earlier hand-written
kernels remain as correctness and performance references. That project exposes
the concrete requirements more precisely than a generic tensor example:

- Model dimensions and sequence lengths come from GGUF metadata and the KV
  cache at runtime. The six-argument `linear` overload now accepts runtime
  `seq`, `d_in`, and `d_out` values for flat float32 buffers and validates them
  before dispatch.
- Linear operations are called inside reusable `req` functions and loops. The
  GPU host rewrite accepts top-level statements, direct function-body
  statements, and calls nested in `if`, `if let`, `while`, `while let`, `for`,
  `loop`, `do while`, block-form `match`, and `try`/`catch`.
  A loop-carried mutable destination stays device-resident between iterations.
  Mutable value semantics currently require a device-to-device copy when that
  result becomes the next kernel's mutable destination.
- GGUF stores each weight output row contiguously as `(d_out, d_in)`. The
  implemented `linear` and `linearTile` operations read that layout directly
  and compute `x * W transpose` without a transposed weight copy.
- Q/K/V projections add a one-dimensional bias. The runtime float32 `linear`
  overload now accepts that bias and fuses its row-wise broadcast into the
  generated kernel, avoiding another dispatch and memory pass.
- Intermediate values are chained through many GPU operations. An immediately
  returned tensor result remains device-resident and is read back only at a
  host-access boundary. wgpu now shares the allocation when a resident value is
  consumed by a read-only kernel field; Metal does the same across function
  boundaries and for explicitly resident locals. Mutable fields retain
  independent-value semantics and still receive a device copy. CUDA and ROCm
  now use shared ownership for read-only resident buffers as well: cloning an
  argument retains an `Arc<CudaSlice<T>>` or `Arc<DeviceBuffer<T>>` and does
  not submit a device copy. Mutable destinations remain uniquely owned.
- The production path keeps weights packed as `uint8`. Tensor `linear` now
  performs fused Q8_0, Q5_0, Q4_0, IQ4_NL, Q6_K, Q4_K, Q3_K, and Q2_K
  dequantization. These cover every packed format currently implemented by
  `boring-llm`.
- Single-token decode (`seq == 1`) is a matrix-vector workload. Dynamic Q8_0
  linear selects a warp-reduction path for `m == 1`, optionally broadcasts
  packed-block scales, and keeps the scalar output-cell path for prefill.
  A destination declared with `[..<count]` is allocated directly on the device
  on every GPU backend; WGPU resizes and rebinds the buffer without staging a
  zero vector. In the current Metal microbenchmark, the tensor path improved
  from about 2.08 seconds to 0.59-0.63 seconds for 200 iterations after direct
  output allocation, 32-lane index specialization, and hoisting the uniform
  output bound outside the decode loop. The hand-written Q8_0 reference took
  about 0.64-0.69 seconds in the same alternating, warmed-up runs, putting the
  generated tensor schedule at parity with the specialized kernel. This
  Q5_0 now uses the same schedule and broadcasts both its scale and 32-bit
  high-bit word. Its latest warmed-up Metal measurements are roughly 0.61-0.69
  seconds versus 0.61-0.75 seconds for the hand-written reference, putting
  both implementations at practical parity. Q4_0 now shares the scale-only
  broadcast schedule with Q8_0; warmed-up Metal runs take roughly 0.56-0.75
  seconds versus 0.51-0.84 seconds for the hand-written reference, again at
  practical parity. IQ4_NL also supports the schedule and is correct on Metal
  and WGPU. Warmed-up Metal runs took about 0.69-0.83 seconds versus 0.56-0.71
  seconds for the specialized reference, but the scalar tensor schedule took
  1.64-1.65 seconds. `auto` therefore selects warp-broadcast for IQ4_NL while
  leaving its nonlinear codebook lowering as the remaining optimization gap.
  A compact packed-word codebook representation was also benchmarked; it was
  correct but did not improve the warmed-up Metal timings, so the clearer
  direct 16-value selection remains. A position-varying IQ4_NL oracle now
  covers all codebook entries across multiple rows, outputs, blocks, and
  positive and negative scales. It matches the interpreter on a real Metal
  device, and its generated kernels build for all four GPU targets.
  Q4_0 has the same position-varying, multi-row and multi-block coverage,
  including signed scales, and also matches the interpreter on real Metal.
  Q8_0 is covered by the same oracle shape with positive and negative int8
  values and scales, and likewise matches real Metal execution.
  Q6_K now has a position-varying oracle covering its low nibbles, packed high
  bits, signed per-group scales, both 128-value halves, multiple outputs, and
  positive and negative block scales. It also matches real Metal execution.
  Q4_K, Q3_K, and Q2_K now have equivalent position-varying oracles. Together
  they exercise Q4_K's split six-bit scales and minima, Q3_K's sign masks and
  split six-bit signed scales, and Q2_K's packed two-bit values with independent
  four-bit scales and minima. Each oracle matches real Metal execution and
  builds for CUDA, Metal, ROCm, and WGPU.
- Attention needs runtime rank-three batched products, a mapping from query
  heads to shared KV heads for GQA, a transposed K operand, and separate causal
  mask and softmax stages. Rank-two matmul can migrate linear layers first but
  cannot replace the attention kernels by itself.

The remaining compiler work after the linear migration is therefore:

1. Add an explicit safe in-place contract where profiling shows that the
   current loop-carried device copy is material.
2. Keep real-GGUF fixtures in the regression sweep. Q5_0, Q6_K, Q4_K, Q3_K,
   IQ4_NL, Q8_0, and Q4_0 tensor paths match both CPU dequantization and the
   specialized Metal kernels for every output of representative Qwen2.5
   tensors. This includes the 151,936-output Q8_0 vocabulary projection.
   Q2_K is covered by an extracted real superblock from Llama-2 7B, independently
   checked against Python's GGUF decoder, so the regular GPU test does not need
   to load the 2.6 GB model.
3. Add a runtime-sized rank-three batched product for attention, including
   explicit transpose and GQA head mapping semantics.
4. Extend fused epilogues only when another tensor operation needs them and
   profiling shows that a separate dispatch is material.

The unquantized and quantized `boring-llm` linear paths have completed this
baseline. The implemented quantized contract describes packed bytes, block
geometry, scales/minima/codebooks, and float32 accumulation. Every supported
format is covered by non-uniform reference vectors and real GGUF data.

Attention should follow only after runtime rank-two scheduling is stable. Its
first extension should be a statically ranked, dynamically sized batched
matmul with explicit batch/head mapping and transpose semantics. Masking and
softmax remain separate operations unless profiling demonstrates that a fused
attention contract is necessary.

## Implementation sequence and acceptance criteria

### Phase 0 — finalize the operation contract

Specify rank-two axis ordering, accepted qualifiers, fixed-shape checking,
non-aliasing, collective participation, disjoint tile ownership, and
float32 numerical behavior. Reuse the existing parser and array types;
add builtin resolution/checking rather than new generic fragment syntax.

### Phase 1 — portable correctness baseline

Implement `matmulTile` and `mmaTile` for fixed-size float32 global/unified
arrays in the interpreter and all four GPU backends. Use distributed scalar
loops. Verify rectangular matrices, tails, `mmaTile` initialization,
more/fewer outputs than threads, multidimensional blocks, and multi-block
tiling. Reject
incompatible shapes, immutable outputs, invalid types, and provable aliasing
or divergent participation. Fixed extents and tile sizes must be positive.

Check generated sources with real backend compilers where available.
Snapshots alone cannot establish shader validity or numerical correctness.
Run device correctness tests on available hardware and state untested targets.

### Phase 2 — performance and native paths

Add tiled shared-memory fallbacks, then a separately validated
float16→float32 profile and native implementations where supported.
Keep a way for tests/benchmarks to force the fallback. Compare both paths for
correctness and end-to-end time, including staging and synchronization.
Measure multiple shapes; selecting native instructions is not itself a win.

### Phase 3 — host dispatch and dynamic shapes

The fixed-shape host-side `matmul`/`mma` operation owns multi-block dispatch.
Next, add runtime shape validation and output allocation integration. Validate
address spaces before admitting shared `'actor` operands. Add equal-shaped
batch axes before considering broadcasting or arbitrary contractions.

### Phase 4 — quantized and specialized workloads

Design quantized operations from actual model requirements and benchmarks.
Consider raw PTX or other low-level paths only when a measured limitation
justifies them. Public fragment types remain unnecessary unless a concrete
use case cannot be expressed through the array operation contract.

## Implementation feasibility review (2026-09-29)

This review predates the implementation described at the end of this file.
It established that the operation was feasible without changing surface
grammar and identified the compiler work required by collective checking and
memory visibility. The progress section records which items are now complete.

| Area | Evidence in the current source | Consequence |
|---|---|---|
| Syntax | `src/parser/parse_expr.rs`, `parse_postfix_inner`, already builds `MethodCall` for dotted calls; labeled types already exist | No parser production is needed for `gpu.tensor.matmulTile(a, b, c, row = ..., col = ..., rows = ..., cols = ...)` |
| Kernel checking | `src/checker/rust_checks.rs`, `check_kernel_decl`, checks field shapes and axis count only; the generic method-call visitor in `src/checker/mod.rs` just visits receiver/arguments | Add a kernel-body validation pass with field types, mutation effects, device context, arity, shapes, and collective restrictions |
| Dynamic shapes | `src/main.rs` runs `desugar_labeled_array` before the checker; `desugar_kernel_decl` in `src/desugar_labeled_array.rs` replaces dynamic labeled fields with flat arrays and extent fields | Preserve shape metadata or resolve tensor operations before this information is erased; fixed-size fields avoid this issue initially |
| Qualifiers | CUDA/ROCm device emitters group Global/Unified into buffer arguments; Metal emits device pointers for both; wgpu also treats them as storage-backed fields | Existing device storage can be reused; no tensor qualifier is needed. Host residency behavior must remain unchanged |
| Backend calls | CUDA/ROCm/Metal/wgpu have explicit warp-intrinsic handling in their device emitters | A tensor call needs explicit lowering too; emitting an ordinary method call is insufficient |
| Synchronization | CUDA/ROCm emit `__syncthreads()` for `sync`; Metal emits a barrier with `mem_threadgroup`; wgpu emits `workgroupBarrier()` | The current generic `sync` lowering is not sufficient evidence for the promised global/storage-buffer visibility. Add tensor-specific ordering for every accessed address space |
| Interpreter storage | `src/interpreter/eval_gpu.rs`, `run_one_kernel_thread`, reconstructs ordinary fields from snapshots; output arrays are merged after execution | A barrier alone cannot make another thread's ordinary-field writes visible. Introduce coherent block storage for tensor operands, including surrounding reads/writes, or an equivalent coordinated execution model |
| Interpreter scheduling | The interpreter already uses OS threads/barriers for shared fields and warp calls | Tensor calls must select a collective-aware path even without shared fields. Existing barriers do not provide cancellation when a participant exits or fails |
| Higher ranks | `check_kernel_decl` rejects labeled kernel fields with more than three axes | Rank-two plus one batch axis fits the current check; additional batch axes require deliberately decoupling data rank from launch rank |

### Recommended first implementation boundaries

These are review recommendations, not already implemented guarantees:

- Accept only standalone tensor call statements directly in the kernel entry
  body, with direct kernel-field operands. Initially reject calls inside
  branches, loops, helper methods, or expressions, and reject early exits
  before a collective. This makes uniform participation checkable without
  claiming a general uniformity analysis exists.
- Accept contiguous, fixed-size, positive-extent rank-two float32 fields,
  global or unified, with a mutable output. Resolve shape by axis position,
  not by special label names. Shared fields, views, aliases, dynamic extents,
  and zero-size buffers follow after explicit support and tests.
- Require explicit tile origins and extents. Accept multi-block launches when
  the caller maps blocks to disjoint output tiles. The checker proves block
  uniformity of accepted origin expressions; output disjointness remains a
  caller obligation for the explicit device API.
- Reject a destination that names either input. Distinct field names do not
  prove distinct allocations: validate underlying buffer identity/ranges
  before dispatch where aliasing is possible. Do not advertise complete
  alias safety until constructor and buffer ownership paths have been audited.
- Represent resolved calls internally with operand identities, M/N/K, types,
  mutation effects, and collective scope. This is internal compiler metadata,
  not a public fragment type. Keep validation shared across backends.
- Lower loops at statement level, initially inline and specialized to the
  fields. This avoids depending on generic array-pointer helper support in
  every shader language. Maintain entry/exit ordering explicitly rather than
  relying on automatic shared-field barrier insertion.
- For interpreter failures, use abortable collective coordination or reject
  unsupported control flow before execution. A plain blocking barrier can
  hang if one participant returns an error. Test failed participation without
  allowing the test runner itself to hang.

### Required examples before implementation acceptance

Accepted examples should include rectangular `matmul`, initialized `mma`,
both output qualifiers, alternate axis labels, and a multidimensional block.
A particularly important visibility test is: threads prepare input elements,
call `matmul`, then read output elements written by other threads in the same
kernel. Also test two successive tensor calls where the second reads the
first's output. These reveal interpreter snapshot and barrier errors that a
single host read after dispatch would miss.

Rejected examples should cover wrong arity, incompatible extents, immutable
outputs, output/input aliasing, calls outside a kernel, conditional calls,
unsupported types, and lane-dependent tile origins. Include distinct fields
backed by an overlapping allocation when the host representation permits it.

### Multi-block decision

Whole-array operations scale over M and N. Each device call remains a
block-scoped collective and receives an explicit output origin and literal
tile extents. The caller maps blocks to disjoint tiles; no grid-wide barrier
or split-K reduction is implied. A host operation will own this mapping and
dispatch a private kernel. This keeps device synchronization local while
providing a PyTorch-like whole-operation entry point.

## References and research boundaries

- [Boring labeled arrays](array-multidim-types.html): syntax, axis order,
  reshape, indexing, and dispatch conventions.
- [Boring GPU module](gpu-module.html): kernel and memory semantics.
- [PyTorch `torch.mm`](https://docs.pytorch.org/docs/stable/generated/torch.mm.html)
  and [torch.matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html):
  inspiration for operation-oriented APIs, not specifications for Boring.
- [NVIDIA CUDA language extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html):
  native WMMA requirements that remain backend responsibilities.
- [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
- [rocWMMA](https://github.com/ROCm/rocWMMA).
- [WGSL specification](https://www.w3.org/TR/WGSL/).

Exact native shape/type tables, architecture gates, and dependency versions
must be verified when implementing each accelerated path. This revision does
not carry forward the old proposal's unverified generation-specific claims
or extrapolate vendor benchmark numbers into expected Boring speedups.

## Implementation progress (2026-09-30)

`src/checker/tensor.rs` now resolves direct field operands into a common tile
operation description: operation kind, operand identities, M/N/K, origin
expressions, and literal tile extents. Direct kernel call statements pass
through this check in both interpreter and GPU-build checker modes.

Current checks cover positional operands, named tile arguments, float32
rank-two positive literal shapes, global/unified storage, mutable output,
obvious same-field aliasing, dimensional compatibility, and size overflow.
Axis labels are descriptive; operand position determines the contraction.

The collective context check now scans nested expressions and control flow.
It rejects calls in constructors/helper methods, nested calls, operand or GPU
namespace shadowing, and calls following an unverified control-flow or
side-effecting prefix. The initial accepted prefix consists of comments and
immutable integer coordinate bindings. Origins may use non-negative literals,
previously validated bindings, block indices/dimensions, and addition or
multiplication. Lane values, memory reads, casts, and arbitrary calls are not
accepted as origins. This deliberately conservative analysis can be extended
without weakening collective participation requirements.

Runtime buffer aliasing and proof of disjoint output ownership remain caller
obligations for explicit device tile calls. Origin addition uses saturating
bounds in the interpreter and pre-addition range guards in generated code.
Top-level host calls are lowered before target emission. Calls nested in
functions or control flow are rejected until their scheduling semantics are
defined.

Unit tests exercise rectangular shapes, tail tiles, accumulation, and invalid
contracts. CLI tests verify diagnostics for `boring run` and all four GPU
build targets without relying on vendor toolchains.

## Scalar backend lowering (2026-09-30)

A common scalar fallback now exists in `src/transpiler/tensor.rs` and is
connected to the CUDA, ROCm, Metal and wgpu device emitters. It consumes the
resolved operation, preserves backend-specific buffer names, linearizes all
three thread dimensions, and assigns each output element to one lane.

The implementation supports rectangular matrices, partial edge tiles, tiles
wholly outside the output, overwrite (`matmulTile`) and accumulation
(`mmaTile`). Origin bounds are checked before adding offsets. Temporary
identifiers are scoped and avoid hiding rendered buffer/origin names.
Entry and exit synchronization uses CUDA/HIP block barriers, Metal device
and threadgroup memory flags, and WGSL storage/workgroup barriers.

Device tile calls are now active. Explicit kernels own their launch geometry
and must map blocks to disjoint output regions. The host API owns this mapping
for fixed-shape calls.

The generated CUDA/HIP-style scalar body is compiled as ordinary C++ in a
CPU harness and compared against matrix multiplication for a 3×5 times 5×7
case, multiple block shapes, multiple output tiles, overwrite followed by
accumulation, and guarded output sentinels. The harness replaces barriers
with no-ops and invokes lanes sequentially: it validates arithmetic and
indexing, **not GPU memory ordering**. Separate emitter tests cover all four
backends and WGSL buffer renaming.

Real frontend compilation was also verified on 2026-09-30 for a rectangular,
multi-block kernel containing consecutive `matmulTile` and `mmaTile` calls:

- Apple Metal Toolchain 27A266a compiled MSL to AIR and linked a metallib;
- Naga CLI 30.0.1 parsed and validated the generated WGSL;
- CUDA 12.6 `nvcc` and `ptxas` compiled the generated CUDA for `sm_70`,
  `sm_75`, `sm_80`, `sm_86`, `sm_89`, and `sm_90`;
- ROCm HIP 6.2 compiled the generated HIP code for `gfx1030`, `gfx1100`,
  `gfx1101`, `gfx90a`, and `gfx942`.

These checks establish source validity across the architectures currently
listed by the repository validation scripts. Host tensor matmul and linear
paths also produce matching numerical results on real Metal, WGPU, and ROCm
hardware. CUDA has real-toolchain compilation coverage; numerical CUDA hardware
coverage and broader cross-kernel memory-ordering stress tests remain open.

The interpreter executes each block collective once, on that block's linear
lane zero, after all lanes have evaluated the arguments. Its existing merge
combines disjoint block writes. This produces the same mathematical result for
validated calls and supports consecutive `matmulTile`/`mmaTile` calls in the
same block. It does not model instruction-level scheduling or barrier timing;
the conservative checker continues to reject divergent or nested calls.

## Host operation progress (2026-09-30)

The common checker now accepts `gpu.tensor.matmul(a, b, c)` and
`gpu.tensor.mma(a, b, c)` for three direct, explicitly typed host variables.
All operands must be fixed-size rank-two float32 labeled arrays with
`'gpu'global` or `'gpu'unified` residency, and the destination binding must be
mutable and distinct from both inputs. Shape ordering and compatibility are
identical to the device API.

`boring run` executes the whole operation and the ordinary Rust backend lowers
it to portable nested loops. Both paths support overwrite followed by
accumulation and validate backing-array lengths.

CUDA, Metal, ROCm, and wgpu builds rewrite each top-level host call into a
private fixed-shape kernel. The generated host code uploads or binds A, B, and
the initial C value, dispatches a 16-by-16 logical output tiling (smaller for
small matrices), then assigns the downloaded result back to C. `matmul`
overwrites C in the device operation; `mma` uses its uploaded value as the
accumulator. Consecutive calls are ordered correctly. Results returned
immediately from a GPU function stay resident; a later read-only tensor input
shares the same allocation on CUDA, Metal, ROCm, and wgpu. A host access
boundary still materializes the value, and mutable destinations keep
independent storage.

Automatic dispatch accepts direct top-level calls and direct statements in
function bodies, including `req` functions. Operands must be named variables
with types visible at the call site. GPU-target functions declare `throws`
because kernel construction and dispatch can fail. When the destination is
returned immediately, the lowering returns the synthesized kernel field
directly; wgpu consequently preserves it as a resident buffer without an
intermediate device-to-host copy. Runtime extents are implemented for the
six-argument `linear` overload and its seven-argument bias form. Calls nested in ordinary loops and branches are
also lowered, and loop-carried results remain resident without a host round
trip. Mutable value semantics may still require a device copy between kernels;
dynamic output allocation is implemented, while batching remains future work.
`linear`/`linearTile` support the transposed row-major weight
orientation without a copy. Native matrix instructions remain a backend
optimization after the portable behavior is validated on real hardware.

CUDA is checked with both the repository's stubbed build step and a real NVIDIA
toolchain. ROCm is compiled and executed through HIP on real AMD hardware. The
synthesized Metal shaders compile to AIR and metallib with Apple Metal
Toolchain 27A266a and execute on real Metal hardware. Naga 30.0.1 validates the
synthesized WGSL, which also executes through WGPU on real adapters. These runs
cover automatically dispatched matmul and linear kernels, including packed
quantized formats.
