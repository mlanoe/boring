# `gpu.tensor.*` — portable matrix operations on labeled arrays

> **Status: proposed design, not implemented.** The examples reuse Boring's
> existing arrays, qualifiers, kernel declarations, and dispatch syntax.
> `gpu.tensor.matmul` and `gpu.tensor.mma`, their collective semantics, and
> their backend implementations are new work. This document replaces the
> earlier proposal for public hardware-fragment types.

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

The initial API is device-side and block-collective. Unlike a host-side
PyTorch call, it executes inside a user-written kernel. A future host API
could arrange an entire multi-block multiplication automatically; that is a
separate execution contract.

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

The checker verifies fixed extents. Dynamic extents require validation before
dispatch: A.K = B.K, C.N = B.N, and C.M = A.M. The initial implementation
should start with fixed extents; runtime shapes must not be advertised until
validation, allocation, and codegen are implemented together.

## API

```boring
gpu.tensor.matmul(a, b, c)   # c = a × b
gpu.tensor.mma(a, b, c)      # c = a × b + c
```

Both functions operate on all elements of the supplied rank-two arrays.
`matmul` does not read the previous destination values. `mma` requires an
initialized destination. Neither operation allocates the public output.

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

The first version supports a single block for a whole-array invocation.
Enforce an explicit one-block launch for kernels using this restricted form;
do not infer a conventional per-element grid from the output shape. Multiple
blocks calling the operation on the same arrays would race. Scalable
multi-block execution requires disjoint tile views or a separate host-level
API, neither of which is silently supplied by this proposal.

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
    let [float32, k = 32, m = 32]'global a
    let [float32, n = 32, k = 32]'global b
    mut [float32, n = 32, m = 32]'unified c

    init([float32, k = 32, m = 32]'global input_a,
         [float32, n = 32, k = 32]'global input_b):
        a = input_a
        b = input_b
        # Fixed-size c is automatically zero-initialized.

    def ():
        gpu.tensor.matmul(a, b, c)

var [float32] host_a = [float32(i % 7) for i in 0..<32 * 32]
var [float32] host_b = [float32(i % 5) for i in 0..<32 * 32]

var multiplication = MatrixMul(
    host_a.reshape(k = 32, m = 32),
    host_b.reshape(n = 32, k = 32)
)

kernel:
    multiplication(block = (32, 1), grid = (1, 1))

let result = multiplication.c.flatten()
with result:
    print "C[0, 0] = {result[0]}"
    print "C[0, 1] = {result[1]}"
```

The one-dimensional block is an example launch configuration, not an API
restriction. A portable lowering linearizes x/y/z thread indices when the
block is multidimensional.

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

Quantized matmul deserves a separate API contract rather than hidden behavior
inside ordinary `matmul`. This document does not choose that API yet.

## Implementation sequence and acceptance criteria

### Phase 0 — finalize the operation contract

Specify rank-two axis ordering, accepted qualifiers, fixed-shape checking,
non-aliasing, collective participation, one-block dispatch validation, and
float32 numerical behavior. Reuse the existing parser and array types;
add builtin resolution/checking rather than new generic fragment syntax.

### Phase 1 — portable correctness baseline

Implement `matmul` and `mma` for fixed-size float32 global/unified arrays in
the interpreter and all four GPU backends. Use distributed scalar loops.
Verify rectangular matrices, tails, `mma` initialization, more/fewer outputs
than threads, multidimensional blocks, and permitted zero extents. Reject
incompatible shapes, immutable outputs, invalid types, and provable aliasing
or divergent participation. Reject multi-block launches for the initial API.

Check generated sources with real backend compilers where available.
Snapshots alone cannot establish shader validity or numerical correctness.
Run device correctness tests on available hardware and state untested targets.

### Phase 2 — performance and native paths

Add tiled shared-memory fallbacks, then a separately validated
float16→float32 profile and native implementations where supported.
Keep a way for tests/benchmarks to force the fallback. Compare both paths for
correctness and end-to-end time, including staging and synchronization.
Measure multiple shapes; selecting native instructions is not itself a win.

### Phase 3 — dynamic shapes and scalable execution

Add runtime shape validation and output allocation integration. Design either
explicit disjoint tile operands for multi-block kernels or a host-side matmul
that owns dispatch. Validate address spaces before admitting shared `'actor`
operands. Add equal-shaped batch axes before considering broadcasting or
arbitrary contractions.

### Phase 4 — quantized and specialized workloads

Design quantized operations from actual model requirements and benchmarks.
Consider raw PTX or other low-level paths only when a measured limitation
justifies them. Public fragment types remain unnecessary unless a concrete
use case cannot be expressed through the array operation contract.

## Implementation feasibility review (2026-09-29)

This is a source review of the current compiler, not a compiled tensor
prototype or a device validation. The operation is feasible without changing
surface grammar, but collective checking and memory visibility are substantial
prerequisites. The following findings refine the phases above.

| Area | Evidence in the current source | Consequence |
|---|---|---|
| Syntax | `src/parser/parse_expr.rs`, `parse_postfix_inner`, already builds `MethodCall` for dotted calls; labeled types already exist | No parser production is needed for `gpu.tensor.matmul(a,b,c)` |
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
- Require an explicit one-block launch. Validate constant launch dimensions
  statically and runtime launch values on the host; every dispatch route,
  including the interpreter, must enforce the same restriction.
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
unsupported types, and multi-block launches. Include distinct fields backed
by an overlapping allocation when the host representation permits it.

### Remaining architectural decision

The one-block API is a valid correctness milestone, not a scalable GEMM
solution. Before optimizing it, choose how whole-array operations scale:
explicit disjoint tile views inside kernels, or a separate host operation
that owns dispatch. Keep the block scope stable; silently turning the same
call into a grid-wide collective would change its synchronization semantics.

Recommendation: implement the restricted collective baseline first, including
its checker and coherent interpreter model. Decide the multi-block extension
before investing in native fragment codegen. No hardware-specific research
is needed to discover these compiler prerequisites.

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
