# `float16`

`float16` (alias `f16`, or `Float16` in type position) is the IEEE 754 binary16
scalar type. Host code uses `half::f16`; arrays therefore occupy exactly two
bytes per element. Conversions are explicit:

```boring
let float16 compact = value as float16
let float32 restored = compact as float32
let float16 parsed = "1.5".parseFloat16() else float16(0.0)
mut [float16]'kernel'unified cache = [0.0 as float16 for ..<count]
```

Metal emits native MSL `half`, CUDA emits `__half`, and ROCm/HIP emits
`_Float16`. The wgpu backend emits WGSL `f16`, enables the optional language
extension, and requests `wgpu::Features::SHADER_F16` whenever a generated
shader uses it. Device creation reports a clear capability error if the selected
adapter lacks `shader-f16`; buffers are never silently widened to `f32`.

The interpreter stores scalar values as `half::f16` and preserves binary16
rounding at every arithmetic and conversion boundary. Transpiled host arrays
and all supported GPU buffers retain real 16-bit storage. `float16` exposes the
same conversions, comparisons, collection operations, math functions, tensor
operations, and quantized tensor paths as `float32` and `float64`; transcendental
operations use `float32` internally before rounding their result back to
binary16.
