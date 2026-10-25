# `float16`

`float16` (alias `f16`, or `Float16` in type position) is the IEEE 754 binary16
scalar type. Host code uses `half::f16`; arrays therefore occupy exactly two
bytes per element. Conversions are explicit:

```boring
let float16 compact = value as float16
let float32 restored = compact as float32
mut [float16]'kernel'unified cache = [0.0 as float16 for ..<count]
```

Metal emits native MSL `half`, CUDA emits `__half`, and ROCm/HIP emits
`_Float16`. The wgpu backend currently rejects `float16` with a shader diagnostic:
its WGSL pipeline does not yet request the optional `shader-f16` feature, so it
must not silently widen storage to `f32`.

The interpreter stores scalar f16 values in its generic numeric value container,
but rounds them through IEEE binary16 on every explicit conversion. Transpiled
host arrays and all supported GPU buffers retain real 16-bit storage.
