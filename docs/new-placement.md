# Kernel placement

`new` is not a Boring placement operator. Ordinary values use their ordinary
constructors; storage decisions are expressed by ownership qualifiers such as
`'owned`, `'shared`, and `'actor`.

GPU and SIMD placement belongs to a declared `kernel` instance. A kernel with
no explicit placement chooses the target's default device:

```boring
mut scale = Scale(n)
scale(block = 256, grid = 64)
```

Use `Device` when several kernels share a backend and physical device:

```boring
let device = Device(target = .WGPU, index = 0)
mut scale = kernel(device) Scale(n)
mut reduce = kernel(device) Reduce(n)
```

`Device` resolves its target/device once. `target` and the selector are both
optional; the selector may be an integer index or a backend-specific name or
address:

```boring
let automatic = Device()
let first = Device(target = .METAL)
let second = Device(target = .WGPU, index = 1)
let address = Device(target = .CUDA, name = "pci:0000:03:00.0")
```

For a one-off instance, put the selection directly on `kernel`:

```boring
mut scale = kernel(target = .WGPU, device = 0) Scale(n)
```

Inline kernels use the same form, followed by their launch dimensions:

```boring
kernel(device, block = 256, grid = 64):
    # device code
    out
```

Only declared kernel constructors accept `kernel(...)`; using it with an
ordinary `struct` is a compile-time error. This makes the allocation and
execution context explicit without giving ordinary construction a special
operator.
