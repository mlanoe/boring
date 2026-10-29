// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// What a kernel must look like to be routed between backends in a multi-target build
// (see `src/multi_unified.rs`): its fields are numeric/bool scalars or plain `[T]` buffers in
// `'unified`/`'global` memory. The CPU host and every backend adapter derive the argument list
// from the same function, so both sides agree on the order and types of a launch's arguments.

use crate::ast::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteField {
    pub name: String,
    /// The Rust type the CPU host and the adapters use for the scalar, or for a buffer's elements.
    pub rust_ty: &'static str,
    pub array: bool,
    /// The adapter must copy this buffer back (`mut`/`var` field).
    pub mutable: bool,
    /// A `Dimension` field (`let Dimension dim`), passed as `Arg::Dim([width, height])`.
    pub dim: bool,
}

impl RouteField {
    /// `Arg` variant holding this field, e.g. `F32` / `BufF32`.
    pub fn variant(&self) -> String {
        if self.dim {
            return "Dim".into();
        }
        if self.rust_ty == "half::f16" {
            return if self.array { "BufF16".into() } else { "F16".into() };
        }
        let mut chars = self.rust_ty.chars();
        let head = chars.next().map(|c| c.to_ascii_uppercase()).unwrap_or('X');
        let scalar = format!("{head}{}", chars.as_str());
        if self.array { format!("Buf{scalar}") } else { scalar }
    }

    /// Accessor on `Arg`: `as_f32` for a scalar, `buf_f32` for a buffer.
    pub fn accessor(&self) -> String {
        if self.dim {
            return "as_dim".into();
        }
        if self.rust_ty == "half::f16" {
            return if self.array { "buf_f16".into() } else { "as_f16".into() };
        }
        if self.array { format!("buf_{}", self.rust_ty) } else { format!("as_{}", self.rust_ty) }
    }
}

fn base(t: &Type) -> &Type {
    match t {
        Type::Mut(x) | Type::Qualified(x, _) => base(x),
        _ => t,
    }
}

/// The CPU host's Rust spelling of a numeric/bool scalar type (`int` is `i32`, `uint` is `u32`).
fn numeric_rust(t: &Type) -> Option<&'static str> {
    Some(match base(t) {
        Type::Int | Type::Int32 => "i32",
        Type::Uint | Type::Uint32 => "u32",
        Type::Int8 => "i8",
        Type::Int16 => "i16",
        Type::Int64 => "i64",
        Type::Uint8 => "u8",
        Type::Uint16 => "u16",
        Type::Uint64 => "u64",
        Type::Float32 => "f32",
        Type::Float16 => "half::f16",
        Type::Float64 => "f64",
        Type::Bool => "bool",
        Type::Named(n) => match n.as_str() {
            "float16" | "f16" => "half::f16",
            "float32" => "f32",
            "float" | "float64" => "f64",
            "int" | "int32" => "i32",
            "uint" | "uint32" => "u32",
            "int8" => "i8",
            "int16" => "i16",
            "int64" => "i64",
            "uint8" => "u8",
            "uint16" => "u16",
            "uint64" => "u64",
            "bool" => "bool",
            _ => return None,
        },
        _ => return None,
    })
}

/// The routable argument list of `k`, in field order, or `None` when the kernel must stay on
/// the CPU (screen surfaces, shared memory, atomics, struct/fixed-array fields, ...).
pub fn route_fields(k: &KernelDecl) -> Option<Vec<RouteField>> {
    let mut fields = Vec::new();
    for f in &k.fields {
        // Block-shared (`'actor`) and per-thread (`'local`) memory lives inside the launch:
        // it has no host data, so it is not an argument.
        if matches!(f.qual, GpuQual::Actor | GpuQual::Local) {
            continue;
        }
        match base(&f.ty) {
            Type::Array(element) => {
                // A screen surface is a buffer like any other here: the CPU host presents its own
                // copy, fetching the newest data from the backend first.
                // `'actor'global` / `'actor'unified` are buffers whose indexed accesses are atomic
                // on the device: the same buffer to the router.
                if !matches!(f.qual, GpuQual::Unified | GpuQual::Global | GpuQual::Surface | GpuQual::ActorGlobal | GpuQual::ActorUnified) {
                    return None;
                }
                fields.push(RouteField {
                    name: f.name.clone(),
                    rust_ty: numeric_rust(element)?,
                    array: true,
                    mutable: !matches!(f.binding, FieldBinding::Let),
                    dim: false,
                });
            }
            // A fixed array in constant memory is a read-only buffer to the router: each backend
            // fills it through its own `copy_<field>_to_device`, as its init code does.
            Type::ArrayN(element, _) if matches!(f.qual, GpuQual::Const) => {
                fields.push(RouteField { name: f.name.clone(), rust_ty: numeric_rust(element)?, array: true, mutable: false, dim: false });
            }
            Type::Named(n) if n == "Dimension" => {
                if !matches!(f.qual, GpuQual::Const) {
                    return None;
                }
                fields.push(RouteField { name: f.name.clone(), rust_ty: "u32", array: false, mutable: false, dim: true });
            }
            other => {
                if !matches!(f.qual, GpuQual::Const) {
                    return None;
                }
                fields.push(RouteField { name: f.name.clone(), rust_ty: numeric_rust(other)?, array: false, mutable: false, dim: false });
            }
        }
    }
    Some(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(fields: &str) -> KernelDecl {
        let src = format!("kernel K:\n{fields}    def ():\n        pass\n");
        let program = crate::parser::parse(crate::lexer::lex(&src).expect("lex")).expect("parse");
        match program.items.into_iter().next() {
            Some(Item::Kernel(k)) => k,
            _ => panic!("expected a kernel"),
        }
    }

    #[test]
    fn plain_scalars_and_buffers_are_routable() {
        let k = kernel("    let float32 alpha\n    let int n\n    let float16 gain\n    let [float32]'unified x\n    mut [uint8]'global y\n    mut [float16]'unified z\n");
        let fields = route_fields(&k).expect("routable");
        let names: Vec<_> = fields.iter().map(|f| (f.name.as_str(), f.rust_ty, f.array, f.mutable)).collect();
        assert_eq!(names, [("alpha", "f32", false, false), ("n", "i32", false, false), ("gain", "half::f16", false, false), ("x", "f32", true, false), ("y", "u8", true, true), ("z", "half::f16", true, true)]);
        assert_eq!(fields[3].variant(), "BufF32");
        assert_eq!(fields[1].accessor(), "as_i32");
        assert_eq!(fields[4].accessor(), "buf_u8");
        assert_eq!(fields[2].variant(), "F16");
        assert_eq!(fields[2].accessor(), "as_f16");
        assert_eq!(fields[5].variant(), "BufF16");
        assert_eq!(fields[5].accessor(), "buf_f16");
    }

    #[test]
    fn other_field_kinds_keep_the_kernel_on_the_cpu() {
        assert!(route_fields(&kernel("    let Dimension'global dim\n")).is_none());
        assert!(route_fields(&kernel("    let [float32]'unified x\n    let Point p\n")).is_none());
    }

    #[test]
    fn shared_memory_and_atomic_buffers_are_routable() {
        // Block-shared and per-thread memory has no host data: not an argument. An atomic
        // buffer is a buffer.
        let k = kernel("    let [float32]'unified input\n    mut [float32, 64]'actor tile\n    mut float32 acc\n    mut [int]'actor'unified counts\n    mut [uint]'actor'global totals\n");
        let fields = route_fields(&k).expect("routable");
        let shapes: Vec<_> = fields.iter().map(|f| (f.name.as_str(), f.rust_ty, f.array, f.mutable)).collect();
        assert_eq!(shapes, [("input", "f32", true, false), ("counts", "i32", true, true), ("totals", "u32", true, true)]);
    }

    #[test]
    fn constant_fixed_arrays_are_routable() {
        // A fixed array in constant memory is a read-only buffer; one in another memory is not.
        let k = kernel("    let [float32, 4]'const w\n    mut [float32]'unified out\n");
        let fields = route_fields(&k).expect("routable");
        let shapes: Vec<_> = fields.iter().map(|f| (f.name.as_str(), f.rust_ty, f.array, f.mutable)).collect();
        assert_eq!(shapes, [("w", "f32", true, false), ("out", "f32", true, true)]);
    }

    #[test]
    fn surfaces_and_dimensions_are_routable() {
        let k = kernel("    mut [uint]'surface pixels\n    let [uint]'surface cells\n    let Dimension dim\n");
        let fields = route_fields(&k).expect("routable");
        let shapes: Vec<_> = fields.iter().map(|f| (f.name.as_str(), f.array, f.mutable, f.dim)).collect();
        assert_eq!(shapes, [("pixels", true, true, false), ("cells", true, false, false), ("dim", false, false, true)]);
        assert_eq!(fields[2].variant(), "Dim");
        assert_eq!(fields[2].accessor(), "as_dim");
    }
}

/// The Rust side of `kernel.backends()`: one entry per backend the program may run kernels on.
/// Named with a `__Boring` prefix so it cannot clash with a program's own `Backend` type.
pub const BACKEND_STRUCT: &str = "#[allow(dead_code)] #[derive(Debug, Clone)] pub struct __BoringBackend { pub name: std::sync::Arc<str>, pub kind: std::sync::Arc<str>, pub class: std::sync::Arc<str>, pub score: f64, pub load: isize }\n";

/// `__boring_backends()` for a single-target build: that one backend.
/// `class` is a Rust expression of type `&str`.
pub fn single_backends_fn(name: &str, kind: &str, class: &str) -> String {
    format!("{BACKEND_STRUCT}#[allow(dead_code)] fn __boring_backends() -> Vec<__BoringBackend> {{ vec![__BoringBackend {{ name: \"{name}\".into(), kind: \"{kind}\".into(), class: ({class}).into(), score: 0.0, load: 0 }}] }}\n")
}

/// `__boring_unified_swap_<name>`: exchange two device buffers of one kernel (of one instance or
/// of two) without leaving the device. `place(obj, field)` is the Rust place holding a field's
/// device buffer on the kernel `obj`, `after(obj)` a statement to run on each kernel once its
/// buffers moved (wgpu rebuilds its bind group), empty when none. Only mutable buffers of one
/// element type (and `allow`ed by the backend) can be exchanged; any other pair is `Unsupported`.
pub fn swap_adapter(name: &str, fields: &[RouteField], allow: &dyn Fn(&RouteField) -> bool, place: &dyn Fn(&str, &str) -> String, extra: &dyn Fn(&RouteField, &str, &RouteField, &str) -> String, after: &dyn Fn(&str) -> String) -> String {
    let swappable: Vec<(usize, &RouteField)> = fields.iter().enumerate().filter(|(_, f)| f.array && f.mutable && allow(f)).collect();
    let mut same = String::new();
    let mut cross = String::new();
    for (i, fi) in &swappable {
        for (j, fj) in &swappable {
            if fi.rust_ty != fj.rust_ty {
                continue;
            }
            if i != j {
                same.push_str(&format!(
                    "            ({i}, {j}) => {{ std::mem::swap(&mut {}, &mut {}); {} {} Ok(()) }}\n",
                    place("k", &fi.name), place("k", &fj.name), extra(fi, "k", fj, "k"), after("k")
                ));
            }
            cross.push_str(&format!(
                "                ({i}, {j}) => {{ std::mem::swap(&mut {}, &mut {}); {} {} {} Ok(()) }}\n",
                place("x.0", &fi.name), place("y.0", &fj.name), extra(fi, "x.0", fj, "y.0"), after("x.0"), after("y.0")
            ));
        }
    }
    format!(
        "pub fn __boring_unified_swap_{name}(a: u64, ia: usize, b: u64, ib: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let unsupported = || -> Box<dyn std::error::Error + Send + Sync> {{ Box::new(crate::unified::Unsupported(\"these buffers cannot be exchanged\".into())) }};
    let mut store = __BORING_STORE_{name}.lock().unwrap();
    let Some(instances) = store.as_mut() else {{ return Err(\"no resident instance\".into()) }};
    if a == b {{
        if ia == ib {{ return Ok(()); }}
        let k = &mut instances.get_mut(&a).ok_or(\"no resident instance\")?.0;
        let r: Result<(), Box<dyn std::error::Error + Send + Sync>> = match (ia, ib) {{
{same}            _ => Err(unsupported()),
        }};
        r
    }} else {{
        let mut x = instances.remove(&a).ok_or(\"no resident instance\")?;
        let r: Result<(), Box<dyn std::error::Error + Send + Sync>> = match instances.get_mut(&b) {{
            Some(y) => match (ia, ib) {{
{cross}                _ => Err(unsupported()),
            }},
            None => Err(\"no resident instance\".into()),
        }};
        instances.insert(a, x);
        r
    }}
}}
"
    )
}

/// CUDA and HIP share one host shape (a context, a cached stream, `clone_htod`/`clone_dtoh`,
/// a consuming `__boring_launch`); they differ only in a few names.
#[derive(Clone, Copy)]
pub enum StreamFlavor {
    Cuda,
    Rocm,
}

/// The CUDA/HIP adapter source appended to the host, and the `(kernel, fields)` it covers.
/// `element_ty`/`scalar_ty` are the backend's own Boring-to-Rust type mapping.
pub fn stream_adapters(
    flavor: StreamFlavor,
    kernels: &[KernelDecl],
    plain_kernel_names: &[String],
    element_ty: fn(&Type) -> String,
    scalar_ty: fn(&Type) -> String,
) -> (String, Vec<(String, Vec<RouteField>)>) {
    let mut out = String::new();
    out.push_str("\n// ---- multi-target adapters ----\n");
    out.push_str("pub fn __boring_unified_init() -> bool { boring_gpu_init().is_ok() }\n");
    let mut routes = Vec::new();
    for k in kernels {
        if !plain_kernel_names.contains(&k.name) {
            continue;
        }
        let Some(fields) = route_fields(k) else { continue };
        // A kernel with no buffer has a different launch signature (no grid inference); it
        // stays on the CPU.
        if !fields.iter().any(|f| f.array) {
            continue;
        }
        out.push_str(&stream_adapter_fn(flavor, k, &fields, element_ty, scalar_ty));
        routes.push((k.name.clone(), fields));
    }
    (out, routes)
}

/// `value as target` unless the two spellings are the same type (`bool as bool` is not a cast).
fn cast(value: &str, from: &str, to: &str) -> String {
    if from == to { value.to_string() } else { format!("{value} as {to}") }
}

fn stream_adapter_fn(flavor: StreamFlavor, k: &KernelDecl, fields: &[RouteField], element_ty: fn(&Type) -> String, scalar_ty: fn(&Type) -> String) -> String {
    let name = &k.name;
    let decl_of = |f: &RouteField| k.fields.iter().find(|d| d.name == f.name).expect("route field of this kernel");
    // A `'const` fixed array is a host `Vec` that the launch copies into the module's constant
    // symbol: it has no device buffer to exchange, copy or read back.
    let is_const = |f: &RouteField| matches!(decl_of(f).qual, GpuQual::Const);
    // A buffer from the CPU copy, converted to the element type the backend uses.
    let buffer = |i: usize, f: &RouteField| {
        let decl = decl_of(f);
        let target = element_ty(&decl.ty);
        let convert = cast("e", f.rust_ty, &target);
        if is_const(f) {
            return format!("args[{i}].{}().iter().map(|&e| {convert}).collect::<Vec<{target}>>()", f.accessor());
        }
        let upload = match flavor {
            StreamFlavor::Cuda => format!("stream.clone_htod::<{target}, _>(&v)?"),
            StreamFlavor::Rocm => format!("stream.clone_htod::<{target}>(&v)?"),
        };
        let upload = format!("{{ let v: Vec<{target}> = args[{i}].{}().iter().map(|&e| {convert}).collect(); {upload} }}", f.accessor());
        if matches!(decl.binding, FieldBinding::Let) { format!("Arc::new({upload})") } else { upload }
    };
    let scalar = |i: usize, f: &RouteField| {
        if f.dim {
            return format!("Dimension(args[{i}].as_dim()[0], args[{i}].as_dim()[1])");
        }
        let target = scalar_ty(&decl_of(f).ty);
        cast(&format!("args[{i}].{}()", f.accessor()), f.rust_ty, &target)
    };
    let mut s = format!(
        "static __BORING_STORE_{name}: std::sync::Mutex<Option<std::collections::HashMap<u64, crate::unified::AssertSend<{name}>>>> = std::sync::Mutex::new(None);
pub fn __boring_unified_{name}(id: u64, args: &mut [crate::unified::Arg<'_>], upload: &[bool], grid: [u32; 3], block: [u32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let ctx = boring_gpu_ctx();
    let stream = boring_new_stream_with_priority(&ctx, 0)?;
    let mut store = __BORING_STORE_{name}.lock().unwrap();
    let instances = store.get_or_insert_with(Default::default);
    let mut k = match instances.remove(&id) {{
        Some(resident) => {{
            let mut k = resident.0;
"
    );
    for (i, f) in fields.iter().enumerate() {
        if f.array {
            s.push_str(&format!("            if upload[{i}] {{ k.{} = {}; }}\n", f.name, buffer(i, f)));
        } else {
            s.push_str(&format!("            k.{} = {};\n", f.name, scalar(i, f)));
        }
    }
    s.push_str(&format!(
        "            k
        }}
        // First launch of this instance on this backend: every buffer comes from the CPU copy.
        None => {name} {{
            __ctx: ctx.clone(),
            __stream: stream.clone(),
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array {
            s.push_str(&format!("            {}: {},\n", f.name, buffer(i, f)));
        } else {
            s.push_str(&format!("            {}: {},\n", f.name, scalar(i, f)));
        }
    }
    s.push_str(
        "        },
    };
    k = k.__boring_launch((block[0], block[1], block[2]), Some((grid[0], grid[1], grid[2])), &[], 0i32)?.inner;
    instances.insert(id, crate::unified::AssertSend(k));
    Ok(())
}
",
    );
    // Copy one buffer back to the CPU host.
    s.push_str(&format!(
        "pub fn __boring_unified_read_{name}(id: u64, index: usize, out: &mut crate::unified::Arg<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let store = __BORING_STORE_{name}.lock().unwrap();
    let Some(resident) = store.as_ref().and_then(|instances| instances.get(&id)) else {{ return Err(\"no resident instance\".into()) }};
    let k = &resident.0;
    match index {{
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array && f.mutable {
            let source = element_ty(&decl_of(f).ty);
            let convert = cast("e", &source, f.rust_ty);
            let (download, sync) = match flavor {
                StreamFlavor::Cuda => (
                    format!("k.__stream.clone_dtoh(&k.{}).map_err(__boring_cuda_classify_error)?", f.name),
                    "k.__stream.synchronize().map_err(__boring_cuda_classify_error)?;",
                ),
                StreamFlavor::Rocm => (format!("k.__stream.clone_dtoh(&k.{})?", f.name), "k.__stream.synchronize()?;"),
            };
            s.push_str(&format!("        {i} => {{ let v = {download}; {sync} *out.{}() = v.into_iter().map(|e| {convert}).collect(); }}\n", f.accessor()));
        }
    }
    s.push_str("        _ => return Err(\"not a buffer\".into()),\n    }\n    Ok(())\n}\n");
    s.push_str(&format!(
        "pub fn __boring_unified_release_{name}(id: u64) {{
    if let Some(instances) = __BORING_STORE_{name}.lock().unwrap().as_mut() {{ instances.remove(&id); }}
}}
"
    ));
    s.push_str(&swap_adapter(name, fields, &|f| !is_const(f), &|obj, field| format!("{obj}.{field}"), &|_, _, _, _| String::new(), &|_| String::new()));
    // Device-to-device copy and ranged read. CUDA: `CudaSlice::clone` and a device view; HIP:
    // `DeviceBuffer::try_clone` (hipMemcpyDtoD) and a ranged hipMemcpyDtoH.
    let (slice_ty, copy) = match flavor {
        StreamFlavor::Cuda => ("CudaSlice", "(%).clone()"),
        StreamFlavor::Rocm => ("DeviceBuffer", "{ k.__stream.synchronize()?; (%).try_clone()? }"),
    };
    s.push_str(&format!(
        "pub fn __boring_unified_export_{name}(id: u64, index: usize) -> Result<Box<dyn std::any::Any + Send>, Box<dyn std::error::Error + Send + Sync>> {{
    let store = __BORING_STORE_{name}.lock().unwrap();
    let Some(resident) = store.as_ref().and_then(|instances| instances.get(&id)) else {{ return Err(\"no resident instance\".into()) }};
    let k = &resident.0;
    match index {{
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array && !is_const(f) {
            let deref = if matches!(decl_of(f).binding, FieldBinding::Let) { format!("(*k.{})", f.name) } else { format!("(k.{})", f.name) };
            s.push_str(&format!("        {i} => Ok(Box::new({})),\n", copy.replace('%', &deref)));
        }
    }
    s.push_str("        _ => Err(\"not a buffer\".into()),\n    }\n}\n");
    s.push_str(&format!(
        "pub fn __boring_unified_import_{name}(id: u64, index: usize, handle: Box<dyn std::any::Any + Send>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let mut store = __BORING_STORE_{name}.lock().unwrap();
    let Some(resident) = store.as_mut().and_then(|instances| instances.get_mut(&id)) else {{ return Err(\"no resident instance\".into()) }};
    let k = &mut resident.0;
    match index {{
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array && !is_const(f) {
            let target = element_ty(&decl_of(f).ty);
            let value = if matches!(decl_of(f).binding, FieldBinding::Let) { "Arc::new(*buffer)" } else { "*buffer" };
            s.push_str(&format!(
                "        {i} => {{ let buffer = handle.downcast::<{slice_ty}<{target}>>().map_err(|_| crate::unified::Unsupported(\"another element type\".into()))?; k.{} = {value}; }}\n",
                f.name
            ));
        }
    }
    s.push_str("        _ => return Err(\"not a buffer\".into()),\n    }\n    Ok(())\n}\n");
    s.push_str(&format!(
        "pub fn __boring_unified_read_range_{name}(id: u64, index: usize, start: usize, len: usize, out: &mut crate::unified::Arg<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let store = __BORING_STORE_{name}.lock().unwrap();
    let Some(resident) = store.as_ref().and_then(|instances| instances.get(&id)) else {{ return Err(\"no resident instance\".into()) }};
    let k = &resident.0;
    match index {{
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array && f.mutable {
            let source = element_ty(&decl_of(f).ty);
            let convert = cast("e", &source, f.rust_ty);
            let read = match flavor {
                StreamFlavor::Cuda => format!(
                    "let end = (start + len).min(k.{field}.len());
            let begin = start.min(end);
            let v = k.__stream.clone_dtoh(&k.{field}.slice(begin..end)).map_err(__boring_cuda_classify_error)?;
            k.__stream.synchronize().map_err(__boring_cuda_classify_error)?;",
                    field = f.name
                ),
                StreamFlavor::Rocm => format!(
                    "let v = k.__stream.clone_dtoh_range(&k.{field}, start, start + len)?;
            k.__stream.synchronize()?;",
                    field = f.name
                ),
            };
            s.push_str(&format!(
                "        {i} => {{
            {read}
            *out.{acc}() = v.into_iter().map(|e| {convert}).collect();
        }}
",
                acc = f.accessor()
            ));
        }
    }
    s.push_str("        _ => return Err(\"not a buffer\".into()),\n    }\n    Ok(())\n}\n");
    s
}
