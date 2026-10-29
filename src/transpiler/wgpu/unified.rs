// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// wgpu adapters for multi-target builds (`src/multi_unified.rs`): one function per routable
// kernel that runs a launch coming from the CPU host on the wgpu device, plus the lazy device
// initialisation the program's own `async_main` would otherwise do.

use crate::ast::*;
use crate::transpiler::unified::{route_fields, RouteField};

/// The adapter source appended to the wgpu host, and the `(kernel, fields)` it covers.
pub(super) fn adapters(
    kernels: &[KernelDecl],
    plain_kernel_names: &[String],
    block_sizes: &std::collections::HashMap<String, (u32, u32, u32)>,
    needs_subgroups: bool,
    needs_f16: bool,
) -> (String, Vec<(String, Vec<RouteField>)>) {
    let mut out = String::new();
    out.push_str("\n// ---- multi-target adapters ----\n");
    out.push_str(&init_fn(needs_subgroups, needs_f16));
    out.push_str(RANGE_HELPER);
    let mut routes = Vec::new();
    for k in kernels {
        if !plain_kernel_names.contains(&k.name) {
            continue; // a monomorphised generic kernel has no counterpart on the CPU host
        }
        let Some(fields) = route_fields(k) else { continue };
        let (bx, by, bz) = block_sizes.get(&k.name).or_else(|| block_sizes.get(&k.name.to_lowercase())).copied().unwrap_or((1, 1, 1));
        out.push_str(&adapter_fn(k, &fields, (bx, by, bz)));
        routes.push((k.name.clone(), fields));
    }
    (out, routes)
}

fn init_fn(needs_subgroups: bool, needs_f16: bool) -> String {
    let subgroup = if needs_subgroups { "        if adapter.features().contains(wgpu::Features::SUBGROUP) { required_features |= wgpu::Features::SUBGROUP; }\n" } else { "" };
    let f16 = if needs_f16 { "        if !adapter.features().contains(wgpu::Features::SHADER_F16) { return false; }\n        required_features |= wgpu::Features::SHADER_F16;\n" } else { "" };
    let features = format!("        let mut required_features = wgpu::Features::empty();\n{subgroup}{f16}");
    format!(
        "pub fn __boring_unified_init() -> bool {{
    if __BORING_GPU_DEVICE.get().is_some() {{ return true; }}
    pollster::block_on(async {{
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Ok(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await else {{ return false }};
{features}        let Ok((device, queue)) = adapter.request_device(&wgpu::DeviceDescriptor {{
            required_features,
            required_limits: adapter.limits(),
            ..Default::default()
        }}).await else {{ return false }};
        let _ = device.poll(wgpu::MaintainBase::Wait);
        device.on_uncaptured_error(Box::new(|e| eprintln!(\"boring: uncaptured GPU error: {{}}\", e)));
        let _ = __BORING_GPU_DEVICE.set(std::sync::Arc::new(device));
        let _ = __BORING_GPU_QUEUE.set(std::sync::Arc::new(queue));
        true
    }})
}}
"
    )
}

fn adapter_fn(k: &KernelDecl, fields: &[RouteField], block: (u32, u32, u32)) -> String {
    let (bx, by, bz) = block;
    let name = &k.name;
    // A kernel with a `Dimension` field is constructed with its size (grid-sized buffers).
    let construct = match fields.iter().position(|f| f.dim) {
        Some(i) => format!("{name}::new(args[{i}].as_dim()[0] as i32, args[{i}].as_dim()[1] as i32, __boring_gpu_device(), __boring_gpu_queue())"),
        None => format!("{name}::new(__boring_gpu_device(), __boring_gpu_queue())"),
    };
    let mut s = format!(
        "static __BORING_STORE_{name}: std::sync::Mutex<Option<std::collections::HashMap<u64, crate::unified::AssertSend<{name}>>>> = std::sync::Mutex::new(None);
pub fn __boring_unified_{name}(id: u64, args: &mut [crate::unified::Arg<'_>], upload: &[bool], grid: [u32; 3], block: [u32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    // The workgroup size is fixed in the shader at build time, from the kernel's launch site.
    if block != [{bx}, {by}, {bz}] {{
        return Err(Box::new(crate::unified::Unsupported(format!(\"block {{:?}}, this build is fixed to [{bx}, {by}, {bz}]\", block))));
    }}
    let mut store = __BORING_STORE_{name}.lock().unwrap();
    let instances = store.get_or_insert_with(Default::default);
    if !instances.contains_key(&id) {{
        // A kernel the device rejects (a shader it cannot validate) is not an error of the
        // program: another backend, or the CPU, runs it.
        let device = __boring_gpu_device();
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = {construct};
        if let Some(rejected) = pollster::block_on(device.pop_error_scope()) {{
            return Err(Box::new(crate::unified::Unsupported(rejected.to_string())));
        }}
        instances.insert(id, crate::unified::AssertSend(built));
    }}
    let k = &mut instances.get_mut(&id).unwrap().0;
"
    );
    for (i, f) in fields.iter().enumerate() {
        if f.dim {
            continue; // fixed at construction
        }
        if f.array {
            s.push_str(&format!("    if upload[{i}] {{ k.copy_{}_to_device(args[{i}].{}().as_slice()); }}\n", f.name, f.accessor()));
        } else {
            s.push_str(&format!("    k.{} = args[{i}].{}();\n", f.name, f.accessor()));
        }
    }
    s.push_str("    k.dispatch(grid[0], grid[1], grid[2])?;\n    Ok(())\n}\n");
    // Copy one buffer back to the CPU host.
    s.push_str(&format!(
        "pub fn __boring_unified_read_{name}(id: u64, index: usize, out: &mut crate::unified::Arg<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let store = __BORING_STORE_{name}.lock().unwrap();
    let Some(k) = store.as_ref().and_then(|instances| instances.get(&id)) else {{ return Err(\"no resident instance\".into()) }};
    match index {{
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array && f.mutable {
            s.push_str(&format!("        {i} => *out.{}() = k.0.copy_{}_to_host(),\n", f.accessor(), f.name));
        }
    }
    s.push_str("        _ => return Err(\"not a buffer\".into()),\n    }\n    Ok(())\n}\n");
    s.push_str(&format!(
        "pub fn __boring_unified_release_{name}(id: u64) {{
    if let Some(instances) = __BORING_STORE_{name}.lock().unwrap().as_mut() {{ instances.remove(&id); }}
}}
"
    ));
    // Packed byte kinds (8/16-bit elements) are stored as `array<u32>` with their element count
    // beside the buffer (`<field>_len`): that count travels with the buffer.
    // A `'const` fixed array lives in the params uniform, not in a buffer of its own.
    let in_uniform = |f: &RouteField| k.fields.iter().any(|d| d.name == f.name && matches!(d.qual, GpuQual::Const));
    let plain = |f: &RouteField| matches!(f.rust_ty, "f32" | "i32" | "u32") && !in_uniform(f);
    let packed = |f: &RouteField| matches!(f.rust_ty, "u8" | "i8" | "u16" | "i16") && !in_uniform(f);
    let exchangeable = |f: &RouteField| plain(f) || packed(f);
    s.push_str(&crate::transpiler::unified::swap_adapter(
        name,
        fields,
        &exchangeable,
        &|obj, field| format!("{obj}.{field}_buf"),
        &|fi, oi, fj, oj| if packed(fi) { format!("std::mem::swap(&mut {oi}.{}_len, &mut {oj}.{}_len);", fi.name, fj.name) } else { String::new() },
        &|obj| format!("{obj}.rebuild_bind_group();"),
    ));
    s.push_str(&format!(
        "pub fn __boring_unified_export_{name}(id: u64, index: usize) -> Result<Box<dyn std::any::Any + Send>, Box<dyn std::error::Error + Send + Sync>> {{
    let store = __BORING_STORE_{name}.lock().unwrap();
    let Some(k) = store.as_ref().and_then(|instances| instances.get(&id)) else {{ return Err(\"no resident instance\".into()) }};
    let (tag, source, len): (&'static str, &std::sync::Arc<wgpu::Buffer>, usize) = match index {{
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array && plain(f) {
            s.push_str(&format!("        {i} => (\"{}\", &k.0.{}_buf, 0),\n", f.rust_ty, f.name));
        } else if f.array && packed(f) {
            s.push_str(&format!("        {i} => (\"{}\", &k.0.{}_buf, k.0.{}_len),\n", f.rust_ty, f.name, f.name));
        }
    }
    s.push_str(
        "        _ => return Err(Box::new(crate::unified::Unsupported(\"not a 32-bit, 16-bit or 8-bit buffer\".into()))),
    };
    let copy = __boring_gpu_copy_d2d(&__boring_gpu_device(), &__boring_gpu_queue(), source);
    Ok(Box::new((tag, copy, len)))
}
",
    );
    s.push_str(&format!(
        "pub fn __boring_unified_import_{name}(id: u64, index: usize, handle: Box<dyn std::any::Any + Send>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let (tag, buffer, len) = *handle.downcast::<(&'static str, std::sync::Arc<wgpu::Buffer>, usize)>().map_err(|_| crate::unified::Unsupported(\"foreign buffer handle\".into()))?;
    let mut store = __BORING_STORE_{name}.lock().unwrap();
    let Some(k) = store.as_mut().and_then(|instances| instances.get_mut(&id)) else {{ return Err(\"no resident instance\".into()) }};
    match index {{
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array && plain(f) {
            s.push_str(&format!("        {i} if tag == \"{}\" => k.0.{}_buf = buffer,\n", f.rust_ty, f.name));
        } else if f.array && packed(f) {
            s.push_str(&format!("        {i} if tag == \"{}\" => {{ k.0.{}_buf = buffer; k.0.{}_len = len; }}\n", f.rust_ty, f.name, f.name));
        }
    }
    s.push_str("        _ => return Err(Box::new(crate::unified::Unsupported(\"another element type\".into()))),\n    }\n    k.0.rebuild_bind_group();\n    Ok(())\n}\n");
    s.push_str(&format!(
        "pub fn __boring_unified_read_range_{name}(id: u64, index: usize, start: usize, len: usize, out: &mut crate::unified::Arg<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let store = __BORING_STORE_{name}.lock().unwrap();
    let Some(k) = store.as_ref().and_then(|instances| instances.get(&id)) else {{ return Err(\"no resident instance\".into()) }};
    match index {{
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array && f.mutable && plain(f) {
            s.push_str(&format!("        {i} => *out.{}() = __boring_gpu_copy_d2h_range::<{}>(&__boring_gpu_device(), &__boring_gpu_queue(), &k.0.{}_buf, start, len),\n", f.accessor(), f.rust_ty, f.name));
        } else if f.array && f.mutable && packed(f) {
            s.push_str(&format!("        {i} => *out.{}() = __boring_gpu_copy_d2h_range_packed::<{}>(&__boring_gpu_device(), &__boring_gpu_queue(), &k.0.{}_buf, start, len, k.0.{}_len),\n", f.accessor(), f.rust_ty, f.name, f.name));
        }
    }
    s.push_str("        _ => return Err(Box::new(crate::unified::Unsupported(\"not a 32-bit, 16-bit or 8-bit buffer\".into()))),\n    }\n    Ok(())\n}\n");
    s
}

/// Reads `len` elements from element `start` of a buffer: a staging buffer of just that range.
const RANGE_HELPER: &str = "fn __boring_gpu_copy_d2h_range_packed<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, start: usize, len: usize, count: usize) -> Vec<T> {
    // 8- and 16-bit elements share 32-bit words: copy the words around the range, cut it out.
    let width = std::mem::size_of::<T>() as u64;
    let end = (start as u64 + len as u64).min(count as u64);
    let begin = (start as u64).min(end);
    if end == begin { return Vec::new(); }
    let first = begin * width / 4 * 4;
    let last = ((end * width + 3) / 4 * 4).min(src.size());
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: last - first,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src, first, &staging, 0, last - first);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    let _ = device.poll(wgpu::MaintainBase::Wait);
    rx.recv().unwrap().unwrap();
    let bytes = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    let from = (begin * width - first) as usize;
    let to = from + ((end - begin) * width) as usize;
    bytemuck::pod_collect_to_vec::<u8, T>(&bytes[from..to])
}
fn __boring_gpu_copy_d2h_range<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, start: usize, len: usize) -> Vec<T> {
    let width = std::mem::size_of::<T>() as u64;
    let total = src.size() / width;
    let end = (start as u64 + len as u64).min(total);
    let begin = (start as u64).min(end);
    let bytes = (end - begin) * width;
    if bytes == 0 { return Vec::new(); }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(src, begin * width, &staging, 0, bytes);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    let _ = device.poll(wgpu::MaintainBase::Wait);
    rx.recv().unwrap().unwrap();
    let data = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    data
}
";
