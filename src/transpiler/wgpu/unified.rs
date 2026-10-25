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
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let Some(adapter) = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await else {{ return false }};
{features}        let Ok((device, queue)) = adapter.request_device(&wgpu::DeviceDescriptor {{
            required_features,
            required_limits: adapter.limits(),
            ..Default::default()
        }}, None).await else {{ return false }};
        device.poll(wgpu::MaintainBase::Wait);
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
    let mut s = format!(
        "static __BORING_STORE_{name}: std::sync::Mutex<Option<std::collections::HashMap<u64, crate::unified::AssertSend<{name}>>>> = std::sync::Mutex::new(None);
pub fn __boring_unified_{name}(id: u64, args: &mut [crate::unified::Arg<'_>], upload: &[bool], grid: [u32; 3], block: [u32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    // The workgroup size is fixed in the shader at build time, from the kernel's launch site.
    if block != [{bx}, {by}, {bz}] {{
        return Err(Box::new(crate::unified::Unsupported(format!(\"block {{:?}}, this build is fixed to [{bx}, {by}, {bz}]\", block))));
    }}
    let mut store = __BORING_STORE_{name}.lock().unwrap();
    let k = &mut store
        .get_or_insert_with(Default::default)
        .entry(id)
        .or_insert_with(|| crate::unified::AssertSend({name}::new(__boring_gpu_device(), __boring_gpu_queue())))
        .0;
"
    );
    for (i, f) in fields.iter().enumerate() {
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
    s
}
