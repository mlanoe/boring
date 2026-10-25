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
) -> (String, Vec<(String, Vec<RouteField>)>) {
    let mut out = String::new();
    out.push_str("\n// ---- multi-target adapters ----\n");
    out.push_str(&init_fn(needs_subgroups));
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

fn init_fn(needs_subgroups: bool) -> String {
    let features = if needs_subgroups {
        "        let use_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);\n        let required_features = if use_subgroups { wgpu::Features::SUBGROUP } else { wgpu::Features::empty() };\n"
    } else {
        "        let required_features = wgpu::Features::empty();\n"
    };
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
    let mut s = format!(
        "pub fn __boring_unified_{name}(args: &mut [crate::unified::Arg<'_>], grid: [u32; 3], block: [u32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    // The workgroup size is fixed in the shader at build time, from the kernel's launch site.
    if block != [{bx}, {by}, {bz}] {{
        return Err(Box::new(crate::unified::Unsupported(format!(\"block {{:?}}, this build is fixed to [{bx}, {by}, {bz}]\", block))));
    }}
    let mut k = {name}::new(__boring_gpu_device(), __boring_gpu_queue());
",
        name = k.name
    );
    for (i, f) in fields.iter().enumerate() {
        if f.array {
            s.push_str(&format!("    k.copy_{}_to_device(args[{i}].{}().as_slice());\n", f.name, f.accessor()));
        } else {
            s.push_str(&format!("    k.{} = args[{i}].{}();\n", f.name, f.accessor()));
        }
    }
    s.push_str("    k.dispatch(grid[0], grid[1], grid[2])?;\n");
    for (i, f) in fields.iter().enumerate() {
        if f.array && f.mutable {
            s.push_str(&format!("    *args[{i}].{}() = k.copy_{}_to_host();\n", f.accessor(), f.name));
        }
    }
    s.push_str("    Ok(())\n}\n");
    s
}
