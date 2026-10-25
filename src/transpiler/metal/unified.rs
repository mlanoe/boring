// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Metal adapters for multi-target builds (`src/multi_unified.rs`): one function per routable
// kernel that runs a launch coming from the CPU host on the Metal device.

use crate::ast::*;
use crate::transpiler::unified::{route_fields, RouteField};

use super::host::{elem_rust_type, rust_type};

/// The adapter source appended to the Metal host, and the `(kernel, fields)` it covers.
pub(super) fn adapters(kernels: &[KernelDecl], plain_kernel_names: &[String]) -> (String, Vec<(String, Vec<RouteField>)>) {
    let mut out = String::new();
    out.push_str("\n// ---- multi-target adapters ----\n");
    out.push_str("pub fn __boring_unified_init() -> bool { metal::Device::system_default().is_some() }\n");
    let mut routes = Vec::new();
    for k in kernels {
        if !plain_kernel_names.contains(&k.name) {
            continue;
        }
        let Some(fields) = route_fields(k) else { continue };
        out.push_str(&adapter_fn(k, &fields));
        routes.push((k.name.clone(), fields));
    }
    (out, routes)
}

/// `value as target` unless the two spellings are the same type (`bool as bool` is not a cast).
fn cast(value: &str, from: &str, to: &str) -> String {
    if from == to { value.to_string() } else { format!("{value} as {to}") }
}

fn adapter_fn(k: &KernelDecl, fields: &[RouteField]) -> String {
    let name = &k.name;
    let mut s = format!(
        "pub fn __boring_unified_{name}(args: &mut [crate::unified::Arg<'_>], grid: [u32; 3], block: [u32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let device = boring_metal_device();
    let mut k = {name} {{
        __queue: __boring_metal_queue(&device),
        __pipeline: __boring_metal_pipeline(&device, \"{name}_kernel\")?,
        __device: device.clone(),
"
    );
    for (i, f) in fields.iter().enumerate() {
        let decl = k.fields.iter().find(|d| d.name == f.name).expect("route field of this kernel");
        if f.array {
            let target = elem_rust_type(&decl.ty);
            let convert = cast("e", f.rust_ty, &target);
            s.push_str(&format!(
                "        {}: {{ let v: Vec<{target}> = args[{i}].{}().iter().map(|&e| {convert}).collect(); device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * std::mem::size_of::<{target}>()) as u64, MTLResourceOptions::StorageModeShared) }},\n",
                f.name,
                f.accessor()
            ));
        } else {
            let target = rust_type(&decl.ty);
            s.push_str(&format!("        {}: {},\n", f.name, cast(&format!("args[{i}].{}()", f.accessor()), f.rust_ty, &target)));
        }
    }
    s.push_str("    };\n");
    s.push_str("    k.__boring_launch((block[0], block[1], block[2]), Some((grid[0], grid[1], grid[2])), &[])?;\n");
    s.push_str("    __boring_metal_flush()?;\n");
    for (i, f) in fields.iter().enumerate() {
        if f.array && f.mutable {
            let decl = k.fields.iter().find(|d| d.name == f.name).expect("route field of this kernel");
            let source = elem_rust_type(&decl.ty);
            let convert = cast("e", &source, f.rust_ty);
            s.push_str(&format!("    *args[{i}].{}() = k.read_{}()?.into_iter().map(|e| {convert}).collect();\n", f.accessor(), f.name));
        }
    }
    s.push_str("    Ok(())\n}\n");
    s
}
