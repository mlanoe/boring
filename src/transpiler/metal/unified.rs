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
    let decl_of = |f: &RouteField| k.fields.iter().find(|d| d.name == f.name).expect("route field of this kernel");
    // A buffer from the CPU copy, converted to the element type Metal uses.
    let buffer = |i: usize, f: &RouteField| {
        let target = elem_rust_type(&decl_of(f).ty);
        let convert = cast("e", f.rust_ty, &target);
        format!(
            "{{ let v: Vec<{target}> = args[{i}].{}().iter().map(|&e| {convert}).collect(); device.new_buffer_with_data(v.as_ptr() as *const _, (v.len() * std::mem::size_of::<{target}>()) as u64, MTLResourceOptions::StorageModeShared) }}",
            f.accessor()
        )
    };
    let scalar = |i: usize, f: &RouteField| {
        let target = rust_type(&decl_of(f).ty);
        cast(&format!("args[{i}].{}()", f.accessor()), f.rust_ty, &target)
    };
    let mut s = format!(
        "static __BORING_STORE_{name}: std::sync::Mutex<Option<std::collections::HashMap<u64, crate::unified::AssertSend<{name}>>>> = std::sync::Mutex::new(None);
pub fn __boring_unified_{name}(id: u64, args: &mut [crate::unified::Arg<'_>], upload: &[bool], grid: [u32; 3], block: [u32; 3]) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let device = boring_metal_device();
    let mut store = __BORING_STORE_{name}.lock().unwrap();
    let instances = store.get_or_insert_with(Default::default);
    if !instances.contains_key(&id) {{
        // First launch of this instance on Metal: every buffer comes from the CPU copy.
        let k = {name} {{
            __queue: __boring_metal_queue(&device),
            __pipeline: __boring_metal_pipeline(&device, \"{name}_kernel\")?,
            __device: device.clone(),
"
    );
    for (i, f) in fields.iter().enumerate() {
        if f.array {
            s.push_str(&format!("            {}: {},\n", f.name, buffer(i, f)));
        } else {
            s.push_str(&format!("            {}: {},\n", f.name, scalar(i, f)));
        }
    }
    s.push_str(&format!(
        "        }};
        instances.insert(id, crate::unified::AssertSend(k));
    }}
    let k = &mut instances.get_mut(&id).unwrap().0;
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array {
            s.push_str(&format!("    if upload[{i}] {{ k.{} = {}; }}\n", f.name, buffer(i, f)));
        } else {
            s.push_str(&format!("    k.{} = {};\n", f.name, scalar(i, f)));
        }
    }
    // The launch is committed, not awaited: reading a buffer back waits for it.
    s.push_str("    k.__boring_launch((block[0], block[1], block[2]), Some((grid[0], grid[1], grid[2])), &[])?;\n    Ok(())\n}\n");
    s.push_str(&format!(
        "pub fn __boring_unified_read_{name}(id: u64, index: usize, out: &mut crate::unified::Arg<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {{
    let store = __BORING_STORE_{name}.lock().unwrap();
    let Some(k) = store.as_ref().and_then(|instances| instances.get(&id)) else {{ return Err(\"no resident instance\".into()) }};
    match index {{
"
    ));
    for (i, f) in fields.iter().enumerate() {
        if f.array && f.mutable {
            let source = elem_rust_type(&decl_of(f).ty);
            let convert = cast("e", &source, f.rust_ty);
            s.push_str(&format!("        {i} => *out.{}() = k.0.read_{}()?.into_iter().map(|e| {convert}).collect(),\n", f.accessor(), f.name));
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
