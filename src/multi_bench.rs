// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Start-up benchmark glue appended to each backend module of a multi-target crate.
//
// Every backend gets `__boring_device_name()` (the device it would run on, `None` when unusable)
// and `__boring_bench()` (millions of multiply-adds per second, best-effort: `None` when the
// backend cannot be measured, which makes the policy fall back to its prior for that backend).
// The kernel is the same everywhere (1M elements, 256 dependent multiply-adds each), so the
// scores are comparable. It is deliberately compute-bound: a memory-bound kernel such as saxpy
// fits in a CPU's cache and would rank the CPU above a GPU on workloads where it is not.
// It measures sustained throughput, not launch latency or host transfers.

use crate::multi_target::Backend;

pub fn device_name_fn(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda => "pub fn __boring_device_name() -> Option<String> { cudarc::driver::CudaContext::new(0).ok()?.name().ok() }",
        Backend::Rocm => "pub fn __boring_device_name() -> Option<String> { HipContext::new(0).ok()?.name().ok() }",
        Backend::Metal => "pub fn __boring_device_name() -> Option<String> { Some(metal::Device::system_default()?.name().to_string()) }",
        Backend::Wgpu => {
            "pub fn __boring_device_name() -> Option<String> { wgpu::Instance::new(wgpu::InstanceDescriptor::default()).enumerate_adapters(wgpu::Backends::all()).first().map(|a| a.get_info().name) }"
        }
        Backend::Simd => "pub fn __boring_device_name() -> Option<String> { Some(\"cpu\".to_string()) }",
    }
}

/// `__boring_device_class()`: `"discrete"`, `"integrated"`, `"gpu"` (memory model unknown) or `"cpu"`.
pub fn device_class_fn(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda | Backend::Rocm => "pub fn __boring_device_class() -> &'static str { \"discrete\" }",
        Backend::Metal => {
            "pub fn __boring_device_class() -> &'static str { if metal::Device::system_default().is_some_and(|d| d.has_unified_memory()) { \"integrated\" } else { \"discrete\" } }"
        }
        Backend::Wgpu => {
            "pub fn __boring_device_class() -> &'static str { wgpu::Instance::new(wgpu::InstanceDescriptor::default()).enumerate_adapters(wgpu::Backends::all()).first().map_or(\"gpu\", |a| match a.get_info().device_type { wgpu::DeviceType::DiscreteGpu => \"discrete\", wgpu::DeviceType::IntegratedGpu => \"integrated\", _ => \"gpu\" }) }"
        }
        Backend::Simd => "pub fn __boring_device_class() -> &'static str { \"cpu\" }",
    }
}

pub fn bench_fn(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda => CUDA,
        // The HIP backend loads a code object compiled ahead of time by hipcc; there is no
        // runtime compiler to build a benchmark kernel with, so it keeps its prior.
        Backend::Rocm => "pub fn __boring_bench() -> Option<f64> { None }",
        Backend::Metal => METAL,
        Backend::Wgpu => WGPU,
        Backend::Simd => SIMD,
    }
}

const SIMD: &str = r##"pub fn __boring_bench() -> Option<f64> {
    let n: usize = 1 << 20;
    let mut y = vec![1.0f32; n];
    let threads = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1);
    let chunk = n.div_ceil(threads);
    let run = |y: &mut Vec<f32>| {
        std::thread::scope(|scope| {
            for part in y.chunks_mut(chunk) {
                scope.spawn(move || {
                    // Tiles that stay in L1, so the loop is bound by arithmetic, not memory.
                    for tile in part.chunks_mut(1024) {
                        for _ in 0..256 {
                            for v in tile.iter_mut() {
                                *v = *v * 1.000001 + 0.5;
                            }
                        }
                    }
                });
            }
        });
    };
    run(&mut y);
    let reps = 64;
    let start = std::time::Instant::now();
    for _ in 0..reps {
        run(&mut y);
    }
    let secs = start.elapsed().as_secs_f64();
    std::hint::black_box(&y);
    Some(n as f64 * 256.0 * reps as f64 / secs / 1e6)
}"##;

const METAL: &str = r##"pub fn __boring_bench() -> Option<f64> {
    let n: u64 = 1 << 20;
    let device = metal::Device::system_default()?;
    let source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void bench_fma(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint i [[thread_position_in_grid]]) { float v = x[i]; for (int k = 0; k < 256; k++) { v = v * 1.000001f + 0.5f; } y[i] = v; }";
    let library = device.new_library_with_source(source, &metal::CompileOptions::new()).ok()?;
    let function = library.get_function("bench_fma", None).ok()?;
    let pipeline = device.new_compute_pipeline_state_with_function(&function).ok()?;
    let x = device.new_buffer(n * 4, metal::MTLResourceOptions::StorageModeShared);
    let y = device.new_buffer(n * 4, metal::MTLResourceOptions::StorageModeShared);
    let queue = device.new_command_queue();
    let run = |reps: u32| {
        let commands = queue.new_command_buffer();
        for _ in 0..reps {
            let encoder = commands.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&x), 0);
            encoder.set_buffer(1, Some(&y), 0);
            encoder.dispatch_thread_groups(metal::MTLSize::new(n / 256, 1, 1), metal::MTLSize::new(256, 1, 1));
            encoder.end_encoding();
        }
        commands.commit();
        commands.wait_until_completed();
    };
    run(1);
    let reps = 64;
    let start = std::time::Instant::now();
    run(reps);
    let secs = start.elapsed().as_secs_f64();
    Some(n as f64 * 256.0 * reps as f64 / secs / 1e6)
}"##;

const WGPU: &str = r##"pub fn __boring_bench() -> Option<f64> {
    pollster::block_on(async {
        let n: u64 = 1 << 20;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await.ok()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read> x: array<f32>;\n@group(0) @binding(1) var<storage, read_write> y: array<f32>;\n@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) { let i = id.x; var v = x[i]; for (var k = 0; k < 256; k++) { v = v * 1.000001 + 0.5; } y[i] = v; }".into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let buffer = || device.create_buffer(&wgpu::BufferDescriptor { label: None, size: n * 4, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let (x, y) = (buffer(), buffer());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: y.as_entire_binding() },
            ],
        });
        let run = |reps: u32| {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            for _ in 0..reps {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((n / 256) as u32, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        };
        run(1);
        let reps = 64;
        let start = std::time::Instant::now();
        run(reps);
        let secs = start.elapsed().as_secs_f64();
        Some(n as f64 * 256.0 * reps as f64 / secs / 1e6)
    })
}"##;

const CUDA: &str = r##"pub fn __boring_bench() -> Option<f64> {
    let n: usize = 1 << 20;
    let ctx = CudaContext::new(0).ok()?;
    let ptx = cudarc::nvrtc::compile_ptx("extern \"C\" __global__ void bench_fma(const float* x, float* y, float a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float v = x[i]; for (int k = 0; k < 256; k++) { v = v * a + 0.5f; } y[i] = v; } }").ok()?;
    let module = ctx.load_module(ptx).ok()?;
    let func = module.load_function("bench_fma").ok()?;
    let stream = ctx.default_stream();
    let x = stream.alloc_zeros::<f32>(n).ok()?;
    let mut y = stream.alloc_zeros::<f32>(n).ok()?;
    let cfg = LaunchConfig { block_dim: (256, 1, 1), grid_dim: ((n / 256) as u32, 1, 1), shared_mem_bytes: 0 };
    let (a, count) = (1.000001f32, n as i32);
    let mut run = |reps: usize| -> Option<()> {
        for _ in 0..reps {
            let mut launcher = stream.launch_builder(&func);
            launcher.arg(&x).arg(&mut y).arg(&a).arg(&count);
            unsafe { launcher.launch(cfg).ok()?; }
        }
        stream.synchronize().ok()
    };
    run(1)?;
    let reps = 64;
    let start = std::time::Instant::now();
    run(reps)?;
    let secs = start.elapsed().as_secs_f64();
    Some(n as f64 * 256.0 * reps as f64 / secs / 1e6)
}"##;
