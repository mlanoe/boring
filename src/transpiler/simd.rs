// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Boring -> portable CPU/SIMD backend.  The generated crate deliberately uses
// no GPU API.  Work-items retain the GPU coordinate model; the innermost x
// dimension is emitted as a tight contiguous loop so LLVM can vectorize it for
// the register set selected by the generated runtime's feature probe.

use crate::ast::*;

pub struct SimdOutput {
    pub host_rs: String,
    pub cargo_toml: String,
    pub kernel_names: Vec<String>,
    pub errors: Vec<crate::transpiler::TranspileError>,
}

pub fn transpile_simd(program: &Program, stem: &str, version: &str) -> SimdOutput {
    // Plain-data structs in kernel bodies / device functions are scalar-replaced first.
    let struct_lowered = crate::desugar_gpu_struct::desugar(program);
    let program = &struct_lowered;
    let kernel_names = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Kernel(k) => Some(k.name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let kernels = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Kernel(k) => Some(k.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let global_consts = program
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Let(value) => {
                simd_top_level_const_type(value).map(|const_ty| (value.name.clone(), const_ty))
            }
            _ => None,
        })
        .collect();
    let mut renamed = crate::transpiler::rename_top_level_main(program, "boring_main");
    // `Screen::present` advances the frame counter/clock, so its binding must be mutable.
    for item in renamed.items.iter_mut() {
        if let Item::Let(binding) = item {
            if let Some(Expr { kind: ExprKind::Call(callee, args), .. }) = &mut binding.value {
                if matches!(&callee.kind, ExprKind::Var(name) if name == "Screen") {
                    binding.binding = BindingKind::Var;
                    // `Screen(Dimension(w, h))` without a title: the runtime constructor
                    // always takes one.
                    if args.len() == 1 {
                        let mut title = args[0].clone();
                        title.label = None;
                        title.value.kind = ExprKind::Str("boring".into());
                        args.push(title);
                    }
                }
            }
        }
    }
    let config = crate::transpiler::TranspileConfig {
        gpu_kernels: kernels.clone(),
        is_gpu_target: true,
        is_simd_target: true,
        ..Default::default()
    };
    SIMD_HOST_TYPES.with(|flag| flag.set(true));
    let general = crate::transpiler::transpile_with_config(&renamed, config);
    SIMD_HOST_TYPES.with(|flag| flag.set(false));
    let (has_main, main_throws) = crate::transpiler::detect_boring_main(&renamed, &general);

    let mut e = Emitter {
        out: String::new(),
        errors: general.errors,
        fields: std::collections::HashSet::new(),
        field_axes: std::collections::HashMap::new(),
        local_axes: std::collections::HashMap::new(),
        arrays: std::collections::HashSet::new(),
        dicts: std::collections::HashSet::new(),
        sets: std::collections::HashSet::new(),
        tuple_arities: std::collections::HashMap::new(),
        function_names: renamed
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(function) => Some(function.name.clone()),
                _ => None,
            })
            .collect(),
        function_sigs: renamed
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Fn(function) => Some((
                    function.name.clone(),
                    (
                        function
                            .params
                            .iter()
                            .map(|param| param.ty.as_ref().and_then(host_numeric_ty))
                            .collect(),
                        function.return_ty.as_ref().and_then(|ret| match base(ret) {
                            Type::Optional(inner) => numeric_ty(inner).map(|device| format!("Option<{device}>")),
                            _ => numeric_ty(ret),
                        }),
                        function
                            .params
                            .iter()
                            .map(|param| match &param.ty {
                                Some(t) if is_array(t) => if matches!(t, Type::Qualified(..)) { 2 } else { 1 },
                                _ => 0,
                            })
                            .collect(),
                        function
                            .params
                            .iter()
                            .map(|param| match param.ty.as_ref().map(base) {
                                Some(Type::Optional(inner)) => Some(host_numeric_ty(inner)),
                                _ => None,
                            })
                            .collect(),
                    ),
                )),
                _ => None,
            })
            .collect(),
        global_consts,
        locals: std::collections::HashSet::new(),
        value_types: std::collections::HashMap::new(),
        replayable_locals: std::collections::HashMap::new(),
        replayed_assignments: std::collections::HashMap::new(),
        collective_uniform_locals: std::collections::HashSet::new(),
        while_let_redirect: std::collections::HashMap::new(),
        while_let_break_sync: None,
        value_loop_depth: 0,
        thread_override: None,
        control_flow_depth: 0,
        work_item_return_uses_active_mask: None,
        coop: false,
        needs_coop: false,
        written_fields: std::collections::HashSet::new(),
        atomic_fields: std::collections::HashMap::new(),
    };
    e.raw(PRELUDE);
    let mirrors = device_struct_mirrors(program, &kernels, &mut e.errors);
    e.raw(&mirrors);
    for kernel in &kernels {
        e.kernel(kernel);
    }
    if e.needs_coop {
        e.raw(COOP_RUNTIME);
    }
    let uses_screen = general.code.contains("Screen::new(");
    let uses_dimension = uses_screen
        || general.code.contains("Dimension::new(")
        || kernels.iter().any(|k| k.fields.iter().any(|f| matches!(base(&f.ty), Type::Named(n) if n == "Dimension")));
    if uses_dimension {
        e.raw(super::simd_display::DIMENSION_RUNTIME);
    }
    if uses_screen {
        e.raw(super::simd_display::SCREEN_RUNTIME);
    }
    e.raw(&general.code);
    if has_main {
        let call = if main_throws {
            "if let Err(e) = boring_main() { eprintln!(\"error: {}\", e); std::process::exit(1); }"
        } else {
            "boring_main();"
        };
        if uses_screen {
            e.raw(&format!("\nfn main() {{ __boring_display_run(|| {{ {call} }}); }}\n"));
        } else {
            e.raw(&format!("\nfn main() {{ {call} }}\n"));
        }
    } else {
        e.raw("\nfn main() {}\n");
    }
    let display_deps = if uses_screen { "[dependencies]\nwinit = \"0.28\"\nsoftbuffer = \"0.3\"\n" } else { "" };
    let cargo_toml = format!("[package]\nname = \"{}-simd\"\nversion = \"{}\"\nedition = \"2021\"\n\n[profile.release]\nopt-level = 3\nlto = \"thin\"\ncodegen-units = 1\n\n{display_deps}", stem.replace('_', "-"), version);
    SimdOutput {
        host_rs: e.out,
        cargo_toml,
        kernel_names,
        errors: e.errors,
    }
}

const PRELUDE: &str = r#"// Generated by `boring build --target simd`.
// CPU execution preserves Boring's 3-D grid/block coordinates.  The x axis is
// traversed contiguously so LLVM can vectorize it for the selected CPU path.
#[derive(Clone, Copy, Debug)] struct __BoringCpu;
fn __boring_gpu_device() -> __BoringCpu { __BoringCpu }
fn __boring_gpu_queue() -> __BoringCpu { __BoringCpu }
fn set_gpu_batch_limit(_: i64) {}
fn __boring_gpu_name<T>(_: T) -> String { format!("CPU SIMD ({})", __boring_simd_name()) }
fn __boring_gpu_total_mem<T>(_: T) -> i64 {
    #[cfg(target_os = "macos")]
    { if let Ok(out) = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output() { if let Ok(v) = String::from_utf8_lossy(&out.stdout).trim().parse::<i64>() { return v; } } }
    #[cfg(target_os = "linux")]
    { if let Ok(text) = std::fs::read_to_string("/proc/meminfo") { if let Some(kb) = text.lines().find_map(|l| l.strip_prefix("MemTotal:")).and_then(|l| l.trim().trim_end_matches("kB").trim().parse::<i64>().ok()) { return kb * 1024; } } }
    0
}
fn __boring_gpu_free_mem<T>(_: T) -> i64 {
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("vm_stat").output() {
            let text = String::from_utf8_lossy(&out.stdout);
            let page = text.split("page size of ").nth(1).and_then(|r| r.split_whitespace().next()).and_then(|v| v.parse::<i64>().ok()).unwrap_or(4096);
            let pages = |label: &str| text.lines().find_map(|l| l.strip_prefix(label)).and_then(|v| v.trim().trim_end_matches('.').parse::<i64>().ok()).unwrap_or(0);
            return (pages("Pages free:") + pages("Pages inactive:")) * page;
        }
    }
    #[cfg(target_os = "linux")]
    { if let Ok(text) = std::fs::read_to_string("/proc/meminfo") { if let Some(kb) = text.lines().find_map(|l| l.strip_prefix("MemAvailable:")).and_then(|l| l.trim().trim_end_matches("kB").trim().parse::<i64>().ok()) { return kb * 1024; } } }
    0
}
fn __boring_gpu_used_mem<T: Copy>(d: T) -> i64 { (__boring_gpu_total_mem(d) - __boring_gpu_free_mem(d)).max(0) }
fn __boring_gpu_all() -> Vec<usize> { vec![0] }
fn __boring_gpu_warp_size<T>(_: T) -> i64 { 32 }
fn __boring_gpu_max_threads<T>(_: T) -> i64 { 1024 }
fn __boring_gpu_max_shared_mem<T>(_: T) -> i64 { 32768 }
fn __boring_gpu_utilization<T>(_: T) -> f64 { 0.0 }
fn __boring_gpu_temperature<T>(_: T) -> f64 { 0.0 }
fn __boring_gpu_power_usage<T>(_: T) -> f64 { 0.0 }
fn __boring_gpu_power_limit<T>(_: T) -> f64 { 0.0 }
fn __boring_gpu_memory_bandwidth<T>(_: T) -> f64 { 0.0 }
fn __boring_gpu_compute_capability<T>(_: T) -> Vec<i64> { vec![__boring_simd_lanes() as i64, 0] }

#[derive(Clone, Copy, Debug)] struct __BoringDim3 { x: i32, y: i32, z: i32 }
fn __boring_simd_lanes() -> usize {
    static LANES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LANES.get_or_init(|| {
        if std::env::var_os("BORING_SIMD_SCALAR").is_some() { return 1; }
        #[cfg(any(target_arch="x86", target_arch="x86_64"))] {
            if std::is_x86_feature_detected!("avx512f") { return 16; }
            if std::is_x86_feature_detected!("avx2") { return 8; }
            if std::is_x86_feature_detected!("sse2") { return 4; }
        }
        #[cfg(target_arch="aarch64")] { if std::arch::is_aarch64_feature_detected!("neon") { return 4; } }
        1
    })
}
fn __boring_simd_name() -> &'static str {
    match __boring_simd_lanes() { 16 => "AVX-512", 8 => "AVX2", 4 => { #[cfg(target_arch="aarch64")] { return "NEON"; } #[cfg(not(target_arch="aarch64"))] { "SSE2" } }, _ => "scalar" }
}
#[inline(always)] fn __boring_fma(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(any(target_feature = "fma", target_arch = "aarch64")) { a.mul_add(b, c) } else { a * b + c }
}
/// Dot product with independent partial sums so LLVM can vectorize the reduction.
#[inline(always)] fn __boring_dot_with(a: &[f32], b: &[f32], init: f32, fma: impl Fn(f32, f32, f32) -> f32) -> f32 {
    let n = a.len().min(b.len());
    let (ca, cb) = (a[..n].chunks_exact(16), b[..n].chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0.0f32; 16];
    for (x, y) in ca.zip(cb) { for l in 0..16 { acc[l] = fma(x[l], y[l], acc[l]); } }
    let mut sum = init;
    for l in 0..16 { sum += acc[l]; }
    for (x, y) in ra.iter().zip(rb) { sum = fma(*x, *y, sum); }
    sum
}
/// `out[j] += scale * row[j]`, contiguous in `j` so it vectorizes.
#[inline(always)] fn __boring_axpy_with(out: &mut [f32], scale: f32, row: &[f32], fma: impl Fn(f32, f32, f32) -> f32) {
    for (o, r) in out.iter_mut().zip(row) { *o = fma(scale, *r, *o); }
}
// Baseline x86-64 has no FMA instruction (`f32::mul_add` would be a libm call), so the
// hot loops get an AVX2+FMA twin chosen at run time.
#[cfg(target_arch = "x86_64")] #[target_feature(enable = "avx2,fma")] unsafe fn __boring_dot_fma(a: &[f32], b: &[f32], init: f32) -> f32 {
    __boring_dot_with(a, b, init, |x, y, z| x.mul_add(y, z))
}
#[cfg(target_arch = "x86_64")] #[target_feature(enable = "avx2,fma")] unsafe fn __boring_axpy_fma(out: &mut [f32], scale: f32, row: &[f32]) {
    __boring_axpy_with(out, scale, row, |x, y, z| x.mul_add(y, z))
}
/// `out[r][c] += sum_k a[row0 + r][k] * b[k][col0 + c]` for a band of rows and columns, using a 4x16
/// register tile so the inner loop is pure fused multiply-adds on in-register accumulators.
#[inline(always)] fn __boring_gemm_with(a: &[f32], lda: usize, row0: usize, rows: usize, b: &[f32], ldb: usize, col0: usize, cols: usize, k: usize, out: &mut [f32], ldo: usize, fma: impl Fn(f32, f32, f32) -> f32 + Copy) {
    const MR: usize = 4;
    const NR: usize = 16;
    let mut r = 0;
    while r + MR <= rows {
        let mut c = 0;
        while c + NR <= cols {
            let mut acc = [[0.0f32; NR]; MR];
            for i in 0..MR {
                acc[i].copy_from_slice(&out[(r + i) * ldo + c..(r + i) * ldo + c + NR]);
            }
            for kk in 0..k {
                let brow: &[f32; NR] = b[kk * ldb + col0 + c..kk * ldb + col0 + c + NR].try_into().unwrap();
                for i in 0..MR {
                    let av = a[(row0 + r + i) * lda + kk];
                    for l in 0..NR { acc[i][l] = fma(av, brow[l], acc[i][l]); }
                }
            }
            for i in 0..MR {
                out[(r + i) * ldo + c..(r + i) * ldo + c + NR].copy_from_slice(&acc[i]);
            }
            c += NR;
        }
        if c < cols {
            for i in 0..MR {
                for kk in 0..k {
                    __boring_axpy_with(&mut out[(r + i) * ldo + c..(r + i) * ldo + cols], a[(row0 + r + i) * lda + kk], &b[kk * ldb + col0 + c..kk * ldb + col0 + cols], fma);
                }
            }
        }
        r += MR;
    }
    while r < rows {
        for kk in 0..k {
            __boring_axpy_with(&mut out[r * ldo..r * ldo + cols], a[(row0 + r) * lda + kk], &b[kk * ldb + col0..kk * ldb + col0 + cols], fma);
        }
        r += 1;
    }
}
/// `out[r][c] = bias[col0 + c] + dot(a[row0 + r], w[col0 + c])` with weights stored `[n, k]`: a 4x4 tile of
/// dot products keeps 16 four-lane accumulators in registers, so each loaded vector is reused four times.
#[inline(always)] fn __boring_linear_with(a: &[f32], lda: usize, row0: usize, rows: usize, w: &[f32], ldw: usize, col0: usize, cols: usize, k: usize, bias: Option<&[f32]>, out: &mut [f32], ldo: usize, fma: impl Fn(f32, f32, f32) -> f32 + Copy) {
    const MR: usize = 4;
    const NR: usize = 4;
    const L: usize = 4;
    let init = |c: usize| bias.map_or(0.0f32, |b| b[col0 + c]);
    let (mut r, kv) = (0usize, k / L * L);
    while r < rows {
        let tile_rows = (rows - r).min(MR);
        let mut c = 0usize;
        while c < cols {
            let tile_cols = (cols - c).min(NR);
            if tile_rows == MR && tile_cols == NR {
                let mut acc = [[[0.0f32; L]; NR]; MR];
                let mut kk = 0;
                while kk < kv {
                    let mut av = [[0.0f32; L]; MR];
                    for i in 0..MR { av[i].copy_from_slice(&a[(row0 + r + i) * lda + kk..(row0 + r + i) * lda + kk + L]); }
                    for j in 0..NR {
                        let wv: &[f32; L] = w[(col0 + c + j) * ldw + kk..(col0 + c + j) * ldw + kk + L].try_into().unwrap();
                        for i in 0..MR { for l in 0..L { acc[i][j][l] = fma(av[i][l], wv[l], acc[i][j][l]); } }
                    }
                    kk += L;
                }
                for i in 0..MR {
                    for j in 0..NR {
                        let mut sum = init(c + j);
                        for l in 0..L { sum += acc[i][j][l]; }
                        for t in kv..k { sum = fma(a[(row0 + r + i) * lda + t], w[(col0 + c + j) * ldw + t], sum); }
                        out[(r + i) * ldo + c + j] = sum;
                    }
                }
            } else {
                for i in 0..tile_rows {
                    for j in 0..tile_cols {
                        out[(r + i) * ldo + c + j] = __boring_dot_with(&a[(row0 + r + i) * lda..(row0 + r + i) * lda + k], &w[(col0 + c + j) * ldw..(col0 + c + j) * ldw + k], init(c + j), fma);
                    }
                }
            }
            c += NR;
        }
        r += MR;
    }
}
#[cfg(target_arch = "x86_64")] #[target_feature(enable = "avx2,fma")] unsafe fn __boring_linear_fma(a: &[f32], lda: usize, row0: usize, rows: usize, w: &[f32], ldw: usize, col0: usize, cols: usize, k: usize, bias: Option<&[f32]>, out: &mut [f32], ldo: usize) {
    __boring_linear_with(a, lda, row0, rows, w, ldw, col0, cols, k, bias, out, ldo, |x, y, z| x.mul_add(y, z))
}
fn __boring_linear(a: &[f32], lda: usize, row0: usize, rows: usize, w: &[f32], ldw: usize, col0: usize, cols: usize, k: usize, bias: Option<&[f32]>, out: &mut [f32], ldo: usize) {
    #[cfg(target_arch = "x86_64")]
    { if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") { return unsafe { __boring_linear_fma(a, lda, row0, rows, w, ldw, col0, cols, k, bias, out, ldo) }; } }
    __boring_linear_with(a, lda, row0, rows, w, ldw, col0, cols, k, bias, out, ldo, __boring_fma)
}
#[cfg(target_arch = "x86_64")] #[target_feature(enable = "avx2,fma")] unsafe fn __boring_gemm_fma(a: &[f32], lda: usize, row0: usize, rows: usize, b: &[f32], ldb: usize, col0: usize, cols: usize, k: usize, out: &mut [f32], ldo: usize) {
    __boring_gemm_with(a, lda, row0, rows, b, ldb, col0, cols, k, out, ldo, |x, y, z| x.mul_add(y, z))
}
fn __boring_gemm(a: &[f32], lda: usize, row0: usize, rows: usize, b: &[f32], ldb: usize, col0: usize, cols: usize, k: usize, out: &mut [f32], ldo: usize) {
    #[cfg(target_arch = "x86_64")]
    { if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") { return unsafe { __boring_gemm_fma(a, lda, row0, rows, b, ldb, col0, cols, k, out, ldo) }; } }
    __boring_gemm_with(a, lda, row0, rows, b, ldb, col0, cols, k, out, ldo, __boring_fma)
}
#[inline(always)] fn __boring_dot(a: &[f32], b: &[f32], init: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    { if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") { return unsafe { __boring_dot_fma(a, b, init) }; } }
    __boring_dot_with(a, b, init, __boring_fma)
}
#[inline(always)] fn __boring_axpy(out: &mut [f32], scale: f32, row: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    { if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") { return unsafe { __boring_axpy_fma(out, scale, row) }; } }
    __boring_axpy_with(out, scale, row, __boring_fma)
}
/// Splits an `m x n` row-major result into worker tasks `(first row, first column, slice)`:
/// whole row bands when there are enough rows, otherwise column bands within each row.
fn __boring_tensor_tasks<'a>(c: &'a mut [f32], m: usize, n: usize, workers: usize) -> Vec<(usize, usize, &'a mut [f32])> {
    if workers > m {
        let splits = workers.div_ceil(m).min(n).max(1);
        let per = n.div_ceil(splits);
        let mut tasks = Vec::new();
        for (row, values) in c.chunks_mut(n).enumerate() {
            for (part, slice) in values.chunks_mut(per).enumerate() { tasks.push((row, part * per, slice)); }
        }
        tasks
    } else {
        let rows = m.div_ceil(workers.max(1));
        c.chunks_mut(rows * n).enumerate().map(|(index, slice)| (index * rows, 0, slice)).collect()
    }
}
fn __boring_blocks(gx: usize, gy: usize, gz: usize, items_per_block: usize, run: &(dyn Fn(usize, usize, usize) + Sync)) {
    static MIN_ITEMS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let min_items = *MIN_ITEMS.get_or_init(|| std::env::var("BORING_SIMD_PARALLEL_ITEMS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(32768));
    let total = gx * gy * gz;
    let threads = __boring_simd_threads().min(total);
    if threads <= 1 || total.saturating_mul(items_per_block) < min_items {
        for bz in 0..gz { for by in 0..gy { for bx in 0..gx { run(bx, by, bz); } } }
        return;
    }
    let chunk = (total / (threads * 8)).max(1);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = || loop {
        let start = next.fetch_add(chunk, std::sync::atomic::Ordering::Relaxed);
        if start >= total { break; }
        for id in start..(start + chunk).min(total) { run(id % gx, (id / gx) % gy, id / (gx * gy)); }
    };
    std::thread::scope(|scope| {
        // The caller works too, so a refused thread spawn only costs parallelism.
        for _ in 1..threads {
            let _ = std::thread::Builder::new().spawn_scoped(scope, &work);
        }
        work();
    });
}
fn __boring_simd_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("BORING_SIMD_THREADS").ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|count| count.get()).unwrap_or(1))
    })
}
fn __boring_simd_parallel_threshold() -> usize {
    static THRESHOLD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("BORING_SIMD_PARALLEL_THRESHOLD").ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(268_435_456)
    })
}
#[allow(dead_code)] #[inline(always)] fn __boring_f16_to_f32(bits: u16) -> f32 {
    static TABLE: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| (0..=u16::MAX).map(__boring_f16_to_f32_slow).collect())[bits as usize]
}
#[allow(dead_code)] fn __boring_f16_to_f32_slow(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0f32 } else { -1.0f32 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x03ff) as u32;
    let magnitude = match exponent {
        0 => (fraction as f32) * 2.0f32.powi(-24),
        31 if fraction == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + fraction as f32 / 1024.0) * 2.0f32.powi(exponent - 15),
    };
    sign * magnitude
}
#[allow(dead_code)] #[derive(Clone, Debug, PartialEq)] enum BoringKernelArg<T> { Host(Vec<T>) }
impl<T> BoringKernelArg<T> { fn len(&self) -> usize { match self { Self::Host(v) => v.len() } } }
impl<T> std::ops::Deref for BoringKernelArg<T> { type Target = Vec<T>; fn deref(&self) -> &Vec<T> { match self { Self::Host(v) => v } } }
impl<T> std::ops::DerefMut for BoringKernelArg<T> { fn deref_mut(&mut self) -> &mut Vec<T> { match self { Self::Host(v) => v } } }
"#;

/// Per free function: host width of each numeric parameter, device type of the numeric
/// (or `Option<numeric>`) return, array-parameter kinds, and for each `T?` parameter the
/// host width of its numeric `T`.
type FunctionSig = (Vec<Option<String>>, Option<String>, Vec<u8>, Vec<Option<Option<String>>>);

struct Emitter {
    out: String,
    errors: Vec<crate::transpiler::TranspileError>,
    fields: std::collections::HashSet<String>,
    field_axes: std::collections::HashMap<String, Vec<LabeledAxis>>,
    local_axes: std::collections::HashMap<String, Vec<LabeledAxis>>,
    arrays: std::collections::HashSet<String>,
    dicts: std::collections::HashSet<String>,
    sets: std::collections::HashSet<String>,
    tuple_arities: std::collections::HashMap<String, usize>,
    function_names: std::collections::HashSet<String>,
    function_sigs: std::collections::HashMap<String, FunctionSig>,
    global_consts: std::collections::HashMap<String, String>,
    locals: std::collections::HashSet<String>,
    value_types: std::collections::HashMap<String, String>,
    replayable_locals: std::collections::HashMap<String, Expr>,
    replayed_assignments: std::collections::HashMap<String, String>,
    collective_uniform_locals: std::collections::HashSet<String>,
    while_let_redirect: std::collections::HashMap<String, String>,
    while_let_break_sync: Option<(String, String)>,
    value_loop_depth: usize,
    thread_override: Option<String>,
    control_flow_depth: usize,
    work_item_return_uses_active_mask: Option<bool>,
    coop: bool,
    needs_coop: bool,
    written_fields: std::collections::HashSet<String>,
    atomic_fields: std::collections::HashMap<String, String>,
}
impl Emitter {
    fn raw(&mut self, s: &str) {
        self.out.push_str(s);
    }
    fn line(&mut self, n: usize, s: &str) {
        self.out.push_str(&"    ".repeat(n));
        self.out.push_str(s);
        self.out.push('\n');
    }
    fn kernel(&mut self, k: &KernelDecl) {
        let auto_synced = auto_sync_kernel(k);
        let k = auto_synced.as_ref().unwrap_or(k);
        self.collective_uniform_locals.clear();
        self.atomic_fields = k
            .fields
            .iter()
            .filter(|f| matches!(f.qual, GpuQual::ActorGlobal | GpuQual::ActorUnified) && is_array(&f.ty))
            .map(|f| (f.name.clone(), elem_ty(&f.ty)))
            .collect();
        self.written_fields = entry_method(k)
            .map(|method| written_buffers(&method.body))
            .unwrap_or_default();
        self.fields = k
            .fields
            .iter()
            .filter(|f| !matches!(f.qual, GpuQual::Local | GpuQual::Actor))
            .map(|f| f.name.clone())
            .collect();
        self.arrays = k
            .fields
            .iter()
            .filter(|field| is_array(&field.ty))
            .map(|field| field.name.clone())
            .collect();
        self.dicts = k
            .fields
            .iter()
            .filter(|field| matches!(base(&field.ty), Type::Dict(_, _)))
            .map(|field| field.name.clone())
            .collect();
        self.sets = k
            .fields
            .iter()
            .filter(|field| matches!(base(&field.ty), Type::Set(_)))
            .map(|field| field.name.clone())
            .collect();
        self.field_axes = k
            .fields
            .iter()
            .filter_map(|f| {
                f.ty.as_labeled_array()
                    .map(|(_, axes)| (f.name.clone(), axes.to_vec()))
            })
            .collect();
        self.locals.clear();
        self.tuple_arities.clear();
        self.tuple_arities.extend(k.fields.iter().filter_map(|field| {
            tuple_type_arity(&field.ty).map(|arity| (field.name.clone(), arity))
        }));
        self.local_axes.clear();
        self.value_types.clear();
        self.replayable_locals.clear();
        self.replayed_assignments.clear();
        self.while_let_redirect.clear();
        self.while_let_break_sync = None;
        self.work_item_return_uses_active_mask = None;
        for field in &k.fields {
            let field_ty = if is_array(&field.ty) {
                elem_ty(&field.ty)
            } else {
                ty(&field.ty)
            };
            self.value_types.insert(field.name.clone(), field_ty);
        }
        if let Some(method) = entry_method(k) {
            self.locals
                .extend(method.params.iter().map(|param| param.name.clone()));
            for param in &method.params {
                if let Some(param_ty) = &param.ty {
                    self.value_types.insert(param.name.clone(), ty(param_ty));
                    if let Some(arity) = tuple_type_arity(param_ty) {
                        self.tuple_arities.insert(param.name.clone(), arity);
                    }
                }
            }
            crate::transpiler::helpers::collect_local_decl_names(&method.body, &mut self.locals);
        }
        self.line(0, &format!("#[derive(Debug)] struct {} {{", k.name));
        for f in &k.fields {
            if matches!(f.qual, GpuQual::Actor)
                || (matches!(f.qual, GpuQual::Local) && is_array(&f.ty))
            {
                continue;
            }
            self.line(1, &format!("{}: {},", f.name, ty(&f.ty)));
        }
        self.line(0, "}");
        self.line(0, &format!("impl {} {{", k.name));
        for helper in k.methods.iter().filter(|method| !method.name.is_empty()) {
            self.kernel_helper(helper);
        }
        // A kernel with a `let Dimension` field is constructed as `K(Dimension(w, h))`:
        // `new` takes the flat extent and sizes every dynamic array field to w*h
        // (`'surface` buffers get the sparse random seed the GPU Screen hosts use).
        let has_dim = k.fields.iter().any(|f| {
            matches!(base(&f.ty), Type::Named(n) if n == "Dimension") && matches!(f.qual, GpuQual::Const)
        });
        if has_dim {
            self.line(1, "fn new(width: i32, height: i32, _: __BoringCpu, _: __BoringCpu) -> Self {");
        } else {
            self.line(1, "fn new(_: __BoringCpu, _: __BoringCpu) -> Self {");
        }
        self.line(2, "Self {");
        for f in &k.fields {
            if matches!(f.qual, GpuQual::Actor)
                || (matches!(f.qual, GpuQual::Local) && is_array(&f.ty))
            {
                continue;
            }
            let initial = if let Some(default) = &f.default {
                let value = self.expr(default);
                if simd_needs_as_cast(&f.ty) {
                    format!("({value}) as {}", ty(&f.ty))
                } else {
                    value
                }
            } else if has_dim && matches!(base(&f.ty), Type::Named(n) if n == "Dimension") {
                "__DeviceDimension { width, height }".to_string()
            } else if has_dim && matches!(base(&f.ty), Type::Array(_)) {
                if matches!(f.qual, GpuQual::Surface) && elem_ty(&f.ty) == "u32" {
                    "__boring_surface_seed((width.max(0) as usize) * (height.max(0) as usize))".to_string()
                } else {
                    "vec![Default::default(); (width.max(0) as usize) * (height.max(0) as usize)]".to_string()
                }
            } else {
                default_value(&f.ty)
            };
            self.line(3, &format!("{}: {},", f.name, initial));
        }
        self.line(2, "}");
        self.line(1, "}");
        for f in &k.fields {
            if is_array(&f.ty) && !matches!(f.qual, GpuQual::Local | GpuQual::Actor) {
                let t = elem_ty(&f.ty);
                self.line(
                    1,
                    &format!(
                        "fn copy_{}_to_host(&self) -> Vec<{}> {{ self.{}.clone() }}",
                        f.name, t, f.name
                    ),
                );
                self.line(
                    1,
                    &format!(
                        "fn copy_{}_to_device(&mut self, v: &[{}]) {{ self.{} = v.to_vec(); }}",
                        f.name, t, f.name
                    ),
                );
                self.line(1, &format!("fn resize_{}(&mut self, n: usize) {{ self.{}.resize(n, Default::default()); }}", f.name, f.name));
            }
        }
        self.line(1, "#[inline(never)] fn dispatch(&mut self, gx: u32, gy: u32, gz: u32, dx: u32, dy: u32, dz: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
        self.line(
            2,
            "#[cfg(any(target_arch=\"x86\", target_arch=\"x86_64\"))] unsafe {",
        );
        self.line(3, "match __boring_simd_lanes() {");
        self.line(4, "16 => return self.dispatch_avx512(gx,gy,gz,dx,dy,dz),");
        self.line(4, "8 => return self.dispatch_avx2(gx,gy,gz,dx,dy,dz),");
        self.line(4, "4 => return self.dispatch_sse2(gx,gy,gz,dx,dy,dz),");
        self.line(4, "_ => {},");
        self.line(3, "}");
        self.line(2, "}");
        self.line(2, "self.dispatch_impl(gx,gy,gz,dx,dy,dz)");
        self.line(1, "}");
        self.line(1, "#[cfg(any(target_arch=\"x86\", target_arch=\"x86_64\"))] #[target_feature(enable=\"avx512f\")] unsafe fn dispatch_avx512(&mut self,gx:u32,gy:u32,gz:u32,dx:u32,dy:u32,dz:u32)->Result<(),Box<dyn std::error::Error+Send+Sync>> { self.dispatch_impl(gx,gy,gz,dx,dy,dz) }");
        self.line(1, "#[cfg(any(target_arch=\"x86\", target_arch=\"x86_64\"))] #[target_feature(enable=\"avx2\")] unsafe fn dispatch_avx2(&mut self,gx:u32,gy:u32,gz:u32,dx:u32,dy:u32,dz:u32)->Result<(),Box<dyn std::error::Error+Send+Sync>> { self.dispatch_impl(gx,gy,gz,dx,dy,dz) }");
        self.line(1, "#[cfg(any(target_arch=\"x86\", target_arch=\"x86_64\"))] #[target_feature(enable=\"sse2\")] unsafe fn dispatch_sse2(&mut self,gx:u32,gy:u32,gz:u32,dx:u32,dy:u32,dz:u32)->Result<(),Box<dyn std::error::Error+Send+Sync>> { self.dispatch_impl(gx,gy,gz,dx,dy,dz) }");
        self.line(1, "#[inline(always)] fn dispatch_impl(&mut self, gx: u32, gy: u32, gz: u32, dx: u32, dy: u32, dz: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {");
        self.line(
            2,
            "let __block_dim = __BoringDim3 { x: dx as i32, y: dy as i32, z: dz as i32 };",
        );
        self.line(
            2,
            "let __grid_dim = __BoringDim3 { x: gx as i32, y: gy as i32, z: gz as i32 };",
        );
        if k.methods
            .iter()
            .any(|method| crate::interpreter::eval_gpu::stmts_use_gpu_warp(&method.body))
        {
            self.line(2, "let __simd_lanes: i32 = 32;");
        }
        let phase_out_mark = self.out.len();
        let phase_error_mark = self.errors.len();
        if let Some(m) = entry_method(k) {
            let has_top_level_sync = m
                .body
                .iter()
                .any(is_simd_barrier_stmt);
            let barrier_for = m.body.iter().enumerate().find_map(|(for_index, stmt)| {
                let Stmt::For(for_stmt) = stmt else { return None };
                let safe_prefix = m.body[..for_index].iter().all(|stmt| {
                    !matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
                        && !contains_nested_simd_barrier(stmt)
                });
                let uniform_locals = self.block_uniform_locals(&m.body[..for_index]);
                (safe_prefix
                    && self.block_uniform_expr_with_locals(
                        &for_stmt.iterable,
                        &uniform_locals,
                    )
                    && for_stmt.body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    }))
                .then_some((for_index, for_stmt))
            });
            let barrier_if = m.body.iter().enumerate().find_map(|(if_index, stmt)| {
                let Stmt::If(if_stmt) = stmt else { return None };
                let has_barrier = if_stmt
                    .branches
                    .iter()
                    .any(|(_, body)| body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    }))
                    || if_stmt
                        .else_body
                        .as_ref()
                        .is_some_and(|body| body.iter().any(|stmt| {
                            is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                        }));
                let safe_prefix = m.body[..if_index].iter().all(|stmt| {
                    !matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
                        && !contains_nested_simd_barrier(stmt)
                });
                let uniform_locals = self.block_uniform_locals(&m.body[..if_index]);
                (has_barrier
                    && safe_prefix
                    && if_stmt
                        .branches
                        .iter()
                        .all(|(condition, _)| {
                            self.block_uniform_expr_with_locals(condition, &uniform_locals)
                        }))
                .then_some((if_index, if_stmt))
            });
            let barrier_if_let = m.body.iter().enumerate().find_map(|(if_index, stmt)| {
                let Stmt::IfLet(if_stmt) = stmt else { return None };
                let has_barrier = if_stmt.then_body.iter().any(|stmt| {
                    is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                })
                    || if_stmt
                        .elif_branches
                        .iter()
                        .any(|branch| branch.body.iter().any(|stmt| {
                            is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                        }))
                    || if_stmt
                        .else_body
                        .as_ref()
                        .is_some_and(|body| body.iter().any(|stmt| {
                            is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                        }));
                let safe_prefix = m.body[..if_index].iter().all(|stmt| {
                    !matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
                        && !contains_nested_simd_barrier(stmt)
                });
                let uniform_locals = self.block_uniform_locals(&m.body[..if_index]);
                (has_barrier
                    && safe_prefix
                    && self.block_uniform_guard_clauses_with_locals(
                        &if_stmt.clauses,
                        &uniform_locals,
                    )
                    && if_stmt
                        .elif_branches
                        .iter()
                        .all(|branch| {
                            self.block_uniform_guard_clauses_with_locals(
                                &branch.clauses,
                                &uniform_locals,
                            )
                        }))
                .then_some((if_index, if_stmt))
            });
            let barrier_match = m.body.iter().enumerate().find_map(|(match_index, stmt)| {
                let Stmt::Match(match_stmt) = stmt else { return None };
                let safe_prefix = m.body[..match_index].iter().all(|stmt| {
                    !matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
                        && !contains_nested_simd_barrier(stmt)
                });
                let uniform_locals = self.block_uniform_locals(&m.body[..match_index]);
                (safe_prefix
                    && self.block_uniform_expr_with_locals(
                        &match_stmt.subject,
                        &uniform_locals,
                    )
                    && match_stmt.arms.iter().all(|arm| {
                        self.block_uniform_match_arm_guard_with_locals(arm, &uniform_locals)
                    })
                    && match_stmt.arms.iter().any(|arm| {
                        matches!(&arm.body, MatchBody::Block(body)
                            if body.iter().any(|stmt| {
                                is_simd_barrier_stmt(stmt)
                                    || contains_nested_simd_barrier(stmt)
                            }))
                    }))
                .then_some((match_index, match_stmt))
            });
            let barrier_guard = m.body.iter().enumerate().find_map(|(guard_index, stmt)| {
                let Stmt::Guard(guard_stmt) = stmt else { return None };
                let safe_prefix = m.body[..guard_index].iter().all(|stmt| {
                    !matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
                        && !contains_nested_simd_barrier(stmt)
                });
                let uniform_locals = self.block_uniform_locals(&m.body[..guard_index]);
                let uniform = match &guard_stmt.cond {
                    GuardCond::Expr(condition) => {
                        self.block_uniform_expr_with_locals(condition, &uniform_locals)
                    }
                    GuardCond::Clauses(clauses) => self
                        .block_uniform_guard_clauses_with_locals(clauses, &uniform_locals),
                };
                (uniform
                    && safe_prefix
                    && guard_stmt.else_body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    }))
                .then_some((guard_index, guard_stmt))
            });
            let barrier_while = m.body.iter().enumerate().find_map(|(while_index, stmt)| {
                let Stmt::While(while_stmt) = stmt else { return None };
                if m.body[..while_index]
                    .iter()
                    .any(contains_nested_simd_barrier)
                {
                    return None;
                }
                if !while_stmt.body.iter().any(|stmt| {
                    is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                }) {
                    return None;
                }
                let Stmt::Expr(update) = while_stmt.body.last()? else { return None };
                let ExprKind::Assign(update_target, update_value) = &update.kind else { return None };
                let ExprKind::Var(update_name) = &update_target.kind else { return None };
                let declaration_index = m.body[..while_index].iter().position(|candidate| {
                    matches!(candidate, Stmt::Let(local)
                        if local.name == *update_name
                            && local.binding.is_mutable()
                            && local.value.as_ref().is_some_and(|value| self.block_uniform_expr(value)))
                })?;
                let Stmt::Let(declaration) = &m.body[declaration_index] else { unreachable!() };
                let uniform_locals = std::collections::HashSet::from([update_name.clone()]);
                if !expr_mentions_var(&while_stmt.condition, update_name)
                    || !self.block_uniform_expr_with_locals(&while_stmt.condition, &uniform_locals)
                    || !self.block_uniform_expr_with_locals(update_value, &uniform_locals)
                {
                    return None;
                }
                Some((while_index, declaration_index, while_stmt, declaration, update_value.as_ref()))
            });
            let barrier_do_while = m.body.iter().enumerate().find_map(|(loop_index, stmt)| {
                let Stmt::DoWhile(loop_stmt) = stmt else { return None };
                if m.body[..loop_index].iter().any(contains_nested_simd_barrier)
                    || !loop_stmt.body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    })
                {
                    return None;
                }
                let Stmt::Expr(update) = loop_stmt.body.last()? else { return None };
                let ExprKind::Assign(update_target, update_value) = &update.kind else { return None };
                let ExprKind::Var(update_name) = &update_target.kind else { return None };
                let declaration_index = m.body[..loop_index].iter().position(|candidate| {
                    matches!(candidate, Stmt::Let(local)
                        if local.name == *update_name
                            && local.binding.is_mutable()
                            && local.value.as_ref().is_some_and(|value| self.block_uniform_expr(value)))
                })?;
                let Stmt::Let(declaration) = &m.body[declaration_index] else { unreachable!() };
                let uniform_locals = std::collections::HashSet::from([update_name.clone()]);
                (expr_mentions_var(&loop_stmt.condition, update_name)
                    && self.block_uniform_expr_with_locals(&loop_stmt.condition, &uniform_locals)
                    && self.block_uniform_expr_with_locals(update_value, &uniform_locals))
                    .then_some((loop_index, declaration_index, loop_stmt, declaration, update_value.as_ref()))
            });
            let barrier_while_let = m.body.iter().enumerate().find_map(|(loop_index, stmt)| {
                let Stmt::WhileLet(loop_stmt) = stmt else { return None };
                let bound_names = collective_while_let_bound_names(loop_stmt);
                if bound_names.is_empty()
                    || m.body[..loop_index].iter().any(contains_nested_simd_barrier)
                    || !loop_stmt.body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    })
                {
                    return None;
                }
                let ExprKind::Var(state_name) = &loop_stmt.value.kind else { return None };
                let Stmt::Expr(update) = loop_stmt.body.last()? else { return None };
                let ExprKind::Assign(update_target, update_value) = &update.kind else { return None };
                if !matches!(&update_target.kind, ExprKind::Var(name) if name == state_name) {
                    return None;
                }
                let declaration_index = m.body[..loop_index].iter().position(|candidate| {
                    matches!(candidate, Stmt::Let(local)
                        if local.name == *state_name
                            && local.binding.is_mutable()
                            && local.value.as_ref().is_some_and(|value| self.block_uniform_expr(value)))
                })?;
                let Stmt::Let(declaration) = &m.body[declaration_index] else { unreachable!() };
                let mut uniform_locals = std::collections::HashSet::from([state_name.clone()]);
                uniform_locals.extend(bound_names.iter().cloned());
                (bound_names.iter().any(|name| expr_mentions_var(update_value, name))
                    && self.block_uniform_expr_with_locals(update_value, &uniform_locals))
                    .then_some((loop_index, declaration_index, loop_stmt, declaration, update_value.as_ref()))
            });
            let barrier_loop = m.body.iter().enumerate().find_map(|(loop_index, stmt)| {
                let Stmt::Loop(loop_stmt) = stmt else { return None };
                if m.body[..loop_index].iter().any(contains_nested_simd_barrier)
                    || !loop_stmt.body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    })
                    || loop_stmt.body.len() < 2
                {
                    return None;
                }
                let Stmt::Expr(update) = &loop_stmt.body[loop_stmt.body.len() - 2] else { return None };
                let ExprKind::Assign(update_target, update_value) = &update.kind else { return None };
                let ExprKind::Var(update_name) = &update_target.kind else { return None };
                let break_condition = canonical_collective_loop_break_condition(loop_stmt.body.last()?)?;
                let declaration_index = m.body[..loop_index].iter().position(|candidate| {
                    matches!(candidate, Stmt::Let(local)
                        if local.name == *update_name
                            && local.binding.is_mutable()
                            && local.value.as_ref().is_some_and(|value| self.block_uniform_expr(value)))
                })?;
                let Stmt::Let(declaration) = &m.body[declaration_index] else { unreachable!() };
                let uniform_locals = std::collections::HashSet::from([update_name.clone()]);
                (expr_mentions_var(break_condition, update_name)
                    && self.block_uniform_expr_with_locals(update_value, &uniform_locals)
                    && self.block_uniform_expr_with_locals(break_condition, &uniform_locals))
                    .then_some((
                        loop_index,
                        declaration_index,
                        loop_stmt,
                        declaration,
                        update_value.as_ref(),
                        break_condition,
                    ))
            });
            if has_top_level_sync
                || barrier_for.is_some()
                || barrier_if.is_some()
                || barrier_if_let.is_some()
                || barrier_match.is_some()
                || barrier_guard.is_some()
                || barrier_while.is_some()
                || barrier_do_while.is_some()
                || barrier_while_let.is_some()
                || barrier_loop.is_some()
            {
                self.line(2, "for bz in 0..gz as usize { for by in 0..gy as usize { for bx in 0..gx as usize {");
                self.line(3, "let __block = __BoringDim3 { x: bx as i32, y: by as i32, z: bz as i32 };");
                for field in k.fields.iter().filter(|f| matches!(f.qual, GpuQual::Actor)) {
                    self.line(
                        3,
                        &format!(
                            "let mut {}: {} = {};",
                            field.name,
                            ty(&field.ty),
                            default_value(&field.ty)
                        ),
                    );
                }
                self.line(3, "let mut __boring_barrier_active = vec![true; (dx * dy * dz) as usize];");
                for field in k.fields.iter().filter(|f| matches!(f.qual, GpuQual::Local)) {
                    let initial = if is_array(&field.ty) {
                        default_value(&field.ty)
                    } else {
                        format!("self.{}.clone()", field.name)
                    };
                    self.line(
                        3,
                        &format!(
                            "let mut __boring_local_{}: Vec<{}> = vec![{}; (dx * dy * dz) as usize];",
                            field.name,
                            ty(&field.ty),
                            initial
                        ),
                    );
                }
                if let Some((for_index, for_stmt)) = barrier_for {
                    let prefix = &m.body[..for_index];
                    let replayed = self.replayable_barrier_locals(prefix);
                    for local in self.block_uniform_local_statements(prefix) {
                        self.stmt(&local, 3);
                    }
                    self.emit_barrier_phases_seeded(k, prefix, 3, &[]);
                    let iterable = self.expr(&for_stmt.iterable);
                    self.line(
                        3,
                        &format!("for {} in {} {{", for_stmt.vars.join(", "), iterable),
                    );
                    let inherited_loop_vars = for_stmt
                        .vars
                        .iter()
                        .filter(|name| self.collective_uniform_locals.contains(*name))
                        .cloned()
                        .collect::<Vec<_>>();
                    self.collective_uniform_locals.extend(for_stmt.vars.iter().cloned());
                    self.emit_following_collective_controls(k, &for_stmt.body, 4, &replayed);
                    for name in &for_stmt.vars {
                        if !inherited_loop_vars.contains(name) {
                            self.collective_uniform_locals.remove(name);
                        }
                    }
                    self.line(3, "}");
                    self.emit_following_collective_controls(
                        k,
                        &m.body[for_index + 1..],
                        3,
                        &replayed,
                    );
                } else if let Some((if_index, if_stmt)) = barrier_if {
                    let prefix = &m.body[..if_index];
                    let replayed = self.replayable_barrier_locals(prefix);
                    for local in self.block_uniform_local_statements(prefix) {
                        self.stmt(&local, 3);
                    }
                    self.emit_barrier_phases_seeded(k, prefix, 3, &[]);
                    for (branch_index, (condition, body)) in if_stmt.branches.iter().enumerate() {
                        let condition = self.expr(condition);
                        self.line(
                            3,
                            &format!(
                                "{}if {condition} {{",
                                if branch_index == 0 { "" } else { "else " }
                            ),
                        );
                        self.emit_following_collective_controls(k, body, 4, &replayed);
                        self.line(3, "}");
                    }
                    if let Some(body) = &if_stmt.else_body {
                        self.line(3, "else {");
                        self.emit_following_collective_controls(k, body, 4, &replayed);
                        self.line(3, "}");
                    }
                    self.emit_following_collective_controls(
                        k,
                        &m.body[if_index + 1..],
                        3,
                        &replayed,
                    );
                } else if let Some((if_index, if_stmt)) = barrier_if_let {
                    let prefix = &m.body[..if_index];
                    let replayed = self.replayable_barrier_locals(prefix);
                    for local in self.block_uniform_local_statements(prefix) {
                        self.stmt(&local, 3);
                    }
                    self.emit_barrier_phases_seeded(k, prefix, 3, &[]);
                    self.emit_collective_if_let_clauses(
                        k,
                        &if_stmt.clauses,
                        0,
                        &if_stmt.then_body,
                        &if_stmt.elif_branches,
                        if_stmt.else_body.as_deref(),
                        3,
                        &replayed,
                    );
                    self.emit_following_collective_controls(
                        k,
                        &m.body[if_index + 1..],
                        3,
                        &replayed,
                    );
                } else if let Some((guard_index, guard_stmt)) = barrier_guard {
                    let prefix = &m.body[..guard_index];
                    let replayed = self.replayable_barrier_locals(prefix);
                    for local in self.block_uniform_local_statements(prefix) {
                        self.stmt(&local, 3);
                    }
                    self.emit_barrier_phases_seeded(k, prefix, 3, &[]);
                    match &guard_stmt.cond {
                        GuardCond::Expr(condition) => {
                            let condition = self.expr(condition);
                            self.line(3, &format!("if !({condition}) {{"));
                            self.emit_following_collective_controls(
                                k,
                                &guard_stmt.else_body,
                                4,
                                &replayed,
                            );
                            self.line(3, "}");
                            self.emit_following_collective_controls(
                                k,
                                &m.body[guard_index + 1..],
                                3,
                                &replayed,
                            );
                        }
                        GuardCond::Clauses(clauses) => self.emit_collective_guard_clauses(
                            k,
                            clauses,
                            0,
                            &guard_stmt.else_body,
                            &m.body[guard_index + 1..],
                            3,
                            &replayed,
                        ),
                    }
                } else if let Some((while_index, declaration_index, while_stmt, declaration, update)) = barrier_while {
                    let uniform_name = &declaration.name;
                    let uniform_ty = declaration
                        .ty
                        .as_ref()
                        .map(ty)
                        .or_else(|| declaration.value.as_ref().and_then(|value| self.expr_type(value)))
                        .unwrap_or_else(|| "i32".into());
                    let initial = self.expr(declaration.value.as_ref().expect("checked above"));
                    self.line(3, &format!("let mut {uniform_name}: {uniform_ty} = {initial};"));
                    self.value_types.insert(uniform_name.clone(), uniform_ty);

                    let prefix = m.body[..while_index]
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != declaration_index)
                        .map(|(_, stmt)| stmt.clone())
                        .collect::<Vec<_>>();
                    let replayed = self.replayable_barrier_locals(&prefix);
                    self.emit_barrier_phases_seeded(k, &prefix, 3, &[]);
                    let condition = self.expr(&while_stmt.condition);
                    self.line(3, &format!("while {condition} {{"));
                    let inserted_uniform = self
                        .collective_uniform_locals
                        .insert(uniform_name.clone());
                    self.emit_following_collective_controls(
                        k,
                        &while_stmt.body[..while_stmt.body.len() - 1],
                        4,
                        &replayed,
                    );
                    if inserted_uniform {
                        self.collective_uniform_locals.remove(uniform_name);
                    }
                    let update = self.expr(update);
                    self.line(4, &format!("{uniform_name} = {update};"));
                    self.line(3, "}");
                    self.emit_following_collective_controls(
                        k,
                        &m.body[while_index + 1..],
                        3,
                        &replayed,
                    );
                } else if let Some((loop_index, declaration_index, loop_stmt, declaration, update)) = barrier_do_while {
                    let uniform_name = &declaration.name;
                    let uniform_ty = declaration.ty.as_ref().map(ty)
                        .or_else(|| declaration.value.as_ref().and_then(|value| self.expr_type(value)))
                        .unwrap_or_else(|| "i32".into());
                    let initial = self.expr(declaration.value.as_ref().expect("checked above"));
                    self.line(3, &format!("let mut {uniform_name}: {uniform_ty} = {initial};"));
                    self.value_types.insert(uniform_name.clone(), uniform_ty);
                    let prefix = m.body[..loop_index].iter().enumerate()
                        .filter(|(index, _)| *index != declaration_index)
                        .map(|(_, stmt)| stmt.clone()).collect::<Vec<_>>();
                    let replayed = self.replayable_barrier_locals(&prefix);
                    self.emit_barrier_phases_seeded(k, &prefix, 3, &[]);
                    self.line(3, "loop {");
                    let inserted_uniform = self.collective_uniform_locals.insert(uniform_name.clone());
                    self.emit_following_collective_controls(k, &loop_stmt.body[..loop_stmt.body.len() - 1], 4, &replayed);
                    if inserted_uniform { self.collective_uniform_locals.remove(uniform_name); }
                    let update = self.expr(update);
                    self.line(4, &format!("{uniform_name} = {update};"));
                    let condition = self.expr(&loop_stmt.condition);
                    self.line(4, &format!("if !({condition}) {{ break; }}"));
                    self.line(3, "}");
                    self.emit_following_collective_controls(k, &m.body[loop_index + 1..], 3, &replayed);
                } else if let Some((loop_index, declaration_index, loop_stmt, declaration, update)) = barrier_while_let {
                    let state_name = &declaration.name;
                    let state_ty = declaration.ty.as_ref().map(ty)
                        .unwrap_or_else(|| "Option<i32>".into());
                    let initial = self.expr(declaration.value.as_ref().expect("checked above"));
                    self.line(3, &format!("let mut {state_name}: {state_ty} = {initial};"));
                    let prefix = m.body[..loop_index].iter().enumerate()
                        .filter(|(index, _)| *index != declaration_index)
                        .map(|(_, stmt)| stmt.clone()).collect::<Vec<_>>();
                    let replayed = self.replayable_barrier_locals(&prefix);
                    self.emit_barrier_phases_seeded(k, &prefix, 3, &[]);
                    let match_pattern = simd_collective_while_let_pattern(loop_stmt);
                    self.line(3, &format!("while let {match_pattern} = {state_name} {{"));
                    let inserted_state = self.collective_uniform_locals.insert(state_name.clone());
                    let bound_names = collective_while_let_bound_names(loop_stmt);
                    let inherited_bound_names = bound_names.iter()
                        .filter(|name| self.collective_uniform_locals.contains(*name))
                        .cloned().collect::<std::collections::HashSet<_>>();
                    self.collective_uniform_locals.extend(bound_names.iter().cloned());
                    self.emit_following_collective_controls(k, &loop_stmt.body[..loop_stmt.body.len() - 1], 4, &replayed);
                    for name in bound_names {
                        if !inherited_bound_names.contains(&name) {
                            self.collective_uniform_locals.remove(&name);
                        }
                    }
                    if inserted_state { self.collective_uniform_locals.remove(state_name); }
                    let update = self.expr(update);
                    self.line(4, &format!("{state_name} = {update};"));
                    self.line(3, "}");
                    self.emit_following_collective_controls(k, &m.body[loop_index + 1..], 3, &replayed);
                } else if let Some((loop_index, declaration_index, loop_stmt, declaration, update, break_condition)) = barrier_loop {
                    let uniform_name = &declaration.name;
                    let uniform_ty = declaration.ty.as_ref().map(ty)
                        .or_else(|| declaration.value.as_ref().and_then(|value| self.expr_type(value)))
                        .unwrap_or_else(|| "i32".into());
                    let initial = self.expr(declaration.value.as_ref().expect("checked above"));
                    self.line(3, &format!("let mut {uniform_name}: {uniform_ty} = {initial};"));
                    self.value_types.insert(uniform_name.clone(), uniform_ty);
                    let prefix = m.body[..loop_index].iter().enumerate()
                        .filter(|(index, _)| *index != declaration_index)
                        .map(|(_, stmt)| stmt.clone()).collect::<Vec<_>>();
                    let replayed = self.replayable_barrier_locals(&prefix);
                    self.emit_barrier_phases_seeded(k, &prefix, 3, &[]);
                    self.line(3, "loop {");
                    let inserted_uniform = self.collective_uniform_locals.insert(uniform_name.clone());
                    self.emit_following_collective_controls(
                        k,
                        &loop_stmt.body[..loop_stmt.body.len() - 2],
                        4,
                        &replayed,
                    );
                    if inserted_uniform { self.collective_uniform_locals.remove(uniform_name); }
                    let update = self.expr(update);
                    self.line(4, &format!("{uniform_name} = {update};"));
                    let break_condition = self.expr(break_condition);
                    self.line(4, &format!("if {break_condition} {{ break; }}"));
                    self.line(3, "}");
                    self.emit_following_collective_controls(k, &m.body[loop_index + 1..], 3, &replayed);
                } else if let Some((match_index, match_stmt)) = barrier_match {
                    let prefix = &m.body[..match_index];
                    let replayed = self.replayable_barrier_locals(prefix);
                    for local in self.block_uniform_local_statements(prefix) {
                        self.stmt(&local, 3);
                    }
                    self.emit_barrier_phases_seeded(k, prefix, 3, &[]);
                    let subject_is_string =
                        self.expr_type(&match_stmt.subject).as_deref() == Some("String");
                    let subject = self.expr(&match_stmt.subject);
                    let subject = if subject_is_string {
                        format!("({subject}).as_str()")
                    } else {
                        subject
                    };
                    self.line(3, &format!("match {subject} {{"));
                    for arm in &match_stmt.arms {
                        let patterns = arm
                            .patterns
                            .iter()
                            .map(simd_pattern)
                            .collect::<Vec<_>>()
                            .join(" | ");
                        let guard = arm
                            .guard
                            .as_ref()
                            .map(|guard| format!(" if {}", self.expr(guard)))
                            .unwrap_or_default();
                        self.line(4, &format!("{patterns}{guard} => {{"));
                        let mut bound_names = std::collections::HashSet::new();
                        for pattern in &arm.patterns {
                            collect_simd_pattern_bind_names(pattern, &mut bound_names);
                        }
                        let inherited_bound_names = bound_names
                            .iter()
                            .filter(|name| self.collective_uniform_locals.contains(*name))
                            .cloned()
                            .collect::<std::collections::HashSet<_>>();
                        self.collective_uniform_locals.extend(bound_names.iter().cloned());
                        match &arm.body {
                            MatchBody::Block(body) => {
                                self.emit_following_collective_controls(k, body, 5, &replayed)
                            }
                            MatchBody::Expr(value) => {
                                self.emit_barrier_phases_seeded(
                                    k,
                                    &[Stmt::Expr(value.clone())],
                                    5,
                                    &replayed,
                                )
                            }
                        }
                        for name in bound_names {
                            if !inherited_bound_names.contains(&name) {
                                self.collective_uniform_locals.remove(&name);
                            }
                        }
                        self.line(4, "},");
                    }
                    self.line(3, "}");
                    self.emit_following_collective_controls(
                        k,
                        &m.body[match_index + 1..],
                        3,
                        &replayed,
                    );
                } else {
                    self.emit_barrier_phases(k, &m.body, 3);
                }
                self.line(2, "} } }");
            } else {
                self.line(2, "for bz in 0..gz as usize { for by in 0..gy as usize { for bx in 0..gx as usize {");
                self.line(3, "let __block = __BoringDim3 { x: bx as i32, y: by as i32, z: bz as i32 };");
                for field in k.fields.iter().filter(|f| matches!(f.qual, GpuQual::Actor)) {
                    self.line(
                        3,
                        &format!(
                            "let mut {}: {} = {};",
                            field.name,
                            ty(&field.ty),
                            default_value(&field.ty)
                        ),
                    );
                }
                self.line(3, "for tz in 0..dz as usize { for ty in 0..dy as usize { '__boring_work_item: for __x in 0..dx as usize {");
                self.line(4, "let __thread = __BoringDim3 { x: __x as i32, y: ty as i32, z: tz as i32 };");
                for field in k.fields.iter().filter(|f| matches!(f.qual, GpuQual::Local)) {
                    let initial = if is_array(&field.ty) {
                        default_value(&field.ty)
                    } else {
                        format!("self.{}.clone()", field.name)
                    };
                    self.line(
                        4,
                        &format!("let mut {}: {} = {};", field.name, ty(&field.ty), initial),
                    );
                }
                self.work_item_return_uses_active_mask = Some(false);
                for stmt in &m.body {
                    self.stmt(stmt, 4);
                }
                self.work_item_return_uses_active_mask = None;
                self.line(3, "} } }");
                self.line(2, "} } }");
            }
        } else {
            self.line(2, "for bz in 0..gz as usize { for by in 0..gy as usize { for bx in 0..gx as usize { for tz in 0..dz as usize { for ty in 0..dy as usize { for __x in 0..dx as usize {");
            self.line(4, "let __thread = __BoringDim3 { x: __x as i32, y: ty as i32, z: tz as i32 };");
            self.line(4, "let __block = __BoringDim3 { x: bx as i32, y: by as i32, z: bz as i32 };");
            self.line(2, "} } } } } }");
        }
        if self.errors[phase_error_mark..]
            .iter()
            .any(|error| is_cooperative_candidate_error(&error.message))
        {
            if let Some(m) = entry_method(k) {
                self.out.truncate(phase_out_mark);
                self.errors.truncate(phase_error_mark);
                self.emit_cooperative_body(k, m);
            }
        }
        let block_method = self.parallelize_blocks(k, phase_out_mark);
        self.line(2, "Ok(())");
        self.line(1, "}");
        self.raw(&block_method);
        self.line(0, "}");
    }
    /// Runs the grid's blocks on several cores: the per-block loop body becomes a method
    /// that worker threads call through `__boring_blocks`. Kernels using atomics keep the
    /// sequential loop (their read-modify-writes are plain on the CPU). Returns the text
    /// of the extracted methods, empty when the body is left in place.
    fn parallelize_blocks(&mut self, k: &KernelDecl, region_start: usize) -> String {
        const OPEN: &str = "        for bz in 0..gz as usize { for by in 0..gy as usize { for bx in 0..gx as usize {\n";
        const CLOSE: &str = "        } } }\n";
        if k.fields.iter().any(|f| {
            matches!(f.qual, GpuQual::ActorGlobal | GpuQual::ActorUnified)
                && !matches!(elem_ty(&f.ty).as_str(), "i32" | "u32" | "i64" | "u64")
        }) {
            return String::new();
        }
        let region = self.out[region_start..].to_string();
        let (Some(open), Some(close)) = (region.find(OPEN), region.rfind(CLOSE)) else {
            return String::new();
        };
        if close <= open || !region[close + CLOSE.len()..].trim().is_empty() {
            return String::new();
        }
        let body = &region[open + OPEN.len()..close];
        let params = "gx: u32, gy: u32, gz: u32, dx: u32, dy: u32, dz: u32, bx: usize, by: usize, bz: usize, __block_dim: __BoringDim3, __grid_dim: __BoringDim3, __simd_lanes: i32";
        let args = "gx, gy, gz, dx, dy, dz, bx, by, bz, __block_dim, __grid_dim, 32";
        let mut call = String::new();
        call.push_str("        let __addr = self as *mut Self as usize;\n");
        call.push_str("        __boring_blocks(gx as usize, gy as usize, gz as usize, (dx as usize) * (dy as usize) * (dz as usize), &|bx, by, bz| {\n");
        call.push_str("            let __this = unsafe { &mut *(__addr as *mut Self) };\n");
        call.push_str("            #[cfg(any(target_arch=\"x86\", target_arch=\"x86_64\"))] unsafe { match __boring_simd_lanes() {\n");
        for (lanes, feature) in [(16, "avx512"), (8, "avx2"), (4, "sse2")] {
            call.push_str(&format!("                {lanes} => return __this.__block_{feature}({args}),\n"));
        }
        call.push_str("                _ => {}\n            } }\n");
        call.push_str(&format!("            __this.__block({args})\n        }});\n"));
        let mut methods = String::new();
        methods.push_str(&format!("    #[inline(always)] #[allow(unused_variables, unused_mut, unused_parens)] fn __block(&mut self, {params}) {{\n{body}    }}\n"));
        for (feature, name) in [("avx512f", "avx512"), ("avx2", "avx2"), ("sse2", "sse2")] {
            methods.push_str(&format!("    #[cfg(any(target_arch=\"x86\", target_arch=\"x86_64\"))] #[target_feature(enable=\"{feature}\")] unsafe fn __block_{name}(&mut self, {params}) {{ self.__block({args}) }}\n"));
        }
        self.out.truncate(region_start + open);
        self.out.push_str(&call);
        methods
    }
    /// General fallback for kernels whose `sync`/`.warp` collectives the phase
    /// splitter cannot express (per-thread state carried across barriers,
    /// divergent shuffle operands): every work-item of a block runs as a
    /// cooperatively scheduled fiber (see `COOP_RUNTIME`), so barriers and
    /// shuffles keep their GPU meaning wherever they appear.
    fn emit_cooperative_body(&mut self, k: &KernelDecl, m: &FnDecl) {
        self.needs_coop = true;
        self.coop = true;
        self.collective_uniform_locals.clear();
        self.replayable_locals.clear();
        self.replayed_assignments.clear();
        self.while_let_redirect.clear();
        self.while_let_break_sync = None;
        self.thread_override = None;
        self.control_flow_depth = 0;
        self.line(2, "let __simd_lanes: i32 = 32;");
        self.line(2, "for bz in 0..gz as usize { for by in 0..gy as usize { for bx in 0..gx as usize {");
        self.line(3, "let __block = __BoringDim3 { x: bx as i32, y: by as i32, z: bz as i32 };");
        for field in k.fields.iter().filter(|f| matches!(f.qual, GpuQual::Actor)) {
            self.line(
                3,
                &format!("let mut {}: {} = {};", field.name, ty(&field.ty), default_value(&field.ty)),
            );
        }
        self.line(3, "let __coop_body: &mut dyn FnMut(usize) = &mut |__tid: usize| {");
        self.line(4, "let __x = __tid % dx as usize;");
        self.line(4, "let ty = (__tid / dx as usize) % dy as usize;");
        self.line(4, "let tz = __tid / (dx as usize * dy as usize);");
        self.line(4, "let __thread = __BoringDim3 { x: __x as i32, y: ty as i32, z: tz as i32 };");
        for field in k.fields.iter().filter(|f| matches!(f.qual, GpuQual::Local)) {
            let initial = if is_array(&field.ty) {
                default_value(&field.ty)
            } else {
                format!("self.{}.clone()", field.name)
            };
            self.line(4, &format!("let mut {}: {} = {};", field.name, ty(&field.ty), initial));
        }
        self.line(4, "'__boring_work_item: for _ in 0..1 {");
        self.work_item_return_uses_active_mask = Some(false);
        for stmt in &m.body {
            self.stmt(stmt, 5);
        }
        self.work_item_return_uses_active_mask = None;
        self.line(4, "}");
        self.line(3, "};");
        self.line(3, "__coop::run_block((dx * dy * dz) as usize, __simd_lanes as usize, __coop_body);");
        self.line(2, "} } }");
        self.coop = false;
    }
    /// A `def f(...)` helper inside a kernel: an ordinary method over the kernel's fields.
    fn kernel_helper(&mut self, helper: &FnDecl) {
        let saved_locals = self.locals.clone();
        let saved_types = self.value_types.clone();
        for param in &helper.params {
            self.locals.insert(param.name.clone());
            if let Some(param_ty) = &param.ty {
                self.value_types.insert(param.name.clone(), ty(param_ty));
            }
        }
        crate::transpiler::helpers::collect_local_decl_names(&helper.body, &mut self.locals);
        let params = helper
            .params
            .iter()
            .map(|param| {
                format!(
                    "{}: {}",
                    param.name,
                    param.ty.as_ref().map(ty).unwrap_or_else(|| "i32".into())
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let ret = helper
            .return_ty
            .as_ref()
            .map(|ret| format!(" -> {}", ty(ret)))
            .unwrap_or_default();
        self.line(1, &format!("#[allow(unused_variables, unused_mut)] fn {}(&mut self, {params}){ret} {{", helper.name));
        let last = helper.body.len().saturating_sub(1);
        for (index, stmt) in helper.body.iter().enumerate() {
            match stmt {
                Stmt::Expr(value) if index == last && helper.return_ty.is_some() => {
                    let text = self.expr(value);
                    self.line(2, &text);
                }
                other => self.stmt(other, 2),
            }
        }
        self.line(1, "}");
        self.locals = saved_locals;
        self.value_types = saved_types;
    }
    fn emit_barrier_phases(&mut self, k: &KernelDecl, body: &[Stmt], indent: usize) {
        self.emit_barrier_phases_seeded(k, body, indent, &[]);
    }
    fn replayable_barrier_locals(&self, body: &[Stmt]) -> Vec<Stmt> {
        let mut names = std::collections::HashSet::new();
        let mut locals = Vec::new();
        for stmt in body {
            if let Stmt::Let(local) = stmt {
                if matches!(local.binding, BindingKind::Let)
                    && local
                        .value
                        .as_ref()
                        .is_some_and(|value| self.barrier_replayable(value, &names))
                {
                    names.insert(local.name.clone());
                    locals.push(stmt.clone());
                }
            }
        }
        locals
    }
    fn emit_following_collective_controls(
        &mut self,
        k: &KernelDecl,
        body: &[Stmt],
        indent: usize,
        initial_replayed: &[Stmt],
    ) {
        enum FollowingCollective<'a> {
            For(&'a ForStmt),
            If(&'a IfStmt),
            IfLet(&'a IfLetStmt),
            Match(&'a MatchStmt),
            Guard(&'a GuardStmt),
            While {
                stmt: &'a WhileStmt,
                declaration_index: usize,
                declaration: &'a LetStmt,
                update: &'a Expr,
            },
            DoWhile {
                stmt: &'a DoWhileStmt,
                declaration_index: usize,
                declaration: &'a LetStmt,
                update: &'a Expr,
            },
            WhileLet {
                stmt: &'a WhileLetStmt,
                declaration_index: usize,
                declaration: &'a LetStmt,
                update: &'a Expr,
            },
            Loop {
                stmt: &'a LoopStmt,
                declaration_index: usize,
                declaration: &'a LetStmt,
                update: &'a Expr,
                break_condition: &'a Expr,
            },
        }
        let next = body.iter().enumerate().find_map(|(index, stmt)| {
            let mut inherited_uniform = self.block_uniform_locals(initial_replayed);
            let prefix_uniform = self
                .block_uniform_local_statements_with_locals(
                    &body[..index],
                    &inherited_uniform,
                )
                .into_iter()
                .filter_map(|stmt| match stmt {
                    Stmt::Let(local) => Some(local.name),
                    _ => None,
                })
                .collect::<Vec<_>>();
            inherited_uniform.extend(prefix_uniform);
            if let Stmt::While(while_stmt) = stmt {
                if !while_stmt.body.iter().any(|stmt| {
                    is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                }) {
                    return None;
                }
                let Stmt::Expr(update) = while_stmt.body.last()? else { return None };
                let ExprKind::Assign(update_target, update_value) = &update.kind else { return None };
                let ExprKind::Var(update_name) = &update_target.kind else { return None };
                let declaration_index = body[..index].iter().position(|candidate| {
                    matches!(candidate, Stmt::Let(local)
                        if local.name == *update_name
                            && local.binding.is_mutable()
                            && local.value.as_ref().is_some_and(|value| {
                                self.block_uniform_expr_with_locals(value, &inherited_uniform)
                            }))
                })?;
                let safe_prefix = body[..index].iter().enumerate().all(|(prefix_index, stmt)| {
                    prefix_index == declaration_index
                        || (!matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
                            && !contains_nested_simd_barrier(stmt))
                });
                let Stmt::Let(declaration) = &body[declaration_index] else { unreachable!() };
                let mut uniform_locals = inherited_uniform;
                uniform_locals.insert(update_name.clone());
                if safe_prefix
                    && expr_mentions_var(&while_stmt.condition, update_name)
                    && self.block_uniform_expr_with_locals(&while_stmt.condition, &uniform_locals)
                    && self.block_uniform_expr_with_locals(update_value, &uniform_locals)
                {
                    return Some((index, FollowingCollective::While {
                        stmt: while_stmt,
                        declaration_index,
                        declaration,
                        update: update_value,
                    }));
                }
                return None;
            }
            if let Stmt::DoWhile(loop_stmt) = stmt {
                if !loop_stmt.body.iter().any(|stmt| {
                    is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                }) {
                    return None;
                }
                let Stmt::Expr(update) = loop_stmt.body.last()? else { return None };
                let ExprKind::Assign(update_target, update_value) = &update.kind else { return None };
                let ExprKind::Var(update_name) = &update_target.kind else { return None };
                let declaration_index = body[..index].iter().position(|candidate| {
                    matches!(candidate, Stmt::Let(local)
                        if local.name == *update_name
                            && local.binding.is_mutable()
                            && local.value.as_ref().is_some_and(|value| {
                                self.block_uniform_expr_with_locals(value, &inherited_uniform)
                            }))
                })?;
                let safe_prefix = body[..index].iter().enumerate().all(|(prefix_index, stmt)| {
                    prefix_index == declaration_index
                        || (!matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
                            && !contains_nested_simd_barrier(stmt))
                });
                let Stmt::Let(declaration) = &body[declaration_index] else { unreachable!() };
                let mut uniform_locals = inherited_uniform;
                uniform_locals.insert(update_name.clone());
                if safe_prefix
                    && expr_mentions_var(&loop_stmt.condition, update_name)
                    && self.block_uniform_expr_with_locals(&loop_stmt.condition, &uniform_locals)
                    && self.block_uniform_expr_with_locals(update_value, &uniform_locals)
                {
                    return Some((index, FollowingCollective::DoWhile {
                        stmt: loop_stmt,
                        declaration_index,
                        declaration,
                        update: update_value,
                    }));
                }
                return None;
            }
            if let Stmt::WhileLet(loop_stmt) = stmt {
                let bound_names = collective_while_let_bound_names(loop_stmt);
                if bound_names.is_empty()
                    || !loop_stmt.body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    })
                {
                    return None;
                }
                let ExprKind::Var(state_name) = &loop_stmt.value.kind else { return None };
                let Stmt::Expr(update) = loop_stmt.body.last()? else { return None };
                let ExprKind::Assign(update_target, update_value) = &update.kind else { return None };
                if !matches!(&update_target.kind, ExprKind::Var(name) if name == state_name) {
                    return None;
                }
                let declaration_index = body[..index].iter().position(|candidate| {
                    matches!(candidate, Stmt::Let(local)
                        if local.name == *state_name
                            && local.binding.is_mutable()
                            && local.value.as_ref().is_some_and(|value| {
                                self.block_uniform_expr_with_locals(value, &inherited_uniform)
                            }))
                })?;
                let safe_prefix = body[..index].iter().enumerate().all(|(prefix_index, stmt)| {
                    prefix_index == declaration_index
                        || (!matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
                            && !contains_nested_simd_barrier(stmt))
                });
                let Stmt::Let(declaration) = &body[declaration_index] else { unreachable!() };
                let mut uniform_locals = inherited_uniform;
                uniform_locals.insert(state_name.clone());
                uniform_locals.extend(bound_names.iter().cloned());
                if safe_prefix
                    && bound_names.iter().any(|name| expr_mentions_var(update_value, name))
                    && self.block_uniform_expr_with_locals(update_value, &uniform_locals)
                {
                    return Some((index, FollowingCollective::WhileLet {
                        stmt: loop_stmt,
                        declaration_index,
                        declaration,
                        update: update_value,
                    }));
                }
                return None;
            }
            if let Stmt::Loop(loop_stmt) = stmt {
                if !loop_stmt.body.iter().any(|stmt| {
                    is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                }) || loop_stmt.body.len() < 2
                {
                    return None;
                }
                let Stmt::Expr(update) = &loop_stmt.body[loop_stmt.body.len() - 2] else { return None };
                let ExprKind::Assign(update_target, update_value) = &update.kind else { return None };
                let ExprKind::Var(update_name) = &update_target.kind else { return None };
                let break_condition = canonical_collective_loop_break_condition(loop_stmt.body.last()?)?;
                let declaration_index = body[..index].iter().position(|candidate| {
                    matches!(candidate, Stmt::Let(local)
                        if local.name == *update_name
                            && local.binding.is_mutable()
                            && local.value.as_ref().is_some_and(|value| {
                                self.block_uniform_expr_with_locals(value, &inherited_uniform)
                            }))
                })?;
                let safe_prefix = body[..index].iter().enumerate().all(|(prefix_index, stmt)| {
                    prefix_index == declaration_index
                        || (!matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
                            && !contains_nested_simd_barrier(stmt))
                });
                let Stmt::Let(declaration) = &body[declaration_index] else { unreachable!() };
                let mut uniform_locals = inherited_uniform;
                uniform_locals.insert(update_name.clone());
                if safe_prefix
                    && expr_mentions_var(break_condition, update_name)
                    && self.block_uniform_expr_with_locals(update_value, &uniform_locals)
                    && self.block_uniform_expr_with_locals(break_condition, &uniform_locals)
                {
                    return Some((index, FollowingCollective::Loop {
                        stmt: loop_stmt,
                        declaration_index,
                        declaration,
                        update: update_value,
                        break_condition,
                    }));
                }
                return None;
            }
            let safe_prefix = body[..index].iter().all(|stmt| {
                !matches!(stmt, Stmt::Let(local) if local.binding.is_mutable())
            });
            if !safe_prefix {
                return None;
            }
            let uniform_locals = inherited_uniform;
            match stmt {
                Stmt::For(for_stmt) => (self.block_uniform_expr_with_locals(
                    &for_stmt.iterable,
                    &uniform_locals,
                )
                    && for_stmt.body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    }))
                .then_some((index, FollowingCollective::For(for_stmt))),
                Stmt::If(if_stmt) => {
                    let has_barrier = if_stmt
                        .branches
                        .iter()
                        .any(|(_, branch)| branch.iter().any(|stmt| {
                            is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                        }))
                        || if_stmt
                            .else_body
                            .as_ref()
                            .is_some_and(|branch| branch.iter().any(|stmt| {
                                is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                            }));
                    (has_barrier
                        && if_stmt
                            .branches
                            .iter()
                            .all(|(condition, _)| {
                                self.block_uniform_expr_with_locals(condition, &uniform_locals)
                            }))
                    .then_some((index, FollowingCollective::If(if_stmt)))
                }
                Stmt::IfLet(if_stmt) => {
                    let has_barrier = if_stmt.then_body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    })
                        || if_stmt
                            .elif_branches
                            .iter()
                            .any(|branch| branch.body.iter().any(|stmt| {
                                is_simd_barrier_stmt(stmt)
                                    || contains_nested_simd_barrier(stmt)
                            }))
                        || if_stmt
                            .else_body
                            .as_ref()
                            .is_some_and(|branch| branch.iter().any(|stmt| {
                                is_simd_barrier_stmt(stmt)
                                    || contains_nested_simd_barrier(stmt)
                            }));
                    (has_barrier
                        && self.block_uniform_guard_clauses_with_locals(
                            &if_stmt.clauses,
                            &uniform_locals,
                        )
                        && if_stmt
                            .elif_branches
                            .iter()
                            .all(|branch| {
                                self.block_uniform_guard_clauses_with_locals(
                                    &branch.clauses,
                                    &uniform_locals,
                                )
                            }))
                    .then_some((index, FollowingCollective::IfLet(if_stmt)))
                }
                Stmt::Match(match_stmt) => {
                    let has_barrier = match_stmt.arms.iter().any(|arm| {
                        matches!(&arm.body, MatchBody::Block(branch)
                            if branch.iter().any(|stmt| {
                                is_simd_barrier_stmt(stmt)
                                    || contains_nested_simd_barrier(stmt)
                            }))
                    });
                    (has_barrier
                        && self.block_uniform_expr_with_locals(
                            &match_stmt.subject,
                            &uniform_locals,
                        )
                        && match_stmt.arms.iter().all(|arm| {
                            self.block_uniform_match_arm_guard_with_locals(
                                arm,
                                &uniform_locals,
                            )
                        }))
                    .then_some((index, FollowingCollective::Match(match_stmt)))
                }
                Stmt::Guard(guard_stmt) => {
                    let uniform = match &guard_stmt.cond {
                        GuardCond::Expr(condition) => {
                            self.block_uniform_expr_with_locals(condition, &uniform_locals)
                        }
                        GuardCond::Clauses(clauses) => {
                            self.block_uniform_guard_clauses_with_locals(
                                clauses,
                                &uniform_locals,
                            )
                        }
                    };
                    (uniform && guard_stmt.else_body.iter().any(|stmt| {
                        is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt)
                    }))
                        .then_some((index, FollowingCollective::Guard(guard_stmt)))
                }
                _ => None,
            }
        });
        let Some((control_index, control)) = next else {
            self.emit_barrier_phases_seeded(k, body, indent, initial_replayed);
            return;
        };
        let excluded_declaration = match &control {
            FollowingCollective::While { declaration_index, .. }
            | FollowingCollective::DoWhile { declaration_index, .. }
            | FollowingCollective::WhileLet { declaration_index, .. }
            | FollowingCollective::Loop { declaration_index, .. } => Some(*declaration_index),
            _ => None,
        };
        let prefix = body[..control_index]
            .iter()
            .enumerate()
            .filter(|(index, _)| Some(*index) != excluded_declaration)
            .map(|(_, stmt)| stmt.clone())
            .collect::<Vec<_>>();
        if let FollowingCollective::While { declaration, .. }
            | FollowingCollective::DoWhile { declaration, .. }
            | FollowingCollective::WhileLet { declaration, .. }
            | FollowingCollective::Loop { declaration, .. } = &control {
            let uniform_name = &declaration.name;
            let uniform_ty = declaration.ty.as_ref().map(ty)
                .or_else(|| declaration.value.as_ref().and_then(|value| self.expr_type(value)))
                .unwrap_or_else(|| "i32".into());
            let initial = self.expr(declaration.value.as_ref().expect("collective while counter initializer was checked"));
            self.line(indent, &format!("let mut {uniform_name}: {uniform_ty} = {initial};"));
            self.value_types.insert(uniform_name.clone(), uniform_ty);
        }
        let mut replayed = initial_replayed.to_vec();
        let mut replayed_names = replayed
            .iter()
            .filter_map(|stmt| match stmt {
                Stmt::Let(local) => Some(local.name.clone()),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        for local in self.replayable_barrier_locals(&prefix) {
            let Stmt::Let(declaration) = &local else { unreachable!() };
            if replayed_names.insert(declaration.name.clone()) {
                replayed.push(local);
            }
        }
        for local in self.block_uniform_local_statements_with_locals(&prefix, &replayed_names) {
            self.stmt(&local, indent);
        }
        self.emit_barrier_phases_seeded(k, &prefix, indent, initial_replayed);
        match control {
            FollowingCollective::For(for_stmt) => {
                let iterable = self.expr(&for_stmt.iterable);
                self.line(
                    indent,
                    &format!("for {} in {} {{", for_stmt.vars.join(", "), iterable),
                );
                let inherited_loop_vars = for_stmt
                    .vars
                    .iter()
                    .filter(|name| self.collective_uniform_locals.contains(*name))
                    .cloned()
                    .collect::<Vec<_>>();
                self.collective_uniform_locals.extend(for_stmt.vars.iter().cloned());
                self.emit_following_collective_controls(
                    k,
                    &for_stmt.body,
                    indent + 1,
                    &replayed,
                );
                for name in &for_stmt.vars {
                    if !inherited_loop_vars.contains(name) {
                        self.collective_uniform_locals.remove(name);
                    }
                }
                self.line(indent, "}");
            }
            FollowingCollective::If(if_stmt) => {
                for (branch_index, (condition, branch)) in if_stmt.branches.iter().enumerate() {
                    let condition = self.expr(condition);
                    self.line(
                        indent,
                        &format!(
                            "{}if {condition} {{",
                            if branch_index == 0 { "" } else { "else " }
                        ),
                    );
                    self.emit_following_collective_controls(k, branch, indent + 1, &replayed);
                    self.line(indent, "}");
                }
                if let Some(branch) = &if_stmt.else_body {
                    self.line(indent, "else {");
                    self.emit_following_collective_controls(k, branch, indent + 1, &replayed);
                    self.line(indent, "}");
                }
            }
            FollowingCollective::IfLet(if_stmt) => self.emit_collective_if_let_clauses(
                k,
                &if_stmt.clauses,
                0,
                &if_stmt.then_body,
                &if_stmt.elif_branches,
                if_stmt.else_body.as_deref(),
                indent,
                &replayed,
            ),
            FollowingCollective::Match(match_stmt) => {
                let subject_is_string =
                    self.expr_type(&match_stmt.subject).as_deref() == Some("String");
                let subject = self.expr(&match_stmt.subject);
                let subject = if subject_is_string {
                    format!("({subject}).as_str()")
                } else {
                    subject
                };
                self.line(indent, &format!("match {subject} {{"));
                for arm in &match_stmt.arms {
                    let patterns = arm
                        .patterns
                        .iter()
                        .map(simd_pattern)
                        .collect::<Vec<_>>()
                        .join(" | ");
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|guard| format!(" if {}", self.expr(guard)))
                        .unwrap_or_default();
                    self.line(indent + 1, &format!("{patterns}{guard} => {{"));
                    let mut bound_names = std::collections::HashSet::new();
                    for pattern in &arm.patterns {
                        collect_simd_pattern_bind_names(pattern, &mut bound_names);
                    }
                    let inherited_bound_names = bound_names
                        .iter()
                        .filter(|name| self.collective_uniform_locals.contains(*name))
                        .cloned()
                        .collect::<std::collections::HashSet<_>>();
                    self.collective_uniform_locals.extend(bound_names.iter().cloned());
                    match &arm.body {
                        MatchBody::Block(branch) => self.emit_following_collective_controls(
                            k,
                            branch,
                            indent + 2,
                            &replayed,
                        ),
                        MatchBody::Expr(value) => self.emit_barrier_phases_seeded(
                            k,
                            &[Stmt::Expr(value.clone())],
                            indent + 2,
                            &replayed,
                        ),
                    }
                    for name in bound_names {
                        if !inherited_bound_names.contains(&name) {
                            self.collective_uniform_locals.remove(&name);
                        }
                    }
                    self.line(indent + 1, "},");
                }
                self.line(indent, "}");
            }
            FollowingCollective::Guard(guard_stmt) => match &guard_stmt.cond {
                GuardCond::Expr(condition) => {
                    let condition = self.expr(condition);
                    self.line(indent, &format!("if !({condition}) {{"));
                    self.emit_following_collective_controls(
                        k,
                        &guard_stmt.else_body,
                        indent + 1,
                        &replayed,
                    );
                    self.line(indent, "}");
                }
                GuardCond::Clauses(clauses) => {
                    self.emit_collective_guard_clauses(
                        k,
                        clauses,
                        0,
                        &guard_stmt.else_body,
                        &body[control_index + 1..],
                        indent,
                        &replayed,
                    );
                    return;
                }
            },
            FollowingCollective::While { stmt, declaration, update, .. } => {
                let condition = self.expr(&stmt.condition);
                self.line(indent, &format!("while {condition} {{"));
                let inserted_uniform = self
                    .collective_uniform_locals
                    .insert(declaration.name.clone());
                self.emit_following_collective_controls(
                    k,
                    &stmt.body[..stmt.body.len() - 1],
                    indent + 1,
                    &replayed,
                );
                if inserted_uniform {
                    self.collective_uniform_locals.remove(&declaration.name);
                }
                let update = self.expr(update);
                self.line(indent + 1, &format!("{} = {update};", declaration.name));
                self.line(indent, "}");
            }
            FollowingCollective::DoWhile { stmt, declaration, update, .. } => {
                self.line(indent, "loop {");
                let inserted_uniform = self.collective_uniform_locals.insert(declaration.name.clone());
                self.emit_following_collective_controls(
                    k,
                    &stmt.body[..stmt.body.len() - 1],
                    indent + 1,
                    &replayed,
                );
                if inserted_uniform { self.collective_uniform_locals.remove(&declaration.name); }
                let update = self.expr(update);
                self.line(indent + 1, &format!("{} = {update};", declaration.name));
                let condition = self.expr(&stmt.condition);
                self.line(indent + 1, &format!("if !({condition}) {{ break; }}"));
                self.line(indent, "}");
            }
            FollowingCollective::WhileLet { stmt, declaration, update, .. } => {
                let state_name = &declaration.name;
                let match_pattern = simd_collective_while_let_pattern(stmt);
                self.line(
                    indent,
                    &format!("while let {match_pattern} = {state_name} {{"),
                );
                let inserted_state = self.collective_uniform_locals.insert(state_name.clone());
                let bound_names = collective_while_let_bound_names(stmt);
                let inherited_bound_names = bound_names.iter()
                    .filter(|name| self.collective_uniform_locals.contains(*name))
                    .cloned().collect::<std::collections::HashSet<_>>();
                self.collective_uniform_locals.extend(bound_names.iter().cloned());
                self.emit_following_collective_controls(
                    k,
                    &stmt.body[..stmt.body.len() - 1],
                    indent + 1,
                    &replayed,
                );
                for name in bound_names {
                    if !inherited_bound_names.contains(&name) {
                        self.collective_uniform_locals.remove(&name);
                    }
                }
                if inserted_state { self.collective_uniform_locals.remove(state_name); }
                let update = self.expr(update);
                self.line(indent + 1, &format!("{state_name} = {update};"));
                self.line(indent, "}");
            }
            FollowingCollective::Loop {
                stmt,
                declaration,
                update,
                break_condition,
                ..
            } => {
                let uniform_name = &declaration.name;
                self.line(indent, "loop {");
                let inserted_uniform = self.collective_uniform_locals.insert(uniform_name.clone());
                self.emit_following_collective_controls(
                    k,
                    &stmt.body[..stmt.body.len() - 2],
                    indent + 1,
                    &replayed,
                );
                if inserted_uniform { self.collective_uniform_locals.remove(uniform_name); }
                let update = self.expr(update);
                self.line(indent + 1, &format!("{uniform_name} = {update};"));
                let break_condition = self.expr(break_condition);
                self.line(indent + 1, &format!("if {break_condition} {{ break; }}"));
                self.line(indent, "}");
            }
        }
        self.emit_following_collective_controls(
            k,
            &body[control_index + 1..],
            indent,
            &replayed,
        );
    }
    fn emit_barrier_phases_seeded(
        &mut self,
        k: &KernelDecl,
        body: &[Stmt],
        indent: usize,
        initial_replayed_locals: &[Stmt],
    ) {
        let phases = body
            .split(is_simd_barrier_stmt)
            .collect::<Vec<_>>();
        let mut carried_mutable_locals = Vec::new();
        for (declaration_phase, phase) in phases.iter().enumerate() {
            for stmt in *phase {
                if let Stmt::Let(local) = stmt {
                    if local.binding.is_mutable() {
                        if let Some(local_ty) = local.ty.as_ref().map(ty).or_else(|| {
                            local.value.as_ref().and_then(|value| self.expr_type(value))
                        }) {
                            self.line(
                                indent,
                                &format!(
                                    "let mut __boring_barrier_{}: Vec<{}> = vec![Default::default(); (dx * dy * dz) as usize];",
                                    local.name, local_ty
                                ),
                            );
                            carried_mutable_locals.push((
                                local.name.clone(),
                                local_ty,
                                declaration_phase,
                            ));
                        }
                    }
                }
            }
        }
        let mut replayed_phase_locals = initial_replayed_locals.to_vec();
        let mut replayed_phase_local_names = initial_replayed_locals
            .iter()
            .filter_map(|stmt| match stmt {
                Stmt::Let(local) => Some(local.name.clone()),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        for (phase_index, phase) in phases.into_iter().enumerate() {
            self.line(indent, "for tz in 0..dz as usize { for ty in 0..dy as usize { '__boring_work_item: for __x in 0..dx as usize {");
            self.line(indent + 1, "let __thread = __BoringDim3 { x: __x as i32, y: ty as i32, z: tz as i32 };");
            self.line(indent + 1, "let __boring_barrier_thread = __x + dx as usize * (ty + dy as usize * tz);");
            self.line(indent + 1, "if !__boring_barrier_active[__boring_barrier_thread] { continue '__boring_work_item; }");
            for field in k.fields.iter().filter(|f| matches!(f.qual, GpuQual::Local)) {
                self.line(
                    indent + 1,
                    &format!(
                        "let mut {}: {} = __boring_local_{}[__boring_barrier_thread].clone();",
                        field.name,
                        ty(&field.ty),
                        field.name
                    ),
                );
            }
            self.work_item_return_uses_active_mask = Some(true);
            if phase_index > 0 || !initial_replayed_locals.is_empty() {
                for local in &replayed_phase_locals {
                    self.stmt(local, indent + 1);
                }
            }
            for (name, local_ty, declaration_phase) in &carried_mutable_locals {
                if *declaration_phase < phase_index {
                    self.line(
                        indent + 1,
                        &format!(
                            "let mut {name}: {local_ty} = __boring_barrier_{name}[__boring_barrier_thread].clone();"
                        ),
                    );
                }
            }
            for stmt in phase {
                self.stmt(stmt, indent + 1);
            }
            self.work_item_return_uses_active_mask = None;
            for field in k.fields.iter().filter(|f| matches!(f.qual, GpuQual::Local)) {
                self.line(
                    indent + 1,
                    &format!(
                        "__boring_local_{}[__boring_barrier_thread] = {}.clone();",
                        field.name, field.name
                    ),
                );
            }
            for (name, _, declaration_phase) in &carried_mutable_locals {
                if *declaration_phase <= phase_index {
                    self.line(
                        indent + 1,
                        &format!(
                            "__boring_barrier_{name}[__boring_barrier_thread] = {name}.clone();"
                        ),
                    );
                }
            }
            for stmt in phase {
                if let Stmt::Let(local) = stmt {
                    if matches!(local.binding, BindingKind::Let)
                        && local.value.as_ref().is_some_and(|value| {
                            self.barrier_replayable(value, &replayed_phase_local_names)
                        })
                    {
                        replayed_phase_local_names.insert(local.name.clone());
                        replayed_phase_locals.push(stmt.clone());
                    }
                }
            }
            self.line(indent, "} } }");
        }
    }
    fn stmt(&mut self, s: &Stmt, n: usize) {
        match s {
            Stmt::Let(v) => {
                let inferred_ty =
                    v.ty.as_ref()
                        .map(ty)
                        .or_else(|| v.value.as_ref().and_then(|value| self.expr_type(value)));
                let optional_destination =
                    v.ty.as_ref().is_some_and(|t| matches!(base(t), Type::Optional(_)));
                let value = v
                    .value
                    .as_ref()
                    .map(|x| {
                        let wrapped;
                        let x = if optional_destination {
                            wrapped = self.implicit_some(x);
                            &wrapped
                        } else {
                            x
                        };
                        // `[T?]` literal: wrap each element, as for a scalar `T?` destination.
                        let array_wrapped;
                        let x = match (v.ty.as_ref().map(base), &x.kind) {
                            (Some(Type::Array(elem)), ExprKind::Array(elements))
                                if matches!(base(elem), Type::Optional(_)) =>
                            {
                                array_wrapped = Expr {
                                    kind: ExprKind::Array(
                                        elements.iter().map(|element| self.implicit_some(element)).collect(),
                                    ),
                                    ..x.clone()
                                };
                                &array_wrapped
                            }
                            _ => x,
                        };
                        let value = self.expr(x);
                        if matches!(&x.kind, ExprKind::Var(source)
                            if self.field_axes.contains_key(source)
                                || self.local_axes.contains_key(source))
                        {
                            format!("{value}.clone()")
                        } else {
                            value
                        }
                    })
                    .unwrap_or_else(|| "Default::default()".into());
                let value = match &v.ty {
                    Some(t) if simd_needs_as_cast(t) => format!("({value}) as {}", ty(t)),
                    _ => value,
                };
                self.line(
                    n,
                    &format!(
                        "let {}{}{} = {};",
                        if v.binding.is_mutable() { "mut " } else { "" },
                        v.name,
                        v.ty.as_ref()
                            .map(|t| format!(": {}", ty(t)))
                            .unwrap_or_default(),
                        value
                    ),
                );
                if let Some(inferred_ty) = inferred_ty {
                    self.value_types.insert(v.name.clone(), inferred_ty);
                } else if v.value.as_ref().is_some_and(|value| self.looks_optional(value)) {
                    self.value_types.insert(v.name.clone(), "Option<_>".into());
                }
                if let Some(arity) = v
                    .ty
                    .as_ref()
                    .and_then(tuple_type_arity)
                    .or_else(|| v.value.as_ref().and_then(|value| self.tuple_arity(value)))
                {
                    self.tuple_arities.insert(v.name.clone(), arity);
                }
                if let Some((_, axes)) = v.ty.as_ref().and_then(Type::as_labeled_array) {
                    self.local_axes.insert(v.name.clone(), axes.to_vec());
                } else if let Some(Expr {
                    kind: ExprKind::Var(source),
                    ..
                }) = &v.value
                {
                    if let Some(axes) = self
                        .field_axes
                        .get(source)
                        .or_else(|| self.local_axes.get(source))
                        .cloned()
                    {
                        self.local_axes.insert(v.name.clone(), axes);
                    }
                }
                let collection_initializer = v.value.as_ref().is_some_and(|value| {
                    matches!(
                        &value.kind,
                        ExprKind::Array(_)
                            | ExprKind::ArrayFill { .. }
                            | ExprKind::ArrayAlloc { .. }
                            | ExprKind::ArrayComp { .. }
                            | ExprKind::ArrayCompIter { .. }
                            | ExprKind::LabeledArrayComp { .. }
                    ) || matches!(
                        &value.kind,
                        ExprKind::MethodCall(_, method, _)
                            if matches!(
                                method.as_str(),
                                "map"
                                    | "filter"
                                    | "keys"
                                    | "values"
                                    | "toArray"
                                    | "reversed"
                                    | "take"
                                    | "drop"
                                    | "slice"
                                    | "flatMap"
                                    | "flat"
                                    | "sorted"
                                    | "sortedBy"
                                    | "zip"
                                    | "enumerate"
                                    | "removeAt"
                                    | "split"
                                    | "chars"
                            ) && !matches!(&value.kind,
                                ExprKind::MethodCall(receiver, method, _)
                                    if matches!(method.as_str(), "map" | "filter")
                                        && matches!(&receiver.kind, ExprKind::Var(source)
                                            if self.dicts.contains(source)))
                    ) || matches!(
                        &value.kind,
                        ExprKind::Pipe(_, method, _)
                            if matches!(
                                method.as_str(),
                                "map"
                                    | "filter"
                                    | "keys"
                                    | "values"
                                    | "toArray"
                                    | "reversed"
                                    | "take"
                                    | "drop"
                                    | "slice"
                                    | "flatMap"
                                    | "flat"
                                    | "sorted"
                                    | "sortedBy"
                                    | "zip"
                                    | "enumerate"
                                    | "removeAt"
                                    | "split"
                                    | "chars"
                            )
                    )
                });
                if v.ty.as_ref().is_some_and(is_array)
                    || collection_initializer
                    || self.local_axes.contains_key(&v.name)
                {
                    self.arrays.insert(v.name.clone());
                }
                if v.ty.as_ref().is_some_and(|ty| matches!(base(ty), Type::Dict(_, _)))
                    || matches!(v.value.as_ref().map(|value| &value.kind), Some(ExprKind::Dict(_)))
                    || matches!(v.value.as_ref().map(|value| &value.kind),
                        Some(ExprKind::Var(source)) if self.dicts.contains(source))
                    || matches!(v.value.as_ref().map(|value| &value.kind),
                        Some(ExprKind::MethodCall(receiver, method, _))
                            if method == "removeAt"
                                && matches!(&receiver.kind, ExprKind::Var(source) if self.dicts.contains(source)))
                    || matches!(v.value.as_ref().map(|value| &value.kind),
                        Some(ExprKind::MethodCall(receiver, method, _))
                            if matches!(method.as_str(), "map" | "filter")
                                && matches!(&receiver.kind, ExprKind::Var(source) if self.dicts.contains(source)))
                {
                    self.dicts.insert(v.name.clone());
                }
                if v.ty.as_ref().is_some_and(|ty| matches!(base(ty), Type::Set(_)))
                    || matches!(v.value.as_ref().map(|value| &value.kind), Some(ExprKind::Set(_)))
                    || matches!(v.value.as_ref().map(|value| &value.kind),
                        Some(ExprKind::Var(source)) if self.sets.contains(source))
                    || matches!(v.value.as_ref().map(|value| &value.kind),
                        Some(ExprKind::MethodCall(_, method, _))
                            if matches!(method.as_str(), "union" | "intersection" | "difference"))
                    || matches!(v.value.as_ref().map(|value| &value.kind),
                        Some(ExprKind::MethodCall(receiver, method, _))
                            if method == "removeAt"
                                && matches!(&receiver.kind, ExprKind::Var(source) if self.sets.contains(source)))
                {
                    self.sets.insert(v.name.clone());
                }
                if let Some(initializer) = v.value.as_ref().filter(|e| self.warp_replayable(e)) {
                    self.replayable_locals
                        .insert(v.name.clone(), initializer.clone());
                }
                self.replayed_assignments.remove(&v.name);
            }
            Stmt::LetDestructure(v) => {
                let bindings = v
                    .bindings
                    .iter()
                    .map(|binding| {
                        if binding.name == "_" {
                            "_".into()
                        } else {
                            format!(
                                "{}{}",
                                if v.binding.is_mutable() { "mut " } else { "" },
                                binding.name
                            )
                        }
                    })
                    .collect::<Vec<String>>()
                    .join(", ");
                let value = self.expr(&v.value);
                self.line(n, &format!("let ({bindings}) = {value};"));
                for (index, binding) in v.bindings.iter().enumerate() {
                    if binding.name == "_" {
                        continue;
                    }
                    if let Some(binding_ty) = &binding.ty {
                        self.value_types
                            .insert(binding.name.clone(), ty(binding_ty));
                        if let Some(arity) = tuple_type_arity(binding_ty) {
                            self.tuple_arities.insert(binding.name.clone(), arity);
                        }
                    } else if let ExprKind::Tuple(elements) = &v.value.kind {
                        if let Some(arity) = elements
                            .get(index)
                            .and_then(|element| self.tuple_arity(element))
                        {
                            self.tuple_arities.insert(binding.name.clone(), arity);
                        }
                    }
                    self.replayable_locals.remove(&binding.name);
                    self.replayed_assignments.remove(&binding.name);
                }
            }
            Stmt::Expr(x) => {
                let x = self.expr(x);
                self.line(n, &format!("{};", x));
            }
            Stmt::If(i) => {
                self.control_flow_depth += 1;
                for (j, (condition, body)) in i.branches.iter().enumerate() {
                    let condition = self.expr(condition);
                    self.line(
                        n,
                        &format!("{}if {} {{", if j == 0 { "" } else { "else " }, condition),
                    );
                    for s in body {
                        self.stmt(s, n + 1);
                    }
                    self.line(n, "}");
                }
                if let Some(body) = &i.else_body {
                    self.line(n, "else {");
                    for s in body {
                        self.stmt(s, n + 1);
                    }
                    self.line(n, "}");
                }
                self.control_flow_depth -= 1;
            }
            Stmt::IfLet(i) => {
                self.control_flow_depth += 1;
                self.emit_if_let_branch(
                    &i.clauses,
                    0,
                    &i.then_body,
                    &i.elif_branches,
                    i.else_body.as_deref(),
                    n,
                );
                self.control_flow_depth -= 1;
            }
            Stmt::Match(m) => {
                let subject = self.expr(&m.subject);
                self.line(n, &format!("match {subject} {{"));
                self.control_flow_depth += 1;
                for arm in &m.arms {
                    let patterns = arm
                        .patterns
                        .iter()
                        .map(simd_pattern)
                        .collect::<Vec<_>>()
                        .join(" | ");
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|guard| format!(" if {}", self.expr(guard)))
                        .unwrap_or_default();
                    match &arm.body {
                        MatchBody::Expr(value) => {
                            let value = self.expr(value);
                            self.line(n + 1, &format!("{patterns}{guard} => {value},"));
                        }
                        MatchBody::Block(body) => {
                            self.line(n + 1, &format!("{patterns}{guard} => {{"));
                            for stmt in body {
                                self.stmt(stmt, n + 2);
                            }
                            self.line(n + 1, "},");
                        }
                    }
                }
                self.control_flow_depth -= 1;
                self.line(n, "}");
            }
            Stmt::Guard(g) => {
                self.control_flow_depth += 1;
                match &g.cond {
                    GuardCond::Expr(condition) => {
                        let condition = self.expr(condition);
                        self.line(n, &format!("if !({condition}) {{"));
                        for stmt in &g.else_body {
                            self.stmt(stmt, n + 1);
                        }
                        self.line(n, "}");
                    }
                    GuardCond::Clauses(clauses) => {
                        for clause in clauses {
                            match clause {
                                CondClause::Let(name, value) => {
                                    let value = self.expr(value);
                                    self.line(n, &format!("let Some({name}) = {value} else {{"));
                                }
                                CondClause::LetPat(pattern, value) => {
                                    let value = self.expr(value);
                                    self.line(
                                        n,
                                        &format!(
                                            "let {} = {value} else {{",
                                            simd_pattern(pattern)
                                        ),
                                    );
                                }
                                CondClause::Expr(condition) => {
                                    let condition = self.expr(condition);
                                    self.line(n, &format!("if !({condition}) {{"));
                                }
                            }
                            for stmt in &g.else_body {
                                self.stmt(stmt, n + 1);
                            }
                            let ending = if matches!(
                                clause,
                                CondClause::Let(..) | CondClause::LetPat(..)
                            ) {
                                "};"
                            } else {
                                "}"
                            };
                            self.line(n, ending);
                            let bound_tuple = match clause {
                                CondClause::Let(name, value) => self
                                    .optional_tuple_arity(value)
                                    .map(|arity| (name.clone(), arity)),
                                CondClause::LetPat(pattern, value) => {
                                    tuple_optional_pattern_binding(pattern)
                                        .zip(self.optional_tuple_arity(value))
                                        .map(|(name, arity)| (name.to_string(), arity))
                                }
                                CondClause::Expr(_) => None,
                            };
                            if let Some((name, arity)) = bound_tuple {
                                self.tuple_arities.insert(name, arity);
                            }
                        }
                    }
                }
                self.control_flow_depth -= 1;
            }
            Stmt::For(f) => {
                let iterable = self.expr(&f.iterable);
                self.line(n, &format!("for {} in {} {{", f.vars.join(", "), iterable));
                let previous_break_sync = self.while_let_break_sync.take();
                self.control_flow_depth += 1;
                for s in &f.body {
                    self.stmt(s, n + 1);
                }
                self.control_flow_depth -= 1;
                self.while_let_break_sync = previous_break_sync;
                self.line(n, "}");
            }
            Stmt::Comment(c) if c == "sync" && self.coop => {
                self.line(n, "__coop::sync();");
            }
            Stmt::Comment(c) if c == "sync" => {
                self.errors.push(crate::errors::SourceError::at_line(
                    "`sync` is not yet supported inside a --target simd kernel",
                    0,
                ));
                self.line(n, "/* rejected unsupported block barrier */");
            }
            Stmt::Comment(c) => self.line(n, &format!("//{}", c)),
            Stmt::While(w) => {
                let condition = self.expr(&w.condition);
                self.line(n, &format!("while {} {{", condition));
                let previous_break_sync = self.while_let_break_sync.take();
                self.control_flow_depth += 1;
                for s in &w.body {
                    self.stmt(s, n + 1);
                }
                self.control_flow_depth -= 1;
                self.while_let_break_sync = previous_break_sync;
                self.line(n, "}");
            }
            Stmt::WhileLet(w) => {
                let self_shorthand = w.pattern.is_none()
                    && matches!(&w.value.kind, ExprKind::Var(name) if name == &w.name);
                if self_shorthand {
                    let outer = format!("__boring_while_let_{}", w.name);
                    self.line(n, "{");
                    self.line(n + 1, &format!("let mut {outer} = {};", w.name));
                    self.line(n + 1, &format!("while let Some({}) = {outer} {{", w.name));
                    let previous_redirect = self
                        .while_let_redirect
                        .insert(w.name.clone(), outer.clone());
                    let previous_break_sync = self
                        .while_let_break_sync
                        .replace((w.name.clone(), outer.clone()));
                    self.control_flow_depth += 1;
                    for stmt in &w.body {
                        self.stmt(stmt, n + 2);
                    }
                    self.control_flow_depth -= 1;
                    self.while_let_break_sync = previous_break_sync;
                    if let Some(previous) = previous_redirect {
                        self.while_let_redirect.insert(w.name.clone(), previous);
                    } else {
                        self.while_let_redirect.remove(&w.name);
                    }
                    self.line(n + 1, "}");
                    self.line(n + 1, &format!("{} = {outer};", w.name));
                    self.line(n, "}");
                    return;
                }
                let value = self.expr(&w.value);
                let pattern = w
                    .pattern
                    .as_ref()
                    .map(simd_pattern)
                    .unwrap_or_else(|| format!("Some({})", w.name));
                self.line(n, &format!("while let {pattern} = {value} {{"));
                let bound_tuple = self.optional_tuple_arity(&w.value).map(|arity| {
                    let name = w
                        .pattern
                        .as_ref()
                        .and_then(tuple_optional_pattern_binding)
                        .unwrap_or(&w.name);
                    (name.to_string(), arity)
                });
                let previous_tuple_arity = bound_tuple.as_ref().and_then(|(name, arity)| {
                    self.tuple_arities.insert(name.clone(), *arity)
                });
                let previous_break_sync = self.while_let_break_sync.take();
                self.control_flow_depth += 1;
                for stmt in &w.body {
                    self.stmt(stmt, n + 1);
                }
                self.control_flow_depth -= 1;
                self.while_let_break_sync = previous_break_sync;
                if let Some((name, _)) = &bound_tuple {
                    if let Some(previous) = previous_tuple_arity {
                        self.tuple_arities.insert(name.clone(), previous);
                    } else {
                        self.tuple_arities.remove(name);
                    }
                }
                self.line(n, "}");
            }
            Stmt::DoWhile(w) => {
                self.line(n, "loop {");
                let previous_break_sync = self.while_let_break_sync.take();
                self.control_flow_depth += 1;
                for stmt in &w.body {
                    self.stmt(stmt, n + 1);
                }
                let condition = self.expr(&w.condition);
                self.line(n + 1, &format!("if !({condition}) {{ break; }}"));
                self.control_flow_depth -= 1;
                self.while_let_break_sync = previous_break_sync;
                self.line(n, "}");
            }
            Stmt::Loop(l) => {
                self.line(n, "loop {");
                let previous_break_sync = self.while_let_break_sync.take();
                self.control_flow_depth += 1;
                for s in &l.body {
                    self.stmt(s, n + 1);
                }
                self.control_flow_depth -= 1;
                self.while_let_break_sync = previous_break_sync;
                self.line(n, "}");
            }
            Stmt::Break(_, value) => {
                if let Some((name, outer)) = self.while_let_break_sync.clone() {
                    self.line(n, &format!("{outer} = Some({name});"));
                }
                if self.value_loop_depth > 0 {
                    if let Some(value) = value {
                        let value = self.expr(value);
                        self.line(n, &format!("break {value};"));
                    } else {
                        self.line(n, "break;");
                    }
                } else {
                    self.line(n, "break;");
                }
            }
            Stmt::Continue(_) => {
                if let Some((name, outer)) = self.while_let_break_sync.clone() {
                    self.line(n, &format!("{outer} = Some({name});"));
                }
                self.line(n, "continue;");
            }
            Stmt::Return(r) => {
                if r.value.is_some() {
                    self.errors.push(crate::errors::SourceError::at(
                        "a SIMD kernel cannot return a value",
                        r.line,
                        r.col,
                    ));
                }
                if self.work_item_return_uses_active_mask == Some(true) {
                    self.line(
                        n,
                        "__boring_barrier_active[__boring_barrier_thread] = false;",
                    );
                }
                self.line(n, "continue '__boring_work_item;");
            }
            other => {
                let (line, col) = stmt_position(other);
                self.errors.push(crate::errors::SourceError::at(
                    "this statement is not yet supported inside a --target simd kernel",
                    line,
                    col,
                ));
                self.line(n, "/* rejected unsupported SIMD kernel statement */");
            }
        }
    }
    fn emit_if_let_branch(
        &mut self,
        clauses: &[CondClause],
        clause_index: usize,
        body: &[Stmt],
        remaining: &[IfLetBranch],
        else_body: Option<&[Stmt]>,
        n: usize,
    ) {
        let Some(clause) = clauses.get(clause_index) else {
            for stmt in body {
                self.stmt(stmt, n);
            }
            return;
        };
        let condition = match clause {
            CondClause::Let(name, value) => {
                let value = self.expr(value);
                format!("let Some({name}) = {value}")
            }
            CondClause::LetPat(pattern, value) => {
                let value = self.expr(value);
                format!("let {} = {value}", simd_pattern(pattern))
            }
            CondClause::Expr(value) => self.expr(value),
        };
        let bound_tuple = match clause {
            CondClause::Let(name, value) => self
                .optional_tuple_arity(value)
                .map(|arity| (name.clone(), arity)),
            CondClause::LetPat(pattern, value) => tuple_optional_pattern_binding(pattern)
                .zip(self.optional_tuple_arity(value))
                .map(|(name, arity)| (name.to_string(), arity)),
            _ => None,
        };
        let previous_tuple_arity = bound_tuple.as_ref().and_then(|(name, arity)| {
            self.tuple_arities.insert(name.clone(), *arity)
        });
        self.line(n, &format!("if {condition} {{"));
        self.emit_if_let_branch(
            clauses,
            clause_index + 1,
            body,
            remaining,
            else_body,
            n + 1,
        );
        if let Some((name, _)) = &bound_tuple {
            if let Some(previous) = previous_tuple_arity {
                self.tuple_arities.insert(name.clone(), previous);
            } else {
                self.tuple_arities.remove(name);
            }
        }
        self.line(n, "} else {");
        self.emit_if_let_fallback(remaining, else_body, n + 1);
        self.line(n, "}");
    }
    fn emit_if_let_fallback(
        &mut self,
        remaining: &[IfLetBranch],
        else_body: Option<&[Stmt]>,
        n: usize,
    ) {
        if let Some((branch, tail)) = remaining.split_first() {
            self.emit_if_let_branch(&branch.clauses, 0, &branch.body, tail, else_body, n);
        } else if let Some(body) = else_body {
            for stmt in body {
                self.stmt(stmt, n);
            }
        }
    }
    fn expr(&mut self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Int(v) => v.to_string(),
            ExprKind::UInt64(v) => format!("{v}u64"),
            ExprKind::Float(v) => float_literal(*v),
            ExprKind::Str(value) => format!("String::from({value:?})"),
            ExprKind::StringInterp(segments) => {
                let mut format_string = String::new();
                let mut arguments = Vec::new();
                for segment in segments {
                    match segment {
                        StringSegment::Lit(value) => {
                            format_string.push_str(&value.replace('{', "{{").replace('}', "}}"));
                        }
                        StringSegment::Expr(value) => {
                            format_string.push_str("{}");
                            arguments.push(self.expr(value));
                        }
                        StringSegment::FormattedExpr(value, spec) => {
                            let spec = spec.trim_end_matches(['f', 'd', 's', 'g', 'G']);
                            format_string.push_str(&format!("{{:{spec}}}"));
                            arguments.push(self.expr(value));
                        }
                    }
                }
                let arguments = arguments
                    .into_iter()
                    .map(|argument| format!(", {argument}"))
                    .collect::<String>();
                format!("format!({format_string:?}{arguments})")
            }
            ExprKind::Bool(v) => v.to_string(),
            ExprKind::Nil => "None".into(),
            ExprKind::Void => "()".into(),
            ExprKind::Var(v) => {
                if self.thread_override.is_some() {
                    if let Some(replayed) = self.replayed_assignments.get(v) {
                        return replayed.clone();
                    }
                    if let Some(initializer) = self.replayable_locals.get(v).cloned() {
                        return self.expr(&initializer);
                    }
                }
                if v == "thread" {
                    self.thread_override
                        .clone()
                        .unwrap_or_else(|| "__thread".into())
                } else if matches!(v.as_str(), "block" | "block_dim") {
                    format!("__{v}")
                } else if v == "warp" {
                    "__warp".into()
                } else if self.fields.contains(v) {
                    format!("self.{v}")
                } else if let Some(const_ty) = self
                    .global_consts
                    .get(v)
                    .filter(|_| !self.locals.contains(v))
                {
                    format!("({} as {const_ty})", v.to_uppercase())
                } else {
                    v.clone()
                }
            }
            ExprKind::DotIdent(v) if v == "thread" => self
                .thread_override
                .clone()
                .unwrap_or_else(|| "__thread".into()),
            ExprKind::DotIdent(v) if matches!(v.as_str(), "block" | "block_dim") => {
                format!("__{v}")
            }
            ExprKind::Field(a, b) => {
                if let ExprKind::Var(name) = &a.kind {
                    if self.arrays.contains(name)
                        || self.dicts.contains(name)
                        || self.sets.contains(name)
                    {
                        let array = self.expr(a);
                        match b.as_str() {
                            "length" | "count" => return format!("({array}.len() as i32)"),
                            "isEmpty" => return format!("{array}.is_empty()"),
                            _ => {}
                        }
                    }
                }
                if let ExprKind::Var(field) = &a.kind {
                    if let Some(axes) = self
                        .field_axes
                        .get(field)
                        .or_else(|| self.local_axes.get(field))
                    {
                        if let Some(axis) = axes.iter().find(|axis| axis.label == *b) {
                            if let Some(size) = &axis.size {
                                return crate::transpiler::helpers::const_expr_to_c_like(&size.0);
                            }
                        }
                    }
                }
                if let ExprKind::Field(root, base) = &a.kind {
                    if matches!(&root.kind,ExprKind::Var(v) if v=="gpu") {
                        let thread_name = self
                            .thread_override
                            .clone()
                            .unwrap_or_else(|| "__thread".into());
                        let builtin = match base.as_str() {
                            "thread" => Some(thread_name.as_str()),
                            "block" => Some("__block"),
                            "block_dim" | "blockDim" => Some("__block_dim"),
                            "grid_dim" | "gridDim" => Some("__grid_dim"),
                            _ => None,
                        };
                        if base == "warp" {
                            match b.as_str() {
                                "size" => "__simd_lanes".into(),
                                "lane" => "((__thread.x + __thread.y * __block_dim.x + __thread.z * __block_dim.x * __block_dim.y).rem_euclid(__simd_lanes))".into(),
                                _ => format!("{}.{}", self.expr(a), b),
                            }
                        } else if let Some(builtin) = builtin {
                            format!("{builtin}.{b}")
                        } else {
                            format!("{}.{}", self.expr(a), b)
                        }
                    } else {
                        format!("{}.{}", self.expr(a), b)
                    }
                } else {
                    format!("{}.{}", self.expr(a), b)
                }
            }
            ExprKind::Index(a, b) => {
                if let ExprKind::SliceRange {
                    start,
                    end,
                    inclusive,
                } = &b.kind
                {
                    let start = start
                        .as_deref()
                        .map(|value| format!("({}) as usize", self.expr(value)));
                    let end = end
                        .as_deref()
                        .map(|value| format!("({}) as usize", self.expr(value)));
                    let range = match (start, end, inclusive) {
                        (Some(start), Some(end), true) => format!("{start}..={end}"),
                        (Some(start), Some(end), false) => format!("{start}..{end}"),
                        (Some(start), None, _) => format!("{start}.."),
                        (None, Some(end), true) => format!("..={end}"),
                        (None, Some(end), false) => format!("..{end}"),
                        (None, None, _) => "..".into(),
                    };
                    format!("{}[{range}].to_vec()", self.expr(a))
                } else if matches!(&a.kind, ExprKind::Var(name) if self.dicts.contains(name)) {
                    format!(
                        "{}.get(&{}).cloned().expect(\"dictionary key not found\")",
                        self.expr(a),
                        self.expr(b)
                    )
                } else {
                    format!("{}[({}) as usize]", self.expr(a), self.expr(b))
                }
            }
            ExprKind::LabeledIndex(a, args) => self.labeled_index(a, args),
            ExprKind::Assign(a, b) => {
                if let ExprKind::Index(object, key) = &a.kind {
                    if matches!(&object.kind, ExprKind::Var(name) if self.dicts.contains(name)) {
                        return format!(
                            "{}.insert({}, {})",
                            self.expr(object),
                            self.expr(key),
                            self.expr(b)
                        );
                    }
                }
                let replayed = if self.control_flow_depth == 0 && self.warp_replayable(b) {
                    self.thread_override = Some("__boring_warp_source_thread".into());
                    let replayed = self.expr(b);
                    self.thread_override = None;
                    Some(replayed)
                } else {
                    None
                };
                if let ExprKind::BinOp(op, left, right) = &b.kind {
                    if let (Some((atomic, element)), Some(method)) = (
                        self.hardware_atomic(a),
                        match op {
                            BinOp::Add => Some("fetch_add"),
                            BinOp::Sub => Some("fetch_sub"),
                            BinOp::BitAnd => Some("fetch_and"),
                            BinOp::BitOr => Some("fetch_or"),
                            BinOp::BitXor => Some("fetch_xor"),
                            _ => None,
                        },
                    ) {
                        let target = self.lvalue(a);
                        if self.lvalue(left) == target {
                            let operand = self.expr(right);
                            return format!(
                                "{{ let __boring_atomic_value = (({operand}) as {element}); unsafe {{ std::sync::atomic::{atomic}::from_ptr(&mut {target} as *mut {element}) }}.{method}(__boring_atomic_value, std::sync::atomic::Ordering::Relaxed); }}"
                            );
                        }
                    }
                }
                let optional_destination = matches!(&a.kind, ExprKind::Var(name)
                    if self.value_types.get(name).is_some_and(|kind| kind.starts_with("Option<")));
                let mut value = if optional_destination {
                    let wrapped = self.implicit_some(b);
                    self.expr(&wrapped)
                } else {
                    self.expr(b)
                };
                // Stores convert to the element type, as in C-like kernel languages.
                if let ExprKind::Index(object, _) | ExprKind::LabeledIndex(object, _) = &a.kind {
                    if let ExprKind::Var(name) = &object.kind {
                        if let (Some(element), Some(source)) = (self.value_types.get(name).cloned(), self.expr_type(b)) {
                            if numeric_rank(&element).is_some()
                                && numeric_rank(&source).is_some()
                                && element != source
                                && !matches!(&b.kind, ExprKind::Int(_) | ExprKind::Float(_))
                            {
                                value = format!("(({value}) as {element})");
                            }
                        }
                    }
                }
                let assignment = format!("{} = {}", self.lvalue(a), value);
                if let ExprKind::Var(name) = &a.kind {
                    self.replayable_locals.remove(name);
                    if let Some(replayed) = replayed {
                        self.replayed_assignments.insert(name.clone(), replayed);
                    } else {
                        self.replayed_assignments.remove(name);
                    }
                }
                assignment
            }
            ExprKind::QuestionAssign(a, b) => {
                let target = self.lvalue(a);
                format!(
                    "if {target}.is_none() {{ {target} = Some({}); }}",
                    self.expr(b)
                )
            }
            ExprKind::BinOp(op, a, b) => {
                if matches!(op, BinOp::Is | BinOp::IsNot) {
                    let optional = if matches!(&a.kind, ExprKind::Nil) {
                        Some(b.as_ref())
                    } else if matches!(&b.kind, ExprKind::Nil) {
                        Some(a.as_ref())
                    } else {
                        None
                    };
                    if let Some(optional) = optional {
                        return format!(
                            "{}.{}()",
                            self.expr(optional),
                            if matches!(op, BinOp::Is) {
                                "is_none"
                            } else {
                                "is_some"
                            }
                        );
                    }
                    let left_ty = self.expr_type(a);
                    let right_ty = self.expr_type(b);
                    let left = self.expr_as(a, right_ty.as_deref());
                    let right = self.expr_as(b, left_ty.as_deref());
                    return format!(
                        "({left} {} {right})",
                        if matches!(op, BinOp::Is) { "==" } else { "!=" }
                    );
                }
                let left_ty = self.expr_type(a);
                let right_ty = self.expr_type(b);
                let mut left = self.expr_as(a, right_ty.as_deref());
                let mut right = self.expr_as(b, left_ty.as_deref());
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem
                    | BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq)
                {
                    // Kernel languages promote mixed numeric operands implicitly.
                    let literal = |e: &Expr| matches!(&e.kind, ExprKind::Int(_) | ExprKind::Float(_));
                    if literal(a) || literal(b) {
                        // `expr_as` already types literals against the other operand.
                    } else if let (Some(l), Some(r)) = (&left_ty, &right_ty) {
                        if let Some(common) = common_numeric(l, r) {
                            if *l != common && !matches!(&a.kind, ExprKind::Int(_) | ExprKind::Float(_)) {
                                left = format!("(({left}) as {common})");
                            }
                            if *r != common && !matches!(&b.kind, ExprKind::Int(_) | ExprKind::Float(_)) {
                                right = format!("(({right}) as {common})");
                            }
                        }
                    }
                }
                match binop(op) {
                    Some(op) => format!("({left} {op} {right})"),
                    None => unreachable!("is/is not handled above"),
                }
            }
            ExprKind::UnaryOp(op, a) => format!(
                "({}{})",
                match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not | UnaryOp::BitNot => "!",
                },
                self.expr(a)
            ),
            ExprKind::Cast(a, t) => format!("({} as {})", self.expr(a), ty(t)),
            ExprKind::RelabelCast(value, _) => self.expr(value),
            ExprKind::Range {
                start,
                end,
                inclusive,
            } => format!(
                "{}..{}{}",
                self.expr(start),
                if *inclusive { "=" } else { "" },
                self.expr(end)
            ),
            ExprKind::Array(elements) => format!(
                "vec![{}]",
                elements
                    .iter()
                    .map(|element| self.expr(element))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ExprKind::Dict(entries) => format!(
                "std::collections::HashMap::from([{}])",
                entries
                    .iter()
                    .map(|(key, value)| format!("({}, {})", self.expr(key), self.expr(value)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ExprKind::Set(elements) => format!(
                "std::collections::HashSet::from([{}])",
                elements
                    .iter()
                    .map(|element| self.expr(element))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ExprKind::ArrayFill { value, count } => {
                format!("vec![{}; ({}) as usize]", self.expr(value), self.expr(count))
            }
            ExprKind::ArrayAlloc { count } => {
                format!("vec![Default::default(); ({}) as usize]", self.expr(count))
            }
            ExprKind::ArrayComp { expr, var, count } => format!(
                "(0..(({}) as usize)).map(|__boring_i| {{ let {var} = __boring_i as i32; {} }}).collect::<Vec<_>>()",
                self.expr(count),
                self.expr(expr),
            ),
            ExprKind::ArrayCompIter { expr, var, iter } => format!(
                "{}.iter().map(|{var}| {{ {} }}).collect::<Vec<_>>()",
                self.expr(iter),
                self.expr(expr),
            ),
            ExprKind::LabeledArrayComp { expr, clauses } => {
                let counts = clauses
                    .iter()
                    .map(|(_, count)| self.expr(count))
                    .collect::<Vec<_>>();
                let value = self.expr(expr);
                let mut generated = String::from("{ let mut __boring_labeled_array = Vec::new(); ");
                for (index, (name, _)) in clauses.iter().enumerate().rev() {
                    generated.push_str(&format!(
                        "for __boring_axis_{index} in 0..(({}) as usize) {{ let {name} = __boring_axis_{index} as i32; ",
                        counts[index]
                    ));
                }
                generated.push_str(&format!("__boring_labeled_array.push({value}); "));
                for _ in clauses {
                    generated.push_str("} ");
                }
                generated.push_str("__boring_labeled_array }");
                generated
            }
            ExprKind::Tuple(elements) => {
                let values = elements
                    .iter()
                    .map(|element| self.expr(element))
                    .collect::<Vec<_>>();
                if values.len() == 1 {
                    format!("({},)", values[0])
                } else {
                    format!("({})", values.join(", "))
                }
            }
            ExprKind::Else(value, fallback) => {
                let expected = self.expr_type(value);
                format!(
                    "{}.unwrap_or_else(|| {})",
                    self.expr(value),
                    self.expr_as(fallback, expected.as_deref())
                )
            }
            ExprKind::Call(c, args) => {
                if let ExprKind::Var(name) = &c.kind {
                    if name == "some" && args.len() == 1 {
                        return format!("Some({})", self.expr(&args[0].value));
                    }
                    if let Some(cast_ty) = primitive_cast_type(name) {
                        if args.len() == 1 {
                            return format!("({} as {cast_ty})", self.expr(&args[0].value));
                        }
                    }
                }
                if let ExprKind::Var(name) = &c.kind {
                    if !self.function_names.contains(name) {
                        if let Some(call) = self.free_math_call(name, args) {
                            return call;
                        }
                    }
                }
                if let ExprKind::Var(name) = &c.kind {
                    if let Some((params, ret, arrays, optionals)) = self.function_sigs.get(name).cloned() {
                        // Host functions use host numeric widths (`int` = isize);
                        // kernels use device widths (`int` = i32): convert at the call.
                        let call = format!(
                            "{}({})",
                            self.expr(c),
                            args.iter()
                                .enumerate()
                                .map(|(index, a)| match params.get(index).and_then(|p| p.as_ref()) {
                                    _ if arrays.get(index).copied().unwrap_or(0) == 1 => {
                                        format!("&({})", self.expr(&a.value))
                                    }
                                    _ if arrays.get(index).copied().unwrap_or(0) == 2 => {
                                        format!("({}).clone()", self.expr(&a.value))
                                    }
                                    _ if a.label.is_none() && optionals.get(index).is_some_and(|o| o.is_some()) => {
                                        // `T?` parameter: `some` is implicit, so wrap a plain value.
                                        let host = optionals[index].clone().flatten();
                                        let wrapped = self.implicit_some(&a.value);
                                        match (&wrapped.kind, host) {
                                            (ExprKind::Call(_, inner), Some(host))
                                                if !matches!(&a.value.kind, ExprKind::Call(..)) =>
                                            {
                                                format!("Some(({}) as {host})", self.expr(&inner[0].value))
                                            }
                                            _ => self.expr(&wrapped),
                                        }
                                    }
                                    Some(host) if a.label.is_none() => {
                                        format!("({}) as {host}", self.expr(&a.value))
                                    }
                                    _ => self.expr(&a.value),
                                })
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                        return match ret {
                            Some(device) if device.starts_with("Option<") => format!(
                                "({call}).map(|__v| __v as {})",
                                &device["Option<".len()..device.len() - 1]
                            ),
                            Some(device) => format!("(({call}) as {device})"),
                            None => call,
                        };
                    }
                }
                format!(
                    "{}({})",
                    self.expr(c),
                    args.iter()
                        .map(|a| self.expr(&a.value))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            ExprKind::GenericCall(callee, type_args, args) => format!(
                "{}::<{}>({})",
                self.expr(callee),
                type_args.iter().map(ty).collect::<Vec<_>>().join(", "),
                args.iter()
                    .map(|arg| self.expr(&arg.value))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ExprKind::Pipe(value, method, args) => {
                if self.function_names.contains(method) {
                    let mut values = vec![self.expr(value)];
                    values.extend(args.iter().map(|arg| self.expr(&arg.value)));
                    format!("{method}({})", values.join(", "))
                } else {
                    self.expr(&Expr {
                        kind: ExprKind::MethodCall(value.clone(), method.clone(), args.clone()),
                        line: e.line,
                        col: e.col,
                        len: e.len,
                    })
                }
            }
            ExprKind::MethodCall(o, m, args) => {
                let tuple_arity = self.tuple_arity(o);
                if let Some(arity) = tuple_arity {
                    match (m.as_str(), args.as_slice()) {
                        ("length" | "count", []) => return arity.to_string(),
                        ("isEmpty", []) => return (arity == 0).to_string(),
                        ("first", []) if arity > 0 => {
                            return format!("{}.0.clone()", self.expr(o));
                        }
                        ("last", []) if arity > 0 => {
                            return format!("{}.{}.clone()", self.expr(o), arity - 1);
                        }
                        ("map", [transform])
                            if matches!(&transform.value.kind, ExprKind::Closure(..)) =>
                        {
                            if arity == 0 {
                                return "()".into();
                            }
                            let (param, body) = self
                                .tuple_closure(&transform.value)
                                .expect("closure checked above");
                            let object = self.expr(o);
                            let values = (0..arity)
                                .map(|index| {
                                    format!(
                                        "{{ let {param} = {object}.{index}.clone(); {body} }}"
                                    )
                                })
                                .collect::<Vec<_>>();
                            return if arity == 1 {
                                format!("({},)", values[0])
                            } else {
                                format!("({})", values.join(", "))
                            };
                        }
                        ("all" | "any", [predicate])
                            if matches!(&predicate.value.kind, ExprKind::Closure(..)) =>
                        {
                            if arity == 0 {
                                return (m == "all").to_string();
                            }
                            let (param, body) = self
                                .tuple_closure(&predicate.value)
                                .expect("closure checked above");
                            let object = self.expr(o);
                            let operator = if m == "all" { " && " } else { " || " };
                            return (0..arity)
                                .map(|index| {
                                    format!(
                                        "{{ let {param} = {object}.{index}.clone(); {body} }}"
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join(operator);
                        }
                        _ => {}
                    }
                }
                let string_receiver = matches!(&o.kind, ExprKind::Str(_) | ExprKind::StringInterp(_))
                    || self.expr_type(o).as_deref() == Some("String");
                if string_receiver {
                    match (m.as_str(), args.as_slice()) {
                        ("length" | "count", []) => {
                            return format!("({}.len() as i32)", self.expr(o));
                        }
                        ("len", []) => {
                            return format!("{}.len()", self.expr(o));
                        }
                        ("isEmpty", []) => {
                            return format!("{}.is_empty()", self.expr(o));
                        }
                        ("trim" | "trimStart" | "trimEnd", []) => {
                            let method = match m.as_str() {
                                "trimStart" => "trim_start",
                                "trimEnd" => "trim_end",
                                _ => "trim",
                            };
                            return format!("{}.{method}().to_string()", self.expr(o));
                        }
                        ("toUpperCase" | "uppercased" | "upper" | "to_upper" | "toUpper", []) => {
                            return format!("{}.to_uppercase()", self.expr(o));
                        }
                        ("toLowerCase" | "lowercased" | "lower" | "to_lower" | "toLower", []) => {
                            return format!("{}.to_lowercase()", self.expr(o));
                        }
                        ("startsWith" | "hasPrefix", [prefix, ..]) => {
                            return format!(
                                "{}.starts_with({}.as_str())",
                                self.expr(o),
                                self.expr(&prefix.value)
                            );
                        }
                        ("endsWith" | "hasSuffix", [suffix, ..]) => {
                            return format!(
                                "{}.ends_with({}.as_str())",
                                self.expr(o),
                                self.expr(&suffix.value)
                            );
                        }
                        ("parseInt", []) => {
                            return format!("{}.trim().parse::<i32>().ok()", self.expr(o));
                        }
                        ("parseFloat", []) => {
                            return format!("{}.trim().parse::<f64>().ok()", self.expr(o));
                        }
                        ("parseFloat32", []) => {
                            return format!("{}.trim().parse::<f32>().ok()", self.expr(o));
                        }
                        ("chars", []) => {
                            return format!(
                                "{}.chars().map(|__boring_char| __boring_char.to_string()).collect::<Vec<String>>()",
                                self.expr(o)
                            );
                        }
                        ("split", [separator, ..]) => {
                            return format!(
                                "{}.split({}.as_str()).map(|__boring_part| __boring_part.to_string()).collect::<Vec<String>>()",
                                self.expr(o),
                                self.expr(&separator.value)
                            );
                        }
                        ("indexOf", [needle, ..]) => {
                            return format!(
                                "{}.find({}.as_str()).map(|__boring_index| __boring_index as i32)",
                                self.expr(o),
                                self.expr(&needle.value)
                            );
                        }
                        ("slice", [start, end, ..]) => {
                            let object = self.expr(o);
                            let start = self.expr(&start.value);
                            let end = self.expr(&end.value);
                            return format!(
                                "{{ let __boring_chars = {object}.chars().collect::<Vec<_>>(); let __boring_len = __boring_chars.len() as i64; let __boring_start = ({start}) as i64; let __boring_end = ({end}) as i64; let __boring_start = if __boring_start < 0 {{ __boring_len + __boring_start }} else {{ __boring_start }}.clamp(0, __boring_len); let __boring_end = if __boring_end < 0 {{ __boring_len + __boring_end }} else {{ __boring_end }}.clamp(__boring_start, __boring_len); __boring_chars[__boring_start as usize..__boring_end as usize].iter().collect::<String>() }}"
                            );
                        }
                        ("contains", [needle, ..]) => {
                            return format!(
                                "{}.contains({}.as_str())",
                                self.expr(o),
                                self.expr(&needle.value)
                            );
                        }
                        ("replace" | "replaceAll", [from, to, ..]) => {
                            return format!(
                                "{}.replace({}.as_str(), {}.as_str())",
                                self.expr(o),
                                self.expr(&from.value),
                                self.expr(&to.value)
                            );
                        }
                        ("repeat", [count, ..]) => {
                            return format!(
                                "{}.repeat(({}) as usize)",
                                self.expr(o),
                                self.expr(&count.value)
                            );
                        }
                        _ => {}
                    }
                }
                let local_collection = matches!(
                    &o.kind,
                    ExprKind::Var(name)
                        if self.arrays.contains(name)
                            || self.dicts.contains(name)
                            || self.sets.contains(name)
                );
                if local_collection && args.is_empty() {
                    match m.as_str() {
                        "length" | "count" => {
                            return format!("({}.len() as i32)", self.expr(o));
                        }
                        "len" => {
                            return format!("{}.len()", self.expr(o));
                        }
                        "isEmpty" => {
                            return format!("{}.is_empty()", self.expr(o));
                        }
                        _ => {}
                    }
                }
                if self.is_local_array_expr(o) {
                    match (m.as_str(), args.as_slice()) {
                        ("first" | "last", []) => {
                            return format!("{}.{}().cloned()", self.expr(o), m);
                        }
                        ("reversed", []) => {
                            return format!(
                                "{}.iter().rev().cloned().collect::<Vec<_>>()",
                                self.expr(o)
                            );
                        }
                        ("count", [predicate])
                            if matches!(&predicate.value.kind, ExprKind::Closure(..)) =>
                        {
                            let predicate = self
                                .owned_filter_predicate(&predicate.value)
                                .expect("closure checked above");
                            return format!(
                                "({}.iter().cloned().filter({}).count() as i32)",
                                self.expr(o),
                                predicate
                            );
                        }
                        ("find", [predicate])
                            if matches!(&predicate.value.kind, ExprKind::Closure(..)) =>
                        {
                            let predicate = self
                                .owned_filter_predicate(&predicate.value)
                                .expect("closure checked above");
                            return format!(
                                "{}.iter().cloned().find({})",
                                self.expr(o),
                                predicate
                            );
                        }
                        ("forEach" | "each", [action])
                            if matches!(&action.value.kind, ExprKind::Closure(..)) =>
                        {
                            return format!(
                                "{}.iter().cloned().for_each({})",
                                self.expr(o),
                                self.expr(&action.value)
                            );
                        }
                        ("join" | "joined", [separator, ..]) => {
                            return format!(
                                "{}.join({}.as_str())",
                                self.expr(o),
                                self.expr(&separator.value)
                            );
                        }
                        ("indexOf" | "position", [predicate])
                            if matches!(&predicate.value.kind, ExprKind::Closure(..)) =>
                        {
                            return format!(
                                "{}.iter().cloned().position({}).map(|__boring_index| __boring_index as i32)",
                                self.expr(o),
                                self.expr(&predicate.value)
                            );
                        }
                        ("indexOf", [value]) => {
                            return format!(
                                "{}.iter().position(|__boring_item| __boring_item == &({})).map(|__boring_index| __boring_index as i32)",
                                self.expr(o),
                                self.expr(&value.value)
                            );
                        }
                        ("take", [count, ..]) => {
                            return format!(
                                "{}.iter().cloned().take(({}) as usize).collect::<Vec<_>>()",
                                self.expr(o),
                                self.expr(&count.value)
                            );
                        }
                        ("drop", [count, ..]) => {
                            return format!(
                                "{}.iter().cloned().skip(({}) as usize).collect::<Vec<_>>()",
                                self.expr(o),
                                self.expr(&count.value)
                            );
                        }
                        ("slice", [start, end, ..]) => {
                            let object = self.expr(o);
                            let start = self.expr(&start.value);
                            let end = self.expr(&end.value);
                            return format!(
                                "{{ let __boring_len = {object}.len() as i64; let __boring_start = ({start}) as i64; let __boring_end = ({end}) as i64; let __boring_start = if __boring_start < 0 {{ __boring_len + __boring_start }} else {{ __boring_start }}.clamp(0, __boring_len); let __boring_end = if __boring_end < 0 {{ __boring_len + __boring_end }} else {{ __boring_end }}.clamp(__boring_start, __boring_len); {object}[__boring_start as usize..__boring_end as usize].to_vec() }}"
                            );
                        }
                        ("flatMap", [transform])
                            if matches!(&transform.value.kind, ExprKind::Closure(..)) =>
                        {
                            return format!(
                                "{}.iter().cloned().flat_map({}).collect::<Vec<_>>()",
                                self.expr(o),
                                self.expr(&transform.value)
                            );
                        }
                        ("flat", []) => {
                            return format!(
                                "{}.iter().cloned().flatten().collect::<Vec<_>>()",
                                self.expr(o)
                            );
                        }
                        ("sorted", []) => {
                            return format!(
                                "{{ let mut __boring_sorted = {}.clone(); __boring_sorted.sort_by(|__boring_a, __boring_b| __boring_a.partial_cmp(__boring_b).unwrap_or(std::cmp::Ordering::Equal)); __boring_sorted }}",
                                self.expr(o)
                            );
                        }
                        ("sortedBy", [key])
                            if matches!(&key.value.kind, ExprKind::Closure(..)) =>
                        {
                            let (param, body) = self
                                .sort_key(&key.value)
                                .expect("closure checked above");
                            return format!(
                                "{{ let mut __boring_sorted = {}.clone(); __boring_sorted.sort_by(|__boring_a, __boring_b| {{ let __boring_key_a = {{ let {param} = __boring_a.clone(); {body} }}; let __boring_key_b = {{ let {param} = __boring_b.clone(); {body} }}; __boring_key_a.partial_cmp(&__boring_key_b).unwrap_or(std::cmp::Ordering::Equal) }}); __boring_sorted }}",
                                self.expr(o)
                            );
                        }
                        ("zip", [other, ..]) => {
                            return format!(
                                "{}.iter().cloned().zip({}.iter().cloned()).collect::<Vec<_>>()",
                                self.expr(o),
                                self.expr(&other.value)
                            );
                        }
                        ("enumerate", []) => {
                            return format!(
                                "{}.iter().cloned().enumerate().map(|(__boring_index, __boring_value)| (__boring_index as i32, __boring_value)).collect::<Vec<_>>()",
                                self.expr(o)
                            );
                        }
                        ("firstIndex", []) => {
                            return format!(
                                "if {}.is_empty() {{ None }} else {{ Some(0i32) }}",
                                self.expr(o)
                            );
                        }
                        ("nextIndex", [index, ..]) => {
                            let object = self.expr(o);
                            let index = self.expr(&index.value);
                            return format!(
                                "{{ let __boring_next = ({index}) as i64 + 1; if __boring_next >= 0 && (__boring_next as usize) < {object}.len() {{ Some(__boring_next as i32) }} else {{ None }} }}"
                            );
                        }
                        ("getAt", [index, ..]) => {
                            return format!(
                                "{}[({}) as usize].clone()",
                                self.expr(o),
                                self.expr(&index.value)
                            );
                        }
                        ("removeAt" | "remove_at", [index, ..]) => {
                            let object = self.expr(o);
                            let index = self.expr(&index.value);
                            return format!(
                                "{{ let mut __boring_collection = {object}.clone(); let __boring_idx = ({index}) as i64; let __boring_idx = if __boring_idx < 0 {{ (__boring_collection.len() as i64 + __boring_idx) as usize }} else {{ __boring_idx as usize }}; __boring_collection.remove(__boring_idx); __boring_collection }}"
                            );
                        }
                        ("push", [value]) => {
                            return format!("{}.push({})", self.expr(o), self.expr(&value.value));
                        }
                        ("insert", [index, value, ..]) => {
                            return format!(
                                "{}.insert(({}) as usize, {})",
                                self.expr(o),
                                self.expr(&index.value),
                                self.expr(&value.value)
                            );
                        }
                        ("append" | "extend", [values, ..]) => {
                            return format!(
                                "{}.extend({}.iter().cloned())",
                                self.expr(o),
                                self.expr(&values.value)
                            );
                        }
                        ("clear", []) => {
                            return format!("{}.clear()", self.expr(o));
                        }
                        ("dedup", []) => {
                            return format!("{}.dedup()", self.expr(o));
                        }
                        ("truncate", [length, ..]) => {
                            return format!(
                                "{}.truncate(({}) as usize)",
                                self.expr(o),
                                self.expr(&length.value)
                            );
                        }
                        ("retain", [predicate])
                            if matches!(&predicate.value.kind, ExprKind::Closure(..)) =>
                        {
                            let predicate = self
                                .owned_filter_predicate(&predicate.value)
                                .expect("closure checked above");
                            return format!("{}.retain({predicate})", self.expr(o));
                        }
                        ("reverse", []) => {
                            return format!("{}.reverse()", self.expr(o));
                        }
                        ("sort", []) => {
                            return format!(
                                "{}.sort_by(|__boring_a, __boring_b| __boring_a.partial_cmp(__boring_b).unwrap_or(std::cmp::Ordering::Equal))",
                                self.expr(o)
                            );
                        }
                        ("sortBy", [key])
                            if matches!(&key.value.kind, ExprKind::Closure(..)) =>
                        {
                            let (param, body) = self
                                .sort_key(&key.value)
                                .expect("closure checked above");
                            return format!(
                                "{}.sort_by(|__boring_a, __boring_b| {{ let __boring_key_a = {{ let {param} = __boring_a.clone(); {body} }}; let __boring_key_b = {{ let {param} = __boring_b.clone(); {body} }}; __boring_key_a.partial_cmp(&__boring_key_b).unwrap_or(std::cmp::Ordering::Equal) }})",
                                self.expr(o)
                            );
                        }
                        ("pop", []) => {
                            return format!("{}.pop().unwrap_or_default()", self.expr(o));
                        }
                        ("remove", [index, ..]) => {
                            let object = self.expr(o);
                            let index = self.expr(&index.value);
                            return format!(
                                "{{ let __boring_idx = ({index}) as i64; let __boring_idx = if __boring_idx < 0 {{ ({object}.len() as i64 + __boring_idx) as usize }} else {{ __boring_idx as usize }}; {object}.remove(__boring_idx) }}"
                            );
                        }
                        ("move", [from, to, ..]) => {
                            let object = self.expr(o);
                            let from = self.expr(&from.value);
                            let to = self.expr(&to.value);
                            return format!(
                                "{{ let __boring_from = ({from}) as i64; let __boring_to = ({to}) as i64; let __boring_len = {object}.len() as i64; assert!(__boring_from >= 0 && __boring_from < __boring_len && __boring_to >= 0 && __boring_to < __boring_len, \"move indices out of range\"); if __boring_from != __boring_to {{ let __boring_item = {object}.remove(__boring_from as usize); {object}.insert(__boring_to as usize, __boring_item); }} }}"
                            );
                        }
                        ("removeIndices", [indices, ..]) => {
                            let object = self.expr(o);
                            let indices = self.expr(&indices.value);
                            return format!(
                                "{{ let __boring_indices = {indices}; let __boring_len = {object}.len() as i64; let mut __boring_previous = -1i64; for &__boring_index in &__boring_indices {{ let __boring_index = __boring_index as i64; assert!(__boring_index >= 0 && __boring_index < __boring_len, \"removeIndices index out of range\"); assert!(__boring_index > __boring_previous, \"removeIndices indices must be unique and strictly ascending\"); __boring_previous = __boring_index; }} for &__boring_index in __boring_indices.iter().rev() {{ {object}.remove(__boring_index as usize); }} }}"
                            );
                        }
                        _ => {}
                    }
                }
                if matches!(&o.kind, ExprKind::Var(name) if self.dicts.contains(name)) {
                    match (m.as_str(), args.as_slice()) {
                        ("firstIndex", []) => {
                            return format!("{}.keys().next().cloned()", self.expr(o));
                        }
                        ("nextIndex", [key, ..]) => {
                            let object = self.expr(o);
                            let key = self.expr(&key.value);
                            return format!(
                                "{{ let __boring_key = {key}; let mut __boring_keys = {object}.keys(); let _ = __boring_keys.find(|__boring_candidate| *__boring_candidate == &__boring_key); __boring_keys.next().cloned() }}"
                            );
                        }
                        ("getAt", [key, ..]) => {
                            return format!(
                                "{}.get(&{}).cloned().expect(\"getAt key not found\")",
                                self.expr(o),
                                self.expr(&key.value)
                            );
                        }
                        ("removeAt", [key, ..]) => {
                            let object = self.expr(o);
                            let key = self.expr(&key.value);
                            return format!(
                                "{{ let mut __boring_collection = {object}.clone(); __boring_collection.remove(&{key}); __boring_collection }}"
                            );
                        }
                        ("map", [transform])
                            if matches!(&transform.value.kind, ExprKind::Closure(..)) =>
                        {
                            let (key, value, body) = self
                                .dict_closure(&transform.value)
                                .expect("closure checked above");
                            return format!(
                                "{}.iter().map(|__boring_entry| {{ let {key} = __boring_entry.0.clone(); let {value} = __boring_entry.1.clone(); ({key}.clone(), {body}) }}).collect::<std::collections::HashMap<_, _>>()",
                                self.expr(o)
                            );
                        }
                        ("filter", [predicate])
                            if matches!(&predicate.value.kind, ExprKind::Closure(..)) =>
                        {
                            let (key, value, body) = self
                                .dict_closure(&predicate.value)
                                .expect("closure checked above");
                            return format!(
                                "{}.iter().filter(|__boring_entry| {{ let {key} = __boring_entry.0.clone(); let {value} = __boring_entry.1.clone(); {body} }}).map(|(__boring_key, __boring_value)| (__boring_key.clone(), __boring_value.clone())).collect::<std::collections::HashMap<_, _>>()",
                                self.expr(o)
                            );
                        }
                        ("get", [key]) => {
                            return format!(
                                "{}.get(&{}).cloned()",
                                self.expr(o),
                                self.expr(&key.value)
                            );
                        }
                        ("get", [key, fallback, ..]) => {
                            return format!(
                                "{}.get(&{}).cloned().unwrap_or_else(|| {})",
                                self.expr(o),
                                self.expr(&key.value),
                                self.expr(&fallback.value)
                            );
                        }
                        ("keys", []) => {
                            return format!("{}.keys().cloned().collect::<Vec<_>>()", self.expr(o));
                        }
                        ("values", []) => {
                            return format!("{}.values().cloned().collect::<Vec<_>>()", self.expr(o));
                        }
                        ("set" | "put", [key, value, ..]) => {
                            return format!(
                                "{}.insert({}, {})",
                                self.expr(o),
                                self.expr(&key.value),
                                self.expr(&value.value)
                            );
                        }
                        ("remove", [key, ..]) => {
                            return format!(
                                "{}.remove(&{})",
                                self.expr(o),
                                self.expr(&key.value)
                            );
                        }
                        _ => {}
                    }
                }
                if matches!(&o.kind, ExprKind::Var(name) if self.sets.contains(name)) {
                    match (m.as_str(), args.as_slice()) {
                        ("firstIndex", []) => {
                            return format!(
                                "if {}.is_empty() {{ None }} else {{ Some(0i32) }}",
                                self.expr(o)
                            );
                        }
                        ("nextIndex", [index, ..]) => {
                            let object = self.expr(o);
                            let index = self.expr(&index.value);
                            return format!(
                                "{{ let __boring_next = ({index}) as i64 + 1; if __boring_next >= 0 && (__boring_next as usize) < {object}.len() {{ Some(__boring_next as i32) }} else {{ None }} }}"
                            );
                        }
                        ("getAt", [index, ..]) => {
                            return format!(
                                "{}.iter().nth(({}) as usize).cloned().expect(\"getAt index out of range\")",
                                self.expr(o),
                                self.expr(&index.value)
                            );
                        }
                        ("removeAt", [index, ..]) => {
                            let object = self.expr(o);
                            let index = self.expr(&index.value);
                            return format!(
                                "{{ let __boring_element = {object}.iter().nth(({index}) as usize).cloned(); let mut __boring_collection = {object}.clone(); if let Some(__boring_element) = __boring_element {{ __boring_collection.remove(&__boring_element); }} __boring_collection }}"
                            );
                        }
                        ("toArray", []) => {
                            return format!("{}.iter().cloned().collect::<Vec<_>>()", self.expr(o));
                        }
                        ("union", [other]) => {
                            return format!(
                                "{}.union(&{}).cloned().collect::<std::collections::HashSet<_>>()",
                                self.expr(o),
                                self.expr(&other.value)
                            );
                        }
                        ("intersection", [other]) => {
                            return format!(
                                "{}.intersection(&{}).cloned().collect::<std::collections::HashSet<_>>()",
                                self.expr(o),
                                self.expr(&other.value)
                            );
                        }
                        ("difference", [other]) => {
                            return format!(
                                "{}.difference(&{}).cloned().collect::<std::collections::HashSet<_>>()",
                                self.expr(o),
                                self.expr(&other.value)
                            );
                        }
                        ("add" | "insert", [value, ..]) => {
                            return format!(
                                "{}.insert({})",
                                self.expr(o),
                                self.expr(&value.value)
                            );
                        }
                        ("remove", [value, ..]) => {
                            return format!(
                                "{}.remove(&{})",
                                self.expr(o),
                                self.expr(&value.value)
                            );
                        }
                        _ => {}
                    }
                }
                if m == "contains"
                    && args.len() == 1
                    && matches!(&o.kind, ExprKind::Var(name) if self.dicts.contains(name))
                {
                    return format!(
                        "{}.contains_key(&{})",
                        self.expr(o),
                        self.expr(&args[0].value)
                    );
                }
                if m == "contains"
                    && args.len() == 1
                    && matches!(&o.kind, ExprKind::Var(name)
                        if self.arrays.contains(name) || self.sets.contains(name))
                {
                    return format!(
                        "{}.contains(&{})",
                        self.expr(o),
                        self.expr(&args[0].value)
                    );
                }
                if matches!(&o.kind, ExprKind::Index(_, _) | ExprKind::LabeledIndex(_, _))
                    && matches!(m.as_str(), "min" | "max" | "swap" | "cas")
                    && args.len() == if m == "cas" { 2 } else { 1 }
                {
                    // SIMD work-items execute in a deterministic CPU loop, so an
                    // indexed atomic operation needs no hardware atomic wrapper.
                    // Keep the GPU contract nonetheless: mutate the element and
                    // yield its previous value.  Taking one mutable reference also
                    // ensures that a computed index is evaluated exactly once.
                    let target = self.lvalue(o);
                    let values = args
                        .iter()
                        .map(|arg| self.expr(&arg.value))
                        .collect::<Vec<_>>();
                    if let Some((atomic, element)) = self.hardware_atomic(o) {
                        let cast = |value: &String| format!("(({value}) as {element})");
                        let head = format!("let __boring_atomic = unsafe {{ std::sync::atomic::{atomic}::from_ptr(&mut {target} as *mut {element}) }};");
                        let relaxed = "std::sync::atomic::Ordering::Relaxed";
                        return match m.as_str() {
                            "min" => format!("{{ let __boring_atomic_value = {}; {head} __boring_atomic.fetch_min(__boring_atomic_value, {relaxed}) }}", cast(&values[0])),
                            "max" => format!("{{ let __boring_atomic_value = {}; {head} __boring_atomic.fetch_max(__boring_atomic_value, {relaxed}) }}", cast(&values[0])),
                            "swap" => format!("{{ let __boring_atomic_value = {}; {head} __boring_atomic.swap(__boring_atomic_value, {relaxed}) }}", cast(&values[0])),
                            _ => format!("{{ let __boring_atomic_expected = {}; let __boring_atomic_desired = {}; {head} match __boring_atomic.compare_exchange(__boring_atomic_expected, __boring_atomic_desired, {relaxed}, {relaxed}) {{ Ok(previous) | Err(previous) => previous }} }}", cast(&values[0]), cast(&values[1])),
                        };
                    }
                    return match m.as_str() {
                        "min" => format!(
                            "{{ let __boring_atomic_value = {}; let __boring_atomic_target = &mut {target}; let __boring_atomic_old = (*__boring_atomic_target).clone(); *__boring_atomic_target = if __boring_atomic_old <= __boring_atomic_value {{ __boring_atomic_old.clone() }} else {{ __boring_atomic_value }}; __boring_atomic_old }}",
                            values[0]
                        ),
                        "max" => format!(
                            "{{ let __boring_atomic_value = {}; let __boring_atomic_target = &mut {target}; let __boring_atomic_old = (*__boring_atomic_target).clone(); *__boring_atomic_target = if __boring_atomic_old >= __boring_atomic_value {{ __boring_atomic_old.clone() }} else {{ __boring_atomic_value }}; __boring_atomic_old }}",
                            values[0]
                        ),
                        "swap" => format!(
                            "{{ let __boring_atomic_value = {}; let __boring_atomic_target = &mut {target}; let __boring_atomic_old = (*__boring_atomic_target).clone(); *__boring_atomic_target = __boring_atomic_value; __boring_atomic_old }}",
                            values[0]
                        ),
                        "cas" => format!(
                            "{{ let __boring_atomic_expected = {}; let __boring_atomic_desired = {}; let __boring_atomic_target = &mut {target}; let __boring_atomic_old = (*__boring_atomic_target).clone(); if __boring_atomic_old == __boring_atomic_expected {{ *__boring_atomic_target = __boring_atomic_desired; }} __boring_atomic_old }}",
                            values[0], values[1]
                        ),
                        _ => unreachable!(),
                    };
                }
                if m == "sum" && args.is_empty() {
                    let sum_ty = self.expr_type(o).unwrap_or_else(|| "i32".into());
                    return format!("{}.iter().cloned().sum::<{sum_ty}>()", self.expr(o));
                }
                if matches!(m.as_str(), "min" | "max") && args.is_empty() {
                    return format!(
                        "{}.iter().cloned().{}().unwrap_or_default()",
                        self.expr(o),
                        m
                    );
                }
                if matches!(m.as_str(), "any" | "all") && args.len() == 1 {
                    return format!(
                        "{}.iter().cloned().{}({})",
                        self.expr(o),
                        m,
                        self.expr(&args[0].value)
                    );
                }
                if m == "map" && args.len() == 1 {
                    return format!(
                        "{}.iter().cloned().map({}).collect::<Vec<_>>()",
                        self.expr(o),
                        self.expr(&args[0].value)
                    );
                }
                if m == "filter" && args.len() == 1 {
                    if let ExprKind::Closure(params, _, body, _, _) = &args[0].value.kind {
                        if let Some(param) = params.first() {
                            let body = match body {
                                ClosureBody::Expr(value) => self.expr(value),
                                ClosureBody::Block(statements) => {
                                    format!("{{ {} }}", self.expr_body_as(statements, None))
                                }
                            };
                            return format!(
                                "{}.iter().cloned().filter(|__boring_item| {{ let {} = __boring_item.clone(); {} }}).collect::<Vec<_>>()",
                                self.expr(o),
                                param.name,
                                body
                            );
                        }
                    }
                }
                if matches!(m.as_str(), "reduce" | "fold") && args.len() == 2 {
                    return format!(
                        "{}.iter().cloned().fold({}, {})",
                        self.expr(o),
                        self.expr(&args[0].value),
                        self.expr(&args[1].value)
                    );
                }
                if self.coop && is_warp_receiver(o) && m == "sync" && args.is_empty() {
                    return "__coop::warp_sync()".into();
                }
                if self.coop && is_warp_receiver(o) && is_warp_shuffle(m) && args.len() == 2 {
                    let kind = match m.as_str() {
                        "shuffle" => 0,
                        "shuffle_down" | "shuffleDown" => 1,
                        "shuffle_up" | "shuffleUp" => 2,
                        _ => 3,
                    };
                    return format!(
                        "__coop::shuffle({kind}u8, {}, ({}) as i32)",
                        self.expr(&args[0].value),
                        self.expr(&args[1].value)
                    );
                }
                if is_warp_receiver(o) && is_warp_shuffle(m) {
                    return self.warp_shuffle(e, m, args);
                }
                format!(
                    "{}.{}({})",
                    self.expr(o),
                    m,
                    args.iter()
                        .map(|a| self.expr(&a.value))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            ExprKind::OptionalField(object, field) => format!(
                "{}.as_ref().map(|__boring_optional| __boring_optional.{field}.clone())",
                self.expr(object)
            ),
            ExprKind::OptionalMethodCall(object, method, args) => format!(
                "{}.as_ref().map(|__boring_optional| __boring_optional.{}({}))",
                self.expr(object),
                method,
                args.iter()
                    .map(|arg| self.expr(&arg.value))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            ExprKind::If(i) => {
                let result_types = i
                    .branches
                    .iter()
                    .filter_map(|(_, body)| self.body_type(body))
                    .chain(i.else_body.iter().filter_map(|body| self.body_type(body)))
                    .collect::<Vec<_>>();
                let result_ty = if result_types.iter().any(|kind| kind == "f32") {
                    Some("f32")
                } else if result_types.iter().any(|kind| kind == "f64") {
                    Some("f64")
                } else {
                    None
                };
                let mut out = String::new();
                for (index, (condition, body)) in i.branches.iter().enumerate() {
                    if index > 0 {
                        out.push_str(" else ");
                    }
                    out.push_str(&format!(
                        "if {} {{ {} }}",
                        self.expr(condition),
                        self.expr_body_as(body, result_ty)
                    ));
                }
                if let Some(body) = &i.else_body {
                    out.push_str(&format!(
                        " else {{ {} }}",
                        self.expr_body_as(body, result_ty)
                    ));
                } else {
                    out.push_str(" else { Default::default() }");
                }
                out
            }
            ExprKind::Match(m) => {
                let subject = self.expr(&m.subject);
                let mut out = format!("match {subject} {{ ");
                for arm in &m.arms {
                    let patterns = arm
                        .patterns
                        .iter()
                        .map(simd_pattern)
                        .collect::<Vec<_>>()
                        .join(" | ");
                    let guard = arm
                        .guard
                        .as_ref()
                        .map(|guard| format!(" if {}", self.expr(guard)))
                        .unwrap_or_default();
                    let body = match &arm.body {
                        MatchBody::Expr(value) => self.expr(value),
                        MatchBody::Block(body) => {
                            format!("{{ {} }}", self.expr_body_as(body, None))
                        }
                    };
                    out.push_str(&format!("{patterns}{guard} => {body}, "));
                }
                out.push('}');
                out
            }
            ExprKind::Loop(loop_stmt) => {
                let outer_output = std::mem::take(&mut self.out);
                let previous_break_sync = self.while_let_break_sync.take();
                self.value_loop_depth += 1;
                self.line(0, "loop {");
                for stmt in &loop_stmt.body {
                    self.stmt(stmt, 1);
                }
                self.line(0, "}");
                self.value_loop_depth -= 1;
                self.while_let_break_sync = previous_break_sync;
                let expression = std::mem::replace(&mut self.out, outer_output);
                expression.trim().to_string()
            }
            ExprKind::Closure(params, return_ty, body, _, _) => {
                let params = params
                    .iter()
                    .map(|param| {
                        param
                            .ty
                            .as_ref()
                            .map(|param_ty| format!("{}: {}", param.name, ty(param_ty)))
                            .unwrap_or_else(|| param.name.clone())
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let return_ty = return_ty
                    .as_ref()
                    .map(|return_ty| format!(" -> {}", ty(return_ty)))
                    .unwrap_or_default();
                let body = match body {
                    ClosureBody::Expr(value) => self.expr(value),
                    ClosureBody::Block(statements) => {
                        format!("{{ {} }}", self.expr_body_as(statements, None))
                    }
                };
                format!("|{params}|{return_ty} {body}")
            }
            ExprKind::Block(statements) | ExprKind::Do(statements) => {
                format!("{{ {} }}", self.expr_body_as(statements, None))
            }
            _ => {
                self.errors.push(crate::errors::SourceError::at(
                    "this expression is not yet supported inside a --target simd kernel",
                    e.line,
                    e.col,
                ));
                "/* rejected unsupported SIMD expression */ 0".into()
            }
        }
    }
    fn owned_filter_predicate(&mut self, predicate: &Expr) -> Option<String> {
        let ExprKind::Closure(params, _, body, _, _) = &predicate.kind else {
            return None;
        };
        let param = params.first()?;
        let body = match body {
            ClosureBody::Expr(value) => self.expr(value),
            ClosureBody::Block(statements) => {
                format!("{{ {} }}", self.expr_body_as(statements, None))
            }
        };
        Some(format!(
            "|__boring_item| {{ let {} = __boring_item.clone(); {} }}",
            param.name, body
        ))
    }
    fn sort_key(&mut self, key: &Expr) -> Option<(String, String)> {
        let ExprKind::Closure(params, _, body, _, _) = &key.kind else {
            return None;
        };
        let param = params.first()?.name.clone();
        let body = match body {
            ClosureBody::Expr(value) => self.expr(value),
            ClosureBody::Block(statements) => {
                format!("{{ {} }}", self.expr_body_as(statements, None))
            }
        };
        Some((param, body))
    }
    fn dict_closure(&mut self, closure: &Expr) -> Option<(String, String, String)> {
        let ExprKind::Closure(params, _, body, _, _) = &closure.kind else {
            return None;
        };
        let key = params.first()?.name.clone();
        let value = params.get(1)?.name.clone();
        let body = match body {
            ClosureBody::Expr(value) => self.expr(value),
            ClosureBody::Block(statements) => {
                format!("{{ {} }}", self.expr_body_as(statements, None))
            }
        };
        Some((key, value, body))
    }
    fn is_local_array_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Var(name) => self.arrays.contains(name),
            ExprKind::Array(_)
            | ExprKind::ArrayFill { .. }
            | ExprKind::ArrayAlloc { .. }
            | ExprKind::ArrayComp { .. }
            | ExprKind::ArrayCompIter { .. }
            | ExprKind::LabeledArrayComp { .. } => true,
            ExprKind::MethodCall(receiver, method, _) | ExprKind::Pipe(receiver, method, _) => {
                matches!(
                    method.as_str(),
                    "map"
                        | "filter"
                        | "reversed"
                        | "take"
                        | "drop"
                        | "slice"
                        | "flatMap"
                        | "flat"
                        | "sorted"
                        | "sortedBy"
                        | "zip"
                        | "enumerate"
                        | "removeAt"
                        | "split"
                        | "chars"
                        | "keys"
                        | "values"
                        | "toArray"
                ) && !matches!(&receiver.kind, ExprKind::Var(name)
                    if self.dicts.contains(name)
                        && matches!(method.as_str(), "map" | "filter" | "removeAt"))
            }
            _ => false,
        }
    }
    fn tuple_closure(&mut self, closure: &Expr) -> Option<(String, String)> {
        let ExprKind::Closure(params, _, body, _, _) = &closure.kind else {
            return None;
        };
        let param = params.first()?.name.clone();
        let body = match body {
            ClosureBody::Expr(value) => self.expr(value),
            ClosureBody::Block(statements) => {
                format!("{{ {} }}", self.expr_body_as(statements, None))
            }
        };
        Some((param, body))
    }
    fn tuple_arity(&self, expr: &Expr) -> Option<usize> {
        match &expr.kind {
            ExprKind::Tuple(elements) => Some(elements.len()),
            ExprKind::Var(name) => self.tuple_arities.get(name).copied(),
            ExprKind::MethodCall(receiver, method, _) if method == "map" => {
                self.tuple_arity(receiver)
            }
            ExprKind::Pipe(receiver, method, _)
                if method == "map" && !self.function_names.contains(method) =>
            {
                self.tuple_arity(receiver)
            }
            ExprKind::If(if_expr) => {
                let else_body = if_expr.else_body.as_ref()?;
                let mut arities = if_expr
                    .branches
                    .iter()
                    .map(|(_, body)| self.body_tuple_arity(body))
                    .chain(std::iter::once(self.body_tuple_arity(else_body)));
                let first = arities.next()??;
                arities.all(|arity| arity == Some(first)).then_some(first)
            }
            ExprKind::Match(match_expr) => {
                let mut arities = match_expr.arms.iter().map(|arm| match &arm.body {
                    MatchBody::Expr(value) => self.tuple_arity(value),
                    MatchBody::Block(body) => self.body_tuple_arity(body),
                });
                let first = arities.next()??;
                arities.all(|arity| arity == Some(first)).then_some(first)
            }
            ExprKind::Else(_, fallback) => self.tuple_arity(fallback),
            ExprKind::Block(body) | ExprKind::Do(body) => self.body_tuple_arity(body),
            _ => None,
        }
    }
    fn optional_tuple_arity(&self, expr: &Expr) -> Option<usize> {
        match &expr.kind {
            ExprKind::Call(callee, args)
                if matches!(&callee.kind, ExprKind::Var(name) if name == "some")
                    && args.len() == 1 =>
            {
                self.tuple_arity(&args[0].value)
            }
            _ => None,
        }
    }
    fn body_tuple_arity(&self, body: &[Stmt]) -> Option<usize> {
        match body.last() {
            Some(Stmt::Expr(expr)) => self.tuple_arity(expr),
            _ => None,
        }
    }
    fn warp_shuffle(&mut self, call: &Expr, method: &str, args: &[Arg]) -> String {
        if args.len() != 2 || !self.warp_replayable(&args[0].value) {
            self.errors.push(crate::errors::SourceError::at(
                "SIMD warp shuffles require two arguments and a lane-replayable value (coordinates, constants, and buffer reads only)",
                call.line,
                call.col,
            ));
            return "/* rejected non-replayable SIMD warp shuffle */ 0".into();
        }
        // Replaying a buffer read in another lane sees later stores of this kernel;
        // lockstep GPU lanes read first. Only the cooperative path orders that right.
        if expr_reads_any(&args[0].value, &self.written_fields) {
            self.errors.push(crate::errors::SourceError::at(
                "SIMD warp shuffles require two arguments and a lane-replayable value (a shuffled buffer element is also written by the kernel)",
                call.line,
                call.col,
            ));
            return "/* rejected shuffle of a written buffer */ 0".into();
        }
        let operand = self.expr(&args[1].value);
        self.thread_override = Some("__boring_warp_source_thread".into());
        let replayed = self.expr(&args[0].value);
        self.thread_override = None;
        let current = self.expr(&args[0].value);
        let target_lane = match method {
            "shuffle_down" | "shuffleDown" => {
                format!("__boring_warp_lane + ({operand} as i32)")
            }
            "shuffle_up" | "shuffleUp" => {
                format!("__boring_warp_lane - ({operand} as i32)")
            }
            "shuffle_xor" | "shuffleXor" => {
                format!("__boring_warp_lane ^ ({operand} as i32)")
            }
            "shuffle" => format!("{operand} as i32"),
            _ => unreachable!(),
        };
        format!(
            "{{ let __boring_warp_linear = __thread.x + __thread.y * __block_dim.x + __thread.z * __block_dim.x * __block_dim.y; let __boring_warp_lane = __boring_warp_linear.rem_euclid(__simd_lanes); let __boring_warp_target_lane = {target_lane}; let __boring_warp_target_linear = (__boring_warp_linear - __boring_warp_lane) + __boring_warp_target_lane; if __boring_warp_target_lane >= 0 && __boring_warp_target_lane < __simd_lanes && __boring_warp_target_linear < __block_dim.x * __block_dim.y * __block_dim.z {{ let __boring_warp_source_thread = __BoringDim3 {{ x: __boring_warp_target_linear.rem_euclid(__block_dim.x), y: (__boring_warp_target_linear / __block_dim.x).rem_euclid(__block_dim.y), z: __boring_warp_target_linear / (__block_dim.x * __block_dim.y) }}; {replayed} }} else {{ {current} }} }}"
        )
    }
    fn warp_replayable(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Int(_)
            | ExprKind::UInt64(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Bool(_)
            | ExprKind::Nil => true,
            ExprKind::StringInterp(segments) => segments.iter().all(|segment| match segment {
                StringSegment::Lit(_) => true,
                StringSegment::Expr(value) | StringSegment::FormattedExpr(value, _) => {
                    self.warp_replayable(value)
                }
            }),
            ExprKind::Var(name) => {
                matches!(name.as_str(), "thread" | "block" | "block_dim" | "gpu")
                    || self.fields.contains(name)
                    || self.global_consts.contains_key(name)
                    || self.replayable_locals.contains_key(name)
                    || self.replayed_assignments.contains_key(name)
            }
            ExprKind::DotIdent(name) => {
                matches!(name.as_str(), "thread" | "block" | "block_dim" | "grid_dim")
            }
            ExprKind::Field(object, _) | ExprKind::UnaryOp(_, object) | ExprKind::Cast(object, _) => {
                self.warp_replayable(object)
            }
            ExprKind::Index(object, index) => {
                self.warp_replayable(object) && self.warp_replayable(index)
            }
            ExprKind::LabeledIndex(object, args) => {
                self.warp_replayable(object)
                    && args.iter().all(|arg| self.warp_replayable(&arg.value))
            }
            ExprKind::BinOp(_, left, right) => {
                self.warp_replayable(left) && self.warp_replayable(right)
            }
            ExprKind::Call(callee, args) => {
                matches!(&callee.kind, ExprKind::Var(name) if primitive_cast_type(name).is_some())
                    && args.iter().all(|arg| self.warp_replayable(&arg.value))
            }
            _ => false,
        }
    }
    fn barrier_replayable(
        &self,
        expr: &Expr,
        locals: &std::collections::HashSet<String>,
    ) -> bool {
        match &expr.kind {
            ExprKind::Int(_)
            | ExprKind::UInt64(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Bool(_)
            | ExprKind::Nil => true,
            ExprKind::StringInterp(segments) => segments.iter().all(|segment| match segment {
                StringSegment::Lit(_) => true,
                StringSegment::Expr(value) | StringSegment::FormattedExpr(value, _) => {
                    self.barrier_replayable(value, locals)
                }
            }),
            ExprKind::Var(name) => {
                matches!(name.as_str(), "thread" | "block" | "block_dim" | "gpu")
                    || self.global_consts.contains_key(name)
                    || locals.contains(name)
            }
            ExprKind::DotIdent(name) => {
                matches!(name.as_str(), "thread" | "block" | "block_dim" | "grid_dim")
            }
            ExprKind::Field(object, _) | ExprKind::UnaryOp(_, object) | ExprKind::Cast(object, _) => {
                self.barrier_replayable(object, locals)
            }
            ExprKind::Index(object, index) => {
                self.barrier_replayable(object, locals)
                    && self.barrier_replayable(index, locals)
            }
            ExprKind::BinOp(_, left, right) => {
                self.barrier_replayable(left, locals)
                    && self.barrier_replayable(right, locals)
            }
            ExprKind::Call(callee, args) => {
                matches!(&callee.kind, ExprKind::Var(name) if primitive_cast_type(name).is_some())
                    && args
                        .iter()
                        .all(|arg| self.barrier_replayable(&arg.value, locals))
            }
            _ => false,
        }
    }
    fn block_uniform_expr(&self, expr: &Expr) -> bool {
        self.block_uniform_expr_with_locals(expr, &std::collections::HashSet::new())
    }
    fn block_uniform_locals(&self, body: &[Stmt]) -> std::collections::HashSet<String> {
        self.block_uniform_local_statements(body)
            .into_iter()
            .filter_map(|stmt| match stmt {
                Stmt::Let(local) => Some(local.name),
                _ => None,
            })
            .collect()
    }
    fn block_uniform_local_statements(&self, body: &[Stmt]) -> Vec<Stmt> {
        self.block_uniform_local_statements_with_locals(
            body,
            &std::collections::HashSet::new(),
        )
    }
    fn block_uniform_local_statements_with_locals(
        &self,
        body: &[Stmt],
        initial_locals: &std::collections::HashSet<String>,
    ) -> Vec<Stmt> {
        let mut locals = initial_locals.clone();
        let mut declarations = Vec::new();
        for stmt in body {
            let Stmt::Let(local) = stmt else { continue };
            if matches!(local.binding, BindingKind::Let)
                && local
                    .value
                    .as_ref()
                    .is_some_and(|value| self.block_uniform_expr_with_locals(value, &locals))
            {
                locals.insert(local.name.clone());
                declarations.push(stmt.clone());
            }
        }
        declarations
    }
    fn block_uniform_match_arm_guard_with_locals(
        &self,
        arm: &MatchArm,
        initial_locals: &std::collections::HashSet<String>,
    ) -> bool {
        let Some(guard) = &arm.guard else {
            return true;
        };
        let mut locals = initial_locals.clone();
        for pattern in &arm.patterns {
            collect_simd_pattern_bind_names(pattern, &mut locals);
        }
        self.block_uniform_expr_with_locals(guard, &locals)
    }
    fn block_uniform_guard_clauses_with_locals(
        &self,
        clauses: &[CondClause],
        initial_locals: &std::collections::HashSet<String>,
    ) -> bool {
        let mut locals = initial_locals.clone();
        for clause in clauses {
            match clause {
                CondClause::Let(name, value) => {
                    if !self.block_uniform_expr_with_locals(value, &locals) {
                        return false;
                    }
                    locals.insert(name.clone());
                }
                CondClause::LetPat(pattern, value) => {
                    if !self.block_uniform_expr_with_locals(value, &locals) {
                        return false;
                    }
                    collect_simd_pattern_bind_names(pattern, &mut locals);
                }
                CondClause::Expr(condition) => {
                    if !self.block_uniform_expr_with_locals(condition, &locals) {
                        return false;
                    }
                }
            }
        }
        true
    }
    fn block_uniform_expr_with_locals(
        &self,
        expr: &Expr,
        locals: &std::collections::HashSet<String>,
    ) -> bool {
        match &expr.kind {
            ExprKind::Int(_)
            | ExprKind::UInt64(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Bool(_)
            | ExprKind::Nil => true,
            ExprKind::StringInterp(segments) => segments.iter().all(|segment| match segment {
                StringSegment::Lit(_) => true,
                StringSegment::Expr(value) | StringSegment::FormattedExpr(value, _) => {
                    self.block_uniform_expr_with_locals(value, locals)
                }
            }),
            ExprKind::Tuple(elements) => elements
                .iter()
                .all(|element| self.block_uniform_expr_with_locals(element, locals)),
            ExprKind::Array(elements) | ExprKind::Set(elements) => elements
                .iter()
                .all(|element| self.block_uniform_expr_with_locals(element, locals)),
            ExprKind::Dict(entries) => entries.iter().all(|(key, value)| {
                self.block_uniform_expr_with_locals(key, locals)
                    && self.block_uniform_expr_with_locals(value, locals)
            }),
            ExprKind::ArrayFill { value, count } => {
                self.block_uniform_expr_with_locals(value, locals)
                    && self.block_uniform_expr_with_locals(count, locals)
            }
            ExprKind::ArrayAlloc { count } => {
                self.block_uniform_expr_with_locals(count, locals)
            }
            ExprKind::ArrayComp { expr, var, count } => {
                let mut element_locals = locals.clone();
                element_locals.insert(var.clone());
                self.block_uniform_expr_with_locals(count, locals)
                    && self.block_uniform_expr_with_locals(expr, &element_locals)
            }
            ExprKind::ArrayCompIter { expr, var, iter } => {
                let mut element_locals = locals.clone();
                element_locals.insert(var.clone());
                self.block_uniform_expr_with_locals(iter, locals)
                    && self.block_uniform_expr_with_locals(expr, &element_locals)
            }
            ExprKind::LabeledArrayComp { expr, clauses } => {
                let mut element_locals = locals.clone();
                for (var, _) in clauses {
                    element_locals.insert(var.clone());
                }
                clauses.iter().all(|(_, count)| {
                    self.block_uniform_expr_with_locals(count, locals)
                }) && self.block_uniform_expr_with_locals(expr, &element_locals)
            }
            ExprKind::Call(callee, args)
                if matches!(&callee.kind, ExprKind::Var(name)
                    if name == "some" || primitive_cast_type(name).is_some()) =>
            {
                args.iter()
                    .all(|arg| self.block_uniform_expr_with_locals(&arg.value, locals))
            }
            ExprKind::MethodCall(receiver, method, args)
                if args.is_empty()
                    && matches!(
                        method.as_str(),
                        "is_some"
                            | "is_none"
                            | "length"
                            | "count"
                            | "len"
                            | "isEmpty"
                            | "first"
                            | "last"
                            | "reversed"
                            | "flat"
                            | "sorted"
                            | "enumerate"
                            | "firstIndex"
                    ) =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
            }
            ExprKind::MethodCall(receiver, method, args)
                if args.is_empty() && matches!(method.as_str(), "sum" | "min" | "max") =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "any" | "all" | "count" | "find" | "indexOf" | "position")
                    && args.len() == 1
                    && matches!(&args[0].value.kind, ExprKind::Closure(..)) =>
            {
                let ExprKind::Closure(params, _, body, _, _) = &args[0].value.kind else {
                    return false;
                };
                let mut closure_locals = locals.clone();
                closure_locals.extend(params.iter().map(|param| param.name.clone()));
                self.block_uniform_expr_with_locals(receiver, locals)
                    && match body {
                        ClosureBody::Expr(value) => {
                            self.block_uniform_expr_with_locals(value, &closure_locals)
                        }
                        ClosureBody::Block(body) => {
                            self.block_uniform_expr_body_with_locals(body, &closure_locals)
                        }
                    }
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "forEach" | "each")
                    && args.len() == 1
                    && matches!(&args[0].value.kind, ExprKind::Closure(..)) =>
            {
                let ExprKind::Closure(params, _, body, _, _) = &args[0].value.kind else {
                    return false;
                };
                let mut closure_locals = locals.clone();
                closure_locals.extend(params.iter().map(|param| param.name.clone()));
                self.block_uniform_expr_with_locals(receiver, locals)
                    && match body {
                        ClosureBody::Expr(value) => {
                            self.block_uniform_expr_with_locals(value, &closure_locals)
                        }
                        ClosureBody::Block(body) => {
                            self.block_uniform_expr_body_with_locals(body, &closure_locals)
                        }
                    }
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "join" | "joined") && args.len() == 1 =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && self.block_uniform_expr_with_locals(&args[0].value, locals)
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "startsWith" | "hasPrefix" | "endsWith" | "hasSuffix")
                    && args.len() == 1 =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && self.block_uniform_expr_with_locals(&args[0].value, locals)
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "indexOf" | "contains" | "repeat")
                    && args.len() == 1 =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && self.block_uniform_expr_with_locals(&args[0].value, locals)
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "slice" | "replace" | "replaceAll")
                    && args.len() >= 2 =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && args.iter().all(|arg| {
                        self.block_uniform_expr_with_locals(&arg.value, locals)
                    })
            }
            ExprKind::MethodCall(receiver, method, args)
                if method == "indexOf" && args.len() == 1 =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && self.block_uniform_expr_with_locals(&args[0].value, locals)
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "take" | "drop") && args.len() == 1 =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && self.block_uniform_expr_with_locals(&args[0].value, locals)
            }
            ExprKind::MethodCall(receiver, method, args)
                if method == "zip" && args.len() == 1 =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && self.block_uniform_expr_with_locals(&args[0].value, locals)
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "nextIndex" | "getAt" | "removeAt" | "remove_at")
                    && args.len() == 1 =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && self.block_uniform_expr_with_locals(&args[0].value, locals)
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "move" | "removeIndices") =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && args.iter().all(|arg| {
                        self.block_uniform_expr_with_locals(&arg.value, locals)
                    })
            }
            ExprKind::MethodCall(receiver, method, args)
                if method == "slice" && args.len() >= 2 =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && args.iter().all(|arg| {
                        self.block_uniform_expr_with_locals(&arg.value, locals)
                    })
            }
            ExprKind::MethodCall(receiver, method, args)
                if method == "flatMap"
                    && args.len() == 1
                    && matches!(&args[0].value.kind, ExprKind::Closure(..)) =>
            {
                let ExprKind::Closure(params, _, body, _, _) = &args[0].value.kind else {
                    return false;
                };
                let mut closure_locals = locals.clone();
                closure_locals.extend(params.iter().map(|param| param.name.clone()));
                self.block_uniform_expr_with_locals(receiver, locals)
                    && match body {
                        ClosureBody::Expr(value) => {
                            self.block_uniform_expr_with_locals(value, &closure_locals)
                        }
                        ClosureBody::Block(body) => {
                            self.block_uniform_expr_body_with_locals(body, &closure_locals)
                        }
                    }
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "sortBy" | "sortedBy")
                    && args.len() == 1
                    && matches!(&args[0].value.kind, ExprKind::Closure(..)) =>
            {
                let ExprKind::Closure(params, _, body, _, _) = &args[0].value.kind else {
                    return false;
                };
                let mut closure_locals = locals.clone();
                closure_locals.extend(params.iter().map(|param| param.name.clone()));
                self.block_uniform_expr_with_locals(receiver, locals)
                    && match body {
                        ClosureBody::Expr(value) => {
                            self.block_uniform_expr_with_locals(value, &closure_locals)
                        }
                        ClosureBody::Block(body) => {
                            self.block_uniform_expr_body_with_locals(body, &closure_locals)
                        }
                    }
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "map" | "filter") && args.len() == 1 =>
            {
                let ExprKind::Closure(params, _, body, _, _) = &args[0].value.kind else {
                    return false;
                };
                let mut closure_locals = locals.clone();
                closure_locals.extend(params.iter().map(|param| param.name.clone()));
                self.block_uniform_expr_with_locals(receiver, locals)
                    && match body {
                        ClosureBody::Expr(value) => {
                            self.block_uniform_expr_with_locals(value, &closure_locals)
                        }
                        ClosureBody::Block(body) => {
                            self.block_uniform_expr_body_with_locals(body, &closure_locals)
                        }
                    }
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(method.as_str(), "reduce" | "fold") && args.len() == 2 =>
            {
                let ExprKind::Closure(params, _, body, _, _) = &args[1].value.kind else {
                    return false;
                };
                let mut closure_locals = locals.clone();
                closure_locals.extend(params.iter().map(|param| param.name.clone()));
                self.block_uniform_expr_with_locals(receiver, locals)
                    && self.block_uniform_expr_with_locals(&args[0].value, locals)
                    && match body {
                        ClosureBody::Expr(value) => {
                            self.block_uniform_expr_with_locals(value, &closure_locals)
                        }
                        ClosureBody::Block(body) => {
                            self.block_uniform_expr_body_with_locals(body, &closure_locals)
                        }
                    }
            }
            ExprKind::MethodCall(receiver, method, args)
                if method == "retain"
                    && args.len() == 1
                    && matches!(&args[0].value.kind, ExprKind::Closure(..)) =>
            {
                let ExprKind::Closure(params, _, body, _, _) = &args[0].value.kind else {
                    return false;
                };
                let mut closure_locals = locals.clone();
                closure_locals.extend(params.iter().map(|param| param.name.clone()));
                self.block_uniform_expr_with_locals(receiver, locals)
                    && match body {
                        ClosureBody::Expr(value) => {
                            self.block_uniform_expr_with_locals(value, &closure_locals)
                        }
                        ClosureBody::Block(body) => {
                            self.block_uniform_expr_body_with_locals(body, &closure_locals)
                        }
                    }
            }
            ExprKind::MethodCall(receiver, method, args)
                if matches!(
                    method.as_str(),
                    "contains"
                        | "get"
                        | "keys"
                        | "values"
                        | "toArray"
                        | "union"
                        | "intersection"
                        | "difference"
                )
                    && matches!(&receiver.kind, ExprKind::Var(name)
                        if locals.contains(name)
                            || self.arrays.contains(name)
                            || self.sets.contains(name)
                            || self.dicts.contains(name)) =>
            {
                self.block_uniform_expr_with_locals(receiver, locals)
                    && args.iter().all(|arg| {
                        self.block_uniform_expr_with_locals(&arg.value, locals)
                    })
            }
            ExprKind::Var(name) => {
                matches!(name.as_str(), "block" | "block_dim" | "grid_dim")
                    || self.global_consts.contains_key(name)
                    || self.fields.contains(name)
                    || self.collective_uniform_locals.contains(name)
                    || locals.contains(name)
            }
            ExprKind::DotIdent(name) => {
                matches!(name.as_str(), "block" | "block_dim" | "grid_dim")
            }
            ExprKind::Field(object, _)
                if matches!(
                    &object.kind,
                    ExprKind::Field(root, base)
                        if matches!(&root.kind, ExprKind::Var(name) if name == "gpu")
                            && matches!(base.as_str(), "block" | "block_dim" | "blockDim" | "grid_dim" | "gridDim")
                ) => true,
            ExprKind::Field(object, _)
            | ExprKind::UnaryOp(_, object)
            | ExprKind::Cast(object, _) => self.block_uniform_expr_with_locals(object, locals),
            ExprKind::Index(object, index) => {
                self.block_uniform_expr_with_locals(object, locals)
                    && self.block_uniform_expr_with_locals(index, locals)
            }
            ExprKind::LabeledIndex(object, indices) => {
                self.block_uniform_expr_with_locals(object, locals)
                    && indices.iter().all(|index| {
                        self.block_uniform_expr_with_locals(&index.value, locals)
                    })
            }
            ExprKind::SliceRange { start, end, .. } => {
                start.as_ref().is_none_or(|bound| {
                    self.block_uniform_expr_with_locals(bound, locals)
                }) && end.as_ref().is_none_or(|bound| {
                    self.block_uniform_expr_with_locals(bound, locals)
                })
            }
            ExprKind::BinOp(_, left, right) => {
                self.block_uniform_expr_with_locals(left, locals)
                    && self.block_uniform_expr_with_locals(right, locals)
            }
            ExprKind::Else(optional, fallback) => {
                self.block_uniform_expr_with_locals(optional, locals)
                    && self.block_uniform_expr_with_locals(fallback, locals)
            }
            ExprKind::If(if_expr) => {
                if_expr.branches.iter().all(|(condition, body)| {
                    self.block_uniform_expr_with_locals(condition, locals)
                        && self.block_uniform_expr_body_with_locals(body, locals)
                }) && if_expr.else_body.as_ref().is_none_or(|body| {
                    self.block_uniform_expr_body_with_locals(body, locals)
                })
            }
            ExprKind::Match(match_expr) => {
                self.block_uniform_expr_with_locals(&match_expr.subject, locals)
                    && match_expr.arms.iter().all(|arm| {
                        let mut arm_locals = locals.clone();
                        for pattern in &arm.patterns {
                            collect_simd_pattern_bind_names(pattern, &mut arm_locals);
                        }
                        arm.guard.as_ref().is_none_or(|guard| {
                            self.block_uniform_expr_with_locals(guard, &arm_locals)
                        }) && match &arm.body {
                            MatchBody::Expr(body) => {
                                self.block_uniform_expr_with_locals(body, &arm_locals)
                            }
                            MatchBody::Block(body) => {
                                self.block_uniform_expr_body_with_locals(body, &arm_locals)
                            }
                        }
                    })
            }
            ExprKind::Block(body) | ExprKind::Do(body) => {
                self.block_uniform_expr_body_with_locals(body, locals)
            }
            ExprKind::Range { start, end, .. } => {
                self.block_uniform_expr_with_locals(start, locals)
                    && self.block_uniform_expr_with_locals(end, locals)
            }
            _ => false,
        }
    }
    fn block_uniform_expr_body_with_locals(
        &self,
        body: &[Stmt],
        initial_locals: &std::collections::HashSet<String>,
    ) -> bool {
        let mut locals = initial_locals.clone();
        for stmt in body {
            match stmt {
                Stmt::Let(local)
                    if matches!(local.binding, BindingKind::Let)
                        && local.value.as_ref().is_some_and(|value| {
                            self.block_uniform_expr_with_locals(value, &locals)
                        }) =>
                {
                    locals.insert(local.name.clone());
                }
                Stmt::Expr(value) if self.block_uniform_expr_with_locals(value, &locals) => {}
                _ => return false,
            }
        }
        true
    }
    #[allow(clippy::too_many_arguments)]
    fn emit_collective_guard_clauses(
        &mut self,
        k: &KernelDecl,
        clauses: &[CondClause],
        clause_index: usize,
        else_body: &[Stmt],
        suffix: &[Stmt],
        indent: usize,
        replayed: &[Stmt],
    ) {
        let Some(clause) = clauses.get(clause_index) else {
            self.emit_following_collective_controls(k, suffix, indent, replayed);
            return;
        };
        let (condition, fallback_pattern) = match clause {
            CondClause::Let(name, value) => {
                (format!("Some({name})"), Some(self.expr(value)))
            }
            CondClause::LetPat(pattern, value) => {
                (simd_pattern(pattern), Some(self.expr(value)))
            }
            CondClause::Expr(value) => (self.expr(value), None),
        };
        let mut bound_names = std::collections::HashSet::new();
        match clause {
            CondClause::Let(name, _) => {
                bound_names.insert(name.clone());
            }
            CondClause::LetPat(pattern, _) => {
                collect_simd_pattern_bind_names(pattern, &mut bound_names);
            }
            CondClause::Expr(_) => {}
        }
        let inherited_bound_names = bound_names
            .iter()
            .filter(|name| self.collective_uniform_locals.contains(*name))
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        if let Some(value) = fallback_pattern {
            self.line(indent, &format!("match {value} {{"));
            self.line(indent + 1, &format!("{condition} => {{"));
            self.collective_uniform_locals.extend(bound_names.iter().cloned());
            self.emit_collective_guard_clauses(
                k,
                clauses,
                clause_index + 1,
                else_body,
                suffix,
                indent + 2,
                replayed,
            );
            for name in &bound_names {
                if !inherited_bound_names.contains(name) {
                    self.collective_uniform_locals.remove(name);
                }
            }
            self.line(indent + 1, "},");
            self.line(indent + 1, "_ => {");
            self.emit_following_collective_controls(k, else_body, indent + 2, replayed);
            self.line(indent + 1, "},");
            self.line(indent, "}");
        } else {
            self.line(indent, &format!("if {condition} {{"));
            self.collective_uniform_locals.extend(bound_names.iter().cloned());
            self.emit_collective_guard_clauses(
                k,
                clauses,
                clause_index + 1,
                else_body,
                suffix,
                indent + 1,
                replayed,
            );
            for name in &bound_names {
                if !inherited_bound_names.contains(name) {
                    self.collective_uniform_locals.remove(name);
                }
            }
            self.line(indent, "} else {");
            self.emit_following_collective_controls(k, else_body, indent + 1, replayed);
            self.line(indent, "}");
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn emit_collective_if_let_clauses(
        &mut self,
        k: &KernelDecl,
        clauses: &[CondClause],
        clause_index: usize,
        then_body: &[Stmt],
        remaining: &[IfLetBranch],
        else_body: Option<&[Stmt]>,
        indent: usize,
        replayed: &[Stmt],
    ) {
        let Some(clause) = clauses.get(clause_index) else {
            self.emit_following_collective_controls(k, then_body, indent, replayed);
            return;
        };
        let condition = match clause {
            CondClause::Let(name, value) => {
                let value = self.expr(value);
                format!("let Some({name}) = {value}")
            }
            CondClause::LetPat(pattern, value) => {
                let value = self.expr(value);
                format!("let {} = {value}", simd_pattern(pattern))
            }
            CondClause::Expr(value) => self.expr(value),
        };
        self.line(indent, &format!("if {condition} {{"));
        let mut bound_names = std::collections::HashSet::new();
        match clause {
            CondClause::Let(name, _) => {
                bound_names.insert(name.clone());
            }
            CondClause::LetPat(pattern, _) => {
                collect_simd_pattern_bind_names(pattern, &mut bound_names);
            }
            CondClause::Expr(_) => {}
        }
        let inherited_bound_names = bound_names
            .iter()
            .filter(|name| self.collective_uniform_locals.contains(*name))
            .cloned()
            .collect::<std::collections::HashSet<_>>();
        self.collective_uniform_locals.extend(bound_names.iter().cloned());
        self.emit_collective_if_let_clauses(
            k,
            clauses,
            clause_index + 1,
            then_body,
            remaining,
            else_body,
            indent + 1,
            replayed,
        );
        for name in bound_names {
            if !inherited_bound_names.contains(&name) {
                self.collective_uniform_locals.remove(&name);
            }
        }
        if !remaining.is_empty() || else_body.is_some() {
            self.line(indent, "} else {");
            self.emit_collective_if_let_fallback(
                k,
                remaining,
                else_body,
                indent + 1,
                replayed,
            );
        }
        self.line(indent, "}");
    }
    fn emit_collective_if_let_fallback(
        &mut self,
        k: &KernelDecl,
        remaining: &[IfLetBranch],
        else_body: Option<&[Stmt]>,
        indent: usize,
        replayed: &[Stmt],
    ) {
        if let Some((branch, rest)) = remaining.split_first() {
            self.emit_collective_if_let_clauses(
                k,
                &branch.clauses,
                0,
                &branch.body,
                rest,
                else_body,
                indent,
                replayed,
            );
        } else if let Some(else_body) = else_body {
            self.emit_following_collective_controls(k, else_body, indent, replayed);
        }
    }
    fn expr_body_as(&mut self, body: &[Stmt], expected: Option<&str>) -> String {
        let mut parts = Vec::new();
        for (index, stmt) in body.iter().enumerate() {
            match stmt {
                Stmt::Expr(e) if index + 1 == body.len() => parts.push(self.expr_as(e, expected)),
                Stmt::Expr(e) => parts.push(self.expr(e)),
                Stmt::Let(v) => {
                    let value = v
                        .value
                        .as_ref()
                        .map(|e| self.expr(e))
                        .unwrap_or_else(|| "Default::default()".into());
                    let value = match &v.ty {
                        Some(t) if simd_needs_as_cast(t) => {
                            format!("({value}) as {}", ty(t))
                        }
                        _ => value,
                    };
                    parts.push(format!(
                        "let {}{}{} = {}",
                        if v.binding.is_mutable() { "mut " } else { "" },
                        v.name,
                        v.ty.as_ref()
                            .map(|t| format!(": {}", ty(t)))
                            .unwrap_or_default(),
                        value
                    ));
                }
                other => {
                    let (line, col) = stmt_position(other);
                    self.errors.push(crate::errors::SourceError::at(
                        "this statement is not supported in a SIMD kernel expression block",
                        line,
                        col,
                    ));
                    parts.push("Default::default()".into());
                }
            }
        }
        parts.join("; ")
    }
    /// Optional-typed destinations take plain values: `some` is implicit, so wrap a
    /// value known to be non-optional (and each branch of an `if` expression).
    fn implicit_some(&self, expr: &Expr) -> Expr {
        let wrap = |inner: &Expr| Expr {
            kind: ExprKind::Call(
                Box::new(Expr { kind: ExprKind::Var("some".into()), line: inner.line, col: inner.col, len: inner.len }),
                vec![Arg { label: None, value: inner.clone(), spread: false, default_rest: false }],
            ),
            line: inner.line,
            col: inner.col,
            len: inner.len,
        };
        match &expr.kind {
            ExprKind::Nil | ExprKind::Else(_, _) => expr.clone(),
            ExprKind::Call(callee, _) if matches!(&callee.kind, ExprKind::Var(name) if name == "some") => expr.clone(),
            ExprKind::If(value) => {
                let mut value = value.clone();
                let tail = |body: &mut Vec<Stmt>, this: &Self| {
                    if let Some(Stmt::Expr(last)) = body.last_mut() {
                        *last = this.implicit_some(last);
                    }
                };
                for (_, body) in &mut value.branches {
                    tail(body, self);
                }
                if let Some(body) = &mut value.else_body {
                    tail(body, self);
                }
                Expr { kind: ExprKind::If(value), ..expr.clone() }
            }
            ExprKind::Int(_) | ExprKind::UInt64(_) | ExprKind::Float(_) | ExprKind::Bool(_)
            | ExprKind::Str(_) | ExprKind::StringInterp(_) | ExprKind::BinOp(_, _, _)
            | ExprKind::UnaryOp(_, _) | ExprKind::Cast(_, _) => wrap(expr),
            ExprKind::Var(name) if self.locals.contains(name) || self.fields.contains(name) => {
                match self.value_types.get(name) {
                    Some(kind) if kind.starts_with("Option<") => expr.clone(),
                    _ => wrap(expr),
                }
            }
            ExprKind::Field(object, _)
                if matches!(&object.kind, ExprKind::DotIdent(name)
                    if matches!(name.as_str(), "thread" | "block" | "block_dim" | "grid_dim")) =>
            {
                wrap(expr)
            }
            _ => match self.expr_type(expr) {
                Some(kind) if !kind.starts_with("Option<") => wrap(expr),
                _ => expr.clone(),
            },
        }
    }
    /// True for initializers that are optional by construction (`nil`, `some(..)`, or an
    /// `if` expression with such a branch), so an unannotated local keeps that type.
    fn looks_optional(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Nil => true,
            ExprKind::Call(callee, _) => matches!(&callee.kind, ExprKind::Var(name) if name == "some"),
            ExprKind::If(value) => value
                .branches
                .iter()
                .map(|(_, body)| body)
                .chain(value.else_body.iter())
                .any(|body| matches!(body.last(), Some(Stmt::Expr(last)) if self.looks_optional(last))),
            _ => false,
        }
    }
    fn expr_as(&mut self, expr: &Expr, expected: Option<&str>) -> String {
        match (&expr.kind, expected) {
            (ExprKind::Int(value), Some("f32")) => format!("{value}.0f32"),
            (ExprKind::Int(value), Some("f64")) => format!("{value}.0f64"),
            (ExprKind::Float(value), Some("f32")) => format!("{value}f32"),
            (ExprKind::Float(value), Some("f64")) => format!("{value}f64"),
            _ => self.expr(expr),
        }
    }
    fn body_type(&self, body: &[Stmt]) -> Option<String> {
        match body.last() {
            Some(Stmt::Expr(expr)) => self.expr_type(expr),
            _ => None,
        }
    }
    /// Free-function math in kernels (`sin(x)`, `sqrt(x)`, `pow(a, b)`, `min(a, b)`, ...),
    /// lowered to the matching Rust numeric method at the argument's own width.
    fn free_math_call(&mut self, name: &str, args: &[Arg]) -> Option<String> {
        if args.iter().any(|a| a.label.is_some()) {
            return None;
        }
        let float_ty = |this: &Self, a: &Expr| match this.expr_type(a).as_deref() {
            Some("f32") => Some("f32"),
            _ => None,
        };
        // A bare literal operand has no type to infer a method receiver from.
        macro_rules! operand {
            ($a:expr, $forced:expr, $int_literal:expr) => {{
                let (a, forced, int_literal): (&Expr, Option<&str>, &str) = ($a, $forced, $int_literal);
                let text = self.expr(a);
                match (forced, &a.kind) {
                    (Some(t), _) => format!("(({text}) as {t})"),
                    (None, ExprKind::Float(_)) => format!("({text} as f64)"),
                    (None, ExprKind::Int(_)) => match int_literal {
                        "" => text,
                        cast => format!("({text} as {cast})"),
                    },
                    _ => format!("({text})"),
                }
            }};
        }
        let unary = match name {
            "sin" => "sin", "cos" => "cos", "tan" => "tan",
            "asin" => "asin", "acos" => "acos", "atan" => "atan",
            "sinh" => "sinh", "cosh" => "cosh", "tanh" => "tanh",
            "exp" => "exp", "exp2" => "exp2", "sqrt" => "sqrt", "cbrt" => "cbrt",
            "floor" => "floor", "ceil" => "ceil", "round" => "round", "trunc" => "trunc",
            "log" | "ln" => "ln", "log2" => "log2", "log10" => "log10",
            _ => "",
        };
        if !unary.is_empty() && args.len() == 1 {
            let forced = float_ty(self, &args[0].value);
            return Some(format!("{}.{unary}()", operand!(&args[0].value, forced, "f64")));
        }
        match (name, args.len()) {
            ("abs", 1) => {
                let forced = float_ty(self, &args[0].value);
                Some(format!("{}.abs()", operand!(&args[0].value, forced, "f64")))
            }
            ("pow", 2) | ("atan2", 2) | ("min", 2) | ("max", 2) => {
                let forced = float_ty(self, &args[0].value).or_else(|| float_ty(self, &args[1].value));
                let method = if name == "pow" { "powf" } else { name };
                let literal = if name == "min" || name == "max" { "" } else { "_" };
                let a = operand!(&args[0].value, forced, if literal.is_empty() { "" } else { "f64" });
                let b = operand!(&args[1].value, forced, literal);
                Some(format!("{a}.{method}({b})"))
            }
            _ => None,
        }
    }
    fn expr_type(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Field(object, field)
                if matches!(gpu_builtin_name(object), Some("thread" | "block" | "block_dim" | "grid_dim" | "blockDim" | "gridDim"))
                    && matches!(field.as_str(), "x" | "y" | "z") =>
            {
                Some("i32".into())
            }
            ExprKind::Field(object, field)
                if gpu_builtin_name(object) == Some("warp")
                    && matches!(field.as_str(), "size" | "lane") =>
            {
                Some("i32".into())
            }
            ExprKind::BinOp(op, left, right)
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem) =>
            {
                let literal = |e: &Expr| matches!(&e.kind, ExprKind::Int(_) | ExprKind::Float(_));
                match (self.expr_type(left), self.expr_type(right)) {
                    (Some(l), _) if literal(right) && numeric_rank(&l).is_some() => Some(l),
                    (_, Some(r)) if literal(left) && numeric_rank(&r).is_some() => Some(r),
                    (Some(l), Some(r)) => common_numeric(&l, &r),
                    (Some(l), None) if numeric_rank(&l).is_some() && matches!(&right.kind, ExprKind::Int(_) | ExprKind::Float(_)) => Some(l),
                    (None, Some(r)) if numeric_rank(&r).is_some() && matches!(&left.kind, ExprKind::Int(_) | ExprKind::Float(_)) => Some(r),
                    _ => None,
                }
            }
            ExprKind::MethodCall(receiver, method, args)
                if is_warp_receiver(receiver) && is_warp_shuffle(method) && args.len() == 2 =>
            {
                self.expr_type(&args[0].value)
            }
            ExprKind::Int(_) => Some("i32".into()),
            ExprKind::UInt64(_) => Some("u64".into()),
            ExprKind::Float(_) => Some("f64".into()),
            ExprKind::Str(_) | ExprKind::StringInterp(_) => Some("String".into()),
            ExprKind::Bool(_) => Some("bool".into()),
            ExprKind::Array(values) => values.first().and_then(|value| self.expr_type(value)),
            ExprKind::Var(name) => self
                .value_types
                .get(name)
                .or_else(|| self.global_consts.get(name))
                .cloned(),
            ExprKind::Index(object, _) | ExprKind::LabeledIndex(object, _) => {
                if let ExprKind::Var(name) = &object.kind {
                    self.value_types.get(name).cloned()
                } else {
                    None
                }
            }
            ExprKind::Cast(_, value_ty) => Some(ty(value_ty)),
            ExprKind::Call(callee, _) => match &callee.kind {
                ExprKind::Var(name) => primitive_cast_type(name).map(str::to_string),
                _ => None,
            },
            ExprKind::MethodCall(object, method, _)
                if matches!(
                    method.as_str(),
                    "trim"
                        | "trimStart"
                        | "trimEnd"
                        | "toUpperCase"
                        | "uppercased"
                        | "upper"
                        | "to_upper"
                        | "toUpper"
                        | "toLowerCase"
                        | "lowercased"
                        | "lower"
                        | "to_lower"
                        | "toLower"
                        | "slice"
                        | "replace"
                        | "replaceAll"
                        | "repeat"
                ) && self.expr_type(object).as_deref() == Some("String") =>
            {
                Some("String".into())
            }
            ExprKind::MethodCall(object, method, _)
                if matches!(method.as_str(), "split" | "chars")
                    && self.expr_type(object).as_deref() == Some("String") =>
            {
                Some("String".into())
            }
            ExprKind::MethodCall(object, method, _)
                if method == "parseInt"
                    && self.expr_type(object).as_deref() == Some("String") =>
            {
                Some("i32".into())
            }
            ExprKind::MethodCall(object, method, _)
                if method == "parseFloat"
                    && self.expr_type(object).as_deref() == Some("String") =>
            {
                Some("f64".into())
            }
            ExprKind::MethodCall(object, method, _)
                if method == "parseFloat32"
                    && self.expr_type(object).as_deref() == Some("String") =>
            {
                Some("f32".into())
            }
            ExprKind::MethodCall(object, method, _) if method == "filter" => self.expr_type(object),
            ExprKind::MethodCall(_, method, args) if method == "map" => args
                .first()
                .and_then(|arg| match &arg.value.kind {
                    ExprKind::Closure(_, _, ClosureBody::Expr(value), _, _) => {
                        self.expr_type(value)
                    }
                    _ => None,
                }),
            ExprKind::Pipe(object, method, _) if method == "filter" => self.expr_type(object),
            ExprKind::Pipe(_, method, args) if method == "map" => args
                .first()
                .and_then(|arg| match &arg.value.kind {
                    ExprKind::Closure(_, _, ClosureBody::Expr(value), _, _) => {
                        self.expr_type(value)
                    }
                    _ => None,
                }),
            ExprKind::Pipe(object, method, _)
                if matches!(
                    method.as_str(),
                    "trim"
                        | "trimStart"
                        | "trimEnd"
                        | "toUpperCase"
                        | "uppercased"
                        | "upper"
                        | "to_upper"
                        | "toUpper"
                        | "toLowerCase"
                        | "lowercased"
                        | "lower"
                        | "to_lower"
                        | "toLower"
                        | "slice"
                        | "replace"
                        | "replaceAll"
                        | "repeat"
                ) && self.expr_type(object).as_deref() == Some("String") =>
            {
                Some("String".into())
            }
            ExprKind::UnaryOp(_, value) => self.expr_type(value),
            ExprKind::BinOp(op, left, right) => {
                if matches!(
                    op,
                    BinOp::Eq
                        | BinOp::NotEq
                        | BinOp::RefEq
                        | BinOp::Lt
                        | BinOp::LtEq
                        | BinOp::Gt
                        | BinOp::GtEq
                        | BinOp::And
                        | BinOp::Or
                        | BinOp::Is
                        | BinOp::IsNot
                ) {
                    Some("bool".into())
                } else {
                    let left = self.expr_type(left);
                    let right = self.expr_type(right);
                    if left.as_deref() == Some("f32") || right.as_deref() == Some("f32") {
                        Some("f32".into())
                    } else {
                        left.or(right)
                    }
                }
            }
            ExprKind::If(value) => value
                .branches
                .iter()
                .filter_map(|(_, body)| self.body_type(body))
                .chain(
                    value
                        .else_body
                        .iter()
                        .filter_map(|body| self.body_type(body)),
                )
                .fold(None, |found, kind| {
                    match (found.as_deref(), kind.as_str()) {
                        (_, "f32") => Some("f32".into()),
                        (None, "f64") => Some("f64".into()),
                        _ => found,
                    }
                }),
            _ => None,
        }
    }
    /// Atomic type and element type when `target` indexes an `'actor'global` /
    /// `'actor'unified` array of 32/64-bit integers (the widths real hardware atomics cover).
    fn hardware_atomic(&self, target: &Expr) -> Option<(&'static str, String)> {
        let (ExprKind::Index(object, _) | ExprKind::LabeledIndex(object, _)) = &target.kind else { return None };
        let ExprKind::Var(name) = &object.kind else { return None };
        let element = self.atomic_fields.get(name)?;
        let atomic = match element.as_str() {
            "i32" => "AtomicI32",
            "u32" => "AtomicU32",
            "i64" => "AtomicI64",
            "u64" => "AtomicU64",
            _ => return None,
        };
        Some((atomic, element.clone()))
    }
    fn lvalue(&mut self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::Var(v) => {
                if let Some(outer) = self.while_let_redirect.get(v) {
                    outer.clone()
                } else if v.starts_with("__") {
                    v.clone()
                } else if self.fields.contains(v) {
                    format!("self.{v}")
                } else {
                    v.clone()
                }
            }
            ExprKind::Index(a, b) => format!("{}[({}) as usize]", self.lvalue(a), self.expr(b)),
            ExprKind::LabeledIndex(a, args) => self.labeled_index(a, args),
            ExprKind::Field(a, b) => format!("{}.{}", self.lvalue(a), b),
            _ => self.expr(e),
        }
    }
    fn labeled_index(&mut self, object: &Expr, args: &[Arg]) -> String {
        let ExprKind::Var(field) = &object.kind else {
            self.errors.push(crate::errors::SourceError::at(
                "SIMD labeled indexing currently requires a direct kernel field",
                object.line,
                object.col,
            ));
            return "/* rejected labeled index */ 0".into();
        };
        let axes = self
            .field_axes
            .get(field)
            .or_else(|| self.local_axes.get(field))
            .cloned();
        let Some(axes) = axes else {
            self.errors.push(crate::errors::SourceError::at(
                format!("array '{}' has no labeled dimensions", field),
                object.line,
                object.col,
            ));
            return "/* rejected labeled index */ 0".into();
        };
        let mut terms = Vec::new();
        let mut stride = "1".to_string();
        for axis in &axes {
            let Some(arg) = args
                .iter()
                .find(|arg| arg.label.as_deref() == Some(axis.label.as_str()))
            else {
                self.errors.push(crate::errors::SourceError::at(
                    format!("missing '{}' axis in labeled index", axis.label),
                    object.line,
                    object.col,
                ));
                continue;
            };
            terms.push(format!(
                "(({}) as usize * ({}))",
                self.expr(&arg.value),
                stride
            ));
            if let Some(size) = &axis.size {
                stride = format!(
                    "({stride}) * (({}) as usize)",
                    crate::transpiler::helpers::const_expr_to_c_like(&size.0)
                );
            }
        }
        let object = if self.fields.contains(field) {
            format!("self.{field}")
        } else {
            field.clone()
        };
        format!("{object}[{}]", terms.join(" + "))
    }
}

fn simd_top_level_const_type(value: &LetStmt) -> Option<String> {
    if value.binding.is_mutable() {
        return None;
    }
    fn scalar_type(value_ty: &Type) -> Option<String> {
        let supported = match value_ty {
            Type::Int
            | Type::Uint
            | Type::Uint8
            | Type::Float32
            | Type::Float64
            | Type::Bool
            | Type::Int8
            | Type::Int16
            | Type::Int32
            | Type::Int64
            | Type::Int128
            | Type::Uint16
            | Type::Uint32
            | Type::Uint64
            | Type::Uint128 => true,
            Type::Named(name) => matches!(
                name.as_str(),
                "int"
                    | "uint"
                    | "uint8"
                    | "float"
                    | "float32"
                    | "float64"
                    | "bool"
                    | "int8"
                    | "int16"
                    | "int32"
                    | "int64"
                    | "int128"
                    | "uint16"
                    | "uint32"
                    | "uint64"
                    | "uint128"
                    | "i8"
                    | "i16"
                    | "i32"
                    | "i64"
                    | "i128"
                    | "isize"
                    | "u8"
                    | "u16"
                    | "u32"
                    | "u64"
                    | "u128"
                    | "usize"
            ),
            Type::Qualified(inner, _) | Type::Mut(inner) => return scalar_type(inner),
            _ => false,
        };
        supported.then(|| ty(value_ty))
    }
    fn scalar_literal(expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Int(_) => Some("i32".into()),
            ExprKind::UInt64(_) => Some("u64".into()),
            ExprKind::Float(_) => Some("f64".into()),
            ExprKind::Bool(_) => Some("bool".into()),
            ExprKind::UnaryOp(_, inner) => scalar_literal(inner),
            _ => None,
        }
    }
    value
        .ty
        .as_ref()
        .and_then(scalar_type)
        .or_else(|| value.value.as_ref().and_then(scalar_literal))
}

fn collect_simd_pattern_bind_names(
    pattern: &Pattern,
    names: &mut std::collections::HashSet<String>,
) {
    match pattern {
        Pattern::Bind(name) => {
            names.insert(name.clone());
        }
        Pattern::Variant(_, fields) | Pattern::Tuple(fields) => {
            for field in fields {
                collect_simd_pattern_bind_names(field, names);
            }
        }
        Pattern::Some(inner) => collect_simd_pattern_bind_names(inner, names),
        Pattern::Wildcard | Pattern::Lit(_) | Pattern::None => {}
    }
}

fn tuple_optional_pattern_binding(pattern: &Pattern) -> Option<&str> {
    match pattern {
        Pattern::Some(inner) => match inner.as_ref() {
            Pattern::Bind(name) => Some(name),
            _ => None,
        },
        _ => None,
    }
}

fn collective_while_let_bound_names(
    stmt: &WhileLetStmt,
) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    if let Some(pattern) = &stmt.pattern {
        collect_simd_pattern_bind_names(pattern, &mut names);
    } else {
        names.insert(stmt.name.clone());
    }
    names
}

fn simd_collective_while_let_pattern(stmt: &WhileLetStmt) -> String {
    stmt.pattern
        .as_ref()
        .map(simd_pattern)
        .unwrap_or_else(|| format!("Some({})", stmt.name))
}

fn simd_pattern(pattern: &Pattern) -> String {
    match pattern {
        Pattern::Wildcard => "_".into(),
        Pattern::Bind(name) => name.clone(),
        Pattern::None => "None".into(),
        Pattern::Some(inner) => format!("Some({})", simd_pattern(inner)),
        Pattern::Variant(name, fields) => {
            if fields.is_empty() {
                name.clone()
            } else {
                let fields = fields
                    .iter()
                    .map(simd_pattern)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{name}({fields})")
            }
        }
        Pattern::Tuple(values) => {
            let values = values
                .iter()
                .map(simd_pattern)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({values})")
        }
        Pattern::Lit(value) => match value {
            LitPattern::Int(value) => value.to_string(),
            LitPattern::Float(value) => value.to_string(),
            LitPattern::Str(value) => format!("{:?}", value),
            LitPattern::Bool(value) => value.to_string(),
            LitPattern::Nil => "None".into(),
        },
    }
}

/// Bit width and kind rank of a numeric Rust type name: floats outrank ints.
fn numeric_rank(name: &str) -> Option<(u8, u8, bool)> {
    Some(match name {
        "i8" => (0, 8, true),
        "i16" => (0, 16, true),
        "i32" => (0, 32, true),
        "i64" => (0, 64, true),
        "i128" => (0, 128, true),
        "u8" => (0, 8, false),
        "u16" => (0, 16, false),
        "u32" => (0, 32, false),
        "u64" => (0, 64, false),
        "u128" => (0, 128, false),
        "f32" => (1, 32, true),
        "f64" => (1, 64, true),
        _ => return None,
    })
}

/// C-style usual arithmetic conversion of two numeric type names.
fn common_numeric(left: &str, right: &str) -> Option<String> {
    let (l, r) = (numeric_rank(left)?, numeric_rank(right)?);
    if left == right {
        return Some(left.to_string());
    }
    let pick = if l.0 != r.0 {
        if l.0 > r.0 { left } else { right }
    } else if l.1 != r.1 {
        if l.1 > r.1 { left } else { right }
    } else if !l.2 {
        left
    } else {
        right
    };
    Some(pick.to_string())
}

thread_local! {
    /// Set while the shared host pipeline runs for `--target simd`, so its buffer
    /// element casts use the CPU kernel's own widths (`float` = f64, not wgpu's f32).
    static SIMD_HOST_TYPES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn simd_scalar_ty(t: &Type) -> Option<&'static str> {
    if !SIMD_HOST_TYPES.with(|flag| flag.get()) {
        return None;
    }
    let name = numeric_ty(t).or_else(|| matches!(base(t), Type::Bool).then(|| "bool".to_string()))?;
    ["i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "f32", "f64", "bool"]
        .into_iter()
        .find(|candidate| *candidate == name)
}

/// `.thread` / `.warp` / ... — spelled `Field(Var(gpu), name)`, `DotIdent(name)` or a bare name.
fn gpu_builtin_name(expr: &Expr) -> Option<&str> {
    match &expr.kind {
        ExprKind::DotIdent(name) | ExprKind::Var(name) => Some(name),
        ExprKind::Field(root, name) if matches!(&root.kind, ExprKind::Var(v) if v == "gpu") => Some(name),
        _ => None,
    }
}

/// Names of buffers the statements store into (`name[i] = ...`).
fn written_buffers(body: &[Stmt]) -> std::collections::HashSet<String> {
    let mut written = std::collections::HashSet::new();
    let mut owned = body.to_vec();
    let mut on_expr = |e: &mut Expr| {
        if let ExprKind::Assign(target, _) = &e.kind {
            if let ExprKind::Index(object, _) | ExprKind::LabeledIndex(object, _) = &target.kind {
                if let ExprKind::Var(name) = &object.kind {
                    written.insert(name.clone());
                }
            }
        }
    };
    let mut on_stmt = |_: &mut Stmt| {};
    crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut owned);
    written
}

fn expr_reads_any(expr: &Expr, names: &std::collections::HashSet<String>) -> bool {
    let mut found = false;
    let mut owned = expr.clone();
    let mut on_expr = |e: &mut Expr| {
        if let ExprKind::Index(object, _) | ExprKind::LabeledIndex(object, _) = &e.kind {
            if matches!(&object.kind, ExprKind::Var(name) if names.contains(name)) {
                found = true;
            }
        }
    };
    let mut on_stmt = |_: &mut Stmt| {};
    crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.expr(&mut owned);
    found
}

fn float_literal(value: f64) -> String {
    let text = format!("{value}");
    if text.contains(['.', 'e', 'E']) || !value.is_finite() {
        text
    } else {
        format!("{text}.0")
    }
}

/// Device width of a numeric scalar type (`int` = i32), `None` for non-numerics.
fn numeric_ty(t: &Type) -> Option<String> {
    match base(t) {
        Type::Int | Type::Uint | Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64
        | Type::Uint8 | Type::Uint16 | Type::Uint32 | Type::Uint64 | Type::Float32
        | Type::Float64 => Some(ty(t)),
        Type::Named(n) if matches!(n.as_str(), "int" | "uint" | "float" | "float32" | "float64"
            | "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32" | "uint64") => Some(ty(t)),
        _ => None,
    }
}

/// Host width of a numeric scalar parameter (`int` = isize, `uint` = usize).
fn host_numeric_ty(t: &Type) -> Option<String> {
    match base(t) {
        Type::Int => Some("isize".into()),
        Type::Uint => Some("usize".into()),
        Type::Named(n) if n == "int" => Some("isize".into()),
        Type::Named(n) if n == "uint" => Some("usize".into()),
        _ => numeric_ty(t),
    }
}

/// Bare `'actor` storage with no explicit `sync` keeps the GPU backends' implicit
/// barrier discipline: one barrier before the first loop and another at the top
/// of every loop iteration that touches shared storage.
fn auto_sync_kernel(k: &KernelDecl) -> Option<KernelDecl> {
    let names = k
        .fields
        .iter()
        .filter(|field| matches!(field.qual, GpuQual::Actor))
        .map(|field| field.name.as_str())
        .collect::<Vec<_>>();
    let method = entry_method(k)?;
    if names.is_empty()
        || method
            .body
            .iter()
            .any(|stmt| is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt))
    {
        return None;
    }
    fn references(stmts: &[Stmt], names: &[&str]) -> bool {
        let mut found = false;
        let mut owned = stmts.to_vec();
        let mut on_expr = |e: &mut Expr| {
            if matches!(&e.kind, ExprKind::Var(name) if names.contains(&name.as_str())) {
                found = true;
            }
        };
        let mut on_stmt = |_: &mut Stmt| {};
        crate::ast_walk::Walker { expr: &mut on_expr, stmt: &mut on_stmt }.stmts(&mut owned);
        found
    }
    fn nested(stmt: &Stmt, names: &[&str]) -> Stmt {
        let loop_body = |body: &[Stmt]| {
            let mut out = Vec::new();
            if references(body, names) {
                out.push(Stmt::Comment("sync".into()));
            }
            out.extend(body.iter().map(|stmt| nested(stmt, names)));
            out
        };
        match stmt {
            Stmt::While(w) => {
                let mut w = w.clone();
                w.body = loop_body(&w.body);
                Stmt::While(w)
            }
            Stmt::For(f) => {
                let mut f = f.clone();
                f.body = loop_body(&f.body);
                Stmt::For(f)
            }
            Stmt::If(i) => {
                let mut i = i.clone();
                for (_, branch) in &mut i.branches {
                    *branch = branch.iter().map(|stmt| nested(stmt, names)).collect();
                }
                if let Some(branch) = &mut i.else_body {
                    *branch = branch.iter().map(|stmt| nested(stmt, names)).collect();
                }
                Stmt::If(i)
            }
            other => other.clone(),
        }
    }
    let split = crate::transpiler::helpers::first_loop_index(&method.body);
    let mut body = method.body[..split]
        .iter()
        .map(|stmt| nested(stmt, &names))
        .collect::<Vec<_>>();
    if split < method.body.len() {
        body.push(Stmt::Comment("sync".into()));
    }
    body.extend(method.body[split..].iter().map(|stmt| nested(stmt, &names)));
    let mut rewritten = k.clone();
    rewritten.methods[0].body = body;
    Some(rewritten)
}

/// The anonymous `def ()` entry point; any other method is a device helper.
fn entry_method(k: &KernelDecl) -> Option<&FnDecl> {
    k.methods.iter().find(|method| method.name.is_empty())
}

fn is_cooperative_candidate_error(message: &str) -> bool {
    message.starts_with("`sync` is not yet supported")
        || message.starts_with("SIMD warp shuffles require")
}

fn is_simd_barrier_stmt(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Comment(comment) => comment == "sync",
        Stmt::Expr(expr) => matches!(
            &expr.kind,
            ExprKind::MethodCall(receiver, method, args)
                if is_warp_receiver(receiver) && method == "sync" && args.is_empty()
        ),
        _ => false,
    }
}

fn contains_nested_simd_barrier(stmt: &Stmt) -> bool {
    let body_has_barrier = |body: &[Stmt]| {
        body.iter()
            .any(|stmt| is_simd_barrier_stmt(stmt) || contains_nested_simd_barrier(stmt))
    };
    match stmt {
        Stmt::If(value) => {
            value
                .branches
                .iter()
                .any(|(_, body)| body_has_barrier(body))
                || value
                    .else_body
                    .as_ref()
                    .is_some_and(|body| body_has_barrier(body))
        }
        Stmt::IfLet(value) => {
            body_has_barrier(&value.then_body)
                || value
                    .elif_branches
                    .iter()
                    .any(|branch| body_has_barrier(&branch.body))
                || value
                    .else_body
                    .as_ref()
                    .is_some_and(|body| body_has_barrier(body))
        }
        Stmt::Match(value) => value.arms.iter().any(|arm| {
            matches!(&arm.body, MatchBody::Block(body) if body_has_barrier(body))
        }),
        Stmt::Guard(value) => body_has_barrier(&value.else_body),
        Stmt::For(value) => body_has_barrier(&value.body),
        Stmt::While(value) => body_has_barrier(&value.body),
        Stmt::WhileLet(value) => body_has_barrier(&value.body),
        Stmt::DoWhile(value) => body_has_barrier(&value.body),
        Stmt::Loop(value) => body_has_barrier(&value.body),
        _ => false,
    }
}

fn canonical_collective_loop_break_condition(stmt: &Stmt) -> Option<&Expr> {
    let Stmt::If(if_stmt) = stmt else { return None };
    if if_stmt.branches.len() != 1 || if_stmt.else_body.is_some() {
        return None;
    }
    let (condition, body) = &if_stmt.branches[0];
    matches!(body.as_slice(), [Stmt::Break(_, None)]).then_some(condition)
}

fn expr_mentions_var(expr: &Expr, sought: &str) -> bool {
    match &expr.kind {
        ExprKind::Var(name) => name == sought,
        ExprKind::BinOp(_, left, right)
        | ExprKind::Assign(left, right)
        | ExprKind::QuestionAssign(left, right) => {
            expr_mentions_var(left, sought) || expr_mentions_var(right, sought)
        }
        ExprKind::UnaryOp(_, value)
        | ExprKind::Cast(value, _)
        | ExprKind::RelabelCast(value, _)
        | ExprKind::Field(value, _) => expr_mentions_var(value, sought),
        ExprKind::Index(value, index) => {
            expr_mentions_var(value, sought) || expr_mentions_var(index, sought)
        }
        ExprKind::Call(callee, args) | ExprKind::MethodCall(callee, _, args) => {
            expr_mentions_var(callee, sought)
                || args.iter().any(|arg| expr_mentions_var(&arg.value, sought))
        }
        ExprKind::If(if_expr) => {
            let body_mentions = |body: &[Stmt]| {
                body.iter().any(|stmt| match stmt {
                    Stmt::Expr(value) => expr_mentions_var(value, sought),
                    Stmt::Let(local) => local
                        .value
                        .as_ref()
                        .is_some_and(|value| expr_mentions_var(value, sought)),
                    _ => false,
                })
            };
            if_expr.branches.iter().any(|(condition, body)| {
                expr_mentions_var(condition, sought) || body_mentions(body)
            }) || if_expr
                .else_body
                .as_ref()
                .is_some_and(|body| body_mentions(body))
        }
        _ => false,
    }
}

fn stmt_position(stmt: &Stmt) -> (usize, usize) {
    match stmt {
        Stmt::Let(s) => (s.line, s.col),
        Stmt::LetDestructure(s) => (s.line, s.col),
        Stmt::Return(s) => (s.line, s.col),
        Stmt::Throw(s) => (s.line, s.col),
        Stmt::If(s) => (s.line, s.col),
        Stmt::IfLet(s) => (s.line, s.col),
        Stmt::Match(s) => (s.line, s.col),
        Stmt::While(s) => (s.line, s.col),
        Stmt::WhileLet(s) => (s.line, s.col),
        Stmt::DoWhile(s) => (s.line, s.col),
        Stmt::Loop(s) => (s.line, s.col),
        Stmt::For(s) => (s.line, s.col),
        Stmt::Guard(s) => (s.line, s.col),
        Stmt::Try(s) => (s.line, s.col),
        Stmt::Expr(e) | Stmt::Wait(e, _) | Stmt::Yield(e, _) => (e.line, e.col),
        Stmt::Fn(s) => (s.line, s.col),
        Stmt::Struct(s) => (s.line, s.col),
        Stmt::Enum(s) => (s.line, s.col),
        Stmt::Mod(s) => (s.line, s.col),
        Stmt::Alias(s) => (s.line, s.col),
        Stmt::KernelBlock(s) => (s.line, s.col),
        Stmt::With(s) => (s.line, s.col),
        Stmt::Break(line, _) | Stmt::Continue(line) => (*line, 0),
        Stmt::Defer(_) | Stmt::Comment(_) => (0, 0),
    }
}
fn base(t: &Type) -> &Type {
    match t {
        Type::Mut(x) | Type::Qualified(x, _) => base(x),
        _ => t,
    }
}
fn tuple_type_arity(value_ty: &Type) -> Option<usize> {
    match base(value_ty) {
        Type::Tuple(elements) => Some(elements.len()),
        _ => None,
    }
}
fn is_array(t: &Type) -> bool {
    matches!(
        base(t),
        Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::LabeledArray(_, _)
    )
}
fn elem_ty(t: &Type) -> String {
    match base(t) {
        Type::Array(x) | Type::ArrayN(x, _) | Type::ArrayNExpr(x, _) => ty(x),
        Type::LabeledArray(x, _) => ty(x),
        _ => "f64".into(),
    }
}
fn ty(t: &Type) -> String {
    match base(t) {
        Type::Int => "i32".into(),
        Type::Uint => "u32".into(),
        Type::Int8 => "i8".into(),
        Type::Int16 => "i16".into(),
        Type::Int32 => "i32".into(),
        Type::Int64 => "i64".into(),
        Type::Int128 => "i128".into(),
        Type::Uint8 => "u8".into(),
        Type::Uint16 => "u16".into(),
        Type::Uint32 => "u32".into(),
        Type::Uint64 => "u64".into(),
        Type::Uint128 => "u128".into(),
        Type::Float64 => "f64".into(),
        Type::Float32 => "f32".into(),
        Type::Str => "String".into(),
        Type::Bool => "bool".into(),
        Type::Named(n) => match n.as_str() {
            "float32" => "f32".into(),
            "float" | "float64" => "f64".into(),
            "string" => "String".into(),
            "int" => "i32".into(),
            "uint" => "u32".into(),
            "int8" => "i8".into(),
            "int16" => "i16".into(),
            "int32" => "i32".into(),
            "int64" => "i64".into(),
            "int128" => "i128".into(),
            "uint8" => "u8".into(),
            "uint16" => "u16".into(),
            "uint32" => "u32".into(),
            "uint64" => "u64".into(),
            "uint128" => "u128".into(),
            "bool" => "bool".into(),
            "Dimension" => "__DeviceDimension".into(),
            other if DEVICE_STRUCTS.with(|s| s.borrow().contains(other)) => {
                format!("__Device_{other}")
            }
            _ => "isize".into(),
        },
        Type::Array(_) | Type::ArrayN(_, _) | Type::ArrayNExpr(_, _) | Type::LabeledArray(_, _) => {
            format!("Vec<{}>", elem_ty(t))
        }
        Type::Dict(key, value) => format!(
            "std::collections::HashMap<{}, {}>",
            ty(key),
            ty(value)
        ),
        Type::Set(element) => format!("std::collections::HashSet<{}>", ty(element)),
        Type::Optional(inner) => format!("Option<{}>", ty(inner)),
        Type::Tuple(elements) => {
            let values = elements.iter().map(ty).collect::<Vec<_>>();
            if values.len() == 1 {
                format!("({},)", values[0])
            } else {
                format!("({})", values.join(", "))
            }
        }
        _ => "isize".into(),
    }
}
thread_local! {
    /// User structs used as kernel field types; their device mirror is `__Device_<Name>`.
    static DEVICE_STRUCTS: std::cell::RefCell<std::collections::HashSet<String>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

/// Device mirrors (`__Device_<Name>`, numeric/bool fields at device widths) of every
/// user struct a kernel field uses, plus the `__device_<Name>_from` host conversion.
fn device_struct_mirrors(program: &Program, kernels: &[KernelDecl], errors: &mut Vec<crate::transpiler::TranspileError>) -> String {
    let structs = program
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Struct(decl) => Some((decl.name.clone(), decl)),
            _ => None,
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mut wanted = Vec::new();
    for kernel in kernels {
        for field in &kernel.fields {
            if let Type::Named(name) = base(&field.ty) {
                if structs.contains_key(name.as_str()) && !wanted.contains(name) {
                    wanted.push(name.clone());
                }
            }
        }
    }
    DEVICE_STRUCTS.with(|s| *s.borrow_mut() = wanted.iter().cloned().collect());
    let mut out = String::new();
    for name in &wanted {
        let decl = structs[name.as_str()];
        let mut mirror = Vec::new();
        let mut convert = Vec::new();
        for field in &decl.fields {
            let device = numeric_ty(&field.ty).or_else(|| matches!(base(&field.ty), Type::Bool).then(|| "bool".to_string()));
            let Some(device) = device else {
                errors.push(crate::errors::SourceError::at(
                    format!("struct `{name}` field `{}` has a type a --target simd kernel cannot hold (only numeric and bool fields)", field.name),
                    decl.line,
                    decl.col,
                ));
                continue;
            };
            mirror.push(format!("{}: {device}", field.name));
            convert.push(if device == "bool" {
                format!("{0}: v.{0}", field.name)
            } else {
                format!("{0}: v.{0} as {device}", field.name)
            });
        }
        out.push_str(&format!(
            "#[derive(Clone, Copy, Debug, Default)] struct __Device_{name} {{ {} }}\n#[allow(non_snake_case)] fn __device_{name}_from(v: &{name}) -> __Device_{name} {{ __Device_{name} {{ {} }} }}\n",
            mirror.join(", "),
            convert.join(", ")
        ));
    }
    out
}

fn simd_needs_as_cast(value_ty: &Type) -> bool {
    match base(value_ty) {
        Type::Int
        | Type::Uint
        | Type::Int8
        | Type::Int16
        | Type::Int32
        | Type::Int64
        | Type::Int128
        | Type::Uint8
        | Type::Uint16
        | Type::Uint32
        | Type::Uint64
        | Type::Uint128
        | Type::Float32
        | Type::Float64
        | Type::Bool => true,
        Type::Named(name) => primitive_cast_type(name).is_some(),
        _ => false,
    }
}
fn default_value(t: &Type) -> String {
    if is_array(t) {
        match base(t) {
            Type::ArrayN(_, n) => format!("vec![Default::default(); {n}]"),
            Type::ArrayNExpr(_, size) => format!(
                "vec![Default::default(); ({}) as usize]",
                crate::transpiler::helpers::const_expr_to_c_like(&size.0)
            ),
            other => match other.labeled_array_len() {
                Some(n) => format!("vec![Default::default(); {n}]"),
                None => "Vec::new()".into(),
            },
        }
    } else {
        "Default::default()".into()
    }
}
fn primitive_cast_type(name: &str) -> Option<&'static str> {
    match name {
        "float32" => Some("f32"),
        "float" | "float64" => Some("f64"),
        "int" | "int32" => Some("i32"),
        "int8" => Some("i8"),
        "int16" => Some("i16"),
        "int64" => Some("i64"),
        "int128" => Some("i128"),
        "uint" | "uint32" => Some("u32"),
        "uint8" => Some("u8"),
        "uint16" => Some("u16"),
        "uint64" => Some("u64"),
        "uint128" => Some("u128"),
        _ => None,
    }
}
fn binop(op: &BinOp) -> Option<&'static str> {
    match op {
        BinOp::Add => Some("+"),
        BinOp::Sub => Some("-"),
        BinOp::Mul => Some("*"),
        BinOp::Div => Some("/"),
        BinOp::Rem => Some("%"),
        BinOp::Eq | BinOp::RefEq => Some("=="),
        BinOp::NotEq => Some("!="),
        BinOp::Lt => Some("<"),
        BinOp::LtEq => Some("<="),
        BinOp::Gt => Some(">"),
        BinOp::GtEq => Some(">="),
        BinOp::And => Some("&&"),
        BinOp::Or => Some("||"),
        BinOp::BitAnd => Some("&"),
        BinOp::BitOr => Some("|"),
        BinOp::BitXor => Some("^"),
        BinOp::Shl => Some("<<"),
        BinOp::Shr => Some(">>"),
        BinOp::Is | BinOp::IsNot => None,
    }
}

fn is_warp_receiver(expr: &Expr) -> bool {
    matches!(&expr.kind,
        ExprKind::Field(root, name)
            if name == "warp" && matches!(&root.kind, ExprKind::Var(value) if value == "gpu"))
}

fn is_warp_shuffle(method: &str) -> bool {
    matches!(
        method,
        "shuffle_down"
            | "shuffleDown"
            | "shuffle_up"
            | "shuffleUp"
            | "shuffle_xor"
            | "shuffleXor"
            | "shuffle"
    )
}

/// Cooperative block scheduler used by kernels the phase splitter rejects.
///
/// One block runs at a time on the calling thread.  Each work-item is a
/// stackful fiber (hand-written context switch on AArch64; one parked OS
/// thread per work-item with strict baton passing elsewhere).  `sync` parks a
/// fiber until every live work-item of the block has arrived, `.warp.sync()`
/// and the shuffles do the same per warp, and a work-item that returns early
/// simply stops taking part.
const COOP_RUNTIME: &str = r#"
#[allow(dead_code, unexpected_cfgs, unused_imports)]
mod __coop {
    use std::cell::{Cell, RefCell};
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

    type Body = *mut (dyn FnMut(usize) + 'static);
    type Payload = Box<dyn std::any::Any + Send>;
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum St { Ready, Block, Warp, Done }

    pub struct Sched {
        n: usize,
        warp: usize,
        st: Vec<St>,
        cur: usize,
        body: Body,
        scratch: Vec<u64>,
        shuffles: Vec<u32>,
        panic: Option<Payload>,
        be: Backend,
    }
    thread_local! { static CUR: Cell<*mut Sched> = const { Cell::new(std::ptr::null_mut()) }; }

    fn sched() -> &'static mut Sched {
        let p = CUR.with(|c| c.get());
        assert!(!p.is_null(), "SIMD kernel barrier used outside a cooperative block");
        unsafe { &mut *p }
    }

    pub fn run_block(n: usize, warp: usize, body: &mut dyn FnMut(usize)) {
        if n == 0 { return; }
        let body: Body = unsafe { std::mem::transmute::<&mut dyn FnMut(usize), Body>(body) };
        let mut s = Sched {
            n, warp: warp.max(1), st: vec![St::Ready; n], cur: 0, body,
            scratch: vec![0; 2 * n], shuffles: vec![0; n], panic: None, be: Backend::new(n),
        };
        let prev = CUR.with(|c| c.replace(&mut s));
        let panic = Backend::drive(&mut s);
        CUR.with(|c| c.set(prev));
        if let Some(p) = panic { resume_unwind(p); }
    }

    impl Sched {
        /// Releases every barrier whose participants have all arrived.
        fn release(&mut self) -> bool {
            let mut released = false;
            if self.st.iter().all(|s| matches!(s, St::Block | St::Done))
                && self.st.iter().any(|s| *s == St::Block)
            {
                for s in self.st.iter_mut() { if *s == St::Block { *s = St::Ready; } }
                released = true;
            }
            let mut base = 0;
            while base < self.n {
                let end = (base + self.warp).min(self.n);
                let lanes = &mut self.st[base..end];
                if lanes.iter().any(|s| *s == St::Warp)
                    && lanes.iter().all(|s| matches!(s, St::Warp | St::Done))
                {
                    for s in lanes.iter_mut() { if *s == St::Warp { *s = St::Ready; } }
                    released = true;
                }
                base = end;
            }
            released
        }
        fn finished(&self) -> bool { self.st.iter().all(|s| *s == St::Done) }
        fn deadlock(&self) -> ! {
            panic!("SIMD kernel deadlock: work-items wait at different barriers (a `sync` or `.warp` call is reached by only part of its block/warp)");
        }
    }

    pub fn sync() {
        let s = sched();
        s.st[s.cur] = St::Block;
        unsafe { Backend::suspend(s); }
    }
    pub fn warp_sync() {
        let s = sched();
        s.st[s.cur] = St::Warp;
        unsafe { Backend::suspend(s); }
    }
    pub fn shuffle<T: Copy>(kind: u8, v: T, arg: i32) -> T {
        assert!(std::mem::size_of::<T>() <= 8, "SIMD warp shuffle supports values of at most 8 bytes");
        let tid = sched().cur;
        let mut bits = 0u64;
        unsafe { std::ptr::copy_nonoverlapping(&v as *const T as *const u8, &mut bits as *mut u64 as *mut u8, std::mem::size_of::<T>()); }
        // Two scratch banks alternate per shuffle: a lane reaches the next shuffle's barrier only
        // after its reads of this one, so a single barrier per shuffle is enough.
        let (bank, n) = { let s = sched(); let bank = (s.shuffles[tid] & 1) as usize; s.shuffles[tid] += 1; (bank, s.n) };
        sched().scratch[bank * n + tid] = bits;
        warp_sync();
        let s = sched();
        let base = tid - tid % s.warp;
        let lane = (tid - base) as i32;
        let target = match kind { 0 => arg, 1 => lane + arg, 2 => lane - arg, _ => lane ^ arg };
        let out = if target >= 0 && (target as usize) < s.warp && base + (target as usize) < s.n
            && s.shuffles[base + target as usize] >= s.shuffles[tid]
        {
            let bits = s.scratch[bank * n + base + target as usize];
            let mut out = v;
            unsafe { std::ptr::copy_nonoverlapping(&bits as *const u64 as *const u8, &mut out as *mut T as *mut u8, std::mem::size_of::<T>()); }
            out
        } else { v };
        out
    }

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix, not(boring_coop_threads)))]
    use fiber::Backend;
    #[cfg(all(windows, not(boring_coop_threads)))]
    use fiber_win::Backend;
    #[cfg(not(any(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix, not(boring_coop_threads)), all(windows, not(boring_coop_threads)))))]
    use threads::Backend;

    #[cfg(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix, not(boring_coop_threads)))]
    mod fiber {
        use super::*;
        const CANARY: u64 = 0xB0121C57ACC0FFEE;
        pub struct Backend { sp: Vec<*mut u8>, sched_sp: *mut u8, stacks: Vec<Vec<u8>> }
        thread_local! { static POOL: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) }; }
        fn stack_bytes() -> usize {
            static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *B.get_or_init(|| std::env::var("BORING_SIMD_STACK_KB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256).max(16) * 1024)
        }
        #[cfg(target_arch = "x86_64")]
        #[unsafe(naked)]
        unsafe extern "C" fn ctx_switch(_save: *mut *mut u8, _load: *mut u8) {
            core::arch::naked_asm!(
                "push rbp", "push rbx", "push r12", "push r13", "push r14", "push r15",
                "mov [rdi], rsp", "mov rsp, rsi",
                "pop r15", "pop r14", "pop r13", "pop r12", "pop rbx", "pop rbp",
                "ret",
            )
        }
        #[cfg(target_arch = "x86_64")]
        #[unsafe(naked)]
        unsafe extern "C" fn trampoline() {
            core::arch::naked_asm!("mov rdi, r12", "call r13", "ud2")
        }
        #[cfg(target_arch = "aarch64")]
        #[unsafe(naked)]
        unsafe extern "C" fn ctx_switch(_save: *mut *mut u8, _load: *mut u8) {
            core::arch::naked_asm!(
                "sub sp, sp, #160",
                "stp x19, x20, [sp, #0]", "stp x21, x22, [sp, #16]", "stp x23, x24, [sp, #32]",
                "stp x25, x26, [sp, #48]", "stp x27, x28, [sp, #64]", "stp x29, x30, [sp, #80]",
                "stp d8, d9, [sp, #96]", "stp d10, d11, [sp, #112]", "stp d12, d13, [sp, #128]", "stp d14, d15, [sp, #144]",
                "mov x9, sp", "str x9, [x0]", "mov sp, x1",
                "ldp x19, x20, [sp, #0]", "ldp x21, x22, [sp, #16]", "ldp x23, x24, [sp, #32]",
                "ldp x25, x26, [sp, #48]", "ldp x27, x28, [sp, #64]", "ldp x29, x30, [sp, #80]",
                "ldp d8, d9, [sp, #96]", "ldp d10, d11, [sp, #112]", "ldp d12, d13, [sp, #128]", "ldp d14, d15, [sp, #144]",
                "add sp, sp, #160",
                "ret",
            )
        }
        #[cfg(target_arch = "aarch64")]
        #[unsafe(naked)]
        unsafe extern "C" fn trampoline() {
            core::arch::naked_asm!("mov x0, x19", "blr x20", "brk #0")
        }
        extern "C" fn fiber_main(_: usize) {
            let s = sched();
            let i = s.cur;
            let body = s.body;
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| unsafe { (*body)(i) })) {
                sched().panic = Some(p);
            }
            let s = sched();
            s.st[i] = St::Done;
            unsafe { Backend::suspend(s); }
            unreachable!("finished SIMD fiber resumed");
        }
        impl Backend {
            pub fn new(n: usize) -> Self {
                let want = stack_bytes();
                let mut stacks = POOL.with(|p| std::mem::take(&mut *p.borrow_mut()));
                stacks.retain(|s| s.len() == want);
                while stacks.len() < n { stacks.push(vec![0u8; want]); }
                stacks.truncate(n);
                Backend { sp: vec![std::ptr::null_mut(); n], sched_sp: std::ptr::null_mut(), stacks }
            }
            pub unsafe fn suspend(s: &mut Sched) {
                let i = s.cur;
                let save: *mut *mut u8 = &mut s.be.sp[i];
                let load = s.be.sched_sp;
                ctx_switch(save, load);
            }
            pub fn drive(s: &mut Sched) -> Option<Payload> {
                for i in 0..s.n {
                    let stack = &mut s.be.stacks[i];
                    let len = stack.len();
                    stack[..8].copy_from_slice(&CANARY.to_le_bytes());
                    let top = (stack.as_mut_ptr() as usize + len) & !15usize;
                    #[cfg(target_arch = "aarch64")]
                    let frame = {
                        let frame = top - 160;
                        unsafe {
                            std::ptr::write_bytes(frame as *mut u8, 0, 160);
                            *((frame + 8) as *mut usize) = fiber_main as extern "C" fn(usize) as usize;
                            *((frame + 88) as *mut usize) = trampoline as unsafe extern "C" fn() as usize;
                        }
                        frame
                    };
                    // x86-64: six callee-saved registers (r15 r14 r13 r12 rbx rbp) popped by
                    // `ctx_switch`, then the return address; `top` is 16-byte aligned when the
                    // trampoline starts, so its `call` enters `fiber_main` correctly aligned.
                    #[cfg(target_arch = "x86_64")]
                    let frame = {
                        let frame = top - 56;
                        unsafe {
                            std::ptr::write_bytes(frame as *mut u8, 0, 56);
                            *((frame + 16) as *mut usize) = fiber_main as extern "C" fn(usize) as usize;
                            *((frame + 48) as *mut usize) = trampoline as unsafe extern "C" fn() as usize;
                        }
                        frame
                    };
                    s.be.sp[i] = frame as *mut u8;
                }
                let mut panic = None;
                'blocks: loop {
                    let mut ran = false;
                    for i in 0..s.n {
                        if s.st[i] != St::Ready { continue; }
                        s.cur = i;
                        ran = true;
                        unsafe {
                            let save: *mut *mut u8 = &mut s.be.sched_sp;
                            let load = s.be.sp[i];
                            ctx_switch(save, load);
                        }
                        if u64::from_le_bytes(s.be.stacks[i][..8].try_into().unwrap()) != CANARY {
                            eprintln!("error: SIMD kernel work-item stack overflow; raise BORING_SIMD_STACK_KB");
                            std::process::abort();
                        }
                        if let Some(p) = s.panic.take() { panic = Some(p); break 'blocks; }
                    }
                    if s.finished() { break; }
                    let released = s.release();
                    if !ran && !released { s.deadlock(); }
                }
                let stacks = std::mem::take(&mut s.be.stacks);
                POOL.with(|p| *p.borrow_mut() = stacks);
                panic
            }
        }
    }

    #[cfg(all(windows, not(boring_coop_threads)))]
    mod fiber_win {
        use super::*;
        use std::ffi::c_void;
        extern "system" {
            fn ConvertThreadToFiber(parameter: *mut c_void) -> *mut c_void;
            fn CreateFiber(stack: usize, start: extern "system" fn(*mut c_void), parameter: *mut c_void) -> *mut c_void;
            fn SwitchToFiber(fiber: *mut c_void);
            fn DeleteFiber(fiber: *mut c_void);
        }
        thread_local! { static MAIN: Cell<*mut c_void> = const { Cell::new(std::ptr::null_mut()) }; }
        pub struct Backend { fibers: Vec<*mut c_void>, main: *mut c_void }
        fn stack_bytes() -> usize {
            static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *B.get_or_init(|| std::env::var("BORING_SIMD_STACK_KB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(256).max(16) * 1024)
        }
        extern "system" fn fiber_main(_: *mut c_void) {
            let s = sched();
            let i = s.cur;
            let body = s.body;
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| unsafe { (*body)(i) })) {
                sched().panic = Some(p);
            }
            let s = sched();
            s.st[i] = St::Done;
            unsafe { Backend::suspend(s); }
            unreachable!("finished SIMD fiber resumed");
        }
        impl Backend {
            pub fn new(n: usize) -> Self {
                let mut main = MAIN.with(|m| m.get());
                if main.is_null() {
                    main = unsafe { ConvertThreadToFiber(std::ptr::null_mut()) };
                    MAIN.with(|m| m.set(main));
                }
                Backend { fibers: vec![std::ptr::null_mut(); n], main }
            }
            pub unsafe fn suspend(s: &mut Sched) {
                SwitchToFiber(s.be.main);
            }
            pub fn drive(s: &mut Sched) -> Option<Payload> {
                for i in 0..s.n {
                    s.be.fibers[i] = unsafe { CreateFiber(stack_bytes(), fiber_main, std::ptr::null_mut()) };
                    assert!(!s.be.fibers[i].is_null(), "cannot allocate a SIMD work-item fiber");
                }
                let mut panic = None;
                'blocks: loop {
                    let mut ran = false;
                    for i in 0..s.n {
                        if s.st[i] != St::Ready { continue; }
                        s.cur = i;
                        ran = true;
                        unsafe { SwitchToFiber(s.be.fibers[i]); }
                        if let Some(p) = s.panic.take() { panic = Some(p); break 'blocks; }
                    }
                    if s.finished() { break; }
                    let released = s.release();
                    if !ran && !released { s.deadlock(); }
                }
                for fiber in s.be.fibers.drain(..) { unsafe { DeleteFiber(fiber); } }
                panic
            }
        }
    }

    #[cfg(not(any(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix, not(boring_coop_threads)), all(windows, not(boring_coop_threads)))))]
    mod threads {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        const SCHED: usize = usize::MAX;
        const ABORT: usize = usize::MAX - 1;
        pub struct Backend { turn: AtomicUsize, sched: Option<std::thread::Thread>, workers: Vec<std::thread::Thread> }
        struct Ptr(*mut Sched);
        unsafe impl Send for Ptr {}
        impl Backend {
            pub fn new(_n: usize) -> Self { Backend { turn: AtomicUsize::new(SCHED), sched: None, workers: Vec::new() } }
            fn wait_turn(s: &Sched, me: usize) {
                loop {
                    let t = s.be.turn.load(Ordering::Acquire);
                    if t == me { return; }
                    if t == ABORT { resume_unwind(Box::new(())); }
                    std::thread::park();
                }
            }
            pub unsafe fn suspend(s: &mut Sched) {
                let me = s.cur;
                s.be.turn.store(SCHED, Ordering::Release);
                s.be.sched.as_ref().unwrap().unpark();
                Self::wait_turn(s, me);
            }
            pub fn drive(s: &mut Sched) -> Option<Payload> {
                s.be.sched = Some(std::thread::current());
                let sp = s as *mut Sched;
                let mut panic = None;
                std::thread::scope(|scope| {
                    let handles: Vec<_> = (0..s.n).map(|i| {
                        let p = Ptr(sp);
                        scope.spawn(move || {
                            let p = p;
                            CUR.with(|c| c.set(p.0));
                            let body = unsafe { (*p.0).body };
                            let r = catch_unwind(AssertUnwindSafe(|| {
                                Self::wait_turn(unsafe { &*p.0 }, i);
                                unsafe { (*body)(i) }
                            }));
                            let s = unsafe { &mut *p.0 };
                            match r {
                                Ok(()) => {}
                                Err(e) if s.be.turn.load(Ordering::Acquire) == ABORT => drop(e),
                                Err(e) => s.panic = Some(e),
                            }
                            s.st[i] = St::Done;
                            s.be.turn.store(SCHED, Ordering::Release);
                            s.be.sched.as_ref().unwrap().unpark();
                        })
                    }).collect();
                    s.be.workers = handles.iter().map(|h| h.thread().clone()).collect();
                    'blocks: loop {
                        let mut ran = false;
                        for i in 0..s.n {
                            if s.st[i] != St::Ready { continue; }
                            s.cur = i;
                            ran = true;
                            s.be.turn.store(i, Ordering::Release);
                            s.be.workers[i].unpark();
                            while s.be.turn.load(Ordering::Acquire) != SCHED { std::thread::park(); }
                            if let Some(p) = s.panic.take() { panic = Some(p); break 'blocks; }
                        }
                        if s.finished() { break; }
                        let released = s.release();
                        if !ran && !released {
                            s.be.turn.store(ABORT, Ordering::Release);
                            for w in &s.be.workers { w.unpark(); }
                            s.deadlock();
                        }
                    }
                    s.be.turn.store(ABORT, Ordering::Release);
                    for w in &s.be.workers { w.unpark(); }
                });
                panic
            }
        }
    }
}
"#;
