// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Metal codegen snapshot tests.
//
// These tests verify the text emitted by `boring build --target metal` without
// requiring a real Metal GPU or macOS.  Each test:
//   1. Writes a Boring source snippet to a temp file.
//   2. Invokes `boring build --target metal <file>`.
//   3. Reads the generated kernels/main.metal and src/main.rs.
//   4. Asserts that the generated text contains the expected patterns.
//
// Run with:
//   cargo test --test metal_codegen

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Invoke `boring build --target metal <file>` and return the generated
/// (kernels/main.metal, src/main.rs) text pair.
///
/// boring names the output directory `<stem>_metal` next to the source file,
/// so we place the source in a dedicated temp dir and read from there.
fn metal_codegen(test_name: &str, src: &str) -> (String, String) {
    let (msl, rs, _toml) = run_metal(test_name, src);
    (msl, rs)
}

fn cargo_toml(test_name: &str, src: &str) -> String {
    let (_, _, toml) = run_metal(test_name, src);
    toml
}

fn run_metal(test_name: &str, src: &str) -> (String, String, String) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("metal_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // Source file named "test.br" → boring creates "test_metal/" next to it.
    let br_file   = tmp.join("test.br");
    let metal_dir = tmp.join("test_metal");
    fs::write(&br_file, src).unwrap();

    let result = Command::new(bin)
        .args(["build", "--target", "metal"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{test_name}] failed to invoke boring: {e}"));

    assert!(
        result.status.success(),
        "[{test_name}] boring build --target metal failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let read = |rel: &str| fs::read_to_string(metal_dir.join(rel)).unwrap_or_default();
    (
        read("kernels/main.metal"),
        read("src/main.rs"),
        read("Cargo.toml"),
    )
}

// ─── MSL header ──────────────────────────────────────────────────────────────

#[test]
fn msl_header_includes_metal_stdlib() {
    let (msl, _) = metal_codegen("msl_header", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("#include <metal_stdlib>"),
        "expected #include <metal_stdlib>;\ngot:\n{msl}");
    assert!(msl.contains("using namespace metal;"),
        "expected using namespace metal;\ngot:\n{msl}");
}

// ─── device — kernel signature ───────────────────────────────────────────────

#[test]
fn device_unified_field_becomes_device_ptr_buffer() {
    let (msl, _) = metal_codegen("unified_ptr", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("device float* buf [[buffer(0)]]"),
        "expected device float* buf [[buffer(0)]];\ngot:\n{msl}");
}

#[test]
fn device_global_field_becomes_device_ptr_buffer() {
    let (msl, _) = metal_codegen("global_ptr", r#"
kernel G:
    mut [float]'global buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] + 1.0
"#);
    assert!(msl.contains("device float* buf [[buffer(0)]]"),
        "expected device float* buf [[buffer(0)]] for 'global;\ngot:\n{msl}");
}

#[test]
fn device_entry_point_has_kernel_attribute() {
    let (msl, _) = metal_codegen("kernel_attr", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("kernel void Scale_kernel("),
        "expected 'kernel void Scale_kernel(' entry point;\ngot:\n{msl}");
}

#[test]
fn device_const_scalar_becomes_constant_ptr_with_deref() {
    let (msl, _) = metal_codegen("const_scalar", r#"
kernel C:
    mut [float]'unified buf
    let float'const     factor
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * factor
"#);
    assert!(msl.contains("constant float* __factor [[buffer("),
        "expected constant float* __factor [[buffer(N)]];\ngot:\n{msl}");
    assert!(msl.contains("const float factor = *__factor;"),
        "expected deref of __factor into const local;\ngot:\n{msl}");
}

#[test]
fn device_shared_dynamic_becomes_threadgroup_ptr() {
    let (msl, _) = metal_codegen("shared_dynamic", r#"
kernel S:
    mut [float]'unified out
    let [float]'sync  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(msl.contains("threadgroup float* scratch [[threadgroup(0)]]"),
        "expected threadgroup pointer param for dynamic 'sync;\ngot:\n{msl}");
    // dynamic 'sync must NOT appear as a device buffer param
    assert!(!msl.contains("device float* scratch"),
        "dynamic 'sync must not appear as device buffer;\ngot:\n{msl}");
}

#[test]
fn device_shared_static_declared_in_body() {
    let (msl, _) = metal_codegen("shared_static", r#"
kernel S:
    mut [float]'unified out
    let [float, 32]'sync tile
    def ():
        let tid = gpu.thread.x
        out[tid] = tile[0]
"#);
    assert!(msl.contains("threadgroup float tile[32];"),
        "expected threadgroup T name[N] for static 'sync;\ngot:\n{msl}");
    // static 'sync must not appear as a threadgroup param
    assert!(!msl.contains("tile [[threadgroup("),
        "static 'sync must not appear as threadgroup param;\ngot:\n{msl}");
}

#[test]
fn device_local_fixed_array_declared_in_body() {
    let (msl, _) = metal_codegen("local_array", r#"
kernel L:
    mut [float]'unified out
    let [float, 8]'local tmp
    def ():
        let tid = gpu.thread.x
        out[tid] = tmp[0]
"#);
    assert!(msl.contains("float tmp[8];"),
        "expected fixed-size local array in body;\ngot:\n{msl}");
}

// ─── device — built-in position parameters ───────────────────────────────────

#[test]
fn device_builtin_position_params_present() {
    let (msl, _) = metal_codegen("builtin_params", r#"
kernel B:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("[[thread_position_in_threadgroup]]"),
        "expected [[thread_position_in_threadgroup]];\ngot:\n{msl}");
    assert!(msl.contains("[[threadgroup_position_in_grid]]"),
        "expected [[threadgroup_position_in_grid]];\ngot:\n{msl}");
    assert!(msl.contains("[[threads_per_threadgroup]]"),
        "expected [[threads_per_threadgroup]];\ngot:\n{msl}");
    assert!(msl.contains("[[threadgroups_per_grid]]"),
        "expected [[threadgroups_per_grid]];\ngot:\n{msl}");
}

#[test]
fn device_gpu_thread_x_maps_to_thread_pos() {
    let (msl, _) = metal_codegen("gpu_thread_x", r#"
kernel B:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(msl.contains("__thread_pos"),
        "expected __thread_pos for gpu.thread;\ngot:\n{msl}");
    assert!(msl.contains("__thread_pos.x"),
        "expected __thread_pos.x for gpu.thread.x;\ngot:\n{msl}");
}

#[test]
fn device_gpu_block_dim_maps_correctly() {
    let (msl, _) = metal_codegen("gpu_block_dim", r#"
kernel B:
    mut [float]'unified buf
    def ():
        let i = gpu.thread.x + gpu.block.x * gpu.block_dim.x
        buf[i] = buf[i] * 2.0
"#);
    assert!(msl.contains("__block_pos.x"),
        "expected __block_pos.x for gpu.block.x;\ngot:\n{msl}");
    assert!(msl.contains("__block_dim.x"),
        "expected __block_dim.x for gpu.block_dim.x;\ngot:\n{msl}");
}

// ─── device — 'actor'global atomics ──────────────────────────────────────────

#[test]
fn device_actor_global_compound_assign_uses_atomic_fetch_add() {
    let (msl, _) = metal_codegen("atomic_add", r#"
kernel A:
    mut [int]'actor'global counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(msl.contains("atomic_fetch_add_explicit"),
        "expected atomic_fetch_add_explicit for 'actor'global += ;\ngot:\n{msl}");
    assert!(msl.contains("memory_order_relaxed"),
        "expected memory_order_relaxed;\ngot:\n{msl}");
}

#[test]
fn device_actor_global_field_has_device_ptr_param() {
    let (msl, _) = metal_codegen("atomic_ptr_param", r#"
kernel A:
    mut [int]'actor'global counts
    def ():
        let tid = gpu.thread.x
        counts[0] += tid
"#);
    assert!(msl.contains("device int64_t* counts [[buffer(0)]]"),
        "expected device int64_t* counts [[buffer(0)]] for 'actor'global;\ngot:\n{msl}");
}

// ─── device — sync barrier ────────────────────────────────────────────────────

#[test]
fn device_sync_emits_threadgroup_barrier() {
    let (msl, _) = metal_codegen("sync_barrier", r#"
kernel S:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = 1.0
        sync
        buf[tid] = buf[tid] + 1.0
"#);
    // Metal uses `threadgroup_barrier` for threadgroup memory sync.
    // The sync comment is emitted by the kernel backend as a Stmt::Comment("sync").
    assert!(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "expected threadgroup_barrier for sync;\ngot:\n{msl}");
}

// ─── host — struct and Metal plumbing ────────────────────────────────────────

#[test]
fn host_prelude_includes_metal_crate() {
    let (_, rs) = metal_codegen("host_prelude", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("use metal::*;"),
        "expected 'use metal::*;' in host prelude;\ngot:\n{rs}");
    assert!(rs.contains("include_str!(\"../kernels/main.metal\")"),
        "expected include_str! for BORING_MSL;\ngot:\n{rs}");
}

#[test]
fn host_pipeline_init_compiles_msl_and_gets_function() {
    let (_, rs) = metal_codegen("host_pipeline", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("new_library_with_source(BORING_MSL"),
        "expected new_library_with_source;\ngot:\n{rs}");
    assert!(rs.contains("\"Scale_kernel\""),
        "expected get_function(\"Scale_kernel\");\ngot:\n{rs}");
    assert!(rs.contains("new_compute_pipeline_state_with_function"),
        "expected new_compute_pipeline_state_with_function;\ngot:\n{rs}");
}

#[test]
fn host_unified_field_is_metal_buffer() {
    let (_, rs) = metal_codegen("host_buffer_field", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("buf: Buffer,"),
        "expected 'buf: Buffer,' for 'unified field;\ngot:\n{rs}");
}

#[test]
fn host_shared_field_absent_from_rust_struct() {
    let (_, rs) = metal_codegen("host_shared_absent", r#"
kernel S:
    mut [float]'unified out
    let [float]'sync  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(!rs.contains("scratch: Buffer"),
        "'sync field must not appear as Buffer in Rust struct;\ngot:\n{rs}");
}

#[test]
fn host_init_uploads_array_via_new_buffer_with_data() {
    let (_, rs) = metal_codegen("host_htod", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("new_buffer_with_data"),
        "expected new_buffer_with_data for array upload;\ngot:\n{rs}");
}

// ─── host — __boring_launch ───────────────────────────────────────────────────

#[test]
fn host_boring_launch_signature() {
    let (_, rs) = metal_codegen("host_launch_sig", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("fn __boring_launch(self, block_dim: (u32,u32,u32), grid_dim: Option<(u32,u32,u32)>"),
        "expected __boring_launch with Option grid_dim;\ngot:\n{rs}");
}

#[test]
fn host_boring_launch_dispatches_thread_groups() {
    let (_, rs) = metal_codegen("host_dispatch", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("dispatch_thread_groups("),
        "expected dispatch_thread_groups in __boring_launch;\ngot:\n{rs}");
    assert!(rs.contains("MTLSize"),
        "expected MTLSize for grid/block dims;\ngot:\n{rs}");
}

#[test]
fn host_boring_launch_waits_for_completion() {
    let (_, rs) = metal_codegen("host_wait", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("wait_until_completed()"),
        "expected wait_until_completed() for Metal synchronous launch;\ngot:\n{rs}");
}

#[test]
fn host_auto_grid_sizing_from_first_array_len() {
    let (_, rs) = metal_codegen("auto_grid", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("(n + block_dim.0 - 1) / block_dim.0"),
        "expected auto grid ceil-div expression;\ngot:\n{rs}");
    assert!(rs.contains("grid_dim: Option<(u32,u32,u32)>"),
        "expected Option grid_dim for auto-grid kernel;\ngot:\n{rs}");
}

#[test]
fn host_dynamic_shared_sets_threadgroup_memory_length() {
    let (_, rs) = metal_codegen("threadgroup_mem", r#"
kernel D:
    mut [float]'unified out
    let [float]'sync  scratch
    def ():
        let tid = gpu.thread.x
        out[tid] = scratch[0]
"#);
    assert!(rs.contains("set_threadgroup_memory_length("),
        "expected set_threadgroup_memory_length for dynamic 'sync;\ngot:\n{rs}");
}

#[test]
fn host_encoder_sets_buffer_for_unified_field() {
    let (_, rs) = metal_codegen("set_buffer", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("set_buffer(0, Some(&self.buf)"),
        "expected set_buffer(0, Some(&self.buf));\ngot:\n{rs}");
}

#[test]
fn host_const_scalar_uses_set_bytes() {
    let (_, rs) = metal_codegen("const_set_bytes", r#"
kernel C:
    mut [float]'unified buf
    let float'const     factor
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * factor
"#);
    assert!(rs.contains("set_bytes("),
        "expected set_bytes for 'const scalar field;\ngot:\n{rs}");
    assert!(rs.contains("&self.factor"),
        "expected &self.factor in set_bytes;\ngot:\n{rs}");
}

// ─── host — read accessor ─────────────────────────────────────────────────────

#[test]
fn host_read_accessor_generated_for_unified_array() {
    let (_, rs) = metal_codegen("read_accessor", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("fn read_buf("),
        "expected read_buf accessor for 'unified array;\ngot:\n{rs}");
    assert!(rs.contains("from_raw_parts"),
        "expected unsafe slice in read accessor;\ngot:\n{rs}");
}

// ─── infrastructure — Cargo.toml ─────────────────────────────────────────────

#[test]
fn cargo_toml_depends_on_metal_crate() {
    let toml = cargo_toml("infra_cargo_toml", r#"
kernel Scale:
    mut [float]'unified buf
    def ():
        let tid = gpu.thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(toml.contains("metal"),
        "Cargo.toml must depend on the metal crate;\ngot:\n{toml}");
}

// ─── saxpy example ────────────────────────────────────────────────────────────

#[test]
fn example_saxpy_metal() {
    let src = std::fs::read_to_string("examples/saxpy.br").expect("examples/saxpy.br not found");
    let (msl, rs) = metal_codegen("saxpy_example", &src);

    // MSL kernel
    assert!(msl.contains("kernel void Saxpy_kernel("), "missing Saxpy_kernel;\ngot:\n{msl}");
    assert!(msl.contains("device float* y [[buffer("), "missing y buffer param;\ngot:\n{msl}");

    // Host struct
    assert!(rs.contains("struct Saxpy"),  "missing struct Saxpy;\ngot:\n{rs}");
    assert!(rs.contains("buf: Buffer,") || rs.contains("y: Buffer,"),
        "missing Buffer field in Saxpy;\ngot:\n{rs}");
    assert!(rs.contains("new_library_with_source(BORING_MSL"),
        "missing MSL compile step;\ngot:\n{rs}");
}
