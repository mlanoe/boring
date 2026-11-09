// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
#![allow(clippy::approx_constant, clippy::useless_format)]
// wgpu codegen snapshot tests.
//
// These tests verify the text emitted by `boring build --target wgpu` without
// requiring a real GPU.  Each test:
//   1. Writes a Boring source snippet to a temp file.
//   2. Invokes `boring build --target wgpu <file>`.
//   3. Reads the generated shaders/main.wgsl and src/main.rs.
//   4. Asserts that the generated text contains the expected patterns.
//
// Run with:
//   cargo test --test wgpu_codegen

mod common;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn init_parameter_used_only_in_host_expressions_is_preserved() {
    let (_wgsl, rs) = wgpu_codegen("init_expression_only_param", r#"
kernel SizedOut:
    let [float32]'global a
    mut [float32]'unified out
    let int n
    let int total
    init([float32]'global ai, int nn, int extra):
        a = ai
        n = nn
        total = nn + extra
        out = [0.0 for ..<(nn + extra)]
    def ():
        let i = .thread.x
        if i < n:
            out[i] = a[i]
pub req [float32]'kernel'unified build([float32]'global x, int n) throws:
    mut k = SizedOut(x, n, 2)
    kernel:
        k(block = 256, grid = 1)
    k.out
"#);
    assert!(rs.contains("k.total = ((n + 2)) as i32;"), "{rs}");
    assert!(rs.contains("vec![(0) as f32; ((n + 2)) as usize]"), "{rs}");
}

#[test]
fn fixed_array_local_is_declared_and_initialized_in_kernel_body() {
    let (wgsl, _) = wgpu_codegen("fixed_array_local", r#"
kernel LocalArr:
    mut [float32]'unified out
    init():
        out = [0.0 for ..<1]
    def ():
        var [float32, 8] acc = [0.0 for ..<8]
        acc[3] = 2.0
        out[0] = acc[3]
"#);
    assert!(wgsl.contains("var acc: array<f32, 8> = array<f32, 8>("), "{wgsl}");
    assert!(!wgsl.contains("/* expr */"), "{wgsl}");
}

#[test]
fn thread_private_fixed_array_kernel_field_is_declared_in_entry_point() {
    let (wgsl, _) = wgpu_codegen("local_fixed_array_field", r#"
kernel LocalField:
    mut [float32]'unified out
    mut [float32, 8] acc
    init():
        out = [0.0 for ..<1]
    def ():
        for e in 0..<8:
            acc[e] = e as float32
        out[0] = acc[7]
"#);
    assert!(wgsl.contains("var acc: array<f32, 8>;"),
        "expected inferred 'local field declaration in entry point;\ngot:\n{wgsl}");
    assert!(wgsl.contains("acc[u32(e)] = f32(e);"), "{wgsl}");
}

#[test]
fn const_fixed_array_filled_by_init_literal_uses_params_and_promotes_thread_index() {
    let (wgsl, rs) = wgpu_codegen("const_array_init_literal", r#"
mut [float32]'kernel'unified out = [0.0 for ..<8]
kernel FA:
    let [float32, 4]'const w
    mut [float32]'unified out
    init(mut [float32]'unified o):
        w = [1.0, 2.0, 3.0, 4.0]
        out = o
    def ():
        let i = .thread.x
        out[i] = w[0] + w[3] + i
mut fa = FA(out)
kernel:
    fa(block = 8)
print "{fa.out[0]} {fa.out[7]}"
"#);
    // The init literal reaches the params-uniform host mirror (was E0599: no `copy_w_to_device`).
    assert!(rs.contains("fn copy_w_to_device(&mut self, data: &[f32])"), "{rs}");
    // `float32 + int` must convert the integer side: WGSL has no implicit i32 -> f32.
    assert!(wgsl.contains("+ f32(i))"), "{wgsl}");
}

#[test]
fn unsupported_fixed_array_local_initializer_is_a_build_error() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("fixed_array_local_bad_initializer");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br_file = tmp.join("test.br");
    fs::write(&br_file, r#"
kernel LocalArr:
    mut [float32]'unified out
    init():
        out = [0.0 for ..<1]
    def ():
        var [float32, 8] acc = [..<8]
        out[0] = acc[0]
"#).unwrap();
    let result = Command::new(bin)
        .args(["build", "--target", "wgpu"])
        .arg(&br_file).output().unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "build unexpectedly succeeded");
    assert!(stderr.contains("fixed-array kernel locals require"), "{stderr}");
}

fn run_wgpu(test_name: &str, src: &str) -> (String, String, String, String) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    // Source file named "test.br" → boring creates "test_wgpu/" next to it.
    let br_file  = tmp.join("test.br");
    let wgpu_dir = tmp.join("test_wgpu");
    fs::write(&br_file, src).unwrap();

    let result = Command::new(bin)
        .args(["build", "--target", "wgpu"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{test_name}] failed to invoke boring: {e}"));

    assert!(
        result.status.success(),
        "[{test_name}] boring build --target wgpu failed:\n{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let read = |rel: &str| fs::read_to_string(wgpu_dir.join(rel)).unwrap_or_default();
    (
        read("shaders/main.wgsl"),
        read("shaders/main_emulated.wgsl"),
        read("src/main.rs"),
        read("Cargo.toml"),
    )
}

fn wgpu_codegen(test_name: &str, src: &str) -> (String, String) {
    let (wgsl, _emulated, rs, _toml) = run_wgpu(test_name, src);
    (wgsl, rs)
}

#[test]
fn shader_f16_is_not_requested_for_float32_only_programs() {
    let (wgsl, host) = wgpu_codegen("no_unneeded_shader_f16", r#"
kernel Scale:
    mut [float32]'unified values
    def ():
        values[.thread.x] *= 2.0
"#);
    assert!(!wgsl.contains("enable f16;"), "{wgsl}");
    assert!(!host.contains("Features::SHADER_F16"), "{host}");
}

#[test]
fn mutable_static_used_by_host_function_is_emitted_once() {
    let (_, host) = wgpu_codegen("mutable_static_once", GPU_STATIC_VAR_REPRO);
    assert_eq!(host.matches("static FLAG:").count(), 1, "{host}");
    assert!(!host.contains("const FLAG:"), "{host}");
}

const GPU_STATIC_VAR_REPRO: &str = r#"
kernel Touch:
    mut [float32]'unified out
    let int n
    init(int nn):
        n = nn
        out = [0.0 for ..<nn]
    def ():
        out[.thread.x] = 1.0

static var bool flag = false
def set_flag(bool v):
    flag = v

pub req [float32]'kernel'unified run(int n) throws:
    mut k = Touch(n)
    kernel:
        k(block = 1, grid = n)
    k.out
"#;

/// Like `run_wgpu`, but for a source expected to fail `boring build --target wgpu`
/// itself (before any Rust is even written out) -- returns stderr instead of the
/// generated files. Mirrors `gpu_kernel_std_target.rs`'s identical "expect a clean
/// diagnostic, not silent success or a raw generated-project compile error" pattern.
fn run_wgpu_expect_failure(test_name: &str, src: &str) -> String {
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join(test_name);
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();

    let br_file = tmp.join("test.br");
    fs::write(&br_file, src).unwrap();

    let result = Command::new(bin)
        .args(["build", "--target", "wgpu"])
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("[{test_name}] failed to invoke boring: {e}"));

    assert!(
        !result.status.success(),
        "[{test_name}] expected `boring build --target wgpu` to fail, but it succeeded"
    );
    // No project directory should have been written -- errors are collected and
    // reported (see `main.rs`'s wgpu branch) before any file is created.
    assert!(
        !tmp.join("test_wgpu").exists(),
        "[{test_name}] expected no test_wgpu/ project dir on a rejected program"
    );
    String::from_utf8_lossy(&result.stderr).into_owned()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn test_simple_vector_add() {
    let src = r#"
kernel VecAdd:
    mut [float]'unified a
    mut [float]'unified b
    mut [float]'unified c
    let int n

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        if i < n:
            c[i] = a[i] + b[i]
"#;
    let (wgsl, rs) = wgpu_codegen("vector_add", src);

    // WGSL device side.
    assert!(wgsl.contains("@group(0) @binding(0)"), "missing binding 0");
    assert!(wgsl.contains("@group(0) @binding(1)"), "missing binding 1");
    assert!(wgsl.contains("@group(0) @binding(2)"), "missing binding 2");
    assert!(wgsl.contains("var<storage"), "missing storage qualifier");
    assert!(wgsl.contains("@compute @workgroup_size("), "missing workgroup_size");
    assert!(wgsl.contains("@builtin(local_invocation_id)"), "missing local_invocation_id builtin");
    assert!(wgsl.contains("@builtin(workgroup_id)"), "missing workgroup_id builtin");
    assert!(wgsl.contains("VecAdd_main"), "missing entry fn name");

    // Host side.
    assert!(rs.contains("wgpu::BufferUsages::STORAGE"), "missing STORAGE usage");
    assert!(rs.contains("wgpu::BufferUsages::COPY_SRC"), "missing COPY_SRC on storage buffer");
    assert!(rs.contains("dispatch_workgroups"), "missing dispatch");
    assert!(rs.contains("queue.submit"), "missing queue submit");
    assert!(rs.contains("device.poll"), "missing device poll");
    assert!(rs.contains("bytemuck"), "missing bytemuck import");
}

#[test]
fn test_dispatch_block_size_resolves_local_let_int_const_not_silently_one() {
    // Regression test: a `k(block = (a, b))` dispatch where `b` is a
    // function-local `let`-bound int (not a literal, not a top-level
    // const) used to silently emit `@workgroup_size(a, 1, 1)` instead of
    // resolving `b`'s value -- WGSL requires `@workgroup_size` to be a
    // compile-time constant, and `collect_block_sizes`/`scan_call_block_size`
    // only ever resolved a *top-level* scalar `let` referenced by name,
    // falling back to `1` for anything function-local (including a
    // trivially constant-foldable one like this). No error was raised
    // either -- the shader compiled and ran with only workgroup y=0 ever
    // dispatching, silently dropping every other row of work. Confirmed
    // against a real GPU: before this fix, a warp-shuffle reduction kernel
    // dispatched this way (`boring-llm`'s `linear_warp_gpu`) produced
    // `3 0 0 0` instead of the correct `3 3 7 7`.
    let src = r#"
kernel WarpLike:
    let [float]'global x
    mut [float]'unified y
    let int warps_per_block

    init([float]'global xi, int wpb):
        x = xi
        warps_per_block = wpb
        y = [0.0 for ..<4]

    def ():
        let warp_in_block = .thread.y
        if warp_in_block < warps_per_block:
            y[warp_in_block] = x[warp_in_block]

pub req [float]'kernel'unified warp_call([float]'global x) throws:
    let int warps_per_block = 8
    mut k = WarpLike(x, warps_per_block)
    kernel:
        k(block = (32, warps_per_block), grid = (1, 1))
    k.y
"#;
    let (wgsl, _rs) = wgpu_codegen("dispatch_block_size_local_let", src);
    assert!(
        wgsl.contains("@workgroup_size(32, 8, 1)"),
        "expected the local `let warps_per_block = 8` to resolve into \
         @workgroup_size's y dimension, not silently fall back to 1;\ngot:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("@workgroup_size(32, 1, 1)"),
        "must not silently drop the non-literal block-size dimension to 1;\ngot:\n{wgsl}"
    );
}

#[test]
fn test_kernel_dispatch_surfaces_validation_errors_instead_of_silent_failure() {
    // Before this fix, `dispatch()` returned `()` and no error scope existed
    // anywhere in the generated code -- a validation failure (e.g. a rejected
    // workgroup count) was never observed by anything Boring generated. Confirmed
    // via a real `cargo check` against the real `wgpu`/`pollster` crates that this
    // whole chain (dispatch -> kernel: block call site -> boring_main()'s own
    // Result) compiles end-to-end.
    let src = r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = .thread.x
        buf[tid] = buf[tid] * 2.0

let data = [1.0, 2.0]
mut k = Scale(data)
kernel:
    k(block = 2)
"#;
    let (_wgsl, rs) = wgpu_codegen("dispatch_error_scope", src);

    assert!(rs.contains("fn dispatch(&self, gx: u32, gy: u32, gz: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>>"),
        "expected dispatch() to return a real Result;\ngot:\n{rs}");
    assert!(rs.contains("push_error_scope(wgpu::ErrorFilter::Validation)"),
        "expected dispatch() to open a validation error scope before encoding;\ngot:\n{rs}");
    assert!(rs.contains("pollster::block_on(self.device.pop_error_scope())"),
        "expected dispatch() to check the error scope after submit;\ngot:\n{rs}");
    assert!(rs.contains("k.dispatch((") && rs.contains(")?;"),
        "expected the kernel: block's dispatch call site to propagate the error via ?;\ngot:\n{rs}");
    assert!(rs.contains("fn boring_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>>"),
        "expected the synthesized boring_main() to be Result-returning so dispatch()'s ? has somewhere to go;\ngot:\n{rs}");
}

#[test]
fn test_scalar_uniform() {
    let src = r#"
kernel Scale:
    mut [float32]'unified data
    let float32 alpha

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        data[i] = data[i] * alpha
"#;
    let (wgsl, rs) = wgpu_codegen("scalar_uniform", src);

    assert!(wgsl.contains("struct ScaleParams"), "missing params struct in WGSL");
    assert!(wgsl.contains("alpha: f32"), "missing alpha field in params");
    assert!(wgsl.contains("var<uniform> scale_params"), "missing uniform binding");

    assert!(rs.contains("struct ScaleParams"), "missing params struct in Rust");
    assert!(rs.contains("queue.write_buffer"), "missing params upload");
}

#[test]
fn test_powf_maps_to_wgsl_pow_builtin() {
    // Regression test: `.powf()` on a float32 used to pass straight through
    // `map_builtin_fn`'s catch-all as a bare `powf(...)` call -- WGSL has no such
    // builtin (only `pow(x, y)`), so the shader compiled fine through `boring
    // build --target wgpu` + `cargo build` but failed at GPU-pipeline-creation
    // time with "no definition in scope for identifier: 'powf'". Found via
    // boring-llm's RoPE inverse-frequency kernel (`1.0 / base.powf(exp)`).
    let src = r#"
kernel Pow:
    mut [float32]'unified data
    let float32 alpha

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        data[i] = data[i].powf(alpha)
"#;
    let (wgsl, _rs) = wgpu_codegen("powf_builtin", src);

    assert!(wgsl.contains("pow("), "expected WGSL `pow(...)` builtin;\ngot:\n{wgsl}");
    assert!(!wgsl.contains("powf("), "must not emit invalid WGSL `powf(...)`;\ngot:\n{wgsl}");
}

#[test]
fn test_ln_maps_to_wgsl_log_builtin() {
    // Same bug class as `test_powf_maps_to_wgsl_pow_builtin`, a different missing
    // `map_builtin_fn` entry: `.ln()` used to pass straight through as a bare
    // `ln(...)` call -- WGSL has no `ln` identifier at all (its natural-log
    // builtin is spelled `log`), so this compiled fine through `cargo build` but
    // failed at GPU-pipeline-creation time ("no definition in scope for
    // identifier: 'ln'"). Found via boring-llm's RoPE frequency kernel.
    let src = r#"
kernel Log:
    mut [float32]'unified data
    let float32 base

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        data[i] = base.ln()
"#;
    let (wgsl, _rs) = wgpu_codegen("ln_builtin", src);

    assert!(wgsl.contains("log("), "expected WGSL `log(...)` builtin;\ngot:\n{wgsl}");
    assert!(!wgsl.contains("ln("), "must not emit invalid WGSL `ln(...)`;\ngot:\n{wgsl}");
}

#[test]
fn test_signum_maps_to_wgsl_sign_builtin() {
    // Same bug class again: WGSL's builtin is spelled `sign`, not `signum`.
    let src = r#"
kernel Signum:
    mut [float32]'unified data

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        data[i] = data[i].signum()
"#;
    let (wgsl, _rs) = wgpu_codegen("signum_builtin", src);

    assert!(wgsl.contains("sign("), "expected WGSL `sign(...)` builtin;\ngot:\n{wgsl}");
    assert!(!wgsl.contains("signum("), "must not emit invalid WGSL `signum(...)`;\ngot:\n{wgsl}");
}

#[test]
fn test_cbrt_expands_to_sign_preserving_pow() {
    // WGSL has no `cbrt` builtin under any name, so this can't be fixed by a
    // `map_builtin_fn` rename like `ln`/`signum` above -- it must expand to a
    // compound expression instead.
    let src = r#"
kernel Cbrt:
    mut [float32]'unified data

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        data[i] = data[i].cbrt()
"#;
    let (wgsl, _rs) = wgpu_codegen("cbrt_expand", src);

    assert!(wgsl.contains("sign(") && wgsl.contains("pow(abs("),
        "expected the sign-preserving cbrt expansion `sign(x) * pow(abs(x), 1.0 / 3.0)`;\ngot:\n{wgsl}");
    assert!(!wgsl.contains("cbrt("), "must not emit invalid WGSL `cbrt(...)`;\ngot:\n{wgsl}");
}

#[test]
fn test_log10_expands_to_change_of_base() {
    // Same story as `cbrt`: WGSL has no `log10` builtin, expand via the
    // change-of-base identity using WGSL's own `log` (natural-log) builtin.
    let src = r#"
kernel Log10:
    mut [float32]'unified data

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        data[i] = data[i].log10()
"#;
    let (wgsl, _rs) = wgpu_codegen("log10_expand", src);

    assert!(wgsl.contains("log(") && wgsl.contains("/ log(10.0)"),
        "expected the change-of-base expansion `log(x) / log(10.0)`;\ngot:\n{wgsl}");
    assert!(!wgsl.contains("log10("), "must not emit invalid WGSL `log10(...)`;\ngot:\n{wgsl}");
}

#[test]
fn test_float_builtin_methods_real_shader_validation() {
    // End-to-end companion to the four textual tests above (`ln`/`signum`/`cbrt`/
    // `log10`) plus the pre-existing `powf` mapping -- a text-only assertion on the
    // generated WGSL can't catch a codegen mistake that only naga's real shader
    // parser/validator would reject (this is exactly how the original `ln` bug
    // surfaced: `cargo build` succeeded, only real shader-module creation failed).
    // Every input below is chosen so the *mathematically* expected result is a
    // round number: ln(1)=0, signum(-5)=-1, cbrt(-1)=-1 (proves the
    // sign-preserving expansion, not just that `cbrt` compiles -- a naive
    // `pow(x, 1/3)` on a negative `x` is NaN), log10(10)=1, pow(2,3)=8. The
    // `log10`/`cbrt` expansions go through the GPU's own approximate `log`/`pow`
    // builtins twice (once on the runtime value, once on a same-valued literal
    // for `log10`), so the result isn't always bit-exact -- compare with a small
    // tolerance rather than an exact string match.
    let src = r#"
kernel FloatBuiltins:
    mut [float32]'unified out
    let float32 ln_in
    let float32 signum_in
    let float32 cbrt_in
    let float32 log10_in
    let float32 pow_base
    let float32 pow_exp

    init(float32 a, float32 b, float32 c, float32 d, float32 e, float32 f):
        out = [0.0, 0.0, 0.0, 0.0, 0.0]
        ln_in = a
        signum_in = b
        cbrt_in = c
        log10_in = d
        pow_base = e
        pow_exp = f

    def ():
        out[0] = ln_in.ln()
        out[1] = signum_in.signum()
        out[2] = cbrt_in.cbrt()
        out[3] = log10_in.log10()
        out[4] = pow_base.powf(pow_exp)

mut k = FloatBuiltins(1.0, -5.0, -1.0, 10.0, 2.0, 3.0)
kernel:
    k(block = 1)

print "ln = {k.out[0]}"
print "signum = {k.out[1]}"
print "cbrt = {k.out[2]}"
print "log10 = {k.out[3]}"
print "pow = {k.out[4]}"
"#;
    let (_wgsl, _emulated, _rs, _toml) = run_wgpu("float_builtin_methods_real_shader", src);

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("float_builtin_methods_real_shader");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU, but it failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    fn parsed_value(stdout: &str, prefix: &str) -> f32 {
        let line = stdout.lines().find(|l| l.starts_with(prefix))
            .unwrap_or_else(|| panic!("missing '{prefix}' line in stdout:\n{stdout}"));
        line[prefix.len()..].trim().parse::<f32>()
            .unwrap_or_else(|e| panic!("failed to parse '{line}' as f32: {e}"))
    }
    let checks: [(&str, f32); 5] = [
        ("ln = ", 0.0),
        ("signum = ", -1.0),
        ("cbrt = ", -1.0),
        ("log10 = ", 1.0),
        ("pow = ", 8.0),
    ];
    for (prefix, expected) in checks {
        let actual = parsed_value(&stdout, prefix);
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {prefix}~{expected}, got {actual} — full stdout:\n{stdout}"
        );
    }
}

#[test]
fn test_sync_barrier_fixed_array() {
    let src = r#"
kernel Tile:
    let [float32, 256]'actor tile
    mut [float32]'unified data

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        tile[.thread.x] = data[i]
        sync
        data[i] = tile[.thread.x]
"#;
    let (wgsl, _rs) = wgpu_codegen("sync_barrier", src);

    assert!(wgsl.contains("var<workgroup>"), "missing workgroup var");
    assert!(wgsl.contains("array<f32, 256>"), "missing fixed-size array type");
    assert!(wgsl.contains("workgroupBarrier()"), "missing explicit barrier");

    // `var<workgroup>` is only legal at WGSL module scope — naga rejects it as a
    // statement inside a function body ("expected identifier, found '<'"). Make
    // sure the declaration appears before the entry point's `@compute` annotation
    // (i.e. outside the function), not after it (i.e. inside the function body).
    let workgroup_pos = wgsl.find("var<workgroup>").expect("workgroup var present");
    let entry_pos = wgsl.find("@compute @workgroup_size(").expect("entry point present");
    assert!(
        workgroup_pos < entry_pos,
        "var<workgroup> must be declared at module scope, before the @compute entry point — \
         found it after, which means it was emitted inside the function body"
    );
}

#[test]
fn test_auto_sync_barrier_found_when_loop_nested_inside_top_level_if() {
    // No explicit `sync` here — relies on the auto-inserted write-phase barrier
    // (`first_loop_index` in transpiler/helpers.rs). The accumulation loop that reads
    // `shared` cross-thread is nested inside a top-level `if`, not a bare top-level
    // `for`/`while` sibling — `first_loop_index` used to only match a bare top-level
    // loop statement directly, so this shape was invisible to it and no barrier at
    // all was emitted before the loop, a real cross-thread race on shared memory.
    let src = r#"
kernel Reduce:
    let [int, 4]'actor shared

    def ():
        let tid = .thread.x
        if tid == 0:
            shared[0] = 0
        if true:
            for i in 0..<4:
                shared[i] = shared[i] + 1
"#;
    let (wgsl, _rs) = wgpu_codegen("auto_sync_nested_if", src);

    assert!(wgsl.contains("workgroupBarrier()"),
        "expected an auto-inserted workgroupBarrier() before the loop nested inside \
         the top-level `if`, even with no explicit `sync`;\ngot:\n{wgsl}");
}

#[test]
fn test_actor_global_atomic() {
    let src = r#"
kernel Histogram:
    mut [int]'actor'global counts
    mut [int]'unified data

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        counts[data[i]] += 1
"#;
    let (wgsl, rs) = wgpu_codegen("actor_global", src);

    assert!(wgsl.contains("atomic<i32>"), "missing atomic type in WGSL");
    assert!(wgsl.contains("atomicAdd"), "missing atomicAdd");
    assert!(rs.contains("COPY_SRC"), "actor global should have COPY_SRC");
}

#[test]
fn test_actor_unified_atomic() {
    let src = r#"
kernel Histogram:
    mut [int]'actor'unified counts
    mut [int]'unified       data

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        counts[data[i]] += 1
"#;
    let (wgsl, rs) = wgpu_codegen("actor_unified", src);

    assert!(wgsl.contains("atomic<i32>"), "missing atomic type in WGSL");
    assert!(wgsl.contains("atomicAdd"), "missing atomicAdd");
    // Same storage-only usage as 'actor'global/'unified — MAP_READ/MAP_WRITE is
    // never combined with the atomic<T> storage buffer itself; host access goes
    // through the staging-buffer copy path instead (see
    // `copy_counts_to_host`/`copy_counts_to_device`), which is what sidesteps the
    // open question of whether WGSL even allows that combination.
    let usage_line = "wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,";
    assert!(rs.contains(usage_line),
        "expected counts_buf's own creation to request only STORAGE|COPY_SRC|COPY_DST \
         (no MAP_READ/MAP_WRITE on the atomic<T> buffer itself);\ngot:\n{rs}");
    // Unlike 'actor'global, 'actor'unified is host-visible — it must get the
    // same read-back/upload accessors 'unified fields get.
    assert!(rs.contains("fn copy_counts_to_host"),
        "expected a host-side copy_counts_to_host() accessor for 'actor'unified;\ngot:\n{rs}");
    assert!(rs.contains("fn copy_counts_to_device"),
        "expected a host-side copy_counts_to_device() accessor for 'actor'unified;\ngot:\n{rs}");
}

#[test]
fn test_gpu_builtins_mapped() {
    let src = r#"
kernel Builtins:
    mut [int]'unified out

    def ():
        let tx = .thread.x
        let bx = .block.x
        let bdx = .block_dim.x
        let gdx = .grid_dim.x
        out[0] = tx + bx + bdx + gdx
"#;
    let (wgsl, _rs) = wgpu_codegen("builtins", src);

    assert!(wgsl.contains("local_invocation_id"), ".thread.x → local_invocation_id");
    assert!(wgsl.contains("workgroup_id"), ".block.x → workgroup_id");
    assert!(wgsl.contains("let bp_bdim = vec3<u32>("), ".block_dim.x → derived from block sizes");
    assert!(wgsl.contains("num_workgroups"), ".grid_dim.x → num_workgroups");
}

#[test]
fn test_gpu_warp_builtins_real_subgroup_path() {
    let src = r#"
kernel WarpBuiltins:
    mut [float]'unified buf

    def ():
        let tid = .thread.x
        let lane = .warp.lane
        let size = .warp.size
        .warp.sync()
        let a = .warp.shuffle_down(buf[tid], 1)
        let b = .warp.shuffle_up(buf[tid], 1)
        let c = .warp.shuffle_xor(buf[tid], 1)
        let d = .warp.shuffle(buf[tid], 0)
        buf[tid] = a + b + c + d + f32(lane) + f32(size)
"#;
    let (wgsl, emulated, _rs, _toml) = run_wgpu("warp_builtins_real", src);

    assert!(wgsl.contains("enable subgroups;"), "expected enable subgroups;\ngot:\n{wgsl}");
    assert!(wgsl.contains("@builtin(subgroup_size)"), "expected @builtin(subgroup_size);\ngot:\n{wgsl}");
    assert!(wgsl.contains("@builtin(subgroup_invocation_id)"), "expected @builtin(subgroup_invocation_id);\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupBarrier()"), "expected subgroupBarrier();\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffleDown("), "expected subgroupShuffleDown;\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffleUp("), "expected subgroupShuffleUp;\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffleXor("), "expected subgroupShuffleXor;\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffle("), "expected subgroupShuffle;\ngot:\n{wgsl}");

    // The emulated fallback module must exist alongside the real one whenever
    // `.warp.*` is used, and never uses the subgroup extension.
    assert!(!emulated.is_empty(), "expected shaders/main_emulated.wgsl to be written");
    assert!(!emulated.contains("enable subgroups;"), "emulated module must not enable subgroups;\ngot:\n{emulated}");
}

#[test]
fn test_gpu_warp_shuffle_emulated_fallback_shape() {
    let src = r#"
kernel WarpEmulated:
    mut [float32]'unified buf

    def ():
        let tid = .thread.x
        .warp.sync()
        let shuffled = .warp.shuffle_down(buf[tid], 1)
        buf[tid] = shuffled
"#;
    let (_wgsl, emulated, _rs, _toml) = run_wgpu("warp_shuffle_emulated", src);

    assert!(emulated.contains("var<workgroup> bp_warp_scratch_warpemulated_f32"),
        "expected a kernel-prefixed f32 workgroup scratch buffer;\ngot:\n{emulated}");
    assert!(emulated.contains("workgroupBarrier()"), "expected workgroupBarrier();\ngot:\n{emulated}");
    assert!(emulated.contains("@builtin(local_invocation_index)"),
        "expected @builtin(local_invocation_index);\ngot:\n{emulated}");
    assert!(emulated.contains("let bp_wsize: u32 = 32u;"), "expected fixed 32-lane fallback constant;\ngot:\n{emulated}");
    assert!(emulated.contains("select("), "expected a select() for the warp-boundary clamp;\ngot:\n{emulated}");
}

/// Regression test for a real bug: `infer_shuffle_elem_type` (and its
/// `collect_shuffle_elem_types_stmts`/`collect_shuffle_types_expr` collector
/// counterparts) only resolved a `Var` expression's type via the kernel's own
/// declared `fields`, with no path at all for a `def()`-body *local* declared
/// via `Stmt::Let` (e.g. `var int v = 0`) — falling back to `f32` regardless of
/// the local's real type. Confirmed via a real wgpu/naga shader-module-creation
/// validation failure on real hardware with the `SUBGROUP` feature disabled
/// (the adapter-without-subgroups path `Emulated` mode exists for): the scratch
/// buffer was declared `array<f32, N>` while an `i32` value was written into it,
/// which `boring build --target wgpu`'s own successful exit code never surfaced
/// (invisible until real `Device::create_shader_module`).
#[test]
fn test_gpu_warp_shuffle_emulated_local_var_non_f32_type() {
    let src = r#"
kernel ShuffleIntKernel:
    mut [float32]'unified out

    init():
        out = [0.0 for i in 0..<32]

    def ():
        let lane = .warp.lane
        var int v = 0
        if lane == 0:
            v = 42
        v = .warp.shuffle(v, 0)
        out[lane] = v as float32
"#;
    let (_wgsl, emulated, _rs, _toml) = run_wgpu("warp_shuffle_emulated_local_var", src);

    assert!(emulated.contains("var<workgroup> bp_warp_scratch_shuffleintkernel_i32: array<i32,"),
        "expected an i32 (not f32) workgroup scratch buffer for a shuffled `int` local;\ngot:\n{emulated}");
    assert!(!emulated.contains("f32") || !emulated.contains("bp_warp_scratch_shuffleintkernel_f32"),
        "must not also declare a stray f32 scratch buffer for this kernel;\ngot:\n{emulated}");
    // The scratch identifier itself must be a clean WGSL identifier -- specifically,
    // it must never contain the narrowing-warning comment `wgsl_scalar` embeds for
    // type-annotation positions (a second, closely related real bug this fix also
    // closes: that comment text was being spliced directly into an identifier).
    assert!(!emulated.contains("bp_warp_scratch_shuffleintkernel_/*"),
        "scratch buffer identifier must not contain an embedded comment;\ngot:\n{emulated}");
    assert!(emulated.contains("bp_warp_scratch_shuffleintkernel_i32[bp_lidx] = v;"),
        "expected the i32 local to be written into the i32 scratch slot directly (no type mismatch);\ngot:\n{emulated}");
}

#[test]
fn test_gpu_warp_not_used_leaves_output_unchanged() {
    let src = r#"
kernel Plain:
    mut [float]'unified buf
    def ():
        let tid = .thread.x
        buf[tid] = buf[tid] * 2.0
"#;
    let (wgsl, emulated, _rs, _toml) = run_wgpu("warp_not_used", src);
    assert!(!wgsl.contains("enable subgroups;"), "no .warp.* usage should never enable subgroups;\ngot:\n{wgsl}");
    assert!(emulated.is_empty(), "no .warp.* usage should not emit an emulated shader file");
}

#[test]
fn test_cargo_toml_deps() {
    let src = r#"
kernel Empty:
    mut [float]'unified data
    def ():
        data[0] = 1.0
"#;
    let (_wgsl, _emulated, _rs, toml) = run_wgpu("cargo_toml", src);

    assert!(toml.contains("wgpu = \"25\""), "missing wgpu dep");
    assert!(toml.contains("bytemuck"), "missing bytemuck dep");
    assert!(toml.contains("pollster"), "missing pollster dep");
    assert!(!toml.contains("winit"), "winit should not be present for compute-only");
}

#[test]
fn test_type_narrowing_int_to_i32() {
    let src = r#"
kernel Narrow:
    mut [int]'unified buf

    def ():
        let int x = 42
        buf[0] = x
"#;
    let (wgsl, _rs) = wgpu_codegen("narrowing", src);

    assert!(wgsl.contains("i32"), "int fields should narrow to i32 in WGSL");
    assert!(!wgsl.contains("i64"), "i64 must not appear in WGSL");
}

#[test]
fn test_int_uint_narrowing_warns_in_generated_wgsl() {
    // `int`/`uint` are 64-bit (isize/usize) on every other GPU backend, but WGSL has no
    // 64-bit integer type -- silently mapping them to i32/u32 here (like the genuinely
    // unsupported 8/16/64/128-bit widths already do via `wgsl_unsupported_width`) must
    // leave a diagnostic in the generated shader instead of narrowing in total silence.
    let src = r#"
kernel Narrow:
    mut [int]'unified buf
    mut [uint]'unified ubuf

    def ():
        let int x = 42
        buf[0] = x
        let uint y = 7
        ubuf[0] = y
"#;
    let (wgsl, _rs) = wgpu_codegen("narrow_warn", src);

    assert!(wgsl.contains("i32"), "int fields should still narrow to i32 in WGSL");
    assert!(wgsl.contains("u32"), "uint fields should still narrow to u32 in WGSL");
    assert!(
        wgsl.contains("64-bit"),
        "narrowing `int`/`uint` to 32-bit on wgpu should emit an explicit diagnostic \
         comment naming the 64-bit narrowing, generated wgsl was:\n{wgsl}"
    );
}

/// Regression test for a real bug: WGSL's `<<`/`>>` require the shift-amount
/// (RHS) operand to be `u32` specifically -- no implicit i32->u32 conversion.
/// Boring's `int` transpiles to `i32` on this backend, so a *variable* or
/// computed shift amount (unlike a literal one, which WGSL's own
/// abstract-int literal inference already coerces) must be wrapped in an
/// explicit `u32(...)` cast by the emitter, or naga rejects the shader at
/// `create_shader_module` time with "automatic conversions cannot convert
/// elements of `i32` to `u32`" -- a failure `cargo build` itself never sees,
/// since WGSL validation happens at shader-creation runtime.
#[test]
fn test_shift_with_variable_amount_casts_to_u32() {
    let src = r#"
kernel ShiftKernel:
    mut [int]'unified buf

    def ():
        let shift_amount = buf[1]
        buf[0] = buf[0] >> shift_amount
        buf[0] = buf[0] << shift_amount
"#;
    let (wgsl, _rs) = wgpu_codegen("shift_variable_amount", src);

    assert!(
        wgsl.contains(">> u32(") && wgsl.contains("<< u32("),
        "expected the variable shift amount to be cast to u32 for both `>>` and `<<`;\ngot:\n{wgsl}"
    );
}

/// End-to-end companion to `test_shift_with_variable_amount_casts_to_u32` --
/// a text-only assertion on the generated WGSL can't catch a codegen mistake
/// that only naga's real shader parser/validator rejects (this is exactly
/// how the bug was found: `boring build --target wgpu` and `cargo build`
/// both succeeded, only real `Device::create_shader_module` failed). Mirrors
/// `test_float_builtin_methods_real_shader_validation`'s pattern. `v = 0xA5`
/// (`10100101`) is chosen so lane 0 (shift 0, bit 0 = 1) and lane 1 (shift 1,
/// bit 1 = 0) disagree -- a broken shift that silently ignored the shift
/// amount (or always shifted by 0) would still happen to produce *some*
/// output without panicking, so the differing expected bits also confirm
/// the shift amount is actually applied per-lane, not just that the shader
/// compiles.
#[test]
fn test_shift_with_variable_amount_real_shader_validation() {
    let src = r#"
kernel ShiftKernel:
    mut [float32]'unified out

    init():
        out = [0.0 for i in 0..<32]

    def ():
        let lane = .warp.lane
        let v = 0xA5
        let shift_amount = lane % 8
        let result = (v >> shift_amount) & 1
        out[lane] = result as float32

mut k = ShiftKernel()
kernel:
    k(block = 32)

print "r0 = {k.out[0]}"
print "r1 = {k.out[1]}"
print "r8 = {k.out[8]}"
"#;
    let (_wgsl, _emulated, _rs, _toml) = run_wgpu("shift_variable_amount_real_shader", src);

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("shift_variable_amount_real_shader");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU (no shader-validation panic from an i32 shift amount), but it failed:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    fn parsed_value(stdout: &str, prefix: &str) -> f32 {
        let line = stdout.lines().find(|l| l.starts_with(prefix))
            .unwrap_or_else(|| panic!("missing '{prefix}' line in stdout:\n{stdout}"));
        line[prefix.len()..].trim().parse::<f32>()
            .unwrap_or_else(|e| panic!("failed to parse '{line}' as f32: {e}"))
    }
    let checks: [(&str, f32); 3] = [
        ("r0 = ", 1.0),
        ("r1 = ", 0.0),
        ("r8 = ", 1.0),
    ];
    for (prefix, expected) in checks {
        let actual = parsed_value(&stdout, prefix);
        assert!(
            (actual - expected).abs() < 1e-4,
            "expected {prefix}~{expected}, got {actual} — full stdout:\n{stdout}"
        );
    }
}

#[test]
fn test_global_buffer_d2h_helper() {
    let src = r#"
kernel Compute:
    mut [float]'global result

    def ():
        result[0] = 1.0
"#;
    let (_wgsl, rs) = wgpu_codegen("global_buffer", src);

    assert!(rs.contains("__boring_gpu_copy_d2h"), "missing D2H staging helper");
    assert!(rs.contains("__boring_gpu_copy_h2d"), "missing H2D staging helper");
    assert!(rs.contains("MAP_READ | wgpu::BufferUsages::COPY_DST"), "staging D2H usages");
}

#[test]
fn test_d2h_staging_buffer_pool_reuse() {
    // Same source as test_global_buffer_d2h_helper -- this test asserts on the
    // *shape* of __boring_gpu_copy_d2h's generated body: repeated readbacks of
    // the same size must reuse a pooled staging buffer instead of allocating a
    // fresh one every call. There's no real GPU in this test harness (these are
    // codegen-shape snapshot tests, see the module doc comment), so we verify
    // the pooling logic is structurally present rather than exercising it at
    // runtime against real device readbacks.
    let src = r#"
kernel Compute:
    mut [float]'global result

    def ():
        result[0] = 1.0
"#;
    let (_wgsl, rs) = wgpu_codegen("d2h_staging_pool", src);

    assert!(rs.contains("thread_local!"), "missing thread_local staging pool declaration:\n{rs}");
    assert!(rs.contains("__BORING_STAGING_POOL"), "missing staging pool storage:\n{rs}");

    // The pool lookup (by exact size match) must happen before falling back to
    // `device.create_buffer` -- i.e. create_buffer is reached only on a pool miss.
    let copy_d2h_start = rs.find("fn __boring_gpu_copy_d2h").expect("missing __boring_gpu_copy_d2h fn");
    let copy_d2h_body = &rs[copy_d2h_start..];
    let pool_lookup_pos = copy_d2h_body.find("pool.iter().position").expect("missing pool lookup by size");
    let create_buffer_pos = copy_d2h_body.find("device.create_buffer").expect("missing create_buffer fallback");
    assert!(pool_lookup_pos < create_buffer_pos,
        "pool lookup must be attempted before falling back to device.create_buffer:\n{copy_d2h_body}");

    // The staging buffer must be unmapped, then returned to the pool -- not dropped.
    let unmap_pos = copy_d2h_body.find("staging.unmap()").expect("missing staging.unmap()");
    let pool_push_pos = copy_d2h_body.find("pool.borrow_mut().push(staging)").expect("missing pool push-back of staging buffer");
    assert!(unmap_pos < pool_push_pos,
        "staging buffer must be fully unmapped before being returned to the pool:\n{copy_d2h_body}");
}

#[test]
fn test_screen_present_and_key() {
    // Minimal game-of-life-style program with Screen + kernel + render loop.
    let src = r#"
kernel Step:
    mut [int, 256]'actor cells_in
    mut [int, 256]'actor cells_out
    let int w

    def ():
        cells_out[0] = cells_in[0]

kernel Render:
    mut [int, 256]'actor pixels
    let int w

    def ():
        pixels[0] = 0

let w = 800
let h = 600
let screen = Screen(Dimension(w, h), title = "Test")
var step = Step(Dimension(w, h))
var render = Render(Dimension(w, h))

kernel:
    loop:
        step(block = (16, 16))
        render(block = (16, 16))
        screen.present(render.pixels)
        if screen.key("\x1B"):
            break
"#;
    let (_wgsl, rs) = wgpu_codegen("screen_present", src);

    assert!(rs.contains("use winit::application::ApplicationHandler"), "missing ApplicationHandler import");
    assert!(rs.contains("use winit::event::{WindowEvent, ElementState}"), "missing winit event imports");
    assert!(rs.contains("use winit::keyboard::{Key, NamedKey}"), "missing NamedKey import");
    assert!(rs.contains("use winit::window::{Window, WindowAttributes, WindowId}"), "missing WindowAttributes import");
    assert!(rs.contains("fn resumed(&mut self, event_loop: &ActiveEventLoop)"), "missing resumed method");
    assert!(rs.contains("fn window_event(&mut self, event_loop: &ActiveEventLoop"), "missing window_event method");
    assert!(rs.contains("let __attrs = WindowAttributes::default()"), "window attributes built from the defaults");
    assert!(rs.contains("event_loop.create_window(__attrs)"), "window created via create_window");
    assert!(rs.contains("fn __boring_present_buffer("), "missing present buffer helper");
    assert!(rs.contains("__boring_present_buffer(&self.device, &self.queue, self.surface.as_ref().unwrap()"), "present call");
    assert!(rs.contains("if self.__keys.contains(&__boring_key_name(\"\\\\x1B\")) { event_loop.exit(); }"), "Escape key exits");
    assert!(rs.contains("surface.get_capabilities(&self.adapter)"), "adapter used for surface caps");
    assert!(rs.contains("surface.configure(&self.device,"), "surface configured");
    assert!(rs.contains("EventLoop::new()"), "event loop created");
    assert!(rs.contains("event_loop.run_app(&mut app)"), "run_app used");
    assert!(rs.contains("NamedKey::Escape"), "Escape named key in key handler");
}

// ── `with` GPU-residency materialization (docs/scoped-access-blocks.md) ────────
//
// `let py'kernel'unified = k.y` followed by `with py:` should read the kernel field
// back exactly once (`k.copy_y_to_host()`), regardless of how many times the
// block's body indexes `py` — the actual bug this exists to fix, confirmed against
// `examples/vector_add_gpu.br`'s `for i in 0..<n: print k.result[i]`, which today
// re-reads the whole buffer on every loop iteration with no `with` available.

#[test]
fn test_with_gpu_resident_read_only_single_readback() {
    let src = r#"
kernel Saxpy:
    let float alpha
    let [float]'unified x
    mut [float]'unified y

    init(float a, [float]'unified xs, [float]'unified ys):
        alpha = a
        x = xs
        y = ys

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        y[i] = alpha * x[i] + y[i]

var [float] hx = [0.0, 1.0]
var [float] hy = [1.0, 1.0]
mut k = Saxpy(2.0, hx, hy)
kernel:
    k(block = 2)

let [float]'kernel'unified py = k.y
with py:
    for i in 0..<2:
        print "{py[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_read", src);

    // Exactly one readback CALL (`k.copy_y_to_host()`), bound before the loop — not
    // one per iteration. `copy_y_to_host` alone also matches the method's own `fn`
    // definition, so count the call form specifically.
    assert_eq!(rs.matches("k.copy_y_to_host()").count(), 1, "expected exactly one copy_y_to_host call:\n{rs}");
    assert!(rs.contains("let py = k.copy_y_to_host()"), "missing single materializing readback:\n{rs}");
    // Read-only block: no write-back targeting `py` (the constructor's own initial
    // upload, `k.copy_y_to_device(&hy...)`, is unrelated and expected), and the
    // alias binding isn't `mut`.
    assert!(!rs.contains("k.copy_y_to_device(&py"), "read-only with-block should not write back:\n{rs}");
    assert!(!rs.contains("let mut py"), "read-only alias should not be `mut`:\n{rs}");
    // No leftover placeholder pointer type for the plain host arrays.
    assert!(!rs.contains("*mut Vec"), "gpu'unified/gpu'global should emit a plain Vec, not a pointer:\n{rs}");
}

#[test]
fn test_with_gpu_resident_write_back_on_mutation() {
    let src = r#"
kernel Saxpy:
    let float alpha
    let [float]'unified x
    mut [float]'unified y

    init(float a, [float]'unified xs, [float]'unified ys):
        alpha = a
        x = xs
        y = ys

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        y[i] = alpha * x[i] + y[i]

var [float] hx = [0.0, 1.0]
var [float] hy = [1.0, 1.0]
mut k = Saxpy(2.0, hx, hy)
kernel:
    k(block = 2)

let [float]'kernel'unified py = k.y
with py:
    py[0] = 0.0
"#;
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_write", src);

    assert!(rs.contains("let mut py = k.copy_y_to_host()"), "mutating block needs a `mut` alias:\n{rs}");
    assert!(rs.contains("k.copy_y_to_device(&py"), "write-back should target the kernel field:\n{rs}");
    // The constructor's own initial upload (`k.copy_y_to_device(&hy...)`) is a
    // separate, expected call — only the with-block's write-back targets `py`.
    assert_eq!(rs.matches("k.copy_y_to_device(&py").count(), 1, "expected exactly one write-back call targeting py:\n{rs}");
}

#[test]
fn test_with_gpu_resident_infers_qualifier_without_annotation() {
    // Same as test_with_gpu_resident_read_only_single_readback, but `py` has no
    // explicit 'kernel'unified annotation at all — the qualifier is inferred from `k.y`
    // being a 'unified array field on a tracked kernel instance.
    let src = r#"
kernel Saxpy:
    let float alpha
    let [float]'unified x
    mut [float]'unified y

    init(float a, [float]'unified xs, [float]'unified ys):
        alpha = a
        x = xs
        y = ys

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        y[i] = alpha * x[i] + y[i]

var [float] hx = [0.0, 1.0]
var [float] hy = [1.0, 1.0]
mut k = Saxpy(2.0, hx, hy)
kernel:
    k(block = 2)

let py = k.y
with py:
    for i in 0..<2:
        print "{py[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_inferred", src);

    assert_eq!(rs.matches("k.copy_y_to_host()").count(), 1, "expected exactly one copy_y_to_host call:\n{rs}");
    assert!(rs.contains("let py = k.copy_y_to_host()"), "missing single materializing readback:\n{rs}");
    assert!(!rs.contains("*mut Vec"), "inferred qualifier should emit a plain Vec, not a pointer:\n{rs}");
}

// ── Interprocedural residency: `with` surviving a function-call boundary ──────
//
// The intra-procedural tests above cover a kernel instance and its field read living
// in the *same* scope. These tests cover the actual motivating case
// (docs/scoped-access-blocks.md): a free function returning a `'kernel'unified`-typed
// value, chained into a second call, with only the *final* consumer paying a host
// round-trip — the shape of whisper-boring's `linear_gpu` -> `gelu_gpu` -> `linear_gpu`.

const SCALE_GPU_KERNEL: &str = r#"
kernel Saxpy:
    let float alpha
    let [float]'unified x
    mut [float]'unified y

    init(float a, [float]'unified xs, [float]'unified ys):
        alpha = a
        x = xs
        y = ys

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        y[i] = alpha * x[i] + y[i]
"#;

#[test]
fn test_with_gpu_resident_call_chain_no_intermediate_roundtrip() {
    // `scale_gpu` wraps kernel construction+dispatch+field-read behind a function
    // boundary with an explicit `'kernel'unified` return type — exactly the shape a
    // real kernel-launcher wrapper (`linear_gpu`, etc.) uses. Called twice in a
    // chain: the second call's argument is the first call's still-resident return
    // value, and only the final `with` pays a real device->host transfer.
    let src = format!(r#"{SCALE_GPU_KERNEL}
req [float]'kernel'unified scale_gpu([float] xv, float factor):
    var [float] zero = [0.0, 0.0]
    mut k = Saxpy(factor, xv, zero)
    kernel:
        k(block = 2)
    k.y

var [float] ha = [1.0, 2.0]
let [float]'kernel'unified fc = scale_gpu(ha, 2.0)
let [float]'kernel'unified fc2 = scale_gpu(fc, 3.0)
with fc2:
    print "{{fc2[0]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_call_chain", &src);

    // Signature: dual-typed param (the only use of `xv` is as a kernel-constructor
    // argument at a 'unified field position), resident return type.
    assert!(rs.contains("fn scale_gpu(xv: BoringKernelArg<f64>, factor: f64) -> BoringKernelArg<f64>"),
        "expected dual-typed param + resident return signature:\n{rs}");

    // Tail expression returns the buffer directly -- no download.
    assert!(rs.contains("BoringKernelArg::Resident(std::sync::Arc::clone(&k.y_buf)"),
        "tail expression should return a Resident handle, not a download:\n{rs}");

    // Kernel-construction consumes `xv` via the dual-mode branch, not an
    // unconditional upload.
    assert!(rs.contains("match &xv {"), "constructor argument for `xv` should branch on BoringKernelArg:\n{rs}");
    assert!(rs.contains("k.x_buf = std::sync::Arc::clone(buf);"), "read-only resident input should reuse the same GPU allocation:\n{rs}");
    assert!(rs.contains("k.rebuild_bind_group();"), "resident branch should rebuild the bind group:\n{rs}");

    // Call sites: `ha` (a plain host array) is wrapped; `fc` (already resident) is
    // passed straight through, not re-wrapped as a host upload.
    assert!(rs.contains("scale_gpu(BoringKernelArg::Host(ha.clone())"), "plain host argument should wrap as BoringKernelArg::Host:\n{rs}");
    assert!(rs.contains("scale_gpu(fc.clone()"), "already-resident argument should pass straight through:\n{rs}");
    assert!(!rs.contains("BoringKernelArg::Host(fc"), "resident value should not be re-wrapped as a host upload:\n{rs}");

    // The whole point: no kernel-field download (`copy_y_to_host`) happens anywhere
    // in the chain -- only the final `with fc2:` materializes, via the free d2h
    // helper directly on the retained buffer (no live kernel instance to call
    // `copy_y_to_host` on at that point).
    assert_eq!(rs.matches("copy_y_to_host()").count(), 0, "no kernel-field download should occur anywhere in the chain:\n{rs}");
    assert!(rs.contains("__boring_gpu_copy_d2h::<f32>(&__boring_gpu_device(), &__boring_gpu_queue(), buf)"),
        "final `with` should materialize via the free d2h helper on the raw buffer:\n{rs}");
    // Read-only `with` block -- no write-back for `fc2` specifically. (The bare
    // `__boring_gpu_copy_h2d` helper still appears elsewhere in the file -- it's
    // also what `copy_x_to_device`/`copy_y_to_device` call internally for the
    // kernel's own H2D uploads, unrelated to this `with` block.)
    assert!(!rs.contains("__fc2_buf"), "read-only with-block should not capture a buffer handle for write-back:\n{rs}");
}

#[test]
fn packed_resident_return_keeps_exact_element_count() {
    let src = r#"
kernel CopyBytes:
    mut [uint8]'unified out
    let int n
    init(int nn):
        n = nn
        out = [0 as uint8 for ..<nn]
    def ():
        let i = .thread.x
        if i < n:
            out[i] = (i + 1) as uint8

req [uint8]'kernel'unified make_bytes(int n):
    mut k = CopyBytes(n)
    kernel:
        k(block = 4, grid = (n + 3) / 4)
    k.out
"#;
    let (_wgsl, rs) = wgpu_codegen("packed_resident_exact_len", src);
    assert!(rs.contains("BoringKernelArg::Resident(std::sync::Arc::clone(&k.out_buf), k.out_len)"),
        "packed resident return must carry the separately tracked logical length:\n{rs}");
    assert!(!rs.contains("k.out_buf.size() as usize) / std::mem::size_of::<u8>()"),
        "padded wgpu byte size is not the logical uint8 element count:\n{rs}");
}

#[test]
fn test_with_gpu_resident_call_infers_qualifier_without_annotation() {
    // Same shape as the chain test above, but neither `fc` nor `fc2` has an explicit
    // `'kernel'unified` annotation -- inferred from `scale_gpu`'s own declared return
    // type, mirroring the same-scope `let py = k.y` inference precedent.
    let src = format!(r#"{SCALE_GPU_KERNEL}
req [float]'kernel'unified scale_gpu([float] xv, float factor):
    var [float] zero = [0.0, 0.0]
    mut k = Saxpy(factor, xv, zero)
    kernel:
        k(block = 2)
    k.y

var [float] ha = [1.0, 2.0]
let fc = scale_gpu(ha, 2.0)
let fc2 = scale_gpu(fc, 3.0)
with fc2:
    print "{{fc2[0]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_call_chain_inferred", &src);

    assert!(rs.contains("fn scale_gpu(xv: BoringKernelArg<f64>, factor: f64) -> BoringKernelArg<f64>"),
        "expected dual-typed param + resident return signature:\n{rs}");
    assert!(rs.contains("scale_gpu(fc.clone()"), "already-resident argument should pass straight through even without an explicit annotation:\n{rs}");
    assert_eq!(rs.matches("copy_y_to_host()").count(), 0, "no kernel-field download should occur anywhere in the chain:\n{rs}");
}

// ── Regression tests: consuming a resident value, not just returning one ──────
//
// The tests above all cover the *return* side of interprocedural residency —
// `scale_gpu`'s own kernel dispatches with a literal block size, never indexing or
// sizing off its dual-typed param. Real kernel-launcher wrappers (whisper-boring's
// `linear_gpu`/`gelu_gpu`/etc.) size their dispatch block off the very array they
// pass to the kernel constructor (`k(block = x.length)`), and real pipelines chain
// kernels directly in one scope as often as across a function boundary. These three
// mirror that consuming shape exactly.

const SCALE_ONE_ARG_KERNEL: &str = r#"
kernel Scale:
    let float factor
    let [float]'unified x
    mut [float]'unified y

    init(float f, [float]'unified xs):
        factor = f
        x = xs
        y = [0.0 for ..<xs.length]

    def ():
        let i = .thread.x
        y[i] = x[i] * factor
"#;

#[test]
fn test_with_gpu_resident_call_param_used_for_dispatch_size() {
    // `x` is used both as the kernel-constructor argument AND to size the dispatch
    // block (`x.length`) -- a second, non-constructor use that must NOT disqualify
    // the exclusive-ctor-arg scan (`ast::scan_var_call_arg_uses`), since a dual-typed
    // `BoringKernelArg<T>` can answer `.length` without ever materializing.
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'kernel'unified scale_gpu([float] x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let fc = scale_gpu(a, 2.0)
    let fc2 = scale_gpu(fc, 3.0)
    with fc2:
        for i in 0..<3:
            print "{{fc2[i]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_param_dispatch_size", &src);

    assert!(rs.contains("fn scale_gpu(x: BoringKernelArg<f64>, factor: f64) -> BoringKernelArg<f64>"),
        "x.length use should not disqualify x from the dual-typed param treatment:\n{rs}");
    assert!(rs.contains("match &x {"), "constructor argument for `x` should branch on BoringKernelArg:\n{rs}");
    assert!(rs.contains("k.x_buf = std::sync::Arc::clone(buf);"), "read-only resident input should reuse the same GPU allocation:\n{rs}");
    assert!(rs.contains("(x.len()) as usize"), "x.length should compile via BoringKernelArg::len(), not a bare field access:\n{rs}");
    assert!(!rs.contains("x::length") && !rs.contains("x::count"), "x.length must not be emitted as a module path:\n{rs}");

    // Chain: plain host array wraps, already-resident value passes straight through.
    assert!(rs.contains("scale_gpu(BoringKernelArg::Host(a.clone())"), "plain host argument should wrap as BoringKernelArg::Host:\n{rs}");
    assert!(rs.contains("scale_gpu(fc.clone()"), "already-resident argument should pass straight through:\n{rs}");
    assert!(!rs.contains("scale_gpu(&fc"), "the by-ref array-argument convention must not apply to a dual-typed param:\n{rs}");
}

#[test]
fn test_resident_input_to_mutable_kernel_field_keeps_copy_semantics() {
    let src = r#"
kernel Mutate:
    mut [float]'unified x
    init([float]'unified input):
        x = input
    def ():
        x[.thread.x] += 1.0

req [float]'kernel'unified mutate_gpu([float] x) throws:
    mut k = Mutate(x)
    kernel:
        k(block = x.length)
    k.x
"#;
    let (_wgsl, rs) = wgpu_codegen("resident_mutable_input_copy", src);
    assert!(rs.contains("k.x_buf = __boring_gpu_copy_d2d(&__boring_gpu_device(), &__boring_gpu_queue(), buf);"),
        "mutable resident input must retain independent-value semantics:\n{rs}");
    assert!(!rs.contains("k.x_buf = std::sync::Arc::clone(buf);"),
        "mutable kernel fields must not alias the caller's allocation:\n{rs}");
}

#[test]
fn test_with_gpu_resident_call_param_explicit_annotation_used_for_dispatch_size() {
    // Same shape as above, but `x` carries an explicit `'kernel'unified` annotation --
    // the annotation must not survive into the emitted parameter type (it should
    // still collapse to the same dual-typed `BoringKernelArg<T>` signature, matching the
    // return-type case), and the same `x.length` use must not disqualify it either.
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'kernel'unified scale_gpu([float]'kernel'unified x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let fc = scale_gpu(a, 2.0)
    let fc2 = scale_gpu(fc, 3.0)
    with fc2:
        for i in 0..<3:
            print "{{fc2[i]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("with_gpu_resident_param_annotated_dispatch_size", &src);

    assert!(rs.contains("fn scale_gpu(x: &BoringKernelArg<f64>, factor: f64) -> BoringKernelArg<f64>"),
        "an explicit 'kernel'unified annotation on the param should collapse to BoringKernelArg<T>, not a plain Vec:\n{rs}");
    assert!(rs.contains("match &x {"), "constructor argument for `x` should branch on BoringKernelArg:\n{rs}");
    assert!(rs.contains("scale_gpu(&fc"), "already-resident argument should pass straight through:\n{rs}");
}

#[test]
fn test_kernel_constructor_consumes_resident_local_no_function_boundary() {
    // No function boundary at all: `k1.y` is aliased to `fc` (`gpu_resident_vars` --
    // a pure compile-time alias with no Rust binding) and then used directly as
    // `k2`'s constructor argument. This isolates the constructor-argument-consumption
    // gap from the fn-parameter dual-typing above -- `fc` never has a Rust identifier
    // to type `BoringKernelArg<T>` in the first place, so the fix must reach into
    // `gpu_resident_vars` directly rather than going through that enum at all.
    let src = r#"
kernel Scale:
    let float factor
    let [float]'unified x
    mut [float]'unified y

    init(float f, [float]'unified xs):
        factor = f
        x = xs
        y = [0.0 for ..<xs.length]

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    mut k1 = Scale(2.0, a)
    kernel:
        k1(block = 3)
    let fc = k1.y

    mut k2 = Scale(3.0, fc)
    kernel:
        k2(block = 3)
    let fc2 = k2.y

    with fc2:
        for i in 0..<3:
            print "{fc2[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("kernel_ctor_consumes_resident_local", src);

    // The second kernel gets its own device-to-device copy of the first
    // kernel's buffer -- no host round-trip, no dangling reference to a `fc`
    // Rust binding that never exists, and (unlike a bare `Arc::clone`, which
    // would silently alias the same `wgpu::Buffer` between k1 and k2) still
    // correct if `k1` were dispatched again afterward.
    assert!(rs.contains("k2.x_buf = __boring_gpu_copy_d2d(&__boring_gpu_device(), &__boring_gpu_queue(), &k1.y_buf);"),
        "second kernel's x field should get a real device-to-device copy of the first kernel's y buffer:\n{rs}");
    assert!(rs.contains("k2.rebuild_bind_group();"), "buffer aliasing should rebuild the bind group:\n{rs}");
    assert!(!rs.contains("k2.copy_x_to_device"), "no host upload should happen for a resident-aliased argument:\n{rs}");
    assert!(!rs.contains("&fc") && !rs.contains("(fc)") && !rs.contains("fc.iter()"),
        "`fc` has no Rust binding at all -- it must never appear as a bare identifier:\n{rs}");

    // `Scale`'s own `y = [0.0 for ..<xs.length]` zero-fill, for k2, must size off the
    // aliased buffer's own length -- not the nonexistent `xs` init-param identifier
    // (the `xs::length` bug) and not a stale reference to `fc`.
    assert!(rs.contains("k2.copy_y_to_device(&vec![(0) as f32; ((k1.y_buf.size() as usize / std::mem::size_of::<f32>())) as usize]);"),
        "k2's output zero-fill should size off k1's buffer directly:\n{rs}");
    assert!(!rs.contains("xs::length") && !rs.contains("xs::count"), "init-param length must not be emitted as a module path:\n{rs}");
}

// ── Transitive parameter propagation: a wrapper function forwarding to another
// Boring function (not a raw kernel constructor) qualifies too, any number of
// call-graph hops deep — see `Checker::collect_gpu_arg_params`'s fixed point.

#[test]
fn test_fn_gpu_arg_param_transitive_two_hop_wrapper() {
    // `wrap_scale_gpu` forwards its own parameter straight into `scale_gpu` — not a
    // raw kernel constructor — so it only qualifies via the *transitive* fixed point,
    // one call-graph hop beyond the base case `scale_gpu` itself uses. Exercises the
    // actual gap this fix closes: a caller passing an already-resident value into the
    // wrapper (confirmed against a real `cargo check` failure before this fix:
    // `BoringKernelArg::Host(xv.clone())` passed where `scale_gpu` expects
    // `BoringKernelArg<f64>` directly).
    let src = format!(r#"{SCALE_GPU_KERNEL}
req [float]'kernel'unified scale_gpu([float] xv, float factor):
    var [float] zero = [0.0, 0.0]
    mut k = Saxpy(factor, xv, zero)
    kernel:
        k(block = 2)
    k.y

req [float]'kernel'unified wrap_scale_gpu([float] xv, float factor):
    scale_gpu(xv, factor)

var [float] ha = [1.0, 2.0]
let [float]'kernel'unified fc = scale_gpu(ha, 2.0)
let [float]'kernel'unified fc2 = wrap_scale_gpu(fc, 3.0)
with fc2:
    print "{{fc2[0]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("fn_gpu_arg_param_transitive_wrapper", &src);

    // The wrapper's own parameter should be dual-typed too, transitively.
    assert!(rs.contains("fn wrap_scale_gpu(xv: BoringKernelArg<f64>, factor: f64) -> BoringKernelArg<f64>"),
        "wrapper's forwarded parameter should qualify transitively:\n{rs}");
    // Forwarding `xv` into `scale_gpu(xv, factor)` inside the wrapper must pass the
    // enum straight through, not re-wrap it as a host upload.
    assert!(rs.contains("scale_gpu(xv.clone()"), "forwarded resident parameter should pass straight through:\n{rs}");
    assert!(!rs.contains("BoringKernelArg::Host(xv"), "forwarded resident parameter must not be re-wrapped as a host upload:\n{rs}");
    // Call site: an already-resident value (`fc`) passed into the wrapper passes
    // straight through too.
    assert!(rs.contains("wrap_scale_gpu(fc.clone()"), "already-resident argument into the wrapper should pass straight through:\n{rs}");
    assert_eq!(rs.matches("copy_y_to_host()").count(), 0, "no kernel-field download should occur anywhere in the chain:\n{rs}");
}

#[test]
fn test_fn_gpu_arg_param_disqualified_when_any_use_is_not_qualifying() {
    // `x` is used in TWO call positions inside `mixed_use`: one at a genuinely
    // qualifying position (`scale_gpu`'s own dual-typed param, transitively valid),
    // and one at a plain, non-qualifying function (`plain_use`, an ordinary host
    // consumer). The "exclusively qualifying, everywhere in the body" rule must still
    // hold under the transitive fixed point — a single disqualifying use anywhere
    // disqualifies the whole parameter, even though another use of the same
    // parameter would, on its own, have qualified.
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'kernel'unified scale_gpu([float] x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

def plain_use([float] x, float factor):
    print "{{x[0]}}"

req [float]'kernel'unified mixed_use([float] x, float factor):
    plain_use(x, factor)
    scale_gpu(x, factor)

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let fc = mixed_use(a, 2.0)
    with fc:
        print "{{fc[0]}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("fn_gpu_arg_param_disqualified_mixed_use", &src);

    assert!(!rs.contains("fn mixed_use(x: BoringKernelArg<f64>"),
        "a parameter with any non-qualifying use anywhere must not dual-type, even transitively:\n{rs}");
}

// ── Tuple-return residency chaining (`mha_step_gpu`-style: a function returning
// `([float]'kernel'unified, [float], ...)`, chaining whichever tail-tuple elements are
// themselves resident while leaving genuinely host-side elements alone) ──────────

#[test]
fn test_gpu_resident_tuple_return_chains_with_explicit_opt_in() {
    // `tuple_fn` returns a tuple whose first element is itself GPU-resident (chained
    // from a kernel-wrapper call) and whose second element is a genuinely host-side
    // array. The tail expression is a bare tuple literal `(doubled, side)` — the case
    // `try_emit_gpu_resident_tuple_return` (emit_kernel.rs) exists for. At the call
    // site, the destructured binding `r` carries an *explicit* `'kernel'unified` opt-in
    // annotation -- unlike the single-value interprocedural case, a resident tuple
    // position stays resident only when asked to (see `emit_resident_tuple_destructure`'s
    // doc for why the default has to run the other way for tuples).
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'kernel'unified scale_gpu([float] x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

req ([float]'kernel'unified, [float]) tuple_fn([float] x, float factor):
    let [float]'kernel'unified doubled = scale_gpu(x, factor)
    let [float] side = [1.0, 2.0]
    (doubled, side)

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let ([float]'kernel'unified r, [float] s) = tuple_fn(a, 2.0)
    with r:
        print "{{r[0]}}"
    print "{{s.length}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("gpu_resident_tuple_return", &src);

    // Return type: position 0 collapses to BoringKernelArg<T>, position 1 stays Vec<T>.
    assert!(rs.contains("fn tuple_fn(x: BoringKernelArg<f64>, factor: f64) -> (BoringKernelArg<f64>, Vec<f64>)"),
        "tuple return type should substitute BoringKernelArg<T> only at the resident position:\n{rs}");
    // Tail expression: element 0 (already a resident local) passes through as a
    // clone, no download; element 1 emits normally.
    // (`side` is the last use of its local, so the last-use analysis moves it: no `.clone()`.)
    assert!(rs.contains("(doubled.clone(), side)"),
        "resident tuple element should pass through as a clone, not a download:\n{rs}");
    assert_eq!(rs.matches("copy_y_to_host()").count(), 0, "no kernel-field download should occur inside tuple_fn:\n{rs}");
    // Destructure at the call site: `r` opted in explicitly, so it binds straight
    // from the call with no materialization; `s` is an ordinary Vec<f64>.
    assert!(rs.contains("let (r, s) = tuple_fn(BoringKernelArg::Host(a.clone()), 2.0);"),
        "opted-in destructure should bind straight from the call, no extra materialization:\n{rs}");
}

#[test]
fn test_gpu_resident_tuple_return_destructure_materializes_by_default() {
    // Same shape as above, but the destructure has NO annotation at all on `r` —
    // the default for a resident tuple position, since tuple destructuring predates
    // this residency feature everywhere in real code (every existing unannotated
    // `let (a, b, c) = some_tuple_fn(...)` already assumes a plain, immediately
    // usable value — see `Checker::check_let_destructure`'s doc for the real
    // `cargo check` failure an opt-*out* default caused against `test_math_gpu.br`).
    // Materializes right at the destructure through a temp binding, since the call
    // must run exactly once (no re-invoking `tuple_fn` to materialize a second time).
    let src = format!(r#"{SCALE_ONE_ARG_KERNEL}
req [float]'kernel'unified scale_gpu([float] x, float factor):
    mut k = Scale(factor, x)
    kernel:
        k(block = x.length)
    k.y

req ([float]'kernel'unified, [float]) tuple_fn([float] x, float factor):
    let [float]'kernel'unified doubled = scale_gpu(x, factor)
    let [float] side = [1.0, 2.0]
    (doubled, side)

def main() throws:
    var [float] a = [1.0, 2.0, 3.0]
    let (r, s) = tuple_fn(a, 2.0)
    print "{{r[0]}}"
    print "{{s.length}}"
"#);
    let (_wgsl, rs) = wgpu_codegen("gpu_resident_tuple_return_default_materialize", &src);

    // The call still runs exactly once, into per-position temp bindings.
    assert_eq!(rs.matches("tuple_fn(BoringKernelArg::Host(a.clone())").count(), 1,
        "tuple_fn should be called exactly once, not re-invoked to materialize a second time:\n{rs}");
    // The unannotated resident position is materialized via a temp binding, not left as a raw enum.
    assert!(rs.contains("BoringKernelArg::Resident(buf, _) => __boring_gpu_copy_d2h::<f32>(&__boring_gpu_device(), &__boring_gpu_queue(), &buf)"),
        "unannotated resident position should materialize through the free d2h helper by default:\n{rs}");
    assert!(rs.contains("let r ="), "default-materialized binding should still end up bound to the plain name `r`:\n{rs}");
}

// ─── typed KernelError ─────────────────────────────────────────────────────────

#[test]
fn dispatch_pushes_outofmemory_and_validation_scopes_and_wraps_typed_gpu_error() {
    // Previously only `ErrorFilter::Validation` was pushed, and any error
    // collapsed into one generic formatted-string message -- no way to tell
    // an out-of-memory failure apart from a rejected launch config, and no
    // way for Boring source to `catch` a specific cause at all. `wgpu`
    // already exposes `ErrorFilter::OutOfMemory` unused (confirmed against
    // real wgpu 22.1.0 source); this wires it up alongside Validation and
    // classifies each into a typed `KernelError` variant wrapped in
    // `BoringError::Other`, exactly the same mechanism `throws CalcError`
    // already uses (`book.md`), so `catch KernelError.OutOfMemory:` genuinely
    // dispatches -- verified end to end via a real `cargo check` against
    // real wgpu.
    let (_, rs) = wgpu_codegen("gpu_error_scopes", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = .thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);"),
        "expected an OutOfMemory error scope pushed alongside Validation;\ngot:\n{rs}");
    assert!(rs.contains("self.device.push_error_scope(wgpu::ErrorFilter::Validation);"),
        "expected the existing Validation error scope to still be pushed;\ngot:\n{rs}");
    assert!(rs.contains("BoringError::Other(std::any::TypeId::of::<KernelError>(), Box::new(KernelError::LaunchError)"),
        "expected the Validation-scope error to classify as KernelError::LaunchError, typed via BoringError::Other;\ngot:\n{rs}");
    assert!(rs.contains("BoringError::Other(std::any::TypeId::of::<KernelError>(), Box::new(KernelError::OutOfMemory)"),
        "expected the OutOfMemory-scope error to classify as KernelError::OutOfMemory, typed via BoringError::Other;\ngot:\n{rs}");
    assert!(rs.contains("enum KernelError") && rs.contains("OutOfMemory,") && rs.contains("DeviceLost,"),
        "expected the built-in KernelError enum (all 6 documented variants) in the generated prelude;\ngot:\n{rs}");
}

#[test]
fn catch_gpu_error_by_variant_downcasts_correctly() {
    // End-to-end: a Boring `catch KernelError.OutOfMemory:` inside a `try:`
    // wrapping a `kernel:` dispatch must lower to a real BoringError
    // downcast + variant match, the same codegen shape `catch CalcError.X:`
    // already produces for a user-declared enum (book.md:6565) -- confirmed
    // this compiles clean via a real `cargo check` against real wgpu.
    let (_, rs) = wgpu_codegen("gpu_error_catch", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = .thread.x
        buf[tid] = buf[tid] * 2.0

def run() throws:
    let data = [1.0, 2.0]
    mut k = Scale(data)
    try:
        kernel:
            k(block = 2)
    catch KernelError.OutOfMemory:
        print "out of memory"
    catch KernelError.LaunchError:
        print "launch error"
"#);
    assert!(rs.contains("__tid == std::any::TypeId::of::<KernelError>()"),
        "expected a TypeId-gated downcast for KernelError;\ngot:\n{rs}");
    assert!(rs.contains(".downcast_ref::<KernelError>()"),
        "expected a downcast_ref::<KernelError> call at the catch site;\ngot:\n{rs}");
    assert!(rs.contains("KernelError::OutOfMemory =>") && rs.contains("KernelError::LaunchError =>"),
        "expected both catch arms to match on the specific KernelError variant;\ngot:\n{rs}");
}

#[test]
fn test_kernel_output_field_plain_array_literal_sized_correctly() {
    // `out`'s init-body assignment is a plain bracketed literal (`ExprKind::Array`),
    // not the `[value for ..<count]` fill (`ExprKind::ArrayFill`) that
    // `kernel_output_fill_map` used to be the only pattern recognized for. Before the
    // fix, this field's buffer was never covered by that map at all, so it stayed at
    // `new()`'s placeholder size (one `f32`, `4u64` bytes -- see wgpu::host's
    // `emit_kernel_new`) instead of the 8 elements the literal actually declares --
    // a real "index out of bounds: the len is 1 but the index is 1" panic on readback,
    // confirmed via a real `cargo run` against the generated project.
    let src = r#"
kernel PlainInit:
    mut [float]'unified out
    let [float]'unified vals
    let int n

    init([float]'unified data, int size):
        vals = data
        n    = size
        out  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def ():
        let tid = .thread.x
        out[tid] = vals[tid] * 2.0

let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
mut k = PlainInit(data, 8)
kernel:
    k(block = 8)
let result = k.out
with result:
    for i in 0..<8:
        print "{i}: {result[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("kernel_output_field_plain_array_literal", src);

    // All 8 literal elements must be uploaded verbatim -- sizing `out_buf` to 8
    // `f32`s (32 bytes) via `copy_out_to_device`'s own `data.len()`-based resize,
    // not left at the constructor's placeholder allocation.
    assert!(
        rs.contains("k.copy_out_to_device(&vec![(0) as f32, (0) as f32, (0) as f32, (0) as f32, (0) as f32, (0) as f32, (0) as f32, (0) as f32]);"),
        "expected the 8-element literal to be uploaded verbatim via copy_out_to_device;\ngot:\n{rs}"
    );
}

#[test]
fn test_kernel_output_field_array_comp_with_loop_var_sized_correctly() {
    // `dst`'s init-body assignment is the *bound* comprehension form (`[0.0 for i in
    // 0..<4]`, `ExprKind::ArrayComp`) — syntactically just as valid as the unbound
    // fill form (`[0.0 for ..<4]`, `ExprKind::ArrayFill`) already covered above, and
    // semantically identical here since `i` never appears in the value expression.
    // Before this fix, `kernel_output_fill_map` recognized `ArrayFill`/`Array` but
    // not `ArrayComp`, so this field's `copy_dst_to_device` resize call was silently
    // dropped entirely — `dst_buf` stayed at `new()`'s one-`f32` placeholder size,
    // a real "index out of bounds: the len is 1 but the index is 1" panic on
    // readback, confirmed via a real `cargo run` against the generated project.
    let src = r#"
kernel Probe:
    let [float32]'global src
    mut [float32]'unified dst
    let int a
    let int b

    init([float32]'global s, int aa, int bb):
        src = s
        a = aa
        b = bb
        dst = [0.0 for i in 0..<4]

    def ():
        let cell = .thread.x
        if cell < a * b:
            dst[cell] = src[cell]

let src = [1.0, 2.0, 3.0, 4.0]
mut k = Probe(src, 2, 2)
kernel:
    k(block = 256, grid = 1)
let result = k.dst
with result:
    for i in 0..<4:
        print "{result[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("kernel_output_field_array_comp_loop_var", src);

    assert!(
        rs.contains("k.copy_dst_to_device(&(0..(4) as usize).map(|i| (0) as f32).collect::<Vec<f32>>());"),
        "expected the bound-comprehension fill to resize+upload dst via copy_dst_to_device;\ngot:\n{rs}"
    );
}

#[test]
fn test_kernel_output_field_array_comp_inside_host_wrapper_fn() {
    // Same fix as `test_kernel_output_field_array_comp_with_loop_var_sized_correctly`,
    // but with the kernel constructed+dispatched *inside a separate `pub req ...
    // throws` host wrapper function* returning a `'kernel'unified` value — the
    // idiomatic pattern every kernel wrapper in boring-llm's `math_gpu.br` uses
    // (`linear_gpu`, `rope_apply_gpu`, `transpose_gpu`, ...), as opposed to inline
    // construction directly in `main()`/`boring_main()`. `emit_kernel_construction`
    // is reached through the same statement emitter regardless of which function
    // body it's in, so this must produce identical codegen to the top-level case —
    // pinned here explicitly since that shared-codegen assumption is exactly what a
    // regression could quietly break.
    let src = r#"
kernel Probe:
    let [float32]'global src
    mut [float32]'unified dst
    let int a
    let int b

    init([float32]'global s, int aa, int bb):
        src = s
        a = aa
        b = bb
        dst = [0.0 for i in 0..<4]

    def ():
        let cell = .thread.x
        if cell < a * b:
            dst[cell] = src[cell]

pub req [float32]'kernel'unified probe_gpu([float32]'global src, int a, int b) throws:
    mut k = Probe(src, a, b)
    kernel:
        k(block = 256, grid = 1)
    k.dst

def main() throws:
    let src = [1.0, 2.0, 3.0, 4.0]
    let result = probe_gpu(src, 2, 2)
    with result:
        for i in 0..<4:
            print "{result[i]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("kernel_output_field_array_comp_host_wrapper", src);

    assert!(
        rs.contains("k.copy_dst_to_device(&(0..(4) as usize).map(|i| (0) as f32).collect::<Vec<f32>>());"),
        "expected the bound-comprehension fill to resize+upload dst via copy_dst_to_device inside the host wrapper fn;\ngot:\n{rs}"
    );
}

// ─── atomic pointer indexing: `u32(...)`, not `... as u32` ────────────────────

#[test]
fn atomic_pointer_index_uses_wgsl_cast_not_rust_cast() {
    // Real, pre-existing bug, found while verifying the new atomic method
    // calls below against real naga (not just `cargo check`, which only
    // validates the Rust host side, never the WGSL a wgpu-target program
    // actually runs): the atomic-pointer helper shared by `try_atomic_assign`
    // and `try_atomic_method_call` used to emit `&buf[i as u32]` -- `as` is
    // Rust cast syntax, not valid inside a WGSL expression at all. A real
    // `naga::front::wgsl::parse_str` on the generated shader failed with
    // "expected ']', found 'as'" for a plain `counts[bucket] += 1`, meaning
    // every atomic op emitted through this path (`+= -= &= |= ^=`, and now
    // min/max/swap/cas) was unparseable WGSL until this fix -- undetected
    // because nothing in this test suite had run generated WGSL through a
    // real WGSL parser before. WGSL casts are `u32(x)`, matching the
    // (already-correct) plain `ExprKind::Index` case elsewhere in this file.
    let (wgsl, _) = wgpu_codegen("atomic_pointer_cast", r#"
kernel Histogram:
    mut [int]'actor'global counts
    init([int]'actor'global data):
        counts = data
    def ():
        let bucket = .thread.x
        counts[bucket] += 1
"#);
    assert!(wgsl.contains("atomicAdd(&histogram_counts[u32(bucket)], 1);"),
        "expected u32(bucket), not 'bucket as u32' (invalid WGSL);\ngot:\n{wgsl}");
    assert!(!wgsl.contains("as u32]"),
        "must not contain the invalid 'expr as u32]' cast anywhere;\ngot:\n{wgsl}");
}

// ─── atomic min/max/swap/cas ───────────────────────────────────────────────────

#[test]
fn device_atomic_method_calls_map_to_wgsl_intrinsics() {
    // min/max/swap map directly onto WGSL's atomicMin/atomicMax/atomicExchange,
    // which already return the previous value. cas doesn't:
    // atomicCompareExchangeWeak returns a struct ({old_value, exchanged}), not
    // a bare value -- `.old_value` field access on the call result gives
    // exactly the previous value, matching every other backend's contract.
    // Verified to both parse and validate against real naga 22.1.0
    // (naga::front::wgsl::parse_str + naga::valid::Validator).
    let (wgsl, _) = wgpu_codegen("atomic_methods", r#"
kernel Histogram:
    mut [int]'actor'global counts
    init([int]'actor'global data):
        counts = data
    def ():
        let bucket = .thread.x
        counts[bucket].min(5)
        counts[bucket].max(5)
        let old_swap = counts[bucket].swap(0)
        let old_cas = counts[bucket].cas(0, 1)
"#);
    assert!(wgsl.contains("atomicMin(&histogram_counts[u32(bucket)], 5);"), "expected atomicMin;\ngot:\n{wgsl}");
    assert!(wgsl.contains("atomicMax(&histogram_counts[u32(bucket)], 5);"), "expected atomicMax;\ngot:\n{wgsl}");
    assert!(wgsl.contains("atomicExchange(&histogram_counts[u32(bucket)], 0)"), "expected atomicExchange;\ngot:\n{wgsl}");
    assert!(wgsl.contains("atomicCompareExchangeWeak(&histogram_counts[u32(bucket)], 0, 1).old_value"),
        "expected atomicCompareExchangeWeak(...).old_value;\ngot:\n{wgsl}");
}

#[test]
fn host_device_installs_on_uncaptured_error_handler() {
    // Pipeline creation (`emit_kernel_new`, inside a `PIPELINE.get_or_init` closure
    // that can't itself return a Result) isn't wrapped in an explicit error scope,
    // unlike dispatch() and shader-module creation -- an oversized fixed-'actor
    // field would otherwise panic via wgpu's default uncaptured-error
    // handler instead of being reported. Fixed by installing a non-panicking
    // handler once at device-creation time.
    let (_wgsl, rs) = wgpu_codegen("uncaptured_error_handler", r#"
kernel Scale:
    mut [float]'unified buf
    init([float]'unified data):
        buf = data
    def ():
        let tid = .thread.x
        buf[tid] = buf[tid] * 2.0
"#);
    assert!(rs.contains("device.on_uncaptured_error(Box::new(|e| eprintln!(\"boring: uncaptured GPU error: {}\", e)));"),
        "expected a non-panicking on_uncaptured_error handler installed at device-creation time;\ngot:\n{rs}");
}

// ─── Labeled multi-dimensional arrays (docs/array-multidim-types.md) ───────
// Note the `img_img` naming (the field's WGSL storage-buffer global gets a
// `{kernel_name_lowercased}_{field}` prefix) — this backend renames
// storage-buffer variables regardless of which syntax declared them.

#[test]
fn device_labeled_index_lowers_to_row_major_index() {
    let (wgsl, _) = wgpu_codegen("labeled_at", r#"
kernel Img:
    mut [float, width = 4, height = 4]'unified img
    init([float, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = .thread.x
        let r = .thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0
"#);
    assert!(wgsl.contains("img_img[u32(c + r * 4)]"),
        "expected [width=c,height=r] to lower to row-major c + r*width, with the u32 index cast this backend's Index already uses;\ngot:\n{wgsl}");
}

#[test]
fn device_labeled_axis_property_lowers_to_literals() {
    let (wgsl, _) = wgpu_codegen("labeled_width_height", r#"
kernel Img:
    mut [float, width = 4, height = 8]'unified img
    init([float, width = 4, height = 8]'unified data):
        img = data
    def ():
        let c = .thread.x
        let r = .thread.y
        if c < img.width and r < img.height:
            img[width = c, height = r] = 0.0
"#);
    assert!(wgsl.contains("c < 4"), "expected img.width to lower to the literal 4;\ngot:\n{wgsl}");
    assert!(wgsl.contains("r < 8"), "expected img.height to lower to the literal 8;\ngot:\n{wgsl}");
}

#[test]
fn device_labeled_array_field_becomes_storage_buffer() {
    let (wgsl, _) = wgpu_codegen("labeled_storage_buffer", r#"
kernel Img:
    mut [float32, width = 4, height = 4]'unified img
    init([float32, width = 4, height = 4]'unified data):
        img = data
    def ():
        let c = .thread.x
        let r = .thread.y
        img[width = c, height = r] = 0.0
"#);
    assert!(wgsl.contains("var<storage, read_write> img_img: array<f32>;"),
        "expected a LabeledArray field to become a flat storage buffer, same as [T]'unified;\ngot:\n{wgsl}");
}

#[test]
fn host_labeled_array_field_dispatch_infers_2d_grid() {
    let src = r#"
kernel Img:
    mut [float, width = 16, height = 32]'unified img
    init([float, width = 16, height = 32]'unified data):
        img = data
    def ():
        let c = .thread.x
        let r = .thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0

let data = [0.0]
mut k = Img(data)
kernel:
    k(block = (8, 8, 1))
"#;
    let (_wgsl, rs) = wgpu_codegen("labeled_2d_grid", src);
    assert!(rs.contains("k.dispatch((((16 + (8) - 1) / (8))) as u32, (((32 + (8) - 1) / (8))) as u32, (1) as u32)?;"),
        "expected the kernel: block with no explicit grid= to default gx/gy from width/height and the block= size;\ngot:\n{rs}");
}

#[test]
fn host_dynamic_labeled_array_field_dispatch_infers_2d_grid_from_shadow_fields() {
    let src = r#"
kernel Img:
    mut [float, width, height]'unified img
    init([float]'unified data, uint w, uint h):
        img = data.reshape(width = w, height = h)
    def ():
        let c = .thread.x
        let r = .thread.y
        img[width = c, height = r] = img[width = c, height = r] * 2.0

let data = [0.0]
mut k = Img(data, 16, 32)
kernel:
    k(block = (8, 8, 1))
"#;
    let (_wgsl, rs) = wgpu_codegen("dynamic_labeled_2d_grid", src);
    assert!(rs.contains("k.__img_axis0"), "expected grid.x inferred from the __img_axis0 shadow field;\ngot:\n{rs}");
    assert!(rs.contains("k.__img_axis1"), "expected grid.y inferred from the __img_axis1 shadow field;\ngot:\n{rs}");
}

#[test]
fn device_shared_labeled_array_becomes_workgroup_decl() {
    let (wgsl, _) = wgpu_codegen("shared_labeled", r#"
kernel Tile:
    mut [float32]'unified out
    let [float32, width = 4, height = 4]'actor tile
    def ():
        let c = .thread.x
        let r = .thread.y
        out[0] = tile[width = c, height = r]
"#);
    assert!(wgsl.contains("var<workgroup> tile_tile: array<f32, 16>;"),
        "expected a kernel-prefixed module-scope var<workgroup> declaration sized \
         width*height;\ngot:\n{wgsl}");
}

// ─── .min/.max/.swap/.cas without 'actor — plain, non-atomic fallback ─────────

#[test]
fn atomic_method_calls_degrade_to_plain_two_statement_form_without_actor() {
    // WGSL has no statement-expression (unlike CUDA/HIP/Metal's `({ ... })`),
    // so the plain (non-atomic) fallback needs two real statements: bind the
    // let-name to the current value, then perform the update -- rather than
    // erroring or silently doing nothing off a non-actor field, matching
    // `+=`/`-=`/etc.'s existing degrade-to-plain-arithmetic behavior.
    // Verified to both parse and validate against real naga 22.1.0.
    let (wgsl, _) = wgpu_codegen("plain_atomic_methods", r#"
kernel Scale:
    mut [int]'unified buf
    init([int]'unified data):
        buf = data
    def ():
        let tid = .thread.x
        let m = buf[tid].min(5)
        let x = buf[tid].max(5)
        let s = buf[tid].swap(0)
        let c = buf[tid].cas(0, 1)
"#);
    assert!(wgsl.contains("let m = scale_buf[u32(tid)];\n    scale_buf[u32(tid)] = min(m, 5);"),
        "expected plain min as two WGSL statements;\ngot:\n{wgsl}");
    assert!(wgsl.contains("let x = scale_buf[u32(tid)];\n    scale_buf[u32(tid)] = max(x, 5);"),
        "expected plain max as two WGSL statements;\ngot:\n{wgsl}");
    assert!(wgsl.contains("let s = scale_buf[u32(tid)];\n    scale_buf[u32(tid)] = 0;"),
        "expected plain swap as two WGSL statements;\ngot:\n{wgsl}");
    assert!(wgsl.contains("let c = scale_buf[u32(tid)];\n    if (c == 0) {\n        scale_buf[u32(tid)] = 1;\n    }"),
        "expected plain cas as an if-guarded WGSL statement;\ngot:\n{wgsl}");
}

#[test]
fn atomic_method_call_discarded_uses_synthetic_name_not_reserved_underscore() {
    // Two real bugs found while verifying this against real WGSL, neither
    // just "unverified" but genuinely wrong: (1) `_ = buf[i].min(v)` used to
    // try reading `_` back to compute the update -- WGSL's `_` is a
    // write-only phony discard target, confirmed via a real naga parse
    // ("no definition in scope for identifier: '_'") when read; (2) the
    // first synthetic name tried (`__boring_discard_0`) hit WGSL's reserved
    // `__` identifier prefix (naga: "Identifier starts with a reserved
    // prefix"), the same constraint this project already documents
    // elsewhere for `__params`. Fixed by declaring a fresh `bp_discard_N`
    // `let` (matching the existing shuffle-hoist temp-naming convention)
    // instead of assigning to `_` directly. Verified to both parse and
    // validate against real naga 22.1.0.
    let (wgsl, _) = wgpu_codegen("plain_atomic_discard", r#"
kernel Scale:
    mut [int]'unified buf
    init([int]'unified data):
        buf = data
    def ():
        let tid = .thread.x
        _ = buf[tid].min(9)
"#);
    assert!(!wgsl.contains("_ ="), "must never assign to or read back WGSL's phony `_` target;\ngot:\n{wgsl}");
    assert!(!wgsl.contains("__boring"), "must never use a WGSL-reserved `__` identifier prefix;\ngot:\n{wgsl}");
    assert!(wgsl.contains("let bp_discard_0 = scale_buf[u32(tid)];\n    scale_buf[u32(tid)] = min(bp_discard_0, 9);"),
        "expected a synthetic bp_discard_N temp instead of `_`;\ngot:\n{wgsl}");
}

#[test]
fn atomic_method_call_in_nested_position_is_a_build_error_not_silent_wrong_wgsl() {
    // Non-atomic min/max/swap/cas only has a real (correct) codegen path when
    // the whole call is the direct RHS of a `let`/assignment -- WGSL can't
    // express "read old, mutate, yield old" as a single expression. Buried
    // inside a larger expression the device emitter leaves an `/* unsupported here: ... */`
    // marker; the placeholder backstop (`helpers::placeholder_errors`) turns that into a
    // build error instead of shipping a shader that fails later.
    let bin = env!("CARGO_BIN_EXE_boring");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wgpu_codegen").join("plain_atomic_nested");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let br_file = tmp.join("test.br");
    fs::write(&br_file, r#"
kernel Scale:
    mut [int]'unified buf
    init([int]'unified data):
        buf = data
    def ():
        let tid = .thread.x
        let x = buf[tid].min(5) + 1
"#).unwrap();
    let result = Command::new(bin).args(["build", "--target", "wgpu"]).arg(&br_file).output().unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(!result.status.success(), "build unexpectedly succeeded");
    assert!(stderr.contains("/* unsupported here:") && stderr.contains(".min(...) needs 'actor'global/'actor'unified"),
        "expected an error quoting the unsupported-position marker;\ngot:\n{stderr}");
}

// ─── Host-side codegen for `Type::LabeledArray` fields (regressions found ───
// while regenerating examples/{matrix_mul_gpu,vector_add_gpu,plasma_metal}_wgpu
// for 0.9.5 — see CHANGELOG's "Known Issues"/"Fixed" entries). The device
// (WGSL) side already recognized `LabeledArray` fields correctly (the tests
// above); these check the *host* Rust side, which a fixed-shape multi-dim
// array field ('global) fell straight through as if it weren't a buffer
// field at all.

#[test]
fn host_labeled_array_field_gets_real_buffer_not_dropped_or_cast_to_i64() {
    let src = r#"
kernel Img:
    let [float32, width = 4, height = 4]'global a
    mut [float32, width = 4, height = 4]'unified c

    init([float32, width = 4, height = 4]'global input_a):
        a = input_a

    def ():
        let col = .thread.x
        let row = .thread.y
        c[width = col, height = row] = a[width = col, height = row] * 2.0

var [float32] data = [float32(i) for i in 0..<16]
mut k = Img(data.reshape(width = 4, height = 4))
kernel:
    k(block = (4, 4))
"#;
    let (_wgsl, rs) = wgpu_codegen("labeled_field_host_buffer", src);
    assert!(rs.contains("a_buf: std::sync::Arc<wgpu::Buffer>,"),
        "a `[T, width=.., height=..]'global` field must still get a host struct buffer field;\ngot:\n{rs}");
    assert!(rs.contains("k.copy_a_to_device(&data.iter().map(|&x| x as f32).collect::<Vec<f32>>());"),
        "the constructor argument must be uploaded via copy_a_to_device, not cast straight to a scalar;\ngot:\n{rs}");
    assert!(!rs.contains("k.a = "),
        "must not fall back to a bare (wrongly-typed) field assignment for a LabeledArray buffer field;\ngot:\n{rs}");
}

#[test]
fn host_array_comprehension_loop_var_is_isize_not_i64() {
    // `int`/`uint` transpile to `isize`/`usize` as of this release (previously
    // `i64`/`u64`) -- this comprehension's implicit loop var was the one
    // codegen path that never followed, producing a `Vec<i64>` that didn't
    // match an explicitly `[int]`-typed (`Vec<isize>`) binding.
    let src = r#"
kernel Dummy:
    mut [int]'unified out
    init([int]'unified data):
        out = data
    def ():
        let tid = .thread.x
        out[tid] = out[tid] * 2

var [int] host = [i for i in 0..<8]
mut k = Dummy(host)
kernel:
    k(block = 8)
"#;
    let (_wgsl, rs) = wgpu_codegen("array_comp_isize", src);
    assert!(rs.contains("let mut host: Vec<isize>") && rs.contains("let i = __boring_i as isize; i "),
        "expected the comprehension's loop var cast `as isize`, matching the `Vec<isize>` binding it initializes;\ngot:\n{rs}");
    assert!(!rs.contains("as i64; i "),
        "must not cast the comprehension loop var to i64 (stale pre-isize-migration codegen);\ngot:\n{rs}");
}

#[test]
fn host_kernel_output_fill_count_resolves_promoted_top_level_const() {
    // `result = [0 for ..<n]` inside `init()` refers to a top-level `let n =
    // ...`, not an init parameter -- `substitute_and_emit`'s fallback used to
    // reproduce the boring-source name verbatim, but a GPU-target top-level
    // scalar `let` is promoted to an uppercased Rust `const`
    // (`gpu_top_level_const_names`), leaving a dangling lowercase reference.
    let src = r#"
let n = 4

kernel Filler:
    let [int]'global a
    mut [int]'unified out

    init([int]'global input_a):
        a = input_a
        out = [0 for ..<n]

    def ():
        let i = .thread.x
        if i < n:
            out[i] = a[i]

var [int] host_a = [i for i in 0..<n]
mut k = Filler(host_a)
kernel:
    k(block = 4)
"#;
    let (_wgsl, rs) = wgpu_codegen("kernel_output_fill_const_promoted", src);
    assert!(rs.contains("k.copy_out_to_device(&vec![(0) as i32; (N) as usize]);"),
        "expected the fill count to reference the uppercased promoted const N;\ngot:\n{rs}");
    assert!(!rs.contains("(n) as usize"),
        "must not leave a dangling lowercase reference to the pre-promotion name;\ngot:\n{rs}");
}

#[test]
fn host_bare_float_scalar_field_assign_casts_to_f32_matching_narrowed_field() {
    // `float(expr)` is a pure alias of `float64`, not its own type — but on
    // `--target wgpu`, `host_scalar_type` narrows `Type::Float64` to `"f32"` (WGSL has
    // no 64-bit float; the device buffer/Params struct backing a kernel field is
    // always f32 in practice regardless of the Boring-level width, see that
    // function's own doc). `var float t` therefore gets an `f32` host struct field,
    // so `k.t = float(screen.time)` must cast to `f32` too, joining
    // `float32(expr)`'s narrowing rather than producing a real `f64` — this used to
    // be split the other way (this cast stayed `f64`, back when `host_scalar_type`
    // itself still returned `"f64"` for `Type::Float64`); flipping one side without
    // the other reintroduces the exact same class of mismatch this split originally
    // fixed, just in the opposite direction (`error[E0308]: expected f32, found
    // f64` assigning this cast's result into the now-`f32` field — confirmed via a
    // real `cargo build`, not caught by this test's own text-only assertions, which
    // is why the pairing needs stating explicitly here rather than left implied).
    //
    // No `cargo build` check here (unlike this file's other float-width regression
    // test, `test_const_generic_kernel_turbofish_construction`) — text-only
    // assertions are enough for the f32-cast behavior under test. This exact
    // `Screen`-driven `'surface pixels` field combination (a kernel field
    // presented implicitly, with no explicit `screen.present(k.field)` call in
    // the loop) used to also hit a separate, unrelated bug in the blit-bind-group
    // codegen (`error[E0609]: no field 'pixels_buf' on type '&mut __App'`,
    // `self.pixels_buf` instead of `self.k.pixels_buf`) — fixed, and covered by
    // a real `cargo build` in `wgpu_screen_surface_field_without_explicit_present_compiles`
    // below.
    let src = r#"
let width = 4
let height = 4
let screen = Screen(Dimension(width, height), title = "test")

kernel T:
    mut [uint]'surface pixels
    let Dimension dim
    var float t

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        dim = d
        t = 0.0

    def ():
        pass

var mut k = T(Dimension(width, height))
kernel:
    loop:
        k.t = float(screen.time)
        k(block = (4, 4))
        break
"#;
    let (_wgsl, rs) = wgpu_codegen("float_alias_scalar_field", src);
    assert!(rs.contains("k.t = (") && rs.contains("__start_time.elapsed().as_secs_f32() as f64)) as f32;"),
        "expected bare float(...) (an f64) to be narrowed to f32 at the assignment, matching `var float t`'s \
         f32 host field;\ngot:\n{rs}");
    assert!(!rs.contains("__start_time.elapsed().as_secs_f32() as f64);"),
        "the f64 value must be narrowed into the f32 field, not assigned directly;\ngot:\n{rs}");
}

/// Regression test for `find_present_buffer`'s fallback in `src/transpiler/wgpu/host.rs`:
/// a `Screen`-driven program whose kernel has a `mut [uint]'surface` field, presented
/// implicitly (no explicit `screen.present(k.field)` call anywhere in the render loop —
/// only `k.t = ...` and `k(block = ...)`). Before the fix, `find_present_buffer` only
/// recognized the explicit-`present()` shape and otherwise fell back to a bare
/// `"pixels_buf"` default, which the blit-bind-group codegen then emitted as
/// `self.pixels_buf` — a field that doesn't exist on `__App` (the kernel instance is
/// `self.k`, a field *of* `__App`, and the buffer is `self.k.pixels_buf`) — failing a
/// real `cargo build` with `error[E0609]: no field 'pixels_buf' on type '&mut __App'`.
/// The fix makes the fallback scan top-level kernel instantiations for a
/// `'surface`-qualified buffer-array field and qualify it with the instance's own
/// variable name, matching the `self.{var}.{field}_buf` shape `emit_kernel_rebuild_bind_group`
/// already uses elsewhere in the same file.
#[test]
fn wgpu_screen_surface_field_without_explicit_present_compiles() {
    let src = r#"
let width = 4
let height = 4
let screen = Screen(Dimension(width, height), title = "test")

kernel T:
    mut [uint]'surface pixels
    let Dimension dim
    var float t

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        dim = d
        t = 0.0

    def ():
        pass

var mut k = T(Dimension(width, height))
kernel:
    loop:
        k.t = float(screen.time)
        k(block = (4, 4))
        break
"#;
    let (_wgsl, _emulated, rs, _toml) = run_wgpu("screen_surface_field_no_explicit_present", src);

    // Scope the assertion to the blit bind group's own creation block -- `self.pixels_buf`
    // (bare, unqualified) is legitimately correct inside `T`'s *own* `rebuild_bind_group`
    // method (there `self` is `&T`, the kernel instance itself), so a blanket
    // `!rs.contains("self.pixels_buf")` over the whole file would false-positive on that
    // unrelated, correct occurrence. Only `__App`'s blit bind group (built in `resumed()`,
    // where `self` is `&mut __App`) needs the `self.k.` prefix.
    let blit_bg_start = rs.find("let blit_bg = self.device.create_bind_group")
        .unwrap_or_else(|| panic!("expected a `let blit_bg = self.device.create_bind_group(...)` \
                                    block in the generated source:\n{rs}"));
    let blit_bg_block = &rs[blit_bg_start..(blit_bg_start + 400).min(rs.len())];
    assert!(blit_bg_block.contains("resource: self.k.pixels_buf.as_entire_binding()"),
        "expected the blit bind group to address the kernel instance's buffer as \
         `self.k.pixels_buf`, not a bare `self.pixels_buf` (no such field exists on \
         `__App`), blit bind group block:\n{blit_bg_block}");
    assert!(!blit_bg_block.contains("resource: self.pixels_buf.as_entire_binding()"),
        "blit bind group must not address `self.pixels_buf` directly on `__App` -- the \
         buffer lives on the kernel instance field `self.k`, blit bind group block:\n{blit_bg_block}");

    // Real `cargo build` of the generated project -- this exact combination
    // (Screen + `'surface` field, no explicit `screen.present(...)` call) used to fail
    // to compile even though the text-only assertions above would not have caught it
    // without this check (the bug was in a totally different generated fn than the one
    // exercised by the assertions), see this test's own doc comment.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("screen_surface_field_no_explicit_present");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let build = common::SharedCargo::new("wgpu_codegen")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    assert!(
        build.status.success(),
        "expected the generated wgpu project to compile, but `cargo build` failed:\n\
         --- stderr ---\n{}",
        String::from_utf8_lossy(&build.stderr),
    );
}

#[test]
fn wgpu_screen_top_level_statement_runs_in_setup() {
    // Before this fix, a bare top-level statement in a `Screen` program (here: a
    // `print` call) was silently dropped from the generated Rust -- the general
    // pass skips every top-level `Stmt`/non-static `Let` for a `Screen` program
    // (`emit_program_items`'s `host_owns_top_level`), on the assumption that
    // `emit_screen_main` handles it instead, but `emit_screen_main` never even
    // looked at anything besides its own hardcoded `let`/`kernel:` shapes. `boring
    // build` used to exit 0 with the `print` simply absent from `main.rs`.
    // `check_screen_program_shape` now rejects it up front instead.
    let src = r#"
let width = 4
let height = 4
let screen = Screen(Dimension(width, height), title = "test")

print "this line has nowhere to go on the wgpu Screen backend"

kernel T:
    mut [uint]'surface pixels
    let Dimension dim

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        dim = d

    def ():
        pass

var mut k = T(Dimension(width, height))
kernel:
    loop:
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;
    // Top-level statements of a Screen program are ordinary code of `__boring_screen_setup`.
    let (_wgsl, rs) = wgpu_codegen("screen_top_level_print", src);
    let setup = &rs[rs.find("fn __boring_screen_setup").expect("setup fn")..];
    assert!(setup.contains("this line has nowhere to go on the wgpu Screen backend"), "print dropped:\n{rs}");
    assert!(rs.contains("__boring_screen_setup()"), "setup not called:\n{rs}");
}

#[test]
fn wgpu_screen_top_level_let_runs_in_setup() {
    // Same gap, `Let` side: a top-level `let` whose initializer isn't a
    // `Screen(...)`/kernel-constructor call or a scalar literal (here, a computed value) used to vanish from the generated Rust
    // with no diagnostic, confirmed against a real generated project (see
    // `test_screen_program_gpu_adapters_global_now_populated`'s original doc
    // comment). It's rejected up front now instead.
    let src = r#"
let width = 4
let height = 4
let screen = Screen(Dimension(width, height), title = "test")
let g = width + 1

kernel T:
    mut [uint]'surface pixels
    let Dimension dim

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        dim = d

    def ():
        pass

var mut k = T(Dimension(width, height))
kernel:
    loop:
        k(block = (4, 4))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;
    // An ordinary top-level `let` is a statement of `__boring_screen_setup` (never dropped).
    let (_wgsl, rs) = wgpu_codegen("screen_top_level_gpu_let", src);
    let setup = &rs[rs.find("fn __boring_screen_setup").expect("setup fn")..];
    assert!(setup.contains("let g = "), "`let g` dropped:\n{rs}");
}

#[test]
fn wgpu_screen_unsupported_render_loop_statement_is_rejected_not_dropped() {
    // Before this fix, any render-loop-body statement outside the five hardcoded
    // shapes `emit_render_stmt` recognizes (`screen.present`, a `block=` dispatch,
    // a field swap, a field assignment, `if screen.key(...): break`) fell through
    // its catch-all `_ => {}` and simply never appeared in the generated Rust --
    // here, a `print` call inside the loop body.
    let src = r#"
let width = 4
let height = 4
let screen = Screen(Dimension(width, height), title = "test")

kernel T:
    mut [uint]'surface pixels
    let Dimension dim

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        dim = d

    def ():
        pass

var mut k = T(Dimension(width, height))
kernel:
    loop:
        k(block = (4, 4))
        screen.present(k.pixels)
        print "frame"
        if screen.key("\x1B"):
            break
"#;
    let stderr = run_wgpu_expect_failure("screen_unsupported_render_loop_print", src);
    assert!(
        stderr.contains("wgpu target") && stderr.contains("render loop"),
        "expected a clear diagnostic naming the unsupported render-loop statement, got:\n{stderr}"
    );
}

#[test]
fn wgpu_screen_render_loop_unconditional_break_now_supported() {
    // A bare, unconditional `break` (no enclosing `if`) inside the render loop
    // used to be silently dropped too (same catch-all as above) -- the loop never
    // actually stopped. There's no real Rust `loop` to `break` out of here (each
    // frame is a `WindowEvent::RedrawRequested` callback driven by winit), so this
    // now maps to the same `event_loop.exit()` call `if screen.key(...): break`
    // already uses, instead of erroring or vanishing.
    let src = r#"
let width = 4
let height = 4
let screen = Screen(Dimension(width, height), title = "test")

kernel T:
    mut [uint]'surface pixels
    let Dimension dim

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        dim = d

    def ():
        pass

var mut k = T(Dimension(width, height))
kernel:
    loop:
        k(block = (4, 4))
        screen.present(k.pixels)
        break
"#;
    let (_wgsl, rs) = wgpu_codegen("screen_unconditional_break", src);
    assert!(rs.contains("event_loop.exit();"),
        "expected a bare `break` to translate to an unconditional event_loop.exit();\ngot:\n{rs}");
}

/// Regression test for a const-generic kernel's top-level turbofish construction
/// (`Blur<3, 1>(...)`) on `--target wgpu` — a non-`Screen` top-level statement, so it's
/// transpiled by the *general* pipeline (`gpu_kernels` set, see `transpiler::wgpu::
/// transpile_wgpu`'s doc), not wgpu's own `host::emit_screen_main`.
///
/// `resolve_effective_kernels` correctly renames the kernel *struct* itself to
/// `Blur_3_1` (mirrored in `TranspileConfig::gpu_kernel_generic_names`), but the general
/// pipeline's own construction codegen (`emit_kernel::try_emit_kernel_let`) used to only
/// recognize a plain `ExprKind::Call` callee — a turbofish `ExprKind::GenericCall` fell
/// through untouched to the generic monomorphization codegen, which has no notion of
/// wgpu's separate kernel-renaming scheme and emitted real Rust turbofish syntax on the
/// *original* name verbatim (`Blur::<3, 1>(w, pixels, result)` — `error[E0425]: cannot
/// find function Blur in this scope`, since only `Blur_3_1` is ever defined).
///
/// Two more bugs surfaced once that one was fixed, both asserted against below:
///   - a `'const`-qualified fixed-size array field (`weights`) fell through
///     `emit_kernel_construction`'s scalar-field assignment branch (which only special-
///     cases `'unified`/`'global`/`'actor'global` buffer fields), casting the whole host
///     array straight to `i64` (`blur.weights = (w) as i64;`) instead of converting it to
///     the field's actual `[f64; 3]` — now `[f32; 3]`, see next point — array type.
///   - `wgpu::host::host_scalar_type` kept `float`/`float64` at `f64` on the host side
///     (struct field type, `copy_*_to_device`'s parameter type) while the *device* WGSL
///     buffer is always `f32` in practice (WGSL has no 64-bit float — `device.rs`'s
///     `wgsl_unsupported_f64` only emits a comment in the generated shader, not a real
///     `TranspileError`, so nothing ever rejects the mismatch) — a real host/device
///     buffer layout mismatch, surfacing as `cargo build`'s `error[E0308]: expected
///     &[f64], found &Vec<f32>` on every kernel with a bare `float`/`float64` field
///     (Saxpy's `alpha`, not just the const-generic `Blur`).
///
/// Exercises the full `linguist/samples/gpu.br` "Const generic params" kernel end to
/// end, including a real `cargo build` of the generated project — the only GPU backend
/// that can be fully compiled locally without extra toolchains (see that file's own
/// header comment), so this is the one target where a full `cargo build` check is
/// meaningful in CI too.
#[test]
fn test_const_generic_kernel_turbofish_construction() {
    let src = r#"
kernel Blur<int W, int H>:
    let [float, W * H]'const weights
    let [float]'global        input
    mut [float]'global        output
    let float                 sigma = 1.0

    init([float] w, [float] inp, [float] out):
        weights = w
        input   = inp
        output  = out

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        var acc = 0.0
        for k in 0..<W * H:
            let idx = i + k
            if idx < input.len():
                acc = acc + weights[k] * input[idx]
        output[i] = acc

let w = [0.25, 0.5, 0.25]
let pixels = [i as float for i in ..<1024]
mut result = [0.0 for ..<1024]

mut blur = Blur<3, 1>(w, pixels, result)
kernel:
    blur(block = 256)
"#;
    let (_wgsl, _emulated, rs, _toml) = run_wgpu("const_generic_kernel_turbofish", src);

    assert!(rs.contains("struct Blur_3_1 {"), "expected the monomorphised kernel struct \
        Blur_3_1, generated source:\n{rs}");
    assert!(rs.contains("let mut blur = Blur_3_1::new(__boring_gpu_device(), __boring_gpu_queue());"),
        "turbofish construction `Blur<3, 1>(...)` should resolve to the monomorphised \
         `Blur_3_1::new(...)`, not real Rust turbofish syntax on the original name, \
         generated source:\n{rs}");
    assert!(!rs.contains("Blur::<3, 1>("),
        "generated source still contains raw (un-mangled) turbofish construction syntax, \
         which doesn't compile (no `Blur` type is ever defined, only `Blur_3_1`):\n{rs}");
    assert!(rs.contains("blur.weights = w.iter().map(|&x| x as f32).collect::<Vec<f32>>().try_into().unwrap();"),
        "expected the `'const` fixed-size array field `weights` to be converted from the \
         constructor argument, not cast straight to a scalar, generated source:\n{rs}");
    assert!(!rs.contains("blur.weights = (w) as"),
        "`weights` (a `[float, W*H]'const` array field) must not be cast to a scalar \
         type, generated source:\n{rs}");

    // Real `cargo build` of the generated project — catches the host/device float-width
    // buffer mismatch (`error[E0308]: expected &[f64], found &Vec<f32>`) that the string
    // assertions above can't see, since it fires against every plain `float` kernel
    // field (Saxpy-style), not just the const-generic one under test here.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("const_generic_kernel_turbofish");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let build = common::SharedCargo::new("wgpu_codegen")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    assert!(
        build.status.success(),
        "expected the generated wgpu project to compile, but `cargo build` failed:\n\
         --- stderr ---\n{}",
        String::from_utf8_lossy(&build.stderr),
    );
}

/// Regression test for a silent `block = N` dispatch-site argument being dropped for
/// any const-generic (turbofish-constructed) kernel on `--target wgpu`.
///
/// `collect_block_sizes`/`resolve_let_kernel_type` (`src/transpiler/wgpu/device.rs`)
/// resolve a `k(block = N)` dispatch call back to its kernel type by first recording,
/// from the `let`/`mut` binding that constructed `k`, a `binding name -> kernel type
/// name` map. That resolution only pattern-matched a plain `ExprKind::Call` callee --
/// a turbofish const-generic construction (`Blur<3, 1>(w, pixels, result)`) parses as
/// the distinct `ExprKind::GenericCall` variant, so `var_to_type` never got an entry
/// for `blur`, the later `blur(block = 256)` dispatch-site lookup could never resolve
/// `blur` back to a kernel type, and `block_sizes` ended up with no entry at all for
/// this kernel -- `emit_kernel_decl`/`emit_entry_point`'s `unwrap_or((1, 1, 1))`
/// default silently kicked in instead, regardless of the `block = 256` actually
/// requested. Silent: no error, no warning, `cargo build` succeeds, the program runs
/// to completion -- it just only ever computes 1 GPU thread's worth of work.
///
/// Confirmed concretely before the fix: this exact source generated
/// `@compute @workgroup_size(1, 1, 1)` for `Blur_3_1` in shaders/main.wgsl (and
/// `blur.dispatch(1u32, 1u32, 1u32)` in src/main.rs) no matter what `block = ...` was
/// requested at the dispatch site.
#[test]
fn test_const_generic_kernel_turbofish_construction_block_size() {
    let src = r#"
kernel Blur<int W, int H>:
    let [float, W * H]'const weights
    let [float]'global        input
    mut [float]'global        output

    init([float] w, [float] inp, [float] out):
        weights = w
        input   = inp
        output  = out

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        var acc = 0.0
        for k in 0..<W * H:
            let idx = i + k
            if idx < input.len():
                acc = acc + weights[k] * input[idx]
        output[i] = acc

let w = [0.25, 0.5, 0.25]
let pixels = [i as float for i in ..<1024]
mut result = [0.0 for ..<1024]

mut blur = Blur<3, 1>(w, pixels, result)
kernel:
    blur(block = 256)
"#;
    let (wgsl, _rs) = wgpu_codegen("const_generic_kernel_turbofish_block_size", src);

    assert!(
        wgsl.contains("@compute @workgroup_size(256, 1, 1)"),
        "expected the `block = 256` dispatch-site argument on the turbofish-constructed \
         kernel `blur` to resolve back to `Blur_3_1` and set its @workgroup_size, but it \
         didn't -- generated shader:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("@compute @workgroup_size(1, 1, 1)"),
        "found the silent (1, 1, 1) default -- `block = 256` was dropped for the \
         turbofish-constructed kernel, generated shader:\n{wgsl}"
    );
}

/// Regression test for three bugs found running the monomorphised `Blur_3_1` kernel's
/// *body* through real WGSL shader validation (`wgpu`'s `Device::create_shader_module`)
/// -- `test_const_generic_kernel_turbofish_construction` above only got this kernel to
/// `cargo build`, since a Rust-level compile never parses the WGSL string it embeds. All
/// three are `DeviceEmitter::expr`/`emit_stmt` bugs (`src/transpiler/wgpu/device.rs`),
/// not this file's mundane text-fixture drift, so a real `wgpu` adapter (available here
/// since this repo's CI runs on macOS/Metal) was used once, by hand, to confirm each
/// fix actually clears shader validation and progresses further -- not just that the
/// assertions below hold, which they trivially would even if the substitution were
/// subtly wrong (e.g. swapped W/H).
///
/// 1. `for k in 0..<W * H` (and any other body reference to a `kernel Blur<int W, int
///    H>` const-generic param) used to stay a bare, unsubstituted `Var("W")`/`Var("H")`
///    after monomorphisation -- `monomorphise`/`monomorphise_type` only rewrites
///    `type_params` references inside a kernel's `fields`, never inside its
///    `methods`/`inits` bodies (see `resolve_effective_kernels`'s doc comment in
///    `src/transpiler/wgpu/mod.rs`) -- so `device::emit_device_wgsl` emitted `W`/`H`
///    straight through as undefined WGSL identifiers ("no definition in scope for
///    identifier: 'W'"). Fixed by threading the per-instantiation `name -> concrete
///    value` substitution map into `DeviceEmitter` (`current_kernel_consts`) and
///    consulting it in the single `ExprKind::Var` choke point every expression
///    emission passes through, so no occurrence (however nested) is missed.
/// 2. `input.len()` on a `'global` storage-buffer field emitted the free function call
///    `len(blur_3_1_input)` -- WGSL has no such builtin (unlike CUDA/Metal's plain
///    array-length arithmetic); the real builtin is `arrayLength(&buf)`, and only
///    applies to a runtime-sized storage buffer, not a fixed-size `'const` array.
/// 3. `weights[k]` (a `'const`-qualified fixed-size array field) emitted a bare,
///    unprefixed `weights[u32(k)]` -- a `'const` array field lives inside the
///    `Blur_3_1Params` uniform-struct binding (`blur_3_1_params.weights`), not as its
///    own module-scope var; `emit_entry_point`'s own comment already said "Fixed
///    arrays are accessed as `{pvar}.field[i]` directly" but nothing implemented that
///    rewrite, so any index into a `'const` array field fell through as an undefined
///    identifier just like bug 1.
///
/// Note: getting `Blur_3_1` to fully execute (not just pass shader validation) hits a
/// fourth, unrelated bug past the scope of this fix -- WGSL's `uniform` address space
/// requires array elements be aligned to a 16-byte stride ("Alignment requirements for
/// address space Uniform are not met"), which a plain `array<f32, N>` params-struct
/// member violates; fixing that needs a host.rs buffer-layout change (`uniform` →
/// `storage` binding for any kernel with a fixed-array params field, or std140-style
/// padding) well beyond a `DeviceEmitter::expr` fix, so it's left for separate work.
#[test]
fn test_const_generic_kernel_body_substitution_and_array_field_access() {
    let src = r#"
kernel Blur<int W, int H>:
    let [float, W * H]'const weights
    let [float]'global        input
    mut [float]'global        output
    let float                 sigma = 1.0

    init([float] w, [float] inp, [float] out):
        weights = w
        input   = inp
        output  = out

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        var acc = 0.0
        for k in 0..<W * H:
            let idx = i + k
            if idx < input.len():
                acc = acc + weights[k] * input[idx]
        output[i] = acc

let w = [0.25, 0.5, 0.25]
let pixels = [i as float for i in ..<1024]
mut result = [0.0 for ..<1024]

mut blur = Blur<3, 1>(w, pixels, result)
kernel:
    blur(block = 256)
"#;
    let (wgsl, _rs) = wgpu_codegen("const_generic_kernel_body_subst", src);

    // Bug 1: the loop bound is the concrete product, not the bare type-param names.
    assert!(wgsl.contains("if !(k < (3 * 1)) { break; }"),
        "expected the monomorphised loop bound `0..<W * H` to substitute down to the \
         concrete `3 * 1`, generated WGSL:\n{wgsl}");
    assert!(!wgsl.contains("(W * H)"),
        "generated WGSL still references the const-generic params `W`/`H` verbatim -- \
         invalid WGSL (\"no definition in scope for identifier\"), generated WGSL:\n{wgsl}");
    // Cheap, generic backstop matching this test's own doc comment: neither type-param
    // name should ever appear as a bare identifier (word-boundary check) anywhere in the
    // module -- catches a stray occurrence in a nested/derived expression this test's
    // specific source snippet doesn't happen to exercise.
    for name in ["W", "H"] {
        assert!(
            !wgsl.split(|c: char| !c.is_alphanumeric() && c != '_')
                .any(|tok| tok == name),
            "generated WGSL contains a bare identifier `{name}` -- a const-generic kernel \
             type param must always be substituted with its concrete value, generated WGSL:\n{wgsl}"
        );
    }

    // Bug 2: `.len()` on a storage-buffer field → `arrayLength(&buf)`, not `len(buf)`.
    assert!(wgsl.contains("arrayLength(&blur_3_1_input)"),
        "expected `input.len()` to lower to WGSL's `arrayLength(&buf)` builtin, \
         generated WGSL:\n{wgsl}");
    assert!(!wgsl.contains("len(blur_3_1_input)"),
        "generated WGSL still contains the invalid WGSL call `len(...)` -- WGSL has no \
         such free function, generated WGSL:\n{wgsl}");

    // Bug 3: a `'const` fixed-array field is accessed through the params struct.
    assert!(wgsl.contains("blur_3_1_params.weights[u32(k)]"),
        "expected the `'const` array field `weights` to be indexed through the params \
         uniform struct (`blur_3_1_params.weights[...]`), generated WGSL:\n{wgsl}");
    assert!(!wgsl.contains(" weights[u32(k)]") && !wgsl.contains("(weights[u32(k)]"),
        "generated WGSL still contains a bare, unprefixed index into `weights` -- an \
         undefined WGSL identifier, generated WGSL:\n{wgsl}");
}

/// Regression test for the fourth blocker named in
/// `test_const_generic_kernel_body_substitution_and_array_field_access`'s doc comment:
/// a kernel whose params struct has a fixed-size scalar array field (here, `weights`,
/// a `[float32, W * H]'const` array) used to emit that struct as
/// `@group(0) @binding(n) var<uniform> ... { weights: array<f32, 3>, ... }`.
///
/// WGSL's `uniform` address space follows std140-like layout rules, which require
/// every array element's *stride* to be a multiple of 16 bytes -- a bare
/// `array<f32, N>` (4-byte stride) violates that and fails real GPU shader
/// validation at pipeline-creation time:
///   Shader validation error: Global variable [N] '..._params' is invalid
///     Alignment requirements for address space Uniform are not met by [...]
///       The array stride 4 is not a multiple of the required alignment 16
/// even though the generated text is syntactically valid WGSL and the host-side
/// Rust compiles cleanly -- neither this file's usual text-only assertions nor a
/// plain `cargo build` of the generated project (a Rust-level compile never parses
/// the embedded WGSL string) can see this; only creating a real `wgpu::Device` and
/// compute pipeline from the generated shader catches it, hence the full `cargo run`
/// below (this repo's CI runs on macOS/Metal, see the sibling test's doc comment).
///
/// Fixed by switching that binding -- and the matching host-side
/// `wgpu::BufferUsages` for `params_buf` -- to `var<storage, read>` /
/// `wgpu::BufferUsages::STORAGE` whenever the params struct has a fixed-array
/// field (`transpiler::wgpu::kernel_params_use_storage`); `storage` follows std430
/// layout instead, which only requires 4-byte alignment for a scalar array's stride.
///
/// Note: this test's kernel body deliberately drops the `if idx < input.len(): ...`
/// guard the sibling test above uses -- `input.len()` lowers to WGSL's `arrayLength`,
/// which returns `u32`, while `idx` is `i32` (derived from `.thread.x`/`.block.x`),
/// and WGSL's `<` operator rejects mixed-signedness operands ("Operation Less can't
/// work with ..."). That's a real, separate bug (a plain `Type::Named("Dimension")`-
/// style comparison-type-promotion gap, unrelated to buffer address spaces) that this
/// fix doesn't touch -- left for separate follow-up so this test stays focused on the
/// one bug it's named for.
#[test]
fn test_const_generic_kernel_fixed_array_params_real_shader_validation() {
    let src = r#"
kernel Blur<int W, int H>:
    let [float32, W * H]'const weights
    let [float32]'global        input
    mut [float32]'global        output

    init([float32] w, [float32] inp, [float32] out):
        weights = w
        input   = inp
        output  = out

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        var acc = 0.0
        for k in 0..<W * H:
            acc = acc + weights[k] * input[i + k]
        output[i] = acc

let w = [0.25, 0.5, 0.25]
let pixels = [i as float32 for i in ..<8]
mut result = [0.0 for ..<8]

mut blur = Blur<3, 1>(w, pixels, result)
kernel:
    blur(block = 8)

print "done"
"#;
    let (wgsl, _emulated, _rs, _toml) = run_wgpu("const_generic_kernel_fixed_array_params_validation", src);

    // The params struct binding must be `storage`, not `uniform`, precisely because
    // it has a fixed-array field (`weights`).
    assert!(wgsl.contains("var<storage, read> blur_3_1_params:"),
        "expected the params struct (has a fixed-array field `weights`) to bind as \
         `var<storage, read>`, not `var<uniform>` (which fails real WGSL alignment \
         validation for an `array<f32, N>` member) -- generated WGSL:\n{wgsl}");
    assert!(!wgsl.contains("var<uniform> blur_3_1_params:"),
        "params struct must not use `var<uniform>` once it has a fixed-array field \
         -- generated WGSL:\n{wgsl}");

    // Real end-to-end validation: build and run the generated project against a real
    // GPU. Before the fix, this compiled fine (`cargo build` never parses the embedded
    // WGSL) but panicked at runtime with wgpu's uncaptured "Alignment requirements for
    // address space Uniform are not met" validation error the moment the compute
    // pipeline was created, well before any dispatch.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("const_generic_kernel_fixed_array_params_validation");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU, but it failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert!(stdout.contains("done"),
        "expected the program to run to completion and print \"done\", but got:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
    assert!(!stderr.contains("Alignment requirements"),
        "the uniform-buffer alignment bug regressed -- generated program's stderr:\n{stderr}");
}

/// Regression test for the separate bug named in the sibling test's doc comment above
/// (left as follow-up there so that test stayed focused on the uniform-alignment bug):
/// `input.len()` lowers to WGSL's `arrayLength(&buf)` builtin, which returns `u32`, but
/// every kernel-body index/loop expression (`.thread.x`/`.block.x` arithmetic,
/// `for`-loop variables) is emitted as `i32`. WGSL requires identical operand types for
/// comparison operators -- no implicit signed/unsigned promotion like C/Rust -- so
/// `idx < input.len()` lowered to `idx < arrayLength(&input)` (`i32 < u32`), which
/// `cargo build` never catches (it doesn't parse the embedded WGSL string) but fails
/// real shader validation at pipeline-creation time:
///   Shader validation error: Entry point ... is invalid
///     Expression [27] is invalid
///       Operation Less can't work with [24] and [26]
///
/// Fixed by casting the `arrayLength(...)` result to `i32` at its single emission choke
/// point (`DeviceEmitter::expr`'s `method == "len"` buffer branch), rather than at every
/// comparison/arithmetic site that might combine a `.len()` against an index.
///
/// This kernel deliberately avoids a `'const` fixed-size array field (unlike the sibling
/// test above) to sidestep that test's own unrelated uniform-alignment bug -- and uses a
/// `'unified` output field, not `'global`, since a `'global` field's buffer is
/// host-write-only (no `COPY_SRC` usage flag) and can't be read back for the `print`
/// assertions below; that asymmetry is itself an existing, separate gap, not something
/// this test is about.
#[test]
fn test_len_comparison_against_i32_index_real_shader_validation() {
    let src = r#"
kernel Sum2:
    let [float]'global   x
    mut [float]'unified  y

    init([float] xs, [float]'unified ys):
        x = xs
        y = ys

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        var acc = 0.0
        for k in 0..<2:
            let idx = i + k
            if idx < x.len():
                acc = acc + x[idx]
        y[i] = acc

let n = 8
let xs = [i as float for i in ..<n]
mut ys = [0.0 for ..<n]

mut k = Sum2(xs, ys)
kernel:
    k(block = 8)

print "y[0] = {k.y[0]}"
print "y[6] = {k.y[6]}"
print "y[7] = {k.y[7]}"
"#;
    let (wgsl, _emulated, _rs, _toml) = run_wgpu("len_comparison_against_i32_index", src);

    // Text-level sanity check: the `arrayLength` result must be cast to `i32` before
    // it's compared against the `i32` loop-derived index.
    assert!(wgsl.contains("i32(arrayLength(&sum2_x))"),
        "expected `x.len()` to lower to an `i32`-cast `arrayLength` call so it compares \
         cleanly against the `i32` index, generated WGSL:\n{wgsl}");

    // Real end-to-end run against a real GPU adapter (this repo's CI runs on
    // macOS/Metal, see the sibling test's doc comment) -- `cargo build` alone can't
    // catch this bug since it never parses the embedded WGSL string; only real shader
    // validation at pipeline-creation time (triggered by actually running the
    // generated binary) does.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("len_comparison_against_i32_index");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU, but it failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert!(
        !stderr.contains("Shader validation error") && !stderr.contains("is invalid"),
        "generated project produced a real WGSL shader validation error at runtime -- \
         `idx < input.len()` (an `i32 < u32` comparison) is invalid WGSL:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    );
    // Values: x = [0..8), for each thread i, acc = x[i] + x[i+1] when both indices are
    // in bounds, else just x[i] (the k=1 tap is dropped once idx == x.len()).
    assert!(stdout.contains("y[0] = 1"), "expected y[0] = x[0]+x[1] = 0+1 = 1, got:\n{stdout}");
    assert!(stdout.contains("y[6] = 13"), "expected y[6] = x[6]+x[7] = 6+7 = 13, got:\n{stdout}");
    assert!(stdout.contains("y[7] = 7"), "expected y[7] = x[7] (k=1 tap out of bounds) = 7, got:\n{stdout}");
}

// ─── kernel/free-function tail expression → explicit `return` ────────────────
//
// Regression tests for a silent-correctness bug: a device function/method's
// implicit tail expression (the last statement, with no explicit `return`)
// used to be emitted as a bare, discarded statement instead of `return <expr>;`
// — compiles cleanly, runs, and produces silently wrong results (the caller
// always got the pre-call value back). WGSL requires an explicit `return`
// for a non-void function, unlike Rust's own implicit-tail-return convention.

#[test]
fn device_kernel_helper_method_tail_expression_emits_return() {
    let (wgsl, _rs) = wgpu_codegen("kernel_helper_tail_return", r#"
kernel AddOneF32:
    mut [float32]'unified data

    def float32 helper(float32 x):
        x + 1.0

    def ():
        let i = .thread.x
        data[i] = self.helper(data[i])
"#);
    assert!(
        wgsl.contains("return (x + 1.0);") || wgsl.contains("return x + 1.0;"),
        "expected the kernel helper method's tail expression to be emitted as \
         an explicit `return ...;`, not a discarded bare statement;\ngot:\n{wgsl}"
    );
    // A trimmed-line-equality check (not a plain substring check) — the bad
    // bare form `(x + 1.0);` is itself a substring of the good `return (x +
    // 1.0);` line, so a naive `!wgsl.contains(...)` would always incorrectly
    // pass once the fix's own `return ` prefix is present.
    let has_bad_bare_stmt = wgsl.lines().any(|l| {
        let t = l.trim();
        t == "(x + 1.0);" || t == "x + 1.0;"
    });
    assert!(
        !has_bad_bare_stmt,
        "the old discarded bare-statement form must not still be present \
         alongside the `return` (that would mean the tail statement was \
         duplicated, not fixed);\ngot:\n{wgsl}"
    );
}

#[test]
fn device_free_function_tail_expression_emits_return() {
    let (wgsl, _rs) = wgpu_codegen("free_fn_tail_return", r#"
def float32 addOne(float32 x):
    x + 1.0

kernel AddOneF32:
    mut [float32]'unified data

    def ():
        let i = .thread.x
        data[i] = addOne(data[i])
"#);
    assert!(
        wgsl.contains("return (x + 1.0);") || wgsl.contains("return x + 1.0;"),
        "expected the free function's tail expression to be emitted as an \
         explicit `return ...;`, not a discarded bare statement;\ngot:\n{wgsl}"
    );
    let has_bad_bare_stmt = wgsl.lines().any(|l| {
        let t = l.trim();
        t == "(x + 1.0);" || t == "x + 1.0;"
    });
    assert!(
        !has_bad_bare_stmt,
        "the old discarded bare-statement form must not still be present \
         alongside the `return`;\ngot:\n{wgsl}"
    );
}

/// Real end-to-end value assertion, against a real GPU adapter (this repo's CI runs
/// on macOS/Metal — see `test_len_comparison_against_i32_index_real_shader_validation`'s
/// doc comment above): the strongest possible regression test for the tail-expression
/// `return`-emission bug, since it exercises the real generated WGSL through an actual
/// compute pipeline dispatch and checks the *numeric* result, not just "compiles" or
/// "contains return". Before the fix, this ran to completion with no error (the bug is
/// silent-correctness, not a crash) and printed the original, unmodified input
/// (`data[i] = 1, 2, 3`) instead of `data[i] + 1`.
#[test]
fn test_kernel_helper_method_real_shader_value() {
    let src = r#"
kernel AddOneF32:
    mut [float32]'unified data

    init([float32]'unified d):
        data = d

    def float32 helper(float32 x):
        x + 1.0

    def ():
        let i = .thread.x
        data[i] = self.helper(data[i])

mut k = AddOneF32([1.0, 2.0, 3.0])
kernel:
    k(block = 3)

print "data[0] = {k.data[0]}"
print "data[1] = {k.data[1]}"
print "data[2] = {k.data[2]}"
"#;
    let (_wgsl, _emulated, _rs, _toml) = run_wgpu("kernel_helper_real_shader_value", src);

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("kernel_helper_real_shader_value");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU, but it failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert!(stdout.contains("data[0] = 2"), "expected data[0] = 1+1 = 2, got:\n{stdout}");
    assert!(stdout.contains("data[1] = 3"), "expected data[1] = 2+1 = 3, got:\n{stdout}");
    assert!(stdout.contains("data[2] = 4"), "expected data[2] = 3+1 = 4, got:\n{stdout}");
}

/// A free function taking a GPU-array-qualified parameter (`[T]'global`/`'unified`/etc.)
/// used to transpile that parameter to a wholesale wrong scalar type (`w_packed: i32`) —
/// see the CHANGELOG's Metal-backend entry for the original find. The naive follow-up fix
/// (a WGSL `ptr<storage, ...>` parameter, `&` at the call site) turned out to be
/// unsupportable on this backend at all: real naga unconditionally rejects any `Storage`-
/// space pointer function parameter (confirmed against `naga-22.1.0` and `naga-29.0.4`;
/// WGSL's own `unrestricted_pointer_parameters` extension is still `Unimplemented`,
/// gfx-rs/naga#5158). The actual fix drops such a parameter from the signature and every
/// call site entirely, substituting the resolved kernel buffer field's own WGSL global
/// name for every reference to it inside the function body instead (a buffer field is
/// already a module-scope global, visible without being passed as a parameter at all).
#[test]
fn device_free_fn_gpu_array_param_dropped_and_substituted_with_kernel_global() {
    let (wgsl, _rs) = wgpu_codegen("free_fn_gpu_array_param_wgpu", r#"
float32 dequant_at([int32]'global w_packed, int idx):
    let b = w_packed[idx]
    (b as float32) * 2.0

kernel Dequant:
    let [int32]'global w_packed
    mut [float32]'unified out

    init([int32] w, [float32] o):
        w_packed = w
        out = o

    def ():
        let tid = .thread.x
        out[tid] = dequant_at(w_packed, tid)
"#);
    assert!(
        wgsl.contains("fn dequant_at__dequant_w_packed(idx: "),
        "expected the GPU-array-qualified `w_packed` parameter to be dropped entirely \
         from the emitted signature (WGSL forbids a storage-buffer pointer function \
         parameter outright), leaving only the ordinary `idx` parameter;\ngot:\n{wgsl}"
    );
    assert!(
        !wgsl.lines().any(|l| l.contains("fn dequant_at__dequant_w_packed(") && l.contains("w_packed:")),
        "the dropped parameter must not survive under any WGSL type at all in the \
         signature — neither the original bug's wrong bare scalar (`w_packed: i32`), nor \
         a `ptr<storage, ...>` type (rejected by naga as a function parameter outright);\n\
         got:\n{wgsl}"
    );
    assert!(
        wgsl.contains("dequant_w_packed[u32(idx)]"),
        "expected every reference to the dropped parameter inside the function body to be \
         substituted with the resolved kernel buffer field's own WGSL global name (the \
         same `{{kernel}}_{{field}}` global the kernel's own entry point already uses for \
         `w_packed`), not left as an unresolved bare identifier;\ngot:\n{wgsl}"
    );
    assert!(
        wgsl.contains("dequant_at__dequant_w_packed(tid)"),
        "expected the call site to drop the corresponding argument too, matching the \
         callee's now-one-parameter-shorter signature;\ngot:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("dequant_at(w_packed, tid)") && !wgsl.contains("dequant_at(&dequant_w_packed"),
        "the call site must not still pass the dropped argument, bare or by reference — a \
         `ptr<storage, ...>` argument position is itself invalid WGSL on this backend, a \
         different and worse failure mode than the original wrong-scalar-type bug;\n\
         got:\n{wgsl}"
    );
}

/// Real end-to-end value assertion for the fix above, against a real GPU adapter (see
/// `test_kernel_helper_method_real_shader_value`'s doc comment for why this backend's
/// tests can verify against real hardware) — this bug class is invisible to `cargo build`
/// on the generated Rust (the broken WGSL is just an embedded string literal) and only
/// surfaces at real WGSL shader validation/dispatch time, so this is the only test that
/// actually proves the fix rather than merely the generated text's shape.
#[test]
fn real_gpu_dispatch_free_fn_with_global_array_param() {
    let src = r#"
float32 dequant_at([int32]'global w_packed, int idx):
    let b = w_packed[idx]
    (b as float32) * 2.0

kernel Dequant:
    let [int32]'global w_packed
    mut [float32]'unified out

    init([int32] w, [float32] o):
        w_packed = w
        out = o

    def ():
        let tid = .thread.x
        out[tid] = dequant_at(w_packed, tid)

mut k = Dequant([1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0])
kernel:
    k(block = 4)

print "out[0] = {k.out[0]}"
print "out[1] = {k.out[1]}"
print "out[2] = {k.out[2]}"
print "out[3] = {k.out[3]}"
"#;
    let (_wgsl, _emulated, _rs, _toml) = run_wgpu("free_fn_gpu_array_param_real_dispatch", src);

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("free_fn_gpu_array_param_real_dispatch");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU, but it failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert!(stdout.contains("out[0] = 2"), "expected out[0] = 1*2 = 2, got:\n{stdout}");
    assert!(stdout.contains("out[1] = 4"), "expected out[1] = 2*2 = 4, got:\n{stdout}");
    assert!(stdout.contains("out[2] = 6"), "expected out[2] = 3*2 = 6, got:\n{stdout}");
    assert!(stdout.contains("out[3] = 8"), "expected out[3] = 4*2 = 8, got:\n{stdout}");
}

/// `DeviceEmitter::errors` (device.rs) accumulates real diagnostics -- a dynamic `'[T]'sync`
/// field with no compile-time size, an out-of-u32-range integer literal, and (see the next
/// test) a GPU-array-qualified parameter that can't be bound to one buffer -- but
/// `emit_device_wgsl` used to return only `(String, Option<String>)`, silently dropping
/// `.errors` at that boundary. `transpile_wgpu`'s own `errors` vec never included anything
/// from the device emitter at all, so `boring build --target wgpu` exited 0 and wrote out a
/// generated shader referencing an undeclared `scratch` variable (the dynamic `'sync` field
/// was dropped from the WGSL entirely, since there's no way to emit a compile-time-sized
/// `var<workgroup>` for it) -- a confusing naga validation failure at real dispatch time
/// instead of this already-computed, much clearer diagnostic. `emit_device_wgsl` now returns
/// its accumulated errors too, wired into the same top-level `errors` vec host/general
/// errors already flow through.
#[test]
fn wgpu_dynamic_sync_field_is_rejected_not_silently_dropped() {
    let src = r#"
kernel S:
    mut [float32]'unified out
    let [float32]'actor  scratch
    def ():
        let tid = .thread.x
        out[tid] = scratch[0]
"#;
    let stderr = run_wgpu_expect_failure("dynamic_sync_field_rejected", src);
    assert!(
        stderr.contains("dynamic '[T]'sync field 'scratch'") && stderr.contains("not supported on --target wgpu"),
        "expected the pre-existing dynamic-'sync diagnostic to actually surface and fail \
         the build, got:\n{stderr}"
    );
}

/// `wgsl_scalar`'s `wgsl_unsupported_width` fallback (device.rs) narrows a genuinely
/// unrepresentable-width element type down to a 4-byte `i32`/`u32` with only an inline WGSL
/// comment -- fine for a *scalar* kernel param (just a narrowed value, still 4 bytes on both
/// host and device), but silently wrong for a storage-*buffer* field: the host side
/// (`host_scalar_type` in host.rs) keeps that field's *real*, narrower-or-wider byte width
/// (`u8`, `i16`, `u64`, ...) for its `Vec`/upload, while the device side's `array<u32>` still
/// indexes by 4-byte word -- every element past the first is read from the wrong byte
/// offset. `int64`/`uint64`/`int128`/`uint128` have no packing story on this target (unlike
/// `uint8`/`int8`/`uint16`/`int16` -- see the `wgpu_packed_byte_buffer_field_*` tests below,
/// which cover the now-supported case this test used to also reject) and stay hard-rejected
/// by `emit_kernel_decl`'s validation loop (device.rs) instead of silently corrupting data.
#[test]
fn wgpu_narrow_width_buffer_field_is_rejected_not_silently_corrupted() {
    for (decl, name) in [
        ("mut [int64]'unified w_packed", "int64"),
        ("mut [uint64]'unified w_packed", "uint64"),
    ] {
        let src = format!(
            r#"
kernel Q:
    mut [float32]'unified out
    {decl}
    def ():
        let tid = .thread.x
        out[tid] = w_packed[tid] as float32
"#
        );
        let stderr = run_wgpu_expect_failure(&format!("narrow_buffer_field_rejected_{name}"), &src);
        assert!(
            stderr.contains("buffer field 'w_packed'") && stderr.contains("not supported as a storage-buffer element"),
            "[{name}] expected the narrow-width buffer field to be rejected with a clear \
             diagnostic instead of silently corrupting data, got:\n{stderr}"
        );
    }
}

/// `'actor` packed `uint8`/`int8`/`uint16`/`int16` buffers: WGSL has no sub-word atomics, so
/// `+=`/`min`/`max`/`swap`/`cas` go through a compare-exchange loop on the containing word
/// (`bp_packed_rmw`). Pairs of threads share one element, neighbouring elements share words.
#[test]
fn wgpu_atomic_packed_byte_buffer_field_uses_cas_loop() {
    let test_name = "atomic_packed_byte_buffer_field";
    let src = r#"
kernel Q:
    mut [uint8]'actor'unified counters
    mut [int8]'actor'unified lows
    mut [uint16]'actor'unified swaps
    init(mut [uint8]'unified c, mut [int8]'unified l, mut [uint16]'unified s):
        counters = c
        lows = l
        swaps = s
    def ():
        let i = .thread.x
        counters[i / 2] += 1
        lows[i / 2].min(0 - i)
        let old = swaps[i / 2].swap(i + 100)
mut k = Q([uint8(0) for i in 0..<10], [int8(0) for i in 0..<10], [uint16(0) for i in 0..<10])
kernel:
    k(block = 10)
print "c {k.counters[0]} {k.counters[1]} {k.counters[4]}"
print "l {k.lows[0]} {k.lows[1]} {k.lows[4]}"
"#;
    let (wgsl, _emulated, _rs, _toml) = run_wgpu(test_name, src);
    assert!(wgsl.contains("fn bp_rmw_q_counters("), "expected the CAS helper:\n{wgsl}");
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wgpu_codegen").join(test_name);
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "run failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
    assert!(stdout.contains("c 2 2 2"), "lost atomic add on packed uint8:\n{stdout}");
    assert!(stdout.contains("l -1 -3 -9"), "wrong signed min on packed int8:\n{stdout}");
}

/// The wgpu-backend regression test for the packed-byte-kind buffer field feature: a
/// `uint8`/`int8`-element `'global` field is transparently packed into `array<u32>` storage
/// words (see `PackedByteKind`'s doc comment in device.rs) instead of being rejected. Modeled
/// on the real motivating case (`boring-llm`'s `Q8LinearKernel`, which reads GGUF's Q8_0
/// quantization format -- pairs of a little-endian 16-bit scale followed by packed int8
/// values, all inside one raw byte buffer): this kernel reads a `[uint8]'global` buffer
/// holding two 5-byte records (a little-endian 16-bit header followed by three 1-byte
/// fields), plus a separate `[int8]'global` buffer to check sign-extension, and copies every
/// individual element straight through to a `'unified` output -- verifying every logical
/// index is read from the correct bit offset, not just the first one per 4-byte word
/// (`bytes`/`sbytes` are 10/6 elements long, deliberately NOT a multiple of the 4-per-word
/// packing, exercising the buffer-size-rounding and host-upload-padding path too, see
/// `round_up_to_word_bytes` in host.rs).
#[test]
fn wgpu_packed_byte_buffer_field_real_gpu_dispatch() {
    let test_name = "packed_byte_buffer_field_real_dispatch";
    let src = r#"
kernel PackedRecords:
    let [uint8]'global bytes
    let [int8]'global sbytes
    mut [float32]'unified out_bytes
    mut [float32]'unified out_sbytes
    mut [float32]'unified out_header

    init([uint8] b, [int8] sb, [float32] ob, [float32] osb, [float32] oh):
        bytes = b
        sbytes = sb
        out_bytes = ob
        out_sbytes = osb
        out_header = oh

    def ():
        let tid = .thread.x
        out_bytes[tid] = bytes[tid] as float32
        out_sbytes[tid] = sbytes[tid] as float32
        if tid < 2:
            let lo = bytes[tid * 5]
            let hi = bytes[tid * 5 + 1]
            out_header[tid] = (lo as float32) + (hi as float32) * 256.0

let bytes = [10, 200, 3, 250, 5, 20, 100, 3, 250, 5]
let sbytes = [-1, -128, 127, 0, -50, 100]
mut out_bytes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mut out_sbytes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mut out_header = [0.0, 0.0]

mut k = PackedRecords(bytes, sbytes, out_bytes, out_sbytes, out_header)
kernel:
    k(block = 10)

for i in 0..<10:
    print "out_bytes[{i}] = {k.out_bytes[i]}"
for i in 0..<6:
    print "out_sbytes[{i}] = {k.out_sbytes[i]}"
for i in 0..<2:
    print "out_header[{i}] = {k.out_header[i]}"
"#;
    let (wgsl, _emulated, _rs, _toml) = run_wgpu(test_name, src);

    // Codegen shape: both packed fields back onto `array<u32>`, not a per-width WGSL type
    // that doesn't exist (`array<i8>` isn't valid WGSL and would fail naga parsing).
    assert!(
        wgsl.contains("array<u32>"),
        "expected the packed-byte buffer fields' storage to be declared as `array<u32>`, got:\n{wgsl}"
    );
    assert!(
        wgsl.contains("extractBits"),
        "expected packed-byte element reads to use WGSL's `extractBits` builtin, got:\n{wgsl}"
    );

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join(test_name);
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a real \
         GPU (dispatching a kernel reading packed `uint8`/`int8` buffer fields), but it \
         failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );

    // Every uint8 element, at every word offset (0..=3), across all 3 backing u32 words
    // (10 elements: word0 = indices 0-3, word1 = 4-7, word2 = 8-9 partial).
    let expected_bytes: [i64; 10] = [10, 200, 3, 250, 5, 20, 100, 3, 250, 5];
    for (i, v) in expected_bytes.iter().enumerate() {
        let line = format!("out_bytes[{i}] = {v}");
        assert!(stdout.contains(&line), "expected `{line}` in:\n{stdout}");
    }
    // Every int8 element, including negative values and the min/max of the type, across
    // both backing u32 words (6 elements: word0 = indices 0-3, word1 = 4-5 partial) --
    // confirms sign-extension via `extractBits(bitcast<i32>(word), ...)` is correct at every
    // offset, not just offset 0.
    let expected_sbytes: [i64; 6] = [-1, -128, 127, 0, -50, 100];
    for (i, v) in expected_sbytes.iter().enumerate() {
        let line = format!("out_sbytes[{i}] = {v}");
        assert!(stdout.contains(&line), "expected `{line}` in:\n{stdout}");
    }
    // The two little-endian 16-bit headers, each decoded from a separate pair of uint8
    // elements straddling a word boundary at record 0 (bytes[0..2], inside word0) and mid-word
    // at record 1 (bytes[5..7], spanning word1) -- matching GGUF's per-block scale field shape.
    assert!(stdout.contains("out_header[0] = 51210"), "expected out_header[0] = 10 + 200*256 = 51210, got:\n{stdout}");
    assert!(stdout.contains("out_header[1] = 25620"), "expected out_header[1] = 20 + 100*256 = 25620, got:\n{stdout}");
}

/// Threads writing neighbouring elements of one packed `uint8`/`int16` word used to lose
/// updates: the store was a plain read-modify-write of the whole `array<u32>` word. A
/// writable packed buffer is now `array<atomic<u32>>` with an `atomicAnd`/`atomicOr` store
/// and `atomicLoad` reads; a `let` (read-only) one stays a plain `array<u32>`.
#[test]
fn wgpu_packed_byte_writes_to_one_word_are_atomic() {
    let test_name = "packed_byte_atomic_writes";
    let src = r#"
kernel Bytes:
    mut [uint8]'unified a
    mut [int16]'unified s
    let [uint8]'global ro
    init(mut [uint8]'unified a0, mut [int16]'unified s0, [uint8] r):
        a = a0
        s = s0
        ro = r
    def ():
        let i = .thread.x
        a[i] = a[i] + ro[i]
        s[i] = s[i] - 1
mut k = Bytes([uint8(i) for i in 0..<10], [int16(i) for i in 0..<10], [uint8(1) for i in 0..<10])
kernel:
    k(block = 10)
print "a {k.a[0]} {k.a[3]} {k.a[9]}"
print "s {k.s[0]} {k.s[3]} {k.s[9]}"
"#;
    let (wgsl, _emulated, _rs, _toml) = run_wgpu(test_name, src);
    assert!(wgsl.contains("array<atomic<u32>>"), "writable packed buffers must be atomic:\n{wgsl}");
    assert!(wgsl.contains("atomicAnd(") && wgsl.contains("atomicOr(") && wgsl.contains("atomicLoad("),
        "packed stores must be atomicAnd/atomicOr pairs:\n{wgsl}");
    assert!(wgsl.contains(": array<u32>;"), "the `let` packed buffer stays a plain array<u32>:\n{wgsl}");

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join(test_name);
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "run failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
    assert!(stdout.contains("a 1 4 10"), "lost update on a packed uint8 word:\n{stdout}");
    assert!(stdout.contains("s -1 2 8"), "lost update on a packed int16 word:\n{stdout}");
}

/// `request_device` used to hardcode `required_limits: wgpu::Limits::default()`
/// (via `..Default::default()`, no `required_limits` field at all) -- wgpu's
/// conservative, portable-across-everything default, capping every buffer at
/// `max_storage_buffer_binding_size` = 128 MiB regardless of what the real
/// adapter actually supports. A single kernel field's buffer larger than that
/// (e.g. a large ML weight matrix) failed at `create_bind_group` time with a
/// validation error, on hardware that could easily have bound it. Fixed by
/// requesting `adapter.limits()` instead, at both `request_device` call sites
/// (the plain headless `async_main` path and the windowed/Screen path).
#[test]
fn wgpu_large_buffer_binding_uses_adapter_limits_not_default_cap() {
    let test_name = "large_buffer_binding_adapter_limits";
    let src = r#"
kernel LargeBuffer:
    mut [float32]'unified out

    init([float32] o):
        out = o

    def ():
        let tid = .thread.x
        if tid < 4:
            out[tid] = (tid as float32) + 1.0

mut out = [0.0 for ..=35999999]

mut k = LargeBuffer(out)
kernel:
    k(block = 4)

print "out[0] = {k.out[0]}"
print "out[1] = {k.out[1]}"
print "out[2] = {k.out[2]}"
print "out[3] = {k.out[3]}"
print "len = {k.out.len()}"
"#;
    let (_wgsl, _emulated, rs, _toml) = run_wgpu(test_name, src);

    // Codegen shape: both `request_device` call sites now request the adapter's own
    // reported limits instead of the conservative portable default.
    assert!(
        rs.contains("required_limits: adapter.limits()"),
        "expected `request_device` to request `adapter.limits()` instead of the \
         conservative `wgpu::Limits::default()` (128 MiB storage-buffer binding cap), got:\n{rs}"
    );
    assert!(
        !rs.contains("DeviceDescriptor::default()"),
        "expected no bare `DeviceDescriptor::default()` request left (it carries the \
         128 MiB cap with no way to raise it), got:\n{rs}"
    );

    // Real GPU dispatch: a single kernel field backed by a ~137 MiB buffer (36,000,000
    // float32 elements) -- comfortably larger than the old artificial 128 MiB
    // (134,217,728 byte) `max_storage_buffer_binding_size` default, but well within any
    // real desktop/discrete GPU's actual capacity. Before the fix, this failed at
    // `Device::create_bind_group` with a validation error regardless of the real
    // adapter's capability; after the fix it binds and dispatches successfully.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join(test_name);
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU (binding a >128 MiB single-buffer kernel field), but it failed -- this \
         is the exact `max_storage_buffer_binding_size` validation error the adapter-limits \
         fix is meant to prevent:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert!(!stderr.contains("exceeds"), "expected no buffer-size validation error, got:\n{stderr}");
    assert!(stdout.contains("out[0] = 1"), "expected out[0] = 1, got:\n{stdout}");
    assert!(stdout.contains("out[1] = 2"), "expected out[1] = 2, got:\n{stdout}");
    assert!(stdout.contains("out[2] = 3"), "expected out[2] = 3, got:\n{stdout}");
    assert!(stdout.contains("out[3] = 4"), "expected out[3] = 4, got:\n{stdout}");
    assert!(stdout.contains("len = 36000000"), "expected len = 36000000, got:\n{stdout}");
}

/// Same silently-dropped-errors bug, different diagnostic source: `build_gpu_array_subst`
/// resolves a GPU-array-qualified free-function parameter (`w_packed` here) to the single
/// kernel buffer field it's always called with across the whole program (see
/// `device_free_fn_gpu_array_param_dropped_and_substituted_with_kernel_global` above for the
/// ordinary, resolvable case) -- when two call sites disagree (two kernels each binding
/// `dequant_at`'s `w_packed` to a *different* field), the parameter is `Poisoned` and
/// unresolvable, since WGSL has no way to represent an unbound storage-buffer parameter at
/// all. That path used to only embed an inline `/* ERROR: ... */` WGSL comment (naga
/// compiles straight through it) without registering a real error anywhere, so this was
/// doubly silent -- dropped by the inline-comment-only path, and would have been dropped a
/// second time by `emit_device_wgsl`'s old signature even if it had been registered.
/// `build_gpu_array_subst` now also pushes a real error, on top of the fix above.
#[test]
fn wgpu_gpu_array_param_binding_conflict_is_rejected_not_silently_dropped() {
    let src = r#"
float32 dequant_at([int32]'global w_packed, int idx):
    let b = w_packed[idx]
    (b as float32) * 2.0

kernel Dequant:
    let [int32]'global w_packed
    mut [float32]'unified out

    init([int32] w, [float32] o):
        w_packed = w
        out = o

    def ():
        let tid = .thread.x
        out[tid] = dequant_at(w_packed, tid)

kernel Dequant2:
    let [int32]'global other_packed
    mut [float32]'unified out

    init([int32] w, [float32] o):
        other_packed = w
        out = o

    def ():
        let tid = .thread.x
        out[tid] = dequant_at(other_packed, tid)
"#;
    // Two kernels binding `dequant_at`'s array parameter to different buffers used to be a
    // hard error; the function is now cloned once per bound buffer instead.
    let (wgsl, _rs) = wgpu_codegen("gpu_array_param_binding_conflict", src);
    assert!(wgsl.contains("fn dequant_at__dequant_w_packed(") && wgsl.contains("fn dequant_at__dequant2_other_packed("), "{wgsl}");
    assert!(wgsl.contains("dequant_w_packed[u32(idx)]") && wgsl.contains("dequant2_other_packed[u32(idx)]"), "{wgsl}");
    assert!(!wgsl.contains("/* ERROR"), "{wgsl}");
}

/// A call whose array argument is not a kernel buffer field (a computed expression) still
/// cannot be bound to a single buffer, and is reported rather than silently miscompiled.
#[test]
fn wgpu_gpu_array_param_unresolvable_argument_is_rejected_not_silently_dropped() {
    let src = r#"
float32 dequant_at([int32]'global w_packed, int idx):
    let b = w_packed[idx]
    (b as float32) * 2.0

kernel Dequant:
    let [int32]'global w_packed
    mut [float32]'unified out

    init([int32] w, [float32] o):
        w_packed = w
        out = o

    def ():
        let tid = .thread.x
        out[tid] = dequant_at(w_packed, tid)

float32 other([int32]'global q):
    dequant_at(q, 0)

kernel Dequant2:
    let [int32]'global p
    mut [float32]'unified out

    init([int32] w, [float32] o):
        p = w
        out = o

    def ():
        out[.thread.x] = dequant_at([1, 2, 3], 0)
"#;
    let stderr = run_wgpu_expect_failure("gpu_array_param_unresolvable", src);
    assert!(stderr.contains("dequant_at") && stderr.contains("not supported on --target wgpu"), "{stderr}");
}

// ─── Identifiers colliding with a real WGSL reserved keyword ─────────────────
//
// Same bug class as the Metal backend's `msl_safe_ident` fix (`device_field_named_half_is_mangled...`
// and its two sibling tests in `tests/metal_codegen.rs`), narrowed to what actually reaches this
// backend: WGSL has no builtin *scalar* type names sharing a namespace with ordinary identifiers
// the way MSL's `half`/`float`/`int` do (Boring has no primitive spelled like a WGSL keyword), but
// it does have real bare-word reserved keywords (naga's own `RESERVED` table — see
// `wgsl_safe_ident`/`WGSL_RESERVED` in `src/transpiler/wgpu/device.rs`) that ARE valid, reachable
// Boring identifiers. A kernel field, a `def()`-body local, a function/method parameter, or a
// `for`-loop variable named e.g. `const` used to be emitted verbatim, producing WGSL naga rejects
// at pipeline-creation time with "name `const` is a reserved keyword" -- invisible to `boring build
// --target wgpu` itself (this backend never parses the WGSL it emits) and to `cargo build`/`cargo
// run` on the generated Rust (the WGSL lives in a plain `include_str!`'d string).
//
// `const` is used throughout (mirroring the Metal tests' reuse of `half` for all three cases) --
// confirmed via a real `naga::front::wgsl::parse_str` that it's genuinely rejected as a plain
// identifier, unlike some of the words a first-pass investigation into this bug guessed at
// (`array`, `atomic`, `ptr`, `bitcast` are NOT actually rejected by naga as plain identifiers --
// they're `BUILTIN_IDENTIFIERS`, not `RESERVED`, and remain ordinary shadowable identifiers outside
// an actual type/builtin-call position).

#[test]
fn device_field_named_const_is_mangled_not_left_colliding_with_wgsl_reserved_keyword() {
    let (wgsl, _rs) = wgpu_codegen("field_named_const", r#"
kernel ConstKernel:
    mut [float32]'unified data
    let float32 const

    def ():
        let i = .thread.x
        data[i] = data[i] * const
"#);
    // Declaration and every use must agree on the same mangled name -- a bare,
    // unmangled `const` declaration is exactly the collision this test guards
    // against (naga rejects `const` as a plain identifier outright).
    assert!(!wgsl.contains("const: f32,"),
        "a params-struct field named `const` must not be emitted verbatim (a real \
         WGSL reserved keyword);\ngot:\n{wgsl}");
    assert!(wgsl.contains("const_: f32,"),
        "expected the params-struct field to be mangled to `const_`;\ngot:\n{wgsl}");
    assert!(wgsl.contains("let const_: f32 = constkernel_params.const_;"),
        "expected the unpacked local to be declared under the same mangled name \
         as the params-struct field;\ngot:\n{wgsl}");
    assert!(wgsl.contains("* const_)"),
        "expected the kernel body's read of the field to use the same mangled \
         name `const_` as its declaration;\ngot:\n{wgsl}");
}

#[test]
fn device_local_let_named_const_is_mangled() {
    let (wgsl, _rs) = wgpu_codegen("local_let_named_const", r#"
kernel LocalConst:
    mut [float32]'unified out
    let int n

    def ():
        let const = n / 2
        let tid = .thread.x
        if tid < const:
            out[tid] = 1.0
"#);
    assert!(!wgsl.contains("let const ="),
        "a local `let const = ...` must not be emitted as a bare `const` \
         identifier (a real WGSL reserved keyword);\ngot:\n{wgsl}");
    assert!(wgsl.contains("let const_ ="),
        "expected the local to be mangled to `const_`;\ngot:\n{wgsl}");
    assert!(wgsl.contains("tid < const_"),
        "expected the later read of the local to use the same mangled name \
         as its declaration;\ngot:\n{wgsl}");
}

#[test]
fn device_for_loop_var_named_const_is_mangled() {
    let (wgsl, _rs) = wgpu_codegen("for_loop_var_named_const", r#"
kernel LoopConst:
    mut [float32]'unified out

    def ():
        for const in 0..<4:
            out[const] = 1.0
"#);
    assert!(!wgsl.contains("var const: i32"),
        "a for-loop variable named `const` must not be emitted verbatim (a real \
         WGSL reserved keyword);\ngot:\n{wgsl}");
    assert!(wgsl.contains("var const_: i32"),
        "expected the loop variable to be mangled to `const_`;\ngot:\n{wgsl}");
    assert!(wgsl.contains("u32(const_)"),
        "expected the loop body's reference to the loop variable to use the \
         same mangled name as its declaration;\ngot:\n{wgsl}");
}

#[test]
fn device_fn_param_named_const_is_mangled() {
    let (wgsl, _rs) = wgpu_codegen("fn_param_named_const", r#"
kernel ParamConst:
    mut [float32]'unified out

    def float32 helper(float32 const):
        const + 1.0

    def ():
        let i = .thread.x
        out[i] = self.helper(out[i])
"#);
    assert!(!wgsl.contains("fn ParamConst_helper(const: f32)"),
        "a method parameter named `const` must not be emitted verbatim (a real \
         WGSL reserved keyword);\ngot:\n{wgsl}");
    assert!(wgsl.contains("fn ParamConst_helper(const_: f32)"),
        "expected the parameter to be mangled to `const_`;\ngot:\n{wgsl}");
    assert!(wgsl.contains("const_ + 1.0"),
        "expected the method body's read of the parameter to use the same \
         mangled name as its declaration;\ngot:\n{wgsl}");
}

// ─── Workgroup-array field self-collision with its own type's WGSL keyword ──
//
// Distinct bug class from the reserved-keyword tests just above (confirmed via a
// real `naga::front::wgsl::parse_str`, naga 30.0.1): `array`/`atomic`/`vec2`/etc.
// are NOT rejected as plain WGSL identifiers (they're `BUILTIN_IDENTIFIERS`, not
// `RESERVED`) -- but naga DOES reject a *declaration* outright when its name
// textually equals the type-constructor keyword spelling its own type, e.g.
// `var<workgroup> array: array<f32, 4>;` fails with "declaration of `array` is
// recursive", even though `var<workgroup> array: i32;` (same name, different
// type) or a same-named *struct field* of any type compiles fine.
//
// `wgsl_workgroup_array_ident` now also kernel-prefixes every workgroup-array
// declaration (`{kernel}_{field}`, see the cross-kernel name-collision fix this
// same function documents) -- which means a field literally named `array` is
// no longer emitted bare at all; it's already disambiguated to `{kernel}_array`
// before the self-recursion check ever gets a chance to matter, since a real
// kernel name is never empty. The trailing-`_` mangling these tests originally
// exercised is kept in the source as a defensive fallback (see the function's
// own doc comment) but is not reachable from valid Boring source any more --
// these tests now assert the (still correct, still non-recursive) prefixed
// name instead.

#[test]
fn device_workgroup_array_field_named_array_is_mangled_not_left_self_recursive() {
    let (wgsl, _rs) = wgpu_codegen("workgroup_array_field_named_array", r#"
kernel Tile:
    let [float32, 256]'actor array
    mut [float32]'unified data

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        array[.thread.x] = data[i]
        sync
        data[i] = array[.thread.x]
"#);
    assert!(!wgsl.contains("var<workgroup> array: array<f32, 256>;"),
        "a workgroup array field literally named `array` must not be emitted \
         verbatim -- naga rejects `var<workgroup> array: array<...>;` outright \
         with \"declaration of `array` is recursive\";\ngot:\n{wgsl}");
    assert!(wgsl.contains("var<workgroup> tile_array: array<f32, 256>;"),
        "expected the field to be kernel-prefixed to `tile_array` (which also \
         happens to sidestep the self-recursive-`array` case);\ngot:\n{wgsl}");
    assert!(wgsl.contains("tile_array[u32(i32(bp_tid.x))] = "),
        "expected the kernel body's write to use the same kernel-prefixed name \
         `tile_array` as its declaration;\ngot:\n{wgsl}");
    assert!(wgsl.contains("= tile_array[u32(i32(bp_tid.x))];"),
        "expected the kernel body's read to use the same kernel-prefixed name \
         `tile_array` as its declaration;\ngot:\n{wgsl}");
}

#[test]
fn device_workgroup_labeled_array_field_named_array_is_mangled() {
    let (wgsl, _rs) = wgpu_codegen("workgroup_labeled_array_field_named_array", r#"
kernel LabeledTile:
    let [float32, width = 4, height = 4]'actor array
    mut [float32]'unified data

    def ():
        let i = .thread.x
        data[i] = array[width = 0, height = 0]
"#);
    assert!(!wgsl.contains("var<workgroup> array: array<f32, 16>;"),
        "a labeled workgroup array field literally named `array` must not be \
         emitted verbatim (same naga \"declaration is recursive\" rejection as \
         the fixed-size-array case);\ngot:\n{wgsl}");
    assert!(wgsl.contains("var<workgroup> labeledtile_array: array<f32, 16>;"),
        "expected the labeled array field to be kernel-prefixed to \
         `labeledtile_array`;\ngot:\n{wgsl}");
}

// ─── host — string indexing/slicing in a kernel-touching function ─────────────

// Unlike the Metal/CUDA/ROCm backends (which each have their own hand-written
// custom host emitter for kernel-touching functions -- see e.g.
// `metal_codegen.rs`'s regression test for this exact bug there), the wgpu
// backend routes ALL non-Screen code, including kernel-touching functions,
// through the SAME general pipeline the plain/std target uses (see
// `wgpu::mod`'s doc comment) -- so it was never affected by the Metal/CUDA/
// ROCm string-indexing bug in the first place. This test just confirms that
// stays true: `s[i]`/`s[a..<b]` on a `string` local inside a function that
// also dispatches a real kernel still emits the general pipeline's correct
// char-safe codegen.
#[test]
fn host_string_indexing_and_slicing_in_kernel_touching_fn_is_char_safe() {
    let (_wgsl, rs) = wgpu_codegen("string_indexing_kernel_touching_fn", r#"
kernel NoopKernel:
    mut [float]'unified out
    init():
        out = [0.0]
    def ():
        out[0] = 1.0

def main() throws:
    let s = "hello world"
    let c = s[1]
    let sub = s[0..<5]
    print "{c}"
    print "{sub}"
    mut k = NoopKernel()
    kernel:
        k(block = 1)
    print "{k.out[0]}"
"#);
    assert!(
        rs.contains(".chars().nth(") || rs.contains("__strchars_"),
        "expected `s[1]` to emit char-safe access -- either `.chars().nth(...)` \
         directly, or the general pipeline's `__strchars_`-cached `Vec<char>` \
         shadow (used when a string local is indexed more than once in the \
         same function, as here);\ngot:\n{rs}"
    );
    assert!(
        rs.contains(".chars().skip(") && rs.contains(".take("),
        "expected `s[0..<5]` to emit a char-safe `.chars().skip(...).take(...)` \
         slice;\ngot:\n{rs}"
    );
}

/// `kernel_param_to_field_map` (emit_kernel.rs) scans a kernel's `init()` body for
/// `field = param` assignments (after `desugar_labeled_array` has already expanded a
/// `.reshape(...)` call into a plain `field = source` assignment plus one
/// `__field_axisN = param` assignment per axis) to build a `param name -> field
/// name(s)` map, later used to translate a constructor call's positional arguments
/// into host-side field assignments. It used to map each param name to a single
/// field (`HashMap<String, String>`), silently overwritten by `.insert()` whenever
/// two different fields' shadow-axis assignments happened to reuse the same
/// dimension parameter -- a common, idiomatic pattern (e.g. two `'global` input
/// buffers reshaped with the same `cols`/`rows` params, or an output field's
/// `[value for col = cols, row = rows]` fill reusing an input's own reshape
/// params). The earlier field(s)' axis fields then permanently stayed at their
/// `i32::default()` (`0`) with no error or warning anywhere -- `boring build`
/// succeeds, `cargo build` on the generated project succeeds, and every WGSL
/// labeled-index read computed with a silently-zero stride, aliasing every access
/// onto the same handful of offsets. This is the single most impactful wgpu bug
/// found auditing the `boring-llm` project against real hardware: `.reshape()` on
/// a `'global` input field is the standard, idiomatic way to give a flat incoming
/// buffer 2D/3D indexing semantics inside a kernel, and even the simplest possible
/// kernel using it (a 2D transpose) returned all zeros.
#[test]
fn host_reshape_axis_fields_not_dropped_when_two_fields_share_dimension_params() {
    let src = r#"
kernel Reshaped2D:
    let [float, col, row]'global src
    let [float, col, row]'global other
    mut [float, col, row]'unified dst
    init([float]'global s, [float]'global o, uint cols, uint rows):
        src = s.reshape(col = cols, row = rows)
        other = o.reshape(col = cols, row = rows)
        dst = [0.0 for col = cols, row = rows]
    def ():
        let c = .thread.x
        let r = .thread.y
        if c < 3 and r < 2:
            dst[col = c, row = r] = src[col = c, row = r] + other[col = c, row = r]

mut k = Reshaped2D([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0], 3, 2)
kernel:
    k(block = (4, 4, 1))
print "dst[0] = {k.dst[0]}"
print "dst[1] = {k.dst[1]}"
print "dst[5] = {k.dst[5]}"
"#;
    let (_wgsl, rs) = wgpu_codegen("reshape_axis_fields_two_fields_share_params", src);

    // Codegen-level assertion: every one of `src`'s and `other`'s own two axis
    // fields must be assigned from the constructor call -- not just declared,
    // defaulted, and cloned (which they still are even when the assignment is
    // missing, so those checks alone would pass on the broken code too).
    for field in ["__src_axis0", "__src_axis1", "__other_axis0", "__other_axis1"] {
        assert!(
            rs.lines().any(|l| l.trim_start().starts_with(&format!("k.{field}")) && l.contains(" = ")),
            "expected an assignment statement for '{field}' in the generated constructor \
             call -- it must not be silently dropped just because another field's \
             dynamic-shape assignment reuses the same `cols`/`rows` init params;\ngot:\n{rs}"
        );
    }

    // Real end-to-end value assertion against a real GPU adapter (same rationale as
    // `real_gpu_dispatch_free_fn_with_global_array_param`'s doc comment: this bug
    // class is invisible to `cargo build` on the generated Rust -- the axis fields
    // stay a validly-typed `0` -- and only surfaces as silently-wrong data at real
    // dispatch time).
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("reshape_axis_fields_two_fields_share_params");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU, but it failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    // dst[i] = src[i] + other[i] = src[i] + 10 -- if either field's axis strides
    // silently stayed 0, these reads would alias onto the wrong offsets (or read
    // uninitialized/zeroed buffer contents) instead of the real per-element sum.
    assert!(stdout.contains("dst[0] = 11"), "expected dst[0] = 1 + 10 = 11, got:\n{stdout}");
    assert!(stdout.contains("dst[1] = 12"), "expected dst[1] = 2 + 10 = 12, got:\n{stdout}");
    assert!(stdout.contains("dst[5] = 16"), "expected dst[5] = 6 + 10 = 16, got:\n{stdout}");
}

/// Regression test for a module-scope WGSL name collision between two unrelated
/// kernels: a `'sync`/`'actor` fixed-array workgroup field (`var<workgroup> {name}:
/// array<...>;`, `emit_kernel_decl`'s "3. Workgroup" section) used to be declared
/// under its own bare field name, with no kernel-specific prefix -- unlike every
/// other kernel-scoped WGSL declaration this backend emits (buffer fields go
/// through `current_buffer_renames`'s `{kernel}_{field}` scheme, helper functions
/// through `{kernel}_{method}`, params structs through `{Kernel}Params`). Two
/// independently-written kernels that happen to name their tile field the same
/// thing -- an entirely ordinary choice for two kernels doing a similar tiled
/// operation, e.g. `tile_x` in two GEMM-shaped kernels -- silently combined into
/// one shader module with two `var<workgroup> tile_x: ...;` declarations sharing
/// one module-scope identifier.
///
/// `boring build --target wgpu` and `cargo build` on the generated project both
/// succeed -- a Rust-level compile never parses the embedded WGSL string -- and
/// the failure only surfaces as a real WGSL parse error at shader-module creation
/// time, the first time the program actually dispatches a kernel:
///   Shader '' parsing error: redefinition of `tile_x`
/// followed by a Rust panic reflecting the (invalid) pipeline. Fixed by
/// kernel-prefixing the declaration (`wgsl_workgroup_array_ident`) and its
/// reference site the same way buffer fields already are. See CHANGELOG.md.
#[test]
fn two_kernels_sharing_actor_field_name_real_shader_validation() {
    let src = r#"
kernel KernelA:
    mut [float32]'unified out
    mut [float32, width=4, height=4]'actor tile_x

    init():
        out = [0.0]

    def ():
        tile_x[width=0, height=0] = 1.0
        out[0] = tile_x[width=0, height=0]

kernel KernelB:
    mut [float32]'unified out
    mut [float32, width=4, height=4]'actor tile_x

    init():
        out = [0.0]

    def ():
        tile_x[width=0, height=0] = 2.0
        out[0] = tile_x[width=0, height=0]

mut a = KernelA()
kernel:
    a(block = 1)
mut b = KernelB()
kernel:
    b(block = 1)
print "{a.out[0]} {b.out[0]}"
"#;
    let (wgsl, _emulated, _rs, _toml) = run_wgpu("two_kernels_sharing_actor_field_name", src);

    // Codegen-level assertion: the two kernels' workgroup declarations must no
    // longer share one bare identifier.
    assert!(
        wgsl.contains("var<workgroup> kernela_tile_x:") && wgsl.contains("var<workgroup> kernelb_tile_x:"),
        "expected each kernel's `'actor` workgroup field to be declared under its own \
         kernel-prefixed name, generated WGSL:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("var<workgroup> tile_x:"),
        "generated WGSL still declares the bare, un-prefixed `tile_x` -- two kernels \
         sharing this field name would collide at module scope again, generated WGSL:\n{wgsl}"
    );

    // Real end-to-end run against a real GPU adapter (this repo's CI runs on
    // macOS/Metal, see `test_const_generic_kernel_fixed_array_params_real_shader_validation`'s
    // doc comment) -- `cargo build` alone can't catch this bug since it never parses
    // the embedded WGSL string; only creating a real `wgpu::Device` shader module at
    // runtime does.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("two_kernels_sharing_actor_field_name");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU, but it failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert!(
        !stderr.contains("redefinition of") && !stderr.contains("parsing error"),
        "the workgroup-variable name collision regressed -- generated program's \
         stderr:\n{stderr}"
    );
    assert!(stdout.contains("1 2"), "expected \"1 2\" (each kernel's own tile write \
        read back independently), got:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
}

/// Regression test for literal float division-by-zero (the ordinary way to construct
/// an IEEE-754 Inf/NaN sentinel value -- e.g. in an f16-to-f32 decoder, needed by any
/// format storing f16 values such as GGUF) under `--target wgpu`. WGSL const-folds a
/// literal/literal division at compile time, and the WGSL spec
/// (https://www.w3.org/TR/WGSL/#floating-point-evaluation) makes a const-expression
/// that evaluates to NaN/Inf a shader-creation error; naga enforces this, unlike
/// Metal/CUDA/ROCm's C-family compilers, which treat float div-by-zero as ordinary
/// runtime arithmetic. `boring build --target wgpu` and `cargo build` on the generated
/// project both succeed -- a Rust-level compile never parses the embedded WGSL string
/// -- and the failure only surfaced as a real naga panic at shader-module creation
/// time, the first time the program actually dispatched the kernel:
///   Shader '' parsing error: failed to convert expression to a concrete type: the
///   concrete type `f32` cannot represent the abstract value `inf` accurately
/// Fixed by detecting the literal-literal division pattern in `device::expr`'s
/// `ExprKind::BinOp` `Div` case and emitting the sentinel via `bitcast<f32>` (a
/// bit-preserving reinterpretation of an already-concrete integer literal, which WGSL
/// does not const-fold the same way) instead of a real division. See CHANGELOG.md.
#[test]
fn literal_float_div_by_zero_nan_inf_real_shader_validation() {
    let src = r#"
kernel NanKernel:
    mut [float32]'unified out
    let int flag

    init(int f):
        out = [0.0]
        flag = f

    def ():
        var float32 mag = 0.0
        if flag == 0:
            mag = 1.0 / 0.0
        else:
            mag = 0.0 / 0.0
        out[0] = mag

mut k = NanKernel(0)
kernel:
    k(block = 1)
print "{k.out[0]}"
"#;
    let (wgsl, rs) = wgpu_codegen("literal_float_div_by_zero_nan_inf", src);

    // Codegen-level assertion: the literal-literal division must no longer be emitted
    // verbatim (which naga rejects as a compile-time-constant NaN/Inf) -- it must go
    // through `bitcast<f32>` instead.
    assert!(
        wgsl.contains("bitcast<f32>(0x7f800000u)") && wgsl.contains("bitcast<f32>(0x7fc00000u)"),
        "expected the Inf and NaN sentinel constructions to be emitted via \
         `bitcast<f32>`, generated WGSL:\n{wgsl}"
    );
    assert!(
        !wgsl.contains("1.0 / 0.0") && !wgsl.contains("0.0 / 0.0"),
        "generated WGSL still contains a literal-literal division that naga would \
         reject at shader-creation time:\n{wgsl}"
    );
    let _ = &rs;

    // Real end-to-end run against a real GPU adapter (this repo's CI runs on
    // macOS/Metal, see `test_const_generic_kernel_fixed_array_params_real_shader_validation`'s
    // doc comment) -- `cargo build` alone can't catch this bug since it never parses
    // the embedded WGSL string; only creating a real `wgpu::Device` shader module at
    // runtime does.
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("literal_float_div_by_zero_nan_inf");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU, but it failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert!(
        !stderr.contains("parsing error") && !stderr.contains("cannot represent the abstract value"),
        "the literal-literal NaN/Inf division regressed -- generated program's \
         stderr:\n{stderr}"
    );
    assert!(stdout.trim() == "inf", "expected the Inf branch's real IEEE-754 value \
        printed back, got:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
}

/// Regression test: a kernel field that `init()` assigns a *computed*
/// expression (`half_n = nn / 2`), not a bare `field = param` passthrough,
/// used to be silently left at its Rust default (`0`) by the generated
/// constructor call -- `kernel_param_to_field_map` (the scan feeding
/// `emit_kernel_construction`'s `k.field = ...` assignments) only recognizes
/// a bare passthrough, so a derived field was never assigned anything at all.
/// The field is declared and used correctly inside the kernel body, so
/// nothing else catches the gap: `cargo build` on the generated Rust succeeds
/// cleanly, and the shader dispatches with no validation error -- it just
/// silently computes nothing, because the kernel body's own guard
/// (`i < half_n`) never holds once `half_n` stayed 0.  Metal/CUDA/ROCm never
/// had this bug -- those backends fully replay `init()`'s body inside their
/// own host-side `new()`, computing a derived field for free. Confirmed
/// against a real GPU adapter (same rationale as
/// `host_reshape_axis_fields_not_dropped_when_two_fields_share_dimension_params`'s
/// doc comment: this bug class is invisible to `cargo build`).
#[test]
fn host_init_derived_scalar_field_not_dropped_real_gpu_dispatch() {
    let src = r#"
let N = 16
var [float32]'unified x = [0.0 for ..<N]
for i in 0..<N:
    x[i] = float32(i) + 1.0

kernel Derived:
    let [float32]'unified x
    mut [float32]'unified out
    let int n
    let int half_n

    init([float32]'unified xs, int nn):
        x = xs
        n = nn
        half_n = nn / 2
        out = [0.0 for ..<nn]

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        if i < half_n:
            out[i] = x[i]

mut k = Derived(x, N)
kernel:
    k(block = N, grid = 1)

for i, v in k.out:
    print "out[{i}] = {v}"
"#;
    let (_wgsl, rs) = wgpu_codegen("init_derived_scalar_field", src);

    // Codegen-level assertion: the derived field must get its own computed
    // assignment in the generated constructor call, not just a declaration
    // and a `0` default (which the broken code still produces, so a check
    // for the field's mere presence would pass either way).
    assert!(
        rs.lines().any(|l| {
            let l = l.trim_start();
            l.starts_with("k.half_n") && l.contains(" = ") && l.contains("N / 2")
        }),
        "expected a computed assignment for 'half_n' (derived from `nn / 2`) in the \
         generated constructor call -- it must not be silently dropped just because its \
         init-body expression isn't a bare `field = param` passthrough;\ngot:\n{rs}"
    );

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("init_derived_scalar_field");
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a \
         real GPU, but it failed:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    // `half_n` = N / 2 = 8 -- if it silently stayed 0, `i < half_n` would never hold
    // and every `out[i]` would stay at its zero-filled default instead of `x[i]`.
    for i in 0..8 {
        let expected = format!("out[{i}] = {}", i as f32 + 1.0);
        assert!(
            stdout.contains(&expected),
            "expected '{expected}' (half_n must resolve to 8, copying the first half of x), \
             got:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
    }
    for i in 8..16 {
        let expected = format!("out[{i}] = 0");
        assert!(
            stdout.contains(&expected),
            "expected '{expected}' (second half of out untouched, still zero-filled), \
             got:\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        );
    }
}

#[test]
fn test_gpu_warp_builtins_real_subgroup_path_camel_case() {
    let src = r#"
kernel WarpBuiltins:
    mut [float]'unified buf

    def ():
        let tid = .thread.x
        let lane = .warp.lane
        let size = .warp.size
        .warp.sync()
        let a = .warp.shuffleDown(buf[tid], 1)
        let b = .warp.shuffleUp(buf[tid], 1)
        let c = .warp.shuffleXor(buf[tid], 1)
        let d = .warp.shuffle(buf[tid], 0)
        buf[tid] = a + b + c + d + f32(lane) + f32(size)
"#;
    let (wgsl, emulated, _rs, _toml) = run_wgpu("warp_builtins_real_camel_case", src);

    assert!(wgsl.contains("enable subgroups;"), "expected enable subgroups;\ngot:\n{wgsl}");
    assert!(wgsl.contains("@builtin(subgroup_size)"), "expected @builtin(subgroup_size);\ngot:\n{wgsl}");
    assert!(wgsl.contains("@builtin(subgroup_invocation_id)"), "expected @builtin(subgroup_invocation_id);\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupBarrier()"), "expected subgroupBarrier();\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffleDown("), "expected subgroupShuffleDown;\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffleUp("), "expected subgroupShuffleUp;\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffleXor("), "expected subgroupShuffleXor;\ngot:\n{wgsl}");
    assert!(wgsl.contains("subgroupShuffle("), "expected subgroupShuffle;\ngot:\n{wgsl}");

    // The emulated fallback module must exist alongside the real one whenever
    // `.warp.*` is used, and never uses the subgroup extension.
    assert!(!emulated.is_empty(), "expected shaders/main_emulated.wgsl to be written");
    assert!(!emulated.contains("enable subgroups;"), "emulated module must not enable subgroups;\ngot:\n{emulated}");
}

#[test]
fn test_gpu_warp_shuffle_emulated_fallback_shape_camel_case() {
    let src = r#"
kernel WarpEmulated:
    mut [float32]'unified buf

    def ():
        let tid = .thread.x
        .warp.sync()
        let shuffled = .warp.shuffleDown(buf[tid], 1)
        buf[tid] = shuffled
"#;
    let (_wgsl, emulated, _rs, _toml) = run_wgpu("warp_shuffle_emulated_camel_case", src);

    assert!(emulated.contains("var<workgroup> bp_warp_scratch_warpemulated_f32"),
        "expected a kernel-prefixed f32 workgroup scratch buffer;\ngot:\n{emulated}");
    assert!(emulated.contains("workgroupBarrier()"), "expected workgroupBarrier();\ngot:\n{emulated}");
    assert!(emulated.contains("@builtin(local_invocation_index)"),
        "expected @builtin(local_invocation_index);\ngot:\n{emulated}");
    assert!(emulated.contains("let bp_wsize: u32 = 32u;"), "expected fixed 32-lane fallback constant;\ngot:\n{emulated}");
    assert!(emulated.contains("select("), "expected a select() for the warp-boundary clamp;\ngot:\n{emulated}");
}

/// A plain, non-resident `int`/`float` local inside a `pub req [T]'kernel'unified`
/// function — used only to compute a grid/block dispatch dimension before a
/// `kernel:` call, never itself GPU-resident and never returned — used to get
/// wrongly wrapped in `BoringKernelArg::Host(...)` by the wgpu host-codegen path,
/// as if it were the function's own return value. Root cause: `ExprKind::If`
/// (and `Match`/`Do`) used as an ordinary expression value clones the parent
/// emitter's `current_fn_returns_resident` flag onto its branch sub-emitter
/// (via `make_sub()`), so the branches of `let int gx = if ...: ... else: ...`
/// were treated as if they were the *enclosing function's* tail position. The
/// Metal backend never exhibited this (its host codegen for scalar locals
/// doesn't consult that flag), only wgpu. Confirmed this test fails to compile
/// with `E0308: expected struct Vec<_>, found isize` against the pre-fix code
/// (mismatched types on `BoringKernelArg::Host(blocks.clone())`), and runs to
/// completion with the correct GPU-computed values after.
#[test]
fn wgpu_scalar_local_if_expr_in_resident_fn_not_wrapped_in_boringgpuarg() {
    let test_name = "scalar_local_if_expr_in_resident_fn";
    let src = r#"
kernel AddKernel:
    let [float32]'global x
    mut [float32]'unified y
    let int n

    init([float32]'global xi, int nn):
        x = xi
        n = nn
        y = [0.0 for i in 0..<nn]

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        if i < n:
            y[i] = x[i] + 1.0

pub req [float32]'kernel'unified add_gpu([float32]'global x, int n) throws:
    let int blocks = (n + 31) / 32
    let int gx = if blocks > 65535: 65535 else: blocks
    mut k = AddKernel(x, n)
    kernel:
        k(block = 32, grid = gx)
    k.y

[float32] passthrough([float32] x):
    x

def main() throws:
    let x = [1.0, 2.0, 3.0]
    let result_raw = add_gpu(x, 3)
    let result = passthrough(result_raw)
    print "result={result[0]} {result[1]} {result[2]}"
"#;
    let (_wgsl, _emulated, rs, _toml) = run_wgpu(test_name, src);

    // Codegen shape: the `if`-expression computing `gx` must emit plain `isize`
    // branches (bare `65535`/`blocks`, no residency wrapping at all) -- that
    // wrapping belongs solely to the function's actual tail expression (`k.y`,
    // emitted as `BoringKernelArg::Resident`). `BoringKernelArg::Host` legitimately
    // appears elsewhere (e.g. the `x` parameter's host/resident match), so assert
    // narrowly against `gx`'s own `let` statement rather than the whole file.
    let gx_let_line = rs.lines().find(|l| l.contains("let gx"))
        .unwrap_or_else(|| panic!("expected a `let gx` statement in generated code;\ngot:\n{rs}"));
    assert!(
        !rs.contains("BoringKernelArg::Host((65535")
            && !rs.contains("BoringKernelArg::Host((blocks")
            && !rs.contains("BoringKernelArg::Host(blocks"),
        "expected `gx`'s if-expression branches (`65535`/`blocks`) to stay plain \
         `isize`, not wrapped in `BoringKernelArg::Host(...)` as if they were the \
         function's own return value;\ngx's `let` line:\n{gx_let_line}\nfull output:\n{rs}"
    );
    assert!(
        rs.contains("BoringKernelArg::Resident"),
        "expected the function's real tail expression (`k.y`) to still emit \
         `BoringKernelArg::Resident(...)`;\ngot:\n{rs}"
    );

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join(test_name);
    let manifest = tmp.join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "expected the generated wgpu project to build AND run to completion against a real \
         GPU (a `'kernel'unified`-returning function with a plain-scalar dispatch-dimension \
         local computed via an if-expression), but it failed:\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
    assert!(
        stdout.contains("result=2 3 4"),
        "expected the GPU-computed `x[i] + 1.0` values;\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
    );
}

#[test]
fn screen_key_pressed_is_edge_triggered_and_mouse_is_exposed() {
    let src = r#"
let width  = 64
let height = 48
let screen = Screen(Dimension(width, height), title = "Test")

kernel Fill:
    mut [uint]'surface pixels
    let Dimension dim
    var float32 mx
    var uint mode

    init(Dimension d):
        pixels = [0 for ..<d.width * d.height]
        dim    = d
        mx     = 0.0
        mode   = 0

    def ():
        pixels[0] = 0xFF000000 | mode

var mut k = Fill(Dimension(width, height))

kernel:
    loop:
        k.mx = float32(screen.mouse.x)
        if screen.key_pressed("c") or screen.mouse.left:
            k.mode = 1
        if screen.key("a") and screen.mouse.right and screen.mouse.middle and screen.mouse.y > 3:
            k.mode = 2
        k(block = (16, 16))
        screen.present(k.pixels)
        if screen.key("\x1B"):
            break
"#;
    let (_, rs) = wgpu_codegen("screen_input", src);
    // Edge-triggered: only a released -> pressed transition feeds the set, cleared after each frame body.
    assert!(rs.contains("if self.__keys.insert(key_str.clone()) { self.__keys_pressed.insert(key_str); }"), "{rs}");
    assert!(rs.contains("self.__keys_pressed.contains(&__boring_key_name(\"c\"))"), "{rs}");
    assert!(rs.contains("self.__keys.contains(&__boring_key_name(\"a\"))"), "{rs}");
    assert!(rs.contains("self.__keys_pressed.clear();"), "{rs}");
    // Mouse: CursorMoved (scaled to buffer pixels) + MouseInput buttons.
    assert!(rs.contains("WindowEvent::CursorMoved"), "{rs}");
    assert!(rs.contains("inner_size()"), "{rs}");
    assert!(rs.contains("winit::event::MouseButton::Middle => { self.__mouse_middle = down;"), "{rs}");
    assert!(rs.contains("self.__mouse_x as f32"), "{rs}");
    assert!(rs.contains("self.__mouse_left") && rs.contains("self.__mouse_right"), "{rs}");
    assert!(rs.contains("self.__mouse_y > 3"), "{rs}");
    // The existing `if screen.key(..): break` form still exits.
    assert!(rs.contains("event_loop.exit()"), "{rs}");
}

#[test]
fn top_level_tensor_arrays_resolve_as_statics_inside_function() {
    let (_dev, host) = wgpu_codegen("toplevel_tensor_in_fn", r#"
let [float32, k = 8, m = 8]'kernel'global a = [1.0 as float32 for ..<64]
let [float32, n = 8, k = 8]'kernel'global b = [1.0 as float32 for ..<64]
mut [float32, n = 8, m = 8]'kernel'unified c = [0.0 as float32 for ..<64]

def run() throws:
    kernel.tensors.matmul(a, b, c)

run()
with c:
    print "{c[0]}"
"#);
    assert!(host.contains("static A: std::sync::LazyLock<Vec<f32>>"), "{host}");
    assert!(host.contains("copy_a_to_device(&A.iter()"), "{host}");
    assert!(host.contains("C.lock()"), "{host}");
}

#[test]
fn wgpu_params_struct_widens_sub_word_scalars_and_compiles() {
    // A `uint8`/`int8`/`uint16`/`bool` scalar field used to be emitted at its own width in
    // the host `{Kernel}Params` struct, which `#[derive(Pod)]` rejects (implicit padding;
    // `bool` is not Pod at all), and which disagreed with the WGSL layout (32-bit slots).
    let src = r#"
kernel Mix:
    mut [float32]'unified out
    let [int32]'unified ints
    let [uint8]'global bytes
    let int n
    let uint m
    let float32 a
    let uint8 b
    let int8 c
    let uint16 d
    let bool e

    init(int nn, uint mm, float32 aa, uint8 bb, int8 cc, uint16 dd, bool ee, [int32]'unified i, [uint8]'global by, [float32]'unified o):
        n = nn
        m = mm
        a = aa
        b = bb
        c = cc
        d = dd
        e = ee
        ints = i
        bytes = by
        out = o

    def ():
        let t = .thread.x
        if t < n:
            out[t] = a * float32(ints[t]) + float32(bytes[t]) + float32(b) + float32(m) + float32(c) + float32(d)

def main() throws:
    mut k = Mix(4, 5, 2.0, 7, -1, 3, true, [1, 2, 3, 4], [10, 20, 30, 40], [0.0, 0.0, 0.0, 0.0])
    kernel:
        k(block = 4, grid = 1)
    print k.out[0]
"#;
    let (_wgsl, rs) = wgpu_codegen("params_sub_word_scalars", src);
    let params = &rs[rs.find("struct MixParams").expect("Params struct")..];
    let params = &params[..params.find('}').unwrap()];
    for bad in [": u8", ": i8", ": u16", ": bool"] {
        assert!(!params.contains(bad), "params struct must use 32-bit slots, found `{bad}`:\n{params}");
    }
    assert!(params.contains("b: u32") && params.contains("c: i32") && params.contains("e: u32"), "{params}");

    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("params_sub_word_scalars");
    let build = common::SharedCargo::new("wgpu_codegen")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(tmp.join("test_wgpu").join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    assert!(build.status.success(), "generated wgpu project failed to compile:\n{}",
        String::from_utf8_lossy(&build.stderr));
}

/// A `continue` in a `for` loop must still run the loop increment: the increment
/// goes in a WGSL `continuing` block, otherwise the shader loops forever (GPU hang).
#[test]
fn continue_in_for_loop_runs_increment_via_continuing_block() {
    let (wgsl, _) = wgpu_codegen("continue_for_continuing", r#"
kernel Neighbors:
    let [int]'global a
    mut [int]'unified out
    let int n

    def ():
        let i = .thread.x
        if i < n:
            var int s = 0
            for dy in -1..<2:
                for dx in -1..<2:
                    if dx == 0 and dy == 0:
                        continue
                    s += a[i]
            for k in 0..<3:
                if k == 1:
                    continue
                s += k
            out[i] = s
"#);
    assert_eq!(wgsl.matches("continuing {").count(), 3, "{wgsl}");
    assert!(wgsl.contains("continue;"), "{wgsl}");
    assert!(wgsl.contains("dx = dx + 1;"), "{wgsl}");
    // The increment must be the sole content of each `continuing` block.
    for v in ["dx", "dy", "k"] {
        let want = format!("continuing {{\n{v} = {v} + 1;", );
        let squashed: String = wgsl.lines().map(|l| l.trim()).collect::<Vec<_>>().join("\n");
        assert!(squashed.contains(&want), "missing continuing increment for {v}:\n{wgsl}");
    }
}

// Every 32-bit integer spelling must get an `array<atomic<..>>` declaration on an
// `'actor'unified`/`'actor'global` buffer (int32/uint32/uint used to be missed,
// leaving `atomicAdd` on a non-atomic `array<i32>`).
#[test]
fn atomic_buffer_declared_atomic_for_every_32bit_int_spelling() {
    for (ty, wgsl_ty) in [("int", "i32"), ("int32", "i32"), ("uint", "u32"), ("uint32", "u32")] {
        for qual in ["'actor'unified", "'actor'global"] {
            let src = format!(r#"
kernel Histogram:
    let [{ty}]'unified values
    mut [{ty}]{qual} counts
    def ():
        let i = .block.x * .block_dim.x + .thread.x
        counts[values[i] % 4] += 1
"#);
            let name = format!("atomic_spelling_{}_{}", ty, qual.replace('\'', "_"));
            let (wgsl, _rs) = wgpu_codegen(&name, &src);
            assert!(wgsl.contains(&format!("array<atomic<{wgsl_ty}>>")),
                "`[{ty}]{qual}` should be declared array<atomic<{wgsl_ty}>>;\ngot:\n{wgsl}");
            assert!(wgsl.contains("atomicAdd"), "missing atomicAdd for `[{ty}]{qual}`;\ngot:\n{wgsl}");
        }
    }
}

// The ping-pong swap `k.a, k.b = k.b, k.a` of two buffer fields used to emit
// `(k.a.clone(), k.b.clone()) = (k.copy_b_to_host()..., ...)` (rustc E0070); it now
// exchanges the device buffers (and the element counts of packed byte kinds) and rebuilds
// the bind group. Runs against a real GPU adapter, like the tests above.
#[test]
fn ping_pong_swap_of_two_buffer_fields_exchanges_device_buffers() {
    for (ty, packed) in [("uint", false), ("uint8", true)] {
        let src = format!(r#"
kernel Step:
    mut [{ty}]'unified a
    mut [{ty}]'unified b
    init(mut [{ty}]'unified a0, mut [{ty}]'unified b0):
        a = a0
        b = b0
    def ():
        let i = .thread.x
        a[i] = a[i] + 1
        b[i] = b[i] * 2
mut k = Step([{ty}(i + 1) for i in 0..<10], [{ty}(i + 1) for i in 0..<10])
kernel:
    k(block = 10)
k.a, k.b = k.b, k.a
kernel:
    k(block = 10)
print "{{k.a[1]}} {{k.b[1]}}"
"#);
        let name = format!("ping_pong_swap_{ty}");
        let (_wgsl, _emulated, rs, _toml) = run_wgpu(&name, &src);
        assert!(rs.contains("std::mem::swap(&mut k.a_buf, &mut k.b_buf); "), "{rs}");
        assert_eq!(rs.contains("std::mem::swap(&mut k.a_len, &mut k.b_len);"), packed, "{rs}");
        assert!(rs.contains("k.rebuild_bind_group();"), "{rs}");
        assert!(!rs.contains("(k.a.clone(), k.b.clone())"), "{rs}");

        let manifest = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("wgpu_codegen").join(&name).join("test_wgpu").join("Cargo.toml");
        let run = common::SharedCargo::new("wgpu_codegen")
            .args(["run", "--quiet", "--manifest-path"])
            .arg(&manifest)
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(run.status.success(), "[{ty}] build/run failed:\n{stdout}\n{}", String::from_utf8_lossy(&run.stderr));
        // Element 1 starts at 2/2: 3/4 after step one; swapped 4/3; step two: 5/6 (no swap gives 4 8).
        assert!(stdout.contains("5 6"), "[{ty}] expected `5 6`, got:\n{stdout}");
    }
}

// A buffer field the kernel body never touches used to make the host's bind group (all
// fields) disagree with the pipeline's automatic layout (only used bindings): wgpu rejected it
// with "Number of bindings in bind group descriptor (2) does not match ... layout (1)".
// The pipeline now has an explicit layout covering every buffer field. Real-GPU run.
#[test]
fn unused_buffer_field_does_not_break_bind_group_creation() {
    let src = r#"
kernel Unused:
    mut [uint]'unified a
    mut [uint]'unified b
    init(mut [uint]'unified a0, mut [uint]'unified b0):
        a = a0
        b = b0
    def ():
        let i = .thread.x
        a[i] = a[i] + 1
mut k = Unused([uint(i) for i in 0..<10], [uint(i * 2) for i in 0..<10])
kernel:
    k(block = 10)
k.a, k.b = k.b, k.a
kernel:
    k(block = 10)
print "{k.a[1]} {k.b[1]}"
"#;
    let (_wgsl, _emulated, rs, _toml) = run_wgpu("unused_buffer_field_bind_group", src);
    assert!(rs.contains("create_pipeline_layout"), "{rs}");
    let manifest = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("unused_buffer_field_bind_group").join("test_wgpu").join("Cargo.toml");
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(run.status.success(), "build/run failed:\n{stdout}\n{}", String::from_utf8_lossy(&run.stderr));
    // a: 1 -> 2; b untouched (2); swapped: a=2, b=2; step two: a=3.
    assert!(stdout.contains("3 2"), "expected `3 2`, got:\n{stdout}");
}

// ── Fixed-point types in kernels (docs/fixed-point-types.md, "wgpu") ─────────────

/// Parses + validates `wgsl` with naga (the front end and validator wgpu itself uses).
fn assert_valid_wgsl(what: &str, wgsl: &str) {
    let module = naga::front::wgsl::parse_str(wgsl)
        .unwrap_or_else(|e| panic!("[{what}] WGSL does not parse: {}\n{wgsl}", e.emit_to_string(wgsl)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("[{what}] WGSL does not validate: {e:?}\n{wgsl}"));
}

/// A kernel exercising every lowered operation on format `t` (`signed`: also negation / `abs`).
fn fixed_ops_kernel(t: &str, signed: bool) -> String {
    let neg = if signed { "        neg[i] = -a[i]\n        absd[i] = a[i].abs()\n" } else { "" };
    let neg_field = if signed { "    mut [@T@]'unified neg\n    mut [@T@]'unified absd\n" } else { "" };
    let neg_init = if signed { ", [@T@]'unified n0, [@T@]'unified ab0" } else { "" };
    let neg_assign = if signed { "        neg = n0\n        absd = ab0\n" } else { "" };
    format!(r#"
@T@ half(@T@ x, @T@ y):
    x * 0.5 + y

@T@ one():
    1

kernel Ops:
    let @T@ k
    let [@T@]'unified a
    let [@T@]'unified b
    mut [@T@]'unified out
    mut [@T@]'unified fl
    mut [int32]'unified ints
    mut [float32]'unified flts
    mut [fixed8_8]'unified small
    mut [ufixed8_24]'unified wide
    mut [fixed4_4]'unified tiny
{neg_field}
    init(@T@ kk, [@T@]'unified a0, [@T@]'unified b0, [@T@]'unified o0, [@T@]'unified f0, [int32]'unified i0, [float32]'unified l0, [fixed8_8]'unified s0, [ufixed8_24]'unified w0, [fixed4_4]'unified t0{neg_init}):
        k = kk
        a = a0
        b = b0
        out = o0
        fl = f0
        ints = i0
        flts = l0
        small = s0
        wide = w0
        tiny = t0
{neg_assign}
    def ():
        let i = .thread.x
        let @T@ s = a[i] + b[i] - 0.5
        let @T@ m = a[i] * b[i]
        let @T@ q = a[i] / (b[i] + 1.0)
        let @T@ r = a[i] % 1.25
        var @T@ acc = half(s, one())
        if acc < k:
            acc = acc + 0.25
        elif a[i] >= b[i]:
            acc = acc - m
        out[i] = acc + m + q + r + @T@.epsilon + @T@.fromRaw(7)
        fl[i] = a[i].floor() + a[i].ceil() + a[i].trunc() + a[i].fract() + a[i].min(b[i]) + a[i].max(b[i])
        ints[i] = a[i].raw + (a[i] as int32) + (i as int32) * 3 + @T@.max.raw + @T@.min.raw
        flts[i] = (a[i] as float32) * 2.0 + ((i as float32) as @T@) as float32
        small[i] = (a[i] as fixed8_8) * 2.5
        wide[i] = (small[i] as ufixed8_24) / 3
        tiny[i] = (a[i] as fixed4_4) + (i as fixed4_4)
{neg}"#).replace("@T@", t)
}

#[test]
fn fixed_point_ops_validate_in_naga_for_every_supported_width() {
    for (t, signed) in [
        ("fixed4_4", true), ("ufixed4_4", false),
        ("fixed8_8", true), ("ufixed8_8", false), ("fixed4_12", true), ("ufixed12_4", false),
        ("fixed16_16", true), ("ufixed16_16", false), ("fixed8_24", true), ("ufixed8_24", false),
        ("fixed2_30", true), ("fixed24_8", true),
        // 64-bit raws: `vec2<u32>` limb pairs.
        ("fixed32_32", true), ("ufixed32_32", false), ("fixed48_16", true), ("ufixed48_16", false),
        ("fixed16_48", true), ("ufixed16_48", false),
    ] {
        let name = format!("fixed_ops_{t}");
        let (wgsl, _, _, _) = run_wgpu(&name, &fixed_ops_kernel(t, signed));
        assert_valid_wgsl(&name, &wgsl);
        assert!(!wgsl.contains("ERROR"), "[{t}] {wgsl}");
        // The WGSL carries no fixed-point type names: everything lowered to i32/u32.
        assert!(!wgsl.contains("fixed"), "[{t}] leaked a fixed-point type name:\n{wgsl}");
    }
}

#[test]
fn fixed_point_w16_lowers_directly_through_32_bit_integers() {
    let (wgsl, _, _, _) = run_wgpu("fixed_w16_direct", r#"
kernel F:
    let [fixed8_8]'unified a
    let [fixed8_8]'unified b
    mut [fixed8_8]'unified out
    mut [ufixed8_8]'unified uo
    init([fixed8_8]'unified a0, [fixed8_8]'unified b0, [fixed8_8]'unified o0, [ufixed8_8]'unified u0):
        a = a0
        b = b0
        out = o0
        uo = u0
    def ():
        let i = .thread.x
        out[i] = a[i] * b[i] + a[i] / b[i]
        uo[i] = (a[i] as ufixed8_8) % 3.5
"#);
    assert_valid_wgsl("fixed_w16_direct", &wgsl);
    // Packed 16-bit raws: a read-only buffer is `array<u32>`, a written one `array<atomic<u32>>`.
    assert!(wgsl.contains("var<storage, read> f_a: array<u32>;"), "{wgsl}");
    assert!(wgsl.contains("var<storage, read_write> f_out: array<atomic<u32>>;"), "{wgsl}");
    // Signed packed read sign-extends; the product divides by 2^8 (never a shift) and re-narrows.
    assert!(wgsl.contains("extractBits(bitcast<i32>(f_a[u32(i) / 2u])"), "{wgsl}");
    assert!(wgsl.contains("/ 256)"), "{wgsl}");
    assert!(wgsl.contains("extractBits("), "{wgsl}");
    assert!(!wgsl.contains(">> 8u"), "{wgsl}");
    // No 64-bit emulation helper is needed for W <= 16.
    assert!(!wgsl.contains("bp_fx_"), "{wgsl}");
    // Untyped literal 3.5 in ufixed8_8 -> raw 896.
    assert!(wgsl.contains("896u"), "{wgsl}");
}

#[test]
fn fixed_point_w32_uses_limb_helpers_only_when_needed() {
    // `+ - %` and compares are plain integer ops: no helper emitted.
    let (wgsl, _, _, _) = run_wgpu("fixed_w32_plain", r#"
kernel F:
    let [fixed16_16]'unified a
    mut [fixed16_16]'unified out
    init([fixed16_16]'unified a0, [fixed16_16]'unified o0):
        a = a0
        out = o0
    def ():
        let i = .thread.x
        out[i] = a[i] + 2.5 - a[i] % 3
"#);
    assert_valid_wgsl("fixed_w32_plain", &wgsl);
    assert!(!wgsl.contains("bp_fx_"), "{wgsl}");
    assert!(wgsl.contains("i32(163840)"), "2.5 -> raw 163840:\n{wgsl}");

    // `*` and `/` pull in exactly their helpers (with dependencies), signed and unsigned.
    let (wgsl, _, _, _) = run_wgpu("fixed_w32_mul_div", r#"
kernel F:
    let [fixed16_16]'unified a
    let [ufixed16_16]'unified u
    mut [fixed16_16]'unified out
    mut [ufixed16_16]'unified uo
    init([fixed16_16]'unified a0, [ufixed16_16]'unified u0, [fixed16_16]'unified o0, [ufixed16_16]'unified uo0):
        a = a0
        u = u0
        out = o0
        uo = uo0
    def ():
        let i = .thread.x
        out[i] = a[i] * a[i]
        uo[i] = u[i] / u[i]
"#);
    assert_valid_wgsl("fixed_w32_mul_div", &wgsl);
    for helper in ["bp_fx_umul64", "bp_fx_shr64", "bp_fx_mul_u32", "bp_fx_mul_i32",
                   "bp_fx_udiv64", "bp_fx_div_u32"] {
        assert_eq!(wgsl.matches(&format!("fn {helper}(")).count(), 1, "{helper} emitted once:\n{wgsl}");
    }
    assert!(!wgsl.contains("fn bp_fx_div_i32("), "unused helper emitted:\n{wgsl}");
    assert!(wgsl.contains("bp_fx_mul_i32(f_a[u32(i)], f_a[u32(i)], 16u)"), "{wgsl}");
    assert!(wgsl.contains("bp_fx_div_u32(f_u[u32(i)], f_u[u32(i)], 16u)"), "{wgsl}");
    // The helpers are module-level functions placed before the kernel code that calls them.
    assert!(wgsl.find("fn bp_fx_mul_i32(").unwrap() < wgsl.find("fn F_main(").unwrap(), "{wgsl}");
}

#[test]
fn fixed_point_literals_fold_to_raw_constants_and_round_to_nearest() {
    let (wgsl, _, _, _) = run_wgpu("fixed_literals", r#"
kernel F:
    mut [fixed16_16]'unified out
    mut [ufixed16_16]'unified uo
    init([fixed16_16]'unified o0, [ufixed16_16]'unified u0):
        out = o0
        uo = u0
    def ():
        let i = .thread.x
        let fixed16_16 h = 0.5
        out[i] = h + 0.1 - 2
        uo[i] = 65535.999990
"#);
    assert_valid_wgsl("fixed_literals", &wgsl);
    assert!(wgsl.contains("i32(32768)"), "0.5:\n{wgsl}");
    assert!(wgsl.contains("i32(6554)"), "0.1 rounds to nearest (6553.6 -> 6554):\n{wgsl}");
    assert!(wgsl.contains("i32(-131072)") || wgsl.contains("i32(131072)"), "2:\n{wgsl}");
    assert!(wgsl.contains("4294967295u"), "max ufixed16_16 literal:\n{wgsl}");
}

#[test]
fn fixed_point_members_and_casts_lower_as_documented() {
    let (wgsl, _, rs, _) = run_wgpu("fixed_members", r#"
kernel F:
    let [fixed16_16]'unified a
    mut [int32]'unified ints
    mut [float32]'unified flts
    mut [fixed16_16]'unified out
    init([fixed16_16]'unified a0, [int32]'unified i0, [float32]'unified f0, [fixed16_16]'unified o0):
        a = a0
        ints = i0
        flts = f0
        out = o0
    def ():
        let i = .thread.x
        ints[i] = (a[i] as int32) + a[i].raw
        flts[i] = a[i] as float32
        out[i] = fixed16_16.fromRaw(ints[i]) + (i as fixed16_16) + (flts[i] as fixed16_16)
        out[i] = a[i].floor() + a[i].ceil() + a[i].trunc() + a[i].fract() + fixed16_16.epsilon + fixed16_16.min
        out[i] = a[i].abs().min(fixed16_16.max).max(0)
"#);
    assert_valid_wgsl("fixed_members", &wgsl);
    // fixed -> int truncates toward zero (helper, since 2^16 does not fit a shift of a negative number).
    assert!(wgsl.contains("bp_fx_tdiv_i32(f_a[u32(i)], 16u)"), "{wgsl}");
    // fixed -> float divides by 2^16.
    assert!(wgsl.contains("f32(f_a[u32(i)]) / 65536.0"), "{wgsl}");
    // int -> fixed shifts left; float -> fixed multiplies by 2^16.
    assert!(wgsl.contains("i32(i) << 16u"), "{wgsl}");
    assert!(wgsl.contains("i32(f32(") && wgsl.contains(") * 65536.0)"), "{wgsl}");
    // floor masks (arithmetic floor), ceil adds the mask first.
    assert!(wgsl.contains("& i32(-65536)"), "{wgsl}");
    assert!(wgsl.contains("i32(65535)"), "{wgsl}");
    // min/max members are the WGSL builtins on raw values; epsilon / min literal.
    assert!(wgsl.contains("min(") && wgsl.contains("max("), "{wgsl}");
    assert!(wgsl.contains("i32(-2147483648)") && wgsl.contains("i32(2147483647)"), "{wgsl}");
    // Host: the fixed buffer crosses as the repr(transparent) newtype, marked Pod, no `as` casts.
    assert!(rs.contains("unsafe impl bytemuck::Pod for Fixed16_16 {}"), "{rs}");
    assert!(!rs.contains(" as Fixed16_16"), "{rs}");
    assert!(rs.contains("a: Vec<Fixed16_16>"), "{rs}");
    assert!(rs.contains("k: i32") || rs.contains("a_buf"), "{rs}");
}

#[test]
fn fixed_point_scalar_param_is_uploaded_as_raw_integer() {
    let (wgsl, _, rs, _) = run_wgpu("fixed_scalar_param", r#"
kernel F:
    let fixed16_16 k
    let ufixed8_8 g
    mut [fixed16_16]'unified out
    init(fixed16_16 kk, ufixed8_8 gg, [fixed16_16]'unified o0):
        k = kk
        g = gg
        out = o0
    def ():
        let i = .thread.x
        out[i] = k * (g as fixed16_16)

mut o = [0.0 as fixed16_16 for ..<4]
mut f = F(1.5 as fixed16_16, 2.25 as ufixed8_8, o)
kernel:
    f(block = 4)
"#);
    assert_valid_wgsl("fixed_scalar_param", &wgsl);
    assert!(wgsl.contains("k: i32,") && wgsl.contains("g: u32,"), "{wgsl}");
    assert!(rs.contains(".0 as i32") && rs.contains(".0 as u32"), "{rs}");
}

#[test]
fn fixed_point_function_params_returns_and_locals_are_typed() {
    let (wgsl, _, _, _) = run_wgpu("fixed_fns", r#"
fixed16_16 lerp(fixed16_16 x, fixed16_16 y, fixed16_16 t):
    x + (y - x) * t

fixed16_16 half():
    0.5

ufixed8_8 up(ufixed8_8 x):
    return x + 1

kernel F:
    mut [fixed16_16]'unified out
    init([fixed16_16]'unified o0):
        out = o0
    def ():
        let i = .thread.x
        let y = lerp(1.0, 2.0, 0.25)
        let z = y * half()
        var [ufixed4_4, 2] arr = [1.0, 2.5]
        out[i] = z + (up(1.5) as fixed16_16) + (arr[1] as fixed16_16)
"#);
    assert_valid_wgsl("fixed_fns", &wgsl);
    assert!(wgsl.contains("fn lerp(x: i32, y: i32, t: i32) -> i32"), "{wgsl}");
    assert!(wgsl.contains("fn half() -> i32 {\n    return i32(32768);"), "{wgsl}");
    assert!(wgsl.contains("fn up(x: u32) -> u32"), "{wgsl}");
    // Untyped literal arguments fold against the callee's declared parameter formats.
    assert!(wgsl.contains("lerp(i32(65536), i32(131072), i32(16384))"), "{wgsl}");
    assert!(wgsl.contains("up(384u)"), "{wgsl}");
    // A local fixed array folds its literals to raw constants (2.5 in ufixed4_4 -> 40).
    assert!(wgsl.contains("array<u32, 2>(16u, 40u)"), "{wgsl}");
}

#[test]
fn fixed_point_unsupported_cases_are_compile_errors_on_wgpu() {
    let cases: [(&str, &str, &str); 4] = [
        ("w128", r#"
kernel F:
    mut [fixed64_64]'unified out
    init([fixed64_64]'unified o):
        out = o
    def ():
        out[.thread.x] = 1.0
"#, "fixed64_64"),
        ("checked_member", r#"
kernel F:
    mut [fixed16_16]'unified out
    init([fixed16_16]'unified o):
        out = o
    def ():
        out[.thread.x] = out[.thread.x].checkedAdd(out[.thread.x]) ?? 0
"#, "checkedAdd"),
        ("literal_out_of_range", r#"
kernel F:
    mut [fixed8_8]'unified out
    init([fixed8_8]'unified o):
        out = o
    def ():
        out[.thread.x] = 1000.0
"#, "out of range"),
        ("actor_fixed_buffer", r#"
kernel F:
    mut [fixed16_16]'actor'unified acc
    init([fixed16_16]'actor'unified a):
        acc = a
    def ():
        acc[0] += 1
"#, "atomic"),
    ];
    for (name, src, needle) in cases {
        let err = run_wgpu_expect_failure(&format!("fixed_err_{name}"), src);
        assert!(err.contains(needle), "[{name}] expected `{needle}` in:\n{err}");
        if name != "literal_out_of_range" && name != "actor_fixed_buffer" && name != "checked_member" {
            assert!(err.contains("--target wgpu"), "[{name}] {err}");
        }
    }
}

/// Builds `src` for wgpu, runs it on the real GPU and compares stdout with `boring run` byte for byte.
fn assert_wgpu_matches_interpreter(name: &str, src: &str) {
    let (_wgsl, _emulated, _rs, _toml) = run_wgpu(name, src);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wgpu_codegen").join(name);
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(dir.join("test_wgpu").join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(run.status.success(), "generated wgpu program failed:\n{stdout}\n{}", String::from_utf8_lossy(&run.stderr));
    let reference = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(dir.join("test.br")).output().unwrap();
    assert!(reference.status.success(), "{}", String::from_utf8_lossy(&reference.stderr));
    assert_eq!(stdout.trim_end(), String::from_utf8_lossy(&reference.stdout).trim_end());
}

/// The full example (fixed16_16, fixed8_8, fixed8_24, ufixed16_16, members, helper fns) built
/// for wgpu and run on the device prints exactly what `boring run` prints.
#[test]
fn fixed_point_kernel_runs_on_wgpu_and_matches_interpreter() {
    let src = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/fixed_point_kernel.br")).unwrap();
    assert_wgpu_matches_interpreter("fixed_point_run", &src);
}

/// Edge-case sweep for the 32-bit limb emulation: `*`, `/`, `%` and the members on mixed signs
/// and magnitudes for signed, unsigned and fraction-only formats (the 64-bit intermediate
/// matters here), plus 8- and 16-bit formats, against the interpreter.
#[test]
fn fixed_point_edge_values_run_on_wgpu_and_match_interpreter() {
    fn program(t: &str, signed: bool, raw_a: &str, raw_b: &str) -> String {
        // `ceil` of a fraction-only format would reach 1.0, which is out of range (an interpreter error).
        let mem = if t == "ufixed0_32" { "x.floor() + x.trunc() + x.fract() - x.min(y) + x.max(y)" }
            else { "x.floor() + x.ceil() - x.trunc() + x.fract() - x.min(y) + x.max(y)" };
        let neg = if signed { "        ng[i] = -a[i]\n        ab[i] = a[i].abs()\n" } else { "" };
        let neg_decl = if signed { "mut [T]'kernel'unified ng = [T.fromRaw(0) for ..<N]\nmut [T]'kernel'unified ab = [T.fromRaw(0) for ..<N]\n" } else { "" };
        let neg_field = if signed { "    mut [T]'unified ng\n    mut [T]'unified ab\n" } else { "" };
        let neg_param = if signed { ", [T]'unified n0, [T]'unified b0x" } else { "" };
        let neg_set = if signed { "        ng = n0\n        ab = b0x\n" } else { "" };
        let neg_arg = if signed { ", ng, ab" } else { "" };
        let neg_print = if signed { " {ng[i]} {ab[i]}" } else { "" };
        format!(r#"
let N = 16
mut [T]'kernel'unified a = [T.fromRaw(0) for ..<N]
mut [T]'kernel'unified b = [T.fromRaw(0) for ..<N]
mut [T]'kernel'unified mu = [T.fromRaw(0) for ..<N]
mut [T]'kernel'unified dv = [T.fromRaw(0) for ..<N]
mut [T]'kernel'unified rm = [T.fromRaw(0) for ..<N]
mut [T]'kernel'unified mem = [T.fromRaw(0) for ..<N]
mut [int32]'kernel'unified ints = [0 for ..<N]
{neg_decl}
for i in 0..<N:
    a[i] = T.fromRaw({raw_a})
    b[i] = T.fromRaw({raw_b})

kernel K:
    let [T]'unified a
    let [T]'unified b
    mut [T]'unified mu
    mut [T]'unified dv
    mut [T]'unified rm
    mut [T]'unified mem
    mut [int32]'unified ints
{neg_field}
    init([T]'unified a0, [T]'unified b0, [T]'unified m0, [T]'unified d0, [T]'unified r0, [T]'unified e0, [int32]'unified i0{neg_param}):
        a = a0
        b = b0
        mu = m0
        dv = d0
        rm = r0
        mem = e0
        ints = i0
{neg_set}
    def ():
        let i = .thread.x
        mu[i] = a[i] * b[i]
        dv[i] = a[i] / b[i]
        rm[i] = a[i] % b[i]
        let x = a[i]
        let y = b[i]
        mem[i] = {mem}
        # `buf[i].min/max(..)` on a fixed-point element is the plain member, in `boring run` too.
        ints[i] = (a[i] as int32) + (mu[i] as int32) * 2 + (a[i].max(b[i]) as int32) - (b[i].min(a[i]) as int32)
{neg}
mut k = K(a, b, mu, dv, rm, mem, ints{neg_arg})
kernel:
    k(block = N, grid = 1)

for i in 0..<N:
    print "{{k.a[i]}} {{k.b[i]}} {{k.mu[i]}} {{k.dv[i]}} {{k.rm[i]}} {{k.mem[i]}} {{k.ints[i]}}{neg_print}"
"#).replace("T.fromRaw", &format!("{t}.fromRaw")).replace("[T]", &format!("[{t}]"))
    }
    let sgn = "(1 - 2 * (i % 2))";
    let cases = [
        ("fixed16_16", true, "(i - 8) * 417233".to_string(), format!("(2 * i + 1) * 73001 * {sgn}")),
        ("fixed8_24", true, "(i - 8) * 9000001".to_string(), format!("(2 * i + 1) * 4000003 * {sgn}")),
        ("ufixed16_16", false, "(i + 1) * 130001".to_string(), "(2 * i + 1) * 100003".to_string()),
        ("ufixed0_32", false, "(i + 1) * 100000007".to_string(), "(i + 1) * 100000007 + 2000000011".to_string()),
        ("fixed8_8", true, "(i - 8) * 400".to_string(), format!("(2 * i + 1) * 40 * {sgn}")),
        ("ufixed8_8", false, "(i + 1) * 500".to_string(), "(2 * i + 1) * 60".to_string()),
        ("fixed4_4", true, "(i - 8) * 2".to_string(), format!("(2 * (i % 4) + 1) * 3 * {sgn}")),
    ];
    for (t, signed, ra, rb) in cases {
        assert_wgpu_matches_interpreter(&format!("fixed_edge_{t}"), &program(t, signed, &ra, &rb));
    }
}

/// 64-bit raws are `vec2<u32>` limb pairs: buffers are `array<vec2<u32>>`, a scalar parameter
/// crosses the params struct as two `u32` slots, and the `bp_fx_*64` helpers are emitted only when used.
#[test]
fn fixed_point_w64_lowers_to_limb_pairs_and_helpers_on_demand() {
    // `+ - neg floor` need only add/sub/neg helpers: no 128-bit product or division machinery.
    let (wgsl, _, rs, _) = run_wgpu("fixed_w64_plain", r#"
kernel F:
    let fixed32_32 k
    let [fixed32_32]'unified a
    mut [fixed32_32]'unified out
    init(fixed32_32 kk, [fixed32_32]'unified a0, [fixed32_32]'unified o0):
        k = kk
        a = a0
        out = o0
    def ():
        let i = .thread.x
        out[i] = -(a[i] + k - 2.5).floor()

mut o = [0.0 as fixed32_32 for ..<4]
mut f = F(1.5 as fixed32_32, o, o)
kernel:
    f(block = 4)
"#);
    assert_valid_wgsl("fixed_w64_plain", &wgsl);
    assert!(wgsl.contains("array<vec2<u32>>"), "{wgsl}");
    assert!(wgsl.contains("k_lo: u32,") && wgsl.contains("k_hi: u32,"), "{wgsl}");
    assert!(wgsl.contains("let k: vec2<u32> = vec2<u32>("), "{wgsl}");
    assert!(wgsl.contains("fn bp_fx_add64(") && wgsl.contains("fn bp_fx_neg64("), "{wgsl}");
    assert!(!wgsl.contains("bp_fx_umulshr64") && !wgsl.contains("bp_fx_udivshl64"), "{wgsl}");
    // 2.5 in fixed32_32 -> raw 0x2_8000_0000 = (hi 2, lo 0x80000000).
    assert!(wgsl.contains("vec2<u32>(2147483648u, 2u)"), "{wgsl}");
    assert!(rs.contains("k_lo:") && rs.contains("k_hi:"), "{rs}");

    // `*` and `/` pull in exactly their helpers (and dependencies), signed vs unsigned.
    let (wgsl, _, _, _) = run_wgpu("fixed_w64_mul_div", r#"
kernel F:
    let [fixed32_32]'unified a
    let [ufixed32_32]'unified u
    mut [fixed32_32]'unified out
    mut [ufixed32_32]'unified uo
    init([fixed32_32]'unified a0, [ufixed32_32]'unified u0, [fixed32_32]'unified o0, [ufixed32_32]'unified uo0):
        a = a0
        u = u0
        out = o0
        uo = uo0
    def ():
        let i = .thread.x
        out[i] = a[i] * a[i]
        uo[i] = u[i] / u[i]
"#);
    assert_valid_wgsl("fixed_w64_mul_div", &wgsl);
    for helper in ["bp_fx_umul64", "bp_fx_umulshr64", "bp_fx_mul_i64", "bp_fx_udivshl64", "bp_fx_div_u64"] {
        assert_eq!(wgsl.matches(&format!("fn {helper}(")).count(), 1, "{helper} emitted once:\n{wgsl}");
    }
    assert!(!wgsl.contains("fn bp_fx_div_i64(") && !wgsl.contains("fn bp_fx_mul_u64("), "unused helper emitted:\n{wgsl}");
    assert!(wgsl.contains("bp_fx_mul_i64(f_a[u32(i)], f_a[u32(i)], 32u)"), "{wgsl}");
    assert!(wgsl.find("fn bp_fx_mul_i64(").unwrap() < wgsl.find("fn F_main(").unwrap(), "{wgsl}");
}

/// The 64-bit example (fixed32_32, fixed48_16, ufixed32_32: carries across the 32-bit halves, negative
/// operands, truncation toward zero, mixed-width casts, members) built for wgpu and run on the device
/// prints what `boring run` prints. `int` is 32-bit on this target, so the example's 64-bit
/// `.raw` sums (which exceed 2^31) are replaced by an `int` expression that fits; the full 64-bit
/// raws are compared by `fixed_point_w64_mul_div_edge_sweep_matches_exact_math_on_wgpu`.
#[test]
fn fixed_point_w64_kernel_runs_on_wgpu_and_matches_interpreter() {
    let src = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/examples/fixed_point_kernel64.br")).unwrap();
    let old1 = "raws[i] = p[i].raw + q[i].floor().raw + (w[i].ceil() as int) + (a[i].trunc().raw) + (b[i].fract().raw)";
    let old2 = "raws[i] = raws[i] + (p[i].max(q[i])).raw";
    assert!(src.contains(old1) && src.contains(old2), "example changed, update this test");
    let src = src
        .replace(old1, "raws[i] = (p[i] as int) + (q[i].floor() as int) + ((w[i].fract() * 1000) as int) + (a[i].trunc() as int) + ((b[i].fract() * 1000) as int)")
        .replace(old2, "raws[i] = raws[i] + (p[i].max(q[i]) as int)");
    assert_wgpu_matches_interpreter("fixed_point64_run", &src);
}

/// `*`, `/` and `%` of 64-bit formats over an edge sweep (raws around 2^32, 2^62, 2^63-1, both signs, and
/// raws with the top bit set for the unsigned formats) match exact 128-bit arithmetic: truncation toward
/// zero, the final raw wrapping. Run on the real GPU, operands and results round-trip as full 64-bit raws.
#[test]
fn fixed_point_w64_mul_div_edge_sweep_matches_exact_math_on_wgpu() {
    fn program(u: &str) -> String {
        let formats: &[&str] = if u.is_empty() { &["fixed32_32", "fixed48_16", "fixed1_63"] }
            else { &["fixed32_32", "fixed48_16", "fixed1_63", "fixed0_64"] };
        let operands = if u.is_empty() { "u = -u" } else { "u = u + u + u\n            v = v + v + v" };
        let mut src = String::from("let vals = [1 as int64, 2, 3, 4294967295, 4294967296, 4294967297, 1234567890123456, 4611686018427387904, 9223372036854775807, 6148914691236517205, 3037000500, 281474976710655]\nlet V = 12\nlet N = 288\n");
        let mut prints = String::new();
        for (k, f) in formats.iter().enumerate() {
            let t = format!("{u}{f}");
            src.push_str(&format!(r#"
mut [@T@]'kernel'unified a@k@ = [@T@.fromRaw(0) for ..<N]
mut [@T@]'kernel'unified b@k@ = [@T@.fromRaw(0) for ..<N]
mut [@T@]'kernel'unified x@k@ = [@T@.fromRaw(0) for ..<N]
mut [@T@]'kernel'unified y@k@ = [@T@.fromRaw(0) for ..<N]
mut [@T@]'kernel'unified m@k@ = [@T@.fromRaw(0) for ..<N]
mut [@T@]'kernel'unified d@k@ = [@T@.fromRaw(0) for ..<N]
mut [@T@]'kernel'unified r@k@ = [@T@.fromRaw(0) for ..<N]
for i in 0..<V:
    for j in 0..<V:
        for h in 0..<2:
            a@k@[h * 144 + i * V + j] = @T@.fromRaw(vals[i])
            b@k@[h * 144 + i * V + j] = @T@.fromRaw(vals[j])

kernel K@k@:
    let [@T@]'unified a
    let [@T@]'unified b
    mut [@T@]'unified x
    mut [@T@]'unified y
    mut [@T@]'unified m
    mut [@T@]'unified d
    mut [@T@]'unified r

    init([@T@]'unified a0, [@T@]'unified b0, [@T@]'unified x0, [@T@]'unified y0, [@T@]'unified m0, [@T@]'unified d0, [@T@]'unified r0):
        a = a0
        b = b0
        x = x0
        y = y0
        m = m0
        d = d0
        r = r0

    def ():
        let i = .thread.x + .block.x * .block_dim.x
        var u = a[i]
        var v = b[i]
        if i >= 144:
            @OPS@
        x[i] = u
        y[i] = v
        m[i] = u * v
        d[i] = u / v
        r[i] = u % v

mut kk@k@ = K@k@(a@k@, b@k@, x@k@, y@k@, m@k@, d@k@, r@k@)
kernel:
    kk@k@(block = 144, grid = 2)
"#).replace("@T@", &t).replace("@k@", &k.to_string()).replace("@OPS@", operands));
            prints.push_str(&format!(
                "    print \"{k} {{kk{k}.x[i].raw}} {{kk{k}.y[i].raw}} {{kk{k}.m[i].raw}} {{kk{k}.d[i].raw}} {{kk{k}.r[i].raw}}\"\n"));
        }
        src.push_str("\nfor i in 0..<N:\n");
        src.push_str(&prints);
        src
    }
    for unsigned in [false, true] {
        let name = if unsigned { "fixed_w64_edge_u" } else { "fixed_w64_edge_s" };
        let (_wgsl, _e, _rs, _toml) = run_wgpu(name, &program(if unsigned { "u" } else { "" }));
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wgpu_codegen").join(name);
        let run = common::SharedCargo::new("wgpu_codegen")
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(dir.join("test_wgpu").join("Cargo.toml"))
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
        let out = String::from_utf8_lossy(&run.stdout).into_owned();
        assert!(run.status.success(), "generated wgpu program failed:\n{out}\n{}", String::from_utf8_lossy(&run.stderr));
        let mut rows = 0;
        for line in out.lines() {
            let v: Vec<i64> = line.split_whitespace().map(|t| t.parse::<i128>().unwrap() as i64).collect();
            if v.len() != 6 { continue; }
            let m = [32u32, 16, 63, 64][v[0] as usize];
            let (a, b) = (v[1], v[2]);
            let want: [i64; 3] = if unsigned {
                let (ua, ub) = (a as u64 as u128, b as u64 as u128);
                [((ua * ub) >> m) as u64 as i64, ((ua << m) / ub) as u64 as i64, (ua % ub) as u64 as i64]
            } else {
                [((a as i128 * b as i128) / (1i128 << m)) as i64, (((a as i128) << m) / b as i128) as i64, (a as i128 % b as i128) as i64]
            };
            assert_eq!(&v[3..], &want[..], "unsigned={unsigned} m={m} a={a} b={b}");
            rows += 1;
        }
        assert_eq!(rows, 288 * if unsigned { 4 } else { 3 }, "unsigned={unsigned}\n{out}");
    }
}

/// 64-bit scalar parameters (two `u32` params slots), int / float / other-format casts in both
/// directions, comparisons, `fromRaw` and the min/max/epsilon members, on the device against `boring run`.
#[test]
fn fixed_point_w64_params_casts_and_compares_run_on_wgpu_and_match_interpreter() {
    assert_wgpu_matches_interpreter("fixed_w64_casts", r#"
let N = 8
mut [fixed32_32]'kernel'unified a = [0.0 as fixed32_32 for ..<N]
mut [fixed32_32]'kernel'unified o = [0.0 as fixed32_32 for ..<N]
mut [fixed16_16]'kernel'unified n = [0.0 as fixed16_16 for ..<N]
mut [ufixed32_32]'kernel'unified u = [0.0 as ufixed32_32 for ..<N]
mut [ufixed8_8]'kernel'unified t = [0.0 as ufixed8_8 for ..<N]
mut [int32]'kernel'unified ints = [0 for ..<N]
mut [float32]'kernel'unified flts = [0.0 as float32 for ..<N]
mut [int32]'kernel'unified cmp = [0 for ..<N]
for i in 0..<N:
    a[i] = ((i as fixed32_32) - 3) * 1.75 + 0.0009765625

kernel C:
    let fixed32_32 k
    let ufixed32_32 ucst
    let [fixed32_32]'unified a
    mut [fixed32_32]'unified o
    mut [fixed16_16]'unified n
    mut [ufixed32_32]'unified u
    mut [ufixed8_8]'unified t
    mut [int32]'unified ints
    mut [float32]'unified flts
    mut [int32]'unified cmp

    init(fixed32_32 kval, ufixed32_32 uval, [fixed32_32]'unified a0, [fixed32_32]'unified o0, [fixed16_16]'unified n0, [ufixed32_32]'unified u0, [ufixed8_8]'unified t0, [int32]'unified i0, [float32]'unified f0, [int32]'unified c0):
        k = kval
        ucst = uval
        a = a0
        o = o0
        n = n0
        u = u0
        t = t0
        ints = i0
        flts = f0
        cmp = c0

    def ():
        let i = .thread.x
        o[i] = a[i] * k + (i as fixed32_32) - fixed32_32.fromRaw(5) + fixed32_32.epsilon
        n[i] = a[i] as fixed16_16
        u[i] = ((n[i].abs() as ufixed32_32) + ucst) * 0.5
        t[i] = u[i] as ufixed8_8
        ints[i] = (a[i] as int32) + (o[i] as int32) * 2
        flts[i] = (a[i] as float32) * 0.25 + ((i as float32) as fixed32_32) as float32
        var c = 0
        if a[i] < k:
            c = c + 1
        if a[i] >= k:
            c = c + 2
        if a[i] == k:
            c = c + 4
        if a[i] != k:
            c = c + 8
        if a[i].max(k) > fixed32_32.fromRaw(0):
            c = c + 16
        if ucst <= u[i]:
            c = c + 32
        cmp[i] = c

mut kk = C(2.5 as fixed32_32, 3.25 as ufixed32_32, a, o, n, u, t, ints, flts, cmp)
kernel:
    kk(block = N, grid = 1)

for i in 0..<N:
    print "{kk.a[i]} {kk.o[i]} {kk.n[i]} {kk.u[i]} {kk.t[i]} {kk.ints[i]} {kk.flts[i]} {kk.cmp[i]}"
"#);
}

/// A `[T, N]'const` array of a 64-bit format lives in the params block as `2N` u32 words and is read
/// back as a limb pair; on the device it matches `boring run` (negative and above-2^32 raws included).
#[test]
fn fixed_point_w64_const_array_runs_on_wgpu_and_matches_interpreter() {
    assert_wgpu_matches_interpreter("fixed_w64_const_array", r#"
let N = 4
mut [fixed32_32]'kernel'unified out = [0.0 as fixed32_32 for ..<N]
mut [ufixed32_32]'kernel'unified uout = [0.0 as ufixed32_32 for ..<N]

kernel W:
    let [fixed32_32, 4]'const w
    let [ufixed32_32, 4]'const uw
    mut [fixed32_32]'unified out
    mut [ufixed32_32]'unified uout
    init([fixed32_32]'unified o, [ufixed32_32]'unified uo):
        out = o
        uout = uo
        w = [1.5, -2.25, 40000.125, -0.0009765625]
        uw = [3.5, 0.25, 4000000000.5, 1.0]
    def ():
        let i = .thread.x
        out[i] = w[i] * w[(i + 1) % 4] - w[(i + 2) % 4]
        uout[i] = uw[i] / uw[(i + 1) % 4] + uw[3 - i]

mut k = W(out, uout)
kernel:
    k(block = N, grid = 1)

for i in 0..<N:
    print "{k.out[i]} {k.uout[i]}"
"#);
}

/// Labeled `'const` arrays (`[T, width = W, height = H]`, stored flat) of a 64-bit format, and the
/// fraction-only `ufixed0_64` (`M = 64`) with its members and casts, on the device against `boring run`.
#[test]
fn fixed_point_w64_labeled_const_array_and_ufixed0_64_run_on_wgpu_and_match_interpreter() {
    assert_wgpu_matches_interpreter("fixed_w64_labeled_m64", r#"
let N = 4
mut [fixed32_32]'kernel'unified out = [0.0 as fixed32_32 for ..<N]
mut [ufixed0_64]'kernel'unified frac = [0.0 as ufixed0_64 for ..<N]
mut [ufixed0_64]'kernel'unified frac2 = [0.0 as ufixed0_64 for ..<N]
mut [ufixed32_32]'kernel'unified wide = [0.0 as ufixed32_32 for ..<N]
mut [int32]'kernel'unified cmp = [0 for ..<N]

kernel W:
    let [fixed32_32, width = 2, height = 2]'const w
    mut [fixed32_32]'unified out
    mut [ufixed0_64]'unified frac
    mut [ufixed0_64]'unified frac2
    mut [ufixed32_32]'unified wide
    mut [int32]'unified cmp
    init([fixed32_32]'unified o, [ufixed0_64]'unified f, [ufixed0_64]'unified f2, [ufixed32_32]'unified wd, [int32]'unified c):
        out = o
        frac = f
        frac2 = f2
        wide = wd
        cmp = c
        w = [1.5, -2.25, 40000.125, -0.0009765625]
    def ():
        let i = .thread.x
        out[i] = w[i] * w[(i + 1) % 4] - w[(i + 2) % 4]
        let ufixed0_64 a = 0.3
        var ufixed0_64 b = 0.7
        if i == 1:
            b = 0.6
        elif i == 2:
            b = 0.5
        elif i == 3:
            b = 0.4
        frac[i] = a * b + a / b + b % a
        let ufixed0_64 mx = ufixed0_64.max
        let ufixed0_64 q = b.min(mx) * 0.25
        frac2[i] = a.fract() + a.floor() + a.trunc() + ufixed0_64.epsilon + q
        wide[i] = (b as ufixed32_32) + (a as ufixed32_32) * 3
        var c = 0
        if a < b:
            c = c + 1
        if a >= b:
            c = c + 2
        cmp[i] = c

mut k = W(out, frac, frac2, wide, cmp)
kernel:
    k(block = N, grid = 1)

for i in 0..<N:
    print "{k.out[i]} {k.frac[i]} {k.frac2[i]} {k.wide[i]} {k.cmp[i]}"
"#);
}

/// `clamp`, `round`, `signum` and `wrapping{Add,Sub,Mul,Div}` over formats of 8..64 bits (signed and
/// unsigned) on the device against `boring run`: raws near the format's extremes, overflowing wraps.
#[test]
fn fixed_point_members_clamp_round_signum_wrapping_run_on_wgpu_and_match_interpreter() {
    fn raws(t: &str, f: impl Fn(i64) -> i128) -> String {
        let _ = t;
        (0..16).map(|i| f(i).to_string()).collect::<Vec<_>>().join(", ")
    }
    for t in ["fixed16_16", "ufixed16_16", "fixed8_8", "ufixed8_8", "fixed4_4", "fixed4_28",
              "fixed32_32", "ufixed32_32", "fixed48_16", "fixed16_48", "ufixed48_16"] {
        let signed = !t.starts_with('u');
        let digits: Vec<u32> = t.trim_start_matches('u').trim_start_matches("fixed").split('_').map(|x| x.parse().unwrap()).collect();
        let w = digits[0] + digits[1];
        // Largest raw, capped to what the (signed 64-bit) host literals can hold.
        let max: i128 = if signed { (1i128 << (w - 1)) - 1 } else { ((1i128 << w) - 1).min(i64::MAX as i128) };
        let k = max / if signed { 9 } else { 17 };
        let (ra, rl, rh, rd) = if signed {
            (raws(t, |i| (i as i128 - 8) * k), raws(t, |i| (((i * 7 + 3) % 16) as i128 - 8) * k),
             raws(t, |i| (((i * 7 + 3) % 16) as i128 - 8) * k + k), raws(t, |i| (i as i128 % 5 + 1) * (k / 2 + 1)))
        } else {
            (raws(t, |i| (i as i128 + 1) * k), raws(t, |i| ((i * 7 + 3) % 16) as i128 * k),
             raws(t, |i| ((i * 7 + 3) % 16) as i128 * k + k), raws(t, |i| (i as i128 % 5 + 1) * (k / 2 + 1)))
        };
        let m = digits[1];
        // Small operands for the rounding variants (|a2| <= 2, 0.5 <= |s2| <= 3.5): no overflow on the CPU.
        let one = 1i128 << m;
        let (rb, rs) = if signed {
            (raws(t, |i| (i as i128 - 8) * one / 4), raws(t, |i| ((i as i128 % 7) + 1) * (one / 2) * if i % 2 == 0 { 1 } else { -1 }))
        } else {
            (raws(t, |i| (i as i128 + 1) * one / 8), raws(t, |i| ((i as i128 % 7) + 1) * (one / 2)))
        };
        let names = ["r", "s", "c", "wa", "ws", "wm", "wd", "sa", "ss", "sm", "sd", "mr", "dr"];
        let ins = ["a", "lo", "hi", "dv", "a2", "s2"];
        let decls: String = ins.iter().chain(names.iter())
            .map(|n| format!("mut [@T@]'kernel'unified {n} = [@T@.fromRaw(0) for ..<N]\n")).collect();
        let fields: String = ins.iter().map(|n| format!("    let [@T@]'unified {n}\n"))
            .chain(names.iter().map(|n| format!("    mut [@T@]'unified {n}\n"))).collect();
        let params = ins.iter().map(|n| format!("[@T@]'unified {n}0"))
            .chain(names.iter().map(|n| format!("[@T@]'unified {n}0"))).collect::<Vec<_>>().join(", ");
        let assigns: String = ins.iter().chain(names.iter()).map(|n| format!("        {n} = {n}0\n")).collect();
        let args = ins.iter().chain(names.iter()).cloned().collect::<Vec<_>>().join(", ");
        let prints = format!("{{k.a[i].raw}} {}", names.iter().map(|n| format!("{{k.{n}[i].raw}}")).collect::<Vec<_>>().join(" "));
        let src = format!(r#"
let N = 16
let RA = [0 as int64, {ra}]
let RL = [0 as int64, {rl}]
let RH = [0 as int64, {rh}]
let RD = [0 as int64, {rd}]
let RB = [0 as int64, {rb}]
let RS = [0 as int64, {rs}]
{decls}
for i in 0..<N:
    a[i] = @T@.fromRaw(RA[i + 1])
    lo[i] = @T@.fromRaw(RL[i + 1])
    hi[i] = @T@.fromRaw(RH[i + 1])
    dv[i] = @T@.fromRaw(RD[i + 1])
    a2[i] = @T@.fromRaw(RB[i + 1])
    s2[i] = @T@.fromRaw(RS[i + 1])

kernel K:
{fields}
    init({params}):
{assigns}
    def ():
        let i = .thread.x
        r[i] = a[i].round()
        s[i] = a[i].signum()
        c[i] = a[i].clamp(lo[i], hi[i])
        wa[i] = a[i].wrappingAdd(lo[i])
        ws[i] = a[i].wrappingSub(lo[i])
        wm[i] = a[i].wrappingMul(lo[i])
        wd[i] = a[i].wrappingDiv(dv[i])
        sa[i] = a[i].saturatingAdd(lo[i])
        ss[i] = a[i].saturatingSub(lo[i])
        sm[i] = a[i].saturatingMul(lo[i])
        sd[i] = a[i].saturatingDiv(dv[i])
        mr[i] = a2[i].mulRound(s2[i])
        dr[i] = a2[i].divRound(s2[i])

mut k = K({args})
kernel:
    k(block = N, grid = 1)

for i in 0..<N:
    print "{prints}"
"#).replace("@T@", t);
        assert_wgpu_matches_interpreter(&format!("fixed_members_{t}"), &src);
    }
}

/// Runs `out[i] = a[i].<method>()` over 16 raws spread across the format's range (and a few tiny ones)
/// on the device and compares with `boring run`.
fn assert_math_member_matches_interpreter(t: &str, method: &str, positive_only: bool) {
    let signed = !t.starts_with('u');
    let digits: Vec<u32> = t.trim_start_matches('u').trim_start_matches("fixed").split('_').map(|x| x.parse().unwrap()).collect();
    let w = digits[0] + digits[1];
    let max: i128 = if signed { (1i128 << (w - 1)) - 1 } else { ((1i128 << w) - 1).min(i64::MAX as i128) };
    let k = max / 17;
    let raws: Vec<i128> = (0..16i128).map(|i| match i {
        0 => 0, 1 => 1, 2 => 3,
        _ if signed && !positive_only => (i - 9) * k,
        _ => i * k,
    }).collect();
    assert_math_member_with_raws(t, method, &raws);
}

/// Runs `<t>.<method>()` on the device over exactly `raws` and compares with `boring run`, raw for raw.
fn assert_math_member_with_raws(t: &str, method: &str, raws: &[i128]) {
    let raws: Vec<String> = raws.iter().map(|v| v.to_string()).collect();
    let n = raws.len();
    let src = format!(r#"
let N = {n}
let RA = [0 as int64, {raws}]
mut [@T@]'kernel'unified a = [@T@.fromRaw(0) for ..<N]
mut [@T@]'kernel'unified r = [@T@.fromRaw(0) for ..<N]
for i in 0..<N:
    a[i] = @T@.fromRaw(RA[i + 1])

kernel K:
    let [@T@]'unified a
    mut [@T@]'unified r
    init([@T@]'unified a0, [@T@]'unified r0):
        a = a0
        r = r0
    def ():
        let i = .thread.x
        r[i] = a[i].@M@()

mut k = K(a, r)
kernel:
    k(block = N, grid = 1)

for i in 0..<N:
    print "{{k.a[i].raw}} {{k.r[i].raw}}"
"#, raws = raws.join(", ")).replace("@T@", t).replace("@M@", method);
    assert_wgpu_matches_interpreter(&format!("fixed_{method}_{t}"), &src);
}

/// Raws of `fixed<N>_<M>` / `ufixed<N>_<M>` for the real values in `xs` whose `f(x)` stays well inside the format's
/// range (so the interpreter neither overflows nor rejects a negative result for an unsigned format).
fn raws_where_result_fits(t: &str, xs: &[f64], f: impl Fn(f64) -> f64) -> Vec<i128> {
    let signed = !t.starts_with('u');
    let d: Vec<i32> = t.trim_start_matches('u').trim_start_matches("fixed").split('_').map(|x| x.parse().unwrap()).collect();
    let (w, m) = (d[0] + d[1], d[1]);
    let max: i128 = if signed { (1i128 << (w - 1)) - 1 } else { ((1i128 << w) - 1).min(i64::MAX as i128) };
    let lim = 2f64.powi(w - m - if signed { 1 } else { 0 }) * 0.9;
    let mut out: Vec<i128> = xs.iter().filter_map(|&x| {
        let raw = (x * 2f64.powi(m)).round() as i128;
        let r = f(x);
        let ok = raw <= max && (if signed { raw >= -max } else { raw >= 0 })
            && r.is_finite() && r.abs() < lim && (signed || r >= 0.0);
        ok.then_some(raw)
    }).collect();
    out.dedup();
    out
}

#[test]
fn fixed_point_sqrt_runs_on_wgpu_and_matches_interpreter() {
    for t in ["fixed16_16", "ufixed16_16", "fixed8_8", "ufixed8_8", "fixed4_4", "fixed4_28", "ufixed0_32",
              "fixed32_32", "ufixed32_32", "fixed48_16", "fixed16_48", "ufixed48_16", "ufixed0_64"] {
        assert_math_member_matches_interpreter(t, "sqrt", true);
    }
}

/// `sin` / `cos` (the interpreter's 60-bit CORDIC, bit for bit) over raws spread across each format's
/// range, so the 128-bit quadrant reduction is exercised, on the device against `boring run`.
#[test]
fn fixed_point_sin_cos_run_on_wgpu_and_match_interpreter() {
    for t in ["fixed16_16", "fixed8_8", "fixed4_28", "fixed2_30", "fixed32_32", "fixed48_16", "fixed16_48"] {
        for m in ["sin", "cos"] {
            assert_math_member_matches_interpreter(t, m, false);
        }
    }
    // `M > 60` (the argument is shifted right into the 60-bit CORDIC domain): inputs below 1, sin only.
    for t in ["ufixed0_32", "ufixed0_64"] {
        assert_math_member_matches_interpreter(t, "sin", true);
    }
}

/// `tan`, `exp` and `ln` (the interpreter's 128-bit-intermediate algorithms, bit for bit) over inputs chosen so the
/// result fits each format, on the device against `boring run`. Covers the normalisation of `ln` (tiny and huge
/// arguments), the `exp` range reduction (negative and positive, underflow to 0), `tan` near its poles, and `M > 60`.
#[test]
fn fixed_point_tan_exp_ln_run_on_wgpu_and_match_interpreter() {
    let ln_xs = [1e-6, 0.001, 0.0625, 0.1, 0.5, 0.75, 1.0, 1.0000001, 1.5, 2.0, 2.718281828, 3.0, 7.0, 10.0, 100.0,
                 1000.0, 12345.678, 65535.5, 1e9, 1e12];
    let exp_xs = [-50.0, -43.5, -20.0, -10.0, -5.0, -2.5, -1.0, -0.5, -0.001, 0.0, 0.001, 0.25, 0.5, 0.6931471805599453,
                  1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0];
    let tan_xs = [-100.0, -4.0, -3.0, -1.5, -1.2, -1.0, -0.5, -0.1, 0.0, 0.0001, 0.1, 0.5, 0.7, 1.0, 1.2, 1.5, 1.5707,
                  2.0, 3.0, 3.14159, 4.0, 10.0, 100.0];
    for t in ["fixed16_16", "fixed8_8", "fixed4_28", "fixed32_32", "fixed48_16", "fixed16_48", "fixed2_62",
              "ufixed16_16", "ufixed32_32", "ufixed48_16"] {
        let mut ln = raws_where_result_fits(t, &ln_xs, f64::ln);
        ln.retain(|&r| r > 0); // x rounded to a zero raw has no logarithm
        let exp = raws_where_result_fits(t, &exp_xs, f64::exp);
        let tan = raws_where_result_fits(t, &tan_xs, f64::tan);
        for (m, raws) in [("ln", ln), ("exp", exp), ("tan", tan)] {
            if raws.len() >= 3 { assert_math_member_with_raws(t, m, &raws); }
        }
    }
    // `M > 60` unsigned: the argument is shifted right into the 60-bit domain; `tan` only (exp >= 1, ln < 0 do not fit).
    for t in ["ufixed0_32", "ufixed0_64"] {
        let tan = raws_where_result_fits(t, &[0.0, 0.01, 0.1, 0.25, 0.5, 0.7], f64::tan);
        assert_math_member_with_raws(t, "tan", &tan);
    }
}

/// Builds `src` for wgpu, runs the generated program on the real device and returns its stdout.
fn run_wgpu_program(name: &str, src: &str) -> (String, String, String) {
    let (wgsl, _emulated, rs, _toml) = run_wgpu(name, src);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wgpu_codegen").join(name);
    let run = common::SharedCargo::new("wgpu_codegen")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(dir.join("test_wgpu").join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    assert!(run.status.success(), "generated wgpu program failed:
{}
{}",
        String::from_utf8_lossy(&run.stdout), String::from_utf8_lossy(&run.stderr));
    (String::from_utf8_lossy(&run.stdout).trim_end().to_string(), wgsl, rs)
}

/// A generic kernel instantiated inside a `pub req` returning a resident array (two
/// instantiations, the first behind an `if` with an explicit `return k0.y`, the second a tail
/// `k1.y`), whose body calls free functions taking a `'global` array parameter (also through a
/// forwarding chain `dec` -> `dec_a`). Used to fail on wgpu: no struct was emitted for the
/// instantiation, the launch was emitted as a tail `Ok(..)`, and the shared `w` parameter was
/// bound to two different buffers.
#[test]
fn generic_kernel_instantiated_in_req_runs_on_wgpu() {
    let src = r#"# D: named tiled kernel (labeled 'actor tiles) + free decode fns with array params, selected by a runtime fmt field
float32 dec_a([uint8]'global w, int idx):
    (int(w[idx]) as float32) * 0.5

float32 dec_b([uint8]'global w, int idx):
    let v = int(w[idx * 2]) | (int(w[idx * 2 + 1]) << 8)
    v as float32

float32 dec([uint8]'global w, int fmt, int idx):
    if fmt == 0:
        dec_a(w, idx)
    else:
        dec_b(w, idx)

kernel TiledKernel<int FMT>:
    let [float32]'global x
    let [uint8]'global w
    mut [float32]'unified y
    let int seq
    let int d_in
    let int d_out
    mut [float32, width = 16, height = 16]'actor tile_x
    mut [float32, width = 16, height = 16]'actor tile_w

    init([float32]'global xi, [uint8]'global wi, int s, int din, int dout):
        x = xi
        w = wi
        seq = s
        d_in = din
        d_out = dout
        y = [0.0 for ..<s * dout]

    def ():
        let tx = .thread.x
        let ty = .thread.y
        let row = .block.y * .block_dim.y + ty
        let col = .block.x * .block_dim.x + tx
        var float32 acc = 0.0
        let num_tiles = (d_in + 15) / 16
        for t in 0..<num_tiles:
            let x_col = t * 16 + tx
            let w_row = t * 16 + ty
            if row < seq and x_col < d_in:
                tile_x[width = tx, height = ty] = x[row * d_in + x_col]
            else:
                tile_x[width = tx, height = ty] = 0.0
            if w_row < d_in and col < d_out:
                tile_w[width = tx, height = ty] = dec(w, FMT, col * d_in + w_row)
            else:
                tile_w[width = tx, height = ty] = 0.0
            sync
            for kk in 0..<16:
                acc += tile_x[width = kk, height = ty] * tile_w[width = tx, height = kk]
            sync
        if row < seq and col < d_out:
            y[row * d_out + col] = acc

pub req [float32]'kernel'unified run_e([float32]'kernel'global x, [uint8]'kernel'global w, int seq, int d_in, int d_out, int fmt) throws:
    if fmt == 0:
        mut k0 = TiledKernel<0>(x, w, seq, d_in, d_out)
        kernel:
            k0(block = (16, 16), grid = ((d_out + 15) / 16, (seq + 15) / 16))
        return k0.y
    guard fmt == 1 else throw "bad fmt {fmt}"
    mut k1 = TiledKernel<1>(x, w, seq, d_in, d_out)
    kernel:
        k1(block = (16, 16), grid = ((d_out + 15) / 16, (seq + 15) / 16))
    k1.y

def main() throws:
    let [float32] x = [1.0 as float32 for ..<4]
    let [uint8] w = [2 as uint8 for ..<16]
    let [float32] r0 = run_e(x, w, 1, 4, 2, 0)
    let [float32] r1 = run_e(x, w, 1, 4, 2, 1)
    print "fmt0 {r0[0]}  fmt1 {r1[0]}"
"#;
    let (out, wgsl, rs) = run_wgpu_program("generic_kernel_in_req", src);
    // fmt 0: dec_a = 2 * 0.5 = 1.0, summed over d_in = 4; fmt 1: dec_b = 2 | 2 << 8 = 514, * 4.
    assert_eq!(out, "fmt0 4  fmt1 2056");
    assert!(rs.contains("struct TiledKernel_0 {") && rs.contains("struct TiledKernel_1 {"), "{rs}");
    assert!(!rs.contains("TiledKernel::<"), "{rs}");
    assert!(rs.contains("return Ok(BoringKernelArg::Resident("), "explicit resident return: {rs}");
    // One clone of each free function per bound buffer, array parameter dropped.
    assert!(wgsl.contains("fn dec__tiledkernel_0_w(") && wgsl.contains("fn dec__tiledkernel_1_w("), "{wgsl}");
    assert!(wgsl.contains("fn dec_a__tiledkernel_0_w(") && wgsl.contains("fn dec_b__tiledkernel_1_w("), "{wgsl}");
}

/// Const-generic array sizes (`[float32, D / 16]'actor`, `width = D + 1`) and the literal
/// `D` itself, per instantiation, from a `req`.
#[test]
fn generic_kernel_array_sizes_depend_on_const_param_on_wgpu() {
    let src = r#"kernel SumKernel<int D>:
    let [float32]'global x
    mut [float32]'unified y
    mut [float32, D / 16]'actor part
    mut [float32, width = D + 1, height = 2]'actor buf

    init([float32]'global xi):
        x = xi
        y = [0.0 for ..<2]

    def ():
        let t = .thread.x
        buf[width = t, height = 0] = x[t]
        sync
        if t < D / 16:
            var float32 s = 0.0
            for k in 0..<16:
                s += buf[width = t * 16 + k, height = 0]
            part[t] = s
        sync
        if t == 0:
            var float32 total = 0.0
            for i in 0..<D / 16:
                total += part[i]
            y[0] = total
            y[1] = D as float32

pub req [float32]'kernel'unified run_sum([float32]'kernel'global x, int which) throws:
    if which == 0:
        mut k0 = SumKernel<32>(x)
        kernel:
            k0(block = 32, grid = 1)
        return k0.y
    mut k1 = SumKernel<64>(x)
    kernel:
        k1(block = 64, grid = 1)
    k1.y

def main() throws:
    let [float32] x = [1.0 as float32 for ..<64]
    let [float32] a = run_sum(x, 0)
    let [float32] b = run_sum(x, 1)
    print "{a[0]} {a[1]} {b[0]} {b[1]}"
"#;
    let (out, wgsl, _rs) = run_wgpu_program("generic_kernel_array_sizes", src);
    assert_eq!(out, "32 32 64 64");
    assert!(wgsl.contains("array<f32, 2>") && wgsl.contains("array<f32, 4>"), "{wgsl}");
}

/// `'kernel'global`/`'kernel'unified` is residency, not ownership: a parameter that is only read or
/// forwarded to a kernel / another GPU function is `&BoringKernelArg<T>` and its call sites lend
/// (`f(&h.a)`) instead of deep-copying the `Host` variant; one that is returned stays by value.
#[test]
fn gpu_array_params_are_borrowed_not_cloned() {
    let src = r#"
struct Holder:
    [float32]'kernel'unified a
    [float32]'kernel'unified b

pub req [float32]'kernel'unified double([float32]'kernel'global x) throws:
    let int n = x.length
    kernel(block = 32, grid = 1):
        mut [float32]'unified out = [0.0 as float32 for ..<n]
        let i = .thread.x
        if i < n:
            out[i] = x[i] * 2.0
        out

pub req [float32]'kernel'unified twice([float32]'kernel'global x) throws:
    let y = double(x)
    double(y)

pub req [float32]'kernel'unified same([float32]'kernel'global x) throws:
    x

def main() throws:
    let h = Holder([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    let r1 = double(h.a)
    let r2 = double(h.a)
    let r3 = twice(h.b)
    let r4 = same(h.a)
    print "done"
"#;
    let (_, rs) = wgpu_codegen("gpu_array_params_are_borrowed_not_cloned", src);
    assert!(rs.contains("pub fn double(x: &BoringKernelArg<f32>)"), "{rs}");
    assert!(rs.contains("pub fn twice(x: &BoringKernelArg<f32>)"), "{rs}");
    assert!(rs.contains("pub fn same(x: BoringKernelArg<f32>)"), "{rs}");
    assert!(rs.contains("double(&h.a)?"), "{rs}");
    assert!(rs.contains("twice(&h.b)?"), "{rs}");
    assert!(rs.contains("let y = double(x)?;"), "{rs}");
    assert!(rs.contains("double(&y)?"), "{rs}");
    assert!(rs.contains("same((h.a).clone())?"), "{rs}");
    assert!(!rs.contains("double((h.a).clone())"), "{rs}");
    let manifest = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("wgpu_codegen").join("gpu_array_params_are_borrowed_not_cloned").join("test_wgpu").join("Cargo.toml");
    let check = common::SharedCargo::new("wgpu_codegen")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {e}"));
    assert!(check.status.success(), "generated Rust must type-check:\n{}\n{rs}", String::from_utf8_lossy(&check.stderr));
}
