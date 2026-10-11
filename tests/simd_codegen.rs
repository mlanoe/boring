use std::{fs, path::PathBuf, process::Command};

const SOURCE: &str = r#"
mut [float32]'kernel'unified x = [0.0 as float32 for ..<8]
mut [float32]'kernel'unified y = [1.0 as float32 for ..<8]
kernel Add:
    let [float32]'global x
    mut [float32]'unified y
    init([float32]'global a, [float32]'unified b):
        x = a
        y = b
    def ():
        let i = .thread.x + .block.x * .block_dim.x
        y[i] = x[i] + y[i]
mut k = Add(x, y)
kernel:
    k(block = 4, grid = 2)
print k.y[7]
"#;

#[test]
fn simd_project_builds_and_preserves_block_coordinates() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_codegen");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("main.br");
    fs::write(&input, SOURCE).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let generated = fs::read_to_string(root.join("main_simd/src/main.rs")).unwrap();
    assert!(generated.contains("is_x86_feature_detected!(\"avx2\")"));
    assert!(generated.contains("is_aarch64_feature_detected!(\"neon\")"));
    assert!(generated.contains("target_feature(enable=\"avx2\")"));
    assert!(generated.contains("BORING_SIMD_SCALAR"));
    assert!(generated.contains("for __x in 0..dx as usize"));
    assert!(!generated.contains("let __end ="));
    assert!(!generated.contains("__lanes"));
    assert!(generated.contains("dispatch((2) as u32, (1) as u32, (1) as u32, (4) as u32"));
    let check = Command::new("cargo")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(root.join("main_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    for scalar in [false, true] {
        let mut command = Command::new("cargo");
        command
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join("main_simd/Cargo.toml"));
        if scalar {
            command.env("BORING_SIMD_SCALAR", "1");
        }
        let run = command.output().unwrap();
        assert!(
            run.status.success(),
            "scalar={scalar}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&run.stdout).trim(),
            "1",
            "scalar={scalar}"
        );
    }
}

#[test]
fn multi_target_syntax_is_not_accepted_yet() {
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "wgpu,simd", "missing.br"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unknown target 'wgpu,simd'"));
}

#[test]
fn warp_lane_and_size_run_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_warp_properties");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("warp.br");
    fs::write(
        &input,
        "mut [int]'kernel'unified lanes = [0 for ..<8]\nmut [int]'kernel'unified sizes = [0 for ..<8]\nkernel W:\n    mut [int]'unified lanes\n    mut [int]'unified sizes\n    init(mut [int]'unified l, mut [int]'unified s):\n        lanes = l\n        sizes = s\n    def ():\n        lanes[.thread.x] = .warp.lane\n        sizes[.thread.x] = .warp.size\nmut w = W(lanes, sizes)\nkernel:\n    w(block = 8)\nprint w.lanes[7]\nprint w.sizes[7]\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let generated = fs::read_to_string(root.join("warp_simd/src/main.rs")).unwrap();
    assert!(generated.contains("rem_euclid(__boring_simd_lanes() as i32)"));
    for scalar in [false, true] {
        let mut command = Command::new("cargo");
        command
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join("warp_simd/Cargo.toml"));
        if scalar {
            command.env("BORING_SIMD_SCALAR", "1");
        }
        let run = command.output().unwrap();
        assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
        let values = String::from_utf8_lossy(&run.stdout)
            .lines()
            .map(|value| value.parse::<i32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values[0], 7 % values[1]);
        assert!(matches!(values[1], 1 | 4 | 8 | 16));
        if scalar {
            assert_eq!(values[1], 1);
        }
    }
}

#[test]
fn warp_collectives_fail_during_codegen() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_warp_error");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("warp.br");
    fs::write(&input, "kernel W:\n    def ():\n        let value = .warp.shuffle(.thread.x, 0)\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("`.warp` collectives are not yet supported"));
}

#[test]
fn block_shared_storage_fails_during_codegen() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_actor_error");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("actor.br");
    fs::write(&input, "kernel Shared:\n    mut [float32, 4]'actor tile\n    def ():\n        tile[.thread.x] = 1.0\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("storage, which is not yet supported"));
}

#[test]
fn fixed_float_tensor_matmul_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_tensor_matmul");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32, k = 5, m = 3]'kernel'global a = [1.0 as float32 for ..<15]
let [float32, n = 7, k = 5]'kernel'global b = [1.0 as float32 for ..<35]
let [float32, k = 5, n = 7]'kernel'global w = [1.0 as float32 for ..<35]
mut [float32, n = 7, m = 3]'kernel'unified c = [0.0 as float32 for ..<21]
mut [float32, n = 7, m = 3]'kernel'unified accum = [1.0 as float32 for ..<21]
mut [float32, n = 7, m = 3]'kernel'unified linear = [0.0 as float32 for ..<21]
kernel.tensors.matmul(a, b, c)
kernel.tensors.mma(a, b, accum)
kernel.tensors.linear(a, w, linear)
print c[0]
print accum[0]
print linear[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(source.contains("for __boring_tensor_k in 0usize..5usize"));
    assert!(source.contains("fn __boring_simd_threads()"));
    assert!(source.contains("__boring_simd_threads().min(3usize)"));
    assert!(source.contains(".chunks_mut(__boring_tensor_rows_per_worker * 7)"));
    assert!(source.contains("__boring_tensor_scope.spawn(move ||"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "5\n6\n5");
}

#[test]
fn gpu_resident_function_return_is_rejected() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_resident_return");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("resident.br");
    fs::write(
        &input,
        "req [float32]'kernel'unified values() throws:\n    [1.0 as float32]\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr)
        .contains("GPU-resident returns are not yet supported"));
}

#[test]
fn local_reassignment_is_not_treated_as_a_kernel_field() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_locals_grid");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("locals.br");
    fs::write(
        &input,
        r#"
kernel K:
    mut [int]'unified out
    init():
        out = [0 for ..<4]
    def ():
        var int value = 0
        value = value + 1
        let i = .block.x + .block.y * 2
        out[i] = value
mut k = K()
kernel:
    k(block = 1, grid = (2, 2))
print k.out[3]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("locals_simd/src/main.rs")).unwrap();
    assert!(source.contains("value = (value + 1)"));
    assert!(!source.contains("self.value"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("locals_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1");
}

#[test]
fn labeled_matrix_kernel_builds_and_runs() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_matrix_example");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("matrix.br");
    let example = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/matrix_mul_gpu.br");
    fs::copy(example, &input).unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("matrix_simd/src/main.rs")).unwrap();
    assert!(source.contains("let mut sum: f32"));
    assert!(source.contains("self.a[((i) as usize"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("matrix_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).starts_with("C[0][0] = 2.4424"));
}

#[test]
fn integer_bitwise_operators_preserve_their_semantics() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_bitwise");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("bitwise.br");
    fs::write(
        &input,
        r#"
kernel Bits:
    mut [int]'unified out
    init():
        out = [0 for ..<1]
    def ():
        let int a = (5 & 3) | (1 << 3)
        let int b = (a ^ 2) >> 1
        out[0] = b + (~0 & 1)
mut bits = Bits()
kernel:
    bits(block = 1)
print bits.out[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("bitwise_simd/src/main.rs")).unwrap();
    assert!(source.contains("(5 & 3)"));
    assert!(source.contains("(1 << 3)"));
    assert!(source.contains("(!0)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("bitwise_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "6");
}

#[test]
fn examples_with_top_level_scalar_constants_build_and_run() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for (example, expected_stdout) in [
        ("vector_add_gpu.br", "c[999] = 2997"),
        ("mandelbrot_gpu.br", "Image saved as 'mandelbrot.ppm'"),
    ] {
        let stem = example.trim_end_matches(".br");
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("simd_{stem}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let input = root.join(example);
        fs::copy(manifest_dir.join("examples").join(example), &input).unwrap();
        let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", "simd"])
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            generated.status.success(),
            "{example}: {}",
            String::from_utf8_lossy(&generated.stderr)
        );
        let run = Command::new("cargo")
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join(format!("{stem}_simd/Cargo.toml")))
            .current_dir(&root)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{example}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(
            String::from_utf8_lossy(&run.stdout).contains(expected_stdout),
            "{example}: {}",
            String::from_utf8_lossy(&run.stdout)
        );
    }
}
