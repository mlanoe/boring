use std::{fs, path::PathBuf, process::Command};

#[test]
fn fixed_float16_tensor_operations_run_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_tensor_cpu");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float16, k = 2, m = 2]'kernel'global a = [float16(1.0), float16(2.0), float16(3.0), float16(4.0)]
let [float16, n = 2, k = 2]'kernel'global b = [float16(5.0), float16(6.0), float16(7.0), float16(8.0)]
let [float16, k = 2, n = 2]'kernel'global w = [float16(5.0), float16(7.0), float16(6.0), float16(8.0)]
mut [float16, n = 2, m = 2]'kernel'unified c = [float16(0.0) for ..<4]
mut [float16, n = 2, m = 2]'kernel'unified accum = [float16(0.5) for ..<4]
mut [float16, n = 2, m = 2]'kernel'unified linear = [float16(0.0) for ..<4]
kernel.tensors.matmul(a, b, c)
kernel.tensors.mma(a, b, accum)
kernel.tensors.linear(a, w, linear)
print c
print accum
print linear
"#,
    )
    .unwrap();

    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        interpreted.status.success(),
        "{}",
        String::from_utf8_lossy(&interpreted.stderr)
    );
    let expected = "[19, 22, 43, 50]\n[19.5, 22.5, 43.5, 50.5]\n[19, 22, 43, 50]";
    assert_eq!(
        String::from_utf8_lossy(&interpreted.stdout).trim(),
        expected
    );

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

    let project = root.join("tensor_simd");
    let source = fs::read_to_string(project.join("src/main.rs")).unwrap();
    assert!(source.contains("f32::from(a["));
    assert!(source.contains("half::f16::from_f32(__boring_tensor_sum)"));

    let run = Command::new("cargo")
        .args(["run", "--offline", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), expected);
}

#[test]
fn dynamic_float16_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_dynamic_tensor_cpu");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("dynamic.br");
    fs::write(
        &input,
        r#"
let [float16]'kernel'global x = [float16(1.0), float16(2.0), float16(3.0), float16(4.0)]
let [float16]'kernel'global w = [float16(5.0), float16(7.0), float16(6.0), float16(8.0)]
let [float16]'kernel'global b = [float16(5.0), float16(6.0), float16(7.0), float16(8.0)]
let [float16]'kernel'global bias = [float16(0.5), float16(-0.5)]
mut [float16]'kernel'unified y = [float16(0.0) for ..<4]
mut [float16]'kernel'unified product = [float16(0.0) for ..<4]
mut [float16]'kernel'unified accumulated = [float16(0.5) for ..<4]
kernel.tensors.linear(x, w, bias, y, m = 2, n = 2, k = 2)
kernel.tensors.matmul(x, b, product, m = 2, n = 2, k = 2)
kernel.tensors.mma(x, b, accumulated, m = 2, n = 2, k = 2)
print y
print product
print accumulated
"#,
    )
    .unwrap();
    let expected = "[19.5, 21.5, 43.5, 49.5]\n[19, 22, 43, 50]\n[19.5, 22.5, 43.5, 50.5]";

    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(&input)
        .output()
        .unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), expected);

    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let project = root.join("dynamic_simd");
    let run = Command::new("cargo")
        .args(["run", "--offline", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), expected);
}

#[test]
fn dynamic_float16_linear_generates_all_gpu_backends() {
    let source_text = r#"
pub req [float16]'kernel'unified compute([float16]'kernel'global x, [float16]'kernel'global w, [float16]'kernel'global bias, int m, int n, int k) throws:
    mut [float16]'kernel'unified y = [float16(0.0) for ..<m * n]
    kernel.tensors.linear(x, w, bias, y, m = m, n = n, k = k)
    y
"#;
    for target in ["wgpu", "metal", "cuda", "rocm"] {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("float16_dynamic_tensor_gpu")
            .join(target);
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let input = root.join("dynamic.br");
        fs::write(&input, source_text).unwrap();
        let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target])
            .arg(&input)
            .output()
            .unwrap();
        assert!(generated.status.success(), "{target}: {}", String::from_utf8_lossy(&generated.stderr));
        let project = root.join(format!("dynamic_{target}"));
        let device_path = match target {
            "wgpu" => project.join("shaders/main.wgsl"),
            "metal" => project.join("kernels/main.metal"),
            "cuda" => project.join("kernels/main.cu"),
            "rocm" => project.join("kernels/main.hip"),
            _ => unreachable!(),
        };
        let device = fs::read_to_string(device_path).unwrap();
        assert!(device.contains("BoringTensorHost"), "{target}: {device}");
        assert!(device.contains("f16") || device.contains("half") || device.contains("Float16"), "{target}: {device}");
    }
}

#[test]
fn fixed_float16_tensor_dispatch_generates_all_gpu_backends() {
    let source_text = r#"
pub req [float16, n = 2, m = 2]'kernel'unified compute([float16, k = 2, m = 2]'kernel'global x, [float16, k = 2, n = 2]'kernel'global w) throws:
    mut [float16, n = 2, m = 2]'kernel'unified y = [float16(0.0) for ..<4]
    kernel.tensors.linear(x, w, y)
    y
"#;
    for target in ["wgpu", "metal", "cuda", "rocm"] {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join("float16_fixed_tensor_gpu")
            .join(target);
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let input = root.join("fixed.br");
        fs::write(&input, source_text).unwrap();
        let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target])
            .arg(&input)
            .output()
            .unwrap();
        assert!(generated.status.success(), "{target}: {}", String::from_utf8_lossy(&generated.stderr));
        let project = root.join(format!("fixed_{target}"));
        let device_path = match target {
            "wgpu" => project.join("shaders/main.wgsl"),
            "metal" => project.join("kernels/main.metal"),
            "cuda" => project.join("kernels/main.cu"),
            "rocm" => project.join("kernels/main.hip"),
            _ => unreachable!(),
        };
        let device = fs::read_to_string(device_path).unwrap();
        assert!(device.contains("BoringTensorHost"), "{target}: {device}");
        assert!(device.contains("f16") || device.contains("half") || device.contains("Float16"), "{target}: {device}");
    }
}

#[test]
fn quantized_float16_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_quant_tensor_cpu");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("quant.br");
    let weights = std::iter::once("0".to_string())
        .chain(std::iter::once("60".to_string()))
        .chain((0..32).map(|_| "1".to_string()))
        .collect::<Vec<_>>()
        .join(", ");
    fs::write(&input, format!(r#"
let [float16]'kernel'global x = [float16(1.0) for ..<32]
let [uint8]'kernel'global weights = [{weights}]
let [float16]'kernel'global bias = [float16(0.5)]
mut [float16]'kernel'unified y = [float16(0.0)]
kernel.tensors.linear(x, weights, bias, y, m = 1, n = 1, k = 32, format = "q8_0")
print y
"#)).unwrap();
    let expected = "[32.5]";
    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&input).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), expected);
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"]).arg(&input).output().unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let project = root.join("quant_simd");
    let run = Command::new("cargo").args(["run", "--offline", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml")).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), expected);
}

#[test]
fn quantized_float16_linear_generates_all_gpu_backends() {
    let source_text = r#"
pub req [float16]'kernel'unified compute([float16]'kernel'global x, [uint8]'kernel'global weights, [float16]'kernel'global bias, int m, int n, int k) throws:
    mut [float16]'kernel'unified y = [float16(0.0) for ..<m * n]
    kernel.tensors.linear(x, weights, bias, y, m = m, n = n, k = k, format = "q8_0")
    y
"#;
    for target in ["wgpu", "metal", "cuda", "rocm"] {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_quant_tensor_gpu").join(target);
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let input = root.join("quant.br");
        fs::write(&input, source_text).unwrap();
        let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&input).output().unwrap();
        assert!(generated.status.success(), "{target}: {}", String::from_utf8_lossy(&generated.stderr));
        let project = root.join(format!("quant_{target}"));
        let device_path = match target {
            "wgpu" => project.join("shaders/main.wgsl"),
            "metal" => project.join("kernels/main.metal"),
            "cuda" => project.join("kernels/main.cu"),
            "rocm" => project.join("kernels/main.hip"),
            _ => unreachable!(),
        };
        let device = fs::read_to_string(device_path).unwrap();
        assert!(device.contains("BoringTensorHost"), "{target}: {device}");
        assert!(device.contains("f16") || device.contains("half") || device.contains("Float16"), "{target}: {device}");
    }
}
