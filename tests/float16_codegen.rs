use std::{fs, path::PathBuf, process::Command};

const SOURCE: &str = r#"
kernel HalfBuffer:
    mut [float16]'unified values
    init([float16]'unified input):
        values = input
    def ():
        let i = .thread.x
        let float32 wide = values[i] as float32
        values[i] = (wide + 1.0) as float16

pub req [float16]'kernel'unified increment([float16]'kernel'unified input) throws:
    mut k = HalfBuffer(input)
    kernel:
        k(block = input.length)
    k.values
"#;

fn generate(target: &str, case: &str) -> (String, String, String) {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen").join(case);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, SOURCE).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target]).arg(&source).output().unwrap();
    assert!(output.status.success(), "{target}: {}", String::from_utf8_lossy(&output.stderr));
    let generated = root.join(format!("main_{target}"));
    let host = fs::read_to_string(generated.join("src/main.rs")).unwrap();
    let manifest = fs::read_to_string(generated.join("Cargo.toml")).unwrap();
    let device = match target {
        "metal" => fs::read_to_string(generated.join("kernels/main.metal")).unwrap(),
        "cuda" => fs::read_to_string(generated.join("kernels/main.cu")).unwrap(),
        "rocm" => fs::read_to_string(generated.join("kernels/main.hip")).unwrap(),
        "wgpu" => fs::read_to_string(generated.join("shaders/main.wgsl")).unwrap(),
        _ => unreachable!(),
    };
    (device, host, manifest)
}

#[test]
fn host_half_is_two_bytes_and_roundtrips() {
    assert_eq!(std::mem::size_of::<half::f16>(), 2);
    let original = 1.337_f32;
    let compact = half::f16::from_f32(original);
    assert_eq!(compact.to_f32(), half::f16::from_f32(compact.to_f32()).to_f32());
    assert_ne!(compact.to_f32(), original);
}

#[test]
fn cpu_interpreter_and_generated_rust_convert_and_compute_f16() {
    let source_text = "def main():\n    let float16 a = float16(1.337)\n    let float16 b = float16(\"2.0\")\n    let float16 c = (a + b) as float16\n    let float16 root = sqrt(float16(4.0)) as float16\n    print \"{c as float32}\"\n    print \"{root as float32}\"\n";
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/cpu");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, source_text).unwrap();
    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["run"]).arg(&source).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), "3.3359375\n2");
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build"]).arg(&source).output().unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let manifest = root.join("main_rust/Cargo.toml");
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"]).arg(manifest).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
}

#[test]
fn cpu_float16_newtype_compiles_with_float_derives() {
    let source_text = "type Half as float16\n\ndef main():\n    let Half x = Half(float16(1.5))\n    print x\n";
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/cpu_newtype");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, source_text).unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build"]).arg(&source).output().unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"])
        .arg(root.join("main_rust/Cargo.toml")).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
}

#[test]
fn supported_gpu_backends_keep_f16_storage() {
    for (target, spelling) in [("metal", "half*"), ("cuda", "__half*"), ("rocm", "_Float16*")] {
        let (device, host, manifest) = generate(target, target);
        assert!(device.contains(spelling), "{target} widened the buffer:\n{device}");
        assert!(host.contains("half::f16"), "{target} host ABI is not f16:\n{host}");
        assert!(manifest.contains("half ="), "{target} manifest lacks half dependency");
    }
}

#[test]
fn wgpu_reports_float16_as_unsupported() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/wgpu_error");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, SOURCE).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "wgpu"]).arg(&source).output().unwrap();
    assert!(!output.status.success(), "wgpu unexpectedly accepted float16");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("does not enable WGSL's optional `f16` feature"), "{stderr}");
}

#[test]
fn simd_kernel_uses_and_compiles_real_f16_storage() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/simd");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, SOURCE).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"]).arg(&source).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let generated = root.join("main_simd");
    let host = fs::read_to_string(generated.join("src/main.rs")).unwrap();
    assert!(host.contains("values: Vec<half::f16>"), "{host}");
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"])
        .arg(generated.join("Cargo.toml")).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
}

#[test]
fn cuda_manifest_enables_cudarc_f16() {
    let (_, _, manifest) = generate("cuda", "cuda_manifest");
    assert!(manifest.contains("\"f16\""), "{manifest}");
}
