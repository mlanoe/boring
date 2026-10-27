use std::{fs, path::PathBuf, process::Command};

const SOURCE: &str = r#"
kernel HalfBuffer:
    let float16 scale
    let [float16, 2] offsets
    mut [float16]'unified values
    mut [float16, 2]'actor scratch
    init(float16 factor, [float16, 2] bias, [float16]'unified input):
        scale = factor
        offsets = bias
        values = input
    def ():
        let i = .thread.x
        let float32 wide = values[i] as float32
        let float16 adjusted = sqrt(float16(wide * (scale as float32) + (offsets[0] as float32)))
        scratch[i] = adjusted
        sync
        values[i] = scratch[i].sqrt()

pub req [float16]'kernel'unified increment([float16]'kernel'unified input) throws:
    mut k = HalfBuffer(float16(1.0), [float16(1.0), float16(0.0)], input)
    kernel:
        k(block = input.length)
    k.values

def main():
    let float16 host_value = float16(1.25)
    let host_root = sqrt(float16(4.0))
    print "{host_value as float32} {host_root as float32}"
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
fn cpu_generated_rust_parses_optional_float16_casts() {
    let source_text = r#"
def main():
    let string valid = "1.5"
    let string invalid = "nope"
    let float16? parsed = valid as float16
    let float16? rejected = invalid as float16
"#;
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("float16_codegen/cpu_optional_cast");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, source_text).unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let host = fs::read_to_string(root.join("main_rust/src/main.rs")).unwrap();
    assert!(
        host.contains("parse::<f32>().ok().map(half::f16::from_f32)"),
        "optional float16 cast did not preserve its width:\n{host}"
    );
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"])
        .arg(root.join("main_rust/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        checked.status.success(),
        "{}",
        String::from_utf8_lossy(&checked.stderr)
    );
}

#[test]
fn cpu_interpreter_and_generated_rust_convert_and_compute_f16() {
    let source_text = "def main():\n    let float16 a = float16(1.337)\n    let float16 b = float16(\"2.0\")\n    let float16 c = (a + b) as float16\n    let int16 narrow_int = 7\n    let float16 from_int = narrow_int as float16\n    let float16 parsed = \"1.5\".parseFloat16() else float16(0.0)\n    let float16 root = sqrt(float16(4.0))\n    let float16 method_root = float16(9.0).sqrt()\n    let float16 squared = float16(4.0).pow(float16(2.0))\n    mut [float16] values = [float16(1.0), float16(2.0)]\n    let float16 total = values.sum()\n    values[0] = 1.337\n    values[1] = float16(1.0) + float16(2.0)\n    mut [float16] ordered = [float16(2.0), float16(1.0)]\n    ordered.sort()\n    let float16 bounded = clamp(float16(-2.0), float16(-1.0), float16(1.0))\n    let float32 smallest = min(float16(-2.0), float16(-1.0)) as float32\n    print \"{c as float32} {parsed as float32} {from_int as float32}\"\n    print \"{root as float32}\"\n    print \"{method_root as float32} {squared as float32}\"\n    print \"{total as float32}\"\n    print \"{bounded as float32} {smallest}\"\n    print \"{(float16(2048.0) + float16(1.0)) as float32} {values[0] as float32}\"\n    print \"{int(float16(3.5))} {float16(float16(1.5)) as float32} {float32(float16(1.5))}\"\n    print \"{float16(1.0) < float16(2.0)} {ordered[0] as float32}\"\n    print json(float16(1.5))\n    print \"{float16(12.5):+.2} {float16(12.5):.1e}\"\n    print \"{sum([float16(1.0), 2]) as float32} {min(float16(2.0), 1) as float32} {max(3, float16(2.0)) as float32} {clamp(float16(4.0), 0, 3) as float32}\"\n";
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/cpu");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, source_text).unwrap();
    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["run"]).arg(&source).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), "3.3359375 1.5 7\n2\n3 16\n3\n-1 -2\n2048 1.3369141\n3 1.5 1.5\ntrue 1\n1.5\n+12.50 1.2e1\n3 1 3 3");
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build"]).arg(&source).output().unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let manifest = root.join("main_rust/Cargo.toml");
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"]).arg(manifest).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
}

#[test]
fn cpu_generated_rust_sorts_and_sums_float16_struct_fields() {
    let source_text = r#"
struct HalfBag:
    var [float16] values

def main():
    mut bag = HalfBag(values = [float16(4.0), float16(3.0)])
    bag.values.sort()
    print "{bag.values.sum() as float32} {bag.values[0] as float32}"
"#;
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/cpu_struct_field");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, source_text).unwrap();
    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["run"]).arg(&source).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), "7 3");
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build"]).arg(&source).output().unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"])
        .arg(root.join("main_rust/Cargo.toml")).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
}

#[test]
fn cpu_generated_rust_float16_compound_assign_and_unwrapped_optional_casts() {
    let source_text = r#"
struct P:
    let float16 x
    let float16? maybe

def main():
    let float16 a = 1.5
    let float16 b = 2.25
    var float16 c = a + b * a - b / a
    c += 1.0
    c *= 2.0
    c -= 0.5
    c /= 2.0
    print "arith {c as float64}"
    let p = P(x = a, maybe = b)
    let pm = p.maybe else a
    print "unwrapped {pm as float64}"
    let parsed = "3.5".parseFloat16() else a
    print "parsed {parsed as float64}"
"#;
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("float16_codegen/cpu_compound_else");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, source_text).unwrap();
    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["run"]).arg(&source).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    let expected = "arith 4.125\nunwrapped 2.25\nparsed 3.5";
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), expected);
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build"]).arg(&source).output().unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--offline", "-q", "--manifest-path"])
        .arg(root.join("main_rust/Cargo.toml")).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), expected);
}

#[test]
fn cpu_generated_rust_coerces_float16_reassignments() {
    let source_text = r#"
struct HalfBox:
    var float16 value

var float16 globalHalf = float16(0.0)

def reset(var float16 value):
    value = 1.75

def resetGlobal():
    globalHalf = 2.25

def main():
    var float16 scalar = float16(0.0)
    scalar = 1.25
    reset(scalar)
    mut box = HalfBox(value = float16(0.0))
    box.value = 1.5
    var float16'guard guarded = float16(0.0)
    guarded = 2.5
    resetGlobal()
    print "{scalar as float32} {box.value as float32} {globalHalf as float32} {guarded as float32}"
"#;
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("float16_codegen/cpu_reassignments");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, source_text).unwrap();
    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["run"]).arg(&source).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), "1.75 1.5 2.25 2.5");
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build"]).arg(&source).output().unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let host = fs::read_to_string(root.join("main_rust/src/main.rs")).unwrap();
    assert!(host.contains("scalar = half::f16::from_f32((1.25) as f32)"), "{host}");
    assert!(host.contains("*value = half::f16::from_f32((1.75) as f32)"), "{host}");
    assert!(host.contains("r#box.value = half::f16::from_f32((1.5) as f32)"), "{host}");
    assert!(host.contains("half::f16::from_f32((2.25) as f32)"), "{host}");
    assert!(host.contains("half::f16::from_f32((2.5) as f32)"), "{host}");
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"])
        .arg(root.join("main_rust/Cargo.toml")).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
}

#[test]
fn cpu_generated_rust_preserves_inferred_float16_math_results() {
    let source_text = r#"
def main():
    let root = sqrt(float16(2.0))
    let method_root = float16(2.0).sqrt()
    let powered = pow(float16(2.0), float16(3.0))
    let method_powered = float16(2.0).pow(float16(3.0))
    let compound_root = sqrt(float16(4.0) + float16(5.0))
    let compound_method_root = (float16(4.0) + float16(5.0)).sqrt()
    let compound_wide = float32(float16(1.0) + float16(2.0))
    let compound_double = float64(float16(1.0) + float16(2.0))
    let compound_int = int(float16(1.0) + float16(2.0))
    let total = root + method_root + powered + method_powered + compound_root + compound_method_root
    let literal_sum = float16(1.5) + 2.0
    let signed = sign(float16(-2.0))
    print "{total as float32}"
    print "{literal_sum as float32}"
    print "{floatToBits(float16(1.5))} {signed as float32} {isNaN(float16(0.0))} {isInfinite(float16(0.0))}"
    print "{float16(1.0) == 1} {float16(1.0) < 2} {float16(1.0) == float32(1.0)}"
    print "{compound_wide} {compound_double} {compound_int}"
"#;
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/cpu_inferred_math");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, source_text).unwrap();
    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["run"]).arg(&source).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build"]).arg(&source).output().unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"])
        .arg(root.join("main_rust/Cargo.toml")).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
    let generated_run = Command::new("cargo")
        .args(["run", "--quiet", "--offline", "--manifest-path"])
        .arg(root.join("main_rust/Cargo.toml")).output().unwrap();
    assert!(generated_run.status.success(), "{}", String::from_utf8_lossy(&generated_run.stderr));
    assert_eq!(generated_run.stdout, interpreted.stdout);
}

#[test]
fn cpu_interpreter_executes_float16_kernel_fields_and_buffers() {
    let source_text = r#"
kernel HalfScale:
    let float16 gain
    mut [float16]'unified values
    init(float16 factor, [float16]'unified input):
        gain = factor
        values = input
    def ():
        values[.thread.x] = values[.thread.x] * gain

mut k = HalfScale(float16(2.0), [float16(1.337), float16(2.0)])
kernel:
    k(block = 2, grid = 1)
print "{k.values[0] as float32} {k.values[1] as float32}"
"#;
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/cpu_kernel");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, source_text).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["run"])
        .arg(&source)
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "2.6738281 4");
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
fn cpu_float16_throw_and_typed_catch_compile() {
    let source_text = "def risky() throws:\n    let float16 value = float16(1.5)\n    throw value\n\ndef main():\n    try:\n        risky()\n    catch float16:\n        print \"{error as float32}\"\n";
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/cpu_catch");
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
        assert!(device.contains("scratch"), "{target} omitted float16 actor storage:\n{device}");
        assert!(device.contains("sqrt("), "{target} did not lower float16 sqrt:\n{device}");
        assert!(!device.contains("float16("), "{target} left an unresolved float16 conversion:\n{device}");
        assert!(!host.contains("float16("), "{target} host left an unresolved float16 conversion:\n{host}");
        assert!(host.contains("half::f16"), "{target} host ABI is not f16:\n{host}");
        assert!(manifest.contains("half ="), "{target} manifest lacks half dependency");
    }
}

#[test]
fn wgpu_enables_shader_f16_and_preserves_two_byte_storage() {
    let (device, host, manifest) = generate("wgpu", "wgpu");
    assert!(device.starts_with("enable f16;"), "missing WGSL f16 directive:\n{device}");
    assert!(device.contains("array<f16>"), "wgpu widened the float16 buffer:\n{device}");
    assert!(device.contains("var<workgroup>") && device.contains("array<f16, 2>"),
        "wgpu did not preserve float16 workgroup storage:\n{device}");
    assert!(device.contains("sqrt("), "wgpu did not lower float16 sqrt:\n{device}");
    assert!(device.contains("scale: f32") && device.contains("offsets: array<f32, 2>"),
        "wgpu params ABI did not widen float16 values consistently:\n{device}");
    assert!(device.contains("let scale: f16 = f16(") && device.contains("f16(halfbuffer_params.offsets"),
        "wgpu did not restore float16 after loading widened params:\n{device}");
    assert!(!device.contains("float16("), "wgpu left an unresolved float16 conversion:\n{device}");
    assert!(host.contains("wgpu::Features::SHADER_F16"), "host does not request shader-f16:\n{host}");
    assert!(host.contains("Vec<half::f16>"), "host ABI is not f16:\n{host}");
    assert!(host.contains("half::f16::from_f32"), "host did not lower float16 construction:\n{host}");
    assert!(host.contains("size.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)"),
        "wgpu uploads do not align odd-sized f16 buffers:\n{host}");
    assert!(manifest.contains("wgpu = \"25\""), "wgpu backend does not use a Naga version with WGSL f16 support:\n{manifest}");
    assert!(manifest.contains("half ="), "wgpu manifest lacks half dependency");

    let generated = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/wgpu/main_wgpu");
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"])
        .arg(generated.join("Cargo.toml")).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
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
    assert!(host.contains("half::f16::from_f32") && host.contains(".sqrt()"),
        "simd did not round float16 sqrt back to half precision:\n{host}");
    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"])
        .arg(generated.join("Cargo.toml")).output().unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
}

#[test]
fn simd_float16_aggregates_coerce_untyped_literals() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen/simd_aggregates");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source = root.join("main.br");
    fs::write(&source, r#"
def main():
    let float16 total = sum([float16(1.0), 2])
    let float16 low = min(float16(2.0), 1)
    let float16 high = max(3, float16(2.0))
    let float16 bounded = clamp(float16(4.0), 0, 3)
    print "{total as float32} {low as float32} {high as float32} {bounded as float32}"
"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"]).arg(&source).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let generated = root.join("main_simd");
    let host = fs::read_to_string(generated.join("src/main.rs")).unwrap();
    assert!(host.contains("half::f16::from_f32"), "SIMD did not coerce aggregate literals:\n{host}");
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

#[test]
fn gpu_host_kernel_constructors_sum_float16_arrays_as_half() {
    let source_text = r#"
kernel HalfScalar:
    let float16 value

    init(float16 input):
        value = input

    def ():
        let copy = value

mut k = HalfScalar([float16(1.0), float16(2.0)].sum())
mut k2 = HalfScalar(float16(float16(1.0) + float16(2.0)))
mut k3 = HalfScalar(sqrt(float16(4.0)))
mut k4 = HalfScalar(float16(float32(float16(2.0))))
mut k5 = HalfScalar(float16(int(float16(2.0))))
mut k6 = HalfScalar(float16(4.0).sqrt())
mut k7 = HalfScalar(float16(2.0).pow(float16(3.0)))
mut k8 = HalfScalar(float16(4.0).clamp(float16(0.0), float16(3.0)))
mut k9 = HalfScalar(float16(float16(1.0) + 2.0))
mut k10 = HalfScalar([float16(1.0), 2.0].sum())
k.value = 1.5
kernel:
    k(block = 1)
    k2(block = 1)
    k3(block = 1)
    k4(block = 1)
    k5(block = 1)
    k6(block = 1)
    k7(block = 1)
    k8(block = 1)
    k9(block = 1)
    k10(block = 1)
"#;
    for target in ["metal", "cuda", "rocm", "wgpu"] {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("float16_codegen").join(format!("host_sum_{target}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let source = root.join("main.br");
        fs::write(&source, source_text).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&source).output().unwrap();
        assert!(output.status.success(), "{target}: {}", String::from_utf8_lossy(&output.stderr));
        let host = fs::read_to_string(root.join(format!("main_{target}/src/main.rs"))).unwrap();
        if target != "wgpu" {
            assert!(host.contains("sum::<half::f16>()"), "{target} widened a float16 host sum:\n{host}");
        }
        assert!(
            !host.contains("half::f16::from_f32(((half::f16::from_f32"),
            "{target} double-converted a compound float16 host expression:\n{host}"
        );
        assert!(
            !host.contains("as half::f16"),
            "{target} emitted an invalid Rust cast to float16:\n{host}"
        );
        assert!(
            host.contains("half::f16::from_f32(f32::from("),
            "{target} did not evaluate float16 host math through f32:\n{host}"
        );
        assert!(
            !host.contains("half::f16::from_f32((4.0) as f32).sqrt()"),
            "{target} called a math method directly on float16:\n{host}"
        );
        assert!(
            !host.contains("half::f16::from_f32((2.0) as f32).pow("),
            "{target} called pow directly on float16:\n{host}"
        );
        assert!(
            !host.contains("half::f16::from_f32((4.0) as f32).clamp("),
            "{target} called clamp directly on float16:\n{host}"
        );
        assert!(
            host.contains("+ half::f16::from_f32((2.0) as f32)"),
            "{target} did not coerce a mixed float16 host literal:\n{host}"
        );
        assert!(
            host.contains("vec![half::f16::from_f32((1.0) as f32), half::f16::from_f32((2.0) as f32)]"),
            "{target} did not coerce a mixed float16 host array:\n{host}"
        );
        assert!(
            host.contains("k.value = half::f16::from_f32((1.5) as f32)"),
            "{target} did not coerce a float16 kernel-field assignment:\n{host}"
        );
        assert!(
            host.contains("f32::from(half::f16::from_f32("),
            "{target} cast float16 to float32 with an invalid primitive cast:\n{host}"
        );
        assert!(
            host.contains("f32::from(half::f16::from_f32((2.0) as f32)) as isize"),
            "{target} cast float16 to int with an invalid primitive cast:\n{host}"
        );
    }
}
