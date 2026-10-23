//! A struct field is a valid `kernel.tensors.matmul/mma/linear` host operand inside a method of
//! its struct (docs/gpu-module.md, "Struct fields as operands"): same result in the interpreter
//! and in every GPU/SIMD build, with a destination field visible to the caller afterwards.
use std::{fs, path::PathBuf, process::Command};

fn dir(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    fs::create_dir_all(&root).unwrap();
    root
}

fn boring(args: &[&str], path: &PathBuf) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_boring")).args(args).arg(path).output().unwrap()
}

/// `forward` accumulates into `out` twice (mma), `dump` reads it back through a parameter
/// (the host cannot index a resident struct field directly, a separate limit).
fn program(w: &str, out: &str, x: &str) -> String {
    format!(
        "struct Layer:\n    let [float32, k = 4, m = 4]{w} w\n    mut [float32, n = 4, m = 4]{out} out\n\n    def forward([float32, n = 4, k = 4]{x} x) throws:\n        kernel.tensors.mma(x, w, out)\n\n    def dump([float32, n = 4, k = 4]'kernel'global eye, mut [float32, n = 4, m = 4]'kernel'unified res) throws:\n        kernel.tensors.matmul(out, eye, res)\n\n\
def main() throws:\n    let [float32, n = 4, k = 4]'kernel'global x = [1.0 as float32 for ..<16]\n    mut [float32, n = 4, k = 4]'kernel'unified eye = [0.0 as float32 for ..<16]\n    with eye:\n        for i in 0..<4:\n            eye[i * 5] = 1.0 as float32\n    mut [float32, n = 4, m = 4]'kernel'unified res = [0.0 as float32 for ..<16]\n    mut l = Layer([2.0 as float32 for ..<16], [0.0 as float32 for ..<16])\n    l.forward(x)\n    l.forward(x)\n    l.dump(eye, res)\n    with res:\n        print \"{{res[0]}} {{res[15]}}\"\n"
    )
}

const VARIANTS: [(&str, &str, &str, &str); 3] = [
    ("explicit", "'kernel'global", "'kernel'unified", "'kernel'global"),
    ("omitted", "", "", ""),
    ("bare", "'kernel", "'kernel", "'kernel"),
];

#[test]
fn field_operands_run_in_the_interpreter_with_a_visible_destination() {
    let root = dir("tensor_struct_fields_run");
    for (name, w, out, x) in VARIANTS {
        let path = root.join(format!("{name}.br"));
        fs::write(&path, program(w, out, x)).unwrap();
        let result = boring(&["run"], &path);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(result.status.success(), "{name}: {stderr}");
        // two accumulations of 1s(4x4) * 2s(4x4) = 8 each: 16 after the second one
        assert_eq!(String::from_utf8_lossy(&result.stdout).trim(), "16 16", "{name}");
    }
}

#[test]
fn field_operands_build_on_every_gpu_and_simd_target() {
    let root = dir("tensor_struct_fields_build");
    for (name, w, out, x) in VARIANTS {
        let path = root.join(format!("{name}.br"));
        fs::write(&path, program(w, out, x)).unwrap();
        for target in ["cuda", "metal", "rocm", "wgpu", "simd"] {
            let result = boring(&["build", "--target", target], &path);
            let stderr = String::from_utf8_lossy(&result.stderr);
            assert!(result.status.success(), "{name}, {target}: {stderr}");
        }
    }
}

#[test]
fn field_destination_rules_are_diagnosed() {
    let root = dir("tensor_struct_fields_errors");
    let base = program("'kernel'global", "'kernel'unified", "'kernel'global");
    for (case, src, expected) in [
        ("req", base.replace("def forward", "req forward"), "must be `def`"),
        (
            "not_mut",
            base.replace("mut [float32, n = 4, m = 4]'kernel'unified out", "[float32, n = 4, m = 4]'kernel'unified out"),
            "must be declared `mut`",
        ),
        ("alias", base.replace("mma(x, w, out)", "mma(x, out, out)"), "must not alias an input"),
    ] {
        let path = root.join(format!("{case}.br"));
        fs::write(&path, src).unwrap();
        let result = boring(&["run"], &path);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success(), "{case}");
        assert!(stderr.contains(expected), "{case}: {stderr}");
    }
}
