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

// --- `obj.field` operands outside a method of the struct -------------------------------------
//
// Outside its struct a field operand (`layer.w` in a function or at the top level) is lowered to a
// call of a generated method of that struct, where it is the `self.w` form every backend handles.

const LAYER: &str = "struct Layer:\n    let [float32, k = 4, m = 4] w\n    mut [float32, n = 4, m = 4] out\n\n";
const EYE: &str = "mut [float32, n = 4, k = 4] eye = [0.0 as float32 for ..<16]\nwith eye:\n    for i in 0..<4:\n        eye[i * 5] = 1.0 as float32\n";

/// `run` writes into `l.out` from a free function; the top level reads it back through `eye`.
fn outside_free_function() -> String {
    format!(
        "{LAYER}def run(mut Layer l, [float32, n = 4, k = 4] x) throws:\n    kernel.tensors.matmul(x, l.w, l.out)\n\n\
def dump([float32, n = 4, k = 4] eye, mut [float32, n = 4, m = 4] res, Layer l) throws:\n    kernel.tensors.matmul(l.out, eye, res)\n\n\
def main() throws:\n    let [float32, n = 4, k = 4] x = [1.0 as float32 for ..<16]\n    mut [float32, n = 4, k = 4] eye = [0.0 as float32 for ..<16]\n    with eye:\n        for i in 0..<4:\n            eye[i * 5] = 1.0 as float32\n    mut [float32, n = 4, m = 4] res = [0.0 as float32 for ..<16]\n    mut l = Layer(w = [2.0 as float32 for ..<16], out = [0.0 as float32 for ..<16])\n    run(l, x)\n    dump(eye, res, l)\n    with res:\n        print \"{{res[0]}} {{res[15]}}\"\n"
    )
}

fn outside_top_level() -> String {
    format!(
        "{LAYER}let [float32, n = 4, k = 4] x = [1.0 as float32 for ..<16]\nmut l = Layer(w = [2.0 as float32 for ..<16], out = [0.0 as float32 for ..<16])\nkernel.tensors.matmul(x, l.w, l.out)\n{EYE}mut [float32, n = 4, m = 4] res = [0.0 as float32 for ..<16]\nkernel.tensors.matmul(l.out, eye, res)\nwith res:\n    print \"{{res[0]}} {{res[15]}}\"\n"
    )
}

/// `linear` with dimensions held in variables: they become parameters of the generated method.
fn outside_linear_variable_dims() -> String {
    "struct Model:\n    let [float32]'kernel'global w\n    mut [float32]'kernel'unified out\n\n\
def apply(mut Model m, [float32]'kernel'global x, int rows, int cols, int inner) throws:\n    kernel.tensors.linear(x, m.w, m.out, m = rows, n = cols, k = inner)\n\n\
def main() throws:\n    let [float32]'kernel'global x = [1.0 as float32 for ..<16]\n    mut m = Model(w = [2.0 as float32 for ..<16], out = [0.0 as float32 for ..<16])\n    apply(m, x, 4, 4, 4)\n    print \"ok\"\n"
        .to_string()
}

/// A field that is only read leaves the object immutable: the generated method is `req`.
fn outside_read_only_field() -> String {
    format!(
        "{LAYER}def read_into([float32, n = 4, k = 4] x, Layer l, mut [float32, n = 4, m = 4] y) throws:\n    kernel.tensors.matmul(x, l.w, y)\n\n\
def main() throws:\n    let [float32, n = 4, k = 4] x = [1.0 as float32 for ..<16]\n    mut [float32, n = 4, m = 4] y = [0.0 as float32 for ..<16]\n    let l = Layer(w = [2.0 as float32 for ..<16], out = [0.0 as float32 for ..<16])\n    read_into(x, l, y)\n    with y:\n        print \"{{y[0]}}\"\n"
    )
}

#[test]
fn outside_field_operands_run_in_the_interpreter() {
    let root = dir("tensor_struct_fields_outside_run");
    for (name, src, expected) in [
        ("free_function", outside_free_function(), "8 8"),
        ("top_level", outside_top_level(), "8 8"),
        ("linear_variable_dims", outside_linear_variable_dims(), "ok"),
        ("read_only_field", outside_read_only_field(), "8"),
    ] {
        let path = root.join(format!("{name}.br"));
        fs::write(&path, src).unwrap();
        let result = boring(&["run"], &path);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(result.status.success(), "{name}: {stderr}");
        assert_eq!(String::from_utf8_lossy(&result.stdout).trim(), expected, "{name}");
    }
}

#[test]
fn outside_field_operands_build_on_every_gpu_and_simd_target() {
    let root = dir("tensor_struct_fields_outside_build");
    for (name, src) in [("free_function", outside_free_function()), ("top_level", outside_top_level())] {
        let path = root.join(format!("{name}.br"));
        fs::write(&path, src).unwrap();
        for target in ["cuda", "metal", "rocm", "wgpu", "simd"] {
            let result = boring(&["build", "--target", target], &path);
            let stderr = String::from_utf8_lossy(&result.stderr);
            assert!(result.status.success(), "{name}, {target}: {stderr}");
        }
    }
}

/// The generated Rust compiles and gives the interpreter's answer (a host-side run: the simd
/// target executes the matmul for real, on any machine).
#[test]
fn outside_field_operands_compile_and_run_on_the_simd_target() {
    let root = dir("tensor_struct_fields_outside_simd");
    for (name, src, expected) in [("freefn", outside_free_function(), "8 8"), ("toplevel", outside_top_level(), "8 8")] {
        let path = root.join(format!("{name}.br"));
        fs::write(&path, src).unwrap();
        let built = boring(&["build", "--target", "simd"], &path);
        assert!(built.status.success(), "{name}: {}", String::from_utf8_lossy(&built.stderr));
        let cargo = Command::new("cargo")
            .args(["build", "--release", "--quiet", "--manifest-path"])
            .arg(root.join(format!("{name}_simd/Cargo.toml")))
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap();
        assert!(cargo.status.success(), "{name}: {}", String::from_utf8_lossy(&cargo.stderr));
        let exe = root.join(format!("{name}_simd/target/release/{name}-simd"));
        let run = Command::new(exe).output().unwrap();
        assert!(run.status.success(), "{name}: {}", String::from_utf8_lossy(&run.stderr));
        assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), expected, "{name}");
    }
}

#[test]
fn outside_field_operand_rules_are_diagnosed() {
    let root = dir("tensor_struct_fields_outside_errors");
    let free = outside_free_function();
    let top = outside_top_level();
    for (case, src, expected) in [
        ("immutable_object", free.replace("def run(mut Layer l,", "def run(Layer l,"), "must be declared `mut` (or `var mut`)"),
        (
            "not_mut_field",
            free.replace("mut [float32, n = 4, m = 4] out", "[float32, n = 4, m = 4] out"),
            "must be declared `mut`",
        ),
        (
            "two_objects",
            top.replace(
                "kernel.tensors.matmul(x, l.w, l.out)",
                "mut l2 = Layer(w = [1.0 as float32 for ..<16], out = [0.0 as float32 for ..<16])\nkernel.tensors.matmul(x, l.w, l2.out)",
            ),
            "two different objects",
        ),
    ] {
        let path = root.join(format!("{case}.br"));
        fs::write(&path, src).unwrap();
        let result = boring(&["run"], &path);
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success(), "{case}");
        assert!(stderr.contains(expected), "{case}: {stderr}");
    }
}
