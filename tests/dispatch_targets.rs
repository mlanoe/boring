//! `dispatch` (docs/book.md, "Compile-time dispatch") on the GPU targets: the ladder it expands
//! to must instantiate one specialised kernel per listed value (`Scale_2`, `Scale_3`), inside a
//! `req` function, on every backend that supports const-generic kernels. Text checks only: no
//! GPU and no kernel build needed (`--emit-rust`).
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const PROGRAM: &str = r#"
kernel Scale<int N>:
    let [float32]'global input
    mut [float32]'global output
    init([float32]'kernel'global inp, [float32]'kernel'global out):
        input  = inp
        output = out
    def ():
        let i = .thread.x
        output[i] = input[i] * N

req [float32]'kernel'unified scaled([float32]'kernel'global x, int n) throws:
    mut out = [0.0 for ..<4]
    dispatch n in [2, 3]:
        mut k = Scale<n>(x, out)
        kernel:
            k(block = 4)
        return k.output

def main() throws:
    print scaled([1.0, 2.0, 3.0, 4.0], 3)
"#;

fn build(target: &str) -> String {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("dispatch_targets").join(target);
    fs::create_dir_all(&dir).unwrap();
    let br = dir.join("t.br");
    fs::write(&br, PROGRAM).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target, "--emit-rust"]).arg(&br).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(out.status.success(), "{target}: {text}");
    text
}

#[test]
fn dispatch_instantiates_one_kernel_per_value_on_every_gpu_target() {
    for target in ["metal", "cuda", "rocm", "wgpu"] {
        let text = build(target);
        assert!(text.contains("Scale_2, Scale_3"), "{target}: {text}");
    }
}
