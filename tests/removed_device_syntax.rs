use std::{fs, path::PathBuf, process::Command};

const KERNEL: &str = "kernel Scale:\n    mut [float]'unified buf\n    init([float]'unified data):\n        buf = data\n    def ():\n        let tid = .thread.x\n        buf[tid] = buf[tid] * 2.0\n\n";

fn build_error(name: &str, body: &str, target: &str) -> String {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join(format!("{name}.br"));
    fs::write(&input, format!("{KERNEL}{body}")).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!out.status.success(), "{target} accepted removed syntax");
    String::from_utf8_lossy(&out.stderr).to_string()
}

#[test]
fn gpu_n_is_rejected_on_every_target() {
    for target in ["metal", "cuda", "rocm", "wgpu", "simd"] {
        let err = build_error("removed_gpu_n", "let g0 = GPU(0)\nlet data = [1.0, 2.0]\n", target);
        assert!(err.contains("`GPU(n)` was removed"), "{target}: {err}");
    }
}

#[test]
fn new_placement_is_rejected_on_every_target() {
    for target in ["metal", "cuda", "rocm", "wgpu", "simd"] {
        let err = build_error(
            "removed_new_placement",
            "let g0 = Device(index = 0)\nlet data = [1.0, 2.0]\nlet k = new(g0) Scale(data)\n",
            target,
        );
        assert!(err.contains("`new(device) Kernel(...)` was removed"), "{target}: {err}");
    }
}
