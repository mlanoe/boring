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
        assert!(err.contains("`GPU(...)` was removed"), "{target}: {err}");
    }
}

#[test]
fn new_placement_is_rejected_on_every_target() {
    for target in ["metal", "cuda", "rocm", "wgpu", "simd"] {
        let err = build_error(
            "removed_new_placement",
            "let data = [1.0, 2.0]\nlet k = new(data) Scale(data)\n",
            target,
        );
        assert!(err.contains("`new(device) Kernel(...)` was removed"), "{target}: {err}");
    }
}

#[test]
fn device_constructor_and_enumeration_are_rejected_on_every_target() {
    for target in ["metal", "cuda", "rocm", "wgpu", "simd"] {
        let err = build_error("removed_device_ctor", "let g = Device(index = 0)\nlet data = [1.0, 2.0]\n", target);
        assert!(err.contains("`Device(...)` was removed"), "{target}: {err}");
        let err = build_error("removed_device_all", "let data = [1.0, 2.0]\nfor g in Device.all():\n    print \"x\"\n", target);
        assert!(err.contains("`Device.all()` was removed"), "{target}: {err}");
    }
}

#[test]
fn kernel_placement_arguments_are_rejected_on_every_target() {
    for target in ["metal", "cuda", "rocm", "wgpu", "simd"] {
        for placement in ["kernel(target = .WGPU) Scale(data)", "kernel(device = 0) Scale(data)", "kernel(0) Scale(data)"] {
            let err = build_error("removed_kernel_placement", &format!("let data = [1.0, 2.0]\nlet k = {placement}\n"), target);
            assert!(err.contains("kernel placement was removed"), "{target} {placement}: {err}");
        }
    }
}
