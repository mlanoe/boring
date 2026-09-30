//! Tensor contracts are shared by interpreter and GPU compilation targets.
use std::{fs, process::Command};

#[test]
fn tensor_validation_is_shared_by_run_and_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_validation");
    fs::create_dir_all(&root).unwrap();
    for (case, output_binding, expected) in [
        (
            "immutable",
            "let",
            Some("tensor destination must be mutable"),
        ),
        ("valid", "mut", None),
    ] {
        let path = root.join(format!("{case}.br"));
        fs::write(&path, format!("kernel Matrix:\n    let [float32, k = 5, m = 3]'global a\n    let [float32, n = 7, k = 5]'global b\n    {output_binding} [float32, n = 7, m = 3]'unified c\n    def ():\n        gpu.tensor.matmulTile(a, b, c, row = 0, col = 0, rows = 4, cols = 8)\n")).unwrap();
        for target in [
            None,
            Some("cuda"),
            Some("metal"),
            Some("rocm"),
            Some("wgpu"),
        ] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_boring"));
            if let Some(target) = target {
                command.args(["build", "--target", target]);
            } else {
                command.arg("run");
            }
            let result = command.arg(&path).output().unwrap();
            let stderr = String::from_utf8_lossy(&result.stderr);
            if let Some(expected) = expected {
                assert!(!result.status.success(), "{case}, {target:?}");
                assert!(stderr.contains(expected), "{case}, {target:?}: {stderr}");
            } else {
                assert!(result.status.success(), "{case}, {target:?}: {stderr}");
            }
        }
    }
}

#[test]
fn tensor_collective_context_is_checked_on_every_target() {
    let root =
        std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_collective_context");
    fs::create_dir_all(&root).unwrap();
    let call = "gpu.tensor.matmulTile(a, b, c, row = ORIGIN, col = 0, rows = 4, cols = 8)";
    let cases = [
        (
            "uniform",
            format!(
                "        let row = gpu.block.y * 4\n        {}",
                call.replace("ORIGIN", "row")
            ),
            None,
        ),
        (
            "thread",
            format!("        {}", call.replace("ORIGIN", "gpu.thread.x")),
            Some("uniform non-negative"),
        ),
        (
            "float",
            format!("        {}", call.replace("ORIGIN", "1.5")),
            Some("uniform non-negative"),
        ),
        (
            "negative",
            format!("        {}", call.replace("ORIGIN", "-1")),
            Some("uniform non-negative"),
        ),
        (
            "branch",
            format!(
                "        if gpu.thread.x == 0:\n            {}",
                call.replace("ORIGIN", "0")
            ),
            Some("standalone entry-point"),
        ),
        (
            "loop",
            format!(
                "        for i in 0..<2:\n            {}",
                call.replace("ORIGIN", "0")
            ),
            Some("standalone entry-point"),
        ),
        (
            "initializer",
            format!("        let result = {}", call.replace("ORIGIN", "0")),
            Some("standalone entry-point"),
        ),
        (
            "return",
            format!("        return\n        {}", call.replace("ORIGIN", "0")),
            Some("participation cannot be proven"),
        ),
        (
            "conditional_return",
            format!(
                "        if gpu.thread.x == 0:\n            return\n        {}",
                call.replace("ORIGIN", "0")
            ),
            Some("participation cannot be proven"),
        ),
        (
            "shadow",
            format!("        let a = 0\n        {}", call.replace("ORIGIN", "0")),
            Some("must not be shadowed"),
        ),
        (
            "gpu_shadow",
            format!(
                "        let gpu = 0\n        {}",
                call.replace("ORIGIN", "0")
            ),
            Some("must not be shadowed"),
        ),
    ];
    for (name, body, expected) in cases {
        let path = root.join(format!("{name}.br"));
        fs::write(&path, format!("kernel Matrix:\n    let [float32, k = 5, m = 3]'global a\n    let [float32, n = 7, k = 5]'global b\n    mut [float32, n = 7, m = 3]'unified c\n    def ():\n{body}\n")).unwrap();
        for target in [
            None,
            Some("cuda"),
            Some("metal"),
            Some("rocm"),
            Some("wgpu"),
        ] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_boring"));
            if let Some(target) = target {
                command.args(["build", "--target", target]);
            } else {
                command.arg("run");
            }
            let result = command.arg(&path).output().unwrap();
            let stderr = String::from_utf8_lossy(&result.stderr);
            if let Some(expected) = expected {
                assert!(!result.status.success(), "{name}, {target:?}");
                assert!(stderr.contains(expected), "{name}, {target:?}: {stderr}");
            } else {
                assert!(result.status.success(), "{name}, {target:?}: {stderr}");
            }
        }
    }
}

#[test]
fn host_tensor_runs_and_gpu_builds_synthesize_dispatch() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("host_tensor");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("host_tensor.br");
    fs::write(&path, "let [float32, k = 5, m = 3]'gpu'global a = [float32(i % 7 - 3) for i in 0..<15]\nlet [float32, n = 7, k = 5]'gpu'unified b = [float32(i % 5 - 2) for i in 0..<35]\nmut [float32, n = 7, m = 3]'gpu'unified c = [float32(0) for ..<21]\ngpu.tensor.matmul(a, b, c)\ngpu.tensor.mma(a, b, c)\nwith c:\n    print \"{c[0]} {c[20]}\"\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(&path)
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 -18");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target])
            .arg(&path)
            .output()
            .unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn host_tensor_linear_runs_and_builds_for_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("host_tensor_linear");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("host_tensor_linear.br");
    fs::write(&path, "let [float32, k = 3, m = 2]'gpu'global x = [1.0 as float32, 2.0 as float32, 3.0 as float32, 4.0 as float32, 5.0 as float32, 6.0 as float32]\nlet [float32, k = 3, n = 2]'gpu'global w = [1.0 as float32, 0.0 as float32, -1.0 as float32, 2.0 as float32, 3.0 as float32, 4.0 as float32]\nmut [float32, n = 2, m = 2]'gpu'unified y = [0.0 as float32 for ..<4]\ngpu.tensor.linear(x, w, y)\nwith y:\n    print \"{y[0]} {y[1]} {y[2]} {y[3]}\"\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "-2 20 -2 47");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn tensor_linear_builds_inside_req_function_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_req_linear");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_req_linear.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32, k = 3, m = 2]'gpu'global x, [float32, k = 3, n = 2]'gpu'global w) throws:\n    mut [float32, n = 2, m = 2]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, w, y)\n    y\n\nlet [float32, k = 3, m = 2]'gpu'global x = [1.0 as float32 for ..<6]\nlet [float32, k = 3, n = 2]'gpu'global w = [1.0 as float32 for ..<6]\nlet result = compute(x, w)\n").unwrap();
    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_runs_and_builds_inside_req_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_req_linear");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_req_linear.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [float32]'gpu'global w, int m, int n, int k) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<m * n]\n    gpu.tensor.linear(x, w, y, m = m, n = n, k = k)\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32, 2.0 as float32, 3.0 as float32, 4.0 as float32, 5.0 as float32, 6.0 as float32]\nlet [float32]'gpu'global w = [1.0 as float32, 0.0 as float32, -1.0 as float32, 2.0 as float32, 3.0 as float32, 4.0 as float32]\nlet result = compute(x, w, 2, 2, 3)\nwith result:\n    print \"{result[0]} {result[1]} {result[2]} {result[3]}\"\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "-2 20 -2 47");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}
