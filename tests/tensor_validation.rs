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

#[test]
fn dynamic_tensor_linear_fuses_bias_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_bias");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_bias.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [float32]'gpu'global w, [float32]'gpu'global bias, int m, int n, int k) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<m * n]\n    gpu.tensor.linear(x, w, bias, y, m = m, n = n, k = k)\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32, 2.0 as float32, 3.0 as float32, 4.0 as float32]\nlet [float32]'gpu'global w = [1.0 as float32, 0.0 as float32, 0.0 as float32, 1.0 as float32]\nlet [float32]'gpu'global bias = [10.0 as float32, 20.0 as float32]\nlet result = compute(x, w, bias, 2, 2, 2)\nwith result:\n    print \"{result[0]} {result[1]} {result[2]} {result[3]}\"\n").unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "11 22 13 24");
    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring")).args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_decodes_q8_0_weights_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q8_0");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q8_0.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, bias, y, m = 1, n = 1, k = 32, format = \"q8_0\")\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32 for ..<32]\nmut [uint8]'gpu'global weight = [uint8(1) for ..<34]\nweight[0] = uint8(0)\nweight[1] = uint8(60)\nlet [float32]'gpu'global bias = [2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print result[0]\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "34");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_decodes_q4_0_weights_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q4_0");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q4_0.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, bias, y, m = 1, n = 1, k = 32, format = \"q4_0\")\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32 for ..<32]\nmut [uint8]'gpu'global weight = [uint8(153) for ..<18]\nweight[0] = uint8(0)\nweight[1] = uint8(60)\nlet [float32]'gpu'global bias = [2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print result[0]\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "34");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_decodes_q5_0_weights_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q5_0");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q5_0.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, bias, y, m = 1, n = 1, k = 32, format = \"q5_0\")\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32 for ..<32]\nmut [uint8]'gpu'global weight = [uint8(17) for ..<22]\nweight[0] = uint8(0)\nweight[1] = uint8(60)\nweight[2] = uint8(255)\nweight[3] = uint8(255)\nweight[4] = uint8(255)\nweight[5] = uint8(255)\nlet [float32]'gpu'global bias = [2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print result[0]\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "34");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_decodes_iq4_nl_weights_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_iq4_nl");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_iq4_nl.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, bias, y, m = 1, n = 1, k = 32, format = \"iq4_nl\")\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32 for ..<32]\nmut [uint8]'gpu'global weight = [uint8(136) for ..<18]\nweight[0] = uint8(0)\nweight[1] = uint8(60)\nlet [float32]'gpu'global bias = [2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print result[0]\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "34");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_decodes_q6_k_weights_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q6_k");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q6_k.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, bias, y, m = 1, n = 1, k = 256, format = \"q6_k\")\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32 for ..<256]\nmut [uint8]'gpu'global weight = [uint8(17) for ..<210]\nfor i in 128..<192:\n    weight[i] = uint8(170)\nfor i in 192..<208:\n    weight[i] = uint8(1)\nweight[208] = uint8(0)\nweight[209] = uint8(60)\nlet [float32]'gpu'global bias = [2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print result[0]\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "258");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_decodes_q4_k_weights_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q4_k");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q4_k.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, bias, y, m = 1, n = 1, k = 256, format = \"q4_k\")\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32 for ..<256]\nmut [uint8]'gpu'global weight = [uint8(0) for ..<144]\nweight[0] = uint8(0)\nweight[1] = uint8(60)\nweight[2] = uint8(0)\nweight[3] = uint8(60)\nfor i in 4..<12:\n    weight[i] = uint8(1)\nfor i in 12..<16:\n    weight[i] = uint8(17)\nfor i in 16..<144:\n    weight[i] = uint8(34)\nlet [float32]'gpu'global bias = [2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print result[0]\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "258");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_decodes_q2_k_weights_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q2_k");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q2_k.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, bias, y, m = 1, n = 1, k = 256, format = \"q2_k\")\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32 for ..<256]\nmut [uint8]'gpu'global weight = [uint8(0) for ..<84]\nfor i in 0..<16:\n    weight[i] = uint8(17)\nfor i in 16..<80:\n    weight[i] = uint8(170)\nweight[80] = uint8(0)\nweight[81] = uint8(60)\nweight[82] = uint8(0)\nweight[83] = uint8(60)\nlet [float32]'gpu'global bias = [2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print result[0]\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "258");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_decodes_q3_k_weights_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q3_k");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q3_k.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, bias, y, m = 1, n = 1, k = 256, format = \"q3_k\")\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32 for ..<256]\nmut [uint8]'gpu'global weight = [uint8(0) for ..<110]\nfor i in 0..<32:\n    weight[i] = uint8(255)\nfor i in 32..<96:\n    weight[i] = uint8(85)\nfor i in 96..<104:\n    weight[i] = uint8(17)\nfor i in 104..<108:\n    weight[i] = uint8(170)\nweight[108] = uint8(0)\nweight[109] = uint8(60)\nlet [float32]'gpu'global bias = [2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print result[0]\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "258");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_chains_resident_results_across_functions() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_chain");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_chain.br");
    fs::write(&path, "req [float32]'gpu'unified project([float32]'gpu'global x, [float32]'gpu'global w) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, w, y, m = 2, n = 2, k = 2)\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32, 2.0 as float32, 3.0 as float32, 4.0 as float32]\nlet [float32]'gpu'global w = [1.0 as float32, 0.0 as float32, 0.0 as float32, 1.0 as float32]\nlet [float32]'gpu'unified first = project(x, w)\nlet [float32]'gpu'unified second = project(first, w)\nwith second:\n    print \"{second[0]} {second[1]} {second[2]} {second[3]}\"\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1 2 3 4");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }

    let wgpu = fs::read_to_string(root.join("tensor_dynamic_chain_wgpu/src/main.rs")).unwrap();
    assert!(wgpu.contains("BoringGpuArg::Resident(buf, _len) =>"));
    assert!(wgpu.contains("std::sync::Arc::clone(buf)"));
    assert!(!wgpu.contains("a_buf = __boring_gpu_copy_d2d"));

    let metal = fs::read_to_string(root.join("tensor_dynamic_chain_metal/src/main.rs")).unwrap();
    assert!(metal.contains("BoringGpuArg::Resident(buf, _) => buf.clone()"));

    let cuda = fs::read_to_string(root.join("tensor_dynamic_chain_cuda/src/main.rs")).unwrap();
    assert!(cuda.contains("Resident(Arc<CudaSlice<T>>, usize)"));
    assert!(cuda.contains("BoringGpuArg::Resident(buf, _) => Arc::clone(buf)"));
    assert!(cuda.contains("a: Arc<CudaSlice<f32>>"));

    let rocm = fs::read_to_string(root.join("tensor_dynamic_chain_rocm/src/main.rs")).unwrap();
    assert!(rocm.contains("Resident(Arc<DeviceBuffer<T>>, usize)"));
    assert!(rocm.contains("BoringGpuArg::Resident(buf, _) => Arc::clone(buf)"));
    assert!(rocm.contains("a: Arc<DeviceBuffer<f32>>"));
}

#[test]
fn tensor_linear_builds_inside_loop_and_branch_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_control_flow");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_control_flow.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32, k = 2, m = 2]'gpu'global x, [float32, k = 2, n = 2]'gpu'global w, int count) throws:\n    mut [float32, n = 2, m = 2]'gpu'unified y = [0.0 as float32 for ..<4]\n    for i in 0..<count:\n        if i >= 0:\n            gpu.tensor.linear(x, w, y)\n    y\n\nlet [float32, k = 2, m = 2]'gpu'global x = [1.0 as float32, 2.0 as float32, 3.0 as float32, 4.0 as float32]\nlet [float32, k = 2, n = 2]'gpu'global w = [1.0 as float32, 0.0 as float32, 0.0 as float32, 1.0 as float32]\nlet result = compute(x, w, 2)\n").unwrap();

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
        let generated = fs::read_to_string(root.join(format!("tensor_control_flow_{target}/src/main.rs"))).unwrap();
        assert!(generated.contains("let mut y") && generated.contains("BoringGpuArg::Host"), "{target}: loop-carried y must use BoringGpuArg");
        assert!(generated.contains("y = BoringGpuArg::Resident") || generated.contains("y = { let __n = __boring_tensor_host_0.c.len(); BoringGpuArg::Resident"), "{target}: loop result must remain resident");
        assert!(!generated.contains("y = __boring_tensor_host_0.read_c()"), "{target}: unexpected loop D2H readback");
        assert!(!generated.contains("y = __boring_tensor_host_0.copy_c_to_host()"), "{target}: unexpected loop D2H readback");
    }
}
