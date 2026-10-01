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

    let metal_root = root.join("tensor_dynamic_q8_0_metal");
    let shader = fs::read_to_string(metal_root.join("kernels/main.metal")).unwrap();
    assert!(shader.contains("simd_shuffle_xor"), "Q8_0 decode must synthesize a warp reduction");
    assert!(shader.contains("simd_shuffle(scale, scaleLane)"), "automatic Q8_0 decode must broadcast each packed-block scale");
    assert!(shader.contains("const auto scaleLane = 0"), "32-lane targets must use a constant Q8_0 scale source lane");
    assert!(shader.contains("(blockByte + 2) + lane"), "32-lane targets must avoid recomputing the Q8_0 position with a modulo");
    assert_eq!(shader.matches("(cell < n)").count(), 1, "the uniform Q8_0 output bound must guard the whole warp schedule, not every decode iteration");
    assert!(!shader.contains("(inner < k)"), "32-lane targets with Q8_0-aligned rows do not need an inner bound check");
    let host = fs::read_to_string(metal_root.join("src/main.rs")).unwrap();
    assert!(host.contains("_blocks = if (1 == 1) { ((1 + 7) / 8)"), "Metal Q8_0 dispatch must use its eight 32-lane warps per block");
    assert!(!host.contains("input_c"), "dynamic linear must allocate its overwritten destination directly on the GPU");
    let rocm_host = fs::read_to_string(root.join("tensor_dynamic_q8_0_rocm/src/main.rs")).unwrap();
    assert!(rocm_host.contains("_blocks = if (1 == 1) { ((1 + 3) / 4)"), "ROCm Q8_0 dispatch must use its four 64-lane warps per block");
    let rocm_shader = fs::read_to_string(root.join("tensor_dynamic_q8_0_rocm/kernels/main.hip")).unwrap();
    assert!(rocm_shader.contains("const auto scaleLane = ((lane / 32) * 32)"), "ROCm wave64 must select lane 0 or 32 for each Q8_0 packed block");
    assert!(rocm_shader.contains("(inner < k)"), "ROCm wave64 must guard the optional second 32-lane half-wave");
    let wgpu_host = fs::read_to_string(root.join("tensor_dynamic_q8_0_wgpu/src/main.rs")).unwrap();
    assert!(wgpu_host.contains("__boring_tensor_host_0.resize_c(((1 * 1)) as usize)"), "WGPU must resize an uninitialized tensor destination without uploading it: {wgpu_host}");
    assert!(!wgpu_host.contains("__boring_tensor_host_0.copy_c_to_device"), "WGPU must not upload an overwritten tensor destination");
}

#[test]
fn project_tensor_config_can_force_scalar_q8_decode() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_config_scalar_q8");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("boring.toml"), "[project]\nname = \"tensor-config\"\n\n[tensor.linear.decode]\nq8_0 = \"scalar\"\n").unwrap();
    let path = root.join("main.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, y, m = 1, n = 1, k = 32, format = \"q8_0\")\n    y\n").unwrap();
    let build = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "metal"]).arg(&path).output().unwrap();
    assert!(build.status.success(), "{}", String::from_utf8_lossy(&build.stderr));
    let shader = fs::read_to_string(root.join("main_metal/kernels/main.metal")).unwrap();
    assert!(!shader.contains("simd_shuffle_xor"), "scalar override must omit the warp reduction");
}

#[test]
fn project_tensor_config_can_select_q5_warp_broadcast_decode() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_config_q5_warp_broadcast");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("boring.toml"), "[project]\nname = \"tensor-config-q5\"\n\n[tensor.linear.decode]\nq5_0 = \"warp-broadcast\"\n").unwrap();
    let path = root.join("main.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, y, m = 1, n = 1, k = 32, format = \"q5_0\")\n    y\n").unwrap();
    let build = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "metal"]).arg(&path).output().unwrap();
    assert!(build.status.success(), "{}", String::from_utf8_lossy(&build.stderr));
    let shader = fs::read_to_string(root.join("main_metal/kernels/main.metal")).unwrap();
    assert!(shader.contains("simd_shuffle((int32_t)(qh), scaleLane)"), "explicit Q5_0 warp-broadcast config must select the Q5 schedule");
}

#[test]
fn project_tensor_config_can_select_q4_warp_broadcast_decode() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_config_q4_warp_broadcast");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("boring.toml"), "[project]\nname = \"tensor-config-q4\"\n\n[tensor.linear.decode]\nq4_0 = \"warp-broadcast\"\n").unwrap();
    let path = root.join("main.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, y, m = 1, n = 1, k = 32, format = \"q4_0\")\n    y\n").unwrap();
    let build = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "metal"]).arg(&path).output().unwrap();
    assert!(build.status.success(), "{}", String::from_utf8_lossy(&build.stderr));
    let shader = fs::read_to_string(root.join("main_metal/kernels/main.metal")).unwrap();
    assert!(shader.contains("const auto blockByte = ((flat / 32) * 18)"), "explicit Q4_0 warp-broadcast config must select the Q4 schedule");
}

#[test]
fn project_tensor_config_can_select_iq4_warp_broadcast_decode() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_config_iq4_warp_broadcast");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("boring.toml"), "[project]\nname = \"tensor-config-iq4\"\n\n[tensor.linear.decode]\niq4_nl = \"warp-broadcast\"\n").unwrap();
    let path = root.join("main.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, y, m = 1, n = 1, k = 32, format = \"iq4_nl\")\n    y\n").unwrap();
    let build = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "metal"]).arg(&path).output().unwrap();
    assert!(build.status.success(), "{}", String::from_utf8_lossy(&build.stderr));
    let shader = fs::read_to_string(root.join("main_metal/kernels/main.metal")).unwrap();
    assert!(shader.contains("nibble == 14"), "explicit IQ4_NL warp-broadcast config must select the nonlinear codebook schedule");
}

#[test]
fn project_tensor_config_auto_selects_iq4_warp_broadcast_decode() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_config_iq4_auto");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("boring.toml"), "[project]\nname = \"tensor-config-iq4-auto\"\n\n[tensor.linear]\nalgorithm = \"auto\"\n").unwrap();
    let path = root.join("main.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, y, m = 1, n = 1, k = 32, format = \"iq4_nl\")\n    y\n").unwrap();
    let build = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "metal"]).arg(&path).output().unwrap();
    assert!(build.status.success(), "{}", String::from_utf8_lossy(&build.stderr));
    let shader = fs::read_to_string(root.join("main_metal/kernels/main.metal")).unwrap();
    assert!(shader.contains("simd_shuffle(scale"), "auto must select IQ4_NL warp-broadcast because it substantially outperforms the scalar tensor schedule");
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
    let metal_root = root.join("tensor_dynamic_q4_0_metal");
    let shader = fs::read_to_string(metal_root.join("kernels/main.metal")).unwrap();
    assert!(shader.contains("simd_shuffle(scale, scaleLane)"), "Q4_0 decode must broadcast its packed-block scale");
    assert!(shader.contains("const auto blockByte = ((flat / 32) * 18)"), "Q4_0 warp decode must use its 18-byte packed-block stride");
    assert_eq!(shader.matches("(cell < n)").count(), 1, "the uniform Q4_0 output bound must guard the whole warp schedule");
    let host = fs::read_to_string(metal_root.join("src/main.rs")).unwrap();
    assert!(host.contains("_blocks = if (1 == 1) { ((1 + 7) / 8)"), "Metal Q4_0 dispatch must schedule one warp per output");
    let rocm_shader = fs::read_to_string(root.join("tensor_dynamic_q4_0_rocm/kernels/main.hip")).unwrap();
    assert!(rocm_shader.contains("const auto scaleLane = ((lane / 32) * 32)"), "ROCm Q4_0 wave64 must select a source lane for each packed block");
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
    let metal_root = root.join("tensor_dynamic_q5_0_metal");
    let shader = fs::read_to_string(metal_root.join("kernels/main.metal")).unwrap();
    assert!(shader.contains("simd_shuffle(scale, scaleLane)"), "Q5_0 decode must broadcast its packed-block scale");
    assert!(shader.contains("simd_shuffle((int32_t)(qh), scaleLane)"), "Q5_0 decode must broadcast its high-bit word");
    assert_eq!(shader.matches("(cell < n)").count(), 1, "the uniform Q5_0 output bound must guard the whole warp schedule");
    let host = fs::read_to_string(metal_root.join("src/main.rs")).unwrap();
    assert!(host.contains("_blocks = if (1 == 1) { ((1 + 7) / 8)"), "Metal Q5_0 dispatch must schedule one warp per output");
    let rocm_shader = fs::read_to_string(root.join("tensor_dynamic_q5_0_rocm/kernels/main.hip")).unwrap();
    assert!(rocm_shader.contains("const auto scaleLane = ((lane / 32) * 32)"), "ROCm Q5_0 wave64 must select a source lane for each packed block");
    assert!(rocm_shader.contains("__shfl_sync(0xffffffff, qh, scaleLane)"), "ROCm Q5_0 must broadcast both high-bit words in a wave64");
}

#[test]
fn dynamic_tensor_linear_q5_0_handles_multiple_rows_outputs_and_blocks() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q5_0_multiblock");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q5_0_multiblock.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, weight, bias, y, m = 2, n = 2, k = 64, format = \"q5_0\")\n    y\n\nmut [float32]'gpu'global x = [1.0 as float32 for ..<128]\nfor i in 64..<128:\n    x[i] = 2.0 as float32\nmut [uint8]'gpu'global weight = [uint8(0) for ..<88]\n# Output 0: 32 * 1 + 32 * (-2) * 0.5 = 0.\nweight[0] = uint8(0)\nweight[1] = uint8(60)\nfor i in 2..<6:\n    weight[i] = uint8(255)\nfor i in 6..<22:\n    weight[i] = uint8(17)\nweight[22] = uint8(0)\nweight[23] = uint8(56)\nfor i in 28..<44:\n    weight[i] = uint8(238)\n# Output 1: 32 * 3 + 32 * (-1) * 2 = 32.\nweight[44] = uint8(0)\nweight[45] = uint8(60)\nfor i in 46..<50:\n    weight[i] = uint8(255)\nfor i in 50..<66:\n    weight[i] = uint8(51)\nweight[66] = uint8(0)\nweight[67] = uint8(64)\nfor i in 72..<88:\n    weight[i] = uint8(255)\nlet [float32]'gpu'global bias = [1.0 as float32, 2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print \"{result[0]} {result[1]} {result[2]} {result[3]}\"\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1 34 1 66");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_q5_0_matches_position_varying_reference() {
    const M: usize = 2;
    const N: usize = 2;
    const K: usize = 64;
    let scales = [1.0f32, 0.5, 2.0, -0.5];
    let mut packed = Vec::with_capacity(N * K / 32 * 22);
    let mut decoded = vec![0.0f32; N * K];
    for block in 0..4 {
        let scale_bits: u16 = match block { 0 => 0x3c00, 1 => 0x3800, 2 => 0x4000, _ => 0xb800 };
        packed.extend_from_slice(&scale_bits.to_le_bytes());
        let mut high = 0u32;
        let mut low = [0u8; 16];
        for position in 0..32 {
            let q = ((position * 7 + block * 5) % 32) as i32 - 16;
            let encoded = (q + 16) as u8;
            if encoded & 16 != 0 { high |= 1 << position; }
            if position < 16 { low[position] |= encoded & 15; }
            else { low[position - 16] |= (encoded & 15) << 4; }
            decoded[block * 32 + position] = q as f32 * scales[block];
        }
        packed.extend_from_slice(&high.to_le_bytes());
        packed.extend_from_slice(&low);
    }
    let x: Vec<f32> = (0..M * K).map(|index| ((index * 3) % 9) as f32 - 4.0).collect();
    let bias = [1.0f32, -2.0];
    let mut expected = Vec::with_capacity(M * N);
    for row in 0..M {
        for col in 0..N {
            let mut sum = bias[col];
            for inner in 0..K { sum += x[row * K + inner] * decoded[col * K + inner]; }
            expected.push(sum);
        }
    }
    let floats = x.iter().map(|value| format!("{value} as float32")).collect::<Vec<_>>().join(", ");
    let bytes = packed.iter().map(|value| format!("uint8({value})")).collect::<Vec<_>>().join(", ");
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q5_0_reference");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q5_0_reference.br");
    fs::write(&path, format!("req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, weight, bias, y, m = 2, n = 2, k = 64, format = \"q5_0\")\n    y\n\nlet [float32]'gpu'global x = [{floats}]\nlet [uint8]'gpu'global weight = [{bytes}]\nlet [float32]'gpu'global bias = [1.0 as float32, -2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print \"{{result[0]}},{{result[1]}},{{result[2]}},{{result[3]}}\"\n")).unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let actual: Vec<f32> = String::from_utf8_lossy(&run.stdout).trim().split(',')
        .map(|value| value.parse().unwrap()).collect();
    assert_eq!(actual, expected);

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_iq4_nl_matches_position_varying_reference() {
    const M: usize = 2;
    const N: usize = 2;
    const K: usize = 64;
    const CODEBOOK: [i32; 16] = [
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    ];
    let scales = [1.0f32, 0.5, 2.0, -0.5];
    let mut packed = Vec::with_capacity(N * K / 32 * 18);
    let mut decoded = vec![0.0f32; N * K];
    for block in 0..4 {
        let scale_bits: u16 = match block {
            0 => 0x3c00,
            1 => 0x3800,
            2 => 0x4000,
            _ => 0xb800,
        };
        packed.extend_from_slice(&scale_bits.to_le_bytes());
        let mut nibbles = [0u8; 16];
        for position in 0..32 {
            let codebook_index = (position * 7 + block * 5) % CODEBOOK.len();
            if position < 16 {
                nibbles[position] |= codebook_index as u8;
            } else {
                nibbles[position - 16] |= (codebook_index as u8) << 4;
            }
            decoded[block * 32 + position] = CODEBOOK[codebook_index] as f32 * scales[block];
        }
        packed.extend_from_slice(&nibbles);
    }
    let x: Vec<f32> = (0..M * K)
        .map(|index| ((index * 3) % 9) as f32 - 4.0)
        .collect();
    let bias = [1.0f32, -2.0];
    let mut expected = Vec::with_capacity(M * N);
    for row in 0..M {
        for col in 0..N {
            let mut sum = bias[col];
            for inner in 0..K {
                sum += x[row * K + inner] * decoded[col * K + inner];
            }
            expected.push(sum);
        }
    }
    let floats = x
        .iter()
        .map(|value| format!("{value} as float32"))
        .collect::<Vec<_>>()
        .join(", ");
    let bytes = packed
        .iter()
        .map(|value| format!("uint8({value})"))
        .collect::<Vec<_>>()
        .join(", ");
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("tensor_dynamic_iq4_nl_reference");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_iq4_nl_reference.br");
    fs::write(&path, format!("req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, weight, bias, y, m = 2, n = 2, k = 64, format = \"iq4_nl\")\n    y\n\nlet [float32]'gpu'global x = [{floats}]\nlet [uint8]'gpu'global weight = [{bytes}]\nlet [float32]'gpu'global bias = [1.0 as float32, -2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print \"{{result[0]}},{{result[1]}},{{result[2]}},{{result[3]}}\"\n")).unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(&path)
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let actual: Vec<f32> = String::from_utf8_lossy(&run.stdout)
        .trim()
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect();
    assert_eq!(actual, expected);

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{target}: {}",
            String::from_utf8_lossy(&build.stderr)
        );
    }
}

#[test]
fn dynamic_tensor_linear_q4_0_matches_position_varying_reference() {
    const M: usize = 2;
    const N: usize = 2;
    const K: usize = 64;
    let scales = [1.0f32, 0.5, 2.0, -0.5];
    let mut packed = Vec::with_capacity(N * K / 32 * 18);
    let mut decoded = vec![0.0f32; N * K];
    for block in 0..4 {
        let scale_bits: u16 = match block {
            0 => 0x3c00,
            1 => 0x3800,
            2 => 0x4000,
            _ => 0xb800,
        };
        packed.extend_from_slice(&scale_bits.to_le_bytes());
        let mut nibbles = [0u8; 16];
        for position in 0..32 {
            let encoded = ((position * 7 + block * 5) % 16) as u8;
            if position < 16 {
                nibbles[position] |= encoded;
            } else {
                nibbles[position - 16] |= encoded << 4;
            }
            decoded[block * 32 + position] = (encoded as i32 - 8) as f32 * scales[block];
        }
        packed.extend_from_slice(&nibbles);
    }
    let x: Vec<f32> = (0..M * K)
        .map(|index| ((index * 3) % 9) as f32 - 4.0)
        .collect();
    let bias = [1.0f32, -2.0];
    let mut expected = Vec::with_capacity(M * N);
    for row in 0..M {
        for col in 0..N {
            let mut sum = bias[col];
            for inner in 0..K {
                sum += x[row * K + inner] * decoded[col * K + inner];
            }
            expected.push(sum);
        }
    }
    let floats = x
        .iter()
        .map(|value| format!("{value} as float32"))
        .collect::<Vec<_>>()
        .join(", ");
    let bytes = packed
        .iter()
        .map(|value| format!("uint8({value})"))
        .collect::<Vec<_>>()
        .join(", ");
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("tensor_dynamic_q4_0_reference");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q4_0_reference.br");
    fs::write(&path, format!("req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, weight, bias, y, m = 2, n = 2, k = 64, format = \"q4_0\")\n    y\n\nlet [float32]'gpu'global x = [{floats}]\nlet [uint8]'gpu'global weight = [{bytes}]\nlet [float32]'gpu'global bias = [1.0 as float32, -2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print \"{{result[0]}},{{result[1]}},{{result[2]}},{{result[3]}}\"\n")).unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(&path)
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let actual: Vec<f32> = String::from_utf8_lossy(&run.stdout)
        .trim()
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect();
    assert_eq!(actual, expected);

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{target}: {}",
            String::from_utf8_lossy(&build.stderr)
        );
    }
}

#[test]
fn dynamic_tensor_linear_q8_0_matches_position_varying_reference() {
    const M: usize = 2;
    const N: usize = 2;
    const K: usize = 64;
    let scales = [1.0f32, 0.5, 2.0, -0.5];
    let mut packed = Vec::with_capacity(N * K / 32 * 34);
    let mut decoded = vec![0.0f32; N * K];
    for block in 0..4 {
        let scale_bits: u16 = match block {
            0 => 0x3c00,
            1 => 0x3800,
            2 => 0x4000,
            _ => 0xb800,
        };
        packed.extend_from_slice(&scale_bits.to_le_bytes());
        for position in 0..32 {
            let quantized = ((position * 17 + block * 29) % 255) as i32 - 127;
            packed.push((quantized & 0xff) as u8);
            decoded[block * 32 + position] = quantized as f32 * scales[block];
        }
    }
    let x: Vec<f32> = (0..M * K)
        .map(|index| ((index * 3) % 9) as f32 - 4.0)
        .collect();
    let bias = [1.0f32, -2.0];
    let mut expected = Vec::with_capacity(M * N);
    for row in 0..M {
        for col in 0..N {
            let mut sum = bias[col];
            for inner in 0..K {
                sum += x[row * K + inner] * decoded[col * K + inner];
            }
            expected.push(sum);
        }
    }
    let floats = x
        .iter()
        .map(|value| format!("{value} as float32"))
        .collect::<Vec<_>>()
        .join(", ");
    let bytes = packed
        .iter()
        .map(|value| format!("uint8({value})"))
        .collect::<Vec<_>>()
        .join(", ");
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("tensor_dynamic_q8_0_reference");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q8_0_reference.br");
    fs::write(&path, format!("req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, weight, bias, y, m = 2, n = 2, k = 64, format = \"q8_0\")\n    y\n\nlet [float32]'gpu'global x = [{floats}]\nlet [uint8]'gpu'global weight = [{bytes}]\nlet [float32]'gpu'global bias = [1.0 as float32, -2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print \"{{result[0]}},{{result[1]}},{{result[2]}},{{result[3]}}\"\n")).unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(&path)
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let actual: Vec<f32> = String::from_utf8_lossy(&run.stdout)
        .trim()
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect();
    assert_eq!(actual, expected);

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{target}: {}",
            String::from_utf8_lossy(&build.stderr)
        );
    }
}

#[test]
fn dynamic_tensor_linear_q6_k_matches_position_varying_reference() {
    const M: usize = 2;
    const N: usize = 2;
    const K: usize = 256;
    const BLOCK_BYTES: usize = 210;
    let block_scales = [1.0f32, -0.5];
    let mut packed = vec![0u8; N * BLOCK_BYTES];
    let mut decoded = vec![0.0f32; N * K];
    for block in 0..N {
        let base = block * BLOCK_BYTES;
        let scale_bits: u16 = if block == 0 { 0x3c00 } else { 0xb800 };
        packed[base + 208..base + 210].copy_from_slice(&scale_bits.to_le_bytes());
        let sub_scales: [i32; 16] = std::array::from_fn(|index| {
            let magnitude = (index % 7 + 1) as i32;
            if (index + block) % 2 == 0 { magnitude } else { -magnitude }
        });
        for (index, sub_scale) in sub_scales.iter().enumerate() {
            packed[base + 192 + index] = (*sub_scale & 0xff) as u8;
        }
        for position in 0..K {
            let iteration = position / 128;
            let within = position % 128;
            let group = within / 32;
            let lane = within % 32;
            let half = lane / 16;
            let scale_index = iteration * 8 + half + group * 2;
            let quantized = ((position * 13 + block * 17) % 64) as i32 - 32;
            let encoded = (quantized + 32) as u8;
            let low_offset = iteration * 64 + lane + if group % 2 == 0 { 0 } else { 32 };
            if group < 2 {
                packed[base + low_offset] |= encoded & 0x0f;
            } else {
                packed[base + low_offset] |= (encoded & 0x0f) << 4;
            }
            packed[base + 128 + iteration * 32 + lane] |=
                ((encoded >> 4) & 0x03) << (group * 2);
            decoded[block * K + position] =
                quantized as f32 * sub_scales[scale_index] as f32 * block_scales[block];
        }
    }
    let x: Vec<f32> = (0..M * K)
        .map(|index| ((index * 3) % 9) as f32 - 4.0)
        .collect();
    let bias = [1.0f32, -2.0];
    let mut expected = Vec::with_capacity(M * N);
    for row in 0..M {
        for col in 0..N {
            let mut sum = bias[col];
            for inner in 0..K {
                sum += x[row * K + inner] * decoded[col * K + inner];
            }
            expected.push(sum);
        }
    }
    let floats = x
        .iter()
        .map(|value| format!("{value} as float32"))
        .collect::<Vec<_>>()
        .join(", ");
    let bytes = packed
        .iter()
        .map(|value| format!("uint8({value})"))
        .collect::<Vec<_>>()
        .join(", ");
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("tensor_dynamic_q6_k_reference");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q6_k_reference.br");
    fs::write(&path, format!("req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, weight, bias, y, m = 2, n = 2, k = 256, format = \"q6_k\")\n    y\n\nlet [float32]'gpu'global x = [{floats}]\nlet [uint8]'gpu'global weight = [{bytes}]\nlet [float32]'gpu'global bias = [1.0 as float32, -2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print \"{{result[0]}},{{result[1]}},{{result[2]}},{{result[3]}}\"\n")).unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(&path)
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let actual: Vec<f32> = String::from_utf8_lossy(&run.stdout)
        .trim()
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect();
    assert_eq!(actual, expected);

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            build.status.success(),
            "{target}: {}",
            String::from_utf8_lossy(&build.stderr)
        );
    }
}

#[test]
fn dynamic_tensor_linear_q4_k_matches_position_varying_reference() {
    const M: usize = 2;
    const N: usize = 2;
    const K: usize = 256;
    const BLOCK_BYTES: usize = 144;
    let block_scales = [0.5f32, -1.0];
    let block_min_scales = [0.25f32, 0.5];
    let sub_scales = [1u8, 5, 17, 31, 33, 47, 55, 63];
    let sub_mins = [2u8, 7, 19, 29, 35, 45, 57, 62];
    let mut packed = vec![0u8; N * BLOCK_BYTES];
    let mut decoded = vec![0.0f32; N * K];
    for block in 0..N {
        let base = block * BLOCK_BYTES;
        let scale_bits: u16 = if block == 0 { 0x3800 } else { 0xbc00 };
        let min_bits: u16 = if block == 0 { 0x3400 } else { 0x3800 };
        packed[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
        packed[base + 2..base + 4].copy_from_slice(&min_bits.to_le_bytes());
        for index in 0..4 {
            packed[base + 4 + index] =
                (sub_scales[index] & 0x3f) | ((sub_scales[index + 4] >> 4) << 6);
            packed[base + 8 + index] =
                (sub_mins[index] & 0x3f) | ((sub_mins[index + 4] >> 4) << 6);
            packed[base + 12 + index] =
                (sub_scales[index + 4] & 0x0f) | ((sub_mins[index + 4] & 0x0f) << 4);
        }
        for position in 0..K {
            let chunk = position / 64;
            let within = position % 64;
            let half = within / 32;
            let lane = within % 32;
            let subblock = chunk * 2 + half;
            let nibble = ((position * 7 + block * 5) % 16) as u8;
            let quant_offset = base + 16 + chunk * 32 + lane;
            if half == 0 {
                packed[quant_offset] |= nibble;
            } else {
                packed[quant_offset] |= nibble << 4;
            }
            decoded[block * K + position] = block_scales[block]
                * sub_scales[subblock] as f32
                * nibble as f32
                - block_min_scales[block] * sub_mins[subblock] as f32;
        }
    }
    let x: Vec<f32> = (0..M * K)
        .map(|index| ((index * 3) % 9) as f32 - 4.0)
        .collect();
    let bias = [1.0f32, -2.0];
    let mut expected = Vec::with_capacity(M * N);
    for row in 0..M {
        for col in 0..N {
            let mut sum = bias[col];
            for inner in 0..K {
                sum += x[row * K + inner] * decoded[col * K + inner];
            }
            expected.push(sum);
        }
    }
    let floats = x.iter().map(|value| format!("{value} as float32")).collect::<Vec<_>>().join(", ");
    let bytes = packed.iter().map(|value| format!("uint8({value})")).collect::<Vec<_>>().join(", ");
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q4_k_reference");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q4_k_reference.br");
    fs::write(&path, format!("req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, weight, bias, y, m = 2, n = 2, k = 256, format = \"q4_k\")\n    y\n\nlet [float32]'gpu'global x = [{floats}]\nlet [uint8]'gpu'global weight = [{bytes}]\nlet [float32]'gpu'global bias = [1.0 as float32, -2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print \"{{result[0]}},{{result[1]}},{{result[2]}},{{result[3]}}\"\n")).unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let actual: Vec<f32> = String::from_utf8_lossy(&run.stdout).trim().split(',')
        .map(|value| value.parse().unwrap()).collect();
    assert_eq!(actual, expected);

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_q3_k_matches_position_varying_reference() {
    const M: usize = 2;
    const N: usize = 2;
    const K: usize = 256;
    const BLOCK_BYTES: usize = 110;
    let block_scales = [0.5f32, -1.0];
    let sub_scales = [-31i32, -24, -17, -9, -1, 3, 8, 14, 19, 23, 27, 31, -28, -13, 6, 16];
    let mut packed = vec![0u8; N * BLOCK_BYTES];
    let mut decoded = vec![0.0f32; N * K];
    for block in 0..N {
        let base = block * BLOCK_BYTES;
        let scale_bits: u16 = if block == 0 { 0x3800 } else { 0xbc00 };
        packed[base + 108..base + 110].copy_from_slice(&scale_bits.to_le_bytes());
        for (index, sub_scale) in sub_scales.iter().enumerate() {
            let encoded = (sub_scale + 32) as u8;
            let word = index / 4;
            let lane = index % 4;
            let low_offset = base + 96 + (word % 2) * 4 + lane;
            if word < 2 {
                packed[low_offset] |= encoded & 0x0f;
            } else {
                packed[low_offset] |= (encoded & 0x0f) << 4;
            }
            packed[base + 104 + lane] |= ((encoded >> 4) & 0x03) << (word * 2);
        }
        for position in 0..K {
            let outer = position / 128;
            let remainder = position % 128;
            let group = remainder / 32;
            let within32 = remainder % 32;
            let sub = within32 / 16;
            let lane = within32 % 16;
            let scale_index = outer * 8 + group * 2 + sub;
            let quantized = ((position * 5 + block * 3) % 8) as i32 - 4;
            packed[base + 32 + outer * 32 + sub * 16 + lane] |=
                ((quantized & 0x03) as u8) << (group * 2);
            if quantized >= 0 {
                packed[base + sub * 16 + lane] |= 1 << (outer * 4 + group);
            }
            decoded[block * K + position] =
                block_scales[block] * sub_scales[scale_index] as f32 * quantized as f32;
        }
    }
    let x: Vec<f32> = (0..M * K)
        .map(|index| ((index * 3) % 9) as f32 - 4.0)
        .collect();
    let bias = [1.0f32, -2.0];
    let mut expected = Vec::with_capacity(M * N);
    for row in 0..M {
        for col in 0..N {
            let mut sum = bias[col];
            for inner in 0..K {
                sum += x[row * K + inner] * decoded[col * K + inner];
            }
            expected.push(sum);
        }
    }
    let floats = x.iter().map(|value| format!("{value} as float32")).collect::<Vec<_>>().join(", ");
    let bytes = packed.iter().map(|value| format!("uint8({value})")).collect::<Vec<_>>().join(", ");
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q3_k_reference");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q3_k_reference.br");
    fs::write(&path, format!("req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, weight, bias, y, m = 2, n = 2, k = 256, format = \"q3_k\")\n    y\n\nlet [float32]'gpu'global x = [{floats}]\nlet [uint8]'gpu'global weight = [{bytes}]\nlet [float32]'gpu'global bias = [1.0 as float32, -2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print \"{{result[0]}},{{result[1]}},{{result[2]}},{{result[3]}}\"\n")).unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let actual: Vec<f32> = String::from_utf8_lossy(&run.stdout).trim().split(',')
        .map(|value| value.parse().unwrap()).collect();
    assert_eq!(actual, expected);

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_q2_k_matches_position_varying_reference() {
    const M: usize = 2;
    const N: usize = 2;
    const K: usize = 256;
    const BLOCK_BYTES: usize = 84;
    let block_scales = [0.5f32, -1.0];
    let block_min_scales = [0.25f32, 0.5];
    let sub_scales = [1u8, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 15];
    let sub_mins = [15u8, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 1];
    let mut packed = vec![0u8; N * BLOCK_BYTES];
    let mut decoded = vec![0.0f32; N * K];
    for block in 0..N {
        let base = block * BLOCK_BYTES;
        let scale_bits: u16 = if block == 0 { 0x3800 } else { 0xbc00 };
        let min_bits: u16 = if block == 0 { 0x3400 } else { 0x3800 };
        packed[base + 80..base + 82].copy_from_slice(&scale_bits.to_le_bytes());
        packed[base + 82..base + 84].copy_from_slice(&min_bits.to_le_bytes());
        for index in 0..16 {
            packed[base + index] = sub_scales[index] | (sub_mins[index] << 4);
        }
        for position in 0..K {
            let outer = position / 128;
            let remainder = position % 128;
            let group = remainder / 32;
            let within32 = remainder % 32;
            let sub = within32 / 16;
            let lane = within32 % 16;
            let scale_index = outer * 8 + group * 2 + sub;
            let quantized = ((position * 3 + block) % 4) as u8;
            packed[base + 16 + outer * 32 + sub * 16 + lane] |= quantized << (group * 2);
            decoded[block * K + position] = block_scales[block]
                * sub_scales[scale_index] as f32
                * quantized as f32
                - block_min_scales[block] * sub_mins[scale_index] as f32;
        }
    }
    let x: Vec<f32> = (0..M * K)
        .map(|index| ((index * 3) % 9) as f32 - 4.0)
        .collect();
    let bias = [1.0f32, -2.0];
    let mut expected = Vec::with_capacity(M * N);
    for row in 0..M {
        for col in 0..N {
            let mut sum = bias[col];
            for inner in 0..K {
                sum += x[row * K + inner] * decoded[col * K + inner];
            }
            expected.push(sum);
        }
    }
    let floats = x.iter().map(|value| format!("{value} as float32")).collect::<Vec<_>>().join(", ");
    let bytes = packed.iter().map(|value| format!("uint8({value})")).collect::<Vec<_>>().join(", ");
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q2_k_reference");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q2_k_reference.br");
    fs::write(&path, format!("req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight, [float32]'gpu'global bias) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32 for ..<4]\n    gpu.tensor.linear(x, weight, bias, y, m = 2, n = 2, k = 256, format = \"q2_k\")\n    y\n\nlet [float32]'gpu'global x = [{floats}]\nlet [uint8]'gpu'global weight = [{bytes}]\nlet [float32]'gpu'global bias = [1.0 as float32, -2.0 as float32]\nlet result = compute(x, weight, bias)\nwith result:\n    print \"{{result[0]}},{{result[1]}},{{result[2]}},{{result[3]}}\"\n")).unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let actual: Vec<f32> = String::from_utf8_lossy(&run.stdout).trim().split(',')
        .map(|value| value.parse().unwrap()).collect();
    assert_eq!(actual, expected);

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_reports_supported_quantized_formats() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_bad_format");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_bad_format.br");
    fs::write(&path, "req [float32]'gpu'unified compute([float32]'gpu'global x, [uint8]'gpu'global weight) throws:\n    mut [float32]'gpu'unified y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, y, m = 1, n = 1, k = 32, format = \"q5_made_up\")\n    y\n\nlet [float32]'gpu'global x = [1.0 as float32 for ..<32]\nlet [uint8]'gpu'global weight = [uint8(0) for ..<22]\nlet result = compute(x, weight)\n").unwrap();

    for target in [None, Some("cuda"), Some("metal"), Some("rocm"), Some("wgpu")] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_boring"));
        if let Some(target) = target { command.args(["build", "--target", target]); }
        else { command.arg("run"); }
        let result = command.arg(&path).output().unwrap();
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success(), "{target:?}");
        assert!(stderr.contains("expected one of: q8_0, q5_0, q4_0, iq4_nl, q6_k, q4_k, q3_k, q2_k"), "{target:?}: {stderr}");
    }
}

#[test]
fn dynamic_tensor_linear_q5_0_supports_no_bias_and_mixed_gpu_storage() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_q5_0_no_bias");
    fs::create_dir_all(&root).unwrap();
    let path = root.join("tensor_dynamic_q5_0_no_bias.br");
    fs::write(&path, "req [float32]'gpu'global compute([float32]'gpu'unified x, [uint8]'gpu'unified weight) throws:\n    mut [float32]'gpu'global y = [0.0 as float32]\n    gpu.tensor.linear(x, weight, y, m = 1, n = 1, k = 32, format = \"q5_0\")\n    y\n\nlet [float32]'gpu'unified x = [1.0 as float32 for ..<32]\nmut [uint8]'gpu'unified weight = [uint8(17) for ..<22]\nweight[0] = uint8(0)\nweight[1] = uint8(60)\nweight[2] = uint8(255)\nweight[3] = uint8(255)\nweight[4] = uint8(255)\nweight[5] = uint8(255)\nlet result = compute(x, weight)\nwith result:\n    print result[0]\n").unwrap();

    let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "32");

    for target in ["cuda", "metal", "rocm", "wgpu"] {
        let build = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", target]).arg(&path).output().unwrap();
        assert!(build.status.success(), "{target}: {}", String::from_utf8_lossy(&build.stderr));
    }
}

#[test]
fn dynamic_tensor_linear_reports_specific_runtime_shape_errors() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_shape_errors");
    fs::create_dir_all(&root).unwrap();
    let cases = [
        ("left", 31, 22, 1, 1, "tensor left operand length mismatch"),
        ("weight", 32, 21, 1, 1, "quantized tensor weight length mismatch"),
        ("bias", 32, 22, 2, 1, "tensor bias length mismatch"),
        ("destination", 32, 22, 1, 2, "tensor destination length mismatch"),
    ];
    for (name, x_len, weight_len, bias_len, output_len, expected) in cases {
        let path = root.join(format!("{name}.br"));
        fs::write(&path, format!("let [float32]'gpu'global x = [1.0 as float32 for ..<{x_len}]\nlet [uint8]'gpu'global weight = [uint8(0) for ..<{weight_len}]\nlet [float32]'gpu'global bias = [0.0 as float32 for ..<{bias_len}]\nmut [float32]'gpu'unified y = [0.0 as float32 for ..<{output_len}]\ngpu.tensor.linear(x, weight, bias, y, m = 1, n = 1, k = 32, format = \"q5_0\")\n")).unwrap();
        let run = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&path).output().unwrap();
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(!run.status.success(), "{name}");
        assert!(stderr.contains(expected), "{name}: {stderr}");
    }
}

#[test]
fn dynamic_tensor_linear_decodes_iq4_nl_weights_on_all_gpu_targets() {
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("tensor_dynamic_iq4_nl");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("boring.toml"), "[project]\nname = \"tensor-iq4-warp\"\n\n[tensor.linear.decode]\niq4_nl = \"warp-broadcast\"\n").unwrap();
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
    let metal_root = root.join("tensor_dynamic_iq4_nl_metal");
    let shader = fs::read_to_string(metal_root.join("kernels/main.metal")).unwrap();
    assert!(shader.contains("simd_shuffle(scale, scaleLane)"), "IQ4_NL decode must broadcast its packed-block scale");
    assert!(shader.contains("nibble == 14"), "IQ4_NL warp decode must preserve the nonlinear 16-value codebook");
    assert_eq!(shader.matches("(cell < n)").count(), 1, "the uniform IQ4_NL output bound must guard the whole warp schedule");
    let host = fs::read_to_string(metal_root.join("src/main.rs")).unwrap();
    assert!(host.contains("_blocks = if (1 == 1) { ((1 + 7) / 8)"), "Metal IQ4_NL dispatch must schedule one warp per output");
    let rocm_shader = fs::read_to_string(root.join("tensor_dynamic_iq4_nl_rocm/kernels/main.hip")).unwrap();
    assert!(rocm_shader.contains("const auto scaleLane = ((lane / 32) * 32)"), "ROCm IQ4_NL wave64 must select a source lane for each packed block");
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
    let wgsl = fs::read_to_string(root.join("tensor_dynamic_q3_k_wgpu/shaders/main.wgsl")).unwrap();
    assert!(wgsl.contains("i32(1) << u32("), "Q3_K high-mask shift must use a concrete i32 left operand: {wgsl}");
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
