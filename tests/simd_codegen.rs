use std::{fs, path::PathBuf, process::Command};

const SOURCE: &str = r#"
mut [float32]'kernel'unified x = [0.0 as float32 for ..<8]
mut [float32]'kernel'unified y = [1.0 as float32 for ..<8]
kernel Add:
    let [float32]'global x
    mut [float32]'unified y
    init([float32]'global a, [float32]'unified b):
        x = a
        y = b
    def ():
        let i = .thread.x + .block.x * .block_dim.x
        y[i] = x[i] + y[i]
mut k = Add(x, y)
kernel:
    k(block = 4, grid = 2)
print k.y[7]
"#;

#[test]
fn simd_project_builds_and_preserves_block_coordinates() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_codegen");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("main.br");
    fs::write(&input, SOURCE).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let generated = fs::read_to_string(root.join("main_simd/src/main.rs")).unwrap();
    assert!(generated.contains("is_x86_feature_detected!(\"avx2\")"));
    assert!(generated.contains("is_aarch64_feature_detected!(\"neon\")"));
    assert!(generated.contains("target_feature(enable=\"avx2\")"));
    assert!(generated.contains("BORING_SIMD_SCALAR"));
    assert!(generated.contains("static LANES: std::sync::OnceLock<usize>"));
    assert!(generated.contains("match __boring_simd_lanes()"));
    assert!(generated.contains("for __x in 0..dx as usize"));
    assert!(!generated.contains("let __end ="));
    assert!(!generated.contains("__lanes"));
    assert!(generated.contains("dispatch((2) as u32, (1) as u32, (1) as u32, (4) as u32"));
    let check = Command::new("cargo")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(root.join("main_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    for scalar in [false, true] {
        let mut command = Command::new("cargo");
        command
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join("main_simd/Cargo.toml"));
        if scalar {
            command.env("BORING_SIMD_SCALAR", "1");
        }
        let run = command.output().unwrap();
        assert!(
            run.status.success(),
            "scalar={scalar}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&run.stdout).trim(),
            "1",
            "scalar={scalar}"
        );
    }
}

#[test]
fn multi_target_syntax_is_not_accepted_yet() {
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "wgpu,simd", "missing.br"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("unknown target 'wgpu,simd'"));
}

#[test]
fn warp_lane_and_size_run_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_warp_properties");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("warp.br");
    fs::write(
        &input,
        "mut [int]'kernel'unified lanes = [0 for ..<8]\nmut [int]'kernel'unified sizes = [0 for ..<8]\nkernel W:\n    mut [int]'unified lanes\n    mut [int]'unified sizes\n    init(mut [int]'unified l, mut [int]'unified s):\n        lanes = l\n        sizes = s\n    def ():\n        let i = .thread.x + .thread.y * .blockDim.x + .thread.z * .blockDim.x * .blockDim.y\n        lanes[i] = .warp.lane\n        sizes[i] = .warp.size\nmut w = W(lanes, sizes)\nkernel:\n    w(block = (2, 2, 2))\nprint w.lanes[7]\nprint w.sizes[7]\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let generated = fs::read_to_string(root.join("warp_simd/src/main.rs")).unwrap();
    assert!(generated.contains("let __simd_lanes = __boring_simd_lanes() as i32;"));
    assert!(generated.contains("rem_euclid(__simd_lanes)"));
    for scalar in [false, true] {
        let mut command = Command::new("cargo");
        command
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join("warp_simd/Cargo.toml"));
        if scalar {
            command.env("BORING_SIMD_SCALAR", "1");
        }
        let run = command.output().unwrap();
        assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
        let values = String::from_utf8_lossy(&run.stdout)
            .lines()
            .map(|value| value.parse::<i32>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values[0], 7 % values[1]);
        assert!(matches!(values[1], 1 | 4 | 8 | 16));
        if scalar {
            assert_eq!(values[1], 1);
        }
    }
}

#[test]
fn replayable_warp_shuffles_run_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_warp_shuffle");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("warp.br");
    fs::write(
        &input,
        r#"
let [int]'kernel'global values = [i for i in 0..<16]
mut [int]'kernel'unified direct = [0 for ..<16]
mut [int]'kernel'unified down = [0 for ..<16]
mut [int]'kernel'unified up = [0 for ..<16]
mut [int]'kernel'unified xor = [0 for ..<16]
mut [int]'kernel'unified sizes = [0 for ..<16]
kernel W:
    let [int]'global values
    mut [int]'unified direct
    mut [int]'unified down
    mut [int]'unified up
    mut [int]'unified xor
    mut [int]'unified sizes
    init([int]'global v, mut [int]'unified d, mut [int]'unified dn, mut [int]'unified u, mut [int]'unified x, mut [int]'unified s):
        values = v
        direct = d
        down = dn
        up = u
        xor = x
        sizes = s
    def ():
        let i = .thread.x
        let value = values[i]
        direct[i] = .warp.shuffle(value, 0)
        down[i] = .warp.shuffle_down(value, 1)
        up[i] = .warp.shuffleUp(value, 1)
        xor[i] = .warp.shuffleXor(value, 1)
        sizes[i] = .warp.size
mut w = W(values, direct, down, up, xor, sizes)
kernel:
    w(block = 16)
print w.direct[15]
print w.down[0]
print w.up[1]
print w.xor[1]
print w.sizes[0]
"#,
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let generated = fs::read_to_string(root.join("warp_simd/src/main.rs")).unwrap();
    assert!(generated.contains("let __boring_warp_target_lane"));
    assert!(generated.contains("let __boring_warp_source_thread"));
    for scalar in [false, true] {
        let mut command = Command::new("cargo");
        command
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join("warp_simd/Cargo.toml"));
        if scalar {
            command.env("BORING_SIMD_SCALAR", "1");
        }
        let run = command.output().unwrap();
        assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
        let output = String::from_utf8_lossy(&run.stdout)
            .lines()
            .map(|value| value.parse::<i32>().unwrap())
            .collect::<Vec<_>>();
        let lanes = output[4];
        assert_eq!(output[0], 15 - 15 % lanes);
        assert_eq!(output[1], if lanes > 1 { 1 } else { 0 });
        assert_eq!(output[2], if lanes > 1 { 0 } else { 1 });
        assert_eq!(output[3], if lanes > 1 { 0 } else { 1 });
    }
}

#[test]
fn top_level_warp_sync_runs_as_a_collective_simd_phase() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_warp_sync");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("warp_sync.br");
    fs::write(
        &input,
        r#"
kernel WarpSync:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i + 1
        .warp.sync()
        output[i] = tile[3 - i]
mut warp_sync = WarpSync()
kernel:
    warp_sync(block = 4)
print "{warp_sync.output[0]} {warp_sync.output[1]} {warp_sync.output[2]} {warp_sync.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("warp_sync_simd/src/main.rs")).unwrap();
    assert!(!source.contains("__warp.sync"));
    assert!(source.matches("'__boring_work_item: for __x").count() >= 2);
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("warp_sync_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "4 3 2 1");
}

#[test]
fn unchanged_mutable_local_warp_shuffle_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_warp_mutable_local");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("warp.br");
    fs::write(
        &input,
        "mut [int]'kernel'unified values = [0 for ..<8]\nkernel W:\n    mut [int]'unified values\n    init(mut [int]'unified v):\n        values = v\n    def ():\n        var int lane = .thread.x\n        values[lane] = .warp.shuffle(lane, 0)\nmut w = W(values)\nkernel:\n    w(block = 8)\nprint w.values[7]\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    for scalar in [false, true] {
        let mut command = Command::new("cargo");
        command
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join("warp_simd/Cargo.toml"));
        if scalar {
            command.env("BORING_SIMD_SCALAR", "1");
        }
        let run = command.output().unwrap();
        assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
        let value = String::from_utf8_lossy(&run.stdout).trim().parse::<i32>().unwrap();
        if scalar {
            assert_eq!(value, 7);
        } else {
            assert!(matches!(value, 0 | 4 | 7));
        }
    }
}

#[test]
fn straight_line_mutable_local_warp_shuffle_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_warp_changed_mutable");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("warp.br");
    fs::write(
        &input,
        "mut [int]'kernel'unified values = [0 for ..<8]\nkernel W:\n    mut [int]'unified values\n    init(mut [int]'unified v):\n        values = v\n    def ():\n        var int lane = .thread.x\n        lane = lane + 1\n        lane = lane + 1\n        values[.thread.x] = .warp.shuffle(lane, 0)\nmut w = W(values)\nkernel:\n    w(block = 8)\nprint w.values[7]\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    for scalar in [false, true] {
        let mut command = Command::new("cargo");
        command
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join("warp_simd/Cargo.toml"));
        if scalar {
            command.env("BORING_SIMD_SCALAR", "1");
        }
        let run = command.output().unwrap();
        assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
        let value = String::from_utf8_lossy(&run.stdout).trim().parse::<i32>().unwrap();
        if scalar {
            assert_eq!(value, 9);
        } else {
            assert!(matches!(value, 2 | 6 | 9));
        }
    }
}

#[test]
fn branched_mutable_local_warp_shuffle_fails_during_codegen() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_warp_branched_mutable_error");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("warp.br");
    fs::write(
        &input,
        "kernel W:\n    def ():\n        var int lane = .thread.x\n        if lane >= 0:\n            lane = lane + 1\n        let value = .warp.shuffle(lane, 0)\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("lane-replayable value"));
}

#[test]
fn block_shared_storage_and_top_level_sync_run_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_actor_sync");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("actor.br");
    fs::write(
        &input,
        "mut [int]'kernel'unified output = [0 for ..<4]\nkernel Shared:\n    mut [int, 4]'actor tile\n    mut [int]'unified output\n    init(mut [int]'unified o):\n        output = o\n    def ():\n        let i = .thread.x\n        var int carried = i * 10\n        carried = carried + 1\n        tile[i] = i + 1\n        sync\n        carried = carried + tile[3 - i]\n        sync\n        output[i] = carried\nmut shared = Shared(output)\nkernel:\n    shared(block = 4)\nprint \"{shared.output[0]} {shared.output[1]} {shared.output[2]} {shared.output[3]}\"\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let generated = fs::read_to_string(root.join("actor_simd/src/main.rs")).unwrap();
    assert!(generated.matches("for tz in 0..dz as usize").count() >= 3);
    assert!(generated.matches("let i = __thread.x").count() >= 3);
    assert!(generated.contains("__boring_barrier_carried"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("actor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "5 14 23 32");
}

#[test]
fn block_shared_storage_is_reset_for_each_simd_block() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_actor_block_scope");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("actor_block_scope.br");
    fs::write(
        &input,
        r#"
kernel Shared:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        if .block.x == 0:
            tile[i] = 7
        sync
        output[i + .block.x * .block_dim.x] = tile[i]
mut shared = Shared()
kernel:
    shared(block = 4, grid = 2)
print "{shared.output[0]} {shared.output[3]} {shared.output[4]} {shared.output[7]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("actor_block_scope_simd/src/main.rs")).unwrap();
    assert!(source.contains("let mut tile: Vec<i32> = vec![Default::default(); 4]"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("actor_block_scope_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "7 7 0 0");
}

#[test]
fn top_level_for_loop_with_sync_runs_in_collective_simd_phases() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_for_loop");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_for_loop.br");
    fs::write(
        &input,
        r#"
kernel CollectiveLoop:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        for round in 0..<2:
            let i = .thread.x
            tile[i] = i + round * 10
            sync
            output[i] = tile[3 - i]
            sync
mut collective = CollectiveLoop()
kernel:
    collective(block = 4)
print "{collective.output[0]} {collective.output[1]} {collective.output[2]} {collective.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("sync_for_loop_simd/src/main.rs")).unwrap();
    assert!(source.contains("for round in 0..2"));
    assert!(source.matches("'__boring_work_item: for __x").count() >= 3);
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_for_loop_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "13 12 11 10");
}

#[test]
fn block_uniform_if_with_sync_runs_in_collective_simd_phases() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_uniform_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_if.br");
    fs::write(
        &input,
        r#"
kernel CollectiveIf:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        if .block.x == 0:
            let i = .thread.x
            tile[i] = i + 1
            sync
            output[i] = tile[3 - i]
mut collective = CollectiveIf()
kernel:
    collective(block = 4, grid = 2)
print "{collective.output[0]} {collective.output[3]} {collective.output[4]} {collective.output[7]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "4 1 0 0");
}

#[test]
fn thread_divergent_if_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_divergent_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_if.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        if .thread.x == 0:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(
        String::from_utf8_lossy(&generated.stderr)
            .contains("`sync` is not yet supported inside a --target simd kernel")
    );
}

#[test]
fn fixed_float_tensor_matmul_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_tensor_matmul");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32, k = 5, m = 3]'kernel'global a = [float32(i + 1) for i in 0..<15]
let [float32, n = 7, k = 5]'kernel'global b = [float32(i + 1) for i in 0..<35]
let [float32, k = 5, n = 7]'kernel'global w = [float32(i + 1) for i in 0..<35]
mut [float32, n = 7, m = 3]'kernel'unified c = [0.0 as float32 for ..<21]
mut [float32, n = 7, m = 3]'kernel'unified accum = [1.0 as float32 for ..<21]
mut [float32, n = 7, m = 3]'kernel'unified linear = [0.0 as float32 for ..<21]
kernel.tensors.matmul(a, b, c)
kernel.tensors.mma(a, b, accum)
kernel.tensors.linear(a, w, linear)
print c[0]
print c[6]
print c[20]
print accum[0]
print accum[6]
print accum[20]
print linear[0]
print linear[6]
print linear[20]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(source.contains("for __boring_tensor_k in 0usize..5usize"));
    assert!(source.contains("fn __boring_simd_threads()"));
    assert!(source.contains("static THREADS: std::sync::OnceLock<usize>"));
    assert!(source.contains("__boring_simd_threads().min(3usize)"));
    assert!(source.contains("if __boring_tensor_work < __boring_simd_parallel_threshold()"));
    assert!(source.contains(".chunks_mut(__boring_tensor_rows_per_worker * 7)"));
    assert!(source.contains("__boring_tensor_task();"));
    assert!(source.contains("__boring_tensor_scope.spawn(__boring_tensor_task)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "295\n385\n1435\n296\n386\n1436\n55\n505\n2155"
    );
}

#[test]
fn dynamic_float_tensor_operations_run_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_dynamic_tensor_ops");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let int m = 2
let int n = 2
let int k = 3
let [float32]'kernel'global x = [1.0 as float32, 2.0 as float32, 3.0 as float32, 4.0 as float32, 5.0 as float32, 6.0 as float32]
let [float32]'kernel'global w = [1.0 as float32, 0.0 as float32, -1.0 as float32, 2.0 as float32, 3.0 as float32, 4.0 as float32]
let [float32]'kernel'global b = [1.0 as float32, 2.0 as float32, 0.0 as float32, 3.0 as float32, -1.0 as float32, 4.0 as float32]
let [float32]'kernel'global bias = [10.0 as float32, 20.0 as float32]
mut [float32]'kernel'unified y = [0.0 as float32 for ..<m * n]
mut [float32]'kernel'unified y_no_bias = [0.0 as float32 for ..<m * n]
mut [float32]'kernel'unified product = [0.0 as float32 for ..<m * n]
mut [float32]'kernel'unified accumulated = [float32(i + 1) for i in 0..<m * n]
kernel.tensors.linear(x, w, bias, y, m = m, n = n, k = k)
kernel.tensors.linear(x, w, y_no_bias, m = m, n = n, k = k)
kernel.tensors.matmul(x, b, product, m = m, n = n, k = k)
kernel.tensors.mma(x, b, accumulated, m = m, n = n, k = k)
print y[0]
print y[1]
print y[2]
print y[3]
print y_no_bias[0]
print y_no_bias[1]
print y_no_bias[2]
print y_no_bias[3]
print product[0]
print product[1]
print product[2]
print product[3]
print accumulated[0]
print accumulated[1]
print accumulated[2]
print accumulated[3]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated_source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(generated_source.contains("__boring_simd_threads().min(__boring_tensor_m)"));
    assert!(generated_source
        .contains("chunks_mut(__boring_tensor_rows_per_worker * __boring_tensor_n)"));
    assert!(generated_source.contains("__boring_tensor_bias[__boring_tensor_col]"));
    assert!(generated_source.contains(".mul_add("));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "8\n40\n8\n67\n-2\n20\n-2\n47\n-2\n20\n-2\n47\n-1\n22\n1\n51"
    );
}

#[test]
fn dynamic_q8_tensor_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_dynamic_q8_linear");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32]'kernel'global x = [1.0 as float32 for ..<32]
mut [uint8]'kernel'global weight = [uint8(1) for ..<34]
weight[0] = uint8(0)
weight[1] = uint8(60)
weight[2] = uint8(255)
let [float32]'kernel'global bias = [2.0 as float32]
mut [float32]'kernel'unified y = [0.0 as float32]
mut [float32]'kernel'unified y_no_bias = [0.0 as float32]
kernel.tensors.linear(x, weight, bias, y, m = 1, n = 1, k = 32, format = "q8_0")
kernel.tensors.linear(x, weight, y_no_bias, m = 1, n = 1, k = 32, format = "q8_0")
print y[0]
print y_no_bias[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated_source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(generated_source.contains("fn __boring_f16_to_f32"));
    assert!(generated_source.contains("as i8 as f32"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "32\n30");
}

#[test]
fn dynamic_q4_tensor_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_dynamic_q4_linear");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32]'kernel'global x = [1.0 as float32 for ..<32]
mut [uint8]'kernel'global weight = [uint8(153) for ..<18]
weight[0] = uint8(0)
weight[1] = uint8(60)
weight[2] = uint8(151)
weight[3] = uint8(121)
let [float32]'kernel'global bias = [2.0 as float32]
mut [float32]'kernel'unified y = [0.0 as float32]
mut [float32]'kernel'unified y_no_bias = [0.0 as float32]
kernel.tensors.linear(x, weight, bias, y, m = 1, n = 1, k = 32, format = "q4_0")
kernel.tensors.linear(x, weight, y_no_bias, m = 1, n = 1, k = 32, format = "q4_0")
print y[0]
print y_no_bias[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated_source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(generated_source.contains("__boring_tensor_nibble"));
    assert!(generated_source.contains("as i32 - 8"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "30\n28");
}

#[test]
fn dynamic_q5_tensor_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_dynamic_q5_linear");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32]'kernel'global x = [1.0 as float32 for ..<32]
mut [uint8]'kernel'global weight = [uint8(17) for ..<22]
weight[0] = uint8(0)
weight[1] = uint8(60)
weight[2] = uint8(254)
weight[3] = uint8(255)
weight[4] = uint8(253)
weight[5] = uint8(255)
weight[6] = uint8(31)
weight[7] = uint8(241)
let [float32]'kernel'global bias = [2.0 as float32]
mut [float32]'kernel'unified y = [0.0 as float32]
mut [float32]'kernel'unified y_no_bias = [0.0 as float32]
kernel.tensors.linear(x, weight, bias, y, m = 1, n = 1, k = 32, format = "q5_0")
kernel.tensors.linear(x, weight, y_no_bias, m = 1, n = 1, k = 32, format = "q5_0")
print y[0]
print y_no_bias[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated_source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(generated_source.contains("__boring_tensor_high_bits"));
    assert!(generated_source.contains("__boring_tensor_quantized"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "30\n28");
}

#[test]
fn dynamic_iq4_tensor_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_dynamic_iq4_linear");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32]'kernel'global x = [1.0 as float32 for ..<32]
mut [uint8]'kernel'global weight = [uint8(136) for ..<18]
weight[0] = uint8(0)
weight[1] = uint8(60)
weight[2] = uint8(135)
weight[3] = uint8(120)
let [float32]'kernel'global bias = [2.0 as float32]
mut [float32]'kernel'unified y = [0.0 as float32]
mut [float32]'kernel'unified y_no_bias = [0.0 as float32]
kernel.tensors.linear(x, weight, bias, y, m = 1, n = 1, k = 32, format = "iq4_nl")
kernel.tensors.linear(x, weight, y_no_bias, m = 1, n = 1, k = 32, format = "iq4_nl")
print y[0]
print y_no_bias[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated_source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(generated_source.contains("[-127i32, -104, -83"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "12\n10");
}

#[test]
fn dynamic_q6k_tensor_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_dynamic_q6k_linear");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32]'kernel'global x = [1.0 as float32 for ..<256]
mut [uint8]'kernel'global weight = [uint8(17) for ..<210]
for i in 128..<192:
    weight[i] = uint8(170)
for i in 192..<208:
    weight[i] = uint8(1)
weight[0] = uint8(31)
weight[65] = uint8(31)
weight[128] = uint8(169)
weight[161] = uint8(169)
weight[208] = uint8(0)
weight[209] = uint8(60)
let [float32]'kernel'global bias = [2.0 as float32]
mut [float32]'kernel'unified y = [0.0 as float32]
mut [float32]'kernel'unified y_no_bias = [0.0 as float32]
kernel.tensors.linear(x, weight, bias, y, m = 1, n = 1, k = 256, format = "q6_k")
kernel.tensors.linear(x, weight, y_no_bias, m = 1, n = 1, k = 256, format = "q6_k")
print y[0]
print y_no_bias[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated_source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(generated_source.contains("__boring_tensor_qh_base"));
    assert!(generated_source.contains("__boring_tensor_scale_at = __boring_tensor_block + 208"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "254\n252");
}

#[test]
fn dynamic_q4k_tensor_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_dynamic_q4k_linear");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32]'kernel'global x = [1.0 as float32 for ..<256]
mut [uint8]'kernel'global weight = [uint8(34) for ..<144]
weight[0] = uint8(0)
weight[1] = uint8(60)
weight[2] = uint8(0)
weight[3] = uint8(60)
for i in 4..<12:
    weight[i] = uint8(1)
for i in 12..<16:
    weight[i] = uint8(17)
weight[16] = uint8(17)
weight[80] = uint8(33)
let [float32]'kernel'global bias = [2.0 as float32]
mut [float32]'kernel'unified y = [0.0 as float32]
mut [float32]'kernel'unified y_no_bias = [0.0 as float32]
kernel.tensors.linear(x, weight, bias, y, m = 1, n = 1, k = 256, format = "q4_k")
kernel.tensors.linear(x, weight, y_no_bias, m = 1, n = 1, k = 256, format = "q4_k")
print y[0]
print y_no_bias[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated_source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(generated_source.contains("__boring_tensor_submin"));
    assert!(generated_source.contains("(__boring_tensor_flat / 256) * 144"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "255\n253");
}

#[test]
fn dynamic_q3k_tensor_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_dynamic_q3k_linear");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32]'kernel'global x = [1.0 as float32 for ..<256]
mut [uint8]'kernel'global weight = [uint8(0) for ..<110]
for i in 0..<32:
    weight[i] = uint8(255)
for i in 32..<96:
    weight[i] = uint8(85)
for i in 96..<104:
    weight[i] = uint8(17)
for i in 104..<108:
    weight[i] = uint8(170)
weight[108] = uint8(0)
weight[109] = uint8(60)
let [float32]'kernel'global bias = [2.0 as float32]
mut [float32]'kernel'unified y = [0.0 as float32]
mut [float32]'kernel'unified y_no_bias = [0.0 as float32]
kernel.tensors.linear(x, weight, bias, y, m = 1, n = 1, k = 256, format = "q3_k")
kernel.tensors.linear(x, weight, y_no_bias, m = 1, n = 1, k = 256, format = "q3_k")
print y[0]
print y_no_bias[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated_source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(generated_source.contains("__boring_tensor_high_mask"));
    assert!(generated_source.contains("__boring_tensor_block + 108"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "258\n256");
}

#[test]
fn dynamic_q2k_tensor_linear_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_dynamic_q2k_linear");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tensor.br");
    fs::write(
        &input,
        r#"
let [float32]'kernel'global x = [1.0 as float32 for ..<256]
mut [uint8]'kernel'global weight = [uint8(0) for ..<84]
for i in 0..<16:
    weight[i] = uint8(17)
for i in 16..<80:
    weight[i] = uint8(170)
weight[80] = uint8(0)
weight[81] = uint8(60)
weight[82] = uint8(0)
weight[83] = uint8(60)
let [float32]'kernel'global bias = [2.0 as float32]
mut [float32]'kernel'unified y = [0.0 as float32]
mut [float32]'kernel'unified y_no_bias = [0.0 as float32]
kernel.tensors.linear(x, weight, bias, y, m = 1, n = 1, k = 256, format = "q2_k")
kernel.tensors.linear(x, weight, y_no_bias, m = 1, n = 1, k = 256, format = "q2_k")
print y[0]
print y_no_bias[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let generated_source = fs::read_to_string(root.join("tensor_simd/src/main.rs")).unwrap();
    assert!(generated_source.contains("__boring_tensor_scale_byte >> 4"));
    assert!(generated_source.contains("__boring_tensor_block + 82"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tensor_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "258\n256");
}

#[test]
fn gpu_resident_function_return_materializes_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_resident_return");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("resident.br");
    fs::write(
        &input,
        "kernel Values:\n    mut [float32]'unified out\n    init():\n        out = [0.0 as float32]\n    def ():\n        out[0] = 42.0 as float32\n\nreq [float32]'kernel'unified values() throws:\n    mut kernel_values = Values()\n    kernel:\n        kernel_values(block = 1)\n    kernel_values.out\n\nlet result = values()\nwith result:\n    print result[0]\n",
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("resident_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "42");
}

#[test]
fn local_reassignment_is_not_treated_as_a_kernel_field() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_locals_grid");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("locals.br");
    fs::write(
        &input,
        r#"
kernel K:
    mut [int]'unified out
    init():
        out = [0 for ..<4]
    def ():
        var int value = 0
        value = value + 1
        let i = .block.x + .block.y * 2
        out[i] = value
mut k = K()
kernel:
    k(block = 1, grid = (2, 2))
print k.out[3]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("locals_simd/src/main.rs")).unwrap();
    assert!(source.contains("value = (value + 1)"));
    assert!(!source.contains("self.value"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("locals_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1");
}

#[test]
fn local_array_expressions_run_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_local_arrays");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("arrays.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel Arrays:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        let [int] literal = [10, 20, 30, 40]
        let [int] filled = [2 for ..<4]
        let [int] computed = [j * 3 for j in ..<4]
        let [int] first = literal[..<2]
        let [int] tail = literal[2..]
        let [int] inclusive = literal[1..=2]
        let half = i % 2
        let scaled = do:
            let base = literal[i] + filled[i]
            base + computed[i]
        output[i] = scaled + first[half] + tail[half] + inclusive[half]
mut arrays = Arrays(output)
kernel:
    arrays(block = 4)
print "{arrays.output[0]} {arrays.output[1]} {arrays.output[2]} {arrays.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("arrays_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "72 115 98 141");
}

#[test]
fn local_optional_expressions_run_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_local_optionals");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("optionals.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel Optionals:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        var int? value = nil
        let int? direct = some(i + 10)
        let direct_value = direct else 0
        if value is nil and i % 2 == 0:
            value ?= -i - 1
        if value is not nil:
            output[i] = (value?.abs() else 99) + direct_value
        else:
            output[i] = (value?.abs() else 99) + direct_value
mut optionals = Optionals(output)
kernel:
    optionals(block = 4)
print "{optionals.output[0]} {optionals.output[1]} {optionals.output[2]} {optionals.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("optionals_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "11 110 15 112");
}

#[test]
fn local_tuple_destructuring_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_tuple_destructure");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("tuples.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel Tuples:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        let pair = (i + 1, i * 2, 100)
        let (left, right, _) = pair
        output[i] = left + right
mut tuples = Tuples(output)
kernel:
    tuples(block = 4)
print "{tuples.output[0]} {tuples.output[1]} {tuples.output[2]} {tuples.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("tuples_simd/src/main.rs")).unwrap();
    assert!(source.contains("let (left, right, _) = pair;"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("tuples_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1 4 7 10");
}

#[test]
fn local_if_let_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_if_let");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("if_let.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel IfLet:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        var int? candidate = nil
        let int? bonus = some(3)
        if i % 2 == 0:
            candidate ?= i + 2
        if let value = candidate, let extra = bonus, value > 2:
            output[i] = value + extra
        elif i == 0:
            output[i] = 10
        else:
            output[i] = 20
mut values = IfLet(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("if_let_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 20 7 20");
}

#[test]
fn local_while_let_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_while_let");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("while_let.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel WhileLet:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        var int? candidate = some(i + 1)
        while let value = candidate:
            output[i] = value
            candidate = nil
mut values = WhileLet(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("while_let_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1 2 3 4");
}

#[test]
fn self_referencing_while_let_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_self_while_let");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("self_while_let.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel SelfWhileLet:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        var int? value = some(i + 1)
        while let value:
            if i == 3 and value == 2:
                break
            output[i] = output[i] + value
            if value > 1:
                value = some(value - 1)
            else:
                value = nil
        if let stopped = value:
            output[i] = output[i] + stopped * 10
mut values = SelfWhileLet(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("self_while_let_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1 3 6 27");
}

#[test]
fn local_do_while_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_do_while");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("do_while.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel DoWhile:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        var value = i
        do: value = value + 1 while value < 3
        output[i] = value
mut values = DoWhile(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("do_while_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "3 3 3 4");
}

#[test]
fn local_match_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("match.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel Matches:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        let offset = match i:
            0: 7
            _: 1
        match i:
            0 | 1: output[i] = 10 + i + offset
            value if value == 2:
                let doubled = value * 2
                output[i] = doubled + offset
            _: output[i] = 99 + offset
mut values = Matches(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("match_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "17 12 5 100");
}

#[test]
fn local_guard_let_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_guard_let");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("guard_let.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel Guards:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        let int? candidate = some(i)
        for _ in 0..<1:
            guard let value = candidate, value < 2 else: continue
            output[i] = value + 10
mut values = Guards(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("guard_let_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 0 0");
}

#[test]
fn local_loop_expression_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_loop_expression");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("loop_expression.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel LoopExpression:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        let selected = loop:
            if i < 2:
                break i + 5
            break 9
        output[i] = selected
mut values = LoopExpression(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("loop_expression_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "5 6 9 9");
}

#[test]
fn local_pipeline_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_pipeline");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("pipeline.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel Pipeline:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        let value = (-i - 1) |> abs() |> clamp(2, 3)
        output[i] = value
mut values = Pipeline(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("pipeline_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "2 2 3 3");
}

#[test]
fn local_generic_call_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_generic_call");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("generic_call.br");
    fs::write(
        &input,
        r#"
def T identity<T>(T value): value

mut [int]'kernel'unified output = [0 for ..<4]
kernel GenericCall:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        output[i] = identity<int>(i + 4)
mut values = GenericCall(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("generic_call_simd/src/main.rs")).unwrap();
    assert!(source.contains("identity::<i32>"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("generic_call_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "4 5 6 7");
}

#[test]
fn local_labeled_array_relabel_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_labeled_local");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("labeled_local.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel LabeledLocal:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        let [int, width = 2, height = 2] original = [10, 20, 30, 40]
        var [int, w = 2, h = 2] renamed = original as [w = width, h = height]
        renamed[w = 1, h = 0] = 25
        output[i] = renamed[w = i % 2, h = i / 2] + renamed.w + renamed.h
mut values = LabeledLocal(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("labeled_local_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "14 29 34 44");
}

#[test]
fn local_array_closure_runs_on_cpu() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_array_closure");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("array_closure.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified output = [0 for ..<4]
kernel ArrayClosure:
    mut [int]'unified output
    init(mut [int]'unified o):
        output = o
    def ():
        let i = .thread.x
        let values = [1, 2, 3, 4].filter((value): value % 2 == 0).map((value): value + i)
        let int total = values.sum()
        let minimum = values.min()
        let maximum = values.max()
        let reduced = values.reduce(0, (sum, value): sum + value)
        output[i] = reduced + total + minimum + maximum + values.length
        if values.isEmpty:
            output[i] = -1
        if output.length != 4:
            output[i] = -2
        if values.any((value): value > 5):
            output[i] = output[i] + 1
mut values = ArrayClosure(output)
kernel:
    values(block = 4)
print "{values.output[0]} {values.output[1]} {values.output[2]} {values.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("array_closure_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "20 26 33 39");
}

#[test]
fn labeled_matrix_kernel_builds_and_runs() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_matrix_example");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("matrix.br");
    let example = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/matrix_mul_gpu.br");
    fs::copy(example, &input).unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("matrix_simd/src/main.rs")).unwrap();
    assert!(source.contains("let mut sum: f32"));
    assert!(source.contains("self.a[((i) as usize"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("matrix_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(String::from_utf8_lossy(&run.stdout).starts_with("C[0][0] = 2.4424"));
}

#[test]
fn integer_bitwise_operators_preserve_their_semantics() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_bitwise");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("bitwise.br");
    fs::write(
        &input,
        r#"
kernel Bits:
    mut [int]'unified out
    init():
        out = [0 for ..<1]
    def ():
        let int a = (5 & 3) | (1 << 3)
        let int b = (a ^ 2) >> 1
        out[0] = b + (~0 & 1)
mut bits = Bits()
kernel:
    bits(block = 1)
print bits.out[0]
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("bitwise_simd/src/main.rs")).unwrap();
    assert!(source.contains("(5 & 3)"));
    assert!(source.contains("(1 << 3)"));
    assert!(source.contains("(!0)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("bitwise_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "6");
}

#[test]
fn early_return_stops_only_the_current_work_item() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_early_return");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("early_return.br");
    fs::write(
        &input,
        r#"
kernel EarlyReturn:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        if i % 2 == 0:
            return
        output[i] = i + 10
mut early = EarlyReturn()
kernel:
    early(block = 4)
print "{early.output[0]} {early.output[1]} {early.output[2]} {early.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("early_return_simd/src/main.rs")).unwrap();
    assert!(source.contains("continue '__boring_work_item;"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("early_return_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "0 11 0 13");
}

#[test]
fn early_return_remains_inactive_after_a_simd_sync_phase() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_early_return");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_early_return.br");
    fs::write(
        &input,
        r#"
kernel EarlyReturn:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        if i == 1:
            return
        sync
        output[i] = i + 10
mut early = EarlyReturn()
kernel:
    early(block = 4)
print "{early.output[0]} {early.output[1]} {early.output[2]} {early.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("sync_early_return_simd/src/main.rs")).unwrap();
    assert!(source.contains("__boring_barrier_active[__boring_barrier_thread] = false;"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_early_return_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 0 12 13");
}

#[test]
fn labeled_array_comprehensions_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_labeled_comprehension");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("labeled_comprehension.br");
    fs::write(
        &input,
        r#"
kernel LabeledComprehension:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let [int, width = 2, height = 2] values = [width + height * 10 for width in ..<2 for height in ..<2]
        let i = .thread.x
        output[i] = values[width = i % 2, height = i / 2]
mut labeled = LabeledComprehension()
kernel:
    labeled(block = 4)
print "{labeled.output[0]} {labeled.output[1]} {labeled.output[2]} {labeled.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("labeled_comprehension_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "0 1 10 11");
}

#[test]
fn thread_local_fields_are_private_and_survive_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_thread_local_fields");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("thread_local_fields.br");
    fs::write(
        &input,
        r#"
kernel ThreadLocal:
    var int register
    mut [int, 2] scratch
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        register = i + 10
        scratch[0] = i
        scratch[1] = i * 100
        sync
        output[i] = register + scratch[0] + scratch[1]
mut local = ThreadLocal()
kernel:
    local(block = 4)
print "{local.output[0]} {local.output[1]} {local.output[2]} {local.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("thread_local_fields_simd/src/main.rs")).unwrap();
    assert!(source.contains("__boring_local_scratch"));
    assert!(source.contains("__boring_local_register"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("thread_local_fields_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 112 214 316");
}

#[test]
fn expression_sized_thread_local_arrays_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_expr_sized_local");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("expr_sized_local.br");
    fs::write(
        &input,
        r#"
kernel ExprSizedLocal:
    mut [int, 2 * 2] scratch
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        scratch[i] = i + 5
        output[i] = scratch[i]
mut sized = ExprSizedLocal()
kernel:
    sized(block = 4)
print "{sized.output[0]} {sized.output[1]} {sized.output[2]} {sized.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("expr_sized_local_simd/src/main.rs")).unwrap();
    assert!(source.contains("let mut scratch: Vec<i32>"));
    assert!(source.contains("vec![Default::default(); ((2 * 2)) as usize]"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("expr_sized_local_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "5 6 7 8");
}

#[test]
fn surface_buffers_run_on_simd_as_cpu_pixel_vectors() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_surface_buffer");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("surface_buffer.br");
    fs::write(
        &input,
        r#"
kernel Paint:
    mut [uint]'surface pixels
    init():
        pixels = [uint(0) for ..<4]
    def ():
        let i = .thread.x
        pixels[i] = uint(i * 3 + 1)
mut paint = Paint()
kernel:
    paint(block = 4)
print "{paint.pixels[0]} {paint.pixels[1]} {paint.pixels[2]} {paint.pixels[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "{}",
        String::from_utf8_lossy(&generated.stderr)
    );
    let source = fs::read_to_string(root.join("surface_buffer_simd/src/main.rs")).unwrap();
    assert!(source.contains("pixels: Vec<u32>"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("surface_buffer_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1 4 7 10");
}

#[test]
fn examples_with_top_level_scalar_constants_build_and_run() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for (example, expected_stdout) in [
        ("vector_add_gpu.br", "c[999] = 2997"),
        ("mandelbrot_gpu.br", "Image saved as 'mandelbrot.ppm'"),
    ] {
        let stem = example.trim_end_matches(".br");
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("simd_{stem}"));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let input = root.join(example);
        fs::copy(manifest_dir.join("examples").join(example), &input).unwrap();
        let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
            .args(["build", "--target", "simd"])
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            generated.status.success(),
            "{example}: {}",
            String::from_utf8_lossy(&generated.stderr)
        );
        let run = Command::new("cargo")
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join(format!("{stem}_simd/Cargo.toml")))
            .current_dir(&root)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{example}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(
            String::from_utf8_lossy(&run.stdout).contains(expected_stdout),
            "{example}: {}",
            String::from_utf8_lossy(&run.stdout)
        );
    }
}
