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
fn collective_for_with_prefix_and_suffix_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_for_context");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_for_context.br");
    fs::write(
        &input,
        r#"
kernel LoopPhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i + 1
        sync
        for round in 0..<2:
            tile[i] = tile[i] + 10 + round
            sync
        output[i] = tile[i]
mut looped = LoopPhases()
kernel:
    looped(block = 4)
print "{looped.output[0]} {looped.output[1]} {looped.output[2]} {looped.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_for_context_simd/src/main.rs")).unwrap();
    assert!(source.contains("for round in 0..2"));
    assert!(source.matches("let i = __thread.x").count() >= 4);
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_for_context_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "22 23 24 25");
}

#[test]
fn collective_for_with_nested_if_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_for_nested_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_for_nested_if.br");
    fs::write(
        &input,
        r#"
kernel NestedLoop:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        for round in 0..<2:
            if round < 2:
                tile[i] = tile[i] + 10 + round
                sync
        output[i] = tile[i]
mut nested = NestedLoop()
kernel:
    nested(block = 4)
print "{nested.output[0]} {nested.output[1]} {nested.output[2]} {nested.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_for_nested_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "21 22 23 24");
}

#[test]
fn collective_for_with_nested_divergent_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_for_nested_divergent_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_for_nested_divergent_if.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        for round in 0..<2:\n            if .thread.x == round:\n                sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn block_uniform_while_reduction_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_uniform_while");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_while.br");
    fs::write(
        &input,
        r#"
kernel Reduce:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let tid = .thread.x
        var int stride = .block_dim.x / 2
        tile[tid] = tid + 1
        sync
        while stride > 0:
            if tid < stride:
                tile[tid] = tile[tid] + tile[tid + stride]
            sync
            stride = stride / 2
        output[tid] = tile[tid]
mut reduce = Reduce()
kernel:
    reduce(block = 4)
print "{reduce.output[0]} {reduce.output[1]} {reduce.output[2]} {reduce.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_uniform_while_simd/src/main.rs")).unwrap();
    assert!(source.contains("let mut stride: i32 = (__block_dim.x / 2)"));
    assert!(source.contains("while (stride > 0)"));
    assert!(source.matches("let tid = __thread.x").count() >= 3);
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_while_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 6 3 4");
}

#[test]
fn collective_while_with_nested_if_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_while_nested_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_while_nested_if.br");
    fs::write(
        &input,
        r#"
kernel NestedWhile:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        var int stride = 2
        tile[i] = i
        while stride > 0:
            if stride > 0:
                tile[i] = tile[i] + stride
                sync
            stride = stride - 1
        output[i] = tile[i]
mut nested = NestedWhile()
kernel:
    nested(block = 4)
print "{nested.output[0]} {nested.output[1]} {nested.output[2]} {nested.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_while_nested_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "3 4 5 6");
}

#[test]
fn collective_while_with_nested_divergent_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_while_nested_divergent_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_while_nested_divergent_if.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        var int stride = 2\n        while stride > 0:\n            if .thread.x < stride:\n                sync\n            stride = stride - 1\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn block_uniform_do_while_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_uniform_do_while");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_do_while.br");
    fs::write(
        &input,
        r#"
kernel CollectiveDoWhile:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        var int round = 0
        tile[i] = i
        do:
            tile[i] = tile[i] + round
            sync
            round = round + 1
        while round < 2
        output[i] = tile[i]
mut looped = CollectiveDoWhile()
kernel:
    looped(block = 4)
print "{looped.output[0]} {looped.output[1]} {looped.output[2]} {looped.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_do_while_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "1 2 3 4");
}

#[test]
fn thread_divergent_do_while_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_do_while");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_do_while.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        var int round = 0\n        do:\n            sync\n            round = round + 1\n        while round < .thread.x\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
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
fn collective_if_with_prefix_and_suffix_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_if_context");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_context.br");
    fs::write(
        &input,
        r#"
kernel ConditionalPhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        tile[i] = i + 1
        sync
        if .block.x == 0:
            tile[i] = tile[i] + 10
            sync
        else:
            tile[i] = tile[i] + 20
            sync
        output[i + .block.x * .block_dim.x] = tile[i]
mut conditional = ConditionalPhases()
kernel:
    conditional(block = 4, grid = 2)
print "{conditional.output[0]} {conditional.output[1]} {conditional.output[2]} {conditional.output[3]}"
print "{conditional.output[4]} {conditional.output[5]} {conditional.output[6]} {conditional.output[7]}"
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
    let source = fs::read_to_string(root.join("sync_if_context_simd/src/main.rs")).unwrap();
    assert!(source.matches("let i = __thread.x").count() >= 4);
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_if_context_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "11 12 13 14\n21 22 23 24"
    );
}

#[test]
fn nested_collective_ifs_with_sync_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_nested_ifs");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_nested_ifs.br");
    fs::write(
        &input,
        r#"
kernel NestedIfs:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i + 1
        if .grid_dim.x == 1:
            tile[i] = tile[i] + 10
            if .block_dim.x == 4:
                tile[i] = tile[i] + 20
                sync
            tile[i] = tile[i] + 30
            sync
        output[i] = tile[i]
mut nested = NestedIfs()
kernel:
    nested(block = 4)
print "{nested.output[0]} {nested.output[1]} {nested.output[2]} {nested.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_nested_ifs_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "61 62 63 64");
}

#[test]
fn nested_divergent_if_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_nested_divergent_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_nested_divergent_if.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        if .block.x == 0:\n            if .thread.x == 0:\n                sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn block_uniform_elif_else_with_different_sync_counts_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_uniform_branches");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_branches.br");
    fs::write(
        &input,
        r#"
kernel CollectiveBranches:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<12]
    def ():
        if .block.x == 0:
            let i = .thread.x
            tile[i] = i + 1
            sync
            output[i] = tile[3 - i]
        elif .block.x == 1:
            let i = .thread.x
            tile[i] = i + 10
            sync
            output[i + 4] = tile[3 - i] + 10
            sync
            output[i + 4] = output[i + 4]
        else:
            let i = .thread.x
            output[i + 8] = 99 + i
mut collective = CollectiveBranches()
kernel:
    collective(block = 4, grid = 3)
print "{collective.output[0]} {collective.output[1]} {collective.output[2]} {collective.output[3]}"
print "{collective.output[4]} {collective.output[5]} {collective.output[6]} {collective.output[7]}"
print "{collective.output[8]} {collective.output[9]} {collective.output[10]} {collective.output[11]}"
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
    let source = fs::read_to_string(root.join("sync_uniform_branches_simd/src/main.rs")).unwrap();
    assert!(source.contains("else if (__block.x == 1)"));
    assert!(source.matches("'__boring_work_item: for __x").count() >= 6);
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_branches_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "4 3 2 1\n23 22 21 20\n99 100 101 102"
    );
}

#[test]
fn block_uniform_match_with_sync_runs_in_collective_simd_phases() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_uniform_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_match.br");
    fs::write(
        &input,
        r#"
kernel CollectiveMatch:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<12]
    def ():
        match .block.x:
            0:
                let i = .thread.x
                tile[i] = i + 1
                sync
                output[i] = tile[3 - i]
            1:
                let i = .thread.x
                tile[i] = i + 10
                sync
                output[i + 4] = tile[3 - i] + 10
                sync
                output[i + 4] = output[i + 4]
            _:
                let i = .thread.x
                output[i + 8] = 99 + i
mut collective = CollectiveMatch()
kernel:
    collective(block = 4, grid = 3)
print "{collective.output[0]} {collective.output[1]} {collective.output[2]} {collective.output[3]}"
print "{collective.output[4]} {collective.output[5]} {collective.output[6]} {collective.output[7]}"
print "{collective.output[8]} {collective.output[9]} {collective.output[10]} {collective.output[11]}"
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
    let source = fs::read_to_string(root.join("sync_uniform_match_simd/src/main.rs")).unwrap();
    assert!(source.contains("match __block.x"));
    assert!(source.matches("'__boring_work_item: for __x").count() >= 6);
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_match_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "4 3 2 1\n23 22 21 20\n99 100 101 102"
    );
}

#[test]
fn block_uniform_tuple_match_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_uniform_tuple_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_tuple_match.br");
    fs::write(
        &input,
        r#"
kernel TupleMatch:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        match (.block.x, .block_dim.x):
            (0, 4):
                output[i] = i + 10
                sync
            _:
                output[i] = 99
                sync
mut matched = TupleMatch()
kernel:
    matched(block = 4)
print "{matched.output[0]} {matched.output[1]} {matched.output[2]} {matched.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_tuple_match_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_divergent_tuple_match_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_tuple_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_tuple_match.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        match (.block.x, .thread.x):\n            (0, 0):\n                sync\n            _:\n                pass\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn pure_optional_and_cast_calls_can_drive_collective_match_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_optional_call_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_optional_call_match.br");
    fs::write(
        &input,
        r#"
kernel OptionalCallMatch:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        match some(int(.block.x)):
            Some(0):
                output[i] = i + 10
                sync
            _:
                output[i] = 99
                sync
mut matched = OptionalCallMatch()
kernel:
    matched(block = 4)
print "{matched.output[0]} {matched.output[1]} {matched.output[2]} {matched.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_optional_call_match_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn optional_call_with_thread_value_remains_divergent_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_optional_call_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_optional_call_match.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        match some(int(.thread.x)):\n            Some(0):\n                sync\n            _:\n                pass\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_optional_predicate_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_optional_predicate");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_optional_predicate.br");
    fs::write(
        &input,
        r#"
kernel OptionalPredicate:
    let int? selected = some(10)
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        if selected.is_some():
            output[i] = i + 10
            sync
mut guarded = OptionalPredicate()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_optional_predicate_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_optional_predicate_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_optional_predicate");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_optional_predicate.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let candidate = some(.thread.x)\n        if candidate.is_some():\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_optional_fallback_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_optional_fallback");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_optional_fallback.br");
    fs::write(
        &input,
        r#"
kernel OptionalFallback:
    let int? selected = some(10)
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        if (selected else 0) == 10:
            output[i] = i + 10
            sync
mut guarded = OptionalFallback()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_optional_fallback_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_optional_fallback_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_optional_fallback");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_optional_fallback.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let candidate = some(.thread.x)\n        if (candidate else 0) == 0:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn collective_match_with_prefix_and_suffix_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_match_context");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_match_context.br");
    fs::write(
        &input,
        r#"
kernel MatchPhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        tile[i] = i + 1
        sync
        match .block.x:
            0:
                tile[i] = tile[i] + 10
                sync
            _:
                tile[i] = tile[i] + 20
                sync
        output[i + .block.x * .block_dim.x] = tile[i]
mut matched = MatchPhases()
kernel:
    matched(block = 4, grid = 2)
print "{matched.output[0]} {matched.output[1]} {matched.output[2]} {matched.output[3]}"
print "{matched.output[4]} {matched.output[5]} {matched.output[6]} {matched.output[7]}"
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
    let source = fs::read_to_string(root.join("sync_match_context_simd/src/main.rs")).unwrap();
    assert!(source.matches("let i = __thread.x").count() >= 4);
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_match_context_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "11 12 13 14\n21 22 23 24"
    );
}

#[test]
fn collective_match_with_nested_bound_if_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_match_nested_bound_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_match_nested_bound_if.br");
    fs::write(
        &input,
        r#"
kernel NestedMatch:
    let int? selected = some(2)
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        match selected:
            Some(value):
                if value == 2:
                    tile[i] = tile[i] + value + 10
                    sync
            _:
                pass
        output[i] = tile[i]
mut nested = NestedMatch()
kernel:
    nested(block = 4)
print "{nested.output[0]} {nested.output[1]} {nested.output[2]} {nested.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_match_nested_bound_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "12 13 14 15");
}

#[test]
fn collective_match_with_nested_divergent_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_match_nested_divergent_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_match_nested_divergent_if.br");
    fs::write(
        &input,
        "kernel Divergent:\n    let int? selected = some(0)\n    def ():\n        match selected:\n            Some(value):\n                if value == .thread.x:\n                    sync\n            _:\n                pass\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn collective_if_let_with_prefix_and_suffix_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_if_let_context");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_let_context.br");
    fs::write(
        &input,
        r#"
kernel IfLetPhases:
    let int? selected = some(40)
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        if let value = selected, value == 40:
            output[i] = value + i
            sync
        else:
            output[i] = 99
            sync
        output[i] = output[i] + 1
        sync
mut conditional = IfLetPhases()
kernel:
    conditional(block = 4)
print "{conditional.output[0]} {conditional.output[1]} {conditional.output[2]} {conditional.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_if_let_context_simd/src/main.rs")).unwrap();
    assert!(source.contains("if let Some(value)"));
    assert!(source.contains("if (value == 40)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_if_let_context_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "41 42 43 44");
}

#[test]
fn collective_if_let_with_nested_bound_if_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_if_let_nested_bound_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_let_nested_bound_if.br");
    fs::write(
        &input,
        r#"
kernel NestedIfLet:
    let int? selected = some(2)
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        if let value = selected:
            if value == 2:
                tile[i] = tile[i] + value + 10
                sync
        output[i] = tile[i]
mut nested = NestedIfLet()
kernel:
    nested(block = 4)
print "{nested.output[0]} {nested.output[1]} {nested.output[2]} {nested.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_if_let_nested_bound_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "12 13 14 15");
}

#[test]
fn collective_if_let_with_nested_divergent_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_if_let_nested_divergent_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_let_nested_divergent_if.br");
    fs::write(
        &input,
        "kernel Divergent:\n    let int? selected = some(0)\n    def ():\n        if let value = selected:\n            if value == .thread.x:\n                sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn successive_collective_ifs_with_sync_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_successive_ifs");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_successive_ifs.br");
    fs::write(
        &input,
        r#"
kernel SuccessivePhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        tile[i] = i
        sync
        if .block.x == 0:
            tile[i] = tile[i] + 10
            sync
        else:
            tile[i] = tile[i] + 20
            sync
        if .block_dim.x == 4:
            tile[i] = tile[i] + 1
            sync
        else:
            tile[i] = 99
            sync
        output[i + .block.x * .block_dim.x] = tile[i]
mut phased = SuccessivePhases()
kernel:
    phased(block = 4, grid = 2)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
print "{phased.output[4]} {phased.output[5]} {phased.output[6]} {phased.output[7]}"
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
    let source = fs::read_to_string(root.join("sync_successive_ifs_simd/src/main.rs")).unwrap();
    assert!(source.contains("if (__block.x == 0)"));
    assert!(source.contains("if (__block_dim.x == 4)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_successive_ifs_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "11 12 13 14\n21 22 23 24"
    );
}

#[test]
fn immutable_uniform_aliases_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_uniform_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_alias.br");
    fs::write(
        &input,
        r#"
kernel AliasPhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        let block_id = .block.x
        let first_block = block_id == 0
        tile[i] = i
        sync
        if first_block:
            tile[i] = tile[i] + 10
            sync
        else:
            tile[i] = tile[i] + 20
            sync
        output[i + block_id * .block_dim.x] = tile[i]
mut phased = AliasPhases()
kernel:
    phased(block = 4, grid = 2)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
print "{phased.output[4]} {phased.output[5]} {phased.output[6]} {phased.output[7]}"
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
    let source = fs::read_to_string(root.join("sync_uniform_alias_simd/src/main.rs")).unwrap();
    assert!(source.matches("let first_block").count() >= 2);
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "10 11 12 13\n20 21 22 23"
    );
}

#[test]
fn uniform_if_expression_alias_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_if_expression_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_if_expression_alias.br");
    fs::write(
        &input,
        r#"
kernel ConditionalAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let enabled = if .block.x == 0: true else: false
        if enabled:
            output[i] = i + 10
            sync
mut guarded = ConditionalAlias()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_if_expression_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_if_expression_alias_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_if_expression_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_if_expression_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let enabled = if .thread.x == 0: true else: false\n        if enabled:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_match_expression_alias_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_match_expression_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_match_expression_alias.br");
    fs::write(
        &input,
        r#"
kernel MatchAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let enabled = match .block.x:
            block_id if block_id == 0:
                let is_first = block_id == 0
                is_first
            _: false
        if enabled:
            output[i] = i + 10
            sync
mut guarded = MatchAlias()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_match_expression_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_match_expression_alias_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_match_expression_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_match_expression_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let enabled = match .thread.x:\n            0: true\n            _: false\n        if enabled:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_do_expression_alias_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_do_expression_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_do_expression_alias.br");
    fs::write(
        &input,
        r#"
kernel DoAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let enabled = do:
            let block_id = .block.x
            block_id == 0
        if enabled:
            output[i] = i + 10
            sync
mut guarded = DoAlias()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_do_expression_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_do_expression_alias_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_do_expression_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_do_expression_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let enabled = do:\n            let lane = .thread.x\n            lane == 0\n        if enabled:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniformly_indexed_collection_alias_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_collection_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_collection_alias.br");
    fs::write(
        &input,
        r#"
kernel CollectionAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        let decisions = [true, false]
        let enabled = decisions[.block.x]
        if enabled:
            output[i + .block.x * .block_dim.x] = i + 10
            sync
mut guarded = CollectionAlias()
kernel:
    guarded(block = 4, grid = 2)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
print "{guarded.output[4]} {guarded.output[5]} {guarded.output[6]} {guarded.output[7]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_collection_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "10 11 12 13\n0 0 0 0"
    );
}

#[test]
fn thread_indexed_collection_alias_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_collection_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_collection_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let decisions = [true, false]\n        let enabled = decisions[.thread.x % 2]\n        if enabled:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_comprehension_alias_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_comprehension_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_comprehension_alias.br");
    fs::write(
        &input,
        r#"
kernel ComprehensionAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        let decisions = [block_id == 0 for block_id in ..<2]
        let enabled = decisions[.block.x]
        if enabled:
            output[i + .block.x * .block_dim.x] = i + 10
            sync
mut guarded = ComprehensionAlias()
kernel:
    guarded(block = 4, grid = 2)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
print "{guarded.output[4]} {guarded.output[5]} {guarded.output[6]} {guarded.output[7]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_comprehension_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "10 11 12 13\n0 0 0 0"
    );
}

#[test]
fn thread_comprehension_alias_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_comprehension_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_comprehension_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let decisions = [block_id == .thread.x for block_id in ..<2]\n        let enabled = decisions[0]\n        if enabled:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_allocated_array_alias_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_allocated_array_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_allocated_array_alias.br");
    fs::write(
        &input,
        r#"
kernel AllocatedAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let [bool] decisions = [..<2]
        let enabled = decisions.length == 2 and not decisions[0]
        if enabled:
            output[i] = i + 10
            sync
mut guarded = AllocatedAlias()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_allocated_array_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_sized_allocated_array_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_allocated_array_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_allocated_array_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let [bool] decisions = [..<(.thread.x + 1)]\n        if decisions.length > 0:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniformly_sliced_collection_alias_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_slice_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_slice_alias.br");
    fs::write(
        &input,
        r#"
kernel SliceAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let decisions = [true, false]
        let selected = decisions[..<1]
        let enabled = selected[0]
        if enabled:
            output[i] = i + 10
            sync
mut guarded = SliceAlias()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_slice_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_sliced_collection_alias_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_slice_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_slice_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let decisions = [true, false]\n        let selected = decisions[..<(.thread.x + 1)]\n        let enabled = selected[0]\n        if enabled:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_dict_and_set_literals_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_dict_set_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_dict_set_alias.br");
    fs::write(
        &input,
        r#"
kernel CollectionLiterals:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let same_dict = {0=true, 1=false} == {1=false, 0=true}
        let same_set = {0, 1, 1} == {1, 0}
        if same_dict and same_set:
            output[i] = i + 10
            sync
mut guarded = CollectionLiterals()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("sync_uniform_dict_set_alias_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("std::collections::HashMap::from"));
    assert!(source.contains("std::collections::HashSet::from"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_dict_set_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn local_dict_lookup_and_set_contains_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_dict_lookup_set_contains");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_dict_lookup_set_contains.br");
    fs::write(
        &input,
        r#"
kernel CollectionLookup:
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        let {int=bool} decisions = {0=true, 1=false}
        let {int} enabled_blocks = {0}
        let enabled = decisions.contains(.block.x) and decisions[.block.x] and enabled_blocks.contains(.block.x) and decisions.length == 2 and enabled_blocks.count == 1 and not decisions.isEmpty
        if enabled:
            output[i + .block.x * .block_dim.x] = i + 10
            sync
mut guarded = CollectionLookup()
kernel:
    guarded(block = 4, grid = 2)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
print "{guarded.output[4]} {guarded.output[5]} {guarded.output[6]} {guarded.output[7]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(
        root.join("local_dict_lookup_set_contains_simd/src/main.rs"),
    )
    .unwrap();
    assert!(source.contains(".get(&__block.x).cloned()"));
    assert!(source.contains(".contains_key(&__block.x)"));
    assert!(source.contains(".contains(&__block.x)"));
    assert!(source.contains("(decisions.len() as i32)"));
    assert!(source.contains("(enabled_blocks.len() as i32)"));
    assert!(source.contains("decisions.is_empty()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_dict_lookup_set_contains_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "10 11 12 13\n0 0 0 0"
    );
}

#[test]
fn thread_set_contains_alias_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_set_contains_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_set_contains_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let enabled_lanes = {0}\n        let enabled = enabled_lanes.contains(.thread.x)\n        if enabled:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_dict_get_alias_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_dict_get_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_dict_get_alias.br");
    fs::write(
        &input,
        r#"
kernel DictGetAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        let {int=bool} decisions = {0=true, 1=false}
        let enabled = decisions.get(.block.x, false)
        if enabled:
            output[i + .block.x * .block_dim.x] = i + 10
            sync
mut guarded = DictGetAlias()
kernel:
    guarded(block = 4, grid = 2)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
print "{guarded.output[4]} {guarded.output[5]} {guarded.output[6]} {guarded.output[7]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_dict_get_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "10 11 12 13\n0 0 0 0"
    );
}

#[test]
fn thread_dict_get_alias_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_dict_get_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_dict_get_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let {int=bool} decisions = {0=true}\n        let enabled = decisions.get(.thread.x, false)\n        if enabled:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_collection_reductions_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_collection_reductions");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_collection_reductions.br");
    fs::write(
        &input,
        r#"
kernel ReductionAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [1, 2, 3]
        let enabled = values.sum() == 6 and values.min() == 1 and values.max() == 3
        if enabled:
            output[i] = i + 10
            sync
mut guarded = ReductionAlias()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_collection_reductions_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_collection_reduction_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_collection_reduction");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_collection_reduction.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let values = [.thread.x, 1]\n        if values.sum() > 0:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_collection_predicates_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_collection_predicates");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_collection_predicates.br");
    fs::write(
        &input,
        r#"
kernel PredicateAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [1, 2, 3]
        let enabled = values.all((value): value > 0) and values.any((value): value == 2)
        if enabled:
            output[i] = i + 10
            sync
mut guarded = PredicateAlias()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_collection_predicates_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_collection_predicate_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_collection_predicate");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_collection_predicate.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let values = [1, 2]\n        if values.any((value): value == .thread.x):\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_collection_transformations_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_collection_transformations");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_collection_transformations.br");
    fs::write(
        &input,
        r#"
kernel TransformAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [1, 2, 3, 4]
        let doubled = values.map((value): value * 2)
        let selected = doubled.filter((value): value >= 4)
        let enabled = doubled.sum() == 20 and selected.sum() == 18
        if enabled:
            output[i] = i + 10
            sync
mut guarded = TransformAlias()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_collection_transformations_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_collection_transformation_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_collection_transformation");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_collection_transformation.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let values = [1, 2]\n        let mapped = values.map((value): value + .thread.x)\n        if mapped.sum() > 0:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn uniform_collection_reduce_can_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_uniform_collection_reduce");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_collection_reduce.br");
    fs::write(
        &input,
        r#"
kernel ReduceAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [1, 2, 3]
        let total = values.reduce(0, (sum, value): sum + value)
        if total == 6:
            output[i] = i + 10
            sync
mut guarded = ReduceAlias()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_collection_reduce_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn thread_collection_reduce_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_collection_reduce");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_collection_reduce.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let values = [1, 2]\n        let total = values.reduce(.thread.x, (sum, value): sum + value)\n        if total > 0:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn local_dict_and_set_mutations_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_dict_set_mutations");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_dict_set_mutations.br");
    fs::write(
        &input,
        r#"
kernel CollectionMutations:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        mut {int=int} values = {0=1}
        values[1] = 10
        values.set(2, 20)
        values.put(3, 30)
        values.remove(0)
        mut {int} selected = {0}
        selected.add(1)
        selected.insert(2)
        selected.remove(0)
        mut copied_values = values
        copied_values[4] = 40
        mut copied_selected = selected
        copied_selected.add(3)
        if copied_values.length == 4 and copied_selected.length == 3 and copied_selected.contains(1):
            output[i] = copied_values[1] + copied_values[2] + copied_values[3] + copied_values[4] + i
mut collections = CollectionMutations()
kernel:
    collections(block = 4)
print "{collections.output[0]} {collections.output[1]} {collections.output[2]} {collections.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_dict_set_mutations_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("values.insert(1, 10)"));
    assert!(source.contains("values.remove(&0)"));
    assert!(source.contains("selected.insert(1)"));
    assert!(source.contains("selected.remove(&0)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_dict_set_mutations_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "100 101 102 103");
}

#[test]
fn local_array_mutation_methods_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_array_mutation_methods");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_mutation_methods.br");
    fs::write(
        &input,
        r#"
kernel ArrayMutations:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        mut values = [10, 20, 30]
        values.push(40)
        values.insert(0, 5)
        let last = values.pop()
        let tail = values.remove(-1)
        let first = values.remove(0)
        output[i] = last + tail + first + values[0] + i
mut arrays = ArrayMutations()
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
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_mutation_methods_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("values.push(40)"));
    assert!(source.contains("values.insert((0) as usize, 5)"));
    assert!(source.contains("values.pop().unwrap_or_default()"));
    assert!(source.contains("__boring_idx < 0"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_mutation_methods_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "85 86 87 88");
}

#[test]
fn local_array_bulk_mutation_methods_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_array_bulk_mutation_methods");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_bulk_mutation_methods.br");
    fs::write(
        &input,
        r#"
kernel ArrayBulkMutations:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        mut values = [3, 1, 2]
        values.append([5, 4])
        values.reverse()
        values.sort()
        let removed = values.remove(-2)
        values.clear()
        values.push(removed + i)
        output[i] = values[0]
mut arrays = ArrayBulkMutations()
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
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_bulk_mutation_methods_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("values.extend(vec![5, 4].iter().cloned())"));
    assert!(source.contains("values.reverse()"));
    assert!(source.contains("values.sort_by("));
    assert!(source.contains("values.clear()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_bulk_mutation_methods_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "4 5 6 7");
}

#[test]
fn local_collection_query_methods_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_collection_query_methods");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_collection_query_methods.br");
    fs::write(
        &input,
        r#"
kernel CollectionQueries:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [10, 20, 30]
        let first = values.first() else 0
        let last = values.last() else 0
        let reversed = values.reversed()
        let empty = values.isEmpty()
        output[i] = first + last + reversed[0] + values.count() + (if empty: 100 else: 0) + i
mut queries = CollectionQueries()
kernel:
    queries(block = 4)
print "{queries.output[0]} {queries.output[1]} {queries.output[2]} {queries.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_collection_query_methods_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("values.first().cloned()"));
    assert!(source.contains("values.last().cloned()"));
    assert!(source.contains("values.iter().rev().cloned().collect::<Vec<_>>()"));
    assert!(source.contains("values.is_empty()"));
    assert!(source.contains("values.len() as i32"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_collection_query_methods_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "73 74 75 76");
}

#[test]
fn local_array_search_methods_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_array_search_methods");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_search_methods.br");
    fs::write(
        &input,
        r#"
kernel ArraySearches:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [10, 20, 30, 40]
        let selected = values.count((value): value >= 20)
        let found = values.find((value): value > 25) else 0
        let index = values.indexOf(30) else -1
        let predicate_index = values.indexOf((value): value == 40) else -1
        output[i] = selected + found + index + predicate_index + i
mut searches = ArraySearches()
kernel:
    searches(block = 4)
print "{searches.output[0]} {searches.output[1]} {searches.output[2]} {searches.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_search_methods_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("iter().cloned().filter"));
    assert!(source.contains("iter().cloned().find"));
    assert!(source.contains("__boring_item == &(30)"));
    assert!(source.contains("iter().cloned().position"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_search_methods_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "38 39 40 41");
}

#[test]
fn local_array_sequence_transforms_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_array_sequence_transforms");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_sequence_transforms.br");
    fs::write(
        &input,
        r#"
kernel ArraySequences:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [1, 2, 3, 4, 5]
        let taken = values.take(3)
        let dropped = values.drop(2)
        let sliced = values.slice(-4, -1)
        let expanded = values.flatMap((value): [value, value * 10])
        let nested = [[1, 2], [3, 4]]
        let flattened = nested.flat()
        output[i] = taken[2] + dropped[0] + sliced[2] + expanded[3] + flattened[2] + i
mut sequences = ArraySequences()
kernel:
    sequences(block = 4)
print "{sequences.output[0]} {sequences.output[1]} {sequences.output[2]} {sequences.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_sequence_transforms_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains(".take((3) as usize)"));
    assert!(source.contains(".skip((2) as usize)"));
    assert!(source.contains(".clamp(__boring_start, __boring_len)"));
    assert!(source.contains(".flat_map("));
    assert!(source.contains(".flatten().collect::<Vec<_>>()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_sequence_transforms_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "33 34 35 36");
}

#[test]
fn local_array_structural_transforms_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_array_structural_transforms");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_structural_transforms.br");
    fs::write(
        &input,
        r#"
kernel ArrayStructures:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [3, 1, 2]
        let sorted = values.sorted()
        let zipped = sorted.zip([10, 20, 30])
        let enumerated = sorted.enumerate()
        output[i] = sorted[0] + zipped.count() + enumerated.count() + i
mut structures = ArrayStructures()
kernel:
    structures(block = 4)
print "{structures.output[0]} {structures.output[1]} {structures.output[2]} {structures.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_structural_transforms_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("__boring_sorted.sort_by"));
    assert!(source.contains(".zip(vec![10, 20, 30].iter().cloned())"));
    assert!(source.contains("__boring_index as i32"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_structural_transforms_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "7 8 9 10");
}

#[test]
fn local_array_key_sorting_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_local_array_key_sorting");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_key_sorting.br");
    fs::write(
        &input,
        r#"
kernel ArrayKeySorting:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        mut values = [3, 1, 2]
        values.sortBy((value): -value)
        let ascending = values.sortedBy((value): value)
        output[i] = values[0] + ascending[0] + i
mut sorting = ArrayKeySorting()
kernel:
    sorting(block = 4)
print "{sorting.output[0]} {sorting.output[1]} {sorting.output[2]} {sorting.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_key_sorting_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("let __boring_key_a"));
    assert!(source.contains("let mut __boring_sorted = values.clone()"));
    assert!(source.contains("partial_cmp(&__boring_key_b)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_key_sorting_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "4 5 6 7");
}

#[test]
fn local_array_cursor_protocol_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_local_array_cursor_protocol");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_cursor_protocol.br");
    fs::write(
        &input,
        r#"
kernel ArrayCursors:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [10, 20, 30]
        let first = values.firstIndex() else -1
        let second = values.nextIndex(first) else -1
        let value = values.getAt(first)
        let trimmed = values.removeAt(first)
        output[i] = first + second + value + trimmed[0] + i
mut cursors = ArrayCursors()
kernel:
    cursors(block = 4)
print "{cursors.output[0]} {cursors.output[1]} {cursors.output[2]} {cursors.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_cursor_protocol_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("Some(0i32)"));
    assert!(source.contains("let __boring_next"));
    assert!(source.contains("let mut __boring_collection = values.clone()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_cursor_protocol_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "31 32 33 34");
}

#[test]
fn local_array_structural_mutations_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_array_structural_mutations");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_structural_mutations.br");
    fs::write(
        &input,
        r#"
kernel ArrayStructuralMutations:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        mut values = [10, 20, 30, 40, 50]
        values.move(1, 3)
        values.removeIndices([0, 2, 4])
        output[i] = values[0] + values[1] + i
mut mutations = ArrayStructuralMutations()
kernel:
    mutations(block = 4)
print "{mutations.output[0]} {mutations.output[1]} {mutations.output[2]} {mutations.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_structural_mutations_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("let __boring_item = values.remove"));
    assert!(source.contains("removeIndices indices must be unique and strictly ascending"));
    assert!(source.contains("__boring_indices.iter().rev()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_structural_mutations_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "50 51 52 53");
}

#[test]
fn local_array_iteration_and_join_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_array_iteration_and_join");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_iteration_and_join.br");
    fs::write(
        &input,
        r#"
kernel ArrayIteration:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [1, 2, 3]
        mut total = 0
        values.forEach (value):
            total += value
        let words = ["a", "bb", "c"]
        let joined = words.join("-")
        output[i] = total + int(joined.len()) + i
mut iteration = ArrayIteration()
kernel:
    iteration(block = 4)
print "{iteration.output[0]} {iteration.output[1]} {iteration.output[2]} {iteration.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_iteration_and_join_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("iter().cloned().for_each"));
    assert!(source.contains("words.join(String::from(\"-\").as_str())"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_iteration_and_join_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "12 13 14 15");
}

#[test]
fn local_dict_and_set_cursor_protocol_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_dict_and_set_cursor_protocol");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_dict_and_set_cursor_protocol.br");
    fs::write(
        &input,
        r#"
kernel CollectionCursors:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let dictionary = {"a" = 10, "b" = 20}
        let key = dictionary.firstIndex() else ""
        let value = dictionary.getAt(key)
        let smaller_dictionary = dictionary.removeAt(key)
        let values = {10, 20, 30}
        let cursor = values.firstIndex() else -1
        let element = values.getAt(cursor)
        let smaller_set = values.removeAt(cursor)
        output[i] = dictionary.count() + smaller_dictionary.count() + values.count() + smaller_set.count() + int(value > 0) + int(element > 0) + i
mut cursors = CollectionCursors()
kernel:
    cursors(block = 4)
print "{cursors.output[0]} {cursors.output[1]} {cursors.output[2]} {cursors.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_dict_and_set_cursor_protocol_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("dictionary.keys().next().cloned()"));
    assert!(source.contains("getAt key not found"));
    assert!(source.contains("values.iter().nth"));
    assert!(source.contains("let __boring_element"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_dict_and_set_cursor_protocol_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn local_array_cleanup_and_fold_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_array_cleanup_and_fold");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_array_cleanup_and_fold.br");
    fs::write(
        &input,
        r#"
kernel ArrayCleanup:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        mut values = [1, 2, 2, 3, 4, 4]
        values.dedup()
        values.retain((value): value % 2 == 1)
        values.append([5, 6])
        values.truncate(3)
        let total = values.fold(0, (sum, value): sum + value)
        output[i] = total + i
mut cleanup = ArrayCleanup()
kernel:
    cleanup(block = 4)
print "{cleanup.output[0]} {cleanup.output[1]} {cleanup.output[2]} {cleanup.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_array_cleanup_and_fold_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("values.dedup()"));
    assert!(source.contains("values.retain("));
    assert!(source.contains("values.truncate((3) as usize)"));
    assert!(source.contains(".iter().cloned().fold(0"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_array_cleanup_and_fold_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "9 10 11 12");
}

#[test]
fn empty_local_array_operations_use_defaults_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_empty_local_array_operations");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("empty_local_array_operations.br");
    fs::write(
        &input,
        r#"
kernel EmptyArrayOperations:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        mut [int] values = []
        let popped = values.pop()
        let minimum = values.min()
        let maximum = values.max()
        output[i] = popped + minimum + maximum + i
mut operations = EmptyArrayOperations()
kernel:
    operations(block = 4)
print "{operations.output[0]} {operations.output[1]} {operations.output[2]} {operations.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("empty_local_array_operations_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("values.pop().unwrap_or_default()"));
    assert!(source.contains("values.iter().cloned().min().unwrap_or_default()"));
    assert!(source.contains("values.iter().cloned().max().unwrap_or_default()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("empty_local_array_operations_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "0 1 2 3");
}

#[test]
fn local_dict_map_and_filter_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_local_dict_map_filter");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_dict_map_filter.br");
    fs::write(
        &input,
        r#"
kernel DictTransforms:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = {0 = 1, 1 = 2, 2 = 3}
        let doubled = values.map((key, value): value * 2)
        let filtered = doubled.filter((key, value): key > 0)
        let high = doubled.get(2) else 0
        let middle = filtered.get(1) else 0
        output[i] = high + middle + i
mut transforms = DictTransforms()
kernel:
    transforms(block = 4)
print "{transforms.output[0]} {transforms.output[1]} {transforms.output[2]} {transforms.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_dict_map_filter_simd/src/main.rs")).unwrap();
    assert!(source.contains("collect::<std::collections::HashMap<_, _>>()"));
    assert!(source.contains("let key = __boring_entry.0.clone()"));
    assert!(source.contains("let value = __boring_entry.1.clone()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_dict_map_filter_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn local_set_map_and_filter_produce_arrays_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_local_set_map_filter");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_set_map_filter.br");
    fs::write(
        &input,
        r#"
kernel SetTransforms:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = {1, 2, 3, 4}
        let doubled = values.map((value): value * 2)
        let even = values.filter((value): value % 2 == 0)
        output[i] = doubled.sum() + even.sum() + i
mut transforms = SetTransforms()
kernel:
    transforms(block = 4)
print "{transforms.output[0]} {transforms.output[1]} {transforms.output[2]} {transforms.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_set_map_filter_simd/src/main.rs")).unwrap();
    assert!(source.contains("values.iter().cloned().map"));
    assert!(source.contains("values.iter().cloned().filter"));
    assert!(source.contains("doubled.iter().cloned().sum::<i32>()"));
    assert!(source.contains("even.iter().cloned().sum::<i32>()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_set_map_filter_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "26 27 28 29");
}

#[test]
fn local_string_alias_methods_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_local_string_alias_methods");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_string_alias_methods.br");
    fs::write(
        &input,
        r#"
kernel StringAliases:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let text = " Hello "
        let trimmed = text.trim()
        let upper = trimmed.toUpperCase()
        let lower = upper.toLowerCase()
        output[i] = trimmed.length() + int(upper == "HELLO") + int(lower.startsWith("he")) + int(lower.endsWith("lo")) + i
mut aliases = StringAliases()
kernel:
    aliases(block = 4)
print "{aliases.output[0]} {aliases.output[1]} {aliases.output[2]} {aliases.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_string_alias_methods_simd/src/main.rs")).unwrap();
    assert!(source.contains("text.trim().to_string()"));
    assert!(source.contains("trimmed.to_uppercase()"));
    assert!(source.contains("upper.to_lowercase()"));
    assert!(source.contains("lower.starts_with(String::from(\"he\").as_str())"));
    assert!(source.contains("lower.ends_with(String::from(\"lo\").as_str())"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_string_alias_methods_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "8 9 10 11");
}

#[test]
fn local_string_conversion_methods_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_string_conversion_methods");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_string_conversion_methods.br");
    fs::write(
        &input,
        r#"
kernel StringConversions:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let text = " 10,20 "
        let parts = text.trim().split(",")
        let digits = parts[0].chars()
        let parsed = parts[1].parseInt() else 0
        let decimal = "2.5".parseFloat() else 0.0
        output[i] = parsed + digits.count() + int(decimal) + i
mut conversions = StringConversions()
kernel:
    conversions(block = 4)
print "{conversions.output[0]} {conversions.output[1]} {conversions.output[2]} {conversions.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_string_conversion_methods_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("collect::<Vec<String>>()"));
    assert!(source.contains("parts[(0) as usize].chars()"));
    assert!(source.contains("parse::<i32>().ok()"));
    assert!(source.contains("parse::<f64>().ok()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_string_conversion_methods_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "24 25 26 27");
}

#[test]
fn local_string_search_and_transform_methods_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_string_search_transforms");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_string_search_transforms.br");
    fs::write(
        &input,
        r#"
kernel StringTransforms:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let text = "  héllo  ".trimStart().trimEnd()
        let middle = text.slice(-4, -1)
        let replaced = middle.replace("é", "a")
        let repeated = replaced.repeat(2)
        let position = repeated.indexOf("ll") else -1
        output[i] = repeated.length() + position + int(repeated.contains("all")) + i
mut transforms = StringTransforms()
kernel:
    transforms(block = 4)
print "{transforms.output[0]} {transforms.output[1]} {transforms.output[2]} {transforms.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_string_search_transforms_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("trim_start().to_string().trim_end().to_string()"));
    assert!(source.contains("__boring_chars"));
    assert!(source.contains(".replace(String::from(\"é\").as_str()"));
    assert!(source.contains(".repeat((2) as usize)"));
    assert!(source.contains(".find(String::from(\"ll\").as_str())"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_string_search_transforms_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "8 9 10 11");
}

#[test]
fn collection_and_string_pipelines_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_collection_string_pipelines");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("collection_string_pipelines.br");
    fs::write(
        &input,
        r#"
kernel Pipelines:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = [1, 2, 3]
        let transformed = values |> map((value): value * 2) |> reversed()
        let text = " hi " |> trim() |> upper()
        output[i] = transformed[0] + int(text == "HI") + i
mut pipelines = Pipelines()
kernel:
    pipelines(block = 4)
print "{pipelines.output[0]} {pipelines.output[1]} {pipelines.output[2]} {pipelines.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("collection_string_pipelines_simd/src/main.rs")).unwrap();
    assert!(source.contains("iter().cloned().map"));
    assert!(source.contains("iter().rev().cloned()"));
    assert!(source.contains("trim().to_string().to_uppercase()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("collection_string_pipelines_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "7 8 9 10");
}

#[test]
fn local_tuple_queries_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_local_tuple_queries");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_tuple_queries.br");
    fs::write(
        &input,
        r#"
kernel TupleQueries:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = (10, 20, 30)
        let alias = values
        output[i] = alias.length() + alias.first() + alias.last() + int(alias.isEmpty()) + i
mut queries = TupleQueries()
kernel:
    queries(block = 4)
print "{queries.output[0]} {queries.output[1]} {queries.output[2]} {queries.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_tuple_queries_simd/src/main.rs")).unwrap();
    assert!(source.contains("alias.0.clone()"));
    assert!(source.contains("alias.2.clone()"));
    assert!(source.contains("3 + alias.0.clone()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_tuple_queries_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "43 44 45 46");
}

#[test]
fn local_tuple_functional_methods_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_tuple_functional_methods");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_tuple_functional_methods.br");
    fs::write(
        &input,
        r#"
kernel TupleFunctions:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let values = (1, 2, 3)
        let doubled = values.map((value): value * 2)
        let all_positive = doubled.all((value): value > 0)
        let has_four = doubled.any((value): value == 4)
        output[i] = doubled.first() + doubled.last() + int(all_positive) + int(has_four) + i
mut functions = TupleFunctions()
kernel:
    functions(block = 4)
print "{functions.output[0]} {functions.output[1]} {functions.output[2]} {functions.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("local_tuple_functional_methods_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("let value = values.0.clone()"));
    assert!(source.contains("let value = doubled.2.clone()"));
    assert!(source.contains(" && "));
    assert!(source.contains(" || "));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_tuple_functional_methods_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn local_dict_and_set_transformations_run_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_local_dict_set_transformations");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("local_dict_set_transformations.br");
    fs::write(
        &input,
        r#"
kernel CollectionTransforms:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let {int=int} values = {0=10, 1=20}
        let present = values.get(0) else -1
        let missing = values.get(9, 30)
        let keys = values.keys()
        let entries = values.values()
        let {int} left = {1, 2}
        let {int} right = {2, 3}
        let united = left.union(right)
        let shared = left.intersection(right)
        let distinct = left.difference(right)
        let flattened = united.toArray()
        if united.contains(3) and shared.contains(2) and distinct.contains(1):
            output[i] = present + missing + united.length + shared.length + distinct.length + flattened.length + keys.length + entries.length + i
mut collections = CollectionTransforms()
kernel:
    collections(block = 4)
print "{collections.output[0]} {collections.output[1]} {collections.output[2]} {collections.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(
        root.join("local_dict_set_transformations_simd/src/main.rs"),
    )
    .unwrap();
    assert!(source.contains(".get(&0).cloned()"));
    assert!(source.contains(".union(&right)"));
    assert!(source.contains(".intersection(&right)"));
    assert!(source.contains(".difference(&right)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("local_dict_set_transformations_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "52 53 54 55");
}

#[test]
fn inherited_uniform_aliases_can_guard_later_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_inherited_uniform_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_inherited_uniform_alias.br");
    fs::write(
        &input,
        r#"
kernel AliasSequence:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let block_id = .block.x
        tile[i] = i
        sync
        if .block_dim.x == 4:
            tile[i] = tile[i] + 1
            sync
        let first_block = block_id == 0
        let selected = first_block == true
        if selected:
            tile[i] = tile[i] + 10
            sync
        output[i] = tile[i]
mut phased = AliasSequence()
kernel:
    phased(block = 4)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_inherited_uniform_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "11 12 13 14");
}

#[test]
fn inherited_thread_alias_for_later_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_inherited_thread_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_inherited_thread_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let lane = .thread.x\n        if .block.x == 0:\n            sync\n        let selected = lane == 0\n        if selected:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn immutable_uniform_aliases_drive_for_and_match_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_uniform_alias_controls");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_alias_controls.br");
    fs::write(
        &input,
        r#"
kernel AliasLoop:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let rounds = .grid_dim.x + 1
        let selector = .block.x
        tile[i] = i
        sync
        for round in 0..<rounds:
            tile[i] = tile[i] + 10 + round
            sync
        match selector:
            0:
                tile[i] = tile[i] + 30
                sync
            _:
                tile[i] = 0
                sync
        output[i] = tile[i]
mut looped = AliasLoop()
kernel:
    looped(block = 4)
print "{looped.output[0]} {looped.output[1]} {looped.output[2]} {looped.output[3]}"
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
        .arg(root.join("sync_uniform_alias_controls_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "51 52 53 54"
    );
}

#[test]
fn immutable_thread_alias_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_thread_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_thread_alias.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let lane = .thread.x\n        if lane == 0:\n            sync\n",
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
fn collective_if_followed_by_match_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_if_then_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_then_match.br");
    fs::write(
        &input,
        r#"
kernel MixedPhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        tile[i] = i
        sync
        if .block_dim.x == 4:
            tile[i] = tile[i] + 1
            sync
        match .block.x:
            0:
                tile[i] = tile[i] + 10
                sync
            _:
                tile[i] = tile[i] + 20
                sync
        output[i + .block.x * .block_dim.x] = tile[i]
mut phased = MixedPhases()
kernel:
    phased(block = 4, grid = 2)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
print "{phased.output[4]} {phased.output[5]} {phased.output[6]} {phased.output[7]}"
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
    let source = fs::read_to_string(root.join("sync_if_then_match_simd/src/main.rs")).unwrap();
    assert!(source.contains("if (__block_dim.x == 4)"));
    assert!(source.contains("match __block.x"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_if_then_match_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "11 12 13 14\n21 22 23 24"
    );
}

#[test]
fn collective_if_followed_by_guard_and_if_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_if_guard_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_guard_if.br");
    fs::write(
        &input,
        r#"
kernel GuardedSequence:
    let int? selected = some(60)
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        sync
        if .block.x == 0:
            tile[i] = tile[i] + 1
            sync
        guard let value = selected else:
            sync
            return
        if .block_dim.x == 4:
            output[i] = value + tile[i]
            sync
mut phased = GuardedSequence()
kernel:
    phased(block = 4)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_if_guard_if_simd/src/main.rs")).unwrap();
    assert!(source.contains("Some(value) =>"));
    assert!(source.contains("if (__block_dim.x == 4)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_if_guard_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "61 62 63 64");
}

#[test]
fn collective_if_followed_by_if_let_and_match_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_if_if_let_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_if_let_match.br");
    fs::write(
        &input,
        r#"
kernel IfLetSequence:
    let int? selected = some(70)
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        sync
        if .block_dim.x == 4:
            tile[i] = tile[i] + 1
            sync
        if let value = selected:
            output[i] = value + tile[i]
            sync
        else:
            output[i] = 99
            sync
        match .block.x:
            0:
                output[i] = output[i] + 10
                sync
            _:
                output[i] = 0
                sync
mut phased = IfLetSequence()
kernel:
    phased(block = 4)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_if_if_let_match_simd/src/main.rs")).unwrap();
    assert!(source.contains("if let Some(value)"));
    assert!(source.contains("match __block.x"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_if_if_let_match_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "81 82 83 84");
}

#[test]
fn collective_if_followed_by_for_and_match_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_if_for_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_for_match.br");
    fs::write(
        &input,
        r#"
kernel LoopSequence:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        sync
        if .block_dim.x == 4:
            tile[i] = tile[i] + 1
            sync
        for round in 0..<2:
            tile[i] = tile[i] + 10 + round
            sync
        match .block.x:
            0:
                tile[i] = tile[i] + 10
                sync
            _:
                tile[i] = 0
                sync
        output[i] = tile[i]
mut phased = LoopSequence()
kernel:
    phased(block = 4)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_if_for_match_simd/src/main.rs")).unwrap();
    assert!(source.contains("for round in 0..2"));
    assert!(source.contains("match __block.x"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_if_for_match_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "32 33 34 35");
}

#[test]
fn collective_if_followed_by_while_and_match_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_if_while_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_while_match.br");
    fs::write(
        &input,
        r#"
kernel WhileSequence:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i + 1
        sync
        if .block_dim.x == 4:
            tile[i] = tile[i] + 1
            sync
        var int stride = 2
        while stride > 0:
            tile[i] = tile[i] + stride
            sync
            stride = stride - 1
        match .block.x:
            0:
                output[i] = tile[i]
                sync
            _:
                output[i] = 0
                sync
mut phased = WhileSequence()
kernel:
    phased(block = 4)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("sync_if_while_match_simd/src/main.rs")).unwrap();
    assert!(source.contains("while (stride > 0)"));
    assert!(source.contains("stride = (stride - 1);"));
    assert!(source.contains("match __block.x"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_if_while_match_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "5 6 7 8");
}

#[test]
fn collective_match_followed_by_for_and_if_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_match_for_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_match_for_if.br");
    fs::write(
        &input,
        r#"
kernel ReverseSequence:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        sync
        match .block.x:
            0:
                tile[i] = tile[i] + 10
                sync
            _:
                tile[i] = 0
                sync
        for round in 0..<2:
            tile[i] = tile[i] + 10 + round
            sync
        if .block_dim.x == 4:
            tile[i] = tile[i] + 1
            sync
        output[i] = tile[i]
mut phased = ReverseSequence()
kernel:
    phased(block = 4)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_match_for_if_simd/src/main.rs")).unwrap();
    assert!(source.contains("match __block.x"));
    assert!(source.contains("for round in 0..2"));
    assert!(source.contains("if (__block_dim.x == 4)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_match_for_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "32 33 34 35");
}

#[test]
fn immutable_literals_and_grid_dim_guard_collective_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_uniform_literals");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_uniform_literals.br");
    fs::write(
        &input,
        r#"
kernel LiteralPhases:
    let int? selected = nil
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        sync
        match "ready":
            "ready":
                tile[i] = tile[i] + 1
                sync
            _:
                tile[i] = 0
                sync
        if selected is nil:
            tile[i] = tile[i] + 10
            sync
        if .grid_dim.x == 1:
            tile[i] = tile[i] + 100
            sync
        output[i] = tile[i]
mut phased = LiteralPhases()
kernel:
    phased(block = 4)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_uniform_literals_simd/src/main.rs")).unwrap();
    assert!(source.contains("String::from(\"ready\")).as_str()"));
    assert!(source.contains("__grid_dim.x == 1"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_uniform_literals_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "111 112 113 114");
}

#[test]
fn divergent_if_after_collective_match_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_match_then_divergent_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_match_then_divergent_if.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        match .block.x:\n            0:\n                sync\n            _:\n                pass\n        if .thread.x == 0:\n            sync\n",
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
fn divergent_while_after_collective_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_if_then_divergent_while");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_then_divergent_while.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        if .block.x == 0:\n            sync\n        var int stride = 2\n        while stride > .thread.x:\n            sync\n            stride = stride - 1\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn divergent_for_after_collective_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_if_then_divergent_for");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_then_divergent_for.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        if .block.x == 0:\n            sync\n        for round in 0..<(.thread.x + 1):\n            sync\n",
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
fn divergent_if_let_after_collective_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_if_then_divergent_if_let");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_then_divergent_if_let.br");
    fs::write(
        &input,
        "kernel Divergent:\n    let int? selected = some(0)\n    def ():\n        if .block.x == 0:\n            sync\n        if let value = selected, value == .thread.x:\n            sync\n",
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
fn divergent_guard_after_collective_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_if_then_divergent_guard");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_then_divergent_guard.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        if .block.x == 0:\n            sync\n        guard .thread.x == 0 else:\n            sync\n            return\n",
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
fn divergent_match_after_collective_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_if_then_divergent_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_then_divergent_match.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        if .block.x == 0:\n            sync\n        match .thread.x:\n            0:\n                sync\n            _:\n                pass\n",
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
fn divergent_if_after_collective_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_successive_if_divergence");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_successive_if_divergence.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        if .block.x == 0:\n            sync\n        if .thread.x == 0:\n            sync\n",
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
fn collective_if_let_elif_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_if_let_elif");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_if_let_elif.br");
    fs::write(
        &input,
        r#"
kernel IfLetElifPhases:
    let int? selected = nil
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        if let value = selected:
            output[i] = value
            sync
        elif .block.x == 0:
            output[i] = 50 + i
            sync
        else:
            output[i] = 99
            sync
        output[i] = output[i] + 1
        sync
mut conditional = IfLetElifPhases()
kernel:
    conditional(block = 4)
print "{conditional.output[0]} {conditional.output[1]} {conditional.output[2]} {conditional.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_if_let_elif_simd/src/main.rs")).unwrap();
    assert!(source.contains("if (__block.x == 0)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_if_let_elif_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "51 52 53 54");
}

#[test]
fn thread_divergent_if_let_elif_with_sync_remains_rejected_on_simd() {
    let root =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_divergent_if_let_elif");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_if_let_elif.br");
    fs::write(
        &input,
        "kernel Divergent:\n    let int? selected = nil\n    def ():\n        if let value = selected:\n            sync\n        elif .thread.x == 0:\n            sync\n",
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
fn thread_divergent_if_let_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_divergent_if_let");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_if_let.br");
    fs::write(
        &input,
        "kernel Divergent:\n    let int? selected = some(0)\n    def ():\n        if let value = selected, value == .thread.x:\n            sync\n",
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
fn block_uniform_match_guard_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_match_guard");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_match_guard.br");
    fs::write(
        &input,
        r#"
kernel GuardedMatch:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i + 1
        sync
        match .block.x:
            0 if .block_dim.x == 4:
                output[i] = tile[3 - i]
                sync
            _:
                output[i] = 99
                sync
mut matched = GuardedMatch()
kernel:
    matched(block = 4)
print "{matched.output[0]} {matched.output[1]} {matched.output[2]} {matched.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_match_guard_simd/src/main.rs")).unwrap();
    assert!(source.contains("__block_dim.x == 4"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_match_guard_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "4 3 2 1");
}

#[test]
fn pattern_bound_uniform_match_guard_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_bound_match_guard");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_bound_match_guard.br");
    fs::write(
        &input,
        r#"
kernel BoundGuardMatch:
    let int? selected = some(5)
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        match selected:
            Some(value) if value == 5:
                output[i] = value + i
                sync
            _:
                output[i] = 99
                sync
mut matched = BoundGuardMatch()
kernel:
    matched(block = 4)
print "{matched.output[0]} {matched.output[1]} {matched.output[2]} {matched.output[3]}"
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
    let source =
        fs::read_to_string(root.join("sync_bound_match_guard_simd/src/main.rs")).unwrap();
    assert!(source.contains("Some(value) if (value == 5)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_bound_match_guard_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "5 6 7 8");
}

#[test]
fn pattern_bound_divergent_match_guard_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_bound_divergent_match_guard");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_bound_divergent_match_guard.br");
    fs::write(
        &input,
        "kernel Divergent:\n    let int? selected = some(0)\n    def ():\n        match selected:\n            Some(value) if value == .thread.x:\n                sync\n            _:\n                pass\n",
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
fn thread_divergent_match_guard_with_sync_remains_rejected_on_simd() {
    let root =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_divergent_match_guard");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_match_guard.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        match .block.x:\n            0 if .thread.x == 0:\n                sync\n            _:\n                pass\n",
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
fn uniform_const_field_condition_can_guard_a_simd_barrier() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_const_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_const_if.br");
    fs::write(
        &input,
        r#"
kernel ConstIf:
    let bool enabled = true
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        if enabled:
            let i = .thread.x
            tile[i] = i + 1
            sync
            output[i] = tile[3 - i]
mut conditional = ConstIf()
kernel:
    conditional(block = 4)
print "{conditional.output[0]} {conditional.output[1]} {conditional.output[2]} {conditional.output[3]}"
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
        .arg(root.join("sync_const_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "4 3 2 1");
}

#[test]
fn collective_guard_with_prefix_and_suffix_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_guard_context");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_guard_context.br");
    fs::write(
        &input,
        r#"
kernel GuardPhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        tile[i] = i + 1
        sync
        guard .block.x == 0 else:
            tile[i] = tile[i] + 20
            sync
            output[i + 4] = tile[i]
            return
        tile[i] = tile[i] + 10
        sync
        output[i] = tile[i]
mut guarded = GuardPhases()
kernel:
    guarded(block = 4, grid = 2)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
print "{guarded.output[4]} {guarded.output[5]} {guarded.output[6]} {guarded.output[7]}"
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
    let source = fs::read_to_string(root.join("sync_guard_context_simd/src/main.rs")).unwrap();
    assert!(source.contains("if !((__block.x == 0))"));
    assert!(source.contains("__boring_barrier_active[__boring_barrier_thread] = false"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_guard_context_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "11 12 13 14\n21 22 23 24"
    );
}

#[test]
fn collective_guard_with_nested_if_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_guard_nested_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_guard_nested_if.br");
    fs::write(
        &input,
        r#"
kernel NestedGuard:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        guard .block.x == 1 else:
            if .block_dim.x == 4:
                tile[i] = tile[i] + 10
                sync
            output[i] = tile[i]
            return
mut nested = NestedGuard()
kernel:
    nested(block = 4)
print "{nested.output[0]} {nested.output[1]} {nested.output[2]} {nested.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_guard_nested_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn collective_guard_with_nested_divergent_if_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_guard_nested_divergent_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_guard_nested_divergent_if.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        guard .block.x == 0 else:\n            if .thread.x == 0:\n                sync\n            return\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn collective_guard_let_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_guard_let");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_guard_let.br");
    fs::write(
        &input,
        r#"
kernel GuardLetPhases:
    let int? selected = some(10)
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        guard let value = selected else:
            sync
            return
        output[i] = value + i
        sync
mut guarded = GuardLetPhases()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
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
    let source = fs::read_to_string(root.join("sync_guard_let_simd/src/main.rs")).unwrap();
    assert!(source.contains("Some(value) =>"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_guard_let_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 11 12 13");
}

#[test]
fn collective_guard_let_binding_guards_later_nested_sync_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_guard_let_nested_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_guard_let_nested_if.br");
    fs::write(
        &input,
        r#"
kernel BoundGuard:
    let int? selected = some(2)
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        guard let value = selected else:
            sync
            return
        if value == 2:
            output[i] = value + i + 10
            sync
mut guarded = BoundGuard()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_guard_let_nested_if_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "12 13 14 15");
}

#[test]
fn collective_guard_let_binding_with_divergent_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_guard_let_nested_divergent_if");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_guard_let_nested_divergent_if.br");
    fs::write(
        &input,
        "kernel Divergent:\n    let int? selected = some(0)\n    def ():\n        guard let value = selected else:\n            sync\n            return\n        if value == .thread.x:\n            sync\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn collective_guard_let_pattern_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_guard_let_pattern");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_guard_let_pattern.br");
    fs::write(
        &input,
        r#"
kernel PatternGuardPhases:
    let int? selected = some(20)
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        guard let Some(value) = selected else:
            sync
            return
        output[i] = value + i
        sync
mut guarded = PatternGuardPhases()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
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
    let source =
        fs::read_to_string(root.join("sync_guard_let_pattern_simd/src/main.rs")).unwrap();
    assert!(source.contains("Some(value) =>"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_guard_let_pattern_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "20 21 22 23");
}

#[test]
fn collective_multiclause_guard_let_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_guard_multiclause");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_guard_multiclause.br");
    fs::write(
        &input,
        r#"
kernel MultiGuardPhases:
    let int? selected = some(30)
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        guard let value = selected, value == 30 else:
            sync
            return
        output[i] = value + i
        sync
mut guarded = MultiGuardPhases()
kernel:
    guarded(block = 4)
print "{guarded.output[0]} {guarded.output[1]} {guarded.output[2]} {guarded.output[3]}"
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
    let source =
        fs::read_to_string(root.join("sync_guard_multiclause_simd/src/main.rs")).unwrap();
    assert!(source.contains("Some(value) =>"));
    assert!(source.contains("if (value == 30)"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("sync_guard_multiclause_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "30 31 32 33");
}

#[test]
fn thread_divergent_multiclause_guard_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_guard_multiclause");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_guard_multiclause.br");
    fs::write(
        &input,
        "kernel Divergent:\n    let int? selected = some(1)\n    def ():\n        guard let value = selected, .thread.x == value else:\n            sync\n            return\n",
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
fn thread_divergent_guard_let_pattern_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_sync_divergent_guard_let_pattern");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_guard_let_pattern.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let int? candidate = some(.thread.x)\n        guard let Some(value) = candidate else:\n            sync\n            return\n",
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
fn thread_divergent_guard_let_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_divergent_guard_let");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_guard_let.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        let int? candidate = some(.thread.x)\n        guard let value = candidate else:\n            sync\n            return\n",
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
fn thread_divergent_guard_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_divergent_guard");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_guard.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        guard .thread.x == 0 else:\n            sync\n            return\n",
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
fn thread_divergent_match_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_divergent_match");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_match.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        match .thread.x:\n            0:\n                sync\n            _:\n                pass\n",
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
fn thread_divergent_for_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_sync_divergent_for");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("sync_divergent_for.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        for round in 0..<(.thread.x + 1):\n            sync\n",
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
fn collective_while_let_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_collective_while_let_sync");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("collective_while_let_sync.br");
    fs::write(
        &input,
        r#"
kernel WhileLetPhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        var int? phase = some(0)
        while let value = phase:
            tile[i] = tile[i] + value
            sync
            phase = if value < 2: some(value + 1) else: nil
        output[i] = tile[i]
mut phased = WhileLetPhases()
kernel:
    phased(block = 4)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let source = fs::read_to_string(root.join("collective_while_let_sync_simd/src/main.rs"))
        .unwrap();
    assert!(source.contains("while let Some(value) = phase"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("collective_while_let_sync_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "3 4 5 6");
}

#[test]
fn thread_divergent_while_let_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_divergent_while_let_sync");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("divergent_while_let_sync.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        var int? phase = some(.thread.x)\n        while let value = phase:\n            sync\n            phase = if value > 0: some(value - 1) else: nil\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn nested_collective_while_let_with_sync_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_nested_collective_while_let_sync");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("nested_collective_while_let_sync.br");
    fs::write(
        &input,
        r#"
kernel NestedWhileLet:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        tile[i] = i
        if .block.x == 0:
            var int? phase = some(0)
            while let Some(value) = phase:
                tile[i] = tile[i] + value
                sync
                phase = if value < 2: some(value + 1) else: nil
        else:
            tile[i] = tile[i] + 20
            sync
        output[i + .block.x * .block_dim.x] = tile[i]
mut phased = NestedWhileLet()
kernel:
    phased(block = 4, grid = 2)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
print "{phased.output[4]} {phased.output[5]} {phased.output[6]} {phased.output[7]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("nested_collective_while_let_sync_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "3 4 5 6\n20 21 22 23"
    );
}

#[test]
fn nested_thread_while_let_with_sync_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_nested_divergent_while_let_sync");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("nested_divergent_while_let_sync.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        if .block.x == 0:\n            var int? phase = some(.thread.x)\n            while let value = phase:\n                sync\n                phase = if value > 0: some(value - 1) else: nil\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn collective_loop_with_uniform_break_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_collective_loop_uniform_break");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("collective_loop_uniform_break.br");
    fs::write(
        &input,
        r#"
kernel LoopPhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        tile[i] = i
        var round = 0
        loop:
            tile[i] = tile[i] + round + 1
            sync
            round = round + 1
            if round >= 3: break
        output[i] = tile[i]
mut phased = LoopPhases()
kernel:
    phased(block = 4)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("collective_loop_uniform_break_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "6 7 8 9");
}

#[test]
fn collective_loop_with_thread_break_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_collective_loop_thread_break");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("collective_loop_thread_break.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        var round = 0\n        loop:\n            sync\n            round = round + 1\n            if round > .thread.x: break\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
}

#[test]
fn nested_collective_loop_with_uniform_break_runs_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_nested_collective_loop_uniform_break");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("nested_collective_loop_uniform_break.br");
    fs::write(
        &input,
        r#"
kernel NestedLoopPhases:
    mut [int, 4]'actor tile
    mut [int]'unified output
    init():
        output = [0 for ..<8]
    def ():
        let i = .thread.x
        tile[i] = i
        if .block.x == 0:
            var round = 0
            loop:
                tile[i] = tile[i] + round + 1
                sync
                round = round + 1
                if round >= 3: break
        else:
            tile[i] = tile[i] + 20
            sync
        output[i + .block.x * .block_dim.x] = tile[i]
mut phased = NestedLoopPhases()
kernel:
    phased(block = 4, grid = 2)
print "{phased.output[0]} {phased.output[1]} {phased.output[2]} {phased.output[3]}"
print "{phased.output[4]} {phased.output[5]} {phased.output[6]} {phased.output[7]}"
"#,
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(generated.status.success(), "{}", String::from_utf8_lossy(&generated.stderr));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("nested_collective_loop_uniform_break_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout).trim(),
        "6 7 8 9\n20 21 22 23"
    );
}

#[test]
fn nested_collective_loop_with_thread_break_remains_rejected_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("simd_nested_collective_loop_thread_break");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("nested_collective_loop_thread_break.br");
    fs::write(
        &input,
        "kernel Divergent:\n    def ():\n        if .block.x == 0:\n            var round = 0\n            loop:\n                sync\n                round = round + 1\n                if round > .thread.x: break\n",
    )
    .unwrap();
    let generated = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", "simd"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(!generated.status.success());
    assert!(String::from_utf8_lossy(&generated.stderr)
        .contains("`sync` is not yet supported inside a --target simd kernel"));
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
fn identity_operators_match_gpu_value_semantics_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_identity_operators");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("identity_operators.br");
    fs::write(
        &input,
        r#"
kernel Identity:
    mut [bool]'unified output
    init():
        output = [false for ..<4]
    def ():
        let i = .thread.x
        let value = i % 2
        let int? optional = if i < 2: some(value) else: nil
        if i == 0:
            output[i] = value is 0
        elif i == 1:
            output[i] = value is not 0
        elif i == 2:
            output[i] = optional is nil
        else:
            output[i] = optional is not some(0)
mut identity = Identity()
kernel:
    identity(block = 4)
print "{identity.output[0]} {identity.output[1]} {identity.output[2]} {identity.output[3]}"
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
    let source = fs::read_to_string(root.join("identity_operators_simd/src/main.rs")).unwrap();
    assert!(source.contains("value == 0"));
    assert!(source.contains("value != 0"));
    assert!(source.contains("optional.is_none()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("identity_operators_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "true true true true");
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
fn inferred_local_alias_preserves_labeled_array_axes_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_labeled_alias");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("labeled_alias.br");
    fs::write(
        &input,
        r#"
kernel LabeledAlias:
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        let [int, width = 2, height = 2] original = [10, 20, 30, 40]
        let alias = original
        output[i] = alias[width = i % alias.width, height = i / alias.height]
mut values = LabeledAlias()
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
    let source = fs::read_to_string(root.join("labeled_alias_simd/src/main.rs")).unwrap();
    assert!(source.contains("let alias = original.clone()"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("labeled_alias_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "10 20 30 40");
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
fn scalar_kernel_field_defaults_seed_const_and_thread_local_values() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_kernel_field_defaults");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("kernel_field_defaults.br");
    fs::write(
        &input,
        r#"
kernel Defaults:
    let int factor = 3
    var int offset = 2
    mut [int]'unified output
    init():
        output = [0 for ..<4]
    def ():
        let i = .thread.x
        output[i] = i * factor + offset
mut defaults = Defaults()
kernel:
    defaults(block = 4)
print "{defaults.output[0]} {defaults.output[1]} {defaults.output[2]} {defaults.output[3]}"
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
    let source = fs::read_to_string(root.join("kernel_field_defaults_simd/src/main.rs")).unwrap();
    assert!(source.contains("factor: (3) as i32"));
    assert!(source.contains("offset: (2) as i32"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("kernel_field_defaults_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "2 5 8 11");
}

#[test]
fn fixed_width_integer_types_are_preserved_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_fixed_width_integers");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("fixed_width_integers.br");
    fs::write(
        &input,
        r#"
kernel Widths:
    let int128 wide = int128(5000000000)
    mut [int128]'unified output
    init():
        output = [int128(0) for ..<1]
    def ():
        output[0] = wide + int128(7)
mut widths = Widths()
kernel:
    widths(block = 1)
print widths.output[0]
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
    let source = fs::read_to_string(root.join("fixed_width_integers_simd/src/main.rs")).unwrap();
    assert!(source.contains("wide: i128"));
    assert!(source.contains("output: Vec<i128>"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("fixed_width_integers_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "5000000007");
}

#[test]
fn string_literals_and_interpolation_run_inside_simd_kernels() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_kernel_strings");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("kernel_strings.br");
    fs::write(
        &input,
        r#"
kernel Strings:
    mut [bool]'unified output
    init():
        output = [false for ..<2]
    def ():
        let i = .thread.x
        let message = "lane {i}"
        output[i] = message == (if i == 0: "lane 0" else: "lane 1")
mut strings = Strings()
kernel:
    strings(block = 2)
print "{strings.output[0]} {strings.output[1]}"
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
    let source = fs::read_to_string(root.join("kernel_strings_simd/src/main.rs")).unwrap();
    assert!(source.contains("format!(\"lane {}\", i)"));
    assert!(source.contains("String::from(\"lane 0\")"));
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--release", "--manifest-path"])
        .arg(root.join("kernel_strings_simd/Cargo.toml"))
        .output()
        .unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    assert_eq!(String::from_utf8_lossy(&run.stdout).trim(), "true true");
}

#[test]
fn indexed_atomic_methods_mutate_and_return_previous_values_on_simd() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simd_atomic_methods");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let input = root.join("atomic_methods.br");
    fs::write(
        &input,
        r#"
mut [int]'kernel'unified values = [10, 10, 10, 10, 1, 2, 4]
mut [int]'kernel'unified old = [0 for ..<4]
kernel Atomics:
    mut [int]'actor'unified values
    mut [int]'unified old
    init(mut [int]'unified input_values, mut [int]'unified input_old):
        values = input_values
        old = input_old
    def ():
        let i = .thread.x
        if i == 0:
            old[0] = values[0].min(5)
        elif i == 1:
            old[1] = values[1].max(15)
        elif i == 2:
            old[2] = values[2].swap(20)
        else:
            old[3] = values[3].cas(10, 30)
            values[4] += 2
            values[5] |= 4
            values[6] ^= 1
mut atomics = Atomics(values, old)
kernel:
    atomics(block = 4)
print "{atomics.old[0]} {atomics.old[1]} {atomics.old[2]} {atomics.old[3]}"
print "{atomics.values[0]} {atomics.values[1]} {atomics.values[2]} {atomics.values[3]} {atomics.values[4]} {atomics.values[5]} {atomics.values[6]}"
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
    let source = fs::read_to_string(root.join("atomic_methods_simd/src/main.rs")).unwrap();
    assert!(source.contains("let __boring_atomic_target = &mut self.values"));
    assert!(source.contains("let __boring_atomic_old = (*__boring_atomic_target).clone()"));
    for scalar in [false, true] {
        let mut command = Command::new("cargo");
        command
            .args(["run", "--quiet", "--release", "--manifest-path"])
            .arg(root.join("atomic_methods_simd/Cargo.toml"));
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
            "10 10 10 10\n5 15 20 30 3 6 5",
            "scalar={scalar}"
        );
    }
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
