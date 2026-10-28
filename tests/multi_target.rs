// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `boring build --target all` / `--target a,b`: several backends in one crate, chosen at run time.
// Generation is checked for every backend; the end-to-end run uses simd + wgpu, which build on any
// host (the run itself only needs the simd backend, forced with `BORING_BACKEND`).
//
// Run with: cargo test --test multi_target

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

// The start-up policy is standard-library only; the test uses it to hold leases like a second
// running program would.
#[path = "../src/multi_policy.rs"]
#[allow(dead_code)]
mod policy;

const FIXTURE: &str = "examples/saxpy.br";

/// Each test project's crate has its own name (its source file's stem): tests share one Cargo
/// target directory, and two crates called `main` would overwrite each other's binary.
fn stem(dir: &Path) -> String {
    dir.file_name().unwrap().to_string_lossy().trim_start_matches("multi_target_").to_string()
}

fn source(dir: &Path) -> PathBuf {
    dir.join(format!("{}.br", stem(dir)))
}

fn root(dir: &Path) -> PathBuf {
    dir.join(format!("{}_multi", stem(dir)))
}

fn project(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("multi_target_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(FIXTURE, source(&dir)).unwrap();
    dir
}

fn build(dir: &Path, target: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target])
        .arg(source(dir))
        .current_dir(dir)
        .output()
        .unwrap()
}

#[test]
fn rejects_unknown_targets_and_linux_combination() {
    let dir = project("reject");
    let out = build(&dir, "cuda,bogus");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown target 'bogus'"));

    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--linux", "--target", "wgpu,simd"])
        .arg(source(&dir))
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--linux cannot be combined"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explicit_list_generates_one_crate_with_every_backend() {
    let dir = project("generate");
    let out = build(&dir, "cuda,rocm,metal,wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let root = root(&dir);

    for backend in ["cuda", "rocm", "metal", "wgpu", "simd"] {
        let module = std::fs::read_to_string(root.join(format!("src/backend_{backend}.rs"))).unwrap();
        assert!(module.contains("pub fn __boring_backend_main("), "{backend}");
        assert!(module.contains("pub fn __boring_device_name()"), "{backend}");
        assert!(module.contains("pub fn __boring_bench()"), "{backend}");
        assert!(!module.contains("\nfn main("), "{backend}: fn main must be renamed");
        assert!(!module.contains("#[macro_use] extern crate"), "{backend}");
    }
    for device_source in ["kernels/main.cu", "kernels/main.hip", "kernels/main.metal", "shaders/main.wgsl"] {
        assert!(root.join(device_source).is_file(), "{device_source}");
    }

    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    assert!(manifest.contains("default = [\"cuda\", \"rocm\", \"metal\", \"wgpu\", \"simd\"]"), "{manifest}");
    assert!(manifest.contains("metal = [\"dep:metal\""), "{manifest}");
    assert!(manifest.contains("edition = \"2024\""), "{manifest}");

    let build_rs = std::fs::read_to_string(root.join("build.rs")).unwrap();
    assert!(build_rs.contains("CARGO_FEATURE_CUDA") && build_rs.contains("CARGO_FEATURE_ROCM"), "{build_rs}");

    assert!(root.join("src/policy.rs").is_file());
    let main_rs = std::fs::read_to_string(root.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("#[macro_use]\nextern crate objc;"), "{main_rs}");
    for backend in ["cuda", "rocm", "metal", "wgpu", "simd"] {
        assert!(main_rs.contains(&format!("mod backend_{backend};")), "{backend}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn float16_is_routed_and_compiles_for_wgpu_and_simd() {
    let dir = project("float16");
    std::fs::write(source(&dir), r#"
kernel HalfScale:
    let float16 gain
    mut [float16]'unified values
    init(float16 factor, [float16]'unified input):
        gain = factor
        values = input
    def ():
        let i = .thread.x
        values[i] = values[i] * gain

mut k = HalfScale(float16(2.0), [float16(1.0), float16(2.0)])
kernel:
    k(block = 2, grid = 1)
print "{k.values[1] as float32}"
"#).unwrap();

    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let generated = root(&dir);
    let unified = std::fs::read_to_string(generated.join("src/unified.rs")).unwrap();
    assert!(unified.contains("(F16, BufF16, half::f16, as_f16, buf_f16)"), "{unified}");

    let wgpu = std::fs::read_to_string(generated.join("src/backend_wgpu.rs")).unwrap();
    assert!(wgpu.contains("wgpu::Features::SHADER_F16"), "{wgpu}");
    assert!(wgpu.contains("args[0].as_f16()"), "{wgpu}");
    assert!(wgpu.contains("args[1].buf_f16()"), "{wgpu}");
    let simd = std::fs::read_to_string(generated.join("src/backend_simd.rs")).unwrap();
    assert!(simd.contains("Arg::F16(self.gain)"), "{simd}");
    assert!(simd.contains("Arg::BufF16(&mut self.values)"), "{simd}");

    let checked = Command::new("cargo")
        .args(["check", "--offline", "--manifest-path"])
        .arg(generated.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(Path::new(env!("CARGO_TARGET_TMPDIR")).join("multi_target_shared"))
        .output()
        .unwrap();
    assert!(checked.status.success(), "{}", String::from_utf8_lossy(&checked.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn all_skips_what_the_host_cannot_build() {
    let dir = project("all");
    let out = build(&dir, "all");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("wgpu") && stderr.contains("simd"), "{stderr}");
    if cfg!(target_os = "macos") {
        assert!(stderr.contains("Skipped cuda") && stderr.contains("Skipped rocm"), "{stderr}");
        assert!(root(&dir).join("src/backend_metal.rs").is_file());
    } else {
        assert!(stderr.contains("Skipped metal"), "{stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn backend_is_chosen_at_run_time() {
    let dir = project("run");
    let out = build(&dir, "simd,wgpu");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let manifest = root(&dir).join("Cargo.toml");
    let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("multi_target_shared");

    let run = |backend: &str| {
        Command::new("cargo")
            .args(["run", "--quiet", "--manifest-path"])
            .arg(&manifest)
            .arg("--target-dir")
            .arg(&target_dir)
            .env("BORING_BACKEND", backend)
            .env("BORING_BACKEND_VERBOSE", "1")
            .output()
            .unwrap()
    };

    let forced = run("simd");
    let stdout = String::from_utf8_lossy(&forced.stdout);
    assert!(forced.status.success(), "{}", String::from_utf8_lossy(&forced.stderr));
    assert!(String::from_utf8_lossy(&forced.stderr).contains("kernel Saxpy -> cpu"));
    assert!(stdout.contains("y[0] = 1") && stdout.contains("y[15] = 31"), "{stdout}");

    // A backend that is not compiled in is skipped; the CPU runs the program.
    let missing = run("cuda");
    assert!(missing.status.success(), "{}", String::from_utf8_lossy(&missing.stderr));
    assert!(String::from_utf8_lossy(&missing.stderr).contains("not compiled into this binary"));
    assert!(String::from_utf8_lossy(&missing.stdout).contains("y[15] = 31"));

    let listed = run("list");
    let stdout = String::from_utf8_lossy(&listed.stdout);
    assert!(stdout.contains("wgpu:") && stdout.contains("simd: available"), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Run the generated crate with an isolated cache; returns (stdout, stderr).
fn run_crate(dir: &Path, cache: &Path, envs: &[(&str, &str)]) -> (String, String) {
    let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("multi_target_shared");
    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--quiet", "--manifest-path"])
        .arg(root(dir).join("Cargo.toml"))
        .arg("--target-dir")
        .arg(target_dir)
        .env("BORING_CACHE_DIR", cache)
        .env("BORING_BACKEND_VERBOSE", "1")
        .env_remove("BORING_BACKEND")
        .env_remove("BORING_BENCH");
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let out = cmd.output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
}

#[test]
fn policy_benchmarks_caches_and_reacts_to_load() {
    let dir = project("policy");
    let out = build(&dir, "simd,wgpu");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");

    // `list` shows each usable backend with its device, e.g. `wgpu: available (Apple M3)`.
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    let device_of = |name: &str| {
        let line = listed.lines().find(|l| l.starts_with(&format!("{name}: available")))?;
        let inside = &line[line.find('(')? + 1..line.rfind(')')?];
        Some(inside.split(", score").next().unwrap().to_string())
    };
    let (Some(wgpu_device), Some(simd_device)) = (device_of("wgpu"), device_of("simd")) else {
        // No GPU adapter on this host: the policy has nothing to compare, simd must just run.
        let (stdout, stderr) = run_crate(&dir, &cache, &[]);
        assert!(stderr.contains("kernels run on []"), "{stderr}");
        assert!(stdout.contains("y[15] = 31"), "{stdout}");
        return;
    };

    // First run measures (and caches) both backends; the second one reuses the cache.
    let (_, first) = run_crate(&dir, &cache, &[]);
    assert!(first.contains("benchmarking"), "{first}");
    let (_, second) = run_crate(&dir, &cache, &[]);
    assert!(!second.contains("benchmarking"), "{second}");
    let cached = std::fs::read_to_string(cache.join("backends.tsv")).unwrap();
    assert!(cached.contains("\tsimd\t") && cached.contains("\twgpu\t"), "{cached}");

    // Seed known scores so the outcome does not depend on this machine's speed.
    let stamp = policy::now();
    std::fs::write(
        cache.join("backends.tsv"),
        format!("v1\twgpu\t{wgpu_device}\t1000\t{stamp}\nv1\tsimd\t{simd_device}\t900\t{stamp}\n"),
    )
    .unwrap();
    let (stdout, stderr) = run_crate(&dir, &cache, &[]);
    assert!(stderr.contains("kernels run on [wgpu]"), "{stderr}");
    assert!(stdout.contains("y[15] = 31"), "{stdout}");

    // Another running program holds wgpu: its effective score halves and simd takes over.
    let _busy = policy::acquire(&cache, "wgpu");
    let (stdout, stderr) = run_crate(&dir, &cache, &[]);
    assert!(stderr.contains("load 1"), "{stderr}");
    assert!(stderr.contains("kernels run on []"), "{stderr}");
    assert!(stdout.contains("y[15] = 31"), "{stdout}");

    // BORING_BENCH=off measures nothing: the embedded device table gives the scores (flagged
    // as priors), and the load still counts.
    drop(_busy);
    let (_, stderr) = run_crate(&dir, &cache, &[("BORING_BENCH", "off")]);
    assert!(!stderr.contains("benchmarking"), "{stderr}");
    assert!(stderr.contains("(prior)") && stderr.contains("wgpu"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn kernels_are_routed_per_launch() {
    let dir = project("routed");
    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let root = root(&dir);
    // The CPU host is always part of a routed build, and the wgpu module carries one adapter per kernel.
    assert!(root.join("src/backend_simd.rs").is_file() && root.join("src/unified.rs").is_file());
    let wgpu = std::fs::read_to_string(root.join("src/backend_wgpu.rs")).unwrap();
    assert!(wgpu.contains("pub fn __boring_unified_Saxpy(") && wgpu.contains("pub fn __boring_unified_init()"), "{wgpu}");
    let simd = std::fs::read_to_string(root.join("src/backend_simd.rs")).unwrap();
    assert!(simd.contains("crate::unified::dispatch(&__BORING_SPEC_Saxpy, self.__id"), "{simd}");
    let main_rs = std::fs::read_to_string(root.join("src/main.rs")).unwrap();
    assert!(main_rs.contains("run: backend_wgpu::__boring_unified_Saxpy,"), "{main_rs}");

    // A launch the adapter cannot run (here: another block size than the one baked into the
    // shader) falls back to the CPU instead of computing the wrong thing.
    let cache = dir.join("cache");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    if !listed.contains("wgpu: available") {
        return; // no GPU adapter on this host: nothing to route to
    }
    let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BACKEND", "wgpu")]);
    assert!(stderr.contains("kernel Saxpy -> wgpu"), "{stderr}");
    assert!(stdout.contains("y[0] = 1") && stdout.contains("y[15] = 31"), "{stdout}");
    // Without forcing, a 16-thread launch is too small to be worth the transfer.
    let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BENCH", "off")]);
    assert!(stderr.contains("kernel Saxpy -> cpu"), "{stderr}");
    assert!(stdout.contains("y[15] = 31"), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

const MIX: &str = r#"
kernel Mix:
    let int n
    let uint m
    let float32 a
    let uint32 b
    let [int32]'unified ints
    let [uint8]'global bytes
    mut [float32]'unified out
    init(int nn, uint mm, float32 aa, uint32 bb, [int32]'unified i, [uint8]'global by, [float32]'unified o):
        n = nn
        m = mm
        a = aa
        b = bb
        ints = i
        bytes = by
        out = o
    def ():
        let t = .thread.x + .block.x * .block_dim.x
        if t < n:
            out[t] = a * float32(ints[t]) + float32(bytes[t]) + float32(b) + float32(m)

mut ints = [1 as int32, 2, 3, 4]
let bytes = [1 as uint8, 2, 3, 4]
mut out = [0.0 as float32, 0.0, 0.0, 0.0]
mut k = Mix(4, 2, 0.5, 3, ints, bytes, out)
kernel:
    k(block = 4, grid = 1)
print "{k.out[0]} {k.out[3]}"
mut k2 = Mix(4, 2, 0.5, 3, ints, bytes, out)
kernel:
    k2(block = 2, grid = 2)
print "{k2.out[0]} {k2.out[3]}"
"#;

/// Scalars of every width and `int`/`uint`/byte/float buffers must cross to each backend and
/// back unchanged, including a launch whose block size differs from the one the wgpu shader
/// was built for (that one falls back to another backend).
#[test]
fn argument_types_survive_every_backend() {
    let dir = project("types");
    std::fs::write(source(&dir), MIX).unwrap();
    let out = build(&dir, "all");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    let mut ran = 0;
    for backend in ["simd", "wgpu", "metal", "cuda", "rocm"] {
        if !listed.contains(&format!("{backend}: available")) {
            continue;
        }
        let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BACKEND", backend)]);
        assert_eq!(stdout.trim(), "6.5 11\n6.5 11", "{backend}: {stderr}");
        ran += 1;
    }
    assert!(ran >= 1, "simd is always available:\n{listed}");
    let _ = std::fs::remove_dir_all(&dir);
}

const PLACEMENT: &str = r#"
@only(.cpu)
kernel Cpu:
    mut [float32]'unified buf
    init([float32]'unified data):
        buf = data
    def ():
        let i = .thread.x
        buf[i] = buf[i] + 1.0

@prefer(.gpu)
kernel Gpu:
    mut [float32]'unified buf
    init([float32]'unified data):
        buf = data
    def ():
        let i = .thread.x
        buf[i] = buf[i] * 2.0

kernel Plain:
    mut [float32]'unified buf
    init([float32]'unified data):
        buf = data
    def ():
        let i = .thread.x
        buf[i] = buf[i] + 10.0

mut a = Cpu([1.0 as float32, 2.0, 3.0, 4.0])
mut b = Gpu([1.0 as float32, 2.0, 3.0, 4.0])
mut c = Plain([1.0 as float32, 2.0, 3.0, 4.0])
mut d = Plain([1.0 as float32, 2.0, 3.0, 4.0])
kernel:
    a(block = 4, grid = 1)
kernel:
    b(block = 4, grid = 1)
kernel:
    c(block = 4, grid = 1)
kernel:
    d(block = 4, grid = 1, prefer = .gpu)
print "{a.buf[3]} {b.buf[3]} {c.buf[3]} {d.buf[3]}"
"#;

/// `@only`, `@prefer` and a launch's `prefer =` steer where a kernel runs; the results do not
/// depend on where that is.
#[test]
fn placement_hints_steer_each_kernel() {
    let dir = project("placement");
    std::fs::write(source(&dir), PLACEMENT).unwrap();
    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BENCH", "off")]);
    assert_eq!(stdout.trim(), "5 8 14 14", "{stderr}");
    assert!(stderr.contains("kernel Cpu -> cpu"), "{stderr}");
    assert!(stderr.contains("kernel Plain -> cpu"), "a 4-thread launch is too small to move: {stderr}");
    if listed.contains("wgpu: available") {
        assert!(stderr.contains("kernel Gpu -> wgpu"), "@prefer(.gpu) ignores the small-launch rule: {stderr}");
        // Forcing a backend overrides `@prefer`, never `@only`.
        let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BACKEND", "wgpu")]);
        assert_eq!(stdout.trim(), "5 8 14 14", "{stderr}");
        assert!(stderr.contains("kernel Cpu -> cpu"), "{stderr}");
        assert!(stderr.contains("kernel Plain -> wgpu"), "{stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn only_gpu_fails_when_no_gpu_backend_runs_the_launch() {
    let dir = project("only_gpu");
    let program = PLACEMENT.replace("@only(.cpu)\nkernel Cpu", "@only(.gpu)\nkernel Cpu");
    std::fs::write(source(&dir), program).unwrap();
    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("multi_target_shared");
    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(root(&dir).join("Cargo.toml"))
        .arg("--target-dir")
        .arg(target_dir)
        .env("BORING_CACHE_DIR", dir.join("cache"))
        .env("BORING_BACKEND", "simd")
        .output()
        .unwrap();
    assert!(!run.status.success(), "stdout {:?} stderr {:?}", String::from_utf8_lossy(&run.stdout), String::from_utf8_lossy(&run.stderr));
    assert!(String::from_utf8_lossy(&run.stderr).contains("`@only(.gpu)`"), "{}", String::from_utf8_lossy(&run.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn placement_hints_are_validated_and_checked_against_a_single_target() {
    let dir = project("placement_errors");
    let kernel = "kernel K:\n    mut [float32]'unified buf\n    init([float32]'unified d):\n        buf = d\n    def ():\n        buf[.thread.x] = 1.0\n";
    let try_build = |source: &str, target: &str| {
        std::fs::write(self::source(&dir), source).unwrap();
        let out = build(&dir, target);
        (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
    };
    let (ok, err) = try_build(&format!("@prefer(.tpu)\n{kernel}"), "wgpu,simd");
    assert!(!ok && err.contains("`@prefer` takes one device class"), "{err}");
    let launch = "mut k = K([0.0 as float32, 0.0])\nkernel:\n    k(block = 2, grid = 1, prefer = 3)\n";
    let (ok, err) = try_build(&format!("{kernel}{launch}"), "wgpu,simd");
    assert!(!ok && err.contains("`prefer` takes a device class"), "{err}");
    // A single-target build has nowhere else to run a kernel it is not allowed to run.
    let (ok, err) = try_build(&format!("@only(.gpu)\n{kernel}"), "simd");
    assert!(!ok && err.contains("`@only(.gpu)` but `--target simd` runs on the CPU"), "{err}");
    let (ok, err) = try_build(&format!("@only(.cpu)\n{kernel}"), "wgpu");
    assert!(!ok && err.contains("`@only(.cpu)` but `--target wgpu` runs on a GPU"), "{err}");
    let (ok, err) = try_build(&format!("@only(.gpu)\n{kernel}"), "wgpu");
    assert!(ok, "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

const LAST_CHOICE: &str = r#"
kernel Bump:
    mut [float32]'unified buf
    init([float32]'unified data):
        buf = data
    def ():
        buf[.thread.x] = buf[.thread.x] + 1.0

mut k = Bump([1.0 as float32, 2.0, 3.0, 4.0])
kernel:
    k(block = 4, grid = 1)
print "{k.buf[3]} on {kernel.lastChoice(k)}"
"#;

/// `kernel.lastChoice(k)` names where a kernel's latest launch ran: the router's answer in a
/// multi-target build, the target itself in a single-target one, the CPU in `boring run`.
#[test]
fn last_choice_names_the_backend_that_ran_the_kernel() {
    let dir = project("last_choice");
    std::fs::write(source(&dir), LAST_CHOICE).unwrap();

    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(source(&dir)).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), "5 on cpu");

    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (stdout, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "simd")]);
    assert_eq!(stdout.trim(), "5 on cpu");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    if listed.contains("wgpu: available") {
        let (stdout, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "wgpu")]);
        assert_eq!(stdout.trim(), "5 on wgpu");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const ITERATE: &str = r#"
kernel Step:
    let float32 gain
    let [float32]'unified bias
    mut [float32]'unified state
    init(float32 g, [float32]'unified b, [float32]'unified s):
        gain = g
        bias = b
        state = s
    def ():
        let i = .thread.x + .block.x * .block_dim.x
        state[i] = state[i] * gain + bias[i]

let n = 32
mut k = Step(1.5, [1.0 as float32 for ..<n], [0.0 as float32 for ..<n])
for round in 0..<20:
    kernel:
        k(block = 16, grid = 2)
print "{k.state[0]} {k.state[31]}"
"#;

const MIXED: &str = r#"
kernel Step:
    let float32 gain
    let [float32]'unified bias
    mut [float32]'unified state
    init(float32 g, [float32]'unified b, [float32]'unified s):
        gain = g
        bias = b
        state = s
    def ():
        let i = .thread.x + .block.x * .block_dim.x
        state[i] = state[i] * gain + bias[i]

let n = 32
mut k = Step(2.0, [1.0 as float32 for ..<n], [0.0 as float32 for ..<n])
for round in 0..<5:
    kernel:
        k(block = 16, grid = 2)
print "a {k.state[0]} {k.state[31]}"
k.state[3] = 100.0
k.state[4] += 1.0
for round in 0..<3:
    kernel:
        k(block = 16, grid = 2)
print "b {k.state[3]} {k.state[4]} {k.state[5]}"
for i in 0..<n:
    k.state[i] = 0.0
kernel:
    k(block = 16, grid = 2)
print "c {k.state[0]} {k.state[31]}"
mut other = Step(1.0, [2.0 as float32 for ..<n], [0.0 as float32 for ..<n])
kernel:
    other(block = 16, grid = 2)
kernel:
    k(block = 16, grid = 2)
kernel:
    other(block = 16, grid = 2)
print "d {other.state[0]} {k.state[0]}"
"#;

/// Run the program on one backend with `BORING_BACKEND` forced; (stdout, stderr).
fn run_forced(dir: &Path, cache: &Path, backend: &str) -> (String, String) {
    run_crate(dir, cache, &[("BORING_BACKEND", backend)])
}

/// A kernel launched 20 times uploads its two buffers once and reads one back per host read,
/// instead of copying everything in and out on every launch.
#[test]
fn buffers_stay_resident_between_launches() {
    let dir = project("resident");
    std::fs::write(source(&dir), ITERATE).unwrap();
    let out = build(&dir, "all");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    let (cpu, cpu_stderr) = run_forced(&dir, &cache, "simd");
    assert!(cpu_stderr.contains("buffer transfers: 0 up, 0 down"), "{cpu_stderr}");
    for backend in ["wgpu", "metal", "cuda", "rocm"] {
        if !listed.contains(&format!("{backend}: available")) {
            continue;
        }
        let (stdout, stderr) = run_forced(&dir, &cache, backend);
        assert_eq!(stdout, cpu, "{backend}: {stderr}");
        // bias + state up once; the two `k.state[..]` reads of the program download one each.
        assert!(stderr.contains("buffer transfers: 2 up, 2 down"), "{backend}: {stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Host reads, element writes, in-place edits, a launch kept on the CPU and a second instance
/// between launches must give what the CPU alone gives, and what the interpreter gives.
#[test]
fn host_access_between_launches_stays_coherent() {
    let dir = project("coherent");
    std::fs::write(source(&dir), MIXED).unwrap();
    let out = build(&dir, "all");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(source(&dir)).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    let expected = String::from_utf8_lossy(&interpreted.stdout).into_owned();
    assert!(expected.starts_with("a 31 31"), "{expected}");
    for backend in ["simd", "wgpu", "metal", "cuda", "rocm"] {
        if !listed.contains(&format!("{backend}: available")) {
            continue;
        }
        let (stdout, stderr) = run_forced(&dir, &cache, backend);
        assert_eq!(stdout, expected, "{backend}: {stderr}");
    }
    // Both GPU backends at once: an instance may move between them.
    if listed.contains("wgpu: available") && listed.contains("metal: available") {
        let (stdout, stderr) = run_forced(&dir, &cache, "metal,wgpu");
        assert_eq!(stdout, expected, "{stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const BACKENDS: &str = r#"
kernel Bump:
    mut [float32]'unified buf
    init([float32]'unified data):
        buf = data
    def ():
        buf[.thread.x] = buf[.thread.x] + 1.0

mut k = Bump([1.0 as float32, 2.0])
kernel:
    k(block = 2, grid = 1)
for b in kernel.backends():
    print "{b.name} {b.kind} {b.load}"
"#;

/// `kernel.backends()` lists the backends a program may run kernels on, best first, with each
/// one's kind, score and load; the CPU is always there.
#[test]
fn backends_lists_what_the_program_can_use() {
    let dir = project("backends");
    std::fs::write(source(&dir), BACKENDS).unwrap();

    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(source(&dir)).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), "cpu cpu 0");

    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (stdout, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "simd")]);
    assert_eq!(stdout.trim(), "cpu cpu 0");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    if listed.contains("wgpu: available") {
        let (stdout, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "wgpu,simd")]);
        assert_eq!(stdout.trim(), "wgpu gpu 0\ncpu cpu 0");
    }
    // A single-target build knows only its own target.
    let single = build(&dir, "simd");
    assert!(single.status.success(), "{}", String::from_utf8_lossy(&single.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

const GPU_KINDS: &str = r#"
kernel Bump:
    mut [float32]'unified buf
    init([float32]'unified data):
        buf = data
    def ():
        buf[.thread.x] = buf[.thread.x] + 1.0

mut k = Bump([1.0 as float32, 2.0, 3.0, 4.0])
kernel:
    k(block = 4, grid = 1)
for b in kernel.backends():
    print "{b.name} {b.class}"
print "{k.buf[3]} on {kernel.lastChoice(k)}"
"#;

/// `.discrete` and `.integrated` narrow `.gpu`: a requirement for the other kind fails the
/// launch, a wish for it falls back to whatever runs the kernel.
#[test]
fn gpu_kinds_narrow_placement() {
    let dir = project("gpu_kinds");
    let interpreted_source = GPU_KINDS.to_string();
    std::fs::write(source(&dir), &interpreted_source).unwrap();
    let interpreted = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(source(&dir)).output().unwrap();
    assert!(interpreted.status.success(), "{}", String::from_utf8_lossy(&interpreted.stderr));
    assert_eq!(String::from_utf8_lossy(&interpreted.stdout).trim(), "cpu cpu\n5 on cpu");

    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    if listed.contains("wgpu: available") {
        let (stdout, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "wgpu,simd")]);
        let class = stdout.lines().next().unwrap().strip_prefix("wgpu ").unwrap().to_string();
        assert!(["discrete", "integrated", "gpu"].contains(&class.as_str()), "{stdout}");
        let other = if class == "discrete" { "integrated" } else { "discrete" };
        let with_hint = |hint: &str| {
            std::fs::write(source(&dir), GPU_KINDS.replace("kernel Bump:", &format!("{hint}\nkernel Bump:"))).unwrap();
            let out = build(&dir, "wgpu,simd");
            assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        };
        if class != "gpu" {
            with_hint(&format!("@only(.{class})"));
            let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BACKEND", "wgpu,simd")]);
            assert!(stdout.contains("5 on wgpu"), "{stdout}{stderr}");
            // A wish for the other kind still gets the kernel run, on the one GPU there is.
            with_hint(&format!("@prefer(.{other})"));
            let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BENCH", "off")]);
            assert!(stdout.contains("5 on "), "{stdout}{stderr}");
            // A requirement for the other kind fails the launch.
            with_hint(&format!("@only(.{other})"));
            let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("multi_target_shared");
            let run = Command::new("cargo")
                .args(["run", "--quiet", "--manifest-path"])
                .arg(root(&dir).join("Cargo.toml"))
                .arg("--target-dir")
                .arg(target_dir)
                .env("BORING_CACHE_DIR", &cache)
                .env("BORING_BACKEND", "wgpu,simd")
                .output()
                .unwrap();
            assert!(!run.status.success(), "{}", String::from_utf8_lossy(&run.stdout));
            assert!(String::from_utf8_lossy(&run.stderr).contains(&format!("`@only(.{other})`")), "{}", String::from_utf8_lossy(&run.stderr));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const DIMENSION_LIFE: &str = r#"
let width = 64
let height = 64

kernel Step:
    mut [uint]'unified cellsIn
    mut [uint]'unified cellsOut
    let Dimension dim

    init(Dimension d):
        cellsIn = [if (i * 7 + i / 5) % 3 == 0: 1 else 0 for i in 0..<d.width * d.height]
        cellsOut = [0 for ..<d.width * d.height]
        dim = d

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < dim.width and row < dim.height:
            var int n = 0
            for dy in -1..<2:
                for dx in -1..<2:
                    if dx != 0 or dy != 0:
                        let nx = (int(col) + dx + int(dim.width)) % int(dim.width)
                        let ny = (int(row) + dy + int(dim.height)) % int(dim.height)
                        n += int(cellsIn[ny * dim.width + nx])
            let alive = cellsIn[row * dim.width + col]
            cellsOut[row * dim.width + col] = if alive == 1: uint(n == 2 or n == 3) else uint(n == 3)

kernel Render:
    let [uint]'unified cells
    mut [uint]'unified pixels
    let Dimension dim

    init(Dimension d):
        cells = [0 for ..<d.width * d.height]
        pixels = [0 for ..<d.width * d.height]
        dim = d

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < dim.width and row < dim.height:
            pixels[row * dim.width + col] = if cells[row * dim.width + col] == 1: uint(255) else uint(17)

var mut step = Step(Dimension(width, height))
var mut render = Render(Dimension(width, height))

for i in 0..<4096:
    if (i * 7 + i / 5) % 3 == 0:
        step.cellsIn[i] = 1
kernel:
    step(block = (16, 16))
let a = step.cellsIn
let b = step.cellsOut
with a:
    with b:
        var sa = 0
        var sb = 0
        for c in a:
            sa += int(c)
        for c in b:
            sb += int(c)
        print "in {sa} out {sb}"
"#;

/// A kernel with a `Dimension` field is routed like any other: the backend builds its buffers
/// from the size, and the results match the CPU.
#[test]
fn dimension_kernels_are_routed() {
    let dir = project("dimension");
    std::fs::write(source(&dir), DIMENSION_LIFE).unwrap();
    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (cpu, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "simd")]);
    assert_eq!(cpu.trim(), "in 1639 out 3224");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    if listed.contains("wgpu: available") {
        let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BACKEND", "wgpu,simd")]);
        assert_eq!(stdout, cpu, "{stderr}");
        assert!(stderr.contains("kernel Step -> wgpu"), "{stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const SCREEN_LIFE: &str = r#"
let width = 64
let height = 64
let screen = Screen(Dimension(width, height), title = "life")

kernel Step:
    mut [uint]'surface cellsIn
    mut [uint]'surface cellsOut
    let Dimension dim

    init(Dimension d):
        cellsIn = [if (i * 7 + i / 5) % 3 == 0: 1 else 0 for i in 0..<d.width * d.height]
        cellsOut = [0 for ..<d.width * d.height]
        dim = d

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < dim.width and row < dim.height:
            var int n = 0
            for dy in -1..<2:
                for dx in -1..<2:
                    if dx != 0 or dy != 0:
                        let nx = (int(col) + dx + int(dim.width)) % int(dim.width)
                        let ny = (int(row) + dy + int(dim.height)) % int(dim.height)
                        n += int(cellsIn[ny * dim.width + nx])
            let alive = cellsIn[row * dim.width + col]
            cellsOut[row * dim.width + col] = if alive == 1: uint(n == 2 or n == 3) else uint(n == 3)

kernel Render:
    let [uint]'surface cells
    mut [uint]'surface pixels
    let Dimension dim

    init(Dimension d):
        cells = [0 for ..<d.width * d.height]
        pixels = [0 for ..<d.width * d.height]
        dim = d

    def ():
        let col = .block.x * .block_dim.x + .thread.x
        let row = .block.y * .block_dim.y + .thread.y
        if col < dim.width and row < dim.height:
            pixels[row * dim.width + col] = if cells[row * dim.width + col] == 1: uint(255) else uint(17)

var mut step = Step(Dimension(width, height))
var mut render = Render(Dimension(width, height))

var generation = 0
kernel:
    loop:
        step(block = (16, 16))
        step.cellsIn, step.cellsOut = step.cellsOut, step.cellsIn
        render.cells = step.cellsIn
        render(block = (16, 16))
        generation += 1
        if generation == 6:
            break
    let pixels = render.pixels
    with pixels:
        var total = 0
        for p in pixels:
            total += int(p)
        print "pixels {total}"
"#;

/// A program with a `Screen` is routed too: its `'surface` buffers are ordinary buffers for the
/// router, and the CPU host that presents them reads the newest pixels back from the backend.
/// The window itself cannot open here, so the program computes and prints a checksum instead of
/// presenting; `examples/game_of_life.br` is only built and started up to the backend list.
#[test]
fn screen_programs_route_their_surfaces() {
    let dir = project("screen_life");
    std::fs::write(source(&dir), SCREEN_LIFE).unwrap();
    let target = if cfg!(target_os = "macos") { "metal,simd" } else { "simd" };
    let out = build(&dir, target);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (cpu, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "simd")]);
    assert_eq!(cpu.trim(), "pixels 306680");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    if listed.contains("metal: available") {
        let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BACKEND", "metal,simd")]);
        assert_eq!(stdout, cpu, "{stderr}");
        assert!(stderr.contains("kernel Step -> metal") && stderr.contains("kernel Render -> metal"), "{stderr}");
        // Six generations of ping-pong swap and kernel-to-kernel copy stay on the device: the
        // buffers go up once and only the final checksum reads them back.
        let counts: Vec<u64> = stderr
            .lines()
            .find_map(|l| l.strip_prefix("boring: buffer transfers: "))
            .map(|l| l.split(|c: char| !c.is_ascii_digit()).filter_map(|n| n.parse().ok()).collect())
            .unwrap_or_default();
        assert!(counts.len() == 2 && counts[0] <= 6 && counts[1] <= 4, "swap and copy went through the CPU: {stderr}");
    }

    // The shipped example, for every backend that can build it, still starts.
    let example = project("screen_example");
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/game_of_life.br"), source(&example)).unwrap();
    let out = build(&example, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let (listed, _) = run_crate(&example, &cache, &[("BORING_BACKEND", "list")]);
    assert!(listed.contains("simd: available"), "{listed}");
    let _ = std::fs::remove_dir_all(&example);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Reading one element of a buffer that lives on a backend reads that element, and the values
/// are the CPU's.
#[test]
fn single_elements_are_read_from_the_backend() {
    let dir = project("element_reads");
    let program = format!("{DIMENSION_LIFE}print \"elements {{step.cellsOut[0]}} {{step.cellsOut[5]}} {{step.cellsOut[63]}} {{step.cellsOut[4095]}} {{step.cellsIn[7]}}\"\n");
    std::fs::write(source(&dir), program).unwrap();
    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (cpu, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "simd")]);
    assert!(cpu.contains("elements "), "{cpu}");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    if listed.contains("wgpu: available") {
        let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BACKEND", "wgpu,simd")]);
        assert_eq!(stdout, cpu, "{stderr}");
        assert!(stderr.contains("kernel Step -> wgpu"), "{stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

const SHARED_REDUCE: &str = r#"
kernel Reduce:
    let [float32]'unified input
    mut [float32]'unified sums
    mut [float32, 64]'actor tile

    init([float32]'unified data, [float32]'unified out):
        input = data
        sums = out

    def ():
        let tid = .thread.x
        tile[tid] = input[.block.x * .block_dim.x + tid]
        if tid == 0:
            var float32 sum = 0.0
            for k in 0..<64:
                sum += tile[k]
            sums[.block.x] = sum

mut reduce = Reduce([float32(i) for i in 0..<256], [float32(0) for ..<4])
kernel:
    reduce(block = 64, grid = 4)
print "sums {reduce.sums[0]} {reduce.sums[1]} {reduce.sums[2]} {reduce.sums[3]}"
"#;

const ATOMIC_HISTOGRAM: &str = r#"
kernel Histogram:
    let [int]'unified values
    mut [int]'actor'unified counts

    init([int]'unified v, [int]'actor'unified c):
        values = v
        counts = c

    def ():
        let i = .block.x * .block_dim.x + .thread.x
        counts[values[i] % 4] += 1

mut hist = Histogram([i * 3 for i in 0..<256], [0, 0, 0, 0])
kernel:
    hist(block = 64, grid = 4)
print "counts {hist.counts[0]} {hist.counts[1]} {hist.counts[2]} {hist.counts[3]}"
"#;

/// Kernels with block-shared memory (`'actor` tiles) and atomic buffers (`'actor'unified`) are
/// routed: shared memory is not an argument, an atomic buffer is a buffer.
#[test]
fn shared_memory_and_atomics_are_routed() {
    let dir = project("shared_atomics");
    let target = if cfg!(target_os = "macos") { "wgpu,metal,simd" } else { "wgpu,simd" };
    for (program, expected, kernel) in [
        (SHARED_REDUCE, "sums 2016 6112 10208 14304", "Reduce"),
        (ATOMIC_HISTOGRAM, "counts 64 64 64 64", "Histogram"),
    ] {
        std::fs::write(source(&dir), program).unwrap();
        // 64-bit atomics (`int`) are not available in Metal Shading Language: a Metal build of the
        // histogram is left to the CPU and wgpu.
        let target = if kernel == "Histogram" { "wgpu,simd" } else { target };
        let out = build(&dir, target);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let cache = dir.join("cache");
        let (cpu, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "simd")]);
        assert_eq!(cpu.trim(), expected);
        let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
        for backend in ["wgpu", "metal"] {
            if listed.contains(&format!("{backend}: available")) {
                let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BACKEND", &format!("{backend},simd"))]);
                assert_eq!(stdout.trim(), expected, "{backend}: {stderr}");
                assert!(stderr.contains(&format!("kernel {kernel} -> {backend}")), "{backend} did not run it: {stderr}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A kernel a backend cannot compile is run elsewhere, not reported as the program's error.
#[test]
fn a_kernel_a_backend_rejects_runs_elsewhere() {
    let dir = project("rejected_kernel");
    // `int32` atomics are rejected by wgpu's validation (a non-atomic buffer declaration).
    std::fs::write(source(&dir), ATOMIC_HISTOGRAM.replace("[int]", "[int32]").replace("[i * 3 for", "[int32(i * 3) for").replace("[0, 0, 0, 0]", "[int32(0), int32(0), int32(0), int32(0)]")).unwrap();
    let out = build(&dir, "wgpu,simd");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let cache = dir.join("cache");
    let (listed, _) = run_crate(&dir, &cache, &[("BORING_BACKEND", "list")]);
    if listed.contains("wgpu: available") {
        let (stdout, stderr) = run_crate(&dir, &cache, &[("BORING_BACKEND", "wgpu,simd")]);
        assert_eq!(stdout.trim(), "counts 64 64 64 64", "{stderr}");
        assert!(stderr.contains("wgpu cannot run kernel Histogram") && stderr.contains("kernel Histogram -> cpu"), "{stderr}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
