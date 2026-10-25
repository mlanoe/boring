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

fn project(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("multi_target_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(FIXTURE, dir.join("main.br")).unwrap();
    dir
}

fn build(dir: &Path, target: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["build", "--target", target, "main.br"])
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
        .args(["build", "--linux", "--target", "wgpu,simd", "main.br"])
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
    let root = dir.join("main_multi");

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
fn all_skips_what_the_host_cannot_build() {
    let dir = project("all");
    let out = build(&dir, "all");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("wgpu") && stderr.contains("simd"), "{stderr}");
    if cfg!(target_os = "macos") {
        assert!(stderr.contains("Skipped cuda") && stderr.contains("Skipped rocm"), "{stderr}");
        assert!(dir.join("main_multi/src/backend_metal.rs").is_file());
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
    let manifest = dir.join("main_multi/Cargo.toml");
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
    assert!(String::from_utf8_lossy(&forced.stderr).contains("using backend simd"));
    assert!(stdout.contains("y[0] = 1") && stdout.contains("y[15] = 31"), "{stdout}");

    let missing = run("cuda");
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("no usable backend"));

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
        .arg(dir.join("main_multi/Cargo.toml"))
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
        assert!(stderr.contains("using backend simd"), "{stderr}");
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
    assert!(stderr.contains("using backend wgpu"), "{stderr}");
    assert!(stdout.contains("y[15] = 31"), "{stdout}");

    // Another running program holds wgpu: its effective score halves and simd takes over.
    let _busy = policy::acquire(&cache, "wgpu");
    let (stdout, stderr) = run_crate(&dir, &cache, &[]);
    assert!(stderr.contains("load 1"), "{stderr}");
    assert!(stderr.contains("using backend simd"), "{stderr}");
    assert!(stdout.contains("y[15] = 31"), "{stdout}");

    // BORING_BENCH=off skips the ranking: the static order (wgpu first) applies again.
    let (_, stderr) = run_crate(&dir, &cache, &[("BORING_BENCH", "off")]);
    assert!(stderr.contains("using backend wgpu"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}
