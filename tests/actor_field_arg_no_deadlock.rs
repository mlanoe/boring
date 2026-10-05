// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a self-deadlock in call-argument codegen: an argument that reads a field
// of an `'actor` value — `helper(interp, params, interp.current_env)` — was emitted with the
// field read's `MutexGuard` (`interp.lock().unwrap().current_env`) as a temporary of the
// *enclosing call statement*, so the guard lived until `helper` returned; `helper` locks `interp`
// again, and a `std::sync::Mutex` is not reentrant. The generated Rust compiles cleanly and then
// hangs forever at ~0% CPU, with no panic. A method call through a locked receiver
// (`x.bump(x.hits, x.hits)` -> `x.lock().unwrap().bump(x.lock().unwrap().hits, ..)`) deadlocked
// the same way. See tests/cases/actor_field_arg_no_deadlock.br's own doc comment for the exact
// before/after.
//
// The failure mode is a hang, not a compile error or a crash, so — like
// actor_self_compound_assign_no_deadlock.rs — this test cannot use `Command::output()` (it would
// block forever if the bug is back, and `transpile_test!`'s shared runner has no timeout either,
// which is why this case is not also a `transpile_test!`; `interp_test!` in tests/run.rs covers
// the interpreter side). It emits and compiles the fixture for each of the four mode/threading
// combinations, then runs each binary under an explicit timeout, polling `try_wait()` and
// killing the process if it doesn't finish in time.
//
// Run with:
//   cargo test --test actor_field_arg_no_deadlock

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Run `child` to completion, killing it (and failing loudly) if it doesn't exit within
/// `timeout` — the hang this test guards against.
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
    context: &str,
) -> std::process::ExitStatus {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("failed to poll child status") {
            return status;
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "{} did not complete within {:?} — this is the self-deadlock hang, not a crash \
                 (a non-reentrant Mutex locked twice: an argument's guard temporary outliving \
                 the call it is an argument of)",
                context, timeout
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn scratch_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("actor_field_arg_no_deadlock_test_scratch")
}

fn run_combo(mode: &str, threading: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let name = "actor_field_arg_no_deadlock";
    let case_br = Path::new("tests/cases").join(format!("{}.br", name));
    let expected = std::fs::read_to_string(Path::new("tests/cases").join(format!("{}.expected", name)))
        .expect("missing .expected file")
        .replace("\r\n", "\n");
    let label = format!("{}@{}+{}", name, mode, threading);

    let root = scratch_root();
    let project = root.join(format!("{}_{}", mode, threading));
    let _ = std::fs::remove_dir_all(&project);

    let emit = Command::new(bin)
        .arg("build").arg(&case_br)
        .arg("--mode").arg(mode)
        .arg("--threading").arg(threading)
        .arg("--output-dir").arg(&project)
        .output()
        .unwrap_or_else(|e| panic!("[{}] failed to invoke boring: {}", label, e));
    assert!(
        emit.status.success(),
        "[{}] boring build failed:\n{}",
        label,
        String::from_utf8_lossy(&emit.stderr)
    );

    // One build directory shared by the four combos (distinct generated projects coexist in it;
    // the combos run sequentially inside the single #[test] below).
    let target_dir = root.join("target");
    let build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(project.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .env("CARGO_TARGET_DIR", &target_dir)
        .output()
        .unwrap_or_else(|e| panic!("[{}] failed to invoke cargo build: {}", label, e));
    assert!(
        build.status.success(),
        "[{}] expected the generated Rust to build, but it failed:\n{}",
        label,
        String::from_utf8_lossy(&build.stderr)
    );

    // The compiled binary is named after the generated package.
    let manifest = std::fs::read_to_string(project.join("Cargo.toml")).expect("missing generated Cargo.toml");
    let pkg = manifest
        .lines()
        .find_map(|l| l.trim().strip_prefix("name").and_then(|r| r.split('"').nth(1)))
        .unwrap_or_else(|| panic!("[{}] no package name in generated Cargo.toml:\n{}", label, manifest))
        .to_string();
    let exe = target_dir.join("debug").join(format!("{}{}", pkg, std::env::consts::EXE_SUFFIX));
    assert!(exe.exists(), "[{}] expected compiled binary at {}", label, exe.display());

    let mut child = Command::new(&exe)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("[{}] failed to spawn compiled binary: {}", label, e));
    let mut stdout_handle = child.stdout.take().expect("child stdout not piped");
    let mut stderr_handle = child.stderr.take().expect("child stderr not piped");

    let status = wait_with_timeout(child, Duration::from_secs(15), &format!("the generated `{}` program", label));

    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();
    let _ = stdout_handle.read_to_string(&mut stdout_buf);
    let _ = stderr_handle.read_to_string(&mut stderr_buf);
    assert!(
        status.success(),
        "[{}] the generated program failed:\n--- stderr ---\n{}",
        label, stderr_buf
    );
    assert_eq!(
        stdout_buf.replace("\r\n", "\n").trim_end(),
        expected.trim_end(),
        "[{}] output mismatch",
        label
    );

    let _ = std::fs::remove_dir_all(&project);
}

#[test]
fn actor_field_arguments_do_not_self_deadlock_in_every_build_mode() {
    for (mode, threading) in [("strict", "multi"), ("strict", "single"), ("managed", "multi"), ("managed", "single")] {
        run_combo(mode, threading);
    }
}
