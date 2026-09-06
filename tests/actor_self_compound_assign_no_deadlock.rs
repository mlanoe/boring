// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a self-deadlock in the assignment codegen of
// `src/transpiler/emit_expr.rs` (the `Mutex field write` / `RwLock field
// write` arms): `p.depth += 1` on an `'actor`-qualified (`Arc<Mutex<T>>`)
// parameter desugars (in the parser) to `p.depth = p.depth + 1`, and used to
// transpile to two nested locks of the same non-reentrant
// `std::sync::Mutex` in a single statement — the write's own guard, plus a
// second lock taken to read the RHS's old value while that guard is still
// held. `std::sync::Mutex` is not reentrant, so this is a guaranteed
// self-deadlock at runtime: the generated Rust compiles cleanly and simply
// hangs forever with near-zero CPU, blocked on the second `.lock()` call.
//
// See tests/cases/actor_self_compound_assign.br's own doc comment for the
// exact before/after generated Rust.
//
// The failure mode here is a hang, not a compile error or a crash, so this
// test cannot just use `Command::output()` (which would block forever if
// the bug is back) — it compiles the generated Rust once with `cargo
// build`, then runs the resulting binary itself under an explicit timeout,
// polling `try_wait()` and killing the process if it doesn't finish in time.
//
// Run with:
//   cargo test --test actor_self_compound_assign_no_deadlock

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Run `child` to completion, killing it (and failing loudly) if it doesn't
/// exit within `timeout` — the hang this test guards against.
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
                 (a non-reentrant Mutex/RwLock locked twice in one statement)",
                context, timeout
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn actor_qualified_self_compound_assign_completes_and_increments() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/actor_self_compound_assign.br");
    let dir = Path::new("tests/cases/actor_self_compound_assign_rust");
    std::fs::create_dir_all(dir.join("src")).expect("failed to create src dir");

    let emit = Command::new(bin)
        .arg("build")
        .arg(case_br)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e));
    assert!(
        emit.status.success(),
        "expected `boring build --emit-rust` to succeed, but it failed:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let generated = String::from_utf8_lossy(&emit.stdout).into_owned();

    std::fs::write(dir.join("src/main.rs"), &generated).expect("failed to write main.rs");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"actor_self_compound_assign_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("failed to write Cargo.toml");

    let manifest_path = dir.join("Cargo.toml");

    // Compile first (this alone never hangs — the deadlock is a runtime
    // property of the generated program, not the Rust compiler).
    let build = Command::new("cargo")
        .args(["build", "--quiet", "--manifest-path"])
        .arg(&manifest_path)
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo build: {}", e));
    assert!(
        build.status.success(),
        "expected the generated Rust to build, but it failed:\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&build.stderr),
        generated,
    );

    // Now run the compiled binary directly (not `cargo run`) so we hold the
    // exact process we need to kill if it hangs.
    let exe_name = format!("actor_self_compound_assign_check{}", std::env::consts::EXE_SUFFIX);
    let exe_path = dir.join("target/debug").join(&exe_name);
    assert!(
        exe_path.exists(),
        "expected compiled binary at {}",
        exe_path.display()
    );

    let mut child = Command::new(&exe_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn compiled binary: {}", e));

    let mut stdout_handle = child.stdout.take().expect("child stdout not piped");
    let mut stderr_handle = child.stderr.take().expect("child stderr not piped");

    let status = wait_with_timeout(
        child,
        Duration::from_secs(15),
        "the generated `actor_self_compound_assign` program",
    );

    let mut stdout_buf = String::new();
    let mut stderr_buf = String::new();
    let _ = stdout_handle.read_to_string(&mut stdout_buf);
    let _ = stderr_handle.read_to_string(&mut stderr_buf);

    assert!(
        status.success(),
        "expected the generated program to exit successfully, but it failed:\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        stderr_buf, generated,
    );

    let actual = stdout_buf.replace("\r\n", "\n");
    assert_eq!(
        actual.trim_end(),
        "3",
        "expected the 'actor-qualified counter to be incremented 3 times — got: {}",
        actual
    );

    let _ = std::fs::remove_dir_all(dir);
}
