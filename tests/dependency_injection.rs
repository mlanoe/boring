// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Checker and codegen tests for the first implementation slice of the
// dependency-injection design (docs/design-notes/boring-di-draft.md): the
// `@singleton` attribute (a general, DI-independent memoization mechanism, §4)
// and `@provide`'s `pub` requirement (§3). `@inject` itself has no
// resolution/registry pass yet and is expected to be rejected with a clear
// "not implemented yet" error rather than silently compiling.
//
// Run with:
//   cargo test --test dependency_injection

use std::path::Path;
use std::process::Command;

fn emit_rust(src: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = tempfile_dir();
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, src).expect("failed to write fixture .br file");

    Command::new(bin)
        .arg("build")
        .arg(&br_file)
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e))
}

fn emit_rust_single_threaded(src: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = tempfile_dir();
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, src).expect("failed to write fixture .br file");

    Command::new(bin)
        .arg("build")
        .arg(&br_file)
        .arg("--threading")
        .arg("single")
        .arg("--emit-rust")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e))
}

fn emit_rust_kernel(src: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = tempfile_dir();
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, src).expect("failed to write fixture .br file");

    Command::new(bin)
        .arg("build")
        .arg("--target")
        .arg("kernel")
        .arg(&br_file)
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke boring: {}", e))
}

// Each test gets its own scratch subdirectory under target/ so parallel test
// threads never race on the same file path.
fn tempfile_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("dependency_injection_test_scratch")
        .join(format!("{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
    dir
}

// ── `@singleton` codegen (checker-level, generated-Rust-text assertions) ───────

#[test]
fn singleton_wraps_in_lazylock_and_clones() {
    let src = "\
@singleton
string expensiveGreeting():
    print \"computing\"
    \"hello\"

def main():
    print expensiveGreeting()
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected @singleton to compile, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("std::sync::LazyLock"),
        "expected a LazyLock-backed static, got:\n{}", stdout
    );
    assert!(
        stdout.contains("fn expensiveGreeting() -> Arc<str> { __BORING_SINGLETON_EXPENSIVEGREETING.clone() }")
            || stdout.contains("fn expensiveGreeting() -> Rc<str> { __BORING_SINGLETON_EXPENSIVEGREETING.clone() }"),
        "expected a clone-from-static wrapper function, got:\n{}", stdout
    );
    // No raw `#[singleton]` Rust attribute should ever leak into the output.
    assert!(
        !stdout.contains("#[singleton]"),
        "the `@singleton` attribute must never be emitted verbatim as a Rust attribute:\n{}", stdout
    );
}

#[test]
fn singleton_attribute_works_without_explicit_def_keyword() {
    // Regression test: an attribute above a return-type-first function declaration
    // with no explicit `def`/`req` keyword (`NetworkClient'shared networkClient(): ...`,
    // the form the DI design doc uses almost exclusively) used to be silently
    // discarded by the parser (see the `TokenKind::At` dispatch in parser/mod.rs) —
    // the function compiled with zero errors and the attribute simply had no effect.
    let src = "\
@singleton
int fortyTwo():
    42

def main():
    print fortyTwo()
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected @singleton to compile, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("std::sync::LazyLock"),
        "expected @singleton codegen to fire even with no explicit `def` keyword, got:\n{}",
        stdout
    );
}

// ── `@singleton` checker rejections ─────────────────────────────────────────────

#[test]
fn singleton_rejects_owned_return_type() {
    let src = "\
struct Widget:
    int x

@singleton
Widget'owned makeWidget():
    Widget(1)

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected `'owned` + `@singleton` to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`@singleton`'s return type cannot be `'owned`"),
        "expected the owned/singleton incompatibility error, got:\n{}", stderr
    );
}

#[test]
fn singleton_rejects_nonzero_params() {
    let src = "\
@singleton
int addOne(int x):
    x + 1

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected @singleton with parameters to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`@singleton` requires a zero-parameter function"),
        "expected the zero-parameter error, got:\n{}", stderr
    );
}

#[test]
fn singleton_rejects_throws() {
    let src = "\
@singleton
string riskyGreeting() throws:
    \"hello\"

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected @singleton + throws to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`@singleton` does not support `throws`"),
        "expected the throws-incompatibility error, got:\n{}", stderr
    );
}

#[test]
fn singleton_rejects_method_form() {
    let src = "\
struct Widget:
    int x

@singleton
def string Widget.describe():
    \"a widget\"

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected @singleton on a `def Type.method()` to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`@singleton` is not yet supported on a method"),
        "expected the method-form rejection, got:\n{}", stderr
    );
}

#[test]
fn singleton_rejects_under_threading_single() {
    // `static`/`LazyLock<T>` requires `T: Sync` regardless of `--threading` — under
    // `single`, `'actor` collapses to `Rc<RefCell<T>>`, which isn't `Sync`. Mirrors
    // `'static`'s own identical, pre-existing restriction (`static_sync_violation`).
    let src = "\
struct Counter:
    var int value = 0

@singleton
Counter'actor makeCounter():
    mut Counter'actor c = Counter(0)
    c

def main():
    print \"ok\"
";
    let out = emit_rust_single_threaded(src);
    assert!(!out.status.success(), "expected @singleton to be rejected under --threading single");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot be `@singleton` under --threading single"),
        "expected the threading/Sync error, got:\n{}", stderr
    );
}

// ── `@provide` checks ────────────────────────────────────────────────────────────

#[test]
fn provide_requires_pub() {
    let src = "\
trait Logger:
    req void log(string msg)

struct ConsoleLogger as Logger:
    req void log(string msg):
        print msg

@provide
Logger'shared consoleLogger():
    ConsoleLogger()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected non-pub @provide to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`@provide` requires `pub`"),
        "expected the pub-required error, got:\n{}", stderr
    );
}

#[test]
fn provide_pub_compiles_cleanly() {
    let src = "\
trait Logger:
    req void log(string msg)

struct ConsoleLogger as Logger:
    req void log(string msg):
        print msg

@provide
pub Logger'shared consoleLogger():
    ConsoleLogger()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected a pub @provide function to compile, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ── `@inject` — not implemented yet, must fail loudly, never silently ──────────

#[test]
fn inject_is_rejected_as_not_yet_implemented() {
    let src = "\
trait NetworkClient:
    req [byte] fetch(string url) throws

struct UserRepository:
    @inject
    NetworkClient'shared client

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected @inject to be rejected (not implemented yet)");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`@inject` is not implemented yet"),
        "expected the not-yet-implemented error, got:\n{}", stderr
    );
}

// ── `--target kernel` rejection ──────────────────────────────────────────────────

#[test]
fn singleton_is_rejected_under_kernel_target() {
    let src = "\
@singleton
int getConfig():
    42

def main():
    print \"ok\"
";
    let out = emit_rust_kernel(src);
    assert!(!out.status.success(), "expected @singleton to be rejected under --target kernel");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`@singleton` is not supported on the Rust-for-Linux kernel target"),
        "expected the kernel-target rejection, got:\n{}", stderr
    );
}

#[test]
fn inject_is_rejected_under_kernel_target() {
    let src = "\
trait NetworkClient:
    req [byte] fetch(string url) throws

struct UserRepository:
    @inject
    NetworkClient'shared client

def main():
    print \"ok\"
";
    let out = emit_rust_kernel(src);
    assert!(!out.status.success(), "expected @inject to be rejected under --target kernel");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("`@inject` is not supported on the Rust-for-Linux kernel target"),
        "expected the kernel-target rejection, got:\n{}", stderr
    );
}
