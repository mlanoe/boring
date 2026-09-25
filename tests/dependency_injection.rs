// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Checker and codegen tests for the dependency-injection design
// (docs/design-notes/boring-di-draft.md): `@singleton` (a general,
// DI-independent memoization mechanism, §4), `@provide`'s `pub` requirement
// (§3), and `@inject` itself (§1-§2) — resolved and desugared into a
// synthesized `init` by `src/desugar_inject.rs`, reusing Boring's existing
// labeled-argument-with-defaults call-site machinery rather than any new
// transpiler codegen. Current `@inject` scope (see that file's own doc
// comment for the full list): same-`Program` providers only (no `[deps]`
// cross-project resolution yet), no `id`/`env`, explicit field qualifier
// required (no bare-field inference yet), and a struct with an `@inject`
// field can't also declare its own `init`.
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

// ── `@inject` — resolution against the `@provide` registry ─────────────────────

#[test]
fn inject_resolves_transient_provider_at_zero_arg_call_site() {
    // `desugar_inject.rs` synthesizes an `init` for `Greeting`; the omitting
    // `Greeting()` call site should be rewritten to call the resolved
    // provider directly, exactly like an ordinary defaulted-parameter call.
    let src = "\
trait Greeter:
    req string greet()

struct EnglishGreeter as Greeter:
    req string greet(): \"hello\"

struct Greeting:
    @inject
    Greeter'shared greeter

    req string say():
        self.greeter.greet()

@provide
pub Greeter'shared greeterProvider():
    EnglishGreeter()

def main():
    let g = Greeting()
    print g.say()
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected a resolvable @inject field to compile, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Greeting::new(greeterProvider())"),
        "expected the omitted @inject argument to be filled in with a call to the resolved \
         provider, got:\n{}", stdout
    );
    assert!(
        !stdout.contains("#[inject]"),
        "the `@inject` attribute must never be emitted verbatim as a Rust attribute:\n{}", stdout
    );
}

#[test]
fn inject_no_provider_found_is_rejected() {
    let src = "\
trait NetworkClient:
    req string fetch()

struct UserRepository:
    @inject
    NetworkClient'shared client

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected @inject with no matching provider to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no `@provide` found for type `NetworkClient`"),
        "expected the no-provider-found error, got:\n{}", stderr
    );
}

#[test]
fn inject_ambiguous_providers_rejected() {
    let src = "\
trait Logger:
    req void log(string msg)

struct FileLogger as Logger:
    def void log(string msg): print msg

struct ConsoleLogger as Logger:
    def void log(string msg): print msg

@provide
pub Logger'shared fileLoggerProvider():
    FileLogger()

@provide
pub Logger'shared consoleLoggerProvider():
    ConsoleLogger()

struct Service:
    @inject
    Logger'shared logger

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected two @provide functions for the same type to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ambiguous provider for `Logger`"),
        "expected the ambiguity error, got:\n{}", stderr
    );
}

#[test]
fn inject_bare_field_without_qualifier_is_rejected() {
    let src = "\
trait NetworkClient:
    req string fetch()

struct RealNetworkClient as NetworkClient:
    req string fetch(): \"data\"

struct UserRepository:
    @inject
    NetworkClient client

@provide
pub NetworkClient'shared networkClient():
    RealNetworkClient()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a bare (unqualified) @inject field to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("has no qualifier"),
        "expected the bare-qualifier error, got:\n{}", stderr
    );
}

#[test]
fn inject_singleton_qualifier_mismatch_is_rejected() {
    let src = "\
trait NetworkClient:
    req string fetch()

struct RealNetworkClient as NetworkClient:
    req string fetch(): \"data\"

struct UserRepository:
    @inject
    NetworkClient'actor client

@provide
@singleton
pub NetworkClient'shared networkClient():
    RealNetworkClient()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a qualifier mismatch against a @singleton provider to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("must be matched exactly"),
        "expected the singleton-qualifier-mismatch error, got:\n{}", stderr
    );
}

#[test]
fn inject_owned_rejected_when_provider_is_singleton() {
    let src = "\
trait NetworkClient:
    req string fetch()

struct RealNetworkClient as NetworkClient:
    req string fetch(): \"data\"

struct UserRepository:
    @inject
    NetworkClient'owned client

@provide
@singleton
pub NetworkClient'owned networkClient():
    RealNetworkClient()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected 'owned + a @singleton provider to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cannot be `'owned`"),
        "expected the owned/singleton-provider rejection, got:\n{}", stderr
    );
}

#[test]
fn inject_rejects_struct_with_existing_init() {
    let src = "\
trait NetworkClient:
    req string fetch()

struct RealNetworkClient as NetworkClient:
    req string fetch(): \"data\"

struct UserRepository:
    @inject
    NetworkClient'shared client

    init():
        self.client = RealNetworkClient()

@provide
pub NetworkClient'shared networkClient():
    RealNetworkClient()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected @inject on a struct with its own init to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("already declares its own `init`"),
        "expected the existing-init-conflict error, got:\n{}", stderr
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
    // A matching `@provide` is required so `desugar_inject` (which runs before the
    // kernel-target checker pass) resolves this field successfully instead of
    // failing earlier with an unrelated "no provider found" error.
    let src = "\
trait NetworkClient:
    req string fetch()

struct RealNetworkClient as NetworkClient:
    req string fetch(): \"data\"

struct UserRepository:
    @inject
    NetworkClient'shared client

@provide
pub NetworkClient'shared networkClient():
    RealNetworkClient()

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
