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
// comment for the full list): same-project only (no `[deps]` cross-project
// resolution yet) — a provider is visible from the entry file itself *or*
// any same-project sibling file reached via a bare `use <name>`
// (`emit_rust_multi_file` below covers this; the reverse direction — a
// struct with its own `@inject` field declared *only* in a sibling file — is
// not covered, see `desugar_inject.rs`'s module doc for why), `id`/`env`
// (§5-§6) and cycle detection (§7) are both implemented (see their own
// sections further down), a struct with an `@inject` field can't also
// declare its own `init`, and a bare (unqualified) field
// against a non-`@singleton` ("transient") provider is only resolved without
// an explicit qualifier when the field's base type is a trait AND the
// provider itself returns that trait bare too (a trait object's bare
// representation is a fixed `Box<dyn Trait>` rule, not something chapter
// 30's per-function usage-based inference decides — see
// `synthesize_init`'s doc comment) — every other bare-field-vs-transient-
// provider combination (a plain struct/generic base type, or a trait base
// type whose only visible provider returns it `'shared`/`'actor`/`'guard`/
// `'owned`/`'new`) still needs an explicit qualifier.
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

fn emit_rust_with_env(src: &str, env: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = tempfile_dir();
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, src).expect("failed to write fixture .br file");

    Command::new(bin)
        .arg("build")
        .arg(&br_file)
        .arg("--env")
        .arg(env)
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

/// Same-project multi-file scenario (`desugar_inject.rs`'s `walk_same_project_uses`):
/// `main_src` is the entry file (`main.br`), `sibling_src` is written alongside it
/// under `sibling_name.br` and reached from `main_src` via a bare `use <sibling_name>`.
fn emit_rust_multi_file(main_src: &str, sibling_name: &str, sibling_src: &str) -> std::process::Output {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = tempfile_dir();
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, main_src).expect("failed to write fixture main.br");
    std::fs::write(dir.join(format!("{sibling_name}.br")), sibling_src)
        .expect("failed to write fixture sibling .br file");

    Command::new(bin)
        .arg("build")
        .arg(&br_file)
        .arg("--emit-rust")
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
fn inject_bare_trait_field_with_bare_transient_provider_compiles_and_runs() {
    // The one bare-field-vs-transient-provider combination that IS resolved without
    // an explicit qualifier: the field's base type is a trait (`NetworkClient`) and
    // the only visible provider returns that trait bare too (no qualifier of its
    // own). A bare trait-typed field's Rust representation is a fixed rule —
    // `Box<dyn Trait>`, unconditionally, since a trait object is unsized — not
    // something chapter 30's per-function usage-based inference decides the way an
    // ordinary bare struct field's representation is, so there's no ordering gap for
    // this specific shape: `struct_init_defaults` (`src/transpiler/mod.rs`) now
    // renders this default through the qualifier-aware `emit_let_value`, which
    // already knows to `Box::new(...)`-wrap a bare trait-typed value.
    let src = "\
trait NetworkClient:
    req string fetch()

struct RealNetworkClient as NetworkClient:
    req string fetch(): \"data\"

struct UserRepository:
    @inject
    NetworkClient client

    req string load():
        self.client.fetch()

@provide
pub NetworkClient networkClient():
    RealNetworkClient()

def main():
    let repo = UserRepository()
    print repo.load()
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected a bare @inject trait field with a bare-returning transient provider to compile, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Box::new(networkClient())"),
        "expected the omitted @inject argument to be boxed to match the bare trait field's \
         `Box<dyn NetworkClient>` representation, got:\n{}", stdout
    );
}

#[test]
fn inject_bare_field_with_transient_provider_is_rejected() {
    // Bare-field inference against a transient provider whose own return type is
    // qualified (`'shared` here) isn't supported: unlike the bare-provider case
    // above, there's no single wrap that turns an `Arc<dyn NetworkClient>`-returning
    // call into the `Box<dyn NetworkClient>` a bare field always is — this is a real
    // representation mismatch (a plain `boring build`-specific gap when the base type
    // isn't even a trait — see the non-trait test below — but a permanent one here),
    // confirmed via a real `cargo build` failure before this rejection was added.
    // Only the `@singleton` case (below) and the bare-trait/bare-provider case above
    // are unaffected.
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
    assert!(!out.status.success(), "expected a bare @inject field with a transient provider to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("needs an explicit qualifier for now"),
        "expected the bare-transient rejection, got:\n{}", stderr
    );
}

#[test]
fn inject_bare_non_trait_field_with_transient_provider_is_rejected() {
    // Same rejection as above, but for a base type that isn't a trait at all (a
    // plain struct) — the general case the bare-trait exception above deliberately
    // does NOT cover. Here chapter 30's ordinary per-function usage-based inference
    // genuinely is what decides this field's eventual representation ('inline vs
    // 'owned vs 'shared/'actor/'guard), and that decision isn't made yet at the
    // point `struct_init_defaults` (`src/transpiler/mod.rs`) renders the omitted
    // default expression — a real, still-open `boring build` ordering gap, unlike
    // the trait case (where the representation is a fixed rule, not an inference
    // outcome, so there's nothing to wait on).
    let src = "\
struct Config:
    int value

struct App:
    @inject
    Config config

@provide
pub Config configProvider():
    Config(value = 42)

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a bare @inject field with a non-trait base type and a transient provider to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("needs an explicit qualifier for now"),
        "expected the bare-transient rejection, got:\n{}", stderr
    );
}

#[test]
fn inject_bare_field_with_no_base_type_is_rejected() {
    // A bare field whose type isn't even a recognizable named type at all (a scalar) —
    // distinct code path from the transient-provider rejection above, exercised
    // separately since there's no provider to resolve against in the first place.
    let src = "\
@provide
pub int configValue():
    42

struct Config:
    @inject
    int value

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a bare scalar @inject field to be rejected");
}

#[test]
fn inject_bare_field_copies_singleton_provider_qualifier_verbatim() {
    // §2: a bare @inject field matched against a @singleton provider copies the
    // provider's own qualifier verbatim, skipping chapter 30 inference entirely — the
    // design doc's flagship case ("the single most likely real case"). No ordering
    // problem here (unlike the transient case above): the field's final type is fixed
    // immediately, before this struct's `init`/defaults are ever registered.
    let src = "\
trait NetworkClient:
    req string fetch()

struct RealNetworkClient as NetworkClient:
    req string fetch(): \"data\"

struct UserRepository:
    @inject
    NetworkClient client

    req string fetchData():
        self.client.fetch()

@provide
@singleton
pub NetworkClient'shared networkClient():
    RealNetworkClient()

def main():
    let repo = UserRepository()
    print repo.fetchData()
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected a bare field matched against a @singleton provider to compile, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("pub client: Arc<dyn NetworkClient>"),
        "expected the bare field's type to be rewritten to the provider's own qualifier \
         ('shared -> Arc<dyn NetworkClient>), got:\n{}", stdout
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

// ── `'static` (§2) ───────────────────────────────────────────────────────────────
//
// Checker/registry-level acceptance only, not a full end-to-end compile+run: a
// bare constructor-call return value isn't wrapped in the `&` reference a
// `'static` return type requires (task_ce5a4ff9, filed this session, a real gap
// unrelated to `@inject` — confirmed via a plain, hand-written function with no DI
// attributes involved at all). Add a full `tests/cases/static_di.br` behavioral
// test (matching `inject_di.br`) once that's fixed.

#[test]
fn inject_static_field_is_accepted() {
    let src = "\
struct RealConfig:
    string apiKey

struct Service:
    @inject
    RealConfig'static config

@provide
pub RealConfig'static loadConfig():
    RealConfig(apiKey = \"secret\")

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected an explicit 'static @inject field to be accepted, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Service::new(loadConfig())") || stdout.contains("Service { config: "),
        "expected the resolved provider to be wired in, got:\n{}", stdout
    );
}

#[test]
fn inject_static_bare_field_requires_explicit_qualifier() {
    // §2: `'static` is the one deliberate exception to "bare field copies a
    // @singleton provider's qualifier verbatim" — never silently inherited.
    let src = "\
struct RealConfig:
    string apiKey

struct Service:
    @inject
    RealConfig config

@provide
pub RealConfig'static loadConfig():
    RealConfig(apiKey = \"secret\")

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a bare field copying 'static to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("must write `'static` explicitly"),
        "expected the must-write-static-explicitly error, got:\n{}", stderr
    );
}

#[test]
fn inject_owned_accepted_with_transient_provider() {
    let src = "\
trait Greeter:
    req string greet()

struct EnglishGreeter as Greeter:
    req string greet(): \"hello\"

struct Greeting:
    @inject
    Greeter'owned greeter

    req string say():
        self.greeter.greet()

@provide
pub Greeter greeterProvider():
    EnglishGreeter()

def main():
    let g = Greeting()
    print g.say()
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected 'owned with a transient (non-@singleton) provider to compile, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
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

// ── `id` (§5) — multiple named bindings of the same type ───────────────────────

#[test]
fn inject_id_disambiguates_multiple_bindings() {
    let src = "\
trait Database:
    req string query()

struct PostgresDatabase as Database:
    string host
    req string query(): \"{self.host}\"

@provide(id = \"primary\")
@singleton
pub Database'shared primaryDb():
    PostgresDatabase(host = \"primary.internal\")

@provide(id = \"replica\")
@singleton
pub Database'shared replicaDb():
    PostgresDatabase(host = \"replica.internal\")

struct ReportGenerator:
    @inject(id = \"replica\")
    Database'shared db

def main():
    let r = ReportGenerator()
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected id-disambiguated providers to compile, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ReportGenerator::new(replicaDb())"),
        "expected @inject(id = \"replica\") to resolve to replicaDb specifically, got:\n{}", stdout
    );
}

#[test]
fn inject_id_with_no_matching_provider_is_rejected() {
    let src = "\
trait Database:
    req string query()

struct PostgresDatabase as Database:
    req string query(): \"data\"

@provide(id = \"primary\")
pub Database'shared primaryDb():
    PostgresDatabase()

struct ReportGenerator:
    @inject(id = \"replica\")
    Database'shared db

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected an id with no matching provider to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no `@provide` found for type `Database`"),
        "expected the no-provider-found error, got:\n{}", stderr
    );
}

#[test]
fn inject_bare_and_id_bindings_never_collide() {
    // A bare `@inject`/`@provide` (no `id`) and an `id`-tagged one for the same base
    // type are two distinct registry keys — never ambiguous with each other.
    let src = "\
trait Database:
    req string query()

struct PostgresDatabase as Database:
    string host
    req string query(): \"{self.host}\"

@provide
pub Database'shared defaultDb():
    PostgresDatabase(host = \"default.internal\")

@provide(id = \"replica\")
pub Database'shared replicaDb():
    PostgresDatabase(host = \"replica.internal\")

struct ServiceA:
    @inject
    Database'shared db

struct ServiceB:
    @inject(id = \"replica\")
    Database'shared db

def main():
    let a = ServiceA()
    let b = ServiceB()
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected a bare and an id-tagged provider for the same type to coexist, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("ServiceA::new(defaultDb())"), "got:\n{}", stdout);
    assert!(stdout.contains("ServiceB::new(replicaDb())"), "got:\n{}", stdout);
}

// ── `env` (§6) — build-specific provider override ───────────────────────────────

#[test]
fn inject_env_overrides_plain_provider_when_matching() {
    let src = "\
trait AuditLogger:
    req string label()

struct FileAuditLogger as AuditLogger:
    req string label(): \"file\"

struct MockAuditLogger as AuditLogger:
    req string label(): \"mock\"

@provide
@singleton
pub AuditLogger'shared auditLogger():
    FileAuditLogger()

@provide(env = \"test\")
pub AuditLogger'shared testAuditLogger():
    MockAuditLogger()

struct ReportGenerator:
    @inject
    AuditLogger'shared logger

def main():
    let r = ReportGenerator()
    print \"ok\"
";
    // No --env: falls back to the plain provider.
    let out = emit_rust(src);
    assert!(out.status.success(), "got:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ReportGenerator::new(auditLogger())"),
        "expected the plain provider with no --env, got:\n{}", stdout
    );

    // --env test: the env-tagged provider outranks the plain one.
    let out = emit_rust_with_env(src, "test");
    assert!(out.status.success(), "got:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ReportGenerator::new(testAuditLogger())"),
        "expected --env test to resolve to testAuditLogger, got:\n{}", stdout
    );

    // --env prod: doesn't match "test", falls back to the plain provider.
    let out = emit_rust_with_env(src, "prod");
    assert!(out.status.success(), "got:\n{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ReportGenerator::new(auditLogger())"),
        "expected --env prod (no match) to fall back to the plain provider, got:\n{}", stdout
    );
}

#[test]
fn inject_ambiguous_same_env_is_rejected() {
    let src = "\
trait Logger:
    req string label()

struct A as Logger:
    req string label(): \"a\"
struct B as Logger:
    req string label(): \"b\"

@provide(env = \"test\")
pub Logger'shared providerA():
    A()

@provide(env = \"test\")
pub Logger'shared providerB():
    B()

struct Service:
    @inject
    Logger'shared logger

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected two providers for the same env to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ambiguous provider for `Logger`"),
        "expected the ambiguity error, got:\n{}", stderr
    );
}

// ── Cycle detection (§7) ─────────────────────────────────────────────────────────

#[test]
fn inject_cycle_between_two_providers_is_rejected() {
    let src = "\
trait TraitA:
    req int value()
trait TraitB:
    req int value()

struct ConcreteA as TraitA:
    @inject
    TraitB'shared b

    req int value(): 1

struct ConcreteB as TraitB:
    @inject
    TraitA'shared a

    req int value(): 2

@provide
pub TraitA'shared provideA():
    ConcreteA()

@provide
pub TraitB'shared provideB():
    ConcreteB()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a cycle between two providers to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cycle detected among `@provide` providers"),
        "expected the cycle-detection error, got:\n{}", stderr
    );
    // The chain should name both providers, in either traversal order.
    assert!(
        stderr.contains("provideA") && stderr.contains("provideB"),
        "expected the full chain naming both providers, got:\n{}", stderr
    );
}

#[test]
fn inject_transitive_non_cyclic_dependency_is_accepted() {
    // §7: "Transitive resolution falls out for free" — a real dependency chain
    // (A needs B, B needs nothing further) is not a cycle and must compile fine.
    let src = "\
struct Logger:
    def log(): print \"log\"

struct RealNetworkClient:
    @inject
    Logger'shared logger

    def fetch(): self.logger.log()

@provide
pub Logger'shared loggerProvider():
    Logger()

@provide
pub RealNetworkClient'shared networkClient():
    RealNetworkClient()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected a non-cyclic transitive dependency to compile, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("RealNetworkClient::new(loggerProvider())"),
        "expected transitive resolution to fall out for free, got:\n{}", stdout
    );
}

#[test]
fn inject_self_cycle_is_rejected() {
    // A struct whose own provider (transitively, through itself) needs another
    // instance of the exact same base type — the degenerate one-node cycle.
    let src = "\
trait Node:
    req int value()

struct LinkedNode as Node:
    @inject
    Node'shared next

    req int value(): 1

@provide
pub Node'shared nodeProvider():
    LinkedNode()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(!out.status.success(), "expected a self-cycle to be rejected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("cycle detected among `@provide` providers"),
        "expected the cycle-detection error, got:\n{}", stderr
    );
}

// ── Same-project multi-file resolution ──────────────────────────────────────────

#[test]
fn inject_resolves_provider_declared_in_sibling_file() {
    let main_src = "\
use providers

struct UserRepository:
    @inject
    NetworkClient'shared client

def main():
    let repo = UserRepository()
    print \"ok\"
";
    let sibling_src = "\
trait NetworkClient:
    req string fetch()

struct RealNetworkClient as NetworkClient:
    req string fetch(): \"data\"

@provide
pub NetworkClient'shared networkClient():
    RealNetworkClient()
";
    let out = emit_rust_multi_file(main_src, "providers", sibling_src);
    assert!(
        out.status.success(),
        "expected a provider in a same-project sibling file to resolve, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("UserRepository::new(networkClient())"),
        "expected the sibling-file provider to be wired in, got:\n{}", stdout
    );
}

#[test]
fn inject_ambiguous_provider_across_entry_and_sibling_file_is_rejected() {
    // The ambiguity check (`collect_providers`) must see both files' providers as
    // one registry, not two independent ones.
    let main_src = "\
use providers

trait Logger:
    req void log(string msg)

struct ConsoleLogger as Logger:
    def void log(string msg): print msg

@provide
pub Logger'shared consoleLoggerProvider():
    ConsoleLogger()

struct Service:
    @inject
    Logger'shared logger

def main():
    print \"ok\"
";
    let sibling_src = "\
struct FileLogger as Logger:
    def void log(string msg): print msg

@provide
pub Logger'shared fileLoggerProvider():
    FileLogger()
";
    let out = emit_rust_multi_file(main_src, "providers", sibling_src);
    assert!(!out.status.success(), "expected providers in two different files to still be caught as ambiguous");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ambiguous provider for `Logger`"),
        "expected the ambiguity error, got:\n{}", stderr
    );
}

#[test]
fn inject_no_provider_found_still_fires_across_files() {
    // A sibling file exists and is reachable, but declares no matching provider at
    // all — must still fail with the ordinary no-provider-found error, not silently
    // succeed just because *some* sibling file was scanned.
    let main_src = "\
use helpers

struct UserRepository:
    @inject
    NetworkClient'shared client

def main():
    print \"ok\"
";
    let sibling_src = "\
trait NetworkClient:
    req string fetch()
";
    let out = emit_rust_multi_file(main_src, "helpers", sibling_src);
    assert!(!out.status.success(), "expected no-provider-found to still fire when the sibling file has no matching @provide");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("no `@provide` found for type `NetworkClient`"),
        "expected the no-provider-found error, got:\n{}", stderr
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
