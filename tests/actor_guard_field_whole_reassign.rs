// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Regression test for a whole-field-reassignment bug in the assignment
// codegen of `src/transpiler/emit_expr.rs`'s `emit_expr_assign`:
// `self.field = expr` (and `outer_var.field = expr`) never re-wrapped the
// RHS for an `'actor`/`'guard`(/`'observed`)-qualified field — it emitted a
// plain, unwrapped assignment that doesn't type-check, since the field's
// real Rust type is `Arc<Mutex<T>>`/`Arc<RwLock<T>>`/`BoringObserved<V>`,
// not the bare `T` on the RHS.
//
// `emit_expr_assign` already had dedicated branches for writing into a field
// *reached through* an already-locked mutex/rwlock field ("self.worker.field
// = v" / "self.data.field = v") and for the `'observed` `.value` escape
// hatch, but nothing for the simpler, more common case where the field
// itself IS the wrapped value being replaced wholesale. The fix wraps the
// RHS the same way construction already does (`emit_actor_new`/
// `emit_guard_new`/`wrap_observed_base` — see emit_struct.rs's per-field-
// default init), while leaving alone the one case that already worked:
// assigning an existing actor/guard-typed *variable* straight into the
// field, which the generic fallback already auto-`.clone()`s.
//
// Run with:
//   cargo test --test actor_guard_field_whole_reassign

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

// Each test gets its own scratch subdirectory under target/ so parallel test
// threads never race on the same file path.
fn tempfile_dir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("actor_guard_field_whole_reassign_test_scratch")
        .join(format!("{}_{}", std::process::id(), n));
    std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
    dir
}

// ── Fast codegen-text checks (no `cargo build`) ─────────────────────────────

#[test]
fn actor_field_whole_reassign_wraps_in_arc_mutex() {
    let src = "\
struct FormModel:
    var string name = \"\"

struct Container:
    mut FormModel'actor model = FormModel()
    def replace():
        self.model = FormModel()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected `boring build --emit-rust` to succeed, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("self.model = Arc::new(std::sync::Mutex::new(FormModel::new()));"),
        "expected the whole-field reassignment to be wrapped in Arc::new(Mutex::new(...)), \
         got generated source:\n{}",
        generated
    );
}

#[test]
fn guard_field_whole_reassign_wraps_in_arc_rwlock() {
    let src = "\
struct FormModel:
    var string name = \"\"

struct Container:
    mut FormModel'guard model = FormModel()
    def replace():
        self.model = FormModel()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected `boring build --emit-rust` to succeed, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("self.model = Arc::new(std::sync::RwLock::new(FormModel::new()));"),
        "expected the whole-field reassignment to be wrapped in Arc::new(RwLock::new(...)), \
         got generated source:\n{}",
        generated
    );
}

#[test]
fn observed_field_whole_reassign_wraps_in_boring_observed_new() {
    let src = "\
struct FormModel:
    var string name = \"\"
    def setName(string s):
        name = s

struct Container:
    mut FormModel'actor'observed model = FormModel()
    def replace():
        self.model = FormModel()

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected `boring build --emit-rust` to succeed, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains(
            "self.model = BoringObserved::new(Arc::new(std::sync::Mutex::new(FormModel::new())));"
        ),
        "expected the whole-field reassignment to be re-wrapped in a fresh \
         BoringObserved::new(Arc::new(Mutex::new(...))), got generated source:\n{}",
        generated
    );
}

// Regression guard: assigning an existing actor-typed *variable* into the
// field (as opposed to a fresh raw value) already worked before this fix via
// a dedicated auto-`.clone()` fallback — the new wrapping branch must not
// fire here and double-wrap it into Arc::new(Mutex::new(Arc<Mutex<...>>)).
#[test]
fn actor_field_reassign_from_existing_actor_var_still_autoclones() {
    let src = "\
struct FormModel:
    var string name = \"\"

struct Container:
    mut FormModel'actor model = FormModel()
    def replace(FormModel'actor other):
        self.model = other

def main():
    print \"ok\"
";
    let out = emit_rust(src);
    assert!(
        out.status.success(),
        "expected `boring build --emit-rust` to succeed, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let generated = String::from_utf8_lossy(&out.stdout);
    assert!(
        generated.contains("self.model = other.clone();"),
        "expected the existing-variable case to stay a plain `.clone()` (not get \
         double-wrapped), got generated source:\n{}",
        generated
    );
}

// ── End-to-end: the original bug report reproduces a real `cargo build` E0308 ──

#[test]
fn actor_field_whole_reassign_compiles_and_runs_correctly() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let case_br = Path::new("tests/cases/actor_field_whole_reassign.br");
    let dir = Path::new("tests/cases/actor_field_whole_reassign_rust");
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
        "[package]\nname = \"actor_field_whole_reassign_check\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("failed to write Cargo.toml");

    let run = Command::new("cargo")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("failed to invoke cargo: {}", e));

    assert!(
        run.status.success(),
        "expected the generated Rust to build and run, but it failed (this is the \
         original bug: `self.model = FormModel::new();` against a field whose Rust \
         type is `Arc<Mutex<FormModel>>` — a real E0308 mismatched-types error):\n\
         --- stderr ---\n{}\n--- generated source ---\n{}",
        String::from_utf8_lossy(&run.stderr),
        generated,
    );

    let actual = String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n");
    assert_eq!(
        actual.trim_end(),
        "Ada",
        "expected the replaced model's name (set through the fresh lock after \
         reassignment) to read back correctly"
    );

    let _ = std::fs::remove_dir_all(dir);
}
