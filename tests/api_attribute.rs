// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// CLI-level tests for `@api` (docs/book.md, "Implicit generics: design notes", D1):
// the attribute needs `pub`, takes only free-text `since`/`deprecated`, and is rejected
// (never silently dropped) on an item it cannot attach to. The positive path (it changes
// nothing at run time and is never emitted to Rust) is `tests/cases/api_attribute.br`;
// the finer checker rules are unit-tested in `src/checker/api.rs`.
//
// Run with:
//   cargo test --test api_attribute

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture_dir() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("api_attribute_{}_{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
    std::fs::create_dir_all(&dir).expect("failed to create fixture dir");
    dir
}

fn boring(mode: &[&str], src: &str) -> Output {
    let file = fixture_dir().join("main.br");
    std::fs::write(&file, src).expect("failed to write fixture");
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(mode)
        .arg(&file)
        .output()
        .expect("failed to invoke boring")
}

fn run(src: &str) -> Output {
    boring(&["run"], src)
}

fn emit(src: &str) -> Output {
    boring(&["build", "--emit-rust"], src)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const NOT_PUB: &str = "@api\ndef int one():\n    return 1\n\nprint one()\n";

#[test]
fn api_without_pub_is_rejected_by_run_and_build() {
    for out in [run(NOT_PUB), emit(NOT_PUB)] {
        assert!(!out.status.success(), "expected a failure");
        assert!(stderr(&out).contains("`@api` requires `pub`"), "got:\n{}", stderr(&out));
    }
}

#[test]
fn unknown_argument_is_rejected() {
    let out = run("@api(version = \"1.0\")\npub def int one():\n    return 1\n\nprint one()\n");
    assert!(!out.status.success());
    assert!(stderr(&out).contains("unknown `@api` argument `version`"), "got:\n{}", stderr(&out));
}

#[test]
fn since_is_free_text() {
    let out = run("@api(since = \"spring 2027, maybe\")\npub def int one():\n    return 1\n\nprint one()\n");
    assert!(out.status.success(), "got:\n{}", stderr(&out));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "1");
}

#[test]
fn api_on_a_trait_is_an_error_not_a_silent_no_op() {
    let out = run("@api\npub trait Shape:\n    req float area()\n\nprint 1\n");
    assert!(!out.status.success(), "expected a failure");
    assert!(stderr(&out).contains("`@api` is only supported on"), "got:\n{}", stderr(&out));
}

#[test]
fn api_is_not_emitted_as_a_rust_attribute() {
    let out = emit("@api(since = \"1.0\")\npub struct P:\n    int x = 0\n\n@api\npub enum E:\n    A\n    B\n\n@api\npub def int f():\n    return 1\n\nprint f()\n");
    assert!(out.status.success(), "got:\n{}", stderr(&out));
    let rust = String::from_utf8_lossy(&out.stdout);
    assert!(!rust.contains("#[api"), "`api` leaked into the generated Rust");
}

// ─── `boring api` / `boring api --check` (snapshot comparison, phase 4 of the implicit-generics draft)

const V1: &str = "@api\npub int twice(int x):\n    x * 2\n\n@api\npub int zero():\n    0\n";

fn api(args: &[&str], dir: &std::path::Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boring")).arg("api").args(args).current_dir(dir).output().expect("failed to invoke boring")
}

#[test]
fn api_snapshot_then_check_detects_breaking_changes_only() {
    let dir = fixture_dir();
    std::fs::write(dir.join("v1.br"), V1).unwrap();
    let snap = api(&["v1.br"], &dir);
    assert!(snap.status.success(), "{}", stderr(&snap));
    std::fs::write(dir.join("api.snapshot"), &snap.stdout).unwrap();
    assert!(String::from_utf8_lossy(&snap.stdout).contains("twice\tdef twice(int x) -> int"));

    // Unchanged: ok.
    assert!(api(&["--check", "api.snapshot", "v1.br"], &dir).status.success());

    // Added item: compatible, reported but not an error.
    std::fs::write(dir.join("v2.br"), format!("{V1}\n@api\npub int one():\n    1\n")).unwrap();
    let out = api(&["--check", "api.snapshot", "v2.br"], &dir);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(stderr(&out).contains("added:    `one`"), "{}", stderr(&out));

    // Changed + removed: breaking, exit code 1.
    std::fs::write(dir.join("v3.br"), "@api\npub float twice(float x):\n    x * 2.0\n").unwrap();
    let out = api(&["--check", "api.snapshot", "v3.br"], &dir);
    assert!(!out.status.success());
    let err = stderr(&out);
    assert!(err.contains("breaking: `twice` changed") && err.contains("breaking: `zero` was removed"), "{err}");
}

#[test]
fn api_check_rejects_a_malformed_snapshot() {
    let dir = fixture_dir();
    std::fs::write(dir.join("v1.br"), V1).unwrap();
    std::fs::write(dir.join("bad.snapshot"), "no tab here\n").unwrap();
    let out = api(&["--check", "bad.snapshot", "v1.br"], &dir);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("bad snapshot"), "{}", stderr(&out));
}

#[test]
fn api_view_rejects_an_untyped_pub_prop_but_not_a_private_bind() {
    let bad = "use boring.ui.*\n\n@api\npub view Badge:\n    pub note\n\n    req string describe():\n        \"{note}\"\n";
    let out = emit(bad);
    assert!(!out.status.success());
    assert!(stderr(&out).contains("field `Badge.note` has no type"), "{}", stderr(&out));
    let ok = "use boring.ui.*\n\n@api\npub view Badge:\n    pub string note\n    bind var label\n\n    req string describe():\n        \"{label}:{note}\"\n";
    let out = emit(ok);
    assert!(out.status.success(), "{}", stderr(&out));
}
