// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// CLI-level edge cases from the test plan of docs/book.md, "Implicit generics: design notes":
// the inputs inference must reject (`run` and `build` agree) rather than guess.
//
// Run with:
//   cargo test --test implicit_generics_errors

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture_dir() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("implicit_generics_errors_{}_{}", std::process::id(), NEXT.fetch_add(1, Ordering::SeqCst)));
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

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}


fn both(src: &str) -> [Output; 2] {
    [boring(&["run"], src), boring(&["build", "--emit-rust"], src)]
}

fn expect_error(src: &str, needle: &str) {
    for out in both(src) {
        assert!(!out.status.success(), "expected a failure for:\n{src}");
        assert!(stderr(&out).contains(needle), "wanted `{needle}`, got:\n{}", stderr(&out));
    }
}

#[test]
fn field_typed_only_by_a_later_push_is_an_error() {
    // D3: an empty collection is no evidence, a later `push` does not count.
    expect_error(
        "struct Bag:\n    var items\n\ndef main():\n    mut b = Bag(items = [])\n    b.items.push(3)\n    print b.items.len()\n",
        "cannot infer the type of `Bag.items`",
    );
}

#[test]
fn recursive_function_without_a_return_type_is_an_error() {
    // D4: a return type that depends on a call to itself needs a fixpoint, so it must be written.
    expect_error(
        "def count(int n):\n    if n > 0:\n        return count(n - 1)\n    return 0\n\nprint count(3)\n",
        "`count` is recursive and returns a value",
    );
}

#[test]
fn mixed_instances_in_one_array_are_an_error() {
    // D8: `[Response]` is homogeneous.
    expect_error(
        "struct R:\n    let a\n\ndef x():\n    return R(a = 1)\n\ndef y():\n    return R(a = \"s\")\n\nlet v = [x(), y()]\nprint v.len()\n",
        "",
    );
}

#[test]
fn one_struct_two_instantiations_run_and_build_agree() {
    let src = "struct Resp:\n    let status\n    let payload\n\ndef a():\n    return Resp(status = 1, payload = \"x\")\n\ndef b():\n    return Resp(status = 2, payload = [1, 2])\n\nprint a().payload\nprint b().payload.len()\nprint a().status\n";
    let out = boring(&["run"], src);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(String::from_utf8_lossy(&out.stdout), "x\n2\n1\n");
    assert!(boring(&["build", "--emit-rust"], src).status.success());
}

#[test]
fn show_types_lists_constructions_of_an_imported_struct() {
    let dir = fixture_dir();
    std::fs::write(dir.join("types.br"), "struct Resp:\n    let status\n    let payload\n").unwrap();
    std::fs::write(
        dir.join("main.br"),
        "use types\n\ndef mk():\n    return Resp(status = 200, payload = \"hi\")\n\nprint mk().payload\n",
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(["check", "--show-types", "main.br"])
        .current_dir(&dir)
        .output()
        .expect("failed to invoke boring");
    assert!(out.status.success(), "{}", stderr(&out));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("Resp<status = int, payload = string>"), "{text}");
}
