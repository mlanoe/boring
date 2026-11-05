// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `'actor`/`'guard` get no exception from the `mut` rule (CLAUDE.md, "Binding × mutability"):
// a `def` call or a field write through a binding that is not `mut` is rejected whether the
// qualifier is written (`let T'actor c`, `T'actor` parameter) or only inferred from a function's
// return type (`let d = mk()`), in `boring build` AND `boring run`. The `mut` spellings are accepted.
//
// Run with:
//   cargo test --test actor_mut_rule

use std::process::Command;

const DECL: &str = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n\
Counter'actor mk():\n    Counter(0)\n";

fn run_mode(args: &[&str], body: &str, name: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("boring_actor_mut_rule_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("t.br");
    std::fs::write(&file, format!("{DECL}{body}")).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).args(args).arg(&file).output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), text)
}

fn rejected_everywhere(name: &str, body: &str) {
    let (ok, text) = run_mode(&["build", "--emit-rust"], body, name);
    assert!(!ok && text.contains("is not declared `mut`"), "build should reject {name}:\n{text}");
    let (ok, text) = run_mode(&["run"], body, name);
    assert!(!ok && text.contains("mut"), "run should reject {name}:\n{text}");
}

#[test]
fn inferred_from_return_type_needs_mut() {
    rejected_everywhere("inferred", "def main():\n    let d = mk()\n    d.inc()\n");
}

#[test]
fn explicit_let_needs_mut() {
    rejected_everywhere("explicit", "def main():\n    let Counter'actor c = Counter(0)\n    c.inc()\n");
}

#[test]
fn actor_param_def_call_needs_mut() {
    rejected_everywhere(
        "param_call",
        "def bump(Counter'actor t):\n    t.inc()\ndef main():\n    mut c = mk()\n    bump(c)\n",
    );
}

#[test]
fn actor_param_field_write_needs_mut() {
    rejected_everywhere(
        "param_field",
        "def bump(Counter'actor t):\n    t.value = 5\ndef main():\n    mut c = mk()\n    bump(c)\n",
    );
}

#[test]
fn mut_spellings_are_accepted() {
    let body = "def bump(mut Counter'actor t):\n    t.inc()\n    t.value = t.value + 1\n\
def main():\n    mut d = mk()\n    d.inc()\n    bump(d)\n    print d.value\n";
    let (ok, text) = run_mode(&["build", "--emit-rust"], body, "ok_build");
    assert!(ok, "build should accept:\n{text}");
    let (ok, text) = run_mode(&["run"], body, "ok_run");
    assert!(ok && text.trim() == "3", "run should print 3:\n{text}");
}
