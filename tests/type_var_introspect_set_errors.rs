// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `boring run`: reflection `set` on a struct's type-level fields reports clear errors
// (`type let` immutable, wrong value type, unknown field). The build side is covered
// by `tests/cases/type_var_introspect_set.br` (errors surface as `Err`, caught with `try?`).
use std::process::Command;

fn run_src(name: &str, body: &str) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("boring_tvis_{}_{}", name, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("t.br");
    let src = format!(
        "struct C as Introspect:\n    int id\n    pub type let int limit = 1\n    pub type var int count = 0\n\ndef main() throws:\n    let info = C(id = 1).introspect()\n    for f in info.fields:\n        if f.name == \"{}\":\n            f.set({})\n    print \"count={{C.count}}\"\n",
        body.split('|').next().unwrap(), body.split('|').nth(1).unwrap());
    std::fs::write(&f, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_boring")).arg("run").arg(&f).output().unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)))
}

#[test]
fn set_on_type_var_works() {
    let (ok, out) = run_src("ok", "count|9");
    assert!(ok, "{out}");
    assert!(out.contains("count=9"), "{out}");
}

#[test]
fn set_on_type_let_is_rejected() {
    let (ok, out) = run_src("let", "limit|5");
    assert!(!ok);
    assert!(out.contains("field 'limit' is not rebindable"), "{out}");
}

#[test]
fn set_with_wrong_value_type_is_rejected() {
    let (ok, out) = run_src("ty", "count|\"x\"");
    assert!(!ok);
    assert!(out.contains("wrong argument type for 'count': expected an integer"), "{out}");
}
