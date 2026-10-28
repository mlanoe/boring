// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A `bind` declared by a view in a `use`d file (sibling file or `[deps]` project) is linked to the caller's `state`:
// the call site `Label(text = name)` hands the child a slot, so the child's write reaches the parent. The lowering
// pass reads the imported file's view signatures (src/lower_view.rs, `collect_imported_bind_views`).
//
// Run with:
//   cargo test --test imported_bind_view

use std::path::{Path, PathBuf};
use std::process::Command;

const LIB: &str = "\
use boring.ui.*

pub view Label:
    bind string text
    body():
        TextField(value = text, label = \"t\", placeholder = \"\")
";

const MAIN: &str = "\
use boring.ui.*
use lib.*

view Holder:
    state string name = \"a\"
    body():
        Label(text = name)

def main():
    let h = Holder()
    let st = h.newState()
    let label = h.expand(st)
    var field = label.expand(label.newState())
    field.viewAct(\"setText\", \"hello\")
    let typed = st as Holder__State else (Holder__State())
    print \"{typed.name.value}\"
";

fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("failed to create project dir");
    for (file, src) in files {
        std::fs::write(dir.join(file), src).expect("failed to write fixture");
    }
    dir
}

fn run(dir: &Path, file: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("run")
        .arg(dir.join(file))
        .current_dir(dir)
        .output()
        .expect("failed to run boring")
}

#[test]
fn bind_of_a_sibling_file_view_is_a_slot() {
    let dir = project("imported_bind_sibling", &[("lib.br", LIB), ("main.br", MAIN)]);
    let o = run(&dir, "main.br");
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "stderr: {}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(out.trim(), "hello");
}

#[test]
fn bind_of_a_deps_project_view_is_a_slot() {
    let dir = project("imported_bind_dep", &[("main.br", &MAIN.replace("use lib.*", "use ui.lib.*"))]);
    let dep = dir.join("dep").join("src");
    std::fs::create_dir_all(&dep).unwrap();
    std::fs::write(dep.join("lib.br"), LIB).unwrap();
    std::fs::write(dir.join("boring.toml"), "[project]\nname = \"t\"\nversion = \"0.1.0\"\n\n[deps]\nui = \"dep\"\n").unwrap();
    let o = run(&dir, "main.br");
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "stderr: {}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(out.trim(), "hello");
}
