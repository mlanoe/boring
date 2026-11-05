// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The method form of a view modifier (`Text("x").styled(1)` = `Styled(n = 1, children = [Text("x")])`) also
// resolves a wrapper view declared in a `use`d file (sibling file or `[deps]` project), not only one declared in
// the calling file (src/desugar_view_modifiers.rs, `collect_imported`).
//
// Run with:
//   cargo test --test imported_view_modifier

use std::path::{Path, PathBuf};
use std::process::Command;

const LIB: &str = "\
use boring.ui.*

pub view Styled:
    int n
    [View] children
    body():
        Padding(insets = 1.0, children = children)
";

const MAIN: &str = "\
use boring.ui.*
use lib.*

def main():
    let v = Text(\"x\").styled(3)
    let w = Styled(n = 3, children = [Text(\"x\")])
    print \"ok\"
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

fn boring(dir: &Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_boring"))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("failed to run boring")
}

fn deps_project(name: &str) -> PathBuf {
    let dir = project(name, &[("main.br", &MAIN.replace("use lib.*", "use ui.lib.*"))]);
    let dep = dir.join("dep").join("src");
    std::fs::create_dir_all(&dep).unwrap();
    std::fs::write(dep.join("lib.br"), LIB).unwrap();
    std::fs::write(dir.join("boring.toml"), "[project]\nname = \"t\"\nversion = \"0.1.0\"\n\n[deps]\nui = \"dep\"\n").unwrap();
    dir
}

#[test]
fn method_form_of_a_sibling_file_view_runs() {
    let dir = project("imported_modifier_sibling", &[("lib.br", LIB), ("main.br", MAIN)]);
    let o = boring(&dir, &["run", "main.br"]);
    assert!(o.status.success(), "stderr: {}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "ok");
}

#[test]
fn method_form_of_a_deps_project_view_runs() {
    let dir = deps_project("imported_modifier_dep");
    let o = boring(&dir, &["run", "main.br"]);
    assert!(o.status.success(), "stderr: {}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "ok");
}

#[test]
fn method_form_of_an_imported_view_transpiles() {
    let dir = deps_project("imported_modifier_emit");
    let o = boring(&dir, &["build", "--emit-rust", "main.br"]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success(), "stderr: {err}");
    assert!(!err.contains("no method"), "stderr: {err}");
}
