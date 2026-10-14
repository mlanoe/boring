// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Imported array-block callables (docs/design-notes/boring-ui-draft.md, §1 "Imported
// array-block callables"): `Column:` must resolve to collect semantics when `Column`'s
// `[dyn Trait]` signature is declared in a `use`d file, not only in the entry file.
//
// Run with:
//   cargo test --test imported_array_block

use std::path::{Path, PathBuf};
use std::process::Command;

const LIB: &str = "\
pub trait Widget:
    req string label()

pub struct Text as Widget:
    string text
    req string label(): text

pub struct Column as Widget:
    [Widget] children
    req string label(): \"Column({self.children.length})\"

struct Hidden as Widget:
    [Widget] children
    req string label(): \"hidden\"

pub Text row([Widget] children):
    Text(\"row of {children.length}\")
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

fn stdout(o: &std::process::Output) -> String { String::from_utf8_lossy(&o.stdout).into_owned() }
fn stderr(o: &std::process::Output) -> String { String::from_utf8_lossy(&o.stderr).into_owned() }

const BODY: &str = "\
def main():
    let c = Column:
        Text(\"a\")
        Text(\"b\")
    print c.label()
    let r = row:
        Text(\"x\")
    print r.label()
";

#[test]
fn bare_sibling_use_resolves_collect() {
    let main = format!("use lib\n\n{BODY}");
    let dir = project("imported_array_block_bare", &[("lib.br", LIB), ("main.br", &main)]);
    let out = run(&dir, "main.br");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "Column(2)\nrow of 1\n");
}

#[test]
fn selective_multi_segment_use_resolves_collect() {
    let main = format!("use lib.Column, row, Text\n\n{BODY}");
    let dir = project("imported_array_block_selective", &[("lib.br", LIB), ("main.br", &main)]);
    let out = run(&dir, "main.br");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "Column(2)\nrow of 1\n");
}

#[test]
fn top_level_use_alias_resolves_imported_constructor_signature() {
    let main = "\
use lib.Column, Text
use Stack as Column

def main():
    let c = Stack:
        Text(\"a\")
        Text(\"b\")
    print c.label()
";
    let dir = project("imported_array_block_alias", &[("lib.br", LIB), ("main.br", main)]);
    let out = run(&dir, "main.br");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "Column(2)\n");
}

#[test]
fn statement_use_alias_resolves_imported_constructor_signature() {
    let main = "\
use lib.Column, Text

def main():
    use Stack as Column
    let c = Stack:
        Text(\"a\")
    print c.label()
";
    let dir = project("imported_array_block_stmt_alias", &[("lib.br", LIB), ("main.br", main)]);
    let out = run(&dir, "main.br");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "Column(1)\n");
}

#[test]
fn local_declaration_shadows_import() {
    // The local `Column` takes a plain closure, so the block must be the trailing-closure
    // form, not collect — proving the local signature wins over the imported one.
    let main = "\
use lib.Text

def Text Column(int body()):
    _ = body()
    Text(\"local\")

def main():
    let t = Column():
        print \"body ran\"
        1
    print t.label()
";
    let dir = project("imported_array_block_shadow", &[("lib.br", LIB), ("main.br", main)]);
    let out = run(&dir, "main.br");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "body ran\nlocal\n");
}

#[test]
fn non_pub_item_is_not_importable_through_multi_segment_use() {
    // `Hidden` is not `pub`: the ordinary import rule rejects `use lib.Hidden` outright, so
    // the array-block pass never has a reason to treat it as a visible callee.
    let main = "\
use lib.Text, Hidden

def main():
    print \"unreachable\"
";
    let dir = project("imported_array_block_nonpub", &[("lib.br", LIB), ("main.br", main)]);
    let out = run(&dir, "main.br");
    assert!(!out.status.success());
    assert!(stderr(&out).contains("not exported"), "stderr: {}", stderr(&out));
}

#[test]
fn array_block_inside_an_imported_module_is_desugared() {
    // The imported file itself uses `Column:` — the interpreter's `use` loader must run the
    // array-block pass on it (against its own imports), not only on the entry file.
    let view = "\
use lib.*

pub Text screen():
    let c = Column:
        Text(\"a\")
        if true:
            Text(\"b\")
        for i in [1, 2]:
            Text(\"c{i}\")
    Text(c.label())
";
    let main = "\
use view

def main():
    print screen().label()
";
    let dir = project("imported_array_block_in_module", &[("lib.br", LIB), ("view.br", view), ("main.br", main)]);
    let out = run(&dir, "main.br");
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(stdout(&out), "Column(4)\n");
}
