// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Array / dict / tuple / set literals whose declared element type carries a pointer qualifier
// (`'actor`, `'guard`, `'shared`, `'owned`, `'inline`): every element is built as that type (a
// fresh value is wrapped, a lock-handle field of the same flavour is the shared handle, not a
// copy), and a method called through an element or tuple slot goes through its lock. Each program
// runs under `boring run` and as a compiled project in all four mode/threading configurations; the
// five outputs must be identical.
//
// Run with:
//   tools/test.sh --test qualified_collection_literals

mod common;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn project(name: &str, src: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("qualified_collection_literals").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("main.br"), src).unwrap();
    dir
}

fn boring(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boring")).current_dir(dir).args(args).output().unwrap()
}

fn text(b: &[u8]) -> String { String::from_utf8_lossy(b).into_owned() }

fn check(name: &str, src: &str, expected: &str) {
    let dir = project(name, src);
    let out = boring(&dir, &["run", "main.br"]);
    assert!(out.status.success(), "{name}: run failed:\n{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), expected, "{name}: interpreter output");
    for (mode, threading) in [("strict", "multi"), ("strict", "single"), ("managed", "multi"), ("managed", "single")] {
        let out_dir = dir.join(format!("out_{mode}_{threading}"));
        let out = boring(&dir, &["build", "main.br", "--mode", mode, "--threading", threading,
                                 "--output-dir", out_dir.to_str().unwrap()]);
        assert!(out.status.success(), "{name} [{mode}+{threading}]: build failed:\n{}", text(&out.stderr));
        let manifest = out_dir.join("Cargo.toml");
        let toml = std::fs::read_to_string(&manifest).unwrap();
        std::fs::write(&manifest, toml.replacen("name = \"main\"", &format!("name = \"qcl_{name}_{mode}_{threading}\""), 1)).unwrap();
        let run = common::SharedCargo::new("qualified_collection_literals")
            .args(["run", "--quiet", "--manifest-path"])
            .arg(out_dir.join("Cargo.toml"))
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap();
        assert!(run.status.success(), "{name} [{mode}+{threading}]: generated project failed:\n{}", text(&run.stderr));
        assert_eq!(text(&run.stdout), expected, "{name} [{mode}+{threading}]: compiled output");
    }
}

/// Lock-qualified elements: a lock-handle field is shared into the collection, a fresh value is
/// wrapped in its own lock, and a `def` call through an element or tuple slot mutates the shared
/// value.
fn lock_program(q: &str) -> String {
    format!(r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Svc:
    Counter'{q} stats = Counter()

mut s = Svc()
var [mut Counter'{q}] a = [s.stats, Counter()]
var {{string = mut Counter'{q}}} d = {{"x" = s.stats, "y" = Counter()}}
var (mut Counter'{q}, int) t = (s.stats, 1)
a[0].inc()
d["x"].inc()
t.0.inc()
a[1].inc()
d["y"].inc()
d["y"].inc()
print "{{s.stats.get()}} {{a[1].get()}} {{d["y"].get()}} {{t.0.get()}}"
"#)
}

#[test]
fn actor_elements_in_array_dict_and_tuple() {
    check("actor_elems", &lock_program("actor"), "3 1 2 3\n");
}

#[test]
fn guard_elements_in_array_dict_and_tuple() {
    check("guard_elems", &lock_program("guard"), "3 1 2 3\n");
}

/// Non-lock qualifiers: `'shared` / `'owned` / `'inline` elements in every literal kind, sets
/// included (a lock handle is not Hash/Eq, so only these three can be set elements).
fn plain_program(q: &str) -> String {
    // A managed-mode `'owned` value is a lock cell, which has no Hash/Eq: no set of them.
    let (set_decl, set_len) = if q == "owned" { ("", "2") } else { ("var {Counter'Q} z = {Counter(n = 1), Counter(n = 9)}\n", "z.length") };
    let set_decl = set_decl.replace('Q', q);
    format!(r#"#derive(Eq, Hash)
struct Counter:
    var int n = 0
    req int get():
        n

var [Counter'{q}] a = [Counter(), Counter(n = 4)]
var {{string = Counter'{q}}} d = {{"x" = Counter(n = 2)}}
var (Counter'{q}, int) t = (Counter(n = 7), 1)
{set_decl}print "{{a[1].get()}} {{d["x"].get()}} {{t.0.get()}} {{{set_len}}}"
"#)
}

#[test]
fn shared_elements_in_every_literal_kind() {
    check("shared_elems", &plain_program("shared"), "4 2 7 2\n");
}

#[test]
fn owned_elements_in_every_literal_kind() {
    check("owned_elems", &plain_program("owned"), "4 2 7 2\n");
}

#[test]
fn inline_elements_in_every_literal_kind() {
    check("inline_elems", &plain_program("inline"), "4 2 7 2\n");
}

#[test]
fn lock_handles_cannot_be_set_elements() {
    for q in ["actor", "guard"] {
        let dir = project(&format!("set_{q}"), &format!(
            "struct Counter:\n    var int n = 0\n\nvar {{Counter'{q}}} z = {{Counter()}}\nprint \"ok\"\n"));
        let out = boring(&dir, &["run", "main.br"]);
        assert!(!out.status.success(), "{q}: a lock handle set must be rejected");
        assert!(text(&out.stderr).contains("cannot be a set element"), "{q}: {}", text(&out.stderr));
    }
}
