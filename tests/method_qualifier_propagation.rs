// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Qualifier inference of a parameter propagates through calls whatever the declaration order,
// for free functions AND for the methods of structs, enums and `ext` blocks: a caller declared
// before its callee, a chain in reverse order, mutual recursion, a method forwarding to a free
// function (and the reverse), and across files. Before, only free functions were pre-inferred, so
// `first(Counter c): self.second(c)` declared before `second(Counter c): spawn(c)` kept a plain
// `&Counter` parameter and emitted `self.second(c.clone())` where an `Arc<Mutex<Counter>>`
// was expected (rustc E0308). Each case runs under `boring run` and as a compiled `boring build`.
//
// Run with:
//   tools/test.sh --test method_qualifier_propagation

mod common;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("method_qualifier_propagation").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    for (file, src) in files {
        let p = dir.join(file);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, src).unwrap();
    }
    dir
}

fn boring(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_boring")).current_dir(dir).args(args).output().unwrap()
}

fn text(b: &[u8]) -> String { String::from_utf8_lossy(b).into_owned() }

/// Runs `main.br` under the interpreter and as a compiled project; both must print `expected`.
fn check(name: &str, files: &[(&str, &str)], expected: &str) {
    let dir = project(name, files);
    let out = boring(&dir, &["run", "main.br"]);
    assert!(out.status.success(), "{name}: run failed:\n{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), expected, "{name}: interpreter output");

    let out = boring(&dir, &["build", "main.br"]);
    assert!(out.status.success(), "{name}: build failed:\n{}", text(&out.stderr));
    // Every generated package is called `main`: give each its own name so the shared target
    // directory never runs another test's executable.
    let manifest = dir.join("main_rust/Cargo.toml");
    let toml = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(&manifest, toml.replacen("name = \"main\"", &format!("name = \"mqp_{name}\""), 1)).unwrap();
    let out = common::SharedCargo::new("method_qualifier_propagation")
        .args(["run", "--quiet", "--manifest-path"])
        .arg(dir.join("main_rust/Cargo.toml"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap();
    assert!(out.status.success(), "{name}: generated project failed to compile/run:\n{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), expected, "{name}: compiled output");
}

const METHOD_BEFORE_METHOD_WITH_ALIAS_RECEIVER: &str = r#"struct Counter:
    var int value = 0
    def inc():
        value += 1

def bump(Counter'actor c):
    print "bump {c.value}"

struct Svc:
    int n
    req first(Counter c):
        self.second(c)
    req second(Counter c):
        bump(c)

def main():
    let c = Counter(value = 7)
    let s = Svc(n = 1)
    s.first(c)
    s.first(c)
"#;

#[test]
fn method_before_method_with_alias_receiver() {
    check("method_before_method_with_alias_receiver", &[("main.br", METHOD_BEFORE_METHOD_WITH_ALIAS_RECEIVER)], "bump 7\nbump 7\n");
}

const METHOD_BEFORE_FREE_FN_AND_FREE_FN_BEFORE_CALLEE: &str = r#"struct Counter:
    var int value = 0
    def inc():
        value += 1

struct Svc:
    int n
    req direct(Counter c):
        bump(c)
    req viaFree(Counter c):
        relay(c)

def relay(Counter c):
    bump(c)

def bump(Counter'actor c):
    print "bump {c.value}"

def main():
    let c = Counter(value = 7)
    let s = Svc(n = 1)
    s.direct(c)
    s.viaFree(c)
"#;

#[test]
fn method_before_free_fn_and_free_fn_before_callee() {
    check("method_before_free_fn_and_free_fn_before_callee", &[("main.br", METHOD_BEFORE_FREE_FN_AND_FREE_FN_BEFORE_CALLEE)], "bump 7\nbump 7\n");
}

const FREE_FN_BEFORE_METHOD: &str = r#"struct Counter:
    var int value = 0
    def inc():
        value += 1

def feed(Svc s, Counter c):
    s.take(c)

def bump(Counter'actor c):
    print "bump {c.value}"

struct Svc:
    int n
    req take(Counter c):
        bump(c)

def main():
    let c = Counter(value = 7)
    let s = Svc(n = 1)
    feed(s, c)
    feed(s, c)
"#;

#[test]
fn free_fn_before_method() {
    check("free_fn_before_method", &[("main.br", FREE_FN_BEFORE_METHOD)], "bump 7\nbump 7\n");
}

const MUTUALLY_RECURSIVE_METHODS: &str = r#"struct Counter:
    var int value = 0
    def inc():
        value += 1

def bump(Counter'actor c):
    print "bump {c.value}"

struct Svc:
    int n
    req ping(Counter c, int k):
        if k > 0:
            self.pong(c, k - 1)
        else:
            bump(c)
    req pong(Counter c, int k):
        if k > 0:
            self.ping(c, k - 1)
        else:
            bump(c)

def main():
    let c = Counter(value = 7)
    let s = Svc(n = 1)
    s.ping(c, 5)
    s.ping(c, 2)
"#;

#[test]
fn mutually_recursive_methods() {
    check("mutually_recursive_methods", &[("main.br", MUTUALLY_RECURSIVE_METHODS)], "bump 7\nbump 7\n");
}

const ENUM_AND_EXT_METHODS: &str = r#"struct Counter:
    var int value = 0
    def inc():
        value += 1

def bump(Counter'actor c):
    print "bump {c.value}"

enum Mode:
    A
    B
    req outer(Counter c):
        self.inner(c)
    req inner(Counter c):
        bump(c)

struct Tag:
    int n

ext Tag:
    req outer(Counter c):
        self.inner(c)
    req inner(Counter c):
        bump(c)

def main():
    let c = Counter(value = 7)
    let Mode m = Mode.A
    m.outer(c)
    let t = Tag(n = 1)
    t.outer(c)
"#;

#[test]
fn enum_and_ext_methods() {
    check("enum_and_ext_methods", &[("main.br", ENUM_AND_EXT_METHODS)], "bump 7\nbump 7\n");
}

const MF_SVC: &str = r#"struct Counter:
    var int value = 0
    def inc():
        value += 1

def bump(Counter'actor c):
    print "bump {c.value}"

struct Svc:
    int n
    req outerLib(Counter c):
        self.innerLib(c)
    req innerLib(Counter c):
        bump(c)
"#;
const MF_MAIN: &str = r#"use svc

ext Svc:
    req outerMain(Counter c):
        self.innerLib(c)

struct Local:
    int k
    req go(Counter c):
        Svc(n = 2).outerMain(c)

def main():
    let c = Counter(value = 3)
    let l = Local(k = 1)
    l.go(c)
    Svc(n = 1).outerLib(c)
"#;

#[test]
fn methods_across_files() {
    check("methods_across_files", &[("svc.br", MF_SVC), ("main.br", MF_MAIN)], "bump 3\nbump 3\n");
}

const FREE_FNS: &str = r#"struct Counter:
    var int value = 0

def a(Counter c):
    b(c)

def b(Counter c):
    d(c)

def d(Counter c):
    bump(c)

def ping(Counter c, int k):
    if k > 0:
        pong(c, k - 1)
    else:
        bump(c)

def pong(Counter c, int k):
    if k > 0:
        ping(c, k - 1)
    else:
        bump(c)

def bump(Counter'actor c):
    print "bump {c.value}"

def main():
    let c = Counter(value = 7)
    a(c)
    ping(c, 3)
"#;

#[test]
fn free_fns_reverse_chain_and_mutual_recursion() {
    check("free_fns", &[("main.br", FREE_FNS)], "bump 7\nbump 7\n");
}

const WRITTEN_QUALIFIER: &str = r#"struct Counter:
    var int value = 0

struct Svc:
    int n
    req first(Counter c):
        self.second(c)
    req second(Counter'actor c):
        print "second {c.value}"

def main():
    let c = Counter(value = 5)
    let s = Svc(n = 1)
    s.first(c)
"#;

#[test]
fn written_qualifier_on_callee_is_kept() {
    check("written", &[("main.br", WRITTEN_QUALIFIER)], "second 5\n");
}
