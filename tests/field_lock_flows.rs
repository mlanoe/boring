// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// A struct field that is a lock handle (`'actor`/`'guard`, written or inferred) used as a VALUE:
// handed to a parameter that demands the same lock (lent, never re-wrapped into a fresh lock
// around a copy), to a plain borrowed parameter (borrowed through its guard), bound to a local,
// returned, or reached through an array element / `for` variable. Each case runs under
// `boring run` and as a compiled project in all four mode/threading configurations; the
// interpreter and the four builds must print the same thing. The last case is a multi-file
// project combining cross-file parameter inference, cross-file field inference and all of the
// above.
//
// Run with:
//   tools/test.sh --test field_lock_flows

mod common;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("field_lock_flows").join(name);
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

fn check(name: &str, files: &[(&str, &str)], expected: &str) {
    let dir = project(name, files);
    let out = boring(&dir, &["run", "main.br"]);
    assert!(out.status.success(), "{name}: run failed:\n{}", text(&out.stderr));
    assert_eq!(text(&out.stdout), expected, "{name}: interpreter output");
    for (mode, threading) in [("strict", "multi"), ("strict", "single"), ("managed", "multi"), ("managed", "single")] {
        let out_dir = dir.join(format!("out_{mode}_{threading}"));
        let out = boring(&dir, &["build", "main.br", "--mode", mode, "--threading", threading,
                                 "--output-dir", out_dir.to_str().unwrap()]);
        assert!(out.status.success(), "{name} [{mode}+{threading}]: build failed:\n{}", text(&out.stderr));
        // Every generated package is called `main`: give each its own name so the shared target
        // directory never runs another test's executable.
        let manifest = out_dir.join("Cargo.toml");
        let toml = std::fs::read_to_string(&manifest).unwrap();
        std::fs::write(&manifest, toml.replacen("name = \"main\"", &format!("name = \"flf_{name}_{mode}_{threading}\""), 1)).unwrap();
        let run = common::SharedCargo::new("field_lock_flows")
            .args(["run", "--quiet", "--manifest-path"])
            .arg(out_dir.join("Cargo.toml"))
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .unwrap();
        assert!(run.status.success(), "{name} [{mode}+{threading}]: generated project failed:\n{}", text(&run.stderr));
        assert_eq!(text(&run.stdout), expected, "{name} [{mode}+{threading}]: compiled output");
    }
}

const FIELD_TO_ACTOR_PARAM: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

def bump(mut Counter'actor c):
    c.inc()

def show(Counter c):
    print "n = {c.get()}"

mut svc = Service(stats = Counter())
bump(svc.stats)
show(svc.stats)
"#;

#[test]
fn field_to_actor_param() {
    check("field_to_actor_param", &[("main.br", FIELD_TO_ACTOR_PARAM)], "n = 1\n");
}

const FIELD_TO_BORROWED_PARAM: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

task def bump(mut Counter'task c):
    c.inc()


def show(Counter c):
    print "n = {c.get()}"

task go():
    mut svc = Service(stats = Counter())
    bump(svc.stats)
    show(svc.stats)

let t = task go()
t.wait
"#;

#[test]
fn field_to_borrowed_param() {
    check("field_to_borrowed_param", &[("main.br", FIELD_TO_BORROWED_PARAM)], "n = 1\n");
}

const LOCAL_AND_FIELD_TO_BORROWED_PARAM: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

def bump(mut Counter'actor c):
    c.inc()

def show(Counter c):
    print "n = {c.get()}"

mut Counter'actor c = Counter()
bump(c)
show(c)
mut svc = Service(stats = Counter())
bump(svc.stats)
show(svc.stats)
"#;

#[test]
fn local_and_field_to_borrowed_param() {
    check("local_and_field_to_borrowed_param", &[("main.br", LOCAL_AND_FIELD_TO_BORROWED_PARAM)], "n = 1\nn = 1\n");
}

const ALIAS_OF_FIELD: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

task def bump(mut Counter'task c):
    c.inc()


task go():
    mut svc = Service(stats = Counter())
    mut c = svc.stats
    bump(c)
    print "n = {svc.stats.get()}"

let t = task go()
t.wait
"#;

#[test]
fn alias_of_field() {
    check("alias_of_field", &[("main.br", ALIAS_OF_FIELD)], "n = 1\n");
}

const ARRAY_ELEMENTS_AND_FOR_VARIABLES: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

task def bump(mut Counter'task c):
    c.inc()


task go():
    mut svcs = [Service(stats = Counter())]
    for s in svcs:
        bump(s.stats)
    bump(svcs[0].stats)
    print "n = {svcs[0].stats.get()}"

let t = task go()
t.wait
"#;

#[test]
fn array_elements_and_for_variables() {
    check("array_elements_and_for_variables", &[("main.br", ARRAY_ELEMENTS_AND_FOR_VARIABLES)], "n = 2\n");
}

const FIELD_RETURNED_BY_EXPLICIT_HANDLE_RETURN_TYPE: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

task def bump(mut Counter'task c):
    c.inc()


Counter'actor'task pick(Service s):
    s.stats

task go():
    mut svc = Service(stats = Counter())
    bump(pick(svc))
    print "n = {svc.stats.get()}"

let t = task go()
t.wait
"#;

#[test]
fn field_returned_by_explicit_handle_return_type() {
    check("field_returned_by_explicit_handle_return_type", &[("main.br", FIELD_RETURNED_BY_EXPLICIT_HANDLE_RETURN_TYPE)], "n = 1\n");
}

const EXPLICIT_HANDLE_FLOWS: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter'actor stats
    int id = 0

def bump(mut Counter'actor c):
    c.inc()

mut svcs = [Service(stats = Counter())]
for s in svcs:
    bump(s.stats)
bump(svcs[0].stats)
print "n = {svcs[0].stats.get()}"
"#;

#[test]
fn explicit_handle_flows() {
    check("explicit_handle_flows", &[("main.br", EXPLICIT_HANDLE_FLOWS)], "n = 2\n");
}

const MUT_METHOD_PARAM_DEMANDING_ACTOR: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

def bump(mut Counter'actor c):
    c.inc()

struct Walker:
    int id = 0
    def ping(mut Counter c, int k):
        if k > 0:
            self.pong(c, k - 1)
    def pong(mut Counter c, int k):
        bump(c)
        if k > 0:
            self.ping(c, k - 1)

mut w = Walker()
mut c = Counter()
w.ping(c, 5)
print "n = {c.get()}"
"#;

#[test]
fn mut_method_param_demanding_actor() {
    check("mut_method_param_demanding_actor", &[("main.br", MUT_METHOD_PARAM_DEMANDING_ACTOR)], "n = 3\n");
}

const TASK_LOCAL_INFERRED_TASK_ACTOR: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

task def bump(mut Counter'task c):
    c.inc()
def show(Counter c):
    print "n = {c.get()}"

task go():
    mut c = Counter()
    bump(c)
    bump(c)
    print "n = {c.get()}"

let t = task go()
t.wait
"#;

#[test]
fn task_local_inferred_task_actor() {
    check("task_local_inferred_task_actor", &[("main.br", TASK_LOCAL_INFERRED_TASK_ACTOR)], "n = 2\n");
}

const TYPED_ARRAY_OF_FIELD_HANDLES_INFERRED_FIELD: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

def bump(mut Counter'actor c):
    c.inc()

mut svc = Service(stats = Counter())
let [Counter'actor] arr = [svc.stats]
bump(arr[0])
print "n = {svc.stats.get()}"
"#;

#[test]
fn typed_array_of_field_handles_inferred_field() {
    check("typed_array_of_field_handles_inferred_field", &[("main.br", TYPED_ARRAY_OF_FIELD_HANDLES_INFERRED_FIELD)], "n = 1\n");
}

const FOR_VARIABLE_OVER_ARRAY_OF_HANDLES: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

def bump(mut Counter'actor c):
    c.inc()

mut svc = Service(stats = Counter())
mut other = Service(stats = Counter())
let [Counter'actor] arr = [svc.stats, other.stats]
for x in arr:
    bump(x)
bump(arr[1])
print "n = {svc.stats.get()} {other.stats.get()}"
"#;

#[test]
fn for_variable_over_array_of_handles() {
    check("for_variable_over_array_of_handles", &[("main.br", FOR_VARIABLE_OVER_ARRAY_OF_HANDLES)], "n = 1 2\n");
}

const XF_MODEL: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Service:
    mut Counter stats
    int id = 0

    def record():
        stats.inc()

struct Fleet:
    mut [Service] members
"#;

const XF_OPS: &str = r#"use model

def bumpTwice(mut Counter'actor c):
    c.inc()
    c.inc()

def relay(mut Counter c):
    bumpTwice(c)

def show(Counter c):
    print "n = {c.get()}"

Counter'actor handleOf(Service s):
    s.stats

def feed(Service s):
    relay(s.stats)
    show(s.stats)
    mut alias = s.stats
    alias.inc()
"#;

const XF_MAIN: &str = r#"use model
use ops

mut svc = Service(stats = Counter())
feed(svc)
svc.record()
show(svc.stats)
relay(handleOf(svc))
mut fleet = Fleet(members = [Service(stats = Counter())])
for m in fleet.members:
    feed(m)
show(fleet.members[0].stats)
print "total = {svc.stats.get() + fleet.members[0].stats.get()}"
"#;

#[test]
fn cross_file_inference_and_fields_end_to_end() {
    check("cross_file", &[("model.br", XF_MODEL), ("ops.br", XF_OPS), ("main.br", XF_MAIN)],
          "n = 2\nn = 4\nn = 2\nn = 3\ntotal = 9\n");
}
