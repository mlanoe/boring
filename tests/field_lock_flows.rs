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

// ---------------------------------------------------------------------------------------------
// Implicit copies of a lock handle are rejected with the fix in the message, instead of an
// opaque rustc error (E0599 / E0308) in the generated project.
// ---------------------------------------------------------------------------------------------

const COPY_PRELUDE: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Svc:
    var Counter stats = Counter()

def bump(mut Counter'actor c):
    c.inc()

"#;

fn build_error(name: &str, body: &str) -> String {
    let dir = project(name, &[("main.br", &format!("{COPY_PRELUDE}{body}"))]);
    let out = boring(&dir, &["build", "main.br", "--output-dir", dir.join("out").to_str().unwrap()]);
    assert!(!out.status.success(), "{name}: build should be rejected");
    // `boring run` must reject the very same program with the very same diagnostic.
    let run = boring(&dir, &["run", "main.br"]);
    assert!(!run.status.success(), "{name}: run should be rejected too");
    let (b, r) = (text(&out.stderr), text(&run.stderr));
    let first = |t: &str| t.lines().find(|l| l.contains("shared lock handle") || l.contains("shared handle")).unwrap_or("").to_string();
    assert_eq!(first(&b), first(&r), "{name}: run and build diagnostics differ:\nbuild: {b}\nrun: {r}");
    b
}

#[test]
fn lock_field_in_untyped_array_literal_is_rejected() {
    let err = build_error("copy_array_literal", "mut s = Svc()\nbump(s.stats)\nvar arr = [s.stats]\nprint \"{arr.length}\"\n");
    assert!(err.contains("`s.stats` is a shared lock handle") && err.contains("array literal"), "{err}");
    assert!(err.contains("[Counter'actor]"), "message must name the fix: {err}");
}

#[test]
fn lock_field_returned_under_bare_type_is_rejected() {
    let err = build_error("copy_bare_return", "Counter pick(Svc s):\n    s.stats\n\nmut s = Svc()\nbump(s.stats)\nvar c = pick(s)\nprint \"{c.get()}\"\n");
    assert!(err.contains("`s.stats` is a shared lock handle") && err.contains("plain return type `Counter`"), "{err}");
    assert!(err.contains("Counter'actor"), "message must name the fix: {err}");
}

#[test]
fn plain_field_in_untyped_array_literal_still_copies() {
    // Not a lock anywhere: value semantics, no diagnostic.
    check("plain_field_array_literal", &[("main.br", r#"struct Counter:
    var int n = 0

struct Svc:
    var Counter stats = Counter()

mut s = Svc()
var arr = [s.stats]
print "{arr.length}"
"#)], "1\n");
}

#[test]
fn no_param_annotation_hint_warning() {
    let dir = project("no_param_hint", &[("main.br", &format!("{COPY_PRELUDE}def bump2(mut Counter c):\n    bump(c)\n\nmut c = Counter()\nbump2(c)\nprint \"{{c.get()}}\"\n"))]);
    let out = boring(&dir, &["build", "main.br", "--output-dir", dir.join("out").to_str().unwrap()]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    assert!(!text(&out.stderr).contains("consider annotating"), "{}", text(&out.stderr));
}

// ---------------------------------------------------------------------------------------------
// `boring run` and the compiled program agree on by-value reads of a struct field: a plain
// field is copied (let, assignment, bare return, method return, nested path, array/dict
// literal element, constructor argument), a lock-handle field is shared.
// ---------------------------------------------------------------------------------------------

#[test]
fn element_read_copies_unless_elements_are_handles() {
    // `let q = arr[0]`: a copy when the element type is plain (`[mut Counter'inline]` opts out of
    // the promotion), the same object when the elements are `'actor` handles. `run` and `build`
    // must print the same thing.
    check("elem_read_copy", &[("main.br", r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

mut [mut Counter'inline] plain = [Counter()]
mut q = plain[0]
q.inc()
print "plain {plain[0].get()} {q.get()}"

mut [mut Counter'actor] shared = [Counter()]
mut h = shared[0]
h.inc()
print "shared {shared[0].get()} {h.get()}"

mut untyped = [Counter()]
mut u = untyped[0]
u.inc()
print "untyped {untyped[0].get()} {u.get()}"

[mut Counter'actor] mkShared():
    mut [mut Counter'actor] r = [Counter()]
    r

mut fromFn = mkShared()
mut f = fromFn[0]
f.inc()
print "fromFn {fromFn[0].get()} {f.get()}"
"#)], "plain 0 1\nshared 1 1\nuntyped 0 1\nfromFn 1 1\n");
}

#[test]
fn plain_field_reads_are_copies_in_run_and_build() {
    check("plain_field_copies", &[("main.br", r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Svc:
    var Counter stats = Counter()

struct Outer:
    var Svc svc = Svc()

struct Holder:
    var mut Counter c = Counter()

struct Box2:
    var Counter inner = Counter()
    def Counter take():
        inner

Counter pick(Svc s):
    s.stats

mut s = Svc()
mut a = s.stats
a.inc()
print "let {s.stats.get()} {a.get()}"

var mut Counter d = Counter()
d = s.stats
d.inc()
print "assign {s.stats.get()} {d.get()}"

mut r = pick(s)
r.inc()
print "return {s.stats.get()} {r.get()}"

mut b = Box2()
mut t = b.take()
t.inc()
print "method {b.inner.get()} {t.get()}"

mut o = Outer()
mut n = o.svc.stats
n.inc()
print "nested {o.svc.stats.get()} {n.get()}"

var [mut Counter] arr = [s.stats]
arr[0].inc()
print "array {s.stats.get()} {arr[0].get()}"

var {string = mut Counter} dd = {"k" = s.stats}
dd["k"].inc()
print "dict {s.stats.get()} {dd["k"].get()}"

mut h = Holder(c = s.stats)
h.c.inc()
print "ctor {s.stats.get()} {h.c.get()}"
"#)], "let 0 1\nassign 0 1\nreturn 0 1\nmethod 0 1\nnested 0 1\narray 0 1\ndict 0 1\nctor 0 1\n");
}

// ---------------------------------------------------------------------------------------------
// A plain (unqualified) struct field copied into an array/dict/tuple literal whose elements then
// need a shared handle: the copy would silently swallow the callee's mutations. Rejected with the
// two fixes in the message; read-only literals and already-handle fields stay valid.
// ---------------------------------------------------------------------------------------------

const BARE_PRELUDE: &str = r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Svc:
    var Counter stats = Counter()
    var Counter other = Counter()

def bump(mut Counter'actor c):
    c.inc()

struct Runner:
    def go(mut Counter'actor c):
        c.inc()

"#;

fn bare_error(name: &str, body: &str) -> String {
    let dir = project(name, &[("main.br", &format!("{BARE_PRELUDE}{body}"))]);
    let out = boring(&dir, &["build", "main.br", "--output-dir", dir.join("out").to_str().unwrap()]);
    assert!(!out.status.success(), "{name}: build should be rejected");
    let err = text(&out.stderr);
    assert!(err.contains("`s.stats` is copied into a literal"), "{name}: {err}");
    assert!(err.contains("Counter'actor stats") && err.contains("mut c = s.stats"), "{name}: message must name both fixes: {err}");
    err
}

#[test]
fn bare_field_copied_into_literal_then_needing_a_handle_is_rejected() {
    bare_error("bare_array", "mut s = Svc()\nvar arr = [s.stats]\nbump(arr[0])\n");
    bare_error("bare_for", "mut s = Svc()\nfor x in [s.stats]:\n    bump(x)\n");
    bare_error("bare_dict", "mut s = Svc()\nvar d = {\"a\" = s.stats}\nbump(d[\"a\"])\n");
    bare_error("bare_tuple", "mut s = Svc()\nvar t = (s.stats, 1)\nbump(t.0)\n");
    bare_error("bare_for_over_local", "mut s = Svc()\nvar arr = [s.stats]\nfor x in arr:\n    bump(x)\n");
    bare_error("bare_method_param", "mut s = Svc()\nmut r = Runner()\nfor x in [s.stats]:\n    r.go(x)\n");
    bare_error("bare_task_capture", "mut s = Svc()\nfor x in [s.stats]:\n    task:\n        x.inc()\n");
}

#[test]
fn bare_field_literals_without_a_handle_demand_stay_valid() {
    // Read-only use of copies: value semantics, no diagnostic.
    check("bare_readonly", &[("main.br", &format!("{BARE_PRELUDE}mut s = Svc()\nvar arr = [s.stats, s.other]\nvar t = 0\nfor x in arr:\n    t += x.get()\nprint \"{{t}}\"\n"))], "0\n");
    // A local that is not a field, and a handle kept in a local: both fine.
    check("bare_local_ok", &[("main.br", &format!("{BARE_PRELUDE}mut s = Svc()\nmut local = Counter()\nfor x in [local]:\n    bump(x)\nmut c = s.stats\nbump(c)\nprint \"{{c.get()}}\"\n"))], "1\n");
    // Field written with the qualifier: a shared handle, not a copy.
    check("bare_actor_field", &[("main.br", r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Holder:
    Counter'actor c = Counter()

def bump(mut Counter'actor c):
    c.inc()

mut h = Holder()
bump(h.c)
print "{h.c.get()}"
"#)], "1\n");
}

#[test]
fn lock_field_in_set_literal_is_rejected() {
    let dir = project("lock_set", &[("main.br", &format!("{BARE_PRELUDE}mut s = Svc()\nbump(s.stats)\nvar st = {{s.stats}}\nprint \"ok\"\n"))]);
    let out = boring(&dir, &["build", "main.br", "--output-dir", dir.join("out").to_str().unwrap()]);
    assert!(!out.status.success());
    assert!(text(&out.stderr).contains("cannot be a set element"), "{}", text(&out.stderr));
}

// ---------------------------------------------------------------------------------------------
// Two read-only calls through the same lock in ONE statement. A guard is a temporary of the
// enclosing statement, so `e.get()` and `arr[0].get()` (the same element) used to deadlock a
// `Mutex` (and panic a `RefCell`): each `req` call now releases its guard right after the call.
// ---------------------------------------------------------------------------------------------

#[test]
fn two_reads_of_one_lock_in_one_statement() {
    check("two_reads_one_statement", &[("main.br", r#"struct Counter:
    var int n = 0
    def inc():
        n += 1
    req int get():
        n

struct Svc:
    var Counter stats = Counter()

def bump(mut Counter'actor c):
    c.inc()

def show(int a, int b):
    print "show {a} {b}"

def main():
    var [mut Counter] arr = [Counter()]
    mut e = arr[0]
    e.inc()
    print "interp {e.get()} {arr[0].get()}"
    let sum = e.get() + arr[0].get()
    print "sum {sum}"
    show(e.get(), arr[0].get())
    mut s = Svc()
    bump(s.stats)
    print "field {s.stats.get()} {s.stats.get()}"
"#)], "interp 1 1\nsum 2\nshow 1 1\nfield 1 1\n");
}

// Scope and receiver precision of the copied-field check, and lock fields in dict/tuple literals.

const AMBIG_PRELUDE: &str = r#"struct Other:
    def go(Counter c):
        print "o {c.get()}"

"#;

#[test]
fn method_receiver_is_resolved_by_type_when_the_name_is_ambiguous() {
    // `go` exists on Runner ('actor param) and Other (plain param): the typed receiver decides.
    let dir = project("bare_ambiguous_method", &[("main.br", &format!("{BARE_PRELUDE}{AMBIG_PRELUDE}mut s = Svc()\nmut r = Runner()\nfor x in [s.stats]:\n    r.go(x)\n"))]);
    let out = boring(&dir, &["build", "main.br", "--output-dir", dir.join("out").to_str().unwrap()]);
    assert!(!out.status.success());
    assert!(text(&out.stderr).contains("`s.stats` is copied into a literal"), "{}", text(&out.stderr));
    let dir = project("bare_ambiguous_method_ok", &[("main.br", &format!("{BARE_PRELUDE}{AMBIG_PRELUDE}mut s = Svc()\nmut o = Other()\nfor x in [s.stats]:\n    o.go(x)\n"))]);
    let out = boring(&dir, &["build", "main.br", "--output-dir", dir.join("out").to_str().unwrap()]);
    assert!(out.status.success(), "{}", text(&out.stderr));
}

#[test]
fn copied_field_check_is_scoped_to_the_binding() {
    // A same-named local in another block does not make the earlier read-only literal an error.
    let dir = project("bare_shadow_ok", &[("main.br", &format!("{BARE_PRELUDE}mut s = Svc()\nvar arr = [s.stats]\nprint \"{{arr.length}}\"\nif true:\n    var arr = [Counter()]\n    bump(arr[0])\n"))]);
    let out = boring(&dir, &["build", "main.br", "--output-dir", dir.join("out").to_str().unwrap()]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    // The same literal inside a nested block is still caught.
    bare_error("bare_nested_block", "mut s = Svc()\nif true:\n    var arr = [s.stats]\n    bump(arr[0])\n");
}

#[test]
fn lock_field_in_untyped_dict_and_tuple_literal_is_rejected() {
    for (name, lit, kind) in [("lock_dict", "var d = {\"a\" = s.stats}", "dict literal"), ("lock_tuple", "var t = (s.stats, 1)", "tuple literal")] {
        let err = build_error(name, &format!("mut s = Svc()\nbump(s.stats)\n{lit}\n"));
        assert!(err.contains("`s.stats` is a shared lock handle") && err.contains(kind), "{name}: {err}");
    }
}
