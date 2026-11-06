// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The qualifier of a struct field with none written is inferred from the uses of the field in
// EVERY file of the project, not only from the struct's own methods: `svc.stats` handed to a
// function that needs `Counter'task`, in another file than `struct Service`, makes the field an
// `Arc<tokio::sync::Mutex<Counter>>` (its constructor argument is wrapped too). Before, only the
// struct's own file was scanned and the field stayed a plain `Counter`.
//
// These tests check the emitted Rust (`boring build`), and `boring run` for the interpreter side.
// They do not cargo-build the result: a project whose field ends up a lock handle does not yet
// compile for reasons independent of the inference (a lock-handle field passed as an argument is
// re-wrapped by move, a struct holding one still derives `Eq`/`Hash`, a read through a local of
// the struct is not locked), which hit a field written `Counter'task` by hand just the same.
//
// Run with:
//   tools/test.sh --test field_inference_across_files

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn project(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("field_inference_across_files").join(name);
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

/// `boring build main.br`: the generated Rust sources, concatenated.
fn build_rust(dir: &Path) -> String {
    let out = boring(dir, &["build", "main.br"]);
    assert!(out.status.success(), "build failed:\n{}", text(&out.stderr));
    // Imported files are emitted as sibling modules: look at all of them.
    let mut files: Vec<_> = std::fs::read_dir(dir.join("main_rust/src")).unwrap()
        .map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "rs")).collect();
    files.sort();
    files.iter().map(|p| std::fs::read_to_string(p).unwrap()).collect::<Vec<_>>().join("\n")
}

fn run_ok(dir: &Path) -> String {
    let out = boring(dir, &["run", "main.br"]);
    assert!(out.status.success(), "run failed:\n{}", text(&out.stderr));
    text(&out.stdout)
}

const COUNTER: &str = "struct Counter:\n    var int n = 0\n    def inc():\n        n += 1\n    req int get():\n        n\n\n";
/// The struct under test, declared in its own file with the field left bare.
fn a_src(field: &str) -> String {
    format!("{COUNTER}struct Service:\n    {field}\n    int id = 0\n")
}

const LOCKED: &str = "stats: Arc<tokio::sync::Mutex<Counter>>,";
const WRAPPED_CTOR: &str = "Service { stats: Arc::new(tokio::sync::Mutex::new(Counter { n: 0 }))";

fn assert_field_is_lock(rs: &str) {
    assert!(rs.contains(LOCKED), "field not inferred as a lock handle:\n{rs}");
}

/// The field is a lock handle and the one construction site wraps its argument accordingly.
fn assert_field_is_lock_and_ctor_wrapped(rs: &str) {
    assert_field_is_lock(rs);
    assert!(rs.contains(WRAPPED_CTOR), "constructor argument not wrapped:\n{rs}");
}

/// The field is a lock handle of either flavour (`'actor` → std Mutex, `'actor'task` → tokio Mutex).
fn assert_field_is_some_lock(rs: &str) {
    assert!(rs.contains("stats: Arc<std::sync::Mutex<Counter>>,") || rs.contains(LOCKED),
        "field not inferred as a lock handle:\n{rs}");
}

fn assert_field_is_plain(rs: &str) {
    assert!(rs.contains("stats: Counter,"), "field should stay a plain Counter:\n{rs}");
}

const BUMP_TASK: &str = "task def bump(mut Counter'task c):\n    c.inc()\n\n";
const MAIN_USES_BUMP: &str = "task go():\n    mut svc = Service(stats = Counter())\n    bump(svc.stats)\n    print \"n = {svc.stats.get()}\"\n\nlet t = task go()\nt.wait\n";

#[test]
fn use_in_the_entry_file_decides_the_field_of_an_imported_struct() {
    let a = a_src("mut Counter stats");
    let main = format!("use a\n\n{BUMP_TASK}{MAIN_USES_BUMP}");
    let dir = project("entry", &[("a.br", &a), ("main.br", &main)]);
    assert_field_is_lock_and_ctor_wrapped(&build_rust(&dir));
}

#[test]
fn use_in_a_third_file_decides_the_field() {
    let a = a_src("mut Counter stats");
    let b = format!("use a\n\n{BUMP_TASK}task def feed(Service s):\n    bump(s.stats)\n");
    let main = "use a\nuse b\n\ndef main():\n    print \"ok\"\n";
    let dir = project("third", &[("a.br", &a), ("b.br", &b), ("main.br", main)]);
    assert_field_is_lock(&build_rust(&dir));
}

#[test]
fn result_does_not_depend_on_the_order_of_the_files() {
    let a = a_src("mut Counter stats");
    let b = format!("use a\n\n{BUMP_TASK}task def feed(Service s):\n    bump(s.stats)\n");
    let c = "use a\n\nint label(Service s):\n    s.id\n";
    for (i, uses) in ["use a\nuse b\nuse c\n", "use c\nuse b\nuse a\n", "use b\nuse a\nuse c\n"].iter().enumerate() {
        let main = format!("{uses}\ndef main():\n    print \"ok\"\n");
        let dir = project(&format!("order{i}"), &[("a.br", &a), ("b.br", &b), ("c.br", c), ("main.br", &main)]);
        assert_field_is_lock(&build_rust(&dir));
    }
}

#[test]
fn cycle_between_the_struct_file_and_the_file_using_it() {
    // a.br declares the struct and uses b.br, which uses a.br back and holds the demand.
    let a = format!("use b\n\n{}", a_src("mut Counter stats"));
    let b = format!("use a\n\n{BUMP_TASK}task def feed(Service s):\n    bump(s.stats)\n");
    let main = "use a\nuse b\n\ndef main():\n    print \"ok\"\n";
    let dir = project("cycle", &[("a.br", &a), ("b.br", &b), ("main.br", main)]);
    assert_field_is_lock(&build_rust(&dir));
}

#[test]
fn demand_reaches_the_field_through_a_forwarding_function() {
    // `relay`'s parameter is bare: it only becomes `Counter'task` because it forwards to `bump`,
    // and the field passed to `relay` follows.
    let a = a_src("mut Counter stats");
    let main = format!("use a\n\n{BUMP_TASK}task def relay(mut Counter c):\n    bump(c)\n\n\
                        task go():\n    mut svc = Service(stats = Counter())\n    relay(svc.stats)\n\n\
                        let t = task go()\nt.wait\n");
    let dir = project("transitive", &[("a.br", &a), ("main.br", &main)]);
    assert_field_is_lock_and_ctor_wrapped(&build_rust(&dir));
}

#[test]
fn field_through_a_chain_of_structs() {
    let a = a_src("mut Counter stats");
    let main = format!("use a\n\nstruct Outer:\n    mut Service inner\n\n{BUMP_TASK}\
                        task go():\n    mut o = Outer(inner = Service(stats = Counter()))\n    bump(o.inner.stats)\n\n\
                        let t = task go()\nt.wait\n");
    let dir = project("chain", &[("a.br", &a), ("main.br", &main)]);
    assert_field_is_lock_and_ctor_wrapped(&build_rust(&dir));
}

#[test]
fn a_field_nobody_constrains_stays_plain() {
    let a = a_src("mut Counter stats");
    let main = "use a\n\ndef main():\n    mut svc = Service(stats = Counter())\n    svc.stats.inc()\n    print \"n = {svc.stats.get()}\"\n";
    let dir = project("plain", &[("a.br", &a), ("main.br", main)]);
    assert_field_is_plain(&build_rust(&dir));
    assert_eq!(run_ok(&dir), "n = 1\n");
}

#[test]
fn a_written_qualifier_is_never_overwritten() {
    let a = a_src("mut Counter'inline stats");
    let main = format!("use a\n\n{BUMP_TASK}{MAIN_USES_BUMP}");
    let dir = project("written", &[("a.br", &a), ("main.br", &main)]);
    assert_field_is_plain(&build_rust(&dir));
}

#[test]
fn interpreter_is_unaffected() {
    let a = a_src("mut Counter stats");
    let main = format!("use a\n\n{BUMP_TASK}{MAIN_USES_BUMP}");
    let dir = project("run", &[("a.br", &a), ("main.br", &main)]);
    assert_eq!(run_ok(&dir), "n = 1\n");
}

#[test]
fn conflicting_files_are_an_error_naming_both() {
    let a = a_src("mut Counter stats");
    let b = format!("use a\n\n{BUMP_TASK}task def feed(Service s):\n    bump(s.stats)\n");
    let c = "use a\n\ndef freeze(Service s):\n    pin(s.stats)\n\ndef pin(Counter'inline c):\n    print c.get()\n";
    let main = "use a\nuse b\nuse c\n\ndef main():\n    print \"ok\"\n";
    let dir = project("conflict", &[("a.br", &a), ("b.br", &b), ("c.br", c), ("main.br", main)]);
    let out = boring(&dir, &["build", "main.br"]);
    assert!(!out.status.success(), "conflicting requirements accepted:\n{}", text(&out.stdout));
    let err = text(&out.stderr);
    assert!(err.contains("conflicting qualifier requirements for field `Service.stats`"), "got:\n{err}");
    assert!(err.contains("`feed`") && err.contains("`freeze`"), "both users should be named:\n{err}");
}

#[test]
fn conflict_message_is_the_same_whatever_the_order() {
    let a = a_src("mut Counter stats");
    let b = format!("use a\n\n{BUMP_TASK}task def feed(Service s):\n    bump(s.stats)\n");
    let c = "use a\n\ndef freeze(Service s):\n    pin(s.stats)\n\ndef pin(Counter'inline c):\n    print c.get()\n";
    let msg = |uses: &str, name: &str| {
        let main = format!("{uses}\ndef main():\n    print \"ok\"\n");
        let dir = project(name, &[("a.br", &a), ("b.br", &b), ("c.br", c), ("main.br", &main)]);
        let out = boring(&dir, &["build", "main.br"]);
        assert!(!out.status.success());
        text(&out.stderr).lines().filter(|l| l.contains("conflicting")).collect::<Vec<_>>().join("\n")
    };
    assert_eq!(msg("use a\nuse b\nuse c\n", "ord1"), msg("use c\nuse b\nuse a\n", "ord2"));
}


// ── Walker gaps: bare fields, task bodies, method parameters, task capture emission ──────────
// Same-file and cross-file share one field walker; each case below is checked in the struct's own
// file, and the bare-field one also from another file's `ext` block.

fn single(name: &str, body: &str) -> String {
    let dir = project(name, &[("main.br", &format!("{COUNTER}{body}"))]);
    build_rust(&dir)
}

const MAIN_TAIL: &str = "\nlet s = Service(stats = Counter())\nprint \"ok\"\n";

#[test]
fn bare_field_name_in_a_method_is_a_use_of_self_field() {
    let rs = single("bare", &format!("{BUMP_TASK}struct Service:\n    mut Counter stats\n\n    task def feed():\n        bump(stats)\n{MAIN_TAIL}"));
    assert_field_is_lock_and_ctor_wrapped(&rs);
}

#[test]
fn bare_field_name_in_an_ext_block_of_another_file() {
    let a = a_src("mut Counter stats");
    let b = format!("use a\n\n{BUMP_TASK}ext Service:\n    task def feed():\n        bump(stats)\n");
    let main = "use a\nuse b\n\ndef main():\n    print \"ok\"\n";
    let dir = project("bare_ext", &[("a.br", &a), ("b.br", &b), ("main.br", main)]);
    assert_field_is_lock(&build_rust(&dir));
}

#[test]
fn a_parameter_shadowing_the_field_is_not_a_use_of_the_field() {
    let rs = single("bare_shadow", &format!("{BUMP_TASK}struct Service:\n    mut Counter stats\n\n    task def feed(mut Counter'task stats):\n        bump(stats)\n{MAIN_TAIL}"));
    assert_field_is_plain(&rs);
}

#[test]
fn a_local_shadowing_the_field_is_not_a_use_of_the_field() {
    let rs = single("bare_local", &format!("{BUMP_TASK}struct Service:\n    mut Counter stats\n\n    task def feed():\n        mut stats = Counter()\n        bump(stats)\n{MAIN_TAIL}"));
    assert_field_is_plain(&rs);
}

#[test]
fn def_call_on_the_field_inside_a_task_body_block_is_seen() {
    let rs = single("task_block", &format!("struct Service:\n    mut Counter stats\n\n    task def go():\n        let t = task:\n            self.stats.inc()\n            self.stats.inc()\n        t.wait\n{MAIN_TAIL}"));
    assert_field_is_some_lock(&rs);
}

#[test]
fn field_passed_to_a_method_parameter_is_a_demand() {
    let worker = "struct Worker:\n    int id = 0\n    task def take(mut Counter'task c):\n        c.inc()\n\n";
    let rs = single("method_param", &format!("{worker}struct Service:\n    mut Counter stats\n\n    task def feed(mut Worker w):\n        w.take(stats)\n{MAIN_TAIL}"));
    assert_field_is_lock(&rs);
}

#[test]
fn field_passed_to_a_method_parameter_of_a_struct_in_another_file() {
    let a = a_src("mut Counter stats");
    let b = "use a\n\nstruct Worker:\n    int id = 0\n    task def take(mut Counter'task c):\n        c.inc()\n\ntask def feed(Service s, mut Worker w):\n    w.take(s.stats)\n";
    let main = "use a\nuse b\n\ndef main():\n    print \"ok\"\n";
    let dir = project("method_param_cross", &[("a.br", &a), ("b.br", b), ("main.br", main)]);
    assert_field_is_lock(&build_rust(&dir));
}

#[test]
fn task_capturing_a_lock_field_clones_the_handle_before_the_spawn() {
    // `tokio::spawn(async move { self.stats ... })` borrows `self` for 'static (E0521).
    let rs = single("task_capture", &format!("struct Service:\n    mut Counter stats\n\n    task def go():\n        let t = task:\n            stats.inc()\n        t.wait\n{MAIN_TAIL}"));
    assert_field_is_some_lock(&rs);
    assert!(rs.contains("let __self_stats = self.stats.clone();"), "handle not cloned before the spawn:\n{rs}");
    assert!(rs.contains("__self_stats.lock()"), "task body must use the clone:\n{rs}");
    assert!(!rs.contains("async move {\n        self.stats"), "task body still borrows self:\n{rs}");
}
