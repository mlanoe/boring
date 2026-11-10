// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Typed specialisation across files (src/specialize_untyped.rs, `specialize_imported`): a `pub`
// function of a library with an untyped parameter used by a method call is specialised by the
// files that call it, each with a typed copy; the library never emits the generic original.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_boring");

const LIB: &str = "\
pub struct Counter:
    var int n = 0
    def inc(): n += 1

pub def bump(var mut a):
    a.inc()

pub def twice(var mut b):
    bump(b)
    bump(b)
";

const MAIN: &str = "\
use lib

def main():
    var mut Counter c = Counter()
    var mut Counter'actor d = Counter()
    bump(c)
    twice(c)
    twice(d)
    print \"{c.n} {d.n}\"
";

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lib.br"), LIB).unwrap();
    std::fs::write(dir.join("main.br"), MAIN).unwrap();
    dir
}

fn boring(dir: &Path, args: &[&str]) -> Output {
    Command::new(BIN).args(args).current_dir(dir).output().unwrap()
}

fn stderr(o: &Output) -> String { String::from_utf8_lossy(&o.stderr).into_owned() }

#[test]
fn run_calls_the_typed_copies() {
    let dir = scratch("spec_cross_run");
    let o = boring(&dir, &["main.br"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "3 2");
}

#[test]
fn the_caller_gets_the_copies_and_the_library_no_generic_original() {
    let dir = scratch("spec_cross_build");
    let o = boring(&dir, &["build", "main.br"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let files: Vec<String> = ["main_rust/src/main.rs", "main_rust/src/lib.rs"].iter()
        .map(|p| std::fs::read_to_string(dir.join(p)).unwrap_or_default()).collect();
    for name in ["fn bump_Counter(", "fn twice_Counter(", "fn bump_Counter_actor(", "fn twice_Counter_actor("] {
        assert!(files[0].contains(name), "main.rs lacks {name}");
    }
    assert!(!files[1].contains("fn bump") && !files[1].contains("fn twice"), "lib.rs:\n{}", files[1]);
}

#[test]
fn a_call_with_an_argument_of_unknown_type_is_an_error_naming_it() {
    let dir = scratch("spec_cross_untyped");
    std::fs::write(dir.join("main.br"), "use lib\n\ndef main():\n    var mut c = Counter()\n    bump(c)\n").unwrap();
    let o = boring(&dir, &["main.br"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("cannot specialise `bump`"), "{}", stderr(&o));
}

#[test]
fn bri_of_the_library_keeps_the_original_and_it_is_still_specialise_only() {
    let dir = scratch("spec_cross_bri");
    let o = boring(&dir, &["check", "--emit-bri", "lib.br"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let bri = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(bri.contains("pub def bump(") && !bri.contains("@specialise_only"), "{bri}");
    std::fs::write(dir.join("lib2.bri"), &bri).unwrap();
    let o = boring(&dir, &["build", "--emit-rust", "lib2.bri"]);
    assert!(o.status.success(), "{}", stderr(&o));
}

const LIB_METHODS: &str = "\
pub struct Counter:
    var int n = 0
    def inc(): n += 1

pub struct Runner:
    var int runs = 0

    pub def bump(var mut a):
        a.inc()
        runs += 1

    pub def twice(var mut b):
        self.bump(b)
        self.bump(b)
";

const MAIN_METHODS: &str = "\
use lib

def main():
    var mut Runner r = Runner()
    var mut Counter c = Counter()
    var mut Counter'actor d = Counter()
    r.bump(c)
    r.twice(c)
    r.twice(d)
    print \"{c.n} {d.n} {r.runs}\"
";

fn scratch_methods(name: &str) -> PathBuf {
    let dir = scratch(name);
    std::fs::write(dir.join("lib.br"), LIB_METHODS).unwrap();
    std::fs::write(dir.join("main.br"), MAIN_METHODS).unwrap();
    dir
}

#[test]
fn a_pub_method_of_a_library_struct_is_specialised_by_its_callers() {
    let dir = scratch_methods("spec_cross_methods_run");
    let o = boring(&dir, &["main.br"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "3 2 5");
}

#[test]
fn a_local_method_of_the_same_name_is_an_error_naming_both() {
    let dir = scratch_methods("spec_cross_methods_shadow");
    std::fs::write(dir.join("main.br"), "use lib\n\nstruct Other:\n    def bump(int x): print \"{x}\"\n\ndef main():\n    var mut Runner r = Runner()\n    var mut Counter c = Counter()\n    r.bump(c)\n").unwrap();
    let o = boring(&dir, &["main.br"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("`bump` of `Runner`") && stderr(&o).contains("rename the local method"), "{}", stderr(&o));
}

#[test]
fn the_copies_of_a_method_go_in_an_ext_of_the_caller_and_the_library_has_none() {
    let dir = scratch_methods("spec_cross_methods_build");
    let o = boring(&dir, &["build", "main.br"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let main = std::fs::read_to_string(dir.join("main_rust/src/main.rs")).unwrap();
    let lib = std::fs::read_to_string(dir.join("main_rust/src/lib.rs")).unwrap_or_default();
    for name in ["fn bump_Counter(", "fn twice_Counter(", "fn bump_Counter_actor(", "fn twice_Counter_actor("] {
        assert!(main.contains(name), "main.rs lacks {name}");
    }
    assert!(!lib.contains("fn bump") && !lib.contains("fn twice"), "lib.rs:\n{lib}");
}

const LIB_MODS: &str = "\
pub struct Counter:
    var int n = 0
    def inc(): n += 1

pub mod tools:
    pub def bump(var mut a):
        a.inc()

    pub def twice(var mut b):
        bump(b)
        bump(b)

    pub mod deep:
        pub def triple(var mut c):
            c.inc()
            c.inc()
            c.inc()
";

const MAIN_MODS: &str = "\
use lib

def main():
    var mut Counter c = Counter()
    var mut Counter'actor d = Counter()
    bump(c)
    twice(d)
    triple(c)
    print \"{c.n} {d.n}\"
";

fn scratch_mods(name: &str) -> PathBuf {
    let dir = scratch(name);
    std::fs::write(dir.join("lib.br"), LIB_MODS).unwrap();
    std::fs::write(dir.join("main.br"), MAIN_MODS).unwrap();
    dir
}

#[test]
fn a_pub_function_of_a_library_mod_is_specialised_by_its_callers() {
    let dir = scratch_mods("spec_cross_mods_run");
    let o = boring(&dir, &["main.br"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "4 2");
}

#[test]
fn the_copies_of_a_mod_function_go_in_the_caller_and_the_library_has_none() {
    let dir = scratch_mods("spec_cross_mods_build");
    let o = boring(&dir, &["build", "main.br"]);
    assert!(o.status.success(), "{}", stderr(&o));
    let main = std::fs::read_to_string(dir.join("main_rust/src/main.rs")).unwrap();
    let lib = std::fs::read_to_string(dir.join("main_rust/src/lib.rs")).unwrap_or_default();
    for name in ["fn bump_Counter(", "fn twice_Counter_actor(", "fn triple_Counter(", "fn bump_Counter_actor("] {
        assert!(main.contains(name), "main.rs lacks {name}");
    }
    assert!(!lib.contains("fn bump") && !lib.contains("fn twice") && !lib.contains("fn triple"), "lib.rs:\n{lib}");
}

#[test]
fn a_pub_method_of_a_struct_declared_in_a_library_mod_is_specialised_by_its_callers() {
    let dir = scratch("spec_cross_mod_struct");
    std::fs::write(dir.join("lib.br"), "\
pub struct Counter:
    var int n = 0
    def inc(): n += 1

pub mod engine:
    pub struct Runner:
        var int runs = 0

        pub def bump(var mut a):
            a.inc()
            runs += 1

        pub def twice(var mut b):
            self.bump(b)
            self.bump(b)
").unwrap();
    std::fs::write(dir.join("main.br"), MAIN_METHODS).unwrap();
    let o = boring(&dir, &["main.br"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert_eq!(String::from_utf8_lossy(&o.stdout).trim(), "3 2 5");
    let b = boring(&dir, &["build", "main.br"]);
    assert!(b.status.success(), "{}", stderr(&b));
    let main = std::fs::read_to_string(dir.join("main_rust/src/main.rs")).unwrap();
    assert!(main.contains("fn bump_Counter(") && main.contains("fn twice_Counter_actor("), "main.rs");
}
