// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `--emit-bri[=mono|full]` of `boring check` and `boring build` (src/bri.rs).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_boring");

const PROGRAM: &str = "\
struct Counter:
    var int n = 0
    def inc(): n += 1

def show(x):
    print \"{x}\"

def main():
    let c = Counter()
    mut m = Counter()
    m.inc()
    show(m.n)
    print \"{c.n}\"
";

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("prog.br"), PROGRAM).unwrap();
    dir
}

fn boring(dir: &Path, args: &[&str]) -> Output {
    Command::new(BIN).args(args).current_dir(dir).output().unwrap()
}

fn text(o: &Output) -> (String, String) {
    (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned())
}

#[test]
fn check_prints_a_valid_program_on_stdout() {
    let dir = scratch("bri_cli_stdout");
    for flag in ["--emit-bri", "--emit-bri=mono"] {
        let o = boring(&dir, &["check", flag, "prog.br"]);
        let (out, err) = text(&o);
        assert!(o.status.success(), "{err}");
        // the implicit generic is written back untyped, with the bounds the compiler derived
        assert!(out.contains("def show(x):  # inferred: T_x as Display"), "{out}");
        assert!(out.contains("struct Counter:"), "{out}");
        // the printed file runs and prints what the source prints
        std::fs::write(dir.join("again.bri"), &out).unwrap();
        let (a, _) = text(&boring(&dir, &["prog.br"]));
        let (b, _) = text(&boring(&dir, &["again.bri"]));
        assert_eq!(a, b);
        assert_eq!(a, "1\n0\n");
    }
}

#[test]
fn check_out_dir_writes_the_file() {
    let dir = scratch("bri_cli_out");
    let o = boring(&dir, &["check", "--emit-bri", "--out", "outdir", "prog.br"]);
    assert!(o.status.success(), "{}", text(&o).1);
    assert!(text(&o).0.is_empty(), "nothing on stdout with --out");
    let written = std::fs::read_to_string(dir.join("outdir/prog.bri")).unwrap();
    assert!(written.contains("def main():"), "{written}");
}

#[test]
fn full_writes_the_inferred_qualifiers_mono_does_not() {
    let dir = scratch("bri_cli_full");
    let mono = text(&boring(&dir, &["check", "--emit-bri=mono", "prog.br"])).0;
    let full = text(&boring(&dir, &["check", "--emit-bri=full", "prog.br"])).0;
    assert!(!mono.contains("'inline"), "{mono}");
    assert!(full.contains("let c'inline = Counter()"), "{full}");
    assert!(full.contains("mut m'inline = Counter()") || full.contains("mut Counter'inline m"), "{full}");
}

#[test]
fn unknown_level_is_an_error() {
    let dir = scratch("bri_cli_level");
    let o = boring(&dir, &["check", "--emit-bri=banana", "prog.br"]);
    assert!(!o.status.success());
    assert!(text(&o).1.contains("unknown --emit-bri level 'banana'"), "{}", text(&o).1);
}

#[test]
fn build_writes_the_bri_into_the_project_and_it_builds_the_same_rust() {
    let dir = scratch("bri_cli_build");
    let o = boring(&dir, &["build", "--emit-bri", "prog.br"]);
    assert!(o.status.success(), "{}", text(&o).1);
    let bri = dir.join("prog_rust/bri/prog.bri");
    assert!(bri.is_file(), "expected {}", bri.display());
    // `boring build x.bri` is a build of Boring source: same Rust as x.br
    let o = boring(&dir, &["build", "prog_rust/bri/prog.bri", "--output-dir", "again_rust"]);
    assert!(o.status.success(), "{}", text(&o).1);
    let a = std::fs::read_to_string(dir.join("prog_rust/src/main.rs")).unwrap();
    let b = std::fs::read_to_string(dir.join("again_rust/src/main.rs")).unwrap();
    assert_eq!(a, b);
}

#[test]
fn build_emit_bri_does_not_combine_with_emit_rust() {
    let dir = scratch("bri_cli_conflict");
    let o = boring(&dir, &["build", "--emit-rust", "--emit-bri", "prog.br"]);
    assert!(!o.status.success());
    assert!(text(&o).1.contains("--emit-bri"), "{}", text(&o).1);
}
