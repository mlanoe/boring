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

#[test]
fn full_shows_a_trait_the_transpiler_made_shared_and_the_rust_is_the_same() {
    // A trait whose methods are all `req`, held by a struct field, is shared (`Trait'shared`): `mono`
    // keeps what was written, `full` writes the decision, and building the `.bri` gives the same Rust.
    let src = "trait Shape:\n    req float area()\n\nstruct Circle as Shape:\n    float r\n\n    req float area():\n        r * r * 3.0\n\nstruct Scene:\n    var [Shape] items = []\n\ndef main():\n    var mut s = Scene()\n    s.items.push(Circle(2.0))\n    print \"{s.items.len()}\"\n";
    let dir = scratch("bri_cli_shared_trait");
    std::fs::write(dir.join("shape.br"), src).unwrap();
    let (mono, _) = text(&boring(&dir, &["check", "--emit-bri=mono", "shape.br"]));
    let (full, _) = text(&boring(&dir, &["check", "--emit-bri=full", "shape.br"]));
    assert!(mono.contains("var [Shape] items"), "{mono}");
    assert!(full.contains("var [Shape'shared] items"), "{full}");
    std::fs::write(dir.join("shape_full.bri"), &full).unwrap();
    let rust = |f: &str| text(&boring(&dir, &["build", "--emit-rust", f])).0;
    assert_eq!(rust("shape.br"), rust("shape_full.bri"));
}

#[test]
fn full_keeps_the_trailing_array_block_of_a_shared_trait_array_working() {
    // The `.bri` writes `[Widget'shared]` on the field the block collects into; the sugar still reads
    // it as a trait-object array, and the builder it makes is shared too (same Rust as the source).
    let dir = scratch("bri_cli_array_block_shared");
    let case = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cases/array_block_for.br");
    std::fs::copy(&case, dir.join("blocks.br")).unwrap();
    let (full, err) = text(&boring(&dir, &["check", "--emit-bri=full", "blocks.br"]));
    assert!(full.contains("[Widget'shared] children"), "{full}\n{err}");
    std::fs::write(dir.join("blocks_full.bri"), &full).unwrap();
    let rust = |f: &str| text(&boring(&dir, &["build", "--emit-rust", f])).0
        .lines().filter(|l| !l.contains("__stream_")).collect::<Vec<_>>().join("\n");
    assert_eq!(rust("blocks.br"), rust("blocks_full.bri"));
}
