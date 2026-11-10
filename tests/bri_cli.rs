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
fn full_leaves_out_what_a_local_or_a_parameter_is_by_default() {
    // `inline` locals and borrowed parameters are the defaults: spelling them would be a
    // constraint (it changes the Rust), not a report of what was inferred.
    let dir = scratch("bri_cli_full");
    let mono = text(&boring(&dir, &["check", "--emit-bri=mono", "prog.br"])).0;
    let full = text(&boring(&dir, &["check", "--emit-bri=full", "prog.br"])).0;
    assert_eq!(mono, full, "nothing here is a non-default decision");
    let again = text(&boring(&dir, &["build", "--emit-rust", "prog.br"])).0;
    std::fs::write(dir.join("full.bri"), &full).unwrap();
    assert_eq!(again, text(&boring(&dir, &["build", "--emit-rust", "full.bri"])).0);
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

#[test]
fn full_on_a_kernel_program_warns_and_prints_the_mono_text() {
    let dir = scratch("bri_cli_full_kernel");
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cases/kernel_aux_method_fast_path.br"), dir.join("k.br")).unwrap();
    let mono = text(&boring(&dir, &["check", "--emit-bri=mono", "k.br"])).0;
    let o = boring(&dir, &["check", "--emit-bri=full", "k.br"]);
    let (out, err) = text(&o);
    assert!(o.status.success(), "{err}");
    assert!(err.contains("the .bri has no inferred qualifiers"), "{err}");
    assert_eq!(out, mono);
}

#[test]
fn full_keeps_a_borrowed_array_parameter_usable_by_the_array_block_sugar() {
    let dir = scratch("bri_cli_full_sugar");
    std::fs::copy(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cases/array_block_lowercase_collect.br"), dir.join("k.br")).unwrap();
    let full = text(&boring(&dir, &["check", "--emit-bri=full", "k.br"])).0;
    std::fs::write(dir.join("k.bri"), full).unwrap();
    let o = boring(&dir, &["k.bri"]);
    assert!(o.status.success(), "{}", text(&o).1);
}

#[test]
fn full_writes_the_residency_of_tensor_operands_and_metal_output_is_the_same() {
    let dir = scratch("bri_cli_full_gpu");
    std::fs::write(dir.join("t.br"), "\
def run([float32, n = 4, k = 4] x, [float32, k = 4, m = 4] w, mut [float32, n = 4, m = 4] y) throws:
    kernel.tensors.matmul(x, w, y)

let [float32, n = 4, k = 4] x = [1.0 as float32 for ..<16]
let [float32, k = 4, m = 4] w = [1.0 as float32 for ..<16]
var [float32, n = 4, m = 4] y = [0.0 as float32 for ..<16]
run(x, w, y)
").unwrap();
    let full = text(&boring(&dir, &["check", "--emit-bri=full", "t.br"])).0;
    assert!(full.contains("    kernel.tensors.matmul(x, w, y)"), "{full}");
    assert!(full.contains("[float32, n = 4, k = 4]'global x"), "{full}");
    assert!(full.contains("mut [float32, n = 4, m = 4]'unified y"), "{full}");
    assert!(full.contains("var [float32, n = 4, m = 4]'unified y"), "{full}");
    std::fs::write(dir.join("t_full.bri"), &full).unwrap();
    for f in ["t.br", "t_full.bri"] {
        let o = boring(&dir, &["build", "--emit-rust", "--target", "metal", f]);
        assert!(o.status.success(), "{f}: {}", text(&o).1);
    }
    let read = |p: &str| std::fs::read_to_string(dir.join(p).join("src/main.rs")).unwrap_or_else(|e| panic!("{p}: {e}"));
    assert_eq!(read("t_metal"), read("t_full_metal"));
}

#[test]
fn a_specialised_copy_says_what_it_is_a_copy_of_and_the_file_builds_the_same_rust() {
    let dir = scratch("bri_cli_origin");
    std::fs::write(dir.join("o.br"), "\
struct Counter:
    var int n = 0
    def inc(): n += 1

def bump(var mut a):
    a.inc()

def main():
    var mut Counter'actor c = Counter()
    bump(c)
    print \"{c.n}\"
").unwrap();
    let bri = text(&boring(&dir, &["check", "--emit-bri", "o.br"])).0;
    assert!(bri.contains("def bump_Counter_actor(var mut Counter'actor a):  # copy of bump (line 5) for a = Counter'actor, first called at line 10"), "{bri}");
    std::fs::write(dir.join("o2.bri"), &bri).unwrap();
    assert_eq!(text(&boring(&dir, &["build", "--emit-rust", "o.br"])).0, text(&boring(&dir, &["build", "--emit-rust", "o2.bri"])).0);
}

#[test]
fn full_shows_the_instances_the_monomorphizer_made_and_the_file_runs_the_same() {
    let dir = scratch("bri_cli_mono");
    let case = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cases");
    for name in ["qualifier_fns", "qualifier_gaps_enums"] {
        std::fs::copy(case.join(format!("{name}.br")), dir.join(format!("{name}.br"))).unwrap();
        let full = text(&boring(&dir, &["check", "--emit-bri=full", &format!("{name}.br")])).0;
        // The generic declarations are gone, their instances are written.
        let code: String = full.lines().filter(|l| !l.trim_start().starts_with('#')).collect::<Vec<_>>().join("\n");
        assert!(!code.contains("<'elem") && !code.contains("<'q"), "{name}:\n{full}");
        std::fs::write(dir.join(format!("{name}_full.bri")), &full).unwrap();
        let original = boring(&dir, &[&format!("{name}.br")]);
        let again = boring(&dir, &[&format!("{name}_full.bri")]);
        assert!(again.status.success(), "{name}: {}", text(&again).1);
        assert_eq!(text(&original).0, text(&again).0, "{name}");
    }
    let full = text(&boring(&dir, &["check", "--emit-bri=full", "qualifier_gaps_enums.br"])).0;
    assert!(full.contains("enum Slot_inline") && full.contains("Slot_inline::") || full.contains("Slot_inline."), "{full}");
}
