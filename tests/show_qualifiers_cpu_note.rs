// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// `--show-qualifiers` explains a plain `'actor` picked by default where the `'task` variant
// was also possible and the value left for a function the inference could not look into
// (docs/qualifiers.md, "undecidable cases"). The choice itself is unchanged: only the report
// gains a note.
//
// Run with:
//   cargo test --test show_qualifiers_cpu_note

use std::path::Path;
use std::process::Command;

fn show_qualifiers(src: &str, name: &str) -> String {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("show_qualifiers_cpu_note_test_scratch")
        .join(format!("{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).expect("failed to create scratch dir");
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, src).expect("failed to write fixture");
    let out = Command::new(env!("CARGO_BIN_EXE_boring"))
        .arg("build")
        .arg(&br_file)
        .arg("--emit-rust")
        .arg("--show-qualifiers")
        .output()
        .expect("failed to invoke boring");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

const COUNTER: &str = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n\n";

#[test]
fn unseen_callee_gets_a_note_on_the_default_plain_variant() {
    let src = format!("{COUNTER}def void run(mut Counter c):\n    task c.inc()\n    external_fn(c)\n");
    let report = show_qualifiers(&src, "unseen");
    assert!(report.contains("run::c  'actor"), "{report}");
    assert!(report.contains("note: plain variant picked by default: passed to `external_fn`"), "{report}");
}

#[test]
fn a_visible_callee_needs_no_note() {
    let src = format!(
        "{COUNTER}def void run(mut Counter c):\n    task c.inc()\n    opaque(c)\n\n\
         task def void opaque(mut Counter c):\n    with c:\n        c.inc()\n        wait(Duration.fromMillis(10))\n"
    );
    let report = show_qualifiers(&src, "visible");
    assert!(report.contains("run::c  'actor'task"), "{report}");
    assert!(!report.contains("note:"), "{report}");
}

#[test]
fn no_note_without_a_task_capture() {
    let src = format!("{COUNTER}def void run(mut Counter c):\n    external_fn(c)\n");
    let report = show_qualifiers(&src, "nocapture");
    assert!(!report.contains("note:"), "{report}");
}

#[test]
fn a_chain_declared_in_reverse_order_propagates_the_task_lock_to_its_head() {
    // `a` calls `b` calls `c`; only `c` awaits while holding its parameter. All three
    // parameters must agree on the tokio lock, whatever the declaration order.
    let src = format!(
        "{COUNTER}task def void a(mut Counter x):\n    b(x)\n\n\
         task def void b(mut Counter y):\n    c(y)\n\n\
         task def void c(mut Counter z):\n    with z:\n        z.inc()\n        wait(Duration.fromMillis(10))\n"
    );
    let report = show_qualifiers(&src, "chain");
    for name in ["a::x", "b::y", "c::z"] {
        assert!(report.contains(&format!("{name}  'actor'task")), "{name}: {report}");
    }
}

// --- the reason printed after each decision -------------------------------------------------

const REPORT_SRC: &str = "struct Counter:\n    var int value = 0\n    def inc():\n        value += 1\n\n\
def bump(mut Counter c):\n    c.inc()\n\ndef show(Counter c):\n    print \"{c.value}\"\n\n\
def grow(mut [int] xs):\n    xs.push(1)\n\ndef main():\n    var mut c = Counter(0)\n    bump(c)\n    show(c)\n    let d = Counter(1)\n    let e = d\n    show(e)\n    mut [int] xs = []\n    grow(xs)\n";

fn line_of<'a>(report: &'a str, name: &str) -> &'a str {
    report.lines().find(|l| l.contains(&format!("{name}  "))).unwrap_or_else(|| panic!("no line for {name}:\n{report}"))
}

#[test]
fn a_read_only_parameter_says_it_is_borrowed_because_nothing_stores_it() {
    let report = show_qualifiers(REPORT_SRC, "why_borrow");
    let l = line_of(&report, "show::c");
    assert!(l.contains("& (borrow)") && l.contains("no storage signal") && l.contains("borrowed read-only"), "{l}");
}

#[test]
fn a_mut_parameter_says_it_is_lent_mutably() {
    let report = show_qualifiers(REPORT_SRC, "why_mutborrow");
    assert!(line_of(&report, "bump::c").contains("`mut`: lent as `&mut`"), "{report}");
    assert!(line_of(&report, "grow::xs").contains("always lent as `&mut`"), "{report}");
}

#[test]
fn an_inline_fallback_names_the_size_and_the_threshold() {
    let report = show_qualifiers(REPORT_SRC, "why_inline");
    let l = line_of(&report, "main::d");
    assert!(l.contains("'inline") && l.contains("8 bytes <= --inline-auto-bytes 256"), "{l}");
}

#[test]
fn an_alias_says_whose_qualifier_it_shares() {
    let report = show_qualifiers(REPORT_SRC, "why_alias");
    assert!(line_of(&report, "main::e").contains("alias of `d`"), "{report}");
}

#[test]
fn a_ruled_out_inline_and_a_task_lock_are_explained() {
    let src = format!(
        "{COUNTER}def void run(mut Counter c):\n    task c.inc()\n\n\
         task def void a(mut Counter x):\n    b(x)\n\n\
         task def void b(mut Counter y):\n    with y:\n        y.inc()\n        wait(Duration.fromMillis(10))\n"
    );
    let report = show_qualifiers(&src, "why_task");
    assert!(line_of(&report, "run::c").contains("`'inline` ruled out by a storage signal"), "{report}");
    assert!(line_of(&report, "a::x").contains("the tokio lock"), "{report}");
}
