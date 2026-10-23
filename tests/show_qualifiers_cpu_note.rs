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
