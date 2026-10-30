// Copyright (C) 2026 Mickaël LANOË
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Pins the by-name rule of method borrow inference (docs/qualifiers.md, "Methods": a read-only
// struct parameter of a method is lent as `&T`). A method name shared by several types is NOT
// disqualifying: what counts is that *every* method bearing the name (and no trait, field,
// piped/bare value or labeled call site) is a plain method and they all borrow the same
// positions. Then each `Type::name` is lent, for resolved receivers and (by name) for receivers of
// unknown type alike. When the types disagree (one only reads the parameter, one returns it), every
// one of them stays by value.
//
// Run with:
//   cargo test --test method_param_borrow_name_rule

use std::path::Path;
use std::process::Command;

fn emit_rust(src: &str, dir: &Path) -> String {
    let bin = env!("CARGO_BIN_EXE_boring");
    std::fs::create_dir_all(dir).expect("failed to create scratch dir");
    let br_file = dir.join("main.br");
    std::fs::write(&br_file, src).expect("failed to write fixture .br file");
    let emit = Command::new(bin).arg("build").arg(&br_file).arg("--emit-rust").output().expect("boring");
    assert!(emit.status.success(), "emit failed:\n{}", String::from_utf8_lossy(&emit.stderr));
    String::from_utf8_lossy(&emit.stdout).into_owned()
}

const PRELUDE: &str = "struct Point:
    float x
    float y

";

#[test]
fn same_name_on_two_types_agreeing_is_borrowed_for_resolved_and_unresolved_receivers() {
    let src = format!("{PRELUDE}struct Counter:
    float k

    req float peek(Point p):
        p.x * k

struct Other:
    float k

    req float peek(Point p):
        p.y + k

def main():
    let p = Point(1.0, 2.0)
    let c = Counter(3.0)
    let o = Other(4.0)
    print c.peek(p)
    print o.peek(p)
    let f = (x): x.peek(p)
    print f(c)
");
    let rs = emit_rust(&src, Path::new("target/method_param_borrow_name_rule_test_scratch/agree"));
    assert_eq!(rs.matches("fn peek(&self, p: &Point)").count(), 2, "both `peek` take `&Point`:\n{rs}");
    assert!(rs.contains("c.peek(&*p)") && rs.contains("o.peek(&*p)"), "resolved call sites lend:\n{rs}");
    assert!(rs.contains("x.peek(&*p)"), "the unresolved receiver is looked up by name and lends too:\n{rs}");
}

#[test]
fn same_name_disagreeing_keeps_every_type_by_value() {
    // `Other.take` destructures a field of its parameter, which needs the owned value
    // (qualifier inference gives up the borrow); `Counter.take` alone would be borrowed.
    let src = "struct Pair:
    (float, float) xy

struct Counter:
    float k

    req float take(Pair p):
        p.xy.0 * k

struct Other:
    float k

    req float take(Pair p):
        let (a, b) = p.xy
        a + b + k

def main():
    let p = Pair((1.0, 2.0))
    let c = Counter(3.0)
    let o = Other(4.0)
    print c.take(p)
    print o.take(p)
";
    let rs = emit_rust(src, Path::new("target/method_param_borrow_name_rule_test_scratch/differ"));
    assert_eq!(rs.matches("fn take(&self, p: Pair)").count(), 2, "both stay by value:\n{rs}");
}
