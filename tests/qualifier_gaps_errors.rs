//! Diagnostics of the qualifier-parameter gaps closed after phase 5 (design note, "Known gaps
//! closed"): `boring run` and `boring build` report the same error at the same position. Only a
//! written qualifier is a proof; a receiver, a `return` or a pattern that does not determine the
//! instance stays an error instead of a guess.
use std::path::Path;
use std::process::Command;

/// (whole file, message fragment, 1-based `line:col` of the reported span).
const CASES: &[(&str, &str, &str)] = &[
    // 1. Method-call arguments: an unknown receiver proves nothing.
    ("struct Holder:\n    var int n = 0\n\n    def put(Board<elem'actor> b):\n        n += b.size()\n\n\
def main():\n    let h = Holder()\n    h.put(Board())\n",
        "cannot infer qualifier parameter `elem` of `Board`, write `Board<elem'...>`", "18:11"),
    // 1. The method's parameter and a written argument disagree: a conflict, not a guess.
    ("struct Holder:\n    var int n = 0\n\n    def put(Board<elem'inline> b):\n        n += b.size()\n\n\
def main():\n    let Holder h = Holder()\n    let Point'actor p = Point(1)\n    h.put(Board(cells = [p]))\n",
        "conflicting qualifiers for `elem` of `Board`: `'inline` (the expected type) and `'actor` (declared on `p`)", "19:11"),
    // 2. A nested function's `return` is read against its own return type.
    ("Board<elem'actor> outer():\n    def Board<elem'inline> helper():\n        let Point'actor p = Point(1)\n        return Board(cells = [p])\n    let h = helper()\n    return Board()\n\n\
def main():\n    print \"x\"\n",
        "conflicting qualifiers for `elem` of `Board`: `'inline` (the expected type) and `'actor` (declared on `p`)", "13:16"),
];

/// Enum cases (declarations on lines 1-11, `main` follows).
const ENUM_CASES: &[(&str, &str, &str)] = &[
    // 3. A bare variant of an enum with no default and no proof stays an error.
    ("def main():\n    let c = Choice.None\n", "cannot infer qualifier parameter `q` of `Choice`, write `Choice<q'...>`", "13:13"),
    // 3. The pattern names an instance, the subject is declared another one.
    ("def main():\n    let Slot<elem'actor> s = Slot.Empty\n    match s:\n        Slot<elem'inline>.Empty: print \"x\"\n        _: print \"y\"\n",
        "the pattern names the instance `Slot<'inline>` but the subject is declared `Slot<'actor>`", "14:11"),
    // 3. A pattern's argument is checked against the bound.
    ("def main():\n    let Choice<q'actor> c = Choice.None\n    match c:\n        Choice<q'inline>.None: print \"x\"\n        _: print \"y\"\n",
        "qualifier `'inline` is outside the bound of `q` in `Choice`", "15:16"),
];

/// 4. Errors inside a header whose list spans several lines point at the right line.
const HEADER_CASES: &[(&str, &str, &str)] = &[
    ("struct Bad<\n    'a,\n    'b as 'weak\n>:\n    int x\n\ndef main():\n    print 'x'\n", "`'weak` cannot bound a qualifier parameter", "3:12"),
    ("int pick<\n    'q as 'inline|owned = 'actor\n>(int x):\n    x\n\ndef main():\n    print 'x'\n", "outside its bound", "2:27"),
];

/// 6. A suffix on a passed-on parameter is checked against the whole bound, at the declaration.
const SUFFIX_CASES: &[(&str, &str, &str)] = &[
    ("struct Bad<'c as 'shared|'actor>:\n    var Held<q'c'observed> inner = Held<q'c'observed>()\n",
        "`c` may be `'shared`, and `'shared'observed` added to it is not a valid chain: cannot combine `'observed` with `'shared`", "8:14"),
    ("struct Bad<'c>:\n    var Held<q'c'observed> inner = Held<q'c'observed>()\n",
        "`'inline'observed` once `'observed` is added is outside it (allowed: actor'observed, guard'observed)", "8:14"),
    ("struct Bad<'c as 'actor'observed|'actor>:\n    var Held<q'c'observed> inner = Held<q'c'observed>()\n",
        "`'actor'observed'observed` added to it is not a valid chain", "8:14"),
    ("struct Bad<'c as 'actor>:\n    var Held<q'unknown'observed> inner = Held<q'actor'observed>()\n",
        "unknown qualifier parameter `unknown`", "8:16"),
];

const SUFFIX_PRELUDE: &str = "struct Point:\n    var int x = 0\n\nstruct Held<'q as 'actor'observed|'guard'observed>:\n    mut Point'q pt = Point()\n\n";

const PRELUDE: &str = "struct Point:\n    var int x = 0\n\nstruct Board<'elem>:\n    var [mut Point'elem] cells = []\n\n    req int size():\n        cells.len()\n\n";

fn diagnostic(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n");
    let msg = stderr.lines().find(|l| l.starts_with("error")).unwrap_or("").to_string();
    let at = stderr.lines().find(|l| l.contains("-->")).and_then(|l| l.rsplit(':').collect::<Vec<_>>().get(..2).map(|p| format!("{}:{}", p[1], p[0]))).unwrap_or_default();
    format!("{msg} @ {at}")
}

fn check(dir: &Path, i: usize, source: &str, fragment: &str, pos: &str) {
    let bin = env!("CARGO_BIN_EXE_boring");
    let file = dir.join(format!("case{i}.br"));
    std::fs::write(&file, source).expect("write case");
    let run = Command::new(bin).arg("run").arg(&file).output().expect("boring run");
    let out_dir = dir.join(format!("case{i}_rust"));
    let build = Command::new(bin).arg("build").arg(&file).arg("--output-dir").arg(&out_dir).output().expect("boring build");
    assert!(!run.status.success() && !build.status.success(), "{source}: both must fail");
    let (r, b) = (diagnostic(&run), diagnostic(&build));
    assert_eq!(r, b, "{source}: run and build disagree");
    assert!(r.contains(fragment), "{source}: expected `{fragment}`, got `{r}`");
    assert!(r.ends_with(&format!("@ {pos}")), "{source}: expected position {pos}, got `{r}`");
    let _ = std::fs::remove_dir_all(&out_dir);
}

const ENUM_PRELUDE: &str = "struct Point:\n    var int x = 0\n\nenum Slot<'elem = 'inline>:\n    Empty\n    Full(Point'elem)\n\nenum Choice<'q as 'actor|owned>:\n    None\n    One(Point'q)\n\n";

#[test]
fn qualifier_gap_errors_are_the_same_for_run_and_build() {
    let dir = Path::new("target/tmp/qualifier_gaps_errors");
    std::fs::create_dir_all(dir).expect("scratch dir");
    for (i, (source, fragment, pos)) in CASES.iter().enumerate() {
        check(dir, i, &format!("{PRELUDE}{source}"), fragment, pos);
    }
    for (i, (source, fragment, pos)) in HEADER_CASES.iter().enumerate() {
        check(dir, 200 + i, source, fragment, pos);
    }
    for (i, (source, fragment, pos)) in SUFFIX_CASES.iter().enumerate() {
        check(dir, 300 + i, &format!("{SUFFIX_PRELUDE}{source}\ndef main():\n    print \"x\"\n"), fragment, pos);
    }
    for (i, (source, fragment, pos)) in ENUM_CASES.iter().enumerate() {
        check(dir, 100 + i, &format!("{ENUM_PRELUDE}{source}"), fragment, pos);
    }
}
