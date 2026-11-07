//! Errors of qualifier-generic functions and methods
//! (docs/book.md "Qualifier parameters"): `boring run` and `boring build`
//! report the same diagnostic at the same position, since every one is found while parsing.
use std::path::Path;
use std::process::Command;

const PRELUDE: &str = "struct Point:\n    var int x = 0\n\nstruct Board<'elem>:\n    var [mut Point'elem] cells = []\n\nint count<'q>(Board<elem'q> b):\n    b.cells.len()\n\ndef add<'q as 'inline|owned>(mut Board<elem'q> b, Point'q p):\n    b.cells.push(p)\n\nstruct Game<'cells = 'inline>:\n    var mut Board<elem'cells> board = Board<elem'cells>()\n    def merge<'q>(Board<elem'q> other):\n        pass\n\n";

/// (statement in `main`, message fragment, 1-based `line:col` of the reported span).
const USE_CASES: &[(&str, &str, &str)] = &[
    // Nothing written determines `q`: a binding without a written type gives no evidence.
    ("let Board<elem'actor> a = Board()\n    let c = a\n    let n = count(c)",
        "cannot infer qualifier parameter `q` of `count`, write `count<q'...>`", "21:13"),
    // A construction that is itself undetermined is no evidence either.
    ("let m = count(Board(cells = []))",
        "cannot infer qualifier parameter `q` of `count`, write `count<q'...>`", "19:13"),
    // Two proofs that disagree.
    ("let Board<elem'owned> b = Board()\n    let Point'inline p = Point(1)\n    add(b, p)",
        "conflicting qualifiers for `q` of `add`: `'owned` (declared on `b`) and `'inline` (declared on `p`); write `add<q'...>` to choose", "21:5"),
    // A proof outside the bound of the parameter.
    ("let Board<elem'actor> b = Board()\n    let Point'actor p = Point(1)\n    add(b, p)",
        "qualifier `'actor` (declared on `b`) is outside the bound of `q` in `add` (allowed: inline, owned)", "21:5"),
    ("let Board<elem'actor> b = Board()\n    let n = count<q'owned>(b)", "", ""),
    ("let Board<elem'actor> b = Board()\n    add<q'actor>(b, Point(1))",
        "qualifier `'actor` is outside the bound of `q` in function `add` (allowed: inline, owned)", "20:9"),
    ("let n = count<z'actor>(Board<elem'actor>())",
        "unknown qualifier parameter `z` of function `count` (qualifier parameters: q)", "19:19"),
    ("let n = count<q'actor, q'owned>(Board<elem'actor>())",
        "generic parameter `q` of function `count` is given more than once", "19:28"),
    // A method's own parameter is inferred from the arguments like a function's.
    ("mut Game<cells'actor> g = Game()\n    let Board<elem'actor> a = Board()\n    let Board<elem'owned> o = Board()\n    g.merge(a)\n    let c = a\n    g.merge(c)",
        "cannot infer qualifier parameter `q` of `merge`, write `merge<q'...>`", "24:7"),
];

/// (whole file, message fragment, position): declaration-side errors.
const DECL_CASES: &[(&str, &str, &str)] = &[
    // A function body does not introduce a parameter: only its signature does.
    ("struct Point:\n    var int x = 0\n\nint bad(Point'q p):\n    let Point'r x = p\n    1\n",
        "unknown qualifier parameter `r`", "5:15"),
    // A struct declares its parameters, it never introduces one by use.
    ("struct Point:\n    var int x = 0\n\nstruct Board<'elem>:\n    var [mut Point'elem] cells = []\n\nstruct Bad:\n    var Board<elem'q> b = Board<elem'q>()\n",
        "unknown qualifier parameter `q`", "8:20"),
    ("struct Point:\n    var int x = 0\n\nenum Slot:\n    Full(Point'q)\n",
        "unknown qualifier parameter `q`", "5:16"),
    // The default must belong to the bound, as on a struct.
    ("struct Point:\n    var int x = 0\n\ndef f<'q as 'inline|owned = 'actor>(Point'q p):\n    pass\n",
        "default `'actor` of qualifier parameter `q` is outside its bound (allowed: inline, owned)", "4:29"),
    ("def f<'actor>(int x):\n    pass\n", "`'actor` is a qualifier, it cannot name a qualifier parameter", "1:8"),
    // The bound of a parameter passed on must fit in the one of the receiving declaration.
    ("struct Point:\n    var int x = 0\n\nstruct Narrow<'e as 'inline|owned>:\n    var [mut Point'e] cells = []\n\nint f<'q>(Narrow<e'q> n):\n    1\n",
        "the bound of `q` (allowed: inline, owned, shared, actor, guard) is not within the bound of `e` in `Narrow`: `'shared` is outside it", "7:18"),
    // A suffix added to a passed parameter is checked against the whole bound, which is also why a
    // polymorphic recursion can never grow its instance: every member of `q` plus `'observed` must
    // be a valid chain, and the chain is a base plus one suffix, so `grow<q'q'observed>` never fits.
    ("struct Point:\n    var int x = 0\n\nstruct Board<'elem as 'actor|'actor'observed = 'actor>:\n    var [mut Point'elem] cells = []\n\nint grow<'q as 'actor|'actor'observed>(Board<elem'q> b, int n):\n    if n == 0:\n        return 0\n    grow<q'q'observed>(b, n - 1)\n",
        "`q` may be `'actor'observed`, and `'actor'observed'observed` added to it is not a valid chain", "10:10"),
    // A method may not reuse the name of a parameter of its struct.
    ("struct Point:\n    var int x = 0\n\nstruct Holder<'q>:\n    var Point'q p = Point()\n    def m<'q>(Point'q x):\n        pass\n",
        "qualifier parameter `q` of `m` has the same name as one of the enclosing declaration", "6:5"),
];

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

#[test]
fn qualifier_fn_errors_are_the_same_for_run_and_build() {
    let dir = Path::new("target/tmp/qualifier_fns_errors");
    std::fs::create_dir_all(dir).expect("scratch dir");
    for (i, (site, fragment, pos)) in USE_CASES.iter().enumerate() {
        if fragment.is_empty() { continue; }
        check(dir, i, &format!("{PRELUDE}def main():\n    {site}\n"), fragment, pos);
    }
    for (i, (source, fragment, pos)) in DECL_CASES.iter().enumerate() {
        check(dir, 100 + i, &format!("{source}\ndef main():\n    print \"x\"\n"), fragment, pos);
    }
}

/// A parameter introduced by one occurrence in the signature and written nowhere else is most
/// likely a typo: a warning, never an error, and `<'q>` silences it.
#[test]
fn isolated_implicit_qualifier_parameter_warns() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = Path::new("target/tmp/qualifier_fns_warning");
    std::fs::create_dir_all(dir).expect("scratch dir");
    let source = |sig: &str| format!("struct Point:\n    var int x = 0\n\n{sig}\n    1\n\ndef main():\n    print \"ok\"\n");
    let run = |name: &str, src: String| {
        let file = dir.join(name);
        std::fs::write(&file, src).expect("write");
        Command::new(bin).arg("run").arg(&file).output().expect("boring run")
    };
    let lone = run("lone.br", source("int lone(Point'q p):"));
    let err = String::from_utf8_lossy(&lone.stderr);
    assert!(lone.status.success(), "a warning is not an error: {err}");
    assert!(err.contains("warning: qualifier parameter `q` is never related to another use"), "{err}");
    let related = run("related.br", source("int related(Point'q p, Point'q r):"));
    assert!(related.status.success());
    assert!(!String::from_utf8_lossy(&related.stderr).contains("warning"), "two uses relate");
    let declared = run("declared.br", source("int declared<'q>(Point'q p):"));
    assert!(declared.status.success());
    assert!(!String::from_utf8_lossy(&declared.stderr).contains("warning"), "an explicit declaration is not a typo");
}
