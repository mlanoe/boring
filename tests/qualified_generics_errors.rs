//! Qualified generics `<T'q>` errors (docs/book.md, "Qualified generics"):
//! `boring run` and `boring build` report the same diagnostic at the same position.
use std::path::Path;
use std::process::Command;

const PRELUDE: &str = "struct Counter:\n    var int n = 0\n\nstruct Entry<T'q>:\n    var T'q res\n    var T'inline fallback\n\nstruct Pair<T'q>:\n    var T'q a\n    var T'q b\n\nstruct Shared<A'q, B'q>:\n    var A'q a\n    var B'q b\n\nstruct Plain<T>:\n    var T v\n\nint peek<T'q>(T'q x, T'inline y):\n    1\n\n";

/// (statement in `main`, message fragment, 1-based `line:col`); `main` starts at line 19.
const USE_CASES: &[(&str, &str, &str)] = &[
    // Row: a `T'inline` slot given a written `'actor` argument.
    ("let Counter'actor c = Counter(1)\n    let e = Entry(res = c, fallback = c)",
        "`T'inline` slot of `Entry` is given an argument written `'actor` (declared on `c`)", "24:13"),
    // Row: two `T'q` slots built from different written qualifiers.
    ("let Counter'actor a = Counter(1)\n    let Counter'owned b = Counter(2)\n    let p = Pair(a = a, b = b)",
        "conflicting qualifiers for `q` of `Pair`: `'actor` (declared on `a`) and `'owned` (declared on `b`)", "25:13"),
    // Row: `T = Counter'actor` on a declaration without `'q`.
    ("let Plain<T = Counter'actor> p = nil", "`T` of `Plain` has no qualifier parameter to bind `'actor`; declare it `<T'q>`", "23:15"),
    // Row: explicit and different.
    ("let Entry<T = Counter'actor, q'owned> e = nil", "conflicting qualifiers for `q` of `Entry`: `'actor` (written on `T`) and `'owned`", "23:15"),
    // A fresh literal is no evidence.
    ("let e = Entry(res = Counter(3), fallback = Counter(1))", "cannot infer qualifier parameter `q` of `Entry`", "23:13"),
    // A function call is resolved the same way.
    ("let Counter'actor c = Counter(1)\n    let Counter'owned o = Counter(2)\n    let r = peek(c, o)",
        "`T'inline` slot of `peek` is given an argument written `'owned`", "25:13"),
];

const DECL_CASES: &[(&str, &str, &str)] = &[
    ("struct E<A, 'q, B'q>:\n    var A'q a\n    var B'q b\n", "qualifier parameter `q` is declared twice", "1:19"),
    ("struct E<T'T>:\n    var T x\n", "`T` is declared both as a type parameter and as a qualifier parameter", "1:13"),
    ("struct E<T'q as Display>:\n    var T'q x\n", "`T'q` takes no bound or default", "1:14"),
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
fn qualified_generics_errors_are_the_same_for_run_and_build() {
    let dir = Path::new("target/tmp/qualified_generics_errors");
    std::fs::create_dir_all(dir).expect("scratch dir");
    for (i, (site, fragment, pos)) in USE_CASES.iter().enumerate() {
        check(dir, i, &format!("{PRELUDE}def main():\n    {site}\n"), fragment, pos);
    }
    for (i, (source, fragment, pos)) in DECL_CASES.iter().enumerate() {
        check(dir, 100 + i, &format!("{source}\ndef main():\n    print \"x\"\n"), fragment, pos);
    }
}
