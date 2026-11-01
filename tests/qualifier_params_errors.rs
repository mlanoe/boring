//! Qualifier parameter errors (docs/design-notes/qualifier-generic-params.md): `boring run` and
//! `boring build` must report the same diagnostic, with the position of the offending token. All
//! of them are found while parsing, so both go through `parser::parse`.
use std::path::Path;
use std::process::Command;

const PRELUDE: &str = "struct Point:\n    var int x = 0\n\nstruct Board<'elem>:\n    var [mut Point'elem] cells = []\n\nstruct Cell<'q as 'new>:\n    var Point'q pt = Point()\n\n";

/// (statement in `main`, message fragment, 1-based `line:col` of the reported span).
const USE_CASES: &[(&str, &str, &str)] = &[
    ("let b = Board()", "cannot infer qualifier parameter `elem` of `Board`, write `Board<elem'...>`", "11:13"),
    ("let Board b = Board<elem'actor>()", "cannot infer qualifier parameter `elem` of `Board`, write `Board<elem'...>`", "11:5"),
    // Inference at construction (phase 2): only a written qualifier is evidence.
    ("let b = Board(cells = [])", "cannot infer qualifier parameter `elem` of `Board`", "11:13"),
    ("let b = Board(cells = [Point(1)])", "cannot infer qualifier parameter `elem` of `Board`", "11:13"),
    ("let p = Point(1)\n    let b = Board(cells = [p])", "cannot infer qualifier parameter `elem` of `Board`", "12:13"),
    ("let Point b = Point(1)\n    let x = Board(cells = [b])", "cannot infer qualifier parameter `elem` of `Board`", "12:13"),
    ("let Point'actor a = Point(1)\n    let Point'inline b = Point(2)\n    let x = Board(cells = [a, b])",
        "conflicting qualifiers for `elem` of `Board`: `'actor` (declared on `a`) and `'inline` (declared on `b`)", "13:13"),
    ("let Point'inline a = Point(1)\n    let Board<elem'actor> x = Board(cells = [a])",
        "conflicting qualifiers for `elem` of `Board`: `'actor` (the expected type) and `'inline` (declared on `a`)", "12:31"),
    ("let Point'inline p = Point(1)\n    let c = Cell(pt = p)",
        "qualifier `'inline` (declared on `p`) is outside the bound of `q` in `Cell` (allowed: owned, shared, actor, guard)", "12:13"),
    ("let b = Board<foo'actor>()", "unknown qualifier parameter `foo` of `Board` (qualifier parameters: elem)", "11:19"),
    ("let b = Board<elem'actor, elem'inline>()", "generic parameter `elem` of `Board` is given more than once", "11:31"),
    ("let b = Cell<q'inline>()", "qualifier `'inline` is outside the bound of `q` in `Cell` (allowed: owned, shared, actor, guard)", "11:18"),
    ("let b = Board<elem'inline|owned>()", "qualifier argument `elem` of `Board` must be one of", "11:19"),
    // A chain is a whole: `'actor'weak` is not in the full bound, which lists single qualifiers.
    ("let b = Board<elem'actor'weak>()", "qualifier `'actor'weak` is outside the bound of `elem` in `Board`", "11:19"),
];

/// (whole file, message fragment, position): declaration-side errors.
const DECL_CASES: &[(&str, &str, &str)] = &[
    ("struct Bad<'actor>:\n    int x\n", "`'actor` is a qualifier, it cannot name a qualifier parameter", "1:13"),
    ("struct Bad<T, 'T>:\n    T x\n", "`T` is declared both as a type parameter and as a qualifier parameter", "1:17"),
    ("struct Bad<'q as 'weak>:\n    int x\n", "`'weak` cannot bound a qualifier parameter", "1:19"),
    ("struct Point:\n    var int x = 0\n\nstruct P:\n    var Point'elem x = Point()\n", "expected an ownership qualifier after '", "5:15"),
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
fn qualifier_param_errors_are_the_same_for_run_and_build() {
    let dir = Path::new("target/tmp/qualifier_params_errors");
    std::fs::create_dir_all(dir).expect("scratch dir");
    for (i, (site, fragment, pos)) in USE_CASES.iter().enumerate() {
        check(dir, i, &format!("{PRELUDE}def main():\n    {site}\n"), fragment, pos);
    }
    for (i, (source, fragment, pos)) in DECL_CASES.iter().enumerate() {
        check(dir, 100 + i, &format!("{source}\ndef main():\n    print \"x\"\n"), fragment, pos);
    }
}

// ── Default values and chained qualifiers (design note, section 10) ──────────────

const PRELUDE2: &str = "struct Point:\n    var int x = 0\n\nstruct Dflt<'elem = 'inline>:\n    var [mut Point'elem] cells = []\n\nstruct Obs<'q as 'actor|'actor'observed>:\n    mut Point'q pt = Point()\n\nstruct OnlyActor<'q as 'actor>:\n    mut Point'q pt = Point()\n\nstruct Capped<'q as 'inline|owned = 'inline>:\n    var Point'q pt = Point()\n\n";

const USE_CASES2: &[(&str, &str, &str)] = &[
    // A bound without a default keeps the "cannot infer" error.
    ("let b = Obs()", "cannot infer qualifier parameter `q` of `Obs`, write `Obs<q'...>`", "17:13"),
    // Membership is by whole chain: `as 'actor` does not admit `'actor'observed`.
    ("let b = OnlyActor<q'actor'observed>()", "qualifier `'actor'observed` is outside the bound of `q` in `OnlyActor` (allowed: actor)", "17:23"),
    ("let b = Obs<q'actor'weak>()", "qualifier `'actor'weak` is outside the bound of `q` in `Obs` (allowed: actor, actor'observed)", "17:17"),
    // Validity of a chain is the validity of the same chain on a type.
    ("let b = Dflt<elem'shared'observed>()", "cannot combine `'observed` with `'shared`", "17:18"),
    // Evidence outside the bound is an error, the default does not rescue it.
    ("let Point'actor p = Point(1)\n    let c = Capped(pt = p)", "qualifier `'actor` (declared on `p`) is outside the bound of `q` in `Capped` (allowed: inline, owned)", "18:13"),
    // A written chain is evidence as a whole.
    ("let Point'actor'observed p = Point(1)\n    let c = OnlyActor(pt = p)", "qualifier `'actor'observed` (declared on `p`) is outside the bound of `q` in `OnlyActor` (allowed: actor)", "18:13"),
];

const DECL_CASES2: &[(&str, &str, &str)] = &[
    ("struct Bad<'q as 'inline|owned = 'actor>:\n    int x\n", "default `'actor` of qualifier parameter `q` is outside its bound (allowed: inline, owned)", "1:34"),
    ("struct Bad<'q as 'actor = 'actor'observed>:\n    int x\n", "default `'actor'observed` of qualifier parameter `q` is outside its bound (allowed: actor)", "1:27"),
    ("struct Bad<'q = 'shared'observed>:\n    int x\n", "cannot combine `'observed` with `'shared`", "1:33"),
    ("struct Bad<'q as 'shared'observed>:\n    int x\n", "cannot combine `'observed` with `'shared`", "1:34"),
    ("struct Bad<'q = 'foo>:\n    int x\n", "cannot be a qualifier as the default of a qualifier parameter", "1:21"),
];

#[test]
fn default_and_chain_errors_are_the_same_for_run_and_build() {
    let dir = Path::new("target/tmp/qualifier_params_errors2");
    std::fs::create_dir_all(dir).expect("scratch dir");
    for (i, (site, fragment, pos)) in USE_CASES2.iter().enumerate() {
        check(dir, i, &format!("{PRELUDE2}def main():\n    {site}\n"), fragment, pos);
    }
    for (i, (source, fragment, pos)) in DECL_CASES2.iter().enumerate() {
        check(dir, 100 + i, &format!("{source}\ndef main():\n    print \"x\"\n"), fragment, pos);
    }
}
