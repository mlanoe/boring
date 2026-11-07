//! Labelled generic argument errors (docs/book.md "Labelled generic arguments"): `boring run`
//! and `boring build` must report the same diagnostic, with the position of the offending
//! argument. Both go through `parser::parse`, where labels are resolved.
use std::path::Path;
use std::process::Command;

const PRELUDE: &str = "struct Pair<K, V>:\n    K a\n    V b\n\nT2 pick<T2, U2>(T2 x, U2 y):\n    x\n\n";

/// (use site, message fragment, 1-based `line:col` of the reported span).
const CASES: &[(&str, &str, &str)] = &[
    ("let p = Pair<K = int, K = string>(1, \"a\")", "generic parameter `K` of `Pair` is given more than once", "9:27"),
    ("let p = Pair<int, K = int>(1, \"a\")", "generic parameter `K` of `Pair` is given more than once", "9:23"),
    ("let p = Pair<X = int, V = string>(1, \"a\")", "unknown generic parameter `X` of `Pair` (parameters: K, V)", "9:18"),
    ("let p = Pair<K = int, string>(1, \"a\")", "positional generic argument after a labelled one", "9:27"),
    ("let p = Pair<K = int>(1, \"a\")", "missing generic argument `V` of `Pair`", "9:17"),
    ("let p = Pair<int, string, V = int>(1, \"a\")", "generic parameter `V` of `Pair` is given more than once", "9:31"),
    ("let p = Pair<int, string, int, K = int>(1, \"a\")", "takes 2 generic argument(s), got more", "9:31"),
    ("let r = pick<T2 = int, Z = int>(1, 2)", "unknown generic parameter `Z` of function `pick` (parameters: T2, U2)", "9:28"),
    ("let r = pick<T2 = int>(1, 2)", "missing generic argument `U2` of function `pick`", "9:17"),
    ("let p = Nope<K = int>(1)", "need the declaration of `Nope`", "9:18"),
    ("let Pair<K = int, V = string, K = int> p = Pair<int, string>(1, \"a\")", "generic parameter `K` of `Pair` is given more than once", "9:35"),
];

fn diagnostic(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n");
    let msg = stderr.lines().find(|l| l.starts_with("error")).unwrap_or("").to_string();
    let at = stderr.lines().find(|l| l.contains("-->")).and_then(|l| l.rsplit(':').collect::<Vec<_>>().get(..2).map(|p| format!("{}:{}", p[1], p[0]))).unwrap_or_default();
    format!("{msg} @ {at}")
}

#[test]
fn labelled_generic_errors_are_the_same_for_run_and_build() {
    let bin = env!("CARGO_BIN_EXE_boring");
    let dir = Path::new("target/tmp/labelled_generics_errors");
    std::fs::create_dir_all(dir).expect("scratch dir");
    for (i, (site, fragment, pos)) in CASES.iter().enumerate() {
        let file = dir.join(format!("case{i}.br"));
        std::fs::write(&file, format!("{PRELUDE}def main():\n    {site}\n")).expect("write case");
        let run = Command::new(bin).arg("run").arg(&file).output().expect("boring run");
        let out_dir = dir.join(format!("case{i}_rust"));
        let build = Command::new(bin).arg("build").arg(&file).arg("--output-dir").arg(&out_dir).output().expect("boring build");
        assert!(!run.status.success() && !build.status.success(), "{site}: both must fail");
        let (r, b) = (diagnostic(&run), diagnostic(&build));
        assert_eq!(r, b, "{site}: run and build disagree");
        assert!(r.contains(fragment), "{site}: expected `{fragment}`, got `{r}`");
        assert!(r.ends_with(&format!("@ {pos}")), "{site}: expected position {pos}, got `{r}`");
        let _ = std::fs::remove_dir_all(&out_dir);
    }
}
